"""
Complete unit test for both backward pass kernels:
- CtaPerRow kernel: long segments (SL >= 32)
- WarpPerRow kernel: short segments (SL < 32)

This test validates:
1. Each kernel separately
2. Both kernels working together
3. Boundary conditions
4. Numerical correctness of gradients
"""
import unittest
import torch

# name for the custom operator module (should match the name in setup.py)
import fbgemm_gpu
import fbgemm_xpu

SEED = 42


class TestDenseEmbeddingCodegenBackward(unittest.TestCase):
    """Comprehensive tests for dense_embedding_codegen backward pass (CtaPerRow and WarpPerRow kernels)."""
    @classmethod
    def setUpClass(cls):
        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            raise unittest.SkipTest("XPU is not available")

    def _create_embedding_setup(self, device, D, num_rows):
        """Helper to create embedding setup configuration."""
        dev_weights = torch.randn(
            num_rows * D, dtype=torch.float32, device=device, requires_grad=True
        )
        weights_offsets = torch.tensor([0], dtype=torch.int64, device=device)
        hash_size_cumsum = torch.tensor([0, num_rows], dtype=torch.int64, device=device)
        D_offsets = torch.tensor([0, D], dtype=torch.int32, device=device)

        return dev_weights, weights_offsets, hash_size_cumsum, D_offsets

    def _run_forward_backward(self, dev_weights, weights_offsets, hash_size_cumsum,
                             D_offsets, indices, D, device):
        """Helper to execute forward and backward pass."""
        B = len(indices)
        offsets = torch.arange(B + 1, dtype=torch.int64, device=device)

        # Forward pass
        output = torch.ops.fbgemm.dense_embedding_codegen_lookup_function(
            dev_weights=dev_weights,
            weights_offsets=weights_offsets,
            D_offsets=D_offsets,
            total_D=D,
            max_D=D,
            hash_size_cumsum=hash_size_cumsum,
            total_hash_size_bits=0,
            indices=indices,
            offsets=offsets,
            pooling_mode=2,  # NONE (no pooling)
            indice_weights=None,
            feature_requires_grad=None,
        )

        # Backward pass with gradients = 1.0
        grad_output = torch.ones_like(output)
        output.backward(grad_output)
        torch.xpu.synchronize()

        return dev_weights.grad

    def test_warp_per_row_only_short_segments(self):
        """Test 1: Only short segments (< 32) - WarpPerRow kernel."""
        device = torch.device("xpu")
        D = 128
        num_rows = 20

        dev_weights, weights_offsets, hash_size_cumsum, D_offsets = \
            self._create_embedding_setup(device, D, num_rows)

        # All segments are short (< 32 repetitions)
        # index 2: 5 reps, index 5: 10 reps, index 8: 15 reps, index 12: 25 reps
        indices_list = [2] * 5 + [5] * 10 + [8] * 15 + [12] * 25
        indices = torch.tensor(indices_list, dtype=torch.int64, device=device)

        grad = self._run_forward_backward(
            dev_weights, weights_offsets, hash_size_cumsum, D_offsets, indices, D, device
        )

        grad_reshaped = grad.view(num_rows, D)

        # Verify that WarpPerRow processed correctly
        self.assertAlmostEqual(
            grad_reshaped[2].sum().item(), 5.0 * D, delta=1e-3,
            msg="WarpPerRow: index 2 (5 reps)"
        )
        self.assertAlmostEqual(
            grad_reshaped[5].sum().item(), 10.0 * D, delta=1e-3,
            msg="WarpPerRow: index 5 (10 reps)"
        )
        self.assertAlmostEqual(
            grad_reshaped[8].sum().item(), 15.0 * D, delta=1e-3,
            msg="WarpPerRow: index 8 (15 reps)"
        )
        self.assertAlmostEqual(
            grad_reshaped[12].sum().item(), 25.0 * D, delta=1e-3,
            msg="WarpPerRow: index 12 (25 reps)"
        )

        # Verify no NaN or Inf
        self.assertFalse(torch.isnan(grad).any().item())
        self.assertFalse(torch.isinf(grad).any().item())

    def test_cta_per_row_only_long_segments(self):
        """Test 2: Only long segments (>= 32) - CtaPerRow kernel."""
        device = torch.device("xpu")
        D = 64
        num_rows = 30

        dev_weights, weights_offsets, hash_size_cumsum, D_offsets = \
            self._create_embedding_setup(device, D, num_rows)

        # All segments are long (>= 32 repetitions)
        # index 3: 32 reps, index 7: 50 reps, index 10: 75 reps, index 15: 100 reps
        indices_list = [3] * 32 + [7] * 50 + [10] * 75 + [15] * 100
        indices = torch.tensor(indices_list, dtype=torch.int64, device=device)

        grad = self._run_forward_backward(
            dev_weights, weights_offsets, hash_size_cumsum, D_offsets, indices, D, device
        )

        grad_reshaped = grad.view(num_rows, D)

        # Verify that CtaPerRow processed correctly
        self.assertAlmostEqual(
            grad_reshaped[3].sum().item(), 32.0 * D, delta=1e-3,
            msg="CtaPerRow: index 3 (32 reps)"
        )
        self.assertAlmostEqual(
            grad_reshaped[7].sum().item(), 50.0 * D, delta=1e-3,
            msg="CtaPerRow: index 7 (50 reps)"
        )
        self.assertAlmostEqual(
            grad_reshaped[10].sum().item(), 75.0 * D, delta=1e-3,
            msg="CtaPerRow: index 10 (75 reps)"
        )
        self.assertAlmostEqual(
            grad_reshaped[15].sum().item(), 100.0 * D, delta=1e-3,
            msg="CtaPerRow: index 15 (100 reps)"
        )

        # Verify no NaN or Inf
        self.assertFalse(torch.isnan(grad).any().item())
        self.assertFalse(torch.isinf(grad).any().item())

    def test_both_kernels_mixed_segments(self):
        """Test 3: Mixed segments - both kernels working together."""
        device = torch.device("xpu")
        D = 128
        num_rows = 25

        dev_weights, weights_offsets, hash_size_cumsum, D_offsets = \
            self._create_embedding_setup(device, D, num_rows)

        # Mix of short and long segments
        indices_list = (
            [1] * 5 +     # WarpPerRow: 5 < 32
            [3] * 10 +    # WarpPerRow: 10 < 32
            [5] * 20 +    # WarpPerRow: 20 < 32
            [7] * 31 +    # WarpPerRow: 31 < 32 (lower limit)
            [10] * 32 +   # CtaPerRow: 32 >= 32 (exact limit)
            [12] * 40 +   # CtaPerRow: 40 >= 32
            [15] * 60 +   # CtaPerRow: 60 >= 32
            [18] * 100    # CtaPerRow: 100 >= 32
        )
        indices = torch.tensor(indices_list, dtype=torch.int64, device=device)

        grad = self._run_forward_backward(
            dev_weights, weights_offsets, hash_size_cumsum, D_offsets, indices, D, device
        )

        grad_reshaped = grad.view(num_rows, D)

        # Verify segments WarpPerRow
        warp_indices = {1: 5, 3: 10, 5: 20, 7: 31}
        for idx, reps in warp_indices.items():
            self.assertAlmostEqual(
                grad_reshaped[idx].sum().item(), reps * D, delta=1e-3,
                msg=f"WarpPerRow: index {idx} ({reps} reps)"
            )

        # Verify segments CtaPerRow
        cta_indices = {10: 32, 12: 40, 15: 60, 18: 100}
        for idx, reps in cta_indices.items():
            self.assertAlmostEqual(
                grad_reshaped[idx].sum().item(), reps * D, delta=1e-3,
                msg=f"CtaPerRow: index {idx} ({reps} reps)"
            )

        # Verify no NaN or Inf
        self.assertFalse(torch.isnan(grad).any().item())
        self.assertFalse(torch.isinf(grad).any().item())

    def test_boundary_case_31_vs_32(self):
        """Test 4: Boundary case between WarpPerRow (31) and CtaPerRow (32)."""
        device = torch.device("xpu")
        D = 128
        num_rows = 10

        dev_weights, weights_offsets, hash_size_cumsum, D_offsets = \
            self._create_embedding_setup(device, D, num_rows)

        # Boundary case: 31 should go to WarpPerRow, 32 should go to CtaPerRow
        indices_list = [3] * 31 + [5] * 32
        indices = torch.tensor(indices_list, dtype=torch.int64, device=device)

        grad = self._run_forward_backward(
            dev_weights, weights_offsets, hash_size_cumsum, D_offsets, indices, D, device
        )

        grad_reshaped = grad.view(num_rows, D)

        # 31 repetitions: WarpPerRow (SL < 32)
        self.assertAlmostEqual(
            grad_reshaped[3].sum().item(), 31.0 * D, delta=1e-3,
            msg="Index 3 (31 reps) should be processed by WarpPerRow"
        )

        # 32 repetitions: CtaPerRow (SL >= 32)
        self.assertAlmostEqual(
            grad_reshaped[5].sum().item(), 32.0 * D, delta=1e-3,
            msg="Index 5 (32 reps) should be processed by CtaPerRow"
        )

    def test_different_dimensions(self):
        """Test 5: Different dimensions D with both kernels."""
        device = torch.device("xpu")
        num_rows = 20

        # Test different dimensions (including D > 256 to test kUseVecBlocking path)
        dimensions = [32, 64, 128, 256, 512]

        for D in dimensions:
            with self.subTest(D=D):
                dev_weights, weights_offsets, hash_size_cumsum, D_offsets = \
                    self._create_embedding_setup(device, D, num_rows)

                # Mix of short and long segments
                indices_list = [2] * 10 + [5] * 50
                indices = torch.tensor(indices_list, dtype=torch.int64, device=device)

                grad = self._run_forward_backward(
                    dev_weights, weights_offsets, hash_size_cumsum, D_offsets,
                    indices, D, device
                )

                grad_reshaped = grad.view(num_rows, D)

                # Verify both kernels
                self.assertAlmostEqual(
                    grad_reshaped[2].sum().item(), 10.0 * D, delta=1e-3,
                    msg=f"D={D}: WarpPerRow index 2 (10 reps)"
                )
                self.assertAlmostEqual(
                    grad_reshaped[5].sum().item(), 50.0 * D, delta=1e-3,
                    msg=f"D={D}: CtaPerRow index 5 (50 reps)"
                )

    def test_kernel_launch_order(self):
        """Test 6: Verify that launch order (CtaPerRow → WarpPerRow) works correctly."""
        device = torch.device("xpu")
        D = 128
        num_rows = 20

        dev_weights, weights_offsets, hash_size_cumsum, D_offsets = \
            self._create_embedding_setup(device, D, num_rows)

        # Interleaved segments: long, short, long, short, short
        indices_list = (
            [2] * 50 +   # CtaPerRow (launched first)
            [4] * 3 +    # WarpPerRow (launched second)
            [6] * 45 +   # CtaPerRow (launched first)
            [8] * 7 +    # WarpPerRow (launched second)
            [12] * 15    # WarpPerRow (launched second)
        )
        indices = torch.tensor(indices_list, dtype=torch.int64, device=device)

        grad = self._run_forward_backward(
            dev_weights, weights_offsets, hash_size_cumsum, D_offsets, indices, D, device
        )

        grad_reshaped = grad.view(num_rows, D)

        # Verify all segments processed correctly
        # regardless of launch order
        expected = {
            2: 50,   # CtaPerRow
            4: 3,    # WarpPerRow
            6: 45,   # CtaPerRow
            8: 7,    # WarpPerRow
            12: 15   # WarpPerRow
        }

        for idx, reps in expected.items():
            self.assertAlmostEqual(
                grad_reshaped[idx].sum().item(), reps * D, delta=1e-3,
                msg=f"Launch order: index {idx} ({reps} reps)"
            )

    def test_very_long_segments(self):
        """Test 7: Very long segments (> 1024) that require special accumulation."""
        device = torch.device("xpu")
        D = 64
        num_rows = 15

        dev_weights, weights_offsets, hash_size_cumsum, D_offsets = \
            self._create_embedding_setup(device, D, num_rows)

        # Very long segments that may require atomic accumulation
        indices_list = [3] * 1500 + [7] * 2000
        indices = torch.tensor(indices_list, dtype=torch.int64, device=device)

        grad = self._run_forward_backward(
            dev_weights, weights_offsets, hash_size_cumsum, D_offsets, indices, D, device
        )

        grad_reshaped = grad.view(num_rows, D)

        # Verify that even very long segments process correctly
        self.assertAlmostEqual(
            grad_reshaped[3].sum().item(), 1500.0 * D, delta=1e-2,
            msg="Very long segment: index 3 (1500 reps)"
        )
        self.assertAlmostEqual(
            grad_reshaped[7].sum().item(), 2000.0 * D, delta=1e-2,
            msg="Very long segment: index 7 (2000 reps)"
        )

    def test_gradient_values_correctness(self):
        """Test 8: Verify exact gradient values, not just sums."""
        device = torch.device("xpu")
        D = 8  # Small dimension for exact verification
        num_rows = 10

        dev_weights, weights_offsets, hash_size_cumsum, D_offsets = \
            self._create_embedding_setup(device, D, num_rows)

        # Simple segments
        indices_list = [2] * 5 + [5] * 40
        indices = torch.tensor(indices_list, dtype=torch.int64, device=device)

        grad = self._run_forward_backward(
            dev_weights, weights_offsets, hash_size_cumsum, D_offsets, indices, D, device
        )

        grad_reshaped = grad.view(num_rows, D)

        # Each embedding dimension should have gradient = number of repetitions
        # (because grad_output is all 1.0)

        # WarpPerRow: index 2, 5 repetitions
        for d in range(D):
            self.assertAlmostEqual(
                grad_reshaped[2][d].item(), 5.0, delta=1e-4,
                msg=f"WarpPerRow: index 2, dimension {d}"
            )

        # CtaPerRow: index 5, 40 repetitions
        for d in range(D):
            self.assertAlmostEqual(
                grad_reshaped[5][d].item(), 40.0, delta=1e-4,
                msg=f"CtaPerRow: index 5, dimension {d}"
            )

    def test_multiple_indices_same_value(self):
        """Test 9: Multiple indices with the same number of repetitions."""
        device = torch.device("xpu")
        D = 64
        num_rows = 30

        dev_weights, weights_offsets, hash_size_cumsum, D_offsets = \
            self._create_embedding_setup(device, D, num_rows)

        # Multiple indices with 10 reps (WarpPerRow) and multiple with 50 reps (CtaPerRow)
        indices_list = (
            [1] * 10 + [3] * 10 + [5] * 10 +  # 3 indices with 10 reps (WarpPerRow)
            [10] * 50 + [12] * 50 + [15] * 50  # 3 indices with 50 reps (CtaPerRow)
        )
        indices = torch.tensor(indices_list, dtype=torch.int64, device=device)

        grad = self._run_forward_backward(
            dev_weights, weights_offsets, hash_size_cumsum, D_offsets, indices, D, device
        )

        grad_reshaped = grad.view(num_rows, D)

        # Verify WarpPerRow indices
        for idx in [1, 3, 5]:
            self.assertAlmostEqual(
                grad_reshaped[idx].sum().item(), 10.0 * D, delta=1e-3,
                msg=f"WarpPerRow: index {idx}"
            )

        # Verify CtaPerRow indices
        for idx in [10, 12, 15]:
            self.assertAlmostEqual(
                grad_reshaped[idx].sum().item(), 50.0 * D, delta=1e-3,
                msg=f"CtaPerRow: index {idx}"
            )

    def test_single_repetition(self):
        """Test 10: Cases with single repetition (minimum for WarpPerRow)."""
        device = torch.device("xpu")
        D = 128
        num_rows = 10

        dev_weights, weights_offsets, hash_size_cumsum, D_offsets = \
            self._create_embedding_setup(device, D, num_rows)

        # Multiple indices with only 1 repetition each
        indices_list = [1, 3, 5, 7, 9]
        indices = torch.tensor(indices_list, dtype=torch.int64, device=device)

        grad = self._run_forward_backward(
            dev_weights, weights_offsets, hash_size_cumsum, D_offsets, indices, D, device
        )

        grad_reshaped = grad.view(num_rows, D)

        # Verify each index has correct gradient
        for idx in [1, 3, 5, 7, 9]:
            self.assertAlmostEqual(
                grad_reshaped[idx].sum().item(), 1.0 * D, delta=1e-4,
                msg=f"Single repetition: index {idx}"
            )

    def test_warp_cta_kernels_single_table_numerical_correctness(self):
        """
        Test 11: Numerical correctness validation for WarpPerRow/CtaPerRow kernels.

        Validates that XPU backward kernels produce numerically identical results to
        PyTorch reference for single table (T=1) without pooling case. Uses PyTorch's
        Embedding as ground truth to verify gradient accumulation correctness.
        """
        device = torch.device("xpu")
        D = 64
        num_rows = 20

        # Setup XPU
        dev_weights, weights_offsets, hash_size_cumsum, D_offsets = \
            self._create_embedding_setup(device, D, num_rows)

        # Create PyTorch reference embedding on CPU
        ref_embedding = torch.nn.Embedding(num_rows, D, sparse=False)
        # Copy same weights for fair comparison
        with torch.no_grad():
            ref_embedding.weight.copy_(dev_weights.view(num_rows, D).cpu())
        ref_embedding.weight.requires_grad = True

        # Create indices with mix of short and long segments
        # Use numpy for reproducibility
        import numpy as np
        np.random.seed(SEED)
        indices_np = np.random.choice(range(num_rows), size=50, replace=True).astype(np.int64)

        # XPU indices
        indices_xpu = torch.from_numpy(indices_np).to(device)

        # CPU reference indices
        indices_cpu = torch.from_numpy(indices_np)

        # Forward pass XPU
        output_xpu = torch.ops.fbgemm.dense_embedding_codegen_lookup_function(
            dev_weights=dev_weights,
            weights_offsets=weights_offsets,
            D_offsets=D_offsets,
            total_D=D,
            max_D=D,
            hash_size_cumsum=hash_size_cumsum,
            total_hash_size_bits=0,
            indices=indices_xpu,
            offsets=torch.arange(len(indices_xpu) + 1, dtype=torch.int64, device=device),
            pooling_mode=2,  # NONE
            indice_weights=None,
            feature_requires_grad=None,
        )

        # Forward pass CPU reference
        output_cpu = ref_embedding(indices_cpu)

        # Compare forward pass outputs
        torch.testing.assert_close(
            output_xpu.cpu(),
            output_cpu,
            atol=1e-5,
            rtol=1e-5,
            msg="Forward pass: XPU output differs from PyTorch reference"
        )

        # Backward pass with same gradient
        grad_output_cpu = torch.ones_like(output_cpu)
        grad_output_xpu = grad_output_cpu.to(device)

        # Backward CPU
        output_cpu.backward(grad_output_cpu)

        # Backward XPU
        output_xpu.backward(grad_output_xpu)
        torch.xpu.synchronize()

        # Compare gradients
        torch.testing.assert_close(
            dev_weights.grad.cpu(),
            ref_embedding.weight.grad.flatten(),
            atol=1e-4,
            rtol=1e-4,
            msg="Backward pass: XPU gradients differ from PyTorch reference"
        )

        # Verify no NaN or Inf
        self.assertFalse(torch.isnan(dev_weights.grad).any().item())
        self.assertFalse(torch.isinf(dev_weights.grad).any().item())


if __name__ == "__main__":
    # Run tests with verbosity
    unittest.main(verbosity=2)
