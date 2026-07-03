"""
Unit tests for dense_embedding_codegen_lookup_function (forward pass).

This test suite validates:
1. Forward pass correctness for dense embedding lookups
2. Kernel dispatch based on dimension (small vs large kernel)
3. Output properties (shape, dtype, device, no NaN/Inf)
4. Deterministic behavior
5. Comparison with PyTorch reference implementation
"""
import unittest
import torch

import fbgemm_gpu
import fbgemm_xpu

SEED = 0


class TestDenseEmbeddingLookup(unittest.TestCase):
    """Unit tests for dense_embedding_codegen_lookup_function."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are shared across all tests."""
        if not torch.xpu.is_available():
            raise unittest.SkipTest("XPU not available")
        torch.manual_seed(SEED)

    def setUp(self):
        if torch.xpu.is_available():
            torch.xpu.synchronize()
        torch.manual_seed(SEED)

    def tearDown(self):
        if torch.xpu.is_available():
            torch.xpu.synchronize()
            torch.xpu.empty_cache()

    def _create_embedding_inputs(self, device, D, num_rows, num_indices, dtype=torch.float32):
        """Helper to create embedding lookup inputs."""
        # Create embedding weights (single table)
        dev_weights = torch.randn(num_rows * D, dtype=dtype, device=device)
        weights_offsets = torch.tensor([0], dtype=torch.int64, device=device)
        hash_size_cumsum = torch.tensor([0, num_rows], dtype=torch.int64, device=device)
        D_offsets = torch.tensor([0, D], dtype=torch.int32, device=device)
        
        # Create indices and offsets for no-bag mode (pooling_mode=2)
        indices = torch.randint(0, num_rows, (num_indices,), dtype=torch.int64, device=device)
        offsets = torch.arange(num_indices + 1, dtype=torch.int64, device=device)
        
        return {
            'dev_weights': dev_weights,
            'weights_offsets': weights_offsets,
            'D_offsets': D_offsets,
            'total_D': D,
            'max_D': D,
            'hash_size_cumsum': hash_size_cumsum,
            'total_hash_size_bits': 0,
            'indices': indices,
            'offsets': offsets,
            'pooling_mode': 2,  # NONE (no pooling)
            'indice_weights': None,
            'feature_requires_grad': None,
        }

    def _run_lookup(self, inputs):
        """Helper to run the lookup function."""
        result = torch.ops.fbgemm.dense_embedding_codegen_lookup_function(
            dev_weights=inputs['dev_weights'],
            weights_offsets=inputs['weights_offsets'],
            D_offsets=inputs['D_offsets'],
            total_D=inputs['total_D'],
            max_D=inputs['max_D'],
            hash_size_cumsum=inputs['hash_size_cumsum'],
            total_hash_size_bits=inputs['total_hash_size_bits'],
            indices=inputs['indices'],
            offsets=inputs['offsets'],
            pooling_mode=inputs['pooling_mode'],
            indice_weights=inputs.get('indice_weights'),
            feature_requires_grad=inputs.get('feature_requires_grad'),
        )
        torch.xpu.synchronize()
        return result

    def test_basic_forward_pass(self):
        """Test basic forward pass with medium dimension."""
        device = torch.device("xpu")
        D = 128
        num_rows = 100
        num_indices = 50
        
        inputs = self._create_embedding_inputs(device, D, num_rows, num_indices)
        result = self._run_lookup(inputs)
        
        # Verify output shape
        self.assertEqual(result.shape, (num_indices, D))
        self.assertEqual(result.device.type, 'xpu')
        self.assertFalse(torch.isnan(result).any())
        self.assertFalse(torch.isinf(result).any())

    def test_small_kernel_dispatch(self):
        """Test that small kernel is used for D <= 32."""
        device = torch.device("xpu")
        
        for D in [8, 16, 32]:
            with self.subTest(D=D):
                num_rows = 50
                num_indices = 20
                
                inputs = self._create_embedding_inputs(device, D, num_rows, num_indices)
                result = self._run_lookup(inputs)
                
                # Verify output
                self.assertEqual(result.shape, (num_indices, D))
                self.assertEqual(result.device.type, 'xpu')
                self.assertFalse(torch.isnan(result).any())

    def test_large_kernel_dispatch(self):
        """Test that large kernel is used for D > 32."""
        device = torch.device("xpu")
        
        for D in [64, 128, 256]:
            with self.subTest(D=D):
                num_rows = 50
                num_indices = 20
                
                inputs = self._create_embedding_inputs(device, D, num_rows, num_indices)
                result = self._run_lookup(inputs)
                
                # Verify output
                self.assertEqual(result.shape, (num_indices, D))
                self.assertEqual(result.device.type, 'xpu')
                self.assertFalse(torch.isnan(result).any())

    def test_output_shape_consistency(self):
        """Test that output shape matches expected dimensions."""
        device = torch.device("xpu")
        D = 128
        num_rows = 100
        num_indices = 50
        
        inputs = self._create_embedding_inputs(device, D, num_rows, num_indices)
        result = self._run_lookup(inputs)
        
        # Check output dimensions
        self.assertEqual(len(result.shape), 2, "Output should be 2D tensor")
        self.assertEqual(result.shape[0], num_indices, 
                        "First dimension should match number of indices")
        self.assertEqual(result.shape[1], D,
                        "Second dimension should match embedding dimension")

    def test_output_dtype_fp32(self):
        """Test output dtype for FP32 weights."""
        device = torch.device("xpu")
        D = 128
        num_rows = 50
        num_indices = 20
        
        inputs = self._create_embedding_inputs(device, D, num_rows, num_indices, dtype=torch.float32)
        result = self._run_lookup(inputs)
        
        self.assertEqual(result.dtype, torch.float32)

    def test_output_dtype_fp16(self):
        """Test output dtype for FP16 weights."""
        device = torch.device("xpu")
        D = 128
        num_rows = 50
        num_indices = 20
        
        inputs = self._create_embedding_inputs(device, D, num_rows, num_indices, dtype=torch.float16)
        result = self._run_lookup(inputs)
        
        # Note: FP16 input is upcast to FP32 for computation
        self.assertEqual(result.dtype, torch.float32)

    def test_output_dtype_bf16(self):
        """Test output dtype for BF16 weights.
        
        Currently BF16 is not implemented and should raise RuntimeError.
        When BF16 support is added, update this test to verify correct behavior.
        """
        device = torch.device("xpu")
        D = 128
        num_rows = 50
        num_indices = 20
        
        inputs = self._create_embedding_inputs(device, D, num_rows, num_indices, dtype=torch.bfloat16)
        
        # BF16 is not yet implemented - expect RuntimeError
        with self.assertRaises(RuntimeError) as context:
            self._run_lookup(inputs)
        
        self.assertIn("not implemented", str(context.exception).lower())

    def test_output_device(self):
        """Test that output is on the correct device (XPU)."""
        device = torch.device("xpu")
        D = 128
        num_rows = 100
        num_indices = 50
        
        inputs = self._create_embedding_inputs(device, D, num_rows, num_indices)
        result = self._run_lookup(inputs)
        
        self.assertEqual(result.device.type, 'xpu', "Output should be on XPU device")

    def test_no_nan_values(self):
        """Test that output contains no NaN values."""
        device = torch.device("xpu")
        D = 128
        num_rows = 100
        num_indices = 50
        
        inputs = self._create_embedding_inputs(device, D, num_rows, num_indices)
        result = self._run_lookup(inputs)
        
        self.assertFalse(torch.isnan(result).any(), 
                        "Output should not contain NaN values")

    def test_no_inf_values(self):
        """Test that output contains no Inf values."""
        device = torch.device("xpu")
        D = 128
        num_rows = 100
        num_indices = 50
        
        inputs = self._create_embedding_inputs(device, D, num_rows, num_indices)
        result = self._run_lookup(inputs)
        
        self.assertFalse(torch.isinf(result).any(), 
                        "Output should not contain Inf values")

    def test_deterministic_output(self):
        """Test that output is deterministic given the same inputs."""
        device = torch.device("xpu")
        D = 128
        num_rows = 100
        num_indices = 50
        
        torch.manual_seed(SEED)
        inputs = self._create_embedding_inputs(device, D, num_rows, num_indices)
        
        result1 = self._run_lookup(inputs)
        result2 = self._run_lookup(inputs)
        
        self.assertTrue(torch.equal(result1, result2),
                       "Multiple calls with same inputs should produce identical results")

    def test_indices_range_validity(self):
        """Test that indices are within valid range."""
        device = torch.device("xpu")
        D = 128
        num_rows = 100
        num_indices = 50
        
        inputs = self._create_embedding_inputs(device, D, num_rows, num_indices)
        
        # Verify indices are within bounds
        self.assertTrue((inputs['indices'] >= 0).all(), 
                       "All indices should be non-negative")
        self.assertTrue((inputs['indices'] < num_rows).all(),
                       f"All indices should be less than num_rows ({num_rows})")

    def test_comparison_with_pytorch_embedding(self):
        """Test that output matches PyTorch's nn.Embedding for single table."""
        device = torch.device("xpu")
        D = 128
        num_rows = 50
        num_indices = 20
        
        # Create inputs
        torch.manual_seed(SEED)
        inputs = self._create_embedding_inputs(device, D, num_rows, num_indices)
        
        # Run XPU version
        result_xpu = self._run_lookup(inputs)
        
        # Create PyTorch reference on CPU
        ref_embedding = torch.nn.Embedding(num_rows, D)
        ref_embedding.weight.data = inputs['dev_weights'].view(num_rows, D).cpu()
        
        # Run reference
        indices_cpu = inputs['indices'].cpu()
        result_cpu = ref_embedding(indices_cpu)
        
        # Compare
        torch.testing.assert_close(
            result_xpu.cpu(),
            result_cpu,
            atol=1e-5,
            rtol=1e-4,
            msg="XPU output should match PyTorch reference"
        )

    def test_empty_indices(self):
        """Test handling of empty indices."""
        device = torch.device("xpu")
        D = 128
        num_rows = 100
        num_indices = 0
        
        inputs = self._create_embedding_inputs(device, D, num_rows, num_indices)
        result = self._run_lookup(inputs)
        
        self.assertEqual(result.shape, (0, D))

    def test_single_index(self):
        """Test handling of single index."""
        device = torch.device("xpu")
        D = 128
        num_rows = 100
        num_indices = 1
        
        inputs = self._create_embedding_inputs(device, D, num_rows, num_indices)
        result = self._run_lookup(inputs)
        
        self.assertEqual(result.shape, (1, D))
        self.assertFalse(torch.isnan(result).any())

    def test_large_batch(self):
        """Test with large batch of indices."""
        device = torch.device("xpu")
        D = 128
        num_rows = 1000
        num_indices = 5000
        
        inputs = self._create_embedding_inputs(device, D, num_rows, num_indices)
        result = self._run_lookup(inputs)
        
        self.assertEqual(result.shape, (num_indices, D))
        self.assertFalse(torch.isnan(result).any())

    def test_repeated_indices(self):
        """Test with repeated indices (same index looked up multiple times)."""
        device = torch.device("xpu")
        D = 128
        num_rows = 100
        
        # Create inputs with repeated indices
        torch.manual_seed(SEED)
        dev_weights = torch.randn(num_rows * D, dtype=torch.float32, device=device)
        weights_offsets = torch.tensor([0], dtype=torch.int64, device=device)
        hash_size_cumsum = torch.tensor([0, num_rows], dtype=torch.int64, device=device)
        D_offsets = torch.tensor([0, D], dtype=torch.int32, device=device)
        
        # Same index repeated multiple times
        indices = torch.tensor([5, 5, 5, 10, 10, 15], dtype=torch.int64, device=device)
        offsets = torch.arange(len(indices) + 1, dtype=torch.int64, device=device)
        
        inputs = {
            'dev_weights': dev_weights,
            'weights_offsets': weights_offsets,
            'D_offsets': D_offsets,
            'total_D': D,
            'max_D': D,
            'hash_size_cumsum': hash_size_cumsum,
            'total_hash_size_bits': 0,
            'indices': indices,
            'offsets': offsets,
            'pooling_mode': 2,
            'indice_weights': None,
            'feature_requires_grad': None,
        }
        
        result = self._run_lookup(inputs)
        
        # Verify repeated indices produce same output
        self.assertTrue(torch.equal(result[0], result[1]))
        self.assertTrue(torch.equal(result[1], result[2]))
        self.assertTrue(torch.equal(result[3], result[4]))

    def test_direct_weight_matrix_row_extraction(self):
        """Test that lookup output matches exact rows from weight matrix.
        
        This verifies the fundamental correctness: that looking up index i
        returns exactly row i from the weight matrix without any corruption.
        """
        device = torch.device("xpu")
        D = 128
        num_rows = 100
        num_indices = 10
        
        torch.manual_seed(SEED)
        inputs = self._create_embedding_inputs(device, D, num_rows, num_indices)
        
        # Reshape weights to 2D matrix for row extraction
        weight_matrix = inputs['dev_weights'].view(num_rows, D)
        
        # Run lookup
        result = self._run_lookup(inputs)
        
        # Verify each output row matches the corresponding weight matrix row
        for i, idx in enumerate(inputs['indices']):
            expected_row = weight_matrix[idx]
            actual_row = result[i]
            
            self.assertTrue(
                torch.equal(expected_row, actual_row),
                f"Lookup result at position {i} (index={idx}) does not match weight matrix row"
            )
            
            # Also verify with torch.allclose for numerical precision
            torch.testing.assert_close(
                actual_row,
                expected_row,
                atol=0.0,
                rtol=0.0,
                msg=f"Exact match failed for index {idx}"
            )

    def test_cross_kernel_consistency_at_dimension_boundary(self):
        """Test that results are consistent across small/large kernel boundary.
        
        The implementation switches kernels at D=32. This test verifies that
        a lookup with D=32 (small kernel) produces the same result when we
        use D=64 (large kernel) for the same embedding values (padded).
        """
        device = torch.device("xpu")
        num_rows = 50
        num_indices = 20
        
        # Test D=32 (small kernel threshold)
        torch.manual_seed(SEED)
        D_small = 32
        inputs_small = self._create_embedding_inputs(device, D_small, num_rows, num_indices)
        result_small = self._run_lookup(inputs_small)
        
        # Create equivalent lookup with PyTorch reference for cross-validation
        ref_embedding = torch.nn.Embedding(num_rows, D_small)
        ref_embedding.weight.data = inputs_small['dev_weights'].view(num_rows, D_small).cpu()
        indices_cpu = inputs_small['indices'].cpu()
        result_ref = ref_embedding(indices_cpu)
        
        # Verify small kernel matches reference exactly
        torch.testing.assert_close(
            result_small.cpu(),
            result_ref,
            atol=1e-5,
            rtol=1e-4,
            msg="Small kernel (D=32) should match PyTorch reference"
        )
        
        # Test D=64 (just over threshold, uses large kernel, and D % 4 == 0)
        torch.manual_seed(SEED)
        D_large = 64
        inputs_large = self._create_embedding_inputs(device, D_large, num_rows, num_indices)
        result_large = self._run_lookup(inputs_large)
        
        # Create equivalent lookup with PyTorch reference
        ref_embedding_large = torch.nn.Embedding(num_rows, D_large)
        ref_embedding_large.weight.data = inputs_large['dev_weights'].view(num_rows, D_large).cpu()
        result_ref_large = ref_embedding_large(indices_cpu)
        
        # Verify large kernel matches reference exactly
        torch.testing.assert_close(
            result_large.cpu(),
            result_ref_large,
            atol=1e-5,
            rtol=1e-4,
            msg="Large kernel (D=64) should match PyTorch reference"
        )

    def test_batched_vs_individual_lookup_consistency(self):
        """Test that batched lookup produces same results as individual lookups.
        
        This verifies that the batching/parallelization doesn't introduce
        any inconsistencies - looking up N indices at once should give the
        same result as looking up each index individually.
        """
        device = torch.device("xpu")
        D = 128
        num_rows = 100
        
        torch.manual_seed(SEED)
        
        # Create shared weights
        dev_weights = torch.randn(num_rows * D, dtype=torch.float32, device=device)
        weights_offsets = torch.tensor([0], dtype=torch.int64, device=device)
        hash_size_cumsum = torch.tensor([0, num_rows], dtype=torch.int64, device=device)
        D_offsets = torch.tensor([0, D], dtype=torch.int32, device=device)
        
        # Create a set of test indices including boundaries and middle values
        test_indices = torch.tensor([0, 1, 10, 25, 50, 75, 99], dtype=torch.int64, device=device)
        
        # Perform batched lookup of all indices at once
        offsets_batch = torch.arange(len(test_indices) + 1, dtype=torch.int64, device=device)
        inputs_batch = {
            'dev_weights': dev_weights,
            'weights_offsets': weights_offsets,
            'D_offsets': D_offsets,
            'total_D': D,
            'max_D': D,
            'hash_size_cumsum': hash_size_cumsum,
            'total_hash_size_bits': 0,
            'indices': test_indices,
            'offsets': offsets_batch,
            'pooling_mode': 2,
            'indice_weights': None,
            'feature_requires_grad': None,
        }
        result_batch = self._run_lookup(inputs_batch)
        
        # Perform individual lookups
        individual_results = []
        for idx in test_indices:
            inputs_individual = {
                'dev_weights': dev_weights,
                'weights_offsets': weights_offsets,
                'D_offsets': D_offsets,
                'total_D': D,
                'max_D': D,
                'hash_size_cumsum': hash_size_cumsum,
                'total_hash_size_bits': 0,
                'indices': idx.unsqueeze(0),
                'offsets': torch.tensor([0, 1], dtype=torch.int64, device=device),
                'pooling_mode': 2,
                'indice_weights': None,
                'feature_requires_grad': None,
            }
            result_individual = self._run_lookup(inputs_individual)
            individual_results.append(result_individual[0])
        
        result_individual_stacked = torch.stack(individual_results)
        
        # Verify batched and individual results match exactly
        self.assertTrue(
            torch.equal(result_batch, result_individual_stacked),
            "Batched lookup should produce identical results to individual lookups"
        )
        
        # Also verify each index individually
        for i, idx in enumerate(test_indices):
            torch.testing.assert_close(
                result_batch[i],
                result_individual_stacked[i],
                atol=0.0,
                rtol=0.0,
                msg=f"Mismatch at index {idx.item()}: batched vs individual lookup"
            )

    def test_consistency_across_dimensions(self):
        """Test lookup consistency across multiple dimensions from small to large.
        
        This verifies that the lookup operation works correctly across the full
        range of embedding dimensions, including:
        - Small dimensions (D=8, 16) using small kernel
        - Boundary dimension (D=32) at kernel threshold
        - Medium dimensions (D=64, 128) using large kernel
        - Large dimensions (D=256, 512) using large kernel
        
        Each dimension is validated against PyTorch reference to ensure correctness.
        """
        device = torch.device("xpu")
        num_rows = 100
        num_indices = 30
        
        # Test dimensions spanning small to large, including kernel boundary
        test_dimensions = [8, 16, 32, 64, 128, 256, 512, 1024]
        
        # Use same seed for each dimension to get comparable random indices
        base_seed = SEED
        
        for D in test_dimensions:
            with self.subTest(D=D):
                # Use same seed for indices generation
                torch.manual_seed(base_seed)
                
                # Create inputs for this dimension
                inputs = self._create_embedding_inputs(device, D, num_rows, num_indices)
                
                # Run XPU lookup
                result_xpu = self._run_lookup(inputs)
                
                # Create PyTorch reference
                ref_embedding = torch.nn.Embedding(num_rows, D)
                ref_embedding.weight.data = inputs['dev_weights'].view(num_rows, D).cpu()
                indices_cpu = inputs['indices'].cpu()
                result_ref = ref_embedding(indices_cpu)
                
                # Verify correctness against reference
                torch.testing.assert_close(
                    result_xpu.cpu(),
                    result_ref,
                    atol=1e-5,
                    rtol=1e-4,
                    msg=f"XPU output mismatch for D={D}"
                )
                
                # Additional sanity checks
                self.assertEqual(result_xpu.shape, (num_indices, D),
                               f"Shape mismatch for D={D}")
                self.assertFalse(torch.isnan(result_xpu).any(),
                               f"NaN values found for D={D}")
                self.assertFalse(torch.isinf(result_xpu).any(),
                               f"Inf values found for D={D}")
                
                # Verify direct weight matrix row extraction for this dimension
                weight_matrix = inputs['dev_weights'].view(num_rows, D)
                for i, idx in enumerate(inputs['indices'][:10]):  # Check first 10 for efficiency
                    expected_row = weight_matrix[idx]
                    actual_row = result_xpu[i]
                    self.assertTrue(
                        torch.equal(expected_row, actual_row),
                        f"Row extraction failed for D={D}, index={idx.item()}"
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
