import copy
import enum
import math
import sys
import unittest
from types import ModuleType
from typing import List, Optional

import torch

import fbgemm_gpu
import fbgemm_xpu

class PoolingMode(enum.IntEnum):
    SUM = 0
    MEAN = 1
    NONE = 2


class PlacementType(enum.IntEnum):
    DEVICE = 0
    MANAGED = 1
    MANAGED_CACHING = 2


def _ensure_fbgemm_gpu_mock() -> None:
    if "fbgemm_gpu" in sys.modules:
        return
    fbgemm_gpu_mock = ModuleType("fbgemm_gpu")
    sys.modules["fbgemm_gpu"] = fbgemm_gpu_mock
    split_table_ops_mock = ModuleType("split_table_batched_embeddings_ops_common")
    split_table_ops_mock.PoolingMode = PoolingMode
    sys.modules["fbgemm_gpu.split_table_batched_embeddings_ops_common"] = split_table_ops_mock
    fbgemm_gpu_mock.split_table_batched_embeddings_ops_common = split_table_ops_mock


_ensure_fbgemm_gpu_mock()


class SparseType(enum.IntEnum):
    FP32 = 0
    FP16 = 1
    BF16 = 5


def _generate_synthetic_data(
    T: int,
    B: int,
    D: int,
    num_rows_per_table: int,
    L: int,
    placements: List[int],
    output_dtype: SparseType = SparseType.FP16,
    num_cache_slots: int = 0,
    cache_hit_ratio: float = 0.7,
    info_B_num_bits: int = 29,
    use_homogeneous_placements: bool = False,
    is_experimental: bool = False,
    seed: int = 42,
):
    """
    Generate synthetic input data for the split_embedding forward pass.

    Supports all placement types:
      - PlacementType.DEVICE (0): table stored in dev_weights
      - PlacementType.MANAGED (1): table stored in uvm_weights
      - PlacementType.MANAGED_CACHING (2): table in uvm_weights with LXU cache
    """
    assert len(placements) == T
    torch.manual_seed(seed)

    total_rows = T * num_rows_per_table
    total_D = T * D
    max_D = D

    weights_placements = torch.tensor(placements, dtype=torch.int32)

    dev_offset = 0
    uvm_offset = 0
    offsets_list = []
    for t in range(T):
        if placements[t] == PlacementType.DEVICE:
            offsets_list.append(dev_offset)
            dev_offset += num_rows_per_table * D
        else:
            offsets_list.append(uvm_offset)
            uvm_offset += num_rows_per_table * D

    weights_offsets = torch.tensor(offsets_list, dtype=torch.int64)

    dev_weights = torch.randn(dev_offset, dtype=torch.float32) if dev_offset > 0 else torch.zeros(0, dtype=torch.float32)
    uvm_weights = torch.randn(uvm_offset, dtype=torch.float32) if uvm_offset > 0 else torch.zeros(0, dtype=torch.float32)

    has_caching = any(p == PlacementType.MANAGED_CACHING for p in placements)
    if has_caching and num_cache_slots > 0:
        lxu_cache_weights = torch.randn(num_cache_slots, D, dtype=torch.float32)
    else:
        lxu_cache_weights = torch.zeros(0, 0, dtype=torch.float32)

    D_offsets = torch.tensor([i * D for i in range(T + 1)], dtype=torch.int32)
    hash_size_cumsum = torch.tensor(
        [i * num_rows_per_table for i in range(T + 1)], dtype=torch.int64
    )
    total_hash_size_bits = int(math.ceil(math.log2(max(total_rows, 1)))) + 1

    total_indices = T * B * L
    indices = torch.randint(0, num_rows_per_table, (total_indices,), dtype=torch.int64)
    offsets = torch.arange(0, total_indices + 1, L, dtype=torch.int64)
    assert offsets.numel() == T * B + 1

    # Generate lxu_cache_locations with globally unique slots per (table, row)
    if has_caching and num_cache_slots > 0:
        lxu_cache_locations = torch.full((total_indices,), -1, dtype=torch.int32)
        global_slot_idx = 0
        for t in range(T):
            if placements[t] != PlacementType.MANAGED_CACHING:
                continue
            table_offset = offsets_list[t]
            slots_remaining = num_cache_slots - global_slot_idx
            max_cacheable = min(num_rows_per_table, slots_remaining)
            num_hits_rows = int(max_cacheable * cache_hit_ratio)
            cached_rows = torch.randperm(num_rows_per_table)[:num_hits_rows]
            row_to_slot = {}
            for row in cached_rows.tolist():
                if global_slot_idx >= num_cache_slots:
                    break
                row_to_slot[row] = global_slot_idx
                src_start = table_offset + row * D
                lxu_cache_weights[global_slot_idx] = uvm_weights[src_start:src_start + D]
                global_slot_idx += 1

            start = t * B * L
            end = (t + 1) * B * L
            for i in range(start, end):
                row_idx = indices[i].item()
                if row_idx in row_to_slot:
                    lxu_cache_locations[i] = row_to_slot[row_idx]
    else:
        lxu_cache_locations = torch.zeros(0, dtype=torch.int32)

    # Momentum (required by the fused op even for forward-only)
    momentum1_dev = torch.zeros(total_rows, dtype=torch.float32)
    momentum1_host = torch.zeros(0, dtype=torch.float32)
    momentum1_placements = torch.zeros(T, dtype=torch.int32)
    momentum1_offsets = torch.tensor(
        [i * num_rows_per_table for i in range(T)], dtype=torch.int64
    )

    weights = [
        dev_weights,
        uvm_weights,
        weights_placements,
        weights_offsets,
        lxu_cache_weights,
    ]
    momentum1 = [
        momentum1_dev,
        momentum1_host,
        momentum1_placements,
        momentum1_offsets,
    ]

    learning_rate_tensor = torch.tensor(0.5, dtype=torch.float32)

    info_B_mask = (1 << info_B_num_bits) - 1
    aux_tensor: List[Optional[torch.Tensor]] = [
        None, None, None, lxu_cache_locations, None, None, None
    ]
    aux_int = [0, info_B_num_bits, info_B_mask]
    aux_float = [0.0, 1.0]
    aux_bool = [
        False,                       # IDX_GRADIENT_CLIPPING
        False,                       # IDX_STOCHASTIC_ROUNDING
        is_experimental,             # IDX_IS_EXPERIMENTAL_TBE
        False,                       # IDX_USE_UNIQ_CACHE_LOCATIONS_BWD
        False,                       # IDX_APPLY_GLOBAL_WEIGHT_DECAY
        use_homogeneous_placements,  # IDX_USE_HOMOGENEOUS_PLACEMENTS
        False,                       # reserved
    ]

    optim_float = [0.2, 0.0, 0.0]
    optim_int = [0]

    inputs = {
        "placeholder_autograd_tensor": torch.zeros(0, dtype=torch.float32),
        "weights": weights,
        "D_offsets": D_offsets,
        "total_D": total_D,
        "max_D": max_D,
        "hash_size_cumsum": hash_size_cumsum,
        "total_hash_size_bits": total_hash_size_bits,
        "indices": indices,
        "offsets": offsets,
        "pooling_mode": int(PoolingMode.NONE),
        "indice_weights": None,
        "feature_requires_grad": None,
        "output_dtype": int(output_dtype),
        "max_B": -1,
        "max_B_feature_rank": -1,
        "vbe_output_size": -1,
        "aux_tensor": aux_tensor,
        "aux_int": aux_int,
        "aux_float": aux_float,
        "aux_bool": aux_bool,
        "learning_rate_tensor": learning_rate_tensor,
        "momentum1": momentum1,
        "optim_int": optim_int,
        "optim_float": optim_float,
    }
    return inputs


def _reference_forward(inputs):
    """
    Reference CPU forward pass: for each index, gather the corresponding
    embedding row from the appropriate weight buffer (or cache).
    """
    weights = inputs["weights"]
    dev_weights = weights[0]
    uvm_weights = weights[1]
    placements = weights[2].tolist()
    offsets_list = weights[3].tolist()
    lxu_cache_weights = weights[4]
    hash_size_cumsum = inputs["hash_size_cumsum"]
    D_offsets = inputs["D_offsets"]
    indices = inputs["indices"]
    input_offsets = inputs["offsets"]
    lxu_cache_locations = inputs["aux_tensor"][3]
    T = len(placements)
    total_B = input_offsets.numel() - 1
    B = total_B // T
    D = inputs["max_D"]

    output_dtype_enum = SparseType(inputs["output_dtype"])
    if output_dtype_enum == SparseType.FP32:
        out_dtype = torch.float32
    elif output_dtype_enum == SparseType.FP16:
        out_dtype = torch.float16
    else:
        out_dtype = torch.bfloat16

    total_indices = indices.numel()
    output = torch.zeros(total_indices, D, dtype=out_dtype)

    for t in range(T):
        table_D = (D_offsets[t + 1] - D_offsets[t]).item()
        buf_offset = offsets_list[t]
        placement = placements[t]

        start = t * B
        for b in range(B):
            bag_idx = start + b
            idx_start = input_offsets[bag_idx].item()
            idx_end = input_offsets[bag_idx + 1].item()

            for l_idx in range(idx_start, idx_end):
                row_idx = indices[l_idx].item()

                # Determine source: cache hit or embedding buffer
                cache_loc = -1
                if lxu_cache_locations.numel() > 0:
                    cache_loc = lxu_cache_locations[l_idx].item()

                if placement == PlacementType.MANAGED_CACHING and cache_loc >= 0:
                    row_data = lxu_cache_weights[cache_loc, :table_D].float()
                elif placement == PlacementType.DEVICE:
                    w_start = buf_offset + row_idx * table_D
                    row_data = dev_weights[w_start:w_start + table_D].float()
                else:
                    w_start = buf_offset + row_idx * table_D
                    row_data = uvm_weights[w_start:w_start + table_D].float()

                output[l_idx, :table_D] = row_data.to(out_dtype)

    return output


def _move_to_device(inputs, device):
    result = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            result[k] = v.to(device)
        elif isinstance(v, list):
            result[k] = [
                x.to(device) if isinstance(x, torch.Tensor) else x for x in v
            ]
        else:
            result[k] = v
    return result


def _run_forward(inputs_xpu):
    weights = inputs_xpu["weights"]

    result = torch.ops.fbgemm.split_embedding_codegen_lookup_rowwise_adagrad_function_pt2(
        placeholder_autograd_tensor=inputs_xpu["placeholder_autograd_tensor"],
        weights=weights,
        D_offsets=inputs_xpu["D_offsets"],
        total_D=inputs_xpu["total_D"],
        max_D=inputs_xpu["max_D"],
        hash_size_cumsum=inputs_xpu["hash_size_cumsum"],
        total_hash_size_bits=inputs_xpu["total_hash_size_bits"],
        indices=inputs_xpu["indices"],
        offsets=inputs_xpu["offsets"],
        pooling_mode=inputs_xpu["pooling_mode"],
        indice_weights=inputs_xpu["indice_weights"],
        feature_requires_grad=inputs_xpu["feature_requires_grad"],
        output_dtype=inputs_xpu["output_dtype"],
        max_B=inputs_xpu["max_B"],
        max_B_feature_rank=inputs_xpu["max_B_feature_rank"],
        vbe_output_size=inputs_xpu["vbe_output_size"],
        aux_tensor=inputs_xpu["aux_tensor"],
        aux_int=inputs_xpu["aux_int"],
        aux_float=inputs_xpu["aux_float"],
        aux_bool=inputs_xpu["aux_bool"],
        learning_rate_tensor=inputs_xpu["learning_rate_tensor"].cpu(),
        momentum1=inputs_xpu["momentum1"],
        optim_int=inputs_xpu["optim_int"],
        optim_float=inputs_xpu["optim_float"],
    )
    torch.xpu.synchronize()
    return result


class TestSplitLookupForwardSynthetic(unittest.TestCase):
    """
    Tests for split_embedding forward pass (nobag, unweighted) using
    synthetically generated data across all placement configurations.
    """

    @classmethod
    def setUpClass(cls):
        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            raise unittest.SkipTest("XPU is not available")

    def setUp(self):
        torch.xpu.synchronize()

    def tearDown(self):
        torch.xpu.empty_cache()
        torch.xpu.synchronize()

    def _verify_forward(self, inputs, atol, rtol):
        ref_output = _reference_forward(inputs)
        inputs_xpu = _move_to_device(copy.deepcopy(inputs), "xpu")
        actual = _run_forward(inputs_xpu).cpu()

        self.assertEqual(actual.shape, ref_output.shape)
        torch.testing.assert_close(
            actual.float(), ref_output.float(), atol=atol, rtol=rtol,
            msg="Forward output mismatch vs reference",
        )

    # ===================================================================
    # Placement: all DEVICE (0)
    # ===================================================================

    def test_forward_all_device_fp32_small(self):
        """All DEVICE, FP32, small config."""
        inputs = _generate_synthetic_data(
            T=2, B=4, D=8, num_rows_per_table=5, L=3,
            placements=[0, 0],
            output_dtype=SparseType.FP32, seed=100,
        )
        self._verify_forward(inputs, atol=1e-5, rtol=1e-5)

    def test_forward_all_device_fp16_small(self):
        """All DEVICE, FP16, small config."""
        inputs = _generate_synthetic_data(
            T=2, B=4, D=8, num_rows_per_table=5, L=3,
            placements=[0, 0],
            output_dtype=SparseType.FP16, seed=101,
        )
        self._verify_forward(inputs, atol=1e-3, rtol=1e-3)

    def test_forward_all_device_fp32_medium(self):
        """All DEVICE, FP32, medium config."""
        inputs = _generate_synthetic_data(
            T=5, B=32, D=64, num_rows_per_table=10, L=18,
            placements=[0, 0, 0, 0, 0],
            output_dtype=SparseType.FP32, seed=102,
        )
        self._verify_forward(inputs, atol=1e-5, rtol=1e-5)

    def test_forward_all_device_bf16_medium(self):
        """All DEVICE, BF16, medium config."""
        inputs = _generate_synthetic_data(
            T=4, B=47, D=200, num_rows_per_table=10, L=18,
            placements=[0, 0, 0, 0],
            output_dtype=SparseType.BF16, seed=103,
        )
        self._verify_forward(inputs, atol=1e-2, rtol=1e-2)

    def test_forward_all_device_large_D(self):
        """All DEVICE, large embedding dim D=256."""
        inputs = _generate_synthetic_data(
            T=3, B=16, D=256, num_rows_per_table=10, L=8,
            placements=[0, 0, 0],
            output_dtype=SparseType.FP32, seed=104,
        )
        self._verify_forward(inputs, atol=1e-5, rtol=1e-5)

    def test_forward_all_device_single_table(self):
        """Single DEVICE table."""
        inputs = _generate_synthetic_data(
            T=1, B=16, D=128, num_rows_per_table=20, L=5,
            placements=[0],
            output_dtype=SparseType.FP32, seed=105,
        )
        self._verify_forward(inputs, atol=1e-5, rtol=1e-5)

    def test_forward_all_device_large_batch(self):
        """Large batch size."""
        inputs = _generate_synthetic_data(
            T=3, B=128, D=256, num_rows_per_table=10, L=20,
            placements=[0, 0, 0],
            output_dtype=SparseType.FP16, seed=106,
        )
        self._verify_forward(inputs, atol=1e-3, rtol=1e-3)

    def test_forward_all_device_info_B_30(self):
        """DEVICE with info_B_num_bits=30."""
        inputs = _generate_synthetic_data(
            T=2, B=82, D=8, num_rows_per_table=10, L=3,
            placements=[0, 0],
            info_B_num_bits=30,
            is_experimental=True,
            output_dtype=SparseType.FP32, seed=107,
        )
        self._verify_forward(inputs, atol=1e-5, rtol=1e-5)

    # ===================================================================
    # Placement: mixed DEVICE (0) + MANAGED (1)
    # ===================================================================

    def test_forward_mixed_device_managed_fp32(self):
        """Mixed DEVICE/MANAGED, FP32."""
        inputs = _generate_synthetic_data(
            T=5, B=86, D=228, num_rows_per_table=10, L=18,
            placements=[0, 1, 1, 1, 0],
            output_dtype=SparseType.FP32, seed=200,
        )
        self._verify_forward(inputs, atol=1e-5, rtol=1e-5)

    def test_forward_mixed_device_managed_fp16(self):
        """Mixed DEVICE/MANAGED, FP16."""
        inputs = _generate_synthetic_data(
            T=5, B=86, D=228, num_rows_per_table=10, L=18,
            placements=[0, 1, 1, 1, 0],
            output_dtype=SparseType.FP16, seed=201,
        )
        self._verify_forward(inputs, atol=1e-3, rtol=1e-3)

    def test_forward_mixed_device_managed_bf16(self):
        """Mixed DEVICE/MANAGED, BF16."""
        inputs = _generate_synthetic_data(
            T=4, B=47, D=200, num_rows_per_table=10, L=18,
            placements=[0, 1, 1, 1],
            output_dtype=SparseType.BF16, seed=202,
        )
        self._verify_forward(inputs, atol=1e-2, rtol=1e-2)

    def test_forward_mixed_device_managed_6_tables(self):
        """6 tables with mixed placements."""
        inputs = _generate_synthetic_data(
            T=6, B=30, D=128, num_rows_per_table=10, L=8,
            placements=[0, 1, 1, 1, 0, 1],
            output_dtype=SparseType.FP16, seed=203,
            use_homogeneous_placements=False,
        )
        self._verify_forward(inputs, atol=1e-3, rtol=1e-3)

    # ===================================================================
    # Placement: all MANAGED_CACHING (2)
    # ===================================================================

    def test_forward_all_caching_fp32(self):
        """All MANAGED_CACHING, FP32, mixed hits/misses."""
        inputs = _generate_synthetic_data(
            T=4, B=62, D=64, num_rows_per_table=10, L=13,
            placements=[2, 2, 2, 2],
            output_dtype=SparseType.FP32, seed=300,
            num_cache_slots=32,
            cache_hit_ratio=0.7,
            use_homogeneous_placements=True,
        )
        self._verify_forward(inputs, atol=1e-5, rtol=1e-5)

    def test_forward_all_caching_fp16(self):
        """All MANAGED_CACHING, FP16."""
        inputs = _generate_synthetic_data(
            T=6, B=61, D=128, num_rows_per_table=10, L=18,
            placements=[2, 2, 2, 2, 2, 2],
            output_dtype=SparseType.FP16, seed=301,
            num_cache_slots=64,
            cache_hit_ratio=0.7,
            use_homogeneous_placements=True,
        )
        self._verify_forward(inputs, atol=1e-3, rtol=1e-3)

    def test_forward_all_caching_full_hits(self):
        """All MANAGED_CACHING, 100% cache hit ratio."""
        inputs = _generate_synthetic_data(
            T=4, B=62, D=64, num_rows_per_table=10, L=13,
            placements=[2, 2, 2, 2],
            output_dtype=SparseType.FP32, seed=302,
            num_cache_slots=64,
            cache_hit_ratio=1.0,
            use_homogeneous_placements=True,
        )
        self._verify_forward(inputs, atol=1e-5, rtol=1e-5)

    def test_forward_all_caching_all_misses(self):
        """All MANAGED_CACHING, 0% hits (fallback to uvm)."""
        inputs = _generate_synthetic_data(
            T=4, B=62, D=64, num_rows_per_table=10, L=13,
            placements=[2, 2, 2, 2],
            output_dtype=SparseType.FP32, seed=303,
            num_cache_slots=32,
            cache_hit_ratio=0.0,
            use_homogeneous_placements=True,
        )
        self._verify_forward(inputs, atol=1e-5, rtol=1e-5)

    def test_forward_all_caching_bf16(self):
        """All MANAGED_CACHING, BF16."""
        inputs = _generate_synthetic_data(
            T=4, B=62, D=376, num_rows_per_table=10, L=13,
            placements=[2, 2, 2, 2],
            output_dtype=SparseType.BF16, seed=304,
            num_cache_slots=32,
            cache_hit_ratio=0.5,
            use_homogeneous_placements=True,
        )
        self._verify_forward(inputs, atol=1e-2, rtol=1e-2)

    def test_forward_all_caching_large_cache(self):
        """All MANAGED_CACHING, 1024 cache slots."""
        inputs = _generate_synthetic_data(
            T=6, B=61, D=428, num_rows_per_table=10, L=18,
            placements=[2, 2, 2, 2, 2, 2],
            output_dtype=SparseType.FP16, seed=305,
            num_cache_slots=1024,
            cache_hit_ratio=0.7,
            use_homogeneous_placements=True,
        )
        self._verify_forward(inputs, atol=1e-3, rtol=1e-3)

    # ===================================================================
    # Placement: mixed MANAGED_CACHING (2) + DEVICE (0)
    # ===================================================================

    def test_forward_mixed_caching_device_fp32(self):
        """Mixed MANAGED_CACHING + DEVICE, FP32."""
        inputs = _generate_synthetic_data(
            T=4, B=10, D=256, num_rows_per_table=10, L=20,
            placements=[2, 0, 0, 0],
            output_dtype=SparseType.FP32, seed=400,
            num_cache_slots=32,
            cache_hit_ratio=0.75,
        )
        self._verify_forward(inputs, atol=1e-5, rtol=1e-5)

    def test_forward_mixed_caching_device_fp16(self):
        """Mixed MANAGED_CACHING + DEVICE, FP16."""
        inputs = _generate_synthetic_data(
            T=3, B=10, D=256, num_rows_per_table=10, L=30,
            placements=[2, 0, 0],
            output_dtype=SparseType.FP16, seed=401,
            num_cache_slots=32,
            cache_hit_ratio=0.7,
            info_B_num_bits=30,
        )
        self._verify_forward(inputs, atol=1e-3, rtol=1e-3)

    def test_forward_mixed_caching_device_partial_misses(self):
        """Mixed placements, low cache hit rate."""
        inputs = _generate_synthetic_data(
            T=3, B=10, D=256, num_rows_per_table=10, L=35,
            placements=[2, 0, 0],
            output_dtype=SparseType.FP32, seed=402,
            num_cache_slots=32,
            cache_hit_ratio=0.25,
        )
        self._verify_forward(inputs, atol=1e-5, rtol=1e-5)

    # ===================================================================
    # Output dtype correctness
    # ===================================================================

    def test_forward_output_dtype_fp32(self):
        """Verify output dtype is FP32 when requested."""
        inputs = _generate_synthetic_data(
            T=2, B=4, D=16, num_rows_per_table=5, L=3,
            placements=[0, 0],
            output_dtype=SparseType.FP32, seed=500,
        )
        inputs_xpu = _move_to_device(inputs, "xpu")
        result = _run_forward(inputs_xpu)
        self.assertEqual(result.dtype, torch.float32)

    def test_forward_output_dtype_fp16(self):
        """Verify output dtype is FP16 when requested."""
        inputs = _generate_synthetic_data(
            T=2, B=4, D=16, num_rows_per_table=5, L=3,
            placements=[0, 0],
            output_dtype=SparseType.FP16, seed=501,
        )
        inputs_xpu = _move_to_device(inputs, "xpu")
        result = _run_forward(inputs_xpu)
        self.assertEqual(result.dtype, torch.float16)

    def test_forward_output_dtype_bf16(self):
        """Verify output dtype is BF16 when requested."""
        inputs = _generate_synthetic_data(
            T=2, B=4, D=16, num_rows_per_table=5, L=3,
            placements=[0, 0],
            output_dtype=SparseType.BF16, seed=502,
        )
        inputs_xpu = _move_to_device(inputs, "xpu")
        result = _run_forward(inputs_xpu)
        self.assertEqual(result.dtype, torch.bfloat16)

    # ===================================================================
    # Shape and basic properties
    # ===================================================================

    def test_forward_output_shape(self):
        """Output shape is (total_indices, D) for nobag mode."""
        inputs = _generate_synthetic_data(
            T=3, B=8, D=64, num_rows_per_table=10, L=5,
            placements=[0, 1, 0],
            output_dtype=SparseType.FP32, seed=600,
        )
        inputs_xpu = _move_to_device(inputs, "xpu")
        result = _run_forward(inputs_xpu)
        total_indices = inputs["indices"].numel()
        self.assertEqual(result.shape, (total_indices, inputs["max_D"]))

    def test_forward_no_nan(self):
        """Output contains no NaN values."""
        inputs = _generate_synthetic_data(
            T=4, B=32, D=128, num_rows_per_table=10, L=10,
            placements=[0, 1, 2, 0],
            output_dtype=SparseType.FP32, seed=601,
            num_cache_slots=16,
            cache_hit_ratio=0.5,
        )
        inputs_xpu = _move_to_device(inputs, "xpu")
        result = _run_forward(inputs_xpu)
        self.assertFalse(torch.isnan(result).any())

    def test_forward_no_inf(self):
        """Output contains no infinite values."""
        inputs = _generate_synthetic_data(
            T=4, B=32, D=128, num_rows_per_table=10, L=10,
            placements=[0, 1, 2, 0],
            output_dtype=SparseType.FP32, seed=602,
            num_cache_slots=16,
            cache_hit_ratio=0.5,
        )
        inputs_xpu = _move_to_device(inputs, "xpu")
        result = _run_forward(inputs_xpu)
        self.assertFalse(torch.isinf(result).any())

    def test_forward_on_xpu_device(self):
        """Output is on XPU device."""
        inputs = _generate_synthetic_data(
            T=2, B=4, D=8, num_rows_per_table=5, L=3,
            placements=[0, 0],
            output_dtype=SparseType.FP32, seed=603,
        )
        inputs_xpu = _move_to_device(inputs, "xpu")
        result = _run_forward(inputs_xpu)
        self.assertEqual(result.device.type, "xpu")

    # ===================================================================
    # Determinism
    # ===================================================================

    def test_forward_determinism_device(self):
        """Forward is deterministic on DEVICE."""
        inputs = _generate_synthetic_data(
            T=3, B=16, D=64, num_rows_per_table=10, L=8,
            placements=[0, 0, 0],
            output_dtype=SparseType.FP32, seed=700,
        )
        self._verify_determinism(inputs)

    def test_forward_determinism_mixed(self):
        """Forward is deterministic with mixed placements."""
        inputs = _generate_synthetic_data(
            T=4, B=16, D=64, num_rows_per_table=10, L=8,
            placements=[0, 1, 1, 0],
            output_dtype=SparseType.FP32, seed=701,
        )
        self._verify_determinism(inputs)

    def test_forward_determinism_caching(self):
        """Forward is deterministic with MANAGED_CACHING."""
        inputs = _generate_synthetic_data(
            T=3, B=10, D=64, num_rows_per_table=10, L=8,
            placements=[2, 0, 0],
            output_dtype=SparseType.FP32, seed=702,
            num_cache_slots=32,
            cache_hit_ratio=0.7,
        )
        self._verify_determinism(inputs)

    def _verify_determinism(self, inputs):
        results = []
        for _ in range(3):
            inputs_xpu = _move_to_device(copy.deepcopy(inputs), "xpu")
            result = _run_forward(inputs_xpu)
            results.append(result.cpu().clone())

        for i in range(1, len(results)):
            torch.testing.assert_close(
                results[0], results[i], atol=0, rtol=0,
                msg=f"Forward run {i} differs from run 0",
            )

    # ===================================================================
    # Small embedding dimensions (D < 32)
    # ===================================================================

    def test_forward_small_dim_d4(self):
        """Test with very small dimension D=4."""
        inputs = _generate_synthetic_data(
            T=2, B=8, D=4, num_rows_per_table=10, L=5,
            placements=[0, 0],
            output_dtype=SparseType.FP32, seed=800,
        )
        self._verify_forward(inputs, atol=1e-5, rtol=1e-5)

    def test_forward_small_dim_d8(self):
        """Test with small dimension D=8."""
        inputs = _generate_synthetic_data(
            T=3, B=16, D=8, num_rows_per_table=10, L=3,
            placements=[0, 1, 0],
            output_dtype=SparseType.FP16, seed=801,
        )
        self._verify_forward(inputs, atol=1e-3, rtol=1e-3)

    def test_forward_small_dim_d12(self):
        """Test with dimension D=12."""
        inputs = _generate_synthetic_data(
            T=2, B=8, D=12, num_rows_per_table=10, L=4,
            placements=[0, 1],
            output_dtype=SparseType.FP32, seed=802,
        )
        self._verify_forward(inputs, atol=1e-5, rtol=1e-5)

    def test_forward_small_dim_d16(self):
        """Test with dimension D=16."""
        inputs = _generate_synthetic_data(
            T=4, B=12, D=16, num_rows_per_table=10, L=6,
            placements=[0, 0, 1, 1],
            output_dtype=SparseType.BF16, seed=803,
        )
        self._verify_forward(inputs, atol=1e-2, rtol=1e-2)

    def test_forward_small_dim_d20(self):
        """Test with dimension D=20."""
        inputs = _generate_synthetic_data(
            T=3, B=10, D=20, num_rows_per_table=10, L=5,
            placements=[0, 1, 2],
            output_dtype=SparseType.FP32, seed=804,
            num_cache_slots=16,
            cache_hit_ratio=0.6,
        )
        self._verify_forward(inputs, atol=1e-5, rtol=1e-5)

    def test_forward_small_dim_d24(self):
        """Test with dimension D=24."""
        inputs = _generate_synthetic_data(
            T=2, B=8, D=24, num_rows_per_table=10, L=4,
            placements=[0, 0],
            output_dtype=SparseType.FP16, seed=805,
        )
        self._verify_forward(inputs, atol=1e-3, rtol=1e-3)

    def test_forward_small_dim_d28(self):
        """Test with dimension D=28."""
        inputs = _generate_synthetic_data(
            T=2, B=8, D=28, num_rows_per_table=10, L=3,
            placements=[0, 1],
            output_dtype=SparseType.FP32, seed=806,
        )
        self._verify_forward(inputs, atol=1e-5, rtol=1e-5)

    def test_forward_small_dim_d32(self):
        """Test with dimension D=32 (exact boundary for small kernel)."""
        inputs = _generate_synthetic_data(
            T=3, B=16, D=32, num_rows_per_table=15, L=8,
            placements=[0, 1, 2],
            output_dtype=SparseType.FP16, seed=807,
            num_cache_slots=16,
            cache_hit_ratio=0.5,
        )
        self._verify_forward(inputs, atol=1e-3, rtol=1e-3)

    # ===================================================================
    # Edge cases and boundary conditions
    # ===================================================================

    def test_forward_single_index_per_batch(self):
        """Each batch has exactly one index (L=1)."""
        inputs = _generate_synthetic_data(
            T=3, B=16, D=64, num_rows_per_table=10, L=1,
            placements=[0, 1, 0],
            output_dtype=SparseType.FP32, seed=900,
        )
        self._verify_forward(inputs, atol=1e-5, rtol=1e-5)

    def test_forward_single_row_table(self):
        """Table with only one embedding row."""
        inputs = _generate_synthetic_data(
            T=2, B=8, D=32, num_rows_per_table=1, L=3,
            placements=[0, 0],
            output_dtype=SparseType.FP32, seed=901,
        )
        self._verify_forward(inputs, atol=1e-5, rtol=1e-5)

    def test_forward_many_indices_per_batch(self):
        """Large L value (many indices per batch)."""
        inputs = _generate_synthetic_data(
            T=2, B=4, D=64, num_rows_per_table=20, L=50,
            placements=[0, 0],
            output_dtype=SparseType.FP32, seed=902,
        )
        self._verify_forward(inputs, atol=1e-5, rtol=1e-5)

    def test_forward_single_batch(self):
        """Single batch (B=1)."""
        inputs = _generate_synthetic_data(
            T=3, B=1, D=128, num_rows_per_table=10, L=5,
            placements=[0, 1, 0],
            output_dtype=SparseType.FP32, seed=903,
        )
        self._verify_forward(inputs, atol=1e-5, rtol=1e-5)

    def test_forward_minimal_config(self):
        """Minimal configuration: T=1, B=1, D=4, L=1."""
        inputs = _generate_synthetic_data(
            T=1, B=1, D=4, num_rows_per_table=5, L=1,
            placements=[0],
            output_dtype=SparseType.FP32, seed=904,
        )
        self._verify_forward(inputs, atol=1e-5, rtol=1e-5)

    # ===================================================================
    # Large scale / performance tests
    # ===================================================================

    def test_forward_very_large_dim(self):
        """Very large embedding dimension D=512."""
        inputs = _generate_synthetic_data(
            T=2, B=8, D=512, num_rows_per_table=10, L=5,
            placements=[0, 0],
            output_dtype=SparseType.FP32, seed=1000,
        )
        self._verify_forward(inputs, atol=1e-5, rtol=1e-5)

    def test_forward_large_dim_1024(self):
        """Large dimension D=1024."""
        inputs = _generate_synthetic_data(
            T=2, B=8, D=1024, num_rows_per_table=10, L=4,
            placements=[0, 1],
            output_dtype=SparseType.FP16, seed=1001,
        )
        self._verify_forward(inputs, atol=1e-3, rtol=1e-3)

    def test_forward_many_tables(self):
        """Many tables (T=12)."""
        inputs = _generate_synthetic_data(
            T=12, B=8, D=64, num_rows_per_table=10, L=5,
            placements=[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            output_dtype=SparseType.FP32, seed=1002,
        )
        self._verify_forward(inputs, atol=1e-5, rtol=1e-5)

    def test_forward_very_large_batch(self):
        """Very large batch size (B=512)."""
        inputs = _generate_synthetic_data(
            T=2, B=512, D=32, num_rows_per_table=10, L=3,
            placements=[0, 0],
            output_dtype=SparseType.FP16, seed=1003,
        )
        self._verify_forward(inputs, atol=1e-3, rtol=1e-3)

    def test_forward_large_total_indices(self):
        """Large total number of indices."""
        inputs = _generate_synthetic_data(
            T=4, B=128, D=128, num_rows_per_table=20, L=20,
            placements=[0, 1, 0, 1],
            output_dtype=SparseType.FP32, seed=1004,
        )
        self._verify_forward(inputs, atol=1e-5, rtol=1e-5)

    # ===================================================================
    # All MANAGED placement (type 1)
    # ===================================================================

    def test_forward_all_managed_fp32(self):
        """All tables with MANAGED placement, FP32."""
        inputs = _generate_synthetic_data(
            T=4, B=16, D=128, num_rows_per_table=10, L=8,
            placements=[1, 1, 1, 1],
            output_dtype=SparseType.FP32, seed=1100,
        )
        self._verify_forward(inputs, atol=1e-5, rtol=1e-5)

    def test_forward_all_managed_fp16(self):
        """All tables with MANAGED placement, FP16."""
        inputs = _generate_synthetic_data(
            T=3, B=20, D=64, num_rows_per_table=10, L=5,
            placements=[1, 1, 1],
            output_dtype=SparseType.FP16, seed=1101,
        )
        self._verify_forward(inputs, atol=1e-3, rtol=1e-3)

    def test_forward_all_managed_bf16(self):
        """All tables with MANAGED placement, BF16."""
        inputs = _generate_synthetic_data(
            T=5, B=24, D=256, num_rows_per_table=10, L=10,
            placements=[1, 1, 1, 1, 1],
            output_dtype=SparseType.BF16, seed=1102,
        )
        self._verify_forward(inputs, atol=1e-2, rtol=1e-2)

    # ===================================================================
    # Mixed all three placement types (0, 1, 2)
    # ===================================================================

    def test_forward_mixed_all_three_placements_fp32(self):
        """Mix of DEVICE, MANAGED, and MANAGED_CACHING, FP32."""
        inputs = _generate_synthetic_data(
            T=6, B=20, D=128, num_rows_per_table=10, L=8,
            placements=[0, 1, 2, 0, 1, 2],
            output_dtype=SparseType.FP32, seed=1200,
            num_cache_slots=32,
            cache_hit_ratio=0.6,
        )
        self._verify_forward(inputs, atol=1e-5, rtol=1e-5)

    def test_forward_mixed_all_three_placements_fp16(self):
        """Mix of all three placements, FP16."""
        inputs = _generate_synthetic_data(
            T=9, B=16, D=64, num_rows_per_table=10, L=5,
            placements=[0, 1, 2, 0, 1, 2, 0, 1, 2],
            output_dtype=SparseType.FP16, seed=1201,
            num_cache_slots=48,
            cache_hit_ratio=0.7,
        )
        self._verify_forward(inputs, atol=1e-3, rtol=1e-3)

    def test_forward_mixed_all_three_placements_bf16(self):
        """Mix of all three placements, BF16."""
        inputs = _generate_synthetic_data(
            T=6, B=32, D=256, num_rows_per_table=10, L=10,
            placements=[2, 1, 0, 2, 1, 0],
            output_dtype=SparseType.BF16, seed=1202,
            num_cache_slots=64,
            cache_hit_ratio=0.5,
        )
        self._verify_forward(inputs, atol=1e-2, rtol=1e-2)

    # ===================================================================
    # Different info_B_num_bits values
    # ===================================================================

    def test_forward_info_B_bits_26(self):
        """Test with info_B_num_bits=26."""
        inputs = _generate_synthetic_data(
            T=3, B=16, D=64, num_rows_per_table=10, L=5,
            placements=[0, 1, 0],
            info_B_num_bits=26,
            output_dtype=SparseType.FP32, seed=1300,
        )
        self._verify_forward(inputs, atol=1e-5, rtol=1e-5)

    def test_forward_info_B_bits_28(self):
        """Test with info_B_num_bits=28."""
        inputs = _generate_synthetic_data(
            T=4, B=32, D=128, num_rows_per_table=10, L=8,
            placements=[0, 0, 1, 1],
            info_B_num_bits=28,
            output_dtype=SparseType.FP16, seed=1301,
        )
        self._verify_forward(inputs, atol=1e-3, rtol=1e-3)

    def test_forward_info_B_bits_31(self):
        """Test with info_B_num_bits=31."""
        inputs = _generate_synthetic_data(
            T=2, B=64, D=32, num_rows_per_table=10, L=4,
            placements=[0, 1],
            info_B_num_bits=31,
            output_dtype=SparseType.FP32, seed=1302,
        )
        self._verify_forward(inputs, atol=1e-5, rtol=1e-5)

    def test_forward_info_B_bits_29_with_caching(self):
        """Test info_B_num_bits=29 with MANAGED_CACHING."""
        inputs = _generate_synthetic_data(
            T=3, B=20, D=64, num_rows_per_table=10, L=6,
            placements=[2, 0, 1],
            info_B_num_bits=29,
            output_dtype=SparseType.FP32, seed=1303,
            num_cache_slots=32,
            cache_hit_ratio=0.7,
        )
        self._verify_forward(inputs, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
