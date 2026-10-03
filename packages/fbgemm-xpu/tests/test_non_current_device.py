"""Operators must run on the queue of their tensors' device, not the current one.

Inputs live on xpu:1 while xpu:0 stays current, and they are produced by a
slow kernel chain on xpu:1. A kernel submitted to the xpu:0 queue would read
them before they are written. Results are compared with FBGEMM's CPU kernels.
"""

import functools
import importlib
import os

import pytest
import torch

ITERATIONS = 5


@pytest.fixture
def devices():
    importlib.import_module("fbgemm_xpu")
    if not torch.xpu.is_available():
        pytest.fail("A real XPU is required; no CPU fallback")
    if torch.xpu.device_count() < 2:
        pytest.fail("Requires two visible XPU devices")
    torch.xpu.set_device(0)
    return torch.device("xpu:0"), torch.device("xpu:1")


def _late(tensors, device):
    """Copy tensors to device so that they become valid only after slow work there."""
    weight = torch.randn((2048, 2048), device=device)
    slow = weight
    for _ in range(8):
        slow = slow @ weight
    zero = slow[0, 0] * 0
    return [
        tensor.to(device) + zero.to(tensor.dtype) if tensor is not None else None
        for tensor in tensors
    ]


def _assert_same(actual, expected, rtol=0.0, atol=0.0):
    if isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_same(actual_item, expected_item, rtol, atol)
    elif expected is None:
        assert actual is None
    else:
        torch.testing.assert_close(actual.cpu(), expected, rtol=rtol, atol=atol)


def _run(operator, inputs, devices, *args, rtol=0.0, atol=0.0):
    current, data = devices
    expected = operator(*inputs, *args)
    for _ in range(ITERATIONS):
        actual = operator(*_late(inputs, data), *args)
        torch.xpu.synchronize(data)
        torch.xpu.synchronize(current)
        assert torch.xpu.current_device() == current.index
        _assert_same(actual, expected, rtol, atol)


def _sparse_features():
    generator = torch.Generator().manual_seed(0)
    lengths = torch.randint(0, 9, (12,), generator=generator, dtype=torch.int32)
    indices = torch.randint(0, 1000, (int(lengths.sum()),), generator=generator, dtype=torch.int32)
    return lengths, indices


pytestmark = pytest.mark.skipif(
    os.environ.get("FBGEMM_XPU_CROSS_TILE_TESTS") != "1",
    reason="Requires two XPU devices and a passed transfer gate",
)


def test_block_bucketize_sparse_features_inference(devices):
    lengths, indices = _sparse_features()
    block_sizes = torch.full((3,), 250, dtype=torch.int32)

    def operator(lengths, indices, block_sizes):
        return torch.ops.fbgemm.block_bucketize_sparse_features_inference(
            lengths, indices, False, True, block_sizes, 4, None,
            return_bucket_mapping=True,
        )

    _run(operator, [lengths, indices, block_sizes], devices)


def test_block_bucketize_sparse_features(devices):
    lengths, indices = _sparse_features()
    block_sizes = torch.full((3,), 250, dtype=torch.int32)

    def operator(lengths, indices, block_sizes):
        return torch.ops.fbgemm.block_bucketize_sparse_features(
            lengths, indices, False, True, block_sizes, 4, None
        )

    _run(operator, [lengths, indices, block_sizes], devices)


def test_populate_bucketized_permute(devices):
    lengths, indices = _sparse_features()
    block_sizes = torch.full((3,), 250, dtype=torch.int32)
    bucketized = torch.ops.fbgemm.block_bucketize_sparse_features_inference(
        lengths, indices, False, True, block_sizes, 4, None, return_bucket_mapping=True
    )
    _run(
        torch.ops.fbgemm.populate_bucketized_permute,
        [lengths, bucketized[0], bucketized[5]],
        devices,
    )


@pytest.mark.parametrize("name", ["permute_2D_sparse_data", "permute_1D_sparse_data"])
def test_permute_sparse_data(devices, name):
    lengths, indices = _sparse_features()
    if name == "permute_2D_sparse_data":
        lengths = lengths.view(3, 4)
    permute = torch.tensor([2, 0, 1], dtype=torch.int32)
    if name == "permute_1D_sparse_data":
        permute = torch.randperm(12, generator=torch.Generator().manual_seed(1)).int()
    _run(getattr(torch.ops.fbgemm, name), [permute, lengths, indices], devices)


def test_invert_permute(devices):
    permute = torch.randperm(1000, generator=torch.Generator().manual_seed(2)).int()
    _run(torch.ops.fbgemm.invert_permute, [permute], devices)


def test_expand_into_jagged_permute(devices):
    permute = torch.tensor([2, 0, 1], dtype=torch.int32)
    lengths = torch.tensor([3, 5, 2], dtype=torch.int32)
    input_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
    output_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths[permute.long()])
    _run(
        torch.ops.fbgemm.expand_into_jagged_permute,
        [permute, input_offsets, output_offsets],
        devices,
        int(lengths.sum()),
    )


def test_jagged_index_select_2d_forward(devices):
    generator = torch.Generator().manual_seed(3)
    lengths = torch.tensor([3, 0, 5, 2], dtype=torch.int64)
    values = torch.randn((int(lengths.sum()), 16), generator=generator)
    indices = torch.tensor([2, 0, 3, 2], dtype=torch.int64)
    input_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)[1:]
    output_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths[indices])[1:]
    _run(
        torch.ops.fbgemm.jagged_index_select_2d_forward,
        [values, indices, input_offsets, output_offsets],
        devices,
        int(output_offsets[-1]),
    )


def _quantized_table(bits, rows, dimension, seed):
    from fbgemm_gpu.split_embedding_configs import SparseType
    from fbgemm_gpu.tbe.utils.quantize import quantize_embs

    values = torch.randn((rows, dimension), generator=torch.Generator().manual_seed(seed))
    packed, scale_bias = quantize_embs(values, SparseType.INT4 if bits == 4 else SparseType.INT8)
    row_bytes = scale_bias.shape[1] + packed.shape[1]
    table = torch.zeros((rows, (row_bytes + 15) // 16 * 16), dtype=torch.uint8)
    table[:, : scale_bias.shape[1]] = scale_bias
    table[:, scale_bias.shape[1] : row_bytes] = packed
    return table.flatten()


def _int_nbit_lookup(
    dev_weights, weights_offsets, weights_tys, D_offsets, indices, offsets, *, dimension, bits
):
    device = dev_weights.device
    tables = weights_offsets.numel()
    return torch.ops.fbgemm.int_nbit_split_embedding_codegen_lookup_function(
        dev_weights=dev_weights,
        uvm_weights=torch.empty(0, dtype=torch.uint8, device=device),
        weights_placements=torch.full(
            (tables,), 0 if device.type == "xpu" else 3, dtype=torch.int32, device=device
        ),
        weights_offsets=weights_offsets,
        weights_tys=weights_tys,
        D_offsets=D_offsets,
        total_D=dimension * tables,
        max_int2_D=0,
        max_int4_D=dimension if bits == 4 else 0,
        max_int8_D=dimension if bits == 8 else 0,
        max_float16_D=0,
        max_float32_D=0,
        indices=indices,
        offsets=offsets,
        pooling_mode=2,
        indice_weights=None,
        output_dtype=0,
        row_alignment=16,
    )


@pytest.mark.parametrize("bits", [4, 8])
def test_int_nbit_split_embedding_codegen_lookup_function(devices, bits):
    dimension, rows, batch = 64, [40, 24], 4
    generator = torch.Generator().manual_seed(4)
    tables = [_quantized_table(bits, count, dimension, seed) for seed, count in enumerate(rows)]
    lengths = torch.randint(0, 6, (len(rows) * batch,), generator=generator)
    indices = torch.cat(
        [
            torch.randint(0, count, (int(lengths[t * batch : (t + 1) * batch].sum()),), generator=generator)
            for t, count in enumerate(rows)
        ]
    ).int()
    _run(
        functools.partial(_int_nbit_lookup, dimension=dimension, bits=bits),
        [
            torch.cat(tables),
            torch.tensor([0, tables[0].numel()], dtype=torch.int64),
            torch.full((len(rows),), 3 if bits == 4 else 2, dtype=torch.uint8),
            torch.arange(len(rows) + 1, dtype=torch.int32) * dimension,
            indices,
            torch.ops.fbgemm.asynchronous_complete_cumsum(lengths).int(),
        ],
        devices,
        rtol=1e-5,
        atol=1e-6,
    )


def _bounds_check_warning(rows_per_table, indices, offsets, warning):
    indices, warning = indices.clone(), warning.clone()
    torch.ops.fbgemm.bounds_check_indices(
        rows_per_table, indices, offsets, 1, warning, None, None, -1, None, -1, -1, 1, False
    )
    return indices, warning


def test_bounds_check_indices(devices):
    _run(
        _bounds_check_warning,
        [
            torch.tensor([3, 2], dtype=torch.int64),
            torch.tensor([0, 3, -2, -1, 2, 0, 2, -3, 1], dtype=torch.int64),
            torch.tensor([0, 3, 5, 7, 9], dtype=torch.int64),
            torch.zeros(1, dtype=torch.int64),
        ],
        devices,
    )


def _jagged_inputs():
    generator = torch.Generator().manual_seed(5)
    offsets = torch.tensor([0, 3, 3, 8, 10], dtype=torch.int64)
    values = torch.randn((10, 8), generator=generator)
    dense = torch.randn((4, 5, 8), generator=generator)
    return values, offsets, dense


def test_jagged_to_padded_dense_forward(devices):
    values, offsets, _ = _jagged_inputs()
    _run(
        lambda values, offsets: torch.ops.fbgemm.jagged_to_padded_dense_forward(
            values, [offsets], [4], -1.0
        ),
        [values, offsets],
        devices,
    )


def test_jagged_to_padded_dense_backward(devices):
    _, offsets, dense = _jagged_inputs()
    _run(
        lambda grad, offsets: torch.ops.fbgemm.jagged_to_padded_dense_backward(
            grad, [offsets], 10
        ),
        [dense, offsets],
        devices,
    )


def test_dense_to_jagged_forward(devices):
    _, offsets, dense = _jagged_inputs()
    _run(
        lambda dense, offsets: torch.ops.fbgemm.dense_to_jagged_forward(dense, [offsets], 10),
        [dense, offsets],
        devices,
    )


def test_jagged_dense_elementwise_add_jagged_output(devices):
    values, offsets, dense = _jagged_inputs()
    _run(
        lambda values, offsets, dense: torch.ops.fbgemm.jagged_dense_elementwise_add_jagged_output(
            values, [offsets], dense
        ),
        [values, offsets, dense],
        devices,
    )


def _batched_ads():
    generator = torch.Generator().manual_seed(6)
    tables, num_ads_in_batch = 2, 6
    batch_offsets = torch.tensor([0, 2, 5, 6], dtype=torch.int32)
    cat_ad_lengths = torch.randint(0, 5, (tables * num_ads_in_batch,), generator=generator).int()
    return batch_offsets, cat_ad_lengths, num_ads_in_batch


def test_reorder_batched_ad_lengths(devices):
    batch_offsets, cat_ad_lengths, num_ads_in_batch = _batched_ads()
    _run(
        torch.ops.fbgemm.reorder_batched_ad_lengths,
        [cat_ad_lengths, batch_offsets],
        devices,
        num_ads_in_batch,
    )


def test_reorder_batched_ad_indices(devices):
    batch_offsets, cat_ad_lengths, num_ads_in_batch = _batched_ads()
    reordered_lengths = torch.ops.fbgemm.reorder_batched_ad_lengths(
        cat_ad_lengths, batch_offsets, num_ads_in_batch
    )
    cat_ad_indices = torch.randint(
        0, 1000, (int(cat_ad_lengths.sum()),), generator=torch.Generator().manual_seed(7)
    )
    _run(
        torch.ops.fbgemm.reorder_batched_ad_indices,
        [
            torch.ops.fbgemm.asynchronous_complete_cumsum(cat_ad_lengths),
            cat_ad_indices,
            torch.ops.fbgemm.asynchronous_complete_cumsum(reordered_lengths),
            batch_offsets,
        ],
        devices,
        num_ads_in_batch,
    )


def test_permute_2D_sparse_preallocated_out(devices):
    lengths, indices = _sparse_features()
    _run(
        torch.ops.fbgemm.permute_2D_sparse_preallocated_out,
        [torch.tensor([2, 0, 1], dtype=torch.int32), lengths.view(3, 4), indices],
        devices,
    )
