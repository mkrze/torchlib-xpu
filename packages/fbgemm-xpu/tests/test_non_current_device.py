"""Operators must run on the queue of their tensors' device, not the current one.

Inputs live on xpu:1 while xpu:0 stays current, and they are produced by a
slow kernel chain on xpu:1. A kernel submitted to the xpu:0 queue would read
them before they are written. Results are compared with FBGEMM's CPU kernels.
"""

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


def _assert_same(actual, expected):
    if isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_same(actual_item, expected_item)
    elif expected is None:
        assert actual is None
    else:
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


def _run(operator, inputs, devices, *args):
    current, data = devices
    expected = operator(*inputs, *args)
    for _ in range(ITERATIONS):
        actual = operator(*_late(inputs, data), *args)
        torch.xpu.synchronize(data)
        torch.xpu.synchronize(current)
        assert torch.xpu.current_device() == current.index
        _assert_same(actual, expected)


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
