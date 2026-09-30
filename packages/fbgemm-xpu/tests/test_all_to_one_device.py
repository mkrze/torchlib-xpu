import importlib
import os

import pytest
import torch


@pytest.fixture
def xpu():
    importlib.import_module("fbgemm_xpu")
    if not torch.xpu.is_available():
        pytest.fail("A real XPU is required; no CPU fallback")
    if not torch._C._dispatch_has_kernel_for_dispatch_key("fbgemm::all_to_one_device", "XPU"):
        pytest.fail("The installed plugin must include all_to_one_device")
    return torch.device("xpu:0")


def test_same_device_preserves_views_and_empty_tensors(xpu):
    torch.xpu.set_device(xpu)
    base = torch.arange(24, device=xpu).reshape(4, 6)
    tensors = [base, base[:, ::2], torch.empty((0, 5), device=xpu), torch.ones((), device=xpu)]
    for target in (xpu, torch.device("xpu")):
        outputs = torch.ops.fbgemm.all_to_one_device(tensors, target)
        if len(outputs) != len(tensors):
            pytest.fail("Output ordering/cardinality must match inputs")
        for actual, expected in zip(outputs, tensors):
            torch.testing.assert_close(actual, expected)
            if actual.data_ptr() != expected.data_ptr() or actual.stride() != expected.stride():
                pytest.fail("Same-device outputs must alias the original tensor views")


def test_rejects_unsupported_device_inputs(xpu):
    tensor = torch.ones(2, device=xpu)
    with pytest.raises(RuntimeError, match="target_device must be XPU"):
        torch.ops.fbgemm.all_to_one_device([tensor], torch.device("cpu"))
    with pytest.raises(RuntimeError, match="all input tensors must be on XPU"):
        torch.ops.fbgemm.all_to_one_device([tensor, torch.ones(2)], xpu)


@pytest.mark.skipif(
    os.environ.get("FBGEMM_XPU_CROSS_TILE_TESTS") != "1",
    reason="Requires an active DUT reservation and a passed two-tile transfer gate",
)
def test_cross_tile_stream_copy_and_lifetime(xpu):
    if torch.xpu.device_count() < 2:
        pytest.fail("Cross-tile validation requires two visible XPU devices")
    for source_index, target_index in ((0, 1), (1, 0)):
        source = torch.device(f"xpu:{source_index}")
        target = torch.device(f"xpu:{target_index}")
        producer = torch.xpu.Stream(device=source)
        consumer = torch.xpu.Stream(device=target)
        with torch.xpu.stream(producer):
            inputs = [
                torch.arange(48, device=source).reshape(6, 8)[:, ::2],
                torch.full((3, 7), 2.5, dtype=torch.float16, device=source),
                torch.empty((0, 4), device=source),
            ]
        torch.xpu.current_stream(source).wait_stream(producer)
        with torch.xpu.stream(consumer):
            outputs = torch.ops.fbgemm.all_to_one_device(inputs, target)
            del inputs
            consumed = [tensor.clone() for tensor in outputs]
        consumer.synchronize()
        expected = [torch.arange(48).reshape(6, 8)[:, ::2], torch.full((3, 7), 2.5, dtype=torch.float16), torch.empty((0, 4))]
        for actual, reference in zip(consumed, expected):
            if actual.device != target:
                pytest.fail("Copy returned a tensor on the wrong target")
            torch.testing.assert_close(actual.cpu(), reference)