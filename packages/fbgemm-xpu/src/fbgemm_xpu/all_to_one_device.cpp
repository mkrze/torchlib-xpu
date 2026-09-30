#include <ATen/core/Tensor.h>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>

#include <vector>

namespace fbgemm_xpu {
namespace {

std::vector<at::Tensor> all_to_one_device_xpu(
    std::vector<at::Tensor> input_tensors,
    at::Device target_device) {
    TORCH_CHECK(target_device.is_xpu(),
                "all_to_one_device: target_device must be XPU");
    for (const auto& tensor : input_tensors) {
        TORCH_CHECK(tensor.is_xpu(),
                    "all_to_one_device: all input tensors must be on XPU");
        TORCH_CHECK(tensor.layout() == at::kStrided,
                    "all_to_one_device: only strided tensors are supported");
    }

    const c10::DeviceGuard guard(target_device);
    const auto device = guard.current_device();
    std::vector<at::Tensor> output_tensors;
    output_tensors.reserve(input_tensors.size());
    for (const auto& tensor : input_tensors) {
        output_tensors.push_back(
            tensor.device() == device
                ? tensor
                : tensor.to(device, tensor.scalar_type(), true, false));
    }
    return output_tensors;
}

}

TORCH_LIBRARY_IMPL(fbgemm, XPU, library) {
    library.impl("all_to_one_device", TORCH_FN(all_to_one_device_xpu));
}

}