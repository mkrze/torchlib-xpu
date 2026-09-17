#pragma once

#include <algorithm>
#include <cstdint>
#include <sycl/sycl.hpp>

namespace fbgemm_xpu {

template <typename index_t, typename output_t>
void lookup_int{{ bit_rate }}_nobag(
    sycl::queue& queue,
    const uint8_t* weights,
    const index_t* indices,
    output_t* output,
    int64_t begin,
    int64_t end,
    int64_t dimension,
    int64_t row_stride,
    int64_t storage_rows,
    int32_t* error) {
    const int64_t elements = (end - begin) * dimension;
    if (elements == 0) {
        return;
    }
    const int64_t work_items = std::min<int64_t>(
        ((elements + 255) / 256) * 256, 4096 * 256);
    queue.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(work_items), sycl::range<1>(256)),
        [=](sycl::nd_item<1> item) {
            for (int64_t element = item.get_global_linear_id();
                 element < elements;
                 element += work_items) {
                const int64_t position = begin + element / dimension;
                const int64_t column = element % dimension;
                const int64_t row_index = indices[position];
                if (row_index < 0 || row_index >= storage_rows) {
                    if (column == 0) {
                        sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
                                         sycl::memory_scope::device,
                                         sycl::access::address_space::global_space>
                            error_ref(*error);
                        error_ref.fetch_or(row_index < 0 ? 1 : 2);
                    }
                    continue;
                }
                const uint8_t* row = weights + row_index * row_stride;
                const uint16_t scale_bits = uint16_t(row[0]) | (uint16_t(row[1]) << 8);
                const uint16_t bias_bits = uint16_t(row[2]) | (uint16_t(row[3]) << 8);
                const float scale = static_cast<float>(sycl::bit_cast<sycl::half>(scale_bits));
                const float bias = static_cast<float>(sycl::bit_cast<sycl::half>(bias_bits));
                const uint8_t packed = row[4 + column / {{ 8 // bit_rate }}];
                const int quantized = (packed >> ((column % {{ 8 // bit_rate }}) * {{ bit_rate }})) & {{ (2 ** bit_rate) - 1 }};
                output[position * dimension + column] =
                    static_cast<output_t>(sycl::fma(static_cast<float>(quantized), scale, bias));
            }
        });
}

}