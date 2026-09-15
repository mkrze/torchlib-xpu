/*
 * Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
 * Copyright (c) 2026 Intel Corporation. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>

#include <sycl/sycl.hpp>

#include <ATen/ATen.h>
#include <c10/macros/Macros.h>

namespace fbgemm_xpu {

// Values are part of FBGEMM's public bounds_check_indices contract.
enum class BoundsCheckMode : int64_t {
  FATAL = 0,
  WARNING = 1,
  IGNORE = 2,
};

template <typename index_t> class BoundsCheckOffsetsKernel {
public:
  BoundsCheckOffsetsKernel(index_t *offsets, int64_t offsets_stride,
                           int64_t total_B, int64_t num_indices,
                           int64_t *warning, int64_t *offsets_invalid,
                           int64_t *fatal_error,
                           BoundsCheckMode bounds_check_mode)
      : offsets_(offsets), offsets_stride_(offsets_stride), total_B_(total_B),
        num_indices_(num_indices), warning_(warning),
        offsets_invalid_(offsets_invalid), fatal_error_(fatal_error),
        bounds_check_mode_(bounds_check_mode) {}

  void operator()(const sycl::nd_item<1> &item) const {
    const index_t num_indices = static_cast<index_t>(num_indices_);
    if (item.get_global_id(0) == 0 &&
        offsets_[total_B_ * offsets_stride_] != num_indices) {
      mark_invalid();
    }

    const uint64_t stride = item.get_global_range(0);
    for (uint64_t b_t = item.get_global_id(0);
         b_t < static_cast<uint64_t>(total_B_); b_t += stride) {
      const index_t indices_start = offsets_[b_t * offsets_stride_];
      // CPU repairs the terminal offset before checking individual bags.
      const index_t indices_end = b_t + 1 == static_cast<uint64_t>(total_B_)
                                      ? num_indices
                                      : offsets_[(b_t + 1) * offsets_stride_];
      if (indices_start < 0 || indices_start > indices_end ||
          indices_end > num_indices) {
        mark_invalid();
      }
    }
  }

private:
  void mark_invalid() const {
    sycl::atomic_ref<int64_t, sycl::memory_order::relaxed,
                     sycl::memory_scope::device,
                     sycl::access::address_space::global_space>
        invalid_ref(offsets_invalid_[0]);
    invalid_ref.store(1);
    if (bounds_check_mode_ == BoundsCheckMode::FATAL) {
      set_fatal_error();
    } else if (bounds_check_mode_ == BoundsCheckMode::WARNING) {
      warn_once();
    }
  }

  int64_t increment_warning() const {
    sycl::atomic_ref<int64_t, sycl::memory_order::relaxed,
                     sycl::memory_scope::device,
                     sycl::access::address_space::global_space>
        warning_ref(warning_[0]);
    return warning_ref.fetch_add(1);
  }

  void warn_once() const {
    if (increment_warning() == 0) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
      static const __attribute__((opencl_constant)) char message[] =
          "EmbeddingBoundsCheck: at least one out-of-bounds access was "
          "corrected.\n";
#else
      static const char message[] =
          "EmbeddingBoundsCheck: at least one out-of-bounds access was "
          "corrected.\n";
#endif
      sycl::ext::oneapi::experimental::printf(message);
    }
  }

  void set_fatal_error() const {
    sycl::atomic_ref<int64_t, sycl::memory_order::relaxed,
                     sycl::memory_scope::device,
                     sycl::access::address_space::global_space>
        error_ref(fatal_error_[0]);
    error_ref.store(1);
  }

  index_t *offsets_;
  int64_t offsets_stride_;
  int64_t total_B_;
  int64_t num_indices_;
  int64_t *warning_;
  int64_t *offsets_invalid_;
  int64_t *fatal_error_;
  BoundsCheckMode bounds_check_mode_;
};

template <typename index_t> class RepairBoundsCheckOffsetsKernel {
public:
  RepairBoundsCheckOffsetsKernel(index_t *offsets, int64_t offsets_stride,
                                 int64_t total_B, int64_t num_indices,
                                 const int64_t *offsets_invalid,
                                 BoundsCheckMode bounds_check_mode)
      : offsets_(offsets), offsets_stride_(offsets_stride), total_B_(total_B),
        num_indices_(num_indices), offsets_invalid_(offsets_invalid),
        bounds_check_mode_(bounds_check_mode) {}

  void operator()(const sycl::id<1> &) const {
    if (offsets_invalid_[0] == 0 ||
        bounds_check_mode_ == BoundsCheckMode::FATAL) {
      return;
    }

    const index_t num_indices = static_cast<index_t>(num_indices_);
    offsets_[total_B_ * offsets_stride_] = num_indices;
    for (int64_t b_t = 0; b_t < total_B_; ++b_t) {
      index_t indices_start = offsets_[b_t * offsets_stride_];
      index_t indices_end = offsets_[(b_t + 1) * offsets_stride_];
      indices_start = std::max(static_cast<index_t>(0),
                               std::min(indices_start, num_indices));
      indices_end = std::max(indices_start, std::min(indices_end, num_indices));
      offsets_[b_t * offsets_stride_] = indices_start;
      offsets_[(b_t + 1) * offsets_stride_] = indices_end;
    }
  }

private:
  index_t *offsets_;
  int64_t offsets_stride_;
  int64_t total_B_;
  int64_t num_indices_;
  const int64_t *offsets_invalid_;
  BoundsCheckMode bounds_check_mode_;
};

template <typename index_t, bool vbe> class BoundsCheckIndicesKernelV1 {
public:
  BoundsCheckIndicesKernelV1(const int64_t *rows_per_table,
                             int64_t rows_per_table_stride, index_t *indices,
                             int64_t indices_stride, index_t *offsets,
                             int64_t offsets_stride, const int32_t *B_offsets,
                             int64_t B_offsets_stride, int64_t *warning,
                             const int64_t *offsets_invalid,
                             int64_t *fatal_error, int64_t T, int64_t total_B,
                             int64_t max_B, BoundsCheckMode bounds_check_mode)
      : rows_per_table_(rows_per_table),
        rows_per_table_stride_(rows_per_table_stride), indices_(indices),
        indices_stride_(indices_stride), offsets_(offsets),
        offsets_stride_(offsets_stride), B_offsets_(B_offsets),
        B_offsets_stride_(B_offsets_stride), warning_(warning),
        offsets_invalid_(offsets_invalid), fatal_error_(fatal_error), T_(T),
        total_B_(total_B), max_B_(max_B),
        bounds_check_mode_(bounds_check_mode) {}

  void operator()(const sycl::nd_item<2> &item) const {
    const int64_t lane = item.get_local_id(1);
    const uint64_t first_bt =
        item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
    const uint64_t bt_stride =
        item.get_group_range(0) * item.get_local_range(0);
    const uint64_t logical_warps =
        static_cast<uint64_t>(max_B_) * static_cast<uint64_t>(T_);

    // The host caps the number of work-groups to keep the flattened SYCL
    // range within INT_MAX. Grid-striding preserves coverage when capped.
    for (uint64_t bt0 = first_bt; bt0 < logical_warps; bt0 += bt_stride) {
      const int64_t t = static_cast<int64_t>(bt0 / max_B_);
      const int64_t b = static_cast<int64_t>(bt0 % max_B_);
      int64_t b_t = static_cast<int64_t>(bt0);
      int64_t B = max_B_;

      if constexpr (vbe) {
        if (t >= T_) {
          continue;
        }
        const int64_t B_start = B_offsets_[t * B_offsets_stride_];
        B = static_cast<int64_t>(B_offsets_[(t + 1) * B_offsets_stride_]) -
            B_start;
        if (b >= B) {
          continue;
        }
        b_t = B_start + b;
      } else if (b_t >= total_B_) {
        continue;
      }

      if (bounds_check_mode_ == BoundsCheckMode::FATAL &&
          offsets_invalid_[0] != 0) {
        continue;
      }

      const int64_t num_rows = rows_per_table_[t * rows_per_table_stride_];
      const index_t indices_start = offsets_[b_t * offsets_stride_];
      const index_t indices_end = offsets_[(b_t + 1) * offsets_stride_];

      const index_t L = indices_end - indices_start;
      for (int64_t i = lane; i < static_cast<int64_t>(L); i += kWarpSize) {
        const int64_t position = static_cast<int64_t>(indices_start) + i;
        const index_t idx = indices_[position * indices_stride_];
        if (idx == static_cast<index_t>(-1)) {
          // -1 is the sentinel for a pruned row.
          continue;
        }

        const bool out_of_bounds = idx < 0 || idx >= num_rows;
        if (bounds_check_mode_ == BoundsCheckMode::FATAL) {
          if (out_of_bounds) {
            set_fatal_error();
          }
        } else if (out_of_bounds) {
          if (bounds_check_mode_ == BoundsCheckMode::WARNING) {
            warn_once();
          }
          indices_[position * indices_stride_] = 0;
        }
      }
    }
  }

private:
  int64_t increment_warning() const {
    sycl::atomic_ref<int64_t, sycl::memory_order::relaxed,
                     sycl::memory_scope::device,
                     sycl::access::address_space::global_space>
        warning_ref(warning_[0]);
    return warning_ref.fetch_add(1);
  }

  void warn_once() const {
    if (increment_warning() == 0) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
      static const __attribute__((opencl_constant)) char message[] =
          "EmbeddingBoundsCheck: at least one out-of-bounds access was "
          "corrected.\n";
#else
      static const char message[] =
          "EmbeddingBoundsCheck: at least one out-of-bounds access was "
          "corrected.\n";
#endif
      sycl::ext::oneapi::experimental::printf(message);
    }
  }

  void set_fatal_error() const {
    sycl::atomic_ref<int64_t, sycl::memory_order::relaxed,
                     sycl::memory_scope::device,
                     sycl::access::address_space::global_space>
        error_ref(fatal_error_[0]);
    error_ref.store(1);
  }

  static constexpr int64_t kWarpSize = 32;

  const int64_t *rows_per_table_;
  int64_t rows_per_table_stride_;
  index_t *indices_;
  int64_t indices_stride_;
  index_t *offsets_;
  int64_t offsets_stride_;
  const int32_t *B_offsets_;
  int64_t B_offsets_stride_;
  int64_t *warning_;
  const int64_t *offsets_invalid_;
  int64_t *fatal_error_;
  int64_t T_;
  int64_t total_B_;
  int64_t max_B_;
  BoundsCheckMode bounds_check_mode_;
};

void bounds_check_indices_xpu(
    at::Tensor &rows_per_table, at::Tensor &indices, at::Tensor &offsets,
    int64_t bounds_check_mode, at::Tensor &warning,
    const std::optional<at::Tensor> &weights,
    const std::optional<at::Tensor> &B_offsets, int64_t max_B,
    const std::optional<at::Tensor> &b_t_map, int64_t info_B_num_bits,
    int64_t info_B_mask, int8_t bounds_check_version, bool prefetch_pipeline);

} // namespace fbgemm_xpu
