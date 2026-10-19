/*
 * Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
 * Copyright (c) 2026 Intel Corporation. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

////////////////////////////////////////////////////////////////////////////////
// HSTU ATTENTION FORWARD FOR XPU (EXPERIMENTAL)
////////////////////////////////////////////////////////////////////////////////
//
// XPU implementation of the inference forward of hstu::hstu_mha from
// generative-recommenders (ops/cpp/hstu_attention, CUDA only). Semantics follow
// the Triton reference _hstu_attn_fwd_one_block in
// generative_recommenders/ops/triton/triton_hstu_attention.py:
//
//   out[m] = sum_n silu(alpha * q[m] . k[n]) / max_seq_len * valid(m, n) * v[n]
//
// with the causal, target and contextual masks of HSTU and jagged sequences
// given by seq_offsets. There is no softmax, so key blocks accumulate
// independently.
//
// Each sub-group of 16 work-items owns 8 query rows of one (sequence, head):
// Q stays in joint_matrix A tiles, S = Q K^T and O += P V run on XMX
// (joint_matrix_mad), the masks and SiLU are applied on the accumulator with
// element coordinates, and joint_matrix_copy turns S into the A operand of P V.
// V tiles load from a row-padded copy of V with plain joint_matrix_load at the
// tile address: a checked load of the first V tile in the second key block
// reads the columns of the last tile instead (DPC++ 2026.1, PVC).
// K is transposed once to [H, D, Z * Npad], every sequence starting at column
// z * Npad (Npad = max_seq_len rounded up to 32), so that K^T tiles load
// row-major from an aligned column; a col_major B load of K gives wrong values
// for bf16 on PVC (DPC++ 2026.1), and so does a K^T column offset that is not
// a multiple of two elements.
// Loads use the checked joint_matrix API from the tensor base with absolute
// coordinates and the extents of the whole tensor: 2D block loads need an
// aligned base and a surface at least 64 bytes wide, so the end of a sequence
// cannot be the surface edge. Keys of the next sequence are masked out and
// rows past the end of a sequence are not stored.
//
// Supported: causal forward, bf16 or fp16, equal head dims for q/k/v that are
// a multiple of 16, num_targets, contextual_seq_len, max_attn_len.
////////////////////////////////////////////////////////////////////////////////

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>
#include <torch/library.h>

#include "fbgemm_utils/tensor_utils.h"
#include "fbgemm_utils/torch_library.h"
#include "fbgemm_utils/utils.h"

namespace fbgemm_xpu {
namespace {

namespace mx = sycl::ext::oneapi::experimental::matrix;
namespace imx = sycl::ext::intel::experimental::matrix;

constexpr int kSubGroupSize = 16;
constexpr int kTileM = 8;    // query rows per sub-group (one XMX tile)
constexpr int kTileN = 16;   // XMX N
constexpr int kTileK = 16;   // XMX K for 16-bit inputs
constexpr int kSubGroups = 8;
constexpr int kRowsPerGroup = kSubGroups * kTileM;

// k_t[h, d, z * npad + n] = k[seq_offsets[z] + n, h, d]
template <typename T>
struct TransposeKeysKernel {
    const T* k;
    T* k_t;
    const int64_t* seq_offsets;
    int64_t stride_k;
    int64_t columns;
    int64_t total;
    int sequences;
    int heads;
    int dim;
    int npad;

    void operator()(sycl::item<1> item) const {
        const int64_t index = item.get_id(0);
        if (index >= total) {
            return;
        }
        const int64_t row = index / (heads * dim);
        const int hd = static_cast<int>(index % (heads * dim));
        int low = 0;
        int high = sequences;
        while (high - low > 1) {
            const int middle = (low + high) / 2;
            if (seq_offsets[middle] <= row) {
                low = middle;
            } else {
                high = middle;
            }
        }
        const int64_t n = row - seq_offsets[low];
        k_t[hd * columns + static_cast<int64_t>(low) * npad + n] = k[row * stride_k + hd];
    }
};

template <typename T, int D>
struct HstuAttentionFwdKernel {
    const T* q;
    const T* k_t;
    const T* v_pad;
    T* out;
    const int64_t* seq_offsets;
    const int64_t* num_targets;
    int64_t stride_q;
    int64_t stride_k_t;
    int64_t stride_o;
    int total_rows;
    int npad;
    int heads;
    float alpha;
    float inv_max_seq_len;
    int contextual_seq_len;
    int max_attn_len;

    [[sycl::reqd_sub_group_size(kSubGroupSize)]] void operator()(
        sycl::nd_item<2> item) const {
        sycl::sub_group sg = item.get_sub_group();
        const int zh = item.get_group(0);
        const int z = zh / heads;
        const int h = zh % heads;
        const int64_t seq_start = seq_offsets[z];
        const int seq_len = static_cast<int>(seq_offsets[z + 1] - seq_start);
        const int m0 = static_cast<int>(item.get_group(1)) * kRowsPerGroup +
                       static_cast<int>(sg.get_group_linear_id()) * kTileM;
        if (m0 >= seq_len) {
            return;
        }
        const int n_targets = num_targets != nullptr ? static_cast<int>(num_targets[z]) : 0;
        const bool has_contextual = contextual_seq_len > 0;
        int max_ids = seq_len;
        if (has_contextual) {
            max_ids = max_ids - contextual_seq_len + 1;
        }
        max_ids -= n_targets;

        int high = sycl::min(seq_len, m0 + kTileM);
        if (has_contextual && m0 < contextual_seq_len) {
            high = seq_len;
        }

        const int row0 = static_cast<int>(seq_start) + m0;
        const int width = heads * D;
        const T* k_t_head = k_t + static_cast<int64_t>(h) * D * stride_k_t;
        auto global = [](const T* ptr) {
            return sycl::address_space_cast<sycl::access::address_space::global_space,
                                            sycl::access::decorated::no>(ptr);
        };

        constexpr int kDTiles = D / kTileK;
        mx::joint_matrix<sycl::sub_group, T, mx::use::a, kTileM, kTileK, mx::layout::row_major>
            qa[kDTiles];
        #pragma unroll
        for (int t = 0; t < kDTiles; ++t) {
            imx::joint_matrix_load_checked(sg, qa[t], global(q), stride_q, total_rows, width, row0,
                                           h * D + t * kTileK);
        }
        mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, kTileM, kTileN>
            acc[kDTiles];
        #pragma unroll
        for (int t = 0; t < kDTiles; ++t) {
            mx::joint_matrix_fill(sg, acc[t], 0.0f);
        }

        for (int n0 = 0; n0 < high; n0 += kTileN) {
            mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, kTileM, kTileN> s;
            mx::joint_matrix_fill(sg, s, 0.0f);
            #pragma unroll
            for (int t = 0; t < kDTiles; ++t) {
                // K^T tile: rows are head dims, columns are keys.
                mx::joint_matrix<sycl::sub_group, T, mx::use::b, kTileK, kTileN,
                                 mx::layout::row_major>
                    kb;
                imx::joint_matrix_load_checked(sg, kb, global(k_t_head), stride_k_t, D,
                                               stride_k_t, t * kTileK, z * npad + n0);
                mx::joint_matrix_mad(sg, s, qa[t], kb, s);
            }
            const float a = alpha;
            const float scale = inv_max_seq_len;
            const int ctx = contextual_seq_len;
            const int attn_len = max_attn_len;
            const bool has_targets = num_targets != nullptr;
            imx::joint_matrix_apply(sg, s, [=](float& x, size_t row, size_t col) {
                const int m = m0 + static_cast<int>(row);
                const int n = n0 + static_cast<int>(col);
                if (n >= seq_len) {
                    x = 0.0f;
                    return;
                }
                bool valid = m == n;
                int mm = m;
                int nn = n;
                if (ctx > 0) {
                    mm = sycl::max(m - ctx + 1, 0);
                    nn = sycl::max(n - ctx + 1, 0);
                }
                if (has_targets) {
                    mm = sycl::min(mm, max_ids);
                    nn = sycl::min(nn, max_ids);
                }
                const int diff = mm - nn;
                valid = valid || diff > 0;
                if (attn_len > 0) {
                    valid = valid && diff <= attn_len;
                }
                if (ctx > 0) {
                    valid = valid || (mm == 0 && nn < max_ids);
                }
                const float qk = x * a;
                x = valid ? qk / (1.0f + sycl::native::exp(-qk)) * scale : 0.0f;
            });
            mx::joint_matrix<sycl::sub_group, T, mx::use::a, kTileM, kTileN,
                             mx::layout::row_major>
                p;
            mx::joint_matrix_copy(sg, s, p);
                        const T* v_tile = v_pad + (seq_start + n0) * width + h * D;
#pragma unroll
            for (int t = 0; t < kDTiles; ++t) {
                mx::joint_matrix<sycl::sub_group, T, mx::use::b, kTileN, kTileN,
                                 mx::layout::row_major>
                    vb;
                mx::joint_matrix_load(sg, vb, global(v_tile + t * kTileN), width);
                mx::joint_matrix_mad(sg, acc[t], p, vb, acc[t]);
            }
        }

        T* out_base = out + seq_start * stride_o + h * D;
        const int64_t stride = stride_o;
        #pragma unroll
        for (int t = 0; t < kDTiles; ++t) {
            imx::joint_matrix_apply(sg, acc[t], [=](float& x, size_t row, size_t col) {
                const int m = m0 + static_cast<int>(row);
                if (m < seq_len) {
                    out_base[m * stride + t * kTileN + static_cast<int>(col)] = static_cast<T>(x);
                }
            });
        }
    }
};

template <typename T, int D>
void launch(const at::Tensor& q, const at::Tensor& k_t, const at::Tensor& v_pad, at::Tensor& out,
            const at::Tensor& seq_offsets, const c10::optional<at::Tensor>& num_targets,
            int64_t max_seq_len, double alpha, int64_t contextual_seq_len, int64_t max_attn_len) {
    const int64_t z = seq_offsets.numel() - 1;
    const int64_t heads = q.size(1);
    const int64_t row_groups = (max_seq_len + kRowsPerGroup - 1) / kRowsPerGroup;
    HstuAttentionFwdKernel<T, D> kernel{
        reinterpret_cast<const T*>(q.data_ptr()),
        reinterpret_cast<const T*>(k_t.data_ptr()),
        reinterpret_cast<const T*>(v_pad.data_ptr()),
        reinterpret_cast<T*>(out.data_ptr()),
        seq_offsets.data_ptr<int64_t>(),
        num_targets.has_value() ? num_targets->data_ptr<int64_t>() : nullptr,
        q.stride(0),
        k_t.stride(1),
        out.stride(0),
        static_cast<int>(q.size(0)),
        static_cast<int>(k_t.size(2) / z),
        static_cast<int>(heads),
        static_cast<float>(alpha),
        1.0f / static_cast<float>(max_seq_len),
        static_cast<int>(contextual_seq_len),
        static_cast<int>(max_attn_len),
    };
    sycl::queue& queue = c10::xpu::getCurrentXPUStream().queue();
    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(z * heads, row_groups * kSubGroups * kSubGroupSize),
                              sycl::range<2>(1, kSubGroups * kSubGroupSize)),
            kernel);
    });
}

void transpose_keys(const at::Tensor& k, at::Tensor& k_t, const at::Tensor& offsets,
                    int64_t npad) {
    const int64_t total = k.size(0) * k.size(1) * k.size(2);
    AT_DISPATCH_SWITCH(
        k.scalar_type(), "hstu_mha_transpose_keys",
        AT_DISPATCH_CASE(at::kBFloat16, [&] {
            using T = sycl::ext::oneapi::bfloat16;
            TransposeKeysKernel<T> kernel{
                reinterpret_cast<const T*>(k.data_ptr()), reinterpret_cast<T*>(k_t.data_ptr()),
                offsets.data_ptr<int64_t>(), k.stride(0), k_t.size(2), total,
                static_cast<int>(offsets.numel() - 1), static_cast<int>(k.size(1)),
                static_cast<int>(k.size(2)), static_cast<int>(npad)};
            c10::xpu::getCurrentXPUStream().queue().parallel_for(sycl::range<1>(total), kernel);
        }) AT_DISPATCH_CASE(at::kHalf, [&] {
            using T = sycl::half;
            TransposeKeysKernel<T> kernel{
                reinterpret_cast<const T*>(k.data_ptr()), reinterpret_cast<T*>(k_t.data_ptr()),
                offsets.data_ptr<int64_t>(), k.stride(0), k_t.size(2), total,
                static_cast<int>(offsets.numel() - 1), static_cast<int>(k.size(1)),
                static_cast<int>(k.size(2)), static_cast<int>(npad)};
            c10::xpu::getCurrentXPUStream().queue().parallel_for(sycl::range<1>(total), kernel);
        }));
}

void check_strides(const at::Tensor& tensor, const char* name, int64_t dim) {
    TORCH_CHECK(tensor.dim() == 3, "hstu_mha: ", name, " must be [L, H, D]");
    TORCH_CHECK(tensor.stride(2) == 1 && tensor.stride(1) == dim,
                "hstu_mha: ", name, " needs contiguous heads and head dims, got strides ",
                tensor.strides());
    TORCH_CHECK(tensor.stride(0) % 8 == 0, "hstu_mha: ", name,
                " row stride must be a multiple of 8 elements, got ", tensor.stride(0));
}

at::Tensor hstu_mha_xpu(
    c10::SymInt max_seq_len, double alpha, const at::Tensor& q, const at::Tensor& k,
    const at::Tensor& v, const c10::optional<at::Tensor>& seq_offsets, bool causal,
    const c10::optional<at::Tensor>& num_targets, const c10::optional<at::Tensor>& attn_scale,
    int64_t max_attn_len, int64_t min_full_attn_seq_len, int64_t contextual_seq_len,
    const c10::optional<at::Tensor>& q_descale, const c10::optional<at::Tensor>& k_descale,
    const c10::optional<at::Tensor>& v_descale, bool sort_by_length, bool deterministic,
    int64_t sm_margin, c10::SymInt max_q_len, const c10::optional<at::Tensor>& seq_offsets_q,
    int64_t num_softmax_heads, bool training,
    const c10::optional<at::Tensor>& max_seq_len_tensor,
    const c10::optional<at::Tensor>& contextual_seq_len_tensor,
    const c10::optional<at::Tensor>& max_attn_len_tensor,
    const c10::optional<at::Tensor>& min_full_attn_seq_len_tensor, int64_t num_groups,
    bool use_bf16_dq_accum) {
    (void)sort_by_length;
    (void)deterministic;
    (void)sm_margin;
    (void)use_bf16_dq_accum;
    TORCH_CHECK(seq_offsets.has_value(), "hstu_mha (XPU): jagged seq_offsets are required");
    TORCH_CHECK(causal, "hstu_mha (XPU): only causal attention is implemented");
    TORCH_CHECK(!training, "hstu_mha (XPU): only the inference forward is implemented");
    TORCH_CHECK(!attn_scale.has_value() && !q_descale.has_value() && !k_descale.has_value() &&
                    !v_descale.has_value(),
                "hstu_mha (XPU): attn_scale and FP8 descales are not implemented");
    TORCH_CHECK(min_full_attn_seq_len == 0 && max_q_len == 0 && !seq_offsets_q.has_value() &&
                    num_softmax_heads == 0 && num_groups == 1,
                "hstu_mha (XPU): only the plain jagged forward is implemented");
    TORCH_CHECK(!max_seq_len_tensor.has_value() && !contextual_seq_len_tensor.has_value() &&
                    !max_attn_len_tensor.has_value() &&
                    !min_full_attn_seq_len_tensor.has_value(),
                "hstu_mha (XPU): tensor-valued lengths are not implemented");
    TENSOR_ON_SYCL_XPU(q);
    TENSORS_ON_SAME_SYCL_XPU_IF_NOT_OPTIONAL(q, k, v, *seq_offsets);
    if (num_targets.has_value()) {
        TENSORS_ON_SAME_SYCL_XPU_IF_NOT_OPTIONAL(q, *num_targets);
    }
    SYCL_DEVICE_GUARD(q);
    const int64_t dim = q.size(2);
    TORCH_CHECK(k.size(2) == dim && v.size(2) == dim,
                "hstu_mha (XPU): q, k and v must have the same head dim");
    TORCH_CHECK(q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type(),
                "hstu_mha (XPU): q, k and v must have the same dtype");
    check_strides(q, "q", dim);
    check_strides(k, "k", dim);
    check_strides(v, "v", dim);
    const int64_t n = max_seq_len.expect_int();
    TORCH_CHECK(n > 0, "hstu_mha (XPU): max_seq_len must be positive");

    const at::Tensor offsets = seq_offsets->to(at::kLong);
    c10::optional<at::Tensor> targets;
    if (num_targets.has_value()) {
        targets = num_targets->to(at::kLong);
    }
    at::Tensor out = at::empty({q.size(0), q.size(1), dim}, q.options());
    if (q.size(0) == 0) {
        return out;
    }
    const int64_t sequences = offsets.numel() - 1;
    const int64_t npad = (n + 31) / 32 * 32;
    at::Tensor k_t = at::zeros({q.size(1), dim, sequences * npad}, k.options());
    transpose_keys(k, k_t, offsets, npad);
    // [L + one key tile, H * D]: the last key tile of the last sequence stays in bounds.
    at::Tensor v_pad = at::zeros({q.size(0) + kTileN, q.size(1) * dim}, v.options());
    v_pad.narrow(0, 0, q.size(0)).copy_(v.reshape({q.size(0), q.size(1) * dim}));

#define HSTU_LAUNCH(SCALAR, DIM)                                                         \
    launch<SCALAR, DIM>(q, k_t, v_pad, out, offsets, targets, n, alpha, contextual_seq_len, \
                        max_attn_len)
    if (q.scalar_type() == at::kBFloat16) {
        using T = sycl::ext::oneapi::bfloat16;
        switch (dim) {
            case 64: HSTU_LAUNCH(T, 64); break;
            case 128: HSTU_LAUNCH(T, 128); break;
            default: TORCH_CHECK(false, "hstu_mha (XPU): head dim must be 64 or 128, got ", dim);
        }
    } else if (q.scalar_type() == at::kHalf) {
        using T = sycl::half;
        switch (dim) {
            case 64: HSTU_LAUNCH(T, 64); break;
            case 128: HSTU_LAUNCH(T, 128); break;
            default: TORCH_CHECK(false, "hstu_mha (XPU): head dim must be 64 or 128, got ", dim);
        }
    } else {
        TORCH_CHECK(false, "hstu_mha (XPU): q must be bfloat16 or float16, got ",
                    q.scalar_type());
    }
#undef HSTU_LAUNCH
    return out;
}

}  // namespace
}  // namespace fbgemm_xpu

// The schema is generative-recommenders' (ops/cpp/hstu_attention/flash_api_cpu.cpp);
// it is defined here only when hstu._C is not loaded.
TORCH_LIBRARY_FRAGMENT(hstu, m) {
    if (!fbgemm_xpu::utils::torch::schemaExists("hstu::hstu_mha")) {
        m.def(
            "hstu_mha(SymInt max_seq_len, float alpha, Tensor q, Tensor k, Tensor v, "
            "Tensor? seq_offsets, bool causal, Tensor? num_targets, Tensor? attn_scale, "
            "int max_attn_len, int min_full_attn_seq_len, int contextual_seq_len, "
            "Tensor? q_descale, Tensor? k_descale, Tensor? v_descale, bool sort_by_length, "
            "bool deterministic, int sm_margin=0, SymInt max_q_len=0, "
            "Tensor? seq_offsets_q=None, int num_softmax_heads=0, bool training=True, "
            "Tensor? max_seq_len_tensor=None, Tensor? contextual_seq_len_tensor=None, "
            "Tensor? max_attn_len_tensor=None, Tensor? min_full_attn_seq_len_tensor=None, "
            "int num_groups=1, bool use_bf16_dq_accum=False) -> Tensor");
    }
}

TORCH_LIBRARY_IMPL(hstu, XPU, m) {
    m.impl("hstu_mha", TORCH_FN(fbgemm_xpu::hstu_mha_xpu));
}
