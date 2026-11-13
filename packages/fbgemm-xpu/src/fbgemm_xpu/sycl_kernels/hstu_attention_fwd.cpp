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
// Each sub-group of 16 work-items owns MT query tiles of 8 rows of one (sequence,
// head): two for bf16, so that every K^T and V tile it loads feeds two products,
// with 256 registers per thread; one for fp16. Q stays in joint_matrix A tiles,
// S = Q K^T and O += P V run on XMX (joint_matrix_mad), the masks and SiLU are
// applied on the accumulator with element coordinates, and joint_matrix_copy
// turns S into the A operand of P V. For fp16, S goes through a tile in local
// memory instead: joint_matrix_copy from the float accumulator to an fp16 A tile
// gives zeros on BMG (head dim 64, DPC++ 2026.1), while S itself and P V are right.
// The K^T and V tiles of the next key block are prefetched to L1, and target rows
// skip the key blocks that hold only other targets, which their mask zeroes. SiLU
// runs on every element and the mask, which needs element coordinates, only on the
// tiles that cross the diagonal, the end of the sequence or the targets.
// V tiles load from a row-padded copy of V with plain joint_matrix_load at the
// tile address: a checked load of the first V tile in the second key block
// reads the columns of the last tile instead (DPC++ 2026.1, PVC).
// K is transposed once to [H, D, Z * Npad] (32 x 32 tiles through local memory,
// zeros past the end of every sequence), every sequence starting at column
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
// With seq_offsets_q, q holds only the last q_len rows of every sequence (for
// example its targets): query row i of sequence z is q[seq_offsets_q[z] + i],
// at position seq_len - q_len + i of the keys, and out has the rows of q. The
// masks are those of the full sequence at that position, as in
// pytorch_target_rows_hstu_mha of generative-recommenders; q_len must not
// exceed seq_len.
//
// Supported: causal forward, bf16 or fp16, equal head dims for q/k/v that are
// a multiple of 16, num_targets, contextual_seq_len, max_attn_len,
// seq_offsets_q with max_q_len.
////////////////////////////////////////////////////////////////////////////////

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/sycl.hpp>
#include <torch/library.h>

#include <type_traits>

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
constexpr int kTransposeTile = 32;
constexpr int kTransposeRows = 8;  // work-items per tile column; each moves 4 elements

// k_t[hd, z * npad + n] = k[seq_offsets[z] + n, hd] for n < the length of sequence z, else 0:
// every element of k_t is written, so it needs no zero fill. A work-group moves a 32 x 32
// tile through local memory, reading rows of k and writing rows of k_t contiguously.
template <typename T>
struct TransposeKeysKernel {
    const T* k;
    T* k_t;
    const int64_t* seq_offsets;
    int64_t stride_k;
    int64_t columns;  // sequences * npad
    int width;        // heads * dim
    int npad;
    sycl::local_accessor<T, 2> tile;

    void operator()(sycl::nd_item<2> item) const {
        const int lx = static_cast<int>(item.get_local_id(1));
        const int ly = static_cast<int>(item.get_local_id(0));
        const int64_t col0 = static_cast<int64_t>(item.get_group(1)) * kTransposeTile;
        const int hd0 = static_cast<int>(item.get_group(0)) * kTransposeTile;
        // npad is a multiple of 32, so a tile of columns lies in one sequence.
        const int z = static_cast<int>(col0 / npad);
        const int n0 = static_cast<int>(col0 - static_cast<int64_t>(z) * npad);
        const int64_t start = seq_offsets[z];
        const int len = static_cast<int>(seq_offsets[z + 1] - start);
        for (int r = ly; r < kTransposeTile; r += kTransposeRows) {
            const int n = n0 + r;
            const int hd = hd0 + lx;
            tile[r][lx] = n < len && hd < width ? k[(start + n) * stride_k + hd] : T(0);
        }
        sycl::group_barrier(item.get_group());
        for (int r = ly; r < kTransposeTile; r += kTransposeRows) {
            const int hd = hd0 + r;
            const int64_t col = col0 + lx;
            if (hd < width && col < columns) {
                k_t[static_cast<int64_t>(hd) * columns + col] = tile[lx][r];
            }
        }
    }
};

// MT: query tiles of kTileM rows per sub-group; every K^T and V tile feeds MT products.
template <typename T, int D, int MT>
struct HstuAttentionFwdKernel {
    static constexpr int kRowsPerSubGroup = kTileM * MT;
    static constexpr int kRowsPerWorkGroup = kSubGroups * kRowsPerSubGroup;

    const T* q;
    const T* k_t;
    const T* v_pad;
    T* out;
    const int64_t* seq_offsets;
    const int64_t* seq_offsets_q;
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
        constexpr bool kStageP = std::is_same_v<T, sycl::half>;
        static_assert(MT == 1 || MT == 2, "one or two query tiles per sub-group");
        static_assert(!kStageP || MT == 1, "the fp16 path stages one S tile");
        // One S tile per sub-group, for the fp16 path; allocated before any return.
        auto stage = sycl::ext::oneapi::group_local_memory_for_overwrite<
            T[kStageP ? kSubGroups * kTileM * kTileN : 1]>(item.get_group());
        sycl::sub_group sg = item.get_sub_group();
        const int sg_id = static_cast<int>(sg.get_group_linear_id());
        const int zh = item.get_group(0);
        const int z = zh / heads;
        const int h = zh % heads;
        const int64_t seq_start = seq_offsets[z];
        const int seq_len = static_cast<int>(seq_offsets[z + 1] - seq_start);
        int64_t q_start = seq_start;
        int q_len = seq_len;
        if (seq_offsets_q != nullptr) {
            q_start = seq_offsets_q[z];
            q_len = static_cast<int>(seq_offsets_q[z + 1] - q_start);
        }
        // i0: first query row of this sub-group in q; m0: its position in the sequence.
        const int i0 = static_cast<int>(item.get_group(1)) * kRowsPerWorkGroup +
                       sg_id * kRowsPerSubGroup;
        if (i0 >= q_len) {
            return;
        }
        const int m0 = seq_len - q_len + i0;
        const int n_targets = num_targets != nullptr ? static_cast<int>(num_targets[z]) : 0;
        const bool has_contextual = contextual_seq_len > 0;
        int max_ids = seq_len;
        if (has_contextual) {
            max_ids = max_ids - contextual_seq_len + 1;
        }
        max_ids -= n_targets;

        int high = sycl::min(seq_len, m0 + kRowsPerSubGroup);
        if (has_contextual && m0 < contextual_seq_len) {
            high = seq_len;
        }
        // Key blocks [0, main_end) and [tail_begin, high): rows that are all targets see
        // the history and themselves, so the blocks of the other targets are skipped.
        int main_end = high;
        int tail_begin = high;
        if (!has_contextual && max_attn_len == 0 && m0 >= max_ids) {
            const int history_end = (max_ids + kTileN - 1) / kTileN * kTileN;
            if (history_end < high) {
                main_end = history_end;
                tail_begin = sycl::max(history_end, m0 / kTileN * kTileN);
            }
        }

        const int row0 = static_cast<int>(q_start) + i0;
        const int width = heads * D;
        const T* k_t_head = k_t + static_cast<int64_t>(h) * D * stride_k_t;
        auto global = [](const T* ptr) {
            return sycl::address_space_cast<sycl::access::address_space::global_space,
                                            sycl::access::decorated::no>(ptr);
        };

        constexpr int kDTiles = D / kTileK;
        using ATile =
            mx::joint_matrix<sycl::sub_group, T, mx::use::a, kTileM, kTileK, mx::layout::row_major>;
        using CTile = mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, kTileM, kTileN>;
        using BTile =
            mx::joint_matrix<sycl::sub_group, T, mx::use::b, kTileK, kTileN, mx::layout::row_major>;
        // The two query tiles are separate variables: with arrays of joint_matrix tiles
        // indexed by query tile, the output is zero (DPC++ 2026.1, PVC).
        ATile qa0[kDTiles];
        ATile qa1[MT == 2 ? kDTiles : 1];
        CTile acc0[kDTiles];
        CTile acc1[MT == 2 ? kDTiles : 1];
        #pragma unroll
        for (int t = 0; t < kDTiles; ++t) {
            imx::joint_matrix_load_checked(sg, qa0[t], global(q), stride_q, total_rows, width,
                                           row0, h * D + t * kTileK);
            mx::joint_matrix_fill(sg, acc0[t], 0.0f);
            if constexpr (MT == 2) {
                imx::joint_matrix_load_checked(sg, qa1[t], global(q), stride_q, total_rows, width,
                                               row0 + kTileM, h * D + t * kTileK);
                mx::joint_matrix_fill(sg, acc1[t], 0.0f);
            }
        }

        const float a = alpha;
        const float scale = inv_max_seq_len;
        const int ctx = contextual_seq_len;
        const int attn_len = max_attn_len;
        const bool has_targets = num_targets != nullptr;

        // One loop over both key ranges: joint_matrix objects stay out of lambdas.
        const int main_blocks = (main_end + kTileN - 1) / kTileN;
        const int tail_blocks =
            high > tail_begin ? (high - tail_begin + kTileN - 1) / kTileN : 0;
        for (int b = 0; b < main_blocks + tail_blocks; ++b) {
            const int n0 =
                b < main_blocks ? b * kTileN : tail_begin + (b - main_blocks) * kTileN;
            const int next = n0 + kTileN;
            if (next < high) {
                namespace syclex = sycl::ext::oneapi::experimental;
                #pragma unroll
                for (int t = 0; t < kDTiles; ++t) {
                    mx::joint_matrix_prefetch<kTileK, kTileN>(
                        sg, k_t_head + static_cast<int64_t>(t * kTileK) * stride_k_t +
                                static_cast<int64_t>(z) * npad + next,
                        stride_k_t, mx::layout::row_major,
                        syclex::properties{syclex::prefetch_hint_L1});
                    mx::joint_matrix_prefetch<kTileN, kTileN>(
                        sg, v_pad + (seq_start + next) * width + h * D + t * kTileN, width,
                        mx::layout::row_major, syclex::properties{syclex::prefetch_hint_L1});
                }
            }
            CTile s0;
            CTile s1;
            mx::joint_matrix_fill(sg, s0, 0.0f);
            if constexpr (MT == 2) {
                mx::joint_matrix_fill(sg, s1, 0.0f);
            }
            #pragma unroll
            for (int t = 0; t < kDTiles; ++t) {
                // K^T tile: rows are head dims, columns are keys.
                BTile kb;
                imx::joint_matrix_load_checked(sg, kb, global(k_t_head), stride_k_t, D,
                                               stride_k_t, t * kTileK, z * npad + n0);
                mx::joint_matrix_mad(sg, s0, qa0[t], kb, s0);
                if constexpr (MT == 2) {
                    mx::joint_matrix_mad(sg, s1, qa1[t], kb, s1);
                }
            }
            // SiLU on every element without coordinates, then the mask on the tiles that
            // need one: a tile of the history below the diagonal is valid everywhere. With
            // the SiLU inside the coordinate apply, the kernel takes 1.6 times as long
            // (DPC++ 2026.1, PVC).
            auto silu = [=](float& x) {
                const float qk = x * a;
                x = qk / (1.0f + sycl::native::exp(-qk)) * scale;
            };
            auto mask = [=](int mb) {
                return [=](float& x, size_t row, size_t col) {
                    const int m = mb + static_cast<int>(row);
                    const int n = n0 + static_cast<int>(col);
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
                    x = valid && n < seq_len ? x : 0.0f;
                };
            };
            const bool plain = ctx == 0 && attn_len == 0;
            mx::joint_matrix_apply(sg, s0, silu);
            if (!(plain && n0 + kTileN <= m0 && n0 + kTileN <= max_ids)) {
                imx::joint_matrix_apply(sg, s0, mask(m0));
            }
            if constexpr (MT == 2) {
                mx::joint_matrix_apply(sg, s1, silu);
                if (!(plain && n0 + kTileN <= m0 + kTileM && n0 + kTileN <= max_ids)) {
                    imx::joint_matrix_apply(sg, s1, mask(m0 + kTileM));
                }
            }
            mx::joint_matrix<sycl::sub_group, T, mx::use::a, kTileM, kTileN,
                             mx::layout::row_major>
                p0;
            mx::joint_matrix<sycl::sub_group, T, mx::use::a, kTileM, kTileN,
                             mx::layout::row_major>
                p1;
            if constexpr (kStageP) {
                T* tile = &(*stage)[sg_id * kTileM * kTileN];
                imx::joint_matrix_apply(sg, s0, [=](float& x, size_t row, size_t col) {
                    tile[row * kTileN + col] = static_cast<T>(x);
                });
                sycl::group_barrier(sg);
                mx::joint_matrix_load(
                    sg, p0,
                    sycl::address_space_cast<sycl::access::address_space::local_space,
                                             sycl::access::decorated::no>(tile),
                    kTileN);
                sycl::group_barrier(sg);
            } else {
                mx::joint_matrix_copy(sg, s0, p0);
                if constexpr (MT == 2) {
                    mx::joint_matrix_copy(sg, s1, p1);
                }
            }
            const T* v_tile = v_pad + (seq_start + n0) * width + h * D;
            #pragma unroll
            for (int t = 0; t < kDTiles; ++t) {
                mx::joint_matrix<sycl::sub_group, T, mx::use::b, kTileN, kTileN,
                                 mx::layout::row_major>
                    vb;
                mx::joint_matrix_load(sg, vb, global(v_tile + t * kTileN), width);
                mx::joint_matrix_mad(sg, acc0[t], p0, vb, acc0[t]);
                if constexpr (MT == 2) {
                    mx::joint_matrix_mad(sg, acc1[t], p1, vb, acc1[t]);
                }
            }
        }

        T* out_base = out + q_start * stride_o + h * D;
        const int64_t stride = stride_o;
        #pragma unroll
        for (int t = 0; t < kDTiles; ++t) {
            imx::joint_matrix_apply(sg, acc0[t], [=](float& x, size_t row, size_t col) {
                const int i = i0 + static_cast<int>(row);
                if (i < q_len) {
                    out_base[i * stride + t * kTileN + static_cast<int>(col)] = static_cast<T>(x);
                }
            });
            if constexpr (MT == 2) {
                imx::joint_matrix_apply(sg, acc1[t], [=](float& x, size_t row, size_t col) {
                    const int i = i0 + kTileM + static_cast<int>(row);
                    if (i < q_len) {
                        out_base[i * stride + t * kTileN + static_cast<int>(col)] =
                            static_cast<T>(x);
                    }
                });
            }
        }
    }
};

// A kernel with 256 general registers per thread (half the threads per vector engine).
template <typename Kernel>
struct LargeGrf : Kernel {
    auto get(sycl::ext::oneapi::experimental::properties_tag) const {
        return sycl::ext::oneapi::experimental::properties{
            sycl::ext::intel::experimental::grf_size<256>};
    }
};

template <typename T, int D>
void launch(const at::Tensor& q, const at::Tensor& k_t, const at::Tensor& v_pad, at::Tensor& out,
            const at::Tensor& seq_offsets, const c10::optional<at::Tensor>& seq_offsets_q,
            const c10::optional<at::Tensor>& num_targets, int64_t max_seq_len,
            int64_t max_rows, double alpha, int64_t contextual_seq_len, int64_t max_attn_len) {
    // Two query tiles per sub-group need the 256 registers of a large-GRF thread.
    constexpr int MT = std::is_same_v<T, sycl::half> ? 1 : 2;
    using Kernel = HstuAttentionFwdKernel<T, D, MT>;
    const int64_t z = seq_offsets.numel() - 1;
    const int64_t heads = q.size(1);
    const int64_t row_groups =
        (max_rows + Kernel::kRowsPerWorkGroup - 1) / Kernel::kRowsPerWorkGroup;
    Kernel kernel{
        reinterpret_cast<const T*>(q.data_ptr()),
        reinterpret_cast<const T*>(k_t.data_ptr()),
        reinterpret_cast<const T*>(v_pad.data_ptr()),
        reinterpret_cast<T*>(out.data_ptr()),
        seq_offsets.data_ptr<int64_t>(),
        seq_offsets_q.has_value() ? seq_offsets_q->data_ptr<int64_t>() : nullptr,
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
    const sycl::nd_range<2> range(
        sycl::range<2>(z * heads, row_groups * kSubGroups * kSubGroupSize),
        sycl::range<2>(1, kSubGroups * kSubGroupSize));
    sycl::queue& queue = c10::xpu::getCurrentXPUStream().queue();
    queue.submit([&](sycl::handler& cgh) {
        if constexpr (MT == 2) {
            cgh.parallel_for(range, LargeGrf<Kernel>{kernel});
        } else {
            cgh.parallel_for(range, kernel);
        }
    });
}

template <typename T>
void launch_transpose_keys(const at::Tensor& k, at::Tensor& k_t, const at::Tensor& offsets,
                           int64_t npad) {
    const int width = static_cast<int>(k.size(1) * k.size(2));
    const int64_t columns = k_t.size(2);
    const sycl::nd_range<2> range(
        sycl::range<2>((width + kTransposeTile - 1) / kTransposeTile * kTransposeRows,
                       (columns + kTransposeTile - 1) / kTransposeTile * kTransposeTile),
        sycl::range<2>(kTransposeRows, kTransposeTile));
    c10::xpu::getCurrentXPUStream().queue().submit([&](sycl::handler& cgh) {
        TransposeKeysKernel<T> kernel{
            reinterpret_cast<const T*>(k.data_ptr()),
            reinterpret_cast<T*>(k_t.data_ptr()),
            offsets.data_ptr<int64_t>(),
            k.stride(0),
            columns,
            width,
            static_cast<int>(npad),
            sycl::local_accessor<T, 2>(sycl::range<2>(kTransposeTile, kTransposeTile + 1), cgh)};
        cgh.parallel_for(range, kernel);
    });
}

void transpose_keys(const at::Tensor& k, at::Tensor& k_t, const at::Tensor& offsets,
                    int64_t npad) {
    AT_DISPATCH_SWITCH(
        k.scalar_type(), "hstu_mha_transpose_keys",
        AT_DISPATCH_CASE(at::kBFloat16,
                         [&] {
                             launch_transpose_keys<sycl::ext::oneapi::bfloat16>(k, k_t, offsets,
                                                                                npad);
                         })
            AT_DISPATCH_CASE(at::kHalf,
                             [&] { launch_transpose_keys<sycl::half>(k, k_t, offsets, npad); }));
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
    TORCH_CHECK(min_full_attn_seq_len == 0 && num_softmax_heads == 0 && num_groups == 1,
                "hstu_mha (XPU): only the plain jagged forward is implemented");
    TORCH_CHECK(seq_offsets_q.has_value() || max_q_len == 0,
                "hstu_mha (XPU): max_q_len needs seq_offsets_q");
    TORCH_CHECK(!max_seq_len_tensor.has_value() && !contextual_seq_len_tensor.has_value() &&
                    !max_attn_len_tensor.has_value() &&
                    !min_full_attn_seq_len_tensor.has_value(),
                "hstu_mha (XPU): tensor-valued lengths are not implemented");
    TENSOR_ON_SYCL_XPU(q);
    TENSORS_ON_SAME_SYCL_XPU_IF_NOT_OPTIONAL(q, k, v, *seq_offsets);
    if (num_targets.has_value()) {
        TENSORS_ON_SAME_SYCL_XPU_IF_NOT_OPTIONAL(q, *num_targets);
    }
    if (seq_offsets_q.has_value()) {
        TENSORS_ON_SAME_SYCL_XPU_IF_NOT_OPTIONAL(q, *seq_offsets_q);
        TORCH_CHECK(seq_offsets_q->numel() == seq_offsets->numel(),
                    "hstu_mha (XPU): seq_offsets_q and seq_offsets must have the same length");
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
    c10::optional<at::Tensor> offsets_q;
    int64_t max_rows = n;
    if (seq_offsets_q.has_value()) {
        offsets_q = seq_offsets_q->to(at::kLong);
        max_rows = max_q_len.expect_int();
    }
    c10::optional<at::Tensor> targets;
    if (num_targets.has_value()) {
        targets = num_targets->to(at::kLong);
    }
    at::Tensor out = at::empty({q.size(0), q.size(1), dim}, q.options());
    if (q.size(0) == 0 || max_rows <= 0) {
        return out;
    }
    const int64_t sequences = offsets.numel() - 1;
    const int64_t npad = (n + 31) / 32 * 32;
    at::Tensor k_t = at::empty({q.size(1), dim, sequences * npad}, k.options());
    transpose_keys(k, k_t, offsets, npad);
    // [L + one key tile, H * D]: the last key tile of the last sequence stays in bounds.
    const int64_t kv_rows = v.size(0);
    at::Tensor v_pad = at::empty({kv_rows + kTileN, v.size(1) * dim}, v.options());
    v_pad.narrow(0, 0, kv_rows).copy_(v.reshape({kv_rows, v.size(1) * dim}));
    v_pad.narrow(0, kv_rows, kTileN).zero_();

#define HSTU_LAUNCH(SCALAR, DIM)                                                            \
    launch<SCALAR, DIM>(q, k_t, v_pad, out, offsets, offsets_q, targets, n, max_rows, alpha, \
                        contextual_seq_len, max_attn_len)
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
