# Copyright (c) 2026 Intel Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""hstu::hstu_mha inference forward on XPU against a PyTorch reference.

The reference follows _hstu_attn_fwd_one_block in generative-recommenders
(ops/triton/triton_hstu_attention.py): SiLU attention scaled by 1/max_seq_len
with the causal, target, contextual and max_attn_len masks, on jagged
sequences. Like the kernel, it rounds the attention weights to the input dtype
before multiplying with v.
"""

import fbgemm_xpu  # noqa: F401  - registers hstu::hstu_mha for XPU
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.xpu.is_available(), reason="requires an XPU device")


def _reference(q, k, v, seq_offsets, num_targets, max_seq_len, alpha, contextual_seq_len,
               max_attn_len):
    out = torch.zeros(v.shape, dtype=torch.float32)
    offsets = seq_offsets.tolist()
    for z in range(len(offsets) - 1):
        start, end = offsets[z], offsets[z + 1]
        length = end - start
        if length == 0:
            continue
        qz, kz, vz = (t[start:end].float() for t in (q, k, v))
        qk = torch.einsum("mhd,nhd->hmn", qz, kz) * alpha
        m = torch.arange(length)[:, None]
        n = torch.arange(length)[None, :]
        valid = m == n
        mm, nn, max_ids = m, n, length
        if contextual_seq_len > 0:
            mm = (m - contextual_seq_len + 1).clamp(min=0)
            nn = (n - contextual_seq_len + 1).clamp(min=0)
            max_ids = max_ids - contextual_seq_len + 1
        if num_targets is not None:
            max_ids = max_ids - int(num_targets[z])
            mm = torch.minimum(mm, torch.tensor(max_ids))
            nn = torch.minimum(nn, torch.tensor(max_ids))
        diff = mm - nn
        valid = valid | (diff > 0)
        if max_attn_len > 0:
            valid = valid & (diff <= max_attn_len)
        if contextual_seq_len > 0:
            valid = valid | ((mm == 0) & (nn < max_ids))
        weights = torch.nn.functional.silu(qk) / max_seq_len * valid
        weights = weights.to(q.dtype).float()
        out[start:end] = torch.einsum("hmn,nhd->mhd", weights, vz)
    return out


def _hstu_mha(q, k, v, seq_offsets, num_targets, max_seq_len, alpha, contextual_seq_len,
              max_attn_len):
    return torch.ops.hstu.hstu_mha(
        max_seq_len, alpha, q, k, v, seq_offsets, True, num_targets, None, max_attn_len, 0,
        contextual_seq_len, None, None, None, False, False, 0, 0, None, 0, False,
        None, None, None, None, 1,
    )


def _inputs(lengths, heads, dim, dtype, device, seed=0, targets=None):
    """q/k/v as views into one uvqk tensor, as GR's hstu_preprocess_and_attention makes them."""
    generator = torch.Generator().manual_seed(seed)
    total = sum(lengths)
    uvqk = torch.randn((total, 4 * heads * dim), generator=generator).to(dtype)
    seq_offsets = torch.tensor([0] + lengths, dtype=torch.int64).cumsum(0)
    num_targets = None if targets is None else torch.tensor(targets, dtype=torch.int64)
    _, v, q, k = (part.view(total, heads, dim) for part in uvqk.split(heads * dim, dim=1))
    cpu = (q, k, v, seq_offsets, num_targets)
    uvqk_device = uvqk.to(device)
    _, v_d, q_d, k_d = (part.view(total, heads, dim) for part in uvqk_device.split(heads * dim, dim=1))
    on_device = (q_d, k_d, v_d, seq_offsets.to(device),
                 None if num_targets is None else num_targets.to(device))
    return cpu, on_device


def _check(lengths, heads=4, dim=128, dtype=torch.bfloat16, targets=None, contextual_seq_len=0,
           max_attn_len=0, device="xpu:0", seed=0):
    max_seq_len = max(max(lengths), 1)
    alpha = dim**-0.5
    cpu, on_device = _inputs(lengths, heads, dim, dtype, device, seed, targets)
    expected = _reference(*cpu, max_seq_len, alpha, contextual_seq_len, max_attn_len)
    actual = _hstu_mha(*on_device, max_seq_len, alpha, contextual_seq_len, max_attn_len)
    assert actual.device == torch.device(device)  # nosec B101
    assert actual.dtype == dtype  # nosec B101
    scale = max(expected.abs().max().item(), 1e-6)
    torch.testing.assert_close(actual.float().cpu(), expected, rtol=0, atol=2e-2 * scale)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("dim", [64, 128])
def test_single_sequence(dtype, dim):
    _check([200], dim=dim, dtype=dtype)


@pytest.mark.parametrize(
    "lengths", [[1], [7], [16], [17], [64], [65], [0, 33], [130, 0, 5, 64, 1]]
)
def test_jagged_lengths(lengths):
    _check(lengths, seed=len(lengths))


@pytest.mark.parametrize("contextual_seq_len", [0, 1, 3])
@pytest.mark.parametrize("with_targets", [False, True])
def test_masks(contextual_seq_len, with_targets):
    lengths = [96, 150, 40]
    targets = [8, 20, 3] if with_targets else None
    _check(lengths, targets=targets, contextual_seq_len=contextual_seq_len, seed=7)


def test_max_attn_len():
    _check([120, 70], targets=[10, 5], max_attn_len=17, seed=3)


def test_dlrm_v3_shape():
    # One request of the DLRM-v3 reduced profile: 4 heads of 128, causal,
    # contextual 1, 256 candidates.
    _check([2048], targets=[256], contextual_seq_len=1, seed=11)


@pytest.mark.skipif(
    torch.xpu.is_available() and torch.xpu.device_count() < 2,
    reason="requires two XPU devices",
)
def test_non_current_device():
    torch.xpu.set_device(0)
    _check([130, 64], targets=[4, 2], contextual_seq_len=1, device="xpu:1", seed=5)
    assert torch.xpu.current_device() == 0  # nosec B101


def test_rejects_training_and_non_causal():
    _, (q, k, v, seq_offsets, _) = _inputs([32], 2, 64, torch.bfloat16, "xpu:0")
    args = [32, 0.125, q, k, v, seq_offsets]
    with pytest.raises(RuntimeError, match="only causal"):
        torch.ops.hstu.hstu_mha(*args, False, None, None, 0, 0, 0, None, None, None, False, False)
    with pytest.raises(RuntimeError, match="inference forward"):
        torch.ops.hstu.hstu_mha(*args, True, None, None, 0, 0, 0, None, None, None, False, False,
                                training=True)
