# Copyright (c) 2026 Intel Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Time hstu::hstu_mha on XPU with DLRM-v3 shapes.

    python -m fbgemm_xpu.bench.hstu_attention [--uih 6588 ...] [--targets 2048] [--batch 10]

Every sequence has its user history followed by its targets, causal with target masking and
no contextual rows, 4 heads of 128 in bf16, as the HSTU layers of DLRM-v3 call it. ``full``
computes every row; ``targets`` only the target rows (seq_offsets_q), as the last layer does
with DLRM_LAST_LAYER_TARGETS_ONLY=1. TFLOPS count the query-key pairs the masks keep, two
matrix products of 2 * head dim FLOPs each per pair and head.
"""

import argparse

import torch

import fbgemm_xpu  # noqa: F401


def _call(max_seq_len, alpha, q, k, v, seq_offsets, num_targets, contextual, max_q_len=0,
          seq_offsets_q=None):
    return torch.ops.hstu.hstu_mha(
        max_seq_len, alpha, q, k, v, seq_offsets, True, num_targets, None, 0, 0, contextual,
        None, None, None, False, False, 0, max_q_len, seq_offsets_q, 0, False, None, None,
        None, None, 1,
    )


def _pairs(uih, targets):
    """Query-key pairs the causal and target masks keep, for all rows and for the targets.

    History row m attends to keys 0..m; a target row to the history and to itself.
    """
    history = sum(n * (n + 1) // 2 for n in uih)
    target_rows = sum(t * (n + 1) for n, t in zip(uih, targets))
    return history + target_rows, target_rows


def _read_energy(path):
    with open(path) as f:
        return int(f.read())


def _time(fn, iters, energy_path=None):
    """Median ms of one call; with energy_path (a hwmon energy1_input in microjoules), also
    the mean millijoules per call over the timed calls."""
    start = torch.xpu.Event(enable_timing=True)
    end = torch.xpu.Event(enable_timing=True)
    fn()
    torch.xpu.synchronize()
    times = []
    energy = _read_energy(energy_path) if energy_path else 0
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    mj = (_read_energy(energy_path) - energy) / 1000 / iters if energy_path else None
    times.sort()
    return times[len(times) // 2], mj


def _steady(fn, seconds, energy_path=None):
    """Calls per second over the second half of `seconds` of back-to-back calls, after the
    tile has settled at its thermal limit; with energy_path also millijoules per call."""
    import time

    fn()
    torch.xpu.synchronize()
    start = time.time()
    half = start + seconds / 2
    calls = 0
    energy = None
    t_half = None
    while True:
        for _ in range(4):
            fn()
        torch.xpu.synchronize()
        now = time.time()
        if t_half is None and now >= half:
            t_half = now
            energy = _read_energy(energy_path) if energy_path else 0
            calls = 0
            continue
        calls += 4
        if now >= start + seconds:
            break
    elapsed = time.time() - t_half
    mj = (_read_energy(energy_path) - energy) / 1000 / calls if energy_path else None
    return calls / elapsed, mj


def _report(name, ms, mj, pairs, flops_per_pair):
    line = f"{name:8s} {ms:8.3f} ms  {pairs * flops_per_pair / ms / 1e9:6.1f} TFLOPS"
    if mj is not None:
        line += f"  {mj:8.1f} mJ/call"
    print(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--uih", type=int, nargs="+", default=[6588],
                        help="history lengths; repeated to --batch sequences")
    parser.add_argument("--targets", type=int, default=2048)
    parser.add_argument("--batch", type=int, default=10)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--contextual", type=int, default=0)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--device", default="xpu:0")
    parser.add_argument("--mode", choices=["full", "targets", "both"], default="both")
    parser.add_argument("--energy", help="hwmon energy1_input of the tile, for mJ per call")
    parser.add_argument("--duration", type=float, default=0,
                        help="seconds of back-to-back full calls; report the steady state")
    args = parser.parse_args()

    torch.xpu.set_device(args.device)
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    uih = [args.uih[i % len(args.uih)] for i in range(args.batch)]
    targets = [args.targets] * args.batch
    lengths = [a + b for a, b in zip(uih, targets)]
    total = sum(lengths)
    max_seq_len = max(lengths)
    alpha = args.dim**-0.5
    generator = torch.Generator().manual_seed(0)
    uvqk = (torch.randn((total, 4 * args.heads * args.dim), generator=generator) * 0.5).to(
        dtype).to(device)
    _, v, q, k = (part.view(total, args.heads, args.dim)
                  for part in uvqk.split(args.heads * args.dim, dim=1))
    seq_offsets = torch.tensor([0] + lengths, dtype=torch.int64).cumsum(0).to(device)
    num_targets = torch.tensor(targets, dtype=torch.int64, device=device)
    full_pairs, target_pairs = _pairs(uih, targets)
    flops_per_pair = 4 * args.dim * args.heads
    print(f"batch {args.batch}, lengths {min(lengths)}-{max(lengths)} ({total} rows), "
          f"targets {args.targets}, heads {args.heads}x{args.dim} {args.dtype}")

    if args.duration > 0:
        rate, mj = _steady(lambda: _call(max_seq_len, alpha, q, k, v, seq_offsets, num_targets,
                                         args.contextual), args.duration, args.energy)
        line = f"steady   {rate:8.2f} calls/s  {1000 / rate:8.3f} ms"
        if mj is not None:
            line += f"  {mj:8.1f} mJ/call"
        print(line)
        return
    if args.mode in ("full", "both"):
        ms, mj = _time(lambda: _call(max_seq_len, alpha, q, k, v, seq_offsets, num_targets,
                                     args.contextual), args.iters, args.energy)
        _report("full", ms, mj, full_pairs, flops_per_pair)
    if args.mode in ("targets", "both"):
        offsets = seq_offsets.tolist()
        rows = torch.cat([torch.arange(offsets[z + 1] - n, offsets[z + 1])
                          for z, n in enumerate(targets)]).to(device)
        q_t = q.index_select(0, rows)
        seq_offsets_q = torch.tensor([0] + targets, dtype=torch.int64).cumsum(0).to(device)
        ms, mj = _time(lambda: _call(max_seq_len, alpha, q_t, k, v, seq_offsets, num_targets,
                                     args.contextual, max(targets), seq_offsets_q), args.iters,
                       args.energy)
        _report("targets", ms, mj, target_pairs, flops_per_pair)


if __name__ == "__main__":
    main()
