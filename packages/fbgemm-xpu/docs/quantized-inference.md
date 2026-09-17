# PTXPULIB-151/150 Quantized Inference Track

## Status and ownership

2026-09-17: supported-version wheel build and CPU/XPU numerical validation PASS
on DUT1013PVC (`gta@10.211.176.215`), using only `xpu:0`. This is a small-table
correctness milestone, not a performance result or full-model validation.

The kernel track uses the isolated worktree below; the GR baseline is maintained
in its separate inference worktree. No TorchRec or FBGEMM source changes or
site-packages patches were needed for the kernel track. The built
wheel was installed into the dedicated remote smoke venv, leaving the existing
Torch 2.14 environment and host drivers untouched.

- Worktree: `/home/mkrzemie/repos/torchrec/torchlib-xpu-mkrze/mlperf-dlrmv3-inference-kernels`
- Branch: `ptxpulib-150-quantized-inference-lookup`
- PR127 base: `f55d8f5b75077be906053573ae0c5091e33f994b`
- Bounds integration commit: `4b87d3e`
- `intel`: `https://github.com/intel/torchlib-xpu.git`
- `origin`: `https://github.com/mkrze/torchlib-xpu.git`

## Pinned sources

| Source | Exact ref / SHA |
| --- | --- |
| Fetched `intel/main` | `e7fd1e2f362df085fbecbd6da94071b73befb1ad` |
| Fetched `intel` PR127, `refs/pull/127/head` | `f55d8f5b75077be906053573ae0c5091e33f994b` |
| Fetched `origin/ptxpulib-151-bounds-check-indices` | `cbb733ce24a23daa5ff74bf647cb145f306c243d` |
| Replayed 151 implementation | `2d42203db4974439f9a4617e0486d459fad1df21` |
| Read-only FBGEMM `v1.8.0` tag object | `f349896c8439444889a49e695807503e8ac68bbf` |
| Read-only FBGEMM `v1.8.0^{}` source commit | `99a76f4ed785d2d00579eae5fd45440a1242d657` |
| Read-only TorchRec frontend API reference | `9cbd52b079323914185e02f7913a10e7d1349f25` |

PR127 is a descendant of the fetched main: their merge base is the main SHA
above. The worktree was created directly at the PR127 commit. The 151 source was
applied with `git cherry-pick --no-commit`, with current-PR127 resolutions in
README, CMake and the operator registry. Its kernel and test paths have no
later changes on the fetched 151 branch. That replay is preserved in the
dedicated integration commit `4b87d3e`, with 150 committed separately above it.
The reproduction manifest in the GR fork pins the final tested kernel SHA.

## Implementation contract

- Uses the exact FBGEMM v1.8.0 `int_nbit_split_embedding_codegen_lookup_function`
  schema, including optional cache, row-alignment and FP8 arguments/defaults.
  The implementation is registered for XPU in the existing `_C` extension.
- Separate `src/codegen/inference` CMake/host/kernel template and
  `generate_forward_quantized.py`, reusing PR127's `CodeTemplate` interface.
  It does not reuse training algorithms or add an autograd path.
- `PoolingMode.NONE`, no weights per index, INT4/INT8 storage, uniform positive
  D divisible by four; one or multiple tables, including reordered/shared
  physical tables and a mixture of INT4/INT8 table types at the same D.
- FP32, FP16 and BF16 output. Indices/offsets independently support int32/int64.
  Used tensors must be contiguous, one-dimensional and on the same XPU.
- A row begins with little-endian FP16 scale and bias (four bytes), then unsigned
  quantized values. INT4 stores the earlier element in the low nibble.
  Stride is `round_up(4 + D * bits / 8, row_alignment)`. Supported alignment is a
  power of two in `[1, 128]`, default 16. Table offsets are byte offsets.
- Quantization stays on CPU using FBGEMM 1.8 `quantize_embs`. Tests assemble the
  same packed rows for CPU and XPU, using HOST placement for the CPU reference
  and DEVICE placement for XPU. CPU host code explicitly selects prefix qparams.
- Each table uses a capped 1D launch with a 64-bit grid-stride loop, including
  work beyond the cap. Empty inputs return an empty XPU tensor without launch.
- Metadata and offsets are gathered on XPU and copied to CPU in one transfer.
  Indices stay on XPU; each packed-row read is guarded and errors are reduced to
  a scalar status. Nonempty calls retain two blocking host reads (metadata and
  error status), down from six. There is no mutable-tensor metadata cache.
  This is not a fully asynchronous operator: synchronous exceptions still wait
  for device completion. Bounds v1 remains enabled and unchanged.
- Raw lookup checks physical storage bounds. It cannot infer the true logical
  row count from padding at the end of a table: frontends must call 151 bounds
  checking with real `rows_per_table`. Direct lookup does not invoke 151 itself.
- Unsupported inputs fail with explicit errors: pooling, weighted lookup,
  nonempty cache/UVM, non-DEVICE placement, other storage/output types, mixed D,
  malformed metadata/offsets, and negative/pruned indices (including `-1`).
  An empty cache tensor is accepted because the no-cache frontend passes one.
  Nondefault/reversed qparam formats are outside this schema's supported layout.
- No cache, pruning/remapping, compile/FakeTensor, DMP, multi-tile, or GR changes.

## Local validation actually performed

All containers used `--network none` and had no GPU device mounts.

- Generator and focused test Ruff checks: PASS.
- `test_codegen_contract`: 1 passed, 104 deselected. This tests generation only.
- Collection: 105 quantized tests total. Explicit plugin-CI deselection leaves
  101 cases and deselects the four TorchRec-dependent collection smokes.
- Exact parsed dispatcher schema versus pinned v1.8 source: PASS.
- IntNBit constructor keyword arguments versus pinned v1.8 source: PASS.
- Generated headers, instantiated host/kernel templates, and replayed 151 bounds
  kernel SYCL syntax: PASS.
- INT4/INT8 host and device code compiled to a generic `spir64` object: PASS.
- Editor diagnostics and Python test syntax: no reported errors.

Compiler image: `fbgemm-xpu-dev:latest`, image ID
`sha256:958cfc9b9f44c44618c7465f98291c3d2f993d91513844624c619ceeb01cc3e4`.
Compiler: Intel oneAPI DPC++/C++ 2026.1.1 (2026.1.1.20260724).
The image has a compiler but no Torch. The installed local Torch is
`2.9.0+hpu_1.23.1-25.git8598e6d`, not the required XPU 2.13 runtime; it also has
no FBGEMM CPU package. The existing 2.13 manylinux builder image has no Torch.
No large dependency installation was attempted.

Limited compile checks borrowed local Torch headers read-only and used
`-DC10_USING_CUSTOM_GENERATED_MACROS` because the HPU distribution lacks the
generated XPU configuration header. This is NOT a supported-version build,
link/import test, PVC AOT validation, or proof of runtime ABI compatibility.
Do not carry that macro override into the real build. The supported-version
build and runtime evidence below supersede this initial compile-only limitation.

Reproduce the dependency-light local test from this worktree:

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m pytest \
  packages/fbgemm-xpu/tests/test_int_nbit_lookup.py -k codegen_contract -v \
  --basetemp="$PWD/build/pytest"
python -m ruff check packages/fbgemm-xpu/tests/test_int_nbit_lookup.py \
  packages/fbgemm-xpu/tests/test_bounds_check_indices.py \
  packages/fbgemm-xpu/src/codegen/genscript/generate_forward_quantized.py
```

## Required build environment

The validated environment is `~/mlperf-dlrmv3-xpu-smoke/venv` on DUT1013PVC.
The existing `~/xpu_smoke` Torch 2.14 venv remains untouched.

Use a separate build environment with oneAPI `icpx` compatible with Torch 2.13
(the repository documents 2026.0), Python >=3.10, Torch `2.13.0+xpu`,
`fbgemm-gpu-cpu==1.8.0`, NumPy `~=2.0`, Jinja2, pybind11,
scikit-build-core >=0.10, CMake and Ninja. Tests additionally need pytest,
hypothesis and expecttest. QuantEmbeddingCollection needs the main agent's
pinned TorchRec installation compatible with this runtime. Do not modify
site-packages or install an unrelated newest TorchRec over that environment.

From the worktree, in that prepared build environment:

```bash
source /opt/intel/oneapi/setvars.sh
python -m pip check
CXX=icpx TORCH_XPU_ARCH_LIST=pvc CMAKE_BUILD_PARALLEL_LEVEL=4 \
  python -m pip wheel --no-build-isolation --no-deps \
  ./packages/fbgemm-xpu --wheel-dir build/wheels
sha256sum build/wheels/fbgemm_xpu-*.whl
```

The owner can install that wheel into the coordinated isolated test environment
with `python -m pip install --no-deps <wheel>`, followed by `python -m pip check`.
Do not claim the wheel is usable until `import fbgemm_xpu` and XPU dispatcher
checks succeed with this exact runtime.

## Validated runtime gates

Runs were serialized on one logical XPU. These tests never skip unavailable XPU
or silently use CPU. Missing dependencies, missing dispatch, all-skipped runs,
and a high-level frontend failure are not successes.

- Runtime: Python 3.12.3, Torch 2.13.0+xpu, FBGEMM CPU 1.8.0.
- TorchRec source: `3d85f3c988ba2c6e809482aeb901ffc89b1defea`, via PYTHONPATH.
- Compiler: existing `fbgemm-xpu-dev:latest`, oneAPI 2026.1.1; mounted venv and
  host Python 3.12 development headers read-only. No GPU used during compilation.
- Wheel: `fbgemm_xpu-0.8.0-cp312-cp312-linux_x86_64.whl`.
- SHA256: `19c6309a3a248742266f111b7736712bef99193e72c35227b179e28841719a5d`.
- Import, XPU dispatcher registration and pip check: PASS.
- Bounds: 12 tests and 22 subtests PASS.
- Original int_nbit suite: 105 tests PASS, including all high-level TorchRec cases.
- Optimized lookup: 127 int_nbit tests plus 12 bounds tests PASS (139 total),
  including mutated metadata, error recovery and late grid-stride invalid IDs.
- Optimized wheel SHA256:
  `23d50001dc90e172857be413a607c020fe4246c7e00d0be658d711ff7c2d3114`.
- CPU packed-layout/generator preflight: five tests PASS before GPU validation.
- Target logs: `~/mlperf-dlrmv3-xpu-smoke/logs/{bounds-unit,intnbit-unit}.log`.
- Build script and complete run evidence are in the GR inference worktree under
  `scripts/mlperf-dlrm-v3-xpu/` and `docs/dlrm-v3-xpu/logs/inference/2026-09-17/`.

```bash
python -m pytest packages/fbgemm-xpu/tests/test_int_nbit_lookup.py \
  -k cpu_packed_layout -v
python -m pytest packages/fbgemm-xpu/tests/test_bounds_check_indices.py -v
python -m pytest packages/fbgemm-xpu/tests/test_int_nbit_lookup.py \
  -k 'not high_level' -v
python -m pytest packages/fbgemm-xpu/tests/test_int_nbit_lookup.py \
  -k high_level_int_nbit -v
python -m pytest packages/fbgemm-xpu/tests/test_int_nbit_lookup.py \
  -k high_level_quant_embedding_collection -v
```

Direct cases cover D=4/512, both index widths, three output types, empty inputs,
empty bags/tables, boundary/repeated IDs, padded rows/tables, multiple/shared
tables, invalid/unsupported inputs, and work beyond the capped grid. Numerical
tests compare explicit dequantization, CPU FBGEMM lookup, and XPU output from
identical packed bytes; tolerances are fixed in the tests.

High-level tests use real IntNBit and QuantEmbeddingCollection forwards for
INT4/INT8 and D=4/512, with WARNING bounds checking and invalid IDs. A forwarding
TorchDispatchMode observes completed XPU `bounds_check_indices` and quantized
lookup invocations; warning counters must change and outputs must match CPU.
Bounds version 1 is selected through the frontend attribute, not a nonexistent
v1.8 constructor argument. These tests do not mock kernels or patch FBGEMM.

Plugin CI lacks TorchRec, so it explicitly deselects only the four
QuantEmbeddingCollection cases. Their separate invocation above is REQUIRED for
the approved track; plugin CI alone cannot close it. No high-level blocker has
yet been reproduced on the target environment. If pristine frontends fail,
report the exact error to the owner before considering a separate FBGEMM source
branch. No speculative frontend patch is part of this worktree.

## Difference from the published 151 branch

The two `embedding_bounds_check_kernel` source files are byte-identical to
`origin/ptxpulib-151-bounds-check-indices` at `cbb733c`. The integration commit
`4b87d3e` replays 151 onto PR127, adapting CMake, registration, README and CI.
The original published 151 branch is not rewritten. Its test now fails instead
of skipping when XPU is unavailable; no bounds algorithm or mode changed.

## Measured lookup optimization

Same host, Torch, inputs, eight warmups, 40 synchronized iterations, bounds
enabled, one table, D=512, FP32 output. Medians in milliseconds:

| Type | Indices | Original | Optimized |
| --- | ---: | ---: | ---: |
| INT4 | 256 | 0.840 | 0.201 |
| INT4 | 32768 | 1.562 | 0.940 |
| INT8 | 256 | 1.343 | 0.199 |
| INT8 | 32768 | 2.032 | 1.475 |

These are local exploratory timings, not MLPerf results or a statistical
performance guarantee. The integrated small model did not speed up: INT4
13.44 -> 13.83 ms, INT8 13.67 -> 14.24 ms. Dense/HSTU and runtime launch costs
dominate. Keep that distinction when reporting operator improvements.