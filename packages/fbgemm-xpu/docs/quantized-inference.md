# Quantized INT4/INT8 XPU Inference Track

## Status and ownership

2026-09-23: the branch was rebuilt on the current `intel/main` after PR #131
merged. A supported-version cp312 wheel build, the full plugin test selection,
CPU/XPU numerical validation, and both high-level frontend smokes pass on BMG.
This is a correctness milestone, not a full-model accuracy or performance
claim.

No TorchRec or FBGEMM source changes or site-packages patches are needed for the
kernel track. TorchRec is supplied read-only through `PYTHONPATH` only for the
four collection smoke cases that plugin CI intentionally deselects.

- Current `intel/main` (merged PR #131): `b59acf9742a1a6b3d4fb7cb05bc87acb9b5de820`
- Quantized lookup commit after rebase: `94bb034`
- Validated branch head before this documentation update: `4a9f86a`
- `intel`: `https://github.com/intel/torchlib-xpu.git`
- `origin`: `https://github.com/mkrze/torchlib-xpu.git`

## Pinned sources

| Source | Exact ref / SHA |
| --- | --- |
| Fetched `intel/main` with merged PR #131 | `b59acf9742a1a6b3d4fb7cb05bc87acb9b5de820` |
| PR #132 lookup implementation after rebase | `94bb034` |
| PR #132 Bandit follow-up after rebase | `80c0c92` |
| PR #132 README follow-up after rebase | `4a9f86a` |
| Read-only FBGEMM `v1.8.0` tag object | `f349896c8439444889a49e695807503e8ac68bbf` |
| Read-only FBGEMM `v1.8.0^{}` source commit | `99a76f4ed785d2d00579eae5fd45440a1242d657` |
| Read-only TorchRec smoke source | `3d85f3c988ba2c6e809482aeb901ffc89b1defea` |

PR #131 was squash-merged as `b59acf9`; its tree is identical to the reviewed
PR head. PR #132 now contains only its three rebased code/review commits and
this documentation follow-up. Relative to `intel/main`, it does not modify
either bounds kernel file.

## Implementation contract

- Uses the exact FBGEMM v1.8.0 `int_nbit_split_embedding_codegen_lookup_function`
  schema, including optional cache, row-alignment and FP8 arguments/defaults.
  The implementation is registered for XPU in the existing `_C` extension.
- Separate `src/codegen/inference` CMake/host/kernel template and
  `generate_forward_quantized.py`, reusing PR127's `CodeTemplate` interface.
  It does not reuse training algorithms or add an autograd path.
- `PoolingMode.NONE`, no weights per index, INT4/INT8 storage, uniform positive
  D divisible by four; one or multiple tables, including reordered/shared
  physical tables and a mixture of INT4/INT8 table types at the same D. The
  four-element boundary preserves the documented FBGEMM IntNBit frontend
  contract; it is a supported-subset decision rather than a scalar-load or row
  padding requirement of the XPU kernel.
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
  for device completion. Bounds v1 is provided by merged PR #131; this branch
  does not modify its kernel files.
- Raw lookup checks physical storage bounds. It cannot infer the true logical
  row count from padding at the end of a table: frontends must call the bounds
  checker from merged PR #131 with real `rows_per_table`. Direct lookup does
  not invoke the bounds checker itself.
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
- Collection: 127 quantized tests total. Explicit plugin-CI deselection leaves
  123 cases and deselects the four TorchRec-dependent collection smokes.
- Exact parsed dispatcher schema versus pinned v1.8 source: PASS.
- IntNBit constructor keyword arguments versus pinned v1.8 source: PASS.
- Generated headers, instantiated host/kernel templates, and dependent bounds
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

Current validation uses an isolated Python 3.12 / Torch 2.13 environment on an
Intel Arc Pro B60 (BMG).

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

- Revision: `4a9f86a`, based on current `intel/main` `b59acf9`.
- Runtime: Python 3.12.3, Torch 2.13.0+xpu, FBGEMM CPU 1.8.0.
- TorchRec source: `3d85f3c988ba2c6e809482aeb901ffc89b1defea`, via PYTHONPATH.
- Compiler image: `pytorch/manylinux2_28-builder:xpu-v2.13.0-rc1`; no GPU was
  used during compilation.
- Wheel: `fbgemm_xpu-0.8.0-cp312-cp312-linux_x86_64.whl`.
- SHA256: `4c6a0b0971a207428f821ad12fe40cfc64a01692ae6c608b3bc234bd05e15166`.
- Import and both XPU dispatcher registrations: PASS.
- Full plugin suite with the exact CI deselection: 335 passed, 9 skipped,
  4 deselected, 36 subtests passed. All skips require two XPU devices.
- Separate TorchRec QuantEmbeddingCollection smoke: 4 passed.
- Local generator contract, Ruff, Bandit, schema/ancestry checks and
  `git diff --check`: PASS.
- Pod-wide `pip check` is not evidence for this run: the supplied pod has a
  pre-existing `torchvision 0.26.0+xpu` requirement for Torch 2.11 while the pod
  runtime is Torch 2.13. Tests used an isolated plugin environment with an
  explicit `torch.__version__ == "2.13.0+xpu"` gate.

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
empty bags, boundary/repeated IDs, padded rows/tables, multiple/shared
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

## Relationship to the merged bounds checker

PR #131 is merged into the base as `b59acf9`. Relative to that base, PR #132
does not modify either `embedding_bounds_check_kernel` file. It changes the
bounds test only so an explicitly selected XPU suite fails instead of skipping
when hardware is unavailable.

## Measured lookup optimization

Same host, Torch, inputs, eight warmups, 40 synchronized iterations, bounds
enabled, one table, D=512, FP32 output. Medians in milliseconds:

| Type | Indices | Original | Optimized |
| --- | ---: | ---: | ---: |
| INT4 | 256 | 0.840 | 0.201 |
| INT4 | 32768 | 1.562 | 0.940 |
| INT8 | 256 | 1.343 | 0.199 |
| INT8 | 32768 | 2.032 | 1.475 |

### Multi-table launch scaling

The current implementation submits one lookup kernel per table. A BMG sweep
held the total work fixed at 4096 indices, `D=128`, FP32 output and 256 physical
rows per table while varying only the table count. Values below are medians
from 30 measured calls after 10 warmups:

| Type | Tables | Wall time (ms) | XPU event time (ms) |
| --- | ---: | ---: | ---: |
| INT4 | 1 | 1.393 | 0.466 |
| INT4 | 8 | 1.875 | 0.638 |
| INT4 | 32 | 1.958 | 0.811 |
| INT4 | 64 | 1.980 | 0.871 |
| INT4 | 128 | 1.761 | 1.114 |
| INT8 | 1 | 1.791 | 0.692 |
| INT8 | 8 | 2.053 | 0.738 |
| INT8 | 32 | 2.175 | 0.875 |
| INT8 | 64 | 2.167 | 0.971 |
| INT8 | 128 | 2.547 | 1.398 |

The per-table launch overhead is measurable but did not make wall time scale
linearly with the number of tables in this sweep. From one to 128 tables, the
absolute wall-time increase was 0.37 ms for INT4 and 0.76 ms for INT8. A fused
multi-table launch remains a possible follow-up optimization.

These are local exploratory timings, not MLPerf results or a statistical
performance guarantee. The integrated small model did not speed up: INT4
13.44 -> 13.83 ms, INT8 13.67 -> 14.24 ms. Dense/HSTU and runtime launch costs
dominate. Keep that distinction when reporting operator improvements.
