# sycl_extension — Custom SYCL Operators tutorial (`mymuladd`)

A standalone reproduction of PyTorch's
[Custom SYCL Operators](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops_sycl.html)
tutorial. It implements a small fused operator `mymuladd` (computing `a * b + c`
element-wise on an Intel GPU / `xpu` device) plus the tutorial's companion ops
`mymul` (`a * b`) and `myadd_out` (in-place `out = a + b`), and exposes them
through PyTorch's dispatcher as `torch.ops.sycl_extension.*`.

See `../docs/sycl-custom-ops-tutorial.md` for a line-by-line walkthrough.

## Layout

```text
sycl_example/
├── setup.py                 # compiles the .sycl sources via SyclExtension
├── sycl_extension/
│   ├── __init__.py          # loads the built .so so the registrations run
│   ├── muladd.sycl          # kernels (functors) + host wrappers + registration
│   └── ops.py               # thin Python wrappers over torch.ops.sycl_extension.*
├── test_sycl_extension.py   # correctness: op output == a * b + c reference
├── scripts/
│   ├── build_wheel.sh       # Stage 1: build the wheel in the compiler image
│   └── deploy_and_test.sh   # Stage 2: ship wheel to a GPU pod + run the test
└── dist/                    # built wheel(s) — the durable, machine-independent artifact
```

## Requirements

- PyTorch >= 2.8 with XPU support (`torch.xpu.is_available()` must be `True`).
- Intel oneAPI compiler (`icpx`) with its environment activated
  (`source /opt/intel/oneapi/setvars.sh`), i.e. *Intel Deep Learning Essentials*.

## Build & test (single XPU box with the compiler)

If you have one machine that has **both** the oneAPI compiler and an Intel GPU:

```bash
cd sycl_example
python -c "import torch; print(torch.xpu.is_available())"   # -> True
pip install -e .                                            # or: python setup.py install
python -m pytest -rsf test_sycl_extension.py
```

## Build & test (split: build box + run-only GPU pod)

The compiler and the GPU may live on different machines (e.g. a run-only k8s
pod that has `torch-xpu` but no oneAPI). In that case build a wheel where the
compiler is, then copy it to the GPU machine to run. Two scripts wrap this:

```bash
# Stage 1 — build the wheel once (on any box with Docker). The wheel is abi3
# (cross-Python); only torch's MINOR version must match the target pod.
./scripts/build_wheel.sh                    # defaults: torch==2.11.0+xpu, cp312
# ./scripts/build_wheel.sh 2.12.1+xpu cp312 # if the pod runs a different torch

# Stage 2 — deploy + test on a GPU pod (pass the pod name; namespace defaults to dev)
./scripts/deploy_and_test.sh <POD_NAME> [NAMESPACE]
```

### Switching GPU machines (pods are preemptible)

The GPU pod may be replaced (e.g. daily). Nothing needs rebuilding when that
happens — the wheel in `dist/` is the durable, machine-independent artifact.
For a new pod just re-run Stage 2:

```bash
./scripts/deploy_and_test.sh <NEW_POD_NAME>
```

It verifies `torch.xpu.is_available()`, copies the wheel + test, installs, and
runs the correctness suite — idempotent, so it's safe to re-run any time.
Only rebuild (Stage 1) if the new machine runs a **different torch minor
version** (then pass it to `build_wheel.sh`).

## Usage

```python
import torch
import sycl_extension

a = torch.randn(8, device="xpu")
b = torch.randn(8, device="xpu")
out = torch.ops.sycl_extension.mymuladd(a, b, 0.5)   # a * b + 0.5
torch.testing.assert_close(out, a * b + 0.5)
```
