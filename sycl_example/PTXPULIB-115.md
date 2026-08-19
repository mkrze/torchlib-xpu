# PTXPULIB-115 — Build and run the PyTorch "Custom SYCL Operators" tutorial (`mymuladd`)

Ticket: <https://jira.devtools.intel.com/browse/PTXPULIB-115>

Standalone reproduction of PyTorch's
[Custom SYCL Operators](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops_sycl.html)
tutorial: a fused `mymuladd` operator (`a * b + c`) implemented as a SYCL
functor, registered under the `XPU` dispatch key, built into a wheel, and
validated on a real Intel GPU.

Project location: `sycl_example/`.

## Acceptance criteria — status

| # | Criterion | Status |
|---|-----------|:------:|
| 1 | `sycl_example/` project created following the tutorial's file layout | Done |
| 2 | `mymuladd` kernel implemented as a SYCL functor (`MulAddKernelFunctor` pattern) | Done |
| 3 | Registered via `TORCH_LIBRARY`/`TORCH_LIBRARY_IMPL(..., XPU, m)`, callable as `torch.ops.sycl_extension.mymuladd` | Done |
| 4 | Extension builds successfully (`python setup.py install` / `pip install -e .`) | Done |
| 5 | `torch.xpu.is_available()` confirmed `True` before running | Done |
| 6 | `test_sycl_extension.py` passes (`mymuladd` == `a * b + c` on `xpu`) | Done |

---

## 1. `sycl_example/` project created following the tutorial's file layout

Created at `sycl_example/`, mirroring the tutorial layout:

```text
sycl_example/
├── setup.py                 # SyclExtension build
├── sycl_extension/
│   ├── __init__.py          # loads the .so so registrations run
│   ├── muladd.sycl          # kernels + wrappers + registration
│   └── ops.py               # Python wrappers
└── test_sycl_extension.py   # correctness test
```

## 2. `mymuladd` kernel implemented as a SYCL functor (`MulAddKernelFunctor`)

Functor with a constructor that captures all inputs into named members and an
`operator()(sycl::nd_item<1>)` that does the per-element work with a tail guard
(`if (idx < numel_)`):

```cpp
class MulAddKernelFunctor {
 public:
  MulAddKernelFunctor(
      int numel, const float* a, const float* b, float c, float* result)
      : numel_(numel), a_(a), b_(b), c_(c), result_(result) {}

  void operator()(const sycl::nd_item<1>& item) const {
    int idx = item.get_global_id(0);
    if (idx < numel_) {
      result_[idx] = a_[idx] * b_[idx] + c_;
    }
  }

 private:
  int numel_;
  const float* a_;
  const float* b_;
  float c_;
  float* result_;
};
```

Host wrapper `mymuladd_xpu` validates with `TORCH_CHECK`, makes inputs
`.contiguous()`, allocates via `at::empty_like`, grabs PyTorch's XPU queue, and
launches over `nd_range<1>(blocks * 256, 256)`:

```cpp
at::Tensor mymuladd_xpu(const at::Tensor& a, const at::Tensor& b, double c) {
  TORCH_CHECK(a.sizes() == b.sizes(), "a and b must have the same shape");
  TORCH_CHECK(a.dtype() == at::kFloat, "a must be a float tensor");
  TORCH_CHECK(a.device().is_xpu(), "a must be an XPU tensor");
  // ... (b checked the same way)

  at::Tensor a_contig = a.contiguous();
  at::Tensor b_contig = b.contiguous();
  at::Tensor result = at::empty_like(a_contig);

  const float* a_ptr = a_contig.data_ptr<float>();
  const float* b_ptr = b_contig.data_ptr<float>();
  float* res_ptr = result.data_ptr<float>();
  int numel = a_contig.numel();

  sycl::queue& queue = c10::xpu::getCurrentXPUStream().queue();
  constexpr int threads = 256;
  int blocks = (numel + threads - 1) / threads;

  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<MulAddKernelFunctor>(
        sycl::nd_range<1>(blocks * threads, threads),
        MulAddKernelFunctor(numel, a_ptr, b_ptr, static_cast<float>(c), res_ptr));
  });
  return result;
}
```

The tutorial's two companion ops (`mymul` = `a * b`, `myadd_out` = in-place
`a + b`) are implemented the same way for faithfulness; they are not required by
the ticket.

## 3. Registration via `TORCH_LIBRARY` / `TORCH_LIBRARY_IMPL(..., XPU, m)`

Schema declared once; XPU implementation bound separately:

```cpp
TORCH_LIBRARY(sycl_extension, m) {
  m.def("mymuladd(Tensor a, Tensor b, float c) -> Tensor");
  m.def("mymul(Tensor a, Tensor b) -> Tensor");
  m.def("myadd_out(Tensor a, Tensor b, Tensor(a!) out) -> ()");
}

TORCH_LIBRARY_IMPL(sycl_extension, XPU, m) {
  m.impl("mymuladd", &mymuladd_xpu);
  m.impl("mymul", &mymul_xpu);
  m.impl("myadd_out", &myadd_out_xpu);
}
```

Verified callable on the pod after `import sycl_extension`:

```text
op sycl_extension.mymuladd    # torch.ops.sycl_extension.mymuladd resolved
```

## 4. Extension builds successfully

Built **inside the recommended `fbgemm-xpu` dev container**
([`fbgemm_dev_tools`](https://github.com/intel-sandbox/fbgemm_dev_tools),
image `fbgemm-xpu-dev:latest`), which already provides the SYCL toolchain
(oneAPI `icpx`), the XPU drivers, and the PyTorch XPU build (`torch 2.13.0+xpu`).
The project was copied into the container and built with `SyclExtension` +
`BuildExtension`:

```bash
docker exec fbgemm-dev bash -lc '
  cd /workspace/sycl_example &&
  SYCL_EXT_SKIP_XPU_CHECK=1 python setup.py bdist_wheel'
```

```text
creating 'dist/sycl_extension-0.0.1-cp39-abi3-linux_x86_64.whl'
adding 'sycl_extension/_C.abi3.so'   # compiled kernel
```

Two build fixes were required:
- Dropped `torch/extension.h` (its pybind11 is incompatible with
  `Py_LIMITED_API`) in favor of `ATen/ATen.h` + `torch/library.h`.
- Guarded the Windows `PyInit__C` stub behind `#ifdef _WIN32` (on Linux the
  `.so` is loaded via `ctypes.CDLL`).

Note on `SYCL_EXT_SKIP_XPU_CHECK=1`: `setup.py` asserts a live XPU at build time
(per the tutorial). The dev container on this host has no local Intel GPU, so the
flag skips that assert — the SYCL compiler does not need a device (kernels are
JIT'd at runtime). Because the extension uses `py_limited_api=True`, the result
is a single `cp39-abi3` wheel that loads on any Python ≥ 3.9.

## 5. `torch.xpu.is_available()` confirmed `True`

Checked on the GPU pod before running:

```text
torch 2.13.0+xpu | xpu True
```

## 6. `test_sycl_extension.py` passes on the `xpu` device

The test compares each op against its plain-PyTorch reference
(`mymuladd` vs `a * b + c`) with `torch.testing.assert_close` on `device="xpu"`.
Run on the real Intel GPU:

```text
test_sycl_extension.py::TestMyMulAdd::test_correctness_xpu PASSED   [ 33%]
test_sycl_extension.py::TestMyMul::test_correctness_xpu PASSED      [ 66%]
test_sycl_extension.py::TestMyAddOut::test_correctness_xpu PASSED   [100%]
============================== 3 passed in 3.22s ===============================
```

---

## Environment notes / deviations from the ticket

- **Dev container:** built inside the ticket's recommended `fbgemm-xpu` dev
  container ([`fbgemm_dev_tools`](https://github.com/intel-sandbox/fbgemm_dev_tools),
  image `fbgemm-xpu-dev:latest`, container `fbgemm-dev`), which ships the SYCL
  toolchain (`icpx`), XPU drivers, and the PyTorch XPU build. This is the same
  container used for the FBGEMM operator work (PTXPULIB-117); see
  [`docs/fbgemm-xpu-dev-container.md`](../docs/fbgemm-xpu-dev-container.md).
- **Torch/GPU used for validation:** `torch 2.13.0+xpu`, Python 3.12, Intel B60
  XPU (k8s namespace `dev`).

### Honest caveat: build box ≠ run box

The extension was **built** in the recommended `fbgemm-xpu` dev container
(`fbgemm-dev`), which provides the SYCL toolchain (`icpx`) and the PyTorch XPU
build — the container the ticket asks for. It was **run/validated** on a separate
Intel B60 GPU pod (`torch213-…-d5qtn`, `torch 2.13.0+xpu`), where
`torch.xpu.is_available()==True` and all 3 correctness tests pass.

Build and run are on **two different boxes** only because no single machine here
has *both* the compiler and a GPU — verified:

| Box | GPU (`/dev/dri`) | `torch.xpu` | `icpx` / oneAPI |
|---|:--:|:--:|:--:|
| `fbgemm-dev` container (build) | none | False | **yes** |
| `torch213-…-d5qtn` pod (run) | 1× B60 | True | none (run-only) |
| `manual-job-…-66dgw` pod | 1× B60 | True | none (run-only) |

Because the extension is built with `py_limited_api=True` (a `cp39-abi3` wheel),
the container-built wheel loads unchanged on the pod's Python 3.12. On a
GPU-equipped host the *same* container would satisfy `torch.xpu.is_available()
==True` and run the test in place, with zero changes — i.e. this is an
infrastructure split, not a deviation in approach or toolchain.
