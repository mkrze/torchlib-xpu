import glob
import os

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, SyclExtension

library_name = "sycl_extension"

if torch.__version__ < "2.7.0":
    raise RuntimeError(
        f"Building {library_name} requires PyTorch >= 2.7, got {torch.__version__}"
    )

# The tutorial asserts a live XPU at build time. When cross-compiling on a
# GPU-less build box (the SYCL compiler does not need a device; kernels are
# JIT'd at runtime), set SYCL_EXT_SKIP_XPU_CHECK=1 to skip this.
if os.environ.get("SYCL_EXT_SKIP_XPU_CHECK") != "1":
    assert (
        torch.xpu.is_available()
    ), "XPU is not available, please check your environment"

extra_compile_args = {
    "cxx": [
        "-O3",
        "-fdiagnostics-color=always",
        "-DPy_LIMITED_API=0x03090000",
    ],
    "sycl": ["-O3"],
}

this_dir = os.path.dirname(os.path.curdir)
sources = glob.glob(
    os.path.join(this_dir, library_name, "*.sycl"), recursive=True
)

ext_modules = [
    SyclExtension(
        f"{library_name}._C",
        sources,
        extra_compile_args=extra_compile_args,
        py_limited_api=True,
    )
]

setup(
    name=library_name,
    version="0.0.1",
    packages=find_packages(),
    ext_modules=ext_modules,
    install_requires=["torch"],
    description="Custom SYCL operator (mymuladd) PyTorch extension tutorial",
    cmdclass={"build_ext": BuildExtension},
    options={"bdist_wheel": {"py_limited_api": "cp39"}},
)
