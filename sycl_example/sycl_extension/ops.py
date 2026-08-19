import torch
from torch import Tensor

__all__ = ["mymuladd", "mymul", "myadd_out"]


def mymuladd(a: Tensor, b: Tensor, c: float) -> Tensor:
    """Performs a * b + c in an efficient fused SYCL kernel."""
    return torch.ops.sycl_extension.mymuladd.default(a, b, c)


def mymul(a: Tensor, b: Tensor) -> Tensor:
    """Performs a * b in a SYCL kernel."""
    return torch.ops.sycl_extension.mymul.default(a, b)


def myadd_out(a: Tensor, b: Tensor, out: Tensor) -> None:
    """Writes a + b into the pre-allocated ``out`` tensor (in-place)."""
    torch.ops.sycl_extension.myadd_out.default(a, b, out)
