# Copyright (c) 2026 Intel Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Let fbgemm_gpu's DENSE TBE module allocate its buffers on an XPU device.

``DenseTableBatchedEmbeddingBagsCodegen.__init__`` in fbgemm_gpu 1.8.0 picks the
buffer device from a fixed three-way choice::

    torch.device("cpu") if use_cpu
    else torch.device(f"mtia:{...}") if use_mtia
    else torch.cuda.current_device()

There is no XPU branch and no device parameter, so constructing it with
``use_cpu=False`` on a machine without CUDA raises "Torch not compiled with CUDA
enabled" before any kernel runs. The FUSED class in the same file already accepts
an explicit device; only the DENSE one does not -- and DENSE is the only compute
kernel TorchRec permits on XPU, so it is unavoidable on the XPU path.

``current_device`` is used solely as the ``device=`` argument when allocating
buffers, and the DENSE class references ``torch.cuda`` nowhere else, so
redirecting that one call is enough. Kernel dispatch, buffer shapes and dtypes
are untouched.

The redirect is skipped when CUDA is present, so a machine with both never has
its CUDA behaviour altered.
"""

import contextlib
import functools
from typing import Iterator

import torch


@contextlib.contextmanager
def _cuda_current_device_returns_xpu() -> Iterator[None]:
    """Point ``torch.cuda.current_device`` at the active XPU device.

    Returns a ``torch.device`` rather than the ``int`` the real function returns.
    That is what the DENSE constructor wants: it passes the result straight to
    ``device=``, and an int there would be read as a CUDA ordinal.

    Not thread-safe. Module construction is single-threaded in TorchRec, and the
    window is one ``__init__`` call.
    """
    original = torch.cuda.current_device
    torch.cuda.current_device = lambda: torch.device("xpu", torch.xpu.current_device())
    try:
        yield
    finally:
        torch.cuda.current_device = original


def install() -> bool:
    """Patch the DENSE TBE constructor. Returns True if the patch is in effect."""
    try:
        from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
            DenseTableBatchedEmbeddingBagsCodegen,
        )
    except ImportError:
        return False

    original_init = DenseTableBatchedEmbeddingBagsCodegen.__init__
    if getattr(original_init, "_fbgemm_xpu_patched", False):
        return True

    @functools.wraps(original_init)
    def __init__(self, *args, **kwargs) -> None:  # pyre-ignore[2, 3]
        if torch.cuda.is_available() or not torch.xpu.is_available():
            original_init(self, *args, **kwargs)
            return
        # use_cpu / use_mtia short-circuit the ternary before it reaches
        # torch.cuda.current_device(), so the redirect is inert for those.
        with _cuda_current_device_returns_xpu():
            original_init(self, *args, **kwargs)

    __init__._fbgemm_xpu_patched = True  # pyre-ignore[16]
    DenseTableBatchedEmbeddingBagsCodegen.__init__ = __init__
    return True
