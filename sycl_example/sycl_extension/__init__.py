import ctypes
import platform
from pathlib import Path

import torch

# The compiled kernels live in a .so (Linux) / .pyd (Windows). Loading that
# shared library runs its TORCH_LIBRARY registrations, which is the side effect
# that makes torch.ops.sycl_extension.* appear. On Linux the extension has no
# PyInit function, so we load it explicitly with ctypes.CDLL rather than import.
_lib_glob = "**/*.pyd" if platform.system() == "Windows" else "**/*.so"

# Search the package dir (wheel install) first, then build/ (editable install).
_pkg_dir = Path(__file__).parent
_search_dirs = [_pkg_dir, _pkg_dir.parent / "build"]
_lib_file = next(
    (f for d in _search_dirs if d.exists() for f in d.glob(_lib_glob)),
    None,
)
if _lib_file is None:
    raise ImportError(
        "Could not find the compiled sycl_extension._C library in "
        f"{[str(d) for d in _search_dirs]}. Did the build succeed?"
    )

with torch._ops.dl_open_guard():
    loaded_lib = ctypes.CDLL(str(_lib_file))

from . import ops  # noqa: E402  (import after the library is loaded)

__all__ = ["ops", "loaded_lib"]
