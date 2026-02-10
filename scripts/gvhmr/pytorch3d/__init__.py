"""Minimal pytorch3d stub used to run GVHMR on Windows.

GVHMR (zju3dv/GVHMR) imports `pytorch3d.transforms` and `pytorch3d.ops.knn`.
Official PyTorch3D wheels are not reliably available on Windows, so we provide
pure-PyTorch fallbacks for the specific APIs GVHMR uses.

This is *not* a full PyTorch3D implementation.
"""

from . import transforms  # noqa: F401
from . import ops  # noqa: F401

