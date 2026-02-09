from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import pinocchio as pin  # type: ignore
except Exception as exc:  # pragma: no cover
    pin = None
    _pin_exc = exc


def _require_pinocchio():  # pragma: no cover
    if pin is None:
        raise ImportError(
            "pinocchio is required for inverse-dynamics contact inference. "
            "Install it (recommended via conda-forge) before calling these functions."
        ) from _pin_exc
    return pin


@dataclass
class ContactInferenceResult:
    tau_ext: np.ndarray
    contact_forces: Optional[np.ndarray] = None


def compute_external_torques(
    urdf_path: str,
    q: np.ndarray,
    qdot: np.ndarray,
    qddot: np.ndarray,
    tau_cmd: np.ndarray,
) -> ContactInferenceResult:
    """
    Compute external torques from inverse dynamics:
        tau_ext = tau_cmd - (M qddot + C qdot + G)
    """
    pin_ = _require_pinocchio()
    model = pin_.buildModelFromUrdf(urdf_path)
    data = model.createData()
    tau_model = pin_.rnea(model, data, q, qdot, qddot)
    tau_ext = tau_cmd - tau_model
    return ContactInferenceResult(tau_ext=tau_ext)


def solve_contact_forces(
    urdf_path: str,
    q: np.ndarray,
    qdot: np.ndarray,
    qddot: np.ndarray,
    tau_cmd: np.ndarray,
    contact_jacobian: np.ndarray,
) -> ContactInferenceResult:
    """
    Solve J^T F = tau_ext for contact forces using least squares.
    """
    result = compute_external_torques(urdf_path, q, qdot, qddot, tau_cmd)
    tau_ext = result.tau_ext
    jt = contact_jacobian.T
    forces, *_ = np.linalg.lstsq(jt, tau_ext, rcond=None)
    result.contact_forces = forces
    return result
