from __future__ import annotations

from typing import Any, Dict, Tuple

from sqlalchemy.orm import Session

from app import crud
from app.core.policy_manifest import build_policy_manifest, upload_policy_manifest


def register_physhoi_policy(
    db: Session,
    dataset_id: str,
    checkpoint_uri: str,
    task: str,
    cfg_env: str,
    cfg_train: str,
    frames_scale: float = 1.0,
    extra: Dict[str, Any] | None = None,
) -> Tuple[str, str]:
    """
    Register a PhysHOI checkpoint as a platform policy.
    Returns (policy_id, manifest_uri).
    """
    params_json: Dict[str, Any] = {
        "source": "physhoi",
        "task": task,
        "cfg_env": cfg_env,
        "cfg_train": cfg_train,
        "frames_scale": frames_scale,
    }
    if extra:
        params_json.update(extra)

    job = crud.create_xmimic_job(
        db,
        dataset_id=dataset_id,
        mode="mocap",
        params_json=params_json,
        idempotency_key=f"physhoi:{task}:{checkpoint_uri}",
    )
    job.status = "IMPORTED"
    db.add(job)
    db.commit()
    db.refresh(job)

    policy = crud.create_policy(db, job.id, checkpoint_uri, metadata_json=params_json)
    manifest = build_policy_manifest(str(policy.id), checkpoint_uri, params_json)
    manifest_uri = upload_policy_manifest(str(policy.id), manifest)
    return str(policy.id), manifest_uri
