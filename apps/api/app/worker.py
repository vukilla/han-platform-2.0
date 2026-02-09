import io
import json
import zipfile
from datetime import datetime
from time import sleep
from uuid import UUID

import numpy as np
from celery import Celery
from kombu import Queue

from app.core.config import get_settings
from app.core.alerts import send_alert
from app.core.storage import upload_bytes, upload_text
from app import crud
from app.quality import evaluate_demo, evaluate_clip
from app.db import SessionLocal
from app import models

settings = get_settings()

celery_app = Celery(
    "han_worker",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

celery_app.conf.update(
    task_default_queue=settings.celery_default_queue,
    task_queues=[
        Queue(settings.celery_cpu_queue),
        Queue(settings.celery_gpu_queue),
    ],
    task_routes={
        "app.worker.run_xgen_job": {"queue": settings.celery_cpu_queue},
        "app.worker.run_xmimic_job": {"queue": settings.celery_gpu_queue},
    },
    worker_prefetch_multiplier=settings.celery_prefetch_multiplier,
    task_acks_late=settings.celery_task_acks_late,
    task_reject_on_worker_lost=settings.celery_reject_on_worker_lost,
)


XGEN_STAGES = [
    "INGEST_VIDEO",
    "ESTIMATE_POSE",
    "RETARGET",
    "CONTACT_SYNTH",
    "NONCONTACT_SIM",
    "AUGMENT",
    "EXPORT_DATASET",
    "RENDER_PREVIEWS",
    "QUALITY_SCORE",
]

XMIMIC_STAGES = [
    "BUILD_ENV",
    "TRAIN_TEACHER",
    "DISTILL_STUDENT",
    "EVAL_POLICY",
    "EXPORT_CHECKPOINT",
    "QUALITY_SCORE",
]


def _update_job_status(db, job, status, error=None):
    job.status = status
    if status in {"COMPLETED", "FAILED"}:
        job.finished_at = datetime.utcnow()
    if job.started_at is None:
        job.started_at = datetime.utcnow()
    if error:
        job.error = error
    db.add(job)
    db.commit()
    db.refresh(job)


def _write_job_log(job_type: str, job_id: str, lines: list[str]) -> str:
    key = f"logs/{job_type}/{job_id}.log"
    return upload_text(key, "\n".join(lines))


def _npz_bytes(**arrays: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


def _build_synthetic_clip(frames: int, nq: int, contact_dim: int, seed: int) -> tuple[bytes, dict, list[str]]:
    """Create a deterministic clip artifact for end-to-end platform plumbing.

    This keeps the platform usable even when the full GPU sim/training stack is not
    running on the control-plane machine.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, frames, dtype=np.float32)

    freqs = rng.uniform(0.5, 2.0, size=(nq,)).astype(np.float32)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=(nq,)).astype(np.float32)
    robot_qpos = 0.35 * np.sin(2.0 * np.pi * t[:, None] * freqs[None, :] + phases[None, :]).astype(np.float32)
    robot_qvel = np.vstack([np.zeros((1, nq), dtype=np.float32), np.diff(robot_qpos, axis=0)]).astype(np.float32)

    root_pose = np.zeros((frames, 7), dtype=np.float32)
    root_pose[:, 0] = 0.1 * t
    root_pose[:, 6] = 1.0

    object_pose = np.zeros((frames, 7), dtype=np.float32)
    object_pose[:, 0] = 0.35 + 0.02 * np.sin(2.0 * np.pi * t)
    object_pose[:, 1] = 0.05 + 0.02 * np.cos(2.0 * np.pi * t)
    object_pose[:, 2] = 0.30
    object_pose[:, 6] = 1.0

    contact_graph = np.zeros((frames, contact_dim), dtype=np.float32)
    c0 = int(frames * 0.35)
    c1 = int(frames * 0.70)
    contact_graph[c0:c1, :] = 1.0

    phase = np.zeros((frames,), dtype=np.int32)
    phase[:c0] = 0
    phase[c0:c1] = 1
    phase[c1:] = 2

    npz = _npz_bytes(
        robot_qpos=robot_qpos,
        robot_qvel=robot_qvel,
        root_pose=root_pose,
        object_pose=object_pose,
        contact_graph=contact_graph,
        phase=phase,
    )

    metadata = {
        "synthetic": True,
        "frames": frames,
        "nq": nq,
        "contact_dim": contact_dim,
        "contact_range": [c0, c1],
    }
    tags = [f"seed:{seed}", "synthetic:v0"]
    return npz, metadata, tags


def _ensure_dataset_for_xgen_job(db, job: models.XGenJob) -> models.Dataset:
    params = dict(job.params_json or {})
    dataset_id = params.get("dataset_id")
    if dataset_id:
        try:
            existing = crud.get_dataset(db, UUID(str(dataset_id)))
            if existing:
                return existing
        except Exception:
            pass

    demo = job.demo
    dataset = crud.create_dataset(
        db,
        demo.project_id,
        source_demo_id=demo.id,
        status="BUILDING",
        summary_json={"source": "xgen", "job_id": str(job.id)},
    )
    params["dataset_id"] = str(dataset.id)
    job.params_json = params
    db.add(job)
    db.commit()
    db.refresh(dataset)
    return dataset


@celery_app.task(bind=True, max_retries=3, retry_backoff=True)
def run_xgen_job(self, job_id: str):
    db = SessionLocal()
    job = None
    log_lines = [f"XGen job {job_id} started at {datetime.utcnow().isoformat()}Z"]
    try:
        job = db.query(models.XGenJob).filter(models.XGenJob.id == job_id).first()
        if not job:
            return
        dataset = _ensure_dataset_for_xgen_job(db, job)
        for stage in XGEN_STAGES:
            _update_job_status(db, job, stage)
            log_lines.append(f"{datetime.utcnow().isoformat()}Z stage={stage}")
            sleep(0.5)
            if stage == "EXPORT_DATASET":
                existing = crud.list_dataset_clips(db, dataset.id)
                if not existing:
                    params = dict(job.params_json or {})
                    frames = int(params.get("frames", 60))
                    nq = int(params.get("nq", 24))
                    contact_dim = int(params.get("contact_dim", 5))
                    clip_count = int(params.get("clip_count", 10))

                    manifest = {
                        "dataset_id": str(dataset.id),
                        "project_id": str(dataset.project_id),
                        "source_demo_id": str(dataset.source_demo_id) if dataset.source_demo_id else None,
                        "created_at": datetime.utcnow().isoformat() + "Z",
                        "clip_count": clip_count,
                        "frames": frames,
                        "nq": nq,
                        "contact_dim": contact_dim,
                    }

                    zip_buf = io.BytesIO()
                    with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                        zf.writestr("manifest.json", json.dumps(manifest, indent=2).encode("utf-8"))
                        for i in range(clip_count):
                            npz_bytes, meta, tags = _build_synthetic_clip(
                                frames=frames,
                                nq=nq,
                                contact_dim=contact_dim,
                                seed=1000 + i,
                            )
                            npz_key = f"datasets/{dataset.id}/clips/{i:04d}.npz"
                            upload_bytes(npz_key, npz_bytes, content_type="application/octet-stream")

                            clip = crud.create_dataset_clip(
                                db,
                                dataset.id,
                                uri_npz=npz_key,
                                # Until clip rendering is implemented, point previews at the original demo video.
                                uri_preview_mp4=job.demo.video_uri,
                                augmentation_tags=tags,
                                stats_json={"status": "Generated", "frames": frames},
                            )

                            zf.writestr(f"clips/{clip.clip_id}/clip.npz", npz_bytes)
                            zf.writestr(
                                f"clips/{clip.clip_id}/metadata.json",
                                json.dumps(meta, indent=2).encode("utf-8"),
                            )

                    dataset_zip_key = f"datasets/{dataset.id}/dataset.zip"
                    upload_bytes(dataset_zip_key, zip_buf.getvalue(), content_type="application/zip")

                    dataset.status = "READY"
                    dataset.summary_json = {
                        **(dataset.summary_json or {}),
                        "clip_count": clip_count,
                        "frames": frames,
                        "nq": nq,
                        "contact_dim": contact_dim,
                        "dataset_zip_key": dataset_zip_key,
                        "name": (dataset.summary_json or {}).get("name", "XGen Dataset"),
                    }
                    db.add(dataset)
                    db.commit()
        _update_job_status(db, job, "COMPLETED")
        log_lines.append(f"{datetime.utcnow().isoformat()}Z status=COMPLETED")
        try:
            demo = job.demo
            frame_count = int((job.params_json or {}).get("frame_count", 0))
            pose_ok = bool((job.params_json or {}).get("pose_ok", True))
            quality = evaluate_demo(frame_count=frame_count, pose_ok=pose_ok)
            crud.create_quality_score(
                db,
                entity_type="demo",
                entity_id=demo.id,
                score=quality.score,
                breakdown_json=quality.breakdown,
                validator_status="pending",
            )
        except Exception:
            pass
    except Exception as exc:
        if job and self.request.retries < self.max_retries:
            _update_job_status(db, job, "RETRYING", error=str(exc))
            log_lines.append(f"{datetime.utcnow().isoformat()}Z status=RETRYING error={exc}")
            send_alert(
                "xgen_retry",
                {"job_id": job_id, "error": str(exc), "retries": self.request.retries},
            )
            raise self.retry(exc=exc)
        if job:
            _update_job_status(db, job, "FAILED", error=str(exc))
            send_alert("xgen_failed", {"job_id": job_id, "error": str(exc)})
        log_lines.append(f"{datetime.utcnow().isoformat()}Z status=FAILED error={exc}")
    finally:
        try:
            logs_uri = _write_job_log("xgen", job_id, log_lines)
            if job:
                job.logs_uri = logs_uri
                db.add(job)
                db.commit()
        except Exception:
            pass
        db.close()


@celery_app.task(bind=True, max_retries=3, retry_backoff=True)
def run_xmimic_job(self, job_id: str):
    db = SessionLocal()
    job = None
    log_lines = [f"XMimic job {job_id} started at {datetime.utcnow().isoformat()}Z"]
    try:
        job = db.query(models.XMimicJob).filter(models.XMimicJob.id == job_id).first()
        if not job:
            return
        for stage in XMIMIC_STAGES:
            _update_job_status(db, job, stage)
            log_lines.append(f"{datetime.utcnow().isoformat()}Z stage={stage}")
            sleep(0.5)
        _update_job_status(db, job, "COMPLETED")
        log_lines.append(f"{datetime.utcnow().isoformat()}Z status=COMPLETED")
        try:
            checkpoint = {
                "synthetic": True,
                "xmimic_job_id": str(job.id),
                "dataset_id": str(job.dataset_id),
                "mode": job.mode,
                "created_at": datetime.utcnow().isoformat() + "Z",
            }
            ckpt_key = f"policies/{job.id}/checkpoint.json"
            upload_bytes(
                ckpt_key,
                json.dumps(checkpoint, indent=2).encode("utf-8"),
                content_type="application/json",
            )
            policy = crud.create_policy(
                db,
                job.id,
                checkpoint_uri=ckpt_key,
                metadata_json={"mode": job.mode, "synthetic": True},
            )
            crud.create_eval_run(
                db,
                policy.id,
                env_task="cargo_pickup_v0",
                sr=0.0,
                gsr=0.0,
                eo=None,
                eh=None,
                report_uri=None,
                videos_uri=None,
            )
        except Exception:
            pass
        try:
            dataset = job.dataset
            quality = evaluate_clip(has_contact_graph=True, joint_limits_ok=True, object_spikes_ok=True)
            crud.create_quality_score(
                db,
                entity_type="dataset",
                entity_id=dataset.id,
                score=quality.score,
                breakdown_json=quality.breakdown,
                validator_status="pending",
            )
        except Exception:
            pass
    except Exception as exc:
        if job and self.request.retries < self.max_retries:
            _update_job_status(db, job, "RETRYING", error=str(exc))
            log_lines.append(f"{datetime.utcnow().isoformat()}Z status=RETRYING error={exc}")
            send_alert(
                "xmimic_retry",
                {"job_id": job_id, "error": str(exc), "retries": self.request.retries},
            )
            raise self.retry(exc=exc)
        if job:
            _update_job_status(db, job, "FAILED", error=str(exc))
            send_alert("xmimic_failed", {"job_id": job_id, "error": str(exc)})
        log_lines.append(f"{datetime.utcnow().isoformat()}Z status=FAILED error={exc}")
    finally:
        try:
            logs_uri = _write_job_log("xmimic", job_id, log_lines)
            if job:
                job.logs_uri = logs_uri
                db.add(job)
                db.commit()
        except Exception:
            pass
        db.close()
