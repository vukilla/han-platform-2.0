from datetime import datetime
from time import sleep
from celery import Celery
from kombu import Queue

from app.core.config import get_settings
from app.core.alerts import send_alert
from app.core.storage import upload_text
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


@celery_app.task(bind=True, max_retries=3, retry_backoff=True)
def run_xgen_job(self, job_id: str):
    db = SessionLocal()
    job = None
    log_lines = [f"XGen job {job_id} started at {datetime.utcnow().isoformat()}Z"]
    try:
        job = db.query(models.XGenJob).filter(models.XGenJob.id == job_id).first()
        if not job:
            return
        for stage in XGEN_STAGES:
            _update_job_status(db, job, stage)
            log_lines.append(f"{datetime.utcnow().isoformat()}Z stage={stage}")
            sleep(0.5)
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
