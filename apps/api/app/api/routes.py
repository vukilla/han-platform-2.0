from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
import redis
from sqlalchemy import text
from sqlalchemy.orm import Session

from app import crud, models, schemas
from app.core.auth import create_access_token, get_current_user
from app.core.config import get_settings
from app.core.queue import is_queue_full, get_queue_depth
from app.core.storage import create_presigned_put, create_presigned_get, get_s3_client
from app.db import get_db
from app.worker import run_xgen_job, run_xmimic_job

router = APIRouter()
settings = get_settings()

def _normalize_s3_key(uri: str) -> str:
    """Return an object-storage key from either a raw key or an s3:// URI."""
    if uri.startswith("s3://"):
        rest = uri[len("s3://") :]
        parts = rest.split("/", 1)
        if len(parts) == 2:
            return parts[1]
    return uri


def _presign_maybe(uri: str | None) -> str | None:
    if not uri:
        return None
    if uri.startswith("http://") or uri.startswith("https://"):
        return uri
    key = _normalize_s3_key(uri)
    # If it's some other scheme, don't attempt to presign it.
    if "://" in key:
        return uri
    return create_presigned_get(key)


class AuthLoginRequest(BaseModel):
    email: str
    name: str | None = None


class AuthLoginResponse(BaseModel):
    user: schemas.UserOut
    token: str


@router.post("/auth/login", response_model=AuthLoginResponse)
def login(payload: AuthLoginRequest, db: Session = Depends(get_db)):
    user = crud.get_user_by_email(db, payload.email)
    if user is None:
        user = crud.create_user(db, payload.email, payload.name)
    token = create_access_token(user.id, user.email)
    return AuthLoginResponse(user=user, token=token)


@router.get("/me", response_model=schemas.UserOut)
def me(current_user: models.User = Depends(get_current_user)):
    return current_user


@router.post("/projects", response_model=schemas.ProjectOut)
def create_project(
    payload: schemas.ProjectCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    project = crud.create_project(db, payload.name, payload.description, owner_id=current_user.id)
    return project


@router.get("/projects", response_model=list[schemas.ProjectOut])
def list_projects(db: Session = Depends(get_db)):
    return crud.list_projects(db)


@router.get("/projects/{project_id}", response_model=schemas.ProjectOut)
def get_project(project_id: UUID, db: Session = Depends(get_db)):
    project = crud.get_project(db, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


@router.patch("/projects/{project_id}", response_model=schemas.ProjectOut)
def update_project(project_id: UUID, payload: schemas.ProjectUpdate, db: Session = Depends(get_db)):
    project = crud.update_project(db, project_id, payload.model_dump())
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


@router.delete("/projects/{project_id}")
def delete_project(project_id: UUID, db: Session = Depends(get_db)):
    ok = crud.delete_project(db, project_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"status": "deleted"}

@router.post("/demos", response_model=schemas.DemoOut)
def create_demo(
    payload: schemas.DemoCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    demo = crud.create_demo(
        db,
        payload.project_id,
        uploader_id=current_user.id,
        robot_model=payload.robot_model,
        object_id=payload.object_id,
    )
    return demo


@router.get("/demos", response_model=list[schemas.DemoOut])
def list_demos(project_id: UUID | None = None, db: Session = Depends(get_db)):
    return crud.list_demos(db, project_id=project_id)


class UploadUrlResponse(BaseModel):
    upload_url: str
    video_uri: str


@router.post("/demos/{demo_id}/upload-url", response_model=UploadUrlResponse)
def create_demo_upload_url(demo_id: UUID, db: Session = Depends(get_db)):
    demo = crud.get_demo(db, demo_id)
    if not demo:
        raise HTTPException(status_code=404, detail="Demo not found")
    key = f"demos/{demo_id}/raw.mp4"
    url = create_presigned_put(key, content_type="video/mp4")
    demo.video_uri = key
    demo.status = "UPLOADED"
    db.add(demo)
    db.commit()
    return UploadUrlResponse(upload_url=url, video_uri=key)


@router.post("/demos/{demo_id}/annotations", response_model=schemas.DemoAnnotationOut)
def annotate_demo(demo_id: UUID, payload: schemas.DemoAnnotationCreate, db: Session = Depends(get_db)):
    demo = crud.get_demo(db, demo_id)
    if not demo:
        raise HTTPException(status_code=404, detail="Demo not found")
    annotation = crud.upsert_demo_annotation(db, demo_id, payload.model_dump())
    return annotation


@router.get("/demos/{demo_id}", response_model=schemas.DemoOut)
def get_demo(demo_id: UUID, db: Session = Depends(get_db)):
    demo = crud.get_demo(db, demo_id)
    if not demo:
        raise HTTPException(status_code=404, detail="Demo not found")
    return demo


@router.patch("/demos/{demo_id}", response_model=schemas.DemoOut)
def update_demo(demo_id: UUID, payload: schemas.DemoUpdate, db: Session = Depends(get_db)):
    demo = crud.update_demo(db, demo_id, payload.model_dump())
    if not demo:
        raise HTTPException(status_code=404, detail="Demo not found")
    return demo


@router.delete("/demos/{demo_id}")
def delete_demo(demo_id: UUID, db: Session = Depends(get_db)):
    ok = crud.delete_demo(db, demo_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Demo not found")
    return {"status": "deleted"}


@router.post("/demos/{demo_id}/xgen/run", response_model=schemas.XGenJobOut)
def run_xgen(
    demo_id: UUID,
    payload: schemas.XGenJobCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    demo = crud.get_demo(db, demo_id)
    if not demo:
        raise HTTPException(status_code=404, detail="Demo not found")
    job = crud.create_xgen_job(
        db,
        demo_id,
        params_json=payload.params_json,
        idempotency_key=payload.idempotency_key,
    )
    requires_gpu = bool((payload.params_json or {}).get("requires_gpu", False))
    queue_name = settings.celery_gpu_queue if requires_gpu else settings.celery_cpu_queue
    max_depth = settings.celery_max_queue_gpu if requires_gpu else settings.celery_max_queue_cpu
    if is_queue_full(queue_name, max_depth):
        raise HTTPException(status_code=429, detail=f"Queue {queue_name} is at capacity")
    run_xgen_job.apply_async(args=[str(job.id)], queue=queue_name)
    return job


@router.get("/xgen/jobs", response_model=list[schemas.XGenJobOut])
def list_xgen_jobs(demo_id: UUID | None = None, db: Session = Depends(get_db)):
    jobs = crud.list_xgen_jobs(db, demo_id=demo_id)
    out: list[schemas.XGenJobOut] = []
    for job in jobs:
        job_out = schemas.XGenJobOut.model_validate(job)
        job_out.logs_uri = _presign_maybe(job_out.logs_uri)
        out.append(job_out)
    return out


@router.get("/xgen/jobs/{job_id}", response_model=schemas.XGenJobOut)
def get_xgen_job(job_id: UUID, db: Session = Depends(get_db)):
    job = crud.get_xgen_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    out = schemas.XGenJobOut.model_validate(job)
    out.logs_uri = _presign_maybe(out.logs_uri)
    return out


@router.get("/datasets/{dataset_id}", response_model=schemas.DatasetOut)
def get_dataset(dataset_id: UUID, db: Session = Depends(get_db)):
    dataset = crud.get_dataset(db, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset


@router.get("/datasets", response_model=list[schemas.DatasetOut])
def list_datasets(project_id: UUID | None = None, db: Session = Depends(get_db)):
    return crud.list_datasets(db, project_id=project_id)


@router.get("/datasets/{dataset_id}/clips", response_model=list[schemas.DatasetClipOut])
def get_dataset_clips(dataset_id: UUID, db: Session = Depends(get_db)):
    clips = crud.list_dataset_clips(db, dataset_id)
    out: list[schemas.DatasetClipOut] = []
    for clip in clips:
        out.append(
            schemas.DatasetClipOut(
                clip_id=clip.clip_id,
                dataset_id=clip.dataset_id,
                uri_npz=create_presigned_get(clip.uri_npz),
                uri_preview_mp4=create_presigned_get(clip.uri_preview_mp4) if clip.uri_preview_mp4 else None,
                augmentation_tags=clip.augmentation_tags,
                stats_json=clip.stats_json,
            )
        )
    return out


class DownloadUrlResponse(BaseModel):
    download_url: str


@router.get("/datasets/{dataset_id}/download-url", response_model=DownloadUrlResponse)
def dataset_download_url(dataset_id: UUID, db: Session = Depends(get_db)):
    dataset = crud.get_dataset(db, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    key = f"datasets/{dataset_id}/dataset.zip"
    url = create_presigned_get(key)
    return DownloadUrlResponse(download_url=url)


@router.post("/datasets/{dataset_id}/xmimic/run", response_model=schemas.XMimicJobOut)
def run_xmimic(
    dataset_id: UUID,
    payload: schemas.XMimicJobCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    dataset = crud.get_dataset(db, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    job = crud.create_xmimic_job(
        db,
        dataset_id,
        payload.mode,
        payload.params_json,
        idempotency_key=payload.idempotency_key,
    )
    queue_name = settings.celery_gpu_queue
    if is_queue_full(queue_name, settings.celery_max_queue_gpu):
        raise HTTPException(status_code=429, detail=f"Queue {queue_name} is at capacity")
    run_xmimic_job.apply_async(args=[str(job.id)], queue=queue_name)
    return job


@router.get("/xmimic/jobs", response_model=list[schemas.XMimicJobOut])
def list_xmimic_jobs(dataset_id: UUID | None = None, db: Session = Depends(get_db)):
    jobs = crud.list_xmimic_jobs(db, dataset_id=dataset_id)
    out: list[schemas.XMimicJobOut] = []
    for job in jobs:
        job_out = schemas.XMimicJobOut.model_validate(job)
        job_out.logs_uri = _presign_maybe(job_out.logs_uri)
        out.append(job_out)
    return out


@router.get("/xmimic/jobs/{job_id}", response_model=schemas.XMimicJobOut)
def get_xmimic_job(job_id: UUID, db: Session = Depends(get_db)):
    job = crud.get_xmimic_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    out = schemas.XMimicJobOut.model_validate(job)
    out.logs_uri = _presign_maybe(out.logs_uri)
    return out


@router.get("/policies/{policy_id}", response_model=schemas.PolicyOut)
def get_policy(policy_id: UUID, db: Session = Depends(get_db)):
    policy = crud.get_policy(db, policy_id)
    if not policy:
        raise HTTPException(status_code=404, detail="Policy not found")
    out = schemas.PolicyOut.model_validate(policy)
    out.checkpoint_uri = create_presigned_get(_normalize_s3_key(out.checkpoint_uri))
    return out


@router.get("/policies", response_model=list[schemas.PolicyOut])
def list_policies(xmimic_job_id: UUID | None = None, db: Session = Depends(get_db)):
    policies = crud.list_policies(db, xmimic_job_id=xmimic_job_id)
    out: list[schemas.PolicyOut] = []
    for policy in policies:
        policy_out = schemas.PolicyOut.model_validate(policy)
        policy_out.checkpoint_uri = create_presigned_get(_normalize_s3_key(policy_out.checkpoint_uri))
        out.append(policy_out)
    return out


class EvalRunRequest(BaseModel):
    env_task: str


@router.post("/policies/{policy_id}/eval/run", response_model=schemas.EvalRunOut)
def run_eval(policy_id: UUID, payload: EvalRunRequest, db: Session = Depends(get_db)):
    policy = crud.get_policy(db, policy_id)
    if not policy:
        raise HTTPException(status_code=404, detail="Policy not found")
    eval_run = crud.create_eval_run(db, policy_id, payload.env_task)
    return eval_run


@router.get("/eval", response_model=list[schemas.EvalRunOut])
def list_eval(policy_id: UUID | None = None, db: Session = Depends(get_db)):
    return crud.list_eval_runs(db, policy_id=policy_id)


@router.get("/eval/{eval_id}", response_model=schemas.EvalRunOut)
def get_eval(eval_id: UUID, db: Session = Depends(get_db)):
    eval_run = crud.get_eval_run(db, eval_id)
    if not eval_run:
        raise HTTPException(status_code=404, detail="Eval not found")
    return eval_run


@router.get("/quality/{entity_type}/{entity_id}", response_model=schemas.QualityScoreOut)
def get_quality(entity_type: str, entity_id: UUID, db: Session = Depends(get_db)):
    score = crud.get_quality_score(db, entity_type, entity_id)
    if not score:
        raise HTTPException(status_code=404, detail="Quality score not found")
    return score


@router.post("/quality/{entity_type}/{entity_id}/review", response_model=schemas.QualityScoreOut)
def review_quality(
    entity_type: str,
    entity_id: UUID,
    payload: schemas.QualityReviewIn,
    db: Session = Depends(get_db),
):
    return crud.review_quality_score(
        db,
        entity_type,
        entity_id,
        status=payload.status,
        notes=payload.notes,
        validator_id=payload.validator_id,
    )


@router.get("/rewards/me", response_model=list[schemas.RewardEventOut])
def get_rewards_me(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    return crud.list_reward_events_for_user(db, current_user.id)


@router.get("/health")
def health(db: Session = Depends(get_db)):
    checks = {"db": False, "redis": False, "s3": False}
    try:
        db.execute(text("SELECT 1"))
        checks["db"] = True
    except Exception:
        pass
    try:
        redis.Redis.from_url(settings.redis_url).ping()
        checks["redis"] = True
    except Exception:
        pass
    try:
        client = get_s3_client()
        client.head_bucket(Bucket=settings.s3_bucket)
        checks["s3"] = True
    except Exception:
        pass
    checks["cpu_queue_depth"] = get_queue_depth(settings.celery_cpu_queue)
    checks["gpu_queue_depth"] = get_queue_depth(settings.celery_gpu_queue)
    checks["status"] = "ok" if all([checks["db"], checks["redis"], checks["s3"]]) else "degraded"
    return checks


@router.get("/ops/jobs/failed", response_model=list[schemas.JobFailureOut])
def list_failed_jobs(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    statuses = ["FAILED", "RETRYING"]
    xgen_jobs = crud.list_xgen_jobs(db, statuses=statuses)
    xmimic_jobs = crud.list_xmimic_jobs(db, statuses=statuses)
    failures: list[schemas.JobFailureOut] = []
    for job in xgen_jobs:
        failures.append(
            schemas.JobFailureOut(
                job_type="xgen",
                id=job.id,
                status=job.status,
                error=job.error,
                logs_uri=_presign_maybe(job.logs_uri),
                demo_id=job.demo_id,
                started_at=job.started_at,
                finished_at=job.finished_at,
            )
        )
    for job in xmimic_jobs:
        failures.append(
            schemas.JobFailureOut(
                job_type="xmimic",
                id=job.id,
                status=job.status,
                error=job.error,
                logs_uri=_presign_maybe(job.logs_uri),
                dataset_id=job.dataset_id,
                started_at=job.started_at,
                finished_at=job.finished_at,
            )
        )
    return failures
