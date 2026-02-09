from datetime import datetime
from sqlalchemy.orm import Session

from app import models


# Users

def get_user_by_email(db: Session, email: str):
    return db.query(models.User).filter(models.User.email == email).first()


def get_user_by_id(db: Session, user_id):
    return db.query(models.User).filter(models.User.id == user_id).first()


def create_user(db: Session, email: str, name: str | None = None):
    user = models.User(email=email, name=name)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


# Projects

def create_project(db: Session, name: str, description: str | None, owner_id=None):
    project = models.Project(name=name, description=description, owner_id=owner_id)
    db.add(project)
    db.commit()
    db.refresh(project)
    return project


def list_projects(db: Session):
    return db.query(models.Project).order_by(models.Project.created_at.desc()).all()


def get_project(db: Session, project_id):
    return db.query(models.Project).filter(models.Project.id == project_id).first()


def update_project(db: Session, project_id, payload):
    project = get_project(db, project_id)
    if not project:
        return None
    for key, value in payload.items():
        if value is not None:
            setattr(project, key, value)
    db.add(project)
    db.commit()
    db.refresh(project)
    return project


def delete_project(db: Session, project_id):
    project = get_project(db, project_id)
    if not project:
        return False
    db.delete(project)
    db.commit()
    return True

# Demos

def create_demo(db: Session, project_id, uploader_id=None, robot_model=None, object_id=None):
    demo = models.Demo(
        project_id=project_id,
        uploader_id=uploader_id,
        robot_model=robot_model,
        object_id=object_id,
        status="CREATED",
    )
    db.add(demo)
    db.commit()
    db.refresh(demo)
    return demo


def list_demos(db: Session, project_id=None):
    query = db.query(models.Demo)
    if project_id:
        query = query.filter(models.Demo.project_id == project_id)
    return query.order_by(models.Demo.created_at.desc()).all()


def get_demo(db: Session, demo_id):
    return db.query(models.Demo).filter(models.Demo.id == demo_id).first()


def update_demo(db: Session, demo_id, payload):
    demo = get_demo(db, demo_id)
    if not demo:
        return None
    for key, value in payload.items():
        if value is not None:
            setattr(demo, key, value)
    db.add(demo)
    db.commit()
    db.refresh(demo)
    return demo


def delete_demo(db: Session, demo_id):
    demo = get_demo(db, demo_id)
    if not demo:
        return False
    db.delete(demo)
    db.commit()
    return True


# Annotations

def upsert_demo_annotation(db: Session, demo_id, payload):
    annotation = db.query(models.DemoAnnotation).filter(models.DemoAnnotation.demo_id == demo_id).first()
    if annotation is None:
        annotation = models.DemoAnnotation(demo_id=demo_id, **payload)
        db.add(annotation)
    else:
        for key, value in payload.items():
            setattr(annotation, key, value)
    db.commit()
    db.refresh(annotation)
    return annotation


# XGen Jobs

def create_xgen_job(db: Session, demo_id, params_json=None, idempotency_key=None):
    if idempotency_key:
        existing = (
            db.query(models.XGenJob)
            .filter(models.XGenJob.demo_id == demo_id, models.XGenJob.idempotency_key == idempotency_key)
            .first()
        )
        if existing:
            return existing
    job = models.XGenJob(
        demo_id=demo_id,
        params_json=params_json,
        status="QUEUED",
        idempotency_key=idempotency_key,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def get_xgen_job(db: Session, job_id):
    return db.query(models.XGenJob).filter(models.XGenJob.id == job_id).first()


def list_xgen_jobs(db: Session, demo_id=None, status=None, statuses=None):
    query = db.query(models.XGenJob)
    if demo_id:
        query = query.filter(models.XGenJob.demo_id == demo_id)
    if status:
        query = query.filter(models.XGenJob.status == status)
    if statuses:
        query = query.filter(models.XGenJob.status.in_(statuses))
    return query.order_by(models.XGenJob.created_at.desc()).all()


# Datasets

def _next_dataset_version(db: Session, project_id):
    latest = (
        db.query(models.Dataset)
        .filter(models.Dataset.project_id == project_id)
        .order_by(models.Dataset.version.desc())
        .first()
    )
    return (latest.version if latest else 0) + 1


def create_dataset(db: Session, project_id, source_demo_id=None, summary_json=None, status="CREATED", version=None):
    dataset = models.Dataset(
        project_id=project_id,
        source_demo_id=source_demo_id,
        status=status,
        summary_json=summary_json,
        version=version or _next_dataset_version(db, project_id),
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    return dataset


def get_dataset(db: Session, dataset_id):
    return db.query(models.Dataset).filter(models.Dataset.id == dataset_id).first()


def list_datasets(db: Session, project_id=None):
    query = db.query(models.Dataset)
    if project_id:
        query = query.filter(models.Dataset.project_id == project_id)
    return query.order_by(models.Dataset.created_at.desc()).all()


def list_dataset_clips(db: Session, dataset_id):
    return db.query(models.DatasetClip).filter(models.DatasetClip.dataset_id == dataset_id).all()


# XMimic Jobs

def create_xmimic_job(db: Session, dataset_id, mode, params_json=None, idempotency_key=None):
    if idempotency_key:
        existing = (
            db.query(models.XMimicJob)
            .filter(models.XMimicJob.dataset_id == dataset_id, models.XMimicJob.idempotency_key == idempotency_key)
            .first()
        )
        if existing:
            return existing
    job = models.XMimicJob(
        dataset_id=dataset_id,
        mode=mode,
        params_json=params_json,
        status="QUEUED",
        idempotency_key=idempotency_key,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def get_xmimic_job(db: Session, job_id):
    return db.query(models.XMimicJob).filter(models.XMimicJob.id == job_id).first()


def list_xmimic_jobs(db: Session, dataset_id=None, status=None, statuses=None):
    query = db.query(models.XMimicJob)
    if dataset_id:
        query = query.filter(models.XMimicJob.dataset_id == dataset_id)
    if status:
        query = query.filter(models.XMimicJob.status == status)
    if statuses:
        query = query.filter(models.XMimicJob.status.in_(statuses))
    return query.order_by(models.XMimicJob.created_at.desc()).all()


# Policies

def create_policy(db: Session, xmimic_job_id, checkpoint_uri, metadata_json=None):
    policy = models.Policy(xmimic_job_id=xmimic_job_id, checkpoint_uri=checkpoint_uri, metadata_json=metadata_json)
    db.add(policy)
    db.commit()
    db.refresh(policy)
    return policy


def get_policy(db: Session, policy_id):
    return db.query(models.Policy).filter(models.Policy.id == policy_id).first()


def list_policies(db: Session, xmimic_job_id=None):
    query = db.query(models.Policy)
    if xmimic_job_id:
        query = query.filter(models.Policy.xmimic_job_id == xmimic_job_id)
    return query.order_by(models.Policy.created_at.desc()).all()


# Eval

def create_eval_run(db: Session, policy_id, env_task, report_uri=None, videos_uri=None, sr=None, gsr=None, eo=None, eh=None):
    eval_run = models.EvalRun(
        policy_id=policy_id,
        env_task=env_task,
        report_uri=report_uri,
        videos_uri=videos_uri,
        sr=sr,
        gsr=gsr,
        eo=eo,
        eh=eh,
    )
    db.add(eval_run)
    db.commit()
    db.refresh(eval_run)
    return eval_run


def get_eval_run(db: Session, eval_id):
    return db.query(models.EvalRun).filter(models.EvalRun.id == eval_id).first()


def list_eval_runs(db: Session, policy_id=None):
    query = db.query(models.EvalRun)
    if policy_id:
        query = query.filter(models.EvalRun.policy_id == policy_id)
    return query.order_by(models.EvalRun.created_at.desc()).all()


# Quality

def get_quality_score(db: Session, entity_type, entity_id):
    return (
        db.query(models.QualityScore)
        .filter(models.QualityScore.entity_type == entity_type, models.QualityScore.entity_id == entity_id)
        .first()
    )


def create_quality_score(db: Session, entity_type, entity_id, score=None, breakdown_json=None, validator_status=None):
    quality = models.QualityScore(
        entity_type=entity_type,
        entity_id=entity_id,
        score=score,
        breakdown_json=breakdown_json,
        validator_status=validator_status,
    )
    db.add(quality)
    db.commit()
    db.refresh(quality)
    return quality


def review_quality_score(db: Session, entity_type, entity_id, status, notes=None, validator_id=None):
    quality = get_quality_score(db, entity_type, entity_id)
    if not quality:
        quality = models.QualityScore(entity_type=entity_type, entity_id=entity_id)
    quality.validator_status = status
    quality.validator_notes = notes
    quality.validator_id = validator_id
    quality.validated_at = datetime.utcnow()
    db.add(quality)
    db.commit()
    db.refresh(quality)
    return quality


# Rewards

def list_reward_events_for_user(db: Session, user_id):
    return db.query(models.RewardEvent).filter(models.RewardEvent.user_id == user_id).order_by(models.RewardEvent.created_at.desc()).all()
