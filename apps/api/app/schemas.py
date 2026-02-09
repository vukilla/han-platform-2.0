from datetime import datetime
from typing import Any, List, Optional
from uuid import UUID

from pydantic import BaseModel


class UserCreate(BaseModel):
    email: str
    name: Optional[str] = None


class UserOut(BaseModel):
    id: UUID
    email: str
    name: Optional[str] = None

    class Config:
        from_attributes = True


class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = None


class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


class ProjectOut(BaseModel):
    id: UUID
    name: str
    description: Optional[str] = None
    owner_id: Optional[UUID] = None

    class Config:
        from_attributes = True


class DemoCreate(BaseModel):
    project_id: UUID
    robot_model: Optional[str] = None
    object_id: Optional[str] = None


class DemoUpdate(BaseModel):
    video_uri: Optional[str] = None
    fps: Optional[float] = None
    duration: Optional[float] = None
    robot_model: Optional[str] = None
    object_id: Optional[str] = None
    status: Optional[str] = None


class DemoOut(BaseModel):
    id: UUID
    project_id: UUID
    uploader_id: Optional[UUID] = None
    video_uri: Optional[str] = None
    fps: Optional[float] = None
    duration: Optional[float] = None
    robot_model: Optional[str] = None
    object_id: Optional[str] = None
    status: str

    class Config:
        from_attributes = True


class DemoAnnotationCreate(BaseModel):
    ts_contact_start: float
    ts_contact_end: float
    anchor_type: str
    key_bodies: Optional[List[str]] = None
    notes: Optional[str] = None


class DemoAnnotationOut(BaseModel):
    demo_id: UUID
    ts_contact_start: float
    ts_contact_end: float
    anchor_type: str
    key_bodies: Optional[List[str]] = None
    notes: Optional[str] = None

    class Config:
        from_attributes = True


class XGenJobCreate(BaseModel):
    params_json: Optional[dict[str, Any]] = None
    idempotency_key: Optional[str] = None


class XGenJobOut(BaseModel):
    id: UUID
    demo_id: UUID
    status: str
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    params_json: Optional[dict[str, Any]] = None
    logs_uri: Optional[str] = None
    error: Optional[str] = None
    idempotency_key: Optional[str] = None

    class Config:
        from_attributes = True


class DatasetOut(BaseModel):
    id: UUID
    project_id: UUID
    source_demo_id: Optional[UUID] = None
    version: int
    status: str
    summary_json: Optional[dict[str, Any]] = None

    class Config:
        from_attributes = True


class DatasetClipOut(BaseModel):
    clip_id: UUID
    dataset_id: UUID
    uri_npz: str
    uri_preview_mp4: Optional[str] = None
    augmentation_tags: Optional[List[str]] = None
    stats_json: Optional[dict[str, Any]] = None

    class Config:
        from_attributes = True


class XMimicJobCreate(BaseModel):
    mode: str = "nep"
    params_json: Optional[dict[str, Any]] = None
    idempotency_key: Optional[str] = None


class XMimicJobOut(BaseModel):
    id: UUID
    dataset_id: UUID
    mode: str
    status: str
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    params_json: Optional[dict[str, Any]] = None
    logs_uri: Optional[str] = None
    error: Optional[str] = None
    idempotency_key: Optional[str] = None

    class Config:
        from_attributes = True


class PolicyOut(BaseModel):
    id: UUID
    xmimic_job_id: UUID
    checkpoint_uri: str
    exported_at: Optional[datetime] = None
    metadata_json: Optional[dict[str, Any]] = None

    class Config:
        from_attributes = True


class EvalRunOut(BaseModel):
    id: UUID
    policy_id: UUID
    env_task: str
    sr: Optional[float] = None
    gsr: Optional[float] = None
    eo: Optional[float] = None
    eh: Optional[float] = None
    report_uri: Optional[str] = None
    videos_uri: Optional[str] = None

    class Config:
        from_attributes = True


class QualityScoreOut(BaseModel):
    id: UUID
    entity_type: str
    entity_id: UUID
    score: Optional[float] = None
    breakdown_json: Optional[dict[str, Any]] = None
    validator_status: Optional[str] = None
    validator_notes: Optional[str] = None
    validator_id: Optional[UUID] = None
    validated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class QualityReviewIn(BaseModel):
    status: str
    notes: Optional[str] = None
    validator_id: Optional[UUID] = None


class RewardEventOut(BaseModel):
    id: UUID
    user_id: UUID
    entity_type: str
    entity_id: UUID
    points: int
    reason: str
    created_at: datetime

    class Config:
        from_attributes = True


class JobFailureOut(BaseModel):
    job_type: str
    id: UUID
    status: str
    error: Optional[str] = None
    logs_uri: Optional[str] = None
    demo_id: Optional[UUID] = None
    dataset_id: Optional[UUID] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
