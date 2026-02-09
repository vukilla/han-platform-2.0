import uuid
from datetime import datetime

from sqlalchemy import (
    Column,
    String,
    Float,
    DateTime,
    Integer,
    ForeignKey,
    Boolean,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.orm import relationship

from app.db import Base


class TimestampMixin:
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


class User(Base, TimestampMixin):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=True)


class Project(Base, TimestampMixin):
    __tablename__ = "projects"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    description = Column(String, nullable=True)
    owner_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)

    owner = relationship("User")


class Demo(Base, TimestampMixin):
    __tablename__ = "demos"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False)
    uploader_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    video_uri = Column(String, nullable=True)
    fps = Column(Float, nullable=True)
    duration = Column(Float, nullable=True)
    robot_model = Column(String, nullable=True)
    object_id = Column(String, nullable=True)
    status = Column(String, nullable=False, default="CREATED")

    project = relationship("Project")
    uploader = relationship("User")


class DemoAnnotation(Base, TimestampMixin):
    __tablename__ = "demo_annotations"

    demo_id = Column(UUID(as_uuid=True), ForeignKey("demos.id"), primary_key=True)
    ts_contact_start = Column(Float, nullable=False)
    ts_contact_end = Column(Float, nullable=False)
    anchor_type = Column(String, nullable=False)
    key_bodies = Column(ARRAY(String), nullable=True)
    notes = Column(String, nullable=True)

    demo = relationship("Demo")


class XGenJob(Base, TimestampMixin):
    __tablename__ = "xgen_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    demo_id = Column(UUID(as_uuid=True), ForeignKey("demos.id"), nullable=False)
    idempotency_key = Column(String, nullable=True, index=True)
    status = Column(String, nullable=False, default="QUEUED")
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)
    params_json = Column(JSONB, nullable=True)
    logs_uri = Column(String, nullable=True)
    error = Column(String, nullable=True)

    demo = relationship("Demo")


class Dataset(Base, TimestampMixin):
    __tablename__ = "datasets"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False)
    source_demo_id = Column(UUID(as_uuid=True), ForeignKey("demos.id"), nullable=True)
    version = Column(Integer, nullable=False, default=1)
    status = Column(String, nullable=False, default="CREATED")
    summary_json = Column(JSONB, nullable=True)

    project = relationship("Project")
    source_demo = relationship("Demo")


class DatasetClip(Base, TimestampMixin):
    __tablename__ = "dataset_clips"

    clip_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dataset_id = Column(UUID(as_uuid=True), ForeignKey("datasets.id"), nullable=False, index=True)
    uri_npz = Column(String, nullable=False)
    uri_preview_mp4 = Column(String, nullable=True)
    augmentation_tags = Column(ARRAY(String), nullable=True)
    stats_json = Column(JSONB, nullable=True)

    dataset = relationship("Dataset")


class XMimicJob(Base, TimestampMixin):
    __tablename__ = "xmimic_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dataset_id = Column(UUID(as_uuid=True), ForeignKey("datasets.id"), nullable=False)
    mode = Column(String, nullable=False, default="nep")
    status = Column(String, nullable=False, default="QUEUED")
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)
    idempotency_key = Column(String, nullable=True, index=True)
    params_json = Column(JSONB, nullable=True)
    logs_uri = Column(String, nullable=True)
    error = Column(String, nullable=True)

    dataset = relationship("Dataset")


class Policy(Base, TimestampMixin):
    __tablename__ = "policies"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    xmimic_job_id = Column(UUID(as_uuid=True), ForeignKey("xmimic_jobs.id"), nullable=False)
    checkpoint_uri = Column(String, nullable=False)
    exported_at = Column(DateTime, nullable=True)
    metadata_json = Column(JSONB, nullable=True)

    xmimic_job = relationship("XMimicJob")


class EvalRun(Base, TimestampMixin):
    __tablename__ = "eval_runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    policy_id = Column(UUID(as_uuid=True), ForeignKey("policies.id"), nullable=False)
    env_task = Column(String, nullable=False)
    sr = Column(Float, nullable=True)
    gsr = Column(Float, nullable=True)
    eo = Column(Float, nullable=True)
    eh = Column(Float, nullable=True)
    report_uri = Column(String, nullable=True)
    videos_uri = Column(String, nullable=True)

    policy = relationship("Policy")


class QualityScore(Base, TimestampMixin):
    __tablename__ = "quality_scores"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    entity_type = Column(String, nullable=False)
    entity_id = Column(UUID(as_uuid=True), nullable=False)
    score = Column(Float, nullable=True)
    breakdown_json = Column(JSONB, nullable=True)
    validator_status = Column(String, nullable=True)
    validator_notes = Column(String, nullable=True)
    validator_id = Column(UUID(as_uuid=True), nullable=True)
    validated_at = Column(DateTime, nullable=True)


class RewardEvent(Base, TimestampMixin):
    __tablename__ = "reward_events"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    entity_type = Column(String, nullable=False)
    entity_id = Column(UUID(as_uuid=True), nullable=False)
    points = Column(Integer, nullable=False)
    reason = Column(String, nullable=False)

    user = relationship("User")
