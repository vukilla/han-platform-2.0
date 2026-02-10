from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    environment: str = "dev"
    database_url: str = "postgresql+psycopg://han:han@db:5432/han"
    redis_url: str = "redis://redis:6379/0"

    # CORS: required for the browser-based Web UI to call the API.
    # Defaults are intentionally permissive for local development; tighten for production.
    cors_allow_origins: str = "http://localhost:3000,http://127.0.0.1:3000"
    cors_allow_origin_regex: str = r"^https?://(localhost|127\\.0\\.0\\.1|192\\.168\\.\\d+\\.\\d+)(:\\d+)?$"

    s3_endpoint: str = "http://minio:9000"
    # Endpoint used for presigned URLs handed to a browser/client.
    # In docker-compose this should generally be a host-reachable address like http://localhost:9000.
    s3_public_endpoint: str = "http://localhost:9000"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"
    s3_bucket: str = "humanx-dev"
    s3_region: str = "us-east-1"
    s3_secure: bool = False

    jwt_secret: str = "dev-secret"

    celery_cpu_queue: str = "cpu"
    celery_gpu_queue: str = "gpu"
    celery_default_queue: str = "cpu"
    celery_max_queue_cpu: int = 100
    celery_max_queue_gpu: int = 20
    celery_prefetch_multiplier: int = 1
    celery_task_acks_late: bool = True
    celery_reject_on_worker_lost: bool = True
    alert_webhook_url: str | None = None

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
