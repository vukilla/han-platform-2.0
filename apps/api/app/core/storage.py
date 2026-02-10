from pathlib import Path
from typing import Optional
import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

from app.core.config import get_settings


def _normalize_key(uri: str) -> str:
    """Return an object-storage key from either a raw key or an s3:// URI."""
    if uri.startswith("s3://"):
        rest = uri[len("s3://") :]
        parts = rest.split("/", 1)
        if len(parts) == 2:
            return parts[1]
    return uri


def get_s3_client(*, endpoint_url: str | None = None):
    settings = get_settings()
    return boto3.client(
        "s3",
        endpoint_url=endpoint_url or settings.s3_endpoint,
        aws_access_key_id=settings.s3_access_key,
        aws_secret_access_key=settings.s3_secret_key,
        region_name=settings.s3_region,
        config=Config(s3={"addressing_style": "path"}),
        use_ssl=settings.s3_secure,
    )


def ensure_bucket_exists():
    settings = get_settings()
    client = get_s3_client()
    try:
        client.head_bucket(Bucket=settings.s3_bucket)
    except ClientError:
        client.create_bucket(Bucket=settings.s3_bucket)


def create_presigned_put(key: str, content_type: Optional[str] = None, expires_in: int = 3600):
    settings = get_settings()
    client = get_s3_client(endpoint_url=settings.s3_public_endpoint or settings.s3_endpoint)
    params = {"Bucket": settings.s3_bucket, "Key": key}
    if content_type:
        params["ContentType"] = content_type
    return client.generate_presigned_url("put_object", Params=params, ExpiresIn=expires_in)


def create_presigned_get(key: str, expires_in: int = 3600):
    settings = get_settings()
    client = get_s3_client(endpoint_url=settings.s3_public_endpoint or settings.s3_endpoint)
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": settings.s3_bucket, "Key": key},
        ExpiresIn=expires_in,
    )


def upload_text(key: str, text: str, content_type: str = "text/plain") -> str:
    settings = get_settings()
    client = get_s3_client()
    client.put_object(Bucket=settings.s3_bucket, Key=key, Body=text.encode("utf-8"), ContentType=content_type)
    return f"s3://{settings.s3_bucket}/{key}"


def upload_bytes(key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
    """Upload raw bytes to object storage and return the storage key."""
    settings = get_settings()
    client = get_s3_client()
    client.put_object(Bucket=settings.s3_bucket, Key=key, Body=data, ContentType=content_type)
    return key


def upload_file(key: str, path: str | Path, content_type: str = "application/octet-stream") -> str:
    """Upload a file from disk to object storage and return the storage key."""
    data = Path(path).read_bytes()
    return upload_bytes(key, data, content_type=content_type)


def download_bytes(key_or_uri: str) -> bytes:
    """Download an object from storage and return its bytes."""
    settings = get_settings()
    key = _normalize_key(key_or_uri)
    client = get_s3_client()
    resp = client.get_object(Bucket=settings.s3_bucket, Key=key)
    return resp["Body"].read()


def download_file(key_or_uri: str, dest_path: str | Path) -> Path:
    """Download an object from storage to a local file path."""
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(download_bytes(key_or_uri))
    return dest


def object_exists(key_or_uri: str) -> bool:
    """Return True if the object exists in storage."""
    settings = get_settings()
    key = _normalize_key(key_or_uri)
    client = get_s3_client()
    try:
        client.head_object(Bucket=settings.s3_bucket, Key=key)
        return True
    except ClientError:
        return False
