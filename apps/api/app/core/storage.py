from typing import Optional
import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

from app.core.config import get_settings


def get_s3_client():
    settings = get_settings()
    return boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint,
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
    client = get_s3_client()
    params = {"Bucket": settings.s3_bucket, "Key": key}
    if content_type:
        params["ContentType"] = content_type
    return client.generate_presigned_url("put_object", Params=params, ExpiresIn=expires_in)


def create_presigned_get(key: str, expires_in: int = 3600):
    settings = get_settings()
    client = get_s3_client()
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
