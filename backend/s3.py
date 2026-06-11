import os
import uuid

import boto3
from botocore.exceptions import ClientError
from fastapi import UploadFile

AWS_REGION = os.getenv("AWS_REGION", "eu-west-2")
ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_ENDPOINT = os.getenv("S3_ENDPOINT_URL")

if not BUCKET_NAME:
    raise ValueError("S3_BUCKET_NAME environment variable is required")

_client_kwargs: dict = {"region_name": AWS_REGION}
if ACCESS_KEY and SECRET_KEY:
    _client_kwargs["aws_access_key_id"] = ACCESS_KEY
    _client_kwargs["aws_secret_access_key"] = SECRET_KEY
if S3_ENDPOINT:
    _client_kwargs["endpoint_url"] = S3_ENDPOINT

s3_client = boto3.client("s3", **_client_kwargs)


def ensure_bucket_exists() -> None:
    """Verify the configured bucket is reachable (creates bucket only for local MinIO)."""
    try:
        s3_client.head_bucket(Bucket=BUCKET_NAME)
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code", "")
        if S3_ENDPOINT and error_code in ("404", "NoSuchBucket", "403"):
            s3_client.create_bucket(Bucket=BUCKET_NAME)
            return
        raise RuntimeError(
            f"S3 bucket '{BUCKET_NAME}' is not accessible in {AWS_REGION}. "
            "Check AWS credentials, IAM permissions, region, and bucket name."
        ) from exc


def upload_video_to_s3(file: UploadFile) -> str:
    file_extension = file.filename.rsplit(".", 1)[-1].lower() if file.filename and "." in file.filename else "mp4"
    s3_key = f"uploads/{uuid.uuid4()}.{file_extension}"

    content_type = file.content_type
    if not content_type or content_type == "application/octet-stream":
        content_type = f"video/{file_extension}" if file_extension in ("mp4", "webm", "mov") else "video/mp4"

    s3_client.upload_fileobj(
        file.file,
        BUCKET_NAME,
        s3_key,
        ExtraArgs={"ContentType": content_type},
    )
    return s3_key


def get_presigned_video_url(s3_key: str, expires_in: int = 3600) -> str:
    return s3_client.generate_presigned_url(
        "get_object",
        Params={"Bucket": BUCKET_NAME, "Key": s3_key},
        ExpiresIn=expires_in,
    )


def delete_video_from_s3(s3_key: str) -> None:
    s3_client.delete_object(Bucket=BUCKET_NAME, Key=s3_key)
