import os
import boto3
from botocore.exceptions import ClientError
from fastapi import UploadFile
import uuid

# Load from environment variables (assuming you are using dotenv/os.environ)
S3_ENDPOINT = os.getenv("S3_ENDPOINT_URL", "http://localhost:9000")
ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID", "dev_access_key")
SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "dev_secret_key")
BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "edgecase-videos")

s3_client = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    region_name=os.getenv("AWS_REGION", "eu-west-2")
)

def ensure_bucket_exists():
    """Creates the MinIO bucket on startup if it doesn't exist."""
    try:
        s3_client.head_bucket(Bucket=BUCKET_NAME)
    except ClientError:
        s3_client.create_bucket(Bucket=BUCKET_NAME)

def upload_video_to_s3(file: UploadFile) -> str:
    """
    Uploads a file to S3/MinIO and returns the generated s3_key.
    """
    file_extension = file.filename.split(".")[-1] if "." in file.filename else "mp4"
    s3_key = f"uploads/{uuid.uuid4()}.{file_extension}"
    
    s3_client.upload_fileobj(
        file.file,
        BUCKET_NAME,
        s3_key,
        ExtraArgs={"ContentType": file.content_type}
    )
    return s3_key


def get_presigned_video_url(s3_key: str, expires_in: int = 3600) -> str:
    return s3_client.generate_presigned_url(
        "get_object",
        Params={"Bucket": BUCKET_NAME, "Key": s3_key},
        ExpiresIn=expires_in,
    )