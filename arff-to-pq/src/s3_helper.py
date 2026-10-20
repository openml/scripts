import os
import typing

from botocore.config import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv
import boto3

if typing.TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client


def create_s3_client() -> S3Client:
    load_dotenv()  # take environment variables from .env.
    return boto3.client(
        's3',
        endpoint_url=os.environ.get("S3_SERVER", default="https://www.openml.org/s3/"),
        aws_access_key_id=os.environ["S3_ACCESS_KEY"],
        aws_secret_access_key=os.environ["S3_SECRET_KEY"],
        config=Config(s3={'addressing_style': 'path'}),
    )


def file_exists(client: S3Client, bucket_name: str, file_path: str) -> bool:
    try:
        client.head_object(Bucket=bucket_name, Key=file_path)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        raise

def _strtobool(input_: str) -> bool:
    return {"yes": True, "y": True, "true": True, "no": False, "n": False, "false": False}[
        input_.lower()]
