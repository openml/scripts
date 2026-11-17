import os
import typing
from functools import partial
from urllib.parse import urlparse, urlunparse

import urllib3
from botocore.config import Config
from dotenv import load_dotenv
import boto3
from loguru import logger

if typing.TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client


def strip_subpath_before_sign(request, subpath, **_):
    """Fixing SignatureDoesNotMatch error when MinIO is hosted on SubPath"""
    parsed = urlparse(request.url)
    if parsed.path.startswith(subpath):
        new_path = parsed.path[len(subpath):] or "/"
        request.url = urlunparse(parsed._replace(path=new_path))

def restore_subpath_before_send(request, subpath, **_):
    """Fixing SignatureDoesNotMatch error when MinIO is hosted on SubPath"""
    parsed = urlparse(request.url)
    if not parsed.path.startswith(subpath):
        clean_path = parsed.path if parsed.path.startswith("/") else f"/{parsed.path}"
        request.url = urlunparse(parsed._replace(path=f"{subpath}{clean_path}"))


def create_s3_client() -> "S3Client":
    load_dotenv()  # take environment variables from .env.
    server = os.environ.get("S3_SERVER", default="https://www.openml.org/s3/")
    verify = os.environ.get("S3_VERIFY", default="true") != "false"
    if not verify:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    logger.info(f"Creating S3 client with server {server} and verify {verify}")
    client =  boto3.client(
        's3',
        endpoint_url=server,
        aws_access_key_id=os.environ["S3_ACCESS_KEY"],
        aws_secret_access_key=os.environ["S3_SECRET_KEY"],
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
        verify=verify,
    )
    subpath = os.environ.get("S3_SUBPATH")
    if subpath:
        client.meta.events.register("before-sign.s3", partial(strip_subpath_before_sign, subpath=subpath))
        client.meta.events.register("before-send.s3", partial(restore_subpath_before_send, subpath=subpath))
    return client