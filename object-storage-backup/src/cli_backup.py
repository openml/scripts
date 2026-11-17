import argparse
import time
from asyncio import IncompleteReadError
from pathlib import Path

from botocore.exceptions import ReadTimeoutError
from dotenv import load_dotenv
from loguru import logger

from s3_helper import create_s3_client

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", "--bucket",
        type=str,
        required=True,
        help = "The S3 bucket"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Output directory (without the bucket name)"
    )
    parser.add_argument(
        "--ignore-existing-files",
        action="store_true"
    )
    return parser.parse_args()


def main():
    load_dotenv()
    args = parse_args()
    s3_client = create_s3_client()

    bucket = args.bucket
    output_dir = args.output / bucket
    ignore_existing = args.ignore_existing_files
    logger.info(f"From {bucket} to {output_dir}.")

    output_dir.mkdir(parents=True, exist_ok=True)
    if not output_dir.is_dir() and next(output_dir.iterdir(), None) is None:
        logger.error(f"{output_dir} is not empty or not a directory")
        exit(1)

    paginator = s3_client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket)
    download_count = 0
    logger.info("Connecting to S3...")
    for i, page in enumerate(pages):
        if i == 0:
            logger.info("Connection established")
        logger.debug(f"Page: {page.get('Name')}...")
        if "Contents" not in page:
            logger.info(f"Empty page: {page.get('Name')}")
            break
        for obj in page["Contents"]:
            s3_key = obj["Key"]
            if s3_key.endswith("/"):
                continue

            output_path = output_dir / s3_key
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if output_path.exists() and ignore_existing:
                logger.debug(f"Ignoring existing file {output_path}")
                continue

            logger.debug(f"Downloading {s3_key} to {output_path}")
            # s3_client.download_file(bucket, s3_key, output_path) does not work for MinIO, throwing
            # a KeyError: 'ContentLength'. So manually getting and writing the objects
            response = s3_client.get_object(Bucket=bucket, Key=s3_key)
            with open(output_path, "wb") as output_file:
                attempt = 0
                try:
                    with response["Body"] as input_stream:
                        while True:
                            chunk = input_stream.read(8 * 1024 * 1024)
                            if chunk:
                                output_file.write(chunk)
                            else:
                                break
                except (IncompleteReadError, ReadTimeoutError) as e:
                    attempt += 1
                    if attempt > 3:
                        raise RuntimeError(f"Failed to download {s3_key} to {output_path}") from e
                    logger.warning(f"Exception while downloading {s3_key} to {output_path}: {e}")
                    time.sleep(2 ** attempt)
            download_count += 1
            if download_count % 100 == 0:
                logger.info(f"Downloaded {download_count} files... Last file {output_path}")
    logger.info(f"Done! Downloaded {download_count} files to {output_dir}")


if __name__ == '__main__':
    main()
