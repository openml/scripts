import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

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
        "--n-threads",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--s3-page-size",
        type=int,
        default=1000
    )
    parser.add_argument(
        "--ignore-existing-files",
        action="store_true"
    )
    return parser.parse_args()


def _s3_keys(s3_client, bucket, page_size):
    paginator = s3_client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, PaginationConfig={"PageSize": page_size})
    logger.info("Reading all s3 keys...")
    for page in pages:
        logger.debug(f"Page: {page.get('Name')}...")
        if "Contents" not in page:
            logger.info(f"Empty page: {page.get('Name')}")
        else:
            for obj in page["Contents"]:
                s3_key = obj["Key"]
                if not s3_key.endswith("/"):
                    yield s3_key


def _download_object(s3_client, bucket, s3_key, output_path):
    logger.debug(f"Downloading {s3_key} to {output_path}...")
    for attempt in range(3):
        try:
            # s3_client.download_file(bucket, s3_key, output_path) does not work for MinIO, throwing
            # a KeyError: 'ContentLength'. So manually getting and writing the objects
            response = s3_client.get_object(Bucket=bucket, Key=s3_key)
            with open(output_path, "wb") as output_file:
                with response["Body"] as input_stream:
                    while chunk := input_stream.read(8 * 1024 * 1024):
                        output_file.write(chunk)
            logger.debug(f"Downloading {s3_key} to {output_path} done.")
            return
        except Exception as e:
            logger.warning(f"Exception while downloading {s3_key} to {output_path}: {e}")
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to download {s3_key} to {output_path}") from e


def main():
    load_dotenv()
    args = parse_args()
    s3_client = create_s3_client()

    bucket = args.bucket
    output_dir = args.output / bucket
    ignore_existing = args.ignore_existing_files
    n_threads = args.n_threads
    page_size = args.s3_page_size
    logger.info(f"From {bucket} to {output_dir} using {n_threads} threads.")
    output_dir.mkdir(parents=True, exist_ok=True)


    futures = []
    download_count = 0
    previous_ignored = False
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        for s3_key in _s3_keys(s3_client, bucket, page_size):
            output_path = output_dir / s3_key
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if output_path.exists() and ignore_existing:
                logger.debug(f"Ignoring existing file {output_path}")
                previous_ignored = True
                continue
            if previous_ignored:
                logger.info(f"Ignored the previous file, continuing with {s3_key}...")
                previous_ignored = False

            future = executor.submit(_download_object, s3_client, bucket, s3_key, output_path)
            futures.append(future)
        for future in as_completed(futures):
            future.result()  # Will raise RuntimeError if a thread failed
            download_count += 1
            if download_count % 100 == 0:
                logger.info(f"Downloaded {download_count} files...")
    logger.info(f"Done! Downloaded {download_count} files to {output_dir}")


if __name__ == '__main__':
    main()
