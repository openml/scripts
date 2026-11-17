import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from boto3.s3.transfer import TransferConfig
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from loguru import logger


from s3_helper import create_s3_client, file_exists, create_public_bucket_if_not_exists


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", "--bucket",
        type=str,
        required=True,
        help = "The S3 bucket"
    )
    parser.add_argument(
        "-input", "--input",
        type=Path,
        required=True,
        help="The input directory (without the bucket name). E.g., /my/path/. All data from "
             "/my/path/[bucket] will be uploaded."
    )
    parser.add_argument(
        "--n_threads",
        type=int,
        default=1,
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
    input_dir = args.input / bucket
    ignore_existing = args.ignore_existing_files
    n_threads = args.n_threads
    logger.info(f"From {input_dir} to {bucket} using {n_threads} threads.")

    config = TransferConfig(
        multipart_threshold=25 * 1024 * 1024,  # 25 MB
        multipart_chunksize=25 * 1024 * 1024,  # 25 MB
        max_concurrency=10,
        use_threads=True,
    )
    files_to_upload = [p for p in input_dir.rglob("*") if p.is_file()]

    def _upload_single_file(file_path: Path):
        s3_key = file_path.relative_to(input_dir).as_posix()
        if ignore_existing and file_exists(s3_client, bucket, s3_key):
            logger.debug(f"File {file_path} already exists! Ignoring the upload.")
            return
        logger.debug(f"Uploading {file_path} to bucket {bucket} key {s3_key}...")
        s3_client.upload_file(Filename=str(file_path), Bucket=bucket, Key=s3_key, Config=config)
        logger.debug(f"Uploading {file_path} to bucket {bucket} key  {s3_key} done.")

    logger.info("Creating public bucket if it doesn't exist...")
    create_public_bucket_if_not_exists(s3_client, bucket)
    logger.info(f"Starting upload of {len(files_to_upload)} files to s3://{bucket}...")
    uploaded_count = 0
    failed_count = 0
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = {
            executor.submit(_upload_single_file, file_path): file_path
            for file_path in files_to_upload
        }
        for future in as_completed(futures):
            file_path = futures[future]
            try:
                future.result()
                uploaded_count += 1
                if uploaded_count % 100 == 0:
                    logger.info(f"Uploaded {uploaded_count} of {len(files_to_upload)} (excluding {failed_count} failed uploads)")
            except ClientError as e:
                logger.warning(f"Failed to upload {file_path}! {e}")
                failed_count += 1
    logger.info(f"Finished! {uploaded_count} of {len(files_to_upload)} files uploaded. There were {failed_count} failed uploads.")

if __name__ == '__main__':
    main()
