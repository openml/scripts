"""
This script can be used to convert arff files to parquet and upload them to the
OpenML S3 server. Dataset files are stored in a single `datasets` bucket, which
has many subdirectories, structured as follows:

datasets/
  0000/
    1/
      dataset_1.pq
      [other files for dataset 1]
    [directories for other datasets 0001-9999]
  0001/
    [directories for datasets 10000-19999]
  [directories for other dataset id brackets]

The bucket is assumed to exist.

The subdirectory structure will be made automatically as needed by this script.
See `--help` for information on how to invoke this script.
"""
import argparse
from loguru import logger
import os
import tempfile
from pathlib import Path

import openml
from dotenv import load_dotenv


from conversion import dataset_to_parquet
from existing_datasets import find_latest, find_latest_dataset, dataset_has_parquet, \
    format_parquet_object_name, DATASET_BUCKET
from s3_helper import create_s3_client


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    data_id_group = parser.add_mutually_exclusive_group(required=True)
    data_id_group.add_argument(
        "--id",
        type=int,
        dest="dataset_id",
        help="Set to convert only this single dataset."
    )
    data_id_group.add_argument(
        "--from",
        type=int,
        dest="from_",
        help="If set, try to convert and upload all dataset files starting from the"
             "provided ID and up."
    )
    data_id_group.add_argument(
        "--latest",
        action='store_true',
        help="If set, determine the last dataset which already has a parquet file "
             "associated with it, and resume from there."
    )
    parser.add_argument(
        "--overwrite",
        action='store_true',
        help="Delete the old object (if it exists), before uploading the new file(s). "
             "If not set, skip datasets which already have a parquet file."
    )
    parser.add_argument(
        "--dry",
        action='store_true',
        help="If True, don't delete or upload anything."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    load_dotenv()
    s3_client = create_s3_client()

    if server := os.environ.get("OPENML_SERVER"):
        openml.config.server = server

    logger.info(f"Script invoked with arguments {args}")
    if args.latest:
        last_with_parquet, highest_on_server = find_latest(s3_client)
        dids = range(last_with_parquet + 1, highest_on_server + 1)
    elif args.from_ is not None:
        latest_dataset_id = find_latest_dataset()
        dids = range(args.from_, latest_dataset_id+1)
    else:
        dids = [args.dataset_id]

    for dataset_id in dids:
        file_exists = dataset_has_parquet(dataset_id, s3_client)
        dataset_object = format_parquet_object_name(dataset_id)
        if file_exists:
            if not args.overwrite:
                logger.info(f"Parquet file {DATASET_BUCKET}/{dataset_object} already exists.")
                continue

        logger.info(f"Converting {dataset_id}")
        with tempfile.TemporaryDirectory() as tmp_dir:
            parquet_file = Path(tmp_dir) / f"{dataset_id}.pq"

            try:
                dataset_to_parquet(dataset_id, output=parquet_file)
            except Exception as e:
                logger.exception(f"Error occurred while converting {dataset_id} to parquet: {e}")
                continue

            if args.dry:
                logger.info("Skipping (removal and) upload in `dry` mode.")
                continue
            if file_exists:
                logger.info(f"Deleting {DATASET_BUCKET}/{dataset_object}")
                s3_client.delete_object(Bucket=DATASET_BUCKET, Key=dataset_object)
            logger.info(f"Uploading {DATASET_BUCKET}/{dataset_object}.")
            s3_client.upload_file(
                Bucket=DATASET_BUCKET,
                Key=dataset_object,
                Filename=str(parquet_file),
            )


if __name__ == "__main__":
    main()
