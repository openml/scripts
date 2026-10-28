import os
import typing

import openml
import pandas as pd

from s3_helper import file_exists

if typing.TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client

DATASET_BUCKET = "datasets"
OPENML_DATASET_OFFSET = int(os.environ.get("OPENML_DATASET_OFFSET", 45400))
BATCH_SIZE = 500

def _format_dataset_directory(dataset_id: int) -> str:
    bracket = dataset_id // 10000
    return f"{bracket:04d}/{dataset_id:04d}"


def format_parquet_object_name(dataset_id: int) -> str:
    dataset_path = _format_dataset_directory(dataset_id)
    return f"{dataset_path}/dataset_{dataset_id}.pq"


def dataset_has_parquet(dataset_id: int, client: S3Client) -> bool:
    parquet_object = format_parquet_object_name(dataset_id)
    return file_exists(client, DATASET_BUCKET, parquet_object)

def find_latest_dataset() -> int:
    last_df = pd.DataFrame()
    # We could use itertools.count, but creating the possibility for an infinite loop
    # which polls production is something I would like to avoid.
    # Current settings should allow up to dataset ID ~95000, which should give us enough
    # time to implement a "latest dataset(s)" endpoint in the Python API.
    for i in range(500):
        datasets = openml.datasets.list_datasets(
            data_id=list(range(OPENML_DATASET_OFFSET + i * BATCH_SIZE, OPENML_DATASET_OFFSET + (i + 1) * BATCH_SIZE)),
            output_format="dataframe",
            status="all",
        )
        if datasets.empty:
            if last_df.empty:
                raise RuntimeError(f"Did not retrieve *any* active datasets. Is the {OPENML_DATASET_OFFSET=} correct?")
            else:
                return int(last_df.did.max())
        last_df = datasets
    raise RuntimeError("Too many datasets on OpenML. Please update the script to use a 'latest "
                       "dataset' endpoint!")


def find_latest(client: S3Client) -> typing.Tuple[int, int]:
    """ Returns a tuple (highest ID with parquet, highest ID on server).

    Latest with Parquet denotes the dataset with the highest ID that already has an
    associated parquet file on MinIO.
    Latest on server denotes the dataset with the highest ID known to OpenML.
    """
    last_dataset_with_parquet = highest_known = find_latest_dataset()
    while last_dataset_with_parquet > 0:
        if dataset_has_parquet(last_dataset_with_parquet, client):
            return last_dataset_with_parquet, highest_known
        last_dataset_with_parquet -= 1
    return last_dataset_with_parquet, highest_known