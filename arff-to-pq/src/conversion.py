"""Functions for converting the openml arff file to the final parquet file."""

import os
import shutil
from pathlib import Path
import re
from typing import Iterable
from loguru import logger

import arff
import openml
import pandas as pd

class IdentityDict(dict):
    def __missing__(self, key):
        return key


# arff interprets *any* backslash+character sequence as an escaped character,
# but unfortunately several are poorly formatted and include e.g., literal file paths
# (C:\\hello\world) where backslashes are not properly escaped. This monkey patch
# allows us to read those files.
arff._ESCAPE_SUB_MAP = IdentityDict(arff._ESCAPE_SUB_MAP)

# ARFF Files lead with a header which specify (among other things) @attributes.
# The end of the header is marked with a @data line.
ATTRIBUTE_MARKER = "@attribute"
QUOTED_ATTRIBUTE_PATTERN = re.compile(r"@ATTRIBUTE ([\'\"].*?[\'\"])", re.IGNORECASE)
DATA_MARKER = "@data"


def remove_attribute_whitespace(arff_file: Path):
    """Remove the leading and trailing whitespace from quoted attribute names, and
    any leading spaces before commas.

    Note that this breaks technically legal categories which have in them a comma
    with leading spaces ("he ,"), but we assume such do not exist.

    E.g.:
    - @ATTRIBUTE "my_feature " {"foo " , "bar"}
    + @ATTRIBUTE "my_feature" {"foo ", "bar"}
    """
    lines = arff_file.read_text().split("\n")

    for i, line in enumerate(lines):
        if line.casefold().startswith(ATTRIBUTE_MARKER):
            if quoted_attribute := re.search(QUOTED_ATTRIBUTE_PATTERN, line):
                # The format specifies that attributes with _any_ whitespace must be quoted
                # However, the OpenML Features Engine automatically removes trailing/leading
                # whitespace of feature names, which leads to a mismatch. Here, we can
                # make sure the new data file feature names match those of the Feature Engine
                # by removing the whitespace surrounding feature names.
                feature_name, = quoted_attribute.groups()
                unquoted_name = feature_name[1:-1]
                line = line.replace(unquoted_name, unquoted_name.strip())
            # The Python ARFF parser can not handle spaces before a comma
            line = re.sub(r" *,", ",", line)
            lines[i] = line
        elif line.casefold().startswith(DATA_MARKER):
            break

    arff_file.write_text('\n'.join(lines))


def datetime_to_string(arff_file: Path) -> Iterable[tuple[str, str]]:
    # The Python ARFF parser does not handle date time, so we mark them as STRING here.
    # Then, after loading the datetime as string with the arff parser, we convert
    # the data back to datetime within Pandas.
    lines = arff_file.read_text().split("\n")

    datetime_pattern = r"@ATTRIBUTE ([\'\"].*[\'\"]|[^\'\"\s]*) (date) ?(.*)?"
    datetime_pattern = re.compile(datetime_pattern, re.IGNORECASE)
    datetime_attributes = []
    for i, line in enumerate(lines):
        if match := re.search(datetime_pattern, line):
            name, dtype, datetime_format = match.groups()
            lines[i] = f"@attribute {name} string\n"
            datetime_attributes.append((name, datetime_format))
        elif line.casefold().startswith("@data"):
            break

    arff_file.write_text('\n'.join(lines))
    return datetime_attributes


def is_uint8(data: pd.Series) -> bool:
    # This is a minor optimization which was already deployed within openml-python
    # this is significant for datasets with embedded images (which typically have
    # "pixel" features which neatly fit into a byte).
    return not data.isna().any() and ((0 <= data) & (data <= 255)).all() and (data == data.astype("uint8")).all()


def unescape_single_slash(arff_file: Path):
    file_text = arff_file.read_text()
    single_slash_percent = re.compile(r"[^\\]\\%")

    def remove_single_slash(match: re.Match) -> str:
        prefix = match.group()[0]
        return f"{prefix}%"

    fixed_text, _ = re.subn(
        single_slash_percent,
        remove_single_slash,
        file_text,
    )
    arff_file.write_text(fixed_text)


def arff_to_pandas(arff_file: Path):
    # The server feature analyzer arff parser behaves differently from the Python arff parser.
    # To account for this, we modify the arff file slightly so the output is more similar.
    # This helps us re-order the features (WHY IS THAT EVEN NEEDED??)
    remove_attribute_whitespace(arff_file)
    unescape_single_slash(arff_file)
    datetime_attributes = datetime_to_string(arff_file)

    with arff_file.open() as fh:
        dataset = arff.load(fh)

    columns = [name for name, dtype in dataset["attributes"]]
    df = pd.DataFrame(dataset["data"], columns=columns)

    for name, type_ in dataset["attributes"]:
        if name in [cname for cname, _ in datetime_attributes]:
            df[name] = pd.to_datetime(df[name])
        elif (
                isinstance(type_, list)
                and len(set(type_)) in (2, 3)
                and all(v is None or isinstance(v, str) for v in type_)
                and {"true", "false"} == {v.casefold() for v in type_ if v is not None}
        ):
            # An optimization/convenience already applied by openml-python:
            # ARFF does not support boolean types, but some categories are boolean
            # beyond reasonable doubt. So we encode this information in the data.
            def str_as_bool(value: str):
                return None if value is None else value.casefold() == "true"
            df[name] = df[name].apply(str_as_bool)
        elif isinstance(type_, list):
            df[name] = df[name].astype("category")
        elif type_.casefold() in ["real", "integer", "numeric"] and is_uint8(df[name]):
            df[name] = df[name].astype("uint8")

    return df


def dataset_to_parquet(dataset_id: int, output: Path, max_cells: int = 1_000_000_000) -> None:
    logger.info(f"Downloading dataset {dataset_id} metadata.")
    dataset = openml.datasets.get_dataset(
        dataset_id,
        download_data=False,
        download_qualities=True,
        download_features_meta_data=False,
    )
    if dataset.format.lower() not in ["sparse_arff", "arff"]:
        raise RuntimeError(f"Unexpected dataset.format: {dataset.format}")
    cells = dataset.qualities["NumberOfInstances"] * dataset.qualities["NumberOfFeatures"]
    if cells > max_cells:
        raise ValueError(f"Dataset has too many cells ({cells}).")
    logger.info(f"Downloading dataset {dataset_id}.")
    dataset = openml.datasets.get_dataset(
        dataset_id,
        download_data=True,
        download_qualities=False,
        download_features_meta_data=False,
    )
    if dataset.parquet_file:
        logger.info("Parquet found using get_dataset. Skipping the conversion.")
        shutil.copy(dataset.parquet_file, output)
        return
    logger.info(f"Converting dataset {dataset_id}.")
    dataframe = arff_to_pandas(Path(dataset.data_file))
    server_feature_names = [f.name for f in dataset.features.values()]
    column_names = [c for c in dataframe.columns]
    only_on_server = set(server_feature_names) - set(dataframe.columns)
    only_in_file = set(dataframe.columns) - set(server_feature_names)
    if server_feature_names != column_names and set(server_feature_names) == set(column_names):
        raise ValueError("Order of columns changed!")
    assert not (only_in_file | only_on_server), (
        "DataFrame and server features diverge, likely due to behavior of the FeatureAnalyzer."
        f"Features only on the server: {only_on_server}"
        f"Features only in the file: {only_in_file}"
    )
    dataframe = dataframe[server_feature_names]
    logger.info(f"Saving dataset to '{output}'.")
    dataframe.to_parquet(path=output)


def main():
    try:
        shutil.rmtree("omlcache")
        os.mkdir("omlcache")
    except FileNotFoundError:
        pass
    openml.config.set_cache_directory("omlcache")

    # This is a data file provided by @prabhant which contains all datasets that do not
    # load out of the box with openml-python.
    with open("bad_arffs.txt", "r") as fh:
        lines = fh.readlines()

    failures = [line.split(":") for line in lines]
    start = 0
    for i, (did, error) in enumerate(failures[start:], start=start):
        # The student below uploaded many datasets of the first ~10 at least have been
        # invalid. So for now I assume all datasets uploaded by her are invalid and we
        # can inspect them later
        student_uploads =[42900, 42901, 42903, 42904, 42908, 42909, 42910, 42911, 42912, 42915, 42967, 42968, 42969, 42964, 42965, 42966, 42918, 42961, 42962, 42963, 42905, 42906, 42907, 43003, 43004, 43005, 42987, 42989, 42990, 42985, 42996, 42997, 42998, 43006, 43007, 43008, 42992, 42993, 42994, 42999, 43000, 42991, 42986, 42877, 42878, 42896, 42897, 42898, 42880, 42887, 42889, 42891, 42893, 42895, 42973, 42974, 42970, 42971, 42972, 42978, 42982, 42984, 42979, 42980, 42981, 42977, 42975, 42976, 43038, 43039, 43040, 43009, 42890]
        if str(did) in ["41190", "41949", "4537", "4546", "4562", "42564", "43147", "43149","40598", *[str(student_did) for student_did in student_uploads]]:
            # 42564 needs a manual fix / new version? Each row is its own categorical variable.
            # 41190 and 41949 have encoding issues and are available as parquet in /parquet-files
            # 4537, 4546, and 4562 have server-side issues (e.g., no features)
            # 42979 is invalid arff, looks like first row is encoded as attributes.
            # 43147, 43149 needs a manual fix: it contains an unescape backslash before the
            # string terminating quote.
            # 40598 has trailing comma
            continue
        d = openml.datasets.get_dataset(
            did,
            download_data=True,
            download_qualities=False,
        )
        print(i, d.data_file)
        df = arff_to_pandas(Path(d.data_file))

if __name__ == "__main__":
    main()
