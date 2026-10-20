# Arff to parquet conversion
This project is used to 

1) List the last uploaded datasets
2) Convert the `.arff` file to `.pq` (parquet)
3) Upload the parquet file to S3 object storage

## Usage
```commandline
python3 src/cli.py --help
 
python3 src/cli.py --id 1083 --dry
```

## Development
Installing dependencies:
```bash
python3 -m venv venv
source venv/bin/activate

python -m pip install uv
uv pip install -e ".[dev]"
```

