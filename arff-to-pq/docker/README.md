# Arff to Parquet
This docker image is used to 

1) List the last uploaded datasets
2) Convert the `.arff` file to `.pq` (parquet)
3) Upload the parquet file to Minio


## Usage

```bash
docker run --rm openml/openml-arff-to-pq:latest --help
docker run --rm --env-file .env openml/openml-arff-to-pq:latest --id 1083 --dry
```


## Build and publish
Make sure to update the tag!
```bash
docker build -f docker/Dockerfile --tag openml/openml-arff-to-pq:v[VERSION] .
docker push openml/openml-arff-to-pq:v[VERSION]
```
