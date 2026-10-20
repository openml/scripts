# Pre-filled-object-storage

Creating a docker test image for OpenML: a S3 (seaweedfs) instance pre-filled with datasets.

## Usage

```bash
docker run --rm openml/seaweedfs-with-datasets:latest
```


## Build and publish
Make sure to update the tag!
```bash
docker build -f docker/Dockerfile --tag openml/seaweedfs-with-datasets:v[VERSION] .
docker push openml/seaweedfs-with-datasets:v[VERSION]
```

