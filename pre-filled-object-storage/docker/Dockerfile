# check=skip=SecretsUsedInArgOrEnv

FROM chrislusf/seaweedfs:4.48 AS builder
RUN apk add --no-cache wget
RUN mkdir /tmp/datasets && wget --wait=1 -r -np -nH -R "index.html*,robots.txt" --cut-dirs=1 -P /tmp/datasets https://test.openml.org/test-server-files/datasets
COPY docker/s3_config.json /etc/seaweedfs/s3_config.json

ENV S3_BUCKET=datasets
ENV AWS_ACCESS_KEY_ID=admin
ENV AWS_SECRET_ACCESS_KEY=adminpassword

RUN weed mini & \
    WEED_PID=$! && \
    sleep 2 && \
    weed filer.copy /tmp/datasets/datasets/* -include=* http://localhost:8888/buckets/datasets/ && \
    kill -s INT $WEED_PID && \
    wait $WEED_PID


FROM chrislusf/seaweedfs:4.48
COPY --from=builder /data /data
COPY --from=builder /etc/seaweedfs/s3_config.json /etc/seaweedfs/s3.json
ENV S3_BUCKET=datasets
ENV AWS_ACCESS_KEY_ID=admin
ENV AWS_SECRET_ACCESS_KEY=adminpassword
EXPOSE 9333 8080 8888 8333
ENTRYPOINT ["weed", "mini", "-s3.config=/etc/seaweedfs/s3.json"]
