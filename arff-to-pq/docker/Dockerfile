FROM python:3.14-slim-bookworm

RUN apt update && apt upgrade -y --no-install-recommends && rm -rf /var/lib/apt/list/*
COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /uvx /bin/

RUN useradd --create-home unprivileged-user

WORKDIR /app
COPY pyproject.toml /app/

ENV UV_NO_DEV=1
RUN uv sync

COPY src /app/src
USER unprivileged-user

ENTRYPOINT ["uv", "run", "src/cli.py"]
