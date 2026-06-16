# Dockerfile — multi-stage build for langembed services.
#
# Stages:
#   base  → annotation service (~400 MB, no torch)
#   ml    → serving + training image (~4 GB, includes torch)
#
# Build examples:
#   docker build --target base -t langembed-annotation .
#   docker build --target ml   -t langembed-ml .
#
# Artifacts (models, data) are mounted as volumes — never baked in.

FROM python:3.11-slim AS base
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir -e ".[serve]"

# ---------------------------------------------------------------------------

FROM base AS ml

RUN pip install --no-cache-dir -e ".[ml]"
