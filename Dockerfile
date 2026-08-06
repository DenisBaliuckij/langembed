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
# Pass --build-arg PIP_INDEX_URL=<mirror> if the build host can't reach the
# default PyPI index (e.g. a network that only routes to a regional mirror).
#
# Artifacts (models, data) are mounted as volumes — never baked in.

FROM python:3.11-slim AS base
ARG PIP_INDEX_URL=https://pypi.org/simple
ENV PIP_INDEX_URL=$PIP_INDEX_URL
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

RUN pip install --no-cache-dir --timeout 300 -e ".[ml,translate]"
