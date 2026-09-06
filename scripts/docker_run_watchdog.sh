#!/bin/bash
# Reusable launcher for the full-pipeline DAG's compute-heavy tasks: runs a
# `docker run` against langembed-ml with a unique container name, a disk/memory
# watchdog, and an outer timeout -- the same hardened pattern already proven in
# all_branches_queue.sh, parameterized instead of hardcoded to one nightly job.
#
# Usage:
#   docker_run_watchdog.sh <unique-name-prefix> <timeout-minutes> <use-gpu:true|false> <script-args...>
#
# <script-args...> is forwarded as-is to `python <script-args...>` inside the
# container, e.g.:
#   docker_run_watchdog.sh mr-run-123-shared-corpus-prep 240 true \
#       scripts/run_pipeline.py --lang mr --raw-input data/raw/mr_nllb.txt \
#       --auto-label --auto-label-method svd --embed-sample-size 200
set -euo pipefail

NAME_PREFIX="$1"; shift
TIMEOUT_MINUTES="$1"; shift
USE_GPU="$1"; shift

BASE=/home/s939/langembed_deploy/langembed
MIN_FREE_GB=50
MIN_AVAIL_MEM_GB=2
CONTAINER="${NAME_PREFIX}-$(date +%s)"

GPU_FLAG=()
if [ "$USE_GPU" = "true" ]; then
  GPU_FLAG=(--gpus all)
fi

(
  while true; do
    sleep 20
    FREE_GB=$(df --output=avail -B1G / | tail -1 | tr -d ' ')
    AVAIL_MEM_GB=$(free -g | awk '/^Mem:/{print $7}')
    if [ -n "$FREE_GB" ] && [ "$FREE_GB" -lt "$MIN_FREE_GB" ]; then
      echo "SEVERE: disk free ${FREE_GB}GB < ${MIN_FREE_GB}GB - killing $CONTAINER" >&2
      docker kill "$CONTAINER" >/dev/null 2>&1 || true
      break
    fi
    if [ -n "$AVAIL_MEM_GB" ] && [ "$AVAIL_MEM_GB" -lt "$MIN_AVAIL_MEM_GB" ]; then
      echo "SEVERE: available memory ${AVAIL_MEM_GB}GB < ${MIN_AVAIL_MEM_GB}GB - killing $CONTAINER" >&2
      docker kill "$CONTAINER" >/dev/null 2>&1 || true
      break
    fi
    if ! docker inspect "$CONTAINER" >/dev/null 2>&1; then
      break
    fi
  done
) &
WATCHDOG_PID=$!

cd "$BASE"
RC=0
# Run as the invoking host user, not the image's default (root) -- otherwise
# every output file/dir written under the $BASE bind mount ends up root-owned,
# and the *next* run's plain (non-sudo) `rm -rf` against that same path fails
# with Permission denied. -e HOME=/tmp gives libraries that expect a writable
# $HOME for caches (pip, HuggingFace, matplotlib, etc.) somewhere to write,
# since this non-root uid/gid has no passwd entry inside the container and
# therefore no real home directory of its own.
timeout "${TIMEOUT_MINUTES}m" docker run --name "$CONTAINER" \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  "${GPU_FLAG[@]}" \
  --add-host host.docker.internal:host-gateway \
  -v "$BASE":/app -v /mnt/nvme-mssql:/mnt/nvme-mssql -w /app \
  langembed-ml:latest \
  python "$@" || RC=$?

kill "$WATCHDOG_PID" 2>/dev/null || true
wait "$WATCHDOG_PID" 2>/dev/null || true
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

exit $RC
