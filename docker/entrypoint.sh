#!/bin/sh
# Diagnostic mode: send shell errors and execution tracing to the container's
# standard output so Railway captures failures that happen before Uvicorn.
exec 2>&1
set -eux

artifact_directory="$(dirname "${MEDTRACK_MODEL_URI}")"
easyocr_directory="${EASYOCR_MODULE_PATH:-/home/app/.EasyOCR}"

# Temporary Railway diagnostic: keep the process running as root. This isolates
# permission failures from entrypoint, model-download, and application failures.
echo "Diagnostic mode: running as uid=$(id -u), gid=$(id -g)"
mkdir -p "${artifact_directory}" "${easyocr_directory}"

if [ "${MEDTRACK_FETCH_MODEL_ON_START:-false}" = "true" ]; then
    python scripts/fetch_model.py --manifest "${MEDTRACK_MODEL_MANIFEST}"
fi

exec "$@"
