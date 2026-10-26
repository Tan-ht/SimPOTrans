#!/usr/bin/env bash
set -euo pipefail

ROOT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAIN_VENV="${MAIN_VENV:-${ROOT_PATH}/.venv}"
COMET_VENV="${COMET_VENV:-${ROOT_PATH}/.comet-venv}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

if [[ ! -x "${MAIN_VENV}/bin/python" ]]; then
  echo "Main virtual environment not found: ${MAIN_VENV}" >&2
  echo "Run uv sync first, or set MAIN_VENV=/path/to/venv." >&2
  exit 1
fi

uv venv "${COMET_VENV}" --python "${PYTHON_VERSION}" --system-site-packages

uv pip install \
  --python "${COMET_VENV}/bin/python" \
  "numpy<2.0" \
  "unbabel-comet>=2.2.0" \
  "pytorch-lightning>=2.0,<3" \
  entmax \
  lightning-utilities \
  torchmetrics \
  jsonargparse \
  sentencepiece \
  protobuf \
  pandas \
  scipy \
  scikit-learn \
  pytz \
  --no-deps

MAIN_SITE_PACKAGES="$("${MAIN_VENV}/bin/python" - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"

COMET_SITE_PACKAGES="$("${COMET_VENV}/bin/python" - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"

echo "${MAIN_SITE_PACKAGES}" > "${COMET_SITE_PACKAGES}/simpotrans-main-venv.pth"

"${COMET_VENV}/bin/python" - <<'PY'
import comet
import numpy
import torch
from comet import download_model, load_from_checkpoint

print(f"COMET import OK: {comet.__file__}")
print(f"numpy: {numpy.__version__}")
print(f"torch: {torch.__version__}")
print("COMET API OK")
PY

echo "COMET environment is ready: ${COMET_VENV}"
