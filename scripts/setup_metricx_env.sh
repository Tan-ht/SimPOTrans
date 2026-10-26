#!/usr/bin/env bash
set -euo pipefail

ROOT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAIN_VENV="${MAIN_VENV:-${ROOT_PATH}/.venv}"
METRICX_REPO="${METRICX_REPO:-${ROOT_PATH}/MetricX}"
METRICX_VENV="${METRICX_VENV:-${ROOT_PATH}/.metricx-venv}"
METRICX_COMMIT="${METRICX_COMMIT:-fc4978eb064670f7cc33e93ea4f52d38396b8ae6}"
PYTHON_VERSION="${METRICX_PYTHON_VERSION:-${PYTHON_VERSION:-3.10}}"

if [[ ! -x "${MAIN_VENV}/bin/python" ]]; then
  echo "Main virtual environment not found: ${MAIN_VENV}" >&2
  echo "Run uv sync first, or set MAIN_VENV=/path/to/venv." >&2
  exit 1
fi

if [[ ! -d "${METRICX_REPO}/.git" ]]; then
  if [[ -e "${METRICX_REPO}" && -n "$(find "${METRICX_REPO}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "METRICX_REPO exists but is not a git checkout: ${METRICX_REPO}" >&2
    exit 1
  fi
  git clone https://github.com/google-research/metricx.git "${METRICX_REPO}"
fi

if ! git -C "${METRICX_REPO}" diff --quiet || ! git -C "${METRICX_REPO}" diff --cached --quiet; then
  echo "MetricX checkout has local changes; refusing to switch revisions: ${METRICX_REPO}" >&2
  exit 1
fi

if ! git -C "${METRICX_REPO}" cat-file -e "${METRICX_COMMIT}^{commit}" 2>/dev/null; then
  git -C "${METRICX_REPO}" fetch origin "${METRICX_COMMIT}"
fi
git -C "${METRICX_REPO}" checkout --detach "${METRICX_COMMIT}"

if [[ ! -x "${METRICX_VENV}/bin/python" ]]; then
  uv venv "${METRICX_VENV}" --python "${PYTHON_VERSION}" --system-site-packages
else
  echo "Reusing MetricX virtual environment: ${METRICX_VENV}"
fi
uv pip install \
  --python "${METRICX_VENV}/bin/python" \
  "transformers==4.30.2" \
  "sentencepiece==0.1.99" \
  "protobuf==3.20.3"

MAIN_SITE_PACKAGES=$("${MAIN_VENV}/bin/python" - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)

METRICX_SITE_PACKAGES=$("${METRICX_VENV}/bin/python" - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)

echo "${MAIN_SITE_PACKAGES}" > "${METRICX_SITE_PACKAGES}/simpotrans-main-venv.pth"

PYTHONPATH="${METRICX_REPO}" "${METRICX_VENV}/bin/python" - <<'PY'
import torch
import transformers
import google.protobuf
from metricx24 import models

print(f"MetricX model API OK: {models.__file__}")
print(f"torch: {torch.__version__}")
print(f"transformers: {transformers.__version__}")
print(f"protobuf: {google.protobuf.__version__}")
PY

echo "MetricX environment is ready: ${METRICX_VENV}"
echo "MetricX source revision: ${METRICX_COMMIT}"
