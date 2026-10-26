#!/usr/bin/env bash
set -euo pipefail

ROOT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COSYVOICE_VENV="${COSYVOICE_VENV:-${ROOT_PATH}/.cosyvoice-venv}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

if [[ ! -f "${ROOT_PATH}/third_party/CosyVoice/requirements.txt" ]]; then
  echo "CosyVoice requirements not found: ${ROOT_PATH}/third_party/CosyVoice/requirements.txt" >&2
  echo "Run bash scripts/fetch_third_party.sh first." >&2
  exit 1
fi

uv venv "${COSYVOICE_VENV}" --python "${PYTHON_VERSION}"

NORMALIZED_REQUIREMENTS="$(mktemp /tmp/simpotrans-cosyvoice-requirements.XXXXXX.txt)"
trap 'rm -f "${NORMALIZED_REQUIREMENTS}"' EXIT
sed 's/^protobuf==4\.25$/protobuf==4.25.3/' \
  "${ROOT_PATH}/third_party/CosyVoice/requirements.txt" > "${NORMALIZED_REQUIREMENTS}"

uv pip install \
  --python "${COSYVOICE_VENV}/bin/python" \
  --index-strategy unsafe-best-match \
  -r "${NORMALIZED_REQUIREMENTS}"

uv pip install \
  --python "${COSYVOICE_VENV}/bin/python" \
  --index-strategy unsafe-best-match \
  "numpy==1.26.4" \
  "Cython>=3"

uv pip install \
  --python "${COSYVOICE_VENV}/bin/python" \
  --index-strategy unsafe-best-match \
  --reinstall \
  --no-build-isolation \
  --no-binary pyworld \
  "numpy==1.26.4" \
  "Cython>=3" \
  "pyworld==0.3.4"

uv pip install \
  --python "${COSYVOICE_VENV}/bin/python" \
  --index https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match \
  "torch==2.9.1" \
  "torchaudio==2.9.1"

uv pip install \
  --python "${COSYVOICE_VENV}/bin/python" \
  --index-strategy unsafe-best-match \
  "fugashi>=1.5.2" \
  "unidic-lite>=1.0.8" \
  "soundfile>=0.12.1" \
  "setuptools>=77,<81"

COSYVOICE_SITE_PACKAGES="$("${COSYVOICE_VENV}/bin/python" - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"

cat > "${COSYVOICE_SITE_PACKAGES}/simpotrans-cosyvoice-paths.pth" <<EOF
${ROOT_PATH}/src
${ROOT_PATH}
${ROOT_PATH}/third_party/CosyVoice
${ROOT_PATH}/third_party/CosyVoice/third_party/Matcha-TTS
EOF

"${COSYVOICE_VENV}/bin/python" - <<'PY'
import torch
import torchaudio
from cosyvoice.cli.cosyvoice import AutoModel

print(f"torch: {torch.__version__}")
print(f"torchaudio: {torchaudio.__version__}")
print(f"CosyVoice AutoModel import OK: {AutoModel}")
PY

echo "CosyVoice environment is ready: ${COSYVOICE_VENV}"
