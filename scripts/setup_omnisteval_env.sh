#!/usr/bin/env bash
set -euo pipefail

ROOT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OMNISTEVAL_DIR="${OMNISTEVAL_DIR:-${ROOT_PATH}/OmniSTEval}"
OMNISTEVAL_VENV="${OMNISTEVAL_VENV:-${OMNISTEVAL_DIR}/.venv}"
PYTHON_VERSION="${OMNISTEVAL_PYTHON_VERSION:-3.11}"

mkdir -p "${OMNISTEVAL_DIR}"

uv venv "${OMNISTEVAL_VENV}" --python "${PYTHON_VERSION}"

uv pip install \
  --python "${OMNISTEVAL_VENV}/bin/python" \
  "OmniSTEval[comet]"

# OmniSTEval 0.1.10 does not publish a simulstream extra, so install it
# explicitly for --hypothesis_format simulstream support.
uv pip install \
  --python "${OMNISTEVAL_VENV}/bin/python" \
  simulstream

"${OMNISTEVAL_VENV}/bin/omnisteval" --help >/dev/null

"${OMNISTEVAL_VENV}/bin/python" - <<'PY'
import comet
import omnisteval
import simulstream

print(f"OmniSTEval import OK: {omnisteval.__file__}")
print(f"COMET import OK: {comet.__file__}")
print(f"SimulStream import OK: {simulstream.__file__}")
PY

echo "OmniSTEval environment is ready: ${OMNISTEVAL_VENV}"
