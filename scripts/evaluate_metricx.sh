#!/usr/bin/env bash
set -euo pipefail
if [[ $# -eq 0 || "${1:-}" == --help || "${1:-}" == -h ]]; then
  echo 'Usage: bash scripts/evaluate_metricx.sh <result_root> [cpu|GPU index|cuda:N]'
  echo 'Set METRICX_SOURCE_SENTENCES_FILE for aligned source text, especially for offline results.'
  exit 0
fi
ROOT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_PATH}"
metricx_python="${METRICX_PYTHON:-${ROOT_PATH}/.metricx-venv/bin/python}"
metricx_repo="${METRICX_REPO:-${ROOT_PATH}/MetricX}"
device="${2:-cpu}"
case "${device}" in
  cpu) export CUDA_VISIBLE_DEVICES="" ;;
  cuda|cuda:*) ;;
  *[!0-9]*|'') echo "Invalid device: ${device}" >&2; exit 2 ;;
  *) export CUDA_VISIBLE_DEVICES="${device}"; device=cuda ;;
esac
if [[ ! -x "${metricx_python}" ]]; then
  echo 'Run bash scripts/setup_metricx_env.sh first.' >&2
  exit 1
fi
# Keep MetricX's older Transformers ahead of packages from the main environment.
metricx_site=$("${metricx_python}" -c 'import site; print(site.getsitepackages()[0])')
export PYTHONPATH="${metricx_repo}:${metricx_site}:${PYTHONPATH:-}"
args=("$1")
if [[ -n "${METRICX_SOURCE_SENTENCES_FILE:-}" ]]; then
  args+=("${METRICX_SOURCE_SENTENCES_FILE}")
fi
if [[ "${FORCE:-0}" == 1 ]]; then
  args+=(--force)
fi
exec "${metricx_python}" "${ROOT_PATH}/src/stream_eval/metricx_eval.py" \
  "${args[@]}" \
  --evaluation-data-root "${ROOT_PATH}/data/evaluation_data" \
  --eval-data "${METRICX_EVAL_DATA:-auto}" \
  --metricx-repo "${metricx_repo}" \
  --model "${METRICX_MODEL:-google/metricx-24-hybrid-xxl-v2p6-bfloat16}" \
  --tokenizer "${METRICX_TOKENIZER:-google/mt5-xl}" \
  --batch-size "${METRICX_BATCH_SIZE:-1}" \
  --max-input-length "${METRICX_MAX_INPUT_LENGTH:-1536}" \
  --device "${device}"
