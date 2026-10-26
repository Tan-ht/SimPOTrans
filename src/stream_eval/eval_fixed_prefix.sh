#!/usr/bin/env zsh
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  zsh src/stream_eval/eval_fixed_prefix.sh <src_lang> <tgt_lang> <device> <eval_data_name> [checkpoint] [output_name]

Example:
  zsh src/stream_eval/eval_fixed_prefix.sh en zh 0 stream_eval_acl checkpoints/covost/st_backbone_covost/ws_2.f3820fc3/checkpoints/step_6000

Environment overrides are the same as eval_fixed_dialogue.sh, including
CHUNK_SIZES="1.0 1.5 2.0 3.0 4.0 5.0". BASE_CONFIG defaults to
src/stream_eval/config/fixed_processor_prefix.yaml, and PROCESSOR_NAME
defaults to SimPOTrans-fixed-prefix.
EOF
}

if [[ $# -lt 4 ]]; then
    usage
    exit 2
fi

SCRIPT_PATH=${(%):-%x}
SCRIPT_DIR=${SCRIPT_PATH:A:h}

export BASE_CONFIG=${BASE_CONFIG:-${SCRIPT_DIR}/config/fixed_processor_prefix.yaml}
export PROCESSOR_NAME=${PROCESSOR_NAME:-SimPOTrans-fixed-prefix}

if [[ $# -lt 6 && -z "${6:-}" ]]; then
    export OUTPUT_SUFFIX=${OUTPUT_SUFFIX:-prefix}
fi

if [[ $# -lt 6 && -n "${OUTPUT_SUFFIX:-}" ]]; then
    exec zsh "${SCRIPT_DIR}/eval_fixed_dialogue.sh" "$1" "$2" "$3" "$4" "${5:-${CKPT_PATH:-}}" "${OUTPUT_SUFFIX}"
fi

exec zsh "${SCRIPT_DIR}/eval_fixed_dialogue.sh" "$@"
