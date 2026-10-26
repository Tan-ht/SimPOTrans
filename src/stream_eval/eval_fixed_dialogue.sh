#!/usr/bin/env zsh
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  zsh src/stream_eval/eval_fixed_dialogue.sh <src_lang> <tgt_lang> <device> <eval_data_name> [checkpoint] [output_name]

Example:
  zsh src/stream_eval/eval_fixed_dialogue.sh en zh 0 stream_eval_acl checkpoints/covost/st_backbone_covost/ws_2.f3820fc3/checkpoints/step_6000

Environment overrides:
  CHUNK_SIZES="1.0 1.5 2.0 3.0 4.0 5.0"
  # Or set CHUNK_SIZE=1.0 to run one chunk size.
  HISTORY_SELECTION_POLICY=soft  # soft | punctuation | fixed | kv
  MAX_SPEECH_CONTEXT_SECONDS=30.0
  SOFT_SPEECH_CONTEXT_RESET_SECONDS=20.0
  HISTORY_SELECTION_FIXED_WORDS=100
  KV_SPEECH_CONTEXT_SECONDS=20.0
  LLM_CONTEXT_TURNS=5
  DECODE=beam  # Set DECODE=sample for sampling decoding.
  BEAM_SIZE=4
  DO_SAMPLE=0
  TEMPERATURE=0.0
  TOP_P=null
  TOP_K=null
  MAX_NEW_TOKENS=128
  # DECODE=sample defaults: BEAM_SIZE=1, DO_SAMPLE=1, TEMPERATURE=0.6, TOP_P=0.95, TOP_K=20
  MODEL_CONFIG=src/st_backbone/config.yaml
  SPEECH_ENCODER_BACKEND=whisper  # whisper | whisper_rt
  WHISPER_RT_MODEL=large-v2
  WHISPER_RT_CHUNK_SIZE_MS=300
  WHISPER_RT_MULTILINGUAL=true
  INFERENCE_PROMPT=src/st_backbone/st_prompt.json
  BASE_CONFIG=src/stream_eval/config/fixed_processor.yaml
  EXP_ROOT=exp_results/stream_results
  PROCESSOR_NAME=SimPOTrans-fixed-dialogue
  RUN_OMNISTEVAL=1
  OMNISTEVAL_BIN=OmniSTEval/.venv/bin/omnisteval
  RUN_METRICX=1
  METRICX_MODEL=google/metricx-24-hybrid-xxl-v2p6-bfloat16
  RUN_COMET=1
  COMET_MODEL=Unbabel/XCOMET-XL
  CHAR_LEVEL=auto
  OFFSET_DELAYS=0
EOF
}

if [[ $# -lt 4 ]]; then
    usage
    exit 2
fi

SCRIPT_PATH=${(%):-%x}
SCRIPT_DIR=${SCRIPT_PATH:A:h}
ROOT_PATH=$(cd "${SCRIPT_DIR}/../.." && pwd)

src_lang=$1
tgt_lang=$2
device_arg=$3
eval_data_name=$4
ckpt_path=${5:-${CKPT_PATH:-}}
output_name=${6:-}

if [[ -n "${CHUNK_SIZES:-}" ]]; then
    chunk_sizes=("${(@s: :)CHUNK_SIZES}")
elif [[ -n "${CHUNK_SIZE:-}" ]]; then
    chunk_sizes=("${CHUNK_SIZE}")
else
    chunk_sizes=(1.0 1.5 2.0 3.0 4.0 5.0)
fi
history_selection_policy=${HISTORY_SELECTION_POLICY:-${HISTORY_POLICY:-soft}}
history_selection_policy=${(L)history_selection_policy}
soft_speech_context_reset_seconds=${SOFT_SPEECH_CONTEXT_RESET_SECONDS:-20.0}
fixed_history_words=${HISTORY_SELECTION_FIXED_WORDS:-${FIXED_HISTORY_WORDS:-100}}
kv_speech_context_seconds=${KV_SPEECH_CONTEXT_SECONDS:-15.0}
llm_context_turns=${LLM_CONTEXT_TURNS:-5}
llm_raw_kv_cache=0
speech_context_policy=fixed_duration
case "${history_selection_policy}" in
    soft|punctuation|fixed)
        max_speech_context_seconds=${MAX_SPEECH_CONTEXT_SECONDS:-30.0}
        runtime_llm_context_turns=null
        ;;
    kv)
        max_speech_context_seconds=${kv_speech_context_seconds}
        llm_raw_kv_cache=1
        runtime_llm_context_turns=${llm_context_turns}
        ;;
    *)
        echo "Unsupported HISTORY_SELECTION_POLICY: ${history_selection_policy}. Expected soft, punctuation, fixed, or kv." >&2
        exit 2
        ;;
esac
decode_mode=${DECODE:-${DECODE_MODE:-${decode:-beam}}}
case "${decode_mode}" in
    beam)
        beam_size=${BEAM_SIZE:-4}
        do_sample=0
        temperature=${TEMPERATURE:-0.0}
        top_p=${TOP_P:-null}
        top_k=${TOP_K:-null}
        ;;
    sample)
        beam_size=${BEAM_SIZE:-1}
        do_sample=1
        temperature=${TEMPERATURE:-0.6}
        top_p=${TOP_P:-0.95}
        top_k=${TOP_K:-20}
        ;;
    *)
        echo "Unsupported DECODE mode: ${decode_mode}. Expected beam or sample." >&2
        exit 2
        ;;
esac
max_new_tokens=${MAX_NEW_TOKENS:-128}
model_config=${MODEL_CONFIG:-${ROOT_PATH}/src/st_backbone/config.yaml}
inference_prompt=${INFERENCE_PROMPT:-${ROOT_PATH}/src/st_backbone/st_prompt.json}
base_config=${BASE_CONFIG:-${SCRIPT_DIR}/config/fixed_processor.yaml}
exp_root=${EXP_ROOT:-${ROOT_PATH}/exp_results/stream_results}
processor_name=${PROCESSOR_NAME:-SimPOTrans-fixed-dialogue}
run_omnisteval=${RUN_OMNISTEVAL:-1}

case "${device_arg}" in
    cpu|cuda|cuda:*)
        runtime_device=${device_arg}
        ;;
    *)
        export CUDA_VISIBLE_DEVICES=${device_arg}
        runtime_device=cuda
        ;;
esac

case "${tgt_lang}" in
    zh|zh-cn|zh-tw)
        latency_unit=char
        ;;
    ja)
        latency_unit=char
        ;;
    *)
        latency_unit=word
        ;;
esac

eval_data_path=${ROOT_PATH}/data/evaluation_data/${src_lang}-${tgt_lang}/${eval_data_name}
wav_list=${eval_data_path}/src.txt

omnisteval_result_dirs=()
typeset -A omnisteval_result_dirs_seen

record_omnisteval_result_dir() {
    local result_dir=$1
    if [[ -z "${omnisteval_result_dirs_seen[${result_dir}]:-}" ]]; then
        omnisteval_result_dirs_seen[${result_dir}]=1
        omnisteval_result_dirs+=("${result_dir}")
    fi
}

run_one_chunk() {
    local chunk_size=$1
    local chunk_output_name=${output_name:-chunk${chunk_size}}
    local out_path=${exp_root}/${src_lang}-${tgt_lang}/${eval_data_name}/${chunk_output_name}/chunk${chunk_size}
    mkdir -p "${out_path}"

for required_file in "${base_config}" "${model_config}" "${inference_prompt}" "${wav_list}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "Missing required file: ${required_file}" >&2
        exit 1
    fi
done

if [[ -n "${ckpt_path}" && "${ckpt_path}" != "none" && "${ckpt_path}" != "null" && ! -e "${ckpt_path}" ]]; then
    echo "Checkpoint does not exist: ${ckpt_path}" >&2
    exit 1
fi

export PYTHONPATH=${ROOT_PATH}/src:${PYTHONPATH:-}
export UV_CACHE_DIR=${ROOT_PATH}/.uv-cache
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/simpotrans-matplotlib}
mkdir -p "${MPLCONFIGDIR}"
export SRC_LANG=${src_lang}
export TGT_LANG=${tgt_lang}
export RUNTIME_DEVICE=${runtime_device}
export CHUNK_SIZE=${chunk_size}
export HISTORY_SELECTION_POLICY=${history_selection_policy}
export MAX_SPEECH_CONTEXT_SECONDS=${max_speech_context_seconds}
export SOFT_SPEECH_CONTEXT_RESET_SECONDS=${soft_speech_context_reset_seconds}
export HISTORY_SELECTION_FIXED_WORDS=${fixed_history_words}
export SPEECH_CONTEXT_POLICY=${speech_context_policy}
export LLM_RAW_KV_CACHE=${llm_raw_kv_cache}
export LLM_CONTEXT_TURNS=${runtime_llm_context_turns}
export DECODE_MODE=${decode_mode}
export BEAM_SIZE=${beam_size}
export DO_SAMPLE=${do_sample}
export TEMPERATURE=${temperature}
export TOP_P=${top_p}
export TOP_K=${top_k}
export MAX_NEW_TOKENS=${max_new_tokens}
export MODEL_CONFIG=${model_config}
export INFERENCE_PROMPT=${inference_prompt}
export SPEECH_ENCODER_BACKEND=${SPEECH_ENCODER_BACKEND:-}
export WHISPER_RT_MODEL=${WHISPER_RT_MODEL:-}
export WHISPER_RT_CHUNK_SIZE_MS=${WHISPER_RT_CHUNK_SIZE_MS:-}
export WHISPER_RT_MULTILINGUAL=${WHISPER_RT_MULTILINGUAL:-}
export CKPT_PATH=${ckpt_path}
export LATENCY_UNIT=${latency_unit}

runtime_config=${out_path}/${chunk_output_name}.processor.yaml
python - "${base_config}" "${runtime_config}" <<'PY'
import os
import sys
from pathlib import Path

import yaml

base_config, runtime_config = sys.argv[1:]
with open(base_config, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f) or {}

config["source_language"] = os.environ["SRC_LANG"]
config["target_language"] = os.environ["TGT_LANG"]
config["device"] = os.environ["RUNTIME_DEVICE"]
config["speech_chunk_size"] = float(os.environ["CHUNK_SIZE"])
config["history_selection_policy"] = os.environ["HISTORY_SELECTION_POLICY"]
config["max_speech_context_seconds"] = float(os.environ["MAX_SPEECH_CONTEXT_SECONDS"])
config["soft_speech_context_reset_seconds"] = float(
    os.environ["SOFT_SPEECH_CONTEXT_RESET_SECONDS"]
)
config["fixed_history_words"] = int(os.environ["HISTORY_SELECTION_FIXED_WORDS"])
config["speech_context_policy"] = os.environ["SPEECH_CONTEXT_POLICY"]
config["llm_raw_kv_cache"] = os.environ["LLM_RAW_KV_CACHE"].lower() in {
    "1", "true", "yes", "y"
}
llm_context_turns = os.environ["LLM_CONTEXT_TURNS"]
config["llm_context_turns"] = (
    None if llm_context_turns.lower() in {"", "none", "null"}
    else int(llm_context_turns)
)
config["model_config"] = os.environ["MODEL_CONFIG"]
model_overrides = {"prompt_path": os.environ["INFERENCE_PROMPT"]}
for env_name, key, caster in (
    ("SPEECH_ENCODER_BACKEND", "speech_encoder_backend", str),
    ("WHISPER_RT_MODEL", "whisper_rt_model", str),
    ("WHISPER_RT_CHUNK_SIZE_MS", "whisper_rt_chunk_size_ms", int),
    ("WHISPER_RT_MULTILINGUAL", "whisper_rt_multilingual", lambda value: value.lower() in {"1", "true", "yes", "y"}),
):
    value = os.environ.get(env_name, "")
    if value.lower() not in {"", "none", "null"}:
        model_overrides[key] = caster(value)
config["model_config_overrides"] = model_overrides
config["detokenizer_type"] = "simuleval"
config["latency_unit"] = os.environ["LATENCY_UNIT"]
decode = dict(config.get("decode") or {})
decode["beam_num"] = int(os.environ["BEAM_SIZE"])
decode["do_sample"] = os.environ["DO_SAMPLE"].lower() in {"1", "true", "yes", "y"}
decode["temperature"] = float(os.environ["TEMPERATURE"])
decode["max_new_tokens"] = int(os.environ["MAX_NEW_TOKENS"])
for env_name, key in (("TOP_P", "top_p"), ("TOP_K", "top_k")):
    value = os.environ[env_name]
    if value.lower() in {"", "none", "null"}:
        decode[key] = None
    elif key == "top_k":
        decode[key] = int(value)
    else:
        decode[key] = float(value)
config["decode"] = decode

checkpoint = os.environ.get("CKPT_PATH", "")
config["checkpoint"] = None if checkpoint in {"", "none", "null"} else checkpoint

Path(runtime_config).parent.mkdir(parents=True, exist_ok=True)
with open(runtime_config, "w", encoding="utf-8") as f:
    yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
PY

metrics_log=${out_path}/${chunk_output_name}.jsonl
runtime_wav_list=${out_path}/${chunk_output_name}.wav_list.txt

# SimulStream 0.3.0 resolves every wav-list entry relative to the list file
# directory, even when the entry is already absolute. Keep the source manifest
# untouched and write a runtime manifest with paths relative to out_path.
python - "${wav_list}" "${runtime_wav_list}" <<'PY'
import os
import sys
from pathlib import Path

src_manifest, dst_manifest = sys.argv[1:]
dst_path = Path(dst_manifest)
dst_path.parent.mkdir(parents=True, exist_ok=True)

with open(src_manifest, "r", encoding="utf-8") as f:
    wav_paths = [line.strip() for line in f if line.strip()]

if not wav_paths:
    raise SystemExit(f"No wav paths found in {src_manifest}")

base_dir = dst_path.parent
runtime_entries = []
missing = []
for wav_path in wav_paths:
    abs_path = Path(wav_path)
    if not abs_path.is_absolute():
        abs_path = Path(src_manifest).parent / abs_path
    abs_path = abs_path.resolve()
    if not abs_path.is_file():
        missing.append(str(abs_path))
    runtime_entries.append(os.path.relpath(abs_path, base_dir))

if missing:
    raise SystemExit("Missing wav file(s):\n" + "\n".join(missing))

with open(dst_path, "w", encoding="utf-8") as f:
    f.write("\n".join(runtime_entries) + "\n")
PY

echo "------------------------------------------------"
echo "SimulStream SimPOTrans ${processor_name} evaluation"
echo "src=${src_lang}, tgt=${tgt_lang}, device=${runtime_device}, chunk=${chunk_size}s"
echo "history_selection_policy=${history_selection_policy}"
echo "max_speech_context=${max_speech_context_seconds}s"
echo "soft_speech_context_reset=${soft_speech_context_reset_seconds}s"
echo "fixed_history_words=${fixed_history_words}, speech_context_policy=${speech_context_policy}, llm_raw_kv_cache=${llm_raw_kv_cache}, llm_context_turns=${runtime_llm_context_turns}"
echo "decode=${decode_mode}, sample=${do_sample}, beam=${beam_size}, temp=${temperature}, top_p=${top_p}, top_k=${top_k}, max_new=${max_new_tokens}"
echo "data=${eval_data_path}"
echo "config=${runtime_config}"
echo "wav_list=${runtime_wav_list}"
echo "output=${out_path}"

echo "Step 1: Inference"
uv run simulstream_inference \
    --speech-processor-config "${runtime_config}" \
    --wav-list-file "${runtime_wav_list}" \
    --tgt-lang "${tgt_lang}" \
    --src-lang "${src_lang}" \
    --metrics-log-file "${metrics_log}"

if [[ ! -s "${metrics_log}" ]]; then
    echo "Inference did not create a non-empty metrics log: ${metrics_log}" >&2
    exit 1
fi

python - "${metrics_log}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
tmp_path = path.with_suffix(path.suffix + ".tmp")

with path.open("r", encoding="utf-8") as src, tmp_path.open("w", encoding="utf-8") as dst:
    for line in src:
        if not line.strip():
            continue
        dst.write(json.dumps(json.loads(line), ensure_ascii=False) + "\n")

tmp_path.replace(path)
PY

echo "Done. Logs:"
echo "  metrics: ${metrics_log}"
record_omnisteval_result_dir "${out_path:h}"
}

for chunk_size in "${chunk_sizes[@]}"; do
    run_one_chunk "${chunk_size}"
done

if [[ "${run_omnisteval}" == "1" ]]; then
    if (( ${#omnisteval_result_dirs} == 0 )); then
        echo "No stream inference result directories were collected for OmniSTEval." >&2
        exit 1
    fi
    for result_dir in "${omnisteval_result_dirs[@]}"; do
        zsh "${SCRIPT_DIR}/run_omnisteval_longform.sh" "${result_dir}" "${src_lang}" "${tgt_lang}" "${eval_data_name}" "${device_arg}"
    done
elif [[ "${run_omnisteval}" != "0" ]]; then
    echo "Unsupported RUN_OMNISTEVAL=${run_omnisteval}. Use 1 or 0." >&2
    exit 2
else
    echo "OmniSTEval longform skipped (RUN_OMNISTEVAL=${run_omnisteval})."
fi
