#!/usr/bin/env zsh
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  zsh src/simul_eval/eval_fixed.sh <src_lang> <tgt_lang> <device> <eval_data> [checkpoint] [exp_name]

Example:
  END_INDEX=10 zsh src/simul_eval/eval_fixed.sh en zh 0 simul_eval_acl checkpoints/covost/st_backbone_covost/ws_2.f3820fc3/checkpoints/step_6000

Environment overrides:
  OFFLINE=1  # source segment size becomes 1000000 ms and output point is offline
  CHUNK_SIZES="1.0 1.5 2.0 3.0 4.0 5.0"
  # Or set CHUNK_SIZE=1.0 to run one chunk size. Ignored when OFFLINE=1.
  END_INDEX=10
  HISTORY_SELECTION_POLICY=none
  MAX_SPEECH_CONTEXT_SECONDS=30.0
  SOFT_SPEECH_CONTEXT_RESET_SECONDS=20.0
  FIXED_HISTORY_WORDS=100
  SPEECH_CONTEXT_POLICY=fixed_duration
  LLM_CONTEXT_TURNS=null
  LLM_RAW_KV_CACHE=0
  DECODE=beam  # Set DECODE=sample for sampling decoding.
  BEAM_SIZE=4
  DO_SAMPLE=0
  TEMPERATURE=0.0
  TOP_P=null
  TOP_K=null
  MAX_NEW_TOKENS=128
  MODEL_CONFIG=src/st_backbone/config.yaml
  INFERENCE_PROMPT=src/st_backbone/st_prompt.json
  EXP_ROOT=exp_results/simuleval_results
  EXP_NAME=  # optional; defaults to POLICY_NAME when unset
  POLICY_NAME=fixed-dialogue
  RUN_COMET=1
  COMET_MODEL=Unbabel/XCOMET-XL
  COMET_SOURCE_TEXT=data/acl6060_src_text.txt
  RUN_OMNISTEVAL=1
  OMNISTEVAL_BIN=OmniSTEval/.venv/bin/omnisteval
EOF
}

if [[ $# -eq 0 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

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
exp_name_arg=${6:-${EXP_NAME:-}}
policy_name=${POLICY_NAME:-fixed-dialogue}

offline=${OFFLINE:-0}
case "${offline}" in
    1|true|yes|y)
        offline=1
        chunk_sizes=(offline)
        ;;
    0|false|no|n)
        offline=0
        if [[ -n "${CHUNK_SIZES:-}" ]]; then
            chunk_sizes=("${(@s: :)CHUNK_SIZES}")
        elif [[ -n "${CHUNK_SIZE:-}" ]]; then
            chunk_sizes=("${CHUNK_SIZE}")
        else
            chunk_sizes=(1.0 1.5 2.0 3.0 4.0 5.0)
        fi
        ;;
    *)
        echo "Unsupported OFFLINE=${offline}. Use 1 or 0." >&2
        exit 2
        ;;
esac

run_name=${exp_name_arg:-${policy_name}}

history_selection_policy=${HISTORY_SELECTION_POLICY:-none}
history_selection_policy=${(L)history_selection_policy}
max_speech_context_seconds=${MAX_SPEECH_CONTEXT_SECONDS:-30.0}
soft_speech_context_reset_seconds=${SOFT_SPEECH_CONTEXT_RESET_SECONDS:-20.0}
fixed_history_words=${FIXED_HISTORY_WORDS:-${HISTORY_SELECTION_FIXED_WORDS:-100}}
speech_context_policy=${SPEECH_CONTEXT_POLICY:-fixed_duration}
llm_context_turns=${LLM_CONTEXT_TURNS:-null}
llm_raw_kv_cache=${LLM_RAW_KV_CACHE:-0}

decode_mode=${DECODE:-${DECODE_MODE:-beam}}
case "${decode_mode}" in
    beam)
        beam_size=${BEAM_SIZE:-4}
        do_sample=${DO_SAMPLE:-0}
        temperature=${TEMPERATURE:-0.0}
        top_p=${TOP_P:-null}
        top_k=${TOP_K:-null}
        ;;
    sample)
        beam_size=${BEAM_SIZE:-1}
        do_sample=${DO_SAMPLE:-1}
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
exp_root=${EXP_ROOT:-${ROOT_PATH}/exp_results/simuleval_results}
end_index=${END_INDEX:-}
run_comet=${RUN_COMET:-1}
comet_model=${COMET_MODEL:-Unbabel/XCOMET-XL}
comet_source_text=${COMET_SOURCE_TEXT:-${ROOT_PATH}/data/acl6060_src_text.txt}
run_omnisteval=${RUN_OMNISTEVAL:-1}
omnisteval_bin=${OMNISTEVAL_BIN:-${ROOT_PATH}/OmniSTEval/.venv/bin/omnisteval}

if [[ "${model_config}" != /* ]]; then
    model_config=${ROOT_PATH}/${model_config}
fi
if [[ "${inference_prompt}" != /* ]]; then
    inference_prompt=${ROOT_PATH}/${inference_prompt}
fi
if [[ "${comet_source_text}" != /* ]]; then
    comet_source_text=${ROOT_PATH}/${comet_source_text}
fi
if [[ "${omnisteval_bin}" != /* ]]; then
    omnisteval_bin=${ROOT_PATH}/${omnisteval_bin}
fi

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
        sacrebleu_tokenizer=zh
        omnisteval_level_arg=(--char_level)
        ;;
    ja)
        latency_unit=char
        sacrebleu_tokenizer=ja-mecab
        omnisteval_level_arg=(--char_level)
        ;;
    ko)
        latency_unit=char
        sacrebleu_tokenizer=ko-mecab
        omnisteval_level_arg=(--char_level)
        ;;
    *)
        latency_unit=word
        sacrebleu_tokenizer=13a
        omnisteval_level_arg=(--word_level)
        ;;
esac

eval_data_path=${ROOT_PATH}/data/evaluation_data/${src_lang}-${tgt_lang}/${eval_data_name}
wav_list=${eval_data_path}/source.txt
ref_list=${eval_data_path}/target.txt

for required_file in "${model_config}" "${inference_prompt}" "${wav_list}" "${ref_list}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "Missing required file: ${required_file}" >&2
        exit 1
    fi
done

if [[ -n "${ckpt_path}" && "${ckpt_path}" != "none" && "${ckpt_path}" != "null" && ! -e "${ckpt_path}" ]]; then
    echo "Checkpoint does not exist: ${ckpt_path}" >&2
    exit 1
fi

if [[ -n "${end_index}" && ! "${end_index}" =~ '^[1-9][0-9]*$' ]]; then
    echo "END_INDEX must be a positive integer, got: ${end_index}" >&2
    exit 2
fi

if [[ "${run_comet}" == "1" && ! -f "${comet_source_text}" ]]; then
    echo "Missing COMET source text file: ${comet_source_text}" >&2
    echo "Set COMET_SOURCE_TEXT=/path/to/source_text.txt or RUN_COMET=0." >&2
    exit 1
fi

if [[ "${run_omnisteval}" == "1" && ! -x "${omnisteval_bin}" ]]; then
    echo "OmniSTEval CLI not found or not executable: ${omnisteval_bin}" >&2
    echo "Install the OmniSTEval environment or set OMNISTEVAL_BIN=/path/to/omnisteval." >&2
    exit 1
fi

prepare_runtime_inputs() {
    local runtime_dir=$1
    local runtime_source=${runtime_dir}/source.txt
    local runtime_target=${runtime_dir}/target.txt
    local runtime_source_text=${runtime_dir}/source_text.txt

    mkdir -p "${runtime_dir}"
    if [[ -n "${end_index}" ]]; then
        python3 - "${end_index}" "${wav_list}" "${ref_list}" "${comet_source_text}" "${runtime_source}" "${runtime_target}" "${runtime_source_text}" <<'MANIFEST_PY'
from pathlib import Path
import sys

end_index = int(sys.argv[1])
source_path = Path(sys.argv[2])
target_path = Path(sys.argv[3])
source_text_path = Path(sys.argv[4])
runtime_source = Path(sys.argv[5])
runtime_target = Path(sys.argv[6])
runtime_source_text = Path(sys.argv[7])

def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()

def write_first(lines: list[str], path: Path, label: str) -> None:
    if len(lines) < end_index:
        raise SystemExit(f"{label} has {len(lines)} line(s), fewer than END_INDEX={end_index}: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines[:end_index]) + "\n", encoding="utf-8")

write_first(read_lines(source_path), runtime_source, "source.txt")
write_first(read_lines(target_path), runtime_target, "target.txt")
if source_text_path.is_file():
    write_first(read_lines(source_text_path), runtime_source_text, "COMET source text")
else:
    runtime_source_text.write_text("", encoding="utf-8")
MANIFEST_PY
    else
        ln -sf "${wav_list:A}" "${runtime_source}"
        ln -sf "${ref_list:A}" "${runtime_target}"
        ln -sf "${comet_source_text:A}" "${runtime_source_text}"
    fi

    echo "${runtime_source}|${runtime_target}|${runtime_source_text}"
}

export PYTHONPATH=${ROOT_PATH}/src:${PYTHONPATH:-}
export UV_CACHE_DIR=${ROOT_PATH}/.uv-cache
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/simpotrans-matplotlib}
mkdir -p "${MPLCONFIGDIR}"

postprocess_out_paths=()
postprocess_ref_lists=()
postprocess_source_texts=()

for chunk_size in "${chunk_sizes[@]}"; do
    if [[ -z "${chunk_size}" ]]; then
        continue
    fi

    point_name=${chunk_size}
    if [[ "${offline}" == "1" ]]; then
        point_name=offline
    fi
    out_path=${exp_root}/${src_lang}-${tgt_lang}/${run_name}/${point_name}
    mkdir -p "${out_path}"

    runtime_inputs=$(prepare_runtime_inputs "${out_path}/eval_data")
    runtime_wav_list=${runtime_inputs%%|*}
    rest=${runtime_inputs#*|}
    runtime_ref_list=${rest%%|*}
    runtime_comet_source_text=${rest#*|}

    if [[ "${offline}" == "1" ]]; then
        chunk_ms=1000000
    else
        chunk_ms=$(python3 - "${chunk_size}" <<'CHUNK_PY'
import sys
print(max(1, int(round(float(sys.argv[1]) * 1000))))
CHUNK_PY
)
    fi

    cmd=(
        uv run simuleval
        --agent "${SCRIPT_DIR}/fixed_policy.py"
        --source "${runtime_wav_list}"
        --target "${runtime_ref_list}"
        --source-type speech
        --target-type text
        --source-segment-size "${chunk_ms}"
        --output "${out_path}"
        --device "${runtime_device}"
        --eval-latency-unit "${latency_unit}"
        --no-scoring
        --model-config "${model_config}"
        --source-language "${src_lang}"
        --target-language "${tgt_lang}"
        --prompt-path "${inference_prompt}"
        --history-selection-policy "${history_selection_policy}"
        --max-speech-context-seconds "${max_speech_context_seconds}"
        --soft-speech-context-reset-seconds "${soft_speech_context_reset_seconds}"
        --fixed-history-words "${fixed_history_words}"
        --speech-context-policy "${speech_context_policy}"
        --beam-size "${beam_size}"
        --max-new-tokens "${max_new_tokens}"
        --temperature "${temperature}"
    )

    if [[ "${llm_context_turns}" != "" && "${llm_context_turns}" != "none" && "${llm_context_turns}" != "null" ]]; then
        cmd+=(--llm-context-turns "${llm_context_turns}")
    fi
    if [[ "${llm_raw_kv_cache}" == "1" || "${llm_raw_kv_cache}" == "true" || "${llm_raw_kv_cache}" == "yes" ]]; then
        cmd+=(--llm-raw-kv-cache)
    fi
    if [[ "${do_sample}" == "1" || "${do_sample}" == "true" || "${do_sample}" == "yes" ]]; then
        cmd+=(--do-sample)
    fi
    if [[ "${top_p}" != "" && "${top_p}" != "none" && "${top_p}" != "null" ]]; then
        cmd+=(--top-p "${top_p}")
    fi
    if [[ "${top_k}" != "" && "${top_k}" != "none" && "${top_k}" != "null" ]]; then
        cmd+=(--top-k "${top_k}")
    fi
    if [[ -n "${ckpt_path}" && "${ckpt_path}" != "none" && "${ckpt_path}" != "null" ]]; then
        cmd+=(--checkpoint "${ckpt_path}")
    fi

    echo "------------------------------------------------"
    echo "SimulEval SimPOTrans ${run_name} evaluation"
    echo "src=${src_lang}, tgt=${tgt_lang}, device=${runtime_device}, chunk=${point_name} (${chunk_ms}ms)"
    echo "history_selection_policy=${history_selection_policy}, offline=${offline}, end_index=${end_index:-all}"
    echo "decode=${decode_mode}, sample=${do_sample}, beam=${beam_size}, temp=${temperature}, top_p=${top_p}, top_k=${top_k}, max_new=${max_new_tokens}"
    echo "latency_unit=${latency_unit}, sacrebleu_tokenizer=${sacrebleu_tokenizer}"
    echo "data=${eval_data_path}"
    echo "source=${runtime_wav_list}"
    echo "target=${runtime_ref_list}"
    echo "source_text=${runtime_comet_source_text}"
    echo "run_name=${run_name}"
    echo "output=${out_path}"
    echo "------------------------------------------------"

    "${cmd[@]}"

    instances_log=${out_path}/instances.log
    if [[ -f "${instances_log}" ]]; then
        python3 - "${instances_log}" <<'JSON_PY'
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
JSON_PY

        postprocess_out_paths+=("${out_path}")
        postprocess_ref_lists+=("${runtime_ref_list}")
        postprocess_source_texts+=("${runtime_comet_source_text}")
    else
        echo "Missing SimulEval instances log: ${instances_log}" >&2
        exit 1
    fi
done

if (( ${#postprocess_out_paths} == 0 )); then
    echo "No SimulEval outputs were collected for post-processing." >&2
    exit 1
fi

if [[ "${run_omnisteval}" == "1" ]]; then
    echo "------------------------------------------------"
    echo "Step 2: OmniSTEval shortform for all SimulEval outputs"
    echo "num_outputs=${#postprocess_out_paths}"
    echo "------------------------------------------------"

    for (( index = 1; index <= ${#postprocess_out_paths}; index++ )); do
        out_path=${postprocess_out_paths[index]}
        runtime_ref_list=${postprocess_ref_lists[index]}
        runtime_comet_source_text=${postprocess_source_texts[index]}
        instances_log=${out_path}/instances.log
        omnisteval_output=${out_path}/omnisteval_shortform

        if [[ ! -f "${instances_log}" ]]; then
            echo "Missing SimulEval instances log for OmniSTEval: ${instances_log}" >&2
            exit 1
        fi

        omnisteval_cmd=(
            "${omnisteval_bin}" shortform
            --hypothesis_file "${instances_log}"
            --ref_sentences_file "${runtime_ref_list}"
            --output_folder "${omnisteval_output}"
            --fix_simuleval_emission_ca
            --bleu_tokenizer "${sacrebleu_tokenizer}"
            "${omnisteval_level_arg[@]}"
        )

        if [[ "${run_comet}" == "1" ]]; then
            omnisteval_cmd+=(
                --comet
                --comet_model "${comet_model}"
                --source_sentences_file "${runtime_comet_source_text}"
            )
        fi

        echo "------------------------------------------------"
        echo "OmniSTEval shortform ${index}/${#postprocess_out_paths}"
        echo "hypothesis=${instances_log}"
        echo "reference=${runtime_ref_list}"
        echo "source_text=${runtime_comet_source_text}"
        echo "output=${omnisteval_output}"
        echo "------------------------------------------------"
        "${omnisteval_cmd[@]}"
    done
else
    echo "OmniSTEval shortform skipped (RUN_OMNISTEVAL=${run_omnisteval})."
fi
