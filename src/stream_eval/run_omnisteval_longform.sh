#!/usr/bin/env zsh
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  zsh src/stream_eval/run_omnisteval_longform.sh <result_dir> [src_lang] [tgt_lang] [eval_data] [device]

Example:
  zsh src/stream_eval/run_omnisteval_longform.sh exp_results/stream_results/en-zh/simulsense_comet-mono-rep
  zsh src/stream_eval/run_omnisteval_longform.sh exp_results/stream_results/en-zh/simulsense_comet-mono-rep en zh stream_eval_acl cpu

Environment overrides:
  OMNISTEVAL_BIN=OmniSTEval/.venv/bin/omnisteval
  RUN_COMET=1
  COMET_MODEL=Unbabel/XCOMET-XL
  RUN_METRICX=1
  METRICX_MODEL=google/metricx-24-hybrid-xxl-v2p6-bfloat16
  METRICX_BATCH_SIZE=1
  CHAR_LEVEL=auto
  OFFSET_DELAYS=0

The script evaluates every direct subdirectory under result_dir with:
  omnisteval longform --hypothesis_format simulstream

Each experiment point writes outputs under:
  <result_dir>/<point>/omnisteval_longform/
EOF
}

if [[ $# -eq 0 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ $# -lt 1 || $# -gt 5 ]]; then
    usage
    exit 2
fi

SCRIPT_PATH=${(%):-%x}
SCRIPT_DIR=${SCRIPT_PATH:A:h}
ROOT_PATH=$(cd "${SCRIPT_DIR}/../.." && pwd)

result_arg=$1
if [[ "${result_arg}" = /* ]]; then
    result_dir=${result_arg:A}
else
    result_dir=${ROOT_PATH}/${result_arg}
    result_dir=${result_dir:A}
fi

if [[ ! -d "${result_dir}" ]]; then
    echo "Result directory does not exist: ${result_dir}" >&2
    exit 1
fi

infer_lang_pair() {
    local path=$1
    local -a parts
    parts=("${(@s:/:)path}")

    local part
    for part in "${parts[@]}"; do
        if [[ "${part}" == *-* && -d "${ROOT_PATH}/data/evaluation_data/${part}" ]]; then
            echo "${part}"
            return 0
        fi
    done

    return 1
}

lang_pair=$(infer_lang_pair "${result_dir}" || true)
src_lang=${2:-}
tgt_lang=${3:-}

if [[ -z "${src_lang}" || -z "${tgt_lang}" ]]; then
    if [[ -z "${lang_pair}" ]]; then
        echo "Could not infer src_lang/tgt_lang from result path. Pass them explicitly." >&2
        exit 1
    fi
    src_lang=${src_lang:-${lang_pair%%-*}}
    tgt_lang=${tgt_lang:-${lang_pair#*-}}
fi

eval_data=${4:-stream_eval_acl}
device=${5:-cpu}

omnisteval_bin=${OMNISTEVAL_BIN:-${ROOT_PATH}/OmniSTEval/.venv/bin/omnisteval}
if [[ "${omnisteval_bin}" != /* ]]; then
    omnisteval_bin=${ROOT_PATH}/${omnisteval_bin}
fi
omnisteval_bin=${omnisteval_bin:A}

if [[ ! -x "${omnisteval_bin}" ]]; then
    echo "OmniSTEval CLI not found or not executable: ${omnisteval_bin}" >&2
    echo "Install the isolated environment first, or set OMNISTEVAL_BIN=/path/to/omnisteval." >&2
    exit 1
fi

case "${device}" in
    cpu)
        export CUDA_VISIBLE_DEVICES=""
        ;;
    <->)
        export CUDA_VISIBLE_DEVICES=${device}
        ;;
    cuda|cuda:*)
        ;;
    *)
        echo "Unsupported device '${device}'. Use cpu, a GPU id like 0, cuda, or cuda:<id>." >&2
        exit 1
        ;;
esac

eval_data_path=${ROOT_PATH}/data/evaluation_data/${src_lang}-${tgt_lang}/${eval_data}
segmentation_yaml_files=("${eval_data_path}"/*.yaml(N.))
if (( ${#segmentation_yaml_files[@]} != 1 )); then
    echo "Expected exactly one speech segmentation *.yaml in ${eval_data_path}, found ${#segmentation_yaml_files[@]}." >&2
    exit 1
fi

speech_segmentation=${segmentation_yaml_files[1]}
ref_sentences_file=${eval_data_path}/tgt_segments.txt
source_sentences_file=${eval_data_path}/src_segments.txt

for required_file in "${speech_segmentation}" "${ref_sentences_file}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "Missing required file: ${required_file}" >&2
        exit 1
    fi
done

run_comet=${RUN_COMET:-1}
run_metricx=${RUN_METRICX:-1}
if [[ ( "${run_comet}" == "1" || "${run_metricx}" == "1" ) && ! -f "${source_sentences_file}" ]]; then
    echo "Missing required source file for COMET/MetricX: ${source_sentences_file}" >&2
    echo "Set RUN_COMET=0 and RUN_METRICX=0 to run without source-aware metrics." >&2
    exit 1
fi
if [[ "${run_metricx}" != "0" && "${run_metricx}" != "1" ]]; then
    echo "Unsupported RUN_METRICX='${run_metricx}'. Use 1 or 0." >&2
    exit 1
fi

case "${tgt_lang}" in
    zh|zh-cn|zh-tw)
        bleu_tokenizer=zh
        default_level=char
        lang_arg=()
        ;;
    ja)
        bleu_tokenizer=ja-mecab
        default_level=char
        lang_arg=()
        ;;
    *)
        bleu_tokenizer=13a
        default_level=word
        lang_arg=(--lang "${tgt_lang}")
        ;;
esac

char_level=${CHAR_LEVEL:-auto}
case "${char_level}" in
    auto)
        if [[ "${default_level}" == "char" ]]; then
            level_arg=(--char_level)
        else
            level_arg=(--word_level)
        fi
        ;;
    1|true|yes|y)
        level_arg=(--char_level)
        ;;
    0|false|no|n)
        level_arg=(--word_level)
        ;;
    *)
        echo "Unsupported CHAR_LEVEL='${char_level}'. Use auto, 1, or 0." >&2
        exit 1
        ;;
esac

offset_delays=${OFFSET_DELAYS:-0}
case "${offset_delays}" in
    1|true|yes|y)
        offset_arg=(--offset_delays)
        ;;
    0|false|no|n)
        offset_arg=()
        ;;
    *)
        echo "Unsupported OFFSET_DELAYS='${offset_delays}'. Use 1 or 0." >&2
        exit 1
        ;;
esac

comet_model=${COMET_MODEL:-Unbabel/XCOMET-XL}
metric_args=(--bleu_tokenizer "${bleu_tokenizer}")
if [[ "${run_comet}" == "1" ]]; then
    metric_args+=(--comet --source_sentences_file "${source_sentences_file}" --comet_model "${comet_model}")
elif [[ "${run_comet}" != "0" ]]; then
    echo "Unsupported RUN_COMET='${run_comet}'. Use 1 or 0." >&2
    exit 1
fi

point_dirs=("${result_dir}"/*(N/))
if (( ${#point_dirs[@]} == 0 )); then
    echo "No experiment point directories found under: ${result_dir}" >&2
    echo "Expected direct subdirectories containing one non-empty *.jsonl and one processor yaml (*.processor.yaml or processor.yaml)." >&2
    exit 1
fi

echo "OmniSTEval longform evaluation"
echo "result_dir=${result_dir}"
echo "src=${src_lang}, tgt=${tgt_lang}, eval_data=${eval_data}, device=${device}"
echo "speech_segmentation=${speech_segmentation}"
echo "ref_sentences_file=${ref_sentences_file}"
echo "run_comet=${run_comet}"
echo "run_metricx=${run_metricx}"
if [[ "${run_comet}" == "1" ]]; then
    echo "comet_model=${comet_model}"
fi
echo
for point_dir in "${point_dirs[@]}"; do
    jsonl_files=("${point_dir}"/*.jsonl(N.))
    yaml_files=("${point_dir}"/*.processor.yaml(N.) "${point_dir}"/processor.yaml(N.))

    nonempty_jsonl_files=()
    local_file=
    for local_file in "${jsonl_files[@]}"; do
        if [[ -s "${local_file}" ]]; then
            nonempty_jsonl_files+=("${local_file}")
        fi
    done

    if (( ${#nonempty_jsonl_files[@]} != 1 )); then
        echo "Expected exactly one non-empty *.jsonl in ${point_dir}, found ${#nonempty_jsonl_files[@]}." >&2
        exit 1
    fi

    if (( ${#yaml_files[@]} != 1 )); then
        echo "Expected exactly one processor yaml (*.processor.yaml or processor.yaml) in ${point_dir}, found ${#yaml_files[@]}." >&2
        exit 1
    fi

    hypothesis_file=${nonempty_jsonl_files[1]}
    simulstream_config_file=${yaml_files[1]}
    output_folder=${point_dir}/omnisteval_longform

    echo "------------------------------------------------"
    echo "point=${point_dir:t}"
    echo "hypothesis_file=${hypothesis_file}"
    echo "simulstream_config_file=${simulstream_config_file}"
    echo "output_folder=${output_folder}"

    "${omnisteval_bin}" longform \
        --speech_segmentation "${speech_segmentation}" \
        --ref_sentences_file "${ref_sentences_file}" \
        --hypothesis_file "${hypothesis_file}" \
        --simulstream_config_file "${simulstream_config_file}" \
        --hypothesis_format simulstream \
        "${(@)lang_arg}" \
        "${(@)level_arg}" \
        "${(@)offset_arg}" \
        "${(@)metric_args}" \
        --output_folder "${output_folder}"
done

echo "Done. OmniSTEval outputs were written under each experiment point's omnisteval_longform directory."

if [[ "${run_metricx}" == "1" ]]; then
    METRICX_SOURCE_SENTENCES_FILE="${source_sentences_file}" \
        bash "${ROOT_PATH}/scripts/evaluate_metricx.sh" \
            "${result_dir}" "${device}"
else
    echo "MetricX-24 skipped (RUN_METRICX=${run_metricx})."
fi
