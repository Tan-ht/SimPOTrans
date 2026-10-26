#!/usr/bin/env zsh
emulate -L zsh
setopt err_exit nounset pipe_fail

usage() {
  cat <<'EOF'
Usage:
  uv run zsh train_multilingual.sh <devices|cpu> <wandb_name> <train_split> <valid_split> <source>... [master_port]

Source format:
  <dataset>:<src_lang>-<tgt_lang>[:manifest_root[:audio_root[:train_split[:valid_split]]]]

Examples:
  uv run zsh train_multilingual.sh 0,1 st_multi train_st_asr_filtered dev_acl6060 \
    covost:en-zh \
    must-c:en-fr \
    must-c:en-nl

  uv run zsh train_multilingual.sh 0 st_multi train_st dev_st \
    covost:ja-en:/path/to/data/covost/ja-en

Optional environment variables:
  SAMPLING_ALPHA                      Multilingual sampling alpha. Default: 0.5
  SAMPLING_UNIT                       lang_pair or source. Default: lang_pair
  MAX_VALIDATION_SENTENCES_PER_LANG_PAIR Validation cap per language pair. Default: 1000
  MAX_BLEU_SENTENCES_PER_LANG_PAIR    BLEU cap per language pair. Default: 1000
  VALIDATE_CHECKPOINT_EVERY_N_STEPS   Validation/checkpoint interval. Default: 2000
  MAX_NUM_FRAMES_PER_BATCH            Override dataset max_num_frames_per_batch. Default: 600000
  SPEECH_ENCODER_BACKEND              Optional: whisper or whisper_rt. Default: config value
  WHISPER_RT_MODEL                    Optional WhisperRT model size/checkpoint. Default: large-v2
  WHISPER_RT_CHUNK_SIZE_MS            Optional WhisperRT chunk size in milliseconds. Default: 300
  WHISPER_RT_MULTILINGUAL             Optional WhisperRT multilingual flag. Default: true
  FREEZE_ENCODER                      Optional override for model.config_overrides.freeze_encoder
EOF
}

if [[ $# -lt 5 ]]; then
  usage
  exit 1
fi

devices="$1"
wandb_name="$2"
TRAIN_SPLIT="$3"
VALID_SPLIT="$4"
shift 4

MASTER_PORT=""
if [[ "$#" -gt 0 && "${argv[-1]}" == <-> ]]; then
  MASTER_PORT="${argv[-1]}"
  argv[-1]=()
fi

ROOT_PATH="${ROOT_PATH:-${0:A:h}/../..}"
ROOT_PATH="${ROOT_PATH:A}"
CONFIG_FILE="${CONFIG_FILE:-${ROOT_PATH}/src/st_backbone/config.yaml}"
TRAIN_ENTRY="${TRAIN_ENTRY:-${ROOT_PATH}/src/st_backbone/train.py}"
WANDB_PROJECT="${WANDB_PROJECT:-SimPOTrans}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-${MASTER_PORT:-29500}}"
SAMPLING_ALPHA="${SAMPLING_ALPHA:-0.5}"
SAMPLING_UNIT="${SAMPLING_UNIT:-lang_pair}"
MAX_VALIDATION_SENTENCES_PER_LANG_PAIR="${MAX_VALIDATION_SENTENCES_PER_LANG_PAIR:-1000}"
MAX_BLEU_SENTENCES_PER_LANG_PAIR="${MAX_BLEU_SENTENCES_PER_LANG_PAIR:-1000}"
VALIDATE_CHECKPOINT_EVERY_N_STEPS="${VALIDATE_CHECKPOINT_EVERY_N_STEPS:-2000}"
MAX_NUM_FRAMES_PER_BATCH="${MAX_NUM_FRAMES_PER_BATCH:-600000}"
DATA_ROOT="${DATA_ROOT:-${ROOT_PATH}/data}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_PATH}/checkpoints/multilingual/${wandb_name}}"
MANIFEST_LIST="${MANIFEST_LIST:-${OUTPUT_DIR}/multilingual_sources.tsv}"

default_manifest_root() {
  local dataset="$1"
  local lang_pair="$2"
  local candidates

  case "${dataset}" in
    must-c|mustc)
      candidates=(
        "${DATA_ROOT}/mustc/${lang_pair}"
        "${DATA_ROOT}/must-c/${lang_pair}"
      )
      ;;
    *)
      candidates=("${DATA_ROOT}/${dataset}/${lang_pair}")
      ;;
  esac

  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -e "${candidate}" ]]; then
      print -r -- "${candidate}"
      return
    fi
  done

  print -r -- "${candidates[1]}"
}

mkdir -p "${OUTPUT_DIR}"
{
  print -r -- $'dataset\tsrc_lang\ttgt_lang\tmanifest_root\taudio_root\ttrain_split\tvalid_split'
  for source in "$@"; do
    parts=("${(@s.:.)source}")
    dataset="${parts[1]}"
    lang_pair="${parts[2]}"
    src_tgt=("${(@s.-.)lang_pair}")
    src_lang="${src_tgt[1]}"
    tgt_lang="${src_tgt[2]}"
    manifest_root="${parts[3]:-$(default_manifest_root "${dataset}" "${lang_pair}")}"
    audio_root="${parts[4]:-${manifest_root}}"
    train_split="${parts[5]:-}"
    valid_split="${parts[6]:-}"

    if [[ -z "${dataset}" || -z "${src_lang}" || -z "${tgt_lang}" ]]; then
      echo "Invalid source spec: ${source}" >&2
      exit 1
    fi

    print -r -- "${dataset}"$'\t'"${src_lang}"$'\t'"${tgt_lang}"$'\t'"${manifest_root}"$'\t'"${audio_root}"$'\t'"${train_split}"$'\t'"${valid_split}"
  done
} > "${MANIFEST_LIST}"

export TF_ENABLE_ONEDNN_OPTS="${TF_ENABLE_ONEDNN_OPTS:-0}"
export WANDB_MODE="${WANDB_MODE:-offline}"

encoder_args=()
if [[ -n "${SPEECH_ENCODER_BACKEND:-}" ]]; then
  encoder_args+=(--config "model.config_overrides.speech_encoder_backend='${SPEECH_ENCODER_BACKEND}'")
fi
if [[ -n "${WHISPER_RT_MODEL:-}" ]]; then
  encoder_args+=(--config "model.config_overrides.whisper_rt_model='${WHISPER_RT_MODEL}'")
fi
if [[ -n "${WHISPER_RT_CHUNK_SIZE_MS:-}" ]]; then
  encoder_args+=(--config "model.config_overrides.whisper_rt_chunk_size_ms=${WHISPER_RT_CHUNK_SIZE_MS}")
fi
if [[ -n "${WHISPER_RT_MULTILINGUAL:-}" ]]; then
  encoder_args+=(--config "model.config_overrides.whisper_rt_multilingual=${WHISPER_RT_MULTILINGUAL}")
fi
if [[ -n "${FREEZE_ENCODER:-}" ]]; then
  encoder_args+=(--config "model.config_overrides.freeze_encoder=${FREEZE_ENCODER}")
fi

common_args=(
  --config-file "${CONFIG_FILE}"
  "${OUTPUT_DIR}"
  --config "dataset.config_overrides.manifest_list='${MANIFEST_LIST}'"
  --config "dataset.config_overrides.splits='${TRAIN_SPLIT}'"
  --config "dataset.config_overrides.default_train_split='train_st'"
  --config "dataset.config_overrides.default_valid_split='dev_st'"
  --config "dataset.config_overrides.sampling_alpha=${SAMPLING_ALPHA}"
  --config "dataset.config_overrides.sampling_unit='${SAMPLING_UNIT}'"
  --config "dataset.config_overrides.max_num_frames_per_batch=${MAX_NUM_FRAMES_PER_BATCH}"
  --config "recipe.valid_splits='${VALID_SPLIT}'"
  --config "recipe.max_validation_sentences_per_lang_pair=${MAX_VALIDATION_SENTENCES_PER_LANG_PAIR}"
  --config "recipe.max_bleu_sentences_per_lang_pair=${MAX_BLEU_SENTENCES_PER_LANG_PAIR}"
  --config "regime.publish_metrics_every_n_steps=100"
  --config "regime.validate_every_n_steps=${VALIDATE_CHECKPOINT_EVERY_N_STEPS}"
  --config "regime.checkpoint_every_n_steps=${VALIDATE_CHECKPOINT_EVERY_N_STEPS}"
  --config "common.metric_recorders.wandb.project='${WANDB_PROJECT}'"
  --config "common.metric_recorders.wandb.run_name='${wandb_name}'"
  "${encoder_args[@]}"
)

echo "SimPOTrans multilingual SFT training"
echo "  sources:       $#"
echo "  manifest list: ${MANIFEST_LIST}"
echo "  train split:   ${TRAIN_SPLIT}"
echo "  valid split:   ${VALID_SPLIT}"
echo "  sampling:      ${SAMPLING_UNIT}, alpha=${SAMPLING_ALPHA}"
echo "  valid cap:     ${MAX_VALIDATION_SENTENCES_PER_LANG_PAIR}/lang-pair"
echo "  BLEU cap:      ${MAX_BLEU_SENTENCES_PER_LANG_PAIR}/lang-pair"
echo "  output dir:    ${OUTPUT_DIR}"
echo "  config file:   ${CONFIG_FILE}"
echo "  train entry:   ${TRAIN_ENTRY}"
echo "  val/ckpt int:  ${VALIDATE_CHECKPOINT_EVERY_N_STEPS}"
echo "  max frames:    ${MAX_NUM_FRAMES_PER_BATCH}"
if [[ -n "${SPEECH_ENCODER_BACKEND:-}" ]]; then
  echo "  encoder:       ${SPEECH_ENCODER_BACKEND}"
fi
if [[ -n "${FREEZE_ENCODER:-}" ]]; then
  echo "  freeze enc:    ${FREEZE_ENCODER}"
fi

if [[ -z "${devices}" || "${devices}" == "cpu" ]]; then
  echo "  device mode:   CPU"
  unset CUDA_VISIBLE_DEVICES
  .venv/bin/python "${TRAIN_ENTRY}" "${common_args[@]}"
else
  export CUDA_VISIBLE_DEVICES="${devices}"
  export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"

  device_array=("${(@s:,:)devices}")
  gpu_nums="${#device_array[@]}"

  echo "  device mode:   GPU"
  echo "  visible GPUs:  ${CUDA_VISIBLE_DEVICES}"
  echo "  processes:     ${gpu_nums}"

  torchrun \
    --nproc_per_node="${gpu_nums}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    "${TRAIN_ENTRY}" \
    "${common_args[@]}"
fi
