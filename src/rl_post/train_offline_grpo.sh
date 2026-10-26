#!/usr/bin/env zsh
emulate -L zsh
setopt err_exit nounset pipe_fail

usage() {
  cat <<'USAGE'
Usage:
  uv run zsh src/rl_post/train_offline_grpo.sh <rl_device|cpu> <xcomet_device|none> <wandb_name> <train_split> <valid_split> <checkpoint> <source>...

Source format:
  <dataset>:<src_lang>-<tgt_lang>[:manifest_root[:audio_root[:train_split[:valid_split]]]]

Example:
  uv run zsh src/rl_post/train_offline_grpo.sh 0 1 st_multi_offline_grpo train_st_asr_filtered dev_acl6060 \
    checkpoints/multilingual/st_multi/checkpoints/step_40000 \
    covost:en-zh \
    must-c:en-fr \
    must-c:en-nl

Optional environment variables:
  CONFIG_FILE                         Default: src/rl_post/offline_config.yaml
  ST_CONFIG_FILE                      Default: src/st_backbone/config.yaml
  DATA_ROOT                           Default: data
  OUTPUT_DIR                          Default: checkpoints/multilingual/<wandb_name>
  MANIFEST_LIST                       Default: <OUTPUT_DIR>/multilingual_sources.tsv
  SAMPLING_ALPHA                      Multilingual sampling alpha. Default: 0.5
  SAMPLING_UNIT                       lang_pair or source. Default: lang_pair
  MAX_NUM_FRAMES_PER_BATCH            Dynamic rollout source-frame budget. Default: 400000
  ROLLOUT_BATCH_SIZE                  Max utterances per rollout step. Default: 12
  GROUP_SIZE                          Candidates per utterance. Default: 12
  MAX_STEPS                           Total optimizer update steps. Default: 4000
  UPDATE_EPOCHS                       Default: 10
  LOGPROB_MICRO_BATCH_SIZE            Default: 4
  USE_REFERENCE_MODEL                 Load reference model for KL. Default: false
  USE_KL                              Apply KL penalty. Default: same as USE_REFERENCE_MODEL
  KL_COEF                             Default: 0.02, ignored when USE_KL=false or USE_REFERENCE_MODEL=false
  LR                                  Default: 2.0e-6
  VALIDATE_EVERY_N_STEPS              Default: 200
  VALIDATION_NUM_SAMPLES              Default: 1000
  MAX_NEW_TOKENS                      Default: 128
  RESUME_MODE                         allow, never, or require. Default: allow
  RESUME_FROM                         Optional checkpoint dir or training_state.pt path. Default: auto
  SPEECH_ENCODER_BACKEND              Optional: whisper or whisper_rt. Default: ST config value
  WHISPER_RT_MODEL                    Optional WhisperRT model size/checkpoint. Default: large-v2
  WHISPER_RT_CHUNK_SIZE_MS            Optional WhisperRT chunk size in milliseconds. Default: 300
  WHISPER_RT_MULTILINGUAL             Optional WhisperRT multilingual flag. Default: true
  FREEZE_ENCODER                      Ignored in RL; speech encoder is always frozen
  WANDB_PROJECT                       Default: SimPOTrans
  WANDB_MODE                          Default: offline
USAGE
}

if [[ $# -lt 7 ]]; then
  usage
  exit 1
fi

rl_device="$1"
xcomet_device="$2"
wandb_name="$3"
TRAIN_SPLIT="$4"
VALID_SPLIT="$5"
checkpoint="$6"
shift 6

ROOT_PATH="${ROOT_PATH:-${0:A:h}/../..}"
CONFIG_FILE="${CONFIG_FILE:-${ROOT_PATH}/src/rl_post/offline_config.yaml}"
TRAIN_ENTRY="${TRAIN_ENTRY:-${ROOT_PATH}/src/rl_post/train.py}"
ST_CONFIG_FILE="${ST_CONFIG_FILE:-${ROOT_PATH}/src/st_backbone/config.yaml}"
WANDB_PROJECT="${WANDB_PROJECT:-SimPOTrans}"
SAMPLING_ALPHA="${SAMPLING_ALPHA:-0.5}"
SAMPLING_UNIT="${SAMPLING_UNIT:-lang_pair}"
MAX_NUM_FRAMES_PER_BATCH="${MAX_NUM_FRAMES_PER_BATCH:-400000}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-12}"
GROUP_SIZE="${GROUP_SIZE:-12}"
MAX_STEPS="${MAX_STEPS:-4000}"
UPDATE_EPOCHS="${UPDATE_EPOCHS:-10}"
LOGPROB_MICRO_BATCH_SIZE="${LOGPROB_MICRO_BATCH_SIZE:-4}"
USE_REFERENCE_MODEL="${USE_REFERENCE_MODEL:-false}"
USE_KL="${USE_KL:-${USE_REFERENCE_MODEL}}"
if [[ "${USE_REFERENCE_MODEL}" != "true" ]]; then
  USE_KL="false"
fi
KL_COEF="${KL_COEF:-0.02}"
EFFECTIVE_KL_COEF="${KL_COEF}"
if [[ "${USE_KL}" != "true" ]]; then
  EFFECTIVE_KL_COEF="0.0"
fi
LR="${LR:-2.0e-6}"
VALIDATE_EVERY_N_STEPS="${VALIDATE_EVERY_N_STEPS:-200}"
VALIDATION_NUM_SAMPLES="${VALIDATION_NUM_SAMPLES:-1000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
RESUME_MODE="${RESUME_MODE:-allow}"
RESUME_FROM="${RESUME_FROM:-}"

DATA_ROOT="${DATA_ROOT:-${ROOT_PATH}/data}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_PATH}/checkpoints/multilingual/${wandb_name}}"
MANIFEST_LIST="${MANIFEST_LIST:-${OUTPUT_DIR}/multilingual_sources.tsv}"
COMET_PYTHON="${COMET_PYTHON:-${ROOT_PATH}/.comet-venv/bin/python}"
COMET_REQUEST_SIZE="${COMET_REQUEST_SIZE:-128}"
WANDB_ENABLED="${WANDB_ENABLED:-true}"
export WANDB_MODE="${WANDB_MODE:-offline}"

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
    manifest_root="${parts[3]:-${DATA_ROOT}/${dataset}/${lang_pair}}"
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

export PYTHONPATH="${ROOT_PATH}/src:${PYTHONPATH:-}"
export UV_CACHE_DIR="${ROOT_PATH}/.uv-cache"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/simpotrans-matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

device_value="cuda"
if [[ -z "${rl_device}" || "${rl_device}" == "cpu" ]]; then
  unset CUDA_VISIBLE_DEVICES
  device_value="cpu"
else
  export CUDA_VISIBLE_DEVICES="${rl_device}"
fi

xcomet_visible_devices="null"
if [[ -n "${xcomet_device}" && "${xcomet_device}" != "none" && "${xcomet_device}" != "null" ]]; then
  xcomet_visible_devices="'${xcomet_device}'"
fi

FREEZE_ENCODER=true

encoder_set_args=()
if [[ -n "${SPEECH_ENCODER_BACKEND:-}" ]]; then
  encoder_set_args+=(--set "model.config_overrides.speech_encoder_backend='${SPEECH_ENCODER_BACKEND}'")
fi
if [[ -n "${WHISPER_RT_MODEL:-}" ]]; then
  encoder_set_args+=(--set "model.config_overrides.whisper_rt_model='${WHISPER_RT_MODEL}'")
fi
if [[ -n "${WHISPER_RT_CHUNK_SIZE_MS:-}" ]]; then
  encoder_set_args+=(--set "model.config_overrides.whisper_rt_chunk_size_ms=${WHISPER_RT_CHUNK_SIZE_MS}")
fi
if [[ -n "${WHISPER_RT_MULTILINGUAL:-}" ]]; then
  encoder_set_args+=(--set "model.config_overrides.whisper_rt_multilingual=${WHISPER_RT_MULTILINGUAL}")
fi
if [[ -n "${FREEZE_ENCODER:-}" ]]; then
  encoder_set_args+=(--set "model.config_overrides.freeze_encoder=${FREEZE_ENCODER}")
fi

set_args=(
  --set "model.config_path='${ST_CONFIG_FILE}'"
  --set "model.device='${device_value}'"
  --set "model.checkpoint='${checkpoint}'"
  --set "dataset.manifest_list='${MANIFEST_LIST}'"
  --set "dataset.splits='${TRAIN_SPLIT}'"
  --set "dataset.default_train_split='train_st'"
  --set "dataset.default_valid_split='dev_st'"
  --set "dataset.sampling_alpha=${SAMPLING_ALPHA}"
  --set "dataset.sampling_unit='${SAMPLING_UNIT}'"
  --set "rollout.mode='offline'"
  --set "rollout.batch_size=${ROLLOUT_BATCH_SIZE}"
  --set "rollout.max_source_frames_per_batch=${MAX_NUM_FRAMES_PER_BATCH}"
  --set "rollout.group_size=${GROUP_SIZE}"
  --set "rollout.max_steps=${MAX_STEPS}"
  --set "rollout.max_new_tokens=${MAX_NEW_TOKENS}"
  --set "reward.scorer='xcomet'"
  --set "reward.comet_python='${COMET_PYTHON}'"
  --set "reward.comet_visible_devices=${xcomet_visible_devices}"
  --set "reward.comet_request_size=${COMET_REQUEST_SIZE}"
  --set "grpo.update_epochs=${UPDATE_EPOCHS}"
  --set "grpo.logprob_micro_batch_size=${LOGPROB_MICRO_BATCH_SIZE}"
  --set "grpo.kl_coef=${EFFECTIVE_KL_COEF}"
  --set "grpo.reference_model=${USE_REFERENCE_MODEL}"
  --set "optimizer.lr=${LR}"
  --set "validation.splits='${VALID_SPLIT}'"
  --set "validation.every_steps=${VALIDATE_EVERY_N_STEPS}"
  --set "validation.num_samples=${VALIDATION_NUM_SAMPLES}"
  --set "validation.max_new_tokens=${MAX_NEW_TOKENS}"
  --set "wandb.enabled=${WANDB_ENABLED}"
  --set "wandb.project='${WANDB_PROJECT}'"
  --set "wandb.run_name='${wandb_name}'"
  --set "output.dir='${OUTPUT_DIR}'"
  --set "output.resume_mode='${RESUME_MODE}'"
  "${encoder_set_args[@]}"
)

if [[ -n "${RESUME_FROM}" ]]; then
  set_args+=(--set "output.resume_from='${RESUME_FROM}'")
fi

echo "SimPOTrans multilingual Offline GRPO post-training"
echo "  sources:       $#"
echo "  manifest list: ${MANIFEST_LIST}"
echo "  checkpoint:    ${checkpoint}"
echo "  train split:   ${TRAIN_SPLIT}"
echo "  valid split:   ${VALID_SPLIT}"
echo "  sampling:      ${SAMPLING_UNIT}, alpha=${SAMPLING_ALPHA}"
echo "  output dir:    ${OUTPUT_DIR}"
echo "  resume mode:   ${RESUME_MODE}"
if [[ -n "${RESUME_FROM}" ]]; then
  echo "  resume from:   ${RESUME_FROM}"
fi
echo "  config file:   ${CONFIG_FILE}"
echo "  train entry:   ${TRAIN_ENTRY}"
echo "  rl device:     ${rl_device}"
echo "  xcomet gpu:    ${xcomet_device}"
echo "  xcomet request:${COMET_REQUEST_SIZE}"
echo "  max frames:    ${MAX_NUM_FRAMES_PER_BATCH}"
if [[ -n "${SPEECH_ENCODER_BACKEND:-}" ]]; then
  echo "  encoder:       ${SPEECH_ENCODER_BACKEND}"
fi
if [[ -n "${FREEZE_ENCODER:-}" ]]; then
  echo "  freeze enc:    ${FREEZE_ENCODER}"
fi
echo "  max utt/step:  ${ROLLOUT_BATCH_SIZE}"
echo "  group size:    ${GROUP_SIZE}"
echo "  update ep:     ${UPDATE_EPOCHS}"
echo "  logprob mb:    ${LOGPROB_MICRO_BATCH_SIZE}"
echo "  max tokens:    ${MAX_NEW_TOKENS}"
echo "  reference:     ${USE_REFERENCE_MODEL}"
echo "  use KL:        ${USE_KL}"
echo "  KL coef:       ${EFFECTIVE_KL_COEF}"

${PYTHON:-python3} "${TRAIN_ENTRY}" --config "${CONFIG_FILE}" --mode offline "${set_args[@]}"
