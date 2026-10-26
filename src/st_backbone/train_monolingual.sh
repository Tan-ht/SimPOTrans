#!/usr/bin/env zsh
emulate -L zsh
setopt err_exit nounset pipe_fail

usage() {
  cat <<'EOF'
Usage:
  uv run zsh train_sft.sh <src_lang> <tgt_lang> <dataset> <devices|cpu> <train_split> <valid_split> <wandb_name> [master_port]

Optional environment variables:
  VALIDATE_CHECKPOINT_EVERY_N_STEPS  Set both validation and checkpoint intervals. Default: 2000
  MAX_NUM_FRAMES_PER_BATCH           Override dataset max_num_frames_per_batch. Default: 400000
EOF
}

if [[ $# -lt 7 ]]; then
  usage
  exit 1
fi

src_lang="$1"
tgt_lang="$2"
dataset="$3"
devices="$4"
train_split="$5"
valid_split="$6"
wandb_name="$7"
if [[ $# -ge 8 ]]; then
  MASTER_PORT="$8"
fi

ROOT_PATH="${ROOT_PATH:-${0:A:h}}"
CONFIG_FILE="${CONFIG_FILE:-${ROOT_PATH}/src/st_backbone/config.yaml}"
TRAIN_ENTRY="${TRAIN_ENTRY:-${ROOT_PATH}/src/st_backbone/train.py}"
WANDB_PROJECT="${WANDB_PROJECT:-SimPOTrans}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
VALIDATE_CHECKPOINT_EVERY_N_STEPS="${VALIDATE_CHECKPOINT_EVERY_N_STEPS:-2000}"
MAX_NUM_FRAMES_PER_BATCH="${MAX_NUM_FRAMES_PER_BATCH:-600000}"

DATA_ROOT="${DATA_ROOT:-${ROOT_PATH}/data/${dataset}/${src_lang}-${tgt_lang}}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_PATH}/checkpoints/${dataset}/${wandb_name}}"

export TF_ENABLE_ONEDNN_OPTS="${TF_ENABLE_ONEDNN_OPTS:-0}"
export WANDB_MODE="${WANDB_MODE:-offline}"

common_args=(
  --config-file "${CONFIG_FILE}"
  "${OUTPUT_DIR}"
  --config "dataset.config_overrides.manifest_root='${DATA_ROOT}'"
  --config "dataset.config_overrides.audio_root='${DATA_ROOT}'"
  --config "dataset.config_overrides.splits='${train_split}'"
  --config "dataset.config_overrides.max_num_frames_per_batch=${MAX_NUM_FRAMES_PER_BATCH}"
  --config "recipe.valid_splits='${valid_split}'"
  --config "regime.publish_metrics_every_n_steps=100"
  --config "regime.validate_every_n_steps=${VALIDATE_CHECKPOINT_EVERY_N_STEPS}"
  --config "regime.checkpoint_every_n_steps=${VALIDATE_CHECKPOINT_EVERY_N_STEPS}"
  --config "model.config_overrides.src_lang='${src_lang}'"
  --config "model.config_overrides.tgt_lang='${tgt_lang}'"
  --config "common.metric_recorders.wandb.project='${WANDB_PROJECT}'"
  --config "common.metric_recorders.wandb.run_name='${wandb_name}'"
)

echo "SimPOTrans SFT training"
echo "  src/tgt:      ${src_lang}-${tgt_lang}"
echo "  dataset:      ${dataset}"
echo "  data root:    ${DATA_ROOT}"
echo "  train split:  ${train_split}"
echo "  valid split:  ${valid_split}"
echo "  output dir:   ${OUTPUT_DIR}"
echo "  config file:  ${CONFIG_FILE}"
echo "  train entry:  ${TRAIN_ENTRY}"
echo "  val/ckpt int: ${VALIDATE_CHECKPOINT_EVERY_N_STEPS}"
echo "  max frames:   ${MAX_NUM_FRAMES_PER_BATCH}"

if [[ -z "${devices}" || "${devices}" == "cpu" ]]; then
  echo "  device mode:  CPU"
  unset CUDA_VISIBLE_DEVICES
  .venv/bin/python "${TRAIN_ENTRY}" "${common_args[@]}"
else
  export CUDA_VISIBLE_DEVICES="${devices}"
  export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"

  device_array=("${(@s:,:)devices}")
  gpu_nums="${#device_array[@]}"

  echo "  device mode:  GPU"
  echo "  visible GPUs: ${CUDA_VISIBLE_DEVICES}"
  echo "  processes:    ${gpu_nums}"

  torchrun \
    --nproc_per_node="${gpu_nums}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    "${TRAIN_ENTRY}" \
    "${common_args[@]}"
fi
