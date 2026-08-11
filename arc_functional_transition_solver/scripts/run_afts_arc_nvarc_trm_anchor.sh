#!/usr/bin/env bash

set -euo pipefail

: "${AFTS_NVARC_TRM_ROOT:?Set AFTS_NVARC_TRM_ROOT to the pinned NVARC TRM source directory.}"
: "${AFTS_NVARC_PYTHON:?Set AFTS_NVARC_PYTHON to the frozen Python interpreter.}"
: "${AFTS_NVARC_DATA:?Set AFTS_NVARC_DATA to the compiled evaluation dataset.}"
: "${AFTS_NVARC_CHECKPOINT:?Set AFTS_NVARC_CHECKPOINT to step_220708.}"
: "${AFTS_NVARC_OUTPUT:?Set AFTS_NVARC_OUTPUT to a new immutable output directory.}"
: "${AFTS_NVARC_TRITON_CACHE:?Set AFTS_NVARC_TRITON_CACHE to an isolated cache directory.}"
: "${AFTS_NVARC_INDUCTOR_CACHE:?Set AFTS_NVARC_INDUCTOR_CACHE to an isolated cache directory.}"
: "${AFTS_CUDA_VISIBLE_DEVICES:?Set AFTS_CUDA_VISIBLE_DEVICES explicitly.}"

if [[ -e "${AFTS_NVARC_OUTPUT}" ]]; then
  printf 'Refusing to overwrite existing output: %s\n' "${AFTS_NVARC_OUTPUT}" >&2
  exit 2
fi

for required_path in \
  "${AFTS_NVARC_TRM_ROOT}/eval-arc-k-10.py" \
  "${AFTS_NVARC_PYTHON}" \
  "${AFTS_NVARC_DATA}" \
  "${AFTS_NVARC_CHECKPOINT}"; do
  if [[ ! -e "${required_path}" ]]; then
    printf 'Required path does not exist: %s\n' "${required_path}" >&2
    exit 2
  fi
done

mkdir -p \
  "${AFTS_NVARC_OUTPUT}" \
  "${AFTS_NVARC_TRITON_CACHE}" \
  "${AFTS_NVARC_INDUCTOR_CACHE}"

date --iso-8601=seconds > "${AFTS_NVARC_OUTPUT}/started_at.txt"
printf '%s\n' "${AFTS_CUDA_VISIBLE_DEVICES}" > "${AFTS_NVARC_OUTPUT}/cuda_visible_devices.txt"
uname -a > "${AFTS_NVARC_OUTPUT}/uname.txt"
nvidia-smi -q > "${AFTS_NVARC_OUTPUT}/nvidia_smi.before.log"
"${AFTS_NVARC_PYTHON}" --version > "${AFTS_NVARC_OUTPUT}/python_version.txt" 2>&1
"${AFTS_NVARC_PYTHON}" -m pip freeze > "${AFTS_NVARC_OUTPUT}/pip_freeze.txt"
git -C "${AFTS_NVARC_TRM_ROOT}" rev-parse HEAD > "${AFTS_NVARC_OUTPUT}/source_commit.txt"
git -C "${AFTS_NVARC_TRM_ROOT}" status --short > "${AFTS_NVARC_OUTPUT}/source_status.txt"
sha256sum "${AFTS_NVARC_CHECKPOINT}" > "${AFTS_NVARC_OUTPUT}/checkpoint.sha256"
find "${AFTS_NVARC_DATA}" -maxdepth 1 -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum > "${AFTS_NVARC_OUTPUT}/dataset.sha256"

cd "${AFTS_NVARC_TRM_ROOT}"
set +e
CUDA_VISIBLE_DEVICES="${AFTS_CUDA_VISIBLE_DEVICES}" \
PYTHONHASHSEED=0 \
PYTHONDONTWRITEBYTECODE=1 \
TRITON_CACHE_DIR="${AFTS_NVARC_TRITON_CACHE}" \
TORCHINDUCTOR_CACHE_DIR="${AFTS_NVARC_INDUCTOR_CACHE}" \
"${AFTS_NVARC_PYTHON}" -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=1 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=localhost:0 \
  eval-arc-k-10.py \
  arch=trm \
  "data_paths=[${AFTS_NVARC_DATA}]" \
  arch.L_layers=2 \
  arch.H_cycles=4 \
  arch.L_cycles=4 \
  arch.halt_max_steps=10 \
  "+load_checkpoint=${AFTS_NVARC_CHECKPOINT}" \
  eval_interval=2000 \
  epochs=2000 \
  global_batch_size=128 \
  ema=True \
  "+checkpoint_path=${AFTS_NVARC_OUTPUT}/checkpoint" \
  lr_warmup_steps=200 \
  lr=0.0001 \
  > "${AFTS_NVARC_OUTPUT}/run.stdout.log" \
  2> "${AFTS_NVARC_OUTPUT}/run.stderr.log"
status=$?
set -e

printf '%s\n' "${status}" > "${AFTS_NVARC_OUTPUT}/exit_code.txt"
date --iso-8601=seconds > "${AFTS_NVARC_OUTPUT}/finished_at.txt"
nvidia-smi -q > "${AFTS_NVARC_OUTPUT}/nvidia_smi.after.log"

exit "${status}"
