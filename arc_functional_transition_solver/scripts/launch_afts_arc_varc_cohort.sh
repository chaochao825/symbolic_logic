#!/usr/bin/env bash

set -euo pipefail

: "${AFTS_VARC_ROOT:?Set AFTS_VARC_ROOT.}"
: "${AFTS_VARC_PYTHON:?Set AFTS_VARC_PYTHON.}"
: "${AFTS_VARC_CHECKPOINT:?Set AFTS_VARC_CHECKPOINT.}"
: "${AFTS_VARC_CHECKPOINT_SHA256:?Set AFTS_VARC_CHECKPOINT_SHA256.}"
: "${AFTS_VARC_SOURCE_COMMIT:?Set AFTS_VARC_SOURCE_COMMIT.}"
: "${AFTS_VARC_BLIND_DATA_ROOT:?Set AFTS_VARC_BLIND_DATA_ROOT.}"
: "${AFTS_VARC_BLIND_COHORT_ID:?Set AFTS_VARC_BLIND_COHORT_ID.}"
: "${AFTS_VARC_DATA_ROOT_NAME:?Set AFTS_VARC_DATA_ROOT_NAME.}"
: "${AFTS_VARC_SAVE_NAME:?Set AFTS_VARC_SAVE_NAME.}"
: "${AFTS_VARC_OUTPUT_DIR:?Set AFTS_VARC_OUTPUT_DIR to a new directory.}"
: "${AFTS_VARC_RUN_DIR:?Set AFTS_VARC_RUN_DIR to a new directory.}"
: "${AFTS_VARC_GPU_IDS:?Set AFTS_VARC_GPU_IDS to space-separated GPU IDs.}"
: "${AFTS_VARC_TASK_RUNNER:?Set AFTS_VARC_TASK_RUNNER.}"
: "${AFTS_VARC_COMPAT_BUILDER:?Set AFTS_VARC_COMPAT_BUILDER.}"

if [[ -e "${AFTS_VARC_OUTPUT_DIR}" || -e "${AFTS_VARC_RUN_DIR}" ]]; then
  printf 'Refusing to replace an existing VARC output or run directory\n' >&2
  exit 2
fi

for required_path in \
  "${AFTS_VARC_ROOT}/test_time_train_ARC.py" \
  "${AFTS_VARC_PYTHON}" \
  "${AFTS_VARC_CHECKPOINT}" \
  "${AFTS_VARC_TASK_RUNNER}" \
  "${AFTS_VARC_COMPAT_BUILDER}"; do
  if [[ ! -e "${required_path}" ]]; then
    printf 'Required path does not exist: %s\n' "${required_path}" >&2
    exit 2
  fi
done

actual_checkpoint_sha256=$(sha256sum "${AFTS_VARC_CHECKPOINT}" | awk '{print $1}')
if [[ "${actual_checkpoint_sha256}" != "${AFTS_VARC_CHECKPOINT_SHA256}" ]]; then
  printf 'VARC checkpoint SHA-256 mismatch\n' >&2
  exit 3
fi
actual_source_commit=$(git -C "${AFTS_VARC_ROOT}" rev-parse HEAD)
if [[ "${actual_source_commit}" != "${AFTS_VARC_SOURCE_COMMIT}" ]]; then
  printf 'VARC source commit mismatch\n' >&2
  exit 3
fi

evaluation_dir="${AFTS_VARC_BLIND_DATA_ROOT}/data/evaluation"
augmentation_dir="${AFTS_VARC_BLIND_DATA_ROOT}/data/eval_color_permute_ttt_9"
if [[ ! -d "${evaluation_dir}" || ! -d "${augmentation_dir}" ]]; then
  printf 'VARC blind data is missing evaluation or augmentation directories\n' >&2
  exit 2
fi
actual_blind_cohort_id=$(
  "${AFTS_VARC_PYTHON}" -c \
    'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["blind_cohort_id"])' \
    "${AFTS_VARC_BLIND_DATA_ROOT}/manifest.json"
)
if [[ "${actual_blind_cohort_id}" != "${AFTS_VARC_BLIND_COHORT_ID}" ]]; then
  printf 'VARC blind cohort ID mismatch\n' >&2
  exit 3
fi

mapfile -t task_ids < <(
  find "${evaluation_dir}" -maxdepth 1 -type f -name '*.json' -printf '%f\n' \
    | sed 's/\.json$//' \
    | sort
)
if (( ${#task_ids[@]} == 0 )); then
  printf 'VARC cohort contains no tasks\n' >&2
  exit 2
fi
read -r -a gpu_ids <<< "${AFTS_VARC_GPU_IDS}"
if (( ${#gpu_ids[@]} == 0 )); then
  printf 'No GPU IDs were provided\n' >&2
  exit 2
fi
if (( ${#gpu_ids[@]} != 1 )); then
  printf 'VARC diagnostic compatibility requires exactly one serialized GPU\n' >&2
  exit 2
fi
declare -A seen_gpu_ids=()
for gpu_id in "${gpu_ids[@]}"; do
  if [[ ! "${gpu_id}" =~ ^[0-9]+$ ]]; then
    printf 'Invalid GPU ID: %s\n' "${gpu_id}" >&2
    exit 2
  fi
  if [[ -n "${seen_gpu_ids[${gpu_id}]+present}" ]]; then
    printf 'Duplicate GPU ID: %s\n' "${gpu_id}" >&2
    exit 2
  fi
  seen_gpu_ids[${gpu_id}]=1
done

data_link="${AFTS_VARC_ROOT}/raw_data/${AFTS_VARC_DATA_ROOT_NAME}"
output_link="${AFTS_VARC_ROOT}/outputs/${AFTS_VARC_SAVE_NAME}_attempt_0"
if [[ -e "${data_link}" || -L "${data_link}" || -e "${output_link}" || -L "${output_link}" ]]; then
  printf 'Refusing to replace an existing VARC data or output link\n' >&2
  exit 2
fi

check_gpu_idle() {
  local gpu_id=$1
  local memory_used utilization
  IFS=, read -r memory_used utilization < <(
    nvidia-smi --id="${gpu_id}" \
      --query-gpu=memory.used,utilization.gpu \
      --format=csv,noheader,nounits
  )
  memory_used=${memory_used// /}
  utilization=${utilization// /}
  if (( memory_used > 1000 || utilization > 10 )); then
    printf 'GPU %s is not idle: memory_used=%s MiB utilization=%s%%\n' \
      "${gpu_id}" "${memory_used}" "${utilization}" >&2
    return 1
  fi
}

for gpu_id in "${gpu_ids[@]}"; do
  check_gpu_idle "${gpu_id}"
done

mkdir -p \
  "${AFTS_VARC_ROOT}/raw_data" \
  "${AFTS_VARC_ROOT}/outputs" \
  "${AFTS_VARC_OUTPUT_DIR}" \
  "${AFTS_VARC_RUN_DIR}"
"${AFTS_VARC_PYTHON}" "${AFTS_VARC_COMPAT_BUILDER}" build \
  --blind-data-root "${AFTS_VARC_BLIND_DATA_ROOT}" \
  --runtime-data-root "${AFTS_VARC_RUN_DIR}/runtime_data" \
  --output "${AFTS_VARC_RUN_DIR}/task_compatibility.json"
ln -s "${AFTS_VARC_RUN_DIR}/runtime_data" "${data_link}"
ln -s "${AFTS_VARC_OUTPUT_DIR}" "${output_link}"
printf '%s\n' "${task_ids[@]}" > "${AFTS_VARC_RUN_DIR}/task_ids.txt"
printf '%s\n' "${gpu_ids[@]}" > "${AFTS_VARC_RUN_DIR}/gpu_ids.txt"
date --iso-8601=seconds > "${AFTS_VARC_RUN_DIR}/started_at.txt"
nvidia-smi -q > "${AFTS_VARC_RUN_DIR}/nvidia_smi.before.log"
"${AFTS_VARC_PYTHON}" --version > "${AFTS_VARC_RUN_DIR}/python_version.txt" 2>&1
"${AFTS_VARC_PYTHON}" -m pip freeze > "${AFTS_VARC_RUN_DIR}/pip_freeze.txt"
sha256sum \
  "${AFTS_VARC_BLIND_DATA_ROOT}/manifest.json" \
  "${AFTS_VARC_CHECKPOINT}" \
  "${AFTS_VARC_COMPAT_BUILDER}" \
  "${AFTS_VARC_TASK_RUNNER}" \
  "$0" > "${AFTS_VARC_RUN_DIR}/source_hashes.sha256"
git -C "${AFTS_VARC_ROOT}" rev-parse HEAD > "${AFTS_VARC_RUN_DIR}/source_commit.txt"
git -C "${AFTS_VARC_ROOT}" status --short > "${AFTS_VARC_RUN_DIR}/source_status.txt"
git -C "${AFTS_VARC_ROOT}" diff --binary --no-ext-diff \
  > "${AFTS_VARC_RUN_DIR}/source_tracked.diff"
find "${AFTS_VARC_BLIND_DATA_ROOT}" -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum > "${AFTS_VARC_RUN_DIR}/blind_data.sha256"

run_worker() {
  local worker_index=$1
  local gpu_id=${gpu_ids[worker_index]}
  local lock="/tmp/afts-varc-gpu${gpu_id}.lock"
  local worker_dir="${AFTS_VARC_RUN_DIR}/gpu${gpu_id}"
  mkdir -p "${worker_dir}" "${AFTS_VARC_RUN_DIR}/tasks"
  exec 9>"${lock}"
  if ! flock -n 9; then
    printf 'GPU lock is already held: %s\n' "${lock}" >&2
    return 3
  fi
  check_gpu_idle "${gpu_id}"
  export CUDA_VISIBLE_DEVICES="${gpu_id}"
  export PYTHONHASHSEED=0
  export PYTHONDONTWRITEBYTECODE=1
  export TORCHINDUCTOR_CACHE_DIR="${worker_dir}/torchinductor_cache"
  local index task_id task_dir start_epoch end_epoch status
  for ((index = worker_index; index < ${#task_ids[@]}; index += ${#gpu_ids[@]})); do
    task_id=${task_ids[index]}
    task_dir="${AFTS_VARC_RUN_DIR}/tasks/${task_id}"
    mkdir -p "${task_dir}"
    start_epoch=$(date +%s)
    "${AFTS_VARC_PYTHON}" "${AFTS_VARC_COMPAT_BUILDER}" prepare-task \
      --runtime-data-root "${AFTS_VARC_RUN_DIR}/runtime_data" \
      --task-id "${task_id}" \
      --archive-dir "${task_dir}/diagnostic_alias_history" \
      --output "${task_dir}/diagnostic_alias.json"
    set +e
    timeout --signal=TERM --kill-after=60s 2400s \
      "${AFTS_VARC_TASK_RUNNER}" "${task_id}" \
      > "${task_dir}/run.log" 2>&1
    status=$?
    set -e
    end_epoch=$(date +%s)
    printf '{"elapsed_seconds":%s,"end_epoch":%s,"exit_code":%s,"gpu_id":%s,"start_epoch":%s,"task_id":"%s"}\n' \
      "$((end_epoch - start_epoch))" "${end_epoch}" "${status}" "${gpu_id}" \
      "${start_epoch}" "${task_id}" > "${task_dir}/status.json"
    if (( status != 0 )); then
      return "${status}"
    fi
  done
  printf '{"complete":true,"gpu_id":%s}\n' "${gpu_id}" \
    > "${worker_dir}/complete.json"
}

pids=()
for worker_index in "${!gpu_ids[@]}"; do
  run_worker "${worker_index}" \
    > "${AFTS_VARC_RUN_DIR}/worker_${worker_index}.stdout.log" \
    2> "${AFTS_VARC_RUN_DIR}/worker_${worker_index}.stderr.log" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
printf '%s\n' "${status}" > "${AFTS_VARC_RUN_DIR}/exit_code.txt"
date --iso-8601=seconds > "${AFTS_VARC_RUN_DIR}/finished_at.txt"
nvidia-smi -q > "${AFTS_VARC_RUN_DIR}/nvidia_smi.after.log"
exit "${status}"
