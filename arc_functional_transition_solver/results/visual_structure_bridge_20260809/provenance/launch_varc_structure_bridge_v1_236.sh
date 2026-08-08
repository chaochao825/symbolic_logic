#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/wangmeiqi/codex_runs/arc_visual_posterior_bridge_20260809
VARC=/home/wangmeiqi/codex_runs/arc_visual_provider_probe_20260807/external/VARC
RUN_DIR="$ROOT/runs/varc_structure_bridge_v1"
OUTPUT_DIR="$VARC/outputs/query_blind_structure_v1_attempt_0"
TASK_RUNNER="$ROOT/run_varc_structure_bridge_task_v1_236.sh"
SESSION_PREFIX=arc_varc_struct_v1_g
GPU_IDS=(0 1)
TASKS=(
  bae5c565 9b30e358 8dab14c2 af726779
  87ab05b8 9841fdad 78e78cff 14b8e18c
  6bcdb01e 57edb29d e39e9282 b5bb5719
)

check_gpu_idle() {
  local gpu_id=$1
  local memory_used utilization
  IFS=, read -r memory_used utilization < <(
    nvidia-smi --id="$gpu_id" \
      --query-gpu=memory.used,utilization.gpu \
      --format=csv,noheader,nounits
  )
  memory_used=${memory_used// /}
  utilization=${utilization// /}
  if (( memory_used > 1000 || utilization > 10 )); then
    printf 'GPU %s is not idle: memory_used=%s MiB utilization=%s%%\n' \
      "$gpu_id" "$memory_used" "$utilization" >&2
    return 1
  fi
}

run_worker() {
  local worker_index=$1
  local gpu_id=${GPU_IDS[$worker_index]}
  local lock="/tmp/codex-arc-varc-gpu${gpu_id}.lock"
  local worker_dir="$RUN_DIR/gpu${gpu_id}"
  mkdir -p "$worker_dir" "$RUN_DIR/tasks" "$RUN_DIR/cache/gpu${gpu_id}"
  exec 9>"$lock"
  if ! flock -n 9; then
    printf 'GPU lock is already held: %s\n' "$lock" >&2
    exit 3
  fi
  check_gpu_idle "$gpu_id"
  nvidia-smi --id="$gpu_id" \
    --query-gpu=index,name,memory.total,memory.used,utilization.gpu,pstate \
    --format=csv,noheader,nounits >"$worker_dir/gpu_prelaunch.csv"

  export CUDA_VISIBLE_DEVICES=$gpu_id
  export TORCHINDUCTOR_CACHE_DIR="$RUN_DIR/cache/gpu${gpu_id}"
  local index task_id task_dir start_epoch end_epoch status
  for ((index = worker_index; index < ${#TASKS[@]}; index += ${#GPU_IDS[@]})); do
    task_id=${TASKS[$index]}
    task_dir="$RUN_DIR/tasks/$task_id"
    mkdir -p "$task_dir"
    if [[ -e "$OUTPUT_DIR/${task_id}_predictions.json" ]]; then
      printf 'prediction artifact already exists: %s\n' \
        "$OUTPUT_DIR/${task_id}_predictions.json" >&2
      exit 2
    fi
    start_epoch=$(date +%s)
    set +e
    timeout --signal=TERM --kill-after=60s 2400s \
      "$TASK_RUNNER" "$task_id" >"$task_dir/run.log" 2>&1
    status=$?
    set -e
    end_epoch=$(date +%s)
    printf '{"task_id":"%s","gpu_id":%s,"exit_code":%s,"start_epoch":%s,"end_epoch":%s,"elapsed_seconds":%s}\n' \
      "$task_id" "$gpu_id" "$status" "$start_epoch" "$end_epoch" \
      "$((end_epoch - start_epoch))" >"$task_dir/status.json"
    if (( status != 0 )); then
      exit "$status"
    fi
  done
  printf '{"gpu_id":%s,"complete":true}\n' "$gpu_id" \
    >"$worker_dir/complete.json"
}

if [[ "${1:-}" == "--worker" ]]; then
  if [[ $# -ne 2 || ! "$2" =~ ^[0-1]$ ]]; then
    printf 'worker requires an index in [0, 1]\n' >&2
    exit 2
  fi
  run_worker "$2"
  exit 0
fi

if [[ -e "$RUN_DIR" || -e "$OUTPUT_DIR" ]]; then
  printf 'confirmation artifact directory already exists\n' >&2
  exit 2
fi
for gpu_id in "${GPU_IDS[@]}"; do
  session="${SESSION_PREFIX}${gpu_id}"
  if tmux has-session -t "$session" 2>/dev/null; then
    printf 'tmux session already exists: %s\n' "$session" >&2
    exit 2
  fi
  check_gpu_idle "$gpu_id"
done

mkdir -p "$RUN_DIR"
printf '%s\n' "${TASKS[@]}" >"$RUN_DIR/task_ids.txt"
nvidia-smi \
  --query-gpu=index,name,memory.total,memory.used,utilization.gpu,pstate \
  --format=csv,noheader,nounits >"$RUN_DIR/gpu_launch.csv"
sha256sum \
  "$ROOT/cohort_v1/manifest.json" \
  "$ROOT/incoming/structure_confirm_v1_blind.tar.gz" \
  "$VARC/saves/offline_train_ViT/checkpoint_best.pt" \
  "$TASK_RUNNER" \
  "$0" >"$RUN_DIR/source_hashes.sha256"

for worker_index in 0 1; do
  gpu_id=${GPU_IDS[$worker_index]}
  session="${SESSION_PREFIX}${gpu_id}"
  tmux new-session -d -s "$session" "$0 --worker $worker_index"
  printf '%s\n' "$session"
done
