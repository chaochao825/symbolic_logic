#!/usr/bin/env bash

set -uo pipefail

if [[ $# -ne 2 ]]; then
  printf 'usage: %s SHARD_INDEX GPU_ID\n' "$0" >&2
  exit 2
fi

shard_index=$1
gpu_id=$2
if [[ ! "${shard_index}" =~ ^[0-2]$ ]] || [[ ! "${gpu_id}" =~ ^[0-9]+$ ]]; then
  printf 'invalid shard or GPU ID\n' >&2
  exit 2
fi

EXP=/home/spco/sow_linear/codex_artifacts/hypothesis_population_20260812/exp004_prospective_v1
PROJ=/home/spco/sow_linear/codex_worktrees/arc_failure_matrix_20260810/arc_functional_transition_solver
VARC=/home/wangmeiqi/codex_runs/arc_visual_provider_probe_20260807/external/VARC
VARC_PY=/home/wangmeiqi/codex_runs/arc_visual_provider_probe_20260807/.venv-varc/bin/python
CHECKPOINT="${VARC}/saves/offline_train_ViT/checkpoint_best.pt"
BLIND="${EXP}/varc_shard_${shard_index}_blind_v1"
blind_cohort_id=$("${VARC_PY}" -c \
  'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["blind_cohort_id"])' \
  "${BLIND}/manifest.json")

date --iso-8601=seconds > "${EXP}/varc_shard_${shard_index}_launcher_started_at.txt"
cd "${PROJ}"
status=0
env \
  AFTS_VARC_ROOT="${VARC}" \
  AFTS_VARC_PYTHON="${VARC_PY}" \
  AFTS_VARC_CHECKPOINT="${CHECKPOINT}" \
  AFTS_VARC_CHECKPOINT_SHA256=c52d719fe9a41066aa8e96548ae6f83a0abf3568facd5103e176d7005e8421b3 \
  AFTS_VARC_SOURCE_COMMIT=bd478ecf362e6499a988b05f33223e5c5fc6a6be \
  AFTS_VARC_BLIND_DATA_ROOT="${BLIND}" \
  AFTS_VARC_BLIND_COHORT_ID="${blind_cohort_id}" \
  AFTS_VARC_DATA_ROOT_NAME="AFTS_EXP004_SHARD${shard_index}_BLIND_V1" \
  AFTS_VARC_SAVE_NAME="afts_exp004_shard${shard_index}_v1" \
  AFTS_VARC_OUTPUT_DIR="${EXP}/varc_shard_${shard_index}_predictions_v1" \
  AFTS_VARC_RUN_DIR="${EXP}/varc_shard_${shard_index}_run_v1" \
  AFTS_VARC_GPU_IDS="${gpu_id}" \
  AFTS_VARC_TASK_RUNNER="${PROJ}/scripts/run_afts_arc_varc_task.sh" \
  AFTS_VARC_COMPAT_BUILDER="${PROJ}/scripts/afts_arc_varc_diagnostic_compat.py" \
  bash scripts/launch_afts_arc_varc_cohort.sh \
  || status=$?

printf '%s\n' "${status}" \
  > "${EXP}/varc_shard_${shard_index}_launcher_exit_code.txt"
date --iso-8601=seconds \
  > "${EXP}/varc_shard_${shard_index}_launcher_finished_at.txt"
exit "${status}"
