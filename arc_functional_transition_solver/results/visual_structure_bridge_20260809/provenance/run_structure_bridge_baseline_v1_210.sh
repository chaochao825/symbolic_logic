#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/wangmeiqi/codex_runs/arc_visual_posterior_bridge_20260809
SOLVER=/home/wangmeiqi/codex_runs/arc_visual_provider_probe_20260807/solver_clean_8c855bf/arc_functional_transition_solver
PYTHON=/home/wangmeiqi/cjlprojects/conda_envs/ditpa/bin/python
DATASET="$ROOT/cohort_v1/gold_data"
OUTPUT_DIR="$ROOT/baseline_structure_v1"
RUN_DIR="$ROOT/runs/portfolio_baseline_structure_v1"
TASK_IDS=bae5c565,9b30e358,8dab14c2,af726779,87ab05b8,9841fdad,78e78cff,14b8e18c,6bcdb01e,57edb29d,e39e9282,b5bb5719

if [[ -e "$OUTPUT_DIR" || -e "$RUN_DIR" ]]; then
  printf 'baseline artifact directory already exists\n' >&2
  exit 2
fi
if [[ ! -e "$ROOT/gold_release_start_v1.json" ]]; then
  printf 'gold-release start receipt is missing\n' >&2
  exit 3
fi
mkdir -p "$RUN_DIR"
cd "$SOLVER"
if [[ -n "$(git status --short)" ]]; then
  printf 'clean baseline worktree is dirty\n' >&2
  exit 3
fi
if [[ "$(git rev-parse HEAD)" != 8c855bf0c2a37c135c9ac0f68fbe2e3fd677bc7c ]]; then
  printf 'baseline source commit mismatch\n' >&2
  exit 3
fi

start_epoch=$(date +%s)
set +e
PYTHONPATH=src /usr/bin/time -v "$PYTHON" \
  scripts/afts_arc_online_matched_budget.py \
  "$DATASET" "$OUTPUT_DIR" \
  --split training \
  --limit 12 \
  --task-ids "$TASK_IDS" >"$RUN_DIR/run.log" 2>&1
status=$?
set -e
end_epoch=$(date +%s)
printf '{"exit_code":%s,"start_epoch":%s,"end_epoch":%s,"elapsed_seconds":%s,"source_commit":"%s"}\n' \
  "$status" "$start_epoch" "$end_epoch" "$((end_epoch - start_epoch))" \
  "$(git rev-parse HEAD)" >"$RUN_DIR/status.json"
exit "$status"
