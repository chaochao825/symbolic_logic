#!/usr/bin/env bash

set -uo pipefail

EXP=/home/spco/sow_linear/codex_artifacts/hypothesis_population_20260812/exp004_prospective_v1
PROJ=/home/spco/sow_linear/codex_worktrees/arc_failure_matrix_20260810/arc_functional_transition_solver
ELIG=/home/spco/sow_linear/codex_artifacts/hypothesis_population_20260812/exp003_prospective_v1/eligible_v1
PY=/home/spco/sow_linear/.venvs/nvarc_trm_20260810/bin/python

date --iso-8601=seconds > "${EXP}/nvarc_launcher_started_at.txt"
printf '%s  %s\n' \
  60682d2ca38496987ddd85033ce9400cfb9a7838995310c1676a6e1156a6d864 \
  "${ELIG}/eligible_challenges.json" \
  | sha256sum --check --strict
"${PY}" - "${EXP}/dataset_boundary_audit_a.json" <<'PY'
import json
import pathlib
import sys

audit = json.loads(pathlib.Path(sys.argv[1]).read_text())
if audit["status"] != "clean":
    raise SystemExit("NVARC dataset boundary audit is not clean")
if audit["audit_id"] != "e7e345dc8490628bcd5271bf8610403de7207d5716c06d9e2ac9c02cd7062a68":
    raise SystemExit("NVARC dataset boundary audit identity drifted")
PY

cd "${PROJ}"
status=0
env \
  AFTS_NVARC_TRM_ROOT=/home/spco/sow_linear/codex_sources/arc_anchor_20260810/nvarc/TRM \
  AFTS_NVARC_PYTHON="${PY}" \
  AFTS_NVARC_DATA="${EXP}/nvarc_eval_aug128" \
  AFTS_NVARC_CHECKPOINT=/home/spco/sow_linear/codex_artifacts/nvarc_trm_anchor_20260810/checkpoint/step_220708 \
  AFTS_NVARC_OUTPUT="${EXP}/nvarc_run_v1" \
  AFTS_NVARC_TRITON_CACHE="${EXP}/nvarc_triton_cache_v1" \
  AFTS_NVARC_INDUCTOR_CACHE="${EXP}/nvarc_inductor_cache_v1" \
  AFTS_CUDA_VISIBLE_DEVICES=0 \
  bash scripts/run_afts_arc_nvarc_trm_anchor.sh \
  || status=$?

printf '%s\n' "${status}" > "${EXP}/nvarc_launcher_exit_code.txt"
date --iso-8601=seconds > "${EXP}/nvarc_launcher_finished_at.txt"
exit "${status}"
