#!/usr/bin/env bash

set -euo pipefail

EXP=/home/spco/sow_linear/codex_artifacts/hypothesis_population_20260812/exp004_prospective_v1
ELIG=/home/spco/sow_linear/codex_artifacts/hypothesis_population_20260812/exp003_prospective_v1/eligible_v1
PROJ=/home/spco/sow_linear/codex_worktrees/arc_failure_matrix_20260810/arc_functional_transition_solver
VARC=/home/wangmeiqi/codex_runs/arc_visual_provider_probe_20260807/external/VARC
VARC_PY=/home/wangmeiqi/codex_runs/arc_visual_provider_probe_20260807/.venv-varc/bin/python
CHECKPOINT="${VARC}/saves/offline_train_ViT/checkpoint_best.pt"
SOURCE_COMMIT=bd478ecf362e6499a988b05f33223e5c5fc6a6be
CHECKPOINT_SHA=c52d719fe9a41066aa8e96548ae6f83a0abf3568facd5103e176d7005e8421b3
COHORT_ID=9706b3af57b19564e4579a20c29f7127271d253bab4e2e5c58657216681ffe15
ASSIGNMENT="${EXP}/varc_shard_assignment_v1/assignment.json"
FULL_BLIND="${EXP}/varc_full_blind_v1"

test -s "${EXP}/pre_visual_commitment.json"
test -s "${EXP}/recruitment_plan_a.json"
test ! -e "${FULL_BLIND}"
test ! -e "${EXP}/varc_predictions_merged_v1"
[[ "$(git -C "${VARC}" rev-parse HEAD)" == "${SOURCE_COMMIT}" ]]
[[ "$(sha256sum "${CHECKPOINT}" | awk '{print $1}')" == "${CHECKPOINT_SHA}" ]]
git -C "${VARC}" diff --binary --no-ext-diff \
  > "${EXP}/varc_source_tracked.diff"
[[ "$(sha256sum "${EXP}/varc_source_tracked.diff" | awk '{print $1}')" == \
  e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 ]]

cd "${PROJ}"
PYTHONPATH=src "${VARC_PY}" scripts/afts_arc_varc_query_blind_cohort.py \
  --challenges "${ELIG}/eligible_challenges.json" \
  --source-cohort-id "${COHORT_ID}" \
  --output-dir "${FULL_BLIND}" \
  > "${EXP}/varc_full_blind.stdout.json"

cd "${VARC}"
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 "${VARC_PY}" - "${FULL_BLIND}" \
  > "${EXP}/varc_full_augmentation.stdout.log" \
  2> "${EXP}/varc_full_augmentation.stderr.log" <<'PY'
import sys

from utils.data_augmentation import augment_raw_data_split_per_task

paths = augment_raw_data_split_per_task(
    dataset_root=sys.argv[1],
    split="evaluation",
    output_subdir="eval_color_permute_ttt_9",
    num_permuate=9,
    only_basic=True,
)
print(f"saved_path_count={len(paths)}")
PY

cd "${PROJ}"
for shard_index in 0 1 2; do
  shard_challenges="${EXP}/varc_shard_assignment_v1/shard_${shard_index}_challenges.json"
  shard_blind="${EXP}/varc_shard_${shard_index}_blind_v1"
  PYTHONPATH=src "${VARC_PY}" scripts/afts_arc_varc_query_blind_cohort.py \
    --challenges "${shard_challenges}" \
    --source-cohort-id "${COHORT_ID}" \
    --output-dir "${shard_blind}" \
    > "${EXP}/varc_shard_${shard_index}_blind.stdout.json"
  mkdir -p "${shard_blind}/data/eval_color_permute_ttt_9"
  while IFS= read -r task_id; do
    cp -a -- \
      "${FULL_BLIND}/data/eval_color_permute_ttt_9/${task_id}" \
      "${shard_blind}/data/eval_color_permute_ttt_9/${task_id}"
  done < <(
    find "${shard_blind}/data/evaluation" -maxdepth 1 -type f -name '*.json' \
      -printf '%f\n' | sed 's/\.json$//' | sort
  )
done

for replay in a b; do
  PYTHONPATH=src "${VARC_PY}" "${EXP}/audit_varc_shard_data.py" \
    --project-root "${PROJ}" \
    --full-blind-root "${FULL_BLIND}" \
    --assignment "${ASSIGNMENT}" \
    --shard-root-prefix "${EXP}/varc_shard_" \
    --output "${EXP}/varc_shard_data_audit_${replay}.json" \
    > "${EXP}/varc_shard_data_audit_${replay}.stdout.json"
done
cmp "${EXP}/varc_shard_data_audit_a.json" \
  "${EXP}/varc_shard_data_audit_b.json"

PYTHONPATH=src "${VARC_PY}" scripts/afts_arc_varc_diagnostic_compat.py build \
  --blind-data-root "${FULL_BLIND}" \
  --runtime-data-root "${EXP}/varc_full_compat_runtime_v1" \
  --output "${EXP}/varc_full_compatibility_v1.json" \
  > "${EXP}/varc_full_compatibility_v1.stdout.json"

PYTHONPATH=src "${VARC_PY}" - "${EXP}" "${VARC}" "${CHECKPOINT}" <<'PY'
import json
import sys
from pathlib import Path

from afts_arc.experiment_safety import atomic_write_json, canonical_sha256, file_sha256

exp = Path(sys.argv[1])
varc = Path(sys.argv[2])
checkpoint = Path(sys.argv[3])
full_manifest_path = exp / "varc_full_blind_v1" / "manifest.json"
full_manifest = json.loads(full_manifest_path.read_text(encoding="utf-8"))
audit_path = exp / "varc_shard_data_audit_a.json"
audit = json.loads(audit_path.read_text(encoding="utf-8"))
body = {
    "audit_script_sha256": file_sha256(exp / "audit_varc_shard_data.py"),
    "augmentation_source_sha256": file_sha256(varc / "utils" / "data_augmentation.py"),
    "checkpoint_sha256": file_sha256(checkpoint),
    "full_blind_cohort_id": full_manifest["blind_cohort_id"],
    "full_blind_manifest_sha256": file_sha256(full_manifest_path),
    "full_compatibility_sha256": file_sha256(
        exp / "varc_full_compatibility_v1.json"
    ),
    "pre_visual_commitment_sha256": file_sha256(
        exp / "pre_visual_commitment.json"
    ),
    "preparation_script_sha256": file_sha256(
        exp / "prepare_varc_data_exp004.sh"
    ),
    "query_gold_read": False,
    "schema": "afts.exp004-varc-data-preparation/v1",
    "shard_data_audit_id": audit["audit_id"],
    "shard_data_audit_sha256": file_sha256(audit_path),
    "source_commit": "bd478ecf362e6499a988b05f33223e5c5fc6a6be",
    "source_tracked_diff_sha256": file_sha256(
        exp / "varc_source_tracked.diff"
    ),
    "task_count": 98,
}
atomic_write_json(
    exp / "varc_data_preparation_manifest.json",
    {"preparation_id": canonical_sha256(body), **body},
)
print(
    json.dumps(
        {
            "full_blind_cohort_id": body["full_blind_cohort_id"],
            "preparation_id": canonical_sha256(body),
            "task_count": body["task_count"],
        },
        sort_keys=True,
    )
)
PY

sha256sum \
  "${FULL_BLIND}/manifest.json" \
  "${EXP}/varc_shard_data_audit_a.json" \
  "${EXP}/varc_full_compatibility_v1.json" \
  "${EXP}/varc_data_preparation_manifest.json" \
  > "${EXP}/varc_prelaunch_data.sha256"
