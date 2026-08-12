#!/usr/bin/env bash

set -euo pipefail

EXP=/home/spco/sow_linear/codex_artifacts/hypothesis_population_20260812/exp004_prospective_v1
RUN="${EXP}/nvarc_run_v1"
DATA="${EXP}/nvarc_eval_aug128"
ELIG=/home/spco/sow_linear/codex_artifacts/hypothesis_population_20260812/exp003_prospective_v1/eligible_v1
PROJ=/home/spco/sow_linear/codex_worktrees/arc_failure_matrix_20260810/arc_functional_transition_solver
NVARC=/home/spco/sow_linear/codex_sources/arc_anchor_20260810/nvarc/TRM
MODEL=/home/spco/sow_linear/codex_sources/arc_anchor_20260810/nvarc/external/TinyRecursiveModels
PY=/home/spco/sow_linear/.venvs/nvarc_trm_20260810/bin/python
CHECKPOINT=/home/spco/sow_linear/codex_artifacts/nvarc_trm_anchor_20260810/checkpoint/step_220708
COHORT_ID=9706b3af57b19564e4579a20c29f7127271d253bab4e2e5c58657216681ffe15
SOURCE_COMMIT=846d0198efa752534594e321fc3289fc0a06c657
MODEL_COMMIT=e7b68717f0a6c4cbb4ce6fbef787b14f42083bd9
RUNNER="${PROJ}/scripts/run_afts_arc_nvarc_trm_anchor.sh"

[[ "$(cat "${EXP}/nvarc_launcher_exit_code.txt")" == 0 ]]
[[ "$(cat "${RUN}/exit_code.txt")" == 0 ]]
[[ "$(cat "${RUN}/source_commit.txt")" == "${SOURCE_COMMIT}" ]]
[[ "$(git -C "${MODEL}" rev-parse HEAD)" == "${MODEL_COMMIT}" ]]
cmp "${EXP}/dataset_boundary_audit_a.json" "${EXP}/dataset_boundary_audit_b.json"
test ! -e "${EXP}/varc_predictions_merged_v1"

git -C "${NVARC}" diff --binary --no-ext-diff > "${RUN}/source_tracked.diff"
git -C "${MODEL}" rev-parse HEAD > "${RUN}/model_source_commit.txt"
git -C "${MODEL}" status --short > "${RUN}/model_source_status.txt"
git -C "${MODEL}" diff --binary --no-ext-diff > "${RUN}/model_source_tracked.diff"
find "${DATA}" -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum > "${RUN}/dataset_recursive.sha256"
sha256sum \
  "${RUNNER}" \
  "${NVARC}/eval-arc-k-10.py" \
  "${MODEL}/evaluators/arc.py" \
  "${MODEL}/models/losses.py" \
  "${MODEL}/models/recursive_reasoning/trm.py" \
  > "${RUN}/executed_source.sha256"

cd "${PROJ}"
for replay in a b; do
  PYTHONPATH=src "${PY}" scripts/afts_arc_nvarc_run_receipt.py \
    --run-dir "${RUN}" \
    --data-dir "${DATA}" \
    --challenges "${ELIG}/eligible_challenges.json" \
    --cohort-manifest "${ELIG}/eligible_seal_manifest.json" \
    --cohort-id "${COHORT_ID}" \
    --initial-checkpoint "${CHECKPOINT}" \
    --runner "${RUNNER}" \
    --schedule-audit "${EXP}/dataset_boundary_audit_a.json" \
    --expected-source-commit "${SOURCE_COMMIT}" \
    --expected-model-source-commit "${MODEL_COMMIT}" \
    --output "${EXP}/nvarc_run_receipt_${replay}.json" \
    > "${EXP}/nvarc_run_receipt_${replay}.stdout.json"
done
cmp "${EXP}/nvarc_run_receipt_a.json" "${EXP}/nvarc_run_receipt_b.json"

receipt_id=$("${PY}" -c \
  'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["receipt_id"])' \
  "${EXP}/nvarc_run_receipt_a.json")
submission="${RUN}/checkpoint/evaluator_ARC_step_5608/submission.json"
for replay in a b; do
  PYTHONPATH=src "${PY}" scripts/afts_arc_nvarc_anchor_candidates.py freeze \
    --challenges "${ELIG}/eligible_challenges.json" \
    --submission "${submission}" \
    --cohort-id "${COHORT_ID}" \
    --anchor-run-id "${receipt_id}" \
    --record-invalid-attempts \
    --output "${EXP}/nvarc_candidate_freeze_${replay}.json" \
    > "${EXP}/nvarc_candidate_freeze_${replay}.stdout.json"
done
cmp "${EXP}/nvarc_candidate_freeze_a.json" \
  "${EXP}/nvarc_candidate_freeze_b.json"

PYTHONPATH=src "${PY}" - \
  "${ELIG}/eligible_seal_manifest.json" \
  "${EXP}/task_universe.json" \
  "${EXP}/varc_provider_contract.json" <<'PY'
import json
import sys
from pathlib import Path

from afts_arc.experiment_safety import atomic_write_json, canonical_sha256
from afts_arc.varc_run import VARC_TTT_CONFIG

seal_path = Path(sys.argv[1])
task_universe_path = Path(sys.argv[2])
provider_contract_path = Path(sys.argv[3])
seal = json.loads(seal_path.read_text(encoding="utf-8"))
task_ids = sorted(row["task_id"] for row in seal["tasks"])
if len(task_ids) != 98 or len(set(task_ids)) != 98:
    raise ValueError("eligible seal does not contain 98 unique tasks")
atomic_write_json(task_universe_path, task_ids)

contract_body = {
    "candidate_contract": {
        "invalid_candidate_policy": "reject",
        "invalid_candidate_rule": (
            "reject non-list, empty, non-rectangular, >30x30, non-integer, "
            "or color-outside-[0,9] samples; never repair or coerce"
        ),
        "query_requires_at_least_one_valid_candidate": True,
        "ranking": "frequency, then first-emission position, then canonical grid JSON",
    },
    "cohort": {
        "cohort_id": "9706b3af57b19564e4579a20c29f7127271d253bab4e2e5c58657216681ffe15",
        "seal_id": "d9c3c71ca6eb10289e3a64eeff3fedbec3107c6c8f947160ff50afc2edd9fe60",
        "task_count": 98,
    },
    "controller_frozen": True,
    "execution": {
        "augmentation": "full-cohort sorted index, then byte-identical task partition",
        "mode": "three isolated serialized single-GPU shards",
        "shard_assignment_id": "7c1cdb12b02dc2323fc9d2ecf6f880cdfac1272a8cf852bcc128bbdd102877b2",
    },
    "name": "VARC ViT static query-blind provider EXP-004",
    "provider_version": "afts-varc-vit-ttt/v1",
    "query_blind_protocol": {
        "candidate_logits_consume": ["attention_mask", "inputs", "task_ids"],
        "provider_visible_test_output": "exact copy of corresponding test input",
        "query_gold_read": False,
        "training_examples": "demonstrations only",
    },
    "schema": "afts.varc-provider-preexecution-contract/v1",
    "scientific_scope": "solver-track static candidate distribution",
    "source": {
        "checkpoint_sha256": "c52d719fe9a41066aa8e96548ae6f83a0abf3568facd5103e176d7005e8421b3",
        "commit": "bd478ecf362e6499a988b05f33223e5c5fc6a6be",
        "repository": "https://github.com/lillian039/VARC",
    },
    "test_time_configuration": VARC_TTT_CONFIG,
}
atomic_write_json(
    provider_contract_path,
    {"contract_id": canonical_sha256(contract_body), **contract_body},
)
PY

for replay in a b; do
  PYTHONPATH=src "${PY}" scripts/afts_arc_functional_recruitment.py freeze \
    --anchor-freeze "${EXP}/nvarc_candidate_freeze_a.json" \
    --anchor-receipt "${EXP}/nvarc_run_receipt_a.json" \
    --protocol "${PROJ}/.research-control/experiments/protocols/EXP-004.md" \
    --task-universe "${EXP}/task_universe.json" \
    --cohort-id "${COHORT_ID}" \
    --anchor-provider-name recursive \
    --recruited-provider-name visual \
    --budget-percent 10 \
    --budget-percent 20 \
    --budget-percent 30 \
    --budget-percent 50 \
    --random-seed-count 256 \
    --output "${EXP}/recruitment_plan_${replay}.json" \
    > "${EXP}/recruitment_plan_${replay}.stdout.json"
done
cmp "${EXP}/recruitment_plan_a.json" "${EXP}/recruitment_plan_b.json"

PYTHONPATH=src "${PY}" - \
  "${EXP}" "${PROJ}" "${ELIG}" <<'PY'
import json
import sys
from pathlib import Path

from afts_arc.experiment_safety import atomic_write_json, canonical_sha256, file_sha256

exp = Path(sys.argv[1])
project = Path(sys.argv[2])
eligible = Path(sys.argv[3])
receipt = json.loads((exp / "nvarc_run_receipt_a.json").read_text(encoding="utf-8"))
freeze = json.loads((exp / "nvarc_candidate_freeze_a.json").read_text(encoding="utf-8"))
plan = json.loads((exp / "recruitment_plan_a.json").read_text(encoding="utf-8"))
provider = json.loads((exp / "varc_provider_contract.json").read_text(encoding="utf-8"))
assignment = json.loads(
    (exp / "varc_shard_assignment_v1" / "assignment.json").read_text(encoding="utf-8")
)
body = {
    "anchor": {
        "candidate_freeze_id": freeze["freeze_id"],
        "candidate_freeze_sha256": file_sha256(exp / "nvarc_candidate_freeze_a.json"),
        "run_receipt_id": receipt["receipt_id"],
        "run_receipt_sha256": file_sha256(exp / "nvarc_run_receipt_a.json"),
    },
    "cohort": {
        "challenge_sha256": file_sha256(eligible / "eligible_challenges.json"),
        "cohort_id": "9706b3af57b19564e4579a20c29f7127271d253bab4e2e5c58657216681ffe15",
        "seal_id": "d9c3c71ca6eb10289e3a64eeff3fedbec3107c6c8f947160ff50afc2edd9fe60",
        "seal_sha256": file_sha256(eligible / "eligible_seal_manifest.json"),
        "task_count": 98,
    },
    "implementation": {
        "base_commit": "c48371547daf36183e4b52b0a90906007b9e8a70",
        "merge_script_sha256": file_sha256(exp / "merge_varc_shard_receipts.py"),
        "prepare_shards_script_sha256": file_sha256(exp / "prepare_varc_shards.py"),
        "protocol_commit": "216800f9c2e9d00e36e5ee8b458e269e58b2138a",
        "protocol_sha256": file_sha256(
            project / ".research-control" / "experiments" / "protocols" / "EXP-004.md"
        ),
    },
    "query_gold_opened": False,
    "recruitment": {
        "budget_task_counts": plan["budget_task_counts"],
        "plan_id": plan["plan_id"],
        "plan_sha256": file_sha256(exp / "recruitment_plan_a.json"),
        "policy_name": plan["policy_name"],
        "random_order_count": len(plan["random_priority_orders"]),
        "recruited_provider_candidates_read": plan[
            "recruited_provider_candidates_read"
        ],
    },
    "schema": "afts.exp004-pre-visual-commitment/v1",
    "visual": {
        "assignment_id": assignment["assignment_id"],
        "assignment_sha256": file_sha256(
            exp / "varc_shard_assignment_v1" / "assignment.json"
        ),
        "provider_contract_id": provider["contract_id"],
        "provider_contract_sha256": file_sha256(exp / "varc_provider_contract.json"),
        "provider_outcome_available": False,
    },
}
atomic_write_json(
    exp / "pre_visual_commitment.json",
    {"commitment_id": canonical_sha256(body), **body},
)
print(
    json.dumps(
        {
            "anchor_freeze_id": freeze["freeze_id"],
            "anchor_receipt_id": receipt["receipt_id"],
            "budget_task_counts": plan["budget_task_counts"],
            "commitment_id": canonical_sha256(body),
            "plan_id": plan["plan_id"],
        },
        sort_keys=True,
    )
)
PY

sha256sum \
  "${EXP}/dataset_boundary_audit_a.json" \
  "${EXP}/nvarc_run_receipt_a.json" \
  "${EXP}/nvarc_candidate_freeze_a.json" \
  "${EXP}/task_universe.json" \
  "${EXP}/recruitment_plan_a.json" \
  "${EXP}/varc_provider_contract.json" \
  "${EXP}/pre_visual_commitment.json" \
  > "${EXP}/pre_visual_artifacts.sha256"
