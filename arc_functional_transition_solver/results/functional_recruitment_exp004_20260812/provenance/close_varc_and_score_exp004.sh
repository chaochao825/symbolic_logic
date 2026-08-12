#!/usr/bin/env bash

set -euo pipefail

EXP=/home/spco/sow_linear/codex_artifacts/hypothesis_population_20260812/exp004_prospective_v1
ELIG=/home/spco/sow_linear/codex_artifacts/hypothesis_population_20260812/exp003_prospective_v1/eligible_v1
PROJ=/home/spco/sow_linear/codex_worktrees/arc_failure_matrix_20260810/arc_functional_transition_solver
VARC=/home/wangmeiqi/codex_runs/arc_visual_provider_probe_20260807/external/VARC
VARC_PY=/home/wangmeiqi/codex_runs/arc_visual_provider_probe_20260807/.venv-varc/bin/python
ARC_TGI_PY=/home/spco/sow_linear/.venvs/afts_arc_tgi_20260810/bin/python
CHECKPOINT="${VARC}/saves/offline_train_ViT/checkpoint_best.pt"
ARC_TGI=/home/spco/sow_linear/codex_sources/arc_anchor_20260810/arc-tgi
ARC_TGI_COMMIT=a614132ff5b2cb3628063d541e7cbd74a2cd2edb
PARTITION="${PROJ}/results/arc_tgi_arcmini_cohort_v2_20260810/partition_manifest.json"
PARENT_PROTOCOL="${PROJ}/notes/research/provenance-relational-effect-v0.4-gate-20260812.md"
COHORT_ID=9706b3af57b19564e4579a20c29f7127271d253bab4e2e5c58657216681ffe15

test -s "${EXP}/visual_launch_commitment.json"
"${ARC_TGI_PY}" -c 'import shortuuid'
for shard_index in 0 1 2; do
  [[ "$(cat "${EXP}/varc_shard_${shard_index}_launcher_exit_code.txt")" == 0 ]]
  [[ "$(cat "${EXP}/varc_shard_${shard_index}_run_v1/exit_code.txt")" == 0 ]]
done

cd "${PROJ}"
for shard_index in 0 1 2; do
  for replay in a b; do
    PYTHONPATH=src "${VARC_PY}" scripts/afts_arc_varc_run_receipt.py \
      --blind-manifest "${EXP}/varc_shard_${shard_index}_blind_v1/manifest.json" \
      --run-dir "${EXP}/varc_shard_${shard_index}_run_v1" \
      --prediction-root "${EXP}/varc_shard_${shard_index}_predictions_v1" \
      --varc-root "${VARC}" \
      --checkpoint "${CHECKPOINT}" \
      --expected-source-commit bd478ecf362e6499a988b05f33223e5c5fc6a6be \
      --expected-checkpoint-sha256 c52d719fe9a41066aa8e96548ae6f83a0abf3568facd5103e176d7005e8421b3 \
      --output "${EXP}/varc_shard_${shard_index}_receipt_${replay}.json" \
      > "${EXP}/varc_shard_${shard_index}_receipt_${replay}.stdout.json"
  done
  cmp "${EXP}/varc_shard_${shard_index}_receipt_a.json" \
    "${EXP}/varc_shard_${shard_index}_receipt_b.json"
done

PYTHONPATH=src "${VARC_PY}" - "${EXP}" <<'PY'
import json
import shutil
import sys
from pathlib import Path

exp = Path(sys.argv[1])
output = exp / "varc_predictions_merged_v1"
if output.exists():
    raise FileExistsError(f"refusing to replace prediction root: {output}")
output.mkdir()
seen = set()
for shard_index in range(3):
    source_root = exp / f"varc_shard_{shard_index}_predictions_v1"
    for source in sorted(source_root.glob("*_predictions.json")):
        if source.name in seen:
            raise ValueError(f"prediction appears in multiple shards: {source.name}")
        seen.add(source.name)
        shutil.copy2(source, output / source.name)
manifest = json.loads(
    (exp / "varc_full_blind_v1" / "manifest.json").read_text(encoding="utf-8")
)
expected = {f"{record['task_id']}_predictions.json" for record in manifest["tasks"]}
if seen != expected:
    raise ValueError("merged prediction set differs from full blind manifest")
print(json.dumps({"prediction_file_count": len(seen)}, sort_keys=True))
PY

for replay in a b; do
  PYTHONPATH=src "${VARC_PY}" "${EXP}/merge_varc_shard_receipts.py" \
    --project-root "${PROJ}" \
    --full-blind-manifest "${EXP}/varc_full_blind_v1/manifest.json" \
    --full-compatibility "${EXP}/varc_full_compatibility_v1.json" \
    --prediction-root "${EXP}/varc_predictions_merged_v1" \
    --checkpoint "${CHECKPOINT}" \
    --orchestration-manifest "${EXP}/visual_launch_commitment.json" \
    --expected-source-commit bd478ecf362e6499a988b05f33223e5c5fc6a6be \
    --expected-checkpoint-sha256 c52d719fe9a41066aa8e96548ae6f83a0abf3568facd5103e176d7005e8421b3 \
    --shard-receipt "${EXP}/varc_shard_0_receipt_a.json" \
    --shard-receipt "${EXP}/varc_shard_1_receipt_a.json" \
    --shard-receipt "${EXP}/varc_shard_2_receipt_a.json" \
    --output "${EXP}/varc_merged_receipt_${replay}.json" \
    > "${EXP}/varc_merged_receipt_${replay}.stdout.json"
done
cmp "${EXP}/varc_merged_receipt_a.json" \
  "${EXP}/varc_merged_receipt_b.json"

for replay in a b; do
  PYTHONPATH=src "${VARC_PY}" scripts/afts_arc_varc_candidates.py freeze \
    --challenges "${ELIG}/eligible_challenges.json" \
    --blind-manifest "${EXP}/varc_full_blind_v1/manifest.json" \
    --prediction-root "${EXP}/varc_predictions_merged_v1" \
    --provider-contract "${EXP}/varc_provider_contract.json" \
    --output "${EXP}/varc_candidate_freeze_${replay}.json" \
    > "${EXP}/varc_candidate_freeze_${replay}.stdout.json"
done
cmp "${EXP}/varc_candidate_freeze_a.json" \
  "${EXP}/varc_candidate_freeze_b.json"

for replay in a b; do
  PYTHONPATH=src "${VARC_PY}" scripts/afts_arc_hypothesis_population.py freeze \
    --cohort-id "${COHORT_ID}" \
    --provider recursive recursive-test-time-learning "${EXP}/nvarc_candidate_freeze_a.json" \
    --provider visual visual-test-time-training "${EXP}/varc_candidate_freeze_a.json" \
    --receipt recursive "${EXP}/nvarc_run_receipt_a.json" \
    --receipt visual "${EXP}/varc_merged_receipt_a.json" \
    --output "${EXP}/population_${replay}.json" \
    > "${EXP}/population_${replay}.stdout.json"
done
cmp "${EXP}/population_a.json" "${EXP}/population_b.json"

for replay in a b; do
  PYTHONPATH=src "${ARC_TGI_PY}" scripts/afts_arc_tgi_indexed_reserve.py \
    open-population-oracle \
    --arc-tgi-root "${ARC_TGI}" \
    --arc-tgi-commit "${ARC_TGI_COMMIT}" \
    --partition-manifest "${PARTITION}" \
    --protocol "${PARENT_PROTOCOL}" \
    --output-root "${EXP}/oracle_open_${replay}" \
    --seal "${ELIG}/eligible_seal_manifest.json" \
    --anchor-freeze "${EXP}/nvarc_candidate_freeze_a.json" \
    --recruited-freeze "${EXP}/varc_candidate_freeze_a.json" \
    --population "${EXP}/population_a.json" \
    --recruitment-plan "${EXP}/recruitment_plan_a.json" \
    > "${EXP}/oracle_open_${replay}.stdout.json"
done
cmp "${EXP}/oracle_open_a/population_oracle_authorization.json" \
  "${EXP}/oracle_open_b/population_oracle_authorization.json"
cmp "${EXP}/oracle_open_a/reserve100_solutions.json" \
  "${EXP}/oracle_open_b/reserve100_solutions.json"
SOLUTIONS="${EXP}/oracle_open_a/reserve100_solutions.json"

for replay in a b; do
  PYTHONPATH=src "${VARC_PY}" scripts/afts_arc_nvarc_anchor_candidates.py score \
    --candidate-freeze "${EXP}/nvarc_candidate_freeze_a.json" \
    --solutions "${SOLUTIONS}" \
    --output "${EXP}/nvarc_score_${replay}.json" \
    > "${EXP}/nvarc_score_${replay}.stdout.json"
  PYTHONPATH=src "${VARC_PY}" scripts/afts_arc_varc_candidates.py score \
    --candidate-freeze "${EXP}/varc_candidate_freeze_a.json" \
    --solutions "${SOLUTIONS}" \
    --output "${EXP}/varc_score_${replay}.json" \
    > "${EXP}/varc_score_${replay}.stdout.json"
  PYTHONPATH=src "${VARC_PY}" scripts/afts_arc_hypothesis_population.py score \
    --population "${EXP}/population_a.json" \
    --solutions "${SOLUTIONS}" \
    --output "${EXP}/population_result_${replay}.json" \
    --summary-output "${EXP}/population_summary_${replay}.json" \
    > "${EXP}/population_score_${replay}.stdout.json"
done
cmp "${EXP}/nvarc_score_a.json" "${EXP}/nvarc_score_b.json"
cmp "${EXP}/varc_score_a.json" "${EXP}/varc_score_b.json"
cmp "${EXP}/population_result_a.json" "${EXP}/population_result_b.json"
cmp "${EXP}/population_summary_a.json" "${EXP}/population_summary_b.json"

PYTHONPATH=src "${VARC_PY}" - "${EXP}" <<'PY'
import json
import sys
from pathlib import Path

from afts_arc.experiment_safety import atomic_write_json, canonical_sha256, file_sha256

exp = Path(sys.argv[1])
summary_path = exp / "population_summary_a.json"
summary = json.loads(summary_path.read_text(encoding="utf-8"))
visual_exclusive = summary["metrics"]["exclusive_strict_task_coverage_by_provider"][
    "visual"
]
body = {
    "minimum_visual_exclusive_tasks": 3,
    "passed": visual_exclusive >= 3,
    "population_summary_id": summary["summary_id"],
    "population_summary_sha256": file_sha256(summary_path),
    "query_gold_read": True,
    "schema": "afts.exp004-complement-gate/v1",
    "task_count": 98,
    "visual_exclusive_tasks": visual_exclusive,
}
atomic_write_json(
    exp / "complement_gate.json",
    {"gate_id": canonical_sha256(body), **body},
)
print(json.dumps(body, sort_keys=True))
PY

complement_passed=$("${VARC_PY}" -c \
  'import json,sys; print(str(json.load(open(sys.argv[1], encoding="utf-8"))["passed"]).lower())' \
  "${EXP}/complement_gate.json")
if [[ "${complement_passed}" == true ]]; then
  for replay in a b; do
    PYTHONPATH=src "${VARC_PY}" scripts/afts_arc_functional_recruitment.py score \
      --plan "${EXP}/recruitment_plan_a.json" \
      --population-result "${EXP}/population_result_a.json" \
      --recruited-provider-receipt "${EXP}/varc_merged_receipt_a.json" \
      --output "${EXP}/recruitment_result_${replay}.json" \
      --summary-output "${EXP}/recruitment_summary_${replay}.json" \
      > "${EXP}/recruitment_score_${replay}.stdout.json"
    PYTHONPATH=src "${VARC_PY}" "${EXP}/score_equal_native_cost.py" \
      --project-root "${PROJ}" \
      --plan "${EXP}/recruitment_plan_a.json" \
      --population-result "${EXP}/population_result_a.json" \
      --recruited-provider-receipt "${EXP}/varc_merged_receipt_a.json" \
      --output "${EXP}/equal_native_cost_${replay}.json" \
      > "${EXP}/equal_native_cost_${replay}.stdout.json"
  done
  cmp "${EXP}/recruitment_result_a.json" \
    "${EXP}/recruitment_result_b.json"
  cmp "${EXP}/recruitment_summary_a.json" \
    "${EXP}/recruitment_summary_b.json"
  cmp "${EXP}/equal_native_cost_a.json" \
    "${EXP}/equal_native_cost_b.json"

  PYTHONPATH=src "${VARC_PY}" - "${EXP}" <<'PY'
import json
import sys
from pathlib import Path

from afts_arc.experiment_safety import atomic_write_json, canonical_sha256, file_sha256

exp = Path(sys.argv[1])
summary = json.loads((exp / "recruitment_summary_a.json").read_text(encoding="utf-8"))
primary = next(row for row in summary["budgets"] if row["activation_percentage"] == 30)
fraction = primary["observed_recruited_gpu_seconds_fraction"]
conditions = {
    "at_least_half_marginal_recall": (
        primary["target_recall"]["numerator"] * 2
        >= primary["target_recall"]["denominator"]
    ),
    "at_least_two_recoveries": primary["incremental_recoveries"] >= 2,
    "cost_at_most_35_percent": fraction["numerator"] * 100 <= fraction["denominator"] * 35,
    "strictly_exceeds_random_median_high": (
        primary["incremental_recoveries"]
        > primary["random_recovery_distribution"]["median_high"]
    ),
}
equal_cost = json.loads((exp / "equal_native_cost_a.json").read_text(encoding="utf-8"))
body = {
    "conditions": conditions,
    "equal_native_cost_comparison": equal_cost["comparison"],
    "passed": all(conditions.values()),
    "query_gold_read": True,
    "recruitment_summary_id": summary["summary_id"],
    "recruitment_summary_sha256": file_sha256(exp / "recruitment_summary_a.json"),
    "schema": "afts.exp004-recruitment-gate/v1",
}
atomic_write_json(
    exp / "recruitment_gate.json",
    {"gate_id": canonical_sha256(body), **body},
)
print(json.dumps(body, sort_keys=True))
PY
fi

find "${EXP}" -maxdepth 2 -type f \
  ! -path '*/trash/*' -print0 \
  | sort -z \
  | xargs -0 sha256sum > "${EXP}/final_artifacts.sha256"
