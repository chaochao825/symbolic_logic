"""Produce a descriptive, post-outcome failure analysis without tuning."""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path


ROOT = Path("/home/wangmeiqi/codex_runs/arc_visual_posterior_bridge_20260809")
REPOSITORY = Path(
    "/home/wangmeiqi/codex_runs/symbolic_logic_arc_online_control_20260724/repo"
    "/arc_functional_transition_solver"
)
sys.path.insert(0, str(REPOSITORY / "src"))

from afts_arc.experiment_safety import atomic_write_json, canonical_sha256  # noqa: E402
from afts_arc.visual_provider_gate import (  # noqa: E402
    _load_provider_predictions,
    _validate_grid,
    _validate_provider_predictions,
    rank_predictions,
)
from afts_arc.visual_structure_bridge import (  # noqa: E402
    _rank_with_structure,
    stable_demo_contract,
)


OUTPUT_PATH = ROOT / "structure_bridge_forensics_v1.json"


def _load(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def main() -> None:
    if OUTPUT_PATH.exists():
        raise FileExistsError(f"forensics artifact already exists: {OUTPUT_PATH}")
    manifest = _load(ROOT / "cohort_v1" / "manifest.json")
    candidates = _load(ROOT / "pre_gold" / "structure_candidates.json")
    visual = _load(ROOT / "visual_score_structure_v1" / "summary.json")
    structure = _load(ROOT / "structure_score_v1.json")
    if structure["decision"]["classification"] != "null":
        raise ValueError("forensics protocol is bound to the frozen null result")
    task_ids = tuple(record["task_id"] for record in manifest["tasks"])
    raw, _ = _load_provider_predictions(
        task_ids, (ROOT / "from236_structure_v1" / "raw_predictions",)
    )
    combined, _ = _validate_provider_predictions(
        raw, invalid_candidate_policy="reject"
    )
    candidate_by_task = {
        record["task_id"]: record for record in candidates["tasks"]
    }
    visual_by_task = {record["task_id"]: record for record in visual["tasks"]}
    structure_by_task = {
        record["task_id"]: record for record in structure["tasks"]
    }

    actions: Counter[str] = Counter()
    tasks = []
    for task_id in task_ids:
        blind = _load(
            ROOT
            / "cohort_v1"
            / "blind_data"
            / "ARC-AGI-2"
            / "data"
            / "evaluation"
            / f"{task_id}.json"
        )
        gold = _load(
            ROOT / "cohort_v1" / "gold_data" / "training" / f"{task_id}.json"
        )
        source = _validate_grid(blind["test"][0]["input"])
        target = _validate_grid(gold["test"][0]["output"])
        contract = stable_demo_contract(blind["train"])
        samples = combined[task_id][0]
        frequency = rank_predictions(samples)
        structural = _rank_with_structure(
            source=source,
            samples=samples,
            contract=contract,
        )
        structural_grids = tuple(record[0] for record in structural)
        match_fractions = {
            key: sum(record[4][key] == value for record in structural)
            / len(structural)
            for key, value in contract.items()
        }
        query_candidate = candidate_by_task[task_id]["queries"][0]
        certificate = query_candidate["certificate"]
        action = "none" if certificate is None else certificate["recommended_action"]
        actions[action] += 1
        selected_second = _validate_grid(query_candidate["hybrid_top_2"][1])
        frequency_second = _validate_grid(query_candidate["frequency_top_2"][1])
        visual_query = visual_by_task[task_id]["queries"][0]
        visual_flags = visual_by_task[task_id]["visual_provider"]
        tasks.append(
            {
                "task_id": task_id,
                "visual_raw_oracle_covered": visual_flags["raw"],
                "frequency_pass_at_1": visual_flags["pass_at_1"],
                "frequency_pass_at_2": visual_flags["pass_at_2"],
                "hybrid_pass_at_2": structure_by_task[task_id][
                    "hybrid_pass_at_2"
                ],
                "gold_frequency_rank": (
                    frequency.index(target) + 1 if target in frequency else None
                ),
                "gold_structural_rank": (
                    structural_grids.index(target) + 1
                    if target in structural_grids
                    else None
                ),
                "unique_candidate_count": len(frequency),
                "stable_contract_field_count": len(contract),
                "candidate_discriminative_field_count": sum(
                    0.0 < fraction < 1.0
                    for fraction in match_fractions.values()
                ),
                "certificate_action": action,
                "contract_shuffle_action_changed": query_candidate[
                    "contract_shuffle"
                ]["action_changed"],
                "hybrid_second_changed": selected_second != frequency_second,
                "selected_second_frequency_rank": frequency.index(selected_second) + 1,
                "selected_second_structural_rank": (
                    structural_grids.index(selected_second) + 1
                ),
                "best_same_shape_mismatch_count": visual_query[
                    "best_same_shape_mismatch_count"
                ],
                "best_same_shape_mismatch_rate": visual_query[
                    "best_same_shape_mismatch_rate"
                ],
            }
        )

    discriminative_counts = [
        record["candidate_discriminative_field_count"] for record in tasks
    ]
    content = {
        "schema": "afts.visual-structure-bridge-forensics/v1",
        "analysis_scope": (
            "post-outcome descriptive diagnosis only; no ranking, feature, or "
            "threshold change is authorized"
        ),
        "frozen_result_id": structure["result_id"],
        "frozen_classification": structure["decision"]["classification"],
        "metrics": {
            "task_count": len(tasks),
            "raw_coverage_gap_count": sum(
                not record["visual_raw_oracle_covered"] for record in tasks
            ),
            "frequency_selection_gap_count": sum(
                record["visual_raw_oracle_covered"]
                and not record["frequency_pass_at_2"]
                for record in tasks
            ),
            "hybrid_second_changed_count": sum(
                record["hybrid_second_changed"] for record in tasks
            ),
            "hybrid_unique_recovery_count": structure["metrics"][
                "hybrid_unique_task_recovery"
            ],
            "certificate_action_counts": dict(actions),
            "median_candidate_discriminative_field_count": statistics.median(
                discriminative_counts
            ),
            "zero_discriminative_contract_count": sum(
                count == 0 for count in discriminative_counts
            ),
        },
        "failure_sets": {
            "raw_coverage_gap": [
                record["task_id"]
                for record in tasks
                if not record["visual_raw_oracle_covered"]
            ],
            "frequency_selection_gap": [
                record["task_id"]
                for record in tasks
                if record["visual_raw_oracle_covered"]
                and not record["frequency_pass_at_2"]
            ],
            "natural_same_shape_near_miss": [
                record["task_id"]
                for record in tasks
                if not record["visual_raw_oracle_covered"]
                and record["best_same_shape_mismatch_count"] is not None
                and record["best_same_shape_mismatch_count"] <= 9
            ],
        },
        "tasks": tasks,
    }
    artifact = {"forensics_id": canonical_sha256(content), **content}
    atomic_write_json(OUTPUT_PATH, artifact)
    print(json.dumps(artifact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
