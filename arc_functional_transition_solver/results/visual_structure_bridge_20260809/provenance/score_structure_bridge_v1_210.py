"""Release query gold after the pre-gold freeze and score the frozen bridge."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/home/wangmeiqi/codex_runs/arc_visual_posterior_bridge_20260809")
REPOSITORY = Path(
    "/home/wangmeiqi/codex_runs/symbolic_logic_arc_online_control_20260724/repo"
    "/arc_functional_transition_solver"
)
sys.path.insert(0, str(REPOSITORY / "src"))

from afts_arc.experiment_safety import atomic_write_json, canonical_sha256  # noqa: E402
from afts_arc.visual_provider_gate import (  # noqa: E402
    _file_sha256,
    _load_json_object,
    freeze_and_score_provider,
)
from afts_arc.visual_structure_bridge import (  # noqa: E402
    score_visual_structure_bridge,
)


MANIFEST_PATH = ROOT / "cohort_v1" / "manifest.json"
GOLD_TASK_DIR = ROOT / "cohort_v1" / "gold_data" / "training"
PREDICTION_ROOT = ROOT / "from236_structure_v1" / "raw_predictions"
CONTRACT_PATH = ROOT / "provider_contract_structure_v1.json"
PRE_GOLD_FROZEN = ROOT / "pre_gold" / "frozen_predictions.json"
PRE_GOLD_CANDIDATE = ROOT / "pre_gold" / "structure_candidates.json"
PRE_GOLD_RECEIPT = ROOT / "pre_gold" / "freeze_receipt.json"
RELEASE_START_PATH = ROOT / "gold_release_start_v1.json"
BASELINE_SUMMARY = ROOT / "baseline_structure_v1" / "summary.json"
VISUAL_SCORE_DIR = ROOT / "visual_score_structure_v1"
STRUCTURE_SCORE_PATH = ROOT / "structure_score_v1.json"
RELEASE_RECEIPT_PATH = ROOT / "gold_release_receipt_v1.json"


def _verify_content_id(payload: dict[str, object], field: str) -> None:
    content = dict(payload)
    expected = content.pop(field)
    if canonical_sha256(content) != expected:
        raise ValueError(f"invalid content ID: {field}")


def main() -> None:
    for path in (VISUAL_SCORE_DIR, STRUCTURE_SCORE_PATH, RELEASE_RECEIPT_PATH):
        if path.exists():
            raise FileExistsError(f"score artifact already exists: {path}")
    receipt = _load_json_object(PRE_GOLD_RECEIPT)
    _verify_content_id(receipt, "freeze_id")
    if receipt["gold_scoring_started"] is not False:
        raise ValueError("pre-gold receipt is not a pre-scoring freeze")
    if receipt["cohort_manifest_sha256"] != _file_sha256(MANIFEST_PATH):
        raise ValueError("pre-gold receipt does not bind the cohort manifest")
    if receipt["provider_contract_sha256"] != _file_sha256(CONTRACT_PATH):
        raise ValueError("pre-gold receipt does not bind the provider contract")
    if receipt["frozen_prediction_sha256"] != _file_sha256(PRE_GOLD_FROZEN):
        raise ValueError("pre-gold receipt does not bind frozen predictions")
    if receipt["structure_candidate_sha256"] != _file_sha256(
        PRE_GOLD_CANDIDATE
    ):
        raise ValueError("pre-gold receipt does not bind structure candidates")
    release_start = _load_json_object(RELEASE_START_PATH)
    _verify_content_id(release_start, "release_start_id")
    if release_start["pre_gold_freeze_id"] != receipt["freeze_id"]:
        raise ValueError("gold-release start does not bind the pre-gold freeze")
    if release_start["post_outcome_tuning_authorized"] is not False:
        raise ValueError("gold-release start permits post-outcome tuning")

    contract = _load_json_object(CONTRACT_PATH)
    _verify_content_id(contract, "contract_id")
    visual = freeze_and_score_provider(
        cohort_manifest=MANIFEST_PATH,
        gold_training_dir=GOLD_TASK_DIR,
        prediction_roots=(PREDICTION_ROOT,),
        baseline_summary=BASELINE_SUMMARY,
        object_provider_summary=None,
        provider_contract=contract,
        output_dir=VISUAL_SCORE_DIR,
        pilot_unique_gate=1,
        invalid_candidate_policy="reject",
    )
    rescored_frozen = _load_json_object(VISUAL_SCORE_DIR / "frozen_predictions.json")
    if rescored_frozen["frozen_prediction_id"] != receipt["frozen_prediction_id"]:
        raise ValueError("post-release freeze differs from the pre-gold freeze")
    structure = score_visual_structure_bridge(
        cohort_manifest=MANIFEST_PATH,
        gold_task_dir=GOLD_TASK_DIR,
        candidate_artifact=PRE_GOLD_CANDIDATE,
        visual_score_summary=VISUAL_SCORE_DIR / "summary.json",
        output_path=STRUCTURE_SCORE_PATH,
    )
    release_content = {
        "schema": "afts.visual-structure-gold-release/v1",
        "gold_scoring_completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "gold_release_start_id": release_start["release_start_id"],
        "gold_release_start_sha256": _file_sha256(RELEASE_START_PATH),
        "pre_gold_freeze_id": receipt["freeze_id"],
        "pre_gold_freeze_sha256": _file_sha256(PRE_GOLD_RECEIPT),
        "baseline_summary_sha256": _file_sha256(BASELINE_SUMMARY),
        "visual_result_id": visual["result_id"],
        "visual_summary_sha256": _file_sha256(VISUAL_SCORE_DIR / "summary.json"),
        "structure_result_id": structure["result_id"],
        "structure_summary_sha256": _file_sha256(STRUCTURE_SCORE_PATH),
    }
    release = {"release_id": canonical_sha256(release_content), **release_content}
    atomic_write_json(RELEASE_RECEIPT_PATH, release)
    print(
        json.dumps(
            {
                "release_id": release["release_id"],
                "visual_metrics": visual["metrics"],
                "structure_metrics": structure["metrics"],
                "decision": structure["decision"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
