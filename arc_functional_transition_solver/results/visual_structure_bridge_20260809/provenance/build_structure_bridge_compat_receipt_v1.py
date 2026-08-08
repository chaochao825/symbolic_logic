"""Record byte-identical replay after the JSON sequence compatibility fix."""

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
from afts_arc.visual_provider_gate import _file_sha256, _load_json_object  # noqa: E402


OUTPUT_PATH = ROOT / "compatibility_fix_receipt_v1.json"


def main() -> None:
    if OUTPUT_PATH.exists():
        raise FileExistsError(f"compatibility receipt already exists: {OUTPUT_PATH}")
    original_frozen = ROOT / "pre_gold" / "frozen_predictions.json"
    replay_frozen = ROOT / "post_result_compat_replay" / "frozen_predictions.json"
    original_candidates = ROOT / "pre_gold" / "structure_candidates.json"
    replay_candidates = (
        ROOT / "post_result_compat_replay" / "structure_candidates.json"
    )
    if original_frozen.read_bytes() != replay_frozen.read_bytes():
        raise ValueError("frozen-prediction replay differs")
    if original_candidates.read_bytes() != replay_candidates.read_bytes():
        raise ValueError("structure-candidate replay differs")
    score = _load_json_object(ROOT / "structure_score_v1.json")
    if score["decision"]["classification"] != "null":
        raise ValueError("compatibility receipt expected the frozen null result")

    content = {
        "schema": "afts.visual-structure-json-compatibility/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "change": (
            "emit shape_delta as a JSON-native list instead of a tuple so a "
            "serialized stable contract compares identically after reload"
        ),
        "algorithm_artifact_byte_identical": True,
        "metric_artifact_recomputed": False,
        "frozen_result_classification": "null",
        "original_frozen_prediction_sha256": _file_sha256(original_frozen),
        "replay_frozen_prediction_sha256": _file_sha256(replay_frozen),
        "original_structure_candidate_sha256": _file_sha256(original_candidates),
        "replay_structure_candidate_sha256": _file_sha256(replay_candidates),
        "updated_source_sha256": _file_sha256(
            REPOSITORY / "src" / "afts_arc" / "visual_structure_bridge.py"
        ),
        "updated_test_sha256": _file_sha256(
            REPOSITORY / "tests" / "test_visual_structure_bridge.py"
        ),
    }
    artifact = {"receipt_id": canonical_sha256(content), **content}
    atomic_write_json(OUTPUT_PATH, artifact)
    print(json.dumps(artifact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
