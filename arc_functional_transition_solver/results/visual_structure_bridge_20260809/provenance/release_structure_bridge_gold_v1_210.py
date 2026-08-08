"""Authorize frozen baseline and endpoint scoring after the pre-gold freeze."""

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


PRE_GOLD_RECEIPT = ROOT / "pre_gold" / "freeze_receipt.json"
OUTPUT_PATH = ROOT / "gold_release_start_v1.json"


def main() -> None:
    if OUTPUT_PATH.exists():
        raise FileExistsError(f"gold-release artifact already exists: {OUTPUT_PATH}")
    for path in (
        ROOT / "baseline_structure_v1",
        ROOT / "visual_score_structure_v1",
        ROOT / "structure_score_v1.json",
    ):
        if path.exists():
            raise FileExistsError(f"post-gold artifact exists before release: {path}")
    freeze = _load_json_object(PRE_GOLD_RECEIPT)
    freeze_content = dict(freeze)
    freeze_id = freeze_content.pop("freeze_id")
    if canonical_sha256(freeze_content) != freeze_id:
        raise ValueError("invalid pre-gold freeze content ID")
    if freeze["gold_scoring_started"] is not False:
        raise ValueError("freeze receipt does not precede gold scoring")
    if freeze["replay"]["byte_identical"] is not True:
        raise ValueError("pre-gold replay is not byte-identical")

    content = {
        "schema": "afts.visual-structure-gold-release-start/v1",
        "released_at_utc": datetime.now(timezone.utc).isoformat(),
        "pre_gold_freeze_id": freeze_id,
        "pre_gold_freeze_sha256": _file_sha256(PRE_GOLD_RECEIPT),
        "authorized_steps": [
            "run frozen portfolio baseline",
            "score frozen visual posterior",
            "score frozen structure candidate",
        ],
        "post_outcome_tuning_authorized": False,
        "controller_training_authorized": False,
    }
    artifact = {"release_start_id": canonical_sha256(content), **content}
    atomic_write_json(OUTPUT_PATH, artifact)
    print(json.dumps(artifact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
