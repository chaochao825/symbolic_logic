from __future__ import annotations

from afts_arc.experiment_safety import canonical_sha256
from afts_arc.object_program_workspace_controls import (
    build_object_program_workspace_controls,
)


def test_control_cohort_is_deterministic_content_addressed_and_balanced() -> None:
    first = build_object_program_workspace_controls(task_count=12)
    second = build_object_program_workspace_controls(task_count=12)

    assert first == second
    manifest = first["manifest"]
    assert manifest["task_count"] == 12
    assert manifest["challenges_sha256"] == canonical_sha256(first["challenges"])
    assert manifest["solutions_sha256"] == canonical_sha256(first["solutions"])
    assert {row["operation"] for row in manifest["cases"]} == {
        "erase_component",
        "recolor_component",
    }
    assert {row["transform_mode"] for row in manifest["cases"]} == {0, 1, 2}
    assert first["visual_freeze"]["query_gold_read"] is False
    assert first["visual_freeze"]["freeze_id"] == canonical_sha256(
        {
            key: value
            for key, value in first["visual_freeze"].items()
            if key != "freeze_id"
        }
    )
