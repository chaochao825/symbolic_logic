from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "summarize_object_program_workspace_results.py"
)


def _script_module() -> object:
    spec = importlib.util.spec_from_file_location(
        "summarize_object_program_workspace_results", SCRIPT
    )
    if spec is None or spec.loader is None:
        raise AssertionError("could not load Object–Program Workspace summarizer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_pair(directory: Path, stem: str, value: dict[str, object]) -> None:
    payload = json.dumps(value, sort_keys=True).encode("utf-8")
    (directory / f"{stem}_a.json").write_bytes(payload)
    (directory / f"{stem}_b.json").write_bytes(payload)


def _fixture(directory: Path) -> None:
    freeze = {
        "dsl_version": "test-dsl/v1",
        "freeze_id": "freeze-id",
        "scientific_lane": "outcome_exposed_development",
        "task_count": 1,
        "tasks": [
            {
                "frontier_opportunity": False,
                "program_opportunity": True,
                "selection_reason": "eligible",
                "task_id": "private-task-id",
            }
        ],
    }
    result = {
        "action_interventions": {
            "visual_vs_residual_selected_set_changes": 0,
            "visual_vs_shuffled_selected_set_changes": 0,
        },
        "candidate_freeze_id": "freeze-id",
        "controlled_semantic_gate": {"applicable": False, "passed": False},
        "controller_training_started": False,
        "development_screen": {
            "applicable": False,
            "prospective_confirmation_authorized": False,
        },
        "opportunity_count": 0,
        "program_opportunity_count": 1,
        "public_evaluation_read": False,
        "query_gold_read_after_freeze": True,
        "result_id": "result-id",
        "scientific_lane": "outcome_exposed_development",
        "strict_task_coverage": {"base_prefix": 1, "visual_typed": 1},
        "task_count": 1,
        "tasks": [
            {
                "frontier_opportunity": False,
                "program_opportunity": True,
                "task_id": "private-task-id",
            }
        ],
        "unique_recovery_over_base_prefix": {"visual_typed": 0},
        "visual_unique_over_all_controls": 0,
    }
    _write_pair(directory, "candidate_freeze", freeze)
    _write_pair(directory, "result", result)
    (directory / "raw_payload.json").write_text('{"secret": true}\n', encoding="utf-8")


def test_build_is_deterministic_and_omits_task_payloads(tmp_path: Path) -> None:
    module = _script_module()
    _fixture(tmp_path)

    summary = module.build(tmp_path)
    first_summary = (tmp_path / "summary.json").read_bytes()
    first_manifest = (tmp_path / "artifact_sha256.json").read_bytes()
    repeated = module.build(tmp_path)

    assert repeated == summary
    assert (tmp_path / "summary.json").read_bytes() == first_summary
    assert (tmp_path / "artifact_sha256.json").read_bytes() == first_manifest
    assert summary["program_only_opportunity_count"] == 1
    assert summary["output_frontier_opportunity_count"] == 0
    assert "private-task-id" not in first_summary.decode("utf-8")


def test_build_rejects_nonidentical_replay(tmp_path: Path) -> None:
    module = _script_module()
    _fixture(tmp_path)
    (tmp_path / "result_b.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="result replay mismatch"):
        module.build(tmp_path)
