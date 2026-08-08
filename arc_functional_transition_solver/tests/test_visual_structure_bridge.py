from __future__ import annotations

import json
from pathlib import Path

import pytest

from afts_arc.experiment_safety import canonical_sha256
from afts_arc.visual_provider_gate import (
    COHORT_SCHEMA,
    FROZEN_PREDICTIONS_SCHEMA,
    SCORE_SCHEMA,
    VisualProviderGateError,
    _load_provider_predictions,
    _validate_provider_predictions,
)
from afts_arc.visual_structure_bridge import (
    compile_structure_certificate,
    freeze_visual_structure_bridge,
    score_visual_structure_bridge,
    stable_demo_contract,
    transition_features,
)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def test_typed_structure_certificates_distinguish_representations() -> None:
    canvas = compile_structure_certificate(
        task_id="deadbeef",
        query_index=0,
        contract={"shape_relation": "same"},
        signature={"shape_relation": "different"},
    )
    background = compile_structure_certificate(
        task_id="deadbeef",
        query_index=0,
        contract={"background_preserved": True},
        signature={"background_preserved": False},
    )
    objects = compile_structure_certificate(
        task_id="deadbeef",
        query_index=0,
        contract={"component_count_relation": "equal"},
        signature={"component_count_relation": "greater"},
    )
    assert canvas is not None
    assert background is not None
    assert objects is not None
    assert canvas["recommended_action"] == "canvas_reinfer"
    assert background["recommended_action"] == "reparse_background"
    assert objects["recommended_action"] == "object_rematch"
    assert (
        compile_structure_certificate(
            task_id="deadbeef",
            query_index=0,
            contract={"shape_relation": "same"},
            signature={"shape_relation": "same"},
        )
        is None
    )


def test_stable_demo_contract_drops_varying_fields() -> None:
    train = [
        {"input": [[0, 1], [0, 0]], "output": [[0, 1], [0, 0]]},
        {"input": [[0, 1], [1, 0]], "output": [[0, 1], [0, 0]]},
    ]
    contract = stable_demo_contract(train)
    assert contract["shape_relation"] == "same"
    assert contract["background_preserved"] is True
    assert "changed_pixel_count" not in contract
    assert json.loads(json.dumps(contract)) == contract


def test_structure_bridge_keeps_top_1_and_recovers_structural_minority(
    tmp_path: Path,
) -> None:
    task_id = "deadbeef"
    input_grid = [[0, 2, 0], [0, 0, 0], [0, 0, 0]]
    correct = input_grid
    wrong_majority = [[0, 3, 0], [0, 0, 0], [0, 0, 0]]
    wrong_second = [[0, 2, 0], [0, 0, 0], [0, 4, 0]]
    gold_task = {
        "train": [
            {"input": [[0, 1], [0, 0]], "output": [[0, 1], [0, 0]]},
            {"input": [[0, 0], [1, 0]], "output": [[0, 0], [1, 0]]},
        ],
        "test": [{"input": input_grid, "output": correct}],
    }
    blind_task = {
        "train": gold_task["train"],
        "test": [{"input": input_grid, "output": input_grid}],
    }
    manifest = {
        "cohort_id": "1" * 64,
        "schema": COHORT_SCHEMA,
        "tasks": [{"task_id": task_id, "query_count": 1}],
    }
    prediction_root = tmp_path / "predictions"
    prediction_path = prediction_root / f"{task_id}_predictions.json"
    samples = [wrong_majority] * 5 + [wrong_second] * 4 + [correct]
    write_json(prediction_path, {"0": samples})
    manifest_path = tmp_path / "manifest.json"
    blind_dir = tmp_path / "blind"
    gold_dir = tmp_path / "gold"
    write_json(manifest_path, manifest)
    write_json(blind_dir / f"{task_id}.json", blind_task)
    write_json(gold_dir / f"{task_id}.json", gold_task)

    raw, _ = _load_provider_predictions((task_id,), (prediction_root,))
    validated, validation = _validate_provider_predictions(
        raw, invalid_candidate_policy="reject"
    )
    frozen = {
        "schema": FROZEN_PREDICTIONS_SCHEMA,
        "cohort_id": manifest["cohort_id"],
        "frozen_prediction_id": "2" * 64,
        "raw_prediction_payload_sha256": canonical_sha256(raw),
        "validated_prediction_payload_sha256": canonical_sha256(validated),
        "prediction_validation": validation,
    }
    frozen_path = tmp_path / "frozen.json"
    write_json(frozen_path, frozen)
    candidate_path = tmp_path / "candidate.json"
    candidate = freeze_visual_structure_bridge(
        cohort_manifest=manifest_path,
        blind_task_dir=blind_dir,
        prediction_roots=(prediction_root,),
        frozen_predictions=frozen_path,
        output_path=candidate_path,
        invalid_candidate_policy="reject",
    )
    query = candidate["tasks"][0]["queries"][0]
    assert query["frequency_top_2"] == [wrong_majority, wrong_second]
    assert query["hybrid_top_2"] == [wrong_majority, correct]
    assert query["certificate"]["recommended_action"] == "object_rematch"

    score_summary = {
        "schema": SCORE_SCHEMA,
        "cohort_id": manifest["cohort_id"],
        "frozen_prediction_id": frozen["frozen_prediction_id"],
        "tasks": [
            {
                "task_id": task_id,
                "queries": [{"raw_oracle_covered": True}],
            }
        ],
    }
    score_summary_path = tmp_path / "visual_score.json"
    write_json(score_summary_path, score_summary)
    result = score_visual_structure_bridge(
        cohort_manifest=manifest_path,
        gold_task_dir=gold_dir,
        candidate_artifact=candidate_path,
        visual_score_summary=score_summary_path,
        output_path=tmp_path / "score.json",
    )
    assert result["metrics"]["frequency_query_pass_at_2"] == 0
    assert result["metrics"]["hybrid_query_pass_at_2"] == 1
    assert result["metrics"]["hybrid_unique_query_recovery"] == 1
    assert result["metrics"]["net_query_pass_at_2_delta"] == 1
    assert transition_features(
        tuple(tuple(row) for row in input_grid),
        tuple(tuple(row) for row in correct),
    )["changed_pixel_count"] == 0


def test_structure_score_rejects_visual_score_task_set_mismatch(
    tmp_path: Path,
) -> None:
    manifest = {
        "cohort_id": "1" * 64,
        "schema": COHORT_SCHEMA,
        "tasks": [{"task_id": "deadbeef", "query_count": 1}],
    }
    candidate_content = {
        "schema": "afts.visual-structure-bridge-candidates/v1",
        "cohort_id": manifest["cohort_id"],
        "frozen_prediction_id": "2" * 64,
        "invalid_candidate_policy": "reject",
        "prediction_files": [],
        "prediction_validation": {},
        "validated_prediction_payload_sha256": "3" * 64,
        "method": {},
        "tasks": [{"task_id": "deadbeef", "queries": []}],
    }
    candidate = {
        "candidate_id": canonical_sha256(candidate_content),
        **candidate_content,
    }
    visual_score = {
        "schema": SCORE_SCHEMA,
        "cohort_id": manifest["cohort_id"],
        "tasks": [{"task_id": "cafebabe", "queries": []}],
    }
    manifest_path = tmp_path / "manifest.json"
    candidate_path = tmp_path / "candidate.json"
    visual_score_path = tmp_path / "visual-score.json"
    write_json(manifest_path, manifest)
    write_json(candidate_path, candidate)
    write_json(visual_score_path, visual_score)

    with pytest.raises(
        VisualProviderGateError, match="visual score task IDs differ from cohort"
    ):
        score_visual_structure_bridge(
            cohort_manifest=manifest_path,
            gold_task_dir=tmp_path / "gold",
            candidate_artifact=candidate_path,
            visual_score_summary=visual_score_path,
            output_path=tmp_path / "output.json",
        )
