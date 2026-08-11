from __future__ import annotations

from copy import deepcopy

import pytest

from afts_arc.experiment_safety import canonical_sha256
from afts_arc.varc_blind import build_query_blind_tasks
from afts_arc.varc_run import VARC_TTT_CONFIG, build_varc_run_receipt


def _manifest() -> dict[str, object]:
    _, manifest = build_query_blind_tasks(
        challenges={
            "task_a": {
                "train": [{"input": [[1]], "output": [[2]]}],
                "test": [{"input": [[3]]}],
            },
            "task_b": {
                "train": [{"input": [[4]], "output": [[5]]}],
                "test": [{"input": [[6]]}],
            },
        },
        source_cohort_id="source",
        source_challenges_sha256="a" * 64,
    )
    return manifest


def _statuses() -> dict[str, object]:
    return {
        "task_a": {
            "elapsed_seconds": 10,
            "end_epoch": 110,
            "exit_code": 0,
            "gpu_id": 0,
            "start_epoch": 100,
            "task_id": "task_a",
        },
        "task_b": {
            "elapsed_seconds": 20,
            "end_epoch": 220,
            "exit_code": 0,
            "gpu_id": 1,
            "start_epoch": 200,
            "task_id": "task_b",
        },
    }


def _predictions() -> dict[str, object]:
    return {
        "task_a": {"bytes": 10, "sha256": "b" * 64},
        "task_b": {"bytes": 20, "sha256": "c" * 64},
    }


def _compatibility() -> dict[str, object]:
    body: dict[str, object] = {
        "blind_cohort_id": _manifest()["blind_cohort_id"],
        "schema": "afts.varc-diagnostic-compatibility/v1",
        "tasks": [
            {
                "augmentation_count": 10,
                "augmentation_set_sha256": "d" * 64,
                "evaluation_sha256": "e" * 64,
                "task_id": "task_a",
            },
            {
                "augmentation_count": 10,
                "augmentation_set_sha256": "f" * 64,
                "evaluation_sha256": "0" * 64,
                "task_id": "task_b",
            },
        ],
    }
    return {"compatibility_id": canonical_sha256(body), **body}


def _receipt() -> dict[str, object]:
    return build_varc_run_receipt(
        blind_manifest=_manifest(),
        task_statuses=_statuses(),
        prediction_files=_predictions(),
        source={"commit": "pinned"},
        runtime={"python": "frozen"},
        gpu_ids=(0, 1),
        compatibility_manifest=_compatibility(),
    )


def test_receipt_closes_costs_configuration_and_content_identity() -> None:
    receipt = _receipt()

    assert receipt["receipt_id"] == canonical_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_id"}
    )
    assert receipt["compute"] == {
        "failure_count": 0,
        "gpu_ids": [0, 1],
        "gpu_seconds": 30,
        "max_task_seconds": 20,
        "task_count": 2,
    }
    assert receipt["test_time_configuration"] == VARC_TTT_CONFIG
    assert receipt["test_time_configuration"]["epoch_argument"] == 100
    assert receipt["test_time_configuration"]["executed_epoch_count"] == 101
    assert receipt["query_blind_protocol"] == {
        "candidate_logits_consume": ["attention_mask", "inputs", "task_ids"],
        "provider_visible_test_output": "exact copy of corresponding test input",
        "query_gold_read": False,
        "test_output_role": "input-derived sentinel for loader shape only",
        "training_examples": "demonstrations only",
        "candidate_path_uses_original_task_and_variant_names": True,
        "diagnostic_alias_role": (
            "post-candidate upstream sentinel diagnostic only"
        ),
        "single_gpu_diagnostic_alias_serialization": True,
        "upstream_logged_score": "sentinel identity diagnostic; not ARC accuracy",
    }


def test_receipt_rejects_failure_elapsed_mismatch_and_missing_prediction() -> None:
    failed = _statuses()
    failed["task_a"]["exit_code"] = 1
    with pytest.raises(ValueError, match="task failed"):
        build_varc_run_receipt(
            blind_manifest=_manifest(),
            task_statuses=failed,
            prediction_files=_predictions(),
            source={"commit": "pinned"},
            runtime={"python": "frozen"},
            gpu_ids=(0, 1),
            compatibility_manifest=_compatibility(),
        )

    elapsed = _statuses()
    elapsed["task_a"]["elapsed_seconds"] = 11
    with pytest.raises(ValueError, match="elapsed time is inconsistent"):
        build_varc_run_receipt(
            blind_manifest=_manifest(),
            task_statuses=elapsed,
            prediction_files=_predictions(),
            source={"commit": "pinned"},
            runtime={"python": "frozen"},
            gpu_ids=(0, 1),
            compatibility_manifest=_compatibility(),
        )

    missing = _predictions()
    del missing["task_b"]
    with pytest.raises(ValueError, match="task sets differ"):
        build_varc_run_receipt(
            blind_manifest=_manifest(),
            task_statuses=_statuses(),
            prediction_files=missing,
            source={"commit": "pinned"},
            runtime={"python": "frozen"},
            gpu_ids=(0, 1),
            compatibility_manifest=_compatibility(),
        )


def test_receipt_rejects_undeclared_gpu_and_forgery_changes_identity() -> None:
    undeclared = _statuses()
    undeclared["task_b"]["gpu_id"] = 2
    with pytest.raises(ValueError, match="undeclared GPU"):
        build_varc_run_receipt(
            blind_manifest=_manifest(),
            task_statuses=undeclared,
            prediction_files=_predictions(),
            source={"commit": "pinned"},
            runtime={"python": "frozen"},
            gpu_ids=(0, 1),
            compatibility_manifest=_compatibility(),
        )

    changed = deepcopy(_predictions())
    changed["task_a"]["sha256"] = "d" * 64
    changed_receipt = build_varc_run_receipt(
        blind_manifest=_manifest(),
        task_statuses=_statuses(),
        prediction_files=changed,
        source={"commit": "pinned"},
        runtime={"python": "frozen"},
        gpu_ids=(0, 1),
        compatibility_manifest=_compatibility(),
    )
    assert changed_receipt["receipt_id"] != _receipt()["receipt_id"]
