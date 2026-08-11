from __future__ import annotations

from copy import deepcopy

import pytest

from afts_arc.experiment_safety import canonical_sha256
from afts_arc.nvarc_run import (
    NVARC_EXPECTED_METRIC_NAMES,
    NVARC_REFERENCE_CONFIG,
    build_nvarc_run_receipt,
)


def _completion() -> dict[str, object]:
    return {
        "attempts_per_query": 10,
        "dropped_final_batch_size": 31,
        "eval_batch_count": 108,
        "exact_replayed_training_steps": 6186,
        "exit_code": 0,
        "group_order_size": 200000,
        "groups_consumed": 200000,
        "metadata_estimated_training_steps": 6288,
        "observed_training_steps": 6186,
        "query_count": 113,
        "task_count": 100,
    }


def _receipt() -> dict[str, object]:
    metrics = {name: 0.1 for name in NVARC_EXPECTED_METRIC_NAMES}
    metrics["all/steps"] = 10.0
    return build_nvarc_run_receipt(
        cohort_id="cohort",
        configuration=NVARC_REFERENCE_CONFIG,
        completion=_completion(),
        expected_completion=_completion(),
        artifacts={"submission": "a" * 64},
        source={"commit": "pinned"},
        runtime={"gpu": "A800"},
        upstream_metrics=metrics,
    )


def test_receipt_closes_reference_workload_and_query_boundary() -> None:
    receipt = _receipt()

    assert receipt["receipt_id"] == canonical_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_id"}
    )
    assert receipt["completion"] == _completion()
    assert receipt["query_label_boundary"] == {
        "candidate_model_updates_use_query_gold": False,
        "candidate_ordering_uses_query_gold": False,
        "project_solution_read_before_submission_freeze": False,
        "public_evaluation_read": False,
        "upstream_evaluator_reads_query_gold_for_metrics": True,
    }


def test_receipt_rejects_configuration_or_incomplete_workload() -> None:
    metrics = {name: 0.1 for name in NVARC_EXPECTED_METRIC_NAMES}
    metrics["all/steps"] = 10.0
    changed_config = dict(NVARC_REFERENCE_CONFIG)
    changed_config["epochs"] = 1
    with pytest.raises(ValueError, match="configuration differs"):
        build_nvarc_run_receipt(
            cohort_id="cohort",
            configuration=changed_config,
            completion=_completion(),
            expected_completion=_completion(),
            artifacts={"submission": "a" * 64},
            source={"commit": "pinned"},
            runtime={"gpu": "A800"},
            upstream_metrics=metrics,
        )

    incomplete = _completion()
    incomplete["observed_training_steps"] = 6185
    with pytest.raises(ValueError, match="did not complete"):
        build_nvarc_run_receipt(
            cohort_id="cohort",
            configuration=NVARC_REFERENCE_CONFIG,
            completion=incomplete,
            expected_completion=_completion(),
            artifacts={"submission": "a" * 64},
            source={"commit": "pinned"},
            runtime={"gpu": "A800"},
            upstream_metrics=metrics,
        )


def test_receipt_rejects_nonfinite_metrics_and_hashes_all_evidence() -> None:
    nonfinite_metrics = {name: 0.1 for name in NVARC_EXPECTED_METRIC_NAMES}
    nonfinite_metrics["all/lm_loss"] = float("nan")
    nonfinite_metrics["all/steps"] = 10.0
    with pytest.raises(ValueError, match="not finite"):
        build_nvarc_run_receipt(
            cohort_id="cohort",
            configuration=NVARC_REFERENCE_CONFIG,
            completion=_completion(),
            expected_completion=_completion(),
            artifacts={"submission": "a" * 64},
            source={"commit": "pinned"},
            runtime={"gpu": "A800"},
            upstream_metrics=nonfinite_metrics,
        )

    changed = deepcopy(_receipt())
    changed["artifacts"]["submission"] = "b" * 64
    changed_content = {
        key: value for key, value in changed.items() if key != "receipt_id"
    }
    assert canonical_sha256(changed_content) != _receipt()["receipt_id"]
