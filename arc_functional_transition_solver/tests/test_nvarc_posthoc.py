from __future__ import annotations

import pytest

from afts_arc.experiment_safety import canonical_sha256
from afts_arc.nvarc_posthoc import build_nvarc_posthoc_attestation


def _attestation() -> dict[str, object]:
    return build_nvarc_posthoc_attestation(
        cohort_id="development",
        completion={
            "eval_batch_count": 2,
            "exit_code": 0,
            "expected_training_steps": 3,
            "observed_training_steps": 3,
            "query_count": 4,
            "task_count": 4,
        },
        artifacts={"submission": "a" * 64},
        source={"commit": "pinned"},
        integrity_checks={
            "checkpoint_loss_source_matches_current": True,
            "checkpoint_model_source_matches_current": True,
            "nvarc_commit_matches_run_start": True,
            "nvarc_status_matches_run_start": True,
        },
        limitations=("source metadata was captured after completion",),
    )


def test_posthoc_attestation_is_content_addressed_and_explicitly_exposed() -> None:
    attestation = _attestation()

    assert attestation["attestation_id"] == canonical_sha256(
        {key: value for key, value in attestation.items() if key != "attestation_id"}
    )
    assert attestation["provenance_capture_phase"] == "post_run"
    assert attestation["project_solution_read_before_candidate_freeze"] is True


def test_posthoc_attestation_rejects_failed_integrity_or_wrong_step() -> None:
    completion = dict(_attestation()["completion"])
    completion["observed_training_steps"] = 2
    with pytest.raises(ValueError, match="observed training steps"):
        build_nvarc_posthoc_attestation(
            cohort_id="development",
            completion=completion,
            artifacts={"submission": "a" * 64},
            source={"commit": "pinned"},
            integrity_checks={
                "checkpoint_loss_source_matches_current": True,
                "checkpoint_model_source_matches_current": True,
                "nvarc_commit_matches_run_start": True,
                "nvarc_status_matches_run_start": True,
            },
            limitations=("captured after completion",),
        )
