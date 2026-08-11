"""Transparent post-run provenance attestation for a development-only anchor."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from afts_arc.experiment_safety import canonical_sha256


NVARC_POSTHOC_ATTESTATION_SCHEMA = "afts.nvarc-posthoc-attestation/v1"


def build_nvarc_posthoc_attestation(
    *,
    cohort_id: str,
    completion: Mapping[str, object],
    artifacts: Mapping[str, str],
    source: Mapping[str, object],
    integrity_checks: Mapping[str, bool],
    limitations: Sequence[str],
) -> dict[str, object]:
    """Record valid development evidence without pretending it was pre-run."""

    if not cohort_id:
        raise ValueError("cohort_id must not be empty")
    required_completion = {
        "eval_batch_count",
        "exit_code",
        "expected_training_steps",
        "observed_training_steps",
        "query_count",
        "task_count",
    }
    if set(completion) != required_completion:
        raise ValueError("posthoc completion fields differ from the contract")
    if completion["exit_code"] != 0:
        raise ValueError("posthoc attestation requires a successful run")
    if completion["observed_training_steps"] != completion["expected_training_steps"]:
        raise ValueError("observed training steps differ from the audited schedule")
    if not artifacts or not all(
        isinstance(digest, str) and len(digest) == 64
        for digest in artifacts.values()
    ):
        raise ValueError("posthoc artifacts must contain SHA-256 digests")
    if not source:
        raise ValueError("posthoc source evidence must not be empty")
    required_checks = {
        "checkpoint_loss_source_matches_current",
        "checkpoint_model_source_matches_current",
        "nvarc_commit_matches_run_start",
        "nvarc_status_matches_run_start",
    }
    if set(integrity_checks) != required_checks:
        raise ValueError("posthoc integrity-check fields differ from the contract")
    if not all(integrity_checks.values()):
        raise ValueError("a posthoc integrity check failed")
    normalized_limitations = tuple(limitations)
    if not normalized_limitations or any(not item for item in normalized_limitations):
        raise ValueError("posthoc limitations must be explicit and non-empty")

    content: dict[str, object] = {
        "schema": NVARC_POSTHOC_ATTESTATION_SCHEMA,
        "scientific_scope": "development-only static anchor diagnostic",
        "cohort_id": cohort_id,
        "provenance_capture_phase": "post_run",
        "project_solution_read_before_candidate_freeze": True,
        "candidate_model_updates_use_query_gold": False,
        "candidate_ordering_uses_query_gold": False,
        "completion": dict(completion),
        "artifacts": dict(sorted(artifacts.items())),
        "source": dict(source),
        "integrity_checks": dict(integrity_checks),
        "limitations": list(normalized_limitations),
    }
    return {"attestation_id": canonical_sha256(content), **content}
