"""Pure, Torch-free shared fixture for terminal M04a budget evidence tests."""

from __future__ import annotations

import hashlib

from afts_arc import m04a_evidence as evidence
from afts_arc import m04a_preflight_command as command
from afts_arc.m04a_contract import canonical_sha256
from afts_arc.m04a_evidence import make_training_cost_ledger_row
from test_m04a_training_evidence_closure import _digest, _preflight_report


def _budget_failure_fixture() -> tuple[
    dict[str, object],
    tuple[dict[str, object], ...],
    dict[str, object],
    bytes,
]:
    rows = tuple(
        make_training_cost_ledger_row(
            phase="preflight_update",
            event_index=index,
            optimizer_step=index + 1,
            optimizer_updates=1,
            microbatches=evidence.PREFLIGHT_MICROBATCHES_PER_UPDATE,
            encoder_forward_calls=evidence.PREFLIGHT_MICROBATCHES_PER_UPDATE,
            decoder_forward_calls=evidence.PREFLIGHT_MICROBATCHES_PER_UPDATE,
            backward_calls=evidence.PREFLIGHT_MICROBATCHES_PER_UPDATE,
            masked_token_predictions=(
                evidence.PREFLIGHT_MASKED_TOKEN_PREDICTIONS_PER_UPDATE
            ),
            wall_time_ns=10_000_000_000,
            cuda_peak_allocated_bytes=2,
            cuda_peak_reserved_bytes=3,
        )
        for index in range(evidence.PREFLIGHT_UPDATES)
    )
    pass_report, parents = _preflight_report(
        rows,
        validation_outer_sha256=_digest("validation-outer"),
        validation_jsonl_sha256=_digest("validation-jsonl"),
        validation_summary_id=_digest("validation-summary"),
        validation_row_count=16,
    )
    training = pass_report["training"]
    inference = pass_report["inference"]
    overhead = parents["overhead_cost_probe_report"]
    diagnostic = parents["diagnostic_checkpoint_snapshot"]
    training_wall = int(training["total_update_wall_ns"])
    total_wall = training_wall + 100
    mean_update = (
        training_wall + evidence.PREFLIGHT_UPDATES - 1
    ) // evidence.PREFLIGHT_UPDATES
    microbatch_count = (
        evidence.PREFLIGHT_UPDATES * evidence.PREFLIGHT_MICROBATCHES_PER_UPDATE
    )
    mean_microbatch = (training_wall + microbatch_count - 1) // microbatch_count
    lock_probe = overhead["lock_handshake_probe"]
    fresh_probe = overhead["fresh_reconstruction_probe"]
    checkpoint_probe = overhead["checkpoint_cost_probe"]
    selection_probe = overhead["final_selection_probe"]
    preflight_started = int(pass_report["preflight_started_perf_counter_ns"])
    handshake_gap = preflight_started - int(
        lock_probe["handshake_completed_perf_counter_ns"]
    )
    validation_count = 16
    primary_wall = mean_update * evidence.OPTIMIZER_UPDATES
    validation_calls = evidence.VALIDATION_PASS_COUNT * validation_count
    validation_wall = mean_microbatch * validation_calls
    checkpoint_wall = evidence.VALIDATION_PASS_COUNT * int(
        checkpoint_probe["roundtrip_wall_ns"]
    )
    projected = (
        int(lock_probe["wall_ns"])
        + handshake_gap
        + total_wall
        + int(fresh_probe["total_wall_ns"])
        + primary_wall
        + validation_wall
        + checkpoint_wall
        + int(selection_probe["selection_wall_ns"])
        + evidence.PREFLIGHT_POST_VALIDATION_MARGIN_NS
    )
    projection: dict[str, object] = {
        "schema": evidence.PREFLIGHT_PROJECTION_SCHEMA_VERSION,
        "projection_method": evidence.PREFLIGHT_PROJECTION_METHOD,
        "measured_preflight_training_wall_ns": training_wall,
        "measured_preflight_total_wall_ns": total_wall,
        "mean_training_update_wall_ns": mean_update,
        "mean_training_microbatch_wall_ns": mean_microbatch,
        "lock_handshake_probe_id": lock_probe["probe_id"],
        "measured_lock_setup_handshake_wall_ns": lock_probe["wall_ns"],
        "measured_handshake_to_preflight_start_ns": handshake_gap,
        "fresh_reconstruction_probe_id": fresh_probe["probe_id"],
        "measured_fresh_reconstruction_wall_ns": fresh_probe["total_wall_ns"],
        "checkpoint_cost_probe_id": checkpoint_probe["probe_id"],
        "measured_checkpoint_roundtrip_wall_ns": checkpoint_probe[
            "roundtrip_wall_ns"
        ],
        "final_selection_probe_id": selection_probe["probe_id"],
        "measured_final_selection_wall_ns": selection_probe["selection_wall_ns"],
        "projected_primary_updates": evidence.OPTIMIZER_UPDATES,
        "projected_primary_wall_ns": primary_wall,
        "projected_validation_passes": evidence.VALIDATION_PASS_COUNT,
        "validation_episodes_per_pass": validation_count,
        "projected_validation_episode_calls": validation_calls,
        "projected_validation_wall_ns": validation_wall,
        "projected_checkpoint_writes": evidence.VALIDATION_PASS_COUNT,
        "projected_checkpoint_wall_ns": checkpoint_wall,
        "projected_final_selection_wall_ns": selection_probe["selection_wall_ns"],
        "post_preflight_validation_margin_ns": (
            evidence.PREFLIGHT_POST_VALIDATION_MARGIN_NS
        ),
        "projected_campaign_wall_ns": projected,
        "budget_limit_ns": evidence.GPU_HOUR_BUDGET_NS,
        "budget_remaining_ns": evidence.GPU_HOUR_BUDGET_NS - projected,
        "budget_status": "BUDGET_EXCEEDED",
    }
    if projected <= evidence.GPU_HOUR_BUDGET_NS:
        raise AssertionError("pure failure fixture must exceed the campaign budget")
    projection["projection_id"] = canonical_sha256(projection)

    commitment = {
        "artifact_filename": overhead["checkpoint_artifact_filename"],
        "artifact_status": overhead["checkpoint_artifact_status"],
        "artifact_disposition": checkpoint_probe["artifact_disposition"],
        "sha256": checkpoint_probe["checkpoint_sha256"],
        "bytes": checkpoint_probe["checkpoint_bytes"],
        "selectable_checkpoint_created": checkpoint_probe[
            "selectable_checkpoint_created"
        ],
    }
    if (
        len(diagnostic) != commitment["bytes"]
        or hashlib.sha256(diagnostic).hexdigest() != commitment["sha256"]
    ):
        raise AssertionError("pure failure fixture diagnostic does not close")
    semantic: dict[str, object] = {
        "schema": command.PREFLIGHT_FAILURE_REPORT_SCHEMA_VERSION,
        "status": "BUDGET_EXCEEDED",
        "failure_code": "BUDGET_EXCEEDED",
        "evidence_completeness": "full_budget_probe",
        "fixture_id": command._expected_preflight_fixture_id(),
        "runtime": pass_report["runtime"],
        "validation_manifest_commitment": pass_report[
            "validation_manifest_commitment"
        ],
        "lock_handshake_artifact_sha256": parents[
            "expected_lock_handshake_artifact_sha256"
        ],
        "preflight_started_perf_counter_ns": preflight_started,
        "completed_updates": evidence.PREFLIGHT_UPDATES,
        "ledger_row_count": len(rows),
        "ledger_rows_sha256": canonical_sha256(list(rows)),
        "training_summary": training,
        "inference_summary": inference,
        "overhead_cost_probe_id": overhead["probe_id"],
        "budget_projection": projection,
        "diagnostic_checkpoint_commitment": commitment,
        "dataset_file_reads": 0,
        "training_checkpoint_writes": 0,
        "diagnostic_checkpoint_writes": 1,
        "fallback_used": False,
        "training_evidence_eligible": False,
        "same_run_retry_allowed": False,
    }
    report = {**semantic, "report_id": canonical_sha256(semantic)}
    return report, rows, parents, diagnostic


__all__ = ["_budget_failure_fixture"]
