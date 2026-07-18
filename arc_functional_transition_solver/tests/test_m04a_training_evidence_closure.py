from __future__ import annotations

import copy
import hashlib
import struct
import unittest

from afts_arc import m04a_evidence as evidence
from afts_arc.m04a_contract import canonical_sha256
from afts_arc.m04a_evidence import (
    CHECKPOINT_SELECTION_RULE,
    PREFLIGHT_REPORT_SCHEMA_VERSION,
    TRAINING_COST_SUMMARY_SCHEMA_VERSION,
    make_checkpoint_manifest_row,
    make_selected_checkpoint_manifest,
    make_training_cost_ledger_row,
    make_training_cost_summary,
    make_training_lock_interval,
    make_validation_metric_row,
    sequential_fp32_mean,
    validate_checkpoint_manifest_row,
    validate_preflight_artifact,
    validate_selected_checkpoint_manifest,
    validate_training_checkpoint_artifacts,
    validate_training_cost_summary,
    validate_training_lock_interval,
    validate_optimizer_schedule_state,
    validate_validation_metric_row,
)
from afts_arc.m04a_train_contract import learning_rate_for_update
from afts_arc.manifest import serialize_json, serialize_jsonl


GPU_BUDGET_NS = 86_400_000_000_000


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


class M04AOptimizerScheduleStateTests(unittest.TestCase):
    def test_state_requires_exact_production_data_closure(self) -> None:
        state = {
            "schema": evidence.OPTIMIZER_SCHEDULE_STATE_SCHEMA_VERSION,
            "optimizer_step": evidence.OPTIMIZER_UPDATES,
            "learning_rate_hex": learning_rate_for_update(
                evidence.OPTIMIZER_UPDATES
            ).hex(),
            "checkpoint_count": evidence.VALIDATION_PASS_COUNT,
            "production_data_closure_id": evidence.PRODUCTION_DATA_CLOSURE_ID,
        }
        self.assertEqual(validate_optimizer_schedule_state(state), state)
        changed = dict(state)
        changed["production_data_closure_id"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "campaign/data closure"):
            validate_optimizer_schedule_state(changed)


def _reseal(payload: dict[str, object], identity: str) -> None:
    semantic = dict(payload)
    semantic.pop(identity, None)
    payload[identity] = canonical_sha256(semantic)


def _preflight_rows() -> tuple[dict[str, object], ...]:
    return tuple(
        make_training_cost_ledger_row(
            phase="preflight_update",
            event_index=index,
            optimizer_step=index + 1,
            optimizer_updates=1,
            microbatches=16,
            encoder_forward_calls=16,
            decoder_forward_calls=16,
            backward_calls=16,
            masked_token_predictions=14_400,
            wall_time_ns=1,
            cuda_peak_allocated_bytes=2,
            cuda_peak_reserved_bytes=3,
        )
        for index in range(100)
    )


def _validation_commitment(
    outer_sha256: str, jsonl_sha256: str, summary_id: str, row_count: int
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": evidence.PREFLIGHT_VALIDATION_COMMITMENT_SCHEMA_VERSION,
        "outer_artifact_manifest_sha256": outer_sha256,
        "jsonl_sha256": jsonl_sha256,
        "summary_id": summary_id,
        "row_count": row_count,
    }
    payload["commitment_id"] = canonical_sha256(payload)
    return payload


def _preflight_report(
    rows: tuple[dict[str, object], ...],
    *,
    validation_outer_sha256: str,
    validation_jsonl_sha256: str,
    validation_summary_id: str,
    validation_row_count: int,
) -> tuple[dict[str, object], dict[str, object]]:
    ledger_sha = canonical_sha256(list(rows))
    mask_trace = evidence._mask_count_trace_900()
    training_wall = sum(int(row["wall_time_ns"]) for row in rows)
    inference: dict[str, object] = {
        "schema": evidence.PREFLIGHT_INFERENCE_SCHEMA_VERSION,
        "lane_count": 8,
        "denoising_steps": 12,
        "encoder_batch_calls": 1,
        "decoder_batch_calls": 12,
        "sample_equivalent_forward_calls": 96,
        "masked_token_predictions": 8 * sum(mask_trace[:-1]),
        "padded_encoder_batch_shape": [21, 901],
        "unpadded_memory_length": 18_921,
        "cache_dtype": "bfloat16",
        "decoder_batch_sizes": [8] * 12,
        "mask_count_trace": list(mask_trace),
        "lane_trace_sha256": [_digest(f"trace-{index}") for index in range(8)],
        "final_output_keys": [_digest(f"output-{index}") for index in range(8)],
        "unique_outputs": 8,
        "h2d_ns": 1,
        "d2h_ns": 1,
        "encoder_gpu_ns": 1,
        "encoder_wall_time_ns": 2,
        "decoder_gpu_ns": 3,
        "decoder_wall_time_ns": 4,
        "cpu_sampling_ns": 1,
        "hashing_ns": 1,
        "inference_wall_time_ns": 20,
        "cuda_peak_allocated_bytes": 5,
        "cuda_peak_reserved_bytes": 6,
    }
    inference["inference_id"] = canonical_sha256(inference)

    total_preflight_wall = 200
    mean_update = 1
    mean_microbatch = 1
    handshake_wall = 2
    fresh_setup = 6
    checkpoint_roundtrip = 5
    selection_wall = 4
    primary = evidence.OPTIMIZER_UPDATES * mean_update
    validation_calls = evidence.VALIDATION_PASS_COUNT * validation_row_count
    validation = validation_calls * mean_microbatch
    checkpoint = evidence.VALIDATION_PASS_COUNT * checkpoint_roundtrip
    selection = selection_wall
    projected = (
        handshake_wall
        + 1
        + total_preflight_wall
        + fresh_setup
        + primary
        + validation
        + checkpoint
        + selection
        + evidence.PREFLIGHT_POST_VALIDATION_MARGIN_NS
    )
    projection: dict[str, object] = {
        "schema": evidence.PREFLIGHT_PROJECTION_SCHEMA_VERSION,
        "projection_method": evidence.PREFLIGHT_PROJECTION_METHOD,
        "measured_preflight_training_wall_ns": training_wall,
        "measured_preflight_total_wall_ns": total_preflight_wall,
        "mean_training_update_wall_ns": mean_update,
        "mean_training_microbatch_wall_ns": mean_microbatch,
        "measured_lock_setup_handshake_wall_ns": handshake_wall,
        "measured_handshake_to_preflight_start_ns": 1,
        "measured_fresh_reconstruction_wall_ns": fresh_setup,
        "measured_checkpoint_roundtrip_wall_ns": checkpoint_roundtrip,
        "measured_final_selection_wall_ns": selection_wall,
        "projected_primary_updates": evidence.OPTIMIZER_UPDATES,
        "projected_primary_wall_ns": primary,
        "projected_validation_passes": evidence.VALIDATION_PASS_COUNT,
        "validation_episodes_per_pass": validation_row_count,
        "projected_validation_episode_calls": validation_calls,
        "projected_validation_wall_ns": validation,
        "projected_checkpoint_writes": evidence.VALIDATION_PASS_COUNT,
        "projected_checkpoint_wall_ns": checkpoint,
        "projected_final_selection_wall_ns": selection,
        "post_preflight_validation_margin_ns": (
            evidence.PREFLIGHT_POST_VALIDATION_MARGIN_NS
        ),
        "projected_campaign_wall_ns": projected,
        "budget_limit_ns": GPU_BUDGET_NS,
        "budget_remaining_ns": GPU_BUDGET_NS - projected,
        "budget_status": "WITHIN_BUDGET",
    }
    projection["projection_id"] = canonical_sha256(projection)
    training = {
        "updates": 100,
        "microbatches_per_update": 16,
        "total_microbatches": 1_600,
        "masked_tokens_per_microbatch": 900,
        "masked_token_predictions": 1_440_000,
        "encoder_forward_calls": 1_600,
        "decoder_forward_calls": 1_600,
        "backward_calls": 1_600,
        "ledger_row_count": 100,
        "ledger_rows_sha256": ledger_sha,
        "total_update_wall_ns": training_wall,
        "total_update_cuda_event_ns": 0,
        "peak_allocated_bytes": 2,
        "peak_reserved_bytes": 3,
        "learning_rate_hex": [
            learning_rate_for_update(step).hex() for step in range(1, 101)
        ],
        "mean_masked_cell_ce_hex": [(1.0).hex()] * 100,
        "gradient_norm_before_clip_hex": [(0.5).hex()] * 100,
    }
    report: dict[str, object] = {
        "schema": PREFLIGHT_REPORT_SCHEMA_VERSION,
        "status": "PASS",
        "fixture": evidence._preflight_fixture_payload(),
        "runtime": {
            "run_id": "m04a-grid-cmlm-v0.1-seed20260711-r1",
            "gpu_uuid": "GPU-00000000-0000-0000-0000-000000000001",
            "logical_device_index": 0,
        },
        "model": {
            "model_semantics_version": "afts-grid-cmlm/v0.1",
            "parameter_count": 8_733_706,
            "training_config_sha256": (
                evidence._train_contract().training_config_sha256()
            ),
            "optimizer": "one_group_adamw_full_weight_decay",
            "precision": "bf16_autocast_forward_fp32_loss_gradients_state",
            "grad_scaler": False,
            "disposition": "discarded_without_checkpoint",
            "primary_requires_fresh_reconstruction": True,
        },
        "training": training,
        "inference": inference,
        "validation_manifest_commitment": _validation_commitment(
            validation_outer_sha256,
            validation_jsonl_sha256,
            validation_summary_id,
            validation_row_count,
        ),
        "fresh_reconstruction_probe": {
            "schema": evidence.PREFLIGHT_FRESH_RECONSTRUCTION_PROBE_SCHEMA_VERSION,
            "probe_id": "",
            "run_id": "m04a-grid-cmlm-v0.1-seed20260711-r1",
            "model_construct_wall_ns": 1,
            "cuda_transfer_wall_ns": 2,
            "optimizer_construct_wall_ns": 3,
            "total_wall_ns": fresh_setup,
            "state_discarded": True,
        },
        "checkpoint_cost_probe": {
            "schema": evidence.PREFLIGHT_CHECKPOINT_COST_PROBE_SCHEMA_VERSION,
            "probe_id": "",
            "probe_scope": "content_addressed_independent_probe",
            "run_id": "m04a-grid-cmlm-v0.1-seed20260711-r1",
            "checkpoint_sha256": _digest("checkpoint-cost-probe"),
            "checkpoint_bytes": 100,
            "write_wall_ns": 2,
            "weights_only_load_wall_ns": 3,
            "roundtrip_wall_ns": checkpoint_roundtrip,
            "weights_only_load": True,
            "artifact_disposition": "retained_nonselectable_diagnostic",
            "selectable_checkpoint_created": False,
        },
        "final_selection_probe": {
            "schema": evidence.PREFLIGHT_FINAL_SELECTION_PROBE_SCHEMA_VERSION,
            "probe_id": "",
            "run_id": "m04a-grid-cmlm-v0.1-seed20260711-r1",
            "candidate_count": evidence.VALIDATION_PASS_COUNT,
            "selection_rule": CHECKPOINT_SELECTION_RULE,
            "selected_checkpoint_index": evidence.SELECTION_TIE_INDICES[0],
            "selection_wall_ns": selection_wall,
        },
        "lock_handshake_probe": {
            "schema": evidence.PREFLIGHT_LOCK_HANDSHAKE_PROBE_SCHEMA_VERSION,
            "probe_id": "",
            "run_id": "m04a-grid-cmlm-v0.1-seed20260711-r1",
            "source": "verified_launcher_lock_handshake",
            "acquisition_started_perf_counter_ns": 1_000,
            "handshake_completed_perf_counter_ns": 1_002,
            "wall_ns": handshake_wall,
        },
        "preflight_started_perf_counter_ns": 1_003,
        "budget_projection": projection,
        "dataset_file_reads": 0,
        "training_checkpoint_writes": 0,
        "diagnostic_checkpoint_writes": 1,
        "fallback_used": False,
        "rng_state_restored": True,
        "ledger_rows_sha256": ledger_sha,
    }
    for field in (
        "fresh_reconstruction_probe",
        "checkpoint_cost_probe",
        "final_selection_probe",
        "lock_handshake_probe",
    ):
        _reseal(report[field], "probe_id")
    projection.update(
        {
            "lock_handshake_probe_id": report["lock_handshake_probe"]["probe_id"],
            "fresh_reconstruction_probe_id": report[
                "fresh_reconstruction_probe"
            ]["probe_id"],
            "checkpoint_cost_probe_id": report["checkpoint_cost_probe"]["probe_id"],
            "final_selection_probe_id": report["final_selection_probe"]["probe_id"],
        }
    )
    _reseal(projection, "projection_id")
    diagnostic_snapshot = b"D" * 100
    diagnostic_sha256 = hashlib.sha256(diagnostic_snapshot).hexdigest()
    report["checkpoint_cost_probe"]["checkpoint_sha256"] = diagnostic_sha256
    _reseal(report["checkpoint_cost_probe"], "probe_id")
    projection["checkpoint_cost_probe_id"] = report["checkpoint_cost_probe"][
        "probe_id"
    ]
    _reseal(projection, "projection_id")

    launch_plan_sha256 = _digest("launch-plan-file")
    launcher_sha256 = _digest("remote-launcher")
    remote_project_root = "/srv/afts"
    lock_semantic: dict[str, object] = {
        "schema": evidence.LOCK_HANDSHAKE_ARTIFACT_SCHEMA_VERSION,
        "run_id": report["runtime"]["run_id"],
        "gpu_uuid": report["runtime"]["gpu_uuid"],
        "lock_path": (
            f"{remote_project_root}/locks/gpu-{report['runtime']['gpu_uuid']}.lock"
        ),
        "lock_st_dev": 1,
        "lock_st_ino": 2,
        "lock_holder_pid": 3,
        "lock_holder_start_ticks": 4,
        "inherited_lock_fd": 5,
        "attempt_nonce": _digest("attempt-nonce"),
        "boot_id": "00000000-0000-0000-0000-000000000001",
        "acquisition_started_perf_counter_ns": 1_000,
        "eligibility_rechecked_perf_counter_ns": 1_001,
        "handshake_completed_perf_counter_ns": 1_002,
        "wall_ns": 2,
        "launcher_sha256": launcher_sha256,
        "launch_plan_sha256": launch_plan_sha256,
        "source": "reviewed_launcher_inherited_posix_flock",
    }
    lock_artifact = {
        **lock_semantic,
        "handshake_id": canonical_sha256(lock_semantic),
    }
    lock_bytes = serialize_json(lock_artifact)
    lock_sha256 = hashlib.sha256(lock_bytes).hexdigest()

    selection_rows = evidence._selection_input_payloads(diagnostic_sha256)
    selection_input_sha256 = canonical_sha256(selection_rows)
    selected_summary: dict[str, object] = {
        "schema": evidence.PREFLIGHT_SELECTED_METRIC_SUMMARY_SCHEMA_VERSION,
        "selection_input_sha256": selection_input_sha256,
        "final_selection_probe_id": report["final_selection_probe"]["probe_id"],
        **selection_rows[evidence.SELECTION_TIE_INDICES[0] - 1],
    }
    selected_summary["summary_id"] = canonical_sha256(selected_summary)
    overhead_projection: dict[str, object] = {
        "schema": evidence.CAMPAIGN_OVERHEAD_PROJECTION_SCHEMA_VERSION,
        "run_id": report["runtime"]["run_id"],
        "projection_method": evidence.OVERHEAD_PROJECTION_METHOD,
        "fixed_checkpoint_multiplier": evidence.VALIDATION_PASS_COUNT,
        "lock_handshake_probe_id": report["lock_handshake_probe"]["probe_id"],
        "measured_lock_setup_handshake_wall_ns": 2,
        "fresh_reconstruction_probe_id": report["fresh_reconstruction_probe"][
            "probe_id"
        ],
        "measured_fresh_reconstruction_wall_ns": 6,
        "checkpoint_cost_probe_id": report["checkpoint_cost_probe"]["probe_id"],
        "measured_checkpoint_write_wall_ns": 2,
        "measured_checkpoint_weights_only_load_wall_ns": 3,
        "measured_checkpoint_roundtrip_wall_ns": 5,
        "final_selection_probe_id": report["final_selection_probe"]["probe_id"],
        "measured_final_selection_wall_ns": 4,
        "projected_checkpoint_writes": evidence.VALIDATION_PASS_COUNT,
        "projected_checkpoint_loads": evidence.VALIDATION_PASS_COUNT,
        "projected_checkpoint_write_wall_ns": (
            evidence.VALIDATION_PASS_COUNT * 2
        ),
        "projected_checkpoint_weights_only_load_wall_ns": (
            evidence.VALIDATION_PASS_COUNT * 3
        ),
        "projected_checkpoint_roundtrip_wall_ns": (
            evidence.VALIDATION_PASS_COUNT * checkpoint_roundtrip
        ),
        "projected_fresh_setup_wall_ns": 6,
        "projected_final_selection_wall_ns": 4,
        "projected_lock_setup_handshake_wall_ns": 2,
        "projected_campaign_overhead_wall_ns": (
            2
            + 6
            + evidence.VALIDATION_PASS_COUNT * checkpoint_roundtrip
            + 4
        ),
    }
    overhead_projection["projection_id"] = canonical_sha256(overhead_projection)
    cost_report: dict[str, object] = {
        "schema": evidence.CAMPAIGN_OVERHEAD_PROBE_SCHEMA_VERSION,
        "status": "PASS",
        "run_id": report["runtime"]["run_id"],
        "runtime_mode": "frozen_cuda",
        "optimizer_step": 100,
        "source_optimizer_state": "fully_initialized_step_100",
        "config_sha256": evidence._train_contract().training_config_sha256(),
        "runtime_source_sha256": _digest("runtime-source-fingerprint"),
        "test_source_sha256": _digest("test-source-fingerprint"),
        "lock_handshake_probe": report["lock_handshake_probe"],
        "fresh_reconstruction_probe": report["fresh_reconstruction_probe"],
        "checkpoint_cost_probe": report["checkpoint_cost_probe"],
        "final_selection_probe": report["final_selection_probe"],
        "selection_input_sha256": selection_input_sha256,
        "selected_metric_summary": selected_summary,
        "overhead_projection": overhead_projection,
        "checkpoint_artifact_filename": (
            f"NOT_TRAINING_CHECKPOINT.diagnostic.{diagnostic_sha256}.pt"
        ),
        "checkpoint_artifact_status": "NOT_TRAINING_CHECKPOINT/diagnostic",
        "dataset_file_reads": 0,
        "fallback_used": False,
    }
    cost_report["probe_id"] = canonical_sha256(cost_report)
    report["overhead_cost_probe_id"] = cost_report["probe_id"]
    report["lock_handshake_artifact_sha256"] = lock_sha256
    report["report_id"] = canonical_sha256(report)
    parents: dict[str, object] = {
        "overhead_cost_probe_report": cost_report,
        "diagnostic_checkpoint_snapshot": diagnostic_snapshot,
        "lock_handshake_artifact": lock_artifact,
        "lock_handshake_artifact_bytes": lock_bytes,
        "expected_lock_handshake_artifact_sha256": lock_sha256,
        "expected_config_sha256": evidence._train_contract().training_config_sha256(),
        "expected_runtime_source_sha256": _digest("runtime-source-fingerprint"),
        "expected_test_source_sha256": _digest("test-source-fingerprint"),
        "expected_launcher_sha256": launcher_sha256,
        "expected_launch_plan_sha256": launch_plan_sha256,
        "expected_remote_project_root": remote_project_root,
        "expected_attempt_nonce": lock_artifact["attempt_nonce"],
    }
    return report, parents


def _training_closure() -> dict[str, object]:
    return {
        "status": "COMPLETE",
        "preflight_optimizer_updates": 100,
        "preflight_microbatches": 1_600,
        "preflight_encoder_forward_calls": 1_600,
        "preflight_decoder_forward_calls": 1_600,
        "preflight_backward_calls": 1_600,
        "preflight_inference_lanes": 8,
        "preflight_inference_steps": 12,
        "preflight_inference_encoder_batch_calls": 1,
        "preflight_inference_decoder_batch_calls": 12,
        "preflight_inference_sample_equivalent_forward_calls": 96,
        "primary_optimizer_updates": evidence.OPTIMIZER_UPDATES,
        "primary_microbatches": (
            evidence.OPTIMIZER_UPDATES * evidence.PREFLIGHT_MICROBATCHES_PER_UPDATE
        ),
        "primary_arc2_episodes": (
            evidence.OPTIMIZER_UPDATES
            * evidence.PREFLIGHT_MICROBATCHES_PER_UPDATE
            // 2
        ),
        "primary_rearc_episodes": (
            evidence.OPTIMIZER_UPDATES
            * evidence.PREFLIGHT_MICROBATCHES_PER_UPDATE
            // 2
        ),
        "primary_encoder_forward_calls": (
            evidence.OPTIMIZER_UPDATES * evidence.PREFLIGHT_MICROBATCHES_PER_UPDATE
        ),
        "primary_decoder_forward_calls": (
            evidence.OPTIMIZER_UPDATES * evidence.PREFLIGHT_MICROBATCHES_PER_UPDATE
        ),
        "primary_backward_calls": (
            evidence.OPTIMIZER_UPDATES * evidence.PREFLIGHT_MICROBATCHES_PER_UPDATE
        ),
        "validation_passes": evidence.VALIDATION_PASS_COUNT,
        "validation_episode_calls": evidence.VALIDATION_PASS_COUNT * 2,
        "validation_encoder_forward_calls": evidence.VALIDATION_PASS_COUNT * 2,
        "validation_decoder_forward_calls": evidence.VALIDATION_PASS_COUNT * 2,
        "checkpoint_writes": evidence.VALIDATION_PASS_COUNT,
        "resume_segments": 0,
        "masked_token_predictions": 1,
        "selected_checkpoint_complete": True,
        "budget_status": "WITHIN_BUDGET",
    }


class M04aPreflightArtifactClosureTests(unittest.TestCase):
    def test_exact_preflight_closes_external_validation_and_all_overheads(self) -> None:
        rows = _preflight_rows()
        outer = _digest("validation-outer")
        jsonl = _digest("validation-jsonl")
        summary_id = _digest("validation-summary")
        report, parents = _preflight_report(
            rows,
            validation_outer_sha256=outer,
            validation_jsonl_sha256=jsonl,
            validation_summary_id=summary_id,
            validation_row_count=2,
        )
        validated = validate_preflight_artifact(
            report,
            rows,
            validation_episodes_per_pass=2,
            expected_validation_outer_manifest_sha256=outer,
            expected_validation_jsonl_sha256=jsonl,
            expected_validation_summary_id=summary_id,
            **parents,
        )
        projection = validated["budget_projection"]
        self.assertEqual(
            projection["projected_checkpoint_writes"],
            evidence.VALIDATION_PASS_COUNT,
        )
        self.assertGreater(projection["measured_fresh_reconstruction_wall_ns"], 0)
        self.assertGreater(projection["projected_final_selection_wall_ns"], 0)

    def test_full_cost_tie_attack_and_lock_nonce_mismatch_are_rejected(self) -> None:
        rows = _preflight_rows()
        outer = _digest("validation-outer")
        jsonl = _digest("validation-jsonl")
        summary_id = _digest("validation-summary")
        report, parents = _preflight_report(
            rows,
            validation_outer_sha256=outer,
            validation_jsonl_sha256=jsonl,
            validation_summary_id=summary_id,
            validation_row_count=2,
        )
        forged_cost = copy.deepcopy(parents["overhead_cost_probe_report"])
        forged_cost["final_selection_probe"]["selected_checkpoint_index"] = (
            evidence.SELECTION_TIE_INDICES[1]
        )
        _reseal(forged_cost["final_selection_probe"], "probe_id")
        selection_rows = evidence._selection_input_payloads(
            forged_cost["checkpoint_cost_probe"]["checkpoint_sha256"]
        )
        forged_summary = {
            "schema": evidence.PREFLIGHT_SELECTED_METRIC_SUMMARY_SCHEMA_VERSION,
            "selection_input_sha256": forged_cost["selection_input_sha256"],
            "final_selection_probe_id": forged_cost["final_selection_probe"][
                "probe_id"
            ],
            **selection_rows[evidence.SELECTION_TIE_INDICES[1] - 1],
        }
        forged_summary["summary_id"] = canonical_sha256(forged_summary)
        forged_cost["selected_metric_summary"] = forged_summary
        forged_cost["overhead_projection"]["final_selection_probe_id"] = (
            forged_cost["final_selection_probe"]["probe_id"]
        )
        _reseal(forged_cost["overhead_projection"], "projection_id")
        _reseal(forged_cost, "probe_id")
        with self.assertRaisesRegex(ValueError, "argmin tie-break"):
            evidence.validate_campaign_overhead_cost_probe_report(
                forged_cost,
                expected_run_id=report["runtime"]["run_id"],
                expected_config_sha256=parents["expected_config_sha256"],
                expected_runtime_source_sha256=parents[
                    "expected_runtime_source_sha256"
                ],
                expected_test_source_sha256=parents[
                    "expected_test_source_sha256"
                ],
                diagnostic_checkpoint_snapshot=parents[
                    "diagnostic_checkpoint_snapshot"
                ],
            )

        wrong_nonce = dict(parents)
        wrong_nonce["expected_attempt_nonce"] = _digest("another-attempt")
        with self.assertRaisesRegex(ValueError, "launch commitments"):
            validate_preflight_artifact(
                report,
                rows,
                validation_episodes_per_pass=2,
                expected_validation_outer_manifest_sha256=outer,
                expected_validation_jsonl_sha256=jsonl,
                expected_validation_summary_id=summary_id,
                **wrong_nonce,
            )

    def test_preflight_rejects_ambient_validation_count_and_incomplete_lanes(self) -> None:
        rows = _preflight_rows()
        outer = _digest("validation-outer")
        jsonl = _digest("validation-jsonl")
        summary_id = _digest("validation-summary")
        report, parents = _preflight_report(
            rows,
            validation_outer_sha256=outer,
            validation_jsonl_sha256=jsonl,
            validation_summary_id=summary_id,
            validation_row_count=2,
        )
        with self.assertRaisesRegex(ValueError, "external evidence"):
            validate_preflight_artifact(
                report,
                rows,
                validation_episodes_per_pass=3,
                expected_validation_outer_manifest_sha256=outer,
                expected_validation_jsonl_sha256=jsonl,
                expected_validation_summary_id=summary_id,
                **parents,
            )

        no_checkpoint_probe = copy.deepcopy(report)
        no_checkpoint_probe.pop("checkpoint_cost_probe")
        _reseal(no_checkpoint_probe, "report_id")
        with self.assertRaisesRegex(ValueError, "fields must be"):
            validate_preflight_artifact(
                no_checkpoint_probe,
                rows,
                validation_episodes_per_pass=2,
                expected_validation_outer_manifest_sha256=outer,
                expected_validation_jsonl_sha256=jsonl,
                expected_validation_summary_id=summary_id,
                **parents,
            )

        tampered = copy.deepcopy(report)
        tampered["inference"]["lane_trace_sha256"].pop()
        _reseal(tampered["inference"], "inference_id")
        _reseal(tampered, "report_id")
        with self.assertRaisesRegex(ValueError, "eight distinct lane"):
            validate_preflight_artifact(
                tampered,
                rows,
                validation_episodes_per_pass=2,
                expected_validation_outer_manifest_sha256=outer,
                expected_validation_jsonl_sha256=jsonl,
                expected_validation_summary_id=summary_id,
                **parents,
            )

        short_rows = list(rows)
        short_rows.pop()
        with self.assertRaisesRegex(ValueError, "exactly 100"):
            validate_preflight_artifact(
                report,
                short_rows,
                validation_episodes_per_pass=2,
                expected_validation_outer_manifest_sha256=outer,
                expected_validation_jsonl_sha256=jsonl,
                expected_validation_summary_id=summary_id,
                **parents,
            )


class M04aSequentialFP32Tests(unittest.TestCase):
    def test_five_consecutive_one_ulp_values_have_pinned_sequential_mean(self) -> None:
        values = [
            struct.unpack("<f", struct.pack("<I", 0x3F800000 + offset))[0]
            for offset in range(5)
        ]
        result = sequential_fp32_mean(values)
        result_bits = struct.unpack("<I", struct.pack("<f", result))[0]
        self.assertEqual(result_bits, 0x3F800002)
        self.assertEqual(result.hex(), "0x1.0000040000000p+0")

    def test_validation_metric_rejects_resealed_non_fp32_episode_ce(self) -> None:
        episode = {
            "ledger_event_id": _digest("fp32-event"),
            "row_ordinal": 0,
            "episode_sha256": _digest("fp32-episode"),
            "semantic_parent_id": "parent",
            "target_group_id": "target",
            "masked_token_predictions": 1,
            "masked_cell_ce_hex": (1.0).hex(),
        }
        metric = make_validation_metric_row(
            validation_pass_index=1,
            validation_episode_outer_manifest_sha256=_digest("validation-outer"),
            validation_episode_manifest_sha256=_digest("validation-jsonl"),
            episode_metrics=(episode,),
            checkpoint_sha256=_digest("checkpoint"),
        )
        forged = copy.deepcopy(metric)
        forged["episode_metrics"][0]["masked_cell_ce_hex"] = (
            1.0 + 2.0**-24
        ).hex()
        _reseal(forged, "metric_id")
        with self.assertRaisesRegex(ValueError, "exact.*FP32"):
            validate_validation_metric_row(forged)


class M04aCheckpointSelectionClosureTests(unittest.TestCase):
    def _artifacts(self):
        validation_outer = _digest("validation-outer")
        validation_jsonl = _digest("validation-jsonl")
        ledger: list[dict[str, object]] = []
        checkpoints: list[dict[str, object]] = []
        metrics: list[dict[str, object]] = []
        validation_manifest_rows = [
            {
                "row_ordinal": ordinal,
                "episode_sha256": _digest(f"validation-episode-{ordinal}"),
                "semantic_parent_id": "validation-parent",
                "target_group_id": "validation-target",
                "masked_linear_indices": [ordinal],
            }
            for ordinal in range(2)
        ]
        event_index = 0
        for checkpoint_index in range(1, evidence.VALIDATION_PASS_COUNT + 1):
            validation_events = []
            for episode_index in range(2):
                event = make_training_cost_ledger_row(
                    phase="validation_episode",
                    event_index=event_index,
                    validation_pass_index=checkpoint_index,
                    episode_index=episode_index,
                    validation_episode_calls=1,
                    validation_encoder_forward_calls=1,
                    validation_decoder_forward_calls=1,
                    masked_token_predictions=1,
                    wall_time_ns=1,
                )
                ledger.append(event)
                validation_events.append(event["event_id"])
                event_index += 1
            checkpoint_sha = _digest(f"checkpoint-{checkpoint_index}")
            checkpoint_bytes = 10_000 + checkpoint_index
            checkpoint_event = make_training_cost_ledger_row(
                phase="checkpoint_operation",
                event_index=event_index,
                checkpoint_index=checkpoint_index,
                checkpoint_writes=1,
                checkpoint_io_bytes=checkpoint_bytes,
                checkpoint_io_ns=1,
                wall_time_ns=1,
            )
            ledger.append(checkpoint_event)
            event_index += 1
            ce = 0.25 if checkpoint_index in {2, 3} else 1.0 + checkpoint_index
            episode_metrics = [
                {
                    "ledger_event_id": validation_events[ordinal],
                    "row_ordinal": ordinal,
                    "episode_sha256": manifest_row["episode_sha256"],
                    "semantic_parent_id": manifest_row["semantic_parent_id"],
                    "target_group_id": manifest_row["target_group_id"],
                    "masked_token_predictions": len(
                        manifest_row["masked_linear_indices"]
                    ),
                    "masked_cell_ce_hex": float(ce).hex(),
                }
                for ordinal, manifest_row in enumerate(validation_manifest_rows)
            ]
            metric = make_validation_metric_row(
                validation_pass_index=checkpoint_index,
                validation_episode_outer_manifest_sha256=validation_outer,
                validation_episode_manifest_sha256=validation_jsonl,
                episode_metrics=episode_metrics,
                checkpoint_sha256=checkpoint_sha,
            )
            metrics.append(metric)
            checkpoints.append(
                make_checkpoint_manifest_row(
                    checkpoint_index=checkpoint_index,
                    checkpoint_sha256=checkpoint_sha,
                    checkpoint_bytes=checkpoint_bytes,
                    checkpoint_event_id=checkpoint_event["event_id"],
                    validation_metric_id=metric["metric_id"],
                )
            )
        checkpoint_jsonl_sha = hashlib.sha256(serialize_jsonl(checkpoints)).hexdigest()
        validation_jsonl_rows_sha = hashlib.sha256(serialize_jsonl(metrics)).hexdigest()
        training_ledger_sha = hashlib.sha256(serialize_jsonl(ledger)).hexdigest()
        lock = make_training_lock_interval(((100, 10_000),))
        selected = make_selected_checkpoint_manifest(
            checkpoint_rows=checkpoints,
            validation_rows=metrics,
            checkpoint_manifests_sha256=checkpoint_jsonl_sha,
            validation_metrics_sha256=validation_jsonl_rows_sha,
            training_cost_ledger_sha256=training_ledger_sha,
            validation_episode_outer_manifest_sha256=validation_outer,
            validation_episode_manifest_sha256=validation_jsonl,
            lock_interval_id=lock["interval_id"],
            selection_completed_perf_counter_ns=10_000,
        )
        return {
            "validation_outer": validation_outer,
            "validation_jsonl": validation_jsonl,
            "validation_manifest_rows": validation_manifest_rows,
            "ledger": ledger,
            "checkpoints": checkpoints,
            "metrics": metrics,
            "checkpoint_jsonl_sha": checkpoint_jsonl_sha,
            "validation_jsonl_rows_sha": validation_jsonl_rows_sha,
            "training_ledger_sha": training_ledger_sha,
            "lock": lock,
            "selected": selected,
        }

    def _validate(
        self,
        fixture,
        selected=None,
        checkpoints=None,
        metrics=None,
        ledger=None,
        training_ledger_sha=None,
        validation_manifest_rows=None,
    ):
        chosen = selected if selected is not None else fixture["selected"]
        checkpoint_rows = checkpoints if checkpoints is not None else fixture["checkpoints"]
        validation_rows = metrics if metrics is not None else fixture["metrics"]
        ledger_rows = ledger if ledger is not None else fixture["ledger"]
        ledger_sha = (
            training_ledger_sha
            if training_ledger_sha is not None
            else fixture["training_ledger_sha"]
        )
        manifest_rows = (
            validation_manifest_rows
            if validation_manifest_rows is not None
            else fixture["validation_manifest_rows"]
        )
        selected_index = chosen["selected_checkpoint_index"]
        selected_checkpoint = checkpoint_rows[selected_index - 1]
        return validate_training_checkpoint_artifacts(
            checkpoint_rows,
            validation_rows,
            chosen,
            training_ledger_rows=ledger_rows,
            checkpoint_manifests_sha256=fixture["checkpoint_jsonl_sha"],
            validation_metrics_sha256=fixture["validation_jsonl_rows_sha"],
            training_cost_ledger_sha256=ledger_sha,
            validation_episode_outer_manifest_sha256=fixture["validation_outer"],
            validation_episode_manifest_sha256=fixture["validation_jsonl"],
            validation_episode_rows=manifest_rows,
            selected_checkpoint_sha256=selected_checkpoint["checkpoint_sha256"],
            selected_checkpoint_bytes=selected_checkpoint["checkpoint_bytes"],
            lock_interval_id=fixture["lock"]["interval_id"],
            lock_selection_completed_perf_counter_ns=fixture["lock"]["segments"][-1][
                "end_perf_counter_ns"
            ],
        )

    def test_all_checkpoint_rows_select_minimum_parent_ce_and_break_tie_earlier(
        self,
    ) -> None:
        fixture = self._artifacts()
        selected = self._validate(fixture)
        self.assertEqual(selected["selection_rule"], CHECKPOINT_SELECTION_RULE)
        self.assertEqual(selected["selected_checkpoint_index"], 2)
        self.assertEqual(selected["optimizer_step"], 4_000)

    def test_rejects_later_tie_reorder_and_extra_schema_field(self) -> None:
        fixture = self._artifacts()
        later = copy.deepcopy(fixture["selected"])
        metric = fixture["metrics"][2]
        checkpoint = fixture["checkpoints"][2]
        later.update(
            {
                "selected_checkpoint_index": 3,
                "optimizer_step": 6_000,
                "parent_grouped_masked_cell_ce_hex": metric[
                    "parent_grouped_masked_cell_ce_hex"
                ],
                "validation_metric_id": metric["metric_id"],
                "checkpoint_manifest_id": checkpoint["manifest_id"],
                "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                "checkpoint_bytes": checkpoint["checkpoint_bytes"],
            }
        )
        _reseal(later, "selection_id")
        with self.assertRaisesRegex(ValueError, "minimum parent-grouped"):
            self._validate(fixture, selected=later)

        reordered = list(fixture["metrics"])
        reordered[0], reordered[1] = reordered[1], reordered[0]
        with self.assertRaisesRegex(
            ValueError,
            rf"ordered rows 1\.\.{evidence.VALIDATION_PASS_COUNT}",
        ):
            self._validate(fixture, metrics=reordered)

        forged_metrics = copy.deepcopy(fixture["metrics"])
        forged_checkpoints = copy.deepcopy(fixture["checkpoints"])
        forged_metrics[0]["episode_metrics"][0]["episode_sha256"] = _digest(
            "forged-validation-episode"
        )
        _reseal(forged_metrics[0], "metric_id")
        forged_checkpoints[0]["validation_metric_id"] = forged_metrics[0]["metric_id"]
        _reseal(forged_checkpoints[0], "manifest_id")
        with self.assertRaisesRegex(ValueError, "manifest row or ledger event"):
            self._validate(
                fixture,
                checkpoints=forged_checkpoints,
                metrics=forged_metrics,
            )

        extra = copy.deepcopy(fixture["checkpoints"][0])
        extra["unverified"] = True
        with self.assertRaisesRegex(ValueError, "fields must be"):
            validate_checkpoint_manifest_row(extra)

    def test_rejects_resealed_noninteger_validation_and_checkpoint_structure(self) -> None:
        fixture = self._artifacts()

        metric_mutations = (
            ("optimizer_step", 2_000.0),
            ("semantic_parent_count", 1.0),
            ("target_group_count", 1.0),
        )
        for field, value in metric_mutations:
            with self.subTest(metric_field=field):
                forged = copy.deepcopy(fixture["metrics"][0])
                forged[field] = value
                _reseal(forged, "metric_id")
                with self.assertRaisesRegex(TypeError, field):
                    validate_validation_metric_row(forged)

        forged = copy.deepcopy(fixture["metrics"][0])
        forged["episode_metrics"][0]["row_ordinal"] = False
        _reseal(forged, "metric_id")
        with self.assertRaisesRegex(TypeError, "row_ordinal"):
            validate_validation_metric_row(forged)

        checkpoint = copy.deepcopy(fixture["checkpoints"][0])
        checkpoint["optimizer_step"] = 2_000.0
        _reseal(checkpoint, "manifest_id")
        with self.assertRaisesRegex(TypeError, "optimizer_step"):
            validate_checkpoint_manifest_row(checkpoint)

        for field, value in (
            ("candidate_count", float(evidence.VALIDATION_PASS_COUNT)),
            ("optimizer_step", 4_000.0),
        ):
            with self.subTest(selected_field=field):
                selected = copy.deepcopy(fixture["selected"])
                selected[field] = value
                _reseal(selected, "selection_id")
                with self.assertRaisesRegex(TypeError, field):
                    validate_selected_checkpoint_manifest(selected)

        manifest_rows = copy.deepcopy(fixture["validation_manifest_rows"])
        manifest_rows[0]["row_ordinal"] = False
        with self.assertRaisesRegex(TypeError, "validation_episode_rows.*row_ordinal"):
            self._validate(fixture, validation_manifest_rows=manifest_rows)

        manifest_rows = copy.deepcopy(fixture["validation_manifest_rows"])
        masked_indices = manifest_rows[0]["masked_linear_indices"]
        masked_indices[0] = bool(masked_indices[0])
        with self.assertRaisesRegex(TypeError, "masked_linear_indices"):
            self._validate(fixture, validation_manifest_rows=manifest_rows)

    def test_rejects_committed_validation_event_outside_registered_passes(self) -> None:
        fixture = self._artifacts()
        ledger = list(fixture["ledger"])
        ledger.append(
            make_training_cost_ledger_row(
                phase="validation_episode",
                event_index=len(ledger),
                validation_pass_index=26,
                episode_index=0,
                validation_episode_calls=1,
                validation_encoder_forward_calls=1,
                validation_decoder_forward_calls=1,
                masked_token_predictions=1,
            )
        )
        ledger_sha = hashlib.sha256(serialize_jsonl(ledger)).hexdigest()
        selected = make_selected_checkpoint_manifest(
            checkpoint_rows=fixture["checkpoints"],
            validation_rows=fixture["metrics"],
            checkpoint_manifests_sha256=fixture["checkpoint_jsonl_sha"],
            validation_metrics_sha256=fixture["validation_jsonl_rows_sha"],
            training_cost_ledger_sha256=ledger_sha,
            validation_episode_outer_manifest_sha256=fixture["validation_outer"],
            validation_episode_manifest_sha256=fixture["validation_jsonl"],
            lock_interval_id=fixture["lock"]["interval_id"],
            selection_completed_perf_counter_ns=fixture["lock"]["segments"][-1][
                "end_perf_counter_ns"
            ],
        )
        with self.assertRaisesRegex(
            ValueError,
            rf"exactly {evidence.VALIDATION_PASS_COUNT} ordered passes",
        ):
            self._validate(
                fixture,
                selected=selected,
                ledger=ledger,
                training_ledger_sha=ledger_sha,
            )


class M04aLockEndpointClosureTests(unittest.TestCase):
    def test_summary_uses_handshake_to_final_selection_endpoint(self) -> None:
        interval = make_training_lock_interval(((10, 110),))
        timing = {
            "preflight_wall_time_ns": 10,
            "primary_training_wall_time_ns": 20,
            "validation_wall_time_ns": 10,
            "checkpoint_wall_time_ns": 10,
            "resume_setup_wall_time_ns": 0,
            "gpu_lock_wall_time_ns": 100,
            "budget_limit_ns": GPU_BUDGET_NS,
            "budget_remaining_ns": GPU_BUDGET_NS - 100,
        }
        summary = make_training_cost_summary(
            ledger_sha256=_digest("ledger"),
            closure=_training_closure(),
            timing=timing,
            lock_interval=interval,
        )
        self.assertEqual(summary["schema"], TRAINING_COST_SUMMARY_SCHEMA_VERSION)
        self.assertEqual(validate_training_cost_summary(summary), summary)

        incomplete = copy.deepcopy(interval)
        incomplete["segments"][0]["end_endpoint"] = "lock_release"
        _reseal(incomplete, "interval_id")
        with self.assertRaisesRegex(ValueError, "handshake-to-selection"):
            validate_training_lock_interval(incomplete)

    def test_resume_segments_and_fragment_sum_cannot_replace_endpoints(self) -> None:
        interval = make_training_lock_interval(((0, 10), (100, 120)))
        validate_training_lock_interval(interval, expected_resume_segments=1)
        with self.assertRaisesRegex(ValueError, "resume manifests"):
            validate_training_lock_interval(interval, expected_resume_segments=0)

        timing = {
            "preflight_wall_time_ns": 6,
            "primary_training_wall_time_ns": 6,
            "validation_wall_time_ns": 6,
            "checkpoint_wall_time_ns": 6,
            "resume_setup_wall_time_ns": 7,
            "gpu_lock_wall_time_ns": 30,
            "budget_limit_ns": GPU_BUDGET_NS,
            "budget_remaining_ns": GPU_BUDGET_NS - 30,
        }
        closure = _training_closure()
        closure["resume_segments"] = 1
        with self.assertRaisesRegex(ValueError, "phase rows exceed"):
            make_training_cost_summary(
                ledger_sha256=_digest("ledger"),
                closure=closure,
                timing=timing,
                lock_interval=interval,
            )


if __name__ == "__main__":
    unittest.main()
