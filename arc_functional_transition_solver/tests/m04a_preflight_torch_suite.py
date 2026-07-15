"""Explicit M04a preflight suite; excluded from default test discovery."""

from __future__ import annotations

import builtins
import hashlib
import inspect
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from afts_arc import m04a_preflight as preflight
from afts_arc import m04a_sample as production_sampler
from afts_arc.m04a_contract import (
    GRADIENT_ACCUMULATION,
    OPTIMIZER_UPDATES,
    VALIDATION_PASS_COUNT,
)
from afts_arc.m04a_cost_probe import (
    CHECKPOINT_COST_PROBE_SCHEMA_VERSION,
    CHECKPOINT_SELECTION_RULE,
    DIAGNOSTIC_DISPOSITION,
    FINAL_SELECTION_PROBE_SCHEMA_VERSION,
    FRESH_RECONSTRUCTION_PROBE_SCHEMA_VERSION,
    LOCK_HANDSHAKE_PROBE_SCHEMA_VERSION,
    SELECTION_TIE_INDICES,
)
from afts_arc.m04a_evidence import make_training_cost_ledger_row
from afts_arc.m04a_launch_plan import read_launch_plan_artifact
from afts_arc.m04a_lock_handshake import read_lock_handshake_artifact
from afts_arc.manifest import serialize_json
from afts_arc.m04a_preflight import (
    PREFLIGHT_BUDGET_EXCEEDED,
    PREFLIGHT_DEMONSTRATIONS,
    PREFLIGHT_GRID_SIDE,
    PREFLIGHT_LANES,
    PREFLIGHT_MEMORY_LENGTH,
    PREFLIGHT_SAMPLE_EQUIVALENT_CALLS,
    PreflightFailure,
    build_preflight_fixture,
    project_campaign_budget,
    run_exact_preflight,
    validate_preflight_fixture_payload,
    validate_preflight_ledger_rows,
)
from afts_arc.m04a_sample import SamplerRuntime
from afts_arc.m04a_torch_runtime import (
    CUBLAS_WORKSPACE_CONFIG,
    configure_deterministic_cuda,
)
from afts_arc.m04a_train_contract import (
    CAMPAIGN_GPU_BUDGET_NS,
    PREFLIGHT_UPDATES,
    training_config_sha256,
)
from afts_arc.m04a_validation_manifest import (
    VALIDATION_MANIFEST_COMMITMENT_SCHEMA_VERSION,
    build_validation_manifest_commitment,
    read_validation_episode_manifest,
)


class _FakeMemory:
    def __init__(self, memory_length: int) -> None:
        self.memory = torch.zeros((1, memory_length, 4), dtype=torch.float32)
        self.key_padding_mask = torch.zeros((1, memory_length), dtype=torch.bool)
        self.grid_lengths = (memory_length,)

    @property
    def memory_length(self) -> int:
        return int(self.memory.shape[1])


class _FakePreflightModel:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.training = True
        self.encoder_calls = 0
        self.decoder_calls = 0
        self.decoder_batch_sizes: list[int] = []

    def eval(self) -> "_FakePreflightModel":
        self.training = False
        return self

    def train(self, mode: bool = True) -> "_FakePreflightModel":
        self.training = mode
        return self

    def encode_task_memory(self, batch) -> _FakeMemory:
        self.encoder_calls += 1
        return _FakeMemory(sum(batch.lengths))

    def decode_target(self, batch, _cache: _FakeMemory):
        self.decoder_calls += 1
        self.decoder_batch_sizes.append(batch.batch_size)
        cell_count = batch.lengths[0]
        base = torch.arange(10, dtype=torch.float32).mul(0.03125)
        logits = base.view(1, 1, 10).expand(
            batch.batch_size, cell_count, 10
        ).contiguous()
        return SimpleNamespace(logits=logits)


def _zero_cost_preflight_rows() -> tuple[dict[str, object], ...]:
    return tuple(
        make_training_cost_ledger_row(
            phase="preflight_update",
            event_index=step - 1,
            optimizer_step=step,
            optimizer_updates=1,
            microbatches=GRADIENT_ACCUMULATION,
            encoder_forward_calls=GRADIENT_ACCUMULATION,
            decoder_forward_calls=GRADIENT_ACCUMULATION,
            backward_calls=GRADIENT_ACCUMULATION,
            masked_token_predictions=(
                GRADIENT_ACCUMULATION * PREFLIGHT_GRID_SIDE * PREFLIGHT_GRID_SIDE
            ),
        )
        for step in range(1, PREFLIGHT_UPDATES + 1)
    )


def _sealed_probe(payload: dict[str, object]) -> dict[str, object]:
    sealed = dict(payload)
    sealed["probe_id"] = preflight.canonical_sha256(sealed)
    return sealed


def _cost_probe_children(*, run_id: str = "test-run") -> dict[str, dict[str, object]]:
    return {
        "lock_handshake_probe": _sealed_probe(
            {
                "schema": LOCK_HANDSHAKE_PROBE_SCHEMA_VERSION,
                "run_id": run_id,
                "acquisition_started_perf_counter_ns": 10,
                "handshake_completed_perf_counter_ns": 11,
                "wall_ns": 1,
                "source": "verified_launcher_lock_handshake",
            }
        ),
        "fresh_reconstruction_probe": _sealed_probe(
            {
                "schema": FRESH_RECONSTRUCTION_PROBE_SCHEMA_VERSION,
                "run_id": run_id,
                "model_construct_wall_ns": 1,
                "cuda_transfer_wall_ns": 1,
                "optimizer_construct_wall_ns": 1,
                "total_wall_ns": 3,
                "state_discarded": True,
            }
        ),
        "checkpoint_cost_probe": _sealed_probe(
            {
                "schema": CHECKPOINT_COST_PROBE_SCHEMA_VERSION,
                "probe_scope": "content_addressed_independent_probe",
                "run_id": run_id,
                "checkpoint_sha256": "a" * 64,
                "checkpoint_bytes": 1,
                "write_wall_ns": 1,
                "weights_only_load_wall_ns": 1,
                "roundtrip_wall_ns": 2,
                "weights_only_load": True,
                "artifact_disposition": DIAGNOSTIC_DISPOSITION,
                "selectable_checkpoint_created": False,
            }
        ),
        "final_selection_probe": _sealed_probe(
            {
                "schema": FINAL_SELECTION_PROBE_SCHEMA_VERSION,
                "run_id": run_id,
                "candidate_count": VALIDATION_PASS_COUNT,
                "selection_rule": CHECKPOINT_SELECTION_RULE,
                "selected_checkpoint_index": SELECTION_TIE_INDICES[0],
                "selection_wall_ns": 1,
            }
        ),
    }


def _validation_commitment_payload(*, row_count: int = 1) -> dict[str, object]:
    semantic: dict[str, object] = {
        "schema": VALIDATION_MANIFEST_COMMITMENT_SCHEMA_VERSION,
        "outer_artifact_manifest_sha256": "b" * 64,
        "jsonl_sha256": "c" * 64,
        "summary_id": "d" * 64,
        "row_count": row_count,
    }
    return {
        **semantic,
        "commitment_id": hashlib.sha256(serialize_json(semantic)).hexdigest(),
    }


class M04aPreflightCpuContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(max(1, min(torch.get_num_threads(), 4)))

    def test_public_preflight_releases_owned_lock_on_failure(self) -> None:
        held = object()
        artifact = object()
        with (
            mock.patch.object(
                preflight,
                "_validate_preflight_launch_inputs",
                return_value="unused",
            ),
            mock.patch.object(
                preflight, "attest_inherited_gpu_lock", return_value=held
            ),
            mock.patch.object(
                preflight,
                "_run_exact_preflight_with_held_lock",
                side_effect=RuntimeError("terminal preflight failure"),
            ),
            mock.patch.object(preflight, "release_held_gpu_lock") as released,
        ):
            with self.assertRaisesRegex(RuntimeError, "terminal preflight"):
                run_exact_preflight(
                    runtime_attestation=object(),
                    validation_manifest_commitment={},
                    run_root="unused",
                    cost_probe_scratch_dir="unused",
                    config_sha256="a" * 64,
                    runtime_source_sha256="b" * 64,
                    test_source_sha256="c" * 64,
                    lock_handshake_artifact=artifact,
                    launch_plan_artifact=artifact,
                )
        released.assert_called_once_with(held)

    def test_launch_plan_mismatch_releases_lock_before_preflight_body(self) -> None:
        held = object()
        artifact = object()
        with (
            mock.patch.object(
                preflight, "attest_inherited_gpu_lock", return_value=held
            ),
            mock.patch.object(
                preflight,
                "_validate_preflight_launch_inputs",
                side_effect=ValueError("launch-plan mismatch"),
            ),
            mock.patch.object(
                preflight, "_run_exact_preflight_with_held_lock"
            ) as body,
            mock.patch.object(preflight, "release_held_gpu_lock") as released,
        ):
            with self.assertRaisesRegex(ValueError, "launch-plan mismatch"):
                run_exact_preflight(
                    runtime_attestation=object(),
                    validation_manifest_commitment={},
                    run_root="unused",
                    cost_probe_scratch_dir="unused",
                    config_sha256="a" * 64,
                    runtime_source_sha256="b" * 64,
                    test_source_sha256="c" * 64,
                    lock_handshake_artifact=artifact,
                    launch_plan_artifact=artifact,
                )
        released.assert_called_once_with(held)
        body.assert_not_called()

    def test_fixture_is_exact_constant_grid_and_opens_no_dataset(self) -> None:
        with mock.patch.object(
            builtins,
            "open",
            side_effect=AssertionError("preflight attempted filesystem access"),
        ):
            fixture = build_preflight_fixture()
            replay = build_preflight_fixture()
        self.assertEqual(fixture, replay)
        self.assertEqual(len(fixture.demonstrations), PREFLIGHT_DEMONSTRATIONS)
        self.assertEqual(
            {cell for row in fixture.demonstrations[7][0] for cell in row}, {7}
        )
        self.assertEqual(
            {cell for row in fixture.demonstrations[7][1] for cell in row}, {8}
        )
        self.assertEqual({cell for row in fixture.query_input for cell in row}, {0})
        self.assertEqual({cell for row in fixture.target_output for cell in row}, {1})
        self.assertEqual(len(fixture.query_input), 30)
        self.assertEqual(len(fixture.query_input[0]), 30)
        self.assertEqual(
            validate_preflight_fixture_payload(fixture.payload), fixture.payload
        )
        self.assertEqual(
            set(inspect.signature(run_exact_preflight).parameters),
            {
                "runtime_attestation",
                "validation_manifest_commitment",
                "run_root",
                "cost_probe_scratch_dir",
                "config_sha256",
                "runtime_source_sha256",
                "test_source_sha256",
                "lock_handshake_artifact",
                "launch_plan_artifact",
            },
        )
        self.assertEqual(
            set(inspect.signature(preflight.validate_preflight_report).parameters),
            {
                "report",
                "ledger_rows",
                "validation_manifest_commitment",
                "overhead_cost_probe_report",
                "expected_config_sha256",
                "expected_runtime_source_sha256",
                "expected_test_source_sha256",
                "lock_handshake_artifact",
                "launch_plan_artifact",
                "expected_run_root",
            },
        )

    def test_preflight_resets_frozen_rng_after_capture_before_model_construction(self) -> None:
        source = inspect.getsource(preflight._run_exact_preflight_with_held_lock)
        capture_at = source.index("rng_before = capture_rng_state")
        protected_try_at = source.index("try:", source.index("run_started ="))
        cpu_seed_at = source.index("torch.manual_seed(TRAINING_SEED)")
        cuda_seed_at = source.index("torch.cuda.manual_seed_all(TRAINING_SEED)")
        construct_at = source.index("model = GridCMLM()")
        self.assertLess(capture_at, protected_try_at)
        self.assertLess(protected_try_at, cpu_seed_at)
        self.assertLess(cpu_seed_at, cuda_seed_at)
        self.assertLess(cuda_seed_at, construct_at)

    def test_exact_preflight_owns_commit_after_complete_outcome_validation(self) -> None:
        outer = inspect.getsource(preflight._run_exact_preflight_with_held_lock)
        restore_at = outer.index("_restore_rng_and_cleanup")
        finish_at = outer.index("_finish_preflight_with_failure_evidence")
        self.assertLess(restore_at, finish_at)

        finish = inspect.getsource(preflight._finish_preflight_and_commit_diagnostic)
        failure_at = finish.index("failure = PreflightFailure")
        first_lock_at = finish.index("assert_held_gpu_lock", failure_at)
        first_commit_at = finish.index("overhead_preparation.commit()", first_lock_at)
        raise_at = finish.index("raise failure", first_commit_at)
        result_at = finish.index("result = PreflightResult")
        final_lock_at = finish.index("assert_held_gpu_lock", result_at)
        final_commit_at = finish.index("overhead_preparation.commit()", final_lock_at)
        return_at = finish.index("return result", final_commit_at)
        self.assertLess(failure_at, first_lock_at)
        self.assertLess(first_lock_at, first_commit_at)
        self.assertLess(first_commit_at, raise_at)
        self.assertLess(result_at, final_lock_at)
        self.assertLess(final_lock_at, final_commit_at)
        self.assertLess(final_commit_at, return_at)

    def test_finish_keeps_reader_commitment_separate_from_json_payload(self) -> None:
        reader_commitment = object()
        commitment_payload = _validation_commitment_payload(row_count=7)
        overhead_report = {
            "lock_handshake_probe": {"probe": "lock"},
            "fresh_reconstruction_probe": {"probe": "reconstruction"},
            "checkpoint_cost_probe": {"probe": "checkpoint"},
            "final_selection_probe": {"probe": "selection"},
        }
        overhead_result = SimpleNamespace(
            report={"unvalidated": True},
            checkpoint_path=Path("NOT_TRAINING_CHECKPOINT.diagnostic.test.pt"),
            checkpoint_snapshot=b"diagnostic checkpoint",
        )
        preparation = SimpleNamespace(
            result=overhead_result,
            commit=mock.Mock(),
        )
        attestation = SimpleNamespace(run_id="reader-provenance-run")
        held_lock = SimpleNamespace(
            artifact=SimpleNamespace(artifact_sha256="a" * 64)
        )
        expected_result = object()
        report = {"status": "PASS"}

        with (
            mock.patch.object(
                preflight,
                "validate_validation_manifest_commitment",
                return_value=commitment_payload,
            ) as validate_commitment,
            mock.patch.object(
                preflight,
                "validate_campaign_overhead_cost_probe_report",
                return_value=overhead_report,
            ),
            mock.patch.object(
                preflight,
                "project_campaign_budget",
                return_value={"budget_status": "WITHIN_BUDGET"},
            ) as project_budget,
            mock.patch.object(
                preflight, "_make_report", return_value=report
            ) as make_report,
            mock.patch.object(
                preflight, "PreflightResult", return_value=expected_result
            ) as result_constructor,
            mock.patch.object(preflight, "assert_held_gpu_lock") as check_lock,
        ):
            result = preflight._finish_preflight_and_commit_diagnostic(
                fixture=object(),
                attestation=attestation,
                validation_manifest_commitment=reader_commitment,
                validation_episodes=7,
                run_root="/reviewed/run-root",
                config_sha256="b" * 64,
                runtime_source_sha256="c" * 64,
                test_source_sha256="d" * 64,
                launch_plan_artifact=object(),
                held_lock=held_lock,
                preflight_started=1,
                run_started=1,
                ledger_rows=(),
                training_summary={"total_update_wall_ns": 1},
                inference_summary={},
                overhead_preparation=preparation,
            )

        self.assertIs(result, expected_result)
        validate_commitment.assert_called_once_with(reader_commitment)
        self.assertIs(
            make_report.call_args.kwargs["validation_manifest_commitment"],
            commitment_payload,
        )
        self.assertIs(
            result_constructor.call_args.kwargs["validation_manifest_commitment"],
            reader_commitment,
        )
        self.assertEqual(
            project_budget.call_args.kwargs["validation_episodes_per_pass"], 7
        )
        check_lock.assert_called_once_with(held_lock)
        preparation.commit.assert_called_once_with()

    def test_held_lock_runner_never_shadows_reader_commitment_with_payload(self) -> None:
        source = inspect.getsource(preflight._run_exact_preflight_with_held_lock)
        self.assertIn(
            "validation_commitment_payload = validate_validation_manifest_commitment",
            source,
        )
        self.assertIn(
            "validation_manifest_commitment=validation_manifest_commitment",
            source,
        )
        self.assertNotIn(
            "validation_commitment = validate_validation_manifest_commitment", source
        )

    def test_budget_finish_embeds_payload_without_constructing_pass_result(self) -> None:
        reader_commitment = object()
        commitment_payload = _validation_commitment_payload(row_count=5)
        overhead_report = {
            "lock_handshake_probe": {"probe": "lock"},
            "fresh_reconstruction_probe": {"probe": "reconstruction"},
            "checkpoint_cost_probe": {"probe": "checkpoint"},
            "final_selection_probe": {"probe": "selection"},
        }
        overhead_result = SimpleNamespace(
            report={"unvalidated": True},
            checkpoint_path=Path("NOT_TRAINING_CHECKPOINT.diagnostic.test.pt"),
            checkpoint_snapshot=b"diagnostic checkpoint",
        )
        preparation = SimpleNamespace(
            result=overhead_result,
            commit=mock.Mock(),
        )
        attestation = SimpleNamespace(run_id="budget-reader-provenance-run")
        held_lock = SimpleNamespace(
            artifact=SimpleNamespace(artifact_sha256="a" * 64)
        )
        failure_report = {
            "failure_code": PREFLIGHT_BUDGET_EXCEEDED,
            "diagnostic_checkpoint_commitment": None,
        }

        with (
            mock.patch.object(
                preflight,
                "validate_validation_manifest_commitment",
                return_value=commitment_payload,
            ),
            mock.patch.object(
                preflight,
                "validate_campaign_overhead_cost_probe_report",
                return_value=overhead_report,
            ),
            mock.patch.object(
                preflight,
                "project_campaign_budget",
                return_value={"budget_status": PREFLIGHT_BUDGET_EXCEEDED},
            ),
            mock.patch.object(
                preflight, "_make_failure_report", return_value=failure_report
            ) as make_failure_report,
            mock.patch.object(
                preflight, "validate_preflight_failure_report"
            ) as validate_failure_report,
            mock.patch.object(preflight, "PreflightResult") as result_constructor,
            mock.patch.object(preflight, "assert_held_gpu_lock") as check_lock,
        ):
            with self.assertRaises(PreflightFailure) as caught:
                preflight._finish_preflight_and_commit_diagnostic(
                    fixture=SimpleNamespace(fixture_id="fixture"),
                    attestation=attestation,
                    validation_manifest_commitment=reader_commitment,
                    validation_episodes=5,
                    run_root="/reviewed/run-root",
                    config_sha256="b" * 64,
                    runtime_source_sha256="c" * 64,
                    test_source_sha256="d" * 64,
                    launch_plan_artifact=object(),
                    held_lock=held_lock,
                    preflight_started=1,
                    run_started=1,
                    ledger_rows=(),
                    training_summary={"total_update_wall_ns": 1},
                    inference_summary={},
                    overhead_preparation=preparation,
                )

        self.assertEqual(caught.exception.failure_code, PREFLIGHT_BUDGET_EXCEEDED)
        self.assertIs(
            make_failure_report.call_args.kwargs[
                "validation_manifest_commitment"
            ],
            commitment_payload,
        )
        validate_failure_report.assert_called_once()
        result_constructor.assert_not_called()
        check_lock.assert_called_once_with(held_lock)
        preparation.commit.assert_called_once_with()

    def test_finish_validator_and_rename_failures_publish_strict_partial_evidence(
        self,
    ) -> None:
        for error in (
            ValueError("injected finish validator failure"),
            FileExistsError("injected no-replace rename failure"),
        ):
            with self.subTest(error=type(error).__name__):
                preparation = SimpleNamespace(
                    committed=False,
                    abort=mock.Mock(),
                )
                with mock.patch.object(
                    preflight,
                    "_finish_preflight_and_commit_diagnostic",
                    side_effect=error,
                ):
                    with self.assertRaises(PreflightFailure) as caught:
                        preflight._finish_preflight_with_failure_evidence(
                            fixture=build_preflight_fixture(),
                            attestation=None,
                            validation_manifest_commitment=None,
                            validation_episodes=0,
                            run_root="unused",
                            config_sha256="unused",
                            runtime_source_sha256="unused",
                            test_source_sha256="unused",
                            launch_plan_artifact=None,
                            held_lock=None,
                            preflight_started=0,
                            run_started=0,
                            ledger_rows=(),
                            training_summary={},
                            inference_summary={},
                            overhead_preparation=preparation,
                        )
                failure = caught.exception
                self.assertEqual(
                    failure.failure_code,
                    preflight.PREFLIGHT_PROJECTION_INCOMPLETE,
                )
                self.assertEqual(failure.report["diagnostic_checkpoint_writes"], 0)
                self.assertIsNone(failure.report["diagnostic_checkpoint_commitment"])
                self.assertEqual(failure.ledger_rows, ())
                preflight.validate_preflight_failure_report(failure.report, ())
                preparation.abort.assert_called_once_with()

    def test_rng_restoration_precedes_cleanup_and_preserves_both_failures(self) -> None:
        order: list[str] = []
        active = RuntimeError("primary failure")

        def fail_restore(_state) -> None:
            order.append("restore")
            raise RuntimeError("restore failed")

        def fail_cleanup() -> None:
            order.append("cleanup")
            raise RuntimeError("cleanup failed")

        with (
            mock.patch.object(preflight, "restore_rng_state", side_effect=fail_restore),
            mock.patch.object(torch.cuda, "empty_cache", side_effect=fail_cleanup),
        ):
            preflight._restore_rng_and_cleanup({}, active_exception=active)
        self.assertEqual(order, ["restore", "cleanup"])
        self.assertIn("restore failed", active.rng_restoration_error)
        self.assertIn("cleanup failed", active.cuda_cleanup_error)

        order.clear()
        cleanup = RuntimeError("cleanup after successful restore")

        def fail_cleanup_after_restore() -> None:
            order.append("cleanup")
            raise cleanup

        with (
            mock.patch.object(
                preflight,
                "restore_rng_state",
                side_effect=lambda _state: order.append("restore"),
            ),
            mock.patch.object(preflight, "capture_rng_state", return_value={}),
            mock.patch.object(preflight, "_rng_states_equal", return_value=True),
            mock.patch.object(
                torch.cuda,
                "empty_cache",
                side_effect=fail_cleanup_after_restore,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "cleanup after successful"):
                preflight._restore_rng_and_cleanup({}, active_exception=None)
        self.assertEqual(order, ["restore", "cleanup"])

    def test_budget_projection_is_integer_ceil_and_fails_over_24_hours(self) -> None:
        probes = _cost_probe_children()
        within = project_campaign_budget(
            measured_preflight_training_wall_ns=100,
            measured_preflight_total_wall_ns=200,
            preflight_started_perf_counter_ns=12,
            validation_episodes_per_pass=3,
            **probes,
        )
        self.assertEqual(within["mean_training_update_wall_ns"], 1)
        self.assertEqual(within["mean_training_microbatch_wall_ns"], 1)
        self.assertEqual(within["projected_primary_wall_ns"], OPTIMIZER_UPDATES)
        self.assertEqual(
            within["projected_validation_episode_calls"],
            VALIDATION_PASS_COUNT * 3,
        )
        self.assertEqual(
            within["projected_checkpoint_wall_ns"], VALIDATION_PASS_COUNT * 2
        )
        self.assertEqual(within["measured_handshake_to_preflight_start_ns"], 1)
        self.assertEqual(
            within["post_preflight_validation_margin_ns"],
            preflight.PREFLIGHT_POST_VALIDATION_MARGIN_NS,
        )
        self.assertEqual(
            within["projected_campaign_wall_ns"],
            (
                1
                + 1
                + 200
                + 3
                + OPTIMIZER_UPDATES
                + VALIDATION_PASS_COUNT * 3
                + VALIDATION_PASS_COUNT * 2
                + 1
                + preflight.PREFLIGHT_POST_VALIDATION_MARGIN_NS
            ),
        )
        self.assertEqual(within["budget_status"], "WITHIN_BUDGET")

        exceeded = project_campaign_budget(
            measured_preflight_training_wall_ns=PREFLIGHT_UPDATES
            * 5_000_000_000,
            measured_preflight_total_wall_ns=PREFLIGHT_UPDATES
            * 5_000_000_000,
            preflight_started_perf_counter_ns=12,
            validation_episodes_per_pass=1,
            **probes,
        )
        self.assertGreater(
            exceeded["projected_campaign_wall_ns"], CAMPAIGN_GPU_BUDGET_NS
        )
        self.assertEqual(exceeded["budget_status"], PREFLIGHT_BUDGET_EXCEEDED)

    def test_real_manifest_projection_accepts_exact_cap_and_rejects_plus_one(self) -> None:
        probes = _cost_probe_children()
        arguments = {
            "measured_preflight_training_wall_ns": 101,
            "measured_preflight_total_wall_ns": 203,
            "preflight_started_perf_counter_ns": 12,
            "validation_episodes_per_pass": 2_656,
            **probes,
        }
        baseline = project_campaign_budget(**arguments)
        exact_cap = baseline["projected_campaign_wall_ns"]
        with mock.patch.object(preflight, "CAMPAIGN_GPU_BUDGET_NS", exact_cap):
            at_cap = project_campaign_budget(**arguments)
        with mock.patch.object(preflight, "CAMPAIGN_GPU_BUDGET_NS", exact_cap - 1):
            plus_one = project_campaign_budget(**arguments)
        self.assertEqual(at_cap["projected_campaign_wall_ns"], exact_cap)
        self.assertEqual(at_cap["budget_status"], "WITHIN_BUDGET")
        self.assertEqual(
            plus_one["projected_campaign_wall_ns"],
            plus_one["budget_limit_ns"] + 1,
        )
        self.assertEqual(plus_one["budget_status"], PREFLIGHT_BUDGET_EXCEEDED)

    def test_maximum_context_episodes_use_production_audit_contract(self) -> None:
        fixture = build_preflight_fixture()
        episodes, audits = preflight._build_synthetic_training_episodes(
            fixture,
            optimizer_step=1,
        )
        self.assertEqual(len(episodes), GRADIENT_ACCUMULATION)
        self.assertEqual(len(audits), GRADIENT_ACCUMULATION)
        self.assertEqual(
            [episode.source for episode in episodes],
            ["arc2" if slot % 2 == 0 else "rearc" for slot in range(16)],
        )
        self.assertTrue(
            all(len(episode.demonstrations) == PREFLIGHT_DEMONSTRATIONS for episode in episodes)
        )
        self.assertTrue(
            all(
                len(example.input_grid) == PREFLIGHT_GRID_SIDE
                and len(example.input_grid[0]) == PREFLIGHT_GRID_SIDE
                and len(example.output_grid) == PREFLIGHT_GRID_SIDE
                and len(example.output_grid[0]) == PREFLIGHT_GRID_SIDE
                for episode in episodes
                for example in episode.demonstrations
            )
        )
        self.assertEqual(
            [episode.seed_u64 for episode in episodes],
            [preflight.episode_seed(0, slot) for slot in range(16)],
        )
        self.assertEqual(
            sum(row["masked_token_predictions"] for row in audits),
            GRADIENT_ACCUMULATION * PREFLIGHT_GRID_SIDE * PREFLIGHT_GRID_SIDE,
        )

    def test_preparation_and_audit_execute_inside_update_wall_timer(self) -> None:
        order: list[str] = []
        clock_values = iter((100, 250))

        def wall_clock() -> int:
            order.append("wall_clock")
            return next(clock_values)

        audits = tuple(
            {"masked_token_predictions": PREFLIGHT_GRID_SIDE * PREFLIGHT_GRID_SIDE}
            for _ in range(GRADIENT_ACCUMULATION)
        )

        def build(_fixture, *, optimizer_step):
            self.assertEqual(optimizer_step, 1)
            order.append("build_and_audit")
            return tuple(object() for _ in range(GRADIENT_ACCUMULATION)), audits

        def execute(_model, _optimizer, episodes, _learning_rate):
            self.assertEqual(len(episodes), GRADIENT_ACCUMULATION)
            order.append("production_cuda_update")
            return torch.tensor(0.0), torch.tensor(0.0)

        def timed(function):
            order.append("cuda_timer")
            return function(), SimpleNamespace(
                gpu_ns=0,
                wall_ns=0,
                peak_allocated_bytes=0,
                peak_reserved_bytes=0,
            )

        with (
            mock.patch.object(preflight, "PREFLIGHT_UPDATES", 1),
            mock.patch.object(
                preflight,
                "_build_synthetic_training_episodes",
                side_effect=build,
            ),
            mock.patch.object(preflight, "_execute_cuda_update", side_effect=execute),
            mock.patch.object(preflight, "timed_cuda_call", side_effect=timed),
            mock.patch.object(preflight, "_assert_no_gradients"),
            mock.patch.object(preflight, "assert_adamw_invariants"),
            mock.patch.object(preflight.time, "perf_counter_ns", side_effect=wall_clock),
            mock.patch.object(preflight.time, "process_time_ns", side_effect=(10, 20)),
        ):
            rows, summary = preflight._run_training_fixture(
                object(),
                object(),
                build_preflight_fixture(),
            )
        self.assertEqual(
            order,
            [
                "wall_clock",
                "build_and_audit",
                "cuda_timer",
                "production_cuda_update",
                "wall_clock",
            ],
        )
        self.assertEqual(rows[0]["wall_time_ns"], 150)
        self.assertEqual(summary["total_update_wall_ns"], 150)

    def test_preflight_ledger_closes_exact_100_by_16_coordinates(self) -> None:
        rows = _zero_cost_preflight_rows()
        normalized = validate_preflight_ledger_rows(rows)
        self.assertEqual(len(normalized), 100)
        self.assertEqual(
            [row["event_index"] for row in normalized], list(range(100))
        )
        self.assertEqual(
            [row["optimizer_step"] for row in normalized], list(range(1, 101))
        )
        tampered = list(rows)
        tampered[-1] = make_training_cost_ledger_row(
            phase="preflight_update",
            event_index=99,
            optimizer_step=100,
            optimizer_updates=1,
            microbatches=GRADIENT_ACCUMULATION,
            encoder_forward_calls=GRADIENT_ACCUMULATION,
            decoder_forward_calls=15,
            backward_calls=GRADIENT_ACCUMULATION,
            masked_token_predictions=(
                GRADIENT_ACCUMULATION * PREFLIGHT_GRID_SIDE * PREFLIGHT_GRID_SIDE
            ),
        )
        with self.assertRaisesRegex(ValueError, "semantic counts"):
            validate_preflight_ledger_rows(tampered)

    def test_eight_lane_inference_reuses_production_sampler_numerics(self) -> None:
        fixture = build_preflight_fixture()
        model = _FakePreflightModel()
        production_step = production_sampler._sample_lane_step
        with mock.patch.object(
            production_sampler,
            "_sample_lane_step",
            wraps=production_step,
        ) as sampled:
            summary = preflight._run_cached_inference_fixture(
                model, fixture, SamplerRuntime.cpu_test()
            )
        self.assertTrue(model.training)
        self.assertEqual(model.encoder_calls, 1)
        self.assertEqual(model.decoder_calls, 12)
        self.assertEqual(model.decoder_batch_sizes, [PREFLIGHT_LANES] * 12)
        self.assertEqual(sampled.call_count, PREFLIGHT_SAMPLE_EQUIVALENT_CALLS)
        self.assertEqual(summary["encoder_batch_calls"], 1)
        self.assertEqual(summary["decoder_batch_calls"], 12)
        self.assertEqual(summary["sample_equivalent_forward_calls"], 96)
        self.assertEqual(summary["unpadded_memory_length"], PREFLIGHT_MEMORY_LENGTH)
        self.assertEqual(summary["cache_dtype"], "float32")
        self.assertEqual(summary["mask_count_trace"][0], 900)
        self.assertEqual(summary["mask_count_trace"][-1], 0)
        self.assertEqual(len(summary["lane_trace_sha256"]), 8)
        self.assertEqual(len(summary["final_output_keys"]), 8)
        preflight._validate_inference_summary(summary, require_cuda=False)

    def test_pass_report_closes_fixture_ledger_inference_and_projection(self) -> None:
        fixture = build_preflight_fixture()
        probes = _cost_probe_children()
        validation_commitment = _validation_commitment_payload()
        rows = [dict(row) for row in _zero_cost_preflight_rows()]
        for row in rows:
            row["wall_time_ns"] = 1
        normalized = validate_preflight_ledger_rows(rows)
        inference = preflight._run_cached_inference_fixture(
            _FakePreflightModel(), fixture, SamplerRuntime.cpu_test()
        )
        inference["cache_dtype"] = "bfloat16"
        inference.pop("inference_id")
        inference["inference_id"] = preflight.canonical_sha256(inference)
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
            "ledger_rows_sha256": preflight.canonical_sha256(list(normalized)),
            "total_update_wall_ns": 100,
            "total_update_cuda_event_ns": 0,
            "peak_allocated_bytes": 0,
            "peak_reserved_bytes": 0,
            "learning_rate_hex": [
                preflight.learning_rate_for_update(step).hex()
                for step in range(1, 101)
            ],
            "mean_masked_cell_ce_hex": [float(0.0).hex()] * 100,
            "gradient_norm_before_clip_hex": [float(0.0).hex()] * 100,
        }
        projection = project_campaign_budget(
            measured_preflight_training_wall_ns=100,
            measured_preflight_total_wall_ns=max(
                200, inference["inference_wall_time_ns"] + 100
            ),
            preflight_started_perf_counter_ns=12,
            validation_episodes_per_pass=1,
            **probes,
        )
        attestation = SimpleNamespace(
            run_id="test-run",
            gpu_uuid="GPU-00000000-0000-0000-0000-000000000001",
            logical_device_index=0,
        )
        report = preflight._make_report(
            fixture=fixture,
            attestation=attestation,
            training_summary=training,
            inference_summary=inference,
            projection=projection,
            ledger_rows=normalized,
            validation_manifest_commitment=validation_commitment,
            overhead_cost_probe_report={"probe_id": "e" * 64, **probes},
            lock_handshake_artifact_sha256="f" * 64,
            preflight_started_perf_counter_ns=12,
        )
        self.assertEqual(
            preflight._validate_preflight_report_fixture(report, normalized), report
        )
        tampered = dict(report)
        tampered["rng_state_restored"] = False
        with self.assertRaisesRegex(ValueError, "RNG"):
            preflight._validate_preflight_report_fixture(tampered, normalized)
        tampered = dict(report)
        tampered["diagnostic_checkpoint_writes"] = 0
        tampered.pop("report_id")
        tampered["report_id"] = preflight.canonical_sha256(tampered)
        with self.assertRaisesRegex(ValueError, "diagnostic checkpoint"):
            preflight._validate_preflight_report_fixture(tampered, normalized)


_FULL_PREFLIGHT_ENABLED = (
    os.environ.get("AFTS_RUN_M04A_FULL_PREFLIGHT_TEST") == "1"
    and torch.cuda.is_available()
    and torch.cuda.device_count() == 1
    and torch.cuda.is_bf16_supported()
    and os.environ.get("CUBLAS_WORKSPACE_CONFIG") == CUBLAS_WORKSPACE_CONFIG
)


@unittest.skipUnless(
    _FULL_PREFLIGHT_ENABLED,
    "set AFTS_RUN_M04A_FULL_PREFLIGHT_TEST=1 in the frozen CUDA launcher",
)
class M04aExactPreflightCudaGate(unittest.TestCase):
    def test_full_exact_preflight_must_return_a_pass_result(self) -> None:
        required_names = (
            "AFTS_M04A_VALIDATION_MANIFEST_DIR",
            "AFTS_M04A_VALIDATION_OUTER_SHA256",
            "AFTS_M04A_VALIDATION_JSONL_SHA256",
            "AFTS_M04A_RUN_ROOT",
            "AFTS_M04A_COST_PROBE_SCRATCH_DIR",
            "AFTS_M04A_RUNTIME_SOURCE_SHA256",
            "AFTS_M04A_TEST_SOURCE_SHA256",
            "AFTS_M04A_LOCK_HANDSHAKE_ARTIFACT_JSON",
            "AFTS_M04A_LOCK_HANDSHAKE_ARTIFACT_SHA256",
            "AFTS_M04A_REMOTE_LAUNCHER_SHA256",
            "AFTS_M04A_LAUNCH_PLAN_JSON",
            "AFTS_M04A_LAUNCH_PLAN_SHA256",
            "AFTS_M04A_REMOTE_PROJECT_ROOT",
            "AFTS_M04A_ATTEMPT_NONCE",
        )
        launch = {name: os.environ.get(name) for name in required_names}
        missing = [name for name, value in launch.items() if not value]
        if missing:
            self.fail(f"frozen CUDA launcher omitted: {', '.join(missing)}")
        validation_manifest = read_validation_episode_manifest(
            str(launch["AFTS_M04A_VALIDATION_MANIFEST_DIR"]),
            expected_artifact_manifest_sha256=str(
                launch["AFTS_M04A_VALIDATION_OUTER_SHA256"]
            ),
        )
        validation_commitment = build_validation_manifest_commitment(
            validation_manifest,
            expected_artifact_manifest_sha256=str(
                launch["AFTS_M04A_VALIDATION_OUTER_SHA256"]
            ),
            expected_jsonl_sha256=str(
                launch["AFTS_M04A_VALIDATION_JSONL_SHA256"]
            ),
        )
        attestation = configure_deterministic_cuda()
        launch_plan_artifact = read_launch_plan_artifact(
            Path(str(launch["AFTS_M04A_LAUNCH_PLAN_JSON"])),
            expected_artifact_sha256=str(
                launch["AFTS_M04A_LAUNCH_PLAN_SHA256"]
            ),
        )
        lock_artifact = read_lock_handshake_artifact(
            Path(str(launch["AFTS_M04A_LOCK_HANDSHAKE_ARTIFACT_JSON"])),
            expected_artifact_sha256=str(
                launch["AFTS_M04A_LOCK_HANDSHAKE_ARTIFACT_SHA256"]
            ),
            expected_run_id=attestation.run_id,
            expected_gpu_uuid=attestation.gpu_uuid,
            expected_launcher_sha256=str(
                launch["AFTS_M04A_REMOTE_LAUNCHER_SHA256"]
            ),
            expected_launch_plan_sha256=str(
                launch["AFTS_M04A_LAUNCH_PLAN_SHA256"]
            ),
            expected_remote_project_root=str(
                launch["AFTS_M04A_REMOTE_PROJECT_ROOT"]
            ),
            expected_attempt_nonce=str(launch["AFTS_M04A_ATTEMPT_NONCE"]),
        )
        try:
            result = run_exact_preflight(
                runtime_attestation=attestation,
                validation_manifest_commitment=validation_commitment,
                run_root=str(launch["AFTS_M04A_RUN_ROOT"]),
                cost_probe_scratch_dir=str(
                    launch["AFTS_M04A_COST_PROBE_SCRATCH_DIR"]
                ),
                config_sha256=training_config_sha256(),
                runtime_source_sha256=str(
                    launch["AFTS_M04A_RUNTIME_SOURCE_SHA256"]
                ),
                test_source_sha256=str(launch["AFTS_M04A_TEST_SOURCE_SHA256"]),
                lock_handshake_artifact=lock_artifact,
                launch_plan_artifact=launch_plan_artifact,
            )
        except PreflightFailure as exc:
            self.fail(
                "terminal preflight failure must make the CUDA gate nonzero: "
                f"{exc.failure_code}; report={exc.report}"
            )
        else:
            self.assertEqual(len(result.ledger_rows), 100)
            self.assertEqual(result.report["status"], "PASS")
            self.assertEqual(result.report["dataset_file_reads"], 0)
            self.assertEqual(result.report["training_checkpoint_writes"], 0)
            self.assertEqual(result.report["diagnostic_checkpoint_writes"], 1)
            self.assertFalse(result.report["fallback_used"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
