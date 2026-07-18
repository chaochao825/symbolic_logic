"""Explicit CPU-fixture tests for the M04a campaign-overhead cost probe."""

from __future__ import annotations

import copy
import hashlib
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import torch

from afts_arc import m04a_cost_probe as cost_probe
from afts_arc.m04a_contract import (
    VALIDATION_INTERVAL,
    VALIDATION_PASS_COUNT,
    canonical_sha256,
)
from afts_arc.m04a_cost_probe import (
    CHECKPOINT_PROJECTION_MULTIPLIER,
    CPU_FIXTURE_RUNTIME_MODE,
    DIAGNOSTIC_DISPOSITION,
    DIAGNOSTIC_STATUS,
    LOCK_HANDSHAKE_PROBE_SCHEMA_VERSION,
    SELECTION_TIE_INDICES,
    run_campaign_overhead_cost_probe,
    run_campaign_overhead_cost_probe_cpu_fixture,
    validate_campaign_overhead_cost_probe_report,
    validate_lock_handshake_probe,
)
from afts_arc.m04a_model import GridCMLM
from afts_arc.m04a_train import build_adamw_optimizer
from afts_arc.m04a_train_contract import (
    PREFLIGHT_UPDATES,
    learning_rate_for_update,
    training_config_sha256,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _handshake(run_id: str) -> dict[str, object]:
    started = time.perf_counter_ns()
    completed = time.perf_counter_ns()
    while completed <= started:
        completed = time.perf_counter_ns()
    payload: dict[str, object] = {
        "schema": LOCK_HANDSHAKE_PROBE_SCHEMA_VERSION,
        "run_id": run_id,
        "acquisition_started_perf_counter_ns": started,
        "handshake_completed_perf_counter_ns": completed,
        "wall_ns": completed - started,
        "source": "verified_launcher_lock_handshake",
    }
    payload["probe_id"] = canonical_sha256(payload)
    return payload


def _materialize_completed_adamw_state(
    model: GridCMLM, optimizer: torch.optim.AdamW
) -> None:
    optimizer.param_groups[0]["lr"] = learning_rate_for_update(PREFLIGHT_UPDATES)
    for parameter in model.parameters():
        optimizer.state[parameter] = {
            "step": torch.tensor(float(PREFLIGHT_UPDATES), dtype=torch.float32),
            "exp_avg": torch.zeros_like(parameter),
            "exp_avg_sq": torch.zeros_like(parameter),
        }


class M04aCostProbePureClosureTests(unittest.TestCase):
    def test_lock_handshake_is_endpoint_derived_and_content_addressed(self) -> None:
        run_id = "cpu-fixture:handshake"
        probe = _handshake(run_id)
        self.assertEqual(
            validate_lock_handshake_probe(probe, expected_run_id=run_id), probe
        )
        tampered = copy.deepcopy(probe)
        tampered["wall_ns"] += 1
        semantic = dict(tampered)
        semantic.pop("probe_id")
        tampered["probe_id"] = canonical_sha256(semantic)
        with self.assertRaisesRegex(ValueError, "endpoints"):
            validate_lock_handshake_probe(tampered, expected_run_id=run_id)

    def test_production_entry_point_cannot_accept_a_cpu_fixture(self) -> None:
        model = GridCMLM()
        optimizer = build_adamw_optimizer(model)
        _materialize_completed_adamw_state(model, optimizer)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises((TypeError, RuntimeError)):
                run_campaign_overhead_cost_probe(
                    model=model,
                    optimizer=optimizer,
                    runtime_attestation=None,  # type: ignore[arg-type]
                    run_root=root,
                    scratch_dir=root / "must-not-exist",
                    config_sha256=training_config_sha256(),
                    runtime_source_sha256="1" * 64,
                    test_source_sha256="2" * 64,
                    lock_handshake_probe=_handshake("cpu-fixture:rejected"),
                )
            self.assertFalse((root / "must-not-exist").exists())


class M04aCostProbeCpuFixtureIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(max(1, min(torch.get_num_threads(), 4)))
        cls.run_id = "cpu-fixture:overhead-roundtrip"
        cls.model = GridCMLM().train()
        cls.optimizer = build_adamw_optimizer(cls.model)
        _materialize_completed_adamw_state(cls.model, cls.optimizer)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.run_root = Path(cls.temporary.name)
        cls.scratch = cls.run_root / "exclusive-cost-probe"
        cls.cpu_rng_before = torch.get_rng_state().clone()
        cls.result = run_campaign_overhead_cost_probe_cpu_fixture(
            model=cls.model,
            optimizer=cls.optimizer,
            run_id=cls.run_id,
            run_root=cls.run_root,
            scratch_dir=cls.scratch,
            config_sha256=training_config_sha256(),
            runtime_source_sha256="3" * 64,
            test_source_sha256="4" * 64,
            lock_handshake_probe=_handshake(cls.run_id),
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.result = None
        cls.optimizer = None
        cls.model = None
        cls.temporary.cleanup()

    def test_checkpoint_is_retained_content_addressed_and_nonselectable(self) -> None:
        report = self.result.report
        checkpoint = report["checkpoint_cost_probe"]
        path = self.result.checkpoint_path
        self.assertTrue(path.is_file())
        self.assertFalse(path.is_symlink())
        self.assertEqual(path.parent, self.scratch)
        self.assertEqual(path.stat().st_size, checkpoint["checkpoint_bytes"])
        self.assertEqual(_sha256_file(path), checkpoint["checkpoint_sha256"])
        self.assertIs(type(self.result.checkpoint_snapshot), bytes)
        self.assertEqual(len(self.result.checkpoint_snapshot), checkpoint["checkpoint_bytes"])
        self.assertEqual(
            hashlib.sha256(self.result.checkpoint_snapshot).hexdigest(),
            checkpoint["checkpoint_sha256"],
        )
        with self.assertRaises(TypeError):
            self.result.checkpoint_snapshot[0] = 0  # type: ignore[index]
        self.assertEqual(
            path.name,
            f"NOT_TRAINING_CHECKPOINT.diagnostic.{checkpoint['checkpoint_sha256']}.pt",
        )
        self.assertEqual(
            checkpoint["artifact_disposition"], DIAGNOSTIC_DISPOSITION
        )
        self.assertFalse(checkpoint["selectable_checkpoint_created"])
        self.assertEqual(report["checkpoint_artifact_status"], DIAGNOSTIC_STATUS)
        self.assertEqual(
            set(child.name for child in self.scratch.iterdir()), {path.name}
        )

    def test_report_and_fixed_validation_projection_close_exactly(self) -> None:
        report = validate_campaign_overhead_cost_probe_report(
            self.result.report, expected_run_id=self.run_id
        )
        self.assertEqual(report["runtime_mode"], CPU_FIXTURE_RUNTIME_MODE)
        with self.assertRaisesRegex(ValueError, "external commitment"):
            validate_campaign_overhead_cost_probe_report(
                report,
                expected_run_id=self.run_id,
                expected_runtime_source_sha256="5" * 64,
            )
        with self.assertRaisesRegex(ValueError, "not frozen CUDA"):
            validate_campaign_overhead_cost_probe_report(
                report,
                expected_run_id=self.run_id,
                require_production_cuda=True,
            )
        relabeled = copy.deepcopy(report)
        relabeled["runtime_mode"] = "frozen_cuda"
        semantic = dict(relabeled)
        semantic.pop("probe_id")
        relabeled["probe_id"] = canonical_sha256(semantic)
        with self.assertRaisesRegex(ValueError, "cannot be relabeled"):
            validate_campaign_overhead_cost_probe_report(
                relabeled,
                expected_run_id=self.run_id,
                expected_config_sha256=training_config_sha256(),
                expected_runtime_source_sha256="3" * 64,
                expected_test_source_sha256="4" * 64,
                require_production_cuda=True,
            )
        self.assertEqual(report["dataset_file_reads"], 0)
        self.assertFalse(report["fallback_used"])
        checkpoint = report["checkpoint_cost_probe"]
        projection = report["overhead_projection"]
        self.assertEqual(
            projection["fixed_checkpoint_multiplier"],
            CHECKPOINT_PROJECTION_MULTIPLIER,
        )
        self.assertEqual(
            projection["projected_checkpoint_writes"], VALIDATION_PASS_COUNT
        )
        self.assertEqual(
            projection["projected_checkpoint_loads"], VALIDATION_PASS_COUNT
        )
        self.assertEqual(
            projection["projected_checkpoint_write_wall_ns"],
            VALIDATION_PASS_COUNT * checkpoint["write_wall_ns"],
        )
        self.assertEqual(
            projection["projected_checkpoint_weights_only_load_wall_ns"],
            VALIDATION_PASS_COUNT * checkpoint["weights_only_load_wall_ns"],
        )
        self.assertEqual(
            projection["projected_checkpoint_roundtrip_wall_ns"],
            VALIDATION_PASS_COUNT * checkpoint["roundtrip_wall_ns"],
        )
        self.assertEqual(
            report["final_selection_probe"]["selected_checkpoint_index"],
            SELECTION_TIE_INDICES[0],
        )
        self.assertEqual(
            report["final_selection_probe"]["candidate_count"],
            VALIDATION_PASS_COUNT,
        )
        self.assertEqual(
            report["selected_metric_summary"]["checkpoint_index"],
            SELECTION_TIE_INDICES[0],
        )
        self.assertEqual(
            report["selected_metric_summary"]["selection_input_sha256"],
            report["selection_input_sha256"],
        )
        self.assertTrue(torch.equal(self.cpu_rng_before, torch.get_rng_state()))

    def test_report_tampering_and_reused_scratch_fail_closed(self) -> None:
        tampered = copy.deepcopy(self.result.report)
        tampered["overhead_projection"][
            "projected_checkpoint_loads"
        ] = VALIDATION_PASS_COUNT - 1
        semantic = dict(tampered["overhead_projection"])
        semantic.pop("projection_id")
        tampered["overhead_projection"]["projection_id"] = canonical_sha256(
            semantic
        )
        semantic = dict(tampered)
        semantic.pop("probe_id")
        tampered["probe_id"] = canonical_sha256(semantic)
        with self.assertRaisesRegex(ValueError, "projection"):
            validate_campaign_overhead_cost_probe_report(tampered)

        selection_tampered = copy.deepcopy(self.result.report)
        selection_tampered["selection_input_sha256"] = "9" * 64
        semantic = dict(selection_tampered)
        semantic.pop("probe_id")
        selection_tampered["probe_id"] = canonical_sha256(semantic)
        with self.assertRaisesRegex(
            ValueError, rf"exact {VALIDATION_PASS_COUNT} metrics"
        ):
            validate_campaign_overhead_cost_probe_report(selection_tampered)

        tie_tampered = copy.deepcopy(self.result.report)
        checkpoint_sha = tie_tampered["checkpoint_cost_probe"][
            "checkpoint_sha256"
        ]
        final_probe = tie_tampered["final_selection_probe"]
        late_tie_index = SELECTION_TIE_INDICES[1]
        final_probe["selected_checkpoint_index"] = late_tie_index
        semantic = dict(final_probe)
        semantic.pop("probe_id")
        final_probe["probe_id"] = canonical_sha256(semantic)
        selected_summary = tie_tampered["selected_metric_summary"]
        selected_summary.update(
            {
                "final_selection_probe_id": final_probe["probe_id"],
                "checkpoint_index": late_tie_index,
                "optimizer_step": late_tie_index * VALIDATION_INTERVAL,
                "parent_grouped_ce_hex": float(0.25).hex(),
                "checkpoint_sha256": hashlib.sha256(
                    f"{checkpoint_sha}:{late_tie_index}".encode("ascii")
                ).hexdigest(),
            }
        )
        semantic = dict(selected_summary)
        semantic.pop("summary_id")
        selected_summary["summary_id"] = canonical_sha256(semantic)
        projection = tie_tampered["overhead_projection"]
        projection["final_selection_probe_id"] = final_probe["probe_id"]
        semantic = dict(projection)
        semantic.pop("projection_id")
        projection["projection_id"] = canonical_sha256(semantic)
        semantic = dict(tie_tampered)
        semantic.pop("probe_id")
        tie_tampered["probe_id"] = canonical_sha256(semantic)
        with self.assertRaisesRegex(ValueError, "argmin/tie-break"):
            validate_campaign_overhead_cost_probe_report(tie_tampered)

        with self.assertRaisesRegex(FileExistsError, "exclusive and fresh"):
            run_campaign_overhead_cost_probe_cpu_fixture(
                model=self.model,
                optimizer=self.optimizer,
                run_id=self.run_id,
                run_root=self.run_root,
                scratch_dir=self.scratch,
                config_sha256=training_config_sha256(),
                runtime_source_sha256="3" * 64,
                test_source_sha256="4" * 64,
                lock_handshake_probe=_handshake(self.run_id),
            )
        self.assertTrue(self.result.checkpoint_path.is_file())

    def test_fallible_probe_phases_never_publish_the_final_checkpoint(self) -> None:
        fault_points = (
            "_verified_weights_only_load",
            "_run_fresh_reconstruction",
            "_run_final_selection",
            "validate_campaign_overhead_cost_probe_report",
            "_read_diagnostic_checkpoint_snapshot",
            "CampaignOverheadCostProbeResult",
        )
        for index, fault_point in enumerate(fault_points):
            with self.subTest(fault_point=fault_point):
                scratch = self.run_root / f"uncommitted-fault-{index}"
                with (
                    mock.patch.object(
                        cost_probe,
                        fault_point,
                        side_effect=RuntimeError(f"injected {fault_point} failure"),
                    ),
                    mock.patch.object(
                        cost_probe,
                        "_rename_no_replace",
                        wraps=cost_probe._rename_no_replace,
                    ) as publish,
                ):
                    with self.assertRaisesRegex(RuntimeError, "injected"):
                        run_campaign_overhead_cost_probe_cpu_fixture(
                            model=self.model,
                            optimizer=self.optimizer,
                            run_id=f"cpu-fixture:fault-{index}",
                            run_root=self.run_root,
                            scratch_dir=scratch,
                            config_sha256=training_config_sha256(),
                            runtime_source_sha256="6" * 64,
                            test_source_sha256="7" * 64,
                            lock_handshake_probe=_handshake(
                                f"cpu-fixture:fault-{index}"
                            ),
                        )
                publish.assert_not_called()
                self.assertEqual(
                    {child.name for child in scratch.iterdir()},
                    {".NOT_TRAINING_CHECKPOINT.diagnostic.partial"},
                )
                self.assertFalse(
                    any(
                        child.name.startswith("NOT_TRAINING_CHECKPOINT.diagnostic.")
                        for child in scratch.iterdir()
                    )
                )

    def test_prepared_probe_restores_rng_and_requires_explicit_commit(self) -> None:
        scratch = self.run_root / "explicit-preflight-owned-commit"
        rng_at_entry = torch.get_rng_state().clone()

        with mock.patch.object(
            cost_probe,
            "_rename_no_replace",
            wraps=cost_probe._rename_no_replace,
        ) as publish:
            prepared = cost_probe._run_probe(
                model=self.model,
                optimizer=self.optimizer,
                run_id="cpu-fixture:preflight-owned-commit",
                runtime_mode=CPU_FIXTURE_RUNTIME_MODE,
                runtime_attestation=None,
                run_root=self.run_root,
                scratch_dir=scratch,
                config_sha256=training_config_sha256(),
                runtime_source_sha256="8" * 64,
                test_source_sha256="9" * 64,
                lock_handshake_probe=_handshake(
                    "cpu-fixture:preflight-owned-commit"
                ),
            )
            self.assertTrue(torch.equal(torch.get_rng_state(), rng_at_entry))
            self.assertFalse(prepared.committed)
            self.assertFalse(prepared.result.checkpoint_path.exists())
            publish.assert_not_called()
            prepared.commit()
            self.assertTrue(prepared.committed)
            self.assertTrue(prepared.result.checkpoint_path.is_file())
            publish.assert_called_once()
        self.assertEqual(
            {child.name for child in scratch.iterdir()},
            {prepared.result.checkpoint_path.name},
        )

    def test_partially_initialized_optimizer_is_rejected_before_scratch_creation(self) -> None:
        parameter = next(iter(self.model.parameters()))
        saved = self.optimizer.state.pop(parameter)
        scratch = self.run_root / "partial-state-must-not-exist"
        try:
            with self.assertRaisesRegex(RuntimeError, "partially materialized"):
                run_campaign_overhead_cost_probe_cpu_fixture(
                    model=self.model,
                    optimizer=self.optimizer,
                    run_id=self.run_id,
                    run_root=self.run_root,
                    scratch_dir=scratch,
                    config_sha256=training_config_sha256(),
                    runtime_source_sha256="3" * 64,
                    test_source_sha256="4" * 64,
                    lock_handshake_probe=_handshake(self.run_id),
                )
        finally:
            self.optimizer.state[parameter] = saved
        self.assertFalse(scratch.exists())

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks are unavailable")
    def test_symlink_scratch_path_is_rejected_without_following_it(self) -> None:
        target = self.run_root / "symlink-target"
        target.mkdir()
        link = self.run_root / "symlink-scratch"
        try:
            os.symlink(target, link, target_is_directory=True)
        except OSError as exc:
            self.skipTest(f"directory symlinks are unavailable: {exc}")
        with self.assertRaisesRegex(FileExistsError, "exclusive and fresh"):
            run_campaign_overhead_cost_probe_cpu_fixture(
                model=self.model,
                optimizer=self.optimizer,
                run_id=self.run_id,
                run_root=self.run_root,
                scratch_dir=link,
                config_sha256=training_config_sha256(),
                runtime_source_sha256="3" * 64,
                test_source_sha256="4" * 64,
                lock_handshake_probe=_handshake(self.run_id),
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
