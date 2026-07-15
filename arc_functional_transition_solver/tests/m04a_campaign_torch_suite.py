"""Explicit CPU tests for the M04a formal-campaign checkpoint boundary."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from afts_arc import m04a_campaign as campaign
from afts_arc.m04a_torch_runtime import save_checkpoint_new
from afts_arc.m04a_train_contract import BASE_LEARNING_RATE, learning_rate_for_update


_CONFIG_SHA256 = "a" * 64
_RUNTIME_SHA256 = "b" * 64
_TEST_SHA256 = "c" * 64


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([[1.0, -2.0]], dtype=torch.float32))
        self.bias = nn.Parameter(torch.tensor([0.5], dtype=torch.float32))


def _tiny_optimizer(model: nn.Module) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        tuple(model.parameters()),
        lr=BASE_LEARNING_RATE,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
        amsgrad=False,
        maximize=False,
        foreach=False,
        capturable=False,
        differentiable=False,
        fused=False,
    )


def _materialized_checkpoint(path: Path) -> tuple[dict[str, object], tuple[str, ...]]:
    model = _TinyModel().train()
    optimizer = _tiny_optimizer(model)
    loss = sum(parameter.square().sum() for parameter in model.parameters())
    loss.backward()
    optimizer.param_groups[0]["lr"] = learning_rate_for_update(1)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    metadata = save_checkpoint_new(
        path,
        optimizer_step=1,
        model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict(),
        config_sha256=_CONFIG_SHA256,
        runtime_source_sha256=_RUNTIME_SHA256,
        test_source_sha256=_TEST_SHA256,
    )
    return metadata, tuple(model.state_dict())


class M04ACampaignCheckpointTests(unittest.TestCase):
    def test_weights_only_roundtrip_restores_rng_and_checks_complete_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "checkpoint.pt"
            metadata, state_keys = _materialized_checkpoint(path)
            torch.manual_seed(98_765)
            rng_before = torch.get_rng_state().clone()
            with (
                patch.object(campaign, "GridCMLM", _TinyModel),
                patch.object(campaign, "build_adamw_optimizer", _tiny_optimizer),
                patch.object(campaign, "assert_grid_cmlm_invariants"),
                patch.object(campaign, "assert_adamw_invariants"),
            ):
                elapsed = campaign.verify_checkpoint_roundtrip(
                    path,
                    checkpoint_sha256=str(metadata["sha256"]),
                    checkpoint_bytes=int(metadata["bytes"]),
                    optimizer_step=1,
                    config_sha256=_CONFIG_SHA256,
                    runtime_source_sha256=_RUNTIME_SHA256,
                    test_source_sha256=_TEST_SHA256,
                    expected_model_state_keys=state_keys,
                )
            self.assertGreater(elapsed, 0)
            self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))

    def test_failed_roundtrip_also_restores_rng(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "checkpoint.pt"
            metadata, state_keys = _materialized_checkpoint(path)
            torch.manual_seed(12_345)
            rng_before = torch.get_rng_state().clone()
            with (
                patch.object(campaign, "GridCMLM", _TinyModel),
                patch.object(campaign, "build_adamw_optimizer", _tiny_optimizer),
                self.assertRaisesRegex(ValueError, "SHA-256 mismatch"),
            ):
                campaign.verify_checkpoint_roundtrip(
                    path,
                    checkpoint_sha256="0" * 64,
                    checkpoint_bytes=int(metadata["bytes"]),
                    optimizer_step=1,
                    config_sha256=_CONFIG_SHA256,
                    runtime_source_sha256=_RUNTIME_SHA256,
                    test_source_sha256=_TEST_SHA256,
                    expected_model_state_keys=state_keys,
                )
            self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))


class M04ACampaignLockLifecycleTests(unittest.TestCase):
    def test_progress_interval_matches_frozen_operational_contract(self) -> None:
        self.assertEqual(campaign.PROGRESS_UPDATE_INTERVAL, 50)

    @staticmethod
    def _shell_preflight_result() -> tuple[object, object]:
        lock = object()
        result = object.__new__(campaign.PreflightResult)
        object.__setattr__(result, "held_lock_handshake", lock)
        return result, lock

    @staticmethod
    def _invoke(result):
        return campaign.run_primary_training_campaign(
            preflight_result=result,
            runtime_attestation=None,
            environment_manifest={},
            python_runtime_lock=None,
            visible_input_snapshots={},
            validation_manifest_dir="validation",
            production_data_root="data",
            working_dir="working",
            output_dir="output",
        )

    def test_success_has_idempotent_release_backstop(self) -> None:
        result, lock = self._shell_preflight_result()
        publication = object()
        with (
            patch.object(
                campaign,
                "_run_primary_training_campaign_held",
                return_value=publication,
            ),
            patch.object(campaign, "release_held_gpu_lock") as release,
        ):
            observed = self._invoke(result)
        self.assertIs(observed, publication)
        release.assert_called_once_with(lock)

    def test_early_failure_releases_before_reraising_original_cause(self) -> None:
        result, lock = self._shell_preflight_result()
        with (
            patch.object(
                campaign,
                "_run_primary_training_campaign_held",
                side_effect=ValueError("injected preparation failure"),
            ),
            patch.object(campaign, "release_held_gpu_lock") as release,
            self.assertRaisesRegex(ValueError, "injected preparation failure"),
        ):
            self._invoke(result)
        release.assert_called_once_with(lock)

    def test_early_failure_materializes_terminal_ineligible_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            run_root = root / "run"
            run_root.mkdir()
            result, lock = self._shell_preflight_result()
            object.__setattr__(
                result,
                "launch_plan_artifact",
                SimpleNamespace(
                    payload={"run_id": "early-failure", "run_root": str(run_root)}
                ),
            )
            working = run_root / "campaign-working"
            with (
                patch.object(
                    campaign,
                    "_run_primary_training_campaign_held",
                    side_effect=ValueError("injected preparation failure"),
                ),
                patch.object(campaign, "release_held_gpu_lock") as release,
                patch.object(campaign.os, "fchmod", create=True),
                self.assertRaisesRegex(ValueError, "injected preparation failure"),
            ):
                campaign.run_primary_training_campaign(
                    preflight_result=result,
                    runtime_attestation=None,
                    environment_manifest={},
                    python_runtime_lock=None,
                    visible_input_snapshots={},
                    validation_manifest_dir="validation",
                    production_data_root="data",
                    working_dir=working,
                    output_dir=run_root / "training-artifact",
                )
            release.assert_called_once_with(lock)
            receipt = json.loads(
                (working / "campaign_failure.json").read_text(encoding="utf-8")
            )
            self.assertEqual(receipt["status"], "CAMPAIGN_INCOMPLETE")
            self.assertEqual(receipt["cause_type"], "ValueError")
            self.assertTrue(receipt["lock_release_succeeded"])
            self.assertFalse(receipt["training_evidence_eligible"])

    def test_nonexact_preflight_after_transfer_still_releases_and_receipts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_root = Path(temporary_directory) / "run"
            run_root.mkdir()
            lock = object()
            preflight = SimpleNamespace(
                held_lock_handshake=lock,
                launch_plan_artifact=SimpleNamespace(
                    payload={"run_id": "wrong-type", "run_root": str(run_root)}
                ),
            )
            working = run_root / "campaign-working"
            with (
                patch.object(campaign, "release_held_gpu_lock") as release,
                patch.object(campaign.os, "fchmod", create=True),
                self.assertRaisesRegex(TypeError, "exact PreflightResult"),
            ):
                campaign.run_primary_training_campaign(
                    preflight_result=preflight,
                    runtime_attestation=None,
                    environment_manifest={},
                    python_runtime_lock=None,
                    visible_input_snapshots={},
                    validation_manifest_dir="validation",
                    production_data_root="data",
                    working_dir=working,
                    output_dir=run_root / "training-artifact",
                )
            release.assert_called_once_with(lock)
            receipt = json.loads(
                (working / "campaign_failure.json").read_text(encoding="utf-8")
            )
            self.assertEqual(receipt["cause_type"], "TypeError")
            self.assertTrue(receipt["lock_release_succeeded"])

    def test_postcommit_reporting_error_never_creates_failure_receipt(self) -> None:
        result, lock = self._shell_preflight_result()
        error = RuntimeError("postcommit reporting failure")
        error.campaign_output_committed = True
        with (
            patch.object(
                campaign,
                "_run_primary_training_campaign_held",
                side_effect=error,
            ),
            patch.object(campaign, "release_held_gpu_lock") as release,
            patch.object(campaign, "ensure_campaign_failure_receipt") as receipt,
            self.assertRaisesRegex(RuntimeError, "postcommit reporting failure"),
        ):
            self._invoke(result)
        release.assert_called_once_with(lock)
        receipt.assert_not_called()

    def test_blocked_working_path_uses_fixed_run_root_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_root = Path(temporary_directory) / "run"
            run_root.mkdir()
            working = run_root / "campaign-working"
            working.write_bytes(b"blocking-file")
            with patch.object(campaign.os, "fchmod", create=True):
                receipt_path = campaign.ensure_campaign_failure_receipt(
                    run_root=run_root,
                    working_dir=working,
                    run_id="blocked-working",
                    failure_code="CAMPAIGN_INCOMPLETE",
                    optimizer_step=0,
                    cause_type="FileExistsError",
                    lock_release_attempted=True,
                    lock_release_succeeded=True,
                )
            self.assertEqual(receipt_path, run_root / "campaign_failure.json")
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertEqual(receipt["cause_type"], "FileExistsError")
            self.assertFalse(receipt["training_evidence_eligible"])

    def test_preexisting_forged_failure_receipts_are_rejected(self) -> None:
        for block_working in (False, True):
            with self.subTest(block_working=block_working):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    run_root = Path(temporary_directory) / "run"
                    run_root.mkdir()
                    working = run_root / "campaign-working"
                    if block_working:
                        working.write_bytes(b"block")
                        receipt = run_root / "campaign_failure.json"
                    else:
                        working.mkdir()
                        receipt = working / "campaign_failure.json"
                    receipt.write_bytes(b"{}\n")
                    with (
                        patch.object(campaign.os, "fchmod", create=True),
                        self.assertRaisesRegex(RuntimeError, "differs"),
                    ):
                        campaign.ensure_campaign_failure_receipt(
                            run_root=run_root,
                            working_dir=working,
                            run_id="forged-receipt",
                            failure_code="CAMPAIGN_INCOMPLETE",
                            optimizer_step=0,
                            cause_type="ValueError",
                            lock_release_attempted=True,
                            lock_release_succeeded=True,
                        )

    def test_budget_failure_preserves_only_ineligible_working_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            run_root = root / "run"
            run_root.mkdir()
            sanitized = root / "sanitized"
            sanitized.mkdir()
            data_split = b"frozen-data-split"
            (sanitized / "data_split_manifest.json").write_bytes(data_split)
            runtime_lock_snapshot = b"runtime-lock"
            visible_snapshots = {
                "data_split_manifest.json": data_split,
                "python-runtime-lock.json": runtime_lock_snapshot,
            }
            expected_inputs = {
                name: hashlib.sha256(content).hexdigest()
                for name, content in visible_snapshots.items()
            }
            run_id = "budget-failure-fixture"
            plan_artifact = SimpleNamespace(
                payload={
                    "run_id": run_id,
                    "run_root": str(run_root),
                    "expected_input_artifacts": expected_inputs,
                }
            )
            commitment = SimpleNamespace(
                outer_artifact_manifest_sha256="a" * 64,
                jsonl_sha256="b" * 64,
                row_count=1,
            )
            lock = object()
            result = object.__new__(campaign.PreflightResult)
            for field, value in {
                "held_lock_handshake": lock,
                "launch_plan_artifact": plan_artifact,
                "validation_manifest_commitment": commitment,
                "ledger_rows": (),
            }.items():
                object.__setattr__(result, field, value)
            binding = SimpleNamespace(
                closure_id=campaign.PRODUCTION_DATA_CLOSURE_ID,
                sanitized_bundle_dir=sanitized,
                training_rearc_cache_dir=root / "train-cache",
                validation_rearc_cache_dir=root / "validation-cache",
            )
            lock_payload = {
                "run_id": run_id,
                "acquisition_started_perf_counter_ns": 1,
            }
            validation_manifest = SimpleNamespace(rows=({"episode": 0},))
            with (
                patch.object(campaign, "validate_runtime_attestation"),
                patch.object(
                    campaign,
                    "assert_held_gpu_lock",
                    return_value=lock_payload,
                ),
                patch.object(
                    campaign,
                    "validate_production_data_root",
                    return_value=binding,
                ),
                patch.object(
                    campaign,
                    "PRODUCTION_DATA_FILES",
                    {
                        "sanitized_split/data_split_manifest.json": (
                            len(data_split),
                            hashlib.sha256(data_split).hexdigest(),
                        )
                    },
                ),
                patch.object(
                    campaign,
                    "read_validation_episode_manifest",
                    return_value=validation_manifest,
                ),
                patch.object(
                    campaign,
                    "require_externally_committed_validation_manifest",
                ),
                patch.object(
                    campaign,
                    "assert_campaign_budget",
                    side_effect=RuntimeError("BUDGET_EXCEEDED"),
                ),
                patch.object(campaign, "release_held_gpu_lock") as release,
                patch.object(campaign.os, "fchmod", create=True),
                self.assertRaises(campaign.CampaignFailure) as raised,
            ):
                campaign._run_primary_training_campaign_held(
                    preflight_result=result,
                    runtime_attestation=None,
                    environment_manifest={},
                    python_runtime_lock=SimpleNamespace(
                        snapshot=runtime_lock_snapshot
                    ),
                    visible_input_snapshots=visible_snapshots,
                    validation_manifest_dir=root / "validation-manifest",
                    production_data_root=root / "data",
                    working_dir=run_root / "campaign-working",
                    output_dir=run_root / "training-artifact",
                )

            self.assertEqual(raised.exception.failure_code, "BUDGET_EXCEEDED")
            self.assertEqual(raised.exception.optimizer_step, 0)
            release.assert_called_once_with(lock)
            receipt_path = run_root / "campaign-working" / "campaign_failure.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertEqual(receipt["status"], "BUDGET_EXCEEDED")
            self.assertIs(receipt["training_evidence_eligible"], False)
            self.assertIs(receipt["selected_checkpoint_complete"], False)
            self.assertIs(receipt["same_run_retry_allowed"], False)
            self.assertFalse((run_root / "training-artifact").exists())

    def test_reduced_success_path_releases_before_cpu_publication(self) -> None:
        """Exercise the whole campaign control path with four cheap updates."""

        repository = Path(__file__).resolve().parents[1]
        split_path = (
            repository
            / "results"
            / "m04a_global_source_v0_1"
            / "sanitized_split"
            / "data_split_manifest.json"
        )
        if not split_path.is_file():
            self.skipTest("canonical production split fixture is absent")
        generated_names = {
            "checkpoint_manifests.jsonl",
            "environment_manifest.json",
            "launch_plan.json",
            "lock_handshake_artifact.json",
            "optimizer_schedule_state.json",
            "preflight.json",
            "preflight_diagnostic_checkpoint.pt",
            "preflight_overhead_cost_report.json",
            "resume_manifests.jsonl",
            "selected_checkpoint.pt",
            "selected_checkpoint_manifest.json",
            "training_cost_ledger.jsonl",
            "training_cost_summary.json",
            "validation_metrics.jsonl",
        }
        input_names = set(campaign.evidence.TRAINING_ARTIFACT_FILES) - generated_names
        snapshots = {name: f"fixture:{name}".encode() for name in input_names}
        snapshots["data_split_manifest.json"] = split_path.read_bytes()
        runtime_lock_snapshot = snapshots["python-runtime-lock.json"]

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            run_root = root / "run"
            run_root.mkdir()
            data_root = root / "data"
            data_root.mkdir()
            binding = campaign.ProductionDataBinding(
                root=data_root,
                sanitized_bundle_dir=data_root / "sanitized_split",
                training_rearc_cache_dir=data_root / "rearc_train_cache",
                validation_rearc_cache_dir=data_root / "rearc_validation_cache",
                closure_id=campaign.PRODUCTION_DATA_CLOSURE_ID,
            )
            plan = {
                "run_id": "reduced-success",
                "run_root": str(run_root),
                "expected_input_artifacts": {
                    name: hashlib.sha256(content).hexdigest()
                    for name, content in snapshots.items()
                },
                "runtime_source_fingerprint_sha256": "d" * 64,
            }
            plan_artifact = SimpleNamespace(
                payload=plan,
                snapshot=b"launch-plan",
                artifact_sha256="e" * 64,
            )
            lock_artifact = SimpleNamespace(
                snapshot=b"held-lock",
                artifact_sha256="f" * 64,
            )
            held_lock = SimpleNamespace(artifact=lock_artifact)
            preflight = object.__new__(campaign.PreflightResult)
            fields = {
                "report": {
                    "budget_projection": {
                        "measured_preflight_total_wall_ns": 1
                    }
                },
                "ledger_rows": (),
                "validation_manifest_commitment": SimpleNamespace(
                    outer_artifact_manifest_sha256="1" * 64,
                    jsonl_sha256="2" * 64,
                    row_count=1,
                ),
                "launch_plan_artifact": plan_artifact,
                "overhead_cost_probe_report": {"status": "PASS"},
                "diagnostic_checkpoint_snapshot": b"diagnostic",
                "expected_config_sha256": "a" * 64,
                "expected_runtime_source_sha256": "b" * 64,
                "expected_test_source_sha256": "c" * 64,
                "held_lock_handshake": held_lock,
            }
            for field, value in fields.items():
                object.__setattr__(preflight, field, value)

            class FakeIndexed:
                training = object()

                def __enter__(self):
                    return self

                def __exit__(self, *_args):
                    return None

            class FakeModel:
                def to(self, **_kwargs):
                    return self

                def train(self):
                    return self

            class FakeValidation:
                def __init__(self, event_index: int) -> None:
                    self.ledger_rows = (
                        {"kind": "validation", "event_index": event_index},
                    )

                @staticmethod
                def evidence_episode_metrics() -> list[object]:
                    return []

            def make_checkpoint(**kwargs):
                index = kwargs["checkpoint_index"]
                path = kwargs["checkpoint_dir"] / f"checkpoint-{index:02d}.pt"
                content = f"checkpoint:{index}".encode()
                path.write_bytes(content)
                return campaign.CheckpointArtifact(
                    checkpoint_index=index,
                    optimizer_step=index * 2,
                    path=path,
                    sha256=hashlib.sha256(content).hexdigest(),
                    bytes=len(content),
                    checkpoint_event={
                        "phase": "checkpoint_operation",
                        "event_id": f"checkpoint-event-{index}",
                    },
                )

            def make_metric(**kwargs):
                index = kwargs["validation_pass_index"]
                return {
                    "metric_id": f"metric-{index}",
                    "validation_pass_index": index,
                    "optimizer_step": index * 2,
                    "checkpoint_sha256": kwargs["checkpoint_sha256"],
                    "parent_grouped_masked_cell_ce_hex": float(3 - index).hex(),
                }

            def make_checkpoint_row(**kwargs):
                return {
                    "checkpoint_index": kwargs["checkpoint_index"],
                    "checkpoint_sha256": kwargs["checkpoint_sha256"],
                    "checkpoint_bytes": kwargs["checkpoint_bytes"],
                }

            selection_endpoints: list[int] = []

            def make_selected(**kwargs):
                selection_endpoints.append(
                    kwargs["selection_completed_perf_counter_ns"]
                )
                winner = kwargs["checkpoint_rows"][1]
                return {
                    "selected_checkpoint_index": 2,
                    "checkpoint_sha256": winner["checkpoint_sha256"],
                    "checkpoint_bytes": winner["checkpoint_bytes"],
                }

            timing = {
                field: 1
                for field in campaign.evidence.TRAINING_ARTIFACT_TIMING_FIELDS
            }
            published_manifest = {"semantic_id": "reduced-success-artifact"}
            events: list[str] = []
            progress_rows: list[dict[str, object]] = []

            def release(_lock):
                events.append("release")

            def publish(output, **_kwargs):
                self.assertEqual(events, ["release"])
                events.append("publish")
                output.mkdir()
                (output / "training_artifact_manifest.json").write_bytes(b"training")
                (output / "artifact_manifest.json").write_bytes(b"outer")
                return SimpleNamespace(
                    manifest=published_manifest,
                    outer_manifest={"bundle_id": "reduced-success-artifact"},
                )

            with ExitStack() as stack:
                stack.enter_context(patch.object(campaign, "OPTIMIZER_UPDATES", 4))
                stack.enter_context(patch.object(campaign, "VALIDATION_INTERVAL", 2))
                stack.enter_context(patch.object(campaign, "VALIDATION_PASS_COUNT", 2))
                stack.enter_context(patch.object(campaign, "PROGRESS_UPDATE_INTERVAL", 2))
                stack.enter_context(patch.object(campaign, "validate_runtime_attestation"))
                stack.enter_context(
                    patch.object(
                        campaign,
                        "assert_held_gpu_lock",
                        return_value={
                            "run_id": plan["run_id"],
                            "acquisition_started_perf_counter_ns": 1,
                        },
                    )
                )
                stack.enter_context(
                    patch.object(
                        campaign,
                        "revalidate_production_data_binding",
                        return_value=binding,
                    )
                )
                stack.enter_context(
                    patch.object(
                        campaign,
                        "read_validation_episode_manifest",
                        return_value=SimpleNamespace(rows=({"episode": 1},)),
                    )
                )
                stack.enter_context(
                    patch.object(
                        campaign, "require_externally_committed_validation_manifest"
                    )
                )
                stack.enter_context(
                    patch.object(
                        campaign, "load_indexed_m04a_data", return_value=FakeIndexed()
                    )
                )
                stack.enter_context(patch.object(campaign, "GridCMLM", FakeModel))
                stack.enter_context(
                    patch.object(campaign, "build_adamw_optimizer", return_value=object())
                )
                stack.enter_context(patch.object(campaign, "assert_grid_cmlm_invariants"))
                stack.enter_context(patch.object(campaign, "assert_adamw_invariants"))
                stack.enter_context(
                    patch.object(
                        campaign,
                        "train_primary_update",
                        side_effect=lambda *_args, **kwargs: SimpleNamespace(
                            ledger_row={
                                "kind": "train",
                                "optimizer_step": kwargs["optimizer_step"],
                            }
                        ),
                    )
                )
                stack.enter_context(
                    patch.object(
                        campaign,
                        "validate_literal_manifest",
                        side_effect=lambda *_args, **kwargs: FakeValidation(
                            kwargs["event_index_start"]
                        ),
                    )
                )
                stack.enter_context(
                    patch.object(campaign, "_make_checkpoint", side_effect=make_checkpoint)
                )
                stack.enter_context(
                    patch.object(
                        campaign, "make_validation_metric_row", side_effect=make_metric
                    )
                )
                stack.enter_context(
                    patch.object(
                        campaign,
                        "make_checkpoint_manifest_row",
                        side_effect=make_checkpoint_row,
                    )
                )
                stack.enter_context(
                    patch.object(
                        campaign,
                        "make_selected_checkpoint_manifest",
                        side_effect=make_selected,
                    )
                )
                stack.enter_context(
                    patch.object(
                        campaign,
                        "validate_training_cost_ledger",
                        return_value=timing,
                    )
                )
                stack.enter_context(
                    patch.object(campaign, "_training_closure", return_value={"ok": True})
                )
                stack.enter_context(
                    patch.object(
                        campaign,
                        "make_training_cost_summary",
                        return_value={"status": "fixture"},
                    )
                )
                stack.enter_context(
                    patch.object(campaign, "validate_training_checkpoint_artifacts")
                )
                stack.enter_context(
                    patch.object(
                        campaign,
                        "runtime_source_fingerprint",
                        return_value=plan["runtime_source_fingerprint_sha256"],
                    )
                )
                stack.enter_context(
                    patch.object(campaign, "_artifact_lineage", return_value={"fixture": "x"})
                )
                stack.enter_context(
                    patch.object(
                        campaign, "publish_training_artifact_bundle", side_effect=publish
                    )
                )
                stack.enter_context(
                    patch.object(campaign, "_budget_guard", return_value=500)
                )
                stack.enter_context(patch.object(campaign, "assert_campaign_budget"))
                stack.enter_context(
                    patch.object(campaign, "release_held_gpu_lock", side_effect=release)
                )
                stack.enter_context(
                    patch.object(
                        campaign.time,
                        "perf_counter_ns",
                        side_effect=(100, 110, 1_000),
                    )
                )
                stack.enter_context(patch.object(campaign.torch.cuda, "empty_cache"))
                stack.enter_context(patch.object(campaign.torch, "manual_seed"))
                stack.enter_context(patch.object(campaign.torch.cuda, "manual_seed_all"))
                stack.enter_context(patch.object(campaign.os, "fchmod", create=True))
                publication = campaign.run_primary_training_campaign(
                    preflight_result=preflight,
                    runtime_attestation=None,
                    environment_manifest={"fixture": True},
                    python_runtime_lock=SimpleNamespace(
                        snapshot=runtime_lock_snapshot
                    ),
                    visible_input_snapshots=snapshots,
                    validation_manifest_dir=root / "validation-manifest",
                    production_data_root=data_root,
                    production_data_binding=binding,
                    working_dir=run_root / "campaign-working",
                    output_dir=run_root / "training-artifact",
                    progress_callback=lambda row: progress_rows.append(dict(row)),
                )

            self.assertEqual(
                publication.report["status"],
                "published_complete_m04a_training_campaign",
            )
            self.assertEqual(selection_endpoints, [1_000])
            self.assertEqual(events, ["release", "publish", "release"])
            self.assertEqual(
                [row["optimizer_step"] for row in progress_rows],
                [2, 2, 4, 4],
            )
            self.assertEqual(
                [row["checkpoint_count"] for row in progress_rows],
                [0, 1, 1, 2],
            )
            self.assertTrue(publication.report["training_evidence_eligible"])

if __name__ == "__main__":
    unittest.main()
