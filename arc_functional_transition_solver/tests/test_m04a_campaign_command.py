"""Torch-free ownership-transfer tests for the formal M04a command."""

from __future__ import annotations

import sys
import tempfile
import types
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from afts_arc import m04a_campaign_command as campaign_command
from test_m04a_preflight_command import GPU_UUID, _CommandFixture


class M04ACampaignCommandTests(unittest.TestCase):
    def _patches(
        self,
        fixture: _CommandFixture,
        *,
        campaign_runner: mock.Mock,
        final_data_validation: object | BaseException = SimpleNamespace(),
    ) -> ExitStack:
        fixture.output = fixture.run_root / "training-artifact"
        data_root = fixture.root / "data"
        data_binding = SimpleNamespace(root=data_root, sealed=True)
        fake_campaign = types.ModuleType("afts_arc.m04a_campaign")
        fake_campaign.run_primary_training_campaign = campaign_runner
        fixture.campaign_receipt = mock.Mock()
        fake_campaign.ensure_campaign_failure_receipt = fixture.campaign_receipt

        stack = fixture.patches()
        stack.enter_context(
            mock.patch.object(
                campaign_command,
                "read_launch_plan_artifact",
                return_value=fixture.launch_artifact,
            )
        )
        stack.enter_context(
            mock.patch.object(
                campaign_command,
                "read_python_runtime_lock_artifact",
                return_value=fixture.runtime_lock_artifact,
            )
        )
        stack.enter_context(
            mock.patch.object(
                campaign_command,
                "validate_live_python_runtime_lock",
                return_value=dict(fixture.runtime_lock_payload),
            )
        )
        stack.enter_context(
            mock.patch.object(campaign_command, "validate_imported_torch_runtime")
        )
        stack.enter_context(
            mock.patch.object(campaign_command, "verify_source_snapshot_zip")
        )
        stack.enter_context(
            mock.patch.object(campaign_command, "verify_test_source_snapshot_zip")
        )
        stack.enter_context(
            mock.patch.object(
                campaign_command,
                "runtime_source_fingerprint",
                return_value=fixture.plan["runtime_source_fingerprint_sha256"],
            )
        )
        stack.enter_context(
            mock.patch.object(
                campaign_command,
                "read_lock_handshake_artifact",
                return_value=fixture.lock_artifact,
            )
        )
        stack.enter_context(
            mock.patch.object(
                campaign_command,
                "_fixed_production_data_root",
                return_value=data_root,
            )
        )
        stack.enter_context(
            mock.patch.object(
                campaign_command,
                "validate_sealed_production_data_snapshot",
                return_value=data_binding,
            )
        )
        if isinstance(final_data_validation, BaseException):
            data_validation = mock.patch.object(
                campaign_command,
                "revalidate_production_data_binding",
                side_effect=final_data_validation,
            )
        else:
            data_validation = mock.patch.object(
                campaign_command,
                "revalidate_production_data_binding",
                return_value=final_data_validation,
            )
        stack.enter_context(data_validation)
        stack.enter_context(
            mock.patch.object(
                campaign_command, "validate_environment_manifest_artifact"
            )
        )
        stack.enter_context(
            mock.patch.dict(sys.modules, {fake_campaign.__name__: fake_campaign})
        )
        return stack

    @staticmethod
    def _invoke(fixture: _CommandFixture) -> dict[str, object]:
        return campaign_command.run_training_campaign_command(
            launch_plan_path=fixture.root / "launch-plan.json",
            launch_plan_sha256=fixture.launch_artifact.artifact_sha256,
            python_runtime_lock_path=fixture.visible_root / "python-runtime-lock.json",
            python_runtime_lock_sha256=fixture.runtime_lock_artifact.artifact_sha256,
            validation_manifest_dir=fixture.root / "validation",
            lock_handshake_path=fixture.root / "lock.json",
            lock_handshake_sha256=fixture.lock_artifact.artifact_sha256,
            gpu_uuid=GPU_UUID,
            visible_root=fixture.visible_root,
            launcher_path=fixture.launcher,
            cost_probe_scratch_dir=fixture.scratch,
            production_data_root=fixture.root / "data",
            working_dir=fixture.run_root / "campaign-working",
            output_dir=fixture.output,
        )

    def test_exact_preflight_result_transfers_to_campaign_in_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _CommandFixture(Path(directory))
            events: list[str] = []
            fixture.run.side_effect = lambda **_kwargs: (
                events.append("exact-preflight") or fixture.result
            )

            def run_campaign(**kwargs):
                events.append("primary-campaign")
                self.assertIs(kwargs["preflight_result"], fixture.result)
                self.assertEqual(
                    kwargs["working_dir"], fixture.run_root / "campaign-working"
                )
                self.assertEqual(kwargs["output_dir"], fixture.output)
                self.assertTrue(kwargs["production_data_binding"].sealed)
                return SimpleNamespace(report={"status": "published"})

            runner = mock.Mock(side_effect=run_campaign)
            with self._patches(fixture, campaign_runner=runner):
                report = self._invoke(fixture)

            self.assertEqual(report, {"status": "published"})
            self.assertEqual(events, ["exact-preflight", "primary-campaign"])
            fixture.release.assert_not_called()
            runner.assert_called_once()

    def test_post_preflight_preparation_failure_releases_lock_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _CommandFixture(Path(directory))
            runner = mock.Mock()
            with (
                self._patches(
                    fixture,
                    campaign_runner=runner,
                    final_data_validation=ValueError("data changed after preflight"),
                ),
                self.assertRaisesRegex(ValueError, "data changed after preflight"),
            ):
                self._invoke(fixture)

            fixture.release.assert_called_once_with(fixture.result.held_lock_handshake)
            fixture.campaign_receipt.assert_called_once()
            self.assertEqual(
                fixture.campaign_receipt.call_args.kwargs["failure_code"],
                "CAMPAIGN_INCOMPLETE",
            )
            runner.assert_not_called()

    def test_campaign_failure_is_not_double_released_by_command(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _CommandFixture(Path(directory))
            runner = mock.Mock(side_effect=RuntimeError("primary campaign failed"))
            with (
                self._patches(fixture, campaign_runner=runner),
                self.assertRaisesRegex(RuntimeError, "primary campaign failed"),
            ):
                self._invoke(fixture)

            runner.assert_called_once()
            fixture.release.assert_not_called()
            fixture.campaign_receipt.assert_not_called()


if __name__ == "__main__":
    unittest.main()
