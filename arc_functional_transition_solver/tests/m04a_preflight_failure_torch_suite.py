"""Explicit Torch-owning failure-evidence suite; excluded from discovery."""

from __future__ import annotations

import copy
import unittest

from afts_arc import m04a_preflight as preflight
from afts_arc.m04a_contract import canonical_sha256
from afts_arc.m04a_evidence import make_training_cost_ledger_row
from m04a_preflight_failure_fixture import _budget_failure_fixture, _digest


class M04aPreflightFailureTests(unittest.TestCase):
    def test_partial_failure_rejects_generic_but_semantically_empty_update(self) -> None:
        row = make_training_cost_ledger_row(
            phase="preflight_update",
            event_index=0,
            optimizer_step=1,
            wall_time_ns=1,
        )
        report = preflight._make_failure_report(
            failure_code=preflight.PREFLIGHT_OOM,
            fixture_id=preflight.build_preflight_fixture().fixture_id,
            completed_updates=1,
            ledger_rows=(row,),
        )
        with self.assertRaisesRegex(ValueError, "semantic counts"):
            preflight.validate_preflight_failure_report(report, (row,))

        wrong_fixture = preflight._make_failure_report(
            failure_code=preflight.PREFLIGHT_OOM,
            fixture_id=_digest("wrong-fixture"),
            completed_updates=0,
            ledger_rows=(),
        )
        with self.assertRaisesRegex(ValueError, "frozen fixture"):
            preflight.validate_preflight_failure_report(wrong_fixture, ())

    def test_full_budget_failure_closes_every_measured_parent(self) -> None:
        report, rows, parents, diagnostic = _budget_failure_fixture()
        overhead = parents["overhead_cost_probe_report"]
        validated = preflight.validate_preflight_failure_report(
            report,
            rows,
            overhead_cost_probe_report=overhead,
            diagnostic_checkpoint_snapshot=diagnostic,
            expected_config_sha256=parents["expected_config_sha256"],
            expected_runtime_source_sha256=parents[
                "expected_runtime_source_sha256"
            ],
            expected_test_source_sha256=parents["expected_test_source_sha256"],
        )
        self.assertEqual(validated["evidence_completeness"], "full_budget_probe")
        self.assertEqual(validated["ledger_row_count"], 100)
        self.assertIs(validated["training_evidence_eligible"], False)
        self.assertEqual(validated["training_checkpoint_writes"], 0)
        self.assertEqual(validated["diagnostic_checkpoint_writes"], 1)
        self.assertIs(
            validated["diagnostic_checkpoint_commitment"][
                "selectable_checkpoint_created"
            ],
            False,
        )

        failure = preflight.PreflightFailure(
            preflight.PREFLIGHT_BUDGET_EXCEEDED,
            "fixture",
            report=report,
            ledger_rows=rows,
            overhead_cost_probe_report=overhead,
            training_summary=report["training_summary"],
            inference_summary=report["inference_summary"],
            diagnostic_checkpoint_snapshot=diagnostic,
        )
        self.assertEqual(len(failure.ledger_rows), 100)
        self.assertEqual(
            failure.overhead_cost_probe_report["probe_id"],
            report["overhead_cost_probe_id"],
        )
        self.assertEqual(
            failure.diagnostic_checkpoint_snapshot,
            diagnostic,
        )

    def test_rejects_resealed_selectable_diagnostic_and_projection_tamper(self) -> None:
        report, rows, parents, diagnostic = _budget_failure_fixture()
        overhead = parents["overhead_cost_probe_report"]
        selectable = copy.deepcopy(report)
        selectable["diagnostic_checkpoint_commitment"][
            "selectable_checkpoint_created"
        ] = True
        semantic = dict(selectable)
        semantic.pop("report_id")
        selectable["report_id"] = canonical_sha256(semantic)
        with self.assertRaisesRegex(ValueError, "commitment"):
            preflight.validate_preflight_failure_report(
                selectable,
                rows,
                overhead_cost_probe_report=overhead,
                diagnostic_checkpoint_snapshot=diagnostic,
            )

        forged_projection = copy.deepcopy(report)
        projection = forged_projection["budget_projection"]
        projection["projected_campaign_wall_ns"] += 1
        projection_semantic = dict(projection)
        projection_semantic.pop("projection_id")
        projection["projection_id"] = canonical_sha256(projection_semantic)
        report_semantic = dict(forged_projection)
        report_semantic.pop("report_id")
        forged_projection["report_id"] = canonical_sha256(report_semantic)
        with self.assertRaisesRegex(ValueError, "projection"):
            preflight.validate_preflight_failure_report(
                forged_projection,
                rows,
                overhead_cost_probe_report=overhead,
                diagnostic_checkpoint_snapshot=diagnostic,
            )


if __name__ == "__main__":
    unittest.main()
