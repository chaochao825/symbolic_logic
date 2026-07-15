from __future__ import annotations

import copy
import json
import hashlib
import subprocess
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from afts_arc.e01 import (
    _content_id,
    _panel_parse_bundle_row,
    _relation_parse_bundle_row,
    _replay_search_artifacts,
    _search_by_task,
    _validate_panel_parse_rows,
    _validate_relation_parse_rows,
    build_public_training_smoke_bundles,
    build_symbolic_pool_bundle,
    build_synthetic_dataset_bundles,
    evaluate_symbolic_pool_bundle,
    evaluate_public_training_pool_bundle,
    snapshot_bundle,
    verify_eval_bundle,
)
from afts_arc.blind import BlindTask
from afts_arc.authority import ARC2_PUBLIC_TRAINING_AUDIT
from afts_arc.grid import as_grid, grid_key
from afts_arc.panel import parse_panels
from afts_arc.manifest import (
    RuntimeSourceCapture,
    capture_runtime_source,
    publish_evidence_bundle,
    serialize_json,
    serialize_jsonl,
    snapshot_task_directory,
    test_source_fingerprint,
)
from afts_arc.search import SearchConfig
from afts_arc.task import ARCPair


class E01EvidenceTests(unittest.TestCase):
    def _context(self) -> tuple[RuntimeSourceCapture, str | None]:
        project = Path(__file__).resolve().parents[1]
        return capture_runtime_source(), test_source_fingerprint(project)

    def test_three_stage_pipeline_is_blind_closed_world_and_recomputable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            blind = root / "blind"
            oracle = root / "oracle"
            pool = root / "pool"
            blind_repeat = root / "blind-repeat"
            oracle_repeat = root / "oracle-repeat"
            pool_repeat = root / "pool-repeat"
            evaluation = root / "evaluation"
            source_capture, tests_sha = self._context()
            dataset_summary = build_synthetic_dataset_bundles(
                blind_output_dir=blind,
                oracle_output_dir=oracle,
                seed=7,
                tasks_per_family=1,
                source_capture=source_capture,
                test_source_sha256=tests_sha,
                source_loader={"test": True},
                command=["test", "build-dataset"],
            )
            blind_snapshot = snapshot_bundle(blind)
            blind_rows = [
                json.loads(line)
                for line in blind_snapshot.artifacts["blind_tasks.jsonl"].splitlines()
            ]
            self.assertEqual(dataset_summary["task_count"], 17)
            self.assertTrue(all(row["task_id"].startswith("blind_") for row in blind_rows))
            forbidden = {
                "family",
                "stratum",
                "seed",
                "generator_program",
                "source_path",
                "source_sha256",
            }
            self.assertTrue(all(not forbidden.intersection(row) for row in blind_rows))
            self.assertTrue(
                all(
                    set(pair) == {"input"}
                    for row in blind_rows
                    for pair in row["test"]
                )
            )
            repeated_dataset = build_synthetic_dataset_bundles(
                blind_output_dir=blind_repeat,
                oracle_output_dir=oracle_repeat,
                seed=7,
                tasks_per_family=1,
                source_capture=source_capture,
                test_source_sha256=tests_sha,
                source_loader={"test": True},
                command=["test", "build-dataset-repeat"],
            )
            self.assertEqual(
                dataset_summary["case_set_id"], repeated_dataset["case_set_id"]
            )
            self.assertEqual(
                snapshot_bundle(blind_repeat).artifacts["blind_tasks.jsonl"],
                blind_snapshot.artifacts["blind_tasks.jsonl"],
            )

            original_read = Path.read_bytes

            def guarded_read(path: Path) -> bytes:
                resolved = path.resolve()
                if oracle.resolve() in (resolved, *resolved.parents):
                    raise AssertionError("pool attempted to read the oracle bundle")
                return original_read(path)

            with patch.object(Path, "read_bytes", guarded_read):
                pool_summary = build_symbolic_pool_bundle(
                    blind_dataset_dir=blind,
                    output_dir=pool,
                    search_config=SearchConfig(
                        max_depth=1,
                        beam_width=16,
                        max_instruction_options=32,
                        max_exact_programs=32,
                    ),
                    source_capture=source_capture,
                    test_source_sha256=tests_sha,
                    source_loader={"test": True},
                    command=["test", "run-pool"],
                )
            self.assertEqual(pool_summary["task_count"], 17)
            repeated_pool = build_symbolic_pool_bundle(
                blind_dataset_dir=blind_repeat,
                output_dir=pool_repeat,
                search_config=SearchConfig(
                    max_depth=1,
                    beam_width=16,
                    max_instruction_options=32,
                    max_exact_programs=32,
                ),
                source_capture=source_capture,
                test_source_sha256=tests_sha,
                source_loader={"test": True},
                command=["test", "run-pool-repeat"],
            )
            self.assertEqual(pool_summary["pool_spec_id"], repeated_pool["pool_spec_id"])
            self.assertEqual(
                pool_summary["pool_content_id"], repeated_pool["pool_content_id"]
            )
            pool_snapshot = snapshot_bundle(pool)
            repeated_pool_snapshot = snapshot_bundle(pool_repeat)
            search_rows = [
                json.loads(line)
                for line in pool_snapshot.artifacts["search.jsonl"].splitlines()
            ]
            sequence_row = next(
                row
                for row in search_rows
                if row["panel_sequence_d4_instruction_proposals"]
            )
            self.assertGreater(sequence_row["panel_sequence_d4_trial_count"], 0)
            self.assertEqual(
                sequence_row["panel_sequence_d4_demo_execution_count"],
                sequence_row["panel_sequence_d4_trial_count"] * 3,
            )
            self.assertEqual(
                sequence_row["instruction_option_count_post_cap"],
                len(sequence_row["instruction_options"]),
            )
            self.assertEqual(
                sequence_row["instruction_option_truncation_count"],
                sequence_row["instruction_option_count_pre_cap"]
                - sequence_row["instruction_option_count_post_cap"],
            )
            self.assertGreater(
                pool_summary["panel_sequence_d4_instruction_proposals"], 0
            )
            self.assertGreater(pool_summary["panel_sequence_d4_trial_count"], 0)
            self.assertGreater(
                pool_summary["panel_sequence_d4_demo_execution_count"], 0
            )
            periodic_row = next(
                row
                for row in search_rows
                if row["panel_lattice_periodic_instruction_proposals"]
            )
            self.assertTrue(periodic_row["panel_lattice_periodic_period_bounds"])
            self.assertGreater(
                periodic_row["panel_lattice_periodic_structural_check_count"], 0
            )
            self.assertEqual(
                periodic_row["panel_lattice_periodic_trial_count"],
                sum(
                    bound["max_row_period"]
                    * bound["max_column_period"]
                    for bound in periodic_row[
                        "panel_lattice_periodic_period_bounds"
                    ]
                ),
            )
            self.assertEqual(
                periodic_row["panel_lattice_periodic_demo_execution_count"],
                periodic_row["panel_lattice_periodic_trial_count"] * 3,
            )
            self.assertGreater(
                pool_summary["panel_lattice_periodic_period_bounds"], 0
            )
            self.assertGreater(
                pool_summary["panel_lattice_periodic_instruction_proposals"], 0
            )
            self.assertGreater(
                pool_summary["panel_lattice_periodic_options_after_cap"], 0
            )
            contact_row = next(
                row
                for row in search_rows
                if row["bbox_contact_instruction_proposals"]
            )
            self.assertTrue(contact_row["bbox_contact_bounds"])
            self.assertEqual(
                contact_row["bbox_contact_structural_check_count"],
                len(contact_row["bbox_contact_bounds"]) * 3,
            )
            self.assertEqual(
                contact_row["bbox_contact_action_trial_count"],
                contact_row["bbox_contact_admissible_binding_count"],
            )
            self.assertEqual(
                contact_row["bbox_contact_demo_execution_count"],
                contact_row["bbox_contact_action_trial_count"] * 3,
            )
            self.assertGreater(pool_summary["relation_parse_records"], 0)
            self.assertGreater(pool_summary["relation_parse_bundles"], 0)
            self.assertGreater(pool_summary["relation_hypotheses"], 0)
            self.assertGreater(pool_summary["bbox_contact_bounds"], 0)
            self.assertGreater(
                pool_summary["bbox_contact_instruction_proposals"], 0
            )
            self.assertGreater(
                pool_summary["bbox_contact_options_after_cap"], 0
            )

            relation_rows = [
                json.loads(line)
                for line in pool_snapshot.artifacts[
                    "relation_parses.jsonl"
                ].splitlines()
            ]
            self.assertTrue(relation_rows)
            for row in relation_rows:
                self.assertEqual(
                    row["background_candidates"],
                    [bundle["background"] for bundle in row["background_bundles"]],
                )

            wrong_step = copy.deepcopy(search_rows)
            wrong_step_row = next(
                row
                for row in wrong_step
                if row["panel_sequence_d4_instruction_proposals"]
            )
            wrong_step_row["panel_sequence_d4_instruction_proposals"][0][
                "arguments"
            ]["step"] = "rotate45"
            with self.assertRaisesRegex(ValueError, "D4"):
                _search_by_task(wrong_step)

            duplicated = copy.deepcopy(search_rows)
            duplicated_row = next(
                row
                for row in duplicated
                if row["panel_sequence_d4_instruction_proposals"]
            )
            duplicated_row["panel_sequence_d4_instruction_proposals"].append(
                copy.deepcopy(
                    duplicated_row["panel_sequence_d4_instruction_proposals"][0]
                )
            )
            with self.assertRaisesRegex(ValueError, "duplicates"):
                _search_by_task(duplicated)

            duplicated_periodic_bound = copy.deepcopy(search_rows)
            duplicated_periodic_bound_row = next(
                row
                for row in duplicated_periodic_bound
                if row["panel_lattice_periodic_period_bounds"]
            )
            duplicated_periodic_bound_row[
                "panel_lattice_periodic_period_bounds"
            ].append(
                copy.deepcopy(
                    duplicated_periodic_bound_row[
                        "panel_lattice_periodic_period_bounds"
                    ][0]
                )
            )
            with self.assertRaisesRegex(ValueError, "bounds are not canonical"):
                _search_by_task(duplicated_periodic_bound)

            forged_periodic_trial = copy.deepcopy(search_rows)
            forged_periodic_trial_row = next(
                row
                for row in forged_periodic_trial
                if row["panel_lattice_periodic_period_bounds"]
            )
            forged_periodic_trial_row["panel_lattice_periodic_trial_count"] += 1
            with self.assertRaisesRegex(ValueError, "trial ledger"):
                _search_by_task(forged_periodic_trial)

            forged_periodic_demo_cost = copy.deepcopy(search_rows)
            forged_periodic_demo_cost_row = next(
                row
                for row in forged_periodic_demo_cost
                if row["panel_lattice_periodic_trial_count"]
            )
            forged_periodic_demo_cost_row[
                "panel_lattice_periodic_demo_execution_count"
            ] += 1
            with self.assertRaisesRegex(ValueError, "demo-execution ledger"):
                _search_by_task(forged_periodic_demo_cost)

            zero_periodic_demo_cost = copy.deepcopy(search_rows)
            zero_periodic_demo_cost_row = next(
                row
                for row in zero_periodic_demo_cost
                if row["panel_lattice_periodic_trial_count"]
            )
            zero_periodic_demo_cost_row[
                "panel_lattice_periodic_demo_execution_count"
            ] = 0
            with self.assertRaisesRegex(ValueError, "demo-execution ledger"):
                _search_by_task(zero_periodic_demo_cost)

            forged_periodic_proposal = copy.deepcopy(search_rows)
            forged_periodic_proposal_row = next(
                row
                for row in forged_periodic_proposal
                if row["panel_lattice_periodic_instruction_proposals"]
            )
            forged_periodic_proposal_row[
                "panel_lattice_periodic_instruction_proposals"
            ][0]["arguments"]["row_period"] = 31
            with self.assertRaisesRegex(ValueError, "period|Positive"):
                _search_by_task(forged_periodic_proposal)

            forged_cap = copy.deepcopy(search_rows)
            forged_cap[0]["instruction_option_count_post_cap"] += 1
            with self.assertRaisesRegex(ValueError, "post-cap"):
                _search_by_task(forged_cap)
            forged_contact_cost = copy.deepcopy(search_rows)
            forged_contact_cost_row = next(
                row
                for row in forged_contact_cost
                if row["bbox_contact_action_trial_count"]
            )
            forged_contact_cost_row["bbox_contact_demo_execution_count"] += 1
            with self.assertRaisesRegex(ValueError, "bbox-contact demo-execution"):
                _search_by_task(forged_contact_cost)
            for artifact in (
                "tasks.jsonl",
                "parses.jsonl",
                "panel_parses.jsonl",
                "relation_parses.jsonl",
                "exact_programs.jsonl",
                "candidates.jsonl",
            ):
                self.assertEqual(
                    pool_snapshot.artifacts[artifact],
                    repeated_pool_snapshot.artifacts[artifact],
                )
            forged_pool_artifacts = dict(pool_snapshot.artifacts)
            forged_pool_summary = json.loads(forged_pool_artifacts["pool_summary.json"])
            forged_pool_summary["candidate_records"] = 999999
            forged_pool_artifacts["pool_summary.json"] = serialize_json(
                forged_pool_summary
            )
            forged_pool = root / "forged-pool-summary"
            publish_evidence_bundle(
                forged_pool,
                run_id="forged-pool-summary",
                artifacts=forged_pool_artifacts,
            )
            with self.assertRaisesRegex(ValueError, "summary"):
                evaluate_symbolic_pool_bundle(
                    blind_dataset_dir=blind,
                    oracle_dataset_dir=oracle,
                    pool_dir=forged_pool,
                    output_dir=root / "forged-pool-evaluation",
                    source_capture=source_capture,
                    test_source_sha256=tests_sha,
                    source_loader={"test": True},
                    command=["test", "reject-forged-pool-summary"],
                )
            forged_cost_artifacts = dict(pool_snapshot.artifacts)
            forged_search_rows = [
                json.loads(line)
                for line in forged_cost_artifacts["search.jsonl"].splitlines()
            ]
            forged_search_rows[0]["wall_time_ns"] = -7
            forged_cost_artifacts["search.jsonl"] = serialize_jsonl(
                forged_search_rows
            )
            forged_cost_pool = root / "forged-pool-cost"
            publish_evidence_bundle(
                forged_cost_pool,
                run_id="forged-pool-cost",
                artifacts=forged_cost_artifacts,
            )
            with self.assertRaisesRegex(ValueError, "non-negative int"):
                evaluate_symbolic_pool_bundle(
                    blind_dataset_dir=blind,
                    oracle_dataset_dir=oracle,
                    pool_dir=forged_cost_pool,
                    output_dir=root / "forged-cost-evaluation",
                    source_capture=source_capture,
                    test_source_sha256=tests_sha,
                    source_loader={"test": True},
                    command=["test", "reject-forged-cost"],
                )
            forged_shape_artifacts = dict(pool_snapshot.artifacts)
            forged_shape_rows = [
                json.loads(line)
                for line in forged_shape_artifacts["search.jsonl"].splitlines()
            ]
            shape_row = next(row for row in forged_shape_rows if row["shape_proposals"])
            proposal = shape_row["shape_proposals"][0]
            proposal["row_factor"] = 2 if proposal["row_factor"] == 1 else 1
            forged_shape_artifacts["search.jsonl"] = serialize_jsonl(
                forged_shape_rows
            )
            forged_shape_pool = root / "forged-shape-proposal"
            publish_evidence_bundle(
                forged_shape_pool,
                run_id="forged-shape-proposal",
                artifacts=forged_shape_artifacts,
            )
            with self.assertRaisesRegex(ValueError, "pool_content_id|proposal_id"):
                evaluate_symbolic_pool_bundle(
                    blind_dataset_dir=blind,
                    oracle_dataset_dir=oracle,
                    pool_dir=forged_shape_pool,
                    output_dir=root / "forged-shape-evaluation",
                    source_capture=source_capture,
                    test_source_sha256=tests_sha,
                    source_loader={"test": True},
                    command=["test", "reject-forged-shape"],
                )
            forged_sequence_artifacts = dict(pool_snapshot.artifacts)
            forged_sequence_rows = copy.deepcopy(search_rows)
            forged_sequence_row = next(
                row
                for row in forged_sequence_rows
                if row["panel_sequence_d4_instruction_proposals"]
            )
            forged_sequence_row["panel_sequence_d4_instruction_proposals"].pop()
            forged_sequence_artifacts["search.jsonl"] = serialize_jsonl(
                forged_sequence_rows
            )
            forged_sequence_pool = root / "forged-panel-sequence-proposal"
            publish_evidence_bundle(
                forged_sequence_pool,
                run_id="forged-panel-sequence-proposal",
                artifacts=forged_sequence_artifacts,
            )
            with self.assertRaisesRegex(ValueError, "pool_content_id"):
                evaluate_symbolic_pool_bundle(
                    blind_dataset_dir=blind,
                    oracle_dataset_dir=oracle,
                    pool_dir=forged_sequence_pool,
                    output_dir=root / "forged-panel-sequence-evaluation",
                    source_capture=source_capture,
                    test_source_sha256=tests_sha,
                    source_loader={"test": True},
                    command=["test", "reject-forged-panel-sequence"],
                )
            forged_periodic_artifacts = dict(pool_snapshot.artifacts)
            forged_periodic_rows = copy.deepcopy(search_rows)
            forged_periodic_row = next(
                row
                for row in forged_periodic_rows
                if row["panel_lattice_periodic_instruction_proposals"]
            )
            forged_periodic_row[
                "panel_lattice_periodic_instruction_proposals"
            ].pop()
            forged_periodic_artifacts["search.jsonl"] = serialize_jsonl(
                forged_periodic_rows
            )
            forged_periodic_pool = root / "forged-panel-periodic-proposal"
            publish_evidence_bundle(
                forged_periodic_pool,
                run_id="forged-panel-periodic-proposal",
                artifacts=forged_periodic_artifacts,
            )
            with self.assertRaisesRegex(ValueError, "pool_content_id"):
                evaluate_symbolic_pool_bundle(
                    blind_dataset_dir=blind,
                    oracle_dataset_dir=oracle,
                    pool_dir=forged_periodic_pool,
                    output_dir=root / "forged-panel-periodic-evaluation",
                    source_capture=source_capture,
                    test_source_sha256=tests_sha,
                    source_loader={"test": True},
                    command=["test", "reject-forged-panel-periodic"],
                )
            replay_rows = {
                name: [json.loads(line) for line in content.splitlines()]
                for name, content in pool_snapshot.artifacts.items()
                if name
                in {
                    "tasks.jsonl",
                    "parses.jsonl",
                    "panel_parses.jsonl",
                    "relation_parses.jsonl",
                    "exact_programs.jsonl",
                    "candidates.jsonl",
                    "search.jsonl",
                }
            }
            replay_rows["search.jsonl"] = forged_periodic_rows
            replay_tasks = {
                task.task_id: task
                for task in (
                    BlindTask.from_json_dict(row)
                    for row in replay_rows["tasks.jsonl"]
                )
            }
            with self.assertRaisesRegex(ValueError, "search ledger"):
                _replay_search_artifacts(
                    tasks=replay_tasks,
                    pool_rows=replay_rows,
                    pool_config=json.loads(pool_snapshot.artifacts["config.json"]),
                )
            future_periodic_config = json.loads(
                pool_snapshot.artifacts["config.json"]
            )
            future_periodic_config[
                "panel_lattice_periodic_semantics_version"
            ] = "future"
            with self.assertRaisesRegex(ValueError, "pool config"):
                _replay_search_artifacts(
                    tasks=replay_tasks,
                    pool_rows=replay_rows,
                    pool_config=future_periodic_config,
                )
            future_relation_config = json.loads(
                pool_snapshot.artifacts["config.json"]
            )
            future_relation_config[
                "relation_parser_semantics_version"
            ] = "future"
            with self.assertRaisesRegex(ValueError, "pool config"):
                _replay_search_artifacts(
                    tasks=replay_tasks,
                    pool_rows=replay_rows,
                    pool_config=future_relation_config,
                )
            future_contact_config = json.loads(
                pool_snapshot.artifacts["config.json"]
            )
            future_contact_config[
                "bbox_contact_proposer_semantics_version"
            ] = "future"
            with self.assertRaisesRegex(ValueError, "pool config"):
                _replay_search_artifacts(
                    tasks=replay_tasks,
                    pool_rows=replay_rows,
                    pool_config=future_contact_config,
                )
            forged_panel_artifacts = dict(pool_snapshot.artifacts)
            forged_panel_rows = [
                json.loads(line)
                for line in forged_panel_artifacts["panel_parses.jsonl"].splitlines()
            ]
            panel_row = next(row for row in forged_panel_rows if row["hypotheses"])
            panel_row["panel_parse_bundle_id"] = "0" * 64
            forged_panel_artifacts["panel_parses.jsonl"] = serialize_jsonl(
                forged_panel_rows
            )
            forged_panel_pool = root / "forged-panel-parse"
            publish_evidence_bundle(
                forged_panel_pool,
                run_id="forged-panel-parse",
                artifacts=forged_panel_artifacts,
            )
            with self.assertRaisesRegex(ValueError, "pool_content_id|panel"):
                evaluate_symbolic_pool_bundle(
                    blind_dataset_dir=blind,
                    oracle_dataset_dir=oracle,
                    pool_dir=forged_panel_pool,
                    output_dir=root / "forged-panel-evaluation",
                    source_capture=source_capture,
                    test_source_sha256=tests_sha,
                    source_loader={"test": True},
                    command=["test", "reject-forged-panel"],
                )

            forged_relation_rows = copy.deepcopy(relation_rows)
            forged_relation_rows[0]["relation_parse_bundle_id"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "relation parse"):
                _validate_relation_parse_rows(replay_tasks, forged_relation_rows)
            forged_relation_artifacts = dict(pool_snapshot.artifacts)
            forged_relation_artifacts["relation_parses.jsonl"] = serialize_jsonl(
                forged_relation_rows
            )
            forged_relation_pool = root / "forged-relation-parse"
            publish_evidence_bundle(
                forged_relation_pool,
                run_id="forged-relation-parse",
                artifacts=forged_relation_artifacts,
            )
            with self.assertRaisesRegex(ValueError, "pool_content_id|relation"):
                evaluate_symbolic_pool_bundle(
                    blind_dataset_dir=blind,
                    oracle_dataset_dir=oracle,
                    pool_dir=forged_relation_pool,
                    output_dir=root / "forged-relation-evaluation",
                    source_capture=source_capture,
                    test_source_sha256=tests_sha,
                    source_loader={"test": True},
                    command=["test", "reject-forged-relation"],
                )

            missing_panel_artifacts = dict(pool_snapshot.artifacts)
            missing_panel_artifacts.pop("panel_parses.jsonl")
            missing_panel_pool = root / "missing-panel-sidecar"
            publish_evidence_bundle(
                missing_panel_pool,
                run_id="missing-panel-sidecar",
                artifacts=missing_panel_artifacts,
            )
            with self.assertRaisesRegex(ValueError, "artifacts"):
                evaluate_symbolic_pool_bundle(
                    blind_dataset_dir=blind,
                    oracle_dataset_dir=oracle,
                    pool_dir=missing_panel_pool,
                    output_dir=root / "missing-panel-evaluation",
                    source_capture=source_capture,
                    test_source_sha256=tests_sha,
                    source_loader={"test": True},
                    command=["test", "reject-missing-panel"],
                )
            missing_relation_artifacts = dict(pool_snapshot.artifacts)
            missing_relation_artifacts.pop("relation_parses.jsonl")
            missing_relation_pool = root / "missing-relation-sidecar"
            publish_evidence_bundle(
                missing_relation_pool,
                run_id="missing-relation-sidecar",
                artifacts=missing_relation_artifacts,
            )
            with self.assertRaisesRegex(ValueError, "artifacts"):
                evaluate_symbolic_pool_bundle(
                    blind_dataset_dir=blind,
                    oracle_dataset_dir=oracle,
                    pool_dir=missing_relation_pool,
                    output_dir=root / "missing-relation-evaluation",
                    source_capture=source_capture,
                    test_source_sha256=tests_sha,
                    source_loader={"test": True},
                    command=["test", "reject-missing-relation"],
                )
            for name, content in pool_snapshot.artifacts.items():
                if name.endswith((".json", ".jsonl")):
                    lowered = content.lower()
                    self.assertNotIn(b'"generator_program"', lowered)
                    self.assertNotIn(b'"family"', lowered)
                    self.assertNotIn(b'"seed"', lowered)

            oracle_snapshot = snapshot_bundle(oracle)
            forged_rows = [
                json.loads(line)
                for line in oracle_snapshot.artifacts["oracle.jsonl"].splitlines()
            ]
            original_output = as_grid(forged_rows[0]["test"][0]["output"])
            forged_output = tuple(
                tuple((cell + 1) % 10 for cell in row) for row in original_output
            )
            forged_rows[0]["test"][0]["output"] = [list(row) for row in forged_output]
            forged_rows[0]["test"][0]["output_key"] = grid_key(forged_output)
            forged_oracle_bytes = serialize_jsonl(forged_rows)
            forged_manifest = json.loads(
                oracle_snapshot.artifacts["oracle_manifest.json"]
            )
            forged_manifest["oracle_sha256"] = hashlib.sha256(forged_oracle_bytes).hexdigest()
            forged_manifest["oracle_set_id"] = _content_id(
                "afts-e01a-oracle-set/v1",
                {
                    "case_set_id": forged_manifest["case_set_id"],
                    "oracle_sha256": forged_manifest["oracle_sha256"],
                },
            )
            forged_oracle = root / "forged-oracle"
            publish_evidence_bundle(
                forged_oracle,
                run_id=forged_manifest["oracle_set_id"][:20],
                artifacts={
                    "oracle_manifest.json": serialize_json(forged_manifest),
                    "oracle.jsonl": forged_oracle_bytes,
                    "source_snapshot.zip": oracle_snapshot.artifacts[
                        "source_snapshot.zip"
                    ],
                },
            )
            with self.assertRaisesRegex(ValueError, "do not reproduce"):
                evaluate_symbolic_pool_bundle(
                    blind_dataset_dir=blind,
                    oracle_dataset_dir=forged_oracle,
                    pool_dir=pool,
                    output_dir=root / "forged-evaluation",
                    source_capture=source_capture,
                    test_source_sha256=tests_sha,
                    source_loader={"test": True},
                    command=["test", "evaluate-forged"],
                )

            summary = evaluate_symbolic_pool_bundle(
                blind_dataset_dir=blind,
                oracle_dataset_dir=oracle,
                pool_dir=pool,
                output_dir=evaluation,
                source_capture=source_capture,
                test_source_sha256=tests_sha,
                source_loader={"test": True},
                command=["test", "evaluate"],
            )
            metrics = summary["metrics"]
            self.assertIsInstance(metrics, dict)
            self.assertEqual(metrics["generator_replay_control_rate"], 1.0)
            integrity = json.loads(
                snapshot_bundle(evaluation).artifacts["integrity_report.json"]
            )
            self.assertTrue(integrity["relation_parse_sidecar_replayed"])
            self.assertTrue(integrity["bbox_contact_bounds_replayed"])
            self.assertTrue(integrity["bbox_contact_proposals_replayed"])
            self.assertTrue(integrity["bbox_contact_cost_ledger_replayed"])
            self.assertTrue(integrity["bbox_contact_cap_survival_replayed"])
            report = verify_eval_bundle(
                evaluation,
                blind_dataset_dir=blind,
                oracle_dataset_dir=oracle,
                pool_dir=pool,
                source_capture=source_capture,
                test_source_sha256=tests_sha,
            )
            self.assertEqual(
                report["verification_status"], "full_parent_replay_pass"
            )

    def test_closed_world_bundle_rejects_unmanifested_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_capture, tests_sha = self._context()
            blind = root / "blind"
            oracle = root / "oracle"
            build_synthetic_dataset_bundles(
                blind_output_dir=blind,
                oracle_output_dir=oracle,
                seed=0,
                tasks_per_family=1,
                source_capture=source_capture,
                test_source_sha256=tests_sha,
                source_loader={"test": True},
                command=["test", "build-dataset"],
            )
            valid_snapshot = snapshot_bundle(blind)
            corrupted_blind = root / "corrupted-blind"
            publish_evidence_bundle(
                corrupted_blind,
                run_id="corrupted-source",
                artifacts={
                    "blind_manifest.json": valid_snapshot.artifacts[
                        "blind_manifest.json"
                    ],
                    "blind_tasks.jsonl": valid_snapshot.artifacts["blind_tasks.jsonl"],
                    "source_snapshot.zip": b"not-a-zip",
                },
            )
            with self.assertRaisesRegex(ValueError, "source snapshot"):
                build_symbolic_pool_bundle(
                    blind_dataset_dir=corrupted_blind,
                    output_dir=root / "corrupted-pool",
                    search_config=SearchConfig(max_depth=1),
                    source_capture=source_capture,
                    test_source_sha256=tests_sha,
                    source_loader={"test": True},
                    command=["test", "reject-corrupted-source"],
                )
            (blind / "poison-oracle.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "closed-world"):
                snapshot_bundle(blind)

    def test_dataset_and_stage_outputs_may_not_overlap_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_capture, tests_sha = self._context()
            with self.assertRaisesRegex(ValueError, "overlap"):
                build_synthetic_dataset_bundles(
                    blind_output_dir=root / "blind",
                    oracle_output_dir=root / "blind" / "oracle",
                    seed=0,
                    tasks_per_family=1,
                    source_capture=source_capture,
                    test_source_sha256=tests_sha,
                    source_loader={"test": True},
                    command=["test", "overlap"],
                )

    def test_panel_parse_sidecar_is_complete_ordered_and_executable(self) -> None:
        scene = as_grid(
            [
                [1, 0, 9, 3, 4],
                [0, 2, 9, 0, 0],
                [9, 9, 9, 9, 9],
                [5, 0, 9, 7, 8],
                [6, 0, 9, 0, 0],
            ]
        )
        target = as_grid([[1, 4], [6, 2]])
        query = as_grid([[1, 2], [3, 4]])
        task = BlindTask.from_observations(
            train=(ARCPair(scene, target),),
            test_inputs=(query,),
        )
        tasks = {task.task_id: task}
        rows = [
            _panel_parse_bundle_row(
                task=task,
                pair_role="train_input",
                pair_index=0,
                bundle=parse_panels(scene),
            ),
            _panel_parse_bundle_row(
                task=task,
                pair_role="train_output",
                pair_index=0,
                bundle=parse_panels(target),
            ),
            _panel_parse_bundle_row(
                task=task,
                pair_role="test_input",
                pair_index=0,
                bundle=parse_panels(query),
            ),
        ]
        _validate_panel_parse_rows(tasks, rows)
        self.assertTrue(rows[0]["hypotheses"])
        self.assertEqual(rows[1]["hypotheses"], [])
        self.assertEqual(rows[2]["hypotheses"], [])

        with self.assertRaisesRegex(ValueError, "canonical order"):
            _validate_panel_parse_rows(tasks, [rows[1], rows[0], rows[2]])
        with self.assertRaisesRegex(ValueError, "cover every"):
            _validate_panel_parse_rows(tasks, rows[:-1])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            _validate_panel_parse_rows(tasks, [rows[0], rows[0], rows[2]])

        wrong_semantics = json.loads(json.dumps(rows))
        wrong_semantics[0]["panel_parser_semantics_version"] = "future"
        with self.assertRaisesRegex(ValueError, "semantics"):
            _validate_panel_parse_rows(tasks, wrong_semantics)

        other_scene = tuple(tuple((cell + 1) % 10 for cell in row) for row in scene)
        replay_forgery = list(rows)
        replay_forgery[0] = _panel_parse_bundle_row(
            task=task,
            pair_role="train_input",
            pair_index=0,
            bundle=parse_panels(other_scene),
        )
        with self.assertRaisesRegex(ValueError, "replay"):
            _validate_panel_parse_rows(tasks, replay_forgery)

    def test_relation_parse_sidecar_is_complete_ordered_and_replayable(self) -> None:
        scene = as_grid(
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [2, 0, 7, 7, 7, 0, 0],
                [0, 0, 7, 7, 7, 0, 3],
                [0, 0, 0, 0, 0, 0, 0],
            ]
        )
        target = as_grid(
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [2, 0, 2, 7, 7, 0, 0],
                [0, 0, 7, 7, 3, 0, 3],
                [0, 0, 0, 0, 0, 0, 0],
            ]
        )
        query = as_grid(
            [
                [0, 4, 0, 0, 0],
                [0, 0, 8, 8, 0],
                [0, 0, 8, 8, 0],
                [0, 0, 0, 0, 0],
            ]
        )
        task = BlindTask.from_observations(
            train=(ARCPair(scene, target),), test_inputs=(query,)
        )
        tasks = {task.task_id: task}
        rows = [
            _relation_parse_bundle_row(
                task=task,
                pair_role="train_input",
                pair_index=0,
                grid=scene,
            ),
            _relation_parse_bundle_row(
                task=task,
                pair_role="train_output",
                pair_index=0,
                grid=target,
            ),
            _relation_parse_bundle_row(
                task=task,
                pair_role="test_input",
                pair_index=0,
                grid=query,
            ),
        ]
        _validate_relation_parse_rows(tasks, rows)
        self.assertTrue(rows[0]["background_bundles"])
        self.assertEqual(
            rows[0]["background_candidates"],
            [bundle["background"] for bundle in rows[0]["background_bundles"]],
        )

        with self.assertRaisesRegex(ValueError, "canonical order"):
            _validate_relation_parse_rows(tasks, [rows[1], rows[0], rows[2]])
        with self.assertRaisesRegex(ValueError, "cover every"):
            _validate_relation_parse_rows(tasks, rows[:-1])
        tampered = copy.deepcopy(rows)
        tampered[0]["relation_parser_semantics_version"] = "future"
        with self.assertRaisesRegex(ValueError, "semantics"):
            _validate_relation_parse_rows(tasks, tampered)
        tampered = copy.deepcopy(rows)
        tampered[0]["background_bundles"][0]["relation_parse_bundle_id"] = (
            "0" * 64
        )
        with self.assertRaises((TypeError, ValueError)):
            _validate_relation_parse_rows(tasks, tampered)

    def test_public_training_smoke_uses_same_blind_pool_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = root / "ARC-AGI-2"
            training = repository / "data" / "training"
            training.mkdir(parents=True)

            def git(*args: str) -> str:
                return subprocess.run(
                    ["git", *args],
                    cwd=repository,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()

            git("init")
            git("config", "user.email", "e01@example.invalid")
            git("config", "user.name", "E01 Tests")
            git("remote", "add", "origin", "https://github.com/arcprize/ARC-AGI-2.git")
            for index in range(4):
                task = {
                    "train": [
                        {"input": [[index]], "output": [[(index + 1) % 10]]}
                    ],
                    "test": [
                        {"input": [[index + 1]], "output": [[(index + 2) % 10]]}
                    ],
                }
                (training / f"task-{index}.json").write_text(
                    json.dumps(task), encoding="utf-8"
                )
            git("add", ".")
            git("commit", "-m", "fixture")
            commit = git("rev-parse", "HEAD")
            snapshot = snapshot_task_directory(
                training,
                dataset_name="ARC-AGI-2",
                split_name="public_training",
                repo_dir=repository,
                expected_origin="https://github.com/arcprize/ARC-AGI-2.git",
                expected_commit=commit,
            )
            blind = root / "public-blind"
            oracle = root / "public-oracle"
            pool = root / "public-pool"
            evaluation = root / "public-eval"
            source_capture, tests_sha = self._context()
            fixture_audit_pin = {
                key: asdict(snapshot.audit)[key] for key in ARC2_PUBLIC_TRAINING_AUDIT
            }
            with self.assertRaisesRegex(ValueError, "complete clean pinned"):
                build_public_training_smoke_bundles(
                    dataset_snapshot=snapshot,
                    blind_output_dir=blind,
                    oracle_output_dir=oracle,
                    smoke_count=2,
                    selection_salt="unit-public-smoke",
                    holdout_count=1,
                    validation_count=1,
                    source_capture=source_capture,
                    test_source_sha256=tests_sha,
                    source_loader={"test": True},
                    command=["test", "reject-unpinned-public"],
                )
            with patch.dict(
                "afts_arc.e01.DATASET_COMMITS", {"ARC-AGI-2": commit}
            ), patch.dict(
                "afts_arc.e01.ARC2_PUBLIC_TRAINING_AUDIT",
                fixture_audit_pin,
                clear=True,
            ):
                build_public_training_smoke_bundles(
                    dataset_snapshot=snapshot,
                    blind_output_dir=blind,
                    oracle_output_dir=oracle,
                    smoke_count=2,
                    selection_salt="unit-public-smoke",
                    holdout_count=1,
                    validation_count=1,
                    source_capture=source_capture,
                    test_source_sha256=tests_sha,
                    source_loader={"test": True},
                    command=["test", "build-public"],
                )
            build_symbolic_pool_bundle(
                blind_dataset_dir=blind,
                output_dir=pool,
                search_config=SearchConfig(max_depth=1, max_instruction_options=16),
                source_capture=source_capture,
                test_source_sha256=tests_sha,
                source_loader={"test": True},
                command=["test", "pool-public"],
            )
            with patch.dict(
                "afts_arc.e01.DATASET_COMMITS", {"ARC-AGI-2": commit}
            ), patch.dict(
                "afts_arc.e01.ARC2_PUBLIC_TRAINING_AUDIT",
                fixture_audit_pin,
                clear=True,
            ):
                summary = evaluate_public_training_pool_bundle(
                    blind_dataset_dir=blind,
                    oracle_dataset_dir=oracle,
                    pool_dir=pool,
                    output_dir=evaluation,
                    source_capture=source_capture,
                    test_source_sha256=tests_sha,
                    source_loader={"test": True},
                    command=["test", "eval-public"],
                )
            self.assertEqual(
                summary["dataset_kind"], "arc_public_training_development_smoke"
            )
            with patch.dict(
                "afts_arc.e01.DATASET_COMMITS", {"ARC-AGI-2": commit}
            ), patch.dict(
                "afts_arc.e01.ARC2_PUBLIC_TRAINING_AUDIT",
                fixture_audit_pin,
                clear=True,
            ):
                verification_status = verify_eval_bundle(
                    evaluation,
                    blind_dataset_dir=blind,
                    oracle_dataset_dir=oracle,
                    pool_dir=pool,
                    source_capture=source_capture,
                    test_source_sha256=tests_sha,
                )["verification_status"]
            self.assertEqual(verification_status, "full_parent_replay_pass")


if __name__ == "__main__":
    unittest.main()
