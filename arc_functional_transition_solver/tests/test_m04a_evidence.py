from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import afts_arc.m04a_evidence as m04a_evidence
from afts_arc.grid import grid_key
from afts_arc.m04a_contract import (
    LANE_TRACE_SCHEMA_VERSION,
    canonical_sha256,
    float64_sequence_sha256,
    lane_seed,
    mask_count_trace,
)
from afts_arc.m04a_evidence import (
    EVALUATION_LINEAGE_FIELDS,
    FIXED_BLIND_TEST_PAIR_COUNT,
    NO_SHAPE_PROPOSAL,
    POOL_ARTIFACT_FILES,
    POOL_LINEAGE_FIELDS,
    SHAPE_READY,
    TRAINING_ARTIFACT_FILES,
    TRAINING_LINEAGE_FIELDS,
    assert_oracle_free_payload,
    build_m04a_blind_input_sidecar,
    build_pool_artifact_manifest,
    build_training_artifact_manifest,
    expected_pair_cost_payload,
    lane_trace_sha256,
    make_decoder_forward_row,
    make_encoder_forward_row,
    make_lane_row,
    make_pair_cost_row,
    make_pool_setup_cost_row,
    make_training_cost_ledger_row,
    publish_pool_artifact_bundle,
    publish_evaluation_artifact_bundle,
    publish_training_artifact_bundle,
    read_closed_world_bundle,
    read_m04a_artifact_bundle,
    select_blind_shapes,
    validate_lane_row,
    validate_lane_trace,
    validate_lane_trace_replay,
    validate_pool_evidence,
    validate_pool_replay_receipt,
    validate_training_cost_ledger,
    validate_training_cost_ledger_row,
    validate_blind_input_shape_sidecar,
)
from afts_arc.manifest import serialize_json, serialize_jsonl
from afts_arc.shape import OutputShapeProposal, ShapeBasis


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BLIND_BUNDLE = (
    PROJECT_ROOT / "results" / "e01a_bbox_contact_v1" / "public_train_dev_blind"
)
DSL_POOL = PROJECT_ROOT / "results" / "e01a_bbox_contact_v1" / "public_train_dev_pool"
PUBLICATION_MANIFEST = PROJECT_ROOT / "PUBLICATION_MANIFEST.json"


def _is_manifest_only_publication_path(path: Path) -> bool:
    if not PUBLICATION_MANIFEST.is_file():
        return False
    relative = path.relative_to(PROJECT_ROOT).as_posix()
    manifest = json.loads(PUBLICATION_MANIFEST.read_text(encoding="utf-8"))
    return any(
        entry.get("path") == relative and entry.get("disposition") == "manifest_only"
        for entry in manifest.get("entries", ())
    )


def _fixed_blind_sidecar():
    source_snapshot = BLIND_BUNDLE / "source_snapshot.zip"
    if not source_snapshot.is_file() and _is_manifest_only_publication_path(
        source_snapshot
    ):
        raise unittest.SkipTest(
            "closed-world blind-bundle payload is manifest-only in this publication"
        )
    return build_m04a_blind_input_sidecar(BLIND_BUNDLE, DSL_POOL)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _lineage(fields: frozenset[str]) -> dict[str, str]:
    return {field: _digest(field) for field in fields}


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
        "primary_optimizer_updates": m04a_evidence.OPTIMIZER_UPDATES,
        "primary_microbatches": (
            m04a_evidence.OPTIMIZER_UPDATES
            * m04a_evidence.PREFLIGHT_MICROBATCHES_PER_UPDATE
        ),
        "primary_arc2_episodes": (
            m04a_evidence.OPTIMIZER_UPDATES
            * m04a_evidence.PREFLIGHT_MICROBATCHES_PER_UPDATE
            // 2
        ),
        "primary_rearc_episodes": (
            m04a_evidence.OPTIMIZER_UPDATES
            * m04a_evidence.PREFLIGHT_MICROBATCHES_PER_UPDATE
            // 2
        ),
        "primary_encoder_forward_calls": (
            m04a_evidence.OPTIMIZER_UPDATES
            * m04a_evidence.PREFLIGHT_MICROBATCHES_PER_UPDATE
        ),
        "primary_decoder_forward_calls": (
            m04a_evidence.OPTIMIZER_UPDATES
            * m04a_evidence.PREFLIGHT_MICROBATCHES_PER_UPDATE
        ),
        "primary_backward_calls": (
            m04a_evidence.OPTIMIZER_UPDATES
            * m04a_evidence.PREFLIGHT_MICROBATCHES_PER_UPDATE
        ),
        "validation_passes": m04a_evidence.VALIDATION_PASS_COUNT,
        "validation_episode_calls": 100,
        "validation_encoder_forward_calls": 100,
        "validation_decoder_forward_calls": 100,
        "checkpoint_writes": m04a_evidence.VALIDATION_PASS_COUNT,
        "resume_segments": 0,
        "masked_token_predictions": 1,
        "selected_checkpoint_complete": True,
        "budget_status": "WITHIN_BUDGET",
    }


def _training_timing(primary_ns: int) -> dict[str, int]:
    return {
        "preflight_wall_time_ns": 1,
        "primary_training_wall_time_ns": primary_ns,
        "validation_wall_time_ns": 2,
        "checkpoint_wall_time_ns": 3,
        "resume_setup_wall_time_ns": 0,
        "gpu_lock_wall_time_ns": primary_ns + 6,
        "budget_limit_ns": 86_400_000_000_000,
        "budget_remaining_ns": 86_400_000_000_000 - primary_ns - 6,
    }


def _pool_closure() -> dict[str, object]:
    return {
        "status": "COMPLETE",
        "task_count": 20,
        "test_pair_count": 21,
        "candidate_count_zero_task_count": 12,
        "shape_supported_test_pair_count": 16,
        "no_shape_test_pair_count": 5,
        "raw_lanes": 1_024,
        "lane_trace_rows": 1_024,
        "candidate_rows": 1_024,
        "encoder_batch_calls": 16,
        "decoder_batch_calls": 1_536,
        "total_actual_batch_calls": 1_552,
        "sample_equivalent_forward_calls": 12_288,
        "masked_token_predictions": 1_024,
        "format_valid": 1_024,
        "format_invalid": 0,
        "unique_outputs": 1_024,
        "duplicate_outputs": 0,
        "pair_cost_rows": 21,
        "replay_verified_trace_count": 1_024,
        "replay_verified_decoder_call_count": 1_536,
        "replay_verified_prediction_count": 1_024,
        "bf16_cache_pair_count": 16,
    }


def _pool_timing() -> dict[str, int]:
    return {
        "setup_wall_time_ns": 1,
        "sequential_pair_wall_time_ns": 2,
        "finalization_wall_time_ns": 3,
        "pool_wall_time_ns": 6,
        "encoder_gpu_ns": 4,
        "decoder_gpu_ns": 5,
        "total_gpu_ns": 9,
    }


def _artifact_content(name: str) -> bytes:
    if name.endswith(".jsonl"):
        return b'{"fixture":true}\n'
    if name.endswith(".json"):
        return b'{"fixture":true}\n'
    return f"fixture:{name}\n".encode("utf-8")


def _file_rows(names: frozenset[str]) -> list[dict[str, object]]:
    return [
        {
            "path": name,
            "sha256": hashlib.sha256(_artifact_content(name)).hexdigest(),
            "bytes": len(_artifact_content(name)),
        }
        for name in sorted(names)
    ]


def _pool_environment_manifest(
    *, runtime_lock_sha256: str, runtime_lock_id: str
) -> dict[str, object]:
    launcher_sha256 = _digest("pool-launcher")
    conda_sha256 = _digest("pool-conda")
    return {
        "schema": "afts-m04a-environment/v0.3",
        "python": "3.10.20 (fixture)",
        "python_implementation": "CPython",
        "python_version": "3.10.20",
        "python_executable": "/opt/mixbit/bin/python",
        "python_executable_sha256": _digest("pool-python"),
        "python_executable_bytes": 17_460_208,
        "platform": "Linux-fixture",
        "hostname": "fixture-host",
        "torch": "2.10.0+cu128",
        "torch_cuda": "12.8",
        "cudnn": 91_002,
        "nvidia_driver": "fixture-driver",
        "cuda_visible_devices": "GPU-00000000-0000-0000-0000-000000000001",
        "gpu_uuid": "GPU-00000000-0000-0000-0000-000000000001",
        "run_id": "pool-fixture",
        "cublas_workspace_config": ":4096:8",
        "gpu": {"name": "fixture-gpu", "total_memory": 1, "major": 9, "minor": 0},
        "deterministic_algorithms": True,
        "cudnn_benchmark": False,
        "tf32_matmul": False,
        "tf32_cudnn": False,
        "flash_sdp": False,
        "mem_efficient_sdp": False,
        "math_sdp": True,
        "launcher_path": "/visible/remote_launcher.py",
        "launcher_sha256": launcher_sha256,
        "ordered_sys_path": ["/stdlib", "/proc/self/fd/200"],
        "ordered_import_roots": ["/srv/afts/project/src"],
        "conda_explicit_path": "/visible/conda-explicit.txt",
        "conda_explicit_sha256": conda_sha256,
        "visible_root": "/visible",
        "runtime_source_sha256": _digest("runtime-source"),
        "test_source_sha256": _digest("test-source"),
        "visible_files": {
            "conda-explicit.txt": conda_sha256,
            "python-runtime-lock.json": runtime_lock_sha256,
            "remote_launcher.py": launcher_sha256,
        },
        "python_runtime_lock_sha256": runtime_lock_sha256,
        "python_runtime_lock_id": runtime_lock_id,
    }


def _fixture_sidecar(rows: tuple[dict[str, object], ...]):
    fixed = _fixed_blind_sidecar()
    manifest = copy.deepcopy(fixed.manifest)
    task_ids = {row["blind_task_id"] for row in rows}
    zero_task_ids = {
        row["blind_task_id"] for row in rows if row["candidate_count_zero_subgroup"]
    }
    manifest.update(
        {
            "task_count": len(task_ids),
            "test_pair_count": len(rows),
            "candidate_count_zero_task_count": len(zero_task_ids),
            "candidate_count_zero_test_pair_count": sum(
                bool(row["candidate_count_zero_subgroup"]) for row in rows
            ),
            "shape_supported_test_pair_count": sum(
                int(row["accepted_shape_count"]) > 0 for row in rows
            ),
            "no_shape_test_pair_count": sum(
                int(row["accepted_shape_count"]) == 0 for row in rows
            ),
            "rows_sha256": hashlib.sha256(serialize_jsonl(rows)).hexdigest(),
        }
    )
    manifest.pop("sidecar_id")
    manifest["sidecar_id"] = canonical_sha256(manifest)
    return validate_blind_input_shape_sidecar(manifest, rows, fixture_mode=True)


def _proposal(
    *, row_factor: int, column_factor: int, query_shapes: tuple[tuple[int, int], ...]
) -> dict[str, object]:
    return OutputShapeProposal.create(
        basis=ShapeBasis.INPUT,
        row_factor=row_factor,
        column_factor=column_factor,
        query_shapes=query_shapes,
    ).to_json_dict()


def _one_cell_trace(
    *,
    greedy: bool = True,
    task_id: str = "blind_trace_fixture",
    proposal_id: str | None = None,
    local_lane: int | None = None,
) -> dict[str, object]:
    proposal_id = proposal_id or _digest("shape-proposal")
    local_lane = (0 if greedy else 1) if local_lane is None else local_lane
    global_lane = local_lane
    color = 0 if greedy else 2
    probability = 0.1
    log_probability = math.log(probability)
    trace_counts = mask_count_trace(1)
    steps: list[dict[str, object]] = []
    for step in range(1, 13):
        remasked = trace_counts[step] == 1
        steps.append(
            {
                "step": step,
                "masked_indices_at_entry": [0],
                "logits_float64_le_sha256": float64_sequence_sha256([0.0] * 10),
                "predictions": [
                    {
                        "linear_index": 0,
                        "row": 0,
                        "column": 0,
                        "chosen_color": color,
                        "probability_hex": probability.hex(),
                        "log_probability_hex": log_probability.hex(),
                        "uniform_hex": None if greedy else (0.25).hex(),
                        "provisional_remasked": remasked,
                    }
                ],
                "remasked_indices": [0] if remasked else [],
                "state_after_step": ["MASK"] if remasked else [color],
            }
        )
    return {
        "schema": LANE_TRACE_SCHEMA_VERSION,
        "blind_task_id": task_id,
        "test_index": 0,
        "shape_order": 0,
        "shape_proposal_id": proposal_id,
        "proposed_height": 1,
        "proposed_width": 1,
        "global_lane": global_lane,
        "local_lane": local_lane,
        "greedy": greedy,
        "seed_u64": lane_seed(task_id, 0, proposal_id, local_lane),
        "initial_masked_indices": [0],
        "steps": steps,
        "final_grid": [[color]],
        "output_key": grid_key(((color,),)),
    }


def _forged_two_cell_tie_trace() -> dict[str, object]:
    """Structurally self-consistent trace whose step-8 tie-break is forged."""

    task_id = "blind_two_cell_tie_fixture"
    proposal_id = _digest("two-cell-shape-proposal")
    schedule = mask_count_trace(2)
    state: list[int | str] = ["MASK", "MASK"]
    steps: list[dict[str, object]] = []
    for step in range(1, 13):
        entry = [index for index, value in enumerate(state) if value == "MASK"]
        predictions: list[dict[str, object]] = []
        for linear_index in entry:
            probability = 0.1
            if step == 8:
                probability += 1e-15 if linear_index == 0 else -1e-15
            predictions.append(
                {
                    "linear_index": linear_index,
                    "row": 0,
                    "column": linear_index,
                    "chosen_color": 0,
                    "probability_hex": probability.hex(),
                    "log_probability_hex": math.log(probability).hex(),
                    "uniform_hex": None,
                    "provisional_remasked": False,
                }
            )
        retained = schedule[step]
        if retained == len(entry):
            remasked = list(entry)
        elif step == 8:
            remasked = [1]
        else:
            remasked = list(entry[:retained])
        next_state = list(state)
        for prediction in predictions:
            linear_index = int(prediction["linear_index"])
            is_remasked = linear_index in remasked
            prediction["provisional_remasked"] = is_remasked
            next_state[linear_index] = "MASK" if is_remasked else 0
        steps.append(
            {
                "step": step,
                "masked_indices_at_entry": entry,
                "logits_float64_le_sha256": float64_sequence_sha256(
                    [0.0] * (10 * len(entry))
                ),
                "predictions": predictions,
                "remasked_indices": remasked,
                "state_after_step": next_state,
            }
        )
        state = next_state
    return {
        "schema": LANE_TRACE_SCHEMA_VERSION,
        "blind_task_id": task_id,
        "test_index": 0,
        "shape_order": 0,
        "shape_proposal_id": proposal_id,
        "proposed_height": 1,
        "proposed_width": 2,
        "global_lane": 0,
        "local_lane": 0,
        "greedy": True,
        "seed_u64": lane_seed(task_id, 0, proposal_id, 0),
        "initial_masked_indices": [0, 1],
        "steps": steps,
        "final_grid": [[0, 0]],
        "output_key": grid_key(((0, 0),)),
    }


def _supported_pool_fixture(
    *, cache_dtype: str = "bfloat16", split_cache: bool = False
) -> dict[str, object]:
    fixed = _fixed_blind_sidecar()
    row = copy.deepcopy(
        next(item for item in fixed.rows if item["accepted_shape_count"] > 0)
    )
    proposal_id = _digest("pool-fixture-shape")
    shape = {
        "shape_order": 0,
        "shape_proposal_id": proposal_id,
        "proposed_height": 1,
        "proposed_width": 1,
    }
    row.update(
        {
            "shape_proposals_pre_dedup": 1,
            "shape_proposals_post_dedup": 1,
            "accepted_shape_count": 1,
            "shape_proposal_truncation_count": 0,
            "accepted_shapes": [shape],
            "status": SHAPE_READY,
        }
    )
    row.pop("row_id")
    row["row_id"] = canonical_sha256(row)
    sidecar = _fixture_sidecar((row,))
    task_id = str(row["blind_task_id"])
    test_index = int(row["test_index"])
    checkpoint = _digest("supported-checkpoint")
    query = tuple(tuple(int(value) for value in values) for values in row["test_input"])
    encoder = make_encoder_forward_row(
        blind_task_id=task_id,
        test_index=test_index,
        ordered_grid_descriptors=[
            {
                "role": "query_input",
                "pair_index": test_index,
                "height": len(query),
                "width": len(query[0]),
                "grid_key": grid_key(query),
            }
        ],
        padded_batch_shape=[1, 1],
        unpadded_memory_length=1,
        cache_dtype=cache_dtype,
        gpu_forward_ns=0,
        wall_time_ns=0,
        cuda_peak_allocated_bytes=0,
        cuda_peak_reserved_bytes=0,
    )
    cache = {
        "encoder_call_id": encoder["call_id"],
        "cache_key": canonical_sha256(
            {
                "checkpoint_sha256": checkpoint,
                "blind_task_id": task_id,
                "test_index": test_index,
                "memory_length": 1,
                "dtype": cache_dtype,
            }
        ),
        "memory_length": 1,
        "dtype": cache_dtype,
    }
    decoder_rows: list[dict[str, object]] = []
    calls_by_lane: dict[int, list[str]] = {lane: [] for lane in range(64)}
    for step in range(1, 13):
        for batch_index, start in enumerate(range(0, 64, 8)):
            global_lanes = list(range(start, start + 8))
            decoder = make_decoder_forward_row(
                blind_task_id=task_id,
                test_index=test_index,
                shape_order=0,
                shape_proposal_id=proposal_id,
                step=step,
                batch_index_within_shape=batch_index,
                global_lanes=global_lanes,
                cached_memory_descriptor=(
                    {
                        **cache,
                        "cache_key": _digest("different-cache"),
                    }
                    if split_cache and step == 1 and batch_index == 0
                    else cache
                ),
                gpu_forward_ns=0,
                wall_time_ns=0,
                cuda_peak_allocated_bytes=0,
                cuda_peak_reserved_bytes=0,
            )
            decoder_rows.append(decoder)
            for global_lane in global_lanes:
                calls_by_lane[global_lane].append(decoder["call_id"])
    traces = tuple(
        _one_cell_trace(
            greedy=local_lane == 0,
            task_id=task_id,
            proposal_id=proposal_id,
            local_lane=local_lane,
        )
        for local_lane in range(64)
    )
    lanes = tuple(
        make_lane_row(
            trace,
            checkpoint_sha256=checkpoint,
            batch_forward_call_ids=calls_by_lane[local_lane],
        )
        for local_lane, trace in enumerate(traces)
    )
    pair = make_pair_cost_row(
        blind_task_id=task_id,
        test_index=test_index,
        accepted_shapes=[shape],
        lane_ids=[lane["lane_id"] for lane in lanes],
        encoder_call_ids=[encoder["call_id"]],
        decoder_call_ids=[row["call_id"] for row in decoder_rows],
        unique_outputs=2,
    )
    setup = make_pool_setup_cost_row(
        checkpoint_sha256=checkpoint,
        warmup_descriptor={},
        model_construction_ns=0,
        checkpoint_load_ns=0,
        warmup_ns=0,
        setup_wall_time_ns=0,
        gpu_warmup_ns=0,
        cuda_peak_allocated_bytes=0,
        cuda_peak_reserved_bytes=0,
    )
    return {
        "sidecar": sidecar,
        "checkpoint": checkpoint,
        "setup": setup,
        "pairs": (pair,),
        "lanes": lanes,
        "traces": traces,
        "encoders": (encoder,),
        "decoders": tuple(decoder_rows),
    }


class M04aBlindSidecarTests(unittest.TestCase):
    def test_fixed_sidecar_includes_every_task_and_test_pair(self) -> None:
        sidecar = _fixed_blind_sidecar()
        self.assertEqual(sidecar.manifest["task_count"], 20)
        self.assertEqual(
            sidecar.manifest["test_pair_count"], FIXED_BLIND_TEST_PAIR_COUNT
        )
        self.assertEqual(sidecar.manifest["candidate_count_zero_task_count"], 12)
        self.assertEqual(sidecar.manifest["candidate_count_zero_test_pair_count"], 13)
        self.assertEqual(sidecar.manifest["shape_supported_test_pair_count"], 16)
        self.assertEqual(sidecar.manifest["no_shape_test_pair_count"], 5)
        self.assertEqual(len(sidecar.rows), 21)
        self.assertEqual(
            len({(row["blind_task_id"], row["test_index"]) for row in sidecar.rows}),
            21,
        )
        zero_tasks = {
            row["blind_task_id"]
            for row in sidecar.rows
            if row["candidate_count_zero_subgroup"]
        }
        supported_zero_tasks = {
            row["blind_task_id"]
            for row in sidecar.rows
            if row["candidate_count_zero_subgroup"] and row["accepted_shape_count"] > 0
        }
        self.assertEqual(len(zero_tasks), 12)
        self.assertEqual(len(supported_zero_tasks), 9)
        for row in sidecar.rows:
            self.assertEqual(
                row["candidate_count_zero_subgroup"], row["dsl_candidate_count"] == 0
            )
            self.assertEqual(
                row["status"],
                SHAPE_READY if row["accepted_shape_count"] else NO_SHAPE_PROPOSAL,
            )

    def test_nonproduction_sidecar_requires_explicit_fixture_mode(self) -> None:
        fixed = _fixed_blind_sidecar()
        rows = (copy.deepcopy(fixed.rows[0]),)
        manifest = copy.deepcopy(fixed.manifest)
        manifest.update(
            {
                "task_count": 1,
                "test_pair_count": 1,
                "candidate_count_zero_task_count": int(
                    rows[0]["candidate_count_zero_subgroup"]
                ),
                "candidate_count_zero_test_pair_count": int(
                    rows[0]["candidate_count_zero_subgroup"]
                ),
                "shape_supported_test_pair_count": int(
                    rows[0]["accepted_shape_count"] > 0
                ),
                "no_shape_test_pair_count": int(rows[0]["accepted_shape_count"] == 0),
                "rows_sha256": hashlib.sha256(serialize_jsonl(rows)).hexdigest(),
            }
        )
        manifest.pop("sidecar_id")
        manifest["sidecar_id"] = canonical_sha256(manifest)
        with self.assertRaisesRegex(ValueError, "production blind shape sidecar"):
            validate_blind_input_shape_sidecar(manifest, rows)
        fixture = validate_blind_input_shape_sidecar(manifest, rows, fixture_mode=True)
        self.assertTrue(fixture.fixture_mode)

    def test_production_sidecar_parent_hash_is_pinned(self) -> None:
        fixed = _fixed_blind_sidecar()
        manifest = copy.deepcopy(fixed.manifest)
        manifest["blind_artifact_manifest_sha256"] = _digest("forged-parent")
        manifest.pop("sidecar_id")
        manifest["sidecar_id"] = canonical_sha256(manifest)
        with self.assertRaisesRegex(ValueError, "blind_artifact_manifest_sha256"):
            validate_blind_input_shape_sidecar(manifest, fixed.rows)

    def test_shape_order_dedup_keeps_first_identity_and_records_truncation(
        self,
    ) -> None:
        first = _proposal(row_factor=1, column_factor=1, query_shapes=((1, 1),))
        duplicate_shape = _proposal(
            row_factor=2, column_factor=2, query_shapes=((1, 1),)
        )
        distinct = [
            _proposal(
                row_factor=index + 3,
                column_factor=1,
                query_shapes=((index + 2, index + 2),),
            )
            for index in range(5)
        ]
        selection = select_blind_shapes(
            [first, duplicate_shape, *distinct], test_index=0, expected_test_count=1
        )
        self.assertEqual(selection.pre_dedup_count, 7)
        self.assertEqual(selection.post_dedup_count, 6)
        self.assertEqual(selection.truncation_count, 2)
        self.assertEqual(len(selection.accepted_shapes), 4)
        self.assertEqual(
            selection.accepted_shapes[0]["shape_proposal_id"], first["proposal_id"]
        )
        self.assertEqual(
            [shape["shape_order"] for shape in selection.accepted_shapes], [0, 1, 2, 3]
        )

    def test_oracle_fields_and_oracle_paths_are_rejected(self) -> None:
        assert_oracle_free_payload({"oracle_inputs_available_to_pool": False})
        for payload in (
            {"query_output_sha256": _digest("answer")},
            {"oracle_shape": [1, 1]},
            {"source": "results/private_oracle/answers.jsonl"},
            {"oracle_inputs_available_to_pool": True},
            {"oracle": _digest("answer")},
            {"oracleHash": _digest("answer")},
            {"queryOutput": [[1]]},
            {"testOutput": [[1]]},
            {"groundTruth": [[1]]},
            {"oracleInputsAvailableToPool": False},
        ):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                assert_oracle_free_payload(payload)
        injected = _proposal(row_factor=1, column_factor=1, query_shapes=((1, 1),))
        injected["oracle_shape"] = [1, 1]
        with self.assertRaises(ValueError):
            select_blind_shapes([injected], test_index=0, expected_test_count=1)


class M04aClosedWorldArtifactTests(unittest.TestCase):
    @unittest.skipUnless(
        os.name == "posix" and hasattr(os, "mkfifo"),
        "FIFO replacement is a POSIX-only boundary",
    )
    def test_closed_world_reader_rejects_fifo_without_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            os.mkfifo(root / "payload.bin", 0o600)
            payload = b""
            artifact_manifest = {
                "schema_version": 1,
                "bundle_status": "complete",
                "run_id": "fixture",
                "artifacts": {
                    "payload.bin": {
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "bytes": 0,
                        "rows": None,
                    }
                },
            }
            (root / "artifact_manifest.json").write_text(
                json.dumps(artifact_manifest), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "regular file"):
                read_closed_world_bundle(root)

    def test_windows_verified_path_handles_pin_directories_against_rename_swap(
        self,
    ) -> None:
        create_file = Mock(return_value=101)
        directory_handle = m04a_evidence._create_windows_component_handle(
            create_file,
            Path("bundle"),
            desired_access=0x80,
            is_leaf=False,
        )
        leaf_handle = m04a_evidence._create_windows_component_handle(
            create_file,
            Path("bundle") / "artifact_manifest.json",
            desired_access=0x80000000,
            is_leaf=True,
        )

        self.assertEqual(directory_handle, 101)
        self.assertEqual(leaf_handle, 101)
        directory_call, leaf_call = create_file.call_args_list
        directory_share_mode = directory_call.args[2]
        leaf_share_mode = leaf_call.args[2]
        self.assertEqual(directory_share_mode, 0x00000001 | 0x00000002)
        self.assertEqual(
            leaf_share_mode,
            0x00000001 | 0x00000002 | 0x00000004,
        )
        self.assertEqual(directory_share_mode & 0x00000004, 0)
        self.assertNotEqual(leaf_share_mode & 0x00000004, 0)

    def test_closed_world_reader_rejects_extra_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = b'{"ok":true}\n'
            (root / "payload.json").write_bytes(payload)
            artifact_manifest = {
                "schema_version": 1,
                "bundle_status": "complete",
                "run_id": "fixture",
                "artifacts": {
                    "payload.json": {
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "bytes": len(payload),
                        "rows": None,
                    }
                },
            }
            (root / "artifact_manifest.json").write_text(
                json.dumps(artifact_manifest), encoding="utf-8"
            )
            read_closed_world_bundle(root)
            (root / "extra.bin").write_bytes(b"injected")
            with self.assertRaisesRegex(ValueError, "extra"):
                read_closed_world_bundle(root)

    def test_verified_bundle_rechecks_bytes_at_consumption_time(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = b'{"ok":0}\n'
            mutated = b'{"ok":1}\n'
            self.assertEqual(len(original), len(mutated))
            payload_path = root / "payload.json"
            payload_path.write_bytes(original)
            artifact_manifest = {
                "schema_version": 1,
                "bundle_status": "complete",
                "run_id": "fixture",
                "artifacts": {
                    "payload.json": {
                        "sha256": hashlib.sha256(original).hexdigest(),
                        "bytes": len(original),
                        "rows": None,
                    }
                },
            }
            (root / "artifact_manifest.json").write_text(
                json.dumps(artifact_manifest), encoding="utf-8"
            )
            bundle = read_closed_world_bundle(root)
            self.assertEqual(bundle.read_bytes("payload.json"), original)
            committed_outer = bundle.artifact_manifest_bytes
            exposed_metadata = bundle.artifacts
            exposed_metadata["payload.json"]["sha256"] = hashlib.sha256(
                mutated
            ).hexdigest()
            exposed_manifest = bundle.artifact_manifest
            exposed_manifest["artifacts"]["payload.json"]["sha256"] = hashlib.sha256(
                mutated
            ).hexdigest()
            payload_path.write_bytes(mutated)
            with self.assertRaisesRegex(ValueError, "metadata mismatch"):
                bundle.read_bytes("payload.json")
            self.assertEqual(bundle.artifact_manifest_bytes, committed_outer)

    def test_closed_world_reader_rejects_unbounded_metadata_before_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = b""
            (root / "payload.bin").write_bytes(payload)
            artifact_manifest = {
                "schema_version": 1,
                "bundle_status": "complete",
                "run_id": "fixture",
                "artifacts": {
                    "payload.bin": {
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "bytes": 512 * 1024 * 1024 + 1,
                        "rows": None,
                    }
                },
            }
            (root / "artifact_manifest.json").write_text(
                json.dumps(artifact_manifest), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "single-file byte limit"):
                read_closed_world_bundle(root)

    def test_file_backed_publication_is_exclusive_and_timing_is_nonsemantic(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources: dict[str, Path] = {}
            for name in TRAINING_ARTIFACT_FILES:
                source = root / "sources" / Path(*name.split("/"))
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_bytes(_artifact_content(name))
                sources[name] = source
            file_rows = _file_rows(TRAINING_ARTIFACT_FILES)
            first = build_training_artifact_manifest(
                lineage=_lineage(TRAINING_LINEAGE_FIELDS),
                files=file_rows,
                closure=_training_closure(),
                timing=_training_timing(1),
            )
            second = build_training_artifact_manifest(
                lineage=_lineage(TRAINING_LINEAGE_FIELDS),
                files=file_rows,
                closure=_training_closure(),
                timing=_training_timing(999),
            )
            self.assertEqual(first["semantic_id"], second["semantic_id"])

            evaluation_source = root / "evaluation.json"
            evaluation_source.write_bytes(b'{"fixture":true}\n')
            pool_lineage = _lineage(POOL_LINEAGE_FIELDS)
            pool_parent = build_pool_artifact_manifest(
                lineage=pool_lineage,
                files=_file_rows(POOL_ARTIFACT_FILES),
                closure=_pool_closure(),
                timing=_pool_timing(),
            )
            pool_parent_source = root / "pool_artifact_manifest.json"
            pool_parent_bytes = serialize_json(pool_parent)
            pool_parent_source.write_bytes(pool_parent_bytes)
            evaluation_lineage = _lineage(EVALUATION_LINEAGE_FIELDS)
            evaluation_lineage.update(
                {
                    "pool_artifact_id": pool_parent["semantic_id"],
                    "pool_artifact_manifest_sha256": hashlib.sha256(
                        pool_parent_bytes
                    ).hexdigest(),
                    "dsl_pool_content_id": pool_lineage["dsl_pool_content_id"],
                    "python_runtime_lock_sha256": pool_lineage[
                        "python_runtime_lock_sha256"
                    ],
                    "python_runtime_lock_id": pool_lineage["python_runtime_lock_id"],
                    "python_runtime_identity_sha256": pool_lineage[
                        "python_runtime_identity_sha256"
                    ],
                }
            )
            target = root / "evaluation_bundle"
            published = publish_evaluation_artifact_bundle(
                target,
                lineage=evaluation_lineage,
                artifact_files={
                    "evaluation.json": evaluation_source,
                    "pool_artifact_manifest.json": pool_parent_source,
                },
                closure={"rows": 1},
                timing={"wall_time_ns": 1},
            )
            loaded = read_m04a_artifact_bundle(target, expected_kind="evaluation")
            self.assertEqual(loaded.manifest, published.manifest)
            drifted_lineage = dict(evaluation_lineage)
            drifted_lineage["python_runtime_lock_id"] = _digest("drifted-runtime")
            with self.assertRaisesRegex(RuntimeError, "runtime/pool lineage"):
                publish_evaluation_artifact_bundle(
                    root / "drifted_evaluation_bundle",
                    lineage=drifted_lineage,
                    artifact_files={
                        "evaluation.json": evaluation_source,
                        "pool_artifact_manifest.json": pool_parent_source,
                    },
                    closure={"rows": 1},
                    timing={"wall_time_ns": 1},
                )
            with self.assertRaises(FileExistsError):
                publish_evaluation_artifact_bundle(
                    target,
                    lineage=evaluation_lineage,
                    artifact_files={
                        "evaluation.json": evaluation_source,
                        "pool_artifact_manifest.json": pool_parent_source,
                    },
                    closure={"rows": 1},
                    timing={"wall_time_ns": 2},
                )
            with self.assertRaisesRegex(RuntimeError, "training payload|valid JSON"):
                publish_training_artifact_bundle(
                    root / "incomplete_training_bundle",
                    lineage=_lineage(TRAINING_LINEAGE_FIELDS),
                    artifact_files=sources,
                    closure=_training_closure(),
                    timing=_training_timing(1),
                )

    def test_complete_training_bundle_cannot_be_one_file_or_over_budget(self) -> None:
        with self.assertRaisesRegex(ValueError, "file set mismatch"):
            build_training_artifact_manifest(
                lineage=_lineage(TRAINING_LINEAGE_FIELDS),
                files=[
                    {
                        "path": "training_cost_ledger.jsonl",
                        "sha256": _digest("x"),
                        "bytes": 1,
                    }
                ],
                closure=_training_closure(),
                timing=_training_timing(1),
            )
        over_budget = _training_timing(1)
        over_budget["primary_training_wall_time_ns"] = 86_400_000_000_000
        over_budget["gpu_lock_wall_time_ns"] = 86_400_000_000_006
        over_budget["budget_remaining_ns"] = 0
        with self.assertRaisesRegex(ValueError, "24 GPU-hour"):
            build_training_artifact_manifest(
                lineage=_lineage(TRAINING_LINEAGE_FIELDS),
                files=_file_rows(TRAINING_ARTIFACT_FILES),
                closure=_training_closure(),
                timing=over_budget,
            )

    def test_complete_pool_bundle_requires_exact_files_including_replay_receipt(
        self,
    ) -> None:
        files = _file_rows(POOL_ARTIFACT_FILES)
        files = [
            row for row in files if row["path"] != "replay_verification_receipt.json"
        ]
        with self.assertRaisesRegex(ValueError, "replay_verification_receipt"):
            build_pool_artifact_manifest(
                lineage=_lineage(POOL_LINEAGE_FIELDS),
                files=files,
                closure=_pool_closure(),
                timing=_pool_timing(),
            )

    def test_pool_publisher_rejects_oracle_json_injection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            injected = root / "injected.json"
            injected.write_text(
                json.dumps({"query_output": [[1]]}) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "oracle|query-output"):
                publish_pool_artifact_bundle(
                    root / "pool",
                    lineage=_lineage(POOL_LINEAGE_FIELDS),
                    artifact_files={"candidate_summary.json": injected},
                    closure={"raw_lanes": 0},
                    timing={"wall_time_ns": 0},
                )

    def test_evaluation_lineage_is_separate_from_blind_pool_lineage(self) -> None:
        self.assertIn("oracle_artifact_manifest_sha256", EVALUATION_LINEAGE_FIELDS)
        self.assertNotIn("oracle_artifact_manifest_sha256", POOL_LINEAGE_FIELDS)

    def test_pool_runtime_lineage_must_equal_training_parent(self) -> None:
        training_lineage = _lineage(TRAINING_LINEAGE_FIELDS)
        parent = {
            "semantic_id": _digest("training-parent"),
            "lineage": training_lineage,
        }
        pool_lineage = _lineage(POOL_LINEAGE_FIELDS)
        pool_lineage.update(
            {
                "training_artifact_id": parent["semantic_id"],
                "selected_checkpoint_manifest_sha256": training_lineage[
                    "selected_checkpoint_manifest_sha256"
                ],
                "checkpoint_sha256": training_lineage["checkpoint_sha256"],
                "python_runtime_lock_sha256": training_lineage[
                    "python_runtime_lock_sha256"
                ],
                "python_runtime_lock_id": training_lineage["python_runtime_lock_id"],
            }
        )
        environment = _pool_environment_manifest(
            runtime_lock_sha256=pool_lineage["python_runtime_lock_sha256"],
            runtime_lock_id=pool_lineage["python_runtime_lock_id"],
        )
        python_identity = m04a_evidence.python_runtime_identity_sha256(
            {
                "implementation": environment["python_implementation"],
                "version": environment["python_version"],
                "executable": environment["python_executable"],
                "executable_sha256": environment["python_executable_sha256"],
                "executable_bytes": environment["python_executable_bytes"],
            }
        )
        training_lineage["python_runtime_identity_sha256"] = python_identity
        pool_lineage["python_runtime_identity_sha256"] = python_identity
        m04a_evidence._validate_pool_parent_lineage(parent, pool_lineage)
        self.assertEqual(
            m04a_evidence._validate_pool_environment_runtime(
                environment,
                lineage=pool_lineage,
                training_parent=parent,
            ),
            environment,
        )
        attacker_environment = dict(environment)
        attacker_environment.update(
            {
                "python_executable": "/attacker/other/python",
                "python_executable_sha256": "a" * 64,
                "python_executable_bytes": 12_345,
            }
        )
        with self.assertRaisesRegex(ValueError, "pool/training lineage"):
            m04a_evidence._validate_pool_environment_runtime(
                attacker_environment,
                lineage=pool_lineage,
                training_parent=parent,
            )
        drifted_environment = dict(environment)
        drifted_environment["python_runtime_lock_id"] = _digest(
            "drifted-environment-runtime"
        )
        with self.assertRaisesRegex(ValueError, "pool/training lineage"):
            m04a_evidence._validate_pool_environment_runtime(
                drifted_environment,
                lineage=pool_lineage,
                training_parent=parent,
            )
        wrong_schema = dict(environment)
        wrong_schema["schema"] = "afts-m04a-environment/v0.2"
        with self.assertRaisesRegex(ValueError, "unsupported.*environment"):
            m04a_evidence._validate_pool_environment_runtime(
                wrong_schema,
                lineage=pool_lineage,
                training_parent=parent,
            )
        for field, drifted, error, message in (
            ("python_executable", "relative/python", ValueError, "executable path"),
            ("python_executable_sha256", "bad", TypeError, "executable_sha256"),
            ("python_executable_bytes", 0, TypeError, "executable_bytes"),
        ):
            malformed_environment = dict(environment)
            malformed_environment[field] = drifted
            with self.subTest(field=field), self.assertRaisesRegex(error, message):
                m04a_evidence._validate_pool_environment_runtime(
                    malformed_environment,
                    lineage=pool_lineage,
                    training_parent=parent,
                )
        pool_lineage["python_runtime_lock_sha256"] = _digest("drifted-runtime")
        with self.assertRaisesRegex(ValueError, "checkpoint/runtime"):
            m04a_evidence._validate_pool_parent_lineage(parent, pool_lineage)


class M04aTraceAndCostTests(unittest.TestCase):
    def test_pool_replay_verifier_binds_checkpoint_sidecar_order_and_receipt(
        self,
    ) -> None:
        fixed = _fixed_blind_sidecar()
        no_shape_rows = tuple(
            copy.deepcopy(row) for row in fixed.rows if row["accepted_shape_count"] == 0
        )[:2]
        sidecar = _fixture_sidecar(no_shape_rows)
        pair_rows = tuple(
            make_pair_cost_row(
                blind_task_id=row["blind_task_id"],
                test_index=row["test_index"],
                accepted_shapes=[],
                lane_ids=[],
                encoder_call_ids=[],
                decoder_call_ids=[],
            )
            for row in no_shape_rows
        )
        checkpoint = _digest("checkpoint")
        setup = make_pool_setup_cost_row(
            checkpoint_sha256=checkpoint,
            warmup_descriptor={},
            model_construction_ns=0,
            checkpoint_load_ns=0,
            warmup_ns=0,
            setup_wall_time_ns=0,
            gpu_warmup_ns=0,
            cuda_peak_allocated_bytes=0,
            cuda_peak_reserved_bytes=0,
        )
        receipt = validate_pool_evidence(
            shape_sidecar=sidecar,
            checkpoint_sha256=checkpoint,
            setup_row=setup,
            pair_rows=pair_rows,
            lane_rows=(),
            traces=(),
            encoder_rows=(),
            decoder_rows=(),
            replay_decoder_call=lambda _row: {},
            replay_cpu_uniforms=lambda _seed, _count: (),
            verifier_source_sha256=_digest("verifier"),
            fixture_mode=True,
        )
        self.assertEqual(receipt["pair_count"], 2)
        self.assertEqual(receipt["checkpoint_sha256"], checkpoint)

        with self.assertRaisesRegex(ValueError, "sidecar order"):
            validate_pool_evidence(
                shape_sidecar=sidecar,
                checkpoint_sha256=checkpoint,
                setup_row=setup,
                pair_rows=tuple(reversed(pair_rows)),
                lane_rows=(),
                traces=(),
                encoder_rows=(),
                decoder_rows=(),
                replay_decoder_call=lambda _row: {},
                replay_cpu_uniforms=lambda _seed, _count: (),
                verifier_source_sha256=_digest("verifier"),
                fixture_mode=True,
            )
        with self.assertRaisesRegex(ValueError, "expected checkpoint"):
            validate_pool_evidence(
                shape_sidecar=sidecar,
                checkpoint_sha256=_digest("other-checkpoint"),
                setup_row=setup,
                pair_rows=pair_rows,
                lane_rows=(),
                traces=(),
                encoder_rows=(),
                decoder_rows=(),
                replay_decoder_call=lambda _row: {},
                replay_cpu_uniforms=lambda _seed, _count: (),
                verifier_source_sha256=_digest("verifier"),
                fixture_mode=True,
            )
        forged = copy.deepcopy(receipt)
        forged["raw_logit_hashes_verified"] = False
        with self.assertRaisesRegex(ValueError, "raw_logit_hashes_verified"):
            validate_pool_replay_receipt(forged)

    def test_pool_replay_verifier_covers_every_lane_call_and_one_bf16_cache(
        self,
    ) -> None:
        fixture = _supported_pool_fixture()

        def replay(decoder: dict[str, object]) -> dict[int, object]:
            return {
                int(global_lane): [[0.0] * 10]
                for global_lane in decoder["global_lanes"]
            }

        receipt = validate_pool_evidence(
            shape_sidecar=fixture["sidecar"],
            checkpoint_sha256=fixture["checkpoint"],
            setup_row=fixture["setup"],
            pair_rows=fixture["pairs"],
            lane_rows=fixture["lanes"],
            traces=fixture["traces"],
            encoder_rows=fixture["encoders"],
            decoder_rows=fixture["decoders"],
            replay_decoder_call=replay,
            replay_cpu_uniforms=lambda _seed, count: [0.25] * count,
            verifier_source_sha256=_digest("verifier"),
            fixture_mode=True,
        )
        self.assertEqual(receipt["lane_count"], 64)
        self.assertEqual(receipt["decoder_call_count"], 96)
        self.assertEqual(receipt["masked_token_predictions"], 768)

        float32_fixture = _supported_pool_fixture(cache_dtype="float32")
        with self.assertRaisesRegex(ValueError, "must be BF16"):
            validate_pool_evidence(
                shape_sidecar=float32_fixture["sidecar"],
                checkpoint_sha256=float32_fixture["checkpoint"],
                setup_row=float32_fixture["setup"],
                pair_rows=float32_fixture["pairs"],
                lane_rows=float32_fixture["lanes"],
                traces=float32_fixture["traces"],
                encoder_rows=float32_fixture["encoders"],
                decoder_rows=float32_fixture["decoders"],
                replay_decoder_call=replay,
                replay_cpu_uniforms=lambda _seed, count: [0.25] * count,
                verifier_source_sha256=_digest("verifier"),
                fixture_mode=True,
            )

        split_fixture = _supported_pool_fixture(split_cache=True)
        with self.assertRaisesRegex(ValueError, "identical bound BF16"):
            validate_pool_evidence(
                shape_sidecar=split_fixture["sidecar"],
                checkpoint_sha256=split_fixture["checkpoint"],
                setup_row=split_fixture["setup"],
                pair_rows=split_fixture["pairs"],
                lane_rows=split_fixture["lanes"],
                traces=split_fixture["traces"],
                encoder_rows=split_fixture["encoders"],
                decoder_rows=split_fixture["decoders"],
                replay_decoder_call=replay,
                replay_cpu_uniforms=lambda _seed, count: [0.25] * count,
                verifier_source_sha256=_digest("verifier"),
                fixture_mode=True,
            )

    def test_trace_reconstruction_and_provisional_tamper_hash(self) -> None:
        trace = _one_cell_trace()
        summary = validate_lane_trace(trace)
        self.assertEqual(summary.mask_count_trace, mask_count_trace(1))
        self.assertEqual(summary.rng_draw_count, 0)
        self.assertEqual(summary.masked_token_predictions, 12)
        calls = [_digest(f"decoder-{step}") for step in range(12)]
        row = make_lane_row(
            trace,
            checkpoint_sha256=_digest("checkpoint"),
            batch_forward_call_ids=calls,
            cpu_sampling_ns=11,
            lane_wall_time_ns=22,
        )
        tampered = copy.deepcopy(trace)
        tampered["steps"][0]["predictions"][0]["chosen_color"] = 4
        self.assertNotEqual(lane_trace_sha256(trace), lane_trace_sha256(tampered))
        with self.assertRaisesRegex(ValueError, "trace_sha256"):
            validate_lane_row(row, trace=tampered)

    def test_neural_replay_binds_logits_probabilities_choices_and_rng(self) -> None:
        greedy = _one_cell_trace(greedy=True)
        logits = [[[0.0] * 10] for _ in range(12)]
        validate_lane_trace_replay(
            greedy, replayed_logits_by_step=logits, replayed_uniforms=[]
        )

        wrong_hash = copy.deepcopy(greedy)
        wrong_hash["steps"][0]["logits_float64_le_sha256"] = _digest("wrong")
        with self.assertRaisesRegex(ValueError, "logits hash"):
            validate_lane_trace_replay(
                wrong_hash, replayed_logits_by_step=logits, replayed_uniforms=[]
            )

        wrong_probability = copy.deepcopy(greedy)
        wrong_probability["steps"][0]["predictions"][0]["probability_hex"] = (0.2).hex()
        wrong_probability["steps"][0]["predictions"][0]["log_probability_hex"] = (
            math.log(0.2).hex()
        )
        with self.assertRaisesRegex(ValueError, "softmax"):
            validate_lane_trace_replay(
                wrong_probability,
                replayed_logits_by_step=logits,
                replayed_uniforms=[],
            )

        wrong_greedy = copy.deepcopy(greedy)
        wrong_greedy["steps"][0]["predictions"][0]["chosen_color"] = 1
        with self.assertRaisesRegex(ValueError, "greedy"):
            validate_lane_trace_replay(
                wrong_greedy, replayed_logits_by_step=logits, replayed_uniforms=[]
            )

        sampled = _one_cell_trace(greedy=False)
        validate_lane_trace_replay(
            sampled,
            replayed_logits_by_step=logits,
            replayed_uniforms=[0.25] * 12,
        )
        wrong_uniform = copy.deepcopy(sampled)
        wrong_uniform["steps"][0]["predictions"][0]["uniform_hex"] = (0.26).hex()
        with self.assertRaisesRegex(ValueError, "seeded CPU RNG"):
            validate_lane_trace_replay(
                wrong_uniform,
                replayed_logits_by_step=logits,
                replayed_uniforms=[0.25] * 12,
            )

    def test_neural_replay_recomputes_confidence_ties_for_remasking(self) -> None:
        forged = _forged_two_cell_tie_trace()
        validate_lane_trace(forged)
        logits = [
            [[0.0] * 10 for _ in step["masked_indices_at_entry"]]
            for step in forged["steps"]
        ]
        with self.assertRaisesRegex(ValueError, "re-mask set differs"):
            validate_lane_trace_replay(
                forged,
                replayed_logits_by_step=logits,
                replayed_uniforms=[],
            )

    def test_structural_trace_rejects_probability_log_mismatch(self) -> None:
        trace = _one_cell_trace()
        trace["steps"][0]["predictions"][0]["log_probability_hex"] = math.log(0.2).hex()
        with self.assertRaisesRegex(ValueError, "natural log"):
            validate_lane_trace(trace)

    def test_lane_id_excludes_timing_fields(self) -> None:
        trace = _one_cell_trace()
        calls = [_digest(f"decoder-{step}") for step in range(12)]
        first = make_lane_row(
            trace,
            checkpoint_sha256=_digest("checkpoint"),
            batch_forward_call_ids=calls,
            cpu_sampling_ns=1,
            lane_wall_time_ns=2,
        )
        second = make_lane_row(
            trace,
            checkpoint_sha256=_digest("checkpoint"),
            batch_forward_call_ids=calls,
            cpu_sampling_ns=999,
            lane_wall_time_ns=1_000,
        )
        self.assertEqual(first["lane_id"], second["lane_id"])

    def test_k0_and_k3_cost_closure(self) -> None:
        k0 = expected_pair_cost_payload([])
        self.assertEqual(k0["shape_lane_allocations"], [])
        self.assertEqual(k0["raw_lanes"], 0)
        self.assertEqual(k0["encoder_batch_calls"], 0)
        self.assertEqual(k0["decoder_batch_calls"], 0)
        no_shape = make_pair_cost_row(
            blind_task_id="blind_fixture",
            test_index=0,
            accepted_shapes=[],
            lane_ids=[],
            encoder_call_ids=[],
            decoder_call_ids=[],
        )
        self.assertEqual(no_shape["status"], NO_SHAPE_PROPOSAL)

        k3 = expected_pair_cost_payload([4, 9, 16])
        self.assertEqual(k3["shape_lane_allocations"], [22, 21, 21])
        self.assertEqual(k3["raw_lanes"], 64)
        self.assertEqual(k3["sample_equivalent_forward_calls"], 768)
        self.assertEqual(k3["encoder_batch_calls"], 1)
        self.assertEqual(k3["decoder_batch_calls"], 108)
        self.assertEqual(k3["total_actual_batch_calls"], 109)
        shapes = [
            {
                "shape_order": index,
                "shape_proposal_id": _digest(f"shape-{index}"),
                "proposed_height": side,
                "proposed_width": side,
            }
            for index, side in enumerate((2, 3, 4))
        ]
        row = make_pair_cost_row(
            blind_task_id="blind_fixture",
            test_index=0,
            accepted_shapes=shapes,
            lane_ids=[_digest(f"lane-{index}") for index in range(64)],
            encoder_call_ids=[_digest("encoder")],
            decoder_call_ids=[_digest(f"decoder-{index}") for index in range(108)],
            format_invalid=3,
            unique_outputs=50,
        )
        self.assertEqual(row["shape_lane_allocations"], [22, 21, 21])
        self.assertEqual(row["format_valid"], 61)
        self.assertEqual(row["duplicate_outputs"], 11)

    def test_training_ledger_requires_exact_preflight_and_cost_only_resume(
        self,
    ) -> None:
        preflight = [
            make_training_cost_ledger_row(
                phase="preflight_update",
                event_index=index,
                optimizer_step=index + 1,
                optimizer_updates=1,
                microbatches=16,
                encoder_forward_calls=16,
                decoder_forward_calls=16,
                backward_calls=16,
                masked_token_predictions=16,
            )
            for index in range(99)
        ]
        with self.assertRaisesRegex(ValueError, "exactly ordered updates 1..100"):
            validate_training_cost_ledger(
                preflight,
                validation_episodes_per_pass=1,
                fresh_setup_wall_time_ns=0,
            )
        preflight.append(
            make_training_cost_ledger_row(
                phase="preflight_update",
                event_index=99,
                optimizer_step=100,
                optimizer_updates=1,
                microbatches=16,
                encoder_forward_calls=16,
                decoder_forward_calls=16,
                backward_calls=16,
                masked_token_predictions=16,
            )
        )
        with self.assertRaisesRegex(ValueError, "primary ledger"):
            validate_training_cost_ledger(
                preflight,
                validation_episodes_per_pass=1,
                fresh_setup_wall_time_ns=0,
            )

        resume = make_training_cost_ledger_row(
            phase="resume_segment",
            event_index=0,
            optimizer_step=2_000,
            checkpoint_index=1,
            resume_segment_index=1,
            checkpoint_io_bytes=1,
            checkpoint_io_ns=1,
            wall_time_ns=1,
        )
        validate_training_cost_ledger_row(resume)
        with self.assertRaisesRegex(ValueError, "zero semantic work"):
            make_training_cost_ledger_row(
                phase="resume_segment",
                event_index=0,
                optimizer_step=2_000,
                checkpoint_index=1,
                resume_segment_index=1,
                encoder_forward_calls=1,
            )

    def test_training_lock_timeline_includes_preflight_gap_and_endpoint(self) -> None:
        interval = m04a_evidence.make_training_lock_interval(((1_000, 1_100),))
        phase_timing = {
            "preflight_wall_time_ns": 40,
            "primary_training_wall_time_ns": 30,
            "validation_wall_time_ns": 10,
            "checkpoint_wall_time_ns": 10,
            "resume_setup_wall_time_ns": 8,
        }
        m04a_evidence._validate_training_lock_timeline_coverage(
            lock_interval=interval,
            preflight_started_perf_counter_ns=1_002,
            phase_timing=phase_timing,
            preflight_endpoint_wall_time_ns=40,
        )

        omitted_gap = dict(phase_timing)
        omitted_gap["resume_setup_wall_time_ns"] = 9
        with self.assertRaisesRegex(ValueError, "acquisition-to-preflight gap"):
            m04a_evidence._validate_training_lock_timeline_coverage(
                lock_interval=interval,
                preflight_started_perf_counter_ns=1_002,
                phase_timing=omitted_gap,
                preflight_endpoint_wall_time_ns=40,
            )

        short_first_segment = m04a_evidence.make_training_lock_interval(
            ((1_000, 1_041),)
        )
        endpoint_only_timing = {field: 0 for field in phase_timing}
        endpoint_only_timing["preflight_wall_time_ns"] = 1
        with self.assertRaisesRegex(ValueError, "preflight endpoint"):
            m04a_evidence._validate_training_lock_timeline_coverage(
                lock_interval=short_first_segment,
                preflight_started_perf_counter_ns=1_002,
                phase_timing=endpoint_only_timing,
                preflight_endpoint_wall_time_ns=40,
            )

    def test_training_manifest_rejects_incomplete_timing_and_oracle_closure(
        self,
    ) -> None:
        timing = _training_timing(1)
        timing.pop("checkpoint_wall_time_ns")
        with self.assertRaisesRegex(ValueError, "timing fields"):
            build_training_artifact_manifest(
                lineage=_lineage(TRAINING_LINEAGE_FIELDS),
                files=_file_rows(TRAINING_ARTIFACT_FILES),
                closure=_training_closure(),
                timing=timing,
            )
        closure = _training_closure()
        closure["oracle_score"] = 1
        with self.assertRaisesRegex(ValueError, "closure fields"):
            build_training_artifact_manifest(
                lineage=_lineage(TRAINING_LINEAGE_FIELDS),
                files=_file_rows(TRAINING_ARTIFACT_FILES),
                closure=closure,
                timing=_training_timing(1),
            )

    def test_import_is_warning_clean_and_does_not_load_torch(self) -> None:
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(PROJECT_ROOT / "src")
        result = subprocess.run(
            [
                sys.executable,
                "-W",
                "error",
                "-c",
                "import sys; import afts_arc.m04a_evidence; "
                "assert 'torch' not in sys.modules",
            ],
            cwd=PROJECT_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_full_preflight_parent_replay_remains_no_torch(self) -> None:
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(PROJECT_ROOT / "src")
        program = """
import runpy
import sys
from afts_arc import m04a_evidence as evidence
ns = runpy.run_path('tests/test_m04a_training_evidence_closure.py')
rows = ns['_preflight_rows']()
digest = ns['_digest']
report, parents = ns['_preflight_report'](
    rows,
    validation_outer_sha256=digest('validation-outer'),
    validation_jsonl_sha256=digest('validation-jsonl'),
    validation_summary_id=digest('validation-summary'),
    validation_row_count=2,
)
evidence.validate_preflight_artifact(
    report,
    rows,
    validation_episodes_per_pass=2,
    expected_validation_outer_manifest_sha256=digest('validation-outer'),
    expected_validation_jsonl_sha256=digest('validation-jsonl'),
    expected_validation_summary_id=digest('validation-summary'),
    **parents,
)
assert 'torch' not in sys.modules
assert 'afts_arc.m04a_cost_probe' not in sys.modules
assert 'afts_arc.m04a_preflight' not in sys.modules
"""
        result = subprocess.run(
            [sys.executable, "-W", "error", "-c", program],
            cwd=PROJECT_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
