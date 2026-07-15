from __future__ import annotations

import hashlib
import io
import tempfile
import unittest
import zipfile
from pathlib import Path

from afts_arc import m04a_evidence as evidence
from afts_arc import m04a_python_runtime_lock as runtime_lock_module
from afts_arc._source_bootstrap import named_bytes_fingerprint
from afts_arc.m04a_contract import canonical_sha256, mask_count_trace
from afts_arc.m04a_data import (
    ARC2Parent,
    M04AExample,
    M04ATrainingData,
    build_validation_episodes,
)
from afts_arc.m04a_evidence import (
    TRAINING_ARTIFACT_FILES,
    make_checkpoint_manifest_row,
    make_selected_checkpoint_manifest,
    make_training_cost_ledger_row,
    make_training_cost_summary,
    make_training_lock_interval,
    make_validation_metric_row,
    publish_training_artifact_bundle,
    read_committed_training_artifact_bundle,
    read_m04a_artifact_bundle,
    validate_training_cost_ledger,
)
from afts_arc.m04a_launch_plan import (
    EXPECTED_INPUT_ARTIFACT_PATHS,
    build_launch_plan,
    canonical_launch_plan_bytes,
    read_launch_plan_artifact,
)
from afts_arc.m04a_train_contract import (
    learning_rate_for_update,
    training_config_sha256,
)
from afts_arc.m04a_validation_manifest import (
    build_validation_manifest_commitment,
    publish_validation_episode_manifest,
)
from afts_arc.manifest import serialize_json, serialize_jsonl
from test_m04a_python_runtime_lock import _SyntheticMixbit, _reidentify


RUN_ID = "m04a-grid-cmlm-v0.1-seed20260711-r1"
REMOTE_ROOT = "/srv/afts"
GPU_UUID = "GPU-00000000-0000-0000-0000-000000000001"
GPU_BUDGET_NS = 86_400_000_000_000


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _reseal(payload: dict[str, object], identity: str) -> None:
    semantic = dict(payload)
    semantic.pop(identity, None)
    payload[identity] = canonical_sha256(semantic)


def _zip_materials(materials: dict[str, bytes]) -> tuple[bytes, str]:
    buffer = io.BytesIO()
    with zipfile.ZipFile(
        buffer, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name, content in sorted(materials.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, content, compresslevel=9)
    return buffer.getvalue(), named_bytes_fingerprint(materials.items())


class _TinyReARC:
    @property
    def parent_ids(self) -> tuple[str, ...]:
        return ("bbbbbbbb",)

    def example(self, parent_id: str, example_index: int) -> M04AExample:
        if parent_id != "bbbbbbbb" or not 0 <= example_index < 1000:
            raise KeyError((parent_id, example_index))
        color = example_index % 10
        return M04AExample.create([[color, 0]], [[0], [color]])

    def semantic_parent_id(self, parent_id: str) -> str:
        if parent_id != "bbbbbbbb":
            raise KeyError(parent_id)
        return parent_id


def _validation_episodes():
    parent = ARC2Parent(
        parent_id="aaaaaaaa",
        train=(
            M04AExample.create([[0, 1]], [[1, 0]]),
            M04AExample.create([[2], [0]], [[0, 2]]),
        ),
        test=(M04AExample.create([[3, 0]], [[3, 3]]),),
    )
    data = M04ATrainingData(
        arc2_parent_ids=(parent.parent_id,),
        arc2_parents={parent.parent_id: parent},
        rearc_parent_ids=("bbbbbbbb",),
        rearc=_TinyReARC(),
    )
    return build_validation_episodes(data)


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


def _preflight_fixture_payload() -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": "afts-grid-cmlm-preflight-fixture/v0.1",
        "grid_side": 30,
        "target_cell_count": 900,
        "demonstration_count": 10,
        "encoder_grid_count": 21,
        "encoder_grid_token_count": 901,
        "expected_memory_length": 18_921,
        "demonstrations": [
            {
                "pair_index": index,
                "input_constant_color": index,
                "output_constant_color": (index + 1) % 10,
                "height": 30,
                "width": 30,
            }
            for index in range(10)
        ],
        "query_constant_color": 0,
        "target_constant_color": 1,
        "training_mask_policy": "all_900_cells_every_microbatch",
        "training_updates": 100,
        "microbatches_per_update": 16,
        "inference_lanes": 8,
        "inference_steps": 12,
    }
    payload["fixture_id"] = canonical_sha256(payload)
    return payload


def _cost_probe_selection_rows(diagnostic_sha256: str) -> list[dict[str, object]]:
    return [
        {
            "checkpoint_index": index,
            "optimizer_step": index * evidence.VALIDATION_INTERVAL,
            "parent_grouped_ce_hex": (
                0.25
                if index in evidence.SELECTION_TIE_INDICES
                else 1.0 + index / 100.0
            ).hex(),
            "checkpoint_sha256": hashlib.sha256(
                f"{diagnostic_sha256}:{index}".encode("ascii")
            ).hexdigest(),
        }
        for index in range(1, evidence.VALIDATION_PASS_COUNT + 1)
    ]


def _build_preflight_evidence(
    rows: tuple[dict[str, object], ...],
    *,
    validation_outer_sha256: str,
    validation_jsonl_sha256: str,
    validation_summary_id: str,
    validation_row_count: int,
    launcher_sha256: str,
    launch_plan_sha256: str,
    runtime_source_fingerprint_sha256: str,
    test_source_fingerprint_sha256: str,
    attempt_nonce: str,
) -> tuple[dict[str, object], dict[str, object], bytes, dict[str, object]]:
    ledger_sha = canonical_sha256(list(rows))
    mask_trace = mask_count_trace(900)
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
    handshake_wall = 2
    handshake_gap = 1
    fresh_setup = 6
    checkpoint_roundtrip = 5
    selection_wall = 4
    primary = evidence.OPTIMIZER_UPDATES
    validation_calls = evidence.VALIDATION_PASS_COUNT * validation_row_count
    validation = validation_calls
    checkpoint = evidence.VALIDATION_PASS_COUNT * checkpoint_roundtrip
    projected = (
        handshake_wall
        + handshake_gap
        + total_preflight_wall
        + fresh_setup
        + primary
        + validation
        + checkpoint
        + selection_wall
        + evidence.PREFLIGHT_POST_VALIDATION_MARGIN_NS
    )
    projection: dict[str, object] = {
        "schema": evidence.PREFLIGHT_PROJECTION_SCHEMA_VERSION,
        "projection_method": evidence.PREFLIGHT_PROJECTION_METHOD,
        "measured_preflight_training_wall_ns": training_wall,
        "measured_preflight_total_wall_ns": total_preflight_wall,
        "mean_training_update_wall_ns": 1,
        "mean_training_microbatch_wall_ns": 1,
        "measured_lock_setup_handshake_wall_ns": handshake_wall,
        "measured_handshake_to_preflight_start_ns": handshake_gap,
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
        "projected_final_selection_wall_ns": selection_wall,
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
        "schema": evidence.PREFLIGHT_REPORT_SCHEMA_VERSION,
        "status": "PASS",
        "fixture": _preflight_fixture_payload(),
        "runtime": {
            "run_id": RUN_ID,
            "gpu_uuid": GPU_UUID,
            "logical_device_index": 0,
        },
        "model": {
            "model_semantics_version": "afts-grid-cmlm/v0.1",
            "parameter_count": 8_733_706,
            "training_config_sha256": training_config_sha256(),
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
            "run_id": RUN_ID,
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
            "run_id": RUN_ID,
            "checkpoint_sha256": "",
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
            "run_id": RUN_ID,
            "candidate_count": evidence.VALIDATION_PASS_COUNT,
            "selection_rule": evidence.CHECKPOINT_SELECTION_RULE,
            "selected_checkpoint_index": evidence.SELECTION_TIE_INDICES[0],
            "selection_wall_ns": selection_wall,
        },
        "lock_handshake_probe": {},
        "preflight_started_perf_counter_ns": 1_003,
        "budget_projection": projection,
        "overhead_cost_probe_id": "",
        "lock_handshake_artifact_sha256": "",
        "dataset_file_reads": 0,
        "training_checkpoint_writes": 0,
        "diagnostic_checkpoint_writes": 1,
        "fallback_used": False,
        "rng_state_restored": True,
        "ledger_rows_sha256": ledger_sha,
    }
    diagnostic_snapshot = b"D" * 100
    diagnostic_sha256 = hashlib.sha256(diagnostic_snapshot).hexdigest()
    report["checkpoint_cost_probe"]["checkpoint_sha256"] = diagnostic_sha256
    for field in (
        "fresh_reconstruction_probe",
        "checkpoint_cost_probe",
        "final_selection_probe",
    ):
        _reseal(report[field], "probe_id")

    lock_semantic: dict[str, object] = {
        "schema": evidence.LOCK_HANDSHAKE_ARTIFACT_SCHEMA_VERSION,
        "run_id": RUN_ID,
        "gpu_uuid": GPU_UUID,
        "lock_path": f"{REMOTE_ROOT}/locks/gpu-{GPU_UUID}.lock",
        "lock_st_dev": 1,
        "lock_st_ino": 2,
        "lock_holder_pid": 3,
        "lock_holder_start_ticks": 4,
        "inherited_lock_fd": 5,
        "attempt_nonce": attempt_nonce,
        "boot_id": "00000000-0000-0000-0000-000000000001",
        "acquisition_started_perf_counter_ns": 1_000,
        "eligibility_rechecked_perf_counter_ns": 1_001,
        "handshake_completed_perf_counter_ns": 1_002,
        "wall_ns": handshake_wall,
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
    lock_probe: dict[str, object] = {
        "schema": evidence.PREFLIGHT_LOCK_HANDSHAKE_PROBE_SCHEMA_VERSION,
        "run_id": RUN_ID,
        "acquisition_started_perf_counter_ns": 1_000,
        "handshake_completed_perf_counter_ns": 1_002,
        "wall_ns": handshake_wall,
        "source": "verified_launcher_lock_handshake",
    }
    lock_probe["probe_id"] = canonical_sha256(lock_probe)
    report["lock_handshake_probe"] = lock_probe

    projection.update(
        {
            "lock_handshake_probe_id": lock_probe["probe_id"],
            "fresh_reconstruction_probe_id": report["fresh_reconstruction_probe"][
                "probe_id"
            ],
            "checkpoint_cost_probe_id": report["checkpoint_cost_probe"]["probe_id"],
            "final_selection_probe_id": report["final_selection_probe"]["probe_id"],
        }
    )
    _reseal(projection, "projection_id")

    selection_rows = _cost_probe_selection_rows(diagnostic_sha256)
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
        "run_id": RUN_ID,
        "projection_method": evidence.OVERHEAD_PROJECTION_METHOD,
        "fixed_checkpoint_multiplier": evidence.VALIDATION_PASS_COUNT,
        "lock_handshake_probe_id": lock_probe["probe_id"],
        "measured_lock_setup_handshake_wall_ns": handshake_wall,
        "fresh_reconstruction_probe_id": report["fresh_reconstruction_probe"][
            "probe_id"
        ],
        "measured_fresh_reconstruction_wall_ns": fresh_setup,
        "checkpoint_cost_probe_id": report["checkpoint_cost_probe"]["probe_id"],
        "measured_checkpoint_write_wall_ns": 2,
        "measured_checkpoint_weights_only_load_wall_ns": 3,
        "measured_checkpoint_roundtrip_wall_ns": checkpoint_roundtrip,
        "final_selection_probe_id": report["final_selection_probe"]["probe_id"],
        "measured_final_selection_wall_ns": selection_wall,
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
        "projected_fresh_setup_wall_ns": fresh_setup,
        "projected_final_selection_wall_ns": selection_wall,
        "projected_lock_setup_handshake_wall_ns": handshake_wall,
        "projected_campaign_overhead_wall_ns": (
            handshake_wall
            + fresh_setup
            + evidence.VALIDATION_PASS_COUNT * checkpoint_roundtrip
            + selection_wall
        ),
    }
    overhead_projection["projection_id"] = canonical_sha256(overhead_projection)
    cost_report: dict[str, object] = {
        "schema": evidence.CAMPAIGN_OVERHEAD_PROBE_SCHEMA_VERSION,
        "status": "PASS",
        "run_id": RUN_ID,
        "runtime_mode": "frozen_cuda",
        "optimizer_step": 100,
        "source_optimizer_state": "fully_initialized_step_100",
        "config_sha256": training_config_sha256(),
        "runtime_source_sha256": runtime_source_fingerprint_sha256,
        "test_source_sha256": test_source_fingerprint_sha256,
        "lock_handshake_probe": lock_probe,
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
    return report, cost_report, diagnostic_snapshot, lock_artifact


def _write_files(root: Path, files: dict[str, bytes]) -> dict[str, Path]:
    root.mkdir(parents=True)
    result: dict[str, Path] = {}
    for name, content in files.items():
        path = root / name
        path.write_bytes(content)
        result[name] = path
    return result


class M04aCompleteTrainingBundleTests(unittest.TestCase):
    def test_publish_read_and_externally_committed_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            validation_root = root / "validation"
            validation = publish_validation_episode_manifest(
                validation_root, _validation_episodes()
            )
            validation_outer = (validation_root / "artifact_manifest.json").read_bytes()
            validation_jsonl = validation.jsonl_bytes
            validation_summary = serialize_json(validation.summary)
            validation_outer_sha = hashlib.sha256(validation_outer).hexdigest()
            validation_jsonl_sha = hashlib.sha256(validation_jsonl).hexdigest()
            validation_commitment = build_validation_manifest_commitment(
                validation,
                expected_artifact_manifest_sha256=validation_outer_sha,
                expected_jsonl_sha256=validation_jsonl_sha,
            ).to_json_dict()

            runtime_zip, runtime_fingerprint = _zip_materials(
                {"afts_arc/_source_bootstrap.py": b"BOOTSTRAP_ANCHOR = True\n"}
            )
            test_zip, test_fingerprint = _zip_materials(
                {"tests/test_fixture.py": b"def test_fixture():\n    assert True\n"}
            )
            self.assertNotEqual(
                runtime_fingerprint, hashlib.sha256(runtime_zip).hexdigest()
            )
            self.assertNotEqual(test_fingerprint, hashlib.sha256(test_zip).hexdigest())
            launcher_bytes = b"def main():\n    return 0\n"
            attempt_nonce = _digest("attempt-nonce")
            runtime_fixture = _SyntheticMixbit(root / "runtime-fixture")
            runtime_patches = runtime_fixture.patches()
            runtime_patches.__enter__()
            self.addCleanup(runtime_patches.close)
            python_runtime_lock = runtime_fixture.build()
            mixbit_prefix = "/opt/mixbit"
            python_runtime_lock["environment"].update(
                {
                    "prefix_path": mixbit_prefix,
                    "site_packages_path": (
                        f"{mixbit_prefix}/lib/python3.10/site-packages"
                    ),
                }
            )
            python_runtime_lock["python"].update(
                {
                    "executable_lexical_path": f"{mixbit_prefix}/bin/python",
                    "executable_resolved_path": f"{mixbit_prefix}/bin/python3.10",
                    "executable_symlink_chain": [
                        {
                            "path": f"{mixbit_prefix}/bin/python",
                            "target": "python3.10",
                        }
                    ],
                }
            )
            python_runtime_lock = _reidentify(python_runtime_lock)
            python_runtime_lock_bytes = (
                runtime_lock_module.canonical_python_runtime_lock_bytes(
                    python_runtime_lock
                )
            )
            source_files: dict[str, bytes] = {
                "data_split_manifest.json": serialize_json({"fixture": "data-split"}),
                "frozen_contract.md": b"# Frozen M04a fixture contract\n",
                "model_config.json": serialize_json({"fixture": "model-config"}),
                "ordered_fold_ids.json": serialize_json({"fixture": "fold-ids"}),
                "parameter_count.json": serialize_json({"parameter_count": 8_733_706}),
                "python-runtime-lock.json": python_runtime_lock_bytes,
                "quarantine_parent_ids.json": serialize_json({"parent_ids": []}),
                "remote_launcher.py": launcher_bytes,
                "reviewed_runtime_source.zip": runtime_zip,
                "reviewed_test_snapshot.zip": test_zip,
                "sanitized_shard_manifest.json": serialize_json(
                    {"fixture": "sanitized-shards"}
                ),
                "schema_config_manifest.json": serialize_json(
                    {"fixture": "schema-config"}
                ),
                "seed_policy.json": serialize_json({"seed": 20_260_711}),
                "short_exact_replay_fixture.json": serialize_json(
                    {"fixture": "short-replay"}
                ),
                "validation_episode_manifest.jsonl": validation_jsonl,
                "validation_episode_manifest_summary.json": validation_summary,
                "validation_episode_outer_manifest.json": validation_outer,
            }
            self.assertEqual(set(source_files), EXPECTED_INPUT_ARTIFACT_PATHS)
            expected_inputs = {
                name: hashlib.sha256(content).hexdigest()
                for name, content in source_files.items()
            }
            launch_plan = build_launch_plan(
                attempt_nonce=attempt_nonce,
                run_id=RUN_ID,
                remote_project_root=REMOTE_ROOT,
                run_root=f"{REMOTE_ROOT}/runs/{RUN_ID}",
                expected_input_artifacts=expected_inputs,
                runtime_source_fingerprint_sha256=runtime_fingerprint,
                test_source_fingerprint_sha256=test_fingerprint,
                training_config_sha256=training_config_sha256(),
                validation_manifest_commitment=validation_commitment,
                conda_explicit_sha256=_digest("conda-explicit"),
                ordered_import_roots=[
                    f"{REMOTE_ROOT}/src",
                    f"{mixbit_prefix}/lib/python3.10/site-packages",
                ],
            )
            launch_plan_bytes = canonical_launch_plan_bytes(launch_plan)
            launch_plan_sha = hashlib.sha256(launch_plan_bytes).hexdigest()
            self.assertEqual(launch_plan["schema"], "afts-m04a-launch-plan/v0.2")
            source_files["launch_plan.json"] = launch_plan_bytes
            launch_plan_path = root / "launch_plan.external.json"
            launch_plan_path.write_bytes(launch_plan_bytes)
            self.assertEqual(
                read_launch_plan_artifact(
                    launch_plan_path,
                    expected_artifact_sha256=launch_plan_sha,
                ).payload,
                launch_plan,
            )

            preflight_rows = _preflight_rows()
            preflight, cost_report, diagnostic, lock_artifact = (
                _build_preflight_evidence(
                    preflight_rows,
                    validation_outer_sha256=validation_outer_sha,
                    validation_jsonl_sha256=validation_jsonl_sha,
                    validation_summary_id=str(validation.summary["summary_id"]),
                    validation_row_count=len(validation.rows),
                    launcher_sha256=hashlib.sha256(launcher_bytes).hexdigest(),
                    launch_plan_sha256=launch_plan_sha,
                    runtime_source_fingerprint_sha256=runtime_fingerprint,
                    test_source_fingerprint_sha256=test_fingerprint,
                    attempt_nonce=attempt_nonce,
                )
            )
            self.assertEqual(
                lock_artifact["schema"],
                "afts-m04a-held-gpu-lock-handshake/v0.2",
            )
            self.assertEqual(
                preflight["schema"], "afts-grid-cmlm-preflight-report/v0.4"
            )
            self.assertEqual(cost_report["status"], "PASS")

            selected_checkpoint_bytes = b"selected-checkpoint-index-seven"
            ledger: list[dict[str, object]] = list(preflight_rows)
            checkpoints: list[dict[str, object]] = []
            metrics: list[dict[str, object]] = []
            event_index = len(ledger)
            for optimizer_step in range(1, evidence.OPTIMIZER_UPDATES + 1):
                ledger.append(
                    make_training_cost_ledger_row(
                        phase="primary_update",
                        event_index=event_index,
                        optimizer_step=optimizer_step,
                        optimizer_updates=1,
                        microbatches=16,
                        arc2_episodes=8,
                        rearc_episodes=8,
                        encoder_forward_calls=16,
                        decoder_forward_calls=16,
                        backward_calls=16,
                        masked_token_predictions=16,
                        wall_time_ns=1,
                    )
                )
                event_index += 1
                if optimizer_step % evidence.VALIDATION_INTERVAL:
                    continue
                pass_index = optimizer_step // evidence.VALIDATION_INTERVAL
                validation_events: list[str] = []
                for episode_index, manifest_row in enumerate(validation.rows):
                    masked_count = len(manifest_row["masked_linear_indices"])
                    event = make_training_cost_ledger_row(
                        phase="validation_episode",
                        event_index=event_index,
                        validation_pass_index=pass_index,
                        episode_index=episode_index,
                        validation_episode_calls=1,
                        validation_encoder_forward_calls=1,
                        validation_decoder_forward_calls=1,
                        masked_token_predictions=masked_count,
                        wall_time_ns=1,
                    )
                    ledger.append(event)
                    validation_events.append(str(event["event_id"]))
                    event_index += 1
                checkpoint_content = (
                    selected_checkpoint_bytes
                    if pass_index == evidence.SELECTION_TIE_INDICES[0]
                    else f"checkpoint-{pass_index}".encode("ascii")
                )
                checkpoint_sha = hashlib.sha256(checkpoint_content).hexdigest()
                checkpoint_event = make_training_cost_ledger_row(
                    phase="checkpoint_operation",
                    event_index=event_index,
                    checkpoint_index=pass_index,
                    checkpoint_writes=1,
                    checkpoint_io_bytes=len(checkpoint_content),
                    checkpoint_io_ns=1,
                    wall_time_ns=1,
                )
                ledger.append(checkpoint_event)
                event_index += 1
                ce = (
                    0.25
                    if pass_index in evidence.SELECTION_TIE_INDICES
                    else float(1 + pass_index)
                )
                episode_metrics = [
                    {
                        "ledger_event_id": validation_events[episode_index],
                        "row_ordinal": episode_index,
                        "episode_sha256": manifest_row["episode_sha256"],
                        "semantic_parent_id": manifest_row["semantic_parent_id"],
                        "target_group_id": manifest_row["target_group_id"],
                        "masked_token_predictions": len(
                            manifest_row["masked_linear_indices"]
                        ),
                        "masked_cell_ce_hex": ce.hex(),
                    }
                    for episode_index, manifest_row in enumerate(validation.rows)
                ]
                metric = make_validation_metric_row(
                    validation_pass_index=pass_index,
                    validation_episode_outer_manifest_sha256=validation_outer_sha,
                    validation_episode_manifest_sha256=validation_jsonl_sha,
                    episode_metrics=episode_metrics,
                    checkpoint_sha256=checkpoint_sha,
                )
                metrics.append(metric)
                checkpoints.append(
                    make_checkpoint_manifest_row(
                        checkpoint_index=pass_index,
                        checkpoint_sha256=checkpoint_sha,
                        checkpoint_bytes=len(checkpoint_content),
                        checkpoint_event_id=str(checkpoint_event["event_id"]),
                        validation_metric_id=str(metric["metric_id"]),
                    )
                )

            ledger_bytes = serialize_jsonl(ledger)
            ledger_sha = hashlib.sha256(ledger_bytes).hexdigest()
            checkpoint_bytes = serialize_jsonl(checkpoints)
            metric_bytes = serialize_jsonl(metrics)
            lock_start = int(lock_artifact["acquisition_started_perf_counter_ns"])
            selection_completed = 400_000_001_000
            lock_interval = make_training_lock_interval(
                ((lock_start, selection_completed),)
            )
            selected = make_selected_checkpoint_manifest(
                checkpoint_rows=checkpoints,
                validation_rows=metrics,
                checkpoint_manifests_sha256=hashlib.sha256(
                    checkpoint_bytes
                ).hexdigest(),
                validation_metrics_sha256=hashlib.sha256(metric_bytes).hexdigest(),
                training_cost_ledger_sha256=ledger_sha,
                validation_episode_outer_manifest_sha256=validation_outer_sha,
                validation_episode_manifest_sha256=validation_jsonl_sha,
                lock_interval_id=str(lock_interval["interval_id"]),
                selection_completed_perf_counter_ns=selection_completed,
            )
            self.assertEqual(
                selected["selected_checkpoint_index"],
                evidence.SELECTION_TIE_INDICES[0],
            )
            self.assertEqual(len(checkpoints), evidence.VALIDATION_PASS_COUNT)
            self.assertEqual(len(metrics), evidence.VALIDATION_PASS_COUNT)
            self.assertEqual(
                selected["checkpoint_sha256"],
                hashlib.sha256(selected_checkpoint_bytes).hexdigest(),
            )

            ledger_closure = validate_training_cost_ledger(
                ledger,
                validation_episodes_per_pass=len(validation.rows),
                fresh_setup_wall_time_ns=6,
                lock_interval=lock_interval,
                preflight_endpoint_wall_time_ns=preflight["budget_projection"][
                    "measured_preflight_total_wall_ns"
                ],
            )
            closure = {
                "status": "COMPLETE",
                "preflight_optimizer_updates": ledger_closure[
                    "preflight_optimizer_updates"
                ],
                "preflight_microbatches": ledger_closure["preflight_microbatches"],
                "preflight_encoder_forward_calls": ledger_closure[
                    "preflight_encoder_forward_calls"
                ],
                "preflight_decoder_forward_calls": ledger_closure[
                    "preflight_decoder_forward_calls"
                ],
                "preflight_backward_calls": ledger_closure["preflight_backward_calls"],
                "preflight_inference_lanes": 8,
                "preflight_inference_steps": 12,
                "preflight_inference_encoder_batch_calls": 1,
                "preflight_inference_decoder_batch_calls": 12,
                "preflight_inference_sample_equivalent_forward_calls": 96,
                "primary_optimizer_updates": ledger_closure["optimizer_updates"],
                "primary_microbatches": ledger_closure["microbatches"],
                "primary_arc2_episodes": ledger_closure["arc2_episodes"],
                "primary_rearc_episodes": ledger_closure["rearc_episodes"],
                "primary_encoder_forward_calls": ledger_closure[
                    "encoder_forward_calls"
                ],
                "primary_decoder_forward_calls": ledger_closure[
                    "decoder_forward_calls"
                ],
                "primary_backward_calls": ledger_closure["backward_calls"],
                "validation_passes": ledger_closure["validation_passes"],
                "validation_episode_calls": ledger_closure["validation_episode_calls"],
                "validation_encoder_forward_calls": ledger_closure[
                    "validation_encoder_forward_calls"
                ],
                "validation_decoder_forward_calls": ledger_closure[
                    "validation_decoder_forward_calls"
                ],
                "checkpoint_writes": ledger_closure["checkpoint_writes"],
                "resume_segments": ledger_closure["resume_segments"],
                "masked_token_predictions": ledger_closure["masked_token_predictions"],
                "selected_checkpoint_complete": True,
                "budget_status": "WITHIN_BUDGET",
            }
            timing = {
                field: int(ledger_closure[field])
                for field in evidence.TRAINING_ARTIFACT_TIMING_FIELDS
            }
            training_summary = make_training_cost_summary(
                ledger_sha256=ledger_sha,
                closure=closure,
                timing=timing,
                lock_interval=lock_interval,
            )

            ordered_import_roots = launch_plan["ordered_import_roots"]
            python_runtime_identity_sha256 = evidence.python_runtime_identity_sha256(
                {
                    "implementation": python_runtime_lock["python"]["implementation"],
                    "version": python_runtime_lock["python"]["version"],
                    "executable": python_runtime_lock["python"][
                        "executable_lexical_path"
                    ],
                    "executable_sha256": python_runtime_lock["python"][
                        "executable_sha256"
                    ],
                    "executable_bytes": python_runtime_lock["python"][
                        "executable_bytes"
                    ],
                }
            )
            environment_manifest = {
                "schema": "afts-m04a-environment/v0.3",
                "python": "3.10.20",
                "python_implementation": python_runtime_lock["python"][
                    "implementation"
                ],
                "python_version": python_runtime_lock["python"]["version"],
                "python_executable": python_runtime_lock["python"][
                    "executable_lexical_path"
                ],
                "python_executable_sha256": python_runtime_lock["python"][
                    "executable_sha256"
                ],
                "python_executable_bytes": python_runtime_lock["python"][
                    "executable_bytes"
                ],
                "platform": "Linux-fixture",
                "hostname": "fixture-host",
                "torch": "2.10.0+cu128",
                "torch_cuda": "12.8",
                "cudnn": 91_002,
                "nvidia_driver": "fixture-driver",
                "cuda_visible_devices": GPU_UUID,
                "gpu_uuid": GPU_UUID,
                "run_id": RUN_ID,
                "cublas_workspace_config": ":4096:8",
                "gpu": {
                    "name": "fixture-gpu",
                    "total_memory": 1,
                    "major": 9,
                    "minor": 0,
                },
                "deterministic_algorithms": True,
                "cudnn_benchmark": False,
                "tf32_matmul": False,
                "tf32_cudnn": False,
                "flash_sdp": False,
                "mem_efficient_sdp": False,
                "math_sdp": True,
                "launcher_path": f"{REMOTE_ROOT}/input/remote_launcher.py",
                "launcher_sha256": hashlib.sha256(launcher_bytes).hexdigest(),
                "ordered_sys_path": [
                    "/stdlib",
                    *[
                        f"/proc/self/fd/{200 + index}"
                        for index in range(len(ordered_import_roots))
                    ],
                ],
                "ordered_import_roots": ordered_import_roots,
                "conda_explicit_path": f"{REMOTE_ROOT}/input/conda-explicit.txt",
                "conda_explicit_sha256": launch_plan["conda_explicit_sha256"],
                "visible_root": f"{REMOTE_ROOT}/input",
                "runtime_source_sha256": runtime_fingerprint,
                "test_source_sha256": test_fingerprint,
                "visible_files": {
                    **expected_inputs,
                    "conda-explicit.txt": launch_plan["conda_explicit_sha256"],
                },
                "python_runtime_lock_sha256": hashlib.sha256(
                    python_runtime_lock_bytes
                ).hexdigest(),
                "python_runtime_lock_id": python_runtime_lock["runtime_lock_id"],
            }
            self.assertEqual(
                evidence.validate_environment_manifest_artifact(
                    environment_manifest,
                    launch_plan=launch_plan,
                    expected_gpu_uuid=GPU_UUID,
                    expected_launcher_sha256=hashlib.sha256(launcher_bytes).hexdigest(),
                    python_runtime_lock=python_runtime_lock,
                ),
                environment_manifest,
            )
            wrong_visible_gpu = dict(environment_manifest)
            wrong_visible_gpu["cuda_visible_devices"] = "0"
            with self.assertRaisesRegex(ValueError, "GPU UUID|launch/runtime"):
                evidence.validate_environment_manifest_artifact(
                    wrong_visible_gpu,
                    launch_plan=launch_plan,
                    expected_gpu_uuid=GPU_UUID,
                    expected_launcher_sha256=hashlib.sha256(launcher_bytes).hexdigest(),
                    python_runtime_lock=python_runtime_lock,
                )
            for field, drifted in (
                ("python", "3.10.19"),
                ("torch", "2.10.0"),
                ("torch_cuda", "12.7"),
                ("cudnn", 91_001),
            ):
                wrong_runtime = dict(environment_manifest)
                wrong_runtime[field] = drifted
                with self.assertRaisesRegex(
                    ValueError, "exact CPython|versions drifted"
                ):
                    evidence.validate_environment_manifest_artifact(
                        wrong_runtime,
                        launch_plan=launch_plan,
                        expected_gpu_uuid=GPU_UUID,
                        expected_launcher_sha256=hashlib.sha256(
                            launcher_bytes
                        ).hexdigest(),
                        python_runtime_lock=python_runtime_lock,
                    )
            for invalid_gpu_uuid in (
                "GPU-1234-abcd",
                "GPU-00000000-0000-0000-0000-00000000000A",
            ):
                wrong_gpu = dict(environment_manifest)
                wrong_gpu["gpu_uuid"] = invalid_gpu_uuid
                wrong_gpu["cuda_visible_devices"] = invalid_gpu_uuid
                with self.assertRaisesRegex(
                    ValueError, "GPU UUID|canonical full NVIDIA GPU UUID"
                ):
                    evidence.validate_environment_manifest_artifact(
                        wrong_gpu,
                        launch_plan=launch_plan,
                        expected_gpu_uuid=invalid_gpu_uuid,
                        expected_launcher_sha256=hashlib.sha256(
                            launcher_bytes
                        ).hexdigest(),
                        python_runtime_lock=python_runtime_lock,
                    )

            wrong_python_executable = dict(environment_manifest)
            wrong_python_executable["python_executable"] = "/opt/conda/bin/python"
            with self.assertRaisesRegex(ValueError, "launch/runtime"):
                evidence.validate_environment_manifest_artifact(
                    wrong_python_executable,
                    launch_plan=launch_plan,
                    expected_gpu_uuid=GPU_UUID,
                    expected_launcher_sha256=hashlib.sha256(launcher_bytes).hexdigest(),
                    python_runtime_lock=python_runtime_lock,
                )

            source_files.update(
                {
                    "checkpoint_manifests.jsonl": checkpoint_bytes,
                    "environment_manifest.json": serialize_json(environment_manifest),
                    "lock_handshake_artifact.json": serialize_json(lock_artifact),
                    "optimizer_schedule_state.json": serialize_json(
                        {
                            "schema": (
                                evidence.OPTIMIZER_SCHEDULE_STATE_SCHEMA_VERSION
                            ),
                            "optimizer_step": evidence.OPTIMIZER_UPDATES,
                            "learning_rate_hex": learning_rate_for_update(
                                evidence.OPTIMIZER_UPDATES
                            ).hex(),
                            "checkpoint_count": evidence.VALIDATION_PASS_COUNT,
                            "production_data_closure_id": (
                                evidence.PRODUCTION_DATA_CLOSURE_ID
                            ),
                        }
                    ),
                    "preflight.json": serialize_json(preflight),
                    "preflight_diagnostic_checkpoint.pt": diagnostic,
                    "preflight_overhead_cost_report.json": serialize_json(cost_report),
                    "resume_manifests.jsonl": b"",
                    "selected_checkpoint.pt": selected_checkpoint_bytes,
                    "selected_checkpoint_manifest.json": serialize_json(selected),
                    "training_cost_ledger.jsonl": ledger_bytes,
                    "training_cost_summary.json": serialize_json(training_summary),
                    "validation_metrics.jsonl": metric_bytes,
                }
            )
            self.assertEqual(set(source_files), TRAINING_ARTIFACT_FILES)
            lineage = {
                "reviewed_runtime_source_sha256": hashlib.sha256(
                    runtime_zip
                ).hexdigest(),
                "reviewed_test_snapshot_sha256": hashlib.sha256(test_zip).hexdigest(),
                "remote_launcher_sha256": hashlib.sha256(launcher_bytes).hexdigest(),
                "launch_plan_sha256": launch_plan_sha,
                "lock_handshake_artifact_sha256": hashlib.sha256(
                    source_files["lock_handshake_artifact.json"]
                ).hexdigest(),
                "preflight_overhead_cost_report_sha256": hashlib.sha256(
                    source_files["preflight_overhead_cost_report.json"]
                ).hexdigest(),
                "preflight_diagnostic_checkpoint_sha256": hashlib.sha256(
                    diagnostic
                ).hexdigest(),
                "frozen_contract_sha256": hashlib.sha256(
                    source_files["frozen_contract.md"]
                ).hexdigest(),
                "sanitized_shard_manifest_sha256": hashlib.sha256(
                    source_files["sanitized_shard_manifest.json"]
                ).hexdigest(),
                "data_split_manifest_sha256": hashlib.sha256(
                    source_files["data_split_manifest.json"]
                ).hexdigest(),
                "validation_episode_outer_manifest_sha256": validation_outer_sha,
                "validation_episode_manifest_sha256": validation_jsonl_sha,
                "validation_episode_summary_sha256": hashlib.sha256(
                    validation_summary
                ).hexdigest(),
                "validation_episode_summary_id": str(validation.summary["summary_id"]),
                "environment_manifest_sha256": hashlib.sha256(
                    source_files["environment_manifest.json"]
                ).hexdigest(),
                "model_config_sha256": hashlib.sha256(
                    source_files["model_config.json"]
                ).hexdigest(),
                "training_cost_ledger_sha256": ledger_sha,
                "selected_checkpoint_manifest_sha256": hashlib.sha256(
                    source_files["selected_checkpoint_manifest.json"]
                ).hexdigest(),
                "checkpoint_sha256": hashlib.sha256(
                    selected_checkpoint_bytes
                ).hexdigest(),
                "python_runtime_lock_sha256": hashlib.sha256(
                    python_runtime_lock_bytes
                ).hexdigest(),
                "python_runtime_lock_id": python_runtime_lock["runtime_lock_id"],
                "python_runtime_identity_sha256": python_runtime_identity_sha256,
            }
            artifact_paths = _write_files(root / "payload", source_files)
            bundle_root = root / "training-bundle"
            published = publish_training_artifact_bundle(
                bundle_root,
                lineage=lineage,
                artifact_files=artifact_paths,
                closure=closure,
                timing=timing,
            )
            loaded = read_m04a_artifact_bundle(bundle_root, expected_kind="training")
            committed = read_committed_training_artifact_bundle(
                bundle_root,
                expected_launch_plan_sha256=launch_plan_sha,
            )
            with self.assertRaisesRegex(ValueError, "external launch-plan"):
                read_committed_training_artifact_bundle(
                    bundle_root,
                    expected_launch_plan_sha256="0" * 64,
                )
            self.assertEqual(
                published.manifest["schema"],
                "afts-m04a-training-artifact-manifest/v0.4",
            )
            self.assertEqual(published.manifest, loaded.manifest)
            self.assertEqual(loaded.manifest, committed.manifest)
            self.assertEqual(published.outer_manifest, loaded.outer_manifest)
            self.assertEqual(loaded.outer_manifest, committed.outer_manifest)
            self.assertEqual(
                published.manifest["semantic_id"], loaded.manifest["semantic_id"]
            )
            self.assertEqual(
                loaded.manifest["semantic_id"], committed.manifest["semantic_id"]
            )
            self.assertEqual(
                selected["selection_completed_perf_counter_ns"],
                training_summary["lock_interval"]["segments"][-1][
                    "end_perf_counter_ns"
                ],
            )
            self.assertEqual(
                len(ledger),
                100
                + evidence.OPTIMIZER_UPDATES
                + evidence.VALIDATION_PASS_COUNT * len(validation.rows)
                + evidence.VALIDATION_PASS_COUNT,
            )
            self.assertEqual(
                sum(row["phase"] == "primary_update" for row in ledger),
                evidence.OPTIMIZER_UPDATES,
            )


if __name__ == "__main__":
    unittest.main()
