"""Measured, dataset-free campaign-overhead probes for M04a.

The production entry point is deliberately CUDA-gated.  A separate CPU-fixture
entry point exercises serialization and schema closure without producing frozen
CUDA evidence.  Successful probes retain a content-addressed diagnostic
checkpoint below a caller-created run root; it is never a selectable training
checkpoint.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import os
import re
import stat
import time
from collections.abc import Mapping
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path
from typing import Any

import torch

from .m04a_contract import (
    MODEL_SEMANTICS_VERSION,
    VALIDATION_INTERVAL,
    VALIDATION_PASS_COUNT,
    canonical_sha256,
)
from .m04a_model import GridCMLM
from .m04a_torch_runtime import (
    CHECKPOINT_SCHEMA_VERSION,
    MAX_CHECKPOINT_BYTES,
    RNG_STATE_SCHEMA_VERSION,
    FrozenRuntimeAttestation,
    capture_rng_state,
    load_checkpoint,
    validate_runtime_attestation,
)
from .m04a_train import (
    assert_adamw_invariants,
    assert_grid_cmlm_invariants,
    build_adamw_optimizer,
)
from .m04a_train_contract import (
    CheckpointMetric,
    PREFLIGHT_UPDATES,
    learning_rate_for_update,
    select_checkpoint,
    training_config_sha256,
)


FRESH_RECONSTRUCTION_PROBE_SCHEMA_VERSION = (
    "afts-grid-cmlm-preflight-fresh-reconstruction-probe/v0.1"
)
CHECKPOINT_COST_PROBE_SCHEMA_VERSION = (
    "afts-grid-cmlm-preflight-checkpoint-cost-probe/v0.1"
)
FINAL_SELECTION_PROBE_SCHEMA_VERSION = (
    "afts-grid-cmlm-preflight-final-selection-probe/v0.1"
)
LOCK_HANDSHAKE_PROBE_SCHEMA_VERSION = (
    "afts-grid-cmlm-preflight-lock-handshake-probe/v0.1"
)
OVERHEAD_PROJECTION_SCHEMA_VERSION = (
    "afts-grid-cmlm-campaign-overhead-projection/v0.1"
)
CAMPAIGN_OVERHEAD_PROBE_SCHEMA_VERSION = (
    "afts-grid-cmlm-campaign-overhead-cost-probe/v0.1"
)
SELECTED_METRIC_SUMMARY_SCHEMA_VERSION = (
    "afts-grid-cmlm-preflight-selected-metric-summary/v0.1"
)

CHECKPOINT_SELECTION_RULE = (
    "minimum_parent_grouped_masked_cell_ce_then_earlier_step"
)
CHECKPOINT_PROJECTION_MULTIPLIER = VALIDATION_PASS_COUNT
OVERHEAD_PROJECTION_METHOD = (
    "measured_lock_handshake_plus_measured_fresh_reconstruction_plus_"
    f"{VALIDATION_PASS_COUNT}_measured_checkpoint_writes_plus_"
    f"{VALIDATION_PASS_COUNT}_verified_weights_only_loads_plus_measured_"
    f"{VALIDATION_PASS_COUNT}_metric_final_selection"
)
SELECTION_TIE_INDICES = (
    VALIDATION_PASS_COUNT - 3,
    VALIDATION_PASS_COUNT - 1,
)
DIAGNOSTIC_DISPOSITION = "retained_nonselectable_diagnostic"
DIAGNOSTIC_STATUS = "NOT_TRAINING_CHECKPOINT/diagnostic"
PRODUCTION_RUNTIME_MODE = "frozen_cuda"
CPU_FIXTURE_RUNTIME_MODE = "cpu_fixture_not_frozen_cuda_evidence"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_REPARSE_POINT_ATTRIBUTE = 0x400


def _strict_int(value: object, *, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise TypeError(f"{field} must be an integer >= {minimum}")
    return value


def _sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _exact_dict(
    value: object, fields: set[str], *, label: str
) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError(f"{label} must be an exact dictionary")
    if set(value) != fields:
        raise ValueError(f"{label} fields do not match the frozen schema")
    return dict(value)


def _sealed(payload: dict[str, Any], *, id_field: str) -> dict[str, Any]:
    if id_field in payload:
        raise ValueError(f"{id_field} must not exist before sealing")
    result = dict(payload)
    result[id_field] = canonical_sha256(result)
    return result


def _validate_seal(payload: Mapping[str, Any], *, id_field: str) -> None:
    semantic = dict(payload)
    identifier = semantic.pop(id_field, None)
    if identifier != canonical_sha256(semantic):
        raise ValueError(f"{id_field} does not match canonical content")


def validate_lock_handshake_probe(
    probe: object, *, expected_run_id: str | None = None
) -> dict[str, Any]:
    payload = _exact_dict(
        probe,
        {
            "schema",
            "probe_id",
            "run_id",
            "acquisition_started_perf_counter_ns",
            "handshake_completed_perf_counter_ns",
            "wall_ns",
            "source",
        },
        label="M04a lock-handshake probe",
    )
    if payload["schema"] != LOCK_HANDSHAKE_PROBE_SCHEMA_VERSION:
        raise ValueError("unsupported lock-handshake probe schema")
    run_id = payload["run_id"]
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("lock-handshake run_id must be non-empty")
    if expected_run_id is not None and run_id != expected_run_id:
        raise ValueError("lock-handshake probe belongs to another run")
    started = _strict_int(
        payload["acquisition_started_perf_counter_ns"],
        field="acquisition_started_perf_counter_ns",
    )
    completed = _strict_int(
        payload["handshake_completed_perf_counter_ns"],
        field="handshake_completed_perf_counter_ns",
        minimum=1,
    )
    wall_ns = _strict_int(payload["wall_ns"], field="wall_ns", minimum=1)
    if completed <= started or wall_ns != completed - started:
        raise ValueError("lock-handshake endpoints do not close")
    if payload["source"] != "verified_launcher_lock_handshake":
        raise ValueError("lock-handshake source is not the verified launcher")
    _validate_seal(payload, id_field="probe_id")
    return payload


def validate_fresh_reconstruction_probe(
    probe: object, *, expected_run_id: str | None = None
) -> dict[str, Any]:
    payload = _exact_dict(
        probe,
        {
            "schema",
            "probe_id",
            "run_id",
            "model_construct_wall_ns",
            "cuda_transfer_wall_ns",
            "optimizer_construct_wall_ns",
            "total_wall_ns",
            "state_discarded",
        },
        label="M04a fresh-reconstruction probe",
    )
    if payload["schema"] != FRESH_RECONSTRUCTION_PROBE_SCHEMA_VERSION:
        raise ValueError("unsupported fresh-reconstruction probe schema")
    if not isinstance(payload["run_id"], str) or not payload["run_id"]:
        raise ValueError("fresh-reconstruction run_id must be non-empty")
    if expected_run_id is not None and payload["run_id"] != expected_run_id:
        raise ValueError("fresh-reconstruction probe belongs to another run")
    components = tuple(
        _strict_int(payload[field], field=field)
        for field in (
            "model_construct_wall_ns",
            "cuda_transfer_wall_ns",
            "optimizer_construct_wall_ns",
        )
    )
    total = _strict_int(payload["total_wall_ns"], field="total_wall_ns", minimum=1)
    if total != sum(components):
        raise ValueError("fresh-reconstruction timings do not close")
    if payload["state_discarded"] is not True:
        raise ValueError("fresh-reconstruction state was not discarded")
    _validate_seal(payload, id_field="probe_id")
    return payload


def validate_checkpoint_cost_probe(
    probe: object, *, expected_run_id: str | None = None
) -> dict[str, Any]:
    payload = _exact_dict(
        probe,
        {
            "schema",
            "probe_id",
            "probe_scope",
            "run_id",
            "checkpoint_sha256",
            "checkpoint_bytes",
            "write_wall_ns",
            "weights_only_load_wall_ns",
            "roundtrip_wall_ns",
            "weights_only_load",
            "artifact_disposition",
            "selectable_checkpoint_created",
        },
        label="M04a checkpoint-cost probe",
    )
    if payload["schema"] != CHECKPOINT_COST_PROBE_SCHEMA_VERSION:
        raise ValueError("unsupported checkpoint-cost probe schema")
    if payload["probe_scope"] != "content_addressed_independent_probe":
        raise ValueError("checkpoint-cost probe scope is unsupported")
    if not isinstance(payload["run_id"], str) or not payload["run_id"]:
        raise ValueError("checkpoint-cost run_id must be non-empty")
    if expected_run_id is not None and payload["run_id"] != expected_run_id:
        raise ValueError("checkpoint-cost probe belongs to another run")
    _sha256(payload["checkpoint_sha256"], field="checkpoint_sha256")
    _strict_int(payload["checkpoint_bytes"], field="checkpoint_bytes", minimum=1)
    write_ns = _strict_int(
        payload["write_wall_ns"], field="write_wall_ns", minimum=1
    )
    load_ns = _strict_int(
        payload["weights_only_load_wall_ns"],
        field="weights_only_load_wall_ns",
        minimum=1,
    )
    if payload["roundtrip_wall_ns"] != write_ns + load_ns:
        raise ValueError("checkpoint-cost roundtrip timing does not close")
    if payload["weights_only_load"] is not True:
        raise ValueError("checkpoint-cost load must use weights_only=True")
    if payload["artifact_disposition"] != DIAGNOSTIC_DISPOSITION:
        raise ValueError("checkpoint diagnostic disposition is invalid")
    if payload["selectable_checkpoint_created"] is not False:
        raise ValueError("checkpoint-cost probe created a selectable checkpoint")
    _validate_seal(payload, id_field="probe_id")
    return payload


def validate_final_selection_probe(
    probe: object, *, expected_run_id: str | None = None
) -> dict[str, Any]:
    payload = _exact_dict(
        probe,
        {
            "schema",
            "probe_id",
            "run_id",
            "candidate_count",
            "selection_rule",
            "selected_checkpoint_index",
            "selection_wall_ns",
        },
        label="M04a final-selection probe",
    )
    if payload["schema"] != FINAL_SELECTION_PROBE_SCHEMA_VERSION:
        raise ValueError("unsupported final-selection probe schema")
    if not isinstance(payload["run_id"], str) or not payload["run_id"]:
        raise ValueError("final-selection run_id must be non-empty")
    if expected_run_id is not None and payload["run_id"] != expected_run_id:
        raise ValueError("final-selection probe belongs to another run")
    if payload["candidate_count"] != VALIDATION_PASS_COUNT:
        raise ValueError(
            "final selection must receive exactly "
            f"{VALIDATION_PASS_COUNT} candidates"
        )
    if payload["selection_rule"] != CHECKPOINT_SELECTION_RULE:
        raise ValueError("final-selection rule drifted")
    selected = _strict_int(
        payload["selected_checkpoint_index"],
        field="selected_checkpoint_index",
        minimum=1,
    )
    if selected > VALIDATION_PASS_COUNT:
        raise ValueError("selected checkpoint index is out of range")
    _strict_int(
        payload["selection_wall_ns"], field="selection_wall_ns", minimum=1
    )
    _validate_seal(payload, id_field="probe_id")
    return payload


_OVERHEAD_PROJECTION_FIELDS = {
    "schema",
    "projection_id",
    "run_id",
    "projection_method",
    "fixed_checkpoint_multiplier",
    "lock_handshake_probe_id",
    "measured_lock_setup_handshake_wall_ns",
    "fresh_reconstruction_probe_id",
    "measured_fresh_reconstruction_wall_ns",
    "checkpoint_cost_probe_id",
    "measured_checkpoint_write_wall_ns",
    "measured_checkpoint_weights_only_load_wall_ns",
    "measured_checkpoint_roundtrip_wall_ns",
    "final_selection_probe_id",
    "measured_final_selection_wall_ns",
    "projected_checkpoint_writes",
    "projected_checkpoint_loads",
    "projected_checkpoint_write_wall_ns",
    "projected_checkpoint_weights_only_load_wall_ns",
    "projected_checkpoint_roundtrip_wall_ns",
    "projected_fresh_setup_wall_ns",
    "projected_final_selection_wall_ns",
    "projected_lock_setup_handshake_wall_ns",
    "projected_campaign_overhead_wall_ns",
}


def validate_overhead_projection(
    projection: object, *, expected_run_id: str | None = None
) -> dict[str, Any]:
    payload = _exact_dict(
        projection,
        _OVERHEAD_PROJECTION_FIELDS,
        label="M04a campaign-overhead projection",
    )
    if payload["schema"] != OVERHEAD_PROJECTION_SCHEMA_VERSION:
        raise ValueError("unsupported campaign-overhead projection schema")
    if not isinstance(payload["run_id"], str) or not payload["run_id"]:
        raise ValueError("overhead projection run_id must be non-empty")
    if expected_run_id is not None and payload["run_id"] != expected_run_id:
        raise ValueError("overhead projection belongs to another run")
    if payload["projection_method"] != OVERHEAD_PROJECTION_METHOD:
        raise ValueError("overhead projection method drifted")
    if payload["fixed_checkpoint_multiplier"] != CHECKPOINT_PROJECTION_MULTIPLIER:
        raise ValueError(
            "checkpoint projection multiplier must be exactly "
            f"{CHECKPOINT_PROJECTION_MULTIPLIER}"
        )
    for field in (
        "lock_handshake_probe_id",
        "fresh_reconstruction_probe_id",
        "checkpoint_cost_probe_id",
        "final_selection_probe_id",
    ):
        _sha256(payload[field], field=field)
    handshake = _strict_int(
        payload["measured_lock_setup_handshake_wall_ns"],
        field="measured_lock_setup_handshake_wall_ns",
        minimum=1,
    )
    fresh = _strict_int(
        payload["measured_fresh_reconstruction_wall_ns"],
        field="measured_fresh_reconstruction_wall_ns",
        minimum=1,
    )
    write = _strict_int(
        payload["measured_checkpoint_write_wall_ns"],
        field="measured_checkpoint_write_wall_ns",
        minimum=1,
    )
    load = _strict_int(
        payload["measured_checkpoint_weights_only_load_wall_ns"],
        field="measured_checkpoint_weights_only_load_wall_ns",
        minimum=1,
    )
    roundtrip = _strict_int(
        payload["measured_checkpoint_roundtrip_wall_ns"],
        field="measured_checkpoint_roundtrip_wall_ns",
        minimum=1,
    )
    selection = _strict_int(
        payload["measured_final_selection_wall_ns"],
        field="measured_final_selection_wall_ns",
        minimum=1,
    )
    if roundtrip != write + load:
        raise ValueError("measured checkpoint roundtrip does not close")
    multiplier = CHECKPOINT_PROJECTION_MULTIPLIER
    expected = {
        "projected_checkpoint_writes": multiplier,
        "projected_checkpoint_loads": multiplier,
        "projected_checkpoint_write_wall_ns": multiplier * write,
        "projected_checkpoint_weights_only_load_wall_ns": multiplier * load,
        "projected_checkpoint_roundtrip_wall_ns": multiplier * roundtrip,
        "projected_fresh_setup_wall_ns": fresh,
        "projected_final_selection_wall_ns": selection,
        "projected_lock_setup_handshake_wall_ns": handshake,
        "projected_campaign_overhead_wall_ns": (
            handshake + fresh + multiplier * roundtrip + selection
        ),
    }
    if any(payload[field] != value for field, value in expected.items()):
        raise ValueError("campaign-overhead projection does not close")
    _validate_seal(payload, id_field="projection_id")
    return payload


_REPORT_FIELDS = {
    "schema",
    "probe_id",
    "status",
    "run_id",
    "runtime_mode",
    "optimizer_step",
    "source_optimizer_state",
    "config_sha256",
    "runtime_source_sha256",
    "test_source_sha256",
    "lock_handshake_probe",
    "fresh_reconstruction_probe",
    "checkpoint_cost_probe",
    "final_selection_probe",
    "selection_input_sha256",
    "selected_metric_summary",
    "overhead_projection",
    "checkpoint_artifact_filename",
    "checkpoint_artifact_status",
    "dataset_file_reads",
    "fallback_used",
}


def validate_campaign_overhead_cost_probe_report(
    report: object,
    *,
    expected_run_id: str | None = None,
    expected_config_sha256: str | None = None,
    expected_runtime_source_sha256: str | None = None,
    expected_test_source_sha256: str | None = None,
    require_production_cuda: bool = False,
) -> dict[str, Any]:
    payload = _exact_dict(
        report, _REPORT_FIELDS, label="M04a campaign-overhead cost probe"
    )
    if payload["schema"] != CAMPAIGN_OVERHEAD_PROBE_SCHEMA_VERSION:
        raise ValueError("unsupported campaign-overhead cost-probe schema")
    if payload["status"] != "PASS":
        raise ValueError("campaign-overhead cost probe is not successful")
    run_id = payload["run_id"]
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("campaign-overhead run_id must be non-empty")
    if expected_run_id is not None and run_id != expected_run_id:
        raise ValueError("campaign-overhead probe belongs to another run")
    runtime_mode = payload["runtime_mode"]
    if runtime_mode not in {PRODUCTION_RUNTIME_MODE, CPU_FIXTURE_RUNTIME_MODE}:
        raise ValueError("campaign-overhead runtime mode is invalid")
    if require_production_cuda and expected_run_id is None:
        raise ValueError("frozen CUDA evidence requires an external expected_run_id")
    if require_production_cuda and runtime_mode != PRODUCTION_RUNTIME_MODE:
        raise ValueError("CPU fixture output is not frozen CUDA evidence")
    external_source_expectations = {
        "config_sha256": expected_config_sha256,
        "runtime_source_sha256": expected_runtime_source_sha256,
        "test_source_sha256": expected_test_source_sha256,
    }
    if require_production_cuda and any(
        value is None for value in external_source_expectations.values()
    ):
        raise ValueError(
            "frozen CUDA evidence requires external config/runtime/test commitments"
        )
    if runtime_mode == PRODUCTION_RUNTIME_MODE and run_id.startswith("cpu-fixture:"):
        raise ValueError("CPU fixture run_id cannot be relabeled as frozen CUDA")
    if runtime_mode == CPU_FIXTURE_RUNTIME_MODE and not run_id.startswith(
        "cpu-fixture:"
    ):
        raise ValueError("CPU fixture run_id must be visibly non-production")
    if payload["optimizer_step"] != PREFLIGHT_UPDATES:
        raise ValueError("cost probe must bind the completed 100-step optimizer")
    if payload["source_optimizer_state"] != "fully_initialized_step_100":
        raise ValueError("source optimizer state is not fully initialized")
    for field in (
        "config_sha256",
        "runtime_source_sha256",
        "test_source_sha256",
    ):
        _sha256(payload[field], field=field)
        expected = external_source_expectations[field]
        if expected is not None and payload[field] != _sha256(
            expected, field=f"expected_{field}"
        ):
            raise ValueError(f"campaign-overhead {field} differs from external commitment")
    if payload["config_sha256"] != training_config_sha256():
        raise ValueError("training config SHA-256 drifted")
    handshake = validate_lock_handshake_probe(
        payload["lock_handshake_probe"], expected_run_id=run_id
    )
    fresh = validate_fresh_reconstruction_probe(
        payload["fresh_reconstruction_probe"], expected_run_id=run_id
    )
    checkpoint = validate_checkpoint_cost_probe(
        payload["checkpoint_cost_probe"], expected_run_id=run_id
    )
    selection = validate_final_selection_probe(
        payload["final_selection_probe"], expected_run_id=run_id
    )
    selection_rows = _selection_input_payloads(checkpoint["checkpoint_sha256"])
    selection_input_sha256 = canonical_sha256(selection_rows)
    if payload["selection_input_sha256"] != selection_input_sha256:
        raise ValueError(
            "final-selection input SHA-256 is not the exact "
            f"{VALIDATION_PASS_COUNT} metrics"
        )
    selected_index = int(selection["selected_checkpoint_index"])
    expected_metrics = _synthetic_checkpoint_metrics(
        checkpoint["checkpoint_sha256"]
    )
    expected_selected = select_checkpoint(expected_metrics)
    expected_selected_index = expected_metrics.index(expected_selected) + 1
    if selected_index != expected_selected_index:
        raise ValueError("final selection violates the frozen argmin/tie-break rule")
    selected_row = selection_rows[selected_index - 1]
    selected_summary = _exact_dict(
        payload["selected_metric_summary"],
        {
            "schema",
            "summary_id",
            "selection_input_sha256",
            "final_selection_probe_id",
            "checkpoint_index",
            "optimizer_step",
            "parent_grouped_ce_hex",
            "checkpoint_sha256",
        },
        label="M04a selected-metric summary",
    )
    expected_selected_summary = {
        "schema": SELECTED_METRIC_SUMMARY_SCHEMA_VERSION,
        "selection_input_sha256": selection_input_sha256,
        "final_selection_probe_id": selection["probe_id"],
        **selected_row,
    }
    if any(
        selected_summary[field] != value
        for field, value in expected_selected_summary.items()
    ):
        raise ValueError("selected-metric summary does not bind the chosen input row")
    _validate_seal(selected_summary, id_field="summary_id")
    projection = validate_overhead_projection(
        payload["overhead_projection"], expected_run_id=run_id
    )
    child_bindings = {
        "lock_handshake_probe_id": handshake["probe_id"],
        "fresh_reconstruction_probe_id": fresh["probe_id"],
        "checkpoint_cost_probe_id": checkpoint["probe_id"],
        "final_selection_probe_id": selection["probe_id"],
    }
    if any(projection[field] != value for field, value in child_bindings.items()):
        raise ValueError("overhead projection does not bind its measured probes")
    measured_bindings = {
        "measured_lock_setup_handshake_wall_ns": handshake["wall_ns"],
        "measured_fresh_reconstruction_wall_ns": fresh["total_wall_ns"],
        "measured_checkpoint_write_wall_ns": checkpoint["write_wall_ns"],
        "measured_checkpoint_weights_only_load_wall_ns": checkpoint[
            "weights_only_load_wall_ns"
        ],
        "measured_checkpoint_roundtrip_wall_ns": checkpoint["roundtrip_wall_ns"],
        "measured_final_selection_wall_ns": selection["selection_wall_ns"],
    }
    if any(
        projection[field] != value for field, value in measured_bindings.items()
    ):
        raise ValueError("overhead projection timings do not bind measured probes")
    checkpoint_sha = checkpoint["checkpoint_sha256"]
    expected_filename = f"NOT_TRAINING_CHECKPOINT.diagnostic.{checkpoint_sha}.pt"
    if payload["checkpoint_artifact_filename"] != expected_filename:
        raise ValueError("diagnostic checkpoint filename is not content addressed")
    if payload["checkpoint_artifact_status"] != DIAGNOSTIC_STATUS:
        raise ValueError("diagnostic checkpoint status is invalid")
    if payload["dataset_file_reads"] != 0 or payload["fallback_used"] is not False:
        raise ValueError("cost probe isolation closure failed")
    _validate_seal(payload, id_field="probe_id")
    return payload


def _path_has_reparse_point(path: Path) -> bool:
    info = path.lstat()
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & _REPARSE_POINT_ATTRIBUTE
    )


def _assert_existing_nonsymlink_chain(path: Path, *, label: str) -> None:
    parts = path.parts
    if not parts:
        raise ValueError(f"{label} path is empty")
    current = Path(parts[0])
    if not os.path.lexists(current):
        raise ValueError(f"{label} ancestor does not exist")
    if _path_has_reparse_point(current):
        raise ValueError(f"{label} path cannot traverse a symlink or reparse point")
    for part in parts[1:]:
        current = current / part
        if not os.path.lexists(current):
            raise ValueError(f"{label} ancestor does not exist")
        if _path_has_reparse_point(current):
            raise ValueError(
                f"{label} path cannot traverse a symlink or reparse point"
            )


def _absolute_lexical_path(value: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(Path(value).expanduser())))


def _prepare_scratch_paths(
    run_root: str | Path, scratch_dir: str | Path
) -> tuple[Path, Path, os.stat_result]:
    root = _absolute_lexical_path(run_root)
    scratch = _absolute_lexical_path(scratch_dir)
    _assert_existing_nonsymlink_chain(root, label="run_root")
    if not root.is_dir():
        raise ValueError("run_root must be an existing directory")
    _assert_existing_nonsymlink_chain(scratch.parent, label="scratch parent")
    if not scratch.parent.is_dir():
        raise ValueError("scratch parent must be an existing directory")
    if os.path.lexists(scratch):
        raise FileExistsError("scratch_dir must be exclusive and fresh")
    try:
        common = os.path.commonpath((os.fspath(root), os.fspath(scratch)))
    except ValueError as exc:
        raise ValueError("scratch_dir is not on the run_root filesystem") from exc
    if os.path.normcase(common) != os.path.normcase(os.fspath(root)):
        raise ValueError("scratch_dir must be contained below run_root")
    root_stat = root.stat(follow_symlinks=False)
    parent_stat = scratch.parent.stat(follow_symlinks=False)
    if root_stat.st_dev != parent_stat.st_dev:
        raise ValueError("scratch_dir parent is not on the run_root filesystem")
    return root, scratch, root_stat


def _create_exclusive_scratch_dir(
    root: Path, scratch: Path, root_stat: os.stat_result
) -> int | None:
    if os.name != "posix":
        os.mkdir(scratch, mode=0o700)
        created_stat = scratch.stat(follow_symlinks=False)
        if (
            not stat.S_ISDIR(created_stat.st_mode)
            or created_stat.st_dev != root_stat.st_dev
        ):
            raise RuntimeError("exclusive scratch_dir identity or filesystem changed")
        return None
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    relative_parts = scratch.relative_to(root).parts
    if not relative_parts:
        raise ValueError("scratch_dir cannot equal run_root")
    current_fd = os.open(root, flags)
    try:
        if not os.path.samestat(os.fstat(current_fd), root_stat):
            raise RuntimeError("run_root identity changed while opening")
        for component in relative_parts[:-1]:
            next_fd = os.open(component, flags, dir_fd=current_fd)
            next_stat = os.fstat(next_fd)
            if (
                not stat.S_ISDIR(next_stat.st_mode)
                or next_stat.st_dev != root_stat.st_dev
            ):
                os.close(next_fd)
                raise RuntimeError("scratch parent chain escaped the run_root filesystem")
            os.close(current_fd)
            current_fd = next_fd
        os.mkdir(relative_parts[-1], mode=0o700, dir_fd=current_fd)
        os.fsync(current_fd)
        directory_fd = os.open(relative_parts[-1], flags, dir_fd=current_fd)
        created_stat = os.fstat(directory_fd)
        if (
            not stat.S_ISDIR(created_stat.st_mode)
            or created_stat.st_dev != root_stat.st_dev
        ):
            os.close(directory_fd)
            raise RuntimeError("exclusive scratch_dir identity or filesystem changed")
        os.fsync(directory_fd)
        return directory_fd
    finally:
        os.close(current_fd)


def _rename_no_replace(
    *,
    scratch: Path,
    directory_fd: int | None,
    source_name: str,
    target_name: str,
) -> None:
    source = scratch / source_name
    target = scratch / target_name
    if os.name == "posix":
        if directory_fd is None:
            raise RuntimeError("POSIX content-address rename lacks a directory handle")
        library = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(library, "renameat2", None)
        if renameat2 is None:
            raise RuntimeError("renameat2(RENAME_NOREPLACE) is required")
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            directory_fd,
            os.fsencode(source_name),
            directory_fd,
            os.fsencode(target_name),
            1,
        )
        if result != 0:
            error_number = ctypes.get_errno()
            if error_number == errno.EEXIST:
                raise FileExistsError("content-addressed checkpoint already exists")
            raise OSError(error_number, os.strerror(error_number), os.fspath(target))
    else:
        os.rename(source, target)


def _open_staging_file(
    *, scratch: Path, directory_fd: int | None, staging_name: str
) -> int:
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    if directory_fd is not None:
        return os.open(staging_name, flags, 0o600, dir_fd=directory_fd)
    return os.open(scratch / staging_name, flags, 0o600)


def _checkpoint_payload(
    model: GridCMLM,
    optimizer: torch.optim.AdamW,
    *,
    config_sha256: str,
    runtime_source_sha256: str,
    test_source_sha256: str,
    runtime_mode: str,
) -> dict[str, object]:
    if runtime_mode == PRODUCTION_RUNTIME_MODE:
        rng_state = capture_rng_state(optimizer_step=PREFLIGHT_UPDATES)
    else:
        rng_state = {
            "schema": RNG_STATE_SCHEMA_VERSION,
            "optimizer_step": PREFLIGHT_UPDATES,
            "torch_cpu": torch.get_rng_state().clone(),
            "torch_cuda": [],
        }
    return {
        "schema": CHECKPOINT_SCHEMA_VERSION,
        "model_semantics_version": MODEL_SEMANTICS_VERSION,
        "optimizer_step": PREFLIGHT_UPDATES,
        "model_state": dict(model.state_dict()),
        "optimizer_state": dict(optimizer.state_dict()),
        "rng_state": rng_state,
        "config_sha256": config_sha256,
        "runtime_source_sha256": runtime_source_sha256,
        "test_source_sha256": test_source_sha256,
    }


@dataclass(frozen=True, slots=True)
class _StagedDiagnosticCheckpoint:
    staging_path: Path
    final_path: Path
    checkpoint_sha256: str
    checkpoint_bytes: int
    write_wall_ns: int
    st_dev: int
    st_ino: int
    st_mtime_ns: int


def _stage_content_addressed_checkpoint(
    *,
    scratch: Path,
    directory_fd: int | None,
    model: GridCMLM,
    optimizer: torch.optim.AdamW,
    config_sha256: str,
    runtime_source_sha256: str,
    test_source_sha256: str,
    runtime_mode: str,
) -> _StagedDiagnosticCheckpoint:
    staging_name = ".NOT_TRAINING_CHECKPOINT.diagnostic.partial"
    write_started = time.perf_counter_ns()
    file_descriptor = _open_staging_file(
        scratch=scratch,
        directory_fd=directory_fd,
        staging_name=staging_name,
    )
    digest = hashlib.sha256()
    checkpoint_bytes = 0
    with os.fdopen(file_descriptor, "w+b") as handle:
        payload = _checkpoint_payload(
            model,
            optimizer,
            config_sha256=config_sha256,
            runtime_source_sha256=runtime_source_sha256,
            test_source_sha256=test_source_sha256,
            runtime_mode=runtime_mode,
        )
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
        written_stat = os.fstat(handle.fileno())
        if not stat.S_ISREG(written_stat.st_mode) or written_stat.st_size <= 0:
            raise RuntimeError("diagnostic checkpoint write did not create a regular file")
        handle.seek(0, os.SEEK_SET)
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
        final_stat = os.fstat(handle.fileno())
        for field in ("st_dev", "st_ino", "st_size", "st_mtime_ns"):
            if getattr(final_stat, field) != getattr(written_stat, field):
                raise RuntimeError("diagnostic checkpoint changed while hashing")
        checkpoint_bytes = int(final_stat.st_size)
    checkpoint_sha256 = digest.hexdigest()
    filename = f"NOT_TRAINING_CHECKPOINT.diagnostic.{checkpoint_sha256}.pt"
    if directory_fd is not None:
        # Make the uncommitted staging entry durable before timing ends.  The final
        # name is deliberately not published until every fallible probe check and
        # caller-supplied lock check has succeeded.
        os.fsync(directory_fd)
    write_wall_ns = time.perf_counter_ns() - write_started
    if write_wall_ns <= 0:
        raise RuntimeError("checkpoint write timer did not advance")
    return _StagedDiagnosticCheckpoint(
        staging_path=scratch / staging_name,
        final_path=scratch / filename,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_bytes=checkpoint_bytes,
        write_wall_ns=write_wall_ns,
        st_dev=int(final_stat.st_dev),
        st_ino=int(final_stat.st_ino),
        st_mtime_ns=int(final_stat.st_mtime_ns),
    )


def _commit_staged_checkpoint(
    staged: _StagedDiagnosticCheckpoint,
    *,
    scratch: Path,
    directory_fd: int | None,
) -> None:
    """Publish a fully verified checkpoint as the probe's final fallible action."""

    if staged.staging_path.parent != scratch or staged.final_path.parent != scratch:
        raise ValueError("staged diagnostic checkpoint escaped its scratch directory")
    path_stat = staged.staging_path.stat(follow_symlinks=False)
    if (
        not stat.S_ISREG(path_stat.st_mode)
        or path_stat.st_size != staged.checkpoint_bytes
        or path_stat.st_dev != staged.st_dev
        or path_stat.st_ino != staged.st_ino
        or path_stat.st_mtime_ns != staged.st_mtime_ns
    ):
        raise RuntimeError("staged diagnostic checkpoint identity changed before commit")
    _rename_no_replace(
        scratch=scratch,
        directory_fd=directory_fd,
        source_name=staged.staging_path.name,
        target_name=staged.final_path.name,
    )


def _validate_loaded_optimizer_state(
    optimizer_state: object, *, parameter_count: int
) -> None:
    if type(optimizer_state) is not dict or set(optimizer_state) != {
        "state",
        "param_groups",
    }:
        raise ValueError("loaded optimizer state_dict schema is invalid")
    state = optimizer_state["state"]
    groups = optimizer_state["param_groups"]
    if type(state) is not dict or type(groups) is not list or len(groups) != 1:
        raise ValueError("loaded optimizer state is not one complete group")
    parameter_ids = groups[0].get("params") if type(groups[0]) is dict else None
    if type(parameter_ids) is not list or len(parameter_ids) != parameter_count:
        raise ValueError("loaded optimizer parameter list is incomplete")
    if set(parameter_ids) != set(state) or len(set(parameter_ids)) != parameter_count:
        raise ValueError("loaded optimizer state is only partially initialized")
    for parameter_id in parameter_ids:
        entry = state[parameter_id]
        if type(entry) is not dict or set(entry) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("loaded AdamW state fields drifted")
        step = entry["step"]
        exp_avg = entry["exp_avg"]
        exp_avg_sq = entry["exp_avg_sq"]
        if (
            not isinstance(step, torch.Tensor)
            or step.dtype != torch.float32
            or step.device.type != "cpu"
            or step.numel() != 1
            or float(step.item()) != float(PREFLIGHT_UPDATES)
        ):
            raise ValueError("loaded AdamW step is not the completed preflight step")
        if (
            not isinstance(exp_avg, torch.Tensor)
            or not isinstance(exp_avg_sq, torch.Tensor)
            or exp_avg.dtype != torch.float32
            or exp_avg_sq.dtype != torch.float32
            or exp_avg.device.type != "cpu"
            or exp_avg_sq.device.type != "cpu"
            or exp_avg.shape != exp_avg_sq.shape
        ):
            raise ValueError("loaded AdamW moments are not complete CPU FP32 tensors")


def _verified_weights_only_load(
    checkpoint_path: Path,
    *,
    checkpoint_sha256: str,
    checkpoint_bytes: int,
    config_sha256: str,
    runtime_source_sha256: str,
    test_source_sha256: str,
    expected_model_state_keys: tuple[str, ...],
) -> int:
    load_started = time.perf_counter_ns()
    verification_model = GridCMLM().train()
    verification_optimizer = build_adamw_optimizer(verification_model)
    payload = load_checkpoint(
        checkpoint_path,
        expected_sha256=checkpoint_sha256,
        expected_optimizer_step=PREFLIGHT_UPDATES,
        expected_config_sha256=config_sha256,
        expected_runtime_source_sha256=runtime_source_sha256,
        expected_test_source_sha256=test_source_sha256,
        expected_bytes=checkpoint_bytes,
        map_location="cpu",
    )
    if tuple(payload["model_state"]) != expected_model_state_keys:
        raise ValueError("loaded checkpoint model state keys drifted")
    incompatibility = verification_model.load_state_dict(
        payload["model_state"], strict=True
    )
    if incompatibility.missing_keys or incompatibility.unexpected_keys:
        raise ValueError("strict checkpoint model load reported incompatible keys")
    verification_optimizer.load_state_dict(payload["optimizer_state"])
    parameter_count = len(tuple(verification_model.parameters()))
    _validate_loaded_optimizer_state(
        payload["optimizer_state"], parameter_count=parameter_count
    )
    assert_grid_cmlm_invariants(verification_model, require_cuda=False)
    assert_adamw_invariants(
        verification_optimizer,
        verification_model,
        expected_learning_rate=learning_rate_for_update(PREFLIGHT_UPDATES),
        expected_completed_updates=PREFLIGHT_UPDATES,
    )
    del verification_optimizer
    del verification_model
    del payload
    load_wall_ns = time.perf_counter_ns() - load_started
    if load_wall_ns <= 0:
        raise RuntimeError("weights-only load timer did not advance")
    return load_wall_ns


def _run_fresh_reconstruction(
    *, run_id: str, runtime_mode: str, attestation: FrozenRuntimeAttestation | None
) -> dict[str, Any]:
    model_started = time.perf_counter_ns()
    fresh_model = GridCMLM()
    model_wall_ns = time.perf_counter_ns() - model_started
    transfer_wall_ns = 0
    if runtime_mode == PRODUCTION_RUNTIME_MODE:
        if attestation is None:
            raise RuntimeError("frozen CUDA reconstruction requires an attestation")
        validate_runtime_attestation(attestation)
        torch.cuda.synchronize()
        transfer_started = time.perf_counter_ns()
        fresh_model = fresh_model.to(device="cuda:0").train()
        torch.cuda.synchronize()
        transfer_wall_ns = time.perf_counter_ns() - transfer_started
        assert_grid_cmlm_invariants(fresh_model, require_cuda=True)
    else:
        fresh_model = fresh_model.train()
        assert_grid_cmlm_invariants(fresh_model, require_cuda=False)
        if next(fresh_model.parameters()).device.type != "cpu":
            raise RuntimeError("CPU fixture reconstruction escaped the CPU")
    optimizer_started = time.perf_counter_ns()
    fresh_optimizer = build_adamw_optimizer(fresh_model)
    optimizer_wall_ns = time.perf_counter_ns() - optimizer_started
    assert_adamw_invariants(
        fresh_optimizer, fresh_model, expected_completed_updates=0
    )
    del fresh_optimizer
    del fresh_model
    if runtime_mode == PRODUCTION_RUNTIME_MODE:
        torch.cuda.empty_cache()
    total_wall_ns = model_wall_ns + transfer_wall_ns + optimizer_wall_ns
    if total_wall_ns <= 0:
        raise RuntimeError("fresh-reconstruction timer did not advance")
    return _sealed(
        {
            "schema": FRESH_RECONSTRUCTION_PROBE_SCHEMA_VERSION,
            "run_id": run_id,
            "model_construct_wall_ns": model_wall_ns,
            "cuda_transfer_wall_ns": transfer_wall_ns,
            "optimizer_construct_wall_ns": optimizer_wall_ns,
            "total_wall_ns": total_wall_ns,
            "state_discarded": True,
        },
        id_field="probe_id",
    )


def _selection_input_payloads(checkpoint_sha256: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(1, VALIDATION_PASS_COUNT + 1):
        # Two in-range indices share the unique minimum; the earlier one must win.
        parent_ce = (
            0.25 if index in SELECTION_TIE_INDICES else 1.0 + index / 100.0
        )
        digest = hashlib.sha256(
            f"{checkpoint_sha256}:{index}".encode("ascii")
        ).hexdigest()
        rows.append(
            {
                "checkpoint_index": index,
                "optimizer_step": index * VALIDATION_INTERVAL,
                "parent_grouped_ce_hex": parent_ce.hex(),
                "checkpoint_sha256": digest,
            }
        )
    return rows


def _synthetic_checkpoint_metrics(checkpoint_sha256: str) -> tuple[CheckpointMetric, ...]:
    rows = _selection_input_payloads(checkpoint_sha256)
    metrics = [
        CheckpointMetric(
            optimizer_step=int(row["optimizer_step"]),
            parent_grouped_ce=float.fromhex(str(row["parent_grouped_ce_hex"])),
            checkpoint_sha256=str(row["checkpoint_sha256"]),
        )
        for row in rows
    ]
    return tuple(metrics)


def _run_final_selection(
    *, run_id: str, checkpoint_sha256: str
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    selection_rows = _selection_input_payloads(checkpoint_sha256)
    selection_input_sha256 = canonical_sha256(selection_rows)
    metrics = _synthetic_checkpoint_metrics(checkpoint_sha256)
    selection_started = time.perf_counter_ns()
    selected = select_checkpoint(metrics)
    selection_wall_ns = time.perf_counter_ns() - selection_started
    if selection_wall_ns <= 0:
        raise RuntimeError("final-selection timer did not advance")
    selected_index = metrics.index(selected) + 1
    if selected_index != SELECTION_TIE_INDICES[0]:
        raise RuntimeError(
            f"{VALIDATION_PASS_COUNT}-metric selection did not apply the frozen "
            "tie-break"
        )
    probe = _sealed(
        {
            "schema": FINAL_SELECTION_PROBE_SCHEMA_VERSION,
            "run_id": run_id,
            "candidate_count": len(metrics),
            "selection_rule": CHECKPOINT_SELECTION_RULE,
            "selected_checkpoint_index": selected_index,
            "selection_wall_ns": selection_wall_ns,
        },
        id_field="probe_id",
    )
    selected_summary = _sealed(
        {
            "schema": SELECTED_METRIC_SUMMARY_SCHEMA_VERSION,
            "selection_input_sha256": selection_input_sha256,
            "final_selection_probe_id": probe["probe_id"],
            **selection_rows[selected_index - 1],
        },
        id_field="summary_id",
    )
    return probe, selection_input_sha256, selected_summary


def _make_overhead_projection(
    *,
    run_id: str,
    handshake: Mapping[str, Any],
    fresh: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    multiplier = CHECKPOINT_PROJECTION_MULTIPLIER
    write = int(checkpoint["write_wall_ns"])
    load = int(checkpoint["weights_only_load_wall_ns"])
    roundtrip = int(checkpoint["roundtrip_wall_ns"])
    handshake_wall = int(handshake["wall_ns"])
    fresh_wall = int(fresh["total_wall_ns"])
    selection_wall = int(selection["selection_wall_ns"])
    return _sealed(
        {
            "schema": OVERHEAD_PROJECTION_SCHEMA_VERSION,
            "run_id": run_id,
            "projection_method": OVERHEAD_PROJECTION_METHOD,
            "fixed_checkpoint_multiplier": multiplier,
            "lock_handshake_probe_id": handshake["probe_id"],
            "measured_lock_setup_handshake_wall_ns": handshake_wall,
            "fresh_reconstruction_probe_id": fresh["probe_id"],
            "measured_fresh_reconstruction_wall_ns": fresh_wall,
            "checkpoint_cost_probe_id": checkpoint["probe_id"],
            "measured_checkpoint_write_wall_ns": write,
            "measured_checkpoint_weights_only_load_wall_ns": load,
            "measured_checkpoint_roundtrip_wall_ns": roundtrip,
            "final_selection_probe_id": selection["probe_id"],
            "measured_final_selection_wall_ns": selection_wall,
            "projected_checkpoint_writes": multiplier,
            "projected_checkpoint_loads": multiplier,
            "projected_checkpoint_write_wall_ns": multiplier * write,
            "projected_checkpoint_weights_only_load_wall_ns": multiplier * load,
            "projected_checkpoint_roundtrip_wall_ns": multiplier * roundtrip,
            "projected_fresh_setup_wall_ns": fresh_wall,
            "projected_final_selection_wall_ns": selection_wall,
            "projected_lock_setup_handshake_wall_ns": handshake_wall,
            "projected_campaign_overhead_wall_ns": (
                handshake_wall + fresh_wall + multiplier * roundtrip + selection_wall
            ),
        },
        id_field="projection_id",
    )


def _read_diagnostic_checkpoint_snapshot(
    checkpoint_path: Path,
    *,
    expected_sha256: str,
    expected_bytes: int,
) -> bytes:
    """Read one immutable, bounded snapshot from a single regular-file handle."""

    path = Path(checkpoint_path)
    expected_sha = _sha256(expected_sha256, field="checkpoint_sha256")
    byte_count = _strict_int(
        expected_bytes, field="checkpoint_bytes", minimum=1
    )
    if byte_count > MAX_CHECKPOINT_BYTES:
        raise ValueError("diagnostic checkpoint exceeds the bounded snapshot limit")
    _assert_existing_nonsymlink_chain(path, label="diagnostic checkpoint")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size != byte_count:
            raise ValueError("diagnostic checkpoint is not the expected regular file")
        chunks: list[bytes] = []
        remaining = byte_count + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
        if (
            not os.path.samestat(before, after)
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
        ):
            raise RuntimeError("diagnostic checkpoint changed while snapshotting")
    finally:
        os.close(descriptor)
    snapshot = b"".join(chunks)
    if len(snapshot) != byte_count:
        raise ValueError("diagnostic checkpoint byte count changed while snapshotting")
    if hashlib.sha256(snapshot).hexdigest() != expected_sha:
        raise ValueError("diagnostic checkpoint snapshot SHA-256 mismatch")
    return snapshot


@dataclass(frozen=True, slots=True)
class CampaignOverheadCostProbeResult:
    report: dict[str, Any]
    checkpoint_path: Path
    checkpoint_snapshot: bytes = dataclass_field(repr=False)

    def __post_init__(self) -> None:
        validate_campaign_overhead_cost_probe_report(self.report)
        if not isinstance(self.checkpoint_path, Path):
            raise TypeError("retained diagnostic checkpoint path must be a Path")
        expected_name = self.report["checkpoint_artifact_filename"]
        if self.checkpoint_path.name != expected_name:
            raise ValueError("retained diagnostic checkpoint path identity is invalid")
        if type(self.checkpoint_snapshot) is not bytes:
            raise TypeError("diagnostic checkpoint snapshot must be immutable bytes")
        checkpoint = self.report["checkpoint_cost_probe"]
        if (
            len(self.checkpoint_snapshot) != checkpoint["checkpoint_bytes"]
            or hashlib.sha256(self.checkpoint_snapshot).hexdigest()
            != checkpoint["checkpoint_sha256"]
        ):
            raise ValueError("diagnostic checkpoint snapshot differs from its report")


_PREPARED_PROBE_TOKEN = object()


class PreparedCampaignOverheadCostProbe:
    """Own one verified staging checkpoint until exact-preflight commits it."""

    __slots__ = (
        "_active",
        "_committed",
        "_directory_fd",
        "_result",
        "_scratch",
        "_staged",
        "_token",
    )

    def __init__(
        self,
        *,
        result: CampaignOverheadCostProbeResult,
        staged: _StagedDiagnosticCheckpoint,
        scratch: Path,
        directory_fd: int | None,
        token: object,
    ) -> None:
        if token is not _PREPARED_PROBE_TOKEN:
            raise TypeError("prepared campaign-overhead probe is internal")
        if result.checkpoint_path != staged.final_path:
            raise ValueError("prepared probe result differs from its staged checkpoint")
        self._result = result
        self._staged = staged
        self._scratch = scratch
        self._directory_fd = directory_fd
        self._active = True
        self._committed = False
        self._token = token

    @property
    def result(self) -> CampaignOverheadCostProbeResult:
        return self._result

    @property
    def committed(self) -> bool:
        return self._committed

    def abort(self) -> None:
        """Close the directory handle while retaining the uncommitted staging file."""

        if not self._active:
            return
        directory_fd = self._directory_fd
        self._directory_fd = None
        self._active = False
        if directory_fd is not None:
            try:
                os.close(directory_fd)
            except OSError:
                pass

    def commit(self) -> CampaignOverheadCostProbeResult:
        """Expose the final diagnostic name after all outer validation has passed."""

        if self._token is not _PREPARED_PROBE_TOKEN or not self._active:
            raise RuntimeError("prepared campaign-overhead probe is no longer active")
        try:
            _commit_staged_checkpoint(
                self._staged,
                scratch=self._scratch,
                directory_fd=self._directory_fd,
            )
        except BaseException:
            self.abort()
            raise
        directory_fd = self._directory_fd
        self._directory_fd = None
        self._active = False
        self._committed = True
        if directory_fd is not None:
            try:
                os.close(directory_fd)
            except OSError:
                # Namespace publication already succeeded; descriptor cleanup does
                # not retroactively turn it into a partial diagnostic write.
                pass
        return self._result

    def __del__(self) -> None:
        try:
            self.abort()
        except Exception:
            pass


def _validate_source_state(
    model: GridCMLM,
    optimizer: torch.optim.AdamW,
    *,
    require_cuda: bool,
) -> None:
    device = assert_grid_cmlm_invariants(model, require_cuda=require_cuda)
    if not require_cuda and device.type != "cpu":
        raise RuntimeError("CPU fixture requires an exact CPU GridCMLM")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("cost probe requires cleared source-model gradients")
    if any(not module.training for module in model.modules()):
        raise RuntimeError("cost probe requires the restored training mode")
    assert_adamw_invariants(
        optimizer,
        model,
        expected_learning_rate=learning_rate_for_update(PREFLIGHT_UPDATES),
        expected_completed_updates=PREFLIGHT_UPDATES,
    )
    if len(optimizer.state) != len(tuple(model.parameters())):
        raise RuntimeError("cost probe requires fully initialized optimizer state")


def _run_probe(
    *,
    model: GridCMLM,
    optimizer: torch.optim.AdamW,
    run_id: str,
    runtime_mode: str,
    runtime_attestation: FrozenRuntimeAttestation | None,
    run_root: str | Path,
    scratch_dir: str | Path,
    config_sha256: str,
    runtime_source_sha256: str,
    test_source_sha256: str,
    lock_handshake_probe: Mapping[str, object],
) -> PreparedCampaignOverheadCostProbe:
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run_id must be non-empty")
    for field, digest in (
        ("config_sha256", config_sha256),
        ("runtime_source_sha256", runtime_source_sha256),
        ("test_source_sha256", test_source_sha256),
    ):
        _sha256(digest, field=field)
    if config_sha256 != training_config_sha256():
        raise ValueError("config_sha256 does not bind the frozen training config")
    handshake = validate_lock_handshake_probe(
        dict(lock_handshake_probe), expected_run_id=run_id
    )
    require_cuda = runtime_mode == PRODUCTION_RUNTIME_MODE
    _validate_source_state(model, optimizer, require_cuda=require_cuda)
    root, scratch, root_stat = _prepare_scratch_paths(run_root, scratch_dir)
    directory_fd = _create_exclusive_scratch_dir(root, scratch, root_stat)
    cpu_rng_before = torch.get_rng_state().clone()
    cuda_rng_before = (
        [state.clone() for state in torch.cuda.get_rng_state_all()]
        if require_cuda
        else []
    )
    try:
        try:
            staged = _stage_content_addressed_checkpoint(
                scratch=scratch,
                directory_fd=directory_fd,
                model=model,
                optimizer=optimizer,
                config_sha256=config_sha256,
                runtime_source_sha256=runtime_source_sha256,
                test_source_sha256=test_source_sha256,
                runtime_mode=runtime_mode,
            )
            checkpoint_sha = staged.checkpoint_sha256
            checkpoint_bytes = staged.checkpoint_bytes
            write_wall = staged.write_wall_ns
            load_wall = _verified_weights_only_load(
                staged.staging_path,
                checkpoint_sha256=checkpoint_sha,
                checkpoint_bytes=checkpoint_bytes,
                config_sha256=config_sha256,
                runtime_source_sha256=runtime_source_sha256,
                test_source_sha256=test_source_sha256,
                expected_model_state_keys=tuple(model.state_dict()),
            )
            checkpoint_probe = _sealed(
                {
                    "schema": CHECKPOINT_COST_PROBE_SCHEMA_VERSION,
                    "probe_scope": "content_addressed_independent_probe",
                    "run_id": run_id,
                    "checkpoint_sha256": checkpoint_sha,
                    "checkpoint_bytes": checkpoint_bytes,
                    "write_wall_ns": write_wall,
                    "weights_only_load_wall_ns": load_wall,
                    "roundtrip_wall_ns": write_wall + load_wall,
                    "weights_only_load": True,
                    "artifact_disposition": DIAGNOSTIC_DISPOSITION,
                    "selectable_checkpoint_created": False,
                },
                id_field="probe_id",
            )
            fresh_probe = _run_fresh_reconstruction(
                run_id=run_id,
                runtime_mode=runtime_mode,
                attestation=runtime_attestation,
            )
            (
                selection_probe,
                selection_input_sha256,
                selected_metric_summary,
            ) = _run_final_selection(
                run_id=run_id, checkpoint_sha256=checkpoint_sha
            )
            projection = _make_overhead_projection(
                run_id=run_id,
                handshake=handshake,
                fresh=fresh_probe,
                checkpoint=checkpoint_probe,
                selection=selection_probe,
            )
            report = _sealed(
                {
                    "schema": CAMPAIGN_OVERHEAD_PROBE_SCHEMA_VERSION,
                    "status": "PASS",
                    "run_id": run_id,
                    "runtime_mode": runtime_mode,
                    "optimizer_step": PREFLIGHT_UPDATES,
                    "source_optimizer_state": "fully_initialized_step_100",
                    "config_sha256": config_sha256,
                    "runtime_source_sha256": runtime_source_sha256,
                    "test_source_sha256": test_source_sha256,
                    "lock_handshake_probe": handshake,
                    "fresh_reconstruction_probe": fresh_probe,
                    "checkpoint_cost_probe": checkpoint_probe,
                    "final_selection_probe": selection_probe,
                    "selection_input_sha256": selection_input_sha256,
                    "selected_metric_summary": selected_metric_summary,
                    "overhead_projection": projection,
                    "checkpoint_artifact_filename": staged.final_path.name,
                    "checkpoint_artifact_status": DIAGNOSTIC_STATUS,
                    "dataset_file_reads": 0,
                    "fallback_used": False,
                },
                id_field="probe_id",
            )
            validate_campaign_overhead_cost_probe_report(
                report,
                expected_run_id=run_id,
                expected_config_sha256=config_sha256,
                expected_runtime_source_sha256=runtime_source_sha256,
                expected_test_source_sha256=test_source_sha256,
                require_production_cuda=require_cuda,
            )
            if require_cuda and runtime_attestation is not None:
                validate_runtime_attestation(runtime_attestation)
            checkpoint_snapshot = _read_diagnostic_checkpoint_snapshot(
                staged.staging_path,
                expected_sha256=checkpoint_sha,
                expected_bytes=checkpoint_bytes,
            )
            prepared_result = CampaignOverheadCostProbeResult(
                report=report,
                checkpoint_path=staged.final_path,
                checkpoint_snapshot=checkpoint_snapshot,
            )
        finally:
            torch.set_rng_state(cpu_rng_before)
            if require_cuda:
                torch.cuda.set_rng_state_all(cuda_rng_before)
    except BaseException:
        if directory_fd is not None:
            try:
                os.close(directory_fd)
            except OSError:
                pass
        raise

    try:
        return PreparedCampaignOverheadCostProbe(
            result=prepared_result,
            staged=staged,
            scratch=scratch,
            directory_fd=directory_fd,
            token=_PREPARED_PROBE_TOKEN,
        )
    except BaseException:
        if directory_fd is not None:
            try:
                os.close(directory_fd)
            except OSError:
                pass
        raise


def run_campaign_overhead_cost_probe(
    *,
    model: GridCMLM,
    optimizer: torch.optim.AdamW,
    runtime_attestation: FrozenRuntimeAttestation,
    run_root: str | Path,
    scratch_dir: str | Path,
    config_sha256: str,
    runtime_source_sha256: str,
    test_source_sha256: str,
    lock_handshake_probe: Mapping[str, object],
) -> PreparedCampaignOverheadCostProbe:
    """Prepare the production probe; exact-preflight owns the final commit."""

    attestation = validate_runtime_attestation(runtime_attestation)
    return _run_probe(
        model=model,
        optimizer=optimizer,
        run_id=attestation.run_id,
        runtime_mode=PRODUCTION_RUNTIME_MODE,
        runtime_attestation=attestation,
        run_root=run_root,
        scratch_dir=scratch_dir,
        config_sha256=config_sha256,
        runtime_source_sha256=runtime_source_sha256,
        test_source_sha256=test_source_sha256,
        lock_handshake_probe=lock_handshake_probe,
    )


def run_campaign_overhead_cost_probe_cpu_fixture(
    *,
    model: GridCMLM,
    optimizer: torch.optim.AdamW,
    run_id: str,
    run_root: str | Path,
    scratch_dir: str | Path,
    config_sha256: str,
    runtime_source_sha256: str,
    test_source_sha256: str,
    lock_handshake_probe: Mapping[str, object],
) -> CampaignOverheadCostProbeResult:
    """Exercise the probe on CPU without claiming frozen CUDA evidence."""

    if not isinstance(run_id, str) or not run_id.startswith("cpu-fixture:"):
        raise ValueError("CPU fixture run_id must start with 'cpu-fixture:'")
    prepared = _run_probe(
        model=model,
        optimizer=optimizer,
        run_id=run_id,
        runtime_mode=CPU_FIXTURE_RUNTIME_MODE,
        runtime_attestation=None,
        run_root=run_root,
        scratch_dir=scratch_dir,
        config_sha256=config_sha256,
        runtime_source_sha256=runtime_source_sha256,
        test_source_sha256=test_source_sha256,
        lock_handshake_probe=lock_handshake_probe,
    )
    return prepared.commit()


__all__ = [
    "CAMPAIGN_OVERHEAD_PROBE_SCHEMA_VERSION",
    "CHECKPOINT_COST_PROBE_SCHEMA_VERSION",
    "CHECKPOINT_PROJECTION_MULTIPLIER",
    "CPU_FIXTURE_RUNTIME_MODE",
    "CampaignOverheadCostProbeResult",
    "DIAGNOSTIC_DISPOSITION",
    "DIAGNOSTIC_STATUS",
    "FINAL_SELECTION_PROBE_SCHEMA_VERSION",
    "FRESH_RECONSTRUCTION_PROBE_SCHEMA_VERSION",
    "LOCK_HANDSHAKE_PROBE_SCHEMA_VERSION",
    "OVERHEAD_PROJECTION_SCHEMA_VERSION",
    "PRODUCTION_RUNTIME_MODE",
    "PreparedCampaignOverheadCostProbe",
    "SELECTED_METRIC_SUMMARY_SCHEMA_VERSION",
    "run_campaign_overhead_cost_probe",
    "run_campaign_overhead_cost_probe_cpu_fixture",
    "validate_campaign_overhead_cost_probe_report",
    "validate_checkpoint_cost_probe",
    "validate_final_selection_probe",
    "validate_fresh_reconstruction_probe",
    "validate_lock_handshake_probe",
    "validate_overhead_projection",
]
