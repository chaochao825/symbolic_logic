"""Same-process exact-preflight-to-training campaign for M04a.

This is the explicit PyTorch campaign boundary.  It consumes a successful
``PreflightResult`` whose inherited GPU lock is still held, discards every
diagnostic model/optimizer state, reconstructs the primary model from the frozen
seed, executes all 20,000 updates, validates and round-trips ten checkpoints, and
releases the lock immediately after the final checkpoint winner is fixed.

Only a complete, within-budget run can publish the v0.4 training bundle.  Any
exception preserves the fresh working directory and writes a small ineligible
failure receipt; it never promotes a partial checkpoint.
"""

from __future__ import annotations

import gc
import hashlib
import os
import stat
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from . import m04a_evidence as evidence
from .m04a_contract import (
    OPTIMIZER_UPDATES,
    TRAINING_SEED,
    VALIDATION_INTERVAL,
    VALIDATION_PASS_COUNT,
)
from .m04a_evidence import (
    make_checkpoint_manifest_row,
    make_selected_checkpoint_manifest,
    make_training_cost_ledger_row,
    make_training_cost_summary,
    make_training_lock_interval,
    make_validation_metric_row,
    publish_training_artifact_bundle,
    python_runtime_identity_sha256,
    validate_training_checkpoint_artifacts,
    validate_training_cost_ledger,
)
from .m04a_lock_handshake import (
    assert_held_gpu_lock,
    release_held_gpu_lock,
)
from .m04a_model import GridCMLM
from .m04a_preflight import PreflightResult
from .m04a_production_data import load_indexed_m04a_data
from .m04a_production_data_contract import (
    PRODUCTION_DATA_CLOSURE_ID,
    PRODUCTION_DATA_FILES,
    SANITIZED_ARTIFACT_MANIFEST_SHA256,
    SANITIZED_DIRECTORY,
    TRAINING_CACHE_ARTIFACT_MANIFEST_SHA256,
    TRAINING_CACHE_DIRECTORY,
    VALIDATION_CACHE_ARTIFACT_MANIFEST_SHA256,
    VALIDATION_CACHE_DIRECTORY,
    ProductionDataBinding,
    revalidate_production_data_binding,
    validate_production_data_root,
)
from .m04a_python_runtime_lock import PythonRuntimeLockArtifact
from .m04a_torch_runtime import (
    FrozenRuntimeAttestation,
    capture_rng_state,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint_new,
    validate_runtime_attestation,
)
from .m04a_train import (
    assert_adamw_invariants,
    assert_grid_cmlm_invariants,
    build_adamw_optimizer,
    train_primary_update,
    validate_literal_manifest,
)
from .m04a_train_contract import (
    assert_campaign_budget,
    learning_rate_for_update,
)
from .m04a_validation_manifest import (
    ValidationEpisodeManifest,
    read_validation_episode_manifest,
    require_externally_committed_validation_manifest,
)
from .manifest import runtime_source_fingerprint, serialize_json, serialize_jsonl


CAMPAIGN_FAILURE_SCHEMA_VERSION = "afts-m04a-training-campaign-failure/v0.1"
OPTIMIZER_SCHEDULE_STATE_SCHEMA_VERSION = (
    evidence.OPTIMIZER_SCHEDULE_STATE_SCHEMA_VERSION
)
CAMPAIGN_PROGRESS_SCHEMA_VERSION = "afts-m04a-training-progress/v0.1"
PROGRESS_UPDATE_INTERVAL = 50


@dataclass(frozen=True, slots=True)
class CheckpointArtifact:
    checkpoint_index: int
    optimizer_step: int
    path: Path
    sha256: str
    bytes: int
    checkpoint_event: dict[str, Any]

    def __post_init__(self) -> None:
        if (
            type(self.checkpoint_index) is not int
            or not 1 <= self.checkpoint_index <= VALIDATION_PASS_COUNT
            or self.optimizer_step != self.checkpoint_index * VALIDATION_INTERVAL
        ):
            raise ValueError("checkpoint artifact coordinates are invalid")
        if not self.path.is_file() or self.path.is_symlink():
            raise ValueError("checkpoint artifact must be a regular non-link file")
        if (
            not isinstance(self.sha256, str)
            or len(self.sha256) != 64
            or self.path.stat().st_size != self.bytes
        ):
            raise ValueError("checkpoint artifact metadata is invalid")
        try:
            if len(bytes.fromhex(self.sha256)) != 32 or self.sha256 != self.sha256.lower():
                raise ValueError
        except ValueError as exc:
            raise ValueError("checkpoint artifact SHA-256 is invalid") from exc
        if self.checkpoint_event.get("phase") != "checkpoint_operation":
            raise ValueError("checkpoint artifact lacks its operation event")


@dataclass(frozen=True, slots=True)
class CampaignPublication:
    report: dict[str, Any]
    selected_checkpoint_manifest: dict[str, Any]
    training_artifact_manifest: dict[str, Any]


class CampaignFailure(RuntimeError):
    """Terminal formal-campaign failure with no selectable training evidence."""

    def __init__(
        self,
        failure_code: str,
        *,
        optimizer_step: int,
        working_dir: Path,
        cause_type: str,
    ) -> None:
        super().__init__(f"{failure_code}: M04a training campaign is incomplete")
        self.failure_code = failure_code
        self.optimizer_step = optimizer_step
        self.working_dir = working_dir
        self.cause_type = cause_type


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _write_all(descriptor: int, content: bytes) -> None:
    view = memoryview(content)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("short write while publishing campaign evidence")
        view = view[written:]


def _write_new_fsync(path: Path, content: bytes, *, mode: int = 0o400) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        _write_all(descriptor, content)
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    if os.name != "posix":
        return
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _strict_campaign_paths(
    *, run_root: Path, working_dir: str | Path, output_dir: str | Path
) -> tuple[Path, Path]:
    root = Path(os.path.abspath(os.fspath(run_root.expanduser())))
    working = Path(os.path.abspath(os.fspath(Path(working_dir).expanduser())))
    output = Path(os.path.abspath(os.fspath(Path(output_dir).expanduser())))
    if not root.is_dir() or root.is_symlink():
        raise ValueError("campaign run root must be an existing real directory")
    if working != root / "campaign-working" or output != root / "training-artifact":
        raise ValueError("campaign paths must be the two fixed run-root children")
    if working == output or os.path.lexists(working) or os.path.lexists(output):
        raise FileExistsError("campaign working/output paths must both be fresh")
    if working.parent != root or output.parent != root:
        raise ValueError("campaign paths escaped their launch-plan run root")
    return working, output


def _rng_states_equal(
    left: Mapping[str, object], right: Mapping[str, object]
) -> bool:
    if left.get("schema") != right.get("schema") or left.get(
        "optimizer_step"
    ) != right.get("optimizer_step"):
        return False
    left_cpu = left.get("torch_cpu")
    right_cpu = right.get("torch_cpu")
    left_cuda = left.get("torch_cuda")
    right_cuda = right.get("torch_cuda")
    return bool(
        isinstance(left_cpu, torch.Tensor)
        and isinstance(right_cpu, torch.Tensor)
        and torch.equal(left_cpu, right_cpu)
        and isinstance(left_cuda, list)
        and isinstance(right_cuda, list)
        and len(left_cuda) == len(right_cuda)
        and all(
            isinstance(left_state, torch.Tensor)
            and isinstance(right_state, torch.Tensor)
            and torch.equal(left_state, right_state)
            for left_state, right_state in zip(left_cuda, right_cuda, strict=True)
        )
    )


def _validate_loaded_optimizer_state(
    optimizer_state: object, *, parameter_count: int, optimizer_step: int
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
    group = groups[0]
    parameter_ids = group.get("params") if type(group) is dict else None
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
            or float(step.item()) != float(optimizer_step)
        ):
            raise ValueError("loaded AdamW step differs from checkpoint coordinate")
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


def verify_checkpoint_roundtrip(
    path: str | Path,
    *,
    checkpoint_sha256: str,
    checkpoint_bytes: int,
    optimizer_step: int,
    config_sha256: str,
    runtime_source_sha256: str,
    test_source_sha256: str,
    expected_model_state_keys: Sequence[str],
) -> int:
    """Weights-only load one checkpoint on CPU without consuming training RNG."""

    source = Path(path)
    rng_before = capture_rng_state(optimizer_step=optimizer_step)
    started = time.perf_counter_ns()
    try:
        verification_model = GridCMLM().train()
        verification_optimizer = build_adamw_optimizer(verification_model)
        payload = load_checkpoint(
            source,
            expected_sha256=checkpoint_sha256,
            expected_optimizer_step=optimizer_step,
            expected_config_sha256=config_sha256,
            expected_runtime_source_sha256=runtime_source_sha256,
            expected_test_source_sha256=test_source_sha256,
            expected_bytes=checkpoint_bytes,
            map_location="cpu",
        )
        if tuple(payload["model_state"]) != tuple(expected_model_state_keys):
            raise ValueError("loaded checkpoint model-state keys drifted")
        incompatibility = verification_model.load_state_dict(
            payload["model_state"], strict=True
        )
        if incompatibility.missing_keys or incompatibility.unexpected_keys:
            raise ValueError("strict checkpoint model load reported incompatible keys")
        _validate_loaded_optimizer_state(
            payload["optimizer_state"],
            parameter_count=len(tuple(verification_model.parameters())),
            optimizer_step=optimizer_step,
        )
        verification_optimizer.load_state_dict(payload["optimizer_state"])
        assert_grid_cmlm_invariants(verification_model, require_cuda=False)
        assert_adamw_invariants(
            verification_optimizer,
            verification_model,
            expected_learning_rate=learning_rate_for_update(optimizer_step),
            expected_completed_updates=optimizer_step,
        )
        cuda_rng = payload["rng_state"]["torch_cuda"]
        if not isinstance(cuda_rng, list) or len(cuda_rng) != torch.cuda.device_count():
            raise ValueError("checkpoint CUDA RNG state count differs from runtime")
        del verification_optimizer
        del verification_model
        del payload
        gc.collect()
    finally:
        restore_rng_state(rng_before)
    rng_after = capture_rng_state(optimizer_step=optimizer_step)
    if not _rng_states_equal(rng_before, rng_after):
        raise RuntimeError("checkpoint roundtrip failed to restore training RNG")
    elapsed = time.perf_counter_ns() - started
    if elapsed <= 0:
        raise RuntimeError("checkpoint roundtrip timer did not advance")
    return elapsed


def _make_checkpoint(
    *,
    checkpoint_dir: Path,
    checkpoint_index: int,
    event_index: int,
    model: GridCMLM,
    optimizer: torch.optim.AdamW,
    config_sha256: str,
    runtime_source_sha256: str,
    test_source_sha256: str,
) -> CheckpointArtifact:
    optimizer_step = checkpoint_index * VALIDATION_INTERVAL
    path = checkpoint_dir / f"checkpoint-{checkpoint_index:02d}.pt"
    started = time.perf_counter_ns()
    metadata = save_checkpoint_new(
        path,
        optimizer_step=optimizer_step,
        model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict(),
        config_sha256=config_sha256,
        runtime_source_sha256=runtime_source_sha256,
        test_source_sha256=test_source_sha256,
    )
    if os.name == "posix":
        os.chmod(path, 0o400)
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _fsync_directory(checkpoint_dir)
    verify_checkpoint_roundtrip(
        path,
        checkpoint_sha256=str(metadata["sha256"]),
        checkpoint_bytes=int(metadata["bytes"]),
        optimizer_step=optimizer_step,
        config_sha256=config_sha256,
        runtime_source_sha256=runtime_source_sha256,
        test_source_sha256=test_source_sha256,
        expected_model_state_keys=tuple(model.state_dict()),
    )
    elapsed = time.perf_counter_ns() - started
    checkpoint_event = make_training_cost_ledger_row(
        phase="checkpoint_operation",
        event_index=event_index,
        checkpoint_index=checkpoint_index,
        checkpoint_writes=1,
        checkpoint_io_bytes=int(metadata["bytes"]),
        checkpoint_io_ns=elapsed,
        wall_time_ns=elapsed,
    )
    return CheckpointArtifact(
        checkpoint_index=checkpoint_index,
        optimizer_step=optimizer_step,
        path=path,
        sha256=str(metadata["sha256"]),
        bytes=int(metadata["bytes"]),
        checkpoint_event=checkpoint_event,
    )


def _budget_guard(preflight: PreflightResult, *, endpoint_ns: int | None = None) -> int:
    lock = assert_held_gpu_lock(preflight.held_lock_handshake)
    endpoint = time.perf_counter_ns() if endpoint_ns is None else endpoint_ns
    start = int(lock["acquisition_started_perf_counter_ns"])
    if endpoint < start:
        raise RuntimeError("campaign budget endpoint predates lock acquisition")
    assert_campaign_budget(endpoint - start)
    return endpoint


def _emit_progress(
    callback: Callable[[Mapping[str, Any]], None] | None,
    *,
    run_id: str,
    optimizer_step: int,
    event_count: int,
    checkpoint_count: int,
) -> None:
    if callback is None:
        return
    callback(
        {
            "schema": CAMPAIGN_PROGRESS_SCHEMA_VERSION,
            "run_id": run_id,
            "optimizer_step": optimizer_step,
            "event_count": event_count,
            "checkpoint_count": checkpoint_count,
            "training_evidence_eligible": False,
        }
    )


def _training_closure(
    ledger_closure: Mapping[str, Any], preflight_report: Mapping[str, Any]
) -> dict[str, Any]:
    inference = preflight_report["inference"]
    return {
        "status": "COMPLETE",
        "preflight_optimizer_updates": ledger_closure["preflight_optimizer_updates"],
        "preflight_microbatches": ledger_closure["preflight_microbatches"],
        "preflight_encoder_forward_calls": ledger_closure[
            "preflight_encoder_forward_calls"
        ],
        "preflight_decoder_forward_calls": ledger_closure[
            "preflight_decoder_forward_calls"
        ],
        "preflight_backward_calls": ledger_closure["preflight_backward_calls"],
        "preflight_inference_lanes": inference["lane_count"],
        "preflight_inference_steps": inference["denoising_steps"],
        "preflight_inference_encoder_batch_calls": inference[
            "encoder_batch_calls"
        ],
        "preflight_inference_decoder_batch_calls": inference[
            "decoder_batch_calls"
        ],
        "preflight_inference_sample_equivalent_forward_calls": inference[
            "sample_equivalent_forward_calls"
        ],
        "primary_optimizer_updates": ledger_closure["optimizer_updates"],
        "primary_microbatches": ledger_closure["microbatches"],
        "primary_arc2_episodes": ledger_closure["arc2_episodes"],
        "primary_rearc_episodes": ledger_closure["rearc_episodes"],
        "primary_encoder_forward_calls": ledger_closure["encoder_forward_calls"],
        "primary_decoder_forward_calls": ledger_closure["decoder_forward_calls"],
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


def _write_failure_receipt(
    working: Path,
    *,
    run_id: str,
    failure_code: str,
    optimizer_step: int,
    cause_type: str,
    lock_release_attempted: bool,
    lock_release_succeeded: bool,
) -> None:
    if not working.is_dir() or working.is_symlink():
        raise ValueError("campaign failure receipt parent is not a real directory")
    payload = _failure_receipt_payload(
        run_id=run_id,
        failure_code=failure_code,
        optimizer_step=optimizer_step,
        cause_type=cause_type,
        lock_release_attempted=lock_release_attempted,
        lock_release_succeeded=lock_release_succeeded,
    )
    _ensure_exact_failure_receipt(
        working / "campaign_failure.json", payload=payload, parent=working
    )


def _failure_receipt_payload(
    *,
    run_id: str,
    failure_code: str,
    optimizer_step: int,
    cause_type: str,
    lock_release_attempted: bool,
    lock_release_succeeded: bool,
) -> dict[str, Any]:
    return {
        "schema": CAMPAIGN_FAILURE_SCHEMA_VERSION,
        "status": failure_code,
        "run_id": run_id,
        "completed_optimizer_step": optimizer_step,
        "cause_type": cause_type,
        "lock_release_attempted": lock_release_attempted,
        "lock_release_succeeded": lock_release_succeeded,
        "selected_checkpoint_complete": False,
        "training_evidence_eligible": False,
        "same_run_retry_allowed": False,
    }


def _read_exact_regular(path: Path, *, maximum_bytes: int) -> bytes:
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > maximum_bytes
        ):
            raise ValueError("campaign failure receipt is not a bounded regular file")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            block = os.read(descriptor, remaining)
            if not block:
                raise RuntimeError("campaign failure receipt became short")
            chunks.append(block)
            remaining -= len(block)
        if os.read(descriptor, 1):
            raise RuntimeError("campaign failure receipt grew while reading")
        after = os.fstat(descriptor)
        path_after = path.stat(follow_symlinks=False)
        if (
            not os.path.samestat(before, after)
            or not os.path.samestat(after, path_after)
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or before.st_ctime_ns != after.st_ctime_ns
        ):
            raise RuntimeError("campaign failure receipt changed while reading")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _ensure_exact_failure_receipt(
    path: Path, *, payload: Mapping[str, Any], parent: Path
) -> None:
    expected = serialize_json(dict(payload))
    try:
        _write_new_fsync(path, expected)
    except FileExistsError:
        if _read_exact_regular(path, maximum_bytes=64 * 1024) != expected:
            raise RuntimeError(
                "existing campaign failure receipt differs from terminal outcome"
            )
    _fsync_directory(parent)


def ensure_campaign_failure_receipt(
    *,
    run_root: str | Path,
    working_dir: str | Path,
    run_id: str,
    failure_code: str,
    optimizer_step: int,
    cause_type: str,
    lock_release_attempted: bool,
    lock_release_succeeded: bool,
) -> Path:
    """Publish one terminal ineligible receipt even for preparation failures."""

    root = Path(os.path.abspath(os.fspath(Path(run_root).expanduser())))
    working = Path(os.path.abspath(os.fspath(Path(working_dir).expanduser())))
    if not root.is_dir() or root.is_symlink():
        raise ValueError("campaign failure receipt requires a real run root")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("campaign failure receipt run_id is invalid")
    if type(optimizer_step) is not int or not 0 <= optimizer_step <= OPTIMIZER_UPDATES:
        raise ValueError("campaign failure receipt optimizer step is invalid")
    use_working = working == root / "campaign-working" and working.parent == root
    if use_working and os.path.lexists(working):
        use_working = working.is_dir() and not working.is_symlink()
    if use_working and not os.path.lexists(working):
        try:
            working.mkdir(mode=0o700)
        except OSError:
            use_working = False
        else:
            _fsync_directory(root)
    if use_working:
        _write_failure_receipt(
            working,
            run_id=run_id,
            failure_code=failure_code,
            optimizer_step=optimizer_step,
            cause_type=cause_type,
            lock_release_attempted=lock_release_attempted,
            lock_release_succeeded=lock_release_succeeded,
        )
        return working / "campaign_failure.json"

    payload = _failure_receipt_payload(
        run_id=run_id,
        failure_code=failure_code,
        optimizer_step=optimizer_step,
        cause_type=cause_type,
        lock_release_attempted=lock_release_attempted,
        lock_release_succeeded=lock_release_succeeded,
    )
    fallback = root / "campaign_failure.json"
    _ensure_exact_failure_receipt(fallback, payload=payload, parent=root)
    return fallback


def _failure_code(cause: BaseException) -> str:
    if isinstance(cause, CampaignFailure):
        return cause.failure_code
    if isinstance(cause, RuntimeError) and str(cause) == "BUDGET_EXCEEDED":
        return "BUDGET_EXCEEDED"
    if isinstance(cause, torch.cuda.OutOfMemoryError):
        return "OOM"
    return "CAMPAIGN_INCOMPLETE"


def _artifact_lineage(
    *,
    snapshots: Mapping[str, bytes],
    environment_bytes: bytes,
    launch_plan_sha256: str,
    lock_handshake_sha256: str,
    overhead_bytes: bytes,
    diagnostic_checkpoint: bytes,
    ledger_sha256: str,
    selected_manifest_bytes: bytes,
    selected_checkpoint_sha256: str,
    validation_manifest: ValidationEpisodeManifest,
    python_runtime_lock: PythonRuntimeLockArtifact,
) -> dict[str, str]:
    python = python_runtime_lock.payload["python"]
    runtime_identity = python_runtime_identity_sha256(
        {
            "implementation": python["implementation"],
            "version": python["version"],
            "executable": python["executable_lexical_path"],
            "executable_sha256": python["executable_sha256"],
            "executable_bytes": python["executable_bytes"],
        }
    )
    return {
        "reviewed_runtime_source_sha256": _sha256_bytes(
            snapshots["reviewed_runtime_source.zip"]
        ),
        "reviewed_test_snapshot_sha256": _sha256_bytes(
            snapshots["reviewed_test_snapshot.zip"]
        ),
        "remote_launcher_sha256": _sha256_bytes(snapshots["remote_launcher.py"]),
        "launch_plan_sha256": launch_plan_sha256,
        "lock_handshake_artifact_sha256": lock_handshake_sha256,
        "preflight_overhead_cost_report_sha256": _sha256_bytes(overhead_bytes),
        "preflight_diagnostic_checkpoint_sha256": _sha256_bytes(
            diagnostic_checkpoint
        ),
        "frozen_contract_sha256": _sha256_bytes(snapshots["frozen_contract.md"]),
        "sanitized_shard_manifest_sha256": _sha256_bytes(
            snapshots["sanitized_shard_manifest.json"]
        ),
        "data_split_manifest_sha256": _sha256_bytes(
            snapshots["data_split_manifest.json"]
        ),
        "validation_episode_outer_manifest_sha256": str(
            validation_manifest.artifact_manifest_sha256
        ),
        "validation_episode_manifest_sha256": _sha256_bytes(
            validation_manifest.jsonl_bytes
        ),
        "validation_episode_summary_sha256": _sha256_bytes(
            snapshots["validation_episode_manifest_summary.json"]
        ),
        "validation_episode_summary_id": str(
            validation_manifest.summary["summary_id"]
        ),
        "environment_manifest_sha256": _sha256_bytes(environment_bytes),
        "model_config_sha256": _sha256_bytes(snapshots["model_config.json"]),
        "training_cost_ledger_sha256": ledger_sha256,
        "selected_checkpoint_manifest_sha256": _sha256_bytes(
            selected_manifest_bytes
        ),
        "checkpoint_sha256": selected_checkpoint_sha256,
        "python_runtime_lock_sha256": python_runtime_lock.artifact_sha256,
        "python_runtime_lock_id": str(python_runtime_lock.payload["runtime_lock_id"]),
        "python_runtime_identity_sha256": runtime_identity,
    }


def _write_payload_files(
    payload_dir: Path, source_bytes: Mapping[str, bytes]
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name, content in sorted(source_bytes.items()):
        if not name or "/" in name or "\\" in name:
            raise ValueError("training artifact payload names must remain flat")
        path = payload_dir / name
        _write_new_fsync(path, content)
        paths[name] = path
    _fsync_directory(payload_dir)
    return paths


def _run_primary_training_campaign_held(
    *,
    preflight_result: PreflightResult,
    runtime_attestation: FrozenRuntimeAttestation,
    environment_manifest: Mapping[str, Any],
    python_runtime_lock: PythonRuntimeLockArtifact,
    visible_input_snapshots: Mapping[str, bytes],
    validation_manifest_dir: str | Path,
    production_data_root: str | Path,
    production_data_binding: ProductionDataBinding | None = None,
    working_dir: str | Path,
    output_dir: str | Path,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> CampaignPublication:
    """Run and publish the one frozen primary campaign under the preflight lock."""

    if type(preflight_result) is not PreflightResult:
        raise TypeError("formal campaign requires an exact PreflightResult")
    validate_runtime_attestation(runtime_attestation)
    lock_payload = assert_held_gpu_lock(preflight_result.held_lock_handshake)
    plan_artifact = preflight_result.launch_plan_artifact
    plan = plan_artifact.payload
    if plan["run_id"] != lock_payload["run_id"]:
        raise ValueError("campaign launch plan differs from its held GPU lock")
    if not isinstance(visible_input_snapshots, Mapping) or set(
        visible_input_snapshots
    ) != set(plan["expected_input_artifacts"]):
        raise ValueError("campaign input snapshots differ from the launch-plan set")
    snapshots = {name: bytes(content) for name, content in visible_input_snapshots.items()}
    for name, expected_sha256 in plan["expected_input_artifacts"].items():
        if _sha256_bytes(snapshots[name]) != expected_sha256:
            raise ValueError(f"campaign input snapshot changed: {name}")
    if python_runtime_lock.snapshot != snapshots["python-runtime-lock.json"]:
        raise ValueError("campaign runtime-lock snapshot differs from visible input")

    if production_data_binding is None:
        data_binding = validate_production_data_root(production_data_root)
    else:
        if type(production_data_binding) is not ProductionDataBinding:
            raise TypeError("formal production data binding must be exact")
        requested_root = Path(
            os.path.abspath(os.fspath(Path(production_data_root).expanduser()))
        )
        if production_data_binding.root != requested_root:
            raise ValueError("production data binding differs from its logical root")
        data_binding = revalidate_production_data_binding(production_data_binding)
    if data_binding.closure_id != PRODUCTION_DATA_CLOSURE_ID:
        raise ValueError("campaign production-data closure drifted")
    expected_split_sha = plan["expected_input_artifacts"]["data_split_manifest.json"]
    if expected_split_sha != PRODUCTION_DATA_FILES[
        "sanitized_split/data_split_manifest.json"
    ][1]:
        raise ValueError("production data split differs from the launch-plan input")

    commitment = preflight_result.validation_manifest_commitment
    validation_manifest = read_validation_episode_manifest(
        validation_manifest_dir,
        expected_artifact_manifest_sha256=(
            commitment.outer_artifact_manifest_sha256
        ),
    )
    require_externally_committed_validation_manifest(
        validation_manifest,
        expected_artifact_manifest_sha256=(
            commitment.outer_artifact_manifest_sha256
        ),
        expected_jsonl_sha256=commitment.jsonl_sha256,
    )
    if len(validation_manifest.rows) != commitment.row_count:
        raise ValueError("campaign validation row count differs from preflight")

    run_root = Path(plan["run_root"])
    working, output = _strict_campaign_paths(
        run_root=run_root, working_dir=working_dir, output_dir=output_dir
    )
    working.mkdir(mode=0o700)
    checkpoint_dir = working / "checkpoints"
    payload_dir = working / "payload"
    checkpoint_dir.mkdir(mode=0o700)
    payload_dir.mkdir(mode=0o700)
    _fsync_directory(working)
    _fsync_directory(run_root)

    ledger: list[dict[str, Any]] = [dict(row) for row in preflight_result.ledger_rows]
    checkpoint_artifacts: list[CheckpointArtifact] = []
    checkpoint_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    event_index = len(ledger)
    completed_step = 0
    fresh_setup_wall_ns = 0
    release_attempted = False
    release_succeeded = False
    publication_committed = False

    try:
        _budget_guard(preflight_result)
        setup_started = time.perf_counter_ns()
        torch.cuda.empty_cache()
        torch.manual_seed(TRAINING_SEED)
        torch.cuda.manual_seed_all(TRAINING_SEED)
        with load_indexed_m04a_data(
            sanitized_bundle_dir=data_binding.sanitized_bundle_dir,
            expected_sanitized_artifact_manifest_sha256=(
                SANITIZED_ARTIFACT_MANIFEST_SHA256
            ),
            training_rearc_cache_dir=data_binding.training_rearc_cache_dir,
            expected_training_cache_artifact_manifest_sha256=(
                TRAINING_CACHE_ARTIFACT_MANIFEST_SHA256
            ),
            validation_rearc_cache_dir=data_binding.validation_rearc_cache_dir,
            expected_validation_cache_artifact_manifest_sha256=(
                VALIDATION_CACHE_ARTIFACT_MANIFEST_SHA256
            ),
            sanitized_artifact_sources=(
                data_binding.directory_artifacts(SANITIZED_DIRECTORY)
                if data_binding.is_sealed_snapshot
                else None
            ),
            training_cache_artifact_sources=(
                data_binding.directory_artifacts(TRAINING_CACHE_DIRECTORY)
                if data_binding.is_sealed_snapshot
                else None
            ),
            validation_cache_artifact_sources=(
                data_binding.directory_artifacts(VALIDATION_CACHE_DIRECTORY)
                if data_binding.is_sealed_snapshot
                else None
            ),
        ) as indexed:
            model = GridCMLM().to(device="cuda:0").train()
            optimizer = build_adamw_optimizer(model)
            assert_grid_cmlm_invariants(model, require_cuda=True)
            assert_adamw_invariants(optimizer, model, expected_completed_updates=0)
            fresh_setup_wall_ns = time.perf_counter_ns() - setup_started
            if fresh_setup_wall_ns <= 0:
                raise RuntimeError("fresh primary setup timer did not advance")
            _budget_guard(preflight_result)

            for optimizer_step in range(1, OPTIMIZER_UPDATES + 1):
                _budget_guard(preflight_result)
                update = train_primary_update(
                    model,
                    optimizer,
                    indexed.training,
                    runtime_attestation=runtime_attestation,
                    optimizer_step=optimizer_step,
                    event_index=event_index,
                )
                ledger.append(update.ledger_row)
                event_index += 1
                completed_step = optimizer_step
                _budget_guard(preflight_result)
                if optimizer_step % PROGRESS_UPDATE_INTERVAL == 0:
                    _emit_progress(
                        progress_callback,
                        run_id=str(plan["run_id"]),
                        optimizer_step=optimizer_step,
                        event_count=len(ledger),
                        checkpoint_count=len(checkpoint_artifacts),
                    )
                if optimizer_step % VALIDATION_INTERVAL:
                    continue

                pass_index = optimizer_step // VALIDATION_INTERVAL
                validation = validate_literal_manifest(
                    model,
                    validation_manifest,
                    runtime_attestation=runtime_attestation,
                    optimizer_step=optimizer_step,
                    validation_pass_index=pass_index,
                    event_index_start=event_index,
                    expected_artifact_manifest_sha256=(
                        commitment.outer_artifact_manifest_sha256
                    ),
                    expected_jsonl_sha256=commitment.jsonl_sha256,
                    budget_check=lambda: _budget_guard(preflight_result),
                )
                ledger.extend(validation.ledger_rows)
                event_index += len(validation.ledger_rows)
                _budget_guard(preflight_result)

                checkpoint = _make_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    checkpoint_index=pass_index,
                    event_index=event_index,
                    model=model,
                    optimizer=optimizer,
                    config_sha256=preflight_result.expected_config_sha256,
                    runtime_source_sha256=(
                        preflight_result.expected_runtime_source_sha256
                    ),
                    test_source_sha256=preflight_result.expected_test_source_sha256,
                )
                checkpoint_artifacts.append(checkpoint)
                ledger.append(checkpoint.checkpoint_event)
                event_index += 1
                metric = make_validation_metric_row(
                    validation_pass_index=pass_index,
                    validation_episode_outer_manifest_sha256=(
                        commitment.outer_artifact_manifest_sha256
                    ),
                    validation_episode_manifest_sha256=commitment.jsonl_sha256,
                    episode_metrics=validation.evidence_episode_metrics(),
                    checkpoint_sha256=checkpoint.sha256,
                )
                validation_rows.append(metric)
                checkpoint_rows.append(
                    make_checkpoint_manifest_row(
                        checkpoint_index=pass_index,
                        checkpoint_sha256=checkpoint.sha256,
                        checkpoint_bytes=checkpoint.bytes,
                        checkpoint_event_id=str(
                            checkpoint.checkpoint_event["event_id"]
                        ),
                        validation_metric_id=str(metric["metric_id"]),
                    )
                )
                _budget_guard(preflight_result)
                _emit_progress(
                    progress_callback,
                    run_id=str(plan["run_id"]),
                    optimizer_step=optimizer_step,
                    event_count=len(ledger),
                    checkpoint_count=len(checkpoint_artifacts),
                )

            del optimizer
            del model
        gc.collect()
        torch.cuda.empty_cache()

        if (
            completed_step != OPTIMIZER_UPDATES
            or len(checkpoint_artifacts) != VALIDATION_PASS_COUNT
            or len(checkpoint_rows) != VALIDATION_PASS_COUNT
            or len(validation_rows) != VALIDATION_PASS_COUNT
        ):
            raise RuntimeError("primary campaign did not close every frozen count")

        ledger_bytes = serialize_jsonl(ledger)
        checkpoint_rows_bytes = serialize_jsonl(checkpoint_rows)
        validation_rows_bytes = serialize_jsonl(validation_rows)
        ledger_sha256 = _sha256_bytes(ledger_bytes)
        checkpoint_rows_sha256 = _sha256_bytes(checkpoint_rows_bytes)
        validation_rows_sha256 = _sha256_bytes(validation_rows_bytes)

        winning_metric = min(
            validation_rows,
            key=lambda row: (
                float.fromhex(str(row["parent_grouped_masked_cell_ce_hex"])),
                int(row["optimizer_step"]),
            ),
        )
        winning_index = int(winning_metric["validation_pass_index"])
        winning_checkpoint = checkpoint_artifacts[winning_index - 1]
        if winning_checkpoint.sha256 != winning_metric["checkpoint_sha256"]:
            raise RuntimeError("selected metric differs from checkpoint bytes")

        # Fix the winner while the lock is still attested, then close the owned
        # flock immediately.  The post-close endpoint is the authoritative end of
        # the billed interval; all manifest construction and replay below is CPU-only.
        selection_decided = _budget_guard(preflight_result)
        release_attempted = True
        release_held_gpu_lock(preflight_result.held_lock_handshake)
        release_succeeded = True
        selection_completed = time.perf_counter_ns()
        if selection_completed < selection_decided:
            raise RuntimeError("GPU-lock release endpoint predates final selection")
        assert_campaign_budget(
            selection_completed
            - int(lock_payload["acquisition_started_perf_counter_ns"])
        )
        lock_interval = make_training_lock_interval(
            (
                (
                    int(lock_payload["acquisition_started_perf_counter_ns"]),
                    selection_completed,
                ),
            )
        )
        selected_manifest = make_selected_checkpoint_manifest(
            checkpoint_rows=checkpoint_rows,
            validation_rows=validation_rows,
            checkpoint_manifests_sha256=checkpoint_rows_sha256,
            validation_metrics_sha256=validation_rows_sha256,
            training_cost_ledger_sha256=ledger_sha256,
            validation_episode_outer_manifest_sha256=(
                commitment.outer_artifact_manifest_sha256
            ),
            validation_episode_manifest_sha256=commitment.jsonl_sha256,
            lock_interval_id=str(lock_interval["interval_id"]),
            selection_completed_perf_counter_ns=selection_completed,
        )
        if (
            selected_manifest["selected_checkpoint_index"] != winning_index
            or selected_manifest["checkpoint_sha256"] != winning_checkpoint.sha256
            or selected_manifest["checkpoint_bytes"] != winning_checkpoint.bytes
        ):
            raise RuntimeError("final selected-checkpoint manifest drifted")

        ledger_closure = validate_training_cost_ledger(
            ledger,
            validation_episodes_per_pass=len(validation_manifest.rows),
            fresh_setup_wall_time_ns=fresh_setup_wall_ns,
            lock_interval=lock_interval,
            preflight_endpoint_wall_time_ns=preflight_result.report[
                "budget_projection"
            ]["measured_preflight_total_wall_ns"],
        )
        closure = _training_closure(ledger_closure, preflight_result.report)
        timing = {
            field: int(ledger_closure[field])
            for field in evidence.TRAINING_ARTIFACT_TIMING_FIELDS
        }
        training_summary = make_training_cost_summary(
            ledger_sha256=ledger_sha256,
            closure=closure,
            timing=timing,
            lock_interval=lock_interval,
        )
        selected_manifest_bytes = serialize_json(selected_manifest)
        validate_training_checkpoint_artifacts(
            checkpoint_rows,
            validation_rows,
            selected_manifest,
            training_ledger_rows=ledger,
            checkpoint_manifests_sha256=checkpoint_rows_sha256,
            validation_metrics_sha256=validation_rows_sha256,
            training_cost_ledger_sha256=ledger_sha256,
            validation_episode_outer_manifest_sha256=(
                commitment.outer_artifact_manifest_sha256
            ),
            validation_episode_manifest_sha256=commitment.jsonl_sha256,
            validation_episode_rows=validation_manifest.rows,
            selected_checkpoint_sha256=winning_checkpoint.sha256,
            selected_checkpoint_bytes=winning_checkpoint.bytes,
            lock_interval_id=str(lock_interval["interval_id"]),
            lock_selection_completed_perf_counter_ns=selection_completed,
        )

        # The GPU cap ends at final selection.  Publication is CPU-only, but the
        # immutable inputs and data closure are rechecked before the bundle moves.
        if runtime_source_fingerprint() != plan["runtime_source_fingerprint_sha256"]:
            raise RuntimeError("reviewed runtime source changed during training")
        final_data_binding = revalidate_production_data_binding(data_binding)
        if final_data_binding != data_binding:
            raise RuntimeError("production data binding changed during training")
        if (
            winning_checkpoint.path.stat().st_size != winning_checkpoint.bytes
            or hashlib.sha256(winning_checkpoint.path.read_bytes()).hexdigest()
            != winning_checkpoint.sha256
        ):
            raise RuntimeError("selected checkpoint changed before publication")

        environment_bytes = serialize_json(dict(environment_manifest))
        overhead_bytes = serialize_json(preflight_result.overhead_cost_probe_report)
        lock_bytes = preflight_result.held_lock_handshake.artifact.snapshot
        generated_bytes: dict[str, bytes] = {
            name: snapshots[name]
            for name in evidence.TRAINING_ARTIFACT_FILES
            if name in snapshots
        }
        generated_bytes.update(
            {
                "checkpoint_manifests.jsonl": checkpoint_rows_bytes,
                "environment_manifest.json": environment_bytes,
                "launch_plan.json": plan_artifact.snapshot,
                "lock_handshake_artifact.json": lock_bytes,
                "optimizer_schedule_state.json": serialize_json(
                    {
                        "schema": OPTIMIZER_SCHEDULE_STATE_SCHEMA_VERSION,
                        "optimizer_step": OPTIMIZER_UPDATES,
                        "learning_rate_hex": learning_rate_for_update(
                            OPTIMIZER_UPDATES
                        ).hex(),
                        "checkpoint_count": VALIDATION_PASS_COUNT,
                        "production_data_closure_id": PRODUCTION_DATA_CLOSURE_ID,
                    }
                ),
                "preflight.json": serialize_json(preflight_result.report),
                "preflight_diagnostic_checkpoint.pt": (
                    preflight_result.diagnostic_checkpoint_snapshot
                ),
                "preflight_overhead_cost_report.json": overhead_bytes,
                "resume_manifests.jsonl": b"",
                "selected_checkpoint_manifest.json": selected_manifest_bytes,
                "training_cost_ledger.jsonl": ledger_bytes,
                "training_cost_summary.json": serialize_json(training_summary),
                "validation_metrics.jsonl": validation_rows_bytes,
            }
        )
        expected_without_selected = set(evidence.TRAINING_ARTIFACT_FILES) - {
            "selected_checkpoint.pt"
        }
        if set(generated_bytes) != expected_without_selected:
            raise RuntimeError("training payload files do not match the frozen set")
        artifact_paths = _write_payload_files(payload_dir, generated_bytes)
        artifact_paths["selected_checkpoint.pt"] = winning_checkpoint.path

        lineage = _artifact_lineage(
            snapshots=snapshots,
            environment_bytes=environment_bytes,
            launch_plan_sha256=plan_artifact.artifact_sha256,
            lock_handshake_sha256=(
                preflight_result.held_lock_handshake.artifact.artifact_sha256
            ),
            overhead_bytes=overhead_bytes,
            diagnostic_checkpoint=preflight_result.diagnostic_checkpoint_snapshot,
            ledger_sha256=ledger_sha256,
            selected_manifest_bytes=selected_manifest_bytes,
            selected_checkpoint_sha256=winning_checkpoint.sha256,
            validation_manifest=validation_manifest,
            python_runtime_lock=python_runtime_lock,
        )
        published = publish_training_artifact_bundle(
            output,
            lineage=lineage,
            artifact_files=artifact_paths,
            closure=closure,
            timing=timing,
        )
        publication_committed = True
        report = {
            "status": "published_complete_m04a_training_campaign",
            "run_id": plan["run_id"],
            "output_dir": str(output),
            "training_artifact_id": published.manifest["semantic_id"],
            "training_artifact_manifest_sha256": _sha256_bytes(
                serialize_json(published.manifest)
            ),
            "outer_artifact_manifest_sha256": _sha256_bytes(
                serialize_json(published.outer_manifest)
            ),
            "selected_checkpoint_index": winning_index,
            "selected_checkpoint_sha256": winning_checkpoint.sha256,
            "optimizer_updates": OPTIMIZER_UPDATES,
            "validation_passes": VALIDATION_PASS_COUNT,
            "gpu_lock_wall_time_ns": lock_interval["gpu_lock_wall_time_ns"],
            "production_data_closure_id": PRODUCTION_DATA_CLOSURE_ID,
            "training_evidence_eligible": True,
        }
        return CampaignPublication(
            report=report,
            selected_checkpoint_manifest=selected_manifest,
            training_artifact_manifest=published.manifest,
        )
    except BaseException as cause:
        if publication_committed:
            try:
                setattr(cause, "campaign_output_committed", True)
            except Exception:
                pass
            raise
        if not release_attempted:
            release_attempted = True
            try:
                release_held_gpu_lock(preflight_result.held_lock_handshake)
            except BaseException:
                release_succeeded = False
            else:
                release_succeeded = True
        failure_code = _failure_code(cause)
        try:
            _write_failure_receipt(
                working,
                run_id=str(plan["run_id"]),
                failure_code=failure_code,
                optimizer_step=completed_step,
                cause_type=type(cause).__name__,
                lock_release_attempted=release_attempted,
                lock_release_succeeded=release_succeeded,
            )
        except BaseException as receipt_error:
            try:
                setattr(cause, "campaign_failure_receipt_error", repr(receipt_error))
            except Exception:
                pass
        raise CampaignFailure(
            failure_code,
            optimizer_step=completed_step,
            working_dir=working,
            cause_type=type(cause).__name__,
        ) from cause


def run_primary_training_campaign(
    *,
    preflight_result: PreflightResult,
    runtime_attestation: FrozenRuntimeAttestation,
    environment_manifest: Mapping[str, Any],
    python_runtime_lock: PythonRuntimeLockArtifact,
    visible_input_snapshots: Mapping[str, bytes],
    validation_manifest_dir: str | Path,
    production_data_root: str | Path,
    production_data_binding: ProductionDataBinding | None = None,
    working_dir: str | Path,
    output_dir: str | Path,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> CampaignPublication:
    """Own the preflight lock lifecycle across preparation, training, and failure."""

    try:
        if type(preflight_result) is not PreflightResult:
            raise TypeError("formal campaign requires an exact PreflightResult")
        publication = _run_primary_training_campaign_held(
            preflight_result=preflight_result,
            runtime_attestation=runtime_attestation,
            environment_manifest=environment_manifest,
            python_runtime_lock=python_runtime_lock,
            visible_input_snapshots=visible_input_snapshots,
            validation_manifest_dir=validation_manifest_dir,
            production_data_root=production_data_root,
            production_data_binding=production_data_binding,
            working_dir=working_dir,
            output_dir=output_dir,
            progress_callback=progress_callback,
        )
    except BaseException as cause:
        output_committed = bool(getattr(cause, "campaign_output_committed", False))
        release_succeeded = False
        release_error: BaseException | None = None
        held_lock = getattr(preflight_result, "held_lock_handshake", None)
        try:
            if held_lock is None:
                raise RuntimeError("campaign ownership lacks its held GPU lock")
            release_held_gpu_lock(held_lock)
        except BaseException as error:
            release_error = error
        else:
            release_succeeded = True
        if not output_committed:
            try:
                plan = preflight_result.launch_plan_artifact.payload
                ensure_campaign_failure_receipt(
                    run_root=plan["run_root"],
                    working_dir=working_dir,
                    run_id=str(plan["run_id"]),
                    failure_code=_failure_code(cause),
                    optimizer_step=int(getattr(cause, "optimizer_step", 0)),
                    cause_type=str(
                        getattr(cause, "cause_type", type(cause).__name__)
                    ),
                    lock_release_attempted=True,
                    lock_release_succeeded=release_succeeded,
                )
            except BaseException as receipt_error:
                try:
                    setattr(
                        cause,
                        "campaign_failure_receipt_error",
                        repr(receipt_error),
                    )
                except Exception:
                    pass
        if release_error is not None:
            raise RuntimeError(
                "campaign failed and the GPU-lock cleanup also failed"
            ) from release_error
        raise
    # The inner campaign releases at the final-selection endpoint.  This call is
    # lifecycle-idempotent and covers any future early-return regression.
    release_held_gpu_lock(preflight_result.held_lock_handshake)
    return publication


__all__ = [
    "CAMPAIGN_FAILURE_SCHEMA_VERSION",
    "CAMPAIGN_PROGRESS_SCHEMA_VERSION",
    "CampaignFailure",
    "CampaignPublication",
    "CheckpointArtifact",
    "ensure_campaign_failure_receipt",
    "OPTIMIZER_SCHEDULE_STATE_SCHEMA_VERSION",
    "PROGRESS_UPDATE_INTERVAL",
    "run_primary_training_campaign",
    "verify_checkpoint_roundtrip",
]
