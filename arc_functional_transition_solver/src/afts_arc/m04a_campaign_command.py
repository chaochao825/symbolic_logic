"""Production command orchestration for the formal M04a training campaign.

The module remains Torch-free at import time.  It repeats every externally
committed preflight input check, executes exact preflight, and transfers the still
held inherited GPU lock directly to :mod:`afts_arc.m04a_campaign` in the same
process.  Unlike ``preflight-m04a``, a PASS is not published as a diagnostic gate;
its report and nonselectable diagnostic checkpoint become parents of the complete
training bundle.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from . import m04a_preflight_command as preflight_command
from .m04a_evidence import (
    validate_environment_manifest_artifact,
    verify_test_source_snapshot_zip,
)
from .m04a_launch_plan import read_launch_plan_artifact
from .m04a_lock_handshake import read_lock_handshake_artifact
from .m04a_production_data_contract import (
    revalidate_production_data_binding,
    validate_sealed_production_data_snapshot,
)
from .m04a_python_runtime_lock import (
    read_python_runtime_lock_artifact,
    validate_imported_torch_runtime,
    validate_live_python_runtime_lock,
)
from .m04a_train_contract import training_config_sha256
from .manifest import (
    runtime_source_fingerprint,
    serialize_json,
    serialize_jsonl,
    verify_source_snapshot_zip,
)


def _fixed_production_data_root(
    value: str | Path, *, remote_project_root: str, run_root: Path, visible_root: Path
) -> Path:
    requested = Path(os.path.abspath(os.fspath(Path(value).expanduser())))
    expected = Path(remote_project_root) / "data"
    if requested != expected:
        raise ValueError("production data root must be remote_project_root/data")
    for forbidden in (run_root, visible_root):
        try:
            requested.relative_to(forbidden)
        except ValueError:
            pass
        else:
            raise ValueError("production data root overlaps a mutable campaign root")
        try:
            forbidden.relative_to(requested)
        except ValueError:
            pass
        else:
            raise ValueError("production data root contains another campaign root")
    return requested


def _failure_artifacts(
    *,
    environment: Mapping[str, Any],
    launch_snapshot: bytes,
    lock_snapshot: bytes,
    runtime_lock_snapshot: bytes,
    failure: BaseException,
) -> tuple[dict[str, bytes], dict[str, Any]]:
    report = getattr(failure, "report", None)
    ledger = list(getattr(failure, "ledger_rows", ()))
    artifacts = {
        "environment_manifest.json": serialize_json(dict(environment)),
        "launch_plan.json": launch_snapshot,
        "lock_handshake_artifact.json": lock_snapshot,
        "preflight_failure.json": serialize_json(report),
        "preflight_training_cost_ledger.jsonl": serialize_jsonl(ledger),
        "python-runtime-lock.json": runtime_lock_snapshot,
    }
    if isinstance(report, dict) and report.get("evidence_completeness") == (
        "full_budget_probe"
    ):
        artifacts.update(
            {
                "preflight_diagnostic_checkpoint.pt": getattr(
                    failure, "diagnostic_checkpoint_snapshot"
                ),
                "preflight_inference_summary.json": serialize_json(
                    getattr(failure, "inference_summary")
                ),
                "preflight_overhead_cost_report.json": serialize_json(
                    getattr(failure, "overhead_cost_probe_report")
                ),
                "preflight_training_summary.json": serialize_json(
                    getattr(failure, "training_summary")
                ),
            }
        )
    return artifacts, report


def run_training_campaign_command(
    *,
    launch_plan_path: str | Path,
    launch_plan_sha256: str,
    python_runtime_lock_path: str | Path,
    python_runtime_lock_sha256: str,
    validation_manifest_dir: str | Path,
    lock_handshake_path: str | Path,
    lock_handshake_sha256: str,
    gpu_uuid: str,
    visible_root: str | Path,
    launcher_path: str | Path,
    cost_probe_scratch_dir: str | Path,
    production_data_root: str | Path,
    working_dir: str | Path,
    output_dir: str | Path,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run exact preflight and the complete primary campaign under one lock."""

    launch_artifact = read_launch_plan_artifact(
        launch_plan_path,
        expected_artifact_sha256=launch_plan_sha256,
    )
    plan = launch_artifact.payload
    if plan["training_config_sha256"] != training_config_sha256():
        raise ValueError("launch plan differs from the frozen training config")
    gpu = preflight_command._validate_process_identity(plan, gpu_uuid=gpu_uuid)
    root, visible_files, visible_snapshots = (
        preflight_command._visible_input_snapshots(
            visible_root,
            expected_input_artifacts=plan["expected_input_artifacts"],
            conda_explicit_sha256=plan["conda_explicit_sha256"],
        )
    )
    run_root, scratch, output, launcher = preflight_command._validate_path_layout(
        plan=plan,
        visible_root=root,
        launch_plan_path=launch_plan_path,
        python_runtime_lock_path=python_runtime_lock_path,
        lock_handshake_path=lock_handshake_path,
        validation_manifest_dir=validation_manifest_dir,
        launcher_path=launcher_path,
        cost_probe_scratch_dir=cost_probe_scratch_dir,
        output_dir=output_dir,
    )
    expected_working = run_root / "campaign-working"
    if Path(os.path.abspath(os.fspath(Path(working_dir).expanduser()))) != (
        expected_working
    ):
        raise ValueError("campaign working directory must be run_root/campaign-working")
    data_root = _fixed_production_data_root(
        production_data_root,
        remote_project_root=str(plan["remote_project_root"]),
        run_root=run_root,
        visible_root=root,
    )
    data_binding = validate_sealed_production_data_snapshot(data_root)
    run_root_identity = run_root.stat(follow_symlinks=False)
    output_parent_identity = output.parent.stat(follow_symlinks=False)

    launcher_snapshot = preflight_command._read_committed_regular(
        launcher,
        expected_sha256=plan["expected_input_artifacts"]["remote_launcher.py"],
        label="executing M04a launcher",
        maximum_bytes=1024 * 1024,
    )
    if (
        os.environ.get("AFTS_EVIDENCE_LAUNCHER") is None
        or preflight_command._absolute_lexical(
            os.environ["AFTS_EVIDENCE_LAUNCHER"]
        )
        != launcher
    ):
        raise RuntimeError("--launcher-path differs from the executing launcher")
    if launcher_snapshot != visible_snapshots["remote_launcher.py"]:
        raise ValueError("executing launcher differs from the visible reviewed copy")

    expected_runtime_lock_sha256 = plan["expected_input_artifacts"][
        "python-runtime-lock.json"
    ]
    if python_runtime_lock_sha256 != expected_runtime_lock_sha256:
        raise ValueError("runtime-lock CLI SHA differs from launch plan")
    python_runtime_lock = read_python_runtime_lock_artifact(
        python_runtime_lock_path,
        expected_artifact_sha256=expected_runtime_lock_sha256,
    )
    if python_runtime_lock.snapshot != visible_snapshots["python-runtime-lock.json"]:
        raise ValueError("runtime-lock reader snapshot differs from visible input")
    runtime_lock_live_kwargs = preflight_command._runtime_lock_live_kwargs(
        plan=plan,
        runtime_lock=python_runtime_lock,
    )
    fresh_runtime_lock = validate_live_python_runtime_lock(
        python_runtime_lock,
        expected_artifact_sha256=expected_runtime_lock_sha256,
        **runtime_lock_live_kwargs,
    )

    runtime_fingerprint = plan["runtime_source_fingerprint_sha256"]
    tests_fingerprint = plan["test_source_fingerprint_sha256"]
    verify_source_snapshot_zip(
        visible_snapshots["reviewed_runtime_source.zip"],
        expected_fingerprint_sha256=runtime_fingerprint,
    )
    verify_test_source_snapshot_zip(
        visible_snapshots["reviewed_test_snapshot.zip"],
        expected_fingerprint_sha256=tests_fingerprint,
    )
    if runtime_source_fingerprint() != runtime_fingerprint:
        raise RuntimeError("executing runtime source differs from the launch plan")

    validation_commitment = preflight_command._validation_commitment_from_plan(
        plan,
        validation_manifest_dir=validation_manifest_dir,
        visible_snapshots=visible_snapshots,
    )
    lock_artifact = read_lock_handshake_artifact(
        lock_handshake_path,
        expected_artifact_sha256=lock_handshake_sha256,
        expected_run_id=plan["run_id"],
        expected_gpu_uuid=gpu,
        expected_launcher_sha256=plan["expected_input_artifacts"][
            "remote_launcher.py"
        ],
        expected_launch_plan_sha256=launch_artifact.artifact_sha256,
        expected_remote_project_root=plan["remote_project_root"],
        expected_attempt_nonce=plan["attempt_nonce"],
    )

    dependencies = preflight_command._load_torch_dependencies()
    from .m04a_campaign import (
        ensure_campaign_failure_receipt,
        run_primary_training_campaign,
    )

    validate_imported_torch_runtime(
        python_runtime_lock,
        expected_artifact_sha256=expected_runtime_lock_sha256,
        imported_torch=dependencies.imported_torch,
    )
    attestation = dependencies.configure_deterministic_cuda()
    environment = dependencies.environment_manifest(
        attestation=attestation,
        launcher_path=launcher,
        runtime_source_sha256=runtime_fingerprint,
        test_source_sha256=tests_fingerprint,
        visible_root=root,
        expected_visible_files=visible_files,
        conda_explicit_path=root / preflight_command.CONDA_EXPLICIT_BASENAME,
        expected_conda_explicit_sha256=plan["conda_explicit_sha256"],
        python_runtime_lock_sha256=expected_runtime_lock_sha256,
        python_runtime_lock_id=fresh_runtime_lock["runtime_lock_id"],
        python_implementation=fresh_runtime_lock["python"]["implementation"],
        python_version=fresh_runtime_lock["python"]["version"],
        python_executable_sha256=fresh_runtime_lock["python"]["executable_sha256"],
        python_executable_bytes=fresh_runtime_lock["python"]["executable_bytes"],
    )
    environment = preflight_command._bind_environment_to_plan(
        environment,
        plan=plan,
        gpu_uuid=gpu,
        visible_root=root,
        visible_files=visible_files,
        launcher=launcher,
        python_runtime_lock=python_runtime_lock,
        runtime_lock_payload=fresh_runtime_lock,
    )
    validate_environment_manifest_artifact(
        environment,
        launch_plan=plan,
        expected_gpu_uuid=gpu,
        expected_launcher_sha256=plan["expected_input_artifacts"][
            "remote_launcher.py"
        ],
        python_runtime_lock=fresh_runtime_lock,
    )

    result: Any | None = None
    campaign_owns_lock = False
    try:
        try:
            result = dependencies.run_exact_preflight(
                runtime_attestation=attestation,
                validation_manifest_commitment=validation_commitment,
                launch_plan_artifact=launch_artifact,
                run_root=run_root,
                cost_probe_scratch_dir=scratch,
                config_sha256=plan["training_config_sha256"],
                runtime_source_sha256=runtime_fingerprint,
                test_source_sha256=tests_fingerprint,
                lock_handshake_artifact=lock_artifact,
            )
        except dependencies.preflight_failure_type as failure:
            failure_report = getattr(failure, "report", None)
            failure_ledger = list(getattr(failure, "ledger_rows", ()))
            overhead = getattr(failure, "overhead_cost_probe_report", None)
            training = getattr(failure, "training_summary", None)
            inference = getattr(failure, "inference_summary", None)
            diagnostic = getattr(failure, "diagnostic_checkpoint_snapshot", None)
            preflight_command._validate_failure_report_payload(
                failure_report,
                failure_ledger,
                run_id=plan["run_id"],
                gpu_uuid=gpu,
                validation_manifest_commitment=plan[
                    "validation_manifest_commitment"
                ],
                lock_handshake_artifact_sha256=lock_artifact.artifact_sha256,
                lock_handshake_payload=lock_artifact.payload,
                config_sha256=plan["training_config_sha256"],
                runtime_source_sha256=runtime_fingerprint,
                test_source_sha256=tests_fingerprint,
                overhead_report=overhead,
                training_summary=training,
                inference_summary=inference,
                diagnostic_checkpoint=diagnostic,
            )
            failure_output = preflight_command._failure_output_path(output)
            preflight_command._assert_publication_parent_unchanged(
                run_root=run_root,
                output=failure_output,
                expected_run_root=run_root_identity,
                expected_output_parent=output_parent_identity,
            )
            failure_artifacts, normalized_report = _failure_artifacts(
                environment=environment,
                launch_snapshot=launch_artifact.snapshot,
                lock_snapshot=lock_artifact.snapshot,
                runtime_lock_snapshot=python_runtime_lock.snapshot,
                failure=failure,
            )
            failure_manifest = preflight_command._publish_preflight_failure(
                failure_output,
                artifacts=failure_artifacts,
                run_id=plan["run_id"],
                launch_plan=launch_artifact,
                lock_artifact=lock_artifact,
                failure_report=normalized_report,
                python_runtime_lock=python_runtime_lock,
                expected_output_parent=output_parent_identity,
            )
            setattr(failure, "published_failure_dir", str(failure_output))
            setattr(failure, "published_failure_id", failure_manifest["failure_id"])
            raise

        if (
            not isinstance(result.report, dict)
            or result.report.get("status") != "PASS"
            or result.overhead_cost_probe_report.get("status") != "PASS"
            or len(result.ledger_rows) != 100
            or type(result.diagnostic_checkpoint_snapshot) is not bytes
            or result.held_lock_handshake.artifact.snapshot != lock_artifact.snapshot
        ):
            raise ValueError("exact preflight result is not training-campaign ready")
        final_root, final_visible_files, final_visible_snapshots = (
            preflight_command._visible_input_snapshots(
                root,
                expected_input_artifacts=plan["expected_input_artifacts"],
                conda_explicit_sha256=plan["conda_explicit_sha256"],
            )
        )
        if (
            not os.path.samefile(final_root, root)
            or final_visible_files != visible_files
            or final_visible_snapshots != visible_snapshots
            or runtime_source_fingerprint() != runtime_fingerprint
        ):
            raise RuntimeError("committed source/input changed during exact preflight")
        validate_live_python_runtime_lock(
            python_runtime_lock,
            expected_artifact_sha256=expected_runtime_lock_sha256,
            **runtime_lock_live_kwargs,
        )
        validate_imported_torch_runtime(
            python_runtime_lock,
            expected_artifact_sha256=expected_runtime_lock_sha256,
            imported_torch=dependencies.imported_torch,
        )
        revalidate_production_data_binding(data_binding)
        preflight_command._assert_publication_parent_unchanged(
            run_root=run_root,
            output=output,
            expected_run_root=run_root_identity,
            expected_output_parent=output_parent_identity,
        )

        campaign_owns_lock = True
        publication = run_primary_training_campaign(
            preflight_result=result,
            runtime_attestation=attestation,
            environment_manifest=environment,
            python_runtime_lock=python_runtime_lock,
            visible_input_snapshots={
                name: visible_snapshots[name]
                for name in plan["expected_input_artifacts"]
            },
            validation_manifest_dir=validation_manifest_dir,
            production_data_root=data_root,
            production_data_binding=data_binding,
            working_dir=working_dir,
            output_dir=output,
            progress_callback=progress_callback,
        )
        return publication.report
    finally:
        if result is not None and not campaign_owns_lock:
            active_exception = sys.exc_info()[1]
            release_succeeded = False
            release_error: BaseException | None = None
            try:
                dependencies.release_held_gpu_lock(result.held_lock_handshake)
            except BaseException as error:
                release_error = error
            else:
                release_succeeded = True
            if active_exception is not None:
                try:
                    ensure_campaign_failure_receipt(
                        run_root=run_root,
                        working_dir=working_dir,
                        run_id=str(plan["run_id"]),
                        failure_code="CAMPAIGN_INCOMPLETE",
                        optimizer_step=0,
                        cause_type=type(active_exception).__name__,
                        lock_release_attempted=True,
                        lock_release_succeeded=release_succeeded,
                    )
                except BaseException as receipt_error:
                    try:
                        setattr(
                            active_exception,
                            "campaign_failure_receipt_error",
                            repr(receipt_error),
                        )
                    except Exception:
                        pass
            if release_error is not None:
                if active_exception is None:
                    raise release_error
                raise RuntimeError(
                    "campaign preparation failed and GPU-lock release also failed"
                ) from release_error


__all__ = ["run_training_campaign_command"]
