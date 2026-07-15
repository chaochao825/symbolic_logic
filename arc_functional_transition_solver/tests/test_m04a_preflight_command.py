from __future__ import annotations

import hashlib
import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from afts_arc import cli
from afts_arc import m04a_preflight_command as command
from afts_arc.m04a_contract import canonical_sha256
from afts_arc.m04a_evidence import read_m04a_artifact_bundle
from afts_arc.m04a_launch_plan import EXPECTED_INPUT_ARTIFACT_PATHS
from afts_arc.m04a_train_contract import training_config_sha256
from afts_arc.manifest import serialize_json
from m04a_preflight_failure_fixture import _budget_failure_fixture


GPU_UUID = "GPU-00000000-0000-0000-0000-000000000001"


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


class _FixturePreflightFailure(RuntimeError):
    pass


def _partial_preflight_failure() -> _FixturePreflightFailure:
    semantic: dict[str, object] = {
        "schema": command.PREFLIGHT_FAILURE_REPORT_SCHEMA_VERSION,
        "status": "OOM",
        "failure_code": "OOM",
        "evidence_completeness": "partial",
        "fixture_id": command._expected_preflight_fixture_id(),
        "runtime": None,
        "validation_manifest_commitment": None,
        "lock_handshake_artifact_sha256": None,
        "preflight_started_perf_counter_ns": None,
        "completed_updates": 0,
        "ledger_row_count": 0,
        "ledger_rows_sha256": canonical_sha256([]),
        "training_summary": None,
        "inference_summary": None,
        "overhead_cost_probe_id": None,
        "budget_projection": None,
        "diagnostic_checkpoint_commitment": None,
        "dataset_file_reads": 0,
        "training_checkpoint_writes": 0,
        "diagnostic_checkpoint_writes": 0,
        "fallback_used": False,
        "training_evidence_eligible": False,
        "same_run_retry_allowed": False,
    }
    failure = _FixturePreflightFailure("OOM: fixture terminal failure")
    failure.report = {**semantic, "report_id": canonical_sha256(semantic)}
    failure.ledger_rows = ()
    failure.overhead_cost_probe_report = None
    failure.training_summary = None
    failure.inference_summary = None
    failure.diagnostic_checkpoint_snapshot = None
    return failure


def _budget_preflight_failure(fixture: "_CommandFixture") -> _FixturePreflightFailure:
    report, rows, parents, diagnostic = _budget_failure_fixture()
    report = copy.deepcopy(report)
    report["validation_manifest_commitment"] = fixture.plan[
        "validation_manifest_commitment"
    ]
    report["lock_handshake_artifact_sha256"] = (
        fixture.lock_artifact.artifact_sha256
    )
    overhead = copy.deepcopy(parents["overhead_cost_probe_report"])
    overhead["runtime_source_sha256"] = fixture.plan[
        "runtime_source_fingerprint_sha256"
    ]
    overhead["test_source_sha256"] = fixture.plan[
        "test_source_fingerprint_sha256"
    ]
    overhead_semantic = dict(overhead)
    overhead_semantic.pop("probe_id")
    overhead["probe_id"] = canonical_sha256(overhead_semantic)
    report["overhead_cost_probe_id"] = overhead["probe_id"]
    report_semantic = dict(report)
    report_semantic.pop("report_id")
    report["report_id"] = canonical_sha256(report_semantic)
    failure = _FixturePreflightFailure(
        "BUDGET_EXCEEDED: fixture terminal failure"
    )
    failure.report = report
    failure.ledger_rows = rows
    failure.overhead_cost_probe_report = overhead
    failure.training_summary = report["training_summary"]
    failure.inference_summary = report["inference_summary"]
    failure.diagnostic_checkpoint_snapshot = diagnostic
    return failure


def _plan() -> dict[str, object]:
    inputs = {name: _digest(name) for name in EXPECTED_INPUT_ARTIFACT_PATHS}
    validation_semantic: dict[str, object] = {
        "schema": "afts-m04a-validation-manifest-commitment/v0.1",
        "outer_artifact_manifest_sha256": inputs[
            "validation_episode_outer_manifest.json"
        ],
        "jsonl_sha256": inputs["validation_episode_manifest.jsonl"],
        "summary_id": _digest("validation-summary"),
        "row_count": 16,
    }
    validation = {
        **validation_semantic,
        "commitment_id": canonical_sha256(validation_semantic),
    }
    semantic = {
        "schema": "afts-m04a-launch-plan/v0.2",
        "attempt_nonce": _digest("attempt"),
        "run_id": "m04a-grid-cmlm-v0.1-seed20260711-r1",
        "remote_project_root": "/srv/afts",
        "run_root": "/srv/afts/runs/m04a-grid-cmlm-v0.1-seed20260711-r1",
        "expected_input_artifacts": inputs,
        "runtime_source_fingerprint_sha256": _digest("runtime-fingerprint"),
        "test_source_fingerprint_sha256": _digest("test-fingerprint"),
        "training_config_sha256": training_config_sha256(),
        "validation_manifest_commitment": validation,
        "conda_explicit_sha256": _digest("conda-explicit"),
        "ordered_import_roots": [
            "/srv/afts/src",
            "/site/purelib",
            "/site/platlib",
        ],
    }
    return {**semantic, "launch_plan_id": canonical_sha256(semantic)}


class _CommandFixture:
    def __init__(self, root: Path, *, run_failure: BaseException | None = None) -> None:
        self.root = root
        self.plan = _plan()
        self.run_root = root / "run"
        self.run_root.mkdir()
        self.run_root.chmod(0o700)
        self.output = self.run_root / "gate"
        self.scratch = self.run_root / "scratch"
        self.visible_root = root / "visible"
        self.visible_root.mkdir()
        self.visible_root.chmod(0o700)
        self.launcher = self.visible_root / "remote_launcher.py"
        self.launcher_bytes = b"raise SystemExit(0)\n"
        self.launcher.write_bytes(self.launcher_bytes)
        self.plan["expected_input_artifacts"]["remote_launcher.py"] = hashlib.sha256(
            self.launcher_bytes
        ).hexdigest()
        self.runtime_lock_payload = {
            "runtime_lock_id": _digest("runtime-lock-id"),
            "python": {
                "implementation": "CPython",
                "version": "3.10.20",
                "executable_lexical_path": (
                    "/opt/mixbit/bin/python"
                ),
                "executable_sha256": _digest("python-executable"),
                "executable_bytes": 17_460_208,
            },
        }
        self.runtime_lock_snapshot = serialize_json(self.runtime_lock_payload)
        self.plan["expected_input_artifacts"]["python-runtime-lock.json"] = (
            hashlib.sha256(self.runtime_lock_snapshot).hexdigest()
        )
        launch_semantic = dict(self.plan)
        launch_semantic.pop("launch_plan_id")
        self.plan["launch_plan_id"] = canonical_sha256(launch_semantic)
        self.visible_files = dict(self.plan["expected_input_artifacts"])
        self.visible_files[command.CONDA_EXPLICIT_BASENAME] = self.plan[
            "conda_explicit_sha256"
        ]
        self.visible_snapshots = {
            name: name.encode("ascii") for name in self.visible_files
        }
        self.visible_snapshots["remote_launcher.py"] = self.launcher_bytes
        self.visible_snapshots["python-runtime-lock.json"] = self.runtime_lock_snapshot
        self.runtime_lock_artifact = SimpleNamespace(
            payload=self.runtime_lock_payload,
            artifact_sha256=self.plan["expected_input_artifacts"][
                "python-runtime-lock.json"
            ],
            snapshot=self.runtime_lock_snapshot,
        )
        launch_snapshot = serialize_json(self.plan)
        self.launch_artifact = SimpleNamespace(
            payload=self.plan,
            artifact_sha256=hashlib.sha256(launch_snapshot).hexdigest(),
            snapshot=launch_snapshot,
        )
        lock_semantic: dict[str, object] = {
            "schema": "afts-m04a-held-gpu-lock-handshake/v0.2",
            "run_id": self.plan["run_id"],
            "gpu_uuid": GPU_UUID,
            "lock_path": f"{self.plan['remote_project_root']}/locks/gpu-{GPU_UUID}.lock",
            "lock_st_dev": 1,
            "lock_st_ino": 2,
            "lock_holder_pid": 3,
            "lock_holder_start_ticks": 4,
            "inherited_lock_fd": 5,
            "attempt_nonce": self.plan["attempt_nonce"],
            "boot_id": "00000000-0000-0000-0000-000000000001",
            "acquisition_started_perf_counter_ns": 1_000,
            "eligibility_rechecked_perf_counter_ns": 1_001,
            "handshake_completed_perf_counter_ns": 1_002,
            "wall_ns": 2,
            "launcher_sha256": self.plan["expected_input_artifacts"][
                "remote_launcher.py"
            ],
            "launch_plan_sha256": self.launch_artifact.artifact_sha256,
            "source": "reviewed_launcher_inherited_posix_flock",
        }
        self.lock_payload = {
            **lock_semantic,
            "handshake_id": canonical_sha256(lock_semantic),
        }
        lock_snapshot = serialize_json(self.lock_payload)
        self.lock_artifact = SimpleNamespace(
            payload=self.lock_payload,
            artifact_sha256=hashlib.sha256(lock_snapshot).hexdigest(),
            snapshot=lock_snapshot,
        )
        self.validation_commitment = SimpleNamespace(
            to_json_dict=lambda: self.plan["validation_manifest_commitment"]
        )
        self.attestation = object()
        self.diagnostic = b"diagnostic-checkpoint"
        self.report = {
            "schema": "afts-grid-cmlm-preflight-report/v0.4",
            "status": "PASS",
            "report_id": _digest("preflight-report"),
            "runtime": {"run_id": self.plan["run_id"], "gpu_uuid": GPU_UUID},
            "overhead_cost_probe_id": _digest("overhead"),
            "lock_handshake_artifact_sha256": self.lock_artifact.artifact_sha256,
        }
        self.overhead = {
            "schema": "afts-grid-cmlm-campaign-overhead-probe/v0.1",
            "status": "PASS",
            "run_id": self.plan["run_id"],
            "probe_id": _digest("overhead"),
            "checkpoint_cost_probe": {
                "checkpoint_bytes": len(self.diagnostic),
                "checkpoint_sha256": hashlib.sha256(self.diagnostic).hexdigest(),
            },
        }
        held = SimpleNamespace(artifact=self.lock_artifact)
        self.result = SimpleNamespace(
            report=self.report,
            ledger_rows=tuple({"event_index": index} for index in range(100)),
            overhead_cost_probe_report=self.overhead,
            diagnostic_checkpoint_snapshot=self.diagnostic,
            held_lock_handshake=held,
        )
        self.release = mock.Mock()
        if run_failure is None:
            run = mock.Mock(return_value=self.result)
        else:
            run = mock.Mock(side_effect=run_failure)
        self.run = run

        def environment_manifest(**kwargs):
            return {
                "schema": "afts-m04a-environment/v0.3",
                "python": "3.10.20",
                "python_implementation": "CPython",
                "python_version": "3.10.20",
                "python_executable": self.runtime_lock_payload["python"][
                    "executable_lexical_path"
                ],
                "python_executable_sha256": self.runtime_lock_payload["python"][
                    "executable_sha256"
                ],
                "python_executable_bytes": self.runtime_lock_payload["python"][
                    "executable_bytes"
                ],
                "platform": "Linux-fixture",
                "hostname": "fixture-host",
                "torch": "2.10.0+cu128",
                "torch_cuda": "12.8",
                "cudnn": 91_002,
                "nvidia_driver": "fixture-driver",
                "cuda_visible_devices": GPU_UUID,
                "run_id": self.plan["run_id"],
                "gpu_uuid": GPU_UUID,
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
                "launcher_path": "/visible/remote_launcher.py",
                "launcher_sha256": self.plan["expected_input_artifacts"][
                    "remote_launcher.py"
                ],
                "ordered_sys_path": [
                    "/stdlib",
                    *[
                        f"/proc/self/fd/{200 + index}"
                        for index in range(len(self.plan["ordered_import_roots"]))
                    ],
                ],
                "ordered_import_roots": self.plan["ordered_import_roots"],
                "conda_explicit_path": "/visible/conda-explicit.txt",
                "conda_explicit_sha256": self.plan["conda_explicit_sha256"],
                "visible_root": "/visible",
                "runtime_source_sha256": self.plan["runtime_source_fingerprint_sha256"],
                "test_source_sha256": self.plan["test_source_fingerprint_sha256"],
                "visible_files": dict(sorted(self.visible_files.items())),
                "python_runtime_lock_sha256": self.runtime_lock_artifact.artifact_sha256,
                "python_runtime_lock_id": self.runtime_lock_payload["runtime_lock_id"],
            }

        self.dependencies = command._TorchDependencies(
            configure_deterministic_cuda=mock.Mock(return_value=self.attestation),
            environment_manifest=mock.Mock(side_effect=environment_manifest),
            run_exact_preflight=self.run,
            release_held_gpu_lock=self.release,
            preflight_failure_type=_FixturePreflightFailure,
            imported_torch=object(),
        )

    def patches(self) -> ExitStack:
        stack = ExitStack()
        stack.enter_context(
            mock.patch.object(
                command,
                "read_launch_plan_artifact",
                return_value=self.launch_artifact,
            )
        )
        stack.enter_context(
            mock.patch.object(
                command,
                "read_python_runtime_lock_artifact",
                return_value=self.runtime_lock_artifact,
            )
        )
        stack.enter_context(
            mock.patch.object(
                command,
                "validate_live_python_runtime_lock",
                return_value=dict(self.runtime_lock_payload),
            )
        )
        stack.enter_context(
            mock.patch.object(command, "validate_imported_torch_runtime")
        )
        stack.enter_context(
            mock.patch.object(command, "_runtime_lock_live_kwargs", return_value={})
        )
        stack.enter_context(
            mock.patch.object(
                command,
                "validate_python_runtime_lock_payload",
                return_value=dict(self.runtime_lock_payload),
            )
        )
        stack.enter_context(
            mock.patch.object(
                command,
                "canonical_python_runtime_lock_bytes",
                return_value=self.runtime_lock_snapshot,
            )
        )
        stack.enter_context(
            mock.patch.object(
                command,
                "_visible_input_snapshots",
                return_value=(
                    self.visible_root,
                    self.visible_files,
                    self.visible_snapshots,
                ),
            )
        )
        stack.enter_context(
            mock.patch.object(
                command,
                "_validate_path_layout",
                return_value=(
                    self.run_root,
                    self.scratch,
                    self.output,
                    self.launcher,
                ),
            )
        )
        stack.enter_context(
            mock.patch.object(
                command,
                "_read_committed_regular",
                return_value=self.launcher_bytes,
            )
        )
        stack.enter_context(mock.patch.object(command, "verify_source_snapshot_zip"))
        stack.enter_context(
            mock.patch.object(command, "verify_test_source_snapshot_zip")
        )
        stack.enter_context(
            mock.patch.object(
                command,
                "runtime_source_fingerprint",
                return_value=self.plan["runtime_source_fingerprint_sha256"],
            )
        )
        stack.enter_context(
            mock.patch.object(
                command,
                "_validation_commitment_from_plan",
                return_value=self.validation_commitment,
            )
        )
        stack.enter_context(
            mock.patch.object(
                command,
                "read_lock_handshake_artifact",
                return_value=self.lock_artifact,
            )
        )
        stack.enter_context(
            mock.patch.object(
                command, "_load_torch_dependencies", return_value=self.dependencies
            )
        )
        stack.enter_context(
            mock.patch.object(
                command,
                "_bind_environment_to_plan",
                side_effect=lambda environment, **_: dict(environment),
            )
        )
        stack.enter_context(
            mock.patch.object(
                command,
                "validate_preflight_artifact",
                return_value=dict(self.report),
            )
        )
        stack.enter_context(
            mock.patch.dict(
                os.environ,
                {
                    "AFTS_M04A_RUN_ID": str(self.plan["run_id"]),
                    "AFTS_M04A_GPU_UUID": GPU_UUID,
                    "CUDA_VISIBLE_DEVICES": GPU_UUID,
                    "AFTS_EVIDENCE_LAUNCHER": str(self.launcher),
                },
            )
        )
        return stack

    def invoke(self) -> dict[str, object]:
        return command.run_preflight_command(
            launch_plan_path=self.root / "launch-plan.json",
            launch_plan_sha256=self.launch_artifact.artifact_sha256,
            python_runtime_lock_path=self.visible_root / "python-runtime-lock.json",
            python_runtime_lock_sha256=self.runtime_lock_artifact.artifact_sha256,
            validation_manifest_dir=self.root / "validation",
            lock_handshake_path=self.root / "lock.json",
            lock_handshake_sha256=self.lock_artifact.artifact_sha256,
            gpu_uuid=GPU_UUID,
            visible_root=self.visible_root,
            launcher_path=self.launcher,
            cost_probe_scratch_dir=self.scratch,
            output_dir=self.output,
        )


class M04aPreflightCommandTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "posix", "FIFO race is POSIX-specific")
    def test_committed_reader_rejects_fifo_without_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fifo = Path(directory) / "swapped-input"
            os.mkfifo(fifo)
            with self.assertRaisesRegex(ValueError, "regular file"):
                command._read_committed_regular(
                    fifo,
                    expected_sha256="0" * 64,
                    label="FIFO race fixture",
                )

    def test_import_and_parser_remain_torch_lazy(self) -> None:
        project = Path(__file__).resolve().parents[1]
        source_root = str(project / "src")
        script = f"""
import sys
sys.path.append({source_root!r})
import afts_arc.cli as cli
import afts_arc.m04a_preflight_command
assert 'torch' not in sys.modules
assert 'afts_arc.m04a_preflight' not in sys.modules
assert 'afts_arc.m04a_torch_runtime' not in sys.modules
args = cli.build_parser().parse_args([
    'preflight-m04a', '--launch-plan', 'plan.json',
    '--launch-plan-sha256', '0'*64,
    '--python-runtime-lock', 'python-runtime-lock.json',
    '--python-runtime-lock-sha256', '2'*64,
    '--validation-manifest-dir', 'validation',
    '--lock-handshake-artifact', 'lock.json',
    '--lock-handshake-artifact-sha256', '1'*64,
    '--gpu-uuid', 'GPU-00000000-0000-0000-0000-000000000001',
    '--visible-root', 'input', '--launcher-path', 'launcher.py',
    '--cost-probe-scratch-dir', 'scratch', '--output-dir', 'output'])
assert args.command == 'preflight-m04a'
assert 'torch' not in sys.modules
"""
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(project / "src")
        completed = subprocess.run(
            [sys.executable, "-I", "-S", "-c", script],
            cwd=project,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_visible_root_is_exact_plan_17_plus_fixed_conda(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "visible"
            root.mkdir()
            expected: dict[str, str] = {}
            expected_bytes: dict[str, bytes] = {}
            for name in EXPECTED_INPUT_ARTIFACT_PATHS:
                content = f"fixture:{name}".encode("ascii")
                (root / name).write_bytes(content)
                expected[name] = hashlib.sha256(content).hexdigest()
                expected_bytes[name] = content
            conda = b"conda-explicit-fixture\n"
            (root / command.CONDA_EXPLICIT_BASENAME).write_bytes(conda)
            resolved, inventory, snapshots = command._visible_input_snapshots(
                root,
                expected_input_artifacts=expected,
                conda_explicit_sha256=hashlib.sha256(conda).hexdigest(),
            )
            self.assertTrue(os.path.samefile(resolved, root))
            self.assertEqual(
                set(inventory), {*expected, command.CONDA_EXPLICIT_BASENAME}
            )
            self.assertEqual(
                {name: snapshots[name] for name in expected}, expected_bytes
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "visible"
            root.mkdir()
            expected = {}
            for name in EXPECTED_INPUT_ARTIFACT_PATHS:
                content = name.encode("ascii")
                (root / name).write_bytes(content)
                expected[name] = hashlib.sha256(content).hexdigest()
            conda = b"conda"
            (root / command.CONDA_EXPLICIT_BASENAME).write_bytes(conda)
            (root / "ambient-extra.txt").write_bytes(b"extra")
            with self.assertRaisesRegex(ValueError, "closed world"):
                command._visible_input_snapshots(
                    root,
                    expected_input_artifacts=expected,
                    conda_explicit_sha256=hashlib.sha256(conda).hexdigest(),
                )

    def test_environment_binding_uses_retained_plan_not_ambient_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _CommandFixture(Path(directory))
            environment = fixture.dependencies.environment_manifest(
                attestation=fixture.attestation
            )
            environment.update(
                {
                    "launcher_path": str(fixture.launcher.resolve()),
                    "conda_explicit_path": str(
                        (
                            fixture.visible_root / command.CONDA_EXPLICIT_BASENAME
                        ).resolve()
                    ),
                    "visible_root": str(fixture.visible_root.resolve()),
                }
            )
            bound = command._bind_environment_to_plan(
                environment,
                plan=fixture.plan,
                gpu_uuid=GPU_UUID,
                visible_root=fixture.visible_root,
                visible_files=fixture.visible_files,
                launcher=fixture.launcher,
                python_runtime_lock=fixture.runtime_lock_artifact,
                runtime_lock_payload=fixture.runtime_lock_payload,
            )
            self.assertEqual(bound["visible_files"], fixture.visible_files)
            wrong_python = dict(environment)
            wrong_python["python_executable"] = "/opt/conda/bin/python"
            with self.assertRaisesRegex(ValueError, "retained launch"):
                command._bind_environment_to_plan(
                    wrong_python,
                    plan=fixture.plan,
                    gpu_uuid=GPU_UUID,
                    visible_root=fixture.visible_root,
                    visible_files=fixture.visible_files,
                    launcher=fixture.launcher,
                    python_runtime_lock=fixture.runtime_lock_artifact,
                    runtime_lock_payload=fixture.runtime_lock_payload,
                )
            forged = dict(environment)
            forged["visible_files"] = {"ambient.txt": _digest("ambient")}
            with self.assertRaisesRegex(ValueError, "retained launch"):
                command._bind_environment_to_plan(
                    forged,
                    plan=fixture.plan,
                    gpu_uuid=GPU_UUID,
                    visible_root=fixture.visible_root,
                    visible_files=fixture.visible_files,
                    launcher=fixture.launcher,
                    python_runtime_lock=fixture.runtime_lock_artifact,
                    runtime_lock_payload=fixture.runtime_lock_payload,
                )

    def test_output_and_scratch_are_disjoint_from_validation_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_root = root / "run"
            visible = root / "visible"
            validation = run_root / "validation-manifest"
            run_root.mkdir()
            visible.mkdir()
            validation.mkdir()
            launcher = visible / "remote_launcher.py"
            launcher.write_bytes(b"fixture")
            runtime_lock_path = visible / "python-runtime-lock.json"
            runtime_lock_path.write_bytes(b"{}")
            launch_plan = root / "launch_plan.json"
            lock = run_root / "lock_handshake_artifact.json"
            launch_plan.write_bytes(b"{}")
            lock.write_bytes(b"{}")
            plan = {"run_root": str(run_root)}
            for field, target in (
                ("cost-probe scratch", validation / "scratch"),
                ("output", validation / "gate"),
            ):
                kwargs = {
                    "cost_probe_scratch_dir": run_root / "scratch",
                    "output_dir": run_root / "gate",
                }
                kwargs[
                    "cost_probe_scratch_dir"
                    if field == "cost-probe scratch"
                    else "output_dir"
                ] = target
                with (
                    self.subTest(field=field),
                    self.assertRaisesRegex(
                        ValueError, "disjoint from the validation bundle"
                    ),
                ):
                    command._validate_path_layout(
                        plan=plan,
                        visible_root=visible,
                        launch_plan_path=launch_plan,
                        python_runtime_lock_path=runtime_lock_path,
                        lock_handshake_path=lock,
                        validation_manifest_dir=validation,
                        launcher_path=launcher,
                        **kwargs,
                    )
            (run_root / command.PREFLIGHT_FAILURE_DIRECTORY).mkdir()
            with self.assertRaisesRegex(FileExistsError, "failure output"):
                command._validate_path_layout(
                    plan=plan,
                    visible_root=visible,
                    launch_plan_path=launch_plan,
                    python_runtime_lock_path=runtime_lock_path,
                    lock_handshake_path=lock,
                    validation_manifest_dir=validation,
                    launcher_path=launcher,
                    cost_probe_scratch_dir=run_root / "scratch",
                    output_dir=run_root / "gate",
                )

    def test_success_publishes_nontraining_gate_then_releases_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _CommandFixture(Path(directory))

            def release_after_publish(held) -> None:
                self.assertIs(held, fixture.result.held_lock_handshake)
                gate = command.read_preflight_gate_bundle(fixture.output)
                self.assertEqual(
                    gate["disposition"], command.PREFLIGHT_GATE_ARTIFACT_KIND
                )

            fixture.release.side_effect = release_after_publish
            with fixture.patches():
                summary = fixture.invoke()

            self.assertEqual(
                summary["disposition"],
                "diagnostic_preflight_gate_not_training_campaign",
            )
            self.assertIs(summary["training_evidence_eligible"], False)
            fixture.release.assert_called_once_with(fixture.result.held_lock_handshake)
            fixture.run.assert_called_once()
            run_kwargs = fixture.run.call_args.kwargs
            self.assertIs(run_kwargs["launch_plan_artifact"], fixture.launch_artifact)
            self.assertIs(
                run_kwargs["validation_manifest_commitment"],
                fixture.validation_commitment,
            )
            environment_kwargs = (
                fixture.dependencies.environment_manifest.call_args.kwargs
            )
            self.assertEqual(
                environment_kwargs["expected_visible_files"], fixture.visible_files
            )

            with (
                mock.patch.object(
                    command,
                    "validate_preflight_artifact",
                    return_value=dict(fixture.report),
                ),
                mock.patch.object(
                    command,
                    "validate_python_runtime_lock_payload",
                    return_value=dict(fixture.runtime_lock_payload),
                ),
                mock.patch.object(
                    command,
                    "canonical_python_runtime_lock_bytes",
                    return_value=fixture.runtime_lock_snapshot,
                ),
            ):
                gate = command.read_preflight_gate_bundle(fixture.output)
            self.assertIs(gate["training_evidence_eligible"], False)
            outer = json.loads(
                (fixture.output / "artifact_manifest.json").read_text("utf-8")
            )
            self.assertEqual(
                outer["artifact_kind"], command.PREFLIGHT_GATE_ARTIFACT_KIND
            )
            self.assertEqual(
                (fixture.output / "launch_plan.json").read_bytes(),
                fixture.launch_artifact.snapshot,
            )
            self.assertEqual(
                (fixture.output / "lock_handshake_artifact.json").read_bytes(),
                fixture.lock_artifact.snapshot,
            )
            with self.assertRaisesRegex(ValueError, "expected training artifact"):
                read_m04a_artifact_bundle(fixture.output, expected_kind="training")

    def test_reader_rejects_shallow_fake_preflight_pass_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _CommandFixture(Path(directory))
            with fixture.patches():
                fixture.invoke()
            with (
                mock.patch.object(
                    command,
                    "validate_python_runtime_lock_payload",
                    return_value=dict(fixture.runtime_lock_payload),
                ),
                mock.patch.object(
                    command,
                    "canonical_python_runtime_lock_bytes",
                    return_value=fixture.runtime_lock_snapshot,
                ),
                self.assertRaisesRegex(ValueError, "preflight report fields"),
            ):
                command.read_preflight_gate_bundle(fixture.output)

    def test_preflight_failure_does_not_publish_or_double_release(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _CommandFixture(
                Path(directory), run_failure=RuntimeError("CUDA preflight failed")
            )
            with (
                fixture.patches(),
                self.assertRaisesRegex(RuntimeError, "CUDA preflight failed"),
            ):
                fixture.invoke()
            self.assertFalse(fixture.output.exists())
            fixture.release.assert_not_called()

    def test_terminal_failure_publishes_closed_world_then_reraises(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _CommandFixture(Path(directory))
            failure = _partial_preflight_failure()
            fixture.run.side_effect = failure
            failure_dir = fixture.run_root / command.PREFLIGHT_FAILURE_DIRECTORY
            with fixture.patches():
                with self.assertRaises(_FixturePreflightFailure) as raised:
                    fixture.invoke()
                manifest = command.read_preflight_failure_bundle(failure_dir)

            self.assertIs(raised.exception, failure)
            self.assertFalse(fixture.output.exists())
            self.assertTrue(failure_dir.is_dir())
            self.assertEqual(
                raised.exception.published_failure_dir, str(failure_dir)
            )
            self.assertEqual(
                raised.exception.published_failure_id, manifest["failure_id"]
            )
            self.assertEqual(manifest["failure_code"], "OOM")
            self.assertEqual(manifest["evidence_completeness"], "partial")
            self.assertIs(manifest["training_evidence_eligible"], False)
            self.assertEqual(
                manifest["lock_release_policy"],
                "release_attempted_by_failed_exact_preflight_process_exit_backstop",
            )
            self.assertIs(manifest["diagnostic_checkpoint_selectable"], False)
            self.assertIs(manifest["same_run_retry_allowed"], False)
            self.assertIs(manifest["preflight_gate_created"], False)
            fixture.release.assert_not_called()

            outer = json.loads(
                (failure_dir / "artifact_manifest.json").read_text("utf-8")
            )
            self.assertEqual(
                outer["artifact_kind"], command.PREFLIGHT_FAILURE_ARTIFACT_KIND
            )
            self.assertNotIn("preflight.json", outer["artifacts"])
            self.assertNotIn(
                "preflight_diagnostic_checkpoint.pt", outer["artifacts"]
            )

    def test_budget_failure_publishes_complete_nonselectable_probe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _CommandFixture(Path(directory))
            failure = _budget_preflight_failure(fixture)
            diagnostic = failure.diagnostic_checkpoint_snapshot
            fixture.run.side_effect = failure
            failure_dir = fixture.run_root / command.PREFLIGHT_FAILURE_DIRECTORY

            with fixture.patches():
                with self.assertRaises(_FixturePreflightFailure):
                    fixture.invoke()
                manifest = command.read_preflight_failure_bundle(failure_dir)

            self.assertFalse(fixture.output.exists())
            self.assertEqual(manifest["failure_code"], "BUDGET_EXCEEDED")
            self.assertEqual(
                manifest["evidence_completeness"], "full_budget_probe"
            )
            self.assertEqual(
                (failure_dir / "preflight_diagnostic_checkpoint.pt").read_bytes(),
                diagnostic,
            )
            self.assertTrue(
                (failure_dir / "preflight_training_cost_ledger.jsonl").is_file()
            )
            self.assertTrue(
                (failure_dir / "preflight_overhead_cost_report.json").is_file()
            )
            persisted_report = json.loads(
                (failure_dir / "preflight_failure.json").read_text("utf-8")
            )
            self.assertEqual(persisted_report["training_checkpoint_writes"], 0)
            self.assertEqual(
                persisted_report["diagnostic_checkpoint_writes"], 1
            )
            self.assertIs(manifest["diagnostic_checkpoint_selectable"], False)
            fixture.release.assert_not_called()

    def test_pure_failure_replay_rejects_resealed_inference_and_projection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _CommandFixture(Path(directory))
            failure = _budget_preflight_failure(fixture)

            def validate(report, inference) -> None:
                command._validate_failure_report_payload(
                    report,
                    list(failure.ledger_rows),
                    run_id=fixture.plan["run_id"],
                    gpu_uuid=GPU_UUID,
                    validation_manifest_commitment=fixture.plan[
                        "validation_manifest_commitment"
                    ],
                    lock_handshake_artifact_sha256=(
                        fixture.lock_artifact.artifact_sha256
                    ),
                    lock_handshake_payload=fixture.lock_artifact.payload,
                    config_sha256=fixture.plan["training_config_sha256"],
                    runtime_source_sha256=fixture.plan[
                        "runtime_source_fingerprint_sha256"
                    ],
                    test_source_sha256=fixture.plan[
                        "test_source_fingerprint_sha256"
                    ],
                    overhead_report=failure.overhead_cost_probe_report,
                    training_summary=failure.training_summary,
                    inference_summary=inference,
                    diagnostic_checkpoint=(
                        failure.diagnostic_checkpoint_snapshot
                    ),
                )

            forged_fixture_report = copy.deepcopy(failure.report)
            forged_fixture_report["fixture_id"] = _digest("attacker-fixture")
            report_semantic = dict(forged_fixture_report)
            report_semantic.pop("report_id")
            forged_fixture_report["report_id"] = canonical_sha256(
                report_semantic
            )
            with self.assertRaisesRegex(ValueError, "frozen fixture"):
                validate(forged_fixture_report, failure.inference_summary)

            forged_inference = copy.deepcopy(failure.inference_summary)
            forged_inference["lane_count"] = 7
            inference_semantic = dict(forged_inference)
            inference_semantic.pop("inference_id")
            forged_inference["inference_id"] = canonical_sha256(
                inference_semantic
            )
            forged_report = copy.deepcopy(failure.report)
            forged_report["inference_summary"] = forged_inference
            report_semantic = dict(forged_report)
            report_semantic.pop("report_id")
            forged_report["report_id"] = canonical_sha256(report_semantic)
            with self.assertRaisesRegex(ValueError, "inference semantic"):
                validate(forged_report, forged_inference)

            for field, value in (
                ("schema", "attacker-controlled-projection/v9"),
                ("projection_method", "attacker-controlled-method"),
            ):
                with self.subTest(field=field):
                    forged_report = copy.deepcopy(failure.report)
                    projection = forged_report["budget_projection"]
                    projection[field] = value
                    projection_semantic = dict(projection)
                    projection_semantic.pop("projection_id")
                    projection["projection_id"] = canonical_sha256(
                        projection_semantic
                    )
                    report_semantic = dict(forged_report)
                    report_semantic.pop("report_id")
                    forged_report["report_id"] = canonical_sha256(
                        report_semantic
                    )
                    with self.assertRaisesRegex(
                        ValueError, "projection version/method"
                    ):
                        validate(forged_report, failure.inference_summary)

    def test_failure_reader_rejects_manifest_retry_policy_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _CommandFixture(Path(directory))
            fixture.run.side_effect = _partial_preflight_failure()
            failure_dir = fixture.run_root / command.PREFLIGHT_FAILURE_DIRECTORY
            with fixture.patches():
                with self.assertRaises(_FixturePreflightFailure):
                    fixture.invoke()
                manifest_path = failure_dir / command.PREFLIGHT_FAILURE_MANIFEST
                manifest = json.loads(manifest_path.read_text("utf-8"))
                manifest["same_run_retry_allowed"] = True
                manifest_path.write_bytes(serialize_json(manifest))
                with self.assertRaisesRegex(
                    ValueError, "SHA-256|hash|differs|metadata mismatch"
                ):
                    command.read_preflight_failure_bundle(failure_dir)

    def test_failure_reader_rejects_resealed_absolute_lock_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _CommandFixture(Path(directory))
            fixture.run.side_effect = _partial_preflight_failure()
            failure_dir = fixture.run_root / command.PREFLIGHT_FAILURE_DIRECTORY
            with fixture.patches():
                with self.assertRaises(_FixturePreflightFailure):
                    fixture.invoke()

                lock_path = failure_dir / "lock_handshake_artifact.json"
                lock = json.loads(lock_path.read_text("utf-8"))
                lock["lock_path"] = "/attacker/absolute.lock"
                lock_semantic = dict(lock)
                lock_semantic.pop("handshake_id")
                lock["handshake_id"] = canonical_sha256(lock_semantic)
                lock_bytes = serialize_json(lock)
                lock_path.write_bytes(lock_bytes)

                manifest_path = failure_dir / command.PREFLIGHT_FAILURE_MANIFEST
                manifest = json.loads(manifest_path.read_text("utf-8"))
                lock_sha = hashlib.sha256(lock_bytes).hexdigest()
                manifest["lock_handshake_artifact_sha256"] = lock_sha
                manifest["lock_handshake_id"] = lock["handshake_id"]
                for row in manifest["files"]:
                    if row["path"] == "lock_handshake_artifact.json":
                        row["sha256"] = lock_sha
                        row["bytes"] = len(lock_bytes)
                manifest_semantic = dict(manifest)
                manifest_semantic.pop("failure_id")
                manifest["failure_id"] = canonical_sha256(manifest_semantic)
                manifest_bytes = serialize_json(manifest)
                manifest_path.write_bytes(manifest_bytes)

                outer_path = failure_dir / "artifact_manifest.json"
                outer = json.loads(outer_path.read_text("utf-8"))
                outer["bundle_id"] = manifest["failure_id"]
                outer["artifacts"]["lock_handshake_artifact.json"].update(
                    {"sha256": lock_sha, "bytes": len(lock_bytes)}
                )
                outer["artifacts"][command.PREFLIGHT_FAILURE_MANIFEST].update(
                    {
                        "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                        "bytes": len(manifest_bytes),
                    }
                )
                outer_path.write_bytes(serialize_json(outer))
                with self.assertRaisesRegex(ValueError, "parent closure"):
                    command.read_preflight_failure_bundle(failure_dir)

    def test_pure_commitment_failure_happens_before_torch_loader(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _CommandFixture(Path(directory))
            with (
                fixture.patches(),
                mock.patch.object(
                    command,
                    "_visible_input_snapshots",
                    side_effect=ValueError("visible commitment failed"),
                ),
                mock.patch.object(command, "_load_torch_dependencies") as loader,
                self.assertRaisesRegex(ValueError, "visible commitment failed"),
            ):
                fixture.invoke()
            loader.assert_not_called()
            self.assertFalse(fixture.output.exists())

    def test_output_is_exclusive_and_cli_forwards_every_argument(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _CommandFixture(Path(directory))
            fixture.output.mkdir()
            with (
                fixture.patches(),
                self.assertRaisesRegex(FileExistsError, "overwrite"),
            ):
                fixture.invoke()
            fixture.run.assert_called_once()
            fixture.release.assert_called_once()

        with tempfile.TemporaryDirectory() as directory:
            fixture = _CommandFixture(Path(directory))
            fixture.output.mkdir()
            fixture.release.side_effect = RuntimeError("release failed")
            with (
                fixture.patches(),
                self.assertRaisesRegex(
                    RuntimeError,
                    "preflight failed and held GPU-lock release also failed",
                ) as raised,
            ):
                fixture.invoke()
            self.assertIsNotNone(raised.exception.__cause__)
            self.assertIn("overwrite", str(raised.exception.__cause__))

        expected = {"status": "fixture"}
        argv = [
            "preflight-m04a",
            "--launch-plan",
            "plan.json",
            "--launch-plan-sha256",
            "0" * 64,
            "--python-runtime-lock",
            "python-runtime-lock.json",
            "--python-runtime-lock-sha256",
            "2" * 64,
            "--validation-manifest-dir",
            "validation",
            "--lock-handshake-artifact",
            "lock.json",
            "--lock-handshake-artifact-sha256",
            "1" * 64,
            "--gpu-uuid",
            GPU_UUID,
            "--visible-root",
            "visible",
            "--launcher-path",
            "launcher.py",
            "--cost-probe-scratch-dir",
            "scratch",
            "--output-dir",
            "output",
        ]
        stdout = StringIO()
        with (
            mock.patch.object(
                command, "run_preflight_command", return_value=expected
            ) as invoked,
            redirect_stdout(stdout),
        ):
            self.assertEqual(cli._main_in_process(argv), 0)
        self.assertEqual(json.loads(stdout.getvalue()), expected)
        invoked.assert_called_once_with(
            launch_plan_path="plan.json",
            launch_plan_sha256="0" * 64,
            python_runtime_lock_path="python-runtime-lock.json",
            python_runtime_lock_sha256="2" * 64,
            validation_manifest_dir="validation",
            lock_handshake_path="lock.json",
            lock_handshake_sha256="1" * 64,
            gpu_uuid=GPU_UUID,
            visible_root="visible",
            launcher_path="launcher.py",
            cost_probe_scratch_dir="scratch",
            output_dir="output",
        )


if __name__ == "__main__":
    unittest.main()
