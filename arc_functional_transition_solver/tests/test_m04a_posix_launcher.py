from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import stat
import sys
import types
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from afts_arc import cli
from afts_arc import m04a_campaign_command as campaign_command
from afts_arc import m04a_preflight_command as preflight_command
from afts_arc.m04a_contract import canonical_sha256
from afts_arc.m04a_launch_plan import build_launch_plan, canonical_launch_plan_bytes
from afts_arc.m04a_lock_handshake import validate_lock_handshake_payload
from afts_arc.m04a_train_contract import training_config_sha256
from afts_arc.manifest import serialize_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_PATH = PROJECT_ROOT / "scripts" / "afts_arc_m04a.py"
SPEC = importlib.util.spec_from_file_location(
    "m04a_posix_launcher_under_test", LAUNCHER_PATH
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load M04a POSIX launcher")
launcher = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = launcher
SPEC.loader.exec_module(launcher)

GPU_A = "GPU-aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
GPU_B = "GPU-11111111-2222-4333-8444-555555555555"
REMOTE_ROOT = "/srv/afts"
RUN_ID = "m04a-test-run"


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _arguments() -> object:
    return launcher._parse_launcher_arguments(
        [
            "remote_launcher.py",
            "preflight-m04a",
            "--projected-artifact-bytes",
            str(15 * (1 << 30)),
            "--minimum-free-inodes",
            "4096",
        ]
    )


def _plan() -> dict[str, object]:
    return {
        "run_id": RUN_ID,
        "run_root": f"{REMOTE_ROOT}/runs/{RUN_ID}",
        "remote_project_root": REMOTE_ROOT,
        "attempt_nonce": _digest("attempt-nonce"),
        "expected_input_artifacts": {
            "remote_launcher.py": _digest("launcher"),
            "reviewed_runtime_source.zip": _digest("runtime-zip"),
            "python-runtime-lock.json": _digest("runtime-lock"),
            "validation_episode_outer_manifest.json": _digest("validation-outer"),
            "validation_episode_manifest.jsonl": _digest("validation-jsonl"),
        },
        "runtime_source_fingerprint_sha256": _digest("runtime-fingerprint"),
        "test_source_fingerprint_sha256": _digest("test-fingerprint"),
        "conda_explicit_sha256": _digest("conda-explicit"),
        "ordered_import_roots": [
            f"{REMOTE_ROOT}/src",
            "/purelib",
            "/platlib",
        ],
    }


def _full_launch_plan_bytes() -> tuple[dict[str, object], bytes]:
    artifacts = {name: _digest(name) for name in launcher._EXPECTED_INPUT_PATHS}
    commitment_semantic: dict[str, object] = {
        "schema": "afts-m04a-validation-manifest-commitment/v0.1",
        "outer_artifact_manifest_sha256": artifacts[
            "validation_episode_outer_manifest.json"
        ],
        "jsonl_sha256": artifacts["validation_episode_manifest.jsonl"],
        "summary_id": _digest("validation-summary"),
        "row_count": 2,
    }
    commitment = {
        **commitment_semantic,
        "commitment_id": canonical_sha256(commitment_semantic),
    }
    plan = build_launch_plan(
        attempt_nonce=_digest("attempt-nonce"),
        run_id=RUN_ID,
        remote_project_root=REMOTE_ROOT,
        run_root=f"{REMOTE_ROOT}/runs/{RUN_ID}",
        expected_input_artifacts=artifacts,
        runtime_source_fingerprint_sha256=_digest("runtime-fingerprint"),
        test_source_fingerprint_sha256=_digest("test-fingerprint"),
        training_config_sha256=training_config_sha256(),
        validation_manifest_commitment=commitment,
        conda_explicit_sha256=_digest("conda-explicit"),
        ordered_import_roots=[f"{REMOTE_ROOT}/src", "/site-packages"],
    )
    return plan, canonical_launch_plan_bytes(plan)


def _validation_snapshots() -> tuple[bytes, bytes, bytes]:
    jsonl = b'{"row":1}\n{"row":2}\n'
    summary = {
        "summary_id": _digest("validation-summary"),
        "jsonl_sha256": hashlib.sha256(jsonl).hexdigest(),
        "jsonl_bytes": len(jsonl),
        "row_count": 2,
    }
    summary_bytes = serialize_json(summary)
    outer = {
        "schema_version": 1,
        "bundle_status": "complete",
        "run_id": summary["summary_id"],
        "artifacts": {
            "validation_episode_manifest.jsonl": {
                "sha256": hashlib.sha256(jsonl).hexdigest(),
                "bytes": len(jsonl),
                "rows": 2,
            },
            "validation_episode_manifest_summary.json": {
                "sha256": hashlib.sha256(summary_bytes).hexdigest(),
                "bytes": len(summary_bytes),
                "rows": None,
            },
        },
    }
    return serialize_json(outer), jsonl, summary_bytes


class M04aLauncherArgumentTests(unittest.TestCase):
    def test_reviewed_capacity_arguments_are_strict_and_not_forwarded(self) -> None:
        parsed = _arguments()
        self.assertEqual(
            parsed.child_argv,
            ("preflight-m04a",),
        )
        self.assertEqual(parsed.projected_artifact_bytes, 15 * (1 << 30))
        self.assertEqual(parsed.minimum_free_inodes, 4096)

        parsed_train = launcher._parse_launcher_arguments(
            [
                "remote_launcher.py",
                "train-m04a",
                "--projected-artifact-bytes",
                str(15 * (1 << 30)),
                "--minimum-free-inodes",
                "4096",
            ]
        )
        self.assertEqual(parsed_train.child_argv, ("train-m04a",))

        bad_cases = (
            ["remote_launcher.py", "preflight-m04a"],
            [
                "remote_launcher.py",
                "preflight-m04a",
                "--projected-artifact-bytes",
                "01",
                "--minimum-free-inodes",
                "1",
            ],
            [
                "remote_launcher.py",
                "preflight-m04a",
                "--projected-artifact-bytes",
                "0",
                "--projected-artifact-bytes",
                "0",
                "--minimum-free-inodes",
                "1",
            ],
            [
                "remote_launcher.py",
                "preflight-m04a",
                "--projected-artifact-bytes",
                "0",
                "--minimum-free-inodes",
                "0",
            ],
        )
        for argv in bad_cases:
            with self.subTest(argv=argv), self.assertRaises(ValueError):
                launcher._parse_launcher_arguments(argv)

        with self.assertRaisesRegex(ValueError, "unreviewed child argument"):
            launcher._parse_launcher_arguments(
                [
                    "remote_launcher.py",
                    "preflight-m04a",
                    "--projected-artifact-bytes",
                    "0",
                    "--minimum-free-inodes",
                    "1",
                    "--gpu-uuid",
                    GPU_A,
                ]
            )

        with self.assertRaisesRegex(SystemExit, "reviewed M04a neural command"):
            launcher._parse_launcher_arguments(["remote_launcher.py", "audit-data"])

    def test_external_plan_path_and_sha_derive_one_exact_remote_root(self) -> None:
        path, digest, root = launcher._external_plan_coordinates(
            {
                "AFTS_M04A_LAUNCH_PLAN_JSON": f"{REMOTE_ROOT}/launch_plan.json",
                "AFTS_M04A_LAUNCH_PLAN_SHA256": "a" * 64,
            }
        )
        self.assertEqual(path.as_posix(), f"{REMOTE_ROOT}/launch_plan.json")
        self.assertEqual(digest, "a" * 64)
        self.assertEqual(root, REMOTE_ROOT)
        for path_value in (
            "relative/launch_plan.json",
            f"{REMOTE_ROOT}/input/launch_plan.json",
            f"{REMOTE_ROOT}/input/not-the-plan.json",
        ):
            with self.subTest(path=path_value), self.assertRaises(ValueError):
                launcher._external_plan_coordinates(
                    {
                        "AFTS_M04A_LAUNCH_PLAN_JSON": path_value,
                        "AFTS_M04A_LAUNCH_PLAN_SHA256": "a" * 64,
                    }
                )

    def test_embedded_plan_reader_matches_canonical_schema_without_importing_source(
        self,
    ) -> None:
        plan, content = _full_launch_plan_bytes()
        digest = hashlib.sha256(content).hexdigest()
        with mock.patch.object(
            launcher, "_read_bounded_regular_posix", return_value=content
        ):
            self.assertEqual(
                launcher._load_committed_launch_plan(
                    Path(f"{REMOTE_ROOT}/launch_plan.json"),
                    expected_sha256=digest,
                    derived_remote_root=REMOTE_ROOT,
                ),
                plan,
            )
            with self.assertRaisesRegex(ValueError, "external SHA-256"):
                launcher._load_committed_launch_plan(
                    Path(f"{REMOTE_ROOT}/launch_plan.json"),
                    expected_sha256="0" * 64,
                    derived_remote_root=REMOTE_ROOT,
                )
        noncanonical = json.dumps(plan, sort_keys=True).encode("utf-8")
        with (
            mock.patch.object(
                launcher, "_read_bounded_regular_posix", return_value=noncanonical
            ),
            self.assertRaisesRegex(ValueError, "canonically serialized"),
        ):
            launcher._load_committed_launch_plan(
                Path(f"{REMOTE_ROOT}/launch_plan.json"),
                expected_sha256=hashlib.sha256(noncanonical).hexdigest(),
                derived_remote_root=REMOTE_ROOT,
            )

    def test_launch_plan_inside_visible_input_is_rejected_as_an_extra_file(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValueError, "remote_project_root/launch_plan.json"):
            launcher._external_plan_coordinates(
                {
                    "AFTS_M04A_LAUNCH_PLAN_JSON": (
                        f"{REMOTE_ROOT}/input/launch_plan.json"
                    ),
                    "AFTS_M04A_LAUNCH_PLAN_SHA256": "a" * 64,
                }
            )

    def test_synthesized_preflight_argv_reaches_real_parser_and_handler(self) -> None:
        child_argv = launcher._neural_cli_argv(
            command="preflight-m04a",
            plan_path=f"{REMOTE_ROOT}/launch_plan.json",
            plan_sha256="a" * 64,
            validation_manifest_dir=f"{REMOTE_ROOT}/runs/{RUN_ID}/validation-manifest",
            handshake_path=f"{REMOTE_ROOT}/runs/{RUN_ID}/lock_handshake_artifact.json",
            handshake_sha256="b" * 64,
            gpu_uuid=GPU_A,
            remote_root=REMOTE_ROOT,
            run_root=f"{REMOTE_ROOT}/runs/{RUN_ID}",
            python_runtime_lock_sha256="c" * 64,
        )
        parsed = cli.build_parser().parse_args(list(child_argv))
        self.assertIs(parsed.handler, cli._cmd_preflight_m04a)
        with (
            mock.patch.object(
                preflight_command,
                "run_preflight_command",
                return_value={"status": "PASS"},
            ) as run,
            redirect_stdout(StringIO()),
        ):
            self.assertEqual(parsed.handler(parsed), 0)
        run.assert_called_once_with(
            launch_plan_path=f"{REMOTE_ROOT}/launch_plan.json",
            launch_plan_sha256="a" * 64,
            python_runtime_lock_path=f"{REMOTE_ROOT}/input/python-runtime-lock.json",
            python_runtime_lock_sha256="c" * 64,
            validation_manifest_dir=f"{REMOTE_ROOT}/runs/{RUN_ID}/validation-manifest",
            lock_handshake_path=f"{REMOTE_ROOT}/runs/{RUN_ID}/lock_handshake_artifact.json",
            lock_handshake_sha256="b" * 64,
            gpu_uuid=GPU_A,
            visible_root=f"{REMOTE_ROOT}/input",
            launcher_path=f"{REMOTE_ROOT}/input/remote_launcher.py",
            cost_probe_scratch_dir=f"{REMOTE_ROOT}/runs/{RUN_ID}/preflight-cost-probe",
            output_dir=f"{REMOTE_ROOT}/runs/{RUN_ID}/preflight-gate",
        )

    def test_synthesized_training_argv_reaches_real_parser_and_handler(self) -> None:
        child_argv = launcher._neural_cli_argv(
            command="train-m04a",
            plan_path=f"{REMOTE_ROOT}/launch_plan.json",
            plan_sha256="a" * 64,
            validation_manifest_dir=f"{REMOTE_ROOT}/runs/{RUN_ID}/validation-manifest",
            handshake_path=f"{REMOTE_ROOT}/runs/{RUN_ID}/lock_handshake_artifact.json",
            handshake_sha256="b" * 64,
            gpu_uuid=GPU_A,
            remote_root=REMOTE_ROOT,
            run_root=f"{REMOTE_ROOT}/runs/{RUN_ID}",
            python_runtime_lock_sha256="c" * 64,
        )
        parsed = cli.build_parser().parse_args(list(child_argv))
        self.assertIs(parsed.handler, cli._cmd_train_m04a)

        def run_campaign(**kwargs):
            kwargs["progress_callback"]({"event": "progress", "update": 100})
            return {"status": "PASS"}

        with (
            mock.patch.object(
                campaign_command,
                "run_training_campaign_command",
                side_effect=run_campaign,
            ) as run,
            redirect_stdout(StringIO()) as stdout,
        ):
            self.assertEqual(parsed.handler(parsed), 0)
        call = run.call_args.kwargs
        progress_callback = call.pop("progress_callback")
        self.assertTrue(callable(progress_callback))
        self.assertEqual(
            call,
            {
                "launch_plan_path": f"{REMOTE_ROOT}/launch_plan.json",
                "launch_plan_sha256": "a" * 64,
                "python_runtime_lock_path": f"{REMOTE_ROOT}/input/python-runtime-lock.json",
                "python_runtime_lock_sha256": "c" * 64,
                "validation_manifest_dir": (
                    f"{REMOTE_ROOT}/runs/{RUN_ID}/validation-manifest"
                ),
                "lock_handshake_path": (
                    f"{REMOTE_ROOT}/runs/{RUN_ID}/lock_handshake_artifact.json"
                ),
                "lock_handshake_sha256": "b" * 64,
                "gpu_uuid": GPU_A,
                "visible_root": f"{REMOTE_ROOT}/input",
                "launcher_path": f"{REMOTE_ROOT}/input/remote_launcher.py",
                "cost_probe_scratch_dir": (
                    f"{REMOTE_ROOT}/runs/{RUN_ID}/preflight-cost-probe"
                ),
                "production_data_root": f"{REMOTE_ROOT}/data",
                "working_dir": f"{REMOTE_ROOT}/runs/{RUN_ID}/campaign-working",
                "output_dir": f"{REMOTE_ROOT}/runs/{RUN_ID}/training-artifact",
            },
        )
        emitted = stdout.getvalue().splitlines()
        self.assertEqual(json.loads(emitted[0]), {"event": "progress", "update": 100})
        self.assertEqual(json.loads("\n".join(emitted[1:])), {"status": "PASS"})


class M04aLauncherGpuTests(unittest.TestCase):
    def test_gpu_snapshot_requires_stable_empty_processes_and_literal_idle_limits(
        self,
    ) -> None:
        replies = iter(
            (
                (b"", b""),
                (
                    f"{GPU_A}, 0, 256\n{GPU_B}, 0, 257\n".encode("ascii"),
                    b"",
                ),
                (b"", b""),
            )
        )
        calls: list[tuple[str, ...]] = []

        def runner(arguments):
            calls.append(tuple(arguments))
            return next(replies)

        rows = launcher._query_gpu_snapshot(runner)
        self.assertEqual(
            calls,
            [launcher._PROCESS_QUERY, launcher._GPU_QUERY, launcher._PROCESS_QUERY],
        )
        self.assertEqual(launcher._select_idle_gpu(rows), GPU_A)
        self.assertTrue(rows[0].idle)
        self.assertFalse(rows[1].idle)

        for utilization, memory, pids in ((1, 0, ()), (0, 257, ()), (0, 0, (7,))):
            row = launcher._GpuRow(GPU_A, utilization, memory, pids)
            with (
                self.subTest(row=row),
                self.assertRaisesRegex(RuntimeError, "NO_IDLE_GPU"),
            ):
                launcher._select_idle_gpu((row,))

    def test_gpu_snapshot_fails_closed_on_changing_processes_stderr_or_bad_rows(
        self,
    ) -> None:
        cases = (
            (
                (
                    (b"", b""),
                    (f"{GPU_A},0,0\n".encode(), b""),
                    (f"{GPU_A},9\n".encode(), b""),
                ),
                RuntimeError,
            ),
            (
                ((b"", b"warning"), (f"{GPU_A},0,0\n".encode(), b""), (b"", b"")),
                RuntimeError,
            ),
            (
                ((b"", b""), (b"not,a,gpu,row,extra\n", b""), (b"", b"")),
                ValueError,
            ),
        )
        for replies, error in cases:
            iterator = iter(replies)
            with self.subTest(replies=replies), self.assertRaises(error):
                launcher._query_gpu_snapshot(lambda _arguments: next(iterator))

    def test_lock_recheck_is_for_the_same_selected_uuid(self) -> None:
        rows = (
            launcher._GpuRow(GPU_A, 0, 0, ()),
            launcher._GpuRow(GPU_B, 0, 0, ()),
        )
        launcher._require_selected_gpu_idle(rows, gpu_uuid=GPU_B)
        with self.assertRaisesRegex(RuntimeError, "NO_LONGER_IDLE"):
            launcher._require_selected_gpu_idle(
                (launcher._GpuRow(GPU_B, 0, 0, (123,)),), gpu_uuid=GPU_B
            )


class M04aLauncherCapacityTests(unittest.TestCase):
    def test_capacity_closes_bytes_and_inode_thresholds(self) -> None:
        gib = 1 << 30
        file_system = SimpleNamespace(
            f_frsize=4096,
            f_bavail=(40 * gib) // 4096,
            f_favail=5000,
        )
        observed = launcher._capacity_observation(
            file_system,
            projected_artifact_bytes=15 * gib,
            minimum_free_inodes=4096,
        )
        self.assertEqual(observed.required_bytes, 35 * gib)
        self.assertEqual(observed.available_bytes, 40 * gib)
        self.assertEqual(observed.free_inodes, 5000)

        with self.assertRaisesRegex(RuntimeError, "FREE_BYTES"):
            launcher._capacity_observation(
                file_system,
                projected_artifact_bytes=21 * gib,
                minimum_free_inodes=4096,
            )
        with self.assertRaisesRegex(RuntimeError, "FREE_INODES"):
            launcher._capacity_observation(
                file_system,
                projected_artifact_bytes=0,
                minimum_free_inodes=5001,
            )
        for projected, inodes in ((True, 1), (0, True)):
            with (
                self.subTest(projected=projected, inodes=inodes),
                self.assertRaises(TypeError),
            ):
                launcher._capacity_observation(
                    file_system,
                    projected_artifact_bytes=projected,
                    minimum_free_inodes=inodes,
                )


class M04aLauncherValidationViewTests(unittest.TestCase):
    def test_validation_snapshots_close_outer_hash_bytes_rows_and_summary(self) -> None:
        outer, jsonl, summary = _validation_snapshots()
        launcher._validate_validation_view_snapshots(
            outer_bytes=outer, jsonl_bytes=jsonl, summary_bytes=summary
        )
        parsed = json.loads(outer)
        bad_cases: list[tuple[str, bytes, bytes, bytes]] = []
        bad_rows = json.loads(outer)
        bad_rows["artifacts"]["validation_episode_manifest.jsonl"]["rows"] = 3
        bad_cases.append(("rows", serialize_json(bad_rows), jsonl, summary))
        extra = json.loads(outer)
        extra["artifacts"]["extra.json"] = {
            "sha256": _digest("extra"),
            "bytes": 1,
            "rows": None,
        }
        bad_cases.append(("closed-world", serialize_json(extra), jsonl, summary))
        bad_summary = json.loads(summary)
        bad_summary["jsonl_bytes"] += 1
        bad_summary_bytes = serialize_json(bad_summary)
        parsed["artifacts"]["validation_episode_manifest_summary.json"]["sha256"] = (
            hashlib.sha256(bad_summary_bytes).hexdigest()
        )
        parsed["artifacts"]["validation_episode_manifest_summary.json"]["bytes"] = len(
            bad_summary_bytes
        )
        bad_cases.append(
            ("summary-closure", serialize_json(parsed), jsonl, bad_summary_bytes)
        )
        for name, outer_bytes, jsonl_bytes, summary_bytes in bad_cases:
            with self.subTest(name=name), self.assertRaises(ValueError):
                launcher._validate_validation_view_snapshots(
                    outer_bytes=outer_bytes,
                    jsonl_bytes=jsonl_bytes,
                    summary_bytes=summary_bytes,
                )

    def test_materialization_renames_verified_outer_and_publishes_manifest_last(
        self,
    ) -> None:
        outer, jsonl, summary = _validation_snapshots()
        snapshots = {
            "validation_episode_outer_manifest.json": outer,
            "validation_episode_manifest.jsonl": jsonl,
            "validation_episode_manifest_summary.json": summary,
        }
        expected = {
            name: hashlib.sha256(content).hexdigest()
            for name, content in snapshots.items()
        }
        writes: list[tuple[str, bytes]] = []

        def read(_directory_fd, *, name, expected_sha256, maximum_bytes):
            self.assertEqual(expected_sha256, expected[name])
            self.assertGreater(maximum_bytes, 0)
            return snapshots[name]

        def write(_directory_fd, *, name, content):
            writes.append((name, content))

        directory_info = SimpleNamespace(st_mode=stat.S_IFDIR)
        with (
            mock.patch.object(launcher, "_read_verified_regular_at", side_effect=read),
            mock.patch.object(launcher.os, "mkdir") as mkdir,
            mock.patch.object(launcher.os, "open", return_value=44),
            mock.patch.object(launcher.os, "fstat", return_value=directory_info),
            mock.patch.object(launcher, "_write_new_readonly_file", side_effect=write),
            mock.patch.object(
                launcher, "_verify_materialized_validation_view"
            ) as verified,
            mock.patch.object(launcher.os, "fsync"),
            mock.patch.object(launcher.os, "fchmod", create=True) as chmod,
            mock.patch.object(launcher.os, "close"),
        ):
            path = launcher._materialize_validation_view(
                input_fd=10,
                run_root_fd=20,
                run_root=f"{REMOTE_ROOT}/runs/{RUN_ID}",
                expected_artifacts=expected,
            )
        self.assertEqual(path, f"{REMOTE_ROOT}/runs/{RUN_ID}/validation-manifest")
        mkdir.assert_called_once_with("validation-manifest", mode=0o700, dir_fd=20)
        self.assertEqual(
            [name for name, _content in writes],
            [
                "validation_episode_manifest.jsonl",
                "validation_episode_manifest_summary.json",
                "artifact_manifest.json",
            ],
        )
        self.assertEqual(writes[-1][1], outer)
        verified.assert_called_once()
        chmod.assert_called_once_with(44, 0o500)

    def test_view_files_are_exclusive_create_and_readonly(self) -> None:
        calls: list[tuple[object, ...]] = []

        def opened(*args, **kwargs):
            calls.append((*args, kwargs))
            return 55

        with (
            mock.patch.object(launcher.os, "open", side_effect=opened),
            mock.patch.object(launcher, "_write_all") as write,
            mock.patch.object(launcher.os, "fsync"),
            mock.patch.object(launcher.os, "fchmod", create=True) as chmod,
            mock.patch.object(launcher.os, "close"),
        ):
            launcher._write_new_readonly_file(20, name="artifact.json", content=b"x")
        flags = calls[0][1]
        self.assertTrue(flags & launcher.os.O_EXCL)
        self.assertTrue(flags & launcher.os.O_CREAT)
        write.assert_called_once_with(55, b"x")
        chmod.assert_called_once_with(55, 0o400)


class M04aLauncherLockAndExecTests(unittest.TestCase):
    def test_project_children_are_opened_relative_to_bound_root_fd(self) -> None:
        directory = SimpleNamespace(st_mode=stat.S_IFDIR)
        with (
            mock.patch.object(
                launcher, "_open_existing_directory_nofollow", return_value=50
            ),
            mock.patch.object(
                launcher.os, "open", side_effect=(51, 52, 53, 54, 55)
            ) as opened,
            mock.patch.object(launcher.os, "fstat", return_value=directory),
        ):
            result = launcher._open_project_directories(REMOTE_ROOT)
        self.assertEqual(
            result,
            {
                "root": 50,
                "input": 51,
                "data": 52,
                "source": 53,
                "runs": 54,
                "locks": 55,
            },
        )
        self.assertEqual(
            [call.args[0] for call in opened.call_args_list],
            ["input", "data", "src", "runs", "locks"],
        )
        self.assertTrue(
            all(call.kwargs["dir_fd"] == 50 for call in opened.call_args_list)
        )

    def test_fixed_fd_duplication_fails_closed_on_occupied_race_and_mismatch(
        self,
    ) -> None:
        fake_fcntl = types.ModuleType("fcntl")
        fake_fcntl.F_DUPFD_CLOEXEC = 1030
        source = SimpleNamespace(st_mode=stat.S_IFDIR, st_dev=1, st_ino=2)
        target = SimpleNamespace(st_mode=stat.S_IFDIR, st_dev=1, st_ino=3)
        with (
            mock.patch.dict(sys.modules, {"fcntl": fake_fcntl}),
            mock.patch.object(launcher.os, "fstat", return_value=target),
            self.assertRaisesRegex(RuntimeError, "already occupied"),
        ):
            launcher._duplicate_directory_to_fixed_fd(10, target_fd=200)

        missing = OSError(launcher.errno.EBADF, "bad fd")
        fake_fcntl.fcntl = mock.Mock(return_value=201)
        with (
            mock.patch.dict(sys.modules, {"fcntl": fake_fcntl}),
            mock.patch.object(launcher.os, "fstat", side_effect=missing),
            mock.patch.object(launcher.os, "close") as closed,
            self.assertRaisesRegex(RuntimeError, "allocation raced"),
        ):
            launcher._duplicate_directory_to_fixed_fd(10, target_fd=200)
        closed.assert_called_once_with(201)

        fake_fcntl.fcntl = mock.Mock(return_value=200)
        with (
            mock.patch.dict(sys.modules, {"fcntl": fake_fcntl}),
            mock.patch.object(
                launcher.os, "fstat", side_effect=(missing, source, target)
            ),
            mock.patch.object(launcher.os.path, "samestat", return_value=False),
            mock.patch.object(launcher.os, "close") as closed,
            self.assertRaisesRegex(RuntimeError, "identity mismatch"),
        ):
            launcher._duplicate_directory_to_fixed_fd(10, target_fd=200)
        closed.assert_called_once_with(200)

    def test_every_visible_input_is_verified_before_gpu_or_run_mutation(self) -> None:
        plan = _full_launch_plan_bytes()[0]
        names = {*launcher._EXPECTED_INPUT_PATHS, "conda-explicit.txt"}
        for drifted in sorted(names):
            with (
                self.subTest(drifted=drifted),
                mock.patch.object(launcher.os, "listdir", return_value=list(names)),
                mock.patch.object(
                    launcher,
                    "_verify_regular_sha_at",
                    side_effect=lambda _fd, *, name, **_kwargs: (
                        (_ for _ in ()).throw(ValueError("drift"))
                        if name == drifted
                        else None
                    ),
                ),
                self.assertRaisesRegex(ValueError, "drift"),
            ):
                launcher._verify_pre_gpu_input_closed_world(11, plan=plan)

    def test_every_production_data_file_is_size_and_hash_verified(self) -> None:
        directory_fds = {
            name: 100 + index
            for index, name in enumerate(sorted(launcher._PRODUCTION_DATA_DIRECTORIES))
        }
        files_by_fd = {
            descriptor: [
                Path(path).name
                for path in launcher._PRODUCTION_DATA_FILES
                if Path(path).parent.as_posix() == directory
            ]
            for directory, descriptor in directory_fds.items()
        }

        def open_directory(name, _flags, *, dir_fd):
            self.assertEqual(dir_fd, 11)
            return directory_fds[name]

        def list_directory(descriptor):
            if descriptor == 11:
                return list(launcher._PRODUCTION_DATA_DIRECTORIES)
            return files_by_fd[descriptor]

        directory_metadata = SimpleNamespace(
            st_mode=stat.S_IFDIR,
            st_dev=1,
            st_ino=2,
            st_mtime_ns=3,
            st_ctime_ns=4,
        )

        with (
            mock.patch.object(launcher.os, "listdir", side_effect=list_directory),
            mock.patch.object(launcher.os, "open", side_effect=open_directory),
            mock.patch.object(
                launcher.os,
                "fstat",
                return_value=directory_metadata,
            ),
            mock.patch.object(launcher.os, "stat", return_value=directory_metadata),
            mock.patch.object(launcher.os.path, "samestat", return_value=True),
            mock.patch.object(launcher.os, "close") as close,
            mock.patch.object(launcher, "_verify_regular_sha_at") as verify,
        ):
            launcher._verify_pre_gpu_data_closed_world(11)

        self.assertEqual(verify.call_count, len(launcher._PRODUCTION_DATA_FILES))
        observed = {
            (call.args[0], call.kwargs["name"]): (
                call.kwargs["expected_bytes"],
                call.kwargs["expected_sha256"],
            )
            for call in verify.call_args_list
        }
        expected = {
            (directory_fds[Path(path).parent.as_posix()], Path(path).name): metadata
            for path, metadata in launcher._PRODUCTION_DATA_FILES.items()
        }
        self.assertEqual(observed, expected)
        self.assertEqual(
            {call.args[0] for call in close.call_args_list}, set(directory_fds.values())
        )

    def test_file_path_swap_after_hash_is_rejected(self) -> None:
        content = b"committed"
        before = SimpleNamespace(
            st_mode=stat.S_IFREG,
            st_size=len(content),
            st_dev=1,
            st_ino=2,
            st_mtime_ns=3,
            st_ctime_ns=4,
        )
        swapped = SimpleNamespace(
            st_mode=stat.S_IFREG,
            st_size=len(content),
            st_dev=1,
            st_ino=9,
            st_mtime_ns=3,
            st_ctime_ns=4,
        )
        with (
            mock.patch.object(launcher.os, "open", return_value=20),
            mock.patch.object(launcher.os, "fstat", return_value=before),
            mock.patch.object(
                launcher.os, "read", side_effect=(content, b"", b"")
            ),
            mock.patch.object(launcher.os, "stat", return_value=swapped),
            mock.patch.object(
                launcher.os.path, "samestat", side_effect=(True, False)
            ),
            mock.patch.object(launcher.os, "close"),
            self.assertRaisesRegex(RuntimeError, "pathname changed"),
        ):
            launcher._verify_regular_sha_at(
                11,
                name="artifact.bin",
                expected_sha256=hashlib.sha256(content).hexdigest(),
                maximum_bytes=1024,
                expected_bytes=len(content),
            )

    def test_production_data_extra_entry_fails_before_file_verification(self) -> None:
        with (
            mock.patch.object(
                launcher.os,
                "listdir",
                return_value=[*launcher._PRODUCTION_DATA_DIRECTORIES, "unexpected"],
            ),
            mock.patch.object(launcher.os, "open") as opened,
            mock.patch.object(launcher, "_verify_regular_sha_at") as verify,
            self.assertRaisesRegex(ValueError, "frozen directory set"),
        ):
            launcher._verify_pre_gpu_data_closed_world(11)
        opened.assert_not_called()
        verify.assert_not_called()

    @unittest.skipUnless(
        os.name == "posix" and hasattr(os, "memfd_create"),
        "sealed production-data snapshots require Linux memfd",
    )
    def test_sealed_snapshot_survives_source_path_replacement(self) -> None:
        import fcntl

        content = b"frozen-production-data"
        digest = hashlib.sha256(content).hexdigest()
        fixed_base = 700
        try:
            os.fstat(fixed_base)
        except OSError:
            pass
        else:
            self.skipTest("fixed descriptor selected for the test is occupied")
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            child = root / "sanitized_split"
            child.mkdir()
            source = child / "artifact_manifest.json"
            source.write_bytes(content)
            root_fd = os.open(
                root,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            bound = None
            try:
                with (
                    mock.patch.object(
                        launcher,
                        "_PRODUCTION_DATA_FILES",
                        {
                            "sanitized_split/artifact_manifest.json": (
                                len(content),
                                digest,
                            )
                        },
                    ),
                    mock.patch.object(
                        launcher,
                        "_PRODUCTION_DATA_DIRECTORIES",
                        frozenset({"sanitized_split"}),
                    ),
                    mock.patch.object(
                        launcher, "_PRODUCTION_DATA_FD_BASE", fixed_base
                    ),
                ):
                    launcher._verify_pre_gpu_data_closed_world(root_fd)
                    bound = launcher._bind_sealed_production_data(root_fd)
                    source.replace(child / "replaced")
                    source.write_bytes(b"attacker")
                    (child / "extra.bin").write_bytes(b"extra")
                    self.assertEqual(
                        os.pread(bound.descriptors[0], len(content), 0), content
                    )
                    required_seals = (
                        fcntl.F_SEAL_SEAL
                        | fcntl.F_SEAL_SHRINK
                        | fcntl.F_SEAL_GROW
                        | fcntl.F_SEAL_WRITE
                    )
                    self.assertEqual(
                        fcntl.fcntl(
                            bound.descriptors[0], fcntl.F_GET_SEALS
                        ),
                        required_seals,
                    )
            finally:
                if bound is not None:
                    for descriptor in bound.descriptors:
                        os.close(descriptor)
                os.close(root_fd)

    def test_unreviewed_empty_source_directory_is_rejected(self) -> None:
        materials = {
            "afts_arc/a.py": b"a = 1\n",
            "afts_arc/m04a_python_runtime_lock.py": b"pass\n",
        }
        with (
            mock.patch.object(
                launcher, "_reviewed_runtime_materials", return_value=materials
            ),
            mock.patch.object(
                launcher,
                "_read_source_tree",
                return_value=(dict(materials), frozenset({"afts_arc", "torch"})),
            ),
            self.assertRaisesRegex(ValueError, "live source tree differs"),
        ):
            launcher._load_reviewed_runtime_lock_module(
                input_fd=11,
                source_fd=12,
                expected_artifacts={"reviewed_runtime_source.zip": "a" * 64},
                expected_runtime_fingerprint_sha256="b" * 64,
            )

    def test_low_lock_fd_is_duplicated_above_stdio_before_flock(self) -> None:
        regular = SimpleNamespace(st_mode=stat.S_IFREG, st_dev=7, st_ino=11)
        fake_fcntl = types.ModuleType("fcntl")
        fake_fcntl.F_DUPFD_CLOEXEC = 1030
        fake_fcntl.LOCK_EX = 2
        fake_fcntl.LOCK_NB = 4
        fake_fcntl.fcntl = mock.Mock(return_value=9)
        fake_fcntl.flock = mock.Mock()
        with (
            mock.patch.dict(sys.modules, {"fcntl": fake_fcntl}),
            mock.patch.object(launcher.os, "open", return_value=1),
            mock.patch.object(launcher.os, "close") as close,
            mock.patch.object(launcher.os, "fstat", return_value=regular),
            mock.patch.object(launcher.os, "stat", return_value=regular),
            mock.patch.object(launcher.os.path, "samestat", return_value=True),
        ):
            descriptor, device, inode = launcher._acquire_gpu_lock(5, gpu_uuid=GPU_A)
        self.assertEqual((descriptor, device, inode), (9, 7, 11))
        fake_fcntl.fcntl.assert_called_once_with(1, fake_fcntl.F_DUPFD_CLOEXEC, 3)
        fake_fcntl.flock.assert_called_once_with(
            9, fake_fcntl.LOCK_EX | fake_fcntl.LOCK_NB
        )
        close.assert_called_once_with(1)

    def test_import_roots_bind_source_first_to_fixed_stable_descriptors(self) -> None:
        with (
            mock.patch.object(
                launcher.sysconfig,
                "get_path",
                side_effect=lambda name: (
                    "/site" if name in {"purelib", "platlib"} else None
                ),
            ),
            mock.patch.object(
                launcher, "_open_existing_directory_nofollow", return_value=8
            ) as opened,
            mock.patch.object(
                launcher,
                "_duplicate_directory_to_fixed_fd",
                side_effect=(200, 201),
            ) as duplicated,
            mock.patch.object(
                launcher.os.path, "abspath", side_effect=lambda path: path
            ),
            mock.patch.object(launcher.os, "close"),
        ):
            self.assertEqual(
                launcher._bind_import_roots(REMOTE_ROOT, source_fd=12),
                launcher._BoundImportRoots(
                    logical_roots=(f"{REMOTE_ROOT}/src", "/site"),
                    bound_roots=("/proc/self/fd/200", "/proc/self/fd/201"),
                    descriptors=(200, 201),
                ),
            )
        opened.assert_called_once_with("/site")
        self.assertEqual(
            duplicated.call_args_list,
            [mock.call(12, target_fd=200), mock.call(8, target_fd=201)],
        )

    def test_full_mocked_flow_writes_canonical_handshake_then_execs_same_pid(
        self,
    ) -> None:
        plan = _plan()
        arguments = launcher._parse_launcher_arguments(
            [
                "remote_launcher.py",
                "train-m04a",
                "--projected-artifact-bytes",
                str(15 * (1 << 30)),
                "--minimum-free-inodes",
                "4096",
            ]
        )
        plan_sha = _digest("launch-plan-file")
        inherited = {
            "AFTS_M04A_LAUNCH_PLAN_JSON": f"{REMOTE_ROOT}/launch_plan.json",
            "AFTS_M04A_LAUNCH_PLAN_SHA256": plan_sha,
            "AFTS_M04A_GPU_UUID": "attacker-value",
            "AFTS_UNRELATED": "attacker-value",
            "PYTHONPATH": "/attacker",
            "LD_PRELOAD": "/attacker.so",
            "LD_AUDIT": "/attacker-audit.so",
            "LD_LIBRARY_PATH": "/attacker-libs",
            "NCCL_DEBUG": "INFO",
            "OMP_NUM_THREADS": "999",
            "HOME": "/home/test",
        }
        before = launcher._CapacityObservation(50 * (1 << 30), 9000, 35 * (1 << 30))
        after = launcher._CapacityObservation(49 * (1 << 30), 8999, 35 * (1 << 30))
        events: list[str] = []
        captured_handshake: dict[str, object] = {}
        captured_exec: dict[str, object] = {}
        directories = {
            "root": 10,
            "input": 11,
            "data": 15,
            "source": 12,
            "runs": 13,
            "locks": 14,
        }

        def capacity(*_args, **_kwargs):
            events.append("capacity")
            return before if events.count("capacity") <= 2 else after

        query_count = 0

        def query():
            nonlocal query_count
            query_count += 1
            events.append(f"gpu-{query_count}")
            return (launcher._GpuRow(GPU_A, 0, 0, ()),)

        def write(_run_fd, *, run_root, payload):
            events.append("write-handshake")
            captured_handshake.update(payload)
            self.assertEqual(run_root, plan["run_root"])
            return f"{run_root}/lock_handshake_artifact.json", _digest("handshake-file")

        def execve(executable, argv, environment):
            events.append("execve")
            captured_exec.update(
                {"executable": executable, "argv": argv, "environment": environment}
            )

        clocks = iter((100, 110, 120))
        with (
            mock.patch.object(
                launcher, "_load_committed_launch_plan", return_value=plan
            ),
            mock.patch.object(
                launcher, "_reviewed_launcher_sha256", return_value=_digest("launcher")
            ),
            mock.patch.object(
                launcher, "_open_project_directories", return_value=directories
            ),
            mock.patch.object(
                launcher,
                "_bind_import_roots",
                return_value=launcher._BoundImportRoots(
                    logical_roots=tuple(plan["ordered_import_roots"]),
                    bound_roots=(
                        "/proc/self/fd/200",
                        "/proc/self/fd/201",
                        "/proc/self/fd/202",
                    ),
                    descriptors=(200, 201, 202),
                ),
            ),
            mock.patch.multiple(
                launcher,
                _verify_pre_gpu_input_closed_world=mock.Mock(
                    side_effect=lambda *_args, **_kwargs: events.append("verify-input")
                ),
                _verify_pre_gpu_data_closed_world=mock.Mock(
                    side_effect=lambda *_args, **_kwargs: events.append("verify-data")
                ),
                _bind_sealed_production_data=mock.Mock(
                    side_effect=lambda *_args, **_kwargs: (
                        events.append("seal-data")
                        or launcher._BoundProductionData(
                            ordered_paths=tuple(
                                sorted(launcher._PRODUCTION_DATA_FILES)
                            ),
                            descriptors=tuple(
                                launcher._PRODUCTION_DATA_FD_BASE + index
                                for index in range(
                                    len(launcher._PRODUCTION_DATA_FILES)
                                )
                            ),
                        )
                    )
                ),
                _load_reviewed_runtime_lock_module=mock.Mock(
                    side_effect=lambda **_kwargs: (
                        events.append("load-runtime-validator") or SimpleNamespace()
                    )
                ),
                _validate_pre_gpu_python_runtime=mock.Mock(
                    side_effect=lambda **_kwargs: (
                        events.append("validate-runtime")
                        or SimpleNamespace(
                            payload={"runtime_lock_id": _digest("runtime-lock-id")}
                        )
                    )
                ),
            ),
            mock.patch.object(launcher, "_check_capacity_fd", side_effect=capacity),
            mock.patch.object(launcher, "_query_gpu_snapshot", side_effect=query),
            mock.patch.object(
                launcher,
                "_create_run_root",
                side_effect=lambda *_args, **_kwargs: events.append("create-run") or 20,
            ),
            mock.patch.object(
                launcher,
                "_materialize_validation_view",
                side_effect=lambda **_kwargs: (
                    events.append("validation-view")
                    or f"{plan['run_root']}/validation-manifest"
                ),
            ),
            mock.patch.object(
                launcher,
                "_acquire_gpu_lock",
                side_effect=lambda *_args, **_kwargs: (
                    events.append("flock") or (30, 77, 88)
                ),
            ),
            mock.patch.object(
                launcher.time, "perf_counter_ns", side_effect=lambda: next(clocks)
            ),
            mock.patch.object(launcher.os, "getpid", return_value=999),
            mock.patch.object(
                launcher,
                "_linux_boot_id",
                return_value="00000000-0000-0000-0000-000000000001",
            ),
            mock.patch.object(
                launcher, "_linux_process_start_ticks", return_value=1234
            ),
            mock.patch.object(launcher.os, "set_inheritable") as inheritable,
            mock.patch.object(launcher, "_write_handshake_artifact", side_effect=write),
            mock.patch.object(
                launcher,
                "_seal_exec_descriptors",
                side_effect=lambda _fd: events.append("seal-fds"),
            ) as sealed,
            mock.patch.object(launcher.os, "execve", side_effect=execve),
            mock.patch.object(launcher.os, "close"),
            self.assertRaisesRegex(RuntimeError, "execve unexpectedly returned"),
        ):
            launcher._run_production_launcher(
                arguments,
                inherited_environment=inherited,
                launcher_path=f"{REMOTE_ROOT}/input/remote_launcher.py",
            )

        self.assertEqual(
            events,
            [
                "verify-input",
                "verify-data",
                "seal-data",
                "load-runtime-validator",
                "validate-runtime",
                "capacity",
                "gpu-1",
                "create-run",
                "validation-view",
                "capacity",
                "flock",
                "capacity",
                "gpu-2",
                "write-handshake",
                "seal-fds",
                "execve",
            ],
        )
        semantic = dict(captured_handshake)
        handshake_id = semantic.pop("handshake_id")
        self.assertEqual(handshake_id, launcher._canonical_sha256(semantic))
        self.assertEqual(
            validate_lock_handshake_payload(captured_handshake), captured_handshake
        )
        self.assertEqual(captured_handshake["lock_holder_pid"], 999)
        self.assertEqual(captured_handshake["inherited_lock_fd"], 30)
        self.assertEqual(captured_handshake["attempt_nonce"], plan["attempt_nonce"])
        self.assertEqual(captured_handshake["launch_plan_sha256"], plan_sha)
        inheritable.assert_called_once_with(30, True)
        sealed.assert_called_once_with(
            frozenset(
                {
                    30,
                    200,
                    201,
                    202,
                    *(
                        launcher._PRODUCTION_DATA_FD_BASE + index
                        for index in range(len(launcher._PRODUCTION_DATA_FILES))
                    ),
                }
            )
        )

        self.assertEqual(captured_exec["executable"], sys.executable)
        exec_argv = captured_exec["argv"]
        self.assertEqual(exec_argv[1:4], ["-I", "-B", "-S"])
        child_argv = exec_argv[exec_argv.index("train-m04a") :]
        self.assertEqual(
            child_argv,
            list(
                launcher._neural_cli_argv(
                    command="train-m04a",
                    plan_path=f"{REMOTE_ROOT}/launch_plan.json",
                    plan_sha256=plan_sha,
                    validation_manifest_dir=f"{plan['run_root']}/validation-manifest",
                    handshake_path=f"{plan['run_root']}/lock_handshake_artifact.json",
                    handshake_sha256=_digest("handshake-file"),
                    gpu_uuid=GPU_A,
                    remote_root=REMOTE_ROOT,
                    run_root=str(plan["run_root"]),
                    python_runtime_lock_sha256=plan["expected_input_artifacts"][
                        "python-runtime-lock.json"
                    ],
                )
            ),
        )
        environment = captured_exec["environment"]
        self.assertEqual(environment["AFTS_M04A_RUN_ID"], RUN_ID)
        self.assertEqual(environment["AFTS_M04A_GPU_UUID"], GPU_A)
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], GPU_A)
        self.assertEqual(environment["AFTS_M04A_INHERITED_LOCK_FD"], "30")
        self.assertEqual(environment["AFTS_M04A_INPUT_ROOT"], f"{REMOTE_ROOT}/input")
        self.assertEqual(
            environment["AFTS_M04A_LAUNCH_PLAN_JSON"],
            f"{REMOTE_ROOT}/launch_plan.json",
        )
        self.assertEqual(
            environment["AFTS_M04A_VALIDATION_MANIFEST_DIR"],
            f"{plan['run_root']}/validation-manifest",
        )
        self.assertEqual(
            environment["AFTS_M04A_PROJECTED_ARTIFACT_BYTES"], str(15 * (1 << 30))
        )
        self.assertEqual(environment["AFTS_M04A_MINIMUM_FREE_INODES"], "4096")
        self.assertNotIn("PYTHONPATH", environment)
        self.assertNotIn("LD_PRELOAD", environment)
        self.assertNotIn("LD_AUDIT", environment)
        self.assertNotIn("LD_LIBRARY_PATH", environment)
        self.assertNotIn("NCCL_DEBUG", environment)
        self.assertNotIn("OMP_NUM_THREADS", environment)
        self.assertNotIn("HOME", environment)
        self.assertNotIn("AFTS_UNRELATED", environment)
        self.assertNotEqual(environment["AFTS_M04A_GPU_UUID"], "attacker-value")
        self.assertEqual(
            environment["AFTS_EVIDENCE_LAUNCHER"],
            f"{REMOTE_ROOT}/input/remote_launcher.py",
        )
        self.assertTrue(
            Path(environment["PYTHONPYCACHEPREFIX"]).name.startswith(
                "afts-m04a-pycache-"
            )
        )
        self.assertEqual(
            json.loads(environment["AFTS_M04A_IMPORT_ROOTS_JSON"]),
            plan["ordered_import_roots"],
        )
        self.assertEqual(
            json.loads(environment["AFTS_M04A_BOUND_IMPORT_ROOTS_JSON"]),
            ["/proc/self/fd/200", "/proc/self/fd/201", "/proc/self/fd/202"],
        )


if __name__ == "__main__":
    unittest.main()
