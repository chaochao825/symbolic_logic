from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from afts_arc import cli
from afts_arc.manifest import git_repo_state


DATASET_ORIGIN = "https://github.com/fchollet/ARC-AGI.git"
SCORER_ORIGIN = "https://github.com/arcprize/arc-agi-benchmarking.git"


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _git_bytes(root: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout


def _init_repo(root: Path, *, origin: str) -> None:
    root.mkdir(parents=True)
    _git(root, "init")
    _git(root, "config", "user.email", "phase0-tests@example.invalid")
    _git(root, "config", "user.name", "Phase 0 Tests")
    _git(root, "remote", "add", "origin", origin)


def _commit_all(root: Path) -> None:
    _git(root, "add", ".")
    _git(root, "commit", "-m", "fixture")


class CliIntegrationTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "posix", "POSIX pipe semantics required")
    def test_m04a_training_survives_real_pipe_reader_disconnect(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            marker = Path(temporary) / "campaign-continued.txt"
            script = f"""
import time
from pathlib import Path

from afts_arc import cli, m04a_campaign_command


def run_campaign(**kwargs):
    callback = kwargs["progress_callback"]
    callback({{"optimizer_step": 50}})
    time.sleep(0.2)
    callback({{"optimizer_step": 100}})
    Path({os.fspath(marker)!r}).write_text("continued", encoding="utf-8")
    return {{"status": "committed_training_artifact"}}


m04a_campaign_command.run_training_campaign_command = run_campaign
arguments = cli.argparse.Namespace(
    launch_plan="plan.json",
    launch_plan_sha256="0" * 64,
    python_runtime_lock="runtime-lock.json",
    python_runtime_lock_sha256="1" * 64,
    validation_manifest_dir="validation",
    lock_handshake_artifact="lock.json",
    lock_handshake_artifact_sha256="2" * 64,
    gpu_uuid="GPU-00000000-0000-0000-0000-000000000000",
    visible_root="visible",
    launcher_path="launcher.py",
    cost_probe_scratch_dir="scratch",
    production_data_root="data",
    working_dir="working",
    output_dir="output",
)
raise SystemExit(cli._cmd_train_m04a(arguments))
"""
            environment = dict(os.environ)
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            environment["PYTHONPATH"] = os.fspath(
                Path(__file__).resolve().parents[1] / "src"
            )
            process = subprocess.Popen(
                [sys.executable, "-B", "-c", script],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=environment,
            )
            assert process.stdout is not None and process.stderr is not None
            first_line = process.stdout.readline()
            self.assertEqual(
                first_line.replace(b"\r\n", b"\n"),
                b'{"optimizer_step":50}\n',
            )
            process.stdout.close()
            stderr = process.stderr.read()
            return_code = process.wait(timeout=10)

            self.assertEqual(return_code, 0, stderr.decode("utf-8", "replace"))
            self.assertEqual(stderr, b"")
            self.assertEqual(marker.read_text("utf-8"), "continued")

    def test_m04a_training_broken_stdout_is_observational_only(self) -> None:
        class BrokenStdout:
            def __init__(self) -> None:
                self.write_count = 0

            def write(self, _text: str) -> int:
                self.write_count += 1
                raise BrokenPipeError

            def flush(self) -> None:
                raise AssertionError("write failure must skip the explicit flush")

            def fileno(self) -> int:
                return 71

        observed_callbacks = 0

        def run_campaign(**kwargs: object) -> dict[str, object]:
            nonlocal observed_callbacks
            callback = kwargs["progress_callback"]
            self.assertTrue(callable(callback))
            callback({"optimizer_step": 50})
            observed_callbacks += 1
            callback({"optimizer_step": 100})
            observed_callbacks += 1
            return {"status": "committed_training_artifact"}

        arguments = cli.argparse.Namespace(
            launch_plan="plan.json",
            launch_plan_sha256="0" * 64,
            python_runtime_lock="runtime-lock.json",
            python_runtime_lock_sha256="1" * 64,
            validation_manifest_dir="validation",
            lock_handshake_artifact="lock.json",
            lock_handshake_artifact_sha256="2" * 64,
            gpu_uuid="GPU-00000000-0000-0000-0000-000000000000",
            visible_root="visible",
            launcher_path="launcher.py",
            cost_probe_scratch_dir="scratch",
            production_data_root="data",
            working_dir="working",
            output_dir="output",
        )
        broken = BrokenStdout()
        with (
            patch(
                "afts_arc.m04a_campaign_command.run_training_campaign_command",
                side_effect=run_campaign,
            ),
            patch.object(cli.sys, "stdout", broken),
            patch.object(cli.os, "open", return_value=73) as open_null,
            patch.object(cli.os, "dup2") as duplicate,
            patch.object(cli.os, "close") as close_null,
        ):
            self.assertEqual(cli._cmd_train_m04a(arguments), 0)

        self.assertEqual(observed_callbacks, 2)
        self.assertEqual(broken.write_count, 1)
        open_null.assert_called_once_with(
            os.devnull,
            os.O_WRONLY | getattr(os, "O_CLOEXEC", 0),
        )
        duplicate.assert_called_once_with(73, 71)
        close_null.assert_called_once_with(73)

    def test_m04a_cpu_artifact_commands_are_registered_without_torch_import(self) -> None:
        self.assertNotIn("torch", sys.modules)
        parser = cli.build_parser()
        cache = parser.parse_args(
            [
                "build-m04a-rearc-cache",
                "--sanitized-bundle-dir",
                "sanitized",
                "--sanitized-artifact-manifest-sha256",
                "0" * 64,
                "--fold",
                "train",
                "--output-dir",
                "cache",
            ]
        )
        self.assertIs(cache.handler, cli._cmd_build_m04a_rearc_cache)
        verify = parser.parse_args(
            [
                "verify-m04a-validation-manifest",
                "--manifest-dir",
                "manifest",
                "--artifact-manifest-sha256",
                "1" * 64,
            ]
        )
        self.assertIs(verify.handler, cli._cmd_verify_m04a_validation_manifest)
        self.assertNotIn("torch", sys.modules)

    def _fixture_repositories(
        self, root: Path
    ) -> tuple[Path, Path, Path, str, str]:
        dataset_repo = root / "ARC-AGI-1"
        _init_repo(dataset_repo, origin=DATASET_ORIGIN)
        training = dataset_repo / "data" / "training"
        evaluation = dataset_repo / "data" / "evaluation"
        training.mkdir(parents=True)
        evaluation.mkdir(parents=True)
        task = {
            "train": [{"input": [[0]], "output": [[1]]}],
            "test": [{"input": [[2]], "output": [[2]]}],
        }
        (training / "a.json").write_text(json.dumps(task), encoding="utf-8")
        (training / "z.json").write_text("not-json", encoding="utf-8")
        (evaluation / "e.json").write_text(json.dumps(task), encoding="utf-8")
        _commit_all(dataset_repo)

        scorer_repo = root / "scorer"
        _init_repo(scorer_repo, origin=SCORER_ORIGIN)
        scorer_file = (
            scorer_repo
            / "src"
            / "arc_agi_benchmarking"
            / "scoring"
            / "scoring.py"
        )
        scorer_file.parent.mkdir(parents=True)
        scorer_file.write_text("# pinned scorer fixture\n", encoding="utf-8")
        _commit_all(scorer_repo)
        return (
            dataset_repo,
            training,
            scorer_repo,
            _git(dataset_repo, "rev-parse", "HEAD"),
            _git(scorer_repo, "rev-parse", "HEAD"),
        )

    def test_run_d4_publishes_self_contained_atomic_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_repo, training, scorer_repo, dataset_commit, scorer_commit = (
                self._fixture_repositories(root)
            )
            scorer_relative = "src/arc_agi_benchmarking/scoring/scoring.py"
            scorer_blob = _git_bytes(
                scorer_repo, "show", f"{scorer_commit}:{scorer_relative}"
            )
            _git(scorer_repo, "update-index", "--skip-worktree", scorer_relative)
            scorer_file = scorer_repo / scorer_relative
            scorer_file.write_text("# local diagnostic differs from blob\n", encoding="utf-8")
            self.assertEqual(_git(scorer_repo, "status", "--porcelain"), "")
            output = root / "evidence"
            argv = [
                "run-d4-baseline",
                "--task-dir",
                str(training),
                "--dataset-name",
                "ARC-AGI-1",
                "--split-name",
                "public_training",
                "--repo-dir",
                str(dataset_repo),
                "--scorer-repo-dir",
                str(scorer_repo),
                "--max-tasks",
                "1",
                "--orchestrator-id",
                "test-goal",
                "--output-dir",
                str(output),
            ]
            with (
                patch.dict(
                    "afts_arc.cli.DATASET_COMMITS",
                    {"ARC-AGI-1": dataset_commit},
                ),
                patch("afts_arc.cli.SCORER_COMMIT", scorer_commit),
                redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(cli._main_in_process(argv), 0)

            manifest = json.loads((output / "run_manifest.json").read_text("utf-8"))
            summary = json.loads((output / "summary.json").read_text("utf-8"))
            artifact_manifest = json.loads(
                (output / "artifact_manifest.json").read_text("utf-8")
            )
            self.assertEqual(manifest["command"], ["afts-arc", *argv])
            self.assertEqual(manifest["extra"]["orchestrator_id"], "test-goal")
            self.assertEqual(manifest["dataset"]["file_count"], 1)
            self.assertEqual(manifest["dataset"]["available_file_count"], 2)
            self.assertEqual(
                manifest["scorer_reference"]["scoring_file_sha256"],
                hashlib.sha256(scorer_blob).hexdigest(),
            )
            self.assertEqual(
                manifest["scorer_reference"]["identity_source"], "pinned_git_blob"
            )
            self.assertNotEqual(
                manifest["scorer_reference"]["scoring_file_sha256"],
                manifest["scorer_reference"][
                    "worktree_scoring_file_sha256_diagnostic"
                ],
            )
            self.assertEqual(summary["run_id"], manifest["run_id"])
            self.assertEqual(
                summary["evidence_status"],
                "verified_harness_smoke_not_research_result",
            )
            self.assertEqual(artifact_manifest["run_id"], manifest["run_id"])
            for name, metadata in artifact_manifest["artifacts"].items():
                content = (output / name).read_bytes()
                self.assertEqual(metadata["sha256"], hashlib.sha256(content).hexdigest())
            self.assertEqual(artifact_manifest["artifacts"]["tasks.jsonl"]["rows"], 1)
            self.assertTrue((output / "source_snapshot.zip").is_file())

    def test_runtime_source_must_match_cli_import_snapshot(self) -> None:
        capture = cli.capture_runtime_source()
        with (
            patch(
                "afts_arc.cli._IMPORTED_RUNTIME_SOURCE_FINGERPRINT",
                "0" * 64,
            ),
            self.assertRaisesRegex(RuntimeError, "start a fresh process"),
        ):
            cli._require_imported_source_identity(capture.fingerprint_sha256)

    def test_package_bootstrap_blocks_preloaded_stale_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(Path(cli.__file__).resolve().parent, root / "afts_arc")
            script = """
import pathlib
import afts_arc.baselines
path = pathlib.Path(afts_arc.baselines.__file__)
path.write_text(path.read_text(encoding='utf-8') + "\\nraise RuntimeError('new disk only')\\n", encoding='utf-8')
import afts_arc.cli as cli
capture = cli.capture_runtime_source()
try:
    cli._require_imported_source_identity(capture.fingerprint_sha256)
except RuntimeError:
    print('stale dependency blocked')
    raise SystemExit(0)
raise SystemExit('guard unexpectedly accepted stale dependency')
"""
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(root)
            completed = subprocess.run(
                [sys.executable, "-B", "-c", script],
                cwd=root,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("stale dependency blocked", completed.stdout)

    def test_audit_publishes_source_bound_atomic_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_repo, training, _, dataset_commit, _ = self._fixture_repositories(root)
            output = root / "audit-evidence"
            argv = [
                "audit-data",
                "--task-dir",
                str(training),
                "--dataset-name",
                "ARC-AGI-1",
                "--split-name",
                "public_training",
                "--repo-dir",
                str(dataset_repo),
                "--hash-only",
                "--output-dir",
                str(output),
            ]
            with (
                patch.dict(
                    "afts_arc.cli.DATASET_COMMITS",
                    {"ARC-AGI-1": dataset_commit},
                ),
                redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(cli._main_in_process(argv), 0)
            audit = json.loads((output / "audit.json").read_text("utf-8"))
            artifacts = json.loads(
                (output / "artifact_manifest.json").read_text("utf-8")
            )
            self.assertEqual(artifacts["run_id"], audit["audit_id"])
            self.assertEqual(audit["dataset"]["file_count"], 2)
            self.assertTrue((output / "source_snapshot.zip").is_file())

    def test_public_evaluation_requires_release_gate_before_reading_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            argv = [
                "run-d4-baseline",
                "--task-dir",
                str(root / "evaluation"),
                "--dataset-name",
                "ARC-AGI-1",
                "--split-name",
                "public_evaluation",
                "--repo-dir",
                str(root / "dataset"),
                "--scorer-repo-dir",
                str(root / "scorer"),
                "--output-dir",
                str(root / "evidence"),
            ]
            with self.assertRaises(PermissionError):
                cli._main_in_process(argv)

    def test_task_directory_must_be_inside_declared_repository(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_repo, training, _, _, _ = self._fixture_repositories(root)
            other_repo = root / "other"
            _init_repo(other_repo, origin=DATASET_ORIGIN)
            (other_repo / "placeholder.txt").write_text("x", encoding="utf-8")
            _commit_all(other_repo)
            argv = [
                "audit-data",
                "--task-dir",
                str(training),
                "--dataset-name",
                "ARC-AGI-1",
                "--split-name",
                "public_training",
                "--repo-dir",
                str(other_repo),
            ]
            with (
                patch.dict(
                    "afts_arc.cli.DATASET_COMMITS",
                    {"ARC-AGI-1": _git(other_repo, "rev-parse", "HEAD")},
                ),
                self.assertRaisesRegex(ValueError, "canonical path"),
            ):
                cli._main_in_process(argv)
            self.assertTrue(dataset_repo.is_dir())

    def test_unpinned_dataset_commit_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_repo, training, _, _, _ = self._fixture_repositories(root)
            argv = [
                "audit-data",
                "--task-dir",
                str(training),
                "--dataset-name",
                "ARC-AGI-1",
                "--split-name",
                "public_training",
                "--repo-dir",
                str(dataset_repo),
            ]
            with self.assertRaisesRegex(ValueError, "repository commit mismatch"):
                cli._main_in_process(argv)

    def test_git_repository_argument_must_be_top_level(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_repo, _, _, _, _ = self._fixture_repositories(root)
            nested = dataset_repo / "data"
            with self.assertRaisesRegex(ValueError, "git top-level"):
                git_repo_state(nested)

    def test_ignored_json_cannot_extend_the_pinned_dataset_tree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_repo, training, _, dataset_commit, _ = self._fixture_repositories(root)
            exclude = dataset_repo / ".git" / "info" / "exclude"
            exclude.write_text("data/training/ignored.json\n", encoding="utf-8")
            (training / "ignored.json").write_text(
                json.dumps(
                    {
                        "train": [{"input": [[0]], "output": [[1]]}],
                        "test": [{"input": [[2]], "output": [[3]]}],
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(_git(dataset_repo, "status", "--porcelain"), "")
            argv = [
                "audit-data",
                "--task-dir",
                str(training),
                "--dataset-name",
                "ARC-AGI-1",
                "--split-name",
                "public_training",
                "--repo-dir",
                str(dataset_repo),
            ]
            with (
                patch.dict(
                    "afts_arc.cli.DATASET_COMMITS",
                    {"ARC-AGI-1": dataset_commit},
                ),
                self.assertRaisesRegex(ValueError, "differs from the pinned git tree"),
            ):
                cli._main_in_process(argv)

    def test_evaluation_audit_also_requires_release_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            argv = [
                "audit-data",
                "--task-dir",
                str(root / "evaluation"),
                "--dataset-name",
                "ARC-AGI-1",
                "--split-name",
                "public_evaluation",
                "--repo-dir",
                str(root / "dataset"),
            ]
            with self.assertRaises(PermissionError):
                cli._main_in_process(argv)

    def test_arbitrary_release_gate_id_is_not_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            argv = [
                "audit-data",
                "--task-dir",
                str(root / "evaluation"),
                "--dataset-name",
                "ARC-AGI-1",
                "--split-name",
                "public_evaluation",
                "--repo-dir",
                str(root / "dataset"),
                "--release-gate-id",
                "made-up-gate",
            ]
            with self.assertRaisesRegex(PermissionError, "not pre-registered"):
                cli._main_in_process(argv)

    def test_registered_evaluation_run_is_labeled_as_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_repo, _, scorer_repo, dataset_commit, scorer_commit = (
                self._fixture_repositories(root)
            )
            output = root / "evaluation-evidence"
            argv = [
                "run-d4-baseline",
                "--task-dir",
                str(dataset_repo / "data" / "evaluation"),
                "--dataset-name",
                "ARC-AGI-1",
                "--split-name",
                "public_evaluation",
                "--repo-dir",
                str(dataset_repo),
                "--scorer-repo-dir",
                str(scorer_repo),
                "--release-gate-id",
                "test-gate",
                "--output-dir",
                str(output),
            ]
            with (
                patch.dict(
                    "afts_arc.cli.DATASET_COMMITS",
                    {"ARC-AGI-1": dataset_commit},
                ),
                patch("afts_arc.cli.SCORER_COMMIT", scorer_commit),
                patch("afts_arc.cli.RELEASE_GATE_IDS", frozenset({"test-gate"})),
                redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(cli._main_in_process(argv), 0)
            summary = json.loads((output / "summary.json").read_text("utf-8"))
            self.assertIn("public_evaluation", summary["note"])
            self.assertNotIn("Training-only", summary["note"])

    def test_public_main_requires_fresh_source_loader(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(cli.sys, "pycache_prefix", None),
            self.assertRaisesRegex(RuntimeError, "afts_arc_evidence.py"),
        ):
            cli.main(["--help"])

    def test_evidence_launcher_establishes_fresh_loader_before_import(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        launcher = project_root / "scripts" / "afts_arc_evidence.py"
        completed = subprocess.run(
            [sys.executable, "-B", str(launcher), "--help"],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("usage: afts-arc", completed.stdout)

    def test_evidence_launcher_isolates_cwd_from_standard_library_imports(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        launcher = project_root / "scripts" / "afts_arc_evidence.py"
        with tempfile.TemporaryDirectory() as directory:
            hostile_cwd = Path(directory)
            (hostile_cwd / "hashlib.py").write_text(
                "raise RuntimeError('CWD_SHADOW_HASHLIB')\n", encoding="utf-8"
            )
            (hostile_cwd / "json.py").write_text(
                "raise RuntimeError('CWD_SHADOW_JSON')\n", encoding="utf-8"
            )
            completed = subprocess.run(
                [sys.executable, "-B", str(launcher), "--help"],
                cwd=hostile_cwd,
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("usage: afts-arc", completed.stdout)
        self.assertNotIn("CWD_SHADOW", completed.stderr)


if __name__ == "__main__":
    unittest.main()
