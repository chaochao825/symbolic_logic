from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from afts_arc._source_bootstrap import named_bytes_fingerprint
from afts_arc import cli
from afts_arc.manifest import (
    capture_runtime_source,
    publish_evidence_bundle,
    serialize_json,
)
from afts_arc.m04a_data import (
    ARC2Parent,
    M04AExample,
    M04ATrainingData,
    build_validation_episodes,
)
import afts_arc.m04a_input_bundle as input_bundle
import afts_arc.m04a_launch_plan as launch_plan
from afts_arc.m04a_validation_manifest import publish_validation_episode_manifest


def _digest_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _zip_one(name: str, content: bytes) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(
        buffer, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
        info.create_system = 3
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o100644 << 16
        archive.writestr(info, content, compresslevel=9)
    return buffer.getvalue()


def _sanitized_fixture(root: Path) -> tuple[Path, str]:
    definitions = (
        ("arc2_train.zip", "arc2", "train", "aaaaaaaa"),
        ("arc2_validation.zip", "arc2", "validation", "bbbbbbbb"),
        ("rearc_train.zip", "rearc", "train", "cccccccc"),
        ("rearc_validation.zip", "rearc", "validation", "dddddddd"),
    )
    artifacts: dict[str, bytes] = {}
    shards: dict[str, dict[str, object]] = {}
    allowlists: dict[str, list[str]] = {}
    counts: dict[str, int] = {}
    for artifact_name, source, fold, parent_id in definitions:
        member_name = f"{source}/{fold}/{parent_id}.json"
        content = serialize_json({"fixture": member_name})
        archive_bytes = _zip_one(member_name, content)
        artifacts[artifact_name] = archive_bytes
        fold_name = f"{source}_{fold}"
        allowlists[fold_name] = [parent_id]
        counts[fold_name] = 1
        shards[artifact_name] = {
            "source": source,
            "fold": fold,
            "member_prefix": f"{source}/{fold}/",
            "compression": "ZIP_DEFLATED_level_9",
            "zip_member_timestamp": "1980-01-01T00:00:00",
            "zip_member_mode": "0100644",
            "member_count": 1,
            "ordered_source_parent_ids": [parent_id],
            "member_aggregate_sha256": named_bytes_fingerprint(
                ((member_name, content),)
            ),
            "members": [
                {
                    "path": member_name,
                    "source_parent_id": parent_id,
                    "semantic_parent_id": parent_id,
                    "sha256": _digest_bytes(content),
                    "bytes": len(content),
                }
            ],
            "zip_sha256": _digest_bytes(archive_bytes),
            "zip_bytes": len(archive_bytes),
        }
    split = {
        "schema": "afts.m04a-sanitized-split/v1",
        "data_folds_semantics_version": "afts-m04a-parent-folds/v0.1",
        "sealed_artifact_manifest_sha256": "0" * 64,
        "source_commitments": {},
        "usable_fold_counts": counts,
        "ordered_allowlists": allowlists,
        "semantic_aliases_in_usable_shards": [],
        "validation_quarantine_ids": [],
        "training_quarantine_ids": [],
        "quarantine_reasons": {},
        "shards": shards,
        "protected_payloads_absent": True,
    }
    artifacts["data_split_manifest.json"] = serialize_json(split)
    publish_evidence_bundle(root, artifacts=artifacts, run_id="sanitized-fixture")
    outer_sha = _digest_bytes((root / "artifact_manifest.json").read_bytes())
    return root, outer_sha


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


def _test_snapshot() -> tuple[bytes, str]:
    materials = {
        "tests/test_reviewed_fixture.py": b"def test_reviewed():\n    assert True\n"
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in materials.items():
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, content)
    return buffer.getvalue(), named_bytes_fingerprint(materials.items())


class M04AInputBundleTests(unittest.TestCase):
    def test_exclusive_activation_refuses_existing_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "staging"
            target = root / "target"
            source.mkdir()
            target.mkdir()
            with self.assertRaises((FileExistsError, OSError)):
                input_bundle._rename_directory_noreplace(source, target)
            self.assertTrue(source.is_dir())
            self.assertTrue(target.is_dir())

    @unittest.skipUnless(os.name == "posix", "FIFO/O_NONBLOCK behavior is POSIX-only")
    def test_safe_reader_rejects_fifo_without_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fifo = Path(directory) / "input.fifo"
            os.mkfifo(fifo)
            with self.assertRaisesRegex(ValueError, "regular file|metadata mismatch"):
                input_bundle._safe_read(
                    fifo,
                    expected_sha256=_digest_bytes(b""),
                    label="FIFO fixture",
                    max_bytes=1024,
                )

    def test_source_drift_and_overlapping_layout_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "source.bin"
            path.write_bytes(b"before")
            reference = input_bundle._SourceReference(
                path, _digest_bytes(b"before"), "drift fixture", 1024
            )
            path.write_bytes(b"after")
            with self.assertRaisesRegex(ValueError, "SHA-256|metadata mismatch"):
                input_bundle._assert_sources_unchanged((reference,))
            self.assertTrue(input_bundle._paths_overlap(root, root / "child"))

    def test_cli_hook_keeps_visible_and_control_outputs_explicit(self) -> None:
        digest = "0" * 64
        args = cli.build_parser().parse_args(
            [
                "materialize-m04a-input-bundle",
                "--sanitized-bundle-dir",
                "sanitized",
                "--sanitized-artifact-manifest-sha256",
                digest,
                "--validation-manifest-dir",
                "validation",
                "--validation-artifact-manifest-sha256",
                digest,
                "--runtime-source-zip",
                "runtime.zip",
                "--runtime-source-zip-sha256",
                digest,
                "--runtime-source-fingerprint-sha256",
                digest,
                "--test-snapshot-zip",
                "tests.zip",
                "--test-snapshot-zip-sha256",
                digest,
                "--test-source-fingerprint-sha256",
                digest,
                "--launcher-path",
                "launcher.py",
                "--launcher-sha256",
                digest,
                "--frozen-contract",
                "contract.md",
                "--frozen-contract-sha256",
                digest,
                "--conda-explicit",
                "conda.txt",
                "--conda-explicit-sha256",
                digest,
                "--python-runtime-lock",
                "python-runtime-lock.json",
                "--python-runtime-lock-sha256",
                digest,
                "--run-id",
                "run",
                "--remote-project-root",
                "/srv/afts",
                "--attempt-nonce",
                digest,
                "--ordered-import-root",
                "/srv/afts/src",
                "--ordered-import-root",
                "/opt/site-packages",
                "--visible-output-dir",
                "input",
                "--control-output-dir",
                "control",
            ]
        )
        self.assertEqual(args.visible_output_dir, "input")
        self.assertEqual(args.control_output_dir, "control")
        self.assertEqual(
            args.ordered_import_root, ["/srv/afts/src", "/opt/site-packages"]
        )

    def test_short_replay_is_exact_and_module_import_is_torch_free(self) -> None:
        fixture = input_bundle.build_short_exact_replay_fixture()
        self.assertEqual(
            fixture["episode_seed_cases"][0]["seed_u64"], 5_146_264_140_296_529_905
        )
        self.assertEqual(fixture["three_shape_lane_case"]["allocations"], [22, 21, 21])
        self.assertEqual(fixture["parameter_count"], 8_733_706)
        input_bundle.validate_short_exact_replay_fixture(fixture)
        forged = dict(fixture)
        forged["fixture_id"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "does not replay"):
            input_bundle.validate_short_exact_replay_fixture(forged)

        project = Path(__file__).resolve().parents[1]
        code = (
            "import sys; sys.modules['torch']=None; "
            "import afts_arc.m04a_input_bundle as m; "
            "assert 'torch' not in m.__dict__; "
            "assert m.build_short_exact_replay_fixture()['schema'].endswith('/v0.1')"
        )
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(project / "src")
        completed = subprocess.run(
            [sys.executable, "-B", "-c", code],
            cwd=project,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_sanitized_reader_is_externally_bound_and_rejects_protected_fields(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, outer_sha = _sanitized_fixture(Path(directory) / "sanitized")
            committed = input_bundle.read_committed_sanitized_input(
                root,
                expected_artifact_manifest_sha256=outer_sha,
                require_production_counts=False,
            )
            self.assertEqual(committed.artifact_manifest_sha256, outer_sha)
            self.assertEqual(committed.data_split["protected_payloads_absent"], True)
            with self.assertRaisesRegex(ValueError, "external|SHA-256|metadata"):
                input_bundle.read_committed_sanitized_input(
                    root,
                    expected_artifact_manifest_sha256="1" * 64,
                    require_production_counts=False,
                )

            split_path = root / "data_split_manifest.json"
            split = json.loads(split_path.read_text("utf-8"))
            split["collision_rows"] = []
            split_path.write_bytes(serialize_json(split))
            outer = json.loads((root / "artifact_manifest.json").read_text("utf-8"))
            outer["artifacts"]["data_split_manifest.json"]["sha256"] = _digest_bytes(
                split_path.read_bytes()
            )
            outer["artifacts"]["data_split_manifest.json"]["bytes"] = (
                split_path.stat().st_size
            )
            (root / "artifact_manifest.json").write_bytes(serialize_json(outer))
            forged_outer_sha = _digest_bytes(
                (root / "artifact_manifest.json").read_bytes()
            )
            with self.assertRaisesRegex(ValueError, "schema|protected"):
                input_bundle.read_committed_sanitized_input(
                    root,
                    expected_artifact_manifest_sha256=forged_outer_sha,
                    require_production_counts=False,
                )

    def test_flat_visible_and_external_control_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sanitized_root, sanitized_sha = _sanitized_fixture(root / "sanitized")
            validation_root = root / "validation"
            validation = publish_validation_episode_manifest(
                validation_root, _validation_episodes()
            )
            validation_sha = str(validation.artifact_manifest_sha256)

            runtime = capture_runtime_source()
            runtime_zip_path = root / "runtime.zip"
            runtime_zip_path.write_bytes(runtime.snapshot_zip)
            test_zip, test_fingerprint = _test_snapshot()
            test_zip_path = root / "tests.zip"
            test_zip_path.write_bytes(test_zip)
            project = Path(__file__).resolve().parents[1]
            launcher_path = project / "scripts" / "afts_arc_m04a.py"
            contract_path = (
                project / "notes" / "design" / "m04a-global-masked-grid-contract.md"
            )
            conda_path = root / "conda-explicit.txt"
            conda_path.write_bytes(
                b"# platform: linux-64\n@EXPLICIT\nhttps://example.invalid/pkg.tar.bz2\n"
            )
            runtime_lock_path = root / input_bundle.PYTHON_RUNTIME_LOCK_NAME
            runtime_lock_path.write_bytes(
                serialize_json(
                    {
                        "schema": "afts-m04a-python-runtime-lock/v0.1",
                        "fixture_nonproduction": True,
                    }
                )
            )
            sources = input_bundle.M04AInputBundleSources(
                sanitized_bundle_dir=sanitized_root,
                sanitized_artifact_manifest_sha256=sanitized_sha,
                validation_manifest_dir=validation_root,
                validation_artifact_manifest_sha256=validation_sha,
                runtime_source_zip_path=runtime_zip_path,
                runtime_source_zip_sha256=_digest_bytes(runtime.snapshot_zip),
                runtime_source_fingerprint_sha256=runtime.fingerprint_sha256,
                test_snapshot_zip_path=test_zip_path,
                test_snapshot_zip_sha256=_digest_bytes(test_zip),
                test_source_fingerprint_sha256=test_fingerprint,
                launcher_path=launcher_path,
                launcher_sha256=_digest_bytes(launcher_path.read_bytes()),
                frozen_contract_path=contract_path,
                frozen_contract_sha256=_digest_bytes(contract_path.read_bytes()),
                conda_explicit_path=conda_path,
                conda_explicit_sha256=_digest_bytes(conda_path.read_bytes()),
                python_runtime_lock_path=runtime_lock_path,
                python_runtime_lock_sha256=_digest_bytes(
                    runtime_lock_path.read_bytes()
                ),
                run_id="m04a-fixture-r1",
                remote_project_root="/srv/afts",
                attempt_nonce=_digest_bytes(b"attempt-nonce"),
                ordered_import_roots=("/srv/afts/src", "/opt/site-packages"),
            )
            visible = root / "input"
            control = root / "input-control"
            expected_paths = set(launch_plan.EXPECTED_INPUT_ARTIFACT_PATHS) | {
                input_bundle.PYTHON_RUNTIME_LOCK_NAME
            }
            with (
                mock.patch.object(
                    input_bundle,
                    "EXPECTED_INPUT_ARTIFACT_PATHS",
                    frozenset(expected_paths),
                ),
                mock.patch.object(
                    launch_plan,
                    "EXPECTED_INPUT_ARTIFACT_PATHS",
                    frozenset(expected_paths),
                ),
                mock.patch.object(
                    input_bundle,
                    "_validate_opaque_runtime_lock",
                    side_effect=lambda content: json.loads(content),
                ),
                mock.patch.object(
                    input_bundle,
                    "_read_committed_runtime_lock",
                    side_effect=lambda path, expected_sha256: path.read_bytes(),
                ),
            ):
                bundle = input_bundle.materialize_m04a_input_bundle(
                    visible,
                    control_output_directory=control,
                    sources=sources,
                    allow_nonproduction_fixture=True,
                )
                self.assertEqual(
                    {entry.name for entry in os.scandir(visible)},
                    expected_paths | {input_bundle.CONDA_EXPLICIT_NAME},
                )
                self.assertNotIn(
                    input_bundle.LAUNCH_PLAN_NAME,
                    {entry.name for entry in os.scandir(visible)},
                )
                self.assertEqual(
                    {entry.name for entry in os.scandir(control)},
                    {input_bundle.LAUNCH_PLAN_NAME, input_bundle.BUNDLE_MANIFEST_NAME},
                )
                replay = input_bundle.replay_validate_m04a_input_bundle(
                    bundle, sources=sources, allow_nonproduction_fixture=True
                )
                self.assertEqual(replay["status"], "VERIFIED")
                self.assertEqual(
                    bundle.launch_plan.payload["expected_input_artifacts"].keys(),
                    expected_paths,
                )
                with self.assertRaises(FileExistsError):
                    input_bundle.materialize_m04a_input_bundle(
                        visible,
                        control_output_directory=control,
                        sources=sources,
                        allow_nonproduction_fixture=True,
                    )


if __name__ == "__main__":
    unittest.main()
