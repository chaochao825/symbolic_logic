from __future__ import annotations

import base64
import copy
import csv
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import afts_arc.m04a_python_runtime_lock as runtime_lock


def _record_hash(content: bytes) -> str:
    encoded = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).decode("ascii")
    return f"sha256={encoded.rstrip('=')}"


def _reidentify(payload: dict[str, object]) -> dict[str, object]:
    changed = copy.deepcopy(payload)
    semantic = dict(changed)
    semantic.pop("runtime_lock_id")
    changed["runtime_lock_id"] = runtime_lock._canonical_sha256(semantic)
    return changed


class _SyntheticMixbit:
    def __init__(self, root: Path) -> None:
        self.prefix = root / "mixbit"
        self.site = self.prefix / "lib" / "python3.10" / "site-packages"
        self.bin = self.prefix / "bin"
        self.site.mkdir(parents=True)
        self.bin.mkdir(parents=True)
        (self.prefix / "share" / "man" / "man1").mkdir(parents=True)
        self.executable = self.prefix / "python3.10"
        self.executable.write_bytes(b"synthetic-cpython-3.10.20\n")
        self.record_paths: dict[str, Path] = {}
        self.owned_files: dict[str, list[Path]] = {}
        self._build_distributions()
        self.record_entry_count = sum(
            len(list(csv.reader(io.StringIO(path.read_text(encoding="utf-8")))))
            for path in self.record_paths.values()
        )
        self.verified_bytes = sum(
            sum(path.stat().st_size for path in paths)
            + self.record_paths[name].stat().st_size
            for name, paths in self.owned_files.items()
        )
        self.unhashed_count = len(self.record_paths) + 1
        self.pyc_count = 1

    def _write(self, relative: str, content: bytes) -> Path:
        path = self.site.joinpath(*relative.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def _actual_for_reference(self, reference: str) -> Path:
        return Path(os.path.normpath(self.site / Path(reference)))

    def _build_distributions(self) -> None:
        versions = runtime_lock.FROZEN_MIXBIT_CRITICAL_DISTRIBUTIONS
        active_dependencies = [name for name in sorted(versions) if name != "torch"]
        for name, version in sorted(versions.items()):
            directory = f"{name.replace('-', '_')}-{version}.dist-info"
            metadata_lines = [
                "Metadata-Version: 2.1",
                f"Name: {name}",
                f"Version: {version}",
            ]
            if name == "torch":
                metadata_lines.extend(
                    f"Requires-Dist: {dependency} (>=0)"
                    for dependency in active_dependencies
                )
                metadata_lines.extend(
                    [
                        'Requires-Dist: pytest; extra == "dev"',
                        "Requires-Dist: gmpy; (platform_python_implementation != \"PyPy\") and extra == 'gmpy'",
                        'Requires-Dist: setuptools; python_version >= "3.12"',
                        'Requires-Dist: importlib-metadata; python_version < "3.10"',
                        'Requires-Dist: cuda-all; sys_platform == "linux" and extra == "all"',
                    ]
                )
            metadata = ("\n".join(metadata_lines) + "\n\n").encode("utf-8")
            metadata_reference = f"{directory}/METADATA"
            metadata_path = self._write(metadata_reference, metadata)
            package_reference = f"payload/{name}.bin"
            package_path = self._write(package_reference, f"payload:{name}\n".encode())
            rows: list[tuple[str, str, str]] = [
                (metadata_reference, _record_hash(metadata), str(len(metadata))),
                (
                    package_reference,
                    _record_hash(package_path.read_bytes()),
                    str(package_path.stat().st_size),
                ),
            ]
            owned = [metadata_path, package_path]
            if name == "torch":
                init_path = self._write("torch/__init__.py", b"# synthetic torch\n")
                version_content = (
                    b"__version__ = '2.10.0+cu128'\n"
                    b"cuda = '12.8'\n"
                    b"git_version = '449b1768410104d3ed79d3bcfe4ba1d65c7f22c0'\n"
                )
                version_path = self._write("torch/version.py", version_content)
                pyc_path = self._write(
                    "torch/__pycache__/cached.cpython-310.pyc", b"synthetic-pyc\n"
                )
                for reference, path in (
                    ("torch/__init__.py", init_path),
                    ("torch/version.py", version_path),
                ):
                    rows.append(
                        (
                            reference,
                            _record_hash(path.read_bytes()),
                            str(path.stat().st_size),
                        )
                    )
                rows.append(("torch/__pycache__/cached.cpython-310.pyc", "", ""))
                owned.extend([init_path, version_path, pyc_path])
            dotdot = runtime_lock._ALLOWED_DOTDOT_RECORD_REFERENCES.get(
                name, frozenset()
            )
            for reference in sorted(dotdot):
                actual = self._actual_for_reference(reference)
                actual.parent.mkdir(parents=True, exist_ok=True)
                actual.write_bytes(f"script:{name}:{reference}\n".encode())
                rows.append(
                    (
                        reference,
                        _record_hash(actual.read_bytes()),
                        str(actual.stat().st_size),
                    )
                )
                owned.append(actual)
            record_reference = f"{directory}/RECORD"
            rows.append((record_reference, "", ""))
            stream = io.StringIO(newline="")
            csv.writer(stream, lineterminator="\n").writerows(rows)
            record_path = self._write(
                record_reference, stream.getvalue().encode("utf-8")
            )
            self.record_paths[name] = record_path
            self.owned_files[name] = owned

    def patches(self) -> ExitStack:
        stack = ExitStack()
        marker_environment = dict(runtime_lock.FROZEN_MIXBIT_MARKER_ENVIRONMENT)
        stack.enter_context(
            mock.patch.object(
                runtime_lock,
                "_runtime_executable_lexical_path",
                return_value=self.executable,
            )
        )
        stack.enter_context(
            mock.patch.object(runtime_lock, "_assert_proc_self_exe_identity")
        )
        stack.enter_context(
            mock.patch.object(
                runtime_lock,
                "_marker_environment",
                return_value=marker_environment,
            )
        )
        stack.enter_context(
            mock.patch.object(
                runtime_lock.platform, "python_version", return_value="3.10.20"
            )
        )
        stack.enter_context(
            mock.patch.object(
                runtime_lock,
                "_runtime_version_info",
                return_value=[3, 10, 20, "final", 0],
            )
        )
        stack.enter_context(
            mock.patch.object(
                runtime_lock,
                "FROZEN_MIXBIT_CRITICAL_RECORD_ENTRY_COUNT",
                self.record_entry_count,
            )
        )
        stack.enter_context(
            mock.patch.object(
                runtime_lock,
                "FROZEN_MIXBIT_CRITICAL_UNHASHED_ENTRY_COUNT",
                self.unhashed_count,
            )
        )
        stack.enter_context(
            mock.patch.object(
                runtime_lock,
                "FROZEN_MIXBIT_CRITICAL_PYC_ENTRY_COUNT",
                self.pyc_count,
            )
        )
        stack.enter_context(
            mock.patch.object(
                runtime_lock,
                "FROZEN_MIXBIT_CRITICAL_VERIFIED_FILE_BYTES",
                self.verified_bytes,
            )
        )
        return stack

    def build(self) -> dict[str, object]:
        with self.patches():
            return runtime_lock.build_python_runtime_lock(
                environment_prefix=self.prefix,
                site_packages_path=self.site,
            )


class RuntimeLockTests(unittest.TestCase):
    def test_regular_tree_paths_enforces_bound_within_single_leaf(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            leaf = root / "leaf"
            leaf.mkdir()
            (leaf / "first.txt").write_bytes(b"first")
            (leaf / "second.txt").write_bytes(b"second")
            with runtime_lock._AnchoredRoot(
                root, descriptor=None, label="synthetic root"
            ) as anchored:
                with self.assertRaisesRegex(ValueError, "entry bound"):
                    anchored.regular_tree_paths(
                        "leaf", maximum_entries=2, label="synthetic"
                    )

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.fixture = _SyntheticMixbit(Path(self.temporary.name))

    def test_build_reader_and_live_roundtrip(self) -> None:
        payload = self.fixture.build()
        with self.fixture.patches():
            content = runtime_lock.canonical_python_runtime_lock_bytes(payload)
            path = Path(self.temporary.name) / runtime_lock.PYTHON_RUNTIME_LOCK_FILENAME
            path.write_bytes(content)
            digest = hashlib.sha256(content).hexdigest()
            artifact = runtime_lock.read_python_runtime_lock_artifact(
                path, expected_artifact_sha256=digest
            )
            validated = runtime_lock.validate_live_python_runtime_lock(
                artifact,
                expected_artifact_sha256=digest,
                ordered_import_roots=[self.fixture.site],
            )
        self.assertEqual(validated, payload)
        self.assertEqual(
            len(payload["critical_distributions"]),
            len(runtime_lock.FROZEN_MIXBIT_CRITICAL_DISTRIBUTIONS),
        )
        torch_record = next(
            item
            for item in payload["critical_distributions"]
            if item["name"] == "torch"
        )
        pyc_rows = [
            row
            for row in torch_record["verified_files"]
            if row["path"].endswith(".pyc")
        ]
        self.assertEqual(len(pyc_rows), 1)
        self.assertIsNone(pyc_rows[0]["declared_sha256"])
        self.assertRegex(pyc_rows[0]["actual_sha256"], r"^[0-9a-f]{64}$")

    def test_tampered_critical_file_is_rejected(self) -> None:
        payload = self.fixture.build()
        with self.fixture.patches():
            content = runtime_lock.canonical_python_runtime_lock_bytes(payload)
            path = Path(self.temporary.name) / "lock.json"
            path.write_bytes(content)
            artifact = runtime_lock.read_python_runtime_lock_artifact(
                path, expected_artifact_sha256=hashlib.sha256(content).hexdigest()
            )
        (self.fixture.site / "torch" / "version.py").write_text(
            "git_version = 'tampered'\n", encoding="utf-8"
        )
        with (
            self.fixture.patches(),
            self.assertRaisesRegex(ValueError, "differs from its RECORD"),
        ):
            runtime_lock.validate_live_python_runtime_lock(
                artifact,
                expected_artifact_sha256=artifact.artifact_sha256,
            )

    def test_imported_torch_origin_and_versions_are_bound(self) -> None:
        payload = self.fixture.build()
        with self.fixture.patches():
            content = runtime_lock.canonical_python_runtime_lock_bytes(payload)
            path = Path(self.temporary.name) / "lock-imported.json"
            path.write_bytes(content)
            artifact = runtime_lock.read_python_runtime_lock_artifact(
                path, expected_artifact_sha256=hashlib.sha256(content).hexdigest()
            )
            origin = str(self.fixture.site / "torch" / "__init__.py")
            package = str(self.fixture.site / "torch")
            fake_torch = SimpleNamespace(
                __file__=origin,
                __path__=[package],
                __spec__=SimpleNamespace(
                    origin=origin, submodule_search_locations=[package]
                ),
                __version__="2.10.0+cu128",
                version=SimpleNamespace(
                    cuda="12.8",
                    git_version="449b1768410104d3ed79d3bcfe4ba1d65c7f22c0",
                ),
            )
            runtime_lock.validate_imported_torch_runtime(
                artifact,
                expected_artifact_sha256=artifact.artifact_sha256,
                imported_torch=fake_torch,
            )
            fake_torch.__spec__.origin = str(self.fixture.site / "torch.py")
            with self.assertRaisesRegex(ValueError, "origin metadata"):
                runtime_lock.validate_imported_torch_runtime(
                    artifact,
                    expected_artifact_sha256=artifact.artifact_sha256,
                    imported_torch=fake_torch,
                )

    def test_non_allowlisted_unhashed_entry_is_rejected(self) -> None:
        record = self.fixture.record_paths["filelock"]
        rows = list(csv.reader(io.StringIO(record.read_text(encoding="utf-8"))))
        path = self.fixture._write("payload/unhashed.txt", b"unhashed\n")
        rows.insert(-1, ("payload/unhashed.txt", "", ""))
        stream = io.StringIO(newline="")
        csv.writer(stream, lineterminator="\n").writerows(rows)
        record.write_text(stream.getvalue(), encoding="utf-8")
        self.fixture.owned_files["filelock"].append(path)
        with (
            self.fixture.patches(),
            self.assertRaisesRegex(ValueError, "non-allowlisted unhashed"),
        ):
            runtime_lock.build_python_runtime_lock(
                environment_prefix=self.fixture.prefix,
                site_packages_path=self.fixture.site,
            )

    def test_unknown_dotdot_record_reference_is_rejected(self) -> None:
        record = self.fixture.record_paths["filelock"]
        rows = list(csv.reader(io.StringIO(record.read_text(encoding="utf-8"))))
        rows.insert(-1, ("../../../bin/evil", "sha256=" + "A" * 43, "1"))
        stream = io.StringIO(newline="")
        csv.writer(stream, lineterminator="\n").writerows(rows)
        record.write_text(stream.getvalue(), encoding="utf-8")
        with (
            self.fixture.patches(),
            self.assertRaisesRegex(ValueError, "non-allowlisted dotdot"),
        ):
            runtime_lock.build_python_runtime_lock(
                environment_prefix=self.fixture.prefix,
                site_packages_path=self.fixture.site,
            )

    def test_missing_record_file_is_rejected(self) -> None:
        missing = self.fixture.site / "payload" / "filelock.bin"
        staged = Path(self.temporary.name) / "staged-filelock.bin"
        missing.replace(staged)
        with self.fixture.patches(), self.assertRaises(FileNotFoundError):
            runtime_lock.build_python_runtime_lock(
                environment_prefix=self.fixture.prefix,
                site_packages_path=self.fixture.site,
            )

    def test_extra_dist_info_file_is_rejected(self) -> None:
        directory = self.fixture.record_paths["filelock"].parent
        (directory / "EXTRA").write_bytes(b"not in RECORD")
        with (
            self.fixture.patches(),
            self.assertRaisesRegex(ValueError, "extra file outside RECORD"),
        ):
            runtime_lock.build_python_runtime_lock(
                environment_prefix=self.fixture.prefix,
                site_packages_path=self.fixture.site,
            )

    @unittest.skipUnless(os.name == "posix", "POSIX symlink semantics")
    def test_symlinked_critical_file_is_rejected(self) -> None:
        target = self.fixture.site / "payload" / "filelock.bin"
        staged = self.fixture.site / "payload" / "filelock.real"
        target.replace(staged)
        target.symlink_to(staged.name)
        with self.fixture.patches(), self.assertRaisesRegex(ValueError, "symlink"):
            runtime_lock.build_python_runtime_lock(
                environment_prefix=self.fixture.prefix,
                site_packages_path=self.fixture.site,
            )

    @unittest.skipUnless(os.name == "posix", "POSIX bound descriptor semantics")
    def test_bound_proc_fd_shadow_scan_uses_descriptors(self) -> None:
        source = Path(self.temporary.name) / "source-root"
        site = Path(self.temporary.name) / "site-root"
        source.mkdir()
        site.mkdir()
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        source_fd = os.open(source, flags)
        site_fd = os.open(site, flags)
        self.addCleanup(os.close, source_fd)
        self.addCleanup(os.close, site_fd)
        runtime_lock._assert_no_critical_shadow_fds([source_fd, site_fd], site_index=1)
        (source / "torch").mkdir()
        with self.assertRaisesRegex(ValueError, "critical package shadow"):
            runtime_lock._assert_no_critical_shadow_fds(
                [source_fd, site_fd], site_index=1
            )

    def test_offline_validator_recomputes_verified_file_aggregate(self) -> None:
        payload = self.fixture.build()
        changed = copy.deepcopy(payload)
        changed["critical_distributions"][0]["verified_files"][0]["actual_sha256"] = (
            "0" * 64
        )
        changed = _reidentify(changed)
        with self.assertRaisesRegex(ValueError, "declared and independently measured"):
            runtime_lock.validate_python_runtime_lock_payload(changed)

    def test_exact_critical_version_gate(self) -> None:
        payload = self.fixture.build()
        changed = copy.deepcopy(payload)
        torch_inventory = next(
            item for item in changed["distributions"] if item["name"] == "torch"
        )
        torch_inventory["version"] = "2.10.1"
        torch_critical = next(
            item
            for item in changed["critical_distributions"]
            if item["name"] == "torch"
        )
        torch_critical["version"] = "2.10.1"
        changed = _reidentify(changed)
        with self.assertRaisesRegex(ValueError, "frozen mixbit"):
            runtime_lock.validate_python_runtime_lock_payload(changed)

    def test_marker_evaluator_and_unknown_syntax(self) -> None:
        environment = runtime_lock.FROZEN_MIXBIT_MARKER_ENVIRONMENT
        self.assertFalse(runtime_lock._evaluate_marker('extra == "dev"', environment))
        self.assertTrue(
            runtime_lock._evaluate_marker(
                'platform_system == "Linux" and platform_machine == "x86_64"',
                environment,
            )
        )
        self.assertFalse(
            runtime_lock._evaluate_marker('python_version >= "3.12"', environment)
        )
        self.assertFalse(
            runtime_lock._evaluate_marker(
                '(os_name == "nt" and implementation_name != "pypy")',
                environment,
            )
        )
        with self.assertRaisesRegex(ValueError, "unsupported requirement marker"):
            runtime_lock._evaluate_marker('unknown_variable == "x"', environment)
        names, active, entries = runtime_lock._requires_dist_names_from_values(
            ['fastapi; extra == "all"', 'fastapi; extra == "all"'],
            marker_environment=environment,
        )
        self.assertEqual(names, ["fastapi"])
        self.assertEqual(active, [])
        self.assertEqual(len(entries), 2)

    def test_noncritical_unknown_marker_is_inventoried_without_blocking_closure(
        self,
    ) -> None:
        metadata_dir = self.fixture.site / "ambient-1.0.dist-info"
        metadata_dir.mkdir()
        (metadata_dir / "METADATA").write_text(
            "Metadata-Version: 2.1\n"
            "Name: ambient\n"
            "Version: 1.0\n"
            'Requires-Dist: optional; unknown_marker == "x"\n\n',
            encoding="utf-8",
        )
        payload = self.fixture.build()
        self.assertIn("ambient", {row["name"] for row in payload["distributions"]})

    def test_offline_linux_paths_validate_on_windows_host(self) -> None:
        payload = self.fixture.build()
        changed = copy.deepcopy(payload)
        prefix = "/opt/mixbit"
        changed["environment"]["prefix_path"] = prefix
        changed["environment"]["site_packages_path"] = (
            f"{prefix}/lib/python3.10/site-packages"
        )
        changed["python"].update(
            {
                "executable_lexical_path": f"{prefix}/bin/python",
                "executable_resolved_path": f"{prefix}/bin/python3.10",
                "executable_symlink_chain": [
                    {"path": f"{prefix}/bin/python", "target": "python3.10"}
                ],
            }
        )
        changed = _reidentify(changed)
        with self.fixture.patches():
            validated = runtime_lock.validate_python_runtime_lock_payload(changed)
        self.assertEqual(validated["environment"]["prefix_path"], prefix)

    def test_offline_metadata_and_marker_edges_are_closed(self) -> None:
        payload = self.fixture.build()
        changed = copy.deepcopy(payload)
        record = changed["critical_distributions"][0]
        metadata_row = next(
            row for row in record["verified_files"] if row["path"].endswith("/METADATA")
        )
        metadata_row["declared_sha256"] = "0" * 64
        metadata_row["actual_sha256"] = "0" * 64
        record["verified_files_sha256"] = runtime_lock._canonical_sha256(
            record["verified_files"]
        )
        changed = _reidentify(changed)
        with (
            self.fixture.patches(),
            self.assertRaisesRegex(ValueError, "METADATA inventory"),
        ):
            runtime_lock.validate_python_runtime_lock_payload(changed)

        changed = copy.deepcopy(payload)
        record = changed["critical_distributions"][0]
        record["active_requires_dist_names"] = ["torch"]
        changed = _reidentify(changed)
        with (
            self.fixture.patches(),
            self.assertRaisesRegex(ValueError, "requirement subset|marker evaluation"),
        ):
            runtime_lock.validate_python_runtime_lock_payload(changed)

    def test_external_artifact_payload_mutation_is_rejected(self) -> None:
        payload = self.fixture.build()
        with self.fixture.patches():
            content = runtime_lock.canonical_python_runtime_lock_bytes(payload)
            path = Path(self.temporary.name) / "mutation-lock.json"
            path.write_bytes(content)
            artifact = runtime_lock.read_python_runtime_lock_artifact(
                path, expected_artifact_sha256=hashlib.sha256(content).hexdigest()
            )
            changed = copy.deepcopy(artifact.payload)
            changed["torch_module"]["git_version"] = "mutated-git"
            changed = _reidentify(changed)
            artifact.payload.clear()
            artifact.payload.update(changed)
            with self.assertRaisesRegex(ValueError, "mutated after external read"):
                runtime_lock.validate_live_python_runtime_lock(
                    artifact,
                    expected_artifact_sha256=artifact.artifact_sha256,
                )

    def test_cross_owned_rows_must_have_identical_actual_identity(self) -> None:
        rows = [
            {
                "verified_files": [
                    {
                        "path": "lib/site-packages/nvidia/__init__.py",
                        "actual_sha256": "a" * 64,
                        "actual_bytes": 1,
                    }
                ]
            },
            {
                "verified_files": [
                    {
                        "path": "lib/site-packages/nvidia/__init__.py",
                        "actual_sha256": "b" * 64,
                        "actual_bytes": 1,
                    }
                ]
            },
        ]
        with self.assertRaisesRegex(ValueError, "cross-owned"):
            runtime_lock._assert_cross_owned_file_identities(rows)

    def test_external_reader_requires_exact_hash_and_canonical_bytes(self) -> None:
        payload = self.fixture.build()
        path = Path(self.temporary.name) / "lock.json"
        with self.fixture.patches():
            content = runtime_lock.canonical_python_runtime_lock_bytes(payload)
            path.write_bytes(content)
            with self.assertRaisesRegex(ValueError, "external commitment"):
                runtime_lock.read_python_runtime_lock_artifact(
                    path, expected_artifact_sha256="0" * 64
                )
            noncanonical = json.dumps(payload, sort_keys=True).encode("utf-8")
            path.write_bytes(noncanonical)
            with self.assertRaisesRegex(ValueError, "not canonical"):
                runtime_lock.read_python_runtime_lock_artifact(
                    path,
                    expected_artifact_sha256=hashlib.sha256(noncanonical).hexdigest(),
                )

    def test_exclusive_writer_refuses_overwrite(self) -> None:
        payload = self.fixture.build()
        path = Path(self.temporary.name) / "exclusive.json"
        with self.fixture.patches():
            digest, byte_count = runtime_lock.write_python_runtime_lock_exclusive(
                path, payload
            )
            self.assertEqual(digest, hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(byte_count, path.stat().st_size)
            with self.assertRaises(FileExistsError):
                runtime_lock.write_python_runtime_lock_exclusive(path, payload)

    def test_import_and_cli_help_do_not_import_torch(self) -> None:
        source = Path(__file__).resolve().parents[1] / "src"
        script = (
            "import sys;"
            f"sys.path.insert(0,{str(source)!r});"
            "import afts_arc.m04a_python_runtime_lock as m;"
            "assert 'torch' not in sys.modules;"
            "m.build_parser().parse_args(['build']);"
            "assert 'torch' not in sys.modules"
        )
        completed = subprocess.run(
            [sys.executable, "-I", "-B", "-S", "-c", script],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
