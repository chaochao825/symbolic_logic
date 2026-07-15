"""Immutable data snapshots and atomic, content-addressed evidence bundles."""

from __future__ import annotations

import hashlib
import io
import json
import os
import platform
import re
import stat
import subprocess
import sys
import uuid
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

from ._source_bootstrap import named_bytes_fingerprint, runtime_materials
from .task import ARCTask, parse_task_bytes


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(root: Path, *args: str) -> str:
    if not root.is_dir():
        raise ValueError(f"git repository directory not found: {root}")
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise ValueError(f"git {' '.join(args)} failed in {root}: {detail}")
    return completed.stdout.strip()


def _git_bytes(root: Path, *args: str) -> bytes:
    if not root.is_dir():
        raise ValueError(f"git repository directory not found: {root}")
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).decode(
            "utf-8", errors="replace"
        ).strip()
        raise ValueError(f"git {' '.join(args)} failed in {root}: {detail}")
    return completed.stdout


@dataclass(frozen=True, slots=True)
class GitRepoState:
    path: str
    commit: str
    origin: str
    clean: bool


def git_repo_state(
    repo_dir: str | Path,
    *,
    expected_origin: str | None = None,
    expected_commit: str | None = None,
    require_clean: bool = True,
) -> GitRepoState:
    root = Path(repo_dir).expanduser().resolve()
    top_level = Path(_git(root, "rev-parse", "--show-toplevel")).resolve()
    if top_level != root:
        raise ValueError(
            f"repository path must be the git top-level: provided {root}, top-level {top_level}"
        )
    commit = _git(root, "rev-parse", "HEAD")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", commit):
        raise ValueError(f"unexpected git commit format in {root}: {commit!r}")
    origin = _git(root, "remote", "get-url", "origin")
    dirty = bool(_git(root, "status", "--porcelain"))
    if expected_origin is not None:
        def normalize(value: str) -> str:
            return value.rstrip("/").removesuffix(".git").lower()

        if normalize(origin) != normalize(expected_origin):
            raise ValueError(
                f"repository origin mismatch: expected {expected_origin}, found {origin}"
            )
    if expected_commit is not None and commit.lower() != expected_commit.lower():
        raise ValueError(
            f"repository commit mismatch: expected {expected_commit.lower()}, "
            f"found {commit.lower()}"
        )
    if require_clean and dirty:
        raise ValueError(f"repository is dirty: {root}")
    return GitRepoState(path=str(root), commit=commit.lower(), origin=origin, clean=not dirty)


def git_commit(repo_dir: str | Path) -> str:
    """Fail-closed compatibility helper returning a verified clean repository HEAD."""

    return git_repo_state(repo_dir).commit


def git_blob_bytes(state: GitRepoState, relative_path: str) -> bytes:
    """Read an exact blob from the already-verified commit, not the worktree."""

    relative = PurePosixPath(relative_path)
    if (
        not relative_path
        or "\\" in relative_path
        or ":" in relative_path
        or relative.is_absolute()
        or relative.as_posix() != relative_path
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError(f"unsafe git blob path: {relative_path}")
    return _git_bytes(Path(state.path), "show", f"{state.commit}:{relative_path}")


def _require_within(path: Path, root: Path, *, label: str) -> None:
    if not path.is_relative_to(root):
        raise ValueError(f"{label} {path} is not inside repository {root}")


@dataclass(frozen=True, slots=True)
class DatasetAudit:
    dataset_name: str
    split_name: str
    task_dir: str
    repository_commit: str | None
    repository_origin: str | None
    repository_clean: bool | None
    available_file_count: int
    file_count: int
    selection_policy: str
    selected_task_ids_sha256: str
    file_manifest_sha256: str
    total_train_pairs: int
    total_test_pairs: int
    multi_test_task_count: int
    test_outputs_present: int
    validated: bool


@dataclass(frozen=True, slots=True)
class DatasetSnapshot:
    audit: DatasetAudit
    tasks: tuple[ARCTask, ...]
    selected_file_names: tuple[str, ...]
    max_tasks: int | None
    repo_state: GitRepoState | None


def snapshot_task_directory(
    task_dir: str | Path,
    *,
    dataset_name: str,
    split_name: str,
    repo_dir: str | Path | None = None,
    expected_origin: str | None = None,
    expected_commit: str | None = None,
    validate: bool = True,
    max_tasks: int | None = None,
) -> DatasetSnapshot:
    """Read each selected task once and derive parsing plus hashes from those bytes."""

    if max_tasks is not None and max_tasks <= 0:
        raise ValueError("max_tasks must be positive")
    root = Path(task_dir).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"task directory not found: {root}")
    repo_state: GitRepoState | None = None
    repo_root: Path | None = None
    if repo_dir is not None:
        repo_state = git_repo_state(
            repo_dir,
            expected_origin=expected_origin,
            expected_commit=expected_commit,
        )
        repo_root = Path(repo_state.path)
        _require_within(root, repo_root, label="task directory")

    worktree_files = sorted(root.glob("*.json"), key=lambda item: item.name)
    if repo_state is not None and repo_root is not None:
        relative_root = root.relative_to(repo_root).as_posix()
        tree_output = _git_bytes(
            repo_root,
            "ls-tree",
            "-r",
            "-z",
            "--name-only",
            repo_state.commit,
            "--",
            relative_root,
        )
        tree_paths = tuple(
            item.decode("utf-8")
            for item in tree_output.split(b"\0")
            if item
        )
        expected_parent = PurePosixPath(relative_root)
        tracked_names = tuple(
            PurePosixPath(item).name
            for item in tree_paths
            if PurePosixPath(item).parent == expected_parent
            and PurePosixPath(item).suffix == ".json"
        )
        worktree_names = tuple(file.name for file in worktree_files)
        if worktree_names != tracked_names:
            raise ValueError(
                "worktree JSON file set differs from the pinned git tree: "
                f"tracked={tracked_names!r}, worktree={worktree_names!r}"
            )
        available_names = tracked_names
    else:
        relative_root = ""
        available_names = tuple(file.name for file in worktree_files)

    if not available_names:
        raise ValueError(f"no JSON tasks found in {root}")
    selected_names = (
        available_names if max_tasks is None else available_names[:max_tasks]
    )

    named_bytes: list[tuple[str, bytes]] = []
    tasks: list[ARCTask] = []
    total_train_pairs = 0
    total_test_pairs = 0
    multi_test_tasks = 0
    outputs_present = 0
    for name in selected_names:
        file = root / name
        if repo_state is not None and repo_root is not None:
            git_path = f"{relative_root}/{name}" if relative_root else name
            raw_bytes = _git_bytes(repo_root, "show", f"{repo_state.commit}:{git_path}")
        else:
            raw_bytes = file.read_bytes()
        named_bytes.append((name, raw_bytes))
        if validate:
            task = parse_task_bytes(raw_bytes, source_path=file, task_id=file.stem)
            tasks.append(task)
            total_train_pairs += len(task.train)
            total_test_pairs += len(task.test)
            multi_test_tasks += int(len(task.test) > 1)
            outputs_present += sum(pair.output is not None for pair in task.test)

    ids = tuple(Path(name).stem for name in selected_names)
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate task IDs in {root}")
    ids_blob = "\n".join(ids).encode("utf-8")
    selection_policy = "all_files" if max_tasks is None else f"first_{max_tasks}_by_filename"
    audit = DatasetAudit(
        dataset_name=dataset_name,
        split_name=split_name,
        task_dir=str(root),
        repository_commit=repo_state.commit if repo_state else None,
        repository_origin=repo_state.origin if repo_state else None,
        repository_clean=repo_state.clean if repo_state else None,
        available_file_count=len(available_names),
        file_count=len(selected_names),
        selection_policy=selection_policy,
        selected_task_ids_sha256=hashlib.sha256(ids_blob).hexdigest(),
        file_manifest_sha256=named_bytes_fingerprint(named_bytes),
        total_train_pairs=total_train_pairs,
        total_test_pairs=total_test_pairs,
        multi_test_task_count=multi_test_tasks,
        test_outputs_present=outputs_present,
        validated=validate,
    )
    return DatasetSnapshot(
        audit=audit,
        tasks=tuple(tasks),
        selected_file_names=tuple(selected_names),
        max_tasks=max_tasks,
        repo_state=repo_state,
    )


def verify_dataset_snapshot_unchanged(snapshot: DatasetSnapshot) -> None:
    current = snapshot_task_directory(
        snapshot.audit.task_dir,
        dataset_name=snapshot.audit.dataset_name,
        split_name=snapshot.audit.split_name,
        repo_dir=snapshot.repo_state.path if snapshot.repo_state else None,
        expected_origin=snapshot.repo_state.origin if snapshot.repo_state else None,
        expected_commit=snapshot.repo_state.commit if snapshot.repo_state else None,
        validate=False,
        max_tasks=snapshot.max_tasks,
    )
    if current.selected_file_names != snapshot.selected_file_names:
        raise RuntimeError("selected dataset file list changed during the run")
    if current.audit.file_manifest_sha256 != snapshot.audit.file_manifest_sha256:
        raise RuntimeError("selected dataset content changed during the run")
    if current.repo_state != snapshot.repo_state:
        raise RuntimeError("dataset repository state changed during the run")


def audit_task_directory(
    task_dir: str | Path,
    *,
    dataset_name: str,
    split_name: str,
    repo_dir: str | Path | None = None,
    expected_origin: str | None = None,
    expected_commit: str | None = None,
    validate: bool = True,
) -> DatasetAudit:
    return snapshot_task_directory(
        task_dir,
        dataset_name=dataset_name,
        split_name=split_name,
        repo_dir=repo_dir,
        expected_origin=expected_origin,
        expected_commit=expected_commit,
        validate=validate,
    ).audit


def _snapshot_zip(materials: Mapping[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in sorted(materials.items()):
            info = zipfile.ZipInfo(name)
            info.date_time = (1980, 1, 1, 0, 0, 0)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, content)
    return buffer.getvalue()


@dataclass(frozen=True, slots=True)
class RuntimeSourceCapture:
    fingerprint_sha256: str
    snapshot_zip: bytes


def capture_runtime_source() -> RuntimeSourceCapture:
    """Fingerprint and archive one in-memory material capture atomically."""

    materials = runtime_materials()
    return RuntimeSourceCapture(
        fingerprint_sha256=named_bytes_fingerprint(materials.items()),
        snapshot_zip=_snapshot_zip(materials),
    )


def runtime_source_fingerprint() -> str:
    return named_bytes_fingerprint(runtime_materials().items())


def source_fingerprint(project_dir: str | Path | None = None) -> str:
    """Return the executing package fingerprint; ``project_dir`` is deprecated."""

    return runtime_source_fingerprint()


def test_source_fingerprint(project_dir: str | Path) -> str | None:
    root = Path(project_dir).expanduser().resolve()
    files = sorted((root / "tests").rglob("*.py"))
    if not files:
        return None
    return named_bytes_fingerprint(
        (f"tests/{path.relative_to(root / 'tests').as_posix()}", path.read_bytes())
        for path in files
    )


# This public helper is imported into test modules; keep pytest from collecting it
# as a fixture-taking test function merely because its API name starts with `test_`.
test_source_fingerprint.__test__ = False


def runtime_source_snapshot_zip() -> bytes:
    """Capture the exact executing package files in a deterministic zip archive."""

    return capture_runtime_source().snapshot_zip


def verify_source_snapshot_zip(
    snapshot_zip: bytes, *, expected_fingerprint_sha256: str
) -> dict[str, bytes]:
    """Validate a source archive and bind its named bytes to a runtime fingerprint."""

    if not isinstance(snapshot_zip, bytes):
        raise TypeError("source snapshot must be bytes")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_fingerprint_sha256):
        raise ValueError("expected source fingerprint must be lowercase SHA-256")
    materials: dict[str, bytes] = {}
    try:
        with zipfile.ZipFile(io.BytesIO(snapshot_zip), mode="r") as archive:
            infos = archive.infolist()
            if not infos or len(infos) > 1000:
                raise ValueError("source snapshot has an invalid entry count")
            total_size = 0
            for info in infos:
                name = info.filename
                relative = PurePosixPath(name)
                if (
                    not name
                    or "\\" in name
                    or ":" in name
                    or relative.is_absolute()
                    or relative.as_posix() != name
                    or any(part in {"", ".", ".."} for part in relative.parts)
                    or info.is_dir()
                    or stat.S_ISLNK(info.external_attr >> 16)
                    or info.flag_bits & 0x1
                    or info.compress_type
                    not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
                ):
                    raise ValueError(f"unsafe source snapshot entry: {name!r}")
                if name in materials:
                    raise ValueError(f"duplicate source snapshot entry: {name}")
                allowed = (
                    name == "pyproject.toml"
                    or name
                    in {
                        "scripts/afts_arc_evidence.py",
                        "scripts/afts_arc_m04a.py",
                    }
                    or (
                        len(relative.parts) >= 2
                        and relative.parts[0] == "afts_arc"
                        and name.endswith(".py")
                    )
                )
                if not allowed:
                    raise ValueError(f"unexpected source snapshot material: {name}")
                total_size += info.file_size
                if info.file_size > 16 * 1024 * 1024 or total_size > 64 * 1024 * 1024:
                    raise ValueError("source snapshot exceeds the uncompressed size limit")
                materials[name] = archive.read(info)
    except (zipfile.BadZipFile, RuntimeError, OSError) as exc:
        raise ValueError(f"invalid source snapshot ZIP: {exc}") from exc
    if "afts_arc/_source_bootstrap.py" not in materials:
        raise ValueError("source snapshot is missing the bootstrap anchor")
    actual = named_bytes_fingerprint(materials.items())
    if actual != expected_fingerprint_sha256:
        raise ValueError("source snapshot bytes do not match the runtime fingerprint")
    return materials


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_run_manifest(
    *,
    command: list[str],
    cwd: str | Path,
    dataset_audit: DatasetAudit,
    started_at: str,
    completed_at: str,
    runtime_source_sha256: str,
    test_source_sha256: str | None,
    scorer_reference: Mapping[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 2,
        "started_at": started_at,
        "completed_at": completed_at,
        "cwd": str(Path(cwd).resolve()),
        "command": command,
        "dataset": asdict(dataset_audit),
        "runtime_source_fingerprint_sha256": runtime_source_sha256,
        "test_source_fingerprint_sha256": test_source_sha256,
        "scorer_reference": dict(scorer_reference),
        "python": sys.version,
        "platform": platform.platform(),
        "process_id": os.getpid(),
        "extra": extra or {},
    }
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    payload["run_id"] = hashlib.sha256(serialized.encode("ascii")).hexdigest()[:20]
    return payload


def serialize_json(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def serialize_jsonl(rows: Iterable[dict[str, Any]]) -> bytes:
    lines = [
        json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) for row in rows
    ]
    return (("\n".join(lines) + "\n") if lines else "").encode("utf-8")


def write_json_new(path: str | Path, payload: Any) -> None:
    """Strictly serialize before exclusive creation, preventing JSON-error partials."""

    content = serialize_json(payload)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as handle:
        handle.write(content)


def write_jsonl_new(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    """Strictly serialize all rows before exclusive creation."""

    content = serialize_jsonl(rows)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as handle:
        handle.write(content)


def publish_evidence_bundle(
    output_dir: str | Path,
    *,
    artifacts: Mapping[str, bytes],
    run_id: str,
) -> dict[str, Any]:
    """Write to a sibling staging directory and atomically publish when complete.

    Failed staging directories are preserved for inspection and must be moved to the
    project ``trash/`` area before manual cleanup.
    """

    target = Path(output_dir).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite evidence directory: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir(exist_ok=False)
    try:
        metadata: dict[str, Any] = {}
        staging_root = staging.resolve()
        for name, content in sorted(artifacts.items()):
            if not isinstance(name, str):
                raise TypeError("artifact name must be a string")
            relative = PurePosixPath(name)
            has_windows_reserved_component = any(
                _is_windows_reserved_component(part) for part in relative.parts
            )
            if (
                not name
                or "\\" in name
                or ":" in name
                or any(ord(character) < 32 or character in '<>"|?*' for character in name)
                or relative.is_absolute()
                or relative.as_posix() != name
                or any(part in {"", ".", ".."} for part in relative.parts)
                or has_windows_reserved_component
            ):
                raise ValueError(f"unsafe artifact path: {name}")
            if name == "artifact_manifest.json":
                raise ValueError("artifact_manifest.json is reserved")
            if not isinstance(content, bytes):
                raise TypeError(f"artifact content must be bytes: {name}")
            destination = staging.joinpath(*relative.parts).resolve()
            _require_within(destination, staging_root, label="artifact path")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as handle:
                handle.write(content)
            written = destination.read_bytes()
            if written != content:
                raise OSError(f"artifact verification mismatch: {name}")
            metadata[name] = {
                "sha256": hashlib.sha256(written).hexdigest(),
                "bytes": len(written),
                "rows": written.count(b"\n") if name.endswith(".jsonl") else None,
            }

        artifact_manifest = {
            "schema_version": 1,
            "bundle_status": "complete",
            "run_id": run_id,
            "artifacts": metadata,
        }
        write_json_new(staging / "artifact_manifest.json", artifact_manifest)
        staging.rename(target)
        return artifact_manifest
    except Exception as exc:
        raise RuntimeError(f"evidence staging failed; preserved at {staging}: {exc}") from exc


def _is_windows_reserved_component(part: str) -> bool:
    if not part or part[-1] in {" ", "."}:
        return True
    stem = part.split(".", 1)[0].rstrip(" ").upper()
    return stem in {"CON", "PRN", "AUX", "NUL"} or bool(
        re.fullmatch(r"(?:COM|LPT)[1-9]", stem)
    )
