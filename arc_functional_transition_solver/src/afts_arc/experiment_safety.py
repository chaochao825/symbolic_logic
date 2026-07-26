"""Small, dependency-free integrity guards for ARC experiment artifacts.

The functions in this module validate provenance and split boundaries only.
They deliberately do not load datasets, score predictions, or alter solver
behavior.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ONLINE_SUMMARY_SCHEMAS = frozenset({"afts.online-matched-budget/v3"})
STRICT_POSTHOC_ORACLE_ACCESS = frozenset({"posthoc_only_after_each_frozen_replay"})
NATIVE_PROFILE_SCHEMAS = frozenset(
    {"afts.native-budget-profile/v1", "afts.native-budget-profile/v2"}
)
POOL_MANIFEST_SCHEMAS = frozenset({"afts.frozen-action-pool/v3"})
SOURCE_PROVENANCE_PATHS = ("src", "scripts", "config")
ARC_EXPERIMENT_SOURCE_PATHS = (
    "src",
    "config",
    "requirements.txt",
    "requirements-lock.txt",
    "requirements-ca.txt",
    "requirements-difflogic-arc.txt",
    "arc_functional_transition_solver/src",
    "arc_functional_transition_solver/scripts",
    "arc_functional_transition_solver/pyproject.toml",
    "arc_functional_transition_solver/requirements-metareasoning-lock.txt",
)


class ExperimentSafetyError(ValueError):
    """Raised when an artifact cannot support its declared safety claim."""


def canonical_json(value: object) -> str:
    """Serialize JSON with a single deterministic, finite-number encoding."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def canonical_sha256(value: object) -> str:
    """Return the SHA-256 digest of :func:`canonical_json`."""

    return hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Hash one file without loading it completely into memory."""

    resolved = Path(path)
    if not resolved.is_file():
        raise ExperimentSafetyError(f"not a regular file: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: str | Path, payload: object) -> bool:
    """Atomically create an immutable JSON artifact.

    Returns ``False`` when an existing file is byte-identical.  A different
    existing artifact is never replaced; callers must choose a new path.
    """

    resolved = Path(path)
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    encoded = text.encode("utf-8")
    if resolved.exists():
        if not resolved.is_file() or resolved.read_bytes() != encoded:
            raise ExperimentSafetyError(
                f"refusing to replace a different artifact: {resolved}"
            )
        return False
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=resolved.parent,
        prefix=f".{resolved.name}.",
        suffix=".partial",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    # The temporary is intentionally left in place if this final atomic move
    # fails, so interrupted writes are inspectable instead of being discarded.
    os.replace(temporary, resolved)
    return True


def runtime_metadata(
    distributions: Iterable[str] = (),
) -> dict[str, object]:
    """Capture interpreter, selected package, and determinism environment data."""

    packages: dict[str, str | None] = {}
    for name in sorted(set(distributions)):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    environment_names = (
        "PYTHONHASHSEED",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    return {
        "python_executable": sys.executable,
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "determinism_environment": {
            name: os.environ.get(name) for name in environment_names
        },
    }


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExperimentSafetyError(f"{name} must be a mapping")
    return value


def _sequence(value: object, name: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise ExperimentSafetyError(f"{name} must be a list")
    return value


def _nonnegative_int(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ExperimentSafetyError(f"{name} must be a non-negative integer")
    return value


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ExperimentSafetyError(f"{name} must be a non-empty string")
    return value


def _sha256_string(value: object, name: str) -> str:
    text = _nonempty_string(value, name)
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ExperimentSafetyError(f"{name} must be a lowercase SHA-256 digest")
    return text


def _verify_content_id(payload: Mapping[str, Any], field: str) -> str:
    declared = _sha256_string(payload.get(field), field)
    body = dict(payload)
    body.pop(field, None)
    observed = canonical_sha256(body)
    if declared != observed:
        raise ExperimentSafetyError(f"{field} does not match canonical content")
    return declared


@dataclass(frozen=True, slots=True, order=True)
class TaskFingerprint:
    """Three independent identities used to guard development boundaries."""

    task_id: str
    task_source_sha256: str
    blind_content_sha256: str

    def __post_init__(self) -> None:
        _nonempty_string(self.task_id, "task_id")
        _sha256_string(self.task_source_sha256, "task_source_sha256")
        _sha256_string(self.blind_content_sha256, "blind_content_sha256")

    @classmethod
    def from_mapping(cls, value: object) -> "TaskFingerprint":
        item = _mapping(value, "task fingerprint")
        return cls(
            task_id=_nonempty_string(item.get("task_id"), "task_id"),
            task_source_sha256=_sha256_string(
                item.get("task_source_sha256"), "task_source_sha256"
            ),
            blind_content_sha256=_sha256_string(
                item.get("blind_content_sha256"), "blind_content_sha256"
            ),
        )

    def to_json_dict(self) -> dict[str, str]:
        return {
            "task_id": self.task_id,
            "task_source_sha256": self.task_source_sha256,
            "blind_content_sha256": self.blind_content_sha256,
        }


def extract_task_fingerprints(
    payload: Mapping[str, Any],
) -> tuple[TaskFingerprint, ...]:
    """Extract completed-task fingerprints from a summary-like mapping."""

    tasks = _sequence(payload.get("tasks"), "tasks")
    fingerprints = tuple(TaskFingerprint.from_mapping(item) for item in tasks)
    _assert_unique_fingerprints(fingerprints, label="task set")
    return fingerprints


def _axis_values(
    fingerprints: Iterable[TaskFingerprint],
) -> dict[str, set[str]]:
    items = tuple(fingerprints)
    return {
        "task_id": {item.task_id for item in items},
        "task_source_sha256": {item.task_source_sha256 for item in items},
        "blind_content_sha256": {item.blind_content_sha256 for item in items},
    }


def _assert_unique_fingerprints(
    fingerprints: Sequence[TaskFingerprint], *, label: str
) -> None:
    axes = _axis_values(fingerprints)
    for axis, values in axes.items():
        if len(values) != len(fingerprints):
            raise ExperimentSafetyError(f"{label} contains duplicate {axis}")


def assert_three_axis_disjoint(
    fit: Iterable[TaskFingerprint], heldout: Iterable[TaskFingerprint]
) -> None:
    """Reject overlap by task ID, source file, or oracle-free task content."""

    fit_items = tuple(fit)
    heldout_items = tuple(heldout)
    _assert_unique_fingerprints(fit_items, label="fit set")
    _assert_unique_fingerprints(heldout_items, label="heldout set")
    fit_axes = _axis_values(fit_items)
    heldout_axes = _axis_values(heldout_items)
    collisions = {
        axis: sorted(fit_axes[axis] & heldout_axes[axis])
        for axis in fit_axes
        if fit_axes[axis] & heldout_axes[axis]
    }
    if collisions:
        axes = ", ".join(sorted(collisions))
        raise ExperimentSafetyError(f"fit/heldout overlap on: {axes}")


@dataclass(frozen=True, slots=True)
class SummaryIntegrity:
    schema: str
    result_id: str
    split: str
    requested_task_count: int
    completed_task_count: int
    failed_task_count: int
    task_fingerprints: tuple[TaskFingerprint, ...]


def validate_summary(
    payload: object,
    *,
    allowed_schemas: Iterable[str] = ONLINE_SUMMARY_SCHEMAS,
    allowed_oracle_access: Iterable[str] = STRICT_POSTHOC_ORACLE_ACCESS,
    required_split: str | None = "training",
    require_no_training: bool = True,
) -> SummaryIntegrity:
    """Validate a matched-budget summary before using it as evidence."""

    summary = _mapping(payload, "summary")
    schemas = frozenset(allowed_schemas)
    schema = _nonempty_string(summary.get("schema"), "schema")
    if not schemas or schema not in schemas:
        raise ExperimentSafetyError(f"summary schema is not allowed: {schema}")
    result_id = _verify_content_id(summary, "result_id")

    oracle_access = _nonempty_string(summary.get("oracle_access"), "oracle_access")
    if oracle_access not in frozenset(allowed_oracle_access):
        raise ExperimentSafetyError("summary oracle_access is not strictly post-hoc")
    if require_no_training and summary.get("training_started") is not False:
        raise ExperimentSafetyError("summary must declare training_started=false")

    dataset = _mapping(summary.get("dataset"), "dataset")
    split = _nonempty_string(dataset.get("split"), "dataset.split")
    if required_split is not None and split != required_split:
        raise ExperimentSafetyError(
            f"dataset split {split!r} is not the required {required_split!r} split"
        )
    requested = _nonnegative_int(
        dataset.get("requested_task_count"), "dataset.requested_task_count"
    )
    completed = _nonnegative_int(
        dataset.get("completed_task_count"), "dataset.completed_task_count"
    )
    failed = _nonnegative_int(
        dataset.get("failed_task_count"), "dataset.failed_task_count"
    )
    if requested != completed + failed:
        raise ExperimentSafetyError(
            "requested_task_count must equal completed_task_count + failed_task_count"
        )

    tasks = _sequence(summary.get("tasks"), "tasks")
    failures = _sequence(summary.get("failures"), "failures")
    if len(tasks) != completed or len(failures) != failed:
        raise ExperimentSafetyError("task/failure records disagree with dataset counts")
    requested_ids_raw = _sequence(dataset.get("task_ids"), "dataset.task_ids")
    requested_ids = tuple(
        _nonempty_string(item, "dataset.task_ids item") for item in requested_ids_raw
    )
    if len(requested_ids) != requested or len(set(requested_ids)) != requested:
        raise ExperimentSafetyError("dataset.task_ids must uniquely enumerate requests")

    fingerprints = extract_task_fingerprints(summary)
    completed_ids = {item.task_id for item in fingerprints}
    failure_ids = {
        _nonempty_string(_mapping(item, "failure").get("task_id"), "failure.task_id")
        for item in failures
    }
    if len(failure_ids) != len(failures) or completed_ids & failure_ids:
        raise ExperimentSafetyError("completed and failed task IDs must be disjoint")
    if completed_ids | failure_ids != set(requested_ids):
        raise ExperimentSafetyError("task and failure records must cover every request")

    return SummaryIntegrity(
        schema=schema,
        result_id=result_id,
        split=split,
        requested_task_count=requested,
        completed_task_count=completed,
        failed_task_count=failed,
        task_fingerprints=fingerprints,
    )


@dataclass(frozen=True, slots=True)
class NativeProfileIntegrity:
    schema: str
    profile_id: str
    overlap_guard: str
    fit_fingerprints: tuple[TaskFingerprint, ...]


def validate_native_budget_profile(
    payload: object,
    *,
    heldout_fingerprints: Iterable[TaskFingerprint] | None = None,
) -> NativeProfileIntegrity:
    """Verify a v1/v2 profile and, for v2, enforce fit/heldout isolation."""

    profile = _mapping(payload, "native budget profile")
    schema = _nonempty_string(profile.get("schema"), "schema")
    if schema not in NATIVE_PROFILE_SCHEMAS:
        raise ExperimentSafetyError(f"native profile schema is not allowed: {schema}")
    profile_id = _verify_content_id(profile, "profile_id")
    if profile.get("oracle_fields_read") != 0:
        raise ExperimentSafetyError("native profile must declare oracle_fields_read=0")

    if schema.endswith("/v1"):
        return NativeProfileIntegrity(schema, profile_id, "unavailable", ())

    validate_clean_source_binding(profile)
    raw_fit = _sequence(profile.get("fit_task_fingerprints"), "fit_task_fingerprints")
    fit = tuple(TaskFingerprint.from_mapping(item) for item in raw_fit)
    if not fit:
        raise ExperimentSafetyError("v2 profile requires fit_task_fingerprints")
    _assert_unique_fingerprints(fit, label="profile fit set")
    declared_count = _nonnegative_int(profile.get("fit_task_count"), "fit_task_count")
    if declared_count != len(fit):
        raise ExperimentSafetyError("fit_task_count disagrees with fit fingerprints")
    declared_ids_digest = _sha256_string(
        profile.get("fit_task_ids_sha256"), "fit_task_ids_sha256"
    )
    if declared_ids_digest != canonical_sha256(sorted(item.task_id for item in fit)):
        raise ExperimentSafetyError(
            "fit_task_ids_sha256 disagrees with fit fingerprints"
        )

    if heldout_fingerprints is None:
        guard = "available_not_checked"
    else:
        assert_three_axis_disjoint(fit, tuple(heldout_fingerprints))
        guard = "verified"
    return NativeProfileIntegrity(schema, profile_id, guard, fit)


def _candidate_id(candidate: object) -> str:
    item = _mapping(candidate, "candidate")
    return _nonempty_string(item.get("hypothesis_id"), "candidate.hypothesis_id")


def candidate_dag_id(manifest: object) -> str:
    """Hash candidate/action content while excluding policy audit trajectories."""

    pool = _mapping(manifest, "pool manifest")
    fingerprint = TaskFingerprint.from_mapping(pool)
    candidates = sorted(
        (
            _mapping(item, "candidate")
            for item in _sequence(pool.get("candidates"), "candidates")
        ),
        key=lambda item: (_candidate_id(item), canonical_json(item)),
    )
    projected_providers: list[dict[str, object]] = []
    for raw_provider in _sequence(pool.get("providers"), "providers"):
        provider = _mapping(raw_provider, "provider")
        batches = sorted(
            (
                _mapping(item, "frozen action batch")
                for item in _sequence(
                    provider.get("frozen_action_batches"),
                    "provider.frozen_action_batches",
                )
            ),
            key=lambda item: canonical_json(item),
        )
        projected_providers.append(
            {
                "provider": _nonempty_string(
                    provider.get("provider"), "provider.provider"
                ),
                "route": _nonempty_string(provider.get("route"), "provider.route"),
                "frozen_action_batches": batches,
            }
        )
    projected_providers.sort(
        key=lambda item: (str(item["provider"]), str(item["route"]))
    )
    projection = {
        "schema": "afts.candidate-dag/v1",
        "task": fingerprint.to_json_dict(),
        "candidates": candidates,
        "providers": projected_providers,
    }
    return canonical_sha256(projection)


@dataclass(frozen=True, slots=True)
class PoolManifestIntegrity:
    schema: str
    pool_id: str
    candidate_dag_id: str
    task_fingerprint: TaskFingerprint


def validate_pool_manifest(
    payload: object,
    *,
    expected_task: TaskFingerprint | None = None,
    allowed_schemas: Iterable[str] = POOL_MANIFEST_SCHEMAS,
) -> PoolManifestIntegrity:
    """Verify pool identity, task binding, references, and oracle-free closure."""

    pool = _mapping(payload, "pool manifest")
    schema = _nonempty_string(pool.get("schema"), "schema")
    if schema not in frozenset(allowed_schemas):
        raise ExperimentSafetyError(f"pool schema is not allowed: {schema}")
    pool_id = _sha256_string(pool.get("pool_id"), "pool_id")
    legacy_body = dict(pool)
    legacy_body.pop("pool_id", None)
    declared_dag_id = legacy_body.pop("candidate_dag_id", None)
    audit_manifest_id = legacy_body.pop("audit_manifest_id", None)
    if pool_id != canonical_sha256(legacy_body):
        raise ExperimentSafetyError("pool_id does not match canonical content")
    if audit_manifest_id is not None and audit_manifest_id != pool_id:
        raise ExperimentSafetyError("audit_manifest_id must equal legacy pool_id")
    fingerprint = TaskFingerprint.from_mapping(pool)
    if expected_task is not None and fingerprint != expected_task:
        raise ExperimentSafetyError("pool task fingerprint does not match summary task")
    if pool.get("oracle_used_during_discovery") is not False:
        raise ExperimentSafetyError("pool discovery must be oracle-free")
    if pool.get("oracle_used_during_pool_closure") is not False:
        raise ExperimentSafetyError("pool closure must be oracle-free")

    candidates = _sequence(pool.get("candidates"), "candidates")
    declared_count = _nonnegative_int(pool.get("candidate_count"), "candidate_count")
    candidate_ids = tuple(_candidate_id(item) for item in candidates)
    if declared_count != len(candidates) or len(set(candidate_ids)) != len(
        candidate_ids
    ):
        raise ExperimentSafetyError("candidate_count or candidate identity is invalid")
    known = set(candidate_ids)
    candidates_by_id = {
        _candidate_id(candidate): _mapping(candidate, "candidate")
        for candidate in candidates
    }
    for raw_provider in _sequence(pool.get("providers"), "providers"):
        provider = _mapping(raw_provider, "provider")
        for raw_batch in _sequence(
            provider.get("frozen_action_batches"),
            "provider.frozen_action_batches",
        ):
            batch = _mapping(raw_batch, "frozen action batch")
            references = _sequence(batch.get("candidate_ids"), "batch.candidate_ids")
            reference_ids = tuple(
                _nonempty_string(item, "batch candidate ID") for item in references
            )
            if any(item not in known for item in reference_ids):
                raise ExperimentSafetyError(
                    "frozen action batch references unknown candidate"
                )
            batch_identity = {
                "operator": _nonempty_string(batch.get("operator"), "batch.operator"),
                "parent_hypothesis_id": batch.get("parent_hypothesis_id"),
                "candidates": [candidates_by_id[item] for item in reference_ids],
                "native_cost": _mapping(batch.get("native_cost"), "batch.native_cost"),
            }
            declared_batch_id = _nonempty_string(
                batch.get("batch_id"), "batch.batch_id"
            )
            if declared_batch_id != canonical_sha256(batch_identity)[:24]:
                raise ExperimentSafetyError(
                    "frozen action batch ID does not match content"
                )

    observed_dag_id = candidate_dag_id(pool)
    if declared_dag_id is not None:
        _sha256_string(declared_dag_id, "candidate_dag_id")
        if declared_dag_id != observed_dag_id:
            raise ExperimentSafetyError(
                "candidate_dag_id does not match candidate/action content"
            )

    return PoolManifestIntegrity(
        schema=schema,
        pool_id=pool_id,
        candidate_dag_id=observed_dag_id,
        task_fingerprint=fingerprint,
    )


@dataclass(frozen=True, slots=True)
class GitSourceProvenance:
    head: str
    dirty: bool
    diff_sha256: str
    tree_sha256: str
    paths: tuple[str, ...] = SOURCE_PROVENANCE_PATHS

    def __post_init__(self) -> None:
        if len(self.head) != 40 or any(
            character not in "0123456789abcdef" for character in self.head
        ):
            raise ExperimentSafetyError(
                "git head must be a lowercase 40-character hash"
            )
        if type(self.dirty) is not bool:
            raise ExperimentSafetyError("git dirty must be boolean")
        _sha256_string(self.diff_sha256, "diff_sha256")
        _sha256_string(self.tree_sha256, "tree_sha256")
        _validate_source_paths(self.paths)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "head": self.head,
            "dirty": self.dirty,
            "diff_sha256": self.diff_sha256,
            "tree_sha256": self.tree_sha256,
            "paths": list(self.paths),
        }


def _git(repo_root: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        check=False,
        capture_output=True,
    )
    if completed.returncode:
        message = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ExperimentSafetyError(f"git provenance command failed: {message}")
    return completed.stdout


def _validate_source_paths(paths: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(paths)
    if not normalized or len(set(normalized)) != len(normalized):
        raise ExperimentSafetyError("source provenance paths must be unique")
    for item in normalized:
        path = Path(item)
        if (
            not isinstance(item, str)
            or not item
            or path.is_absolute()
            or ".." in path.parts
        ):
            raise ExperimentSafetyError(
                "source provenance paths must be safe repository-relative paths"
            )
    return normalized


def _source_tree_sha256(repo_root: Path, paths: Sequence[str]) -> str:
    entries: list[dict[str, str]] = []
    ignored_suffixes = {".pyc", ".pyo"}
    for root_name in paths:
        source_root = repo_root / root_name
        if not source_root.exists():
            continue
        paths = (source_root,) if source_root.is_file() else source_root.rglob("*")
        for path in sorted(paths, key=lambda item: item.as_posix()):
            relative = path.relative_to(repo_root)
            if "__pycache__" in relative.parts or path.suffix in ignored_suffixes:
                continue
            if path.is_symlink():
                entries.append(
                    {
                        "path": relative.as_posix(),
                        "kind": "symlink",
                        "sha256": hashlib.sha256(
                            os.readlink(path).encode("utf-8")
                        ).hexdigest(),
                    }
                )
            elif path.is_file():
                entries.append(
                    {
                        "path": relative.as_posix(),
                        "kind": "file",
                        "sha256": file_sha256(path),
                    }
                )
    return canonical_sha256(entries)


def capture_git_source_provenance(
    repo_root: str | Path,
    *,
    paths: Sequence[str] = SOURCE_PROVENANCE_PATHS,
) -> GitSourceProvenance:
    """Capture HEAD and scoped worktree evidence for src/scripts/config only."""

    root = Path(repo_root).resolve()
    normalized_paths = _validate_source_paths(paths)
    head = _git(root, "rev-parse", "HEAD").decode("ascii").strip()
    status = _git(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        *normalized_paths,
    )
    patch = _git(
        root,
        "diff",
        "--binary",
        "--no-ext-diff",
        "HEAD",
        "--",
        *normalized_paths,
    )
    diff_sha256 = hashlib.sha256(status + b"\0" + patch).hexdigest()
    return GitSourceProvenance(
        head=head,
        dirty=bool(status.strip()),
        diff_sha256=diff_sha256,
        tree_sha256=_source_tree_sha256(root, normalized_paths),
        paths=normalized_paths,
    )


def verify_source_provenance_unchanged(
    start: GitSourceProvenance, end: GitSourceProvenance
) -> None:
    """Require identical source snapshots at experiment start and end."""

    if not isinstance(start, GitSourceProvenance) or not isinstance(
        end, GitSourceProvenance
    ):
        raise TypeError("start and end must be GitSourceProvenance values")
    if start != end:
        changed = tuple(
            field
            for field in ("head", "dirty", "diff_sha256", "tree_sha256", "paths")
            if getattr(start, field) != getattr(end, field)
        )
        raise ExperimentSafetyError(
            "source provenance changed during experiment: " + ", ".join(changed)
        )


def validate_clean_source_binding(payload: object) -> GitSourceProvenance:
    """Require an embedded clean source snapshot bound to ``source_commit``."""

    artifact = _mapping(payload, "source-bound artifact")
    raw = _mapping(artifact.get("source_provenance"), "source_provenance")
    provenance = GitSourceProvenance(
        head=_nonempty_string(raw.get("head"), "source_provenance.head"),
        dirty=raw.get("dirty"),
        diff_sha256=_sha256_string(
            raw.get("diff_sha256"), "source_provenance.diff_sha256"
        ),
        tree_sha256=_sha256_string(
            raw.get("tree_sha256"), "source_provenance.tree_sha256"
        ),
        paths=tuple(
            _nonempty_string(item, "source_provenance.paths item")
            for item in _sequence(raw.get("paths"), "source_provenance.paths")
        ),
    )
    if provenance.dirty:
        raise ExperimentSafetyError("source-bound artifact was produced dirty")
    if artifact.get("source_commit") != provenance.head:
        raise ExperimentSafetyError(
            "source_commit does not match embedded source provenance"
        )
    return provenance
