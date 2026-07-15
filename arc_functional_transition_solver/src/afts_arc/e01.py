"""Leakage-resistant E01a symbolic-coverage evidence pipeline.

The three public entry points intentionally form separate filesystem/process stages:

1. build two sibling dataset bundles (blind observations and oracle labels),
2. generate a frozen candidate pool from the blind bundle alone,
3. evaluate the frozen pool after loading the oracle bundle.

This is a grammar-aligned control experiment.  It is not evidence that the DSL
expresses the distribution of real ARC tasks.
"""

from __future__ import annotations

import hashlib
import json
import time
import tracemalloc
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .blind import BlindTask
from .authority import ARC2_PUBLIC_TRAINING_AUDIT, DATASET_COMMITS, DATASET_ORIGINS
from .candidate import CandidateRecord
from .dsl import (
    DSL_SEMANTICS_VERSION,
    Instruction,
    InvalidCode,
    Program,
    execute_program,
)
from .grid import Grid, as_grid, grid_key, grid_to_lists
from .manifest import (
    DatasetSnapshot,
    RuntimeSourceCapture,
    publish_evidence_bundle,
    runtime_source_fingerprint,
    serialize_json,
    serialize_jsonl,
    utc_now,
    verify_dataset_snapshot_unchanged,
    verify_source_snapshot_zip,
)
from .parse import (
    PARSER_SEMANTICS_VERSION,
    GridParseBundle,
    background_hypotheses,
    parse_grid,
)
from .panel import (
    PANEL_PARSER_SEMANTICS_VERSION,
    PANEL_PARSER_STAGE,
    PanelParseBundle,
    parse_panels,
)
from .panel_actions import (
    D4Step,
    PANEL_LATTICE_PERIODIC_SEMANTICS_VERSION,
    PANEL_LATTICE_PERIODIC_STAGE,
    PANEL_SEQUENCE_D4_SEMANTICS_VERSION,
    PANEL_SEQUENCE_D4_STAGE,
)
from .contact_actions import (
    BBOX_CONTACT_ACTION_SEMANTICS_VERSION,
    BBOX_CONTACT_ACTION_STAGE,
)
from .relation import (
    RELATION_PARSER_SEMANTICS_VERSION,
    RELATION_PARSER_STAGE,
    BBoxContactParseBundle,
    parse_bbox_contacts,
)
from .search import (
    BBOX_CONTACT_PROPOSER_SEMANTICS_VERSION,
    PANEL_LATTICE_PERIODIC_PROPOSER_SEMANTICS_VERSION,
    PANEL_SEQUENCE_D4_PROPOSER_SEMANTICS_VERSION,
    SearchConfig,
    dsl_candidates,
    search_programs,
)
from .shape import SHAPE_PROPOSER_SEMANTICS_VERSION, OutputShapeProposal
from .synthetic import (
    SYNTHETIC_ORACLE_SCHEMA_VERSION,
    SYNTHETIC_SUITE_VERSION,
    generate_suite,
)
from .task import ARCTask

BLIND_MANIFEST_SCHEMA = "afts.e01a-blind-manifest/v1"
ORACLE_MANIFEST_SCHEMA = "afts.e01a-oracle-manifest/v1"
ORACLE_ROW_SCHEMA = SYNTHETIC_ORACLE_SCHEMA_VERSION
PUBLIC_ORACLE_MANIFEST_SCHEMA = "afts.e01a-public-oracle-manifest/v1"
PUBLIC_ORACLE_ROW_SCHEMA = "afts.public-train-oracle/v1"
POOL_MANIFEST_SCHEMA = "afts.e01a-symbolic-pool/v6"
EVAL_MANIFEST_SCHEMA = "afts.e01a-symbolic-eval/v6"
PANEL_PARSE_ROW_SCHEMA = "afts.e01a-panel-parse-row/v1"
RELATION_PARSE_ROW_SCHEMA = "afts.e01a-relation-parse-row/v1"
PARSER_STAGE = "M02a_connected_components_only"
OUTPUT_CUTOFFS: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)
PROGRAM_CUTOFFS: tuple[int, ...] = (1, 8, 32, 128, 512, 2048)
_POOL_FORBIDDEN_KEYS = frozenset(
    {
        "family",
        "stratum",
        "seed",
        "generator_program",
        "ground_truth_program",
        "query_outputs",
        "test_outputs",
        "oracle_output",
        "oracle_exact",
    }
)


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _content_id(label: str, payload: object) -> str:
    return hashlib.sha256(label.encode("ascii") + b"\0" + _canonical_bytes(payload)).hexdigest()


def _strict_json(content: bytes, *, label: str) -> Any:
    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key in {label}: {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant in {label}: {value}")

    try:
        return json.loads(
            content.decode("utf-8"),
            object_pairs_hook=reject_duplicate,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON in {label}: {exc}") from exc


def _strict_jsonl(content: bytes, *, label: str) -> list[dict[str, Any]]:
    if content and not content.endswith(b"\n"):
        raise ValueError(f"JSONL artifact must end with a newline: {label}")
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(content.splitlines()):
        if not line:
            raise ValueError(f"blank JSONL row in {label} at line {index + 1}")
        row = _strict_json(line, label=f"{label}:{index + 1}")
        if not isinstance(row, dict):
            raise TypeError(f"JSONL row must be an object: {label}:{index + 1}")
        rows.append(row)
    return rows


def _require_fields(payload: object, expected: set[str], *, label: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must be a JSON object")
    if set(payload) != expected:
        raise ValueError(
            f"{label} fields must be {sorted(expected)}, found {sorted(payload)}"
        )
    return payload


@dataclass(frozen=True, slots=True)
class BundleSnapshot:
    root: Path
    artifact_manifest: dict[str, Any]
    artifact_manifest_sha256: str
    artifacts: dict[str, bytes]


def _require_nonoverlapping_paths(**paths: str | Path) -> dict[str, Path]:
    resolved = {
        name: Path(value).expanduser().resolve() for name, value in paths.items()
    }
    items = list(resolved.items())
    for index, (first_name, first) in enumerate(items):
        for second_name, second in items[index + 1 :]:
            if first == second or first in second.parents or second in first.parents:
                raise ValueError(
                    f"evidence paths must not overlap: {first_name}={first}, "
                    f"{second_name}={second}"
                )
    return resolved


def snapshot_bundle(path: str | Path) -> BundleSnapshot:
    """Read and verify a closed-world bundle once, avoiding TOCTOU parsing."""

    supplied = Path(path).expanduser().absolute()
    if supplied.is_symlink():
        raise ValueError(f"bundle root may not be a symlink: {supplied}")
    root = supplied.resolve()
    if not root.is_dir():
        raise ValueError(f"bundle must be a real directory: {root}")
    manifest_path = root / "artifact_manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError(f"bundle artifact manifest missing or symlinked: {manifest_path}")
    manifest_bytes = manifest_path.read_bytes()
    manifest = _require_fields(
        _strict_json(manifest_bytes, label=str(manifest_path)),
        {"schema_version", "bundle_status", "run_id", "artifacts"},
        label="artifact manifest",
    )
    if manifest["schema_version"] != 1 or manifest["bundle_status"] != "complete":
        raise ValueError("unsupported or incomplete artifact manifest")
    if not isinstance(manifest["run_id"], str) or not manifest["run_id"]:
        raise ValueError("artifact manifest run_id must be non-empty")
    metadata = manifest["artifacts"]
    if not isinstance(metadata, dict) or not metadata:
        raise ValueError("artifact manifest must declare artifacts")

    declared = set(metadata)
    actual: set[str] = set()
    for entry in root.rglob("*"):
        if entry.is_symlink():
            raise ValueError(f"bundle may not contain symlinks: {entry}")
        if entry.is_file():
            actual.add(entry.relative_to(root).as_posix())
        elif not entry.is_dir():
            raise ValueError(f"unsupported bundle entry: {entry}")
    expected_files = declared | {"artifact_manifest.json"}
    if actual != expected_files:
        raise ValueError(
            f"bundle is not closed-world; expected {sorted(expected_files)}, found {sorted(actual)}"
        )

    captured: dict[str, bytes] = {}
    for name in sorted(declared):
        if not isinstance(name, str) or name.startswith("/") or ".." in Path(name).parts:
            raise ValueError(f"unsafe artifact name in manifest: {name!r}")
        record = _require_fields(
            metadata[name], {"sha256", "bytes", "rows"}, label=f"metadata[{name}]"
        )
        destination = (root / Path(name)).resolve()
        try:
            destination.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"artifact escapes bundle root: {name}") from exc
        if not destination.is_file() or destination.is_symlink():
            raise ValueError(f"artifact missing or symlinked: {name}")
        content = destination.read_bytes()
        if record["sha256"] != _sha256(content) or record["bytes"] != len(content):
            raise ValueError(f"artifact digest or byte count mismatch: {name}")
        expected_rows = content.count(b"\n") if name.endswith(".jsonl") else None
        if record["rows"] != expected_rows:
            raise ValueError(f"artifact row count mismatch: {name}")
        captured[name] = content
    return BundleSnapshot(
        root=root,
        artifact_manifest=manifest,
        artifact_manifest_sha256=_sha256(manifest_bytes),
        artifacts=captured,
    )


def _require_artifacts(snapshot: BundleSnapshot, expected: set[str]) -> None:
    found = set(snapshot.artifacts)
    if found != expected:
        raise ValueError(
            f"bundle artifacts must be {sorted(expected)}, found {sorted(found)}"
        )


def _runtime_record(
    *, source_capture: RuntimeSourceCapture, test_source_sha256: str | None
) -> dict[str, object]:
    return {
        "runtime_source_fingerprint_sha256": source_capture.fingerprint_sha256,
        "test_source_fingerprint_sha256": test_source_sha256,
    }


def _oracle_row(case: Any) -> dict[str, object]:
    return case.oracle_json_dict()


def build_synthetic_dataset_bundles(
    *,
    blind_output_dir: str | Path,
    oracle_output_dir: str | Path,
    seed: int,
    tasks_per_family: int,
    source_capture: RuntimeSourceCapture,
    test_source_sha256: str | None,
    source_loader: Mapping[str, object],
    command: list[str],
) -> dict[str, object]:
    blind_target = Path(blind_output_dir).expanduser().resolve()
    oracle_target = Path(oracle_output_dir).expanduser().resolve()
    _require_nonoverlapping_paths(blind=blind_target, oracle=oracle_target)
    if blind_target.parent != oracle_target.parent:
        raise ValueError("blind and oracle dataset bundles must be siblings")
    if blind_target.exists() or oracle_target.exists():
        raise FileExistsError("refusing to overwrite a dataset evidence directory")
    started_at = utc_now()
    cases = generate_suite(seed=seed, tasks_per_family=tasks_per_family)
    by_id = sorted(cases, key=lambda case: case.blind_task.task_id)
    if len({case.blind_task.task_id for case in by_id}) != len(by_id):
        raise RuntimeError("synthetic suite contains duplicate blind task identities")

    blind_rows = [case.blind_task.to_json_dict() for case in by_id]
    oracle_rows: list[dict[str, object]] = []
    for case in by_id:
        for pair in (*case.task.train, *case.task.test):
            outcome = execute_program(case.generator_program, pair.input)
            if outcome.invalid_code is InvalidCode.INTERNAL_ERROR:
                raise RuntimeError("generator program encountered a DSL internal error")
            if not outcome.ok or outcome.output != pair.output:
                raise RuntimeError("generator program failed its own positive control")
        oracle_rows.append(_oracle_row(case))

    blind_bytes = serialize_jsonl(blind_rows)
    case_set_id = _content_id(
        "afts-e01a-case-set/v1",
        {
            "dataset_descriptor": SYNTHETIC_SUITE_VERSION,
            "blind_tasks_sha256": _sha256(blind_bytes),
            "task_count": len(blind_rows),
        },
    )
    oracle_bytes = serialize_jsonl(oracle_rows)
    oracle_set_id = _content_id(
        "afts-e01a-oracle-set/v1",
        {
            "case_set_id": case_set_id,
            "oracle_sha256": _sha256(oracle_bytes),
        },
    )
    completed_at = utc_now()
    runtime = _runtime_record(
        source_capture=source_capture, test_source_sha256=test_source_sha256
    )
    blind_manifest = {
        "schema": BLIND_MANIFEST_SCHEMA,
        "evidence_status": "generator_known_symbolic_control_blind_observations",
        "case_set_id": case_set_id,
        "dataset_kind": "synthetic_control",
        "dataset_descriptor": SYNTHETIC_SUITE_VERSION,
        "task_count": len(blind_rows),
        "blind_tasks_sha256": _sha256(blind_bytes),
        "started_at": started_at,
        "completed_at": completed_at,
        "source_loader": dict(source_loader),
        "command": command,
        **runtime,
        "claim_boundary": (
            "Grammar-aligned synthetic control only; it does not estimate real ARC "
            "DSL expressivity or compositional OOD generalization."
        ),
    }
    oracle_manifest = {
        "schema": ORACLE_MANIFEST_SCHEMA,
        "evidence_status": "generator_known_symbolic_control_oracle_labels",
        "case_set_id": case_set_id,
        "oracle_set_id": oracle_set_id,
        "suite_version": SYNTHETIC_SUITE_VERSION,
        "task_count": len(oracle_rows),
        "oracle_sha256": _sha256(oracle_bytes),
        "generator_config": {"seed": seed, "tasks_per_family": tasks_per_family},
        "started_at": started_at,
        "completed_at": completed_at,
        "source_loader": dict(source_loader),
        "command": command,
        **runtime,
    }
    if runtime_source_fingerprint() != source_capture.fingerprint_sha256:
        raise RuntimeError("executing source changed during synthetic dataset construction")
    blind_artifact_manifest = publish_evidence_bundle(
        blind_target,
        run_id=case_set_id[:20],
        artifacts={
            "blind_manifest.json": serialize_json(blind_manifest),
            "blind_tasks.jsonl": blind_bytes,
            "source_snapshot.zip": source_capture.snapshot_zip,
        },
    )
    if runtime_source_fingerprint() != source_capture.fingerprint_sha256:
        raise RuntimeError("executing source changed before oracle bundle publication")
    oracle_artifact_manifest = publish_evidence_bundle(
        oracle_target,
        run_id=oracle_set_id[:20],
        artifacts={
            "oracle_manifest.json": serialize_json(oracle_manifest),
            "oracle.jsonl": oracle_bytes,
            "source_snapshot.zip": source_capture.snapshot_zip,
        },
    )
    return {
        "case_set_id": case_set_id,
        "oracle_set_id": oracle_set_id,
        "task_count": len(blind_rows),
        "blind_output_dir": str(blind_target),
        "oracle_output_dir": str(oracle_target),
        "blind_artifact_manifest_run_id": blind_artifact_manifest["run_id"],
        "oracle_artifact_manifest_run_id": oracle_artifact_manifest["run_id"],
        "evidence_status": "published_sibling_blind_and_oracle_control_bundles",
    }


def _validate_pinned_arc2_training_audit(payload: object) -> dict[str, Any]:
    audit = _require_fields(
        payload,
        {
            "dataset_name",
            "split_name",
            "task_dir",
            "repository_commit",
            "repository_origin",
            "repository_clean",
            "available_file_count",
            "file_count",
            "selection_policy",
            "selected_task_ids_sha256",
            "file_manifest_sha256",
            "total_train_pairs",
            "total_test_pairs",
            "multi_test_task_count",
            "test_outputs_present",
            "validated",
        },
        label="ARC-AGI-2 public-training audit",
    )
    normalize_origin = lambda value: value.rstrip("/").removesuffix(".git").lower()
    if (
        audit["dataset_name"] != "ARC-AGI-2"
        or audit["split_name"] != "public_training"
        or audit["repository_commit"] != DATASET_COMMITS["ARC-AGI-2"]
        or not isinstance(audit["repository_origin"], str)
        or normalize_origin(audit["repository_origin"])
        != normalize_origin(DATASET_ORIGINS["ARC-AGI-2"])
        or audit["repository_clean"] is not True
        or audit["validated"] is not True
        or any(audit[key] != value for key, value in ARC2_PUBLIC_TRAINING_AUDIT.items())
    ):
        raise ValueError("audit is not the complete clean pinned ARC-AGI-2 training tree")
    return audit


def build_public_training_smoke_bundles(
    *,
    dataset_snapshot: DatasetSnapshot,
    blind_output_dir: str | Path,
    oracle_output_dir: str | Path,
    smoke_count: int,
    selection_salt: str,
    source_capture: RuntimeSourceCapture,
    test_source_sha256: str | None,
    source_loader: Mapping[str, object],
    command: list[str],
    holdout_count: int = 150,
    validation_count: int = 150,
) -> dict[str, object]:
    """Materialize a deterministic ARC-AGI-2 public-training development smoke.

    The first fixed hash buckets are reserved for future holdout and validation use;
    the repeatedly runnable smoke is selected only from the remaining development
    bucket.  This v1 audit quarantines exact semantic duplicates but deliberately
    does not claim D4/color-normalized near-duplicate detection yet.
    """

    blind_target = Path(blind_output_dir).expanduser().resolve()
    oracle_target = Path(oracle_output_dir).expanduser().resolve()
    _require_nonoverlapping_paths(blind=blind_target, oracle=oracle_target)
    if blind_target.parent != oracle_target.parent:
        raise ValueError("blind and oracle dataset bundles must be siblings")
    if blind_target.exists() or oracle_target.exists():
        raise FileExistsError("refusing to overwrite a dataset evidence directory")
    if type(smoke_count) is not int or smoke_count <= 0:
        raise ValueError("smoke_count must be positive")
    if type(holdout_count) is not int or holdout_count < 0:
        raise ValueError("holdout_count must be non-negative")
    if type(validation_count) is not int or validation_count < 0:
        raise ValueError("validation_count must be non-negative")
    if not isinstance(selection_salt, str) or not selection_salt:
        raise ValueError("selection_salt must be a non-empty string")
    dataset_audit = asdict(dataset_snapshot.audit)
    tasks = dataset_snapshot.tasks
    _validate_pinned_arc2_training_audit(dataset_audit)
    repo_state = dataset_snapshot.repo_state
    normalize_origin = lambda value: value.rstrip("/").removesuffix(".git").lower()
    if (
        repo_state is None
        or repo_state.commit != DATASET_COMMITS["ARC-AGI-2"]
        or normalize_origin(repo_state.origin)
        != normalize_origin(DATASET_ORIGINS["ARC-AGI-2"])
        or not repo_state.clean
        or dataset_snapshot.max_tasks is not None
        or dataset_audit.get("file_count") != len(tasks)
        or Path(str(dataset_audit.get("task_dir"))).resolve()
        != (Path(repo_state.path) / "data" / "training").resolve()
    ):
        raise ValueError(
            "public smoke requires the complete clean pinned ARC-AGI-2 training tree"
        )
    if not tasks:
        raise ValueError("public development smoke requires tasks")

    observed: list[tuple[BlindTask, ARCTask]] = []
    for task in tasks:
        if any(pair.output is None for pair in task.test):
            raise ValueError("public training smoke requires query outputs in oracle source")
        observed.append((BlindTask.from_task(task), task))
    semantic_groups: defaultdict[str, list[tuple[BlindTask, ARCTask]]] = defaultdict(list)
    for item in observed:
        semantic_groups[item[0].semantic_fingerprint_sha256].append(item)
    quarantined = [
        item for group in semantic_groups.values() if len(group) > 1 for item in group
    ]
    eligible = [group[0] for group in semantic_groups.values() if len(group) == 1]

    split_salt = "afts-e01-public-train-split-v1"

    def assignment_key(item: tuple[BlindTask, ARCTask]) -> str:
        return _content_id(
            split_salt, {"blind_content_sha256": item[0].blind_content_sha256}
        )

    assigned = sorted(eligible, key=lambda item: (assignment_key(item), item[0].task_id))
    required = holdout_count + validation_count + smoke_count
    if len(assigned) < required:
        raise ValueError(
            f"not enough singleton tasks for reserved splits and smoke: {len(assigned)} < {required}"
        )
    development = assigned[holdout_count + validation_count :]

    def smoke_key(item: tuple[BlindTask, ARCTask]) -> str:
        return _content_id(
            selection_salt, {"blind_content_sha256": item[0].blind_content_sha256}
        )

    selected = sorted(
        sorted(development, key=lambda item: (smoke_key(item), item[0].task_id))[
            :smoke_count
        ],
        key=lambda item: item[0].task_id,
    )
    blind_rows = [blind.to_json_dict() for blind, _ in selected]
    blind_bytes = serialize_jsonl(blind_rows)
    dataset_descriptor = (
        f"ARC-AGI-2@{dataset_audit.get('repository_commit')}/public_training/"
        "development-smoke-v1"
    )
    case_set_id = _content_id(
        "afts-e01a-case-set/v1",
        {
            "dataset_descriptor": dataset_descriptor,
            "blind_tasks_sha256": _sha256(blind_bytes),
            "task_count": len(blind_rows),
        },
    )
    oracle_rows = [
        {
            "schema": PUBLIC_ORACLE_ROW_SCHEMA,
            "task_id": blind.task_id,
            "source_task_id": task.task_id,
            "source_sha256": task.source_sha256,
            "test": [
                {
                    "test_index": index,
                    "output": grid_to_lists(pair.output),
                    "output_key": grid_key(pair.output),
                }
                for index, pair in enumerate(task.test)
            ],
        }
        for blind, task in selected
    ]
    oracle_bytes = serialize_jsonl(oracle_rows)
    oracle_set_id = _content_id(
        "afts-e01a-public-oracle-set/v1",
        {"case_set_id": case_set_id, "oracle_sha256": _sha256(oracle_bytes)},
    )
    started_at = utc_now()
    completed_at = utc_now()
    runtime = _runtime_record(
        source_capture=source_capture, test_source_sha256=test_source_sha256
    )
    blind_manifest = {
        "schema": BLIND_MANIFEST_SCHEMA,
        "evidence_status": "arc_agi_2_public_training_development_smoke_blind",
        "case_set_id": case_set_id,
        "dataset_kind": "arc_public_training_development_smoke",
        "dataset_descriptor": dataset_descriptor,
        "task_count": len(blind_rows),
        "blind_tasks_sha256": _sha256(blind_bytes),
        "started_at": started_at,
        "completed_at": completed_at,
        "source_loader": dict(source_loader),
        "command": command,
        **runtime,
        "claim_boundary": (
            "Repeated development smoke on pinned ARC-AGI-2 public training; not a "
            "holdout, public-evaluation, or release result."
        ),
    }
    oracle_manifest = {
        "schema": PUBLIC_ORACLE_MANIFEST_SCHEMA,
        "evidence_status": "arc_agi_2_public_training_development_smoke_oracle",
        "case_set_id": case_set_id,
        "oracle_set_id": oracle_set_id,
        "dataset_descriptor": dataset_descriptor,
        "task_count": len(oracle_rows),
        "oracle_sha256": _sha256(oracle_bytes),
        "dataset_audit": dict(dataset_audit),
        "selection_audit": {
            "policy": "semantic-singletons_then_hash_reserved_splits_v1",
            "split_salt": split_salt,
            "smoke_salt": selection_salt,
            "available_tasks": len(tasks),
            "exact_semantic_quarantine_tasks": len(quarantined),
            "singleton_tasks": len(eligible),
            "reserved_holdout_tasks": holdout_count,
            "reserved_validation_tasks": validation_count,
            "development_tasks": len(development),
            "smoke_tasks": len(selected),
            "near_duplicate_limit": "exact_semantic_only_pending_d4_color_audit",
        },
        "started_at": started_at,
        "completed_at": completed_at,
        "source_loader": dict(source_loader),
        "command": command,
        **runtime,
    }
    if runtime_source_fingerprint() != source_capture.fingerprint_sha256:
        raise RuntimeError("executing source changed during public smoke construction")
    verify_dataset_snapshot_unchanged(dataset_snapshot)
    publish_evidence_bundle(
        blind_target,
        run_id=case_set_id[:20],
        artifacts={
            "blind_manifest.json": serialize_json(blind_manifest),
            "blind_tasks.jsonl": blind_bytes,
            "source_snapshot.zip": source_capture.snapshot_zip,
        },
    )
    if runtime_source_fingerprint() != source_capture.fingerprint_sha256:
        raise RuntimeError("executing source changed before public oracle publication")
    verify_dataset_snapshot_unchanged(dataset_snapshot)
    publish_evidence_bundle(
        oracle_target,
        run_id=oracle_set_id[:20],
        artifacts={
            "oracle_manifest.json": serialize_json(oracle_manifest),
            "oracle.jsonl": oracle_bytes,
            "source_snapshot.zip": source_capture.snapshot_zip,
        },
    )
    return {
        "evidence_status": "published_public_training_development_smoke_bundles",
        "case_set_id": case_set_id,
        "oracle_set_id": oracle_set_id,
        "task_count": len(selected),
        "quarantined_exact_semantic_tasks": len(quarantined),
        "blind_output_dir": str(blind_target),
        "oracle_output_dir": str(oracle_target),
    }


def _load_blind_dataset(snapshot: BundleSnapshot) -> tuple[dict[str, Any], tuple[BlindTask, ...]]:
    _require_artifacts(
        snapshot,
        {"blind_manifest.json", "blind_tasks.jsonl", "source_snapshot.zip"},
    )
    manifest = _require_fields(
        _strict_json(snapshot.artifacts["blind_manifest.json"], label="blind_manifest.json"),
        {
            "schema",
            "evidence_status",
            "case_set_id",
            "dataset_kind",
            "dataset_descriptor",
            "task_count",
            "blind_tasks_sha256",
            "started_at",
            "completed_at",
            "source_loader",
            "command",
            "runtime_source_fingerprint_sha256",
            "test_source_fingerprint_sha256",
            "claim_boundary",
        },
        label="blind manifest",
    )
    if manifest["schema"] != BLIND_MANIFEST_SCHEMA:
        raise ValueError("unsupported blind manifest schema")
    verify_source_snapshot_zip(
        snapshot.artifacts["source_snapshot.zip"],
        expected_fingerprint_sha256=manifest["runtime_source_fingerprint_sha256"],
    )
    blind_bytes = snapshot.artifacts["blind_tasks.jsonl"]
    if manifest["blind_tasks_sha256"] != _sha256(blind_bytes):
        raise ValueError("blind manifest does not bind blind_tasks.jsonl")
    tasks = tuple(
        BlindTask.from_json_dict(row)
        for row in _strict_jsonl(blind_bytes, label="blind_tasks.jsonl")
    )
    if manifest["task_count"] != len(tasks):
        raise ValueError("blind manifest task count mismatch")
    ids = tuple(task.task_id for task in tasks)
    if ids != tuple(sorted(ids)) or len(ids) != len(set(ids)):
        raise ValueError("blind tasks must be uniquely sorted by content-derived task_id")
    expected_case_set = _content_id(
        "afts-e01a-case-set/v1",
        {
            "dataset_descriptor": manifest["dataset_descriptor"],
            "blind_tasks_sha256": _sha256(blind_bytes),
            "task_count": len(tasks),
        },
    )
    if manifest["case_set_id"] != expected_case_set:
        raise ValueError("blind case_set_id does not match observable content")
    return manifest, tasks


def _parse_bundle_row(
    *, task: BlindTask, pair_role: str, pair_index: int, bundle: GridParseBundle
) -> dict[str, object]:
    hypotheses = [hypothesis.to_json_dict() for hypothesis in bundle.hypotheses]
    parse_content = {
        "parser_semantics_version": PARSER_SEMANTICS_VERSION,
        "parser_stage": PARSER_STAGE,
        "grid_key": bundle.grid_key,
        "background_candidates": list(bundle.background_candidates),
        "hypotheses": hypotheses,
    }
    return {
        "task_id": task.task_id,
        "blind_content_sha256": task.blind_content_sha256,
        "pair_role": pair_role,
        "pair_index": pair_index,
        "parse_bundle_id": _content_id("afts-parse-bundle/v1", parse_content),
        **parse_content,
    }


def _panel_parse_bundle_row(
    *, task: BlindTask, pair_role: str, pair_index: int, bundle: PanelParseBundle
) -> dict[str, object]:
    hypotheses = [hypothesis.to_json_dict() for hypothesis in bundle.hypotheses]
    panel_content = {
        "schema": PANEL_PARSE_ROW_SCHEMA,
        "panel_parser_semantics_version": PANEL_PARSER_SEMANTICS_VERSION,
        "panel_parser_stage": PANEL_PARSER_STAGE,
        "grid_key": bundle.grid_key,
        "hypotheses": hypotheses,
    }
    return {
        "task_id": task.task_id,
        "blind_content_sha256": task.blind_content_sha256,
        "pair_role": pair_role,
        "pair_index": pair_index,
        "panel_parse_bundle_id": _content_id(
            "afts-panel-parse-bundle/v1", panel_content
        ),
        **panel_content,
    }


def _relation_parse_bundle_row(
    *, task: BlindTask, pair_role: str, pair_index: int, grid: Grid
) -> dict[str, object]:
    """Serialize every blind, proposer-visible bbox-contact parse for one grid."""

    backgrounds = tuple(
        background
        for background in background_hypotheses(
            grid, include_none=True, max_backgrounds=3
        )
        if background is not None
    )
    bundles = [
        parse_bbox_contacts(grid, background=background).to_json_dict()
        for background in backgrounds
    ]
    relation_content = {
        "schema": RELATION_PARSE_ROW_SCHEMA,
        "relation_parser_semantics_version": RELATION_PARSER_SEMANTICS_VERSION,
        "relation_parser_stage": RELATION_PARSER_STAGE,
        "grid_key": grid_key(grid),
        "background_candidates": list(backgrounds),
        "background_bundles": bundles,
    }
    return {
        "task_id": task.task_id,
        "blind_content_sha256": task.blind_content_sha256,
        "pair_role": pair_role,
        "pair_index": pair_index,
        "relation_parse_bundle_id": _content_id(
            "afts-relation-parse-bundle/v1", relation_content
        ),
        **relation_content,
    }


def _outcome_json(test_index: int, outcome: Any) -> dict[str, object]:
    return {
        "test_index": test_index,
        "status": outcome.status,
        "output": grid_to_lists(outcome.output) if outcome.output is not None else None,
        "output_key": grid_key(outcome.output) if outcome.output is not None else None,
        "invalid_code": outcome.invalid_code.value if outcome.invalid_code else None,
        "failing_instruction_index": outcome.failing_instruction_index,
        "semantic_trace": [list(item) for item in outcome.semantic_trace],
    }


def _assert_no_forbidden_pool_keys(value: object, *, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in _POOL_FORBIDDEN_KEYS:
                raise RuntimeError(f"oracle-only key leaked into pool at {path}.{key}")
            _assert_no_forbidden_pool_keys(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_no_forbidden_pool_keys(child, path=f"{path}[{index}]")


def _pool_behavior_hashes(
    *,
    task_bytes: bytes,
    parse_bytes: bytes,
    panel_parse_bytes: bytes,
    relation_parse_bytes: bytes,
    program_bytes: bytes,
    candidate_bytes: bytes,
    search_rows: list[dict[str, object]],
) -> dict[str, str]:
    return {
        "tasks": _sha256(task_bytes),
        "parses": _sha256(parse_bytes),
        "panel_parses": _sha256(panel_parse_bytes),
        "relation_parses": _sha256(relation_parse_bytes),
        "programs": _sha256(program_bytes),
        "candidates": _sha256(candidate_bytes),
        "search_semantics": _content_id(
            "afts-e01a-search-semantics/v6",
            [
                {
                    key: value
                    for key, value in row.items()
                    if key
                    not in {
                        "wall_time_ns",
                        "process_time_ns",
                        "tracemalloc_peak_bytes",
                    }
                }
                for row in search_rows
            ],
        ),
    }


def _panel_sequence_d4_options_after_cap(
    search_rows: Iterable[Mapping[str, Any]],
) -> int:
    total = 0
    for row in search_rows:
        option_keys = {
            json.dumps(item, sort_keys=True, separators=(",", ":"))
            for item in row["instruction_options"]
        }
        total += sum(
            json.dumps(item, sort_keys=True, separators=(",", ":"))
            in option_keys
            for item in row["panel_sequence_d4_instruction_proposals"]
        )
    return total


def _panel_lattice_periodic_options_after_cap(
    search_rows: Iterable[Mapping[str, Any]],
) -> int:
    total = 0
    for row in search_rows:
        option_keys = {
            json.dumps(item, sort_keys=True, separators=(",", ":"))
            for item in row["instruction_options"]
        }
        total += sum(
            json.dumps(item, sort_keys=True, separators=(",", ":"))
            in option_keys
            for item in row["panel_lattice_periodic_instruction_proposals"]
        )
    return total


def _bbox_contact_options_after_cap(
    search_rows: Iterable[Mapping[str, Any]],
) -> int:
    total = 0
    for row in search_rows:
        option_keys = {
            json.dumps(item, sort_keys=True, separators=(",", ":"))
            for item in row["instruction_options"]
        }
        total += sum(
            json.dumps(item, sort_keys=True, separators=(",", ":"))
            in option_keys
            for item in row["bbox_contact_instruction_proposals"]
        )
    return total


def _pool_summary_payload(
    *,
    blind_manifest: Mapping[str, Any],
    pool_spec_id: str,
    pool_content_id: str,
    task_rows: list[dict[str, Any]],
    parse_rows: list[dict[str, Any]],
    panel_parse_rows: list[dict[str, Any]],
    relation_parse_rows: list[dict[str, Any]],
    program_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    search_rows: list[dict[str, Any]],
) -> dict[str, object]:
    dataset_kind = blind_manifest["dataset_kind"]
    return {
        "schema_version": 6,
        "evidence_status": "blind_pool_counts_only_no_coverage_claim",
        "dataset_kind": dataset_kind,
        "dataset_descriptor": blind_manifest["dataset_descriptor"],
        "pool_spec_id": pool_spec_id,
        "pool_content_id": pool_content_id,
        "case_set_id": blind_manifest["case_set_id"],
        "task_count": len(task_rows),
        "shape_proposals": sum(
            len(row["shape_proposals"]) for row in search_rows
        ),
        "parse_records": len(parse_rows),
        "panel_parse_records": len(panel_parse_rows),
        "panel_hypotheses": sum(
            len(row["hypotheses"]) for row in panel_parse_rows
        ),
        "relation_parse_records": len(relation_parse_rows),
        "relation_parse_bundles": sum(
            len(row["background_bundles"]) for row in relation_parse_rows
        ),
        "relation_hypotheses": sum(
            len(bundle["hypotheses"])
            for row in relation_parse_rows
            for bundle in row["background_bundles"]
        ),
        "panel_instruction_proposals": sum(
            len(row["panel_instruction_proposals"]) for row in search_rows
        ),
        "panel_sequence_d4_instruction_proposals": sum(
            len(row["panel_sequence_d4_instruction_proposals"])
            for row in search_rows
        ),
        "panel_sequence_d4_options_after_cap": (
            _panel_sequence_d4_options_after_cap(search_rows)
        ),
        "panel_sequence_d4_trial_count": sum(
            row["panel_sequence_d4_trial_count"] for row in search_rows
        ),
        "panel_sequence_d4_demo_execution_count": sum(
            row["panel_sequence_d4_demo_execution_count"]
            for row in search_rows
        ),
        "panel_lattice_periodic_period_bounds": sum(
            len(row["panel_lattice_periodic_period_bounds"])
            for row in search_rows
        ),
        "panel_lattice_periodic_instruction_proposals": sum(
            len(row["panel_lattice_periodic_instruction_proposals"])
            for row in search_rows
        ),
        "panel_lattice_periodic_options_after_cap": (
            _panel_lattice_periodic_options_after_cap(search_rows)
        ),
        "panel_lattice_periodic_structural_check_count": sum(
            row["panel_lattice_periodic_structural_check_count"]
            for row in search_rows
        ),
        "panel_lattice_periodic_trial_count": sum(
            row["panel_lattice_periodic_trial_count"]
            for row in search_rows
        ),
        "panel_lattice_periodic_demo_execution_count": sum(
            row["panel_lattice_periodic_demo_execution_count"]
            for row in search_rows
        ),
        "bbox_contact_bounds": sum(
            len(row["bbox_contact_bounds"]) for row in search_rows
        ),
        "bbox_contact_instruction_proposals": sum(
            len(row["bbox_contact_instruction_proposals"])
            for row in search_rows
        ),
        "bbox_contact_options_after_cap": _bbox_contact_options_after_cap(
            search_rows
        ),
        "bbox_contact_structural_check_count": sum(
            row["bbox_contact_structural_check_count"] for row in search_rows
        ),
        "bbox_contact_anchor_candidate_count": sum(
            row["bbox_contact_anchor_candidate_count"] for row in search_rows
        ),
        "bbox_contact_relation_check_count": sum(
            row["bbox_contact_relation_check_count"] for row in search_rows
        ),
        "bbox_contact_admissible_binding_count": sum(
            row["bbox_contact_admissible_binding_count"] for row in search_rows
        ),
        "bbox_contact_action_trial_count": sum(
            row["bbox_contact_action_trial_count"] for row in search_rows
        ),
        "bbox_contact_demo_execution_count": sum(
            row["bbox_contact_demo_execution_count"] for row in search_rows
        ),
        "instruction_option_count_pre_cap": sum(
            row["instruction_option_count_pre_cap"] for row in search_rows
        ),
        "instruction_option_count_post_cap": sum(
            row["instruction_option_count_post_cap"] for row in search_rows
        ),
        "instruction_option_truncation_count": sum(
            row["instruction_option_truncation_count"] for row in search_rows
        ),
        "retained_exact_programs": len(program_rows),
        "candidate_records": len(candidate_rows),
        "unique_candidate_outputs": len(
            {
                (
                    row["candidate"]["task_id"],
                    row["candidate"]["test_index"],
                    row["candidate"]["output_key"],
                )
                for row in candidate_rows
            }
        ),
        "expansions": sum(int(row["expansions"]) for row in search_rows),
        "program_execution_count": sum(
            int(row["program_execution_count"]) for row in search_rows
        ),
        "semantic_duplicates": sum(
            int(row["semantic_duplicates"]) for row in search_rows
        ),
        "claim_boundary": (
            "Counts describe an oracle-unread grammar-aligned synthetic control pool; "
            "no query coverage or real-ARC claim is computed in this stage."
            if dataset_kind == "synthetic_control"
            else "Counts describe an oracle-unread ARC public-training development "
            "pool; no query coverage, holdout, or public-evaluation claim is computed "
            "in this stage."
        ),
    }


def _representation_funnel_counts(
    pool_rows: Mapping[str, list[dict[str, Any]]],
) -> dict[str, int]:
    return {
        "parse_records": len(pool_rows["parses.jsonl"]),
        "panel_parse_records": len(pool_rows["panel_parses.jsonl"]),
        "panel_hypotheses": sum(
            len(row["hypotheses"]) for row in pool_rows["panel_parses.jsonl"]
        ),
        "relation_parse_records": len(pool_rows["relation_parses.jsonl"]),
        "relation_parse_bundles": sum(
            len(row["background_bundles"])
            for row in pool_rows["relation_parses.jsonl"]
        ),
        "relation_hypotheses": sum(
            len(bundle["hypotheses"])
            for row in pool_rows["relation_parses.jsonl"]
            for bundle in row["background_bundles"]
        ),
        "panel_instruction_proposals": sum(
            len(row["panel_instruction_proposals"])
            for row in pool_rows["search.jsonl"]
        ),
        "panel_sequence_d4_instruction_proposals": sum(
            len(row["panel_sequence_d4_instruction_proposals"])
            for row in pool_rows["search.jsonl"]
        ),
        "panel_sequence_d4_options_after_cap": (
            _panel_sequence_d4_options_after_cap(pool_rows["search.jsonl"])
        ),
        "panel_sequence_d4_trial_count": sum(
            row["panel_sequence_d4_trial_count"]
            for row in pool_rows["search.jsonl"]
        ),
        "panel_sequence_d4_demo_execution_count": sum(
            row["panel_sequence_d4_demo_execution_count"]
            for row in pool_rows["search.jsonl"]
        ),
        "panel_lattice_periodic_period_bounds": sum(
            len(row["panel_lattice_periodic_period_bounds"])
            for row in pool_rows["search.jsonl"]
        ),
        "panel_lattice_periodic_instruction_proposals": sum(
            len(row["panel_lattice_periodic_instruction_proposals"])
            for row in pool_rows["search.jsonl"]
        ),
        "panel_lattice_periodic_options_after_cap": (
            _panel_lattice_periodic_options_after_cap(
                pool_rows["search.jsonl"]
            )
        ),
        "panel_lattice_periodic_structural_check_count": sum(
            row["panel_lattice_periodic_structural_check_count"]
            for row in pool_rows["search.jsonl"]
        ),
        "panel_lattice_periodic_trial_count": sum(
            row["panel_lattice_periodic_trial_count"]
            for row in pool_rows["search.jsonl"]
        ),
        "panel_lattice_periodic_demo_execution_count": sum(
            row["panel_lattice_periodic_demo_execution_count"]
            for row in pool_rows["search.jsonl"]
        ),
        "bbox_contact_bounds": sum(
            len(row["bbox_contact_bounds"])
            for row in pool_rows["search.jsonl"]
        ),
        "bbox_contact_instruction_proposals": sum(
            len(row["bbox_contact_instruction_proposals"])
            for row in pool_rows["search.jsonl"]
        ),
        "bbox_contact_options_after_cap": _bbox_contact_options_after_cap(
            pool_rows["search.jsonl"]
        ),
        "bbox_contact_structural_check_count": sum(
            row["bbox_contact_structural_check_count"]
            for row in pool_rows["search.jsonl"]
        ),
        "bbox_contact_anchor_candidate_count": sum(
            row["bbox_contact_anchor_candidate_count"]
            for row in pool_rows["search.jsonl"]
        ),
        "bbox_contact_relation_check_count": sum(
            row["bbox_contact_relation_check_count"]
            for row in pool_rows["search.jsonl"]
        ),
        "bbox_contact_admissible_binding_count": sum(
            row["bbox_contact_admissible_binding_count"]
            for row in pool_rows["search.jsonl"]
        ),
        "bbox_contact_action_trial_count": sum(
            row["bbox_contact_action_trial_count"]
            for row in pool_rows["search.jsonl"]
        ),
        "bbox_contact_demo_execution_count": sum(
            row["bbox_contact_demo_execution_count"]
            for row in pool_rows["search.jsonl"]
        ),
        "instruction_option_count_pre_cap": sum(
            row["instruction_option_count_pre_cap"]
            for row in pool_rows["search.jsonl"]
        ),
        "instruction_option_count_post_cap": sum(
            row["instruction_option_count_post_cap"]
            for row in pool_rows["search.jsonl"]
        ),
        "instruction_option_truncation_count": sum(
            row["instruction_option_truncation_count"]
            for row in pool_rows["search.jsonl"]
        ),
    }


def build_symbolic_pool_bundle(
    *,
    blind_dataset_dir: str | Path,
    output_dir: str | Path,
    search_config: SearchConfig,
    source_capture: RuntimeSourceCapture,
    test_source_sha256: str | None,
    source_loader: Mapping[str, object],
    command: list[str],
) -> dict[str, object]:
    target = Path(output_dir).expanduser().resolve()
    _require_nonoverlapping_paths(blind=blind_dataset_dir, output=target)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite evidence directory: {target}")
    blind_snapshot = snapshot_bundle(blind_dataset_dir)
    blind_manifest, tasks = _load_blind_dataset(blind_snapshot)
    if (
        blind_manifest["runtime_source_fingerprint_sha256"]
        != source_capture.fingerprint_sha256
    ):
        raise ValueError("blind control dataset source differs from pool runtime source")
    started_at = utc_now()
    config = {
        "search": asdict(search_config),
        "dsl_semantics_version": DSL_SEMANTICS_VERSION,
        "shape_proposer_semantics_version": SHAPE_PROPOSER_SEMANTICS_VERSION,
        "parser_semantics_version": PARSER_SEMANTICS_VERSION,
        "parser_stage": PARSER_STAGE,
        "panel_parser_semantics_version": PANEL_PARSER_SEMANTICS_VERSION,
        "panel_parser_stage": PANEL_PARSER_STAGE,
        "panel_sequence_d4_semantics_version": (
            PANEL_SEQUENCE_D4_SEMANTICS_VERSION
        ),
        "panel_sequence_d4_stage": PANEL_SEQUENCE_D4_STAGE,
        "panel_sequence_d4_proposer_semantics_version": (
            PANEL_SEQUENCE_D4_PROPOSER_SEMANTICS_VERSION
        ),
        "panel_lattice_periodic_semantics_version": (
            PANEL_LATTICE_PERIODIC_SEMANTICS_VERSION
        ),
        "panel_lattice_periodic_stage": PANEL_LATTICE_PERIODIC_STAGE,
        "panel_lattice_periodic_proposer_semantics_version": (
            PANEL_LATTICE_PERIODIC_PROPOSER_SEMANTICS_VERSION
        ),
        "relation_parser_semantics_version": RELATION_PARSER_SEMANTICS_VERSION,
        "relation_parser_stage": RELATION_PARSER_STAGE,
        "bbox_contact_action_semantics_version": (
            BBOX_CONTACT_ACTION_SEMANTICS_VERSION
        ),
        "bbox_contact_action_stage": BBOX_CONTACT_ACTION_STAGE,
        "bbox_contact_proposer_semantics_version": (
            BBOX_CONTACT_PROPOSER_SEMANTICS_VERSION
        ),
        "oracle_inputs_available_to_pool": False,
    }
    pool_spec_id = _content_id(
        "afts-e01a-pool-spec/v6",
        {
            "case_set_id": blind_manifest["case_set_id"],
            "blind_tasks_sha256": blind_manifest["blind_tasks_sha256"],
            "runtime_source_fingerprint_sha256": source_capture.fingerprint_sha256,
            "config": config,
        },
    )

    task_rows = [task.to_json_dict() for task in tasks]
    parse_rows: list[dict[str, object]] = []
    panel_parse_rows: list[dict[str, object]] = []
    relation_parse_rows: list[dict[str, object]] = []
    program_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    search_rows: list[dict[str, object]] = []
    for task in tasks:
        for index, pair in enumerate(task.train):
            parse_rows.append(
                _parse_bundle_row(
                    task=task,
                    pair_role="train_input",
                    pair_index=index,
                    bundle=parse_grid(pair.input),
                )
            )
            panel_parse_rows.append(
                _panel_parse_bundle_row(
                    task=task,
                    pair_role="train_input",
                    pair_index=index,
                    bundle=parse_panels(pair.input),
                )
            )
            relation_parse_rows.append(
                _relation_parse_bundle_row(
                    task=task,
                    pair_role="train_input",
                    pair_index=index,
                    grid=pair.input,
                )
            )
            parse_rows.append(
                _parse_bundle_row(
                    task=task,
                    pair_role="train_output",
                    pair_index=index,
                    bundle=parse_grid(pair.output),
                )
            )
            panel_parse_rows.append(
                _panel_parse_bundle_row(
                    task=task,
                    pair_role="train_output",
                    pair_index=index,
                    bundle=parse_panels(pair.output),
                )
            )
            relation_parse_rows.append(
                _relation_parse_bundle_row(
                    task=task,
                    pair_role="train_output",
                    pair_index=index,
                    grid=pair.output,
                )
            )
        for index, grid in enumerate(task.test_inputs):
            parse_rows.append(
                _parse_bundle_row(
                    task=task,
                    pair_role="test_input",
                    pair_index=index,
                    bundle=parse_grid(grid),
                )
            )
            panel_parse_rows.append(
                _panel_parse_bundle_row(
                    task=task,
                    pair_role="test_input",
                    pair_index=index,
                    bundle=parse_panels(grid),
                )
            )
            relation_parse_rows.append(
                _relation_parse_bundle_row(
                    task=task,
                    pair_role="test_input",
                    pair_index=index,
                    grid=grid,
                )
            )

        wall_start = time.perf_counter_ns()
        cpu_start = time.process_time_ns()
        tracemalloc.start()
        try:
            result = search_programs(task, config=search_config)
            _, peak_bytes = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        wall_ns = time.perf_counter_ns() - wall_start
        cpu_ns = time.process_time_ns() - cpu_start
        if result.expansions != len(result.evaluated_program_ids):
            raise RuntimeError("search expansion ledger is not one-to-one with program IDs")

        retained_ids: list[str] = []
        for rank, evaluation in enumerate(result.exact_evaluations):
            if not evaluation.all_demo_exact:
                raise RuntimeError("non-exact program entered the retained exact pool")
            if any(
                outcome.invalid_code is InvalidCode.INTERNAL_ERROR
                for outcome in (*evaluation.demo_outcomes, *evaluation.query_outcomes)
            ):
                raise RuntimeError("DSL internal error entered the evidence pool")
            retained_ids.append(evaluation.program.program_id)
            program_rows.append(
                {
                    "task_id": task.task_id,
                    "retained_rank": rank,
                    "program": evaluation.program.to_json_dict(),
                    "all_demo_exact": True,
                    "query_outcomes": [
                        _outcome_json(index, outcome)
                        for index, outcome in enumerate(evaluation.query_outcomes)
                    ],
                }
            )

        ranks: defaultdict[int, int] = defaultdict(int)
        for candidate in dsl_candidates(task, search_result=result):
            if candidate.program_hash not in set(retained_ids):
                raise RuntimeError("candidate does not reference a retained exact program")
            rank = ranks[candidate.test_index]
            candidate_rows.append(
                {
                    "emission_rank": rank,
                    "candidate": candidate.to_json_dict(),
                }
            )
            ranks[candidate.test_index] += 1
        search_rows.append(
            {
                "task_id": task.task_id,
                "blind_content_sha256": task.blind_content_sha256,
                "search_config": asdict(result.config),
                "shape_proposals": [
                    item.to_json_dict() for item in result.shape_proposals
                ],
                "panel_instruction_proposals": [
                    item.to_json_dict()
                    for item in result.panel_instruction_proposals
                ],
                "panel_sequence_d4_instruction_proposals": [
                    item.to_json_dict()
                    for item in result.panel_sequence_d4_instruction_proposals
                ],
                "panel_sequence_d4_trial_count": (
                    result.panel_sequence_d4_trial_count
                ),
                "panel_sequence_d4_demo_execution_count": (
                    result.panel_sequence_d4_demo_execution_count
                ),
                "panel_lattice_periodic_period_bounds": [
                    item.to_json_dict()
                    for item in result.panel_lattice_periodic_period_bounds
                ],
                "panel_lattice_periodic_instruction_proposals": [
                    item.to_json_dict()
                    for item in result.panel_lattice_periodic_instruction_proposals
                ],
                "panel_lattice_periodic_structural_check_count": (
                    result.panel_lattice_periodic_structural_check_count
                ),
                "panel_lattice_periodic_trial_count": (
                    result.panel_lattice_periodic_trial_count
                ),
                "panel_lattice_periodic_demo_execution_count": (
                    result.panel_lattice_periodic_demo_execution_count
                ),
                "bbox_contact_bounds": [
                    item.to_json_dict() for item in result.bbox_contact_bounds
                ],
                "bbox_contact_instruction_proposals": [
                    item.to_json_dict()
                    for item in result.bbox_contact_instruction_proposals
                ],
                "bbox_contact_structural_check_count": (
                    result.bbox_contact_structural_check_count
                ),
                "bbox_contact_anchor_candidate_count": (
                    result.bbox_contact_anchor_candidate_count
                ),
                "bbox_contact_relation_check_count": (
                    result.bbox_contact_relation_check_count
                ),
                "bbox_contact_admissible_binding_count": (
                    result.bbox_contact_admissible_binding_count
                ),
                "bbox_contact_action_trial_count": (
                    result.bbox_contact_action_trial_count
                ),
                "bbox_contact_demo_execution_count": (
                    result.bbox_contact_demo_execution_count
                ),
                "instruction_option_count_pre_cap": (
                    result.instruction_option_count_pre_cap
                ),
                "instruction_option_count_post_cap": (
                    result.instruction_option_count_post_cap
                ),
                "instruction_option_truncation_count": (
                    result.instruction_option_truncation_count
                ),
                "instruction_options": [item.to_json_dict() for item in result.instruction_options],
                "expansions": result.expansions,
                "program_execution_count": result.expansions
                * (len(task.train) + len(task.test_inputs)),
                "semantic_duplicates": result.semantic_duplicates,
                "first_exact_expansion": result.first_exact_expansion,
                "evaluated_program_ids": list(result.evaluated_program_ids),
                "retained_exact_program_ids": retained_ids,
                "candidate_records": sum(ranks.values()),
                "wall_time_ns": wall_ns,
                "process_time_ns": cpu_ns,
                "tracemalloc_peak_bytes": peak_bytes,
            }
        )

    for collection in (
        task_rows,
        parse_rows,
        panel_parse_rows,
        relation_parse_rows,
        program_rows,
        candidate_rows,
        search_rows,
        config,
    ):
        _assert_no_forbidden_pool_keys(collection)
    task_bytes = serialize_jsonl(task_rows)
    parse_bytes = serialize_jsonl(parse_rows)
    panel_parse_bytes = serialize_jsonl(panel_parse_rows)
    relation_parse_bytes = serialize_jsonl(relation_parse_rows)
    program_bytes = serialize_jsonl(program_rows)
    candidate_bytes = serialize_jsonl(candidate_rows)
    search_bytes = serialize_jsonl(search_rows)
    behavior_hashes = _pool_behavior_hashes(
        task_bytes=task_bytes,
        parse_bytes=parse_bytes,
        panel_parse_bytes=panel_parse_bytes,
        relation_parse_bytes=relation_parse_bytes,
        program_bytes=program_bytes,
        candidate_bytes=candidate_bytes,
        search_rows=search_rows,
    )
    pool_content_id = _content_id(
        "afts-e01a-pool-content/v6",
        {"pool_spec_id": pool_spec_id, "behavior_hashes": behavior_hashes},
    )
    completed_at = utc_now()
    pool_manifest = {
        "schema": POOL_MANIFEST_SCHEMA,
        "evidence_status": "frozen_oracle_unread_symbolic_candidate_pool",
        "pool_spec_id": pool_spec_id,
        "pool_content_id": pool_content_id,
        "case_set_id": blind_manifest["case_set_id"],
        "parent_blind_bundle": {
            "artifact_manifest_sha256": blind_snapshot.artifact_manifest_sha256,
            "blind_tasks_sha256": blind_manifest["blind_tasks_sha256"],
        },
        "config": config,
        "started_at": started_at,
        "completed_at": completed_at,
        "source_loader": dict(source_loader),
        "command": command,
        **_runtime_record(
            source_capture=source_capture, test_source_sha256=test_source_sha256
        ),
    }
    pool_summary = _pool_summary_payload(
        blind_manifest=blind_manifest,
        pool_spec_id=pool_spec_id,
        pool_content_id=pool_content_id,
        task_rows=task_rows,
        parse_rows=parse_rows,
        panel_parse_rows=panel_parse_rows,
        relation_parse_rows=relation_parse_rows,
        program_rows=program_rows,
        candidate_rows=candidate_rows,
        search_rows=search_rows,
    )
    for payload in (pool_manifest, pool_summary):
        _assert_no_forbidden_pool_keys(payload)
    if runtime_source_fingerprint() != source_capture.fingerprint_sha256:
        raise RuntimeError("executing source changed during symbolic pool construction")
    publish_evidence_bundle(
        target,
        run_id=pool_content_id[:20],
        artifacts={
            "pool_manifest.json": serialize_json(pool_manifest),
            "config.json": serialize_json(config),
            "tasks.jsonl": task_bytes,
            "parses.jsonl": parse_bytes,
            "panel_parses.jsonl": panel_parse_bytes,
            "relation_parses.jsonl": relation_parse_bytes,
            "exact_programs.jsonl": program_bytes,
            "candidates.jsonl": candidate_bytes,
            "search.jsonl": search_bytes,
            "pool_summary.json": serialize_json(pool_summary),
            "source_snapshot.zip": source_capture.snapshot_zip,
        },
    )
    return pool_summary


def _load_oracle_dataset(
    snapshot: BundleSnapshot,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    _require_artifacts(
        snapshot, {"oracle_manifest.json", "oracle.jsonl", "source_snapshot.zip"}
    )
    manifest = _require_fields(
        _strict_json(snapshot.artifacts["oracle_manifest.json"], label="oracle_manifest.json"),
        {
            "schema",
            "evidence_status",
            "case_set_id",
            "oracle_set_id",
            "suite_version",
            "task_count",
            "oracle_sha256",
            "generator_config",
            "started_at",
            "completed_at",
            "source_loader",
            "command",
            "runtime_source_fingerprint_sha256",
            "test_source_fingerprint_sha256",
        },
        label="oracle manifest",
    )
    if manifest["schema"] != ORACLE_MANIFEST_SCHEMA:
        raise ValueError("unsupported oracle manifest schema")
    verify_source_snapshot_zip(
        snapshot.artifacts["source_snapshot.zip"],
        expected_fingerprint_sha256=manifest["runtime_source_fingerprint_sha256"],
    )
    oracle_bytes = snapshot.artifacts["oracle.jsonl"]
    if manifest["oracle_sha256"] != _sha256(oracle_bytes):
        raise ValueError("oracle manifest does not bind oracle.jsonl")
    rows = _strict_jsonl(oracle_bytes, label="oracle.jsonl")
    by_id: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(rows):
        row = _require_fields(
            raw,
            {"schema", "task_id", "family", "stratum", "seed", "generator_program", "test"},
            label=f"oracle row {index}",
        )
        if row["schema"] != ORACLE_ROW_SCHEMA:
            raise ValueError("unsupported oracle row schema")
        if row["task_id"] in by_id:
            raise ValueError("duplicate task in oracle bundle")
        Program.from_json_dict(row["generator_program"])
        if not isinstance(row["test"], list) or not row["test"]:
            raise ValueError("oracle row must contain test outputs")
        for test_index, test_row in enumerate(row["test"]):
            test = _require_fields(
                test_row, {"test_index", "output", "output_key"}, label="oracle test row"
            )
            if test["test_index"] != test_index:
                raise ValueError("oracle test indices must be contiguous and ordered")
            output = as_grid(test["output"])
            if test["output_key"] != grid_key(output):
                raise ValueError("oracle output_key mismatch")
        by_id[row["task_id"]] = row
    if manifest["task_count"] != len(rows):
        raise ValueError("oracle task count mismatch")
    config = _require_fields(
        manifest["generator_config"],
        {"seed", "tasks_per_family"},
        label="oracle generator config",
    )
    if type(config["seed"]) is not int or type(config["tasks_per_family"]) is not int:
        raise TypeError("oracle generator seed and tasks_per_family must be integers")
    if config["tasks_per_family"] <= 0:
        raise ValueError("oracle tasks_per_family must be positive")
    regenerated = sorted(
        generate_suite(seed=config["seed"], tasks_per_family=config["tasks_per_family"]),
        key=lambda case: case.blind_task.task_id,
    )
    expected_rows = [_oracle_row(case) for case in regenerated]
    if rows != expected_rows:
        raise ValueError("oracle rows do not reproduce from the declared generator config")
    expected_oracle_set = _content_id(
        "afts-e01a-oracle-set/v1",
        {"case_set_id": manifest["case_set_id"], "oracle_sha256": _sha256(oracle_bytes)},
    )
    if manifest["oracle_set_id"] != expected_oracle_set:
        raise ValueError("oracle_set_id does not match oracle content")
    return manifest, by_id


def _load_public_oracle_dataset(
    snapshot: BundleSnapshot,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    _require_artifacts(
        snapshot, {"oracle_manifest.json", "oracle.jsonl", "source_snapshot.zip"}
    )
    manifest = _require_fields(
        _strict_json(snapshot.artifacts["oracle_manifest.json"], label="oracle_manifest.json"),
        {
            "schema",
            "evidence_status",
            "case_set_id",
            "oracle_set_id",
            "dataset_descriptor",
            "task_count",
            "oracle_sha256",
            "dataset_audit",
            "selection_audit",
            "started_at",
            "completed_at",
            "source_loader",
            "command",
            "runtime_source_fingerprint_sha256",
            "test_source_fingerprint_sha256",
        },
        label="public oracle manifest",
    )
    if manifest["schema"] != PUBLIC_ORACLE_MANIFEST_SCHEMA:
        raise ValueError("unsupported public oracle manifest schema")
    verify_source_snapshot_zip(
        snapshot.artifacts["source_snapshot.zip"],
        expected_fingerprint_sha256=manifest["runtime_source_fingerprint_sha256"],
    )
    audit = manifest["dataset_audit"]
    _validate_pinned_arc2_training_audit(audit)
    oracle_bytes = snapshot.artifacts["oracle.jsonl"]
    if manifest["oracle_sha256"] != _sha256(oracle_bytes):
        raise ValueError("public oracle manifest does not bind oracle.jsonl")
    rows = _strict_jsonl(oracle_bytes, label="oracle.jsonl")
    by_id: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(rows):
        row = _require_fields(
            raw,
            {"schema", "task_id", "source_task_id", "source_sha256", "test"},
            label=f"public oracle row {index}",
        )
        if row["schema"] != PUBLIC_ORACLE_ROW_SCHEMA:
            raise ValueError("unsupported public oracle row schema")
        if not isinstance(row["task_id"], str) or row["task_id"] in by_id:
            raise ValueError("public oracle task IDs must be unique strings")
        if not isinstance(row["source_task_id"], str) or not row["source_task_id"]:
            raise ValueError("public oracle source_task_id must be non-empty")
        digest = row["source_sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest.lower())
        ):
            raise ValueError("public oracle source_sha256 must be hexadecimal")
        if not isinstance(row["test"], list) or not row["test"]:
            raise ValueError("public oracle row must contain query outputs")
        for test_index, test_raw in enumerate(row["test"]):
            test = _require_fields(
                test_raw,
                {"test_index", "output", "output_key"},
                label="public oracle test row",
            )
            if test["test_index"] != test_index:
                raise ValueError("public oracle test indices must be contiguous")
            output = as_grid(test["output"])
            if test["output_key"] != grid_key(output):
                raise ValueError("public oracle output_key mismatch")
        by_id[row["task_id"]] = row
    if manifest["task_count"] != len(rows):
        raise ValueError("public oracle task count mismatch")
    if tuple(by_id) != tuple(sorted(by_id)):
        raise ValueError("public oracle rows must be sorted by blind task_id")
    expected_oracle_set = _content_id(
        "afts-e01a-public-oracle-set/v1",
        {"case_set_id": manifest["case_set_id"], "oracle_sha256": _sha256(oracle_bytes)},
    )
    if manifest["oracle_set_id"] != expected_oracle_set:
        raise ValueError("public oracle_set_id does not match content")
    return manifest, by_id


def _load_pool(snapshot: BundleSnapshot) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    _require_artifacts(
        snapshot,
        {
            "pool_manifest.json",
            "config.json",
            "tasks.jsonl",
            "parses.jsonl",
            "panel_parses.jsonl",
            "relation_parses.jsonl",
            "exact_programs.jsonl",
            "candidates.jsonl",
            "search.jsonl",
            "pool_summary.json",
            "source_snapshot.zip",
        },
    )
    manifest = _require_fields(
        _strict_json(snapshot.artifacts["pool_manifest.json"], label="pool_manifest.json"),
        {
            "schema",
            "evidence_status",
            "pool_spec_id",
            "pool_content_id",
            "case_set_id",
            "parent_blind_bundle",
            "config",
            "started_at",
            "completed_at",
            "source_loader",
            "command",
            "runtime_source_fingerprint_sha256",
            "test_source_fingerprint_sha256",
        },
        label="pool manifest",
    )
    if manifest["schema"] != POOL_MANIFEST_SCHEMA:
        raise ValueError("unsupported pool manifest schema")
    verify_source_snapshot_zip(
        snapshot.artifacts["source_snapshot.zip"],
        expected_fingerprint_sha256=manifest["runtime_source_fingerprint_sha256"],
    )
    if _strict_json(snapshot.artifacts["config.json"], label="config.json") != manifest["config"]:
        raise ValueError("pool config artifact and manifest disagree")
    rows = {
        name: _strict_jsonl(snapshot.artifacts[name], label=name)
        for name in (
            "tasks.jsonl",
            "parses.jsonl",
            "panel_parses.jsonl",
            "relation_parses.jsonl",
            "exact_programs.jsonl",
            "candidates.jsonl",
            "search.jsonl",
        )
    }
    for payload in (manifest, *rows.values()):
        _assert_no_forbidden_pool_keys(payload)
    return manifest, rows


def _candidate_groups(
    rows: Iterable[dict[str, Any]],
) -> tuple[
    dict[tuple[str, int], list[tuple[int, CandidateRecord]]],
    list[CandidateRecord],
]:
    groups: defaultdict[tuple[str, int], list[tuple[int, CandidateRecord]]] = defaultdict(list)
    all_candidates: list[CandidateRecord] = []
    for index, raw in enumerate(rows):
        row = _require_fields(raw, {"emission_rank", "candidate"}, label=f"candidate row {index}")
        if type(row["emission_rank"]) is not int or row["emission_rank"] < 0:
            raise ValueError("candidate emission_rank must be non-negative")
        candidate = CandidateRecord.from_json_dict(row["candidate"])
        groups[(candidate.task_id, candidate.test_index)].append(
            (row["emission_rank"], candidate)
        )
        all_candidates.append(candidate)
    for key, group in groups.items():
        group.sort(key=lambda item: item[0])
        if [rank for rank, _ in group] != list(range(len(group))):
            raise ValueError(f"candidate ranks are not contiguous for {key}")
    return dict(groups), all_candidates


def _program_groups(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, raw in enumerate(rows):
        row = _require_fields(
            raw,
            {"task_id", "retained_rank", "program", "all_demo_exact", "query_outcomes"},
            label=f"program row {index}",
        )
        program = Program.from_json_dict(row["program"])
        if row["all_demo_exact"] is not True:
            raise ValueError("retained program is not marked demo exact")
        if type(row["retained_rank"]) is not int or row["retained_rank"] < 0:
            raise ValueError("retained_rank must be non-negative")
        normalized = dict(row)
        normalized["_program_object"] = program
        groups[row["task_id"]].append(normalized)
    for task_id, group in groups.items():
        group.sort(key=lambda item: item["retained_rank"])
        if [row["retained_rank"] for row in group] != list(range(len(group))):
            raise ValueError(f"program ranks are not contiguous for {task_id}")
    return dict(groups)


def _search_by_task(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_task: dict[str, dict[str, Any]] = {}
    expected_fields = {
        "task_id",
        "blind_content_sha256",
        "search_config",
        "shape_proposals",
        "panel_instruction_proposals",
        "panel_sequence_d4_instruction_proposals",
        "panel_sequence_d4_trial_count",
        "panel_sequence_d4_demo_execution_count",
        "panel_lattice_periodic_period_bounds",
        "panel_lattice_periodic_instruction_proposals",
        "panel_lattice_periodic_structural_check_count",
        "panel_lattice_periodic_trial_count",
        "panel_lattice_periodic_demo_execution_count",
        "bbox_contact_bounds",
        "bbox_contact_instruction_proposals",
        "bbox_contact_structural_check_count",
        "bbox_contact_anchor_candidate_count",
        "bbox_contact_relation_check_count",
        "bbox_contact_admissible_binding_count",
        "bbox_contact_action_trial_count",
        "bbox_contact_demo_execution_count",
        "instruction_option_count_pre_cap",
        "instruction_option_count_post_cap",
        "instruction_option_truncation_count",
        "instruction_options",
        "expansions",
        "program_execution_count",
        "semantic_duplicates",
        "first_exact_expansion",
        "evaluated_program_ids",
        "retained_exact_program_ids",
        "candidate_records",
        "wall_time_ns",
        "process_time_ns",
        "tracemalloc_peak_bytes",
    }
    for index, raw in enumerate(rows):
        row = _require_fields(raw, expected_fields, label=f"search row {index}")
        task_id = row["task_id"]
        if not isinstance(task_id, str) or task_id in by_task:
            raise ValueError("search rows must have unique task IDs")
        if not isinstance(row["shape_proposals"], list):
            raise TypeError("search shape_proposals must be a list")
        for proposal in row["shape_proposals"]:
            OutputShapeProposal.from_json_dict(proposal)
        if not isinstance(row["panel_instruction_proposals"], list):
            raise TypeError("search panel_instruction_proposals must be a list")
        for proposal in row["panel_instruction_proposals"]:
            if not isinstance(proposal, dict) or set(proposal) != {
                "op",
                "arguments",
            }:
                raise ValueError("panel instruction proposal has invalid fields")
            if not isinstance(proposal["arguments"], dict):
                raise TypeError("panel instruction proposal arguments must be an object")
            instruction = Instruction.create(
                proposal["op"], **proposal["arguments"]
            )
            if instruction.op != "overlay_panel_grid" or instruction.to_json_dict() != proposal:
                raise ValueError("panel instruction proposal is not canonical overlay")
        sequence_proposals = row["panel_sequence_d4_instruction_proposals"]
        if not isinstance(sequence_proposals, list):
            raise TypeError("panel-sequence D4 proposals must be a list")
        canonical_sequence_proposals: list[dict[str, object]] = []
        for proposal in sequence_proposals:
            if not isinstance(proposal, dict) or set(proposal) != {
                "op",
                "arguments",
            }:
                raise ValueError("panel-sequence D4 proposal has invalid fields")
            if not isinstance(proposal["arguments"], dict):
                raise TypeError("panel-sequence D4 proposal arguments must be an object")
            instruction = Instruction.create(
                proposal["op"], **proposal["arguments"]
            )
            if (
                instruction.op != "broadcast_panel_sequence_d4"
                or instruction.to_json_dict() != proposal
            ):
                raise ValueError("panel-sequence D4 proposal is not canonical")
            canonical_sequence_proposals.append(proposal)
        if len(
            {
                json.dumps(item, sort_keys=True, separators=(",", ":"))
                for item in canonical_sequence_proposals
            }
        ) != len(canonical_sequence_proposals):
            raise ValueError("panel-sequence D4 proposals contain duplicates")
        step_order = {step.value: index for index, step in enumerate(D4Step)}
        if canonical_sequence_proposals != sorted(
            canonical_sequence_proposals,
            key=lambda item: (
                item["arguments"]["background"],
                step_order[item["arguments"]["step"]],
            ),
        ):
            raise ValueError("panel-sequence D4 proposals are not canonically ordered")
        periodic_bounds = row["panel_lattice_periodic_period_bounds"]
        if not isinstance(periodic_bounds, list):
            raise TypeError("panel-lattice periodic bounds must be a list")
        canonical_periodic_bounds: list[dict[str, int]] = []
        for bound in periodic_bounds:
            if not isinstance(bound, dict) or set(bound) != {
                "background",
                "max_row_period",
                "max_column_period",
            }:
                raise ValueError("panel-lattice periodic bound has invalid fields")
            if (
                type(bound["background"]) is not int
                or not 0 <= bound["background"] <= 9
                or type(bound["max_row_period"]) is not int
                or not 1 <= bound["max_row_period"] <= 29
                or type(bound["max_column_period"]) is not int
                or not 1 <= bound["max_column_period"] <= 29
            ):
                raise ValueError("panel-lattice periodic bound is outside its domain")
            canonical_periodic_bounds.append(bound)
        if canonical_periodic_bounds != sorted(
            canonical_periodic_bounds,
            key=lambda item: item["background"],
        ) or len({item["background"] for item in canonical_periodic_bounds}) != len(
            canonical_periodic_bounds
        ):
            raise ValueError("panel-lattice periodic bounds are not canonical")
        if len(canonical_periodic_bounds) > 3:
            raise ValueError("panel-lattice periodic bounds exceed the background cap")
        bounds_by_background = {
            item["background"]: item for item in canonical_periodic_bounds
        }
        periodic_proposals = row[
            "panel_lattice_periodic_instruction_proposals"
        ]
        if not isinstance(periodic_proposals, list):
            raise TypeError("panel-lattice periodic proposals must be a list")
        canonical_periodic_proposals: list[dict[str, object]] = []
        for proposal in periodic_proposals:
            if not isinstance(proposal, dict) or set(proposal) != {
                "op",
                "arguments",
            }:
                raise ValueError("panel-lattice periodic proposal has invalid fields")
            if not isinstance(proposal["arguments"], dict):
                raise TypeError(
                    "panel-lattice periodic proposal arguments must be an object"
                )
            instruction = Instruction.create(
                proposal["op"], **proposal["arguments"]
            )
            if (
                instruction.op != "broadcast_panel_lattice_periodic"
                or instruction.to_json_dict() != proposal
            ):
                raise ValueError("panel-lattice periodic proposal is not canonical")
            background = proposal["arguments"]["background"]
            bound = bounds_by_background.get(background)
            if bound is None or not (
                proposal["arguments"]["row_period"]
                <= bound["max_row_period"]
                and proposal["arguments"]["column_period"]
                <= bound["max_column_period"]
            ):
                raise ValueError("panel-lattice periodic proposal exceeds its bound")
            canonical_periodic_proposals.append(proposal)
        if len(
            {
                json.dumps(item, sort_keys=True, separators=(",", ":"))
                for item in canonical_periodic_proposals
            }
        ) != len(canonical_periodic_proposals):
            raise ValueError("panel-lattice periodic proposals contain duplicates")
        if canonical_periodic_proposals != sorted(
            canonical_periodic_proposals,
            key=lambda item: (
                item["arguments"]["background"],
                item["arguments"]["row_period"],
                item["arguments"]["column_period"],
            ),
        ):
            raise ValueError(
                "panel-lattice periodic proposals are not canonically ordered"
            )
        contact_bounds = row["bbox_contact_bounds"]
        if not isinstance(contact_bounds, list):
            raise TypeError("bbox-contact bounds must be a list")
        canonical_contact_bounds: list[dict[str, int]] = []
        for bound in contact_bounds:
            if not isinstance(bound, dict) or set(bound) != {
                "background",
                "max_object_count",
                "max_anchor_candidate_count",
                "max_relation_count",
            }:
                raise ValueError("bbox-contact bound has invalid fields")
            if (
                type(bound["background"]) is not int
                or not 0 <= bound["background"] <= 9
            ):
                raise ValueError("bbox-contact bound background is outside ARC colors")
            for field in (
                "max_object_count",
                "max_anchor_candidate_count",
                "max_relation_count",
            ):
                if type(bound[field]) is not int or not 0 <= bound[field] <= 900:
                    raise ValueError("bbox-contact bound count is outside its domain")
            if bound["max_anchor_candidate_count"] > bound["max_object_count"]:
                raise ValueError("bbox-contact anchor bound exceeds object bound")
            if bound["max_relation_count"] > max(
                0, bound["max_object_count"] - 1
            ):
                raise ValueError("bbox-contact relation bound exceeds object bound")
            canonical_contact_bounds.append(bound)
        if canonical_contact_bounds != sorted(
            canonical_contact_bounds, key=lambda item: item["background"]
        ) or len({item["background"] for item in canonical_contact_bounds}) != len(
            canonical_contact_bounds
        ):
            raise ValueError("bbox-contact bounds are not canonical")
        if len(canonical_contact_bounds) > 3:
            raise ValueError("bbox-contact bounds exceed the background cap")
        contact_bound_backgrounds = {
            item["background"] for item in canonical_contact_bounds
        }
        contact_proposals = row["bbox_contact_instruction_proposals"]
        if not isinstance(contact_proposals, list):
            raise TypeError("bbox-contact proposals must be a list")
        canonical_contact_proposals: list[dict[str, object]] = []
        for proposal in contact_proposals:
            if not isinstance(proposal, dict) or set(proposal) != {
                "op",
                "arguments",
            }:
                raise ValueError("bbox-contact proposal has invalid fields")
            if not isinstance(proposal["arguments"], dict):
                raise TypeError("bbox-contact proposal arguments must be an object")
            instruction = Instruction.create(
                proposal["op"], **proposal["arguments"]
            )
            if (
                instruction.op != "paint_bbox_contacts"
                or instruction.to_json_dict() != proposal
                or proposal["arguments"].get("background")
                not in contact_bound_backgrounds
            ):
                raise ValueError("bbox-contact proposal is not canonical or bounded")
            canonical_contact_proposals.append(proposal)
        if canonical_contact_proposals != sorted(
            canonical_contact_proposals,
            key=lambda item: item["arguments"]["background"],
        ) or len(
            {
                item["arguments"]["background"]
                for item in canonical_contact_proposals
            }
        ) != len(canonical_contact_proposals):
            raise ValueError("bbox-contact proposals are not unique and canonical")
        if not isinstance(row["instruction_options"], list):
            raise TypeError("search instruction_options must be a list")
        for option in row["instruction_options"]:
            if not isinstance(option, dict) or set(option) != {"op", "arguments"}:
                raise ValueError("search instruction option has invalid fields")
            if not isinstance(option["arguments"], dict):
                raise TypeError("search instruction option arguments must be an object")
            if Instruction.create(option["op"], **option["arguments"]).to_json_dict() != option:
                raise ValueError("search instruction option is not canonical")
        evaluated = row["evaluated_program_ids"]
        retained = row["retained_exact_program_ids"]
        if not isinstance(evaluated, list) or not isinstance(retained, list):
            raise TypeError("search program ID ledgers must be lists")
        for field in (
            "expansions",
            "program_execution_count",
            "semantic_duplicates",
            "candidate_records",
            "panel_sequence_d4_trial_count",
            "panel_sequence_d4_demo_execution_count",
            "panel_lattice_periodic_structural_check_count",
            "panel_lattice_periodic_trial_count",
            "panel_lattice_periodic_demo_execution_count",
            "bbox_contact_structural_check_count",
            "bbox_contact_anchor_candidate_count",
            "bbox_contact_relation_check_count",
            "bbox_contact_admissible_binding_count",
            "bbox_contact_action_trial_count",
            "bbox_contact_demo_execution_count",
            "instruction_option_count_pre_cap",
            "instruction_option_count_post_cap",
            "instruction_option_truncation_count",
            "wall_time_ns",
            "process_time_ns",
            "tracemalloc_peak_bytes",
        ):
            if type(row[field]) is not int or row[field] < 0:
                raise ValueError(f"search cost/count field {field} must be a non-negative int")
        search_config = _require_fields(
            row["search_config"],
            {"max_depth", "beam_width", "max_instruction_options", "max_exact_programs"},
            label="search row config",
        )
        if row["panel_sequence_d4_trial_count"] > 24:
            raise ValueError("panel-sequence D4 trial count exceeds the fixed domain")
        expected_periodic_trials = sum(
            item["max_row_period"] * item["max_column_period"]
            for item in canonical_periodic_bounds
        )
        if row["panel_lattice_periodic_trial_count"] != expected_periodic_trials:
            raise ValueError("panel-lattice periodic trial ledger is inconsistent")
        if row["panel_lattice_periodic_trial_count"] > 3 * 29 * 29:
            raise ValueError("panel-lattice periodic trial count exceeds the domain")
        if (
            row["panel_lattice_periodic_trial_count"] == 0
            and row["panel_lattice_periodic_demo_execution_count"] != 0
        ) or (
            row["panel_lattice_periodic_trial_count"] > 0
            and row["panel_lattice_periodic_demo_execution_count"] == 0
        ) or (
            row["panel_lattice_periodic_trial_count"] > 0
            and row["panel_lattice_periodic_demo_execution_count"]
            % row["panel_lattice_periodic_trial_count"]
            != 0
        ):
            raise ValueError(
                "panel-lattice periodic demo-execution ledger is inconsistent"
            )
        if len(canonical_periodic_proposals) > row[
            "panel_lattice_periodic_trial_count"
        ]:
            raise ValueError("panel-lattice periodic proposals exceed trial count")
        if canonical_periodic_bounds and row[
            "panel_lattice_periodic_structural_check_count"
        ] == 0:
            raise ValueError("panel-lattice periodic structural ledger is empty")
        contact_bound_count = len(canonical_contact_bounds)
        contact_structural_count = row["bbox_contact_structural_check_count"]
        if contact_bound_count == 0:
            if contact_structural_count != 0:
                raise ValueError("bbox-contact structural ledger has no bounds")
            contact_demo_count = 0
        else:
            if (
                contact_structural_count == 0
                or contact_structural_count % contact_bound_count != 0
            ):
                raise ValueError("bbox-contact structural ledger is inconsistent")
            contact_demo_count = contact_structural_count // contact_bound_count
        if row["bbox_contact_action_trial_count"] != row[
            "bbox_contact_admissible_binding_count"
        ]:
            raise ValueError("bbox-contact action trials do not close to bindings")
        if row["bbox_contact_admissible_binding_count"] > contact_bound_count:
            raise ValueError("bbox-contact admissible bindings exceed bounds")
        if row["bbox_contact_demo_execution_count"] != (
            row["bbox_contact_action_trial_count"] * contact_demo_count
        ):
            raise ValueError("bbox-contact demo-execution ledger is inconsistent")
        if len(canonical_contact_proposals) > row[
            "bbox_contact_action_trial_count"
        ]:
            raise ValueError("bbox-contact proposals exceed action trials")
        max_anchor_sum = sum(
            item["max_anchor_candidate_count"]
            for item in canonical_contact_bounds
        )
        if not (
            max_anchor_sum
            <= row["bbox_contact_anchor_candidate_count"]
            <= contact_demo_count * max_anchor_sum
        ):
            raise ValueError("bbox-contact anchor-candidate ledger is inconsistent")
        relation_check_upper_bound = contact_demo_count * sum(
            item["max_anchor_candidate_count"]
            * max(0, item["max_object_count"] - 1)
            for item in canonical_contact_bounds
        )
        if row["bbox_contact_relation_check_count"] > relation_check_upper_bound:
            raise ValueError("bbox-contact relation-check ledger exceeds its bounds")
        contact_option_keys = {
            json.dumps(item, sort_keys=True, separators=(",", ":"))
            for item in row["instruction_options"]
        }
        contact_options_after_cap = sum(
            json.dumps(item, sort_keys=True, separators=(",", ":"))
            in contact_option_keys
            for item in canonical_contact_proposals
        )
        if contact_options_after_cap > len(canonical_contact_proposals):
            raise ValueError("bbox-contact options-after-cap exceeds proposals")
        if row["instruction_option_count_post_cap"] != len(row["instruction_options"]):
            raise ValueError("instruction post-cap count does not match options")
        if row["instruction_option_count_post_cap"] != min(
            row["instruction_option_count_pre_cap"],
            search_config["max_instruction_options"],
        ):
            raise ValueError("instruction option cap ledger is inconsistent")
        if row["instruction_option_truncation_count"] != (
            row["instruction_option_count_pre_cap"]
            - row["instruction_option_count_post_cap"]
        ):
            raise ValueError("instruction option truncation ledger is inconsistent")
        first_exact = row["first_exact_expansion"]
        if first_exact is not None and (
            type(first_exact) is not int
            or first_exact <= 0
            or first_exact > row["expansions"]
        ):
            raise ValueError("first_exact_expansion must lie within the expansion ledger")
        if row["expansions"] != len(evaluated) or len(evaluated) != len(set(evaluated)):
            raise ValueError("search expansion ledger mismatch")
        if not set(retained).issubset(evaluated):
            raise ValueError("retained exact programs must be evaluated")
        by_task[task_id] = row
    return by_task


def _program_query_outputs(row: dict[str, Any], query_count: int) -> tuple[Grid | None, ...]:
    outcomes = row["query_outcomes"]
    if not isinstance(outcomes, list) or len(outcomes) != query_count:
        raise ValueError("program query outcome count mismatch")
    outputs: list[Grid | None] = []
    for expected_index, outcome in enumerate(outcomes):
        value = _require_fields(
            outcome,
            {
                "test_index",
                "status",
                "output",
                "output_key",
                "invalid_code",
                "failing_instruction_index",
                "semantic_trace",
            },
            label="query outcome",
        )
        if value["test_index"] != expected_index:
            raise ValueError("query outcomes must be contiguous and ordered")
        if value["output"] is None:
            if value["output_key"] is not None:
                raise ValueError("invalid query outcome may not have output_key")
            outputs.append(None)
        else:
            output = as_grid(value["output"])
            if value["status"] != "ok" or value["output_key"] != grid_key(output):
                raise ValueError("query outcome output/status/hash mismatch")
            outputs.append(output)
    return tuple(outputs)


def _validate_parse_rows(
    tasks: Mapping[str, BlindTask], rows: list[dict[str, Any]]
) -> None:
    expected: dict[tuple[str, str, int], Grid] = {}
    for task in tasks.values():
        for index, pair in enumerate(task.train):
            expected[(task.task_id, "train_input", index)] = pair.input
            expected[(task.task_id, "train_output", index)] = pair.output
        for index, grid in enumerate(task.test_inputs):
            expected[(task.task_id, "test_input", index)] = grid
    observed: set[tuple[str, str, int]] = set()
    observed_order: list[tuple[str, str, int]] = []
    for row in rows:
        task_id = row.get("task_id")
        role = row.get("pair_role")
        index = row.get("pair_index")
        if not isinstance(task_id, str) or not isinstance(role, str) or type(index) is not int:
            raise ValueError("parse provenance key is malformed")
        key = (task_id, role, index)
        if key in observed or key not in expected:
            raise ValueError("parse rows contain a duplicate or unexpected grid role")
        observed.add(key)
        observed_order.append(key)
        replayed = parse_grid(expected[key])
        expected_row = _parse_bundle_row(
            task=tasks[task_id],
            pair_role=role,
            pair_index=index,
            bundle=replayed,
        )
        if row != expected_row:
            raise ValueError("parse sidecar row does not exactly replay from its blind grid")
    if observed != set(expected):
        raise ValueError("parse sidecar does not cover every observable grid exactly once")
    if observed_order != list(expected):
        raise ValueError("parse sidecar rows are not in canonical order")


def _validate_panel_parse_rows(
    tasks: Mapping[str, BlindTask], rows: list[dict[str, Any]]
) -> None:
    expected: dict[tuple[str, str, int], Grid] = {}
    for task in tasks.values():
        for index, pair in enumerate(task.train):
            expected[(task.task_id, "train_input", index)] = pair.input
            expected[(task.task_id, "train_output", index)] = pair.output
        for index, grid in enumerate(task.test_inputs):
            expected[(task.task_id, "test_input", index)] = grid
    observed: set[tuple[str, str, int]] = set()
    observed_order: list[tuple[str, str, int]] = []
    row_fields = {
        "schema",
        "task_id",
        "blind_content_sha256",
        "pair_role",
        "pair_index",
        "panel_parse_bundle_id",
        "panel_parser_semantics_version",
        "panel_parser_stage",
        "grid_key",
        "hypotheses",
    }
    for index, raw in enumerate(rows):
        row = _require_fields(raw, row_fields, label=f"panel parse row {index}")
        task_id = row["task_id"]
        role = row["pair_role"]
        pair_index = row["pair_index"]
        if (
            not isinstance(task_id, str)
            or not isinstance(role, str)
            or type(pair_index) is not int
        ):
            raise ValueError("panel parse provenance key is malformed")
        key = (task_id, role, pair_index)
        if key in observed or key not in expected:
            raise ValueError("panel parse rows contain a duplicate or unexpected grid role")
        observed.add(key)
        observed_order.append(key)
        if (
            row["schema"] != PANEL_PARSE_ROW_SCHEMA
            or row["panel_parser_semantics_version"]
            != PANEL_PARSER_SEMANTICS_VERSION
            or row["panel_parser_stage"] != PANEL_PARSER_STAGE
            or not isinstance(row["hypotheses"], list)
        ):
            raise ValueError("panel parse row semantics do not match the executable parser")
        PanelParseBundle.from_json_dict(
            {"grid_key": row["grid_key"], "hypotheses": row["hypotheses"]}
        )
        replayed = parse_panels(expected[key])
        expected_row = _panel_parse_bundle_row(
            task=tasks[task_id],
            pair_role=role,
            pair_index=pair_index,
            bundle=replayed,
        )
        if row != expected_row:
            raise ValueError(
                "panel parse sidecar row does not exactly replay from its blind grid"
            )
    if observed != set(expected):
        raise ValueError(
            "panel parse sidecar does not cover every observable grid exactly once"
        )
    if observed_order != list(expected):
        raise ValueError("panel parse sidecar rows are not in canonical order")


def _validate_relation_parse_rows(
    tasks: Mapping[str, BlindTask], rows: list[dict[str, Any]]
) -> None:
    expected: dict[tuple[str, str, int], Grid] = {}
    for task in tasks.values():
        for index, pair in enumerate(task.train):
            expected[(task.task_id, "train_input", index)] = pair.input
            expected[(task.task_id, "train_output", index)] = pair.output
        for index, grid in enumerate(task.test_inputs):
            expected[(task.task_id, "test_input", index)] = grid
    observed: set[tuple[str, str, int]] = set()
    observed_order: list[tuple[str, str, int]] = []
    row_fields = {
        "schema",
        "task_id",
        "blind_content_sha256",
        "pair_role",
        "pair_index",
        "relation_parse_bundle_id",
        "relation_parser_semantics_version",
        "relation_parser_stage",
        "grid_key",
        "background_candidates",
        "background_bundles",
    }
    for index, raw in enumerate(rows):
        row = _require_fields(raw, row_fields, label=f"relation parse row {index}")
        task_id = row["task_id"]
        role = row["pair_role"]
        pair_index = row["pair_index"]
        if (
            not isinstance(task_id, str)
            or not isinstance(role, str)
            or type(pair_index) is not int
        ):
            raise ValueError("relation parse provenance key is malformed")
        key = (task_id, role, pair_index)
        if key in observed or key not in expected:
            raise ValueError(
                "relation parse rows contain a duplicate or unexpected grid role"
            )
        observed.add(key)
        observed_order.append(key)
        backgrounds = row["background_candidates"]
        bundles = row["background_bundles"]
        if (
            row["schema"] != RELATION_PARSE_ROW_SCHEMA
            or row["relation_parser_semantics_version"]
            != RELATION_PARSER_SEMANTICS_VERSION
            or row["relation_parser_stage"] != RELATION_PARSER_STAGE
            or not isinstance(backgrounds, list)
            or not isinstance(bundles, list)
            or len(backgrounds) != len(bundles)
            or len(backgrounds) != len(set(backgrounds))
            or any(type(background) is not int or not 0 <= background <= 9 for background in backgrounds)
        ):
            raise ValueError(
                "relation parse row semantics or background domain is malformed"
            )
        parsed_bundles = tuple(
            BBoxContactParseBundle.from_json_dict(bundle) for bundle in bundles
        )
        if tuple(bundle.background for bundle in parsed_bundles) != tuple(backgrounds):
            raise ValueError("relation parse bundles do not match ordered backgrounds")
        expected_row = _relation_parse_bundle_row(
            task=tasks[task_id],
            pair_role=role,
            pair_index=pair_index,
            grid=expected[key],
        )
        if row != expected_row:
            raise ValueError(
                "relation parse sidecar row does not exactly replay from its blind grid"
            )
    if observed != set(expected):
        raise ValueError(
            "relation parse sidecar does not cover every observable grid exactly once"
        )
    if observed_order != list(expected):
        raise ValueError("relation parse sidecar rows are not in canonical order")


def _validate_candidates_against_programs(
    *,
    tasks: Mapping[str, BlindTask],
    candidate_groups: Mapping[
        tuple[str, int], list[tuple[int, CandidateRecord]]
    ],
    program_groups: Mapping[str, list[dict[str, Any]]],
    search_by_task: Mapping[str, dict[str, Any]],
) -> None:
    expected_groups: dict[
        tuple[str, int], list[tuple[int, dict[str, Any], Grid]]
    ] = defaultdict(list)
    for task_id, task in tasks.items():
        for row in program_groups.get(task_id, []):
            outputs = _program_query_outputs(row, len(task.test_inputs))
            for test_index, output in enumerate(outputs):
                if output is not None:
                    expected_groups[(task_id, test_index)].append(
                        (row["retained_rank"], row["program"], output)
                    )
    all_keys = set(expected_groups) | set(candidate_groups)
    for key in all_keys:
        expected = expected_groups.get(key, [])
        actual = candidate_groups.get(key, [])
        if len(actual) != len(expected):
            raise ValueError("candidate count does not match replayable retained programs")
        task = tasks[key[0]]
        for (wrapper_rank, candidate), (program_rank, program_json, output) in zip(
            actual, expected
        ):
            if wrapper_rank < 0:
                raise ValueError("candidate wrapper rank must be non-negative")
            if candidate.program_hash != program_json["program_id"] or candidate.output != output:
                raise ValueError("candidate output does not replay from its program_hash")
            parameters = json.loads(candidate.generation_parameters_json)
            if (
                parameters.get("blind_content_sha256") != task.blind_content_sha256
                or parameters.get("emission_rank") != program_rank
                or parameters.get("program") != program_json
                or parameters.get("search_config")
                != search_by_task[key[0]]["search_config"]
            ):
                raise ValueError("candidate provenance does not match its pool program")


def _replay_search_artifacts(
    *,
    tasks: Mapping[str, BlindTask],
    pool_rows: Mapping[str, list[dict[str, Any]]],
    pool_config: object,
) -> None:
    config = _require_fields(
        pool_config,
        {
            "search",
            "dsl_semantics_version",
            "shape_proposer_semantics_version",
            "parser_semantics_version",
            "parser_stage",
            "panel_parser_semantics_version",
            "panel_parser_stage",
            "panel_sequence_d4_semantics_version",
            "panel_sequence_d4_stage",
            "panel_sequence_d4_proposer_semantics_version",
            "panel_lattice_periodic_semantics_version",
            "panel_lattice_periodic_stage",
            "panel_lattice_periodic_proposer_semantics_version",
            "relation_parser_semantics_version",
            "relation_parser_stage",
            "bbox_contact_action_semantics_version",
            "bbox_contact_action_stage",
            "bbox_contact_proposer_semantics_version",
            "oracle_inputs_available_to_pool",
        },
        label="pool config",
    )
    if (
        config["dsl_semantics_version"] != DSL_SEMANTICS_VERSION
        or config["shape_proposer_semantics_version"]
        != SHAPE_PROPOSER_SEMANTICS_VERSION
        or config["parser_semantics_version"] != PARSER_SEMANTICS_VERSION
        or config["parser_stage"] != PARSER_STAGE
        or config["panel_parser_semantics_version"]
        != PANEL_PARSER_SEMANTICS_VERSION
        or config["panel_parser_stage"] != PANEL_PARSER_STAGE
        or config["panel_sequence_d4_semantics_version"]
        != PANEL_SEQUENCE_D4_SEMANTICS_VERSION
        or config["panel_sequence_d4_stage"] != PANEL_SEQUENCE_D4_STAGE
        or config["panel_sequence_d4_proposer_semantics_version"]
        != PANEL_SEQUENCE_D4_PROPOSER_SEMANTICS_VERSION
        or config["panel_lattice_periodic_semantics_version"]
        != PANEL_LATTICE_PERIODIC_SEMANTICS_VERSION
        or config["panel_lattice_periodic_stage"]
        != PANEL_LATTICE_PERIODIC_STAGE
        or config["panel_lattice_periodic_proposer_semantics_version"]
        != PANEL_LATTICE_PERIODIC_PROPOSER_SEMANTICS_VERSION
        or config["relation_parser_semantics_version"]
        != RELATION_PARSER_SEMANTICS_VERSION
        or config["relation_parser_stage"] != RELATION_PARSER_STAGE
        or config["bbox_contact_action_semantics_version"]
        != BBOX_CONTACT_ACTION_SEMANTICS_VERSION
        or config["bbox_contact_action_stage"] != BBOX_CONTACT_ACTION_STAGE
        or config["bbox_contact_proposer_semantics_version"]
        != BBOX_CONTACT_PROPOSER_SEMANTICS_VERSION
        or config["oracle_inputs_available_to_pool"] is not False
    ):
        raise ValueError("pool config does not match the executable blind semantics")
    search_payload = _require_fields(
        config["search"],
        {"max_depth", "beam_width", "max_instruction_options", "max_exact_programs"},
        label="search config",
    )
    search_config = SearchConfig(**search_payload)
    expected_program_rows: list[dict[str, object]] = []
    expected_candidate_rows: list[dict[str, object]] = []
    expected_search_rows: list[dict[str, object]] = []
    for task in tasks.values():
        result = search_programs(task, config=search_config)
        retained_ids: list[str] = []
        for rank, evaluation in enumerate(result.exact_evaluations):
            retained_ids.append(evaluation.program.program_id)
            expected_program_rows.append(
                {
                    "task_id": task.task_id,
                    "retained_rank": rank,
                    "program": evaluation.program.to_json_dict(),
                    "all_demo_exact": True,
                    "query_outcomes": [
                        _outcome_json(index, outcome)
                        for index, outcome in enumerate(evaluation.query_outcomes)
                    ],
                }
            )
        ranks: defaultdict[int, int] = defaultdict(int)
        for candidate in dsl_candidates(task, search_result=result):
            expected_candidate_rows.append(
                {
                    "emission_rank": ranks[candidate.test_index],
                    "candidate": candidate.to_json_dict(),
                }
            )
            ranks[candidate.test_index] += 1
        expected_search_rows.append(
            {
                "task_id": task.task_id,
                "blind_content_sha256": task.blind_content_sha256,
                "search_config": asdict(result.config),
                "shape_proposals": [
                    item.to_json_dict() for item in result.shape_proposals
                ],
                "panel_instruction_proposals": [
                    item.to_json_dict()
                    for item in result.panel_instruction_proposals
                ],
                "panel_sequence_d4_instruction_proposals": [
                    item.to_json_dict()
                    for item in result.panel_sequence_d4_instruction_proposals
                ],
                "panel_sequence_d4_trial_count": (
                    result.panel_sequence_d4_trial_count
                ),
                "panel_sequence_d4_demo_execution_count": (
                    result.panel_sequence_d4_demo_execution_count
                ),
                "panel_lattice_periodic_period_bounds": [
                    item.to_json_dict()
                    for item in result.panel_lattice_periodic_period_bounds
                ],
                "panel_lattice_periodic_instruction_proposals": [
                    item.to_json_dict()
                    for item in result.panel_lattice_periodic_instruction_proposals
                ],
                "panel_lattice_periodic_structural_check_count": (
                    result.panel_lattice_periodic_structural_check_count
                ),
                "panel_lattice_periodic_trial_count": (
                    result.panel_lattice_periodic_trial_count
                ),
                "panel_lattice_periodic_demo_execution_count": (
                    result.panel_lattice_periodic_demo_execution_count
                ),
                "bbox_contact_bounds": [
                    item.to_json_dict() for item in result.bbox_contact_bounds
                ],
                "bbox_contact_instruction_proposals": [
                    item.to_json_dict()
                    for item in result.bbox_contact_instruction_proposals
                ],
                "bbox_contact_structural_check_count": (
                    result.bbox_contact_structural_check_count
                ),
                "bbox_contact_anchor_candidate_count": (
                    result.bbox_contact_anchor_candidate_count
                ),
                "bbox_contact_relation_check_count": (
                    result.bbox_contact_relation_check_count
                ),
                "bbox_contact_admissible_binding_count": (
                    result.bbox_contact_admissible_binding_count
                ),
                "bbox_contact_action_trial_count": (
                    result.bbox_contact_action_trial_count
                ),
                "bbox_contact_demo_execution_count": (
                    result.bbox_contact_demo_execution_count
                ),
                "instruction_option_count_pre_cap": (
                    result.instruction_option_count_pre_cap
                ),
                "instruction_option_count_post_cap": (
                    result.instruction_option_count_post_cap
                ),
                "instruction_option_truncation_count": (
                    result.instruction_option_truncation_count
                ),
                "instruction_options": [
                    item.to_json_dict() for item in result.instruction_options
                ],
                "expansions": result.expansions,
                "program_execution_count": result.expansions
                * (len(task.train) + len(task.test_inputs)),
                "semantic_duplicates": result.semantic_duplicates,
                "first_exact_expansion": result.first_exact_expansion,
                "evaluated_program_ids": list(result.evaluated_program_ids),
                "retained_exact_program_ids": retained_ids,
                "candidate_records": sum(ranks.values()),
            }
        )
    if pool_rows["exact_programs.jsonl"] != expected_program_rows:
        raise ValueError("exact program sidecar does not replay from blind search")
    if pool_rows["candidates.jsonl"] != expected_candidate_rows:
        raise ValueError("candidate sidecar does not replay from blind search")
    actual_search_rows = [
        {
            key: value
            for key, value in row.items()
            if key not in {"wall_time_ns", "process_time_ns", "tracemalloc_peak_bytes"}
        }
        for row in pool_rows["search.jsonl"]
    ]
    if actual_search_rows != expected_search_rows:
        raise ValueError("search ledger does not replay from blind tasks and config")


@dataclass(frozen=True, slots=True)
class ValidatedBlindPool:
    blind_snapshot: BundleSnapshot
    pool_snapshot: BundleSnapshot
    blind_manifest: dict[str, Any]
    pool_manifest: dict[str, Any]
    tasks: dict[str, BlindTask]
    pool_rows: dict[str, list[dict[str, Any]]]
    candidate_groups: dict[tuple[str, int], list[tuple[int, CandidateRecord]]]
    all_candidates: list[CandidateRecord]
    program_groups: dict[str, list[dict[str, Any]]]
    search_by_task: dict[str, dict[str, Any]]


def _validated_blind_pool(
    *,
    blind_dataset_dir: str | Path,
    pool_dir: str | Path,
    source_capture: RuntimeSourceCapture,
) -> ValidatedBlindPool:
    blind_snapshot = snapshot_bundle(blind_dataset_dir)
    pool_snapshot = snapshot_bundle(pool_dir)
    blind_manifest, tasks_tuple = _load_blind_dataset(blind_snapshot)
    pool_manifest, pool_rows = _load_pool(pool_snapshot)
    expected_source = source_capture.fingerprint_sha256
    if blind_manifest["runtime_source_fingerprint_sha256"] != expected_source or pool_manifest[
        "runtime_source_fingerprint_sha256"
    ] != expected_source:
        raise ValueError("blind dataset, pool, and evaluator source fingerprints differ")
    tasks = {task.task_id: task for task in tasks_tuple}
    task_ids = set(tasks)
    if pool_manifest["case_set_id"] != blind_manifest["case_set_id"]:
        raise ValueError("pool and blind case_set_id differ")
    if pool_manifest["parent_blind_bundle"] != {
        "artifact_manifest_sha256": blind_snapshot.artifact_manifest_sha256,
        "blind_tasks_sha256": blind_manifest["blind_tasks_sha256"],
    }:
        raise ValueError("pool does not bind the supplied blind bundle")
    expected_pool_spec = _content_id(
        "afts-e01a-pool-spec/v6",
        {
            "case_set_id": blind_manifest["case_set_id"],
            "blind_tasks_sha256": blind_manifest["blind_tasks_sha256"],
            "runtime_source_fingerprint_sha256": expected_source,
            "config": pool_manifest["config"],
        },
    )
    if pool_manifest["pool_spec_id"] != expected_pool_spec:
        raise ValueError("pool_spec_id does not match its scientific inputs")
    behavior_hashes = _pool_behavior_hashes(
        task_bytes=pool_snapshot.artifacts["tasks.jsonl"],
        parse_bytes=pool_snapshot.artifacts["parses.jsonl"],
        panel_parse_bytes=pool_snapshot.artifacts["panel_parses.jsonl"],
        relation_parse_bytes=pool_snapshot.artifacts["relation_parses.jsonl"],
        program_bytes=pool_snapshot.artifacts["exact_programs.jsonl"],
        candidate_bytes=pool_snapshot.artifacts["candidates.jsonl"],
        search_rows=pool_rows["search.jsonl"],
    )
    if pool_manifest["pool_content_id"] != _content_id(
        "afts-e01a-pool-content/v6",
        {"pool_spec_id": expected_pool_spec, "behavior_hashes": behavior_hashes},
    ):
        raise ValueError("pool_content_id does not match frozen behavior")
    expected_pool_summary = _pool_summary_payload(
        blind_manifest=blind_manifest,
        pool_spec_id=pool_manifest["pool_spec_id"],
        pool_content_id=pool_manifest["pool_content_id"],
        task_rows=pool_rows["tasks.jsonl"],
        parse_rows=pool_rows["parses.jsonl"],
        panel_parse_rows=pool_rows["panel_parses.jsonl"],
        relation_parse_rows=pool_rows["relation_parses.jsonl"],
        program_rows=pool_rows["exact_programs.jsonl"],
        candidate_rows=pool_rows["candidates.jsonl"],
        search_rows=pool_rows["search.jsonl"],
    )
    actual_pool_summary = _strict_json(
        pool_snapshot.artifacts["pool_summary.json"], label="pool_summary.json"
    )
    if actual_pool_summary != expected_pool_summary:
        raise ValueError("pool summary does not recompute exactly from frozen rows")
    if pool_rows["tasks.jsonl"] != _strict_jsonl(
        blind_snapshot.artifacts["blind_tasks.jsonl"], label="blind_tasks.jsonl"
    ):
        raise ValueError("pool task rows are not identical to blind task rows")
    if {row.get("task_id") for row in pool_rows["tasks.jsonl"]} != task_ids:
        raise ValueError("pool task set differs from blind task set")
    _validate_parse_rows(tasks, pool_rows["parses.jsonl"])
    _validate_panel_parse_rows(tasks, pool_rows["panel_parses.jsonl"])
    _validate_relation_parse_rows(tasks, pool_rows["relation_parses.jsonl"])
    _replay_search_artifacts(
        tasks=tasks,
        pool_rows=pool_rows,
        pool_config=pool_manifest["config"],
    )
    candidate_groups, all_candidates = _candidate_groups(pool_rows["candidates.jsonl"])
    program_groups = _program_groups(pool_rows["exact_programs.jsonl"])
    search_by_task = _search_by_task(pool_rows["search.jsonl"])
    if set(search_by_task) != task_ids or not set(program_groups).issubset(task_ids):
        raise ValueError("program/search sidecar task sets differ from blind tasks")
    for candidate in all_candidates:
        task = tasks.get(candidate.task_id)
        if task is None or candidate.test_index >= len(task.test_inputs):
            raise ValueError("candidate references an unknown task or query index")
        parameters = json.loads(candidate.generation_parameters_json)
        if parameters.get("blind_content_sha256") != task.blind_content_sha256:
            raise ValueError("candidate does not bind its ordered blind task content")
    for task_id in task_ids:
        retained = [
            row["program"]["program_id"] for row in program_groups.get(task_id, [])
        ]
        if retained != search_by_task[task_id]["retained_exact_program_ids"]:
            raise ValueError("retained search ledger and program sidecar disagree")
        task = tasks[task_id]
        for row in program_groups.get(task_id, []):
            program = row["_program_object"]
            for pair in task.train:
                outcome = execute_program(program, pair.input)
                if outcome.invalid_code is InvalidCode.INTERNAL_ERROR:
                    raise RuntimeError("retained program replay encountered an internal error")
                if not outcome.ok or outcome.output != pair.output:
                    raise ValueError("retained program is not demo exact on replay")
            replayed = [
                _outcome_json(index, execute_program(program, grid))
                for index, grid in enumerate(task.test_inputs)
            ]
            if any(
                item["invalid_code"] == InvalidCode.INTERNAL_ERROR.value for item in replayed
            ):
                raise RuntimeError("retained query replay encountered an internal error")
            if replayed != row["query_outcomes"]:
                raise ValueError("retained query outcomes do not replay exactly")
    retained_keys = {
        (task_id, row["program"]["program_id"])
        for task_id, rows in program_groups.items()
        for row in rows
    }
    if any(
        (candidate.task_id, candidate.program_hash) not in retained_keys
        for candidate in all_candidates
    ):
        raise ValueError("candidate references a non-retained program")
    _validate_candidates_against_programs(
        tasks=tasks,
        candidate_groups=candidate_groups,
        program_groups=program_groups,
        search_by_task=search_by_task,
    )
    return ValidatedBlindPool(
        blind_snapshot=blind_snapshot,
        pool_snapshot=pool_snapshot,
        blind_manifest=blind_manifest,
        pool_manifest=pool_manifest,
        tasks=tasks,
        pool_rows=pool_rows,
        candidate_groups=candidate_groups,
        all_candidates=all_candidates,
        program_groups=program_groups,
        search_by_task=search_by_task,
    )


def _aggregate_task_metrics(rows: list[dict[str, Any]]) -> dict[str, object]:
    if not rows:
        raise ValueError("cannot aggregate an empty task metric set")
    task_count = len(rows)
    pair_count = sum(int(row["query_pair_count"]) for row in rows)
    return {
        "task_count": task_count,
        "query_pair_count": pair_count,
        "task_first_pair_coverage_by_k": {
            str(k): sum(row["pair_fraction_by_k"][str(k)] for row in rows) / task_count
            for k in OUTPUT_CUTOFFS
        },
        "micro_pair_coverage_by_k": {
            str(k): sum(
                row["pair_fraction_by_k"][str(k)] * row["query_pair_count"]
                for row in rows
            )
            / pair_count
            for k in OUTPUT_CUTOFFS
        },
        "strict_task_coverage_by_k": {
            str(k): sum(row["strict_covered_by_k"][str(k)] for row in rows) / task_count
            for k in OUTPUT_CUTOFFS
        },
        "generator_program_evaluated_rate": sum(
            bool(row["generator_program_evaluated"]) for row in rows
        )
        / task_count,
        "generator_program_retained_rate": sum(
            row["generator_program_first_rank"] is not None for row in rows
        )
        / task_count,
        "generator_program_recall_by_k": {
            str(k): sum(row["generator_program_recall_by_k"][str(k)] for row in rows)
            / task_count
            for k in PROGRAM_CUTOFFS
        },
        "semantic_program_coverage_by_k": {
            str(k): sum(row["semantic_program_coverage_by_k"][str(k)] for row in rows)
            / task_count
            for k in PROGRAM_CUTOFFS
        },
        "generator_replay_control_rate": sum(
            bool(row["generator_replays_all_pairs"]) for row in rows
        )
        / task_count,
    }


def _recompute_evaluation_rows(
    *,
    context: ValidatedBlindPool,
    oracle_by_task: Mapping[str, dict[str, Any]],
    dataset_kind: str,
) -> tuple[list[dict[str, object]], list[dict[str, Any]]]:
    if set(oracle_by_task) != set(context.tasks):
        raise ValueError("oracle and blind task sets differ")
    pair_rows: list[dict[str, object]] = []
    task_rows: list[dict[str, Any]] = []
    for task_id in sorted(context.tasks):
        task = context.tasks[task_id]
        oracle = oracle_by_task[task_id]
        if len(oracle["test"]) != len(task.test_inputs):
            raise ValueError("oracle query indices do not match blind query inputs")
        oracle_outputs = tuple(as_grid(row["output"]) for row in oracle["test"])
        pair_vectors: list[dict[str, bool]] = []
        total_unique = 0
        for test_index, oracle_output in enumerate(oracle_outputs):
            ordered = context.candidate_groups.get((task_id, test_index), [])
            unique: list[CandidateRecord] = []
            seen: set[str] = set()
            for _, candidate in ordered:
                if candidate.output_key not in seen:
                    seen.add(candidate.output_key)
                    unique.append(candidate)
            total_unique += len(unique)
            first_hit = next(
                (
                    index + 1
                    for index, candidate in enumerate(unique)
                    if candidate.output == oracle_output
                ),
                None,
            )
            hits = {
                str(k): first_hit is not None and first_hit <= k for k in OUTPUT_CUTOFFS
            }
            pair_vectors.append(hits)
            pair_row: dict[str, object] = {
                "task_id": task_id,
                "test_index": test_index,
                "oracle_output_key": grid_key(oracle_output),
                "raw_candidate_count": len(ordered),
                "unique_output_count": len(unique),
                "first_hit_unique_rank": first_hit,
                "covered_by_k": hits,
            }
            if dataset_kind == "synthetic_control":
                pair_row.update(
                    {"family": oracle["family"], "stratum": oracle["stratum"]}
                )
            elif dataset_kind == "arc_public_training_development_smoke":
                pair_row["source_task_id"] = oracle["source_task_id"]
            else:
                raise ValueError("unsupported evaluation dataset_kind")
            pair_rows.append(pair_row)

        semantic_first: int | None = None
        for row in context.program_groups.get(task_id, []):
            outputs = _program_query_outputs(row, len(oracle_outputs))
            if semantic_first is None and outputs == oracle_outputs:
                semantic_first = int(row["retained_rank"]) + 1
        common_task: dict[str, Any] = {
            "task_id": task_id,
            "query_pair_count": len(oracle_outputs),
            "raw_candidate_count": sum(
                len(context.candidate_groups.get((task_id, index), []))
                for index in range(len(oracle_outputs))
            ),
            "unique_output_count": total_unique,
            "pair_fraction_by_k": {
                str(k): sum(vector[str(k)] for vector in pair_vectors)
                / len(pair_vectors)
                for k in OUTPUT_CUTOFFS
            },
            "strict_covered_by_k": {
                str(k): all(vector[str(k)] for vector in pair_vectors)
                for k in OUTPUT_CUTOFFS
            },
            "semantic_solving_program_first_rank": semantic_first,
            "semantic_program_coverage_by_k": {
                str(k): semantic_first is not None and semantic_first <= k
                for k in PROGRAM_CUTOFFS
            },
        }
        if dataset_kind == "synthetic_control":
            generator = Program.from_json_dict(oracle["generator_program"])
            replay_ok = True
            for pair in task.train:
                outcome = execute_program(generator, pair.input)
                if outcome.invalid_code is InvalidCode.INTERNAL_ERROR:
                    raise RuntimeError("generator replay encountered an internal error")
                replay_ok &= outcome.ok and outcome.output == pair.output
            for index, grid in enumerate(task.test_inputs):
                outcome = execute_program(generator, grid)
                if outcome.invalid_code is InvalidCode.INTERNAL_ERROR:
                    raise RuntimeError("generator replay encountered an internal error")
                replay_ok &= outcome.ok and outcome.output == oracle_outputs[index]
            generator_first = next(
                (
                    int(row["retained_rank"]) + 1
                    for row in context.program_groups.get(task_id, [])
                    if row["program"]["program_id"] == generator.program_id
                ),
                None,
            )
            common_task.update(
                {
                    "family": oracle["family"],
                    "stratum": oracle["stratum"],
                    "generator_program_id": generator.program_id,
                    "generator_program_evaluated": generator.program_id
                    in context.search_by_task[task_id]["evaluated_program_ids"],
                    "generator_program_first_rank": generator_first,
                    "generator_program_recall_by_k": {
                        str(k): generator_first is not None and generator_first <= k
                        for k in PROGRAM_CUTOFFS
                    },
                    "generator_replays_all_pairs": replay_ok,
                }
            )
        else:
            common_task["source_task_id"] = oracle["source_task_id"]
        task_rows.append(common_task)
    return pair_rows, task_rows


def evaluate_symbolic_pool_bundle(
    *,
    blind_dataset_dir: str | Path,
    oracle_dataset_dir: str | Path,
    pool_dir: str | Path,
    output_dir: str | Path,
    source_capture: RuntimeSourceCapture,
    test_source_sha256: str | None,
    source_loader: Mapping[str, object],
    command: list[str],
) -> dict[str, object]:
    target = Path(output_dir).expanduser().resolve()
    _require_nonoverlapping_paths(
        blind=blind_dataset_dir,
        oracle=oracle_dataset_dir,
        pool=pool_dir,
        output=target,
    )
    if target.exists():
        raise FileExistsError(f"refusing to overwrite evidence directory: {target}")
    validated_pool = _validated_blind_pool(
        blind_dataset_dir=blind_dataset_dir,
        pool_dir=pool_dir,
        source_capture=source_capture,
    )
    blind_snapshot = validated_pool.blind_snapshot
    oracle_snapshot = snapshot_bundle(oracle_dataset_dir)
    pool_snapshot = validated_pool.pool_snapshot
    blind_manifest = validated_pool.blind_manifest
    tasks_tuple = tuple(
        validated_pool.tasks[task_id] for task_id in sorted(validated_pool.tasks)
    )
    oracle_manifest, oracle_by_task = _load_oracle_dataset(oracle_snapshot)
    pool_manifest = validated_pool.pool_manifest
    pool_rows = validated_pool.pool_rows
    expected_source = source_capture.fingerprint_sha256
    if any(
        manifest["runtime_source_fingerprint_sha256"] != expected_source
        for manifest in (blind_manifest, oracle_manifest, pool_manifest)
    ):
        raise ValueError(
            "dataset, pool, and evaluator must use the same runtime source fingerprint"
        )
    tasks = {task.task_id: task for task in tasks_tuple}
    task_ids = set(tasks)
    if set(oracle_by_task) != task_ids:
        raise ValueError("blind and oracle task sets differ")
    if any(
        len(oracle_by_task[task_id]["test"]) != len(tasks[task_id].test_inputs)
        for task_id in task_ids
    ):
        raise ValueError("oracle query indices do not match blind query inputs")
    if oracle_manifest["case_set_id"] != blind_manifest["case_set_id"]:
        raise ValueError("oracle and blind case_set_id differ")
    if pool_manifest["case_set_id"] != blind_manifest["case_set_id"]:
        raise ValueError("pool and blind case_set_id differ")
    parent = pool_manifest["parent_blind_bundle"]
    if parent != {
        "artifact_manifest_sha256": blind_snapshot.artifact_manifest_sha256,
        "blind_tasks_sha256": blind_manifest["blind_tasks_sha256"],
    }:
        raise ValueError("pool does not bind the supplied blind bundle")
    expected_pool_spec = _content_id(
        "afts-e01a-pool-spec/v6",
        {
            "case_set_id": blind_manifest["case_set_id"],
            "blind_tasks_sha256": blind_manifest["blind_tasks_sha256"],
            "runtime_source_fingerprint_sha256": pool_manifest[
                "runtime_source_fingerprint_sha256"
            ],
            "config": pool_manifest["config"],
        },
    )
    if pool_manifest["pool_spec_id"] != expected_pool_spec:
        raise ValueError("pool_spec_id does not match its scientific inputs")
    pool_behavior_hashes = _pool_behavior_hashes(
        task_bytes=pool_snapshot.artifacts["tasks.jsonl"],
        parse_bytes=pool_snapshot.artifacts["parses.jsonl"],
        panel_parse_bytes=pool_snapshot.artifacts["panel_parses.jsonl"],
        relation_parse_bytes=pool_snapshot.artifacts["relation_parses.jsonl"],
        program_bytes=pool_snapshot.artifacts["exact_programs.jsonl"],
        candidate_bytes=pool_snapshot.artifacts["candidates.jsonl"],
        search_rows=pool_rows["search.jsonl"],
    )
    expected_pool_content = _content_id(
        "afts-e01a-pool-content/v6",
        {
            "pool_spec_id": expected_pool_spec,
            "behavior_hashes": pool_behavior_hashes,
        },
    )
    if pool_manifest["pool_content_id"] != expected_pool_content:
        raise ValueError("pool_content_id does not match frozen pool behavior")
    if pool_rows["tasks.jsonl"] != _strict_jsonl(
        blind_snapshot.artifacts["blind_tasks.jsonl"], label="blind_tasks.jsonl"
    ):
        raise ValueError("pool task rows are not identical to the blind dataset rows")
    pool_task_ids = {row.get("task_id") for row in pool_rows["tasks.jsonl"]}
    if pool_task_ids != task_ids:
        raise ValueError("pool task set differs from blind task set")
    _validate_parse_rows(tasks, pool_rows["parses.jsonl"])
    _validate_panel_parse_rows(tasks, pool_rows["panel_parses.jsonl"])
    _validate_relation_parse_rows(tasks, pool_rows["relation_parses.jsonl"])

    candidate_groups, all_candidates = _candidate_groups(pool_rows["candidates.jsonl"])
    program_groups = _program_groups(pool_rows["exact_programs.jsonl"])
    search_by_task = _search_by_task(pool_rows["search.jsonl"])
    if set(search_by_task) != task_ids:
        raise ValueError("search ledger task set differs from blind task set")
    for candidate in all_candidates:
        task = tasks.get(candidate.task_id)
        if task is None or candidate.test_index >= len(task.test_inputs):
            raise ValueError("candidate references an unknown task or query index")
        parameters = json.loads(candidate.generation_parameters_json)
        if parameters.get("blind_content_sha256") != task.blind_content_sha256:
            raise ValueError("candidate does not bind ordered blind task content")
    for task_id in task_ids:
        retained = [
            row["program"]["program_id"] for row in program_groups.get(task_id, [])
        ]
        if retained != search_by_task[task_id]["retained_exact_program_ids"]:
            raise ValueError("program sidecar and retained search ledger disagree")
        task = tasks[task_id]
        for row in program_groups.get(task_id, []):
            program = row["_program_object"]
            for pair in task.train:
                outcome = execute_program(program, pair.input)
                if outcome.invalid_code is InvalidCode.INTERNAL_ERROR:
                    raise RuntimeError("retained program replay encountered an internal error")
                if not outcome.ok or outcome.output != pair.output:
                    raise ValueError("retained program does not replay as demo exact")
            recorded_query = row["query_outcomes"]
            replayed_query = [
                _outcome_json(index, execute_program(program, grid))
                for index, grid in enumerate(task.test_inputs)
            ]
            if any(
                item["invalid_code"] == InvalidCode.INTERNAL_ERROR.value
                for item in replayed_query
            ):
                raise RuntimeError("retained query replay encountered an internal error")
            if recorded_query != replayed_query:
                raise ValueError("retained program query trace does not replay exactly")
    retained_keys = {
        (task_id, row["program"]["program_id"])
        for task_id, rows in program_groups.items()
        for row in rows
    }
    if any((candidate.task_id, candidate.program_hash) not in retained_keys for candidate in all_candidates):
        raise ValueError("candidate references a non-retained program")

    pair_rows: list[dict[str, object]] = []
    task_metric_rows: list[dict[str, Any]] = []
    for task_id in sorted(task_ids):
        task = tasks[task_id]
        oracle = oracle_by_task[task_id]
        oracle_outputs = tuple(as_grid(row["output"]) for row in oracle["test"])
        gt_program = Program.from_json_dict(oracle["generator_program"])
        replay_ok = True
        for pair in task.train:
            outcome = execute_program(gt_program, pair.input)
            if outcome.invalid_code is InvalidCode.INTERNAL_ERROR:
                raise RuntimeError("generator replay encountered a DSL internal error")
            replay_ok &= outcome.ok and outcome.output == pair.output
        for index, grid in enumerate(task.test_inputs):
            outcome = execute_program(gt_program, grid)
            if outcome.invalid_code is InvalidCode.INTERNAL_ERROR:
                raise RuntimeError("generator replay encountered a DSL internal error")
            replay_ok &= outcome.ok and outcome.output == oracle_outputs[index]

        pair_hit_vectors: list[dict[str, bool]] = []
        unique_count = 0
        for test_index, oracle_output in enumerate(oracle_outputs):
            ordered = candidate_groups.get((task_id, test_index), [])
            seen: set[str] = set()
            unique: list[CandidateRecord] = []
            for _, candidate in ordered:
                if candidate.output_key not in seen:
                    seen.add(candidate.output_key)
                    unique.append(candidate)
            unique_count += len(unique)
            hits = {
                str(k): any(candidate.output == oracle_output for candidate in unique[:k])
                for k in OUTPUT_CUTOFFS
            }
            first_hit = next(
                (index + 1 for index, candidate in enumerate(unique) if candidate.output == oracle_output),
                None,
            )
            pair_hit_vectors.append(hits)
            pair_rows.append(
                {
                    "task_id": task_id,
                    "test_index": test_index,
                    "family": oracle["family"],
                    "stratum": oracle["stratum"],
                    "oracle_output_key": grid_key(oracle_output),
                    "raw_candidate_count": len(ordered),
                    "unique_output_count": len(unique),
                    "first_hit_unique_rank": first_hit,
                    "covered_by_k": hits,
                }
            )

        programs = program_groups.get(task_id, [])
        generator_first_rank: int | None = None
        semantic_first_rank: int | None = None
        for row in programs:
            one_based_rank = int(row["retained_rank"]) + 1
            if row["program"]["program_id"] == gt_program.program_id:
                generator_first_rank = one_based_rank
            outputs = _program_query_outputs(row, len(oracle_outputs))
            if semantic_first_rank is None and outputs == oracle_outputs:
                semantic_first_rank = one_based_rank
            for test_index, output in enumerate(outputs):
                if output is None:
                    continue
                matching = [
                    candidate
                    for _, candidate in candidate_groups.get((task_id, test_index), [])
                    if candidate.program_hash == row["program"]["program_id"]
                ]
                if not any(candidate.output == output for candidate in matching):
                    raise ValueError("candidate sidecar cannot replay a program query output")
        pair_fraction = {
            str(k): sum(vector[str(k)] for vector in pair_hit_vectors) / len(pair_hit_vectors)
            for k in OUTPUT_CUTOFFS
        }
        strict = {
            str(k): all(vector[str(k)] for vector in pair_hit_vectors)
            for k in OUTPUT_CUTOFFS
        }
        task_metric_rows.append(
            {
                "task_id": task_id,
                "family": oracle["family"],
                "stratum": oracle["stratum"],
                "query_pair_count": len(oracle_outputs),
                "raw_candidate_count": sum(
                    len(candidate_groups.get((task_id, index), []))
                    for index in range(len(oracle_outputs))
                ),
                "unique_output_count": unique_count,
                "pair_fraction_by_k": pair_fraction,
                "strict_covered_by_k": strict,
                "generator_program_id": gt_program.program_id,
                "generator_program_evaluated": gt_program.program_id
                in search_by_task[task_id]["evaluated_program_ids"],
                "generator_program_first_rank": generator_first_rank,
                "generator_program_recall_by_k": {
                    str(k): generator_first_rank is not None and generator_first_rank <= k
                    for k in PROGRAM_CUTOFFS
                },
                "semantic_solving_program_first_rank": semantic_first_rank,
                "semantic_program_coverage_by_k": {
                    str(k): semantic_first_rank is not None and semantic_first_rank <= k
                    for k in PROGRAM_CUTOFFS
                },
                "generator_replays_all_pairs": replay_ok,
            }
        )

    metrics = _aggregate_task_metrics(task_metric_rows)
    family_metrics = {
        family: _aggregate_task_metrics(
            [row for row in task_metric_rows if row["family"] == family]
        )
        for family in sorted({row["family"] for row in task_metric_rows})
    }
    stratum_metrics = {
        stratum: _aggregate_task_metrics(
            [row for row in task_metric_rows if row["stratum"] == stratum]
        )
        for stratum in sorted({row["stratum"] for row in task_metric_rows})
    }
    if metrics["generator_replay_control_rate"] != 1.0:
        raise RuntimeError("generator replay positive control did not reach 100%")
    funnel = {
        "tasks": len(task_metric_rows),
        **_representation_funnel_counts(pool_rows),
        "shape_proposals": sum(
            len(row["shape_proposals"]) for row in search_by_task.values()
        ),
        "instruction_options": sum(
            len(row["instruction_options"]) for row in search_by_task.values()
        ),
        "evaluated_programs": sum(
            len(row["evaluated_program_ids"]) for row in search_by_task.values()
        ),
        "semantic_duplicates": sum(
            row["semantic_duplicates"] for row in search_by_task.values()
        ),
        "retained_exact_programs": sum(len(rows) for rows in program_groups.values()),
        "candidate_records": len(all_candidates),
        "unique_candidate_outputs": len(
            {(item.task_id, item.test_index, item.output_key) for item in all_candidates}
        ),
        "program_execution_count": sum(
            row["program_execution_count"] for row in search_by_task.values()
        ),
    }
    integrity = {
        "all_checks_pass": True,
        "closed_world_bundles_verified": True,
        "artifact_hashes_verified_from_captured_bytes": True,
        "case_set_ids_match": True,
        "pool_binds_exact_blind_artifact_manifest": True,
        "pool_tasks_equal_blind_tasks": True,
        "pool_oracle_keys_absent": True,
        "candidate_ranks_contiguous": True,
        "program_and_candidate_parent_closure": True,
        "expansion_ledgers_match_program_ids": True,
        "generator_replay_control_100_percent": True,
        "panel_parse_sidecar_replayed": True,
        "panel_sequence_d4_proposals_replayed": True,
        "panel_sequence_d4_cost_ledger_replayed": True,
        "panel_lattice_periodic_bounds_replayed": True,
        "panel_lattice_periodic_proposals_replayed": True,
        "panel_lattice_periodic_cost_ledger_replayed": True,
        "relation_parse_sidecar_replayed": True,
        "bbox_contact_bounds_replayed": True,
        "bbox_contact_proposals_replayed": True,
        "bbox_contact_cost_ledger_replayed": True,
        "bbox_contact_cap_survival_replayed": True,
        "instruction_cap_ledger_replayed": True,
    }
    started_at = utc_now()
    eval_spec_id = _content_id(
        "afts-e01a-eval-spec/v6",
        {
            "case_set_id": blind_manifest["case_set_id"],
            "oracle_set_id": oracle_manifest["oracle_set_id"],
            "pool_content_id": pool_manifest["pool_content_id"],
            "runtime_source_fingerprint_sha256": source_capture.fingerprint_sha256,
            "output_cutoffs": OUTPUT_CUTOFFS,
            "program_cutoffs": PROGRAM_CUTOFFS,
        },
    )
    completed_at = utc_now()
    eval_manifest = {
        "schema": EVAL_MANIFEST_SCHEMA,
        "evidence_status": "post_hoc_oracle_evaluation_of_frozen_symbolic_pool",
        "dataset_kind": "synthetic_control",
        "eval_spec_id": eval_spec_id,
        "case_set_id": blind_manifest["case_set_id"],
        "oracle_set_id": oracle_manifest["oracle_set_id"],
        "pool_spec_id": pool_manifest["pool_spec_id"],
        "pool_content_id": pool_manifest["pool_content_id"],
        "parent_artifact_manifests": {
            "blind_sha256": blind_snapshot.artifact_manifest_sha256,
            "oracle_sha256": oracle_snapshot.artifact_manifest_sha256,
            "pool_sha256": pool_snapshot.artifact_manifest_sha256,
        },
        "output_cutoffs": list(OUTPUT_CUTOFFS),
        "program_cutoffs": list(PROGRAM_CUTOFFS),
        "started_at": started_at,
        "completed_at": completed_at,
        "source_loader": dict(source_loader),
        "command": command,
        **_runtime_record(
            source_capture=source_capture, test_source_sha256=test_source_sha256
        ),
    }
    summary = {
        "schema_version": 6,
        "evidence_status": "verified_grammar_aligned_symbolic_control",
        "dataset_kind": "synthetic_control",
        "eval_spec_id": eval_spec_id,
        "case_set_id": blind_manifest["case_set_id"],
        "pool_content_id": pool_manifest["pool_content_id"],
        "metrics": metrics,
        "claim_boundary": (
            f"{SYNTHETIC_SUITE_VERSION} is a grammar-aligned execution/search "
            "control. It is not "
            "real ARC coverage, a public-evaluation result, or evidence of heterogeneous "
            "diffusion/code-source complementarity."
        ),
    }
    if runtime_source_fingerprint() != source_capture.fingerprint_sha256:
        raise RuntimeError("executing source changed during post-hoc evaluation")
    publish_evidence_bundle(
        target,
        run_id=eval_spec_id[:20],
        artifacts={
            "eval_manifest.json": serialize_json(eval_manifest),
            "oracle_labels.jsonl": serialize_jsonl(
                [oracle_by_task[task_id] for task_id in sorted(oracle_by_task)]
            ),
            "per_pair_metrics.jsonl": serialize_jsonl(pair_rows),
            "per_task_metrics.jsonl": serialize_jsonl(task_metric_rows),
            "per_slice_metrics.json": serialize_json(
                {"by_family": family_metrics, "by_stratum": stratum_metrics}
            ),
            "funnel.json": serialize_json(funnel),
            "integrity_report.json": serialize_json(integrity),
            "summary.json": serialize_json(summary),
            "source_snapshot.zip": source_capture.snapshot_zip,
        },
    )
    return summary


def _aggregate_public_task_metrics(rows: list[dict[str, Any]]) -> dict[str, object]:
    if not rows:
        raise ValueError("cannot aggregate an empty public task metric set")
    task_count = len(rows)
    pair_count = sum(int(row["query_pair_count"]) for row in rows)
    return {
        "task_count": task_count,
        "query_pair_count": pair_count,
        "task_first_pair_coverage_by_k": {
            str(k): sum(row["pair_fraction_by_k"][str(k)] for row in rows) / task_count
            for k in OUTPUT_CUTOFFS
        },
        "micro_pair_coverage_by_k": {
            str(k): sum(
                row["pair_fraction_by_k"][str(k)] * row["query_pair_count"]
                for row in rows
            )
            / pair_count
            for k in OUTPUT_CUTOFFS
        },
        "strict_task_coverage_by_k": {
            str(k): sum(row["strict_covered_by_k"][str(k)] for row in rows) / task_count
            for k in OUTPUT_CUTOFFS
        },
        "semantic_program_coverage_by_k": {
            str(k): sum(row["semantic_program_coverage_by_k"][str(k)] for row in rows)
            / task_count
            for k in PROGRAM_CUTOFFS
        },
    }


def evaluate_public_training_pool_bundle(
    *,
    blind_dataset_dir: str | Path,
    oracle_dataset_dir: str | Path,
    pool_dir: str | Path,
    output_dir: str | Path,
    source_capture: RuntimeSourceCapture,
    test_source_sha256: str | None,
    source_loader: Mapping[str, object],
    command: list[str],
) -> dict[str, object]:
    """Post-hoc output coverage for a frozen public-training development smoke."""

    target = Path(output_dir).expanduser().resolve()
    _require_nonoverlapping_paths(
        blind=blind_dataset_dir,
        oracle=oracle_dataset_dir,
        pool=pool_dir,
        output=target,
    )
    if target.exists():
        raise FileExistsError(f"refusing to overwrite evidence directory: {target}")
    context = _validated_blind_pool(
        blind_dataset_dir=blind_dataset_dir,
        pool_dir=pool_dir,
        source_capture=source_capture,
    )
    oracle_snapshot = snapshot_bundle(oracle_dataset_dir)
    oracle_manifest, oracle_by_task = _load_public_oracle_dataset(oracle_snapshot)
    if oracle_manifest["runtime_source_fingerprint_sha256"] != source_capture.fingerprint_sha256:
        raise ValueError("public oracle and evaluator source fingerprints differ")
    if context.blind_manifest["dataset_kind"] != "arc_public_training_development_smoke":
        raise ValueError("blind bundle is not a public-training development smoke")
    if oracle_manifest["case_set_id"] != context.blind_manifest["case_set_id"]:
        raise ValueError("public oracle and blind case_set_id differ")
    if oracle_manifest["dataset_descriptor"] != context.blind_manifest[
        "dataset_descriptor"
    ]:
        raise ValueError("public oracle and blind dataset descriptors differ")
    if set(oracle_by_task) != set(context.tasks):
        raise ValueError("public oracle and blind task sets differ")
    if any(
        len(oracle_by_task[task_id]["test"])
        != len(context.tasks[task_id].test_inputs)
        for task_id in context.tasks
    ):
        raise ValueError("public oracle query indices do not match blind query inputs")

    pair_rows: list[dict[str, object]] = []
    task_rows: list[dict[str, Any]] = []
    for task_id in sorted(context.tasks):
        task = context.tasks[task_id]
        oracle = oracle_by_task[task_id]
        oracle_outputs = tuple(as_grid(row["output"]) for row in oracle["test"])
        pair_vectors: list[dict[str, bool]] = []
        total_unique = 0
        for test_index, oracle_output in enumerate(oracle_outputs):
            ordered = context.candidate_groups.get((task_id, test_index), [])
            unique: list[CandidateRecord] = []
            seen: set[str] = set()
            for _, candidate in ordered:
                if candidate.output_key not in seen:
                    seen.add(candidate.output_key)
                    unique.append(candidate)
            total_unique += len(unique)
            hits = {
                str(k): any(candidate.output == oracle_output for candidate in unique[:k])
                for k in OUTPUT_CUTOFFS
            }
            pair_vectors.append(hits)
            pair_rows.append(
                {
                    "task_id": task_id,
                    "source_task_id": oracle["source_task_id"],
                    "test_index": test_index,
                    "oracle_output_key": grid_key(oracle_output),
                    "raw_candidate_count": len(ordered),
                    "unique_output_count": len(unique),
                    "first_hit_unique_rank": next(
                        (
                            index + 1
                            for index, candidate in enumerate(unique)
                            if candidate.output == oracle_output
                        ),
                        None,
                    ),
                    "covered_by_k": hits,
                }
            )

        semantic_first: int | None = None
        for row in context.program_groups.get(task_id, []):
            outputs = _program_query_outputs(row, len(oracle_outputs))
            if semantic_first is None and outputs == oracle_outputs:
                semantic_first = int(row["retained_rank"]) + 1
            for test_index, output in enumerate(outputs):
                if output is None:
                    continue
                matching = [
                    candidate
                    for _, candidate in context.candidate_groups.get(
                        (task_id, test_index), []
                    )
                    if candidate.program_hash == row["program"]["program_id"]
                ]
                if not any(candidate.output == output for candidate in matching):
                    raise ValueError("public candidate sidecar cannot replay query output")
        task_rows.append(
            {
                "task_id": task_id,
                "source_task_id": oracle["source_task_id"],
                "query_pair_count": len(oracle_outputs),
                "raw_candidate_count": sum(
                    len(context.candidate_groups.get((task_id, index), []))
                    for index in range(len(oracle_outputs))
                ),
                "unique_output_count": total_unique,
                "pair_fraction_by_k": {
                    str(k): sum(vector[str(k)] for vector in pair_vectors)
                    / len(pair_vectors)
                    for k in OUTPUT_CUTOFFS
                },
                "strict_covered_by_k": {
                    str(k): all(vector[str(k)] for vector in pair_vectors)
                    for k in OUTPUT_CUTOFFS
                },
                "semantic_solving_program_first_rank": semantic_first,
                "semantic_program_coverage_by_k": {
                    str(k): semantic_first is not None and semantic_first <= k
                    for k in PROGRAM_CUTOFFS
                },
            }
        )

    metrics = _aggregate_public_task_metrics(task_rows)
    funnel = {
        "tasks": len(task_rows),
        **_representation_funnel_counts(context.pool_rows),
        "shape_proposals": sum(
            len(row["shape_proposals"])
            for row in context.search_by_task.values()
        ),
        "instruction_options": sum(
            len(row["instruction_options"])
            for row in context.search_by_task.values()
        ),
        "evaluated_programs": sum(
            len(row["evaluated_program_ids"])
            for row in context.search_by_task.values()
        ),
        "semantic_duplicates": sum(
            row["semantic_duplicates"] for row in context.search_by_task.values()
        ),
        "retained_exact_programs": sum(
            len(rows) for rows in context.program_groups.values()
        ),
        "candidate_records": len(context.all_candidates),
        "unique_candidate_outputs": len(
            {
                (candidate.task_id, candidate.test_index, candidate.output_key)
                for candidate in context.all_candidates
            }
        ),
        "program_execution_count": sum(
            row["program_execution_count"]
            for row in context.search_by_task.values()
        ),
    }
    eval_spec_id = _content_id(
        "afts-e01a-public-eval-spec/v6",
        {
            "case_set_id": context.blind_manifest["case_set_id"],
            "oracle_set_id": oracle_manifest["oracle_set_id"],
            "pool_content_id": context.pool_manifest["pool_content_id"],
            "runtime_source_fingerprint_sha256": source_capture.fingerprint_sha256,
            "output_cutoffs": OUTPUT_CUTOFFS,
            "program_cutoffs": PROGRAM_CUTOFFS,
        },
    )
    started_at = utc_now()
    completed_at = utc_now()
    eval_manifest = {
        "schema": EVAL_MANIFEST_SCHEMA,
        "evidence_status": "post_hoc_public_training_development_smoke_evaluation",
        "dataset_kind": "arc_public_training_development_smoke",
        "eval_spec_id": eval_spec_id,
        "case_set_id": context.blind_manifest["case_set_id"],
        "oracle_set_id": oracle_manifest["oracle_set_id"],
        "pool_spec_id": context.pool_manifest["pool_spec_id"],
        "pool_content_id": context.pool_manifest["pool_content_id"],
        "parent_artifact_manifests": {
            "blind_sha256": context.blind_snapshot.artifact_manifest_sha256,
            "oracle_sha256": oracle_snapshot.artifact_manifest_sha256,
            "pool_sha256": context.pool_snapshot.artifact_manifest_sha256,
        },
        "output_cutoffs": list(OUTPUT_CUTOFFS),
        "program_cutoffs": list(PROGRAM_CUTOFFS),
        "started_at": started_at,
        "completed_at": completed_at,
        "source_loader": dict(source_loader),
        "command": command,
        **_runtime_record(
            source_capture=source_capture, test_source_sha256=test_source_sha256
        ),
    }
    integrity = {
        "all_checks_pass": True,
        "closed_world_bundles_verified": True,
        "case_set_ids_match": True,
        "pool_binds_exact_blind_artifact_manifest": True,
        "pool_content_id_recomputed": True,
        "parse_program_candidate_closure_replayed": True,
        "pool_oracle_keys_absent": True,
        "panel_parse_sidecar_replayed": True,
        "panel_sequence_d4_proposals_replayed": True,
        "panel_sequence_d4_cost_ledger_replayed": True,
        "panel_lattice_periodic_bounds_replayed": True,
        "panel_lattice_periodic_proposals_replayed": True,
        "panel_lattice_periodic_cost_ledger_replayed": True,
        "relation_parse_sidecar_replayed": True,
        "bbox_contact_bounds_replayed": True,
        "bbox_contact_proposals_replayed": True,
        "bbox_contact_cost_ledger_replayed": True,
        "bbox_contact_cap_survival_replayed": True,
        "instruction_cap_ledger_replayed": True,
    }
    summary = {
        "schema_version": 6,
        "evidence_status": "verified_public_training_development_smoke",
        "dataset_kind": "arc_public_training_development_smoke",
        "eval_spec_id": eval_spec_id,
        "case_set_id": context.blind_manifest["case_set_id"],
        "pool_content_id": context.pool_manifest["pool_content_id"],
        "metrics": metrics,
        "claim_boundary": (
            "Pinned ARC-AGI-2 public-training development smoke only. Outputs were "
            "loaded after pool freeze; this is not a holdout, public-evaluation, "
            "pass@2, release, or heterogeneous-source result."
        ),
    }
    if runtime_source_fingerprint() != source_capture.fingerprint_sha256:
        raise RuntimeError("executing source changed during public smoke evaluation")
    publish_evidence_bundle(
        target,
        run_id=eval_spec_id[:20],
        artifacts={
            "eval_manifest.json": serialize_json(eval_manifest),
            "oracle_labels.jsonl": serialize_jsonl(
                [oracle_by_task[task_id] for task_id in sorted(oracle_by_task)]
            ),
            "per_pair_metrics.jsonl": serialize_jsonl(pair_rows),
            "per_task_metrics.jsonl": serialize_jsonl(task_rows),
            "per_slice_metrics.json": serialize_json({"development_smoke": metrics}),
            "funnel.json": serialize_json(funnel),
            "integrity_report.json": serialize_json(integrity),
            "summary.json": serialize_json(summary),
            "source_snapshot.zip": source_capture.snapshot_zip,
        },
    )
    return summary


def verify_eval_bundle(
    path: str | Path,
    *,
    blind_dataset_dir: str | Path,
    oracle_dataset_dir: str | Path,
    pool_dir: str | Path,
    source_capture: RuntimeSourceCapture,
    test_source_sha256: str | None,
) -> dict[str, object]:
    """Replay parent bundles and independently recompute every result artifact."""

    _require_nonoverlapping_paths(
        evaluation=path,
        blind=blind_dataset_dir,
        oracle=oracle_dataset_dir,
        pool=pool_dir,
    )
    snapshot = snapshot_bundle(path)
    _require_artifacts(
        snapshot,
        {
            "eval_manifest.json",
            "oracle_labels.jsonl",
            "per_pair_metrics.jsonl",
            "per_task_metrics.jsonl",
            "per_slice_metrics.json",
            "funnel.json",
            "integrity_report.json",
            "summary.json",
            "source_snapshot.zip",
        },
    )
    manifest = _require_fields(
        _strict_json(snapshot.artifacts["eval_manifest.json"], label="eval_manifest.json"),
        {
            "schema",
            "evidence_status",
            "dataset_kind",
            "eval_spec_id",
            "case_set_id",
            "oracle_set_id",
            "pool_spec_id",
            "pool_content_id",
            "parent_artifact_manifests",
            "output_cutoffs",
            "program_cutoffs",
            "started_at",
            "completed_at",
            "source_loader",
            "command",
            "runtime_source_fingerprint_sha256",
            "test_source_fingerprint_sha256",
        },
        label="evaluation manifest",
    )
    if manifest["schema"] != EVAL_MANIFEST_SCHEMA:
        raise ValueError("unsupported evaluation manifest")
    verify_source_snapshot_zip(
        snapshot.artifacts["source_snapshot.zip"],
        expected_fingerprint_sha256=manifest["runtime_source_fingerprint_sha256"],
    )
    if manifest["output_cutoffs"] != list(OUTPUT_CUTOFFS) or manifest[
        "program_cutoffs"
    ] != list(PROGRAM_CUTOFFS):
        raise ValueError("evaluation cutoffs differ from the registered protocol")
    if (
        manifest["runtime_source_fingerprint_sha256"]
        != source_capture.fingerprint_sha256
        or manifest["test_source_fingerprint_sha256"] != test_source_sha256
        or snapshot.artifacts["source_snapshot.zip"] != source_capture.snapshot_zip
    ):
        raise ValueError("evaluation source or test fingerprint does not match verifier")
    if (
        not isinstance(manifest["started_at"], str)
        or not isinstance(manifest["completed_at"], str)
        or not isinstance(manifest["command"], list)
        or any(not isinstance(item, str) for item in manifest["command"])
    ):
        raise ValueError("evaluation execution metadata is malformed")

    context = _validated_blind_pool(
        blind_dataset_dir=blind_dataset_dir,
        pool_dir=pool_dir,
        source_capture=source_capture,
    )
    oracle_snapshot = snapshot_bundle(oracle_dataset_dir)
    dataset_kind = manifest["dataset_kind"]
    if dataset_kind == "synthetic_control":
        oracle_manifest, oracle_by_task = _load_oracle_dataset(oracle_snapshot)
        eval_label = "afts-e01a-eval-spec/v6"
        expected_evidence_status = "post_hoc_oracle_evaluation_of_frozen_symbolic_pool"
    elif dataset_kind == "arc_public_training_development_smoke":
        oracle_manifest, oracle_by_task = _load_public_oracle_dataset(oracle_snapshot)
        eval_label = "afts-e01a-public-eval-spec/v6"
        expected_evidence_status = "post_hoc_public_training_development_smoke_evaluation"
    else:
        raise ValueError("evaluation manifest has an unsupported dataset_kind")
    if (
        context.blind_manifest["dataset_kind"] != dataset_kind
        or oracle_manifest["runtime_source_fingerprint_sha256"]
        != source_capture.fingerprint_sha256
        or oracle_manifest["case_set_id"] != context.blind_manifest["case_set_id"]
    ):
        raise ValueError("evaluation parents disagree on dataset kind, source, or case set")
    if dataset_kind == "arc_public_training_development_smoke" and oracle_manifest[
        "dataset_descriptor"
    ] != context.blind_manifest["dataset_descriptor"]:
        raise ValueError("public evaluation parents disagree on dataset descriptor")
    parent_hashes = {
        "blind_sha256": context.blind_snapshot.artifact_manifest_sha256,
        "oracle_sha256": oracle_snapshot.artifact_manifest_sha256,
        "pool_sha256": context.pool_snapshot.artifact_manifest_sha256,
    }
    if manifest["parent_artifact_manifests"] != parent_hashes:
        raise ValueError("evaluation manifest does not bind the supplied parent bundles")
    if (
        manifest["evidence_status"] != expected_evidence_status
        or manifest["case_set_id"] != context.blind_manifest["case_set_id"]
        or manifest["oracle_set_id"] != oracle_manifest["oracle_set_id"]
        or manifest["pool_spec_id"] != context.pool_manifest["pool_spec_id"]
        or manifest["pool_content_id"] != context.pool_manifest["pool_content_id"]
    ):
        raise ValueError("evaluation manifest scientific identities do not match parents")
    expected_eval_spec = _content_id(
        eval_label,
        {
            "case_set_id": context.blind_manifest["case_set_id"],
            "oracle_set_id": oracle_manifest["oracle_set_id"],
            "pool_content_id": context.pool_manifest["pool_content_id"],
            "runtime_source_fingerprint_sha256": source_capture.fingerprint_sha256,
            "output_cutoffs": OUTPUT_CUTOFFS,
            "program_cutoffs": PROGRAM_CUTOFFS,
        },
    )
    if (
        manifest["eval_spec_id"] != expected_eval_spec
        or snapshot.artifact_manifest["run_id"] != expected_eval_spec[:20]
    ):
        raise ValueError("eval_spec_id or artifact run_id does not match bound inputs")

    expected_oracle_rows = [oracle_by_task[key] for key in sorted(oracle_by_task)]
    actual_oracle_rows = _strict_jsonl(
        snapshot.artifacts["oracle_labels.jsonl"], label="oracle_labels.jsonl"
    )
    if actual_oracle_rows != expected_oracle_rows:
        raise ValueError("evaluation oracle labels differ from the bound oracle bundle")
    expected_pair_rows, expected_task_rows = _recompute_evaluation_rows(
        context=context,
        oracle_by_task=oracle_by_task,
        dataset_kind=dataset_kind,
    )
    actual_pair_rows = _strict_jsonl(
        snapshot.artifacts["per_pair_metrics.jsonl"], label="per_pair_metrics.jsonl"
    )
    actual_task_rows = _strict_jsonl(
        snapshot.artifacts["per_task_metrics.jsonl"], label="per_task_metrics.jsonl"
    )
    if actual_pair_rows != expected_pair_rows or actual_task_rows != expected_task_rows:
        raise ValueError("pair or task metrics do not recompute from pool and oracle")

    if dataset_kind == "synthetic_control":
        metrics = _aggregate_task_metrics(expected_task_rows)
        slice_metrics = {
            "by_family": {
                family: _aggregate_task_metrics(
                    [row for row in expected_task_rows if row["family"] == family]
                )
                for family in sorted({row["family"] for row in expected_task_rows})
            },
            "by_stratum": {
                stratum: _aggregate_task_metrics(
                    [row for row in expected_task_rows if row["stratum"] == stratum]
                )
                for stratum in sorted({row["stratum"] for row in expected_task_rows})
            },
        }
        funnel = {
            "tasks": len(expected_task_rows),
            **_representation_funnel_counts(context.pool_rows),
            "shape_proposals": sum(
                len(row["shape_proposals"])
                for row in context.search_by_task.values()
            ),
            "instruction_options": sum(
                len(row["instruction_options"])
                for row in context.search_by_task.values()
            ),
            "evaluated_programs": sum(
                len(row["evaluated_program_ids"])
                for row in context.search_by_task.values()
            ),
            "semantic_duplicates": sum(
                row["semantic_duplicates"] for row in context.search_by_task.values()
            ),
            "retained_exact_programs": sum(
                len(rows) for rows in context.program_groups.values()
            ),
            "candidate_records": len(context.all_candidates),
            "unique_candidate_outputs": len(
                {
                    (item.task_id, item.test_index, item.output_key)
                    for item in context.all_candidates
                }
            ),
            "program_execution_count": sum(
                row["program_execution_count"]
                for row in context.search_by_task.values()
            ),
        }
        integrity = {
            "all_checks_pass": True,
            "closed_world_bundles_verified": True,
            "artifact_hashes_verified_from_captured_bytes": True,
            "case_set_ids_match": True,
            "pool_binds_exact_blind_artifact_manifest": True,
            "pool_tasks_equal_blind_tasks": True,
            "pool_oracle_keys_absent": True,
            "candidate_ranks_contiguous": True,
            "program_and_candidate_parent_closure": True,
            "expansion_ledgers_match_program_ids": True,
            "generator_replay_control_100_percent": True,
            "panel_parse_sidecar_replayed": True,
            "panel_sequence_d4_proposals_replayed": True,
            "panel_sequence_d4_cost_ledger_replayed": True,
            "panel_lattice_periodic_bounds_replayed": True,
            "panel_lattice_periodic_proposals_replayed": True,
            "panel_lattice_periodic_cost_ledger_replayed": True,
            "relation_parse_sidecar_replayed": True,
            "bbox_contact_bounds_replayed": True,
            "bbox_contact_proposals_replayed": True,
            "bbox_contact_cost_ledger_replayed": True,
            "bbox_contact_cap_survival_replayed": True,
            "instruction_cap_ledger_replayed": True,
        }
        summary = {
            "schema_version": 6,
            "evidence_status": "verified_grammar_aligned_symbolic_control",
            "dataset_kind": dataset_kind,
            "eval_spec_id": expected_eval_spec,
            "case_set_id": context.blind_manifest["case_set_id"],
            "pool_content_id": context.pool_manifest["pool_content_id"],
            "metrics": metrics,
            "claim_boundary": (
                f"{SYNTHETIC_SUITE_VERSION} is a grammar-aligned execution/search "
                "control. It is not "
                "real ARC coverage, a public-evaluation result, or evidence of heterogeneous "
                "diffusion/code-source complementarity."
            ),
        }
    else:
        metrics = _aggregate_public_task_metrics(expected_task_rows)
        slice_metrics = {"development_smoke": metrics}
        funnel = {
            "tasks": len(expected_task_rows),
            **_representation_funnel_counts(context.pool_rows),
            "shape_proposals": sum(
                len(row["shape_proposals"])
                for row in context.search_by_task.values()
            ),
            "instruction_options": sum(
                len(row["instruction_options"])
                for row in context.search_by_task.values()
            ),
            "evaluated_programs": sum(
                len(row["evaluated_program_ids"])
                for row in context.search_by_task.values()
            ),
            "semantic_duplicates": sum(
                row["semantic_duplicates"] for row in context.search_by_task.values()
            ),
            "retained_exact_programs": sum(
                len(rows) for rows in context.program_groups.values()
            ),
            "candidate_records": len(context.all_candidates),
            "unique_candidate_outputs": len(
                {
                    (item.task_id, item.test_index, item.output_key)
                    for item in context.all_candidates
                }
            ),
            "program_execution_count": sum(
                row["program_execution_count"]
                for row in context.search_by_task.values()
            ),
        }
        integrity = {
            "all_checks_pass": True,
            "closed_world_bundles_verified": True,
            "case_set_ids_match": True,
            "pool_binds_exact_blind_artifact_manifest": True,
            "pool_content_id_recomputed": True,
            "parse_program_candidate_closure_replayed": True,
            "pool_oracle_keys_absent": True,
            "panel_parse_sidecar_replayed": True,
            "panel_sequence_d4_proposals_replayed": True,
            "panel_sequence_d4_cost_ledger_replayed": True,
            "panel_lattice_periodic_bounds_replayed": True,
            "panel_lattice_periodic_proposals_replayed": True,
            "panel_lattice_periodic_cost_ledger_replayed": True,
            "relation_parse_sidecar_replayed": True,
            "bbox_contact_bounds_replayed": True,
            "bbox_contact_proposals_replayed": True,
            "bbox_contact_cost_ledger_replayed": True,
            "bbox_contact_cap_survival_replayed": True,
            "instruction_cap_ledger_replayed": True,
        }
        summary = {
            "schema_version": 6,
            "evidence_status": "verified_public_training_development_smoke",
            "dataset_kind": dataset_kind,
            "eval_spec_id": expected_eval_spec,
            "case_set_id": context.blind_manifest["case_set_id"],
            "pool_content_id": context.pool_manifest["pool_content_id"],
            "metrics": metrics,
            "claim_boundary": (
                "Pinned ARC-AGI-2 public-training development smoke only. Outputs were "
                "loaded after pool freeze; this is not a holdout, public-evaluation, "
                "pass@2, release, or heterogeneous-source result."
            ),
        }
    if _strict_json(
        snapshot.artifacts["per_slice_metrics.json"], label="per_slice_metrics.json"
    ) != slice_metrics:
        raise ValueError("slice metrics do not recompute from task rows")
    if _strict_json(snapshot.artifacts["funnel.json"], label="funnel.json") != funnel:
        raise ValueError("funnel does not recompute from the bound pool")
    if _strict_json(
        snapshot.artifacts["integrity_report.json"], label="integrity_report.json"
    ) != integrity:
        raise ValueError("integrity report does not match completed checks")
    if _strict_json(snapshot.artifacts["summary.json"], label="summary.json") != summary:
        raise ValueError("evaluation summary does not recompute from parent bundles")
    return {
        "verification_status": "full_parent_replay_pass",
        "eval_spec_id": expected_eval_spec,
        "task_count": metrics["task_count"],
        "artifact_manifest_sha256": snapshot.artifact_manifest_sha256,
    }
