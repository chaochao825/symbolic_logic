"""Command-line entry points for Phase-0 data and scoring audits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ._source_bootstrap import BOOTSTRAP_RUNTIME_SOURCE_FINGERPRINT
from .authority import DATASET_COMMITS, DATASET_ORIGINS, SPLIT_DIRECTORIES
from .baselines import dihedral_candidates
from .candidate import CandidateRecord, CandidateStore
from .e01 import (
    build_public_training_smoke_bundles,
    build_symbolic_pool_bundle,
    build_synthetic_dataset_bundles,
    evaluate_public_training_pool_bundle,
    evaluate_symbolic_pool_bundle,
    verify_eval_bundle,
)
from .manifest import (
    DatasetSnapshot,
    RuntimeSourceCapture,
    build_run_manifest,
    capture_runtime_source,
    file_sha256,
    git_blob_bytes,
    git_repo_state,
    publish_evidence_bundle,
    runtime_source_fingerprint,
    serialize_json,
    serialize_jsonl,
    snapshot_task_directory,
    test_source_fingerprint,
    utc_now,
    verify_dataset_snapshot_unchanged,
)
from .scoring import score_dataset
from .search import SearchConfig
from .task import ARCTask

SCORER_ORIGIN = "https://github.com/arcprize/arc-agi-benchmarking.git"
SCORER_COMMIT = "688c9be4fb270ad7eede09fe8bba6f8187be3bed"
SCORER_RELATIVE_PATH = Path("src/arc_agi_benchmarking/scoring/scoring.py")
RELEASE_GATE_IDS: frozenset[str] = frozenset()
_IMPORTED_RUNTIME_SOURCE_FINGERPRINT = runtime_source_fingerprint()


def _local_project_dir() -> Path | None:
    candidate = Path(__file__).resolve().parents[2]
    return candidate if (candidate / "pyproject.toml").exists() else None


def _validate_split_path(args: argparse.Namespace) -> None:
    task_dir = Path(args.task_dir).expanduser().resolve()
    repo_dir = Path(args.repo_dir).expanduser().resolve()
    expected_name = SPLIT_DIRECTORIES[args.split_name]
    expected_path = (repo_dir / "data" / expected_name).resolve()
    if task_dir != expected_path:
        raise ValueError(
            f"split {args.split_name} must use canonical path {expected_path}, found {task_dir}"
        )


def _snapshot_from_args(
    args: argparse.Namespace, *, validate: bool = True, max_tasks: int | None = None
) -> DatasetSnapshot:
    _validate_split_path(args)
    return snapshot_task_directory(
        args.task_dir,
        dataset_name=args.dataset_name,
        split_name=args.split_name,
        repo_dir=args.repo_dir,
        expected_origin=DATASET_ORIGINS[args.dataset_name],
        expected_commit=DATASET_COMMITS[args.dataset_name],
        validate=validate,
        max_tasks=max_tasks,
    )


def _require_release_gate(args: argparse.Namespace) -> None:
    if args.split_name != "public_evaluation":
        return
    if not args.release_gate_id:
        raise PermissionError(
            "public evaluation access is sealed; provide a pre-registered --release-gate-id"
        )
    if args.release_gate_id not in RELEASE_GATE_IDS:
        raise PermissionError(
            f"release gate {args.release_gate_id!r} is not pre-registered in this source version"
        )


def _require_imported_source_identity(runtime_fingerprint: str) -> None:
    if (
        runtime_fingerprint
        not in {
            BOOTSTRAP_RUNTIME_SOURCE_FINGERPRINT,
            _IMPORTED_RUNTIME_SOURCE_FINGERPRINT,
        }
        or BOOTSTRAP_RUNTIME_SOURCE_FINGERPRINT != _IMPORTED_RUNTIME_SOURCE_FINGERPRINT
    ):
        raise RuntimeError(
            "package source differs from the package bootstrap or CLI import snapshot; "
            "start a fresh process before producing evidence"
        )


def _require_fresh_source_loader() -> None:
    prefix = sys.pycache_prefix
    launcher_value = os.environ.get("AFTS_EVIDENCE_LAUNCHER")
    neural_mode = os.environ.get("AFTS_M04A_TORCH_MODE") == "1"
    launcher_name = "afts_arc_m04a.py" if neural_mode else "afts_arc_evidence.py"
    prefix_name = "afts-m04a-pycache-" if neural_mode else "afts-arc-pycache-"
    input_root_value = os.environ.get("AFTS_M04A_INPUT_ROOT")
    if neural_mode and input_root_value:
        expected_launcher = Path(input_root_value).resolve() / "remote_launcher.py"
    else:
        expected_launcher = (
            Path(__file__).resolve().parents[2] / "scripts" / launcher_name
        )
    launcher = Path(launcher_value).resolve() if launcher_value else None
    source_root = Path(__file__).resolve().parents[1]
    cwd = Path.cwd().resolve()
    resolved_sys_path = tuple(
        Path(entry).resolve() for entry in sys.path if isinstance(entry, str) and entry
    )
    raw_import_roots = os.environ.get("AFTS_M04A_IMPORT_ROOTS_JSON")
    raw_bound_import_roots = os.environ.get("AFTS_M04A_BOUND_IMPORT_ROOTS_JSON")
    try:
        import_roots = json.loads(raw_import_roots or "") if neural_mode else []
        bound_import_roots = (
            json.loads(raw_bound_import_roots or "") if neural_mode else []
        )
    except json.JSONDecodeError:
        import_roots = []
        bound_import_roots = []
    valid_import_records = (
        isinstance(import_roots, list)
        and len(import_roots) in {2, 3}
        and all(isinstance(entry, str) and entry for entry in import_roots)
        and isinstance(bound_import_roots, list)
        and len(bound_import_roots) == len(import_roots)
        and bound_import_roots
        == [f"/proc/self/fd/{200 + index}" for index in range(len(import_roots))]
    )
    resolved_import_roots = (
        tuple(Path(entry).resolve() for entry in import_roots)
        if valid_import_records
        else ()
    )
    resolved_bound_import_roots = (
        tuple(Path(entry).resolve() for entry in bound_import_roots)
        if valid_import_records
        else ()
    )
    custom_root_count = len(resolved_bound_import_roots) if neural_mode else 1
    if (
        os.environ.get("AFTS_EVIDENCE_FRESH_SOURCE_LOADER") != "1"
        or not sys.flags.isolated
        or not sys.dont_write_bytecode
        or not prefix
        or Path(prefix).exists()
        or not Path(prefix).name.startswith(prefix_name)
        or launcher is None
        or not launcher.is_file()
        or (expected_launcher.is_file() and launcher != expected_launcher.resolve())
        or not resolved_sys_path
        or (
            neural_mode
            and (
                not resolved_import_roots
                or resolved_import_roots[0] != source_root
                or resolved_bound_import_roots != resolved_import_roots
                or resolved_sys_path[-len(resolved_import_roots) :]
                != resolved_bound_import_roots
            )
        )
        or (not neural_mode and resolved_sys_path[-1] != source_root)
        or resolved_sys_path.count(source_root) != 1
        or cwd in resolved_sys_path[:-custom_root_count]
    ):
        raise RuntimeError(
            "formal evidence must be launched with: "
            f"python scripts/{launcher_name} <command> ..."
        )


def _source_loader_record() -> dict[str, object]:
    neural_mode = os.environ.get("AFTS_M04A_TORCH_MODE") == "1"
    return {
        "policy": "fresh_nonexistent_pycache_prefix_with_bytecode_writes_disabled",
        "launcher_material": (
            "scripts/afts_arc_m04a.py"
            if neural_mode
            else "scripts/afts_arc_evidence.py"
        ),
        "bootstrap_import_run_source_match": True,
        "python_isolated_mode": True,
        "cwd_removed_from_import_path": True,
        "project_source_after_standard_library": True,
        "explicit_site_packages_without_site_initialization": neural_mode,
    }


def _cmd_audit_data(args: argparse.Namespace) -> int:
    _require_release_gate(args)
    started_at = utc_now()
    source_capture = capture_runtime_source()
    _require_imported_source_identity(source_capture.fingerprint_sha256)
    snapshot = _snapshot_from_args(args, validate=not args.hash_only)
    verify_dataset_snapshot_unchanged(snapshot)
    if runtime_source_fingerprint() != source_capture.fingerprint_sha256:
        raise RuntimeError("executing package source changed during the audit")
    payload = {
        "schema_version": 2,
        "evidence_status": "verified_pinned_dataset_audit",
        "started_at": started_at,
        "completed_at": utc_now(),
        "command": args.effective_command,
        "runtime_source_fingerprint_sha256": source_capture.fingerprint_sha256,
        "source_loader": _source_loader_record(),
        "release_gate_id": args.release_gate_id,
        "dataset": asdict(snapshot.audit),
    }
    canonical_without_id = serialize_json(payload)
    payload["audit_id"] = hashlib.sha256(canonical_without_id).hexdigest()[:20]
    if args.output_dir:
        publish_evidence_bundle(
            args.output_dir,
            run_id=payload["audit_id"],
            artifacts={
                "audit.json": serialize_json(payload),
                "source_snapshot.zip": source_capture.snapshot_zip,
            },
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _candidate_groups(
    candidates: tuple[CandidateRecord, ...],
) -> dict[int, list[CandidateRecord]]:
    groups: dict[int, list[CandidateRecord]] = defaultdict(list)
    for candidate in candidates:
        groups[candidate.test_index].append(candidate)
    return groups


def _oracle_pair_coverage(
    task: ARCTask, groups: dict[int, list[CandidateRecord]]
) -> tuple[bool, ...]:
    coverage: list[bool] = []
    for test_index, pair in enumerate(task.test):
        if pair.output is None:
            raise ValueError(
                f"task {task.task_id} test[{test_index}] has no reference output"
            )
        coverage.append(
            any(
                candidate.output == pair.output
                for candidate in groups.get(test_index, [])
            )
        )
    return tuple(coverage)


def _scorer_reference(args: argparse.Namespace) -> dict[str, Any]:
    state = git_repo_state(
        args.scorer_repo_dir,
        expected_origin=SCORER_ORIGIN,
        expected_commit=SCORER_COMMIT,
        require_clean=True,
    )
    scorer_file = Path(state.path) / SCORER_RELATIVE_PATH
    if not scorer_file.is_file():
        raise ValueError(f"official scorer file not found: {scorer_file}")
    scorer_blob = git_blob_bytes(state, SCORER_RELATIVE_PATH.as_posix())
    return {
        **asdict(state),
        "scoring_file": SCORER_RELATIVE_PATH.as_posix(),
        "identity_source": "pinned_git_blob",
        "scoring_file_sha256": hashlib.sha256(scorer_blob).hexdigest(),
        "worktree_scoring_file_sha256_diagnostic": file_sha256(scorer_file),
        "attempt_limit_enforced_locally": 2,
        "scoring_semantics": "mean across tasks of within-task exact test-pair fractions",
    }


def _verify_scorer_reference(
    args: argparse.Namespace, expected: dict[str, Any]
) -> None:
    current = _scorer_reference(args)
    if current != expected:
        raise RuntimeError("official scorer repository changed during the run")


def _cmd_run_d4(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite evidence directory: {output_dir}")
    _require_release_gate(args)

    started_at = utc_now()
    source_capture = capture_runtime_source()
    _require_imported_source_identity(source_capture.fingerprint_sha256)
    runtime_fingerprint_before = source_capture.fingerprint_sha256
    source_snapshot = source_capture.snapshot_zip
    project_dir = _local_project_dir()
    tests_fingerprint_before = (
        test_source_fingerprint(project_dir) if project_dir is not None else None
    )
    scorer_reference = _scorer_reference(args)
    dataset_snapshot = _snapshot_from_args(
        args, validate=True, max_tasks=args.max_tasks
    )
    tasks = dataset_snapshot.tasks

    store = CandidateStore()
    submissions: dict[str, dict[int, list[object]]] = {}
    task_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    oracle_pair_hits = 0
    oracle_pair_total = 0
    strict_oracle_tasks = 0

    for task in tasks:
        candidates = dihedral_candidates(task)
        for candidate in candidates:
            store.add(candidate)
            candidate_rows.append(candidate.to_json_dict())
        groups = _candidate_groups(candidates)
        submissions[task.task_id] = {
            test_index: [candidate.output for candidate in group[:2]]
            for test_index, group in groups.items()
        }
        oracle = _oracle_pair_coverage(task, groups)
        oracle_pair_hits += sum(oracle)
        oracle_pair_total += len(oracle)
        strict_oracle_tasks += int(all(oracle))
        task_rows.append(
            {
                "task_id": task.task_id,
                "test_pair_count": len(task.test),
                "candidate_records": len(candidates),
                "unique_outputs": sum(
                    len({candidate.output_key for candidate in group})
                    for group in groups.values()
                ),
                "oracle_pair_coverage": list(oracle),
                "strict_oracle_task_covered": all(oracle),
            }
        )

    dataset_score = score_dataset(tasks, submissions)
    scores_by_id = {score.task_id: score for score in dataset_score.task_scores}
    for row in task_rows:
        score = scores_by_id[row["task_id"]]
        row.update(
            {
                "official_pass_at_1": score.official_pass_at_1,
                "official_pass_at_2": score.official_pass_at_2,
                "strict_task_solved_at_1": score.strict_task_solved_at_1,
                "strict_task_solved_at_2": score.strict_task_solved_at_2,
            }
        )

    verify_dataset_snapshot_unchanged(dataset_snapshot)
    _verify_scorer_reference(args, scorer_reference)
    runtime_fingerprint_after = runtime_source_fingerprint()
    if runtime_fingerprint_after != runtime_fingerprint_before:
        raise RuntimeError("executing package source changed during the run")
    tests_fingerprint_after = (
        test_source_fingerprint(project_dir) if project_dir is not None else None
    )
    if tests_fingerprint_after != tests_fingerprint_before:
        raise RuntimeError("local test source changed during the run")

    completed_at = utc_now()
    manifest = build_run_manifest(
        command=args.effective_command,
        cwd=Path.cwd(),
        dataset_audit=dataset_snapshot.audit,
        started_at=started_at,
        completed_at=completed_at,
        runtime_source_sha256=runtime_fingerprint_before,
        test_source_sha256=tests_fingerprint_before,
        scorer_reference=scorer_reference,
        extra={
            "max_tasks": args.max_tasks,
            "tasks_scored": len(tasks),
            "release_gate_id": args.release_gate_id,
            "orchestrator_id": args.orchestrator_id,
            "source_loader": _source_loader_record(),
        },
    )
    summary = {
        "schema_version": 2,
        "evidence_status": "verified_harness_smoke_not_research_result",
        "baseline": "deterministic_d4_phase0",
        "run_id": manifest["run_id"],
        "dataset": asdict(dataset_snapshot.audit),
        "scorer_reference_commit": scorer_reference["commit"],
        "attempt_limit": 2,
        "tasks_scored": len(tasks),
        "candidate_records": len(store),
        "unique_output_groups": store.unique_output_count,
        "official_pass_at_1": dataset_score.official_pass_at_1,
        "official_pass_at_2": dataset_score.official_pass_at_2,
        "strict_task_solved_rate_at_1": dataset_score.strict_task_solved_rate_at_1,
        "strict_task_solved_rate_at_2": dataset_score.strict_task_solved_rate_at_2,
        "oracle_pair_coverage": oracle_pair_hits / oracle_pair_total
        if oracle_pair_total
        else 0.0,
        "strict_oracle_task_coverage": strict_oracle_tasks / len(tasks)
        if tasks
        else 0.0,
        "note": (
            f"{args.split_name} D4 harness smoke, not a solver claim. Oracle coverage uses "
            "every unique D4 transform; pass@2 uses only the first two."
        ),
    }
    publish_evidence_bundle(
        output_dir,
        run_id=manifest["run_id"],
        artifacts={
            "summary.json": serialize_json(summary),
            "run_manifest.json": serialize_json(manifest),
            "tasks.jsonl": serialize_jsonl(task_rows),
            "candidates.jsonl": serialize_jsonl(candidate_rows),
            "source_snapshot.zip": source_snapshot,
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _phase1_source_context() -> tuple[RuntimeSourceCapture, str | None]:
    source_capture = capture_runtime_source()
    _require_imported_source_identity(source_capture.fingerprint_sha256)
    project_dir = _local_project_dir()
    tests_fingerprint = (
        test_source_fingerprint(project_dir) if project_dir is not None else None
    )
    return source_capture, tests_fingerprint


def _cmd_build_e01a_dataset(args: argparse.Namespace) -> int:
    source_capture, tests_fingerprint = _phase1_source_context()
    summary = build_synthetic_dataset_bundles(
        blind_output_dir=args.blind_output_dir,
        oracle_output_dir=args.oracle_output_dir,
        seed=args.seed,
        tasks_per_family=args.tasks_per_family,
        source_capture=source_capture,
        test_source_sha256=tests_fingerprint,
        source_loader=_source_loader_record(),
        command=args.effective_command,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _cmd_build_e01a_public_dataset(args: argparse.Namespace) -> int:
    if args.dataset_name != "ARC-AGI-2" or args.split_name != "public_training":
        raise ValueError(
            "E01a public development smoke requires ARC-AGI-2 public_training"
        )
    source_capture, tests_fingerprint = _phase1_source_context()
    dataset_snapshot = _snapshot_from_args(args, validate=True)
    summary = build_public_training_smoke_bundles(
        dataset_snapshot=dataset_snapshot,
        blind_output_dir=args.blind_output_dir,
        oracle_output_dir=args.oracle_output_dir,
        smoke_count=args.smoke_count,
        selection_salt=args.selection_salt,
        source_capture=source_capture,
        test_source_sha256=tests_fingerprint,
        source_loader=_source_loader_record(),
        command=args.effective_command,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _cmd_run_e01a_pool(args: argparse.Namespace) -> int:
    source_capture, tests_fingerprint = _phase1_source_context()
    config = SearchConfig(
        max_depth=args.max_depth,
        beam_width=args.beam_width,
        max_instruction_options=args.max_instruction_options,
        max_exact_programs=args.max_exact_programs,
    )
    summary = build_symbolic_pool_bundle(
        blind_dataset_dir=args.blind_dataset_dir,
        output_dir=args.output_dir,
        search_config=config,
        source_capture=source_capture,
        test_source_sha256=tests_fingerprint,
        source_loader=_source_loader_record(),
        command=args.effective_command,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _cmd_evaluate_e01a_pool(args: argparse.Namespace) -> int:
    source_capture, tests_fingerprint = _phase1_source_context()
    summary = evaluate_symbolic_pool_bundle(
        blind_dataset_dir=args.blind_dataset_dir,
        oracle_dataset_dir=args.oracle_dataset_dir,
        pool_dir=args.pool_dir,
        output_dir=args.output_dir,
        source_capture=source_capture,
        test_source_sha256=tests_fingerprint,
        source_loader=_source_loader_record(),
        command=args.effective_command,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _cmd_evaluate_e01a_public_pool(args: argparse.Namespace) -> int:
    source_capture, tests_fingerprint = _phase1_source_context()
    summary = evaluate_public_training_pool_bundle(
        blind_dataset_dir=args.blind_dataset_dir,
        oracle_dataset_dir=args.oracle_dataset_dir,
        pool_dir=args.pool_dir,
        output_dir=args.output_dir,
        source_capture=source_capture,
        test_source_sha256=tests_fingerprint,
        source_loader=_source_loader_record(),
        command=args.effective_command,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _cmd_verify_e01a_eval(args: argparse.Namespace) -> int:
    source_capture, tests_fingerprint = _phase1_source_context()
    report = verify_eval_bundle(
        args.eval_bundle,
        blind_dataset_dir=args.blind_dataset_dir,
        oracle_dataset_dir=args.oracle_dataset_dir,
        pool_dir=args.pool_dir,
        source_capture=source_capture,
        test_source_sha256=tests_fingerprint,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _cmd_build_m04a_rearc_cache(args: argparse.Namespace) -> int:
    source_capture = capture_runtime_source()
    _require_imported_source_identity(source_capture.fingerprint_sha256)
    from .m04a_rearc_index import build_rearc_mmap_cache

    result = build_rearc_mmap_cache(
        sanitized_split_dir=args.sanitized_bundle_dir,
        fold=args.fold,
        output_dir=args.output_dir,
        expected_sanitized_artifact_manifest_sha256=(
            args.sanitized_artifact_manifest_sha256
        ),
    )
    if runtime_source_fingerprint() != source_capture.fingerprint_sha256:
        raise RuntimeError("executing source changed while building the ReARC cache")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _indexed_m04a_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sanitized-bundle-dir", required=True)
    parser.add_argument("--sanitized-artifact-manifest-sha256", required=True)
    parser.add_argument("--training-rearc-cache-dir", required=True)
    parser.add_argument(
        "--training-rearc-cache-artifact-manifest-sha256", required=True
    )
    parser.add_argument("--validation-rearc-cache-dir", required=True)
    parser.add_argument(
        "--validation-rearc-cache-artifact-manifest-sha256", required=True
    )


def _cmd_build_m04a_validation_manifest(args: argparse.Namespace) -> int:
    source_capture = capture_runtime_source()
    _require_imported_source_identity(source_capture.fingerprint_sha256)
    from .m04a_data import build_validation_episodes
    from .m04a_production_data import load_indexed_m04a_data
    from .m04a_validation_manifest import publish_validation_episode_manifest

    with load_indexed_m04a_data(
        sanitized_bundle_dir=args.sanitized_bundle_dir,
        expected_sanitized_artifact_manifest_sha256=(
            args.sanitized_artifact_manifest_sha256
        ),
        training_rearc_cache_dir=args.training_rearc_cache_dir,
        expected_training_cache_artifact_manifest_sha256=(
            args.training_rearc_cache_artifact_manifest_sha256
        ),
        validation_rearc_cache_dir=args.validation_rearc_cache_dir,
        expected_validation_cache_artifact_manifest_sha256=(
            args.validation_rearc_cache_artifact_manifest_sha256
        ),
    ) as indexed:
        episodes = build_validation_episodes(indexed.validation)
        manifest = publish_validation_episode_manifest(args.output_dir, episodes)
    if runtime_source_fingerprint() != source_capture.fingerprint_sha256:
        raise RuntimeError(
            "executing source changed while building validation evidence"
        )
    report = {
        "status": "published_m04a_validation_episode_manifest",
        "output_dir": str(Path(args.output_dir).expanduser().resolve()),
        "artifact_manifest_sha256": manifest.artifact_manifest_sha256,
        "summary": manifest.summary,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _cmd_verify_m04a_validation_manifest(args: argparse.Namespace) -> int:
    source_capture = capture_runtime_source()
    _require_imported_source_identity(source_capture.fingerprint_sha256)
    from .m04a_validation_manifest import read_validation_episode_manifest

    manifest = read_validation_episode_manifest(
        args.manifest_dir,
        expected_artifact_manifest_sha256=args.artifact_manifest_sha256,
    )
    report = {
        "status": "verified_m04a_validation_episode_manifest",
        "artifact_manifest_sha256": manifest.artifact_manifest_sha256,
        "summary": manifest.summary,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _cmd_preflight_m04a(args: argparse.Namespace) -> int:
    from .m04a_preflight_command import run_preflight_command

    report = run_preflight_command(
        launch_plan_path=args.launch_plan,
        launch_plan_sha256=args.launch_plan_sha256,
        python_runtime_lock_path=args.python_runtime_lock,
        python_runtime_lock_sha256=args.python_runtime_lock_sha256,
        validation_manifest_dir=args.validation_manifest_dir,
        lock_handshake_path=args.lock_handshake_artifact,
        lock_handshake_sha256=args.lock_handshake_artifact_sha256,
        gpu_uuid=args.gpu_uuid,
        visible_root=args.visible_root,
        launcher_path=args.launcher_path,
        cost_probe_scratch_dir=args.cost_probe_scratch_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _cmd_train_m04a(args: argparse.Namespace) -> int:
    from .m04a_campaign_command import run_training_campaign_command

    stdout_available = True

    def emit_stdout(payload: Mapping[str, Any], *, compact: bool) -> None:
        nonlocal stdout_available
        if not stdout_available:
            return
        rendered = json.dumps(
            dict(payload),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":") if compact else None,
            indent=None if compact else 2,
        )
        try:
            sys.stdout.write(rendered + "\n")
            sys.stdout.flush()
        except BrokenPipeError:
            # Progress and the post-commit report are observational side channels.
            # A disconnected SSH/client reader must not invalidate otherwise exact
            # training.  Replacing the broken descriptor also prevents CPython's
            # shutdown flush from turning a completed command into exit status 120.
            stdout_available = False
            null_descriptor = None
            try:
                null_descriptor = os.open(
                    os.devnull,
                    os.O_WRONLY | getattr(os, "O_CLOEXEC", 0),
                )
                os.dup2(null_descriptor, sys.stdout.fileno())
            except (OSError, ValueError):
                pass
            finally:
                if null_descriptor is not None:
                    os.close(null_descriptor)

    def emit_progress(row: Mapping[str, Any]) -> None:
        emit_stdout(row, compact=True)

    report = run_training_campaign_command(
        launch_plan_path=args.launch_plan,
        launch_plan_sha256=args.launch_plan_sha256,
        python_runtime_lock_path=args.python_runtime_lock,
        python_runtime_lock_sha256=args.python_runtime_lock_sha256,
        validation_manifest_dir=args.validation_manifest_dir,
        lock_handshake_path=args.lock_handshake_artifact,
        lock_handshake_sha256=args.lock_handshake_artifact_sha256,
        gpu_uuid=args.gpu_uuid,
        visible_root=args.visible_root,
        launcher_path=args.launcher_path,
        cost_probe_scratch_dir=args.cost_probe_scratch_dir,
        production_data_root=args.production_data_root,
        working_dir=args.working_dir,
        output_dir=args.output_dir,
        progress_callback=emit_progress,
    )
    emit_stdout(report, compact=False)
    return 0


def _m04a_runtime_command_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--launch-plan", required=True)
    parser.add_argument("--launch-plan-sha256", required=True)
    parser.add_argument("--python-runtime-lock", required=True)
    parser.add_argument("--python-runtime-lock-sha256", required=True)
    parser.add_argument("--validation-manifest-dir", required=True)
    parser.add_argument("--lock-handshake-artifact", required=True)
    parser.add_argument("--lock-handshake-artifact-sha256", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--visible-root", required=True)
    parser.add_argument("--launcher-path", required=True)
    parser.add_argument("--cost-probe-scratch-dir", required=True)
    parser.add_argument("--output-dir", required=True)


def _cmd_materialize_m04a_input_bundle(args: argparse.Namespace) -> int:
    from .m04a_input_bundle import (
        M04AInputBundleSources,
        materialize_m04a_input_bundle,
        replay_validate_m04a_input_bundle,
    )

    sources = M04AInputBundleSources(
        sanitized_bundle_dir=Path(args.sanitized_bundle_dir),
        sanitized_artifact_manifest_sha256=args.sanitized_artifact_manifest_sha256,
        validation_manifest_dir=Path(args.validation_manifest_dir),
        validation_artifact_manifest_sha256=(args.validation_artifact_manifest_sha256),
        runtime_source_zip_path=Path(args.runtime_source_zip),
        runtime_source_zip_sha256=args.runtime_source_zip_sha256,
        runtime_source_fingerprint_sha256=(args.runtime_source_fingerprint_sha256),
        test_snapshot_zip_path=Path(args.test_snapshot_zip),
        test_snapshot_zip_sha256=args.test_snapshot_zip_sha256,
        test_source_fingerprint_sha256=args.test_source_fingerprint_sha256,
        launcher_path=Path(args.launcher_path),
        launcher_sha256=args.launcher_sha256,
        frozen_contract_path=Path(args.frozen_contract),
        frozen_contract_sha256=args.frozen_contract_sha256,
        conda_explicit_path=Path(args.conda_explicit),
        conda_explicit_sha256=args.conda_explicit_sha256,
        python_runtime_lock_path=Path(args.python_runtime_lock),
        python_runtime_lock_sha256=args.python_runtime_lock_sha256,
        run_id=args.run_id,
        remote_project_root=args.remote_project_root,
        attempt_nonce=args.attempt_nonce,
        ordered_import_roots=tuple(args.ordered_import_root),
    )
    bundle = materialize_m04a_input_bundle(
        args.visible_output_dir,
        control_output_directory=args.control_output_dir,
        sources=sources,
    )
    report = replay_validate_m04a_input_bundle(bundle, sources=sources)
    report = {
        **report,
        "visible_root": str(bundle.visible_root),
        "control_root": str(bundle.control_root),
        "launch_plan_path": str(bundle.launch_plan_path),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _add_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--task-dir",
        required=True,
        help="Official split directory containing task JSON files",
    )
    parser.add_argument(
        "--dataset-name", required=True, choices=sorted(DATASET_ORIGINS)
    )
    parser.add_argument(
        "--split-name", required=True, choices=sorted(SPLIT_DIRECTORIES)
    )
    parser.add_argument(
        "--repo-dir",
        required=True,
        help="Clean official dataset repository containing task-dir",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="afts-arc")
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser(
        "audit-data", help="Validate and fingerprint a task directory"
    )
    _add_dataset_args(audit)
    audit.add_argument(
        "--hash-only", action="store_true", help="Do not parse task contents"
    )
    audit.add_argument(
        "--release-gate-id",
        help="Required for any public_evaluation access; recorded in the audit output",
    )
    audit.add_argument(
        "--output-dir",
        help="Optional new atomic evidence directory containing audit JSON and source snapshot",
    )
    audit.set_defaults(handler=_cmd_audit_data)

    d4 = subparsers.add_parser(
        "run-d4-baseline", help="Run the deterministic D4 harness baseline"
    )
    _add_dataset_args(d4)
    d4.add_argument("--output-dir", required=True)
    d4.add_argument(
        "--max-tasks",
        type=int,
        help="Select only the first N task files by stable filename order and bind that subset hash",
    )
    d4.add_argument(
        "--scorer-repo-dir",
        required=True,
        help="Clean pinned official arc-agi-benchmarking repository",
    )
    d4.add_argument(
        "--release-gate-id",
        help="Required only for public_evaluation scoring; recorded in the manifest",
    )
    d4.add_argument(
        "--orchestrator-id", help="Optional parent invocation or goal identifier"
    )
    d4.set_defaults(handler=_cmd_run_d4)

    e01_dataset = subparsers.add_parser(
        "build-e01a-synthetic-dataset",
        help="Publish separate blind and oracle bundles for symbolic controls",
    )
    e01_dataset.add_argument("--blind-output-dir", required=True)
    e01_dataset.add_argument("--oracle-output-dir", required=True)
    e01_dataset.add_argument("--seed", type=int, default=0)
    e01_dataset.add_argument("--tasks-per-family", type=int, default=3)
    e01_dataset.set_defaults(handler=_cmd_build_e01a_dataset)

    e01_public_dataset = subparsers.add_parser(
        "build-e01a-public-train-smoke-dataset",
        help="Publish a pinned ARC-AGI-2 public-training development smoke",
    )
    _add_dataset_args(e01_public_dataset)
    e01_public_dataset.add_argument("--blind-output-dir", required=True)
    e01_public_dataset.add_argument("--oracle-output-dir", required=True)
    e01_public_dataset.add_argument("--smoke-count", type=int, default=20)
    e01_public_dataset.add_argument(
        "--selection-salt", default="afts-e01-public-train-development-smoke-v1"
    )
    e01_public_dataset.set_defaults(handler=_cmd_build_e01a_public_dataset)

    e01_pool = subparsers.add_parser(
        "run-e01a-symbolic-pool",
        help="Build an oracle-unread typed-DSL pool from a blind dataset bundle",
    )
    e01_pool.add_argument("--blind-dataset-dir", required=True)
    e01_pool.add_argument("--output-dir", required=True)
    e01_pool.add_argument("--max-depth", type=int, default=2)
    e01_pool.add_argument("--beam-width", type=int, default=64)
    e01_pool.add_argument("--max-instruction-options", type=int, default=64)
    e01_pool.add_argument("--max-exact-programs", type=int, default=128)
    e01_pool.set_defaults(handler=_cmd_run_e01a_pool)

    e01_eval = subparsers.add_parser(
        "evaluate-e01a-symbolic-pool",
        help="Evaluate a frozen symbolic pool against its separate oracle bundle",
    )
    e01_eval.add_argument("--blind-dataset-dir", required=True)
    e01_eval.add_argument("--oracle-dataset-dir", required=True)
    e01_eval.add_argument("--pool-dir", required=True)
    e01_eval.add_argument("--output-dir", required=True)
    e01_eval.set_defaults(handler=_cmd_evaluate_e01a_pool)

    e01_public_eval = subparsers.add_parser(
        "evaluate-e01a-public-train-pool",
        help="Evaluate a frozen symbolic pool on the public-training development smoke",
    )
    e01_public_eval.add_argument("--blind-dataset-dir", required=True)
    e01_public_eval.add_argument("--oracle-dataset-dir", required=True)
    e01_public_eval.add_argument("--pool-dir", required=True)
    e01_public_eval.add_argument("--output-dir", required=True)
    e01_public_eval.set_defaults(handler=_cmd_evaluate_e01a_public_pool)

    e01_verify = subparsers.add_parser(
        "verify-e01a-eval",
        help="Recompute E01a aggregate metrics from a published evaluation bundle",
    )
    e01_verify.add_argument("--eval-bundle", required=True)
    e01_verify.add_argument("--blind-dataset-dir", required=True)
    e01_verify.add_argument("--oracle-dataset-dir", required=True)
    e01_verify.add_argument("--pool-dir", required=True)
    e01_verify.set_defaults(handler=_cmd_verify_e01a_eval)

    rearc_cache = subparsers.add_parser(
        "build-m04a-rearc-cache",
        help="Build one production ReARC mmap cache from a committed sanitized shard",
    )
    rearc_cache.add_argument("--sanitized-bundle-dir", required=True)
    rearc_cache.add_argument("--sanitized-artifact-manifest-sha256", required=True)
    rearc_cache.add_argument("--fold", required=True, choices=("train", "validation"))
    rearc_cache.add_argument("--output-dir", required=True)
    rearc_cache.set_defaults(handler=_cmd_build_m04a_rearc_cache)

    validation_manifest = subparsers.add_parser(
        "build-m04a-validation-manifest",
        help="Materialize the frozen full validation episode JSONL",
    )
    _indexed_m04a_args(validation_manifest)
    validation_manifest.add_argument("--output-dir", required=True)
    validation_manifest.set_defaults(handler=_cmd_build_m04a_validation_manifest)

    verify_validation_manifest = subparsers.add_parser(
        "verify-m04a-validation-manifest",
        help="Verify a committed M04a validation episode bundle",
    )
    verify_validation_manifest.add_argument("--manifest-dir", required=True)
    verify_validation_manifest.add_argument("--artifact-manifest-sha256", required=True)
    verify_validation_manifest.set_defaults(
        handler=_cmd_verify_m04a_validation_manifest
    )

    preflight_m04a = subparsers.add_parser(
        "preflight-m04a",
        help="Run the exact committed CUDA preflight and publish a diagnostic gate",
    )
    _m04a_runtime_command_args(preflight_m04a)
    preflight_m04a.set_defaults(handler=_cmd_preflight_m04a)

    train_m04a = subparsers.add_parser(
        "train-m04a",
        help="Run same-process exact preflight and the complete primary campaign",
    )
    _m04a_runtime_command_args(train_m04a)
    train_m04a.add_argument("--production-data-root", required=True)
    train_m04a.add_argument("--working-dir", required=True)
    train_m04a.set_defaults(handler=_cmd_train_m04a)

    input_bundle = subparsers.add_parser(
        "materialize-m04a-input-bundle",
        help="Publish the flat committed M04a input root and external launch plan",
    )
    input_bundle.add_argument("--sanitized-bundle-dir", required=True)
    input_bundle.add_argument("--sanitized-artifact-manifest-sha256", required=True)
    input_bundle.add_argument("--validation-manifest-dir", required=True)
    input_bundle.add_argument("--validation-artifact-manifest-sha256", required=True)
    input_bundle.add_argument("--runtime-source-zip", required=True)
    input_bundle.add_argument("--runtime-source-zip-sha256", required=True)
    input_bundle.add_argument("--runtime-source-fingerprint-sha256", required=True)
    input_bundle.add_argument("--test-snapshot-zip", required=True)
    input_bundle.add_argument("--test-snapshot-zip-sha256", required=True)
    input_bundle.add_argument("--test-source-fingerprint-sha256", required=True)
    input_bundle.add_argument("--launcher-path", required=True)
    input_bundle.add_argument("--launcher-sha256", required=True)
    input_bundle.add_argument("--frozen-contract", required=True)
    input_bundle.add_argument("--frozen-contract-sha256", required=True)
    input_bundle.add_argument("--conda-explicit", required=True)
    input_bundle.add_argument("--conda-explicit-sha256", required=True)
    input_bundle.add_argument("--python-runtime-lock", required=True)
    input_bundle.add_argument("--python-runtime-lock-sha256", required=True)
    input_bundle.add_argument("--run-id", required=True)
    input_bundle.add_argument("--remote-project-root", required=True)
    input_bundle.add_argument("--attempt-nonce", required=True)
    input_bundle.add_argument(
        "--ordered-import-root",
        required=True,
        action="append",
        help="Repeat in exact source, purelib, optional-platlib order",
    )
    input_bundle.add_argument("--visible-output-dir", required=True)
    input_bundle.add_argument("--control-output-dir", required=True)
    input_bundle.set_defaults(handler=_cmd_materialize_m04a_input_bundle)
    return parser


def _main_in_process(argv: list[str] | None = None) -> int:
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(effective_argv)
    if getattr(args, "max_tasks", None) is not None and args.max_tasks <= 0:
        raise ValueError("--max-tasks must be positive")
    if (
        getattr(args, "tasks_per_family", None) is not None
        and args.tasks_per_family <= 0
    ):
        raise ValueError("--tasks-per-family must be positive")
    if getattr(args, "smoke_count", None) is not None and args.smoke_count <= 0:
        raise ValueError("--smoke-count must be positive")
    args.effective_command = ["afts-arc", *effective_argv]
    return int(args.handler(args))


def main(argv: list[str] | None = None) -> int:
    _require_fresh_source_loader()
    return _main_in_process(argv)


if __name__ == "__main__":
    raise SystemExit(main())
