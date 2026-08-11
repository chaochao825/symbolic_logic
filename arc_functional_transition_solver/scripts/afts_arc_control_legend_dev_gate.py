"""Freeze and score the outcome-exposed control-legend v0.1 dev gate."""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = str((PROJECT_ROOT / "src").resolve())
if SOURCE_ROOT not in sys.path:
    sys.path.insert(0, SOURCE_ROOT)

from afts_arc.blind import BlindTask  # noqa: E402
from afts_arc.control_legend import (  # noqa: E402
    CONTROL_LEGEND_DSL_VERSION,
    CONTROL_LEGEND_PROGRAM_CAP,
    ControlLegendProgram,
    ControlLegendProgramScore,
    control_legend_program_id,
    enumerate_control_legend_programs,
    execute_control_legend,
    score_control_legend_program,
)
from afts_arc.experiment_safety import (  # noqa: E402
    atomic_write_json,
    canonical_sha256,
    file_sha256,
    runtime_metadata,
)
from afts_arc.grid import Grid, as_grid, grid_to_lists  # noqa: E402
from afts_arc.hybrid.types import canonical_json  # noqa: E402
from afts_arc.task import ARCPair  # noqa: E402


FREEZE_SCHEMA = "afts.control-legend-dev-candidate-freeze/v0.1"
RESULT_SCHEMA = "afts.control-legend-dev-result/v0.1"
EXPECTED_COHORT_ID = "dfee67cf54613ba721e1fe46723967c95020715dff1a40500fe498055805e3ac"
EXPECTED_V02_FREEZE_ID = (
    "85811f28942fc447d529be60e217d4cd4b5326c485ea901dda3d19b983fe32e5"
)
EXPECTED_V02_RESULT_ID = (
    "14d4ad29bec0da59fc6e405b72677cc917cf1f42d41e25af776003d3ccb3b884"
)
SELECTED_DEVELOPMENT_TASK_IDS = ("5adee1b2", "e4888269")
SELECTED_CANDIDATE_CAP = 2
EXPECTED_SEMANTIC_TEST_NAMES = frozenset(
    {
        "test_ambiguous_control_lanes_are_rejected",
        "test_demo_exact_synthesis_rejects_near_miss_render",
        "test_enumeration_is_bounded_and_query_output_free",
        "test_fill_exterior_bbox_preserves_enclosed_holes",
        "test_ordered_rewrite_applies_rules_sequentially_and_protects_control",
        "test_program_round_trip_is_strict_and_content_addressed",
        "test_repeated_pairs_collapse_without_losing_control_cells",
        "test_vertical_lane_is_supported_by_the_same_typed_program",
    }
)


def _load_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON artifact must be an object: {path}")
    return payload


def _write_new(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to replace artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, payload)


def _validate_content_id(payload: Mapping[str, object], field: str) -> None:
    body = dict(payload)
    declared = body.pop(field)
    if declared != canonical_sha256(body):
        raise ValueError(f"{field} does not match canonical content")


def _blind_task(payload: Mapping[str, object]) -> BlindTask:
    if set(payload) != {"train", "test"}:
        raise ValueError("cohort blind task has missing or unknown fields")
    train = payload["train"]
    test = payload["test"]
    if not isinstance(train, list) or not train:
        raise ValueError("blind task requires demonstrations")
    if not isinstance(test, list) or not test:
        raise ValueError("blind task requires query inputs")
    for query in test:
        if not isinstance(query, Mapping) or set(query) != {"input", "output"}:
            raise ValueError("blind query must contain input and sentinel output")
        if query["output"] != query["input"]:
            raise ValueError("blind query output is not the input-copy sentinel")
    return BlindTask.from_observations(
        train=tuple(
            ARCPair(as_grid(pair["input"]), as_grid(pair["output"])) for pair in train
        ),
        test_inputs=tuple(as_grid(pair["input"]) for pair in test),
    )


def _outputs_payload(outputs: Sequence[Grid]) -> list[list[list[int]]]:
    return [grid_to_lists(output) for output in outputs]


def _score_key(score: ControlLegendProgramScore) -> tuple[object, ...]:
    return (
        -score.exact_demo_count,
        score.mismatch_count,
        score.program.description_bits,
        control_legend_program_id(score.program),
    )


def _candidate(
    program: ControlLegendProgram, task: BlindTask
) -> dict[str, object] | None:
    executions = tuple(
        execute_control_legend(program, query) for query in task.test_inputs
    )
    if any(not execution.ok or execution.output is None for execution in executions):
        return None
    outputs = tuple(
        execution.output for execution in executions if execution.output is not None
    )
    return {
        "source": "control_legend_program",
        "program_id": control_legend_program_id(program),
        "description_bits": program.description_bits,
        "program": program.to_json_dict(),
        "functional_trace": list(program.functional_trace),
        "query_outputs": _outputs_payload(outputs),
    }


def _candidate_key(candidate: Mapping[str, object]) -> tuple[object, ...]:
    return (
        candidate["description_bits"],
        candidate["program_id"],
        candidate["source"],
    )


def _output_signature(candidate: Mapping[str, object]) -> str:
    return canonical_json(candidate["query_outputs"])


def _deduplicate_candidates(
    candidates: Sequence[dict[str, object]],
) -> tuple[dict[str, object], ...]:
    by_outputs: dict[str, dict[str, object]] = {}
    for candidate in candidates:
        signature = _output_signature(candidate)
        if signature not in by_outputs or _candidate_key(candidate) < _candidate_key(
            by_outputs[signature]
        ):
            by_outputs[signature] = candidate
    return tuple(sorted(by_outputs.values(), key=_candidate_key))


def _freeze_task(
    task_record: Mapping[str, object],
    blind_root: Path,
    existing_task: Mapping[str, object],
) -> dict[str, object]:
    task_id = task_record["task_id"]
    if existing_task["task_id"] != task_id:
        raise ValueError("existing candidate task order does not match cohort")
    blind_path = blind_root / f"{task_id}.json"
    if file_sha256(blind_path) != task_record["blind_sha256"]:
        raise ValueError(f"blind task hash mismatch: {task_id}")
    task = _blind_task(_load_object(blind_path))
    programs = enumerate_control_legend_programs(task)
    if len(programs) != CONTROL_LEGEND_PROGRAM_CAP:
        raise ValueError(f"control-legend grammar violates its cap: {task_id}")
    scores = tuple(score_control_legend_program(program, task) for program in programs)
    exact_scores = tuple(
        sorted((score for score in scores if score.all_demo_exact), key=_score_key)
    )
    candidates = tuple(
        candidate
        for score in exact_scores
        if (candidate := _candidate(score.program, task)) is not None
    )
    deduplicated = _deduplicate_candidates(candidates)
    existing_candidates = existing_task["full"]["candidates"]
    if not isinstance(existing_candidates, list):
        raise TypeError("existing v0.2 candidates must be a list")
    existing_signatures = {
        _output_signature(candidate) for candidate in existing_candidates
    }
    novel = tuple(
        candidate
        for candidate in deduplicated
        if _output_signature(candidate) not in existing_signatures
    )
    return {
        "task_id": task_id,
        "blind_sha256": task_record["blind_sha256"],
        "source_sha256": task_record["source_sha256"],
        "selected_development_task": task_id in SELECTED_DEVELOPMENT_TASK_IDS,
        "demo_count": len(task.train),
        "query_count": len(task.test_inputs),
        "native_cost": {
            "program_trials": len(programs),
            "demo_program_executions": len(programs) * len(task.train),
            "query_program_executions": len(exact_scores) * len(task.test_inputs),
            "total_program_executions": len(programs) * len(task.train)
            + len(exact_scores) * len(task.test_inputs),
        },
        "demo_exact_program_count": len(exact_scores),
        "candidate_count": len(deduplicated),
        "candidates": list(deduplicated),
        "selected_candidates": list(deduplicated[:SELECTED_CANDIDATE_CAP]),
        "existing_v02_candidate_count": len(existing_candidates),
        "novel_frontier_count": len(novel),
        "frontier_changed": bool(novel),
        "novel_candidates": list(novel),
    }


def _source_contract() -> dict[str, object]:
    paths = (
        PROJECT_ROOT / "src" / "afts_arc" / "control_legend.py",
        Path(__file__).resolve(),
        PROJECT_ROOT / "tests" / "test_control_legend.py",
        PROJECT_ROOT
        / "notes"
        / "design"
        / "control-legend-program-development-gate-v0.1.md",
    )
    return {
        str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): file_sha256(path)
        for path in paths
    }


def _semantic_test_receipt(path: Path) -> dict[str, object]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    if not suites:
        raise ValueError("semantic test report contains no test suites")
    tests = sum(int(suite.attrib["tests"]) for suite in suites)
    failures = sum(int(suite.attrib["failures"]) for suite in suites)
    errors = sum(int(suite.attrib["errors"]) for suite in suites)
    skipped = sum(int(suite.attrib["skipped"]) for suite in suites)
    test_names = tuple(
        sorted(test_case.attrib["name"] for test_case in root.findall(".//testcase"))
    )
    passed = (
        tests == len(EXPECTED_SEMANTIC_TEST_NAMES)
        and failures == 0
        and errors == 0
        and skipped == 0
        and frozenset(test_names) == EXPECTED_SEMANTIC_TEST_NAMES
    )
    if not passed:
        raise ValueError("semantic test report does not pass the frozen fixture gate")
    return {
        "report_sha256": file_sha256(path),
        "tests": tests,
        "failures": failures,
        "errors": errors,
        "skipped": skipped,
        "test_names": list(test_names),
        "passed": passed,
    }


def _construct_freeze(
    manifest_path: Path,
    blind_root: Path,
    existing_freeze_path: Path,
    semantic_test_report: Path,
) -> dict[str, object]:
    manifest = _load_object(manifest_path)
    if manifest["cohort_id"] != EXPECTED_COHORT_ID:
        raise ValueError("cohort ID does not match the exposed development cohort")
    tasks = manifest["tasks"]
    if not isinstance(tasks, list) or len(tasks) != 50:
        raise ValueError("development cohort must contain exactly 50 tasks")
    existing_freeze = _load_object(existing_freeze_path)
    _validate_content_id(existing_freeze, "freeze_id")
    if existing_freeze["freeze_id"] != EXPECTED_V02_FREEZE_ID:
        raise ValueError("existing v0.2 candidate freeze ID mismatch")
    if existing_freeze["cohort_id"] != EXPECTED_COHORT_ID:
        raise ValueError("existing v0.2 candidate cohort mismatch")
    existing_tasks = existing_freeze["tasks"]
    if not isinstance(existing_tasks, list) or len(existing_tasks) != len(tasks):
        raise ValueError("existing v0.2 task count mismatch")
    records = [
        _freeze_task(task, blind_root, existing_task)
        for task, existing_task in zip(tasks, existing_tasks, strict=True)
    ]
    content = {
        "schema": FREEZE_SCHEMA,
        "status": "outcome_exposed_development_only",
        "cohort_id": manifest["cohort_id"],
        "cohort_manifest_sha256": file_sha256(manifest_path),
        "existing_v02_freeze_id": existing_freeze["freeze_id"],
        "existing_v02_freeze_sha256": file_sha256(existing_freeze_path),
        "control_legend_dsl_version": CONTROL_LEGEND_DSL_VERSION,
        "source_contract": _source_contract(),
        "semantic_test_receipt": _semantic_test_receipt(semantic_test_report),
        "budgets": {
            "program_cap_per_task": CONTROL_LEGEND_PROGRAM_CAP,
            "selected_candidate_cap_per_task": SELECTED_CANDIDATE_CAP,
        },
        "selected_development_task_ids": list(SELECTED_DEVELOPMENT_TASK_IDS),
        "query_gold_read": False,
        "public_evaluation_read": False,
        "controller_training_started": False,
        "runtime": runtime_metadata(),
        "tasks": records,
    }
    return {"freeze_id": canonical_sha256(content), **content}


def freeze(args: argparse.Namespace) -> None:
    first = _construct_freeze(
        args.cohort_manifest,
        args.blind_root,
        args.existing_freeze,
        args.semantic_test_report,
    )
    second = _construct_freeze(
        args.cohort_manifest,
        args.blind_root,
        args.existing_freeze,
        args.semantic_test_report,
    )
    if canonical_json(first) != canonical_json(second):
        raise ValueError("candidate construction is not replay-identical")
    first["construction_replay_identical"] = True
    body = dict(first)
    body.pop("freeze_id")
    first["freeze_id"] = canonical_sha256(body)
    _write_new(args.output, first)
    print(json.dumps(first, indent=2, sort_keys=True))


def _gold_outputs(path: Path) -> tuple[Grid, ...]:
    payload = _load_object(path)
    if set(payload) != {"train", "test"}:
        raise ValueError(f"ARC task has missing or unknown fields: {path}")
    test = payload["test"]
    if not isinstance(test, list) or not test:
        raise ValueError(f"ARC task has no query outputs: {path}")
    return tuple(as_grid(pair["output"]) for pair in test)


def _candidate_query_exact(
    candidate: Mapping[str, object], gold_outputs: Sequence[Grid]
) -> tuple[bool, ...]:
    outputs = candidate["query_outputs"]
    if not isinstance(outputs, list) or len(outputs) != len(gold_outputs):
        raise ValueError("candidate query outputs do not close against gold")
    return tuple(
        as_grid(output) == gold
        for output, gold in zip(outputs, gold_outputs, strict=True)
    )


def _pool_metrics(
    task: Mapping[str, object], gold_outputs: Sequence[Grid]
) -> dict[str, object]:
    candidates = task["candidates"]
    selected = task["selected_candidates"]
    if not isinstance(candidates, list) or not isinstance(selected, list):
        raise TypeError("candidate pools must be lists")
    oracle_pairs = [False] * len(gold_outputs)
    for candidate in candidates:
        exact = _candidate_query_exact(candidate, gold_outputs)
        oracle_pairs = [
            previous or current
            for previous, current in zip(oracle_pairs, exact, strict=True)
        ]
    pass2_pairs = [False] * len(gold_outputs)
    for candidate in selected[:SELECTED_CANDIDATE_CAP]:
        exact = _candidate_query_exact(candidate, gold_outputs)
        pass2_pairs = [
            previous or current
            for previous, current in zip(pass2_pairs, exact, strict=True)
        ]
    return {
        "candidate_count": len(candidates),
        "selected_candidate_count": len(selected[:SELECTED_CANDIDATE_CAP]),
        "oracle_pair_coverage": oracle_pairs,
        "pass2_pair_coverage": pass2_pairs,
        "raw_oracle": all(oracle_pairs),
        "pass2": all(pass2_pairs),
    }


def score(args: argparse.Namespace) -> None:
    freeze_payload = _load_object(args.candidate_freeze)
    _validate_content_id(freeze_payload, "freeze_id")
    if freeze_payload["schema"] != FREEZE_SCHEMA:
        raise ValueError("candidate freeze schema mismatch")
    if freeze_payload["query_gold_read"] is not False:
        raise ValueError("candidate freeze is not query blind")
    if freeze_payload["construction_replay_identical"] is not True:
        raise ValueError("candidate freeze did not pass construction replay")
    if freeze_payload["source_contract"] != _source_contract():
        raise ValueError("candidate freeze source contract no longer matches checkout")
    if freeze_payload["semantic_test_receipt"]["passed"] is not True:
        raise ValueError("candidate freeze lacks a passing semantic test receipt")
    existing_result = _load_object(args.existing_result)
    _validate_content_id(existing_result, "result_id")
    if existing_result["result_id"] != EXPECTED_V02_RESULT_ID:
        raise ValueError("existing combined v0.2 result ID mismatch")
    existing_by_task = {task["task_id"]: task for task in existing_result["tasks"]}

    rows = []
    for task in freeze_payload["tasks"]:
        task_id = task["task_id"]
        task_path = args.task_root / f"{task_id}.json"
        if file_sha256(task_path) != task["source_sha256"]:
            raise ValueError(f"gold task hash mismatch: {task_id}")
        gold_outputs = _gold_outputs(task_path)
        metrics = _pool_metrics(task, gold_outputs)
        existing = existing_by_task[task_id]
        existing_solved = existing["baseline_pass2"] or existing["full"]["pass2"]
        rows.append(
            {
                "task_id": task_id,
                "selected_development_task": task["selected_development_task"],
                "demo_exact_program_count": task["demo_exact_program_count"],
                "novel_frontier_count": task["novel_frontier_count"],
                "frontier_changed": task["frontier_changed"],
                "native_cost": task["native_cost"],
                "full": metrics,
                "existing_solved": existing_solved,
                "unique_pass2_vs_existing": metrics["pass2"] and not existing_solved,
            }
        )
    selected = [row for row in rows if row["selected_development_task"]]
    if tuple(row["task_id"] for row in selected) != SELECTED_DEVELOPMENT_TASK_IDS:
        raise ValueError("selected development task order changed")
    counts = {
        "task_count": len(rows),
        "demo_exact_task_count": sum(
            row["demo_exact_program_count"] > 0 for row in rows
        ),
        "novel_frontier_task_count": sum(row["frontier_changed"] for row in rows),
        "raw_oracle": sum(row["full"]["raw_oracle"] for row in rows),
        "pass2": sum(row["full"]["pass2"] for row in rows),
        "unique_pass2_vs_existing": sum(
            row["unique_pass2_vs_existing"] for row in rows
        ),
        "selected_demo_exact": sum(
            row["demo_exact_program_count"] > 0 for row in selected
        ),
        "selected_novel_frontier": sum(row["frontier_changed"] for row in selected),
        "selected_pass2": sum(row["full"]["pass2"] for row in selected),
        "selected_unique_pass2": sum(
            row["unique_pass2_vs_existing"] for row in selected
        ),
    }
    selected_task_count = len(SELECTED_DEVELOPMENT_TASK_IDS)
    gates = {
        "semantic_fixture_gate": freeze_payload["semantic_test_receipt"]["passed"],
        "selected_demo_exact_gate": counts["selected_demo_exact"]
        == selected_task_count,
        "selected_novel_frontier_gate": counts["selected_novel_frontier"]
        == selected_task_count,
        "selected_query_exact_gate": counts["selected_pass2"] == selected_task_count,
        "selected_unique_coverage_gate": counts["selected_unique_pass2"]
        == selected_task_count,
        "representation_development_gate": all(
            (
                counts["selected_demo_exact"] == selected_task_count,
                counts["selected_novel_frontier"] == selected_task_count,
                counts["selected_pass2"] == selected_task_count,
                counts["selected_unique_pass2"] == selected_task_count,
            )
        ),
        "fresh_5_per_100_gate": False,
        "typed_repair_gate": False,
        "controller_training_authorized": False,
        "causal_function_switching_supported": False,
    }
    content = {
        "schema": RESULT_SCHEMA,
        "status": "outcome_exposed_development_only",
        "candidate_freeze_id": freeze_payload["freeze_id"],
        "candidate_freeze_sha256": file_sha256(args.candidate_freeze),
        "existing_v02_result_id": existing_result["result_id"],
        "existing_v02_result_sha256": file_sha256(args.existing_result),
        "cohort_id": freeze_payload["cohort_id"],
        "query_gold_read_after_freeze": True,
        "public_evaluation_read": False,
        "controller_training_started": False,
        "counts": counts,
        "gates": gates,
        "tasks": rows,
    }
    result = {"result_id": canonical_sha256(content), **content}
    _write_new(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--cohort-manifest", type=Path, required=True)
    freeze_parser.add_argument("--blind-root", type=Path, required=True)
    freeze_parser.add_argument("--existing-freeze", type=Path, required=True)
    freeze_parser.add_argument("--semantic-test-report", type=Path, required=True)
    freeze_parser.add_argument("--output", type=Path, required=True)
    freeze_parser.set_defaults(function=freeze)
    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--candidate-freeze", type=Path, required=True)
    score_parser.add_argument("--existing-result", type=Path, required=True)
    score_parser.add_argument("--task-root", type=Path, required=True)
    score_parser.add_argument("--output", type=Path, required=True)
    score_parser.set_defaults(function=score)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
