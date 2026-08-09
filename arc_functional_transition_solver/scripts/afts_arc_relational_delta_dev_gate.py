"""Freeze and score the outcome-exposed relational-delta v0.2 dev gate."""

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
from afts_arc.experiment_safety import (  # noqa: E402
    atomic_write_json,
    canonical_sha256,
    file_sha256,
    runtime_metadata,
)
from afts_arc.grid import Grid, as_grid, grid_to_lists  # noqa: E402
from afts_arc.hybrid.types import canonical_json  # noqa: E402
from afts_arc.relational_delta import (  # noqa: E402
    RELATIONAL_DELTA_DSL_VERSION,
    RelationalDeltaProgram,
    RelationalDeltaProgramScore,
    compile_relational_delta_failure_certificate,
    enumerate_relational_delta_programs,
    execute_relational_delta,
    relational_delta_near_miss_quality,
    relational_delta_program_id,
    score_relational_delta_program,
    single_slot_relational_delta_variants,
)
from afts_arc.task import ARCPair  # noqa: E402


FREEZE_SCHEMA = "afts.relational-delta-dev-candidate-freeze/v0.2"
RESULT_SCHEMA = "afts.relational-delta-dev-result/v0.2"
EXPECTED_COHORT_ID = "dfee67cf54613ba721e1fe46723967c95020715dff1a40500fe498055805e3ac"
EXPECTED_BASELINE_RESULT_ID = (
    "916221630a9a3472e43e05a33932486f1d57f6a49d274bc75677c8b4ae678f6e"
)
DIAGNOSTIC_TASK_IDS = ("7acdf6d3", "320afe60", "3ad05f52")
PROGRAM_CAP = 256
INITIAL_PROGRAM_TRIALS = 64
ACTION_PROGRAM_TRIALS = 16
EXPECTED_SEMANTIC_TEST_NAMES = frozenset(
    {
        "test_add_uses_a_coordinate_target_and_preserves_the_actor",
        "test_enumeration_is_query_blind_bounded_and_reaches_recolor_fixture",
        "test_erase_add_matches_actor_area_to_a_complete_row_interior",
        "test_failure_certificate_reaches_a_novel_single_slot_palette_edit",
        "test_fill_relation_region_writes_the_complete_enclosed_region",
        "test_near_miss_requires_identity_gain_precision_recall_and_topology",
        "test_overlapping_component_targets_are_rejected_as_topology_invalid",
        "test_program_round_trip_is_strict_and_content_addressed",
        "test_recolor_component_uses_topology_role_and_preserves_shapes",
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
            ARCPair(as_grid(pair["input"]), as_grid(pair["output"]))
            for pair in train
        ),
        test_inputs=tuple(as_grid(pair["input"]) for pair in test),
    )


def _outputs_payload(outputs: Sequence[Grid]) -> list[list[list[int]]]:
    return [grid_to_lists(output) for output in outputs]


def _candidate(
    program: RelationalDeltaProgram,
    task: BlindTask,
    *,
    source: str,
) -> dict[str, object] | None:
    executions = tuple(
        execute_relational_delta(program, query) for query in task.test_inputs
    )
    if any(not execution.ok or execution.output is None for execution in executions):
        return None
    outputs = tuple(
        execution.output for execution in executions if execution.output is not None
    )
    return {
        "source": source,
        "program_id": relational_delta_program_id(program),
        "description_bits": program.description_bits,
        "program": program.to_json_dict(),
        "query_outputs": _outputs_payload(outputs),
    }


def _candidate_key(candidate: Mapping[str, object]) -> tuple[object, ...]:
    return (
        candidate["description_bits"],
        candidate["program_id"],
        candidate["source"],
    )


def _deduplicate_candidates(
    candidates: Sequence[dict[str, object]],
) -> tuple[dict[str, object], ...]:
    by_outputs: dict[str, dict[str, object]] = {}
    for candidate in candidates:
        signature = canonical_json(candidate["query_outputs"])
        incumbent = by_outputs.get(signature)
        if incumbent is None or _candidate_key(candidate) < _candidate_key(incumbent):
            by_outputs[signature] = candidate
    return tuple(sorted(by_outputs.values(), key=_candidate_key))


def _score_key(score: RelationalDeltaProgramScore) -> tuple[object, ...]:
    return (
        -score.exact_demo_count,
        score.mismatch_count,
        score.program.description_bits,
        relational_delta_program_id(score.program),
    )


def _arm(
    *,
    programs: Sequence[RelationalDeltaProgram],
    padding_program: RelationalDeltaProgram,
    task: BlindTask,
    source: str,
    initial_program_ids: Sequence[str],
) -> dict[str, object]:
    actual = tuple(programs[:ACTION_PROGRAM_TRIALS])
    candidates = []
    trials = []
    for trial_index in range(ACTION_PROGRAM_TRIALS):
        padding = trial_index >= len(actual)
        program = padding_program if padding else actual[trial_index]
        demo_executions = tuple(
            execute_relational_delta(program, pair.input) for pair in task.train
        )
        query_executions = tuple(
            execute_relational_delta(program, query) for query in task.test_inputs
        )
        demo_exact = all(
            execution.ok
            and execution.output is not None
            and execution.output == pair.output
            for execution, pair in zip(demo_executions, task.train, strict=True)
        )
        query_valid = all(
            execution.ok and execution.output is not None
            for execution in query_executions
        )
        trials.append(
            {
                "trial_index": trial_index,
                "padding": padding,
                "program_id": relational_delta_program_id(program),
                "demo_exact": demo_exact,
                "query_valid": query_valid,
            }
        )
        if not padding and demo_exact and query_valid:
            candidate = _candidate(program, task, source=source)
            if candidate is not None:
                candidates.append(candidate)
    initial = frozenset(initial_program_ids)
    novel_ids = {
        relational_delta_program_id(program)
        for program in actual
        if relational_delta_program_id(program) not in initial
    }
    deduplicated = _deduplicate_candidates(candidates)
    return {
        "reserved_program_trials": ACTION_PROGRAM_TRIALS,
        "actual_program_trials": len(actual),
        "padding_program_trials": ACTION_PROGRAM_TRIALS - len(actual),
        "demo_execution_count": ACTION_PROGRAM_TRIALS * len(task.train),
        "query_execution_count": ACTION_PROGRAM_TRIALS * len(task.test_inputs),
        "total_execution_count": ACTION_PROGRAM_TRIALS
        * (len(task.train) + len(task.test_inputs)),
        "novel_frontier_count": len(novel_ids),
        "frontier_changed": bool(novel_ids),
        "trials": trials,
        "candidates": list(deduplicated),
        "selected_candidates": list(deduplicated[:2]),
    }


def _freeze_task(
    task_record: Mapping[str, object], blind_root: Path
) -> dict[str, object]:
    task_id = task_record["task_id"]
    blind_path = blind_root / f"{task_id}.json"
    if file_sha256(blind_path) != task_record["blind_sha256"]:
        raise ValueError(f"blind task hash mismatch: {task_id}")
    task = _blind_task(_load_object(blind_path))
    programs = enumerate_relational_delta_programs(task)
    if not programs or len(programs) > PROGRAM_CAP:
        raise ValueError(f"relational delta grammar violates its cap: {task_id}")
    scores = tuple(score_relational_delta_program(program, task) for program in programs)
    score_by_id = {
        relational_delta_program_id(score.program): score for score in scores
    }
    exact_scores = tuple(sorted((score for score in scores if score.all_demo_exact), key=_score_key))
    full_candidates = tuple(
        candidate
        for score in exact_scores
        if (candidate := _candidate(score.program, task, source="relational_delta_full"))
        is not None
    )
    full_deduplicated = _deduplicate_candidates(full_candidates)

    initial_programs = programs[:INITIAL_PROGRAM_TRIALS]
    initial_ids = tuple(relational_delta_program_id(program) for program in initial_programs)
    initial_candidates = tuple(
        candidate
        for program in initial_programs
        if score_by_id[relational_delta_program_id(program)].all_demo_exact
        if (candidate := _candidate(program, task, source="relational_delta_initial"))
        is not None
    )
    initial_deduplicated = _deduplicate_candidates(initial_candidates)

    eligible_scores = tuple(
        score
        for score in scores[: len(initial_programs)]
        if relational_delta_near_miss_quality(score.program, task).eligible
    )
    def quality_key(score: RelationalDeltaProgramScore) -> tuple[object, ...]:
        quality = relational_delta_near_miss_quality(score.program, task)
        return (
            quality.parent_mismatch_count,
            -quality.delta_precision,
            -quality.delta_recall,
            score.program.description_bits,
            relational_delta_program_id(score.program),
        )

    parent_score = min(eligible_scores, key=quality_key) if eligible_scores else None
    certificate = None
    variants: tuple[RelationalDeltaProgram, ...] = ()
    if parent_score is not None:
        parent = parent_score.program
        certificate_value = compile_relational_delta_failure_certificate(
            task_id=task_id,
            task=task,
            parent=parent,
        )
        certificate = certificate_value.to_json_dict()
        edits = single_slot_relational_delta_variants(
            parent,
            allowed_slots=certificate_value.affected_slots,
            candidate_programs=programs,
            existing_program_ids=initial_ids,
        )
        variants = tuple(
            edit.program
            for edit in sorted(
                edits,
                key=lambda edit: _score_key(
                    score_by_id[relational_delta_program_id(edit.program)]
                ),
            )
        )
    padding_program = parent_score.program if parent_score is not None else programs[0]
    typed = _arm(
        programs=variants,
        padding_program=padding_program,
        task=task,
        source="relational_delta_typed_repair",
        initial_program_ids=initial_ids,
    )
    cold = _arm(
        programs=programs[
            INITIAL_PROGRAM_TRIALS : INITIAL_PROGRAM_TRIALS + ACTION_PROGRAM_TRIALS
        ],
        padding_program=padding_program,
        task=task,
        source="relational_delta_cold_restart",
        initial_program_ids=initial_ids,
    )
    if typed["total_execution_count"] != cold["total_execution_count"]:
        raise ValueError("typed and cold arms do not have equal execution budgets")
    action_declared = parent_score is not None and typed["novel_frontier_count"] > 0
    if typed["frontier_changed"] is not (typed["novel_frontier_count"] > 0):
        raise ValueError("typed frontier accounting is inconsistent")
    parent_payload = None
    if parent_score is not None:
        parent_payload = {
            "program_id": relational_delta_program_id(parent_score.program),
            "program": parent_score.program.to_json_dict(),
            "quality": relational_delta_near_miss_quality(
                parent_score.program, task
            ).to_json_dict(),
        }
    return {
        "task_id": task_id,
        "blind_sha256": task_record["blind_sha256"],
        "source_sha256": task_record["source_sha256"],
        "demo_count": len(task.train),
        "query_count": len(task.test_inputs),
        "grammar_program_count": len(programs),
        "initial_program_count": len(initial_programs),
        "full": {
            "demo_exact_program_count": len(exact_scores),
            "candidates": list(full_deduplicated),
            "selected_candidates": list(full_deduplicated[:2]),
        },
        "initial": {
            "candidates": list(initial_deduplicated),
            "selected_candidates": list(initial_deduplicated[:2]),
        },
        "repair": {
            "quality_parent_count": len(eligible_scores),
            "parent": parent_payload,
            "certificate": certificate,
            "action_declared": action_declared,
            "typed": typed,
            "cold": cold,
        },
    }


def _source_contract() -> dict[str, object]:
    paths = (
        PROJECT_ROOT / "src" / "afts_arc" / "relational_delta.py",
        Path(__file__).resolve(),
        PROJECT_ROOT / "tests" / "test_relational_delta.py",
        PROJECT_ROOT
        / "notes"
        / "design"
        / "relational-delta-program-development-gate-v0.2.md",
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
    semantic_test_report: Path,
) -> dict[str, object]:
    manifest = _load_object(manifest_path)
    if manifest["cohort_id"] != EXPECTED_COHORT_ID:
        raise ValueError("cohort ID does not match the exposed development cohort")
    tasks = manifest["tasks"]
    if not isinstance(tasks, list) or len(tasks) != 50:
        raise ValueError("development cohort must contain exactly 50 tasks")
    records = [_freeze_task(task, blind_root) for task in tasks]
    content = {
        "schema": FREEZE_SCHEMA,
        "status": "outcome_exposed_development_only",
        "cohort_id": manifest["cohort_id"],
        "cohort_manifest_sha256": file_sha256(manifest_path),
        "relational_delta_dsl_version": RELATIONAL_DELTA_DSL_VERSION,
        "source_contract": _source_contract(),
        "semantic_test_receipt": _semantic_test_receipt(semantic_test_report),
        "budgets": {
            "program_cap": PROGRAM_CAP,
            "initial_program_trials": INITIAL_PROGRAM_TRIALS,
            "action_program_trials": ACTION_PROGRAM_TRIALS,
        },
        "near_miss_contract": {
            "strict_aggregate_identity_improvement": True,
            "per_demo_not_worse_than_identity": True,
            "minimum_delta_precision": 0.5,
            "minimum_delta_recall": 0.5,
            "topology_valid": True,
            "novel_frontier_required_to_declare_action": True,
        },
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
        args.semantic_test_report,
    )
    second = _construct_freeze(
        args.cohort_manifest,
        args.blind_root,
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
    pool: Mapping[str, object], gold_outputs: Sequence[Grid]
) -> dict[str, object]:
    candidates = pool["candidates"]
    selected = pool["selected_candidates"]
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
    for candidate in selected[:2]:
        exact = _candidate_query_exact(candidate, gold_outputs)
        pass2_pairs = [
            previous or current
            for previous, current in zip(pass2_pairs, exact, strict=True)
        ]
    return {
        "candidate_count": len(candidates),
        "selected_candidate_count": len(selected[:2]),
        "oracle_pair_coverage": oracle_pairs,
        "pass2_pair_coverage": pass2_pairs,
        "raw_oracle": all(oracle_pairs),
        "pass2": all(pass2_pairs),
    }


def _gold_outputs(path: Path) -> tuple[Grid, ...]:
    payload = _load_object(path)
    if set(payload) != {"train", "test"}:
        raise ValueError(f"ARC task has missing or unknown fields: {path}")
    test = payload["test"]
    if not isinstance(test, list) or not test:
        raise ValueError(f"ARC task has no query outputs: {path}")
    return tuple(as_grid(pair["output"]) for pair in test)


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
    baseline = _load_object(args.baseline_result)
    if baseline["result_id"] != EXPECTED_BASELINE_RESULT_ID:
        raise ValueError("baseline result ID mismatch")
    baseline_by_task = {task["task_id"]: task for task in baseline["tasks"]}

    rows = []
    for task in freeze_payload["tasks"]:
        task_id = task["task_id"]
        task_path = args.task_root / f"{task_id}.json"
        if file_sha256(task_path) != task["source_sha256"]:
            raise ValueError(f"gold task hash mismatch: {task_id}")
        gold_outputs = _gold_outputs(task_path)
        full = _pool_metrics(task["full"], gold_outputs)
        initial = _pool_metrics(task["initial"], gold_outputs)
        typed = _pool_metrics(task["repair"]["typed"], gold_outputs)
        cold = _pool_metrics(task["repair"]["cold"], gold_outputs)
        baseline_task = baseline_by_task[task_id]
        baseline_pass2 = baseline_task["combined_pass2"]
        typed_recovery = typed["raw_oracle"] and not initial["raw_oracle"]
        cold_recovery = cold["raw_oracle"] and not initial["raw_oracle"]
        rows.append(
            {
                "task_id": task_id,
                "diagnostic_task": task_id in DIAGNOSTIC_TASK_IDS,
                "full": full,
                "initial": initial,
                "typed": typed,
                "cold": cold,
                "baseline_pass2": baseline_pass2,
                "unique_pass2_vs_v0_1": full["pass2"] and not baseline_pass2,
                "quality_parent": task["repair"]["parent"] is not None,
                "action_declared": task["repair"]["action_declared"],
                "novel_frontier_count": task["repair"]["typed"][
                    "novel_frontier_count"
                ],
                "typed_recovery": typed_recovery,
                "cold_recovery": cold_recovery,
                "typed_unique_vs_cold": typed_recovery and not cold_recovery,
            }
        )
    diagnostic = [row for row in rows if row["diagnostic_task"]]
    counts = {
        "full_raw_oracle": sum(row["full"]["raw_oracle"] for row in rows),
        "full_pass2": sum(row["full"]["pass2"] for row in rows),
        "unique_pass2_vs_v0_1": sum(row["unique_pass2_vs_v0_1"] for row in rows),
        "diagnostic_full_raw_oracle": sum(
            row["full"]["raw_oracle"] for row in diagnostic
        ),
        "diagnostic_full_pass2": sum(row["full"]["pass2"] for row in diagnostic),
        "quality_parent": sum(row["quality_parent"] for row in rows),
        "action_declared": sum(row["action_declared"] for row in rows),
        "typed_recovery": sum(row["typed_recovery"] for row in rows),
        "cold_recovery": sum(row["cold_recovery"] for row in rows),
        "typed_unique_vs_cold": sum(row["typed_unique_vs_cold"] for row in rows),
    }
    gates = {
        "semantic_fixture_gate": freeze_payload["semantic_test_receipt"]["passed"],
        "diagnostic_representation_gate": counts["diagnostic_full_raw_oracle"] >= 2,
        "exposed_unique_coverage_descriptive": counts["unique_pass2_vs_v0_1"] >= 2,
        "natural_quality_parent_gate": counts["quality_parent"] >= 2,
        "natural_frontier_gate": counts["action_declared"] >= 2,
        "typed_repair_gate": (
            counts["typed_recovery"] >= 1
            and counts["typed_recovery"] > counts["cold_recovery"]
        ),
        "controller_training_authorized": False,
        "visual_lodo_authorized": False,
    }
    content = {
        "schema": RESULT_SCHEMA,
        "status": "outcome_exposed_development_only",
        "candidate_freeze_id": freeze_payload["freeze_id"],
        "candidate_freeze_sha256": file_sha256(args.candidate_freeze),
        "baseline_result_id": baseline["result_id"],
        "cohort_id": freeze_payload["cohort_id"],
        "task_count": len(rows),
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
    freeze_parser.add_argument("--semantic-test-report", type=Path, required=True)
    freeze_parser.add_argument("--output", type=Path, required=True)
    freeze_parser.set_defaults(function=freeze)
    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--candidate-freeze", type=Path, required=True)
    score_parser.add_argument("--baseline-result", type=Path, required=True)
    score_parser.add_argument("--task-root", type=Path, required=True)
    score_parser.add_argument("--output", type=Path, required=True)
    score_parser.set_defaults(function=score)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
