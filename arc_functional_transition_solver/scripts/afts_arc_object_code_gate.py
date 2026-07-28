"""Audit the first object/code provider and decide whether neural work may resume.

The experiment is intentionally phase-gated.  It first runs generator-known
positive controls, then freezes and replays candidate artifacts on the exact task
IDs from an earlier baseline summary.  Public query outputs are consulted only
after each new pool artifact has been written.  A negative result is classified
as implementation, search-budget, representation, inductive-ambiguity, or typed-
repair failure instead of being reported as an undifferentiated score decrease.

No training, model download, API call, or controller fitting is performed.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path


def _bootstrap() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    repository_root = project_root.parent
    for source_root in (repository_root / "src", project_root / "src"):
        resolved = str(source_root.resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)
    return repository_root


REPOSITORY_ROOT = _bootstrap()

from afts_arc.blind import BlindTask  # noqa: E402
from afts_arc.experiment_safety import (  # noqa: E402
    ARC_EXPERIMENT_SOURCE_PATHS,
    ExperimentSafetyError,
    atomic_write_json,
    canonical_sha256,
    capture_git_source_provenance,
    file_sha256,
    runtime_metadata,
    verify_source_provenance_unchanged,
)
from afts_arc.grid import Grid, as_grid, grid_key  # noqa: E402
from afts_arc.hybrid import (  # noqa: E402
    D4LabelCompletionProgram,
    RoleStampProgram,
    enumerate_object_code_programs,
    evaluate_hypothesis,
    execute_object_code_program,
    make_object_code_hypothesis,
    rank_verified,
    synthesize_object_code_programs,
    typed_repair_frontier,
)
from afts_arc.hybrid.object_code import (  # noqa: E402
    OBJECT_CODE_DSL_VERSION,
    OBJECT_CODE_PROVIDER_VERSION,
    ObjectCodeProgramScore,
    ObjectCodeSynthesisResult,
)
from afts_arc.hybrid.types import CandidateEvaluation  # noqa: E402
from afts_arc.task import ARCPair, ARCTask, load_task  # noqa: E402


SCHEMA_VERSION = "afts.object-code-gate/v1"
POOL_SCHEMA_VERSION = "afts.object-code-frozen-pool/v1"


def _load_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExperimentSafetyError(f"cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ExperimentSafetyError(f"JSON artifact is not an object: {path}")
    return payload


def _blind(
    train: tuple[tuple[list[list[int]], list[list[int]]], ...],
    queries: tuple[list[list[int]], ...],
) -> BlindTask:
    return BlindTask.from_observations(
        train=tuple(
            ARCPair(as_grid(source), as_grid(target)) for source, target in train
        ),
        test_inputs=tuple(as_grid(grid) for grid in queries),
    )


def _role_control() -> tuple[BlindTask, tuple[Grid, ...]]:
    source = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 8, 0, 0, 0, 0],
        [0, 8, 3, 8, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 2, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ]
    target = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 8, 0, 0],
        [0, 0, 0, 8, 0, 8, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ]
    query = [
        [0, 0, 0, 0, 8, 0, 0],
        [0, 0, 0, 8, 3, 8, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ]
    expected = as_grid(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 8, 0, 0, 0, 0],
            [0, 8, 0, 8, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
    )
    return _blind(((source, target),), (query,)), (expected,)


def _d4_control() -> BlindTask:
    source = [[0 for _ in range(12)] for _ in range(7)]
    for row, column in (
        (1, 1),
        (2, 1),
        (3, 1),
        (3, 2),
        (1, 7),
        (2, 7),
        (3, 7),
        (3, 8),
    ):
        source[row][column] = 4
    source[1][2] = 1
    source[4][1] = 3
    source[1][8] = 1
    target = [row[:] for row in source]
    target[4][7] = 3
    return _blind(((source, target),), (source,))


def _score_query_outputs(
    scores: Sequence[ObjectCodeProgramScore],
    query_inputs: Sequence[Grid],
    oracle_outputs: Sequence[Grid],
) -> bool:
    for score in scores:
        outputs = tuple(
            execute_object_code_program(score.program, grid).output
            for grid in query_inputs
        )
        if outputs == tuple(oracle_outputs):
            return True
    return False


def _positive_controls(repair_trial_budget: int) -> dict[str, object]:
    role_task, role_oracle = _role_control()
    role_result = synthesize_object_code_programs(role_task)
    role_hit = _score_query_outputs(
        role_result.exact_scores, role_task.test_inputs, role_oracle
    )

    d4_task = _d4_control()
    d4_program = D4LabelCompletionProgram(0, 4, 4, 1)
    d4_execution = execute_object_code_program(d4_program, d4_task.train[0].input)
    d4_pass = d4_execution.ok and d4_execution.output == d4_task.train[0].output

    fault_cases = (
        (
            "canvas",
            RoleStampProgram(0, 8, 3, 2, "identity", "crop"),
            (),
            "canvas_reinfer",
        ),
        (
            "ast_hole",
            RoleStampProgram(0, 8, 3, 1, "identity", "blank"),
            ("target_anchor_color",),
            "fill_ast_hole",
        ),
        (
            "background",
            RoleStampProgram(9, 8, 3, 2, "identity", "blank"),
            (),
            "reparse_background",
        ),
        (
            "correspondence",
            RoleStampProgram(0, 8, 3, 2, "rotate90", "blank"),
            (),
            "object_rematch",
        ),
    )
    repairs = []
    full_frontier = enumerate_object_code_programs(role_task)
    for name, program, holes, expected_action in fault_cases:
        parent = make_object_code_hypothesis(
            program,
            demo_exact=False,
            ast_holes=holes,
        )
        evaluation = evaluate_hypothesis(parent, role_task)
        certificate, typed_frontier = typed_repair_frontier(role_task, evaluation)
        trials = min(repair_trial_budget, len(typed_frontier))
        typed = synthesize_object_code_programs(
            role_task,
            max_program_trials=max(1, trials),
            programs=typed_frontier[:trials],
        )
        cold = synthesize_object_code_programs(
            role_task,
            max_program_trials=max(1, trials),
            programs=full_frontier[:trials],
        )
        typed_oracle = _score_query_outputs(
            typed.exact_scores, role_task.test_inputs, role_oracle
        )
        cold_oracle = _score_query_outputs(
            cold.exact_scores, role_task.test_inputs, role_oracle
        )
        repairs.append(
            {
                "fault": name,
                "expected_action": expected_action,
                "observed_action": certificate.recommended_action,
                "certificate_id": certificate.certificate_id,
                "typed_frontier_size": len(typed_frontier),
                "matched_program_trials": trials,
                "typed_demo_recovered": bool(typed.exact_scores),
                "cold_demo_recovered": bool(cold.exact_scores),
                "typed_oracle_recovered": typed_oracle,
                "cold_oracle_recovered": cold_oracle,
            }
        )
    action_causality = all(
        item["expected_action"] == item["observed_action"] for item in repairs
    ) and len({item["observed_action"] for item in repairs}) == len(fault_cases)
    repair_pass = all(
        item["typed_oracle_recovered"] and not item["cold_oracle_recovered"]
        for item in repairs
    )
    return {
        "role_stamp_oracle_recovered": role_hit,
        "d4_completion_demo_exact": d4_pass,
        "typed_action_causality": action_causality,
        "typed_repair_beats_equal_trial_cold_restart": repair_pass,
        "fault_cases": repairs,
        "passed": role_hit and d4_pass and action_causality and repair_pass,
    }


def _candidate_artifacts(
    result: ObjectCodeSynthesisResult,
    blind: BlindTask,
) -> tuple[tuple[CandidateEvaluation, ...], list[dict[str, object]]]:
    candidates = tuple(
        make_object_code_hypothesis(
            score.program,
            demo_exact=score.all_demo_exact,
        )
        for score in (*result.exact_scores, *result.near_miss_scores)
    )
    evaluations = tuple(
        evaluate_hypothesis(candidate, blind) for candidate in candidates
    )
    records = []
    for evaluation in evaluations:
        records.append(
            {
                "candidate": evaluation.hypothesis.to_json_dict(),
                "demo_exact": evaluation.demo_exact,
                "hard_verified": evaluation.hard_verified,
                "eligible": evaluation.eligible,
                "rejection_reason": evaluation.rejection_reason,
                "agreement": evaluation.agreement,
                "total_mdl_bits": evaluation.total_mdl_bits,
                "demo_output_keys": [
                    None if output is None else grid_key(output)
                    for output in evaluation.demo_outputs
                ],
                "query_output_keys": [
                    None if output is None else grid_key(output)
                    for output in evaluation.query_outputs
                ],
            }
        )
    return evaluations, records


def _evaluation_signature(evaluations: Sequence[CandidateEvaluation]) -> str:
    return canonical_sha256(
        [
            {
                "candidate": item.hypothesis.to_json_dict(),
                "demo_outputs": [
                    None if output is None else grid_key(output)
                    for output in item.demo_outputs
                ],
                "query_outputs": [
                    None if output is None else grid_key(output)
                    for output in item.query_outputs
                ],
                "demo_exact": item.demo_exact,
                "hard_verified": item.hard_verified,
                "rejection_reason": item.rejection_reason,
            }
            for item in evaluations
        ]
    )


def _oracle_hit(evaluation: CandidateEvaluation, task: ARCTask) -> bool:
    oracle = tuple(pair.output for pair in task.test)
    return (
        all(output is not None for output in oracle)
        and evaluation.query_outputs == oracle
    )


def _baseline_flags(summary: Mapping[str, object]) -> dict[str, dict[str, bool]]:
    tasks = summary.get("tasks")
    if not isinstance(tasks, list):
        raise ExperimentSafetyError("baseline summary has no task records")
    flags: dict[str, dict[str, bool]] = {}
    for record in tasks:
        if not isinstance(record, Mapping) or not isinstance(
            record.get("task_id"), str
        ):
            raise ExperimentSafetyError("baseline task record is malformed")
        policies = record.get("policies")
        if not isinstance(policies, Mapping):
            raise ExperimentSafetyError("baseline task record has no policies")
        raw = False
        selectable = False
        passed = False
        for policy in policies.values():
            if not isinstance(policy, Mapping):
                continue
            metrics = policy.get("metrics")
            if not isinstance(metrics, Mapping):
                continue
            raw |= metrics.get("pool_oracle_covered") is True
            selectable |= metrics.get("pool_selectable_oracle_covered") is True
            passed |= metrics.get("pass_at_k") is True
        flags[record["task_id"]] = {
            "raw_oracle_covered": raw,
            "selectable_oracle_covered": selectable,
            "pass_at_2": passed,
        }
    return flags


def _repair_result_manifest(
    result: ObjectCodeSynthesisResult,
    blind: BlindTask,
) -> dict[str, object]:
    scores = (*result.exact_scores, *result.near_miss_scores)
    return {
        "program_trial_count": result.program_trial_count,
        "demo_execution_count": result.demo_execution_count,
        "query_execution_count_during_synthesis": result.query_execution_count,
        "artifact_query_replay_count": len(scores) * len(blind.test_inputs),
        "programs": [
            {
                "program": score.program.to_json_dict(),
                "demo_exact": score.all_demo_exact,
                "agreement": score.agreement,
                "query_output_keys": [
                    (
                        None
                        if (
                            execution := execute_object_code_program(
                                score.program, grid
                            )
                        ).output
                        is None
                        else grid_key(execution.output)
                    )
                    for grid in blind.test_inputs
                ],
            }
            for score in scores
        ],
    }


def _natural_repairs(
    task: ARCTask,
    blind: BlindTask,
    result: ObjectCodeSynthesisResult,
    all_programs: Sequence[object],
    *,
    repair_trial_budget: int,
    case_limit: int,
    output_dir: Path,
) -> list[dict[str, object]]:
    initial_pool_program_ids = {
        canonical_sha256(score.program.to_json_dict())
        for score in (*result.exact_scores, *result.near_miss_scores)
    }
    pending: list[
        tuple[
            dict[str, object],
            ObjectCodeSynthesisResult | None,
            ObjectCodeSynthesisResult | None,
        ]
    ] = []
    for case_index, parent_score in enumerate(result.near_miss_scores[:case_limit]):
        parent = make_object_code_hypothesis(parent_score.program, demo_exact=False)
        evaluation = evaluate_hypothesis(parent, blind)
        certificate, frontier = typed_repair_frontier(blind, evaluation)
        trials = min(repair_trial_budget, len(frontier))
        base: dict[str, object] = {
            "case_index": case_index,
            "parent_hypothesis_id": parent.hypothesis_id,
            "parent_program": parent_score.program.to_json_dict(),
            "parent_agreement": parent_score.agreement,
            "certificate": certificate.to_json_dict(),
            "matched_program_trials": trials,
            "typed_frontier_size": len(frontier),
            "initial_pool_program_count": len(initial_pool_program_ids),
        }
        if trials < 1:
            base["status"] = "empty_typed_frontier"
            pending.append((base, None, None))
            continue
        typed = synthesize_object_code_programs(
            blind,
            max_program_trials=trials,
            programs=frontier[:trials],
        )
        cold = synthesize_object_code_programs(
            blind,
            max_program_trials=trials,
            programs=tuple(all_programs[:trials]),
        )
        if not (
            typed.program_trial_count == cold.program_trial_count == trials
            and typed.demo_execution_count == cold.demo_execution_count
        ):
            raise ExperimentSafetyError(
                "typed repair and cold restart violated matched generation cost"
            )
        base.update(
            {
                "status": "generated_and_frozen",
                "native_cost_reservation": {
                    "program_trials": trials,
                    "demo_program_executions": trials * len(blind.train),
                    "query_program_executions": trials * len(blind.test_inputs),
                },
                "strict_generation_cost_comparable": True,
                "typed": _repair_result_manifest(typed, blind),
                "cold_restart": _repair_result_manifest(cold, blind),
            }
        )
        pending.append((base, typed, cold))

    if not pending:
        return []
    artifact_content = {
        "schema": "afts.object-code-repair-pool/v1",
        "task_id": task.task_id,
        "task_source_sha256": task.source_sha256,
        "blind_task_id": blind.task_id,
        "blind_content_sha256": blind.blind_content_sha256,
        "provider_version": OBJECT_CODE_PROVIDER_VERSION,
        "repair_trial_budget": repair_trial_budget,
        "cases": [base for base, _, _ in pending],
    }
    artifact_id = canonical_sha256(artifact_content)
    relative = Path("repairs") / f"{task.task_id}.{artifact_id[:16]}.json"
    atomic_write_json(
        output_dir / relative,
        {"repair_pool_id": artifact_id, **artifact_content},
    )

    # Query labels become visible only after every repair/cold candidate in this
    # task has been frozen in the content-addressed artifact above.
    oracle = tuple(pair.output for pair in task.test)
    if any(output is None for output in oracle):
        raise ExperimentSafetyError("public gate task omits a query output")
    records = []
    for base, typed, cold in pending:
        record = {
            **{
                key: value
                for key, value in base.items()
                if key not in {"typed", "cold_restart"}
            },
            "repair_pool_id": artifact_id,
            "repair_pool_manifest": relative.as_posix(),
        }
        if typed is not None and cold is not None:
            typed_raw_oracle = _score_query_outputs(
                typed.exact_scores, blind.test_inputs, oracle
            )
            cold_raw_oracle = _score_query_outputs(
                cold.exact_scores, blind.test_inputs, oracle
            )
            typed_novel_scores = tuple(
                score
                for score in typed.exact_scores
                if canonical_sha256(score.program.to_json_dict())
                not in initial_pool_program_ids
            )
            cold_novel_scores = tuple(
                score
                for score in cold.exact_scores
                if canonical_sha256(score.program.to_json_dict())
                not in initial_pool_program_ids
            )
            typed_novel_oracle = _score_query_outputs(
                typed_novel_scores, blind.test_inputs, oracle
            )
            cold_novel_oracle = _score_query_outputs(
                cold_novel_scores, blind.test_inputs, oracle
            )
            record.update(
                {
                    "typed_demo_recovered": bool(typed.exact_scores),
                    "cold_demo_recovered": bool(cold.exact_scores),
                    "typed_raw_oracle_recovered": typed_raw_oracle,
                    "cold_raw_oracle_recovered": cold_raw_oracle,
                    "typed_novel_demo_recovered": bool(typed_novel_scores),
                    "cold_novel_demo_recovered": bool(cold_novel_scores),
                    "typed_novel_oracle_recovered": typed_novel_oracle,
                    "cold_novel_oracle_recovered": cold_novel_oracle,
                    "unique_typed_novel_oracle_recovery": (
                        typed_novel_oracle and not cold_novel_oracle
                    ),
                    "status": "completed",
                }
            )
        records.append(record)
    return records


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("baseline_summary", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--split", default="training")
    parser.add_argument("--max-program-trials", type=int, default=20_000)
    parser.add_argument("--max-exact-programs", type=int, default=32)
    parser.add_argument("--max-near-misses", type=int, default=8)
    parser.add_argument("--minimum-near-miss-agreement", type=float, default=0.2)
    parser.add_argument("--repair-trial-budget", type=int, default=16)
    parser.add_argument("--natural-near-miss-limit", type=int, default=100)
    parser.add_argument("--selectable-union-gate", type=int, default=25)
    parser.add_argument("--provider-unique-gate", type=int, default=3)
    parser.add_argument("--repair-unique-rate-gate", type=float, default=0.05)
    parser.add_argument("--allow-dirty-source", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.max_program_trials < 1 or args.max_exact_programs < 1:
        raise ExperimentSafetyError("program limits must be positive")
    if args.max_near_misses < 1:
        raise ExperimentSafetyError(
            "natural near-miss audit requires max_near_misses >= 1"
        )
    if args.repair_trial_budget < 1:
        raise ExperimentSafetyError("repair trial budget must be positive")
    if args.natural_near_miss_limit < 1:
        raise ExperimentSafetyError("natural near-miss limit must be positive")
    if not 0.0 <= args.minimum_near_miss_agreement <= 1.0:
        raise ExperimentSafetyError("near-miss agreement must be in [0, 1]")
    if not 0.0 <= args.repair_unique_rate_gate <= 1.0:
        raise ExperimentSafetyError("repair rate gate must be in [0, 1]")

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise ExperimentSafetyError(
            "output directory already exists; choose a new run ID"
        )
    baseline_path = args.baseline_summary.resolve()
    baseline = _load_json(baseline_path)
    baseline_dataset = baseline.get("dataset")
    if not isinstance(baseline_dataset, Mapping):
        raise ExperimentSafetyError("baseline summary has no dataset contract")
    task_ids = baseline_dataset.get("task_ids")
    if (
        not isinstance(task_ids, list)
        or not task_ids
        or any(not isinstance(task_id, str) or not task_id for task_id in task_ids)
    ):
        raise ExperimentSafetyError("baseline dataset task IDs are malformed")
    if len(set(task_ids)) != len(task_ids):
        raise ExperimentSafetyError("baseline dataset task IDs are not unique")
    if baseline_dataset.get("split") != args.split:
        raise ExperimentSafetyError("requested split differs from baseline summary")
    baseline_by_task = _baseline_flags(baseline)
    if set(task_ids) != set(baseline_by_task):
        raise ExperimentSafetyError("baseline task IDs and task records differ")

    source_start = capture_git_source_provenance(
        REPOSITORY_ROOT, paths=ARC_EXPERIMENT_SOURCE_PATHS
    )
    if source_start.dirty and not args.allow_dirty_source:
        raise ExperimentSafetyError(
            "refusing canonical gate from dirty source; commit first or pass "
            "--allow-dirty-source for a diagnostic"
        )
    controls = _positive_controls(args.repair_trial_budget)
    task_records = []
    failures = []
    natural_repairs = []
    dataset_root = args.dataset_root.resolve()
    split_root = dataset_root / args.split

    for task_id in task_ids:
        task_path = split_root / f"{task_id}.json"
        try:
            task = load_task(task_path)
            blind = BlindTask.from_task(task)
            all_programs = enumerate_object_code_programs(blind)
            result = synthesize_object_code_programs(
                blind,
                max_program_trials=args.max_program_trials,
                max_exact_programs=args.max_exact_programs,
                max_near_misses=args.max_near_misses,
                minimum_near_miss_agreement=args.minimum_near_miss_agreement,
            )
            evaluations, candidate_records = _candidate_artifacts(result, blind)
            replay_result = synthesize_object_code_programs(
                blind,
                max_program_trials=args.max_program_trials,
                max_exact_programs=args.max_exact_programs,
                max_near_misses=args.max_near_misses,
                minimum_near_miss_agreement=args.minimum_near_miss_agreement,
            )
            replay_evaluations, _ = _candidate_artifacts(replay_result, blind)
            signature = _evaluation_signature(evaluations)
            replay_signature = _evaluation_signature(replay_evaluations)
            replay_exact = signature == replay_signature and result == replay_result
            grid_count = len(blind.train) + len(blind.test_inputs)
            verification_executions = (
                len(evaluations) * grid_count
                + sum(
                    item.rejection_reason != "execution_failed" for item in evaluations
                )
                * 2
                * grid_count
            )
            pool_content = {
                "schema": POOL_SCHEMA_VERSION,
                "task_id": task_id,
                "task_source_sha256": task.source_sha256,
                "blind_task_id": blind.task_id,
                "blind_content_sha256": blind.blind_content_sha256,
                "provider_version": OBJECT_CODE_PROVIDER_VERSION,
                "dsl_version": OBJECT_CODE_DSL_VERSION,
                "config": {
                    "max_program_trials": args.max_program_trials,
                    "max_exact_programs": args.max_exact_programs,
                    "max_near_misses": args.max_near_misses,
                    "minimum_near_miss_agreement": args.minimum_near_miss_agreement,
                },
                "grammar_program_count": len(all_programs),
                "program_trial_count": result.program_trial_count,
                "demo_execution_count": result.demo_execution_count,
                "query_execution_count": result.query_execution_count,
                "candidate_verification_execution_count": verification_executions,
                "replay_validation": {
                    "program_trial_count": replay_result.program_trial_count,
                    "demo_execution_count": replay_result.demo_execution_count,
                    "query_execution_count": replay_result.query_execution_count,
                    "candidate_verification_execution_count": (
                        len(replay_evaluations) * grid_count
                        + sum(
                            item.rejection_reason != "execution_failed"
                            for item in replay_evaluations
                        )
                        * 2
                        * grid_count
                    ),
                },
                "candidate_count": len(evaluations),
                "candidate_evaluation_signature": signature,
                "replay_evaluation_signature": replay_signature,
                "replay_exact": replay_exact,
                "candidates": candidate_records,
            }
            pool_id = canonical_sha256(pool_content)
            pool_artifact = {"pool_id": pool_id, **pool_content}
            pool_relative = Path("pools") / f"{task_id}.{pool_id[:16]}.json"
            # Freeze before reading any public query output below.
            atomic_write_json(output_dir / pool_relative, pool_artifact)

            remaining_near_misses = max(
                0, args.natural_near_miss_limit - len(natural_repairs)
            )
            natural_cases = _natural_repairs(
                task,
                blind,
                result,
                all_programs,
                repair_trial_budget=args.repair_trial_budget,
                case_limit=remaining_near_misses,
                output_dir=output_dir,
            )
            natural_repairs.extend(
                {"task_id": task_id, **case} for case in natural_cases
            )

            raw_hit = any(_oracle_hit(item, task) for item in evaluations)
            selectable_hit = any(
                item.eligible and _oracle_hit(item, task) for item in evaluations
            )
            selected = rank_verified(evaluations, limit=2)
            pass_at_2 = any(_oracle_hit(item, task) for item in selected)
            baseline_flags = baseline_by_task[task_id]
            if not replay_exact:
                failure_class = "implementation_replay_failure"
            elif selectable_hit:
                failure_class = "solved"
            elif raw_hit:
                failure_class = "selectability_or_verifier_failure"
            elif result.exact_scores:
                failure_class = "inductive_ambiguity_or_query_generalization"
            elif result.program_trial_count < len(all_programs):
                failure_class = "search_budget_failure"
            else:
                failure_class = "candidate_language_failure"

            task_records.append(
                {
                    "task_id": task_id,
                    "task_source_sha256": task.source_sha256,
                    "blind_content_sha256": blind.blind_content_sha256,
                    "pool_id": pool_id,
                    "pool_manifest": pool_relative.as_posix(),
                    "grammar_program_count": len(all_programs),
                    "program_trial_count": result.program_trial_count,
                    "grammar_exhaustive_under_limit": (
                        result.program_trial_count == len(all_programs)
                    ),
                    "exact_demo_program_count": len(result.exact_scores),
                    "near_miss_program_count": len(result.near_miss_scores),
                    "candidate_count": len(evaluations),
                    "replay_exact": replay_exact,
                    "provider_raw_oracle_covered": raw_hit,
                    "provider_selectable_oracle_covered": selectable_hit,
                    "provider_pass_at_2": pass_at_2,
                    "baseline": baseline_flags,
                    "new_unique_selectable_coverage": (
                        selectable_hit
                        and not baseline_flags["selectable_oracle_covered"]
                    ),
                    "failure_class": failure_class,
                    "natural_repair_case_count": len(natural_cases),
                    "natural_repair_pool_ids": sorted(
                        {
                            case["repair_pool_id"]
                            for case in natural_cases
                            if "repair_pool_id" in case
                        }
                    ),
                }
            )
        except Exception as exc:
            failures.append(
                {
                    "task_id": task_id,
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                }
            )

    source_end = capture_git_source_provenance(
        REPOSITORY_ROOT, paths=ARC_EXPERIMENT_SOURCE_PATHS
    )
    verify_source_provenance_unchanged(source_start, source_end)
    completed = len(task_records)
    baseline_raw = sum(
        record["baseline"]["raw_oracle_covered"] for record in task_records
    )
    baseline_selectable = sum(
        record["baseline"]["selectable_oracle_covered"] for record in task_records
    )
    provider_raw = sum(record["provider_raw_oracle_covered"] for record in task_records)
    provider_selectable = sum(
        record["provider_selectable_oracle_covered"] for record in task_records
    )
    provider_pass = sum(record["provider_pass_at_2"] for record in task_records)
    unique_selectable = sum(
        record["new_unique_selectable_coverage"] for record in task_records
    )
    union_selectable = sum(
        record["baseline"]["selectable_oracle_covered"]
        or record["provider_selectable_oracle_covered"]
        for record in task_records
    )
    repair_completed = [
        item for item in natural_repairs if item.get("status") == "completed"
    ]
    typed_raw_repairs = sum(
        item["typed_raw_oracle_recovered"] for item in repair_completed
    )
    cold_raw_repairs = sum(
        item["cold_raw_oracle_recovered"] for item in repair_completed
    )
    typed_repairs = sum(
        item["typed_novel_oracle_recovered"] for item in repair_completed
    )
    cold_repairs = sum(item["cold_novel_oracle_recovered"] for item in repair_completed)
    unique_repairs = sum(
        item["unique_typed_novel_oracle_recovery"] for item in repair_completed
    )
    repair_unique_rate = (
        unique_repairs / len(natural_repairs) if natural_repairs else 0.0
    )
    failure_counts = Counter(record["failure_class"] for record in task_records)
    action_counts = Counter(
        item["certificate"]["recommended_action"] for item in repair_completed
    )
    gates = {
        "implementation_controls": controls["passed"] and not failures,
        "all_candidate_replays_exact": all(
            record["replay_exact"] for record in task_records
        ),
        "selectable_union": {
            "observed": union_selectable,
            "required": args.selectable_union_gate,
            "passed": union_selectable >= args.selectable_union_gate,
        },
        "provider_unique_selectable": {
            "observed": unique_selectable,
            "required": args.provider_unique_gate,
            "passed": unique_selectable >= args.provider_unique_gate,
        },
        "natural_near_miss_sample": {
            "observed": len(natural_repairs),
            "required_for_100_case_claim": 100,
            "passed": len(natural_repairs) >= 100,
        },
        "repair_unique_rate": {
            "observed": repair_unique_rate,
            "required": args.repair_unique_rate_gate,
            "passed": repair_unique_rate >= args.repair_unique_rate_gate,
        },
        "repair_beats_equal_trial_cold_restart": {
            "typed_novel_recoveries": typed_repairs,
            "cold_novel_recoveries": cold_repairs,
            "unique_typed_novel_recoveries": unique_repairs,
            "passed": unique_repairs > 0 and typed_repairs > cold_repairs,
        },
        "controlled_residual_causality": controls["typed_action_causality"],
    }
    proceed_to_neural_provider = all(
        (
            gates["implementation_controls"],
            gates["all_candidate_replays_exact"],
            gates["selectable_union"]["passed"],
            gates["provider_unique_selectable"]["passed"],
            gates["natural_near_miss_sample"]["passed"],
            gates["repair_unique_rate"]["passed"],
            gates["repair_beats_equal_trial_cold_restart"]["passed"],
            gates["controlled_residual_causality"],
        )
    )
    if not gates["implementation_controls"] or not gates["all_candidate_replays_exact"]:
        negative_result_attribution = "implementation_or_protocol_failure"
    elif (
        not gates["selectable_union"]["passed"]
        or not gates["provider_unique_selectable"]["passed"]
    ):
        negative_result_attribution = "current_object_code_representation_insufficient"
    elif not gates["natural_near_miss_sample"]["passed"]:
        negative_result_attribution = "insufficient_natural_near_miss_evidence"
    elif not gates["repair_beats_equal_trial_cold_restart"]["passed"]:
        negative_result_attribution = "typed_repair_hypothesis_not_supported"
    else:
        negative_result_attribution = "none"

    publication_blockers = []
    if source_start.dirty:
        publication_blockers.append("dirty_source_override")
    if failures:
        publication_blockers.append("incomplete_task_execution")
    if not controls["passed"]:
        publication_blockers.append("positive_control_failure")
    if not all(record["replay_exact"] for record in task_records):
        publication_blockers.append("candidate_replay_mismatch")
    summary = {
        "schema": SCHEMA_VERSION,
        "result_id": "",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": source_start.head,
        "source_provenance": source_start.to_json_dict(),
        "runtime": runtime_metadata(),
        "claim_scope": {
            "phase": "object_code_candidate_and_repair_gate",
            "development_only": True,
            "training_performed": False,
            "query_oracle_used_by_generation": False,
            "query_oracle_used_after_pool_freeze_for_scoring": True,
            "controller_claim_allowed": False,
            "neural_provider_claim_allowed": proceed_to_neural_provider,
            "cost_comparison_scope": (
                "equal program/demo/query reservation for typed repair versus "
                "cold restart only; no cross-provider scalar cost claim"
            ),
        },
        "dataset": {
            "root": str(dataset_root),
            "split": args.split,
            "selection": "exact_baseline_task_ids",
            "task_ids": task_ids,
            "requested_task_count": len(task_ids),
            "completed_task_count": completed,
            "failed_task_count": len(failures),
        },
        "baseline": {
            "summary_path": str(baseline_path),
            "summary_sha256": file_sha256(baseline_path),
            "source_commit": baseline.get("source_commit"),
            "raw_oracle_covered_tasks": baseline_raw,
            "selectable_oracle_covered_tasks": baseline_selectable,
        },
        "provider_config": {
            "provider_version": OBJECT_CODE_PROVIDER_VERSION,
            "dsl_version": OBJECT_CODE_DSL_VERSION,
            "max_program_trials": args.max_program_trials,
            "max_exact_programs": args.max_exact_programs,
            "max_near_misses": args.max_near_misses,
            "minimum_near_miss_agreement": args.minimum_near_miss_agreement,
            "repair_trial_budget": args.repair_trial_budget,
            "natural_near_miss_limit": args.natural_near_miss_limit,
        },
        "positive_controls": controls,
        "coverage": {
            "provider_raw_oracle_covered_tasks": provider_raw,
            "provider_selectable_oracle_covered_tasks": provider_selectable,
            "provider_pass_at_2_tasks": provider_pass,
            "new_unique_selectable_tasks": unique_selectable,
            "baseline_plus_provider_selectable_union_tasks": union_selectable,
        },
        "natural_near_miss_repair": {
            "constructed_case_count": len(natural_repairs),
            "completed_equal_trial_case_count": len(repair_completed),
            "repair_rate_denominator_case_count": len(natural_repairs),
            "typed_raw_oracle_recoveries": typed_raw_repairs,
            "cold_restart_raw_oracle_recoveries": cold_raw_repairs,
            "typed_novel_oracle_recoveries": typed_repairs,
            "cold_restart_novel_oracle_recoveries": cold_repairs,
            "unique_typed_novel_oracle_recoveries": unique_repairs,
            "unique_typed_novel_recovery_rate": repair_unique_rate,
            "recommended_action_counts": dict(sorted(action_counts.items())),
            "cases": natural_repairs,
        },
        "failure_analysis": {
            "task_failure_class_counts": dict(sorted(failure_counts.items())),
            "negative_result_attribution": negative_result_attribution,
            "interpretation": (
                "A coverage failure after passing controls and exhaustive grammar "
                "replay is a limitation of this representation slice, not evidence "
                "that residuals are inherently uninformative."
            ),
        },
        "gates": gates,
        "decision": {
            "proceed_to_masked_neural_provider": proceed_to_neural_provider,
            "controller_remains_frozen": not proceed_to_neural_provider,
        },
        "publication_blockers": publication_blockers,
        "failures": failures,
        "tasks": task_records,
    }
    summary["result_id"] = canonical_sha256(
        {key: value for key, value in summary.items() if key != "result_id"}
    )
    atomic_write_json(output_dir / "summary.json", summary)
    print(
        json.dumps(
            {
                "result_id": summary["result_id"],
                "completed": completed,
                "failed": len(failures),
                "coverage": summary["coverage"],
                "negative_result_attribution": negative_result_attribution,
                "proceed_to_masked_neural_provider": proceed_to_neural_provider,
                "output": str(output_dir / "summary.json"),
            },
            sort_keys=True,
        )
    )
    return 0 if not failures and controls["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
