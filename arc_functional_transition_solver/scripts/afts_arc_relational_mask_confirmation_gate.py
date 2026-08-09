"""Run the frozen ARC-2 relational-mask representation and repair gate."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT_STRING = str(PROJECT_ROOT)
SOURCE_ROOT = str((PROJECT_ROOT / "src").resolve())
if PROJECT_ROOT_STRING not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STRING)
if SOURCE_ROOT not in sys.path:
    sys.path.insert(0, SOURCE_ROOT)

from afts_arc.blind import BlindTask  # noqa: E402
from afts_arc.experiment_safety import (  # noqa: E402
    atomic_write_json,
    canonical_sha256,
)
from afts_arc.grid import Grid, as_grid, grid_to_lists  # noqa: E402
from afts_arc.hybrid.object_code import (  # noqa: E402
    ObjectCodeProgram,
    enumerate_object_code_programs,
    execute_object_code_program,
    object_code_program_id,
    synthesize_object_code_programs,
)
from afts_arc.hybrid.types import canonical_json  # noqa: E402
from afts_arc.relational_mask import (  # noqa: E402
    RelationalMaskProgram,
    RelationalMaskProgramScore,
    compile_relational_mask_failure_certificate,
    enumerate_relational_mask_programs,
    execute_relational_mask,
    relational_mask_program_id,
    score_relational_mask_program,
    single_slot_relational_mask_variants,
)
from afts_arc.task import ARCPair  # noqa: E402
from afts_arc.visual_provider_gate import _file_sha256  # noqa: E402
from scripts.build_arc2_relational_mask_confirmation import (  # noqa: E402
    COHORT_SCHEMA,
    EXPECTED_ARC2_COMMIT,
)


FREEZE_SCHEMA = "afts.arc2-relational-mask-candidate-freeze/v1"
RESULT_SCHEMA = "afts.arc2-relational-mask-confirmation-result/v1"
EXPECTED_COHORT_ID = "dfee67cf54613ba721e1fe46723967c95020715dff1a40500fe498055805e3ac"
LEGACY_PROGRAM_TRIALS = 512
RELATIONAL_PROGRAM_CAP = 1_280
INITIAL_RELATIONAL_TRIALS = 64
ACTION_PROGRAM_TRIALS = 16
MINIMUM_PARENT_AGREEMENT = 0.5


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


def _blind_task(payload: Mapping[str, object]) -> BlindTask:
    train = payload["train"]
    test = payload["test"]
    if not isinstance(train, list) or not train:
        raise ValueError("blind task requires demonstrations")
    if not isinstance(test, list) or not test:
        raise ValueError("blind task requires query inputs")
    for query in test:
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


def _legacy_candidate(
    program: ObjectCodeProgram,
    task: BlindTask,
) -> dict[str, object] | None:
    executions = tuple(
        execute_object_code_program(program, query) for query in task.test_inputs
    )
    if any(not execution.ok or execution.output is None for execution in executions):
        return None
    outputs = tuple(
        execution.output for execution in executions if execution.output is not None
    )
    return {
        "source": "legacy_object_code_512",
        "program_id": object_code_program_id(program),
        "description_bits": program.description_bits,
        "program": program.to_json_dict(),
        "query_outputs": _outputs_payload(outputs),
    }


def _relational_candidate(
    program: RelationalMaskProgram,
    task: BlindTask,
    *,
    source: str,
) -> dict[str, object] | None:
    executions = tuple(
        execute_relational_mask(program, query) for query in task.test_inputs
    )
    if any(not execution.ok or execution.output is None for execution in executions):
        return None
    outputs = tuple(
        execution.output for execution in executions if execution.output is not None
    )
    return {
        "source": source,
        "program_id": relational_mask_program_id(program),
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


def _relational_score_key(
    score: RelationalMaskProgramScore,
) -> tuple[object, ...]:
    return (
        -score.exact_demo_count,
        -score.shape_match_count,
        -score.agreement,
        score.mismatch_count,
        score.program.description_bits,
        relational_mask_program_id(score.program),
    )


def _score_action_arm(
    *,
    programs: Sequence[RelationalMaskProgram],
    padding_program: RelationalMaskProgram,
    task: BlindTask,
    source: str,
) -> dict[str, object]:
    actual = tuple(programs[:ACTION_PROGRAM_TRIALS])
    candidates = []
    trials = []
    for trial_index in range(ACTION_PROGRAM_TRIALS):
        padding = trial_index >= len(actual)
        program = padding_program if padding else actual[trial_index]
        demo_executions = tuple(
            execute_relational_mask(program, pair.input) for pair in task.train
        )
        query_executions = tuple(
            execute_relational_mask(program, query) for query in task.test_inputs
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
                "program_id": relational_mask_program_id(program),
                "demo_exact": demo_exact,
                "query_valid": query_valid,
            }
        )
        if not padding and demo_exact and query_valid:
            outputs = tuple(
                execution.output
                for execution in query_executions
                if execution.output is not None
            )
            candidates.append(
                {
                    "source": source,
                    "program_id": relational_mask_program_id(program),
                    "description_bits": program.description_bits,
                    "program": program.to_json_dict(),
                    "query_outputs": _outputs_payload(outputs),
                }
            )
    deduplicated = _deduplicate_candidates(candidates)
    return {
        "reserved_program_trials": ACTION_PROGRAM_TRIALS,
        "actual_program_trials": len(actual),
        "padding_program_trials": ACTION_PROGRAM_TRIALS - len(actual),
        "demo_execution_count": ACTION_PROGRAM_TRIALS * len(task.train),
        "query_execution_count": ACTION_PROGRAM_TRIALS * len(task.test_inputs),
        "total_execution_count": ACTION_PROGRAM_TRIALS
        * (len(task.train) + len(task.test_inputs)),
        "novel_frontier_count": len(
            {relational_mask_program_id(program) for program in actual}
        ),
        "frontier_changed": bool(actual),
        "trials": trials,
        "candidates": list(deduplicated),
        "selected_candidates": list(deduplicated[:2]),
    }


def _freeze_task(
    task_record: Mapping[str, object],
    blind_root: Path,
) -> dict[str, object]:
    task_id = task_record["task_id"]
    blind_path = blind_root / f"{task_id}.json"
    if _file_sha256(blind_path) != task_record["blind_sha256"]:
        raise ValueError(f"blind task hash mismatch: {task_id}")
    task = _blind_task(_load_object(blind_path))

    all_legacy_programs = enumerate_object_code_programs(task)
    legacy_programs = tuple(all_legacy_programs[:LEGACY_PROGRAM_TRIALS])
    if len(legacy_programs) != LEGACY_PROGRAM_TRIALS:
        raise ValueError(f"legacy program prefix is incomplete: {task_id}")
    legacy_synthesis = synthesize_object_code_programs(
        task,
        max_program_trials=LEGACY_PROGRAM_TRIALS,
        max_exact_programs=LEGACY_PROGRAM_TRIALS,
        max_near_misses=0,
        programs=legacy_programs,
    )
    legacy_candidates = []
    for score in legacy_synthesis.exact_scores:
        candidate = _legacy_candidate(score.program, task)
        if candidate is not None:
            legacy_candidates.append(candidate)
    legacy_deduplicated = _deduplicate_candidates(legacy_candidates)

    relational_programs = enumerate_relational_mask_programs(task)
    if not relational_programs or len(relational_programs) > RELATIONAL_PROGRAM_CAP:
        raise ValueError(f"relational grammar violates its bound: {task_id}")
    relational_scores = tuple(
        score_relational_mask_program(program, task)
        for program in relational_programs
    )
    relational_candidates = []
    for score in relational_scores:
        if not score.all_demo_exact:
            continue
        candidate = _relational_candidate(
            score.program,
            task,
            source="relational_mask_full",
        )
        if candidate is not None:
            relational_candidates.append(candidate)
    relational_deduplicated = _deduplicate_candidates(relational_candidates)

    initial_scores = relational_scores[:INITIAL_RELATIONAL_TRIALS]
    initial_candidates = []
    for score in initial_scores:
        if not score.all_demo_exact:
            continue
        candidate = _relational_candidate(
            score.program,
            task,
            source="relational_mask_initial64",
        )
        if candidate is not None:
            initial_candidates.append(candidate)
    initial_deduplicated = _deduplicate_candidates(initial_candidates)

    parent_scores = tuple(
        sorted(
            (
                score
                for score in initial_scores
                if not score.all_demo_exact
                and score.execution_valid
                and score.agreement >= MINIMUM_PARENT_AGREEMENT
            ),
            key=_relational_score_key,
        )
    )
    padding_program = relational_programs[0]
    if parent_scores:
        parent_score = parent_scores[0]
        parent = parent_score.program
        padding_program = parent
        certificate = compile_relational_mask_failure_certificate(
            task_id=task_id,
            task=task,
            parent=parent,
        )
        variants = single_slot_relational_mask_variants(
            task,
            parent,
            allowed_slots=certificate.affected_slots,
            existing_program_ids=tuple(
                relational_mask_program_id(program)
                for program in relational_programs[:INITIAL_RELATIONAL_TRIALS]
            ),
            candidate_programs=relational_programs,
        )
        typed_programs = tuple(edit.program for edit in variants)
        parent_payload: dict[str, object] | None = {
            "program_id": relational_mask_program_id(parent),
            "program": parent.to_json_dict(),
            "exact_demo_count": parent_score.exact_demo_count,
            "shape_match_count": parent_score.shape_match_count,
            "agreement": parent_score.agreement,
            "mismatch_count": parent_score.mismatch_count,
            "certificate": certificate.to_json_dict(),
            "legal_variant_count": len(variants),
        }
    else:
        typed_programs = ()
        parent_payload = None
    typed_arm = _score_action_arm(
        programs=typed_programs,
        padding_program=padding_program,
        task=task,
        source="relational_mask_typed_repair",
    )
    cold_programs = relational_programs[
        INITIAL_RELATIONAL_TRIALS : INITIAL_RELATIONAL_TRIALS
        + ACTION_PROGRAM_TRIALS
    ]
    cold_arm = _score_action_arm(
        programs=cold_programs,
        padding_program=padding_program,
        task=task,
        source="relational_mask_cold_restart",
    )
    if (
        typed_arm["reserved_program_trials"]
        != cold_arm["reserved_program_trials"]
        or typed_arm["total_execution_count"]
        != cold_arm["total_execution_count"]
    ):
        raise RuntimeError("typed repair and cold restart costs differ")

    combined = _deduplicate_candidates(
        (*legacy_deduplicated, *relational_deduplicated)
    )
    return {
        "task_id": task_id,
        "blind_sha256": task_record["blind_sha256"],
        "blind_structural_signature": task_record["blind_structural_signature"],
        "demo_count": len(task.train),
        "query_count": len(task.test_inputs),
        "legacy": {
            "program_trial_count": legacy_synthesis.program_trial_count,
            "demo_execution_count": legacy_synthesis.demo_execution_count,
            "synthesis_query_execution_count": (
                legacy_synthesis.query_execution_count
            ),
            "freeze_query_execution_count": len(legacy_synthesis.exact_scores)
            * len(task.test_inputs),
            "exact_program_count": len(legacy_synthesis.exact_scores),
            "candidates": list(legacy_deduplicated),
            "selected_candidates": list(legacy_deduplicated[:2]),
        },
        "relational": {
            "program_trial_count": len(relational_programs),
            "demo_execution_count": len(relational_programs) * len(task.train),
            "freeze_query_execution_count": sum(
                score.all_demo_exact for score in relational_scores
            )
            * len(task.test_inputs),
            "exact_program_count": sum(
                score.all_demo_exact for score in relational_scores
            ),
            "candidates": list(relational_deduplicated),
            "selected_candidates": list(relational_deduplicated[:2]),
        },
        "combined_selected_candidates": list(combined[:2]),
        "repair": {
            "initial_program_trial_count": INITIAL_RELATIONAL_TRIALS,
            "minimum_parent_agreement": MINIMUM_PARENT_AGREEMENT,
            "initial_candidates": list(initial_deduplicated),
            "initial_selected_candidates": list(initial_deduplicated[:2]),
            "parent": parent_payload,
            "typed": typed_arm,
            "cold_restart": cold_arm,
            "cost_equal": True,
        },
    }


def _build_freeze(manifest: Mapping[str, object]) -> dict[str, object]:
    blind_root = Path(manifest["blind_task_root"])
    task_records = manifest["tasks"]
    tasks = []
    for index, task_record in enumerate(task_records, start=1):
        print(
            f"freeze {index}/{len(task_records)}: {task_record['task_id']}",
            file=sys.stderr,
            flush=True,
        )
        tasks.append(_freeze_task(task_record, blind_root))
    return {
        "schema": FREEZE_SCHEMA,
        "cohort_id": manifest["cohort_id"],
        "query_gold_read": False,
        "public_evaluation_read": False,
        "visual_provider_run": False,
        "controller_training_started": False,
        "budgets": {
            "legacy_program_trials": LEGACY_PROGRAM_TRIALS,
            "relational_program_cap": RELATIONAL_PROGRAM_CAP,
            "initial_relational_trials": INITIAL_RELATIONAL_TRIALS,
            "action_program_trials": ACTION_PROGRAM_TRIALS,
            "minimum_parent_agreement": MINIMUM_PARENT_AGREEMENT,
        },
        "tasks": tasks,
    }


def _candidate_solves(
    candidate: Mapping[str, object],
    gold_outputs: Sequence[object],
) -> bool:
    return candidate["query_outputs"] == list(gold_outputs)


def _arm_solved(
    candidates: Sequence[Mapping[str, object]],
    gold_outputs: Sequence[object],
) -> bool:
    return any(_candidate_solves(candidate, gold_outputs) for candidate in candidates)


def _score_frozen_task(
    frozen: Mapping[str, object],
    task_record: Mapping[str, object],
    arc2_training_dir: Path,
) -> dict[str, object]:
    task_id = frozen["task_id"]
    source_path = arc2_training_dir / f"{task_id}.json"
    if _file_sha256(source_path) != task_record["source_sha256"]:
        raise ValueError(f"gold source hash mismatch: {task_id}")
    gold = _load_object(source_path)
    gold_outputs = [pair["output"] for pair in gold["test"]]
    legacy_raw = _arm_solved(frozen["legacy"]["candidates"], gold_outputs)
    legacy_pass2 = _arm_solved(
        frozen["legacy"]["selected_candidates"], gold_outputs
    )
    relational_raw = _arm_solved(
        frozen["relational"]["candidates"], gold_outputs
    )
    relational_pass2 = _arm_solved(
        frozen["relational"]["selected_candidates"], gold_outputs
    )
    combined_pass2 = _arm_solved(
        frozen["combined_selected_candidates"], gold_outputs
    )
    repair = frozen["repair"]
    initial_raw = _arm_solved(repair["initial_candidates"], gold_outputs)
    initial_pass2 = _arm_solved(
        repair["initial_selected_candidates"], gold_outputs
    )
    typed_raw = _arm_solved(repair["typed"]["candidates"], gold_outputs)
    typed_pass2 = _arm_solved(
        repair["typed"]["selected_candidates"], gold_outputs
    )
    cold_raw = _arm_solved(
        repair["cold_restart"]["candidates"], gold_outputs
    )
    cold_pass2 = _arm_solved(
        repair["cold_restart"]["selected_candidates"], gold_outputs
    )
    repair_eligible = (
        repair["parent"] is not None
        and repair["typed"]["novel_frontier_count"] > 0
    )
    return {
        "task_id": task_id,
        "legacy_raw_oracle": legacy_raw,
        "legacy_pass2": legacy_pass2,
        "relational_raw_oracle": relational_raw,
        "relational_pass2": relational_pass2,
        "relational_unique_vs_legacy": relational_raw and not legacy_raw,
        "combined_pass2": combined_pass2,
        "repair_eligible": repair_eligible,
        "initial_raw_oracle": initial_raw,
        "initial_pass2": initial_pass2,
        "typed_raw_oracle": typed_raw,
        "typed_pass2": typed_pass2,
        "cold_raw_oracle": cold_raw,
        "cold_pass2": cold_pass2,
        "typed_recovery": typed_raw and not initial_raw,
        "cold_recovery": cold_raw and not initial_raw,
        "typed_unique_vs_cold": typed_raw and not initial_raw and not cold_raw,
        "cold_unique_vs_typed": cold_raw and not initial_raw and not typed_raw,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cohort_manifest", type=Path)
    parser.add_argument("arc2_training_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"gate output already exists: {args.output_dir}")
    manifest = _load_object(args.cohort_manifest)
    if manifest["schema"] != COHORT_SCHEMA:
        raise ValueError("cohort schema mismatch")
    content = {key: value for key, value in manifest.items() if key != "cohort_id"}
    if manifest["cohort_id"] != canonical_sha256(content):
        raise ValueError("cohort ID mismatch")
    if manifest["cohort_id"] != EXPECTED_COHORT_ID:
        raise ValueError("unexpected cohort ID")
    if manifest["source"]["arc2_commit"] != EXPECTED_ARC2_COMMIT:
        raise ValueError("ARC-AGI-2 source commit mismatch")

    freeze_content = _build_freeze(manifest)
    freeze = {"freeze_id": canonical_sha256(freeze_content), **freeze_content}
    freeze_path = args.output_dir / "candidate_freeze.json"
    _write_new(freeze_path, freeze)
    reread = _load_object(freeze_path)
    if reread != freeze:
        raise RuntimeError("candidate freeze changed after atomic write")
    replay_content = _build_freeze(manifest)
    if replay_content != freeze_content:
        raise RuntimeError("candidate freeze is not byte-identical on replay")

    task_records = {
        record["task_id"]: record for record in manifest["tasks"]
    }
    scores = tuple(
        _score_frozen_task(
            task,
            task_records[task["task_id"]],
            args.arc2_training_dir,
        )
        for task in freeze["tasks"]
    )
    total = len(scores)
    metric_names = tuple(key for key in scores[0] if key != "task_id")
    counts = {
        metric: sum(bool(score[metric]) for score in scores)
        for metric in metric_names
    }
    typed_net_vs_cold = counts["typed_recovery"] - counts["cold_recovery"]
    representation_gate = (
        counts["relational_unique_vs_legacy"] >= 2
        and counts["relational_unique_vs_legacy"] / total >= 0.04
    )
    frontier_gate = counts["repair_eligible"] >= 3
    repair_gate = counts["typed_recovery"] >= 3 and typed_net_vs_cold >= 1
    all_gates = representation_gate and frontier_gate and repair_gate
    result_content = {
        "schema": RESULT_SCHEMA,
        "cohort_id": manifest["cohort_id"],
        "candidate_freeze_id": freeze["freeze_id"],
        "candidate_freeze_sha256": _file_sha256(freeze_path),
        "query_gold_read_after_freeze": True,
        "public_evaluation_read": False,
        "visual_provider_run": False,
        "controller_training_started": False,
        "task_count": total,
        "counts": counts,
        "rates": {metric: counts[metric] / total for metric in metric_names},
        "typed_net_recovery_vs_cold_restart": typed_net_vs_cold,
        "gates": {
            "representation_unique_coverage": representation_gate,
            "natural_typed_frontier": frontier_gate,
            "typed_repair_over_equal_cost_restart": repair_gate,
            "all_representation_preconditions": all_gates,
            "visual_lodo_authorized": all_gates,
            "controller_training_authorized": False,
        },
        "cost_contract": {
            "action_program_trials_per_arm": ACTION_PROGRAM_TRIALS,
            "typed_restart_cost_equal_all_tasks": all(
                task["repair"]["cost_equal"] for task in freeze["tasks"]
            ),
        },
        "tasks": list(scores),
    }
    result = {"result_id": canonical_sha256(result_content), **result_content}
    _write_new(args.output_dir / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
