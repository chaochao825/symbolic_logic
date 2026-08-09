"""Freeze and score the visual relational trace repair gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = str((PROJECT_ROOT / "src").resolve())
if SOURCE_ROOT not in sys.path:
    sys.path.insert(0, SOURCE_ROOT)

from afts_arc.blind import BlindTask  # noqa: E402
from afts_arc.experiment_safety import atomic_write_json, canonical_sha256  # noqa: E402
from afts_arc.grid import Grid, as_grid, grid_to_lists  # noqa: E402
from afts_arc.hybrid.object_code import (  # noqa: E402
    enumerate_object_code_programs,
    object_code_program_from_json,
    object_code_program_id,
)
from afts_arc.hybrid.scene_graph import (  # noqa: E402
    ScenePipelineProgram,
    enumerate_scene_pipeline_programs,
    execute_scene_pipeline,
)
from afts_arc.task import load_task  # noqa: E402
from afts_arc.visual_provider_gate import (  # noqa: E402
    COHORT_SCHEMA as VISUAL_PROVIDER_COHORT_SCHEMA,
    FROZEN_PREDICTIONS_SCHEMA,
    _file_sha256,
    _load_json_object,
    _load_provider_predictions,
    _validate_grid,
    _validate_provider_predictions,
)
from afts_arc.visual_trace_repair import (  # noqa: E402
    compile_visual_trace_certificate,
    query_posterior_rank_key,
    relational_difference_groups,
    single_slot_scene_variants,
)


TRACE_REPAIR_COHORT_SCHEMA = "afts.visual-trace-repair-cohort/v1"
TRACE_REPAIR_CANDIDATE_SCHEMA = "afts.visual-trace-repair-candidates/v1"
TRACE_REPAIR_SCORE_SCHEMA = "afts.visual-trace-repair-score/v1"
ACTION_TRIAL_BUDGET = 16
COLD_ORDER_SEED = "visual-trace-repair-cold-v1-20260809"
ARM_NAMES = ("visual_typed", "bridge_lesion", "cold_restart")


def _normalize_samples(samples: Sequence[object]) -> tuple[Grid, ...]:
    return tuple(_validate_grid(sample) for sample in samples)


def _episode_samples(
    task_record: Mapping[str, object],
    predictions: Mapping[str, Mapping[int, Sequence[object]]],
) -> tuple[tuple[tuple[Grid, ...], ...], tuple[Grid, ...]]:
    episodes = task_record["episodes"]
    if not isinstance(episodes, list) or len(episodes) != 4:
        raise ValueError("trace-repair task must contain four provider episodes")
    by_kind = {episode["kind"]: episode for episode in episodes}
    if set(by_kind) != {"query", "lodo0", "lodo1", "lodo2"}:
        raise ValueError("trace-repair provider episode kinds are malformed")
    lodo = tuple(
        _normalize_samples(predictions[by_kind[f"lodo{index}"]["episode_id"]][0])
        for index in range(3)
    )
    query = _normalize_samples(predictions[by_kind["query"]["episode_id"]][0])
    return lodo, query


def _candidate_record(
    *,
    program: ScenePipelineProgram,
    slot: str,
    task: BlindTask,
    query_samples: Sequence[Grid],
) -> dict[str, object]:
    demo_executions = tuple(
        execute_scene_pipeline(program, pair.input) for pair in task.train
    )
    query_executions = tuple(
        execute_scene_pipeline(program, query) for query in task.test_inputs
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
    query_outputs = tuple(
        execution.output
        for execution in query_executions
        if execution.output is not None
    )
    exact_support_count = 0
    relation_distance_sum = 0
    if query_valid:
        if len(query_outputs) != 1:
            raise ValueError("trace-repair gate currently requires one query")
        output = query_outputs[0]
        exact_support_count = sum(sample == output for sample in query_samples)
        relation_distance_sum = sum(
            len(relational_difference_groups(program, output, sample))
            for sample in query_samples
        )
    return {
        "program_id": object_code_program_id(program),
        "program": program.to_json_dict(),
        "edited_slot": slot,
        "demo_exact": demo_exact,
        "demo_execution_valid": [execution.ok for execution in demo_executions],
        "demo_outputs": [
            None if execution.output is None else grid_to_lists(execution.output)
            for execution in demo_executions
        ],
        "query_execution_valid": query_valid,
        "query_outputs": [grid_to_lists(output) for output in query_outputs],
        "query_posterior_exact_support_count": exact_support_count,
        "query_posterior_relation_distance_sum": relation_distance_sum,
        "query_posterior_sample_count": len(query_samples),
        "description_bits": program.description_bits,
    }


def _rank_selectable(
    records: Sequence[dict[str, object]],
    query_samples: Sequence[Grid],
) -> tuple[dict[str, object], ...]:
    selectable = []
    for record in records:
        if record["demo_exact"] is not True or record["query_execution_valid"] is not True:
            continue
        program = object_code_program_from_json(record["program"])
        if not isinstance(program, ScenePipelineProgram):
            raise TypeError("trace-repair candidates must be scene programs")
        outputs = record["query_outputs"]
        if not isinstance(outputs, list) or len(outputs) != 1:
            raise ValueError("trace-repair selectable candidate has malformed output")
        output = as_grid(outputs[0])
        selectable.append((query_posterior_rank_key(program, output, query_samples), record))
    return tuple(record for _, record in sorted(selectable, key=lambda item: item[0])[:2])


def _arm_record(
    *,
    name: str,
    programs: Sequence[tuple[str, ScenePipelineProgram]],
    parent: ScenePipelineProgram,
    task: BlindTask,
    query_samples: Sequence[Grid],
    initial_ids: frozenset[str],
) -> dict[str, object]:
    if name not in ARM_NAMES:
        raise ValueError("unknown trace-repair arm")
    selected_programs = tuple(programs[:ACTION_TRIAL_BUDGET])
    program_ids = tuple(object_code_program_id(program) for _, program in selected_programs)
    if len(program_ids) != len(set(program_ids)):
        raise ValueError("trace-repair arm contains duplicate programs")
    if any(program_id in initial_ids for program_id in program_ids):
        raise ValueError("trace-repair arm re-emits an initial-pool program")
    records = tuple(
        _candidate_record(
            program=program,
            slot=slot,
            task=task,
            query_samples=query_samples,
        )
        for slot, program in selected_programs
    )
    padding_count = ACTION_TRIAL_BUDGET - len(selected_programs)
    for _ in range(padding_count):
        for pair in task.train:
            execute_scene_pipeline(parent, pair.input)
        for query in task.test_inputs:
            execute_scene_pipeline(parent, query)
    novel_frontier_count = len(records)
    selected = _rank_selectable(records, query_samples)
    return {
        "name": name,
        "candidate_records": list(records),
        "selected_program_ids": [record["program_id"] for record in selected],
        "selected_query_outputs": [record["query_outputs"] for record in selected],
        "demo_exact_candidate_count": sum(record["demo_exact"] is True for record in records),
        "novel_frontier_count": novel_frontier_count,
        "frontier_changed": novel_frontier_count > 0,
        "frontier_change_basis": "content-addressed one-existing-slot program ID",
        "padding_execution_count": padding_count,
        "native_cost": {
            "reserved_program_trials": ACTION_TRIAL_BUDGET,
            "observed_program_trials": len(records) + padding_count,
            "reserved_demo_executions": ACTION_TRIAL_BUDGET * len(task.train),
            "observed_demo_executions": ACTION_TRIAL_BUDGET * len(task.train),
            "reserved_query_executions": ACTION_TRIAL_BUDGET * len(task.test_inputs),
            "observed_query_executions": ACTION_TRIAL_BUDGET * len(task.test_inputs),
        },
    }


def _cold_programs(
    task_id: str,
    task: BlindTask,
    initial_ids: frozenset[str],
) -> tuple[tuple[str, ScenePipelineProgram], ...]:
    programs = tuple(
        program
        for program in enumerate_scene_pipeline_programs(task)
        if object_code_program_id(program) not in initial_ids
    )
    return tuple(
        ("cold_restart", program)
        for program in sorted(
            programs,
            key=lambda program: (
                hashlib.sha256(
                    (
                        f"{COLD_ORDER_SEED}\0{task_id}\0"
                        f"{object_code_program_id(program)}"
                    ).encode("ascii")
                ).hexdigest(),
                object_code_program_id(program),
            ),
        )
    )


def _verify_initial_pool(
    task: BlindTask,
    task_record: Mapping[str, object],
) -> tuple[frozenset[str], ScenePipelineProgram]:
    count = task_record["initial_program_trial_count"]
    if type(count) is not int or count < 1:
        raise ValueError("initial program trial count is malformed")
    observed = tuple(
        object_code_program_id(program)
        for program in enumerate_object_code_programs(task)[:count]
    )
    expected = tuple(task_record["initial_program_ids"])
    if observed != expected:
        raise ValueError("initial object/code pool does not replay")
    parent_payload = task_record["parent"]
    parent = object_code_program_from_json(parent_payload["program"])
    if not isinstance(parent, ScenePipelineProgram):
        raise TypeError("visual trace repair parent is not a scene program")
    parent_id = object_code_program_id(parent)
    if parent_id != parent_payload["program_id"] or parent_id not in observed:
        raise ValueError("visual trace repair parent identity is invalid")
    return frozenset(observed), parent


def freeze_candidates(args: argparse.Namespace) -> dict[str, object]:
    if args.output_path.exists():
        raise FileExistsError(f"candidate artifact already exists: {args.output_path}")
    manifest = _load_json_object(args.cohort_manifest)
    provider_manifest = _load_json_object(args.provider_manifest)
    frozen = _load_json_object(args.frozen_predictions)
    if manifest["schema"] != TRACE_REPAIR_COHORT_SCHEMA:
        raise ValueError("trace-repair cohort schema mismatch")
    if provider_manifest["schema"] != VISUAL_PROVIDER_COHORT_SCHEMA:
        raise ValueError("provider cohort schema mismatch")
    if manifest["provider_cohort_id"] != provider_manifest["cohort_id"]:
        raise ValueError("trace-repair and provider cohort IDs differ")
    if manifest["provider_manifest_sha256"] != _file_sha256(args.provider_manifest):
        raise ValueError("provider manifest hash differs from trace-repair cohort")
    if frozen["schema"] != FROZEN_PREDICTIONS_SCHEMA:
        raise ValueError("frozen visual prediction schema mismatch")
    if frozen["cohort_id"] != provider_manifest["cohort_id"]:
        raise ValueError("frozen predictions bind a different provider cohort")

    provider_ids = tuple(record["task_id"] for record in provider_manifest["tasks"])
    raw_predictions, prediction_files = _load_provider_predictions(
        provider_ids, tuple(args.prediction_root)
    )
    combined, prediction_validation = _validate_provider_predictions(
        raw_predictions,
        invalid_candidate_policy=args.invalid_candidate_policy,
    )
    if frozen["raw_prediction_payload_sha256"] != canonical_sha256(raw_predictions):
        raise ValueError("raw visual predictions differ from the frozen artifact")
    if frozen["validated_prediction_payload_sha256"] != canonical_sha256(combined):
        raise ValueError("validated visual predictions differ from the frozen artifact")
    if frozen["prediction_validation"] != prediction_validation:
        raise ValueError("visual prediction validation differs from the freeze")

    contexts = []
    task_records = manifest["tasks"]
    if not isinstance(task_records, list) or len(task_records) != 12:
        raise ValueError("trace-repair gate requires exactly 12 tasks")
    for task_record in task_records:
        task_id = task_record["task_id"]
        blind_path = args.blind_task_dir / f"{task_id}.json"
        if _file_sha256(blind_path) != task_record["blind_sha256"]:
            raise ValueError("blind task hash differs from cohort manifest")
        blind_payload = _load_json_object(blind_path)
        for query in blind_payload["test"]:
            if query["output"] != query["input"]:
                raise ValueError("blind task query sentinel differs from its input")
        task = BlindTask.from_task(load_task(blind_path))
        if task.task_id != task_record["blind_task_id"]:
            raise ValueError("blind task content identity differs from cohort manifest")
        initial_ids, parent = _verify_initial_pool(task, task_record)
        demo_executions = tuple(
            execute_scene_pipeline(parent, pair.input) for pair in task.train
        )
        query_execution = execute_scene_pipeline(parent, task.test_inputs[0])
        lodo_samples, query_samples = _episode_samples(task_record, combined)
        contexts.append(
            {
                "record": task_record,
                "task": task,
                "initial_ids": initial_ids,
                "parent": parent,
                "demo_executions": demo_executions,
                "query_execution": query_execution,
                "lodo_samples": lodo_samples,
                "query_samples": query_samples,
            }
        )

    frozen_tasks = []
    for index, context in enumerate(contexts):
        task_record = context["record"]
        task = context["task"]
        parent = context["parent"]
        visual = compile_visual_trace_certificate(
            task_id=task_record["task_id"],
            parent_program=parent,
            demo_executions=context["demo_executions"],
            demo_gold_outputs=tuple(pair.output for pair in task.train),
            lodo_samples=context["lodo_samples"],
            query_execution=context["query_execution"],
            query_samples=context["query_samples"],
            mode="visual_lodo",
        )
        lesion = compile_visual_trace_certificate(
            task_id=task_record["task_id"],
            parent_program=parent,
            demo_executions=context["demo_executions"],
            demo_gold_outputs=tuple(pair.output for pair in task.train),
            lodo_samples=context["lodo_samples"],
            query_execution=context["query_execution"],
            query_samples=context["query_samples"],
            mode="bridge_lesion",
        )
        shuffled_source = contexts[(index + 1) % len(contexts)]
        shuffled = compile_visual_trace_certificate(
            task_id=task_record["task_id"],
            parent_program=parent,
            demo_executions=context["demo_executions"],
            demo_gold_outputs=tuple(pair.output for pair in task.train),
            lodo_samples=shuffled_source["lodo_samples"],
            query_execution=context["query_execution"],
            query_samples=context["query_samples"],
            mode="visual_lodo",
        )
        scene_programs = enumerate_scene_pipeline_programs(task)
        visual_edits = single_slot_scene_variants(
            task,
            parent,
            allowed_slots=visual.affected_slots,
            existing_program_ids=tuple(context["initial_ids"]),
            candidate_programs=scene_programs,
        )
        lesion_edits = single_slot_scene_variants(
            task,
            parent,
            allowed_slots=lesion.affected_slots,
            existing_program_ids=tuple(context["initial_ids"]),
            candidate_programs=scene_programs,
        )
        arms = {
            "visual_typed": _arm_record(
                name="visual_typed",
                programs=tuple((edit.slot, edit.program) for edit in visual_edits),
                parent=parent,
                task=task,
                query_samples=context["query_samples"],
                initial_ids=context["initial_ids"],
            ),
            "bridge_lesion": _arm_record(
                name="bridge_lesion",
                programs=tuple((edit.slot, edit.program) for edit in lesion_edits),
                parent=parent,
                task=task,
                query_samples=context["query_samples"],
                initial_ids=context["initial_ids"],
            ),
            "cold_restart": _arm_record(
                name="cold_restart",
                programs=_cold_programs(
                    task_record["task_id"], task, context["initial_ids"]
                ),
                parent=parent,
                task=task,
                query_samples=context["query_samples"],
                initial_ids=context["initial_ids"],
            ),
        }
        costs = tuple(arms[name]["native_cost"] for name in ARM_NAMES)
        if any(cost != costs[0] for cost in costs[1:]):
            raise ValueError("trace-repair arms have unequal native action cost")
        frozen_tasks.append(
            {
                "task_id": task_record["task_id"],
                "blind_task_id": task.task_id,
                "parent_program_id": object_code_program_id(parent),
                "visual_certificate": visual.to_json_dict(),
                "lesion_certificate": lesion.to_json_dict(),
                "shuffled_certificate": shuffled.to_json_dict(),
                "visual_vs_lesion_changed": (
                    visual.diagnosis_group != lesion.diagnosis_group
                    or visual.affected_slots != lesion.affected_slots
                ),
                "visual_vs_shuffle_changed": (
                    visual.diagnosis_group != shuffled.diagnosis_group
                    or visual.affected_slots != shuffled.affected_slots
                ),
                "arms": arms,
                "strict_action_cost_comparable": True,
            }
        )

    content = {
        "schema": TRACE_REPAIR_CANDIDATE_SCHEMA,
        "cohort_id": manifest["cohort_id"],
        "cohort_manifest_sha256": _file_sha256(args.cohort_manifest),
        "provider_cohort_id": provider_manifest["cohort_id"],
        "provider_manifest_sha256": _file_sha256(args.provider_manifest),
        "frozen_prediction_id": frozen["frozen_prediction_id"],
        "frozen_prediction_sha256": _file_sha256(args.frozen_predictions),
        "prediction_files": prediction_files,
        "prediction_validation": prediction_validation,
        "configuration": {
            "action_trial_budget": ACTION_TRIAL_BUDGET,
            "cold_order_seed": COLD_ORDER_SEED,
            "complete_grid_posterior_only": True,
            "grid_synthesis": False,
            "controller_frozen": True,
        },
        "tasks": frozen_tasks,
    }
    artifact = {"candidate_id": canonical_sha256(content), **content}
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_path, artifact)
    return artifact


def _selected_outputs(arm: Mapping[str, object]) -> tuple[Grid, ...]:
    selected = arm["selected_query_outputs"]
    if not isinstance(selected, list):
        raise ValueError("selected query outputs are malformed")
    outputs = []
    for record in selected:
        if not isinstance(record, list) or len(record) != 1:
            raise ValueError("selected trace-repair candidate must have one query output")
        outputs.append(as_grid(record[0]))
    return tuple(outputs)


def score_candidates(args: argparse.Namespace) -> dict[str, object]:
    if args.output_path.exists():
        raise FileExistsError(f"score artifact already exists: {args.output_path}")
    manifest = _load_json_object(args.cohort_manifest)
    candidate = _load_json_object(args.candidate_artifact)
    replay = _load_json_object(args.replay_artifact)
    if candidate != replay or _file_sha256(args.candidate_artifact) != _file_sha256(
        args.replay_artifact
    ):
        raise ValueError("pre-gold trace-repair candidate replay is not byte-identical")
    if candidate["schema"] != TRACE_REPAIR_CANDIDATE_SCHEMA:
        raise ValueError("trace-repair candidate schema mismatch")
    if candidate["cohort_id"] != manifest["cohort_id"]:
        raise ValueError("trace-repair candidate binds a different cohort")
    manifest_by_id = {record["task_id"]: record for record in manifest["tasks"]}
    task_results = []
    recoveries = {name: 0 for name in ARM_NAMES}
    raw_recoveries = {name: 0 for name in ARM_NAMES}
    for task_record in candidate["tasks"]:
        task_id = task_record["task_id"]
        manifest_record = manifest_by_id[task_id]
        gold_path = args.gold_training_dir / f"{task_id}.json"
        if _file_sha256(gold_path) != manifest_record["gold_sha256"]:
            raise ValueError("query-gold file differs from cohort manifest")
        gold_payload = _load_json_object(gold_path)
        gold_examples = gold_payload["test"]
        if not isinstance(gold_examples, list) or len(gold_examples) != 1:
            raise ValueError("trace-repair gate requires one gold query")
        gold = _validate_grid(gold_examples[0]["output"])
        arm_results = {}
        for name in ARM_NAMES:
            arm = task_record["arms"][name]
            selected_outputs = _selected_outputs(arm)
            pass_at_2 = any(output == gold for output in selected_outputs)
            raw = any(
                record["demo_exact"] is True
                and record["query_execution_valid"] is True
                and any(as_grid(output) == gold for output in record["query_outputs"])
                for record in arm["candidate_records"]
            )
            recoveries[name] += int(pass_at_2)
            raw_recoveries[name] += int(raw)
            arm_results[name] = {
                "demo_recovered": arm["demo_exact_candidate_count"] > 0,
                "raw_query_recovered": raw,
                "pass_at_2": pass_at_2,
                "selected_program_ids": arm["selected_program_ids"],
            }
        visual_unique = (
            arm_results["visual_typed"]["pass_at_2"]
            and not arm_results["bridge_lesion"]["pass_at_2"]
            and not arm_results["cold_restart"]["pass_at_2"]
        )
        task_results.append(
            {
                "task_id": task_id,
                "visual_vs_lesion_changed": task_record["visual_vs_lesion_changed"],
                "visual_vs_shuffle_changed": task_record["visual_vs_shuffle_changed"],
                "visual_unique_over_both_controls": visual_unique,
                "arms": arm_results,
            }
        )

    lesion_changes = sum(record["visual_vs_lesion_changed"] for record in task_results)
    shuffle_changes = sum(record["visual_vs_shuffle_changed"] for record in task_results)
    unique_visual = sum(
        record["visual_unique_over_both_controls"] for record in task_results
    )
    cost_guard = all(
        task_record["strict_action_cost_comparable"] is True
        and all(
            arm["frontier_changed"] is (arm["novel_frontier_count"] > 0)
            for arm in task_record["arms"].values()
        )
        for task_record in candidate["tasks"]
    )
    validity_guard = (
        len(task_results) == 12
        and cost_guard
        and candidate == replay
        and candidate["configuration"]["controller_frozen"] is True
    )
    causal_guard = lesion_changes >= 3 or shuffle_changes >= 3
    passed = (
        validity_guard
        and causal_guard
        and unique_visual >= 1
        and recoveries["visual_typed"] > recoveries["bridge_lesion"]
        and recoveries["visual_typed"] > recoveries["cold_restart"]
    )
    if not validity_guard:
        classification = "invalid"
    elif (
        recoveries["visual_typed"] < recoveries["bridge_lesion"]
        or recoveries["visual_typed"] < recoveries["cold_restart"]
    ):
        classification = "adverse"
    elif passed:
        classification = "pass"
    elif causal_guard and (
        recoveries["visual_typed"] > recoveries["bridge_lesion"]
        or recoveries["visual_typed"] > recoveries["cold_restart"]
    ):
        classification = "boundary"
    else:
        classification = "null"
    content = {
        "schema": TRACE_REPAIR_SCORE_SCHEMA,
        "cohort_id": manifest["cohort_id"],
        "candidate_id": candidate["candidate_id"],
        "candidate_sha256": _file_sha256(args.candidate_artifact),
        "candidate_replay_byte_identical": True,
        "metrics": {
            "task_count": len(task_results),
            "pass_at_2_recoveries": recoveries,
            "raw_query_recoveries": raw_recoveries,
            "visual_unique_over_both_controls": unique_visual,
            "visual_vs_lesion_certificate_changes": lesion_changes,
            "visual_vs_shuffle_certificate_changes": shuffle_changes,
        },
        "guards": {
            "validity": validity_guard,
            "strict_action_cost_and_frontier_integrity": cost_guard,
            "residual_causality": causal_guard,
        },
        "decision": {
            "classification": classification,
            "feasibility_gate_passed": passed,
            "advance_to_fresh_100_case_confirmation": passed,
            "controller_remains_frozen": True,
        },
        "claim_scope": {
            "repair_opportunity_cohort": True,
            "general_arc_coverage_estimate": False,
            "matched_comparison_conditional_on_shared_visual_posterior": True,
            "cross_provider_scalar_cost_claim": False,
        },
        "tasks": task_results,
    }
    artifact = {"result_id": canonical_sha256(content), **content}
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_path, artifact)
    return artifact


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    freeze = commands.add_parser("freeze")
    freeze.add_argument("cohort_manifest", type=Path)
    freeze.add_argument("provider_manifest", type=Path)
    freeze.add_argument("blind_task_dir", type=Path)
    freeze.add_argument("frozen_predictions", type=Path)
    freeze.add_argument("output_path", type=Path)
    freeze.add_argument("--prediction-root", action="append", type=Path, required=True)
    freeze.add_argument(
        "--invalid-candidate-policy", choices=("error", "reject"), required=True
    )

    score = commands.add_parser("score")
    score.add_argument("cohort_manifest", type=Path)
    score.add_argument("gold_training_dir", type=Path)
    score.add_argument("candidate_artifact", type=Path)
    score.add_argument("replay_artifact", type=Path)
    score.add_argument("output_path", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = freeze_candidates(args) if args.command == "freeze" else score_candidates(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
