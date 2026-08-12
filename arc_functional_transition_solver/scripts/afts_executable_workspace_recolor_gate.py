from __future__ import annotations

import json
from itertools import product

from afts_arc.blind import BlindTask
from afts_arc.executable_workspace import (
    RECOLOR_NODE_DOMAIN,
    TypedProgramSketch,
    abstract_execute,
    abstract_recolor_domain,
    analyze_recolor_reachability,
    execute_recolor_grid,
    execute_typed_sketch,
    fill_obligation,
    insert_recolor_hole,
)
from afts_arc.grid import as_grid, grid_to_lists
from afts_arc.hybrid.scene_graph import (
    CanvasNode,
    CorrespondObjectsNode,
    ObjectOperationNode,
    ParseObjectsNode,
    RenderObjectsNode,
    ScenePipelineProgram,
    SelectObjectsNode,
)
from afts_arc.task import ARCPair


SCHEMA = "afts.executable-workspace-recolor-gate/v1"


def _crop_program() -> ScenePipelineProgram:
    return ScenePipelineProgram(
        ParseObjectsNode(0, 4, "monochrome_components"),
        CorrespondObjectsNode(),
        SelectObjectsNode("smallest_area"),
        ObjectOperationNode("crop"),
        CanvasNode("bbox", 0),
        RenderObjectsNode("source_crop"),
    )


def _task(
    demonstrations: tuple[tuple[list[list[int]], list[list[int]]], ...],
    query: list[list[int]],
) -> BlindTask:
    return BlindTask.from_observations(
        train=tuple(
            ARCPair(as_grid(source), as_grid(target))
            for source, target in demonstrations
        ),
        test_inputs=(as_grid(query),),
    )


def _cases() -> tuple[dict[str, object], ...]:
    square_four = [
        [0, 0, 0, 0],
        [0, 4, 4, 0],
        [0, 4, 4, 0],
        [0, 0, 0, 0],
    ]
    square_five = [
        [0, 0, 0, 0],
        [0, 5, 5, 0],
        [0, 5, 5, 0],
        [0, 0, 0, 0],
    ]
    ell_four = [
        [0, 4, 0],
        [0, 4, 4],
        [0, 0, 0],
    ]
    ring_four = [
        [0, 0, 0, 0, 0],
        [0, 4, 4, 4, 0],
        [0, 4, 0, 4, 0],
        [0, 4, 4, 4, 0],
        [0, 0, 0, 0, 0],
    ]
    return (
        {
            "name": "positive_single_demo_foreground_recolor",
            "kind": "positive",
            "expected_status": "reachable",
            "task": _task(
                ((square_four, [[2, 2], [2, 2]]),),
                [[0, 4], [4, 4]],
            ),
        },
        {
            "name": "positive_multi_demo_shared_recolor_with_exact_constraint",
            "kind": "positive",
            "expected_status": "reachable",
            "task": _task(
                (
                    (square_four, [[2, 2], [2, 2]]),
                    (ell_four, [[2, 0], [2, 2]]),
                    (square_five, [[5, 5], [5, 5]]),
                ),
                [[0, 4], [4, 4]],
            ),
        },
        {
            "name": "positive_crop_then_background_recolor",
            "kind": "positive",
            "expected_status": "reachable",
            "task": _task(
                (
                    (
                        ring_four,
                        [
                            [4, 4, 4],
                            [4, 8, 4],
                            [4, 4, 4],
                        ],
                    ),
                ),
                ring_four,
            ),
        },
        {
            "name": "negative_shape_unreachable",
            "kind": "negative",
            "expected_status": "unreachable",
            "task": _task(
                ((square_four, [[2, 2], [2, 2], [2, 2]]),),
                square_four,
            ),
        },
        {
            "name": "negative_one_source_requires_two_targets",
            "kind": "negative",
            "expected_status": "unreachable",
            "task": _task(
                ((square_four, [[2, 1], [2, 2]]),),
                square_four,
            ),
        },
        {
            "name": "negative_cross_demo_inconsistent",
            "kind": "negative",
            "expected_status": "unreachable",
            "task": _task(
                (
                    (square_four, [[2, 2], [2, 2]]),
                    (square_four, [[3, 3], [3, 3]]),
                ),
                square_four,
            ),
        },
        {
            "name": "negative_base_exact_suppresses_insertion",
            "kind": "negative",
            "expected_status": "base_exact",
            "task": _task(
                ((square_four, [[4, 4], [4, 4]]),),
                square_four,
            ),
        },
        {
            "name": "negative_invalid_base_execution",
            "kind": "negative",
            "expected_status": "unreachable",
            "task": _task(
                (([[0, 0], [0, 0]], [[2]]),),
                [[0]],
            ),
        },
    )


def _evaluate_case(
    base: TypedProgramSketch,
    case: dict[str, object],
) -> dict[str, object]:
    task = case["task"]
    if not isinstance(task, BlindTask):
        raise TypeError("controlled recolor case must contain a BlindTask")
    reachability = analyze_recolor_reachability(base, task)
    base_result = abstract_execute(base, task)
    closure_records: list[dict[str, object]] = []
    inserted_sketch_id: str | None = None
    inserted_topology: str | None = None
    insertion_obligation_id: str | None = None
    if reachability.status == "reachable":
        obligation = reachability.obligations[0]
        inserted = insert_recolor_hole(
            base,
            obligation,
            hole_id="controlled-post-recolor-hole",
        )
        incomplete = abstract_execute(inserted, task)
        inserted_sketch_id = inserted.sketch_id
        inserted_topology = inserted.topology
        insertion_obligation_id = obligation.obligation_id
        for candidate in reachability.candidates:
            completed = fill_obligation(
                inserted,
                incomplete.obligations[0],
                candidate,
            )
            exact = abstract_execute(completed, task)
            query_execution = execute_typed_sketch(completed, task.test_inputs[0])
            closure_records.append(
                {
                    "node": candidate.to_json_dict(),
                    "node_id": candidate.node_id,
                    "completed_sketch_id": completed.sketch_id,
                    "demo_exact": exact.exact,
                    "query_execution_ok": query_execution.ok,
                    "query_output": (
                        None
                        if query_execution.output is None
                        else grid_to_lists(query_execution.output)
                    ),
                }
            )
    expected_status = case["expected_status"]
    if not isinstance(expected_status, str):
        raise TypeError("controlled recolor expected status must be a string")
    kind = case["kind"]
    if kind not in {"positive", "negative"}:
        raise ValueError("controlled recolor case has an unknown kind")
    all_candidates_close = bool(closure_records) and all(
        record["demo_exact"] is True for record in closure_records
    )
    return {
        "name": case["name"],
        "kind": kind,
        "blind_task_id": task.task_id,
        "base_status": base_result.status,
        "reachability_status": reachability.status,
        "expected_status": expected_status,
        "status_matches": reachability.status == expected_status,
        "recolor_candidates": [
            node.to_json_dict() for node in reachability.candidates
        ],
        "novel_frontier_count": reachability.novel_frontier_count,
        "obligation_types": [
            obligation.obligation_type for obligation in reachability.obligations
        ],
        "insertion_obligation_id": insertion_obligation_id,
        "inserted_sketch_id": inserted_sketch_id,
        "inserted_topology": inserted_topology,
        "closure_records": closure_records,
        "all_candidates_close": all_candidates_close,
        "false_positive": kind == "negative" and reachability.status == "reachable",
    }


def _exhaustive_domain_audit() -> dict[str, object]:
    palette = range(3)
    comparisons = 0
    mismatch_count = 0
    for predicted_values in product(palette, repeat=2):
        predicted = as_grid([list(predicted_values)])
        for expected_values in product(palette, repeat=2):
            expected = as_grid([list(expected_values)])
            abstract = abstract_recolor_domain(predicted, expected)
            concrete = tuple(
                node
                for node in RECOLOR_NODE_DOMAIN
                if execute_recolor_grid(node, predicted) == expected
            )
            comparisons += 1
            mismatch_count += int(abstract != concrete)
    return {
        "palette": [0, 1, 2],
        "grid_shape": [1, 2],
        "abstract_concrete_comparisons": comparisons,
        "domain_program_count": len(RECOLOR_NODE_DOMAIN),
        "mismatch_count": mismatch_count,
        "exact": mismatch_count == 0,
        "sound": mismatch_count == 0,
    }


def construct_result() -> dict[str, object]:
    base = TypedProgramSketch.from_scene_pipeline(_crop_program())
    case_results = tuple(_evaluate_case(base, case) for case in _cases())
    positive = tuple(case for case in case_results if case["kind"] == "positive")
    negative = tuple(case for case in case_results if case["kind"] == "negative")
    exhaustive = _exhaustive_domain_audit()
    positive_closed = sum(
        case["status_matches"] is True
        and case["novel_frontier_count"] > 0
        and case["all_candidates_close"] is True
        for case in positive
    )
    negative_false_positive = sum(case["false_positive"] is True for case in negative)
    status_match_count = sum(case["status_matches"] is True for case in case_results)
    gate_pass = (
        positive_closed == len(positive)
        and negative_false_positive == 0
        and status_match_count == len(case_results)
        and exhaustive["exact"] is True
    )
    return {
        "schema": SCHEMA,
        "candidate": "scene-pipeline-plus-one-global-recolor-node",
        "base_sketch_id": base.sketch_id,
        "case_results": list(case_results),
        "exhaustive_domain_audit": exhaustive,
        "summary": {
            "case_count": len(case_results),
            "positive_count": len(positive),
            "positive_closed": positive_closed,
            "negative_count": len(negative),
            "negative_false_positive": negative_false_positive,
            "status_match_count": status_match_count,
            "gate_pass": gate_pass,
        },
    }


def main() -> None:
    print(
        json.dumps(
            construct_result(),
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
