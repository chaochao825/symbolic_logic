"""Query-blind structural use of a frozen visual-provider posterior."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

from afts_arc.experiment_safety import atomic_write_json, canonical_sha256
from afts_arc.visual_provider_gate import (
    COHORT_SCHEMA,
    FROZEN_PREDICTIONS_SCHEMA,
    SCORE_SCHEMA,
    VisualProviderGateError,
    _grid_json,
    _load_json_object,
    _load_provider_predictions,
    _validate_grid,
    _validate_provider_predictions,
)


STRUCTURE_CANDIDATE_SCHEMA = "afts.visual-structure-bridge-candidates/v1"
STRUCTURE_CERTIFICATE_SCHEMA = "afts.visual-structure-certificate/v1"
STRUCTURE_SCORE_SCHEMA = "afts.visual-structure-bridge/v1"

Grid = tuple[tuple[int, ...], ...]
Cell = tuple[int, int]
Component = tuple[int, tuple[Cell, ...]]

CANVAS_FEATURES = frozenset(
    {
        "shape_delta",
        "shape_relation",
    }
)
BACKGROUND_FEATURES = frozenset({"background_preserved"})
OBJECT_FEATURES = frozenset(
    {
        "added_color_count",
        "changed_component_count",
        "changed_pixel_count",
        "colored_shape_relation",
        "component_area_relation",
        "component_count_delta",
        "component_count_relation",
        "foreground_count_relation",
        "foreground_bbox_is_output_shape",
        "foreground_palette_relation",
        "input_contains_output",
        "object_bbox_is_output_shape",
        "output_contains_input",
        "removed_color_count",
        "uncolored_shape_relation",
    }
)

ACTION_SLOTS = {
    "canvas_reinfer": (
        "ast.canvas.background",
        "ast.canvas.height",
        "ast.canvas.mode",
        "ast.canvas.padding",
        "ast.canvas.width",
        "ast.render.mode",
    ),
    "object_rematch": (
        "ast.correspond.d4_invariant",
        "ast.correspond.features",
        "ast.correspond.policy",
        "ast.operate.transform",
        "ast.parse.connectivity",
        "ast.parse.grouping",
        "ast.select.role",
    ),
    "reparse_background": ("ast.parse.background",),
}


def _shape(grid: Grid) -> tuple[int, int]:
    return len(grid), len(grid[0])


def _modal_color(grid: Grid) -> int:
    counts = Counter(cell for row in grid for cell in row)
    return min(counts, key=lambda color: (-counts[color], color))


def _neighbors(cell: Cell) -> tuple[Cell, ...]:
    row, column = cell
    return (
        (row - 1, column),
        (row, column - 1),
        (row, column + 1),
        (row + 1, column),
    )


def _components(grid: Grid, background: int) -> tuple[Component, ...]:
    height, width = _shape(grid)
    remaining = {
        (row, column)
        for row in range(height)
        for column in range(width)
        if grid[row][column] != background
    }
    found: list[Component] = []
    while remaining:
        seed = min(remaining)
        color = grid[seed[0]][seed[1]]
        remaining.remove(seed)
        frontier = [seed]
        cells: set[Cell] = set()
        while frontier:
            cell = frontier.pop()
            cells.add(cell)
            for neighbor in _neighbors(cell):
                row, column = neighbor
                if (
                    0 <= row < height
                    and 0 <= column < width
                    and neighbor in remaining
                    and grid[row][column] == color
                ):
                    remaining.remove(neighbor)
                    frontier.append(neighbor)
        found.append((color, tuple(sorted(cells))))
    return tuple(sorted(found))


def _component_shape(component: Component) -> tuple[Cell, ...]:
    _, cells = component
    top = min(row for row, _ in cells)
    left = min(column for _, column in cells)
    return tuple((row - top, column - left) for row, column in cells)


def _component_bbox_shape(component: Component) -> tuple[int, int]:
    _, cells = component
    return (
        max(row for row, _ in cells) - min(row for row, _ in cells) + 1,
        max(column for _, column in cells) - min(column for _, column in cells) + 1,
    )


def _foreground_bbox_shape(grid: Grid, background: int) -> tuple[int, int] | None:
    cells = [
        (row, column)
        for row, values in enumerate(grid)
        for column, color in enumerate(values)
        if color != background
    ]
    if not cells:
        return None
    return (
        max(row for row, _ in cells) - min(row for row, _ in cells) + 1,
        max(column for _, column in cells) - min(column for _, column in cells) + 1,
    )


def _contains_subgrid(container: Grid, candidate: Grid) -> bool:
    container_height, container_width = _shape(container)
    height, width = _shape(candidate)
    if height > container_height or width > container_width:
        return False
    return any(
        all(
            container[top + row][left : left + width] == candidate[row]
            for row in range(height)
        )
        for top in range(container_height - height + 1)
        for left in range(container_width - width + 1)
    )


def _relation(first: int, second: int) -> str:
    if first == second:
        return "equal"
    return "less" if first < second else "greater"


def _multiset_relation(first: Counter[object], second: Counter[object]) -> str:
    if first == second:
        return "equal"
    if all(first[key] <= second[key] for key in first):
        return "first_subset"
    if all(second[key] <= first[key] for key in second):
        return "first_superset"
    if set(first) & set(second):
        return "overlap"
    return "disjoint"


def transition_features(source: Grid, target: Grid) -> dict[str, object]:
    """Return deterministic object/canvas transition features for one grid pair."""

    source_shape = _shape(source)
    target_shape = _shape(target)
    source_background = _modal_color(source)
    target_background = _modal_color(target)
    source_objects = _components(source, source_background)
    target_objects = _components(target, target_background)
    source_palette = frozenset(cell for row in source for cell in row)
    target_palette = frozenset(cell for row in target for cell in row)
    source_foreground_palette = source_palette - {source_background}
    target_foreground_palette = target_palette - {target_background}
    source_areas = Counter(len(cells) for _, cells in source_objects)
    target_areas = Counter(len(cells) for _, cells in target_objects)
    source_shapes = Counter(_component_shape(item) for item in source_objects)
    target_shapes = Counter(_component_shape(item) for item in target_objects)
    source_colored_shapes = Counter(
        (color, _component_shape(item)) for item in source_objects for color in (item[0],)
    )
    target_colored_shapes = Counter(
        (color, _component_shape(item)) for item in target_objects for color in (item[0],)
    )
    changed_count: int | None = None
    changed_component_count: int | None = None
    if source_shape == target_shape:
        changed_grid = tuple(
            tuple(
                int(source[row][column] != target[row][column])
                for column in range(source_shape[1])
            )
            for row in range(source_shape[0])
        )
        changed_count = sum(cell for row in changed_grid for cell in row)
        changed_component_count = len(_components(changed_grid, 0))
    source_foreground_count = sum(
        cell != source_background for row in source for cell in row
    )
    target_foreground_count = sum(
        cell != target_background for row in target for cell in row
    )
    return {
        "added_color_count": len(target_foreground_palette - source_foreground_palette),
        "background_preserved": source_background == target_background,
        "changed_component_count": changed_component_count,
        "changed_pixel_count": changed_count,
        "colored_shape_relation": _multiset_relation(
            source_colored_shapes, target_colored_shapes
        ),
        "component_area_relation": _multiset_relation(source_areas, target_areas),
        "component_count_delta": len(target_objects) - len(source_objects),
        "component_count_relation": _relation(len(target_objects), len(source_objects)),
        "foreground_bbox_is_output_shape": _foreground_bbox_shape(
            source, source_background
        )
        == target_shape,
        "foreground_count_relation": _relation(
            target_foreground_count, source_foreground_count
        ),
        "foreground_palette_relation": _multiset_relation(
            Counter(source_foreground_palette), Counter(target_foreground_palette)
        ),
        "input_contains_output": _contains_subgrid(source, target),
        "object_bbox_is_output_shape": target_shape
        in {_component_bbox_shape(item) for item in source_objects},
        "output_contains_input": _contains_subgrid(target, source),
        "removed_color_count": len(source_foreground_palette - target_foreground_palette),
        "shape_delta": [
            target_shape[0] - source_shape[0],
            target_shape[1] - source_shape[1],
        ],
        "shape_relation": (
            "same"
            if source_shape == target_shape
            else "transpose"
            if source_shape == (target_shape[1], target_shape[0])
            else "different"
        ),
        "uncolored_shape_relation": _multiset_relation(source_shapes, target_shapes),
    }


def stable_demo_contract(train: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Keep only transition fields identical across all demonstrations."""

    if not train:
        raise VisualProviderGateError("structural contract requires demonstrations")
    signatures = tuple(
        transition_features(
            _validate_grid(pair["input"]),
            _validate_grid(pair["output"]),
        )
        for pair in train
    )
    return {
        key: signatures[0][key]
        for key in signatures[0]
        if all(signature[key] == signatures[0][key] for signature in signatures[1:])
    }


def compile_structure_certificate(
    *,
    task_id: str,
    query_index: int,
    contract: Mapping[str, object],
    signature: Mapping[str, object],
) -> dict[str, object] | None:
    """Compile stable-contract violations into one legal representation action."""

    violations = tuple(
        sorted(key for key, value in contract.items() if signature[key] != value)
    )
    if not violations:
        return None
    violation_set = frozenset(violations)
    if violation_set & CANVAS_FEATURES:
        diagnosis = "visual_canvas_contract_mismatch"
        action = "canvas_reinfer"
    elif violation_set & BACKGROUND_FEATURES:
        diagnosis = "visual_background_role_mismatch"
        action = "reparse_background"
    elif violation_set & OBJECT_FEATURES:
        diagnosis = "visual_object_transition_mismatch"
        action = "object_rematch"
    else:
        raise VisualProviderGateError("unmapped structural-contract violation")
    content = {
        "schema": STRUCTURE_CERTIFICATE_SCHEMA,
        "task_id": task_id,
        "query_index": query_index,
        "diagnosis": diagnosis,
        "recommended_action": action,
        "affected_slots": list(ACTION_SLOTS[action]),
        "violated_features": list(violations),
        "cross_representation": True,
    }
    return {"certificate_id": canonical_sha256(content), **content}


def _rank_with_structure(
    *,
    source: Grid,
    samples: Sequence[object],
    contract: Mapping[str, object],
) -> tuple[tuple[Grid, int, int, int, dict[str, object]], ...]:
    normalized = tuple(_validate_grid(sample) for sample in samples)
    counts = Counter(normalized)
    first_index: dict[Grid, int] = {}
    for index, grid in enumerate(normalized):
        if grid not in first_index:
            first_index[grid] = index
    scored = []
    for candidate, count in counts.items():
        signature = transition_features(source, candidate)
        score = sum(signature[key] == value for key, value in contract.items())
        scored.append((candidate, count, first_index[candidate], score, signature))
    return tuple(
        sorted(
            scored,
            key=lambda item: (
                -item[3],
                -item[1],
                item[2],
                json.dumps(item[4], sort_keys=True, separators=(",", ":")),
                _grid_json(item[0]),
            ),
        )
    )


def freeze_visual_structure_bridge(
    *,
    cohort_manifest: Path,
    blind_task_dir: Path,
    prediction_roots: Sequence[Path],
    frozen_predictions: Path,
    output_path: Path,
    invalid_candidate_policy: str,
) -> dict[str, object]:
    """Freeze structural candidates and certificates without reading query gold."""

    if output_path.exists():
        raise VisualProviderGateError(
            f"structure candidate artifact already exists: {output_path}"
        )
    manifest = _load_json_object(cohort_manifest)
    if manifest["schema"] != COHORT_SCHEMA:
        raise VisualProviderGateError("cohort schema mismatch")
    task_records = manifest["tasks"]
    if not isinstance(task_records, list):
        raise VisualProviderGateError("cohort task records are malformed")
    task_ids = tuple(record["task_id"] for record in task_records)
    query_counts = {
        record["task_id"]: record["query_count"] for record in task_records
    }
    raw_predictions, prediction_files = _load_provider_predictions(
        task_ids, prediction_roots
    )
    combined, prediction_validation = _validate_provider_predictions(
        raw_predictions,
        invalid_candidate_policy=invalid_candidate_policy,
    )
    frozen = _load_json_object(frozen_predictions)
    if frozen["schema"] != FROZEN_PREDICTIONS_SCHEMA:
        raise VisualProviderGateError("frozen-prediction schema mismatch")
    if frozen["cohort_id"] != manifest["cohort_id"]:
        raise VisualProviderGateError("frozen-prediction cohort mismatch")
    if frozen["raw_prediction_payload_sha256"] != canonical_sha256(raw_predictions):
        raise VisualProviderGateError("raw prediction payload differs from freeze")
    if frozen["validated_prediction_payload_sha256"] != canonical_sha256(combined):
        raise VisualProviderGateError("validated prediction payload differs from freeze")

    tasks: list[dict[str, object]] = []
    for task_id in task_ids:
        blind = _load_json_object(blind_task_dir / f"{task_id}.json")
        train = blind["train"]
        tests = blind["test"]
        if not isinstance(train, list) or not isinstance(tests, list):
            raise VisualProviderGateError(f"blind task is malformed: {task_id}")
        if len(tests) != query_counts[task_id]:
            raise VisualProviderGateError(f"blind query count mismatch: {task_id}")
        contract = stable_demo_contract(train)
        queries: list[dict[str, object]] = []
        for query_index, query in enumerate(tests):
            if not isinstance(query, Mapping):
                raise VisualProviderGateError(f"blind query is malformed: {task_id}")
            source = _validate_grid(query["input"])
            samples = combined[task_id][query_index]
            normalized = tuple(_validate_grid(sample) for sample in samples)
            counts = Counter(normalized)
            first_index: dict[Grid, int] = {}
            for index, grid in enumerate(normalized):
                if grid not in first_index:
                    first_index[grid] = index
            frequency_ranked = tuple(
                sorted(
                    counts,
                    key=lambda grid: (
                        -counts[grid],
                        first_index[grid],
                        _grid_json(grid),
                    ),
                )
            )
            if len(frequency_ranked) < 2:
                raise VisualProviderGateError(
                    f"structural pass@2 requires two unique candidates: {task_id}"
                )
            structural_ranked = _rank_with_structure(
                source=source,
                samples=samples,
                contract=contract,
            )
            top_1 = frequency_ranked[0]
            structural_second = next(
                item for item in structural_ranked if item[0] != top_1
            )
            top_1_signature = transition_features(source, top_1)
            certificate = compile_structure_certificate(
                task_id=task_id,
                query_index=query_index,
                contract=contract,
                signature=top_1_signature,
            )
            queries.append(
                {
                    "query_index": query_index,
                    "frequency_top_2": [
                        [list(row) for row in candidate]
                        for candidate in frequency_ranked[:2]
                    ],
                    "hybrid_top_2": [
                        [list(row) for row in top_1],
                        [list(row) for row in structural_second[0]],
                    ],
                    "frequency_top_1_sha256": canonical_sha256({"grid": top_1}),
                    "structural_second": {
                        "grid_sha256": canonical_sha256(
                            {"grid": structural_second[0]}
                        ),
                        "frequency": structural_second[1],
                        "first_emission_index": structural_second[2],
                        "contract_match_count": structural_second[3],
                        "contract_field_count": len(contract),
                    },
                    "certificate": certificate,
                    "native_cost": {
                        "candidate_signature_count": len(structural_ranked),
                        "feature_comparison_count": len(structural_ranked)
                        * len(contract),
                    },
                }
            )
        tasks.append(
            {
                "task_id": task_id,
                "stable_demo_contract": contract,
                "queries": queries,
            }
        )

    contract_by_task = {
        record["task_id"]: record["stable_demo_contract"] for record in tasks
    }
    rotated_contract = {
        task_id: contract_by_task[task_ids[(index + 1) % len(task_ids)]]
        for index, task_id in enumerate(task_ids)
    }
    task_by_id = {record["task_id"]: record for record in tasks}
    for task_id in task_ids:
        blind = _load_json_object(blind_task_dir / f"{task_id}.json")
        for query in task_by_id[task_id]["queries"]:
            query_index = query["query_index"]
            source = _validate_grid(blind["test"][query_index]["input"])
            top_1 = _validate_grid(query["hybrid_top_2"][0])
            shuffled = compile_structure_certificate(
                task_id=task_id,
                query_index=query_index,
                contract=rotated_contract[task_id],
                signature=transition_features(source, top_1),
            )
            original_action = (
                None
                if query["certificate"] is None
                else query["certificate"]["recommended_action"]
            )
            shuffled_action = (
                None if shuffled is None else shuffled["recommended_action"]
            )
            query["contract_shuffle"] = {
                "source_task_id": task_ids[
                    (task_ids.index(task_id) + 1) % len(task_ids)
                ],
                "recommended_action": shuffled_action,
                "action_changed": original_action != shuffled_action,
            }

    content = {
        "schema": STRUCTURE_CANDIDATE_SCHEMA,
        "cohort_id": manifest["cohort_id"],
        "frozen_prediction_id": frozen["frozen_prediction_id"],
        "invalid_candidate_policy": invalid_candidate_policy,
        "prediction_files": prediction_files,
        "prediction_validation": prediction_validation,
        "validated_prediction_payload_sha256": canonical_sha256(combined),
        "method": {
            "candidate_1": "unchanged frequency top-1",
            "candidate_2": "highest stable-demo-contract match excluding candidate 1",
            "object_connectivity": 4,
            "grid_synthesis": False,
        },
        "tasks": tasks,
    }
    candidate_id = canonical_sha256(content)
    artifact = {"candidate_id": candidate_id, **content}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_path, artifact)
    return artifact


def score_visual_structure_bridge(
    *,
    cohort_manifest: Path,
    gold_task_dir: Path,
    candidate_artifact: Path,
    visual_score_summary: Path,
    output_path: Path,
) -> dict[str, object]:
    """Score one frozen structural bridge after query-gold release."""

    if output_path.exists():
        raise VisualProviderGateError(
            f"structure score artifact already exists: {output_path}"
        )
    manifest = _load_json_object(cohort_manifest)
    candidates = _load_json_object(candidate_artifact)
    if manifest["schema"] != COHORT_SCHEMA:
        raise VisualProviderGateError("cohort schema mismatch")
    if candidates["schema"] != STRUCTURE_CANDIDATE_SCHEMA:
        raise VisualProviderGateError("structure candidate schema mismatch")
    if candidates["cohort_id"] != manifest["cohort_id"]:
        raise VisualProviderGateError("structure candidate cohort mismatch")
    candidate_content = dict(candidates)
    candidate_id = candidate_content.pop("candidate_id")
    if canonical_sha256(candidate_content) != candidate_id:
        raise VisualProviderGateError("structure candidate content ID mismatch")

    visual_score = _load_json_object(visual_score_summary)
    if visual_score["schema"] != SCORE_SCHEMA:
        raise VisualProviderGateError("visual score schema mismatch")
    if visual_score["cohort_id"] != manifest["cohort_id"]:
        raise VisualProviderGateError("visual score cohort mismatch")
    score_by_task = {
        record["task_id"]: record for record in visual_score["tasks"]
    }

    task_ids = tuple(record["task_id"] for record in manifest["tasks"])
    if set(score_by_task) != set(task_ids):
        raise VisualProviderGateError("visual score task IDs differ from cohort")
    candidate_by_task = {
        record["task_id"]: record for record in candidates["tasks"]
    }
    if set(candidate_by_task) != set(task_ids):
        raise VisualProviderGateError("structure candidate task IDs differ from cohort")
    tasks: list[dict[str, object]] = []
    for task_id in task_ids:
        gold = _load_json_object(gold_task_dir / f"{task_id}.json")
        gold_queries = gold["test"]
        candidate_queries = candidate_by_task[task_id]["queries"]
        score_queries = score_by_task[task_id]["queries"]
        if len(gold_queries) != len(candidate_queries):
            raise VisualProviderGateError(f"query count mismatch: {task_id}")
        if len(score_queries) != len(candidate_queries):
            raise VisualProviderGateError(f"visual score query mismatch: {task_id}")
        queries: list[dict[str, object]] = []
        for query_index, query in enumerate(candidate_queries):
            target = _validate_grid(gold_queries[query_index]["output"])
            frequency_top_2 = tuple(
                _validate_grid(candidate) for candidate in query["frequency_top_2"]
            )
            hybrid_top_2 = tuple(
                _validate_grid(candidate) for candidate in query["hybrid_top_2"]
            )
            if hybrid_top_2[0] != frequency_top_2[0]:
                raise VisualProviderGateError("hybrid candidate 1 changed frequency top-1")
            queries.append(
                {
                    "query_index": query_index,
                    "frequency_pass_at_1": target == frequency_top_2[0],
                    "frequency_pass_at_2": target in frequency_top_2,
                    "hybrid_pass_at_2": target in hybrid_top_2,
                    "hybrid_unique_recovery": (
                        target in hybrid_top_2 and target not in frequency_top_2
                    ),
                    "hybrid_regression": (
                        target in frequency_top_2 and target not in hybrid_top_2
                    ),
                    "raw_oracle_covered": score_queries[query_index][
                        "raw_oracle_covered"
                    ],
                    "contract_shuffle_action_changed": query["contract_shuffle"][
                        "action_changed"
                    ],
                }
            )
        tasks.append(
            {
                "task_id": task_id,
                "frequency_pass_at_2": all(
                    query["frequency_pass_at_2"] for query in queries
                ),
                "hybrid_pass_at_2": all(query["hybrid_pass_at_2"] for query in queries),
                "hybrid_unique_recovery": (
                    all(query["hybrid_pass_at_2"] for query in queries)
                    and not all(query["frequency_pass_at_2"] for query in queries)
                ),
                "hybrid_regression": (
                    all(query["frequency_pass_at_2"] for query in queries)
                    and not all(query["hybrid_pass_at_2"] for query in queries)
                ),
                "queries": queries,
            }
        )

    all_queries = [query for task in tasks for query in task["queries"]]
    metrics = {
        "task_count": len(tasks),
        "query_count": len(all_queries),
        "frequency_task_pass_at_2": sum(task["frequency_pass_at_2"] for task in tasks),
        "hybrid_task_pass_at_2": sum(task["hybrid_pass_at_2"] for task in tasks),
        "hybrid_unique_task_recovery": sum(
            task["hybrid_unique_recovery"] for task in tasks
        ),
        "hybrid_task_regression": sum(task["hybrid_regression"] for task in tasks),
        "frequency_query_pass_at_1": sum(
            query["frequency_pass_at_1"] for query in all_queries
        ),
        "frequency_query_pass_at_2": sum(
            query["frequency_pass_at_2"] for query in all_queries
        ),
        "hybrid_query_pass_at_2": sum(
            query["hybrid_pass_at_2"] for query in all_queries
        ),
        "hybrid_unique_query_recovery": sum(
            query["hybrid_unique_recovery"] for query in all_queries
        ),
        "hybrid_query_regression": sum(
            query["hybrid_regression"] for query in all_queries
        ),
        "contract_shuffle_action_change_count": sum(
            query["contract_shuffle_action_changed"] for query in all_queries
        ),
        "raw_oracle_covered_query_count": sum(
            query["raw_oracle_covered"] for query in all_queries
        ),
    }
    metrics["net_task_pass_at_2_delta"] = (
        metrics["hybrid_task_pass_at_2"] - metrics["frequency_task_pass_at_2"]
    )
    metrics["net_query_pass_at_2_delta"] = (
        metrics["hybrid_query_pass_at_2"] - metrics["frequency_query_pass_at_2"]
    )
    pass_gate = (
        metrics["hybrid_unique_task_recovery"] >= 1
        and metrics["net_task_pass_at_2_delta"] > 0
        and metrics["contract_shuffle_action_change_count"] >= 1
    )
    boundary = (
        not pass_gate
        and metrics["net_query_pass_at_2_delta"] > 0
        and metrics["contract_shuffle_action_change_count"] >= 1
    )
    adverse = (
        metrics["net_task_pass_at_2_delta"] < 0
        or metrics["net_query_pass_at_2_delta"] < 0
    )
    decision = {
        "classification": (
            "pass" if pass_gate else "adverse" if adverse else "boundary" if boundary else "null"
        ),
        "advance_to_lodo_typed_repair_gate": pass_gate,
        "controller_remains_frozen": True,
        "novel_candidate_coverage_tested": False,
        "typed_repair_utility_tested": False,
    }
    content = {
        "schema": STRUCTURE_SCORE_SCHEMA,
        "cohort_id": manifest["cohort_id"],
        "candidate_id": candidate_id,
        "metrics": metrics,
        "decision": decision,
        "tasks": tasks,
    }
    result_id = canonical_sha256(content)
    result = {"result_id": result_id, **content}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_path, result)
    return result
