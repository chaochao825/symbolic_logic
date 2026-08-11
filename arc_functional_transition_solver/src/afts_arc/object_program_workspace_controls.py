"""Deterministic semantic controls for Object–Program Workspace v1."""

from __future__ import annotations

from afts_arc.experiment_safety import canonical_sha256


OBJECT_PROGRAM_WORKSPACE_CONTROL_SCHEMA = (
    "afts.object-program-workspace-controls/v1"
)

Coordinate = tuple[int, int]


def _transform_cell(
    cell: Coordinate,
    *,
    height: int,
    width: int,
    mode: int,
) -> Coordinate:
    row, column = cell
    if mode == 0:
        return row, column
    if mode == 1:
        return row, width - 1 - column
    if mode == 2:
        return height - 1 - row, column
    raise ValueError("control transform mode is invalid")


def _transform_cells(
    cells: tuple[Coordinate, ...],
    *,
    height: int,
    width: int,
    mode: int,
) -> tuple[Coordinate, ...]:
    return tuple(
        sorted(
            _transform_cell(cell, height=height, width=width, mode=mode)
            for cell in cells
        )
    )


def _paint(
    height: int,
    width: int,
    objects: tuple[tuple[int, tuple[Coordinate, ...]], ...],
) -> list[list[int]]:
    grid = [[0 for _ in range(width)] for _ in range(height)]
    for color, cells in objects:
        for row, column in cells:
            if grid[row][column] != 0:
                raise ValueError("controlled objects overlap")
            grid[row][column] = color
    return grid


def _apply_effect(
    grid: list[list[int]],
    cells: tuple[Coordinate, ...],
    *,
    operation: str,
    target_color: int,
) -> list[list[int]]:
    output = [row[:] for row in grid]
    color = 0 if operation == "erase_component" else target_color
    for row, column in cells:
        output[row][column] = color
    return output


def build_object_program_workspace_controls(
    *,
    task_count: int,
) -> dict[str, object]:
    """Build correspondence faults with declared whole-grid posterior mass."""

    if type(task_count) is not int or not 1 <= task_count <= 100:
        raise ValueError("controlled task count must be between one and 100")
    challenges: dict[str, object] = {}
    solutions: dict[str, object] = {}
    visual_tasks = []
    case_rows = []
    palette = (1, 2, 3, 4, 5, 6, 7, 8, 9)
    for index in range(task_count):
        mode = index % 3
        operation = (
            "recolor_component" if index % 2 == 0 else "erase_component"
        )
        offset = (index * 4) % len(palette)
        actor_a = palette[offset]
        actor_b = palette[(offset + 1) % len(palette)]
        border_color = palette[(offset + 2) % len(palette)]
        target_color = palette[(offset + 3) % len(palette)]

        first_height = first_width = 9
        first_interior = (
            (2, 2),
            (3, 2),
            (3, 3),
            (5, 5),
        )
        first_objects = (
            (actor_a, ((2, 2), (3, 2), (3, 3))),
            (actor_b, ((5, 5),)),
            (border_color, ((0, 1), (0, 2))),
            (border_color, ((8, 6), (8, 7))),
        )
        first_objects = tuple(
            (
                color,
                _transform_cells(
                    cells,
                    height=first_height,
                    width=first_width,
                    mode=mode,
                ),
            )
            for color, cells in first_objects
        )
        first_interior = _transform_cells(
            first_interior,
            height=first_height,
            width=first_width,
            mode=mode,
        )
        first = _paint(first_height, first_width, first_objects)
        first_gold = _apply_effect(
            first,
            first_interior,
            operation=operation,
            target_color=target_color,
        )

        second_height = second_width = 10
        second_interior = (
            (2, 2),
            (2, 3),
            (3, 2),
            (3, 3),
            (6, 6),
            (7, 6),
        )
        second_objects = (
            (actor_a, ((2, 2), (2, 3), (3, 2), (3, 3))),
            (actor_b, ((6, 6), (7, 6))),
            (border_color, ((0, 1),)),
            (border_color, ((0, 5),)),
            (border_color, ((9, 8),)),
        )
        second_objects = tuple(
            (
                color,
                _transform_cells(
                    cells,
                    height=second_height,
                    width=second_width,
                    mode=mode,
                ),
            )
            for color, cells in second_objects
        )
        second_interior = _transform_cells(
            second_interior,
            height=second_height,
            width=second_width,
            mode=mode,
        )
        second = _paint(second_height, second_width, second_objects)
        second_gold = _apply_effect(
            second,
            second_interior,
            operation=operation,
            target_color=target_color,
        )

        query_height = query_width = 9
        query_interior = ((3, 3), (5, 5))
        query_border = ((0, 1), (0, 2))
        query_objects = (
            (actor_a, ((3, 3),)),
            (actor_b, ((5, 5),)),
            (border_color, query_border),
        )
        query_objects = tuple(
            (
                color,
                _transform_cells(
                    cells,
                    height=query_height,
                    width=query_width,
                    mode=mode,
                ),
            )
            for color, cells in query_objects
        )
        query_interior = _transform_cells(
            query_interior,
            height=query_height,
            width=query_width,
            mode=mode,
        )
        query_border = _transform_cells(
            query_border,
            height=query_height,
            width=query_width,
            mode=mode,
        )
        query = _paint(query_height, query_width, query_objects)
        query_gold = _apply_effect(
            query,
            query_interior,
            operation=operation,
            target_color=target_color,
        )
        query_distractor = _apply_effect(
            query,
            query_border,
            operation=operation,
            target_color=target_color,
        )

        task_id = f"object_workspace_control_{index:03d}"
        challenges[task_id] = {
            "train": [
                {"input": first, "output": first_gold},
                {"input": second, "output": second_gold},
            ],
            "test": [{"input": query}],
        }
        solutions[task_id] = [query_gold]
        visual_tasks.append(
            {
                "task_id": task_id,
                "queries": [
                    {
                        "candidates": [
                            {"output": query_gold, "sample_count": 8},
                            {"output": query_distractor, "sample_count": 2},
                        ]
                    }
                ],
            }
        )
        case_rows.append(
            {
                "task_id": task_id,
                "operation": operation,
                "intended_selector": "interior",
                "posterior_exact_weight": 8,
                "posterior_distractor_weight": 2,
                "transform_mode": mode,
            }
        )
    visual_content: dict[str, object] = {
        "schema": "afts.controlled-whole-grid-posterior/v1",
        "query_gold_read": False,
        "construction": "declared latent program, not a learned visual model",
        "tasks": visual_tasks,
    }
    visual = {"freeze_id": canonical_sha256(visual_content), **visual_content}
    cohort_content: dict[str, object] = {
        "schema": OBJECT_PROGRAM_WORKSPACE_CONTROL_SCHEMA,
        "task_count": task_count,
        "challenges_sha256": canonical_sha256(challenges),
        "solutions_sha256": canonical_sha256(solutions),
        "visual_freeze_id": visual["freeze_id"],
        "cases": case_rows,
    }
    return {
        "manifest": {
            "cohort_id": canonical_sha256(cohort_content),
            **cohort_content,
        },
        "challenges": challenges,
        "solutions": solutions,
        "visual_freeze": visual,
    }
