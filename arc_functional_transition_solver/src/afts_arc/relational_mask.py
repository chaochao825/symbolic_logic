"""Opt-in relational-mask programs for candidate-conditioned local marking.

This module is deliberately not part of the default object/code enumeration.
It adds a versioned representation without changing legacy provider order,
candidate identities, checkpoints, or previously reported numerical results.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations

from .blind import BlindTask
from .grid import Grid, as_grid
from .hybrid.scene_graph import ExecutionTraceNode
from .hybrid.types import canonical_json
from .residual import compare_grids


RELATIONAL_MASK_DSL_VERSION = "afts-relational-mask-dsl/v0.1"
RELATIONAL_MASK_PROVIDER_VERSION = "afts-relational-mask-provider/v0.1"
RELATIONAL_MASK_CERTIFICATE_SCHEMA = "afts.relational-mask-certificate/v0.1"
RELATIONAL_MASK_DIRECTION_SETS: dict[
    str, tuple[tuple[int, int], ...]
] = {
    "cardinal4": ((-1, 0), (0, -1), (0, 1), (1, 0)),
    "diagonal4": ((-1, -1), (-1, 1), (1, -1), (1, 1)),
}
RELATIONAL_MASK_CANVAS_MODES = ("copy", "blank")
RELATIONAL_MASK_MAX_RADIUS = 4

Coordinate = tuple[int, int]


def _arc_color(value: object, *, field_name: str) -> int:
    if type(value) is not int or not 0 <= value <= 9:
        raise ValueError(f"{field_name} must be an ARC color")
    return value


@dataclass(frozen=True, slots=True)
class RelationalMaskProgram:
    """Mark centers jointly supported along a typed set of relative rays."""

    background: int
    support_color: int
    output_color: int
    direction_set: str
    radii: tuple[int, ...]
    canvas_mode: str = "copy"

    def __post_init__(self) -> None:
        colors = (
            _arc_color(self.background, field_name="background"),
            _arc_color(self.support_color, field_name="support color"),
            _arc_color(self.output_color, field_name="output color"),
        )
        if len(set(colors)) != len(colors):
            raise ValueError("relational-mask colors must denote distinct roles")
        if self.direction_set not in RELATIONAL_MASK_DIRECTION_SETS:
            raise ValueError("unknown relational-mask direction set")
        if (
            not 1 <= len(self.radii) <= 2
            or tuple(sorted(set(self.radii))) != self.radii
            or any(
                type(radius) is not int
                or not 1 <= radius <= RELATIONAL_MASK_MAX_RADIUS
                for radius in self.radii
            )
        ):
            raise ValueError("relational-mask radii must be one or two sorted values")
        if self.canvas_mode not in RELATIONAL_MASK_CANVAS_MODES:
            raise ValueError("unknown relational-mask canvas mode")

    @property
    def description_bits(self) -> int:
        return 16 + 12 + len(self.radii) * math.ceil(
            math.log2(RELATIONAL_MASK_MAX_RADIUS)
        )

    @property
    def functional_trace(self) -> tuple[str, ...]:
        radii = ",".join(str(radius) for radius in self.radii)
        return (
            f"node:parse:background={self.background}",
            (
                "node:relate:all_direction_rays:"
                f"directions={self.direction_set}:radii={radii}:"
                f"support={self.support_color}"
            ),
            f"node:mask:center_background={self.background}",
            (
                "node:render:relational_marker:"
                f"color={self.output_color}:canvas={self.canvas_mode}"
            ),
        )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "relational_mask_dsl_version": RELATIONAL_MASK_DSL_VERSION,
            "kind": "relational_marker",
            "background": self.background,
            "support_color": self.support_color,
            "output_color": self.output_color,
            "direction_set": self.direction_set,
            "radii": list(self.radii),
            "canvas_mode": self.canvas_mode,
        }

    @classmethod
    def from_json_dict(cls, payload: object) -> "RelationalMaskProgram":
        expected = {
            "relational_mask_dsl_version",
            "kind",
            "background",
            "support_color",
            "output_color",
            "direction_set",
            "radii",
            "canvas_mode",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("relational-mask program has missing or unknown fields")
        if payload["relational_mask_dsl_version"] != RELATIONAL_MASK_DSL_VERSION:
            raise ValueError("unsupported relational-mask DSL version")
        if payload["kind"] != "relational_marker":
            raise ValueError("program kind is not relational_marker")
        radii = payload["radii"]
        if not isinstance(radii, list):
            raise TypeError("relational-mask radii must be a list")
        return cls(
            payload["background"],
            payload["support_color"],
            payload["output_color"],
            payload["direction_set"],
            tuple(radii),
            payload["canvas_mode"],
        )


def relational_mask_program_id(program: RelationalMaskProgram) -> str:
    return hashlib.sha256(
        canonical_json(program.to_json_dict()).encode("ascii")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class RelationalMaskExecution:
    status: str
    output: Grid | None
    reason: str | None
    mask: tuple[Coordinate, ...]
    node_trace: tuple[ExecutionTraceNode, ...]

    def __post_init__(self) -> None:
        if self.status not in {"ok", "invalid"}:
            raise ValueError("relational-mask status must be ok or invalid")
        if (self.status == "ok") != (self.output is not None):
            raise ValueError("only successful relational-mask runs carry output")
        if tuple(sorted(set(self.mask))) != self.mask:
            raise ValueError("relational-mask coordinates must be unique and sorted")

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def _relation_mask(program: RelationalMaskProgram, grid: Grid) -> tuple[Coordinate, ...]:
    height, width = len(grid), len(grid[0])
    directions = RELATIONAL_MASK_DIRECTION_SETS[program.direction_set]
    return tuple(
        (row, column)
        for row in range(height)
        for column in range(width)
        if grid[row][column] == program.background
        and all(
            0 <= row + dr * radius < height
            and 0 <= column + dc * radius < width
            and grid[row + dr * radius][column + dc * radius]
            == program.support_color
            for dr, dc in directions
            for radius in program.radii
        )
    )


def execute_relational_mask(
    program: RelationalMaskProgram,
    grid: Grid,
) -> RelationalMaskExecution:
    normalized = as_grid(grid)
    mask = _relation_mask(program, normalized)
    trace = (
        ExecutionTraceNode.create(
            "parse",
            "parse_background",
            "ok",
            background=program.background,
        ),
        ExecutionTraceNode.create(
            "relate",
            "all_direction_rays",
            "ok" if mask else "invalid",
            direction_set=program.direction_set,
            mask_count=len(mask),
            radii=list(program.radii),
            support_color=program.support_color,
        ),
    )
    if not mask:
        return RelationalMaskExecution(
            "invalid",
            None,
            "no_relational_centers",
            (),
            trace,
        )
    if program.canvas_mode == "copy":
        canvas = [list(row) for row in normalized]
    else:
        canvas = [
            [program.background for _ in range(len(normalized[0]))]
            for _ in range(len(normalized))
        ]
    for row, column in mask:
        canvas[row][column] = program.output_color
    output = as_grid(canvas)
    return RelationalMaskExecution(
        "ok",
        output,
        None,
        mask,
        (
            *trace,
            ExecutionTraceNode.create(
                "mask",
                "relation_to_mask",
                "ok",
                selected_count=len(mask),
            ),
            ExecutionTraceNode.create(
                "render",
                "relational_marker",
                "ok",
                canvas_mode=program.canvas_mode,
                output_color=program.output_color,
            ),
        ),
    )


def _modal_color(grid: Grid) -> int:
    counts = Counter(cell for row in grid for cell in row)
    return min(counts, key=lambda color: (-counts[color], color))


def _observable_counts(task: BlindTask) -> tuple[Counter[int], Counter[int]]:
    input_counts: Counter[int] = Counter()
    output_counts: Counter[int] = Counter()
    for pair in task.train:
        input_counts.update(cell for row in pair.input for cell in row)
        assert pair.output is not None
        output_counts.update(cell for row in pair.output for cell in row)
    for grid in task.test_inputs:
        input_counts.update(cell for row in grid for cell in row)
    return input_counts, output_counts


def enumerate_relational_mask_programs(
    task: BlindTask,
) -> tuple[RelationalMaskProgram, ...]:
    """Enumerate a bounded grammar using demonstrations and query inputs only."""

    if not isinstance(task, BlindTask):
        raise TypeError("relational-mask synthesis accepts BlindTask only")
    input_counts, output_counts = _observable_counts(task)
    modal_counts = Counter(
        _modal_color(grid)
        for grid in (
            *(pair.input for pair in task.train),
            *task.test_inputs,
        )
    )
    backgrounds = tuple(
        sorted(
            input_counts,
            key=lambda color: (-modal_counts[color], -input_counts[color], color),
        )[:2]
    )
    output_delta = {
        color: output_counts[color]
        - sum(
            Counter(cell for row in pair.input for cell in row)[color]
            for pair in task.train
        )
        for color in output_counts
    }
    output_colors = tuple(
        sorted(
            output_counts,
            key=lambda color: (-output_delta[color], -output_counts[color], color),
        )[:4]
    )
    radii = tuple(
        (radius,)
        for radius in range(1, RELATIONAL_MASK_MAX_RADIUS + 1)
    ) + tuple(combinations(range(1, RELATIONAL_MASK_MAX_RADIUS + 1), 2))
    programs = []
    for background in backgrounds:
        support_colors = tuple(
            sorted(
                (color for color in input_counts if color != background),
                key=lambda color: (-input_counts[color], color),
            )[:4]
        )
        for support_color in support_colors:
            for output_color in output_colors:
                if output_color in {background, support_color}:
                    continue
                for direction_set in RELATIONAL_MASK_DIRECTION_SETS:
                    for radius_set in radii:
                        for canvas_mode in RELATIONAL_MASK_CANVAS_MODES:
                            programs.append(
                                RelationalMaskProgram(
                                    background,
                                    support_color,
                                    output_color,
                                    direction_set,
                                    tuple(radius_set),
                                    canvas_mode,
                                )
                            )
    unique = {
        canonical_json(program.to_json_dict()): program for program in programs
    }
    return tuple(unique[key] for key in sorted(unique))


@dataclass(frozen=True, slots=True)
class RelationalMaskProgramScore:
    program: RelationalMaskProgram
    demo_outputs: tuple[Grid | None, ...]
    exact_demo_count: int
    shape_match_count: int
    agreement: float
    mismatch_count: int
    execution_valid: bool

    @property
    def all_demo_exact(self) -> bool:
        return self.execution_valid and self.exact_demo_count == len(self.demo_outputs)


def score_relational_mask_program(
    program: RelationalMaskProgram,
    task: BlindTask,
) -> RelationalMaskProgramScore:
    outputs = []
    exact_count = 0
    shape_count = 0
    matches = 0
    cells = 0
    mismatches = 0
    for index, pair in enumerate(task.train):
        execution = execute_relational_mask(program, pair.input)
        output = execution.output if execution.ok else None
        outputs.append(output)
        residual = compare_grids(
            output,
            pair.output,
            pair_index=index,
            invalid_code=execution.reason,
        )
        exact_count += int(residual.exact)
        shape_count += int(residual.shape_match)
        matches += residual.overlap_matches
        cells += residual.comparison_cells
        mismatches += residual.mismatch_count
    return RelationalMaskProgramScore(
        program,
        tuple(outputs),
        exact_count,
        shape_count,
        matches / cells if cells else 0.0,
        mismatches,
        all(output is not None for output in outputs),
    )


RELATIONAL_MASK_DIAGNOSIS_SLOTS: dict[str, tuple[str, ...]] = {
    "relation_geometry": (
        "relation.background",
        "relation.direction_set",
        "relation.radii",
        "relation.support_color",
    ),
    "render_canvas": ("render.canvas_mode",),
    "render_palette": ("render.output_color",),
}


@dataclass(frozen=True, slots=True)
class RelationalMaskFailureCertificate:
    certificate_id: str
    task_id: str
    parent_program_id: str
    diagnosis: str
    action: str
    affected_slots: tuple[str, ...]
    evidence: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.diagnosis not in RELATIONAL_MASK_DIAGNOSIS_SLOTS:
            raise ValueError("unknown relational-mask diagnosis")
        if self.action != "fill_relational_mask_slot":
            raise ValueError("unknown relational-mask certificate action")
        if self.affected_slots != RELATIONAL_MASK_DIAGNOSIS_SLOTS[self.diagnosis]:
            raise ValueError("certificate slots do not match its diagnosis")
        if self.certificate_id != self._content_id():
            raise ValueError("relational-mask certificate ID does not match content")

    def _content(self) -> dict[str, object]:
        return {
            "schema": RELATIONAL_MASK_CERTIFICATE_SCHEMA,
            "task_id": self.task_id,
            "parent_program_id": self.parent_program_id,
            "diagnosis": self.diagnosis,
            "action": self.action,
            "affected_slots": list(self.affected_slots),
            "evidence": dict(self.evidence),
        }

    def _content_id(self) -> str:
        return hashlib.sha256(
            canonical_json(self._content()).encode("ascii")
        ).hexdigest()

    def to_json_dict(self) -> dict[str, object]:
        return {"certificate_id": self.certificate_id, **self._content()}


def _delta_coordinates(first: Grid, second: Grid) -> tuple[Coordinate, ...]:
    if (len(first), len(first[0])) != (len(second), len(second[0])):
        return ()
    return tuple(
        (row, column)
        for row in range(len(first))
        for column in range(len(first[0]))
        if first[row][column] != second[row][column]
    )


def compile_relational_mask_failure_certificate(
    *,
    task_id: str,
    task: BlindTask,
    parent: RelationalMaskProgram,
) -> RelationalMaskFailureCertificate:
    """Compile visible demonstration failures into legal existing-slot edits."""

    executions = tuple(
        execute_relational_mask(parent, pair.input) for pair in task.train
    )
    rows = []
    for index, (pair, execution) in enumerate(
        zip(task.train, executions, strict=True)
    ):
        assert pair.output is not None
        output = execution.output
        rows.append(
            {
                "demo_index": index,
                "execution_status": execution.status,
                "execution_reason": execution.reason,
                "parent_delta": (
                    []
                    if output is None
                    else [list(item) for item in _delta_coordinates(pair.input, output)]
                ),
                "gold_delta": [
                    list(item) for item in _delta_coordinates(pair.input, pair.output)
                ],
                "parent_mask": [list(item) for item in execution.mask],
            }
        )
    if any(not execution.ok for execution in executions):
        diagnosis = "relation_geometry"
    elif parent.canvas_mode == "blank" and any(
        row["parent_delta"] != row["gold_delta"] for row in rows
    ):
        diagnosis = "render_canvas"
    elif all(
        row["parent_delta"] == row["gold_delta"] and row["gold_delta"]
        for row in rows
    ):
        diagnosis = "render_palette"
    else:
        diagnosis = "relation_geometry"
    slots = RELATIONAL_MASK_DIAGNOSIS_SLOTS[diagnosis]
    evidence: dict[str, object] = {
        "demo_count": len(rows),
        "demo_residuals": rows,
        "query_gold_read": False,
        "existing_slots_only": True,
    }
    content = {
        "schema": RELATIONAL_MASK_CERTIFICATE_SCHEMA,
        "task_id": task_id,
        "parent_program_id": relational_mask_program_id(parent),
        "diagnosis": diagnosis,
        "action": "fill_relational_mask_slot",
        "affected_slots": list(slots),
        "evidence": evidence,
    }
    certificate_id = hashlib.sha256(
        canonical_json(content).encode("ascii")
    ).hexdigest()
    return RelationalMaskFailureCertificate(
        certificate_id,
        task_id,
        content["parent_program_id"],
        diagnosis,
        content["action"],
        slots,
        evidence,
    )


def relational_mask_program_fields(
    program: RelationalMaskProgram,
) -> dict[str, object]:
    return {
        "relation.background": program.background,
        "relation.direction_set": program.direction_set,
        "relation.radii": program.radii,
        "relation.support_color": program.support_color,
        "render.canvas_mode": program.canvas_mode,
        "render.output_color": program.output_color,
    }


@dataclass(frozen=True, slots=True)
class RelationalMaskSlotEdit:
    slot: str
    program: RelationalMaskProgram


def single_slot_relational_mask_variants(
    task: BlindTask,
    parent: RelationalMaskProgram,
    *,
    allowed_slots: Sequence[str],
    existing_program_ids: Sequence[str] = (),
    candidate_programs: Sequence[RelationalMaskProgram] | None = None,
) -> tuple[RelationalMaskSlotEdit, ...]:
    """Return content-novel variants that change exactly one typed leaf slot."""

    allowed = frozenset(allowed_slots)
    parent_fields = relational_mask_program_fields(parent)
    if not allowed or any(slot not in parent_fields for slot in allowed):
        raise ValueError("allowed slots must name existing relational-mask fields")
    existing = frozenset(existing_program_ids)
    programs = (
        enumerate_relational_mask_programs(task)
        if candidate_programs is None
        else tuple(candidate_programs)
    )
    edits = []
    for program in programs:
        program_id = relational_mask_program_id(program)
        if program_id in existing:
            continue
        fields = relational_mask_program_fields(program)
        differences = tuple(
            sorted(
                field
                for field in parent_fields
                if parent_fields[field] != fields[field]
            )
        )
        if len(differences) == 1 and differences[0] in allowed:
            edits.append(RelationalMaskSlotEdit(differences[0], program))
    return tuple(
        sorted(
            edits,
            key=lambda edit: (
                edit.slot,
                canonical_json(edit.program.to_json_dict()),
            ),
        )
    )
