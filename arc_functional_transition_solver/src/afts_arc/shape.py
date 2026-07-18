"""Blind, deterministic output-shape proposals for bounded ARC search."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum

from .blind import BlindTask
from .grid import ARC_MAX_HEIGHT, ARC_MAX_WIDTH, Grid

SHAPE_PROPOSER_SEMANTICS_VERSION = "afts-output-shape/v0.1"


class ShapeBasis(str, Enum):
    """Input dimension order used before applying multiplicative factors."""

    INPUT = "input"


@dataclass(frozen=True, slots=True)
class OutputShapeProposal:
    """A content-addressed shape rule and its blind query projections."""

    proposal_id: str
    basis: ShapeBasis
    row_factor: int
    column_factor: int
    query_shapes: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        try:
            basis = ShapeBasis(self.basis)
        except (TypeError, ValueError) as exc:
            raise ValueError("basis must be a valid ShapeBasis") from exc
        for name in ("row_factor", "column_factor"):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= max(
                ARC_MAX_HEIGHT, ARC_MAX_WIDTH
            ):
                raise ValueError(f"{name} must be an integer in [1, 30]")
        if not isinstance(self.query_shapes, (tuple, list)) or not self.query_shapes:
            raise ValueError("query_shapes must be a non-empty sequence")
        normalized_shapes: list[tuple[int, int]] = []
        for index, shape in enumerate(self.query_shapes):
            if not isinstance(shape, (tuple, list)) or len(shape) != 2:
                raise TypeError(f"query_shapes[{index}] must contain height and width")
            height, width = shape
            if (
                type(height) is not int
                or type(width) is not int
                or not 1 <= height <= ARC_MAX_HEIGHT
                or not 1 <= width <= ARC_MAX_WIDTH
            ):
                raise ValueError(f"query_shapes[{index}] exceeds ARC shape bounds")
            normalized_shapes.append((height, width))
        normalized = tuple(normalized_shapes)
        expected = _proposal_id(
            basis=basis,
            row_factor=self.row_factor,
            column_factor=self.column_factor,
            query_shapes=normalized,
        )
        if self.proposal_id != expected:
            raise ValueError("proposal_id does not match canonical shape content")
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "query_shapes", normalized)

    @classmethod
    def create(
        cls,
        *,
        basis: ShapeBasis,
        row_factor: int,
        column_factor: int,
        query_shapes: tuple[tuple[int, int], ...],
    ) -> "OutputShapeProposal":
        normalized_basis = ShapeBasis(basis)
        normalized_shapes = tuple(tuple(shape) for shape in query_shapes)
        return cls(
            proposal_id=_proposal_id(
                basis=normalized_basis,
                row_factor=row_factor,
                column_factor=column_factor,
                query_shapes=normalized_shapes,
            ),
            basis=normalized_basis,
            row_factor=row_factor,
            column_factor=column_factor,
            query_shapes=normalized_shapes,
        )

    @classmethod
    def from_json_dict(cls, payload: object) -> "OutputShapeProposal":
        if not isinstance(payload, dict):
            raise TypeError("output-shape proposal must be a JSON object")
        expected = {
            "shape_proposer_semantics_version",
            "proposal_id",
            "basis",
            "row_factor",
            "column_factor",
            "query_shapes",
        }
        if set(payload) != expected:
            raise ValueError(
                f"output-shape fields must be {sorted(expected)}, found {sorted(payload)}"
            )
        if payload["shape_proposer_semantics_version"] != SHAPE_PROPOSER_SEMANTICS_VERSION:
            raise ValueError("unsupported output-shape proposer semantics version")
        raw_shapes = payload["query_shapes"]
        if not isinstance(raw_shapes, list):
            raise TypeError("query_shapes must be a list")
        return cls(
            proposal_id=payload["proposal_id"],
            basis=payload["basis"],
            row_factor=payload["row_factor"],
            column_factor=payload["column_factor"],
            query_shapes=tuple(raw_shapes),
        )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "shape_proposer_semantics_version": SHAPE_PROPOSER_SEMANTICS_VERSION,
            "proposal_id": self.proposal_id,
            "basis": self.basis.value,
            "row_factor": self.row_factor,
            "column_factor": self.column_factor,
            "query_shapes": [list(shape) for shape in self.query_shapes],
        }


def _proposal_id(
    *,
    basis: ShapeBasis,
    row_factor: int,
    column_factor: int,
    query_shapes: tuple[tuple[int, int], ...],
) -> str:
    payload = {
        "shape_proposer_semantics_version": SHAPE_PROPOSER_SEMANTICS_VERSION,
        "basis": basis.value,
        "row_factor": row_factor,
        "column_factor": column_factor,
        "query_shapes": [list(shape) for shape in query_shapes],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def _base_shape(grid: Grid, basis: ShapeBasis) -> tuple[int, int]:
    if basis is not ShapeBasis.INPUT:
        raise AssertionError(f"unsupported shape basis: {basis}")
    return len(grid), len(grid[0])


def infer_output_shape_proposals(task: BlindTask) -> tuple[OutputShapeProposal, ...]:
    """Infer shape multipliers using demonstrations and query inputs only.

    A rule is emitted only when every demonstration has the same positive integer
    factors and all projected query shapes remain within the ARC 30-by-30 limit.
    """

    proposals: list[OutputShapeProposal] = []
    for basis in (ShapeBasis.INPUT,):
        factors: tuple[int, int] | None = None
        consistent = True
        for pair in task.train:
            if pair.output is None:
                raise ValueError("blind task training output is missing")
            base_height, base_width = _base_shape(pair.input, basis)
            output_height, output_width = len(pair.output), len(pair.output[0])
            if output_height % base_height or output_width % base_width:
                consistent = False
                break
            candidate = (output_height // base_height, output_width // base_width)
            if factors is None:
                factors = candidate
            elif factors != candidate:
                consistent = False
                break
        if not consistent or factors is None:
            continue
        row_factor, column_factor = factors
        query_shapes = tuple(
            (
                _base_shape(grid, basis)[0] * row_factor,
                _base_shape(grid, basis)[1] * column_factor,
            )
            for grid in task.test_inputs
        )
        if any(
            height > ARC_MAX_HEIGHT or width > ARC_MAX_WIDTH
            for height, width in query_shapes
        ):
            continue
        proposals.append(
            OutputShapeProposal.create(
                basis=basis,
                row_factor=row_factor,
                column_factor=column_factor,
                query_shapes=query_shapes,
            )
        )
    return tuple(proposals)
