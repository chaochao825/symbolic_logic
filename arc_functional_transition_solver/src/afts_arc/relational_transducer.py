"""Demo-induced D4 correspondence and aggregate-color transducer programs."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction

from afts_arc.blind import BlindTask
from afts_arc.experiment_safety import canonical_sha256
from afts_arc.grid import Grid, as_grid


RELATIONAL_TRANSDUCER_DSL_VERSION = "afts-relational-transducer-dsl/v1"
RELATIONAL_TRANSDUCER_CERTIFICATE_SCHEMA = (
    "afts.visual-relational-transducer-certificate/v1"
)
COORDINATE_MODES = (
    "identity",
    "rotate_90_clockwise",
    "rotate_180",
    "rotate_270_clockwise",
    "reflect_vertical_axis",
    "reflect_horizontal_axis",
    "transpose_main",
    "transpose_anti",
)
COLOR_MODES = (
    "global_homomorphism",
    "row_foreground_count_offset",
    "column_foreground_count_offset",
)
STRUCTURAL_HYPOTHESIS_COUNT = len(COORDINATE_MODES) * len(COLOR_MODES)
COLD_ALLOCATION_SEED = "visual-trace-repair-cold-v1-20260809"
VISUAL_ALLOCATION_NUMERATOR = 3
VISUAL_ALLOCATION_DENOMINATOR = 10


def _shape(grid: Grid) -> tuple[int, int]:
    return len(grid), len(grid[0])


def modal_background(grid: Grid) -> int:
    counts = Counter(value for row in grid for value in row)
    return min(counts, key=lambda color: (-counts[color], color))


def foreground_count(grid: Grid) -> int:
    background = modal_background(grid)
    return sum(value != background for row in grid for value in row)


def transform_coordinates(grid: Grid, mode: str) -> Grid:
    """Apply one typed D4 coordinate transform."""

    source = as_grid(grid)
    height, width = _shape(source)
    if mode == "identity":
        return source
    if mode == "rotate_90_clockwise":
        return tuple(
            tuple(source[height - 1 - row][column] for row in range(height))
            for column in range(width)
        )
    if mode == "rotate_180":
        return tuple(tuple(reversed(row)) for row in reversed(source))
    if mode == "rotate_270_clockwise":
        return tuple(
            tuple(source[row][width - 1 - column] for row in range(height))
            for column in range(width)
        )
    if mode == "reflect_vertical_axis":
        return tuple(tuple(reversed(row)) for row in source)
    if mode == "reflect_horizontal_axis":
        return tuple(reversed(source))
    if mode == "transpose_main":
        return tuple(
            tuple(source[row][column] for row in range(height))
            for column in range(width)
        )
    if mode == "transpose_anti":
        return tuple(
            tuple(
                source[height - 1 - row][width - 1 - column]
                for row in range(height)
            )
            for column in range(width)
        )
    raise ValueError("unknown relational transducer coordinate mode")


@dataclass(frozen=True, slots=True)
class RelationalTransducerProgram:
    coordinate_mode: str
    color_mode: str
    color_mapping: tuple[tuple[int, int], ...] = ()
    count_offset: int | None = None

    def __post_init__(self) -> None:
        if self.coordinate_mode not in COORDINATE_MODES:
            raise ValueError("unknown transducer coordinate mode")
        if self.color_mode not in COLOR_MODES:
            raise ValueError("unknown transducer color mode")
        if self.color_mode == "global_homomorphism":
            if self.count_offset is not None or not self.color_mapping:
                raise ValueError("global homomorphism requires only a color mapping")
            sources = [source for source, _ in self.color_mapping]
            if sources != sorted(set(sources)):
                raise ValueError("transducer color-map sources must be sorted and unique")
            if any(
                not 0 <= source <= 9 or not 0 <= target <= 9
                for source, target in self.color_mapping
            ):
                raise ValueError("transducer color mapping contains an invalid color")
        elif self.color_mapping or isinstance(self.count_offset, bool) or not isinstance(
            self.count_offset, int
        ):
            raise ValueError("aggregate color mode requires only an integer offset")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "dsl_version": RELATIONAL_TRANSDUCER_DSL_VERSION,
            "coordinate": {"op": "d4_correspond", "mode": self.coordinate_mode},
            "color": {
                "op": self.color_mode,
                "mapping": [list(pair) for pair in self.color_mapping],
                "count_offset": self.count_offset,
            },
        }

    @classmethod
    def from_json_dict(cls, payload: object) -> "RelationalTransducerProgram":
        if not isinstance(payload, Mapping) or set(payload) != {
            "dsl_version",
            "coordinate",
            "color",
        }:
            raise ValueError("transducer program fields differ")
        if payload["dsl_version"] != RELATIONAL_TRANSDUCER_DSL_VERSION:
            raise ValueError("unsupported transducer DSL version")
        coordinate = payload["coordinate"]
        color = payload["color"]
        if not isinstance(coordinate, Mapping) or set(coordinate) != {"op", "mode"}:
            raise ValueError("transducer coordinate node fields differ")
        if coordinate["op"] != "d4_correspond":
            raise ValueError("transducer coordinate node has the wrong operator")
        if not isinstance(color, Mapping) or set(color) != {
            "op",
            "mapping",
            "count_offset",
        }:
            raise ValueError("transducer color node fields differ")
        mapping = color["mapping"]
        if not isinstance(mapping, list) or any(
            not isinstance(pair, list) or len(pair) != 2 for pair in mapping
        ):
            raise TypeError("transducer color mapping must contain pairs")
        return cls(
            coordinate_mode=coordinate["mode"],
            color_mode=color["op"],
            color_mapping=tuple((pair[0], pair[1]) for pair in mapping),
            count_offset=color["count_offset"],
        )

    @property
    def program_id(self) -> str:
        return canonical_sha256(self.to_json_dict())


def execute_relational_transducer(
    program: RelationalTransducerProgram, grid: Grid
) -> Grid | None:
    transformed = transform_coordinates(as_grid(grid), program.coordinate_mode)
    background = modal_background(transformed)
    if program.color_mode == "global_homomorphism":
        mapping = dict(program.color_mapping)
        if any(value not in mapping for row in transformed for value in row):
            return None
        return tuple(tuple(mapping[value] for value in row) for row in transformed)

    offset = program.count_offset
    if offset is None:
        raise AssertionError("validated aggregate transducer has no offset")
    height, width = _shape(transformed)
    output = [list(row) for row in transformed]
    if program.color_mode == "row_foreground_count_offset":
        for row in range(height):
            count = sum(value != background for value in transformed[row])
            color = count + offset
            if count and not 0 <= color <= 9:
                return None
            for column in range(width):
                output[row][column] = (
                    background if transformed[row][column] == background else color
                )
    else:
        for column in range(width):
            count = sum(
                transformed[row][column] != background for row in range(height)
            )
            color = count + offset
            if count and not 0 <= color <= 9:
                return None
            for row in range(height):
                output[row][column] = (
                    background if transformed[row][column] == background else color
                )
    return as_grid(output)


def _global_mapping(
    pairs: Sequence[tuple[Grid, Grid]], coordinate_mode: str
) -> tuple[tuple[int, int], ...] | None:
    mapping: dict[int, int] = {}
    for source, target in pairs:
        transformed = transform_coordinates(source, coordinate_mode)
        if _shape(transformed) != _shape(target):
            return None
        for source_row, target_row in zip(transformed, target, strict=True):
            for source_color, target_color in zip(source_row, target_row, strict=True):
                if source_color in mapping and mapping[source_color] != target_color:
                    return None
                mapping[source_color] = target_color
    return tuple(sorted(mapping.items()))


def _aggregate_offset(
    pairs: Sequence[tuple[Grid, Grid]], coordinate_mode: str, *, axis: str
) -> int | None:
    offsets: set[int] = set()
    observed_foreground = False
    for source, target in pairs:
        transformed = transform_coordinates(source, coordinate_mode)
        if _shape(transformed) != _shape(target):
            return None
        background = modal_background(transformed)
        height, width = _shape(transformed)
        groups = (
            tuple(tuple((row, column) for column in range(width)) for row in range(height))
            if axis == "row"
            else tuple(
                tuple((row, column) for row in range(height))
                for column in range(width)
            )
        )
        for cells in groups:
            foreground = [
                (row, column)
                for row, column in cells
                if transformed[row][column] != background
            ]
            for row, column in cells:
                if transformed[row][column] == background and target[row][column] != background:
                    return None
            if not foreground:
                continue
            observed_foreground = True
            colors = {target[row][column] for row, column in foreground}
            if len(colors) != 1:
                return None
            offsets.add(next(iter(colors)) - len(foreground))
    if not observed_foreground or len(offsets) != 1:
        return None
    return next(iter(offsets))


def synthesize_relational_transducers(
    task: BlindTask,
) -> tuple[RelationalTransducerProgram, ...]:
    """Enumerate the fixed 24 hypotheses and retain demo-exact programs."""

    if not isinstance(task, BlindTask):
        raise TypeError("relational transducer synthesis requires BlindTask")
    pairs = tuple((pair.input, pair.output) for pair in task.train)
    programs: list[RelationalTransducerProgram] = []
    for coordinate_mode in COORDINATE_MODES:
        mapping = _global_mapping(pairs, coordinate_mode)
        if mapping is not None:
            programs.append(
                RelationalTransducerProgram(
                    coordinate_mode=coordinate_mode,
                    color_mode="global_homomorphism",
                    color_mapping=mapping,
                )
            )
        for color_mode, axis in (
            ("row_foreground_count_offset", "row"),
            ("column_foreground_count_offset", "column"),
        ):
            offset = _aggregate_offset(pairs, coordinate_mode, axis=axis)
            if offset is not None:
                programs.append(
                    RelationalTransducerProgram(
                        coordinate_mode=coordinate_mode,
                        color_mode=color_mode,
                        count_offset=offset,
                    )
                )
    exact = []
    for program in programs:
        outputs = tuple(
            execute_relational_transducer(program, pair.input) for pair in task.train
        )
        if all(
            output == pair.output
            for output, pair in zip(outputs, task.train, strict=True)
        ):
            exact.append(program)
    unique = {program.program_id: program for program in exact}
    return tuple(unique[key] for key in sorted(unique))


def build_visual_transducer_certificate(
    *,
    task_id: str,
    query_inputs: Sequence[Grid],
    posterior_queries: Sequence[Sequence[tuple[Grid, int]]],
    visual_unique_candidate_count: int,
    recursive_unique_candidate_count: int,
    exact_pool_overlap_count: int,
) -> dict[str, object]:
    """Compile foreground-cardinality preservation into a typed action score."""

    if len(query_inputs) != len(posterior_queries) or not query_inputs:
        raise ValueError("certificate query inputs and posterior views must align")
    query_rows = []
    ratios = []
    for query_index, (raw_input, weighted_samples) in enumerate(
        zip(query_inputs, posterior_queries, strict=True)
    ):
        query = as_grid(raw_input)
        if not weighted_samples:
            raise ValueError("certificate posterior view must not be empty")
        target_count = foreground_count(query)
        total_weight = 0
        preserving_weight = 0
        for raw_sample, weight in weighted_samples:
            if isinstance(weight, bool) or not isinstance(weight, int) or weight <= 0:
                raise ValueError("posterior sample weights must be positive integers")
            sample = as_grid(raw_sample)
            total_weight += weight
            if _shape(sample) == _shape(query) and foreground_count(sample) == target_count:
                preserving_weight += weight
        ratios.append(Fraction(preserving_weight, total_weight))
        query_rows.append(
            {
                "query_index": query_index,
                "posterior_sample_weight": total_weight,
                "preserving_sample_weight": preserving_weight,
            }
        )
    minimum_ratio = min(ratios)
    content: dict[str, object] = {
        "schema": RELATIONAL_TRANSDUCER_CERTIFICATE_SCHEMA,
        "task_id": task_id,
        "recommended_action": "synthesize_relational_transducer",
        "affected_slots": ["ast.coordinate", "ast.color"],
        "score": {
            "minimum_preserving_numerator": minimum_ratio.numerator,
            "minimum_preserving_denominator": minimum_ratio.denominator,
            "total_unique_candidate_count": visual_unique_candidate_count
            + recursive_unique_candidate_count,
            "exact_pool_overlap_count": exact_pool_overlap_count,
        },
        "queries": query_rows,
    }
    return {"certificate_id": canonical_sha256(content), **content}


def visual_allocation_key(certificate: Mapping[str, object]) -> tuple[object, ...]:
    score = certificate["score"]
    if not isinstance(score, Mapping):
        raise TypeError("certificate score must be an object")
    ratio = Fraction(
        score["minimum_preserving_numerator"],
        score["minimum_preserving_denominator"],
    )
    return (
        -ratio,
        -score["total_unique_candidate_count"],
        score["exact_pool_overlap_count"],
        certificate["task_id"],
    )


def allocation_count(task_count: int) -> int:
    if isinstance(task_count, bool) or not isinstance(task_count, int) or task_count < 1:
        raise ValueError("allocation task count must be a positive integer")
    return (
        task_count * VISUAL_ALLOCATION_NUMERATOR
        + VISUAL_ALLOCATION_DENOMINATOR
        - 1
    ) // VISUAL_ALLOCATION_DENOMINATOR


def cold_allocation_key(task_id: str) -> tuple[str, str]:
    digest = hashlib.sha256(
        f"{COLD_ALLOCATION_SEED}\0{task_id}".encode("utf-8")
    ).hexdigest()
    return digest, task_id
