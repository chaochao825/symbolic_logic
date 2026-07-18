"""Demonstration-only representation router for hybrid ARC solving."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from ..blind import BlindTask
from ..grid import Grid, dihedral_variants


ROUTES = (
    "dsl_program",
    "code_llm",
    "sparse_ca",
    "difflogic_hard",
    "masked_diffusion",
)


def _shape(grid: Grid) -> tuple[int, int]:
    return len(grid), len(grid[0])


def _modal_color(grid: Grid) -> int:
    counts = Counter(cell for row in grid for cell in row)
    return min(counts, key=lambda color: (-counts[color], color))


def _component_count(grid: Grid) -> int:
    background = _modal_color(grid)
    height, width = _shape(grid)
    seen: set[tuple[int, int]] = set()
    count = 0
    for row in range(height):
        for column in range(width):
            if grid[row][column] == background or (row, column) in seen:
                continue
            count += 1
            color = grid[row][column]
            frontier = [(row, column)]
            seen.add((row, column))
            while frontier:
                current_row, current_column = frontier.pop()
                for next_row, next_column in (
                    (current_row - 1, current_column),
                    (current_row + 1, current_column),
                    (current_row, current_column - 1),
                    (current_row, current_column + 1),
                ):
                    if (
                        0 <= next_row < height
                        and 0 <= next_column < width
                        and (next_row, next_column) not in seen
                        and grid[next_row][next_column] == color
                    ):
                        seen.add((next_row, next_column))
                        frontier.append((next_row, next_column))
    return count


@dataclass(frozen=True, slots=True)
class TaskFeatures:
    demonstration_count: int
    query_count: int
    all_same_shape: bool
    shape_change: bool
    mean_changed_fraction: float
    center_transition_support: float
    d4_consistent: bool
    object_relation_evidence: bool
    compressible_local_transition: bool
    open_hypothesis_needed: bool

    def to_json_dict(self) -> dict[str, object]:
        return {
            "demonstration_count": self.demonstration_count,
            "query_count": self.query_count,
            "all_same_shape": self.all_same_shape,
            "shape_change": self.shape_change,
            "mean_changed_fraction": self.mean_changed_fraction,
            "center_transition_support": self.center_transition_support,
            "d4_consistent": self.d4_consistent,
            "object_relation_evidence": self.object_relation_evidence,
            "compressible_local_transition": self.compressible_local_transition,
            "open_hypothesis_needed": self.open_hypothesis_needed,
        }


@dataclass(frozen=True, slots=True)
class RouteAssignment:
    route: str
    priority: int
    budget: int
    reasons: tuple[str, ...]

    def to_json_dict(self) -> dict[str, object]:
        return {
            "route": self.route,
            "priority": self.priority,
            "budget": self.budget,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True, slots=True)
class RouteDecision:
    assignments: tuple[RouteAssignment, ...]

    def __post_init__(self) -> None:
        routes = tuple(item.route for item in self.assignments)
        if set(routes) != set(ROUTES) or len(routes) != len(ROUTES):
            raise ValueError("route decision must contain every specialist exactly once")
        if tuple(sorted(self.assignments, key=lambda item: (item.priority, item.route))) != self.assignments:
            raise ValueError("route assignments must be canonically ordered")

    @property
    def route_order(self) -> tuple[str, ...]:
        return tuple(item.route for item in self.assignments)

    def priority_for(self, route: str) -> int:
        for item in self.assignments:
            if item.route == route:
                return item.priority
        return len(self.assignments)

    def budget_for(self, route: str) -> int:
        for item in self.assignments:
            if item.route == route:
                return item.budget
        return 0

    def to_json_dict(self) -> dict[str, object]:
        return {
            "route_order": list(self.route_order),
            "assignments": [item.to_json_dict() for item in self.assignments],
        }


def extract_task_features(task: BlindTask) -> TaskFeatures:
    """Extract only observable demonstration/query-input features."""

    if not isinstance(task, BlindTask):
        raise TypeError("router accepts BlindTask only")
    same_shape_flags = tuple(_shape(pair.input) == _shape(pair.output) for pair in task.train)
    all_same_shape = all(same_shape_flags)
    changed_fractions: list[float] = []
    transition_targets: dict[int, set[int]] = defaultdict(set)
    transition_counts: Counter[int] = Counter()
    total_transition_cells = 0
    component_change = False
    shared_d4_names: set[str] | None = None
    for pair, same_shape in zip(task.train, same_shape_flags):
        assert pair.output is not None
        component_change |= _component_count(pair.input) != _component_count(pair.output)
        transforms = {name for name, output in dihedral_variants(pair.input) if output == pair.output}
        shared_d4_names = transforms if shared_d4_names is None else shared_d4_names & transforms
        if not same_shape:
            continue
        cells = len(pair.input) * len(pair.input[0])
        changed = 0
        for source_row, target_row in zip(pair.input, pair.output):
            for source, target in zip(source_row, target_row):
                changed += source != target
                transition_targets[source].add(target)
                transition_counts[source] += 1
                total_transition_cells += 1
        changed_fractions.append(changed / cells)
    unambiguous_cells = sum(
        transition_counts[color]
        for color, targets in transition_targets.items()
        if len(targets) == 1
    )
    center_support = unambiguous_cells / total_transition_cells if total_transition_cells else 0.0
    mean_changed = sum(changed_fractions) / len(changed_fractions) if changed_fractions else 1.0
    shape_change = not all_same_shape
    d4_consistent = bool(shared_d4_names)
    object_relation = shape_change or component_change
    compressible = all_same_shape and center_support >= 0.8 and mean_changed <= 0.6
    open_needed = not (object_relation or d4_consistent or compressible)
    return TaskFeatures(
        demonstration_count=len(task.train),
        query_count=len(task.test_inputs),
        all_same_shape=all_same_shape,
        shape_change=shape_change,
        mean_changed_fraction=mean_changed,
        center_transition_support=center_support,
        d4_consistent=d4_consistent,
        object_relation_evidence=object_relation,
        compressible_local_transition=compressible,
        open_hypothesis_needed=open_needed,
    )


def route_task(task: BlindTask) -> tuple[TaskFeatures, RouteDecision]:
    features = extract_task_features(task)
    budgets = {
        "dsl_program": 128,
        "code_llm": 8,
        "sparse_ca": 16,
        "difflogic_hard": 8,
        "masked_diffusion": 32,
    }
    if features.object_relation_evidence:
        order = (
            "dsl_program",
            "code_llm",
            "masked_diffusion",
            "sparse_ca",
            "difflogic_hard",
        )
    elif features.all_same_shape and (
        features.compressible_local_transition or features.d4_consistent
    ):
        order = (
            "sparse_ca",
            "difflogic_hard",
            "dsl_program",
            "masked_diffusion",
            "code_llm",
        )
    else:
        order = (
            "code_llm",
            "masked_diffusion",
            "dsl_program",
            "sparse_ca",
            "difflogic_hard",
        )

    reasons_by_route = {
        "dsl_program": tuple(
            reason
            for reason, enabled in (
                ("shape_change", features.shape_change),
                ("object_or_relation_change", features.object_relation_evidence),
                ("d4_exact_on_demonstrations", features.d4_consistent),
                ("symbolic_fallback", True),
            )
            if enabled
        ),
        "code_llm": (
            ("open_hypothesis_needed",) if features.open_hypothesis_needed else ("open_hypothesis_fallback",)
        ),
        "sparse_ca": tuple(
            reason
            for reason, enabled in (
                ("same_shape", features.all_same_shape),
                ("high_center_transition_support", features.center_transition_support >= 0.8),
                ("local_change_fraction", features.mean_changed_fraction <= 0.6),
            )
            if enabled
        ) or ("shape_change_low_priority",),
        "difflogic_hard": (
            ("compressible_local_transition",)
            if features.compressible_local_transition
            else ("requires_verified_hard_circuit",)
        ),
        "masked_diffusion": ("broad_candidate_and_repair_source",),
    }
    assignments = tuple(
        RouteAssignment(route, priority, budgets[route], reasons_by_route[route])
        for priority, route in enumerate(order)
    )
    return features, RouteDecision(assignments)
