"""Bounded object-predicate DSL and provider for structured ARC deliberation.

This module is intentionally independent of the legacy sequential grid DSL.  A
scene rule first fixes a replayable object parse, then applies one typed selector
and one typed action.  Synthesis may read demonstration outputs, but replay and
controller decisions never receive query outputs.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..blind import BlindTask
from ..grid import Grid, as_grid, grid_key
from ..parse import ColorMode, ObjectView, background_hypotheses, parse_grid
from .router import RouteDecision, TaskFeatures
from .types import CandidateHypothesis, ProviderResult, canonical_json

if TYPE_CHECKING:
    from .control import Blackboard, ControlAction


SCENE_DSL_VERSION = "afts-scene-predicate-dsl/v0.1"
SCENE_PROVIDER_VERSION = "afts-hybrid-scene-predicate/v0.1"

SELECTORS = (
    "all",
    "largest_area",
    "smallest_area",
    "singleton",
    "has_hole",
    "no_hole",
    "filled_rectangle",
    "touches_border",
    "interior",
    "color",
    "area",
)
ACTIONS = (
    "keep",
    "erase",
    "crop",
    "recolor",
    "fill_bbox",
    "outline_bbox",
    "fill_bbox_background",
)
COLOR_ACTIONS = frozenset(
    {"recolor", "fill_bbox", "outline_bbox", "fill_bbox_background"}
)
ARG_SELECTORS = frozenset({"color", "area"})


@dataclass(frozen=True, slots=True)
class SceneRule:
    background: int
    connectivity: int
    color_mode: str
    selector: str
    selector_argument: int | None
    action: str
    action_color: int | None

    def __post_init__(self) -> None:
        if type(self.background) is not int or not 0 <= self.background <= 9:
            raise ValueError("scene background must be an ARC color")
        if self.connectivity not in {4, 8}:
            raise ValueError("scene connectivity must be 4 or 8")
        try:
            ColorMode(self.color_mode)
        except ValueError as exc:
            raise ValueError("unknown scene color mode") from exc
        if self.selector not in SELECTORS:
            raise ValueError("unknown scene selector")
        if self.action not in ACTIONS:
            raise ValueError("unknown scene action")
        if self.selector in ARG_SELECTORS:
            if type(self.selector_argument) is not int:
                raise ValueError("argument selector requires an integer argument")
            if self.selector == "color" and not 0 <= self.selector_argument <= 9:
                raise ValueError("color selector argument must be an ARC color")
            if self.selector == "area" and self.selector_argument < 1:
                raise ValueError("area selector argument must be positive")
        elif self.selector_argument is not None:
            raise ValueError("argument-free selector cannot carry an argument")
        if self.action in COLOR_ACTIONS:
            if type(self.action_color) is not int or not 0 <= self.action_color <= 9:
                raise ValueError("color action requires an ARC color")
        elif self.action_color is not None:
            raise ValueError("color-free action cannot carry a color")

    @classmethod
    def from_json_dict(cls, payload: object) -> "SceneRule":
        expected = {
            "scene_dsl_version",
            "background",
            "connectivity",
            "color_mode",
            "selector",
            "selector_argument",
            "action",
            "action_color",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("scene rule has missing or unknown fields")
        if payload["scene_dsl_version"] != SCENE_DSL_VERSION:
            raise ValueError("unsupported scene DSL version")
        return cls(
            background=payload["background"],
            connectivity=payload["connectivity"],
            color_mode=payload["color_mode"],
            selector=payload["selector"],
            selector_argument=payload["selector_argument"],
            action=payload["action"],
            action_color=payload["action_color"],
        )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "scene_dsl_version": SCENE_DSL_VERSION,
            "background": self.background,
            "connectivity": self.connectivity,
            "color_mode": self.color_mode,
            "selector": self.selector,
            "selector_argument": self.selector_argument,
            "action": self.action,
            "action_color": self.action_color,
        }

    @property
    def functional_trace(self) -> tuple[str, ...]:
        return (
            f"parse:objects:bg={self.background}:c={self.connectivity}:{self.color_mode}",
            f"predicate:{self.selector}",
            f"theorem:{self.action}",
        )

    @property
    def description_bits(self) -> int:
        selector_bits = max(1, math.ceil(math.log2(len(SELECTORS))))
        action_bits = max(1, math.ceil(math.log2(len(ACTIONS))))
        argument_bits = 0
        if self.selector_argument is not None:
            argument_bits = max(4, self.selector_argument.bit_length() + 1)
        return 4 + 1 + 1 + selector_bits + argument_bits + action_bits + (
            4 if self.action_color is not None else 0
        )


@dataclass(frozen=True, slots=True)
class SceneExecution:
    status: str
    output: Grid | None
    reason: str | None
    selected_object_count: int

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def _scene_objects(rule: SceneRule, grid: Grid) -> tuple[ObjectView, ...]:
    bundle = parse_grid(
        grid,
        backgrounds=(rule.background,),
        connectivities=(rule.connectivity,),
        color_modes=(ColorMode(rule.color_mode),),
        max_backgrounds=None,
    )
    if len(bundle.hypotheses) != 1:
        raise AssertionError("a fixed scene parse must produce exactly one hypothesis")
    return bundle.hypotheses[0].objects


def _select_objects(
    rule: SceneRule,
    objects: Sequence[ObjectView],
    *,
    height: int,
    width: int,
) -> tuple[ObjectView, ...]:
    if not objects:
        return ()
    if rule.selector == "all":
        selected = tuple(objects)
    elif rule.selector == "largest_area":
        target = max(item.area for item in objects)
        selected = tuple(item for item in objects if item.area == target)
    elif rule.selector == "smallest_area":
        target = min(item.area for item in objects)
        selected = tuple(item for item in objects if item.area == target)
    elif rule.selector == "singleton":
        selected = tuple(item for item in objects if item.area == 1)
    elif rule.selector == "has_hole":
        selected = tuple(item for item in objects if item.hole_count > 0)
    elif rule.selector == "no_hole":
        selected = tuple(item for item in objects if item.hole_count == 0)
    elif rule.selector == "filled_rectangle":
        selected = tuple(
            item
            for item in objects
            if item.area == item.bbox.height * item.bbox.width
        )
    elif rule.selector == "touches_border":
        selected = tuple(
            item
            for item in objects
            if item.bbox.top == 0
            or item.bbox.left == 0
            or item.bbox.bottom == height - 1
            or item.bbox.right == width - 1
        )
    elif rule.selector == "interior":
        selected = tuple(
            item
            for item in objects
            if item.bbox.top > 0
            and item.bbox.left > 0
            and item.bbox.bottom < height - 1
            and item.bbox.right < width - 1
        )
    elif rule.selector == "color":
        selected = tuple(
            item for item in objects if rule.selector_argument in item.colors
        )
    elif rule.selector == "area":
        selected = tuple(
            item for item in objects if item.area == rule.selector_argument
        )
    else:
        raise AssertionError(f"selector is not implemented: {rule.selector}")
    return tuple(sorted(selected, key=lambda item: item.object_id))


def _execute_with_objects(
    rule: SceneRule,
    grid: Grid,
    objects: Sequence[ObjectView],
) -> SceneExecution:
    normalized = as_grid(grid)
    height, width = len(normalized), len(normalized[0])
    selected = _select_objects(rule, objects, height=height, width=width)
    if not selected:
        return SceneExecution("invalid", None, "empty_selection", 0)
    if rule.action == "crop":
        if len(selected) != 1:
            return SceneExecution(
                "invalid", None, "crop_requires_unique_selection", len(selected)
            )
        box = selected[0].bbox
        output = tuple(
            tuple(row[box.left : box.right + 1])
            for row in normalized[box.top : box.bottom + 1]
        )
        return SceneExecution("ok", output, None, 1)

    if rule.action == "keep":
        canvas = [[rule.background for _ in range(width)] for _ in range(height)]
        for item in selected:
            for row, column in item.pixels:
                canvas[row][column] = normalized[row][column]
        return SceneExecution("ok", as_grid(canvas), None, len(selected))

    canvas = [list(row) for row in normalized]
    if rule.action == "erase":
        for item in selected:
            for row, column in item.pixels:
                canvas[row][column] = rule.background
    elif rule.action == "recolor":
        assert rule.action_color is not None
        for item in selected:
            for row, column in item.pixels:
                canvas[row][column] = rule.action_color
    elif rule.action == "fill_bbox":
        assert rule.action_color is not None
        for item in selected:
            for row in range(item.bbox.top, item.bbox.bottom + 1):
                for column in range(item.bbox.left, item.bbox.right + 1):
                    canvas[row][column] = rule.action_color
    elif rule.action == "outline_bbox":
        assert rule.action_color is not None
        for item in selected:
            for row in range(item.bbox.top, item.bbox.bottom + 1):
                for column in range(item.bbox.left, item.bbox.right + 1):
                    if row in {item.bbox.top, item.bbox.bottom} or column in {
                        item.bbox.left,
                        item.bbox.right,
                    }:
                        canvas[row][column] = rule.action_color
    elif rule.action == "fill_bbox_background":
        assert rule.action_color is not None
        for item in selected:
            for row in range(item.bbox.top, item.bbox.bottom + 1):
                for column in range(item.bbox.left, item.bbox.right + 1):
                    if canvas[row][column] == rule.background:
                        canvas[row][column] = rule.action_color
    else:
        raise AssertionError(f"action is not implemented: {rule.action}")
    return SceneExecution("ok", as_grid(canvas), None, len(selected))


def execute_scene_rule(rule: SceneRule, grid: Grid) -> SceneExecution:
    try:
        return _execute_with_objects(rule, grid, _scene_objects(rule, grid))
    except Exception:
        return SceneExecution("invalid", None, "internal_error", 0)


@dataclass(frozen=True, slots=True)
class SceneSynthesisResult:
    rules: tuple[SceneRule, ...]
    parse_count: int
    rule_trial_count: int
    demo_execution_count: int
    exact_semantic_duplicate_count: int


def _candidate_backgrounds(task: BlindTask, *, limit: int = 3) -> tuple[int, ...]:
    grids = tuple(pair.input for pair in task.train) + task.test_inputs
    ranked_by_grid = tuple(
        tuple(
            color
            for color in background_hypotheses(
                grid, include_none=False, max_backgrounds=3
            )
            if color is not None
        )
        for grid in grids
    )
    common = set(ranked_by_grid[0])
    for ranked in ranked_by_grid[1:]:
        common &= set(ranked)
    if not common:
        common = set.intersection(
            *({cell for row in grid for cell in row} for grid in grids)
        )
    score = Counter()
    for ranked in ranked_by_grid:
        for index, color in enumerate(ranked):
            score[color] += index
    return tuple(sorted(common, key=lambda color: (score[color], color))[:limit])


def _action_colors(task: BlindTask, *, limit: int = 8) -> tuple[int, ...]:
    input_counts = Counter(
        cell for pair in task.train for row in pair.input for cell in row
    )
    output_counts = Counter(
        cell
        for pair in task.train
        for row in (pair.output or ())
        for cell in row
    )
    query_counts = Counter(cell for grid in task.test_inputs for row in grid for cell in row)
    colors = set(output_counts) | set(input_counts) | set(query_counts)
    return tuple(
        sorted(
            colors,
            key=lambda color: (
                color in input_counts,
                -output_counts[color],
                -query_counts[color],
                color,
            ),
        )[:limit]
    )


def synthesize_scene_rules(
    task: BlindTask,
    *,
    max_exact_rules: int = 64,
) -> SceneSynthesisResult:
    if not isinstance(task, BlindTask):
        raise TypeError("scene synthesis accepts BlindTask only")
    if type(max_exact_rules) is not int or max_exact_rules < 1:
        raise ValueError("max_exact_rules must be positive")

    demo_inputs = tuple(pair.input for pair in task.train)
    demo_outputs = tuple(pair.output for pair in task.train)
    if any(output is None for output in demo_outputs):
        raise ValueError("blind demonstrations must retain their outputs")
    query_inputs = task.test_inputs
    all_inputs = demo_inputs + query_inputs
    action_colors = _action_colors(task)
    exact_by_semantics: dict[tuple[object, ...], SceneRule] = {}
    parse_count = 0
    rule_trials = 0
    demo_executions = 0
    duplicate_count = 0

    for background in _candidate_backgrounds(task):
        for connectivity in (4, 8):
            for color_mode in (
                ColorMode.SINGLE_COLOR,
                ColorMode.MULTICOLOR_FOREGROUND,
            ):
                probe = SceneRule(
                    background,
                    connectivity,
                    color_mode.value,
                    "all",
                    None,
                    "keep",
                    None,
                )
                objects_by_input = tuple(
                    _scene_objects(probe, grid) for grid in all_inputs
                )
                parse_count += len(all_inputs)
                selector_colors = sorted(
                    {
                        color
                        for objects in objects_by_input
                        for item in objects
                        for color in item.colors
                    }
                )
                selector_areas = sorted(
                    {
                        item.area
                        for objects in objects_by_input
                        for item in objects
                        if item.area <= 900
                    }
                )
                selectors: list[tuple[str, int | None]] = [
                    (name, None) for name in SELECTORS if name not in ARG_SELECTORS
                ]
                selectors.extend(("color", color) for color in selector_colors)
                selectors.extend(("area", area) for area in selector_areas)
                actions: list[tuple[str, int | None]] = [
                    ("keep", None),
                    ("erase", None),
                    ("crop", None),
                ]
                actions.extend(
                    (action, color)
                    for action in sorted(COLOR_ACTIONS)
                    for color in action_colors
                )

                for selector, selector_argument in selectors:
                    for action, action_color in actions:
                        rule = SceneRule(
                            background,
                            connectivity,
                            color_mode.value,
                            selector,
                            selector_argument,
                            action,
                            action_color,
                        )
                        rule_trials += 1
                        demo_results = tuple(
                            _execute_with_objects(rule, grid, objects)
                            for grid, objects in zip(
                                demo_inputs,
                                objects_by_input[: len(demo_inputs)],
                            )
                        )
                        demo_executions += len(demo_results)
                        if any(not result.ok for result in demo_results):
                            continue
                        if tuple(result.output for result in demo_results) != demo_outputs:
                            continue
                        query_results = tuple(
                            _execute_with_objects(rule, grid, objects)
                            for grid, objects in zip(
                                query_inputs,
                                objects_by_input[len(demo_inputs) :],
                            )
                        )
                        if any(not result.ok for result in query_results):
                            continue
                        signature = (
                            tuple(grid_key(result.output) for result in demo_results),
                            tuple(grid_key(result.output) for result in query_results),
                        )
                        incumbent = exact_by_semantics.get(signature)
                        if incumbent is None or (
                            rule.description_bits,
                            canonical_json(rule.to_json_dict()),
                        ) < (
                            incumbent.description_bits,
                            canonical_json(incumbent.to_json_dict()),
                        ):
                            duplicate_count += int(incumbent is not None)
                            exact_by_semantics[signature] = rule
                        else:
                            duplicate_count += 1

    rules = tuple(
        sorted(
            exact_by_semantics.values(),
            key=lambda rule: (
                rule.description_bits,
                canonical_json(rule.to_json_dict()),
            ),
        )[:max_exact_rules]
    )
    return SceneSynthesisResult(
        rules,
        parse_count,
        rule_trials,
        demo_executions,
        duplicate_count,
    )


def _scene_hypothesis(rule: SceneRule) -> CandidateHypothesis:
    serialized = rule.to_json_dict()
    rule_digest = hashlib.sha256(canonical_json(serialized).encode("ascii")).hexdigest()

    def replay(grid: Grid) -> object | None:
        result = execute_scene_rule(rule, grid)
        return result.output if result.ok else None

    def hard_verify(task: BlindTask) -> bool:
        reconstructed = SceneRule.from_json_dict(serialized)
        grids = tuple(pair.input for pair in task.train) + task.test_inputs
        return all(execute_scene_rule(reconstructed, grid).ok for grid in grids)

    return CandidateHypothesis.create(
        name=f"scene_dsl:{rule_digest[:16]}",
        source="scene_predicate_dsl",
        source_version=SCENE_PROVIDER_VERSION,
        route="dsl_program",
        description_bits=rule.description_bits,
        verification_mode="replayable",
        functional_trace=rule.functional_trace,
        spec={"scene_rule": serialized},
        metadata={
            "demo_exact": True,
            "emission_lane": "demo_exact_scene_predicate",
            "parse_predicates": [
                "background",
                "connectivity",
                "color_mode",
            ],
            "selector_predicate": rule.selector,
            "action_theorem": rule.action,
        },
        replay=replay,
        hard_verifier=hard_verify,
    )


@dataclass(slots=True)
class SceneProgramProvider:
    max_exact_rules: int = 64
    name: str = "scene_predicate_dsl"
    route: str = "dsl_program"
    strict_budget_contract: bool = field(default=False, init=False)
    supports_residual_actions: bool = field(default=False, init=False)
    supports_repeated_batches: bool = field(default=False, init=False)
    max_control_calls: int = field(default=1, init=False)
    parent_sensitive_operators: frozenset[str] = field(
        default=frozenset(), init=False
    )

    def __post_init__(self) -> None:
        if type(self.max_exact_rules) is not int or self.max_exact_rules < 1:
            raise ValueError("max_exact_rules must be positive")

    def propose(
        self,
        task: BlindTask,
        features: TaskFeatures,
        decision: RouteDecision,
    ) -> ProviderResult:
        del features, decision
        result = synthesize_scene_rules(task, max_exact_rules=self.max_exact_rules)
        candidates = tuple(_scene_hypothesis(rule) for rule in result.rules)
        diagnostics: dict[str, Any] = {
            "scene_dsl_version": SCENE_DSL_VERSION,
            "exact_rule_count": len(result.rules),
            "exact_semantic_duplicates_removed": result.exact_semantic_duplicate_count,
            "native_cost": {
                "scene_parses": result.parse_count,
                "scene_rule_trials": result.rule_trial_count,
                "demo_scene_rule_executions": result.demo_execution_count,
            },
        }
        if not candidates:
            return ProviderResult.abstained(
                self.name, self.route, "no_demo_exact_scene_rule", diagnostics
            )
        return ProviderResult.ok(self.name, self.route, candidates, diagnostics)

    def act(
        self,
        task: BlindTask,
        features: TaskFeatures,
        decision: RouteDecision,
        blackboard: "Blackboard",
        action: "ControlAction",
    ) -> ProviderResult:
        del blackboard
        if (
            action.kind != "propose"
            or action.actor != self.name
            or action.route != self.route
            or action.operator != "synthesize"
            or action.parent_hypothesis_id is not None
        ):
            raise ValueError("scene DSL provider received an incompatible action")
        raw = self.propose(task, features, decision)
        if raw.status != "ok":
            return raw
        selected = raw.candidates[: action.budget.candidate_slots]
        return ProviderResult.ok(
            self.name,
            self.route,
            selected,
            {
                **raw.diagnostics,
                "action_operator": action.operator,
                "candidate_slot_limit": action.budget.candidate_slots,
                "emitted_candidate_count": len(selected),
            },
        )

