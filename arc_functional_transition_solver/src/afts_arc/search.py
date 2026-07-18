"""Bounded deterministic search over the Phase-1 typed grid DSL."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .blind import BlindTask
from .candidate import CandidateRecord
from .dsl import (
    DSL_SEMANTICS_VERSION,
    ExecutionOutcome,
    Instruction,
    InvalidCode,
    Program,
    execute_program,
)
from .grid import Grid, grid_key
from .parse import ColorMode, background_hypotheses, parse_grid
from .panel import overlay_panel_grid
from .panel_actions import (
    D4Step,
    broadcast_panel_lattice_periodic,
    broadcast_panel_sequence_d4,
    panel_lattice_period_bounds,
)
from .contact_actions import paint_bbox_contacts
from .relation import ContactStructuralStatus, parse_bbox_contacts
from .residual import DemoResidual, compare_grids
from .shape import OutputShapeProposal, infer_output_shape_proposals

PANEL_SEQUENCE_D4_PROPOSER_SEMANTICS_VERSION = (
    "afts-panel-sequence-d4-proposer/v0.1"
)
PANEL_LATTICE_PERIODIC_PROPOSER_SEMANTICS_VERSION = (
    "afts-panel-lattice-periodic-proposer/v0.1"
)
BBOX_CONTACT_PROPOSER_SEMANTICS_VERSION = "afts-bbox-contact-proposer/v0.1"


@dataclass(frozen=True, slots=True)
class SearchConfig:
    max_depth: int = 2
    beam_width: int = 64
    max_instruction_options: int = 64
    max_exact_programs: int = 128

    def __post_init__(self) -> None:
        if not 1 <= self.max_depth <= 3:
            raise ValueError("max_depth must be in [1, 3]")
        for name in ("beam_width", "max_instruction_options", "max_exact_programs"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True, slots=True)
class ProgramEvaluation:
    program: Program
    demo_outcomes: tuple[ExecutionOutcome, ...]
    demo_residuals: tuple[DemoResidual, ...]
    query_outcomes: tuple[ExecutionOutcome, ...]
    exact_demo_count: int
    shape_match_count: int
    overlap_matches: int
    comparison_cells: int

    @property
    def all_demo_exact(self) -> bool:
        return self.exact_demo_count == len(self.demo_residuals)

    @property
    def agreement(self) -> float:
        return self.overlap_matches / self.comparison_cells

    def semantic_signature(self) -> tuple[object, ...]:
        demo = tuple(
            (outcome.status, grid_key(outcome.output) if outcome.output is not None else None)
            for outcome in self.demo_outcomes
        )
        query = tuple(
            (outcome.status, grid_key(outcome.output) if outcome.output is not None else None)
            for outcome in self.query_outcomes
        )
        return (demo, query)


@dataclass(frozen=True, slots=True)
class PanelSequenceD4ProposalResult:
    instructions: tuple[Instruction, ...]
    trial_count: int
    demo_execution_count: int

    def __post_init__(self) -> None:
        instructions = tuple(self.instructions)
        if any(not isinstance(item, Instruction) for item in instructions):
            raise TypeError("panel-sequence proposals must contain Instructions")
        if (
            type(self.trial_count) is not int
            or type(self.demo_execution_count) is not int
            or self.trial_count < 0
            or self.demo_execution_count < 0
        ):
            raise ValueError("panel-sequence proposal costs must be non-negative")
        if any(
            item.op != "broadcast_panel_sequence_d4"
            for item in instructions
        ):
            raise ValueError("panel-sequence proposal result contains a wrong op")
        object.__setattr__(self, "instructions", instructions)


@dataclass(frozen=True, slots=True)
class PanelLatticePeriodicBound:
    background: int
    max_row_period: int
    max_column_period: int

    def __post_init__(self) -> None:
        if type(self.background) is not int or not 0 <= self.background <= 9:
            raise ValueError("periodic bound background must be an ARC color")
        for name in ("max_row_period", "max_column_period"):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= 29:
                raise ValueError(
                    f"periodic bound {name} must be an integer in [1, 29]"
                )

    def to_json_dict(self) -> dict[str, int]:
        return {
            "background": self.background,
            "max_row_period": self.max_row_period,
            "max_column_period": self.max_column_period,
        }


@dataclass(frozen=True, slots=True)
class PanelLatticePeriodicProposalResult:
    instructions: tuple[Instruction, ...]
    period_bounds: tuple[PanelLatticePeriodicBound, ...]
    structural_check_count: int
    trial_count: int
    demo_execution_count: int

    def __post_init__(self) -> None:
        instructions = tuple(self.instructions)
        bounds = tuple(self.period_bounds)
        if any(not isinstance(item, Instruction) for item in instructions):
            raise TypeError("periodic panel proposals must contain Instructions")
        if any(
            item.op != "broadcast_panel_lattice_periodic"
            for item in instructions
        ):
            raise ValueError("periodic panel proposal result contains a wrong op")
        if any(not isinstance(item, PanelLatticePeriodicBound) for item in bounds):
            raise TypeError(
                "periodic panel proposal bounds must contain bound records"
            )
        if tuple(item.background for item in bounds) != tuple(
            sorted({item.background for item in bounds})
        ):
            raise ValueError("periodic panel bounds must have unique sorted backgrounds")
        if len(bounds) > 3:
            raise ValueError("periodic panel bounds exceed the background cap")
        for name in (
            "structural_check_count",
            "trial_count",
            "demo_execution_count",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(
                    f"periodic panel proposal {name} must be non-negative"
                )
        expected_trials = sum(
            item.max_row_period * item.max_column_period for item in bounds
        )
        if self.trial_count != expected_trials:
            raise ValueError("periodic panel trial count does not match its bounds")
        if self.trial_count > 3 * 29 * 29:
            raise ValueError("periodic panel trial count exceeds the fixed domain")
        if (self.trial_count == 0) != (self.demo_execution_count == 0):
            raise ValueError(
                "periodic panel trial and demo-execution costs must both be zero or positive"
            )
        bound_by_background = {item.background: item for item in bounds}
        proposal_keys: list[tuple[int, int, int]] = []
        for instruction in instructions:
            arguments = instruction.arguments
            background = arguments["background"]
            row_period = arguments["row_period"]
            column_period = arguments["column_period"]
            bound = bound_by_background.get(background)
            if bound is None or not (
                row_period <= bound.max_row_period
                and column_period <= bound.max_column_period
            ):
                raise ValueError("periodic panel proposal lies outside its bounds")
            proposal_keys.append((background, row_period, column_period))
        if proposal_keys != sorted(set(proposal_keys)):
            raise ValueError(
                "periodic panel proposals must be unique and canonically ordered"
            )
        if len(instructions) > self.trial_count:
            raise ValueError("periodic panel proposals exceed the trial count")
        if bounds and self.structural_check_count == 0:
            raise ValueError("periodic panel structural check ledger is empty")
        object.__setattr__(self, "instructions", instructions)
        object.__setattr__(self, "period_bounds", bounds)


@dataclass(frozen=True, slots=True)
class BBoxContactBound:
    background: int
    max_object_count: int
    max_anchor_candidate_count: int
    max_relation_count: int

    def __post_init__(self) -> None:
        if type(self.background) is not int or not 0 <= self.background <= 9:
            raise ValueError("bbox-contact bound background must be an ARC color")
        for name in (
            "max_object_count",
            "max_anchor_candidate_count",
            "max_relation_count",
        ):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value <= 900:
                raise ValueError(
                    f"bbox-contact bound {name} must be an integer in [0, 900]"
                )
        if self.max_anchor_candidate_count > self.max_object_count:
            raise ValueError("bbox-contact anchors exceed the object bound")
        if self.max_relation_count > max(0, self.max_object_count - 1):
            raise ValueError("bbox-contact relations exceed the per-anchor bound")

    def to_json_dict(self) -> dict[str, int]:
        return {
            "background": self.background,
            "max_object_count": self.max_object_count,
            "max_anchor_candidate_count": self.max_anchor_candidate_count,
            "max_relation_count": self.max_relation_count,
        }


@dataclass(frozen=True, slots=True)
class BBoxContactProposalResult:
    instructions: tuple[Instruction, ...]
    bounds: tuple[BBoxContactBound, ...]
    structural_check_count: int
    anchor_candidate_count: int
    relation_check_count: int
    admissible_binding_count: int
    action_trial_count: int
    demo_execution_count: int

    def __post_init__(self) -> None:
        instructions = tuple(self.instructions)
        bounds = tuple(self.bounds)
        if any(not isinstance(item, Instruction) for item in instructions):
            raise TypeError("bbox-contact proposals must contain Instructions")
        if any(item.op != "paint_bbox_contacts" for item in instructions):
            raise ValueError("bbox-contact proposal result contains a wrong op")
        if any(not isinstance(item, BBoxContactBound) for item in bounds):
            raise TypeError("bbox-contact bounds must contain bound records")
        if tuple(item.background for item in bounds) != tuple(
            sorted({item.background for item in bounds})
        ):
            raise ValueError("bbox-contact bounds must have unique sorted backgrounds")
        if len(bounds) > 3:
            raise ValueError("bbox-contact bounds exceed the background cap")
        for name in (
            "structural_check_count",
            "anchor_candidate_count",
            "relation_check_count",
            "admissible_binding_count",
            "action_trial_count",
            "demo_execution_count",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"bbox-contact {name} must be non-negative")
        if self.action_trial_count != self.admissible_binding_count:
            raise ValueError("bbox-contact trials must equal admissible bindings")
        if bounds:
            if (
                self.structural_check_count == 0
                or self.structural_check_count % len(bounds)
            ):
                raise ValueError(
                    "bbox-contact structural checks do not close to its bounds"
                )
            demonstration_count = self.structural_check_count // len(bounds)
            if self.action_trial_count > len(bounds):
                raise ValueError("bbox-contact trials exceed the background domain")
            if self.demo_execution_count != (
                self.action_trial_count * demonstration_count
            ):
                raise ValueError(
                    "bbox-contact demo executions do not close to trials and demonstrations"
                )
            if self.anchor_candidate_count > sum(
                item.max_anchor_candidate_count * demonstration_count
                for item in bounds
            ):
                raise ValueError("bbox-contact anchor count exceeds its bounds")
        elif any(
            (
                self.structural_check_count,
                self.anchor_candidate_count,
                self.relation_check_count,
                self.admissible_binding_count,
                self.action_trial_count,
                self.demo_execution_count,
            )
        ):
            raise ValueError("bbox-contact empty bounds require a zero cost ledger")
        if len(instructions) > self.action_trial_count:
            raise ValueError("bbox-contact proposals exceed action trials")
        bound_backgrounds = {item.background for item in bounds}
        proposal_backgrounds = [
            instruction.arguments["background"] for instruction in instructions
        ]
        if proposal_backgrounds != sorted(set(proposal_backgrounds)):
            raise ValueError(
                "bbox-contact proposals must be unique and canonically ordered"
            )
        if any(item not in bound_backgrounds for item in proposal_backgrounds):
            raise ValueError("bbox-contact proposal lies outside its bounds")
        object.__setattr__(self, "instructions", instructions)
        object.__setattr__(self, "bounds", bounds)


@dataclass(frozen=True, slots=True)
class ProgramSearchResult:
    config: SearchConfig
    shape_proposals: tuple[OutputShapeProposal, ...]
    panel_instruction_proposals: tuple[Instruction, ...]
    panel_sequence_d4_instruction_proposals: tuple[Instruction, ...]
    panel_sequence_d4_trial_count: int
    panel_sequence_d4_demo_execution_count: int
    panel_lattice_periodic_period_bounds: tuple[PanelLatticePeriodicBound, ...]
    panel_lattice_periodic_instruction_proposals: tuple[Instruction, ...]
    panel_lattice_periodic_structural_check_count: int
    panel_lattice_periodic_trial_count: int
    panel_lattice_periodic_demo_execution_count: int
    bbox_contact_bounds: tuple[BBoxContactBound, ...]
    bbox_contact_instruction_proposals: tuple[Instruction, ...]
    bbox_contact_structural_check_count: int
    bbox_contact_anchor_candidate_count: int
    bbox_contact_relation_check_count: int
    bbox_contact_admissible_binding_count: int
    bbox_contact_action_trial_count: int
    bbox_contact_demo_execution_count: int
    instruction_option_count_pre_cap: int
    instruction_option_count_post_cap: int
    instruction_option_truncation_count: int
    instruction_options: tuple[Instruction, ...]
    exact_evaluations: tuple[ProgramEvaluation, ...]
    evaluated_program_ids: tuple[str, ...]
    expansions: int
    semantic_duplicates: int
    first_exact_expansion: int | None

    @property
    def exact_programs(self) -> tuple[Program, ...]:
        return tuple(item.program for item in self.exact_evaluations)


def panel_lattice_periodic_options_after_cap(
    proposals: tuple[Instruction, ...],
    instruction_options: tuple[Instruction, ...],
) -> int:
    """Count canonical periodic proposals retained by the option cap."""

    option_keys = {
        (item.op, item.arguments_json) for item in instruction_options
    }
    return sum(
        (item.op, item.arguments_json) in option_keys for item in proposals
    )


def bbox_contact_options_after_cap(
    proposals: tuple[Instruction, ...],
    instruction_options: tuple[Instruction, ...],
) -> int:
    """Count canonical bbox-contact proposals retained by the option cap."""

    option_keys = {(item.op, item.arguments_json) for item in instruction_options}
    return sum(
        (item.op, item.arguments_json) in option_keys for item in proposals
    )


def _task_colors(task: BlindTask) -> tuple[tuple[int, ...], tuple[int, ...]]:
    input_colors = sorted(
        {
            cell
            for pair in task.train
            for row in pair.input
            for cell in row
        }
    )
    output_colors = sorted(
        {
            cell
            for pair in task.train
            for row in pair.output or ()
            for cell in row
        }
    )
    return tuple(input_colors), tuple(output_colors)


def _infer_color_map(task: BlindTask) -> Instruction | None:
    mapping: dict[int, int] = {}
    for pair in task.train:
        if pair.output is None or (
            len(pair.input) != len(pair.output)
            or len(pair.input[0]) != len(pair.output[0])
        ):
            return None
        for input_row, output_row in zip(pair.input, pair.output):
            for input_color, output_color in zip(input_row, output_row):
                previous = mapping.get(input_color)
                if previous is not None and previous != output_color:
                    return None
                mapping[input_color] = output_color
    if not mapping:
        return None
    return Instruction.create("map_colors", pairs=sorted(mapping.items()))


def instruction_proposals(
    task: BlindTask,
    *,
    shape_proposals: tuple[OutputShapeProposal, ...] | None = None,
    panel_proposals: tuple[Instruction, ...] | None = None,
    panel_sequence_d4_proposals: tuple[Instruction, ...] | None = None,
    panel_lattice_periodic_proposals: tuple[Instruction, ...] | None = None,
    bbox_contact_proposals: tuple[Instruction, ...] | None = None,
) -> tuple[Instruction, ...]:
    """Derive a deterministic, demo-only finite instruction vocabulary."""

    proposals: list[Instruction] = [
        Instruction.create("identity"),
        Instruction.create("rotate90"),
        Instruction.create("rotate180"),
        Instruction.create("rotate270"),
        Instruction.create("flip_horizontal"),
        Instruction.create("flip_vertical"),
        Instruction.create("transpose"),
        Instruction.create("anti_transpose"),
    ]
    inferred_map = _infer_color_map(task)
    if inferred_map is not None:
        proposals.append(inferred_map)
    selected_shape_proposals = (
        shape_proposals
        if shape_proposals is not None
        else infer_output_shape_proposals(task)
    )
    factor_pairs = sorted(
        {
            (proposal.row_factor, proposal.column_factor)
            for proposal in selected_shape_proposals
            if (proposal.row_factor, proposal.column_factor) != (1, 1)
        }
    )
    for row_factor, column_factor in factor_pairs:
        proposals.append(
            Instruction.create(
                "scale_pixels",
                row_factor=row_factor,
                column_factor=column_factor,
            )
        )
        proposals.append(
            Instruction.create(
                "tile_grid",
                row_repeats=row_factor,
                column_repeats=column_factor,
            )
        )
    selected_panel_proposals = (
        panel_proposals
        if panel_proposals is not None
        else panel_instruction_proposals(task)
    )
    proposals.extend(selected_panel_proposals)
    selected_panel_sequence_d4_proposals = (
        panel_sequence_d4_proposals
        if panel_sequence_d4_proposals is not None
        else panel_sequence_d4_instruction_proposals(task)
    )
    proposals.extend(selected_panel_sequence_d4_proposals)
    selected_panel_lattice_periodic_proposals = (
        panel_lattice_periodic_proposals
        if panel_lattice_periodic_proposals is not None
        else panel_lattice_periodic_instruction_proposals(task)
    )
    proposals.extend(selected_panel_lattice_periodic_proposals)
    selected_bbox_contact_proposals = (
        bbox_contact_proposals
        if bbox_contact_proposals is not None
        else bbox_contact_instruction_proposals(task)
    )
    proposals.extend(selected_bbox_contact_proposals)
    backgrounds: list[int] = []
    for pair in task.train:
        for background in background_hypotheses(pair.input, max_backgrounds=3):
            if background is not None and background not in backgrounds:
                backgrounds.append(background)
    output_colors = _task_colors(task)[1]
    for background in backgrounds:
        proposals.append(Instruction.create("crop_non_background", background=background))
        for connectivity in (4, 8):
            for color_mode in (
                ColorMode.SINGLE_COLOR.value,
                ColorMode.MULTICOLOR_FOREGROUND.value,
            ):
                proposals.append(
                    Instruction.create(
                        "crop_largest_object",
                        background=background,
                        connectivity=connectivity,
                        color_mode=color_mode,
                    )
                )
                proposals.append(
                    Instruction.create(
                        "keep_largest_object",
                        background=background,
                        connectivity=connectivity,
                        color_mode=color_mode,
                    )
                )
        for color in output_colors:
            proposals.append(
                Instruction.create(
                    "render_foreground_bbox", background=background, color=color
                )
            )

    input_colors, output_colors = _task_colors(task)
    for old in input_colors:
        for new in output_colors:
            if old != new:
                proposals.append(Instruction.create("recolor", old=old, new=new))

    unique: list[Instruction] = []
    seen: set[tuple[str, str]] = set()
    for instruction in proposals:
        key = (instruction.op, instruction.arguments_json)
        if key not in seen:
            seen.add(key)
            unique.append(instruction)
    return tuple(unique)


def panel_instruction_proposals(task: BlindTask) -> tuple[Instruction, ...]:
    """Propose demo-compatible panel overlay instructions without query labels."""

    background_sets = [
        {
            color
            for color in background_hypotheses(pair.input, max_backgrounds=3)
            if color is not None
        }
        for pair in task.train
    ]
    common_backgrounds = (
        set.intersection(*background_sets) if background_sets else set()
    )
    proposals: list[Instruction] = []
    for background in sorted(common_backgrounds):
        compatible = True
        for pair in task.train:
            if pair.output is None:
                raise ValueError("blind task training output is missing")
            output, invalid = overlay_panel_grid(pair.input, background=background)
            if invalid is not None or output is None:
                compatible = False
                break
            output_shape = (len(output), len(output[0]))
            target_shape = (len(pair.output), len(pair.output[0]))
            if target_shape not in {output_shape, output_shape[::-1]}:
                compatible = False
                break
        if compatible:
            proposals.append(
                Instruction.create("overlay_panel_grid", background=background)
            )
    return tuple(proposals)


def panel_sequence_d4_instruction_proposals(
    task: BlindTask,
) -> tuple[Instruction, ...]:
    """Enumerate structurally valid D4 sequence actions without query labels."""

    return _panel_sequence_d4_proposal_result(task).instructions


def _panel_sequence_d4_proposal_result(
    task: BlindTask,
) -> PanelSequenceD4ProposalResult:
    """Return canonical proposals plus their explicit structural trial cost."""

    background_sets = [
        {
            color
            for color in background_hypotheses(pair.input, max_backgrounds=3)
            if color is not None
        }
        for pair in task.train
    ]
    common_backgrounds = (
        set.intersection(*background_sets) if background_sets else set()
    )
    proposals: list[Instruction] = []
    trial_count = 0
    demo_execution_count = 0
    for background in sorted(common_backgrounds):
        for step in D4Step:
            trial_count += 1
            observations: list[bool] = []
            for pair in task.train:
                if pair.output is None:
                    raise ValueError("blind task training output is missing")
                demo_execution_count += 1
                output, invalid = broadcast_panel_sequence_d4(
                    pair.input,
                    background=background,
                    step=step,
                )
                if invalid is not None or output is None:
                    observations.append(False)
                    continue
                produced_shape = (len(output), len(output[0]))
                target_shape = (len(pair.output), len(pair.output[0]))
                observations.append(
                    target_shape in {produced_shape, produced_shape[::-1]}
                )
            if observations and all(observations):
                proposals.append(
                    Instruction.create(
                        "broadcast_panel_sequence_d4",
                        background=background,
                        step=step.value,
                    )
                )
    return PanelSequenceD4ProposalResult(
        instructions=tuple(proposals),
        trial_count=trial_count,
        demo_execution_count=demo_execution_count,
    )


def panel_lattice_periodic_instruction_proposals(
    task: BlindTask,
) -> tuple[Instruction, ...]:
    """Enumerate bounded two-axis periodic actions without query labels."""

    return _panel_lattice_periodic_proposal_result(task).instructions


def _panel_lattice_periodic_proposal_result(
    task: BlindTask,
) -> PanelLatticePeriodicProposalResult:
    """Return shape-compatible periodic proposals and their explicit cost."""

    background_sets = [
        {
            color
            for color in background_hypotheses(pair.input, max_backgrounds=3)
            if color is not None
        }
        for pair in task.train
    ]
    common_backgrounds = (
        set.intersection(*background_sets) if background_sets else set()
    )
    proposals: list[Instruction] = []
    bounds: list[PanelLatticePeriodicBound] = []
    structural_check_count = 0
    trial_count = 0
    demo_execution_count = 0
    for background in sorted(common_backgrounds):
        observed_bounds: list[tuple[int, int] | None] = []
        for pair in task.train:
            structural_check_count += 1
            observed_bounds.append(
                panel_lattice_period_bounds(
                    pair.input, background=background
                )
            )
        if not observed_bounds or any(item is None for item in observed_bounds):
            continue
        eligible_bounds = tuple(
            item for item in observed_bounds if item is not None
        )
        bound = PanelLatticePeriodicBound(
            background=background,
            max_row_period=max(item[0] for item in eligible_bounds),
            max_column_period=max(item[1] for item in eligible_bounds),
        )
        bounds.append(bound)
        for row_period in range(1, bound.max_row_period + 1):
            for column_period in range(1, bound.max_column_period + 1):
                trial_count += 1
                observations: list[bool] = []
                for pair in task.train:
                    if pair.output is None:
                        raise ValueError("blind task training output is missing")
                    demo_execution_count += 1
                    output, invalid = broadcast_panel_lattice_periodic(
                        pair.input,
                        background=background,
                        row_period=row_period,
                        column_period=column_period,
                    )
                    if invalid is not None or output is None:
                        observations.append(False)
                        continue
                    produced_shape = (len(output), len(output[0]))
                    target_shape = (len(pair.output), len(pair.output[0]))
                    observations.append(
                        target_shape
                        in {produced_shape, produced_shape[::-1]}
                    )
                if observations and all(observations):
                    proposals.append(
                        Instruction.create(
                            "broadcast_panel_lattice_periodic",
                            background=background,
                            row_period=row_period,
                            column_period=column_period,
                        )
                    )
    return PanelLatticePeriodicProposalResult(
        instructions=tuple(proposals),
        period_bounds=tuple(bounds),
        structural_check_count=structural_check_count,
        trial_count=trial_count,
        demo_execution_count=demo_execution_count,
    )


def bbox_contact_instruction_proposals(
    task: BlindTask,
) -> tuple[Instruction, ...]:
    """Enumerate blind axis-ray contact actions without query labels."""

    return _bbox_contact_proposal_result(task).instructions


def _bbox_contact_proposal_result(
    task: BlindTask,
) -> BBoxContactProposalResult:
    """Return structural bbox-contact proposals and a closed cost ledger."""

    background_sets = [
        {
            color
            for color in background_hypotheses(pair.input, max_backgrounds=3)
            if color is not None
        }
        for pair in task.train
    ]
    common_backgrounds = (
        set.intersection(*background_sets) if background_sets else set()
    )
    proposals: list[Instruction] = []
    bounds: list[BBoxContactBound] = []
    structural_check_count = 0
    anchor_candidate_count = 0
    relation_check_count = 0
    admissible_binding_count = 0
    action_trial_count = 0
    demo_execution_count = 0
    for background in sorted(common_backgrounds):
        complete_counts: list[int] = []
        max_object_count = 0
        max_anchor_count = 0
        max_relation_count = 0
        for pair in task.train:
            if pair.output is None:
                raise ValueError("blind task training output is missing")
            structural_check_count += 1
            object_hypothesis = parse_grid(
                pair.input,
                backgrounds=(background,),
                connectivities=(4,),
                color_modes=(ColorMode.SINGLE_COLOR,),
                max_backgrounds=None,
            ).hypotheses[0]
            objects = object_hypothesis.objects
            anchors = tuple(
                item
                for item in objects
                if item.bbox.height >= 2
                and item.bbox.width >= 2
                and item.area == item.bbox.height * item.bbox.width
                and len(item.colors) == 1
            )
            max_object_count = max(max_object_count, len(objects))
            max_anchor_count = max(max_anchor_count, len(anchors))
            anchor_candidate_count += len(anchors)
            relation_check_count += len(anchors) * max(
                0, len(objects) - len(anchors)
            )
            bundle = parse_bbox_contacts(pair.input, background=background)
            complete = tuple(
                hypothesis
                for hypothesis in bundle.hypotheses
                if hypothesis.structural_status is ContactStructuralStatus.COMPLETE
            )
            complete_counts.append(len(complete))
            max_relation_count = max(
                max_relation_count,
                max(
                    (
                        len(hypothesis.relations)
                        for hypothesis in bundle.hypotheses
                        if hypothesis.anchor_fills_bbox
                    ),
                    default=0,
                ),
            )
        bounds.append(
            BBoxContactBound(
                background=background,
                max_object_count=max_object_count,
                max_anchor_candidate_count=max_anchor_count,
                max_relation_count=max_relation_count,
            )
        )
        if not complete_counts or not all(count == 1 for count in complete_counts):
            continue
        admissible_binding_count += 1
        action_trial_count += 1
        observations: list[bool] = []
        for pair in task.train:
            if pair.output is None:
                raise ValueError("blind task training output is missing")
            demo_execution_count += 1
            output, invalid = paint_bbox_contacts(
                pair.input,
                background=background,
            )
            if invalid is not None or output is None:
                observations.append(False)
                continue
            produced_shape = (len(output), len(output[0]))
            target_shape = (len(pair.output), len(pair.output[0]))
            observations.append(
                target_shape in {produced_shape, produced_shape[::-1]}
            )
        if observations and all(observations):
            proposals.append(
                Instruction.create("paint_bbox_contacts", background=background)
            )
    return BBoxContactProposalResult(
        instructions=tuple(proposals),
        bounds=tuple(bounds),
        structural_check_count=structural_check_count,
        anchor_candidate_count=anchor_candidate_count,
        relation_check_count=relation_check_count,
        admissible_binding_count=admissible_binding_count,
        action_trial_count=action_trial_count,
        demo_execution_count=demo_execution_count,
    )


def evaluate_program(program: Program, task: BlindTask) -> ProgramEvaluation:
    demo_outcomes: list[ExecutionOutcome] = []
    residuals: list[DemoResidual] = []
    for pair_index, pair in enumerate(task.train):
        outcome = execute_program(program, pair.input)
        demo_outcomes.append(outcome)
        residuals.append(
            compare_grids(
                outcome.output,
                pair.output,
                pair_index=pair_index,
                invalid_code=outcome.invalid_code.value if outcome.invalid_code else None,
            )
        )
    query_outcomes = tuple(execute_program(program, grid) for grid in task.test_inputs)
    return ProgramEvaluation(
        program=program,
        demo_outcomes=tuple(demo_outcomes),
        demo_residuals=tuple(residuals),
        query_outcomes=query_outcomes,
        exact_demo_count=sum(item.exact for item in residuals),
        shape_match_count=sum(item.shape_match for item in residuals),
        overlap_matches=sum(item.overlap_matches for item in residuals),
        comparison_cells=sum(item.comparison_cells for item in residuals),
    )


def _rank_key(evaluation: ProgramEvaluation) -> tuple[object, ...]:
    return (
        -evaluation.exact_demo_count,
        -evaluation.shape_match_count,
        -evaluation.agreement,
        evaluation.program.node_count,
        evaluation.program.program_id,
    )


def search_programs(
    task: BlindTask, *, config: SearchConfig | None = None
) -> ProgramSearchResult:
    selected_config = config or SearchConfig()
    shape_proposals = infer_output_shape_proposals(task)
    panel_proposals = panel_instruction_proposals(task)
    panel_sequence_d4_result = _panel_sequence_d4_proposal_result(task)
    panel_sequence_d4_proposals = panel_sequence_d4_result.instructions
    panel_lattice_periodic_result = _panel_lattice_periodic_proposal_result(task)
    panel_lattice_periodic_proposals = (
        panel_lattice_periodic_result.instructions
    )
    bbox_contact_result = _bbox_contact_proposal_result(task)
    bbox_contact_proposals = bbox_contact_result.instructions
    full_options = instruction_proposals(
        task,
        shape_proposals=shape_proposals,
        panel_proposals=panel_proposals,
        panel_sequence_d4_proposals=panel_sequence_d4_proposals,
        panel_lattice_periodic_proposals=panel_lattice_periodic_proposals,
        bbox_contact_proposals=bbox_contact_proposals,
    )
    options = full_options[: selected_config.max_instruction_options]
    if not options:
        raise RuntimeError("instruction proposal set is empty")

    exact: dict[str, ProgramEvaluation] = {}
    evaluated_ids: set[str] = set()
    expansions = 0
    semantic_duplicates = 0
    first_exact: int | None = None
    frontier_programs: tuple[Program, ...] = ()

    for depth in range(1, selected_config.max_depth + 1):
        if depth == 1:
            programs = tuple(Program.create((instruction,)) for instruction in options)
        else:
            expanded: list[Program] = []
            for prefix in frontier_programs:
                for suffix in options:
                    if suffix.op == "identity":
                        continue
                    expanded.append(Program.create((*prefix.instructions, suffix)))
            programs = tuple(expanded)

        representatives: dict[tuple[object, ...], ProgramEvaluation] = {}
        for program in programs:
            if program.program_id in evaluated_ids:
                continue
            evaluated_ids.add(program.program_id)
            expansions += 1
            evaluation = evaluate_program(program, task)
            if any(
                outcome.invalid_code is InvalidCode.INTERNAL_ERROR
                for outcome in (*evaluation.demo_outcomes, *evaluation.query_outcomes)
            ):
                raise RuntimeError(
                    f"DSL internal error while evaluating program {program.program_id}"
                )
            signature = evaluation.semantic_signature()
            incumbent = representatives.get(signature)
            if incumbent is None or _rank_key(evaluation) < _rank_key(incumbent):
                if incumbent is not None:
                    semantic_duplicates += 1
                representatives[signature] = evaluation
            else:
                semantic_duplicates += 1
            if evaluation.all_demo_exact:
                exact.setdefault(program.program_id, evaluation)
                if first_exact is None:
                    first_exact = expansions

        ranked = tuple(sorted(representatives.values(), key=_rank_key))
        frontier_programs = tuple(
            item.program for item in ranked[: selected_config.beam_width]
        )

    exact_evaluations = tuple(
        sorted(exact.values(), key=lambda item: (item.program.node_count, item.program.program_id))[
            : selected_config.max_exact_programs
        ]
    )
    return ProgramSearchResult(
        config=selected_config,
        shape_proposals=shape_proposals,
        panel_instruction_proposals=panel_proposals,
        panel_sequence_d4_instruction_proposals=panel_sequence_d4_proposals,
        panel_sequence_d4_trial_count=panel_sequence_d4_result.trial_count,
        panel_sequence_d4_demo_execution_count=(
            panel_sequence_d4_result.demo_execution_count
        ),
        panel_lattice_periodic_period_bounds=(
            panel_lattice_periodic_result.period_bounds
        ),
        panel_lattice_periodic_instruction_proposals=(
            panel_lattice_periodic_proposals
        ),
        panel_lattice_periodic_structural_check_count=(
            panel_lattice_periodic_result.structural_check_count
        ),
        panel_lattice_periodic_trial_count=(
            panel_lattice_periodic_result.trial_count
        ),
        panel_lattice_periodic_demo_execution_count=(
            panel_lattice_periodic_result.demo_execution_count
        ),
        bbox_contact_bounds=bbox_contact_result.bounds,
        bbox_contact_instruction_proposals=bbox_contact_proposals,
        bbox_contact_structural_check_count=(
            bbox_contact_result.structural_check_count
        ),
        bbox_contact_anchor_candidate_count=(
            bbox_contact_result.anchor_candidate_count
        ),
        bbox_contact_relation_check_count=(
            bbox_contact_result.relation_check_count
        ),
        bbox_contact_admissible_binding_count=(
            bbox_contact_result.admissible_binding_count
        ),
        bbox_contact_action_trial_count=(
            bbox_contact_result.action_trial_count
        ),
        bbox_contact_demo_execution_count=(
            bbox_contact_result.demo_execution_count
        ),
        instruction_option_count_pre_cap=len(full_options),
        instruction_option_count_post_cap=len(options),
        instruction_option_truncation_count=len(full_options) - len(options),
        instruction_options=options,
        exact_evaluations=exact_evaluations,
        evaluated_program_ids=tuple(sorted(evaluated_ids)),
        expansions=expansions,
        semantic_duplicates=semantic_duplicates,
        first_exact_expansion=first_exact,
    )


def dsl_candidates(
    task: BlindTask,
    *,
    search_result: ProgramSearchResult | None = None,
    config: SearchConfig | None = None,
) -> tuple[CandidateRecord, ...]:
    result = search_result or search_programs(task, config=config)
    candidates: list[CandidateRecord] = []
    for emission_rank, evaluation in enumerate(result.exact_evaluations):
        for test_index, outcome in enumerate(evaluation.query_outcomes):
            if not outcome.ok or outcome.output is None:
                continue
            candidates.append(
                CandidateRecord.create(
                    task_id=task.task_id,
                    test_index=test_index,
                    source_type="typed_dsl",
                    source_version=DSL_SEMANTICS_VERSION,
                    output=outcome.output,
                    program_hash=evaluation.program.program_id,
                    functional_trace=evaluation.program.functional_trace,
                    generation_parameters={
                        "blind_content_sha256": task.blind_content_sha256,
                        "emission_rank": emission_rank,
                        "program": evaluation.program.to_json_dict(),
                        "search_config": asdict(result.config),
                    },
                    evidence_status="strict_demo_fit_candidate",
                )
            )
    return tuple(candidates)
