"""Generator-known synthetic tasks for symbolic coverage controls."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass

from .blind import BlindTask
from .dsl import Instruction, Program, execute_program
from .grid import Grid, as_grid, grid_key, grid_to_lists
from .panel_actions import D4Step
from .task import ARCPair, ARCTask

SYNTHETIC_SUITE_VERSION = "afts-symbolic-controls/v0.6"
SYNTHETIC_ORACLE_SCHEMA_VERSION = "afts.synthetic-oracle/v1"


@dataclass(frozen=True, slots=True)
class SyntheticCase:
    task: ARCTask
    blind_task: BlindTask
    family: str
    stratum: str
    seed: int
    generator_program: Program

    def oracle_json_dict(self) -> dict[str, object]:
        return {
            "schema": SYNTHETIC_ORACLE_SCHEMA_VERSION,
            "task_id": self.task.task_id,
            "family": self.family,
            "stratum": self.stratum,
            "seed": self.seed,
            "generator_program": self.generator_program.to_json_dict(),
            "test": [
                {
                    "test_index": index,
                    "output": grid_to_lists(pair.output),
                    "output_key": grid_key(pair.output),
                }
                for index, pair in enumerate(self.task.test)
            ],
        }


def _generic_grid(rng: random.Random, *, require_non_square: bool = False) -> Grid:
    while True:
        height = rng.randint(3, 6)
        width = rng.randint(3, 6)
        if require_non_square and height == width:
            continue
        background = rng.randint(0, 2)
        palette = tuple(color for color in range(1, 7) if color != background)
        values = [[background for _ in range(width)] for _ in range(height)]
        for _ in range(rng.randint(3, max(4, height * width // 2))):
            row = rng.randrange(height)
            column = rng.randrange(width)
            values[row][column] = rng.choice(palette)
        grid = as_grid(values)
        if any(cell != background for row in grid for cell in row):
            return grid


def _recolor_grid(rng: random.Random, *, old: int, background: int = 0) -> Grid:
    height = rng.randint(3, 6)
    width = rng.randint(3, 6)
    values = [[background for _ in range(width)] for _ in range(height)]
    positions = [(row, column) for row in range(height) for column in range(width)]
    rng.shuffle(positions)
    for row, column in positions[: rng.randint(2, max(2, len(positions) // 3))]:
        values[row][column] = old
    if len(positions) > 3:
        row, column = positions[-1]
        values[row][column] = 2 if old != 2 else 3
    return as_grid(values)


def _crop_grid(rng: random.Random, *, background: int = 0) -> Grid:
    height = rng.randint(5, 8)
    width = rng.randint(5, 8)
    object_height = rng.randint(2, min(4, height - 2))
    object_width = rng.randint(2, min(4, width - 2))
    top = rng.randint(1, height - object_height - 1)
    left = rng.randint(1, width - object_width - 1)
    values = [[background for _ in range(width)] for _ in range(height)]
    for row in range(top, top + object_height):
        for column in range(left, left + object_width):
            values[row][column] = 3 + ((row - top) + 2 * (column - left)) % 3
    return as_grid(values)


def _largest_grid(rng: random.Random, *, background: int = 0) -> Grid:
    height = rng.randint(6, 8)
    width = rng.randint(6, 8)
    values = [[background for _ in range(width)] for _ in range(height)]
    top = rng.randint(0, 1)
    left = rng.randint(0, 1)
    for row in range(top, top + 2):
        for column in range(left, left + 3):
            values[row][column] = 4
    values[height - 1][width - 1] = 5
    return as_grid(values)


def _shape_grid(rng: random.Random) -> Grid:
    """Generate a small grid that distinguishes cell scaling from whole-grid tiling."""

    while True:
        grid = _generic_grid(rng, require_non_square=True)
        scaled_2x2 = tuple(
            row
            for source_row in grid
            for row in (
                tuple(cell for cell in source_row for _ in range(2)),
            )
            for _ in range(2)
        )
        tiled_2x2 = tuple((row * 2) for row in grid) * 2
        scaled_horizontal = tuple(
            tuple(cell for cell in row for _ in range(2)) for row in grid
        )
        tiled_horizontal = tuple(row * 2 for row in grid)
        if scaled_2x2 != tiled_2x2 and scaled_horizontal != tiled_horizontal:
            return grid


def _panel_grid(rng: random.Random) -> Grid:
    """Generate a separator lattice with row-major overlay conflicts."""

    panel_rows = 2
    panel_columns = rng.choice((2, 3))
    panel_height = rng.randint(2, 4)
    panel_width = rng.randint(2, 4)
    separator = rng.randint(1, 9)
    palette = tuple(color for color in range(1, 10) if color != separator)
    panels = [
        [[0 for _ in range(panel_width)] for _ in range(panel_height)]
        for _ in range(panel_rows * panel_columns)
    ]
    for panel in panels:
        for _ in range(rng.randint(1, max(1, panel_height * panel_width // 2))):
            row = rng.randrange(panel_height)
            column = rng.randrange(panel_width)
            panel[row][column] = rng.choice(palette)
    panels[0][0][0] = max(palette)
    panels[1][0][0] = min(palette)
    panels[0][0][1] = min(palette)
    panels[1][0][1] = max(palette)

    height = panel_rows * panel_height + panel_rows - 1
    width = panel_columns * panel_width + panel_columns - 1
    values = [[separator for _ in range(width)] for _ in range(height)]
    for panel_row in range(panel_rows):
        for panel_column in range(panel_columns):
            panel = panels[panel_row * panel_columns + panel_column]
            top = panel_row * (panel_height + 1)
            left = panel_column * (panel_width + 1)
            for row in range(panel_height):
                for column in range(panel_width):
                    values[top + row][left + column] = panel[row][column]
    return as_grid(values)


def _panel_sequence_parameters(family: str, seed: int) -> tuple[int, D4Step]:
    index = seed % 3
    if family == "broadcast_panel_sequence_d4":
        steps = (D4Step.ROTATE90, D4Step.FLIP_HORIZONTAL, D4Step.TRANSPOSE)
    elif family == "broadcast_panel_sequence_d4_flip_horizontal":
        steps = (D4Step.ROTATE90, D4Step.TRANSPOSE, D4Step.ROTATE270)
    else:
        raise ValueError(f"not a panel-sequence family: {family}")
    background = (0, 2, 3)[random.Random(seed ^ 0xD405D4).randrange(3)]
    return background, steps[index]


def _panel_sequence_grid(
    rng: random.Random,
    *,
    background: int,
    step: D4Step,
    example_index: int,
) -> Grid:
    panel_counts = (2, 3, 4, 5, 2)
    seed_indices = (0, 1, 2, 2, 1)
    panel_count = panel_counts[example_index]
    seed_index = seed_indices[example_index]
    horizontal = example_index % 2 == 0
    separator_thickness = 1 if example_index % 2 == 0 else 2
    swaps_shape = step in {
        D4Step.ROTATE90,
        D4Step.ROTATE270,
        D4Step.TRANSPOSE,
        D4Step.ANTI_TRANSPOSE,
    }
    if swaps_shape:
        panel_height = panel_width = 3
    elif example_index % 2:
        panel_height, panel_width = 2, 3
    else:
        panel_height, panel_width = 3, 2
    separator_choices = tuple(
        color
        for color in range(10)
        if color != background
    )
    separator = separator_choices[example_index % len(separator_choices)]
    palette = tuple(
        color
        for color in range(10)
        if color not in {background, separator}
    )
    seed = [
        [background for _ in range(panel_width)]
        for _ in range(panel_height)
    ]
    coordinates = (
        (0, 0),
        (0, panel_width - 1),
        (panel_height - 1, 0),
        (panel_height - 1, panel_width - 1),
        (panel_height // 2, panel_width // 2),
    )
    offset = rng.randrange(len(palette))
    for index, (row, column) in enumerate(coordinates):
        seed[row][column] = palette[(offset + index) % len(palette)]
    panels = [
        [
            [background for _ in range(panel_width)]
            for _ in range(panel_height)
        ]
        for _ in range(panel_count)
    ]
    panels[seed_index] = seed
    if horizontal:
        values = [
            [
                cell
                for panel_index, panel in enumerate(panels)
                for cell in (
                    ([separator] * separator_thickness if panel_index else [])
                    + panel[row]
                )
            ]
            for row in range(panel_height)
        ]
    else:
        values: list[list[int]] = []
        for panel_index, panel in enumerate(panels):
            if panel_index:
                values.extend(
                    [[separator for _ in range(panel_width)]]
                    * separator_thickness
                )
            values.extend([list(row) for row in panel])
    return as_grid(values)


def _panel_lattice_periodic_parameters(
    family: str, seed: int
) -> tuple[int, int, int]:
    if family not in {
        "broadcast_panel_lattice_periodic",
        "broadcast_panel_lattice_periodic_rotate180",
    }:
        raise ValueError(f"not a panel-lattice periodic family: {family}")
    index = seed % 3
    row_period, column_period = ((2, 2), (2, 3), (3, 2))[index]
    background = (0, 2, 3)[index]
    return background, row_period, column_period


def _panel_lattice_periodic_grid(
    rng: random.Random,
    *,
    background: int,
    row_period: int,
    column_period: int,
    example_index: int,
) -> Grid:
    """Generate a two-axis lattice with only bottom/right suffix raggedness."""

    if not 0 <= example_index < 5:
        raise ValueError("periodic panel example_index must be in [0, 4]")
    nominal_height = 3
    nominal_width = 3
    if example_index == 0:
        row_count = 2 * row_period
        column_count = 2 * column_period
        source_row = row_period - 1
        source_column = column_period - 1
        bottom_height = nominal_height
        right_width = nominal_width
    elif example_index == 1:
        row_count = row_period + 2
        column_count = 2 * column_period
        source_row = 1
        source_column = column_period - 1
        bottom_height = 1
        right_width = nominal_width
    elif example_index == 2:
        row_count = 2 * row_period
        column_count = column_period + 2
        source_row = row_period - 1
        source_column = 1
        bottom_height = nominal_height
        right_width = 1
    elif example_index == 3:
        row_count = row_period + 2
        column_count = column_period + 2
        source_row = 1
        source_column = 1
        bottom_height = 2
        right_width = 2
    else:
        row_count = row_period + 2
        column_count = column_period + 2
        source_row = 0
        source_column = 0
        bottom_height = 1
        right_width = 1

    row_heights = [nominal_height for _ in range(row_count)]
    column_widths = [nominal_width for _ in range(column_count)]
    row_heights[-1] = bottom_height
    column_widths[-1] = right_width
    separator_thickness = 1 + (example_index % 2)
    separator_choices = tuple(color for color in range(10) if color != background)
    separator = separator_choices[example_index % len(separator_choices)]
    palette = tuple(
        color for color in range(10) if color not in {background, separator}
    )

    height = sum(row_heights) + separator_thickness * (row_count - 1)
    width = sum(column_widths) + separator_thickness * (column_count - 1)
    values = [[separator for _ in range(width)] for _ in range(height)]
    row_offsets: list[int] = []
    cursor = 0
    for panel_height in row_heights:
        row_offsets.append(cursor)
        cursor += panel_height + separator_thickness
    column_offsets: list[int] = []
    cursor = 0
    for panel_width in column_widths:
        column_offsets.append(cursor)
        cursor += panel_width + separator_thickness
    for panel_row, panel_height in enumerate(row_heights):
        for panel_column, panel_width in enumerate(column_widths):
            top = row_offsets[panel_row]
            left = column_offsets[panel_column]
            for row in range(panel_height):
                for column in range(panel_width):
                    values[top + row][left + column] = background

    seed = [[background for _ in range(nominal_width)] for _ in range(nominal_height)]
    offset = rng.randrange(len(palette))
    for index, (row, column) in enumerate(
        ((0, 0), (0, 1), (1, 2), (2, 0), (2, 2))
    ):
        seed[row][column] = palette[(offset + index) % len(palette)]
    seed_top = row_offsets[source_row]
    seed_left = column_offsets[source_column]
    for row, cells in enumerate(seed):
        for column, cell in enumerate(cells):
            values[seed_top + row][seed_left + column] = cell
    return as_grid(values)


def _bbox_contact_parameters(family: str, seed: int) -> int:
    if family not in {
        "paint_bbox_contacts",
        "paint_bbox_contacts_rotate180",
    }:
        raise ValueError(f"not a bbox-contact family: {family}")
    return (0, 2, 3)[seed % 3]


def _bbox_contact_grid(
    rng: random.Random,
    *,
    background: int,
    example_index: int,
) -> Grid:
    """Generate a solid anchor with distinct clear external singleton rays."""

    if not 0 <= example_index < 5:
        raise ValueError("bbox-contact example_index must be in [0, 4]")
    anchor_height = 4 + (example_index % 2)
    anchor_width = 4 + ((example_index // 2) % 2)
    top = 4 + (example_index % 2)
    left = 4 + ((example_index + 1) % 2)
    height = top + anchor_height + 5
    width = left + anchor_width + 5
    values = [[background for _ in range(width)] for _ in range(height)]
    palette = tuple(color for color in range(10) if color != background)
    palette_offset = rng.randrange(len(palette))
    anchor_color = palette[palette_offset]
    marker_palette = tuple(
        color for color in range(10) if color not in {background, anchor_color}
    )
    for row in range(top, top + anchor_height):
        for column in range(left, left + anchor_width):
            values[row][column] = anchor_color

    contacts = (
        ("left", 1),
        ("right", anchor_height - 2),
        ("top", 1),
        ("bottom", anchor_width - 2),
        ("left", anchor_height - 2),
        ("top", anchor_width - 2),
    )
    marker_counts = (4, 3, 5, 6, 4)
    color_offset = (palette_offset + example_index) % len(marker_palette)
    for index, (side, offset) in enumerate(contacts[: marker_counts[example_index]]):
        gap = (example_index + index) % 3
        if side == "left":
            row = top + offset
            column = left - gap - 1
        elif side == "right":
            row = top + offset
            column = left + anchor_width + gap
        elif side == "top":
            row = top - gap - 1
            column = left + offset
        else:
            row = top + anchor_height + gap
            column = left + offset
        values[row][column] = marker_palette[
            (color_offset + index // 2) % len(marker_palette)
        ]
    return as_grid(values)


def _program_for_family(family: str, *, seed: int) -> Program:
    if family == "rotate90":
        instructions = (Instruction.create("rotate90"),)
    elif family == "flip_horizontal":
        instructions = (Instruction.create("flip_horizontal"),)
    elif family == "recolor":
        instructions = (Instruction.create("recolor", old=1, new=7),)
    elif family == "rotate90_recolor":
        instructions = (
            Instruction.create("rotate90"),
            Instruction.create("recolor", old=1, new=7),
        )
    elif family == "crop_non_background":
        instructions = (Instruction.create("crop_non_background", background=0),)
    elif family == "crop_rotate90":
        instructions = (
            Instruction.create("crop_non_background", background=0),
            Instruction.create("rotate90"),
        )
    elif family == "keep_largest_object":
        instructions = (
            Instruction.create(
                "keep_largest_object",
                background=0,
                connectivity=4,
                color_mode="single_color",
            ),
        )
    elif family == "scale_pixels_2x2":
        instructions = (
            Instruction.create("scale_pixels", row_factor=2, column_factor=2),
        )
    elif family == "tile_horizontal_2":
        instructions = (
            Instruction.create("tile_grid", row_repeats=1, column_repeats=2),
        )
    elif family == "overlay_panel_grid":
        instructions = (Instruction.create("overlay_panel_grid", background=0),)
    elif family == "overlay_panel_rotate90":
        instructions = (
            Instruction.create("overlay_panel_grid", background=0),
            Instruction.create("rotate90"),
        )
    elif family in {
        "broadcast_panel_sequence_d4",
        "broadcast_panel_sequence_d4_flip_horizontal",
    }:
        background, step = _panel_sequence_parameters(family, seed)
        sequence = Instruction.create(
            "broadcast_panel_sequence_d4",
            background=background,
            step=step.value,
        )
        instructions = (
            (sequence,)
            if family == "broadcast_panel_sequence_d4"
            else (
                sequence,
                Instruction.create("flip_horizontal"),
            )
        )
    elif family in {
        "broadcast_panel_lattice_periodic",
        "broadcast_panel_lattice_periodic_rotate180",
    }:
        background, row_period, column_period = (
            _panel_lattice_periodic_parameters(family, seed)
        )
        periodic = Instruction.create(
            "broadcast_panel_lattice_periodic",
            background=background,
            row_period=row_period,
            column_period=column_period,
        )
        instructions = (
            (periodic,)
            if family == "broadcast_panel_lattice_periodic"
            else (periodic, Instruction.create("rotate180"))
        )
    elif family in {
        "paint_bbox_contacts",
        "paint_bbox_contacts_rotate180",
    }:
        background = _bbox_contact_parameters(family, seed)
        contact = Instruction.create(
            "paint_bbox_contacts", background=background
        )
        instructions = (
            (contact,)
            if family == "paint_bbox_contacts"
            else (contact, Instruction.create("rotate180"))
        )
    else:
        raise ValueError(f"unknown synthetic family: {family}")
    return Program.create(instructions)


def _input_for_family(
    family: str,
    rng: random.Random,
    *,
    program: Program,
    example_index: int,
) -> Grid:
    if family in {"rotate90", "flip_horizontal"}:
        return _generic_grid(rng, require_non_square=family == "rotate90")
    if family in {"recolor", "rotate90_recolor"}:
        grid = _recolor_grid(rng, old=1)
        if family == "rotate90_recolor" and len(grid) == len(grid[0]):
            return _recolor_grid(rng, old=1)
        return grid
    if family in {"crop_non_background", "crop_rotate90"}:
        return _crop_grid(rng)
    if family == "keep_largest_object":
        return _largest_grid(rng)
    if family in {"scale_pixels_2x2", "tile_horizontal_2"}:
        return _shape_grid(rng)
    if family in {"overlay_panel_grid", "overlay_panel_rotate90"}:
        return _panel_grid(rng)
    if family in {
        "broadcast_panel_sequence_d4",
        "broadcast_panel_sequence_d4_flip_horizontal",
    }:
        arguments = program.instructions[0].arguments
        return _panel_sequence_grid(
            rng,
            background=arguments["background"],
            step=D4Step(arguments["step"]),
            example_index=example_index,
        )
    if family in {
        "broadcast_panel_lattice_periodic",
        "broadcast_panel_lattice_periodic_rotate180",
    }:
        arguments = program.instructions[0].arguments
        return _panel_lattice_periodic_grid(
            rng,
            background=arguments["background"],
            row_period=arguments["row_period"],
            column_period=arguments["column_period"],
            example_index=example_index,
        )
    if family in {
        "paint_bbox_contacts",
        "paint_bbox_contacts_rotate180",
    }:
        return _bbox_contact_grid(
            rng,
            background=program.instructions[0].arguments["background"],
            example_index=example_index,
        )
    raise ValueError(f"unknown synthetic family: {family}")


def _task_payload(
    train: tuple[ARCPair, ...], test: tuple[ARCPair, ...]
) -> dict[str, object]:
    return {
        "train": [
            {"input": grid_to_lists(pair.input), "output": grid_to_lists(pair.output)}
            for pair in train
        ],
        "test": [
            {"input": grid_to_lists(pair.input), "output": grid_to_lists(pair.output)}
            for pair in test
        ],
    }


def generate_case(*, family: str, seed: int) -> SyntheticCase:
    rng = random.Random(seed)
    program = _program_for_family(family, seed=seed)
    pairs: list[ARCPair] = []
    for example_index in range(5):
        for _attempt in range(100):
            input_grid = _input_for_family(
                family,
                rng,
                program=program,
                example_index=example_index,
            )
            outcome = execute_program(program, input_grid)
            if not outcome.ok or outcome.output is None or outcome.output == input_grid:
                continue
            if family in {
                "broadcast_panel_sequence_d4_flip_horizontal",
                "broadcast_panel_lattice_periodic_rotate180",
                "paint_bbox_contacts_rotate180",
            }:
                prefix = execute_program(
                    Program.create((program.instructions[0],)), input_grid
                )
                suffix = execute_program(
                    Program.create((program.instructions[1],)), input_grid
                )
                if outcome.output in {prefix.output, suffix.output}:
                    continue
            pairs.append(ARCPair(input_grid, outcome.output))
            break
        else:
            raise RuntimeError(f"could not generate non-degenerate {family} case")
    train = tuple(pairs[:3])
    test = tuple(pairs[3:])
    payload = _task_payload(train, test)
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    content_hash = hashlib.sha256(serialized).hexdigest()
    blind_task = BlindTask.from_observations(
        train=train, test_inputs=tuple(pair.input for pair in test)
    )
    task = ARCTask(
        task_id=blind_task.task_id,
        train=train,
        test=test,
        source_path=f"synthetic://{SYNTHETIC_SUITE_VERSION}/{content_hash[:16]}",
        source_sha256=content_hash,
    )
    return SyntheticCase(
        task=task,
        blind_task=blind_task,
        family=family,
        stratum="atomic" if program.node_count == 1 else "composition",
        seed=seed,
        generator_program=program,
    )


def generate_suite(*, seed: int = 0, tasks_per_family: int = 3) -> tuple[SyntheticCase, ...]:
    if tasks_per_family <= 0:
        raise ValueError("tasks_per_family must be positive")
    families = (
        "rotate90",
        "flip_horizontal",
        "recolor",
        "rotate90_recolor",
        "crop_non_background",
        "crop_rotate90",
        "keep_largest_object",
        "scale_pixels_2x2",
        "tile_horizontal_2",
        "overlay_panel_grid",
        "overlay_panel_rotate90",
        "broadcast_panel_sequence_d4",
        "broadcast_panel_sequence_d4_flip_horizontal",
        "broadcast_panel_lattice_periodic",
        "broadcast_panel_lattice_periodic_rotate180",
        "paint_bbox_contacts",
        "paint_bbox_contacts_rotate180",
    )
    cases = [
        generate_case(family=family, seed=seed * 100_000 + family_index * 1_000 + index)
        for family_index, family in enumerate(families)
        for index in range(tasks_per_family)
    ]
    return tuple(cases)
