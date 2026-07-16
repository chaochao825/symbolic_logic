"""Label-isolated ARC canvas, shape, augmentation, and object features.

All selectors in this module consume demonstrations and test inputs only.
Held-out outputs are used by the runner's evaluator after prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np

from arc_ca import encode_binary4, predict_direct, select_demo_rule
from arc_data import ARC_MAX_SIDE, ArcExample


COLOR_BITS = 4
COORD_BITS = 5
SHAPE_KINDS = ("same", "bbox", "transpose", "scale", "constant")


def modal_color(grid: object) -> int:
    values = np.asarray(grid, dtype=np.uint8).reshape(-1)
    return int(np.argmax(np.bincount(values, minlength=10)))


def encode_color_channels(grid: object) -> np.ndarray:
    """Return little-endian binary4 color channels as ``[4,H,W]``."""
    bits = encode_binary4(np.asarray(grid, dtype=np.uint8))
    return np.moveaxis(bits, -1, 0).astype(np.float32)


def decode_color_channels(bits: object, fallback: object) -> tuple[np.ndarray, np.ndarray]:
    """Decode hard ``[4,H,W]`` bits and replace invalid codes by fallback."""
    array = np.asarray(bits)
    if array.ndim != 3 or array.shape[0] != COLOR_BITS:
        raise ValueError("binary color state must have shape [4,H,W]")
    if not bool(np.all((array == 0) | (array == 1))):
        raise ValueError("hard color state must contain only zero or one")
    values = np.sum(array.astype(np.uint8) << np.arange(COLOR_BITS, dtype=np.uint8)[:, None, None], axis=0).astype(np.uint8)
    valid = values < 10
    fallback_array = np.asarray(fallback, dtype=np.uint8)
    if fallback_array.shape != values.shape:
        raise ValueError("fallback grid shape mismatch")
    return np.where(valid, values, fallback_array).astype(np.uint8), valid


def _int_bits(value: int, width: int) -> np.ndarray:
    if value < 0 or value >= 1 << width:
        raise ValueError(f"value {value} does not fit in {width} bits")
    return ((int(value) >> np.arange(width)) & 1).astype(np.float32)


def _foreground_bbox_shape(grid: np.ndarray) -> tuple[int, int]:
    background = modal_color(grid)
    rows, columns = np.where(grid != background)
    if not len(rows):
        return tuple(int(value) for value in grid.shape)
    return int(rows.max() - rows.min() + 1), int(columns.max() - columns.min() + 1)


@dataclass(frozen=True)
class ShapeProgram:
    kind: str
    params: tuple[int, ...]
    description_bits: int

    @property
    def name(self) -> str:
        suffix = ":".join(str(value) for value in self.params)
        return self.kind + (":" + suffix if suffix else "")

    def predict_shape(self, input_grid: object) -> tuple[int, int]:
        grid = np.asarray(input_grid)
        rows, columns = (int(value) for value in grid.shape)
        if self.kind == "same":
            result = (rows, columns)
        elif self.kind == "transpose":
            result = (columns, rows)
        elif self.kind == "bbox":
            result = _foreground_bbox_shape(grid)
        elif self.kind == "scale":
            row_scale, column_scale = self.params
            result = (rows * row_scale, columns * column_scale)
        elif self.kind == "constant":
            result = tuple(int(value) for value in self.params)
        else:
            raise ValueError(f"unknown shape program {self.kind}")
        if min(result) < 1 or max(result) > ARC_MAX_SIDE:
            raise ValueError(f"predicted output shape {result} is outside ARC limits")
        return result


def rank_shape_programs(examples: Sequence[ArcExample]) -> tuple[ShapeProgram, ...]:
    """Return all predeclared programs that explain demonstration shapes."""
    if not examples:
        raise ValueError("shape inference requires demonstrations")
    candidates: list[ShapeProgram] = [
        ShapeProgram("same", (), 3),
        ShapeProgram("bbox", (), 5),
        ShapeProgram("transpose", (), 5),
    ]
    output_shapes = {tuple(int(value) for value in example.output_grid.shape) for example in examples}
    if len(output_shapes) == 1:
        candidates.append(ShapeProgram("constant", next(iter(output_shapes)), 3 + 2 * COORD_BITS))
    scales: set[tuple[int, int]] = set()
    valid_scale = True
    for example in examples:
        in_rows, in_columns = example.input_grid.shape
        out_rows, out_columns = example.output_grid.shape
        if out_rows % in_rows or out_columns % in_columns:
            valid_scale = False
            break
        scales.add((out_rows // in_rows, out_columns // in_columns))
    if valid_scale and len(scales) == 1 and min(next(iter(scales))) >= 1:
        candidates.append(ShapeProgram("scale", next(iter(scales)), 3 + COORD_BITS))

    accepted: dict[str, ShapeProgram] = {}
    for candidate in candidates:
        try:
            correct = all(candidate.predict_shape(example.input_grid) == example.output_grid.shape for example in examples)
        except ValueError:
            correct = False
        if correct:
            accepted[candidate.name] = candidate
    return tuple(sorted(accepted.values(), key=lambda item: (item.description_bits, item.name)))


def select_shape_program(examples: Sequence[ArcExample]) -> ShapeProgram | None:
    ranked = rank_shape_programs(examples)
    return ranked[0] if ranked else None


def d4_arrays(array: object) -> tuple[np.ndarray, ...]:
    """Return the eight dihedral transforms in a fixed public order."""
    source = np.asarray(array)
    rotations = tuple(np.rot90(source, k).copy() for k in range(4))
    reflected = np.fliplr(source)
    return rotations + tuple(np.rot90(reflected, k).copy() for k in range(4))


def augment_examples(examples: Sequence[ArcExample], policy: str) -> tuple[ArcExample, ...]:
    if policy == "none":
        return tuple(examples)
    if policy != "d4":
        raise ValueError("augmentation policy must be 'none' or 'd4'")
    augmented: list[ArcExample] = []
    for example in examples:
        inputs = d4_arrays(example.input_grid)
        outputs = d4_arrays(example.output_grid)
        augmented.extend(ArcExample(source, target) for source, target in zip(inputs, outputs))
    return tuple(augmented)


@dataclass(frozen=True)
class AugmentationSelection:
    policy: str
    lodo_pair_exact: float
    lodo_cell_accuracy: float
    description_bits: int


def _augmentation_lodo(examples: Sequence[ArcExample], policy: str) -> tuple[float, float]:
    if len(examples) < 2:
        return 0.0, 0.0
    exact: list[bool] = []
    correct = 0
    cells = 0
    for held_out in range(len(examples)):
        training = tuple(example for index, example in enumerate(examples) if index != held_out)
        selected = select_demo_rule(augment_examples(training, policy))
        reference = np.asarray(examples[held_out].output_grid)
        prediction = (
            predict_direct(selected.rule, examples[held_out].input_grid)
            if selected.rule is not None
            else np.asarray(examples[held_out].input_grid)
        )
        match = prediction.shape == reference.shape
        exact.append(bool(match and np.array_equal(prediction, reference)))
        if match:
            correct += int(np.sum(prediction == reference))
            cells += int(reference.size)
    return float(np.mean(exact)), float(correct / cells) if cells else 0.0


def select_augmentation_policy(examples: Sequence[ArcExample]) -> AugmentationSelection:
    candidates = []
    for policy, bits in (("none", 1), ("d4", 2)):
        pair, cell = _augmentation_lodo(examples, policy)
        candidates.append(AugmentationSelection(policy, pair, cell, bits))
    candidates.sort(key=lambda item: (-item.lodo_pair_exact, -item.lodo_cell_accuracy, item.description_bits, item.policy))
    return candidates[0]


def _component_features(grid: np.ndarray) -> np.ndarray:
    """Rasterize declared same-color 4-connected component predicates."""
    rows, columns = grid.shape
    background = modal_color(grid)
    features = np.zeros((9, rows, columns), dtype=np.float32)
    foreground = grid != background
    features[0] = foreground
    visited = np.zeros_like(foreground, dtype=bool)
    for start_row in range(rows):
        for start_column in range(columns):
            if visited[start_row, start_column] or not foreground[start_row, start_column]:
                continue
            color = int(grid[start_row, start_column])
            stack = [(start_row, start_column)]
            visited[start_row, start_column] = True
            component: list[tuple[int, int]] = []
            while stack:
                row, column = stack.pop()
                component.append((row, column))
                for delta_row, delta_column in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    other_row, other_column = row + delta_row, column + delta_column
                    if (
                        0 <= other_row < rows
                        and 0 <= other_column < columns
                        and not visited[other_row, other_column]
                        and int(grid[other_row, other_column]) == color
                    ):
                        visited[other_row, other_column] = True
                        stack.append((other_row, other_column))
            component_rows = [item[0] for item in component]
            component_columns = [item[1] for item in component]
            top, bottom = min(component_rows), max(component_rows)
            left, right = min(component_columns), max(component_columns)
            size = len(component)
            component_set = set(component)
            for row, column in component:
                perimeter = any(
                    (row + dr, column + dc) not in component_set
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                )
                features[1, row, column] = float(perimeter)
                features[2, row, column] = float((row, column) == (top, left))
                features[3, row, column] = float(size == 1)
                features[4, row, column] = float(size <= 4)
                features[5, row, column] = float(row == top)
                features[6, row, column] = float(row == bottom)
                features[7, row, column] = float(column == left)
                features[8, row, column] = float(column == right)
    return features


def _task_context(examples: Sequence[ArcExample], shape_program: ShapeProgram) -> np.ndarray:
    input_palette = np.zeros(10, dtype=np.float32)
    output_palette = np.zeros(10, dtype=np.float32)
    backgrounds = np.zeros(10, dtype=np.int64)
    changed_pairs: set[tuple[int, int]] = set()
    for example in examples:
        input_palette[np.unique(example.input_grid)] = 1
        output_palette[np.unique(example.output_grid)] = 1
        backgrounds[modal_color(example.input_grid)] += 1
        if example.input_grid.shape == example.output_grid.shape:
            changed_pairs.update(
                (int(source), int(target))
                for source, target in zip(example.input_grid.reshape(-1), example.output_grid.reshape(-1))
                if source != target
            )
    background_bits = _int_bits(int(np.argmax(backgrounds)), COLOR_BITS)
    shape_bits = np.zeros(len(SHAPE_KINDS), dtype=np.float32)
    shape_bits[SHAPE_KINDS.index(shape_program.kind)] = 1
    change_bits = np.asarray(
        [len(changed_pairs) == 0, len(changed_pairs) == 1, len(changed_pairs) > 1],
        dtype=np.float32,
    )
    return np.concatenate((background_bits, input_palette, output_palette, shape_bits, change_bits))


def _input_global_context(grid: np.ndarray) -> np.ndarray:
    palette = np.zeros(10, dtype=np.float32)
    palette[np.unique(grid)] = 1
    background = _int_bits(modal_color(grid), COLOR_BITS)
    foreground_count = int(np.sum(grid != modal_color(grid)))
    count_bits = _int_bits(min(foreground_count, 31), COORD_BITS)
    return np.concatenate((palette, background, count_bits))


@dataclass(frozen=True)
class CanvasExample:
    initial_state: np.ndarray
    static_features: np.ndarray
    target_bits: np.ndarray | None
    target_mask: np.ndarray
    update_mask: np.ndarray
    fallback_colors: np.ndarray
    output_shape: tuple[int, int]
    changed_mask: np.ndarray | None


def infer_workspace_shape(
    examples: Sequence[ArcExample],
    test_inputs: Sequence[np.ndarray],
    shape_program: ShapeProgram,
    *,
    augmentation: str = "none",
) -> tuple[int, int]:
    grids: list[tuple[int, int]] = []
    augmented = augment_examples(examples, augmentation)
    for example in augmented:
        grids.extend((example.input_grid.shape, example.output_grid.shape))
    for grid in test_inputs:
        grids.extend((grid.shape, shape_program.predict_shape(grid)))
    rows = max(int(shape[0]) for shape in grids)
    columns = max(int(shape[1]) for shape in grids)
    if max(rows, columns) > ARC_MAX_SIDE:
        raise ValueError("workspace exceeds ARC limits")
    return rows, columns


def build_canvas_example(
    input_grid: object,
    output_shape: tuple[int, int],
    canvas_shape: tuple[int, int],
    examples: Sequence[ArcExample],
    shape_program: ShapeProgram,
    *,
    hidden_bits: int,
    target_grid: object | None = None,
    include_original: bool = False,
    include_geometry: bool = False,
    include_objects: bool = False,
    include_context: bool = False,
) -> CanvasExample:
    grid = np.asarray(input_grid, dtype=np.uint8)
    canvas_rows, canvas_columns = canvas_shape
    if grid.ndim != 2 or grid.shape[0] > canvas_rows or grid.shape[1] > canvas_columns:
        raise ValueError("input does not fit workspace")
    if output_shape[0] > canvas_rows or output_shape[1] > canvas_columns:
        raise ValueError("output does not fit workspace")

    background = modal_color(grid)
    fallback = np.full(canvas_shape, background, dtype=np.uint8)
    fallback[: grid.shape[0], : grid.shape[1]] = grid
    dynamic = np.zeros((COLOR_BITS + int(hidden_bits), canvas_rows, canvas_columns), dtype=np.float32)
    dynamic[:COLOR_BITS, : grid.shape[0], : grid.shape[1]] = encode_color_channels(grid)

    input_mask = np.zeros(canvas_shape, dtype=np.float32)
    input_mask[: grid.shape[0], : grid.shape[1]] = 1
    output_mask = np.zeros(canvas_shape, dtype=np.float32)
    output_mask[: output_shape[0], : output_shape[1]] = 1
    update_mask = np.maximum(input_mask, output_mask)
    static: list[np.ndarray] = [input_mask[None], output_mask[None]]

    if include_original:
        original = np.zeros((COLOR_BITS, canvas_rows, canvas_columns), dtype=np.float32)
        original[:, : grid.shape[0], : grid.shape[1]] = encode_color_channels(grid)
        static.append(original)

    if include_geometry:
        row_grid = np.broadcast_to(np.arange(canvas_rows)[:, None], canvas_shape)
        column_grid = np.broadcast_to(np.arange(canvas_columns)[None, :], canvas_shape)
        row_bits = ((row_grid[None] >> np.arange(COORD_BITS)[:, None, None]) & 1).astype(np.float32)
        column_bits = ((column_grid[None] >> np.arange(COORD_BITS)[:, None, None]) & 1).astype(np.float32)
        border = np.zeros((4, canvas_rows, canvas_columns), dtype=np.float32)
        border[0, 0, : output_shape[1]] = 1
        border[1, output_shape[0] - 1, : output_shape[1]] = 1
        border[2, : output_shape[0], 0] = 1
        border[3, : output_shape[0], output_shape[1] - 1] = 1
        size_bits = np.concatenate((_int_bits(output_shape[0], COORD_BITS), _int_bits(output_shape[1], COORD_BITS)))
        size_raster = np.broadcast_to(size_bits[:, None, None], (len(size_bits), canvas_rows, canvas_columns)).copy()
        static.extend((row_bits, column_bits, border, size_raster))

    if include_objects:
        objects = np.zeros((9, canvas_rows, canvas_columns), dtype=np.float32)
        objects[:, : grid.shape[0], : grid.shape[1]] = _component_features(grid)
        static.append(objects)

    if include_context:
        context = np.concatenate((_task_context(examples, shape_program), _input_global_context(grid)))
        static.append(np.broadcast_to(context[:, None, None], (len(context), canvas_rows, canvas_columns)).copy())

    target_bits: np.ndarray | None = None
    changed_mask: np.ndarray | None = None
    if target_grid is not None:
        target = np.asarray(target_grid, dtype=np.uint8)
        if target.shape != output_shape:
            raise ValueError("target grid disagrees with declared output shape")
        target_bits = np.zeros((COLOR_BITS, canvas_rows, canvas_columns), dtype=np.float32)
        target_bits[:, : output_shape[0], : output_shape[1]] = encode_color_channels(target)
        changed_mask = np.zeros(canvas_shape, dtype=np.float32)
        overlap_rows = min(grid.shape[0], target.shape[0])
        overlap_columns = min(grid.shape[1], target.shape[1])
        changed_mask[: target.shape[0], : target.shape[1]] = 1
        changed_mask[:overlap_rows, :overlap_columns] = (grid[:overlap_rows, :overlap_columns] != target[:overlap_rows, :overlap_columns])

    return CanvasExample(
        initial_state=dynamic,
        static_features=np.concatenate(static, axis=0).astype(np.float32),
        target_bits=target_bits,
        target_mask=output_mask,
        update_mask=update_mask,
        fallback_colors=fallback,
        output_shape=tuple(int(value) for value in output_shape),
        changed_mask=changed_mask,
    )


def paired_grid_metrics(references: Iterable[np.ndarray], predictions: Iterable[np.ndarray]) -> dict[str, float]:
    references = list(references)
    predictions = list(predictions)
    if len(references) != len(predictions):
        raise ValueError("reference and prediction counts differ")
    exact = []
    correct = 0
    cells = 0
    for reference, prediction in zip(references, predictions):
        reference = np.asarray(reference)
        prediction = np.asarray(prediction)
        same = reference.shape == prediction.shape
        exact.append(bool(same and np.array_equal(reference, prediction)))
        if same:
            correct += int(np.sum(reference == prediction))
            cells += int(reference.size)
    return {
        "pair_exact_count": float(sum(exact)),
        "pair_exact_rate": float(np.mean(exact)) if exact else math.nan,
        "task_exact": float(bool(exact) and all(exact)),
        "cell_accuracy": float(correct / cells) if cells else 0.0,
    }
