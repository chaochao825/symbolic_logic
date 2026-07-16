"""Run the frozen categorical-CA and hard-gate compatibility experiments.

The ARC selector receives :class:`ArcProblem` only.  Test labels are held by a
separate evaluator object.  Public-evaluation mode additionally requires a
clean committed source tree and emits a one-run receipt in metadata.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from arc_ca import (
    BINARY4_BITS,
    BOUNDARY_COLOR,
    FROZEN_NEIGHBORHOODS,
    NeighborhoodSpec,
    SparseCategoricalRule,
    compiled_gate_upper_bounds,
    find_oracle_local_rule,
    predict_binary4,
    predict_direct,
    predict_onehot,
    rollout_rule,
    rule_from_transition_table,
    select_demo_rule,
    support_fraction,
)
from arc_data import (
    ArcExample,
    ArcLabels,
    ArcProblem,
    load_arc_split,
    sha256_file,
    split_digest,
    task_id_digest,
)
from arc_ca_programs import select_demo_program


ROOT = Path(__file__).resolve().parents[1]
ARC_AGI_1_COMMIT = "aa922be204204ec148a1137fe6ed4d34ddde812b"
ARC_AGI_2_COMMIT = "f3283f727488ad98fe575ea6a5ac981e4a188e49"
EXPECTED_ORIGIN_URLS = {
    "git@github.com:chaochao825/symbolic_logic.git",
    "https://github.com/chaochao825/symbolic_logic.git",
}
CONFIG_SCHEMA_VERSION = 2
DECLARED_CA_WRAPPER_BITS = 8
# Public container prefix: dimension (1), boundary policy (2), synchronous
# schedule (1), recurrence class (1), and lattice-interface family (3).
FROZEN_SPLITS = {
    ("ARC-AGI-1", "training"): {
        "count": 400,
        "split_digest": "20f5ece7b6ee70e6f058e8ba1d077764d69c6452b52dc50f0856a63be3a552de",
        "task_id_digest": "c4e1b145daad71b4f02ac3482be215f1ab07863b67f13fc1bde1572f501ff75a",
    },
    ("ARC-AGI-1", "evaluation"): {
        "count": 400,
        "split_digest": "b27020a211f2d55f5556cdc27e30b42771394ecaecfc2e0a53da3061b6f5de76",
        "task_id_digest": "911bf838d8843606d5fc1ba0550a0c23670027bf8f6e32e4e4fbf93854957ddd",
    },
    ("ARC-AGI-2", "training"): {
        "count": 1000,
        "split_digest": "10a0146ab6374f18f44a770b37154ccb6d579fddb9a321eb3bf936666bc85aa8",
        "task_id_digest": "6031afc0c6f31a05793362bbf162b4d75448efdcf43afb1e79d4bc54ada9c2a2",
    },
    ("ARC-AGI-2", "evaluation"): {
        "count": 120,
        "split_digest": "ddb0a9f8794407f47ff9dc692ef7d3e578ddd503c92f555a4f38b509b0ba05aa",
        "task_id_digest": "54ca25cdc4444e5669e272e25cbe301bbfe3aa81da8f555126095153aff69425",
    },
}


def git_state() -> dict[str, object]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        porcelain = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
        remote = subprocess.check_output(
            ["git", "remote", "get-url", "origin"], cwd=ROOT, text=True
        ).strip()
        containing = subprocess.check_output(
            ["git", "branch", "-r", "--contains", commit], cwd=ROOT, text=True
        ).splitlines()
        return {
            "commit": commit,
            "dirty": bool(porcelain),
            "porcelain": porcelain.splitlines(),
            "origin": remote,
            "origin_contains_head": any(line.strip().startswith("origin/") for line in containing),
            "origin_is_expected_repository": remote in EXPECTED_ORIGIN_URLS,
        }
    except (OSError, subprocess.CalledProcessError):
        return {
            "commit": "unavailable",
            "dirty": None,
            "porcelain": [],
            "origin": "unavailable",
            "origin_contains_head": False,
            "origin_is_expected_repository": False,
        }


def dataset_git_state(root: Path) -> dict[str, object]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        porcelain = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True
        ).strip()
        return {"commit": commit, "dirty": bool(porcelain), "porcelain": porcelain.splitlines()}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unavailable", "dirty": None, "porcelain": []}


def _assert_label_alignment(problem: ArcProblem, labels: ArcLabels) -> None:
    if problem.task_id != labels.task_id:
        raise ValueError(f"problem/label task-id mismatch: {problem.task_id} != {labels.task_id}")
    if len(problem.test_inputs) != len(labels.test_outputs):
        raise ValueError(
            f"{problem.task_id}: {len(problem.test_inputs)} inputs but {len(labels.test_outputs)} labels"
        )


def _reserve_receipt(path: Path, payload: dict[str, object]) -> None:
    """Atomically reserve the declared one-shot public-evaluation JSONL ledger."""
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"), sort_keys=True)
        handle.write("\n")


def _complete_receipt(path: Path, output_dir: Path, finished_utc: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "status": "completed",
                    "finished_utc": finished_utc,
                    "metadata_sha256": sha256_file(output_dir / "metadata.json"),
                },
                sort_keys=True,
            )
            + "\n"
        )


def _paired_metrics(
    references: Sequence[np.ndarray | None],
    predictions: Sequence[np.ndarray],
) -> dict[str, float | int]:
    if len(references) != len(predictions):
        raise ValueError(f"reference/prediction length mismatch: {len(references)} != {len(predictions)}")
    exact: list[bool] = []
    matching_shape_pairs = 0
    correct = 0
    cells = 0
    for reference, prediction in zip(references, predictions):
        if reference is None:
            continue
        same_shape = reference.shape == prediction.shape
        exact.append(bool(same_shape and np.array_equal(reference, prediction)))
        if same_shape:
            matching_shape_pairs += 1
            correct += int(np.sum(reference == prediction))
            cells += int(reference.size)
    return {
        "labeled_pairs": len(exact),
        "pair_exact_count": int(sum(exact)),
        "pair_exact_rate": float(np.mean(exact)) if exact else math.nan,
        "task_exact": float(bool(exact) and all(exact)),
        "matching_shape_pairs": matching_shape_pairs,
        "cell_accuracy": float(correct / cells) if cells else math.nan,
        "scored_cells": cells,
    }


def _top2_metrics(
    references: Sequence[np.ndarray | None],
    first: Sequence[np.ndarray],
    second: Sequence[np.ndarray] | None,
) -> dict[str, float | int]:
    if len(references) != len(first) or (second is not None and len(references) != len(second)):
        raise ValueError("top-2 reference/prediction length mismatch")
    exact: list[bool] = []
    for index, reference in enumerate(references):
        if reference is None:
            continue
        candidates = [first[index]]
        if second is not None:
            candidates.append(second[index])
        exact.append(any(candidate.shape == reference.shape and np.array_equal(candidate, reference) for candidate in candidates))
    return {
        "pair_exact_count_pass_at_2": int(sum(exact)),
        "pair_exact_rate_pass_at_2": float(np.mean(exact)) if exact else math.nan,
        "task_exact_pass_at_2": float(bool(exact) and all(exact)),
    }


def _benchmark(function: Callable[[], object], repeats: int) -> tuple[float, float]:
    samples: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        samples.append(time.perf_counter() - started)
    ordered = sorted(samples)
    index95 = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    return float(statistics.median(samples)), float(ordered[index95])


def _novel_test_colors(problem: ArcProblem) -> tuple[int, ...]:
    train_colors = set(
        int(value)
        for example in problem.demonstrations
        for grid in (example.input_grid, example.output_grid)
        for value in np.unique(grid)
    )
    test_colors = set(int(value) for grid in problem.test_inputs for value in np.unique(grid))
    return tuple(sorted(test_colors - train_colors))


def evaluate_arc_task(
    problem: ArcProblem,
    labels: ArcLabels,
    *,
    repeats: int,
    max_compiled_entries: int | None,
) -> tuple[dict[str, object], dict[str, object] | None, dict[str, object]]:
    """Select from demonstrations, then score through the isolated evaluator."""
    _assert_label_alignment(problem, labels)
    selected = select_demo_rule(problem.demonstrations)
    references = list(labels.test_outputs)
    identity_predictions = [np.asarray(grid).copy() for grid in problem.test_inputs]
    identity_metrics = _paired_metrics(references, identity_predictions)

    oracle_rule: SparseCategoricalRule | None = None
    labels_complete = bool(labels.test_outputs) and all(output is not None for output in labels.test_outputs)
    if labels_complete:
        oracle_examples = list(problem.demonstrations)
        oracle_examples.extend(
            ArcExample(input_grid, output_grid)
            for input_grid, output_grid in zip(problem.test_inputs, labels.test_outputs)
            if output_grid is not None
        )
        oracle_rule = find_oracle_local_rule(oracle_examples)

    row: dict[str, object] = {
        "task_id": problem.task_id,
        "selection_status": selected.status,
        "demonstration_pairs": len(problem.demonstrations),
        "test_pairs": len(problem.test_inputs),
        "candidates_considered": selected.candidates_considered,
        "exact_training_candidates": selected.exact_training_candidates,
        "collision_candidates": selected.collision_candidates,
        "selection_seconds": selected.search_seconds,
        "lodo_pair_exact": selected.lodo_pair_exact,
        "lodo_cell_accuracy": selected.lodo_cell_accuracy,
        "novel_test_colors": ";".join(map(str, _novel_test_colors(problem))),
        "novel_test_color_count": len(_novel_test_colors(problem)),
        "identity_pair_exact_rate": identity_metrics["pair_exact_rate"],
        "identity_task_exact": identity_metrics["task_exact"],
        "posthoc_oracle_local_exists": float(oracle_rule is not None) if labels_complete else math.nan,
        "posthoc_oracle_neighborhood": oracle_rule.spec.name if oracle_rule is not None else "",
        "posthoc_oracle_model_bits": oracle_rule.model_description_bits if oracle_rule is not None else math.nan,
    }
    if selected.rule is None:
        row.update(
            {
                "outcome": "abstained",
                "selected_neighborhood": "",
                "offset_count": 0,
                "exception_count": 0,
                "model_description_bits": math.nan,
                "wrapper_description_bits": math.nan,
                "total_description_bits": math.nan,
                "mean_test_support": math.nan,
                "min_test_support": math.nan,
                "labeled_pairs": identity_metrics["labeled_pairs"],
                "pair_exact_count": 0,
                "pair_exact_rate": 0.0 if identity_metrics["labeled_pairs"] else math.nan,
                "task_exact": 0.0,
                "pair_exact_count_pass_at_2": 0,
                "pair_exact_rate_pass_at_2": 0.0 if identity_metrics["labeled_pairs"] else math.nan,
                "task_exact_pass_at_2": 0.0,
                "second_neighborhood": "",
                "matching_shape_pairs": 0,
                "cell_accuracy": math.nan,
                "scored_cells": 0,
                "compiled_checked": 0.0,
                "compiled_equivalent": math.nan,
                "binary4_invalid_rate": math.nan,
                "onehot_invalid_rate": math.nan,
            }
        )
        return row, None, {
            "task_id": problem.task_id,
            "attempt_1": None,
            "attempt_2": None,
            "status": "abstained",
        }

    rule = selected.rule
    direct_predictions = [predict_direct(rule, grid) for grid in problem.test_inputs]
    direct_metrics = _paired_metrics(references, direct_predictions)
    second_rule: SparseCategoricalRule | None = None
    second_predictions: list[np.ndarray] | None = None
    first_signature = tuple(prediction.tobytes() for prediction in direct_predictions)
    for alternative in selected.ranked_rules[1:]:
        predictions = [predict_direct(alternative, grid) for grid in problem.test_inputs]
        if tuple(prediction.tobytes() for prediction in predictions) != first_signature:
            second_rule = alternative
            second_predictions = predictions
            break
    top2 = _top2_metrics(references, direct_predictions, second_predictions)
    supports = [support_fraction(rule, grid) for grid in problem.test_inputs]
    compiled_checked = max_compiled_entries is None or rule.exception_count <= max_compiled_entries
    compiled_equivalent = math.nan
    binary_invalid_rate = math.nan
    onehot_invalid_rate = math.nan
    binary_predictions: list[np.ndarray] = []
    onehot_predictions: list[np.ndarray] = []
    if compiled_checked:
        binary_valid: list[np.ndarray] = []
        onehot_valid: list[np.ndarray] = []
        for grid in problem.test_inputs:
            binary, valid4 = predict_binary4(rule, grid)
            onehot, valid10 = predict_onehot(rule, grid)
            binary_predictions.append(binary)
            onehot_predictions.append(onehot)
            binary_valid.append(valid4)
            onehot_valid.append(valid10)
        compiled_equivalent = float(
            all(np.array_equal(a, b) and np.array_equal(a, c) for a, b, c in zip(direct_predictions, binary_predictions, onehot_predictions))
        )
        binary_invalid_rate = float(1.0 - np.mean(np.concatenate([valid.reshape(-1) for valid in binary_valid])))
        onehot_invalid_rate = float(1.0 - np.mean(np.concatenate([valid.reshape(-1) for valid in onehot_valid])))
        if compiled_equivalent != 1.0:
            raise AssertionError(f"compiled codec disagrees with direct rule on {problem.task_id}")
    row.update(
        {
            "outcome": "solved" if direct_metrics["task_exact"] == 1.0 else "predicted_incorrect",
            "selected_neighborhood": rule.spec.name,
            "offset_count": len(rule.spec.offsets),
            "exception_count": rule.exception_count,
            "model_description_bits": rule.model_description_bits,
            "wrapper_description_bits": DECLARED_CA_WRAPPER_BITS,
            "total_description_bits": rule.model_description_bits + DECLARED_CA_WRAPPER_BITS,
            "mean_test_support": float(np.mean(supports)),
            "min_test_support": float(np.min(supports)),
            **direct_metrics,
            **top2,
            "second_neighborhood": second_rule.spec.name if second_rule is not None else "",
            "compiled_checked": float(compiled_checked),
            "compiled_equivalent": compiled_equivalent,
            "binary4_invalid_rate": binary_invalid_rate,
            "onehot_invalid_rate": onehot_invalid_rate,
        }
    )

    direct_median, direct_p95 = _benchmark(
        lambda: [predict_direct(rule, grid) for grid in problem.test_inputs], repeats
    )
    bounds = compiled_gate_upper_bounds(rule)
    codec_row: dict[str, object] = {
        "task_id": problem.task_id,
        "neighborhood": rule.spec.name,
        "offset_count": len(rule.spec.offsets),
        "exception_count": rule.exception_count,
        "model_description_bits": rule.model_description_bits,
        "wrapper_description_bits": DECLARED_CA_WRAPPER_BITS,
        "total_description_bits": rule.model_description_bits + DECLARED_CA_WRAPPER_BITS,
        **bounds,
        "test_cells": int(sum(grid.size for grid in problem.test_inputs)),
        "binary4_dynamic_gate_upper_bound": int(bounds["binary4_gate_upper_bound"] * sum(grid.size for grid in problem.test_inputs)),
        "onehot_dynamic_gate_upper_bound": int(bounds["onehot_gate_upper_bound"] * sum(grid.size for grid in problem.test_inputs)),
        "compiled_checked": float(compiled_checked),
        "compiled_equivalent": compiled_equivalent,
        "binary4_invalid_rate": binary_invalid_rate,
        "onehot_invalid_rate": onehot_invalid_rate,
        "direct_median_seconds": direct_median,
        "direct_p95_seconds": direct_p95,
        "binary4_median_seconds": math.nan,
        "binary4_p95_seconds": math.nan,
        "onehot_median_seconds": math.nan,
        "onehot_p95_seconds": math.nan,
        "compile_skip_reason": "" if compiled_checked else "exception_budget",
    }
    if compiled_checked:
        codec_row["binary4_median_seconds"], codec_row["binary4_p95_seconds"] = _benchmark(
            lambda: [predict_binary4(rule, grid) for grid in problem.test_inputs], repeats
        )
        codec_row["onehot_median_seconds"], codec_row["onehot_p95_seconds"] = _benchmark(
            lambda: [predict_onehot(rule, grid) for grid in problem.test_inputs], repeats
        )
    submission = {
        "task_id": problem.task_id,
        "attempt_1": [prediction.tolist() for prediction in direct_predictions],
        "attempt_2": (
            [prediction.tolist() for prediction in second_predictions]
            if second_predictions is not None
            else None
        ),
        "status": "predicted",
    }
    return row, codec_row, submission


def evaluate_program_task(
    problem: ArcProblem,
    labels: ArcLabels,
    *,
    repeats: int,
) -> dict[str, object]:
    """Evaluate the independently bounded copy/propagation CA program library."""
    _assert_label_alignment(problem, labels)
    selected = select_demo_program(problem.demonstrations)
    references = list(labels.test_outputs)
    row: dict[str, object] = {
        "task_id": problem.task_id,
        "selection_status": selected.status,
        "candidates_evaluated": selected.candidates_evaluated,
        "exact_training_candidates": selected.exact_candidates,
        "selection_seconds": selected.search_seconds,
        "program": selected.program.name if selected.program is not None else "",
        "program_kind": selected.program.kind if selected.program is not None else "",
        "description_bits": selected.program.description_bits if selected.program is not None else math.nan,
        "wrapper_description_bits": DECLARED_CA_WRAPPER_BITS if selected.program is not None else math.nan,
        "total_description_bits": selected.program.description_bits + DECLARED_CA_WRAPPER_BITS if selected.program is not None else math.nan,
        "onehot_gates_per_cell_upper": selected.program.onehot_gates_per_cell_upper if selected.program is not None else math.nan,
    }
    if selected.program is None:
        labeled = sum(output is not None for output in references)
        row.update(
            {
                "outcome": "abstained",
                "labeled_pairs": labeled,
                "pair_exact_count": 0,
                "pair_exact_rate": 0.0 if labeled else math.nan,
                "task_exact": 0.0,
                "matching_shape_pairs": 0,
                "cell_accuracy": math.nan,
                "scored_cells": 0,
                "mean_steps": math.nan,
                "max_steps": math.nan,
                "stop_reasons": "",
                "dynamic_gate_upper_bound": math.nan,
                "local_dynamic_gate_upper_bound": math.nan,
                "fixed_point_detection_gate_upper_bound": math.nan,
                "median_seconds": math.nan,
                "p95_seconds": math.nan,
            }
        )
        return row
    predictions: list[np.ndarray] = []
    steps: list[int] = []
    stops: list[str] = []
    for grid in problem.test_inputs:
        prediction, executed, reason = selected.program.run(grid)
        predictions.append(prediction)
        steps.append(executed)
        stops.append(reason)
    metrics = _paired_metrics(references, predictions)
    median, p95 = _benchmark(
        lambda: [selected.program.run(grid)[0] for grid in problem.test_inputs], repeats
    )
    local_dynamic = int(
        selected.program.onehot_gates_per_cell_upper
        * sum(grid.size * executed for grid, executed in zip(problem.test_inputs, steps))
    )
    fixed_point_policy = selected.program.kind == "propagate" and selected.program.params[-1] == "fixed_point"
    halt_dynamic = int(
        sum((20 * grid.size - 1) * executed for grid, executed in zip(problem.test_inputs, steps))
        if fixed_point_policy
        else 0
    )
    row.update(
        {
            "outcome": "solved" if metrics["task_exact"] == 1.0 else "predicted_incorrect",
            **metrics,
            "mean_steps": float(np.mean(steps)),
            "max_steps": int(max(steps)),
            "stop_reasons": ";".join(stops),
            "local_dynamic_gate_upper_bound": local_dynamic,
            "fixed_point_detection_gate_upper_bound": halt_dynamic,
            "dynamic_gate_upper_bound": local_dynamic + halt_dynamic,
            "median_seconds": median,
            "p95_seconds": p95,
        }
    )
    return row


def _reference_shift_north(grid: np.ndarray) -> np.ndarray:
    result = grid.copy()
    result[1:] = grid[:-1]
    return result


def _cross_reference(grid: np.ndarray, operation: str) -> np.ndarray:
    padded = np.pad(grid, 1, mode="constant")
    values = np.stack(
        (
            grid,
            padded[:-2, 1:-1],
            padded[1:-1, :-2],
            padded[1:-1, 2:],
            padded[2:, 1:-1],
        ),
        axis=-1,
    )
    if operation == "dilation":
        return np.any(values == 1, axis=-1).astype(np.uint8)
    if operation == "erosion":
        return np.all(values == 1, axis=-1).astype(np.uint8)
    if operation == "parity":
        return (np.sum(values, axis=-1) % 2).astype(np.uint8)
    raise ValueError(operation)


def _binary_cross_rule(operation: str) -> SparseCategoricalRule:
    spec = next(spec for spec in FROZEN_NEIGHBORHOODS if spec.name == "von_neumann_r1")
    table: dict[tuple[int, ...], int] = {}
    for center in (0, 1):
        for neighbors in itertools.product((0, 1, BOUNDARY_COLOR), repeat=4):
            pattern = (center, *neighbors)
            bits = tuple(0 if value == BOUNDARY_COLOR else value for value in pattern)
            if operation == "dilation":
                output = int(any(bits))
            elif operation == "erosion":
                output = int(all(bits))
            elif operation == "parity":
                output = int(sum(bits) % 2)
            else:
                raise ValueError(operation)
            table[pattern] = output
    return rule_from_transition_table(spec, table)


def _synthetic_rules() -> list[
    tuple[str, str, SparseCategoricalRule, Callable[[np.ndarray], np.ndarray], int, int, int]
]:
    center = next(spec for spec in FROZEN_NEIGHBORHOODS if spec.name == "center")
    north = next(spec for spec in FROZEN_NEIGHBORHOODS if spec.name == "center_wire_-1_+0")
    identity = rule_from_transition_table(center, {(color,): color for color in range(10)})
    permutation = np.asarray((0, 2, 3, 4, 5, 6, 7, 8, 9, 1), dtype=np.uint8)
    recolor = rule_from_transition_table(center, {(color,): int(permutation[color]) for color in range(10)})
    copy_north = rule_from_transition_table(
        north,
        {
            (center_color, north_color): center_color if north_color == BOUNDARY_COLOR else north_color
            for center_color in range(10)
            for north_color in range(11)
        },
    )
    conditional = rule_from_transition_table(
        north,
        {
            (center_color, north_color): 3 if center_color == 1 and north_color == 2 else center_color
            for center_color in range(10)
            for north_color in range(11)
        },
    )
    return [
        ("identity_10color", "L0", identity, lambda grid: grid.copy(), 0, 4, 40),
        ("palette_cycle_10color", "L0", recolor, lambda grid: permutation[grid], 0, 8, 40),
        ("copy_north", "L0", copy_north, _reference_shift_north, 0, 8, 400),
        (
            "conditional_recolor",
            "L0",
            conditional,
            lambda grid: np.where((grid == 1) & (np.pad(grid, 1)[:-2, 1:-1] == 2), 3, grid).astype(np.uint8),
            3,
            20,
            400,
        ),
        ("binary_dilation", "L1", _binary_cross_rule("dilation"), lambda grid: _cross_reference(grid, "dilation"), 4, 8, 32),
        ("binary_erosion", "L1", _binary_cross_rule("erosion"), lambda grid: _cross_reference(grid, "erosion"), 4, 8, 32),
        ("binary_parity", "L3", _binary_cross_rule("parity"), lambda grid: _cross_reference(grid, "parity"), 4, 8, 32),
    ]


def run_synthetic(repeats: int) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rng = np.random.default_rng(20_260_716)
    rows: list[dict[str, object]] = []
    codec_rows: list[dict[str, object]] = []
    for name, level, rule, reference_function, known_gates, structured_bits, raw_lut_bits in _synthetic_rules():
        binary_task = name.startswith("binary_")
        grid = rng.integers(0, 2 if binary_task else 10, size=(24, 31), dtype=np.uint8)
        reference = reference_function(grid)
        direct = predict_direct(rule, grid)
        binary, valid4 = predict_binary4(rule, grid)
        onehot, valid10 = predict_onehot(rule, grid)
        direct_exact = float(np.array_equal(reference, direct))
        compiled_exact = float(np.array_equal(direct, binary) and np.array_equal(direct, onehot))
        if direct_exact != 1.0 or compiled_exact != 1.0:
            raise AssertionError(f"synthetic semantic mismatch: {name}")
        bounds = compiled_gate_upper_bounds(rule)
        direct_median, direct_p95 = _benchmark(lambda: predict_direct(rule, grid), repeats)
        binary_median, binary_p95 = _benchmark(lambda: predict_binary4(rule, grid), repeats)
        onehot_median, onehot_p95 = _benchmark(lambda: predict_onehot(rule, grid), repeats)
        rows.append(
            {
                "task": name,
                "level": level,
                "status": "exact",
                "domain_patterns": len(rule.seen_patterns),
                "exception_count": rule.exception_count,
                "known_structured_gate_count": known_gates,
                "raw_interior_lut_bits": raw_lut_bits,
                "raw_description_bits_with_wrapper": raw_lut_bits + DECLARED_CA_WRAPPER_BITS,
                "sparse_model_bits": rule.model_description_bits,
                "known_structured_description_bits": structured_bits,
                "structured_description_bits_with_wrapper": structured_bits + DECLARED_CA_WRAPPER_BITS,
                "preferred_description_route": (
                    "structured_program"
                    if structured_bits + DECLARED_CA_WRAPPER_BITS < raw_lut_bits + DECLARED_CA_WRAPPER_BITS
                    else "raw_lut"
                ),
                "direct_reference_exact": direct_exact,
                "compiled_equivalent": compiled_exact,
                "binary4_invalid_rate": float(1.0 - np.mean(valid4)),
                "onehot_invalid_rate": float(1.0 - np.mean(valid10)),
                **bounds,
            }
        )
        codec_rows.append(
            {
                "task_id": f"synthetic:{name}",
                "neighborhood": rule.spec.name,
                "offset_count": len(rule.spec.offsets),
                "exception_count": rule.exception_count,
                "model_description_bits": rule.model_description_bits,
                "wrapper_description_bits": DECLARED_CA_WRAPPER_BITS,
                "total_description_bits": rule.model_description_bits + DECLARED_CA_WRAPPER_BITS,
                **bounds,
                "test_cells": grid.size,
                "binary4_dynamic_gate_upper_bound": bounds["binary4_gate_upper_bound"] * grid.size,
                "onehot_dynamic_gate_upper_bound": bounds["onehot_gate_upper_bound"] * grid.size,
                "compiled_checked": 1.0,
                "compiled_equivalent": compiled_exact,
                "binary4_invalid_rate": float(1.0 - np.mean(valid4)),
                "onehot_invalid_rate": float(1.0 - np.mean(valid10)),
                "direct_median_seconds": direct_median,
                "direct_p95_seconds": direct_p95,
                "binary4_median_seconds": binary_median,
                "binary4_p95_seconds": binary_p95,
                "onehot_median_seconds": onehot_median,
                "onehot_p95_seconds": onehot_p95,
                "compile_skip_reason": "",
            }
        )

    # Random local LUT: exact representation is trivial, but the predeclared raw
    # 32-bit table beats the generic sparse categorical equality/mux encoding.
    cross = next(spec for spec in FROZEN_NEIGHBORHOODS if spec.name == "von_neumann_r1")
    random_bits = rng.integers(0, 2, size=32, dtype=np.uint8)
    random_lut = random_bits.reshape((2, 2, 2, 2, 2))
    random_table: dict[tuple[int, ...], int] = {}
    for center in (0, 1):
        for neighbors in itertools.product((0, 1, BOUNDARY_COLOR), repeat=4):
            pattern = (center, *neighbors)
            interior = (center, *(0 if value == BOUNDARY_COLOR else value for value in neighbors))
            random_table[pattern] = int(random_lut[interior])
    random_rule = rule_from_transition_table(cross, random_table)
    random_bounds = compiled_gate_upper_bounds(random_rule)
    random_grid = rng.integers(0, 2, size=(24, 31), dtype=np.uint8)
    random_padded = np.pad(random_grid, 1, mode="constant")
    random_reference = random_lut[
        random_grid,
        random_padded[:-2, 1:-1],
        random_padded[1:-1, :-2],
        random_padded[1:-1, 2:],
        random_padded[2:, 1:-1],
    ]
    random_direct = predict_direct(random_rule, random_grid)
    random_binary, random_valid4 = predict_binary4(random_rule, random_grid)
    random_onehot, random_valid10 = predict_onehot(random_rule, random_grid)
    random_direct_exact = float(np.array_equal(random_reference, random_direct))
    random_compiled_exact = float(
        np.array_equal(random_direct, random_binary) and np.array_equal(random_direct, random_onehot)
    )
    if random_direct_exact != 1.0 or random_compiled_exact != 1.0:
        raise AssertionError("random LUT boundary wrapper or compiled codec mismatch")
    random_direct_median, random_direct_p95 = _benchmark(
        lambda: predict_direct(random_rule, random_grid), repeats
    )
    random_binary_median, random_binary_p95 = _benchmark(
        lambda: predict_binary4(random_rule, random_grid), repeats
    )
    random_onehot_median, random_onehot_p95 = _benchmark(
        lambda: predict_onehot(random_rule, random_grid), repeats
    )
    rows.append(
        {
            "task": "random_binary_lut5",
            "level": "N",
            "status": "exact_but_raw_route",
            "domain_patterns": len(random_table),
            "exception_count": random_rule.exception_count,
            "known_structured_gate_count": math.nan,
            "raw_interior_lut_bits": 32,
            "raw_description_bits_with_wrapper": 32 + DECLARED_CA_WRAPPER_BITS,
            "sparse_model_bits": random_rule.model_description_bits,
            "known_structured_description_bits": math.nan,
            "structured_description_bits_with_wrapper": math.nan,
            "preferred_description_route": "raw_lut",
            "direct_reference_exact": random_direct_exact,
            "compiled_equivalent": random_compiled_exact,
            "binary4_invalid_rate": float(1.0 - np.mean(random_valid4)),
            "onehot_invalid_rate": float(1.0 - np.mean(random_valid10)),
            **random_bounds,
        }
    )
    codec_rows.append(
        {
            "task_id": "synthetic:random_binary_lut5",
            "neighborhood": random_rule.spec.name,
            "offset_count": len(random_rule.spec.offsets),
            "exception_count": random_rule.exception_count,
            "model_description_bits": random_rule.model_description_bits,
            "wrapper_description_bits": DECLARED_CA_WRAPPER_BITS,
            "total_description_bits": random_rule.model_description_bits + DECLARED_CA_WRAPPER_BITS,
            **random_bounds,
            "test_cells": random_grid.size,
            "binary4_dynamic_gate_upper_bound": random_bounds["binary4_gate_upper_bound"] * random_grid.size,
            "onehot_dynamic_gate_upper_bound": random_bounds["onehot_gate_upper_bound"] * random_grid.size,
            "compiled_checked": 1.0,
            "compiled_equivalent": random_compiled_exact,
            "binary4_invalid_rate": float(1.0 - np.mean(random_valid4)),
            "onehot_invalid_rate": float(1.0 - np.mean(random_valid10)),
            "direct_median_seconds": random_direct_median,
            "direct_p95_seconds": random_direct_p95,
            "binary4_median_seconds": random_binary_median,
            "binary4_p95_seconds": random_binary_p95,
            "onehot_median_seconds": random_onehot_median,
            "onehot_p95_seconds": random_onehot_p95,
            "compile_skip_reason": "",
        }
    )

    # Global majority creates an explicit local collision: identical radius-two
    # center patches demand different outputs in two 11x11 grids.
    low = np.zeros((11, 11), dtype=np.uint8)
    high = np.ones((11, 11), dtype=np.uint8)
    high[3:8, 3:8] = 0
    global_examples = (
        ArcExample(low, np.zeros_like(low)),
        ArcExample(high, np.ones_like(high)),
    )
    global_selection = select_demo_rule(global_examples)
    if global_selection.rule is not None:
        raise AssertionError("global majority negative control unexpectedly fit the frozen local grammar")
    rows.append(
        {
            "task": "global_majority_recolor",
            "level": "L3",
            "status": global_selection.status,
            "domain_patterns": math.nan,
            "exception_count": math.nan,
            "known_structured_gate_count": math.nan,
            "raw_interior_lut_bits": math.nan,
            "raw_description_bits_with_wrapper": math.nan,
            "sparse_model_bits": math.nan,
            "known_structured_description_bits": math.nan,
            "structured_description_bits_with_wrapper": math.nan,
            "preferred_description_route": "reject_local_ca",
            "direct_reference_exact": 0.0,
            "compiled_equivalent": math.nan,
            "binary4_invalid_rate": math.nan,
            "onehot_invalid_rate": math.nan,
        }
    )
    resize_example = ArcExample(np.zeros((4, 4), dtype=np.uint8), np.zeros((2, 2), dtype=np.uint8))
    resize_selection = select_demo_rule((resize_example,))
    rows.append(
        {
            "task": "resize_crop",
            "level": "L4",
            "status": resize_selection.status,
            "domain_patterns": math.nan,
            "exception_count": math.nan,
            "known_structured_gate_count": math.nan,
            "raw_interior_lut_bits": math.nan,
            "raw_description_bits_with_wrapper": math.nan,
            "sparse_model_bits": math.nan,
            "known_structured_description_bits": math.nan,
            "structured_description_bits_with_wrapper": math.nan,
            "preferred_description_route": "reject_fixed_lattice",
            "direct_reference_exact": 0.0,
            "compiled_equivalent": math.nan,
            "binary4_invalid_rate": math.nan,
            "onehot_invalid_rate": math.nan,
        }
    )

    # A task switch absent from cell state makes identical observations demand
    # incompatible outputs.  This is an honest missing-context L5 refusal.
    context_input = np.zeros((7, 7), dtype=np.uint8)
    context_selection = select_demo_rule(
        (
            ArcExample(context_input, context_input.copy()),
            ArcExample(context_input, np.ones_like(context_input)),
        )
    )
    if context_selection.rule is not None:
        raise AssertionError("missing task context unexpectedly fit a shared local rule")
    rows.append(
        {
            "task": "task_context_switch",
            "level": "L5",
            "status": context_selection.status,
            "domain_patterns": math.nan,
            "exception_count": math.nan,
            "known_structured_gate_count": math.nan,
            "raw_interior_lut_bits": math.nan,
            "raw_description_bits_with_wrapper": math.nan,
            "sparse_model_bits": math.nan,
            "known_structured_description_bits": math.nan,
            "structured_description_bits_with_wrapper": math.nan,
            "preferred_description_route": "reject_missing_task_context",
            "direct_reference_exact": 0.0,
            "compiled_equivalent": math.nan,
            "binary4_invalid_rate": math.nan,
            "onehot_invalid_rate": math.nan,
        }
    )

    # Random full-grid labels deliberately collide on identical local patches;
    # storing the target itself is the honest route, not a compact local CA.
    random_global_input = np.zeros((11, 11), dtype=np.uint8)
    random_global_output = rng.integers(0, 2, size=(11, 11), dtype=np.uint8)
    random_global_selection = select_demo_rule(
        (ArcExample(random_global_input, random_global_output),)
    )
    if random_global_selection.rule is not None:
        raise AssertionError("random global output unexpectedly fit the frozen local grammar")
    rows.append(
        {
            "task": "random_global_output",
            "level": "N",
            "status": random_global_selection.status,
            "domain_patterns": math.nan,
            "exception_count": math.nan,
            "known_structured_gate_count": math.nan,
            "raw_interior_lut_bits": math.nan,
            "raw_description_bits_with_wrapper": int(
                random_global_output.size + DECLARED_CA_WRAPPER_BITS
            ),
            "sparse_model_bits": math.nan,
            "known_structured_description_bits": math.nan,
            "structured_description_bits_with_wrapper": math.nan,
            "preferred_description_route": "raw_global_labels",
            "direct_reference_exact": 0.0,
            "compiled_equivalent": math.nan,
            "binary4_invalid_rate": math.nan,
            "onehot_invalid_rate": math.nan,
        }
    )

    dilation = _binary_cross_rule("dilation")
    initial = np.zeros((31, 31), dtype=np.uint8)
    initial[15, 15] = 1
    fixed8 = rollout_rule(dilation, initial, 8, stop="max_steps")
    fixed_point = rollout_rule(dilation, initial, 64, stop="fixed_point")
    rows.append(
        {
            "task": "iterative_dilation_horizon",
            "level": "L2",
            "status": fixed_point.stop_reason,
            "domain_patterns": len(dilation.seen_patterns),
            "exception_count": dilation.exception_count,
            "known_structured_gate_count": 4,
            "raw_interior_lut_bits": 32,
            "raw_description_bits_with_wrapper": 32 + DECLARED_CA_WRAPPER_BITS,
            "sparse_model_bits": dilation.model_description_bits,
            "known_structured_description_bits": 8,
            "structured_description_bits_with_wrapper": 8 + DECLARED_CA_WRAPPER_BITS,
            "preferred_description_route": "structured_program",
            "direct_reference_exact": float(np.all(fixed_point.terminal == 1)),
            "compiled_equivalent": 1.0,
            "binary4_invalid_rate": 0.0,
            "onehot_invalid_rate": 0.0,
            "fixed8_terminal_accuracy": float(np.mean(fixed8.terminal == 1)),
            "fixed_point_steps": fixed_point.steps_executed,
            "local_dynamic_gate_upper_bound": int(4 * initial.size * fixed_point.steps_executed),
            "fixed_point_detection_gate_upper_bound": int(
                (6 * initial.size - 1) * fixed_point.steps_executed
            ),
            "fixed_point_gate_evaluations": int(
                (10 * initial.size - 1) * fixed_point.steps_executed
            ),
        }
    )
    return rows, codec_rows


def summarize_arc(rows: pd.DataFrame) -> pd.DataFrame:
    summaries: list[dict[str, object]] = []

    def add(group: str, metric: str, numerator: float, denominator: float) -> None:
        summaries.append(
            {
                "group": group,
                "metric": metric,
                "numerator": numerator,
                "denominator": denominator,
                "value": float(numerator / denominator) if denominator else math.nan,
            }
        )

    add("all_tasks", "task_exact_pass_at_1", float(rows["task_exact"].sum()), float(len(rows)))
    add("all_tasks", "task_exact_pass_at_2", float(rows["task_exact_pass_at_2"].sum()), float(len(rows)))
    add("all_tasks", "identity_task_exact", float(rows["identity_task_exact"].sum()), float(len(rows)))
    add("all_tasks", "solver_coverage", float((rows["selection_status"] == "selected").sum()), float(len(rows)))
    add("all_tasks", "posthoc_oracle_local_representability", float(rows["posthoc_oracle_local_exists"].sum()), float(len(rows)))
    add("all_pairs", "pair_exact_pass_at_1", float(rows["pair_exact_count"].sum()), float(rows["labeled_pairs"].sum()))
    add("all_pairs", "pair_exact_pass_at_2", float(rows["pair_exact_count_pass_at_2"].sum()), float(rows["labeled_pairs"].sum()))
    supported = rows[rows["selection_status"] == "selected"]
    add("supported_tasks", "task_exact_pass_at_1", float(supported["task_exact"].sum()), float(len(supported)))
    add("supported_tasks", "task_exact_pass_at_2", float(supported["task_exact_pass_at_2"].sum()), float(len(supported)))
    compiled = rows[rows["compiled_checked"] == 1.0]
    add("compiled_tasks", "semantic_equivalence", float(compiled["compiled_equivalent"].sum()), float(len(compiled)))
    for status, count in rows["selection_status"].value_counts().sort_index().items():
        add("refusal_taxonomy", str(status), float(count), float(len(rows)))
    return pd.DataFrame(summaries)


def summarize_programs(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    summaries: list[dict[str, object]] = []

    def add(group: str, metric: str, numerator: float, denominator: float) -> None:
        summaries.append(
            {
                "group": group,
                "metric": metric,
                "numerator": numerator,
                "denominator": denominator,
                "value": float(numerator / denominator) if denominator else math.nan,
            }
        )

    add("program_all_tasks", "task_exact_pass_at_1", float(rows["task_exact"].sum()), float(len(rows)))
    add("program_all_tasks", "solver_coverage", float((rows["selection_status"] == "selected").sum()), float(len(rows)))
    add("program_all_pairs", "pair_exact_pass_at_1", float(rows["pair_exact_count"].sum()), float(rows["labeled_pairs"].sum()))
    supported = rows[rows["selection_status"] == "selected"]
    add("program_supported_tasks", "task_exact_pass_at_1", float(supported["task_exact"].sum()), float(len(supported)))
    for status, count in rows["selection_status"].value_counts().sort_index().items():
        add("program_refusal_taxonomy", str(status), float(count), float(len(rows)))
    return pd.DataFrame(summaries)


def write_artifacts(
    output_dir: Path,
    *,
    task_rows: list[dict[str, object]],
    program_rows: list[dict[str, object]],
    synthetic_rows: list[dict[str, object]],
    codec_rows: list[dict[str, object]],
    submissions: list[dict[str, object]],
    metadata: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    task_frame = pd.DataFrame(task_rows)
    program_frame = pd.DataFrame(program_rows)
    synthetic_frame = pd.DataFrame(synthetic_rows)
    codec_frame = pd.DataFrame(codec_rows)
    summary_parts: list[pd.DataFrame] = []
    if not task_frame.empty:
        table_summary = summarize_arc(task_frame)
        table_summary["group"] = "sparse_table/" + table_summary["group"].astype(str)
        summary_parts.append(table_summary)
    if not program_frame.empty:
        summary_parts.append(summarize_programs(program_frame))
    if not task_frame.empty and not program_frame.empty:
        joined = task_frame[["task_id", "task_exact"]].merge(
            program_frame[["task_id", "task_exact"]], on="task_id", suffixes=("_table", "_program")
        )
        union = float(np.maximum(joined["task_exact_table"], joined["task_exact_program"]).sum())
        summary_parts.append(
            pd.DataFrame(
                [
                    {
                        "group": "two_family_lower_bound",
                        "metric": "task_exact_pass_at_2",
                        "numerator": union,
                        "denominator": float(len(joined)),
                        "value": union / len(joined) if len(joined) else math.nan,
                    }
                ]
            )
        )
    summary_frame = pd.concat(summary_parts, ignore_index=True) if summary_parts else pd.DataFrame()
    outputs = {
        "arc_ca_task_results.csv": task_frame,
        "arc_ca_program_results.csv": program_frame,
        "arc_ca_synthetic_results.csv": synthetic_frame,
        "arc_ca_codec_results.csv": codec_frame,
        "arc_ca_summary.csv": summary_frame,
    }
    for filename, frame in outputs.items():
        frame.to_csv(output_dir / filename, index=False)
    metadata["row_counts"] = {
        **{filename: len(frame) for filename, frame in outputs.items()},
        "arc_ca_predictions.jsonl": len(submissions),
    }
    metadata["artifacts"] = {
        filename: {
            "bytes": (output_dir / filename).stat().st_size,
            "sha256": sha256_file(output_dir / filename),
        }
        for filename in outputs
    }
    if submissions:
        predictions_path = output_dir / "arc_ca_predictions.jsonl"
        predictions_path.write_text(
            "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in submissions),
            encoding="utf-8",
        )
        metadata["artifacts"][predictions_path.name] = {
            "bytes": predictions_path.stat().st_size,
            "sha256": sha256_file(predictions_path),
        }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("synthetic", "arc", "all"), default="all")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--dataset-name", choices=("ARC-AGI-1", "ARC-AGI-2"))
    parser.add_argument("--split", choices=("training", "evaluation"), default="training")
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--task-limit", type=int)
    parser.add_argument("--benchmark-repeats", type=int, default=3)
    parser.add_argument("--max-compiled-entries", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--public-evaluation-receipt", action="store_true")
    parser.add_argument("--receipt-ledger", type=Path)
    args = parser.parse_args()
    if args.benchmark_repeats < 1 or (
        args.max_compiled_entries is not None and args.max_compiled_entries < 0
    ):
        parser.error("benchmark repeats must be positive and compiled-entry budget nonnegative")
    if args.suite in ("arc", "all") and (args.dataset_root is None or args.dataset_name is None):
        parser.error("ARC suites require --dataset-root and --dataset-name")
    if args.public_evaluation_receipt and not (
        args.suite in ("arc", "all") and args.dataset_name == "ARC-AGI-2" and args.split == "evaluation" and args.mode == "full"
    ):
        parser.error("public evaluation receipt requires full ARC-AGI-2 evaluation")
    if args.public_evaluation_receipt and args.task_limit is not None:
        parser.error("public evaluation receipt forbids a task limit")
    if (
        args.suite in ("arc", "all")
        and args.dataset_name == "ARC-AGI-2"
        and args.split == "evaluation"
        and not args.public_evaluation_receipt
    ):
        parser.error("ARC-AGI-2 evaluation requires --public-evaluation-receipt")
    if args.public_evaluation_receipt and args.receipt_ledger is None:
        parser.error("public evaluation receipt requires --receipt-ledger")
    if args.receipt_ledger is not None and not args.public_evaluation_receipt:
        parser.error("--receipt-ledger is only valid with --public-evaluation-receipt")
    if args.mode == "full" and args.max_compiled_entries is not None:
        parser.error("full mode compiles every selected rule; do not set --max-compiled-entries")
    if args.output_dir.exists():
        parser.error("output directory must not already exist")
    if args.public_evaluation_receipt:
        for destination, label in (
            (args.output_dir, "output"),
            (args.receipt_ledger, "receipt ledger"),
        ):
            try:
                destination.resolve().relative_to(ROOT.resolve())
            except ValueError:
                pass
            else:
                parser.error(f"public-evaluation {label} must be outside the source repository")
        try:
            args.receipt_ledger.resolve().relative_to(args.output_dir.resolve())
        except ValueError:
            pass
        else:
            parser.error("receipt ledger must not be inside the output directory")
        if args.receipt_ledger.resolve().parent != args.output_dir.resolve().parent:
            parser.error("receipt ledger and output directory must be sibling paths")

    start_time = datetime.now(timezone.utc)
    source_state = git_state()
    if args.public_evaluation_receipt:
        if (
            source_state["commit"] == "unavailable"
            or source_state["dirty"]
            or not source_state["origin_contains_head"]
            or not source_state["origin_is_expected_repository"]
        ):
            parser.error("public evaluation requires a clean source commit already present on the expected origin")

    synthetic_rows: list[dict[str, object]] = []
    codec_rows: list[dict[str, object]] = []
    task_rows: list[dict[str, object]] = []
    program_rows: list[dict[str, object]] = []
    submissions: list[dict[str, object]] = []
    if args.suite in ("synthetic", "all"):
        synthetic_rows, synthetic_codec = run_synthetic(args.benchmark_repeats)
        codec_rows.extend(synthetic_codec)

    dataset_metadata: dict[str, object] = {"requested": False}
    if args.suite in ("arc", "all"):
        limit = args.task_limit
        if args.mode == "smoke" and limit is None:
            limit = 20
        loaded = load_arc_split(args.dataset_root, args.split, limit=limit)
        dataset_state = dataset_git_state(args.dataset_root)
        dataset_commit = str(dataset_state["commit"])
        expected = ARC_AGI_2_COMMIT if args.dataset_name == "ARC-AGI-2" else ARC_AGI_1_COMMIT
        if dataset_commit != expected:
            raise ValueError(f"dataset commit {dataset_commit} does not match frozen {expected}")
        if dataset_state["dirty"]:
            raise ValueError("dataset repository worktree is not clean")
        loaded_paths = [path for _, _, path in loaded]
        actual_split_digest = split_digest(loaded_paths)
        actual_task_id_digest = task_id_digest(loaded_paths)
        expected_split = FROZEN_SPLITS[(args.dataset_name, args.split)]
        if args.mode == "full":
            if len(loaded) != expected_split["count"]:
                raise ValueError(
                    f"full {args.dataset_name} {args.split} requires {expected_split['count']} tasks, found {len(loaded)}"
                )
            if actual_split_digest != expected_split["split_digest"]:
                raise ValueError("dataset split content digest does not match the frozen manifest")
            if actual_task_id_digest != expected_split["task_id_digest"]:
                raise ValueError("dataset task-id digest does not match the frozen manifest")
        if args.public_evaluation_receipt:
            if any(output is None for _, labels, _ in loaded for output in labels.test_outputs):
                raise ValueError("public evaluation receipt requires complete labels for every test pair")
            _reserve_receipt(
                args.receipt_ledger,
                {
                    "status": "reserved_before_scoring",
                    "started_utc": start_time.isoformat(),
                    "source_commit": source_state["commit"],
                    "dataset_commit": dataset_commit,
                    "split_digest": actual_split_digest,
                    "task_id_digest": actual_task_id_digest,
                    "output_dir": str(args.output_dir.resolve()),
                },
            )
        for index, (problem, labels, _) in enumerate(loaded, 1):
            row, codec, submission = evaluate_arc_task(
                problem,
                labels,
                repeats=args.benchmark_repeats,
                max_compiled_entries=(
                    None
                    if args.mode == "full"
                    else (128 if args.max_compiled_entries is None else args.max_compiled_entries)
                ),
            )
            row.update(
                {
                    "dataset": args.dataset_name,
                    "dataset_commit": dataset_commit,
                    "split": args.split,
                    "task_index": index,
                }
            )
            task_rows.append(row)
            submissions.append(submission)
            program_row = evaluate_program_task(problem, labels, repeats=args.benchmark_repeats)
            program_row.update(
                {
                    "dataset": args.dataset_name,
                    "dataset_commit": dataset_commit,
                    "split": args.split,
                    "task_index": index,
                }
            )
            program_rows.append(program_row)
            if codec is not None:
                codec.update({"dataset": args.dataset_name, "split": args.split})
                codec_rows.append(codec)
        dataset_metadata = {
            "requested": True,
            "name": args.dataset_name,
            "commit": dataset_commit,
            "split": args.split,
            "task_count": len(loaded),
            "split_digest_for_loaded_tasks": actual_split_digest,
            "task_id_digest_for_loaded_tasks": actual_task_id_digest,
            "expected_full_split": expected_split,
            "git_state": dataset_state,
            "paths_are_external_not_vendored": True,
        }

    end_time = datetime.now(timezone.utc)
    source_state_end = git_state()
    if args.public_evaluation_receipt and (
        source_state_end["commit"] != source_state["commit"] or source_state_end["dirty"]
    ):
        raise RuntimeError("source commit or cleanliness changed during public evaluation")
    metadata: dict[str, object] = {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "argv": [str(ROOT / "src" / "run_arc_ca.py"), *sys.argv[1:]],
        "suite": args.suite,
        "mode": args.mode,
        "source_git_at_start": source_state,
        "source_git_at_end": source_state_end,
        "dataset": dataset_metadata,
        "frozen_neighborhoods": [
            {"name": spec.name, "offsets": list(spec.offsets)} for spec in FROZEN_NEIGHBORHOODS
        ],
        "configuration": {
            "fallback": "center_color",
            "selection_order": "LODO pair exact, LODO cell accuracy, prefix upper bound, lexical name",
            "prediction_budget": 2,
            "benchmark_repeats": args.benchmark_repeats,
            "max_compiled_entries": (
                None
                if args.mode == "full"
                else (128 if args.max_compiled_entries is None else args.max_compiled_entries)
            ),
            "compile_policy": "all_selected_rules" if args.mode == "full" else "exception_budget",
            "declared_ca_wrapper_bits": DECLARED_CA_WRAPPER_BITS,
            "task_limit": args.task_limit,
            "synthetic_seed": 20_260_716,
        },
        "public_evaluation_receipt": {
            "enabled": args.public_evaluation_receipt,
            "source_was_clean_at_start": bool(not source_state["dirty"]),
            "source_was_clean_at_end": bool(not source_state_end["dirty"]),
            "source_commit_was_on_origin": bool(source_state["origin_contains_head"]),
            "source_origin_was_expected_repository": bool(source_state["origin_is_expected_repository"]),
            "declared_one_shot_ledger": str(args.receipt_ledger.resolve()) if args.receipt_ledger else None,
            "two_distinct_demo_ranked_local_rules_when_available": True,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "platform": platform.platform(),
            "hostname": platform.node(),
        },
        "started_utc": start_time.isoformat(),
        "finished_utc": end_time.isoformat(),
        "elapsed_seconds": (end_time - start_time).total_seconds(),
        "claim_boundary": "Sparse compiler counts are no-sharing upper bounds; ARC posthoc oracle uses labels and is never a solver score; CPU timing is not hardware PPA.",
        "sources": {
            str(path.relative_to(ROOT)).replace("\\", "/"): {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in (
                ROOT / "src" / "arc_data.py",
                ROOT / "src" / "arc_ca.py",
                ROOT / "src" / "arc_ca_programs.py",
                ROOT / "src" / "run_arc_ca.py",
                ROOT / "notes" / "design" / "arc-ca-contract.md",
                ROOT / "third_party" / "arc_agi_manifest.json",
            )
        },
    }
    write_artifacts(
        args.output_dir,
        task_rows=task_rows,
        program_rows=program_rows,
        synthetic_rows=synthetic_rows,
        codec_rows=codec_rows,
        submissions=submissions,
        metadata=metadata,
    )
    if args.public_evaluation_receipt:
        _complete_receipt(args.receipt_ledger, args.output_dir, end_time.isoformat())
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "task_rows": len(task_rows),
                "program_rows": len(program_rows),
                "synthetic_rows": len(synthetic_rows),
                "codec_rows": len(codec_rows),
                "elapsed_seconds": metadata["elapsed_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
