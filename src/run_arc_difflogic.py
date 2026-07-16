"""Run label-isolated DiffLogic-ARC development and confirmatory cohorts.

ARC-AGI-2 evaluation is intentionally rejected.  The consumed public
evaluation receipt belongs to the frozen categorical experiment and is never
reopened by this runner.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Iterable, Sequence

import numpy as np

from arc_ca import predict_direct, select_demo_rule
from arc_data import ArcExample, ArcLabels, ArcProblem, load_arc_split, sha256_file, split_digest, task_id_digest
from arc_difflogic_features import paired_grid_metrics
from arc_difflogic_model import VARIANT_CONFIGS
from arc_difflogic_train import (
    TrainingConfig,
    augmented_sparse_predictions,
    candidate_selection_key,
    evaluate_candidate,
    train_task_candidate,
)

try:
    import torch
except ImportError:  # pragma: no cover - pure protocol helpers remain importable.
    torch = None


ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    "src/trainable_difflogic.py",
    "src/arc_difflogic_features.py",
    "src/arc_difflogic_model.py",
    "src/arc_difflogic_train.py",
    "src/run_arc_difflogic.py",
    "src/arc_data.py",
    "src/arc_ca.py",
)

COHORTS: dict[str, tuple[str, ...]] = {
    "dev_low_mdl": (
        "9968a131", "aabf363d", "32e9702f", "3aa6fb7a",
        "aedd82e4", "ba97ae07", "4347f46a", "25d8a9c8",
    ),
    "conf_induction_hash": (
        "a8d7556c", "7ee1c6ea", "a834deea", "1e0a9b12", "a04b2602", "b60334d2",
        "68b16354", "9caf5b84", "3345333e", "bda2d7a6", "cc9053aa", "9b5080bb",
    ),
    "positive": ("bb43febb", "c8f0f002", "84f2aca1", "dc1df850"),
    "representation_gap": ("fc754716", "1b8318e3", "e734a0e8", "6e82a1ae"),
    "local_conflict": ("94be5b80", "342ae2ed", "465b7d93", "a3f84088"),
    "shape_change": ("80214e03", "ce602527", "ae4f1146", "9110e3c5"),
}


def _grid(rows: Sequence[Sequence[int]]) -> np.ndarray:
    return np.asarray(rows, dtype=np.uint8)


def _dilate(grid: np.ndarray, steps: int) -> np.ndarray:
    state = grid.copy()
    for _ in range(steps):
        padded = np.pad(state == 1, 1)
        neighbors = np.zeros_like(state, dtype=bool)
        for row in range(3):
            for column in range(3):
                neighbors |= padded[row : row + state.shape[0], column : column + state.shape[1]]
        state = neighbors.astype(np.uint8)
    return state


def _marker_rule(grid: np.ndarray) -> np.ndarray:
    output = grid.copy()
    output[output == 1] = 2 if 9 in grid else 3
    return output


def _scale2(grid: np.ndarray) -> np.ndarray:
    return np.repeat(np.repeat(grid, 2, axis=0), 2, axis=1)


def _shift_right(grid: np.ndarray, steps: int) -> np.ndarray:
    output = np.zeros_like(grid)
    if steps < grid.shape[1]:
        output[:, steps:] = grid[:, :-steps]
    return output


def _object_anchor_rule(grid: np.ndarray) -> np.ndarray:
    output = grid.copy()
    visited = np.zeros_like(grid, dtype=bool)
    for row in range(grid.shape[0]):
        for column in range(grid.shape[1]):
            if visited[row, column] or grid[row, column] != 1:
                continue
            stack = [(row, column)]
            visited[row, column] = True
            component = []
            while stack:
                r, c = stack.pop()
                component.append((r, c))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1] and not visited[rr, cc] and grid[rr, cc] == 1:
                        visited[rr, cc] = True
                        stack.append((rr, cc))
            if len(component) >= 3:
                anchor = min(component)
                output[anchor] = 2
    return output


def synthetic_tasks() -> list[tuple[ArcProblem, ArcLabels, Path | None]]:
    not_inputs = (
        _grid([[0, 1, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 1, 0]]),
        _grid([[1, 0, 0], [0, 1, 0], [1, 1, 0]]),
    )
    not_examples = tuple(ArcExample(item, 1 - item) for item in not_inputs)
    not_test = _grid([[0, 0, 1, 0, 1], [1, 0, 1, 1, 0], [0, 1, 0, 0, 1]])

    local_inputs = (
        _grid([[0, 1, 0], [1, 0, 1], [0, 0, 0]]),
        _grid([[0, 0, 0], [0, 1, 0], [0, 0, 1]]),
    )
    local_examples = tuple(ArcExample(item, _dilate(item, 1)) for item in local_inputs)
    local_test = _grid([[1, 0, 0], [0, 0, 0], [0, 0, 0]])

    propagation_inputs = (
        _grid([[1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]),
        _grid([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]),
    )
    propagation_examples = tuple(ArcExample(item, _dilate(item, 2)) for item in propagation_inputs)
    propagation_test = _grid([[0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])

    shift_inputs = (
        _grid([[1, 0, 0, 1, 0, 0, 0], [0, 1, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0, 0]]),
        _grid([[0, 1, 1, 0, 0, 0, 0], [1, 0, 0, 0, 1, 0, 0], [0, 0, 1, 0, 1, 0, 0]]),
        _grid([[1, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0, 0]]),
    )
    shift_examples = tuple(ArcExample(item, _shift_right(item, 2)) for item in shift_inputs)
    shift_test = _grid([[1, 0, 1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 1, 0, 0, 0, 0, 0], [1, 1, 0, 0, 1, 0, 0, 0, 0]])

    marker_inputs = (
        _grid([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
        _grid([[1, 0, 0, 0], [0, 0, 0, 9], [0, 0, 0, 0], [0, 0, 0, 0]]),
        _grid([[0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 9], [0, 0, 0, 0]]),
    )
    marker_examples = tuple(ArcExample(item, _marker_rule(item)) for item in marker_inputs)
    marker_test = _grid([[0, 1, 0, 0], [0, 0, 0, 0], [9, 0, 0, 0], [0, 0, 0, 0]])

    object_inputs = (
        _grid([[0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 0, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0]]),
        _grid([[0, 1, 0, 0, 0, 0, 0], [0, 1, 0, 1, 1, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0]]),
    )
    object_examples = tuple(ArcExample(item, _object_anchor_rule(item)) for item in object_inputs)
    object_test = _grid([[0, 0, 0, 0, 1, 1, 0], [0, 1, 1, 1, 0, 0, 0], [0, 0, 0, 0, 1, 1, 1], [0, 1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]])

    scale_inputs = (_grid([[1, 2], [3, 4]]), _grid([[4, 0, 2], [1, 3, 0]]))
    scale_examples = tuple(ArcExample(item, _scale2(item)) for item in scale_inputs)
    scale_test = _grid([[2, 1], [0, 4], [3, 0]])

    constant_inputs = (_grid([[0, 2], [1, 0]]), _grid([[3, 0, 1], [0, 2, 0], [1, 0, 3]]))
    constant_target = np.ones((2, 3), dtype=np.uint8)
    constant_examples = tuple(ArcExample(item, constant_target) for item in constant_inputs)
    constant_test = _grid([[4, 0, 0, 2], [0, 1, 3, 0]])

    tasks = [
        ("syn_local_not", not_examples, not_test, 1 - not_test),
        ("syn_local_dilation1", local_examples, local_test, _dilate(local_test, 1)),
        ("syn_hidden_shift2", shift_examples, shift_test, _shift_right(shift_test, 2)),
        ("syn_hidden_dilation2", propagation_examples, propagation_test, _dilate(propagation_test, 2)),
        ("syn_global_marker", marker_examples, marker_test, _marker_rule(marker_test)),
        ("syn_object_anchor", object_examples, object_test, _object_anchor_rule(object_test)),
        ("syn_shape_constant", constant_examples, constant_test, constant_target),
        ("syn_shape_scale2", scale_examples, scale_test, _scale2(scale_test)),
    ]
    return [
        (ArcProblem(name, tuple(examples), (test_input,)), ArcLabels(name, (test_output,)), None)
        for name, examples, test_input, test_output in tasks
    ]


def _git(*arguments: str) -> str:
    completed = subprocess.run(("git", *arguments), cwd=ROOT, text=True, capture_output=True, check=True)
    return completed.stdout.strip()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(_json_safe(row), sort_keys=True, separators=(",", ":")) + "\n")


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"))
                    if isinstance(value, (dict, list, tuple))
                    else _json_safe(value)
                    for key, value in row.items()
                }
            )


def _flat_candidate_record(candidate: Any, evaluation: dict[str, Any], selected: bool) -> dict[str, Any]:
    record = dict(candidate.record)
    record.pop("config", None)
    record.pop("training_config", None)
    record["test_fixed_point_steps"] = json.dumps(record.get("test_fixed_point_steps", []), separators=(",", ":"))
    record.update(evaluation)
    record["selected_by_demo"] = float(selected)
    return record


def _zero_metrics() -> dict[str, float]:
    return {
        "pair_exact_count": 0.0,
        "pair_exact_rate": 0.0,
        "task_exact": 0.0,
        "cell_accuracy": 0.0,
    }


def _evaluate_grid_predictions(predictions: Sequence[np.ndarray], labels: ArcLabels) -> dict[str, float]:
    references = [np.asarray(output) for output in labels.test_outputs if output is not None]
    if len(references) != len(labels.test_outputs):
        raise ValueError("evaluation requires complete public labels")
    if len(predictions) != len(references):
        return _zero_metrics()
    return paired_grid_metrics(references, predictions)


def _baseline_candidates(problem: ArcProblem) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Produce and demo-audit baseline predictions without receiving test labels."""
    rows: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    sparse = select_demo_rule(problem.demonstrations)
    direct_predictions = (
        [predict_direct(sparse.rule, grid) for grid in problem.test_inputs]
        if sparse.rule is not None
        else []
    )
    if direct_predictions:
        predictions.append({"task_id": problem.task_id, "variant": "sparse", "seed": -1, "grids": direct_predictions})
    sparse_demo_predictions = (
        [predict_direct(sparse.rule, example.input_grid) for example in problem.demonstrations]
        if sparse.rule is not None
        else []
    )
    sparse_demo = (
        paired_grid_metrics([example.output_grid for example in problem.demonstrations], sparse_demo_predictions)
        if sparse_demo_predictions
        else {"task_exact": 0.0, "pair_exact_rate": 0.0, "cell_accuracy": 0.0}
    )
    rows.append(
        {
            "task_id": problem.task_id,
            "variant": "sparse",
            "seed": -1,
            "status": sparse.status,
            "hard_demo_task_exact": sparse_demo["task_exact"],
            "hard_demo_pair_exact_rate": sparse_demo["pair_exact_rate"],
            "hard_demo_cell_accuracy": sparse_demo["cell_accuracy"],
            "deployment_eligible": float(bool(direct_predictions) and sparse_demo["task_exact"] == 1.0),
            "selected_by_demo": 1.0,
        }
    )
    d4_predictions, d4 = augmented_sparse_predictions(problem)
    if d4_predictions:
        predictions.append({"task_id": problem.task_id, "variant": "d4_sparse", "seed": -1, "grids": d4_predictions})
    rows.append(
        {
            "task_id": problem.task_id,
            "variant": "d4_sparse",
            "seed": -1,
            **d4,
            "hard_demo_task_exact": float(d4.get("deployment_demo_task_exact", 0.0)),
            "hard_demo_pair_exact_rate": float(d4.get("deployment_demo_task_exact", 0.0)),
            "hard_demo_cell_accuracy": float(d4.get("deployment_demo_cell_accuracy", 0.0)),
            "selected_by_demo": 1.0,
        }
    )
    padded_predictions, padded = augmented_sparse_predictions(problem, "d4_bgpad")
    if padded_predictions:
        predictions.append({"task_id": problem.task_id, "variant": "d4_bgpad_sparse", "seed": -1, "grids": padded_predictions})
    rows.append(
        {
            "task_id": problem.task_id,
            "variant": "d4_bgpad_sparse",
            "seed": -1,
            **padded,
            "hard_demo_task_exact": float(padded.get("deployment_demo_task_exact", 0.0)),
            "hard_demo_pair_exact_rate": float(padded.get("deployment_demo_task_exact", 0.0)),
            "hard_demo_cell_accuracy": float(padded.get("deployment_demo_cell_accuracy", 0.0)),
            "selected_by_demo": 1.0,
        }
    )
    return rows, predictions


def _score_baseline_candidates(
    rows: Sequence[dict[str, Any]],
    predictions: Sequence[dict[str, Any]],
    labels: ArcLabels,
) -> list[dict[str, Any]]:
    """Evaluator-only stage called after every model/seed choice is frozen."""
    if any(str(row["task_id"]) != labels.task_id for row in rows):
        raise ValueError("baseline and label task ids differ")
    by_variant = {str(record["variant"]): record["grids"] for record in predictions}
    scored = []
    for row in rows:
        metrics = _evaluate_grid_predictions(by_variant.get(str(row["variant"]), ()), labels)
        scored.append({**row, **{"hard_" + key: value for key, value in metrics.items()}})
    return scored


def _summary(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [row for row in rows if float(row.get("selected_by_demo", 0)) == 1.0]
    output = []
    for variant in sorted({str(row["variant"]) for row in selected}):
        group = [row for row in selected if row["variant"] == variant]
        output.extend(
            (
                {"variant": variant, "metric": "tasks", "numerator": len(group), "denominator": len(group), "value": 1.0},
                {
                    "variant": variant,
                    "metric": "hard_task_exact",
                    "numerator": sum(float(row.get("hard_task_exact", 0) or 0) for row in group),
                    "denominator": len(group),
                    "value": sum(float(row.get("hard_task_exact", 0) or 0) for row in group) / len(group) if group else math.nan,
                },
                {
                    "variant": variant,
                    "metric": "hard_demo_exact",
                    "numerator": sum(float(row.get("hard_demo_task_exact", 0) or 0) for row in group),
                    "denominator": len(group),
                    "value": sum(float(row.get("hard_demo_task_exact", 0) or 0) for row in group) / len(group) if group else math.nan,
                },
            )
        )
    return output


def _validate_confirmatory_config(
    confirmation: dict[str, Any],
    *,
    source_head: str,
    args: argparse.Namespace,
    variants: Sequence[str],
    seeds: Sequence[int],
    training_config: TrainingConfig,
    task_ids: Sequence[str],
    task_id_digest_value: str,
    task_content_digest_value: str,
) -> None:
    """Require a confirmatory receipt to freeze every result-affecting CLI choice."""
    expected = {
        "protocol": "arc_difflogic_v1",
        "source_commit": source_head,
        "suite": args.suite,
        "cohort": args.cohort,
        "mode": args.mode,
        "device": args.device,
        "augmentation": args.augmentation,
        "variants": list(variants),
        "seeds": list(seeds),
        "training_config": _json_safe(asdict(training_config)),
        "task_ids": list(task_ids),
        "task_id_digest": task_id_digest_value,
        "task_content_digest": task_content_digest_value,
    }
    mismatches = {
        key: {"expected": value, "observed": confirmation.get(key)}
        for key, value in expected.items()
        if confirmation.get(key) != value
    }
    if mismatches:
        raise ValueError("confirmatory config mismatch: " + json.dumps(mismatches, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("synthetic", "arc"), required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--cohort", choices=tuple(COHORTS))
    parser.add_argument("--variants", default="dl1,dlr,dlo,dlf,mlp")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--soft-fraction", type=float, default=0.65)
    parser.add_argument("--entropy-weight", type=float, default=2e-4)
    parser.add_argument("--changed-cell-weight", type=float, default=4.0)
    parser.add_argument("--intermediate-loss-weight", type=float, default=0.0)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--augmentation", choices=("auto", "none", "d4", "bgpad", "d4_bgpad"), default="auto")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confirmatory-config", type=Path)
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--task-ids", help="comma-separated development subset; forbidden for full confirmatory runs")
    args = parser.parse_args()

    if torch is None:
        parser.error("PyTorch is required; see requirements-difflogic-arc.txt")

    variants = tuple(item.strip() for item in args.variants.split(",") if item.strip())
    unknown = sorted(set(variants) - set(VARIANT_CONFIGS))
    if unknown:
        parser.error("unknown variants: " + ", ".join(unknown))
    seeds = tuple(int(item) for item in args.seeds.split(",") if item.strip())
    if not seeds:
        parser.error("at least one seed is required")
    if args.suite == "arc":
        if args.dataset_root is None or args.cohort is None:
            parser.error("ARC suite requires --dataset-root and --cohort")
        if args.cohort == "conf_induction_hash" and args.confirmatory_config is None:
            parser.error("confirmatory cohort requires --confirmatory-config")
        if args.cohort == "conf_induction_hash" and args.mode != "full":
            parser.error("confirmatory cohort requires --mode full")
        if args.cohort == "conf_induction_hash" and (args.task_ids or args.max_tasks is not None):
            parser.error("confirmatory cohort forbids task subsets")
        tasks = load_arc_split(args.dataset_root, "training", task_ids=COHORTS[args.cohort])
    else:
        tasks = synthetic_tasks()
    if args.task_ids:
        if args.mode == "full" and args.cohort == "conf_induction_hash":
            parser.error("full confirmatory runs may not select a task subset")
        wanted = {item.strip() for item in args.task_ids.split(",") if item.strip()}
        tasks = [item for item in tasks if item[0].task_id in wanted]
        missing = sorted(wanted - {item[0].task_id for item in tasks})
        if missing:
            parser.error("unknown task ids: " + ", ".join(missing))
    if args.max_tasks is not None:
        tasks = tasks[: args.max_tasks]
    if not tasks:
        parser.error("no tasks selected")

    status_start = _git("status", "--porcelain")
    if args.mode == "full" and status_start:
        parser.error("full runs require a clean source worktree")
    source_head = _git("rev-parse", "HEAD")
    if args.confirmatory_config is not None:
        confirmation = json.loads(args.confirmatory_config.read_text(encoding="utf-8"))
        if not isinstance(confirmation, dict):
            parser.error("confirmatory config must be a JSON object")
        confirmation_sha = sha256_file(args.confirmatory_config)
    else:
        confirmation = None
        confirmation_sha = ""

    training_config = TrainingConfig(
        epochs=args.epochs if args.epochs is not None else (40 if args.mode == "smoke" else 240),
        learning_rate=args.learning_rate,
        soft_fraction=args.soft_fraction,
        entropy_weight=args.entropy_weight,
        changed_cell_weight=args.changed_cell_weight,
        intermediate_loss_weight=args.intermediate_loss_weight,
        trace_every=10 if args.mode == "smoke" else 20,
        early_stop_patience=2 if args.mode == "smoke" else 3,
    )
    task_paths = [path for _, _, path in tasks if path is not None]
    task_id_digest_value = task_id_digest(task_paths) if task_paths else "synthetic"
    task_content_digest_value = split_digest(task_paths) if task_paths else "synthetic"
    if confirmation is not None:
        try:
            _validate_confirmatory_config(
                confirmation,
                source_head=source_head,
                args=args,
                variants=variants,
                seeds=seeds,
                training_config=training_config,
                task_ids=[problem.task_id for problem, _, _ in tasks],
                task_id_digest_value=task_id_digest_value,
                task_content_digest_value=task_content_digest_value,
            )
        except ValueError as error:
            parser.error(str(error))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    all_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    circuit_rows: list[dict[str, Any]] = []
    pending_evaluation: list[
        tuple[
            ArcLabels,
            list[dict[str, Any]],
            list[dict[str, Any]],
            dict[str, tuple[list[Any], Any]],
        ]
    ] = []

    for task_number, (problem, labels, _) in enumerate(tasks, start=1):
        baseline_rows, baseline_predictions = _baseline_candidates(problem)
        prediction_rows.extend(baseline_predictions)
        variant_candidates: dict[str, tuple[list[Any], Any]] = {}
        for variant in variants:
            config = VARIANT_CONFIGS[variant]
            candidates = []
            for seed in seeds:
                candidate = train_task_candidate(
                    problem,
                    config,
                    training_config,
                    seed=seed,
                    device=args.device,
                    augmentation_policy=args.augmentation,
                )
                candidates.append(candidate)
                prediction_rows.append(
                    {
                        "task_id": problem.task_id,
                        "variant": variant,
                        "seed": seed,
                        "status": candidate.status,
                        "hard_grids": candidate.hard_predictions,
                        "soft_grids": candidate.soft_predictions,
                    }
                )
                trace_rows.extend(
                    {"task_id": problem.task_id, "variant": variant, "seed": seed, **trace}
                    for trace in candidate.trace
                )
                if candidate.hard_specifications:
                    circuit_rows.append(
                        {
                            "task_id": problem.task_id,
                            "variant": variant,
                            "seed": seed,
                            "sha256": candidate.record["hard_export_sha256"],
                            "layers": candidate.hard_specifications,
                        }
                    )
            chosen = min(candidates, key=candidate_selection_key)
            variant_candidates[variant] = (candidates, chosen)
        pending_evaluation.append((labels, baseline_rows, baseline_predictions, variant_candidates))
        print(f"[predict {task_number}/{len(tasks)}] {problem.task_id}", flush=True)

    # Label-bearing objects enter only this evaluator pass, after every task,
    # seed, horizon, hard circuit, and demo-selected candidate is frozen.
    for task_number, (labels, baseline_rows, baseline_predictions, variant_candidates) in enumerate(
        pending_evaluation,
        start=1,
    ):
        scored_baselines = _score_baseline_candidates(baseline_rows, baseline_predictions, labels)
        all_rows.extend(scored_baselines)
        selected_rows.extend(scored_baselines)
        for candidates, chosen in variant_candidates.values():
            for candidate in candidates:
                evaluation = evaluate_candidate(candidate, labels)
                row = _flat_candidate_record(candidate, evaluation, candidate is chosen)
                all_rows.append(row)
                if candidate is chosen:
                    selected_rows.append(row)
        print(f"[evaluate {task_number}/{len(tasks)}] {labels.task_id}", flush=True)

    summary = _summary(selected_rows)
    artifacts = {
        "arc_difflogic_seed_results.csv": all_rows,
        "arc_difflogic_selected_results.csv": selected_rows,
        "arc_difflogic_summary.csv": summary,
    }
    for name, rows in artifacts.items():
        _write_csv(output_dir / name, rows)
    _write_jsonl(output_dir / "arc_difflogic_predictions.jsonl", prediction_rows)
    _write_jsonl(output_dir / "arc_difflogic_traces.jsonl", trace_rows)
    _write_jsonl(output_dir / "arc_difflogic_hard_circuits.jsonl", circuit_rows)

    artifact_names = tuple(artifacts) + (
        "arc_difflogic_predictions.jsonl",
        "arc_difflogic_traces.jsonl",
        "arc_difflogic_hard_circuits.jsonl",
    )
    status_end = _git("status", "--porcelain")
    if args.mode == "full" and status_end != status_start:
        raise RuntimeError("source worktree changed during full run")
    metadata = {
        "experiment": "arc_difflogic_v1",
        "suite": args.suite,
        "cohort": args.cohort or "synthetic",
        "mode": args.mode,
        "source_commit": source_head,
        "source_status_start": status_start,
        "source_status_end": status_end,
        "source_files": {
            path: {"bytes": (ROOT / path).stat().st_size, "sha256": sha256_file(ROOT / path)}
            for path in SOURCE_FILES
        },
        "variants": variants,
        "seeds": seeds,
        "training_config": asdict(training_config),
        "device": args.device,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
        },
        "confirmatory_config": confirmation,
        "confirmatory_config_sha256": confirmation_sha,
        "task_ids": [problem.task_id for problem, _, _ in tasks],
        "task_id_digest": task_id_digest_value,
        "task_content_digest": task_content_digest_value,
        "elapsed_seconds": time.perf_counter() - started,
        "result_rows": len(all_rows),
        "selected_rows": len(selected_rows),
        "claim_boundary": "ARC2 evaluation is untouched; cohorts are ARC2-training mechanism probes, not a full benchmark score; timing is not PPA.",
        "artifacts": {
            name: {"bytes": (output_dir / name).stat().st_size, "sha256": sha256_file(output_dir / name)}
            for name in artifact_names
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(_json_safe(metadata), indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(output_dir)


if __name__ == "__main__":
    main()
