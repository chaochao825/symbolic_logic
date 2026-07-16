"""Run label-isolated DiffLogic-ARC development and confirmatory cohorts.

ARC-AGI-2 evaluation is intentionally rejected.  The consumed public
evaluation receipt belongs to the frozen categorical experiment and is never
reopened by this runner.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
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
except ImportError as error:  # pragma: no cover
    raise SystemExit("run_arc_difflogic.py requires PyTorch; see requirements-difflogic-arc.txt") from error


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


def synthetic_tasks() -> list[tuple[ArcProblem, ArcLabels, Path | None]]:
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

    marker_inputs = (
        _grid([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
        _grid([[1, 0, 0, 0], [0, 0, 0, 9], [0, 0, 0, 0], [0, 0, 0, 0]]),
        _grid([[0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 9], [0, 0, 0, 0]]),
    )
    marker_examples = tuple(ArcExample(item, _marker_rule(item)) for item in marker_inputs)
    marker_test = _grid([[0, 1, 0, 0], [0, 0, 0, 0], [9, 0, 0, 0], [0, 0, 0, 0]])

    scale_inputs = (_grid([[1, 2], [3, 4]]), _grid([[4, 0, 2], [1, 3, 0]]))
    scale_examples = tuple(ArcExample(item, _scale2(item)) for item in scale_inputs)
    scale_test = _grid([[2, 1], [0, 4], [3, 0]])

    tasks = [
        ("syn_local_dilation1", local_examples, local_test, _dilate(local_test, 1)),
        ("syn_hidden_dilation2", propagation_examples, propagation_test, _dilate(propagation_test, 2)),
        ("syn_global_marker", marker_examples, marker_test, _marker_rule(marker_test)),
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


def _baseline_rows(problem: ArcProblem, labels: ArcLabels) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    references = [np.asarray(output) for output in labels.test_outputs]
    sparse = select_demo_rule(problem.demonstrations)
    direct_predictions = (
        [predict_direct(sparse.rule, grid) for grid in problem.test_inputs]
        if sparse.rule is not None
        else []
    )
    if direct_predictions:
        predictions.append({"task_id": problem.task_id, "variant": "sparse", "seed": -1, "grids": direct_predictions})
        metrics = paired_grid_metrics(references, direct_predictions)
    else:
        metrics = {"pair_exact_count": 0.0, "pair_exact_rate": 0.0, "task_exact": 0.0, "cell_accuracy": 0.0}
    rows.append(
        {
            "task_id": problem.task_id,
            "variant": "sparse",
            "seed": -1,
            "status": sparse.status,
            **{"hard_" + key: value for key, value in metrics.items()},
            "selected_by_demo": 1.0,
        }
    )
    d4_predictions, d4 = augmented_sparse_predictions(problem)
    if d4_predictions:
        predictions.append({"task_id": problem.task_id, "variant": "d4_sparse", "seed": -1, "grids": d4_predictions})
        d4_metrics = paired_grid_metrics(references, d4_predictions)
    else:
        d4_metrics = {"pair_exact_count": 0.0, "pair_exact_rate": 0.0, "task_exact": 0.0, "cell_accuracy": 0.0}
    rows.append(
        {
            "task_id": problem.task_id,
            "variant": "d4_sparse",
            "seed": -1,
            **d4,
            **{"hard_" + key: value for key, value in d4_metrics.items()},
            "selected_by_demo": 1.0,
        }
    )
    return rows, predictions


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("synthetic", "arc"), required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--cohort", choices=tuple(COHORTS))
    parser.add_argument("--variants", default="dl1,dlr,dlo,dlf,mlp")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--augmentation", choices=("auto", "none", "d4"), default="auto")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confirmatory-config", type=Path)
    parser.add_argument("--max-tasks", type=int)
    args = parser.parse_args()

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
        tasks = load_arc_split(args.dataset_root, "training", task_ids=COHORTS[args.cohort])
    else:
        tasks = synthetic_tasks()
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
        if confirmation.get("source_commit") != source_head:
            parser.error("confirmatory config source_commit does not match HEAD")
        confirmation_sha = sha256_file(args.confirmatory_config)
    else:
        confirmation = None
        confirmation_sha = ""

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    training_config = TrainingConfig(
        epochs=args.epochs if args.epochs is not None else (40 if args.mode == "smoke" else 240),
        trace_every=10 if args.mode == "smoke" else 20,
        early_stop_patience=2 if args.mode == "smoke" else 3,
    )
    started = time.perf_counter()
    all_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    circuit_rows: list[dict[str, Any]] = []

    for task_number, (problem, labels, _) in enumerate(tasks, start=1):
        baseline_rows, baseline_predictions = _baseline_rows(problem, labels)
        all_rows.extend(baseline_rows)
        selected_rows.extend(baseline_rows)
        prediction_rows.extend(baseline_predictions)
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
                # Predictions already exist before labels enter this evaluator.
                evaluation = evaluate_candidate(candidate, labels)
                candidates.append((candidate, evaluation))
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
            chosen, chosen_evaluation = min(candidates, key=lambda item: candidate_selection_key(item[0]))
            for candidate, evaluation in candidates:
                row = _flat_candidate_record(candidate, evaluation, candidate is chosen)
                all_rows.append(row)
                if candidate is chosen:
                    selected_rows.append(row)
        print(f"[{task_number}/{len(tasks)}] {problem.task_id}", flush=True)

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
    task_paths = [path for _, _, path in tasks if path is not None]
    metadata = {
        "experiment": "arc_difflogic_v1",
        "suite": args.suite,
        "cohort": args.cohort or "synthetic",
        "mode": args.mode,
        "source_commit": source_head,
        "source_status_start": status_start,
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
        "task_id_digest": task_id_digest(task_paths) if task_paths else "synthetic",
        "task_content_digest": split_digest(task_paths) if task_paths else "synthetic",
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
