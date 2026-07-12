"""Train and evaluate the pixel-to-symbol-to-solver gridworld model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from neurosymbolic_gridworld import (
    LearnedBinaryGate,
    PatchMLPEncoder,
    collect_edge_training_data,
    evaluate_methods,
    extract_patches,
    infer_scores,
    make_dataset,
    tune_hybrid,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MODEL_DIR = RESULTS / "gridworld_models"


def git_state() -> dict[str, object]:
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip())
        return {"commit": revision, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unavailable", "dirty": None}


def save_model(seed: int, encoder: PatchMLPEncoder, gate: LearnedBinaryGate, soft_threshold: float, confidence_threshold: float) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        MODEL_DIR / f"seed_{seed}_encoder.npz",
        w1=encoder.w1,
        b1=encoder.b1,
        w2=encoder.w2,
        b2=encoder.b2,
        temperature=np.asarray(encoder.temperature),
    )
    (MODEL_DIR / f"seed_{seed}_gate.json").write_text(
        json.dumps(
            {
                "selected_op": gate.selected_op(),
                "logits": gate.logits.tolist(),
                "weights": gate.weights().tolist(),
                "soft_threshold": soft_threshold,
                "confidence_threshold": confidence_threshold,
                "input_predicates": ["source_passable_probability", "target_passable_probability"],
                "candidate_topology": "four-neighbor grid adjacency",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def run(mode: str) -> pd.DataFrame:
    patch_size = 4
    seeds = range(1 if mode == "smoke" else 3)
    n_train = 320 if mode == "smoke" else 1200
    n_validation = 100 if mode == "smoke" else 300
    n_test = 120 if mode == "smoke" else 400
    encoder_steps = 260 if mode == "smoke" else 650
    rows: list[dict] = []
    for seed in seeds:
        train = make_dataset(n_train, 8, patch_size, 100_000 + seed, "clean")
        validation = make_dataset(n_validation, 8, patch_size, 110_000 + seed, "clean")
        encoder = PatchMLPEncoder(patch_size * patch_size * 3, hidden=40, seed=200_000 + seed, lr=0.008)
        train_patches = extract_patches(train.images, 8, patch_size)
        validation_patches = extract_patches(validation.images, 8, patch_size)
        encoder.fit(train_patches, train.cell_labels.reshape(-1), steps=encoder_steps, batch_size=1024)
        encoder.calibrate_temperature(validation_patches, validation.cell_labels.reshape(-1))
        a, b, edge_labels = collect_edge_training_data(train, encoder)
        gate = LearnedBinaryGate(seed=300_000 + seed).fit(a, b, edge_labels, steps=700)
        validation_scores = infer_scores(validation, encoder, gate)
        soft_threshold, confidence_threshold = tune_hybrid(validation.task_labels, validation_scores)
        save_model(seed, encoder, gate, soft_threshold, confidence_threshold)
        test_sets = [
            make_dataset(n_test, 8, patch_size, 120_000 + seed, "clean"),
            make_dataset(n_test, 8, patch_size, 130_000 + seed, "correlated_occlusion"),
            make_dataset(n_test, 10, patch_size, 140_000 + seed, "size_ood"),
        ]
        for dataset in test_sets:
            scores = infer_scores(dataset, encoder, gate)
            rows.extend(evaluate_methods(dataset, scores, soft_threshold, confidence_threshold, seed, gate, encoder))
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "end_to_end_gridworld_results.csv", index=False)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    initial_git_state = git_state()
    started = time.perf_counter()
    frame = run(args.mode)
    source_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (ROOT / "src").glob("*.py")}
    metadata = {
        "mode": args.mode,
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "elapsed_seconds": time.perf_counter() - started,
        "rows": len(frame),
        "conditions": sorted(frame["condition"].unique().tolist()),
        "methods": sorted(frame["method"].unique().tolist()),
        "git_at_start": initial_git_state,
        "dataset": {
            "train_grid_size": 8,
            "size_ood_grid_size": 10,
            "patch_size": 4,
            "full_train_samples_per_seed": 1200,
            "full_validation_samples_per_seed": 300,
            "full_test_samples_per_condition_per_seed": 400,
        },
        "source_sha256": source_hashes,
    }
    (RESULTS / "end_to_end_gridworld_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
