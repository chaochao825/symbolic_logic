"""Produce reproducible aggregate tables and bootstrap intervals from raw CSVs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def mean_ci(values: pd.Series, rng: np.random.Generator, draws: int = 10_000) -> tuple[float, float, float, float, int]:
    array = values.to_numpy(dtype=float)
    n = len(array)
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if n > 1 else 0.0
    if n <= 1:
        return mean, std, mean, mean, n
    sampled = rng.choice(array, size=(draws, n), replace=True).mean(axis=1)
    low, high = np.quantile(sampled, (0.025, 0.975))
    return mean, std, float(low), float(high), n


def emit_grouped(frame: pd.DataFrame, group_columns: list[str], metric_columns: list[str], section: str, rng: np.random.Generator) -> list[dict]:
    rows: list[dict] = []
    for key, group in frame.groupby(group_columns, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        shared = dict(zip(group_columns, key))
        for metric in metric_columns:
            mean, std, low, high, n = mean_ci(group[metric], rng)
            rows.append({"section": section, **shared, "metric": metric, "mean": mean, "std": std, "ci95_low": low, "ci95_high": high, "n": n})
    return rows


def main() -> None:
    rng = np.random.default_rng(20260711)
    summary: list[dict] = []
    predicate = pd.read_csv(RESULTS / "predicate_results.csv")
    summary += emit_grouped(
        predicate,
        ["family", "train_fraction", "method"],
        ["iid_accuracy", "ood_accuracy", "ood_balanced_accuracy", "ood_tpr", "ood_tnr", "ood_f1", "ood_mcc", "full_accuracy", "fit_seconds", "gate_count"],
        "predicate",
        rng,
    )
    efficiency = pd.read_csv(RESULTS / "efficiency_results.csv")
    summary += emit_grouped(efficiency, ["samples", "method"], ["median_ms", "samples_per_second", "source_input_bytes", "explicit_buffer_bytes"], "efficiency", rng)
    relation = pd.read_csv(RESULTS / "relation_filter_results.csv")
    summary += emit_grouped(relation, ["objects", "method"], ["median_ms", "candidate_ratio", "evaluated_pairs"], "relation", rng)
    reachability = pd.read_csv(RESULTS / "reachability_results.csv")
    summary += emit_grouped(reachability, ["width", "layers", "method"], ["balanced_accuracy", "positive_accuracy", "negative_accuracy"], "reachability", rng)
    noise = pd.read_csv(RESULTS / "noise_results.csv")
    summary += emit_grouped(noise, ["epsilon", "method"], ["accuracy", "brier"], "noise", rng)
    learned_gate = pd.read_csv(RESULTS / "learned_gate_results.csv")
    summary += emit_grouped(
        learned_gate,
        ["train_fraction", "method"],
        ["full_accuracy", "full_balanced_accuracy", "ood_accuracy", "ood_balanced_accuracy", "ood_brier", "fit_seconds", "hardening_seconds", "soft_to_hard_ood_balanced_drop"],
        "learned_gate",
        rng,
    )
    learned_bfs = pd.read_csv(RESULTS / "learned_gate_bfs_results.csv")
    summary += emit_grouped(
        learned_bfs,
        ["train_fraction", "width", "layers", "method"],
        ["accuracy", "balanced_accuracy", "positive_accuracy", "negative_accuracy", "median_query_ms"],
        "learned_gate_bfs",
        rng,
    )
    pd.DataFrame(summary).to_csv(RESULTS / "summary_metrics.csv", index=False)
    print(f"wrote {len(summary)} aggregate rows to {RESULTS / 'summary_metrics.csv'}")


if __name__ == "__main__":
    main()
