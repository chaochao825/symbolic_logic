"""Run the bounded symbolic-logic experiment suite.

Usage (from the repository root):
    ..\\..\\work\\symbolic_env\\Scripts\\python.exe src\\run_experiments.py --mode smoke

All summary tables are written as CSV so figures and the report can be checked
against the measured artifacts rather than hand-entered values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from logic_core import (
    RULES,
    GateBeamSynthesizer,
    TinyMLP,
    TruthMemorizer,
    accuracy,
    all_assignments,
    balanced_accuracy,
    binary_metrics,
    brier_score,
    compositional_rule,
    hamming_weight,
    packed_compositional_rule,
    soft_probability_rule,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def timed(function: Callable[[], object], repeats: int = 15, warmup: int = 3) -> tuple[float, float, list[float]]:
    """Return median, IQR, and raw execution times in milliseconds."""
    for _ in range(warmup):
        function()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        function()
        samples.append((time.perf_counter_ns() - start) / 1_000_000)
    values = np.asarray(samples)
    return float(np.median(values)), float(np.percentile(values, 75) - np.percentile(values, 25)), [float(value) for value in values]


def write_csv(rows: list[dict], filename: str) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / filename, index=False)
    return frame


def run_predicate_learning(mode: str) -> pd.DataFrame:
    x_full = all_assignments(8)
    weights = hamming_weight(x_full)
    middle = np.flatnonzero((weights >= 2) & (weights <= 6))
    ood = np.flatnonzero((weights <= 1) | (weights >= 7))
    fractions = (0.50, 0.75) if mode == "smoke" else (0.25, 0.50, 0.75)
    seeds = range(3 if mode == "smoke" else 10)
    rows: list[dict] = []
    for family, rule in RULES.items():
        y_full = rule(x_full)
        for fraction in fractions:
            for seed in seeds:
                split_rng = np.random.default_rng(10_000 * (list(RULES).index(family) + 1) + int(fraction * 100) + seed)
                permutation = split_rng.permutation(middle)
                n_train = int(round(len(middle) * fraction))
                train_index = permutation[:n_train]
                iid_index = permutation[n_train:]
                x_train, y_train = x_full[train_index], y_full[train_index]
                methods: list[tuple[str, object, float, int | float, int | float, str]] = []
                gate = GateBeamSynthesizer(max_depth=4, beam_width=192).fit(x_train, y_train)
                assert gate.expression is not None
                methods.append(("GateBeam", gate, gate.fit_seconds, gate.expression.gates, gate.expression.depth, gate.expression.text))
                mlp = TinyMLP(8, seed=seed + 100, steps=1400).fit(x_train, y_train)
                methods.append(("MLP", mlp, mlp.fit_seconds, np.nan, np.nan, ""))
                memorizer = TruthMemorizer().fit(x_train, y_train)
                methods.append(("TruthMemorizer", memorizer, 0.0, np.nan, np.nan, ""))
                for method_name, method, fit_seconds, gates, depth, expression in methods:
                    prediction = method.predict(x_full)  # type: ignore[attr-defined]
                    ood_metrics = binary_metrics(y_full[ood], prediction[ood])
                    rows.append(
                        {
                            "family": family,
                            "train_fraction": fraction,
                            "seed": seed,
                            "method": method_name,
                            "train_accuracy": accuracy(y_train, prediction[train_index]),
                            "iid_accuracy": accuracy(y_full[iid_index], prediction[iid_index]),
                            "ood_accuracy": accuracy(y_full[ood], prediction[ood]),
                            "ood_balanced_accuracy": balanced_accuracy(y_full[ood], prediction[ood]),
                            "ood_tpr": ood_metrics["tpr"],
                            "ood_tnr": ood_metrics["tnr"],
                            "ood_f1": ood_metrics["f1"],
                            "ood_mcc": ood_metrics["mcc"],
                            "full_accuracy": accuracy(y_full, prediction),
                            "full_balanced_accuracy": balanced_accuracy(y_full, prediction),
                            "fit_seconds": fit_seconds,
                            "gate_count": gates,
                            "depth": depth,
                            "expression": expression,
                        }
                    )
                # The oracle is execution-only; it provides the semantic ceiling.
                rows.append(
                    {
                        "family": family,
                        "train_fraction": fraction,
                        "seed": seed,
                        "method": "OracleRule",
                        "train_accuracy": 1.0,
                        "iid_accuracy": 1.0,
                        "ood_accuracy": 1.0,
                        "ood_balanced_accuracy": 1.0,
                        "ood_tpr": 1.0,
                        "ood_tnr": 1.0,
                        "ood_f1": 1.0,
                        "ood_mcc": 1.0,
                        "full_accuracy": 1.0,
                        "full_balanced_accuracy": 1.0,
                        "fit_seconds": 0.0,
                        "gate_count": np.nan,
                        "depth": np.nan,
                        "expression": "known target rule (not learned)",
                    }
                )
    return write_csv(rows, "predicate_results.csv")


def _float_rule(x: np.ndarray) -> np.ndarray:
    """Float t-norm equivalent on binary inputs; it intentionally keeps float work."""
    x = np.asarray(x, dtype=np.float32)
    term_a = x[:, 0] * x[:, 1]
    term_b = x[:, 2] * x[:, 3]
    term_c = x[:, 4] * (1.0 - x[:, 5]) + (1.0 - x[:, 4]) * x[:, 5]
    body = 1.0 - (1.0 - term_a) * (1.0 - term_b) * (1.0 - term_c)
    return body * (1.0 - x[:, 6] * x[:, 7])


def run_efficiency(mode: str) -> tuple[pd.DataFrame, list[dict]]:
    sizes = (1_024, 16_384, 262_144) if mode == "smoke" else (1_024, 16_384, 262_144, 1_048_576)
    rows: list[dict] = []
    timing_rows: list[dict] = []
    rng = np.random.default_rng(20260711)
    for n_samples in sizes:
        x_bool = rng.integers(0, 2, size=(n_samples, 8), dtype=np.uint8)
        x_float = x_bool.astype(np.float32)
        packed = [np.packbits(x_bool[:, i], bitorder="little") for i in range(8)]
        expected = compositional_rule(x_bool)

        def float_eval() -> np.ndarray:
            return (_float_rule(x_float) >= 0.5).astype(np.uint8)

        def bool_eval() -> np.ndarray:
            return compositional_rule(x_bool)

        def packed_kernel() -> np.ndarray:
            return packed_compositional_rule(packed)

        def packed_end_to_end() -> np.ndarray:
            local_packed = [np.packbits(x_bool[:, i], bitorder="little") for i in range(8)]
            result = packed_compositional_rule(local_packed)
            return np.unpackbits(result, bitorder="little")[:n_samples]

        checks = {
            "float_tnorm": float_eval(),
            "bool_numpy": bool_eval(),
            "packed_end_to_end": packed_end_to_end(),
        }
        for name, value in checks.items():
            if not np.array_equal(value, expected):
                raise AssertionError(f"{name} changed Boolean semantics")
        packed_input_bytes = sum(column.nbytes for column in packed)
        packed_result_bytes = int(packed[0].nbytes)
        unpacked_result_bytes = int(expected.nbytes)
        measurements = [
            ("float_tnorm", float_eval, int(x_float.nbytes), int(x_float.nbytes + unpacked_result_bytes), "float source plus explicit output", True),
            ("bool_numpy", bool_eval, int(x_bool.nbytes), int(x_bool.nbytes + unpacked_result_bytes), "byte source plus explicit output", True),
            ("packed_kernel", packed_kernel, int(packed_input_bytes), int(packed_input_bytes + packed_result_bytes), "already-packed source plus packed result", False),
            (
                "packed_end_to_end",
                packed_end_to_end,
                int(x_bool.nbytes),
                int(x_bool.nbytes + packed_input_bytes + packed_result_bytes + unpacked_result_bytes),
                "byte source plus explicit packed input, packed result, and unpacked result",
                True,
            ),
        ]
        for method, function, source_input_bytes, explicit_buffer_bytes, storage_scope, output_unpacked in measurements:
            repeats = 12 if n_samples < 1_000_000 else 8
            median_ms, iqr_ms, raw_ms = timed(function, repeats=repeats)
            rows.append(
                {
                    "samples": n_samples,
                    "method": method,
                    "median_ms": median_ms,
                    "iqr_ms": iqr_ms,
                    "samples_per_second": n_samples / (median_ms / 1_000),
                    "source_input_bytes": source_input_bytes,
                    "explicit_buffer_bytes": explicit_buffer_bytes,
                    "storage_scope": storage_scope,
                    "output_unpacked": output_unpacked,
                    "semantic_check": True,
                }
            )
            timing_rows.extend(
                {
                    "benchmark": "efficiency",
                    "samples": n_samples,
                    "method": method,
                    "repeat": repeat,
                    "milliseconds": value,
                }
                for repeat, value in enumerate(raw_ms)
            )
    return write_csv(rows, "efficiency_results.csv"), timing_rows


def _dense_relation_count(color: np.ndarray, shape: np.ndarray, position: np.ndarray, window: int) -> int:
    valid = (color[:, None] == color[None, :]) & (shape[:, None] != shape[None, :])
    valid &= np.abs(position[:, None] - position[None, :]) <= window
    np.fill_diagonal(valid, False)
    return int(valid.sum())


def _indexed_relation_count(color: np.ndarray, shape: np.ndarray, position: np.ndarray, window: int) -> tuple[int, int]:
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, (c, p) in enumerate(zip(color, position)):
        buckets[(int(c), int(p // window))].append(index)
    valid_count = 0
    candidate_count = 0
    for index, (c, p) in enumerate(zip(color, position)):
        candidate_indices: list[int] = []
        bucket = int(p // window)
        for neighbor in (bucket - 1, bucket, bucket + 1):
            candidate_indices.extend(buckets.get((int(c), neighbor), []))
        for other in candidate_indices:
            if other == index:
                continue
            candidate_count += 1
            if shape[index] != shape[other] and abs(int(p) - int(position[other])) <= window:
                valid_count += 1
    return valid_count, candidate_count


def run_relation_filter(mode: str) -> tuple[pd.DataFrame, list[dict]]:
    sizes = (64, 128, 256, 512) if mode == "smoke" else (64, 128, 256, 512, 1024, 2048)
    rows: list[dict] = []
    timing_rows: list[dict] = []
    for seed in range(3 if mode == "smoke" else 8):
        rng = np.random.default_rng(9000 + seed)
        for n_objects in sizes:
            window = 8
            color = rng.integers(0, 8, size=n_objects, dtype=np.int16)
            shape = rng.integers(0, 4, size=n_objects, dtype=np.int16)
            position = rng.integers(0, max(window + 1, n_objects * window // 2), size=n_objects, dtype=np.int32)
            dense_count = _dense_relation_count(color, shape, position, window)
            indexed_count, candidate_count = _indexed_relation_count(color, shape, position, window)
            if dense_count != indexed_count:
                raise AssertionError("spatial indexing changed relation semantics")
            dense_ms, dense_iqr, dense_raw = timed(lambda: _dense_relation_count(color, shape, position, window), repeats=10)
            indexed_ms, indexed_iqr, indexed_raw = timed(lambda: _indexed_relation_count(color, shape, position, window), repeats=10)
            dense_pairs = n_objects * (n_objects - 1)
            rows.extend(
                [
                    {
                        "seed": seed,
                        "objects": n_objects,
                        "method": "dense_boolean_gate",
                        "median_ms": dense_ms,
                        "iqr_ms": dense_iqr,
                        "evaluated_pairs": dense_pairs,
                        "candidate_ratio": 1.0,
                        "valid_pairs": dense_count,
                        "semantic_check": True,
                    },
                    {
                        "seed": seed,
                        "objects": n_objects,
                        "method": "indexed_gate_filter",
                        "median_ms": indexed_ms,
                        "iqr_ms": indexed_iqr,
                        "evaluated_pairs": candidate_count,
                        "candidate_ratio": candidate_count / dense_pairs,
                        "valid_pairs": indexed_count,
                        "semantic_check": True,
                    },
                ]
            )
            timing_rows.extend(
                {
                    "benchmark": "relation_filter",
                    "objects": n_objects,
                    "seed": seed,
                    "method": "dense_boolean_gate",
                    "repeat": repeat,
                    "milliseconds": value,
                }
                for repeat, value in enumerate(dense_raw)
            )
            timing_rows.extend(
                {
                    "benchmark": "relation_filter",
                    "objects": n_objects,
                    "seed": seed,
                    "method": "indexed_gate_filter",
                    "repeat": repeat,
                    "milliseconds": value,
                }
                for repeat, value in enumerate(indexed_raw)
            )
    return write_csv(rows, "relation_filter_results.csv"), timing_rows


def _reachable(valid_edges: np.ndarray, source: int) -> np.ndarray:
    width = valid_edges.shape[1]
    active = np.zeros(width, dtype=bool)
    active[source] = True
    for layer in range(valid_edges.shape[0]):
        active = np.any(valid_edges[layer, active, :], axis=0) if active.any() else np.zeros(width, dtype=bool)
    return active


def _fixed_k_reachable(valid_edges: np.ndarray, source: int, k: int) -> np.ndarray:
    """Execute at most K layer transitions while retaining layer identity.

    When K is shorter than the graph, the terminal layer remains all false.
    This represents a genuine K-step unrolling rather than an early-return
    shortcut, and prevents reused local node indices from faking a hit.
    """
    width = valid_edges.shape[1]
    states = np.zeros((valid_edges.shape[0] + 1, width), dtype=bool)
    states[0, source] = True
    for layer in range(min(k, valid_edges.shape[0])):
        active = states[layer]
        states[layer + 1] = np.any(valid_edges[layer, active, :], axis=0) if active.any() else False
    return states[-1]


def _make_layered_graph(width: int, layers: int, rng: np.random.Generator) -> tuple[np.ndarray, int, int, int]:
    """Generate a graph with a forced positive path and an independently negative endpoint."""
    for _ in range(100):
        features = rng.integers(0, 2, size=(layers, width, width, 8), dtype=np.uint8)
        present = rng.random((layers, width, width)) < (0.60 / width)
        source = int(rng.integers(width))
        path = [source] + [int(rng.integers(width)) for _ in range(layers)]
        force_features = np.array([1, 1, 0, 0, 0, 0, 0, 0], dtype=np.uint8)
        for layer in range(layers):
            features[layer, path[layer], path[layer + 1]] = force_features
            present[layer, path[layer], path[layer + 1]] = True
        valid = present & compositional_rule(features.reshape(-1, 8)).reshape(layers, width, width).astype(bool)
        reachable = _reachable(valid, source)
        negatives = np.flatnonzero(~reachable)
        if len(negatives):
            return valid, source, path[-1], int(rng.choice(negatives))
    raise RuntimeError("could not produce a graph with both query classes")


def run_reachability(mode: str) -> pd.DataFrame:
    widths = (8, 16, 32)
    depths = (2, 4, 6, 8, 12)
    graphs_per_cell = 50 if mode == "smoke" else 200
    rng = np.random.default_rng(424242)
    rows: list[dict] = []
    for width in widths:
        for layers in depths:
            labels: list[int] = []
            bfs_predictions: list[int] = []
            fixed_predictions: list[int] = []
            bfs_ms: list[float] = []
            fixed_ms: list[float] = []
            for _ in range(graphs_per_cell):
                valid, source, positive_target, negative_target = _make_layered_graph(width, layers, rng)
                for label, target in ((1, positive_target), (0, negative_target)):
                    started = time.perf_counter_ns()
                    bfs_result = int(_reachable(valid, source)[target])
                    bfs_ms.append((time.perf_counter_ns() - started) / 1_000_000)
                    started = time.perf_counter_ns()
                    fixed_result = int(_fixed_k_reachable(valid, source, k=4)[target])
                    fixed_ms.append((time.perf_counter_ns() - started) / 1_000_000)
                    labels.append(label)
                    bfs_predictions.append(bfs_result)
                    fixed_predictions.append(fixed_result)
            for method, predictions, latencies in (
                ("OracleGate+BFS", bfs_predictions, bfs_ms),
                ("FixedK=4GateCircuit", fixed_predictions, fixed_ms),
            ):
                rows.append(
                    {
                        "width": width,
                        "layers": layers,
                        "queries": len(labels),
                        "method": method,
                        "accuracy": accuracy(np.asarray(labels), np.asarray(predictions)),
                        "balanced_accuracy": balanced_accuracy(np.asarray(labels), np.asarray(predictions)),
                        "positive_accuracy": accuracy(np.ones(sum(labels), dtype=np.uint8), np.asarray(predictions)[np.asarray(labels) == 1]),
                        "negative_accuracy": accuracy(np.zeros(len(labels) - sum(labels), dtype=np.uint8), np.asarray(predictions)[np.asarray(labels) == 0]),
                        "median_query_ms": float(np.median(latencies)),
                        "iqr_query_ms": float(np.percentile(latencies, 75) - np.percentile(latencies, 25)),
                    }
                )
    return write_csv(rows, "reachability_results.csv")


def run_noise(mode: str) -> pd.DataFrame:
    epsilons = (0.0, 0.01, 0.05, 0.10, 0.20)
    rows: list[dict] = []
    n_examples = 20_000 if mode == "smoke" else 100_000
    n_seeds = 3 if mode == "smoke" else 10
    for epsilon in epsilons:
        for seed in range(n_seeds):
            rng = np.random.default_rng(70_000 + seed)
            clean = rng.integers(0, 2, size=(n_examples, 8), dtype=np.uint8)
            label = compositional_rule(clean)
            flips = (rng.random(clean.shape) < epsilon).astype(np.uint8)
            observed = clean ^ flips
            hard_prediction = compositional_rule(observed)
            if epsilon == 0:
                posterior = observed.astype(float)
            else:
                posterior = np.where(observed == 1, 1.0 - epsilon, epsilon)
            soft_probability = soft_probability_rule(posterior)
            soft_prediction = (soft_probability >= 0.5).astype(np.uint8)
            rows.extend(
                [
                    {
                        "epsilon": epsilon,
                        "seed": seed,
                        "method": "HardGate",
                        "accuracy": accuracy(label, hard_prediction),
                        "balanced_accuracy": balanced_accuracy(label, hard_prediction),
                        "brier": brier_score(label, hard_prediction),
                        "mean_probability": float(hard_prediction.mean()),
                    },
                    {
                        "epsilon": epsilon,
                        "seed": seed,
                        "method": "SoftProbabilityCircuit",
                        "accuracy": accuracy(label, soft_prediction),
                        "balanced_accuracy": balanced_accuracy(label, soft_prediction),
                        "brier": brier_score(label, soft_probability),
                        "mean_probability": float(soft_probability.mean()),
                    },
                ]
            )
    return write_csv(rows, "noise_results.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    predicate = run_predicate_learning(args.mode)
    efficiency, efficiency_timing = run_efficiency(args.mode)
    relation, relation_timing = run_relation_filter(args.mode)
    reachability = run_reachability(args.mode)
    noise = run_noise(args.mode)
    timing_samples = efficiency_timing + relation_timing
    write_csv(timing_samples, "timing_samples.csv")
    source_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (ROOT / "src").glob("*.py")
    }
    metadata = {
        "mode": args.mode,
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "thread_environment": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "timing_policy": {"warmup": 3, "efficiency_repeats": "12 (8 for 2^20)", "relation_repeats": 10},
        "source_sha256": source_hashes,
        "elapsed_seconds": time.perf_counter() - started,
        "row_counts": {
            "predicate": len(predicate),
            "efficiency": len(efficiency),
            "relation": len(relation),
            "reachability": len(reachability),
            "noise": len(noise),
            "timing_samples": len(timing_samples),
        },
    }
    (RESULTS / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
