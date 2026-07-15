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
    state_transition = pd.read_csv(RESULTS / "state_transition_results.csv")
    summary += emit_grouped(
        state_transition,
        ["width", "path_length", "method"],
        ["accuracy", "balanced_accuracy", "positive_accuracy", "negative_accuracy", "median_query_ms", "mean_loop_iterations"],
        "state_transition",
        rng,
    )
    planning = pd.read_csv(RESULTS / "planning_frontier_results.csv")
    summary += emit_grouped(
        planning,
        ["n_bits", "goal_depth", "method"],
        ["positive_accuracy", "negative_accuracy", "balanced_accuracy", "median_pair_ms", "mean_expansions"],
        "planning_frontier",
        rng,
    )
    scaling = pd.read_csv(RESULTS / "noncompressible_scaling_results.csv")
    summary += emit_grouped(
        scaling,
        ["family", "n_bits", "method"],
        ["holdout_accuracy", "full_accuracy", "fit_seconds", "gate_count"],
        "noncompressible_scaling",
        rng,
    )
    probability = pd.read_csv(RESULTS / "probability_marginalization_results.csv")
    summary += emit_grouped(
        probability,
        ["experiment", "n_bits", "method"],
        ["exact_probability", "estimate", "absolute_error", "elapsed_ms"],
        "probability_marginalization",
        rng,
    )
    end_to_end = pd.read_csv(RESULTS / "end_to_end_gridworld_results.csv")
    summary += emit_grouped(
        end_to_end,
        ["condition", "grid_size", "method"],
        [
            "accuracy",
            "balanced_accuracy",
            "brier",
            "cell_grounding_accuracy",
            "cell_grounding_ece",
            "source_localization_accuracy",
            "target_localization_accuracy",
            "fallback_rate",
            "encoder_ms_per_image",
            "median_solver_ms",
        ],
        "end_to_end_gridworld",
        rng,
    )
    logic_discovery = pd.read_csv(RESULTS / "logic_discovery_results.csv")
    summary += emit_grouped(
        logic_discovery,
        ["suite", "task", "method"],
        [
            "test_balanced_accuracy",
            "operator_recovered",
            "inputs_recovered",
            "topology_recovered",
            "rejected_hardening",
            "description_bits",
            "gate_count",
            "fit_seconds",
            "margin",
            "inference_us_per_1000",
            "validation_balanced_accuracy",
        ],
        "logic_discovery",
        rng,
    )
    rate_reduction = pd.read_csv(RESULTS / "rate_reduction_results.csv")
    summary += emit_grouped(
        rate_reduction,
        ["suite", "task", "method"],
        [
            "global_rate_bits",
            "within_rate_bits",
            "rate_reduction_bits",
            "zero_fraction",
            "subspace_coherence",
            "empirical_code_entropy_bits",
            "fixed_code_bits",
            "reconstruction_mse",
            "gate_count",
            "boolean_accuracy",
        ],
        "rate_reduction",
        rng,
    )
    discrete_function = pd.read_csv(RESULTS / "discrete_function_mdl_results.csv")
    discrete_function = discrete_function[discrete_function["selected"] == 1]
    summary += emit_grouped(
        discrete_function,
        ["family", "n_inputs", "best_function_language"],
        ["best_function_bits", "saving_vs_truth_table_bits", "anf_terms", "anf_degree", "best_robdd_nodes"],
        "discrete_function_mdl",
        rng,
    )
    discrete_task = pd.read_csv(RESULTS / "discrete_task_mdl_results.csv")
    discrete_task = discrete_task[discrete_task["selected"] == 1]
    summary += emit_grouped(
        discrete_task,
        ["family", "n_inputs", "language"],
        ["model_bits", "residual_bits", "total_bits", "errors", "safe_compression_gain_bits"],
        "discrete_task_mdl",
        rng,
    )
    discrete_representation = pd.read_csv(RESULTS / "discrete_representation_code_results.csv")
    summary += emit_grouped(
        discrete_representation,
        ["case", "code_family"],
        [
            "global_bits",
            "conditional_bits_side_info",
            "signed_reduction_side_info_bits",
            "gain_vs_global_route_side_info_bits",
            "routed_global_baseline_bits",
            "routed_best_bits",
            "signed_reduction_labels_encoded_bits",
            "asymmetric_helper_control_bits",
        ],
        "discrete_representation_code",
        rng,
    )
    exact_balanced = pd.read_csv(RESULTS / "exact_formula_balanced4_results.csv")
    exact_histogram = (
        exact_balanced.groupby("minimum_formula_gates", as_index=False)
        .size()
        .rename(columns={"size": "function_count"})
        .sort_values("minimum_formula_gates")
    )
    exact_histogram["population_fraction"] = exact_histogram["function_count"] / len(exact_balanced)
    exact_histogram["cumulative_fraction"] = exact_histogram["population_fraction"].cumsum()
    summary += emit_grouped(
        exact_histogram,
        ["minimum_formula_gates"],
        ["function_count", "population_fraction", "cumulative_fraction"],
        "exact_formula_balanced4",
        rng,
    )
    ca_local = pd.read_csv(RESULTS / "cellular_automata_local_rule_results.csv")
    ca_life = ca_local[ca_local["task"] == "game_of_life"]
    summary += emit_grouped(
        ca_life,
        ["task", "method"],
        ["local_accuracy", "local_balanced_accuracy", "cell_accuracy", "exact_trajectory", "first_divergence_step", "gate_count", "fit_seconds"],
        "cellular_automata_local",
        rng,
    )
    ca_tasks = pd.read_csv(RESULTS / "cellular_automata_task_results.csv")
    ca_collective = ca_tasks[ca_tasks["task"].isin(["density_classification", "global_synchronization"])]
    summary += emit_grouped(
        ca_collective,
        ["task", "method", "width"],
        ["accuracy", "unresolved_rate", "runtime_seconds", "published_accuracy", "anf_description_bits"],
        "cellular_automata_collective",
        rng,
    )
    ca_path = ca_tasks[ca_tasks["task"] == "boolean_pathfinding"]
    summary += emit_grouped(
        ca_path,
        ["case", "method", "width"],
        ["accuracy", "distance_exact", "unresolved_rate", "runtime_seconds", "steps"],
        "cellular_automata_pathfinding",
        rng,
    )
    ca_official = pd.read_csv(RESULTS / "cellular_automata_official_results.csv")
    summary += emit_grouped(
        ca_official,
        ["task", "condition", "size"],
        ["local_accuracy", "channel0_accuracy", "full_state_accuracy", "exact_final_state", "runtime_seconds", "logic_node_count", "critical_depth"],
        "cellular_automata_official",
        rng,
    )
    ca_complexity = pd.read_csv(RESULTS / "cellular_automata_complexity_results.csv")
    summary += emit_grouped(
        ca_complexity,
        ["experiment", "task", "method"],
        ["model_description_bits", "residual_bits", "single_state_storage_bits", "expected_cell_updates", "expected_gate_evaluations", "amortized_model_bits_per_cell_update"],
        "cellular_automata_complexity",
        rng,
    )
    pd.DataFrame(summary).to_csv(RESULTS / "summary_metrics.csv", index=False, lineterminator="\n")
    print(f"wrote {len(summary)} aggregate rows to {RESULTS / 'summary_metrics.csv'}")


if __name__ == "__main__":
    main()
