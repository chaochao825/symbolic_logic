"""Data-bound plotting script for the symbolic-logic experiment suite.

Reads only CSV files under ../results and writes vector PDF figures next to this
script.  No numerical value is embedded in the plotting code.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = Path(__file__).resolve().parent
COLORS = {
    "GateBeam": "#1f77b4",
    "MLP": "#ff7f0e",
    "TruthMemorizer": "#7f7f7f",
    "OracleRule": "#2ca02c",
    "float_tnorm": "#d62728",
    "bool_numpy": "#1f77b4",
    "packed_kernel": "#2ca02c",
    "packed_end_to_end": "#9467bd",
    "OracleGate+BFS": "#1f77b4",
    "FixedK=4GateCircuit": "#d62728",
    "HardGate": "#d62728",
    "SoftProbabilityCircuit": "#1f77b4",
    "dense_boolean_gate": "#d62728",
    "indexed_gate_filter": "#2ca02c",
    "LearnedSoftGate": "#1f77b4",
    "LearnedHardenedGate": "#d62728",
    "LearnedSoftGate+BFS": "#1f77b4",
    "LearnedHardenedGate+BFS": "#d62728",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def finish(fig: plt.Figure, name: str) -> None:
    for axis in fig.axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / name, format="pdf")
    fig.savefig(OUT / name.replace(".pdf", ".png"), format="png", dpi=180)
    plt.close(fig)


def plot_predicate() -> None:
    data = pd.read_csv(RESULTS / "predicate_results.csv")
    data = data[data["train_fraction"] == 0.50]
    methods = ["GateBeam", "MLP", "TruthMemorizer", "OracleRule"]
    families = ["compositional", "xor_heavy", "random_lut"]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0), sharey=True)
    for axis, family, panel in zip(axes, families, ("(a)", "(b)", "(c)")):
        subset = data[data["family"] == family]
        aggregate = subset.groupby("method")["ood_balanced_accuracy"].agg(["mean", "std"])
        positions = np.arange(len(methods))
        values = [aggregate.loc[m, "mean"] for m in methods]
        errors = [0 if np.isnan(aggregate.loc[m, "std"]) else aggregate.loc[m, "std"] for m in methods]
        axis.bar(positions, values, yerr=errors, capsize=2.5, color=[COLORS[m] for m in methods])
        axis.set_xticks(positions, ["Gate\nBeam", "MLP", "Memo", "Oracle"])
        axis.set_ylim(0, 1.06)
        axis.set_xlabel(f"{panel} {family.replace('_', ' ')}")
        axis.grid(axis="y", linestyle="--", alpha=0.25)
    axes[0].set_ylabel("OOD balanced accuracy")
    finish(fig, "predicate_accuracy.pdf")


def plot_efficiency() -> None:
    data = pd.read_csv(RESULTS / "efficiency_results.csv")
    methods = ["float_tnorm", "bool_numpy", "packed_kernel", "packed_end_to_end"]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.0))
    for method in methods:
        subset = data[data["method"] == method].sort_values("samples")
        label = method.replace("_", " ")
        axes[0].plot(subset["samples"], subset["median_ms"], marker="o", label=label, color=COLORS[method])
        axes[1].plot(subset["samples"], subset["explicit_buffer_bytes"], marker="o", label=label, color=COLORS[method])
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Batch samples")
    axes[0].set_ylabel("Median time (ms)")
    axes[0].legend(loc="upper left")
    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Batch samples")
    axes[1].set_ylabel("Explicit live buffers (bytes)")
    axes[1].legend(loc="upper left")
    finish(fig, "efficiency.pdf")


def plot_reachability() -> None:
    data = pd.read_csv(RESULTS / "reachability_results.csv")
    methods = ["OracleGate+BFS", "FixedK=4GateCircuit"]
    widths = sorted(data["width"].unique())
    fig, axes = plt.subplots(1, len(widths), figsize=(9.3, 3.0), sharey=True)
    for axis, width, panel in zip(axes, widths, ("(a)", "(b)", "(c)")):
        for method in methods:
            subset = data[(data["method"] == method) & (data["width"] == width)].sort_values("layers")
            axis.plot(
                subset["layers"],
                subset["balanced_accuracy"],
                marker="o",
                label=method.replace("OracleGate+BFS", "Oracle gate + BFS").replace("FixedK=4GateCircuit", "Fixed K=4 gate circuit"),
                color=COLORS[method],
            )
        axis.axvline(4, color="black", linestyle="--", linewidth=0.9, alpha=0.6)
        axis.text(4.12, 0.54, "K=4", fontsize=8)
        axis.set_ylim(0.42, 1.05)
        axis.set_xlabel(f"{panel} path length (width={width})")
        axis.grid(axis="y", linestyle="--", alpha=0.25)
    axes[0].set_ylabel("Balanced accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.06))
    finish(fig, "reachability.pdf")


def plot_noise() -> None:
    data = pd.read_csv(RESULTS / "noise_results.csv")
    methods = ["HardGate", "SoftProbabilityCircuit"]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.0))
    for method in methods:
        aggregate = data[data["method"] == method].groupby("epsilon")[["accuracy", "brier"]].mean().reset_index()
        label = "hard gate" if method == "HardGate" else "soft probability circuit"
        axes[0].plot(aggregate["epsilon"], aggregate["accuracy"], marker="o", label=label, color=COLORS[method])
        axes[1].plot(aggregate["epsilon"], aggregate["brier"], marker="o", label=label, color=COLORS[method])
    axes[0].set_xlabel("Predicate bit-flip rate")
    axes[0].set_ylabel("Accuracy")
    axes[1].set_xlabel("Predicate bit-flip rate")
    axes[1].set_ylabel("Brier score (lower is better)")
    axes[0].legend(loc="lower left")
    axes[1].legend(loc="upper left")
    finish(fig, "grounding_noise.pdf")


def plot_relation_filter() -> None:
    data = pd.read_csv(RESULTS / "relation_filter_results.csv")
    methods = ["dense_boolean_gate", "indexed_gate_filter"]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.0))
    for method in methods:
        aggregate = data[data["method"] == method].groupby("objects")[["median_ms", "candidate_ratio"]].mean().reset_index()
        label = method.replace("_", " ")
        axes[0].plot(aggregate["objects"], aggregate["median_ms"], marker="o", label=label, color=COLORS[method])
        axes[1].plot(aggregate["objects"], aggregate["candidate_ratio"], marker="o", label=label, color=COLORS[method])
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Objects")
    axes[0].set_ylabel("Median time (ms)")
    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Objects")
    axes[1].set_ylabel("Fraction of directed pairs evaluated")
    axes[0].legend(loc="upper left")
    axes[1].legend(loc="upper right")
    finish(fig, "relation_filter.pdf")


def plot_learned_gate() -> None:
    """Plot the newly measured soft-to-hard and learned-BFS interfaces."""
    local = pd.read_csv(RESULTS / "learned_gate_results.csv")
    graph = pd.read_csv(RESULTS / "learned_gate_bfs_results.csv")
    local = local[local["train_fraction"] == 0.50]
    graph = graph[graph["train_fraction"] == 0.50]
    methods = ["LearnedSoftGate", "LearnedHardenedGate"]
    bfs_methods = ["LearnedSoftGate+BFS", "LearnedHardenedGate+BFS"]
    aggregate = local.groupby("method")["ood_balanced_accuracy"].agg(["mean", "std"])
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.0))
    positions = np.arange(len(methods))
    axes[0].bar(
        positions,
        [aggregate.loc[m, "mean"] for m in methods],
        yerr=[aggregate.loc[m, "std"] for m in methods],
        capsize=2.5,
        color=[COLORS[m] for m in methods],
    )
    axes[0].set_xticks(positions, ["soft gate", "hardened gate"])
    axes[0].set_ylim(0, 1.06)
    axes[0].set_ylabel("Local OOD balanced accuracy")
    axes[0].set_xlabel("(a) fixed-wiring hardening")
    axes[0].grid(axis="y", linestyle="--", alpha=0.25)
    for method in bfs_methods:
        subset = graph[graph["method"] == method].groupby("layers")["balanced_accuracy"].agg(["mean", "std"]).reset_index()
        axes[1].plot(subset["layers"], subset["mean"], marker="o", label=method.replace("Learned", "").replace("+BFS", " + BFS"), color=COLORS[method])
    axes[1].set_ylim(0.42, 1.05)
    axes[1].set_xlabel("(b) path length")
    axes[1].set_ylabel("Graph balanced accuracy")
    axes[1].legend(loc="lower left")
    axes[1].grid(axis="y", linestyle="--", alpha=0.25)
    finish(fig, "learned_gate_integration.pdf")


def main() -> None:
    setup_style()
    plot_predicate()
    plot_efficiency()
    plot_reachability()
    plot_noise()
    plot_relation_filter()
    plot_learned_gate()


if __name__ == "__main__":
    main()
