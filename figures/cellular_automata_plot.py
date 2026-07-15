"""Generate the cellular-automata result figure from committed raw CSV files."""

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
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "gray": "#6B6B6B",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9.5,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 7.8,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(-0.14, 1.05, label, transform=axis.transAxes, fontweight="bold", va="top")


def plot_local_logic(axis: plt.Axes, local: pd.DataFrame) -> None:
    data = local[local["task"] == "game_of_life"].copy()
    order = ["FixedAND(center,north)", "GateBeam", "TinyMLP", "LifeLikeB3S23"]
    aggregate = data.groupby("method")[["local_balanced_accuracy", "cell_accuracy"]].mean()
    x = np.arange(len(order))
    width = 0.36
    axis.bar(
        x - width / 2,
        [aggregate.loc[method, "local_balanced_accuracy"] for method in order],
        width,
        color=COLORS["blue"],
        label="local balanced accuracy",
    )
    axis.bar(
        x + width / 2,
        [aggregate.loc[method, "cell_accuracy"] for method in order],
        width,
        color=COLORS["orange"],
        hatch="//",
        label="rollout cell accuracy",
    )
    axis.set_xticks(x, ["fixed\nAND", "Gate\nBeam", "MLP", "B3/S23"])
    axis.set_ylim(0.4, 1.04)
    axis.set_ylabel("Accuracy")
    axis.grid(axis="y", linestyle="--", alpha=0.25)
    axis.legend(loc="lower right")
    panel_label(axis, "(a)")


def plot_collective(axis: plt.Axes, tasks: pd.DataFrame, task: str) -> None:
    data = tasks[tasks["task"] == task]
    if task == "density_classification":
        methods = ["majority", "block_expand", "particle", "gkl"]
        palette = [COLORS["gray"], COLORS["orange"], COLORS["green"], COLORS["blue"]]
    else:
        methods = ["naive_oscillator", "phi_sync"]
        palette = [COLORS["gray"], COLORS["purple"]]
    for method, color in zip(methods, palette):
        subset = data[data["method"] == method]
        aggregate = subset.groupby("width")["accuracy"].agg(["mean", "std"]).reset_index()
        axis.errorbar(
            aggregate["width"],
            aggregate["mean"],
            yerr=aggregate["std"].fillna(0),
            marker="o",
            capsize=2,
            linewidth=1.5,
            color=color,
            label=method.replace("_", " "),
        )
        published = subset.groupby("width")["published_accuracy"].mean().dropna()
        if len(published):
            axis.scatter(
                published.index,
                published.values,
                marker="x",
                s=32,
                linewidths=1.2,
                color=color,
                alpha=0.75,
            )
    axis.set_xticks([149, 599, 999])
    axis.set_ylim(-0.03, 1.04)
    axis.set_xlabel("Lattice width N")
    axis.set_ylabel("Task success rate")
    axis.grid(axis="y", linestyle="--", alpha=0.25)
    axis.legend(loc="best")
    panel_label(axis, "(b)" if task == "density_classification" else "(c)")
    if task == "global_synchronization":
        axis.text(
            0.98,
            0.06,
            "decision M=2N\n+1 validation step",
            transform=axis.transAxes,
            ha="right",
            color=COLORS["gray"],
        )


def plot_cost(axis: plt.Axes, complexity: pd.DataFrame) -> None:
    selected = complexity[complexity["experiment"].isin(["E18", "E20", "E21", "E22"])].copy()
    # Aggregate exact repeated configurations to keep the panel readable.
    selected["family"] = selected["experiment"].map(
        {"E18": "official GoL", "E20": "collective LUT", "E21": "wavefront", "E22": "checkerboard"}
    )
    palette = {
        "official GoL": COLORS["red"],
        "collective LUT": COLORS["green"],
        "wavefront": COLORS["blue"],
        "checkerboard": COLORS["purple"],
    }
    for family, group in selected.groupby("family"):
        axis.scatter(
            group["expected_cell_updates"],
            group["model_description_bits"],
            s=22,
            alpha=0.55,
            color=palette[family],
            label=family,
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Cell updates per evaluated rollout")
    axis.set_ylabel("Declared model description (bits)")
    axis.grid(which="both", linestyle="--", alpha=0.20)
    axis.legend(loc="best")
    panel_label(axis, "(d)")


def main() -> None:
    setup_style()
    local = pd.read_csv(RESULTS / "cellular_automata_local_rule_results.csv")
    tasks = pd.read_csv(RESULTS / "cellular_automata_task_results.csv")
    complexity = pd.read_csv(RESULTS / "cellular_automata_complexity_results.csv")
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.8))
    plot_local_logic(axes[0, 0], local)
    plot_collective(axes[0, 1], tasks, "density_classification")
    plot_collective(axes[1, 0], tasks, "global_synchronization")
    plot_cost(axes[1, 1], complexity)
    fig.tight_layout(w_pad=2.0, h_pad=2.0)
    fig.savefig(OUT / "cellular_automata_results.pdf", format="pdf")
    fig.savefig(OUT / "cellular_automata_results.png", format="png", dpi=180)
    plt.close(fig)
    print(OUT / "cellular_automata_results.pdf")


if __name__ == "__main__":
    main()
