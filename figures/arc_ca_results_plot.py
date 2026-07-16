"""Plot ARC cellular-automaton results from committed, hash-checked run artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs"
OUT = Path(__file__).resolve().parent
FROZEN_COMMIT = "888573d15de3c865e339677df8d2e804e999ff88"
RUN_ORDER = (
    ("ARC1 train", "arc_ca_v1_arc1_training_full_210_888573d"),
    ("ARC1 eval", "arc_ca_v1_arc1_evaluation_full_210_888573d"),
    ("ARC2 train", "arc_ca_v1_arc2_training_full_210_888573d"),
    ("ARC2 eval", "arc_ca_v1_arc2_evaluation_full_210_888573d"),
)
COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "gray": "#777777",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9.5,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 7.8,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_csv_allow_empty(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def load_run(name: str) -> dict[str, object]:
    directory = RUNS / name
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if metadata["source_git_at_start"]["commit"] != FROZEN_COMMIT:
        raise ValueError(f"{name}: unexpected source commit")
    for filename, declared in metadata["artifacts"].items():
        path = directory / filename
        if path.stat().st_size != declared["bytes"] or _sha256(path) != declared["sha256"]:
            raise ValueError(f"{name}: artifact integrity failure for {filename}")
    return {
        "metadata": metadata,
        "tasks": _read_csv_allow_empty(directory / "arc_ca_task_results.csv"),
        "programs": _read_csv_allow_empty(directory / "arc_ca_program_results.csv"),
        "codecs": _read_csv_allow_empty(directory / "arc_ca_codec_results.csv"),
        "synthetic": _read_csv_allow_empty(directory / "arc_ca_synthetic_results.csv"),
    }


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(-0.14, 1.05, label, transform=axis.transAxes, fontweight="bold", va="top")


def plot_exact(axis: plt.Axes, loaded: list[tuple[str, dict[str, object]]]) -> None:
    values: list[tuple[float, float, float]] = []
    for _, run in loaded:
        tasks = run["tasks"]
        programs = run["programs"]
        joined = tasks[["task_id", "task_exact"]].merge(
            programs[["task_id", "task_exact"]], on="task_id", suffixes=("_table", "_program")
        )
        denominator = len(tasks)
        values.append(
            (
                100 * float(tasks["task_exact"].sum()) / denominator,
                100 * float(programs["task_exact"].sum()) / denominator,
                100
                * float(np.maximum(joined["task_exact_table"], joined["task_exact_program"]).sum())
                / denominator,
            )
        )
    x = np.arange(len(loaded))
    width = 0.24
    labels = ("sparse table", "bounded program", "two-family union")
    colors = (COLORS["blue"], COLORS["orange"], COLORS["green"])
    for index, (label, color) in enumerate(zip(labels, colors)):
        heights = [row[index] for row in values]
        bars = axis.bar(x + (index - 1) * width, heights, width, label=label, color=color)
        for bar, height in zip(bars, heights):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.08,
                f"{height:.1f}",
                ha="center",
                va="bottom",
                fontsize=6.8,
            )
    axis.set_xticks(x, [label.replace(" ", "\n") for label, _ in loaded])
    axis.set_ylim(0, 4.2)
    axis.set_ylabel("Full-task exact (%)")
    axis.grid(axis="y", linestyle="--", alpha=0.25)
    axis.legend(loc="upper right")
    panel_label(axis, "(a)")


def plot_failure_decomposition(axis: plt.Axes, loaded: list[tuple[str, dict[str, object]]]) -> None:
    categories = (
        ("exact", COLORS["green"]),
        ("representable, induction fail", COLORS["orange"]),
        ("selected, representation fail", COLORS["purple"]),
        ("refused by grammar", COLORS["gray"]),
    )
    shares = {label: [] for label, _ in categories}
    for _, run in loaded:
        tasks = run["tasks"]
        exact = tasks["task_exact"].eq(1)
        selected = tasks["selection_status"].eq("selected")
        representable = tasks["posthoc_oracle_local_exists"].eq(1)
        masks = (
            exact,
            representable & ~exact,
            selected & ~representable,
            ~selected,
        )
        if sum(int(mask.sum()) for mask in masks) != len(tasks):
            raise ValueError("failure decomposition does not cover each task exactly once")
        for (label, _), mask in zip(categories, masks):
            shares[label].append(100 * float(mask.sum()) / len(tasks))
    x = np.arange(len(loaded))
    bottom = np.zeros(len(loaded))
    for label, color in categories:
        values = np.asarray(shares[label])
        axis.bar(x, values, bottom=bottom, color=color, label=label)
        bottom += values
    axis.set_xticks(x, [label.replace(" ", "\n") for label, _ in loaded])
    axis.set_ylim(0, 100)
    axis.set_ylabel("Sparse-table outcome share (%)")
    axis.grid(axis="y", linestyle="--", alpha=0.20)
    axis.legend(loc="lower left", ncol=2)
    panel_label(axis, "(b)")


def plot_codec_latency(axis: plt.Axes, loaded: list[tuple[str, dict[str, object]]]) -> None:
    frames = []
    for _, run in loaded:
        frame = run["codecs"]
        frames.append(frame[frame["dataset"].notna()])
    codecs = pd.concat(frames, ignore_index=True)
    binary = (codecs["binary4_median_seconds"] / codecs["direct_median_seconds"]).to_numpy()
    onehot = (codecs["onehot_median_seconds"] / codecs["direct_median_seconds"]).to_numpy()
    plot = axis.boxplot(
        [binary, onehot],
        positions=[1, 2],
        widths=0.55,
        whis=(5, 95),
        showfliers=False,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.2},
    )
    for patch, color in zip(plot["boxes"], (COLORS["blue"], COLORS["purple"])):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    axis.axhline(1.0, color=COLORS["gray"], linestyle="--", linewidth=1.0, label="direct baseline")
    axis.set_yscale("log")
    axis.set_xticks([1, 2], ["binary4", "one-hot10"])
    axis.set_ylabel("CPU latency / direct categorical")
    axis.grid(axis="y", which="both", linestyle="--", alpha=0.22)
    axis.text(1, np.median(binary) * 1.10, f"median {np.median(binary):.2f}x", ha="center", fontsize=7.4)
    axis.text(2, np.median(onehot) * 1.10, f"median {np.median(onehot):.2f}x", ha="center", fontsize=7.4)
    axis.legend(loc="upper right")
    panel_label(axis, "(c)")


def plot_description_routes(axis: plt.Axes, loaded: list[tuple[str, dict[str, object]]]) -> None:
    arc2_training = next(run for label, run in loaded if label == "ARC2 train")
    synthetic = arc2_training["synthetic"].copy()
    synthetic = synthetic[synthetic["direct_reference_exact"].eq(1)].reset_index(drop=True)
    labels = {
        "identity_10color": "identity",
        "palette_cycle_10color": "palette",
        "copy_north": "copy",
        "conditional_recolor": "condition",
        "binary_dilation": "dilation",
        "binary_erosion": "erosion",
        "binary_parity": "parity",
        "random_binary_lut5": "random LUT",
        "iterative_dilation_horizon": "iterative",
    }
    synthetic = synthetic[synthetic["task"].isin(labels)]
    x = np.arange(len(synthetic))
    series = (
        ("structured prefix", synthetic["structured_description_bits_with_wrapper"], "o", COLORS["green"]),
        ("raw route", synthetic["raw_description_bits_with_wrapper"], "s", COLORS["orange"]),
        ("generic sparse table", synthetic["sparse_model_bits"] + 8, "^", COLORS["blue"]),
    )
    for label, values, marker, color in series:
        axis.scatter(x, values, label=label, marker=marker, color=color, s=28, zorder=3)
    axis.set_yscale("log")
    axis.set_xticks(x, [labels[name] for name in synthetic["task"]], rotation=34, ha="right")
    axis.set_ylabel("Declared model prefix (bits)")
    axis.grid(axis="y", which="both", linestyle="--", alpha=0.22)
    axis.legend(loc="upper left")
    panel_label(axis, "(d)")


def main() -> None:
    setup_style()
    loaded = [(label, load_run(name)) for label, name in RUN_ORDER]
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.0))
    plot_exact(axes[0, 0], loaded)
    plot_failure_decomposition(axes[0, 1], loaded)
    plot_codec_latency(axes[1, 0], loaded)
    plot_description_routes(axes[1, 1], loaded)
    fig.tight_layout(w_pad=1.8, h_pad=2.0)
    fig.savefig(OUT / "arc_ca_results.pdf", format="pdf")
    fig.savefig(OUT / "arc_ca_results.png", format="png", dpi=180)
    plt.close(fig)
    print(OUT / "arc_ca_results.pdf")


if __name__ == "__main__":
    main()
