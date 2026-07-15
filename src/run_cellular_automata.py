"""Run the bounded cellular-automata extension.

The runner deliberately separates four evidence levels:

* exact local transition identification and recurrent rollout;
* replay of commit-pinned official hard DiffLogic-CA artifacts;
* classical collective CA tasks with published rule tables; and
* a Boolean wavefront algorithm compared with an independent BFS oracle.

Frozen-circuit replay is never labelled as retraining.  Description length,
state storage, recurrent work, and task error are written to separate columns.
"""

from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import math
import platform
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from boolean_mdl import (
    CircuitIR,
    CircuitNode,
    anf_description_bits,
    circuit_description_bits,
    elias_delta_bits,
    exact_formula_library,
    formula_to_circuit,
    residual_code_bits,
)
from cellular_automata import (
    BooleanWavefrontResult,
    ca_complexity_accounting,
    eca_lut,
    game_of_life_next_from_patches,
    game_of_life_rollout,
    game_of_life_step,
    moore_patches,
    radius_lut_step,
    radius_neighborhood_codes,
    trajectory_metrics,
)
from logic_core import GateBeamSynthesizer, TinyMLP, balanced_accuracy


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OFFICIAL_PAGES_COMMIT = "4c0246d9f7a2912cb7201f6bfe5fcda0fe373904"
OFFICIAL_NOTEBOOK_ORIGIN_REPRODUCTION_COMMIT = "c4f07027a2e05dd8c0f8e0fcfb31d2dcbd644558"

# Mitchell/Das author-source rule tables.  Each hexadecimal string is expanded
# from the output for 0000000 through the output for 1111111.
DENSITY_RULE_HEX = {
    "majority": "000101170117177f0117177f177f7fff",
    "block_expand": "0505408305c90101200b0efb94c7cff7",
    "particle": "0504058705000f77037755837bffb77f",
}
SYNC_RULE_HEX = {
    "phi_sync": "FEB1C6EAB8E0C4DA6484A5AAF410C8A0",
}
PUBLISHED_DENSITY = {
    "majority": {149: 0.000, 599: 0.000, 999: 0.000},
    "block_expand": {149: 0.652, 599: 0.515, 999: 0.503},
    "particle": {149: 0.769, 599: 0.725, 999: 0.714},
}
PUBLISHED_SYNC = {
    "naive_oscillator": {149: 0.54, 599: 0.09, 999: 0.02},
    "phi_sync": {149: 1.00, 599: 1.00, 999: 1.00},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state() -> dict:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unavailable", "dirty": None}


def assignments(n_inputs: int) -> np.ndarray:
    codes = np.arange(1 << n_inputs, dtype=np.uint64)[:, None]
    return ((codes >> np.arange(n_inputs, dtype=np.uint64)) & 1).astype(np.uint8)


def lexicographic_hex_lut(text: str) -> np.ndarray:
    """Decode outputs ordered from the all-zero to the all-one neighborhood."""
    if len(text) != 32 or any(character not in "0123456789abcdefABCDEF" for character in text):
        raise ValueError("a radius-3 binary rule must be exactly 32 hexadecimal digits")
    bits = "".join(f"{int(character, 16):04b}" for character in text)
    return np.fromiter((int(bit) for bit in bits), dtype=np.uint8, count=128)


def gkl_density_lut() -> np.ndarray:
    """GKL rule under bit0=+3,...,bit3=center,...,bit6=-3."""
    codes = np.arange(128, dtype=np.uint8)
    bits = ((codes[:, None] >> np.arange(7, dtype=np.uint8)) & 1).astype(np.uint8)

    def majority(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        return ((a + b + c) >= 2).astype(np.uint8)

    center = bits[:, 3]
    left_vote = majority(center, bits[:, 4], bits[:, 6])
    right_vote = majority(center, bits[:, 2], bits[:, 0])
    return np.where(center == 0, left_vote, right_vote).astype(np.uint8)


def fixed_and_life(patches: np.ndarray) -> np.ndarray:
    """Declared weak ablation: center AND north, not an optimized AND rule."""
    return (np.asarray(patches, dtype=np.uint8)[..., 4] & np.asarray(patches, dtype=np.uint8)[..., 1]).astype(np.uint8)


def predictor_rollout(
    initial: np.ndarray,
    steps: int,
    predictor: Callable[[np.ndarray], np.ndarray],
    boundary: str = "periodic",
) -> np.ndarray:
    grid = np.asarray(initial, dtype=np.uint8).copy()
    trajectory = np.empty((steps + 1, *grid.shape), dtype=np.uint8)
    trajectory[0] = grid
    for step in range(steps):
        patches = moore_patches(grid, boundary=boundary)
        grid = np.asarray(predictor(patches.reshape(-1, 9)), dtype=np.uint8).reshape(grid.shape)
        trajectory[step + 1] = grid
    return trajectory


def circuit_eca_step(state: np.ndarray, circuit: CircuitIR) -> np.ndarray:
    codes = radius_neighborhood_codes(state, radius=1, boundary="periodic")
    features = ((codes.reshape(-1, 1) >> np.arange(3, dtype=np.uint64)) & 1).astype(np.uint8)
    return circuit.evaluate(features).reshape(state.shape)


def circuit_eca_rollout(initial: np.ndarray, steps: int, circuit: CircuitIR) -> np.ndarray:
    state = np.asarray(initial, dtype=np.uint8).copy()
    trajectory = np.empty((steps + 1, *state.shape), dtype=np.uint8)
    trajectory[0] = state
    for step in range(steps):
        state = circuit_eca_step(state, circuit)
        trajectory[step + 1] = state
    return trajectory


def ca_wrapper_bits(*, dimension: int, radius: int, channels: int, boundary: str, steps: int) -> int:
    """A declared prefix wrapper, separate from the local-rule language."""
    if boundary not in ("fixed", "periodic"):
        raise ValueError(boundary)
    # dimension route (1D/2D/other), radius, channels, boundary, synchronous
    # schedule, multi-channel layout, and a finite horizon. Initial data are
    # public side information. The declared CA language maps 2D radius one to
    # the full Moore neighborhood and 1D radius r to all 2r+1 offsets.
    return 2 + elias_delta_bits(radius + 1) + elias_delta_bits(channels) + 1 + 1 + 1 + elias_delta_bits(steps + 1)


def append_complexity(
    rows: list[dict],
    *,
    experiment: str,
    task: str,
    method: str,
    spatial_shape: Sequence[int],
    steps: int,
    gates_per_cell: int,
    state_bits_per_cell: int,
    model_description_bits: int,
    residual_bits: int,
    seed: int = -1,
    update_fraction: float = 1.0,
    total_network_gates: int | None = None,
    reachable_network_gates: int | None = None,
    active_network_gates: int | None = None,
    note: str = "",
) -> None:
    accounting = ca_complexity_accounting(
        spatial_shape,
        steps,
        gates_per_cell,
        state_bits_per_cell=state_bits_per_cell,
        update_fraction=update_fraction,
        model_description_bits=model_description_bits,
        total_network_gates=total_network_gates,
        reachable_network_gates=reachable_network_gates,
        active_network_gates=active_network_gates,
    )
    rows.append(
        {
            "experiment": experiment,
            "task": task,
            "method": method,
            "seed": seed,
            **accounting,
            "residual_bits": int(residual_bits),
            "total_description_bits": int(model_description_bits + residual_bits),
            "note": note,
        }
    )


def run_local_rules(mode: str) -> tuple[list[dict], list[dict], dict[str, np.ndarray]]:
    local_rows: list[dict] = []
    complexity_rows: list[dict] = []
    trajectories: dict[str, np.ndarray] = {}
    formulae = exact_formula_library(3, max_gates=4)

    for rule in range(256):
        formula = formulae[rule]
        circuit = formula_to_circuit(formula, 3, deduplicate=True)
        prediction = circuit.evaluate(assignments(3))
        target = eca_lut(rule)
        local_rows.append(
            {
                "experiment": "E19",
                "task": "eca_catalog",
                "case": f"rule_{rule}",
                "method": "ExactFormulaCatalog",
                "seed": -1,
                "samples": 8,
                "local_accuracy": float(np.mean(prediction == target)),
                "local_balanced_accuracy": balanced_accuracy(target, prediction),
                "gate_count": len(circuit.nodes),
                "depth": circuit.depth,
                "model_bits": circuit_description_bits(circuit),
                "truth_table_bits": 8,
                "expression": formula.text,
                "cell_accuracy": 1.0,
                "exact_trajectory": 1.0,
                "first_divergence_step": -1,
            }
        )

    rng = np.random.default_rng(19_110)
    eca_width = 64 if mode == "smoke" else 257
    eca_steps = 32 if mode == "smoke" else 128
    for rule in (30, 90, 110, 150, 184, 204):
        formula = formulae[rule]
        circuit = formula_to_circuit(formula, 3, deduplicate=True)
        initial = rng.integers(0, 2, size=eca_width, dtype=np.uint8)
        oracle = np.empty((eca_steps + 1, eca_width), dtype=np.uint8)
        oracle[0] = initial
        for step in range(eca_steps):
            oracle[step + 1] = radius_lut_step(oracle[step], rule, radius=1, boundary="periodic")
        compiled = circuit_eca_rollout(initial, eca_steps, circuit)
        metrics = trajectory_metrics(oracle, compiled)
        local_rows.append(
            {
                "experiment": "E19",
                "task": "eca_rollout",
                "case": f"rule_{rule}",
                "method": "ExactFormulaCircuit",
                "seed": 19_110,
                "samples": eca_width * eca_steps,
                "local_accuracy": 1.0,
                "local_balanced_accuracy": 1.0,
                "gate_count": len(circuit.nodes),
                "depth": circuit.depth,
                "model_bits": circuit_description_bits(circuit),
                "truth_table_bits": 8,
                "expression": formula.text,
                **{key: metrics[key] for key in ("cell_accuracy", "exact_trajectory", "exact_final_state", "first_divergence_step", "exact_step_fraction")},
            }
        )
        if rule in (90, 110):
            trajectories[f"eca_rule_{rule}_oracle"] = oracle
        wrapper = ca_wrapper_bits(dimension=1, radius=1, channels=1, boundary="periodic", steps=eca_steps)
        residual = residual_code_bits(oracle[1:], compiled[1:])
        append_complexity(
            complexity_rows,
            experiment="E19",
            task=f"eca_rule_{rule}",
            method="ExactFormulaCircuit",
            spatial_shape=(eca_width,),
            steps=eca_steps,
            gates_per_cell=len(circuit.nodes),
            state_bits_per_cell=1,
            model_description_bits=3 + wrapper + circuit_description_bits(circuit),
            residual_bits=residual,
            seed=19_110,
            total_network_gates=len(circuit.nodes),
            reachable_network_gates=len(circuit.nodes),
            active_network_gates=len(circuit.nodes),
            note="Formula-tree optimum under AND/OR/XOR/NAND; not a hardware minimum.",
        )
        append_complexity(
            complexity_rows,
            experiment="E19",
            task=f"eca_rule_{rule}",
            method="Wolfram8BitLUT",
            spatial_shape=(eca_width,),
            steps=eca_steps,
            gates_per_cell=0,
            state_bits_per_cell=1,
            model_description_bits=3 + wrapper + 8,
            residual_bits=residual_code_bits(oracle[1:], oracle[1:]),
            seed=19_110,
            note="One LUT lookup per cell update is recorded outside binary-gate evaluations.",
        )

    # Rule 110 system identification on the complete local truth table.
    x3, y110 = assignments(3), eca_lut(110)
    gate110 = GateBeamSynthesizer(max_depth=4, beam_width=192).fit(x3, y110)
    pred110 = gate110.predict(x3)
    local_rows.append(
        {
            "experiment": "E19",
            "task": "eca_system_identification",
            "case": "rule_110",
            "method": "GateBeam",
            "seed": 0,
            "samples": 8,
            "local_accuracy": float(np.mean(pred110 == y110)),
            "local_balanced_accuracy": balanced_accuracy(y110, pred110),
            "gate_count": gate110.expression.gates,
            "depth": gate110.expression.depth,
            "model_bits": np.nan,
            "truth_table_bits": 8,
            "expression": gate110.expression.text,
            "fit_seconds": gate110.fit_seconds,
            "cell_accuracy": np.nan,
            "exact_trajectory": np.nan,
            "first_divergence_step": np.nan,
        }
    )

    # Complete 3x3 transition table, followed by long recurrent rollouts.
    x9 = assignments(9)
    life_target = game_of_life_next_from_patches(x9)
    life_models: list[tuple[str, int, Callable[[np.ndarray], np.ndarray], int, float, str]] = [
        ("LifeLikeB3S23", -1, game_of_life_next_from_patches, 2, 0.0, "count frontend plus two Boolean predicates"),
        ("FixedAND(center,north)", -1, fixed_and_life, 1, 0.0, "declared weak fixed-AND ablation"),
    ]
    gate_life = GateBeamSynthesizer(max_depth=4, beam_width=192).fit(x9, life_target)
    life_models.append(("GateBeam", 0, gate_life.predict, gate_life.expression.gates, gate_life.fit_seconds, gate_life.expression.text))
    mlp_seeds = range(1 if mode == "smoke" else 3)
    for seed in mlp_seeds:
        mlp = TinyMLP(9, hidden=(32, 16), seed=seed, lr=0.02, steps=900 if mode == "smoke" else 1800).fit(x9, life_target)
        life_models.append(("TinyMLP", seed, mlp.predict, 0, mlp.fit_seconds, "32x16 float MLP"))

    board_rng = np.random.default_rng(25_006)
    board_size = 24 if mode == "smoke" else 64
    life_steps = 12 if mode == "smoke" else 64
    initial_board = (board_rng.random((board_size, board_size)) < 0.22).astype(np.uint8)
    oracle_life = game_of_life_rollout(initial_board, life_steps, boundary="periodic")
    trajectories["game_of_life_oracle"] = oracle_life
    for method, seed, predictor, gate_count, fit_seconds, detail in life_models:
        local_prediction = np.asarray(predictor(x9), dtype=np.uint8)
        rollout = predictor_rollout(initial_board, life_steps, predictor, boundary="periodic")
        metrics = trajectory_metrics(oracle_life, rollout)
        local_rows.append(
            {
                "experiment": "E19",
                "task": "game_of_life",
                "case": "all_512_and_random_rollout",
                "method": method,
                "seed": seed,
                "samples": 512,
                "local_accuracy": float(np.mean(local_prediction == life_target)),
                "local_balanced_accuracy": balanced_accuracy(life_target, local_prediction),
                "gate_count": gate_count,
                "depth": np.nan,
                "model_bits": 18 if method == "LifeLikeB3S23" else np.nan,
                "truth_table_bits": 512,
                "expression": detail,
                "fit_seconds": fit_seconds,
                **{key: metrics[key] for key in ("cell_accuracy", "exact_trajectory", "exact_final_state", "first_divergence_step", "exact_step_fraction")},
            }
        )
        if method in ("LifeLikeB3S23", "GateBeam"):
            trajectories[f"game_of_life_{method.replace('(', '_').replace(')', '').replace(',', '').replace(' ', '_')}"] = rollout
        if method == "LifeLikeB3S23":
            wrapper = ca_wrapper_bits(dimension=2, radius=1, channels=1, boundary="periodic", steps=life_steps)
            residual = residual_code_bits(oracle_life[1:], rollout[1:])
            append_complexity(
                complexity_rows,
                experiment="E19",
                task="game_of_life",
                method=method,
                spatial_shape=(board_size, board_size),
                steps=life_steps,
                gates_per_cell=2,
                state_bits_per_cell=1,
                model_description_bits=3 + wrapper + 18,
                residual_bits=residual,
                note="B/S rule is 18 bits; neighbor counting/comparisons are separate arithmetic work.",
            )

    return local_rows, complexity_rows, trajectories


def collective_configs(mode: str) -> list[tuple[int, int, int]]:
    """Return (width, samples_per_seed, seeds), bounded below paper K=10,000."""
    if mode == "smoke":
        return [(149, 256, 1)]
    return [(149, 1_000, 3), (599, 256, 2), (999, 128, 2)]


def run_radius3_to_horizon(initial: np.ndarray, lut: np.ndarray, steps: int) -> np.ndarray:
    state = np.asarray(initial, dtype=np.uint8).copy()
    for _ in range(steps):
        state = radius_lut_step(state, lut, radius=3, boundary="periodic")
    return state


def run_collective_tasks(mode: str) -> tuple[list[dict], list[dict]]:
    task_rows: list[dict] = []
    complexity_rows: list[dict] = []
    density_rules = {name: lexicographic_hex_lut(value) for name, value in DENSITY_RULE_HEX.items()}
    density_rules["gkl"] = gkl_density_lut()
    sync_rules = {name: lexicographic_hex_lut(value) for name, value in SYNC_RULE_HEX.items()}
    naive = np.zeros(128, dtype=np.uint8)
    naive[0] = 1
    sync_rules["naive_oscillator"] = naive

    for width, samples, seeds in collective_configs(mode):
        horizon = 2 * width
        for seed in range(seeds):
            rng = np.random.default_rng(94_000 + width * 10 + seed)
            initial = rng.integers(0, 2, size=(samples, width), dtype=np.uint8)
            desired = (initial.sum(axis=1) > width // 2).astype(np.uint8)
            for name, lut in density_rules.items():
                started = time.perf_counter()
                final = run_radius3_to_horizon(initial, lut, horizon)
                elapsed = time.perf_counter() - started
                success = np.all(final == desired[:, None], axis=1)
                accuracy = float(np.mean(success))
                task_rows.append(
                    {
                        "experiment": "E20",
                        "task": "density_classification",
                        "case": "iid_bernoulli_half",
                        "method": name,
                        "seed": seed,
                        "width": width,
                        "samples": samples,
                        "steps": horizon,
                        "horizon_policy": "exact_2N",
                        "accuracy": accuracy,
                        "successes": int(success.sum()),
                        "unresolved_rate": float(np.mean(~np.all(final == final[:, :1], axis=1))),
                        "mean_final_density_error": float(np.mean(np.abs(final.mean(axis=1) - desired))),
                        "runtime_seconds": elapsed,
                        "published_accuracy": PUBLISHED_DENSITY.get(name, {}).get(width, np.nan),
                        "rule_lut_bits": 128,
                        "anf_description_bits": anf_description_bits(lut),
                        "output_ones": int(lut.sum()),
                        "local_semantics_source": "published_rule_replay" if name != "gkl" else "published_structural_rule",
                    }
                )
                wrapper = ca_wrapper_bits(dimension=1, radius=3, channels=1, boundary="periodic", steps=horizon)
                # Task labels, not full trajectories, are coded as the residual target.
                prediction = np.all(final == 1, axis=1).astype(np.uint8)
                residual = residual_code_bits(desired, prediction)
                append_complexity(
                    complexity_rows,
                    experiment="E20",
                    task="density_classification",
                    method=name,
                    spatial_shape=(samples, width),
                    steps=horizon,
                    gates_per_cell=0,
                    state_bits_per_cell=1,
                    model_description_bits=3 + wrapper + 128,
                    residual_bits=residual,
                    seed=seed,
                    note="128-bit radius-3 LUT paid once; dynamic column is LUT updates, not binary-gate PPA.",
                )

            for name, lut in sync_rules.items():
                started = time.perf_counter()
                current = run_radius3_to_horizon(initial, lut, horizon)
                following = radius_lut_step(current, lut, radius=3, boundary="periodic")
                elapsed = time.perf_counter() - started
                current_uniform = np.all(current == current[:, :1], axis=1)
                next_uniform = np.all(following == following[:, :1], axis=1)
                alternates = current[:, 0] != following[:, 0]
                success = current_uniform & next_uniform & alternates
                task_rows.append(
                    {
                        "experiment": "E20",
                        "task": "global_synchronization",
                        "case": "iid_bernoulli_half",
                        "method": name,
                        "seed": seed,
                        "width": width,
                        "samples": samples,
                        "steps": horizon,
                        "horizon_policy": "exact_2N_plus_one_validation_step",
                        "accuracy": float(np.mean(success)),
                        "successes": int(success.sum()),
                        "unresolved_rate": float(np.mean(~(current_uniform & next_uniform))),
                        "mean_final_density_error": np.nan,
                        "runtime_seconds": elapsed,
                        "published_accuracy": PUBLISHED_SYNC[name].get(width, np.nan),
                        "rule_lut_bits": 128,
                        "anf_description_bits": anf_description_bits(lut),
                        "output_ones": int(lut.sum()),
                        "local_semantics_source": "published_rule_replay",
                    }
                )
                wrapper = ca_wrapper_bits(dimension=1, radius=3, channels=1, boundary="periodic", steps=horizon)
                residual = residual_code_bits(np.ones(samples, dtype=np.uint8), success.astype(np.uint8))
                append_complexity(
                    complexity_rows,
                    experiment="E20",
                    task="global_synchronization",
                    method=name,
                    spatial_shape=(samples, width),
                    steps=horizon + 1,
                    gates_per_cell=0,
                    state_bits_per_cell=1,
                    model_description_bits=3 + wrapper + 128,
                    residual_bits=residual,
                    seed=seed,
                    note="Success requires two uniform complementary consecutive states; no partial credit.",
                )
    return task_rows, complexity_rows


def bfs_reference(passable: np.ndarray, source: tuple[int, int], target: tuple[int, int]) -> tuple[bool, int]:
    """Independent queue-based shortest-path oracle."""
    grid = np.asarray(passable, dtype=np.uint8)
    distance = np.full(grid.shape, -1, dtype=np.int32)
    distance[source] = 0
    queue: deque[tuple[int, int]] = deque([source])
    while queue:
        row, column = queue.popleft()
        if (row, column) == target:
            return True, int(distance[target])
        for row_delta, column_delta in ((-1, 0), (0, -1), (0, 1), (1, 0)):
            neighbor = (row + row_delta, column + column_delta)
            if (
                0 <= neighbor[0] < grid.shape[0]
                and 0 <= neighbor[1] < grid.shape[1]
                and grid[neighbor]
                and distance[neighbor] < 0
            ):
                distance[neighbor] = distance[row, column] + 1
                queue.append(neighbor)
    return False, -1


def snake_maze(size: int, blocked: bool) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    grid = np.zeros((size, size), dtype=np.uint8)
    corridor_rows = list(range(0, size, 2))
    for index, row in enumerate(corridor_rows):
        grid[row, :] = 1
        if row + 1 < size and index + 1 < len(corridor_rows):
            grid[row + 1, -1 if index % 2 == 0 else 0] = 1
    last_index = len(corridor_rows) - 1
    target = (corridor_rows[-1], size - 1 if last_index % 2 == 0 else 0)
    source = (0, 0)
    if blocked:
        middle_row = corridor_rows[len(corridor_rows) // 2]
        grid[middle_row, size // 2] = 0
    return grid, source, target


def random_maze(size: int, seed: int, reachable: bool) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    rng = np.random.default_rng(seed)
    grid = (rng.random((size, size)) > 0.30).astype(np.uint8)
    source, target = (0, 0), (size - 1, size - 1)
    if reachable:
        # Public tie order: traverse top row and then the rightmost column.
        grid[0, :] = 1
        grid[:, -1] = 1
    else:
        grid[max(0, size - 2), size - 1] = 0
        grid[size - 1, max(0, size - 2)] = 0
    grid[source] = grid[target] = 1
    return grid, source, target


def wavefront_circuit_ir() -> CircuitIR:
    """Return the decoded five-gate local reachability recurrence.

    Inputs are reached, passable, north, west, east, south. CircuitIR reserves
    references 6 and 7 for constants, so the first emitted gate starts at 8.
    """
    return CircuitIR(
        n_inputs=6,
        nodes=(
            CircuitNode("OR", 0, 2),
            CircuitNode("OR", 8, 3),
            CircuitNode("OR", 9, 4),
            CircuitNode("OR", 10, 5),
            CircuitNode("AND", 1, 11),
        ),
        output=12,
    )


def run_pathfinding(mode: str) -> tuple[list[dict], list[dict]]:
    from cellular_automata import boolean_wavefront_pathfind

    task_rows: list[dict] = []
    complexity_rows: list[dict] = []
    sizes = (16,) if mode == "smoke" else (16, 32, 64)
    cases_per_size = 8 if mode == "smoke" else 40
    fixed_horizon = 16
    # reached OR north OR west OR east OR south, then AND passable.
    wavefront_circuit = wavefront_circuit_ir()
    local_model_bits = circuit_description_bits(wavefront_circuit)
    for size in sizes:
        for case_index in range(cases_per_size):
            family_index = case_index % 4
            if family_index == 0:
                grid, source, target = snake_maze(size, blocked=False)
                family = "long_snake_reachable"
            elif family_index == 1:
                grid, source, target = snake_maze(size, blocked=True)
                family = "long_snake_blocked"
            elif family_index == 2:
                grid, source, target = random_maze(size, 21_000 + size * 100 + case_index, reachable=True)
                family = "random_carved_reachable"
            else:
                grid, source, target = random_maze(size, 21_000 + size * 100 + case_index, reachable=False)
                family = "random_isolated_target"
            started = time.perf_counter()
            truth_reached, truth_distance = bfs_reference(grid, source, target)
            bfs_seconds = time.perf_counter() - started
            methods: list[tuple[str, bool, int, int, float]] = [
                ("OracleBFS", truth_reached, truth_distance, max(0, truth_distance), bfs_seconds)
            ]
            for method, max_steps in (("BooleanWavefrontFull", grid.size - 1), ("FixedK16Wavefront", fixed_horizon)):
                started = time.perf_counter()
                result: BooleanWavefrontResult = boolean_wavefront_pathfind(grid, source, target, max_steps=max_steps)
                elapsed = time.perf_counter() - started
                methods.append((method, result.target_reached, result.target_distance, result.iterations, elapsed))
            for method, predicted, predicted_distance, iterations, elapsed in methods:
                task_rows.append(
                    {
                        "experiment": "E21",
                        "task": "boolean_pathfinding",
                        "case": family,
                        "method": method,
                        "seed": case_index,
                        "width": size,
                        "samples": 1,
                        "steps": iterations,
                        "horizon_policy": "until_target_or_fixed_grid_bound" if method != "FixedK16Wavefront" else "fixed_16",
                        "accuracy": float(predicted == truth_reached),
                        "truth_reachable": float(truth_reached),
                        "predicted_reachable": float(predicted),
                        "truth_distance": truth_distance,
                        "predicted_distance": predicted_distance,
                        "distance_exact": float((not truth_reached and not predicted) or predicted_distance == truth_distance),
                        "unresolved_rate": float(truth_reached and not predicted),
                        "runtime_seconds": elapsed,
                        "published_accuracy": np.nan,
                        "rule_lut_bits": np.nan,
                        "anf_description_bits": np.nan,
                        "output_ones": np.nan,
                        "local_semantics_source": "hand_coded_boolean_recurrence" if method != "OracleBFS" else "independent_queue_oracle",
                    }
                )
                if method != "OracleBFS":
                    wrapper = ca_wrapper_bits(dimension=2, radius=1, channels=2, boundary="fixed", steps=iterations)
                    append_complexity(
                        complexity_rows,
                        experiment="E21",
                        task=family,
                        method=method,
                        spatial_shape=grid.shape,
                        steps=iterations,
                        gates_per_cell=5,
                        state_bits_per_cell=2,
                        model_description_bits=3 + wrapper + local_model_bits,
                        residual_bits=residual_code_bits(np.asarray([truth_reached], dtype=np.uint8), np.asarray([predicted], dtype=np.uint8)),
                        seed=case_index,
                        total_network_gates=5,
                        reachable_network_gates=5,
                        active_network_gates=5,
                        note="Exact symbolic grid input; iteration bound is charged separately from the five-gate local rule.",
                    )
    return task_rows, complexity_rows


def checkerboard_target(size: int, channels: int = 8, shape_size: int = 2) -> np.ndarray:
    row, column = np.indices((size, size))
    target = np.zeros((size, size, channels), dtype=np.uint8)
    target[..., 0] = ((row // shape_size + column // shape_size) % 2).astype(np.uint8)
    return target


def declared_digitaljs_netlist_bits(circuit: object) -> int:
    """A conservative fixed-width code for an ordered multi-output DAG.

    The code pays for sparse absolute input orders, every operator, two parent
    slots per logic node (dummy slots for unary/constants), and output roots.
    It is an auditable upper code, not minimum circuit complexity.
    """
    stats = circuit.stats()
    logic_nodes = int(stats["logic_node_count"])
    input_width = int(circuit.required_input_width)
    input_count = int(stats["input_count"])
    output_count = int(stats["output_count"])
    wire_space = max(1, input_width + logic_nodes + 2)
    wire_bits = int(math.ceil(math.log2(wire_space))) if wire_space > 1 else 0
    input_order_bits = int(math.ceil(math.log2(max(1, input_width)))) if input_width > 1 else 0
    return (
        elias_delta_bits(logic_nodes + 1)
        + elias_delta_bits(input_width + 1)
        + elias_delta_bits(input_count + 1)
        + elias_delta_bits(output_count + 1)
        + input_count * input_order_bits
        + logic_nodes * (4 + 2 * wire_bits)
        + output_count * wire_bits
    )


def asynchronous_digitaljs_rollout(
    circuit: object,
    initial: np.ndarray,
    steps: int,
    fire_rate: float,
    seed: int,
    boundary: str = "fixed",
    input_layout: str = "spatial_major",
) -> np.ndarray:
    from difflogic_ca import synchronous_ca_step

    rng = np.random.default_rng(seed)
    state = np.asarray(initial, dtype=np.uint8).copy()
    trajectory = np.empty((steps + 1, *state.shape), dtype=np.uint8)
    trajectory[0] = state
    for step in range(steps):
        proposal = synchronous_ca_step(circuit, state, boundary=boundary, input_layout=input_layout)
        update = rng.random(state.shape[:2]) <= fire_rate
        state = np.where(update[..., None], proposal, state).astype(np.uint8)
        trajectory[step + 1] = state
    return trajectory


def run_official_replay(
    mode: str,
    cache_dir: Path,
) -> tuple[list[dict], list[dict], dict[str, np.ndarray], dict]:
    from difflogic_ca import (
        GOOGLE_DIFFLOGIC_CA_ARTIFACTS,
        download_google_difflogic_ca_json,
        load_digitaljs_json,
        synchronous_ca_step,
        synchronous_ca_rollout,
    )

    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    circuits = {}
    for filename in ("gol_circuit.json", "checkerboard.json"):
        path = download_google_difflogic_ca_json(cache_dir / filename, artifact=filename, timeout=120.0)
        paths[filename] = path
        circuits[filename] = load_digitaljs_json(path)

    official_rows: list[dict] = []
    complexity_rows: list[dict] = []
    trajectories: dict[str, np.ndarray] = {}
    common = {
        "experiment": "E18",
        "reproduction_level": "released_hard_circuit_semantic_reproduction",
        "training_reproduction": "not_claimed",
        "pages_commit": OFFICIAL_PAGES_COMMIT,
    }

    gol = circuits["gol_circuit.json"]
    gol_stats = gol.stats()
    x9 = assignments(9)
    gol_target = game_of_life_next_from_patches(x9)
    started = time.perf_counter()
    gol_prediction = gol.evaluate(x9)[:, 0]
    truth_seconds = time.perf_counter() - started
    official_rows.append(
        {
            **common,
            "task": "game_of_life",
            "condition": "all_512_local_transitions",
            "artifact": "gol_circuit.json",
            "input_layout": "spatial_major_f*C+c (C=1)",
            "method": "OfficialFrozenDigitalJS",
            "seed": -1,
            "size": 3,
            "steps": 1,
            "samples": 512,
            "local_accuracy": float(np.mean(gol_prediction == gol_target)),
            "local_balanced_accuracy": balanced_accuracy(gol_target, gol_prediction),
            "cell_accuracy": float(np.mean(gol_prediction == gol_target)),
            "channel0_accuracy": float(np.mean(gol_prediction == gol_target)),
            "full_state_accuracy": float(np.mean(gol_prediction == gol_target)),
            "exact_final_state": float(np.array_equal(gol_prediction, gol_target)),
            "exact_full_state": float(np.array_equal(gol_prediction, gol_target)),
            "first_divergence_step": -1 if np.array_equal(gol_prediction, gol_target) else 0,
            "runtime_seconds": truth_seconds,
            "logic_node_count": gol_stats["logic_node_count"],
            "paper_reported_active_gates_excluding_passthrough": 336,
            "equivalent_functional_gates": np.nan,
            "critical_depth": gol_stats["critical_depth"],
            "raw_connector_count": gol_stats["raw_connector_count"],
            "unique_connector_count": gol_stats["unique_connector_count"],
            "artifact_sha256": GOOGLE_DIFFLOGIC_CA_ARTIFACTS["gol_circuit.json"].sha256,
            "note": "Released output-reachable hard circuit; the paper's 336 active-gate count uses a different artifact/counting scope.",
        }
    )

    gol_size = 32 if mode == "smoke" else 128
    gol_steps = 30 if mode == "smoke" else 300
    rng = np.random.default_rng(25_006)
    gol_initial = rng.integers(0, 2, size=(gol_size, gol_size, 1), dtype=np.uint8)
    oracle = game_of_life_rollout(gol_initial[..., 0], gol_steps, boundary="periodic")[..., None]
    started = time.perf_counter()
    replay = synchronous_ca_rollout(gol, gol_initial, gol_steps, boundary="periodic")
    elapsed = time.perf_counter() - started
    metrics = trajectory_metrics(oracle, replay)
    official_rows.append(
        {
            **common,
            "task": "game_of_life",
            "condition": "random_periodic_rollout",
            "artifact": "gol_circuit.json",
            "input_layout": "spatial_major_f*C+c (C=1)",
            "method": "OfficialFrozenDigitalJS",
            "seed": 25_006,
            "size": gol_size,
            "steps": gol_steps,
            "samples": gol_size * gol_size * gol_steps,
            "local_accuracy": 1.0,
            "local_balanced_accuracy": 1.0,
            "cell_accuracy": metrics["cell_accuracy"],
            "channel0_accuracy": metrics["final_cell_accuracy"],
            "full_state_accuracy": metrics["final_cell_accuracy"],
            "exact_final_state": metrics["exact_final_state"],
            "exact_full_state": metrics["exact_final_state"],
            "first_divergence_step": metrics["first_divergence_step"],
            "runtime_seconds": elapsed,
            "logic_node_count": gol_stats["logic_node_count"],
            "paper_reported_active_gates_excluding_passthrough": 336,
            "equivalent_functional_gates": np.nan,
            "critical_depth": gol_stats["critical_depth"],
            "raw_connector_count": gol_stats["raw_connector_count"],
            "unique_connector_count": gol_stats["unique_connector_count"],
            "artifact_sha256": GOOGLE_DIFFLOGIC_CA_ARTIFACTS["gol_circuit.json"].sha256,
            "note": "Matches the paper's 128x128/300-step deployment scale in full mode.",
        }
    )
    trajectories["official_gol_oracle"] = oracle
    trajectories["official_gol_replay"] = replay
    gol_netlist_bits = 3 + declared_digitaljs_netlist_bits(gol)
    gol_model_bits = gol_netlist_bits + ca_wrapper_bits(
        dimension=2,
        radius=1,
        channels=1,
        boundary="periodic",
        steps=gol_steps,
    )
    append_complexity(
        complexity_rows,
        experiment="E18",
        task="official_game_of_life",
        method="OfficialFrozenDigitalJS",
        spatial_shape=(gol_size, gol_size),
        steps=gol_steps,
        gates_per_cell=gol.logic_node_count,
        state_bits_per_cell=1,
        model_description_bits=gol_model_bits,
        residual_bits=residual_code_bits(oracle[1:], replay[1:]),
        total_network_gates=gol.logic_node_count,
        reachable_network_gates=gol.logic_node_count,
        active_network_gates=gol.logic_node_count,
        note="Generic fixed-width DigitalJS DAG upper code; the paper's 336 count is a separate artifact/counting scope.",
    )

    checker = circuits["checkerboard.json"]
    checker_stats = checker.stats()
    clean_configs = [(16, 20)] if mode == "smoke" else [(16, 20), (64, 80)]
    clean_seeds = range(3 if mode == "smoke" else 5)
    checker_netlist_bits = 3 + declared_digitaljs_netlist_bits(checker)
    for size, steps in clean_configs:
        target = checkerboard_target(size)
        for seed in clean_seeds:
            rng = np.random.default_rng(23_000 + size * 100 + seed)
            initial = rng.integers(0, 2, size=(size, size, 8), dtype=np.uint8)
            started = time.perf_counter()
            rollout = synchronous_ca_rollout(
                checker,
                initial,
                steps,
                boundary="fixed",
                input_layout="channel_major",
            )
            elapsed = time.perf_counter() - started
            final = rollout[-1]
            channel_accuracy = float(np.mean(final[..., 0] == target[..., 0]))
            full_accuracy = float(np.mean(final == target))
            official_rows.append(
                {
                    **common,
                    "task": "checkerboard",
                    "condition": "clean_sync",
                    "artifact": "checkerboard.json",
                    "input_layout": "channel_major_c*9+f",
                    "method": "OfficialFrozenDigitalJS",
                    "seed": seed,
                    "size": size,
                    "steps": steps,
                    "samples": size * size,
                    "local_accuracy": np.nan,
                    "local_balanced_accuracy": np.nan,
                    "cell_accuracy": channel_accuracy,
                    "channel0_accuracy": channel_accuracy,
                    "full_state_accuracy": full_accuracy,
                    "exact_final_state": float(np.array_equal(final[..., 0], target[..., 0])),
                    "exact_full_state": float(np.array_equal(final, target)),
                    "first_divergence_step": np.nan,
                    "runtime_seconds": elapsed,
                    "logic_node_count": checker_stats["logic_node_count"],
                    "paper_reported_active_gates_excluding_passthrough": 22,
                    "equivalent_functional_gates": 5,
                    "critical_depth": checker_stats["critical_depth"],
                    "raw_connector_count": checker_stats["raw_connector_count"],
                    "unique_connector_count": checker_stats["unique_connector_count"],
                    "artifact_sha256": GOOGLE_DIFFLOGIC_CA_ARTIFACTS["checkerboard.json"].sha256,
                    "note": "Notebook loss supervises channel 0 only; hidden-channel accuracy is diagnostic. Six released nodes include one removable same-input AND.",
                }
            )
            append_complexity(
                complexity_rows,
                experiment="E22",
                task=f"checkerboard_{size}",
                method="OfficialFrozenDigitalJS",
                spatial_shape=(size, size),
                steps=steps,
                gates_per_cell=checker.logic_node_count,
                state_bits_per_cell=8,
                model_description_bits=checker_netlist_bits
                + ca_wrapper_bits(
                    dimension=2,
                    radius=1,
                    channels=8,
                    boundary="fixed",
                    steps=steps,
                ),
                residual_bits=residual_code_bits(target[..., 0], final[..., 0]),
                seed=seed,
                total_network_gates=checker.logic_node_count,
                reachable_network_gates=checker.logic_node_count,
                active_network_gates=5,
                note="Interpreter evaluates six released nodes; five remain after algebraic same-input simplification.",
            )
            if seed == 0:
                trajectories[f"official_checkerboard_{size}"] = rollout

    async_seeds = range(2 if mode == "smoke" else 4)
    target16 = checkerboard_target(16)
    for seed in async_seeds:
        rng = np.random.default_rng(60_000 + seed)
        initial = rng.integers(0, 2, size=(16, 16, 8), dtype=np.uint8)
        started = time.perf_counter()
        rollout = asynchronous_digitaljs_rollout(
            checker,
            initial,
            100,
            0.6,
            60_000 + seed,
            input_layout="channel_major",
        )
        elapsed = time.perf_counter() - started
        final = rollout[-1]
        channel_accuracy = float(np.mean(final[..., 0] == target16[..., 0]))
        official_rows.append(
            {
                **common,
                "task": "checkerboard",
                "condition": "async_fire_0.6",
                "artifact": "checkerboard.json",
                "input_layout": "channel_major_c*9+f",
                "method": "OfficialFrozenDigitalJS",
                "seed": seed,
                "size": 16,
                "steps": 100,
                "samples": 16 * 16,
                "local_accuracy": np.nan,
                "local_balanced_accuracy": np.nan,
                "cell_accuracy": channel_accuracy,
                "channel0_accuracy": channel_accuracy,
                "full_state_accuracy": float(np.mean(final == target16)),
                "exact_final_state": float(np.array_equal(final[..., 0], target16[..., 0])),
                "exact_full_state": float(np.array_equal(final, target16)),
                "first_divergence_step": np.nan,
                "runtime_seconds": elapsed,
                "logic_node_count": checker.logic_node_count,
                "paper_reported_active_gates_excluding_passthrough": 22,
                "equivalent_functional_gates": 5,
                "critical_depth": checker.critical_depth,
                "raw_connector_count": checker.raw_connector_count,
                "unique_connector_count": checker.unique_connector_count,
                "artifact_sha256": GOOGLE_DIFFLOGIC_CA_ARTIFACTS["checkerboard.json"].sha256,
                "note": "NumPy default_rng (PCG64), seed, and fire rate are public; this is not the notebook's JAX PRNG stream and the schedule is not learned topology.",
            }
        )

    damage_size = 16 if mode == "smoke" else 64
    damage_seeds = range(1 if mode == "smoke" else 3)
    target_damage = checkerboard_target(damage_size)
    for seed in damage_seeds:
        rng = np.random.default_rng(80_000 + seed)
        initial = rng.integers(0, 2, size=(damage_size, damage_size, 8), dtype=np.uint8)
        half = 2 if mode == "smoke" else 10
        center = damage_size // 2
        started = time.perf_counter()
        state = initial.copy()
        recovered_rollout = np.empty((81, *state.shape), dtype=np.uint8)
        recovered_rollout[0] = state
        for step in range(80):
            state = synchronous_ca_step(
                checker,
                state,
                boundary="fixed",
                input_layout="channel_major",
            )
            # The official notebook clamps the actual 20x20 central slice after
            # each of the first 40 updates, then releases it for 40 updates.
            # Smoke mode retains the timing but scales the lesion to 4x4.
            if step < 40:
                state[center - half : center + half, center - half : center + half, :] = 0
            recovered_rollout[step + 1] = state
        elapsed = time.perf_counter() - started
        damage_accuracy = float(np.mean(recovered_rollout[40, ..., 0] == target_damage[..., 0]))
        final = recovered_rollout[-1]
        final_accuracy = float(np.mean(final[..., 0] == target_damage[..., 0]))
        official_rows.append(
            {
                **common,
                "task": "checkerboard",
                "condition": "persistent_damage_40_then_release_40",
                "artifact": "checkerboard.json",
                "input_layout": "channel_major_c*9+f",
                "method": "OfficialFrozenDigitalJS",
                "seed": seed,
                "size": damage_size,
                "steps": 80,
                "damage_steps": 40,
                "release_steps": 40,
                "damage_side": 2 * half,
                "samples": damage_size * damage_size,
                "local_accuracy": np.nan,
                "local_balanced_accuracy": np.nan,
                "cell_accuracy": final_accuracy,
                "channel0_accuracy": final_accuracy,
                "full_state_accuracy": float(np.mean(final == target_damage)),
                "exact_final_state": float(np.array_equal(final[..., 0], target_damage[..., 0])),
                "exact_full_state": float(np.array_equal(final, target_damage)),
                "first_divergence_step": np.nan,
                "damage_channel0_accuracy": damage_accuracy,
                "recovery_gain": final_accuracy - damage_accuracy,
                "runtime_seconds": elapsed,
                "logic_node_count": checker.logic_node_count,
                "paper_reported_active_gates_excluding_passthrough": 22,
                "equivalent_functional_gates": 5,
                "critical_depth": checker.critical_depth,
                "raw_connector_count": checker.raw_connector_count,
                "unique_connector_count": checker.unique_connector_count,
                "artifact_sha256": GOOGLE_DIFFLOGIC_CA_ARTIFACTS["checkerboard.json"].sha256,
                "note": "Full mode exactly follows the notebook timing: clamp its actual 20x20 center slice after each of the first 40 synchronous updates, then release for 40 updates; smoke scales only the lesion size.",
            }
        )
        if seed == 0:
            trajectories["official_checkerboard_damage_recovery"] = recovered_rollout

    direct_bits = 3 + 2 + elias_delta_bits(2) + 1
    append_complexity(
        complexity_rows,
        experiment="E22",
        task="checkerboard_direct_description",
        method="DirectCheckerboardProgram",
        spatial_shape=(damage_size, damage_size),
        steps=1,
        gates_per_cell=0,
        state_bits_per_cell=1,
        model_description_bits=direct_bits,
        residual_bits=residual_code_bits(target_damage[..., 0], target_damage[..., 0]),
        note="Competing target-language code pays family, shape-size, and phase; dimensions are public side information.",
    )
    provenance = {
        "pages_commit": OFFICIAL_PAGES_COMMIT,
        "notebook_origin_reproduction_commit": OFFICIAL_NOTEBOOK_ORIGIN_REPRODUCTION_COMMIT,
        "artifacts": {
            filename: {
                "path": str(path),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
                "circuit_stats": circuits[filename].stats(),
            }
            for filename, path in paths.items()
        },
        "training_reproduction_status": "not_part_of_frozen_replay; no upstream checkpoint or embedded notebook output",
    }
    return official_rows, complexity_rows, trajectories, provenance


def write_results(
    local_rows: list[dict],
    task_rows: list[dict],
    complexity_rows: list[dict],
    official_rows: list[dict],
    trajectories: dict[str, np.ndarray],
    metadata: dict,
) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    outputs = {
        "cellular_automata_local_rule_results.csv": pd.DataFrame(local_rows),
        "cellular_automata_task_results.csv": pd.DataFrame(task_rows),
        "cellular_automata_complexity_results.csv": pd.DataFrame(complexity_rows),
        "cellular_automata_official_results.csv": pd.DataFrame(official_rows),
    }
    for filename, frame in outputs.items():
        frame.to_csv(RESULTS / filename, index=False)
    np.savez_compressed(RESULTS / "cellular_automata_trajectories.npz", **trajectories)
    artifact_paths = [RESULTS / filename for filename in outputs]
    artifact_paths.append(RESULTS / "cellular_automata_trajectories.npz")
    training_attempt = RESULTS / "difflogic_ca_training_attempt.json"
    if training_attempt.exists():
        artifact_paths.append(training_attempt)
    metadata["artifacts"] = {
        path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
        for path in artifact_paths
    }
    metadata_path = RESULTS / "cellular_automata_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument(
        "--suite",
        choices=("all", "local", "collective", "pathfinding", "official"),
        default="all",
    )
    parser.add_argument(
        "--official-cache-dir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "symbolic_logic_difflogic_ca",
    )
    args = parser.parse_args()
    started = time.perf_counter()
    local_rows: list[dict] = []
    task_rows: list[dict] = []
    complexity_rows: list[dict] = []
    official_rows: list[dict] = []
    trajectories: dict[str, np.ndarray] = {}
    official_provenance: dict = {"training_reproduction_status": "not_requested_in_selected_suite"}

    if args.suite in ("all", "local"):
        rows, complexity, arrays = run_local_rules(args.mode)
        local_rows.extend(rows)
        complexity_rows.extend(complexity)
        trajectories.update(arrays)
    if args.suite in ("all", "collective"):
        rows, complexity = run_collective_tasks(args.mode)
        task_rows.extend(rows)
        complexity_rows.extend(complexity)
    if args.suite in ("all", "pathfinding"):
        rows, complexity = run_pathfinding(args.mode)
        task_rows.extend(rows)
        complexity_rows.extend(complexity)
    if args.suite in ("all", "official"):
        rows, complexity, arrays, official_provenance = run_official_replay(args.mode, args.official_cache_dir)
        official_rows.extend(rows)
        complexity_rows.extend(complexity)
        trajectories.update(arrays)

    metadata = {
        "schema_version": 1,
        "mode": args.mode,
        "suite": args.suite,
        "started_from_git": git_state(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "elapsed_seconds": time.perf_counter() - started,
        "seeds": {
            "eca_and_life": [19_110, 25_006],
            "collective_formula": "94000 + width*10 + seed",
            "pathfinding_formula": "21000 + size*100 + case",
            "official_checkerboard_formulas": ["23000 + size*100 + seed", "60000 + seed", "80000 + seed"],
        },
        "row_counts": {
            "local": len(local_rows),
            "task": len(task_rows),
            "complexity": len(complexity_rows),
            "official": len(official_rows),
        },
        "official": official_provenance,
        "claim_boundary": "Frozen hard-circuit replay is not optimization or multi-seed training reproduction; CPU and gate counts are not hardware PPA.",
    }
    source_paths = [
        ROOT / "src" / "cellular_automata.py",
        ROOT / "src" / "difflogic_ca.py",
        ROOT / "src" / "run_cellular_automata.py",
        ROOT / "src" / "run_official_difflogic_gol.py",
        ROOT / "src" / "boolean_mdl.py",
        ROOT / "src" / "logic_core.py",
        ROOT / "third_party" / "difflogic_ca_manifest.json",
    ]
    metadata["sources"] = {
        str(path.relative_to(ROOT)).replace("\\", "/"): {"sha256": sha256(path), "bytes": path.stat().st_size}
        for path in source_paths
    }
    training_attempt_path = RESULTS / "difflogic_ca_training_attempt.json"
    if training_attempt_path.exists():
        metadata["official"]["training_attempt"] = json.loads(training_attempt_path.read_text(encoding="utf-8"))
    write_results(local_rows, task_rows, complexity_rows, official_rows, trajectories, metadata)
    print(json.dumps({"mode": args.mode, "suite": args.suite, "rows": metadata["row_counts"], "elapsed_seconds": metadata["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
