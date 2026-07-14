"""Run the executable Boolean/Circuit-MDL theory checks and benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from boolean_mdl import (
    anf_statistics,
    best_robdd,
    discrete_rate_reduction,
    exact_formula_library,
    joint_dirichlet_code_bits,
    kt_independent_matrix_bits,
    mask_to_values,
    occam_error_bound,
    route_task_mdl,
    routed_function_description,
    task_mdl_candidates,
    threshold_library,
    truth_table_bits,
    values_to_mask,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


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


def binary_entropy(values: np.ndarray) -> float:
    probability = float(np.mean(values))
    if probability in (0.0, 1.0):
        return 0.0
    return float(-probability * np.log2(probability) - (1.0 - probability) * np.log2(1.0 - probability))


def balanced_random(n_inputs: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = np.zeros(1 << n_inputs, dtype=np.uint8)
    values[: len(values) // 2] = 1
    rng.shuffle(values)
    return values


def benchmark_cases(mode: str) -> list[dict]:
    cases: list[dict] = []
    for n_inputs in (3, 4, 6):
        x = assignments(n_inputs)
        if n_inputs == 3:
            cases.extend(
                [
                    {"case": "rule110", "family": "cellular_rule", "n_inputs": 3, "seed": -1, "values": np.asarray([0, 1, 1, 1, 0, 1, 1, 0], dtype=np.uint8)},
                    {"case": "majority3", "family": "threshold", "n_inputs": 3, "seed": -1, "values": (x.sum(axis=1) >= 2).astype(np.uint8)},
                ]
            )
        if n_inputs == 4:
            shared = x[:, 0] ^ x[:, 1]
            cases.extend(
                [
                    {"case": "parity4", "family": "parity", "n_inputs": 4, "seed": -1, "values": np.bitwise_xor.reduce(x, axis=1)},
                    {"case": "sparse_dnf4", "family": "sparse_dnf", "n_inputs": 4, "seed": -1, "values": (x[:, 0] & x[:, 1]) | (x[:, 2] & x[:, 3])},
                    {"case": "mux4", "family": "decision", "n_inputs": 4, "seed": -1, "values": (x[:, 0] & x[:, 1]) | ((1 - x[:, 0]) & x[:, 2])},
                    {"case": "reused_subexpr4", "family": "reuse", "n_inputs": 4, "seed": -1, "values": (shared & x[:, 2]) | (shared & x[:, 3])},
                ]
            )
            for seed in range(2 if mode == "smoke" else 10):
                cases.append({"case": f"random_balanced4_s{seed}", "family": "random_balanced", "n_inputs": 4, "seed": seed, "values": balanced_random(4, 44_000 + seed)})
        if n_inputs == 6:
            shared = x[:, 0] ^ x[:, 1]
            cases.extend(
                [
                    {"case": "and2_of_6", "family": "sparse_conjunction", "n_inputs": 6, "seed": -1, "values": x[:, 0] & x[:, 1]},
                    {"case": "parity6", "family": "parity", "n_inputs": 6, "seed": -1, "values": np.bitwise_xor.reduce(x, axis=1)},
                    {"case": "majority6", "family": "threshold", "n_inputs": 6, "seed": -1, "values": (x.sum(axis=1) >= 3).astype(np.uint8)},
                    {"case": "sparse_dnf6", "family": "sparse_dnf", "n_inputs": 6, "seed": -1, "values": (x[:, 0] & x[:, 1]) | (x[:, 2] & x[:, 3])},
                    {"case": "permuted_local6", "family": "variable_permutation", "n_inputs": 6, "seed": -1, "values": (x[:, 5] & x[:, 2]) | (x[:, 4] & x[:, 1])},
                    {"case": "reused_subexpr6", "family": "reuse", "n_inputs": 6, "seed": -1, "values": (shared & x[:, 2]) | (shared & x[:, 3]) | (shared & x[:, 4])},
                ]
            )
            for seed in range(2 if mode == "smoke" else 10):
                cases.append({"case": f"random_balanced6_s{seed}", "family": "random_balanced", "n_inputs": 6, "seed": seed, "values": balanced_random(6, 66_000 + seed)})
    return cases


def run_function_and_task_mdl(mode: str) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict[int, object], dict[int, dict]]:
    formulae: dict[int, object] = {
        3: exact_formula_library(3, max_gates=4),
        4: exact_formula_library(4, max_gates=4 if mode == "smoke" else 8),
    }
    thresholds = {
        3: threshold_library(3, max_abs_weight=2),
        4: threshold_library(4, max_abs_weight=2),
        6: threshold_library(6, max_abs_weight=1),
    }
    function_rows: list[dict] = []
    task_rows: list[dict] = []
    for case in benchmark_cases(mode):
        values = np.asarray(case["values"], dtype=np.uint8)
        n_inputs = int(case["n_inputs"])
        target = values_to_mask(values)
        local_formulae = formulae.get(n_inputs)
        local_thresholds = thresholds.get(n_inputs)
        language, best_bits, lengths = routed_function_description(values, n_inputs, formula_library=local_formulae, thresholds=local_thresholds)
        anf_terms, anf_degree, anf_gate_upper = anf_statistics(values)
        bdd, _ = best_robdd(values)
        formula = local_formulae.get(target) if local_formulae is not None else None
        shared = {
            "case": case["case"],
            "family": case["family"],
            "n_inputs": n_inputs,
            "seed": case["seed"],
            "samples": len(values),
            "output_ones": int(values.sum()),
            "output_entropy_bits": binary_entropy(values),
            "truth_table_bits": truth_table_bits(n_inputs),
            "best_function_language": language,
            "best_function_bits": best_bits,
            "anf_terms": anf_terms,
            "anf_degree": anf_degree,
            "anf_gate_upper": anf_gate_upper,
            "best_robdd_nodes": len(bdd.nodes),
            "best_robdd_order": "-".join(str(item) for item in bdd.order),
            "threshold_exact": float(local_thresholds is not None and target in local_thresholds),
            "minimum_formula_gates": formula.gates if formula is not None else np.nan,
            "minimum_formula_exact": float(formula is not None),
        }
        for candidate_language, description_bits in lengths.items():
            function_rows.append(
                {
                    **shared,
                    "language": candidate_language,
                    "description_bits": description_bits,
                    "saving_vs_truth_table_bits": truth_table_bits(n_inputs) - description_bits,
                    "selected": float(candidate_language == language),
                }
            )
        candidates = task_mdl_candidates(values, n_inputs, formula_library=local_formulae, thresholds=local_thresholds)
        best_task, baseline_bits, safe_gain = route_task_mdl(candidates)
        for candidate in candidates:
            empirical_error = candidate.errors / len(values)
            model_code = ceil_log2_public_languages() + candidate.model_bits
            task_rows.append(
                {
                    **shared,
                    "language": candidate.language,
                    "model_bits": candidate.model_bits,
                    "residual_bits": candidate.residual_bits,
                    "total_bits": candidate.total_bits,
                    "errors": candidate.errors,
                    "empirical_error": empirical_error,
                    "exact": float(candidate.exact),
                    "detail": candidate.detail,
                    "selected": float(candidate == best_task),
                    "raw_label_baseline_bits": baseline_bits,
                    "safe_compression_gain_bits": safe_gain,
                    "occam_bound_delta05": occam_error_bound(empirical_error, model_code, len(values)) if candidate.language not in ("RawLabels", "KTLabels") else np.nan,
                }
            )
    checks = {
        "formula3_coverage": len(formulae[3]),
        "formula4_coverage": len(formulae[4]),
        "formula4_complete": len(formulae[4]) == 65536,
        "formula4_maximum_min_gates": max(item.gates for item in formulae[4].values()),
    }
    return pd.DataFrame(function_rows), pd.DataFrame(task_rows), checks, formulae, thresholds


def ceil_log2_public_languages() -> int:
    # Must match task_mdl_candidates' ten public routes.
    return 4


def run_balanced_formula_distribution(formula4: dict[int, object]) -> pd.DataFrame:
    rows = []
    parity_mask = values_to_mask(np.bitwise_xor.reduce(assignments(4), axis=1))
    parity_gates = formula4[parity_mask].gates
    for mask, formula in formula4.items():
        weight = int(mask).bit_count() if hasattr(int(mask), "bit_count") else bin(int(mask)).count("1")
        if weight == 8:
            rows.append(
                {
                    "truth_mask": mask,
                    "minimum_formula_gates": formula.gates,
                    "parity4_minimum_formula_gates": parity_gates,
                    "no_more_complex_than_parity": float(formula.gates <= parity_gates),
                }
            )
    return pd.DataFrame(rows)


def run_kraft_check(formula3: dict[int, object], threshold3: dict) -> dict:
    lengths = []
    selected_languages: dict[str, int] = {}
    for mask in range(256):
        values = mask_to_values(mask, 8)
        language, bits, _ = routed_function_description(values, 3, formula_library=formula3, thresholds=threshold3)
        lengths.append(bits)
        selected_languages[language] = selected_languages.get(language, 0) + 1
    return {
        "functions": 256,
        "kraft_sum": float(sum(2.0 ** (-bits) for bits in lengths)),
        "kraft_valid": bool(sum(2.0 ** (-bits) for bits in lengths) <= 1.0 + 1e-12),
        "minimum_bits": min(lengths),
        "maximum_bits": max(lengths),
        "selected_language_counts": selected_languages,
    }


def run_representation_codes() -> pd.DataFrame:
    rng = np.random.default_rng(20260715)
    samples = 256
    balanced = np.tile(np.asarray([0, 1], dtype=np.uint8), samples // 2)
    noise = rng.integers(0, 2, size=samples, dtype=np.uint8)
    cases = {
        "redundant_basis": (np.stack((balanced, balanced), axis=1), balanced),
        "invertible_xor_basis": (np.stack((balanced, balanced ^ balanced), axis=1), balanced),
        "label_plus_noise": (np.stack((balanced, noise), axis=1), balanced),
        "constant_irrelevant_labels": (np.zeros((samples, 2), dtype=np.uint8), balanced),
    }
    rows = []
    for name, (codes, labels) in cases.items():
        for family, joint in (("IndependentKT", False), ("JointDirichletKT", True)):
            side = discrete_rate_reduction(codes, labels, joint=joint, labels_are_side_information=True)
            no_side = discrete_rate_reduction(codes, labels, joint=joint, labels_are_side_information=False)
            rows.append(
                {
                    "case": name,
                    "code_family": family,
                    "samples": samples,
                    "dimensions": codes.shape[1],
                    "global_bits": side.global_bits,
                    "conditional_bits_side_info": side.conditional_bits,
                    "signed_reduction_side_info_bits": side.raw_reduction_bits,
                    "routed_reduction_side_info_bits": side.routed_reduction_bits,
                    "conditional_bits_labels_encoded": no_side.conditional_bits,
                    "signed_reduction_labels_encoded_bits": no_side.raw_reduction_bits,
                    "direct_global_bits": joint_dirichlet_code_bits(codes) if joint else kt_independent_matrix_bits(codes),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    started = time.perf_counter()
    start_state = git_state()
    function_frame, task_frame, checks, formulae, thresholds = run_function_and_task_mdl(args.mode)
    balanced_frame = run_balanced_formula_distribution(formulae[4])
    representation_frame = run_representation_codes()
    checks["n3_prefix_code"] = run_kraft_check(formulae[3], thresholds[3])
    checks["balanced4_functions"] = len(balanced_frame)
    checks["balanced4_fraction_no_more_complex_than_parity"] = float(balanced_frame["no_more_complex_than_parity"].mean())
    checks["minimum_safe_task_gain_bits"] = int(task_frame["safe_compression_gain_bits"].min())
    paths = {
        "function": RESULTS / "discrete_function_mdl_results.csv",
        "task": RESULTS / "discrete_task_mdl_results.csv",
        "balanced": RESULTS / "exact_formula_balanced4_results.csv",
        "representation": RESULTS / "discrete_representation_code_results.csv",
        "checks": RESULTS / "discrete_theory_checks.json",
    }
    function_frame.to_csv(paths["function"], index=False)
    task_frame.to_csv(paths["task"], index=False)
    balanced_frame.to_csv(paths["balanced"], index=False)
    representation_frame.to_csv(paths["representation"], index=False)
    paths["checks"].write_text(json.dumps(checks, indent=2), encoding="utf-8")
    source_paths = [ROOT / "src" / name for name in ("boolean_mdl.py", "run_discrete_theory.py")]
    metadata = {
        "mode": args.mode,
        "elapsed_seconds": time.perf_counter() - started,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "git_at_start": start_state,
        "public_conditions": {
            "dimensions_known": True,
            "gate_library": ["AND", "OR", "XOR", "NAND"],
            "constants_free_sources": [0, 1],
            "formula_exactness": "minimum formula-tree gate count, not minimum DAG",
            "labels_side_information": "reported both as known side information and explicitly encoded",
        },
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in source_paths},
        "artifact_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in paths.values()},
    }
    metadata_path = RESULTS / "discrete_theory_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"function_rows": len(function_frame), "task_rows": len(task_frame), "balanced_rows": len(balanced_frame), "representation_rows": len(representation_frame), "checks": checks, **metadata}, indent=2))


if __name__ == "__main__":
    main()
