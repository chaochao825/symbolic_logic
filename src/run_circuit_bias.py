"""Run the bounded basis-aware formula-complexity benchmark.

Unlike the older runners, this command writes to a fresh run directory and
refuses to overwrite existing artifacts.  The resulting manifest explicitly
labels the exact oracle as a three-input *formula* oracle, not a general circuit
or DAG optimum.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Iterable

import numpy as np

from circuit_bias_experiments import BASIS_LIBRARY, benchmark_exact_bases, named_truth_masks


ROOT = Path(__file__).resolve().parents[1]


def normalized_sha256(path: Path) -> str:
    """Hash text with LF newlines so provenance survives Windows checkout."""

    payload = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(payload).hexdigest()


def installed_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "not-installed"


def git_state() -> dict:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unavailable", "dirty": None}


def _csv_value(value):
    if isinstance(value, (float, np.floating)) and math.isnan(float(value)):
        return ""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty artifact: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in fields})


def _mean(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if value is not None and not math.isnan(float(value))]
    return float(np.mean(finite)) if finite else float("nan")


def summarize_bases(oracle_rows: list[dict], diagnostic_rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    diagnostics = {int(row["function_id"]): row for row in diagnostic_rows}
    by_basis: dict[str, list[dict]] = defaultdict(list)
    by_basis_family: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in oracle_rows:
        by_basis[str(row["basis"])].append(row)
        family = str(diagnostics[int(row["function_id"])]["family"])
        by_basis_family[(str(row["basis"]), family)].append(row)

    summary_rows = []
    for basis, rows in sorted(by_basis.items()):
        exact = [row for row in rows if int(row["exact"]) == 1]
        complete = len(rows) == 256 and all(int(row["exact"]) == 1 and int(row["search_complete"]) == 1 for row in rows)
        scope = (
            "exact minimum formula size on the complete 3-input function space"
            if complete
            else "bounded formula search on a named or incomplete 3-input subset"
        )
        summary_rows.append(
            {
                "basis": basis,
                "functions": len(rows),
                "exact_functions": len(exact),
                "exact_coverage": len(exact) / len(rows),
                "mean_formula_gate_refs": _mean(row["formula_gate_refs"] for row in exact),
                "mean_observed_hashconsed_dag_upper_bound_gates": _mean(
                    row["observed_hashconsed_dag_upper_bound_gates"] for row in exact
                ),
                "mean_basis_formula_dag_code_upper_bound_bits": _mean(
                    row["basis_formula_dag_code_upper_bound_bits"] for row in exact
                ),
                "formula_optimal_functions": sum(int(row["formula_optimal_across_bases"]) for row in exact),
                "smallest_observed_hashconsed_dag_upper_bound_functions": sum(
                    int(row["smallest_observed_hashconsed_dag_upper_bound_across_bases"]) for row in exact
                ),
                "smallest_observed_witness_code_functions": sum(
                    int(row["smallest_observed_witness_code_across_basis_formula_witnesses"]) for row in exact
                ),
                "compile_seconds": max(float(row["compile_seconds"]) for row in rows),
                "candidates_evaluated": max(int(row["candidates_evaluated"]) for row in rows),
                "scope": scope,
            }
        )

    family_rows = []
    for (basis, family), rows in sorted(by_basis_family.items()):
        exact = [row for row in rows if int(row["exact"]) == 1]
        family_rows.append(
            {
                "basis": basis,
                "family": family,
                "functions": len(rows),
                "exact_coverage": len(exact) / len(rows),
                "mean_formula_gate_refs": _mean(row["formula_gate_refs"] for row in exact),
                "mean_observed_hashconsed_dag_upper_bound_gates": _mean(
                    row["observed_hashconsed_dag_upper_bound_gates"] for row in exact
                ),
                "mean_basis_formula_dag_code_upper_bound_bits": _mean(
                    row["basis_formula_dag_code_upper_bound_bits"] for row in exact
                ),
                "formula_optimal_rate": _mean(row["formula_optimal_across_bases"] for row in exact),
                "smallest_observed_witness_code_rate": _mean(
                    row["smallest_observed_witness_code_across_basis_formula_witnesses"] for row in exact
                ),
            }
        )

    named = set(named_truth_masks())
    named_rows = [row for row in oracle_rows if row["task"] in named]
    return summary_rows, family_rows, named_rows


def search_representation_bridge(path: Path) -> list[dict]:
    """Correct the parity/random-LUT interpretation in the legacy result table."""

    if not path.exists():
        return []
    grouped: dict[tuple[str, int, str], list[dict]] = defaultdict(list)
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped[(row["family"], int(row["n_bits"]), row["method"])].append(row)

    output = []
    for (family, n_bits, method), rows in sorted(grouped.items()):
        gate_values = [float(row["gate_count"]) for row in rows if method == "GateBeam" and row.get("gate_count")]
        if family == "parity":
            constructive_basis = "XAG"
            constructive_gates = n_bits - 1
            representation_status = "circuit_compressible_with_constructive_xag_upper_bound"
            observed_status = "method_failed_exact_recovery_despite_constructive_upper_bound" if not any(row["exact_full_recovery"].lower() == "true" for row in rows) else "exact_recovery"
        else:
            constructive_basis = ""
            constructive_gates = float("nan")
            representation_status = "random_lut_negative_control_no_instance_lower_bound"
            observed_status = "no_compositional_generalization" if _mean(float(row["holdout_accuracy"]) for row in rows) < 0.6 else "inspect"
        output.append(
            {
                "family": family,
                "n_bits": n_bits,
                "method": method,
                "seeds": len(rows),
                "mean_holdout_accuracy": _mean(float(row["holdout_accuracy"]) for row in rows),
                "mean_full_accuracy": _mean(float(row["full_accuracy"]) for row in rows),
                "exact_full_recovery_rate": _mean(1.0 if row["exact_full_recovery"].lower() == "true" else 0.0 for row in rows),
                "mean_gatebeam_formula_gate_refs": _mean(gate_values),
                "constructive_basis": constructive_basis,
                "constructive_formula_gate_upper_bound": constructive_gates,
                "representation_status": representation_status,
                "observed_method_status": observed_status,
                "interpretation_limit": "GateBeam gate count is not an exact circuit lower bound; sampled random LUTs have no proved per-instance lower bound",
            }
        )
    return output


def default_output_dir(mode: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return ROOT / "results" / "runs" / f"circuit_bias_v0_{mode}_{stamp}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--max-gates", type=int, default=12)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve() if args.output_dir else default_output_dir(args.mode)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"immutable run directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc)
    start_git = git_state()
    oracle_rows, diagnostic_rows = benchmark_exact_bases(args.mode, max_gates=args.max_gates)
    summary_rows, family_rows, named_rows = summarize_bases(oracle_rows, diagnostic_rows)
    legacy_path = ROOT / "results" / "noncompressible_scaling_results.csv"
    bridge_rows = search_representation_bridge(legacy_path)

    artifacts = {
        "basis_oracle_results.csv": oracle_rows,
        "boolean_diagnostics.csv": diagnostic_rows,
        "basis_summary.csv": summary_rows,
        "basis_family_summary.csv": family_rows,
        "named_task_basis_results.csv": named_rows,
    }
    if bridge_rows:
        artifacts["search_vs_representation_results.csv"] = bridge_rows
    for name, rows in artifacts.items():
        write_rows(output_dir / name, rows)

    sources = [
        ROOT / "src" / "boolean_mdl.py",
        ROOT / "src" / "circuit_bias_experiments.py",
        ROOT / "src" / "run_circuit_bias.py",
    ]
    metadata = {
        "benchmark": "Circuit Bias Benchmark v0",
        "mode": args.mode,
        "max_gates": args.max_gates,
        "elapsed_seconds": time.perf_counter() - started,
        "started_utc": started_utc.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "working_directory": str(Path.cwd()),
        "invocation": [sys.executable, *sys.argv],
        "python_executable": sys.executable,
        "python_prefix": sys.prefix,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": installed_version("pandas"),
        "matplotlib": installed_version("matplotlib"),
        "platform": platform.platform(),
        "hash_mode": "sha256 after CRLF-to-LF normalization for tracked text",
        "requirements_lock_normalized_sha256": normalized_sha256(ROOT / "requirements-lock.txt"),
        "cpu_count": os.cpu_count(),
        "thread_environment": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
        },
        "git_at_start": start_git,
        "scope": (
            "Exact minimum Boolean formula size for the complete three-input function space"
            if args.mode == "full"
            and len(diagnostic_rows) == 256
            and all(int(row["exact"]) == 1 and int(row["search_complete"]) == 1 for row in oracle_rows)
            else "Bounded Boolean formula search for a named or incomplete three-input subset"
        )
        + " under AIG/XAG/MIG-style and MIXED formula bases; hash-consed DAG metrics are derived from the retained minimum-formula witness and are neither an exact DAG optimum nor an exact minimum-code search",
        "basis_semantics": {name: {"operations": list(basis.operations), "free_complemented_edges": True, "free_constants": True} for name, basis in BASIS_LIBRARY.items()},
        "code_accounting": {
            "public_n_inputs": 3,
            "basis_routes": len(BASIS_LIBRARY),
            "basis_route_tag_bits": math.ceil(math.log2(len(BASIS_LIBRARY))),
            "selected_witness_kraft_sum": sum(
                2.0 ** -float(row["basis_formula_dag_code_upper_bound_bits"])
                for row in oracle_rows
                if int(row["exact"]) == 1
            ),
            "kraft_check_scope": "sanity check over retained witnesses only; the field grammar supplies the prefix-code argument",
            "residual_semantics": "enumerative error-mask code; exact witnesses still pay the zero-error count header",
            "not_included": "no RawLabels/ANF/threshold/ROBDD global route and no runtime confidence fallback in this runner",
        },
        "source_normalized_sha256": {str(path.relative_to(ROOT)): normalized_sha256(path) for path in sources},
        "input_normalized_sha256": {str(legacy_path.relative_to(ROOT)): normalized_sha256(legacy_path)} if legacy_path.exists() else {},
        "artifact_normalized_sha256": {name: normalized_sha256(output_dir / name) for name in artifacts},
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "oracle_rows": len(oracle_rows), "diagnostic_rows": len(diagnostic_rows), **metadata}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
