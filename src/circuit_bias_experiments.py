"""Exact small-function diagnostics for basis-aware Boolean induction.

This module deliberately solves a bounded problem: minimum *formula* size over
the complete three-input Boolean-function space.  It is not a SAT circuit
oracle and it does not claim minimum DAG size.  The distinction is useful:

* formula gate references expose the inductive bias used during synthesis;
* structural hash-consing reports the DAG obtained by sharing identical
  subexpressions in that formula;
* the paid basis/witness code keeps a prefix residual field compatible with the
  repository's global Circuit-MDL router.  This benchmark itself emits exact
  formula witnesses, so it does not replace the global approximate-task MDL.

The supported AIG/XAG/MIG-style *formula* bases use constants and free
complemented edges, matching common logic-network accounting conventions while
remaining distinct from minimum graph-network synthesis:

* AIG: AND;
* XAG: AND and XOR;
* MIG: three-input majority;
* MIXED: AND, XOR, and three-input majority.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, combinations_with_replacement, product
from math import ceil, log2
from time import perf_counter
from typing import Iterable, Iterator, Sequence

import numpy as np

from boolean_mdl import anf_coefficients as upstream_anf_coefficients
from boolean_mdl import mask_to_values, residual_code_bits, values_to_mask


@dataclass(frozen=True)
class GateBasis:
    """A commutative gate basis with free complemented edges and constants."""

    name: str
    operations: tuple[str, ...]

    def arity(self, operation: str) -> int:
        if operation in ("AND", "XOR"):
            return 2
        if operation == "MAJ":
            return 3
        raise ValueError(f"unsupported operation: {operation}")


BASIS_LIBRARY: dict[str, GateBasis] = {
    "AIG": GateBasis("AIG", ("AND",)),
    "XAG": GateBasis("XAG", ("AND", "XOR")),
    "MIG": GateBasis("MIG", ("MAJ",)),
    "MIXED": GateBasis("MIXED", ("AND", "XOR", "MAJ")),
}


def all_assignments(n_inputs: int) -> np.ndarray:
    """Return little-endian Boolean assignments for a small exact problem."""

    if not 1 <= n_inputs <= 12:
        raise ValueError("exact circuit-bias experiments support 1 through 12 inputs")
    codes = np.arange(1 << n_inputs, dtype=np.uint64)[:, None]
    shifts = np.arange(n_inputs, dtype=np.uint64)[None, :]
    return ((codes >> shifts) & 1).astype(np.uint8)


def vector_to_mask(values: Iterable[int]) -> int:
    """Pack a Boolean vector into a Python integer truth mask."""

    return values_to_mask(values)


def mask_to_vector(mask: int, n_rows: int) -> np.ndarray:
    """Unpack a Python integer truth mask into a Boolean vector."""

    return mask_to_values(mask, n_rows)


def _apply_mask_gate(operation: str, masks: Sequence[int]) -> int:
    if operation == "AND":
        return masks[0] & masks[1]
    if operation == "XOR":
        return masks[0] ^ masks[1]
    if operation == "MAJ":
        a, b, c = masks
        return (a & b) | (a & c) | (b & c)
    raise ValueError(operation)


def _apply_array_gate(operation: str, values: Sequence[np.ndarray]) -> np.ndarray:
    if operation == "AND":
        return values[0] & values[1]
    if operation == "XOR":
        return values[0] ^ values[1]
    if operation == "MAJ":
        a, b, c = values
        return (a & b) | (a & c) | (b & c)
    raise ValueError(operation)


@dataclass(frozen=True)
class BasisFormula:
    """Immutable formula with output complementation represented as a free edge."""

    mask: int
    op: str
    text: str
    gate_count: int
    depth: int
    args: tuple["BasisFormula", ...] = ()
    input_index: int | None = None
    constant: int | None = None
    negated: bool = False

    def inverted(self, universe: int) -> "BasisFormula":
        text = self.text[1:] if self.negated and self.text.startswith("~") else f"~{self.text}"
        return BasisFormula(
            mask=universe ^ self.mask,
            op=self.op,
            text=text,
            gate_count=self.gate_count,
            depth=self.depth,
            args=self.args,
            input_index=self.input_index,
            constant=self.constant,
            negated=not self.negated,
        )

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.uint8)
        if self.op == "INPUT":
            assert self.input_index is not None
            raw = x[:, self.input_index]
        elif self.op == "CONST":
            assert self.constant is not None
            raw = np.full(len(x), self.constant, dtype=np.uint8)
        else:
            raw = _apply_array_gate(self.op, [argument.evaluate(x) for argument in self.args])
        return (1 - raw).astype(np.uint8) if self.negated else raw.astype(np.uint8)

    def core_key(self) -> tuple:
        """Return a structural node key; output polarity is an edge attribute."""

        if self.op == "INPUT":
            return ("INPUT", self.input_index)
        if self.op == "CONST":
            return ("CONST", self.constant)
        child_edges = tuple((argument.core_key(), argument.negated) for argument in self.args)
        return (self.op, child_edges)


@dataclass(frozen=True)
class FormulaSearchResult:
    formulas: dict[int, BasisFormula]
    exact_cost_levels: tuple[dict[int, BasisFormula], ...]
    compile_seconds: float
    candidates_evaluated: int
    complete: bool
    max_gate_count: int


def _formula_tie_key(formula: BasisFormula) -> tuple[int, str]:
    return (formula.depth, formula.text)


def _base_formulas(x: np.ndarray) -> dict[int, BasisFormula]:
    n_rows, n_inputs = x.shape
    universe = (1 << n_rows) - 1
    base: dict[int, BasisFormula] = {}
    for index in range(n_inputs):
        mask = vector_to_mask(x[:, index])
        positive = BasisFormula(mask, "INPUT", f"x{index}", 0, 0, input_index=index)
        negative = positive.inverted(universe)
        base.setdefault(positive.mask, positive)
        base.setdefault(negative.mask, negative)
    zero = BasisFormula(0, "CONST", "0", 0, 0, constant=0)
    one = BasisFormula(universe, "CONST", "1", 0, 0, constant=1)
    base.setdefault(0, zero)
    base.setdefault(universe, one)
    return base


def _cost_partitions(total: int, arity: int) -> Iterator[tuple[int, ...]]:
    """Yield nondecreasing nonnegative tuples that sum to ``total``."""

    if arity == 2:
        for left in range(total + 1):
            right = total - left
            if left <= right:
                yield (left, right)
        return
    if arity == 3:
        for first in range(total + 1):
            for second in range(first, total + 1):
                third = total - first - second
                if second <= third:
                    yield (first, second, third)
        return
    raise ValueError(arity)


def _argument_tuples(levels: Sequence[dict[int, BasisFormula]], costs: tuple[int, ...]) -> Iterator[tuple[BasisFormula, ...]]:
    """Iterate commutative argument tuples without permutation duplicates."""

    if len(costs) == 2:
        left, right = costs
        if left == right:
            yield from combinations_with_replacement(levels[left].values(), 2)
        else:
            yield from product(levels[left].values(), levels[right].values())
        return

    first, second, third = costs
    if first == second == third:
        yield from combinations_with_replacement(levels[first].values(), 3)
    elif first == second:
        for pair in combinations_with_replacement(levels[first].values(), 2):
            for final in levels[third].values():
                yield pair + (final,)
    elif second == third:
        for initial in levels[first].values():
            for pair in combinations_with_replacement(levels[second].values(), 2):
                yield (initial,) + pair
    else:
        yield from product(levels[first].values(), levels[second].values(), levels[third].values())


def enumerate_formula_space(
    x: np.ndarray,
    basis: GateBasis,
    max_gates: int = 12,
    stop_masks: Iterable[int] | None = None,
    max_candidates: int = 20_000_000,
) -> FormulaSearchResult:
    """Enumerate exact minimum-formula costs over the observed assignments.

    Dynamic programming is by total formula gate references.  Therefore the
    first time a truth mask appears is an exact formula-size optimum within the
    selected basis.  ``complete`` means every Boolean labeling of the supplied
    rows has been reached, which is practical for the full three-input space.
    """

    started = perf_counter()
    x = np.asarray(x, dtype=np.uint8)
    if x.ndim != 2 or len(x) == 0:
        raise ValueError("x must be a nonempty 2-D Boolean array")
    if not np.all((x == 0) | (x == 1)):
        raise ValueError("x must be Boolean")
    if max_gates < 0:
        raise ValueError("max_gates must be nonnegative")

    universe = (1 << len(x)) - 1
    base = _base_formulas(x)
    formulas = dict(base)
    levels: list[dict[int, BasisFormula]] = [dict(base)]
    targets = set(stop_masks or ())
    total_functions = 1 << len(x)
    candidates_evaluated = 0
    reached_gate_count = 0

    if targets and targets.issubset(formulas):
        return FormulaSearchResult(formulas, tuple(levels), perf_counter() - started, 0, len(formulas) == total_functions, 0)

    for gate_count in range(1, max_gates + 1):
        level: dict[int, BasisFormula] = {}
        for operation in basis.operations:
            arity = basis.arity(operation)
            for costs in _cost_partitions(gate_count - 1, arity):
                for arguments in _argument_tuples(levels, costs):
                    candidates_evaluated += 1
                    if candidates_evaluated > max_candidates:
                        raise RuntimeError(
                            f"formula enumeration exceeded {max_candidates} candidates for {basis.name}; "
                            "reduce max_gates or use a SAT backend"
                        )
                    mask = _apply_mask_gate(operation, [argument.mask for argument in arguments])
                    text = f"{operation}({','.join(argument.text for argument in arguments)})"
                    candidate = BasisFormula(
                        mask=mask,
                        op=operation,
                        text=text,
                        gate_count=gate_count,
                        depth=1 + max(argument.depth for argument in arguments),
                        args=tuple(arguments),
                    )
                    for option in (candidate, candidate.inverted(universe)):
                        if option.mask in formulas:
                            continue
                        prior = level.get(option.mask)
                        if prior is None or _formula_tie_key(option) < _formula_tie_key(prior):
                            level[option.mask] = option
        levels.append(level)
        formulas.update(level)
        reached_gate_count = gate_count
        if len(formulas) == total_functions or (targets and targets.issubset(formulas)):
            break
        # Exact formula-size levels can contain gaps: a later tree may duplicate
        # an already-optimal subtree and thereby reach a new function.  Do not
        # stop merely because one exact-cost level is empty.

    return FormulaSearchResult(
        formulas=formulas,
        exact_cost_levels=tuple(levels),
        compile_seconds=perf_counter() - started,
        candidates_evaluated=candidates_evaluated,
        complete=len(formulas) == total_functions,
        max_gate_count=reached_gate_count,
    )


def synthesize_min_formula(
    x: np.ndarray,
    y: np.ndarray,
    basis: GateBasis,
    max_gates: int = 12,
    max_candidates: int = 20_000_000,
) -> tuple[BasisFormula | None, FormulaSearchResult]:
    """Return an exact minimum formula for ``y`` when it fits the gate budget."""

    x = np.asarray(x, dtype=np.uint8)
    y = np.asarray(y, dtype=np.uint8).reshape(-1)
    if len(x) != len(y):
        raise ValueError("x and y must have the same number of rows")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("y must be Boolean")
    target = vector_to_mask(y)
    result = enumerate_formula_space(x, basis, max_gates, stop_masks=(target,), max_candidates=max_candidates)
    return result.formulas.get(target), result


def formula_dag_metrics(formula: BasisFormula) -> dict[str, int]:
    """Hash-cons identical formula subexpressions and report structural metrics."""

    gates: dict[tuple, tuple[tuple, ...]] = {}

    def visit(node: BasisFormula) -> None:
        if node.op in ("INPUT", "CONST"):
            return
        key = node.core_key()
        if key in gates:
            return
        edges = tuple((argument.core_key(), argument.negated) for argument in node.args)
        gates[key] = edges
        for argument in node.args:
            visit(argument)

    visit(formula)
    fanout = {key: 0 for key in gates}
    for edges in gates.values():
        for child_key, _ in edges:
            if child_key in fanout:
                fanout[child_key] += 1
    output_key = formula.core_key()
    if output_key in fanout:
        fanout[output_key] += 1
    return {
        "formula_gate_refs": formula.gate_count,
        "observed_hashconsed_dag_upper_bound_gates": len(gates),
        "depth": formula.depth,
        "unique_gate_input_edges": sum(len(edges) for edges in gates.values()),
        "internal_gate_fanout_max_including_output": max(fanout.values(), default=0),
    }


def basis_formula_dag_upper_bound_bits(
    formula: BasisFormula, basis: GateBasis, n_inputs: int, n_bases: int | None = None
) -> int:
    """Return a fixed-width upper bound for the hash-consed DAG description."""

    if n_inputs < 1:
        raise ValueError("n_inputs must be positive")
    if not basis.operations:
        raise ValueError("basis must contain at least one operation")

    def validate(node: BasisFormula) -> None:
        if not isinstance(node.negated, bool):
            raise ValueError("edge polarity must be Boolean")
        if node.op == "INPUT":
            if node.args or node.input_index is None or not 0 <= node.input_index < n_inputs:
                raise ValueError("input node is outside the public input range")
            return
        if node.op == "CONST":
            if node.args or node.constant not in (0, 1):
                raise ValueError("constant node must encode public source 0 or 1")
            return
        if node.op not in basis.operations:
            raise ValueError(f"operation {node.op} is not in basis {basis.name}")
        if len(node.args) != basis.arity(node.op):
            raise ValueError(f"operation {node.op} has the wrong arity")
        for argument in node.args:
            validate(argument)

    validate(formula)
    metrics = formula_dag_metrics(formula)
    gates = metrics["observed_hashconsed_dag_upper_bound_gates"]
    source_bits = ceil(log2(max(2, n_inputs + 2 + gates)))
    operation_bits = ceil(log2(max(1, len(basis.operations))))
    basis_bits = ceil(log2(max(1, n_bases))) if n_bases is not None else 0
    # Elias-gamma length of gates+1, followed by gate records and one output edge.
    count_bits = 2 * int(log2(gates + 1)) + 1
    gate_bits = 0
    seen: set[tuple] = set()

    def visit(node: BasisFormula) -> None:
        nonlocal gate_bits
        if node.op in ("INPUT", "CONST"):
            return
        key = node.core_key()
        if key in seen:
            return
        seen.add(key)
        gate_bits += operation_bits + len(node.args) * (source_bits + 1)
        for argument in node.args:
            visit(argument)

    visit(formula)
    return int(basis_bits + count_bits + gate_bits + source_bits + 1)


def residual_description_bits(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Prefix residual code shared with the repository's Circuit-MDL theory."""

    y_true = np.asarray(y_true, dtype=np.uint8).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.uint8).reshape(-1)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal length")
    return float(residual_code_bits(y_true, y_pred))


def _fwht(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    size = len(result)
    if size == 0 or size & (size - 1):
        raise ValueError("Walsh-Hadamard transform requires a power-of-two length")
    width = 1
    while width < size:
        for start in range(0, size, width * 2):
            left = result[start : start + width].copy()
            right = result[start + width : start + 2 * width].copy()
            result[start : start + width] = left + right
            result[start + width : start + 2 * width] = left - right
        width *= 2
    return result


def _popcount(value: int) -> int:
    """Python 3.8-compatible population count used on the 210 server."""

    integer = int(value)
    return integer.bit_count() if hasattr(integer, "bit_count") else bin(integer).count("1")


def _anf_coefficients(y: np.ndarray, n_inputs: int) -> np.ndarray:
    coefficients = upstream_anf_coefficients(y)
    if len(coefficients) != 1 << n_inputs:
        raise ValueError("ANF input width does not match n_inputs")
    return coefficients


def _certificate_sizes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    n_inputs = x.shape[1]
    sizes = np.empty(len(x), dtype=np.int64)
    for row_index, row in enumerate(x):
        found = n_inputs
        for size in range(n_inputs + 1):
            certified = False
            for subset in combinations(range(n_inputs), size):
                if subset:
                    matches = np.all(x[:, subset] == row[list(subset)], axis=1)
                else:
                    matches = np.ones(len(x), dtype=bool)
                if np.all(y[matches] == y[row_index]):
                    found = size
                    certified = True
                    break
            if certified:
                break
        sizes[row_index] = found
    return sizes


def boolean_diagnostics(y: np.ndarray, n_inputs: int, noise_rate: float = 0.1) -> dict[str, float | int | str]:
    """Compute exact small-function diagnostics on a complete truth table."""

    x = all_assignments(n_inputs)
    y = np.asarray(y, dtype=np.uint8).reshape(-1)
    if len(y) != len(x):
        raise ValueError("diagnostics require the complete truth table")
    if not 0.0 <= noise_rate <= 0.5:
        raise ValueError("noise_rate must be in [0, 0.5]")

    influences = np.empty(n_inputs, dtype=np.float64)
    indices = np.arange(len(y), dtype=np.int64)
    for bit in range(n_inputs):
        influences[bit] = np.mean(y != y[indices ^ (1 << bit)])

    signed = 1.0 - 2.0 * y.astype(np.float64)
    fourier = _fwht(signed) / len(signed)
    masses = fourier * fourier
    degrees = np.asarray([_popcount(index) for index in range(len(y))], dtype=np.int64)
    support = np.abs(fourier) > 1e-10
    fourier_degree = int(degrees[support].max(initial=0))
    noise_correlation = float(np.sum(masses * (1.0 - 2.0 * noise_rate) ** degrees))

    anf = _anf_coefficients(y, n_inputs)
    anf_support = np.flatnonzero(anf)
    anf_degree = int(max((_popcount(index) for index in anf_support), default=0))

    violations = 0
    directed_edges = 0
    for index in range(len(y)):
        for bit in range(n_inputs):
            if not (index & (1 << bit)):
                directed_edges += 1
                violations += int(y[index] > y[index | (1 << bit)])

    symmetry_errors = 0
    for weight in range(n_inputs + 1):
        group = y[x.sum(axis=1) == weight]
        if len(group):
            ones = int(group.sum())
            symmetry_errors += min(ones, len(group) - ones)

    certificates = _certificate_sizes(x, y)
    c0_values = certificates[y == 0]
    c1_values = certificates[y == 1]
    c0 = int(c0_values.max(initial=0))
    c1 = int(c1_values.max(initial=0))

    if np.all(y == y[0]):
        family = "constant"
    elif int(np.sum(influences > 0)) == 1:
        family = "literal"
    elif anf_degree <= 1:
        family = "affine"
    elif violations == 0 and symmetry_errors == 0:
        family = "monotone_symmetric"
    elif violations == 0:
        family = "monotone"
    elif symmetry_errors == 0:
        family = "symmetric"
    else:
        family = "general"

    return {
        "family": family,
        "positive_fraction": float(np.mean(y)),
        "relevant_inputs": int(np.sum(influences > 0)),
        "total_influence": float(influences.sum()),
        "max_influence": float(influences.max(initial=0.0)),
        "fourier_degree": fourier_degree,
        "fourier_mass_degree_le_1": float(masses[degrees <= 1].sum()),
        "fourier_mass_degree_le_2": float(masses[degrees <= 2].sum()),
        "anf_degree": anf_degree,
        "anf_terms": int(len(anf_support)),
        "monotonicity_violation_rate": float(violations / max(1, directed_edges)),
        "symmetry_error_rate": float(symmetry_errors / len(y)),
        "certificate_c0": c0,
        "certificate_c1": c1,
        "certificate_mean": float(np.mean(certificates)),
        "noise_sensitivity": float(0.5 * (1.0 - noise_correlation)),
        "noise_rate": float(noise_rate),
    }


def named_truth_masks(n_inputs: int = 3) -> dict[str, int]:
    """Canonical named functions used to make basis specialization readable."""

    if n_inputs != 3:
        raise ValueError("the v0 named exact-oracle suite uses three inputs")
    x = all_assignments(n_inputs)
    functions = {
        "zero": np.zeros(len(x), dtype=np.uint8),
        "literal_x0": x[:, 0],
        "and3": np.bitwise_and.reduce(x, axis=1),
        "or3": np.bitwise_or.reduce(x, axis=1),
        "parity3": np.bitwise_xor.reduce(x, axis=1),
        "majority3": (x.sum(axis=1) >= 2).astype(np.uint8),
        "mux3": np.where(x[:, 0] == 1, x[:, 1], x[:, 2]).astype(np.uint8),
        "exactly_one3": (x.sum(axis=1) == 1).astype(np.uint8),
    }
    return {name: vector_to_mask(values) for name, values in functions.items()}


def benchmark_exact_bases(mode: str = "full", max_gates: int = 12) -> tuple[list[dict], list[dict]]:
    """Run the full three-input oracle or its named-task smoke subset."""

    if mode not in ("smoke", "full"):
        raise ValueError("mode must be smoke or full")
    x = all_assignments(3)
    named = named_truth_masks(3)
    names_by_mask = {mask: name for name, mask in named.items()}
    target_masks = sorted(named.values()) if mode == "smoke" else list(range(1 << len(x)))
    oracle_rows: list[dict] = []

    for basis in BASIS_LIBRARY.values():
        search = enumerate_formula_space(x, basis, max_gates=max_gates)
        for function_id in target_masks:
            formula = search.formulas.get(function_id)
            if formula is None:
                oracle_rows.append(
                    {
                        "function_id": function_id,
                        "task": names_by_mask.get(function_id, f"truth_table_{function_id:03d}"),
                        "basis": basis.name,
                        "exact": 0,
                        "search_complete": int(search.complete),
                        "formula_gate_refs": np.nan,
                        "observed_hashconsed_dag_upper_bound_gates": np.nan,
                        "depth": np.nan,
                        "unique_gate_input_edges": np.nan,
                        "internal_gate_fanout_max_including_output": np.nan,
                        "basis_formula_dag_code_upper_bound_bits": np.nan,
                        "residual_bits": np.nan,
                        "basis_formula_witness_plus_residual_bits": np.nan,
                        "compile_seconds": search.compile_seconds,
                        "candidates_evaluated": search.candidates_evaluated,
                        "expression": "",
                    }
                )
                continue
            prediction = formula.evaluate(x)
            target = mask_to_vector(function_id, len(x))
            metrics = formula_dag_metrics(formula)
            model_bits = basis_formula_dag_upper_bound_bits(formula, basis, 3, n_bases=len(BASIS_LIBRARY))
            residual_bits = residual_description_bits(target, prediction)
            oracle_rows.append(
                {
                    "function_id": function_id,
                    "task": names_by_mask.get(function_id, f"truth_table_{function_id:03d}"),
                    "basis": basis.name,
                    "exact": int(np.array_equal(prediction, target)),
                    "search_complete": int(search.complete),
                    **metrics,
                    "basis_formula_dag_code_upper_bound_bits": model_bits,
                    "residual_bits": residual_bits,
                    "basis_formula_witness_plus_residual_bits": model_bits + residual_bits,
                    "compile_seconds": search.compile_seconds,
                    "candidates_evaluated": search.candidates_evaluated,
                    "expression": formula.text,
                }
            )

    by_function: dict[int, list[dict]] = {}
    for row in oracle_rows:
        by_function.setdefault(int(row["function_id"]), []).append(row)
    for rows in by_function.values():
        exact_rows = [row for row in rows if row["exact"]]
        min_gates = min((row["formula_gate_refs"] for row in exact_rows), default=np.nan)
        min_dag = min((row["observed_hashconsed_dag_upper_bound_gates"] for row in exact_rows), default=np.nan)
        min_witness_code = min((row["basis_formula_witness_plus_residual_bits"] for row in exact_rows), default=np.nan)
        for row in rows:
            row["formula_optimal_across_bases"] = int(bool(row["exact"]) and row["formula_gate_refs"] == min_gates)
            # These two comparisons are only over the minimum-formula witness
            # retained for each basis.  Hash-consing gives a DAG upper bound;
            # it is neither an exact minimum-DAG search nor an exact code search
            # over all formula ties.
            row["smallest_observed_hashconsed_dag_upper_bound_across_bases"] = int(
                bool(row["exact"]) and row["observed_hashconsed_dag_upper_bound_gates"] == min_dag
            )
            row["smallest_observed_witness_code_across_basis_formula_witnesses"] = int(
                bool(row["exact"]) and row["basis_formula_witness_plus_residual_bits"] == min_witness_code
            )
            row["formula_regret_gates"] = float(row["formula_gate_refs"] - min_gates) if row["exact"] else np.nan

    diagnostic_rows = []
    for function_id in target_masks:
        y = mask_to_vector(function_id, len(x))
        diagnostic_rows.append(
            {
                "function_id": function_id,
                "task": names_by_mask.get(function_id, f"truth_table_{function_id:03d}"),
                **boolean_diagnostics(y, 3),
            }
        )
    return oracle_rows, diagnostic_rows
