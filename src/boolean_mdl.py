"""Executable discrete coding theory for Boolean representations and circuits.

The central rule is that every reported bit count must correspond to a
decodable code under an explicitly named model family.  Continuous log-det
rates, empirical entropy, gate count, and hardware cost are deliberately kept
separate.

All description lengths are conditional on the public benchmark dimensions
(``n_inputs``, sample count, and gate library).  This convention is used for
every competing language, so shared headers cancel in comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement, permutations, product
from math import ceil, factorial, lgamma, log, log2
from typing import Iterable, Sequence

import numpy as np


GATE_OPS = ("AND", "OR", "XOR", "NAND")


def ceil_log2_count(count: int) -> int:
    """Fixed-width bits required to index ``count`` alternatives."""
    if count <= 1:
        return 0
    return (count - 1).bit_length()


def elias_delta_bits(value: int) -> int:
    """Length of the Elias-delta prefix code for a positive integer."""
    if value < 1:
        raise ValueError("Elias-delta encodes positive integers")
    leading = value.bit_length() - 1
    return leading + 2 * ((leading + 1).bit_length() - 1) + 1


def signed_integer_bits(value: int) -> int:
    """Prefix length after a zig-zag map from integers to positive integers."""
    mapped = 2 * value + 1 if value >= 0 else -2 * value
    return elias_delta_bits(mapped)


def log2_binomial(n: int, k: int) -> float:
    if not 0 <= k <= n:
        raise ValueError("k must be between zero and n")
    if k == 0 or k == n:
        return 0.0
    return (lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)) / log(2.0)


def enumerative_subset_bits(universe: int, selected: int) -> int:
    """Prefix-code a subset by count then lexicographic rank.

    ``selected`` is encoded by Elias-delta as ``selected + 1``.  Conditional
    on that count, the subset rank uses a fixed-width enumerative code.
    """
    if not 0 <= selected <= universe:
        raise ValueError("selected must be between zero and universe")
    combinations = 1 if selected in (0, universe) else __import__("math").comb(universe, selected)
    return elias_delta_bits(selected + 1) + ceil_log2_count(combinations)


def truth_table_bits(n_inputs: int, n_outputs: int = 1) -> int:
    if n_inputs < 0 or n_outputs < 1:
        raise ValueError("invalid truth-table dimensions")
    return n_outputs * (1 << n_inputs)


def _binary_array(values: object, name: str) -> np.ndarray:
    """Validate binary values before uint8 conversion so integers cannot wrap."""
    raw = np.asarray(values)
    if raw.dtype.kind not in "buifc" or not np.all(np.isfinite(raw)):
        raise ValueError(f"{name} expects finite zeros and ones")
    if np.any((raw != 0) & (raw != 1)):
        raise ValueError(f"{name} expects zeros and ones")
    return raw.astype(np.uint8, copy=False)


def kt_binary_ideal_bits(values: Iterable[int]) -> float:
    """Krichevsky-Trofimov mixture codelength for one binary sequence."""
    array = _binary_array(list(values), "KT binary code").reshape(-1)
    ones = int(array.sum())
    zeros = len(array) - ones
    log_probability = (
        lgamma(ones + 0.5)
        + lgamma(zeros + 0.5)
        - lgamma(len(array) + 1.0)
        - 2.0 * lgamma(0.5)
    )
    return float(-log_probability / log(2.0))


def kt_binary_prefix_bits(values: Iterable[int]) -> int:
    """Shannon-Fano integer length induced by the KT mixture probability."""
    return int(ceil(kt_binary_ideal_bits(values)))


def kt_independent_matrix_bits(binary_codes: np.ndarray) -> int:
    """Product KT code across public bit positions; captures marginal bias."""
    codes = _binary_array(binary_codes, "independent KT code")
    if codes.ndim == 1:
        codes = codes[:, None]
    ideal = sum(kt_binary_ideal_bits(codes[:, column]) for column in range(codes.shape[1]))
    return int(ceil(ideal))


def joint_dirichlet_code_bits(binary_codes: np.ndarray, alpha: float = 0.5) -> int:
    """Joint codeword KT/Dirichlet-mixture length for a finite binary alphabet."""
    codes = _binary_array(binary_codes, "joint Dirichlet code")
    if codes.ndim == 1:
        codes = codes[:, None]
    if codes.shape[1] > 20:
        raise ValueError("joint code is bounded to at most 20 bits")
    alphabet = 1 << codes.shape[1]
    symbols = np.sum(codes.astype(np.uint64) << np.arange(codes.shape[1], dtype=np.uint64), axis=1)
    counts = np.bincount(symbols.astype(np.int64), minlength=alphabet)
    n = len(codes)
    log_probability = lgamma(alphabet * alpha) - lgamma(n + alphabet * alpha)
    log_probability += sum(lgamma(int(count) + alpha) - lgamma(alpha) for count in counts)
    return int(ceil(-log_probability / log(2.0)))


@dataclass(frozen=True)
class DiscreteRateReduction:
    global_bits: int
    conditional_bits: int
    raw_reduction_bits: int
    gain_vs_global_route_bits: int
    labels_are_side_information: bool
    label_bits: int
    route_tag_bits: int
    routed_global_baseline_bits: int
    routed_best_bits: int

    @property
    def routed_reduction_bits(self) -> int:
        """Compatibility alias; prefer ``gain_vs_global_route_bits``."""
        return self.gain_vs_global_route_bits


def discrete_rate_reduction(
    binary_codes: np.ndarray,
    labels: np.ndarray,
    *,
    joint: bool = True,
    labels_are_side_information: bool = True,
) -> DiscreteRateReduction:
    """Compare one global universal code against class-conditional codes.

    If labels are side information, their cost is zero on both sides.  If they
    are not, the same KT label length is added to both competing joint codes
    and therefore cancels in the reduction.  The safe comparison is a real
    two-route prefix code: both its global baseline and selected best route pay
    one route-tag bit.  The common tag cancels in
    ``gain_vs_global_route_bits``; this is not a gain against the standalone,
    untagged global code.
    """
    codes = _binary_array(binary_codes, "binary representation")
    labels = _binary_array(labels, "binary labels").reshape(-1)
    if len(codes) != len(labels):
        raise ValueError("codes and labels must contain the same samples")
    encoder = joint_dirichlet_code_bits if joint else kt_independent_matrix_bits
    global_bits = encoder(codes)
    conditional = sum(encoder(codes[labels == label]) for label in np.unique(labels))
    label_bits = 0 if labels_are_side_information else kt_binary_prefix_bits(labels)
    global_total = global_bits + label_bits
    conditional_total = conditional + label_bits
    raw = global_total - conditional_total
    route_tag = 1
    routed_global = route_tag + global_total
    routed_best = route_tag + min(global_total, conditional_total)
    return DiscreteRateReduction(
        global_bits=global_total,
        conditional_bits=conditional_total,
        raw_reduction_bits=raw,
        gain_vs_global_route_bits=routed_global - routed_best,
        labels_are_side_information=labels_are_side_information,
        label_bits=label_bits,
        route_tag_bits=route_tag,
        routed_global_baseline_bits=routed_global,
        routed_best_bits=routed_best,
    )


def residual_code_bits(y_true: np.ndarray, y_pred: np.ndarray) -> int:
    """Prefix-code a binary error mask by its count and subset rank."""
    truth = _binary_array(y_true, "residual truth").reshape(-1)
    prediction = _binary_array(y_pred, "residual prediction").reshape(-1)
    if len(truth) != len(prediction):
        raise ValueError("truth and prediction must have equal length")
    errors = int(np.sum(truth != prediction))
    return enumerative_subset_bits(len(truth), errors)


def occam_error_bound(empirical_error: float, description_bits: float, samples: int, delta: float = 0.05) -> float:
    """Hoeffding-Kraft bound for a prefix-coded deterministic classifier."""
    if not 0.0 <= empirical_error <= 1.0 or samples < 1 or not 0.0 < delta < 1.0:
        raise ValueError("invalid bound arguments")
    penalty = np.sqrt((description_bits * log(2.0) + log(1.0 / delta)) / (2.0 * samples))
    return float(min(1.0, empirical_error + penalty))


def apply_gate_mask(op: str, left: int, right: int, universe: int) -> int:
    if op == "AND":
        return left & right
    if op == "OR":
        return left | right
    if op == "XOR":
        return left ^ right
    if op == "NAND":
        return universe ^ (left & right)
    raise ValueError(op)


def values_to_mask(values: Iterable[int]) -> int:
    result = 0
    for index, value in enumerate(_binary_array(list(values), "truth values").reshape(-1)):
        if int(value):
            result |= 1 << index
    return result


def mask_to_values(mask: int, samples: int) -> np.ndarray:
    return np.asarray([(mask >> index) & 1 for index in range(samples)], dtype=np.uint8)


def input_truth_masks(n_inputs: int) -> tuple[int, ...]:
    samples = 1 << n_inputs
    return tuple(values_to_mask((np.arange(samples, dtype=np.uint64) >> bit) & 1) for bit in range(n_inputs))


@dataclass(frozen=True)
class Formula:
    op: str
    mask: int
    gates: int
    text: str
    left: "Formula | None" = None
    right: "Formula | None" = None
    source_index: int | None = None


def exact_formula_library(n_inputs: int, max_gates: int = 5) -> dict[int, Formula]:
    """Dynamic program returning minimum *formula* size under ``GATE_OPS``.

    One representative per truth function is sufficient because formula cost
    is additive and has no sub-DAG sharing.  Minimality therefore holds within
    the declared gate library and ``max_gates``.  This is not a minimum-DAG
    circuit oracle.
    """
    if not 1 <= n_inputs <= 4:
        raise ValueError("exact formula synthesis supports one through four inputs")
    samples = 1 << n_inputs
    universe = (1 << samples) - 1
    best: dict[int, Formula] = {}
    by_cost: list[dict[int, Formula]] = [dict() for _ in range(max_gates + 1)]
    for index, mask in enumerate(input_truth_masks(n_inputs)):
        formula = Formula("INPUT", mask, 0, f"x{index}", source_index=index)
        best.setdefault(mask, formula)
        by_cost[0].setdefault(mask, formula)
    for value, text, source in ((0, "0", n_inputs), (universe, "1", n_inputs + 1)):
        formula = Formula("CONST", value, 0, text, source_index=source)
        best.setdefault(value, formula)
        by_cost[0].setdefault(value, formula)
    for cost in range(1, max_gates + 1):
        current: dict[int, Formula] = {}
        for left_cost in range((cost - 1) // 2 + 1):
            right_cost = cost - 1 - left_cost
            left_items = list(by_cost[left_cost].items())
            right_items = list(by_cost[right_cost].items())
            for left_mask, left in left_items:
                for right_mask, right in right_items:
                    if left_cost == right_cost and right_mask < left_mask:
                        continue
                    for op in GATE_OPS:
                        output = apply_gate_mask(op, left_mask, right_mask, universe)
                        if output in best or output in current:
                            continue
                        current[output] = Formula(op, output, cost, f"({left.text} {op} {right.text})", left, right)
        by_cost[cost] = current
        best.update(current)
        if len(best) == 1 << samples:
            break
    return best


@dataclass(frozen=True)
class CircuitNode:
    op: str
    left: int
    right: int


@dataclass(frozen=True)
class CircuitIR:
    n_inputs: int
    nodes: tuple[CircuitNode, ...]
    output: int

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        values = [np.asarray(x, dtype=np.uint8)[:, index] for index in range(self.n_inputs)]
        values.extend((np.zeros(len(x), dtype=np.uint8), np.ones(len(x), dtype=np.uint8)))
        for node in self.nodes:
            left, right = values[node.left], values[node.right]
            if node.op == "AND":
                output = left & right
            elif node.op == "OR":
                output = left | right
            elif node.op == "XOR":
                output = left ^ right
            elif node.op == "NAND":
                output = 1 - (left & right)
            else:
                raise ValueError(node.op)
            values.append(output)
        return values[self.output]

    @property
    def depth(self) -> int:
        depths = [0] * (self.n_inputs + 2)
        for node in self.nodes:
            depths.append(1 + max(depths[node.left], depths[node.right]))
        return depths[self.output]


def formula_to_circuit(formula: Formula, n_inputs: int, deduplicate: bool = False) -> CircuitIR:
    nodes: list[CircuitNode] = []
    memo: dict[tuple, int] = {}

    def emit(item: Formula) -> int:
        if item.op in ("INPUT", "CONST"):
            assert item.source_index is not None
            return item.source_index
        assert item.left is not None and item.right is not None
        key = (item.op, item.left.text, item.right.text)
        if deduplicate and key in memo:
            return memo[key]
        left = emit(item.left)
        right = emit(item.right)
        if right < left:
            left, right = right, left
        reference = n_inputs + 2 + len(nodes)
        nodes.append(CircuitNode(item.op, left, right))
        if deduplicate:
            memo[key] = reference
        return reference

    output = emit(formula)
    return CircuitIR(n_inputs, tuple(nodes), output)


def circuit_description_bits(circuit: CircuitIR, gate_ops: Sequence[str] = GATE_OPS) -> int:
    """Prefix length for a topologically ordered two-input netlist."""
    gates = len(circuit.nodes)
    bits = elias_delta_bits(gates + 1)
    op_bits = ceil_log2_count(len(gate_ops))
    for index, node in enumerate(circuit.nodes):
        if node.op not in gate_ops:
            raise ValueError(f"operator {node.op} is not in the public library")
        available = circuit.n_inputs + 2 + index
        parent_pairs = available * (available + 1) // 2
        bits += op_bits + ceil_log2_count(parent_pairs)
    bits += ceil_log2_count(circuit.n_inputs + 2 + gates)
    return bits


def anf_coefficients(values: np.ndarray) -> np.ndarray:
    values = _binary_array(values, "ANF truth table").reshape(-1).copy()
    n_inputs = int(round(log2(len(values))))
    if len(values) != 1 << n_inputs:
        raise ValueError("ANF expects a complete power-of-two truth table")
    for bit in range(n_inputs):
        for mask in range(len(values)):
            if mask & (1 << bit):
                values[mask] ^= values[mask ^ (1 << bit)]
    return values


def anf_description_bits(values: np.ndarray) -> int:
    coefficients = anf_coefficients(values)
    return enumerative_subset_bits(len(coefficients), int(coefficients.sum()))


def anf_statistics(values: np.ndarray) -> tuple[int, int, int]:
    coefficients = anf_coefficients(values)
    support = np.flatnonzero(coefficients)
    degrees = [int(index).bit_count() if hasattr(int(index), "bit_count") else bin(int(index)).count("1") for index in support]
    terms = len(support)
    degree = max(degrees, default=0)
    and_gates = sum(max(0, item - 1) for item in degrees)
    xor_gates = max(0, terms - 1)
    return terms, degree, and_gates + xor_gates


@dataclass(frozen=True)
class BDDNode:
    variable: int
    low: int
    high: int


@dataclass(frozen=True)
class ROBDD:
    n_inputs: int
    order: tuple[int, ...]
    nodes: tuple[BDDNode, ...]
    root: int

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.uint8)
        output = np.empty(len(x), dtype=np.uint8)
        for row_index, row in enumerate(x):
            reference = self.root
            while reference >= 2:
                node = self.nodes[reference - 2]
                reference = node.high if row[node.variable] else node.low
            output[row_index] = reference
        return output


def build_robdd(values: np.ndarray, order: Sequence[int]) -> ROBDD:
    values = _binary_array(values, "ROBDD truth table").reshape(-1)
    n_inputs = int(round(log2(len(values))))
    if len(values) != 1 << n_inputs or sorted(order) != list(range(n_inputs)):
        raise ValueError("invalid truth table or BDD order")
    indices = np.arange(len(values), dtype=np.uint64)
    nodes: list[BDDNode] = []
    unique: dict[tuple[int, int, int], int] = {}
    memo: dict[tuple[int, bytes], int] = {}

    def recurse(active: np.ndarray, level: int) -> int:
        subset = values[active]
        if np.all(subset == 0):
            return 0
        if np.all(subset == 1):
            return 1
        key = (level, subset.tobytes())
        if key in memo:
            return memo[key]
        variable = int(order[level])
        low = recurse(active[((active >> variable) & 1) == 0], level + 1)
        high = recurse(active[((active >> variable) & 1) == 1], level + 1)
        if low == high:
            memo[key] = low
            return low
        node_key = (variable, low, high)
        if node_key not in unique:
            unique[node_key] = 2 + len(nodes)
            nodes.append(BDDNode(*node_key))
        memo[key] = unique[node_key]
        return memo[key]

    root = recurse(indices, 0)
    return ROBDD(n_inputs, tuple(order), tuple(nodes), root)


def robdd_description_bits(bdd: ROBDD, encode_order: bool = True) -> int:
    bits = elias_delta_bits(len(bdd.nodes) + 1)
    if encode_order:
        bits += ceil_log2_count(factorial(bdd.n_inputs))
    variable_bits = ceil_log2_count(bdd.n_inputs)
    for index, _ in enumerate(bdd.nodes):
        available = 2 + index
        bits += variable_bits + 2 * ceil_log2_count(available)
    bits += ceil_log2_count(2 + len(bdd.nodes))
    return bits


def best_robdd(values: np.ndarray) -> tuple[ROBDD, int]:
    n_inputs = int(round(log2(len(values))))
    best: tuple[int, ROBDD] | None = None
    for order in permutations(range(n_inputs)):
        bdd = build_robdd(values, order)
        bits = robdd_description_bits(bdd, encode_order=True)
        if best is None or (bits, len(bdd.nodes), order) < (best[0], len(best[1].nodes), best[1].order):
            best = (bits, bdd)
    assert best is not None
    return best[1], best[0]


@dataclass(frozen=True)
class ThresholdModel:
    weights: tuple[int, ...]
    threshold: int
    description_bits: int
    mask: int


def threshold_library(n_inputs: int, max_abs_weight: int = 3) -> dict[int, ThresholdModel]:
    x = ((np.arange(1 << n_inputs, dtype=np.uint64)[:, None] >> np.arange(n_inputs, dtype=np.uint64)) & 1).astype(int)
    models: dict[int, ThresholdModel] = {}
    for weights in product(range(-max_abs_weight, max_abs_weight + 1), repeat=n_inputs):
        scores = x @ np.asarray(weights, dtype=int)
        for threshold in range(int(scores.min()), int(scores.max()) + 2):
            prediction = (scores >= threshold).astype(np.uint8)
            mask = values_to_mask(prediction)
            bits = sum(signed_integer_bits(value) for value in weights) + signed_integer_bits(threshold)
            candidate = ThresholdModel(tuple(weights), threshold, bits, mask)
            prior = models.get(mask)
            if prior is None or (bits, sum(abs(v) for v in weights), weights, threshold) < (
                prior.description_bits,
                sum(abs(v) for v in prior.weights),
                prior.weights,
                prior.threshold,
            ):
                models[mask] = candidate
    return models


@dataclass(frozen=True)
class MDLCandidate:
    language: str
    model_bits: int
    residual_bits: int
    total_bits: int
    errors: int
    exact: bool
    detail: str


def _best_literal(values: np.ndarray, n_inputs: int) -> tuple[np.ndarray, int, str]:
    x = ((np.arange(1 << n_inputs, dtype=np.uint64)[:, None] >> np.arange(n_inputs, dtype=np.uint64)) & 1).astype(np.uint8)
    best = None
    model_bits = ceil_log2_count(n_inputs) + 1
    for index in range(n_inputs):
        for inverted in (0, 1):
            prediction = x[:, index] ^ inverted
            residual = residual_code_bits(values, prediction)
            candidate = (model_bits + residual, prediction, f"{'~' if inverted else ''}x{index}")
            if best is None or candidate[0] < best[0]:
                best = candidate
    assert best is not None
    return best[1], model_bits, best[2]


def _best_single_gate(values: np.ndarray, n_inputs: int) -> tuple[np.ndarray, int, str]:
    x = ((np.arange(1 << n_inputs, dtype=np.uint64)[:, None] >> np.arange(n_inputs, dtype=np.uint64)) & 1).astype(np.uint8)
    pairs = list(combinations_with_replacement(range(n_inputs), 2))
    model_bits = ceil_log2_count(len(pairs)) + ceil_log2_count(len(GATE_OPS))
    best = None
    for left, right in pairs:
        for op in GATE_OPS:
            a, b = x[:, left], x[:, right]
            if op == "AND":
                prediction = a & b
            elif op == "OR":
                prediction = a | b
            elif op == "XOR":
                prediction = a ^ b
            else:
                prediction = 1 - (a & b)
            residual = residual_code_bits(values, prediction)
            candidate = (model_bits + residual, prediction, f"x{left} {op} x{right}")
            if best is None or candidate[0] < best[0]:
                best = candidate
    assert best is not None
    return best[1], model_bits, best[2]


def task_mdl_candidates(
    values: np.ndarray,
    n_inputs: int,
    *,
    formula_library: dict[int, Formula] | None = None,
    thresholds: dict[int, ThresholdModel] | None = None,
) -> list[MDLCandidate]:
    """Build a decodable multi-language two-part MDL comparison."""
    values = _binary_array(values, "task truth table").reshape(-1)
    if len(values) != 1 << n_inputs:
        raise ValueError("values must be a complete truth table")
    target_mask = values_to_mask(values)
    languages = ("RawLabels", "KTLabels", "Constant", "Literal", "SingleGate", "TruthTable", "ANF", "ROBDD", "Threshold", "ExactFormula")
    header = ceil_log2_count(len(languages))
    candidates: list[MDLCandidate] = []

    def append(language: str, model_bits: int, prediction: np.ndarray, detail: str) -> None:
        residual = residual_code_bits(values, prediction)
        errors = int(np.sum(values != prediction))
        candidates.append(MDLCandidate(language, model_bits, residual, header + model_bits + residual, errors, errors == 0, detail))

    candidates.append(MDLCandidate("RawLabels", 0, len(values), header + len(values), 0, True, "literal truth-table labels"))
    kt_bits = kt_binary_prefix_bits(values)
    candidates.append(MDLCandidate("KTLabels", 0, kt_bits, header + kt_bits, 0, True, "KT mixture over labels"))
    for constant in (0, 1):
        prediction = np.full(len(values), constant, dtype=np.uint8)
        append("Constant", 1, prediction, str(constant))
    literal_prediction, literal_bits, literal_detail = _best_literal(values, n_inputs)
    append("Literal", literal_bits, literal_prediction, literal_detail)
    gate_prediction, gate_bits, gate_detail = _best_single_gate(values, n_inputs)
    append("SingleGate", gate_bits, gate_prediction, gate_detail)
    append("TruthTable", truth_table_bits(n_inputs), values, "exact raw truth table")
    append("ANF", anf_description_bits(values), values, f"{anf_statistics(values)[0]} nonzero coefficients")
    bdd, bdd_bits = best_robdd(values)
    append("ROBDD", bdd_bits, values, f"{len(bdd.nodes)} nodes; order={bdd.order}")
    if thresholds is not None and target_mask in thresholds:
        model = thresholds[target_mask]
        append("Threshold", model.description_bits, values, f"w={model.weights}; t={model.threshold}")
    if formula_library is not None and target_mask in formula_library:
        formula = formula_library[target_mask]
        circuit = formula_to_circuit(formula, n_inputs, deduplicate=False)
        append("ExactFormula", circuit_description_bits(circuit), values, f"{formula.gates} minimum formula gates; {formula.text}")
    return candidates


def route_task_mdl(candidates: Sequence[MDLCandidate]) -> tuple[MDLCandidate, int, int]:
    """Return shortest route, raw-label baseline, and nonnegative gain."""
    if not candidates:
        raise ValueError("at least one candidate is required")
    best = min(candidates, key=lambda item: (item.total_bits, item.model_bits, item.language))
    raw = next(item for item in candidates if item.language == "RawLabels")
    return best, raw.total_bits, raw.total_bits - best.total_bits


def exact_function_language_lengths(
    values: np.ndarray,
    n_inputs: int,
    *,
    formula_library: dict[int, Formula] | None = None,
    thresholds: dict[int, ThresholdModel] | None = None,
) -> dict[str, int]:
    """Prefix lengths for exact function descriptions under several languages."""
    values = _binary_array(values, "function truth table").reshape(-1)
    target = values_to_mask(values)
    language_count = 6
    header = ceil_log2_count(language_count)
    lengths = {
        "TruthTable": header + truth_table_bits(n_inputs),
        "Onset": header + enumerative_subset_bits(len(values), int(values.sum())),
        "ANF": header + anf_description_bits(values),
    }
    bdd, bdd_bits = best_robdd(values)
    lengths["ROBDD"] = header + bdd_bits
    if thresholds is not None and target in thresholds:
        lengths["Threshold"] = header + thresholds[target].description_bits
    if formula_library is not None and target in formula_library:
        circuit = formula_to_circuit(formula_library[target], n_inputs, deduplicate=False)
        lengths["ExactFormula"] = header + circuit_description_bits(circuit)
    return lengths


def routed_function_description(values: np.ndarray, n_inputs: int, **kwargs) -> tuple[str, int, dict[str, int]]:
    lengths = exact_function_language_lengths(values, n_inputs, **kwargs)
    language = min(lengths, key=lambda item: (lengths[item], item))
    return language, lengths[language], lengths


def counting_incompressibility_probability(saved_bits: int) -> float:
    """Upper bound P[L(f) <= 2^n-saved_bits] for a uniform Boolean function."""
    if saved_bits < 0:
        raise ValueError("saved_bits must be nonnegative")
    return 2.0 ** (-saved_bits)
