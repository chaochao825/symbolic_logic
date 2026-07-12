"""Core models and exact Boolean tasks for the symbolic-logic experiments.

The code intentionally stays dependency-light.  It does not claim to reproduce
the training system of published differentiable logic-gate networks.  Instead,
it offers a transparent gate-DAG synthesizer, an equally small float MLP, and
exact/soft reference evaluators for controlled tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np


PREDICATE_NAMES = (
    "near",
    "same_color",
    "aligned",
    "same_shape",
    "src_active",
    "dst_active",
    "blocked",
    "reserved",
)


def all_assignments(n_bits: int = 8) -> np.ndarray:
    """Enumerate every binary assignment in little-endian predicate order."""
    if not 1 <= n_bits <= 20:
        raise ValueError("all_assignments supports 1 through 20 bits to bound memory use")
    codes = np.arange(1 << n_bits, dtype=np.uint64)[:, None]
    shifts = np.arange(n_bits, dtype=np.uint64)[None, :]
    return ((codes >> shifts) & 1).astype(np.uint8)


def compositional_rule(x: np.ndarray) -> np.ndarray:
    """A compact local predicate with AND/OR/XOR/NAND-like structure.

    valid = ((near & same_color) | (aligned & same_shape) |
             (src_active XOR dst_active)) & NOT(blocked & reserved)
    """
    x = np.asarray(x, dtype=np.uint8)
    body = (x[..., 0] & x[..., 1]) | (x[..., 2] & x[..., 3]) | (x[..., 4] ^ x[..., 5])
    return (body & ~(x[..., 6] & x[..., 7]) & 1).astype(np.uint8)


def xor_heavy_rule(x: np.ndarray) -> np.ndarray:
    """A parity-heavy rule that tests whether an XOR-capable library matters."""
    x = np.asarray(x, dtype=np.uint8)
    return (x[..., 0] ^ x[..., 1] ^ x[..., 2] ^ (x[..., 3] & x[..., 4]) ^ x[..., 5]).astype(np.uint8)


def parity_rule(x: np.ndarray) -> np.ndarray:
    """Parity over all input bits; circuit size grows with input width."""
    x = np.asarray(x, dtype=np.uint8)
    return np.bitwise_xor.reduce(x, axis=-1).astype(np.uint8)


def soft_parity_probability(probabilities: np.ndarray) -> np.ndarray:
    """Exact probability that independent Bernoulli bits have odd parity."""
    p = np.asarray(probabilities, dtype=np.float64)
    return 0.5 * (1.0 - np.prod(1.0 - 2.0 * p, axis=-1))


def random_lut_rule(x: np.ndarray, seed: int = 1701) -> np.ndarray:
    """A deterministic balanced random truth table: a non-compressible control."""
    x = np.asarray(x, dtype=np.uint8)
    n_bits = x.shape[-1]
    table_rng = np.random.default_rng(seed)
    table = np.zeros(1 << n_bits, dtype=np.uint8)
    table[: len(table) // 2] = 1
    table_rng.shuffle(table)
    codes = np.sum(x.astype(np.uint64) << np.arange(n_bits, dtype=np.uint64), axis=-1)
    return table[codes]


RULES: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "compositional": compositional_rule,
    "xor_heavy": xor_heavy_rule,
    "random_lut": random_lut_rule,
}


def hamming_weight(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.uint8).sum(axis=1)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.uint8)
    y_pred = np.asarray(y_pred, dtype=np.uint8)
    return float(np.mean(y_true == y_pred))


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.uint8)
    y_pred = np.asarray(y_pred, dtype=np.uint8)
    positives = y_true == 1
    negatives = ~positives
    tpr = np.mean(y_pred[positives] == 1) if positives.any() else 1.0
    tnr = np.mean(y_pred[negatives] == 0) if negatives.any() else 1.0
    return float((tpr + tnr) / 2)


def brier_score(y_true: np.ndarray, probability: np.ndarray) -> float:
    return float(np.mean((np.asarray(probability, dtype=float) - np.asarray(y_true, dtype=float)) ** 2))


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return class-sensitive binary metrics without external dependencies."""
    y_true = np.asarray(y_true, dtype=np.uint8)
    y_pred = np.asarray(y_pred, dtype=np.uint8)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tpr = tp / (tp + fn) if (tp + fn) else 1.0
    tnr = tn / (tn + fp) if (tn + fp) else 1.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    denominator = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn - fp * fn) / np.sqrt(denominator)) if denominator else 0.0
    return {"tpr": float(tpr), "tnr": float(tnr), "f1": float(f1), "mcc": float(mcc)}


@dataclass(frozen=True)
class Expr:
    """Immutable Boolean expression used by the beam synthesizer."""

    op: str
    text: str
    mask: int
    gates: int
    depth: int
    left: "Expr | None" = None
    right: "Expr | None" = None
    literal_index: int | None = None
    literal_negated: bool = False

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.uint8)
        if self.op == "LITERAL":
            out = x[:, self.literal_index]  # type: ignore[index]
            return (1 - out) if self.literal_negated else out
        if self.op == "CONST":
            return np.full(len(x), int(self.text), dtype=np.uint8)
        assert self.left is not None and self.right is not None
        a = self.left.evaluate(x)
        b = self.right.evaluate(x)
        if self.op == "AND":
            return a & b
        if self.op == "OR":
            return a | b
        if self.op == "XOR":
            return a ^ b
        if self.op == "NAND":
            return 1 - (a & b)
        raise ValueError(f"unknown operation: {self.op}")


def _vector_to_mask(values: Iterable[int]) -> int:
    result = 0
    for index, value in enumerate(values):
        if int(value):
            result |= 1 << index
    return result


def _mask_accuracy(mask: int, label_mask: int, n_examples: int) -> float:
    incorrect = (mask ^ label_mask).bit_count()
    return 1.0 - incorrect / n_examples


def _apply_mask_gate(op: str, a: int, b: int, universe: int) -> int:
    if op == "AND":
        return a & b
    if op == "OR":
        return a | b
    if op == "XOR":
        return a ^ b
    if op == "NAND":
        return universe ^ (a & b)
    raise ValueError(op)


class GateBeamSynthesizer:
    """Greedy beam search for a compact Boolean expression.

    Search is performed over packed truth values, so the structure search itself
    already uses bitwise primitives.  It is a transparent rule-induction proxy,
    not a claim of differentiable-LGN equivalence.
    """

    def __init__(self, max_depth: int = 4, beam_width: int = 192, gates: tuple[str, ...] = ("AND", "OR", "XOR", "NAND")):
        self.max_depth = max_depth
        self.beam_width = beam_width
        self.gates = gates
        self.expression: Expr | None = None
        self.fit_seconds: float = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> "GateBeamSynthesizer":
        import time

        started = time.perf_counter()
        x = np.asarray(x, dtype=np.uint8)
        y = np.asarray(y, dtype=np.uint8)
        n_examples, n_bits = x.shape
        if n_examples > 4096:
            raise ValueError("beam synthesizer is intentionally for small finite truth tables")
        universe = (1 << n_examples) - 1
        label_mask = _vector_to_mask(y)
        base: dict[int, Expr] = {}
        for bit in range(n_bits):
            mask = _vector_to_mask(x[:, bit])
            positive = Expr("LITERAL", f"x{bit}", mask, 0, 0, literal_index=bit)
            negative = Expr("LITERAL", f"~x{bit}", universe ^ mask, 0, 0, literal_index=bit, literal_negated=True)
            base.setdefault(mask, positive)
            base.setdefault(universe ^ mask, negative)
        base.setdefault(0, Expr("CONST", "0", 0, 0, 0))
        base.setdefault(universe, Expr("CONST", "1", universe, 0, 0))

        best = max(base.values(), key=lambda item: (_mask_accuracy(item.mask, label_mask, n_examples), -item.gates))
        all_frontiers: list[list[Expr]] = [list(base.values())]

        def rank(item: Expr) -> tuple[float, int, int, str]:
            # Accuracy is primary; smaller/deeper structures lose deterministic ties.
            return (_mask_accuracy(item.mask, label_mask, n_examples), -item.gates, -item.depth, item.text)

        for depth in range(1, self.max_depth + 1):
            sources: dict[int, Expr] = dict(base)
            for frontier in all_frontiers[1:]:
                for item in frontier:
                    current = sources.get(item.mask)
                    if current is None or (item.gates, item.text) < (current.gates, current.text):
                        sources[item.mask] = item
            ordered = sorted(sources.values(), key=rank, reverse=True)
            # Retaining base literals plus the best intermediate signals preserves
            # building blocks even when a constituent has weak standalone accuracy.
            literals = list(base.values())
            non_literals = [item for item in ordered if item.depth > 0][: self.beam_width]
            candidates_input = literals + non_literals
            candidate_by_mask: dict[int, Expr] = {}
            for left_i, left in enumerate(candidates_input):
                for right in candidates_input[left_i:]:
                    for op in self.gates:
                        mask = _apply_mask_gate(op, left.mask, right.mask, universe)
                        candidate = Expr(
                            op,
                            f"({left.text} {op} {right.text})",
                            mask,
                            left.gates + right.gates + 1,
                            depth,
                            left,
                            right,
                        )
                        prior = candidate_by_mask.get(mask)
                        if prior is None or (candidate.gates, candidate.text) < (prior.gates, prior.text):
                            candidate_by_mask[mask] = candidate
            frontier = sorted(candidate_by_mask.values(), key=rank, reverse=True)[: self.beam_width]
            all_frontiers.append(frontier)
            if frontier and rank(frontier[0]) > rank(best):
                best = frontier[0]
            if _mask_accuracy(best.mask, label_mask, n_examples) == 1.0:
                break
        self.expression = best
        self.fit_seconds = time.perf_counter() - started
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.expression is None:
            raise RuntimeError("fit must be called before predict")
        return self.expression.evaluate(x).astype(np.uint8)


class TruthMemorizer:
    """Lookup baseline with no compositional inductive bias."""

    def __init__(self) -> None:
        self.table: dict[tuple[int, ...], int] = {}
        self.prior = 0

    def fit(self, x: np.ndarray, y: np.ndarray) -> "TruthMemorizer":
        x = np.asarray(x, dtype=np.uint8)
        y = np.asarray(y, dtype=np.uint8)
        self.prior = int(np.mean(y) >= 0.5)
        self.table = {tuple(row.tolist()): int(label) for row, label in zip(x, y)}
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray([self.table.get(tuple(row.tolist()), self.prior) for row in np.asarray(x, dtype=np.uint8)], dtype=np.uint8)


class TinyMLP:
    """A tiny float baseline trained with full-batch Adam and binary cross entropy."""

    def __init__(self, n_inputs: int, hidden: tuple[int, ...] = (24, 16), seed: int = 0, lr: float = 0.02, steps: int = 1400):
        self.n_inputs = n_inputs
        self.hidden = hidden
        self.seed = seed
        self.lr = lr
        self.steps = steps
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        self.fit_seconds = 0.0

    def _initialize(self) -> None:
        rng = np.random.default_rng(self.seed)
        widths = (self.n_inputs,) + self.hidden + (1,)
        self.weights = [rng.normal(0.0, np.sqrt(2 / widths[i]), size=(widths[i], widths[i + 1])).astype(np.float64) for i in range(len(widths) - 1)]
        self.biases = [np.zeros((1, widths[i + 1]), dtype=np.float64) for i in range(len(widths) - 1)]

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))

    def fit(self, x: np.ndarray, y: np.ndarray) -> "TinyMLP":
        import time

        started = time.perf_counter()
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        self._initialize()
        m_w = [np.zeros_like(w) for w in self.weights]
        v_w = [np.zeros_like(w) for w in self.weights]
        m_b = [np.zeros_like(b) for b in self.biases]
        v_b = [np.zeros_like(b) for b in self.biases]
        beta1, beta2 = 0.9, 0.999
        n = len(x)
        for step in range(1, self.steps + 1):
            activations = [x]
            preactivations: list[np.ndarray] = []
            current = x
            for layer, (w, b) in enumerate(zip(self.weights, self.biases)):
                z = current @ w + b
                preactivations.append(z)
                current = self._sigmoid(z) if layer == len(self.weights) - 1 else np.tanh(z)
                activations.append(current)
            delta = (activations[-1] - y) / n
            grad_w: list[np.ndarray] = [np.empty_like(w) for w in self.weights]
            grad_b: list[np.ndarray] = [np.empty_like(b) for b in self.biases]
            for layer in range(len(self.weights) - 1, -1, -1):
                grad_w[layer] = activations[layer].T @ delta
                grad_b[layer] = delta.sum(axis=0, keepdims=True)
                if layer:
                    delta = (delta @ self.weights[layer].T) * (1.0 - np.tanh(preactivations[layer - 1]) ** 2)
            for index in range(len(self.weights)):
                m_w[index] = beta1 * m_w[index] + (1 - beta1) * grad_w[index]
                v_w[index] = beta2 * v_w[index] + (1 - beta2) * (grad_w[index] ** 2)
                m_b[index] = beta1 * m_b[index] + (1 - beta1) * grad_b[index]
                v_b[index] = beta2 * v_b[index] + (1 - beta2) * (grad_b[index] ** 2)
                m_w_hat = m_w[index] / (1 - beta1**step)
                v_w_hat = v_w[index] / (1 - beta2**step)
                m_b_hat = m_b[index] / (1 - beta1**step)
                v_b_hat = v_b[index] / (1 - beta2**step)
                self.weights[index] -= self.lr * m_w_hat / (np.sqrt(v_w_hat) + 1e-8)
                self.biases[index] -= self.lr * m_b_hat / (np.sqrt(v_b_hat) + 1e-8)
        self.fit_seconds = time.perf_counter() - started
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        current = np.asarray(x, dtype=np.float64)
        for layer, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = current @ w + b
            current = self._sigmoid(z) if layer == len(self.weights) - 1 else np.tanh(z)
        return current[:, 0]

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.predict_proba(x) >= 0.5).astype(np.uint8)


class SoftGateCircuit:
    """A small differentiable gate DAG used for the soft-to-hard audit.

    The wiring is intentionally fixed so that the experiment isolates the
    *operator learning and hardening* question rather than conflating it with
    architecture search.  Each node is a softmax mixture of AND/OR/XOR/NAND
    fuzzy operators.  ``predict_hardened`` compiles every mixture to its
    highest-probability Boolean operator and evaluates the resulting DAG with
    crisp inputs.

    This is a controlled proxy for a learned soft-LGN, not a reproduction of a
    published LGN implementation.  The explicit distinction is recorded in
    the report and keeps the end-to-end result falsifiable.
    """

    OP_NAMES = ("AND", "OR", "XOR", "NAND")
    # The final node implements the compositional rule when the target
    # operators are AND, AND, XOR, OR, OR, NAND, AND respectively.
    PAIRS = ((0, 1), (2, 3), (4, 5), (8, 9), (11, 10), (6, 7), (12, 13))

    def __init__(self, seed: int = 0, lr: float = 0.08, steps: int = 1600) -> None:
        self.seed = seed
        self.lr = lr
        self.steps = steps
        self.logits = np.zeros((len(self.PAIRS), len(self.OP_NAMES)), dtype=np.float64)
        self.fit_seconds = 0.0
        self.loss = float("nan")

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exp = np.exp(shifted)
        return exp / np.sum(exp, axis=-1, keepdims=True)

    @staticmethod
    def _operator_values(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        values = np.stack((a * b, a + b - a * b, a + b - 2 * a * b, 1.0 - a * b), axis=-1)
        da = np.stack((b, 1.0 - b, 1.0 - 2.0 * b, -b), axis=-1)
        db = np.stack((a, 1.0 - a, 1.0 - 2.0 * a, -a), axis=-1)
        return values, da, db

    def _forward(self, x: np.ndarray, with_cache: bool = False):
        values: list[np.ndarray] = [np.asarray(x, dtype=np.float64)[:, i] for i in range(8)]
        caches: list[tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        probabilities = self._softmax(self.logits)
        for node, (left_index, right_index) in enumerate(self.PAIRS):
            a, b = values[left_index], values[right_index]
            op_values, da, db = self._operator_values(a, b)
            pi = probabilities[node]
            output = op_values @ pi
            values.append(output)
            if with_cache:
                caches.append((left_index, right_index, op_values, da, db, pi))
        return (values[-1], values, caches) if with_cache else values[-1]

    def fit(self, x: np.ndarray, y: np.ndarray) -> "SoftGateCircuit":
        import time

        started = time.perf_counter()
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        rng = np.random.default_rng(self.seed)
        self.logits = rng.normal(0.0, 0.04, size=self.logits.shape)
        m = np.zeros_like(self.logits)
        v = np.zeros_like(self.logits)
        beta1, beta2 = 0.9, 0.999
        n = max(1, len(x))
        for step in range(1, self.steps + 1):
            output, values, caches = self._forward(x, with_cache=True)
            # MSE keeps the proxy stable when hard Boolean inputs make a
            # fuzzy operator output exactly 0 or 1 at initialization.
            grad_values = [np.zeros_like(item) for item in values]
            grad_values[-1] = (output - y) / n
            grad_logits = np.zeros_like(self.logits)
            for node in range(len(self.PAIRS) - 1, -1, -1):
                left_index, right_index, op_values, da, db, pi = caches[node]
                g = grad_values[node + 8]
                mixed = op_values @ pi
                grad_logits[node] = np.sum(g[:, None] * pi[None, :] * (op_values - mixed[:, None]), axis=0)
                grad_values[left_index] += g * (da @ pi)
                grad_values[right_index] += g * (db @ pi)
            m = beta1 * m + (1.0 - beta1) * grad_logits
            v = beta2 * v + (1.0 - beta2) * (grad_logits * grad_logits)
            m_hat = m / (1.0 - beta1**step)
            v_hat = v / (1.0 - beta2**step)
            self.logits -= self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)
        output = self.predict_proba(x)
        self.loss = float(np.mean((output - y) ** 2))
        self.fit_seconds = time.perf_counter() - started
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return np.clip(self._forward(np.asarray(x, dtype=np.float64)), 0.0, 1.0)

    def selected_ops(self) -> tuple[str, ...]:
        return tuple(self.OP_NAMES[int(index)] for index in np.argmax(self.logits, axis=1))

    @staticmethod
    def _apply_boolean(op: str, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if op == "AND":
            return a & b
        if op == "OR":
            return a | b
        if op == "XOR":
            return a ^ b
        if op == "NAND":
            return 1 - (a & b)
        raise ValueError(op)

    def predict_hardened(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the argmax-compiled Boolean DAG on thresholded inputs."""
        source = (np.asarray(x, dtype=np.float64) >= 0.5).astype(np.uint8)
        values: list[np.ndarray] = [source[:, i] for i in range(8)]
        for op, (left_index, right_index) in zip(self.selected_ops(), self.PAIRS):
            values.append(self._apply_boolean(op, values[left_index], values[right_index]))
        return values[-1].astype(np.uint8)


def soft_probability_rule(probabilities: np.ndarray) -> np.ndarray:
    """Exact Bernoulli probability for the compositional rule under independence."""
    p = np.asarray(probabilities, dtype=np.float64)
    term_a = p[..., 0] * p[..., 1]
    term_b = p[..., 2] * p[..., 3]
    term_c = p[..., 4] * (1 - p[..., 5]) + (1 - p[..., 4]) * p[..., 5]
    body = 1 - (1 - term_a) * (1 - term_b) * (1 - term_c)
    return body * (1 - p[..., 6] * p[..., 7])


def packed_compositional_rule(packed_columns: list[np.ndarray]) -> np.ndarray:
    """Evaluate the local rule over packed byte lanes, with no unpacked math."""
    if len(packed_columns) != 8:
        raise ValueError("expected eight packed predicate columns")
    x = packed_columns
    body = (x[0] & x[1]) | (x[2] & x[3]) | (x[4] ^ x[5])
    return body & np.bitwise_not(x[6] & x[7])
