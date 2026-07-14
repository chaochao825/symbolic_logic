"""Controlled logic-discovery and coding-rate-reduction experiments.

The module separates three questions that are often conflated:

1. Can a learner recover both the variables and the Boolean operator?
2. Can local rules/topology be identified using only graph-level supervision?
3. Does the MCR2 log-det coding-rate proxy predict literal bit cost or gate count?

This is a dependency-light NumPy reproduction of the *core objective and
gradient flow* behind MCR2/ReduNet, not a reproduction of the full ReduNet or
CRATE training systems.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import ceil, log2
from typing import Iterable

import numpy as np

from logic_core import GateBeamSynthesizer, TinyMLP, accuracy, all_assignments, balanced_accuracy


OPS = ("AND", "OR", "XOR", "NAND")


def apply_gate(op: str, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Apply a crisp Boolean gate to equally-shaped arrays."""
    a = np.asarray(a, dtype=np.uint8)
    b = np.asarray(b, dtype=np.uint8)
    if op == "AND":
        return a & b
    if op == "OR":
        return a | b
    if op == "XOR":
        return a ^ b
    if op == "NAND":
        return 1 - (a & b)
    raise ValueError(op)


def fuzzy_gate(op: str, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Standard multilinear fuzzy extension of the four gate primitives."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if op == "AND":
        return a * b
    if op == "OR":
        return a + b - a * b
    if op == "XOR":
        return a + b - 2.0 * a * b
    if op == "NAND":
        return 1.0 - a * b
    raise ValueError(op)


@dataclass(frozen=True)
class GateHypothesis:
    left: int
    right: int
    op: str
    train_balanced_accuracy: float
    runner_up_margin: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        return apply_gate(self.op, x[:, self.left], x[:, self.right])

    @property
    def description_bits(self) -> int:
        n_pairs = max(1, self.right + 1)
        # The runner does not use this property for accounting because the full
        # input width is needed.  It is kept as a compact self-description.
        return 2 + ceil(log2(n_pairs))


def fit_gate_hypothesis(x: np.ndarray, y: np.ndarray) -> GateHypothesis:
    """Exhaustively select two inputs and one gate from a finite library."""
    x = np.asarray(x, dtype=np.uint8)
    y = np.asarray(y, dtype=np.uint8)
    candidates: list[tuple[float, int, int, str]] = []
    for left, right in combinations(range(x.shape[1]), 2):
        for op in OPS:
            score = balanced_accuracy(y, apply_gate(op, x[:, left], x[:, right]))
            candidates.append((score, left, right, op))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2], OPS.index(item[3])))
    best = candidates[0]
    second = candidates[1][0]
    return GateHypothesis(best[1], best[2], best[3], best[0], best[0] - second)


class DifferentiableGateSelector:
    """Softmax mixture over every (input pair, Boolean operator) hypothesis.

    This extends the earlier operator-only learned gate: both variable binding
    and operator choice receive gradients.  Hardening selects one finite
    hypothesis.  The candidate topology is still enumerated, so this is not an
    unrestricted wiring-search network.
    """

    def __init__(self, n_inputs: int, seed: int = 0, steps: int = 900, lr: float = 0.08) -> None:
        self.n_inputs = n_inputs
        self.seed = seed
        self.steps = steps
        self.lr = lr
        self.specs = [(left, right, op) for left, right in combinations(range(n_inputs), 2) for op in OPS]
        self.logits = np.zeros(len(self.specs), dtype=np.float64)
        self.fit_seconds = 0.0

    @staticmethod
    def _softmax(values: np.ndarray) -> np.ndarray:
        shifted = values - values.max()
        exp = np.exp(shifted)
        return exp / exp.sum()

    def _candidate_matrix(self, x: np.ndarray) -> np.ndarray:
        return np.stack([apply_gate(op, x[:, left], x[:, right]) for left, right, op in self.specs], axis=1).astype(np.float64)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "DifferentiableGateSelector":
        import time

        started = time.perf_counter()
        candidates = self._candidate_matrix(np.asarray(x, dtype=np.uint8))
        target = np.asarray(y, dtype=np.float64)
        rng = np.random.default_rng(self.seed)
        self.logits = rng.normal(0.0, 0.02, size=len(self.specs))
        first = np.zeros_like(self.logits)
        second = np.zeros_like(self.logits)
        beta1, beta2 = 0.9, 0.999
        for step in range(1, self.steps + 1):
            weights = self._softmax(self.logits)
            output = candidates @ weights
            grad_weights = candidates.T @ ((output - target) / max(1, len(target)))
            grad = weights * (grad_weights - np.dot(weights, grad_weights))
            first = beta1 * first + (1.0 - beta1) * grad
            second = beta2 * second + (1.0 - beta2) * grad * grad
            corrected_first = first / (1.0 - beta1**step)
            corrected_second = second / (1.0 - beta2**step)
            self.logits -= self.lr * corrected_first / (np.sqrt(corrected_second) + 1e-8)
        self.fit_seconds = time.perf_counter() - started
        return self

    def weights(self) -> np.ndarray:
        return self._softmax(self.logits)

    def selected(self) -> tuple[int, int, str]:
        return self.specs[int(np.argmax(self.logits))]

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self._candidate_matrix(np.asarray(x, dtype=np.uint8)) @ self.weights()

    def predict(self, x: np.ndarray) -> np.ndarray:
        left, right, op = self.selected()
        return apply_gate(op, np.asarray(x)[:, left], np.asarray(x)[:, right])


def gate_description_bits(n_inputs: int) -> int:
    """Fixed-length index cost for one unordered input pair plus one of 4 ops."""
    return ceil(log2(max(1, n_inputs * (n_inputs - 1) // 2))) + 2


def _bfs(n_nodes: int, edges: np.ndarray, valid: np.ndarray, source: int = 0, target: int | None = None) -> bool:
    target = n_nodes - 1 if target is None else target
    adjacency = [[] for _ in range(n_nodes)]
    for (left, right), enabled in zip(edges, valid):
        if enabled:
            adjacency[int(left)].append(int(right))
            adjacency[int(right)].append(int(left))
    queue = [source]
    visited = {source}
    while queue:
        node = queue.pop(0)
        if node == target:
            return True
        for neighbor in adjacency[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return False


@dataclass
class GraphExample:
    n_nodes: int
    edges: np.ndarray
    edge_features: np.ndarray
    relation_types: np.ndarray
    label: int


def make_graph_dataset(
    rng: np.random.Generator,
    samples: int,
    n_bits: int,
    true_pair: tuple[int, int],
    true_op: str,
    true_relation_mask: int,
    n_relation_types: int = 4,
) -> list[GraphExample]:
    """Generate balanced task-only reachability observations.

    Relation types are an explicit finite topology vocabulary.  Learning its
    mask is weaker than discovering arbitrary object-to-object adjacency, but
    stronger than fixing the four-neighbor graph in advance.
    """
    positives: list[GraphExample] = []
    negatives: list[GraphExample] = []
    attempts = 0
    target_per_class = samples // 2
    while min(len(positives), len(negatives)) < target_per_class and attempts < samples * 200:
        attempts += 1
        n_nodes = int(rng.integers(7, 11))
        all_edges = np.asarray(list(combinations(range(n_nodes), 2)), dtype=np.int16)
        chosen = rng.choice(len(all_edges), size=min(len(all_edges), int(rng.integers(n_nodes + 3, n_nodes * 3))), replace=False)
        edges = all_edges[chosen]
        features = rng.integers(0, 2, size=(len(edges), n_bits), dtype=np.uint8)
        relations = rng.integers(0, n_relation_types, size=len(edges), dtype=np.uint8)
        relation_valid = ((true_relation_mask >> relations) & 1).astype(np.uint8)
        local_valid = apply_gate(true_op, features[:, true_pair[0]], features[:, true_pair[1]])
        label = int(_bfs(n_nodes, edges, relation_valid & local_valid))
        example = GraphExample(n_nodes, edges, features, relations, label)
        bucket = positives if label else negatives
        if len(bucket) < target_per_class:
            bucket.append(example)
    if min(len(positives), len(negatives)) < target_per_class:
        raise RuntimeError("could not generate a balanced reachability dataset")
    result = positives + negatives
    rng.shuffle(result)
    return result


def predict_graphs(examples: Iterable[GraphExample], pair: tuple[int, int], op: str, relation_mask: int) -> np.ndarray:
    predictions = []
    for example in examples:
        relation_valid = ((relation_mask >> example.relation_types) & 1).astype(np.uint8)
        local = apply_gate(op, example.edge_features[:, pair[0]], example.edge_features[:, pair[1]])
        predictions.append(int(_bfs(example.n_nodes, example.edges, relation_valid & local)))
    return np.asarray(predictions, dtype=np.uint8)


def fit_task_supervised_graph_rule(examples: list[GraphExample], n_bits: int, n_relation_types: int = 4) -> tuple[tuple[int, int], str, int, float, float]:
    """Jointly search variable pair, gate, and relation-type mask from task labels."""
    labels = np.asarray([item.label for item in examples], dtype=np.uint8)
    scored: list[tuple[float, tuple[int, int], str, int]] = []
    for pair in combinations(range(n_bits), 2):
        for op in OPS:
            for relation_mask in range(1, 1 << n_relation_types):
                pred = predict_graphs(examples, pair, op, relation_mask)
                scored.append((balanced_accuracy(labels, pred), pair, op, relation_mask))
    scored.sort(key=lambda item: (-item[0], item[1], OPS.index(item[2]), item[3]))
    return scored[0][1], scored[0][2], scored[0][3], scored[0][0], scored[0][0] - scored[1][0]


def coding_rate(z: np.ndarray, epsilon: float = 0.5) -> float:
    """Finite-sample Gaussian coding-rate proxy in bits (base-2 log-det)."""
    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError("z must have shape [features, samples]")
    d, m = z.shape
    if m == 0:
        return 0.0
    matrix = np.eye(d) + (d / (m * epsilon**2)) * (z @ z.T)
    sign, logdet = np.linalg.slogdet(matrix)
    if sign <= 0:
        raise FloatingPointError("coding-rate matrix is not positive definite")
    return float(0.5 * logdet / np.log(2.0))


def mcr2(z: np.ndarray, labels: np.ndarray, epsilon: float = 0.5) -> tuple[float, float, float]:
    """Return global rate, weighted within-class rate, and their reduction."""
    z = np.asarray(z, dtype=np.float64)
    labels = np.asarray(labels)
    m = z.shape[1]
    global_rate = coding_rate(z, epsilon)
    within = 0.0
    for label in np.unique(labels):
        subset = z[:, labels == label]
        within += subset.shape[1] / m * coding_rate(subset, epsilon)
    return global_rate, within, global_rate - within


def mcr2_gradient(z: np.ndarray, labels: np.ndarray, epsilon: float = 0.5) -> np.ndarray:
    """Analytic gradient of the base-2 MCR2 objective with respect to z."""
    z = np.asarray(z, dtype=np.float64)
    labels = np.asarray(labels)
    d, m = z.shape
    alpha = d / (m * epsilon**2)
    gradient = (alpha / np.log(2.0)) * np.linalg.solve(np.eye(d) + alpha * (z @ z.T), z)
    for label in np.unique(labels):
        mask = labels == label
        subset = z[:, mask]
        alpha_class = d / (subset.shape[1] * epsilon**2)
        weighted = subset.shape[1] / m
        gradient[:, mask] -= (weighted * alpha_class / np.log(2.0)) * np.linalg.solve(
            np.eye(d) + alpha_class * (subset @ subset.T), subset
        )
    return gradient


def normalize_columns(z: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(z, axis=0, keepdims=True)
    return z / np.maximum(norm, 1e-12)


def soft_threshold(z: np.ndarray, threshold: float) -> np.ndarray:
    return np.sign(z) * np.maximum(np.abs(z) - threshold, 0.0)


def run_rate_flow(z: np.ndarray, labels: np.ndarray, layers: int = 24, step_size: float = 0.35, sparse_lambda: float = 0.0, epsilon: float = 0.5) -> np.ndarray:
    """Unroll repeated MCR2 ascent, optionally followed by an ISTA shrinkage step."""
    current = normalize_columns(np.asarray(z, dtype=np.float64))
    for _ in range(layers):
        update = current + step_size * mcr2_gradient(current, labels, epsilon)
        if sparse_lambda:
            update = soft_threshold(update, step_size * sparse_lambda)
        current = normalize_columns(update)
    return current


def empirical_code_entropy(binary_codes: np.ndarray) -> float:
    """Empirical entropy of observed codewords in bits per sample."""
    binary_codes = np.asarray(binary_codes, dtype=np.uint8).T
    _, counts = np.unique(binary_codes, axis=0, return_counts=True)
    probabilities = counts / counts.sum()
    return float(-np.sum(probabilities * np.log2(probabilities)))


def linear_reconstruction_mse(codes: np.ndarray, targets: np.ndarray, ridge: float = 1e-4) -> float:
    """Decode targets from codes with a finite linear ridge decoder."""
    x = np.asarray(codes, dtype=np.float64).T
    y = np.asarray(targets, dtype=np.float64).T
    design = np.concatenate((x, np.ones((len(x), 1))), axis=1)
    regularizer = ridge * np.eye(design.shape[1])
    weights = np.linalg.solve(design.T @ design + regularizer, design.T @ y)
    reconstructed = design @ weights
    return float(np.mean((reconstructed - y) ** 2))


def subspace_coherence(z: np.ndarray, labels: np.ndarray, rank: int = 3) -> float:
    """Largest cross-class principal correlation; lower is more orthogonal."""
    bases = []
    for label in np.unique(labels):
        u, _, _ = np.linalg.svd(z[:, labels == label], full_matrices=False)
        bases.append(u[:, : min(rank, u.shape[1])])
    values = [np.linalg.svd(a.T @ b, compute_uv=False)[0] for a, b in combinations(bases, 2)]
    return float(max(values)) if values else 0.0


def make_union_of_subspaces(rng: np.random.Generator, classes: int = 3, samples_per_class: int = 128, ambient: int = 12, intrinsic: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic finite-sample union of noisy low-dimensional subspaces."""
    shared, _ = np.linalg.qr(rng.normal(size=(ambient, intrinsic)))
    samples = []
    labels = []
    for class_index in range(classes):
        specific, _ = np.linalg.qr(rng.normal(size=(ambient, intrinsic)))
        basis, _ = np.linalg.qr(0.55 * shared + 0.85 * specific)
        coefficients = rng.normal(size=(intrinsic, samples_per_class))
        block = basis[:, :intrinsic] @ coefficients + 0.04 * rng.normal(size=(ambient, samples_per_class))
        samples.append(block)
        labels.extend([class_index] * samples_per_class)
    return normalize_columns(np.concatenate(samples, axis=1)), np.asarray(labels, dtype=np.int16)


def nearest_subspace_accuracy(train_z: np.ndarray, train_y: np.ndarray, test_z: np.ndarray, test_y: np.ndarray, rank: int = 3) -> float:
    bases = []
    classes = np.unique(train_y)
    for label in classes:
        u, _, _ = np.linalg.svd(train_z[:, train_y == label], full_matrices=False)
        bases.append(u[:, : min(rank, u.shape[1])])
    residuals = np.stack([np.sum((test_z - basis @ (basis.T @ test_z)) ** 2, axis=0) for basis in bases], axis=1)
    prediction = classes[np.argmin(residuals, axis=1)]
    return float(np.mean(prediction == test_y))


def shared_projection_attention(z: np.ndarray, bases: np.ndarray) -> np.ndarray:
    """CRATE-style subspace attention proxy reusing each U for analysis/synthesis.

    ``bases`` has shape [heads, features, head_dim].  This function exists to
    make the parameter-sharing claim executable; it is not a full CRATE block.
    """
    outputs = np.zeros_like(z, dtype=np.float64)
    scale = max(1.0, np.sqrt(bases.shape[2]))
    for basis in bases:
        projected = basis.T @ z
        logits = (projected.T @ projected) / scale
        logits -= logits.max(axis=1, keepdims=True)
        attention = np.exp(logits)
        attention /= attention.sum(axis=1, keepdims=True)
        outputs += basis @ (projected @ attention.T)
    return outputs


def run_logic_discovery(seed: int, n_bits: int = 8) -> list[dict]:
    rng = np.random.default_rng(91_000 + seed)
    x = all_assignments(n_bits)
    rows: list[dict] = []
    learned_ops: dict[str, str] = {}
    for task_index, op in enumerate(OPS):
        pair = tuple(sorted(rng.choice(n_bits, size=2, replace=False).tolist()))
        y = apply_gate(op, x[:, pair[0]], x[:, pair[1]])
        order = rng.permutation(len(x))
        train = order[: len(x) // 2]
        test = order[len(x) // 2 :]
        learned = DifferentiableGateSelector(n_bits, seed=seed * 100 + task_index).fit(x[train], y[train])
        selected_left, selected_right, selected_op = learned.selected()
        learned_ops[op] = selected_op
        fixed = x[:, 0] & x[:, 1]
        mlp = TinyMLP(n_bits, hidden=(16,), seed=seed * 10 + task_index, steps=700).fit(x[train], y[train])
        for method, prediction, fit_seconds, description_bits in (
            ("FixedAND", fixed[test], 0.0, 2),
            ("DifferentiableGateSelector", learned.predict(x[test]), learned.fit_seconds, gate_description_bits(n_bits)),
            ("TinyMLP", mlp.predict(x[test]), mlp.fit_seconds, sum(w.size for w in mlp.weights) * 32),
        ):
            rows.append(
                {
                    "suite": "multi_rule_distractors",
                    "seed": seed,
                    "task": op,
                    "method": method,
                    "test_balanced_accuracy": balanced_accuracy(y[test], prediction),
                    "operator_recovered": float(selected_op == op) if method == "DifferentiableGateSelector" else np.nan,
                    "inputs_recovered": float((selected_left, selected_right) == pair) if method == "DifferentiableGateSelector" else np.nan,
                    "topology_recovered": np.nan,
                    "rejected_hardening": np.nan,
                    "description_bits": description_bits,
                    "gate_count": 1 if method != "TinyMLP" else np.nan,
                    "fit_seconds": fit_seconds,
                    "margin": float(np.sort(learned.weights())[-1] - np.sort(learned.weights())[-2]) if method == "DifferentiableGateSelector" else np.nan,
                }
            )

    # Operator semantics are learned only from depth-1 tasks, then reused in
    # depth-2/3 expressions without fitting on composed labels.
    for depth in (2, 3):
        leaves = [x[:, index] for index in range(2**depth)]
        true_nodes = leaves
        learned_nodes = leaves
        for level in range(depth):
            next_true = []
            next_learned = []
            for node in range(0, len(true_nodes), 2):
                op = OPS[(seed + level + node // 2) % len(OPS)]
                next_true.append(apply_gate(op, true_nodes[node], true_nodes[node + 1]))
                next_learned.append(apply_gate(learned_ops[op], learned_nodes[node], learned_nodes[node + 1]))
            true_nodes, learned_nodes = next_true, next_learned
        rows.append(
            {
                "suite": "unseen_composition",
                "seed": seed,
                "task": f"depth_{depth}",
                "method": "Depth1OperatorLibrary",
                "test_balanced_accuracy": balanced_accuracy(true_nodes[0], learned_nodes[0]),
                "operator_recovered": float(all(learned_ops[name] == name for name in OPS)),
                "inputs_recovered": np.nan,
                "topology_recovered": np.nan,
                "rejected_hardening": np.nan,
                "description_bits": depth * 2,
                "gate_count": 2**depth - 1,
                "fit_seconds": 0.0,
                "margin": np.nan,
            }
        )

    # Joint rule/topology discovery from reachability labels only.
    true_pair = tuple(sorted(rng.choice(n_bits, size=2, replace=False).tolist()))
    true_op = OPS[seed % len(OPS)]
    true_mask = (1 << (seed % 4)) | (1 << ((seed + 1) % 4))
    train_graphs = make_graph_dataset(rng, 160, n_bits, true_pair, true_op, true_mask)
    test_graphs = make_graph_dataset(rng, 240, n_bits, true_pair, true_op, true_mask)
    pair, op, relation_mask, train_score, margin = fit_task_supervised_graph_rule(train_graphs, n_bits)
    test_labels = np.asarray([item.label for item in test_graphs], dtype=np.uint8)
    learned_prediction = predict_graphs(test_graphs, pair, op, relation_mask)
    fixed_prediction = predict_graphs(test_graphs, (0, 1), "AND", (1 << 4) - 1)
    for method, prediction in (("TaskOnlyRuleTopologySearch", learned_prediction), ("FixedANDDenseTopology", fixed_prediction)):
        rows.append(
            {
                "suite": "task_only_reachability",
                "seed": seed,
                "task": true_op,
                "method": method,
                "test_balanced_accuracy": balanced_accuracy(test_labels, prediction),
                "operator_recovered": float(op == true_op) if method.startswith("TaskOnly") else np.nan,
                "inputs_recovered": float(pair == true_pair) if method.startswith("TaskOnly") else np.nan,
                "topology_recovered": float(relation_mask == true_mask) if method.startswith("TaskOnly") else np.nan,
                "rejected_hardening": np.nan,
                "description_bits": gate_description_bits(n_bits) + 4,
                "gate_count": 1,
                "fit_seconds": 0.0,
                "margin": margin if method.startswith("TaskOnly") else np.nan,
                "train_balanced_accuracy": train_score if method.startswith("TaskOnly") else np.nan,
            }
        )

    # Negative controls ask whether a one-gate hypothesis should be rejected.
    negative_tasks = {
        "majority3": (x[:, 0] + x[:, 1] + x[:, 2] >= 2).astype(np.uint8),
        "parity4": np.bitwise_xor.reduce(x[:, :4], axis=1),
        "random_lut": np.random.default_rng(44_000 + seed).integers(0, 2, size=len(x), dtype=np.uint8),
    }
    for name, y in negative_tasks.items():
        order = rng.permutation(len(x))
        train, test = order[: len(x) // 2], order[len(x) // 2 :]
        learned = DifferentiableGateSelector(n_bits, seed=seed * 1000 + len(name)).fit(x[train], y[train])
        score = balanced_accuracy(y[test], learned.predict(x[test]))
        rows.append(
            {
                "suite": "hardening_negative_control",
                "seed": seed,
                "task": name,
                "method": "DifferentiableGateSelector",
                "test_balanced_accuracy": score,
                "operator_recovered": np.nan,
                "inputs_recovered": np.nan,
                "topology_recovered": np.nan,
                "rejected_hardening": float(score < 0.90),
                "description_bits": gate_description_bits(n_bits),
                "gate_count": 1,
                "fit_seconds": learned.fit_seconds,
                "margin": float(np.sort(learned.weights())[-1] - np.sort(learned.weights())[-2]),
            }
        )
    continuous = rng.uniform(0.0, 1.0, size=(1024, n_bits))
    continuous_y = (continuous[:, 0] + 0.7 * continuous[:, 1] > 0.85).astype(np.uint8)
    thresholded = (continuous >= 0.5).astype(np.uint8)
    train = np.arange(0, 512)
    test = np.arange(512, 1024)
    learned = DifferentiableGateSelector(n_bits, seed=seed + 55_000).fit(thresholded[train], continuous_y[train])
    score = balanced_accuracy(continuous_y[test], learned.predict(thresholded[test]))
    rows.append(
        {
            "suite": "hardening_negative_control",
            "seed": seed,
            "task": "continuous_threshold",
            "method": "DifferentiableGateSelector",
            "test_balanced_accuracy": score,
            "operator_recovered": np.nan,
            "inputs_recovered": np.nan,
            "topology_recovered": np.nan,
            "rejected_hardening": float(score < 0.90),
            "description_bits": gate_description_bits(n_bits),
            "gate_count": 1,
            "fit_seconds": learned.fit_seconds,
            "margin": float(np.sort(learned.weights())[-1] - np.sort(learned.weights())[-2]),
        }
    )
    probabilities = rng.uniform(0.02, 0.98, size=(512, n_bits))
    target_probability = probabilities[:, 0] * probabilities[:, 1]
    hard_output = apply_gate("AND", probabilities[:, 0] >= 0.5, probabilities[:, 1] >= 0.5).astype(float)
    rows.append(
        {
            "suite": "hardening_negative_control",
            "seed": seed,
            "task": "probability_product",
            "method": "HardenedAND",
            "test_balanced_accuracy": np.nan,
            "operator_recovered": np.nan,
            "inputs_recovered": np.nan,
            "topology_recovered": np.nan,
            "rejected_hardening": float(np.mean((hard_output - target_probability) ** 2) > 0.05),
            "description_bits": 2,
            "gate_count": 1,
            "fit_seconds": 0.0,
            "margin": np.nan,
            "probability_mse": float(np.mean((hard_output - target_probability) ** 2)),
            "soft_probability_mse": float(np.mean((fuzzy_gate("AND", probabilities[:, 0], probabilities[:, 1]) - target_probability) ** 2)),
        }
    )
    return rows


def run_rate_reduction(seed: int, epsilon: float = 0.5) -> list[dict]:
    rng = np.random.default_rng(121_000 + seed)
    train_z, train_y = make_union_of_subspaces(rng, samples_per_class=128)
    test_z, test_y = make_union_of_subspaces(rng, samples_per_class=96)
    # The test distribution uses new bases, so nearest-subspace accuracy is an
    # intentionally harsh negative control rather than a standard IID split.
    representations = {
        "Raw": train_z,
        "MCR2Flow": run_rate_flow(train_z, train_y, sparse_lambda=0.0, epsilon=epsilon),
        "MCR2Flow+ISTA": run_rate_flow(train_z, train_y, sparse_lambda=0.035, epsilon=epsilon),
    }
    rows: list[dict] = []
    for method, z in representations.items():
        global_rate, within_rate, reduction = mcr2(z, train_y, epsilon)
        binary = (z >= 0).astype(np.uint8)
        rows.append(
            {
                "suite": "union_subspaces",
                "seed": seed,
                "task": "three_class_subspaces",
                "method": method,
                "global_rate_bits": global_rate,
                "within_rate_bits": within_rate,
                "rate_reduction_bits": reduction,
                "zero_fraction": float(np.mean(np.abs(z) < 1e-10)),
                "subspace_coherence": subspace_coherence(z, train_y),
                "empirical_code_entropy_bits": empirical_code_entropy(binary),
                "fixed_code_bits": binary.shape[0],
                "reconstruction_mse": linear_reconstruction_mse(binary, train_z),
                "classification_accuracy": nearest_subspace_accuracy(z, train_y, z, train_y),
                "ood_accuracy": np.nan,
                "gate_count": np.nan,
                "boolean_accuracy": np.nan,
            }
        )

    # Executable parameter-sharing accounting for a CRATE-style subspace head.
    d, tokens, heads = 12, 32, 3
    head_dim = d // heads
    bases = rng.normal(size=(heads, d, head_dim))
    bases = np.stack([np.linalg.qr(item)[0][:, :head_dim] for item in bases])
    tokens_z = rng.normal(size=(d, tokens))
    shared_output = shared_projection_attention(tokens_z, bases)
    rows.append(
        {
            "suite": "shared_projection_attention",
            "seed": seed,
            "task": "parameter_accounting",
            "method": "SharedSubspaceMSSAProxy",
            "global_rate_bits": np.nan,
            "within_rate_bits": np.nan,
            "rate_reduction_bits": np.nan,
            "zero_fraction": np.nan,
            "subspace_coherence": np.nan,
            "empirical_code_entropy_bits": np.nan,
            "fixed_code_bits": np.nan,
            "reconstruction_mse": np.nan,
            "classification_accuracy": np.nan,
            "ood_accuracy": np.nan,
            "gate_count": np.nan,
            "boolean_accuracy": np.nan,
            "projection_parameters": int(bases.size),
            "standard_qkv_parameters": int(3 * d * d),
            "parameter_ratio_vs_qkv": float(bases.size / (3 * d * d)),
            "output_norm": float(np.linalg.norm(shared_output)),
        }
    )

    # Directly test whether the log-det proxy orders literal bit/gate costs.
    x = all_assignments(8)
    signed = (2.0 * x - 1.0).T
    boolean_tasks = {
        "and2": x[:, 0] & x[:, 1],
        "majority3": (x[:, 0] + x[:, 1] + x[:, 2] >= 2).astype(np.uint8),
        "parity4": np.bitwise_xor.reduce(x[:, :4], axis=1),
        "random_lut": np.random.default_rng(131_000 + seed).integers(0, 2, size=len(x), dtype=np.uint8),
    }
    for task, y in boolean_tasks.items():
        raw = normalize_columns(signed)
        reduced = run_rate_flow(raw, y, layers=20, step_size=0.3, sparse_lambda=0.025, epsilon=epsilon)
        for method, z in (("RawBits", raw), ("RateReducedSign", reduced)):
            codes = (z >= 0).astype(np.uint8).T
            synthesizer = GateBeamSynthesizer(max_depth=3, beam_width=128).fit(codes, y)
            prediction = synthesizer.predict(codes)
            _, _, reduction = mcr2(z, y, epsilon)
            rows.append(
                {
                    "suite": "booleanization",
                    "seed": seed,
                    "task": task,
                    "method": method,
                    "global_rate_bits": mcr2(z, y, epsilon)[0],
                    "within_rate_bits": mcr2(z, y, epsilon)[1],
                    "rate_reduction_bits": reduction,
                    "zero_fraction": float(np.mean(np.abs(z) < 1e-10)),
                    "subspace_coherence": np.nan,
                    "empirical_code_entropy_bits": empirical_code_entropy(codes.T),
                    "fixed_code_bits": codes.shape[1],
                    "reconstruction_mse": linear_reconstruction_mse(codes.T, raw),
                    "classification_accuracy": np.nan,
                    "ood_accuracy": np.nan,
                    "gate_count": synthesizer.expression.gates if synthesizer.expression else np.nan,
                    "boolean_accuracy": accuracy(y, prediction),
                    "expression": synthesizer.expression.text if synthesizer.expression else "",
                }
            )
    return rows
