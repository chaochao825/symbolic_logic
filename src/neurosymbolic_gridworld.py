"""Pixel-to-symbol-to-solver neuro-symbolic gridworld experiment.

The task starts from rendered RGB images.  A learned patch MLP grounds four
named cell concepts (free, wall, source, target), a learned differentiable gate
maps passability probabilities to local edge validity, and a symbolic solver
answers source-to-target reachability.  The model is deliberately small enough
to run with NumPy while still closing the raw-pixel-to-task-output loop.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from logic_core import accuracy, balanced_accuracy, brier_score


FREE, WALL, SOURCE, TARGET = 0, 1, 2, 3
BASE_COLORS = np.asarray(
    [
        [0.82, 0.82, 0.82],
        [0.12, 0.12, 0.16],
        [0.15, 0.72, 0.25],
        [0.82, 0.18, 0.18],
    ],
    dtype=np.float64,
)


@dataclass
class GridDataset:
    images: np.ndarray
    cell_labels: np.ndarray
    source_indices: np.ndarray
    target_indices: np.ndarray
    task_labels: np.ndarray
    condition: str
    grid_size: int
    patch_size: int


def _grid_reachable(labels: np.ndarray, source: int, target: int) -> bool:
    size = labels.shape[0]
    visited = np.zeros((size, size), dtype=bool)
    frontier = [divmod(source, size)]
    visited[frontier[0]] = True
    while frontier:
        row, col = frontier.pop()
        if row * size + col == target:
            return True
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = row + dr, col + dc
            if 0 <= nr < size and 0 <= nc < size and not visited[nr, nc] and labels[nr, nc] != WALL:
                visited[nr, nc] = True
                frontier.append((nr, nc))
    return False


def _make_grid(size: int, reachable: bool, rng: np.random.Generator) -> tuple[np.ndarray, int, int]:
    labels = np.full((size, size), FREE, dtype=np.uint8)
    if reachable:
        labels[rng.random((size, size)) < 0.24] = WALL
        source_rc = (int(rng.integers(size)), int(rng.integers(size)))
        target_rc = (int(rng.integers(size)), int(rng.integers(size)))
        while target_rc == source_rc:
            target_rc = (int(rng.integers(size)), int(rng.integers(size)))
        row, col = source_rc
        labels[row, col] = FREE
        while col != target_rc[1]:
            col += 1 if target_rc[1] > col else -1
            labels[row, col] = FREE
        while row != target_rc[0]:
            row += 1 if target_rc[0] > row else -1
            labels[row, col] = FREE
    else:
        vertical = bool(rng.integers(2))
        barrier = int(rng.integers(1, size - 1))
        labels[rng.random((size, size)) < 0.16] = WALL
        if vertical:
            labels[:, barrier] = WALL
            source_rc = (int(rng.integers(size)), int(rng.integers(0, barrier)))
            target_rc = (int(rng.integers(size)), int(rng.integers(barrier + 1, size)))
        else:
            labels[barrier, :] = WALL
            source_rc = (int(rng.integers(0, barrier)), int(rng.integers(size)))
            target_rc = (int(rng.integers(barrier + 1, size)), int(rng.integers(size)))
    labels[source_rc] = SOURCE
    labels[target_rc] = TARGET
    source = source_rc[0] * size + source_rc[1]
    target = target_rc[0] * size + target_rc[1]
    if _grid_reachable(labels, source, target) != reachable:
        raise AssertionError("grid construction changed requested reachability")
    return labels, source, target


def _render_grid(labels: np.ndarray, patch_size: int, rng: np.random.Generator, condition: str) -> np.ndarray:
    size = labels.shape[0]
    global_shift = rng.normal(0.0, 0.035, size=3)
    image = np.empty((size * patch_size, size * patch_size, 3), dtype=np.float64)
    for row in range(size):
        for col in range(size):
            color = BASE_COLORS[int(labels[row, col])] + global_shift + rng.normal(0.0, 0.035, size=3)
            patch = color + rng.normal(0.0, 0.045, size=(patch_size, patch_size, 3))
            image[row * patch_size : (row + 1) * patch_size, col * patch_size : (col + 1) * patch_size] = patch
    if condition == "correlated_occlusion":
        height = int(rng.integers(2, min(4, size) + 1))
        width = int(rng.integers(2, min(4, size) + 1))
        top = int(rng.integers(0, size - height + 1))
        left = int(rng.integers(0, size - width + 1))
        r0, r1 = top * patch_size, (top + height) * patch_size
        c0, c1 = left * patch_size, (left + width) * patch_size
        tint = rng.uniform(0.18, 0.72, size=3)
        image[r0:r1, c0:c1] = 0.38 * image[r0:r1, c0:c1] + 0.62 * tint
        image = image * rng.uniform(0.78, 1.12, size=3) + rng.normal(0.0, 0.025, size=image.shape)
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def make_dataset(
    n_samples: int,
    grid_size: int,
    patch_size: int,
    seed: int,
    condition: str = "clean",
) -> GridDataset:
    rng = np.random.default_rng(seed)
    images = []
    labels = []
    sources = []
    targets = []
    tasks = []
    for index in range(n_samples):
        desired = bool(index % 2)
        grid, source, target = _make_grid(grid_size, desired, rng)
        images.append(_render_grid(grid, patch_size, rng, condition))
        labels.append(grid)
        sources.append(source)
        targets.append(target)
        tasks.append(int(desired))
    permutation = rng.permutation(n_samples)
    return GridDataset(
        images=np.asarray(images)[permutation],
        cell_labels=np.asarray(labels)[permutation],
        source_indices=np.asarray(sources, dtype=np.int32)[permutation],
        target_indices=np.asarray(targets, dtype=np.int32)[permutation],
        task_labels=np.asarray(tasks, dtype=np.uint8)[permutation],
        condition=condition,
        grid_size=grid_size,
        patch_size=patch_size,
    )


def extract_patches(images: np.ndarray, grid_size: int, patch_size: int) -> np.ndarray:
    n = len(images)
    return (
        images.reshape(n, grid_size, patch_size, grid_size, patch_size, 3)
        .transpose(0, 1, 3, 2, 4, 5)
        .reshape(n * grid_size * grid_size, patch_size * patch_size * 3)
        .astype(np.float64)
    )


class PatchMLPEncoder:
    """One-hidden-layer neural grounding model for rendered cell patches."""

    def __init__(self, n_inputs: int, hidden: int = 32, seed: int = 0, lr: float = 0.01) -> None:
        rng = np.random.default_rng(seed)
        self.lr = lr
        self.rng = rng
        self.w1 = rng.normal(0, np.sqrt(2 / n_inputs), size=(n_inputs, hidden))
        self.b1 = np.zeros(hidden)
        self.w2 = rng.normal(0, np.sqrt(2 / hidden), size=(hidden, 4))
        self.b2 = np.zeros(4)
        self.temperature = 1.0
        self.fit_seconds = 0.0

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)

    def predict_logits(self, x: np.ndarray) -> np.ndarray:
        hidden = np.maximum(np.asarray(x, dtype=np.float64) @ self.w1 + self.b1, 0.0)
        return hidden @ self.w2 + self.b2

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self._softmax(self.predict_logits(x) / self.temperature)

    def fit(self, x: np.ndarray, y: np.ndarray, steps: int = 500, batch_size: int = 1024) -> "PatchMLPEncoder":
        started = time.perf_counter()
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        parameters = [self.w1, self.b1, self.w2, self.b2]
        m = [np.zeros_like(item) for item in parameters]
        v = [np.zeros_like(item) for item in parameters]
        beta1, beta2 = 0.9, 0.999
        class_indices = [np.flatnonzero(y == label) for label in range(4)]
        for step in range(1, steps + 1):
            per_class = max(1, min(batch_size, len(x)) // 4)
            indices = np.concatenate([self.rng.choice(pool, size=per_class, replace=True) for pool in class_indices])
            self.rng.shuffle(indices)
            xb, yb = x[indices], y[indices]
            z1 = xb @ self.w1 + self.b1
            h1 = np.maximum(z1, 0.0)
            probabilities = self._softmax(h1 @ self.w2 + self.b2)
            probabilities[np.arange(len(yb)), yb] -= 1.0
            probabilities /= len(yb)
            gradients = [
                xb.T @ ((probabilities @ self.w2.T) * (z1 > 0)),
                ((probabilities @ self.w2.T) * (z1 > 0)).sum(axis=0),
                h1.T @ probabilities,
                probabilities.sum(axis=0),
            ]
            for index, (parameter, gradient) in enumerate(zip(parameters, gradients)):
                m[index] = beta1 * m[index] + (1 - beta1) * gradient
                v[index] = beta2 * v[index] + (1 - beta2) * gradient * gradient
                parameter -= self.lr * (m[index] / (1 - beta1**step)) / (np.sqrt(v[index] / (1 - beta2**step)) + 1e-8)
        self.fit_seconds = time.perf_counter() - started
        return self

    def calibrate_temperature(self, x: np.ndarray, y: np.ndarray) -> float:
        logits = self.predict_logits(x)
        y = np.asarray(y, dtype=np.int64)
        best = (float("inf"), 1.0)
        temperatures = np.unique(np.concatenate((np.geomspace(0.08, 1.0, 40), np.linspace(1.1, 5.0, 40))))
        for temperature in temperatures:
            probability = self._softmax(logits / temperature)
            nll = float(-np.log(np.clip(probability[np.arange(len(y)), y], 1e-12, 1.0)).mean())
            if nll < best[0]:
                best = (nll, float(temperature))
        self.temperature = best[1]
        return self.temperature


class LearnedBinaryGate:
    """Differentiable mixture over four Boolean operators for edge validity."""

    OPS = ("AND", "OR", "XOR", "NAND")

    def __init__(self, seed: int = 0, lr: float = 0.08) -> None:
        self.rng = np.random.default_rng(seed)
        self.logits = self.rng.normal(0.0, 0.05, size=4)
        self.lr = lr
        self.fit_seconds = 0.0

    def weights(self) -> np.ndarray:
        shifted = self.logits - self.logits.max()
        exp = np.exp(shifted)
        return exp / exp.sum()

    @staticmethod
    def op_values(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.stack((a * b, a + b - a * b, a + b - 2 * a * b, 1 - a * b), axis=-1)

    def predict_proba(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.clip(self.op_values(np.asarray(a), np.asarray(b)) @ self.weights(), 0.0, 1.0)

    def selected_op(self) -> str:
        return self.OPS[int(np.argmax(self.logits))]

    def predict_hard(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        av = np.asarray(a) >= 0.5
        bv = np.asarray(b) >= 0.5
        op = self.selected_op()
        if op == "AND":
            return av & bv
        if op == "OR":
            return av | bv
        if op == "XOR":
            return av ^ bv
        return ~(av & bv)

    def fit(self, a: np.ndarray, b: np.ndarray, y: np.ndarray, steps: int = 800) -> "LearnedBinaryGate":
        started = time.perf_counter()
        values = self.op_values(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))
        y = np.asarray(y, dtype=np.float64)
        m = np.zeros_like(self.logits)
        v = np.zeros_like(self.logits)
        for step in range(1, steps + 1):
            weights = self.weights()
            output = values @ weights
            derivative = (output - y) / len(y)
            gradient = np.sum(derivative[:, None] * weights[None, :] * (values - output[:, None]), axis=0)
            m = 0.9 * m + 0.1 * gradient
            v = 0.999 * v + 0.001 * gradient * gradient
            self.logits -= self.lr * (m / (1 - 0.9**step)) / (np.sqrt(v / (1 - 0.999**step)) + 1e-8)
        self.fit_seconds = time.perf_counter() - started
        return self


def _adjacent_edges(size: int) -> tuple[np.ndarray, np.ndarray]:
    sources = []
    targets = []
    for row in range(size):
        for col in range(size):
            source = row * size + col
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = row + dr, col + dc
                if 0 <= nr < size and 0 <= nc < size:
                    sources.append(source)
                    targets.append(nr * size + nc)
    return np.asarray(sources, dtype=np.int32), np.asarray(targets, dtype=np.int32)


def _bfs_from_edges(size: int, edge_source: np.ndarray, edge_target: np.ndarray, valid: np.ndarray, source: int, target: int) -> bool:
    adjacency = [[] for _ in range(size * size)]
    for left, right in zip(edge_source[valid], edge_target[valid]):
        adjacency[int(left)].append(int(right))
    frontier = [int(source)]
    visited = {int(source)}
    while frontier:
        node = frontier.pop()
        if node == int(target):
            return True
        for neighbor in adjacency[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                frontier.append(neighbor)
    return False


def _soft_path_score(size: int, edge_source: np.ndarray, edge_target: np.ndarray, edge_probability: np.ndarray, source: int, target: int) -> float:
    score = np.zeros(size * size, dtype=np.float64)
    score[int(source)] = 1.0
    for _ in range(size * size):
        updated = score.copy()
        np.maximum.at(updated, edge_target, score[edge_source] * edge_probability)
        if np.max(np.abs(updated - score)) < 1e-10:
            break
        score = updated
    return float(score[int(target)])


def _decode_dataset(dataset: GridDataset, encoder: PatchMLPEncoder) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    patches = extract_patches(dataset.images, dataset.grid_size, dataset.patch_size)
    started = time.perf_counter()
    probabilities = encoder.predict_proba(patches).reshape(len(dataset.images), dataset.grid_size * dataset.grid_size, 4)
    milliseconds_per_image = (time.perf_counter() - started) * 1000 / len(dataset.images)
    source = np.argmax(probabilities[:, :, SOURCE], axis=1)
    target = np.argmax(probabilities[:, :, TARGET], axis=1)
    return probabilities, source.astype(np.int32), target.astype(np.int32), milliseconds_per_image


def _multiclass_ece(probabilities: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    flat_probability = probabilities.reshape(-1, probabilities.shape[-1])
    flat_labels = labels.reshape(-1)
    confidence = flat_probability.max(axis=1)
    correct = flat_probability.argmax(axis=1) == flat_labels
    error = 0.0
    for lower, upper in zip(np.linspace(0.0, 1.0, bins + 1)[:-1], np.linspace(0.0, 1.0, bins + 1)[1:]):
        selected = (confidence > lower) & (confidence <= upper)
        if selected.any():
            error += float(selected.mean()) * abs(float(correct[selected].mean()) - float(confidence[selected].mean()))
    return error


def collect_edge_training_data(dataset: GridDataset, encoder: PatchMLPEncoder, max_pairs: int = 100_000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probabilities, _, _, _ = _decode_dataset(dataset, encoder)
    edge_source, edge_target = _adjacent_edges(dataset.grid_size)
    a_values = []
    b_values = []
    labels = []
    for index in range(len(dataset.images)):
        passable = 1.0 - probabilities[index, :, WALL]
        truth = dataset.cell_labels[index].reshape(-1) != WALL
        a_values.append(passable[edge_source])
        b_values.append(passable[edge_target])
        labels.append((truth[edge_source] & truth[edge_target]).astype(np.uint8))
    a = np.concatenate(a_values)
    b = np.concatenate(b_values)
    y = np.concatenate(labels)
    if len(y) > max_pairs:
        indices = np.linspace(0, len(y) - 1, max_pairs, dtype=np.int64)
        return a[indices], b[indices], y[indices]
    return a, b, y


def infer_scores(dataset: GridDataset, encoder: PatchMLPEncoder, gate: LearnedBinaryGate) -> dict[str, np.ndarray | float]:
    probabilities, predicted_source, predicted_target, encoder_ms = _decode_dataset(dataset, encoder)
    edge_source, edge_target = _adjacent_edges(dataset.grid_size)
    hard = []
    soft = []
    oracle = []
    confidence = []
    hard_solver_ms = []
    soft_solver_ms = []
    oracle_solver_ms = []
    for index in range(len(dataset.images)):
        passable = 1.0 - probabilities[index, :, WALL]
        edge_probability = gate.predict_proba(passable[edge_source], passable[edge_target])
        hard_edges = gate.predict_hard(passable[edge_source], passable[edge_target])
        started = time.perf_counter_ns()
        hard.append(_bfs_from_edges(dataset.grid_size, edge_source, edge_target, hard_edges, predicted_source[index], predicted_target[index]))
        hard_solver_ms.append((time.perf_counter_ns() - started) / 1_000_000)
        started = time.perf_counter_ns()
        soft.append(_soft_path_score(dataset.grid_size, edge_source, edge_target, edge_probability, predicted_source[index], predicted_target[index]))
        soft_solver_ms.append((time.perf_counter_ns() - started) / 1_000_000)
        truth = dataset.cell_labels[index].reshape(-1) != WALL
        oracle_edges = truth[edge_source] & truth[edge_target]
        started = time.perf_counter_ns()
        oracle.append(_bfs_from_edges(dataset.grid_size, edge_source, edge_target, oracle_edges, dataset.source_indices[index], dataset.target_indices[index]))
        oracle_solver_ms.append((time.perf_counter_ns() - started) / 1_000_000)
        cell_confidence = probabilities[index].max(axis=1).mean()
        confidence.append(
            min(
                probabilities[index, predicted_source[index], SOURCE],
                probabilities[index, predicted_target[index], TARGET],
                cell_confidence,
            )
        )
    cell_prediction = probabilities.argmax(axis=2).reshape(dataset.cell_labels.shape)
    return {
        "hard": np.asarray(hard, dtype=np.uint8),
        "soft": np.asarray(soft, dtype=np.float64),
        "oracle": np.asarray(oracle, dtype=np.uint8),
        "confidence": np.asarray(confidence, dtype=np.float64),
        "cell_accuracy": float(np.mean(cell_prediction == dataset.cell_labels)),
        "cell_ece": _multiclass_ece(probabilities, dataset.cell_labels),
        "source_accuracy": float(np.mean(predicted_source == dataset.source_indices)),
        "target_accuracy": float(np.mean(predicted_target == dataset.target_indices)),
        "encoder_ms_per_image": encoder_ms,
        "hard_solver_ms": np.asarray(hard_solver_ms, dtype=np.float64),
        "soft_solver_ms": np.asarray(soft_solver_ms, dtype=np.float64),
        "oracle_solver_ms": np.asarray(oracle_solver_ms, dtype=np.float64),
    }


def _best_threshold(labels: np.ndarray, score: np.ndarray) -> float:
    best = (-1.0, 0.5)
    thresholds = np.unique(np.concatenate((np.geomspace(1e-4, 0.1, 30), np.linspace(0.12, 0.95, 34))))
    for threshold in thresholds:
        value = balanced_accuracy(labels, score >= threshold)
        if value > best[0]:
            best = (value, float(threshold))
    return best[1]


def tune_hybrid(labels: np.ndarray, scores: dict[str, np.ndarray | float]) -> tuple[float, float]:
    soft = np.asarray(scores["soft"])
    hard = np.asarray(scores["hard"])
    confidence = np.asarray(scores["confidence"])
    soft_threshold = _best_threshold(labels, soft)
    best = (-1.0, 0.75)
    for threshold in np.linspace(0.10, 0.99, 46):
        prediction = np.where(confidence >= threshold, hard, soft >= soft_threshold)
        value = balanced_accuracy(labels, prediction)
        if value > best[0]:
            best = (value, float(threshold))
    return soft_threshold, best[1]


def evaluate_methods(
    dataset: GridDataset,
    scores: dict[str, np.ndarray | float],
    soft_threshold: float,
    confidence_threshold: float,
    seed: int,
    gate: LearnedBinaryGate,
    encoder: PatchMLPEncoder,
) -> list[dict]:
    labels = dataset.task_labels
    hard = np.asarray(scores["hard"], dtype=np.uint8)
    soft_probability = np.asarray(scores["soft"], dtype=np.float64)
    soft = (soft_probability >= soft_threshold).astype(np.uint8)
    confidence = np.asarray(scores["confidence"], dtype=np.float64)
    fallback = confidence < confidence_threshold
    hybrid_probability = np.where(fallback, soft_probability, hard.astype(float))
    hybrid = (hybrid_probability >= soft_threshold).astype(np.uint8)
    oracle = np.asarray(scores["oracle"], dtype=np.uint8)
    rows = []
    hard_solver_ms = np.asarray(scores["hard_solver_ms"], dtype=np.float64)
    soft_solver_ms = np.asarray(scores["soft_solver_ms"], dtype=np.float64)
    oracle_solver_ms = np.asarray(scores["oracle_solver_ms"], dtype=np.float64)
    hybrid_solver_ms = np.where(fallback, soft_solver_ms, hard_solver_ms)
    for method, prediction, probability, solver_ms in (
        ("OracleBFS", oracle, oracle.astype(float), oracle_solver_ms),
        ("HardNeuroSymbolic", hard, hard.astype(float), hard_solver_ms),
        ("SoftNeuroSymbolic", soft, soft_probability, soft_solver_ms),
        ("HybridFallback", hybrid, hybrid_probability, hybrid_solver_ms),
    ):
        rows.append(
            {
                "seed": seed,
                "condition": dataset.condition,
                "grid_size": dataset.grid_size,
                "samples": len(labels),
                "method": method,
                "accuracy": accuracy(labels, prediction),
                "balanced_accuracy": balanced_accuracy(labels, prediction),
                "brier": brier_score(labels, probability),
                "cell_grounding_accuracy": scores["cell_accuracy"],
                "cell_grounding_ece": scores["cell_ece"],
                "source_localization_accuracy": scores["source_accuracy"],
                "target_localization_accuracy": scores["target_accuracy"],
                "fallback_rate": float(fallback.mean()) if method == "HybridFallback" else 0.0,
                "soft_threshold": soft_threshold,
                "confidence_threshold": confidence_threshold,
                "encoder_ms_per_image": scores["encoder_ms_per_image"],
                "median_solver_ms": float(np.median(solver_ms)),
                "selected_gate": gate.selected_op(),
                "encoder_temperature": encoder.temperature,
                "encoder_fit_seconds": encoder.fit_seconds,
                "gate_fit_seconds": gate.fit_seconds,
            }
        )
    return rows
