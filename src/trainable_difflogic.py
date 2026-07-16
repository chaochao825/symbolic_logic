"""Trainable fixed-wiring networks over all 16 two-input Boolean gates.

This module is deliberately independent of the frozen DigitalJS replay in
``difflogic_ca.py``.  It provides a small, auditable PyTorch reference for
soft, straight-through, and argmax-hard gate evaluation.  PyTorch is optional
for the dependency-light hard-CA test suite; NumPy truth tables and hard
exports remain available without it.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable, Sequence

import numpy as np

try:  # Optional experiment dependency.
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised by dependency-light CI.
    torch = None
    nn = None


GATE_NAMES: tuple[str, ...] = (
    "FALSE",
    "AND",
    "A_AND_NOT_B",
    "A",
    "NOT_A_AND_B",
    "B",
    "XOR",
    "OR",
    "NOR",
    "XNOR",
    "NOT_B",
    "A_OR_NOT_B",
    "NOT_A",
    "NOT_A_OR_B",
    "NAND",
    "TRUE",
)
PASS_THROUGH_GATE_IDS = frozenset((3, 5))


def numpy_all_binary_gates(a: Any, b: Any) -> np.ndarray:
    """Return the public 16-gate order for NumPy inputs in ``[0, 1]``."""
    a = np.asarray(a)
    b = np.asarray(b)
    ab = a * b
    return np.stack(
        (
            np.zeros_like(ab),
            ab,
            a - ab,
            a,
            b - ab,
            b,
            a + b - 2 * ab,
            a + b - ab,
            1 - a - b + ab,
            1 - a - b + 2 * ab,
            1 - b,
            1 - b + ab,
            1 - a,
            1 - a + ab,
            1 - ab,
            np.ones_like(ab),
        ),
        axis=-1,
    )


def gate_truth_tables() -> np.ndarray:
    """Return a ``[16, 4]`` table ordered by inputs 00, 01, 10, 11."""
    a = np.asarray([0, 0, 1, 1], dtype=np.uint8)
    b = np.asarray([0, 1, 0, 1], dtype=np.uint8)
    return numpy_all_binary_gates(a, b).T.astype(np.uint8)


def coverage_balanced_indices(in_features: int, out_features: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Create fixed connections while covering every reachable input.

    When two input slots per output are sufficient, each input occurs at least
    once.  Remaining slots use independent seeded permutations.  The function
    is deterministic across NumPy versions because only ``PCG64`` integer
    permutations are used.
    """
    if in_features < 2 or out_features < 1:
        raise ValueError("logic wiring needs at least two inputs and one output")
    slots = 2 * out_features
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    base = np.arange(slots, dtype=np.int64) % in_features
    rng.shuffle(base)
    left = base[:out_features].copy()
    right = base[out_features:].copy()
    same = left == right
    if np.any(same):
        right[same] = (right[same] + 1 + rng.integers(0, in_features - 1, size=int(same.sum()))) % in_features
    return left, right


def _require_torch() -> None:
    if torch is None or nn is None:
        raise ImportError("DiffLogic training requires PyTorch; install requirements-difflogic-arc.txt")


def torch_all_binary_gates(a: "torch.Tensor", b: "torch.Tensor") -> "torch.Tensor":
    """Torch counterpart of :func:`numpy_all_binary_gates`."""
    _require_torch()
    ab = a * b
    return torch.stack(
        (
            torch.zeros_like(ab),
            ab,
            a - ab,
            a,
            b - ab,
            b,
            a + b - 2 * ab,
            a + b - ab,
            1 - a - b + ab,
            1 - a - b + 2 * ab,
            1 - b,
            1 - b + ab,
            1 - a,
            1 - a + ab,
            1 - ab,
            torch.ones_like(ab),
        ),
        dim=-1,
    )


@dataclass(frozen=True)
class HardLogicLayerSpec:
    in_features: int
    out_features: int
    left_indices: tuple[int, ...]
    right_indices: tuple[int, ...]
    gate_ids: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "left_indices": list(self.left_indices),
            "right_indices": list(self.right_indices),
            "gate_ids": list(self.gate_ids),
        }


def hard_numpy_layer(x: Any, specification: HardLogicLayerSpec) -> np.ndarray:
    """Execute one exported hard layer on an arbitrary leading batch shape."""
    x = np.asarray(x)
    if x.shape[-1] != specification.in_features:
        raise ValueError("hard layer input width mismatch")
    a = x[..., np.asarray(specification.left_indices, dtype=np.int64)]
    b = x[..., np.asarray(specification.right_indices, dtype=np.int64)]
    functions = numpy_all_binary_gates(a, b)
    gate_ids = np.asarray(specification.gate_ids, dtype=np.int64)
    return np.take_along_axis(functions, gate_ids.reshape((1,) * (functions.ndim - 2) + (-1, 1)), axis=-1)[..., 0]


def hard_numpy_network(x: Any, specifications: Sequence[HardLogicLayerSpec]) -> np.ndarray:
    result = np.asarray(x)
    for specification in specifications:
        result = hard_numpy_layer(result, specification)
    return result


def hard_spec_sha256(specifications: Sequence[HardLogicLayerSpec]) -> str:
    payload = json.dumps([item.as_dict() for item in specifications], sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


if nn is not None:

    class DifferentiableLogicLayer(nn.Module):
        """One fixed-wiring layer that learns a distribution over 16 gates."""

        def __init__(
            self,
            in_features: int,
            out_features: int,
            *,
            wiring_seed: int,
            pass_bias: float = 0.35,
        ) -> None:
            super().__init__()
            left, right = coverage_balanced_indices(in_features, out_features, wiring_seed)
            self.in_features = int(in_features)
            self.out_features = int(out_features)
            self.register_buffer("left_indices", torch.as_tensor(left, dtype=torch.long))
            self.register_buffer("right_indices", torch.as_tensor(right, dtype=torch.long))
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(wiring_seed) + 10_000_019)
            initial = 0.01 * torch.randn(out_features, len(GATE_NAMES), generator=generator)
            initial[:, 3] += float(pass_bias)
            initial[:, 5] += float(pass_bias)
            self.logits = nn.Parameter(initial)

        def gate_probabilities(self, temperature: float) -> "torch.Tensor":
            if not temperature > 0:
                raise ValueError("temperature must be positive")
            return torch.softmax(self.logits / float(temperature), dim=-1)

        def forward(self, x: "torch.Tensor", *, temperature: float = 1.0, mode: str = "soft") -> "torch.Tensor":
            if x.shape[-1] != self.in_features:
                raise ValueError(f"logic input width {x.shape[-1]} != {self.in_features}")
            a = x.index_select(-1, self.left_indices)
            b = x.index_select(-1, self.right_indices)
            functions = torch_all_binary_gates(a, b)
            probabilities = self.gate_probabilities(temperature)
            if mode == "soft":
                weights = probabilities
            elif mode == "st":
                chosen = torch.nn.functional.one_hot(probabilities.argmax(dim=-1), len(GATE_NAMES)).to(probabilities.dtype)
                weights = chosen - probabilities.detach() + probabilities
            elif mode == "hard":
                indices = probabilities.argmax(dim=-1)
                view = (1,) * (functions.ndim - 2) + (self.out_features, 1)
                return torch.gather(functions, -1, indices.reshape(view).expand(functions.shape[:-1] + (1,)))[..., 0]
            else:
                raise ValueError("mode must be 'soft', 'st', or 'hard'")
            view = (1,) * (functions.ndim - 2) + probabilities.shape
            return torch.sum(functions * weights.reshape(view), dim=-1)

        def entropy(self, temperature: float = 1.0) -> "torch.Tensor":
            probabilities = self.gate_probabilities(temperature)
            return -(probabilities * torch.log(probabilities.clamp_min(1e-12))).sum(dim=-1).mean()

        def hard_specification(self) -> HardLogicLayerSpec:
            chosen = self.logits.detach().argmax(dim=-1).cpu().numpy()
            return HardLogicLayerSpec(
                self.in_features,
                self.out_features,
                tuple(int(value) for value in self.left_indices.detach().cpu().numpy()),
                tuple(int(value) for value in self.right_indices.detach().cpu().numpy()),
                tuple(int(value) for value in chosen),
            )


    class DifferentiableLogicNetwork(nn.Module):
        """A stack of fixed-wiring differentiable logic layers."""

        def __init__(
            self,
            in_features: int,
            layer_widths: Sequence[int],
            *,
            wiring_seed: int,
            pass_bias: float = 0.35,
        ) -> None:
            super().__init__()
            if not layer_widths or any(int(width) < 1 for width in layer_widths):
                raise ValueError("layer_widths must be nonempty positive integers")
            widths = (int(in_features), *(int(width) for width in layer_widths))
            self.layers = nn.ModuleList(
                DifferentiableLogicLayer(
                    widths[index],
                    widths[index + 1],
                    wiring_seed=int(wiring_seed) + 1_000_003 * index,
                    pass_bias=pass_bias,
                )
                for index in range(len(widths) - 1)
            )

        def forward(self, x: "torch.Tensor", *, temperature: float = 1.0, mode: str = "soft") -> "torch.Tensor":
            for layer in self.layers:
                x = layer(x, temperature=temperature, mode=mode)
            return x

        def entropy(self, temperature: float = 1.0) -> "torch.Tensor":
            return torch.stack([layer.entropy(temperature) for layer in self.layers]).mean()

        def hard_specifications(self) -> tuple[HardLogicLayerSpec, ...]:
            return tuple(layer.hard_specification() for layer in self.layers)

        def active_non_passthrough_gates(self) -> int:
            return sum(
                int(sum(gate not in PASS_THROUGH_GATE_IDS for gate in layer.hard_specification().gate_ids))
                for layer in self.layers
            )


else:  # pragma: no cover - allows import and clear errors without PyTorch.

    class DifferentiableLogicLayer:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any) -> None:
            _require_torch()


    class DifferentiableLogicNetwork:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any) -> None:
            _require_torch()


def count_hard_gate_ids(specifications: Iterable[HardLogicLayerSpec]) -> dict[str, int]:
    counts = {name: 0 for name in GATE_NAMES}
    for specification in specifications:
        for gate_id in specification.gate_ids:
            counts[GATE_NAMES[int(gate_id)]] += 1
    return counts
