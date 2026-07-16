"""ARC-adapted recurrent DiffLogic CA and matched MLP-NCA control."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np

from trainable_difflogic import (
    DifferentiableLogicNetwork,
    HardLogicLayerSpec,
    hard_numpy_network,
)

try:  # Optional experiment dependency.
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError:  # pragma: no cover - dependency-light CI.
    torch = None
    nn = None
    F = None


@dataclass(frozen=True)
class ArcDiffLogicConfig:
    name: str
    model_kind: str = "difflogic"
    hidden_bits: int = 0
    max_steps: int = 1
    hidden_widths: tuple[int, ...] = (96, 48, 24)
    include_original: bool = False
    include_geometry: bool = False
    include_objects: bool = False
    include_context: bool = False
    allow_shape_change: bool = False
    pass_bias: float = 0.35

    @property
    def state_bits(self) -> int:
        return 4 + int(self.hidden_bits)

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["state_bits"] = self.state_bits
        return result


VARIANT_CONFIGS: dict[str, ArcDiffLogicConfig] = {
    "dl1": ArcDiffLogicConfig("dl1", hidden_bits=0, max_steps=1, hidden_widths=(96, 48, 16)),
    "dlr": ArcDiffLogicConfig("dlr", hidden_bits=8, max_steps=4, hidden_widths=(128, 64, 32, 24)),
    "dlo": ArcDiffLogicConfig(
        "dlo",
        hidden_bits=8,
        max_steps=4,
        hidden_widths=(160, 96, 48, 24),
        include_original=True,
        include_geometry=True,
        include_objects=True,
        include_context=True,
    ),
    "dlf": ArcDiffLogicConfig(
        "dlf",
        hidden_bits=8,
        max_steps=8,
        hidden_widths=(192, 96, 48, 24),
        include_original=True,
        include_geometry=True,
        include_objects=True,
        include_context=True,
        allow_shape_change=True,
    ),
    "mlp": ArcDiffLogicConfig(
        "mlp",
        model_kind="mlp",
        hidden_bits=8,
        max_steps=8,
        hidden_widths=(192, 96),
        include_original=True,
        include_geometry=True,
        include_objects=True,
        include_context=True,
        allow_shape_change=True,
        pass_bias=0.0,
    ),
}


def _require_torch() -> None:
    if torch is None or nn is None or F is None:
        raise ImportError("ARC DiffLogic training requires PyTorch")


if nn is not None:

    class ArcDiffLogicCA(nn.Module):
        def __init__(self, config: ArcDiffLogicConfig, static_channels: int, *, wiring_seed: int) -> None:
            super().__init__()
            if config.model_kind != "difflogic":
                raise ValueError("ArcDiffLogicCA requires model_kind='difflogic'")
            self.config = config
            self.static_channels = int(static_channels)
            input_width = 9 * config.state_bits + self.static_channels
            widths = (*config.hidden_widths, config.state_bits)
            self.logic = DifferentiableLogicNetwork(
                input_width,
                widths,
                wiring_seed=int(wiring_seed),
                pass_bias=config.pass_bias,
            )

        def step(
            self,
            state: "torch.Tensor",
            static: "torch.Tensor",
            update_mask: "torch.Tensor",
            *,
            temperature: float,
            mode: str,
        ) -> "torch.Tensor":
            batch, channels, rows, columns = state.shape
            if channels != self.config.state_bits or static.shape != (batch, self.static_channels, rows, columns):
                raise ValueError("state/static dimensions disagree with model")
            patches = F.unfold(state, kernel_size=3, padding=1).transpose(1, 2)
            center_static = static.permute(0, 2, 3, 1).reshape(batch, rows * columns, self.static_channels)
            inputs = torch.cat((patches, center_static), dim=-1)
            outputs = self.logic(inputs, temperature=temperature, mode=mode)
            next_state = outputs.reshape(batch, rows, columns, channels).permute(0, 3, 1, 2)
            return next_state * update_mask[:, None, :, :]

        def rollout(
            self,
            initial_state: "torch.Tensor",
            static: "torch.Tensor",
            update_mask: "torch.Tensor",
            *,
            steps: int | None = None,
            temperature: float = 1.0,
            mode: str = "soft",
        ) -> tuple["torch.Tensor", ...]:
            count = self.config.max_steps if steps is None else int(steps)
            if count < 1:
                raise ValueError("rollout steps must be positive")
            states = []
            state = initial_state
            for _ in range(count):
                state = self.step(state, static, update_mask, temperature=temperature, mode=mode)
                states.append(state)
            return tuple(states)

        def gate_entropy(self, temperature: float = 1.0) -> "torch.Tensor":
            return self.logic.entropy(temperature)

        def hard_specifications(self) -> tuple[HardLogicLayerSpec, ...]:
            return self.logic.hard_specifications()

        def active_non_passthrough_gates(self) -> int:
            return self.logic.active_non_passthrough_gates()


    class MatchedMLPNCA(nn.Module):
        def __init__(self, config: ArcDiffLogicConfig, static_channels: int, *, wiring_seed: int) -> None:
            super().__init__()
            if config.model_kind != "mlp":
                raise ValueError("MatchedMLPNCA requires model_kind='mlp'")
            torch.manual_seed(int(wiring_seed))
            self.config = config
            self.static_channels = int(static_channels)
            input_width = 9 * config.state_bits + self.static_channels
            widths = (input_width, *config.hidden_widths, config.state_bits)
            modules: list[nn.Module] = []
            for index in range(len(widths) - 1):
                modules.append(nn.Linear(widths[index], widths[index + 1]))
                if index + 1 < len(widths) - 1:
                    modules.append(nn.ReLU())
            self.network = nn.Sequential(*modules)

        def step(
            self,
            state: "torch.Tensor",
            static: "torch.Tensor",
            update_mask: "torch.Tensor",
            *,
            temperature: float,
            mode: str,
        ) -> "torch.Tensor":
            del temperature
            batch, channels, rows, columns = state.shape
            patches = F.unfold(state, kernel_size=3, padding=1).transpose(1, 2)
            center_static = static.permute(0, 2, 3, 1).reshape(batch, rows * columns, self.static_channels)
            probabilities = torch.sigmoid(self.network(torch.cat((patches, center_static), dim=-1)))
            if mode == "hard":
                outputs = (probabilities >= 0.5).to(probabilities.dtype)
            elif mode == "st":
                hard = (probabilities >= 0.5).to(probabilities.dtype)
                outputs = hard - probabilities.detach() + probabilities
            elif mode == "soft":
                outputs = probabilities
            else:
                raise ValueError("mode must be 'soft', 'st', or 'hard'")
            next_state = outputs.reshape(batch, rows, columns, channels).permute(0, 3, 1, 2)
            return next_state * update_mask[:, None, :, :]

        def rollout(
            self,
            initial_state: "torch.Tensor",
            static: "torch.Tensor",
            update_mask: "torch.Tensor",
            *,
            steps: int | None = None,
            temperature: float = 1.0,
            mode: str = "soft",
        ) -> tuple["torch.Tensor", ...]:
            count = self.config.max_steps if steps is None else int(steps)
            states = []
            state = initial_state
            for _ in range(count):
                state = self.step(state, static, update_mask, temperature=temperature, mode=mode)
                states.append(state)
            return tuple(states)

        def gate_entropy(self, temperature: float = 1.0) -> "torch.Tensor":
            del temperature
            return next(self.parameters()).new_zeros(())

        def active_non_passthrough_gates(self) -> int:
            return 0


else:  # pragma: no cover

    class ArcDiffLogicCA:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any) -> None:
            _require_torch()


    class MatchedMLPNCA:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any) -> None:
            _require_torch()


def create_model(config: ArcDiffLogicConfig, static_channels: int, *, wiring_seed: int) -> Any:
    _require_torch()
    if config.model_kind == "difflogic":
        return ArcDiffLogicCA(config, static_channels, wiring_seed=wiring_seed)
    if config.model_kind == "mlp":
        return MatchedMLPNCA(config, static_channels, wiring_seed=wiring_seed)
    raise ValueError(f"unknown model kind {config.model_kind}")


def invalid_color_probability(color_probabilities: "torch.Tensor") -> "torch.Tensor":
    """Probability mass assigned to independent binary4 codes 10--15."""
    _require_torch()
    if color_probabilities.shape[1] != 4:
        raise ValueError("color probabilities must be channel-first binary4")
    probabilities = color_probabilities.clamp(1e-6, 1 - 1e-6)
    invalid = probabilities.new_zeros(probabilities.shape[0], *probabilities.shape[2:])
    for code in range(10, 16):
        bits = torch.as_tensor([(code >> index) & 1 for index in range(4)], device=probabilities.device, dtype=probabilities.dtype)
        terms = torch.where(bits[None, :, None, None] > 0, probabilities, 1 - probabilities)
        invalid = invalid + torch.prod(terms, dim=1)
    return invalid


def hard_numpy_ca_step(
    state: np.ndarray,
    static: np.ndarray,
    update_mask: np.ndarray,
    specifications: Sequence[HardLogicLayerSpec],
) -> np.ndarray:
    """Execute an exported DiffLogic CA step without PyTorch."""
    state = np.asarray(state)
    static = np.asarray(static)
    update_mask = np.asarray(update_mask)
    if state.ndim != 3 or static.ndim != 3 or state.shape[1:] != static.shape[1:]:
        raise ValueError("hard NumPy CA expects channel-first state/static")
    channels, rows, columns = state.shape
    padded = np.pad(state, ((0, 0), (1, 1), (1, 1)), constant_values=0)
    patch_channels = []
    for channel in range(channels):
        for delta_row in range(3):
            for delta_column in range(3):
                patch_channels.append(padded[channel, delta_row : delta_row + rows, delta_column : delta_column + columns])
    patches = np.stack(patch_channels, axis=-1)
    center_static = np.moveaxis(static, 0, -1)
    inputs = np.concatenate((patches, center_static), axis=-1).reshape(rows * columns, -1)
    outputs = hard_numpy_network(inputs, specifications)
    result = outputs.reshape(rows, columns, channels).transpose(2, 0, 1)
    return result * update_mask[None]


def hard_numpy_ca_rollout(
    initial_state: np.ndarray,
    static: np.ndarray,
    update_mask: np.ndarray,
    specifications: Sequence[HardLogicLayerSpec],
    steps: int,
) -> tuple[np.ndarray, ...]:
    state = np.asarray(initial_state)
    states = []
    for _ in range(int(steps)):
        state = hard_numpy_ca_step(state, static, update_mask, specifications)
        states.append(state)
    return tuple(states)
