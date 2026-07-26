"""Small TRM-style recurrent Transformer building blocks.

This is an independent, minimal implementation of the recurrent block family
described by USRL and TRM.  It intentionally uses only PyTorch primitives.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def rms_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    dtype = x.dtype
    normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
    return normalized.to(dtype)


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_length: int, theta: float) -> None:
        super().__init__()
        if head_dim % 2:
            raise ValueError("head_dim must be even")
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        positions = torch.arange(max_length, dtype=torch.float32)
        frequencies = torch.outer(positions, inv_freq)
        angles = torch.cat((frequencies, frequencies), dim=-1)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def values(self, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        if length > self.cos.shape[0]:
            raise ValueError(f"sequence length {length} exceeds RoPE limit {self.cos.shape[0]}")
        return self.cos[:length], self.sin[:length]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    left, right = x.chunk(2, dim=-1)
    return torch.cat((-right, left), dim=-1)


def _apply_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # query/key: [batch, heads, sequence, head_dim]
    original_dtype = query.dtype
    cos = cos.to(query.device, dtype=torch.float32)[None, None, :, :]
    sin = sin.to(query.device, dtype=torch.float32)[None, None, :, :]
    query_float = query.float()
    key_float = key.float()
    query = query_float * cos + _rotate_half(query_float) * sin
    key = key_float * cos + _rotate_half(key_float) * sin
    return query.to(original_dtype), key.to(original_dtype)


class SelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.output = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self,
        hidden: torch.Tensor,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        batch, length, _ = hidden.shape
        qkv = self.qkv(hidden).view(batch, length, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        query, key = _apply_rope(query, key, *cos_sin)
        attended = F.scaled_dot_product_attention(query, key, value, is_causal=False)
        attended = attended.transpose(1, 2).contiguous().view(batch, length, self.hidden_size)
        return self.output(attended)


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, expansion: float) -> None:
        super().__init__()
        # TRM rounds to 256.  For smoke-sized networks a 64 multiple avoids a
        # disproportionately large MLP while preserving the same construction.
        multiple = 256 if hidden_size >= 256 else 64
        intermediate = _round_up(round(expansion * hidden_size * 2 / 3), multiple)
        self.gate_up = nn.Linear(hidden_size, 2 * intermediate, bias=False)
        self.down = nn.Linear(intermediate, hidden_size, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        gate, value = self.gate_up(hidden).chunk(2, dim=-1)
        return self.down(F.silu(gate) * value)


class RecurrentTransformerLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        expansion: float,
        norm_eps: float,
    ) -> None:
        super().__init__()
        self.attention = SelfAttention(hidden_size, num_heads)
        self.mlp = SwiGLU(hidden_size, expansion)
        self.norm_eps = norm_eps

    def forward(
        self,
        hidden: torch.Tensor,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden = rms_norm(hidden + self.attention(hidden, cos_sin), self.norm_eps)
        hidden = rms_norm(hidden + self.mlp(hidden), self.norm_eps)
        return hidden


class RecurrentTransformer(nn.Module):
    """Two-state recurrence used independently by UM and SM.

    For every outer cycle, the scratch state is updated ``inner_loops`` times
    from ``answer + input_injection`` and the answer state is updated once from
    the scratch state.  The first ``outer_loops - 1`` cycles are warm-up cycles
    and may run without gradients, matching the public TRM implementation.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        layers: int,
        inner_loops: int,
        outer_loops: int,
        expansion: float,
        norm_eps: float,
        max_length: int,
        rope_theta: float,
        no_grad_warmup: bool,
    ) -> None:
        super().__init__()
        self.inner_loops = inner_loops
        self.outer_loops = outer_loops
        self.no_grad_warmup = no_grad_warmup
        self.layers = nn.ModuleList(
            RecurrentTransformerLayer(hidden_size, num_heads, expansion, norm_eps)
            for _ in range(layers)
        )
        self.rotary = RotaryEmbedding(hidden_size // num_heads, max_length, rope_theta)
        self.register_buffer("answer_init", torch.empty(hidden_size), persistent=True)
        self.register_buffer("scratch_init", torch.empty(hidden_size), persistent=True)
        nn.init.normal_(self.answer_init, mean=0.0, std=1.0)
        nn.init.normal_(self.scratch_init, mean=0.0, std=1.0)

    def _block(self, hidden: torch.Tensor, injection: torch.Tensor) -> torch.Tensor:
        hidden = hidden + injection
        cos_sin = self.rotary.values(hidden.shape[1])
        for layer in self.layers:
            hidden = layer(hidden, cos_sin)
        return hidden

    def initial_state(self, injection: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, length, _ = injection.shape
        answer = self.answer_init.to(injection).view(1, 1, -1).expand(batch, length, -1).clone()
        scratch = self.scratch_init.to(injection).view(1, 1, -1).expand(batch, length, -1).clone()
        return answer, scratch

    def _cycle(
        self,
        answer: torch.Tensor,
        scratch: torch.Tensor,
        injection: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for _ in range(self.inner_loops):
            scratch = self._block(scratch, answer + injection)
        answer = self._block(answer, scratch)
        return answer, scratch

    def forward(
        self,
        injection: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        answer, scratch = self.initial_state(injection) if state is None else state
        warmup = max(self.outer_loops - 1, 0)
        if warmup:
            if self.no_grad_warmup:
                with torch.no_grad():
                    for _ in range(warmup):
                        answer, scratch = self._cycle(answer, scratch, injection)
            else:
                for _ in range(warmup):
                    answer, scratch = self._cycle(answer, scratch, injection)
        answer, scratch = self._cycle(answer, scratch, injection)
        return answer, (answer, scratch)


def reset_parameters(module: nn.Module) -> None:
    """LeCun-style initialization for linear and embedding weights."""

    for child in module.modules():
        if isinstance(child, nn.Linear):
            nn.init.trunc_normal_(child.weight, std=1.0 / math.sqrt(child.in_features))
            if child.bias is not None:
                nn.init.zeros_(child.bias)
        elif isinstance(child, nn.Embedding):
            nn.init.trunc_normal_(child.weight, std=1.0 / math.sqrt(child.embedding_dim))
