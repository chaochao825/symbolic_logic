"""Configuration contracts for the USRL paper-spec reproduction.

The values returned by :meth:`USRLConfig.paper` are transcribed from the
CVPR 2026 paper and its supplementary material.  Values that the paper does
not specify are kept explicit so that future official code can replace the
current assumptions without changing the model API.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class USRLConfig:
    """Architecture and reasoning-loop configuration.

    Paper-specified defaults:
      * hidden_size=368, num_heads=8
      * two Transformer blocks in both UM and SM
      * four inner loops and three outer loops
      * four query tokens
      * at most 16 understand/solve steps
      * contrastive temperature 0.07 and weight 0.5

    Reproduction assumptions:
      * TRM-style RoPE, RMSNorm, and SwiGLU blocks
      * vocabulary uses PAD=0, EOS=1, ARC colours=2..11
      * outer warm-up cycles use no gradients, matching TRM
      * the recurrent state is detached between supervised reasoning steps
    """

    max_grid_size: int = 30
    vocab_size: int = 12
    pad_token_id: int = 0
    eos_token_id: int = 1
    mask_token_id: int = 12

    hidden_size: int = 368
    num_heads: int = 8
    um_layers: int = 2
    sm_layers: int = 2
    inner_loops: int = 4
    outer_loops: int = 3
    query_tokens: int = 4
    max_demos: int = 5
    max_reasoning_steps: int = 16

    expansion: float = 4.0
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10_000.0
    contrastive_temperature: float = 0.07
    contrastive_weight: float = 0.5
    detach_between_steps: bool = True
    no_grad_warmup_cycles: bool = True

    @property
    def grid_tokens(self) -> int:
        return self.max_grid_size * self.max_grid_size

    @property
    def pair_embedding_size(self) -> int:
        if self.hidden_size % 2:
            raise ValueError("hidden_size must be even for paired grid embeddings")
        return self.hidden_size // 2

    @property
    def max_sequence_length(self) -> int:
        um_length = self.grid_tokens + self.query_tokens
        sm_length = self.grid_tokens + (self.max_demos + 1) * self.query_tokens
        return max(um_length, sm_length)

    def validate(self) -> None:
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        head_dim = self.hidden_size // self.num_heads
        if head_dim % 2:
            raise ValueError("RoPE requires an even attention head dimension")
        if self.vocab_size <= 11:
            raise ValueError("vocab_size must represent PAD, EOS, and ten ARC colours")
        if min(self.inner_loops, self.outer_loops, self.max_reasoning_steps) < 1:
            raise ValueError("all recurrence depths must be positive")
        if self.max_demos < 2:
            raise ValueError("contrastive rule learning requires at least two demos")
        _ = self.pair_embedding_size

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def paper(cls) -> "USRLConfig":
        config = cls()
        config.validate()
        return config

    @classmethod
    def smoke(cls) -> "USRLConfig":
        config = cls(
            max_grid_size=6,
            hidden_size=64,
            num_heads=4,
            um_layers=1,
            sm_layers=1,
            inner_loops=1,
            outer_loops=1,
            query_tokens=2,
            max_demos=3,
            max_reasoning_steps=2,
            expansion=2.0,
            no_grad_warmup_cycles=False,
        )
        config.validate()
        return config

    @classmethod
    def pilot(cls, *, discrete_mask: bool = False) -> "USRLConfig":
        """Compute-bounded 30x30 profile; never label its score as paper-scale."""

        config = cls(
            max_grid_size=30,
            vocab_size=13 if discrete_mask else 12,
            hidden_size=96,
            num_heads=4,
            um_layers=1,
            sm_layers=1,
            inner_loops=1,
            outer_loops=1,
            query_tokens=2,
            max_demos=5,
            max_reasoning_steps=2,
            expansion=2.0,
            no_grad_warmup_cycles=False,
        )
        config.validate()
        return config
