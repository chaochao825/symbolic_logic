"""USRL deep-supervision and rule-isomorphism losses."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from .model import USRLOutput


def supervised_contrastive_rule_loss(
    rules: torch.Tensor,
    demo_mask: torch.Tensor,
    *,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Supervised InfoNCE over demo rules, with same-task positives.

    ``rules`` is ``[B,M,Q,D]``. Query tokens are mean pooled before cosine
    similarity; padded demonstrations are removed. Every other demonstration
    from the same batch item is a positive and all other batch items provide
    negatives, matching the paper's definition.
    """

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    batch, demos, _, hidden = rules.shape
    pooled = F.normalize(rules.mean(dim=2).float(), dim=-1).reshape(batch * demos, hidden)
    valid = demo_mask.reshape(-1)
    pooled = pooled[valid]
    task_ids = torch.arange(batch, device=rules.device)[:, None].expand(batch, demos).reshape(-1)
    task_ids = task_ids[valid]
    if pooled.shape[0] < 2:
        return rules.sum() * 0.0

    logits = pooled @ pooled.T / temperature
    self_mask = torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
    positives = task_ids[:, None].eq(task_ids[None, :]) & ~self_mask
    usable = positives.any(dim=1)
    if not usable.any():
        return rules.sum() * 0.0

    logits = logits.masked_fill(self_mask, float("-inf"))
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positive_count = positives.sum(dim=1).clamp_min(1)
    per_anchor = -(log_prob.masked_fill(~positives, 0.0).sum(dim=1) / positive_count)
    return per_anchor[usable].mean()


def usrl_loss(
    output: USRLOutput,
    target: torch.Tensor,
    demo_mask: torch.Tensor,
    *,
    pad_token_id: int = 0,
    contrastive_temperature: float = 0.07,
    contrastive_weight: float = 0.5,
    reasoning_depths: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compute per-step CE plus the UM rule-isomorphism objective.

    ``reasoning_depths`` implements the paper's variable-depth supervision at
    sample granularity. A value of ``d`` supervises steps ``0..d-1`` for that
    sample. Dynamic batch replacement is a training-loop concern and is kept
    outside this pure loss function.
    """

    batch, steps, length, vocab = output.logits.shape
    if target.shape != (batch, length):
        raise ValueError("target shape does not match model output")
    expanded_target = target[:, None, :].expand(batch, steps, length)
    token_losses = F.cross_entropy(
        output.logits.reshape(batch * steps * length, vocab),
        expanded_target.reshape(-1),
        ignore_index=pad_token_id,
        reduction="none",
    ).view(batch, steps, length)
    valid_tokens = expanded_target.ne(pad_token_id)
    per_sample_step = (token_losses * valid_tokens).sum(dim=-1) / valid_tokens.sum(dim=-1).clamp_min(1)

    if reasoning_depths is None:
        step_mask = torch.ones(batch, steps, dtype=torch.bool, device=target.device)
    else:
        if reasoning_depths.shape != (batch,):
            raise ValueError("reasoning_depths must have shape [batch]")
        indices = torch.arange(steps, device=target.device)[None, :]
        step_mask = indices < reasoning_depths[:, None]
    ce = (per_sample_step * step_mask).sum() / step_mask.sum().clamp_min(1)
    contrastive = supervised_contrastive_rule_loss(
        output.raw_rule_representations,
        demo_mask,
        temperature=contrastive_temperature,
    )
    total = ce + contrastive_weight * contrastive
    return {"loss": total, "cross_entropy": ce, "contrastive": contrastive}
