"""Paper-spec implementation of the USRL understand/solve loop.

The public paper fixes the main tensor flow but omits the gate equation.  The
gate below is therefore deliberately isolated: it uses the current solution
summary and answer-rule representation to scale each cached demonstration rule
token, then concatenates the current answer rule.  For three demonstrations and
four query tokens this produces the paper's 16 rule tokens and 916-token SM
state exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .config import USRLConfig
from .layers import RecurrentTransformer, reset_parameters


@dataclass
class USRLOutput:
    logits: torch.Tensor
    predictions: torch.Tensor
    raw_rule_representations: torch.Tensor
    answer_rule_representations: torch.Tensor
    intra_consistency: torch.Tensor
    answer_consistency: torch.Tensor
    halted_step: torch.Tensor


def _masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    weights = mask.to(values.dtype)
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(-1)
    numerator = (values * weights).sum(dim=dim)
    denominator = weights.sum(dim=dim).clamp_min(1.0)
    return numerator / denominator


class UnderstandingModule(nn.Module):
    def __init__(self, config: USRLConfig, token_embedding: nn.Embedding) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = token_embedding
        self.query = nn.Parameter(torch.zeros(config.query_tokens, config.hidden_size))
        self.recurrence = RecurrentTransformer(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            layers=config.um_layers,
            inner_loops=config.inner_loops,
            outer_loops=config.outer_loops,
            expansion=config.expansion,
            norm_eps=config.rms_norm_eps,
            max_length=config.max_sequence_length,
            rope_theta=config.rope_theta,
            no_grad_warmup=config.no_grad_warmup_cycles,
        )

    def pair_embeddings(self, inputs: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        input_emb = self.token_embedding(inputs)
        output_emb = self.token_embedding(outputs)
        return torch.cat((input_emb, output_emb), dim=-1)

    def forward(self, inputs: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        """Encode pairs shaped ``[B, M, L]`` into ``[B, M, Q, D]``."""

        if inputs.shape != outputs.shape or inputs.ndim != 3:
            raise ValueError("UM expects equally-shaped [batch, demos, grid_tokens] tensors")
        batch, demos, _ = inputs.shape
        pairs = self.pair_embeddings(inputs, outputs).flatten(0, 1)
        queries = self.query.to(pairs).unsqueeze(0).expand(batch * demos, -1, -1)
        injection = torch.cat((queries, pairs), dim=1)
        hidden, _ = self.recurrence(injection)
        return hidden[:, : self.config.query_tokens].view(
            batch, demos, self.config.query_tokens, self.config.hidden_size
        )


class RuleGate(nn.Module):
    """Paper-compatible scalar gate with an explicit ambiguity boundary."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.raw_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.context_projection = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.scalar = nn.Linear(hidden_size, 1, bias=True)

    def forward(
        self,
        raw_rules: torch.Tensor,
        answer_rule: torch.Tensor,
        demo_mask: torch.Tensor,
        solution_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # raw_rules [B,M,Q,D], answer_rule [B,1,Q,D]
        batch, demos, queries, hidden = raw_rules.shape
        answer_summary = answer_rule.mean(dim=(1, 2))
        state_summary = (
            raw_rules.new_zeros(batch, hidden)
            if solution_state is None
            else solution_state.mean(dim=1)
        )
        context = torch.tanh(
            self.context_projection(torch.cat((state_summary, answer_summary), dim=-1))
        )
        gate = torch.sigmoid(
            self.scalar(torch.tanh(self.raw_projection(raw_rules) + context[:, None, None, :]))
        )
        gate = gate * demo_mask[:, :, None, None].to(gate.dtype)
        gated_rules = raw_rules * gate
        rule_tokens = torch.cat(
            (gated_rules.reshape(batch, demos * queries, hidden), answer_rule[:, 0]), dim=1
        )
        return rule_tokens, gate.squeeze(-1)


class SolvingModule(nn.Module):
    def __init__(self, config: USRLConfig, token_embedding: nn.Embedding) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = token_embedding
        self.recurrence = RecurrentTransformer(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            layers=config.sm_layers,
            inner_loops=config.inner_loops,
            outer_loops=config.outer_loops,
            expansion=config.expansion,
            norm_eps=config.rms_norm_eps,
            max_length=config.max_sequence_length,
            rope_theta=config.rope_theta,
            no_grad_warmup=config.no_grad_warmup_cycles,
        )

    def pair_embeddings(self, problem: torch.Tensor, draft: torch.Tensor) -> torch.Tensor:
        return torch.cat((self.token_embedding(problem), self.token_embedding(draft)), dim=-1)

    def forward(
        self,
        problem: torch.Tensor,
        draft: torch.Tensor,
        rule_tokens: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        injection = torch.cat((self.pair_embeddings(problem, draft), rule_tokens), dim=1)
        return self.recurrence(injection, state)


class USRLModel(nn.Module):
    """Understanding and Solving Reasoning Loop.

    Inputs are already padded/tokenized.  ``demo_inputs`` and ``demo_outputs``
    have shape ``[B,M,L]``; ``problem`` has shape ``[B,L]``.  ``demo_mask`` is
    true for real demonstrations and false for batch padding.
    """

    def __init__(self, config: USRLConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.pair_embedding_size)
        self.understanding = UnderstandingModule(config, self.token_embedding)
        self.gate = RuleGate(config.hidden_size)
        self.solving = SolvingModule(config, self.token_embedding)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        reset_parameters(self)
        # The paper explicitly initializes query tokens to zero.
        nn.init.zeros_(self.understanding.query)

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    @staticmethod
    def _intra_consistency(raw_rules: torch.Tensor, demo_mask: torch.Tensor) -> torch.Tensor:
        pooled = F.normalize(raw_rules.mean(dim=2).float(), dim=-1)
        similarities = torch.einsum("bmd,bnd->bmn", pooled, pooled)
        valid = demo_mask[:, :, None] & demo_mask[:, None, :]
        diagonal = torch.eye(raw_rules.shape[1], dtype=torch.bool, device=raw_rules.device)
        valid = valid & ~diagonal.unsqueeze(0)
        numerator = (similarities * valid).sum(dim=(1, 2))
        denominator = valid.sum(dim=(1, 2)).clamp_min(1)
        return numerator / denominator

    @staticmethod
    def _answer_consistency(
        answer_rule: torch.Tensor,
        raw_rules: torch.Tensor,
        demo_mask: torch.Tensor,
    ) -> torch.Tensor:
        answer = F.normalize(answer_rule.mean(dim=(1, 2)).float(), dim=-1)
        demos = F.normalize(raw_rules.mean(dim=2).float(), dim=-1)
        similarities = torch.einsum("bd,bmd->bm", answer, demos)
        return _masked_mean(similarities, demo_mask, dim=1)

    def forward(
        self,
        demo_inputs: torch.Tensor,
        demo_outputs: torch.Tensor,
        problem: torch.Tensor,
        *,
        demo_mask: torch.Tensor | None = None,
        initial_answer: torch.Tensor | None = None,
        reasoning_steps: int | None = None,
        adaptive_halting: bool = False,
    ) -> USRLOutput:
        if problem.ndim != 2 or problem.shape[-1] != self.config.grid_tokens:
            raise ValueError("problem must have shape [batch, config.grid_tokens]")
        if demo_inputs.shape[:1] != problem.shape[:1]:
            raise ValueError("demo and problem batch sizes differ")
        batch, demos, length = demo_inputs.shape
        if demos > self.config.max_demos or length != self.config.grid_tokens:
            raise ValueError("demo count or grid length exceeds configuration")
        if demo_mask is None:
            demo_mask = torch.ones(batch, demos, dtype=torch.bool, device=problem.device)
        if demo_mask.shape != (batch, demos):
            raise ValueError("demo_mask must have shape [batch, demos]")

        steps = self.config.max_reasoning_steps if reasoning_steps is None else reasoning_steps
        if not 1 <= steps <= self.config.max_reasoning_steps:
            raise ValueError("reasoning_steps is outside the configured range")

        raw_rules = self.understanding(demo_inputs, demo_outputs)
        intra = self._intra_consistency(raw_rules, demo_mask)
        draft = (
            torch.full_like(problem, self.config.pad_token_id)
            if initial_answer is None
            else initial_answer
        )
        state: tuple[torch.Tensor, torch.Tensor] | None = None
        logits_steps: list[torch.Tensor] = []
        prediction_steps: list[torch.Tensor] = []
        answer_rule_steps: list[torch.Tensor] = []
        consistency_steps: list[torch.Tensor] = []
        halted_step = torch.full((batch,), steps, dtype=torch.long, device=problem.device)
        active = torch.ones(batch, dtype=torch.bool, device=problem.device)
        answer_rule = self.understanding(problem[:, None, :], draft[:, None, :])

        for step in range(steps):
            solution_summary = None if state is None else state[0]
            rule_tokens, _ = self.gate(raw_rules, answer_rule, demo_mask, solution_summary)
            solution, state = self.solving(problem, draft, rule_tokens, state)
            logits = self.decoder(solution[:, : self.config.grid_tokens])
            prediction = logits.argmax(dim=-1)
            # Algorithm 1 line 14: the newly generated draft is passed through
            # UM before answer/example consistency is measured.
            answer_rule = self.understanding(problem[:, None, :], prediction[:, None, :])
            answer_consistency = self._answer_consistency(answer_rule, raw_rules, demo_mask)

            logits_steps.append(logits)
            prediction_steps.append(prediction)
            answer_rule_steps.append(answer_rule[:, 0])
            consistency_steps.append(answer_consistency)

            if adaptive_halting:
                newly_halted = active & (answer_consistency > intra)
                halted_step = torch.where(newly_halted, step + 1, halted_step)
                active = active & ~newly_halted

            draft = prediction.detach()
            if self.config.detach_between_steps:
                state = tuple(item.detach() for item in state)

        return USRLOutput(
            logits=torch.stack(logits_steps, dim=1),
            predictions=torch.stack(prediction_steps, dim=1),
            raw_rule_representations=raw_rules,
            answer_rule_representations=torch.stack(answer_rule_steps, dim=1),
            intra_consistency=intra,
            answer_consistency=torch.stack(consistency_steps, dim=1),
            halted_step=halted_step,
        )
