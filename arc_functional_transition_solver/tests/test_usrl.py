from __future__ import annotations

from types import SimpleNamespace

import pytest


def _usrl_imports() -> SimpleNamespace:
    """Import the optional Torch runtime only while a USRL test is executing.

    Several existing CPU-only CLI tests assert that collecting the default test
    suite does not import Torch. Keeping these imports out of module scope
    preserves that repository-wide optional-runtime contract.
    """

    torch = pytest.importorskip("torch")
    from afts_arc.usrl.config import USRLConfig
    from afts_arc.usrl.data import (
        ARCEpisode,
        ARCPair,
        collate_episodes,
        decode_grid,
        dihedral_grid,
        encode_grid,
    )
    from afts_arc.usrl.losses import supervised_contrastive_rule_loss, usrl_loss
    from afts_arc.usrl.model import USRLModel

    return SimpleNamespace(
        torch=torch,
        USRLConfig=USRLConfig,
        ARCEpisode=ARCEpisode,
        ARCPair=ARCPair,
        collate_episodes=collate_episodes,
        decode_grid=decode_grid,
        dihedral_grid=dihedral_grid,
        encode_grid=encode_grid,
        supervised_contrastive_rule_loss=supervised_contrastive_rule_loss,
        usrl_loss=usrl_loss,
        USRLModel=USRLModel,
    )


def _episode(index: int = 0, *, api: SimpleNamespace | None = None):
    api = _usrl_imports() if api is None else api
    demos = (
        api.ARCPair([[1, 0], [0, 1]], [[2, 0], [0, 2]]),
        api.ARCPair([[0, 1], [1, 0]], [[0, 2], [2, 0]]),
        api.ARCPair([[1, 1], [0, 0]], [[2, 2], [0, 0]]),
    )
    return api.ARCEpisode(
        task_id=f"unit-{index}",
        demonstrations=demos,
        problem=[[1, 0], [1, 0]],
        target=[[2, 0], [2, 0]],
        source_split="unit",
    )


def test_paper_contract_parameter_count_is_approximately_seven_million() -> None:
    api = _usrl_imports()
    model = api.USRLModel(api.USRLConfig.paper())
    assert 6_500_000 <= model.trainable_parameter_count <= 7_500_000


def test_smoke_forward_and_losses_are_finite() -> None:
    api = _usrl_imports()
    api.torch.manual_seed(7)
    config = api.USRLConfig.smoke()
    batch = api.collate_episodes([_episode(0, api=api), _episode(1, api=api)], config)
    model = api.USRLModel(config)
    output = model(
        batch["demo_inputs"],
        batch["demo_outputs"],
        batch["problem"],
        demo_mask=batch["demo_mask"],
        reasoning_steps=2,
        adaptive_halting=True,
    )
    assert output.logits.shape == (2, 2, config.grid_tokens, config.vocab_size)
    assert output.raw_rule_representations.shape == (
        2,
        3,
        config.query_tokens,
        config.hidden_size,
    )
    losses = api.usrl_loss(
        output,
        batch["target"],
        batch["demo_mask"],
        pad_token_id=config.pad_token_id,
        reasoning_depths=api.torch.tensor([1, 2]),
    )
    assert all(api.torch.isfinite(value) for value in losses.values())


def test_contrastive_loss_prefers_clustered_same_task_rules() -> None:
    api = _usrl_imports()
    demo_mask = api.torch.ones(2, 2, dtype=api.torch.bool)
    separated = api.torch.tensor(
        [
            [[[1.0, 0.0]], [[0.9, 0.1]]],
            [[[-1.0, 0.0]], [[-0.9, -0.1]]],
        ]
    )
    shuffled = separated.clone()
    shuffled[0, 1] = separated[1, 0]
    assert api.supervised_contrastive_rule_loss(
        separated, demo_mask
    ) < api.supervised_contrastive_rule_loss(
        shuffled, demo_mask
    )


def test_grid_codec_round_trip_and_d4_closure() -> None:
    api = _usrl_imports()
    config = api.USRLConfig.smoke()
    grid = [[1, 2, 3], [4, 5, 6]]
    assert api.decode_grid(api.encode_grid(grid, config), config) == grid
    transformed = grid
    for _ in range(4):
        transformed = api.dihedral_grid(transformed, 1)
    assert transformed == grid


def test_collate_padding_mask() -> None:
    api = _usrl_imports()
    config = api.USRLConfig.smoke()
    short = _episode(0, api=api)
    short = api.ARCEpisode(
        task_id=short.task_id,
        demonstrations=short.demonstrations[:2],
        problem=short.problem,
        target=short.target,
        source_split=short.source_split,
    )
    batch = api.collate_episodes([short, _episode(1, api=api)], config)
    assert batch["demo_mask"].tolist() == [[True, True, False], [True, True, True]]


def test_adaptive_consistency_reencodes_the_new_prediction() -> None:
    """Algorithm 1 evaluates the new draft, not the previous-step draft."""

    api = _usrl_imports()
    torch = api.torch
    config = api.USRLConfig.smoke()
    batch = api.collate_episodes([_episode(0, api=api)], config)
    model = api.USRLModel(config)

    class FakeUnderstanding(torch.nn.Module):
        def forward(self, inputs: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
            values = outputs[:, :, :1].float().unsqueeze(-1)
            result = torch.zeros(
                outputs.shape[0],
                outputs.shape[1],
                config.query_tokens,
                config.hidden_size,
                device=outputs.device,
            )
            result[..., :1] = values
            return result

    class FakeSolving(torch.nn.Module):
        def forward(self, problem, draft, rule_tokens, state):
            solution = torch.zeros(
                problem.shape[0], config.grid_tokens, config.hidden_size
            )
            recurrent_state = (solution, solution.clone())
            return solution, recurrent_state

    class TokenThreeDecoder(torch.nn.Module):
        def forward(self, hidden: torch.Tensor) -> torch.Tensor:
            logits = torch.zeros(*hidden.shape[:-1], config.vocab_size)
            logits[..., 3] = 1.0
            return logits

    observed: list[float] = []

    def capture(answer_rule, raw_rules, demo_mask):
        observed.append(float(answer_rule[0, 0, 0, 0]))
        return torch.zeros(answer_rule.shape[0])

    model.understanding = FakeUnderstanding()
    model.solving = FakeSolving()
    model.decoder = TokenThreeDecoder()
    model._answer_consistency = capture
    model(
        batch["demo_inputs"],
        batch["demo_outputs"],
        batch["problem"],
        demo_mask=batch["demo_mask"],
        reasoning_steps=1,
        adaptive_halting=True,
    )
    assert observed == [3.0]
