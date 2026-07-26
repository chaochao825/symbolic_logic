#!/usr/bin/env python3
"""Reproducible audit and bounded experiment driver for USRL.

This script intentionally separates three claims:
  1. ``paper-audit`` verifies architecture/parameter contracts only.
  2. ``protocol-audit`` materializes strict and paper-transductive data receipts.
  3. ``overfit-smoke`` tests whether the loop is trainable on fixed synthetic rules.

None of these bounded checks is presented as a reproduction of 47.2%.
"""

# ruff: noqa: E402 -- the source-tree CLI bootstraps `src/` before local imports.

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import time
from dataclasses import replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from afts_arc.usrl.config import USRLConfig
from afts_arc.usrl.data import (
    ARCEpisode,
    ARCPair,
    augment_episode,
    build_training_episodes,
    collate_episodes,
    decode_grid,
)
from afts_arc.usrl.losses import usrl_loss
from afts_arc.usrl.model import USRLModel
from afts_arc.usrl.protocol import audit_protocol, write_json


def environment_receipt(device: torch.device) -> dict:
    receipt = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
    }
    if device.type == "cuda":
        receipt["gpu"] = torch.cuda.get_device_name(device)
        receipt["gpu_memory_bytes"] = torch.cuda.get_device_properties(device).total_memory
    return receipt


def paper_audit(output: Path) -> None:
    config = USRLConfig.paper()
    model = USRLModel(config)
    payload = {
        "status": "architecture_contract_only",
        "config": config.to_dict(),
        "trainable_parameters": model.trainable_parameter_count,
        "paper_reported_parameters_rounded": 7_000_000,
        "parameter_ratio_to_paper_rounded": model.trainable_parameter_count / 7_000_000,
        "shape_contract_three_demos": {
            "um_input": [3, config.grid_tokens + config.query_tokens, config.hidden_size],
            "raw_rules": [3, config.query_tokens, config.hidden_size],
            "gated_rule_tokens": [(3 + 1) * config.query_tokens, config.hidden_size],
            "sm_state": [
                config.grid_tokens + (3 + 1) * config.query_tokens,
                config.hidden_size,
            ],
        },
        "warning": "No official USRL code/checkpoint was available; matching parameter and tensor contracts does not validate the omitted training details.",
    }
    write_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


def _random_grid(rng: random.Random, source_colour: int) -> list[list[int]]:
    grid = [[0 for _ in range(4)] for _ in range(4)]
    for _ in range(5):
        grid[rng.randrange(4)][rng.randrange(4)] = source_colour
    return grid


def _map_colour(grid: list[list[int]], source: int, target: int) -> list[list[int]]:
    return [[target if value == source else value for value in row] for row in grid]


def synthetic_episodes(seed: int, count: int = 8) -> list[ARCEpisode]:
    rng = random.Random(seed)
    episodes = []
    for index in range(count):
        source = 1 + index % 4
        target = 5 + index % 4
        demos = []
        for _ in range(3):
            inp = _random_grid(rng, source)
            demos.append(ARCPair(inp, _map_colour(inp, source, target)))
        problem = _random_grid(rng, source)
        episodes.append(
            ARCEpisode(
                task_id=f"synthetic-recolor-{index}",
                demonstrations=tuple(demos),
                problem=problem,
                target=_map_colour(problem, source, target),
                source_split="synthetic-fixed",
            )
        )
    return episodes


def _quality(logits: torch.Tensor, target: torch.Tensor, pad_token_id: int) -> dict[str, float]:
    prediction = logits.argmax(dim=-1)
    valid = target.ne(pad_token_id)
    cell = ((prediction == target) & valid).sum() / valid.sum().clamp_min(1)
    exact = ((prediction == target) | ~valid).all(dim=-1).float().mean()
    return {"valid_cell_accuracy": float(cell), "valid_grid_exact": float(exact)}


def overfit_smoke(output: Path, device_name: str, steps: int, seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    config = USRLConfig.smoke()
    model = USRLModel(config).to(device)
    batch = {key: value.to(device) for key, value in collate_episodes(
        synthetic_episodes(seed), config
    ).items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    depths = torch.arange(batch["problem"].shape[0], device=device) % config.max_reasoning_steps + 1

    model.eval()
    with torch.no_grad():
        initial = model(
            batch["demo_inputs"],
            batch["demo_outputs"],
            batch["problem"],
            demo_mask=batch["demo_mask"],
            reasoning_steps=config.max_reasoning_steps,
        )
        initial_losses = usrl_loss(
            initial,
            batch["target"],
            batch["demo_mask"],
            pad_token_id=config.pad_token_id,
            contrastive_temperature=config.contrastive_temperature,
            contrastive_weight=config.contrastive_weight,
            reasoning_depths=depths,
        )
        initial_quality = _quality(initial.logits[:, -1], batch["target"], config.pad_token_id)

    started = time.perf_counter()
    trace = []
    model.train()
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        result = model(
            batch["demo_inputs"],
            batch["demo_outputs"],
            batch["problem"],
            demo_mask=batch["demo_mask"],
            reasoning_steps=config.max_reasoning_steps,
        )
        losses = usrl_loss(
            result,
            batch["target"],
            batch["demo_mask"],
            pad_token_id=config.pad_token_id,
            contrastive_temperature=config.contrastive_temperature,
            contrastive_weight=config.contrastive_weight,
            reasoning_depths=depths,
        )
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step == steps or step % max(steps // 5, 1) == 0:
            trace.append(
                {
                    "step": step,
                    "loss": float(losses["loss"].detach()),
                    "cross_entropy": float(losses["cross_entropy"].detach()),
                    "contrastive": float(losses["contrastive"].detach()),
                }
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    model.eval()
    with torch.no_grad():
        final = model(
            batch["demo_inputs"],
            batch["demo_outputs"],
            batch["problem"],
            demo_mask=batch["demo_mask"],
            reasoning_steps=config.max_reasoning_steps,
            adaptive_halting=True,
        )
        final_losses = usrl_loss(
            final,
            batch["target"],
            batch["demo_mask"],
            pad_token_id=config.pad_token_id,
            contrastive_temperature=config.contrastive_temperature,
            contrastive_weight=config.contrastive_weight,
            reasoning_depths=depths,
        )
        final_quality = _quality(final.logits[:, -1], batch["target"], config.pad_token_id)

    payload = {
        "status": "bounded_synthetic_overfit_smoke",
        "seed": seed,
        "steps": steps,
        "elapsed_seconds": elapsed,
        "environment": environment_receipt(device),
        "config": config.to_dict(),
        "trainable_parameters": model.trainable_parameter_count,
        "initial": {
            "loss": float(initial_losses["loss"]),
            **initial_quality,
        },
        "final": {
            "loss": float(final_losses["loss"]),
            **final_quality,
            "mean_adaptive_halt_step": float(final.halted_step.float().mean()),
        },
        "trace": trace,
        "claim_boundary": "This verifies optimization and tensor flow on eight fixed synthetic recolouring episodes; it is not ARC accuracy evidence.",
    }
    write_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


def _evaluate_arc(
    model: USRLModel,
    episodes: list[ARCEpisode],
    config: USRLConfig,
    device: torch.device,
    batch_size: int,
    curriculum: str,
) -> dict:
    model.eval()
    final_exact = 0
    adaptive_exact = 0
    final_valid_correct = 0
    adaptive_valid_correct = 0
    valid_total = 0
    foreground_correct = 0
    foreground_total = 0
    background_correct = 0
    background_total = 0
    eos_correct = 0
    eos_total = 0
    shape_correct = 0
    residual_cells: list[int] = []
    adaptive_by_task: dict[str, list[bool]] = {}
    solved_task_ids: list[str] = []
    halting_steps = []
    with torch.inference_mode():
        for start in range(0, len(episodes), batch_size):
            current_episodes = episodes[start : start + batch_size]
            batch = {
                key: value.to(device)
                for key, value in collate_episodes(current_episodes, config).items()
            }
            context = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if device.type == "cuda"
                else torch.autocast("cpu", enabled=False)
            )
            with context:
                result = model(
                    batch["demo_inputs"],
                    batch["demo_outputs"],
                    batch["problem"],
                    demo_mask=batch["demo_mask"],
                    initial_answer=(
                        torch.full_like(batch["problem"], config.mask_token_id)
                        if curriculum == "drm-cosine-mask"
                        else None
                    ),
                    reasoning_steps=config.max_reasoning_steps,
                    adaptive_halting=True,
                )
            prediction = result.predictions[:, -1]
            row_indices = torch.arange(prediction.shape[0], device=device)
            adaptive_prediction = result.predictions[
                row_indices, result.halted_step.clamp_min(1) - 1
            ]
            target = batch["target"]
            valid = target.ne(config.pad_token_id)
            final_valid_correct += int(((prediction == target) & valid).sum())
            adaptive_valid_correct += int(((adaptive_prediction == target) & valid).sum())
            valid_total += int(valid.sum())
            foreground = target.ge(3) & target.le(11)
            background = target.eq(2)
            eos = target.eq(config.eos_token_id)
            foreground_correct += int(((adaptive_prediction == target) & foreground).sum())
            foreground_total += int(foreground.sum())
            background_correct += int(((adaptive_prediction == target) & background).sum())
            background_total += int(background.sum())
            eos_correct += int(((adaptive_prediction == target) & eos).sum())
            eos_total += int(eos.sum())
            for final_row, adaptive_row, episode, valid_row, target_row in zip(
                prediction,
                adaptive_prediction,
                current_episodes,
                valid,
                target,
            ):
                final_grid = decode_grid(final_row, config)
                adaptive_grid = decode_grid(adaptive_row, config)
                final_ok = final_grid == episode.target
                adaptive_ok = adaptive_grid == episode.target
                final_exact += final_ok
                adaptive_exact += adaptive_ok
                adaptive_by_task.setdefault(episode.task_id, []).append(adaptive_ok)
                target_shape = (len(episode.target), len(episode.target[0]))
                predicted_shape = (len(adaptive_grid), len(adaptive_grid[0]))
                shape_correct += predicted_shape == target_shape
                residual_cells.append(
                    int(((adaptive_row != target_row) & valid_row).sum().item())
                )
            halting_steps.extend(result.halted_step.cpu().tolist())
    for task_id, outcomes in adaptive_by_task.items():
        if outcomes and all(outcomes):
            solved_task_ids.append(task_id)
    task_weighted = sum(
        sum(outcomes) / len(outcomes) for outcomes in adaptive_by_task.values()
    ) / max(len(adaptive_by_task), 1)
    residual_sorted = sorted(residual_cells)
    median_residual = (
        residual_sorted[len(residual_sorted) // 2] if residual_sorted else 0
    )
    return {
        "episode_count": len(episodes),
        "task_count": len(adaptive_by_task),
        "fixed_final_exact_accuracy": final_exact / max(len(episodes), 1),
        "adaptive_exact_accuracy": adaptive_exact / max(len(episodes), 1),
        "adaptive_task_weighted_accuracy": task_weighted,
        "adaptive_valid_cell_accuracy": adaptive_valid_correct / max(valid_total, 1),
        "fixed_final_valid_cell_accuracy": final_valid_correct / max(valid_total, 1),
        "adaptive_foreground_cell_accuracy": foreground_correct / max(foreground_total, 1),
        "adaptive_background_cell_accuracy": background_correct / max(background_total, 1),
        "adaptive_eos_accuracy": eos_correct / max(eos_total, 1),
        "adaptive_shape_accuracy": shape_correct / max(len(episodes), 1),
        "adaptive_mean_residual_cells": sum(residual_cells) / max(len(residual_cells), 1),
        "adaptive_median_residual_cells": median_residual,
        "adaptive_within_5_residual_cells": sum(value <= 5 for value in residual_cells)
        / max(len(residual_cells), 1),
        "solved_task_ids": solved_task_ids,
        "mean_adaptive_halt_step": sum(halting_steps) / max(len(halting_steps), 1),
    }


def _cosine_mask_corruption(
    target: torch.Tensor,
    config: USRLConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """DRM-inspired corruption only; this is not the full iterative DRM sampler."""

    tau = torch.rand(target.shape[0], device=target.device)
    offset = 0.008
    scaled = ((tau + offset) / (1.0 + offset)).clamp(0.0, 1.0)
    mask_ratio = 1.0 - torch.cos(0.5 * torch.pi * scaled).square()
    mask = torch.rand_like(target, dtype=torch.float32) < mask_ratio[:, None]
    corrupted = torch.where(mask, config.mask_token_id, target)
    return corrupted, mask_ratio


def arc_pilot(
    output: Path,
    arc_root: Path,
    protocol: str,
    device_name: str,
    steps: int,
    batch_size: int,
    seed: int,
    contrastive_weight: float,
    curriculum: str,
    inner_loops: int,
    outer_loops: int,
) -> None:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    if curriculum not in {"none", "drm-cosine-mask"}:
        raise ValueError("unsupported curriculum")
    config = replace(
        USRLConfig.pilot(discrete_mask=curriculum == "drm-cosine-mask"),
        inner_loops=inner_loops,
        outer_loops=outer_loops,
        no_grad_warmup_cycles=outer_loops > 1,
    )
    config.validate()
    fit, validation = build_training_episodes(arc_root, protocol=protocol)
    fit_by_task: dict[str, list[ARCEpisode]] = {}
    for episode in fit:
        fit_by_task.setdefault(episode.task_id, []).append(episode)
    task_ids = sorted(fit_by_task)
    if batch_size > len(task_ids):
        raise ValueError("batch_size exceeds the number of fit tasks")
    model = USRLModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    trace = []
    started = time.perf_counter()
    model.train()
    for step in range(1, steps + 1):
        sampled_task_ids = rng.sample(task_ids, batch_size)
        sampled = [
            augment_episode(
                fit_by_task[task_id][rng.randrange(len(fit_by_task[task_id]))], rng
            )
            for task_id in sampled_task_ids
        ]
        batch = {
            key: value.to(device)
            for key, value in collate_episodes(
                sampled, config, rng=rng, random_translation=True
            ).items()
        }
        depths = torch.tensor(
            [1 + rng.randrange(config.max_reasoning_steps) for _ in sampled], device=device
        )
        optimizer.zero_grad(set_to_none=True)
        context = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if device.type == "cuda"
            else torch.autocast("cpu", enabled=False)
        )
        with context:
            initial_answer = None
            mean_mask_ratio = 0.0
            if curriculum == "drm-cosine-mask":
                initial_answer, sampled_mask_ratios = _cosine_mask_corruption(
                    batch["target"], config
                )
                mean_mask_ratio = float(sampled_mask_ratios.mean())
            result = model(
                batch["demo_inputs"],
                batch["demo_outputs"],
                batch["problem"],
                demo_mask=batch["demo_mask"],
                initial_answer=initial_answer,
                reasoning_steps=config.max_reasoning_steps,
            )
            losses = usrl_loss(
                result,
                batch["target"],
                batch["demo_mask"],
                pad_token_id=config.pad_token_id,
                contrastive_temperature=config.contrastive_temperature,
                contrastive_weight=contrastive_weight,
                reasoning_depths=depths,
            )
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step == steps or step % max(steps // 10, 1) == 0:
            trace.append(
                {
                    "step": step,
                    "loss": float(losses["loss"].detach()),
                    "cross_entropy": float(losses["cross_entropy"].detach()),
                    "contrastive": float(losses["contrastive"].detach()),
                    "mean_sampled_mask_ratio": mean_mask_ratio,
                }
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter() - started
    metrics = _evaluate_arc(
        model, validation, config, device, batch_size, curriculum
    )
    payload = {
        "status": "compute_bounded_arc_pilot",
        "protocol": protocol,
        "seed": seed,
        "steps": steps,
        "batch_size": batch_size,
        "optimizer_updates": steps,
        "episodes_seen_with_replacement": steps * batch_size,
        "fit_episode_count": len(fit),
        "fit_task_count": len(task_ids),
        "validation_episode_count": len(validation),
        "contrastive_weight": contrastive_weight,
        "curriculum": curriculum,
        "train_seconds": train_seconds,
        "environment": environment_receipt(device),
        "config": config.to_dict(),
        "trainable_parameters": model.trainable_parameter_count,
        "sampler": "uniform_without_replacement_over_tasks_then_uniform_leave-one-out_episode",
        "augmentation": "random D4 + non-background colour permutation + shared pair translation",
        "metric_contract": {
            "background": "ARC colour 0 / black token only; not an inferred modal background",
            "foreground": "ARC colours 1..9 / non-black tokens",
            "valid": "all non-PAD target tokens, including EOS",
        },
        "curriculum_contract": (
            "cosine training corruption with all-MASK initialization at evaluation; "
            "the official DRM multi-timestep remask/reinjection sampler is not implemented"
            if curriculum == "drm-cosine-mask"
            else "direct target prediction from a PAD draft"
        ),
        "metrics": metrics,
        "trace": trace,
        "claim_boundary": (
            "This 30x30 pilot uses width 96 and one block in each of UM/SM, "
            f"configured for {inner_loops}x{outer_loops} inner/outer recurrence, "
            "for a small number of optimizer updates. The cosine-mask option "
            "does not implement DRM's iterative inference sampler. This is a "
            "pipeline diagnostic, not a 7M/100k-epoch USRL or DRM reproduction."
        ),
    }
    write_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    paper = subparsers.add_parser("paper-audit")
    paper.add_argument("--output", type=Path, required=True)

    protocol = subparsers.add_parser("protocol-audit")
    protocol.add_argument("--arc-root", type=Path, required=True)
    protocol.add_argument("--output", type=Path, required=True)

    smoke = subparsers.add_parser("overfit-smoke")
    smoke.add_argument("--output", type=Path, required=True)
    smoke.add_argument("--device", default="auto")
    smoke.add_argument("--steps", type=int, default=100)
    smoke.add_argument("--seed", type=int, default=20260726)

    pilot = subparsers.add_parser("arc-pilot")
    pilot.add_argument("--arc-root", type=Path, required=True)
    pilot.add_argument("--protocol", choices=("strict", "paper-transductive"), required=True)
    pilot.add_argument("--output", type=Path, required=True)
    pilot.add_argument("--device", default="auto")
    pilot.add_argument("--steps", type=int, default=300)
    pilot.add_argument("--batch-size", type=int, default=8)
    pilot.add_argument("--seed", type=int, default=20260726)
    pilot.add_argument("--contrastive-weight", type=float, default=0.5)
    pilot.add_argument(
        "--curriculum", choices=("none", "drm-cosine-mask"), default="none"
    )
    pilot.add_argument("--inner-loops", type=int, default=1)
    pilot.add_argument("--outer-loops", type=int, default=1)

    args = parser.parse_args()
    if args.command == "paper-audit":
        paper_audit(args.output)
    elif args.command == "protocol-audit":
        payload = audit_protocol(args.arc_root)
        write_json(args.output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif args.command == "overfit-smoke":
        overfit_smoke(args.output, args.device, args.steps, args.seed)
    else:
        arc_pilot(
            args.output,
            args.arc_root,
            args.protocol,
            args.device,
            args.steps,
            args.batch_size,
            args.seed,
            args.contrastive_weight,
            args.curriculum,
            args.inner_loops,
            args.outer_loops,
        )


if __name__ == "__main__":
    main()
