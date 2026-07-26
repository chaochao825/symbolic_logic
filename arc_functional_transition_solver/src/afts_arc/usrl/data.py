"""ARC tokenization, augmentation, and protocol-safe episode construction."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch

from .config import USRLConfig


Grid = list[list[int]]


@dataclass(frozen=True)
class ARCPair:
    input: Grid
    output: Grid


@dataclass(frozen=True)
class ARCEpisode:
    task_id: str
    demonstrations: tuple[ARCPair, ...]
    problem: Grid
    target: Grid
    source_split: str


def _validate_grid(grid: Grid, max_size: int) -> None:
    if not grid or not grid[0]:
        raise ValueError("ARC grids must be non-empty")
    width = len(grid[0])
    if len(grid) > max_size or width > max_size:
        raise ValueError("grid exceeds configured maximum size")
    if any(len(row) != width for row in grid):
        raise ValueError("grid is ragged")
    if any(not 0 <= value <= 9 for row in grid for value in row):
        raise ValueError("ARC colours must be in [0,9]")


def encode_grid(
    grid: Grid,
    config: USRLConfig,
    *,
    top: int = 0,
    left: int = 0,
) -> torch.Tensor:
    """Encode a grid using the public TRM PAD/EOS/colour convention."""

    _validate_grid(grid, config.max_grid_size)
    height, width = len(grid), len(grid[0])
    if top < 0 or left < 0 or top + height > config.max_grid_size or left + width > config.max_grid_size:
        raise ValueError("translation places the grid outside the canvas")
    encoded = torch.full(
        (config.max_grid_size, config.max_grid_size),
        config.pad_token_id,
        dtype=torch.long,
    )
    values = torch.tensor(grid, dtype=torch.long) + 2
    encoded[top : top + height, left : left + width] = values
    if top + height < config.max_grid_size:
        encoded[top + height, left : left + width] = config.eos_token_id
    if left + width < config.max_grid_size:
        encoded[top : top + height, left + width] = config.eos_token_id
    return encoded.flatten()


def decode_grid(tokens: torch.Tensor, config: USRLConfig) -> Grid:
    """Crop a token canvas at its first EOS/PAD frontier."""

    canvas = tokens.detach().cpu().view(config.max_grid_size, config.max_grid_size)
    max_width = config.max_grid_size
    best_height = 0
    best_width = 0
    best_area = 0
    for row in range(config.max_grid_size):
        width = max_width
        for column in range(max_width):
            value = int(canvas[row, column])
            if value < 2 or value > 11:
                width = column
                break
        max_width = width
        area = (row + 1) * max_width
        if area > best_area:
            best_area = area
            best_height = row + 1
            best_width = max_width
    if best_area == 0:
        return [[0]]
    return (canvas[:best_height, :best_width] - 2).tolist()


def dihedral_grid(grid: Grid, transform: int) -> Grid:
    """Apply one of the eight D4 transformations without NumPy."""

    if transform not in range(8):
        raise ValueError("transform must be in [0,7]")
    result = [row[:] for row in grid]
    if transform >= 4:
        result = [list(reversed(row)) for row in result]
        transform -= 4
    for _ in range(transform):
        result = [list(row) for row in zip(*reversed(result))]
    return result


def recolor_grid(grid: Grid, mapping: Sequence[int]) -> Grid:
    if len(mapping) != 10 or sorted(mapping) != list(range(10)):
        raise ValueError("mapping must be a permutation of ARC colours")
    return [[mapping[value] for value in row] for row in grid]


def augment_episode(episode: ARCEpisode, rng: random.Random) -> ARCEpisode:
    transform = rng.randrange(8)
    non_background = list(range(1, 10))
    rng.shuffle(non_background)
    mapping = [0, *non_background]

    def aug(grid: Grid) -> Grid:
        return dihedral_grid(recolor_grid(grid, mapping), transform)

    return ARCEpisode(
        task_id=episode.task_id,
        demonstrations=tuple(ARCPair(aug(pair.input), aug(pair.output)) for pair in episode.demonstrations),
        problem=aug(episode.problem),
        target=aug(episode.target),
        source_split=episode.source_split,
    )


def load_arc_task(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        task = json.load(handle)
    if not isinstance(task.get("train"), list) or not isinstance(task.get("test"), list):
        raise ValueError(f"invalid ARC task: {path}")
    return task


def leave_one_out_episodes(task_id: str, task: dict, source_split: str) -> list[ARCEpisode]:
    """Turn known demonstration pairs into training episodes.

    The USRL paper does not state how an 800-task collection becomes supervised
    episodes. Leave-one-demonstration-out is the minimal non-leaking choice:
    every target pair is removed from the UM context before prediction.
    """

    pairs = tuple(ARCPair(item["input"], item["output"]) for item in task["train"])
    episodes: list[ARCEpisode] = []
    for target_index, target in enumerate(pairs):
        demonstrations = tuple(pair for index, pair in enumerate(pairs) if index != target_index)
        # Tasks with two provided examples necessarily leave only one UM
        # demonstration. They still provide a valid CE episode; the supervised
        # contrastive loss has no positive pair and skips that anchor.
        if len(demonstrations) < 1:
            continue
        episodes.append(
            ARCEpisode(
                task_id=task_id,
                demonstrations=demonstrations,
                problem=target.input,
                target=target.output,
                source_split=source_split,
            )
        )
    return episodes


def test_episodes(task_id: str, task: dict, source_split: str) -> list[ARCEpisode]:
    demonstrations = tuple(ARCPair(item["input"], item["output"]) for item in task["train"])
    episodes = []
    for item in task["test"]:
        if "output" not in item:
            continue
        episodes.append(
            ARCEpisode(
                task_id=task_id,
                demonstrations=demonstrations,
                problem=item["input"],
                target=item["output"],
                source_split=source_split,
            )
        )
    return episodes


def discover_tasks(arc_root: Path, split: str) -> list[tuple[str, dict]]:
    directory = arc_root / split
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    return [(path.stem, load_arc_task(path)) for path in sorted(directory.glob("*.json"))]


def build_training_episodes(
    arc_root: Path,
    *,
    protocol: str,
    strict_train_fraction: float = 0.8,
) -> tuple[list[ARCEpisode], list[ARCEpisode]]:
    """Build either a strict unseen-task split or the paper's transductive split."""

    if protocol not in {"strict", "paper-transductive"}:
        raise ValueError("protocol must be 'strict' or 'paper-transductive'")
    train_tasks = discover_tasks(arc_root, "training")
    if protocol == "strict":
        cutoff = round(len(train_tasks) * strict_train_fraction)
        fit_tasks, validation_tasks = train_tasks[:cutoff], train_tasks[cutoff:]
        fit = [
            episode
            for task_id, task in fit_tasks
            for episode in leave_one_out_episodes(task_id, task, "training")
        ]
        validation = [
            episode
            for task_id, task in validation_tasks
            for episode in test_episodes(task_id, task, "training-heldout")
        ]
        return fit, validation

    evaluation_tasks = discover_tasks(arc_root, "evaluation")
    fit = [
        episode
        for split, tasks in (("training", train_tasks), ("evaluation-demo", evaluation_tasks))
        for task_id, task in tasks
        for episode in leave_one_out_episodes(task_id, task, split)
    ]
    validation = [
        episode
        for task_id, task in evaluation_tasks
        for episode in test_episodes(task_id, task, "evaluation-test")
    ]
    return fit, validation


def _shared_translation(
    first: Grid,
    second: Grid,
    config: USRLConfig,
    rng: random.Random,
) -> tuple[int, int]:
    max_height = max(len(first), len(second))
    max_width = max(len(first[0]), len(second[0]))
    return (
        rng.randrange(config.max_grid_size - max_height + 1),
        rng.randrange(config.max_grid_size - max_width + 1),
    )


def collate_episodes(
    episodes: Sequence[ARCEpisode],
    config: USRLConfig,
    *,
    rng: random.Random | None = None,
    random_translation: bool = False,
) -> dict[str, torch.Tensor]:
    if not episodes:
        raise ValueError("cannot collate an empty episode batch")
    demos = min(max(len(episode.demonstrations) for episode in episodes), config.max_demos)
    batch = len(episodes)
    demo_inputs = torch.full(
        (batch, demos, config.grid_tokens), config.pad_token_id, dtype=torch.long
    )
    demo_outputs = torch.full_like(demo_inputs, config.pad_token_id)
    demo_mask = torch.zeros(batch, demos, dtype=torch.bool)
    problems = torch.empty(batch, config.grid_tokens, dtype=torch.long)
    targets = torch.empty_like(problems)

    for batch_index, episode in enumerate(episodes):
        selected = episode.demonstrations[:demos]
        for demo_index, pair in enumerate(selected):
            top, left = (0, 0)
            if random_translation:
                if rng is None:
                    raise ValueError("random_translation requires an explicit RNG")
                top, left = _shared_translation(pair.input, pair.output, config, rng)
            demo_inputs[batch_index, demo_index] = encode_grid(
                pair.input, config, top=top, left=left
            )
            demo_outputs[batch_index, demo_index] = encode_grid(
                pair.output, config, top=top, left=left
            )
            demo_mask[batch_index, demo_index] = True
        top, left = (0, 0)
        if random_translation:
            if rng is None:
                raise ValueError("random_translation requires an explicit RNG")
            top, left = _shared_translation(episode.problem, episode.target, config, rng)
        problems[batch_index] = encode_grid(episode.problem, config, top=top, left=left)
        targets[batch_index] = encode_grid(episode.target, config, top=top, left=left)
    return {
        "demo_inputs": demo_inputs,
        "demo_outputs": demo_outputs,
        "demo_mask": demo_mask,
        "problem": problems,
        "target": targets,
    }


def canonical_episode_hash(episode: ARCEpisode) -> str:
    payload = {
        "task_id": episode.task_id,
        "source_split": episode.source_split,
        "demonstrations": [
            {"input": pair.input, "output": pair.output} for pair in episode.demonstrations
        ],
        "problem": episode.problem,
        "target": episode.target,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def dataset_manifest(episodes: Iterable[ARCEpisode]) -> dict:
    episode_list = list(episodes)
    by_split: dict[str, int] = {}
    for episode in episode_list:
        by_split[episode.source_split] = by_split.get(episode.source_split, 0) + 1
    hashes = sorted(canonical_episode_hash(episode) for episode in episode_list)
    digest = hashlib.sha256("".join(hashes).encode("ascii")).hexdigest()
    return {
        "episode_count": len(episode_list),
        "task_count": len({episode.task_id for episode in episode_list}),
        "by_source_split": by_split,
        "content_sha256": digest,
    }
