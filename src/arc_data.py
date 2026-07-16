"""Strict ARC-AGI JSON loading with solver/evaluator label separation.

The solver-facing :class:`ArcProblem` contains demonstrations and test inputs.
Held-out test outputs live in a separate :class:`ArcLabels` object so a rule
selector cannot receive them accidentally through its normal API.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


ARC_COLOR_COUNT = 10
ARC_MAX_SIDE = 30


def _grid(value: object, *, where: str) -> np.ndarray:
    """Validate one nonempty rectangular ARC grid and return it read-only."""
    if not isinstance(value, list) or not value or not all(isinstance(row, list) for row in value):
        raise ValueError(f"{where} must be a nonempty list of rows")
    width = len(value[0])
    if width == 0 or any(len(row) != width for row in value):
        raise ValueError(f"{where} must be a nonempty rectangular grid")
    if len(value) > ARC_MAX_SIDE or width > ARC_MAX_SIDE:
        raise ValueError(f"{where} exceeds the ARC {ARC_MAX_SIDE}x{ARC_MAX_SIDE} limit")
    array = np.asarray(value)
    if array.ndim != 2 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{where} must contain integer colors")
    if not bool(np.all((0 <= array) & (array < ARC_COLOR_COUNT))):
        raise ValueError(f"{where} colors must be between 0 and 9")
    result = array.astype(np.uint8, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class ArcExample:
    input_grid: np.ndarray
    output_grid: np.ndarray


@dataclass(frozen=True)
class ArcProblem:
    task_id: str
    demonstrations: tuple[ArcExample, ...]
    test_inputs: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class ArcLabels:
    task_id: str
    test_outputs: tuple[np.ndarray | None, ...]


def _pair_list(value: object, *, field: str) -> list[dict]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a nonempty list")
    if not all(isinstance(pair, dict) for pair in value):
        raise ValueError(f"{field} entries must be JSON objects")
    return value


def parse_arc_task(document: object, task_id: str) -> tuple[ArcProblem, ArcLabels]:
    """Parse one official ARC task into disjoint solver and evaluator objects."""
    if not isinstance(document, dict) or "train" not in document or "test" not in document:
        raise ValueError(f"{task_id}: task must contain train and test fields")
    train = _pair_list(document["train"], field=f"{task_id}.train")
    test = _pair_list(document["test"], field=f"{task_id}.test")

    demonstrations: list[ArcExample] = []
    for index, pair in enumerate(train):
        if "input" not in pair or "output" not in pair:
            raise ValueError(f"{task_id}.train[{index}] requires input and output")
        demonstrations.append(
            ArcExample(
                _grid(pair["input"], where=f"{task_id}.train[{index}].input"),
                _grid(pair["output"], where=f"{task_id}.train[{index}].output"),
            )
        )

    inputs: list[np.ndarray] = []
    outputs: list[np.ndarray | None] = []
    for index, pair in enumerate(test):
        if "input" not in pair:
            raise ValueError(f"{task_id}.test[{index}] requires input")
        inputs.append(_grid(pair["input"], where=f"{task_id}.test[{index}].input"))
        outputs.append(
            _grid(pair["output"], where=f"{task_id}.test[{index}].output")
            if "output" in pair
            else None
        )
    return (
        ArcProblem(task_id, tuple(demonstrations), tuple(inputs)),
        ArcLabels(task_id, tuple(outputs)),
    )


def load_arc_task(path: str | Path) -> tuple[ArcProblem, ArcLabels]:
    path = Path(path)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load ARC task {path}: {error}") from error
    return parse_arc_task(document, path.stem)


def resolve_split_directory(root: str | Path, split: str) -> Path:
    """Accept an ARC repository root, its data directory, or a split directory."""
    if split not in ("training", "evaluation"):
        raise ValueError("split must be 'training' or 'evaluation'")
    root = Path(root)
    candidates = (root / "data" / split, root / split, root)
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("*.json")):
            return candidate
    raise ValueError(f"could not find {split} JSON files below {root}")


def load_arc_split(
    root: str | Path,
    split: str,
    *,
    task_ids: Sequence[str] | None = None,
    limit: int | None = None,
) -> list[tuple[ArcProblem, ArcLabels, Path]]:
    directory = resolve_split_directory(root, split)
    paths = sorted(directory.glob("*.json"))
    if task_ids is not None:
        wanted = set(task_ids)
        paths = [path for path in paths if path.stem in wanted]
        missing = sorted(wanted - {path.stem for path in paths})
        if missing:
            raise ValueError(f"task ids absent from {directory}: {', '.join(missing)}")
    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")
        paths = paths[:limit]
    return [(*load_arc_task(path), path) for path in paths]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_digest(paths: Iterable[str | Path]) -> str:
    """Hash sorted task ids and payload hashes into one split identity."""
    digest = hashlib.sha256()
    for path in sorted((Path(path) for path in paths), key=lambda item: item.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def task_id_digest(paths: Iterable[str | Path]) -> str:
    """Hash the sorted task-id set independently of task payloads."""
    digest = hashlib.sha256()
    for path in sorted((Path(path) for path in paths), key=lambda item: item.name):
        digest.update(path.stem.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()
