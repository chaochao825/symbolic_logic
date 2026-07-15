"""ARC task loading with strict schema and source provenance."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from .grid import Grid, as_grid


@dataclass(frozen=True, slots=True)
class ARCPair:
    input: Grid
    output: Grid | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "input", as_grid(self.input))
        if self.output is not None:
            object.__setattr__(self, "output", as_grid(self.output))


@dataclass(frozen=True, slots=True)
class ARCTask:
    task_id: str
    train: tuple[ARCPair, ...]
    test: tuple[ARCPair, ...]
    source_path: str
    source_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id:
            raise TypeError("task_id must be a non-empty string")
        if not isinstance(self.source_path, str) or not self.source_path:
            raise TypeError("source_path must be a non-empty string")
        if not isinstance(self.source_sha256, str) or not re.fullmatch(
            r"[0-9a-fA-F]{64}", self.source_sha256
        ):
            raise ValueError("source_sha256 must be a 64-character hexadecimal digest")
        if not isinstance(self.train, (tuple, list)):
            raise TypeError("train must be an ordered tuple or list of ARCPair values")
        if not isinstance(self.test, (tuple, list)):
            raise TypeError("test must be an ordered tuple or list of ARCPair values")
        train = tuple(self.train)
        test = tuple(self.test)
        if any(not isinstance(pair, ARCPair) for pair in (*train, *test)):
            raise TypeError("train and test must contain only ARCPair values")
        object.__setattr__(self, "train", train)
        object.__setattr__(self, "test", test)
        object.__setattr__(self, "source_sha256", self.source_sha256.lower())
        if not train:
            raise ValueError("task must contain at least one training pair")
        if not test:
            raise ValueError("task must contain at least one test pair")
        if any(pair.output is None for pair in train):
            raise ValueError("every training pair must contain an output grid")


class _DuplicateJSONKeyError(ValueError):
    pass


def _reject_duplicate_object_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJSONKeyError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _parse_pair(raw: object, *, require_output: bool, label: str) -> ARCPair:
    if not isinstance(raw, dict):
        raise ValueError(f"{label} must be a JSON object")
    unknown = set(raw) - {"input", "output"}
    if unknown:
        raise ValueError(f"{label} contains unknown fields: {sorted(unknown)}")
    if "input" not in raw:
        raise ValueError(f"{label} is missing input")
    if require_output and "output" not in raw:
        raise ValueError(f"{label} is missing output")
    output = as_grid(raw["output"]) if "output" in raw else None
    return ARCPair(input=as_grid(raw["input"]), output=output)


def load_task(path: str | Path) -> ARCTask:
    """Load one official-format ARC JSON task.

    Hidden test tasks may omit test outputs; public tasks may include them. Training
    outputs are always required.
    """

    source = Path(path).expanduser().resolve()
    raw_bytes = source.read_bytes()
    return parse_task_bytes(raw_bytes, source_path=source, task_id=source.stem)


def parse_task_bytes(
    raw_bytes: bytes, *, source_path: str | Path, task_id: str
) -> ARCTask:
    """Parse one task from the same bytes used for provenance hashing."""

    source = Path(source_path).expanduser().resolve()
    try:
        payload = json.loads(
            raw_bytes.decode("utf-8-sig"),
            object_pairs_hook=_reject_duplicate_object_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, _DuplicateJSONKeyError) as exc:
        raise ValueError(f"invalid ARC JSON at {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"task root must be a JSON object: {source}")
    unknown = set(payload) - {"train", "test", "name"}
    if unknown:
        raise ValueError(f"task contains unknown fields {sorted(unknown)}: {source}")
    if "name" in payload:
        if not isinstance(payload["name"], str) or payload["name"] != task_id:
            raise ValueError(
                f"legacy task name must exactly match task_id {task_id!r}: {source}"
            )

    train_raw = payload.get("train")
    test_raw = payload.get("test")
    if not isinstance(train_raw, list) or not isinstance(test_raw, list):
        raise ValueError(f"task requires list-valued train and test fields: {source}")

    train = tuple(
        _parse_pair(pair, require_output=True, label=f"train[{index}]")
        for index, pair in enumerate(train_raw)
    )
    test = tuple(
        _parse_pair(pair, require_output=False, label=f"test[{index}]")
        for index, pair in enumerate(test_raw)
    )
    return ARCTask(
        task_id=task_id,
        train=train,
        test=test,
        source_path=str(source),
        source_sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )


def task_semantic_fingerprint(
    task: ARCTask, *, include_test_outputs: bool = True
) -> str:
    """Hash task semantics while ignoring pair order within train and test.

    Train and test roles remain distinct. Public-data overlap audits should include
    test outputs; hidden-task comparisons can omit them explicitly.
    """

    def pair_payload(pair: ARCPair, *, include_output: bool) -> str:
        payload: dict[str, object] = {"input": [list(row) for row in pair.input]}
        if include_output and pair.output is not None:
            payload["output"] = [list(row) for row in pair.output]
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    payload = {
        "train": sorted(pair_payload(pair, include_output=True) for pair in task.train),
        "test": sorted(
            pair_payload(pair, include_output=include_test_outputs) for pair in task.test
        ),
        "test_outputs_included": include_test_outputs,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_task_directory(path: str | Path) -> tuple[ARCTask, ...]:
    """Load every JSON task in a directory in stable task-id order."""

    root = Path(path).expanduser().resolve()
    files = sorted(root.glob("*.json"), key=lambda item: item.name)
    if not files:
        raise ValueError(f"no JSON task files found in {root}")
    tasks = tuple(load_task(file) for file in files)
    ids = [task.task_id for task in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate task IDs in {root}")
    return tasks
