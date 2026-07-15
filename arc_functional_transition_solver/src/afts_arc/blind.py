"""Oracle-free task view used by candidate generation and search."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from .grid import Grid, as_grid, grid_to_lists
from .task import ARCPair, ARCTask

BLIND_TASK_SCHEMA_VERSION = "afts.blind-task/v1"


@dataclass(frozen=True, slots=True)
class BlindTask:
    task_id: str
    train: tuple[ARCPair, ...]
    test_inputs: tuple[Grid, ...]
    blind_content_sha256: str
    semantic_fingerprint_sha256: str

    def __post_init__(self) -> None:
        train = tuple(self.train)
        test_inputs = tuple(as_grid(grid) for grid in self.test_inputs)
        if not train or any(
            not isinstance(pair, ARCPair) or pair.output is None for pair in train
        ):
            raise ValueError("blind task training pairs must have outputs")
        if not test_inputs:
            raise ValueError("blind task must contain at least one test input")
        object.__setattr__(self, "train", train)
        object.__setattr__(self, "test_inputs", test_inputs)
        content_sha = _blind_content_sha256(train, test_inputs)
        semantic_sha = _semantic_fingerprint(train, test_inputs)
        if self.task_id != f"blind_{content_sha}":
            raise ValueError("task_id must be derived from ordered blind content")
        if self.blind_content_sha256 != content_sha:
            raise ValueError("blind_content_sha256 does not match ordered observable content")
        if self.semantic_fingerprint_sha256 != semantic_sha:
            raise ValueError(
                "semantic_fingerprint_sha256 does not match observable semantics"
            )

    @classmethod
    def from_task(cls, task: ARCTask) -> "BlindTask":
        test_inputs = tuple(pair.input for pair in task.test)
        return cls.from_observations(train=task.train, test_inputs=test_inputs)

    @classmethod
    def from_observations(
        cls, *, train: tuple[ARCPair, ...], test_inputs: tuple[Grid, ...]
    ) -> "BlindTask":
        normalized_train = tuple(train)
        normalized_test = tuple(as_grid(grid) for grid in test_inputs)
        content_sha = _blind_content_sha256(normalized_train, normalized_test)
        return cls(
            task_id=f"blind_{content_sha}",
            train=normalized_train,
            test_inputs=normalized_test,
            blind_content_sha256=content_sha,
            semantic_fingerprint_sha256=_semantic_fingerprint(
                normalized_train, normalized_test
            ),
        )

    @classmethod
    def from_json_dict(cls, payload: object) -> "BlindTask":
        """Load the exact oracle-free interchange schema and reject extensions."""

        if not isinstance(payload, dict):
            raise TypeError("blind task must be a JSON object")
        expected = {
            "schema",
            "task_id",
            "train",
            "test",
            "blind_content_sha256",
            "semantic_fingerprint_sha256",
        }
        if set(payload) != expected:
            raise ValueError(
                f"blind task fields must be {sorted(expected)}, found {sorted(payload)}"
            )
        train_payload = payload["train"]
        test_payload = payload["test"]
        if not isinstance(train_payload, list) or not isinstance(test_payload, list):
            raise TypeError("blind task train and test fields must be lists")
        if payload["schema"] != BLIND_TASK_SCHEMA_VERSION:
            raise ValueError("unsupported blind task schema")
        train: list[ARCPair] = []
        for index, pair in enumerate(train_payload):
            if not isinstance(pair, dict) or set(pair) != {"input", "output"}:
                raise ValueError(
                    f"blind train[{index}] must contain exactly input and output"
                )
            train.append(ARCPair(as_grid(pair["input"]), as_grid(pair["output"])))
        test_inputs: list[Grid] = []
        for index, pair in enumerate(test_payload):
            if not isinstance(pair, dict) or set(pair) != {"input"}:
                raise ValueError(
                    f"blind test[{index}] must contain input and no output"
                )
            test_inputs.append(as_grid(pair["input"]))
        return cls(
            task_id=payload["task_id"],
            train=tuple(train),
            test_inputs=tuple(test_inputs),
            blind_content_sha256=payload["blind_content_sha256"],
            semantic_fingerprint_sha256=payload["semantic_fingerprint_sha256"],
        )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema": BLIND_TASK_SCHEMA_VERSION,
            "task_id": self.task_id,
            "train": [
                {
                    "input": grid_to_lists(pair.input),
                    "output": grid_to_lists(pair.output),
                }
                for pair in self.train
            ],
            "test": [{"input": grid_to_lists(grid)} for grid in self.test_inputs],
            "blind_content_sha256": self.blind_content_sha256,
            "semantic_fingerprint_sha256": self.semantic_fingerprint_sha256,
        }


def _observable_payload(
    train: tuple[ARCPair, ...], test_inputs: tuple[Grid, ...]
) -> dict[str, object]:
    return {
        "schema": BLIND_TASK_SCHEMA_VERSION,
        "train": [
            {
                "input": grid_to_lists(pair.input),
                "output": grid_to_lists(pair.output),
            }
            for pair in train
        ],
        "test": [{"input": grid_to_lists(grid)} for grid in test_inputs],
    }


def _blind_content_sha256(
    train: tuple[ARCPair, ...], test_inputs: tuple[Grid, ...]
) -> str:
    canonical = json.dumps(
        _observable_payload(train, test_inputs),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def _semantic_fingerprint(
    train: tuple[ARCPair, ...], test_inputs: tuple[Grid, ...]
) -> str:
    payload = {
        "train": sorted(
            json.dumps(
                {
                    "input": grid_to_lists(pair.input),
                    "output": grid_to_lists(pair.output),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            for pair in train
        ),
        "test_inputs": sorted(
            json.dumps(grid_to_lists(grid), separators=(",", ":"))
            for grid in test_inputs
        ),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
