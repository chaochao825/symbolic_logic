from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.afts_arc_nvarc_anchor_audit import (
    audit_probe,
    original_identifier,
    summarize_identifier_overlap,
)


def _encode(grid: list[list[int]]) -> np.ndarray:
    values = np.asarray(grid, dtype=np.uint8)
    encoded = np.zeros((30, 30), dtype=np.uint8)
    encoded[: values.shape[0], : values.shape[1]] = values + 2
    if values.shape[0] < 30:
        encoded[values.shape[0], : values.shape[1]] = 1
    if values.shape[1] < 30:
        encoded[: values.shape[0], values.shape[1]] = 1
    return encoded


def test_original_identifier_removes_only_trm_augmentation() -> None:
    assert original_identifier("abc12345|||t3|||0123456789") == "abc12345"
    assert original_identifier("task_with_underscore") == "task_with_underscore"


def test_overlap_counts_augmentation_multiplicity() -> None:
    result = summarize_identifier_overlap(
        [
            "<blank>",
            "task_a",
            "task_a|||t1|||0123456789",
            "task_b",
        ],
        ["task_a", "task_c"],
    )

    assert result["target_count"] == 2
    assert result["overlap_count"] == 1
    assert result["overlap_task_ids"] == ["task_a"]
    assert result["augmentation_multiplicity"] == {"task_a": 2}


def test_probe_detects_query_output_in_training_labels(tmp_path: Path) -> None:
    task = {
        "train": [{"input": [[1, 0], [0, 0]], "output": [[2, 0], [0, 0]]}],
        "test": [{"input": [[0, 1], [0, 0]], "output": [[0, 2], [0, 0]]}],
    }
    task_path = tmp_path / "task.json"
    task_path.write_text(json.dumps(task), encoding="utf-8")
    input_path = tmp_path / "task.inputs.bin"
    label_path = tmp_path / "task.labels.bin"
    np.stack(
        [_encode(pair["input"]) for split in ("train", "test") for pair in task[split]]
    ).tofile(input_path)
    np.stack(
        [_encode(pair["output"]) for split in ("train", "test") for pair in task[split]]
    ).tofile(label_path)

    result = audit_probe("task", task_path, input_path, label_path)

    assert result["query_pair_count"] == 1
    assert result["query_label_match_count"] == 1
    assert result["query_outputs_present_in_pretraining_labels"] is True
    assert result["pretraining_pair_matches"] == [
        {"split": "train", "pair_index": 0},
        {"split": "test", "pair_index": 0},
    ]
