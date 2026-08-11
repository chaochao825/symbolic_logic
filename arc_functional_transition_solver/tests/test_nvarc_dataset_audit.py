from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from afts_arc.experiment_safety import canonical_sha256
from afts_arc.nvarc_dataset_audit import (
    _decode_sequence,
    _inverse_augmented_grid,
    build_nvarc_dataset_boundary_audit,
)


def _encode(grid: list[list[int]], *, row: int = 0, column: int = 0) -> np.ndarray:
    values = np.asarray(grid, dtype=np.uint8)
    canvas = np.zeros((30, 30), dtype=np.uint8)
    height, width = values.shape
    canvas[row : row + height, column : column + width] = values + 2
    if row + height < 30:
        canvas[row + height, column : column + width] = 1
    if column + width < 30:
        canvas[row : row + height, column + width] = 1
    return canvas.reshape(-1)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _dataset(tmp_path: Path) -> tuple[Path, Path]:
    data_dir = tmp_path / "compiled"
    (data_dir / "train").mkdir(parents=True)
    (data_dir / "test").mkdir()
    challenges = {
        "a": {
            "train": [{"input": [[1]], "output": [[2]]}],
            "test": [{"input": [[3]]}],
        },
        "b": {
            "train": [{"input": [[4, 4]], "output": [[5], [5]]}],
            "test": [{"input": [[6, 6]]}],
        },
    }
    challenges_path = tmp_path / "challenges.json"
    _write_json(challenges_path, challenges)
    _write_json(data_dir / "identifiers.json", ["<blank>", "a", "b"])
    metadata = {
        "blank_identifier_id": 0,
        "ignore_label_id": 0,
        "mean_puzzle_examples": 1.0,
        "num_puzzle_identifiers": 3,
        "pad_id": 0,
        "seq_len": 900,
        "sets": ["all"],
        "total_groups": 2,
        "total_puzzles": 2,
        "vocab_size": 12,
    }
    for split in ("train", "test"):
        split_dir = data_dir / split
        _write_json(split_dir / "dataset.json", metadata)
        np.save(split_dir / "all__group_indices.npy", np.asarray([0, 1, 2]))
        np.save(split_dir / "all__puzzle_identifiers.npy", np.asarray([1, 2]))
        np.save(split_dir / "all__puzzle_indices.npy", np.asarray([0, 1, 2]))
    np.save(
        data_dir / "train" / "all__inputs.npy",
        np.stack([_encode([[1]], row=4, column=2), _encode([[4, 4]])]),
    )
    np.save(
        data_dir / "train" / "all__labels.npy",
        np.stack([_encode([[2]], row=4, column=2), _encode([[5], [5]])]),
    )
    np.save(
        data_dir / "test" / "all__inputs.npy",
        np.stack([_encode([[3]]), _encode([[6, 6]])]),
    )
    np.save(
        data_dir / "test" / "all__labels.npy",
        np.stack([_encode([[9]]), _encode([[9]])]),
    )
    return data_dir, challenges_path


def test_decode_and_inverse_augmentation() -> None:
    grid = np.asarray([[1, 2], [3, 4]], dtype=np.uint8)
    assert np.array_equal(_decode_sequence(_encode(grid.tolist(), row=5, column=7)), grid)

    mapped = np.rot90(np.asarray([[2, 1], [3, 4]], dtype=np.uint8), k=1)
    restored = _inverse_augmented_grid("task|||t1|||0213456789", mapped)
    assert np.array_equal(restored, grid)


def test_boundary_audit_reconstructs_only_demo_pairs_without_test_labels(
    tmp_path: Path,
) -> None:
    data_dir, challenges_path = _dataset(tmp_path)

    audit = build_nvarc_dataset_boundary_audit(
        data_dir=data_dir,
        challenges_path=challenges_path,
        epochs=2,
        global_batch_size=2,
        dataset_seed=0,
    )

    assert audit["audit_id"] == canonical_sha256(
        {key: value for key, value in audit.items() if key != "audit_id"}
    )
    assert audit["status"] == "clean"
    assert audit["violations"] == {
        "demo_query_input_collision_count": 0,
        "group_base_identifier_duplicate_count": 0,
        "represented_task_count_difference": 0,
        "test_inputs_outside_query_set": 0,
        "test_puzzles_wrong_example_count": 0,
        "train_pairs_outside_demo_set": 0,
        "train_puzzles_wrong_example_count": 0,
    }
    assert audit["query_label_boundary"]["test_label_semantics_read"] is False


def test_boundary_audit_reports_compiled_training_pair_violation(
    tmp_path: Path,
) -> None:
    data_dir, challenges_path = _dataset(tmp_path)
    labels = np.load(data_dir / "train" / "all__labels.npy")
    labels[0] = _encode([[8]])
    np.save(data_dir / "train" / "all__labels.npy", labels)

    audit = build_nvarc_dataset_boundary_audit(
        data_dir=data_dir,
        challenges_path=challenges_path,
        epochs=2,
        global_batch_size=2,
        dataset_seed=0,
    )

    assert audit["status"] == "failed"
    assert audit["violations"]["train_pairs_outside_demo_set"] == 1
