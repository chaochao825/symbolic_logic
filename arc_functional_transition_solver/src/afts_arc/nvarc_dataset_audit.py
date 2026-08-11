"""Audit NVARC's compiled split boundary and replay its train schedule."""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from afts_arc.experiment_safety import canonical_sha256, file_sha256


NVARC_DATASET_BOUNDARY_AUDIT_SCHEMA = "afts.nvarc-dataset-boundary-audit/v1"
_PUZZLE_ID_SEPARATOR = "|||"
_DIHEDRAL_INVERSE = (0, 3, 2, 1, 4, 5, 6, 7)


def _object(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise TypeError(f"{field} keys must be strings")
    return value


def _sequence(value: object, *, field: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise TypeError(f"{field} must be an array")
    return value


def _grid(value: object, *, field: str) -> np.ndarray:
    rows = _sequence(value, field=field)
    if not rows:
        raise ValueError(f"{field} must not be empty")
    arrays = [np.asarray(_sequence(row, field=f"{field} row")) for row in rows]
    width = arrays[0].size
    if width == 0 or any(row.ndim != 1 or row.size != width for row in arrays):
        raise ValueError(f"{field} must be a non-empty rectangular grid")
    grid = np.stack(arrays).astype(np.uint8)
    if np.any(grid > 9):
        raise ValueError(f"{field} contains a color outside 0..9")
    return grid


def _grid_key(grid: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(value) for value in row) for row in grid.tolist())


def _decode_sequence(sequence: np.ndarray) -> np.ndarray:
    if sequence.shape != (900,):
        raise ValueError("compiled ARC sequence must contain 900 tokens")
    canvas = np.asarray(sequence).reshape(30, 30)
    color_mask = canvas >= 2
    if not np.any(color_mask):
        raise ValueError("compiled ARC sequence contains no encoded grid")
    rows, columns = np.nonzero(color_mask)
    row_start = int(rows.min())
    row_stop = int(rows.max()) + 1
    column_start = int(columns.min())
    column_stop = int(columns.max()) + 1
    rectangle = canvas[row_start:row_stop, column_start:column_stop]
    if not np.all(rectangle >= 2):
        raise ValueError("compiled ARC grid rectangle contains PAD or EOS tokens")
    return (rectangle - 2).astype(np.uint8)


def _dihedral_transform(grid: np.ndarray, transform_id: int) -> np.ndarray:
    if transform_id == 0:
        return grid
    if transform_id == 1:
        return np.rot90(grid, k=1)
    if transform_id == 2:
        return np.rot90(grid, k=2)
    if transform_id == 3:
        return np.rot90(grid, k=3)
    if transform_id == 4:
        return np.fliplr(grid)
    if transform_id == 5:
        return np.flipud(grid)
    if transform_id == 6:
        return grid.T
    if transform_id == 7:
        return np.fliplr(np.rot90(grid, k=1))
    raise ValueError("dihedral transform ID must be in 0..7")


def _base_identifier(identifier: str) -> str:
    return identifier.split(_PUZZLE_ID_SEPARATOR)[0]


def _inverse_augmented_grid(identifier: str, grid: np.ndarray) -> np.ndarray:
    parts = identifier.split(_PUZZLE_ID_SEPARATOR)
    if len(parts) == 1:
        return grid.copy()
    if len(parts) != 3:
        raise ValueError("augmented puzzle identifier must have three fields")
    transform_field = parts[1]
    permutation_field = parts[2]
    if len(transform_field) != 2 or transform_field[0] != "t":
        raise ValueError("augmented puzzle identifier has an invalid transform")
    transform_id = int(transform_field[1])
    if not 0 <= transform_id <= 7:
        raise ValueError("augmented puzzle transform must be in 0..7")
    if len(permutation_field) != 10 or not permutation_field.isdigit():
        raise ValueError("augmented puzzle identifier has an invalid color map")
    permutation = np.asarray([int(value) for value in permutation_field])
    if sorted(permutation.tolist()) != list(range(10)) or permutation[0] != 0:
        raise ValueError("augmented puzzle color map must permute 0..9 and fix zero")
    geometrically_inverted = _dihedral_transform(
        grid, _DIHEDRAL_INVERSE[transform_id]
    )
    inverse_permutation = np.argsort(permutation).astype(np.uint8)
    return inverse_permutation[geometrically_inverted]


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _split_arrays(data_dir: Path, split: str) -> dict[str, np.ndarray]:
    split_dir = data_dir / split
    return {
        "group_indices": np.load(split_dir / "all__group_indices.npy"),
        "inputs": np.load(split_dir / "all__inputs.npy", mmap_mode="r"),
        "puzzle_identifiers": np.load(
            split_dir / "all__puzzle_identifiers.npy"
        ),
        "puzzle_indices": np.load(split_dir / "all__puzzle_indices.npy"),
    }


def _replay_schedule(
    *,
    puzzle_indices: np.ndarray,
    group_indices: np.ndarray,
    epochs: int,
    global_batch_size: int,
    dataset_seed: int,
    mean_puzzle_examples: float,
) -> dict[str, object]:
    if epochs <= 0 or global_batch_size <= 0:
        raise ValueError("epochs and global_batch_size must be positive")
    rng = np.random.Generator(np.random.Philox(seed=dataset_seed + 1))
    group_count = int(group_indices.size - 1)
    group_order = np.concatenate(
        [rng.permutation(group_count) for _ in range(epochs)]
    )
    trace = hashlib.sha256()
    cursor = 0
    full_batch_count = 0
    final_effective_batch_size = 0
    while cursor < group_order.size:
        effective_batch_size = 0
        while cursor < group_order.size and effective_batch_size < global_batch_size:
            group_id = int(group_order[cursor])
            puzzle_id = int(
                rng.integers(group_indices[group_id], group_indices[group_id + 1])
            )
            cursor += 1
            puzzle_size = int(
                puzzle_indices[puzzle_id + 1] - puzzle_indices[puzzle_id]
            )
            append_size = min(
                puzzle_size, global_batch_size - effective_batch_size
            )
            trace.update(
                struct.pack("<qqqq", group_id, puzzle_id, puzzle_size, append_size)
            )
            effective_batch_size += append_size
        if effective_batch_size < global_batch_size:
            final_effective_batch_size = effective_batch_size
            break
        full_batch_count += 1
    return {
        "dataset_seed": dataset_seed,
        "dropped_final_batch_size": final_effective_batch_size,
        "epochs": epochs,
        "full_batch_count": full_batch_count,
        "global_batch_size": global_batch_size,
        "group_order_sha256": hashlib.sha256(group_order.tobytes()).hexdigest(),
        "group_order_size": int(group_order.size),
        "groups_consumed": cursor,
        "metadata_estimated_training_steps": int(
            epochs * group_count * mean_puzzle_examples / global_batch_size
        ),
        "sampling_trace_sha256": trace.hexdigest(),
    }


def build_nvarc_dataset_boundary_audit(
    *,
    data_dir: Path,
    challenges_path: Path,
    epochs: int,
    global_batch_size: int,
    dataset_seed: int,
) -> dict[str, object]:
    """Prove that updates use demos and replay the exact upstream batch count."""

    data_dir = data_dir.resolve()
    challenges_path = challenges_path.resolve()
    challenges = _object(_load_json(challenges_path), field="challenges")
    identifiers = _sequence(
        _load_json(data_dir / "identifiers.json"), field="identifiers"
    )
    train_metadata = _object(
        _load_json(data_dir / "train" / "dataset.json"),
        field="train metadata",
    )
    test_metadata = _object(
        _load_json(data_dir / "test" / "dataset.json"), field="test metadata"
    )
    train = _split_arrays(data_dir, "train")
    test = _split_arrays(data_dir, "test")
    train_labels = np.load(data_dir / "train" / "all__labels.npy", mmap_mode="r")

    if not np.array_equal(
        train["puzzle_identifiers"], test["puzzle_identifiers"]
    ):
        raise ValueError("train and test puzzle identifier arrays differ")
    if not np.array_equal(train["group_indices"], test["group_indices"]):
        raise ValueError("train and test group boundaries differ")
    if len(identifiers) != int(train_metadata["num_puzzle_identifiers"]):
        raise ValueError("identifier count differs from train metadata")
    if int(test_metadata["num_puzzle_identifiers"]) != len(identifiers):
        raise ValueError("identifier count differs from test metadata")
    if int(train_metadata["total_puzzles"]) != train["puzzle_identifiers"].size:
        raise ValueError("train puzzle count differs from metadata")
    if int(test_metadata["total_puzzles"]) != test["puzzle_identifiers"].size:
        raise ValueError("test puzzle count differs from metadata")
    if int(train_metadata["total_groups"]) != train["group_indices"].size - 1:
        raise ValueError("train group count differs from metadata")
    if int(test_metadata["total_groups"]) != test["group_indices"].size - 1:
        raise ValueError("test group count differs from metadata")

    demo_pairs: dict[
        str, set[tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]]
    ] = {}
    query_inputs: dict[str, set[tuple[tuple[int, ...], ...]]] = {}
    demo_example_counts: dict[str, int] = {}
    query_example_counts: dict[str, int] = {}
    demo_query_input_collision_count = 0
    for task_id in sorted(challenges):
        task = _object(challenges[task_id], field=f"challenge[{task_id}]")
        train_examples = _sequence(
            task["train"], field=f"challenge[{task_id}].train"
        )
        test_examples = _sequence(task["test"], field=f"challenge[{task_id}].test")
        task_demo_pairs = set()
        task_demo_inputs = set()
        task_query_inputs = set()
        for example_index, raw_example in enumerate(train_examples):
            example = _object(
                raw_example, field=f"challenge[{task_id}].train[{example_index}]"
            )
            input_key = _grid_key(
                _grid(example["input"], field="demonstration input")
            )
            output_key = _grid_key(
                _grid(example["output"], field="demonstration output")
            )
            task_demo_inputs.add(input_key)
            task_demo_pairs.add((input_key, output_key))
        for query_index, raw_query in enumerate(test_examples):
            query = _object(
                raw_query, field=f"challenge[{task_id}].test[{query_index}]"
            )
            if set(query) != {"input"}:
                raise ValueError("challenge query must contain input only")
            task_query_inputs.add(
                _grid_key(_grid(query["input"], field="query input"))
            )
        demo_query_input_collision_count += len(
            task_demo_inputs.intersection(task_query_inputs)
        )
        demo_pairs[task_id] = task_demo_pairs
        query_inputs[task_id] = task_query_inputs
        demo_example_counts[task_id] = len(train_examples)
        query_example_counts[task_id] = len(test_examples)

    puzzle_ids = train["puzzle_identifiers"]
    train_indices = train["puzzle_indices"]
    test_indices = test["puzzle_indices"]
    train_pairs_outside_demo_set = 0
    train_puzzles_wrong_example_count = 0
    test_inputs_outside_query_set = 0
    test_puzzles_wrong_example_count = 0
    represented_base_identifiers: set[str] = set()
    for puzzle_index, raw_identifier_id in enumerate(puzzle_ids):
        identifier_id = int(raw_identifier_id)
        identifier = identifiers[identifier_id]
        if not isinstance(identifier, str):
            raise TypeError("puzzle identifier must be a string")
        base_identifier = _base_identifier(identifier)
        if base_identifier not in demo_pairs:
            raise ValueError("compiled puzzle identifier is absent from challenges")
        represented_base_identifiers.add(base_identifier)

        train_start = int(train_indices[puzzle_index])
        train_stop = int(train_indices[puzzle_index + 1])
        if train_stop - train_start != demo_example_counts[base_identifier]:
            train_puzzles_wrong_example_count += 1
        for example_index in range(train_start, train_stop):
            input_grid = _inverse_augmented_grid(
                identifier, _decode_sequence(train["inputs"][example_index])
            )
            output_grid = _inverse_augmented_grid(
                identifier, _decode_sequence(train_labels[example_index])
            )
            pair = (_grid_key(input_grid), _grid_key(output_grid))
            train_pairs_outside_demo_set += int(
                pair not in demo_pairs[base_identifier]
            )

        test_start = int(test_indices[puzzle_index])
        test_stop = int(test_indices[puzzle_index + 1])
        if test_stop - test_start != query_example_counts[base_identifier]:
            test_puzzles_wrong_example_count += 1
        for example_index in range(test_start, test_stop):
            input_grid = _inverse_augmented_grid(
                identifier, _decode_sequence(test["inputs"][example_index])
            )
            test_inputs_outside_query_set += int(
                _grid_key(input_grid) not in query_inputs[base_identifier]
            )

    group_base_identifiers: list[str] = []
    for group_index in range(train["group_indices"].size - 1):
        group_start = int(train["group_indices"][group_index])
        group_stop = int(train["group_indices"][group_index + 1])
        group_bases = {
            _base_identifier(str(identifiers[int(puzzle_ids[puzzle_index])]))
            for puzzle_index in range(group_start, group_stop)
        }
        if len(group_bases) != 1:
            raise ValueError("compiled augmentation group mixes base tasks")
        group_base_identifiers.append(next(iter(group_bases)))

    schedule = _replay_schedule(
        puzzle_indices=train_indices,
        group_indices=train["group_indices"],
        epochs=epochs,
        global_batch_size=global_batch_size,
        dataset_seed=dataset_seed,
        mean_puzzle_examples=float(train_metadata["mean_puzzle_examples"]),
    )
    violations = {
        "demo_query_input_collision_count": demo_query_input_collision_count,
        "group_base_identifier_duplicate_count": len(group_base_identifiers)
        - len(set(group_base_identifiers)),
        "represented_task_count_difference": len(challenges)
        - len(represented_base_identifiers),
        "test_inputs_outside_query_set": test_inputs_outside_query_set,
        "test_puzzles_wrong_example_count": test_puzzles_wrong_example_count,
        "train_pairs_outside_demo_set": train_pairs_outside_demo_set,
        "train_puzzles_wrong_example_count": train_puzzles_wrong_example_count,
    }
    artifacts = {
        "challenges": file_sha256(challenges_path),
        "identifiers": file_sha256(data_dir / "identifiers.json"),
        "test_inputs": file_sha256(data_dir / "test" / "all__inputs.npy"),
        "test_labels_opaque": file_sha256(data_dir / "test" / "all__labels.npy"),
        "train_inputs": file_sha256(data_dir / "train" / "all__inputs.npy"),
        "train_labels": file_sha256(data_dir / "train" / "all__labels.npy"),
    }
    body: dict[str, object] = {
        "artifacts": artifacts,
        "counts": {
            "augmented_puzzle_count": int(puzzle_ids.size),
            "base_task_count": len(represented_base_identifiers),
            "test_input_count": int(test["inputs"].shape[0]),
            "train_pair_count": int(train["inputs"].shape[0]),
        },
        "query_label_boundary": {
            "test_label_file_content_addressed": True,
            "test_label_semantics_read": False,
            "training_pairs_reconstructed_against_demonstrations": True,
        },
        "schedule": schedule,
        "schema": NVARC_DATASET_BOUNDARY_AUDIT_SCHEMA,
        "status": "clean" if not any(violations.values()) else "failed",
        "violations": violations,
    }
    return {"audit_id": canonical_sha256(body), **body}
