"""Audit whether the released NVARC TRM checkpoint is a clean ARC anchor."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = str((PROJECT_ROOT / "src").resolve())
if SOURCE_ROOT not in sys.path:
    sys.path.insert(0, SOURCE_ROOT)

from afts_arc.experiment_safety import (  # noqa: E402
    atomic_write_json,
    canonical_sha256,
    file_sha256,
    runtime_metadata,
)


AUDIT_SCHEMA = "afts.nvarc-trm-anchor-leakage-audit/v1"
PROBE_TASK_IDS = ("e4888269", "5adee1b2", "984d8a3e")
EXPECTED_CHECKPOINT_KEYS = (
    "_orig_mod.model.inner.H_init",
    "_orig_mod.model.inner.L_init",
    "_orig_mod.model.inner.embed_tokens.embedding_weight",
    "_orig_mod.model.inner.lm_head.weight",
    "_orig_mod.model.inner.q_head.weight",
    "_orig_mod.model.inner.q_head.bias",
    "_orig_mod.model.inner.puzzle_emb.weights",
    "_orig_mod.model.inner.L_level.layers.0.self_attn.qkv_proj.weight",
    "_orig_mod.model.inner.L_level.layers.0.self_attn.o_proj.weight",
    "_orig_mod.model.inner.L_level.layers.0.mlp.gate_up_proj.weight",
    "_orig_mod.model.inner.L_level.layers.0.mlp.down_proj.weight",
    "_orig_mod.model.inner.L_level.layers.1.self_attn.qkv_proj.weight",
    "_orig_mod.model.inner.L_level.layers.1.self_attn.o_proj.weight",
    "_orig_mod.model.inner.L_level.layers.1.mlp.gate_up_proj.weight",
    "_orig_mod.model.inner.L_level.layers.1.mlp.down_proj.weight",
)


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _object(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a JSON object")
    return value


def _list(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a JSON list")
    return value


def original_identifier(identifier: str) -> str:
    """Remove only the D4/color augmentation suffix used by TRM."""

    return identifier.split("|||", maxsplit=1)[0]


def summarize_identifier_overlap(
    training_identifiers: Sequence[str], target_ids: Sequence[str]
) -> dict[str, object]:
    originals = tuple(
        original_identifier(identifier)
        for identifier in training_identifiers
        if identifier != "<blank>"
    )
    counts = Counter(originals)
    target = tuple(sorted(target_ids))
    overlap = tuple(task_id for task_id in target if task_id in counts)
    return {
        "target_count": len(target),
        "overlap_count": len(overlap),
        "overlap_fraction": len(overlap) / len(target) if target else 0.0,
        "overlap_task_ids": list(overlap),
        "augmentation_multiplicity": {task_id: counts[task_id] for task_id in overlap},
    }


def _crop_encoded_grid(encoded: np.ndarray) -> np.ndarray:
    grid = encoded.reshape(30, 30)
    valid = (grid >= 2) & (grid <= 11)
    rows = np.flatnonzero(valid.any(axis=1))
    columns = np.flatnonzero(valid.any(axis=0))
    if rows.size == 0 or columns.size == 0:
        raise ValueError("encoded ARC grid contains no color token")
    return (grid[rows[0] : rows[-1] + 1, columns[0] : columns[-1] + 1] - 2).astype(
        np.uint8
    )


def _official_pairs(
    task: Mapping[str, object],
) -> list[tuple[str, int, np.ndarray, np.ndarray]]:
    pairs: list[tuple[str, int, np.ndarray, np.ndarray]] = []
    for split in ("train", "test"):
        values = _list(task[split], f"official task {split}")
        for index, raw_pair in enumerate(values):
            pair = _object(raw_pair, f"official {split} pair")
            if set(pair) != {"input", "output"}:
                raise ValueError(f"official {split} pair has unexpected fields")
            pairs.append(
                (
                    split,
                    index,
                    np.asarray(pair["input"], dtype=np.uint8),
                    np.asarray(pair["output"], dtype=np.uint8),
                )
            )
    return pairs


def audit_probe(
    task_id: str,
    official_task_path: Path,
    input_rows_path: Path,
    label_rows_path: Path,
) -> dict[str, object]:
    task = _object(_load_json(official_task_path), f"official task {task_id}")
    pairs = _official_pairs(task)
    inputs = np.fromfile(input_rows_path, dtype=np.uint8)
    labels = np.fromfile(label_rows_path, dtype=np.uint8)
    expected_size = len(pairs) * 30 * 30
    if inputs.size != expected_size or labels.size != expected_size:
        raise ValueError(f"probe row count mismatch for {task_id}")
    inputs = inputs.reshape(len(pairs), 30, 30)
    labels = labels.reshape(len(pairs), 30, 30)
    matches = []
    for input_row, label_row in zip(inputs, labels, strict=True):
        decoded_input = _crop_encoded_grid(input_row)
        decoded_label = _crop_encoded_grid(label_row)
        row_matches = [
            {"split": split, "pair_index": index}
            for split, index, pair_input, pair_output in pairs
            if np.array_equal(decoded_input, pair_input)
            and np.array_equal(decoded_label, pair_output)
        ]
        if len(row_matches) != 1:
            raise ValueError(f"probe row is not uniquely attributable for {task_id}")
        matches.append(row_matches[0])
    expected_matches = [
        {"split": split, "pair_index": index} for split, index, _, _ in pairs
    ]
    if matches != expected_matches:
        raise ValueError(f"probe row order mismatch for {task_id}")
    query_matches = [match for match in matches if match["split"] == "test"]
    return {
        "task_id": task_id,
        "official_task_sha256": file_sha256(official_task_path),
        "input_rows_sha256": file_sha256(input_rows_path),
        "label_rows_sha256": file_sha256(label_rows_path),
        "pretraining_pair_matches": matches,
        "query_pair_count": len(_list(task["test"], "official test pairs")),
        "query_label_match_count": len(query_matches),
        "query_outputs_present_in_pretraining_labels": len(query_matches) > 0,
    }


def audit_checkpoint(path: Path) -> dict[str, object]:
    import torch

    state = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    if not isinstance(state, Mapping):
        raise TypeError("checkpoint must contain a state dictionary")
    keys = tuple(state)
    if keys != EXPECTED_CHECKPOINT_KEYS:
        raise ValueError("checkpoint state keys do not match the frozen contract")
    tensor_rows = []
    total_numel = 0
    total_bytes = 0
    all_finite = True
    for key, tensor in state.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"checkpoint value is not a tensor: {key}")
        flat = tensor.reshape(-1)
        finite = True
        for start in range(0, flat.numel(), 4 * 1024 * 1024):
            if not torch.isfinite(flat[start : start + 4 * 1024 * 1024]).all().item():
                finite = False
                break
        all_finite &= finite
        tensor_rows.append(
            {
                "key": key,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "numel": tensor.numel(),
                "bytes": tensor.numel() * tensor.element_size(),
                "all_finite": finite,
            }
        )
        total_numel += tensor.numel()
        total_bytes += tensor.numel() * tensor.element_size()
    if not all_finite:
        raise ValueError("checkpoint contains a non-finite tensor")
    return {
        "sha256": file_sha256(path),
        "file_bytes": path.stat().st_size,
        "state_key_count": len(keys),
        "total_numel": total_numel,
        "tensor_bytes": total_bytes,
        "all_finite": all_finite,
        "tensors": tensor_rows,
    }


def _project_task_ids(result_path: Path) -> list[str]:
    result = _object(_load_json(result_path), "project cohort result")
    tasks = _list(result["tasks"], "project cohort tasks")
    ids = []
    for raw_task in tasks:
        task = _object(raw_task, "project cohort task")
        task_id = task["task_id"]
        if not isinstance(task_id, str) or not task_id:
            raise TypeError("project task_id must be a non-empty string")
        ids.append(task_id)
    if len(ids) != len(set(ids)):
        raise ValueError("project cohort contains duplicate task IDs")
    return sorted(ids)


def _training_identifiers(path: Path) -> list[str]:
    raw = _list(_load_json(path), "training identifiers")
    if not all(isinstance(value, str) for value in raw):
        raise TypeError("training identifiers must contain only strings")
    return list(raw)  # type: ignore[arg-type]


def _evaluation_task_ids(path: Path) -> list[str]:
    value = _object(_load_json(path), "evaluation test puzzles")
    return sorted(value)


def _validate_training_indices(
    identifiers: Sequence[str], puzzle_identifiers_path: Path, puzzle_indices_path: Path
) -> dict[str, object]:
    puzzle_identifiers = np.load(puzzle_identifiers_path, mmap_mode="r")
    puzzle_indices = np.load(puzzle_indices_path, mmap_mode="r")
    if puzzle_identifiers.ndim != 1 or puzzle_indices.ndim != 1:
        raise ValueError("training index arrays must be one-dimensional")
    if puzzle_indices.size != puzzle_identifiers.size + 1:
        raise ValueError("training puzzle index arrays are inconsistent")
    expected = np.arange(1, len(identifiers), dtype=puzzle_identifiers.dtype)
    if not np.array_equal(puzzle_identifiers, expected):
        raise ValueError("training puzzle identifiers are not the declared sequence")
    if puzzle_indices[0] != 0 or np.any(np.diff(puzzle_indices) <= 0):
        raise ValueError("training puzzle row ranges are invalid")
    return {
        "puzzle_identifiers_sha256": file_sha256(puzzle_identifiers_path),
        "puzzle_indices_sha256": file_sha256(puzzle_indices_path),
        "puzzle_count": int(puzzle_identifiers.size),
        "example_row_count": int(puzzle_indices[-1]),
    }


def construct_audit(args: argparse.Namespace) -> dict[str, object]:
    training_identifiers = _training_identifiers(args.training_identifiers)
    evaluation_ids = _evaluation_task_ids(args.evaluation_test_puzzles)
    project_ids = _project_task_ids(args.project_cohort_result)
    index_audit = _validate_training_indices(
        training_identifiers,
        args.training_puzzle_identifiers,
        args.training_puzzle_indices,
    )
    original_ids = {
        original_identifier(identifier)
        for identifier in training_identifiers
        if identifier != "<blank>"
    }
    hex_ids = sorted(
        identifier
        for identifier in original_ids
        if re.fullmatch(r"[0-9a-f]{8}", identifier)
    )
    probes = []
    for task_id in PROBE_TASK_IDS:
        probes.append(
            audit_probe(
                task_id,
                args.official_arc2_training_root / f"{task_id}.json",
                args.probe_root / f"{task_id}.inputs.bin",
                args.probe_root / f"{task_id}.labels.bin",
            )
        )
    if not all(
        probe["query_outputs_present_in_pretraining_labels"] for probe in probes
    ):
        raise ValueError("not every frozen leakage probe exposes a query label")
    evaluation_overlap = summarize_identifier_overlap(
        training_identifiers, evaluation_ids
    )
    project_overlap = summarize_identifier_overlap(training_identifiers, project_ids)
    clean_heldout = (
        evaluation_overlap["overlap_count"] == 0
        and project_overlap["overlap_count"] == 0
        and not any(
            probe["query_outputs_present_in_pretraining_labels"] for probe in probes
        )
    )
    content = {
        "schema": AUDIT_SCHEMA,
        "status": "completed_negative_eligibility_audit",
        "checkpoint": audit_checkpoint(args.checkpoint),
        "assets": {
            "training_identifiers_sha256": file_sha256(args.training_identifiers),
            "evaluation_test_puzzles_sha256": file_sha256(args.evaluation_test_puzzles),
            "project_cohort_result_sha256": file_sha256(args.project_cohort_result),
            **index_audit,
        },
        "training_identifier_summary": {
            "identifier_count_including_blank": len(training_identifiers),
            "original_task_count": len(original_ids),
            "official_hex_task_count": len(hex_ids),
        },
        "evaluation_120_overlap": evaluation_overlap,
        "project_50_overlap": project_overlap,
        "query_label_probes": probes,
        "decision": {
            "clean_heldout_anchor_eligible": clean_heldout,
            "official_public_replication": "allowed_only_as_contaminated_software_replication",
            "current_50_solver_comparison": "forbidden_training_label_exposure",
            "fresh_nonoverlapping_cohort_required": True,
            "gpu_run_authorized_as_clean_evidence": False,
        },
        "source_contract": {
            str(Path(__file__).resolve().relative_to(PROJECT_ROOT)).replace(
                "\\", "/"
            ): file_sha256(Path(__file__).resolve()),
            str(args.protocol.relative_to(PROJECT_ROOT)).replace(
                "\\", "/"
            ): file_sha256(args.protocol),
            "nvarc_eval_script": file_sha256(args.nvarc_eval_script),
            "nvarc_readme": file_sha256(args.nvarc_readme),
            "trm_dataset_builder": file_sha256(args.trm_dataset_builder),
            "trm_evaluator": file_sha256(args.trm_evaluator),
            "trm_loss": file_sha256(args.trm_loss),
            "trm_model": file_sha256(args.trm_model),
        },
        "runtime": runtime_metadata(("numpy", "torch")),
    }
    return {"audit_id": canonical_sha256(content), **content}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-identifiers", type=Path, required=True)
    parser.add_argument("--training-puzzle-identifiers", type=Path, required=True)
    parser.add_argument("--training-puzzle-indices", type=Path, required=True)
    parser.add_argument("--evaluation-test-puzzles", type=Path, required=True)
    parser.add_argument("--project-cohort-result", type=Path, required=True)
    parser.add_argument("--official-arc2-training-root", type=Path, required=True)
    parser.add_argument("--probe-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--nvarc-eval-script", type=Path, required=True)
    parser.add_argument("--nvarc-readme", type=Path, required=True)
    parser.add_argument("--trm-dataset-builder", type=Path, required=True)
    parser.add_argument("--trm-evaluator", type=Path, required=True)
    parser.add_argument("--trm-loss", type=Path, required=True)
    parser.add_argument("--trm-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = construct_audit(args)
    atomic_write_json(args.output, audit)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
