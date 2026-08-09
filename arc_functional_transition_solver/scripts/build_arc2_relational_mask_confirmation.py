"""Build the frozen query-blind ARC-2 relational-mask confirmation cohort."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import random
import subprocess
import sys
from collections import Counter, deque
from collections.abc import Mapping, Sequence
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT_STRING = str(PROJECT_ROOT)
SOURCE_ROOT = str((PROJECT_ROOT / "src").resolve())
if PROJECT_ROOT_STRING not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STRING)
if SOURCE_ROOT not in sys.path:
    sys.path.insert(0, SOURCE_ROOT)

from afts_arc.experiment_safety import atomic_write_json, canonical_sha256  # noqa: E402
from afts_arc.grid import as_grid, grid_to_lists  # noqa: E402
from afts_arc.visual_provider_gate import (  # noqa: E402
    EXPOSURE_REGISTRY_SCHEMA,
    _file_sha256,
    make_query_blind_payload,
)
from scripts.build_arcgen_visual_trace_repair_cohort import (  # noqa: E402
    EXPECTED_ARC1_COMMIT,
    EXPECTED_ARC1_SPLIT_COUNT,
    EXPECTED_ARCGEN_COMMIT,
    _pair_seed,
)


COHORT_SCHEMA = "afts.arc2-relational-mask-confirmation/v1"
SELECTION_SEED = "relational-mask-arc2-confirm-v1-20260809"
EXPECTED_ARC2_COMMIT = "f3283f727488ad98fe575ea6a5ac981e4a188e49"
EXPECTED_ARC2_TRAINING_COUNT = 1_000
EXPECTED_ELIGIBLE_COUNT = 50
DEVELOPMENT_FAMILY_ID = "9f5f939b"

Coordinate = tuple[int, int]


def _git_head(root: Path) -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _load_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON payload must be an object: {path}")
    return payload


def _write_new(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to replace artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, payload)


def _component_sizes(grid: Sequence[Sequence[int]]) -> tuple[int, ...]:
    height, width = len(grid), len(grid[0])
    counts = Counter(cell for row in grid for cell in row)
    background = min(counts, key=lambda color: (-counts[color], color))
    unseen = {
        (row, column)
        for row in range(height)
        for column in range(width)
        if grid[row][column] != background
    }
    sizes = []
    while unseen:
        start = min(unseen)
        color = grid[start[0]][start[1]]
        unseen.remove(start)
        queue: deque[Coordinate] = deque((start,))
        size = 0
        while queue:
            row, column = queue.popleft()
            size += 1
            for dr, dc in ((-1, 0), (0, -1), (0, 1), (1, 0)):
                neighbor = (row + dr, column + dc)
                if neighbor in unseen and grid[neighbor[0]][neighbor[1]] == color:
                    unseen.remove(neighbor)
                    queue.append(neighbor)
        sizes.append(size)
    return tuple(sorted(sizes))


def _grid_signature(grid: object) -> dict[str, object]:
    normalized = grid_to_lists(as_grid(grid))
    counts = Counter(cell for row in normalized for cell in row)
    return {
        "shape": [len(normalized), len(normalized[0])],
        "color_count_multiset": sorted(counts.values()),
        "foreground_component_sizes": list(_component_sizes(normalized)),
    }


def _change_signature(first: object, second: object) -> dict[str, object]:
    source = grid_to_lists(as_grid(first))
    target = grid_to_lists(as_grid(second))
    same_shape = (len(source), len(source[0])) == (len(target), len(target[0]))
    if not same_shape:
        return {"same_shape": False}
    changed = tuple(
        (row, column)
        for row in range(len(source))
        for column in range(len(source[0]))
        if source[row][column] != target[row][column]
    )
    if not changed:
        return {
            "same_shape": True,
            "changed_count": 0,
            "changed_bbox_shape": [0, 0],
        }
    top = min(row for row, _ in changed)
    bottom = max(row for row, _ in changed)
    left = min(column for _, column in changed)
    right = max(column for _, column in changed)
    return {
        "same_shape": True,
        "changed_count": len(changed),
        "changed_bbox_shape": [bottom - top + 1, right - left + 1],
    }


def blind_structural_signature(payload: Mapping[str, object]) -> str:
    train = payload["train"]
    test = payload["test"]
    if not isinstance(train, list) or not train:
        raise ValueError("blind structural signature requires demonstrations")
    if not isinstance(test, list) or not test:
        raise ValueError("blind structural signature requires query inputs")
    content = {
        "train": [
            {
                "input": _grid_signature(pair["input"]),
                "output": _grid_signature(pair["output"]),
                "change": _change_signature(pair["input"], pair["output"]),
            }
            for pair in train
        ],
        "test_inputs": [_grid_signature(pair["input"]) for pair in test],
    }
    return canonical_sha256(content)


def _development_signature(arcgen_root: Path) -> str:
    sys.path.insert(0, str(arcgen_root.resolve()))
    generator = importlib.import_module("task_list").task_list()[
        DEVELOPMENT_FAMILY_ID
    ][0]
    pairs = []
    for pair_index in range(4):
        random.seed(_pair_seed(DEVELOPMENT_FAMILY_ID, pair_index))
        generated = generator()
        pairs.append(
            {
                "input": grid_to_lists(as_grid(generated["input"])),
                "output": grid_to_lists(as_grid(generated["output"])),
            }
        )
    payload = {
        "train": pairs[:3],
        "test": [{"input": pairs[3]["input"], "output": pairs[3]["input"]}],
    }
    return blind_structural_signature(payload)


def _load_exposure_registry(path: Path) -> tuple[set[str], dict[str, object]]:
    payload = _load_object(path)
    if payload["schema"] != EXPOSURE_REGISTRY_SCHEMA:
        raise ValueError("exposure registry schema mismatch")
    task_ids = payload["task_ids"]
    if not isinstance(task_ids, list) or len(task_ids) != len(set(task_ids)):
        raise ValueError("exposure registry task IDs are malformed")
    content = {
        "schema": payload["schema"],
        "sources": payload["sources"],
        "task_ids": task_ids,
    }
    if payload["registry_id"] != canonical_sha256(content):
        raise ValueError("exposure registry ID mismatch")
    return set(task_ids), {
        "registry_id": payload["registry_id"],
        "file_sha256": _file_sha256(path),
        "task_count": len(task_ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("arc2_root", type=Path)
    parser.add_argument("arcgen_root", type=Path)
    parser.add_argument("exposure_registry", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"cohort output already exists: {args.output_dir}")
    if _git_head(args.arc2_root) != EXPECTED_ARC2_COMMIT:
        raise ValueError("ARC-AGI-2 commit mismatch")
    if _git_head(args.arcgen_root) != EXPECTED_ARCGEN_COMMIT:
        raise ValueError("ARC-GEN commit mismatch")
    arc1_root = args.arcgen_root / "external" / "ARC-AGI"
    if _git_head(arc1_root) != EXPECTED_ARC1_COMMIT:
        raise ValueError("ARC-AGI-1 submodule commit mismatch")

    arc2_training = args.arc2_root / "data" / "training"
    arc2_ids = {path.stem for path in arc2_training.glob("*.json")}
    if len(arc2_ids) != EXPECTED_ARC2_TRAINING_COUNT:
        raise ValueError("ARC-AGI-2 training identity source is incomplete")
    arc1_split_ids = tuple(
        {
            path.stem
            for path in (arc1_root / "data" / split).glob("*.json")
        }
        for split in ("training", "evaluation")
    )
    if any(len(ids) != EXPECTED_ARC1_SPLIT_COUNT for ids in arc1_split_ids):
        raise ValueError("ARC-AGI-1 identity source is incomplete")
    arc1_ids = arc1_split_ids[0] | arc1_split_ids[1]
    exposure_ids, exposure_record = _load_exposure_registry(
        args.exposure_registry
    )
    eligible = arc2_ids - arc1_ids - exposure_ids
    if len(eligible) != EXPECTED_ELIGIBLE_COUNT:
        raise ValueError(
            f"eligible ARC-2 task count changed: {len(eligible)}"
        )
    ordered = tuple(
        sorted(
            eligible,
            key=lambda task_id: (
                hashlib.sha256(
                    f"{SELECTION_SEED}\0{task_id}".encode("ascii")
                ).hexdigest(),
                task_id,
            ),
        )
    )
    development_signature = _development_signature(args.arcgen_root)
    retained: list[dict[str, object]] = []
    seen_signatures: set[str] = set()
    duplicate_signature_count = 0
    development_signature_exclusion_count = 0
    for task_id in ordered:
        source_path = arc2_training / f"{task_id}.json"
        source_payload = _load_object(source_path)
        blind_payload = make_query_blind_payload(source_payload)
        signature = blind_structural_signature(blind_payload)
        if signature == development_signature:
            development_signature_exclusion_count += 1
            continue
        if signature in seen_signatures:
            duplicate_signature_count += 1
            continue
        seen_signatures.add(signature)
        blind_path = args.output_dir / "blind_tasks" / source_path.name
        _write_new(blind_path, blind_payload)
        retained.append(
            {
                "task_id": task_id,
                "source_sha256": _file_sha256(source_path),
                "blind_sha256": _file_sha256(blind_path),
                "blind_structural_signature": signature,
                "query_count": len(source_payload["test"]),
            }
        )
    if not retained:
        raise ValueError("structural deduplication removed the entire cohort")

    content = {
        "schema": COHORT_SCHEMA,
        "selection": {
            "source_split": "ARC-AGI-2/training",
            "seed": SELECTION_SEED,
            "algorithm": "sha256(seed + NUL + task_id), ascending",
            "arc2_training_count": len(arc2_ids),
            "arc1_task_count": len(arc1_ids),
            "arc2_arc1_overlap_count": len(arc2_ids & arc1_ids),
            "exposure_task_count": len(exposure_ids),
            "eligible_count": len(eligible),
            "duplicate_signature_count": duplicate_signature_count,
            "development_signature_exclusion_count": (
                development_signature_exclusion_count
            ),
            "retained_count": len(retained),
            "structural_family_claim": (
                "task-ID disjoint plus blind structural-signature deduplication; "
                "not ground-truth semantic family labels"
            ),
        },
        "source": {
            "arc2_commit": EXPECTED_ARC2_COMMIT,
            "arc2_training_dir": str(arc2_training.resolve()),
            "arcgen_commit": EXPECTED_ARCGEN_COMMIT,
            "arc1_submodule_commit": EXPECTED_ARC1_COMMIT,
            "exposure_registry": exposure_record,
        },
        "query_blind_contract": {
            "test_output_sentinel": "exact copy of test input",
            "selection_reads": "task IDs, demonstrations, and query inputs",
            "selection_does_not_read": "query outputs or evaluation split",
            "gold_release": "after content-addressed candidate freeze and replay",
        },
        "development_signature": {
            "family_id": DEVELOPMENT_FAMILY_ID,
            "blind_structural_signature": development_signature,
            "role": "excluded development-only representation probe",
        },
        "blind_task_root": str((args.output_dir / "blind_tasks").resolve()),
        "tasks": retained,
    }
    manifest = {"cohort_id": canonical_sha256(content), **content}
    _write_new(args.output_dir / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "cohort_id": manifest["cohort_id"],
                "eligible_count": len(eligible),
                "retained_count": len(retained),
                "duplicate_signature_count": duplicate_signature_count,
                "development_signature_exclusion_count": (
                    development_signature_exclusion_count
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
