"""Post-hoc identity-normalized audit of relational-mask parent quality."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = str((PROJECT_ROOT / "src").resolve())
if SOURCE_ROOT not in sys.path:
    sys.path.insert(0, SOURCE_ROOT)

from afts_arc.experiment_safety import atomic_write_json, canonical_sha256  # noqa: E402
from afts_arc.grid import Grid, as_grid  # noqa: E402
from afts_arc.relational_mask import (  # noqa: E402
    RelationalMaskProgram,
    execute_relational_mask,
)


DIAGNOSTIC_SCHEMA = "afts.relational-mask-near-miss-quality/v1"
EXPECTED_FREEZE_ID = "ddeb3e4c6d057d927359b22eb2e7e5a7369e80be5d9c0298112337d6835b6556"
DELTA_F1_THRESHOLD = 0.5

Coordinate = tuple[int, int]


def _load_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON artifact must be an object: {path}")
    return payload


def _delta(first: Grid, second: Grid) -> set[Coordinate]:
    if (len(first), len(first[0])) != (len(second), len(second[0])):
        raise ValueError("near-miss quality requires same-shape grids")
    return {
        (row, column)
        for row in range(len(first))
        for column in range(len(first[0]))
        if first[row][column] != second[row][column]
    }


def parent_quality(
    parent: RelationalMaskProgram,
    train: list[object],
) -> dict[str, object]:
    rows = []
    identity_mismatches = 0
    parent_mismatches = 0
    intersection_count = 0
    parent_delta_count = 0
    gold_delta_count = 0
    for index, pair in enumerate(train):
        source = as_grid(pair["input"])
        gold = as_grid(pair["output"])
        execution = execute_relational_mask(parent, source)
        if not execution.ok or execution.output is None:
            raise ValueError("post-hoc parent is not execution-valid")
        parent_output = execution.output
        gold_delta = _delta(source, gold)
        predicted_delta = _delta(source, parent_output)
        intersection = len(gold_delta & predicted_delta)
        identity_mismatch = len(gold_delta)
        parent_mismatch = len(_delta(parent_output, gold))
        identity_mismatches += identity_mismatch
        parent_mismatches += parent_mismatch
        intersection_count += intersection
        parent_delta_count += len(predicted_delta)
        gold_delta_count += len(gold_delta)
        rows.append(
            {
                "demo_index": index,
                "identity_mismatch_count": identity_mismatch,
                "parent_mismatch_count": parent_mismatch,
                "gold_delta_count": len(gold_delta),
                "parent_delta_count": len(predicted_delta),
                "delta_intersection_count": intersection,
            }
        )
    precision = (
        intersection_count / parent_delta_count if parent_delta_count else 0.0
    )
    recall = intersection_count / gold_delta_count if gold_delta_count else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    improves_identity = parent_mismatches < identity_mismatches
    return {
        "identity_mismatch_count": identity_mismatches,
        "parent_mismatch_count": parent_mismatches,
        "parent_improves_identity": improves_identity,
        "delta_precision": precision,
        "delta_recall": recall,
        "delta_f1": f1,
        "quality_near_miss": improves_identity and f1 >= DELTA_F1_THRESHOLD,
        "demos": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate_freeze", type=Path)
    parser.add_argument("cohort_manifest", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()
    if args.output_path.exists():
        raise FileExistsError(f"diagnostic output already exists: {args.output_path}")
    freeze = _load_object(args.candidate_freeze)
    manifest = _load_object(args.cohort_manifest)
    if freeze["freeze_id"] != EXPECTED_FREEZE_ID:
        raise ValueError("candidate freeze ID mismatch")
    if freeze["query_gold_read"] is not False:
        raise ValueError("candidate freeze is not query blind")
    if freeze["cohort_id"] != manifest["cohort_id"]:
        raise ValueError("candidate freeze and cohort differ")
    blind_root = Path(manifest["blind_task_root"])
    rows = []
    for task in freeze["tasks"]:
        parent = task["repair"]["parent"]
        if parent is None:
            continue
        blind = _load_object(blind_root / f"{task['task_id']}.json")
        quality = parent_quality(
            RelationalMaskProgram.from_json_dict(parent["program"]),
            blind["train"],
        )
        rows.append(
            {
                "task_id": task["task_id"],
                "preregistered_pixel_agreement": parent["agreement"],
                **quality,
            }
        )
    content = {
        "schema": DIAGNOSTIC_SCHEMA,
        "status": "posthoc_diagnostic_not_preregistered",
        "candidate_freeze_id": freeze["freeze_id"],
        "cohort_id": freeze["cohort_id"],
        "query_gold_read": False,
        "criterion": {
            "parent_improves_identity": True,
            "minimum_delta_f1": DELTA_F1_THRESHOLD,
        },
        "preregistered_parent_count": len(rows),
        "quality_near_miss_count": sum(row["quality_near_miss"] for row in rows),
        "background_dominated_false_near_miss_count": sum(
            not row["quality_near_miss"] for row in rows
        ),
        "tasks": rows,
    }
    result = {"diagnostic_id": canonical_sha256(content), **content}
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
