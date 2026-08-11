from __future__ import annotations

import json
from pathlib import Path

import pytest

from afts_arc.cohort_eligibility import (
    build_collision_free_amendment,
    find_demo_query_input_collisions,
)
from afts_arc.experiment_safety import canonical_sha256


def _artifacts() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    challenges = {
        "collision": {
            "train": [{"input": [[1]], "output": [[2]]}],
            "test": [{"input": [[1]]}],
        },
        "eligible": {
            "train": [{"input": [[3]], "output": [[4]]}],
            "test": [{"input": [[5]]}],
        },
    }
    solutions = {"collision": [[[2]]], "eligible": [[[6]]]}
    witnesses = {"collision": {"trace": 1}, "eligible": {"trace": 2}}
    return challenges, solutions, witnesses


def _source_paths(tmp_path: Path, artifacts: tuple[dict[str, object], ...]) -> dict[str, Path]:
    paths = {}
    for name, value in zip(("challenges", "solutions", "witnesses"), artifacts):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        paths[name] = path
    return paths


def test_collision_filter_is_query_gold_free_and_uses_no_replacement(
    tmp_path: Path,
) -> None:
    challenges, solutions, witnesses = _artifacts()
    source_paths = _source_paths(tmp_path, (challenges, solutions, witnesses))

    result = build_collision_free_amendment(
        challenges=challenges,
        solutions=solutions,
        witnesses=witnesses,
        source_paths=source_paths,
    )

    assert result["challenges"] == {"eligible": challenges["eligible"]}
    assert result["solutions"] == {"eligible": solutions["eligible"]}
    assert result["witnesses"] == {"eligible": witnesses["eligible"]}
    amendment = result["amendment"]
    assert amendment["eligibility_decision_id"] == canonical_sha256(
        {
            key: value
            for key, value in amendment.items()
            if key != "eligibility_decision_id"
        }
    )
    assert amendment["query_gold_used_for_eligibility"] is False
    assert amendment["replacement_policy"] == "none"
    assert amendment["counts"] == {
        "source_tasks": 2,
        "eligible_tasks": 1,
        "excluded_tasks": 1,
    }
    assert amendment["excluded"] == [
        {
            "task_id": "collision",
            "reason": "exact_demo_query_input_collision",
            "collision_indices": [{"demo_index": 0, "query_index": 0}],
        }
    ]


def test_collision_detection_rejects_query_gold_in_challenge() -> None:
    challenges, _, _ = _artifacts()
    challenges["eligible"]["test"][0]["output"] = [[6]]

    with pytest.raises(ValueError, match="query must contain input only"):
        find_demo_query_input_collisions(challenges)


def test_amendment_rejects_misaligned_artifact_task_ids(tmp_path: Path) -> None:
    challenges, solutions, witnesses = _artifacts()
    del witnesses["collision"]
    source_paths = _source_paths(tmp_path, (challenges, solutions, witnesses))

    with pytest.raises(ValueError, match="task IDs must match"):
        build_collision_free_amendment(
            challenges=challenges,
            solutions=solutions,
            witnesses=witnesses,
            source_paths=source_paths,
        )
