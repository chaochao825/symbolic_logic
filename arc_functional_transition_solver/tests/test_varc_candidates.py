from __future__ import annotations

from copy import deepcopy

import pytest

from afts_arc.experiment_safety import canonical_sha256
from afts_arc.varc_blind import build_query_blind_tasks
from afts_arc.varc_candidates import freeze_varc_predictions, score_varc_freeze


def _challenges() -> dict[str, object]:
    return {
        "task_a": {
            "train": [{"input": [[1]], "output": [[2]]}],
            "test": [{"input": [[3]]}, {"input": [[4]]}],
        },
        "task_b": {
            "train": [{"input": [[5]], "output": [[6]]}],
            "test": [{"input": [[7]]}],
        },
    }


def _manifest() -> dict[str, object]:
    _, manifest = build_query_blind_tasks(
        challenges=_challenges(),
        source_cohort_id="source",
        source_challenges_sha256="a" * 64,
    )
    return manifest


def _predictions() -> dict[str, object]:
    return {
        "task_a": {
            "0": [[[8]], [[8]], [[0]], [[10]]],
            "1": [[[0]], [[9]], [[9]]],
        },
        "task_b": {"0": [[[1]], [[1]], [[0]]]},
    }


def _freeze() -> dict[str, object]:
    return freeze_varc_predictions(
        challenges=_challenges(),
        blind_manifest=_manifest(),
        predictions=_predictions(),
        provider_contract={"checkpoint": "hash", "seed": 42},
        source_files={"challenges": "a" * 64, "predictions": "b" * 64},
    )


def test_freeze_is_query_gold_free_ranked_deduplicated_and_content_addressed() -> None:
    first = _freeze()
    second = _freeze()

    assert first == second
    assert first["query_gold_read"] is False
    assert first["freeze_id"] == canonical_sha256(
        {key: value for key, value in first.items() if key != "freeze_id"}
    )
    assert first["aggregate"] == {
        "raw_sample_count": 10,
        "rejected_sample_count": 1,
        "unique_candidate_count": 6,
        "valid_sample_count": 9,
    }
    query = first["tasks"][0]["queries"][0]
    assert [candidate["output"] for candidate in query["candidates"]] == [
        [[8]],
        [[0]],
    ]
    assert query["candidates"][0]["sample_count"] == 2


def test_freeze_rejects_task_query_and_challenge_hash_mismatch() -> None:
    missing_task = _predictions()
    del missing_task["task_b"]
    with pytest.raises(ValueError, match="task sets differ"):
        freeze_varc_predictions(
            challenges=_challenges(),
            blind_manifest=_manifest(),
            predictions=missing_task,
            provider_contract={},
            source_files={"challenges": "a" * 64},
        )

    wrong_query = _predictions()
    del wrong_query["task_a"]["1"]
    with pytest.raises(ValueError, match="query indices differ"):
        freeze_varc_predictions(
            challenges=_challenges(),
            blind_manifest=_manifest(),
            predictions=wrong_query,
            provider_contract={},
            source_files={"challenges": "a" * 64},
        )

    with pytest.raises(ValueError, match="SHA-256 differs"):
        freeze_varc_predictions(
            challenges=_challenges(),
            blind_manifest=_manifest(),
            predictions=_predictions(),
            provider_contract={},
            source_files={"challenges": "c" * 64},
        )


def test_score_separates_rank_two_raw_and_strict_task_endpoints() -> None:
    result = score_varc_freeze(
        freeze=_freeze(),
        solutions={"task_a": [[[8]], [[0]]], "task_b": [[[1]]]},
        solution_source_sha256="d" * 64,
    )

    assert result["metrics"]["1"]["strict_task_pass"]["correct"] == 1
    assert result["metrics"]["2"]["strict_task_pass"]["correct"] == 2
    assert result["metrics"]["raw"]["strict_task_pass"]["correct"] == 2
    assert result["metrics"]["1"]["query_pass"]["correct"] == 2
    assert result["metrics"]["1"]["task_mean_pair_pass"]["rate"] == 0.75
    assert result["query_gold_read"] is True


def test_score_rejects_forged_freeze_and_solution_task_set() -> None:
    forged = deepcopy(_freeze())
    forged["tasks"][0]["queries"][0]["candidates"][0]["output"] = [[7]]
    with pytest.raises(ValueError, match="freeze_id"):
        score_varc_freeze(
            freeze=forged,
            solutions={"task_a": [[[8]], [[0]]], "task_b": [[[1]]]},
            solution_source_sha256="d" * 64,
        )

    with pytest.raises(ValueError, match="task sets differ"):
        score_varc_freeze(
            freeze=_freeze(),
            solutions={"task_a": [[[8]], [[0]]]},
            solution_source_sha256="d" * 64,
        )
