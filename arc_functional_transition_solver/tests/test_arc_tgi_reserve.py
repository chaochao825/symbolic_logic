from __future__ import annotations

import pytest

from afts_arc.arc_tgi_reserve import (
    authorize_oracle_open,
    build_reserve_seal,
    opened_solution_payload,
    reserve_episode_schedule,
)
from afts_arc.experiment_safety import canonical_sha256


def _episode() -> dict[str, object]:
    blind = {
        "train": [{"input": [[1]], "output": [[2]]}],
        "test": [{"input": [[3]]}],
    }
    oracle = {"test_outputs": [[[4]]]}
    witness = {"code": "sealed"}
    return {
        "task_id": "reserve-task",
        "episode_id": "episode-id",
        "family_id": "family-a",
        "family_source_id": "family-source-id",
        "selected_source_sha256": "source-sha256",
        "blind_content_sha256": canonical_sha256(blind),
        "oracle_sha256": canonical_sha256(oracle),
        "witness_sha256": canonical_sha256(witness),
        "blind": blind,
        "oracle": oracle,
        "witness": witness,
    }


def _freeze(seal: dict[str, object], opportunities: int) -> dict[str, object]:
    content = {
        "scientific_lane": "prospective_reserve",
        "query_gold_read": False,
        "cohort_id": seal["cohort_id"],
        "opportunity_count": opportunities,
        "tasks": [{"task_id": "reserve-task"}],
    }
    return {"freeze_id": canonical_sha256(content), **content}


def test_reserve_seal_contains_hashes_but_not_oracle_payload() -> None:
    challenges, seal = build_reserve_seal(
        episodes=(_episode(),),
        cohort_id="cohort-a",
        partition_id="partition-a",
        source_files={"protocol": "sha256"},
    )

    assert challenges["reserve-task"]["test"] == [{"input": [[3]]}]
    assert seal["query_gold_written"] is False
    assert "oracle" not in seal["tasks"][0]
    assert seal["tasks"][0]["oracle_sha256"] == _episode()["oracle_sha256"]


def test_oracle_open_fails_closed_below_demo_only_gate() -> None:
    _, seal = build_reserve_seal(
        episodes=(_episode(),),
        cohort_id="cohort-a",
        partition_id="partition-a",
        source_files={"protocol": "sha256"},
    )

    with pytest.raises(PermissionError, match="did not pass"):
        authorize_oracle_open(
            seal=seal,
            candidate_freeze=_freeze(seal, 1),
            minimum_opportunities=2,
        )


def test_authorized_regeneration_must_match_every_sealed_hash() -> None:
    episode = _episode()
    _, seal = build_reserve_seal(
        episodes=(episode,),
        cohort_id="cohort-a",
        partition_id="partition-a",
        source_files={"protocol": "sha256"},
    )
    authorization = authorize_oracle_open(
        seal=seal,
        candidate_freeze=_freeze(seal, 2),
        minimum_opportunities=2,
    )

    assert authorization["authorized"] is True
    assert opened_solution_payload(seal=seal, episodes=(episode,)) == {
        "reserve-task": [[[4]]]
    }
    changed = dict(episode)
    changed["oracle_sha256"] = "changed"
    with pytest.raises(ValueError, match="oracle_sha256"):
        opened_solution_payload(seal=seal, episodes=(changed,))


def test_reserve_schedule_is_balanced_family_disjoint_and_deterministic() -> None:
    family_ids = tuple(f"family-{index:02d}" for index in reversed(range(18)))
    first = reserve_episode_schedule(family_ids, episode_count=100)
    second = reserve_episode_schedule(family_ids, episode_count=100)

    counts = {
        family_id: sum(item[0] == family_id for item in first)
        for family_id in family_ids
    }
    assert first == second
    assert len(first) == 100
    assert sorted(counts.values()) == [5] * 8 + [6] * 10
    assert {item[0] for item in first} == set(family_ids)

    with pytest.raises(ValueError, match="repeats"):
        reserve_episode_schedule(("family", "family"), episode_count=2)
