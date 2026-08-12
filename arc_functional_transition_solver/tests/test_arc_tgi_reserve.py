from __future__ import annotations

import pytest

from afts_arc.arc_tgi_reserve import (
    authorize_oracle_open,
    authorize_population_oracle_open,
    build_collision_free_reserve_subset,
    build_reserve_seal,
    opened_solution_payload,
    reserve_episode_schedule,
)
from afts_arc.experiment_safety import canonical_sha256
from afts_arc.functional_recruitment import freeze_functional_recruitment_plan
from afts_arc.hypothesis_population import freeze_hypothesis_population
from afts_arc.nvarc_anchor import freeze_submission
from afts_arc.varc_blind import build_query_blind_tasks
from afts_arc.varc_candidates import freeze_varc_predictions


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


def test_population_oracle_requires_bound_query_blind_population_and_plan() -> None:
    episode = _episode()
    challenges, seal = build_reserve_seal(
        episodes=(episode,),
        cohort_id="cohort-a",
        partition_id="partition-a",
        source_files={"protocol": "sha256"},
    )
    attempts = {f"attempt_{rank}": [[rank % 2]] for rank in range(1, 11)}
    anchor = freeze_submission(
        challenges=challenges,
        submission={"reserve-task": [attempts]},
        cohort_id="cohort-a",
        anchor_run_id="anchor-run",
        source_files={"challenges": "a" * 64, "submission": "b" * 64},
    )
    _, blind_manifest = build_query_blind_tasks(
        challenges=challenges,
        source_cohort_id="cohort-a",
        source_challenges_sha256="a" * 64,
    )
    visual = freeze_varc_predictions(
        challenges=challenges,
        blind_manifest=blind_manifest,
        predictions={"reserve-task": {"0": [[[4]], [[4]], [[3]]]}},
        provider_contract={"checkpoint": "visual", "seed": 42},
        source_files={"challenges": "a" * 64, "predictions": "c" * 64},
    )
    plan = freeze_functional_recruitment_plan(
        anchor_freeze=anchor,
        cohort_id="cohort-a",
        anchor_provider_name="recursive",
        recruited_provider_name="visual",
        budget_percentages=(100,),
        random_seed_count=2,
        source_files={
            "anchor_candidate_freeze": "d" * 64,
            "anchor_cost_receipt": "e" * 64,
            "protocol": "f" * 64,
            "task_universe_manifest": "0" * 64,
        },
        task_universe=("reserve-task",),
    )
    population = freeze_hypothesis_population(
        cohort_id="cohort-a",
        provider_freezes={"recursive": anchor, "visual": visual},
        provider_families={
            "recursive": "recursive-test-time-learning",
            "visual": "visual-test-time-training",
        },
        provider_freeze_sha256s={"recursive": "1" * 64, "visual": "2" * 64},
        provider_receipt_sha256s={"recursive": "3" * 64, "visual": "4" * 64},
    )

    authorization = authorize_population_oracle_open(
        seal=seal,
        anchor_freeze=anchor,
        recruited_freeze=visual,
        population=population,
        recruitment_plan=plan,
    )
    assert authorization["authorized"] is True
    assert authorization["task_count"] == 1

    forged_plan = dict(plan)
    forged_plan["anchor_candidate_freeze_id"] = "5" * 64
    forged_body = {
        key: value for key, value in forged_plan.items() if key != "plan_id"
    }
    forged_plan["plan_id"] = canonical_sha256(forged_body)
    with pytest.raises(ValueError, match="anchor freeze differs"):
        authorize_population_oracle_open(
            seal=seal,
            anchor_freeze=anchor,
            recruited_freeze=visual,
            population=population,
            recruitment_plan=forged_plan,
        )


def test_collision_free_subset_uses_only_blind_inputs_and_retains_oracle_hashes() -> None:
    colliding = _episode()
    clean = dict(_episode())
    clean["task_id"] = "clean-task"
    clean["episode_id"] = "clean-episode"
    clean["blind"] = {
        "train": [{"input": [[1]], "output": [[2]]}],
        "test": [{"input": [[5]]}],
    }
    clean["blind_content_sha256"] = canonical_sha256(clean["blind"])
    colliding["blind"] = {
        "train": [{"input": [[3]], "output": [[2]]}],
        "test": [{"input": [[3]]}],
    }
    colliding["blind_content_sha256"] = canonical_sha256(colliding["blind"])
    challenges, parent_seal = build_reserve_seal(
        episodes=(colliding, clean),
        cohort_id="parent-cohort",
        partition_id="partition-a",
        source_files={"protocol": "sha256"},
    )

    eligible, subset_seal = build_collision_free_reserve_subset(
        challenges=challenges,
        seal=parent_seal,
        source_files={"parent": "a" * 64},
    )

    assert set(eligible) == {"clean-task"}
    assert subset_seal["query_gold_written"] is False
    assert subset_seal["parent_seal_id"] == parent_seal["seal_id"]
    assert subset_seal["eligibility"]["excluded_tasks"] == [
        {
            "query_indices": [0],
            "reason": "demo_query_input_collision",
            "task_id": "reserve-task",
        }
    ]
    assert subset_seal["tasks"][0]["oracle_sha256"] == clean["oracle_sha256"]
