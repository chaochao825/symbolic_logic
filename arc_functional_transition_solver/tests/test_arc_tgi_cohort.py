from __future__ import annotations

from scripts.afts_arc_tgi_cohort import (
    MANUALLY_EXPOSED_FAMILIES,
    QUARANTINED_FAMILIES,
    canonical_family_name,
    partition_generator_sources,
    split_full_record,
)


def _sources() -> list[dict[str, object]]:
    family_ids = [
        *MANUALLY_EXPOSED_FAMILIES,
        *QUARANTINED_FAMILIES,
        *(f"family_{index:03d}" for index in range(167)),
    ]
    return [
        {
            "family_id": family_id,
            "family_source_id": f"{index:064x}",
            "selected_relative_path": f"{family_id}.py",
            "selected_source_sha256": (
                QUARANTINED_FAMILIES[family_id]["selected_source_sha256"]
                if family_id in QUARANTINED_FAMILIES
                else f"{index + 1000:064x}"
            ),
            "variants": [],
            "manually_exposed_before_partition": (
                family_id in MANUALLY_EXPOSED_FAMILIES
            ),
        }
        for index, family_id in enumerate(family_ids)
    ]


def test_canonical_family_name_merges_only_known_revision_suffixes() -> None:
    assert canonical_family_name("taskABC_1") == "taskABC"
    assert canonical_family_name("taskABC_new") == "taskABC"
    assert canonical_family_name("taskABC-new") == "taskABC"
    assert canonical_family_name("taskABC_2") == "taskABC_2"


def test_family_partition_is_deterministic_disjoint_and_exposure_safe() -> None:
    first = partition_generator_sources(_sources())
    second = partition_generator_sources(_sources())

    assert first == second
    assert len(first["development"]) == 50
    assert len(first["confirmatory"]) == 100
    assert len(first["reserve"]) == 18
    assert len(first["quarantine"]) == 2
    development = {row["family_id"] for row in first["development"]}
    confirmatory = {row["family_id"] for row in first["confirmatory"]}
    reserve = {row["family_id"] for row in first["reserve"]}
    quarantine = {row["family_id"] for row in first["quarantine"]}
    assert MANUALLY_EXPOSED_FAMILIES <= development
    assert not development & confirmatory
    assert not development & reserve
    assert not confirmatory & reserve
    assert quarantine == set(QUARANTINED_FAMILIES)
    assert not quarantine & (development | confirmatory | reserve)


def test_split_full_record_keeps_query_gold_and_witness_out_of_challenge() -> None:
    full = {
        "train": [{"input": [[1]], "output": [[2]]}],
        "test": [{"input": [[3]], "output": [[4]]}],
        "input_reasoning_chain": ["input"],
        "transformation_reasoning_chain": ["transform"],
        "task_variables": {"color": 4},
        "partial_transform_code": "def transform_input(self, grid): return grid",
    }

    blind, oracle, witness = split_full_record(full)

    assert blind == {
        "train": [{"input": [[1]], "output": [[2]]}],
        "test": [{"input": [[3]]}],
    }
    assert oracle == {"test_outputs": [[[4]]]}
    assert "output" not in blind["test"][0]
    assert witness["task_variables"] == {"color": 4}
