"""Two-phase sealing and oracle authorization for the ARC-TGI reserve lane."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .experiment_safety import canonical_sha256


ARC_TGI_RESERVE_SEAL_SCHEMA = "afts.arc-tgi-reserve-seal/v1"
ARC_TGI_ORACLE_AUTHORIZATION_SCHEMA = "afts.arc-tgi-oracle-authorization/v1"
POPULATION_ORACLE_AUTHORIZATION_SCHEMA = (
    "afts.population-recruitment-oracle-authorization/v1"
)


def reserve_episode_schedule(
    family_ids: Sequence[str],
    *,
    episode_count: int,
) -> tuple[tuple[str, int], ...]:
    """Allocate deterministic indexed episodes as evenly as possible by family."""

    if not family_ids or any(not isinstance(item, str) or not item for item in family_ids):
        raise ValueError("reserve schedule requires non-empty family IDs")
    if len(set(family_ids)) != len(family_ids):
        raise ValueError("reserve schedule repeats a family ID")
    if type(episode_count) is not int or episode_count < len(family_ids):
        raise ValueError("reserve episode count must cover every family")
    ordered = tuple(sorted(family_ids))
    quotient, remainder = divmod(episode_count, len(ordered))
    return tuple(
        (family_id, episode_index)
        for family_offset, family_id in enumerate(ordered)
        for episode_index in range(quotient + int(family_offset < remainder))
    )


def _object(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise TypeError(f"{field} keys must be strings")
    return value


def build_reserve_seal(
    *,
    episodes: Sequence[Mapping[str, object]],
    cohort_id: str,
    partition_id: str,
    source_files: Mapping[str, str],
) -> tuple[dict[str, object], dict[str, object]]:
    """Commit oracle hashes while returning only blind challenges to solvers."""

    if not episodes:
        raise ValueError("reserve seal requires at least one episode")
    challenges: dict[str, object] = {}
    task_rows = []
    for raw_episode in episodes:
        episode = _object(raw_episode, field="reserve episode")
        task_id = episode["task_id"]
        if not isinstance(task_id, str) or not task_id or task_id in challenges:
            raise ValueError("reserve episode task ID is invalid or duplicated")
        blind = _object(episode["blind"], field="reserve blind task")
        test = blind["test"]
        if not isinstance(test, list) or any(
            not isinstance(query, Mapping) or set(query) != {"input"}
            for query in test
        ):
            raise ValueError("reserve blind challenge exposes a query output")
        challenges[task_id] = dict(blind)
        task_rows.append(
            {
                "task_id": task_id,
                "episode_id": episode["episode_id"],
                "family_id": episode["family_id"],
                "family_source_id": episode["family_source_id"],
                "selected_source_sha256": episode["selected_source_sha256"],
                "blind_content_sha256": episode["blind_content_sha256"],
                "oracle_sha256": episode["oracle_sha256"],
                "witness_sha256": episode["witness_sha256"],
            }
        )
    task_rows.sort(key=lambda row: row["task_id"])
    content: dict[str, object] = {
        "schema": ARC_TGI_RESERVE_SEAL_SCHEMA,
        "status": "blind_frozen_oracle_hash_sealed",
        "cohort_id": cohort_id,
        "partition_id": partition_id,
        "query_gold_written": False,
        "witness_written": False,
        "solver_artifact": "reserve_challenges.json",
        "task_count": len(task_rows),
        "challenge_content_sha256": canonical_sha256(challenges),
        "source_files": dict(sorted(source_files.items())),
        "tasks": task_rows,
    }
    return challenges, {"seal_id": canonical_sha256(content), **content}


def authorize_oracle_open(
    *,
    seal: Mapping[str, object],
    candidate_freeze: Mapping[str, object],
    minimum_opportunities: int = 2,
) -> dict[str, object]:
    """Authorize oracle regeneration only after the prospective demo-only gate."""

    if type(minimum_opportunities) is not int or minimum_opportunities < 1:
        raise ValueError("minimum opportunities must be a positive integer")
    seal_content = {key: value for key, value in seal.items() if key != "seal_id"}
    if seal["seal_id"] != canonical_sha256(seal_content):
        raise ValueError("reserve seal ID differs from canonical content")
    freeze_content = {
        key: value for key, value in candidate_freeze.items() if key != "freeze_id"
    }
    if candidate_freeze["freeze_id"] != canonical_sha256(freeze_content):
        raise ValueError("reserve candidate freeze ID differs from canonical content")
    if candidate_freeze["scientific_lane"] != "prospective_reserve":
        raise ValueError("oracle authorization requires the prospective reserve lane")
    if candidate_freeze["query_gold_read"] is not False:
        raise ValueError("reserve candidate freeze is not query-gold-free")
    if candidate_freeze["cohort_id"] != seal["cohort_id"]:
        raise ValueError("reserve seal and candidate cohort IDs differ")
    sealed_tasks = {
        _object(item, field="sealed task")["task_id"]
        for item in seal["tasks"]
    }
    frozen_tasks = {
        _object(item, field="frozen task")["task_id"]
        for item in candidate_freeze["tasks"]
    }
    if sealed_tasks != frozen_tasks:
        raise ValueError("reserve seal and candidate task sets differ")
    opportunity_count = candidate_freeze["opportunity_count"]
    if type(opportunity_count) is not int:
        raise TypeError("candidate opportunity count must be an integer")
    if opportunity_count < minimum_opportunities:
        raise PermissionError("prospective demo-only frontier gate did not pass")
    content: dict[str, object] = {
        "schema": ARC_TGI_ORACLE_AUTHORIZATION_SCHEMA,
        "seal_id": seal["seal_id"],
        "freeze_id": candidate_freeze["freeze_id"],
        "minimum_opportunities": minimum_opportunities,
        "observed_opportunities": opportunity_count,
        "task_count": len(sealed_tasks),
        "authorized": True,
    }
    return {"authorization_id": canonical_sha256(content), **content}


def authorize_population_oracle_open(
    *,
    seal: Mapping[str, object],
    anchor_freeze: Mapping[str, object],
    recruited_freeze: Mapping[str, object],
    population: Mapping[str, object],
    recruitment_plan: Mapping[str, object],
) -> dict[str, object]:
    """Authorize gold only after a query-blind population and plan are frozen."""

    from .functional_recruitment import _validated_plan
    from .hypothesis_population import _validated_population
    from .nvarc_anchor import _validated_candidates as _validated_nvarc_candidates
    from .varc_candidates import _validated_freeze as _validated_varc_freeze

    seal_content = {key: value for key, value in seal.items() if key != "seal_id"}
    if seal["seal_id"] != canonical_sha256(seal_content):
        raise ValueError("reserve seal ID differs from canonical content")
    if seal["query_gold_written"] is not False:
        raise ValueError("reserve seal already exposes query gold")
    sealed_task_ids = {
        _object(item, field="sealed task")["task_id"] for item in seal["tasks"]
    }

    anchor_cohort_id, anchor_tasks = _validated_nvarc_candidates(anchor_freeze)
    _, recruited_tasks = _validated_varc_freeze(recruited_freeze)
    if anchor_cohort_id != seal["cohort_id"]:
        raise ValueError("anchor candidate cohort differs from reserve seal")
    if not set(anchor_tasks) <= sealed_task_ids:
        raise ValueError("anchor candidate freeze contains an unsealed task")
    if set(recruited_tasks) != sealed_task_ids:
        raise ValueError("recruited candidate task set differs from reserve seal")

    provider_names, population_tasks = _validated_population(population)
    priority_order, _ = _validated_plan(recruitment_plan)
    if set(population_tasks) != sealed_task_ids:
        raise ValueError("population task set differs from reserve seal")
    if set(priority_order) != sealed_task_ids:
        raise ValueError("recruitment plan task set differs from reserve seal")
    if population["cohort_id"] != seal["cohort_id"]:
        raise ValueError("population cohort differs from reserve seal")
    if recruitment_plan["cohort_id"] != seal["cohort_id"]:
        raise ValueError("recruitment plan cohort differs from reserve seal")

    anchor_name = recruitment_plan["anchor_provider_name"]
    recruited_name = recruitment_plan["recruited_provider_name"]
    if set(provider_names) != {anchor_name, recruited_name}:
        raise ValueError("population providers differ from recruitment plan")
    provider_freeze_ids = {
        _object(row, field="population provider")["name"]: _object(
            row, field="population provider"
        )["candidate_freeze_id"]
        for row in population["providers"]
    }
    expected_freeze_ids = {
        anchor_name: anchor_freeze["freeze_id"],
        recruited_name: recruited_freeze["freeze_id"],
    }
    if provider_freeze_ids != expected_freeze_ids:
        raise ValueError("population provider freezes differ from supplied freezes")
    if recruitment_plan["anchor_candidate_freeze_id"] != anchor_freeze["freeze_id"]:
        raise ValueError("recruitment plan anchor freeze differs")

    content: dict[str, object] = {
        "anchor_candidate_freeze_id": anchor_freeze["freeze_id"],
        "authorized": True,
        "population_id": population["population_id"],
        "recruited_candidate_freeze_id": recruited_freeze["freeze_id"],
        "recruitment_plan_id": recruitment_plan["plan_id"],
        "schema": POPULATION_ORACLE_AUTHORIZATION_SCHEMA,
        "seal_id": seal["seal_id"],
        "task_count": len(sealed_task_ids),
    }
    return {"authorization_id": canonical_sha256(content), **content}


def opened_solution_payload(
    *,
    seal: Mapping[str, object],
    episodes: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Verify deterministic regeneration against the seal before returning gold."""

    sealed = {
        row["task_id"]: row
        for row in (
            _object(item, field="sealed task") for item in seal["tasks"]
        )
    }
    solutions: dict[str, object] = {}
    for raw_episode in episodes:
        episode = _object(raw_episode, field="regenerated reserve episode")
        task_id = episode["task_id"]
        if task_id not in sealed:
            raise ValueError("regenerated reserve task is not sealed")
        row = sealed[task_id]
        for field in (
            "episode_id",
            "family_id",
            "family_source_id",
            "selected_source_sha256",
            "blind_content_sha256",
            "oracle_sha256",
            "witness_sha256",
        ):
            if episode[field] != row[field]:
                raise ValueError(f"regenerated reserve {field} differs from seal")
        oracle = _object(episode["oracle"], field="regenerated reserve oracle")
        solutions[task_id] = oracle["test_outputs"]
    if set(solutions) != set(sealed):
        raise ValueError("regenerated reserve does not cover every sealed task")
    return solutions
