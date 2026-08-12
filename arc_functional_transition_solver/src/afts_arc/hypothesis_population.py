"""Content-addressed populations of heterogeneous ARC hypotheses.

The population freeze is deliberately query-gold-free.  Provider-specific
candidate identities remain intact as provenance, while identical raster
hypotheses are merged under a provider-independent content identity.  Scoring
is a separate post-freeze operation so oracle complement cannot influence
candidate construction or provider ranking.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

from afts_arc.experiment_safety import canonical_sha256
from afts_arc.grid import as_grid, grid_to_lists
from afts_arc.nvarc_anchor import (
    CANDIDATE_FREEZE_SCHEMA,
    CANDIDATE_FREEZE_WITH_REJECTIONS_SCHEMA,
    _validated_candidates as _validated_nvarc_candidates,
)
from afts_arc.varc_candidates import (
    VARC_FREEZE_SCHEMA,
    _validated_freeze as _validated_varc_freeze,
)


HYPOTHESIS_POPULATION_SCHEMA = "afts.heterogeneous-hypothesis-population/v1"
HYPOTHESIS_POPULATION_WITH_ABSTENTIONS_SCHEMA = (
    "afts.heterogeneous-hypothesis-population/v2"
)
HYPOTHESIS_POPULATION_RESULT_SCHEMA = (
    "afts.heterogeneous-hypothesis-population-result/v1"
)
HYPOTHESIS_POPULATION_SUMMARY_SCHEMA = (
    "afts.heterogeneous-hypothesis-population-summary/v1"
)


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


def _nonempty_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{field} must be a non-empty string")
    return value


def _sha256(value: object, *, field: str) -> str:
    digest = _nonempty_string(value, field=field)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return digest


def _validate_query_blind_flags(freeze: Mapping[str, object], *, provider: str) -> None:
    if freeze["query_gold_read"] is not False:
        raise ValueError(f"{provider} candidate freeze is not query-gold-free")
    if freeze["controller_training_started"] is not False:
        raise ValueError(f"{provider} candidate freeze was produced after controller training")
    if freeze["public_evaluation_read"] is not False:
        raise ValueError(f"{provider} candidate freeze read public evaluation labels")


def _provider_tasks(
    freeze: Mapping[str, object], *, provider: str
) -> tuple[str, str, Mapping[str, Sequence[object]]]:
    schema = freeze["schema"]
    _validate_query_blind_flags(freeze, provider=provider)
    if schema in {
        CANDIDATE_FREEZE_SCHEMA,
        CANDIDATE_FREEZE_WITH_REJECTIONS_SCHEMA,
    }:
        dataset_id, tasks = _validated_nvarc_candidates(freeze)
        return "nvarc", dataset_id, tasks
    if schema == VARC_FREEZE_SCHEMA:
        dataset_id, tasks = _validated_varc_freeze(freeze)
        return "varc", dataset_id, tasks
    raise ValueError(f"unsupported candidate-freeze schema for {provider}: {schema}")


def _candidate_contribution(
    *,
    adapter: str,
    candidate: Mapping[str, object],
    provider_name: str,
    provider_family: str,
) -> tuple[list[list[int]], dict[str, object]]:
    output = grid_to_lists(as_grid(candidate["output"]))
    provider_candidate_id = _nonempty_string(
        candidate["candidate_id"], field="provider candidate_id"
    )
    if adapter == "nvarc":
        raw_ranks = _sequence(candidate["ranks"], field="NVARC candidate ranks")
        ranks = [int(rank) for rank in raw_ranks]
        contributor = {
            "provider_candidate_id": provider_candidate_id,
            "provider_family": provider_family,
            "provider_name": provider_name,
            "provider_rank": min(ranks),
            "support_count": len(ranks),
            "provider_evidence": {"ranks": ranks},
        }
        return output, contributor
    if adapter == "varc":
        rank = candidate["rank"]
        support = candidate["sample_count"]
        first_index = candidate["first_valid_sample_index"]
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (rank, support, first_index)
        ):
            raise TypeError("VARC rank, support, and first sample index must be integers")
        contributor = {
            "provider_candidate_id": provider_candidate_id,
            "provider_family": provider_family,
            "provider_name": provider_name,
            "provider_rank": rank,
            "support_count": support,
            "provider_evidence": {"first_valid_sample_index": first_index},
        }
        return output, contributor
    raise AssertionError(f"unhandled provider adapter: {adapter}")


def freeze_hypothesis_population(
    *,
    cohort_id: str,
    provider_freezes: Mapping[str, Mapping[str, object]],
    provider_families: Mapping[str, str],
    provider_freeze_sha256s: Mapping[str, str],
    provider_receipt_sha256s: Mapping[str, str],
) -> dict[str, object]:
    """Merge frozen provider outputs without reading query solutions."""

    cohort_id = _nonempty_string(cohort_id, field="cohort_id")
    provider_names = sorted(provider_freezes)
    if len(provider_names) < 2:
        raise ValueError("a heterogeneous population requires at least two providers")
    expected_names = set(provider_names)
    for field, values in (
        ("provider_families", provider_families),
        ("provider_freeze_sha256s", provider_freeze_sha256s),
        ("provider_receipt_sha256s", provider_receipt_sha256s),
    ):
        if set(values) != expected_names:
            raise ValueError(f"{field} provider set differs from provider_freezes")
    families = {
        _nonempty_string(provider_families[name], field=f"family[{name}]")
        for name in provider_names
    }
    if len(families) < 2:
        raise ValueError("a heterogeneous population requires distinct provider families")

    provider_views: dict[str, tuple[str, str, Mapping[str, Sequence[object]]]] = {}
    provider_rows = []
    for name in provider_names:
        freeze = _object(provider_freezes[name], field=f"freeze[{name}]")
        adapter, dataset_id, tasks = _provider_tasks(freeze, provider=name)
        freeze_sha256 = _sha256(
            provider_freeze_sha256s[name], field=f"freeze_sha256[{name}]"
        )
        receipt_sha256 = _sha256(
            provider_receipt_sha256s[name], field=f"receipt_sha256[{name}]"
        )
        provider_views[name] = (adapter, dataset_id, tasks)
        provider_rows.append(
            {
                "adapter": adapter,
                "candidate_freeze_id": _sha256(
                    freeze["freeze_id"], field=f"freeze_id[{name}]"
                ),
                "candidate_freeze_sha256": freeze_sha256,
                "cost_receipt_sha256": receipt_sha256,
                "dataset_id": dataset_id,
                "family": provider_families[name],
                "name": name,
                "provider_schema": freeze["schema"],
                "provider_version": freeze["provider_version"],
            }
        )

    provider_task_sets = {
        name: set(provider_views[name][2]) for name in provider_names
    }
    reference_tasks = set().union(*provider_task_sets.values())
    if not reference_tasks:
        raise ValueError("provider candidate freezes contain no tasks")
    has_abstentions = any(
        provider_task_sets[name] != reference_tasks for name in provider_names
    )

    provider_memberships: Counter[str] = Counter()
    exclusive_hypotheses: Counter[str] = Counter()
    shared_hypothesis_count = 0
    total_hypothesis_count = 0
    total_query_count = 0
    task_rows = []
    for task_id in sorted(reference_tasks):
        present_provider_names = [
            name for name in provider_names if task_id in provider_task_sets[name]
        ]
        reference_queries = provider_views[present_provider_names[0]][2][task_id]
        query_count = len(reference_queries)
        for name in present_provider_names[1:]:
            if len(provider_views[name][2][task_id]) != query_count:
                raise ValueError(f"provider query count differs for {task_id}: {name}")
        frozen_queries = []
        for query_index in range(query_count):
            by_hypothesis_id: dict[str, dict[str, object]] = {}
            reference_query_input_sha256: str | None = None
            for name in present_provider_names:
                adapter, _, tasks = provider_views[name]
                query = _object(
                    tasks[task_id][query_index],
                    field=f"{name}[{task_id}][{query_index}]",
                )
                query_input_sha256 = _sha256(
                    query["query_input_sha256"], field="query_input_sha256"
                )
                if reference_query_input_sha256 is None:
                    reference_query_input_sha256 = query_input_sha256
                elif query_input_sha256 != reference_query_input_sha256:
                    raise ValueError(
                        f"provider query input identity differs for {task_id}[{query_index}]"
                    )
                candidates = _sequence(query["candidates"], field="provider candidates")
                for raw_candidate in candidates:
                    candidate = _object(raw_candidate, field="provider candidate")
                    output, contributor = _candidate_contribution(
                        adapter=adapter,
                        candidate=candidate,
                        provider_name=name,
                        provider_family=provider_families[name],
                    )
                    hypothesis_content = {
                        "cohort_id": cohort_id,
                        "output": output,
                        "query_index": query_index,
                        "query_input_sha256": reference_query_input_sha256,
                        "task_id": task_id,
                    }
                    hypothesis_id = canonical_sha256(hypothesis_content)
                    if hypothesis_id not in by_hypothesis_id:
                        by_hypothesis_id[hypothesis_id] = {
                            "hypothesis_id": hypothesis_id,
                            **hypothesis_content,
                            "contributors": [],
                        }
                    contributors = by_hypothesis_id[hypothesis_id]["contributors"]
                    if not isinstance(contributors, list):
                        raise AssertionError("contributors must remain mutable during freeze")
                    contributors.append(contributor)
                    provider_memberships[name] += 1
            if reference_query_input_sha256 is None:
                raise AssertionError("query identity was not initialized")
            hypotheses = []
            for hypothesis_id in sorted(by_hypothesis_id):
                hypothesis = by_hypothesis_id[hypothesis_id]
                contributors = hypothesis["contributors"]
                if not isinstance(contributors, list):
                    raise AssertionError("contributors must be an array")
                contributors.sort(
                    key=lambda row: (
                        row["provider_name"],
                        row["provider_rank"],
                        row["provider_candidate_id"],
                    )
                )
                contributor_names = [row["provider_name"] for row in contributors]
                if len(contributor_names) != len(set(contributor_names)):
                    raise ValueError("provider emitted duplicate exact candidates")
                hypothesis["provider_count"] = len(contributor_names)
                if len(contributor_names) > 1:
                    shared_hypothesis_count += 1
                else:
                    exclusive_hypotheses[contributor_names[0]] += 1
                hypotheses.append(hypothesis)
            total_hypothesis_count += len(hypotheses)
            total_query_count += 1
            frozen_queries.append(
                {
                    "hypotheses": hypotheses,
                    "hypothesis_count": len(hypotheses),
                    "query_index": query_index,
                    "query_input_sha256": reference_query_input_sha256,
                }
            )
        task_rows.append(
            {"queries": frozen_queries, "query_count": query_count, "task_id": task_id}
        )

    aggregate = {
        "exclusive_hypothesis_count_by_provider": {
            name: exclusive_hypotheses[name] for name in provider_names
        },
        "provider_candidate_membership_count": {
            name: provider_memberships[name] for name in provider_names
        },
        "query_count": total_query_count,
        "shared_hypothesis_count": shared_hypothesis_count,
        "task_count": len(task_rows),
        "unique_hypothesis_count": total_hypothesis_count,
    }
    if has_abstentions:
        aggregate["provider_abstention_task_count"] = {
            name: len(reference_tasks - provider_task_sets[name])
            for name in provider_names
        }
    body: dict[str, object] = {
        "aggregate": aggregate,
        "cohort_id": cohort_id,
        "controller_training_started": False,
        "providers": provider_rows,
        "public_evaluation_read": False,
        "query_gold_read": False,
        "schema": (
            HYPOTHESIS_POPULATION_WITH_ABSTENTIONS_SCHEMA
            if has_abstentions
            else HYPOTHESIS_POPULATION_SCHEMA
        ),
        "tasks": task_rows,
    }
    return {"population_id": canonical_sha256(body), **body}


def _validated_population(
    population: Mapping[str, object],
) -> tuple[tuple[str, ...], Mapping[str, Mapping[str, object]]]:
    body = dict(population)
    declared_id = body.pop("population_id")
    if declared_id != canonical_sha256(body):
        raise ValueError("population_id does not match canonical population content")
    if population["schema"] not in {
        HYPOTHESIS_POPULATION_SCHEMA,
        HYPOTHESIS_POPULATION_WITH_ABSTENTIONS_SCHEMA,
    }:
        raise ValueError("unsupported hypothesis-population schema")
    if population["query_gold_read"] is not False:
        raise ValueError("hypothesis population is not query-gold-free")
    if population["controller_training_started"] is not False:
        raise ValueError("hypothesis population was frozen after controller training")
    if population["public_evaluation_read"] is not False:
        raise ValueError("hypothesis population read public evaluation labels")
    cohort_id = _nonempty_string(population["cohort_id"], field="cohort_id")
    provider_rows = _sequence(population["providers"], field="providers")
    provider_names = tuple(
        _nonempty_string(_object(row, field="provider")["name"], field="provider name")
        for row in provider_rows
    )
    if provider_names != tuple(sorted(set(provider_names))) or len(provider_names) < 2:
        raise ValueError("population providers must be sorted, unique, and heterogeneous")
    tasks: dict[str, Mapping[str, object]] = {}
    for raw_task in _sequence(population["tasks"], field="tasks"):
        task = _object(raw_task, field="task")
        task_id = _nonempty_string(task["task_id"], field="task_id")
        if task_id in tasks:
            raise ValueError("population task_id is duplicated")
        queries = _sequence(task["queries"], field=f"queries[{task_id}]")
        if task["query_count"] != len(queries):
            raise ValueError(f"population query_count differs for {task_id}")
        for query_index, raw_query in enumerate(queries):
            query = _object(raw_query, field="query")
            if query["query_index"] != query_index:
                raise ValueError(f"population query indices are not ordered for {task_id}")
            query_input_sha256 = _sha256(
                query["query_input_sha256"], field="query_input_sha256"
            )
            hypotheses = _sequence(query["hypotheses"], field="hypotheses")
            if query["hypothesis_count"] != len(hypotheses):
                raise ValueError("population hypothesis_count differs")
            observed_ids = []
            for raw_hypothesis in hypotheses:
                hypothesis = _object(raw_hypothesis, field="hypothesis")
                output = grid_to_lists(as_grid(hypothesis["output"]))
                content = {
                    "cohort_id": cohort_id,
                    "output": output,
                    "query_index": query_index,
                    "query_input_sha256": query_input_sha256,
                    "task_id": task_id,
                }
                hypothesis_id = canonical_sha256(content)
                if hypothesis["hypothesis_id"] != hypothesis_id:
                    raise ValueError("hypothesis_id does not match hypothesis content")
                for field, expected in content.items():
                    if hypothesis[field] != expected:
                        raise ValueError(f"hypothesis field {field} is inconsistent")
                contributors = _sequence(hypothesis["contributors"], field="contributors")
                contributor_names = [
                    _nonempty_string(
                        _object(row, field="contributor")["provider_name"],
                        field="contributor provider",
                    )
                    for row in contributors
                ]
                if (
                    contributor_names != sorted(set(contributor_names))
                    or any(name not in provider_names for name in contributor_names)
                    or hypothesis["provider_count"] != len(contributor_names)
                ):
                    raise ValueError("hypothesis contributor set is inconsistent")
                observed_ids.append(hypothesis_id)
            if observed_ids != sorted(set(observed_ids)):
                raise ValueError("population hypothesis IDs must be sorted and unique")
        tasks[task_id] = task
    if list(tasks) != sorted(tasks):
        raise ValueError("population tasks must be sorted")
    return provider_names, tasks


def score_hypothesis_population(
    *,
    population: Mapping[str, object],
    solutions: Mapping[str, object],
    solution_source_sha256: str,
) -> dict[str, object]:
    """Measure strict provider complement only after population freeze."""

    solution_source_sha256 = _sha256(
        solution_source_sha256, field="solution_source_sha256"
    )
    provider_names, tasks = _validated_population(population)
    if set(tasks) != set(solutions):
        raise ValueError("hypothesis population and solution task sets differ")
    strict_counts = {name: 0 for name in provider_names}
    query_counts = {name: 0 for name in provider_names}
    exclusive_counts = {name: 0 for name in provider_names}
    pairwise = {
        f"{first}|{second}": {
            "both": 0,
            "first_only": 0,
            "neither": 0,
            "second_only": 0,
        }
        for first_index, first in enumerate(provider_names)
        for second in provider_names[first_index + 1 :]
    }
    union_strict_count = 0
    union_query_count = 0
    cross_provider_composed_count = 0
    total_query_count = 0
    coverage_patterns: Counter[str] = Counter()
    task_rows = []
    for task_id in sorted(tasks):
        task = tasks[task_id]
        queries = _sequence(task["queries"], field=f"queries[{task_id}]")
        task_solutions = _sequence(solutions[task_id], field=f"solutions[{task_id}]")
        if len(queries) != len(task_solutions):
            raise ValueError(f"solution query count differs for {task_id}")
        provider_query_hits = {name: [] for name in provider_names}
        union_query_hits = []
        query_rows = []
        for query_index, (raw_query, raw_solution) in enumerate(
            zip(queries, task_solutions, strict=True)
        ):
            query = _object(raw_query, field="query")
            gold = grid_to_lists(as_grid(raw_solution))
            gold_hypothesis_id = canonical_sha256(
                {
                    "cohort_id": population["cohort_id"],
                    "output": gold,
                    "query_index": query_index,
                    "query_input_sha256": query["query_input_sha256"],
                    "task_id": task_id,
                }
            )
            matching_providers: set[str] = set()
            for raw_hypothesis in _sequence(query["hypotheses"], field="hypotheses"):
                hypothesis = _object(raw_hypothesis, field="hypothesis")
                if hypothesis["hypothesis_id"] != gold_hypothesis_id:
                    continue
                for raw_contributor in _sequence(
                    hypothesis["contributors"], field="contributors"
                ):
                    contributor = _object(raw_contributor, field="contributor")
                    matching_providers.add(str(contributor["provider_name"]))
            hits = {name: name in matching_providers for name in provider_names}
            for name in provider_names:
                provider_query_hits[name].append(hits[name])
                query_counts[name] += int(hits[name])
            union_hit = bool(matching_providers)
            union_query_hits.append(union_hit)
            union_query_count += int(union_hit)
            total_query_count += 1
            query_rows.append(
                {
                    "matching_providers": sorted(matching_providers),
                    "provider_hits": hits,
                    "query_index": query_index,
                    "solution_sha256": canonical_sha256(gold),
                    "union_hit": union_hit,
                }
            )
        provider_task_hits = {
            name: all(provider_query_hits[name]) for name in provider_names
        }
        union_task_hit = all(union_query_hits)
        for name in provider_names:
            strict_counts[name] += int(provider_task_hits[name])
        union_strict_count += int(union_task_hit)
        strict_provider_set = {
            name for name in provider_names if provider_task_hits[name]
        }
        if len(strict_provider_set) == 1:
            exclusive_counts[next(iter(strict_provider_set))] += 1
        if union_task_hit and not strict_provider_set:
            cross_provider_composed_count += 1
        pattern = "+".join(sorted(strict_provider_set)) if strict_provider_set else "none"
        coverage_patterns[pattern] += 1
        for first_index, first in enumerate(provider_names):
            for second in provider_names[first_index + 1 :]:
                first_hit = provider_task_hits[first]
                second_hit = provider_task_hits[second]
                if first_hit and second_hit:
                    category = "both"
                elif first_hit:
                    category = "first_only"
                elif second_hit:
                    category = "second_only"
                else:
                    category = "neither"
                pairwise[f"{first}|{second}"][category] += 1
        task_rows.append(
            {
                "provider_hits": provider_task_hits,
                "queries": query_rows,
                "query_count": len(queries),
                "task_id": task_id,
                "union_hit": union_task_hit,
            }
        )

    task_count = len(task_rows)
    best_provider_count = max(strict_counts.values())
    content: dict[str, object] = {
        "schema": HYPOTHESIS_POPULATION_RESULT_SCHEMA,
        "population_id": population["population_id"],
        "solution_source_sha256": solution_source_sha256,
        "query_gold_read": True,
        "metrics": {
            "best_single_provider_strict_task_coverage": best_provider_count,
            "cross_provider_composed_strict_task_coverage": cross_provider_composed_count,
            "exclusive_strict_task_coverage_by_provider": exclusive_counts,
            "provider_query_coverage": {
                name: {"correct": query_counts[name], "total": total_query_count}
                for name in provider_names
            },
            "provider_strict_task_coverage": {
                name: {"correct": strict_counts[name], "total": task_count}
                for name in provider_names
            },
            "union_increment_over_best_single": union_strict_count
            - best_provider_count,
            "union_query_coverage": {
                "correct": union_query_count,
                "total": total_query_count,
            },
            "union_strict_task_coverage": {
                "correct": union_strict_count,
                "total": task_count,
            },
        },
        "coverage_pattern_counts": dict(sorted(coverage_patterns.items())),
        "pairwise_contingencies": pairwise,
        "tasks": task_rows,
    }
    return {"result_id": canonical_sha256(content), **content}


def summarize_hypothesis_population_result(
    result: Mapping[str, object],
) -> dict[str, object]:
    """Publish aggregate evidence while committing to hidden task outcomes."""

    if result["schema"] != HYPOTHESIS_POPULATION_RESULT_SCHEMA:
        raise ValueError("unsupported hypothesis-population result schema")
    body = dict(result)
    declared_id = body.pop("result_id")
    if declared_id != canonical_sha256(body):
        raise ValueError("result_id does not match canonical result content")
    content: dict[str, object] = {
        "schema": HYPOTHESIS_POPULATION_SUMMARY_SCHEMA,
        "population_id": result["population_id"],
        "full_result_id_commitment": declared_id,
        "solution_source_sha256": result["solution_source_sha256"],
        "query_gold_read": True,
        "task_level_outcomes_exposed": False,
        "metrics": result["metrics"],
        "coverage_pattern_counts": result["coverage_pattern_counts"],
        "pairwise_contingencies": result["pairwise_contingencies"],
    }
    return {"summary_id": canonical_sha256(content), **content}
