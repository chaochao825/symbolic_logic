"""Freeze and independently score NVARC/TRM ARC candidate submissions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from afts_arc.experiment_safety import canonical_sha256
from afts_arc.grid import GridValidationError, as_grid, grid_to_lists


ANCHOR_PROVIDER_VERSION = "afts-nvarc-trm-anchor/v1"
CANDIDATE_FREEZE_SCHEMA = "afts.nvarc-trm-candidate-freeze/v1"
CANDIDATE_FREEZE_WITH_REJECTIONS_SCHEMA = (
    "afts.nvarc-trm-candidate-freeze-with-rejections/v2"
)
ANCHOR_RESULT_SCHEMA = "afts.nvarc-trm-anchor-result/v1"
ANCHOR_GATE_SUMMARY_SCHEMA = "afts.nvarc-trm-anchor-gate-summary/v1"
PASS_KS = (1, 2, 5, 10)


def _object(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise TypeError(f"{field} keys must be strings")
    return value


def _sequence(value: object, *, field: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise TypeError(f"{field} must be a JSON array")
    return value


def _grid(value: object, *, field: str) -> list[list[int]]:
    del field
    return grid_to_lists(as_grid(value))


def _validate_task_sets(
    first: Mapping[str, object],
    second: Mapping[str, object],
    *,
    first_name: str,
    second_name: str,
) -> None:
    if set(first) != set(second):
        missing = sorted(set(first) - set(second))
        unknown = sorted(set(second) - set(first))
        raise ValueError(
            f"{second_name} task set differs from {first_name}: "
            f"missing={missing}, unknown={unknown}"
        )


def _candidate_content(
    *,
    cohort_id: str,
    task_id: str,
    query_index: int,
    query_input_sha256: str,
    output: list[list[int]],
) -> dict[str, object]:
    return {
        "cohort_id": cohort_id,
        "output": output,
        "provider_version": ANCHOR_PROVIDER_VERSION,
        "query_index": query_index,
        "query_input_sha256": query_input_sha256,
        "task_id": task_id,
    }


def _validate_freeze_id(freeze: Mapping[str, object]) -> None:
    if "schema" not in freeze:
        raise ValueError("candidate freeze has no schema")
    schema = freeze["schema"]
    expected = {
        "anchor_run_id",
        "cohort_id",
        "controller_training_started",
        "freeze_id",
        "provider_version",
        "public_evaluation_read",
        "query_gold_read",
        "schema",
        "source_files",
        "tasks",
    }
    if schema == CANDIDATE_FREEZE_WITH_REJECTIONS_SCHEMA:
        expected.add("rejected_attempt_count")
    elif schema != CANDIDATE_FREEZE_SCHEMA:
        raise ValueError("unsupported candidate freeze schema")
    if set(freeze) != expected:
        raise ValueError("candidate freeze has missing or unknown top-level fields")
    body = dict(freeze)
    declared = body.pop("freeze_id")
    if declared != canonical_sha256(body):
        raise ValueError("freeze_id does not match canonical freeze content")


def _freeze_submission(
    *,
    challenges: Mapping[str, object],
    submission: Mapping[str, object],
    cohort_id: str,
    anchor_run_id: str,
    source_files: Mapping[str, str],
    reject_invalid_attempts: bool,
) -> dict[str, object]:
    """Build a query-gold-free, content-addressed top-10 candidate artifact."""

    if not cohort_id:
        raise ValueError("cohort_id must not be empty")
    if not anchor_run_id:
        raise ValueError("anchor_run_id must not be empty")
    if not source_files or not all(source_files[key] for key in source_files):
        raise ValueError("source_files must contain non-empty digests")
    _validate_task_sets(
        challenges,
        submission,
        first_name="challenges",
        second_name="submission",
    )
    attempt_fields = {f"attempt_{rank}" for rank in range(1, 11)}
    frozen_tasks: list[dict[str, object]] = []
    total_rejected_attempts = 0
    for task_id in sorted(challenges):
        challenge = _object(challenges[task_id], field=f"challenges[{task_id}]")
        if set(challenge) != {"train", "test"}:
            raise ValueError(f"challenge {task_id} has missing or unknown fields")
        queries = _sequence(challenge["test"], field=f"challenges[{task_id}].test")
        submitted_queries = _sequence(
            submission[task_id], field=f"submission[{task_id}]"
        )
        if len(submitted_queries) != len(queries):
            raise ValueError(f"submission query count differs for task {task_id}")
        frozen_queries: list[dict[str, object]] = []
        for query_index, (raw_query, raw_attempts) in enumerate(
            zip(queries, submitted_queries, strict=True)
        ):
            query = _object(
                raw_query, field=f"challenges[{task_id}].test[{query_index}]"
            )
            if set(query) != {"input"}:
                raise ValueError(
                    f"challenge query {task_id}[{query_index}] must contain input only"
                )
            query_input = _grid(
                query["input"], field=f"challenges[{task_id}].test[{query_index}].input"
            )
            query_input_sha256 = canonical_sha256(query_input)
            attempts = _object(
                raw_attempts, field=f"submission[{task_id}][{query_index}]"
            )
            if set(attempts) != attempt_fields:
                raise ValueError(
                    f"submission attempts for {task_id}[{query_index}] "
                    "must be exactly attempt_1..attempt_10"
                )
            by_candidate_id: dict[str, dict[str, object]] = {}
            ordered_candidate_ids: list[str] = []
            rejected_attempts: list[dict[str, object]] = []
            for rank in range(1, 11):
                raw_output = attempts[f"attempt_{rank}"]
                if reject_invalid_attempts:
                    try:
                        output = _grid(
                            raw_output,
                            field=(
                                f"submission[{task_id}][{query_index}]"
                                f".attempt_{rank}"
                            ),
                        )
                    except GridValidationError:
                        rejected_attempts.append(
                            {
                                "rank": rank,
                                "raw_output_sha256": canonical_sha256(raw_output),
                                "reason": "invalid_arc_grid",
                            }
                        )
                        total_rejected_attempts += 1
                        continue
                else:
                    output = _grid(
                        raw_output,
                        field=(
                            f"submission[{task_id}][{query_index}].attempt_{rank}"
                        ),
                    )
                content = _candidate_content(
                    cohort_id=cohort_id,
                    task_id=task_id,
                    query_index=query_index,
                    query_input_sha256=query_input_sha256,
                    output=output,
                )
                candidate_id = canonical_sha256(content)
                if candidate_id not in by_candidate_id:
                    ordered_candidate_ids.append(candidate_id)
                    by_candidate_id[candidate_id] = {
                        "candidate_id": candidate_id,
                        **content,
                        "ranks": [],
                    }
                ranks = by_candidate_id[candidate_id]["ranks"]
                if not isinstance(ranks, list):
                    raise AssertionError(
                        "candidate ranks are not mutable during freeze"
                    )
                ranks.append(rank)
            query_content: dict[str, object] = {
                "attempt_count": 10,
                "candidates": [
                    by_candidate_id[candidate_id]
                    for candidate_id in ordered_candidate_ids
                ],
                "query_index": query_index,
                "query_input_sha256": query_input_sha256,
                "unique_candidate_count": len(ordered_candidate_ids),
            }
            if reject_invalid_attempts:
                query_content["rejected_attempts"] = rejected_attempts
            frozen_queries.append(query_content)
        frozen_tasks.append(
            {
                "queries": frozen_queries,
                "query_count": len(frozen_queries),
                "task_id": task_id,
            }
        )
    body: dict[str, object] = {
        "anchor_run_id": anchor_run_id,
        "cohort_id": cohort_id,
        "controller_training_started": False,
        "provider_version": ANCHOR_PROVIDER_VERSION,
        "public_evaluation_read": False,
        "query_gold_read": False,
        "schema": (
            CANDIDATE_FREEZE_WITH_REJECTIONS_SCHEMA
            if reject_invalid_attempts
            else CANDIDATE_FREEZE_SCHEMA
        ),
        "source_files": dict(sorted(source_files.items())),
        "tasks": frozen_tasks,
    }
    if reject_invalid_attempts:
        body["rejected_attempt_count"] = total_rejected_attempts
    return {"freeze_id": canonical_sha256(body), **body}


def freeze_submission(
    *,
    challenges: Mapping[str, object],
    submission: Mapping[str, object],
    cohort_id: str,
    anchor_run_id: str,
    source_files: Mapping[str, str],
) -> dict[str, object]:
    """Build the legacy v1 freeze and reject every malformed attempt."""

    return _freeze_submission(
        challenges=challenges,
        submission=submission,
        cohort_id=cohort_id,
        anchor_run_id=anchor_run_id,
        source_files=source_files,
        reject_invalid_attempts=False,
    )


def freeze_submission_with_rejections(
    *,
    challenges: Mapping[str, object],
    submission: Mapping[str, object],
    cohort_id: str,
    anchor_run_id: str,
    source_files: Mapping[str, str],
) -> dict[str, object]:
    """Freeze valid grids and explicitly preserve malformed ranks as misses."""

    return _freeze_submission(
        challenges=challenges,
        submission=submission,
        cohort_id=cohort_id,
        anchor_run_id=anchor_run_id,
        source_files=source_files,
        reject_invalid_attempts=True,
    )


def _validated_candidates(
    freeze: Mapping[str, object],
) -> tuple[str, dict[str, Sequence[object]]]:
    _validate_freeze_id(freeze)
    schema = freeze["schema"]
    with_rejections = schema == CANDIDATE_FREEZE_WITH_REJECTIONS_SCHEMA
    if freeze["provider_version"] != ANCHOR_PROVIDER_VERSION:
        raise ValueError("unsupported anchor provider version")
    if freeze["query_gold_read"] is not False:
        raise ValueError("candidate freeze is not query-gold-free")
    cohort_id = freeze["cohort_id"]
    if not isinstance(cohort_id, str) or not cohort_id:
        raise ValueError("invalid freeze cohort_id")
    tasks: dict[str, Sequence[object]] = {}
    observed_rejected_attempts = 0
    for raw_task in _sequence(freeze["tasks"], field="freeze.tasks"):
        task = _object(raw_task, field="freeze task")
        if set(task) != {"queries", "query_count", "task_id"}:
            raise ValueError("freeze task has missing or unknown fields")
        task_id = task["task_id"]
        if not isinstance(task_id, str) or not task_id or task_id in tasks:
            raise ValueError("freeze task_id is invalid or duplicated")
        queries = _sequence(task["queries"], field=f"freeze task {task_id}.queries")
        if task["query_count"] != len(queries):
            raise ValueError(f"freeze query_count differs for {task_id}")
        for expected_query_index, raw_query in enumerate(queries):
            query = _object(raw_query, field=f"freeze task {task_id} query")
            expected_query_fields = {
                "attempt_count",
                "candidates",
                "query_index",
                "query_input_sha256",
                "unique_candidate_count",
            }
            if with_rejections:
                expected_query_fields.add("rejected_attempts")
            if set(query) != expected_query_fields:
                raise ValueError("freeze query has missing or unknown fields")
            if query["query_index"] != expected_query_index:
                raise ValueError(f"freeze query indices are not ordered for {task_id}")
            if query["attempt_count"] != 10:
                raise ValueError("freeze query attempt_count must be 10")
            query_input_sha256 = query["query_input_sha256"]
            if not isinstance(query_input_sha256, str):
                raise TypeError("query_input_sha256 must be a string")
            candidates = _sequence(
                query["candidates"], field=f"freeze task {task_id} candidates"
            )
            if query["unique_candidate_count"] != len(candidates):
                raise ValueError("unique_candidate_count differs from candidates")
            seen_ids: set[str] = set()
            seen_ranks: set[int] = set()
            for raw_candidate in candidates:
                candidate = _object(raw_candidate, field="freeze candidate")
                if set(candidate) != {
                    "candidate_id",
                    "cohort_id",
                    "output",
                    "provider_version",
                    "query_index",
                    "query_input_sha256",
                    "ranks",
                    "task_id",
                }:
                    raise ValueError("freeze candidate has missing or unknown fields")
                output = _grid(candidate["output"], field="freeze candidate output")
                ranks = _sequence(candidate["ranks"], field="freeze candidate ranks")
                if not ranks or any(
                    isinstance(rank, bool) or not isinstance(rank, int)
                    for rank in ranks
                ):
                    raise TypeError("candidate ranks must be non-empty integers")
                normalized_ranks = [int(rank) for rank in ranks]
                if normalized_ranks != sorted(set(normalized_ranks)):
                    raise ValueError("candidate ranks must be sorted and unique")
                if any(not 1 <= rank <= 10 for rank in normalized_ranks):
                    raise ValueError("candidate rank is outside 1..10")
                if seen_ranks.intersection(normalized_ranks):
                    raise ValueError("candidate ranks overlap")
                seen_ranks.update(normalized_ranks)
                content = _candidate_content(
                    cohort_id=cohort_id,
                    task_id=task_id,
                    query_index=expected_query_index,
                    query_input_sha256=query_input_sha256,
                    output=output,
                )
                if candidate["candidate_id"] != canonical_sha256(content):
                    raise ValueError("candidate_id does not match candidate content")
                if candidate["candidate_id"] in seen_ids:
                    raise ValueError("duplicate candidate_id in query")
                seen_ids.add(str(candidate["candidate_id"]))
                for field, expected in content.items():
                    if candidate[field] != expected:
                        raise ValueError(f"candidate field {field} is inconsistent")
            rejected_ranks: set[int] = set()
            if with_rejections:
                rejected = _sequence(
                    query["rejected_attempts"], field="freeze rejected attempts"
                )
                observed_rejected_attempts += len(rejected)
                for raw_rejection in rejected:
                    rejection = _object(raw_rejection, field="freeze rejection")
                    if set(rejection) != {"rank", "raw_output_sha256", "reason"}:
                        raise ValueError("freeze rejection fields differ")
                    rank = rejection["rank"]
                    if isinstance(rank, bool) or not isinstance(rank, int):
                        raise TypeError("rejected attempt rank must be an integer")
                    if not 1 <= rank <= 10 or rank in rejected_ranks:
                        raise ValueError("rejected attempt rank is invalid or duplicated")
                    digest = rejection["raw_output_sha256"]
                    if not isinstance(digest, str) or len(digest) != 64:
                        raise ValueError("rejected raw output identity is invalid")
                    if rejection["reason"] != "invalid_arc_grid":
                        raise ValueError("unknown rejected attempt reason")
                    rejected_ranks.add(rank)
            if seen_ranks & rejected_ranks:
                raise ValueError("valid and rejected candidate ranks overlap")
            if seen_ranks | rejected_ranks != set(range(1, 11)):
                raise ValueError("candidate ranks do not cover attempt_1..attempt_10")
        tasks[task_id] = queries
    if with_rejections:
        rejected_attempt_count = freeze["rejected_attempt_count"]
        if isinstance(rejected_attempt_count, bool) or not isinstance(
            rejected_attempt_count, int
        ):
            raise TypeError("top-level rejected attempt count must be an integer")
        if rejected_attempt_count != observed_rejected_attempts:
            raise ValueError("top-level rejected attempt count differs")
    return cohort_id, tasks


def score_candidate_freeze(
    *,
    freeze: Mapping[str, object],
    solutions: Mapping[str, object],
    solution_source_sha256: str,
) -> dict[str, object]:
    """Score a frozen top-10 pool without modifying or reranking candidates."""

    if not solution_source_sha256:
        raise ValueError("solution_source_sha256 must not be empty")
    cohort_id, tasks = _validated_candidates(freeze)
    _validate_task_sets(
        tasks,
        solutions,
        first_name="candidate freeze",
        second_name="solutions",
    )
    strict_counts = {k: 0 for k in PASS_KS}
    query_counts = {k: 0 for k in PASS_KS}
    upstream_task_sums = {k: 0.0 for k in PASS_KS}
    task_rows: list[dict[str, object]] = []
    total_queries = 0
    for task_id in sorted(tasks):
        queries = tasks[task_id]
        raw_solutions = _sequence(solutions[task_id], field=f"solutions[{task_id}]")
        if len(raw_solutions) != len(queries):
            raise ValueError(f"solution query count differs for task {task_id}")
        task_hits = {k: 0 for k in PASS_KS}
        query_rows: list[dict[str, object]] = []
        for query_index, (raw_query, raw_solution) in enumerate(
            zip(queries, raw_solutions, strict=True)
        ):
            query = _object(raw_query, field=f"freeze task {task_id} query")
            solution = _grid(raw_solution, field=f"solutions[{task_id}][{query_index}]")
            solution_sha256 = canonical_sha256(solution)
            candidates = _sequence(
                query["candidates"], field=f"freeze task {task_id} candidates"
            )
            hits: dict[str, bool] = {}
            for k in PASS_KS:
                hit = False
                for raw_candidate in candidates:
                    candidate = _object(raw_candidate, field="freeze candidate")
                    ranks = _sequence(candidate["ranks"], field="candidate ranks")
                    if min(int(rank) for rank in ranks) <= k:
                        hit = hit or candidate["output"] == solution
                hits[str(k)] = hit
                task_hits[k] += int(hit)
                query_counts[k] += int(hit)
            query_rows.append(
                {
                    "hits": hits,
                    "query_index": query_index,
                    "solution_sha256": solution_sha256,
                }
            )
        total_queries += len(queries)
        strict_hits: dict[str, bool] = {}
        pair_rates: dict[str, float] = {}
        for k in PASS_KS:
            strict_hit = task_hits[k] == len(queries)
            strict_hits[str(k)] = strict_hit
            strict_counts[k] += int(strict_hit)
            pair_rate = task_hits[k] / len(queries)
            pair_rates[str(k)] = pair_rate
            upstream_task_sums[k] += pair_rate
        task_rows.append(
            {
                "pair_hit_rates": pair_rates,
                "queries": query_rows,
                "query_count": len(queries),
                "strict_task_hits": strict_hits,
                "task_id": task_id,
            }
        )
    task_count = len(tasks)
    metrics: dict[str, object] = {}
    for k in PASS_KS:
        metrics[str(k)] = {
            "query_pass": {
                "correct": query_counts[k],
                "rate": query_counts[k] / total_queries,
                "total": total_queries,
            },
            "strict_task_pass": {
                "correct": strict_counts[k],
                "rate": strict_counts[k] / task_count,
                "total": task_count,
            },
            "upstream_task_mean_pair_pass": {
                "rate": upstream_task_sums[k] / task_count,
                "task_count": task_count,
            },
        }
    body: dict[str, Any] = {
        "candidate_freeze_id": freeze["freeze_id"],
        "cohort_id": cohort_id,
        "metrics_by_k": metrics,
        "query_gold_read": True,
        "schema": ANCHOR_RESULT_SCHEMA,
        "solution_source_sha256": solution_source_sha256,
        "tasks": task_rows,
    }
    return {"result_id": canonical_sha256(body), **body}


def score_candidate_freeze_gate(
    *,
    freeze: Mapping[str, object],
    solutions: Mapping[str, object],
    solution_source_sha256: str,
) -> dict[str, object]:
    """Expose aggregate anchor metrics while committing to hidden task rows."""

    full_result = score_candidate_freeze(
        freeze=freeze,
        solutions=solutions,
        solution_source_sha256=solution_source_sha256,
    )
    body: dict[str, object] = {
        "candidate_freeze_id": full_result["candidate_freeze_id"],
        "cohort_id": full_result["cohort_id"],
        "full_result_id_commitment": full_result["result_id"],
        "metrics_by_k": full_result["metrics_by_k"],
        "query_gold_read": True,
        "schema": ANCHOR_GATE_SUMMARY_SCHEMA,
        "solution_source_sha256": solution_source_sha256,
        "task_level_outcomes_exposed": False,
    }
    return {"gate_summary_id": canonical_sha256(body), **body}
