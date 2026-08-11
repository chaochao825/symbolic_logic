"""Freeze and independently score query-blind VARC posterior candidates."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

from afts_arc.experiment_safety import canonical_json, canonical_sha256
from afts_arc.grid import Grid, GridValidationError, as_grid, grid_to_lists
from afts_arc.varc_blind import VARC_BLIND_SCHEMA


VARC_PROVIDER_VERSION = "afts-varc-vit-ttt/v1"
VARC_FREEZE_SCHEMA = "afts.varc-posterior-candidate-freeze/v1"
VARC_RESULT_SCHEMA = "afts.varc-posterior-result/v1"


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


def _task_records(manifest: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    if manifest["schema"] != VARC_BLIND_SCHEMA:
        raise ValueError("unsupported VARC blind-manifest schema")
    records: dict[str, Mapping[str, object]] = {}
    for raw_record in _sequence(manifest["tasks"], field="manifest.tasks"):
        record = _object(raw_record, field="manifest task")
        task_id = record["task_id"]
        if not isinstance(task_id, str) or not task_id or task_id in records:
            raise ValueError("manifest task_id is invalid or duplicated")
        records[task_id] = record
    if not records:
        raise ValueError("VARC blind manifest contains no tasks")
    return records


def _rank_samples(
    samples: Sequence[object],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    valid: list[Grid] = []
    rejection_reasons: Counter[str] = Counter()
    for sample in samples:
        try:
            valid.append(as_grid(sample))
        except GridValidationError as error:
            rejection_reasons[str(error)] += 1
    if not valid:
        raise ValueError("VARC query contains no valid candidates")
    counts = Counter(valid)
    first_indices: dict[Grid, int] = {}
    for sample_index, grid in enumerate(valid):
        if grid not in first_indices:
            first_indices[grid] = sample_index
    ranked = sorted(
        counts,
        key=lambda grid: (
            -counts[grid],
            first_indices[grid],
            canonical_json(grid_to_lists(grid)),
        ),
    )
    candidates = [
        {
            "first_valid_sample_index": first_indices[grid],
            "output": grid_to_lists(grid),
            "rank": rank,
            "sample_count": counts[grid],
        }
        for rank, grid in enumerate(ranked, start=1)
    ]
    validation = {
        "raw_sample_count": len(samples),
        "rejected_sample_count": len(samples) - len(valid),
        "rejection_reasons": [
            {"count": count, "reason": reason}
            for reason, count in sorted(rejection_reasons.items())
        ],
        "unique_candidate_count": len(candidates),
        "valid_sample_count": len(valid),
    }
    return candidates, validation


def _candidate_content(
    *,
    blind_cohort_id: str,
    task_id: str,
    query_index: int,
    query_input_sha256: str,
    output: list[list[int]],
) -> dict[str, object]:
    return {
        "blind_cohort_id": blind_cohort_id,
        "output": output,
        "provider_version": VARC_PROVIDER_VERSION,
        "query_index": query_index,
        "query_input_sha256": query_input_sha256,
        "task_id": task_id,
    }


def freeze_varc_predictions(
    *,
    challenges: Mapping[str, object],
    blind_manifest: Mapping[str, object],
    predictions: Mapping[str, object],
    provider_contract: Mapping[str, object],
    source_files: Mapping[str, str],
) -> dict[str, object]:
    """Freeze all valid unique VARC grids before query-gold access."""

    records = _task_records(blind_manifest)
    if set(records) != set(challenges) or set(records) != set(predictions):
        raise ValueError("challenge, manifest, and prediction task sets differ")
    if source_files["challenges"] != blind_manifest["source_challenges_sha256"]:
        raise ValueError("challenge SHA-256 differs from blind manifest")
    blind_cohort_id = blind_manifest["blind_cohort_id"]
    if not isinstance(blind_cohort_id, str) or not blind_cohort_id:
        raise ValueError("blind_cohort_id is invalid")
    frozen_tasks: list[dict[str, object]] = []
    total_raw_samples = 0
    total_valid_samples = 0
    total_rejected_samples = 0
    total_unique_candidates = 0
    for task_id in sorted(records):
        challenge = _object(challenges[task_id], field=f"challenges[{task_id}]")
        if set(challenge) != {"train", "test"}:
            raise ValueError(f"challenge {task_id} has missing or unknown fields")
        queries = _sequence(challenge["test"], field=f"challenges[{task_id}].test")
        if records[task_id]["query_count"] != len(queries):
            raise ValueError(f"manifest query_count differs for {task_id}")
        task_predictions = _object(
            predictions[task_id], field=f"predictions[{task_id}]"
        )
        expected_query_keys = {str(index) for index in range(len(queries))}
        if set(task_predictions) != expected_query_keys:
            raise ValueError(f"prediction query indices differ for {task_id}")
        frozen_queries: list[dict[str, object]] = []
        for query_index, raw_query in enumerate(queries):
            query = _object(
                raw_query, field=f"challenges[{task_id}].test[{query_index}]"
            )
            if set(query) != {"input"}:
                raise ValueError(
                    f"challenge query {task_id}[{query_index}] must contain input only"
                )
            query_input = grid_to_lists(as_grid(query["input"]))
            query_input_sha256 = canonical_sha256(query_input)
            samples = _sequence(
                task_predictions[str(query_index)],
                field=f"predictions[{task_id}][{query_index}]",
            )
            candidates, validation = _rank_samples(samples)
            frozen_candidates: list[dict[str, object]] = []
            for candidate in candidates:
                output = candidate["output"]
                if not isinstance(output, list):
                    raise AssertionError("ranked candidate output is not a grid")
                content = _candidate_content(
                    blind_cohort_id=blind_cohort_id,
                    task_id=task_id,
                    query_index=query_index,
                    query_input_sha256=query_input_sha256,
                    output=output,
                )
                frozen_candidates.append(
                    {"candidate_id": canonical_sha256(content), **content, **candidate}
                )
            total_raw_samples += int(validation["raw_sample_count"])
            total_valid_samples += int(validation["valid_sample_count"])
            total_rejected_samples += int(validation["rejected_sample_count"])
            total_unique_candidates += int(validation["unique_candidate_count"])
            frozen_queries.append(
                {
                    "candidates": frozen_candidates,
                    "query_index": query_index,
                    "query_input_sha256": query_input_sha256,
                    "validation": validation,
                }
            )
        frozen_tasks.append(
            {
                "queries": frozen_queries,
                "query_count": len(frozen_queries),
                "task_id": task_id,
            }
        )
    aggregate = {
        "raw_sample_count": total_raw_samples,
        "rejected_sample_count": total_rejected_samples,
        "unique_candidate_count": total_unique_candidates,
        "valid_sample_count": total_valid_samples,
    }
    body: dict[str, object] = {
        "aggregate": aggregate,
        "blind_cohort_id": blind_cohort_id,
        "controller_training_started": False,
        "provider_contract": dict(provider_contract),
        "provider_version": VARC_PROVIDER_VERSION,
        "public_evaluation_read": False,
        "query_gold_read": False,
        "schema": VARC_FREEZE_SCHEMA,
        "source_files": dict(sorted(source_files.items())),
        "tasks": frozen_tasks,
    }
    return {"freeze_id": canonical_sha256(body), **body}


def _validated_freeze(
    freeze: Mapping[str, object],
) -> tuple[str, dict[str, Sequence[object]]]:
    body = dict(freeze)
    declared = body.pop("freeze_id")
    if declared != canonical_sha256(body):
        raise ValueError("freeze_id does not match canonical freeze content")
    if freeze["schema"] != VARC_FREEZE_SCHEMA:
        raise ValueError("unsupported VARC candidate-freeze schema")
    if freeze["provider_version"] != VARC_PROVIDER_VERSION:
        raise ValueError("unsupported VARC provider version")
    if freeze["query_gold_read"] is not False:
        raise ValueError("VARC candidate freeze is not query-gold-free")
    blind_cohort_id = freeze["blind_cohort_id"]
    if not isinstance(blind_cohort_id, str) or not blind_cohort_id:
        raise ValueError("candidate freeze blind_cohort_id is invalid")
    tasks: dict[str, Sequence[object]] = {}
    for raw_task in _sequence(freeze["tasks"], field="freeze.tasks"):
        task = _object(raw_task, field="freeze task")
        task_id = task["task_id"]
        if not isinstance(task_id, str) or not task_id or task_id in tasks:
            raise ValueError("freeze task_id is invalid or duplicated")
        queries = _sequence(task["queries"], field=f"freeze task {task_id}.queries")
        if task["query_count"] != len(queries):
            raise ValueError(f"freeze query_count differs for {task_id}")
        for expected_query_index, raw_query in enumerate(queries):
            query = _object(raw_query, field="freeze query")
            if query["query_index"] != expected_query_index:
                raise ValueError(f"freeze query indices are not ordered for {task_id}")
            query_input_sha256 = query["query_input_sha256"]
            if not isinstance(query_input_sha256, str):
                raise TypeError("query_input_sha256 must be a string")
            candidates = _sequence(query["candidates"], field="freeze candidates")
            seen_ids: set[str] = set()
            for expected_rank, raw_candidate in enumerate(candidates, start=1):
                candidate = _object(raw_candidate, field="freeze candidate")
                if candidate["rank"] != expected_rank:
                    raise ValueError("VARC candidate ranks are not contiguous")
                output = grid_to_lists(as_grid(candidate["output"]))
                content = _candidate_content(
                    blind_cohort_id=blind_cohort_id,
                    task_id=task_id,
                    query_index=expected_query_index,
                    query_input_sha256=query_input_sha256,
                    output=output,
                )
                candidate_id = candidate["candidate_id"]
                if candidate_id != canonical_sha256(content):
                    raise ValueError("VARC candidate_id does not match content")
                if candidate_id in seen_ids:
                    raise ValueError("duplicate VARC candidate_id in query")
                seen_ids.add(str(candidate_id))
                for field, expected in content.items():
                    if candidate[field] != expected:
                        raise ValueError(
                            f"VARC candidate field {field} is inconsistent"
                        )
        tasks[task_id] = queries
    return blind_cohort_id, tasks


def score_varc_freeze(
    *,
    freeze: Mapping[str, object],
    solutions: Mapping[str, object],
    solution_source_sha256: str,
) -> dict[str, object]:
    """Score rank-1, rank-2, and raw-oracle endpoints after candidate freeze."""

    if not solution_source_sha256:
        raise ValueError("solution_source_sha256 must not be empty")
    blind_cohort_id, tasks = _validated_freeze(freeze)
    if set(tasks) != set(solutions):
        raise ValueError("candidate-freeze and solution task sets differ")
    endpoint_names = ("1", "2", "raw")
    strict_counts = {name: 0 for name in endpoint_names}
    query_counts = {name: 0 for name in endpoint_names}
    task_pair_sums = {name: 0.0 for name in endpoint_names}
    total_queries = 0
    task_rows: list[dict[str, object]] = []
    for task_id in sorted(tasks):
        queries = tasks[task_id]
        task_solutions = _sequence(solutions[task_id], field=f"solutions[{task_id}]")
        if len(queries) != len(task_solutions):
            raise ValueError(f"solution query count differs for {task_id}")
        task_hits = {name: 0 for name in endpoint_names}
        query_rows: list[dict[str, object]] = []
        for query_index, (raw_query, raw_solution) in enumerate(
            zip(queries, task_solutions, strict=True)
        ):
            query = _object(raw_query, field="freeze query")
            solution = grid_to_lists(as_grid(raw_solution))
            candidates = _sequence(query["candidates"], field="freeze candidates")
            hits = {
                "1": any(
                    candidate["rank"] <= 1 and candidate["output"] == solution
                    for candidate in candidates
                ),
                "2": any(
                    candidate["rank"] <= 2 and candidate["output"] == solution
                    for candidate in candidates
                ),
                "raw": any(candidate["output"] == solution for candidate in candidates),
            }
            for name in endpoint_names:
                task_hits[name] += int(hits[name])
                query_counts[name] += int(hits[name])
            query_rows.append(
                {
                    "hits": hits,
                    "query_index": query_index,
                    "solution_sha256": canonical_sha256(solution),
                }
            )
        total_queries += len(queries)
        strict_hits: dict[str, bool] = {}
        pair_rates: dict[str, float] = {}
        for name in endpoint_names:
            strict_hit = task_hits[name] == len(queries)
            strict_hits[name] = strict_hit
            strict_counts[name] += int(strict_hit)
            pair_rate = task_hits[name] / len(queries)
            pair_rates[name] = pair_rate
            task_pair_sums[name] += pair_rate
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
    for name in endpoint_names:
        metrics[name] = {
            "query_pass": {
                "correct": query_counts[name],
                "rate": query_counts[name] / total_queries,
                "total": total_queries,
            },
            "strict_task_pass": {
                "correct": strict_counts[name],
                "rate": strict_counts[name] / task_count,
                "total": task_count,
            },
            "task_mean_pair_pass": {
                "rate": task_pair_sums[name] / task_count,
                "task_count": task_count,
            },
        }
    body: dict[str, object] = {
        "blind_cohort_id": blind_cohort_id,
        "candidate_freeze_id": freeze["freeze_id"],
        "metrics": metrics,
        "query_gold_read": True,
        "schema": VARC_RESULT_SCHEMA,
        "solution_source_sha256": solution_source_sha256,
        "tasks": task_rows,
    }
    return {"result_id": canonical_sha256(body), **body}
