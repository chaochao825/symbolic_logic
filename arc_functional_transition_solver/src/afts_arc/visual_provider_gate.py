"""Query-blind cohort preparation and frozen visual-provider scoring.

The visual provider is intentionally treated as a static candidate source.  This
module does not train or invoke a controller.  It creates a task view that cannot
contain query labels, freezes provider predictions, and only then reads the gold
task files for coverage accounting.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from afts_arc.experiment_safety import atomic_write_json, canonical_sha256


COHORT_SCHEMA = "afts.visual-provider-cohort/v1"
EXPOSURE_REGISTRY_SCHEMA = "afts.visual-provider-exposure-registry/v1"
FROZEN_PREDICTIONS_SCHEMA = "afts.visual-provider-frozen-predictions/v2"
SCORE_SCHEMA = "afts.visual-provider-gate/v2"
POSTERIOR_AUDIT_SCHEMA = "afts.visual-posterior-localization/v1"
POSTERIOR_BRIDGE_SCHEMA = "afts.visual-posterior-consensus-bridge/v1"
AGGREGATE_SCHEMA = "afts.visual-provider-aggregate/v1"
TASK_ID_PATTERN = re.compile(r"^[0-9a-f]{8}$")
TASK_ID_TEXT_PATTERN = re.compile(r"(?<![0-9a-f])[0-9a-f]{8}(?![0-9a-f])")


class VisualProviderGateError(RuntimeError):
    """Raised when a visual-provider experiment violates its frozen contract."""


def _load_json_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise VisualProviderGateError(f"JSON artifact is not an object: {path}")
    return payload


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total < 1:
        raise VisualProviderGateError("Wilson interval requires a positive total")
    if not 0 <= successes <= total:
        raise VisualProviderGateError("Wilson successes must lie in [0, total]")
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return center - radius, center + radius


def _task_ids(task_dir: Path) -> set[str]:
    if not task_dir.is_dir():
        raise VisualProviderGateError(f"task directory does not exist: {task_dir}")
    task_ids = {path.stem for path in task_dir.glob("*.json")}
    invalid = sorted(task_id for task_id in task_ids if not TASK_ID_PATTERN.fullmatch(task_id))
    if invalid:
        raise VisualProviderGateError(
            f"task directory contains malformed task IDs: {invalid[:5]}"
        )
    return task_ids


def discover_previously_exposed_task_ids(project_root: Path) -> set[str]:
    """Collect task IDs already named in authored evidence or result artifacts."""

    exposed: set[str] = set()
    results_root = project_root / "results"
    if results_root.is_dir():
        for path in results_root.rglob("*"):
            if path.is_file():
                prefix = path.name[:8]
                if TASK_ID_PATTERN.fullmatch(prefix):
                    exposed.add(prefix)
        for path in results_root.rglob("summary.json"):
            text = path.read_text(encoding="utf-8")
            exposed.update(TASK_ID_TEXT_PATTERN.findall(text))

    authored_roots = (
        project_root / "brief",
        project_root / "notes",
        project_root / "plan",
        project_root / "scripts",
        project_root / "tests",
    )
    text_suffixes = {".md", ".json", ".csv", ".yaml", ".yml", ".py"}
    for root in authored_roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in text_suffixes:
                continue
            exposed.update(TASK_ID_TEXT_PATTERN.findall(path.read_text(encoding="utf-8")))
    return exposed


def _load_exposure_registries(
    paths: Sequence[Path],
) -> tuple[set[str], list[dict[str, object]]]:
    excluded: set[str] = set()
    records: list[dict[str, object]] = []
    for path in paths:
        payload = _load_json_object(path)
        if payload["schema"] != EXPOSURE_REGISTRY_SCHEMA:
            raise VisualProviderGateError(f"exposure registry schema mismatch: {path}")
        task_ids = payload["task_ids"]
        sources = payload["sources"]
        if (
            not isinstance(task_ids, list)
            or not task_ids
            or not isinstance(sources, list)
            or not sources
        ):
            raise VisualProviderGateError(f"exposure registry is malformed: {path}")
        invalid = sorted(
            task_id
            for task_id in task_ids
            if not isinstance(task_id, str) or not TASK_ID_PATTERN.fullmatch(task_id)
        )
        if invalid:
            raise VisualProviderGateError(
                f"exposure registry contains malformed task IDs: {invalid[:5]}"
            )
        if len(task_ids) != len(set(task_ids)):
            raise VisualProviderGateError(
                f"exposure registry contains duplicate task IDs: {path}"
            )
        content = {
            "schema": payload["schema"],
            "sources": sources,
            "task_ids": task_ids,
        }
        expected_registry_id = canonical_sha256(content)
        if payload["registry_id"] != expected_registry_id:
            raise VisualProviderGateError(f"exposure registry ID mismatch: {path}")
        excluded.update(task_ids)
        records.append(
            {
                "path": str(path.resolve()),
                "file_sha256": _file_sha256(path),
                "registry_id": expected_registry_id,
                "task_count": len(task_ids),
            }
        )
    return excluded, records


def make_query_blind_payload(payload: Mapping[str, object]) -> dict[str, object]:
    """Replace every hidden query output with an input-derived sentinel.

    VARC's released augmentation code serializes a test output even during
    inference.  An input copy preserves that code path while carrying no label
    information.  The sentinel is never used as an oracle.
    """

    train = payload["train"]
    test = payload["test"]
    if not isinstance(train, list) or not isinstance(test, list) or not test:
        raise VisualProviderGateError("ARC task must contain train and non-empty test lists")
    blind_test: list[dict[str, object]] = []
    for example in test:
        if not isinstance(example, Mapping):
            raise VisualProviderGateError("ARC test example is not an object")
        query_input = example["input"]
        blind_test.append({"input": query_input, "output": query_input})
    blind_payload = dict(payload)
    blind_payload["test"] = blind_test
    return blind_payload


def select_query_blind_cohort(
    *,
    arc2_training_dir: Path,
    arc1_task_dirs: Sequence[Path],
    project_root: Path,
    explicit_excluded_task_ids: Iterable[str],
    seed: str,
    limit: int,
) -> tuple[tuple[str, ...], dict[str, int]]:
    if not seed:
        raise VisualProviderGateError("selection seed must be non-empty")
    if limit < 1:
        raise VisualProviderGateError("cohort limit must be positive")

    arc2_ids = _task_ids(arc2_training_dir)
    arc1_ids: set[str] = set()
    for task_dir in arc1_task_dirs:
        arc1_ids.update(_task_ids(task_dir))
    exposed_ids = discover_previously_exposed_task_ids(project_root)
    explicit_excluded_ids = set(explicit_excluded_task_ids)
    invalid_explicit = sorted(
        task_id
        for task_id in explicit_excluded_ids
        if not isinstance(task_id, str) or not TASK_ID_PATTERN.fullmatch(task_id)
    )
    if invalid_explicit:
        raise VisualProviderGateError(
            f"explicit exposure set contains malformed task IDs: {invalid_explicit[:5]}"
        )
    eligible = arc2_ids - arc1_ids - exposed_ids - explicit_excluded_ids
    ordered = sorted(
        eligible,
        key=lambda task_id: (
            hashlib.sha256(f"{seed}\0{task_id}".encode("utf-8")).hexdigest(),
            task_id,
        ),
    )
    if len(ordered) < limit:
        raise VisualProviderGateError(
            f"only {len(ordered)} eligible tasks remain for requested limit {limit}"
        )
    counts = {
        "arc2_task_count": len(arc2_ids),
        "arc1_overlap_count": len(arc2_ids & arc1_ids),
        "previously_exposed_count": len(arc2_ids & exposed_ids),
        "explicit_exposure_count": len(arc2_ids & explicit_excluded_ids),
        "explicit_exposure_non_arc1_count": len(
            (arc2_ids & explicit_excluded_ids) - arc1_ids
        ),
        "eligible_count": len(eligible),
    }
    return tuple(ordered[:limit]), counts


def prepare_query_blind_cohort(
    *,
    arc2_training_dir: Path,
    arc1_task_dirs: Sequence[Path],
    project_root: Path,
    exposure_registry_paths: Sequence[Path],
    output_dir: Path,
    seed: str,
    limit: int,
    arc1_source_commit: str,
    arc2_source_commit: str,
) -> dict[str, object]:
    if output_dir.exists():
        raise VisualProviderGateError(f"output directory already exists: {output_dir}")
    explicit_excluded_ids, exposure_registry_records = _load_exposure_registries(
        exposure_registry_paths
    )
    task_ids, counts = select_query_blind_cohort(
        arc2_training_dir=arc2_training_dir,
        arc1_task_dirs=arc1_task_dirs,
        project_root=project_root,
        explicit_excluded_task_ids=explicit_excluded_ids,
        seed=seed,
        limit=limit,
    )
    blind_task_dir = output_dir / "blind_data" / "ARC-AGI-2" / "data" / "evaluation"
    blind_task_dir.mkdir(parents=True)

    records: list[dict[str, object]] = []
    for task_id in task_ids:
        source_path = arc2_training_dir / f"{task_id}.json"
        source_payload = _load_json_object(source_path)
        blind_payload = make_query_blind_payload(source_payload)
        blind_path = blind_task_dir / source_path.name
        atomic_write_json(blind_path, blind_payload)
        records.append(
            {
                "task_id": task_id,
                "source_sha256": _file_sha256(source_path),
                "blind_sha256": _file_sha256(blind_path),
                "query_count": len(source_payload["test"]),
            }
        )

    content: dict[str, object] = {
        "schema": COHORT_SCHEMA,
        "selection": {
            "source_split": "ARC-AGI-2/training",
            "seed": seed,
            "algorithm": "sha256(seed + NUL + task_id), ascending",
            "limit": limit,
            "exclusions": (
                "all ARC-AGI-1 IDs, task IDs named in prior authored evidence, "
                "and task IDs in explicit semantic-exposure registries"
            ),
            **counts,
        },
        "source": {
            "arc1_commit": arc1_source_commit,
            "arc1_task_dirs": [
                str(task_dir.resolve()) for task_dir in arc1_task_dirs
            ],
            "arc2_commit": arc2_source_commit,
            "arc2_training_dir": str(arc2_training_dir.resolve()),
            "exposure_registries": exposure_registry_records,
        },
        "query_blind_contract": {
            "test_output_sentinel": "exact copy of the corresponding test input",
            "gold_output_present": False,
            "provider_may_read": "demonstration input/output pairs and test inputs only",
            "scoring_phase": "after content-addressed prediction freeze",
        },
        "blind_data_root": str((output_dir / "blind_data" / "ARC-AGI-2").resolve()),
        "tasks": records,
    }
    cohort_id = canonical_sha256(content)
    manifest = {"cohort_id": cohort_id, **content}
    atomic_write_json(output_dir / "manifest.json", manifest)
    return manifest


def _validate_grid(grid: object) -> tuple[tuple[int, ...], ...]:
    if not isinstance(grid, list) or not grid:
        raise VisualProviderGateError("prediction grid must be a non-empty list")
    width: int | None = None
    normalized: list[tuple[int, ...]] = []
    for row in grid:
        if not isinstance(row, list) or not row:
            raise VisualProviderGateError("prediction row must be a non-empty list")
        if width is None:
            width = len(row)
        if len(row) != width:
            raise VisualProviderGateError("prediction grid must be rectangular")
        values: list[int] = []
        for value in row:
            if type(value) is not int or not 0 <= value <= 9:
                raise VisualProviderGateError("prediction colors must be integers in [0, 9]")
            values.append(value)
        normalized.append(tuple(values))
    if len(normalized) > 30 or width is None or width > 30:
        raise VisualProviderGateError("prediction grid exceeds ARC's 30x30 limit")
    return tuple(normalized)


def _grid_json(grid: tuple[tuple[int, ...], ...]) -> str:
    return json.dumps(grid, separators=(",", ":"))


def rank_predictions(predictions: Iterable[object]) -> tuple[tuple[tuple[int, ...], ...], ...]:
    normalized = [_validate_grid(prediction) for prediction in predictions]
    if not normalized:
        raise VisualProviderGateError("provider emitted no valid predictions")
    counts = Counter(normalized)
    first_index: dict[tuple[tuple[int, ...], ...], int] = {}
    for index, grid in enumerate(normalized):
        if grid not in first_index:
            first_index[grid] = index
    return tuple(
        sorted(
            counts,
            key=lambda grid: (-counts[grid], first_index[grid], _grid_json(grid)),
        )
    )


def _load_provider_predictions(
    task_ids: Sequence[str], prediction_roots: Sequence[Path]
) -> tuple[dict[str, dict[int, list[object]]], list[dict[str, object]]]:
    if not prediction_roots:
        raise VisualProviderGateError("at least one prediction root is required")
    combined: dict[str, dict[int, list[object]]] = {task_id: {} for task_id in task_ids}
    files: list[dict[str, object]] = []
    for root in prediction_roots:
        for task_id in task_ids:
            path = root / f"{task_id}_predictions.json"
            payload = _load_json_object(path)
            files.append(
                {
                    "task_id": task_id,
                    "path": str(path.resolve()),
                    "sha256": _file_sha256(path),
                }
            )
            for index_text, predictions in payload.items():
                index = int(index_text)
                if not isinstance(predictions, list):
                    raise VisualProviderGateError(
                        f"prediction list is malformed for {task_id} query {index}"
                    )
                combined[task_id].setdefault(index, []).extend(predictions)
    return combined, files


def _validate_provider_predictions(
    predictions: Mapping[str, Mapping[int, Sequence[object]]],
    *,
    invalid_candidate_policy: str,
) -> tuple[
    dict[str, dict[int, list[object]]],
    dict[str, object],
]:
    if invalid_candidate_policy not in {"error", "reject"}:
        raise VisualProviderGateError(
            "invalid-candidate policy must be either 'error' or 'reject'"
        )

    validated: dict[str, dict[int, list[object]]] = {}
    query_records: list[dict[str, object]] = []
    rejection_reasons: Counter[str] = Counter()
    raw_candidate_count = 0
    valid_candidate_count = 0
    for task_id, queries in predictions.items():
        validated[task_id] = {}
        for query_index, samples in sorted(queries.items()):
            valid_samples: list[object] = []
            query_reasons: Counter[str] = Counter()
            for sample_index, sample in enumerate(samples):
                raw_candidate_count += 1
                try:
                    _validate_grid(sample)
                except VisualProviderGateError as error:
                    if invalid_candidate_policy == "error":
                        raise VisualProviderGateError(
                            f"invalid prediction for {task_id} query {query_index} "
                            f"sample {sample_index}: {error}"
                        ) from error
                    reason = str(error)
                    query_reasons[reason] += 1
                    rejection_reasons[reason] += 1
                    continue
                valid_samples.append(sample)
                valid_candidate_count += 1
            if not valid_samples:
                raise VisualProviderGateError(
                    f"provider emitted no valid predictions for {task_id} "
                    f"query {query_index}"
                )
            validated[task_id][query_index] = valid_samples
            query_records.append(
                {
                    "task_id": task_id,
                    "query_index": query_index,
                    "raw_candidate_count": len(samples),
                    "valid_candidate_count": len(valid_samples),
                    "rejected_candidate_count": len(samples) - len(valid_samples),
                    "rejection_reasons": [
                        {"reason": reason, "count": count}
                        for reason, count in sorted(query_reasons.items())
                    ],
                }
            )

    return validated, {
        "invalid_candidate_policy": invalid_candidate_policy,
        "raw_candidate_count": raw_candidate_count,
        "valid_candidate_count": valid_candidate_count,
        "rejected_candidate_count": raw_candidate_count - valid_candidate_count,
        "rejection_reasons": [
            {"reason": reason, "count": count}
            for reason, count in sorted(rejection_reasons.items())
        ],
        "queries": query_records,
    }


def _baseline_flags(payload: Mapping[str, object]) -> dict[str, dict[str, bool]]:
    records = payload["tasks"]
    if not isinstance(records, list):
        raise VisualProviderGateError("baseline summary tasks field is malformed")
    flags: dict[str, dict[str, bool]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise VisualProviderGateError("baseline task record is malformed")
        task_id = record["task_id"]
        policies = record["policies"]
        if not isinstance(task_id, str) or not isinstance(policies, Mapping):
            raise VisualProviderGateError("baseline task identity or policies are malformed")
        raw = False
        selectable = False
        passed = False
        for policy in policies.values():
            if not isinstance(policy, Mapping):
                raise VisualProviderGateError("baseline policy record is malformed")
            metrics = policy["metrics"]
            if not isinstance(metrics, Mapping):
                raise VisualProviderGateError("baseline policy metrics are malformed")
            raw |= metrics["pool_oracle_covered"] is True
            selectable |= metrics["pool_selectable_oracle_covered"] is True
            passed |= metrics["pass_at_k"] is True
        flags[task_id] = {"raw": raw, "selectable": selectable, "pass_at_2": passed}
    return flags


def _object_provider_flags(payload: Mapping[str, object]) -> dict[str, dict[str, bool]]:
    records = payload["tasks"]
    if not isinstance(records, list):
        raise VisualProviderGateError("object-provider summary tasks field is malformed")
    flags: dict[str, dict[str, bool]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise VisualProviderGateError("object-provider task record is malformed")
        task_id = record["task_id"]
        if not isinstance(task_id, str):
            raise VisualProviderGateError("object-provider task ID is malformed")
        flags[task_id] = {
            "raw": record["provider_raw_oracle_covered"] is True,
            "selectable": record["provider_selectable_oracle_covered"] is True,
            "pass_at_2": record["provider_pass_at_2"] is True,
        }
    return flags


def freeze_and_score_provider(
    *,
    cohort_manifest: Path,
    gold_training_dir: Path,
    prediction_roots: Sequence[Path],
    baseline_summary: Path,
    object_provider_summary: Path | None,
    provider_contract: Mapping[str, object],
    output_dir: Path,
    pilot_unique_gate: int,
    invalid_candidate_policy: str,
) -> dict[str, object]:
    if output_dir.exists():
        raise VisualProviderGateError(f"score output directory already exists: {output_dir}")
    if pilot_unique_gate < 1:
        raise VisualProviderGateError("pilot unique-coverage gate must be positive")

    manifest = _load_json_object(cohort_manifest)
    if manifest["schema"] != COHORT_SCHEMA:
        raise VisualProviderGateError("cohort schema mismatch")
    task_records = manifest["tasks"]
    if not isinstance(task_records, list):
        raise VisualProviderGateError("cohort task records are malformed")
    task_ids = tuple(record["task_id"] for record in task_records)
    if any(not isinstance(task_id, str) for task_id in task_ids):
        raise VisualProviderGateError("cohort task ID is malformed")

    raw_predictions, prediction_files = _load_provider_predictions(
        task_ids, prediction_roots
    )
    combined, prediction_validation = _validate_provider_predictions(
        raw_predictions,
        invalid_candidate_policy=invalid_candidate_policy,
    )
    frozen_content: dict[str, object] = {
        "schema": FROZEN_PREDICTIONS_SCHEMA,
        "cohort_id": manifest["cohort_id"],
        "provider_contract": dict(provider_contract),
        "prediction_files": prediction_files,
        "raw_prediction_payload_sha256": canonical_sha256(raw_predictions),
        "validated_prediction_payload_sha256": canonical_sha256(combined),
        "prediction_validation": prediction_validation,
    }
    frozen_id = canonical_sha256(frozen_content)
    output_dir.mkdir(parents=True)
    atomic_write_json(
        output_dir / "frozen_predictions.json",
        {"frozen_prediction_id": frozen_id, **frozen_content},
    )

    baseline = _baseline_flags(_load_json_object(baseline_summary))
    object_flags = (
        _object_provider_flags(_load_json_object(object_provider_summary))
        if object_provider_summary is not None
        else {}
    )
    if set(task_ids) != set(baseline):
        raise VisualProviderGateError("baseline task IDs differ from the frozen cohort")
    if object_provider_summary is not None and set(task_ids) != set(object_flags):
        raise VisualProviderGateError("object-provider task IDs differ from the frozen cohort")

    scored_tasks: list[dict[str, object]] = []
    validation_by_query = {
        (record["task_id"], record["query_index"]): record
        for record in prediction_validation["queries"]
    }
    for task_id in task_ids:
        gold_payload = _load_json_object(gold_training_dir / f"{task_id}.json")
        gold_examples = gold_payload["test"]
        if not isinstance(gold_examples, list) or not gold_examples:
            raise VisualProviderGateError(f"gold queries are malformed for {task_id}")
        query_predictions = combined[task_id]
        if set(query_predictions) != set(range(len(gold_examples))):
            raise VisualProviderGateError(
                f"provider query indices differ from gold queries for {task_id}"
            )
        query_records: list[dict[str, object]] = []
        raw_hits: list[bool] = []
        top1_hits: list[bool] = []
        top2_hits: list[bool] = []
        for index, gold_example in enumerate(gold_examples):
            if not isinstance(gold_example, Mapping):
                raise VisualProviderGateError(f"gold query is malformed for {task_id}")
            gold = _validate_grid(gold_example["output"])
            samples = query_predictions[index]
            validation_record = validation_by_query[(task_id, index)]
            normalized_samples = tuple(_validate_grid(sample) for sample in samples)
            ranked = rank_predictions(samples)
            sample_counts = Counter(normalized_samples)
            raw_hit = gold in ranked
            top1_hit = ranked[0] == gold
            top2_hit = gold in ranked[:2]
            gold_height = len(gold)
            gold_width = len(gold[0])
            same_shape = tuple(
                candidate
                for candidate in ranked
                if len(candidate) == gold_height and len(candidate[0]) == gold_width
            )
            mismatch_counts = tuple(
                sum(
                    predicted != expected
                    for predicted_row, expected_row in zip(candidate, gold, strict=True)
                    for predicted, expected in zip(
                        predicted_row, expected_row, strict=True
                    )
                )
                for candidate in same_shape
            )
            shape_counts = Counter(
                (len(candidate), len(candidate[0]))
                for candidate in normalized_samples
            )
            shape_first_index: dict[tuple[int, int], int] = {}
            for sample_index, candidate in enumerate(normalized_samples):
                shape = (len(candidate), len(candidate[0]))
                if shape not in shape_first_index:
                    shape_first_index[shape] = sample_index
            ranked_shapes = sorted(
                shape_counts,
                key=lambda shape: (
                    -shape_counts[shape],
                    shape_first_index[shape],
                    shape,
                ),
            )
            modal_shape = ranked_shapes[0]
            raw_hits.append(raw_hit)
            top1_hits.append(top1_hit)
            top2_hits.append(top2_hit)
            query_records.append(
                {
                    "query_index": index,
                    "raw_sample_count": validation_record["raw_candidate_count"],
                    "valid_sample_count": len(samples),
                    "rejected_sample_count": validation_record[
                        "rejected_candidate_count"
                    ],
                    "unique_candidate_count": len(ranked),
                    "raw_oracle_covered": raw_hit,
                    "pass_at_1": top1_hit,
                    "pass_at_2": top2_hit,
                    "gold_rank_in_unique_candidates": (
                        ranked.index(gold) + 1 if raw_hit else None
                    ),
                    "gold_sample_support_count": sample_counts[gold],
                    "gold_sample_support_fraction": (
                        sample_counts[gold] / len(normalized_samples)
                    ),
                    "gold_shape": [gold_height, gold_width],
                    "shape_matched_unique_candidate_count": len(same_shape),
                    "best_same_shape_mismatch_count": (
                        min(mismatch_counts) if mismatch_counts else None
                    ),
                    "best_same_shape_mismatch_rate": (
                        min(mismatch_counts) / (gold_height * gold_width)
                        if mismatch_counts
                        else None
                    ),
                    "top_1_same_shape_mismatch_count": (
                        sum(
                            predicted != expected
                            for predicted_row, expected_row in zip(
                                ranked[0], gold, strict=True
                            )
                            for predicted, expected in zip(
                                predicted_row, expected_row, strict=True
                            )
                        )
                        if len(ranked[0]) == gold_height
                        and len(ranked[0][0]) == gold_width
                        else None
                    ),
                    "modal_sample_shape": list(modal_shape),
                    "modal_sample_shape_matches_gold": (
                        modal_shape == (gold_height, gold_width)
                    ),
                    "modal_sample_shape_fraction": (
                        shape_counts[modal_shape] / len(normalized_samples)
                    ),
                    "sample_shape_histogram": [
                        {
                            "height": height,
                            "width": width,
                            "sample_count": shape_counts[(height, width)],
                        }
                        for height, width in ranked_shapes
                    ],
                }
            )

        visual = {
            "raw": all(raw_hits),
            "selectable": all(top2_hits),
            "pass_at_1": all(top1_hits),
            "pass_at_2": all(top2_hits),
        }
        base = dict(baseline[task_id])
        if task_id in object_flags:
            base = {
                key: base[key] or object_flags[task_id][key]
                for key in ("raw", "selectable", "pass_at_2")
            }
        scored_tasks.append(
            {
                "task_id": task_id,
                "base_pool": base,
                "visual_provider": visual,
                "visual_unique_raw_coverage": visual["raw"] and not base["raw"],
                "visual_unique_selectable_coverage": (
                    visual["selectable"] and not base["selectable"]
                ),
                "base_plus_visual_raw_coverage": base["raw"] or visual["raw"],
                "base_plus_visual_selectable_coverage": (
                    base["selectable"] or visual["selectable"]
                ),
                "queries": query_records,
            }
        )

    def count(field: str) -> int:
        return sum(record[field] is True for record in scored_tasks)

    metrics = {
        "task_count": len(scored_tasks),
        "base_raw_coverage": sum(record["base_pool"]["raw"] is True for record in scored_tasks),
        "base_selectable_coverage": sum(
            record["base_pool"]["selectable"] is True for record in scored_tasks
        ),
        "visual_raw_coverage": sum(
            record["visual_provider"]["raw"] is True for record in scored_tasks
        ),
        "visual_selectable_coverage": sum(
            record["visual_provider"]["selectable"] is True for record in scored_tasks
        ),
        "visual_pass_at_1": sum(
            record["visual_provider"]["pass_at_1"] is True for record in scored_tasks
        ),
        "visual_pass_at_2": sum(
            record["visual_provider"]["pass_at_2"] is True for record in scored_tasks
        ),
        "visual_unique_raw_coverage": count("visual_unique_raw_coverage"),
        "visual_unique_selectable_coverage": count("visual_unique_selectable_coverage"),
        "base_plus_visual_raw_coverage": count("base_plus_visual_raw_coverage"),
        "base_plus_visual_selectable_coverage": count(
            "base_plus_visual_selectable_coverage"
        ),
    }
    decision = {
        "pilot_unique_gate": pilot_unique_gate,
        "advance_to_100_task_provider_gate": (
            metrics["visual_unique_selectable_coverage"] >= pilot_unique_gate
        ),
        "controller_remains_frozen": True,
        "full_provider_gate": "not evaluated; requires >=3 unique selectable tasks on 100 fixed tasks",
    }
    content = {
        "schema": SCORE_SCHEMA,
        "cohort_id": manifest["cohort_id"],
        "frozen_prediction_id": frozen_id,
        "provider_contract": dict(provider_contract),
        "metrics": metrics,
        "decision": decision,
        "tasks": scored_tasks,
    }
    result_id = canonical_sha256(content)
    summary = {"result_id": result_id, **content}
    atomic_write_json(output_dir / "summary.json", summary)
    return summary


def _binary_auc(scores: Sequence[float], labels: Sequence[bool]) -> float | None:
    positives = [score for score, label in zip(scores, labels, strict=True) if label]
    negatives = [score for score, label in zip(scores, labels, strict=True) if not label]
    if not positives or not negatives:
        return None
    concordance = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                concordance += 1.0
            elif positive == negative:
                concordance += 0.5
    return concordance / (len(positives) * len(negatives))


def audit_posterior_error_localization(
    *,
    cohort_manifest: Path,
    gold_training_dir: Path,
    prediction_roots: Sequence[Path],
    frozen_predictions: Path,
    output_dir: Path,
    invalid_candidate_policy: str,
    mask_fractions: Sequence[float],
    disagreement_thresholds: Sequence[float],
) -> dict[str, object]:
    """Test whether query-blind sample disagreement localizes top-1 errors.

    This is a diagnostic, not a repair action.  It replays and validates the
    content-frozen provider payload before reading gold, then measures whether
    per-pixel disagreement with the top-ranked grid predicts its error mask.
    """

    if output_dir.exists():
        raise VisualProviderGateError(f"audit output directory already exists: {output_dir}")
    if not mask_fractions or any(not 0.0 < value <= 1.0 for value in mask_fractions):
        raise VisualProviderGateError("mask fractions must lie in (0, 1]")
    if len(mask_fractions) != len(set(mask_fractions)):
        raise VisualProviderGateError("mask fractions must be unique")
    if 0.1 not in mask_fractions:
        raise VisualProviderGateError(
            "mask fractions must include 0.1 for the fixed decision rule"
        )
    if not disagreement_thresholds or any(
        not 0.0 <= value <= 1.0 for value in disagreement_thresholds
    ):
        raise VisualProviderGateError("disagreement thresholds must lie in [0, 1]")
    if len(disagreement_thresholds) != len(set(disagreement_thresholds)):
        raise VisualProviderGateError("disagreement thresholds must be unique")

    manifest = _load_json_object(cohort_manifest)
    if manifest["schema"] != COHORT_SCHEMA:
        raise VisualProviderGateError("cohort schema mismatch")
    task_records = manifest["tasks"]
    if not isinstance(task_records, list):
        raise VisualProviderGateError("cohort task records are malformed")
    task_ids = tuple(record["task_id"] for record in task_records)

    raw_predictions, prediction_files = _load_provider_predictions(
        task_ids, prediction_roots
    )
    combined, prediction_validation = _validate_provider_predictions(
        raw_predictions,
        invalid_candidate_policy=invalid_candidate_policy,
    )
    frozen = _load_json_object(frozen_predictions)
    if frozen["schema"] != FROZEN_PREDICTIONS_SCHEMA:
        raise VisualProviderGateError("frozen-prediction schema mismatch")
    if frozen["cohort_id"] != manifest["cohort_id"]:
        raise VisualProviderGateError("frozen-prediction cohort mismatch")
    if frozen["raw_prediction_payload_sha256"] != canonical_sha256(raw_predictions):
        raise VisualProviderGateError("raw prediction payload differs from freeze")
    if frozen["validated_prediction_payload_sha256"] != canonical_sha256(combined):
        raise VisualProviderGateError("validated prediction payload differs from freeze")
    if (
        frozen["prediction_validation"]["invalid_candidate_policy"]
        != invalid_candidate_policy
    ):
        raise VisualProviderGateError("invalid-candidate policy differs from freeze")

    replay_content: dict[str, object] = {
        "schema": "afts.visual-posterior-replay/v1",
        "cohort_id": manifest["cohort_id"],
        "frozen_prediction_id": frozen["frozen_prediction_id"],
        "prediction_files": prediction_files,
        "raw_prediction_payload_sha256": canonical_sha256(raw_predictions),
        "validated_prediction_payload_sha256": canonical_sha256(combined),
        "invalid_candidate_policy": invalid_candidate_policy,
        "mask_fractions": sorted(mask_fractions),
        "disagreement_thresholds": sorted(disagreement_thresholds),
    }
    replay_id = canonical_sha256(replay_content)
    output_dir.mkdir(parents=True)
    atomic_write_json(
        output_dir / "replay_receipt.json",
        {"replay_id": replay_id, **replay_content},
    )

    query_records: list[dict[str, object]] = []
    for task_id in task_ids:
        gold_payload = _load_json_object(gold_training_dir / f"{task_id}.json")
        gold_examples = gold_payload["test"]
        if not isinstance(gold_examples, list) or not gold_examples:
            raise VisualProviderGateError(f"gold queries are malformed for {task_id}")
        query_predictions = combined[task_id]
        if set(query_predictions) != set(range(len(gold_examples))):
            raise VisualProviderGateError(
                f"provider query indices differ from gold queries for {task_id}"
            )
        for query_index, gold_example in enumerate(gold_examples):
            if not isinstance(gold_example, Mapping):
                raise VisualProviderGateError(f"gold query is malformed for {task_id}")
            gold = _validate_grid(gold_example["output"])
            samples = query_predictions[query_index]
            normalized_samples = tuple(_validate_grid(sample) for sample in samples)
            top_1 = rank_predictions(samples)[0]
            shape_eligible = (
                len(top_1) == len(gold) and len(top_1[0]) == len(gold[0])
            )
            record: dict[str, object] = {
                "task_id": task_id,
                "query_index": query_index,
                "top_1_exact": top_1 == gold,
                "top_1_shape": [len(top_1), len(top_1[0])],
                "gold_shape": [len(gold), len(gold[0])],
                "shape_eligible": shape_eligible,
            }
            if not shape_eligible:
                record["localization"] = None
                query_records.append(record)
                continue

            same_shape_samples = tuple(
                sample
                for sample in normalized_samples
                if len(sample) == len(top_1) and len(sample[0]) == len(top_1[0])
            )
            disagreement_scores: list[float] = []
            error_labels: list[bool] = []
            for row_index, row in enumerate(top_1):
                for column_index, value in enumerate(row):
                    disagreement_scores.append(
                        sum(
                            sample[row_index][column_index] != value
                            for sample in same_shape_samples
                        )
                        / len(same_shape_samples)
                    )
                    error_labels.append(value != gold[row_index][column_index])
            cell_count = len(error_labels)
            error_count = sum(error_labels)
            ranked_indices = sorted(
                range(cell_count),
                key=lambda index: (-disagreement_scores[index], index),
            )
            masks: list[dict[str, object]] = []
            for fraction in sorted(mask_fractions):
                selected_count = max(1, math.ceil(cell_count * fraction))
                selected = ranked_indices[:selected_count]
                hit_count = sum(error_labels[index] for index in selected)
                masks.append(
                    {
                        "fraction": fraction,
                        "selected_count": selected_count,
                        "error_hit_count": hit_count,
                        "precision": hit_count / selected_count,
                        "recall": hit_count / error_count if error_count else None,
                        "random_expected_recall": selected_count / cell_count,
                        "recall_lift_over_random": (
                            (hit_count / error_count) / (selected_count / cell_count)
                            if error_count
                            else None
                        ),
                    }
                )
            threshold_masks: list[dict[str, object]] = []
            for threshold in sorted(disagreement_thresholds):
                selected = [
                    index
                    for index, score in enumerate(disagreement_scores)
                    if score >= threshold
                ]
                hit_count = sum(error_labels[index] for index in selected)
                threshold_masks.append(
                    {
                        "threshold": threshold,
                        "selected_count": len(selected),
                        "error_hit_count": hit_count,
                        "precision": hit_count / len(selected) if selected else None,
                        "recall": hit_count / error_count if error_count else None,
                    }
                )
            error_scores = [
                score
                for score, is_error in zip(
                    disagreement_scores, error_labels, strict=True
                )
                if is_error
            ]
            correct_scores = [
                score
                for score, is_error in zip(
                    disagreement_scores, error_labels, strict=True
                )
                if not is_error
            ]
            record["localization"] = {
                "same_shape_sample_count": len(same_shape_samples),
                "cell_count": cell_count,
                "error_count": error_count,
                "error_rate": error_count / cell_count,
                "mean_disagreement": statistics.fmean(disagreement_scores),
                "mean_error_pixel_disagreement": (
                    statistics.fmean(error_scores) if error_scores else None
                ),
                "mean_correct_pixel_disagreement": (
                    statistics.fmean(correct_scores) if correct_scores else None
                ),
                "error_localization_auc": _binary_auc(
                    disagreement_scores, error_labels
                ),
                "fraction_masks": masks,
                "threshold_masks": threshold_masks,
            }
            query_records.append(record)

    incorrect_shape_eligible = [
        record
        for record in query_records
        if record["shape_eligible"] is True and record["top_1_exact"] is False
    ]
    aucs = [
        record["localization"]["error_localization_auc"]
        for record in incorrect_shape_eligible
        if record["localization"]["error_localization_auc"] is not None
    ]
    if not incorrect_shape_eligible:
        raise VisualProviderGateError(
            "posterior localization requires an incorrect shape-eligible query"
        )
    fraction_aggregates: list[dict[str, object]] = []
    for fraction in sorted(mask_fractions):
        masks = [
            next(
                mask
                for mask in record["localization"]["fraction_masks"]
                if mask["fraction"] == fraction
            )
            for record in incorrect_shape_eligible
        ]
        fraction_aggregates.append(
            {
                "fraction": fraction,
                "query_count": len(masks),
                "macro_precision": statistics.fmean(
                    mask["precision"] for mask in masks
                ),
                "macro_recall": statistics.fmean(mask["recall"] for mask in masks),
                "macro_recall_lift_over_random": statistics.fmean(
                    mask["recall_lift_over_random"] for mask in masks
                ),
            }
        )
    ten_percent = next(
        aggregate
        for aggregate in fraction_aggregates
        if aggregate["fraction"] == 0.1
    )
    aggregate = {
        "task_count": len(task_ids),
        "query_count": len(query_records),
        "top_1_exact_query_count": sum(
            record["top_1_exact"] is True for record in query_records
        ),
        "shape_eligible_query_count": sum(
            record["shape_eligible"] is True for record in query_records
        ),
        "incorrect_shape_eligible_query_count": len(incorrect_shape_eligible),
        "auc_query_count": len(aucs),
        "mean_error_localization_auc": statistics.fmean(aucs) if aucs else None,
        "median_error_localization_auc": statistics.median(aucs) if aucs else None,
        "fraction_masks": fraction_aggregates,
    }
    decision = {
        "exploratory_diagnostic_only": True,
        "frontier_change_tested": False,
        "repair_success_tested": False,
        "strong_localization_signal": (
            bool(aucs)
            and aggregate["median_error_localization_auc"] >= 0.70
            and ten_percent["macro_recall_lift_over_random"] >= 2.0
        ),
        "strong_signal_rule": (
            "median pixel-error AUROC >= 0.70 and macro recall lift of the "
            "fixed 10% disagreement mask >= 2.0"
        ),
    }
    content = {
        "schema": POSTERIOR_AUDIT_SCHEMA,
        "cohort_id": manifest["cohort_id"],
        "frozen_prediction_id": frozen["frozen_prediction_id"],
        "replay_id": replay_id,
        "aggregate": aggregate,
        "decision": decision,
        "queries": query_records,
    }
    result_id = canonical_sha256(content)
    summary = {"result_id": result_id, **content}
    atomic_write_json(output_dir / "summary.json", summary)
    return summary


def audit_posterior_consensus_bridge(
    *,
    cohort_manifest: Path,
    gold_training_dir: Path,
    prediction_roots: Sequence[Path],
    frozen_predictions: Path,
    visual_score_summary: Path,
    output_dir: Path,
    invalid_candidate_policy: str,
) -> dict[str, object]:
    """Compose one query-blind pixel-consensus candidate per visual posterior."""

    if output_dir.exists():
        raise VisualProviderGateError(
            f"bridge output directory already exists: {output_dir}"
        )
    manifest = _load_json_object(cohort_manifest)
    if manifest["schema"] != COHORT_SCHEMA:
        raise VisualProviderGateError("cohort schema mismatch")
    task_records = manifest["tasks"]
    if not isinstance(task_records, list):
        raise VisualProviderGateError("cohort task records are malformed")
    task_ids = tuple(record["task_id"] for record in task_records)
    query_counts = {
        record["task_id"]: record["query_count"] for record in task_records
    }

    raw_predictions, prediction_files = _load_provider_predictions(
        task_ids, prediction_roots
    )
    combined, prediction_validation = _validate_provider_predictions(
        raw_predictions,
        invalid_candidate_policy=invalid_candidate_policy,
    )
    frozen = _load_json_object(frozen_predictions)
    if frozen["schema"] != FROZEN_PREDICTIONS_SCHEMA:
        raise VisualProviderGateError("frozen-prediction schema mismatch")
    if frozen["cohort_id"] != manifest["cohort_id"]:
        raise VisualProviderGateError("frozen-prediction cohort mismatch")
    if frozen["raw_prediction_payload_sha256"] != canonical_sha256(raw_predictions):
        raise VisualProviderGateError("raw prediction payload differs from freeze")
    if frozen["validated_prediction_payload_sha256"] != canonical_sha256(combined):
        raise VisualProviderGateError("validated prediction payload differs from freeze")
    if (
        frozen["prediction_validation"]["invalid_candidate_policy"]
        != invalid_candidate_policy
    ):
        raise VisualProviderGateError("invalid-candidate policy differs from freeze")

    candidate_tasks: list[dict[str, object]] = []
    for task_id in task_ids:
        queries = combined[task_id]
        if set(queries) != set(range(query_counts[task_id])):
            raise VisualProviderGateError(
                f"provider query indices differ from cohort manifest for {task_id}"
            )
        candidate_queries: list[dict[str, object]] = []
        for query_index in range(query_counts[task_id]):
            samples = queries[query_index]
            normalized_samples = tuple(_validate_grid(sample) for sample in samples)
            ranked = rank_predictions(samples)
            top_1 = ranked[0]
            same_shape_samples = tuple(
                sample
                for sample in normalized_samples
                if len(sample) == len(top_1) and len(sample[0]) == len(top_1[0])
            )
            disagreement_scores: list[float] = []
            consensus_rows: list[list[int]] = []
            for row_index, row in enumerate(top_1):
                consensus_row: list[int] = []
                for column_index, top_1_color in enumerate(row):
                    color_counts = Counter(
                        sample[row_index][column_index]
                        for sample in same_shape_samples
                    )
                    consensus_color = min(
                        color_counts,
                        key=lambda color: (
                            -color_counts[color],
                            color != top_1_color,
                            color,
                        ),
                    )
                    consensus_row.append(consensus_color)
                    disagreement_scores.append(
                        1.0
                        - color_counts[top_1_color] / len(same_shape_samples)
                    )
                consensus_rows.append(consensus_row)
            consensus = _validate_grid(consensus_rows)
            raw_pool = set(normalized_samples)
            action_triggered = any(score > 0.0 for score in disagreement_scores)
            novel_frontier = action_triggered and consensus not in raw_pool
            bridge_top_2 = [top_1]
            if consensus != top_1:
                bridge_top_2.append(consensus)
            elif len(ranked) > 1:
                bridge_top_2.append(ranked[1])
            candidate_queries.append(
                {
                    "query_index": query_index,
                    "certificate": {
                        "type": "pixel_disagreement",
                        "same_shape_sample_count": len(same_shape_samples),
                        "disagreement_pixel_count": sum(
                            score > 0.0 for score in disagreement_scores
                        ),
                        "mean_disagreement": statistics.fmean(disagreement_scores),
                        "max_disagreement": max(disagreement_scores),
                    },
                    "action": {
                        "type": "posterior_consensus_compose",
                        "triggered": action_triggered,
                        "consensus_grid": [list(row) for row in consensus],
                        "consensus_grid_sha256": canonical_sha256(
                            {"grid": consensus_rows}
                        ),
                        "differs_from_top_1": consensus != top_1,
                        "raw_pool_contains_consensus": consensus in raw_pool,
                        "novel_frontier": novel_frontier,
                        "cell_vote_cost": (
                            len(same_shape_samples) * len(top_1) * len(top_1[0])
                        ),
                    },
                    "bridge_top_2": [
                        [list(row) for row in candidate]
                        for candidate in bridge_top_2
                    ],
                }
            )
        candidate_tasks.append({"task_id": task_id, "queries": candidate_queries})

    candidate_content: dict[str, object] = {
        "schema": "afts.visual-posterior-consensus-candidates/v1",
        "cohort_id": manifest["cohort_id"],
        "frozen_prediction_id": frozen["frozen_prediction_id"],
        "prediction_files": prediction_files,
        "validated_prediction_payload_sha256": canonical_sha256(combined),
        "invalid_candidate_policy": invalid_candidate_policy,
        "prediction_validation": prediction_validation,
        "tasks": candidate_tasks,
    }
    candidate_id = canonical_sha256(candidate_content)
    output_dir.mkdir(parents=True)
    atomic_write_json(
        output_dir / "bridge_candidates.json",
        {"candidate_id": candidate_id, **candidate_content},
    )

    score = _load_json_object(visual_score_summary)
    if score["schema"] != SCORE_SCHEMA:
        raise VisualProviderGateError("visual score schema mismatch")
    if score["cohort_id"] != manifest["cohort_id"]:
        raise VisualProviderGateError("visual score cohort mismatch")
    if score["frozen_prediction_id"] != frozen["frozen_prediction_id"]:
        raise VisualProviderGateError("visual score freeze mismatch")
    score_records = score["tasks"]
    if not isinstance(score_records, list):
        raise VisualProviderGateError("visual score task records are malformed")
    score_by_task = {record["task_id"]: record for record in score_records}
    if set(score_by_task) != set(task_ids):
        raise VisualProviderGateError("visual score task IDs differ from cohort")

    scored_tasks: list[dict[str, object]] = []
    candidate_by_task = {
        record["task_id"]: record for record in candidate_tasks
    }
    for task_id in task_ids:
        gold_payload = _load_json_object(gold_training_dir / f"{task_id}.json")
        gold_examples = gold_payload["test"]
        if not isinstance(gold_examples, list) or not gold_examples:
            raise VisualProviderGateError(f"gold queries are malformed for {task_id}")
        score_record = score_by_task[task_id]
        score_queries = score_record["queries"]
        candidate_queries = candidate_by_task[task_id]["queries"]
        if len(gold_examples) != len(score_queries) or len(gold_examples) != len(
            candidate_queries
        ):
            raise VisualProviderGateError(f"query count mismatch for {task_id}")
        query_results: list[dict[str, object]] = []
        for query_index, gold_example in enumerate(gold_examples):
            if not isinstance(gold_example, Mapping):
                raise VisualProviderGateError(f"gold query is malformed for {task_id}")
            gold = _validate_grid(gold_example["output"])
            candidate_record = candidate_queries[query_index]
            consensus = _validate_grid(
                candidate_record["action"]["consensus_grid"]
            )
            bridge_top_2 = tuple(
                _validate_grid(candidate)
                for candidate in candidate_record["bridge_top_2"]
            )
            consensus_exact = consensus == gold
            query_results.append(
                {
                    "query_index": query_index,
                    "existing_raw_oracle_covered": score_queries[query_index][
                        "raw_oracle_covered"
                    ],
                    "existing_pass_at_2": score_queries[query_index]["pass_at_2"],
                    "consensus_exact": consensus_exact,
                    "consensus_novel_exact": (
                        consensus_exact
                        and candidate_record["action"]["novel_frontier"] is True
                    ),
                    "bridge_raw_oracle_covered": (
                        score_queries[query_index]["raw_oracle_covered"] is True
                        or consensus_exact
                    ),
                    "bridge_pass_at_2": gold in bridge_top_2,
                    "novel_frontier": candidate_record["action"][
                        "novel_frontier"
                    ],
                }
            )
        bridge_raw = all(
            result["bridge_raw_oracle_covered"] is True for result in query_results
        )
        bridge_pass_2 = all(
            result["bridge_pass_at_2"] is True for result in query_results
        )
        visual_raw = score_record["visual_provider"]["raw"] is True
        visual_pass_2 = score_record["visual_provider"]["pass_at_2"] is True
        base_raw = score_record["base_pool"]["raw"] is True
        base_pass_2 = score_record["base_pool"]["pass_at_2"] is True
        missing_query_recovered_by_novel = any(
            result["existing_raw_oracle_covered"] is False
            and result["consensus_novel_exact"] is True
            for result in query_results
        )
        unique_selectable = bridge_pass_2 and not (visual_pass_2 or base_pass_2)
        scored_tasks.append(
            {
                "task_id": task_id,
                "bridge_raw_coverage": bridge_raw,
                "bridge_pass_at_2": bridge_pass_2,
                "unique_raw_recovery": bridge_raw and not (visual_raw or base_raw),
                "unique_selectable_recovery": unique_selectable,
                "unique_selectable_novel_recovery": (
                    unique_selectable and missing_query_recovered_by_novel
                ),
                "queries": query_results,
            }
        )

    def count(field: str) -> int:
        return sum(record[field] is True for record in scored_tasks)

    candidate_queries = [
        query for task in candidate_tasks for query in task["queries"]
    ]
    scored_queries = [query for task in scored_tasks for query in task["queries"]]
    metrics = {
        "task_count": len(task_ids),
        "query_count": len(candidate_queries),
        "triggered_action_query_count": sum(
            query["action"]["triggered"] is True for query in candidate_queries
        ),
        "novel_frontier_query_count": sum(
            query["action"]["novel_frontier"] is True for query in candidate_queries
        ),
        "consensus_exact_query_count": sum(
            query["consensus_exact"] is True for query in scored_queries
        ),
        "consensus_novel_exact_query_count": sum(
            query["consensus_novel_exact"] is True for query in scored_queries
        ),
        "bridge_raw_coverage": count("bridge_raw_coverage"),
        "bridge_pass_at_2": count("bridge_pass_at_2"),
        "unique_raw_recovery": count("unique_raw_recovery"),
        "unique_selectable_recovery": count("unique_selectable_recovery"),
        "unique_selectable_novel_recovery": count(
            "unique_selectable_novel_recovery"
        ),
        "cell_vote_cost": sum(
            query["action"]["cell_vote_cost"] for query in candidate_queries
        ),
    }
    decision = {
        "advance_to_matched_cost_bridge_gate": (
            metrics["novel_frontier_query_count"] > 0
            and metrics["unique_selectable_novel_recovery"] >= 1
        ),
        "controller_remains_frozen": True,
        "matched_cost_restart_tested": False,
        "repair_gate_5_per_100_tested": False,
    }
    content = {
        "schema": POSTERIOR_BRIDGE_SCHEMA,
        "cohort_id": manifest["cohort_id"],
        "frozen_prediction_id": frozen["frozen_prediction_id"],
        "candidate_id": candidate_id,
        "metrics": metrics,
        "decision": decision,
        "tasks": scored_tasks,
    }
    result_id = canonical_sha256(content)
    summary = {"result_id": result_id, **content}
    atomic_write_json(output_dir / "summary.json", summary)
    return summary


def _verify_result_artifact(
    payload: Mapping[str, object], expected_schema: str, path: Path
) -> str:
    if payload["schema"] != expected_schema:
        raise VisualProviderGateError(f"result schema mismatch: {path}")
    body = dict(payload)
    declared = body.pop("result_id")
    if not isinstance(declared, str) or declared != canonical_sha256(body):
        raise VisualProviderGateError(f"result content ID mismatch: {path}")
    return declared


def aggregate_visual_provider_runs(
    *,
    score_summaries: Sequence[Path],
    posterior_summaries: Sequence[Path],
    output_dir: Path,
    confirmation_score_index: int,
    bridge_summary: Path | None,
) -> dict[str, object]:
    """Aggregate disjoint, content-verified visual-provider cohorts.

    The aggregation consumes only already-scored immutable summaries.  It does
    not rerank predictions, alter candidate validity, or invoke a controller.
    """

    if output_dir.exists():
        raise VisualProviderGateError(
            f"aggregate output directory already exists: {output_dir}"
        )
    if not score_summaries or len(score_summaries) != len(posterior_summaries):
        raise VisualProviderGateError(
            "score and posterior summary lists must be non-empty and equal length"
        )
    if not 0 <= confirmation_score_index < len(score_summaries):
        raise VisualProviderGateError("confirmation score index is out of range")

    scores = [_load_json_object(path) for path in score_summaries]
    posteriors = [_load_json_object(path) for path in posterior_summaries]
    input_records: list[dict[str, object]] = []
    all_task_ids: set[str] = set()
    anchor_contract_fingerprint: str | None = None
    score_metric_fields = (
        "base_raw_coverage",
        "base_selectable_coverage",
        "visual_raw_coverage",
        "visual_selectable_coverage",
        "visual_pass_at_1",
        "visual_pass_at_2",
        "visual_unique_raw_coverage",
        "visual_unique_selectable_coverage",
        "base_plus_visual_raw_coverage",
        "base_plus_visual_selectable_coverage",
    )
    totals = {field: 0 for field in score_metric_fields}
    total_tasks = 0
    total_gpu_seconds = 0
    total_raw_candidates = 0
    total_valid_candidates = 0
    total_rejected_candidates = 0
    scored_queries: list[Mapping[str, object]] = []
    posterior_queries: list[Mapping[str, object]] = []

    for score_path, posterior_path, score, posterior in zip(
        score_summaries,
        posterior_summaries,
        scores,
        posteriors,
        strict=True,
    ):
        score_result_id = _verify_result_artifact(score, SCORE_SCHEMA, score_path)
        posterior_result_id = _verify_result_artifact(
            posterior, POSTERIOR_AUDIT_SCHEMA, posterior_path
        )
        if score["cohort_id"] != posterior["cohort_id"]:
            raise VisualProviderGateError(
                f"score/posterior cohort mismatch: {score_path}"
            )
        if score["frozen_prediction_id"] != posterior["frozen_prediction_id"]:
            raise VisualProviderGateError(
                f"score/posterior prediction freeze mismatch: {score_path}"
            )

        tasks = score["tasks"]
        if not isinstance(tasks, list) or not tasks:
            raise VisualProviderGateError(f"score tasks are malformed: {score_path}")
        task_ids = [task["task_id"] for task in tasks]
        if len(task_ids) != len(set(task_ids)):
            raise VisualProviderGateError(f"score contains duplicate tasks: {score_path}")
        overlap = all_task_ids.intersection(task_ids)
        if overlap:
            raise VisualProviderGateError(
                f"aggregate cohorts overlap on task IDs: {sorted(overlap)}"
            )
        all_task_ids.update(task_ids)
        metrics = score["metrics"]
        if metrics["task_count"] != len(task_ids):
            raise VisualProviderGateError(f"score task count mismatch: {score_path}")
        if posterior["aggregate"]["task_count"] != len(task_ids):
            raise VisualProviderGateError(
                f"posterior task count mismatch: {posterior_path}"
            )
        posterior_task_ids = {query["task_id"] for query in posterior["queries"]}
        if posterior_task_ids != set(task_ids):
            raise VisualProviderGateError(
                f"posterior task IDs differ from score tasks: {posterior_path}"
            )

        contract = score["provider_contract"]
        if contract["controller_frozen"] is not True:
            raise VisualProviderGateError(f"controller was not frozen: {score_path}")
        contract_fingerprint = canonical_sha256(
            {
                "source": contract["source"],
                "test_time_configuration": contract["test_time_configuration"],
                "candidate_policy": {
                    "invalid_candidate_policy": contract["candidate_contract"][
                        "invalid_candidate_policy"
                    ],
                    "invalid_candidate_rule": contract["candidate_contract"][
                        "invalid_candidate_rule"
                    ],
                    "ranking": contract["candidate_contract"]["ranking"],
                },
                "query_blind_protocol": {
                    "gold_output_present_in_provider_root": contract[
                        "query_blind_protocol"
                    ]["gold_output_present_in_provider_root"],
                    "provider_visible_test_output": contract["query_blind_protocol"][
                        "provider_visible_test_output"
                    ],
                    "released_loader_repair": contract["query_blind_protocol"][
                        "released_loader_repair"
                    ],
                },
            }
        )
        if anchor_contract_fingerprint is None:
            anchor_contract_fingerprint = contract_fingerprint
        elif anchor_contract_fingerprint != contract_fingerprint:
            raise VisualProviderGateError(
                f"provider semantics differ across cohorts: {score_path}"
            )

        for field in score_metric_fields:
            totals[field] += metrics[field]
        total_tasks += len(task_ids)
        total_gpu_seconds += contract["compute"]["gpu_seconds"]
        total_raw_candidates += contract["candidate_contract"][
            "raw_candidate_count"
        ]
        total_valid_candidates += contract["candidate_contract"][
            "valid_candidate_count"
        ]
        total_rejected_candidates += contract["candidate_contract"][
            "rejected_candidate_count"
        ]
        scored_queries.extend(query for task in tasks for query in task["queries"])
        posterior_queries.extend(posterior["queries"])
        input_records.append(
            {
                "cohort_id": score["cohort_id"],
                "frozen_prediction_id": score["frozen_prediction_id"],
                "posterior_result_id": posterior_result_id,
                "posterior_summary_sha256": _file_sha256(posterior_path),
                "score_result_id": score_result_id,
                "score_summary_sha256": _file_sha256(score_path),
                "task_count": len(task_ids),
            }
        )

    intervals: dict[str, dict[str, float | int]] = {}
    for field in (
        "visual_raw_coverage",
        "visual_selectable_coverage",
        "visual_unique_raw_coverage",
        "visual_unique_selectable_coverage",
    ):
        lower, upper = _wilson_interval(totals[field], total_tasks)
        intervals[field] = {
            "successes": totals[field],
            "total": total_tasks,
            "rate": totals[field] / total_tasks,
            "wilson_95_lower": lower,
            "wilson_95_upper": upper,
        }

    raw_hit_ranks = sorted(
        query["gold_rank_in_unique_candidates"]
        for query in scored_queries
        if query["gold_rank_in_unique_candidates"] is not None
    )
    raw_absent_shape_eligible = [
        query
        for query in scored_queries
        if query["raw_oracle_covered"] is False
        and query["best_same_shape_mismatch_rate"] is not None
    ]
    posterior_auc_values: list[float] = []
    posterior_mask_records: dict[float, list[Mapping[str, object]]] = {}
    for query in posterior_queries:
        localization = query["localization"]
        if localization is None or query["top_1_exact"] is True:
            continue
        auc = localization["error_localization_auc"]
        if auc is not None:
            posterior_auc_values.append(auc)
        for record in localization["fraction_masks"]:
            fraction = record["fraction"]
            posterior_mask_records.setdefault(fraction, []).append(record)

    if not posterior_auc_values:
        raise VisualProviderGateError(
            "aggregate has no incorrect shape-eligible posterior queries"
        )
    if 0.1 not in posterior_mask_records:
        raise VisualProviderGateError(
            "posterior summaries do not contain the fixed 10% mask"
        )

    fraction_masks: list[dict[str, object]] = []
    for fraction, records in sorted(posterior_mask_records.items()):
        precision_values = [
            record["precision"] for record in records if record["precision"] is not None
        ]
        recall_values = [
            record["recall"] for record in records if record["recall"] is not None
        ]
        lift_values = [
            record["recall_lift_over_random"]
            for record in records
            if record["recall_lift_over_random"] is not None
        ]
        fraction_masks.append(
            {
                "fraction": fraction,
                "macro_precision": statistics.fmean(precision_values),
                "macro_recall": statistics.fmean(recall_values),
                "macro_recall_lift_over_random": statistics.fmean(lift_values),
                "query_count": len(records),
            }
        )

    fixed_ten_percent = next(
        record for record in fraction_masks if record["fraction"] == 0.1
    )
    posterior_summary = {
        "auc_query_count": len(posterior_auc_values),
        "fraction_masks": fraction_masks,
        "incorrect_shape_eligible_query_count": len(posterior_auc_values),
        "mean_error_localization_auc": statistics.fmean(posterior_auc_values),
        "median_error_localization_auc": statistics.median(posterior_auc_values),
        "query_count": len(posterior_queries),
        "strong_localization_signal": (
            statistics.median(posterior_auc_values) >= 0.70
            and fixed_ten_percent["macro_recall_lift_over_random"] >= 2.0
        ),
    }

    bridge_record: dict[str, object] | None = None
    if bridge_summary is not None:
        bridge = _load_json_object(bridge_summary)
        bridge_result_id = _verify_result_artifact(
            bridge, POSTERIOR_BRIDGE_SCHEMA, bridge_summary
        )
        bridge_record = {
            "decision": bridge["decision"],
            "metrics": bridge["metrics"],
            "result_id": bridge_result_id,
            "summary_sha256": _file_sha256(bridge_summary),
        }

    confirmation_metrics = scores[confirmation_score_index]["metrics"]
    content = {
        "schema": AGGREGATE_SCHEMA,
        "inputs": input_records,
        "provider_semantics_fingerprint": anchor_contract_fingerprint,
        "solver_track": {
            "metrics": {"task_count": total_tasks, **totals},
            "confidence_intervals": intervals,
            "query_diagnostics": {
                "query_count": len(scored_queries),
                "raw_hit_query_count": len(raw_hit_ranks),
                "raw_hit_ranks": raw_hit_ranks,
                "raw_hits_outside_top_2": sum(rank > 2 for rank in raw_hit_ranks),
                "raw_absent_shape_eligible_query_count": len(
                    raw_absent_shape_eligible
                ),
                "raw_absent_best_mismatch_at_most_1_percent": sum(
                    query["best_same_shape_mismatch_rate"] <= 0.01
                    for query in raw_absent_shape_eligible
                ),
                "raw_absent_best_mismatch_at_most_5_percent": sum(
                    query["best_same_shape_mismatch_rate"] <= 0.05
                    for query in raw_absent_shape_eligible
                ),
                "raw_absent_best_mismatch_at_most_10_percent": sum(
                    query["best_same_shape_mismatch_rate"] <= 0.10
                    for query in raw_absent_shape_eligible
                ),
            },
            "compute": {
                "gpu_seconds": total_gpu_seconds,
                "raw_candidate_count": total_raw_candidates,
                "rejected_candidate_count": total_rejected_candidates,
                "valid_candidate_count": total_valid_candidates,
            },
        },
        "mechanism_track": {
            "posterior_localization": posterior_summary,
            "pixel_consensus_bridge": bridge_record,
            "controller_remained_frozen": True,
        },
        "decision": {
            "confirmation_nonzero_unique_selectable_replicated": (
                confirmation_metrics["visual_unique_selectable_coverage"] >= 1
            ),
            "confirmation_strong_small_cohort_replication": (
                confirmation_metrics["visual_unique_selectable_coverage"] >= 2
            ),
            "full_100_task_provider_gate_evaluated": total_tasks >= 100,
            "controller_training_authorized": False,
            "next_action": (
                "compile posterior disagreement into object/program search constraints; "
                "do not use pixelwise consensus or train a router"
            ),
        },
    }
    result_id = canonical_sha256(content)
    summary = {"result_id": result_id, **content}
    output_dir.mkdir(parents=True)
    atomic_write_json(output_dir / "summary.json", summary)
    return summary
