#!/usr/bin/env python3
"""Audit whether residuals add predictive value to ARC option switching.

This is an offline development audit over already-opened frozen summaries.  It
never reads query outputs.  Post-hoc pass labels train/evaluate source-utility
models; inference features contain only task facts and observable option
outcomes from single-source runs.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

from afts_arc.experiment_safety import (
    ARC_EXPERIMENT_SOURCE_PATHS,
    ExperimentSafetyError,
    TaskFingerprint,
    assert_three_axis_disjoint,
    atomic_write_json,
    capture_git_source_provenance,
    file_sha256,
    runtime_metadata,
    validate_clean_source_binding,
    validate_summary,
    verify_source_provenance_unchanged,
)
from afts_arc.hybrid.metareasoning import conditional_mutual_information


SCHEMA_VERSION = "afts.option-switching-audit/v2"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
SOURCES = ("dsl", "ca", "scene")
SOURCE_POLICIES = {
    "dsl": "dsl_only",
    "ca": "ca_only",
    "scene": "scene_only",
}
CANONICAL_POLICY = "structured_deliberation"


def _sha256(value: object) -> str:
    text = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def _numeric(value: object) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError("expected a numeric summary field")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError("summary numeric fields must be finite and non-negative")
    return result


@dataclass(frozen=True, slots=True)
class OptionOutcome:
    success: bool
    residual_tokens: tuple[str, ...]
    phase: str
    numeric: tuple[tuple[str, float], ...]
    native_cost: tuple[tuple[str, float], ...]

    def numeric_mapping(self) -> dict[str, float]:
        return dict(self.numeric)

    def cost_mapping(self) -> dict[str, float]:
        return dict(self.native_cost)


@dataclass(frozen=True, slots=True)
class TaskRecord:
    task_id: str
    task_source_sha256: str
    blind_content_sha256: str
    static_tokens: tuple[str, ...]
    options: tuple[tuple[str, OptionOutcome], ...]
    pool_oracle_covered: bool
    pool_selectable_oracle_covered: bool
    canonical_pass: bool

    def option(self, source: str) -> OptionOutcome:
        return dict(self.options)[source]


def _option_outcome(payload: Mapping[str, object]) -> OptionOutcome:
    sketch = payload["deliberation_sketch"]
    budget = payload["budget_used"]
    actions = payload["actions"]
    metrics = payload["metrics"]
    if not isinstance(sketch, Mapping) or not isinstance(budget, Mapping):
        raise TypeError("invalid option summary")
    if not isinstance(actions, list) or not isinstance(metrics, Mapping):
        raise TypeError("invalid option actions/metrics")

    status_counts = Counter(
        str(item["status"])
        for item in actions
        if isinstance(item, Mapping) and item.get("kind") != "stop"
    )
    kind_counts = Counter(
        str(item["kind"])
        for item in actions
        if isinstance(item, Mapping) and item.get("kind") != "stop"
    )
    numeric = {
        "accepted_candidates": _numeric(payload["accepted_candidate_count"]),
        "emitted_candidates": _numeric(payload["emitted_candidate_count"]),
        "semantic_novel": _numeric(payload["semantic_novel_candidate_count"]),
        "semantic_clusters": _numeric(sketch["semantic_cluster_count"]),
        "demo_exact_verified": _numeric(sketch["demo_exact_verified_count"]),
        "action_count": float(sum(status_counts.values())),
        "ok_actions": float(status_counts["ok"]),
        "abstained_actions": float(status_counts["abstained"]),
        "error_actions": float(status_counts["error"]),
        "proposal_actions": float(kind_counts["propose"]),
        "repair_actions": float(kind_counts["repair"]),
    }
    for key, value in budget.items():
        numeric[f"budget.{key}"] = _numeric(value)
    raw_native = payload.get("native_cost_vector", {})
    if not isinstance(raw_native, Mapping):
        raise TypeError("native_cost_vector must be a mapping")
    native = {str(key): _numeric(value) for key, value in raw_native.items()}
    return OptionOutcome(
        success=bool(metrics["pass_at_k"]),
        residual_tokens=tuple(sorted(str(item) for item in sketch["residual_tokens"])),
        phase=str(sketch["phase"]),
        numeric=tuple(sorted(numeric.items())),
        native_cost=tuple(
            sorted((key, value) for key, value in native.items() if value)
        ),
    )


def _load_records(path: Path) -> tuple[TaskRecord, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    integrity = validate_summary(payload)
    if integrity.failed_task_count:
        raise ExperimentSafetyError(f"option audit requires a complete summary: {path}")
    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError(f"summary has no tasks: {path}")
    records: list[TaskRecord] = []
    for task in tasks:
        policies = task["policies"]
        canonical = policies[CANONICAL_POLICY]
        sketch = canonical["deliberation_sketch"]
        options = tuple(
            (source, _option_outcome(policies[SOURCE_POLICIES[source]]))
            for source in SOURCES
        )
        metrics = canonical["metrics"]
        records.append(
            TaskRecord(
                task_id=str(task["task_id"]),
                task_source_sha256=str(task["task_source_sha256"]),
                blind_content_sha256=str(task["blind_content_sha256"]),
                static_tokens=tuple(
                    sorted(str(item) for item in sketch["factual_tokens"])
                ),
                options=options,
                pool_oracle_covered=bool(metrics["pool_oracle_covered"]),
                pool_selectable_oracle_covered=bool(
                    metrics["pool_selectable_oracle_covered"]
                ),
                canonical_pass=bool(metrics["pass_at_k"]),
            )
        )
    task_ids = [item.task_id for item in records]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError(f"summary contains duplicate task IDs: {path}")
    return tuple(records)


def _median_scales(records: Sequence[TaskRecord]) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for record in records:
        for source in SOURCES:
            for key, value in record.option(source).native_cost:
                if value > 0.0:
                    values[key].append(value)
    return {
        key: float(np.median(raw)) if raw else 1.0
        for key, raw in sorted(values.items())
    }


def _normalized_cost(outcome: OptionOutcome, scales: Mapping[str, float]) -> float:
    return sum(
        value / scales.get(key, max(value, 1.0)) for key, value in outcome.native_cost
    )


@dataclass(frozen=True, slots=True)
class FeatureSpace:
    static_tokens: tuple[str, ...]
    residual_tokens: tuple[str, ...]
    numeric_keys: tuple[str, ...]
    cost_keys: tuple[str, ...]
    cost_scales: tuple[tuple[str, float], ...]

    @classmethod
    def fit(cls, records: Sequence[TaskRecord]) -> "FeatureSpace":
        static = sorted({token for record in records for token in record.static_tokens})
        residual = sorted(
            {
                token
                for record in records
                for source in SOURCES
                for token in record.option(source).residual_tokens
            }
        )
        numeric = sorted(
            {
                key
                for record in records
                for source in SOURCES
                for key, _ in record.option(source).numeric
            }
        )
        scales = _median_scales(records)
        return cls(
            tuple(static),
            tuple(residual),
            tuple(numeric),
            tuple(scales),
            tuple(scales.items()),
        )

    def _task(self, record: TaskRecord) -> list[float]:
        present = set(record.static_tokens)
        return [float(token in present) for token in self.static_tokens]

    @staticmethod
    def _source(source: str) -> list[float]:
        return [float(source == candidate) for candidate in SOURCES]

    def static(self, record: TaskRecord, candidate_source: str) -> list[float]:
        return [*self._task(record), *self._source(candidate_source)]

    def dynamic(
        self,
        record: TaskRecord,
        first_source: str,
        candidate_source: str,
        *,
        include_residual: bool,
        residual_override: Sequence[str] | None = None,
    ) -> list[float]:
        outcome = record.option(first_source)
        numeric = outcome.numeric_mapping()
        cost = outcome.cost_mapping()
        scales = dict(self.cost_scales)
        residual = set(
            outcome.residual_tokens
            if residual_override is None
            else tuple(residual_override)
        )
        features = [
            *self._task(record),
            *self._source(first_source),
            *self._source(candidate_source),
            float(outcome.phase == "scope"),
            float(outcome.phase == "explore"),
            float(outcome.phase == "refine"),
            float(outcome.phase == "answer"),
        ]
        features.extend(math.log1p(numeric.get(key, 0.0)) for key in self.numeric_keys)
        features.extend(cost.get(key, 0.0) / scales[key] for key in self.cost_keys)
        if include_residual:
            features.extend(float(token in residual) for token in self.residual_tokens)
        return features


def _fit_classifier(
    kind: str, x: Sequence[Sequence[float]], y: Sequence[int], seed: int
):
    labels = set(y)
    if len(labels) < 2:
        model = DummyClassifier(strategy="constant", constant=next(iter(labels)))
    elif kind == "tree":
        model = DecisionTreeClassifier(
            max_depth=3,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=seed,
        )
    elif kind == "boosting":
        model = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=3,
            max_iter=120,
            min_samples_leaf=8,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=seed,
        )
    else:
        raise ValueError(f"unknown classifier kind: {kind}")
    model.fit(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.int64))
    return model


def _positive_probability(model: object, rows: Sequence[Sequence[float]]) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float64)
    probabilities = model.predict_proba(values)
    classes = list(model.classes_)
    if 1 not in classes:
        return np.ones(len(values)) if classes == [1] else np.zeros(len(values))
    return probabilities[:, classes.index(1)]


def _fit_models(
    records: Sequence[TaskRecord], feature_space: FeatureSpace, seed: int
) -> dict[str, object]:
    static_x: list[list[float]] = []
    static_y: list[int] = []
    dynamic_full_x: list[list[float]] = []
    dynamic_blind_x: list[list[float]] = []
    dynamic_y: list[int] = []
    for record in records:
        for source in SOURCES:
            static_x.append(feature_space.static(record, source))
            static_y.append(int(record.option(source).success))
        for first_source in SOURCES:
            for candidate_source in SOURCES:
                if candidate_source == first_source:
                    continue
                dynamic_full_x.append(
                    feature_space.dynamic(
                        record,
                        first_source,
                        candidate_source,
                        include_residual=True,
                    )
                )
                dynamic_blind_x.append(
                    feature_space.dynamic(
                        record,
                        first_source,
                        candidate_source,
                        include_residual=False,
                    )
                )
                dynamic_y.append(int(record.option(candidate_source).success))
    return {
        "static_tree": _fit_classifier("tree", static_x, static_y, seed),
        "static_boosting": _fit_classifier("boosting", static_x, static_y, seed),
        "dynamic_blind": _fit_classifier(
            "boosting", dynamic_blind_x, dynamic_y, seed + 1
        ),
        "dynamic_residual": _fit_classifier(
            "boosting", dynamic_full_x, dynamic_y, seed + 2
        ),
    }


def _static_order(
    model: object, feature_space: FeatureSpace, record: TaskRecord
) -> tuple[str, ...]:
    scores = _positive_probability(
        model, [feature_space.static(record, source) for source in SOURCES]
    )
    return tuple(
        source
        for _, source in sorted(
            zip(scores, SOURCES), key=lambda item: (-item[0], SOURCES.index(item[1]))
        )
    )


def _dynamic_sequence(
    static_model: object,
    dynamic_model: object,
    feature_space: FeatureSpace,
    record: TaskRecord,
    *,
    include_residual: bool,
    residual_override: Sequence[str] | None = None,
) -> tuple[str, str]:
    first = _static_order(static_model, feature_space, record)[0]
    remaining = tuple(source for source in SOURCES if source != first)
    rows = [
        feature_space.dynamic(
            record,
            first,
            source,
            include_residual=include_residual,
            residual_override=residual_override,
        )
        for source in remaining
    ]
    scores = _positive_probability(dynamic_model, rows)
    second = min(
        zip(scores, remaining),
        key=lambda item: (-item[0], SOURCES.index(item[1])),
    )[1]
    return first, second


def _sequence_success(record: TaskRecord, sequence: Sequence[str]) -> bool:
    return any(record.option(source).success for source in sequence)


def _sequence_cost(
    record: TaskRecord,
    sequence: Sequence[str],
    scales: Mapping[str, float],
) -> tuple[float, dict[str, float]]:
    normalized = 0.0
    raw: dict[str, float] = defaultdict(float)
    for source in sequence:
        outcome = record.option(source)
        normalized += _normalized_cost(outcome, scales)
        for key, value in outcome.native_cost:
            raw[key] += value
    return normalized, dict(raw)


def _aggregate_policy(
    records: Sequence[TaskRecord],
    sequences: Mapping[str, Sequence[str]],
    scales: Mapping[str, float],
) -> dict[str, object]:
    solved: list[bool] = []
    normalized_cost = 0.0
    native: dict[str, float] = defaultdict(float)
    sequence_counts: Counter[tuple[str, ...]] = Counter()
    for record in records:
        sequence = tuple(sequences[record.task_id])
        sequence_counts[sequence] += 1
        solved.append(_sequence_success(record, sequence))
        task_cost, raw = _sequence_cost(record, sequence, scales)
        normalized_cost += task_cost
        for key, value in raw.items():
            native[key] += value
    return {
        "solved_tasks": sum(solved),
        "task_count": len(records),
        "coverage_rate": sum(solved) / len(records),
        "normalized_native_work": normalized_cost,
        "solved_per_normalized_native_work": (
            sum(solved) / normalized_cost if normalized_cost else 0.0
        ),
        "native_cost_vector": dict(sorted(native.items())),
        "sequence_counts": {
            "->".join(sequence): count
            for sequence, count in sorted(sequence_counts.items())
        },
        "task_success": {
            record.task_id: success for record, success in zip(records, solved)
        },
    }


def _best_fixed_pair(
    records: Sequence[TaskRecord], scales: Mapping[str, float]
) -> tuple[str, str]:
    ranked: list[tuple[int, float, tuple[str, str]]] = []
    for pair in itertools.combinations(SOURCES, 2):
        solved = sum(_sequence_success(record, pair) for record in records)
        cost = sum(_sequence_cost(record, pair, scales)[0] for record in records)
        ranked.append((-solved, cost, pair))
    return min(ranked)[2]


def _bootstrap_difference(
    first: Sequence[bool], second: Sequence[bool], *, seed: int, samples: int = 10000
) -> tuple[float, float]:
    if len(first) != len(second) or not first:
        raise ValueError("paired bootstrap inputs must be non-empty and equal")
    delta = np.asarray(first, dtype=np.float64) - np.asarray(second, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(delta), size=(samples, len(delta)))
    draws = delta[indices].mean(axis=1)
    low, high = np.quantile(draws, (0.025, 0.975))
    return float(low), float(high)


def _sign_test(wins: int, losses: int) -> float:
    discordant = wins + losses
    if not discordant:
        return 1.0
    tail = sum(
        math.comb(discordant, index) for index in range(min(wins, losses) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def _paired_comparison(
    first: Mapping[str, object], second: Mapping[str, object], *, seed: int
) -> dict[str, object]:
    first_success = first["task_success"]
    second_success = second["task_success"]
    task_ids = sorted(first_success)
    a = [bool(first_success[item]) for item in task_ids]
    b = [bool(second_success[item]) for item in task_ids]
    wins = sum(x and not y for x, y in zip(a, b))
    losses = sum(y and not x for x, y in zip(a, b))
    low, high = _bootstrap_difference(a, b, seed=seed)
    return {
        "wins": wins,
        "losses": losses,
        "ties": len(a) - wins - losses,
        "rate_difference": (sum(a) - sum(b)) / len(a),
        "paired_bootstrap_95_interval": [low, high],
        "two_sided_exact_sign_p": _sign_test(wins, losses),
    }


def _stratified_cmi(
    records: Sequence[TaskRecord],
    first_by_task: Mapping[str, str],
    source_cost: Mapping[str, float],
    *,
    seed: int,
    permutations: int = 1000,
) -> dict[str, object]:
    residual: list[tuple[str, ...]] = []
    best_second: list[str] = []
    condition: list[tuple[object, ...]] = []
    for record in records:
        first = first_by_task[record.task_id]
        remaining = tuple(source for source in SOURCES if source != first)
        successful = tuple(
            source for source in remaining if record.option(source).success
        )
        target = (
            min(successful, key=lambda source: (source_cost[source], source))
            if successful
            else "none"
        )
        core_facts = tuple(
            token
            for token in record.static_tokens
            if token.startswith(("shape:", "change:", "support:"))
        )
        residual.append(record.option(first).residual_tokens)
        best_second.append(target)
        condition.append((core_facts, first))
    observed = conditional_mutual_information(residual, best_second, condition)
    groups: dict[object, list[int]] = defaultdict(list)
    for index, key in enumerate(condition):
        groups[key].append(index)
    rng = random.Random(seed)
    null: list[float] = []
    for _ in range(permutations):
        shuffled = list(residual)
        for indices in groups.values():
            values = [shuffled[index] for index in indices]
            rng.shuffle(values)
            for index, value in zip(indices, values):
                shuffled[index] = value
        null.append(conditional_mutual_information(shuffled, best_second, condition))
    p_value = (1 + sum(value >= observed - 1e-12 for value in null)) / (
        permutations + 1
    )
    return {
        "conditional_mutual_information_bits": observed,
        "stratified_permutations": permutations,
        "permutation_mean_bits": float(np.mean(null)),
        "permutation_95_percentile_bits": float(np.quantile(null, 0.95)),
        "permutation_p": p_value,
        "condition": "core task facts plus first option",
        "conditioning_definition": (
            "phi_core(shape,change,support) plus first option; not full phi(T)"
        ),
        "stratum_count": len(groups),
        "stratum_size_histogram": dict(
            sorted(Counter(len(indices) for indices in groups.values()).items())
        ),
        "permutable_sample_count": sum(
            len(indices) for indices in groups.values() if len(indices) > 1
        ),
        "interpretation_limit": (
            "p-value applies only to this plug-in estimator, stratification, "
            "sample, and permutation null"
        ),
    }


def _residual_shuffle(
    records: Sequence[TaskRecord], first_by_task: Mapping[str, str]
) -> dict[str, tuple[str, ...]]:
    groups: dict[str, list[TaskRecord]] = defaultdict(list)
    for record in records:
        groups[first_by_task[record.task_id]].append(record)
    shuffled: dict[str, tuple[str, ...]] = {}
    for source, group in groups.items():
        ordered = sorted(group, key=lambda item: item.task_id)
        values = [record.option(source).residual_tokens for record in ordered]
        rotated = values[1:] + values[:1]
        for record, residual in zip(ordered, rotated):
            shuffled[record.task_id] = residual
    return shuffled


def _dataset_diagnostics(records: Sequence[TaskRecord]) -> dict[str, object]:
    return {
        "task_count": len(records),
        "task_ids_sha256": _sha256(sorted(item.task_id for item in records)),
        "task_source_sha256_set": _sha256(
            sorted(item.task_source_sha256 for item in records)
        ),
        "blind_content_sha256_set": _sha256(
            sorted(item.blind_content_sha256 for item in records)
        ),
        "heterogeneous_pool_oracle_covered_tasks": sum(
            item.pool_oracle_covered for item in records
        ),
        "heterogeneous_pool_selectable_oracle_covered_tasks": sum(
            item.pool_selectable_oracle_covered for item in records
        ),
        "canonical_pass_tasks": sum(item.canonical_pass for item in records),
        "source_success_tasks": {
            source: sum(item.option(source).success for item in records)
            for source in SOURCES
        },
        "three_source_union_tasks": sum(
            any(item.option(source).success for source in SOURCES) for item in records
        ),
    }


def run_audit(
    fit_records: Sequence[TaskRecord],
    test_records: Sequence[TaskRecord],
    *,
    seed: int,
) -> dict[str, object]:
    fit_fingerprints = tuple(
        TaskFingerprint(
            item.task_id, item.task_source_sha256, item.blind_content_sha256
        )
        for item in fit_records
    )
    test_fingerprints = tuple(
        TaskFingerprint(
            item.task_id, item.task_source_sha256, item.blind_content_sha256
        )
        for item in test_records
    )
    assert_three_axis_disjoint(fit_fingerprints, test_fingerprints)
    feature_space = FeatureSpace.fit(fit_records)
    scales = dict(feature_space.cost_scales)
    models = _fit_models(fit_records, feature_space, seed)

    fixed_pair = _best_fixed_pair(fit_records, scales)
    sequences: dict[str, dict[str, tuple[str, ...]]] = {
        "best_fixed_pair": {item.task_id: fixed_pair for item in test_records},
        "static_tree": {
            item.task_id: _static_order(models["static_tree"], feature_space, item)[:2]
            for item in test_records
        },
        "static_boosting": {
            item.task_id: _static_order(models["static_boosting"], feature_space, item)[
                :2
            ]
            for item in test_records
        },
        "dynamic_history": {
            item.task_id: _dynamic_sequence(
                models["static_boosting"],
                models["dynamic_blind"],
                feature_space,
                item,
                include_residual=False,
            )
            for item in test_records
        },
        "dynamic_residual": {
            item.task_id: _dynamic_sequence(
                models["static_boosting"],
                models["dynamic_residual"],
                feature_space,
                item,
                include_residual=True,
            )
            for item in test_records
        },
        "residual_ablated": {
            item.task_id: _dynamic_sequence(
                models["static_boosting"],
                models["dynamic_residual"],
                feature_space,
                item,
                include_residual=True,
                residual_override=(),
            )
            for item in test_records
        },
    }
    first_by_task = {
        task_id: sequence[0]
        for task_id, sequence in sequences["dynamic_residual"].items()
    }
    shuffled_residual = _residual_shuffle(test_records, first_by_task)
    sequences["residual_shuffled"] = {
        item.task_id: _dynamic_sequence(
            models["static_boosting"],
            models["dynamic_residual"],
            feature_space,
            item,
            include_residual=True,
            residual_override=shuffled_residual[item.task_id],
        )
        for item in test_records
    }
    sequences["deterministic_random_pair"] = {}
    for item in test_records:
        ranked = sorted(
            SOURCES,
            key=lambda source: hashlib.sha256(
                f"{seed}:{item.task_id}:{source}".encode("ascii")
            ).hexdigest(),
        )
        sequences["deterministic_random_pair"][item.task_id] = tuple(ranked[:2])

    aggregates = {
        name: _aggregate_policy(test_records, value, scales)
        for name, value in sequences.items()
    }
    oracle_sequences = {
        item.task_id: tuple(
            sorted(
                SOURCES,
                key=lambda source: (
                    not item.option(source).success,
                    _normalized_cost(item.option(source), scales),
                    source,
                ),
            )[:2]
        )
        for item in test_records
    }
    aggregates["oracle_source_scheduler"] = _aggregate_policy(
        test_records, oracle_sequences, scales
    )

    source_cost = {
        source: float(
            np.mean(
                [_normalized_cost(item.option(source), scales) for item in fit_records]
            )
        )
        for source in SOURCES
    }
    residual_effect = {
        comparison: sum(
            sequences["dynamic_residual"][item.task_id]
            != sequences[comparison][item.task_id]
            for item in test_records
        )
        for comparison in (
            "dynamic_history",
            "residual_ablated",
            "residual_shuffled",
        )
    }
    dynamic_vs_history = _paired_comparison(
        aggregates["dynamic_residual"], aggregates["dynamic_history"], seed=seed
    )
    dynamic_vs_ablated = _paired_comparison(
        aggregates["dynamic_residual"],
        aggregates["residual_ablated"],
        seed=seed + 1,
    )
    dynamic_vs_static = _paired_comparison(
        aggregates["dynamic_residual"], aggregates["static_boosting"], seed=seed + 2
    )
    residual_gain = dynamic_vs_ablated["rate_difference"]
    if residual_effect["residual_ablated"] == 0:
        gate = "residual_behaviorally_inert"
    elif residual_gain <= 0.0:
        gate = "residual_changes_actions_without_positive_coverage_gain"
    elif residual_gain < 0.03:
        gate = "positive_but_below_three_point_development_gate"
    elif dynamic_vs_ablated["paired_bootstrap_95_interval"][0] <= 0.0:
        gate = "exploratory_gain_with_interval_crossing_zero"
    else:
        gate = "development_gate_passed_requires_200_task_physical_budget_replication"

    return {
        "schema": SCHEMA_VERSION,
        "training_started": True,
        "neural_provider_training_started": False,
        "offline_controller_fitting_performed": True,
        "oracle_access": "posthoc source-success labels on disjoint fit block only",
        "fit_label_provenance": (
            "posthoc per-source success labels; query oracle grids are never features"
        ),
        "offline_controller_fitting": {
            "seed": seed,
            "fit_task_count": len(fit_records),
            "static_training_rows": len(fit_records) * len(SOURCES),
            "dynamic_training_rows": len(fit_records)
            * len(SOURCES)
            * (len(SOURCES) - 1),
            "estimators": {
                "static_tree": "DecisionTreeClassifier(max_depth=3,min_samples_leaf=5,class_weight=balanced)",
                "static_boosting": "HistGradientBoostingClassifier(max_depth=3,max_iter=120)",
                "dynamic_history": "HistGradientBoostingClassifier(max_depth=3,max_iter=120)",
                "dynamic_residual": "HistGradientBoostingClassifier(max_depth=3,max_iter=120)",
            },
        },
        "claim_scope": (
            "development option-level residual-increment audit; two option slots; "
            "not a physical-compute-matched end-to-end ARC claim"
        ),
        "fit": _dataset_diagnostics(fit_records),
        "test": _dataset_diagnostics(test_records),
        "task_sets_disjoint": True,
        "task_set_disjointness_axes": [
            "task_id",
            "task_source_sha256",
            "blind_content_sha256",
        ],
        "features": {
            "static_token_count": len(feature_space.static_tokens),
            "residual_token_count": len(feature_space.residual_tokens),
            "numeric_outcome_feature_count": len(feature_space.numeric_keys),
            "native_cost_feature_count": len(feature_space.cost_keys),
            "query_oracle_features": 0,
        },
        "native_cost_normalization": {
            "rule": "divide each named dimension by fit-block positive median",
            "scales": scales,
            "warning": "descriptive scalarization only; raw vectors are authoritative",
        },
        "selected_fit_fixed_pair": list(fixed_pair),
        "source_mean_normalized_fit_cost": source_cost,
        "policies": aggregates,
        "residual_action_intervention": residual_effect,
        "residual_increment_vs_history": dynamic_vs_history,
        "residual_causal_effect_vs_same_model_ablation": dynamic_vs_ablated,
        "dynamic_vs_static_boosting": dynamic_vs_static,
        "conditional_information": _stratified_cmi(
            test_records, first_by_task, source_cost, seed=seed
        ),
        "decision_gate": gate,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fit_summary", type=Path)
    parser.add_argument("test_summary", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument(
        "--allow-dirty-source",
        action="store_true",
        help="allow a noncanonical diagnostic audit from modified source",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    source_start = capture_git_source_provenance(
        REPOSITORY_ROOT, paths=ARC_EXPERIMENT_SOURCE_PATHS
    )
    if source_start.dirty and not args.allow_dirty_source:
        raise ExperimentSafetyError(
            "refusing audit from a dirty source tree; commit first or use "
            "--allow-dirty-source for a noncanonical diagnostic"
        )
    fit_path = args.fit_summary.resolve()
    test_path = args.test_summary.resolve()
    fit_summary = json.loads(fit_path.read_text(encoding="utf-8"))
    test_summary = json.loads(test_path.read_text(encoding="utf-8"))
    input_integrity = {
        "fit": validate_summary(fit_summary),
        "test": validate_summary(test_summary),
    }
    input_source_blockers: list[str] = []
    for role, summary in (("fit", fit_summary), ("test", test_summary)):
        try:
            validate_clean_source_binding(summary)
        except ExperimentSafetyError:
            input_source_blockers.append(f"{role}_summary_source_binding_unavailable")
    fit_records = _load_records(fit_path)
    test_records = _load_records(test_path)
    payload: dict[str, Any] = run_audit(fit_records, test_records, seed=args.seed)
    source_end = capture_git_source_provenance(
        REPOSITORY_ROOT, paths=ARC_EXPERIMENT_SOURCE_PATHS
    )
    verify_source_provenance_unchanged(source_start, source_end)
    payload["inputs"] = {
        "fit_summary": str(fit_path),
        "fit_summary_sha256": file_sha256(fit_path),
        "fit_summary_result_id": input_integrity["fit"].result_id,
        "test_summary": str(test_path),
        "test_summary_sha256": file_sha256(test_path),
        "test_summary_result_id": input_integrity["test"].result_id,
        "seed": args.seed,
    }
    payload["source_commit"] = source_start.head
    payload["source_provenance"] = source_start.to_json_dict()
    payload["runtime"] = runtime_metadata(("numpy", "scikit-learn"))
    publication_blockers = list(input_source_blockers)
    if source_start.dirty:
        publication_blockers.append("dirty_source_override")
    payload["publication_eligibility"] = {
        "eligible": not publication_blockers,
        "blockers": sorted(publication_blockers),
    }
    payload["result_id"] = _sha256(payload)
    atomic_write_json(args.output.resolve(), payload)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "result_id": payload["result_id"],
                "decision_gate": payload["decision_gate"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
