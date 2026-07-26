#!/usr/bin/env python3
"""Calibrate pre-action native reservations from a fit-block frozen pool."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from afts_arc.experiment_safety import (
    ARC_EXPERIMENT_SOURCE_PATHS,
    ExperimentSafetyError,
    GitSourceProvenance,
    atomic_write_json,
    canonical_sha256,
    capture_git_source_provenance,
    file_sha256,
    runtime_metadata,
    validate_pool_manifest,
    validate_clean_source_binding,
    validate_summary,
    verify_source_provenance_unchanged,
)
from afts_arc.hybrid.metareasoning import (
    NativeCostContract,
    NativeCostReservation,
    NativeCostVector,
)


SCHEMA_VERSION = "afts.native-budget-profile/v2"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
EXPECTED_ACTIONS = {
    "typed_dsl": ("synthesize", "shape_resynthesize", "suffix_resynthesize"),
    "sparse_ca_d4_bgpad": ("d4_bgpad_search", "local_transition_search"),
    "scene_predicate_dsl": ("synthesize",),
    "residual_repair": ("global_color_map", "local_transition"),
}


def _sha256(value: object) -> str:
    return canonical_sha256(value)


def _file_sha256(path: Path) -> str:
    return file_sha256(path)


def _higher_quantile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    return float(np.quantile(values, quantile, method="higher"))


def _cost_mapping(
    value: object,
    *,
    prefix: str,
    legacy_conversions: dict[str, int],
) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise TypeError("native cost diagnostics must be a mapping")

    def numeric_only(
        raw: Mapping[str, object], path: tuple[str, ...] = ()
    ) -> dict[str, object]:
        cleaned: dict[str, object] = {}
        for key, item in raw.items():
            text_key = str(key)
            item_path = (*path, text_key)
            if isinstance(item, Mapping):
                nested = numeric_only(item, item_path)
                if nested:
                    cleaned[text_key] = nested
            elif isinstance(item, (int, float)) and not isinstance(item, bool):
                cleaned[text_key] = item
            elif item_path == ("bounded_program_trials_enabled",) and isinstance(
                item, bool
            ):
                # The v3 legacy pool schema placed this one descriptive flag
                # under native_cost.  It was never a cost dimension.
                legacy_key = ".".join((*prefix.split("."), *item_path))
                legacy_conversions[legacy_key] = (
                    legacy_conversions.get(legacy_key, 0) + 1
                )
            else:
                raise TypeError(
                    "native cost leaves must be numeric; unsupported leaf at "
                    + ".".join(item_path)
                )
        return cleaned

    return NativeCostVector.from_mapping(
        numeric_only(value), prefix=prefix
    ).to_mapping()


def _calibrated_vector(
    observations: Sequence[Mapping[str, float]],
    *,
    actor: str,
    quantile: float,
) -> NativeCostVector:
    keys = sorted({key for item in observations for key in item})
    reserved = {
        key: _higher_quantile([item.get(key, 0.0) for item in observations], quantile)
        for key in keys
    }
    reserved[f"{actor}.native_cost.option_calls"] = 1.0
    return NativeCostVector(tuple(reserved.items()))


def calibrate(
    summary_path: Path,
    *,
    quantile: float,
    source_provenance: GitSourceProvenance,
    runtime: Mapping[str, object],
) -> dict[str, object]:
    if not 0.0 < quantile <= 1.0 or not math.isfinite(quantile):
        raise ValueError("quantile must be finite and in (0, 1]")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    integrity = validate_summary(summary)
    input_source = validate_clean_source_binding(summary)
    if integrity.failed_task_count:
        raise ExperimentSafetyError("fit summary must be complete with no failures")
    tasks = summary["tasks"]
    if not tasks:
        raise ValueError("fit summary contains no tasks")
    task_ids = [str(item["task_id"]) for item in tasks]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("fit summary contains duplicate task IDs")

    observations: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    legacy_conversions: dict[str, int] = {}
    fingerprints = {item.task_id: item for item in integrity.task_fingerprints}
    summary_root = summary_path.parent.resolve()
    for task in tasks:
        manifest_path = (summary_root / str(task["pool_manifest"])).resolve()
        if not manifest_path.is_relative_to(summary_root):
            raise ExperimentSafetyError("pool manifest escapes summary directory")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_integrity = validate_pool_manifest(
            manifest, expected_task=fingerprints[str(task["task_id"])]
        )
        if manifest_integrity.pool_id != task.get("pool_id"):
            raise ExperimentSafetyError("summary/pool manifest ID mismatch")
        for provider in manifest["providers"]:
            actor = str(provider["provider"])
            for action in provider["actions"]:
                operator = str(action["operator"])
                diagnostics = action.get("diagnostics", {})
                if not isinstance(diagnostics, Mapping):
                    raise TypeError("provider diagnostics must be a mapping")
                if "native_cost" not in diagnostics:
                    raise ExperimentSafetyError(
                        f"missing actual native cost for {actor}:{operator}"
                    )
                native = diagnostics["native_cost"]
                observations[(actor, operator)].append(
                    _cost_mapping(
                        native,
                        prefix=f"{actor}.native_cost",
                        legacy_conversions=legacy_conversions,
                    )
                )

    repair_cost = {
        "residual_repair.native_cost.repair_attempts": 1.0,
    }
    for operator in EXPECTED_ACTIONS["residual_repair"]:
        observations[("residual_repair", operator)].append(repair_cost)

    actor_fallback: dict[str, list[dict[str, float]]] = defaultdict(list)
    for (actor, _), values in observations.items():
        actor_fallback[actor].extend(values)

    reservations: list[NativeCostReservation] = []
    support: dict[str, dict[str, object]] = {}
    for actor, operators in EXPECTED_ACTIONS.items():
        for operator in operators:
            values = observations.get((actor, operator), ())
            used_fallback = not values
            calibration_values = values or actor_fallback.get(actor, ({},))
            reservation = NativeCostReservation(
                actor,
                operator,
                _calibrated_vector(
                    calibration_values,
                    actor=actor,
                    quantile=quantile,
                ),
            )
            reservations.append(reservation)
            support[f"{actor}:{operator}"] = {
                "observed_action_outcomes": len(values),
                "used_actor_pooled_observed_fallback": used_fallback,
            }
    contract = NativeCostContract(tuple(reservations))

    limit_values: dict[str, float] = defaultdict(float)
    for reservation in contract.reservations:
        for key, value in reservation.cost.items:
            limit_values[key] = max(limit_values[key], value)
    limit = NativeCostVector(tuple(limit_values.items()))
    payload: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "source_commit": source_provenance.head,
        "source_provenance": source_provenance.to_json_dict(),
        "runtime": dict(runtime),
        "publication_eligibility": {
            "eligible": not source_provenance.dirty,
            "blockers": (
                [] if not source_provenance.dirty else ["dirty_source_override"]
            ),
        },
        "fit_summary_sha256": _file_sha256(summary_path),
        "fit_summary_result_id": integrity.result_id,
        "fit_summary_source_commit": input_source.head,
        "fit_task_count": len(task_ids),
        "fit_task_ids_sha256": _sha256(sorted(task_ids)),
        "fit_task_fingerprints": [
            item.to_json_dict() for item in sorted(integrity.task_fingerprints)
        ],
        "oracle_fields_read": 0,
        "reservation_rule": {
            "estimator": "dimensionwise_empirical_higher_quantile",
            "quantile": quantile,
            "abstentions_are_included": True,
            "unseen_operator": "actor_pooled_observed_higher_quantile_fallback",
        },
        "budget_rule": (
            "one calibrated option-call token per actor; per-dimension cap is "
            "the maximum reservation across that actor's operators"
        ),
        "contract": contract.to_json_dict(),
        "budget_limit": limit.to_json_dict(),
        "support": support,
        "legacy_cost_conversions": dict(sorted(legacy_conversions.items())),
    }
    payload["profile_id"] = _sha256(payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fit_summary", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--quantile", type=float, default=1.0)
    parser.add_argument(
        "--allow-dirty-source",
        action="store_true",
        help="allow a noncanonical diagnostic profile from modified source",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    source_start = capture_git_source_provenance(
        REPOSITORY_ROOT, paths=ARC_EXPERIMENT_SOURCE_PATHS
    )
    if source_start.dirty and not args.allow_dirty_source:
        raise ExperimentSafetyError(
            "refusing calibration from a dirty source tree; commit first or use "
            "--allow-dirty-source for a noncanonical diagnostic"
        )
    summary = args.fit_summary.resolve()
    payload = calibrate(
        summary,
        quantile=args.quantile,
        source_provenance=source_start,
        runtime=runtime_metadata(("numpy",)),
    )
    source_end = capture_git_source_provenance(
        REPOSITORY_ROOT, paths=ARC_EXPERIMENT_SOURCE_PATHS
    )
    verify_source_provenance_unchanged(source_start, source_end)
    output = args.output.resolve()
    registry_root = (
        output.parent
        if output.parent.name == "native_budget_profiles"
        else output.parent / "native_budget_profiles"
    )
    registry_path = registry_root / f"{payload['profile_id']}.json"
    atomic_write_json(registry_path, payload)
    atomic_write_json(output, payload)
    print(
        json.dumps(
            {
                "output": str(output),
                "content_addressed_profile": str(registry_path),
                "profile_id": payload["profile_id"],
                "fit_task_count": payload["fit_task_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
