#!/usr/bin/env python3
"""Calibrate pre-action native reservations from a fit-block frozen pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from afts_arc.hybrid.metareasoning import (
    NativeCostContract,
    NativeCostReservation,
    NativeCostVector,
)


SCHEMA_VERSION = "afts.native-budget-profile/v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ACTIONS = {
    "typed_dsl": ("synthesize", "shape_resynthesize", "suffix_resynthesize"),
    "sparse_ca_d4_bgpad": ("d4_bgpad_search", "local_transition_search"),
    "scene_predicate_dsl": ("synthesize",),
    "residual_repair": ("global_color_map", "local_transition"),
}


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _higher_quantile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    return float(np.quantile(values, quantile, method="higher"))


def _cost_mapping(value: object, *, prefix: str) -> dict[str, float]:
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise TypeError("native cost diagnostics must be a mapping")

    def numeric_only(raw: Mapping[str, object]) -> dict[str, object]:
        cleaned: dict[str, object] = {}
        for key, item in raw.items():
            if isinstance(item, Mapping):
                nested = numeric_only(item)
                if nested:
                    cleaned[str(key)] = nested
            elif isinstance(item, (int, float)) and not isinstance(item, bool):
                cleaned[str(key)] = item
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


def calibrate(summary_path: Path, *, quantile: float) -> dict[str, object]:
    if not 0.0 < quantile <= 1.0 or not math.isfinite(quantile):
        raise ValueError("quantile must be finite and in (0, 1]")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    tasks = summary.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("fit summary contains no tasks")
    task_ids = [str(item["task_id"]) for item in tasks]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("fit summary contains duplicate task IDs")

    observations: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    for task in tasks:
        manifest_path = summary_path.parent / str(task["pool_manifest"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("task_id") != task["task_id"]:
            raise ValueError("pool manifest task ID mismatch")
        for provider in manifest["providers"]:
            actor = str(provider["provider"])
            for action in provider["actions"]:
                operator = str(action["operator"])
                diagnostics = action.get("diagnostics", {})
                native = (
                    diagnostics.get("native_cost", {})
                    if isinstance(diagnostics, Mapping)
                    else {}
                )
                observations[(actor, operator)].append(
                    _cost_mapping(
                        native,
                        prefix=f"{actor}.native_cost",
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
                "used_actor_worst_observed_fallback": used_fallback,
            }
    contract = NativeCostContract(tuple(reservations))

    limit_values: dict[str, float] = defaultdict(float)
    for reservation in contract.reservations:
        for key, value in reservation.cost.items:
            limit_values[key] = max(limit_values[key], value)
    limit = NativeCostVector(tuple(limit_values.items()))
    payload: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "source_commit": _git_head(),
        "fit_summary_sha256": _file_sha256(summary_path),
        "fit_task_count": len(task_ids),
        "fit_task_ids_sha256": _sha256(sorted(task_ids)),
        "oracle_fields_read": 0,
        "reservation_rule": {
            "estimator": "dimensionwise_empirical_higher_quantile",
            "quantile": quantile,
            "abstentions_are_included": True,
            "unseen_operator": "actor_worst_observed_fallback",
        },
        "budget_rule": (
            "one calibrated option-call token per actor; per-dimension cap is "
            "the maximum reservation across that actor's operators"
        ),
        "contract": contract.to_json_dict(),
        "budget_limit": limit.to_json_dict(),
        "support": support,
    }
    payload["profile_id"] = _sha256(payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fit_summary", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--quantile", type=float, default=1.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = args.fit_summary.resolve()
    payload = calibrate(summary, quantile=args.quantile)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "profile_id": payload["profile_id"],
                "fit_task_count": payload["fit_task_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
