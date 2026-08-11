"""Freeze and score query-blind VARC posterior candidates."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = str((PROJECT_ROOT / "src").resolve())
if SOURCE_ROOT not in sys.path:
    sys.path.insert(0, SOURCE_ROOT)

from afts_arc.experiment_safety import atomic_write_json, file_sha256  # noqa: E402
from afts_arc.varc_candidates import (  # noqa: E402
    freeze_varc_predictions,
    score_varc_freeze,
)


def _load_object(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"JSON artifact must be an object: {path}")
    return payload


def _write_new(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to replace artifact: {path}")
    atomic_write_json(path, payload)


def _freeze(arguments: argparse.Namespace) -> None:
    challenges = Path(arguments.challenges).resolve()
    blind_manifest = Path(arguments.blind_manifest).resolve()
    prediction_root = Path(arguments.prediction_root).resolve()
    provider_contract = Path(arguments.provider_contract).resolve()
    output = Path(arguments.output).resolve()
    manifest = _load_object(blind_manifest)
    task_records = manifest["tasks"]
    if not isinstance(task_records, list):
        raise TypeError("blind manifest tasks must be an array")
    task_ids = [record["task_id"] for record in task_records]
    expected_files = {f"{task_id}_predictions.json" for task_id in task_ids}
    actual_files = {path.name for path in prediction_root.glob("*_predictions.json")}
    if actual_files != expected_files:
        raise ValueError("prediction-root file set differs from blind manifest")
    predictions: dict[str, object] = {}
    prediction_hashes: dict[str, str] = {}
    for task_id in task_ids:
        path = prediction_root / f"{task_id}_predictions.json"
        predictions[task_id] = _load_object(path)
        prediction_hashes[f"prediction:{task_id}"] = file_sha256(path)
    source_files = {
        "blind_manifest": file_sha256(blind_manifest),
        "challenges": file_sha256(challenges),
        "provider_contract": file_sha256(provider_contract),
        **prediction_hashes,
    }
    payload = freeze_varc_predictions(
        challenges=_load_object(challenges),
        blind_manifest=manifest,
        predictions=predictions,
        provider_contract=_load_object(provider_contract),
        source_files=source_files,
    )
    _write_new(output, payload)
    print(json.dumps({"freeze_id": payload["freeze_id"], "output": str(output)}))


def _score(arguments: argparse.Namespace) -> None:
    candidate_freeze = Path(arguments.candidate_freeze).resolve()
    solutions = Path(arguments.solutions).resolve()
    output = Path(arguments.output).resolve()
    payload = score_varc_freeze(
        freeze=_load_object(candidate_freeze),
        solutions=_load_object(solutions),
        solution_source_sha256=file_sha256(solutions),
    )
    _write_new(output, payload)
    print(json.dumps({"result_id": payload["result_id"], "output": str(output)}))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--challenges", required=True)
    freeze.add_argument("--blind-manifest", required=True)
    freeze.add_argument("--prediction-root", required=True)
    freeze.add_argument("--provider-contract", required=True)
    freeze.add_argument("--output", required=True)
    freeze.set_defaults(handler=_freeze)
    score = subparsers.add_parser("score")
    score.add_argument("--candidate-freeze", required=True)
    score.add_argument("--solutions", required=True)
    score.add_argument("--output", required=True)
    score.set_defaults(handler=_score)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    arguments.handler(arguments)


if __name__ == "__main__":
    main()
