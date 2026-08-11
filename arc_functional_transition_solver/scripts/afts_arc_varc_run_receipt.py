"""Close an immutable receipt for a completed query-blind VARC run."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from collections.abc import Mapping
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = str((PROJECT_ROOT / "src").resolve())
if SOURCE_ROOT not in sys.path:
    sys.path.insert(0, SOURCE_ROOT)

from afts_arc.experiment_safety import atomic_write_json, file_sha256  # noqa: E402
from afts_arc.varc_run import build_varc_run_receipt  # noqa: E402


def _load_object(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"JSON artifact must be an object: {path}")
    return payload


def _pip_versions(path: Path) -> dict[str, str]:
    selected = {"diffusers", "einops", "timm", "torch", "torchvision"}
    versions: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "==" not in line:
            continue
        name, version = line.split("==", 1)
        normalized = name.lower()
        if normalized in selected:
            versions[normalized] = version
    if set(versions) != selected:
        raise ValueError("pip freeze does not close the required VARC packages")
    return dict(sorted(versions.items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blind-manifest", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--prediction-root", required=True)
    parser.add_argument("--varc-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()

    blind_manifest_path = Path(arguments.blind_manifest).resolve()
    run_dir = Path(arguments.run_dir).resolve()
    prediction_root = Path(arguments.prediction_root).resolve()
    varc_root = Path(arguments.varc_root).resolve()
    checkpoint = Path(arguments.checkpoint).resolve()
    output = Path(arguments.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace artifact: {output}")
    if (run_dir / "exit_code.txt").read_text(encoding="utf-8").strip() != "0":
        raise ValueError("VARC cohort launcher did not exit successfully")
    source_commit = (run_dir / "source_commit.txt").read_text(encoding="utf-8").strip()
    if source_commit != arguments.expected_source_commit:
        raise ValueError("VARC source commit differs from the frozen contract")
    checkpoint_sha256 = file_sha256(checkpoint)
    if checkpoint_sha256 != arguments.expected_checkpoint_sha256:
        raise ValueError("VARC checkpoint differs from the frozen contract")
    gpu_ids = tuple(
        int(line)
        for line in (run_dir / "gpu_ids.txt").read_text(encoding="utf-8").splitlines()
    )
    blind_manifest = _load_object(blind_manifest_path)
    compatibility_manifest = _load_object(
        run_dir / "task_compatibility.json"
    )
    records = blind_manifest["tasks"]
    if not isinstance(records, list):
        raise TypeError("blind manifest tasks must be an array")
    task_ids = [record["task_id"] for record in records]
    statuses: dict[str, object] = {}
    predictions: dict[str, object] = {}
    for task_id in task_ids:
        statuses[task_id] = _load_object(run_dir / "tasks" / task_id / "status.json")
        prediction_path = prediction_root / f"{task_id}_predictions.json"
        predictions[task_id] = {
            "bytes": prediction_path.stat().st_size,
            "sha256": file_sha256(prediction_path),
        }
    executed_files = (
        "src/ARC_ViT.py",
        "src/ARC_loader.py",
        "test_time_train_ARC.py",
        "utils/analyze_prediction.py",
        "utils/args.py",
        "utils/distribution.py",
        "utils/eval_utils_ttt.py",
        "utils/load_model.py",
    )
    source = {
        "blind_manifest_sha256": file_sha256(blind_manifest_path),
        "checkpoint_sha256": checkpoint_sha256,
        "executed_source_sha256": {
            relative: file_sha256(varc_root / relative) for relative in executed_files
        },
        "source_commit": source_commit,
        "source_hash_manifest_sha256": file_sha256(
            run_dir / "source_hashes.sha256"
        ),
        "diagnostic_compatibility_sha256": file_sha256(
            run_dir / "task_compatibility.json"
        ),
        "diagnostic_records_sha256": {
            task_id: file_sha256(
                run_dir / "tasks" / task_id / "diagnostic_alias.json"
            )
            for task_id in task_ids
        },
    }
    runtime = {
        "packages": _pip_versions(run_dir / "pip_freeze.txt"),
        "platform": platform.platform(),
        "python": (run_dir / "python_version.txt").read_text(encoding="utf-8").strip(),
    }
    receipt = build_varc_run_receipt(
        blind_manifest=blind_manifest,
        task_statuses=statuses,
        prediction_files=predictions,
        source=source,
        runtime=runtime,
        gpu_ids=gpu_ids,
        compatibility_manifest=compatibility_manifest,
    )
    atomic_write_json(output, receipt)
    print(json.dumps({"output": str(output), "receipt_id": receipt["receipt_id"]}))


if __name__ == "__main__":
    main()
