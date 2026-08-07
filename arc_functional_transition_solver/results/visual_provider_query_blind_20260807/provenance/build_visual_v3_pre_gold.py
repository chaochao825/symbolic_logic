"""Build the confirmatory VARC provider contract and pre-gold freeze receipt."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/home/wangmeiqi/codex_runs/arc_visual_provider_probe_20260807")
TRANSFER_ROOT = ROOT / "from236_v3_confirm"
PREDICTION_ROOT = (
    TRANSFER_ROOT
    / "external"
    / "VARC"
    / "outputs"
    / "query_blind_confirm_v3_attempt_0"
)


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def write_new_json(path: Path, payload: object) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise FileExistsError(f"refusing to replace different artifact: {path}")
        return
    path.write_text(encoded, encoding="utf-8")


def main() -> None:
    template_path = ROOT / "provider_contract_v2_1.json"
    validation_path = TRANSFER_ROOT / "provider_raw_validation_v3_confirm.json"
    manifest_path = ROOT / "cohort_v3_confirm" / "manifest.json"
    registry_path = ROOT / "exposure_registry_v3_confirm.json"
    package_path = ROOT / "visual_provider_raw_v3_confirm_compact_from236.tar.gz"

    template = load_json(template_path)
    template_body = dict(template)
    del template_body["contract_id"]
    if canonical_sha256(template_body) != template["contract_id"]:
        raise ValueError("v2.1 provider contract content ID is invalid")

    validation = load_json(validation_path)
    manifest = load_json(manifest_path)
    if validation["gold_accessed"] is not False:
        raise ValueError("raw validation is not pre-gold")
    if validation["cohort_id"] != manifest["cohort_id"]:
        raise ValueError("validation and cohort IDs differ")
    if validation["cohort_manifest_sha256"] != file_sha256(manifest_path):
        raise ValueError("validation does not bind the transferred cohort manifest")
    if validation["exposure_registry_sha256"] != file_sha256(registry_path):
        raise ValueError("validation does not bind the exposure registry")

    raw = validation["predictions"]
    run = validation["run"]
    audit = validation["query_blind_audit"]
    rejection_reasons = raw["rejection_reasons"]
    prediction_files = raw["files"]
    if run["failure_count"] != 0 or run["task_count"] != len(prediction_files):
        raise ValueError("provider run is incomplete")

    transferred_files: list[dict[str, object]] = []
    for record in prediction_files:
        task_id = record["task_id"]
        path = PREDICTION_ROOT / f"{task_id}_predictions.json"
        if file_sha256(path) != record["sha256"]:
            raise ValueError(f"prediction hash mismatch: {task_id}")
        if path.stat().st_size != record["size_bytes"]:
            raise ValueError(f"prediction size mismatch: {task_id}")
        transferred_files.append(
            {
                "path": str(path),
                "sha256": record["sha256"],
                "size_bytes": record["size_bytes"],
                "task_id": task_id,
            }
        )

    contract = template_body
    contract["name"] = "VARC ViT static query-blind provider confirm v3"
    contract["cohort"] = {
        "cohort_id": manifest["cohort_id"],
        "exposure_registry_id": validation["exposure_registry_id"],
        "exposure_registry_sha256": file_sha256(registry_path),
        "manifest_sha256": file_sha256(manifest_path),
        "task_count": len(prediction_files),
    }
    contract["compute"] = {
        "gpu_launch_inventory": [
            "0, NVIDIA GeForce RTX 4090, 24564, 15, 0, P8",
            "1, NVIDIA GeForce RTX 4090, 24564, 15, 0, P8",
            "2, NVIDIA H200 NVL, 143771, 14, 0, P0",
            "3, NVIDIA H200 NVL, 143771, 14, 0, P0",
        ],
        "gpu_seconds": run["gpu_seconds"],
        "max_task_seconds": run["max_task_seconds"],
        "task_failure_count": run["failure_count"],
    }
    contract["candidate_contract"] = {
        "invalid_candidate_amendment": (
            "v2 strict protocol invalidated before scoring; v2.1 reject rule reused "
            "unchanged for confirmatory v3"
        ),
        "invalid_candidate_policy": "reject",
        "invalid_candidate_rule": (
            "reject non-list, empty, non-rectangular, >30x30, non-integer, or "
            "color-outside-[0,9] samples; never repair or coerce"
        ),
        "query_requires_at_least_one_valid_candidate": True,
        "ranking": "frequency, then first-emission position, then canonical grid JSON",
        "raw_candidate_count": raw["raw_candidate_count"],
        "raw_files_frozen_before_gold": True,
        "rejected_candidate_count": raw["rejected_candidate_count"],
        "rejection_reasons": rejection_reasons,
        "valid_candidate_count": raw["valid_candidate_count"],
    }
    contract["query_blind_protocol"] = {
        "gold_output_present_in_provider_root": False,
        "provider_visible_test_output": "exact copy of corresponding test input",
        "released_loader_repair": (
            "sentinel preserves the released preprocessing interface without carrying "
            "a query label; this is not an exact paper-script replication"
        ),
        "sentinel_file_count": audit["provider_json_file_count"],
        "sentinel_mismatch_count": audit["sentinel_mismatch_count"],
        "sentinel_query_count": audit["sentinel_query_count"],
    }
    contract["raw_artifact"] = {
        "transfer_package_sha256": file_sha256(package_path),
        "validation_id": validation["validation_id"],
        "validation_sha256": file_sha256(validation_path),
    }
    contract["contract_id"] = canonical_sha256(contract)
    contract_path = ROOT / "provider_contract_v3_confirm.json"
    write_new_json(contract_path, contract)

    receipt = {
        "cohort_id": manifest["cohort_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "gold_scoring_started": False,
        "prediction_files": transferred_files,
        "provider_contract_id": contract["contract_id"],
        "provider_contract_sha256": file_sha256(contract_path),
        "raw_validation_id": validation["validation_id"],
        "raw_validation_sha256": file_sha256(validation_path),
        "schema": "afts.visual-provider-pre-gold-freeze/v1",
        "transfer_package_sha256": file_sha256(package_path),
    }
    receipt["freeze_id"] = canonical_sha256(receipt)
    receipt_path = ROOT / "pre_gold_freeze_receipt_v3_confirm.json"
    write_new_json(receipt_path, receipt)

    print(
        json.dumps(
            {
                "contract_id": contract["contract_id"],
                "contract_sha256": file_sha256(contract_path),
                "freeze_id": receipt["freeze_id"],
                "freeze_sha256": file_sha256(receipt_path),
                "prediction_file_count": len(transferred_files),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
