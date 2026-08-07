"""Freeze all confirmatory comparison inputs before visual-provider scoring."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/home/wangmeiqi/codex_runs/arc_visual_provider_probe_20260807")
EXPECTED_SOURCE_COMMIT = "8c855bf0c2a37c135c9ac0f68fbe2e3fd677bc7c"


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


def bound_file(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": file_sha256(path)}


def main() -> None:
    baseline_path = ROOT / "baseline_v3_confirm" / "summary.json"
    object_path = ROOT / "object_code_v3_confirm" / "summary.json"
    manifest_path = ROOT / "cohort_v3_confirm" / "manifest.json"
    pre_gold_path = ROOT / "pre_gold_freeze_receipt_v3_confirm.json"
    contract_path = ROOT / "provider_contract_v3_confirm.json"
    validation_path = (
        ROOT
        / "from236_v3_confirm"
        / "provider_raw_validation_v3_confirm.json"
    )

    baseline = load_json(baseline_path)
    object_summary = load_json(object_path)
    manifest = load_json(manifest_path)
    pre_gold = load_json(pre_gold_path)
    contract = load_json(contract_path)
    validation = load_json(validation_path)

    task_ids = [record["task_id"] for record in manifest["tasks"]]
    if baseline["dataset"]["task_ids"] != task_ids:
        raise ValueError("baseline task order differs from frozen cohort")
    if object_summary["dataset"]["task_ids"] != task_ids:
        raise ValueError("object/code task order differs from frozen cohort")
    if baseline["dataset"]["failed_task_count"] != 0:
        raise ValueError("baseline run is incomplete")
    if object_summary["dataset"]["failed_task_count"] != 0:
        raise ValueError("object/code run is incomplete")
    if baseline["source_commit"] != EXPECTED_SOURCE_COMMIT:
        raise ValueError("baseline source commit differs from the frozen implementation")
    if object_summary["baseline"]["source_commit"] != EXPECTED_SOURCE_COMMIT:
        raise ValueError("object/code source commit differs from the frozen implementation")

    contract_body = dict(contract)
    del contract_body["contract_id"]
    if canonical_sha256(contract_body) != contract["contract_id"]:
        raise ValueError("provider contract content ID is invalid")
    pre_gold_body = dict(pre_gold)
    del pre_gold_body["freeze_id"]
    if canonical_sha256(pre_gold_body) != pre_gold["freeze_id"]:
        raise ValueError("pre-gold freeze content ID is invalid")
    if pre_gold["gold_scoring_started"] is not False:
        raise ValueError("visual gold scoring was already declared started")
    if validation["gold_accessed"] is not False:
        raise ValueError("provider validation is not pre-gold")

    receipt = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "gold_scoring_started": False,
        "ids": {
            "baseline_result_id": baseline["result_id"],
            "cohort_id": manifest["cohort_id"],
            "object_provider_result_id": object_summary["result_id"],
            "provider_contract_id": contract["contract_id"],
        },
        "inputs": {
            "baseline_summary": bound_file(baseline_path),
            "cohort_manifest": bound_file(manifest_path),
            "object_provider_summary": bound_file(object_path),
            "pre_gold_freeze_receipt": bound_file(pre_gold_path),
            "provider_contract": bound_file(contract_path),
            "raw_validation": bound_file(validation_path),
        },
        "schema": "afts.visual-provider-comparison-freeze/v1",
        "task_count": len(task_ids),
        "task_ids": task_ids,
    }
    receipt["freeze_id"] = canonical_sha256(receipt)
    output_path = ROOT / "comparison_freeze_receipt_v3_confirm.json"
    write_new_json(output_path, receipt)
    print(
        json.dumps(
            {
                "freeze_id": receipt["freeze_id"],
                "freeze_sha256": file_sha256(output_path),
                "task_count": len(task_ids),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
