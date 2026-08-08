"""Freeze the VARC posterior and structural bridge before query-gold scoring."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/home/wangmeiqi/codex_runs/arc_visual_posterior_bridge_20260809")
REPOSITORY = Path(
    "/home/wangmeiqi/codex_runs/symbolic_logic_arc_online_control_20260724/repo"
    "/arc_functional_transition_solver"
)
sys.path.insert(0, str(REPOSITORY / "src"))

from afts_arc.experiment_safety import atomic_write_json, canonical_sha256  # noqa: E402
from afts_arc.visual_provider_gate import (  # noqa: E402
    _file_sha256,
    _load_json_object,
    freeze_provider_predictions,
)
from afts_arc.visual_structure_bridge import (  # noqa: E402
    freeze_visual_structure_bridge,
)


TRANSFER_ROOT = ROOT / "from236_structure_v1"
MANIFEST_PATH = ROOT / "cohort_v1" / "manifest.json"
TRANSFER_MANIFEST_PATH = TRANSFER_ROOT / "cohort_v1" / "manifest.json"
VALIDATION_PATH = TRANSFER_ROOT / "provider_raw_validation_structure_v1.json"
PREDICTION_ROOT = TRANSFER_ROOT / "raw_predictions"
RAW_PACKAGE = ROOT / "visual_structure_v1_raw_compact_from236.tar.gz"
CONTRACT_PATH = ROOT / "provider_contract_structure_v1.json"
FROZEN_PATH = ROOT / "pre_gold" / "frozen_predictions.json"
CANDIDATE_PATH = ROOT / "pre_gold" / "structure_candidates.json"
REPLAY_FROZEN_PATH = ROOT / "pre_gold" / "replay" / "frozen_predictions.json"
REPLAY_CANDIDATE_PATH = ROOT / "pre_gold" / "replay" / "structure_candidates.json"
RECEIPT_PATH = ROOT / "pre_gold" / "freeze_receipt.json"


def _write_new(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(f"artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, payload)


def _verify_content_id(payload: dict[str, object], field: str) -> None:
    content = dict(payload)
    expected = content.pop(field)
    if canonical_sha256(content) != expected:
        raise ValueError(f"invalid content ID: {field}")


def _contract(
    *,
    manifest: dict[str, object],
    validation: dict[str, object],
) -> dict[str, object]:
    prediction_validation = validation["predictions"]
    source = validation["source"]
    content = {
        "schema": "afts.visual-provider-contract/v1",
        "name": "VARC ViT static query-blind provider structure bridge v1",
        "scientific_scope": "solver-track structural selection over frozen whole grids",
        "controller_frozen": True,
        "cohort": {
            "cohort_id": manifest["cohort_id"],
            "manifest_sha256": _file_sha256(MANIFEST_PATH),
            "task_count": len(manifest["tasks"]),
            "source": manifest["source"],
            "selection": manifest["selection"],
        },
        "query_blind_protocol": {
            "gold_output_present_in_provider_root": False,
            "provider_visible_test_output": "exact copy of corresponding test input",
            "released_loader_repair": (
                "sentinel preserves the released preprocessing interface without "
                "carrying a query label; this is not an exact paper-script replication"
            ),
            **validation["query_blind_audit"],
        },
        "source": {
            "repository": "https://github.com/lillian039/VARC",
            "commit": source["varc_commit"],
            "checkpoint_sha256": source["checkpoint_sha256"],
            "executed_source_sha256": source["executed_source_sha256"],
            "checkpoint_epoch": 95,
            "checkpoint_best_eval_accuracy": 0.7908653846153846,
            "checkpoint_training_provenance": {
                "status": (
                    "self-described by checkpoint metadata; dataset contents not "
                    "reconstructed"
                ),
                "data_root": "raw_data/ARC-AGI",
                "train_split": "training",
                "eval_split": "training",
                "include_rearc": True,
                "rearc_path": "raw_data/re_arc",
                "rearc_limit": -1,
            },
        },
        "test_time_configuration": {
            "architecture": "vit",
            "epochs": 100,
            "depth": 10,
            "batch_size": 8,
            "image_size": 64,
            "patch_size": 2,
            "learning_rate": 0.0003,
            "weight_decay": 0.0,
            "embed_dim": 512,
            "num_heads": 8,
            "num_colors": 12,
            "lr_scheduler": "cosine",
            "resume_skip_task_token": True,
            "num_attempts": 10,
            "ttt_num_each": 1,
            "seed": 42,
            "per_task_timeout_seconds": 2400,
            "worker_limit": 2,
        },
        "candidate_contract": {
            "invalid_candidate_policy": "reject",
            "invalid_candidate_rule": (
                "reject non-list, empty, non-rectangular, >30x30, non-integer, or "
                "color-outside-[0,9] samples; never repair or coerce"
            ),
            "ranking": (
                "candidate 1 is frequency/first-emission/canonical-grid top-1; "
                "candidate 2 is highest stable-demo structural-contract match among "
                "different complete raw grids"
            ),
            "grid_synthesis": False,
            "raw_files_frozen_before_gold": True,
            "raw_candidate_count": prediction_validation["raw_candidate_count"],
            "valid_candidate_count": prediction_validation["valid_candidate_count"],
            "rejected_candidate_count": prediction_validation[
                "rejected_candidate_count"
            ],
            "rejection_reasons": prediction_validation["rejection_reasons"],
        },
        "compute": {
            "gpu_seconds": validation["run"]["gpu_seconds"],
            "max_task_seconds": validation["run"]["max_task_seconds"],
            "task_failure_count": validation["run"]["failure_count"],
            "gpu_launch_inventory": (
                TRANSFER_ROOT
                / "runs"
                / "varc_structure_bridge_v1"
                / "gpu_launch.csv"
            ).read_text(encoding="utf-8").splitlines(),
        },
        "runtime": validation["runtime"],
        "raw_artifact": {
            "transfer_package_sha256": _file_sha256(RAW_PACKAGE),
            "validation_id": validation["validation_id"],
            "validation_sha256": _file_sha256(VALIDATION_PATH),
        },
    }
    return {"contract_id": canonical_sha256(content), **content}


def main() -> None:
    for path in (
        CONTRACT_PATH,
        FROZEN_PATH,
        CANDIDATE_PATH,
        REPLAY_FROZEN_PATH,
        REPLAY_CANDIDATE_PATH,
        RECEIPT_PATH,
    ):
        if path.exists():
            raise FileExistsError(f"pre-gold artifact already exists: {path}")
    manifest = _load_json_object(MANIFEST_PATH)
    transferred_manifest = _load_json_object(TRANSFER_MANIFEST_PATH)
    validation = _load_json_object(VALIDATION_PATH)
    if manifest != transferred_manifest:
        raise ValueError("transferred and scoring cohort manifests differ")
    if validation["cohort_id"] != manifest["cohort_id"]:
        raise ValueError("validation and cohort IDs differ")
    if validation["cohort_manifest_sha256"] != _file_sha256(MANIFEST_PATH):
        raise ValueError("validation does not bind the scoring cohort manifest")
    if validation["gold_accessed"] is not False:
        raise ValueError("raw validation is not query-blind")
    _verify_content_id(validation, "validation_id")

    contract = _contract(manifest=manifest, validation=validation)
    _write_new(CONTRACT_PATH, contract)
    frozen = freeze_provider_predictions(
        cohort_manifest=MANIFEST_PATH,
        prediction_roots=(PREDICTION_ROOT,),
        provider_contract=contract,
        output_path=FROZEN_PATH,
        invalid_candidate_policy="reject",
    )
    candidates = freeze_visual_structure_bridge(
        cohort_manifest=MANIFEST_PATH,
        blind_task_dir=ROOT / "cohort_v1" / "blind_data" / "ARC-AGI-2" / "data" / "evaluation",
        prediction_roots=(PREDICTION_ROOT,),
        frozen_predictions=FROZEN_PATH,
        output_path=CANDIDATE_PATH,
        invalid_candidate_policy="reject",
    )
    replay_frozen = freeze_provider_predictions(
        cohort_manifest=MANIFEST_PATH,
        prediction_roots=(PREDICTION_ROOT,),
        provider_contract=contract,
        output_path=REPLAY_FROZEN_PATH,
        invalid_candidate_policy="reject",
    )
    replay_candidates = freeze_visual_structure_bridge(
        cohort_manifest=MANIFEST_PATH,
        blind_task_dir=ROOT / "cohort_v1" / "blind_data" / "ARC-AGI-2" / "data" / "evaluation",
        prediction_roots=(PREDICTION_ROOT,),
        frozen_predictions=REPLAY_FROZEN_PATH,
        output_path=REPLAY_CANDIDATE_PATH,
        invalid_candidate_policy="reject",
    )
    if replay_frozen != frozen or _file_sha256(REPLAY_FROZEN_PATH) != _file_sha256(
        FROZEN_PATH
    ):
        raise ValueError("frozen-prediction replay differs")
    if replay_candidates != candidates or _file_sha256(
        REPLAY_CANDIDATE_PATH
    ) != _file_sha256(CANDIDATE_PATH):
        raise ValueError("structure-candidate replay differs")
    receipt_content = {
        "schema": "afts.visual-structure-pre-gold-freeze/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "gold_scoring_started": False,
        "cohort_id": manifest["cohort_id"],
        "cohort_manifest_sha256": _file_sha256(MANIFEST_PATH),
        "provider_contract_id": contract["contract_id"],
        "provider_contract_sha256": _file_sha256(CONTRACT_PATH),
        "raw_validation_id": validation["validation_id"],
        "raw_validation_sha256": _file_sha256(VALIDATION_PATH),
        "raw_transfer_package_sha256": _file_sha256(RAW_PACKAGE),
        "frozen_prediction_id": frozen["frozen_prediction_id"],
        "frozen_prediction_sha256": _file_sha256(FROZEN_PATH),
        "structure_candidate_id": candidates["candidate_id"],
        "structure_candidate_sha256": _file_sha256(CANDIDATE_PATH),
        "replay": {
            "byte_identical": True,
            "frozen_prediction_id": replay_frozen["frozen_prediction_id"],
            "frozen_prediction_sha256": _file_sha256(REPLAY_FROZEN_PATH),
            "structure_candidate_id": replay_candidates["candidate_id"],
            "structure_candidate_sha256": _file_sha256(REPLAY_CANDIDATE_PATH),
        },
    }
    receipt = {"freeze_id": canonical_sha256(receipt_content), **receipt_content}
    _write_new(RECEIPT_PATH, receipt)
    print(
        json.dumps(
            {
                "contract_id": contract["contract_id"],
                "frozen_prediction_id": frozen["frozen_prediction_id"],
                "structure_candidate_id": candidates["candidate_id"],
                "freeze_id": receipt["freeze_id"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
