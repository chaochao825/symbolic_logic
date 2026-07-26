"""Leakage and reporting contracts for USRL reproduction experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .data import build_training_episodes, dataset_manifest, discover_tasks


PAPER_SPEC = {
    "reported_arc_agi_1_pass_at_2": 0.472,
    "adaptive_halting_pass_at_2": 0.470,
    "parameters": 7_000_000,
    "augmented_tasks": 876_000,
    "base_arc_tasks": 800,
    "concept_arc_tasks": 160,
    "global_batch_size": 768,
    "epochs_as_named_by_paper": 100_000,
    "optimizer": "AdamW",
    "learning_rate": 1e-4,
    "weight_decay": 0.01,
    "ema": True,
    "training_hardware": "8x NVIDIA L40S + 16x NVIDIA V100",
}


UNSPECIFIED_BY_PAPER = [
    "gate equation and whether answer-rule tokens are concatenated before SM",
    "construction of supervised episodes from each task's demonstration pairs",
    "positional encoding, normalization, activation, and exact Transformer block",
    "stochastic-halting probability/distribution and dynamic replacement algorithm",
    "EMA decay, learning-rate warmup/schedule, gradient clipping, and random seeds",
    "initial answer/state values and gradient detachment boundaries",
    "variable output-shape encoding and pass@2 candidate generation/voting",
    "meaning of one epoch relative to 876k augmented tasks",
]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_protocol(arc_root: Path) -> dict[str, Any]:
    strict_fit, strict_validation = build_training_episodes(arc_root, protocol="strict")
    paper_fit, paper_validation = build_training_episodes(
        arc_root, protocol="paper-transductive"
    )
    files = sorted((arc_root / "training").glob("*.json")) + sorted(
        (arc_root / "evaluation").glob("*.json")
    )
    file_digest = hashlib.sha256(
        "".join(f"{path.name}:{_sha256_file(path)}" for path in files).encode("utf-8")
    ).hexdigest()
    train_tasks = discover_tasks(arc_root, "training")
    eval_tasks = discover_tasks(arc_root, "evaluation")
    return {
        "arc_root": str(arc_root.resolve()),
        "arc_json_file_count": len(files),
        "arc_content_sha256": file_digest,
        "training_task_count": len(train_tasks),
        "evaluation_task_count": len(eval_tasks),
        "strict": {
            "fit": dataset_manifest(strict_fit),
            "validation": dataset_manifest(strict_validation),
            "evaluation_demonstrations_used_for_gradient_updates": False,
        },
        "paper_transductive": {
            "fit": dataset_manifest(paper_fit),
            "validation": dataset_manifest(paper_validation),
            "evaluation_demonstrations_used_for_gradient_updates": True,
            "evaluation_test_outputs_used_for_gradient_updates": False,
        },
        "paper_spec": PAPER_SPEC,
        "paper_unspecified_fields": UNSPECIFIED_BY_PAPER,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
