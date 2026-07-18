"""Pure-Python optimizer, schedule, selection, and budget rules for M04a."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from .m04a_contract import (
    DATA_FOLDS_SEMANTICS_VERSION,
    GRADIENT_ACCUMULATION,
    MODEL_SEMANTICS_VERSION,
    OPTIMIZER_UPDATES,
    SAMPLER_SEMANTICS_VERSION,
    TRAINING_SEED,
    VALIDATION_INTERVAL,
    VALIDATION_PASS_COUNT,
    VALIDATION_SEMANTICS_VERSION,
    canonical_sha256,
)


TRAINING_CONFIG_SCHEMA_VERSION = "afts-grid-cmlm-training-config/v0.1"
CHECKPOINT_SELECTION_SCHEMA_VERSION = "afts-grid-cmlm-checkpoint-selection/v0.1"
BASE_LEARNING_RATE = 3e-4
WARMUP_UPDATES = 2_000
COSINE_UPDATES = OPTIMIZER_UPDATES - WARMUP_UPDATES
CAMPAIGN_GPU_BUDGET_NS = 24 * 60 * 60 * 1_000_000_000
PREFLIGHT_UPDATES = 100


def _strict_int(value: object, *, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise TypeError(f"{field} must be an integer >= {minimum}")
    return value


def learning_rate_for_update(update_number: int) -> float:
    """Return the frozen LR assigned immediately before optimizer update ``t``."""

    update_number = _strict_int(update_number, field="update_number", minimum=1)
    if update_number > OPTIMIZER_UPDATES:
        raise ValueError(f"update_number must be in 1..{OPTIMIZER_UPDATES}")
    if update_number <= WARMUP_UPDATES:
        return BASE_LEARNING_RATE * update_number / WARMUP_UPDATES
    return BASE_LEARNING_RATE * 0.5 * (
        1.0
        + math.cos(
            math.pi * (update_number - WARMUP_UPDATES) / COSINE_UPDATES
        )
    )


def training_config_payload() -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": TRAINING_CONFIG_SCHEMA_VERSION,
        "model_semantics_version": MODEL_SEMANTICS_VERSION,
        "sampler_semantics_version": SAMPLER_SEMANTICS_VERSION,
        "data_folds_semantics_version": DATA_FOLDS_SEMANTICS_VERSION,
        "validation_semantics_version": VALIDATION_SEMANTICS_VERSION,
        "training_seed": TRAINING_SEED,
        "optimizer": {
            "name": "AdamW",
            "base_learning_rate": BASE_LEARNING_RATE,
            "betas": [0.9, 0.95],
            "weight_decay": 0.1,
            "eps": 1e-8,
            "amsgrad": False,
            "maximize": False,
            "foreach": False,
            "fused": False,
            "capturable": False,
            "differentiable": False,
            "parameter_groups": 1,
            "decay_exclusions": [],
        },
        "optimizer_updates": OPTIMIZER_UPDATES,
        "warmup_updates": WARMUP_UPDATES,
        "post_warmup_schedule": "cosine_to_zero",
        "gradient_accumulation": GRADIENT_ACCUMULATION,
        "microbatch_episodes": 1,
        "precision": "cuda_bfloat16_autocast_forward_fp32_loss",
        "grad_scaler": False,
        "gradient_clip_global_norm": 1.0,
        "validation_interval": VALIDATION_INTERVAL,
        "validation_pass_count": VALIDATION_PASS_COUNT,
        "preflight_updates": PREFLIGHT_UPDATES,
        "campaign_gpu_budget_ns": CAMPAIGN_GPU_BUDGET_NS,
        "checkpoint_selection": "minimum_parent_grouped_masked_cell_ce_then_earlier_step",
    }
    payload["config_id"] = canonical_sha256(payload)
    return payload


def training_config_sha256() -> str:
    return str(training_config_payload()["config_id"])


@dataclass(frozen=True, slots=True)
class CheckpointMetric:
    optimizer_step: int
    parent_grouped_ce: float
    checkpoint_sha256: str

    def __post_init__(self) -> None:
        step = _strict_int(self.optimizer_step, field="optimizer_step", minimum=1)
        if step > OPTIMIZER_UPDATES or step % VALIDATION_INTERVAL != 0:
            raise ValueError("checkpoint step must be a validation-interval multiple")
        if (
            isinstance(self.parent_grouped_ce, bool)
            or not isinstance(self.parent_grouped_ce, (int, float))
            or not math.isfinite(float(self.parent_grouped_ce))
            or float(self.parent_grouped_ce) < 0.0
        ):
            raise ValueError("parent_grouped_ce must be finite and non-negative")
        if not isinstance(self.checkpoint_sha256, str) or re.fullmatch(
            r"[0-9a-f]{64}", self.checkpoint_sha256
        ) is None:
            raise ValueError("checkpoint_sha256 must be lowercase SHA-256")


def select_checkpoint(metrics: tuple[CheckpointMetric, ...]) -> CheckpointMetric:
    if type(metrics) is not tuple or not metrics:
        raise TypeError("metrics must be a non-empty tuple")
    if any(type(row) is not CheckpointMetric for row in metrics):
        raise TypeError("every checkpoint metric must be an exact CheckpointMetric")
    expected_steps = tuple(
        range(VALIDATION_INTERVAL, OPTIMIZER_UPDATES + 1, VALIDATION_INTERVAL)
    )
    if tuple(row.optimizer_step for row in metrics) != expected_steps:
        raise ValueError(
            "checkpoint metrics must contain all "
            f"{VALIDATION_PASS_COUNT} literal validation steps"
        )
    return min(metrics, key=lambda row: (float(row.parent_grouped_ce), row.optimizer_step))


def assert_campaign_budget(cumulative_lock_held_ns: int) -> None:
    cumulative = _strict_int(
        cumulative_lock_held_ns, field="cumulative_lock_held_ns"
    )
    if cumulative > CAMPAIGN_GPU_BUDGET_NS:
        raise RuntimeError("BUDGET_EXCEEDED")


__all__ = [
    "BASE_LEARNING_RATE",
    "CAMPAIGN_GPU_BUDGET_NS",
    "CHECKPOINT_SELECTION_SCHEMA_VERSION",
    "COSINE_UPDATES",
    "CheckpointMetric",
    "PREFLIGHT_UPDATES",
    "TRAINING_CONFIG_SCHEMA_VERSION",
    "WARMUP_UPDATES",
    "assert_campaign_budget",
    "learning_rate_for_update",
    "select_checkpoint",
    "training_config_payload",
    "training_config_sha256",
]
