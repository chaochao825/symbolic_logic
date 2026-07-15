"""Frozen single-update training and literal-manifest validation for M04a.

This is an explicit optional-PyTorch boundary.  It deliberately implements only
the reusable numerical core: one primary optimizer update and one validation
pass.  Campaign coordination, checkpoint publication, preflight construction,
and resume orchestration remain outside this module.

The public runtime functions fail closed on model, device, optimizer, episode,
and manifest drift.  CUDA synchronization occurs only at the update boundary or
at the explicitly timed boundary around each validation episode; no loss value
is copied to the host from an individual training microbatch.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .m04a_contract import (
    GRADIENT_ACCUMULATION,
    MODEL_PARAMETER_COUNT,
    OPTIMIZER_UPDATES,
    VALIDATION_INTERVAL,
    VALIDATION_PASS_COUNT,
    canonical_sha256,
    episode_seed,
)
from .m04a_data import (
    ARC2_SOURCE,
    REARC_SOURCE,
    M04AEpisode,
    M04ATrainingData,
    build_training_episode,
)
from .m04a_evidence import make_training_cost_ledger_row, sequential_fp32_mean
from .m04a_model import (
    GridCMLM,
    TokenBatch,
    mask_target_grid,
    tokenize_target_batch,
    tokenize_task_memory,
    trainable_parameter_count,
)
from .m04a_torch_runtime import (
    CUDATiming,
    FrozenRuntimeAttestation,
    bf16_autocast,
    capture_rng_state,
    restore_rng_state,
    timed_cuda_call,
    validate_runtime_attestation,
)
from .m04a_train_contract import BASE_LEARNING_RATE, learning_rate_for_update
from .m04a_validation_manifest import (
    ValidationEpisodeManifest,
    build_validation_episode_artifacts,
    require_externally_committed_validation_manifest,
    validate_validation_episode_rows,
)


TRAINING_MASK_AUDIT_SCHEMA_VERSION = "afts-m04a-training-mask-audit/v0.1"
GRADIENT_CLIP_GLOBAL_NORM = 1.0
ADAMW_BETAS = (0.9, 0.95)
ADAMW_WEIGHT_DECAY = 0.1
ADAMW_EPSILON = 1e-8

_MASK_AUDIT_FIELDS = {
    "schema",
    "row_id",
    "optimizer_step",
    "episode_counter_step",
    "microbatch_slot",
    "source",
    "source_parent_id",
    "semantic_parent_id",
    "target_descriptor",
    "episode_seed_u64",
    "episode_sha256",
    "target_height",
    "target_width",
    "target_cell_count",
    "corruption_kind",
    "masked_linear_indices",
    "masked_token_predictions",
    "mask_sha256",
}


def _strict_int(value: object, *, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise TypeError(f"{field} must be an integer >= {minimum}")
    return value


def _strict_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{field} must be a non-empty string")
    return value


def _finite_nonnegative(value: object, *, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{field} must be finite and non-negative")
    return float(value)


def _sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise TypeError(f"{field} must be a lowercase SHA-256 digest")
    try:
        decoded = bytes.fromhex(value)
    except ValueError as exc:
        raise TypeError(f"{field} must be a lowercase SHA-256 digest") from exc
    if len(decoded) != 32 or value != value.lower():
        raise TypeError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _mask_semantic_payload(row: Mapping[str, Any]) -> dict[str, object]:
    return {
        "target_height": row["target_height"],
        "target_width": row["target_width"],
        "masked_linear_indices": list(row["masked_linear_indices"]),
    }


def make_training_mask_audit_row(
    episode: M04AEpisode, *, optimizer_step: int
) -> dict[str, Any]:
    """Commit the compact, replayable mask identity for one microbatch."""

    optimizer_step = _strict_int(
        optimizer_step, field="optimizer_step", minimum=1
    )
    if optimizer_step > OPTIMIZER_UPDATES:
        raise ValueError(f"optimizer_step must be in 1..{OPTIMIZER_UPDATES}")
    if not isinstance(episode, M04AEpisode):
        raise TypeError("episode must be an M04AEpisode")
    if episode.optimizer_step != optimizer_step - 1:
        raise ValueError("episode counter step does not match the optimizer update")
    if episode.microbatch_slot is None:
        raise ValueError("training mask audit cannot contain a validation episode")
    height = len(episode.target_output)
    width = len(episode.target_output[0])
    episode_payload = episode.to_json_dict()
    row: dict[str, Any] = {
        "schema": TRAINING_MASK_AUDIT_SCHEMA_VERSION,
        "optimizer_step": optimizer_step,
        "episode_counter_step": episode.optimizer_step,
        "microbatch_slot": episode.microbatch_slot,
        "source": episode.source,
        "source_parent_id": episode.parent_id,
        "semantic_parent_id": episode.semantic_parent_id,
        "target_descriptor": episode.target_descriptor,
        "episode_seed_u64": episode.seed_u64,
        "episode_sha256": episode_payload["episode_sha256"],
        "target_height": height,
        "target_width": width,
        "target_cell_count": height * width,
        "corruption_kind": episode.corruption_kind,
        "masked_linear_indices": list(episode.masked_linear_indices),
        "masked_token_predictions": len(episode.masked_linear_indices),
    }
    row["mask_sha256"] = canonical_sha256(_mask_semantic_payload(row))
    row["row_id"] = canonical_sha256(row)
    return validate_training_mask_audit_row(row)


def validate_training_mask_audit_row(row: object) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise TypeError("training mask audit row must be a dictionary")
    if set(row) != _MASK_AUDIT_FIELDS:
        raise ValueError("training mask audit row fields do not match its schema")
    payload = dict(row)
    if payload["schema"] != TRAINING_MASK_AUDIT_SCHEMA_VERSION:
        raise ValueError("unsupported training mask audit schema")
    optimizer_step = _strict_int(
        payload["optimizer_step"], field="optimizer_step", minimum=1
    )
    if optimizer_step > OPTIMIZER_UPDATES:
        raise ValueError("training mask audit optimizer step is out of range")
    counter_step = _strict_int(
        payload["episode_counter_step"], field="episode_counter_step"
    )
    if counter_step != optimizer_step - 1:
        raise ValueError("training mask audit counter/update mismatch")
    slot = _strict_int(payload["microbatch_slot"], field="microbatch_slot")
    if slot >= GRADIENT_ACCUMULATION:
        raise ValueError("training mask audit microbatch slot is out of range")
    expected_source = ARC2_SOURCE if slot % 2 == 0 else REARC_SOURCE
    if payload["source"] != expected_source:
        raise ValueError("training mask audit source does not alternate by slot")
    for field in (
        "source_parent_id",
        "semantic_parent_id",
        "target_descriptor",
        "corruption_kind",
    ):
        _strict_string(payload[field], field=field)
    seed = _strict_int(payload["episode_seed_u64"], field="episode_seed_u64")
    if seed >= 2**64 or seed != episode_seed(counter_step, slot):
        raise ValueError("training mask audit seed does not replay")
    for field in ("episode_sha256", "mask_sha256", "row_id"):
        _sha256(payload[field], field=field)
    height = _strict_int(payload["target_height"], field="target_height", minimum=1)
    width = _strict_int(payload["target_width"], field="target_width", minimum=1)
    cell_count = _strict_int(
        payload["target_cell_count"], field="target_cell_count", minimum=1
    )
    if cell_count != height * width:
        raise ValueError("training mask audit target shape does not close")
    masked = payload["masked_linear_indices"]
    if not isinstance(masked, list) or not masked:
        raise TypeError("masked_linear_indices must be a non-empty JSON array")
    if any(type(index) is not int for index in masked):
        raise TypeError("masked_linear_indices must contain integers")
    if masked != sorted(set(masked)) or masked[0] < 0 or masked[-1] >= cell_count:
        raise ValueError("training mask audit indices are not canonical and in bounds")
    masked_count = _strict_int(
        payload["masked_token_predictions"],
        field="masked_token_predictions",
        minimum=1,
    )
    if masked_count != len(masked):
        raise ValueError("training mask audit masked count does not close")
    corruption_kind = payload["corruption_kind"]
    if cell_count == 1:
        if corruption_kind != "single_cell" or masked != [0]:
            raise ValueError("single-cell audit mask is invalid")
    elif len(masked) == cell_count:
        if corruption_kind != "all_mask":
            raise ValueError("complete audit mask must be all_mask")
    elif corruption_kind != "partial":
        raise ValueError("incomplete multi-cell audit mask must be partial")
    if payload["mask_sha256"] != canonical_sha256(_mask_semantic_payload(payload)):
        raise ValueError("training mask audit mask_sha256 mismatch")
    semantic = dict(payload)
    row_id = semantic.pop("row_id")
    if row_id != canonical_sha256(semantic):
        raise ValueError("training mask audit row_id mismatch")
    return payload


def assert_grid_cmlm_invariants(
    model: nn.Module, *, require_cuda: bool
) -> torch.device:
    """Validate the exact model class, complete FP32 state, and one device."""

    if type(model) is not GridCMLM:
        raise TypeError("training requires the exact GridCMLM class")
    parameters = tuple(model.parameters())
    if not parameters or len({id(parameter) for parameter in parameters}) != len(
        parameters
    ):
        raise RuntimeError("GridCMLM parameters are empty or aliased")
    if trainable_parameter_count(model) != MODEL_PARAMETER_COUNT:
        raise RuntimeError("GridCMLM trainable parameter count drifted")
    if any(not parameter.requires_grad for parameter in parameters):
        raise RuntimeError("every GridCMLM parameter must remain trainable")
    if any(parameter.dtype != torch.float32 for parameter in parameters):
        raise RuntimeError("all GridCMLM parameters must remain FP32")
    devices = {parameter.device for parameter in parameters}
    buffers = tuple(model.buffers())
    devices.update(buffer.device for buffer in buffers)
    if len(devices) != 1:
        raise RuntimeError("all GridCMLM parameters and buffers must share one device")
    device = next(iter(devices))
    if any(buffer.is_floating_point() and buffer.dtype != torch.float32 for buffer in buffers):
        raise RuntimeError("floating GridCMLM buffers must remain FP32")
    if require_cuda:
        if device.type != "cuda" or (device.index not in {None, 0}):
            raise RuntimeError("M04a training requires the sole visible CUDA device")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("M04a training requires exactly one visible CUDA device")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("M04a training requires CUDA BF16 support")
    elif device.type not in {"cpu", "cuda"}:
        raise RuntimeError("GridCMLM supports only CPU or CUDA placement")
    return device


def _optimizer_hyperparameters(group: Mapping[str, object]) -> None:
    expected: dict[str, object] = {
        "betas": ADAMW_BETAS,
        "eps": ADAMW_EPSILON,
        "weight_decay": ADAMW_WEIGHT_DECAY,
        "amsgrad": False,
        "maximize": False,
        "foreach": False,
        "capturable": False,
        "differentiable": False,
        "fused": False,
    }
    for field, value in expected.items():
        if group.get(field) != value:
            raise RuntimeError(f"AdamW {field} drifted from the frozen contract")


def assert_adamw_invariants(
    optimizer: torch.optim.Optimizer,
    model: GridCMLM,
    *,
    expected_learning_rate: float | None = None,
    expected_completed_updates: int | None = None,
) -> None:
    """Validate the exact one-group optimizer and all materialized FP32 state."""

    if type(optimizer) is not torch.optim.AdamW:
        raise TypeError("M04a requires the exact torch.optim.AdamW class")
    if len(optimizer.param_groups) != 1:
        raise RuntimeError("M04a AdamW must contain exactly one parameter group")
    parameters = tuple(model.parameters())
    group = optimizer.param_groups[0]
    grouped = tuple(group["params"])
    if len(grouped) != len(parameters) or any(
        grouped[index] is not parameter
        for index, parameter in enumerate(parameters)
    ):
        raise RuntimeError("AdamW parameter group does not preserve every model parameter")
    _optimizer_hyperparameters(group)
    _optimizer_hyperparameters(optimizer.defaults)
    learning_rate = group.get("lr")
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, (int, float))
        or not math.isfinite(float(learning_rate))
        or float(learning_rate) < 0.0
    ):
        raise RuntimeError("AdamW learning rate must be a finite non-negative scalar")
    if expected_learning_rate is not None and float(learning_rate) != float(
        expected_learning_rate
    ):
        raise RuntimeError("AdamW learning rate does not match the current update")

    state_keys = set(optimizer.state)
    parameter_keys = set(parameters)
    if state_keys and state_keys != parameter_keys:
        raise RuntimeError("AdamW state is only partially materialized")
    if expected_completed_updates is not None:
        expected_completed_updates = _strict_int(
            expected_completed_updates,
            field="expected_completed_updates",
        )
        if expected_completed_updates == 0 and state_keys:
            raise RuntimeError("AdamW has state before the first optimizer update")
        if expected_completed_updates > 0 and not state_keys:
            raise RuntimeError("AdamW lacks state for completed optimizer updates")
    for parameter in parameters:
        state = optimizer.state.get(parameter)
        if state is None:
            continue
        if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
            raise RuntimeError("AdamW state fields drifted from non-AMSGrad AdamW")
        step = state["step"]
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        if (
            not isinstance(step, Tensor)
            or step.dtype != torch.float32
            or step.device.type != "cpu"
            or step.numel() != 1
        ):
            raise RuntimeError("AdamW step state must be one CPU FP32 scalar")
        if (
            not isinstance(exp_avg, Tensor)
            or not isinstance(exp_avg_sq, Tensor)
            or exp_avg.dtype != torch.float32
            or exp_avg_sq.dtype != torch.float32
            or exp_avg.device != parameter.device
            or exp_avg_sq.device != parameter.device
            or exp_avg.shape != parameter.shape
            or exp_avg_sq.shape != parameter.shape
        ):
            raise RuntimeError("AdamW moment state must match its FP32 parameter")
        if (
            expected_completed_updates is not None
            and float(step.item()) != float(expected_completed_updates)
        ):
            raise RuntimeError("AdamW step state does not match the requested update")


def build_adamw_optimizer(model: GridCMLM) -> torch.optim.AdamW:
    """Create the frozen one-group AdamW with no decay exclusions."""

    assert_grid_cmlm_invariants(model, require_cuda=False)
    optimizer = torch.optim.AdamW(
        tuple(model.parameters()),
        lr=BASE_LEARNING_RATE,
        betas=ADAMW_BETAS,
        eps=ADAMW_EPSILON,
        weight_decay=ADAMW_WEIGHT_DECAY,
        amsgrad=False,
        maximize=False,
        foreach=False,
        capturable=False,
        differentiable=False,
        fused=False,
    )
    assert_adamw_invariants(
        optimizer,
        model,
        expected_learning_rate=BASE_LEARNING_RATE,
        expected_completed_updates=0,
    )
    return optimizer


def _assert_no_gradients(model: GridCMLM) -> None:
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("model gradients must be cleared at this boundary")


def _assert_uniform_model_mode(model: GridCMLM, expected: bool) -> None:
    if any(bool(module.training) != expected for module in model.modules()):
        raise RuntimeError("GridCMLM module modes are not uniformly restored")


@dataclass(frozen=True, slots=True)
class _PreparedEpisode:
    task_batch: TokenBatch
    target_batch: TokenBatch
    labels: Tensor
    masked_indices: Tensor


def _prepare_episode(episode: M04AEpisode, device: torch.device) -> _PreparedEpisode:
    masked_target = mask_target_grid(
        episode.target_output, episode.masked_linear_indices
    )
    task_batch = tokenize_task_memory(
        episode.demonstrations, episode.query_input, device="cpu"
    ).to(device)
    target_batch = tokenize_target_batch((masked_target,), device="cpu").to(device)
    labels = torch.tensor(
        [cell for row in episode.target_output for cell in row],
        dtype=torch.long,
        device=device,
    )
    masked_indices = torch.tensor(
        episode.masked_linear_indices, dtype=torch.long, device=device
    )
    if target_batch.batch_size != 1 or target_batch.lengths != (labels.numel(),):
        raise RuntimeError("training target tokenization is not microbatch-one")
    return _PreparedEpisode(
        task_batch=task_batch,
        target_batch=target_batch,
        labels=labels,
        masked_indices=masked_indices,
    )


def _episode_masked_ce(model: GridCMLM, prepared: _PreparedEpisode) -> Tensor:
    with bf16_autocast():
        decoded = model(prepared.task_batch, prepared.target_batch)
    masked_logits = decoded.logits[0].index_select(
        0, prepared.masked_indices
    ).float()
    masked_labels = prepared.labels.index_select(0, prepared.masked_indices)
    loss = F.cross_entropy(masked_logits, masked_labels, reduction="mean")
    if loss.dtype != torch.float32 or loss.ndim != 0:
        raise RuntimeError("masked-cell cross entropy must be one FP32 scalar")
    return loss


def _execute_cuda_update(
    model: GridCMLM,
    optimizer: torch.optim.AdamW,
    episodes: Sequence[M04AEpisode],
    learning_rate: float,
) -> tuple[Tensor, Tensor]:
    device = next(model.parameters()).device
    loss_sum = torch.zeros((), dtype=torch.float32, device=device)
    for episode in episodes:
        prepared = _prepare_episode(episode, device)
        loss = _episode_masked_ce(model, prepared)
        (loss / GRADIENT_ACCUMULATION).backward()
        loss_sum = loss_sum + loss.detach()

    parameters = tuple(model.parameters())
    if any(
        parameter.grad is None
        or parameter.grad.dtype != torch.float32
        or parameter.grad.device != parameter.device
        for parameter in parameters
    ):
        raise RuntimeError("every trainable parameter must have one FP32 accumulated gradient")
    # This is the sole accumulation-boundary finite check.  There is no per-forward
    # .item(), synchronize(), or D2H copy in the sixteen-microbatch hot path.
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        parameters,
        max_norm=GRADIENT_CLIP_GLOBAL_NORM,
        norm_type=2.0,
        error_if_nonfinite=True,
        foreach=False,
    )
    optimizer.param_groups[0]["lr"] = learning_rate
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return loss_sum / GRADIENT_ACCUMULATION, gradient_norm.detach().float()


@dataclass(frozen=True, slots=True)
class TrainingUpdateResult:
    optimizer_step: int
    learning_rate: float
    mean_masked_cell_ce: float
    gradient_norm_before_clip: float
    masked_token_predictions: int
    mask_audit_rows: tuple[dict[str, Any], ...]
    ledger_row: dict[str, Any]
    timing: CUDATiming

    def __post_init__(self) -> None:
        _strict_int(self.optimizer_step, field="optimizer_step", minimum=1)
        _finite_nonnegative(self.learning_rate, field="learning_rate")
        _finite_nonnegative(self.mean_masked_cell_ce, field="mean_masked_cell_ce")
        _finite_nonnegative(
            self.gradient_norm_before_clip, field="gradient_norm_before_clip"
        )
        if len(self.mask_audit_rows) != GRADIENT_ACCUMULATION:
            raise ValueError("training update must contain sixteen mask audit rows")


def train_primary_update(
    model: GridCMLM,
    optimizer: torch.optim.AdamW,
    data: M04ATrainingData,
    *,
    runtime_attestation: FrozenRuntimeAttestation,
    optimizer_step: int,
    event_index: int,
) -> TrainingUpdateResult:
    """Run one exact 16-episode primary update and emit its semantic ledger row."""

    optimizer_step = _strict_int(
        optimizer_step, field="optimizer_step", minimum=1
    )
    if optimizer_step > OPTIMIZER_UPDATES:
        raise ValueError(f"optimizer_step must be in 1..{OPTIMIZER_UPDATES}")
    event_index = _strict_int(event_index, field="event_index")
    if not isinstance(data, M04ATrainingData):
        raise TypeError("data must be M04ATrainingData")
    device = assert_grid_cmlm_invariants(model, require_cuda=True)
    validate_runtime_attestation(runtime_attestation)
    if not model.training:
        raise RuntimeError("primary update requires model.train() mode")
    _assert_uniform_model_mode(model, True)
    _assert_no_gradients(model)
    assert_adamw_invariants(
        optimizer,
        model,
        expected_completed_updates=optimizer_step - 1,
    )
    learning_rate = learning_rate_for_update(optimizer_step)

    wall_start = time.perf_counter_ns()
    cpu_start = time.process_time_ns()
    episodes: list[M04AEpisode] = []
    audit_rows: list[dict[str, Any]] = []
    for slot in range(GRADIENT_ACCUMULATION):
        episode = build_training_episode(data, optimizer_step - 1, slot)
        expected_source = ARC2_SOURCE if slot % 2 == 0 else REARC_SOURCE
        if episode.source != expected_source or episode.microbatch_slot != slot:
            raise RuntimeError("training episode stream violated frozen slot alternation")
        episodes.append(episode)
        audit_rows.append(
            make_training_mask_audit_row(episode, optimizer_step=optimizer_step)
        )

    outputs, timing = timed_cuda_call(
        lambda: _execute_cuda_update(model, optimizer, episodes, learning_rate)
    )
    mean_loss_tensor, gradient_norm_tensor = outputs
    mean_loss = float(mean_loss_tensor.item())
    gradient_norm = float(gradient_norm_tensor.item())
    if not math.isfinite(mean_loss) or mean_loss < 0.0:
        raise RuntimeError("primary update produced non-finite masked-cell CE")
    if not math.isfinite(gradient_norm) or gradient_norm < 0.0:
        raise RuntimeError("primary update produced non-finite gradient norm")
    _assert_no_gradients(model)
    assert_grid_cmlm_invariants(model, require_cuda=True)
    assert_adamw_invariants(
        optimizer,
        model,
        expected_learning_rate=learning_rate,
        expected_completed_updates=optimizer_step,
    )
    if next(model.parameters()).device != device:
        raise RuntimeError("model device changed during the optimizer update")

    wall_time_ns = max(time.perf_counter_ns() - wall_start, timing.gpu_ns)
    cpu_time_ns = time.process_time_ns() - cpu_start
    masked_predictions = sum(
        int(row["masked_token_predictions"]) for row in audit_rows
    )
    ledger_row = make_training_cost_ledger_row(
        phase="primary_update",
        event_index=event_index,
        optimizer_step=optimizer_step,
        optimizer_updates=1,
        microbatches=GRADIENT_ACCUMULATION,
        arc2_episodes=GRADIENT_ACCUMULATION // 2,
        rearc_episodes=GRADIENT_ACCUMULATION // 2,
        encoder_forward_calls=GRADIENT_ACCUMULATION,
        decoder_forward_calls=GRADIENT_ACCUMULATION,
        backward_calls=GRADIENT_ACCUMULATION,
        masked_token_predictions=masked_predictions,
        cuda_event_ns=timing.gpu_ns,
        cpu_time_ns=cpu_time_ns,
        wall_time_ns=wall_time_ns,
        cuda_peak_allocated_bytes=timing.peak_allocated_bytes,
        cuda_peak_reserved_bytes=timing.peak_reserved_bytes,
    )
    return TrainingUpdateResult(
        optimizer_step=optimizer_step,
        learning_rate=learning_rate,
        mean_masked_cell_ce=mean_loss,
        gradient_norm_before_clip=gradient_norm,
        masked_token_predictions=masked_predictions,
        mask_audit_rows=tuple(audit_rows),
        ledger_row=ledger_row,
        timing=timing,
    )


@dataclass(frozen=True, slots=True)
class ValidationTargetMetric:
    target_group_id: str
    semantic_parent_id: str
    view_count: int
    mean_masked_cell_ce: float

    def __post_init__(self) -> None:
        _strict_string(self.target_group_id, field="target_group_id")
        _strict_string(self.semantic_parent_id, field="semantic_parent_id")
        _strict_int(self.view_count, field="view_count", minimum=1)
        _finite_nonnegative(
            self.mean_masked_cell_ce, field="mean_masked_cell_ce"
        )


@dataclass(frozen=True, slots=True)
class ValidationParentMetric:
    semantic_parent_id: str
    target_count: int
    mean_masked_cell_ce: float

    def __post_init__(self) -> None:
        _strict_string(self.semantic_parent_id, field="semantic_parent_id")
        _strict_int(self.target_count, field="target_count", minimum=1)
        _finite_nonnegative(
            self.mean_masked_cell_ce, field="mean_masked_cell_ce"
        )


@dataclass(frozen=True, slots=True)
class ValidationAggregation:
    target_metrics: tuple[ValidationTargetMetric, ...]
    parent_metrics: tuple[ValidationParentMetric, ...]
    parent_grouped_ce: float

    def __post_init__(self) -> None:
        if not self.target_metrics or not self.parent_metrics:
            raise ValueError("validation aggregation must be non-empty")
        _finite_nonnegative(self.parent_grouped_ce, field="parent_grouped_ce")


def aggregate_validation_fp32(
    semantic_parent_ids: Sequence[str],
    target_group_ids: Sequence[str],
    episode_losses: Sequence[Tensor],
) -> ValidationAggregation:
    """Average FP32 views -> targets -> semantic parents in literal order."""

    if isinstance(semantic_parent_ids, (str, bytes)) or not isinstance(
        semantic_parent_ids, Sequence
    ):
        raise TypeError("semantic_parent_ids must be an ordered sequence")
    if isinstance(target_group_ids, (str, bytes)) or not isinstance(
        target_group_ids, Sequence
    ):
        raise TypeError("target_group_ids must be an ordered sequence")
    if isinstance(episode_losses, (str, bytes)) or not isinstance(
        episode_losses, Sequence
    ):
        raise TypeError("episode_losses must be an ordered sequence")
    if not episode_losses or not (
        len(semantic_parent_ids) == len(target_group_ids) == len(episode_losses)
    ):
        raise ValueError("validation aggregation inputs must align and be non-empty")

    target_order: list[str] = []
    target_parent: dict[str, str] = {}
    target_losses: dict[str, list[float]] = {}
    closed_targets: set[str] = set()
    previous_target: str | None = None
    for index, (parent_id, target_id, loss) in enumerate(
        zip(semantic_parent_ids, target_group_ids, episode_losses, strict=True)
    ):
        parent_id = _strict_string(parent_id, field=f"semantic_parent_ids[{index}]")
        target_id = _strict_string(target_id, field=f"target_group_ids[{index}]")
        if (
            not isinstance(loss, Tensor)
            or loss.device.type != "cpu"
            or loss.dtype != torch.float32
            or loss.ndim != 0
        ):
            raise TypeError("validation losses must be CPU FP32 scalar tensors")
        if not bool(torch.isfinite(loss).item()) or float(loss.item()) < 0.0:
            raise ValueError("validation losses must be finite and non-negative")
        if target_id != previous_target:
            if target_id in closed_targets:
                raise ValueError("validation target groups must form literal contiguous runs")
            if previous_target is not None:
                closed_targets.add(previous_target)
            previous_target = target_id
        if target_id not in target_losses:
            target_order.append(target_id)
            target_parent[target_id] = parent_id
            target_losses[target_id] = []
        elif target_parent[target_id] != parent_id:
            raise ValueError("one target group cannot cross semantic parents")
        target_losses[target_id].append(float(loss.item()))

    target_means: dict[str, float] = {}
    target_metrics: list[ValidationTargetMetric] = []
    parent_order: list[str] = []
    parent_targets: dict[str, list[float]] = {}
    for target_id in target_order:
        mean = sequential_fp32_mean(target_losses[target_id])
        parent_id = target_parent[target_id]
        target_means[target_id] = mean
        target_metrics.append(
            ValidationTargetMetric(
                target_group_id=target_id,
                semantic_parent_id=parent_id,
                view_count=len(target_losses[target_id]),
                mean_masked_cell_ce=mean,
            )
        )
        if parent_id not in parent_targets:
            parent_order.append(parent_id)
            parent_targets[parent_id] = []
        parent_targets[parent_id].append(target_means[target_id])

    parent_means: list[float] = []
    parent_metrics: list[ValidationParentMetric] = []
    for parent_id in parent_order:
        parent_mean = sequential_fp32_mean(parent_targets[parent_id])
        parent_means.append(parent_mean)
        parent_metrics.append(
            ValidationParentMetric(
                semantic_parent_id=parent_id,
                target_count=len(parent_targets[parent_id]),
                mean_masked_cell_ce=parent_mean,
            )
        )
    overall = sequential_fp32_mean(parent_means)
    return ValidationAggregation(
        target_metrics=tuple(target_metrics),
        parent_metrics=tuple(parent_metrics),
        parent_grouped_ce=overall,
    )


@dataclass(frozen=True, slots=True)
class ValidationEpisodeMetric:
    ledger_event_id: str
    row_ordinal: int
    episode_sha256: str
    target_group_id: str
    semantic_parent_id: str
    masked_token_predictions: int
    masked_cell_ce: float

    def __post_init__(self) -> None:
        _sha256(self.ledger_event_id, field="ledger_event_id")
        _strict_int(self.row_ordinal, field="row_ordinal")
        _sha256(self.episode_sha256, field="episode_sha256")
        _strict_string(self.target_group_id, field="target_group_id")
        _strict_string(self.semantic_parent_id, field="semantic_parent_id")
        _strict_int(
            self.masked_token_predictions,
            field="masked_token_predictions",
            minimum=1,
        )
        _finite_nonnegative(self.masked_cell_ce, field="masked_cell_ce")

    def to_evidence_row(self) -> dict[str, Any]:
        return {
            "ledger_event_id": self.ledger_event_id,
            "row_ordinal": self.row_ordinal,
            "episode_sha256": self.episode_sha256,
            "semantic_parent_id": self.semantic_parent_id,
            "target_group_id": self.target_group_id,
            "masked_token_predictions": self.masked_token_predictions,
            "masked_cell_ce_hex": self.masked_cell_ce.hex(),
        }


@dataclass(frozen=True, slots=True)
class ValidationPassResult:
    optimizer_step: int
    validation_pass_index: int
    episode_metrics: tuple[ValidationEpisodeMetric, ...]
    aggregation: ValidationAggregation
    ledger_rows: tuple[dict[str, Any], ...]

    def __post_init__(self) -> None:
        _strict_int(self.optimizer_step, field="optimizer_step", minimum=1)
        _strict_int(
            self.validation_pass_index,
            field="validation_pass_index",
            minimum=1,
        )
        if not self.episode_metrics or len(self.episode_metrics) != len(
            self.ledger_rows
        ):
            raise ValueError("validation metrics and ledger rows must align")
        for ordinal, (metric, ledger_row) in enumerate(
            zip(self.episode_metrics, self.ledger_rows, strict=True)
        ):
            if (
                metric.row_ordinal != ordinal
                or metric.ledger_event_id != ledger_row.get("event_id")
                or ledger_row.get("episode_index") != ordinal
            ):
                raise ValueError("validation episode metric does not bind its ledger row")

    def evidence_episode_metrics(self) -> tuple[dict[str, Any], ...]:
        return tuple(metric.to_evidence_row() for metric in self.episode_metrics)


def _validated_literal_manifest(
    manifest: ValidationEpisodeManifest,
) -> tuple[tuple[M04AEpisode, ...], tuple[dict[str, Any], ...]]:
    if not isinstance(manifest, ValidationEpisodeManifest):
        raise TypeError("manifest must be a ValidationEpisodeManifest")
    rows = tuple(manifest.rows)
    episodes = validate_validation_episode_rows(rows)
    if episodes != tuple(manifest.episodes):
        raise ValueError("validation manifest episodes do not match literal rows")
    rebuilt = build_validation_episode_artifacts(episodes)
    if (
        rebuilt.rows != rows
        or rebuilt.summary != manifest.summary
        or rebuilt.jsonl_bytes != manifest.jsonl_bytes
    ):
        raise ValueError("validation manifest object is not its canonical materialization")
    return episodes, rows


def _rng_states_equal(
    left: Mapping[str, object], right: Mapping[str, object]
) -> bool:
    if left.get("schema") != right.get("schema") or left.get(
        "optimizer_step"
    ) != right.get("optimizer_step"):
        return False
    left_cpu = left.get("torch_cpu")
    right_cpu = right.get("torch_cpu")
    left_cuda = left.get("torch_cuda")
    right_cuda = right.get("torch_cuda")
    if (
        not isinstance(left_cpu, Tensor)
        or not isinstance(right_cpu, Tensor)
        or not torch.equal(left_cpu, right_cpu)
        or not isinstance(left_cuda, list)
        or not isinstance(right_cuda, list)
        or len(left_cuda) != len(right_cuda)
    ):
        return False
    return all(
        isinstance(left_state, Tensor)
        and isinstance(right_state, Tensor)
        and torch.equal(left_state, right_state)
        for left_state, right_state in zip(left_cuda, right_cuda, strict=True)
    )


def _validation_episode_loss(
    model: GridCMLM, episode: M04AEpisode, device: torch.device
) -> Tensor:
    prepared = _prepare_episode(episode, device)
    if prepared.target_batch.batch_size != 1:
        raise RuntimeError("validation must evaluate exactly one episode at a time")
    with torch.no_grad():
        return _episode_masked_ce(model, prepared).detach()


def validate_literal_manifest(
    model: GridCMLM,
    manifest: ValidationEpisodeManifest,
    *,
    runtime_attestation: FrozenRuntimeAttestation,
    optimizer_step: int,
    validation_pass_index: int,
    event_index_start: int,
    expected_artifact_manifest_sha256: str,
    expected_jsonl_sha256: str,
    budget_check: Callable[[], None] | None = None,
) -> ValidationPassResult:
    """Evaluate one stored validation pass without consuming training RNG state."""

    optimizer_step = _strict_int(
        optimizer_step, field="optimizer_step", minimum=1
    )
    validation_pass_index = _strict_int(
        validation_pass_index,
        field="validation_pass_index",
        minimum=1,
    )
    if validation_pass_index > VALIDATION_PASS_COUNT:
        raise ValueError(
            "validation_pass_index exceeds the frozen "
            f"{VALIDATION_PASS_COUNT} passes"
        )
    if optimizer_step != validation_pass_index * VALIDATION_INTERVAL:
        raise ValueError("validation pass does not bind its checkpoint optimizer step")
    event_index_start = _strict_int(event_index_start, field="event_index_start")
    require_externally_committed_validation_manifest(
        manifest,
        expected_artifact_manifest_sha256=expected_artifact_manifest_sha256,
        expected_jsonl_sha256=expected_jsonl_sha256,
    )
    if budget_check is not None and not callable(budget_check):
        raise TypeError("budget_check must be callable when supplied")
    device = assert_grid_cmlm_invariants(model, require_cuda=True)
    validate_runtime_attestation(runtime_attestation)
    _assert_no_gradients(model)
    episodes, rows = _validated_literal_manifest(manifest)

    original_training_mode = bool(model.training)
    _assert_uniform_model_mode(model, original_training_mode)
    rng_before = capture_rng_state(optimizer_step=optimizer_step)
    loss_tensors: list[Tensor] = []
    ledger_rows: list[dict[str, Any]] = []
    try:
        model.eval()
        for episode_index, (episode, row) in enumerate(
            zip(episodes, rows, strict=True)
        ):
            if budget_check is not None:
                budget_check()
            wall_start = time.perf_counter_ns()
            cpu_start = time.process_time_ns()
            loss, timing = timed_cuda_call(
                lambda episode=episode: _validation_episode_loss(
                    model, episode, device
                )
            )
            cpu_loss = loss.to(device="cpu", dtype=torch.float32).reshape(()).clone()
            if not bool(torch.isfinite(cpu_loss).item()) or float(cpu_loss.item()) < 0.0:
                raise RuntimeError("validation produced non-finite masked-cell CE")
            loss_tensors.append(cpu_loss)
            wall_time_ns = max(time.perf_counter_ns() - wall_start, timing.gpu_ns)
            cpu_time_ns = time.process_time_ns() - cpu_start
            ledger_rows.append(
                make_training_cost_ledger_row(
                    phase="validation_episode",
                    event_index=event_index_start + episode_index,
                    validation_pass_index=validation_pass_index,
                    episode_index=episode_index,
                    validation_episode_calls=1,
                    validation_encoder_forward_calls=1,
                    validation_decoder_forward_calls=1,
                    masked_token_predictions=len(episode.masked_linear_indices),
                    cuda_event_ns=timing.gpu_ns,
                    cpu_time_ns=cpu_time_ns,
                    wall_time_ns=wall_time_ns,
                    cuda_peak_allocated_bytes=timing.peak_allocated_bytes,
                    cuda_peak_reserved_bytes=timing.peak_reserved_bytes,
                )
            )
            if row["episode_sha256"] != episode.to_json_dict()["episode_sha256"]:
                raise RuntimeError("literal validation row changed during evaluation")
            if budget_check is not None:
                budget_check()
        rng_after = capture_rng_state(optimizer_step=optimizer_step)
        if not _rng_states_equal(rng_before, rng_after):
            raise RuntimeError("validation consumed or altered the training RNG state")
    finally:
        restore_rng_state(rng_before)
        model.train(original_training_mode)
        restored = capture_rng_state(optimizer_step=optimizer_step)
        if not _rng_states_equal(rng_before, restored):
            raise RuntimeError("validation failed to restore the training RNG state")
        if bool(model.training) != original_training_mode:
            raise RuntimeError("validation failed to restore model mode")
        _assert_uniform_model_mode(model, original_training_mode)

    aggregation = aggregate_validation_fp32(
        tuple(str(row["semantic_parent_id"]) for row in rows),
        tuple(str(row["target_group_id"]) for row in rows),
        tuple(loss_tensors),
    )
    episode_metrics = tuple(
        ValidationEpisodeMetric(
            ledger_event_id=str(ledger_row["event_id"]),
            row_ordinal=int(row["row_ordinal"]),
            episode_sha256=str(row["episode_sha256"]),
            target_group_id=str(row["target_group_id"]),
            semantic_parent_id=str(row["semantic_parent_id"]),
            masked_token_predictions=len(episode.masked_linear_indices),
            masked_cell_ce=float(loss.item()),
        )
        for episode, row, loss, ledger_row in zip(
            episodes, rows, loss_tensors, ledger_rows, strict=True
        )
    )
    _assert_no_gradients(model)
    assert_grid_cmlm_invariants(model, require_cuda=True)
    return ValidationPassResult(
        optimizer_step=optimizer_step,
        validation_pass_index=validation_pass_index,
        episode_metrics=episode_metrics,
        aggregation=aggregation,
        ledger_rows=tuple(ledger_rows),
    )


__all__ = [
    "ADAMW_BETAS",
    "ADAMW_EPSILON",
    "ADAMW_WEIGHT_DECAY",
    "GRADIENT_CLIP_GLOBAL_NORM",
    "TRAINING_MASK_AUDIT_SCHEMA_VERSION",
    "TrainingUpdateResult",
    "ValidationAggregation",
    "ValidationEpisodeMetric",
    "ValidationParentMetric",
    "ValidationPassResult",
    "ValidationTargetMetric",
    "aggregate_validation_fp32",
    "assert_adamw_invariants",
    "assert_grid_cmlm_invariants",
    "build_adamw_optimizer",
    "make_training_mask_audit_row",
    "train_primary_update",
    "validate_literal_manifest",
    "validate_training_mask_audit_row",
]
