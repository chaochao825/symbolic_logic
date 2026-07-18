"""Exact, dataset-free CUDA preflight for the frozen M04a campaign.

The preflight owns a fresh model and optimizer, runs the literal 100 by 16
maximum-context training fixture through the production episode preparation,
mask-audit, and CUDA update path, then exercises one eight-lane cached-memory
MaskGIT pass through the production sampler numerics.  It never accepts a path,
dataset, task loader, blind task, oracle, checkpoint, or caller-owned model.
"""

from __future__ import annotations

import hashlib
import math
import os
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .grid import Grid, grid_key, grid_to_lists
from .m04a_contract import (
    DENOISING_STEPS,
    GRADIENT_ACCUMULATION,
    MAX_GRID_SIDE,
    MODEL_PARAMETER_COUNT,
    MODEL_SEMANTICS_VERSION,
    OPTIMIZER_UPDATES,
    OUTPUT_COLOR_COUNT,
    TRAINING_SEED,
    VALIDATION_PASS_COUNT,
    canonical_sha256,
    episode_seed,
    lane_seed,
    mask_count_trace,
)
from .m04a_cost_probe import (
    CampaignOverheadCostProbeResult,
    PreparedCampaignOverheadCostProbe,
    run_campaign_overhead_cost_probe,
    validate_campaign_overhead_cost_probe_report,
    validate_checkpoint_cost_probe,
    validate_final_selection_probe,
    validate_fresh_reconstruction_probe,
    validate_lock_handshake_probe,
)
from .m04a_evidence import (
    PREFLIGHT_PROJECTION_METHOD,
    make_training_cost_ledger_row,
    validate_preflight_artifact as validate_preflight_evidence_artifact,
    validate_training_cost_ledger_row,
)
from .m04a_lock_handshake import (
    MAX_HANDSHAKE_TO_PREFLIGHT_NS,
    HeldGPULockHandshake,
    LockHandshakeArtifact,
    assert_held_gpu_lock,
    attest_inherited_gpu_lock,
    make_cost_probe_lock_handshake,
    make_cost_probe_lock_handshake_from_artifact,
    release_held_gpu_lock,
)
from .m04a_launch_plan import LaunchPlanArtifact
from .m04a_data import ARC2_SOURCE, REARC_SOURCE, M04AEpisode, M04AExample
from .m04a_model import GridCMLM, tokenize_target_batch, tokenize_task_memory
from . import m04a_sample as production_sampler
from .m04a_sample import SamplerRuntime
from .m04a_torch_runtime import (
    FrozenRuntimeAttestation,
    capture_rng_state,
    restore_rng_state,
    timed_cuda_call,
    validate_runtime_attestation,
)
from .m04a_train import (
    _execute_cuda_update,
    assert_adamw_invariants,
    assert_grid_cmlm_invariants,
    build_adamw_optimizer,
    make_training_mask_audit_row,
)
from .m04a_train_contract import (
    CAMPAIGN_GPU_BUDGET_NS,
    PREFLIGHT_UPDATES,
    learning_rate_for_update,
    training_config_sha256,
)
from .m04a_validation_manifest import (
    ValidationManifestCommitment,
    validate_validation_manifest_commitment,
)


PREFLIGHT_FIXTURE_SCHEMA_VERSION = "afts-grid-cmlm-preflight-fixture/v0.1"
PREFLIGHT_INFERENCE_SCHEMA_VERSION = "afts-grid-cmlm-preflight-inference/v0.1"
PREFLIGHT_LANE_TRACE_SCHEMA_VERSION = "afts-grid-cmlm-preflight-lane-trace/v0.1"
PREFLIGHT_PROJECTION_SCHEMA_VERSION = "afts-grid-cmlm-preflight-projection/v0.4"
PREFLIGHT_REPORT_SCHEMA_VERSION = "afts-grid-cmlm-preflight-report/v0.4"
PREFLIGHT_FAILURE_REPORT_SCHEMA_VERSION = (
    "afts-grid-cmlm-preflight-failure-report/v0.2"
)

PREFLIGHT_GRID_SIDE = MAX_GRID_SIDE
PREFLIGHT_DEMONSTRATIONS = 10
PREFLIGHT_LANES = 8
PREFLIGHT_ENCODER_GRID_COUNT = 2 * PREFLIGHT_DEMONSTRATIONS + 1
PREFLIGHT_GRID_TOKEN_COUNT = PREFLIGHT_GRID_SIDE * PREFLIGHT_GRID_SIDE + 1
PREFLIGHT_MEMORY_LENGTH = PREFLIGHT_ENCODER_GRID_COUNT * PREFLIGHT_GRID_TOKEN_COUNT
PREFLIGHT_MASKED_TOKENS_PER_MICROBATCH = PREFLIGHT_GRID_SIDE * PREFLIGHT_GRID_SIDE
PREFLIGHT_MASKED_TOKENS_PER_UPDATE = (
    GRADIENT_ACCUMULATION * PREFLIGHT_MASKED_TOKENS_PER_MICROBATCH
)
PREFLIGHT_SAMPLE_EQUIVALENT_CALLS = PREFLIGHT_LANES * DENOISING_STEPS
PREFLIGHT_POST_VALIDATION_MARGIN_NS = 5 * 60 * 1_000_000_000

PREFLIGHT_OOM = "OOM"
PREFLIGHT_BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
PREFLIGHT_NONFINITE_LOGITS = "NONFINITE_LOGITS"
PREFLIGHT_PROJECTION_INCOMPLETE = "PROJECTION_INCOMPLETE"

_INFERENCE_FIELDS = {
    "schema",
    "inference_id",
    "lane_count",
    "denoising_steps",
    "encoder_batch_calls",
    "decoder_batch_calls",
    "sample_equivalent_forward_calls",
    "masked_token_predictions",
    "padded_encoder_batch_shape",
    "unpadded_memory_length",
    "cache_dtype",
    "decoder_batch_sizes",
    "mask_count_trace",
    "lane_trace_sha256",
    "final_output_keys",
    "unique_outputs",
    "h2d_ns",
    "d2h_ns",
    "encoder_gpu_ns",
    "encoder_wall_time_ns",
    "decoder_gpu_ns",
    "decoder_wall_time_ns",
    "cpu_sampling_ns",
    "hashing_ns",
    "inference_wall_time_ns",
    "cuda_peak_allocated_bytes",
    "cuda_peak_reserved_bytes",
}
_TRAINING_SUMMARY_FIELDS = {
    "updates",
    "microbatches_per_update",
    "total_microbatches",
    "masked_tokens_per_microbatch",
    "masked_token_predictions",
    "encoder_forward_calls",
    "decoder_forward_calls",
    "backward_calls",
    "ledger_row_count",
    "ledger_rows_sha256",
    "total_update_wall_ns",
    "total_update_cuda_event_ns",
    "peak_allocated_bytes",
    "peak_reserved_bytes",
    "learning_rate_hex",
    "mean_masked_cell_ce_hex",
    "gradient_norm_before_clip_hex",
}
_REPORT_FIELDS = {
    "schema",
    "report_id",
    "status",
    "fixture",
    "runtime",
    "model",
    "training",
    "inference",
    "validation_manifest_commitment",
    "overhead_cost_probe_id",
    "lock_handshake_artifact_sha256",
    "preflight_started_perf_counter_ns",
    "fresh_reconstruction_probe",
    "checkpoint_cost_probe",
    "final_selection_probe",
    "lock_handshake_probe",
    "budget_projection",
    "dataset_file_reads",
    "training_checkpoint_writes",
    "diagnostic_checkpoint_writes",
    "fallback_used",
    "rng_state_restored",
    "ledger_rows_sha256",
}
_FAILURE_REPORT_FIELDS = {
    "schema",
    "report_id",
    "status",
    "failure_code",
    "evidence_completeness",
    "fixture_id",
    "runtime",
    "validation_manifest_commitment",
    "lock_handshake_artifact_sha256",
    "preflight_started_perf_counter_ns",
    "completed_updates",
    "ledger_row_count",
    "ledger_rows_sha256",
    "training_summary",
    "inference_summary",
    "overhead_cost_probe_id",
    "budget_projection",
    "diagnostic_checkpoint_commitment",
    "dataset_file_reads",
    "training_checkpoint_writes",
    "diagnostic_checkpoint_writes",
    "fallback_used",
    "training_evidence_eligible",
    "same_run_retry_allowed",
}


def _strict_int(value: object, *, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise TypeError(f"{field} must be an integer >= {minimum}")
    return value


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        return False
    try:
        return len(bytes.fromhex(value)) == 32
    except ValueError:
        return False


def _constant_grid(color: int) -> Grid:
    if type(color) is not int or not 0 <= color < OUTPUT_COLOR_COUNT:
        raise ValueError("constant-grid color must be in 0..9")
    row = (color,) * PREFLIGHT_GRID_SIDE
    return (row,) * PREFLIGHT_GRID_SIDE


def _grid_is_constant(grid: Grid, color: int) -> bool:
    return (
        len(grid) == PREFLIGHT_GRID_SIDE
        and len(grid[0]) == PREFLIGHT_GRID_SIDE
        and all(cell == color for row in grid for cell in row)
    )


@dataclass(frozen=True, slots=True)
class SyntheticPreflightFixture:
    demonstrations: tuple[tuple[Grid, Grid], ...]
    query_input: Grid
    target_output: Grid
    payload: dict[str, Any]

    def __post_init__(self) -> None:
        if len(self.demonstrations) != PREFLIGHT_DEMONSTRATIONS:
            raise ValueError("preflight requires exactly ten demonstrations")
        for index, (input_grid, output_grid) in enumerate(self.demonstrations):
            if not _grid_is_constant(input_grid, index):
                raise ValueError("preflight demonstration input constant drifted")
            if not _grid_is_constant(output_grid, (index + 1) % 10):
                raise ValueError("preflight demonstration output constant drifted")
        if not _grid_is_constant(self.query_input, 0):
            raise ValueError("preflight query constant drifted")
        if not _grid_is_constant(self.target_output, 1):
            raise ValueError("preflight target constant drifted")
        validate_preflight_fixture_payload(self.payload)

    @property
    def fixture_id(self) -> str:
        return str(self.payload["fixture_id"])


def _fixture_payload() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": PREFLIGHT_FIXTURE_SCHEMA_VERSION,
        "grid_side": PREFLIGHT_GRID_SIDE,
        "target_cell_count": PREFLIGHT_MASKED_TOKENS_PER_MICROBATCH,
        "demonstration_count": PREFLIGHT_DEMONSTRATIONS,
        "encoder_grid_count": PREFLIGHT_ENCODER_GRID_COUNT,
        "encoder_grid_token_count": PREFLIGHT_GRID_TOKEN_COUNT,
        "expected_memory_length": PREFLIGHT_MEMORY_LENGTH,
        "demonstrations": [
            {
                "pair_index": index,
                "input_constant_color": index,
                "output_constant_color": (index + 1) % OUTPUT_COLOR_COUNT,
                "height": PREFLIGHT_GRID_SIDE,
                "width": PREFLIGHT_GRID_SIDE,
            }
            for index in range(PREFLIGHT_DEMONSTRATIONS)
        ],
        "query_constant_color": 0,
        "target_constant_color": 1,
        "training_mask_policy": "all_900_cells_every_microbatch",
        "training_updates": PREFLIGHT_UPDATES,
        "microbatches_per_update": GRADIENT_ACCUMULATION,
        "inference_lanes": PREFLIGHT_LANES,
        "inference_steps": DENOISING_STEPS,
    }
    payload["fixture_id"] = canonical_sha256(payload)
    return payload


def validate_preflight_fixture_payload(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("preflight fixture payload must be a dictionary")
    expected = _fixture_payload()
    if payload != expected:
        raise ValueError("preflight fixture payload drifted from its exact version")
    return dict(payload)


def build_preflight_fixture() -> SyntheticPreflightFixture:
    """Build the complete versioned fixture without opening any filesystem path."""

    demonstrations = tuple(
        (_constant_grid(index), _constant_grid((index + 1) % 10))
        for index in range(PREFLIGHT_DEMONSTRATIONS)
    )
    return SyntheticPreflightFixture(
        demonstrations=demonstrations,
        query_input=_constant_grid(0),
        target_output=_constant_grid(1),
        payload=_fixture_payload(),
    )


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def project_campaign_budget(
    *,
    measured_preflight_training_wall_ns: int,
    measured_preflight_total_wall_ns: int,
    preflight_started_perf_counter_ns: int,
    validation_episodes_per_pass: int,
    lock_handshake_probe: Mapping[str, object],
    fresh_reconstruction_probe: Mapping[str, object],
    checkpoint_cost_probe: Mapping[str, object],
    final_selection_probe: Mapping[str, object],
) -> dict[str, Any]:
    """Conservatively extrapolate measured maximum-context preflight rates."""

    training_wall = _strict_int(
        measured_preflight_training_wall_ns,
        field="measured_preflight_training_wall_ns",
        minimum=1,
    )
    total_wall = _strict_int(
        measured_preflight_total_wall_ns,
        field="measured_preflight_total_wall_ns",
        minimum=1,
    )
    validation_episodes = _strict_int(
        validation_episodes_per_pass,
        field="validation_episodes_per_pass",
        minimum=1,
    )
    if total_wall < training_wall:
        raise ValueError("preflight total wall time cannot be below training wall time")
    handshake = validate_lock_handshake_probe(dict(lock_handshake_probe))
    preflight_started = _strict_int(
        preflight_started_perf_counter_ns,
        field="preflight_started_perf_counter_ns",
        minimum=1,
    )
    handshake_completed = int(handshake["handshake_completed_perf_counter_ns"])
    if preflight_started < handshake_completed:
        raise ValueError("preflight started before the lock handshake completed")
    handshake_to_preflight = preflight_started - handshake_completed
    if handshake_to_preflight > MAX_HANDSHAKE_TO_PREFLIGHT_NS:
        raise ValueError("lock handshake is too old for this preflight attempt")
    run_id = str(handshake["run_id"])
    fresh = validate_fresh_reconstruction_probe(
        dict(fresh_reconstruction_probe), expected_run_id=run_id
    )
    checkpoint = validate_checkpoint_cost_probe(
        dict(checkpoint_cost_probe), expected_run_id=run_id
    )
    selection = validate_final_selection_probe(
        dict(final_selection_probe), expected_run_id=run_id
    )
    mean_update = _ceil_div(training_wall, PREFLIGHT_UPDATES)
    mean_microbatch = _ceil_div(
        training_wall, PREFLIGHT_UPDATES * GRADIENT_ACCUMULATION
    )
    primary_wall = mean_update * OPTIMIZER_UPDATES
    validation_calls = VALIDATION_PASS_COUNT * validation_episodes
    validation_wall = mean_microbatch * validation_calls
    checkpoint_wall = VALIDATION_PASS_COUNT * int(checkpoint["roundtrip_wall_ns"])
    projected_total = (
        int(handshake["wall_ns"])
        + handshake_to_preflight
        + total_wall
        + int(fresh["total_wall_ns"])
        + primary_wall
        + validation_wall
        + checkpoint_wall
        + int(selection["selection_wall_ns"])
        + PREFLIGHT_POST_VALIDATION_MARGIN_NS
    )
    payload: dict[str, Any] = {
        "schema": PREFLIGHT_PROJECTION_SCHEMA_VERSION,
        "projection_method": PREFLIGHT_PROJECTION_METHOD,
        "measured_preflight_training_wall_ns": training_wall,
        "measured_preflight_total_wall_ns": total_wall,
        "mean_training_update_wall_ns": mean_update,
        "mean_training_microbatch_wall_ns": mean_microbatch,
        "lock_handshake_probe_id": handshake["probe_id"],
        "measured_lock_setup_handshake_wall_ns": handshake["wall_ns"],
        "measured_handshake_to_preflight_start_ns": handshake_to_preflight,
        "fresh_reconstruction_probe_id": fresh["probe_id"],
        "measured_fresh_reconstruction_wall_ns": fresh["total_wall_ns"],
        "checkpoint_cost_probe_id": checkpoint["probe_id"],
        "measured_checkpoint_roundtrip_wall_ns": checkpoint[
            "roundtrip_wall_ns"
        ],
        "final_selection_probe_id": selection["probe_id"],
        "measured_final_selection_wall_ns": selection["selection_wall_ns"],
        "projected_primary_updates": OPTIMIZER_UPDATES,
        "projected_primary_wall_ns": primary_wall,
        "projected_validation_passes": VALIDATION_PASS_COUNT,
        "validation_episodes_per_pass": validation_episodes,
        "projected_validation_episode_calls": validation_calls,
        "projected_validation_wall_ns": validation_wall,
        "projected_checkpoint_writes": VALIDATION_PASS_COUNT,
        "projected_checkpoint_wall_ns": checkpoint_wall,
        "projected_final_selection_wall_ns": selection["selection_wall_ns"],
        "post_preflight_validation_margin_ns": (
            PREFLIGHT_POST_VALIDATION_MARGIN_NS
        ),
        "projected_campaign_wall_ns": projected_total,
        "budget_limit_ns": CAMPAIGN_GPU_BUDGET_NS,
        "budget_remaining_ns": CAMPAIGN_GPU_BUDGET_NS - projected_total,
        "budget_status": (
            "WITHIN_BUDGET"
            if projected_total <= CAMPAIGN_GPU_BUDGET_NS
            else PREFLIGHT_BUDGET_EXCEEDED
        ),
    }
    payload["projection_id"] = canonical_sha256(payload)
    return payload


def validate_preflight_ledger_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise TypeError("preflight ledger rows must be an ordered sequence")
    normalized = tuple(validate_training_cost_ledger_row(row) for row in rows)
    if len(normalized) != PREFLIGHT_UPDATES:
        raise ValueError("preflight ledger must contain exactly 100 update rows")
    for index, row in enumerate(normalized):
        expected_counts = {
            "optimizer_updates": 1,
            "microbatches": GRADIENT_ACCUMULATION,
            "arc2_episodes": 0,
            "rearc_episodes": 0,
            "encoder_forward_calls": GRADIENT_ACCUMULATION,
            "decoder_forward_calls": GRADIENT_ACCUMULATION,
            "backward_calls": GRADIENT_ACCUMULATION,
            "validation_episode_calls": 0,
            "validation_encoder_forward_calls": 0,
            "validation_decoder_forward_calls": 0,
            "checkpoint_writes": 0,
            "masked_token_predictions": PREFLIGHT_MASKED_TOKENS_PER_UPDATE,
        }
        if row["phase"] != "preflight_update":
            raise ValueError("preflight ledger contains a non-preflight phase")
        if row["event_index"] != index or row["optimizer_step"] != index + 1:
            raise ValueError("preflight ledger coordinates are not literal 0/1 based")
        if any(row[field] != expected for field, expected in expected_counts.items()):
            raise ValueError("preflight ledger semantic counts drifted")
    return normalized


def _build_synthetic_training_episodes(
    fixture: SyntheticPreflightFixture,
    *,
    optimizer_step: int,
) -> tuple[tuple[M04AEpisode, ...], tuple[dict[str, Any], ...]]:
    """Build and audit one all-maximum update through production episode types."""

    if type(optimizer_step) is not int or not 1 <= optimizer_step <= PREFLIGHT_UPDATES:
        raise ValueError(f"optimizer_step must be in 1..{PREFLIGHT_UPDATES}")
    masked = tuple(range(PREFLIGHT_MASKED_TOKENS_PER_MICROBATCH))
    identity_permutation = tuple(range(OUTPUT_COLOR_COUNT))
    descriptors = tuple(
        f"maximum-context-demonstration:{index}"
        for index in range(PREFLIGHT_DEMONSTRATIONS)
    )
    episodes: list[M04AEpisode] = []
    audit_rows: list[dict[str, Any]] = []
    for slot in range(GRADIENT_ACCUMULATION):
        # Reconstruct every example inside the measured update.  Production
        # build_training_episode likewise materializes transformed examples;
        # doing this for ten 30x30 demonstrations in every slot is the frozen
        # dataset-free upper-context preparation path.
        demonstrations = tuple(
            M04AExample.create(input_grid, output_grid)
            for input_grid, output_grid in fixture.demonstrations
        )
        source = ARC2_SOURCE if slot % 2 == 0 else REARC_SOURCE
        parent_id = f"preflight-maximum-context-{source}"
        episode = M04AEpisode(
            source=source,
            parent_id=parent_id,
            semantic_parent_id=parent_id,
            target_descriptor=f"maximum-context:{optimizer_step}:{slot}",
            demonstration_descriptors=descriptors,
            demonstrations=demonstrations,
            query_input=fixture.query_input,
            target_output=fixture.target_output,
            d4_index=0,
            color_permutation=identity_permutation,
            seed_u64=episode_seed(optimizer_step - 1, slot),
            masked_linear_indices=masked,
            corruption_kind="all_mask",
            optimizer_step=optimizer_step - 1,
            microbatch_slot=slot,
        )
        audit = make_training_mask_audit_row(
            episode,
            optimizer_step=optimizer_step,
        )
        if (
            len(episode.demonstrations) != PREFLIGHT_DEMONSTRATIONS
            or audit["microbatch_slot"] != slot
            or audit["masked_token_predictions"]
            != PREFLIGHT_MASKED_TOKENS_PER_MICROBATCH
        ):
            raise RuntimeError("maximum-context preflight episode/audit drifted")
        episodes.append(episode)
        audit_rows.append(audit)
    return tuple(episodes), tuple(audit_rows)


def _assert_no_gradients(model: GridCMLM) -> None:
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("preflight model gradients were not cleared")


def _run_training_fixture(
    model: GridCMLM,
    optimizer: torch.optim.AdamW,
    fixture: SyntheticPreflightFixture,
    *,
    ledger_sink: list[dict[str, Any]] | None = None,
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    ledger_rows = [] if ledger_sink is None else ledger_sink
    if ledger_rows:
        raise ValueError("preflight ledger sink must be empty at run start")
    losses: list[str] = []
    gradient_norms: list[str] = []
    learning_rates: list[str] = []
    for optimizer_step in range(1, PREFLIGHT_UPDATES + 1):
        learning_rate = learning_rate_for_update(optimizer_step)
        wall_started = time.perf_counter_ns()
        cpu_started = time.process_time_ns()
        episodes, audit_rows = _build_synthetic_training_episodes(
            fixture,
            optimizer_step=optimizer_step,
        )
        if (
            len(audit_rows) != GRADIENT_ACCUMULATION
            or sum(int(row["masked_token_predictions"]) for row in audit_rows)
            != PREFLIGHT_MASKED_TOKENS_PER_UPDATE
        ):
            raise RuntimeError("maximum-context preflight audit window drifted")
        outputs, timing = timed_cuda_call(
            lambda episodes=episodes, learning_rate=learning_rate: _execute_cuda_update(
                model,
                optimizer,
                episodes,
                learning_rate,
            )
        )
        mean_loss_tensor, gradient_norm_tensor = outputs
        mean_loss = float(mean_loss_tensor.item())
        gradient_norm = float(gradient_norm_tensor.item())
        if (
            not math.isfinite(mean_loss)
            or mean_loss < 0.0
            or not math.isfinite(gradient_norm)
            or gradient_norm < 0.0
        ):
            raise RuntimeError("preflight update produced a non-finite metric")
        _assert_no_gradients(model)
        assert_adamw_invariants(
            optimizer,
            model,
            expected_learning_rate=learning_rate,
            expected_completed_updates=optimizer_step,
        )
        wall_time_ns = max(time.perf_counter_ns() - wall_started, timing.gpu_ns)
        ledger_rows.append(
            make_training_cost_ledger_row(
                phase="preflight_update",
                event_index=optimizer_step - 1,
                optimizer_step=optimizer_step,
                optimizer_updates=1,
                microbatches=GRADIENT_ACCUMULATION,
                encoder_forward_calls=GRADIENT_ACCUMULATION,
                decoder_forward_calls=GRADIENT_ACCUMULATION,
                backward_calls=GRADIENT_ACCUMULATION,
                masked_token_predictions=PREFLIGHT_MASKED_TOKENS_PER_UPDATE,
                cuda_event_ns=timing.gpu_ns,
                cpu_time_ns=time.process_time_ns() - cpu_started,
                wall_time_ns=wall_time_ns,
                cuda_peak_allocated_bytes=timing.peak_allocated_bytes,
                cuda_peak_reserved_bytes=timing.peak_reserved_bytes,
            )
        )
        losses.append(mean_loss.hex())
        gradient_norms.append(gradient_norm.hex())
        learning_rates.append(learning_rate.hex())
    normalized = validate_preflight_ledger_rows(ledger_rows)
    training_summary = {
        "updates": PREFLIGHT_UPDATES,
        "microbatches_per_update": GRADIENT_ACCUMULATION,
        "total_microbatches": PREFLIGHT_UPDATES * GRADIENT_ACCUMULATION,
        "masked_tokens_per_microbatch": PREFLIGHT_MASKED_TOKENS_PER_MICROBATCH,
        "masked_token_predictions": (
            PREFLIGHT_UPDATES * PREFLIGHT_MASKED_TOKENS_PER_UPDATE
        ),
        "encoder_forward_calls": PREFLIGHT_UPDATES * GRADIENT_ACCUMULATION,
        "decoder_forward_calls": PREFLIGHT_UPDATES * GRADIENT_ACCUMULATION,
        "backward_calls": PREFLIGHT_UPDATES * GRADIENT_ACCUMULATION,
        "ledger_row_count": len(normalized),
        "ledger_rows_sha256": canonical_sha256(list(normalized)),
        "total_update_wall_ns": sum(int(row["wall_time_ns"]) for row in normalized),
        "total_update_cuda_event_ns": sum(
            int(row["cuda_event_ns"]) for row in normalized
        ),
        "peak_allocated_bytes": max(
            int(row["cuda_peak_allocated_bytes"]) for row in normalized
        ),
        "peak_reserved_bytes": max(
            int(row["cuda_peak_reserved_bytes"]) for row in normalized
        ),
        "learning_rate_hex": learning_rates,
        "mean_masked_cell_ce_hex": losses,
        "gradient_norm_before_clip_hex": gradient_norms,
    }
    return normalized, training_summary


def _run_cached_inference_fixture(
    model: Any,
    fixture: SyntheticPreflightFixture,
    runtime: SamplerRuntime,
) -> dict[str, Any]:
    """Exercise eight lanes through the production sampler's numeric primitives."""

    if not isinstance(runtime, SamplerRuntime):
        raise TypeError("runtime must be a SamplerRuntime")
    if production_sampler._model_device(model) != runtime.device:
        raise ValueError("preflight model and sampler runtime devices differ")
    task_cpu = tokenize_task_memory(
        fixture.demonstrations, fixture.query_input, device="cpu"
    )
    shape_proposal_id = canonical_sha256(
        {
            "fixture_id": fixture.fixture_id,
            "height": PREFLIGHT_GRID_SIDE,
            "width": PREFLIGHT_GRID_SIDE,
        }
    )
    lanes: list[Any] = []
    for local_lane in range(PREFLIGHT_LANES):
        seed = lane_seed(fixture.fixture_id, 0, shape_proposal_id, local_lane)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        lanes.append(
            production_sampler._LaneState(
                shape_order=0,
                shape_proposal_id=shape_proposal_id,
                height=PREFLIGHT_GRID_SIDE,
                width=PREFLIGHT_GRID_SIDE,
                global_lane=local_lane,
                local_lane=local_lane,
                greedy=local_lane == 0,
                seed_u64=seed,
                generator=generator,
                cells=[production_sampler._MASK]
                * PREFLIGHT_MASKED_TOKENS_PER_MICROBATCH,
                started_ns=time.perf_counter_ns(),
            )
        )

    previous_training = bool(getattr(model, "training", False))
    model.eval()
    endpoint_started = time.perf_counter_ns()
    h2d_ns = 0
    d2h_ns = 0
    decoder_gpu_ns = 0
    decoder_wall_ns = 0
    cpu_sampling_ns = 0
    hashing_ns = 0
    peak_allocated = 0
    peak_reserved = 0
    decoder_batch_sizes: list[int] = []
    try:
        with production_sampler._frozen_inference_context(runtime.device):
            task_batch, transfer_ns = production_sampler._move_token_batch(
                task_cpu, device=runtime.device
            )
            h2d_ns += transfer_ns
            encoder = production_sampler._measure_forward(
                lambda: model.encode_task_memory(task_batch), device=runtime.device
            )
            cache = production_sampler._cache_for_runtime(encoder.value, runtime)
            cache_tensor = production_sampler._memory_tensor(cache)
            memory_length = production_sampler._memory_length(cache)
            peak_allocated = max(peak_allocated, encoder.peak_allocated_bytes)
            peak_reserved = max(peak_reserved, encoder.peak_reserved_bytes)

            for step in range(1, DENOISING_STEPS + 1):
                targets_cpu = tokenize_target_batch(
                    tuple(production_sampler._target_grid(lane) for lane in lanes),
                    device="cpu",
                )
                targets, transfer_ns = production_sampler._move_token_batch(
                    targets_cpu, device=runtime.device
                )
                h2d_ns += transfer_ns
                decoder = production_sampler._measure_forward(
                    lambda targets=targets: model.decode_target(targets, cache),
                    device=runtime.device,
                )
                logits = getattr(decoder.value, "logits", None)
                if not isinstance(logits, Tensor) or tuple(logits.shape) != (
                    PREFLIGHT_LANES,
                    PREFLIGHT_MASKED_TOKENS_PER_MICROBATCH,
                    OUTPUT_COLOR_COUNT,
                ):
                    raise ValueError("preflight sampler decoder logits shape drifted")
                logits_cpu, transfer_ns = production_sampler._logits_to_cpu_float64(
                    logits, device=runtime.device
                )
                d2h_ns += transfer_ns
                if not bool(torch.isfinite(logits_cpu).all().item()):
                    raise PreflightFailure(
                        PREFLIGHT_NONFINITE_LOGITS,
                        "preflight sampler produced non-finite logits",
                    )
                decoder_batch_sizes.append(PREFLIGHT_LANES)
                decoder_gpu_ns += decoder.gpu_forward_ns
                decoder_wall_ns += decoder.wall_time_ns
                peak_allocated = max(peak_allocated, decoder.peak_allocated_bytes)
                peak_reserved = max(peak_reserved, decoder.peak_reserved_bytes)
                for lane_offset, lane in enumerate(lanes):
                    sampling_started = time.perf_counter_ns()
                    step_row, step_hashing_ns = production_sampler._sample_lane_step(
                        lane,
                        step=step,
                        logits=logits_cpu[lane_offset],
                    )
                    step_wall_ns = time.perf_counter_ns() - sampling_started
                    if step_hashing_ns > step_wall_ns:
                        raise RuntimeError("preflight sampler hashing time exceeds step wall time")
                    hashing_ns += step_hashing_ns
                    cpu_sampling_ns += step_wall_ns - step_hashing_ns
                    lane.steps.append(step_row)
    finally:
        model.train(previous_training)

    lane_trace_sha256: list[str] = []
    final_output_keys: list[str] = []
    for lane in lanes:
        final_grid = production_sampler._final_grid(lane)
        trace = {
            "schema": PREFLIGHT_LANE_TRACE_SCHEMA_VERSION,
            "fixture_id": fixture.fixture_id,
            "shape_proposal_id": shape_proposal_id,
            "local_lane": lane.local_lane,
            "greedy": lane.greedy,
            "seed_u64": lane.seed_u64,
            "initial_masked_indices": list(
                range(PREFLIGHT_MASKED_TOKENS_PER_MICROBATCH)
            ),
            "steps": lane.steps,
            "final_grid": grid_to_lists(final_grid),
            "output_key": grid_key(final_grid),
        }
        lane_trace_sha256.append(canonical_sha256(trace))
        final_output_keys.append(grid_key(final_grid))
    cache_dtype = production_sampler._dtype_name(cache_tensor.dtype)
    summary: dict[str, Any] = {
        "schema": PREFLIGHT_INFERENCE_SCHEMA_VERSION,
        "lane_count": PREFLIGHT_LANES,
        "denoising_steps": DENOISING_STEPS,
        "encoder_batch_calls": 1,
        "decoder_batch_calls": DENOISING_STEPS,
        "sample_equivalent_forward_calls": PREFLIGHT_SAMPLE_EQUIVALENT_CALLS,
        "masked_token_predictions": (
            PREFLIGHT_LANES
            * sum(mask_count_trace(PREFLIGHT_MASKED_TOKENS_PER_MICROBATCH)[:-1])
        ),
        "padded_encoder_batch_shape": list(task_cpu.token_ids.shape),
        "unpadded_memory_length": memory_length,
        "cache_dtype": cache_dtype,
        "decoder_batch_sizes": decoder_batch_sizes,
        "mask_count_trace": list(
            mask_count_trace(PREFLIGHT_MASKED_TOKENS_PER_MICROBATCH)
        ),
        "lane_trace_sha256": lane_trace_sha256,
        "final_output_keys": final_output_keys,
        "unique_outputs": len(set(final_output_keys)),
        "h2d_ns": h2d_ns,
        "d2h_ns": d2h_ns,
        "encoder_gpu_ns": encoder.gpu_forward_ns,
        "encoder_wall_time_ns": encoder.wall_time_ns,
        "decoder_gpu_ns": decoder_gpu_ns,
        "decoder_wall_time_ns": decoder_wall_ns,
        "cpu_sampling_ns": cpu_sampling_ns,
        "hashing_ns": hashing_ns,
        "inference_wall_time_ns": time.perf_counter_ns() - endpoint_started,
        "cuda_peak_allocated_bytes": peak_allocated,
        "cuda_peak_reserved_bytes": peak_reserved,
    }
    summary["inference_id"] = canonical_sha256(summary)
    return summary


def _validate_inference_summary(
    summary: object, *, require_cuda: bool
) -> dict[str, Any]:
    if not isinstance(summary, dict):
        raise TypeError("preflight inference summary must be a dictionary")
    if set(summary) != _INFERENCE_FIELDS:
        raise ValueError("preflight inference fields do not match the exact schema")
    expected_scalars = {
        "schema": PREFLIGHT_INFERENCE_SCHEMA_VERSION,
        "lane_count": PREFLIGHT_LANES,
        "denoising_steps": DENOISING_STEPS,
        "encoder_batch_calls": 1,
        "decoder_batch_calls": DENOISING_STEPS,
        "sample_equivalent_forward_calls": PREFLIGHT_SAMPLE_EQUIVALENT_CALLS,
        "masked_token_predictions": (
            PREFLIGHT_LANES
            * sum(mask_count_trace(PREFLIGHT_MASKED_TOKENS_PER_MICROBATCH)[:-1])
        ),
        "padded_encoder_batch_shape": [
            PREFLIGHT_ENCODER_GRID_COUNT,
            PREFLIGHT_GRID_TOKEN_COUNT,
        ],
        "unpadded_memory_length": PREFLIGHT_MEMORY_LENGTH,
        "decoder_batch_sizes": [PREFLIGHT_LANES] * DENOISING_STEPS,
        "mask_count_trace": list(
            mask_count_trace(PREFLIGHT_MASKED_TOKENS_PER_MICROBATCH)
        ),
    }
    if any(summary.get(field) != value for field, value in expected_scalars.items()):
        raise ValueError("preflight inference semantic closure drifted")
    if require_cuda and summary.get("cache_dtype") != "bfloat16":
        raise ValueError("CUDA preflight task-memory cache must be bfloat16")
    if not require_cuda and summary.get("cache_dtype") not in {
        "float32",
        "bfloat16",
    }:
        raise ValueError("test preflight cache dtype is unsupported")
    for field in ("lane_trace_sha256", "final_output_keys"):
        values = summary.get(field)
        if not isinstance(values, list) or len(values) != PREFLIGHT_LANES:
            raise ValueError(f"preflight {field} must contain eight rows")
    if any(not _is_sha256(value) for value in summary["lane_trace_sha256"]):
        raise ValueError("preflight lane traces must be bound by SHA-256")
    if any(
        not isinstance(value, str) or not value
        for value in summary["final_output_keys"]
    ):
        raise ValueError("preflight final output keys must be non-empty strings")
    if summary.get("unique_outputs") != len(set(summary["final_output_keys"])):
        raise ValueError("preflight inference unique-output count does not close")
    measurement_fields = (
        "h2d_ns",
        "d2h_ns",
        "encoder_gpu_ns",
        "encoder_wall_time_ns",
        "decoder_gpu_ns",
        "decoder_wall_time_ns",
        "cpu_sampling_ns",
        "hashing_ns",
        "inference_wall_time_ns",
        "cuda_peak_allocated_bytes",
        "cuda_peak_reserved_bytes",
    )
    for field in measurement_fields:
        _strict_int(summary[field], field=field)
    if summary["encoder_gpu_ns"] > summary["encoder_wall_time_ns"]:
        raise ValueError("preflight encoder GPU time exceeds wall time")
    if summary["decoder_gpu_ns"] > summary["decoder_wall_time_ns"]:
        raise ValueError("preflight decoder GPU time exceeds wall time")
    sequential_components = (
        summary["h2d_ns"]
        + summary["d2h_ns"]
        + summary["encoder_wall_time_ns"]
        + summary["decoder_wall_time_ns"]
        + summary["cpu_sampling_ns"]
        + summary["hashing_ns"]
    )
    if sequential_components > summary["inference_wall_time_ns"]:
        raise ValueError("preflight inference endpoint wall time does not close")
    semantic = dict(summary)
    inference_id = semantic.pop("inference_id", None)
    if inference_id != canonical_sha256(semantic):
        raise ValueError("preflight inference_id mismatch")
    return dict(summary)


def _rng_states_equal(
    left: Mapping[str, object], right: Mapping[str, object]
) -> bool:
    left_cpu = left.get("torch_cpu")
    right_cpu = right.get("torch_cpu")
    left_cuda = left.get("torch_cuda")
    right_cuda = right.get("torch_cuda")
    return (
        left.get("schema") == right.get("schema")
        and left.get("optimizer_step") == right.get("optimizer_step")
        and isinstance(left_cpu, Tensor)
        and isinstance(right_cpu, Tensor)
        and torch.equal(left_cpu, right_cpu)
        and isinstance(left_cuda, list)
        and isinstance(right_cuda, list)
        and len(left_cuda) == len(right_cuda)
        and all(
            isinstance(left_state, Tensor)
            and isinstance(right_state, Tensor)
            and torch.equal(left_state, right_state)
            for left_state, right_state in zip(left_cuda, right_cuda, strict=True)
        )
    )


def _restore_rng_and_cleanup(
    rng_before: Mapping[str, object],
    *,
    active_exception: BaseException | None,
) -> None:
    """Always attempt RNG restoration before fallible CUDA cache cleanup."""

    restoration_error: Exception | None = None
    cleanup_error: Exception | None = None
    try:
        restore_rng_state(rng_before)
        rng_restored = capture_rng_state(optimizer_step=0)
        if not _rng_states_equal(rng_before, rng_restored):
            raise RuntimeError("preflight failed to restore CPU/CUDA RNG state")
    except Exception as exc:
        restoration_error = exc
    try:
        torch.cuda.empty_cache()
    except Exception as exc:
        cleanup_error = exc
    if active_exception is not None:
        if restoration_error is not None:
            setattr(active_exception, "rng_restoration_error", repr(restoration_error))
        if cleanup_error is not None:
            setattr(active_exception, "cuda_cleanup_error", repr(cleanup_error))
    elif restoration_error is not None:
        if cleanup_error is not None:
            setattr(restoration_error, "cuda_cleanup_error", repr(cleanup_error))
        raise restoration_error
    elif cleanup_error is not None:
        raise cleanup_error


def _make_failure_report(
    *,
    failure_code: str,
    fixture_id: str,
    completed_updates: int,
    ledger_rows: Sequence[Mapping[str, Any]],
    projection: Mapping[str, Any] | None = None,
    training_summary: Mapping[str, Any] | None = None,
    inference_summary: Mapping[str, Any] | None = None,
    overhead_cost_probe_report: Mapping[str, Any] | None = None,
    diagnostic_checkpoint_snapshot: bytes | None = None,
    runtime_attestation: FrozenRuntimeAttestation | None = None,
    validation_manifest_commitment: Mapping[str, Any] | None = None,
    lock_handshake_artifact_sha256: str | None = None,
    preflight_started_perf_counter_ns: int | None = None,
) -> dict[str, Any]:
    full_budget_probe = failure_code == PREFLIGHT_BUDGET_EXCEEDED
    if full_budget_probe:
        if (
            projection is None
            or training_summary is None
            or inference_summary is None
            or overhead_cost_probe_report is None
            or diagnostic_checkpoint_snapshot is None
            or runtime_attestation is None
            or validation_manifest_commitment is None
            or lock_handshake_artifact_sha256 is None
            or preflight_started_perf_counter_ns is None
        ):
            raise ValueError(
                "budget-exceeded failure requires the complete measured probe"
            )
        checkpoint_probe = overhead_cost_probe_report["checkpoint_cost_probe"]
        diagnostic_commitment: dict[str, Any] | None = {
            "artifact_filename": overhead_cost_probe_report[
                "checkpoint_artifact_filename"
            ],
            "artifact_status": overhead_cost_probe_report[
                "checkpoint_artifact_status"
            ],
            "artifact_disposition": checkpoint_probe["artifact_disposition"],
            "sha256": checkpoint_probe["checkpoint_sha256"],
            "bytes": checkpoint_probe["checkpoint_bytes"],
            "selectable_checkpoint_created": checkpoint_probe[
                "selectable_checkpoint_created"
            ],
        }
        if (
            len(diagnostic_checkpoint_snapshot) != diagnostic_commitment["bytes"]
            or hashlib.sha256(diagnostic_checkpoint_snapshot).hexdigest()
            != diagnostic_commitment["sha256"]
        ):
            raise ValueError(
                "budget failure diagnostic checkpoint differs from its cost probe"
            )
        runtime: dict[str, Any] | None = {
            "run_id": runtime_attestation.run_id,
            "gpu_uuid": runtime_attestation.gpu_uuid,
            "logical_device_index": runtime_attestation.logical_device_index,
        }
        overhead_cost_probe_id: str | None = str(
            overhead_cost_probe_report["probe_id"]
        )
    else:
        runtime = None
        overhead_cost_probe_id = None
        diagnostic_commitment = None
    report: dict[str, Any] = {
        "schema": PREFLIGHT_FAILURE_REPORT_SCHEMA_VERSION,
        "status": failure_code,
        "failure_code": failure_code,
        "evidence_completeness": (
            "full_budget_probe" if full_budget_probe else "partial"
        ),
        "fixture_id": fixture_id,
        "runtime": runtime,
        "validation_manifest_commitment": (
            None
            if validation_manifest_commitment is None
            else dict(validation_manifest_commitment)
        ),
        "lock_handshake_artifact_sha256": lock_handshake_artifact_sha256,
        "preflight_started_perf_counter_ns": preflight_started_perf_counter_ns,
        "completed_updates": completed_updates,
        "ledger_row_count": len(ledger_rows),
        "ledger_rows_sha256": canonical_sha256(list(ledger_rows)),
        "training_summary": (
            None if training_summary is None else dict(training_summary)
        ),
        "inference_summary": (
            None if inference_summary is None else dict(inference_summary)
        ),
        "overhead_cost_probe_id": overhead_cost_probe_id,
        "budget_projection": None if projection is None else dict(projection),
        "diagnostic_checkpoint_commitment": diagnostic_commitment,
        "dataset_file_reads": 0,
        "training_checkpoint_writes": 0,
        "diagnostic_checkpoint_writes": 1 if full_budget_probe else 0,
        "fallback_used": False,
        "training_evidence_eligible": False,
        "same_run_retry_allowed": False,
    }
    report["report_id"] = canonical_sha256(report)
    return report


class PreflightFailure(RuntimeError):
    """Terminal preflight failure; it authorizes no smaller-context fallback."""

    def __init__(
        self,
        failure_code: str,
        message: str,
        *,
        report: Mapping[str, Any] | None = None,
        ledger_rows: Sequence[Mapping[str, Any]] = (),
        overhead_cost_probe_report: Mapping[str, Any] | None = None,
        training_summary: Mapping[str, Any] | None = None,
        inference_summary: Mapping[str, Any] | None = None,
        diagnostic_checkpoint_snapshot: bytes | None = None,
    ) -> None:
        super().__init__(f"{failure_code}: {message}")
        self.failure_code = failure_code
        self.report = None if report is None else dict(report)
        self.ledger_rows = tuple(dict(row) for row in ledger_rows)
        self.overhead_cost_probe_report = (
            None
            if overhead_cost_probe_report is None
            else dict(overhead_cost_probe_report)
        )
        self.training_summary = (
            None if training_summary is None else dict(training_summary)
        )
        self.inference_summary = (
            None if inference_summary is None else dict(inference_summary)
        )
        if diagnostic_checkpoint_snapshot is not None and type(
            diagnostic_checkpoint_snapshot
        ) is not bytes:
            raise TypeError("diagnostic checkpoint snapshot must be immutable bytes")
        self.diagnostic_checkpoint_snapshot = diagnostic_checkpoint_snapshot
        self.diagnostic_checkpoint_commitment = (
            None
            if self.report is None
            else self.report.get("diagnostic_checkpoint_commitment")
        )


def validate_preflight_failure_report(
    report: object,
    ledger_rows: Sequence[Mapping[str, Any]],
    *,
    overhead_cost_probe_report: Mapping[str, Any] | None = None,
    diagnostic_checkpoint_snapshot: bytes | None = None,
    expected_config_sha256: str | None = None,
    expected_runtime_source_sha256: str | None = None,
    expected_test_source_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate terminal failure evidence without upgrading it to a PASS gate."""

    if not isinstance(report, dict) or set(report) != _FAILURE_REPORT_FIELDS:
        raise ValueError("preflight failure report fields do not match the exact schema")
    if report.get("schema") != PREFLIGHT_FAILURE_REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported preflight failure report schema")
    failure_code = report.get("failure_code")
    if failure_code not in {
        PREFLIGHT_OOM,
        PREFLIGHT_BUDGET_EXCEEDED,
        PREFLIGHT_NONFINITE_LOGITS,
        PREFLIGHT_PROJECTION_INCOMPLETE,
    } or report.get("status") != failure_code:
        raise ValueError("preflight failure code/status is invalid")
    if report.get("fixture_id") != _fixture_payload()["fixture_id"]:
        raise ValueError("preflight failure fixture_id drifted from frozen fixture")
    if (
        report.get("dataset_file_reads") != 0
        or report.get("training_checkpoint_writes") != 0
        or report.get("diagnostic_checkpoint_writes")
        != (1 if failure_code == PREFLIGHT_BUDGET_EXCEEDED else 0)
        or report.get("fallback_used") is not False
        or report.get("training_evidence_eligible") is not False
        or report.get("same_run_retry_allowed") is not False
    ):
        raise ValueError("preflight failure disposition is unsafe")
    if isinstance(ledger_rows, (str, bytes)) or not isinstance(
        ledger_rows, Sequence
    ):
        raise TypeError("preflight failure ledger must be an ordered sequence")
    normalized = tuple(validate_training_cost_ledger_row(row) for row in ledger_rows)
    completed_updates = _strict_int(
        report.get("completed_updates"), field="completed_updates"
    )
    if (
        completed_updates != len(normalized)
        or completed_updates > PREFLIGHT_UPDATES
        or report.get("ledger_row_count") != len(normalized)
        or report.get("ledger_rows_sha256")
        != canonical_sha256(list(normalized))
    ):
        raise ValueError("preflight failure ledger commitment does not close")
    expected_partial_counts = {
        "optimizer_updates": 1,
        "microbatches": GRADIENT_ACCUMULATION,
        "arc2_episodes": 0,
        "rearc_episodes": 0,
        "encoder_forward_calls": GRADIENT_ACCUMULATION,
        "decoder_forward_calls": GRADIENT_ACCUMULATION,
        "backward_calls": GRADIENT_ACCUMULATION,
        "validation_episode_calls": 0,
        "validation_encoder_forward_calls": 0,
        "validation_decoder_forward_calls": 0,
        "checkpoint_writes": 0,
        "masked_token_predictions": PREFLIGHT_MASKED_TOKENS_PER_UPDATE,
    }
    for index, row in enumerate(normalized):
        if (
            row["phase"] != "preflight_update"
            or row["event_index"] != index
            or row["optimizer_step"] != index + 1
            or any(
                row[field] != value
                for field, value in expected_partial_counts.items()
            )
        ):
            raise ValueError("preflight failure ledger semantic counts drifted")
    semantic = dict(report)
    report_id = semantic.pop("report_id", None)
    if not _is_sha256(report_id) or report_id != canonical_sha256(semantic):
        raise ValueError("preflight failure report_id mismatch")

    completeness = report.get("evidence_completeness")
    full_budget_probe = failure_code == PREFLIGHT_BUDGET_EXCEEDED
    if completeness != ("full_budget_probe" if full_budget_probe else "partial"):
        raise ValueError("preflight failure evidence-completeness state drifted")
    if not full_budget_probe:
        nullable = (
            "runtime",
            "validation_manifest_commitment",
            "lock_handshake_artifact_sha256",
            "preflight_started_perf_counter_ns",
            "training_summary",
            "inference_summary",
            "overhead_cost_probe_id",
            "budget_projection",
            "diagnostic_checkpoint_commitment",
        )
        if any(report[field] is not None for field in nullable):
            raise ValueError("partial preflight failure contains full-probe fields")
        if overhead_cost_probe_report is not None or diagnostic_checkpoint_snapshot is not None:
            raise ValueError("partial preflight failure has uncommitted probe artifacts")
        return dict(report)

    normalized = validate_preflight_ledger_rows(normalized)
    runtime = report.get("runtime")
    if (
        not isinstance(runtime, dict)
        or set(runtime) != {"run_id", "gpu_uuid", "logical_device_index"}
        or not isinstance(runtime["run_id"], str)
        or not runtime["run_id"]
        or not isinstance(runtime["gpu_uuid"], str)
        or not runtime["gpu_uuid"]
        or runtime["logical_device_index"] != 0
    ):
        raise ValueError("budget failure runtime identity is invalid")
    embedded_commitment = report.get("validation_manifest_commitment")
    commitment_fields = {
        "schema",
        "commitment_id",
        "outer_artifact_manifest_sha256",
        "jsonl_sha256",
        "summary_id",
        "row_count",
    }
    if (
        not isinstance(embedded_commitment, dict)
        or set(embedded_commitment) != commitment_fields
        or not _is_sha256(
            embedded_commitment.get("outer_artifact_manifest_sha256")
        )
        or not _is_sha256(embedded_commitment.get("jsonl_sha256"))
        or not _is_sha256(embedded_commitment.get("summary_id"))
        or type(embedded_commitment.get("row_count")) is not int
        or embedded_commitment["row_count"] < 1
    ):
        raise ValueError("budget failure validation commitment is invalid")
    commitment_semantic = dict(embedded_commitment)
    commitment_id = commitment_semantic.pop("commitment_id")
    if commitment_id != canonical_sha256(commitment_semantic):
        raise ValueError("budget failure validation commitment_id mismatch")
    if not _is_sha256(report.get("lock_handshake_artifact_sha256")):
        raise ValueError("budget failure lacks its lock-handshake commitment")
    preflight_started = _strict_int(
        report.get("preflight_started_perf_counter_ns"),
        field="preflight_started_perf_counter_ns",
        minimum=1,
    )
    training = report.get("training_summary")
    if not isinstance(training, dict) or set(training) != _TRAINING_SUMMARY_FIELDS:
        raise ValueError("budget failure training summary schema is invalid")
    expected_training_bindings = {
        "updates": PREFLIGHT_UPDATES,
        "ledger_row_count": PREFLIGHT_UPDATES,
        "ledger_rows_sha256": canonical_sha256(list(normalized)),
        "total_update_wall_ns": sum(int(row["wall_time_ns"]) for row in normalized),
        "total_update_cuda_event_ns": sum(
            int(row["cuda_event_ns"]) for row in normalized
        ),
        "peak_allocated_bytes": max(
            int(row["cuda_peak_allocated_bytes"]) for row in normalized
        ),
        "peak_reserved_bytes": max(
            int(row["cuda_peak_reserved_bytes"]) for row in normalized
        ),
    }
    if any(training.get(key) != value for key, value in expected_training_bindings.items()):
        raise ValueError("budget failure training summary does not close from ledger")
    _validate_inference_summary(report.get("inference_summary"), require_cuda=True)
    if overhead_cost_probe_report is None:
        raise ValueError("budget failure lacks the complete overhead report")
    cost_report = validate_campaign_overhead_cost_probe_report(
        dict(overhead_cost_probe_report),
        expected_run_id=runtime["run_id"],
        expected_config_sha256=expected_config_sha256,
        expected_runtime_source_sha256=expected_runtime_source_sha256,
        expected_test_source_sha256=expected_test_source_sha256,
        require_production_cuda=all(
            value is not None
            for value in (
                expected_config_sha256,
                expected_runtime_source_sha256,
                expected_test_source_sha256,
            )
        ),
    )
    if report.get("overhead_cost_probe_id") != cost_report["probe_id"]:
        raise ValueError("budget failure does not bind its overhead report")
    projection = report.get("budget_projection")
    if not isinstance(projection, dict):
        raise TypeError("budget failure projection must be a dictionary")
    expected_projection = project_campaign_budget(
        measured_preflight_training_wall_ns=int(training["total_update_wall_ns"]),
        measured_preflight_total_wall_ns=projection[
            "measured_preflight_total_wall_ns"
        ],
        preflight_started_perf_counter_ns=preflight_started,
        validation_episodes_per_pass=projection["validation_episodes_per_pass"],
        lock_handshake_probe=cost_report["lock_handshake_probe"],
        fresh_reconstruction_probe=cost_report["fresh_reconstruction_probe"],
        checkpoint_cost_probe=cost_report["checkpoint_cost_probe"],
        final_selection_probe=cost_report["final_selection_probe"],
    )
    if projection != expected_projection or projection["budget_status"] != (
        PREFLIGHT_BUDGET_EXCEEDED
    ):
        raise ValueError("budget failure projection does not reproduce")
    commitment = report.get("diagnostic_checkpoint_commitment")
    checkpoint_probe = cost_report["checkpoint_cost_probe"]
    expected_commitment = {
        "artifact_filename": cost_report["checkpoint_artifact_filename"],
        "artifact_status": cost_report["checkpoint_artifact_status"],
        "artifact_disposition": checkpoint_probe["artifact_disposition"],
        "sha256": checkpoint_probe["checkpoint_sha256"],
        "bytes": checkpoint_probe["checkpoint_bytes"],
        "selectable_checkpoint_created": False,
    }
    if commitment != expected_commitment or diagnostic_checkpoint_snapshot is None:
        raise ValueError("budget failure diagnostic checkpoint commitment drifted")
    if (
        type(diagnostic_checkpoint_snapshot) is not bytes
        or len(diagnostic_checkpoint_snapshot) != expected_commitment["bytes"]
        or hashlib.sha256(diagnostic_checkpoint_snapshot).hexdigest()
        != expected_commitment["sha256"]
    ):
        raise ValueError("budget failure diagnostic checkpoint bytes drifted")
    return dict(report)


@dataclass(frozen=True, slots=True)
class PreflightResult:
    report: dict[str, Any]
    ledger_rows: tuple[dict[str, Any], ...]
    validation_manifest_commitment: ValidationManifestCommitment
    launch_plan_artifact: LaunchPlanArtifact
    overhead_cost_probe_report: dict[str, Any]
    diagnostic_checkpoint_path: Path
    diagnostic_checkpoint_snapshot: bytes
    expected_config_sha256: str
    expected_runtime_source_sha256: str
    expected_test_source_sha256: str
    expected_run_root: str
    held_lock_handshake: HeldGPULockHandshake

    def __post_init__(self) -> None:
        commitment = validate_validation_manifest_commitment(
            self.validation_manifest_commitment
        )
        if type(self.launch_plan_artifact) is not LaunchPlanArtifact:
            raise TypeError("preflight result requires a committed launch-plan artifact")
        launch_plan = self.launch_plan_artifact.payload
        runtime = self.report.get("runtime")
        if not isinstance(runtime, dict) or not isinstance(runtime.get("run_id"), str):
            raise ValueError("preflight result lacks its runtime run_id")
        lock_payload = assert_held_gpu_lock(self.held_lock_handshake)
        if (
            lock_payload["run_id"] != runtime["run_id"]
            or lock_payload["gpu_uuid"] != runtime.get("gpu_uuid")
            or self.report.get("lock_handshake_artifact_sha256")
            != self.held_lock_handshake.artifact.artifact_sha256
            or self.report.get("preflight_started_perf_counter_ns")
            != self.held_lock_handshake.preflight_started_perf_counter_ns
            or make_cost_probe_lock_handshake(self.held_lock_handshake)
            != self.report.get("lock_handshake_probe")
            or self.launch_plan_artifact.artifact_sha256
            != self.held_lock_handshake.artifact.expected_launch_plan_sha256
        ):
            raise ValueError("preflight result differs from its held GPU lock")
        for field in (
            "expected_config_sha256",
            "expected_runtime_source_sha256",
            "expected_test_source_sha256",
        ):
            if not _is_sha256(getattr(self, field)):
                raise ValueError(f"{field} must be an external SHA-256 commitment")
        if self.expected_config_sha256 != training_config_sha256():
            raise ValueError("preflight result config commitment drifted")
        if not isinstance(self.expected_run_root, str):
            raise TypeError("preflight expected_run_root must be a POSIX string")
        if (
            self.expected_run_root != launch_plan["run_root"]
            or runtime["run_id"] != launch_plan["run_id"]
            or self.expected_config_sha256 != launch_plan["training_config_sha256"]
            or self.expected_runtime_source_sha256
            != launch_plan["runtime_source_fingerprint_sha256"]
            or self.expected_test_source_sha256
            != launch_plan["test_source_fingerprint_sha256"]
            or commitment != launch_plan["validation_manifest_commitment"]
            or self.held_lock_handshake.artifact.expected_launcher_sha256
            != launch_plan["expected_input_artifacts"]["remote_launcher.py"]
            or self.held_lock_handshake.artifact.expected_remote_project_root
            != launch_plan["remote_project_root"]
            or self.held_lock_handshake.artifact.expected_attempt_nonce
            != launch_plan["attempt_nonce"]
        ):
            raise ValueError("preflight result differs from its committed launch plan")
        cost_report = validate_campaign_overhead_cost_probe_report(
            self.overhead_cost_probe_report,
            expected_run_id=runtime["run_id"],
            expected_config_sha256=self.expected_config_sha256,
            expected_runtime_source_sha256=self.expected_runtime_source_sha256,
            expected_test_source_sha256=self.expected_test_source_sha256,
            require_production_cuda=True,
        )
        validate_preflight_report(
            self.report,
            self.ledger_rows,
            validation_manifest_commitment=self.validation_manifest_commitment,
            overhead_cost_probe_report=cost_report,
            expected_config_sha256=self.expected_config_sha256,
            expected_runtime_source_sha256=self.expected_runtime_source_sha256,
            expected_test_source_sha256=self.expected_test_source_sha256,
            lock_handshake_artifact=self.held_lock_handshake.artifact,
            launch_plan_artifact=self.launch_plan_artifact,
            expected_run_root=self.expected_run_root,
        )
        validate_preflight_evidence_artifact(
            self.report,
            self.ledger_rows,
            validation_episodes_per_pass=int(commitment["row_count"]),
            expected_validation_outer_manifest_sha256=str(
                commitment["outer_artifact_manifest_sha256"]
            ),
            expected_validation_jsonl_sha256=str(commitment["jsonl_sha256"]),
            expected_validation_summary_id=str(commitment["summary_id"]),
            overhead_cost_probe_report=cost_report,
            diagnostic_checkpoint_snapshot=self.diagnostic_checkpoint_snapshot,
            lock_handshake_artifact=self.held_lock_handshake.artifact.payload,
            lock_handshake_artifact_bytes=(
                self.held_lock_handshake.artifact.snapshot
            ),
            expected_lock_handshake_artifact_sha256=(
                self.held_lock_handshake.artifact.artifact_sha256
            ),
            expected_config_sha256=self.expected_config_sha256,
            expected_runtime_source_sha256=self.expected_runtime_source_sha256,
            expected_test_source_sha256=self.expected_test_source_sha256,
            expected_launcher_sha256=(
                self.held_lock_handshake.artifact.expected_launcher_sha256
            ),
            expected_launch_plan_sha256=(
                self.held_lock_handshake.artifact.expected_launch_plan_sha256
            ),
            expected_remote_project_root=(
                self.held_lock_handshake.artifact.expected_remote_project_root
            ),
            expected_attempt_nonce=(
                self.held_lock_handshake.artifact.expected_attempt_nonce
            ),
        )
        checkpoint = Path(self.diagnostic_checkpoint_path)
        checkpoint_probe = cost_report["checkpoint_cost_probe"]
        if type(self.diagnostic_checkpoint_snapshot) is not bytes:
            raise TypeError("preflight diagnostic checkpoint must be immutable bytes")
        if (
            checkpoint.name != cost_report["checkpoint_artifact_filename"]
            or len(self.diagnostic_checkpoint_snapshot)
            != checkpoint_probe["checkpoint_bytes"]
            or hashlib.sha256(self.diagnostic_checkpoint_snapshot).hexdigest()
            != checkpoint_probe["checkpoint_sha256"]
            or self.report["validation_manifest_commitment"] != commitment
        ):
            raise ValueError("preflight result artifact commitment mismatch")


def _make_report(
    *,
    fixture: SyntheticPreflightFixture,
    attestation: FrozenRuntimeAttestation,
    training_summary: Mapping[str, Any],
    inference_summary: Mapping[str, Any],
    projection: Mapping[str, Any],
    ledger_rows: Sequence[Mapping[str, Any]],
    validation_manifest_commitment: Mapping[str, Any],
    overhead_cost_probe_report: Mapping[str, Any],
    lock_handshake_artifact_sha256: str,
    preflight_started_perf_counter_ns: int,
) -> dict[str, Any]:
    cost_report = dict(overhead_cost_probe_report)
    if not _is_sha256(cost_report.get("probe_id")):
        raise ValueError("preflight report requires the full overhead cost-probe ID")
    if not _is_sha256(lock_handshake_artifact_sha256):
        raise ValueError("preflight report requires the lock-handshake artifact SHA-256")
    preflight_started = _strict_int(
        preflight_started_perf_counter_ns,
        field="preflight_started_perf_counter_ns",
        minimum=1,
    )
    report: dict[str, Any] = {
        "schema": PREFLIGHT_REPORT_SCHEMA_VERSION,
        "status": "PASS",
        "fixture": dict(fixture.payload),
        "runtime": {
            "run_id": attestation.run_id,
            "gpu_uuid": attestation.gpu_uuid,
            "logical_device_index": attestation.logical_device_index,
        },
        "model": {
            "model_semantics_version": MODEL_SEMANTICS_VERSION,
            "parameter_count": MODEL_PARAMETER_COUNT,
            "training_config_sha256": training_config_sha256(),
            "optimizer": "one_group_adamw_full_weight_decay",
            "precision": "bf16_autocast_forward_fp32_loss_gradients_state",
            "grad_scaler": False,
            "disposition": "discarded_without_checkpoint",
            "primary_requires_fresh_reconstruction": True,
        },
        "training": dict(training_summary),
        "inference": dict(inference_summary),
        "validation_manifest_commitment": dict(validation_manifest_commitment),
        "overhead_cost_probe_id": cost_report["probe_id"],
        "lock_handshake_artifact_sha256": lock_handshake_artifact_sha256,
        "preflight_started_perf_counter_ns": preflight_started,
        "fresh_reconstruction_probe": dict(
            cost_report["fresh_reconstruction_probe"]
        ),
        "checkpoint_cost_probe": dict(cost_report["checkpoint_cost_probe"]),
        "final_selection_probe": dict(cost_report["final_selection_probe"]),
        "lock_handshake_probe": dict(cost_report["lock_handshake_probe"]),
        "budget_projection": dict(projection),
        "dataset_file_reads": 0,
        "training_checkpoint_writes": 0,
        "diagnostic_checkpoint_writes": 1,
        "fallback_used": False,
        "rng_state_restored": True,
        "ledger_rows_sha256": canonical_sha256(list(ledger_rows)),
    }
    report["report_id"] = canonical_sha256(report)
    return report


def _validate_preflight_report_core(
    report: object,
    ledger_rows: Sequence[Mapping[str, Any]],
    *,
    expected_validation_commitment: Mapping[str, Any] | None = None,
    overhead_cost_probe_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(report, dict):
        raise TypeError("preflight report must be a dictionary")
    if set(report) != _REPORT_FIELDS:
        raise ValueError("preflight report fields do not match the exact schema")
    normalized = validate_preflight_ledger_rows(ledger_rows)
    if report.get("schema") != PREFLIGHT_REPORT_SCHEMA_VERSION or report.get(
        "status"
    ) != "PASS":
        raise ValueError("unsupported or unsuccessful preflight report")
    validate_preflight_fixture_payload(report.get("fixture"))
    if report.get("dataset_file_reads") != 0:
        raise ValueError("preflight report must prove zero dataset file reads")
    if (
        report.get("training_checkpoint_writes") != 0
        or report.get("diagnostic_checkpoint_writes") != 1
        or report.get("fallback_used") is not False
    ):
        raise ValueError(
            "preflight must write exactly one diagnostic checkpoint, no training "
            "checkpoint, and use no fallback"
        )
    if report.get("rng_state_restored") is not True:
        raise ValueError("preflight report must bind successful RNG restoration")
    if report.get("ledger_rows_sha256") != canonical_sha256(list(normalized)):
        raise ValueError("preflight report does not bind its exact ledger rows")
    training = report.get("training")
    if not isinstance(training, dict):
        raise TypeError("preflight training summary must be a dictionary")
    if set(training) != _TRAINING_SUMMARY_FIELDS:
        raise ValueError("preflight training fields do not match the exact schema")
    expected_training = {
        "updates": PREFLIGHT_UPDATES,
        "microbatches_per_update": GRADIENT_ACCUMULATION,
        "total_microbatches": PREFLIGHT_UPDATES * GRADIENT_ACCUMULATION,
        "masked_tokens_per_microbatch": PREFLIGHT_MASKED_TOKENS_PER_MICROBATCH,
        "masked_token_predictions": (
            PREFLIGHT_UPDATES * PREFLIGHT_MASKED_TOKENS_PER_UPDATE
        ),
        "encoder_forward_calls": PREFLIGHT_UPDATES * GRADIENT_ACCUMULATION,
        "decoder_forward_calls": PREFLIGHT_UPDATES * GRADIENT_ACCUMULATION,
        "backward_calls": PREFLIGHT_UPDATES * GRADIENT_ACCUMULATION,
        "ledger_row_count": PREFLIGHT_UPDATES,
        "ledger_rows_sha256": canonical_sha256(list(normalized)),
        "total_update_wall_ns": sum(int(row["wall_time_ns"]) for row in normalized),
        "total_update_cuda_event_ns": sum(
            int(row["cuda_event_ns"]) for row in normalized
        ),
        "peak_allocated_bytes": max(
            int(row["cuda_peak_allocated_bytes"]) for row in normalized
        ),
        "peak_reserved_bytes": max(
            int(row["cuda_peak_reserved_bytes"]) for row in normalized
        ),
    }
    if any(training.get(field) != value for field, value in expected_training.items()):
        raise ValueError("preflight training summary does not close from ledger rows")
    for field in (
        "learning_rate_hex",
        "mean_masked_cell_ce_hex",
        "gradient_norm_before_clip_hex",
    ):
        values = training.get(field)
        if not isinstance(values, list) or len(values) != PREFLIGHT_UPDATES:
            raise ValueError(f"preflight {field} must contain 100 values")
        for value in values:
            if not isinstance(value, str):
                raise TypeError(f"preflight {field} values must be float.hex strings")
            parsed = float.fromhex(value)
            if not math.isfinite(parsed) or parsed < 0.0:
                raise ValueError(f"preflight {field} contains an invalid value")
    if training["learning_rate_hex"] != [
        learning_rate_for_update(step).hex()
        for step in range(1, PREFLIGHT_UPDATES + 1)
    ]:
        raise ValueError("preflight learning-rate trace drifted")
    _validate_inference_summary(report.get("inference"), require_cuda=True)
    embedded_commitment = report.get("validation_manifest_commitment")
    if expected_validation_commitment is not None:
        if embedded_commitment != dict(expected_validation_commitment):
            raise ValueError("preflight report differs from external validation commitment")
    elif not isinstance(embedded_commitment, dict):
        raise TypeError("preflight report requires a validation commitment")
    runtime_record = report.get("runtime")
    if not isinstance(runtime_record, dict) or not isinstance(
        runtime_record.get("run_id"), str
    ):
        raise TypeError("preflight runtime identity is invalid")
    run_id = runtime_record["run_id"]
    child_probes = {
        "lock_handshake_probe": validate_lock_handshake_probe(
            report["lock_handshake_probe"], expected_run_id=run_id
        ),
        "fresh_reconstruction_probe": validate_fresh_reconstruction_probe(
            report["fresh_reconstruction_probe"], expected_run_id=run_id
        ),
        "checkpoint_cost_probe": validate_checkpoint_cost_probe(
            report["checkpoint_cost_probe"], expected_run_id=run_id
        ),
        "final_selection_probe": validate_final_selection_probe(
            report["final_selection_probe"], expected_run_id=run_id
        ),
    }
    if not _is_sha256(report.get("overhead_cost_probe_id")):
        raise ValueError("preflight report lacks a full overhead cost-probe ID")
    if not _is_sha256(report.get("lock_handshake_artifact_sha256")):
        raise ValueError("preflight report lacks its lock-handshake artifact SHA-256")
    if overhead_cost_probe_report is not None:
        cost_report = validate_campaign_overhead_cost_probe_report(
            dict(overhead_cost_probe_report), expected_run_id=run_id
        )
        if any(cost_report[name] != probe for name, probe in child_probes.items()):
            raise ValueError("preflight child probes differ from the full cost-probe report")
        if report["overhead_cost_probe_id"] != cost_report["probe_id"]:
            raise ValueError("preflight report does not bind the full cost-probe report")
    projection = report.get("budget_projection")
    if not isinstance(projection, dict):
        raise TypeError("preflight budget projection must be a dictionary")
    expected_projection = project_campaign_budget(
        measured_preflight_training_wall_ns=training["total_update_wall_ns"],
        measured_preflight_total_wall_ns=projection[
            "measured_preflight_total_wall_ns"
        ],
        preflight_started_perf_counter_ns=report[
            "preflight_started_perf_counter_ns"
        ],
        validation_episodes_per_pass=projection["validation_episodes_per_pass"],
        lock_handshake_probe=child_probes["lock_handshake_probe"],
        fresh_reconstruction_probe=child_probes["fresh_reconstruction_probe"],
        checkpoint_cost_probe=child_probes["checkpoint_cost_probe"],
        final_selection_probe=child_probes["final_selection_probe"],
    )
    if projection != expected_projection or projection["budget_status"] != (
        "WITHIN_BUDGET"
    ):
        raise ValueError("preflight budget projection is invalid or over budget")
    model = report.get("model")
    expected_model = {
        "model_semantics_version": MODEL_SEMANTICS_VERSION,
        "parameter_count": MODEL_PARAMETER_COUNT,
        "training_config_sha256": training_config_sha256(),
        "optimizer": "one_group_adamw_full_weight_decay",
        "precision": "bf16_autocast_forward_fp32_loss_gradients_state",
        "grad_scaler": False,
        "disposition": "discarded_without_checkpoint",
        "primary_requires_fresh_reconstruction": True,
    }
    if model != expected_model:
        raise ValueError("preflight model/disposition record drifted")
    runtime = report.get("runtime")
    if (
        not isinstance(runtime, dict)
        or set(runtime) != {"run_id", "gpu_uuid", "logical_device_index"}
        or not isinstance(runtime["run_id"], str)
        or not runtime["run_id"]
        or not isinstance(runtime["gpu_uuid"], str)
        or not runtime["gpu_uuid"]
        or runtime["logical_device_index"] != 0
    ):
        raise ValueError("preflight runtime identity record is invalid")
    semantic = dict(report)
    report_id = semantic.pop("report_id", None)
    if report_id != canonical_sha256(semantic):
        raise ValueError("preflight report_id mismatch")
    return dict(report)


def validate_preflight_report(
    report: object,
    ledger_rows: Sequence[Mapping[str, Any]],
    *,
    validation_manifest_commitment: ValidationManifestCommitment,
    overhead_cost_probe_report: Mapping[str, Any],
    expected_config_sha256: str,
    expected_runtime_source_sha256: str,
    expected_test_source_sha256: str,
    lock_handshake_artifact: LockHandshakeArtifact,
    launch_plan_artifact: LaunchPlanArtifact,
    expected_run_root: str,
) -> dict[str, Any]:
    """Validate production PASS evidence against externally committed parents."""

    commitment = validate_validation_manifest_commitment(
        validation_manifest_commitment
    )
    runtime = report.get("runtime") if isinstance(report, dict) else None
    if not isinstance(runtime, dict) or not isinstance(runtime.get("run_id"), str):
        raise ValueError("preflight report lacks a production runtime identity")
    if type(launch_plan_artifact) is not LaunchPlanArtifact:
        raise TypeError("production preflight requires a committed launch plan")
    if type(lock_handshake_artifact) is not LockHandshakeArtifact:
        raise TypeError("production preflight requires a committed lock artifact")
    launch_plan = launch_plan_artifact.payload
    if (
        not isinstance(expected_run_root, str)
        or expected_run_root != launch_plan["run_root"]
        or runtime["run_id"] != launch_plan["run_id"]
        or expected_config_sha256 != launch_plan["training_config_sha256"]
        or expected_runtime_source_sha256
        != launch_plan["runtime_source_fingerprint_sha256"]
        or expected_test_source_sha256
        != launch_plan["test_source_fingerprint_sha256"]
        or commitment != launch_plan["validation_manifest_commitment"]
        or launch_plan_artifact.artifact_sha256
        != lock_handshake_artifact.expected_launch_plan_sha256
        or lock_handshake_artifact.expected_launcher_sha256
        != launch_plan["expected_input_artifacts"]["remote_launcher.py"]
        or lock_handshake_artifact.expected_remote_project_root
        != launch_plan["remote_project_root"]
        or lock_handshake_artifact.expected_attempt_nonce
        != launch_plan["attempt_nonce"]
    ):
        raise ValueError("preflight inputs differ from the committed launch plan")
    cost_report = validate_campaign_overhead_cost_probe_report(
        dict(overhead_cost_probe_report),
        expected_run_id=runtime["run_id"],
        expected_config_sha256=expected_config_sha256,
        expected_runtime_source_sha256=expected_runtime_source_sha256,
        expected_test_source_sha256=expected_test_source_sha256,
        require_production_cuda=True,
    )
    lock_payload = lock_handshake_artifact.payload
    if (
        report.get("lock_handshake_artifact_sha256")
        != lock_handshake_artifact.artifact_sha256
        or report.get("lock_handshake_probe")
        != make_cost_probe_lock_handshake_from_artifact(lock_handshake_artifact)
        or lock_payload["run_id"] != runtime["run_id"]
        or lock_payload["gpu_uuid"] != runtime.get("gpu_uuid")
    ):
        raise ValueError("preflight report differs from its committed lock artifact")
    return _validate_preflight_report_core(
        report,
        ledger_rows,
        expected_validation_commitment=commitment,
        overhead_cost_probe_report=cost_report,
    )


def _validate_preflight_report_fixture(
    report: object,
    ledger_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """CPU-test-only structural replay; it cannot certify production evidence."""

    if not isinstance(report, dict) or not isinstance(
        report.get("validation_manifest_commitment"), dict
    ):
        raise TypeError("fixture preflight report lacks an embedded commitment")
    return _validate_preflight_report_core(
        report,
        ledger_rows,
        expected_validation_commitment=report["validation_manifest_commitment"],
    )


def _validate_preflight_launch_inputs(
    *,
    runtime_attestation: FrozenRuntimeAttestation,
    validation_manifest_commitment: ValidationManifestCommitment,
    run_root: str | Path,
    config_sha256: str,
    runtime_source_sha256: str,
    test_source_sha256: str,
    lock_handshake_artifact: LockHandshakeArtifact,
    launch_plan_artifact: LaunchPlanArtifact,
) -> str:
    if type(launch_plan_artifact) is not LaunchPlanArtifact:
        raise TypeError("exact preflight requires a committed launch-plan artifact")
    if type(lock_handshake_artifact) is not LockHandshakeArtifact:
        raise TypeError("exact preflight requires a committed lock artifact")
    attestation = validate_runtime_attestation(runtime_attestation)
    commitment = validate_validation_manifest_commitment(
        validation_manifest_commitment
    )
    launch_plan = launch_plan_artifact.payload
    run_root_text = os.fspath(run_root)
    if (
        run_root_text != launch_plan["run_root"]
        or attestation.run_id != launch_plan["run_id"]
        or config_sha256 != launch_plan["training_config_sha256"]
        or runtime_source_sha256
        != launch_plan["runtime_source_fingerprint_sha256"]
        or test_source_sha256 != launch_plan["test_source_fingerprint_sha256"]
        or commitment != launch_plan["validation_manifest_commitment"]
        or launch_plan_artifact.artifact_sha256
        != lock_handshake_artifact.expected_launch_plan_sha256
        or lock_handshake_artifact.expected_run_id != launch_plan["run_id"]
        or lock_handshake_artifact.expected_gpu_uuid != attestation.gpu_uuid
        or lock_handshake_artifact.expected_launcher_sha256
        != launch_plan["expected_input_artifacts"]["remote_launcher.py"]
        or lock_handshake_artifact.expected_remote_project_root
        != launch_plan["remote_project_root"]
        or lock_handshake_artifact.expected_attempt_nonce
        != launch_plan["attempt_nonce"]
    ):
        raise ValueError("exact preflight inputs differ from the committed launch plan")
    return run_root_text


def _finish_preflight_and_commit_diagnostic(
    *,
    fixture: SyntheticPreflightFixture,
    attestation: FrozenRuntimeAttestation,
    validation_manifest_commitment: ValidationManifestCommitment,
    validation_episodes: int,
    run_root: str | Path,
    config_sha256: str,
    runtime_source_sha256: str,
    test_source_sha256: str,
    launch_plan_artifact: LaunchPlanArtifact,
    held_lock: HeldGPULockHandshake,
    preflight_started: int,
    run_started: int,
    ledger_rows: tuple[dict[str, Any], ...],
    training_summary: dict[str, Any],
    inference_summary: dict[str, Any],
    overhead_preparation: PreparedCampaignOverheadCostProbe,
) -> PreflightResult:
    """Validate the complete outcome before exposing the diagnostic filename."""

    validation_commitment_payload = validate_validation_manifest_commitment(
        validation_manifest_commitment
    )
    if validation_episodes != validation_commitment_payload["row_count"]:
        raise ValueError("preflight validation episode count differs from commitment")
    overhead_result = overhead_preparation.result
    overhead_report = validate_campaign_overhead_cost_probe_report(
        overhead_result.report,
        expected_run_id=attestation.run_id,
        expected_config_sha256=config_sha256,
        expected_runtime_source_sha256=runtime_source_sha256,
        expected_test_source_sha256=test_source_sha256,
        require_production_cuda=True,
    )
    preflight_total_wall_ns = max(
        time.perf_counter_ns() - run_started,
        int(training_summary["total_update_wall_ns"]),
    )
    projection = project_campaign_budget(
        measured_preflight_training_wall_ns=int(
            training_summary["total_update_wall_ns"]
        ),
        measured_preflight_total_wall_ns=preflight_total_wall_ns,
        preflight_started_perf_counter_ns=preflight_started,
        validation_episodes_per_pass=validation_episodes,
        lock_handshake_probe=overhead_report["lock_handshake_probe"],
        fresh_reconstruction_probe=overhead_report[
            "fresh_reconstruction_probe"
        ],
        checkpoint_cost_probe=overhead_report["checkpoint_cost_probe"],
        final_selection_probe=overhead_report["final_selection_probe"],
    )
    if projection["budget_status"] == PREFLIGHT_BUDGET_EXCEEDED:
        failure_report = _make_failure_report(
            failure_code=PREFLIGHT_BUDGET_EXCEEDED,
            fixture_id=fixture.fixture_id,
            completed_updates=PREFLIGHT_UPDATES,
            ledger_rows=ledger_rows,
            projection=projection,
            training_summary=training_summary,
            inference_summary=inference_summary,
            overhead_cost_probe_report=overhead_report,
            diagnostic_checkpoint_snapshot=overhead_result.checkpoint_snapshot,
            runtime_attestation=attestation,
            validation_manifest_commitment=validation_commitment_payload,
            lock_handshake_artifact_sha256=held_lock.artifact.artifact_sha256,
            preflight_started_perf_counter_ns=preflight_started,
        )
        validate_preflight_failure_report(
            failure_report,
            ledger_rows,
            overhead_cost_probe_report=overhead_report,
            diagnostic_checkpoint_snapshot=overhead_result.checkpoint_snapshot,
            expected_config_sha256=config_sha256,
            expected_runtime_source_sha256=runtime_source_sha256,
            expected_test_source_sha256=test_source_sha256,
        )
        failure = PreflightFailure(
            PREFLIGHT_BUDGET_EXCEEDED,
            "measured-rate campaign projection exceeds 24 GPU-hours",
            report=failure_report,
            ledger_rows=ledger_rows,
            overhead_cost_probe_report=overhead_report,
            training_summary=training_summary,
            inference_summary=inference_summary,
            diagnostic_checkpoint_snapshot=overhead_result.checkpoint_snapshot,
        )
        assert_held_gpu_lock(held_lock)
        overhead_preparation.commit()
        raise failure

    report = _make_report(
        fixture=fixture,
        attestation=attestation,
        training_summary=training_summary,
        inference_summary=inference_summary,
        projection=projection,
        ledger_rows=ledger_rows,
        validation_manifest_commitment=validation_commitment_payload,
        overhead_cost_probe_report=overhead_report,
        lock_handshake_artifact_sha256=held_lock.artifact.artifact_sha256,
        preflight_started_perf_counter_ns=preflight_started,
    )
    result = PreflightResult(
        report=report,
        ledger_rows=ledger_rows,
        validation_manifest_commitment=validation_manifest_commitment,
        launch_plan_artifact=launch_plan_artifact,
        overhead_cost_probe_report=overhead_report,
        diagnostic_checkpoint_path=overhead_result.checkpoint_path,
        diagnostic_checkpoint_snapshot=overhead_result.checkpoint_snapshot,
        expected_config_sha256=config_sha256,
        expected_runtime_source_sha256=runtime_source_sha256,
        expected_test_source_sha256=test_source_sha256,
        expected_run_root=os.fspath(run_root),
        held_lock_handshake=held_lock,
    )
    assert_held_gpu_lock(held_lock)
    overhead_preparation.commit()
    return result


def _finish_preflight_with_failure_evidence(
    *,
    fixture: SyntheticPreflightFixture,
    attestation: FrozenRuntimeAttestation,
    validation_manifest_commitment: ValidationManifestCommitment,
    validation_episodes: int,
    run_root: str | Path,
    config_sha256: str,
    runtime_source_sha256: str,
    test_source_sha256: str,
    launch_plan_artifact: LaunchPlanArtifact,
    held_lock: HeldGPULockHandshake,
    preflight_started: int,
    run_started: int,
    ledger_rows: tuple[dict[str, Any], ...],
    training_summary: dict[str, Any],
    inference_summary: dict[str, Any],
    overhead_preparation: PreparedCampaignOverheadCostProbe,
) -> PreflightResult:
    """Convert every ordinary pre-commit finish error to strict partial evidence."""

    try:
        return _finish_preflight_and_commit_diagnostic(
            fixture=fixture,
            attestation=attestation,
            validation_manifest_commitment=validation_manifest_commitment,
            validation_episodes=validation_episodes,
            run_root=run_root,
            config_sha256=config_sha256,
            runtime_source_sha256=runtime_source_sha256,
            test_source_sha256=test_source_sha256,
            launch_plan_artifact=launch_plan_artifact,
            held_lock=held_lock,
            preflight_started=preflight_started,
            run_started=run_started,
            ledger_rows=ledger_rows,
            training_summary=training_summary,
            inference_summary=inference_summary,
            overhead_preparation=overhead_preparation,
        )
    except PreflightFailure:
        overhead_preparation.abort()
        raise
    except Exception as exc:
        committed = overhead_preparation.committed
        overhead_preparation.abort()
        if committed:
            raise
        failure_report = _make_failure_report(
            failure_code=PREFLIGHT_PROJECTION_INCOMPLETE,
            fixture_id=fixture.fixture_id,
            completed_updates=len(ledger_rows),
            ledger_rows=ledger_rows,
        )
        raise PreflightFailure(
            PREFLIGHT_PROJECTION_INCOMPLETE,
            "complete exact-preflight outcome did not close before diagnostic commit",
            report=failure_report,
            ledger_rows=ledger_rows,
        ) from exc
    except BaseException:
        overhead_preparation.abort()
        raise


def _run_exact_preflight_with_held_lock(
    *,
    runtime_attestation: FrozenRuntimeAttestation,
    validation_manifest_commitment: ValidationManifestCommitment,
    run_root: str | Path,
    cost_probe_scratch_dir: str | Path,
    config_sha256: str,
    runtime_source_sha256: str,
    test_source_sha256: str,
    launch_plan_artifact: LaunchPlanArtifact,
    held_lock: HeldGPULockHandshake,
    preflight_started: int,
) -> PreflightResult:
    """Run after the public boundary has taken ownership of the inherited lock."""

    lock_payload = assert_held_gpu_lock(held_lock)
    lock_handshake_probe = make_cost_probe_lock_handshake(held_lock)
    attestation = validate_runtime_attestation(runtime_attestation)
    if (
        lock_payload["run_id"] != attestation.run_id
        or lock_payload["gpu_uuid"] != attestation.gpu_uuid
    ):
        raise ValueError("runtime attestation differs from the inherited GPU lock")
    validation_commitment_payload = validate_validation_manifest_commitment(
        validation_manifest_commitment
    )
    validation_episodes = int(validation_commitment_payload["row_count"])
    if config_sha256 != training_config_sha256():
        raise ValueError("preflight config SHA differs from the frozen training config")
    fixture = build_preflight_fixture()
    rng_before = capture_rng_state(optimizer_step=0)
    model: GridCMLM | None = None
    optimizer: torch.optim.AdamW | None = None
    ledger_rows: tuple[dict[str, Any], ...] = ()
    ledger_buffer: list[dict[str, Any]] = []
    training_summary: dict[str, Any] | None = None
    inference_summary: dict[str, Any] | None = None
    overhead_result: CampaignOverheadCostProbeResult | None = None
    overhead_preparation: PreparedCampaignOverheadCostProbe | None = None
    run_started = preflight_started
    try:
        torch.manual_seed(TRAINING_SEED)
        torch.cuda.manual_seed_all(TRAINING_SEED)
        model = GridCMLM()
        assert_grid_cmlm_invariants(model, require_cuda=False)
        model = model.to(device="cuda:0").train()
        assert_grid_cmlm_invariants(model, require_cuda=True)
        optimizer = build_adamw_optimizer(model)
        ledger_rows, training_summary = _run_training_fixture(
            model, optimizer, fixture, ledger_sink=ledger_buffer
        )
        validate_runtime_attestation(attestation)
        sampler_runtime = SamplerRuntime.cuda(attestation)
        inference_summary = _run_cached_inference_fixture(
            model, fixture, sampler_runtime
        )
        _validate_inference_summary(inference_summary, require_cuda=True)
        try:
            assert_held_gpu_lock(held_lock)
            overhead_preparation = run_campaign_overhead_cost_probe(
                model=model,
                optimizer=optimizer,
                runtime_attestation=attestation,
                run_root=run_root,
                scratch_dir=cost_probe_scratch_dir,
                config_sha256=config_sha256,
                runtime_source_sha256=runtime_source_sha256,
                test_source_sha256=test_source_sha256,
                lock_handshake_probe=lock_handshake_probe,
            )
            overhead_result = overhead_preparation.result
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception as exc:
            raise PreflightFailure(
                PREFLIGHT_PROJECTION_INCOMPLETE,
                "campaign overhead cost probe did not close",
            ) from exc
    except PreflightFailure as exc:
        partial_rows = tuple(ledger_buffer)
        if exc.report is not None:
            raise
        failure_report = _make_failure_report(
            failure_code=exc.failure_code,
            fixture_id=fixture.fixture_id,
            completed_updates=len(partial_rows),
            ledger_rows=partial_rows,
        )
        raise PreflightFailure(
            exc.failure_code,
            str(exc),
            report=failure_report,
            ledger_rows=partial_rows,
        ) from exc
    except torch.cuda.OutOfMemoryError as exc:
        partial_rows = tuple(ledger_buffer)
        failure_report = _make_failure_report(
            failure_code=PREFLIGHT_OOM,
            fixture_id=fixture.fixture_id,
            completed_updates=len(partial_rows),
            ledger_rows=partial_rows,
        )
        raise PreflightFailure(
            PREFLIGHT_OOM,
            "maximum-context preflight exhausted CUDA memory; no fallback is allowed",
            report=failure_report,
            ledger_rows=partial_rows,
        ) from exc
    finally:
        active_exception = sys.exc_info()[1]
        optimizer = None
        model = None
        try:
            _restore_rng_and_cleanup(rng_before, active_exception=active_exception)
        except BaseException:
            if overhead_preparation is not None:
                overhead_preparation.abort()
            raise
    if (
        training_summary is None
        or inference_summary is None
        or overhead_result is None
        or overhead_preparation is None
    ):
        if overhead_preparation is not None:
            overhead_preparation.abort()
        raise RuntimeError("preflight ended without a result or terminal failure")
    return _finish_preflight_with_failure_evidence(
        fixture=fixture,
        attestation=attestation,
        validation_manifest_commitment=validation_manifest_commitment,
        validation_episodes=validation_episodes,
        run_root=run_root,
        config_sha256=config_sha256,
        runtime_source_sha256=runtime_source_sha256,
        test_source_sha256=test_source_sha256,
        launch_plan_artifact=launch_plan_artifact,
        held_lock=held_lock,
        preflight_started=preflight_started,
        run_started=run_started,
        ledger_rows=ledger_rows,
        training_summary=training_summary,
        inference_summary=inference_summary,
        overhead_preparation=overhead_preparation,
    )


def run_exact_preflight(
    *,
    runtime_attestation: FrozenRuntimeAttestation,
    validation_manifest_commitment: ValidationManifestCommitment,
    run_root: str | Path,
    cost_probe_scratch_dir: str | Path,
    config_sha256: str,
    runtime_source_sha256: str,
    test_source_sha256: str,
    lock_handshake_artifact: LockHandshakeArtifact,
    launch_plan_artifact: LaunchPlanArtifact,
) -> PreflightResult:
    """Run the frozen preflight, releasing the owned lock on every failure path."""

    preflight_started = time.perf_counter_ns()
    held_lock = attest_inherited_gpu_lock(
        lock_handshake_artifact,
        preflight_started_perf_counter_ns=preflight_started,
    )
    try:
        expected_run_root = _validate_preflight_launch_inputs(
            runtime_attestation=runtime_attestation,
            validation_manifest_commitment=validation_manifest_commitment,
            run_root=run_root,
            config_sha256=config_sha256,
            runtime_source_sha256=runtime_source_sha256,
            test_source_sha256=test_source_sha256,
            lock_handshake_artifact=lock_handshake_artifact,
            launch_plan_artifact=launch_plan_artifact,
        )
        return _run_exact_preflight_with_held_lock(
            runtime_attestation=runtime_attestation,
            validation_manifest_commitment=validation_manifest_commitment,
            run_root=expected_run_root,
            cost_probe_scratch_dir=cost_probe_scratch_dir,
            config_sha256=config_sha256,
            runtime_source_sha256=runtime_source_sha256,
            test_source_sha256=test_source_sha256,
            launch_plan_artifact=launch_plan_artifact,
            held_lock=held_lock,
            preflight_started=preflight_started,
        )
    except BaseException as exc:
        try:
            release_held_gpu_lock(held_lock)
        except BaseException as release_error:
            try:
                setattr(exc, "gpu_lock_release_error", repr(release_error))
            except Exception:
                pass
        raise


__all__ = [
    "PREFLIGHT_BUDGET_EXCEEDED",
    "PREFLIGHT_DEMONSTRATIONS",
    "PREFLIGHT_FAILURE_REPORT_SCHEMA_VERSION",
    "PREFLIGHT_FIXTURE_SCHEMA_VERSION",
    "PREFLIGHT_GRID_SIDE",
    "PREFLIGHT_INFERENCE_SCHEMA_VERSION",
    "PREFLIGHT_LANES",
    "PREFLIGHT_MEMORY_LENGTH",
    "PREFLIGHT_NONFINITE_LOGITS",
    "PREFLIGHT_OOM",
    "PREFLIGHT_POST_VALIDATION_MARGIN_NS",
    "PREFLIGHT_PROJECTION_SCHEMA_VERSION",
    "PREFLIGHT_PROJECTION_INCOMPLETE",
    "PREFLIGHT_REPORT_SCHEMA_VERSION",
    "PREFLIGHT_SAMPLE_EQUIVALENT_CALLS",
    "PreflightFailure",
    "PreflightResult",
    "SyntheticPreflightFixture",
    "build_preflight_fixture",
    "project_campaign_budget",
    "run_exact_preflight",
    "validate_preflight_fixture_payload",
    "validate_preflight_failure_report",
    "validate_preflight_ledger_rows",
    "validate_preflight_report",
]
