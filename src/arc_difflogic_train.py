"""Task adaptation, hardening, and label-isolated evaluation for DiffLogic-ARC."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time
from typing import Any, Sequence

import numpy as np

from arc_ca import predict_direct, select_demo_rule
from arc_data import ArcExample, ArcLabels, ArcProblem
from arc_difflogic_features import (
    CanvasExample,
    augment_examples,
    build_canvas_example,
    decode_color_channels,
    infer_workspace_shape,
    modal_color,
    paired_grid_metrics,
    select_augmentation_policy,
    select_shape_program,
)
from arc_difflogic_model import (
    ArcDiffLogicConfig,
    create_model,
    hard_numpy_ca_rollout,
    invalid_color_probability,
)
from trainable_difflogic import hard_spec_sha256

try:  # Optional experiment dependency.
    import torch
    from torch.nn import functional as F
except ImportError:  # pragma: no cover
    torch = None
    F = None


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 240
    learning_rate: float = 0.02
    mlp_learning_rate: float = 0.003
    temperature_start: float = 2.0
    temperature_soft_end: float = 0.5
    temperature_hard_end: float = 0.2
    soft_fraction: float = 0.65
    entropy_weight: float = 2e-4
    invalid_weight: float = 0.05
    binary_weight: float = 0.01
    changed_cell_weight: float = 4.0
    intermediate_loss_weight: float = 0.0
    gradient_clip: float = 1.0
    trace_every: int = 20
    early_stop_patience: int = 3

    def __post_init__(self) -> None:
        if self.epochs < 1 or not 0 < self.soft_fraction <= 1:
            raise ValueError("invalid training schedule")


@dataclass
class PreparedTask:
    shape_program_name: str
    augmentation_policy: str
    augmentation_lodo_pair_exact: float
    augmentation_lodo_cell_accuracy: float
    workspace_shape: tuple[int, int]
    training: tuple[CanvasExample, ...]
    tests: tuple[CanvasExample, ...]


@dataclass
class TrainedTaskCandidate:
    task_id: str
    variant: str
    seed: int
    status: str
    record: dict[str, Any]
    hard_predictions: list[np.ndarray]
    soft_predictions: list[np.ndarray]
    hard_specifications: list[dict[str, Any]]
    trace: list[dict[str, float]]


def _require_torch() -> None:
    if torch is None or F is None:
        raise ImportError("DiffLogic-ARC training requires PyTorch")


def prepare_task(problem: ArcProblem, config: ArcDiffLogicConfig, augmentation_policy: str = "auto") -> PreparedTask | None:
    shape_program = select_shape_program(problem.demonstrations)
    if shape_program is None:
        return None
    shape_changes = any(example.input_grid.shape != example.output_grid.shape for example in problem.demonstrations)
    if shape_changes and not config.allow_shape_change:
        return None
    if augmentation_policy == "auto":
        augmentation = select_augmentation_policy(problem.demonstrations)
        policy = augmentation.policy
    else:
        if augmentation_policy not in ("none", "d4", "bgpad", "d4_bgpad"):
            raise ValueError("augmentation must be auto, none, d4, bgpad, or d4_bgpad")
        selected = select_augmentation_policy(problem.demonstrations)
        policy = augmentation_policy
        policy_bits = {"none": 1, "d4": 2, "bgpad": 2, "d4_bgpad": 3}
        augmentation = type(selected)(policy, math.nan, math.nan, policy_bits[policy])
    uses_background_pad = policy in ("bgpad", "d4_bgpad")
    if uses_background_pad and shape_changes:
        raise ValueError("background padding is defined only for same-shape tasks")
    augmented = augment_examples(problem.demonstrations, policy)
    workspace = infer_workspace_shape(problem.demonstrations, problem.test_inputs, shape_program, augmentation=policy)
    training = tuple(
        build_canvas_example(
            example.input_grid,
            tuple(int(value) for value in example.output_grid.shape),
            workspace,
            problem.demonstrations,
            shape_program,
            hidden_bits=config.hidden_bits,
            target_grid=example.output_grid,
            crop_margin=1 if uses_background_pad else 0,
            include_masks=config.include_masks,
            include_original=config.include_original,
            include_geometry=config.include_geometry,
            include_objects=config.include_objects,
            include_context=config.include_context,
        )
        for example in augmented
    )
    tests = tuple(
        build_canvas_example(
            (
                np.pad(grid, 1, constant_values=modal_color(grid))
                if uses_background_pad
                else grid
            ),
            (
                (int(grid.shape[0]) + 2, int(grid.shape[1]) + 2)
                if uses_background_pad
                else shape_program.predict_shape(grid)
            ),
            workspace,
            problem.demonstrations,
            shape_program,
            hidden_bits=config.hidden_bits,
            crop_margin=1 if uses_background_pad else 0,
            include_masks=config.include_masks,
            include_original=config.include_original,
            include_geometry=config.include_geometry,
            include_objects=config.include_objects,
            include_context=config.include_context,
        )
        for grid in problem.test_inputs
    )
    return PreparedTask(
        shape_program.name,
        policy,
        augmentation.lodo_pair_exact,
        augmentation.lodo_cell_accuracy,
        workspace,
        training,
        tests,
    )


def _stack_training(examples: Sequence[CanvasExample], device: str) -> dict[str, "torch.Tensor"]:
    _require_torch()
    targets = [example.target_bits for example in examples]
    if any(target is None for target in targets):
        raise ValueError("training examples require targets")
    changed = [example.changed_mask for example in examples]
    if any(mask is None for mask in changed):
        raise ValueError("training examples require changed masks")
    return {
        "initial": torch.as_tensor(np.stack([item.initial_state for item in examples]), device=device),
        "static": torch.as_tensor(np.stack([item.static_features for item in examples]), device=device),
        "target": torch.as_tensor(np.stack(targets), device=device),
        "target_mask": torch.as_tensor(np.stack([item.target_mask for item in examples]), device=device),
        "update_mask": torch.as_tensor(np.stack([item.update_mask for item in examples]), device=device),
        "changed": torch.as_tensor(np.stack(changed), device=device),
    }


def _temperature_and_mode(epoch: int, config: TrainingConfig) -> tuple[float, str]:
    split = max(1, int(round(config.epochs * config.soft_fraction)))
    if epoch < split:
        fraction = epoch / max(1, split - 1)
        temperature = config.temperature_start + fraction * (config.temperature_soft_end - config.temperature_start)
        return float(temperature), "soft"
    fraction = (epoch - split) / max(1, config.epochs - split - 1)
    temperature = config.temperature_soft_end + fraction * (config.temperature_hard_end - config.temperature_soft_end)
    return float(temperature), "st"


def _color_loss(state: "torch.Tensor", tensors: dict[str, "torch.Tensor"], config: TrainingConfig) -> "torch.Tensor":
    # Preserve the straight-through gradient at hard forward values 0/1.
    probabilities = state[:, :4] * (1 - 2e-6) + 1e-6
    per_bit = F.binary_cross_entropy(probabilities, tensors["target"], reduction="none").mean(dim=1)
    weights = tensors["target_mask"] * (1 + config.changed_cell_weight * tensors["changed"])
    return torch.sum(per_bit * weights) / weights.sum().clamp_min(1)


def _training_objective(
    model: Any,
    states: Sequence["torch.Tensor"],
    tensors: dict[str, "torch.Tensor"],
    config: TrainingConfig,
    temperature: float,
) -> tuple["torch.Tensor", dict[str, float]]:
    color = _color_loss(states[-1], tensors, config)
    if config.intermediate_loss_weight and len(states) > 1:
        earlier = torch.stack([_color_loss(state, tensors, config) for state in states[:-1]]).mean()
        color = color + config.intermediate_loss_weight * earlier
    invalid = (
        invalid_color_probability(states[-1][:, :4]) * tensors["target_mask"]
    ).sum() / tensors["target_mask"].sum().clamp_min(1)
    binary = (
        states[-1] * (1 - states[-1]) * tensors["update_mask"][:, None]
    ).sum() / (tensors["update_mask"].sum().clamp_min(1) * states[-1].shape[1])
    entropy = model.gate_entropy(temperature)
    total = color + config.invalid_weight * invalid + config.binary_weight * binary + config.entropy_weight * entropy
    return total, {
        "loss": float(total.detach().cpu()),
        "color_loss": float(color.detach().cpu()),
        "invalid_probability": float(invalid.detach().cpu()),
        "binary_penalty": float(binary.detach().cpu()),
        "gate_entropy": float(entropy.detach().cpu()),
    }


def _decode_soft_state(
    state: np.ndarray,
    fallback: np.ndarray,
    output_shape: tuple[int, int],
    crop_margin: int = 0,
) -> np.ndarray:
    probabilities = np.clip(state[:4], 1e-6, 1 - 1e-6)
    scores = []
    for code in range(10):
        bits = np.asarray([(code >> index) & 1 for index in range(4)], dtype=np.float32)[:, None, None]
        log_probability = np.sum(np.where(bits > 0, np.log(probabilities), np.log(1 - probabilities)), axis=0)
        scores.append(log_probability)
    colors = np.argmax(np.stack(scores, axis=0), axis=0).astype(np.uint8)
    output = colors[: output_shape[0], : output_shape[1]]
    if crop_margin:
        output = output[crop_margin:-crop_margin, crop_margin:-crop_margin]
    return output


def _decode_hard_state(state: np.ndarray, example: CanvasExample) -> tuple[np.ndarray, int, int]:
    hard_bits = np.asarray(state[:4] >= 0.5, dtype=np.uint8)
    colors, valid = decode_color_channels(hard_bits, example.fallback_colors)
    rows, columns = example.output_shape
    active_valid = valid[:rows, :columns]
    output = colors[:rows, :columns]
    if example.crop_margin:
        margin = example.crop_margin
        output = output[margin:-margin, margin:-margin]
        active_valid = active_valid[margin:-margin, margin:-margin]
    return output, int(np.sum(~active_valid)), int(active_valid.size)


def _rollout_examples(
    model: Any,
    examples: Sequence[CanvasExample],
    device: str,
    *,
    mode: str,
    temperature: float,
    steps: int,
) -> list[list[np.ndarray]]:
    _require_torch()
    initial = torch.as_tensor(np.stack([item.initial_state for item in examples]), device=device)
    static = torch.as_tensor(np.stack([item.static_features for item in examples]), device=device)
    mask = torch.as_tensor(np.stack([item.update_mask for item in examples]), device=device)
    with torch.no_grad():
        states = model.rollout(initial, static, mask, steps=steps, temperature=temperature, mode=mode)
    return [[state[index].detach().cpu().numpy() for index in range(len(examples))] for state in states]


def _training_references(examples: Sequence[CanvasExample]) -> list[np.ndarray]:
    references = []
    for example in examples:
        if example.target_bits is None:
            raise ValueError("missing training target")
        probabilities = example.target_bits
        references.append(
            _decode_soft_state(
                probabilities,
                example.fallback_colors,
                example.output_shape,
                example.crop_margin,
            )
        )
    return references


def candidate_horizons(max_steps: int) -> tuple[int, ...]:
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    horizons = []
    step = 1
    while step <= max_steps:
        horizons.append(step)
        step *= 2
    if horizons[-1] != max_steps:
        horizons.append(max_steps)
    return tuple(horizons)


def _select_horizon(model: Any, examples: Sequence[CanvasExample], device: str, max_steps: int) -> tuple[int, dict[str, float]]:
    trajectories = _rollout_examples(model, examples, device, mode="hard", temperature=0.1, steps=max_steps)
    references = _training_references(examples)
    scored = []
    for step in candidate_horizons(max_steps):
        states = trajectories[step - 1]
        predictions = [_decode_hard_state(state, example)[0] for state, example in zip(states, examples)]
        metrics = paired_grid_metrics(references, predictions)
        scored.append((step, metrics))
    exact = [item for item in scored if item[1]["task_exact"] == 1.0]
    if exact:
        return exact[0]
    return sorted(scored, key=lambda item: (-item[1]["cell_accuracy"], -item[1]["pair_exact_rate"], item[0]))[0]


def _fixed_point_steps(trajectory: Sequence[np.ndarray], initial: np.ndarray) -> int | None:
    previous = initial
    for step, state in enumerate(trajectory, start=1):
        if np.array_equal(previous, state):
            return step
        previous = state
    return None


def _predict_at_horizon(
    model: Any,
    examples: Sequence[CanvasExample],
    device: str,
    horizon: int,
    *,
    mode: str,
) -> tuple[list[np.ndarray], int, int, list[int | None]]:
    trajectories = _rollout_examples(model, examples, device, mode=mode, temperature=0.1, steps=horizon)
    final = trajectories[-1]
    predictions: list[np.ndarray] = []
    invalid = 0
    cells = 0
    fixed_points: list[int | None] = []
    for index, (state, example) in enumerate(zip(final, examples)):
        if mode == "hard":
            prediction, bad, count = _decode_hard_state(state, example)
            invalid += bad
            cells += count
        else:
            prediction = _decode_soft_state(
                state,
                example.fallback_colors,
                example.output_shape,
                example.crop_margin,
            )
        predictions.append(prediction)
        sample_trajectory = [states[index] for states in trajectories]
        fixed_points.append(_fixed_point_steps(sample_trajectory, example.initial_state))
    return predictions, invalid, cells, fixed_points


def train_task_candidate(
    problem: ArcProblem,
    config: ArcDiffLogicConfig,
    training_config: TrainingConfig,
    *,
    seed: int,
    device: str,
    augmentation_policy: str = "auto",
) -> TrainedTaskCandidate:
    _require_torch()
    started = time.perf_counter()
    prepared = prepare_task(problem, config, augmentation_policy)
    base_record: dict[str, Any] = {
        "task_id": problem.task_id,
        "variant": config.name,
        "model_kind": config.model_kind,
        "seed": int(seed),
        "device": str(device),
        "config": config.as_dict(),
        "training_config": asdict(training_config),
    }
    if prepared is None:
        base_record.update({"status": "unsupported_shape_program", "elapsed_seconds": time.perf_counter() - started})
        return TrainedTaskCandidate(problem.task_id, config.name, seed, "unsupported_shape_program", base_record, [], [], [], [])

    torch.manual_seed(int(seed))
    tensors = _stack_training(prepared.training, device)
    static_channels = int(tensors["static"].shape[1])
    model = create_model(config, static_channels, wiring_seed=int(seed)).to(device)
    learning_rate = training_config.mlp_learning_rate if config.model_kind == "mlp" else training_config.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    trace: list[dict[str, float]] = []
    exact_checkpoints = 0
    for epoch in range(training_config.epochs):
        temperature, mode = _temperature_and_mode(epoch, training_config)
        optimizer.zero_grad(set_to_none=True)
        states = model.rollout(
            tensors["initial"],
            tensors["static"],
            tensors["update_mask"],
            steps=config.max_steps,
            temperature=temperature,
            mode=mode,
        )
        loss, diagnostics = _training_objective(model, states, tensors, training_config, temperature)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.gradient_clip)
        optimizer.step()
        if epoch % training_config.trace_every == 0 or epoch + 1 == training_config.epochs:
            diagnostics.update({"epoch": float(epoch), "temperature": temperature, "straight_through": float(mode == "st")})
            trace.append(diagnostics)
            horizon, metrics = _select_horizon(model, prepared.training, device, config.max_steps)
            if metrics["task_exact"] == 1.0 and mode == "st":
                exact_checkpoints += 1
                if exact_checkpoints >= training_config.early_stop_patience:
                    break
            else:
                exact_checkpoints = 0

    horizon, demo_metrics = _select_horizon(model, prepared.training, device, config.max_steps)
    hard_predictions, invalid, invalid_denominator, fixed_points = _predict_at_horizon(
        model, prepared.tests, device, horizon, mode="hard"
    )
    soft_predictions, _, _, _ = _predict_at_horizon(model, prepared.tests, device, horizon, mode="soft")
    specifications = model.hard_specifications() if config.model_kind == "difflogic" else ()
    export_equivalent = math.nan
    export_sha = ""
    gate_counts: list[dict[str, Any]] = []
    if specifications:
        export_sha = hard_spec_sha256(specifications)
        gate_counts = [item.as_dict() for item in specifications]
        export_examples = prepared.training + prepared.tests
        torch_hard = _rollout_examples(
            model,
            export_examples,
            device,
            mode="hard",
            temperature=0.1,
            steps=horizon,
        )
        export_equivalent = float(
            all(
                np.array_equal(torch_hard[step][example_index], numpy_state)
                for example_index, example in enumerate(export_examples)
                for step, numpy_state in enumerate(
                    hard_numpy_ca_rollout(
                        example.initial_state,
                        example.static_features,
                        example.update_mask,
                        specifications,
                        horizon,
                    )
                )
            )
        )
        if export_equivalent != 1.0:
            raise RuntimeError("exported hard circuit disagrees with PyTorch rollout")

    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    parameter_storage_bits = int(sum(parameter.numel() * parameter.element_size() * 8 for parameter in model.parameters()))
    active_test_cells = int(sum(np.sum(example.update_mask) for example in prepared.tests))
    hard_gates_per_cell_step: int | float = math.nan
    hard_fixed_width_circuit_bits: int | float = math.nan
    test_dynamic_gate_evaluations: int | float = math.nan
    dense_macs_per_cell_step: int | float = math.nan
    test_dense_macs: int | float = math.nan
    if specifications:
        hard_gates_per_cell_step = int(sum(len(specification.gate_ids) for specification in specifications))
        hard_fixed_width_circuit_bits = 0
        for specification in specifications:
            source_bits = max(1, math.ceil(math.log2(specification.in_features)))
            hard_fixed_width_circuit_bits += specification.out_features * (2 * source_bits + 4)
        test_dynamic_gate_evaluations = active_test_cells * int(horizon) * hard_gates_per_cell_step
    else:
        dense_macs_per_cell_step = int(
            sum(parameter.numel() for name, parameter in model.named_parameters() if name.endswith("weight"))
        )
        test_dense_macs = active_test_cells * int(horizon) * dense_macs_per_cell_step
    test_state_bit_updates = active_test_cells * int(horizon) * config.state_bits
    base_record.update(
        {
            "status": "trained",
            "shape_program": prepared.shape_program_name,
            "augmentation_policy": prepared.augmentation_policy,
            "augmentation_lodo_pair_exact": prepared.augmentation_lodo_pair_exact,
            "augmentation_lodo_cell_accuracy": prepared.augmentation_lodo_cell_accuracy,
            "workspace_rows": prepared.workspace_shape[0],
            "workspace_columns": prepared.workspace_shape[1],
            "static_channels": static_channels,
            "epochs_executed": int(trace[-1]["epoch"] + 1) if trace else training_config.epochs,
            "selected_horizon": int(horizon),
            "hard_demo_task_exact": demo_metrics["task_exact"],
            "hard_demo_pair_exact_rate": demo_metrics["pair_exact_rate"],
            "hard_demo_cell_accuracy": demo_metrics["cell_accuracy"],
            "hard_invalid_cells": int(invalid),
            "hard_invalid_denominator": int(invalid_denominator),
            "hard_invalid_rate": float(invalid / invalid_denominator) if invalid_denominator else math.nan,
            "test_fixed_point_steps": fixed_points,
            "parameter_count": parameter_count,
            "parameter_storage_bits": parameter_storage_bits,
            "state_bits_per_cell": config.state_bits,
            "test_active_cells": active_test_cells,
            "test_executed_steps_per_example": int(horizon),
            "test_state_bit_updates": int(test_state_bit_updates),
            "hard_gates_per_cell_step": hard_gates_per_cell_step,
            "test_dynamic_gate_evaluations": test_dynamic_gate_evaluations,
            "hard_fixed_width_circuit_bits": hard_fixed_width_circuit_bits,
            "dense_macs_per_cell_step": dense_macs_per_cell_step,
            "test_dense_macs": test_dense_macs,
            "active_non_passthrough_gates": int(model.active_non_passthrough_gates()),
            "hard_export_sha256": export_sha,
            "hard_export_equivalent": export_equivalent,
            "elapsed_seconds": time.perf_counter() - started,
        }
    )
    return TrainedTaskCandidate(
        problem.task_id,
        config.name,
        seed,
        "trained",
        base_record,
        hard_predictions,
        soft_predictions,
        gate_counts,
        trace,
    )


def candidate_selection_key(candidate: TrainedTaskCandidate) -> tuple[float, float, int, int]:
    record = candidate.record
    if candidate.status != "trained":
        return (1.0, 1.0, 1 << 30, candidate.seed)
    return (
        -float(record["hard_demo_task_exact"]),
        -float(record["hard_demo_cell_accuracy"]),
        int(record["active_non_passthrough_gates"] or record["parameter_count"]),
        candidate.seed,
    )


def evaluate_candidate(candidate: TrainedTaskCandidate, labels: ArcLabels) -> dict[str, Any]:
    """Score already-produced predictions; never pass labels into training."""
    if candidate.task_id != labels.task_id:
        raise ValueError("candidate and label task ids differ")
    references = [np.asarray(output) for output in labels.test_outputs if output is not None]
    if len(references) != len(labels.test_outputs):
        raise ValueError("evaluation requires complete public labels")
    if candidate.status != "trained":
        return {
            "hard_pair_exact_count": 0.0,
            "hard_pair_exact_rate": 0.0,
            "hard_task_exact": 0.0,
            "hard_cell_accuracy": 0.0,
            "soft_pair_exact_count": 0.0,
            "soft_pair_exact_rate": 0.0,
            "soft_task_exact": 0.0,
            "soft_cell_accuracy": 0.0,
            "hardening_task_drop": 0.0,
        }
    hard = paired_grid_metrics(references, candidate.hard_predictions)
    soft = paired_grid_metrics(references, candidate.soft_predictions)
    return {
        **{"hard_" + key: value for key, value in hard.items()},
        **{"soft_" + key: value for key, value in soft.items()},
        "hardening_task_drop": float(soft["task_exact"] - hard["task_exact"]),
    }


def augmented_sparse_predictions(problem: ArcProblem, policy: str = "auto") -> tuple[list[np.ndarray], dict[str, Any]]:
    selection = select_augmentation_policy(problem.demonstrations)
    selected_policy = selection.policy if policy == "auto" else policy
    if selected_policy not in ("none", "d4", "bgpad", "d4_bgpad"):
        raise ValueError("unknown sparse augmentation policy")
    fitted = select_demo_rule(augment_examples(problem.demonstrations, selected_policy))
    if fitted.rule is None:
        return [], {
            "status": fitted.status,
            "augmentation_policy": selected_policy,
            "lodo_pair_exact": selection.lodo_pair_exact,
            "lodo_cell_accuracy": selection.lodo_cell_accuracy,
            "deployment_demo_task_exact": 0.0,
            "deployment_eligible": 0.0,
        }

    def deploy(grid: np.ndarray) -> np.ndarray:
        if selected_policy in ("bgpad", "d4_bgpad"):
            background = modal_color(grid)
            padded = np.pad(grid, 1, constant_values=background)
            return predict_direct(fitted.rule, padded)[1:-1, 1:-1]
        return predict_direct(fitted.rule, grid)

    demo_predictions = [deploy(np.asarray(example.input_grid)) for example in problem.demonstrations]
    demo_metrics = paired_grid_metrics(
        [np.asarray(example.output_grid) for example in problem.demonstrations],
        demo_predictions,
    )
    if demo_metrics["task_exact"] != 1.0:
        return [], {
            "status": "rejected_original_demo_mismatch",
            "augmentation_policy": selected_policy,
            "lodo_pair_exact": selection.lodo_pair_exact,
            "lodo_cell_accuracy": selection.lodo_cell_accuracy,
            "deployment_demo_task_exact": demo_metrics["task_exact"],
            "deployment_demo_cell_accuracy": demo_metrics["cell_accuracy"],
            "deployment_eligible": 0.0,
        }

    predictions = [deploy(np.asarray(grid)) for grid in problem.test_inputs]
    return predictions, {
        "status": "selected",
        "augmentation_policy": selected_policy,
        "lodo_pair_exact": selection.lodo_pair_exact,
        "lodo_cell_accuracy": selection.lodo_cell_accuracy,
        "deployment_demo_task_exact": demo_metrics["task_exact"],
        "deployment_demo_cell_accuracy": demo_metrics["cell_accuracy"],
        "deployment_eligible": 1.0,
        "neighborhood": fitted.rule.spec.name,
        "model_description_bits": fitted.rule.model_description_bits,
    }
