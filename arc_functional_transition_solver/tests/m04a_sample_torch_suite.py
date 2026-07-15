"""Explicit CPU-fake sampler suite; excluded from default ``test_*.py`` discovery."""

from __future__ import annotations

import hashlib
import math
import tempfile
import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from pathlib import Path

import torch
from torch import Tensor

from afts_arc.blind import BlindTask
from afts_arc.grid import grid_to_lists
from afts_arc.m04a_contract import (
    OUTPUT_SHAPE_SEMANTICS_VERSION,
    canonical_sha256,
    mask_count_trace,
)
from afts_arc.m04a_evidence import (
    BLIND_INPUT_ROW_SCHEMA_VERSION,
    BLIND_SHAPE_SIDECAR_SCHEMA_VERSION,
    FIXED_DSL_SEMANTICS_VERSION,
    NO_SHAPE_PROPOSAL,
    SHAPE_READY,
    validate_blind_input_row,
    validate_blind_input_shape_sidecar,
)
from afts_arc.m04a_sample import (
    NONFINITE_LOGITS,
    CheckpointModel,
    NonFiniteLogitsError,
    PoolSamplingAborted,
    SamplerRuntime,
    _LaneState,
    _VERIFIED_CHECKPOINT_MODEL_TOKEN,
    _finalize_streamed_pool,
    _frozen_inference_context,
    _inverse_cdf,
    _model_execution_guard,
    _model_parameter_guard,
    _move_token_batch,
    _register_model_parameter_value_seal,
    _require_exact_production_pool_sink,
    _sample_lane_step,
    sample_blind_pool,
    sample_blind_test_pair,
)
from afts_arc.m04a_model import (
    MASK_TOKEN_ID,
    GridCMLM,
    parameter_state_sha256,
    tokenize_target_batch,
)
from afts_arc.m04a_pool_staging import M04APoolPairStagingSink
from afts_arc.manifest import serialize_jsonl
from afts_arc.task import ARCPair


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _blind_task() -> BlindTask:
    return BlindTask.from_observations(
        train=(ARCPair(input=((0, 1),), output=((1, 0),)),),
        test_inputs=(((2, 3),),),
    )


def _sidecar_row(
    task: BlindTask, shapes: tuple[tuple[int, int], ...]
) -> dict[str, object]:
    accepted = [
        {
            "shape_order": index,
            "shape_proposal_id": _digest(f"proposal-{index}-{height}x{width}"),
            "proposed_height": height,
            "proposed_width": width,
        }
        for index, (height, width) in enumerate(shapes)
    ]
    row: dict[str, object] = {
        "schema": BLIND_INPUT_ROW_SCHEMA_VERSION,
        "blind_task_id": task.task_id,
        "blind_content_sha256": task.blind_content_sha256,
        "test_index": 0,
        "test_input": grid_to_lists(task.test_inputs[0]),
        "test_input_sha256": canonical_sha256(grid_to_lists(task.test_inputs[0])),
        "dsl_candidate_count": 0,
        "candidate_count_zero_subgroup": True,
        "shape_proposals_pre_dedup": len(accepted),
        "shape_proposals_post_dedup": len(accepted),
        "accepted_shape_count": len(accepted),
        "shape_proposal_truncation_count": 0,
        "accepted_shapes": accepted,
        "status": SHAPE_READY if accepted else NO_SHAPE_PROPOSAL,
    }
    row["row_id"] = canonical_sha256(row)
    return validate_blind_input_row(row)


def _one_row_sidecar(row: dict[str, object]):
    rows = (row,)
    rows_sha256 = hashlib.sha256(serialize_jsonl(rows)).hexdigest()
    manifest: dict[str, object] = {
        "schema": BLIND_SHAPE_SIDECAR_SCHEMA_VERSION,
        "case_set_id": _digest("case-set"),
        "blind_artifact_manifest_sha256": _digest("blind-artifact"),
        "blind_tasks_sha256": _digest("blind-tasks"),
        "dsl_pool_artifact_manifest_sha256": _digest("dsl-artifact"),
        "dsl_pool_content_id": _digest("dsl-content"),
        "dsl_pool_spec_id": _digest("dsl-spec"),
        "dsl_semantics_version": FIXED_DSL_SEMANTICS_VERSION,
        "shape_proposer_semantics_version": OUTPUT_SHAPE_SEMANTICS_VERSION,
        "task_count": 1,
        "test_pair_count": 1,
        "candidate_count_zero_task_count": 1,
        "candidate_count_zero_test_pair_count": 1,
        "shape_supported_test_pair_count": int(row["accepted_shape_count"] > 0),
        "no_shape_test_pair_count": int(row["accepted_shape_count"] == 0),
        "rows_sha256": rows_sha256,
    }
    manifest["sidecar_id"] = canonical_sha256(manifest)
    return validate_blind_input_shape_sidecar(manifest, rows, fixture_mode=True)


@dataclass(frozen=True, slots=True)
class _FakeMemory:
    memory: Tensor
    key_padding_mask: Tensor
    grid_lengths: tuple[int, ...]

    @property
    def memory_length(self) -> int:
        return int(self.memory.shape[1])


class _DeterministicFakeModel:
    def __init__(
        self,
        *,
        nonfinite: bool = False,
        nonfinite_call: int | None = None,
        nonfinite_cell: int = 0,
    ) -> None:
        self.device = torch.device("cpu")
        self.training = True
        self.nonfinite = nonfinite
        self.nonfinite_call = nonfinite_call
        self.nonfinite_cell = nonfinite_cell
        self.encoder_calls = 0
        self.decoder_calls = 0
        self.decoder_batches: list[tuple[int, int, int]] = []

    def eval(self) -> "_DeterministicFakeModel":
        self.training = False
        return self

    def train(self, mode: bool = True) -> "_DeterministicFakeModel":
        self.training = mode
        return self

    def encode_task_memory(self, batch) -> _FakeMemory:
        self.encoder_calls += 1
        memory_length = sum(batch.lengths)
        return _FakeMemory(
            memory=torch.zeros((1, memory_length, 4), dtype=torch.float32),
            key_padding_mask=torch.zeros((1, memory_length), dtype=torch.bool),
            grid_lengths=batch.lengths,
        )

    def decode_target(self, batch, memory: _FakeMemory):
        self.decoder_calls += 1
        heights = set(batch.height_ids[:, 0].tolist())
        widths = set(batch.width_ids[:, 0].tolist())
        if len(heights) != 1 or len(widths) != 1:
            raise AssertionError("fake decoder received a mixed-shape batch")
        height = next(iter(heights))
        width = next(iter(widths))
        self.decoder_batches.append((height, width, batch.batch_size))
        cell_count = height * width
        logits = torch.empty(
            (batch.batch_size, cell_count, 10), dtype=torch.float32
        )
        for batch_index in range(batch.batch_size):
            for linear_index in range(cell_count):
                for color in range(10):
                    logits[batch_index, linear_index, color] = (
                        (color - 4.5) * 0.31
                        + ((linear_index + 2 * color) % 5) * 0.07
                    )
        if self.nonfinite and self.decoder_calls == 1:
            logits[0, 0, 0] = math.nan
        if self.nonfinite_call == self.decoder_calls:
            logits[0, self.nonfinite_cell, 0] = math.inf
        return SimpleNamespace(logits=logits)


def _checkpoint(model: _DeterministicFakeModel) -> CheckpointModel:
    return CheckpointModel.fixture(
        model, checkpoint_sha256=_digest("checkpoint")
    )


class M04aSamplerCpuFakeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(max(1, min(torch.get_num_threads(), 4)))

    def test_arbitrary_model_cannot_claim_a_production_checkpoint(self) -> None:
        with self.assertRaisesRegex(TypeError, "load_checkpoint_model"):
            CheckpointModel(
                model=_DeterministicFakeModel(),
                checkpoint_sha256=_digest("checkpoint"),
            )

    def test_device_seal_catches_data_mutation_and_finalizer_aborts(self) -> None:
        model = GridCMLM().eval()
        checkpoint = CheckpointModel(
            model=model,
            checkpoint_sha256=_digest("checkpoint"),
            weight_state_sha256=parameter_state_sha256(model),
            optimizer_step=7,
            fixture_mode=False,
            _parameter_guard=_model_parameter_guard(model),
            _execution_guard=_model_execution_guard(model),
            _parameter_value_registration_token=(
                _register_model_parameter_value_seal(model)
            ),
            _construction_token=_VERIFIED_CHECKPOINT_MODEL_TOKEN,
        )
        self.assertFalse(hasattr(checkpoint, "_parameter_value_seal"))
        self.assertFalse(hasattr(checkpoint, "_parameter_value_verifier"))
        cuda_boundary = SimpleNamespace(
            allow_cpu_test=False, device=torch.device("cuda:0")
        )
        checkpoint.assert_intact_for_runtime(
            cuda_boundary, verify_host_weight_sha256=False
        )
        parameter = next(model.parameters())
        original = parameter.detach().clone()
        original_version = int(parameter._version)
        parameter.data.add_(1.0)
        self.assertEqual(int(parameter._version), original_version)
        with self.assertRaisesRegex(RuntimeError, "device seal"):
            checkpoint.assert_intact_for_runtime(
                cuda_boundary, verify_host_weight_sha256=False
            )
        with tempfile.TemporaryDirectory() as temporary_directory:
            staging = Path(temporary_directory) / "guard-failure-staging"
            sink = M04APoolPairStagingSink(
                staging, expected_pairs=(("task", 0),)
            )
            with self.assertRaisesRegex(RuntimeError, "device seal"):
                _finalize_streamed_pool(
                    checkpoint,
                    cuda_boundary,
                    sink,
                    pair_cost_rows=(),
                )
            self.assertTrue((staging / "ABORTED.json").is_file())
            self.assertFalse((staging / "pair_materials_manifest.json").exists())

        parameter.data.copy_(original)
        self.assertEqual(int(parameter._version), original_version)
        checkpoint.assert_intact_for_runtime(
            cuda_boundary, verify_host_weight_sha256=True
        )

        attention = model.decoder_layers[0].self_attn
        original_num_heads = attention.num_heads
        attention.num_heads = original_num_heads // 2
        with self.assertRaisesRegex(RuntimeError, "execution path"):
            checkpoint.assert_intact_for_runtime(
                cuda_boundary, verify_host_weight_sha256=False
            )
        attention.num_heads = original_num_heads
        checkpoint.assert_intact_for_runtime(
            cuda_boundary, verify_host_weight_sha256=True
        )

    def test_production_sink_requires_exact_unmodified_methods(self) -> None:
        task = _blind_task()
        with tempfile.TemporaryDirectory() as temporary_directory:
            sink = M04APoolPairStagingSink(
                Path(temporary_directory) / "pair-staging",
                expected_pairs=((task.task_id, 0),),
            )
            self.assertIs(_require_exact_production_pool_sink(sink), sink)
            sink.append_pair = lambda **_: {}  # type: ignore[method-assign]
            with self.assertRaisesRegex(TypeError, "cannot be overridden"):
                _require_exact_production_pool_sink(sink)
            del sink.append_pair
            original_handle = sink._handles["lane_traces.jsonl"]
            sink._handles["lane_traces.jsonl"] = object()  # type: ignore[assignment]
            with self.assertRaisesRegex(TypeError, "file handle changed"):
                _require_exact_production_pool_sink(sink)
            sink._handles["lane_traces.jsonl"] = original_handle
            sink.abort(failure_code="TEST_COMPLETE")

    def test_k0_returns_no_encoder_decoder_or_lane(self) -> None:
        task = _blind_task()
        row = _sidecar_row(task, ())
        model = _DeterministicFakeModel()
        with self.assertRaisesRegex(ValueError, "explicit fixture sidecar"):
            sample_blind_test_pair(
                task,
                row,
                _checkpoint(model),
                runtime=SamplerRuntime.cpu_test(),
            )
        result = sample_blind_test_pair(
            task,
            row,
            _checkpoint(model),
            runtime=SamplerRuntime.cpu_test(),
            fixture_sidecar=_one_row_sidecar(row),
        )
        self.assertTrue(result.fixture_mode)
        self.assertEqual(model.encoder_calls, 0)
        self.assertEqual(model.decoder_calls, 0)
        self.assertEqual(result.traces, ())
        self.assertEqual(result.lane_rows, ())
        self.assertEqual(result.encoder_rows, ())
        self.assertEqual(result.decoder_rows, ())
        self.assertEqual(result.pair_cost_row["status"], NO_SHAPE_PROPOSAL)
        self.assertEqual(result.pair_cost_row["raw_lanes"], 0)

    def test_fixture_pool_cannot_write_production_pair_staging(self) -> None:
        task = _blind_task()
        row = _sidecar_row(task, ())
        sidecar = _one_row_sidecar(row)
        model = _DeterministicFakeModel()
        with tempfile.TemporaryDirectory() as directory:
            sink = M04APoolPairStagingSink(
                Path(directory) / "pair-staging",
                expected_pairs=((task.task_id, 0),),
            )
            try:
                with self.assertRaisesRegex(ValueError, "fixture pools cannot write"):
                    sample_blind_pool(
                        (task,),
                        sidecar,
                        _checkpoint(model),
                        runtime=SamplerRuntime.cpu_test(),
                        artifact_sink=sink,
                    )
            finally:
                sink.abort(failure_code="FIXTURE_STAGING_FORBIDDEN")

    def test_rng_scalar_draw_replay_and_greedy_zero_draw(self) -> None:
        task = _blind_task()
        row = _sidecar_row(task, ((2, 2),))
        model = _DeterministicFakeModel()
        result = sample_blind_test_pair(
            task,
            row,
            _checkpoint(model),
            runtime=SamplerRuntime.cpu_test(),
            fixture_sidecar=_one_row_sidecar(row),
        )
        self.assertEqual(model.encoder_calls, 1)
        self.assertEqual(model.decoder_calls, 96)
        self.assertEqual(len(result.lane_rows), 64)
        greedy_trace = result.traces[0]
        self.assertTrue(greedy_trace["greedy"])
        self.assertTrue(
            all(
                prediction["uniform_hex"] is None
                for step in greedy_trace["steps"]
                for prediction in step["predictions"]
            )
        )

        sampled_trace = result.traces[1]
        observed = [
            prediction["uniform_hex"]
            for step in sampled_trace["steps"]
            for prediction in step["predictions"]
        ]
        expected_count = sum(mask_count_trace(4)[:12])
        self.assertEqual(len(observed), expected_count)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(sampled_trace["seed_u64"])
        expected = [
            float(
                torch.rand((), dtype=torch.float64, generator=generator).item()
            ).hex()
            for _ in range(expected_count)
        ]
        self.assertEqual(observed, expected)
        self.assertEqual(len(sampled_trace["steps"]), 12)
        self.assertGreater(len(observed), 4)

    def test_strict_inverse_cdf_ties_remask_order_and_committed_cells(self) -> None:
        self.assertEqual(_inverse_cdf((0.25, 0.25, 0.5) + (0.0,) * 7, 0.25), 1)
        self.assertEqual(_inverse_cdf((0.0,) * 10, 0.75), 9)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(7)
        lane = _LaneState(
            shape_order=0,
            shape_proposal_id=_digest("two-cell-shape"),
            height=1,
            width=2,
            global_lane=0,
            local_lane=0,
            greedy=True,
            seed_u64=7,
            generator=generator,
            cells=["MASK", "MASK"],
            started_ns=0,
        )
        tied_logits = torch.zeros((2, 10), dtype=torch.float64)
        step8, _ = _sample_lane_step(lane, step=8, logits=tied_logits)
        self.assertEqual(
            [row["chosen_color"] for row in step8["predictions"]], [0, 0]
        )
        self.assertEqual(step8["remasked_indices"], [0])
        self.assertEqual(lane.cells, ["MASK", 0])

        next_logits = torch.zeros((2, 10), dtype=torch.float64)
        next_logits[0, 2] = 9.0
        next_logits[1, 9] = 99.0
        _sample_lane_step(lane, step=9, logits=next_logits)
        self.assertEqual(lane.cells[1], 0)

    def test_k3_has_108_shape_pure_batches_and_frozen_lane_offsets(self) -> None:
        task = _blind_task()
        row = _sidecar_row(task, ((1, 1), (1, 2), (2, 2)))
        model = _DeterministicFakeModel()
        result = sample_blind_test_pair(
            task,
            row,
            _checkpoint(model),
            runtime=SamplerRuntime.cpu_test(),
            fixture_sidecar=_one_row_sidecar(row),
        )
        self.assertEqual(model.encoder_calls, 1)
        self.assertEqual(model.decoder_calls, 108)
        self.assertEqual(len(result.encoder_rows), 1)
        self.assertEqual(len(result.decoder_rows), 108)
        self.assertEqual(len(result.lane_rows), 64)
        self.assertEqual(result.pair_cost_row["shape_lane_allocations"], [22, 21, 21])
        self.assertEqual(result.pair_cost_row["decoder_batch_calls"], 108)
        self.assertEqual(result.pair_cost_row["total_actual_batch_calls"], 109)
        self.assertEqual(
            [
                min(
                    lane["global_lane"]
                    for lane in result.lane_rows
                    if lane["shape_order"] == shape_order
                )
                for shape_order in range(3)
            ],
            [0, 22, 43],
        )
        self.assertTrue(all(batch_size <= 8 for _, _, batch_size in model.decoder_batches))
        self.assertEqual(
            {shape for height, width, _ in model.decoder_batches for shape in [(height, width)]},
            {(1, 1), (1, 2), (2, 2)},
        )

    def test_nonfinite_logits_abort_entire_pool_without_partial_result(self) -> None:
        task = _blind_task()
        row = _sidecar_row(task, ((2, 2),))
        sidecar = _one_row_sidecar(row)
        model = _DeterministicFakeModel(nonfinite=True)
        with self.assertRaises(PoolSamplingAborted) as context:
            sample_blind_pool(
                (task,),
                sidecar,
                _checkpoint(model),
                runtime=SamplerRuntime.cpu_test(),
            )
        self.assertEqual(context.exception.failure_code, NONFINITE_LOGITS)
        self.assertIsInstance(context.exception.__cause__, NonFiniteLogitsError)
        self.assertFalse(hasattr(context.exception, "pairs"))
        self.assertEqual(model.encoder_calls, 1)
        self.assertEqual(model.decoder_calls, 1)

        # At step 9, cell 1 in lane 0 is already committed for the two-cell
        # schedule.  The sampler must still reject a non-finite decoder logit there.
        late_model = _DeterministicFakeModel(
            nonfinite_call=65, nonfinite_cell=1
        )
        late_row = _sidecar_row(task, ((1, 2),))
        with self.assertRaises(NonFiniteLogitsError):
            sample_blind_test_pair(
                task,
                late_row,
                _checkpoint(late_model),
                runtime=SamplerRuntime.cpu_test(),
                fixture_sidecar=_one_row_sidecar(late_row),
            )
        self.assertEqual(late_model.decoder_calls, 65)

    def test_sampler_rejects_nonblind_task_payload(self) -> None:
        task = _blind_task()
        row = _sidecar_row(task, ())
        model = _DeterministicFakeModel()
        with self.assertRaisesRegex(TypeError, "BlindTask"):
            sample_blind_test_pair(
                {"test": [{"input": [[2]], "output": [[9]]}]},
                row,
                _checkpoint(model),
                runtime=SamplerRuntime.cpu_test(),
            )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for device-seal checks")
class M04aCheckpointIntegrityCudaTests(unittest.TestCase):
    def test_production_inference_context_preserves_transfer_versions(self) -> None:
        device = torch.device("cuda", torch.cuda.current_device())
        cpu_batch = tokenize_target_batch(
            (((MASK_TOKEN_ID, 1), (2, 3)),), device="cpu"
        )
        model = GridCMLM().to(device).eval()
        with _frozen_inference_context(device):
            moved, transfer_ns = _move_token_batch(cpu_batch, device=device)
            tensors = moved._all_tensors()
            self.assertGreater(transfer_ns, 0)
            self.assertTrue(all(not tensor.is_inference() for tensor in tensors))
            versions = tuple(int(tensor._version) for tensor in tensors)
            self.assertTrue(model.embed(moved).is_cuda)
            moved.token_ids.add_(0)
            self.assertEqual(int(moved.token_ids._version), versions[0] + 1)
            with self.assertRaisesRegex(RuntimeError, "validated transfer"):
                model.embed(moved)

    def test_parameter_buffer_data_mutation_and_final_abort(self) -> None:
        device = torch.device("cuda", torch.cuda.current_device())
        model = GridCMLM().to(device).eval()
        model.register_buffer(
            "_integrity_test_buffer", torch.ones(1, device=device)
        )
        checkpoint = CheckpointModel(
            model=model,
            checkpoint_sha256=_digest("cuda-checkpoint"),
            weight_state_sha256=parameter_state_sha256(model),
            optimizer_step=7,
            fixture_mode=False,
            _parameter_guard=_model_parameter_guard(model),
            _execution_guard=_model_execution_guard(model),
            _parameter_value_registration_token=(
                _register_model_parameter_value_seal(model)
            ),
            _construction_token=_VERIFIED_CHECKPOINT_MODEL_TOKEN,
        )
        runtime = SimpleNamespace(allow_cpu_test=False, device=device)
        checkpoint.assert_intact_for_runtime(
            runtime, verify_host_weight_sha256=False
        )

        buffer = model._integrity_test_buffer
        buffer.data.add_(1.0)
        with self.assertRaisesRegex(RuntimeError, "device seal"):
            checkpoint.assert_intact_for_runtime(
                runtime, verify_host_weight_sha256=False
            )
        buffer.data.sub_(1.0)

        parameter = next(model.parameters())
        parameter.data.add_(1.0)
        with tempfile.TemporaryDirectory() as temporary_directory:
            staging = Path(temporary_directory) / "cuda-guard-failure"
            sink = M04APoolPairStagingSink(
                staging, expected_pairs=(("task", 0),)
            )
            with self.assertRaisesRegex(RuntimeError, "device seal"):
                _finalize_streamed_pool(
                    checkpoint,
                    runtime,
                    sink,
                    pair_cost_rows=(),
                )
            self.assertTrue((staging / "ABORTED.json").is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
