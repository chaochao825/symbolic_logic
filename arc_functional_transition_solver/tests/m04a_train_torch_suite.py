"""Explicit M04a train/validation suite; excluded from ``test_*.py`` discovery."""

from __future__ import annotations

import hashlib
import math
import os
import struct
import tempfile
import unittest
from pathlib import Path

import torch

from afts_arc.m04a_contract import GRADIENT_ACCUMULATION
from afts_arc.m04a_data import (
    ARC2_SOURCE,
    REARC_SOURCE,
    ARC2Parent,
    M04AExample,
    M04ATrainingData,
    build_training_episode,
    build_validation_episodes,
)
from afts_arc.m04a_model import GridCMLM
from afts_arc.m04a_evidence import make_validation_metric_row
from afts_arc.m04a_train import (
    ADAMW_BETAS,
    ADAMW_EPSILON,
    ADAMW_WEIGHT_DECAY,
    aggregate_validation_fp32,
    assert_adamw_invariants,
    build_adamw_optimizer,
    make_training_mask_audit_row,
    train_primary_update,
    validate_literal_manifest,
    validate_training_mask_audit_row,
)
from afts_arc.m04a_train_contract import learning_rate_for_update
from afts_arc.m04a_torch_runtime import configure_deterministic_cuda
from afts_arc.m04a_validation_manifest import publish_validation_episode_manifest


class _TinyReARC:
    def __init__(self, example: M04AExample, *, semantic_parent_id: str) -> None:
        self._example = example
        self._semantic_parent_id = semantic_parent_id
        self._parent_ids = ("rearc-parent",)

    @property
    def parent_ids(self) -> tuple[str, ...]:
        return self._parent_ids

    def example(self, parent_id: str, example_index: int) -> M04AExample:
        if parent_id != self._parent_ids[0] or not 0 <= example_index < 1000:
            raise IndexError((parent_id, example_index))
        return self._example

    def semantic_parent_id(self, parent_id: str) -> str:
        if parent_id != self._parent_ids[0]:
            raise KeyError(parent_id)
        return self._semantic_parent_id


def _tiny_data(*, cross_source_alias: bool = False) -> M04ATrainingData:
    demonstration = M04AExample.create(((0,),), ((1,),))
    target = M04AExample.create(((2,),), ((3,),))
    parent_id = "arc-parent"
    parent = ARC2Parent(
        parent_id=parent_id,
        train=(demonstration,),
        test=(target,),
    )
    rearc = _TinyReARC(
        target,
        semantic_parent_id=parent_id if cross_source_alias else "rearc-parent",
    )
    return M04ATrainingData(
        arc2_parent_ids=(parent_id,),
        arc2_parents={parent_id: parent},
        rearc_parent_ids=rearc.parent_ids,
        rearc=rearc,
    )


class M04aTrainCpuContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(max(1, min(torch.get_num_threads(), 4)))

    def test_optimizer_is_one_exact_all_parameter_group(self) -> None:
        model = GridCMLM()
        optimizer = build_adamw_optimizer(model)
        self.assertIs(type(optimizer), torch.optim.AdamW)
        self.assertEqual(len(optimizer.param_groups), 1)
        group = optimizer.param_groups[0]
        self.assertEqual(
            [id(parameter) for parameter in group["params"]],
            [id(parameter) for parameter in model.parameters()],
        )
        self.assertEqual(group["betas"], ADAMW_BETAS)
        self.assertEqual(group["eps"], ADAMW_EPSILON)
        self.assertEqual(group["weight_decay"], ADAMW_WEIGHT_DECAY)
        self.assertFalse(group["amsgrad"])
        self.assertFalse(group["maximize"])
        self.assertFalse(group["foreach"])
        self.assertFalse(group["fused"])
        self.assertFalse(group["capturable"])
        self.assertFalse(group["differentiable"])
        self.assertEqual(optimizer.state, {})
        assert_adamw_invariants(
            optimizer, model, expected_completed_updates=0
        )
        with self.assertRaisesRegex(RuntimeError, "CUDA"):
            train_primary_update(
                model,
                optimizer,
                _tiny_data(),
                runtime_attestation=None,
                optimizer_step=1,
                event_index=0,
            )
        group["weight_decay"] = 0.0
        with self.assertRaisesRegex(RuntimeError, "weight_decay"):
            assert_adamw_invariants(optimizer, model)

    def test_mask_audit_replays_counter_source_and_hashes(self) -> None:
        data = _tiny_data()
        rows = []
        for slot in range(GRADIENT_ACCUMULATION):
            episode = build_training_episode(data, 0, slot)
            row = make_training_mask_audit_row(episode, optimizer_step=1)
            rows.append(row)
            self.assertEqual(row["microbatch_slot"], slot)
            self.assertEqual(
                row["source"], ARC2_SOURCE if slot % 2 == 0 else REARC_SOURCE
            )
            self.assertEqual(row["masked_linear_indices"], [0])
            self.assertEqual(row["masked_token_predictions"], 1)
            self.assertEqual(validate_training_mask_audit_row(row), row)
        replay = [
            make_training_mask_audit_row(
                build_training_episode(data, 0, slot), optimizer_step=1
            )
            for slot in range(GRADIENT_ACCUMULATION)
        ]
        self.assertEqual(rows, replay)
        tampered = dict(rows[0])
        tampered["episode_seed_u64"] += 1
        with self.assertRaisesRegex(ValueError, "seed"):
            validate_training_mask_audit_row(tampered)

    def test_fp32_hierarchy_is_views_then_targets_then_parents(self) -> None:
        losses = tuple(
            torch.tensor(value, dtype=torch.float32)
            for value in (1.0, 3.0, 9.0, 5.0)
        )
        aggregation = aggregate_validation_fp32(
            ("parent-a", "parent-a", "parent-a", "parent-b"),
            ("target-a", "target-a", "target-b", "target-c"),
            losses,
        )
        self.assertEqual(
            [metric.view_count for metric in aggregation.target_metrics], [2, 1, 1]
        )
        self.assertEqual(
            [metric.mean_masked_cell_ce for metric in aggregation.target_metrics],
            [2.0, 9.0, 5.0],
        )
        self.assertEqual(
            [metric.target_count for metric in aggregation.parent_metrics], [2, 1]
        )
        self.assertEqual(
            [metric.mean_masked_cell_ce for metric in aggregation.parent_metrics],
            [5.5, 5.0],
        )
        self.assertEqual(aggregation.parent_grouped_ce, 5.25)
        with self.assertRaisesRegex(ValueError, "contiguous"):
            aggregate_validation_fp32(
                ("parent-a", "parent-a", "parent-a"),
                ("target-a", "target-b", "target-a"),
                losses[:3],
            )

    def test_production_uses_pinned_five_value_one_ulp_sequential_mean(self) -> None:
        values = [
            struct.unpack("<f", struct.pack("<I", 0x3F800000 + offset))[0]
            for offset in range(5)
        ]
        aggregation = aggregate_validation_fp32(
            ("parent",) * 5,
            ("target",) * 5,
            tuple(torch.tensor(value, dtype=torch.float32) for value in values),
        )
        result_bits = struct.unpack(
            "<I", struct.pack("<f", aggregation.parent_grouped_ce)
        )[0]
        self.assertEqual(result_bits, 0x3F800002)

    def test_rounding_sensitive_hierarchy_matches_pure_metric_replay(self) -> None:
        def value_from_bits(bits: int) -> float:
            return struct.unpack("<f", struct.pack("<I", bits))[0]

        def bits_from_value(value: float) -> int:
            return struct.unpack("<I", struct.pack("<f", value))[0]

        parent_ids: list[str] = []
        target_ids: list[str] = []
        losses: list[torch.Tensor] = []
        episode_metrics: list[dict[str, object]] = []

        def append_view(parent_id: str, target_id: str, value_bits: int) -> None:
            ordinal = len(losses)
            value = value_from_bits(value_bits)
            parent_ids.append(parent_id)
            target_ids.append(target_id)
            losses.append(torch.tensor(value, dtype=torch.float32))
            episode_metrics.append(
                {
                    "ledger_event_id": hashlib.sha256(
                        f"event-{ordinal}".encode("ascii")
                    ).hexdigest(),
                    "row_ordinal": ordinal,
                    "episode_sha256": hashlib.sha256(
                        f"episode-{ordinal}".encode("ascii")
                    ).hexdigest(),
                    "semantic_parent_id": parent_id,
                    "target_group_id": target_id,
                    "masked_token_predictions": 1,
                    "masked_cell_ce_hex": value.hex(),
                }
            )

        base = 0x3F800102
        append_view("parent-0", "parent-0-target-0", base)
        for target_offset in range(5):
            target_bits = base + target_offset
            target_id = f"parent-1-target-{target_offset}"
            if target_offset == 1:
                for view_offset in range(5):
                    append_view("parent-1", target_id, base + view_offset)
            else:
                append_view("parent-1", target_id, target_bits)
        for parent_offset in range(2, 5):
            append_view(
                f"parent-{parent_offset}",
                f"parent-{parent_offset}-target-0",
                base + parent_offset,
            )

        aggregation = aggregate_validation_fp32(
            tuple(parent_ids), tuple(target_ids), tuple(losses)
        )
        metric = make_validation_metric_row(
            validation_pass_index=1,
            validation_episode_outer_manifest_sha256="1" * 64,
            validation_episode_manifest_sha256="2" * 64,
            episode_metrics=episode_metrics,
            checkpoint_sha256="3" * 64,
        )

        expected_target_bits = [
            base,
            base,
            base + 1,
            base + 2,
            base + 3,
            base + 4,
            base + 2,
            base + 3,
            base + 4,
        ]
        expected_parent_bits = [base + offset for offset in range(5)]
        self.assertEqual(
            [
                bits_from_value(target.mean_masked_cell_ce)
                for target in aggregation.target_metrics
            ],
            expected_target_bits,
        )
        self.assertEqual(
            [
                bits_from_value(parent.mean_masked_cell_ce)
                for parent in aggregation.parent_metrics
            ],
            expected_parent_bits,
        )
        self.assertEqual(bits_from_value(aggregation.parent_grouped_ce), base + 1)
        self.assertEqual(
            metric["parent_metrics"],
            [
                {
                    "semantic_parent_id": parent.semantic_parent_id,
                    "target_count": parent.target_count,
                    "mean_masked_cell_ce_hex": parent.mean_masked_cell_ce.hex(),
                }
                for parent in aggregation.parent_metrics
            ],
        )
        self.assertEqual(
            metric["parent_grouped_masked_cell_ce_hex"],
            aggregation.parent_grouped_ce.hex(),
        )

        single_round_bits = bits_from_value(
            sum(value_from_bits(base + offset) for offset in range(5)) / 5
        )
        self.assertEqual(single_round_bits, base + 2)
        self.assertNotEqual(single_round_bits, base + 1)


_CUDA_CONTRACT_READY = (
    torch.cuda.is_available()
    and torch.cuda.device_count() == 1
    and torch.cuda.is_bf16_supported()
    and os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
)


@unittest.skipUnless(
    _CUDA_CONTRACT_READY,
    "one BF16 CUDA device and CUBLAS_WORKSPACE_CONFIG=:4096:8 are required",
)
class M04aTrainCudaRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runtime_attestation = configure_deterministic_cuda()

    def tearDown(self) -> None:
        torch.cuda.empty_cache()

    def test_one_update_uses_sixteen_sequential_microbatches_and_fp32_state(self) -> None:
        model = GridCMLM().to("cuda").train()
        optimizer = build_adamw_optimizer(model)
        initial_bias = model.output_head.bias.detach().clone()
        result = train_primary_update(
            model,
            optimizer,
            _tiny_data(),
            runtime_attestation=self.runtime_attestation,
            optimizer_step=1,
            event_index=0,
        )
        self.assertEqual(result.optimizer_step, 1)
        self.assertEqual(result.learning_rate, learning_rate_for_update(1))
        self.assertTrue(math.isfinite(result.mean_masked_cell_ce))
        self.assertTrue(math.isfinite(result.gradient_norm_before_clip))
        self.assertEqual(result.masked_token_predictions, GRADIENT_ACCUMULATION)
        self.assertEqual(len(result.mask_audit_rows), GRADIENT_ACCUMULATION)
        self.assertEqual(
            [row["source"] for row in result.mask_audit_rows],
            [
                ARC2_SOURCE if slot % 2 == 0 else REARC_SOURCE
                for slot in range(GRADIENT_ACCUMULATION)
            ],
        )
        ledger = result.ledger_row
        self.assertEqual(ledger["phase"], "primary_update")
        self.assertEqual(ledger["optimizer_updates"], 1)
        self.assertEqual(ledger["microbatches"], GRADIENT_ACCUMULATION)
        self.assertEqual(ledger["arc2_episodes"], 8)
        self.assertEqual(ledger["rearc_episodes"], 8)
        self.assertEqual(ledger["encoder_forward_calls"], 16)
        self.assertEqual(ledger["decoder_forward_calls"], 16)
        self.assertEqual(ledger["backward_calls"], 16)
        self.assertFalse(torch.equal(initial_bias, model.output_head.bias))
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        self.assertTrue(all(parameter.dtype == torch.float32 for parameter in model.parameters()))
        assert_adamw_invariants(
            optimizer,
            model,
            expected_learning_rate=learning_rate_for_update(1),
            expected_completed_updates=1,
        )

    def test_literal_validation_restores_cpu_cuda_rng_and_training_mode(self) -> None:
        data = _tiny_data(cross_source_alias=True)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        manifest = publish_validation_episode_manifest(
            Path(temporary.name) / "validation-manifest",
            build_validation_episodes(data),
        )
        model = GridCMLM().to("cuda").train()
        torch.manual_seed(987654)
        torch.cuda.manual_seed_all(123456)
        cpu_before = torch.get_rng_state().clone()
        cuda_before = [state.clone() for state in torch.cuda.get_rng_state_all()]
        budget_checks = 0

        def check_budget() -> None:
            nonlocal budget_checks
            budget_checks += 1

        result = validate_literal_manifest(
            model,
            manifest,
            runtime_attestation=self.runtime_attestation,
            optimizer_step=2_000,
            validation_pass_index=1,
            event_index_start=10,
            expected_artifact_manifest_sha256=manifest.artifact_manifest_sha256,
            expected_jsonl_sha256=manifest.summary["jsonl_sha256"],
            budget_check=check_budget,
        )
        self.assertEqual(budget_checks, 2 * len(manifest.rows))
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(cpu_before, torch.get_rng_state()))
        self.assertEqual(len(cuda_before), len(torch.cuda.get_rng_state_all()))
        self.assertTrue(
            all(
                torch.equal(before, after)
                for before, after in zip(
                    cuda_before, torch.cuda.get_rng_state_all(), strict=True
                )
            )
        )
        self.assertEqual(len(result.episode_metrics), len(manifest.rows))
        self.assertEqual(len(result.ledger_rows), len(manifest.rows))
        self.assertEqual(
            [row["event_index"] for row in result.ledger_rows],
            list(range(10, 10 + len(manifest.rows))),
        )
        self.assertTrue(
            all(row["validation_encoder_forward_calls"] == 1 for row in result.ledger_rows)
        )
        self.assertTrue(
            all(row["validation_decoder_forward_calls"] == 1 for row in result.ledger_rows)
        )
        self.assertEqual(len(result.aggregation.parent_metrics), 1)
        self.assertTrue(math.isfinite(result.aggregation.parent_grouped_ce))
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
