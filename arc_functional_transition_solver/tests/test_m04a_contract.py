from __future__ import annotations

import ast
import dataclasses
import math
import unittest
from pathlib import Path

import afts_arc.m04a_contract as contract


class M04aContractTests(unittest.TestCase):
    def test_module_has_no_torch_import(self) -> None:
        source = Path(contract.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported_roots: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".", 1)[0])
        self.assertNotIn("torch", imported_roots)

    def test_all_frozen_version_and_schema_identities(self) -> None:
        self.assertEqual(contract.MODEL_SEMANTICS_VERSION, "afts-grid-cmlm/v0.1")
        self.assertEqual(contract.MODEL_STAGE, "M04a_global_masked_grid_generation")
        self.assertEqual(contract.SAMPLER_SEMANTICS_VERSION, "afts-maskgit-cosine/v0.1")
        self.assertEqual(
            contract.DATA_FOLDS_SEMANTICS_VERSION, "afts-m04a-parent-folds/v0.1"
        )
        self.assertEqual(
            contract.EPISODE_SEMANTICS_VERSION, "afts-grid-cmlm-episodes/v0.1"
        )
        self.assertEqual(
            contract.VALIDATION_SEMANTICS_VERSION, "afts-grid-cmlm-validation/v0.1"
        )
        self.assertEqual(
            contract.REARC_PARENT_SPLIT_SEMANTICS_VERSION,
            "m04a-rearc-parent-split/v0.1",
        )
        self.assertEqual(
            contract.TASK_ORBIT_SEMANTICS_VERSION,
            "afts-task-d4-color-orbit/v0.1",
        )
        self.assertEqual(
            contract.SAMPLE_ORBIT_SEMANTICS_VERSION,
            "afts-sample-d4-color-orbit/v0.1",
        )
        self.assertEqual(
            contract.OUTPUT_SHAPE_SEMANTICS_VERSION, "afts-output-shape/v0.1"
        )
        self.assertEqual(
            contract.TRAINING_COST_LEDGER_SCHEMA_VERSION,
            "afts-training-cost-ledger/v0.1",
        )
        self.assertEqual(contract.LANE_ROW_SCHEMA_VERSION, "afts-m04a-lane/v0.1")
        self.assertEqual(contract.LANE_TRACE_SCHEMA_VERSION, "afts-maskgit-trace/v0.1")
        self.assertEqual(
            contract.ENCODER_FORWARD_LEDGER_SCHEMA_VERSION,
            "afts-encoder-forward-ledger/v0.1",
        )
        self.assertEqual(
            contract.DECODER_FORWARD_LEDGER_SCHEMA_VERSION,
            "afts-decoder-forward-ledger/v0.1",
        )
        self.assertEqual(contract.PAIR_COST_SCHEMA_VERSION, "afts-m04a-pair-cost/v0.1")
        self.assertEqual(
            contract.POOL_SETUP_COST_SCHEMA_VERSION,
            "afts-m04a-pool-setup-cost/v0.1",
        )

    def test_model_parameter_arithmetic_closes_exactly(self) -> None:
        self.assertEqual(contract.D_MODEL, 256)
        self.assertEqual(contract.ATTENTION_HEADS, 8)
        self.assertEqual(contract.FFN_WIDTH, 1_024)
        self.assertEqual(contract.ENCODER_LAYER_COUNT, 3)
        self.assertEqual(contract.DECODER_LAYER_COUNT, 6)
        self.assertEqual(contract.TOKEN_EMBEDDING_COUNT, 13)
        self.assertEqual(contract.POSITION_EMBEDDING_COUNT, 31)
        self.assertEqual(contract.ROLE_EMBEDDING_COUNT, 4)
        self.assertEqual(contract.PAIR_SLOT_EMBEDDING_COUNT, 16)
        self.assertEqual(contract.QUERY_TARGET_PAIR_SLOT, 15)
        self.assertEqual(contract.OUTPUT_COLOR_COUNT, 10)
        self.assertEqual(contract.LAYER_NORM_EPSILON, 1e-5)
        self.assertEqual(contract.DROPOUT_PROBABILITY, 0.1)
        closure = contract.model_parameter_closure()
        self.assertEqual(closure.one_mha, 263_168)
        self.assertEqual(closure.one_ffn, 525_568)
        self.assertEqual(closure.one_encoder_layer, 789_760)
        self.assertEqual(closure.encoder_layers, 2_369_280)
        self.assertEqual(closure.one_decoder_layer, 1_053_440)
        self.assertEqual(closure.decoder_layers, 6_320_640)
        self.assertEqual(closure.transformer_core, 8_689_920)
        self.assertEqual(closure.embeddings, 40_192)
        self.assertEqual(closure.final_layer_norms, 1_024)
        self.assertEqual(closure.output_head, 2_570)
        self.assertEqual(closure.total, 8_733_706)
        self.assertEqual(contract.MODEL_PARAMETER_CLOSURE, closure)
        self.assertEqual(contract.MODEL_PARAMETER_COUNT, 8_733_706)

    def test_indexed_d4_exact_order_on_non_square_grid(self) -> None:
        grid = ((0, 1, 2), (3, 4, 5))
        expected = (
            ((0, 1, 2), (3, 4, 5)),
            ((3, 0), (4, 1), (5, 2)),
            ((5, 4, 3), (2, 1, 0)),
            ((2, 5), (1, 4), (0, 3)),
            ((2, 1, 0), (5, 4, 3)),
            ((5, 2), (4, 1), (3, 0)),
            ((3, 4, 5), (0, 1, 2)),
            ((0, 3), (1, 4), (2, 5)),
        )
        self.assertEqual(
            contract.D4_INDEX_NAMES,
            (
                "identity",
                "rotate90",
                "rotate180",
                "rotate270",
                "reflect_left_to_right",
                "reflect_left_to_right_rotate90",
                "reflect_left_to_right_rotate180",
                "reflect_left_to_right_rotate270",
            ),
        )
        self.assertEqual(tuple(contract.apply_indexed_d4(grid, i) for i in range(8)), expected)
        with self.assertRaisesRegex(ValueError, "0..7"):
            contract.apply_indexed_d4(grid, 8)

    def test_orbit_helpers_do_not_reject_dimension_quarantine_inputs(self) -> None:
        over_model_bound = tuple((index % 10,) for index in range(31))
        self.assertEqual(len(contract.apply_indexed_d4(over_model_bound, 0)), 31)
        self.assertRegex(
            contract.sample_orbit_id(over_model_bound, over_model_bound), r"^[0-9a-f]{64}$"
        )

    def test_sample_orbit_is_d4_and_global_color_invariant(self) -> None:
        input_grid = ((0, 1, 2), (3, 0, 1))
        output_grid = ((2, 2), (0, 3))
        expected = "e1659e29f112ad1846837f58f938bf84113765a715f7402b0f06d49525d537b5"
        self.assertEqual(contract.sample_orbit_id(input_grid, output_grid), expected)

        color_map = {0: 9, 1: 4, 2: 7, 3: 1}
        recolored_input = tuple(tuple(color_map[cell] for cell in row) for row in input_grid)
        recolored_output = tuple(tuple(color_map[cell] for cell in row) for row in output_grid)
        self.assertEqual(contract.sample_orbit_id(recolored_input, recolored_output), expected)
        for index in range(8):
            self.assertEqual(
                contract.sample_orbit_id(
                    contract.apply_indexed_d4(input_grid, index),
                    contract.apply_indexed_d4(output_grid, index),
                ),
                expected,
            )
        self.assertEqual(len(contract.sample_orbit_serializations(input_grid, output_grid)), 8)

    def test_task_orbit_preserves_pair_order_and_roles(self) -> None:
        first = (((0, 1, 1),), ((2, 0, 1),))
        second = (((3,), (0,)), ((0,), (3,)))
        test = [(((4, 4), (0, 5)), ((5, 0), (4, 4)))]
        first_id = contract.task_orbit_id([first, second], test)
        second_id = contract.task_orbit_id([second, first], test)
        self.assertEqual(
            first_id, "6307e12cb74bc5c0db98edc3f1383ec49128e9b5a7ac34b0ede19c103e106c0f"
        )
        self.assertEqual(
            second_id, "aa4ee46b14c755a43e2123ea75a2424cb342ba00c344ce079ea661ea585ad74b"
        )
        self.assertNotEqual(first_id, second_id)
        self.assertNotEqual(
            first_id,
            contract.task_orbit_id(test, [first, second]),
        )

    def test_zero_based_seed_derivations_are_pinned(self) -> None:
        self.assertEqual(contract.episode_seed(0, 0), 5_146_264_140_296_529_905)
        self.assertEqual(
            contract.episode_seed(
                contract.OPTIMIZER_UPDATES - 1,
                contract.GRADIENT_ACCUMULATION - 1,
            ),
            18_041_118_754_428_078_861,
        )
        self.assertEqual(
            contract.lane_seed("blind_demo", 0, "shape_demo", 1),
            2_885_997_511_158_729_330,
        )
        self.assertEqual(
            contract.validation_episode_seed("parent", "target", 0),
            13_244_406_371_838_620_521,
        )
        self.assertEqual(contract.rearc_parent_split_bucket("40853293"), 45)
        with self.assertRaises(ValueError):
            contract.episode_seed(contract.OPTIMIZER_UPDATES, 0)
        with self.assertRaises(ValueError):
            contract.episode_seed(0, 16)
        with self.assertRaises(ValueError):
            contract.lane_seed("blind_demo", 0, "shape_demo", 64)
        with self.assertRaises(ValueError):
            contract.validation_episode_seed("parent", "target", 4)

    def test_lane_allocations_plans_and_batch_calls_close(self) -> None:
        expected = {
            0: ((), 0, 0),
            1: ((64,), 64, 96),
            2: ((32, 32), 64, 96),
            3: ((22, 21, 21), 64, 108),
            4: ((16, 16, 16, 16), 64, 96),
        }
        for shape_count, (allocations, lane_count, batch_calls) in expected.items():
            with self.subTest(shape_count=shape_count):
                self.assertEqual(contract.shape_lane_allocations(shape_count), allocations)
                plan = contract.lane_plan(shape_count)
                self.assertEqual(len(plan), lane_count)
                self.assertEqual(contract.decoder_batch_calls(shape_count), batch_calls)
                self.assertEqual(
                    sum(entry.greedy for entry in plan),
                    shape_count,
                )
                self.assertEqual(
                    tuple(entry.global_lane for entry in plan), tuple(range(lane_count))
                )
        third = contract.lane_plan(3)
        self.assertEqual(third[21].local_lane, 21)
        self.assertEqual(third[22].shape_order, 1)
        self.assertEqual(third[22].local_lane, 0)
        self.assertTrue(third[22].greedy)
        with self.assertRaises(ValueError):
            contract.shape_lane_allocations(5)

    def test_complete_mask_schedule_is_pinned_and_monotone(self) -> None:
        self.assertEqual(
            contract.MASK_SCHEDULE_TABLE_SHA256,
            "fe74fb990c24252dcd3d798d4303aba869b2e4403e49d6d22e2fb4a609f350c6",
        )
        self.assertEqual(
            contract.mask_schedule_table_sha256(), contract.MASK_SCHEDULE_TABLE_SHA256
        )
        table = contract.mask_schedule_table()
        self.assertEqual(len(table), 900)
        for cell_count, trace in enumerate(table, start=1):
            self.assertEqual(len(trace), 13)
            self.assertEqual(trace[0], cell_count)
            self.assertEqual(trace[-1], 0)
            self.assertTrue(all(after <= before for before, after in zip(trace, trace[1:])))
            self.assertEqual(trace[8], (cell_count + 1) // 2)
            self.assertEqual(contract.masked_token_predictions(cell_count), sum(trace[:12]))
        self.assertEqual(contract.retained_mask_count(2, 8), 1)
        self.assertEqual(contract.mask_count_trace(1), (1,) * 12 + (0,))
        with self.assertRaises(ValueError):
            contract.mask_count_trace(901)

    def test_canonical_json_trace_and_float64_hashes(self) -> None:
        payload = {"b": [2, 1], "a": "é"}
        self.assertEqual(
            contract.canonical_json_bytes(payload), '{"a":"é","b":[2,1]}'.encode("utf-8")
        )
        with self.assertRaises(ValueError):
            contract.canonical_json_bytes({"not_finite": math.nan})
        trace = {"schema": contract.LANE_TRACE_SCHEMA_VERSION, "x": [1, "é"]}
        self.assertEqual(
            contract.trace_sha256(trace),
            "d72d05ba7e99ca89845c63951acb4707969d0a8923dfc5dbbd5eddc2d9cae5d8",
        )
        with self.assertRaisesRegex(ValueError, "schema"):
            contract.trace_sha256({"schema": "wrong"})
        self.assertEqual(
            contract.float64_sequence_sha256([0.0, 1.0, -2.5]),
            "dac183dbebdeba8a99ace5f31b2b38adca59fe5e1d5354147acd11db7eb1eaf5",
        )
        with self.assertRaises(ValueError):
            contract.float64_sequence_sha256([math.inf])

    def test_primary_training_cost_closure(self) -> None:
        closure = contract.TrainingCostClosure.expected(validation_episodes_per_pass=37)
        primary_microbatches = (
            contract.OPTIMIZER_UPDATES * contract.GRADIENT_ACCUMULATION
        )
        validation_calls = contract.VALIDATION_PASS_COUNT * 37
        self.assertEqual(closure.optimizer_updates, contract.OPTIMIZER_UPDATES)
        self.assertEqual(closure.microbatches, primary_microbatches)
        self.assertEqual(closure.arc2_episodes, primary_microbatches // 2)
        self.assertEqual(closure.rearc_episodes, primary_microbatches // 2)
        self.assertEqual(closure.encoder_forward_calls, primary_microbatches)
        self.assertEqual(closure.decoder_forward_calls, primary_microbatches)
        self.assertEqual(closure.backward_calls, primary_microbatches)
        self.assertEqual(closure.validation_passes, contract.VALIDATION_PASS_COUNT)
        self.assertEqual(closure.validation_episode_calls, validation_calls)
        self.assertEqual(closure.validation_encoder_forward_calls, validation_calls)
        self.assertEqual(closure.validation_decoder_forward_calls, validation_calls)
        self.assertEqual(closure.checkpoint_writes, contract.VALIDATION_PASS_COUNT)
        self.assertIs(contract.validate_training_cost_closure(closure), closure)
        with self.assertRaisesRegex(ValueError, "training cost closure"):
            dataclasses.replace(closure, microbatches=primary_microbatches - 1)

    def test_pair_cost_closure_for_no_shape_and_three_shapes(self) -> None:
        no_shape = contract.PairCostClosure.expected(shape_cell_counts=())
        self.assertEqual(no_shape.shape_lane_allocations, ())
        self.assertEqual(no_shape.shape_cell_counts, ())
        self.assertEqual(no_shape.raw_lanes, 0)
        self.assertEqual(no_shape.encoder_batch_calls, 0)
        self.assertEqual(no_shape.decoder_batch_calls, 0)
        self.assertEqual(no_shape.total_actual_batch_calls, 0)
        self.assertEqual(no_shape.candidate_rows, 0)

        closure = contract.PairCostClosure.expected(
            shape_cell_counts=(1, 2, 900),
            format_invalid=2,
            unique_outputs=10,
        )
        self.assertEqual(closure.shape_lane_allocations, (22, 21, 21))
        self.assertEqual(closure.shape_cell_counts, (1, 2, 900))
        self.assertEqual(closure.raw_lanes, 64)
        self.assertEqual(closure.sample_equivalent_forward_calls, 768)
        self.assertEqual(closure.encoder_batch_calls, 1)
        self.assertEqual(closure.decoder_batch_calls, 108)
        self.assertEqual(closure.total_actual_batch_calls, 109)
        self.assertEqual(closure.format_valid, 62)
        self.assertEqual(closure.format_invalid, 2)
        self.assertEqual(closure.unique_outputs, 10)
        self.assertEqual(closure.duplicate_outputs, 52)
        self.assertEqual(closure.candidate_rows, 64)
        self.assertEqual(len(closure.mask_traces), 64)
        self.assertEqual(closure.mask_traces[0], contract.mask_count_trace(1))
        self.assertEqual(closure.mask_traces[22], contract.mask_count_trace(2))
        self.assertEqual(closure.mask_traces[43], contract.mask_count_trace(900))
        self.assertIs(contract.validate_pair_cost_closure(closure), closure)
        with self.assertRaisesRegex(ValueError, "raw_lanes"):
            dataclasses.replace(closure, raw_lanes=63)
        all_single_cell_traces = tuple(
            contract.mask_count_trace(1) for _ in range(closure.raw_lanes)
        )
        all_single_cell_predictions = tuple(
            sum(trace[: contract.DENOISING_STEPS])
            for trace in all_single_cell_traces
        )
        with self.assertRaisesRegex(ValueError, "declared shape size"):
            dataclasses.replace(
                closure,
                mask_traces=all_single_cell_traces,
                masked_token_predictions_per_lane=all_single_cell_predictions,
            )
        with self.assertRaises(ValueError):
            contract.PairCostClosure.expected(shape_cell_counts=(1, 2, 3, 4, 5))


if __name__ == "__main__":
    unittest.main()
