from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from arc_data import ArcExample  # noqa: E402
from arc_difflogic_features import (  # noqa: E402
    augment_examples,
    build_canvas_example,
    decode_color_channels,
    encode_color_channels,
    rank_shape_programs,
    select_augmentation_policy,
    select_shape_program,
)
from arc_difflogic_model import hard_numpy_ca_step  # noqa: E402
from trainable_difflogic import (  # noqa: E402
    GATE_NAMES,
    HardLogicLayerSpec,
    coverage_balanced_indices,
    gate_truth_tables,
    hard_numpy_layer,
)


class ArcDiffLogicDependencyLightTests(unittest.TestCase):
    def test_all_sixteen_gate_truth_tables_use_public_order(self) -> None:
        expected = np.asarray(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
                [0, 0, 1, 1],
                [0, 1, 0, 0],
                [0, 1, 0, 1],
                [0, 1, 1, 0],
                [0, 1, 1, 1],
                [1, 0, 0, 0],
                [1, 0, 0, 1],
                [1, 0, 1, 0],
                [1, 0, 1, 1],
                [1, 1, 0, 0],
                [1, 1, 0, 1],
                [1, 1, 1, 0],
                [1, 1, 1, 1],
            ],
            dtype=np.uint8,
        )
        self.assertEqual(len(GATE_NAMES), 16)
        np.testing.assert_array_equal(gate_truth_tables(), expected)

    def test_coverage_balanced_wiring_is_deterministic_and_covers_inputs(self) -> None:
        left, right = coverage_balanced_indices(17, 20, 123)
        again = coverage_balanced_indices(17, 20, 123)
        np.testing.assert_array_equal(left, again[0])
        np.testing.assert_array_equal(right, again[1])
        self.assertEqual(set(np.concatenate((left, right))), set(range(17)))
        self.assertFalse(bool(np.any(left == right)))

    def test_hard_layer_executes_each_gate_without_torch(self) -> None:
        inputs = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.uint8)
        for gate_id in range(16):
            with self.subTest(gate=GATE_NAMES[gate_id]):
                spec = HardLogicLayerSpec(2, 1, (0,), (1,), (gate_id,))
                np.testing.assert_array_equal(hard_numpy_layer(inputs, spec)[:, 0], gate_truth_tables()[gate_id])

    def test_binary4_canvas_roundtrip_and_invalid_fallback(self) -> None:
        grid = np.asarray([[0, 1, 9], [2, 3, 4]], dtype=np.uint8)
        bits = encode_color_channels(grid)
        decoded, valid = decode_color_channels(bits, np.zeros_like(grid))
        np.testing.assert_array_equal(decoded, grid)
        self.assertTrue(bool(np.all(valid)))
        invalid = bits.copy()
        invalid[:, 0, 0] = np.asarray([0, 1, 0, 1])  # code 10
        decoded, valid = decode_color_channels(invalid, np.full_like(grid, 7))
        self.assertEqual(int(decoded[0, 0]), 7)
        self.assertFalse(bool(valid[0, 0]))

    def test_shape_programs_are_demo_only_and_prefer_structured_scale(self) -> None:
        first = ArcExample(np.zeros((2, 3), dtype=np.uint8), np.zeros((4, 6), dtype=np.uint8))
        second = ArcExample(np.zeros((3, 2), dtype=np.uint8), np.zeros((6, 4), dtype=np.uint8))
        ranked = rank_shape_programs((first, second))
        self.assertEqual(ranked[0].name, "scale:2:2")
        self.assertEqual(ranked[0].predict_shape(np.zeros((4, 5), dtype=np.uint8)), (8, 10))

    def test_d4_augmentation_is_fixed_and_selector_uses_only_demos(self) -> None:
        a = np.zeros((3, 3), dtype=np.uint8)
        a[0, 0] = 1
        b = np.zeros((3, 3), dtype=np.uint8)
        b[0, 0] = 2
        c = np.zeros((3, 3), dtype=np.uint8)
        c[2, 2] = 1
        d = np.zeros((3, 3), dtype=np.uint8)
        d[2, 2] = 2
        examples = (ArcExample(a, b), ArcExample(c, d))
        self.assertEqual(len(augment_examples(examples, "d4")), 16)
        selected = select_augmentation_policy(examples)
        self.assertIn(selected.policy, ("none", "d4"))

    def test_canvas_features_separate_input_and_output_masks(self) -> None:
        source = np.asarray([[1, 0], [0, 0]], dtype=np.uint8)
        target = np.repeat(np.repeat(source, 2, axis=0), 2, axis=1)
        example = ArcExample(source, target)
        shape = select_shape_program((example,))
        self.assertIsNotNone(shape)
        canvas = build_canvas_example(
            source,
            target.shape,
            target.shape,
            (example,),
            shape,
            hidden_bits=2,
            target_grid=target,
            include_original=True,
            include_geometry=True,
            include_objects=True,
            include_context=True,
        )
        self.assertEqual(canvas.initial_state.shape, (6, 4, 4))
        self.assertEqual(float(canvas.static_features[0].sum()), 4.0)
        self.assertEqual(float(canvas.static_features[1].sum()), 16.0)
        self.assertGreater(canvas.static_features.shape[0], 20)

    def test_exported_identity_center_step_matches_binary4_state(self) -> None:
        grid = np.asarray([[1, 2], [3, 4]], dtype=np.uint8)
        state = encode_color_channels(grid)
        static = np.zeros((2, 2, 2), dtype=np.uint8)
        # unfold order is channel-major then 3x3 row-major; center is c*9+4.
        centers = tuple(channel * 9 + 4 for channel in range(4))
        spec = HardLogicLayerSpec(38, 4, centers, (0, 0, 0, 0), (3, 3, 3, 3))
        result = hard_numpy_ca_step(state, static, np.ones((2, 2), dtype=np.uint8), (spec,))
        np.testing.assert_array_equal(result, state)


try:
    import torch
    from trainable_difflogic import DifferentiableLogicLayer
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "optional PyTorch training dependency is absent")
class ArcDiffLogicTorchTests(unittest.TestCase):
    def test_straight_through_gate_has_finite_gradient(self) -> None:
        layer = DifferentiableLogicLayer(2, 4, wiring_seed=7)
        values = torch.tensor([[0.0, 1.0], [1.0, 0.0]], requires_grad=True)
        output = layer(values, temperature=0.7, mode="st")
        output.sum().backward()
        self.assertTrue(bool(torch.isfinite(values.grad).all()))
        self.assertTrue(bool(torch.isfinite(layer.logits.grad).all()))

    def test_argmax_hard_layer_matches_numpy_export(self) -> None:
        layer = DifferentiableLogicLayer(2, 6, wiring_seed=11)
        values = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        torch_result = layer(values, temperature=0.3, mode="hard").detach().numpy()
        numpy_result = hard_numpy_layer(values.numpy(), layer.hard_specification())
        np.testing.assert_array_equal(torch_result, numpy_result)


if __name__ == "__main__":
    unittest.main()
