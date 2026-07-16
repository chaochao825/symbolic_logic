"""Categorical CA induction, codec, and boundary regressions."""

from __future__ import annotations

import itertools
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arc_ca import (  # noqa: E402
    BOUNDARY_COLOR,
    FROZEN_NEIGHBORHOODS,
    decode_binary4,
    extract_neighborhoods,
    find_oracle_local_rule,
    predict_binary4,
    predict_direct,
    predict_onehot,
    rollout_rule,
    rule_from_transition_table,
    select_demo_rule,
)
from arc_data import ArcExample  # noqa: E402


class ArcCATests(unittest.TestCase):
    def test_frozen_neighborhood_names_and_offsets_are_unique(self) -> None:
        self.assertEqual(len({spec.name for spec in FROZEN_NEIGHBORHOODS}), len(FROZEN_NEIGHBORHOODS))
        self.assertTrue(all(spec.offsets[0] == (0, 0) for spec in FROZEN_NEIGHBORHOODS))
        self.assertEqual(len(FROZEN_NEIGHBORHOODS), 29)

    def test_distinct_boundary_symbol_does_not_wrap(self) -> None:
        spec = next(spec for spec in FROZEN_NEIGHBORHOODS if spec.name == "center_wire_-1_+0")
        grid = np.asarray([[1, 2], [3, 4]], dtype=np.uint8)
        patches = extract_neighborhoods(grid, spec)
        self.assertEqual(tuple(patches[0, 0]), (1, BOUNDARY_COLOR))
        self.assertEqual(tuple(patches[1, 1]), (4, 2))

    def test_public_grid_api_rejects_fractional_colors(self) -> None:
        spec = next(spec for spec in FROZEN_NEIGHBORHOODS if spec.name == "center")
        with self.assertRaises(ValueError):
            extract_neighborhoods(np.asarray([[0.5]]), spec)

    def test_demo_selection_compiles_recolor_without_semantic_loss(self) -> None:
        source = np.asarray([[0, 1, 2], [2, 1, 0]], dtype=np.uint8)
        target = np.where(source == 1, 3, source).astype(np.uint8)
        selected = select_demo_rule((ArcExample(source, target),))
        self.assertEqual(selected.status, "selected")
        self.assertEqual(selected.rule.spec.name, "center")
        direct = predict_direct(selected.rule, source)
        binary, valid4 = predict_binary4(selected.rule, source)
        onehot, valid10 = predict_onehot(selected.rule, source)
        self.assertTrue(np.array_equal(direct, target))
        self.assertTrue(np.array_equal(binary, target))
        self.assertTrue(np.array_equal(onehot, target))
        self.assertTrue(valid4.all())
        self.assertTrue(valid10.all())

    def test_global_collision_and_shape_change_are_honest_refusals(self) -> None:
        low = np.zeros((11, 11), dtype=np.uint8)
        high = np.ones((11, 11), dtype=np.uint8)
        high[3:8, 3:8] = 0
        collision = select_demo_rule((ArcExample(low, np.zeros_like(low)), ArcExample(high, np.ones_like(high))))
        self.assertEqual(collision.status, "no_deterministic_local_rule")
        resize = select_demo_rule((ArcExample(np.zeros((3, 3), dtype=np.uint8), np.zeros((2, 2), dtype=np.uint8)),))
        self.assertEqual(resize.status, "unsupported_shape_change")

    def test_posthoc_oracle_is_a_separate_explicit_api(self) -> None:
        source = np.asarray([[0, 1], [1, 0]], dtype=np.uint8)
        examples = (ArcExample(source, 1 - source),)
        self.assertIsNotNone(find_oracle_local_rule(examples))

    def test_binary4_decoder_marks_unused_arc_codes_invalid(self) -> None:
        bits = ((np.asarray([9, 10, 15], dtype=np.uint8)[:, None] >> np.arange(4)) & 1).astype(np.uint8)
        colors, valid = decode_binary4(bits)
        self.assertEqual(colors.tolist(), [9, 10, 15])
        self.assertEqual(valid.tolist(), [True, False, False])

    def test_rollout_reports_fixed_point_and_cycle(self) -> None:
        center = next(spec for spec in FROZEN_NEIGHBORHOODS if spec.name == "center")
        identity = rule_from_transition_table(center, {(color,): color for color in range(10)})
        fixed = rollout_rule(identity, np.asarray([[1, 2]], dtype=np.uint8), 4)
        self.assertEqual(fixed.stop_reason, "fixed_point")
        self.assertEqual(fixed.steps_executed, 1)
        toggle = rule_from_transition_table(center, {(0,): 1, (1,): 0})
        cycle = rollout_rule(toggle, np.asarray([[0]], dtype=np.uint8), 4, stop="cycle")
        self.assertEqual(cycle.stop_reason, "cycle_length_2")

    def test_complete_small_valid_domain_codec_equivalence(self) -> None:
        spec = next(spec for spec in FROZEN_NEIGHBORHOODS if spec.name == "center_wire_-1_+0")
        table = {(a, b): (a if b == BOUNDARY_COLOR else b) for a in range(10) for b in range(11)}
        rule = rule_from_transition_table(spec, table)
        for a, b in itertools.product(range(10), repeat=2):
            grid = np.asarray([[b], [a]], dtype=np.uint8)
            direct = predict_direct(rule, grid)
            binary, _ = predict_binary4(rule, grid)
            onehot, _ = predict_onehot(rule, grid)
            self.assertTrue(np.array_equal(direct, binary))
            self.assertTrue(np.array_equal(direct, onehot))


if __name__ == "__main__":
    unittest.main()
