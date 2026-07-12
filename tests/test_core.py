"""Regression checks for exact semantics and the intended counterexample."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from logic_core import (  # noqa: E402
    GateBeamSynthesizer,
    SoftGateCircuit,
    all_assignments,
    compositional_rule,
    packed_compositional_rule,
    parity_rule,
    soft_probability_rule,
    soft_parity_probability,
)
from run_experiments import (  # noqa: E402
    _dense_relation_count,
    _fixed_k_reachable,
    _fixed_k_cyclic,
    _cyclic_reachable,
    _make_layered_graph_with_features,
    _indexed_relation_count,
    _planning_bfs,
    _reachable,
)


class LogicCoreTests(unittest.TestCase):
    def test_soft_probability_equals_rule_on_hard_bits(self) -> None:
        x = all_assignments(8)
        expected = compositional_rule(x)
        actual = soft_probability_rule(x.astype(float))
        self.assertTrue(np.array_equal(actual.astype(np.uint8), expected))

    def test_assignment_enumerator_preserves_seventeenth_bit(self) -> None:
        x = all_assignments(17)
        self.assertEqual(x.shape, (1 << 17, 17))
        self.assertEqual(len(np.unique(x, axis=0)), 1 << 17)

    def test_packed_rule_preserves_truth_table(self) -> None:
        x = all_assignments(8)
        packed = [np.packbits(x[:, i], bitorder="little") for i in range(8)]
        actual = np.unpackbits(packed_compositional_rule(packed), bitorder="little")[: len(x)]
        self.assertTrue(np.array_equal(actual, compositional_rule(x)))

    def test_packed_rule_preserves_non_byte_aligned_batch(self) -> None:
        rng = np.random.default_rng(11)
        x = rng.integers(0, 2, size=(13, 8), dtype=np.uint8)
        packed = [np.packbits(x[:, i], bitorder="little") for i in range(8)]
        actual = np.unpackbits(packed_compositional_rule(packed), bitorder="little")[: len(x)]
        self.assertTrue(np.array_equal(actual, compositional_rule(x)))

    def test_soft_probability_matches_enumerated_bernoulli_semantics(self) -> None:
        assignments = all_assignments(8)
        probabilities = np.array([0.15, 0.35, 0.55, 0.80, 0.20, 0.60, 0.70, 0.10])
        assignment_weights = np.prod(np.where(assignments == 1, probabilities, 1 - probabilities), axis=1)
        expected = float(np.sum(assignment_weights * compositional_rule(assignments)))
        actual = float(soft_probability_rule(probabilities[None, :])[0])
        self.assertAlmostEqual(actual, expected, places=12)

    def test_soft_parity_matches_enumerated_bernoulli_semantics(self) -> None:
        assignments = all_assignments(6)
        probabilities = np.array([0.15, 0.35, 0.55, 0.80, 0.20, 0.60])
        weights = np.prod(np.where(assignments == 1, probabilities, 1 - probabilities), axis=1)
        expected = float(np.sum(weights * parity_rule(assignments)))
        actual = float(soft_parity_probability(probabilities[None, :])[0])
        self.assertAlmostEqual(actual, expected, places=12)

    def test_beam_can_recover_known_compact_rule_from_full_table(self) -> None:
        x = all_assignments(8)
        learner = GateBeamSynthesizer(max_depth=4, beam_width=192).fit(x, compositional_rule(x))
        self.assertTrue(np.array_equal(learner.predict(x), compositional_rule(x)))
        self.assertIsNotNone(learner.expression)
        assert learner.expression is not None
        self.assertLessEqual(learner.expression.depth, 4)

    def test_learned_soft_gate_hardening_matches_clean_truth_table(self) -> None:
        x = all_assignments(8)
        y = compositional_rule(x)
        learner = SoftGateCircuit(seed=4, steps=600).fit(x, y)
        soft = (learner.predict_proba(x) >= 0.5).astype(np.uint8)
        hard = learner.predict_hardened(x)
        self.assertGreaterEqual(float(np.mean(soft == y)), 0.99)
        self.assertGreaterEqual(float(np.mean(hard == y)), 0.99)

    def test_indexed_filter_matches_dense_filter(self) -> None:
        rng = np.random.default_rng(3)
        color = rng.integers(0, 8, size=256)
        shape = rng.integers(0, 4, size=256)
        position = rng.integers(0, 1024, size=256)
        dense = _dense_relation_count(color, shape, position, 8)
        indexed, _ = _indexed_relation_count(color, shape, position, 8)
        self.assertEqual(dense, indexed)

    def test_fixed_depth_does_not_confuse_layer_indices(self) -> None:
        # A three-hop path reaches terminal node index 0.  After only two hops a
        # reused local index 0 must not be treated as a terminal-layer success.
        valid = np.zeros((3, 2, 2), dtype=bool)
        valid[0, 0, 1] = True
        valid[1, 1, 0] = True
        valid[2, 0, 0] = True
        self.assertTrue(_reachable(valid, 0)[0])
        self.assertFalse(_fixed_k_reachable(valid, 0, 2)[0])
        self.assertTrue(_fixed_k_reachable(valid, 0, 3)[0])

    def test_featured_layered_graph_has_balanced_query_targets(self) -> None:
        features, present, source, positive_target, negative_target = _make_layered_graph_with_features(
            8, 4, np.random.default_rng(19)
        )
        valid = present & compositional_rule(features.reshape(-1, 8)).reshape(4, 8, 8).astype(bool)
        reached = _reachable(valid, source)
        self.assertTrue(bool(reached[positive_target]))
        self.assertFalse(bool(reached[negative_target]))

    def test_looped_state_transition_reaches_cycle_without_fixed_unroll(self) -> None:
        adjacency = np.zeros((5, 5), dtype=bool)
        adjacency[0, 1] = adjacency[1, 2] = adjacency[2, 3] = adjacency[3, 4] = True
        adjacency[4, 1] = True
        reached, _, iterations = _cyclic_reachable(adjacency, 0)
        self.assertTrue(bool(reached[4]))
        self.assertGreater(iterations, 4)
        self.assertFalse(bool(_fixed_k_cyclic(adjacency, 0, 2)[4]))

    def test_planning_frontier_loop_and_blocked_goal_semantics(self) -> None:
        positive = 0b111111
        blocked = positive | (1 << 7)
        self.assertTrue(_planning_bfs(8, 7, positive, max_steps=None)[0])
        self.assertFalse(_planning_bfs(8, 7, positive, max_steps=4)[0])
        self.assertFalse(_planning_bfs(8, 7, blocked, max_steps=None)[0])

    def test_shared_wire_probability_is_not_independent_product(self) -> None:
        p = 0.37
        exact_and = p  # P(x AND x)
        exact_or = p  # P(x OR x)
        self.assertGreater(abs(exact_and - p * p), 0.1)
        self.assertGreater(abs(exact_or - (1.0 - (1.0 - p) ** 2)), 0.1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
