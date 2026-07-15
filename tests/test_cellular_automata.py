"""Exact semantic regressions for the bounded cellular-automata layer."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cellular_automata import (  # noqa: E402
    BINARY_GATE16_NAMES,
    RULE110,
    all_binary_gates16,
    binary_gate16,
    boolean_wavefront_pathfind,
    boolean_wavefront_step,
    ca_complexity_accounting,
    eca_lut,
    eca_predict,
    eca_rollout,
    eca_step,
    encode_radius_neighborhoods,
    game_of_life_next_from_patches,
    game_of_life_rollout,
    game_of_life_step,
    moore_patches,
    radius_lut_predict,
    radius_lut_rollout,
    radius_lut_values,
    trajectory_metrics,
)
from run_cellular_automata import wavefront_circuit_ir  # noqa: E402


class CellularAutomataTests(unittest.TestCase):
    def test_all_sixteen_binary_gates_match_their_truth_tables(self) -> None:
        a = np.asarray([0, 0, 1, 1], dtype=np.uint8)
        b = np.asarray([0, 1, 0, 1], dtype=np.uint8)
        all_outputs = all_binary_gates16(a, b)
        self.assertEqual(all_outputs.shape, (4, 16))
        self.assertEqual(len({tuple(all_outputs[:, index]) for index in range(16)}), 16)
        for index, name in enumerate(BINARY_GATE16_NAMES):
            expected = np.asarray([(index >> bit) & 1 for bit in (3, 2, 1, 0)], dtype=np.uint8)
            self.assertTrue(np.array_equal(all_outputs[:, index], expected), name)
            self.assertTrue(np.array_equal(binary_gate16(index, a, b), expected), name)
            self.assertTrue(np.array_equal(binary_gate16(name, a, b), expected), name)

    def test_every_eca_integer_decodes_and_predicts_with_public_bit_order(self) -> None:
        codes = np.arange(8, dtype=np.uint8)
        # Columns are x0=right, x1=center, x2=left; x0 is the LSB.
        neighborhoods = ((codes[:, None] >> np.arange(3, dtype=np.uint8)) & 1).astype(np.uint8)
        self.assertTrue(np.array_equal(encode_radius_neighborhoods(neighborhoods, radius=1), codes))
        for rule in range(256):
            expected = np.asarray([(rule >> code) & 1 for code in range(8)], dtype=np.uint8)
            self.assertTrue(np.array_equal(eca_lut(rule), expected))
            self.assertTrue(np.array_equal(eca_predict(neighborhoods, rule), expected))

    def test_rule110_truth_table_and_single_seed_transition(self) -> None:
        self.assertTrue(
            np.array_equal(
                eca_lut(RULE110),
                np.asarray([0, 1, 1, 1, 0, 1, 1, 0], dtype=np.uint8),
            )
        )
        initial = np.asarray([0, 0, 0, 1, 0, 0, 0], dtype=np.uint8)
        expected = np.asarray([0, 0, 1, 1, 0, 0, 0], dtype=np.uint8)
        self.assertTrue(np.array_equal(eca_step(initial, RULE110), expected))
        rollout = eca_rollout(initial, RULE110, steps=1)
        self.assertTrue(np.array_equal(rollout, np.stack((initial, expected))))

    def test_radius_two_lut_uses_rightmost_to_leftmost_little_endian_order(self) -> None:
        neighborhoods = np.asarray(
            [
                [1, 0, 1, 0, 1],
                [0, 1, 0, 1, 0],
            ],
            dtype=np.uint8,
        )
        codes = encode_radius_neighborhoods(neighborhoods, radius=2)
        self.assertTrue(np.array_equal(codes, np.asarray([21, 10], dtype=np.uint64)))
        lut = np.zeros(32, dtype=np.uint8)
        lut[[10, 21]] = [1, 1]
        self.assertTrue(np.array_equal(radius_lut_predict(neighborhoods, lut, radius=2), [1, 1]))
        rule = (1 << 10) | (1 << 21)
        self.assertTrue(np.array_equal(radius_lut_values(rule, radius=2), lut))

    def test_one_dimensional_updates_are_synchronous(self) -> None:
        # ECA 170 copies the right neighbor.  An in-place sequential update
        # would not produce this one-cell cyclic shift.
        initial = np.asarray([1, 0, 0, 0], dtype=np.uint8)
        expected = np.asarray([0, 0, 0, 1], dtype=np.uint8)
        self.assertTrue(np.array_equal(eca_step(initial, 170), expected))
        trajectory = radius_lut_rollout(initial, 170, steps=4, radius=1)
        self.assertTrue(np.array_equal(trajectory[0], trajectory[4]))
        self.assertTrue(np.array_equal(trajectory[1], expected))

    def test_one_dimensional_rollout_preserves_leading_batch_axes(self) -> None:
        initial = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.uint8)
        expected = np.asarray([[0, 0, 0, 1], [1, 0, 0, 0]], dtype=np.uint8)
        self.assertTrue(np.array_equal(eca_step(initial, 170), expected))
        trajectory = eca_rollout(initial, 170, steps=2)
        self.assertEqual(trajectory.shape, (3, 2, 4))
        self.assertTrue(np.array_equal(trajectory[1], expected))

    def test_all_512_game_of_life_patches_match_counting_rule(self) -> None:
        codes = np.arange(512, dtype=np.uint16)
        flat = ((codes[:, None] >> np.arange(9, dtype=np.uint16)) & 1).astype(np.uint8)
        patches = flat.reshape(-1, 3, 3)
        center = flat[:, 4]
        neighbors = flat.sum(axis=1) - center
        expected = ((neighbors == 3) | ((center == 1) & (neighbors == 2))).astype(np.uint8)
        self.assertTrue(np.array_equal(game_of_life_next_from_patches(flat), expected))
        self.assertTrue(np.array_equal(game_of_life_next_from_patches(patches), expected))

    def test_moore_patches_are_row_major_and_share_the_gol_oracle(self) -> None:
        grid = np.asarray(
            [
                [1, 0, 1, 0, 0],
                [0, 1, 0, 1, 0],
                [1, 1, 0, 0, 1],
                [0, 0, 1, 1, 0],
                [0, 1, 0, 1, 1],
            ],
            dtype=np.uint8,
        )
        patches = moore_patches(grid, boundary="fixed")
        self.assertEqual(patches.shape, (5, 5, 9))
        self.assertTrue(np.array_equal(patches[2, 2], grid[1:4, 1:4].reshape(-1)))
        self.assertTrue(np.array_equal(game_of_life_next_from_patches(patches), game_of_life_step(grid)))

    def test_game_of_life_block_is_stable(self) -> None:
        block = np.zeros((6, 6), dtype=np.uint8)
        block[2:4, 2:4] = 1
        self.assertTrue(np.array_equal(game_of_life_step(block), block))
        rollout = game_of_life_rollout(block, steps=5)
        self.assertTrue(np.all(rollout == block[None, :, :]))

    def test_game_of_life_blinker_has_period_two(self) -> None:
        horizontal = np.zeros((7, 7), dtype=np.uint8)
        horizontal[3, 2:5] = 1
        vertical = np.zeros_like(horizontal)
        vertical[2:5, 3] = 1
        rollout = game_of_life_rollout(horizontal, steps=2)
        self.assertTrue(np.array_equal(rollout[1], vertical))
        self.assertTrue(np.array_equal(rollout[2], horizontal))

    def test_boolean_wavefront_finds_shortest_non_wrapping_path(self) -> None:
        passable = np.asarray(
            [
                [1, 1, 0, 0],
                [0, 1, 0, 1],
                [0, 1, 1, 1],
            ],
            dtype=np.uint8,
        )
        result = boolean_wavefront_pathfind(passable, (0, 0), (2, 3))
        self.assertTrue(result.target_reached)
        self.assertEqual(result.target_distance, 5)
        self.assertEqual(len(result.path), 6)
        self.assertEqual(result.path[0], (0, 0))
        self.assertEqual(result.path[-1], (2, 3))
        for left, right in zip(result.path[:-1], result.path[1:]):
            self.assertEqual(abs(left[0] - right[0]) + abs(left[1] - right[1]), 1)

        frontier = np.zeros_like(passable)
        frontier[0, 0] = 1
        next_frontier = boolean_wavefront_step(frontier, frontier, passable)
        self.assertTrue(np.array_equal(next_frontier, np.asarray([[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])))

    def test_declared_wavefront_circuit_matches_all_sixty_four_inputs(self) -> None:
        codes = np.arange(64, dtype=np.uint8)
        inputs = ((codes[:, None] >> np.arange(6, dtype=np.uint8)) & 1).astype(np.uint8)
        reached, passable, north, west, east, south = inputs.T
        expected = passable & (reached | north | west | east | south)
        circuit = wavefront_circuit_ir()
        self.assertEqual(len(circuit.nodes), 5)
        self.assertEqual(circuit.depth, 5)
        self.assertTrue(np.array_equal(circuit.evaluate(inputs), expected))

    def test_wavefront_respects_step_bound_and_blocked_target(self) -> None:
        passable = np.ones((1, 6), dtype=np.uint8)
        bounded = boolean_wavefront_pathfind(passable, (0, 0), (0, 5), max_steps=4)
        self.assertFalse(bounded.target_reached)
        self.assertEqual(bounded.target_distance, -1)
        blocked = passable.copy()
        blocked[0, 5] = 0
        unreachable = boolean_wavefront_pathfind(blocked, (0, 0), (0, 5))
        self.assertFalse(unreachable.target_reached)
        self.assertEqual(unreachable.path, ())

    def test_trajectory_metrics_identify_first_divergence(self) -> None:
        reference = np.zeros((3, 2, 2), dtype=np.uint8)
        prediction = reference.copy()
        prediction[1, 0, 0] = 1
        prediction[2, :, :] = 1
        metrics = trajectory_metrics(reference, prediction)
        self.assertEqual(metrics["first_divergence_step"], 1)
        self.assertEqual(metrics["exact_trajectory"], 0.0)
        self.assertEqual(metrics["final_cell_accuracy"], 0.0)
        self.assertAlmostEqual(metrics["cell_accuracy"], 7 / 12)
        exact = trajectory_metrics(reference, reference)
        self.assertEqual(exact["first_divergence_step"], -1)
        self.assertEqual(exact["exact_trajectory"], 1.0)

    def test_complexity_accounting_separates_static_and_dynamic_counts(self) -> None:
        result = ca_complexity_accounting(
            (16, 16),
            steps=20,
            gates_per_cell=5,
            state_bits_per_cell=8,
            update_fraction=0.6,
            model_description_bits=320,
            total_network_gates=100,
            reachable_network_gates=22,
            active_network_gates=5,
        )
        self.assertEqual(result["spatial_cells"], 256)
        self.assertEqual(result["single_state_storage_bits"], 2048)
        self.assertEqual(result["synchronous_double_buffer_bits"], 4096)
        self.assertEqual(result["expected_cell_updates"], 3072.0)
        self.assertEqual(result["expected_gate_evaluations"], 15360.0)
        self.assertAlmostEqual(result["amortized_model_bits_per_cell_update"], 320 / 3072)


if __name__ == "__main__":
    unittest.main(verbosity=2)
