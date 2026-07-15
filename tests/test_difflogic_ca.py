"""Offline regressions for the Google DiffLogic-CA DigitalJS adapter."""

from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from difflogic_ca import (  # noqa: E402
    GOOGLE_DIFFLOGIC_CA_ARTIFACTS,
    GOOGLE_DIFFLOGIC_CA_COMMIT,
    ArtifactIntegrityError,
    DigitalJSCircuitFormatError,
    download_json_with_sha256,
    flatten_3x3_patches,
    parse_digitaljs_json,
    sha256_hexdigest,
    synchronous_ca_rollout,
    synchronous_ca_step,
)


def _binary_fixture(gate_type: str) -> dict:
    devices = {
        "output": {"type": "Output", "order": 0},
        "gate": {"type": gate_type},
        "b": {"type": "Button", "order": 1},
        "a": {"type": "Button", "order": 0},
    }
    connectors = [
        {"from": {"id": "gate", "port": "out"}, "to": {"id": "output", "port": "in"}},
        {"from": {"id": "a", "port": "out"}, "to": {"id": "gate", "port": "in1"}},
        {"from": {"id": "b", "port": "out"}, "to": {"id": "gate", "port": "in2"}},
    ]
    return {"devices": devices, "connectors": connectors, "subcircuits": {}}


def _unary_fixture(gate_type: str) -> dict:
    return {
        "devices": {
            "output": {"type": "Output", "order": 0},
            "gate": {"type": gate_type},
            "a": {"type": "Button", "order": 0},
        },
        "connectors": [
            {"from": {"id": "gate", "port": "out"}, "to": {"id": "output", "port": "in"}},
            {"from": {"id": "a", "port": "out"}, "to": {"id": "gate", "port": "in"}},
        ],
        "subcircuits": {},
    }


class DiffLogicCAAdapterTests(unittest.TestCase):
    def test_google_artifact_urls_are_commit_pinned_and_have_sha256(self) -> None:
        self.assertEqual(len(GOOGLE_DIFFLOGIC_CA_COMMIT), 40)
        self.assertNotIn("master", GOOGLE_DIFFLOGIC_CA_COMMIT)
        self.assertIn("gol_circuit.json", GOOGLE_DIFFLOGIC_CA_ARTIFACTS)
        self.assertIn("checkerboard.json", GOOGLE_DIFFLOGIC_CA_ARTIFACTS)
        for artifact in GOOGLE_DIFFLOGIC_CA_ARTIFACTS.values():
            self.assertIn(GOOGLE_DIFFLOGIC_CA_COMMIT, artifact.url)
            self.assertEqual(len(artifact.sha256), 64)

    def test_integrity_checked_downloader_uses_an_offline_opener(self) -> None:
        payload = json.dumps(
            {"devices": {}, "connectors": [], "subcircuits": {}},
            sort_keys=True,
        ).encode("utf-8")
        digest = sha256_hexdigest(payload)

        def opener(url: str, timeout: float) -> io.BytesIO:
            self.assertEqual(url, "https://example.invalid/fixture.json")
            self.assertGreater(timeout, 0)
            return io.BytesIO(payload)

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "fixture.json"
            result = download_json_with_sha256(
                "https://example.invalid/fixture.json",
                destination,
                digest,
                opener=opener,
            )
            self.assertEqual(result.read_bytes(), payload)

    def test_integrity_mismatch_never_writes_destination(self) -> None:
        payload = b'{"devices": {}, "connectors": []}'

        def opener(url: str, timeout: float) -> io.BytesIO:
            return io.BytesIO(payload)

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "fixture.json"
            with self.assertRaises(ArtifactIntegrityError):
                download_json_with_sha256(
                    "https://example.invalid/fixture.json",
                    destination,
                    "0" * 64,
                    opener=opener,
                )
            self.assertFalse(destination.exists())

    def test_all_published_boolean_gate_types_match_truth_tables(self) -> None:
        inputs = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.uint8)
        expected = {
            "And": [0, 0, 0, 1],
            "Or": [0, 1, 1, 1],
            "Xor": [0, 1, 1, 0],
            "Xnor": [1, 0, 0, 1],
            "Nand": [1, 1, 1, 0],
            "Nor": [1, 0, 0, 0],
            "AAndNotB": [0, 0, 1, 0],
            "NotAAndB": [0, 1, 0, 0],
            "AOrNotB": [1, 0, 1, 1],
            "NotAOrB": [1, 1, 0, 1],
            "A": [0, 0, 1, 1],
            "B": [0, 1, 0, 1],
            "NotA": [1, 1, 0, 0],
            "NotB": [1, 0, 1, 0],
        }
        for gate_type, truth_table in expected.items():
            with self.subTest(gate_type=gate_type):
                circuit = parse_digitaljs_json(_binary_fixture(gate_type))
                actual = circuit.evaluate(inputs)[:, 0]
                self.assertTrue(np.array_equal(actual, truth_table))

        unary_not = parse_digitaljs_json(_unary_fixture("Not"))
        self.assertTrue(
            np.array_equal(
                unary_not.evaluate(np.asarray([[0], [1]], dtype=np.uint8))[:, 0],
                [1, 0],
            )
        )

    def test_constants_are_batched(self) -> None:
        for gate_type, value in (("False", 0), ("True", 1)):
            document = {
                "devices": {
                    "a": {"type": "Button", "order": 0},
                    "constant": {"type": gate_type},
                    "output": {"type": "Output", "order": 0},
                },
                "connectors": [
                    {
                        "from": {"id": "constant", "port": "out"},
                        "to": {"id": "output", "port": "in"},
                    }
                ],
                "subcircuits": {},
            }
            result = parse_digitaljs_json(document).evaluate(np.zeros((7, 1), dtype=np.uint8))
            self.assertTrue(np.array_equal(result, np.full((7, 1), value, dtype=np.uint8)))

    def test_device_dictionary_order_does_not_control_io_or_evaluation(self) -> None:
        document = {
            "devices": {
                "out_xor": {"type": "Output", "order": 1},
                "xor": {"type": "Xor"},
                "input_b": {"type": "Button", "order": 2},
                "out_and": {"type": "Output", "order": 0},
                "and": {"type": "And"},
                "input_a": {"type": "Button", "order": 0},
            },
            "connectors": [
                {"from": {"id": "xor", "port": "out"}, "to": {"id": "out_xor", "port": "in"}},
                {"from": {"id": "and", "port": "out"}, "to": {"id": "out_and", "port": "in"}},
                {"from": {"id": "input_a", "port": "out"}, "to": {"id": "xor", "port": "in1"}},
                {"from": {"id": "input_b", "port": "out"}, "to": {"id": "xor", "port": "in2"}},
                {"from": {"id": "input_a", "port": "out"}, "to": {"id": "and", "port": "in1"}},
                {"from": {"id": "input_b", "port": "out"}, "to": {"id": "and", "port": "in2"}},
            ],
            "subcircuits": {},
        }
        circuit = parse_digitaljs_json(json.dumps(document))
        self.assertEqual(circuit.input_orders, (0, 2))
        self.assertEqual(circuit.output_orders, (0, 1))
        self.assertEqual(circuit.required_input_width, 3)
        ordered_inputs = np.asarray([[1, 0, 1], [1, 1, 0]], dtype=np.uint8)
        self.assertTrue(
            np.array_equal(
                circuit.evaluate(ordered_inputs),
                np.asarray([[1, 0], [0, 1]], dtype=np.uint8),
            )
        )
        positions = {device_id: i for i, device_id in enumerate(circuit.topological_device_ids)}
        self.assertLess(positions["input_a"], positions["and"])
        self.assertLess(positions["and"], positions["out_and"])

    def test_identical_duplicate_connectors_are_idempotent(self) -> None:
        document = _binary_fixture("And")
        document["connectors"].append(dict(document["connectors"][1]))
        circuit = parse_digitaljs_json(document)
        inputs = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.uint8)
        self.assertTrue(np.array_equal(circuit.evaluate(inputs)[:, 0], [0, 0, 0, 1]))
        self.assertEqual(circuit.raw_connector_count, 4)
        self.assertEqual(circuit.unique_connector_count, 3)
        self.assertEqual(circuit.logic_node_count, 1)
        self.assertEqual(circuit.critical_depth, 1)
        self.assertEqual(
            circuit.device_type_counts,
            {"And": 1, "Button": 2, "Output": 1},
        )
        stats = circuit.stats()
        self.assertEqual(stats["logic_node_count"], 1)
        stats["device_type_counts"]["And"] = 99
        self.assertEqual(circuit.device_type_counts["And"], 1)

    def test_different_sources_cannot_drive_the_same_target_port(self) -> None:
        document = _binary_fixture("And")
        document["connectors"].append(
            {"from": {"id": "b", "port": "out"}, "to": {"id": "gate", "port": "in1"}}
        )
        with self.assertRaises(DigitalJSCircuitFormatError):
            parse_digitaljs_json(document)

    def test_cycle_is_rejected(self) -> None:
        document = {
            "devices": {
                "a": {"type": "Not"},
                "b": {"type": "Not"},
                "output": {"type": "Output", "order": 0},
            },
            "connectors": [
                {"from": {"id": "b", "port": "out"}, "to": {"id": "a", "port": "in"}},
                {"from": {"id": "a", "port": "out"}, "to": {"id": "b", "port": "in"}},
                {"from": {"id": "a", "port": "out"}, "to": {"id": "output", "port": "in"}},
            ],
            "subcircuits": {},
        }
        with self.assertRaises(DigitalJSCircuitFormatError):
            parse_digitaljs_json(document)

    def test_patch_flattening_is_spatial_major_order_f_times_c_plus_c(self) -> None:
        patch = np.arange(18).reshape(3, 3, 2)
        flattened = flatten_3x3_patches(patch)
        self.assertTrue(np.array_equal(flattened, np.arange(18)))
        for spatial_index in range(9):
            for channel in range(2):
                self.assertEqual(flattened[spatial_index * 2 + channel], patch.reshape(9, 2)[spatial_index, channel])

    def test_patch_flattening_channel_major_is_c_times_nine_plus_f(self) -> None:
        patch = np.arange(18).reshape(3, 3, 2)
        flattened = flatten_3x3_patches(patch, layout="channel_major")
        for spatial_index in range(9):
            for channel in range(2):
                self.assertEqual(
                    flattened[channel * 9 + spatial_index],
                    patch.reshape(9, 2)[spatial_index, channel],
                )

    def test_ca_step_is_synchronous_and_uses_f_times_c_plus_c_order(self) -> None:
        # Output channel 0 copies the right neighbor's channel 0: f=5,c=0.
        # Output channel 1 copies the left neighbor's channel 1:  f=3,c=1.
        document = {
            "devices": {
                "right_c0": {"type": "Button", "order": 5 * 2 + 0},
                "left_c1": {"type": "Button", "order": 3 * 2 + 1},
                "output_c1": {"type": "Output", "order": 1},
                "output_c0": {"type": "Output", "order": 0},
            },
            "connectors": [
                {"from": {"id": "left_c1", "port": "out"}, "to": {"id": "output_c1", "port": "in"}},
                {"from": {"id": "right_c0", "port": "out"}, "to": {"id": "output_c0", "port": "in"}},
            ],
            "subcircuits": {},
        }
        circuit = parse_digitaljs_json(document)
        grid = np.asarray(
            [
                [[1, 0], [0, 1], [0, 0], [1, 1]],
                [[0, 1], [1, 1], [0, 0], [0, 1]],
            ],
            dtype=np.uint8,
        )
        expected = np.empty_like(grid)
        expected[..., 0] = np.roll(grid[..., 0], -1, axis=1)
        expected[..., 1] = np.roll(grid[..., 1], 1, axis=1)
        actual = synchronous_ca_step(circuit, grid, boundary="periodic")
        self.assertTrue(np.array_equal(actual, expected))

        rollout = synchronous_ca_rollout(circuit, grid, steps=1, boundary="periodic")
        self.assertTrue(np.array_equal(rollout[0], grid))
        self.assertTrue(np.array_equal(rollout[1], expected))

    def test_ca_rollout_can_use_official_checkerboard_channel_major_layout(self) -> None:
        document = {
            "devices": {
                # channel-major: order = c*9+f
                "right_c0": {"type": "Button", "order": 0 * 9 + 5},
                "left_c1": {"type": "Button", "order": 1 * 9 + 3},
                "output_c1": {"type": "Output", "order": 1},
                "output_c0": {"type": "Output", "order": 0},
            },
            "connectors": [
                {"from": {"id": "left_c1", "port": "out"}, "to": {"id": "output_c1", "port": "in"}},
                {"from": {"id": "right_c0", "port": "out"}, "to": {"id": "output_c0", "port": "in"}},
            ],
            "subcircuits": {},
        }
        circuit = parse_digitaljs_json(document)
        grid = np.asarray(
            [
                [[1, 0], [0, 1], [0, 0], [1, 1]],
                [[0, 1], [1, 1], [0, 0], [0, 1]],
            ],
            dtype=np.uint8,
        )
        expected = np.empty_like(grid)
        expected[..., 0] = np.roll(grid[..., 0], -1, axis=1)
        expected[..., 1] = np.roll(grid[..., 1], 1, axis=1)
        rollout = synchronous_ca_rollout(
            circuit,
            grid,
            steps=1,
            boundary="periodic",
            input_layout="channel_major",
        )
        self.assertTrue(np.array_equal(rollout[1], expected))


if __name__ == "__main__":
    unittest.main(verbosity=2)
