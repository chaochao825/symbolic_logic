from __future__ import annotations

import copy
import unittest

from afts_arc.grid import as_grid, grid_key
from afts_arc.panel import (
    PanelOverlayCode,
    PanelParseBundle,
    parse_panels,
    overlay_panel_grid,
)
from afts_arc.parse import ColorMode, parse_grid


class PanelParserTests(unittest.TestCase):
    def test_cross_lattice_is_indexed_and_overlay_is_row_major_first_wins(self) -> None:
        grid = as_grid(
            [
                [1, 0, 9, 3, 4],
                [0, 2, 9, 0, 0],
                [9, 9, 9, 9, 9],
                [5, 0, 9, 7, 8],
                [6, 0, 9, 0, 0],
            ]
        )
        bundle = parse_panels(grid)
        self.assertEqual(len(bundle.hypotheses), 1)
        hypothesis = bundle.hypotheses[0]
        self.assertEqual(
            grid_key(grid),
            "dd93c6686ede47acc0c7be84c28300eadfa32340003831817e9ebd003f99cd03",
        )
        self.assertEqual(hypothesis.parse_id, "d75754ac21bc655b9ee4")
        self.assertEqual(
            [panel.panel_id for panel in hypothesis.panels],
            [
                "b4e8c76cdba33ace1514",
                "dd015b7b57e4ad7f4b79",
                "6e0100b2e8257a59bf31",
                "d357e5767b68546a44d7",
            ],
        )
        self.assertEqual(hypothesis.separator_color, 9)
        self.assertEqual(
            [(item.start, item.end) for item in hypothesis.row_separator_bands],
            [(2, 2)],
        )
        self.assertEqual(
            [(item.start, item.end) for item in hypothesis.column_separator_bands],
            [(2, 2)],
        )
        self.assertEqual(
            [
                (
                    panel.row_index,
                    panel.column_index,
                    panel.bbox.top,
                    panel.bbox.left,
                    panel.bbox.bottom,
                    panel.bbox.right,
                )
                for panel in hypothesis.panels
            ],
            [
                (0, 0, 0, 0, 1, 1),
                (0, 1, 0, 3, 1, 4),
                (1, 0, 3, 0, 4, 1),
                (1, 1, 3, 3, 4, 4),
            ],
        )
        output, invalid = overlay_panel_grid(grid, background=0)
        self.assertIsNone(invalid)
        self.assertEqual(output, ((1, 4), (6, 2)))

    def test_single_axis_and_thick_separator_bands_are_preserved(self) -> None:
        grid = as_grid(
            [
                [1, 0, 5, 5, 2, 0, 5, 3],
                [0, 4, 5, 5, 0, 6, 5, 0],
                [7, 0, 5, 5, 8, 0, 5, 9],
            ]
        )
        hypothesis = parse_panels(grid).hypotheses[0]
        self.assertEqual(hypothesis.row_separator_bands, ())
        self.assertEqual(
            [(item.start, item.end) for item in hypothesis.column_separator_bands],
            [(2, 3), (6, 6)],
        )
        self.assertEqual(
            [(panel.bbox.height, panel.bbox.width) for panel in hypothesis.panels],
            [(3, 2), (3, 2), (3, 1)],
        )
        output, invalid = overlay_panel_grid(grid, background=0)
        self.assertIsNone(output)
        self.assertEqual(invalid, PanelOverlayCode.EMPTY_SELECTION)

    def test_ragged_cross_lattice_parses_but_overlay_is_incompatible(self) -> None:
        values = [[0 for _ in range(7)] for _ in range(7)]
        for index in (2, 5):
            values[index] = [8 for _ in range(7)]
            for row in range(7):
                values[row][index] = 8
        grid = as_grid(values)
        hypothesis = parse_panels(grid).hypotheses[0]
        self.assertEqual(
            [(panel.bbox.height, panel.bbox.width) for panel in hypothesis.panels],
            [
                (2, 2),
                (2, 2),
                (2, 1),
                (2, 2),
                (2, 2),
                (2, 1),
                (1, 2),
                (1, 2),
                (1, 1),
            ],
        )
        output, invalid = overlay_panel_grid(grid, background=0)
        self.assertIsNone(output)
        self.assertEqual(invalid, PanelOverlayCode.INCOMPATIBLE_PANEL_SHAPES)

    def test_outer_frame_and_no_separator_do_not_form_panel_hypotheses(self) -> None:
        frame = as_grid([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        plain = as_grid([[1, 2], [3, 4]])
        self.assertEqual(parse_panels(frame).hypotheses, ())
        self.assertEqual(parse_panels(plain).hypotheses, ())

    def test_strict_round_trip_rejects_forged_panel_content_and_structure(self) -> None:
        grid = as_grid(
            [
                [1, 0, 9, 3, 4],
                [0, 2, 9, 0, 0],
                [9, 9, 9, 9, 9],
                [5, 0, 9, 7, 8],
                [6, 0, 9, 0, 0],
            ]
        )
        bundle = parse_panels(grid)
        payload = bundle.to_json_dict()
        self.assertEqual(PanelParseBundle.from_json_dict(payload), bundle)

        forged_id = copy.deepcopy(payload)
        forged_id["hypotheses"][0]["panels"][0]["panel_id"] = "forged"
        with self.assertRaisesRegex(ValueError, "panel_id"):
            PanelParseBundle.from_json_dict(forged_id)

        forged_content = copy.deepcopy(payload)
        forged_content["hypotheses"][0]["panels"][0]["content"][0][0] = 8
        with self.assertRaisesRegex(ValueError, "content_key"):
            PanelParseBundle.from_json_dict(forged_content)

        forged_band = copy.deepcopy(payload)
        forged_band["hypotheses"][0]["row_separator_bands"][0]["start"] = 1
        with self.assertRaises(ValueError):
            PanelParseBundle.from_json_dict(forged_band)

        unknown = copy.deepcopy(payload)
        unknown["unexpected"] = True
        with self.assertRaisesRegex(ValueError, "unknown"):
            PanelParseBundle.from_json_dict(unknown)

    def test_m02a_content_addresses_are_unchanged_by_independent_panel_view(self) -> None:
        hypothesis = parse_grid(
            as_grid([[0, 1], [0, 1]]),
            backgrounds=(0,),
            connectivities=(4,),
            color_modes=(ColorMode.SINGLE_COLOR,),
        ).hypotheses[0]
        self.assertEqual(hypothesis.parse_id, "99556c7f8f90a537f485")
        self.assertEqual(hypothesis.objects[0].object_id, "8c30f36fe2a5be88b579")


if __name__ == "__main__":
    unittest.main()
