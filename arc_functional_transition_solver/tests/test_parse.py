from __future__ import annotations

import unittest
from dataclasses import replace

from afts_arc.grid import as_grid
from afts_arc.parse import ColorMode, GridParseBundle, background_hypotheses, parse_grid


class ParseTests(unittest.TestCase):
    def test_background_hypotheses_are_evidence_ordered_and_deterministic(self) -> None:
        grid = as_grid([[2, 2, 2], [2, 0, 1], [2, 1, 1]])
        self.assertEqual(background_hypotheses(grid), (None, 2, 0, 1))
        self.assertEqual(background_hypotheses(grid, max_backgrounds=2), (None, 2))

    def test_four_and_eight_connectivity_keep_distinct_views(self) -> None:
        grid = as_grid([[1, 0], [0, 1]])
        bundle = parse_grid(
            grid,
            backgrounds=(0,),
            color_modes=(ColorMode.SINGLE_COLOR,),
        )
        by_connectivity = {item.connectivity: item for item in bundle.hypotheses}
        self.assertEqual(len(by_connectivity[4].objects), 2)
        self.assertEqual(len(by_connectivity[8].objects), 1)
        self.assertNotEqual(by_connectivity[4].parse_id, by_connectivity[8].parse_id)

    def test_single_color_and_multicolor_views_disagree_without_collapsing(self) -> None:
        grid = as_grid([[1, 2], [0, 0]])
        bundle = parse_grid(grid, backgrounds=(0,), connectivities=(4,))
        by_mode = {item.color_mode: item for item in bundle.hypotheses}
        self.assertEqual(len(by_mode[ColorMode.SINGLE_COLOR].objects), 2)
        multicolor = by_mode[ColorMode.MULTICOLOR_FOREGROUND].objects
        self.assertEqual(len(multicolor), 1)
        self.assertEqual(multicolor[0].colors, (1, 2))

    def test_holes_and_bbox_containment_are_recorded(self) -> None:
        grid = as_grid(
            [
                [1, 1, 1, 1, 1],
                [1, 0, 0, 0, 1],
                [1, 0, 2, 0, 1],
                [1, 0, 0, 0, 1],
                [1, 1, 1, 1, 1],
            ]
        )
        hypothesis = parse_grid(
            grid,
            backgrounds=(0,),
            connectivities=(4,),
            color_modes=(ColorMode.SINGLE_COLOR,),
        ).hypotheses[0]
        ring = next(item for item in hypothesis.objects if item.colors == (1,))
        center = next(item for item in hypothesis.objects if item.colors == (2,))
        self.assertEqual(ring.hole_count, 1)
        relation = hypothesis.relations[0]
        contains = (
            relation.first_bbox_contains_second
            if relation.first_object_id == ring.object_id
            else relation.second_bbox_contains_first
        )
        self.assertTrue(contains)
        self.assertEqual(center.area, 1)

    def test_parse_records_are_repeatable_and_immutable(self) -> None:
        grid = as_grid([[0, 3, 0], [3, 3, 0]])
        first = parse_grid(grid)
        second = parse_grid(grid)
        self.assertEqual(first, second)
        with self.assertRaises(Exception):
            first.hypotheses[0].background_color = 9  # type: ignore[misc]

    def test_default_parser_enumerates_none_and_every_observed_background(self) -> None:
        grid = as_grid([[0, 1], [2, 0]])
        bundle = parse_grid(grid)
        self.assertEqual(bundle.background_candidates, (None, 0, 1, 2))
        self.assertEqual(len(bundle.hypotheses), 16)
        no_background = next(
            item
            for item in bundle.hypotheses
            if item.background_color is None
            and item.connectivity == 4
            and item.color_mode is ColorMode.MULTICOLOR_FOREGROUND
        )
        self.assertEqual(sum(obj.area for obj in no_background.objects), 4)

    def test_content_addressed_parse_records_reject_forged_derived_fields(self) -> None:
        bundle = parse_grid(
            as_grid([[0, 1], [0, 1]]),
            backgrounds=(0,),
            connectivities=(4,),
            color_modes=(ColorMode.SINGLE_COLOR,),
        )
        hypothesis = bundle.hypotheses[0]
        self.assertEqual(GridParseBundle.from_json_dict(bundle.to_json_dict()), bundle)
        with self.assertRaisesRegex(ValueError, "object_id"):
            replace(hypothesis.objects[0], object_id="forged")
        with self.assertRaisesRegex(ValueError, "parse_id"):
            replace(hypothesis, parse_id="forged")


if __name__ == "__main__":
    unittest.main()
