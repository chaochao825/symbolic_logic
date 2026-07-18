from __future__ import annotations

import copy
import unittest

from afts_arc.grid import as_grid
from afts_arc.relation import (
    BBoxContactParseBundle,
    ContactAxis,
    ContactDirection,
    ContactStructuralStatus,
    parse_bbox_contacts,
)


class RelationTests(unittest.TestCase):
    def test_valid_bundle_records_zero_based_direction_gap_and_round_trips(self) -> None:
        grid = as_grid(
            [
                [0, 0, 0, 0, 4, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 8, 8, 8, 0, 0],
                [2, 0, 0, 8, 8, 8, 0, 0],
                [0, 0, 0, 8, 8, 8, 0, 6],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 7, 0, 0, 0],
            ]
        )
        bundle = parse_bbox_contacts(grid, background=0)
        self.assertEqual(bundle.structural_status, ContactStructuralStatus.COMPLETE)
        complete = [
            item
            for item in bundle.hypotheses
            if item.structural_status is ContactStructuralStatus.COMPLETE
        ]
        self.assertEqual(len(complete), 1)
        by_marker = {
            relation.marker_coordinate: relation
            for relation in complete[0].relations
        }
        self.assertEqual(
            (
                by_marker[(3, 0)].axis,
                by_marker[(3, 0)].direction,
                by_marker[(3, 0)].projected_boundary_coordinate,
                by_marker[(3, 0)].gap,
            ),
            (ContactAxis.ROW, ContactDirection.RIGHT, (3, 3), 2),
        )
        self.assertEqual(
            (
                by_marker[(0, 4)].axis,
                by_marker[(0, 4)].direction,
                by_marker[(0, 4)].projected_boundary_coordinate,
                by_marker[(0, 4)].gap,
            ),
            (ContactAxis.COLUMN, ContactDirection.DOWN, (2, 4), 1),
        )
        self.assertEqual(by_marker[(4, 7)].direction, ContactDirection.LEFT)
        self.assertEqual(by_marker[(6, 4)].direction, ContactDirection.UP)
        self.assertTrue(all(item.ray_clear for item in by_marker.values()))
        payload = bundle.to_json_dict()
        self.assertEqual(BBoxContactParseBundle.from_json_dict(payload), bundle)

        forged = copy.deepcopy(payload)
        forged["bundle_id"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "bundle_id"):
            BBoxContactParseBundle.from_json_dict(forged)
        forged = copy.deepcopy(payload)
        forged["hypotheses"][0]["relations"][0]["gap"] += 1
        with self.assertRaisesRegex(ValueError, "gap|ray_clear"):
            BBoxContactParseBundle.from_json_dict(forged)
        forged = copy.deepcopy(payload)
        forged["hypotheses"][0]["relations"][0]["marker_coordinate"][0] += 1
        with self.assertRaisesRegex(ValueError, "geometry|direction|gap|relation_id"):
            BBoxContactParseBundle.from_json_dict(forged)
        forged = copy.deepcopy(payload)
        forged["hypotheses"][0]["relations"][0]["source_m02a_parse_id"] = "forged"
        with self.assertRaisesRegex(ValueError, "relation_id|source parse"):
            BBoxContactParseBundle.from_json_dict(forged)
        forged = copy.deepcopy(payload)
        forged["semantics_version"] = "future"
        with self.assertRaisesRegex(ValueError, "semantics"):
            BBoxContactParseBundle.from_json_dict(forged)

    def test_collision_occlusion_incompatible_and_empty_statuses(self) -> None:
        collision = as_grid(
            [
                [0, 0, 2, 0, 0],
                [0, 0, 0, 0, 0],
                [3, 0, 8, 8, 0],
                [0, 0, 8, 8, 0],
                [0, 0, 0, 0, 0],
            ]
        )
        self.assertEqual(
            parse_bbox_contacts(collision, background=0).structural_status,
            ContactStructuralStatus.TARGET_COLLISION,
        )

        occluded = as_grid(
            [
                [0, 0, 2, 0, 0],
                [0, 0, 3, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 8, 8, 8, 0],
                [0, 8, 8, 8, 0],
            ]
        )
        bundle = parse_bbox_contacts(occluded, background=0)
        self.assertEqual(bundle.structural_status, ContactStructuralStatus.OCCLUDED_RAY)
        relations = bundle.hypotheses[0].relations
        self.assertEqual(sum(item.ray_clear for item in relations), 1)
        self.assertEqual(sum(not item.ray_clear for item in relations), 1)

        incompatible_grids = (
            as_grid(
                [
                    [0, 0, 2, 0, 0],
                    [0, 0, 0, 8, 8],
                    [0, 0, 0, 8, 8],
                ]
            ),
            as_grid([[8, 8], [8, 0]]),
            as_grid(
                [
                    [0, 2, 2, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 8, 8, 0, 0],
                    [0, 8, 8, 0, 0],
                ]
            ),
            as_grid(
                [
                    [0, 0, 8, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 8, 8, 0, 0],
                    [0, 8, 8, 0, 0],
                ]
            ),
        )
        for grid in incompatible_grids:
            with self.subTest(grid=grid):
                self.assertEqual(
                    parse_bbox_contacts(grid, background=0).structural_status,
                    ContactStructuralStatus.INCOMPATIBLE_RELATION_GEOMETRY,
                )
        self.assertEqual(
            parse_bbox_contacts(as_grid([[0, 1, 0]]), background=0).structural_status,
            ContactStructuralStatus.EMPTY_SELECTION,
        )

    def test_two_complete_competing_anchors_are_non_unique(self) -> None:
        grid = as_grid(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 8, 8, 0, 3, 0, 7, 7, 0],
                [0, 8, 8, 0, 0, 0, 7, 7, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )
        bundle = parse_bbox_contacts(grid, background=0)
        self.assertEqual(
            sum(
                item.structural_status is ContactStructuralStatus.COMPLETE
                for item in bundle.hypotheses
            ),
            2,
        )
        self.assertEqual(
            bundle.structural_status,
            ContactStructuralStatus.NON_UNIQUE_SELECTION,
        )

    def test_background_argument_is_strict(self) -> None:
        for invalid in (None, False, -1, 10):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                ValueError, "ARC color"
            ):
                parse_bbox_contacts(as_grid([[0]]), background=invalid)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
