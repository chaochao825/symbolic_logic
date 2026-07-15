from __future__ import annotations

import unittest

from afts_arc.blind import BlindTask
from afts_arc.grid import as_grid
from afts_arc.shape import (
    OutputShapeProposal,
    ShapeBasis,
    infer_output_shape_proposals,
)
from afts_arc.task import ARCPair


def _blind(
    train: tuple[tuple[list[list[int]], list[list[int]]], ...],
    test_inputs: tuple[list[list[int]], ...] = ([[0, 1], [2, 3]],),
) -> BlindTask:
    return BlindTask.from_observations(
        train=tuple(
            ARCPair(as_grid(source), as_grid(target)) for source, target in train
        ),
        test_inputs=tuple(as_grid(grid) for grid in test_inputs),
    )


class OutputShapeProposalTests(unittest.TestCase):
    def test_consistent_integer_axis_factors_and_query_shapes_are_content_addressed(
        self,
    ) -> None:
        task = _blind(
            (
                (
                    [[1, 2, 3], [4, 5, 6]],
                    [[0] * 9 for _ in range(4)],
                ),
                (
                    [[1, 2]],
                    [[0] * 6 for _ in range(2)],
                ),
            ),
            test_inputs=([[1, 2, 3, 4], [5, 6, 7, 8], [9, 0, 1, 2]],),
        )
        proposals = infer_output_shape_proposals(task)
        self.assertEqual(len(proposals), 1)
        proposal = proposals[0]
        self.assertEqual(proposal.basis, ShapeBasis.INPUT)
        self.assertEqual((proposal.row_factor, proposal.column_factor), (2, 3))
        self.assertEqual(proposal.query_shapes, ((6, 12),))
        self.assertEqual(
            OutputShapeProposal.from_json_dict(proposal.to_json_dict()), proposal
        )
        forged = proposal.to_json_dict()
        forged["row_factor"] = 3
        with self.assertRaisesRegex(ValueError, "proposal_id"):
            OutputShapeProposal.from_json_dict(forged)

    def test_inconsistent_noninteger_and_shrinking_shapes_are_not_proposed(self) -> None:
        inconsistent = _blind(
            (
                ([[1]], [[1], [1]]),
                ([[2]], [[2], [2], [2]]),
            )
        )
        noninteger = _blind((([[1, 2], [3, 4]], [[0, 0], [0, 0], [0, 0]]),))
        shrinking = _blind((([[1, 2], [3, 4]], [[1]]),))
        self.assertEqual(infer_output_shape_proposals(inconsistent), ())
        self.assertEqual(infer_output_shape_proposals(noninteger), ())
        self.assertEqual(infer_output_shape_proposals(shrinking), ())

    def test_query_projection_over_arc_limit_is_rejected(self) -> None:
        task = _blind(
            (([[1]], [[1, 1], [1, 1]]),),
            test_inputs=([[1] for _ in range(16)],),
        )
        self.assertEqual(infer_output_shape_proposals(task), ())


if __name__ == "__main__":
    unittest.main()
