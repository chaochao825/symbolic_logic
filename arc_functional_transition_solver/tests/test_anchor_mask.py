from __future__ import annotations

import pytest

from afts_arc.anchor_mask import (
    ANCHOR_MASK_PROGRAMS_PER_DELTA,
    AnchorRasterizedDeltaNode,
    anchor_mask_domain,
    execute_anchor_rasterized_delta,
    rasterize_anchor_mask,
)
from afts_arc.grid import as_grid
from afts_arc.hybrid.scene_graph import extract_scene_graph


def _scene(grid: list[list[int]]):
    return extract_scene_graph(
        as_grid(grid),
        background=0,
        connectivity=4,
        grouping="monochrome_components",
    )


def test_frozen_mask_language_has_exactly_twelve_rasterizers_per_anchor() -> None:
    domain = anchor_mask_domain(0, 5)

    assert ANCHOR_MASK_PROGRAMS_PER_DELTA == 120
    assert len(domain) == 120
    assert len(set(domain)) == 120


def test_stencil_project_and_bbox_have_distinct_concrete_semantics() -> None:
    source = [[2, 0, 2], [0, 0, 0], [0, 0, 0]]
    parent = as_grid([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    scene = _scene(source)
    east = AnchorRasterizedDeltaNode(2, "stencil", "east", 0, 5)
    columns = AnchorRasterizedDeltaNode(2, "axis_project", "column", 0, 5)
    box = AnchorRasterizedDeltaNode(2, "bbox_fill", "inset0", 0, 5)

    assert execute_anchor_rasterized_delta(east, as_grid(source), parent, scene) == as_grid(
        [[0, 5, 0], [0, 0, 0], [0, 0, 0]]
    )
    assert execute_anchor_rasterized_delta(
        columns, as_grid(source), parent, scene
    ) == as_grid([[5, 0, 5], [5, 0, 5], [5, 0, 5]])
    assert rasterize_anchor_mask(box, as_grid(source), parent, scene) == (
        (0, 0),
        (0, 1),
        (0, 2),
    )


def test_anchor_mask_node_is_strict_content_addressed_and_context_checked() -> None:
    node = AnchorRasterizedDeltaNode(2, "axis_project", "column", 0, 5)
    reconstructed = AnchorRasterizedDeltaNode.from_json_dict(node.to_json_dict())

    assert reconstructed == node
    assert reconstructed.node_id == node.node_id
    with pytest.raises(ValueError, match="missing or unknown"):
        AnchorRasterizedDeltaNode.from_json_dict(
            {**node.to_json_dict(), "unknown": True}
        )
    with pytest.raises(ValueError, match="incompatible"):
        AnchorRasterizedDeltaNode(2, "axis_project", "north", 0, 5)
    with pytest.raises(ValueError, match="canvases must match"):
        execute_anchor_rasterized_delta(
            node,
            as_grid([[2, 0], [0, 0]]),
            as_grid([[0]]),
            _scene([[2, 0], [0, 0]]),
        )


def test_background_color_cannot_become_an_implicit_anchor_object() -> None:
    source = as_grid([[0, 2], [0, 0]])
    parent = as_grid([[0, 0], [0, 0]])
    node = AnchorRasterizedDeltaNode(0, "axis_project", "row", 0, 5)

    assert execute_anchor_rasterized_delta(
        node,
        source,
        parent,
        _scene([[0, 2], [0, 0]]),
    ) == parent
