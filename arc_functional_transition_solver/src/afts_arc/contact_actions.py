"""M05f local renderer for the additive M02c bbox-contact relation view."""

from __future__ import annotations

from enum import Enum

from .grid import Grid, as_grid
from .relation import ContactStructuralStatus, parse_bbox_contacts

BBOX_CONTACT_ACTION_SEMANTICS_VERSION = "afts-bbox-contact-render/v0.1"
BBOX_CONTACT_ACTION_STAGE = "M05f_bbox_contact_renderer"


class BBoxContactCode(str, Enum):
    EMPTY_SELECTION = "empty_selection"
    NON_UNIQUE_SELECTION = "non_unique_selection"
    TARGET_COLLISION = "target_collision"
    OCCLUDED_RAY = "occluded_ray"
    INCOMPATIBLE_RELATION_GEOMETRY = "incompatible_relation_geometry"


def paint_bbox_contacts(
    grid: Grid, *, background: int
) -> tuple[Grid | None, BBoxContactCode | None]:
    """Copy marker colors to unique clear rectangle-boundary contacts."""

    normalized = as_grid(grid)
    if type(background) is not int or not 0 <= background <= 9:
        raise ValueError("background must be an ARC color")
    bundle = parse_bbox_contacts(normalized, background=background)
    complete = tuple(
        item
        for item in bundle.hypotheses
        if item.structural_status is ContactStructuralStatus.COMPLETE
    )
    if len(complete) == 1:
        output = [list(row) for row in normalized]
        for relation in complete[0].relations:
            marker_row, marker_column = relation.marker_coordinate
            target_row, target_column = relation.projected_boundary_coordinate
            output[target_row][target_column] = normalized[marker_row][marker_column]
        return as_grid(output), None
    status_map = {
        ContactStructuralStatus.NON_UNIQUE_SELECTION: (
            BBoxContactCode.NON_UNIQUE_SELECTION
        ),
        ContactStructuralStatus.TARGET_COLLISION: BBoxContactCode.TARGET_COLLISION,
        ContactStructuralStatus.OCCLUDED_RAY: BBoxContactCode.OCCLUDED_RAY,
        ContactStructuralStatus.INCOMPATIBLE_RELATION_GEOMETRY: (
            BBoxContactCode.INCOMPATIBLE_RELATION_GEOMETRY
        ),
        ContactStructuralStatus.EMPTY_SELECTION: BBoxContactCode.EMPTY_SELECTION,
    }
    return None, status_map[bundle.structural_status]
