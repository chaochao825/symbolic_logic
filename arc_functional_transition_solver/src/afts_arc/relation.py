"""Additive M02c axis-ray relation views over the frozen M02a objects."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum

from .grid import Grid, as_grid, grid_key
from .parse import BoundingBox, ColorMode, ObjectView, parse_grid

Coordinate = tuple[int, int]
RELATION_PARSER_SEMANTICS_VERSION = "afts-axis-aligned-bbox-contact/v0.1"
RELATION_PARSER_STAGE = "M02c_axis_aligned_singleton_bbox_contact"


class ContactAxis(str, Enum):
    ROW = "row"
    COLUMN = "column"


class ContactDirection(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    UP = "up"
    DOWN = "down"


class ContactStructuralStatus(str, Enum):
    COMPLETE = "complete"
    NON_UNIQUE_SELECTION = "non_unique_selection"
    TARGET_COLLISION = "target_collision"
    OCCLUDED_RAY = "occluded_ray"
    INCOMPATIBLE_RELATION_GEOMETRY = "incompatible_relation_geometry"
    EMPTY_SELECTION = "empty_selection"


def _canonical_hash(label: str, payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(label.encode("ascii") + b"\0" + encoded).hexdigest()


def _validate_coordinate(value: object, *, label: str) -> Coordinate:
    if (
        not isinstance(value, (tuple, list))
        or len(value) != 2
        or any(type(item) is not int or item < 0 for item in value)
    ):
        raise ValueError(f"{label} must be a non-negative coordinate pair")
    return value[0], value[1]


@dataclass(frozen=True, slots=True)
class BBoxContactRelation:
    relation_id: str
    marker_object_id: str
    anchor_object_id: str
    source_m02a_parse_id: str
    marker_coordinate: Coordinate
    projected_boundary_coordinate: Coordinate
    axis: ContactAxis
    direction: ContactDirection
    gap: int
    intervening_background_count: int
    ray_clear: bool

    def __post_init__(self) -> None:
        if (
            not isinstance(self.marker_object_id, str)
            or not self.marker_object_id
            or not isinstance(self.anchor_object_id, str)
            or not self.anchor_object_id
            or self.marker_object_id == self.anchor_object_id
        ):
            raise ValueError("contact relation endpoints must be distinct object IDs")
        if (
            not isinstance(self.source_m02a_parse_id, str)
            or not self.source_m02a_parse_id
        ):
            raise ValueError("contact relation source parse ID must not be empty")
        marker = _validate_coordinate(
            self.marker_coordinate, label="marker_coordinate"
        )
        projected = _validate_coordinate(
            self.projected_boundary_coordinate,
            label="projected_boundary_coordinate",
        )
        object.__setattr__(self, "marker_coordinate", marker)
        object.__setattr__(self, "projected_boundary_coordinate", projected)
        if not isinstance(self.axis, ContactAxis):
            raise TypeError("contact axis must be a ContactAxis")
        if not isinstance(self.direction, ContactDirection):
            raise TypeError("contact direction must be a ContactDirection")
        if type(self.gap) is not int or self.gap < 0:
            raise ValueError("contact gap must be non-negative")
        if (
            type(self.intervening_background_count) is not int
            or not 0 <= self.intervening_background_count <= self.gap
        ):
            raise ValueError("intervening background count must lie in [0, gap]")
        if type(self.ray_clear) is not bool or self.ray_clear != (
            self.intervening_background_count == self.gap
        ):
            raise ValueError("ray_clear must close to the open-ray background count")
        row_delta = projected[0] - marker[0]
        column_delta = projected[1] - marker[1]
        if self.axis is ContactAxis.ROW:
            if row_delta != 0 or column_delta == 0 or self.direction not in {
                ContactDirection.LEFT,
                ContactDirection.RIGHT,
            }:
                raise ValueError("row-axis contact geometry is inconsistent")
            expected_direction = (
                ContactDirection.RIGHT
                if column_delta > 0
                else ContactDirection.LEFT
            )
        else:
            if column_delta != 0 or row_delta == 0 or self.direction not in {
                ContactDirection.UP,
                ContactDirection.DOWN,
            }:
                raise ValueError("column-axis contact geometry is inconsistent")
            expected_direction = (
                ContactDirection.DOWN if row_delta > 0 else ContactDirection.UP
            )
        if self.direction is not expected_direction:
            raise ValueError("contact direction is not marker-to-anchor travel")
        if self.gap != abs(row_delta) + abs(column_delta) - 1:
            raise ValueError("contact gap must equal Manhattan distance minus one")
        expected_id = _canonical_hash(
            "afts-bbox-contact-relation/v1",
            {
                "semantics": RELATION_PARSER_SEMANTICS_VERSION,
                "marker_object_id": self.marker_object_id,
                "anchor_object_id": self.anchor_object_id,
                "source_m02a_parse_id": self.source_m02a_parse_id,
                "marker_coordinate": list(marker),
                "projected_boundary_coordinate": list(projected),
                "axis": self.axis.value,
                "direction": self.direction.value,
                "gap": self.gap,
                "intervening_background_count": self.intervening_background_count,
                "ray_clear": self.ray_clear,
            },
        )[:20]
        if self.relation_id != expected_id:
            raise ValueError("relation_id does not match canonical relation content")

    @classmethod
    def create(
        cls,
        *,
        marker_object_id: str,
        anchor_object_id: str,
        source_m02a_parse_id: str,
        marker_coordinate: Coordinate,
        projected_boundary_coordinate: Coordinate,
        axis: ContactAxis,
        direction: ContactDirection,
        gap: int,
        intervening_background_count: int,
    ) -> "BBoxContactRelation":
        payload = {
            "semantics": RELATION_PARSER_SEMANTICS_VERSION,
            "marker_object_id": marker_object_id,
            "anchor_object_id": anchor_object_id,
            "source_m02a_parse_id": source_m02a_parse_id,
            "marker_coordinate": list(marker_coordinate),
            "projected_boundary_coordinate": list(projected_boundary_coordinate),
            "axis": axis.value,
            "direction": direction.value,
            "gap": gap,
            "intervening_background_count": intervening_background_count,
            "ray_clear": intervening_background_count == gap,
        }
        return cls(
            relation_id=_canonical_hash(
                "afts-bbox-contact-relation/v1", payload
            )[:20],
            marker_object_id=marker_object_id,
            anchor_object_id=anchor_object_id,
            source_m02a_parse_id=source_m02a_parse_id,
            marker_coordinate=marker_coordinate,
            projected_boundary_coordinate=projected_boundary_coordinate,
            axis=axis,
            direction=direction,
            gap=gap,
            intervening_background_count=intervening_background_count,
            ray_clear=intervening_background_count == gap,
        )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "relation_id": self.relation_id,
            "marker_object_id": self.marker_object_id,
            "anchor_object_id": self.anchor_object_id,
            "source_m02a_parse_id": self.source_m02a_parse_id,
            "marker_coordinate": list(self.marker_coordinate),
            "projected_boundary_coordinate": list(
                self.projected_boundary_coordinate
            ),
            "axis": self.axis.value,
            "direction": self.direction.value,
            "gap": self.gap,
            "intervening_background_count": self.intervening_background_count,
            "ray_clear": self.ray_clear,
        }

    @classmethod
    def from_json_dict(cls, payload: object) -> "BBoxContactRelation":
        expected = {
            "relation_id",
            "marker_object_id",
            "anchor_object_id",
            "source_m02a_parse_id",
            "marker_coordinate",
            "projected_boundary_coordinate",
            "axis",
            "direction",
            "gap",
            "intervening_background_count",
            "ray_clear",
        }
        if not isinstance(payload, dict) or set(payload) != expected:
            raise ValueError("bbox contact relation has missing or unknown fields")
        return cls(
            relation_id=payload["relation_id"],
            marker_object_id=payload["marker_object_id"],
            anchor_object_id=payload["anchor_object_id"],
            source_m02a_parse_id=payload["source_m02a_parse_id"],
            marker_coordinate=_validate_coordinate(
                payload["marker_coordinate"], label="marker_coordinate"
            ),
            projected_boundary_coordinate=_validate_coordinate(
                payload["projected_boundary_coordinate"],
                label="projected_boundary_coordinate",
            ),
            axis=ContactAxis(payload["axis"]),
            direction=ContactDirection(payload["direction"]),
            gap=payload["gap"],
            intervening_background_count=payload[
                "intervening_background_count"
            ],
            ray_clear=payload["ray_clear"],
        )


@dataclass(frozen=True, slots=True)
class ContactCollisionGroup:
    projected_boundary_coordinate: Coordinate
    marker_object_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        coordinate = _validate_coordinate(
            self.projected_boundary_coordinate,
            label="collision projected_boundary_coordinate",
        )
        object.__setattr__(self, "projected_boundary_coordinate", coordinate)
        ids = tuple(self.marker_object_ids)
        if (
            len(ids) < 2
            or ids != tuple(sorted(set(ids)))
            or any(not isinstance(item, str) or not item for item in ids)
        ):
            raise ValueError("collision marker IDs must be sorted unique IDs")
        object.__setattr__(self, "marker_object_ids", ids)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "projected_boundary_coordinate": list(
                self.projected_boundary_coordinate
            ),
            "marker_object_ids": list(self.marker_object_ids),
        }

    @classmethod
    def from_json_dict(cls, payload: object) -> "ContactCollisionGroup":
        if not isinstance(payload, dict) or set(payload) != {
            "projected_boundary_coordinate",
            "marker_object_ids",
        }:
            raise ValueError("contact collision group has invalid fields")
        if not isinstance(payload["marker_object_ids"], list):
            raise TypeError("collision marker_object_ids must be a list")
        return cls(
            projected_boundary_coordinate=_validate_coordinate(
                payload["projected_boundary_coordinate"],
                label="collision projected_boundary_coordinate",
            ),
            marker_object_ids=tuple(payload["marker_object_ids"]),
        )


def _expected_hypothesis_status(
    *,
    anchor_fills_bbox: bool,
    relations: tuple[BBoxContactRelation, ...],
    unmatched_object_ids: tuple[str, ...],
    non_singleton_object_ids: tuple[str, ...],
    same_color_object_ids: tuple[str, ...],
    collision_groups: tuple[ContactCollisionGroup, ...],
) -> ContactStructuralStatus:
    same_color = set(same_color_object_ids)
    if anchor_fills_bbox and collision_groups:
        return ContactStructuralStatus.TARGET_COLLISION
    if anchor_fills_bbox and any(
        not item.ray_clear and item.marker_object_id not in same_color
        for item in relations
    ):
        return ContactStructuralStatus.OCCLUDED_RAY
    if (
        not anchor_fills_bbox
        or unmatched_object_ids
        or non_singleton_object_ids
        or same_color_object_ids
    ):
        return ContactStructuralStatus.INCOMPATIBLE_RELATION_GEOMETRY
    if not relations:
        return ContactStructuralStatus.EMPTY_SELECTION
    return ContactStructuralStatus.COMPLETE


@dataclass(frozen=True, slots=True)
class BBoxContactHypothesis:
    relation_parse_id: str
    grid_key: str
    source_m02a_parse_id: str
    background: int
    anchor_object_id: str
    anchor_color: int
    anchor_bbox: BoundingBox
    anchor_fills_bbox: bool
    relations: tuple[BBoxContactRelation, ...]
    unmatched_object_ids: tuple[str, ...]
    non_singleton_object_ids: tuple[str, ...]
    same_color_object_ids: tuple[str, ...]
    destination_collision_groups: tuple[ContactCollisionGroup, ...]
    structural_status: ContactStructuralStatus

    def __post_init__(self) -> None:
        if not isinstance(self.grid_key, str) or not self.grid_key:
            raise ValueError("contact hypothesis grid_key must not be empty")
        if not isinstance(self.source_m02a_parse_id, str) or not self.source_m02a_parse_id:
            raise ValueError("contact hypothesis source parse ID must not be empty")
        if type(self.background) is not int or not 0 <= self.background <= 9:
            raise ValueError("contact hypothesis background must be an ARC color")
        if not isinstance(self.anchor_object_id, str) or not self.anchor_object_id:
            raise ValueError("contact hypothesis anchor ID must not be empty")
        if type(self.anchor_color) is not int or not 0 <= self.anchor_color <= 9:
            raise ValueError("contact hypothesis anchor color must be an ARC color")
        if not isinstance(self.anchor_bbox, BoundingBox):
            raise TypeError("contact hypothesis anchor_bbox must be a BoundingBox")
        if type(self.anchor_fills_bbox) is not bool:
            raise TypeError("anchor_fills_bbox must be boolean")
        relations = tuple(self.relations)
        if any(not isinstance(item, BBoxContactRelation) for item in relations):
            raise TypeError("contact relations must contain BBoxContactRelation values")
        if relations != tuple(sorted(relations, key=lambda item: item.marker_object_id)):
            raise ValueError("contact relations must be ordered by marker object ID")
        if len({item.marker_object_id for item in relations}) != len(relations):
            raise ValueError("a marker may have at most one contact relation")
        if any(item.anchor_object_id != self.anchor_object_id for item in relations):
            raise ValueError("contact relations must bind the hypothesis anchor")
        if any(
            item.source_m02a_parse_id != self.source_m02a_parse_id
            for item in relations
        ):
            raise ValueError("contact relations must bind the hypothesis source parse")
        object.__setattr__(self, "relations", relations)
        diagnostic_names = (
            "unmatched_object_ids",
            "non_singleton_object_ids",
            "same_color_object_ids",
        )
        for name in diagnostic_names:
            values = tuple(getattr(self, name))
            if (
                values != tuple(sorted(set(values)))
                or any(not isinstance(item, str) or not item for item in values)
                or self.anchor_object_id in values
            ):
                raise ValueError(f"{name} must contain sorted unique non-anchor IDs")
            object.__setattr__(self, name, values)
        if set(self.unmatched_object_ids) & set(self.non_singleton_object_ids):
            raise ValueError("unmatched and non-singleton diagnostics must be disjoint")
        collisions = tuple(self.destination_collision_groups)
        if any(not isinstance(item, ContactCollisionGroup) for item in collisions):
            raise TypeError("collision groups must contain ContactCollisionGroup values")
        if collisions != tuple(
            sorted(collisions, key=lambda item: item.projected_boundary_coordinate)
        ):
            raise ValueError("collision groups must be canonically ordered")
        same_color = set(self.same_color_object_ids)
        grouped: dict[Coordinate, list[str]] = {}
        for relation in relations:
            if relation.ray_clear and relation.marker_object_id not in same_color:
                grouped.setdefault(
                    relation.projected_boundary_coordinate, []
                ).append(relation.marker_object_id)
        expected_collisions = tuple(
            ContactCollisionGroup(coordinate, tuple(sorted(ids)))
            for coordinate, ids in sorted(grouped.items())
            if len(ids) >= 2
        )
        if collisions != expected_collisions:
            raise ValueError("collision groups do not match clear admissible relations")
        object.__setattr__(self, "destination_collision_groups", collisions)
        if not isinstance(self.structural_status, ContactStructuralStatus):
            raise TypeError("structural_status must be a ContactStructuralStatus")
        expected_status = _expected_hypothesis_status(
            anchor_fills_bbox=self.anchor_fills_bbox,
            relations=relations,
            unmatched_object_ids=self.unmatched_object_ids,
            non_singleton_object_ids=self.non_singleton_object_ids,
            same_color_object_ids=self.same_color_object_ids,
            collision_groups=collisions,
        )
        if self.structural_status is not expected_status:
            raise ValueError("contact hypothesis status does not match its diagnostics")
        expected_id = _canonical_hash(
            "afts-bbox-contact-hypothesis/v1",
            self._content_payload(),
        )[:20]
        if self.relation_parse_id != expected_id:
            raise ValueError("relation_parse_id does not match hypothesis content")

    def _content_payload(self) -> dict[str, object]:
        return {
            "semantics": RELATION_PARSER_SEMANTICS_VERSION,
            "grid_key": self.grid_key,
            "source_m02a_parse_id": self.source_m02a_parse_id,
            "background": self.background,
            "anchor_object_id": self.anchor_object_id,
            "anchor_color": self.anchor_color,
            "anchor_bbox": self.anchor_bbox.to_json_dict(),
            "anchor_fills_bbox": self.anchor_fills_bbox,
            "relations": [item.to_json_dict() for item in self.relations],
            "unmatched_object_ids": list(self.unmatched_object_ids),
            "non_singleton_object_ids": list(self.non_singleton_object_ids),
            "same_color_object_ids": list(self.same_color_object_ids),
            "destination_collision_groups": [
                item.to_json_dict() for item in self.destination_collision_groups
            ],
            "structural_status": self.structural_status.value,
        }

    def to_json_dict(self) -> dict[str, object]:
        return {"relation_parse_id": self.relation_parse_id, **self._content_payload()}

    @classmethod
    def from_json_dict(cls, payload: object) -> "BBoxContactHypothesis":
        expected = {
            "relation_parse_id",
            "semantics",
            "grid_key",
            "source_m02a_parse_id",
            "background",
            "anchor_object_id",
            "anchor_color",
            "anchor_bbox",
            "anchor_fills_bbox",
            "relations",
            "unmatched_object_ids",
            "non_singleton_object_ids",
            "same_color_object_ids",
            "destination_collision_groups",
            "structural_status",
        }
        if not isinstance(payload, dict) or set(payload) != expected:
            raise ValueError("bbox contact hypothesis has missing or unknown fields")
        if payload["semantics"] != RELATION_PARSER_SEMANTICS_VERSION:
            raise ValueError("unsupported bbox contact hypothesis semantics")
        for name in (
            "relations",
            "unmatched_object_ids",
            "non_singleton_object_ids",
            "same_color_object_ids",
            "destination_collision_groups",
        ):
            if not isinstance(payload[name], list):
                raise TypeError(f"bbox contact hypothesis {name} must be a list")
        return cls(
            relation_parse_id=payload["relation_parse_id"],
            grid_key=payload["grid_key"],
            source_m02a_parse_id=payload["source_m02a_parse_id"],
            background=payload["background"],
            anchor_object_id=payload["anchor_object_id"],
            anchor_color=payload["anchor_color"],
            anchor_bbox=BoundingBox.from_json_dict(payload["anchor_bbox"]),
            anchor_fills_bbox=payload["anchor_fills_bbox"],
            relations=tuple(
                BBoxContactRelation.from_json_dict(item)
                for item in payload["relations"]
            ),
            unmatched_object_ids=tuple(payload["unmatched_object_ids"]),
            non_singleton_object_ids=tuple(payload["non_singleton_object_ids"]),
            same_color_object_ids=tuple(payload["same_color_object_ids"]),
            destination_collision_groups=tuple(
                ContactCollisionGroup.from_json_dict(item)
                for item in payload["destination_collision_groups"]
            ),
            structural_status=ContactStructuralStatus(payload["structural_status"]),
        )


def _bundle_status(
    hypotheses: tuple[BBoxContactHypothesis, ...],
) -> ContactStructuralStatus:
    complete_count = sum(
        item.structural_status is ContactStructuralStatus.COMPLETE
        for item in hypotheses
    )
    if complete_count == 1:
        return ContactStructuralStatus.COMPLETE
    if complete_count > 1:
        return ContactStructuralStatus.NON_UNIQUE_SELECTION
    for status in (
        ContactStructuralStatus.TARGET_COLLISION,
        ContactStructuralStatus.OCCLUDED_RAY,
        ContactStructuralStatus.INCOMPATIBLE_RELATION_GEOMETRY,
    ):
        if any(item.structural_status is status for item in hypotheses):
            return status
    return ContactStructuralStatus.EMPTY_SELECTION


@dataclass(frozen=True, slots=True)
class BBoxContactParseBundle:
    bundle_id: str
    semantics_version: str
    stage: str
    grid_key: str
    background: int
    source_m02a_parse_id: str
    hypotheses: tuple[BBoxContactHypothesis, ...]
    structural_status: ContactStructuralStatus

    def __post_init__(self) -> None:
        if self.semantics_version != RELATION_PARSER_SEMANTICS_VERSION:
            raise ValueError("unsupported bbox contact parser semantics")
        if self.stage != RELATION_PARSER_STAGE:
            raise ValueError("unsupported bbox contact parser stage")
        if not isinstance(self.grid_key, str) or not self.grid_key:
            raise ValueError("bbox contact bundle grid_key must not be empty")
        if type(self.background) is not int or not 0 <= self.background <= 9:
            raise ValueError("bbox contact bundle background must be an ARC color")
        if not isinstance(self.source_m02a_parse_id, str) or not self.source_m02a_parse_id:
            raise ValueError("bbox contact bundle source parse ID must not be empty")
        hypotheses = tuple(self.hypotheses)
        if any(not isinstance(item, BBoxContactHypothesis) for item in hypotheses):
            raise TypeError("bbox contact bundle hypotheses have a wrong type")
        if hypotheses != tuple(sorted(hypotheses, key=lambda item: item.relation_parse_id)):
            raise ValueError("bbox contact hypotheses must be canonically ordered")
        if len({item.relation_parse_id for item in hypotheses}) != len(hypotheses):
            raise ValueError("bbox contact hypotheses must have unique IDs")
        if any(
            item.grid_key != self.grid_key
            or item.background != self.background
            or item.source_m02a_parse_id != self.source_m02a_parse_id
            for item in hypotheses
        ):
            raise ValueError("bbox contact hypotheses do not bind the bundle")
        object.__setattr__(self, "hypotheses", hypotheses)
        if not isinstance(self.structural_status, ContactStructuralStatus):
            raise TypeError("bbox contact bundle status has a wrong type")
        if self.structural_status is not _bundle_status(hypotheses):
            raise ValueError("bbox contact bundle status does not match hypotheses")
        expected_id = _canonical_hash(
            "afts-bbox-contact-bundle/v1", self._content_payload()
        )
        if self.bundle_id != expected_id:
            raise ValueError("bbox contact bundle_id does not match its content")

    def _content_payload(self) -> dict[str, object]:
        return {
            "semantics_version": self.semantics_version,
            "stage": self.stage,
            "grid_key": self.grid_key,
            "background": self.background,
            "source_m02a_parse_id": self.source_m02a_parse_id,
            "hypotheses": [item.to_json_dict() for item in self.hypotheses],
            "structural_status": self.structural_status.value,
        }

    def to_json_dict(self) -> dict[str, object]:
        return {"bundle_id": self.bundle_id, **self._content_payload()}

    @classmethod
    def from_json_dict(cls, payload: object) -> "BBoxContactParseBundle":
        expected = {
            "bundle_id",
            "semantics_version",
            "stage",
            "grid_key",
            "background",
            "source_m02a_parse_id",
            "hypotheses",
            "structural_status",
        }
        if not isinstance(payload, dict) or set(payload) != expected:
            raise ValueError("bbox contact bundle has missing or unknown fields")
        if not isinstance(payload["hypotheses"], list):
            raise TypeError("bbox contact bundle hypotheses must be a list")
        return cls(
            bundle_id=payload["bundle_id"],
            semantics_version=payload["semantics_version"],
            stage=payload["stage"],
            grid_key=payload["grid_key"],
            background=payload["background"],
            source_m02a_parse_id=payload["source_m02a_parse_id"],
            hypotheses=tuple(
                BBoxContactHypothesis.from_json_dict(item)
                for item in payload["hypotheses"]
            ),
            structural_status=ContactStructuralStatus(payload["structural_status"]),
        )


def _projection(
    marker: Coordinate, bbox: BoundingBox
) -> tuple[ContactAxis, ContactDirection, Coordinate, tuple[Coordinate, ...]] | None:
    row, column = marker
    if bbox.top <= row <= bbox.bottom and column < bbox.left:
        projected = (row, bbox.left)
        open_cells = tuple((row, item) for item in range(column + 1, bbox.left))
        return ContactAxis.ROW, ContactDirection.RIGHT, projected, open_cells
    if bbox.top <= row <= bbox.bottom and column > bbox.right:
        projected = (row, bbox.right)
        open_cells = tuple(
            (row, item) for item in range(bbox.right + 1, column)
        )
        return ContactAxis.ROW, ContactDirection.LEFT, projected, open_cells
    if bbox.left <= column <= bbox.right and row < bbox.top:
        projected = (bbox.top, column)
        open_cells = tuple((item, column) for item in range(row + 1, bbox.top))
        return ContactAxis.COLUMN, ContactDirection.DOWN, projected, open_cells
    if bbox.left <= column <= bbox.right and row > bbox.bottom:
        projected = (bbox.bottom, column)
        open_cells = tuple(
            (item, column) for item in range(bbox.bottom + 1, row)
        )
        return ContactAxis.COLUMN, ContactDirection.UP, projected, open_cells
    return None


def _is_filled_anchor(item: ObjectView) -> bool:
    return (
        item.bbox.height >= 2
        and item.bbox.width >= 2
        and item.area == item.bbox.height * item.bbox.width
    )


def _hypothesis(
    *,
    grid: Grid,
    grid_fingerprint: str,
    source_m02a_parse_id: str,
    background: int,
    anchor: ObjectView,
    objects: tuple[ObjectView, ...],
    filled_anchor_ids: frozenset[str],
) -> BBoxContactHypothesis:
    anchor_color = anchor.colors[0]
    anchor_fills_bbox = _is_filled_anchor(anchor)
    relations: list[BBoxContactRelation] = []
    unmatched: list[str] = []
    non_singleton: list[str] = []
    same_color: list[str] = []
    for item in objects:
        if item.object_id == anchor.object_id or (
            item.object_id in filled_anchor_ids
            and item.object_id != anchor.object_id
        ):
            continue
        if item.area != 1:
            non_singleton.append(item.object_id)
            continue
        marker = item.pixels[0]
        same = item.colors[0] == anchor_color
        if same:
            same_color.append(item.object_id)
        projection = _projection(marker, anchor.bbox)
        if projection is None:
            if not same:
                unmatched.append(item.object_id)
            continue
        axis, direction, projected, open_cells = projection
        background_count = sum(
            grid[row][column] == background for row, column in open_cells
        )
        relations.append(
            BBoxContactRelation.create(
                marker_object_id=item.object_id,
                anchor_object_id=anchor.object_id,
                source_m02a_parse_id=source_m02a_parse_id,
                marker_coordinate=marker,
                projected_boundary_coordinate=projected,
                axis=axis,
                direction=direction,
                gap=len(open_cells),
                intervening_background_count=background_count,
            )
        )
    relations_tuple = tuple(sorted(relations, key=lambda item: item.marker_object_id))
    same_color_tuple = tuple(sorted(same_color))
    same_color_set = set(same_color_tuple)
    grouped: dict[Coordinate, list[str]] = {}
    for relation in relations_tuple:
        if relation.ray_clear and relation.marker_object_id not in same_color_set:
            grouped.setdefault(
                relation.projected_boundary_coordinate, []
            ).append(relation.marker_object_id)
    collisions = tuple(
        ContactCollisionGroup(coordinate, tuple(sorted(ids)))
        for coordinate, ids in sorted(grouped.items())
        if len(ids) >= 2
    )
    unmatched_tuple = tuple(sorted(unmatched))
    non_singleton_tuple = tuple(sorted(non_singleton))
    status = _expected_hypothesis_status(
        anchor_fills_bbox=anchor_fills_bbox,
        relations=relations_tuple,
        unmatched_object_ids=unmatched_tuple,
        non_singleton_object_ids=non_singleton_tuple,
        same_color_object_ids=same_color_tuple,
        collision_groups=collisions,
    )
    content = {
        "semantics": RELATION_PARSER_SEMANTICS_VERSION,
        "grid_key": grid_fingerprint,
        "source_m02a_parse_id": source_m02a_parse_id,
        "background": background,
        "anchor_object_id": anchor.object_id,
        "anchor_color": anchor_color,
        "anchor_bbox": anchor.bbox.to_json_dict(),
        "anchor_fills_bbox": anchor_fills_bbox,
        "relations": [item.to_json_dict() for item in relations_tuple],
        "unmatched_object_ids": list(unmatched_tuple),
        "non_singleton_object_ids": list(non_singleton_tuple),
        "same_color_object_ids": list(same_color_tuple),
        "destination_collision_groups": [item.to_json_dict() for item in collisions],
        "structural_status": status.value,
    }
    return BBoxContactHypothesis(
        relation_parse_id=_canonical_hash(
            "afts-bbox-contact-hypothesis/v1", content
        )[:20],
        grid_key=grid_fingerprint,
        source_m02a_parse_id=source_m02a_parse_id,
        background=background,
        anchor_object_id=anchor.object_id,
        anchor_color=anchor_color,
        anchor_bbox=anchor.bbox,
        anchor_fills_bbox=anchor_fills_bbox,
        relations=relations_tuple,
        unmatched_object_ids=unmatched_tuple,
        non_singleton_object_ids=non_singleton_tuple,
        same_color_object_ids=same_color_tuple,
        destination_collision_groups=collisions,
        structural_status=status,
    )


def parse_bbox_contacts(grid: Grid, *, background: int) -> BBoxContactParseBundle:
    """Build one background-specific M02c contact bundle from a blind grid."""

    normalized = as_grid(grid)
    if type(background) is not int or not 0 <= background <= 9:
        raise ValueError("background must be an ARC color")
    source = parse_grid(
        normalized,
        backgrounds=(background,),
        connectivities=(4,),
        color_modes=(ColorMode.SINGLE_COLOR,),
        max_backgrounds=None,
    ).hypotheses[0]
    pre_candidates = tuple(
        item
        for item in source.objects
        if item.bbox.height >= 2 and item.bbox.width >= 2
    )
    filled_anchor_ids = frozenset(
        item.object_id for item in pre_candidates if _is_filled_anchor(item)
    )
    fingerprint = grid_key(normalized)
    hypotheses = tuple(
        sorted(
            (
                _hypothesis(
                    grid=normalized,
                    grid_fingerprint=fingerprint,
                    source_m02a_parse_id=source.parse_id,
                    background=background,
                    anchor=anchor,
                    objects=source.objects,
                    filled_anchor_ids=filled_anchor_ids,
                )
                for anchor in pre_candidates
            ),
            key=lambda item: item.relation_parse_id,
        )
    )
    status = _bundle_status(hypotheses)
    content = {
        "semantics_version": RELATION_PARSER_SEMANTICS_VERSION,
        "stage": RELATION_PARSER_STAGE,
        "grid_key": fingerprint,
        "background": background,
        "source_m02a_parse_id": source.parse_id,
        "hypotheses": [item.to_json_dict() for item in hypotheses],
        "structural_status": status.value,
    }
    return BBoxContactParseBundle(
        bundle_id=_canonical_hash("afts-bbox-contact-bundle/v1", content),
        semantics_version=RELATION_PARSER_SEMANTICS_VERSION,
        stage=RELATION_PARSER_STAGE,
        grid_key=fingerprint,
        background=background,
        source_m02a_parse_id=source.parse_id,
        hypotheses=hypotheses,
        structural_status=status,
    )
