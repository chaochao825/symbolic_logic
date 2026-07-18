"""Closed-world materialization of the frozen M04a validation episodes.

The training runtime must consume the literal rows published here rather than
rebuilding validation episodes while a checkpoint is being selected.  This
module intentionally uses only the Python standard library and the pure M04a
data/evidence modules; importing it must never initialize PyTorch.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .grid import Grid, grid_to_lists
from .m04a_contract import (
    VALIDATION_SEMANTICS_VERSION,
    apply_indexed_d4,
    canonical_sha256,
)
from .m04a_data import (
    ARC2_SOURCE,
    REARC_SOURCE,
    M04AEpisode,
    M04AExample,
)
from .m04a_evidence import read_closed_world_bundle
from .manifest import publish_evidence_bundle, serialize_json, serialize_jsonl


VALIDATION_EPISODE_ROW_SCHEMA_VERSION = (
    "afts-grid-cmlm-validation-episode-row/v0.1"
)
VALIDATION_EPISODE_SUMMARY_SCHEMA_VERSION = (
    "afts-grid-cmlm-validation-episode-summary/v0.1"
)
VALIDATION_EPISODE_JSONL = "validation_episode_manifest.jsonl"
VALIDATION_EPISODE_SUMMARY_JSON = "validation_episode_manifest_summary.json"
VALIDATION_MANIFEST_COMMITMENT_SCHEMA_VERSION = (
    "afts-m04a-validation-manifest-commitment/v0.1"
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_READER_ATTESTATION_TOKEN = object()
_INDEXED_DESCRIPTOR = re.compile(r"(test|train|rearc):(0|[1-9][0-9]*)\Z")
_ROW_FIELDS = {
    "schema",
    "row_ordinal",
    "validation_semantics_version",
    "source",
    "source_parent_id",
    "semantic_parent_id",
    "target_descriptor",
    "target_group_id",
    "demonstration_descriptors",
    "demonstrations",
    "query_input",
    "target_output",
    "d4_index",
    "old_to_new_color_permutation",
    "seed_u64",
    "masked_linear_indices",
    "masked_coordinates",
    "corruption_kind",
    "view_index",
    "mask_fraction",
    "episode_sha256",
}
_SUMMARY_FIELDS = {
    "schema",
    "summary_id",
    "validation_semantics_version",
    "jsonl_artifact_name",
    "jsonl_sha256",
    "jsonl_bytes",
    "row_count",
    "target_group_count",
    "ordered_target_group_ids",
    "ordered_episode_sha256",
    "rows_semantic_sha256",
}


def _strict_int(value: object, *, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise TypeError(f"{field} must be an integer >= {minimum}")
    return value


def _strict_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{field} must be a non-empty string")
    return value


def _sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise TypeError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _exact_dict(value: object, fields: set[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object")
    if set(value) != fields:
        raise ValueError(
            f"{label} fields must be {sorted(fields)}, found {sorted(value)}"
        )
    return value


def _strict_json_bytes(content: bytes, *, label: str) -> object:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key in {label}: {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant in {label}: {value}")

    try:
        return json.loads(
            content.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON in {label}: {exc}") from exc


def _strict_jsonl_bytes(content: bytes) -> tuple[dict[str, Any], ...]:
    if not content:
        raise ValueError("validation episode JSONL must not be empty")
    if not content.endswith(b"\n"):
        raise ValueError("validation episode JSONL must end with one newline")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(content.splitlines(), start=1):
        if not line:
            raise ValueError(f"blank validation episode JSONL row {line_number}")
        row = _strict_json_bytes(line, label=f"validation JSONL row {line_number}")
        if not isinstance(row, dict):
            raise TypeError(f"validation JSONL row {line_number} must be an object")
        rows.append(row)
    return tuple(rows)


def validation_target_group_id(
    semantic_parent_id: str,
    source: str,
    source_parent_id: str,
    target_descriptor: str,
) -> str:
    """Hash the frozen target aggregation identity in its declared tuple order."""

    return canonical_sha256(
        [
            _strict_string(semantic_parent_id, field="semantic_parent_id"),
            _strict_string(source, field="source"),
            _strict_string(source_parent_id, field="source_parent_id"),
            _strict_string(target_descriptor, field="target_descriptor"),
        ]
    )


def _descriptor(value: str) -> tuple[str, int]:
    match = _INDEXED_DESCRIPTOR.fullmatch(value)
    if match is None:
        raise ValueError(f"invalid validation descriptor: {value!r}")
    return match.group(1), int(match.group(2))


def _rearc_validation_indices(parent_id: str, view_index: int) -> tuple[int, ...]:
    prefix = (
        f"{VALIDATION_SEMANTICS_VERSION}\0{parent_id}\0{view_index}\0"
    ).encode("utf-8")
    ranked = sorted(
        range(1000),
        key=lambda index: (
            hashlib.sha256(prefix + str(index).encode("ascii")).digest(),
            index,
        ),
    )
    return tuple(ranked[:4])


def _validate_seeded_augmentation(episode: M04AEpisode) -> None:
    """Replay every validation RNG call that is observable in a stored row."""

    if episode.validation_view_index is None:
        raise ValueError("validation manifest cannot contain a training episode")
    rng = random.Random(episode.seed_u64)
    expected_d4 = rng.randrange(8)
    expected_permutation = list(range(10))
    rng.shuffle(expected_permutation)

    kind, target_index = _descriptor(episode.target_descriptor)
    if episode.source == ARC2_SOURCE:
        if kind not in {"test", "train"}:
            raise ValueError("ARC2 target descriptor must be test:<n> or train:<n>")
        if kind == "test":
            pre_shuffle = tuple(
                f"train:{index}" for index in range(len(episode.demonstrations))
            )
        else:
            original_train_count = len(episode.demonstrations) + 1
            if target_index >= original_train_count:
                raise ValueError("ARC2 train target index exceeds inferred train count")
            pre_shuffle = tuple(
                f"train:{index}"
                for index in range(original_train_count)
                if index != target_index
            )
    elif episode.source == REARC_SOURCE:
        if kind != "rearc":
            raise ValueError("ReARC target descriptor must be rearc:<n>")
        selected = _rearc_validation_indices(
            episode.parent_id, episode.validation_view_index
        )
        if target_index != selected[3]:
            raise ValueError("ReARC target descriptor does not match frozen hash rank")
        pre_shuffle = tuple(f"rearc:{index}" for index in selected[:3])
    else:  # M04AEpisode rejects this; retained as a defensive boundary.
        raise ValueError("unsupported validation source")

    order = list(range(len(pre_shuffle)))
    rng.shuffle(order)
    expected_descriptors = tuple(pre_shuffle[index] for index in order)
    if episode.d4_index != expected_d4:
        raise ValueError("stored D4 index does not replay from validation seed")
    if episode.color_permutation != tuple(expected_permutation):
        raise ValueError("stored old-to-new color permutation does not replay")
    if episode.demonstration_descriptors != expected_descriptors:
        raise ValueError("stored demonstration order does not replay from validation seed")


def _deaugment_grid(episode: M04AEpisode, grid: Grid) -> Grid:
    inverse_permutation = [0] * 10
    for old_color, new_color in enumerate(episode.color_permutation):
        inverse_permutation[new_color] = old_color
    uncolored = tuple(
        tuple(inverse_permutation[color] for color in row)
        for row in grid
    )
    inverse_d4 = (0, 3, 2, 1, 4, 5, 6, 7)[episode.d4_index]
    return apply_indexed_d4(uncolored, inverse_d4)


def _deaugmented_examples(
    episode: M04AEpisode,
) -> tuple[dict[str, M04AExample], M04AExample]:
    demos = {
        descriptor: M04AExample.create(
            _deaugment_grid(episode, example.input_grid),
            _deaugment_grid(episode, example.output_grid),
        )
        for descriptor, example in zip(
            episode.demonstration_descriptors,
            episode.demonstrations,
            strict=True,
        )
    }
    if len(demos) != len(episode.demonstrations):
        raise ValueError("validation demonstration descriptors must be unique")
    target = M04AExample.create(
        _deaugment_grid(episode, episode.query_input),
        _deaugment_grid(episode, episode.target_output),
    )
    return demos, target


def _episode_to_row(episode: M04AEpisode, row_ordinal: int) -> dict[str, Any]:
    if episode.validation_view_index is None or episode.validation_mask_fraction is None:
        raise ValueError("validation manifest cannot contain a training episode")
    target_width = len(episode.target_output[0])
    episode_payload = episode.to_json_dict()
    return {
        "schema": VALIDATION_EPISODE_ROW_SCHEMA_VERSION,
        "row_ordinal": row_ordinal,
        "validation_semantics_version": VALIDATION_SEMANTICS_VERSION,
        "source": episode.source,
        "source_parent_id": episode.parent_id,
        "semantic_parent_id": episode.semantic_parent_id,
        "target_descriptor": episode.target_descriptor,
        "target_group_id": validation_target_group_id(
            episode.semantic_parent_id,
            episode.source,
            episode.parent_id,
            episode.target_descriptor,
        ),
        "demonstration_descriptors": list(episode.demonstration_descriptors),
        "demonstrations": [
            {
                "input": grid_to_lists(example.input_grid),
                "output": grid_to_lists(example.output_grid),
            }
            for example in episode.demonstrations
        ],
        "query_input": grid_to_lists(episode.query_input),
        "target_output": grid_to_lists(episode.target_output),
        "d4_index": episode.d4_index,
        "old_to_new_color_permutation": list(episode.color_permutation),
        "seed_u64": episode.seed_u64,
        "masked_linear_indices": list(episode.masked_linear_indices),
        "masked_coordinates": [
            [linear // target_width, linear % target_width]
            for linear in episode.masked_linear_indices
        ],
        "corruption_kind": episode.corruption_kind,
        "view_index": episode.validation_view_index,
        "mask_fraction": episode.validation_mask_fraction,
        "episode_sha256": episode_payload["episode_sha256"],
    }


def validate_validation_episode_row(
    row: object, *, expected_ordinal: int
) -> M04AEpisode:
    """Strictly reconstruct one immutable episode and replay its semantics."""

    payload = _exact_dict(row, _ROW_FIELDS, label="validation episode row")
    expected_ordinal = _strict_int(expected_ordinal, field="expected_ordinal")
    if payload["schema"] != VALIDATION_EPISODE_ROW_SCHEMA_VERSION:
        raise ValueError("unsupported validation episode row schema")
    row_ordinal = _strict_int(payload["row_ordinal"], field="row_ordinal")
    if row_ordinal != expected_ordinal:
        raise ValueError("validation row_ordinal does not match literal JSONL order")
    if payload["validation_semantics_version"] != VALIDATION_SEMANTICS_VERSION:
        raise ValueError("validation row semantics version mismatch")
    if payload["corruption_kind"] != "validation":
        raise ValueError("validation row corruption_kind must be validation")

    descriptors = payload["demonstration_descriptors"]
    demonstrations = payload["demonstrations"]
    if not isinstance(descriptors, list) or not isinstance(demonstrations, list):
        raise TypeError("validation demonstrations and descriptors must be JSON arrays")
    parsed_demos: list[M04AExample] = []
    for index, raw_demo in enumerate(demonstrations):
        demo = _exact_dict(
            raw_demo,
            {"input", "output"},
            label=f"validation demonstrations[{index}]",
        )
        parsed_demos.append(M04AExample.create(demo["input"], demo["output"]))
    if any(not isinstance(value, str) or not value for value in descriptors):
        raise TypeError("demonstration descriptors must be non-empty strings")

    permutation = payload["old_to_new_color_permutation"]
    masked = payload["masked_linear_indices"]
    coordinates = payload["masked_coordinates"]
    if not isinstance(permutation, list) or not isinstance(masked, list):
        raise TypeError("permutation and masked indices must be JSON arrays")
    for index, value in enumerate(permutation):
        _strict_int(
            value,
            field=f"old_to_new_color_permutation[{index}]",
        )
    for index, value in enumerate(masked):
        _strict_int(value, field=f"masked_linear_indices[{index}]")
    if not isinstance(coordinates, list):
        raise TypeError("masked_coordinates must be a JSON array")
    for index, coordinate in enumerate(coordinates):
        if (
            not isinstance(coordinate, list)
            or len(coordinate) != 2
            or any(type(value) is not int or value < 0 for value in coordinate)
        ):
            raise TypeError(
                f"masked_coordinates[{index}] must be [nonnegative row, column]"
            )

    episode = M04AEpisode(
        source=_strict_string(payload["source"], field="source"),
        parent_id=_strict_string(
            payload["source_parent_id"], field="source_parent_id"
        ),
        semantic_parent_id=_strict_string(
            payload["semantic_parent_id"], field="semantic_parent_id"
        ),
        target_descriptor=_strict_string(
            payload["target_descriptor"], field="target_descriptor"
        ),
        demonstration_descriptors=tuple(descriptors),
        demonstrations=tuple(parsed_demos),
        query_input=payload["query_input"],
        target_output=payload["target_output"],
        d4_index=payload["d4_index"],
        color_permutation=tuple(permutation),
        seed_u64=payload["seed_u64"],
        masked_linear_indices=tuple(masked),
        corruption_kind="validation",
        validation_view_index=payload["view_index"],
        validation_mask_fraction=payload["mask_fraction"],
    )
    _validate_seeded_augmentation(episode)

    target_width = len(episode.target_output[0])
    expected_coordinates = [
        [linear // target_width, linear % target_width]
        for linear in episode.masked_linear_indices
    ]
    if coordinates != expected_coordinates:
        raise ValueError("masked coordinates do not match masked linear indices")

    expected_group_id = validation_target_group_id(
        episode.semantic_parent_id,
        episode.source,
        episode.parent_id,
        episode.target_descriptor,
    )
    if _sha256(payload["target_group_id"], field="target_group_id") != expected_group_id:
        raise ValueError("target_group_id does not match its frozen identity tuple")
    expected_episode_sha = episode.to_json_dict()["episode_sha256"]
    if _sha256(payload["episode_sha256"], field="episode_sha256") != expected_episode_sha:
        raise ValueError("episode_sha256 does not match reconstructed episode")
    return episode


def _validate_population(episodes: Sequence[M04AEpisode]) -> None:
    if not episodes:
        raise ValueError("validation episode population must not be empty")
    identities: set[tuple[str, str, str, int]] = set()
    by_parent: dict[tuple[str, str], list[M04AEpisode]] = {}
    for episode in episodes:
        if episode.validation_view_index is None:
            raise ValueError("validation population contains a training episode")
        identity = (
            episode.source,
            episode.parent_id,
            episode.target_descriptor,
            episode.validation_view_index,
        )
        if identity in identities:
            raise ValueError("duplicate validation source/parent/target/view identity")
        identities.add(identity)
        by_parent.setdefault((episode.source, episode.parent_id), []).append(episode)

    for (source, parent_id), parent_rows in by_parent.items():
        semantic_ids = {row.semantic_parent_id for row in parent_rows}
        if len(semantic_ids) != 1:
            raise ValueError("one source parent maps to multiple semantic parents")
        views = [row.validation_view_index for row in parent_rows]
        if source == REARC_SOURCE:
            if sorted(views) != [0, 1, 2, 3]:
                raise ValueError("each ReARC validation parent requires views 0..3 once")
            observed_examples: dict[str, M04AExample] = {}
            for row in parent_rows:
                demonstrations, target = _deaugmented_examples(row)
                for descriptor, example in {
                    **demonstrations,
                    row.target_descriptor: target,
                }.items():
                    previous = observed_examples.setdefault(descriptor, example)
                    if previous != example:
                        raise ValueError(
                            "one ReARC example descriptor has inconsistent augmented grids"
                        )
            continue
        if source != ARC2_SOURCE:
            raise ValueError("unsupported validation source")
        if semantic_ids != {parent_id}:
            raise ValueError("ARC2 validation semantic parent must equal source parent")

        groups: dict[str, list[M04AEpisode]] = {}
        for row in parent_rows:
            groups.setdefault(row.target_descriptor, []).append(row)
        if any(
            sorted(row.validation_view_index for row in group) != [0, 1, 2, 3]
            for group in groups.values()
        ):
            raise ValueError("each ARC2 validation target requires views 0..3 once")
        recovered_groups: dict[
            str, tuple[dict[str, M04AExample], M04AExample]
        ] = {}
        for descriptor, group in groups.items():
            recovered = tuple(_deaugmented_examples(row) for row in group)
            if any(value != recovered[0] for value in recovered[1:]):
                raise ValueError(
                    "ARC2 target views do not deaugment to one source episode"
                )
            recovered_groups[descriptor] = recovered[0]
        test_indices: list[int] = []
        train_indices: list[int] = []
        train_counts: set[int] = set()
        for descriptor, group in groups.items():
            kind, index = _descriptor(descriptor)
            if kind == "test":
                test_indices.append(index)
                train_counts.add(len(group[0].demonstrations))
            elif kind == "train":
                train_indices.append(index)
                train_counts.add(len(group[0].demonstrations) + 1)
            else:
                raise ValueError("ARC2 validation target has a ReARC descriptor")
        if sorted(test_indices) != list(range(len(test_indices))) or not test_indices:
            raise ValueError("ARC2 test targets must be contiguous test:0..<n>")
        if len(train_counts) != 1:
            raise ValueError("ARC2 validation rows infer inconsistent train counts")
        train_count = next(iter(train_counts))
        expected_train = list(range(train_count)) if train_count > 1 else []
        if sorted(train_indices) != expected_train:
            raise ValueError("ARC2 train pseudo-targets do not close to the train pairs")
        canonical_training_examples = recovered_groups["test:0"][0]
        if set(canonical_training_examples) != {
            f"train:{index}" for index in range(train_count)
        }:
            raise ValueError("ARC2 test views do not expose the complete train set")
        for test_index in test_indices:
            if recovered_groups[f"test:{test_index}"][0] != canonical_training_examples:
                raise ValueError("ARC2 test targets have inconsistent demonstrations")
        for train_index in train_indices:
            descriptor = f"train:{train_index}"
            demos, target = recovered_groups[descriptor]
            if target != canonical_training_examples[descriptor]:
                raise ValueError("ARC2 train pseudo-target grid differs from its train pair")
            expected_demos = {
                key: value
                for key, value in canonical_training_examples.items()
                if key != descriptor
            }
            if demos != expected_demos:
                raise ValueError("ARC2 train pseudo-target demonstrations do not close")


def _canonicalize_episodes(
    episodes: Sequence[M04AEpisode],
) -> tuple[M04AEpisode, ...]:
    normalized = tuple(episodes)
    if any(not isinstance(row, M04AEpisode) for row in normalized):
        raise TypeError("validation episodes must be M04AEpisode values")
    _validate_population(normalized)

    group_first_view: dict[tuple[str, str, str], int] = {}
    for episode in normalized:
        if episode.validation_view_index is None:  # Checked above; narrows type.
            raise RuntimeError("validation view unexpectedly absent")
        group = (episode.source, episode.parent_id, episode.target_descriptor)
        group_first_view[group] = min(
            group_first_view.get(group, episode.validation_view_index),
            episode.validation_view_index,
        )

    def key(episode: M04AEpisode) -> tuple[object, ...]:
        if episode.validation_view_index is None:
            raise RuntimeError("validation view unexpectedly absent")
        source_rank = 0 if episode.source == ARC2_SOURCE else 1
        kind, index = _descriptor(episode.target_descriptor)
        if episode.source == ARC2_SOURCE:
            target_key: tuple[object, ...] = (
                0 if kind == "test" else 1,
                index,
            )
        else:
            target_key = (
                group_first_view[
                    (episode.source, episode.parent_id, episode.target_descriptor)
                ],
                episode.target_descriptor,
            )
        return (
            source_rank,
            episode.parent_id,
            *target_key,
            episode.validation_view_index,
        )

    return tuple(sorted(normalized, key=key))


def _validate_group_runs(rows: Sequence[Mapping[str, Any]]) -> None:
    closed: set[str] = set()
    previous: str | None = None
    for row in rows:
        group_id = row["target_group_id"]
        if group_id != previous:
            if group_id in closed:
                raise ValueError("target-group rows must be contiguous")
            if previous is not None:
                closed.add(previous)
            previous = group_id


def validate_validation_episode_rows(
    rows: Sequence[dict[str, Any]],
) -> tuple[M04AEpisode, ...]:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise TypeError("validation episode rows must be an ordered sequence")
    episodes = tuple(
        validate_validation_episode_row(row, expected_ordinal=ordinal)
        for ordinal, row in enumerate(rows)
    )
    _validate_group_runs(rows)
    if episodes != _canonicalize_episodes(episodes):
        raise ValueError("validation episode rows are not in canonical order")
    return episodes


def _ordered_group_ids(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    result: list[str] = []
    for row in rows:
        group_id = row["target_group_id"]
        if not result or result[-1] != group_id:
            result.append(group_id)
    return result


def _make_summary(
    rows: Sequence[Mapping[str, Any]], jsonl_bytes: bytes
) -> dict[str, Any]:
    ordered_group_ids = _ordered_group_ids(rows)
    summary: dict[str, Any] = {
        "schema": VALIDATION_EPISODE_SUMMARY_SCHEMA_VERSION,
        "validation_semantics_version": VALIDATION_SEMANTICS_VERSION,
        "jsonl_artifact_name": VALIDATION_EPISODE_JSONL,
        "jsonl_sha256": hashlib.sha256(jsonl_bytes).hexdigest(),
        "jsonl_bytes": len(jsonl_bytes),
        "row_count": len(rows),
        "target_group_count": len(ordered_group_ids),
        "ordered_target_group_ids": ordered_group_ids,
        "ordered_episode_sha256": [row["episode_sha256"] for row in rows],
        "rows_semantic_sha256": canonical_sha256(list(rows)),
    }
    summary["summary_id"] = canonical_sha256(summary)
    return summary


def _validate_summary(
    summary: object,
    *,
    rows: Sequence[Mapping[str, Any]],
    jsonl_bytes: bytes,
) -> dict[str, Any]:
    payload = _exact_dict(
        summary, _SUMMARY_FIELDS, label="validation episode summary"
    )
    if payload["schema"] != VALIDATION_EPISODE_SUMMARY_SCHEMA_VERSION:
        raise ValueError("unsupported validation episode summary schema")
    if payload["validation_semantics_version"] != VALIDATION_SEMANTICS_VERSION:
        raise ValueError("validation episode summary semantics mismatch")
    if payload["jsonl_artifact_name"] != VALIDATION_EPISODE_JSONL:
        raise ValueError("validation summary names the wrong JSONL artifact")
    for field_name in ("jsonl_bytes", "row_count", "target_group_count"):
        _strict_int(payload[field_name], field=field_name)
    for field_name in ("jsonl_sha256", "rows_semantic_sha256", "summary_id"):
        _sha256(payload[field_name], field=field_name)
    for field_name in ("ordered_target_group_ids", "ordered_episode_sha256"):
        values = payload[field_name]
        if not isinstance(values, list):
            raise TypeError(f"{field_name} must be a JSON array")
        for index, value in enumerate(values):
            _sha256(value, field=f"{field_name}[{index}]")
    expected = _make_summary(rows, jsonl_bytes)
    if payload != expected:
        raise ValueError("validation episode summary does not close over exact rows/bytes")
    return payload


@dataclass(frozen=True, slots=True)
class ValidationEpisodeManifest:
    episodes: tuple[M04AEpisode, ...]
    rows: tuple[dict[str, Any], ...]
    summary: dict[str, Any]
    jsonl_bytes: bytes
    root: Path | None = None
    artifact_manifest_sha256: str | None = None
    _reader_attestation: object = field(default=None, repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class ValidationManifestCommitment:
    outer_artifact_manifest_sha256: str
    jsonl_sha256: str
    summary_id: str
    row_count: int
    commitment_id: str
    _reader_attestation: object = field(default=None, repr=False, compare=False)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema": VALIDATION_MANIFEST_COMMITMENT_SCHEMA_VERSION,
            "commitment_id": self.commitment_id,
            "outer_artifact_manifest_sha256": self.outer_artifact_manifest_sha256,
            "jsonl_sha256": self.jsonl_sha256,
            "summary_id": self.summary_id,
            "row_count": self.row_count,
        }


def validate_validation_manifest_commitment(
    commitment: ValidationManifestCommitment,
) -> dict[str, object]:
    if type(commitment) is not ValidationManifestCommitment:
        raise TypeError("validation commitment must be reader-produced")
    if commitment._reader_attestation is not _READER_ATTESTATION_TOKEN:
        raise ValueError("validation commitment lacks reader attestation")
    payload = commitment.to_json_dict()
    _sha256(
        payload["outer_artifact_manifest_sha256"],
        field="outer_artifact_manifest_sha256",
    )
    _sha256(payload["jsonl_sha256"], field="jsonl_sha256")
    _sha256(payload["summary_id"], field="summary_id")
    _strict_int(payload["row_count"], field="row_count", minimum=1)
    semantic = dict(payload)
    claimed_id = semantic.pop("commitment_id")
    _sha256(claimed_id, field="commitment_id")
    expected_id = canonical_sha256(semantic)
    if claimed_id != expected_id:
        raise ValueError("validation commitment_id does not match semantic content")
    return payload


def build_validation_manifest_commitment(
    manifest: ValidationEpisodeManifest,
    *,
    expected_artifact_manifest_sha256: str,
    expected_jsonl_sha256: str,
) -> ValidationManifestCommitment:
    attested = require_externally_committed_validation_manifest(
        manifest,
        expected_artifact_manifest_sha256=expected_artifact_manifest_sha256,
        expected_jsonl_sha256=expected_jsonl_sha256,
    )
    semantic: dict[str, object] = {
        "schema": VALIDATION_MANIFEST_COMMITMENT_SCHEMA_VERSION,
        "outer_artifact_manifest_sha256": expected_artifact_manifest_sha256,
        "jsonl_sha256": expected_jsonl_sha256,
        "summary_id": attested.summary["summary_id"],
        "row_count": attested.summary["row_count"],
    }
    commitment = ValidationManifestCommitment(
        outer_artifact_manifest_sha256=expected_artifact_manifest_sha256,
        jsonl_sha256=expected_jsonl_sha256,
        summary_id=str(attested.summary["summary_id"]),
        row_count=int(attested.summary["row_count"]),
        commitment_id=canonical_sha256(semantic),
        _reader_attestation=_READER_ATTESTATION_TOKEN,
    )
    validate_validation_manifest_commitment(commitment)
    return commitment


def require_externally_committed_validation_manifest(
    manifest: ValidationEpisodeManifest,
    *,
    expected_artifact_manifest_sha256: str,
    expected_jsonl_sha256: str,
) -> ValidationEpisodeManifest:
    """Require a reader-attested artifact bound by the launch plan and lineage."""

    if not isinstance(manifest, ValidationEpisodeManifest):
        raise TypeError("manifest must be a ValidationEpisodeManifest")
    expected_outer = _sha256(
        expected_artifact_manifest_sha256,
        field="expected_artifact_manifest_sha256",
    )
    expected_jsonl = _sha256(
        expected_jsonl_sha256,
        field="expected_jsonl_sha256",
    )
    if (
        manifest._reader_attestation is not _READER_ATTESTATION_TOKEN
        or manifest.root is None
        or manifest.artifact_manifest_sha256 != expected_outer
    ):
        raise ValueError(
            "production validation requires the externally committed reader-attested bundle"
        )
    if manifest.summary.get("jsonl_sha256") != expected_jsonl:
        raise ValueError("validation JSONL digest differs from training lineage")
    if hashlib.sha256(manifest.jsonl_bytes).hexdigest() != expected_jsonl:
        raise ValueError("validation manifest bytes differ from training lineage")
    return manifest


def build_validation_episode_artifacts(
    episodes: Sequence[M04AEpisode],
) -> ValidationEpisodeManifest:
    """Canonicalize and fully materialize the pre-training validation artifact."""

    canonical = _canonicalize_episodes(episodes)
    rows = tuple(
        _episode_to_row(episode, ordinal)
        for ordinal, episode in enumerate(canonical)
    )
    reconstructed = validate_validation_episode_rows(rows)
    if reconstructed != canonical:
        raise RuntimeError("validation row reconstruction changed an episode")
    jsonl_bytes = serialize_jsonl(rows)
    summary = _make_summary(rows, jsonl_bytes)
    _validate_summary(summary, rows=rows, jsonl_bytes=jsonl_bytes)
    return ValidationEpisodeManifest(
        episodes=canonical,
        rows=rows,
        summary=summary,
        jsonl_bytes=jsonl_bytes,
    )


def publish_validation_episode_manifest(
    output_dir: str | Path,
    episodes: Sequence[M04AEpisode],
) -> ValidationEpisodeManifest:
    """Publish once via a sibling staging directory and verify the result."""

    materialized = build_validation_episode_artifacts(episodes)
    outer_manifest = publish_evidence_bundle(
        output_dir,
        artifacts={
            VALIDATION_EPISODE_JSONL: materialized.jsonl_bytes,
            VALIDATION_EPISODE_SUMMARY_JSON: serialize_json(materialized.summary),
        },
        run_id=materialized.summary["summary_id"],
    )
    root = Path(output_dir).expanduser().resolve()
    artifact_sha = hashlib.sha256(serialize_json(outer_manifest)).hexdigest()
    return read_validation_episode_manifest(
        root, expected_artifact_manifest_sha256=artifact_sha
    )


def read_validation_episode_manifest(
    directory: str | Path,
    *,
    expected_artifact_manifest_sha256: str,
) -> ValidationEpisodeManifest:
    """Read only an externally committed, closed-world validation artifact."""

    expected_sha = _sha256(
        expected_artifact_manifest_sha256,
        field="expected_artifact_manifest_sha256",
    )
    bundle = read_closed_world_bundle(Path(directory).expanduser())
    root = bundle.root
    actual_sha = hashlib.sha256(bundle.artifact_manifest_bytes).hexdigest()
    if actual_sha != expected_sha:
        raise ValueError("validation artifact-manifest external commitment mismatch")

    if set(bundle.artifacts) != {
        VALIDATION_EPISODE_JSONL,
        VALIDATION_EPISODE_SUMMARY_JSON,
    }:
        raise ValueError("validation episode artifact set is not closed-world")

    summary_bytes = bundle.read_bytes(VALIDATION_EPISODE_SUMMARY_JSON)
    summary_raw = _strict_json_bytes(
        summary_bytes, label=VALIDATION_EPISODE_SUMMARY_JSON
    )
    if not isinstance(summary_raw, dict):
        raise TypeError("validation episode summary must be a JSON object")
    if serialize_json(summary_raw) != summary_bytes:
        raise ValueError("validation episode summary is not canonically serialized")

    jsonl_bytes = bundle.read_bytes(VALIDATION_EPISODE_JSONL)
    rows = _strict_jsonl_bytes(jsonl_bytes)
    if serialize_jsonl(rows) != jsonl_bytes:
        raise ValueError("validation episode JSONL is not canonically serialized")
    episodes = validate_validation_episode_rows(rows)
    summary = _validate_summary(summary_raw, rows=rows, jsonl_bytes=jsonl_bytes)

    outer = bundle.artifact_manifest
    if outer.get("schema_version") != 1 or outer.get("run_id") != summary["summary_id"]:
        raise ValueError("outer artifact manifest does not bind validation summary_id")
    if bundle.artifacts[VALIDATION_EPISODE_JSONL]["rows"] != summary["row_count"]:
        raise ValueError("outer JSONL row count does not bind validation summary")
    return ValidationEpisodeManifest(
        episodes=episodes,
        rows=rows,
        summary=summary,
        jsonl_bytes=jsonl_bytes,
        root=root,
        artifact_manifest_sha256=actual_sha,
        _reader_attestation=_READER_ATTESTATION_TOKEN,
    )


__all__ = [
    "VALIDATION_EPISODE_JSONL",
    "VALIDATION_EPISODE_ROW_SCHEMA_VERSION",
    "VALIDATION_EPISODE_SUMMARY_JSON",
    "VALIDATION_EPISODE_SUMMARY_SCHEMA_VERSION",
    "VALIDATION_MANIFEST_COMMITMENT_SCHEMA_VERSION",
    "ValidationEpisodeManifest",
    "ValidationManifestCommitment",
    "build_validation_manifest_commitment",
    "build_validation_episode_artifacts",
    "publish_validation_episode_manifest",
    "require_externally_committed_validation_manifest",
    "read_validation_episode_manifest",
    "validate_validation_episode_row",
    "validate_validation_episode_rows",
    "validate_validation_manifest_commitment",
    "validation_target_group_id",
]
