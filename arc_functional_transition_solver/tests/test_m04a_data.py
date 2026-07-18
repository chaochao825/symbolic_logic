from __future__ import annotations

import hashlib
import io
import json
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path

from afts_arc._source_bootstrap import named_bytes_fingerprint
from afts_arc.manifest import publish_evidence_bundle, serialize_json
from afts_arc.m04a_contract import episode_seed
from afts_arc.m04a_data import (
    ARC2Parent,
    M04AExample,
    M04ATrainingData,
    build_training_episode,
    build_validation_episodes,
    load_sanitized_m04a_data,
)


class _TinyReARC:
    def __init__(self, parent_ids: tuple[str, ...] = ("bbbbbbbb",)) -> None:
        self._parent_ids = parent_ids

    @property
    def parent_ids(self) -> tuple[str, ...]:
        return self._parent_ids

    def example(self, parent_id: str, example_index: int) -> M04AExample:
        if parent_id not in self._parent_ids or not 0 <= example_index < 1000:
            raise KeyError((parent_id, example_index))
        color = example_index % 10
        return M04AExample.create([[color, 0]], [[0], [color]])

    def semantic_parent_id(self, parent_id: str) -> str:
        if parent_id not in self._parent_ids:
            raise KeyError(parent_id)
        return parent_id


def _data() -> M04ATrainingData:
    parent = ARC2Parent(
        parent_id="aaaaaaaa",
        train=(
            M04AExample.create([[0, 1]], [[1, 0]]),
            M04AExample.create([[2], [0]], [[0, 2]]),
        ),
        test=(M04AExample.create([[3, 0]], [[3, 3]]),),
    )
    return M04ATrainingData(
        arc2_parent_ids=(parent.parent_id,),
        arc2_parents={parent.parent_id: parent},
        rearc_parent_ids=("bbbbbbbb",),
        rearc=_TinyReARC(),
    )


def _zip_one(name: str, content: bytes) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(
        buffer, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
        info.create_system = 3
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o100644 << 16
        archive.writestr(info, content, compresslevel=9)
    return buffer.getvalue()


def _sanitized_fixture(root: Path) -> Path:
    arc_raw = json.dumps(
        {
            "train": [{"input": [[0]], "output": [[1]]}],
            "test": [{"input": [[1]], "output": [[0]]}],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    rearc_raw = json.dumps(
        [{"input": [[index % 10]], "output": [[(index + 1) % 10]]} for index in range(1000)],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    definitions = (
        ("arc2_train.zip", "arc2", "train", "aaaaaaaa", arc_raw),
        ("arc2_validation.zip", "arc2", "validation", "bbbbbbbb", arc_raw),
        ("rearc_train.zip", "rearc", "train", "cccccccc", rearc_raw),
        ("rearc_validation.zip", "rearc", "validation", "dddddddd", rearc_raw),
    )
    artifacts: dict[str, bytes] = {}
    shards: dict[str, dict[str, object]] = {}
    allowlists: dict[str, list[str]] = {}
    counts: dict[str, int] = {}
    for artifact_name, source, fold, parent_id, content in definitions:
        member_name = f"{source}/{fold}/{parent_id}.json"
        archive_bytes = _zip_one(member_name, content)
        artifacts[artifact_name] = archive_bytes
        key = f"{source}_{fold}"
        allowlists[key] = [parent_id]
        counts[key] = 1
        shards[artifact_name] = {
            "source": source,
            "fold": fold,
            "member_prefix": f"{source}/{fold}/",
            "compression": "ZIP_DEFLATED_level_9",
            "zip_member_timestamp": "1980-01-01T00:00:00",
            "zip_member_mode": "0100644",
            "member_count": 1,
            "ordered_source_parent_ids": [parent_id],
            "member_aggregate_sha256": named_bytes_fingerprint(
                ((member_name, content),)
            ),
            "members": [
                {
                    "path": member_name,
                    "source_parent_id": parent_id,
                    "semantic_parent_id": parent_id,
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "bytes": len(content),
                }
            ],
            "zip_sha256": hashlib.sha256(archive_bytes).hexdigest(),
            "zip_bytes": len(archive_bytes),
        }
    manifest = {
        "schema": "afts.m04a-sanitized-split/v1",
        "data_folds_semantics_version": "afts-m04a-parent-folds/v0.1",
        "sealed_artifact_manifest_sha256": "0" * 64,
        "source_commitments": {},
        "usable_fold_counts": counts,
        "ordered_allowlists": allowlists,
        "semantic_aliases_in_usable_shards": [],
        "validation_quarantine_ids": [],
        "training_quarantine_ids": [],
        "quarantine_reasons": {},
        "shards": shards,
        "protected_payloads_absent": True,
    }
    artifacts["data_split_manifest.json"] = serialize_json(manifest)
    publish_evidence_bundle(root, artifacts=artifacts, run_id="fixture")
    return root


def _rehash_sanitized_fixture(root: Path) -> str:
    artifact_path = root / "artifact_manifest.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    split_bytes = (root / "data_split_manifest.json").read_bytes()
    artifact["artifacts"]["data_split_manifest.json"]["sha256"] = hashlib.sha256(
        split_bytes
    ).hexdigest()
    artifact["artifacts"]["data_split_manifest.json"]["bytes"] = len(split_bytes)
    artifact_path.write_bytes(serialize_json(artifact))
    return hashlib.sha256(artifact_path.read_bytes()).hexdigest()


class M04ADataTests(unittest.TestCase):
    def test_training_stream_alternates_and_replays_exactly(self) -> None:
        data = _data()
        arc_first = build_training_episode(data, 0, 0)
        arc_repeat = build_training_episode(data, 0, 0)
        rearc = build_training_episode(data, 0, 1)
        self.assertEqual(arc_first, arc_repeat)
        self.assertEqual(arc_first.source, "arc2")
        self.assertEqual(rearc.source, "rearc")
        self.assertEqual(arc_first.seed_u64, episode_seed(0, 0))
        self.assertEqual(rearc.seed_u64, episode_seed(0, 1))
        self.assertEqual(sorted(arc_first.color_permutation), list(range(10)))
        self.assertTrue(arc_first.masked_linear_indices)
        self.assertTrue(rearc.masked_linear_indices)
        self.assertEqual(
            arc_first.to_json_dict()["episode_sha256"],
            arc_repeat.to_json_dict()["episode_sha256"],
        )
        self.assertEqual(
            {
                "source": arc_first.source,
                "parent_id": arc_first.parent_id,
                "target_descriptor": arc_first.target_descriptor,
                "demonstration_descriptors": arc_first.demonstration_descriptors,
                "d4_index": arc_first.d4_index,
                "color_permutation": arc_first.color_permutation,
                "masked_linear_indices": arc_first.masked_linear_indices,
                "corruption_kind": arc_first.corruption_kind,
                "episode_sha256": arc_first.to_json_dict()["episode_sha256"],
            },
            {
                "source": "arc2",
                "parent_id": "aaaaaaaa",
                "target_descriptor": "train:0",
                "demonstration_descriptors": ("train:1",),
                "d4_index": 2,
                "color_permutation": (9, 3, 2, 0, 7, 5, 4, 8, 1, 6),
                "masked_linear_indices": (0,),
                "corruption_kind": "partial",
                "episode_sha256": "dbd379757efaf0b62195b2c63026d5f36085007007286f585de38ba24514bb0f",
            },
        )
        self.assertEqual(
            {
                "target_descriptor": rearc.target_descriptor,
                "demonstration_descriptors": rearc.demonstration_descriptors,
                "d4_index": rearc.d4_index,
                "color_permutation": rearc.color_permutation,
                "masked_linear_indices": rearc.masked_linear_indices,
                "episode_sha256": rearc.to_json_dict()["episode_sha256"],
            },
            {
                "target_descriptor": "rearc:33",
                "demonstration_descriptors": (
                    "rearc:84",
                    "rearc:449",
                    "rearc:608",
                ),
                "d4_index": 1,
                "color_permutation": (0, 6, 9, 8, 3, 4, 1, 5, 2, 7),
                "masked_linear_indices": (0,),
                "episode_sha256": "716ab73709d873cc83138a0453f18874e30fa9c4e6db0e3c305cef76b565ce4b",
            },
        )

    def test_validation_views_have_exact_nonzero_mask_counts(self) -> None:
        episodes = build_validation_episodes(_data())
        arc = [episode for episode in episodes if episode.source == "arc2"]
        rearc = [episode for episode in episodes if episode.source == "rearc"]
        # Three ARC targets (one test then two train pseudo-queries), four views each.
        self.assertEqual(len(arc), 12)
        self.assertEqual(len(rearc), 4)
        first_target = arc[:4]
        cell_count = len(first_target[0].target_output) * len(
            first_target[0].target_output[0]
        )
        self.assertEqual(
            [len(episode.masked_linear_indices) for episode in first_target],
            [1, 1, 2, cell_count],
        )

    def test_more_than_ten_demos_and_mapping_mismatch_fail_closed(self) -> None:
        example = M04AExample.create([[0]], [[1]])
        with self.assertRaisesRegex(ValueError, "ten"):
            ARC2Parent(
                parent_id="too_many",
                train=tuple(example for _ in range(11)),
                test=(example,),
            )
        parent = ARC2Parent(parent_id="one", train=(example,), test=(example,))
        with self.assertRaisesRegex(ValueError, "mapping"):
            M04ATrainingData(
                arc2_parent_ids=("other",),
                arc2_parents={"one": parent},
                rearc_parent_ids=("bbbbbbbb",),
                rearc=_TinyReARC(),
            )

    def test_sanitized_loader_checks_external_commitment_and_closed_world(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle = _sanitized_fixture(Path(directory) / "sanitized")
            artifact_sha = hashlib.sha256(
                (bundle / "artifact_manifest.json").read_bytes()
            ).hexdigest()
            loaded = load_sanitized_m04a_data(
                bundle,
                require_production_counts=False,
                expected_artifact_manifest_sha256=artifact_sha,
            )
            self.assertEqual(loaded.training.arc2_parent_ids, ("aaaaaaaa",))
            self.assertEqual(loaded.validation.rearc_parent_ids, ("dddddddd",))
            self.assertEqual(
                loaded.training.rearc.example("cccccccc", 999).input_grid,
                ((9,),),
            )
            with self.assertRaisesRegex(ValueError, "external commitment"):
                load_sanitized_m04a_data(
                    bundle,
                    require_production_counts=False,
                    expected_artifact_manifest_sha256="f" * 64,
                )
            (bundle / "unexpected.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "extra"):
                load_sanitized_m04a_data(
                    bundle,
                    require_production_counts=False,
                    expected_artifact_manifest_sha256=artifact_sha,
                )

    def test_sanitized_loader_closes_versions_aliases_and_quarantines(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = _sanitized_fixture(root / "base")
            cases = root / "cases"
            cases.mkdir()

            wrong_version = cases / "wrong-version"
            shutil.copytree(base, wrong_version)
            manifest = json.loads(
                (wrong_version / "data_split_manifest.json").read_text("utf-8")
            )
            manifest["data_folds_semantics_version"] = "wrong"
            (wrong_version / "data_split_manifest.json").write_bytes(
                serialize_json(manifest)
            )
            commitment = _rehash_sanitized_fixture(wrong_version)
            with self.assertRaisesRegex(ValueError, "semantics version"):
                load_sanitized_m04a_data(
                    wrong_version,
                    require_production_counts=False,
                    expected_artifact_manifest_sha256=commitment,
                )

            alias_mismatch = cases / "alias-mismatch"
            shutil.copytree(base, alias_mismatch)
            manifest = json.loads(
                (alias_mismatch / "data_split_manifest.json").read_text("utf-8")
            )
            manifest["semantic_aliases_in_usable_shards"] = [
                {"rearc_parent_id": "cccccccc", "arc2_parent_id": "070dd51e"}
            ]
            (alias_mismatch / "data_split_manifest.json").write_bytes(
                serialize_json(manifest)
            )
            commitment = _rehash_sanitized_fixture(alias_mismatch)
            with self.assertRaisesRegex(ValueError, "alias closure"):
                load_sanitized_m04a_data(
                    alias_mismatch,
                    require_production_counts=False,
                    expected_artifact_manifest_sha256=commitment,
                )

            quarantined = cases / "quarantined"
            shutil.copytree(base, quarantined)
            manifest = json.loads(
                (quarantined / "data_split_manifest.json").read_text("utf-8")
            )
            manifest["training_quarantine_ids"] = ["aaaaaaaa"]
            manifest["quarantine_reasons"] = {"aaaaaaaa": ["fixture"]}
            (quarantined / "data_split_manifest.json").write_bytes(
                serialize_json(manifest)
            )
            commitment = _rehash_sanitized_fixture(quarantined)
            with self.assertRaisesRegex(ValueError, "usable ARC2"):
                load_sanitized_m04a_data(
                    quarantined,
                    require_production_counts=False,
                    expected_artifact_manifest_sha256=commitment,
                )


if __name__ == "__main__":
    unittest.main()
