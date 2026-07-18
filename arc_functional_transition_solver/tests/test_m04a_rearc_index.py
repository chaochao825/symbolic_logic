from __future__ import annotations

import hashlib
import io
import json
import os
import random
import shutil
import struct
import tempfile
import unittest
import zipfile
from pathlib import Path

from afts_arc.m04a_contract import canonical_sha256
from afts_arc.m04a_rearc_index import (
    CACHE_SEMANTICS_VERSION,
    DATA_MAGIC,
    EXAMPLES_PER_PARENT,
    INDEX_MAGIC,
    MMapReARCExampleSource,
    build_rearc_mmap_cache,
)
from afts_arc.manifest import serialize_json


def _example(index: int, *, seed: int) -> dict[str, object]:
    return {
        "input": [[(index + seed) % 10]],
        "output": [[(index + seed + 1) % 10, (index // 10 + seed) % 10]],
    }


def _member(seed: int) -> tuple[bytes, tuple[dict[str, object], ...]]:
    rows = tuple(_example(index, seed=seed) for index in range(EXAMPLES_PER_PARENT))
    return json.dumps(rows, separators=(",", ":"), sort_keys=True).encode("utf-8"), rows


def _zip(materials: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(
        buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name, content in sorted(materials.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, content, compresslevel=9)
    return buffer.getvalue()


def _artifact_manifest(artifacts: dict[str, bytes], *, run_id: str) -> bytes:
    return serialize_json(
        {
            "schema_version": 1,
            "bundle_status": "complete",
            "run_id": run_id,
            "artifacts": {
                name: {
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "bytes": len(content),
                    "rows": None,
                }
                for name, content in sorted(artifacts.items())
            },
        }
    )


def _sanitized_fixture(root: Path) -> tuple[Path, dict[str, tuple[dict[str, object], ...]]]:
    sanitized = root / "sanitized"
    sanitized.mkdir()
    parents_by_fold = {
        "train": (("aaaaaaaa", 1), ("cccccccc", 7)),
        "validation": (("bbbbbbbb", 4),),
    }
    expected: dict[str, tuple[dict[str, object], ...]] = {}
    artifacts: dict[str, bytes] = {
        "arc2_train.zip": _zip({}),
        "arc2_validation.zip": _zip({}),
    }
    shard_rows: dict[str, dict[str, object]] = {}
    allowlists: dict[str, list[str]] = {
        "arc2_train": [],
        "arc2_validation": [],
    }
    for fold, parent_specs in parents_by_fold.items():
        materials: dict[str, bytes] = {}
        member_rows: list[dict[str, object]] = []
        parent_ids: list[str] = []
        for parent_id, seed in parent_specs:
            content, rows = _member(seed)
            expected[parent_id] = rows
            parent_ids.append(parent_id)
            member_name = f"rearc/{fold}/{parent_id}.json"
            materials[member_name] = content
            member_rows.append(
                {
                    "path": member_name,
                    "source_parent_id": parent_id,
                    "semantic_parent_id": (
                        "070dd51e" if parent_id == "cccccccc" else parent_id
                    ),
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "bytes": len(content),
                }
            )
        archive = _zip(materials)
        artifact_name = f"rearc_{fold}.zip"
        artifacts[artifact_name] = archive
        allowlists[f"rearc_{fold}"] = parent_ids
        shard_rows[artifact_name] = {
            "source": "rearc",
            "fold": fold,
            "member_prefix": f"rearc/{fold}/",
            "compression": "ZIP_DEFLATED_level_9",
            "zip_member_timestamp": "1980-01-01T00:00:00",
            "zip_member_mode": "0100644",
            "member_count": len(parent_ids),
            "ordered_source_parent_ids": parent_ids,
            "member_aggregate_sha256": canonical_sha256(parent_ids),
            "members": member_rows,
            "zip_sha256": hashlib.sha256(archive).hexdigest(),
            "zip_bytes": len(archive),
        }
    split_manifest = serialize_json(
        {
            "schema": "afts.m04a-sanitized-split/v1",
            "protected_payloads_absent": True,
            "ordered_allowlists": allowlists,
            "shards": shard_rows,
        }
    )
    artifacts["data_split_manifest.json"] = split_manifest
    for name, content in artifacts.items():
        (sanitized / name).write_bytes(content)
    (sanitized / "artifact_manifest.json").write_bytes(
        _artifact_manifest(artifacts, run_id="sanitized-fixture")
    )
    return sanitized, expected


def _rehash_cache(cache: Path) -> str:
    manifest_path = cache / "cache_manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    for key, name in (("data", "data.bin"), ("index", "index.bin")):
        content = (cache / name).read_bytes()
        manifest[key]["sha256"] = hashlib.sha256(content).hexdigest()
        manifest[key]["bytes"] = len(content)
    manifest_path.write_bytes(serialize_json(manifest))
    artifact_path = cache / "artifact_manifest.json"
    artifact = json.loads(artifact_path.read_text("utf-8"))
    for name in ("cache_manifest.json", "data.bin", "index.bin"):
        content = (cache / name).read_bytes()
        artifact["artifacts"][name]["sha256"] = hashlib.sha256(content).hexdigest()
        artifact["artifacts"][name]["bytes"] = len(content)
    artifact_path.write_bytes(serialize_json(artifact))
    return hashlib.sha256(artifact_path.read_bytes()).hexdigest()


class M04aReARCIndexTests(unittest.TestCase):
    def test_round_trip_random_access_and_fold_binding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sanitized, expected = _sanitized_fixture(root)
            sanitized_sha = hashlib.sha256(
                (sanitized / "artifact_manifest.json").read_bytes()
            ).hexdigest()
            train_cache = root / "train-cache"
            result = build_rearc_mmap_cache(
                sanitized_split_dir=sanitized,
                fold="train",
                output_dir=train_cache,
                expected_sanitized_artifact_manifest_sha256=sanitized_sha,
                allow_nonproduction_fixture=True,
            )
            self.assertEqual(result["parent_count"], 2)
            self.assertEqual(result["total_example_count"], 2 * EXAMPLES_PER_PARENT)
            self.assertEqual(
                result["input_sanitized_artifact_manifest_sha256"], sanitized_sha
            )
            self.assertEqual(
                {path.name for path in train_cache.iterdir()},
                {
                    "artifact_manifest.json",
                    "cache_manifest.json",
                    "data.bin",
                    "index.bin",
                },
            )
            with MMapReARCExampleSource(
                train_cache,
                expected_artifact_manifest_sha256=result[
                    "artifact_manifest_sha256"
                ],
                expected_sanitized_artifact_manifest_sha256=sanitized_sha,
                allow_nonproduction_fixture=True,
            ) as source:
                self.assertEqual(source.parent_ids, ("aaaaaaaa", "cccccccc"))
                self.assertEqual(source.semantic_parent_id("aaaaaaaa"), "aaaaaaaa")
                self.assertEqual(source.semantic_parent_id("cccccccc"), "070dd51e")
                self.assertEqual(
                    source.input_sanitized_artifact_manifest_sha256, sanitized_sha
                )
                indices = [0, 123, 999]
                indices.extend(random.Random(20260711).sample(range(1000), 10))
                for index in indices:
                    row = expected["aaaaaaaa"][index]
                    actual = source.example("aaaaaaaa", index)
                    self.assertEqual(actual.input_grid, tuple(tuple(r) for r in row["input"]))
                    self.assertEqual(
                        actual.output_grid, tuple(tuple(r) for r in row["output"])
                    )
                with self.assertRaises(KeyError):
                    source.example("dddddddd", 0)
                with self.assertRaises(KeyError):
                    source.semantic_parent_id("dddddddd")
                with self.assertRaises(IndexError):
                    source.example("aaaaaaaa", 1000)
                second_parent_row = expected["cccccccc"][777]
                self.assertEqual(
                    source.example("cccccccc", 777).output_grid,
                    tuple(tuple(r) for r in second_parent_row["output"]),
                )
            with self.assertRaises(RuntimeError):
                source.example("aaaaaaaa", 0)
            with self.assertRaises(RuntimeError):
                source.semantic_parent_id("aaaaaaaa")
            with self.assertRaisesRegex(ValueError, "sanitized artifact manifest"):
                MMapReARCExampleSource(
                    train_cache,
                    expected_artifact_manifest_sha256=result[
                        "artifact_manifest_sha256"
                    ],
                    expected_sanitized_artifact_manifest_sha256="0" * 64,
                    allow_nonproduction_fixture=True,
                )
            with self.assertRaisesRegex(ValueError, "production reader"):
                MMapReARCExampleSource(
                    train_cache,
                    expected_artifact_manifest_sha256=result[
                        "artifact_manifest_sha256"
                    ],
                    expected_sanitized_artifact_manifest_sha256=sanitized_sha,
                )

            validation_cache = root / "validation-cache"
            validation_result = build_rearc_mmap_cache(
                sanitized_split_dir=sanitized,
                fold="validation",
                output_dir=validation_cache,
                expected_sanitized_artifact_manifest_sha256=sanitized_sha,
                allow_nonproduction_fixture=True,
            )
            with MMapReARCExampleSource(
                validation_cache,
                expected_artifact_manifest_sha256=validation_result[
                    "artifact_manifest_sha256"
                ],
                expected_sanitized_artifact_manifest_sha256=sanitized_sha,
                allow_nonproduction_fixture=True,
            ) as validation_source:
                self.assertEqual(validation_source.parent_ids, ("bbbbbbbb",))
                self.assertEqual(
                    validation_source.example("bbbbbbbb", 999).output_grid,
                    tuple(tuple(r) for r in expected["bbbbbbbb"][999]["output"]),
                )

            with self.assertRaises(FileExistsError):
                build_rearc_mmap_cache(
                    sanitized_split_dir=sanitized,
                    fold="train",
                    output_dir=train_cache,
                    expected_sanitized_artifact_manifest_sha256=sanitized_sha,
                    allow_nonproduction_fixture=True,
                )

    def test_reader_rejects_hash_truncate_extra_and_manifest_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sanitized, _ = _sanitized_fixture(root)
            sanitized_sha = hashlib.sha256(
                (sanitized / "artifact_manifest.json").read_bytes()
            ).hexdigest()
            base = root / "base-cache"
            result = build_rearc_mmap_cache(
                sanitized_split_dir=sanitized,
                fold="train",
                output_dir=base,
                expected_sanitized_artifact_manifest_sha256=sanitized_sha,
                allow_nonproduction_fixture=True,
            )
            expected_sha = result["artifact_manifest_sha256"]

            hash_tamper = root / "hash-tamper"
            shutil.copytree(base, hash_tamper)
            with (hash_tamper / "data.bin").open("r+b") as handle:
                handle.seek(-1, io.SEEK_END)
                handle.write(b"\xff")
            with self.assertRaisesRegex(ValueError, "hash/size mismatch"):
                MMapReARCExampleSource(
                    hash_tamper,
                    expected_artifact_manifest_sha256=expected_sha,
                    expected_sanitized_artifact_manifest_sha256=sanitized_sha,
                    allow_nonproduction_fixture=True,
                )

            truncated = root / "truncated"
            shutil.copytree(base, truncated)
            content = (truncated / "index.bin").read_bytes()
            (truncated / "index.bin").write_bytes(content[:-8])
            with self.assertRaisesRegex(ValueError, "hash/size mismatch"):
                MMapReARCExampleSource(
                    truncated,
                    expected_artifact_manifest_sha256=expected_sha,
                    expected_sanitized_artifact_manifest_sha256=sanitized_sha,
                    allow_nonproduction_fixture=True,
                )

            extra = root / "extra"
            shutil.copytree(base, extra)
            (extra / "unexpected.bin").write_bytes(b"bad")
            with self.assertRaisesRegex(ValueError, "file set mismatch"):
                MMapReARCExampleSource(
                    extra,
                    expected_artifact_manifest_sha256=expected_sha,
                    expected_sanitized_artifact_manifest_sha256=sanitized_sha,
                    allow_nonproduction_fixture=True,
                )

            manifest_tamper = root / "manifest-tamper"
            shutil.copytree(base, manifest_tamper)
            manifest = json.loads(
                (manifest_tamper / "cache_manifest.json").read_text("utf-8")
            )
            manifest["fold"] = "validation"
            (manifest_tamper / "cache_manifest.json").write_bytes(serialize_json(manifest))
            with self.assertRaisesRegex(ValueError, "hash/size mismatch"):
                MMapReARCExampleSource(
                    manifest_tamper,
                    expected_artifact_manifest_sha256=expected_sha,
                    expected_sanitized_artifact_manifest_sha256=sanitized_sha,
                    allow_nonproduction_fixture=True,
                )

    def test_reader_rejects_rehashed_magic_and_offset_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sanitized, _ = _sanitized_fixture(root)
            sanitized_sha = hashlib.sha256(
                (sanitized / "artifact_manifest.json").read_bytes()
            ).hexdigest()
            base = root / "base-cache"
            build_rearc_mmap_cache(
                sanitized_split_dir=sanitized,
                fold="train",
                output_dir=base,
                expected_sanitized_artifact_manifest_sha256=sanitized_sha,
                allow_nonproduction_fixture=True,
            )

            bad_magic = root / "bad-magic"
            shutil.copytree(base, bad_magic)
            with (bad_magic / "data.bin").open("r+b") as handle:
                handle.write(bytes([DATA_MAGIC[0] ^ 0x01]))
            bad_magic_sha = _rehash_cache(bad_magic)
            with self.assertRaisesRegex(ValueError, "data magic"):
                MMapReARCExampleSource(
                    bad_magic,
                    expected_artifact_manifest_sha256=bad_magic_sha,
                    expected_sanitized_artifact_manifest_sha256=sanitized_sha,
                    allow_nonproduction_fixture=True,
                )

            bad_offset = root / "bad-offset"
            shutil.copytree(base, bad_offset)
            schema_length = len(CACHE_SEMANTICS_VERSION.encode("ascii"))
            index_prefix = len(INDEX_MAGIC) + 2 + schema_length + struct.calcsize("<IIQ")
            offset_base = index_prefix + 8
            with (bad_offset / "index.bin").open("r+b") as handle:
                handle.seek(offset_base)
                first = handle.read(8)
                handle.seek(offset_base + 8)
                handle.write(first)
            bad_offset_sha = _rehash_cache(bad_offset)
            with self.assertRaisesRegex(ValueError, "offset table"):
                MMapReARCExampleSource(
                    bad_offset,
                    expected_artifact_manifest_sha256=bad_offset_sha,
                    expected_sanitized_artifact_manifest_sha256=sanitized_sha,
                    allow_nonproduction_fixture=True,
                )

    def test_builder_rejects_tampered_or_extended_sanitized_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sanitized, _ = _sanitized_fixture(root)
            sanitized_sha = hashlib.sha256(
                (sanitized / "artifact_manifest.json").read_bytes()
            ).hexdigest()
            (sanitized / "unexpected.txt").write_text("bad", encoding="utf-8")
            output = root / "cache"
            with self.assertRaisesRegex(ValueError, "file set mismatch"):
                build_rearc_mmap_cache(
                    sanitized_split_dir=sanitized,
                    fold="train",
                    output_dir=output,
                    expected_sanitized_artifact_manifest_sha256=sanitized_sha,
                    allow_nonproduction_fixture=True,
                )
            self.assertFalse(output.exists())

    def test_external_trust_defaults_and_symlink_artifacts_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sanitized, _ = _sanitized_fixture(root)
            sanitized_sha = hashlib.sha256(
                (sanitized / "artifact_manifest.json").read_bytes()
            ).hexdigest()
            with self.assertRaisesRegex(ValueError, "external commitment"):
                build_rearc_mmap_cache(
                    sanitized_split_dir=sanitized,
                    fold="train",
                    output_dir=root / "wrong-commitment",
                    expected_sanitized_artifact_manifest_sha256="0" * 64,
                    allow_nonproduction_fixture=True,
                )
            with self.assertRaises(ValueError):
                build_rearc_mmap_cache(
                    sanitized_split_dir=sanitized,
                    fold="train",
                    output_dir=root / "production-default",
                    expected_sanitized_artifact_manifest_sha256=sanitized_sha,
                )

            cache = root / "cache"
            result = build_rearc_mmap_cache(
                sanitized_split_dir=sanitized,
                fold="train",
                output_dir=cache,
                expected_sanitized_artifact_manifest_sha256=sanitized_sha,
                allow_nonproduction_fixture=True,
            )
            outside = root / "outside-data.bin"
            (cache / "data.bin").replace(outside)
            try:
                os.symlink(outside, cache / "data.bin")
            except OSError as exc:
                self.skipTest(f"symlink creation is unavailable: {exc}")
            with self.assertRaisesRegex(ValueError, "symlinked artifact"):
                MMapReARCExampleSource(
                    cache,
                    expected_artifact_manifest_sha256=result[
                        "artifact_manifest_sha256"
                    ],
                    expected_sanitized_artifact_manifest_sha256=sanitized_sha,
                    allow_nonproduction_fixture=True,
                )


if __name__ == "__main__":
    unittest.main()
