from __future__ import annotations

import copy
import hashlib
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from afts_arc import m04a_evidence as evidence
from afts_arc.m04a_contract import canonical_sha256
import afts_arc.m04a_validation_manifest as validation_manifest
from afts_arc.manifest import publish_evidence_bundle, serialize_json, serialize_jsonl
from afts_arc.m04a_data import (
    ARC2Parent,
    M04AExample,
    M04ATrainingData,
    build_validation_episodes,
)
from afts_arc.m04a_validation_manifest import (
    VALIDATION_EPISODE_JSONL,
    VALIDATION_EPISODE_SUMMARY_JSON,
    build_validation_manifest_commitment,
    build_validation_episode_artifacts,
    publish_validation_episode_manifest,
    read_validation_episode_manifest,
    require_externally_committed_validation_manifest,
    validate_validation_manifest_commitment,
)


class _TinyReARC:
    @property
    def parent_ids(self) -> tuple[str, ...]:
        return ("bbbbbbbb",)

    def example(self, parent_id: str, example_index: int) -> M04AExample:
        if parent_id != "bbbbbbbb" or not 0 <= example_index < 1000:
            raise KeyError((parent_id, example_index))
        color = example_index % 10
        return M04AExample.create([[color, 0]], [[0], [color]])

    def semantic_parent_id(self, parent_id: str) -> str:
        if parent_id != "bbbbbbbb":
            raise KeyError(parent_id)
        return "bbbbbbbb"


def _episodes():
    parent = ARC2Parent(
        parent_id="aaaaaaaa",
        train=(
            M04AExample.create([[0, 1]], [[1, 0]]),
            M04AExample.create([[2], [0]], [[0, 2]]),
        ),
        test=(M04AExample.create([[3, 0]], [[3, 3]]),),
    )
    data = M04ATrainingData(
        arc2_parent_ids=(parent.parent_id,),
        arc2_parents={parent.parent_id: parent},
        rearc_parent_ids=("bbbbbbbb",),
        rearc=_TinyReARC(),
    )
    return build_validation_episodes(data)


def _artifact_sha(root: Path) -> str:
    return hashlib.sha256((root / "artifact_manifest.json").read_bytes()).hexdigest()


def _publish_forged_rows(
    root: Path,
    rows: list[dict[str, object]],
    *,
    jsonl_override: bytes | None = None,
    summary_override: dict[str, object] | None = None,
) -> str:
    jsonl = serialize_jsonl(rows) if jsonl_override is None else jsonl_override
    summary = (
        validation_manifest._make_summary(rows, jsonl)
        if summary_override is None
        else summary_override
    )
    publish_evidence_bundle(
        root,
        artifacts={
            VALIDATION_EPISODE_JSONL: jsonl,
            VALIDATION_EPISODE_SUMMARY_JSON: serialize_json(summary),
        },
        run_id=str(summary["summary_id"]),
    )
    return _artifact_sha(root)


class M04AValidationManifestTests(unittest.TestCase):
    def test_publish_read_roundtrip_and_distinct_rearc_target_groups(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "validation"
            published = publish_validation_episode_manifest(root, _episodes())
            expected_sha = _artifact_sha(root)
            loaded = read_validation_episode_manifest(
                root, expected_artifact_manifest_sha256=expected_sha
            )
            require_externally_committed_validation_manifest(
                loaded,
                expected_artifact_manifest_sha256=expected_sha,
                expected_jsonl_sha256=loaded.summary["jsonl_sha256"],
            )
            commitment = build_validation_manifest_commitment(
                loaded,
                expected_artifact_manifest_sha256=expected_sha,
                expected_jsonl_sha256=loaded.summary["jsonl_sha256"],
            )
            commitment_payload = validate_validation_manifest_commitment(commitment)
            with self.assertRaisesRegex(TypeError, "reader-produced"):
                validate_validation_manifest_commitment(  # type: ignore[arg-type]
                    commitment.to_json_dict()
                )
            commitment_semantic = dict(commitment_payload)
            commitment_semantic.pop("commitment_id")
            self.assertEqual(
                commitment_payload["commitment_id"],
                canonical_sha256(commitment_semantic),
            )
            self.assertEqual(
                evidence._validate_preflight_validation_commitment(
                    commitment_payload,
                    expected_outer_manifest_sha256=expected_sha,
                    expected_jsonl_sha256=loaded.summary["jsonl_sha256"],
                    expected_summary_id=loaded.summary["summary_id"],
                    expected_row_count=len(loaded.rows),
                ),
                commitment_payload,
            )
            self.assertEqual(commitment_payload["row_count"], len(loaded.rows))
            self.assertEqual(
                commitment_payload["summary_id"], loaded.summary["summary_id"]
            )
            self.assertEqual(published, loaded)
            self.assertEqual(len(loaded.episodes), 16)
            self.assertEqual(loaded.summary["row_count"], 16)
            self.assertEqual(
                loaded.summary["jsonl_sha256"],
                hashlib.sha256(loaded.jsonl_bytes).hexdigest(),
            )
            self.assertEqual(loaded.summary["jsonl_bytes"], len(loaded.jsonl_bytes))
            self.assertEqual(
                loaded.summary["ordered_episode_sha256"],
                [row["episode_sha256"] for row in loaded.rows],
            )
            for ordinal, row in enumerate(loaded.rows):
                self.assertEqual(row["row_ordinal"], ordinal)
                width = len(row["target_output"][0])
                self.assertEqual(
                    row["masked_coordinates"],
                    [
                        [linear // width, linear % width]
                        for linear in row["masked_linear_indices"]
                    ],
                )
            rearc = [row for row in loaded.rows if row["source"] == "rearc"]
            self.assertEqual(len({row["target_descriptor"] for row in rearc}), 4)
            self.assertEqual(len({row["target_group_id"] for row in rearc}), 4)
            with self.assertRaises(FileExistsError):
                publish_validation_episode_manifest(root, _episodes())

    def test_commitment_typed_and_pure_validators_reject_resealed_type_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "validation"
            publish_validation_episode_manifest(root, _episodes())
            expected_sha = _artifact_sha(root)
            loaded = read_validation_episode_manifest(
                root, expected_artifact_manifest_sha256=expected_sha
            )
            commitment = build_validation_manifest_commitment(
                loaded,
                expected_artifact_manifest_sha256=expected_sha,
                expected_jsonl_sha256=loaded.summary["jsonl_sha256"],
            )

            wrong_id = replace(commitment, commitment_id="0" * 64)
            with self.assertRaisesRegex(ValueError, "commitment_id"):
                validate_validation_manifest_commitment(wrong_id)

            bool_semantic = commitment.to_json_dict()
            bool_semantic["row_count"] = True
            bool_semantic.pop("commitment_id")
            bool_count = replace(
                commitment,
                row_count=True,
                commitment_id=canonical_sha256(bool_semantic),
            )
            with self.assertRaisesRegex(TypeError, "row_count"):
                validate_validation_manifest_commitment(bool_count)

            pure_payload = bool_count.to_json_dict()
            with self.assertRaisesRegex(TypeError, "row_count"):
                evidence._validate_preflight_validation_commitment(
                    pure_payload,
                    expected_outer_manifest_sha256=expected_sha,
                    expected_jsonl_sha256=loaded.summary["jsonl_sha256"],
                    expected_summary_id=loaded.summary["summary_id"],
                    expected_row_count=len(loaded.rows),
                )

    def test_in_memory_manifest_cannot_enter_production_validation(self) -> None:
        materialized = build_validation_episode_artifacts(_episodes())
        with self.assertRaisesRegex(ValueError, "reader-attested"):
            require_externally_committed_validation_manifest(
                materialized,
                expected_artifact_manifest_sha256="0" * 64,
                expected_jsonl_sha256=materialized.summary["jsonl_sha256"],
            )
        with self.assertRaisesRegex(ValueError, "reader-attested"):
            build_validation_manifest_commitment(
                materialized,
                expected_artifact_manifest_sha256="0" * 64,
                expected_jsonl_sha256=materialized.summary["jsonl_sha256"],
            )

    def test_external_commitment_tamper_and_extra_file_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "validation"
            publish_validation_episode_manifest(root, _episodes())
            expected_sha = _artifact_sha(root)
            with self.assertRaisesRegex(ValueError, "external commitment"):
                read_validation_episode_manifest(
                    root,
                    expected_artifact_manifest_sha256="f" * 64,
                )
            jsonl = root / VALIDATION_EPISODE_JSONL
            jsonl.write_bytes(jsonl.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "metadata mismatch"):
                read_validation_episode_manifest(
                    root, expected_artifact_manifest_sha256=expected_sha
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "validation"
            publish_validation_episode_manifest(root, _episodes())
            expected_sha = _artifact_sha(root)
            (root / "extra.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "closed-world"):
                read_validation_episode_manifest(
                    root, expected_artifact_manifest_sha256=expected_sha
                )

    def test_root_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "validation"
            alias = base / "validation-link"
            publish_validation_episode_manifest(root, _episodes())
            try:
                os.symlink(root, alias, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"directory symlinks unavailable: {exc}")
            with self.assertRaisesRegex(ValueError, "symlink"):
                read_validation_episode_manifest(
                    alias, expected_artifact_manifest_sha256=_artifact_sha(root)
                )

    def test_coordinate_order_and_group_forgery_are_rejected_by_reader(self) -> None:
        materialized = build_validation_episode_artifacts(_episodes())

        with tempfile.TemporaryDirectory() as directory:
            rows = copy.deepcopy(list(materialized.rows))
            rows[0]["masked_coordinates"][0] = [29, 29]
            root = Path(directory) / "bad-coordinate"
            expected_sha = _publish_forged_rows(root, rows)
            with self.assertRaisesRegex(ValueError, "coordinates"):
                read_validation_episode_manifest(
                    root, expected_artifact_manifest_sha256=expected_sha
                )

        with tempfile.TemporaryDirectory() as directory:
            rows = copy.deepcopy(list(materialized.rows))
            rows[0], rows[1] = rows[1], rows[0]
            for ordinal, row in enumerate(rows):
                row["row_ordinal"] = ordinal
            root = Path(directory) / "bad-order"
            expected_sha = _publish_forged_rows(root, rows)
            with self.assertRaisesRegex(ValueError, "canonical order"):
                read_validation_episode_manifest(
                    root, expected_artifact_manifest_sha256=expected_sha
                )

        with tempfile.TemporaryDirectory() as directory:
            original = copy.deepcopy(list(materialized.rows))
            rows = [original[0], *original[4:8], *original[1:4], *original[8:]]
            for ordinal, row in enumerate(rows):
                row["row_ordinal"] = ordinal
            root = Path(directory) / "split-group"
            expected_sha = _publish_forged_rows(root, rows)
            with self.assertRaisesRegex(ValueError, "contiguous"):
                read_validation_episode_manifest(
                    root, expected_artifact_manifest_sha256=expected_sha
                )

    def test_resealed_structural_boolean_integers_are_rejected_by_reader(self) -> None:
        materialized = build_validation_episode_artifacts(_episodes())

        def forge_row_ordinal(rows: list[dict[str, object]]) -> None:
            rows[0]["row_ordinal"] = False

        def forge_masked_index(rows: list[dict[str, object]]) -> None:
            masked = rows[0]["masked_linear_indices"]
            assert isinstance(masked, list)
            masked[0] = bool(masked[0])

        def forge_color_permutation(rows: list[dict[str, object]]) -> None:
            permutation = rows[0]["old_to_new_color_permutation"]
            assert isinstance(permutation, list)
            one_index = permutation.index(1)
            permutation[one_index] = True

        for label, forge in (
            ("row ordinal", forge_row_ordinal),
            ("masked index", forge_masked_index),
            ("color permutation", forge_color_permutation),
        ):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                rows = copy.deepcopy(list(materialized.rows))
                forge(rows)
                root = Path(directory) / "structural-type-drift"
                expected_sha = _publish_forged_rows(root, rows)
                with self.assertRaisesRegex(TypeError, "must be an integer"):
                    read_validation_episode_manifest(
                        root, expected_artifact_manifest_sha256=expected_sha
                    )

    def test_duplicate_keys_and_nonfinite_json_are_rejected(self) -> None:
        materialized = build_validation_episode_artifacts(_episodes())
        original_lines = materialized.jsonl_bytes.splitlines(keepends=True)
        corruptions = {
            "duplicate JSON key": (
                b'{"row_ordinal": 0,' + original_lines[0][1:]
                + b"".join(original_lines[1:])
            ),
            "non-finite JSON constant": materialized.jsonl_bytes.replace(
                b'"row_ordinal": 0', b'"row_ordinal": NaN', 1
            ),
        }
        for expected_error, jsonl in corruptions.items():
            with self.subTest(expected_error=expected_error):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory) / "invalid-json"
                    expected_sha = _publish_forged_rows(
                        root,
                        copy.deepcopy(list(materialized.rows)),
                        jsonl_override=jsonl,
                        summary_override=copy.deepcopy(materialized.summary),
                    )
                    with self.assertRaisesRegex(ValueError, expected_error):
                        read_validation_episode_manifest(
                            root,
                            expected_artifact_manifest_sha256=expected_sha,
                        )

    def test_isolated_import_does_not_load_torch(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        code = (
            "import sys;"
            f"sys.path.insert(0, {str(project_root / 'src')!r});"
            "import afts_arc.m04a_validation_manifest;"
            "assert not any(name == 'torch' or name.startswith('torch.') "
            "for name in sys.modules)"
        )
        completed = subprocess.run(
            [sys.executable, "-I", "-B", "-S", "-c", code],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
