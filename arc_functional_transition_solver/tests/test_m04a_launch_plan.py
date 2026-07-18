from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import afts_arc.m04a_launch_plan as launch_plan
from afts_arc.m04a_contract import canonical_sha256
from afts_arc.m04a_launch_plan import (
    EXPECTED_INPUT_ARTIFACT_PATHS,
    LAUNCH_PLAN_SCHEMA_VERSION,
    MAX_LAUNCH_PLAN_BYTES,
    LaunchPlanArtifact,
    build_launch_plan,
    canonical_launch_plan_bytes,
    read_launch_plan_artifact,
    validate_launch_plan_payload,
)
from afts_arc.m04a_train_contract import training_config_sha256
from afts_arc.manifest import serialize_json


RUN_ID = "m04a-grid-cmlm-v0.1-seed20260711-r1"
REMOTE_ROOT = "/srv/afts/arc_functional_transition_solver"
RUN_ROOT = f"{REMOTE_ROOT}/runs/{RUN_ID}"


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _validation_commitment(artifacts: dict[str, str]) -> dict[str, object]:
    semantic: dict[str, object] = {
        "schema": "afts-m04a-validation-manifest-commitment/v0.1",
        "outer_artifact_manifest_sha256": artifacts[
            "validation_episode_outer_manifest.json"
        ],
        "jsonl_sha256": artifacts["validation_episode_manifest.jsonl"],
        "summary_id": _digest("validation-summary"),
        "row_count": 25,
    }
    return {**semantic, "commitment_id": canonical_sha256(semantic)}


def _payload() -> dict[str, object]:
    artifacts = {name: _digest(name) for name in EXPECTED_INPUT_ARTIFACT_PATHS}
    return build_launch_plan(
        attempt_nonce=_digest("attempt-nonce"),
        run_id=RUN_ID,
        remote_project_root=REMOTE_ROOT,
        run_root=RUN_ROOT,
        expected_input_artifacts=artifacts,
        runtime_source_fingerprint_sha256=_digest("runtime-fingerprint"),
        test_source_fingerprint_sha256=_digest("test-fingerprint"),
        training_config_sha256=training_config_sha256(),
        validation_manifest_commitment=_validation_commitment(artifacts),
        conda_explicit_sha256=_digest("conda-explicit"),
        ordered_import_roots=[
            f"{REMOTE_ROOT}/src",
            "/opt/conda/envs/mixbit/lib/python3.10/pure-site-packages",
            "/opt/conda/envs/mixbit/lib/python3.10/plat-site-packages",
        ],
    )


class M04aLaunchPlanTests(unittest.TestCase):
    def test_exact_frozen_paths_identity_and_external_reader_roundtrip(self) -> None:
        self.assertEqual(
            EXPECTED_INPUT_ARTIFACT_PATHS,
            frozenset(
                {
                    "reviewed_runtime_source.zip",
                    "reviewed_test_snapshot.zip",
                    "remote_launcher.py",
                    "frozen_contract.md",
                    "sanitized_shard_manifest.json",
                    "data_split_manifest.json",
                    "validation_episode_outer_manifest.json",
                    "validation_episode_manifest.jsonl",
                    "validation_episode_manifest_summary.json",
                    "model_config.json",
                    "parameter_count.json",
                    "python-runtime-lock.json",
                    "ordered_fold_ids.json",
                    "quarantine_parent_ids.json",
                    "schema_config_manifest.json",
                    "seed_policy.json",
                    "short_exact_replay_fixture.json",
                }
            ),
        )
        payload = _payload()
        self.assertEqual(payload["schema"], LAUNCH_PLAN_SCHEMA_VERSION)
        semantic = dict(payload)
        claimed_id = semantic.pop("launch_plan_id")
        self.assertEqual(claimed_id, canonical_sha256(semantic))
        content = canonical_launch_plan_bytes(payload)
        self.assertEqual(content, serialize_json(payload))
        artifact_digest = hashlib.sha256(content).hexdigest()
        self.assertNotEqual(artifact_digest, claimed_id)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "launch_plan.json"
            path.write_bytes(content)
            artifact = read_launch_plan_artifact(
                path, expected_artifact_sha256=artifact_digest
            )
            self.assertIs(type(artifact), LaunchPlanArtifact)
            self.assertEqual(artifact.snapshot, content)
            first = artifact.payload
            first["run_id"] = "mutated-copy"
            first["expected_input_artifacts"]["remote_launcher.py"] = "0" * 64
            self.assertEqual(artifact.payload, payload)

            with self.assertRaisesRegex(ValueError, "external SHA-256"):
                read_launch_plan_artifact(path, expected_artifact_sha256="0" * 64)

    def test_exact_schema_and_frozen_inputs_fail_closed(self) -> None:
        payload = _payload()
        for mutation in ("missing", "extra"):
            changed = copy.deepcopy(payload)
            if mutation == "missing":
                changed.pop("conda_explicit_sha256")
            else:
                changed["unexpected"] = "field"
            with (
                self.subTest(mutation=mutation),
                self.assertRaisesRegex(ValueError, "exact schema"),
            ):
                validate_launch_plan_payload(changed)

        for mutation in ("missing", "extra"):
            changed = copy.deepcopy(payload)
            artifacts = changed["expected_input_artifacts"]
            if mutation == "missing":
                artifacts.pop("remote_launcher.py")
            else:
                artifacts["optimizer_schedule_state.json"] = _digest("extra")
            semantic = dict(changed)
            semantic.pop("launch_plan_id")
            changed["launch_plan_id"] = canonical_sha256(semantic)
            with (
                self.subTest(artifact_mutation=mutation),
                self.assertRaisesRegex(ValueError, "frozen set"),
            ):
                validate_launch_plan_payload(changed)

    def test_validation_commitment_and_config_are_cross_bound(self) -> None:
        payload = _payload()
        cases: list[tuple[str, object, str]] = []

        wrong_id = copy.deepcopy(payload)
        wrong_id["validation_manifest_commitment"]["commitment_id"] = "0" * 64
        cases.append(("commitment-id", wrong_id, "commitment_id mismatch"))

        bool_count = copy.deepcopy(payload)
        commitment = bool_count["validation_manifest_commitment"]
        commitment["row_count"] = True
        semantic_commitment = dict(commitment)
        semantic_commitment.pop("commitment_id")
        commitment["commitment_id"] = canonical_sha256(semantic_commitment)
        cases.append(("bool-count", bool_count, "integer"))

        wrong_outer = copy.deepcopy(payload)
        commitment = wrong_outer["validation_manifest_commitment"]
        commitment["outer_artifact_manifest_sha256"] = _digest("other-outer")
        semantic_commitment = dict(commitment)
        semantic_commitment.pop("commitment_id")
        commitment["commitment_id"] = canonical_sha256(semantic_commitment)
        cases.append(("outer-cross-bind", wrong_outer, "input artifacts"))

        wrong_config = copy.deepcopy(payload)
        wrong_config["training_config_sha256"] = _digest("other-config")
        cases.append(("frozen-config", wrong_config, "frozen config"))

        wrong_nonce = copy.deepcopy(payload)
        wrong_nonce["attempt_nonce"] = "A" * 64
        cases.append(("attempt-nonce", wrong_nonce, "lowercase SHA-256"))

        for name, changed, message in cases:
            semantic = dict(changed)
            semantic.pop("launch_plan_id")
            changed["launch_plan_id"] = canonical_sha256(semantic)
            with (
                self.subTest(name=name),
                self.assertRaisesRegex((TypeError, ValueError), message),
            ):
                validate_launch_plan_payload(changed)

    def test_run_and_import_paths_are_absolute_lexical_posix(self) -> None:
        payload = _payload()
        cases = [
            ("relative-root", "remote_project_root", "relative/root"),
            ("double-root", "remote_project_root", "//srv/afts"),
            ("dot-root", "remote_project_root", "/srv/./afts"),
            ("trailing-root", "remote_project_root", "/srv/afts/"),
            ("wrong-run-root", "run_root", "/srv/afts/runs/not-this-run"),
        ]
        for name, field_name, value in cases:
            changed = copy.deepcopy(payload)
            changed[field_name] = value
            semantic = dict(changed)
            semantic.pop("launch_plan_id")
            changed["launch_plan_id"] = canonical_sha256(semantic)
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_launch_plan_payload(changed)

        bad_run_id = copy.deepcopy(payload)
        bad_run_id["run_id"] = "nested/run"
        bad_run_id["run_root"] = f"{REMOTE_ROOT}/runs/nested/run"
        semantic = dict(bad_run_id)
        semantic.pop("launch_plan_id")
        bad_run_id["launch_plan_id"] = canonical_sha256(semantic)
        with self.assertRaisesRegex(ValueError, "one safe"):
            validate_launch_plan_payload(bad_run_id)

        for roots in (
            ["/one"],
            [f"/root/{index}" for index in range(4)],
            [f"{REMOTE_ROOT}/src", "relative"],
            [f"{REMOTE_ROOT}/src", f"{REMOTE_ROOT}/src"],
            ["/logical/site", f"{REMOTE_ROOT}/src"],
        ):
            changed = copy.deepcopy(payload)
            changed["ordered_import_roots"] = roots
            semantic = dict(changed)
            semantic.pop("launch_plan_id")
            changed["launch_plan_id"] = canonical_sha256(semantic)
            with self.subTest(roots=roots), self.assertRaises(ValueError):
                validate_launch_plan_payload(changed)

        two_roots = copy.deepcopy(payload)
        two_roots["ordered_import_roots"] = [
            f"{REMOTE_ROOT}/src",
            "/opt/conda/envs/mixbit/lib/python3.10/site-packages",
        ]
        semantic = dict(two_roots)
        semantic.pop("launch_plan_id")
        two_roots["launch_plan_id"] = canonical_sha256(semantic)
        self.assertEqual(
            validate_launch_plan_payload(two_roots)["ordered_import_roots"],
            two_roots["ordered_import_roots"],
        )

    def test_reader_rejects_noncanonical_duplicate_nonfinite_and_oversized_json(
        self,
    ) -> None:
        payload = _payload()
        canonical = serialize_json(payload)
        noncanonical = json.dumps(payload, sort_keys=False).encode("utf-8")
        duplicate = b'{"schema":"first","schema":"duplicate"}\n'
        nonfinite = canonical.replace(b'"row_count": 25', b'"row_count": NaN', 1)
        cases = (
            ("noncanonical", noncanonical, "canonically serialized"),
            ("duplicate", duplicate, "duplicate"),
            ("nonfinite", nonfinite, "non-finite"),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, content, expected in cases:
                path = root / f"{name}.json"
                path.write_bytes(content)
                with (
                    self.subTest(name=name),
                    self.assertRaisesRegex(ValueError, expected),
                ):
                    read_launch_plan_artifact(
                        path,
                        expected_artifact_sha256=hashlib.sha256(content).hexdigest(),
                    )

            oversized = root / "oversized.json"
            oversized.write_bytes(b"{" + b" " * MAX_LAUNCH_PLAN_BYTES + b"}")
            with self.assertRaisesRegex(ValueError, "bounded size"):
                read_launch_plan_artifact(
                    oversized,
                    expected_artifact_sha256=hashlib.sha256(
                        oversized.read_bytes()
                    ).hexdigest(),
                )
            with self.assertRaisesRegex(ValueError, "regular file"):
                read_launch_plan_artifact(
                    root, expected_artifact_sha256=_digest("directory")
                )

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks are unavailable")
    def test_reader_rejects_symlink_without_following_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.json"
            content = serialize_json(_payload())
            target.write_bytes(content)
            link = root / "launch_plan.json"
            try:
                os.symlink(target, link)
            except OSError as exc:
                self.skipTest(f"symlink creation is unavailable: {exc}")
            with self.assertRaisesRegex((OSError, ValueError), "symlink|reparse"):
                read_launch_plan_artifact(
                    link,
                    expected_artifact_sha256=hashlib.sha256(content).hexdigest(),
                )

    @unittest.skipUnless(os.name == "nt", "Windows handle race regression")
    def test_windows_reader_rejects_reparse_swap_between_traversal_and_leaf_open(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            content = serialize_json(_payload())
            path = root / "launch_plan.json"
            path.write_bytes(content)
            moved = root / "original.json"
            alternate = root / "same-bytes-alternate.json"
            alternate.write_bytes(content)
            original_open = launch_plan._create_windows_component_handle
            swapped = False

            def swap_before_leaf(
                create_file,
                component_path,
                *,
                desired_access,
                is_leaf,
            ):
                nonlocal swapped
                if is_leaf and not swapped:
                    os.replace(component_path, moved)
                    try:
                        os.symlink(alternate, component_path)
                    except OSError as exc:
                        self.skipTest(f"symlink creation is unavailable: {exc}")
                    swapped = True
                return original_open(
                    create_file,
                    component_path,
                    desired_access=desired_access,
                    is_leaf=is_leaf,
                )

            with (
                mock.patch.object(
                    launch_plan,
                    "_create_windows_component_handle",
                    side_effect=swap_before_leaf,
                ),
                self.assertRaisesRegex(ValueError, "symlink|reparse"),
            ):
                read_launch_plan_artifact(
                    path,
                    expected_artifact_sha256=hashlib.sha256(content).hexdigest(),
                )
            self.assertTrue(swapped)


if __name__ == "__main__":
    unittest.main()
