from __future__ import annotations

import json
from pathlib import Path

from afts_arc.experiment_safety import canonical_sha256, file_sha256


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = (
    PROJECT_ROOT / "results" / "residual_compiled_metareasoning_p1_20260726"
)
REGISTRY_PATH = RESULT_ROOT / "artifact_registry.json"


def test_registered_historical_artifacts_remain_byte_exact() -> None:
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    body = dict(registry)
    declared_registry_id = body.pop("registry_id")
    assert declared_registry_id == canonical_sha256(body)

    entries = (
        *registry["canonical_artifacts"],
        *registry["diagnostic_artifacts"],
    )
    for entry in entries:
        artifact = RESULT_ROOT / entry["artifact"]
        assert artifact.is_file(), entry["artifact"]
        assert file_sha256(artifact) == entry["file_sha256"]


def test_content_addressed_profile_registry_is_closed() -> None:
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    for entry in registry["profile_registry"]:
        artifact = RESULT_ROOT / entry["path"]
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        assert payload["profile_id"] == entry["profile_id"]
        assert artifact.stem == entry["profile_id"]
        assert file_sha256(artifact) == entry["file_sha256"]
