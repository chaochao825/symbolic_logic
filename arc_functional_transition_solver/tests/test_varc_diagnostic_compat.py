from __future__ import annotations

import json
from pathlib import Path

from afts_arc.experiment_safety import canonical_sha256
from afts_arc.varc_diagnostic_compat import (
    build_varc_diagnostic_overlay,
    prepare_varc_diagnostic_alias,
    validate_varc_diagnostic_compatibility,
)


def _blind_root(tmp_path: Path) -> Path:
    blind_root = tmp_path / "blind"
    evaluation = blind_root / "data" / "evaluation"
    augmentation = blind_root / "data" / "eval_color_permute_ttt_9"
    evaluation.mkdir(parents=True)
    augmentation.mkdir()
    tasks = []
    for task_id in ("arc_task_a", "arc_task_b"):
        payload = {"test": [{"input": [[1]], "output": [[1]]}], "train": []}
        (evaluation / f"{task_id}.json").write_text(json.dumps(payload))
        task_augmentation = augmentation / task_id
        task_augmentation.mkdir()
        (task_augmentation / f"{task_id}.json").write_text(json.dumps(payload))
        tasks.append({"task_id": task_id})
    body: dict[str, object] = {
        "provider_visible_test_output": "exact copy of corresponding test input",
        "schema": "afts.varc-query-blind-cohort/v1",
        "source_challenges_sha256": "a" * 64,
        "source_cohort_id": "source",
        "tasks": tasks,
    }
    (blind_root / "manifest.json").write_text(
        json.dumps({"blind_cohort_id": canonical_sha256(body), **body})
    )
    return blind_root


def test_diagnostic_overlay_preserves_model_visible_names_and_bytes(
    tmp_path: Path,
) -> None:
    blind_root = _blind_root(tmp_path)
    runtime_root = tmp_path / "runtime"

    compatibility = build_varc_diagnostic_overlay(
        blind_data_root=blind_root, runtime_data_root=runtime_root
    )

    assert validate_varc_diagnostic_compatibility(compatibility) == (
        "arc_task_a",
        "arc_task_b",
    )
    assert compatibility["compatibility_id"] == canonical_sha256(
        {
            key: value
            for key, value in compatibility.items()
            if key != "compatibility_id"
        }
    )
    assert sorted(
        path.name
        for path in (runtime_root / "data" / "evaluation").glob("*.json")
    ) == ["arc_task_a.json", "arc_task_b.json"]
    assert (
        runtime_root / "data" / "eval_color_permute_ttt_9" / "arc_task_a"
    ).is_symlink()


def test_prepare_diagnostic_alias_archives_previous_link(tmp_path: Path) -> None:
    runtime_root = tmp_path / "runtime"
    build_varc_diagnostic_overlay(
        blind_data_root=_blind_root(tmp_path), runtime_data_root=runtime_root
    )

    first = prepare_varc_diagnostic_alias(
        runtime_data_root=runtime_root,
        task_id="arc_task_a",
        archive_dir=tmp_path / "first_archive",
    )
    second = prepare_varc_diagnostic_alias(
        runtime_data_root=runtime_root,
        task_id="arc_task_b",
        archive_dir=tmp_path / "second_archive",
    )

    diagnostic = runtime_root / "data" / "evaluation" / "arc.json"
    assert first["replaced_previous_alias"] is False
    assert second["replaced_previous_alias"] is True
    assert diagnostic.resolve().name == "arc_task_b.json"
    assert (tmp_path / "second_archive" / "arc.json").resolve().name == (
        "arc_task_a.json"
    )
