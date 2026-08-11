from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "build_github_publication.py"
)


def _script_module() -> object:
    spec = importlib.util.spec_from_file_location("build_github_publication", SCRIPT)
    if spec is None or spec.loader is None:
        raise AssertionError("could not load GitHub publication builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "relative",
    [
        Path(
            "results/arc_tgi_arcmini_cohort_v2_20260810/"
            "confirmatory_solutions.json"
        ),
        Path(
            "results/object_program_workspace_arc_tgi_dev_20260811/"
            "candidate_freeze_a.json"
        ),
        Path("results/object_program_workspace_controls_20260811/solutions.json"),
        Path(
            "results/object_graph_rewrite_v2_arc_tgi_dev_20260811/"
            "candidate_freeze_a.json"
        ),
        Path(
            "results/object_graph_rewrite_v2_arc_tgi_reserve_20260811/"
            "sealed_oracle.json"
        ),
        Path("results/object_graph_rewrite_v2_qa_20260811/pytest.stdout.log"),
    ],
)
def test_protected_result_payloads_are_manifest_only(
    tmp_path: Path, relative: Path
) -> None:
    module = _script_module()
    source = tmp_path / "source"
    payload = source / relative
    payload.parent.mkdir(parents=True)
    payload.write_text('{"hidden": true}\n', encoding="utf-8")
    output = tmp_path / "publication"

    manifest = module.build(source, output)

    entry = next(
        item
        for item in manifest["entries"]
        if item["path"] == relative.as_posix()
    )
    assert entry["disposition"] == "manifest_only"
    assert entry["reason"] == "protected_sealed_audit_payload"
    assert not (output / relative).exists()


@pytest.mark.parametrize("name", ["README.md", "artifact_sha256.json", "summary.json"])
def test_object_workspace_public_metadata_is_copied(
    tmp_path: Path, name: str
) -> None:
    module = _script_module()
    source = tmp_path / "source"
    relative = Path("results/object_program_workspace_controls_20260811") / name
    payload = source / relative
    payload.parent.mkdir(parents=True)
    payload.write_text('{"public": true}\n', encoding="utf-8")
    output = tmp_path / "publication"

    manifest = module.build(source, output)

    entry = next(
        item
        for item in manifest["entries"]
        if item["path"] == relative.as_posix()
    )
    assert entry["disposition"] == "copied"
    assert (output / relative).read_bytes() == payload.read_bytes()


@pytest.mark.parametrize(
    "directory",
    [
        "object_graph_rewrite_v2_arc_tgi_dev_20260811",
        "object_graph_rewrite_v2_arc_tgi_reserve_20260811",
        "object_graph_rewrite_v2_qa_20260811",
    ],
)
@pytest.mark.parametrize("name", ["README.md", "artifact_sha256.json", "summary.json"])
def test_object_graph_rewrite_public_metadata_is_copied(
    tmp_path: Path, directory: str, name: str
) -> None:
    module = _script_module()
    source = tmp_path / "source"
    relative = Path("results") / directory / name
    payload = source / relative
    payload.parent.mkdir(parents=True)
    payload.write_text('{"public": true}\n', encoding="utf-8")
    output = tmp_path / "publication"

    manifest = module.build(source, output)

    entry = next(
        item
        for item in manifest["entries"]
        if item["path"] == relative.as_posix()
    )
    assert entry["disposition"] == "copied"
    assert (output / relative).read_bytes() == payload.read_bytes()
