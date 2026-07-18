"""Earliest package-source anchor, loaded before any solver submodule."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable


def named_bytes_fingerprint(items: Iterable[tuple[str, bytes]]) -> str:
    digest = hashlib.sha256()
    for name, content in sorted(items, key=lambda item: item[0]):
        encoded_name = name.replace("\\", "/").encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def runtime_materials() -> dict[str, bytes]:
    package_root = Path(__file__).resolve().parent
    materials = {
        f"afts_arc/{path.relative_to(package_root).as_posix()}": path.read_bytes()
        for path in package_root.rglob("*.py")
    }
    source_project = package_root.parent.parent
    pyproject = source_project / "pyproject.toml"
    if pyproject.exists():
        materials["pyproject.toml"] = pyproject.read_bytes()
    for launcher_name in ("afts_arc_evidence.py", "afts_arc_m04a.py"):
        launcher = source_project / "scripts" / launcher_name
        if launcher.exists():
            materials[f"scripts/{launcher_name}"] = launcher.read_bytes()
    return materials


# Package __init__ imports this module before importing candidate, grid, task,
# scoring, baselines, manifest, or cli. Any later disk mutation is therefore
# detectable even when a programmatic caller preloads a dependency before cli.
BOOTSTRAP_RUNTIME_SOURCE_FINGERPRINT = named_bytes_fingerprint(
    runtime_materials().items()
)
