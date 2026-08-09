from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from scripts.build_arc2_relational_mask_confirmation import (
    blind_structural_signature,
)


def _payload(first: int, second: int, *, sentinel: int) -> dict[str, object]:
    return {
        "train": [
            {
                "input": [[0, first], [first, 0]],
                "output": [[0, second], [second, 0]],
            }
        ],
        "test": [
            {
                "input": [[0, first], [first, 0]],
                "output": [[sentinel]],
            }
        ],
    }


def test_blind_structural_signature_is_color_permutation_invariant() -> None:
    assert blind_structural_signature(_payload(1, 2, sentinel=1)) == (
        blind_structural_signature(_payload(7, 4, sentinel=9))
    )


def test_blind_structural_signature_ignores_query_output() -> None:
    assert blind_structural_signature(_payload(1, 2, sentinel=1)) == (
        blind_structural_signature(_payload(1, 2, sentinel=8))
    )


def test_builder_entrypoint_imports_without_project_pythonpath() -> None:
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        (
            sys.executable,
            str(project_root / "scripts" / "build_arc2_relational_mask_confirmation.py"),
            "--help",
        ),
        cwd=project_root.parent,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
