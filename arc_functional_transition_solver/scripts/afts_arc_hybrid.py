"""Repository launcher that exposes both the AFTS and root CA source trees."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    repository_root = project_root.parent
    for source_root in (repository_root / "src", project_root / "src"):
        resolved = str(source_root.resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)
    from afts_arc.functional_cli import main as functional_main

    return functional_main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
