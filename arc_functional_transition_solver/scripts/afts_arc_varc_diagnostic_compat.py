"""Build and operate VARC's post-candidate diagnostic compatibility layer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = str((PROJECT_ROOT / "src").resolve())
if SOURCE_ROOT not in sys.path:
    sys.path.insert(0, SOURCE_ROOT)

from afts_arc.experiment_safety import atomic_write_json  # noqa: E402
from afts_arc.varc_diagnostic_compat import (  # noqa: E402
    build_varc_diagnostic_overlay,
    prepare_varc_diagnostic_alias,
)


def _write_new(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to replace artifact: {path}")
    atomic_write_json(path, payload)


def _build(arguments: argparse.Namespace) -> None:
    output = Path(arguments.output).resolve()
    payload = build_varc_diagnostic_overlay(
        blind_data_root=Path(arguments.blind_data_root),
        runtime_data_root=Path(arguments.runtime_data_root),
    )
    _write_new(output, payload)
    print(json.dumps({"compatibility_id": payload["compatibility_id"]}))


def _prepare_task(arguments: argparse.Namespace) -> None:
    output = Path(arguments.output).resolve()
    payload = prepare_varc_diagnostic_alias(
        runtime_data_root=Path(arguments.runtime_data_root),
        task_id=arguments.task_id,
        archive_dir=Path(arguments.archive_dir),
    )
    _write_new(output, payload)
    print(json.dumps({"record_id": payload["record_id"]}))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--blind-data-root", required=True)
    build.add_argument("--runtime-data-root", required=True)
    build.add_argument("--output", required=True)
    build.set_defaults(handler=_build)
    prepare = subparsers.add_parser("prepare-task")
    prepare.add_argument("--runtime-data-root", required=True)
    prepare.add_argument("--task-id", required=True)
    prepare.add_argument("--archive-dir", required=True)
    prepare.add_argument("--output", required=True)
    prepare.set_defaults(handler=_prepare_task)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    arguments.handler(arguments)


if __name__ == "__main__":
    main()
