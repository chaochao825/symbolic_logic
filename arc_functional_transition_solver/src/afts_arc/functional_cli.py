"""Ordinary (non-evidence) CLI for the hybrid functional ARC solver."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .blind import BlindTask
from .hybrid import (
    CodeModelProvider,
    DiffLogicHardProvider,
    DslProgramProvider,
    FunctionalRouterConfig,
    FunctionalRouterSolver,
    MaskedDiffusionProvider,
    SparseCAProvider,
)
from .search import SearchConfig
from .task import load_task


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Route an ARC task through DSL, sparse CA/D4/bgpad, optional external "
            "sources, demo-exact/MDL/hard verification, and residual repair."
        )
    )
    parser.add_argument("task", help="official-format ARC JSON task")
    parser.add_argument("--output", help="optional JSON report path")
    parser.add_argument("--max-depth", type=int, default=2, choices=(1, 2, 3))
    parser.add_argument("--beam-width", type=int, default=64)
    parser.add_argument("--max-instruction-options", type=int, default=64)
    parser.add_argument("--max-exact-programs", type=int, default=128)
    parser.add_argument("--max-selected", type=int, default=2)
    parser.add_argument("--max-repair-parents", type=int, default=8)
    parser.add_argument("--minimum-repair-agreement", type=float, default=0.5)
    parser.add_argument(
        "--no-ca",
        action="store_true",
        help="disable the optional repository-root NumPy CA backend",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    repository_root = Path(__file__).resolve().parents[3]
    root_source = repository_root / "src"
    if (root_source / "arc_ca.py").is_file() and str(root_source) not in sys.path:
        sys.path.insert(0, str(root_source))
    args = build_parser().parse_args(argv)
    source_task = load_task(args.task)
    blind = BlindTask.from_task(source_task)
    providers = [
        DslProgramProvider(
            SearchConfig(
                max_depth=args.max_depth,
                beam_width=args.beam_width,
                max_instruction_options=args.max_instruction_options,
                max_exact_programs=args.max_exact_programs,
            )
        )
    ]
    if not args.no_ca:
        providers.append(SparseCAProvider())
    providers.extend(
        (
            CodeModelProvider(),
            DiffLogicHardProvider(),
            MaskedDiffusionProvider(),
        )
    )
    solver = FunctionalRouterSolver(
        providers=providers,
        config=FunctionalRouterConfig(
            max_selected_hypotheses=args.max_selected,
            max_repair_parents=args.max_repair_parents,
            minimum_repair_agreement=args.minimum_repair_agreement,
        ),
    )
    report = solver.solve(blind).to_json_dict(task_id=source_task.task_id)
    serialized = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        destination = Path(args.output).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
