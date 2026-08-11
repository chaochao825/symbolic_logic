"""Materialize the frozen ARC-TGI reserve without prematurely opening oracle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = str((PROJECT_ROOT / "src").resolve())
SCRIPT_ROOT = str(Path(__file__).resolve().parent)
for path in (SOURCE_ROOT, SCRIPT_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

import afts_arc_tgi_cohort as cohort  # noqa: E402
from afts_arc.arc_tgi_reserve import (  # noqa: E402
    authorize_oracle_open,
    build_reserve_seal,
    opened_solution_payload,
)
from afts_arc.experiment_safety import (  # noqa: E402
    atomic_write_json,
    canonical_sha256,
    file_sha256,
)


def _read_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _reserve_sources(args: argparse.Namespace) -> tuple[dict[str, object], ...]:
    revision = cohort._git_revision(args.arc_tgi_root)
    if revision != args.arc_tgi_commit:
        raise ValueError("ARC-TGI source revision differs from frozen commit")
    generator_root = args.arc_tgi_root / "Generators" / "ARC-Mini"
    sources = cohort._generator_sources(generator_root)
    partitions = cohort.partition_generator_sources(sources)
    manifest = _read_object(args.partition_manifest)
    manifest_content = {
        key: value for key, value in manifest.items() if key != "partition_id"
    }
    if manifest["partition_id"] != canonical_sha256(manifest_content):
        raise ValueError("partition manifest ID differs")
    if manifest["partitions"] != partitions:
        raise ValueError("reconstructed partition differs from frozen manifest")
    reserve = tuple(dict(item) for item in partitions["reserve"])
    if any(item["manually_exposed_before_partition"] is not False for item in reserve):
        raise ValueError("manually exposed family entered the reserve")
    return reserve


def _episodes(args: argparse.Namespace) -> tuple[dict[str, object], ...]:
    generator_root = args.arc_tgi_root / "Generators" / "ARC-Mini"
    return tuple(
        cohort._episode(args.arc_tgi_root, generator_root, source)
        for source in _reserve_sources(args)
    )


def _freeze_blind(args: argparse.Namespace) -> dict[str, object]:
    episodes = _episodes(args)
    partition = _read_object(args.partition_manifest)
    source_files = {
        str(Path(__file__).resolve().relative_to(PROJECT_ROOT)).replace("\\", "/"):
        file_sha256(Path(__file__).resolve()),
        "scripts/afts_arc_tgi_cohort.py": file_sha256(
            PROJECT_ROOT / "scripts" / "afts_arc_tgi_cohort.py"
        ),
        "src/afts_arc/arc_tgi_reserve.py": file_sha256(
            PROJECT_ROOT / "src" / "afts_arc" / "arc_tgi_reserve.py"
        ),
        str(args.protocol.resolve().relative_to(PROJECT_ROOT.resolve())).replace(
            "\\", "/"
        ): file_sha256(args.protocol),
    }
    challenges, seal = build_reserve_seal(
        episodes=episodes,
        cohort_id=args.cohort_id,
        partition_id=partition["partition_id"],
        source_files=source_files,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_root / "reserve_challenges.json", challenges)
    atomic_write_json(args.output_root / "reserve_seal_manifest.json", seal)
    return {
        "seal_id": seal["seal_id"],
        "task_count": seal["task_count"],
        "query_gold_written": seal["query_gold_written"],
    }


def _open_oracle(args: argparse.Namespace) -> dict[str, object]:
    seal = _read_object(args.seal)
    candidate_freeze = _read_object(args.candidate_freeze)
    authorization = authorize_oracle_open(
        seal=seal,
        candidate_freeze=candidate_freeze,
        minimum_opportunities=args.minimum_opportunities,
    )
    episodes = _episodes(args)
    solutions = opened_solution_payload(seal=seal, episodes=episodes)
    args.output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_root / "oracle_authorization.json", authorization)
    atomic_write_json(args.output_root / "reserve_solutions.json", solutions)
    return {
        "authorization_id": authorization["authorization_id"],
        "task_count": authorization["task_count"],
        "solutions_sha256": file_sha256(
            args.output_root / "reserve_solutions.json"
        ),
    }


def _shared(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--arc-tgi-root", type=Path, required=True)
    parser.add_argument("--arc-tgi-commit", required=True)
    parser.add_argument("--partition-manifest", type=Path, required=True)
    parser.add_argument("--cohort-id", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze-blind")
    _shared(freeze)
    freeze.set_defaults(handler=_freeze_blind)
    open_oracle = commands.add_parser("open-oracle")
    _shared(open_oracle)
    open_oracle.add_argument("--seal", type=Path, required=True)
    open_oracle.add_argument("--candidate-freeze", type=Path, required=True)
    open_oracle.add_argument("--minimum-opportunities", type=int, default=2)
    open_oracle.set_defaults(handler=_open_oracle)
    args = parser.parse_args()
    print(json.dumps(args.handler(args), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
