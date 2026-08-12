"""Seal or deterministically open the 100-episode indexed ARC-TGI reserve."""

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
    authorize_population_oracle_open,
    build_collision_free_reserve_subset,
    build_reserve_seal,
    opened_solution_payload,
    reserve_episode_schedule,
)
from afts_arc.experiment_safety import (  # noqa: E402
    atomic_write_json,
    canonical_json,
    canonical_sha256,
    file_sha256,
)


INDEXED_RESERVE_SEED = "afts-arc-tgi-indexed-reserve-20260812-v1"
INDEXED_RESERVE_EPISODE_COUNT = 100


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


def _indexed_episode(
    *,
    args: argparse.Namespace,
    source: dict[str, object],
    episode_index: int,
) -> dict[str, object]:
    family_id = str(source["family_id"])
    generator_root = args.arc_tgi_root / "Generators" / "ARC-Mini"
    source_path = generator_root / str(source["selected_relative_path"])
    seed_contract = (
        f"{INDEXED_RESERVE_SEED}|{family_id}|{source['family_source_id']}|"
        f"{episode_index}"
    )
    episode_seed = cohort._seed(seed_contract)
    first = cohort._generate_full_record(args.arc_tgi_root, source_path, episode_seed)
    second = cohort._generate_full_record(args.arc_tgi_root, source_path, episode_seed)
    if canonical_json(first) != canonical_json(second):
        raise ValueError(f"indexed ARC-TGI replay differs: {family_id}/{episode_index}")
    blind, oracle, witness = cohort.split_full_record(first)
    blind_sha256 = canonical_sha256(blind)
    identity = {
        "seed_contract": INDEXED_RESERVE_SEED,
        "family_id": family_id,
        "family_source_id": source["family_source_id"],
        "selected_source_sha256": source["selected_source_sha256"],
        "episode_index": episode_index,
        "episode_seed": episode_seed,
        "blind_content_sha256": blind_sha256,
    }
    episode_id = canonical_sha256(identity)
    return {
        "task_id": f"arc_tgi_{family_id}_r{episode_index:02d}_{episode_id[:16]}",
        "episode_id": episode_id,
        "episode_index": episode_index,
        "episode_seed": episode_seed,
        "family_id": family_id,
        "family_source_id": source["family_source_id"],
        "selected_relative_path": source["selected_relative_path"],
        "selected_source_sha256": source["selected_source_sha256"],
        "blind_content_sha256": blind_sha256,
        "oracle_sha256": canonical_sha256(oracle),
        "witness_sha256": canonical_sha256(witness),
        "replay_identical": True,
        "blind": blind,
        "oracle": oracle,
        "witness": witness,
    }


def _episodes(args: argparse.Namespace) -> tuple[dict[str, object], ...]:
    sources = _reserve_sources(args)
    by_family = {str(source["family_id"]): source for source in sources}
    schedule = reserve_episode_schedule(
        tuple(by_family),
        episode_count=INDEXED_RESERVE_EPISODE_COUNT,
    )
    episodes = tuple(
        _indexed_episode(
            args=args,
            source=by_family[family_id],
            episode_index=episode_index,
        )
        for family_id, episode_index in schedule
    )
    if len({episode["episode_id"] for episode in episodes}) != len(episodes):
        raise ValueError("indexed reserve repeats an episode identity")
    return episodes


def _cohort_id(args: argparse.Namespace, episodes: tuple[dict[str, object], ...]) -> str:
    partition = _read_object(args.partition_manifest)
    return canonical_sha256(
        {
            "schema": "afts.arc-tgi-indexed-reserve-cohort/v1",
            "partition_id": partition["partition_id"],
            "arc_tgi_commit": args.arc_tgi_commit,
            "seed_contract": INDEXED_RESERVE_SEED,
            "task_count": len(episodes),
            "episode_ids": [episode["episode_id"] for episode in episodes],
        }
    )


def _source_files(args: argparse.Namespace) -> dict[str, str]:
    return {
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


def _freeze_blind(args: argparse.Namespace) -> dict[str, object]:
    episodes = _episodes(args)
    partition = _read_object(args.partition_manifest)
    cohort_id = _cohort_id(args, episodes)
    challenges, seal = build_reserve_seal(
        episodes=episodes,
        cohort_id=cohort_id,
        partition_id=str(partition["partition_id"]),
        source_files=_source_files(args),
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_root / "reserve100_challenges.json", challenges)
    atomic_write_json(args.output_root / "reserve100_seal_manifest.json", seal)
    return {
        "cohort_id": cohort_id,
        "seal_id": seal["seal_id"],
        "task_count": seal["task_count"],
        "family_count": len({episode["family_id"] for episode in episodes}),
        "query_gold_written": seal["query_gold_written"],
    }


def _filter_blind_collisions(args: argparse.Namespace) -> dict[str, object]:
    if args.output_root.exists():
        raise FileExistsError(f"refusing to replace output root: {args.output_root}")
    challenges_path = args.challenges.resolve()
    seal_path = args.seal.resolve()
    protocol_path = args.protocol.resolve()
    challenges, seal = build_collision_free_reserve_subset(
        challenges=_read_object(challenges_path),
        seal=_read_object(seal_path),
        source_files={
            "parent_challenges": file_sha256(challenges_path),
            "parent_seal": file_sha256(seal_path),
            "protocol": file_sha256(protocol_path),
            "subset_builder": file_sha256(Path(__file__).resolve()),
        },
    )
    args.output_root.mkdir(parents=True)
    atomic_write_json(args.output_root / "eligible_challenges.json", challenges)
    atomic_write_json(args.output_root / "eligible_seal_manifest.json", seal)
    return {
        "cohort_id": seal["cohort_id"],
        "excluded_task_count": seal["eligibility"]["excluded_task_count"],
        "retained_task_count": seal["task_count"],
        "seal_id": seal["seal_id"],
    }


def _open_oracle(args: argparse.Namespace) -> dict[str, object]:
    episodes = _episodes(args)
    seal = _read_object(args.seal)
    candidate_freeze = _read_object(args.candidate_freeze)
    if _cohort_id(args, episodes) != seal["cohort_id"]:
        raise ValueError("regenerated indexed cohort differs from the seal")
    if candidate_freeze["frontier_gate_passed"] is not True:
        raise PermissionError("family-diverse frontier gate did not authorize oracle open")
    authorization = authorize_oracle_open(
        seal=seal,
        candidate_freeze=candidate_freeze,
        minimum_opportunities=args.minimum_opportunities,
    )
    solutions = opened_solution_payload(seal=seal, episodes=episodes)
    args.output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_root / "oracle_authorization.json", authorization)
    atomic_write_json(args.output_root / "reserve100_solutions.json", solutions)
    return {
        "authorization_id": authorization["authorization_id"],
        "task_count": authorization["task_count"],
        "solutions_sha256": file_sha256(
            args.output_root / "reserve100_solutions.json"
        ),
    }


def _open_population_oracle(args: argparse.Namespace) -> dict[str, object]:
    episodes = _episodes(args)
    seal = _read_object(args.seal)
    regenerated_cohort_id = _cohort_id(args, episodes)
    if "parent_cohort_id" in seal:
        if regenerated_cohort_id != seal["parent_cohort_id"]:
            raise ValueError("regenerated indexed cohort differs from parent seal")
        sealed_task_ids = {row["task_id"] for row in seal["tasks"]}
        episodes = tuple(
            episode for episode in episodes if episode["task_id"] in sealed_task_ids
        )
        if {episode["task_id"] for episode in episodes} != sealed_task_ids:
            raise ValueError("regenerated episodes differ from subset seal")
    elif regenerated_cohort_id != seal["cohort_id"]:
        raise ValueError("regenerated indexed cohort differs from the seal")
    authorization = authorize_population_oracle_open(
        seal=seal,
        anchor_freeze=_read_object(args.anchor_freeze),
        recruited_freeze=_read_object(args.recruited_freeze),
        population=_read_object(args.population),
        recruitment_plan=_read_object(args.recruitment_plan),
    )
    solutions = opened_solution_payload(seal=seal, episodes=episodes)
    args.output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        args.output_root / "population_oracle_authorization.json", authorization
    )
    atomic_write_json(args.output_root / "reserve100_solutions.json", solutions)
    return {
        "authorization_id": authorization["authorization_id"],
        "task_count": authorization["task_count"],
        "solutions_sha256": file_sha256(
            args.output_root / "reserve100_solutions.json"
        ),
    }


def _shared(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--arc-tgi-root", type=Path, required=True)
    parser.add_argument("--arc-tgi-commit", required=True)
    parser.add_argument("--partition-manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze-blind")
    _shared(freeze)
    freeze.set_defaults(handler=_freeze_blind)
    filter_blind = commands.add_parser("filter-blind-collisions")
    filter_blind.add_argument("--challenges", type=Path, required=True)
    filter_blind.add_argument("--seal", type=Path, required=True)
    filter_blind.add_argument("--protocol", type=Path, required=True)
    filter_blind.add_argument("--output-root", type=Path, required=True)
    filter_blind.set_defaults(handler=_filter_blind_collisions)
    open_oracle = commands.add_parser("open-oracle")
    _shared(open_oracle)
    open_oracle.add_argument("--seal", type=Path, required=True)
    open_oracle.add_argument("--candidate-freeze", type=Path, required=True)
    open_oracle.add_argument("--minimum-opportunities", type=int, default=5)
    open_oracle.set_defaults(handler=_open_oracle)
    open_population = commands.add_parser("open-population-oracle")
    _shared(open_population)
    open_population.add_argument("--seal", type=Path, required=True)
    open_population.add_argument("--anchor-freeze", type=Path, required=True)
    open_population.add_argument("--recruited-freeze", type=Path, required=True)
    open_population.add_argument("--population", type=Path, required=True)
    open_population.add_argument("--recruitment-plan", type=Path, required=True)
    open_population.set_defaults(handler=_open_population_oracle)
    args = parser.parse_args()
    print(json.dumps(args.handler(args), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
