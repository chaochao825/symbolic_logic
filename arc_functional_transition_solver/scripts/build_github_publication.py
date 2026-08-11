#!/usr/bin/env python3
"""Build a deterministic, public-safe GitHub staging tree.

Scientific text/JSONL evidence is retained. Large JSON/JSONL files and runtime-lock
snapshots are deterministically gzip-compressed. Binary caches, third-party archives,
deployment ZIPs, bytecode, external repositories, and trash are represented by exact
SHA-256/byte metadata rather than copied into normal Git history.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import shutil
from pathlib import Path


SCHEMA = "afts-github-publication-manifest/v0.1"
MAX_INLINE_TEXT_BYTES = 10 * 1024 * 1024
MAX_COPIED_FILE_BYTES = 50 * 1024 * 1024
TOP_LEVEL_FILES = (
    ".gitignore",
    "README.md",
    "pyproject.toml",
    "external/SOURCES.lock.md",
    "external/ARC-AGI-1/LICENSE",
    "external/ARC-AGI-2/LICENSE",
    "external/arc-agi-benchmarking/LICENSE.md",
    "external/re-arc/LICENSE",
)
TOP_LEVEL_DIRECTORIES = (
    "brief",
    "issues",
    "notes",
    "plan",
    "results",
    "scripts",
    "src",
    "tests",
)
IGNORED_PARTS = frozenset(
    {".git", ".pytest_cache", ".ruff_cache", "__pycache__", "build", "trash"}
)
MANIFEST_ONLY_FILE_ENDINGS = (
    ".bin",
    ".pt",
    ".pyc",
    ".zip",
    ".tar.gz",
    ".tgz",
    ".7z",
    ".rar",
)
MANIFEST_ONLY_RESULT_SUFFIXES = (".printonly.txt",)
COMPRESSIBLE_SUFFIXES = frozenset({".json", ".jsonl"})
PROTECTED_RESULT_PREFIXES = (
    "results/m04a_global_source_v0_1/privileged_split_sealed",
    "results/arc_tgi_arcmini_cohort_v2_20260810",
    "results/object_program_workspace_arc_tgi_dev_20260811",
    "results/object_program_workspace_controls_20260811",
    "results/object_graph_rewrite_v2_arc_tgi_dev_20260811",
    "results/object_graph_rewrite_v2_arc_tgi_reserve_20260811",
    "results/object_graph_rewrite_v2_qa_20260811",
)
PROTECTED_RESULT_PUBLIC_METADATA = frozenset(
    {
        "results/object_program_workspace_arc_tgi_dev_20260811/README.md",
        "results/object_program_workspace_arc_tgi_dev_20260811/artifact_sha256.json",
        "results/object_program_workspace_arc_tgi_dev_20260811/summary.json",
        "results/object_program_workspace_controls_20260811/README.md",
        "results/object_program_workspace_controls_20260811/artifact_sha256.json",
        "results/object_program_workspace_controls_20260811/summary.json",
        "results/object_graph_rewrite_v2_arc_tgi_dev_20260811/README.md",
        "results/object_graph_rewrite_v2_arc_tgi_dev_20260811/artifact_sha256.json",
        "results/object_graph_rewrite_v2_arc_tgi_dev_20260811/summary.json",
        "results/object_graph_rewrite_v2_arc_tgi_reserve_20260811/README.md",
        "results/object_graph_rewrite_v2_arc_tgi_reserve_20260811/artifact_sha256.json",
        "results/object_graph_rewrite_v2_arc_tgi_reserve_20260811/summary.json",
        "results/object_graph_rewrite_v2_qa_20260811/README.md",
        "results/object_graph_rewrite_v2_qa_20260811/artifact_sha256.json",
        "results/object_graph_rewrite_v2_qa_20260811/summary.json",
    }
)
INTERNAL_INFRASTRUCTURE_PATTERNS = (
    re.compile(
        rb"\b(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|"
        rb"172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})\b"
    ),
    re.compile(rb"/home/[A-Za-z0-9._-]+"),
    re.compile(rb"[A-Za-z]:\\Users\\[A-Za-z0-9._-]+", re.IGNORECASE),
)
CREDENTIAL_PATTERNS = (
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(rb"\bghp_[A-Za-z0-9]{36,}\b"),
    re.compile(rb"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
)
INFRASTRUCTURE_FIXTURE_ALLOWLIST = {
    "tests/test_m04a_posix_launcher.py": (b"/home/" + b"test",),
    "src/afts_arc/m04a_python_runtime_lock.py": (b"10.3." + b"9.90",),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def gzip_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as input_handle, target.open("xb") as output_handle:
        with gzip.GzipFile(
            fileobj=output_handle, mode="wb", compresslevel=9, mtime=0
        ) as compressed:
            shutil.copyfileobj(input_handle, compressed, length=1024 * 1024)


def scanned_chunks(path: Path):
    """Yield raw or transparently decompressed chunks for publication scanning."""

    if path.name.lower().endswith(".gz"):
        handle = gzip.open(path, "rb")
    else:
        handle = path.open("rb")
    with handle:
        yield from iter(lambda: handle.read(1024 * 1024), b"")


def contains_pattern(
    path: Path,
    patterns: tuple[re.Pattern[bytes], ...],
    *,
    allowed_values: tuple[bytes, ...] = (),
) -> bool:
    """Scan chunk boundaries without returning or logging matched values."""

    overlap = 255
    previous = b""
    for chunk in scanned_chunks(path):
        window = previous + chunk
        for pattern in patterns:
            for match in pattern.finditer(window):
                if match.group(0) not in allowed_values:
                    return True
        previous = window[-overlap:]
    return False


def reserve_published_path(
    published_paths: dict[str, str], published: Path, source: str
) -> None:
    key = published.as_posix()
    prior = published_paths.get(key)
    if prior is not None:
        raise ValueError(f"published-path collision: {key} from {prior} and {source}")
    published_paths[key] = source


def source_files(source_root: Path) -> tuple[Path, ...]:
    selected: list[Path] = []
    for name in TOP_LEVEL_FILES:
        path = source_root / name
        if path.is_file():
            selected.append(path)
    for name in TOP_LEVEL_DIRECTORIES:
        directory = source_root / name
        if not directory.is_dir():
            continue
        for path in directory.rglob("*"):
            relative = path.relative_to(source_root)
            if any(part in IGNORED_PARTS for part in relative.parts):
                continue
            if path.is_file() or path.is_symlink():
                selected.append(path)
    return tuple(
        sorted(set(selected), key=lambda item: item.relative_to(source_root).as_posix())
    )


def build(source_root: Path, output_root: Path) -> dict[str, object]:
    source_root = source_root.resolve(strict=True)
    output_root = output_root.resolve(strict=False)
    if output_root.exists():
        raise FileExistsError(f"output directory already exists: {output_root}")
    if output_root == source_root or source_root in output_root.parents:
        raise ValueError("output directory must not be inside the source tree")
    output_root.mkdir(parents=True)

    entries: list[dict[str, object]] = []
    copied_bytes = 0
    compressed_source_bytes = 0
    manifest_only_bytes = 0
    published_paths: dict[str, str] = {}
    for source in source_files(source_root):
        relative = source.relative_to(source_root)
        relative_text = relative.as_posix()
        if source.is_symlink():
            entries.append(
                {
                    "path": relative_text,
                    "disposition": "manifest_only",
                    "reason": "symlink_not_published",
                }
            )
            continue
        size = source.stat().st_size
        digest = sha256_file(source)
        common: dict[str, object] = {
            "path": relative_text,
            "bytes": size,
            "sha256": digest,
        }
        suffix = source.suffix.lower()
        lower_name = source.name.lower()
        if (
            relative_text not in PROTECTED_RESULT_PUBLIC_METADATA
            and any(
                relative_text == prefix or relative_text.startswith(prefix + "/")
                for prefix in PROTECTED_RESULT_PREFIXES
            )
        ):
            manifest_only_bytes += size
            entries.append(
                {
                    **common,
                    "disposition": "manifest_only",
                    "reason": "protected_sealed_audit_payload",
                }
            )
            continue
        if lower_name.endswith(MANIFEST_ONLY_FILE_ENDINGS):
            manifest_only_bytes += size
            entries.append(
                {
                    **common,
                    "disposition": "manifest_only",
                    "reason": "binary_cache_archive_or_checkpoint",
                }
            )
            continue
        if contains_pattern(source, CREDENTIAL_PATTERNS):
            raise ValueError(f"credential-like material in publishable file: {relative_text}")
        infrastructure_match = contains_pattern(
            source,
            INTERNAL_INFRASTRUCTURE_PATTERNS,
            allowed_values=INFRASTRUCTURE_FIXTURE_ALLOWLIST.get(relative_text, ()),
        )
        if relative.parts[0] == "results" and (
            relative_text.endswith(MANIFEST_ONLY_RESULT_SUFFIXES)
            or infrastructure_match
        ):
            manifest_only_bytes += size
            entries.append(
                {
                    **common,
                    "disposition": "manifest_only",
                    "reason": "machine_specific_infrastructure_metadata",
                }
            )
            continue
        if infrastructure_match:
            raise ValueError(
                "machine-specific infrastructure metadata in publishable file: "
                f"{relative_text}"
            )
        should_compress = suffix in COMPRESSIBLE_SUFFIXES and (
            size > MAX_INLINE_TEXT_BYTES or source.name == "python-runtime-lock.json"
        )
        if should_compress:
            published = relative.with_name(relative.name + ".gz")
            reserve_published_path(published_paths, published, relative_text)
            gzip_file(source, output_root / published)
            published_size = (output_root / published).stat().st_size
            if published_size > MAX_COPIED_FILE_BYTES:
                raise ValueError(
                    f"compressed publication blob exceeds size bound: {published.as_posix()}"
                )
            compressed_source_bytes += size
            entries.append(
                {
                    **common,
                    "disposition": "gzip",
                    "published_path": published.as_posix(),
                    "published_bytes": published_size,
                    "published_sha256": sha256_file(output_root / published),
                }
            )
            continue
        if size > MAX_COPIED_FILE_BYTES:
            manifest_only_bytes += size
            entries.append(
                {
                    **common,
                    "disposition": "manifest_only",
                    "reason": "file_exceeds_normal_git_publication_bound",
                }
            )
            continue
        reserve_published_path(published_paths, relative, relative_text)
        copy_file(source, output_root / relative)
        copied_bytes += size
        entries.append({**common, "disposition": "copied", "published_path": relative_text})

    disposition_counts: dict[str, int] = {}
    for entry in entries:
        disposition = str(entry["disposition"])
        disposition_counts[disposition] = disposition_counts.get(disposition, 0) + 1
    manifest: dict[str, object] = {
        "schema": SCHEMA,
        "source_root_name": source_root.name,
        "policy": {
            "top_level_files": list(TOP_LEVEL_FILES),
            "top_level_directories": list(TOP_LEVEL_DIRECTORIES),
            "excluded_top_level_directories": ["trash"],
            "external_working_tree_policy": (
                "only SOURCES.lock.md and pinned upstream license texts are published"
            ),
            "max_inline_text_bytes": MAX_INLINE_TEXT_BYTES,
            "max_copied_file_bytes": MAX_COPIED_FILE_BYTES,
            "manifest_only_file_endings": list(MANIFEST_ONLY_FILE_ENDINGS),
            "manifest_only_result_suffixes": list(MANIFEST_ONLY_RESULT_SUFFIXES),
            "protected_result_prefixes": list(PROTECTED_RESULT_PREFIXES),
            "machine_specific_result_records": "manifest_only",
            "deterministic_gzip_mtime": 0,
        },
        "summary": {
            "source_entry_count": len(entries),
            "disposition_counts": disposition_counts,
            "copied_source_bytes": copied_bytes,
            "gzip_source_bytes": compressed_source_bytes,
            "manifest_only_source_bytes": manifest_only_bytes,
        },
        "entries": entries,
    }
    manifest_path = output_root / "PUBLICATION_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root", type=Path)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    manifest = build(args.source_root, args.output_root)
    print(json.dumps(manifest["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
