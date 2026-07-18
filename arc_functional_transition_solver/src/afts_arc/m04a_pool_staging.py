"""Streaming, incomplete-by-construction staging for large M04a pair evidence."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, BinaryIO

from .m04a_evidence import validate_pair_cost_row
from .manifest import serialize_json, serialize_jsonl


PAIR_MATERIALS_SCHEMA_VERSION = "afts-m04a-pair-materials-staging/v0.1"
_STREAM_FILES = (
    "lane_traces.jsonl",
    "candidate_rows.jsonl",
    "encoder_forward_ledger.jsonl",
    "batch_forward_ledger.jsonl",
)


class M04APoolPairStagingSink:
    """Write each pair once without ever claiming a complete pool artifact."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        expected_pairs: Sequence[tuple[str, int]],
    ) -> None:
        target = Path(output_dir).expanduser().resolve()
        if target.exists():
            raise FileExistsError(f"refusing to overwrite pair staging: {target}")
        pairs = tuple(expected_pairs)
        if not pairs or any(
            not isinstance(task_id, str)
            or not task_id
            or type(test_index) is not int
            or test_index < 0
            for task_id, test_index in pairs
        ):
            raise ValueError("expected_pairs must be non-empty task/test coordinates")
        if len(pairs) != len(set(pairs)):
            raise ValueError("expected pair coordinates must be unique")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.mkdir(exist_ok=False)
        self._root = target
        self._expected_pairs = pairs
        self._next_pair = 0
        self._closed = False
        self._aborted = False
        self._handles: dict[str, BinaryIO] = {
            name: (target / name).open("xb") for name in _STREAM_FILES
        }
        self._digests = {name: hashlib.sha256() for name in _STREAM_FILES}
        self._bytes = {name: 0 for name in _STREAM_FILES}
        self._rows = {name: 0 for name in _STREAM_FILES}

    @property
    def root(self) -> Path:
        return self._root

    def append_pair(
        self,
        *,
        blind_task_id: str,
        test_index: int,
        materials: Mapping[str, bytes],
    ) -> Mapping[str, object]:
        if self._closed or self._aborted:
            raise RuntimeError("pair staging is already closed")
        if self._next_pair >= len(self._expected_pairs):
            raise ValueError("pair staging received more pairs than declared")
        coordinate = (blind_task_id, test_index)
        if coordinate != self._expected_pairs[self._next_pair]:
            raise ValueError("pair staging order differs from the frozen sidecar")
        if set(materials) != set(_STREAM_FILES):
            raise ValueError("pair materials do not have the exact streaming file set")
        chunk_metadata: dict[str, dict[str, object]] = {}
        for name in _STREAM_FILES:
            content = materials[name]
            if not isinstance(content, bytes):
                raise TypeError("pair material chunks must be bytes")
            if content and not content.endswith(b"\n"):
                raise ValueError("non-empty JSONL pair material must end with newline")
            self._handles[name].write(content)
            self._handles[name].flush()
            self._digests[name].update(content)
            self._bytes[name] += len(content)
            row_count = content.count(b"\n")
            self._rows[name] += row_count
            chunk_metadata[name] = {
                "bytes": len(content),
                "rows": row_count,
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        receipt: dict[str, object] = {
            "schema": PAIR_MATERIALS_SCHEMA_VERSION,
            "pair_index": self._next_pair,
            "blind_task_id": blind_task_id,
            "test_index": test_index,
            "chunks": chunk_metadata,
        }
        receipt["receipt_id"] = hashlib.sha256(serialize_json(receipt)).hexdigest()
        self._next_pair += 1
        return receipt

    def _close_streams(self) -> None:
        if self._closed:
            return
        for handle in self._handles.values():
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
        self._closed = True

    def finalize(
        self, *, pair_cost_rows: Sequence[Mapping[str, Any]]
    ) -> Mapping[str, object]:
        if self._aborted:
            raise RuntimeError("aborted pair staging cannot be finalized")
        if self._next_pair != len(self._expected_pairs):
            raise ValueError("pair staging is missing declared pairs")
        rows = tuple(validate_pair_cost_row(row) for row in pair_cost_rows)
        if [
            (row["blind_task_id"], row["test_index"]) for row in rows
        ] != list(self._expected_pairs):
            raise ValueError("pair-cost rows differ from the staged pair order")
        self._close_streams()
        pair_cost_bytes = serialize_jsonl(rows)
        pair_cost_path = self._root / "pair_cost_ledger.jsonl"
        with pair_cost_path.open("xb") as handle:
            handle.write(pair_cost_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        files: dict[str, dict[str, object]] = {
            name: {
                "sha256": self._digests[name].hexdigest(),
                "bytes": self._bytes[name],
                "rows": self._rows[name],
            }
            for name in _STREAM_FILES
        }
        files["pair_cost_ledger.jsonl"] = {
            "sha256": hashlib.sha256(pair_cost_bytes).hexdigest(),
            "bytes": len(pair_cost_bytes),
            "rows": len(rows),
        }
        manifest: dict[str, object] = {
            "schema": PAIR_MATERIALS_SCHEMA_VERSION,
            "status": "PAIR_MATERIALS_ONLY_NOT_POOL_COMPLETE",
            "expected_pairs": [list(value) for value in self._expected_pairs],
            "pair_count": len(rows),
            "files": files,
        }
        manifest["manifest_id"] = hashlib.sha256(serialize_json(manifest)).hexdigest()
        with (self._root / "pair_materials_manifest.json").open("xb") as handle:
            content = serialize_json(manifest)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        return {
            "status": manifest["status"],
            "staging_dir": str(self._root),
            "manifest_id": manifest["manifest_id"],
            "artifact_files": {
                name: str(self._root / name) for name in sorted(files)
            },
        }

    def abort(self, *, failure_code: str) -> None:
        if self._aborted:
            return
        self._close_streams()
        self._aborted = True
        marker = {
            "schema": PAIR_MATERIALS_SCHEMA_VERSION,
            "status": "ABORTED_NOT_POOL_COMPLETE",
            "failure_code": str(failure_code),
            "completed_pair_count": self._next_pair,
        }
        with (self._root / "ABORTED.json").open("xb") as handle:
            content = serialize_json(marker)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())


__all__ = ["M04APoolPairStagingSink", "PAIR_MATERIALS_SCHEMA_VERSION"]
