"""Create a compact, provenance-checked bridge to the 210 Hard-LGN results.

The source experiment lives in the separate ``learnable_logic`` project.  This
script does not relabel those rows as symbolic_logic experiments; it extracts a
small auditable subset and records hashes of the full source tables.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


EXPECTED = (
    "abc_synthesis_comparison.csv",
    "hardening_method_comparison.csv",
    "truth_refit_final_test_comparison.csv",
)


def normalized_sha256(path: Path) -> str:
    """Hash text after CRLF-to-LF normalization for checkout portability."""

    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize(abc_rows: list[dict[str, str]], hardening_rows: list[dict[str, str]], truth_rows: list[dict[str, str]]) -> tuple[list[dict], list[dict]]:
    selected = [row for row in hardening_rows if row.get("selected") == "1" and row.get("method") == "redundant_task_hardened"]
    candidate_classes = Counter()
    for row in selected:
        candidate = row["candidate"]
        if candidate.startswith("gumbel_sample_"):
            candidate_classes["best_of_32_gumbel"] += 1
        elif candidate == "truth_table_refit":
            candidate_classes["truth_table_refit"] += 1
        else:
            candidate_classes[candidate] += 1

    abc_ok = [row for row in abc_rows if row.get("abc_status") == "ok"]
    abc_preserved = [row for row in abc_ok if abs(float(row.get("post_acc_delta_vs_pre_discrete") or "nan")) <= 1e-12]
    reductions = [float(row["abc_and_reduction_vs_pre_nd"]) for row in abc_ok]
    truth_task_wins = sum(row.get("status_vs_task_selected") == "WIN" for row in truth_rows)
    truth_task_losses = sum(row.get("status_vs_task_selected") == "LOSS" for row in truth_rows)
    truth_strong_wins = sum(row.get("status_vs_strong_baseline") == "WIN" for row in truth_rows)
    truth_strong_losses = sum(row.get("status_vs_strong_baseline") == "LOSS" for row in truth_rows)

    summary = [
        {
            "axis": "hardening_selection",
            "metric": "selected_task_aware_rows",
            "value": len(selected),
            "denominator": len(selected),
            "status": "measured",
            "interpretation": "validation-selected candidates; scope is the evaluated rows, not a universal winner claim",
            "source": "hardening_method_comparison.csv",
        },
    ]
    for candidate, count in sorted(candidate_classes.items()):
        summary.append(
            {
                "axis": "hardening_selection",
                "metric": f"selected_{candidate}",
                "value": count,
                "denominator": len(selected),
                "status": "measured",
                "interpretation": "no candidate class is selected on every evaluated row",
                "source": "hardening_method_comparison.csv",
            }
        )
    summary.extend(
        [
            {
                "axis": "truth_table_refit",
                "metric": "heldout_wins_vs_validation_selected",
                "value": truth_task_wins,
                "denominator": len(truth_rows),
                "status": "partial",
                "interpretation": f"wins={truth_task_wins}; losses={truth_task_losses}",
                "source": "truth_refit_final_test_comparison.csv",
            },
            {
                "axis": "truth_table_refit",
                "metric": "heldout_wins_vs_strong_baseline",
                "value": truth_strong_wins,
                "denominator": len(truth_rows),
                "status": "partial",
                "interpretation": f"wins={truth_strong_wins}; losses={truth_strong_losses}",
                "source": "truth_refit_final_test_comparison.csv",
            },
            {
                "axis": "abc",
                "metric": "successful_synthesis_rows",
                "value": len(abc_ok),
                "denominator": len(abc_rows),
                "status": "no_data" if not abc_rows else ("pass" if len(abc_ok) == len(abc_rows) else "partial"),
                "interpretation": "ABC/AIG structural optimization completed",
                "source": "abc_synthesis_comparison.csv",
            },
            {
                "axis": "abc",
                "metric": "hard_accuracy_preserved_rows",
                "value": len(abc_preserved),
                "denominator": len(abc_ok),
                "status": "no_data" if not abc_ok else ("pass" if len(abc_preserved) == len(abc_ok) else "fail"),
                "interpretation": "optimized BLIF was re-evaluated; equivalence is empirical over the task inputs",
                "source": "abc_synthesis_comparison.csv",
            },
            {
                "axis": "abc",
                "metric": "minimum_reported_post_abc_and_vs_pre_blif_node_reduction",
                "value": min(reductions) if reductions else "",
                "denominator": "",
                "status": "measured" if reductions else "no_data",
                "interpretation": "cross-representation statistic: pre-BLIF node count versus post-ABC AIG AND count; not like-for-like AIG reduction or hardware PPA",
                "source": "abc_synthesis_comparison.csv",
            },
            {
                "axis": "abc",
                "metric": "maximum_reported_post_abc_and_vs_pre_blif_node_reduction",
                "value": max(reductions) if reductions else "",
                "denominator": "",
                "status": "measured" if reductions else "no_data",
                "interpretation": "cross-representation statistic: pre-BLIF node count versus post-ABC AIG AND count; not like-for-like AIG reduction or hardware PPA",
                "source": "abc_synthesis_comparison.csv",
            },
        ]
    )
    return summary, selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-root", default="/home/spco/sow_linear/hard_lgn_gap_proto/runs/reports_next_stage_v23_valid_st_fields_fixed")
    parser.add_argument("--source-commit", default="39c73d099bdcd7068f99bdea7667ae54578c193c")
    parser.add_argument("--source-hostname", default="hi-X640-G40")
    args = parser.parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    missing = [name for name in EXPECTED if not (input_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"missing source artifacts: {missing}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"immutable output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    abc_rows = read_csv(input_dir / EXPECTED[0])
    hardening_rows = read_csv(input_dir / EXPECTED[1])
    truth_rows = read_csv(input_dir / EXPECTED[2])
    summary, selected = summarize(abc_rows, hardening_rows, truth_rows)
    write_csv(output_dir / "hard_lgn_external_summary.csv", summary)
    write_csv(output_dir / "selected_task_hardening_rows.csv", selected)
    write_csv(output_dir / EXPECTED[0], abc_rows)
    write_csv(output_dir / EXPECTED[2], truth_rows)

    derived_names = (
        "hard_lgn_external_summary.csv",
        "selected_task_hardening_rows.csv",
        EXPECTED[0],
        EXPECTED[2],
    )

    metadata = {
        "source_project": "chaochao825/learnable_logic hard_lgn_gap_proto",
        "source_commit": args.source_commit,
        "source_hostname": args.source_hostname,
        "source_root_on_210": args.source_root,
        "source_environment": "not recorded in the selected external artifacts",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "provenance bridge only; these are external Hard-LGN results, not reruns by symbolic_logic",
        "hash_mode": "sha256 after CRLF-to-LF normalization for CSV text",
        "source_normalized_sha256": {name: normalized_sha256(input_dir / name) for name in EXPECTED},
        "source_rows": {EXPECTED[0]: len(abc_rows), EXPECTED[1]: len(hardening_rows), EXPECTED[2]: len(truth_rows)},
        "derived_rows": {"hard_lgn_external_summary.csv": len(summary), "selected_task_hardening_rows.csv": len(selected)},
        "derived_normalized_sha256": {name: normalized_sha256(output_dir / name) for name in derived_names},
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), **metadata}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
