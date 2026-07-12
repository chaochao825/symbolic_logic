# Provenance

## User-provided reference documents

The supplied documents were read as UTF-8. They were not copied into this repository; their external locations and SHA-256 hashes are recorded so the interpretation can be audited.

| Document | SHA-256 | Role in this project |
|---|---|---|
| `D:\xwechat_files\wxid_zqxxbf2n217z22_a684\msg\file\2026-07\txt1.txt` | `89C7DF27C692D7263E4D156F966309C3C8E0A3D4C08A5BB330C1B1D47EA204A9` | Positions LGN as a trainable, hardenable predicate compiler rather than a complete symbolic reasoner. |
| `D:\xwechat_files\wxid_zqxxbf2n217z22_a684\msg\file\2026-07\txt2.txt` | `8AE3E23F4E15251D81008D7F02E99B18546092856D59EEC52670368656A183DF` | Identifies soft/dense computation, relation enumeration, rule selection, and constraint verification as candidate gate-compilation targets. |

## Result provenance

- Raw results: `results/*.csv`.
- Aggregate bootstrap summaries: `results/summary_metrics.csv`.
- Runtime metadata: `results/run_metadata.json`.
- New hardening artifacts: `results/learned_gate_results.csv` and `results/learned_gate_bfs_results.csv` (full run: 40 local rows and 360 graph rows).
- New state/complexity artifacts: `results/state_transition_results.csv` (26 rows), `results/planning_frontier_results.csv` (22 rows), `results/noncompressible_scaling_results.csv` (60 rows), and `results/probability_marginalization_results.csv` (100 rows).
- Figure source: `figures/results_plot.py`; it reads only the raw CSV files, including `figures/learned_gate_integration.pdf`.
- Core exactness tests: `tests/test_core.py`.
- The previous mojibake-encoded report is preserved at `trash/20260712_evidence_report_revision/evidence_report_zh_legacy.md`; the current report is UTF-8.
