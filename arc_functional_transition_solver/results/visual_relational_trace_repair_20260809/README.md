# Visual relational trace-repair evidence

This directory contains the valid, content-addressed evidence for the
2026-08-09 relational actuator and ARC-AGI-2 public-training confirmation gate.
It excludes all invalid or superseded dry runs and contains no ARC task payloads,
query gold, model checkpoint, or public-evaluation data.

| Artifact | Role |
|---|---|
| `arcgen_feasibility_scan.json` | Corrected complete query-gold-blind scan of 139 eligible ARC-GEN families. |
| `arcgen_cohort_build_stderr.log` | Progress/terminal-reason log for the corrected scan. |
| `arc2_exposure_registry.json` | Frozen set of prior task/family exposures. |
| `arc2_cohort_manifest.json` | Query-blind 50-task ARC-AGI-2 training cohort definition and source identities. |
| `arc2_candidate_freeze.json` | Pre-gold programs, candidates, ledgers, and replay identities. |
| `arc2_result.json` | Frozen exact-coverage, pass@2, repair, cost, and gate endpoints. |
| `arc2_near_miss_quality_posthoc.json` | Explicitly post-hoc, query-blind diagnosis of the pixel-agreement false positives. |
| `arc2_gate_stderr.log` | Gate execution progress log. |
| `artifact_sha256.txt` | SHA-256 of every published artifact in this directory except itself. |

The interpretation and claim boundary are recorded in
`notes/results/relational-mask-arc2-confirmation-20260809.md`.
