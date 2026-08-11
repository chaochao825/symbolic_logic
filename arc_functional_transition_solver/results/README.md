# Result status map

This directory contains evidence artifacts, build products, and control material.
Directory existence is not evidence that an experiment ran successfully.

| Area | Status | Claim boundary |
|---|---|---|
| `e00_harness*` | VERIFIED INFRASTRUCTURE | Harness/provenance smoke only; not solver accuracy. |
| `e01a_symbolic_v1` through `e01a_bbox_contact_v1` | VERIFIED PUBLIC-TRAIN DEVELOPMENT SMOKE | Frozen 20-task development-smoke candidate coverage; not public-evaluation or leaderboard performance. |
| `m04a_global_source_v0_1` | BUILD/PREFLIGHT EVIDENCE | Data/runtime/input construction. Protected split payloads and derived data caches are not public Git content. |
| `m04a_global_source_v0_2` through `v0_6` | INCOMPLETE REMEDIATION ITERATIONS | Deployment/preflight iterations; no successful production training artifact. |
| `m04a_global_source_v0_7` | TRAINING CAMPAIGN INCOMPLETE | Training reached step 13,150; six checkpoints through step 12,000 survive remotely. No atomically published validation metrics, selected checkpoint, candidate pool, or ARC result. |
| `m04a_global_source_v0_8` | **UNRUN** | Local robustness/deployment candidate only. Never deployed and never trained. |
| `m04a_r*_remote_control` and `run_m04a_*` | CONTROL MATERIAL, NOT RESULT | Operational scripts/receipts. Public bundles retain hashes but omit files containing machine-specific infrastructure metadata. |
| `m04a_r7_posthoc_evaluation_20260715.json` | READ-ONLY DIAGNOSTIC | Reconstructed frozen masked-validation metrics. It is not formal campaign evidence or ARC solve rate. |
| `m04a_r7_failure_receipt.json` | EXACT FAILURE RECEIPT | Campaign terminal state; SHA-256 is pinned in the analysis report. |
| `object_code_gate_v1_*_20260728` | VERIFIED NEGATIVE GATE | Two disjoint public-training 100-task audits. v0.1 adds 1/200 unique selectable coverage and 0/65 novel natural repairs; controller remains frozen. |
| `object_code_gate_v2_*_20260728` | VERIFIED NEGATIVE DEVELOPMENT GATE | Role reachability adds 2,784 trials and 40 role near misses on offset-100, but 0 exact/unique candidates and 0/76 novel repairs; offset-200 confirmation was not scored. |
| `object_code_gate_v3_scene_ast_*_20260729` | VERIFIED MIXED DEVELOPMENT GATE | Scene roles/canvas inference add exactly 3/100 unique selectable tasks, passing the provider-only gate, but union is 12/100 and natural typed repair is 0/100. This is representation reachability at about 333x v0.2 trials, not an efficiency, router, or neural claim. |
| `visual_provider_query_blind_20260807` | VERIFIED MIXED DEVELOPMENT GATE | A query-blind static VARC provider adds 8/31 unique selectable and 13/31 unique raw solutions on two disjoint exposure-audited ARC-AGI-2 training cohorts. Posterior disagreement localizes errors, but pixel-consensus repair adds 0 unique recoveries. This is not a public/private evaluation score or functional-switching claim. |
| `visual_relational_trace_repair_20260809` | VERIFIED NEGATIVE REPRESENTATION GATE | A corrected bounded ARC-GEN scan retains 0/139 families: 134 have no existing scene-AST single-slot neighbor and five end in typed construction exclusions. An opt-in relational-mask DSL then adds 0/50 unique solutions on a frozen ARC-AGI-2 public-training cohort; typed repair and equal-cost restart both recover 0. The three pixel-agreement parents are post-hoc diagnosed as background-dominated false near misses. LODO and controller training remain frozen. |
| `relational_delta_failure_matrix_20260810_v2` | VERIFIED QUERY-BLIND DEVELOPMENT DIAGNOSTIC | Complete terminal-stage assignment of 48 misses: 18 parse failures, 14 missing relations, 12 incompatible canvases, three legal-but-inexact programs, and one insufficient AST. This is a decomposition of one frozen language, not a causal ARC taxonomy. |
| `control_legend_dev_20260810` | OUTCOME-EXPOSED REPRESENTATION PROBE | A narrowly typed border/control-legend family adds two unique development candidates. It is not fresh-set generalization evidence and does not authorize controller training. |
| `arc_tgi_arcmini_cohort_v2_20260810` | VERIFIED COHORT CONSTRUCTION | Frozen generator-family-disjoint ARC-TGI development/confirmation/reserve partitions and contamination commitments. Generator-family disjointness does not imply semantic or program-distribution disjointness. |
| `nvarc_trm_arc_tgi_anchor_20260811` | VERIFIED SYNTHETIC STATIC ANCHOR | The reference-scale NVARC/TRM provider reaches 87/100 strict pass@1 and 92/100 strict top-10 oracle on the frozen 100-task ARC-TGI confirmation cohort. This is not an ARC-AGI score or functional-switching result. |
| `relational_transducer_development_20260811` | MIXED, OUTCOME-EXPOSED MECHANISM DIAGNOSTIC | On 49 eligible development tasks, the VARC/NVARC raw union is 46/49. A frozen 24-program D4/color transducer recovers the remaining three; composite allocation recovers 3 versus 0 for equal-charged-cost cold allocation. Posterior clearing replaces 7/15 selected tasks but also recovers 3, so posterior performance attribution fails. The preregistered 100-task confirmation gate remains required. |
| `object_program_workspace_controls_20260811` | VERIFIED CONTROLLED ACTUATOR PASS / SENSOR BOUNDARY | On 12 typed selector-fault controls, visual, residual-cleared, and shuffled arms each recover 12/12 versus 5/12 cold; visual has zero recovery unique over controls. This validates execution, intervention, replay, and cost accounting, not generalization. |
| `object_program_workspace_arc_tgi_dev_20260811` | VERIFIED REPRESENTATION/OUTPUT-FRONTIER NULL | On the outcome-exposed 100-task ARC-TGI cohort, v1 yields three new programs but zero new output candidates and zero unique recoveries; 94 tasks lack a demo-exact relational child. Prospective confirmation and controller training remain closed. |
| `object_graph_rewrite_v2_arc_tgi_dev_20260811` | VERIFIED NEGATIVE REPRESENTATION GATE WITH ONE UNIQUE RECOVERY | On the outcome-exposed 50-family ARC-TGI development cohort, a query-blind two-stage composition creates one novel output and it is exact/unique over baseline and equal-cost cold. The preregistered opportunity gate requires 5/50, so reserve remains unmaterialized and controller training remains closed. Forty-eight tasks have parents but no demo-exact second-stage composition. |
| `object_graph_rewrite_v2_qa_20260811` | TARGETED PASS / FULL SUITE NOT GREEN DUE TO MISSING PROTECTED FIXTURES | New-module tests pass 12/12. The complete run reports 546 passed, three skipped, and six pre-existing M04a evidence failures caused by absent protected snapshot/cache payloads; this is not reported as a green full suite. |
| `stateful_object_graph_rewrite_v3_arc_tgi_dev_20260811` | VERIFIED REPRESENTATION NULL / CONTROLLED STATEFUL ACTUATOR PASS | Persistent identities, typed node rewrites, suffix replay, and exact verifier parity pass a 60-test adjacent regression selection, but the exposed dev50 gate yields 0/50 novel-output opportunities. The compiler-selected node improves 1/196 parents versus 94/196 under a demo-only all-node audit, yet no legal node is demo-exact. The full suite remains non-green only on the same six missing protected M04a payload tests; reserve and controller remain closed. |

The public publication bundle is closed by `PUBLICATION_MANIFEST.json`:

- copied files retain their original SHA-256;
- large JSON/JSONL evidence may be stored as deterministic `.gz`, with hashes for
  both source and compressed bytes;
- binary caches, third-party/data archives, any local checkpoint files, protected
  split payloads, and machine-specific operational records found in the local source
  tree are represented by path, byte count, SHA-256, and an explicit `manifest_only`
  reason.

Checkpoint binaries are not ordinary Git blobs. The six surviving r7 checkpoint
files remain only on the remote experiment host and were never copied into this local
source tree, so they are not `PUBLICATION_MANIFEST.json` entries. Their basenames,
hashes, and sizes are recorded only in `m04a_r7_posthoc_evaluation_20260715.json`;
publishing the binaries themselves requires a separately authorized GitHub Release or
Git LFS policy.

When `PUBLICATION_MANIFEST.json` is present, byte-level tests whose required payloads
are explicitly `manifest_only` skip with that reason. Algorithmic and metadata-contract
tests still run; in the full private evidence tree the same tests verify the payload
bytes normally.
