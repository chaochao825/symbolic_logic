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
