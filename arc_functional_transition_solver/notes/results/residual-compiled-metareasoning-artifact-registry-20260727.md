# Residual-Compiled Metareasoning Artifact Registry

Date: 2026-07-27

This sidecar registry distinguishes primary evidence from early engineering
artifacts without rewriting historical JSON. Existing result IDs remain
content addresses of the payloads originally written; changing those files
would destroy rather than repair their provenance.

## Status vocabulary

- `canonical`: generated or cleanly regenerated under the recorded committed
  producer source, with the currently referenced profile content available.
- `noncanonical_dirty_worktree`: generated while HEAD did not describe the
  actual source tree. It may be useful for debugging but is not source-bound
  primary evidence.
- `broken_external_profile_reference`: the JSON records a profile ID and path,
  but the file now present at that path has different content or ID.
- `historical_json_immutable`: retain the original JSON exactly; corrections
  live in this registry or a future content-addressed sidecar.

## Canonical primary artifacts

| Role | Repository-relative path | Primary ID | File SHA256 | Producer source | Status |
|---|---|---|---|---|---|
| option-switching audit | `results/residual_compiled_metareasoning_p1_20260726/option_switching_audit.json` | result `3e6df17d9e2127df7f40a095272207f43d9494bbab65788dfb88a6650057cbd8` | `ab56f4ee50dcfc9abf3f6acce26045b042aa2830dc5f82fc9a154bf7311ebe6d` | `f62a0fe5ea326a0b20bcccd6543575e4f6230925` | `canonical` |
| fit-calibrated native profile | `results/residual_compiled_metareasoning_p1_20260726/native_budget_profiles/8f54b70c3f3c1b926fc873c58a7d7e6e4bbac3e7b4da9e943aef11c1320b95b8.json` | profile `8f54b70c3f3c1b926fc873c58a7d7e6e4bbac3e7b4da9e943aef11c1320b95b8`; contract `168b711f134e84d4f8e0aac41304718875fa1057661ea7b8fa351b401d9eb98a` | `32cb972d126e2ccd3a23459ed272272ecca9c3c2a95c8a1c01fe0563baa6a70d` | `f62a0fe5ea326a0b20bcccd6543575e4f6230925` | `canonical`; legacy v1 does not embed fit task source/blind fingerprints |
| native-budget phase ablation | `results/residual_compiled_metareasoning_p1_20260726/native_budget_phase_ablation_offset300_n20/summary.json` | result `fbefd0b0b17131eb21980943a746ad2efaf342a72729852fed10aaaa83f6bb98` | `6760ef94fced66cdd9f94c1c16431ad48639ab3e38cdd3c5d7d0b3d5e43a323d` | `f62a0fe5ea326a0b20bcccd6543575e4f6230925` | `canonical` for pass, trajectory, and reservation ledger; not canonical for complete physical actual cost |

The canonical source above is the producer-code commit, not the source commit
of every upstream summary consumed by the audit. Input-artifact file hashes and
producer commits are not fully bound by the current option-audit schema; that
is a known provenance gap rather than an implied claim of transitive closure.

## Early non-canonical engineering artifacts

| Artifact | Repository-relative path | Known ID and file SHA256 | Profile-reference status | Registry status |
|---|---|---|---|---|
| schema-v3 cost-DAG smoke | `results/residual_compiled_metareasoning_p1_20260726/cost_dag_smoke/summary.json` | result `aa50e1d9847a8bdd61eb4196fb37682c0565508804e7b7d81d3907292e0369d3`; file `33400d23601ddfdfe3c3113bbfc6a75ce641baa041d5580ddd0f5c8929b6b983` | no native profile was required | `noncanonical_dirty_worktree`, `historical_json_immutable` |
| native-budget smoke | `results/residual_compiled_metareasoning_p1_20260726/native_budget_smoke/summary.json` | result `d126f84172266ff1c52d442433b4501ca85b8c49ee025622fcca5aa6f4134e39`; file `27a279dbecfde3d8a08dc135408f1bfafa9fa60dec5b1b7ef47162092e80e091` | literal path is broken; declared `eb7abb...` content is resolved by the immutable profile-registry path below | `noncanonical_dirty_worktree`, `broken_external_profile_reference`, `historical_json_immutable` |
| CA native-cost regression smoke | `results/residual_compiled_metareasoning_p1_20260726/native_budget_ca_cost_smoke/summary.json` | result `e96fe1f72fad9daa3de290da83520823378bbfa5ea28f4c5d095a6c17b293ce0`; file `161e05f0eebf29c57d408c5b40e1b81687201cb637201da9d5a40fe80e0b265e` | literal path is broken; declared `eb7abb...` content is resolved by the immutable profile-registry path below | `noncanonical_dirty_worktree`, `broken_external_profile_reference`, `historical_json_immutable` |
| native-budget 20-task pilot | `results/residual_compiled_metareasoning_p1_20260726/native_budget_pilot_offset300_n20/summary.json` | result `bb0cc80d5744f01c011d292f4566e8183673cc3897ad3680bf52c327e00aa313`; file `2e313d5aac0552f0b2cd94d79f3ec2441b44bec144cc5812d25852d37a410570` | literal path is broken; declared `eb7abb...` content is resolved by the immutable profile-registry path below | `noncanonical_dirty_worktree`, `broken_external_profile_reference`, `historical_json_immutable` |

All four artifacts above declared the base HEAD
`ab9442facfc92bb04ebfd3c3289f9d6396803f8a` during development, but that commit
does not contain the native-metareasoning implementation used to produce the
native files. HEAD therefore cannot serve as their exact source snapshot.

## Repair plan

1. Do not edit, regenerate in place, or relabel any historical JSON.
2. Treat the three canonical artifacts above as the only primary evidence for
   the switching audit, calibrated contract, and 20-task phase ablation.
3. Keep this Markdown file as the human-readable registry. The machine-readable
   registry is
   `results/residual_compiled_metareasoning_p1_20260726/artifact_registry.json`.
4. Resolve the old declared profile content through
   `results/residual_compiled_metareasoning_p1_20260726/native_budget_profiles/eb7abb42ae146c6b34913ba61f0c0af0b6c955691abc3887735c4b81a567405a.json`.
   Its recomputed profile ID is `eb7abb...` and its file SHA256 is
   `a63f33d88a85b84379bd341bb06fa4a2a3cc313258940025eef0694795a99814`.
   This recovers the declared profile **content**, but the legacy profile does
   not bind a clean source tree and byte-for-byte provenance of the original
   path remains unavailable.
5. Retain `broken_external_profile_reference` for the historical summaries:
   their literal recorded path still resolves to different `8f54b7...` content.
   The sidecar resolution must not be represented as an in-place repair, and
   the canonical `8f54b7...` profile must not be substituted under the old ID.
6. Future summaries should embed a profile file SHA256 or immutable profile
   snapshot, validate `profile_id` on load, capture clean source state at start
   and end, and record input-artifact hashes.

## Claim boundary

The registry correction does not alter the canonical negative result:
phase-enabled and phase-disabled grounding have identical actions and pass@2 on
the canonical 20-task ablation, and direct residual ablation or shuffling is
inert in the canonical 100-task option audit. It only narrows what can be
claimed about early smoke provenance, full native-cost completeness, and
independent JSON replayability.
