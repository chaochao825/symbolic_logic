# Object/code failure-control gate v1 (2026-07-28)

## Decision

Version 1 does **not** pass the candidate or repair gates. The controller remains
frozen, and these runs do not authorize a masked-neural provider or a learned
router experiment.

The negative result is not an execution or replay failure. It is a failure of the
current representation/enumeration slice, followed by a failure to demonstrate a
novel repair frontier on naturally produced near misses.

## Frozen protocol

- Source commit: `018e3bf04b472df648e1a642d1a04291b9326418`.
- Dataset: two three-axis-disjoint 100-task slices from ARC-AGI-1 public training.
- Generation input: demonstrations plus query inputs through `BlindTask`; query
  outputs are read only after content-addressed pools are written.
- Candidate budget: exhaustive v0.1 grammar under the 20,000-trial cap.
- Repair comparison: identical observed program trials, demo executions, and
  query executions for typed repair and cold restart. Prediction-blind padding is
  used only to consume the same query-execution reservation; artifact replay and
  oracle scoring are separate audit overhead.
- Selection: demo-exact, hard replay verification, then MDL rank with pass@2.
- Controller training, model training, model downloads, and external API calls:
  none.

Implementation evidence was positive: 393 tests passed and 9 were skipped; all
200 task pools replayed exactly; both runs used clean source, had no publication
blocker, and had no task exception. Generator-known controls produced the four
predicted actions (`object_rematch`, `canvas_reinfer`, `fill_ast_hole`, and
`reparse_background`) and recovered their synthetic targets.

## Coverage results

| Slice | Frozen baseline selectable | Provider raw/selectable/pass@2 | New unique selectable | Baseline + provider selectable union | Gate |
|---|---:|---:|---:|---:|---|
| development offset 0 | 10/100 | 1/1/1 | 1/100 | 11/100 | fail |
| disjoint offset 100 | 9/100 | 0/0/0 | 0/100 | 9/100 | fail |
| combined diagnostic | 19/200 | 1/1/1 | 1/200 | 20/200 | not a pooled gate |

The only provider solve was development task `36d67576`, using
`d4_label_completion(background=0, structure_color=4, connectivity=8,
attachment_radius=2)`. That task was inspected during provider development, so it
is not confirmatory evidence. The disjoint slice contributed no solve.

The required per-slice gates were selectable union at least 25/100 and provider
unique coverage at least 3/100. Both failed by a wide margin. Of the 200 tasks,
199 were classified as candidate-language failures and one was solved. All 200
finite grammars were exhaustive under the declared cap, so this is not a trial-cap
or router failure.

## Why candidate coverage failed

The implementation exposed two program families, but the task-derived v0.1
enumerator instantiated only one on these slices:

| Slice | Enumerated programs | `d4_label_completion` | `role_stamp` | Emitted candidates |
|---|---:|---:|---:|---:|
| development offset 0 | 1,736 | 1,736 | 0 | 30 |
| disjoint offset 100 | 1,516 | 1,516 | 0 | 36 |

The role enumerator required source and target colors to disappear from every
demonstration output before considering any canvas mode. That condition is valid
for some blank/erase transforms, but it incorrectly prevents copy-preserving role
hypotheses from entering the search. In this audit, the intended two-family
provider therefore collapsed to a narrow D4 annotation completer. This is an
algorithm-design/reachability defect in v0.1, not a runtime implementation defect
and not evidence that object-centric code synthesis is generally ineffective.

## Natural repair results

| Slice | Natural cases (tasks) | Typed/cold matched program trials | Matched demo executions | Matched query executions | Typed raw oracle rediscoveries | Typed novel oracle recoveries | Cold novel recoveries |
|---|---:|---:|---:|---:|---:|---:|---:|
| development offset 0 | 29 (7) | 183 / 183 | 624 / 624 | 183 / 183 | 6 | 0 | 0 |
| disjoint offset 100 | 36 (8) | 122 / 122 | 394 / 394 | 122 / 122 | 0 | 0 | 0 |
| combined | 65 (15) | 305 / 305 | 1,018 / 1,018 | 305 / 305 | 6 | 0 | 0 |

The sample gate also failed: only 65 natural near misses were produced, below the
required 100. Natural diagnoses contained only object/correspondence and
background cases; no natural `canvas_reinfer` or `fill_ast_hole` case appeared.

All six raw typed recoveries came from six parents on the already solved
development task `36d67576`. The recovered exact program was already present in
that task's frozen initial pool. Thus typed filtering navigated the local grammar
more efficiently than the deterministic cold prefix, but it did not add coverage
or change the candidate frontier. Counting these six rediscoveries as repair
success would have produced a false method claim. The hard metric therefore
records 0/65 unique novel recoveries.

## Failure attribution

1. **Implementation/protocol:** supported. Positive controls, strict types,
   content IDs, clean source binding, replay, hidden-query-output invariance, and
   matched native cost all passed.
2. **Candidate representation v0.1:** rejected. It provided only 1/200 unique
   coverage and had no disjoint-slice solve; 199/200 tasks had no demo-exact
   program.
3. **Typed repair v0.1 on natural failures:** not supported. Controlled injections
   prove that the plumbing can select four actions, but natural repair produced no
   novel oracle recovery and did not reach the 100-case sample gate.
4. **General residual-control hypothesis:** not rejected by this experiment. The
   available natural actions only filtered an already enumerated D4 grammar, so
   they did not yet implement the required cross-representation frontier change.

## Next bounded iteration

Create a versioned v0.2 provider that broadens role candidates without changing
legacy solvers or v1 artifacts. Source and target role eligibility should depend
on the canvas contract instead of globally requiring disappearance from outputs.
Re-run the same two frozen slices and accept v0.2 only if it yields additional
demo-exact and unique selectable candidates. Do not train a controller; if role
reachability improves but unique coverage remains below 3/100, the next change
must add new object/canvas/composition primitives rather than a more complex
router.

## Artifact identities

- Development result:
  `ee6ca60d0964f601881eda3fad24da818140c85048fc21c90161dbc34976121d`.
- Disjoint result:
  `07b756c26c6ddee4a65f8b424ad55b48c751688784af73dbc8f991aad746aa79`.

Each summary contains the baseline hash, task/source/blind hashes, pool and repair
artifact IDs, runtime determinism environment, native cost ledger, gate decision,
and exact source provenance.
