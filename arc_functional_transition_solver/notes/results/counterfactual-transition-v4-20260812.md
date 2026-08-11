# Counterfactual transition v4 result

## Decision

The preregistered exposed-development representation gate fails.  Bounded
one-to-three-node counterfactual transitions produce zero query-blind output
opportunities on the frozen 50-family ARC-TGI cohort.  None of the three
single-arm policies reaches the required 5/50 threshold, so the reserve remains
unmaterialized and controller training remains prohibited.

This is a valid method-level null, not a failed run.  The canonical run finished
with status zero, the two candidate freezes and two score files are byte
identical, all content IDs and hashes verify, no query gold was present during
candidate construction, and all 36 targeted semantic tests pass.

The result also corrects an overly strong interpretation of v3.  The current
language does contain many local residual-descending transitions.  What it does
not contain is a transition path that reaches a demonstration-exact program.
The observed causal chain is therefore:

```text
narrow single-operation scene language
-> many legal one-to-three-node rewrites
-> residual descent on a subset of tasks
-> no demonstration-exact state
-> exact-verification guard emits no candidate
-> no new query frontier and no recovery
```

## Frozen contract and reproducibility

- Source commit: `c4bd579b0f0d6f58b1cf30166276dc719676aff7`.
- Frozen cohort ID:
  `b237836bbaf9340497292f632efb7c2208e140a877b75a5bd3783ff11b4e2a27`.
- Parent v3 freeze ID:
  `66fb434846843584cde0198be17af637fb9af2d61521e9699ecb63c1f5712269`.
- v4 freeze ID:
  `f2663571ea9f75268d0a110b5ad714e2431196e2c87d423ca1106af716f5ef48`.
- v4 result ID:
  `b13a4339e7f877eefda628a5907d9e47b56d4f3158a75f0e22e07f96f8f20b8c`.
- v4 failure-audit ID:
  `3fb3d51336e1e531fa6b8a264d61df35deb1dd97893d306672b9f54deebacbac`.
- Paired freeze SHA-256:
  `b1be49c2d74d5bf43958a429b49314a49ff8b89767e6585e2615bd4f5ec1b310`.
- Paired result SHA-256:
  `fea1957f27408093a44ebd8a51b9f0935dffed63028da5a96105993de0078e0a`.
- Search contract per task and arm: 2,000 first-stage trials, at most four
  parents, 512 transition trials per parent, 2,048 reserved transition trials,
  and at most 32 emitted candidates.
- The three-arm union consumes three single-arm reservations and is explicitly
  marked non-comparable to one cold arm.
- Candidate generation used only blind tasks.  Development solutions were read
  only after both candidate freezes were complete and byte-identical.  No
  public evaluation data was read and no controller was trained.

The two eight-worker freezes took 55:02 and 55:06 wall time.  GNU `time`
reported about 7,278 seconds of user CPU per replicate, average utilization of
222%, peak RSS of about 12.56 GB, zero swaps, zero major page faults, and exit
status zero.  These physical costs are environment-dependent and are not used
as algorithmic credits, but they expose severe object-count-conditioned compute
variance that the logical trial ledger alone does not show.

## Main gate result

| Single arm | Novel opportunity tasks | Demo-exact candidates | Exact / pass@2 | Unique recovery |
|---|---:|---:|---:|---:|
| A: distance-tiered | 0/50 | 0 | 0 / 0 | 0 |
| B: typed semantic bundles | 0/50 | 0 | 0 / 0 | 0 |
| C: residual beam | 0/50 | 0 | 0 / 0 | 0 |

The incumbent remains exact on 2/50 tasks and the frozen equal-reservation cold
arm remains exact on 0/50.  The three-arm oracle union is also 0/50.  Forty-nine
tasks have four execution-valid stateful parents; one has none.  Every reachable
task terminates with legal transitions but no demonstration-exact candidate.
Consequently no transition query execution is needed and the novelty guard
correctly reports zero frontier changes.

## Residual descent is real but insufficient

The three policies differ before the exact-verification barrier:

| Arm | Tasks with any residual descent | Improving trials | Realized / padding trials | Improvement / realized | Improvement / reserved |
|---|---:|---:|---:|---:|---:|
| A: distance-tiered | 30 | 4,201 | 99,115 / 3,285 | 4.24% | 4.10% |
| B: typed semantic bundles | 24 | 3,781 | 85,640 / 16,760 | 4.41% | 3.69% |
| C: residual beam | 27 | 4,164 | 47,506 / 54,894 | 8.77% | 4.07% |

The apparently high residual-beam rate is conditional on realized trials.  It
pads 53.6% of its frozen reservation.  Under the fair reserved denominator it
does not beat distance-tiered search.  More importantly, the task sets are
nested rather than complementary:

| Improvement pattern | Tasks |
|---|---:|
| all A, B, and C improve | 24 |
| A and C improve, B does not | 3 |
| only A improves | 3 |
| no arm improves | 20 |

Thus `B subset C subset A` at task level.  Typed diagnosis and residual beam add
no unique descent coverage, let alone an exact recovery.  C is a more selective
sampler of some already reachable downhill transitions, not evidence for a
better repair policy.

The 196 parent certificates diagnose 180 `assignment` nodes, 12 `operate`
nodes, and four `canvas` nodes.  Assignment therefore receives 91.8% of all
diagnoses.  This concentration, the nested improvement sets, and the lack of an
exact state show that the certificate is not yet a causal localization of a
missing program variable.  It is mainly a coarse label attached to correlated
raster residuals.

## What failed: implementation versus method

Two invalid attempts are retained separately and do not update the scientific
claim.

1. Attempt 1 used MDL tie-break fields when counting residual improvements.
   Equal residuals with shorter descriptions were incorrectly counted as
   repairs.  Candidate generation, exactness, and novelty were unaffected, but
   the diagnostic rate was invalid.  The run was stopped before either freeze
   completed, and the metric now compares only exactness, shape, mismatch, and
   execution validity.
2. Attempt 2 passed semantic tests but failed to materialize either freeze after
   more than 50 minutes because high-object-count tasks repeatedly serialized
   grammar nodes and constructed unused pairwise relations.  No solution was
   read.  The retry precomputes immutable node/program keys, reuses one canonical
   grammar, and omits relations only in grammar construction and raster-only
   first-stage scoring.  Stateful replay still constructs the complete
   persistent relation state.

The canonical result remains expensive but completes reproducibly.  A normal
preflight task preserved its exact trial/candidate trace after the performance
changes.  A high-cardinality preflight took 987.7 seconds and found 44 genuine
distance-tiered residual improvements but zero exact candidates, anticipating
the full-cohort pattern.  Controlled counterfactual tests still recover an
injected assignment-plus-canvas fault and reject wrong declarations and bridge
lesions.  Therefore the actuator and replay implementation work on representable
faults; the natural null is a representation/search failure.

The full test suite requires `PYTHONPATH=src:tests`.  A first collection attempt
without `tests` on the path is retained as an invalid harness attempt.  The
correct run reports 584 passed, three skipped, and the same six pre-existing
M04a evidence failures.  Those failures concern absent protected sidecar,
cache, and `source_snapshot.zip` payloads; none touches the v4 implementation or
artifacts.  The 36-test v4/scene/object/stateful regression selection is green.

## Comparison with earlier attempts

| Stage | Positive evidence | Limitation exposed |
|---|---|---|
| object-graph rewrite v2 | 1/50 apparent novel exact output | recovery came from structurally unfaithful raster composition and is not a valid mechanism anchor |
| stateful rewrite v3 | stable persistent identities and exact suffix replay | 0/50; compiler-selected node improved only 1/196 parents, while any legal node improved 94/196; no node reached exact |
| counterfactual transition v4 | 1--3-node legal rewrites create genuine descent on 30/50 tasks | 0/50; specialized policies have no unique descent and no arm reaches demo exact |
| object-code v0.3 | 4/100 pass@2 and three unique tasks in its 100-task audit | all unique coverage came from crop; protocol/cohort differs from this 50-task gate |
| visual posterior studies | selectable visual candidates and strong error localization (`AUROC=0.942`, top-10% mask recall lift `5.87x`) | pixel consensus and structural repair produced zero unique recovery |
| workspace/controller studies | replay, provenance, budgets, and intervention plumbing are auditable | visual/clear/shuffle controls were identical; no natural frontier-changing action |
| reference-scale NVARC/TRM anchor | 90/100 pass@2 on its synthetic family-disjoint ARC-TGI anchor | different training scale and cohort; evidence for strong candidate generation, not for current residual control |

The progression is informative rather than repetitive.  V2 showed why output
novelty without structural fidelity is unsafe.  V3 established persistent state
but showed single-node causal localization was poor.  V4 demonstrates that
multi-node closure contains downhill moves, while falsifying the stronger claim
that typed bundles or a shallow residual beam can convert them into verified
programs.

## Why strong ARC systems remain far ahead

Strong neural, code, program-synthesis, and test-time-adaptation systems change
the candidate distribution itself.  Their model or synthesizer can introduce
new object variables, sequential operators, latent correspondences, learned
visual concepts, and task-conditioned programs.  The current v4 search only
recombines fields of a fixed six-region scene pipeline with one object operation.
Changing up to three existing regions does not add a second operation, a loop,
a new relation predicate, a new parse variable, or a new canvas construction.

This distinction also explains why a lower pixel residual is not enough.  The
current residual is not an admissible heuristic over program space: a raster can
move closer to demonstrations while entering a state with no legal completion.
Residual beam probes one-node edits and completes them with a bounded two/three
node bundle, but it does not preserve an evolving belief state, recompile the
failure after every action, invent an operator, or plan through a temporarily
worse state.  It is a shallow local search over a narrow language, not yet the
long-horizon functional switching mechanism in the original motivation.

## Next bounded sequence

Do not tune A/B/C on these now-exposed 50 families and do not train a router.
The next work should use a fresh, preregistered family-disjoint cohort and run
the following gates in order:

1. **Strong static anchor.** Freeze one reference-scale neural/code provider and
   measure selectable union and unique coverage before adding control.  Require
   at least 25/100 selectable coverage and 3--5/100 unique provider coverage.
2. **Joint typed-hole solver.** Convert a node-level trace into explicit object,
   relation, canvas, mask, parameter, and sequential-AST holes, then solve the
   holes jointly across all demonstrations.  Compare against equal-cost full
   restart.  This tests whether the state exposes the missing variable rather
   than merely a low-resolution node label.
3. **Iterative counterfactual planning.** Only if step 2 recovers at least 5/100
   near misses, maintain a content-addressed state graph, recompile the
   certificate after every action, allow bounded non-monotone paths, replay only
   affected subtrees, and charge primitive-equivalent execution cost.  Compare
   against distance-tiered search, not only against a weak typed baseline.
4. **Visual-to-structural proposals.** Use leave-one-demo-out visual posterior
   only to restrict hole domains, object matches, masks, and canvas hypotheses.
   It must not directly vote pixels.  Require unique exact recovery above the
   same cold-restart cost before claiming a useful cross-representation bridge.
5. **Functional-switching interventions.** Only after a natural repair gate
   passes should residual injection, bridge lesion, module-cost intervention,
   and new-module insertion test causal switching.  XGBoost, bandit, GRU, and
   diffusion controllers remain out of scope until then.

The brain-inspired objective is retained, but the current evidence narrows its
necessary mechanism.  Persistent identity and a shared executable trace are
infrastructure.  Functional switching additionally requires causal state
variables, operators that create genuinely new candidate regions, iterative
failure recompilation, and memory over successful state transitions.  V4 has
the first two infrastructure pieces but not those four algorithmic capabilities.
