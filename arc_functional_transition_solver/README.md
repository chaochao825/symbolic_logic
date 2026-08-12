# ARC Functional Transition Solver

This project studies an accuracy-first solver for ARC-AGI-1 and ARC-AGI-2.
It treats solving as a budgeted, closed-loop search over three coupled spaces:

1. output grids proposed or repaired by masked discrete denoising;
2. executable rules synthesized in a typed DSL or generated as code;
3. functional traces that decide which specialist should act next.

The system uses a shared blackboard of demonstrations, parse hypotheses,
candidates, execution traces, residuals, verification evidence, and remaining
compute. A controller may be implemented by rules, XGBoost, an MLP, or a
differentiable logic-gate network (DLGN). DLGN is an ablation and efficiency
candidate, not a presupposed winner.

## Scientific priorities

1. Raise oracle candidate coverage under a fixed task budget.
2. Raise exact-match pass@2 given a fixed candidate pool.
3. Recover near misses through local repair instead of unconditional restarts.
4. Reduce search cost and timeout rate without pruning the true hypothesis.
5. Compare control policies fairly on identical features, candidates, and budgets.

## Evidence status

All method claims are hypotheses until backed by result files in `results/`.
The numerical results pasted into the initiating discussion are recorded as
user-provided but unverified because their linked experiment artifacts are not
present in this workspace.

Phase 0 is complete as infrastructure evidence. The current `e00_harness_v2`
runs bind exact dataset and scorer commits, read dataset/scorer identity from Git
blobs, archive the executing source, enforce two-attempt pair-fraction scoring,
and atomically publish content-hashed bundles. Their D4 scores are harness smoke,
not evidence for the proposed solver. Public evaluation access is disabled until
a release-gate ID is added to the source-controlled allowlist.

Formal evidence commands must use `python scripts/afts_arc_evidence.py ...`.
The launcher chooses a fresh, nonexistent bytecode-cache prefix before importing
the package, preventing a timestamp-valid stale `.pyc` from diverging from the
archived source. Direct `python -m afts_arc` and programmatic `main(argv)` calls are
not accepted as evidence entry points.

E01a now has six source-bound symbolic checkpoints. The compact M02a/DSL v0.1 run
measures 10% strict coverage on a fixed 20-task public-training development smoke;
the post-hoc M03a/DSL v0.2 scale/tile slice raises the same smoke to 20% while
leaving 16 tasks without candidates; and the independent M02b/DSL v0.3 panel-overlay
slice raises it to 25%. The DSL v0.4 indexed panel-sequence D4 slice then reaches
30%, the DSL v0.5 periodic panel-lattice slice reaches 35%, and the additive
M02c/DSL v0.6 axis-ray bbox-contact slice reaches 40%, while preserving all prior
solved tasks. All six have frozen blind/oracle/pool/evaluation bundles and
full-parent replay. The M02c/M05f slice adds only `1f642eb9` and explicitly defers
multi-anchor move-and-erase behavior. Twelve of the 20 smoke tasks still lack a
correct candidate. A subsequent limited input-only scan found that the two cleanest
remaining diagnosed signatures each identify only their already known task; it does
not exclude broader symbolic opportunities. To stop accumulating one-task evidence,
the next direction is M04a global masked-grid generation. Its independently reviewed
pre-implementation contract is now frozen at
`notes/design/m04a-global-masked-grid-contract.md`. The exact 8,733,706-parameter
Grid-CMLM, MaskGIT sampler, deterministic training/validation primitives, production
preflight, lock/launch evidence, and closed-world training-bundle verifier are now
implemented and locally tested. The first production campaign later reached optimizer
step 13,150/20,000 before an observational stdout `BrokenPipeError`; six complete
checkpoints through step 12,000 remain, but the campaign is formally incomplete and
published no selected checkpoint or candidate pool. A read-only reconstruction on the
frozen 2,656-episode validation manifest finds that the parent-grouped masked-cell CE
improves from 1.2439 at step 2,000 to 0.8713 at step 12,000, while full-mask one-pass
exact grids remain 1/664 at step 12,000. These diagnostics support conditional grid
completion, not an ARC solve-rate claim. See
`notes/results/m04a-r7-training-analysis.md` and
`results/m04a_r7_posthoc_evaluation_20260715.json`. Code-model and rule sources must
likewise be frozen and evaluated independently before heterogeneous complementarity
is tested.
Masked grid repair and the adaptive controller remain downstream of broader,
provenance-complete candidate coverage.

## Executable functional router

The repository now contains a runnable integration MVP in `src/afts_arc/hybrid/`:

- typed DSL/program synthesis and repair-seed replay;
- inference-only sparse categorical CA, D4, modal-background padding, and bounded
  local CA programs from the repository-root implementation;
- explicit, non-training callback boundaries for masked diffusion, code/LLM, and
  DiffLogic hard circuits;
- a demonstration-only representation router;
- whole-task demo-exact replay, MDL ranking, built-in hard-rule reconstruction,
  and an explicit attestation boundary for external verifiers;
- residual-directed global-color and high-support radius-one local repair.

The v2 path adds the previously missing online controller:

- immutable content-addressed blackboard states, typed actions, action results,
  and explicit STOP;
- a multi-dimensional normalized-compute ledger that reserves and charges every
  provider/repair action before execution;
- execution-failure and demonstration-residual compilation into legal local
  repair or cross-representation actions;
- parent-conditioned DSL shape/suffix resynthesis and operator-specific sparse
  CA policy families through the context-aware `act(...)` boundary;
- strict `(operator, parent) -> batch` frozen heterogeneous pools for
  matched-budget policy comparisons, with no cross-action fallback;
- post-hoc separation of full-pool, observed, and selected oracle coverage,
  plus pass@k, exploration recall, selection utilization, and unique repair
  recovery per compute.

Run it from the repository root with:

```bash
python arc_functional_transition_solver/scripts/afts_arc_hybrid.py path/to/task.json
```

The CLI uses the online residual controller by default.  Pass
`--controller static` to reproduce the v1 eager run-all orchestration.  Budget
flags (`--compute-units`, `--controller-steps`, `--provider-calls`,
`--repair-attempts`, and `--candidate-slots`) define the exact controller ledger.
See `notes/design/online-residual-controller.md` for action typing, accounting,
provider fidelity labels, and the matched-budget experiment contract.

The no-training matched-budget evaluator builds content-addressed action pools,
replays residual-first/fixed/round-robin/static/random and single-source
baselines, and scores public oracles only after each replay:

```bash
python arc_functional_transition_solver/scripts/afts_arc_online_matched_budget.py \
  /path/to/ARC-AGI-1/data \
  arc_functional_transition_solver/results/online_control_frozen_v1 \
  --split training --limit 32 --sample-seed 20260726
```

The first frozen evaluation is recorded in
`notes/results/online-control-frozen-v1-20260726.md`, with its content-addressed
32-task and 100-task summaries and pool manifests under `results/`.  The result
shows DSL/CA pool complementarity but does not support the current
residual-first controller or unique-repair hypothesis; see the report for the
claim boundary and next gate.

The v2/v3 follow-up is recorded in
`notes/results/online-control-frozen-v2-v3-20260726.md`.  Exact-first emission
and action-aware frozen replay reach all 14 selectable pool tasks on an
untouched disjoint 100-task block.  A same-task paired ablation keeps the
candidate pool identical while reducing coverage-aware control cost from 945
to 630 NCU.  All heterogeneous schedules nevertheless pass the same 14 tasks,
and residual repair adds no unique union coverage; the report therefore treats
candidate-language expansion, physical-cost accounting, and uniquely useful
typed repair as the next research gates.

The first scene-graph object/code expansion is recorded in
`notes/results/object-code-gate-v3-scene-ast-20260729.md`. On the frozen
offset-100 development slice it adds exactly 3/100 unique selectable tasks, but
raises union only to 12/100 and recovers 0/100 natural near misses. The provider
gate passes narrowly while the broader solver and repair gates fail, so learned
controllers and masked-neural claims remain frozen.

Unconfigured neural/code providers return explicit abstention receipts. The
default CLI never resumes M04a or DiffLogic training; programmatically supplied
callbacks are explicit trusted integration boundaries rather than implicit
model calls. External candidates are centrally route/budget checked, conflicting
artifact IDs are quarantined as a group, and DiffLogic callbacks must bind a
self-hashed circuit manifest to the blind task. The archived
DiffLogic artifacts are not treated as deployable checkpoints because they lack
the complete preprocessing/configuration bundle required for independent replay.
The implementation and trust boundary are documented in
`notes/design/functional-router-implementation.md`. This is implementation
evidence, not a new ARC accuracy result; any claim of improved task coverage
still requires a frozen evaluation run.

The query-blind visual-posterior follow-up is recorded in
`notes/results/visual-structure-bridge-20260809.md`. On 12 frozen, generated
ARC-GEN families, VARC supplies 8/12 raw and 7/12 selectable candidates while
the frozen legacy portfolio covers 0/12. A demo-derived structural second-view
selector nevertheless leaves pass@2 unchanged at 7/12, with zero recovery and
zero regression. The result is therefore a clean null for the current
structural selector: whole-grid visual hypotheses are complementary, but coarse
4-connected object-transition marginals do not reliably identify the useful
minority hypothesis. The controller remains frozen. Content-addressed raw,
pre-gold, replay, score, and failure-analysis artifacts are under
`results/visual_structure_bridge_20260809/`.

The next relational actuator gate is recorded in
`notes/results/relational-mask-arc2-confirmation-20260809.md`. After correcting
and replaying the freshness audit, the frozen bounded scene-AST construction
retains 0/139 eligible ARC-GEN families (134 clear no-neighbor outcomes and five
typed construction exclusions).
An opt-in relational-mask DSL then adds 0/50 exact or unique candidates on a
frozen ARC-AGI-2 public-training cohort; typed repair and equal-cost restart
both recover zero. A query-blind post-hoc audit shows that all three
pixel-agreement parents are background-dominated false near misses. The result
rejects this representation and near-miss metric, not the unexecuted LODO
posterior hypothesis. Visual execution and controller training remain frozen.
Artifacts are under `results/visual_relational_trace_repair_20260809/`.

The subsequent narrow candidate-frontier program is recorded in the
query-blind 48-task terminal failure matrix, a reference-scale NVARC/TRM
anchor, and the exposed visual relational-transducer development gate. The
failure matrix assigns all residual tasks to 18 parse failures, 14 missing
relations, 12 canvas incompatibilities, three legal-but-inexact programs, and
one insufficient AST. The static NVARC/TRM anchor reaches 92/100 strict top-10
oracle on a synthetic, generator-family-disjoint ARC-TGI confirmation cohort;
this is not an ARC-AGI score.

On the collision-free 49-task development amendment, the raw VARC/NVARC union
covers 46/49. A frozen same-canvas family containing exactly 24 D4/color
transducers adds the other three. Under equal charged native cost, a
query-blind visual-posterior allocation recovers 3 tasks and the pre-existing
cold allocation recovers 0; clearing the posterior term replaces 7/15 selected
tasks but leaves unique recovery unchanged at 3. The development signal is
therefore candidate-language and composite-allocation evidence, not posterior
performance attribution. The controller remains frozen until the unchanged 100-task gate
obtains at least five unique recoveries and strictly exceeds cold restart. See
`notes/results/relational-delta-v0.2-failure-matrix-20260810.md`,
`notes/results/nvarc-trm-arc-tgi-anchor-20260811.md`, and
`notes/results/visual-relational-transducer-development-20260811.md`.

Object–Program Workspace v1 then separated program novelty from output-candidate
novelty.  A 12-task controlled intervention passes typed execution, replay, and
matched-cost checks, but visual, residual-cleared, and shuffled arms all recover
12/12, so visual attribution is absent.  On the frozen outcome-exposed 100-task
ARC-TGI development cohort, the workspace finds three new program IDs but zero
new query-output content IDs and zero unique recoveries; 94 tasks have no
demo-exact relational child.  This closes selector-only recolor/erase v1 as a
representation/output-frontier null and keeps controller training frozen.  See
`notes/results/object-program-workspace-v1-20260811.md`.

Cognitive Workspace + Object-Graph Rewrite v2 next tested a content-addressed
goal/memory/failure workspace and a two-stage executable object-program bridge.
On the frozen 50-family ARC-TGI development cohort it creates one query-blind
novel output; that output is exact and unique over both the frozen baseline and
equal-cost cold arm.  This is below the preregistered 5/50 opportunity gate.
Forty-nine tasks have executable parents, but 48 cannot close a demo-exact
second stage, localizing the failure to representation and repairability rather
than routing.  The reserve remains unmaterialized and the controller remains
frozen.  See
`notes/results/cognitive-workspace-object-graph-rewrite-v2-20260811.md`.

Stateful Object-Graph Rewrite v3 replaces that lossy raster boundary with
persistent object/relation identities, node-level failure certificates,
single-node counterfactual rewrites, and affected-subtree replay. Controlled
tests validate the mechanism, but the frozen 50-family development gate finds
0/50 novel-output opportunities. A demo-only all-node audit shows that the
compiler-selected node improves only 1/196 parents versus 94/196 for an oracle
over legal nodes, while no node is demo-exact. The result is jointly a typed
diagnosis and action-language null; reserve materialization and controller
training remain closed. See
`notes/results/stateful-object-graph-rewrite-v3-20260811.md`.

The next bounded topology experiment inserts one finite
`Grid x Grid x Scene -> Grid` anchor-rasterized delta node.  Its controlled
semantics, content addressing, typed insertion, and replay pass, but the valid
dev50 run is a preregistered sensor null: only 1/50 tasks passes strict LODO,
0/50 creates a novel query frontier, and unique recovery is 0/50.  Exact
reachability rejects all 48 previously identified spatially conditioned
single-color parents, showing that independent anchor-color raster marginals
are too weak.  The active research direction now derives support as the causal
footprint of a typed intervention on persistent objects, relations, or an AST
node; it does not add another raster primitive on the exposed cohort.  See
`notes/results/anchor-rasterized-delta-v0.3-20260812.md` and
`notes/decisions/RDR-20260812-relational-causal-footprints.md`.

Provenance-aligned Relational Effect v0.4 implements the next minimal bridge:
persistent entity/relation IDs, exact cell lineage through supported crop/D4
traces, typed effect summaries, and strict-LODO failure cores.  Controlled tests
and byte-identical eight-worker replay pass.  On a fresh sealed reserve-v2
cohort, however, the frozen twelve-effect sensor reaches only 2/100 tasks from
one family, exactly tied by global and whole-component recolor.  G1 therefore
fails; query targets remain sealed, G2--G4 are not run, and the controller stays
frozen.  The failure matrix localizes 67/100 tasks to single-delta or
cross-demo role inconsistency and 24/100 to the finite effect language.  Exact
source provenance covers 97.76% of observed residual cells, so the next bounded
candidate must add typed multi-effect obligations and relational role binding,
not another post-hoc raster mask.  See
`notes/results/provenance-relational-effect-v0.4-reserve100-20260812.md`.

## Project map

- `brief/`: topic, contribution, and evidence contracts.
- `notes/literature/`: source-checked literature map.
- `notes/design/`: architecture, baselines, and experiment matrices.
- `notes/innovation/`: candidate framings and rejection decisions.
- `plan/`: paper routing, outline, and implementation milestones.
- `src/`, `tests/`: implementation, added only after the research contract is stable.
- `results/`: generated evidence; no claim may be upgraded without a backing file.
- `results/README.md`: status map distinguishing verified, incomplete, ineligible,
  and unrun artifacts.
- `trash/`: ignored staging area for removals; never publish or commit it.

## USRL paper-spec reproduction

An independent reconstruction of the CVPR 2026 Understanding and Solving
Reasoning Loop now lives in `src/afts_arc/usrl/`, with a bounded experiment
driver at `scripts/afts_arc_usrl_reproduce.py`. It matches the public 7M-class
tensor contract, separates strict unseen-task and paper-transductive protocols,
and records deterministic architecture, data, optimization, and cost receipts.

This is not presented as a reproduction of the paper's 47.2% pass@2: no official
code/checkpoint was available, several algorithmic details are unspecified, and
the committed ARC pilots use a 177K-or-smaller model for only 3,000 updates. See
`notes/design/usrl-paper-spec-reproduction.md`,
`notes/literature/usrl-and-recursive-reasoning-20260726.md`, and
`notes/results/usrl-cvpr2026-reproduction-20260726.md` for the exact claim
boundary and results.

## Non-goals

- claiming a biologically faithful model of cortical areas;
- using a fixed task taxonomy as the main solver;
- making DLGN directly emit arbitrary output grids;
- tuning repeatedly on the public evaluation set;
- reporting leaderboard, accuracy, latency, or efficiency claims without provenance.
