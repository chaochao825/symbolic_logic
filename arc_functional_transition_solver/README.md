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
