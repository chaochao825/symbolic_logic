# ARC-CA experiment contract (frozen before public-evaluation scoring)

## Question and evidence boundary

This extension asks whether a finite, spatially shared cellular-automaton
transition can be induced from ARC demonstrations, compiled without semantic
loss into Boolean gates, and executed with a useful description/runtime
trade-off.  It does **not** assume that a fixed lattice can solve arbitrary ARC
tasks, that a hand-written CA proves rule learning, or that a gate-count proxy is
hardware speed.

The primary static-grid benchmark is ARC-AGI-2 at commit
`f3283f727488ad98fe575ea6a5ac981e4a188e49`.  ARC-AGI-1 release `v1.0.2`
(`aa922be204204ec148a1137fe6ed4d34ddde812b`) is a historical diagnostic only.
Dataset files are not vendored; every run records repository commit and content
hashes.  Apache-2.0 licenses and upstream changelogs remain authoritative.

## Frozen task ladder

The mechanism suite is ordered by required computation, not by observed score:

1. `L0`: one-step local rules (identity, recolor, shift, Boolean local LUTs);
2. `L1`: bounded morphology (dilation, erosion, local denoising);
3. `L2`: iterative propagation and fixed-point control (flood fill, wavefront);
4. `L3`: global aggregation (parity, count, majority, component selection);
5. `L4`: output-shape change (crop, resize, tile, object extraction);
6. `L5`: task-conditioned/composed rules whose required context is absent from
   the cell state;
7. `N`: incompressible controls (random local LUT and random global mapping).

The ARC solver uses the following fixed candidate order.  It receives only a
task's demonstration pairs during selection:

- center-only categorical map;
- every single offset in the radius-two square;
- radius-one von Neumann and Moore neighborhoods;
- radius-two von Neumann and Moore neighborhoods.

A second, independently reported program family contains identity, every
radius-two copy wire with center/modal-color boundary handling, and monotone
4/8-neighbor color propagation with horizons `1,2,3,4,8` or fixed point.  Its
parameters are enumerated only when demonstrations contain one changed color
pair.  It is not merged silently with the sparse-table score: both pass@1
results and their two-family pass@2 lower bound are reported.

A sparse-table candidate is admissible only when one shared deterministic table reproduces
all demonstration outputs exactly and all demonstration input/output shapes
match.  Ties are decided by leave-one-demonstration-out exactness, then cell
accuracy, then a declared prefix upper bound and a stable lexical name.  An
unseen test neighborhood falls back to the center color and is counted.  There
is no inspection of test outputs during candidate selection.

The post-hoc `train+test oracle representable` field is diagnostic only: it
separates representation failure from few-shot induction failure and is never a
submitted prediction.

The bounded program family is selected independently by exact demonstration
fit, then program-description bits, per-cell gate upper bound, and lexical
name.  It does not inherit the sparse table's leave-one-demonstration-out score.

## Public evaluation discipline

ARC-AGI-2 public training may be used for implementation diagnostics.  Solver
source, candidate grammar, codec, task manifest, seed policy, and output budget
must be committed and present on `origin` before the public evaluation split is
scored.  ARC-AGI-2 evaluation cannot run without a clean full-split receipt: it
checks the frozen 120-task ID/content digests and clean dataset worktree,
reserves an exclusive external JSONL ledger before scoring, saves both attempted
grids, and checks that source HEAD remains unchanged.  This is a declared
one-shot; it cannot prove that public labels were never viewed outside this
program.  Subsequent edits may repair reporting or verified
implementation bugs, but may not tune the solver from per-task evaluation
outcomes without invalidating the run.

Pass@1 is the primary metric.  Pass@2 uses the first two distinct predictions
under the same frozen demonstration-only ranking; a duplicated grid is not
counted as a second candidate.  Full task exact requires every test pair to match.  The all-task denominator retains
shape-changing and unsupported tasks; supported-subset scores are secondary.

## Gate compatibility and accounting

Every selected sparse categorical rule in a full run is executed three ways;
smoke runs may declare and report an exception-entry budget:

- direct categorical table;
- 4-bit binary equality/multiplexer construction;
- 10-state one-hot equality/multiplexer construction (plus an input-only
  boundary symbol).

The two compiled paths must equal the direct path exactly on every evaluated
cell.  Complete finite valid domains are enumerated for the small synthetic
rules; ARC claims are limited to observed demonstration/test neighborhoods.
The report records compilation coverage as a denominator, invalid decode rate,
rule entries, state bits, an explicit
gate upper bound, description bits, wall-clock time, and test support coverage.
No minimum-circuit claim is made.

Executable model prefixes include rule grammar, neighborhood wiring, mapping
entries, boundary, and horizon/stop policy.  The shared eight-bit CA container
prefix declares dimension (one bit), boundary policy (two), update schedule
(one), recurrence class (one), and lattice-interface family (three).  ARC
failure rows do not encode residual outputs, so `total_description_bits` is a
model-prefix upper bound rather than an end-to-end two-part MDL score.  Codec
choice and state storage are deployment representations reported in separate
columns; they are not silently folded into the shared model-prefix bit count.
Recurrent work is reported as
`active gates x updated cells x steps`; it is never replaced by model size.
For binary synthetic controls, a raw LUT stores the 2-state interior truth
table while the same declared boundary-to-zero wrapper is charged to both the
raw and structured routes.  Description-route decisions compare bits with
bits; gate counts are retained only as execution-cost proxies.

## Predeclared failure attribution

- soft/direct succeeds but hard compilation differs: codec or compilation bug;
- demonstrations fit but test fails with high unseen support: induction/data
  coverage bottleneck;
- post-hoc local oracle is impossible: locality/state/topology bottleneck;
- shape changes: fixed-lattice interface bottleneck;
- exact gates are slower than the direct program: execution bottleneck;
- random local LUT routes shorter to raw storage: compressibility bottleneck;
- long monotone propagation succeeds only with larger horizon: iteration/stop
  bottleneck.

## Required artifacts

- `arc_ca_task_results.csv`: one row per task, including all refusals;
- `arc_ca_program_results.csv`: bounded copy/propagation program search;
- `arc_ca_synthetic_results.csv`: L0--N mechanism controls;
- `arc_ca_codec_results.csv`: paired direct/binary4/one-hot equivalence and cost;
- `arc_ca_summary.csv`: all-task and supported-subset metrics;
- `arc_ca_predictions.jsonl`: auditable per-input attempt grids or explicit abstentions;
- `metadata.json`: source/data hashes, environment, frozen configuration, and
  public-evaluation receipt;
- unit, artifact, and clean-clone tests plus independent read-only review.
