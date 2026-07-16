# DiffLogic-ARC v1 Experimental Contract

Status: development protocol amended after `DEV-LOW-MDL`/synthetic debugging;
`CONF-INDUCTION-HASH` remains unread and will be run only from an externally
hashed, source-commit-bound configuration.

The amendments made before confirmatory freeze are: preserving a visible-state
backbone through fixed wiring after a reachability failure, keeping gradients
alive at straight-through 0/1 values, making final-state loss the default,
adding an explicitly charged modal-background padding probe, restricting
adaptive horizons to the declared doubling schedule, and deferring every test
label evaluation until all tasks/seeds/circuits have been selected.  These are
development-informed fixes, not preregistered findings; all corresponding
development numbers are rerun from the clean implementation commit.

## Question and evidence boundary

This experiment asks whether a genuine differentiable-logic cellular automaton
can improve the failure layers identified by the frozen categorical ARC-CA
baseline.  It does not reopen or rerun the consumed ARC-AGI-2 evaluation
receipt.  All development and confirmatory tasks come from ARC-AGI-2 public
`training` at dataset commit
`f3283f727488ad98fe575ea6a5ac981e4a188e49`.

The model-facing API receives demonstrations and test inputs only.  Test
outputs remain evaluator-only.  Task cohorts below were constructed from the
already-published baseline diagnostics, so they are mechanism cohorts rather
than an unbiased full-ARC score.  Cohort membership may use held-out labels;
model fitting, seed selection, augmentation selection, horizon selection, and
prediction may not.

## Model family

The core operator is a fixed-wiring network of gates.  Gate `g` has logits
`theta_g` over all 16 two-input Boolean functions.  With temperature `tau`,

```
p_g = softmax(theta_g / tau)
y_g(a,b) = sum_k p_g[k] f_k(a,b).
```

Training uses soft mixtures followed by straight-through hard selections.
Deployment uses `argmax(theta_g)` and binary state only.  The public bit order
is the same 16-function order used by DiffLogic-CA.  Fixed connections are
coverage-balanced and seed-recorded; wiring is not learned.

ARC colors use four visible state bits.  Invalid hard codes 10--15 are counted
before a declared center-color fallback.  Recurrent variants add eight hidden
bits.  A CA update reads a `3x3` patch of dynamic state plus immutable binary
context at the center cell and directly emits the next state.

The staged additions are:

1. `DL1`: one step, no hidden state, same-shape canvas;
   `DL1-wide` is a capacity control with the same state and horizon;
2. `DLR`: shared recurrent rule, eight hidden bits, candidate horizons
   `1,2,4,8`, demo-selected smallest hard-exact horizon and fixed-point stop;
3. `DLO`: immutable connected-component raster, color-role, coordinate, border,
   input-mask, and task-context bits;
4. `DLS`: demo-only shape router and padded workspace, with the output crop
   fixed by the selected shape program rather than a held-out label;
5. `DLF`: the complete staged model.

Every neural variant has a matched MLP-NCA control with the same state,
context, recurrent horizon, optimizer budget, and decoder.  The frozen
`DemoSelectedLocalCA`, a D4-augmented sparse-rule control, and the bounded CA
program library remain non-neural baselines.

## Shape, objects, task context, and stopping

The shape router considers only predeclared programs: same shape, a constant
demo output shape, integer row/column scaling, and foreground bounding-box
shape.  A program is eligible only if it explains every demonstration and can
compute a test shape from its input.  Ties use declared prefix length then a
lexical name.  Unsupported shape relations abstain.

Connected components are deterministic input features, not learned ground
truth objects.  Per-cell context records foreground membership, component
boundary/anchor, small-size predicates, absolute coordinate bits, canvas
border, input/output masks, demonstration-derived background color, and shape
program bits.  This tests whether explicit object/layout state removes local
collisions; it does not claim task-independent object discovery.

Adaptive computation is demo-selected.  The solver chooses the smallest hard
horizon that is exact on all demonstrations; otherwise it chooses the horizon
with best hard demonstration cell accuracy and records that failure.  A hard
rollout may stop earlier only after the complete dynamic state is unchanged.
All executed steps and gate evaluations are charged.

## Data augmentation and candidate selection

`none` and dihedral `D4` augmentation are demo-selected candidates.  Explicit
background-padding and `D4+background-padding` probes test whether an ARC grid
edge should behave like the task's modal background rather than a distinct CA
boundary symbol.  Padding is applied identically to demonstration and test
inputs, predictions are cropped by the declared one-cell margin, and the
candidate abstains unless the cropped rule is exact on every original
demonstration.  These remain separately reported predictions.  The selector
uses leave-one-demonstration-out exactness, then cell accuracy, then declared
augmentation bits.  Neural seeds are ranked by hard demonstration exactness,
hard cell accuracy, active non-pass-through gates, and seed.  No test output is
used for routing.  The runner produces and freezes predictions for every task,
seed, horizon, and hard circuit before the evaluator receives any test-label
object.

## Frozen cohorts

`DEV-LOW-MDL` is explicitly diagnostic and may be used for implementation and
hyperparameter development:

```
9968a131 aabf363d 32e9702f 3aa6fb7a
aedd82e4 ba97ae07 4347f46a 25d8a9c8
```

`CONF-INDUCTION-HASH` contains twelve baseline-selected, post-hoc representable
induction failures chosen by the smallest
`SHA256("diffarc-v1:" + task_id)` values.  Its labels may be read only after a
confirmatory configuration is frozen:

```
a8d7556c 7ee1c6ea a834deea 1e0a9b12 a04b2602 b60334d2
68b16354 9caf5b84 3345333e bda2d7a6 cc9053aa 9b5080bb
```

Additional four-task probes use the same hash rule:

```
POSITIVE: bb43febb c8f0f002 84f2aca1 dc1df850
REPRESENTATION-GAP: fc754716 1b8318e3 e734a0e8 6e82a1ae
LOCAL-CONFLICT: 94be5b80 342ae2ed 465b7d93 a3f84088
SHAPE-CHANGE: 80214e03 ce602527 ae4f1146 9110e3c5
```

## Runs, seeds, metrics, and claims

Synthetic mechanism tests cover all 16 gates, one-step color rules, hidden
multi-step propagation, context-disambiguated rules, shape expansion, and
fixed-point stopping.  Training claims use seeds `0,1,2` and report each seed;
a smoke run is not a success-rate claim.

The primary ARC metric is hard full-task exactness.  Secondary metrics are
pair exactness, active-cell and all-canvas accuracy, demonstration exactness,
soft-to-hard drop, invalid-code rate, selected augmentation/horizon, fixed-point
status, active gates, state bits, dynamic gate evaluations, and wall time.

The following conclusions require the corresponding evidence:

- "true DiffLogic training works" requires optimization from initialized gate
  logits and a separately evaluated hardened circuit, not replay or table
  compilation;
- "induction improves" requires at least one confirmatory hard exact task above
  the frozen zero-exact cohort baseline, with no confirmatory retuning;
- "hidden recurrence helps" requires a task solved by `DLR/DLF` but not `DL1`;
- "object/context helps" requires a task solved by `DLO/DLF` but not the matched
  context-free model;
- "shape support helps" requires a shape-changing task with exact predicted
  shape and grid, not padded-canvas pixel accuracy alone;
- "hardening preserves behavior" is reported as a rate and never inferred from
  soft loss;
- CPU/GPU timing and abstract gate counts are not FPGA/ASIC PPA.

Zero improvements, hardening collapse, unstable seeds, and refused shape
programs remain reportable results.  ARC-AGI-2 evaluation remains untouched.
