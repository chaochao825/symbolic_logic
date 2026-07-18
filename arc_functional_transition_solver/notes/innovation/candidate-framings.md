# Candidate Contribution Framings

All framings are hypotheses. A framing is promoted only after the stated
falsification test is run on a frozen protocol.

## F1. Residual-conditioned functional switching

**Claim.** On the pre-registered 150-task ARC-AGI-2 public-training holdout, a
preselected XGBoost next-action controller improves official two-attempt pair-fraction
score by at least 0.02 at a 180-second non-API budget, relative to the strongest
validation-tuned static weighted schedule, with a paired task-bootstrap 95% confidence
interval whose lower bound is above zero.

**Why it matters.** Existing systems show that candidate generation and refinement
work, but adding generators can waste compute or flood the selector with correlated
candidates. The allocation policy becomes the scientific object.

**Prior coverage.** Refinement loops, evolutionary synthesis, recursive masked
diffusion, modular routing, and multi-model judging already exist separately.

**Candidate gap.** A single, source-agnostic blackboard controller with exact
execution evidence, typed legality, matched-budget evaluation, and representation
switching has not yet been established as the cause of ARC gains.

**Evidence needed.** E06 is the primary test. Budget curves, source drops, other
controllers, and ARC-AGI-1 are secondary analyses and cannot rescue a failed E06.

**Falsifier.** The E06 point delta is below 0.02, its 95% interval includes zero, or
resource matching is violated.

**Decision.** Primary framing for the first implementation.

## F2. Dual-space denoising for local repair

**Claim.** Near misses are repaired more efficiently when both output grids and
functional/program traces can be selectively masked and reconstructed.

**Why it matters.** Grid diffusion can correct pixels but may not correct the causal
rule; program search can correct a rule but may struggle with perceptual uncertainty.

**Prior coverage.** ARChitects already perform recursive grid-level soft masking.
Program mutation and evolutionary refinement are also established.

**Candidate gap.** Coupled repair masks derived from the same verifier residual, with
cross-space transfer and exact recovery measurement.

**Evidence needed.** Matched-call repair versus restart experiments in both spaces,
plus a coupling ablation.

**Falsifier.** Independent restarts recover at least as many exact solutions, or one
repair space provides no complementary coverage.

**Decision.** High-value second-stage framing; too expensive for the first baseline.

## F3. Typed stochastic transition control

**Claim.** Dynamic type/precondition masks provide most of the search compression of
a fixed transition graph without its catastrophic false-negative ceiling.

**Why it matters.** Search must be pruned, but a missing legal edge makes the true
program unreachable.

**Prior coverage.** Typed DSLs and execution-guided search are established; modular
routers and options formalize function selection.

**Candidate gap.** Four-valued predicates, escapable hard masks, and recall-constrained
evaluation for ARC function switching.

**Evidence needed.** Fixed/dynamic/unmasked comparison with true-path recall and
expanded-node measurements.

**Falsifier.** The dynamic mask is dominated by unmasked search or loses rare true
paths.

**Decision.** Retain as a mechanism, not the main title claim.

## F4. DLGN as the ARC control plane

**Claim.** A discretized logic-gate policy provides a useful recall-cost tradeoff for
early filtering or repair selection.

**Why it matters.** Large candidate pools require cheap decisions, and a hard logic
policy is inspectable and hardware friendly.

**Prior coverage.** DLGN efficiency is established on classification tasks, not ARC
control. Scaling and discretization limitations are known.

**Evidence needed.** Same-feature comparison with XGBoost, MLP, trees, and rules;
soft/hard agreement; CPU latency; model size; final solver effect.

**Falsifier.** DLGN is Pareto-dominated or silently prunes correct candidates.

**Decision.** Keep as an ablation. Do not make it the primary framing unless evidence
later shows a clear frontier advantage.

## Rejected framings

- **Task classifier selects a solver once.** Rejected because demonstrations define
  the task and later execution states can change the best action.
- **Fixed brain-area transition matrix.** Rejected because a single missing edge
  creates zero recall; the matrix must be conditional and escapable.
- **End-to-end LGN grid solver.** Rejected because variable shapes, binding, loops,
  parsing ambiguity, and an open rule vocabulary are outside its strongest regime.
- **Brain-inspired ARC.** Rejected as a paper claim; the biological analogy does not
  itself yield a mechanism or falsifiable neuroscience result.
