# Residual-Compiled Metareasoning: Theory and Pre-Registered Development Audit

Date: 2026-07-26
Status: development-only pre-registration, written before the new audit is run
Sealed evaluation: **not opened by this study**

## Decision

The project will no longer treat a list of modules as evidence for a
brain-like solver.  Its falsifiable core is narrowed to:

> Execution residuals and failures are compiled into legal cross-representation
> options.  A stateful controller allocates provider-native compute to those
> options only when their estimated value of computation is positive.

The neuroscience analogy is functional, not anatomical.  It is retained only
through five testable properties: specialist selectivity, residual-conditioned
coupling, compositional reuse, selective lesion effects, and resource-rational
switching.  Flexible-hub results motivate changing connectivity across task
states; they do not imply that an ARC provider is a biological brain region.

## Relation to prior work and novelty boundary

- [Flexible hubs](https://pmc.ncbi.nlm.nih.gov/articles/PMC3758404/) motivate
  task-dependent coupling among otherwise specialized systems.
- [Rational metareasoning](https://pmc.ncbi.nlm.nih.gov/articles/PMC5937797/)
  treats internal computation as a cost-benefit decision and permits strategies
  to be represented as options.
- [Option-Critic](https://arxiv.org/abs/1609.05140) supplies the language of
  temporally extended options and learned termination.
- [USRL](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_Human-like_Abstract_Visual_Reasoning_via_Understanding_and_Solving_Reasoning_Loop_CVPR_2026_paper.html)
  supplies a neural understanding-solving recurrence and adaptive halting, but
  not typed cross-language residual compilation or a physical-cost portfolio.
- The [ARC Prize 2025 report](https://arxiv.org/abs/2601.10904) identifies
  feedback-driven per-task refinement as a major trend.
- [Modality-Driven Search](https://arxiv.org/abs/2606.31543) shows that diverse
  text/image/code search operators can outperform a single channel and also
  reports that iterative refinement can reduce diversity.

Therefore, modules, recurrence, refinement, and routing are not standalone
novelty claims.  The candidate contribution is their specific coupling:

`typed residual -> legal cross-representation option -> immutable action DAG -> value-of-computation under native budgets`.

## Formal object

The controller operates on a metalevel semi-MDP state

\[
b_t=(\phi(T), C_t, R_t, F_t, H_t, B_t),
\]

where task features, content-addressed candidates, structured residuals,
failures, action history, and remaining native budget are all observable without
query labels.  An option is

\[
a_t=(m, op, parents, quota),
\]

plus `STOP`.  Its objective is

\[
r_t=\Delta\widehat U_2(C_t)-\lambda^\top c(a_t)
    -\kappa\mathbf 1[m_t\ne m_{t-1}],
\]

and it stops when every legal option has non-positive estimated value of
computation.  Types establish syntactic legality and executability, not the
truth of an underdetermined ARC rule.

## Evidence baseline that this study must not obscure

Across the two comparable 100-task ARC-AGI-1 training blocks already opened:

- heterogeneous pool coverage is 29/200 and structured pass@2 is 27/200;
- structured and grounding-only execute identical action traces on 200/200;
- structured beats fixed CA-first on 4 tasks and loses on 0, but this is not a
  sealed or compute-matched significance result;
- 76 repair actions yield zero uniquely recovered tasks;
- structured uses 371,439 scene-rule trials versus 9,002 for fixed CA-first.

The dominant bottleneck remains candidate language, not final selection.

## Phase P1 implemented in this study

### P1.1 Cost-carrying frozen actions

Every frozen action batch will carry a canonical vector of realized native work.
The vector remains multi-dimensional; no hidden exchange rate converts scene
trials, program expansions, GPU seconds, or model tokens into NCU.  This is an
audit prerequisite, not yet a deployable cost reservation model: realized cost
must not be used as inference-time lookahead on a sealed task.

### P1.2 Option-level residual-increment audit

Use the already opened offset 0-99 block for model fitting and the already
opened offset 300-399 block as an analysis test block.  This is development
evidence only.  Treat DSL, sparse CA, and scene DSL single-source runs as
temporally extended options.

Models:

1. best fixed two-option pair selected on the fit block;
2. shallow static decision tree using task facts only;
3. static gradient boosting using the same task facts;
4. dynamic gradient boosting using facts plus first-option outcome/history but
   excluding residual tokens;
5. the same dynamic model including residual tokens;
6. residual-ablated and residual-shuffled interventions on model 5;
7. oracle source scheduler, for diagnostic ceiling only.

The label is single-source selectable success and is used only for offline
training/evaluation.  Inference features contain no query output or oracle
field.  All methods receive exactly two option slots.  Native work is reported
as a vector and via a development-fitted normalization only for descriptive
efficiency; no matched-physical-compute accuracy claim is allowed in P1.

Primary mechanistic question:

> Does adding the observed first-option residual change the second option and
> improve held-out selectable option coverage beyond the same model with only
> static facts and non-residual history?

Report empirical
\(I(m^*_{2};R_1\mid\phi(T),m_1)\) with a stratified permutation null, paired
wins/losses, paired bootstrap confidence intervals, intervention-induced action
changes, native cost vectors, and all negative results.

## Pre-registered interpretation gates

- If residual and history variants choose the same actions, learned functional
  switching is behaviorally inert and no GRU/diffusion planner is justified.
- If residual changes actions but not coverage, the compiler has causal control
  but no demonstrated utility.
- If dynamic residual routing improves fewer than three tasks per 100 or its
  paired 95% interval includes zero, treat it as exploratory only.
- A main-paper controller claim requires at least 200 sealed tasks, at least a
  three-point gain over the strongest static schedule, and a 95% lower bound
  above zero under at least two physical budget definitions.
- Natural-ARC repair remains unsupported until it uniquely recovers at least
  five of 100 pre-defined root-pool near misses and beats an equal-cost cold
  restart.

## Next phases, conditional on P1

1. Build a parent/family-disjoint controlled near-miss benchmark and compare
   residual-conditioned versus blind, same- versus cross-representation repair.
2. Add reservation costs calibrated only on the fit block, then enforce native
   token buckets during frozen replay.
3. Improve candidate coverage with one credible masked-grid provider and one
   code/program provider; use certified and exploratory answer lanes.
4. Train a controller only after the new providers add stable unique selectable
   coverage.  Compare static tree, boosting, contextual bandit, MLP/GRU, then a
   discrete diffusion planner in that order.
5. Run residual injection, representation-bridge lesion, cost intervention, and
   unseen-option insertion before any claim of brain-like functional switching.

The existing ARC-AGI-1 training results are development data from this point
forward.  Final ARC evaluation remains one-shot and is outside P1.
