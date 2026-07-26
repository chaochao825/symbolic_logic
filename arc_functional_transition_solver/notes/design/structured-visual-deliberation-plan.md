# Structured visual deliberation for ARC: pre-implementation contract

Date: 2026-07-26

## Decision

The five papers do not justify adding a longer free-form chain of thought to the
current ARC solver.  Their transferable common denominator is narrower and more
testable:

1. compile perception into a concise, queryable intermediate state;
2. separate stable factual state from diverse hypothesis search;
3. make the next action depend on typed step prerequisites and observed failures;
4. use boundary/counterexample cases to allocate exploration;
5. execute hypotheses in a formal language and verify the result.

The current solver already has (3) and (5) at the system level, but its factual
state is not summarized as an explicit visual cache, its policy does not expose
an adaptive grounding/diversity decomposition, and its DSL has weak object-level
predicate coverage.  This experiment therefore adds two separable components:

- a controller-only structured-deliberation policy over a content-addressed task
  sketch; and
- an independently ablatable object-predicate DSL provider.

No neural model is trained, downloaded, or called in this experiment.

## Primary-source mechanism audit

### Hilbert-Geo

Primary sources: [CVPR paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Xu_Hilbert-Geo_Solving_Solid_Geometric_Problems_by_Neural-Symbolic_Reasoning_CVPR_2026_paper.pdf),
[official code](https://github.com/PremiLab-Math/Hilbert-Geo).

The useful mechanism is Parse2Reason: text and diagrams are parsed into a shared
Conditional Description Language, followed by theorem-bank search.  The released
system contains 120 predicates and more than 200 theorems.  Its reported errors
are concentrated in formalization distortion and combinatorial search, not in
the deterministic theorem executor.  The transferable ARC lesson is a typed
scene predicate/action language with independent parse and execution receipts.
The geometry-specific predicate inventory and 45-shot MLLM parser are not directly
transferable.

### Step-CoT

Primary sources: [paper](https://arxiv.org/abs/2603.13878),
[official benchmark/code](https://github.com/hahaha111111/Step-CoT).

The useful mechanism is a graph of clinically meaningful steps with a global
memory node.  The teacher writes a compact step prediction back through a gated
memory update, and removing memory causes the largest reported ablation drop.
The transferable ARC lesson is to expose phase/prerequisite state and retain a
compact cross-step cache.  The seven radiology steps and expert labels are domain
specific and must not be hard-coded as ARC task classes.

### Summary-Driven RL / structured video CoT

Primary source: [paper](https://arxiv.org/abs/2603.25942).

The method separates `Summarize -> Think -> Answer`.  CVK concentrates summaries
around a factual anchor, while DVR increases reasoning diversity in proportion
to uncertainty and multiplies it by `1 - group_accuracy`.  The transferable ARC
lesson is to keep observable task facts stable while increasing representation
diversity only when verified support is weak.  Token-level GRPO on 32 A100 GPUs
is neither necessary nor a fair baseline for the present deterministic solver.

### Visual Thoughts

Primary source: [NeurIPS paper](https://arxiv.org/abs/2505.15510).

The paper compares natural-language descriptions, scene graphs, edited images,
and generated images.  Its central empirical claim is that concise, clear visual
thoughts act as a cache that carries task-relevant image information deeper than
repeatedly consulting the raw image.  The transferable ARC lesson is a compact,
content-addressed scene/residual sketch, not prose.  The paper did not provide
open code at publication time, so its mechanism should be treated as motivation,
not imported as a reproduced result.

### Agile Deliberation

Primary source: [paper](https://arxiv.org/abs/2512.10821).

The framework first scopes a concept into positive/negative subconcepts, then
retrieves semantically borderline examples, clusters coherent ambiguities, uses
UCB to allocate attention, and greedily refines the definition while preserving
performance on all prior labels.  The transferable ARC lesson is to treat
representation disagreement and residual clusters as an exploration frontier.
Its claimed gains come from 18 human sessions on two subjective concepts and do
not establish an autonomous ARC method.

## Competitiveness assessment before implementation

The current method has a stronger verification and systems substrate than these
papers for ARC: candidates are heterogeneous, content-addressed, replayable,
budgeted, demo-exact, MDL-ranked, and hard-rule checked.  None of the five papers
provides this exact cross-representation, matched-pool controller protocol.

It is not currently competitive as an ARC solver.  On the prior untouched
100-task block, selectable union coverage and pass@2 were 14/100; all seven
heterogeneous policies reached the same 14 tasks, and unique repair recovery was
zero.  This identifies candidate language coverage as the main bottleneck.  A
controller-only gain can improve anytime efficiency, but cannot exceed a frozen
pool ceiling.

The defensible research opportunity is the conjunction:

> Typed visual-state summaries and residuals control a budgeted, heterogeneous,
> formally verified candidate portfolio; stable grounding and adaptive branch
> diversity are measured separately from candidate-language expansion.

## Frozen hypotheses and ablations

H1 -- summary grounding: a content-addressed task sketch plus phase-compatible
action scoring improves pass-versus-NCU area under tight budgets, without changing
the candidate pool.

H2 -- adaptive diversity: rewarding untried representation families when the
observed demo-verification rate is low improves observed selectable-pool recall
under tight budgets.  It need not improve the full-budget endpoint.

H3 -- predicate language: an object-predicate DSL containing explicit selectors
and verified object actions adds selectable oracle coverage beyond the existing
grid DSL and sparse CA on a disjoint task block.

H4 -- combined method: structured deliberation plus the predicate provider has
higher final pass@2 than the original two-provider controller at the same NCU
limit.  Failure to beat the base union falsifies the capability claim even if
the new controller is more efficient.

Controller variants, all oracle-free:

- existing `coverage_aware_v2`;
- `summary_grounded` (task sketch and phase grounding only);
- `adaptive_diversity` (dynamic representation diversity only);
- `structured_deliberation` (grounding + diversity + borderline/UCB frontier).

Provider ablations:

- existing typed grid DSL + sparse CA;
- object-predicate DSL only;
- all three providers.

## Object-predicate DSL scope

The first slice is deliberately generic and bounded.  A rule contains:

- parse predicates: background, 4/8 connectivity, single- or multicolor objects;
- selectors: all, area extrema, singleton, hole/no-hole, filled rectangle,
  border/interior, literal color, or literal area;
- actions: keep, erase, crop, recolor, fill/outline bounding box, or fill holes.

Rules are enumerated from demonstration/query-input parse domains and
demonstration-output colors, retained only when every demonstration is exact,
and reloaded by a hard verifier before selection.  Query outputs are never read
during synthesis or control.  This is a small theorem/action bank, not a claim
to reproduce Hilbert-Geo's full language.

## Data and decision protocol

- Existing positions 1--200 of the SHA-256 order are development data.
- Positions 201--300 have already been viewed and are validation/diagnosis data.
- Positions 301--400 are reserved as the new final block and must not be inspected
  until code, policy weights, action space, budgets, and task IDs are frozen.
- Primary cap: 11 NCU, four steps, three provider calls, one repair, seven slots,
  provider batch size two.
- Tight-budget sensitivity is configured before the final launch; it is not used
  to tune after viewing final outcomes.
- Every controller comparison uses the same action-frozen pool.  Provider ablation
  is reported separately because changing the language necessarily changes the
  pool.
- Primary metrics: selectable pool coverage, observed selectable coverage,
  pass@2, NCU, pass-vs-NCU AUC, pool utilization, unique provider contribution,
  and unique repair recovery.
- A positive method result requires new disjoint-block pass@2, not merely more raw
  candidates, lower NCU, or recovery already present in another provider.

