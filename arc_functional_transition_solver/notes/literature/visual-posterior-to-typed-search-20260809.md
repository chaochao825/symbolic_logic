# Visual posterior to typed search: related-work synthesis

## Claim under study

The current evidence supports treating a query-blind visual posterior as a
diagnostic sensor over joint output hypotheses, not as a pixel actuator. The
candidate mechanism is therefore:

```text
whole-grid visual samples
  -> demo-derived structural contract
  -> candidate-conditioned typed certificate
  -> representation-specific search action
  -> demo-exact verification
```

The decisive distinction is between the joint posterior over complete grids,
`q(Y | X, D)`, and independently composed marginals. Pixel voting estimates
separate `q(Y_rc | X, D)` terms. Connected-object voting is less granular, but
still loses cross-object correspondence, topology, and shared program-node
dependencies. Neither operation is guaranteed to produce a sample from the
support of the original joint posterior.

## What the closest work contributes

| Work | Relevant mechanism | Constraint for this project |
|---|---|---|
| [VARC](https://arxiv.org/abs/2511.14761) | Vision-only image-to-image modeling, per-task test-time training, and multi-view inference | Preserve complete outputs: VARC regards two views as agreeing only when their entire grids are identical and ranks whole-grid modes. Pixel consensus is not a faithful reproduction of its inference rule. |
| [VLSR/MSSC](https://arxiv.org/abs/2511.15703) | Vision for global pattern abstraction and verification; language for rule formulation and precise execution | Use visual uncertainty to diagnose or verify a symbolic hypothesis, not to replace exact execution. |
| [Loop-OWM](https://arxiv.org/abs/2606.12316) | Color-prototype slots and composable transitions over visual-symbolic object state | The next representation should expose object slots and relations; independent connected components are insufficient. |
| [Synthesize, Execute and Debug](https://proceedings.neurips.cc/paper/2020/hash/cd0f74b5955dc87fd0605745c4b49ee8-Abstract.html) | Candidate program generation followed by execution-conditioned debugging | An `AST-hole` certificate is valid only when it references an executable candidate and a localized trace failure. |
| [Kintsugi](https://arxiv.org/abs/2605.09487) | Localized typed edits to executable knowledge, admitted by deterministic verification | Typed actions need type checks, protected-regression checks, and measurable frontier change; a plausible diagnosis alone is not repair utility. |
| [Modality-Driven Search](https://arxiv.org/abs/2606.31543) | Independent modality search and holistic comparison can recover minority hypotheses | Preserve modality diversity and compare complete hypotheses. Iterative prescriptive refinement can collapse diversity, so the bridge must not overwrite the raw pool. Its reported performance remains a recent preprint result. |
| [Tycho](https://arxiv.org/abs/2607.28287) | Explicit decisions to build, repair, use, or bypass an executable model | Better transition reconstruction does not imply better action utility. Our endpoint must be pass@2 or repair-per-cost, not certificate fidelity alone. This is ARC-AGI-3 and a recent preprint, so it is conceptual evidence rather than an ARC-AGI-2 baseline. |
| [ARC Prize 2025 report](https://arxiv.org/abs/2601.10904) | Refinement loops and feedback-guided per-task optimization emerged as a common successful pattern | Feedback is useful only when the hypothesis language covers the task and feedback changes the reachable search region. The report also warns that apparent performance can be constrained by knowledge coverage and contamination. |
| [ARC-GEN](https://github.com/google/ARC-GEN) | Procedural generation of task-family episodes, including ARC-AGI-2 support | Use family-controlled generated episodes for intervention and near-miss tests, while stating that generated-family results are not a substitute for hidden benchmark generalization. |

## Difference from the original portfolio

The original system-level idea assigned heterogeneous modules different
inductive biases and trained a router to switch between them. The negative
controller experiments showed that the available residual token did not alter
the legal action distribution. The revised proposal changes the unit of
control:

- A certificate is conditioned on a concrete candidate, not just the task.
- It names the violated representation invariant and editable slots.
- Its action must alter the candidate frontier under a native cost ledger.
- Acceptance still requires demo-exact execution and protected replay checks.

This is stronger than static modality selection but narrower than a biological
brain-region claim. A functional-switching claim would additionally require
selective lesions, residual injection, cost intervention, and module insertion
effects.

## Frozen bridge v1

The first confirmation gate intentionally stops before repair. It keeps the
frequency-ranked whole-grid top-1 and chooses the second candidate from existing
complete visual samples by agreement with only those canvas, background, object,
and transition fields invariant across all demonstrations. A top-1 contract
violation compiles to exactly one of:

- `canvas_reinfer` for shape/canvas violations;
- `reparse_background` for a background-role violation;
- `object_rematch` for object, palette, topology, or transition violations.

It does not emit `fill_ast_hole`, because this visual provider has no executable
AST or node-level execution trace. It does not synthesize a new grid and cannot
raise raw oracle coverage. Its only possible solver contribution is improved
utilization of an already-covered minority hypothesis within pass@2.

## If the selector gate passes

The next experiment should be a separate leave-one-demo-out (LODO) typed-repair
gate:

1. For each demonstration, hide its output and obtain a visual posterior from
   the remaining demonstrations.
2. Execute a concrete object/code candidate on the hidden input.
3. Compare its execution trace with whole-grid samples and demo gold to localize
   the earliest violated object correspondence, canvas parameter, mask, or AST
   node.
4. Compile only type-legal edits to existing holes or parameters.
5. Measure unique recovery and native cost against an equal-cost cold restart.

The LODO gate should require natural and injected near-misses, at least five
unique recoveries per 100 near-misses, a positive repair-per-cost advantage over
restart, and nonzero novel-frontier count. Controller training remains frozen
until residual removal, shuffling, and targeted injection cause predictable
action changes and a dynamic policy gains at least three tasks per 100.

## If the selector gate is null or adverse

A null result means the current hand-engineered structural contract is too weak
to identify useful minority hypotheses; it does not refute the already observed
visual raw coverage or localization signal. An adverse result means even this
selector destroys useful frequency information and should be parked. Neither
outcome licenses trying more post-outcome weights on the same cohort. The next
independent research direction would be learned relational slots or a
candidate-conditioned object graph trained on procedurally injected failure
types, not another router.

## Evidence boundary

VARC, VLSR, Loop-OWM, Modality-Driven Search, Kintsugi, and Tycho are recent
preprints as of August 2026; their reported scores and mechanisms are cited as
author-reported evidence. The NeurIPS SED paper is peer-reviewed. The local
bridge conclusions must be based on content-addressed artifacts from the frozen
cohort, not on the external papers' endpoint claims.
