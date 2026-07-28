# Object/code failure-control gate v1

## Claim boundary

This phase tests a narrower claim than the original brain-inspired portfolio:

> A demonstration failure can be compiled into a typed, replayable action that
> changes an object/program search frontier, and the resulting repair can be
> compared with a cold restart under the same native search reservation.

It does **not** train a controller, add a neural provider, or support a biological
claim. Existing controller trajectories, residual/state schemas, candidate IDs,
and result artifacts remain frozen. The new implementation is opt-in.

## First representation slice

`afts-object-code-dsl/v0.1` contains two multi-stage program families:

1. `role_stamp`: bind background, payload, source-anchor, and target-anchor
   roles; extract the payload relative to the source; apply one D4 transform;
   stamp at every target; render on a typed canvas.
2. `d4_label_completion`: parse connected structural objects; group them by a
   canonical D4 shape; bind the uniquely most-annotated prototype; use existing
   annotations to disambiguate correspondence; fill only background cells.

Both families are synthesized from demonstrations and observable query inputs.
They never receive query outputs. Each candidate serializes its complete program,
is content addressed, is reconstructed by its verifier, and must replay on the
whole task.

The diagnostic lane may retain a candidate that executes on only a subset of
demonstrations when its aggregate observed agreement still clears the configured
near-miss threshold. Completely unsupported invalid programs remain excluded;
partial execution failures are kept so the natural repair audit is not restricted
to pixel residuals by construction.

## Failure certificate and legal actions

The versioned certificate binds the blind task hash, parent candidate, diagnosis,
affected AST slots, compact demonstration evidence, and one legal action:

| Diagnosis | Action | Frontier change |
|---|---|---|
| object/role mismatch | `object_rematch` | vary object roles, D4 registration, or prototype binding |
| canvas shape mismatch | `canvas_reinfer` | vary only the canvas/render slot |
| declared provisional slot | `fill_ast_hole` | vary only named AST holes |
| background role mismatch | `reparse_background` | vary only the background parse |

The compiler does not run counterfactual search while diagnosing. Search work is
therefore visible in the subsequent action ledger rather than hidden in the
residual compiler.

## Negative-result attribution

The experiment classifies failures in this order:

1. **Implementation/protocol failure**: a generator-known control, type check,
   content hash, replay, leakage invariant, or matched-cost assertion fails.
2. **Search-budget failure**: the declared trial cap truncates the finite grammar
   before a demo-exact program is reached.
3. **Candidate-language failure**: the complete v0.1 grammar has no demo-exact
   program.
4. **Inductive ambiguity/generalization failure**: demo-exact programs exist but
   none predicts the public query output.
5. **Selectability/verifier failure**: an oracle query output exists among raw
   candidates but cannot pass demo-exact and hard-verification gates.
6. **Typed-repair failure**: controls and candidate coverage pass, but repair does
   not beat an equal-reservation cold restart.

Low coverage alone is not interpreted as proof that residuals are uninformative.
It rejects only this representation/compiler/action instance.

## Sequential hard gates

The canonical run reuses the exact task IDs from an immutable baseline summary.
For every task it writes the candidate pool before scoring public query outputs,
then repeats synthesis and replay to verify identity. Repair and cold-restart
pools are also frozen before query scoring.

Controller and neural work remain frozen unless all conditions hold:

- selectable baseline-plus-provider union at least 25/100;
- new provider unique selectable coverage at least 3/100;
- 100 natural near-miss cases are available;
- typed repair has at least 5% unique recovery and beats equal-reservation cold
  restart;
- controlled residual injections trigger all predicted action types;
- no implementation, replay, provenance, leakage, or cost-contract failure.

Observed program trials and demonstration executions must be identical between a
repair and its cold restart. Because synthesis normally executes query inputs only
for demo-exact programs, the lower-usage side is deterministically padded without
using predictions until both sides also consume their identical query-execution
reservation. Artifact replay and oracle scoring are reported as audit overhead;
no scalar cost comparison is made across unrelated provider types.
Natural cases whose typed frontier is empty remain in the recovery-rate
denominator as zero recoveries; only executable pairs contribute to the separate
typed-versus-cold recovery counts.
Raw rediscovery of an exact program already present in the frozen initial pool is
reported only as a diagnostic. The hard repair gate counts a recovery only when
the oracle-correct exact program is absent from that initial pool, so filtering or
reordering an already enumerated grammar cannot masquerade as a frontier change.
