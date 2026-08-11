# Stateful Object-Graph Rewrite v3 result

## Decision

The preregistered exposed-development representation gate fails. Preserving
executable state and applying a typed single-node counterfactual produces zero
query-blind output opportunities on 50 frozen ARC-TGI families, below the 5/50
threshold. The family-disjoint reserve remains unmaterialized and the
controller remains frozen.

The implementation does establish the requested interface as a testable
mechanism:

```text
executable trace
-> persistent object/relation identities
-> node-level typed failure
-> counterfactual state rewrite
-> affected-subtree re-execution
-> exact verification
```

Controlled tests verify stable identities, split/merge ancestry, strict
single-node legality, unchanged upstream state IDs, suffix-only replay, bridge
necessity, AST round trips, and equality with both fresh stateful execution and
the historical scene executor. Those are actuator and infrastructure results,
not natural ARC recovery evidence.

## Reproducibility and boundary

- Main source commit: `7e66cdb1b0cdae8a65809f0cd8d39e4de90fa4e9`.
- Failure-audit retry source commit:
  `7f9bee4e9def2c6a6deff6eca36ea81331dca6fa`.
- Frozen v2 incumbent ID:
  `79816f8458b39d6b29837d2d62eb1823e3127b7925cc811f86dd56616cf866de`.
- v3 candidate freeze ID:
  `66fb434846843584cde0198be17af637fb9af2d61521e9699ecb63c1f5712269`.
- Paired freeze files are byte-identical with SHA-256
  `7d8d0a6fff29b55ffe5dbd5cbdbcb1c40d62dfe194cc22a810ed166340e2432c`.
- Paired score files are byte-identical with SHA-256
  `82ff4d5adafd250fa2f47326c350aa4de053c2da9f71ebada3b83bba57cb7e2a`.
- Candidate construction and the post-hoc failure audit read no query gold.
  Development query gold was opened only after both candidate freezes for
  diagnostic scoring. No public evaluation data was read.
- The adjacent v3/v2/workspace/publication regression selection passes 60/60.
  The complete suite reports 575 passed, three skipped, and the same six
  pre-existing M04a evidence failures caused by missing protected
  sidecar/cache/source-snapshot payloads. It is not a green full-suite result.

## Gate result

| Metric | Result |
|---|---:|
| tasks | 50 |
| query-blind novel-output opportunities | 0 |
| required to continue | 5 |
| complete v2 incumbent exact | 2 |
| v3 rewrite exact / pass@2 | 0 / 0 |
| unique over incumbent and cold | 0 |
| equal-cost cold exact | 0 |

Forty-nine tasks retain four execution-valid parents; one has none. All 49
reachable tasks terminate with legal rewrites but no demo-exact candidate. The
main compiler constructs 196 certificates: 180 target `assignment`, four
target `canvas`, and 12 target `operate`. No query execution is charged for the
rewrite arm because no candidate passes the demonstration-exact guard.

Candidate and cold arms each reserve 102,400 program trials, 403,456 demo
executions, and 122,880 query executions. The rewrite arm realizes 4,534 trials
and pads 97,866; cold realizes 92,570 and pads 9,830. This preserves the frozen
reservation comparison while revealing that the legal single-node
neighborhood is very small. It does not establish better unit-compute repair:
both numerator and unique recovery are zero.

## Failure audit: compiler versus action language

After classifying the main result, a demonstration-only audit enumerated every
legal node, holding each parent and the 512-trial-per-node bound fixed. Across
196 parents and 7,446 trials:

| Diagnostic | Parents/tasks |
|---|---:|
| compiler-selected node reduces mismatch | 1/196 parents |
| at least one legal node reduces mismatch | 94/196 parents |
| selected node tied for best | 99/196 parents |
| selected-node demo-exact | 0/50 tasks |
| alternative-node demo-exact | 0/50 tasks |
| any-node demo-exact | 0/50 tasks |

Canvas rewrites improve 87 parents, parse rewrites improve ten, and assignment
rewrites improve one; operate and render improve none. Yet no node produces a
demo-exact program. This supports two simultaneous conclusions:

1. **Typed diagnosis failure.** The residual compiler over-assigns failures to
   `assignment` and usually fails to select the most useful intervention site.
2. **Single-node frontier failure.** Even an oracle node choice cannot close a
   demonstration. The parent grammar and one-node delta set lack the required
   joint transformations.

Therefore it would be incorrect to explain the null only as a routing error,
or only as out-of-language candidates. Persistent state removes the v2 raster
information bottleneck, but does not by itself create correct causal state
variables or sufficiently expressive transitions.

## Engineering incident

The first failure-audit attempt terminated on a render wider than the ARC grid
limit. The historical executor treats that case as an invalid program, whereas
the new executor leaked `GridValidationError` across its boundary. This was an
implementation defect in the diagnostic path. The failed attempt is retained
with status 1 and stderr SHA-256
`7e8ad829b692acd9ad545e5c031f9beabff438790bebe824d7169890ebed0af1`.

The executor now translates exactly that validation exception to the existing
typed invalid-execution state. A regression test covers oversized suffix
replay, and the retry completed with empty stderr and audit ID
`fd8f53b9ae263a61bdfa49df19c94cd8ea73f7c63fa4506ad21e004754d75fed`.
The preregistered main run had already completed without taking the failing
path, so this incident does not invalidate its zero-opportunity result.

## Consequence

Do not tune the compiler on these exposed 50 families and do not expand the
router. The next method gate must separate two preregistered questions on fresh
family-disjoint data:

1. can node-level certificates recover injected and naturally generated typed
   failures better than node-frequency and identity baselines; and
2. can a bounded multi-node state transition (for example parse plus canvas,
   or assignment plus operation) produce exact novel outputs under the same
   reservation as cold restart?

Only a positive first result licenses a learned residual selector. Only a
positive second result licenses a richer planning policy. The current evidence
supports the value of an auditable shared execution state, but not the claim
that the present state variables implement causal functional switching.
