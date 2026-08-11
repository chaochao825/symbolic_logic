# Object–Program Workspace v1: controlled pass, natural frontier null

## Decision

Object–Program Workspace v1 is closed as a **valid mixed/negative gate**.  It
establishes that a typed selector repair can be executed, replayed, intervened
on, and charged fairly when the required object representation is present.  It
does not establish a useful natural candidate frontier or causal utility from
the frozen visual posterior.

The long-term objective remains brain-inspired functional specialization and
switching.  This experiment isolates one necessary bridge in that objective;
it does not test a learned controller, a biological claim, or ARC-AGI
leaderboard performance.  GRU, MLP, bandit, and diffusion-router training
remain frozen.

## Frozen mechanism

The opt-in `afts-object-program-workspace-dsl/v1` adds:

- deterministic multi-parse object graphs over background, connectivity, and
  grouping hypotheses;
- n-ary shape, size, topology, alignment, containment, border, and relation
  profiles;
- typed parse/correspond/effect/render traces;
- complete-object `recolor_component` and `erase_component` effects;
- one-slot `object_rematch` actions that may change only
  `ast.correspond.selector`;
- a joint visual bridge that maps a whole output sample to a complete selected
  object set or rejects it, without pixelwise marginal synthesis; and
- separate checks for a new program ID and a genuinely new query-output
  content ID.

Every arm reserves four native program trials with matched demonstration and
query executions.  `visual_typed`, `residual_cleared`,
`posterior_shuffled`, and `cold_restart` share the same executable grammar and
budget.  Existing providers, identifiers, checkpoints, and historical results
are unchanged.

## Lane A: controlled semantic intervention

The 12 deterministic tasks contain a legal relational child by construction.
They test the actuator and intervention contract, not visual generalization.

| Metric | Result |
|---|---:|
| tasks / program opportunities / output-frontier opportunities | 12 / 12 / 12 |
| base-prefix / complete-grammar strict coverage | 6 / 12 |
| visual / residual-cleared / shuffled / cold strict coverage | 12 / 12 / 12 / 5 |
| unique over base: visual / cleared / shuffled / cold | 6 / 6 / 6 / 2 |
| visual vs cleared / shuffled selected-set changes | 11 / 11 |
| visual unique over every control | 0 |
| typed recovery rate | 1.0, above the frozen 0.8 gate |

Candidate freeze A/B and scoring result A/B are byte-identical.  The controlled
actuator gate therefore passes.  However, clearing or shuffling the posterior
does not reduce recovery, so this lane is a **sensor boundary**, not evidence
that visual evidence caused successful repair.  By protocol it cannot license a
generalization claim.

## Lane B: outcome-exposed family-disjoint development

The already frozen 100-task ARC-TGI confirmation cohort was reused only as an
outcome-exposed development screen.  Candidate construction read
demonstrations, query inputs, and a previously frozen VARC posterior; solutions
were opened only after two byte-identical candidate freezes.

| Metric | Result |
|---|---:|
| no demo-exact relational child | 94/100 |
| no quality base parent | 3/100 |
| new-program opportunities | 3/100 |
| new-output-frontier opportunities | **0/100** |
| base-prefix / complete-grammar strict coverage | 7 / 7 |
| visual / residual-cleared / shuffled / cold arm coverage | 3 / 3 / 3 / 0 |
| unique recovery over base, every arm | 0 |
| visual unique over every control | 0 |
| visual vs cleared / shuffled selected-set changes | 0 / 0 |

The three typed arms solve three queries, but all three query-output content IDs
already exist in the base pool.  They are alternative programs for existing
candidates, not new candidate regions.  Consequently the preregistered minimum
of five demo-only output-frontier opportunities is not met, and prospective
Lane C is not authorized.

The lower per-arm coverage (3) than base-prefix coverage (7) is not a portfolio
regression: arm coverage counts only candidates emitted by that four-trial arm,
whereas base coverage audits the frozen base pool.  Their union is deliberately
not relabelled as repair.

## Failure attribution

This is not a canonical-run implementation failure.  A same-shape scoring bug
and an inefficient repeated scene extraction were corrected before the final
freeze; the canonical artifacts then passed byte replay, content-address,
typing, topology, leakage-lane, and matched-reservation checks.

The natural result is a representation/output-frontier null:

1. v1 can only change a correspondence selector while preserving parse, effect,
   and same-size canvas semantics;
2. 94 tasks have no demonstration-exact relational child in that language;
3. the three reachable relational programs reproduce outputs already present in
   the base pool; and
4. no visual ordering can create utility when the legal action set contains no
   novel output candidate.

Thus the experiment does **not** show that residual evidence is intrinsically
uninformative.  It shows that this certificate/action pair cannot translate the
available evidence into a useful new representation region.

## Claim ledger

Supported:

- multi-parse object states, typed one-slot edits, full-object writes, content
  novelty, deterministic replay, and native-cost matching work as specified;
- the controlled actuator is causally manipulable;
- program novelty must not be equated with output-candidate novelty.

Not supported:

- natural candidate-frontier expansion from selector-only recolor/erase;
- visual-posterior utility over clear, shuffle, or equal-cost cold controls;
- typed repair recovery on this 100-task development cohort;
- dynamic functional switching or a competitive ARC solver claim.

Unknown:

- whether an object graph rewrite language with input/output correspondence,
  move/copy/layout/canvas operations, and typed AST holes yields enough natural
  opportunities;
- whether visual posterior evidence becomes causal after such a frontier exists;
- prospective family-disjoint generalization.

The single next research decision is **revise the representation protocol**, not
train a controller.  A separately frozen v2 may test object-relation graph
rewrites, but it must first demonstrate at least five demo-only novel-output
opportunities on a fresh development lane before any visual allocation or query
scoring is run.

## Reproducibility identifiers

| Lane | Candidate freeze ID | Result ID | Freeze file SHA-256 | Result file SHA-256 |
|---|---|---|---|---|
| controlled | `937da81861d6b544670d25ff82cc7c80efe20513343aad9c9a72c6c67c3e023f` | `a40b9a180084b6a76d91dd55fe5279e9255c4545e932e72ae0e388b55ebbfe17` | `7f5f41a9bbad3cb3b91f8625fcb82a2d8bcc94c1d1db472d406ddfd99e6ae603` | `8c643950a26aff4e978ae78c42927f668d70f65eefc52bc9cf64781b078d1714` |
| exposed development | `835cfc0c8a51475c8942ddff461d708c10f361627ed0db360eb2aed5e6463ae0` | `dd4abc81975df8fcd603e4a686aadb32d301623d85026bf917b0194703d66eb3` | `de3f2d52489188d71c7c1c74baeb63587ec4671e5bafe7cfe4969a99642a21b9` | `82a9ab5c94926c3ac991e67d5c4b7e543a1a6ba9e078d881bfcb743c2986fec0` |

The public repository contains compact summaries and complete SHA-256 manifests.
Task payloads, solutions, posterior samples, and candidate freezes remain
manifest-only in the publication bundle.
