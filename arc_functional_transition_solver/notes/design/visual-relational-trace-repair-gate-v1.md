# Visual relational trace repair gate v1

## Decision question

On fresh ARC-GEN families, can leave-one-demonstration-out (LODO) VARC
posteriors and an executable scene-AST parent trace select a type-legal set of
existing AST slots whose bounded edit frontier recovers a query more often than
both a posterior-lesioned typed edit and an equal-action-cost cold restart?

This is a mechanism feasibility gate on prospectively selected natural search
near misses. It is not an ARC-AGI-2 benchmark estimate, a new raw-coverage gate,
a learned-controller experiment, or evidence for a biological brain mechanism.

## Evidence and candidate

The preceding frozen gate established complementary visual raw coverage but
returned a valid null for a task-level structural ranker. Three of its four raw
coverage gaps had same-shape visual near misses at one, five, and nine pixels;
the fourth was a canvas failure. Pixel consensus, connected-object consensus,
and an unconditioned structural contract all produced zero unique recovery.

The active candidate therefore preserves complete visual samples and conditions
their interpretation on one executable parent program:

```text
LODO complete-grid posterior + executable parent trace
  -> relational difference groups supported by visible demo residuals
  -> legal existing-slot action
  -> content-novel single-slot programs
  -> demo-exact verification
  -> query-blind whole-grid selection
```

No pixel, object, or feature marginal is rendered into a grid. Connected
components are only nodes in a joint object-relation graph. Every emitted grid
is produced by replaying a serialized program.

## Fresh cohort

- Source: ARC-GEN commit
  `a15cbdb44c776610aeeb9f487a06af875d3d0878`, ARC-AGI submodule commit
  `399030444e0ab0cc8b4e199870fb20b863846f34`.
- ARC-AGI-1 exclusions must come from that locked ARC-GEN submodule. Both
  400-task splits are validated before selection; an empty, incomplete, or
  alternate identity source makes the run invalid.
- Episode seed: `visual-relational-trace-repair-v1-20260809`.
- Exclude every family/task ID named in prior authored evidence, including all
  31 visual-development and 12 structural-confirmation families, plus every
  ARC-AGI-1 family.
- Order eligible families by `sha256(seed + NUL + family_id)`.
- Generate three demonstrations and one query with deterministic per-pair seeds.
- Retain the first 12 families for which a deterministic 512-trial initial
  object/code search yields an execution-valid scene-AST near miss and at least
  one content-novel, one-existing-slot edit is exact on all demonstrations.
- Cap query-blind construction at 30 seconds and 100,000 scene programs per
  family. A cap hit is recorded as construction-budget exclusion, not a
  representation failure.
- Cohort selection may read demonstration outputs and query inputs. It may not
  inspect query outputs, VARC predictions, or final query scores.

This construction creates a query-blind repair-opportunity benchmark. Its
denominator must not be reported as general ARC coverage.

Each retained task creates four provider episodes: one full three-demo query
episode and three LODO episodes. A LODO episode trains on the other two demos
and predicts the held-out demo input. Every provider-visible test output is an
exact input-copy sentinel. Gold query files remain on the 210 server until the
candidate artifact and a byte-identical replay are frozen.

## Frozen mechanism

### Parent and graph evidence

The parent is the first retained scene-AST near miss under the existing score
order. It is serialized, content addressed, and replayed on every demo and query
input. Node traces must contain the six declared nodes: parse, correspond,
select, operate, canvas, and render.

For a parent output and one complete alternative grid, the compiler constructs
a candidate-conditioned joint relation graph using the parent's parse
connectivity/grouping and canvas background. It records differences in seven
groups:

- `canvas`: output height or width;
- `background`: observed modal background;
- `object_identity`: object count, D4 shape, size, or topology;
- `transform`: oriented shape with D4 identity preserved;
- `spatial_relation`: pairwise above/left/containment/distance relations;
- `palette`: object-role colors;
- `mask_render`: grids differ after the preceding graph contracts agree.

For each demo, only groups present in the visible parent-to-gold residual are
eligible. LODO support is the fraction of complete posterior samples whose
parent-to-sample graph contains that same group. The selected group maximizes,
in order: mean LODO support on affected demos, affected-demo count, full-query
posterior support, then a fixed group order. No learned weights or thresholds
are fitted.

The group maps to legal existing slots:

| Group | Action | Existing slots |
|---|---|---|
| canvas | `canvas_reinfer` | canvas mode, padding, height, width |
| background | `reparse_background` | parse/canvas background |
| object_identity | `object_rematch` | parse, correspondence, selection |
| transform | `object_rematch` | operation transform |
| spatial_relation | `object_rematch` | correspondence, selection, operation layout |
| palette | `object_rematch` | operation color, parse/canvas background |
| mask_render | `mask_rerender` | render mode and conflict policy |

Every proposed program must differ from the parent in exactly one listed leaf
slot. `fill_ast_hole` is illegal because the parent has no hole. An action may
claim `frontier_changed` if and only if at least one emitted program ID was not
in the 512-trial initial pool.

### Arms and cost

The provider posterior and parent diagnosis cost are shared and charged to all
arms. Three arms each reserve 16 program trials, `16 * demo_count` demo
executions, and `16 * query_count` query executions:

1. `visual_typed`: slots selected by LODO support;
2. `bridge_lesion`: the same compiler with visual support removed, using only
   visible residual frequency and the fixed group order;
3. `cold_restart`: a deterministic hash order over all content-novel scene-AST
   programs in the same representation, without the parent certificate.

If an arm has fewer than 16 novel programs, replaying the parent consumes the
unused execution reservation but does not create a candidate or frontier claim.
All actual trial programs are executed on all demos and the query input before
gold release. Only demo-exact programs are selectable. Up to two selectable
programs are ordered by exact complete-grid query-posterior support, then joint
relation-graph distance, description length, and program ID. The same selector
is used for all arms.

The matched comparison is conditional on the shared visual observation. Native
VARC GPU time and AST execution counts are reported separately; no scalar
conversion between neural and symbolic cost is allowed.

## Interventions and guards

- Deterministically rotate LODO posteriors across tasks and recompile
  certificates without executing new candidates.
- Remove LODO and query supports for the bridge-lesion arm.
- Verify all provider files use input-copy sentinels and no gold directory is
  present on 236.
- Verify source, generator, checkpoint, manifests, raw predictions, programs,
  traces, candidates, and scores by content hash.
- Rebuild the pre-gold artifact and require byte identity.
- Reject malformed visual samples; never repair or coerce them.
- Require identical reserved and observed program/demo/query execution counts
  across all three arms.
- Keep the controller frozen.

## Outcome mapping

- **Pass:** 12 valid tasks; zero leakage/replay/cost/frontier violations; at
  least three shuffled or lesioned certificate changes; at least one
  `visual_typed` query recovery unique over both controls; and strictly more
  visual recoveries than both controls. This licenses a fresh 100-case repair
  confirmation requiring at least 5/100 unique recoveries. It does not license
  controller training.
- **Boundary:** certificates change causally and the visual arm improves demo
  recovery or one control comparison, but has no unique query recovery over
  both controls. Retain as diagnosis evidence only.
- **Null:** no unique query recovery and no endpoint advantage with all guards
  passing. The candidate-conditioned relational compiler is not useful under
  this action language and budget.
- **Adverse:** visual typed repair recovers fewer queries than either matched
  control or violates a correctness/cost/frontier guard.
- **Invalid:** leakage, wrong cohort/checkpoint/source, malformed traces,
  incomplete provider jobs, non-identical replay, or mismatched cost. Repair
  infrastructure once without changing the frozen method, then rerun.

No group definition, ordering, slot map, budget, family, provider setting, or
tie-break may change after the pre-run source commit. A null/adverse result ends
this gate; it must not trigger same-cohort reweighting.
