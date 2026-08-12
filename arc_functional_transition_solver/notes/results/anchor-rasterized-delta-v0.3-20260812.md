# Anchor-rasterized delta v0.3: sensor null

## Outcome

The frozen v0.3 run is valid and stops at the preregistered sensor gate.  Only
one of 50 outcome-exposed development tasks has a parent whose finite mask
version space predicts every held-out demonstration.  The promotion threshold
was three tasks.  The result is therefore a **sensor null**, not evidence about
the utility of a more general mask language.

The one reachable task is not a useful localization success.  All four
reconstructed parents admit seven demo-exact mask programs, but their query
outputs deduplicate to content already present in the incumbent frontier.  The
same task is also solved under leave-one-demo-out by both global recolor and
whole-component recolor.  Consequently:

- strict LODO tasks: 1/50;
- reachable and demo-exact candidate tasks: 1/50;
- content-novel query-frontier tasks: 0/50;
- raw mask-exact tasks: 1/50, already incumbent-exact;
- mask pass@2: 0/50;
- unique recovery over incumbent plus frozen equal-cost cold: 0/50;
- incumbent/frozen-cold union: 2/50;
- promotion threshold: 3/50, not met.

No reserve, router training, or grammar amendment is licensed.

## Failure matrix

The gate reconstructed 196 content-bound executable parents over 49 tasks; one
task had no execution-valid parent.  Parent-level terminal causes are:

| terminal cause | parents | interpretation |
|---|---:|---|
| `multi_delta_unreachable` | 102 | one post-render color delta cannot repair the parent |
| `anchor_mask_unreachable` | 48 | a shared single-color delta exists, but none of the 12 frozen support programs matches it |
| `cross_demo_delta_inconsistent` | 34 | demonstrations require different color deltas |
| `shape_unreachable` | 4 | parent and target output shapes differ |
| `canvas_incompatible` | 4 | input, parent, and target do not share the required canvas |
| reachable | 4 | all four belong to the same redundant task |

At task level, 25 tasks are exclusively multi-delta, 12 exclusively
anchor-mask-unreachable, eight exclusively cross-demo-inconsistent, one mixes
multi-delta and cross-demo inconsistency, one is shape-incompatible, one is
canvas-incompatible, one has no valid parent, and one is reachable.

This sharpens the previous recolor audit.  The 48 parents from 12 tasks that
were identified as spatially conditioned single-color residuals are exactly the
48 parents rejected by the finite anchor-mask language.  Cardinal stencils,
anchored 2x2 quadrants, row/column projections, and bounding-box fills do not
express their support.  The grammar did not miss them through search or
ranking; exact bounded reachability proves they are outside this language.

## Engineering validity and cost

The controlled abstract/concrete semantics, schema round trips, explicit graph
dependencies, LODO baselines, query-blind construction, certificate guards,
and budget guards pass.  The dependency suite reports 75/75 tests and Ruff
passes.  Two independent freezes and two independent score replays are
byte-identical:

- candidate freeze SHA-256:
  `d7e708b1187ec5a10f27e46f5a6b54c0d6d9b24bf9504f5000d94b737b88c4d8`;
- result SHA-256:
  `8589f7bcdcc9a8d1ff3b940c982106d6ba852cde3dfe44bceec808435787b7c0`;
- source commit:
  `c34120cd5e46454f11f9b16ee626904865714e32`.

Across 50 tasks the run reserves 102,400 native program trials, realizes
23,520 finite-language trials, records 78,880 padding trials, and accounts for
93,120 separate LODO-only audit trials.  Candidate construction never reads
query targets; scoring occurs only after both candidate freezes match.

The failure is therefore algorithm-semantic rather than an implementation,
replay, leakage, or accounting failure.

## Scientific interpretation

The experiment preserves the earlier controlled result that a failure
certificate can license a typed topology insertion.  It rejects a stronger
hypothesis: that a small raster predicate over anchor-color marginals is a
useful natural sensor for deciding the inserted function.

The main insight is that the residual support should not be predicted as an
independent pixel mask.  It should be the observable footprint of a typed
counterfactual intervention on persistent objects, relations, or an AST node:

```text
typed intervention on executable state
-> affected identities and relations
-> causal raster footprint
-> local effect
-> affected-subtree replay
```

This preserves the joint correlation between *which structure changed* and
*which cells may change*.  It also gives a principled reason to reject a mask:
no legal intervention in the active representation has that footprint.  The
next research object is therefore a relational causal-footprint certificate,
not another hand-added rasterizer.

The v0.3 stop rule remains in force.  Any relational-footprint grammar must be
preregistered and evaluated on a fresh family-disjoint cohort; the exposed 50
tasks may be used only as a failure-analysis set.
