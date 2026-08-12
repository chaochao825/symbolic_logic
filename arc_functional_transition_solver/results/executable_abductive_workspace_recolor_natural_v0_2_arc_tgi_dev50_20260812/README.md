# Recolor topology v0.2 natural utility audit

Status: **valid algorithm-semantics null** on the outcome-exposed ARC-TGI
development-50 cohort.

The controlled mechanism gate passed before this run: the finite 90-program
abstract domain matched concrete execution in all 81 exhaustive small-grid
comparisons; all three positive topology loops closed; all five negative guards
held. The natural audit then reconstructed all v4 parent hashes exactly, ran two
byte-identical query-blind freezes, and scored only after freezing.

## Natural result

| Endpoint | Result |
|---|---:|
| Frozen tasks | 50 |
| Reconstructed scene parents | 196 |
| Reachable parents | 4 |
| Reachable tasks | 1 |
| Demo-exact candidate tasks | 1 |
| Globally novel frontier tasks | 0 |
| Recolor exact tasks | 1 |
| Unique recovery over incumbent + cold | 0 |
| Promotion threshold | 3/50 |

The only reachable task was
`arc_tgi_taskEpt4U7XM7kW2qULwGBFcaN_e53dce966b6f5db3`. All four parents
collapsed to one query-output bundle. It was exact, but the same bundle was
already present in the incumbent frontier. The novel-frontier guard therefore
correctly rejected it as a new system-level frontier.

Of the 192 unreachable parents, 188 cannot be made demo-exact by any single
global recolor and four have incompatible output shapes. One task has no frozen
scene parent. Thus the null is not an implementation or replay failure: the
candidate language is too narrow and overlaps the incumbent where it does fit.

A post-hoc, demonstration-only refinement further separates those 188 parents.
No parent is solvable by merely widening the node to a global multi-color map.
Forty-eight parents across 12 tasks have one recolor pair but require a spatial
mask; in every case the changed cells cut through a 4- and 8-connected source
color component, so whole-component recolor is also insufficient. The other
140 parents across 35 tasks require multiple spatially conditional mappings.
This refinement is diagnostic, not a preregistered endpoint, but it rules out
`global ColorMapGridNode` as the obvious next widening.

## Fairness and reproducibility

- Every task reserved 2,048 native program trials, matching frozen cold restart.
- Recolor realized 17,640 trials and recorded 84,760 padding trials; padding was
  not reallocated.
- Incumbent + cold exact: 2/50; cold alone: 0/50.
- Dependency regression: 62/62 passed; ruff passed.
- Both freeze and score replays were byte-identical.
- Freeze SHA-256:
  `5f2d9807868493865b28327e53271059cf7f3db611a245f1fbccc394aef743ad`.
- Result SHA-256:
  `00fb9e081fd61e75ca548026c5cc27c74b66b63f34958603a8f86cdc3a691949`.

This exposed-development null does not disprove typed topology insertion in
general. It rejects this single global-recolor node over this frozen parent
distribution as a source of novel natural utility. It does not authorize a
second node by default; the next proposal must be chosen from the observed
failure certificates rather than from the query score. If another output-edge
node is tested, the evidence points to a structured relation/region mask feeding
a typed local recolor or transition—not a larger unconditional color map.
