# Relational-mask frontier and ARC-AGI-2 confirmation gate

## Decision

The frozen representation gate is **null**. The opt-in
`afts-relational-mask-dsl/v0.1` implementation and its audit harness replayed
correctly, but the representation added no exact candidate on the fresh
50-task ARC-AGI-2 public-training cohort. Typed repair and equal-cost cold
restart both recovered zero tasks. The visual LODO experiment and every learned
controller therefore remain unauthorized.

This result does not test whether a visual posterior can diagnose a useful
repair when an executable near miss exists. It establishes the earlier
precondition: the current action language does not reach such parents on fresh
data.

## Prospective evidence

### Complete fresh ARC-GEN actuator scan

The corrected feasibility scan used ARC-GEN commit
`a15cbdb44c776610aeeb9f487a06af875d3d0878` and its locked ARC-AGI submodule
commit `399030444e0ab0cc8b4e199870fb20b863846f34`. The source validator required
exactly 400 training and 400 evaluation ARC-AGI-1 task identities before any
family could be called fresh.

| Quantity | Count |
|---|---:|
| ARC-GEN families | 900 |
| ARC-AGI-1 identities | 800 |
| ARC-AGI-1 / generator overlap | 728 |
| Authored exposures | 479 |
| Generator exposures excluded | 440 |
| Eligible and scanned families | 139 |
| Retained demo-exact single-slot repair opportunities | **0** |

The 139 terminal reasons were 134
`no_demo_exact_single_slot_neighbor`, two duplicate generated pairs, two
per-family timeouts, and one out-of-scope grid. The two timeouts are typed
construction-budget exclusions, not representation failures. Query gold and
VARC predictions were not read. A second complete scan reproduced all 139
ordered terminal reasons and the canonical scan ID.

An earlier dry run incorrectly used empty repository-local ARC-AGI-1
directories and produced an apparent 460-family fresh pool. That eligibility
claim is invalid. The run was moved to ignored `trash/`, was not used for model
or cohort selection, and is not published as evidence. The source-count
validation and its regression tests were added before the corrected scan.

### Frozen ARC-AGI-2 public-training confirmation

Because the existing scene-AST actuator exposed no candidate, a minimal
relational representation was implemented behind an opt-in API. It recognizes
a background center jointly supported in all cardinal or diagonal directions
at one or two radii and renders a learned marker on a copy or blank canvas. Its
six editable leaves are relation background, support color, direction set,
radii, render color, and canvas mode. It is not appended to the legacy provider,
so existing enumeration order, candidate IDs, checkpoints, and results are
unchanged.

The confirmation source was ARC-AGI-2 commit
`f3283f727488ad98fe575ea6a5ac981e4a188e49`, `data/training` only. The public
evaluation split was not opened. All ARC-AGI-1 identities, the frozen exposure
registry, the 139-family scan, and the development signature were excluded.
Fifty eligible tasks were ordered by the preregistered hash and retained before
execution. Blind payloads contained demonstrations and query inputs only.

ARC-AGI-2 supplies no semantic family labels. Consequently, this cohort is
task-ID disjoint and blind-structural-signature deduplicated, but it must not be
described as proven semantic-family-disjoint.

| Frozen endpoint (50 tasks) | Result |
|---|---:|
| Legacy 512-program raw oracle / pass@2 | 0 / 0 |
| Complete relational raw oracle / pass@2 | 0 / 0 |
| Combined pass@2 | 0 |
| Mechanically repair-eligible parents | 3 |
| Typed-edit recovery | 0 |
| Equal-cost cold-restart recovery | 0 |
| Relational unique coverage | **0** |

Both repair arms reserved 16 trials and executed identical demo/query costs on
all 50 tasks. Candidate construction replayed byte-for-byte before gold release.
There were no cost, replay, or frontier-accounting violations.

The final source audit additionally corrected the same-MDL tie-break to use
program ID before provider name and made the exposure freezer recompute the
scan content ID. A complete post-fix 50-task rebuild produced byte-identical
`candidate_freeze.json` and `result.json` files with the same hashes and IDs
listed below. These correctness changes therefore did not alter this endpoint.

The preregistered gates required at least two and at least 4% relational unique
recoveries, at least three natural typed frontiers, and at least three typed
recoveries with a net gain of one over restart. The unique-coverage and repair
gates failed, so `visual_lodo_authorized=false` and
`controller_training_authorized=false`.

## Post-hoc near-miss validity audit

The preregistration counted three parents because full-grid pixel agreement was
at least 0.5. A query-blind post-hoc diagnostic compared each parent with the
identity transform and measured F1 on the demonstration change mask. It is
explicitly labeled non-preregistered and does not rewrite the frozen endpoint.

| Task | Pixel agreement | Parent mismatches | Identity mismatches | Delta F1 | Quality near miss |
|---|---:|---:|---:|---:|---|
| `7acdf6d3` | 0.965 | 15 | 10 | 0.000 | no |
| `320afe60` | 0.806 | 264 | 267 | 0.255 | no |
| `3ad05f52` | 0.528 | 916 | 406 | 0.000 | no |

All three are background-dominated false near misses. Visual inspection of the
demonstrations identifies different missing program factors:

- `7acdf6d3` moves/reassigns markers and needs an `erase + add` delta program;
- `320afe60` recolors complete components by role, shape, and relative position;
- `3ad05f52` fills relation-defined regions with an input payload color.

The mechanically true `natural_typed_frontier` field in the frozen v1 result
therefore cannot support a natural-near-miss or residual-control claim. The
near-miss definition itself was underspecified.

## Failure classification

| Observation | Classification | Consequence |
|---|---|---|
| Empty ARC-AGI-1 identity source in an early dry run | Harness implementation defect | Fixed with exact source-count validation and fail-fast tests; invalid run quarantined. |
| Script entrypoint import failure | Packaging defect before data access | Fixed and covered by a real subprocess entrypoint test. |
| Generator exceptions, timeouts, and oversized grids | Expected external boundaries | Converted to explicit terminal reasons without silent fallback. |
| Corrected ARC-GEN scan retains 0/139 | Existing scene-AST actuator/reachability failure | Do not run VARC on this gate. |
| Relational provider is exact on 0/50 | Representation-language failure | Radial center marking is too task-specific. |
| Typed edit and restart both recover 0/50 | Action-language failure | Existing six slots cannot express the required delta programs. |
| Three high-agreement parents all fail delta-quality audit | Metric-design failure | Replace raw agreement with identity-normalized delta support and topology/object constraints. |

The negative endpoint is therefore not explained by broken execution or an
untrained router. It is an algorithmic null after the identified harness bugs
were repaired without changing the frozen method.

## Next bounded design

The next development candidate should factor the executable program as:

```text
parse + role assignment
  -> n-ary relational correspondence
  -> target component / coordinate / region
  -> typed delta mask
  -> render {add, erase+add, recolor-component, fill-relation-region}
```

Parent eligibility should require improvement over the identity baseline,
minimum delta-mask precision/recall, preserved object/topology constraints, and
at least one legal edit that changes the content-addressed frontier. Controlled
faults and already exposed public-training tasks may be used for development,
but a later confirmation requires a newly frozen semantic-family source. The
ARC-AGI-2 public evaluation split must remain one-shot and unopened during this
iteration.

Only after this representation produces unique exact coverage and typed repair
beats equal-cost restart should LODO posterior evidence be introduced to select
the target/operation slots. Controller training remains frozen.

## Immutable identifiers

- ARC-GEN feasibility scan ID:
  `459b1667edbfb6d9af5aa180f1a402b8309890991b6fa22bcf32decd758a1440`
- ARC-GEN scan file SHA-256:
  `fdd8d424e97710c3afa619dd11533413007b62bb9499b190ae3a7f5686cc43f1`
- Exposure registry ID:
  `dbb0c5643cc4f077bc7c24dcca1e81863ab56cfe38dfdefe75e419bae202a87e`
- Exposure registry file SHA-256:
  `955655acaba23176d2f8835c3b295d516fb101c7c2e99a0f3b652fdc3ffb275d`
- Cohort ID:
  `dfee67cf54613ba721e1fe46723967c95020715dff1a40500fe498055805e3ac`
- Cohort manifest file SHA-256:
  `7610301e1ac359d035f1949f70e3b33ea22ef990b07a5434d01e875beae0ba45`
- Candidate freeze ID:
  `ddeb3e4c6d057d927359b22eb2e7e5a7369e80be5d9c0298112337d6835b6556`
- Candidate freeze file SHA-256:
  `0454b2a799f34b940e22c817a3d83598872fb216102a5a0d18e3df1f6c3cbe7f`
- Result ID:
  `916221630a9a3472e43e05a33932486f1d57f6a49d274bc75677c8b4ae678f6e`
- Result file SHA-256:
  `88e33eb39b0c693fe5e733ecacf90388389556280a4d0ff734b5abbe1d07cb08`
- Post-hoc diagnostic ID:
  `cae5ed1e3f0d7f13daf08d20384d0948aa3add1562b08ff8b1c38a881e75ec9c`
- Post-hoc diagnostic file SHA-256:
  `f488bdee8b1ef662b506136e65296f2756e597ca84d288bd5f48ed6f34b692a2`

Machine-readable evidence is under
`results/visual_relational_trace_repair_20260809/`.
