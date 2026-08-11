# Cognitive Workspace + Object-Graph Rewrite v2 result

## Decision

The outcome-exposed development representation gate fails.  The two-stage
rewrite creates one query-blind novel-output opportunity on 50 frozen ARC-TGI
families, below the preregistered 5/50 threshold.  The sole candidate is exact
and unique over the frozen baseline and equal-cost cold arm, so the mechanism
has a small natural positive example; it does not have enough support or unit
compute efficiency to open prospective reserve or train a controller.

## Reproducibility and boundary

- Frozen source commit:
  `afded52f17cfd90f220953308c8e0c4449a5e402`.
- Candidate freeze ID:
  `79816f8458b39d6b29837d2d62eb1823e3127b7925cc811f86dd56616cf866de`.
- Both candidate files are byte-identical with SHA-256
  `21c050cbbaa9df5cd57a5cc21572de3f94d3930e16dea6c059af6a06b967ce5c`.
- Both post-freeze score files are byte-identical with SHA-256
  `6f5d7ac8e61c66c0e69d980db70b6f3c469d275d1b17c94aac0acca68529fecc`.
- Candidate construction read challenges only.  Development query gold and
  witnesses were read after both freezes.  No public evaluation was read.
- The unmaterialized 18-family reserve remains sealed.  No oracle-open command
  was run.

## Result

| Metric | Result |
|---|---:|
| tasks | 50 |
| query-blind novel-output opportunities | 1 |
| required to continue | 5 |
| frozen baseline exact | 1 |
| two-stage composition exact / pass@2 | 1 / 1 |
| unique over baseline and cold | 1 |
| equal-cost cold exact | 0 |
| baseline-or-composition union | 2 |

The one recovered family recolors two of four fixed-color 2x2 objects.  The
synthesized composition reaches the correct extensional output through two
scene/crop stages with different background assumptions.  It is valid under
hard replay but is not a structurally faithful explanation of the witness's
recolor rule.  This makes it evidence that depth-two composition can enter one
natural frontier, not evidence of robust relational diagnosis.

## Failure localization

Parent construction is not the dominant termination:

- 49/50 tasks have an execution-valid first-stage parent;
- 48 retain the maximum four parents;
- 193 parent certificates are constructed; and
- only one certificate has incompatible output shape.

The dominant terminal state is `parents_but_no_demo_exact_composition` on
48/50 tasks.  The best available parent has a median of 103 total mismatched
demo pixels and a mean of 204.78; only four tasks are within eight pixels and
ten are within 32.  These are mostly out-of-language failures rather than
local repair opportunities.

The post-freeze witness audit agrees with that diagnosis.  Most families need
cell/region propagation, filling rows or columns, adjacency-conditioned
drawing, checkerboards, geometric extension, reflection, motion, or canvas
resizing.  The bounded scene grammar is dominated by whole-object selection,
crop, correspondence, and rendering.  Applying that same grammar again to an
information-losing parent output rarely closes the demonstrations.

## Implementation versus theory

No replay, content hash, AST round trip, query-gold boundary, or native-cost
check failed.  The candidate and cold arms each reserve 102,400 program trials,
403,456 demo executions, and 122,880 query executions; unused work is explicit
padding.  This is therefore an algorithm/representation null, not a broken run.

The result rejects the current instantiation:

1. a generic `stages[1]` hole is not a typed diagnosis of an object, relation,
   canvas parameter, or AST node;
2. pixel-agreement parent ranking does not ensure information preservation or
   editability;
3. stage 2 receives only the parent grid, so information discarded by stage 1
   cannot be recovered; and
4. composing the same narrow grammar is still largely out of language.

It does not reject a shared-state, memory, and planning architecture.  The
current workspace implements content addressing, provenance, legal options,
and a deterministic budget-feasible representation path.  It does not yet
implement schema consolidation, a learned transition model, counterfactual
rollouts, delayed credit, or value-of-computation metacontrol.

## Next bounded hypothesis

Do not add a third generic stage and do not train a router.  A subsequent gate
should preserve a joint state containing the original input, parent output,
object graph, executable trace, and residual mask, then expose a single typed
node delta such as `reselect_object`, `change_canvas`, `replace_relation`, or
`compose_with_original`.  Parents should be ranked by information preservation
and trace-localized editability, with the current v2 parent-output-only path and
an equal-cost cold restart as controls.

The stronger thought experiment also changes which provider should enter this
gate: the existing reference-scale visual/NVARC anchor should supply frozen
perceptual proposals or traces, while the symbolic bridge supplies executable
state and verification.  Repeating self-composition inside the weak object DSL
does not instantiate the assumption that strong specialist capabilities are
already available.

Long-term memory and planning remain later gates.  A schema may enter reusable
memory only after family-disjoint validation; bounded rollouts must then beat a
matched myopic policy before any learned metacontroller is trained.  Static ARC
continues to test exact abstraction and solver coverage, while ARC-AGI-3-style
interactive tasks are the better eventual mechanism assay for goal acquisition,
belief update, memory compression, and long-horizon planning.
