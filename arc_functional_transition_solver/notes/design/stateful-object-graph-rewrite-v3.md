# Stateful Object-Graph Rewrite v3 preregistration

## Decision question

Does preserving executable object/relation state and repairing one diagnosed
AST node create a larger query-blind candidate frontier than the frozen v2
incumbent under matched native trial reservations?

This gate tests the stage-boundary representation. It does not test a learned
router, visual-posterior ranking, cross-task memory, leaderboard performance,
or biological brain-region equivalence.

## Frozen incumbent and candidate

- **Incumbent:** Object-Graph Rewrite v2 freeze
  `79816f8458b39d6b29837d2d62eb1823e3127b7925cc811f86dd56616cf866de`,
  including all one-stage, Object-Program Workspace v1, and v2 composition
  query-output bundles.
- **Candidate:** one `StatefulSceneRewrite` whose parent is an execution-valid
  `ScenePipelineProgram` from the same first 2,000 object/code trials.
- **Cold control:** an equal reserved number of unconditioned object/code
  programs after the shared 2,000-program prefix, ordered by a frozen content
  hash.

Historical provider implementations and result artifacts remain immutable.

## Typed execution state

For each input, parsing creates immutable records for:

- a grid and parse-contract content ID;
- persistent entity IDs derived from the source grid, occupied cells, and
  colors rather than transient object indices;
- relation IDs keyed by persistent source/target entity IDs and predicates;
- assignment, operation, canvas, and render state IDs; and
- an append-only trace identifying which nodes were reused or executed.

A reparse may split or merge objects. New records therefore retain overlapping
ancestor entity IDs; identity continuity is explicit rather than silently
assumed.

The replay dependency order is:

```text
parse -> assignment(correspond + select) -> operate -> canvas -> render
```

Changing one node must reuse every preceding state ID and execute exactly its
dependency suffix. `assignment` is a typed compound node because the existing
scene AST couples correspondence policy and signature-based selection.

## Failure compilation and legal actions

Demonstration-only execution residuals compile to exactly one node:

- first invalid trace node, with an empty scene assigned to `parse`;
- output-shape disagreement assigned to `canvas`;
- equal support with wrong values assigned to `operate`;
- complete-object or selection-conditioned disagreement assigned to
  `assignment`; or
- other shape-compatible transformation disagreement assigned to `operate`.

The legal frontier contains only programs whose serialized AST differs from
the parent in that node. There is no implicit fallback to a full program
restart. A missing trace, multi-node difference, stale input identity, or
certificate/parent mismatch is rejected.

## Search and cost contract

- Shared first-stage trials: **2,000 per task**.
- Retained stateful parents: **at most 4 per task**.
- Legal node rewrites: **at most 512 per parent**.
- Emitted demo-exact candidates: **at most 32 per task** before output
  deduplication.
- Candidate and cold arms reserve **2,048 program trials per task**, with
  matched demonstration and query execution reservations; unused trials are
  explicit padding.
- Actual executed and reused node counts are logged separately. They are an
  efficiency diagnostic and do not relax the matched reservation.
- Parent selection, failure compilation, rewrite ordering, exact verification,
  and novelty use demonstrations and query inputs only.

## Verification guards

Every emitted candidate must satisfy all of the following:

1. demonstration exactness;
2. typed single-node difference and complete dependency suffix;
3. serialized rewrite and scene-program round-trip;
4. identical output from cached suffix replay, fresh stateful execution, and
   the historical scene executor;
5. unchanged upstream state IDs;
6. content-address validation for trace, rewrite, candidate, and output bundle;
7. `novel_frontier_count > 0` before claiming a frontier-changing action; and
8. a query-gold-free candidate freeze reproduced byte-for-byte twice.

Controlled tests also inject node certificates, remove the parent-trace bridge,
change a downstream-node budget, and exercise parse split/merge lineage.

## Data lanes and outcome mapping

### Lane A: controlled semantic tests

Generated examples establish executor parity, persistent identities, node
localization, legal rewrite typing, dependency reuse, bridge necessity, and
exact replay. These are implementation/mechanism tests only.

### Lane B: exposed ARC-TGI dev50 opportunity screen

Use the same frozen 50-family challenge file and cohort ID as v2. The candidate
freeze is produced twice before opening the already exposed development oracle.

- **Representation pass:** at least **5/50** tasks have a demo-exact v3
  query-output bundle absent from the complete v2 incumbent.
- **Representation null:** fewer than **5/50** opportunities; do not materialize
  the reserve.
- Development exact recovery is reported diagnostically but does not replace
  the query-blind opportunity gate.

### Lane C: family-disjoint reserve18

Only after the development pass, materialize blind challenges and sealed oracle
commitments. Require at least **2/18** query-blind opportunities before opening
the reserve oracle. If opened, require at least **2/18** unique recoveries and
at least one more than equal-cost cold restart.

## Failure classification

- Replay, hashing, AST, identity, leakage, source-drift, or reservation failure:
  **invalid/engineering failure**, with no scientific belief update.
- Valid run below the Lane B or C opportunity threshold: **representation
  null** for this node-rewrite grammar.
- Opportunity exists but exact recovery does not exceed cold restart:
  **selection/semantics null**.
- Gains require extra native reservations or outcome-exposed retrieval:
  **adverse/unfair**.

Controller training remains prohibited until the prospective gate and residual
intervention tests both pass.
