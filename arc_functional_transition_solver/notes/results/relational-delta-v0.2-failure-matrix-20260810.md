# Relational-delta v0.2 terminal failure matrix

## Decision

The frozen 48-task audit is complete and supports a **narrow** decision.

The largest terminal bucket is `parse_failure` (18/48), but it is not one
semantic family.  It contains palette legends, mirror completion, panel
rearrangement, path extension, object relocation, anomaly removal, and several
other transformations.  In v0.2, `parse_failure` specifically means that the
fixed `minority_foreground` / `majority_foreground` role contract cannot be
satisfied on every demonstration.  It does not show that a more permissive
color-count parser would make the existing relation and render language solve
those tasks.

Accordingly, the next representation family is limited to the clearest
reusable subcluster inside the largest bucket:

```text
border/control legend
  -> ordered color correspondence
  -> payload-object assignment
  -> typed legend-conditioned rendering
```

This is an opt-in candidate language.  It must not modify v0.2 enumeration,
candidate IDs, budgets, scores, or prior artifacts.  It is a development probe
on an outcome-exposed cohort, not generalization evidence.

## Frozen matrix

| Mutually exclusive terminal category | Tasks | Share |
|---|---:|---:|
| `parse_failure` | 18 | 37.5% |
| `relation_missing` | 14 | 29.2% |
| `canvas_incompatible` | 12 | 25.0% |
| `legal_but_inexact` | 3 | 6.2% |
| `ast_insufficient` | 1 | 2.1% |
| **Total** | **48** | **100.0%** |

The categories use the preregistered precedence
`canvas_incompatible -> parse_failure -> relation_missing -> ast_insufficient
-> legal_but_inexact`.  Each stage must be reached on every demonstration.
Therefore, the table is a terminal-stage decomposition of one frozen language,
not a causal taxonomy of ARC tasks.

### Complete task assignment

| Category | Task IDs |
|---|---|
| `canvas_incompatible` | `25c199f5`, `5587a8d0`, `6165ea8f`, `83eb0a57`, `878187ab`, `8a6d367c`, `9ba4a9aa`, `a6953f00`, `aaecdb9a`, `b71a7747`, `da6e95e5`, `e9fc42f2` |
| `parse_failure` | `17829a00`, `2de01db2`, `3d588dc9`, `54dc2872`, `5623160b`, `5a719d11`, `5adee1b2`, `902510d5`, `9720b24f`, `9f669b64`, `bc93ec48`, `c4d1a9ae`, `c9680e90`, `db615bd4`, `e4888269`, `e729b7be`, `ecb67b6d`, `f341894c` |
| `relation_missing` | `2f767503`, `4df5b0ae`, `52364a65`, `66ac4c3b`, `825aa9e9`, `97c75046`, `984d8a3e`, `98c475bf`, `9b5080bb`, `ad3b40cf`, `c6141b15`, `d6542281`, `df978a02`, `fe45cba4` |
| `ast_insufficient` | `753ea09b` |
| `legal_but_inexact` | `396d80d7`, `3ad05f52`, `c3fa4749` |

The machine-readable matrix records, for every task, blind-task hashes, demo
and query counts, shape relations, palette deltas, demonstrated-delta
components, identity residual, execution-reason histogram, grammar size, and
the number of programs reaching each typed boundary.

## Why the largest bucket cannot be repaired by role relaxation alone

Manual review used only exposed development demonstrations.  The 18 tasks
split into the following structural hypotheses:

| Task | Demonstration-level structural hypothesis |
|---|---|
| `17829a00` | project objects toward matching colored boundary markers |
| `2de01db2` | reorganize a panel or periodic pattern |
| `3d588dc9` | marker/orientation-conditioned object edit |
| `54dc2872` | complete colored L-shaped objects |
| `5623160b` | decompose and relocate composite colored objects |
| `5a719d11` | compare or align congruent panel shapes |
| `5adee1b2` | decode a border legend and fill mapped payload-object neighborhoods |
| `902510d5` | extend a path from isolated colored markers |
| `9720b24f` | remove anomalous interior colors |
| `9f669b64` | split and arrange composite objects |
| `bc93ec48` | correct or complete object structure |
| `c4d1a9ae` | repair a stripe or repeated pattern |
| `c9680e90` | correspond objects across a separator and recolor/reposition them |
| `db615bd4` | encode motifs into a framed panel |
| `e4888269` | decode an explicit border palette and recolor payload objects |
| `e729b7be` | mirror-complete an object across a separator |
| `ecb67b6d` | fill a relation-defined region or boundary |
| `f341894c` | orient/recolor domino objects from border markers |

These labels are diagnostic hypotheses, not ground-truth ARC family labels.
They show why the machine terminal reason has low action specificity.  A
generic role relaxation would still lack the required relation and render AST
for almost every row.

## Selected coherent subcluster

`5adee1b2` and `e4888269` share an input-side control structure:

1. ordered, adjacent color pairs occur in a compact border lane;
2. the first color of each pair appears in spatially separate payload objects;
3. the second color specifies a per-object render color;
4. one task recolors payload cells, while the other writes the mapped color in
   a topology-preserving padded object region.

The proposed language is one representation family because both candidates
share the same typed parse, correspondence, payload assignment, and mapping;
only the final bounded render node differs.  The grammar must parse legends
from inputs alone, choose a render program by demonstration-exact synthesis,
exclude control cells from payload edits, produce content-addressed programs,
and reject ambiguous or inconsistent legends.

The development gate is:

- controlled semantic fixtures pass, including negative ambiguity and payload
  leakage cases;
- both selected development tasks have a demonstration-exact program;
- any emitted query candidate is content-novel relative to the frozen v0.2
  candidate artifact;
- at least one selected task adds raw query-oracle coverage over v0.2;
- the implementation reads no query output while generating candidates.

Failure to recover either task is an implementation or semantic-specification
failure to diagnose before judging the representation.  Success is only
outcome-exposed reachability evidence; it does not satisfy the future fresh-set
gate of at least 5/100 unique recoveries or the equal-cost cold-restart test.

## Reproducibility and immutable identity

- candidate freeze ID:
  `85811f28942fc447d529be60e217d4cd4b5326c485ea901dda3d19b983fe32e5`;
- cohort ID:
  `dfee67cf54613ba721e1fe46723967c95020715dff1a40500fe498055805e3ac`;
- matrix ID:
  `ce11afec96e717dd44a2c4354a44eb84fdc523b515f5bf62c3d23ee0906ddf3a`;
- matrix JSON SHA-256:
  `72f12424dbf8adc5de4c4410cec1ee158f1edde12fcb4c1a73a566e407b8feb1`;
- matrix CSV SHA-256:
  `d13146cb766553c747335c36680aa9fea6d6d6c15425c56e341cdb94d6332f88`;
- query gold read: `false`;
- public evaluation read: `false`;
- deterministic replay: the matrix was independently constructed twice and
  was byte-identical.

The superseding artifacts are under
`results/relational_delta_failure_matrix_20260810_v2/`.  The first artifact
(`matrix_id=4af69e7b...`) is retained unchanged.  Repository formatting changed
only the diagnostic-script source hash: all 48 task records, category counts,
and CSV bytes are identical between v1 and v2.  The demonstration review sheets
remain in the retained v1 directory.

## Claim boundary

The accurate project status remains:

> We have established a highly trustworthy heterogeneous-candidate and
> negative-result auditing infrastructure, and observed local complementarity
> between visual and symbolic candidates.  Strong candidate generation,
> executable cross-representation repair, and causal functional switching have
> not yet been achieved.

No controller training or brain-inspired switching claim is authorized by
this audit.
