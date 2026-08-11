# Stateful Object-Graph Rewrite v3 development gate

Status: **verified representation null; stateful actuator passes controlled
tests, but natural node diagnosis and rewrite coverage fail**.

The v3 interface preserves executable traces and persistent object/relation
identities, compiles demonstration failures to one typed AST node, rewrites
that node, re-executes only its dependency suffix, and checks the result against
fresh stateful and historical execution. On the frozen, outcome-exposed
50-family ARC-TGI development cohort, it creates 0 query-blind output-frontier
opportunities, below the preregistered 5/50 threshold. The family-disjoint
reserve was therefore not materialized and controller training remains closed.

The valid demo-only failure audit separates two causes. The compiler-selected
node improves only 1/196 parents, while an oracle over all five legal nodes
finds an improving rewrite for 94/196. However, none of 7,446 legal single-node
trials is demo-exact. The current result is therefore both a typed-diagnosis
failure and a single-node action-language/parent-frontier failure. It is not a
failure of persistent identity, affected-subtree replay, content addressing,
or matched-cost accounting.

The first diagnostic audit terminated on an oversized render that the
stateful executor had not translated to the historical invalid-execution
boundary. That diagnostic attempt is marked invalid, its receipt is retained,
and a regression-tested fix produced the successful query-gold-free retry. The
main preregistered gate had already completed and did not encounter this path.

The v3, v2, workspace, publication-policy, and adjacent regression selection
passes 60/60. The complete repository suite reports 575 passed, three skipped,
and the same six pre-existing M04a evidence failures caused by absent protected
sidecar/cache/source-snapshot payloads; it is not reported as a green full
suite.

`summary.json` contains payload-free aggregate evidence. The paired freezes
and paired score files are byte-identical; `artifact_sha256.json` records raw
artifact sizes and hashes. Raw task-level candidates, traces, logs, and audits
remain protected as manifest-only publication entries.

See `notes/results/stateful-object-graph-rewrite-v3-20260811.md` for the full
failure attribution and decision boundary.
