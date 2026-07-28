# Object/code role reachability gate v0.2 (2026-07-28)

## Decision

The v0.2 role-reachability change works mechanically but does not improve
coverage. It fails the development gate, so the pre-registered offset-200
confirmation is not scored. Neural providers and learned controllers remain
frozen.

## Change under test

Source commit `36617915c6df658bdacbbb63bb7f6c2f3bf29741` changes only the
task-derived `role_stamp` enumeration and bumps the provider version to
`afts-hybrid-object-code/v0.2`. Program execution and the v0.1 DSL schema are
unchanged. Persistent source/target colors may enter the bounded role lists so a
copy canvas can be tested by demo replay instead of being rejected during parse.

The v0.1 offset-100 slice was already observed and is used as development. The
offset-200 query labels were not read because the sequential development gate
failed. A label-free blind enumeration preflight had only established that the
new family was reachable before the rule was frozen.

## Paired development result

| Metric on identical offset-100 tasks | v0.1 | v0.2 | Change |
|---|---:|---:|---:|
| grammar program trials | 1,516 | 4,300 | +2,784 (2.84x) |
| demo program executions | 4,648 | 13,544 | +8,896 (2.91x) |
| emitted candidates | 36 | 76 | +40 (2.11x) |
| emitted `role_stamp` candidates | 0 | 40 | +40 |
| natural near-miss cases | 36 | 76 | +40 |
| demo-exact provider candidates | 0 | 0 | 0 |
| provider raw/selectable/pass@2 | 0/0/0 | 0/0/0 | 0 |
| new unique selectable coverage | 0/100 | 0/100 | 0 |
| baseline + provider selectable union | 9/100 | 9/100 | 0 |
| typed novel oracle recoveries | 0 | 0 | 0 |

All 4,300 v0.2 programs were exhausted under the unchanged 20,000-trial cap.
Every task was classified as a candidate-language failure, not a search-cap,
selection, or runtime failure. Thus this result is not explained by the router or
by insufficient search within the declared grammar.

The reachability intervention did exactly what it was intended to do: the 40 new
natural candidates are all `role_stamp`, alongside the original 36 D4 near
misses. They simply remain too far from any demo-exact program. More candidates
did not create a better candidate distribution.

## Repair analysis

Of 76 natural cases, 60 had an executable typed/cold pair and 16 had an empty
typed frontier. The completed pairs consumed identical native budgets on each
side: 490 program trials, 1,498 demo executions, and 490 query executions. Neither
side produced a novel oracle recovery, and typed repair produced no raw oracle
rediscovery.

All 16 empty cases were `role_stamp` parents diagnosed as
`background_role_mismatch` and mapped to `reparse_background`. The action was
schema-legal, but no alternative program satisfying the fixed roles existed.
This exposes a missing semantic condition:

> A typed action is not frontier-changing merely because its operator and fields
> type-check; it must produce at least one candidate absent from the frozen parent
> pool.

The natural sample remains below the required 100 and contains no novel recovery.
Controlled fault injections still produce four distinct actions, but they prove
only deterministic compiler plumbing, not natural utility.

## Failure attribution

1. **Implementation/protocol:** supported. Full regression was 394 passed and 9
   skipped. Clean source binding, pool replay, hidden-output isolation, content
   IDs, v0.1 program replay under v0.2, and matched native costs passed.
2. **Reachability fix:** supported. The intervention activated 2,784 role programs
   and emitted 40 new role near misses.
3. **Coverage hypothesis:** rejected on development. A 2.84x larger search produced
   no demo-exact or oracle candidate.
4. **Natural typed-repair hypothesis:** unsupported. There were no novel
   recoveries, and 16 legal actions had an empty semantic frontier.
5. **General object/code or residual-control direction:** not rejected. The tested
   representation still binds global colors rather than object components, has no
   learned/code-generated program prior, and cannot compose multi-stage object,
   canvas, counting, or arrangement operations.

## Relation to stronger ARC candidate generators

This ablation sharpens the difference from NVARC/SOAR-style code generation and
masked-neural systems: increasing a hand-enumerated candidate count is not the
same as improving the probability mass around correct programs. Our auditable
typed-control infrastructure remains useful, but it currently operates on a weak
candidate distribution and cannot compensate for that weakness.

The next object/code version should therefore add representation power, not relax
more color-count filters:

- connected-component scene graphs and typed object roles;
- object correspondence by shape, size, topology, and relative position;
- explicit output-canvas and size hypotheses;
- multi-stage copy, count, arrange, compose, and crop AST nodes with node-level
  traces and holes;
- a hard `novel_frontier_count > 0` action precondition;
- a generated/code-model proposal lane evaluated first as a static provider.

No router training should resume until one such version contributes at least
3/100 unique selectable coverage and natural typed repair reaches the existing
novel-recovery gate.

## Artifact identity

- Result ID:
  `ca3f0d8e7a6d879ab3f245a755a301101863deb12b0f6487136765c99c64d9a3`.
- Result directory:
  `results/object_code_gate_v2_arc1_train_dev_offset100_n100_20260728`.

The summary is bound to the clean source commit and includes all candidate pools,
repair pools, native cost ledgers, baseline hash, runtime metadata, gate outcomes,
and failure classes.
