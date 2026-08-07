# Query-blind visual-provider confirmation v3

## Decision

The v2.1 pilot produced 4/20 unique selectable and 7/20 unique raw coverage
against the frozen portfolio plus object/code provider.  This confirmation uses
all 11 tasks that remained eligible under the original v2 selection rule.  No
task is selected or removed using the pilot's error types.

The controller remains frozen.  The confirmation changes neither the VARC
checkpoint nor its test-time configuration, augmentation, candidate validation,
or ranking.

## Frozen cohort

- exposure registry ID:
  `2a91ea020bb6e978b3069eec3e86b356aee5becf525aa896e717770e29ad7825`;
- exposure registry SHA-256:
  `d9f73d8e879ab76488f0f2dae5f80f6d3df09c863a93695470edcb4cddc8180d`;
- cohort ID:
  `b070322ec6e90cb02f08fb3a4d41172588839f3a024dba31c0b76792ad7fb19a`;
- manifest SHA-256:
  `3917256a0cf026f0ef8d8320fec66771357f0c41bc592c12cf19a9125e69858d`;
- blind-data package SHA-256:
  `c71c81bc52bb31b1b4e21baff7d64be6f8d486eea90ef978bbbcf3a30010087b`.

The ordered task IDs are:

```text
cc9053aa 230f2e48 b1986d4b 7e2bad24 880c1354 a2d730bd
20fb2937 252143c9 470c91de d753a70b f0100645
```

The cohort is the complete deterministic remainder after excluding ARC-AGI-1,
all prior project semantic exposure, the aborted v1 run, and all 20 gold-scored
v2 tasks.  There were exactly 11 eligible tasks.  Its 572 provider JSON files
contain 624 query instances; every query output is the input-copy sentinel.

## Fixed provider and comparison contract

- VARC source commit and checkpoint are identical to v2.1.
- TTT uses 100 epochs, seed 42, one replica, ten inference attempts, and the
  released five basic perspectives with nine color permutations.
- Invalid samples are rejected by the v2.1 syntax-only rule.  No candidate is
  cropped, recolored, coerced, or repaired.  Every query must retain at least
  one valid candidate.
- Valid candidates are ranked by frequency, first-emission position, and
  canonical JSON.
- The static portfolio and object/code v0.3 use the same task order and source
  commit `8c855bf0c2a37c135c9ac0f68fbe2e3fd677bc7c`.
- Predictions and contracts are frozen before gold scoring.

## Endpoints

The primary endpoint remains unique selectable coverage over the union of the
static portfolio and object/code provider.

- at least 1/11: replicates nonzero independent selectable coverage;
- at least 2/11: strong small-cohort replication;
- 0/11: confirmation failure; report the aggregate 31-task estimate but do not
  claim a stable provider rate.

Raw coverage, pass@1/pass@2, invalid-sample rate, modal-canvas correctness,
oracle rank, support, and same-shape Hamming distance are secondary.  No result
from this gate supports residual control or brain-like switching.

## Frozen outcome

The run completed on all 11 tasks with zero provider, baseline, or object/code
failures.  Of 6,120 raw samples, 5,989 were valid and 131 were rejected by the
unchanged color-range rule.  The visual provider achieved 6/11 unique raw and
4/11 unique selectable coverage; pass@1 and pass@2 were both 4/11.  The frozen
base and object/code union was 0/11.

The confirmation therefore passes both preregistered small-cohort criteria.  In
the combined disjoint 31-task set, visual unique selectable coverage is 8/31
and unique raw coverage is 13/31.  This licenses reporting a stable positive
development signal, but the 100-task provider gate remains unevaluated.

The exact v3 score result is
`97f605f13dad6bb9e5cea6fbefdf36eb9763c7d418cea0f74355bbcb3981491c`.
The fixed posterior-localization audit also replicated its exploratory signal:
median AUROC 0.945 over seven incorrect same-shape queries and 5.62x recall lift
for the fixed top-10% mask.  No additional bridge variant was run after the
pixel-consensus bridge failed its v2.1 advancement gate.
