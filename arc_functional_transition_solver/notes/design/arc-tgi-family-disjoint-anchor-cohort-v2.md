# ARC-TGI family-disjoint anchor cohort v2

This protocol supersedes v1 after its preflight stopped before writing any task
artifact. The v1 failure is retained as evidence; no solver or anchor output was
observed.

## Why v1 stopped

At the frozen v1 episode seeds, two of 170 canonical ARC-Mini generators emit
outputs wider than ARC's 30-cell maximum:

| canonical family | selected source SHA-256 | frozen shapes | disposition |
|---|---|---|---|
| `taskGqqVrV4CnAUNRgZGQZCUQK` | `6050e25b52f8c4f669b7ddb0cdcc519ec25a1b5d42a59835292e0abe3353558e` | inputs `10x30`; outputs `10x38`–`10x45` | quarantine |
| `taskaasAJ4e5NPRnnWF5HTmp35` | `3159f8fab3c1fdeeec3aebc91b4d4b021f655a8e586af573ff7c184491e02726` | inputs `11x25`; outputs `22x50` | quarantine |

The failure is upstream generator/data validity, not solver or bridge behavior.
Changing the 30-cell rule would change ARC semantics, while resampling until a
valid seed appears would create an undisclosed adaptive filter. V2 therefore
quarantines exactly these source-hash-bound families before partitioning.

## Frozen source and partition

- ARC-TGI commit: `a614132ff5b2cb3628063d541e7cbd74a2cd2edb`
- source files: 180
- conservative canonical families: 170
- quarantined by the table above: 2
- eligible families: 168
- development: 50
- confirmatory: 100
- unmaterialized reserve: 18
- partition seed: `afts-arc-tgi-arcmini-family-split-20260810-v2`
- episode seed: unchanged from v1,
  `afts-arc-tgi-arcmini-episode-20260810-v1`

Keeping the episode seed unchanged prevents the v1 format failure from causing
post-hoc resampling of any retained family. The manually inspected generator
`task2fJ984g27gSFKHfq53RTVH` remains forced into development.

## Remaining contract

All source selection, replay, witness, grid, split, identifier-overlap, and
gold-boundary conditions from v1 remain unchanged. Every retained episode must
be ARC-valid and must pass both direct and partially evaluated witness replay;
there is no additional fallback or skipping path.

The anchor and bridge gate also remains unchanged: on the 100 family-disjoint
confirmatory episodes, typed visual proposals must uniquely recover at least
5/100 tasks and beat an equal-native-cost cold restart. The benchmark remains
mechanism-track evidence only.
