# NVARC/TRM static anchor on the frozen ARC-TGI cohort

This directory contains only aggregate, query-boundary-safe evidence. The
complete candidate freeze and task-level outcomes remain external until the
visual bridge and its confirmatory candidates are frozen.

## Outcome

The reference-scale static anchor passes its predeclared 10/100 usefulness
gate on the 100-task synthetic, generator-family-disjoint confirmatory cohort.

| k | strict task pass | query pass | upstream mean pair pass |
|---:|---:|---:|---:|
| 1 | 87/100 | 99/113 | 0.890 |
| 2 | 90/100 | 102/113 | 0.915 |
| 5 | 92/100 | 104/113 | 0.935 |
| 10 | 92/100 | 105/113 | 0.940 |

This is not an ARC-AGI benchmark score. It establishes a strong static anchor
only for this synthetic mechanism-track cohort. Family-ID disjointness does
not imply semantic or program-distribution disjointness from checkpoint
pretraining.

## Boundary and replay evidence

- dataset audit ID:
  `dd294946bba1fba41e783011965a312df52dd01bd85bf581fdfd01336ee84128`;
- run receipt ID:
  `94cbc61b67ce97f042a60ddad8b2df801a3a0b4e62ac0093f8db076d87bbe280`;
- candidate freeze ID:
  `1f6979b548c1623940eab563a7550895060496c0efd04b84ff36acd19ca337d2`;
- gate summary ID:
  `55251c16a95dfac55ebe76008bd027b97cc423fc27144541d27f3f617fe99630`;
- hidden full-result commitment:
  `6df6fa44649a6506f4e31cd543edcfdac1db4590963bbdd6dffbae81e3cf29c5`.

All three included JSON artifacts reproduced byte-for-byte. The external
candidate freeze also reproduced byte-for-byte with SHA-256
`69430b414afe5b8e66d284dad942c0fe7c3d11b025ae0e85403b007811b587ad`.
It contains 437 content-distinct candidates across 113 queries; top-5 to
top-10 adds no strict task recovery.

The dataset audit reconstructs all 48,498 training pairs to demonstrations,
all 13,728 test inputs to query inputs, and finds zero demo/query input
collisions. Test-label semantics were not opened by that audit. Candidate
construction was frozen before project-side solution scoring.

## Corrected completion semantics

Upstream metadata estimates 6,288 steps, but exact Philox schedule replay
emits 6,186 full batches, consumes all 200,000 group-order entries, and drops
a final partial batch of 31 examples. The run observed exactly 6,186 steps,
evaluated 108 batches, exited zero, and took 3,260 seconds on one A800.

See `notes/design/nvarc-trm-family-disjoint-anchor-v3.md` for the transparent
post-run correction and `notes/results/nvarc-trm-arc-tgi-anchor-20260811.md`
for interpretation.
