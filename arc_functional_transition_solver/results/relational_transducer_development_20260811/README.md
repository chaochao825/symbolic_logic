# Visual relational-transducer development gate

This directory contains the compact result for the 49-task, collision-free
ARC-TGI development amendment. The complete candidate freeze and task-level
score remain in the external, content-addressed artifact store.

## Outcome

The raw VARC/NVARC top-10 union solves 46/49 tasks. The frozen 24-program
relational-transducer family recovers all three remaining tasks. The
query-blind visual allocation selects 15/49 tasks and obtains three unique
recoveries; the pre-existing seeded cold allocation selects 15/49 tasks and
obtains zero.

| Arm | selected tasks | program trials | demo executions | query executions | novel frontier | unique recoveries |
|---|---:|---:|---:|---:|---:|---:|
| visual typed | 15 | 360 | 2,160 | 1,080 | 3 | 3 |
| cold restart | 15 | 360 | 2,160 | 1,080 | 0 | 0 |
| static family audit | 49 | 1,176 | 7,056 | 3,528 | 3 | 3 |

The three recovered programs are generic members of the frozen grammar:

- identity coordinates plus a global color homomorphism;
- identity coordinates plus row foreground count with offset `+2`;
- vertical-axis reflection plus a global color homomorphism.

Every admitted candidate is demo-exact and content-distinct from the frozen
VARC/NVARC pool. Candidate construction does not read query gold. Scoring does.
Two freezes and two scores are byte-identical.

The direct allocation ablation is also query-gold-free and byte-identically
replayed. Clearing only the visual-posterior term while retaining static pool
descriptors replaces 7/15 selected tasks. Clearing all certificate features
replaces 9/15. Thus the posterior changes actions, but a separate post-freeze
outcome audit finds no development recovery advantage from that term: the
observed and posterior-cleared allocations each recover three tasks, while the
all-features-cleared allocation recovers one. The positive visual-versus-cold
comparison therefore cannot be attributed to posterior preservation mass.

## Interpretation boundary

This is a development-set mechanism signal, not a passed confirmation gate,
an ARC-AGI result, or an end-to-end pass@2 gain. The development solutions and
provider outcomes had already been exposed before this family was evaluated.
The result justifies keeping the representation and allocation protocol
frozen for the untouched 100-task confirmation cohort. Confirmation still
requires at least five unique recoveries and strictly more recoveries than the
equal-native-cost cold allocation.

External artifact commitments:

- candidate freeze ID: `e6da04fe59388978ce94f7a68617531f6b1cb5d56ed108d98fe7cba69e01902a`;
- candidate freeze SHA-256: `76a208e8e2e963e9bdd483ccd0a5387958dcb44e533308d5954df692a86d83af`;
- result ID: `7901e74eab8f8cc17b0a5528879208e44ca1197642bbcf03d547dfac75273583`;
- result SHA-256: `5aa644391f2d7c78b13850434b6b2fc45c9aa97443907859cc2577207c8a9925`;
- allocation-ablation ID: `fb4c8d94a674fead99798888261a2499f539070cd4616f9acd2d8b924b44f9a8`;
- allocation-ablation SHA-256: `8e5c05eeede0e07109277d9e42a95aa1dc3dd491d57c6bbb451a29e5472d1a66`;
- allocation-outcome ID: `c94b63d6dbdf2b88bb76605afe5448c46dad951ea03e01da921c28cad159ec0e`;
- allocation-outcome SHA-256: `2af472fab500a4acdeef7f296f44614ed9c59104e47f52535d4159ecf3418751`.
