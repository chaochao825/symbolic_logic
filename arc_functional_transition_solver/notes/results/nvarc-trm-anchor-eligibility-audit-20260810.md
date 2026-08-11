# NVARC TRM anchor eligibility audit: 2026-08-10

## Decision

The released NVARC checkpoint is **not eligible as clean held-out evidence** on
the official ARC-AGI-2 evaluation tasks or this project's previously exposed
50-task ARC-AGI-2 development cohort. It remains eligible for software
replication and as a frozen initializer on post-checkpoint, family-disjoint
generated tasks, provided those new candidate runs are independently audited.

This finding changes the dataset used by the anchor experiment; it does not
change the checkpoint or claim that the upstream authors leaked a competition
secret. The released workflow deliberately pretrains on public ARC task
variants. Our stricter mechanism experiment cannot reuse those identities as a
held-out test.

## Frozen identities

- NVARC commit: `846d0198efa752534594e321fc3289fc0a06c657`;
- TinyRecursiveModels commit:
  `e7b68717f0a6c4cbb4ce6fbef787b14f42083bd9`;
- checkpoint `step_220708`: 2,159,719,349 bytes, SHA-256
  `dbf771737d9799b84faf009c24c5376831f095d2cfd66debbe9a505a445bfc31`;
- evaluation archive SHA-256:
  `735150bcc19f2e9f06b1ab30a2190b0bb969073121e763a62c63ba8104408d42`;
- training identifier archive SHA-256:
  `79a3a2cca7328aeb16bdcc803de82696e35178d94b9e49b5175abf416880b541`.

The checkpoint contains 15 tensors and 539,928,578 tensor elements. The
`1041208 x 512` puzzle-identifier embedding accounts for most of its 2.16 GB;
all tensors passed a complete finite-value scan.

## Exposure result

The published pretraining identifier table contains 1,041,208 entries,
including 4,073 original-task names and 1,120 official eight-hex ARC task IDs.
Identity overlap is:

- official ARC-AGI-2 public evaluation: **120/120**;
- this project's old exposed ARC-AGI-2 cohort: **50/50**.

This is not merely a name collision. Range-addressed reads from the 19.7 GB
training arrays found the exact query outputs in the pretraining labels for all
four inspected query pairs:

- `e4888269`: 2/2;
- `5adee1b2`: 1/1;
- `984d8a3e`: 1/1.

Downloading the full training arrays was unnecessary; the puzzle index and
identifier tables locate the byte ranges deterministically. The access plan,
range receipts, identifiers, hashes, and overlap rows are retained in the
content-addressed audit artifact.

## Consequence for claims

The following uses are prohibited:

- treating an NVARC score on the old 50 tasks as held-out generalization;
- comparing that score to our query-blind providers as if exposure were equal;
- using the official 120-task result to validate residual control;
- tuning a bridge on those scores and reporting a clean confirmation result.

The following remain licensed:

- reproducing the upstream software result with an explicit contamination
  label;
- checking checkpoint portability and numerical integrity;
- resetting task embeddings and test-time adapting on post-checkpoint,
  generator-family-disjoint tasks;
- freezing the resulting candidates before any solution access.

The audit also confirmed by source inspection that evaluation-time logits use
only `inputs` and `puzzle_identifiers`; labels enter the loss and metrics after
logit construction, and evaluation halting is fixed to ten steps. The fresh
anchor run must nevertheless freeze its submission before independent scoring
and retain the exact executed-source hashes.

## Evidence identity

- audit ID:
  `f9cc0868f0049e032583fb9d7a77c855b78d3c8c5839c1b2682a664411cca1da`;
- `audit.json` SHA-256:
  `180e62bbb542596632b090564eda3830df2b5ad4a3690584a7efea830453f12a`;
- training puzzle identifiers SHA-256:
  `c42158e3af3ec9241dfd02089790dbfd47006171ee2fa72a66fcbdaa27bd38c6`;
- training puzzle indices SHA-256:
  `287895d2c826589427b601d4edd4bf0c0b086dd617f94884ea0ca6d12b2ec919`.

Machine-readable evidence is under
`results/nvarc_trm_anchor_audit_20260810/`. Controller training remains closed.
