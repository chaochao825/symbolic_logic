# Exposure-audited query-blind visual-provider gate v2

## Status

Invalidated before gold scoring.  During query-blind prediction validation, the
raw VARC files contained ARC-domain-invalid samples: internal color `11` and an
empty grid.  The preregistered v2 contract declared any invalid grid fatal, so
v2 receives no scientific score.  The exact raw predictions are retained and
are evaluated only under the separately versioned v2.1 amendment, whose
candidate-rejection rule was frozen before any gold output was read.

## Decision

This supersedes the invalid v1 cohort before gold scoring.  The controller
remains frozen.  The experiment asks only whether a static VARC visual provider
adds independent selectable coverage to the existing portfolio on tasks that
were not used by prior project training, validation, or the aborted v1 dry run.

## Exposure contract

The explicit exposure registry has schema
`afts.visual-provider-exposure-registry/v1`, registry ID
`72c6bf70b2d0ac2104cfd141c899edffee53ea30b9a253ce0e9cc82f3f1fb285`,
and file SHA-256
`fdb70b6e5f6029acdc4800f56c5912e78bcdfd4786ac674955efb6740f51b90a`.
It contains the union of:

- 807 ARC-2 task IDs from the M04a `arc2_train` and `arc2_validation`
  semantic allowlists; and
- all 20 IDs touched by the aborted v1 infrastructure run.

The union has 811 IDs because 16 v1 IDs overlap the M04a allowlists.  Selection
also excludes all 767 ARC-AGI-1 overlaps and task IDs found in prior authored
notes and per-task summaries.  An independent set audit requires zero selected
overlap with every exclusion source.

After exclusions, 31 eligible ARC-2 public-training tasks remain.  The fixed 20
are selected by ascending
`sha256("visual-provider-v2-exposure-audited-20260807" + NUL + task_id)`:

```text
2ccd9fef e45ef808 f18ec8cc 1d61978c 8fff9e47
f8f52ecc 2a28add5 ac0c2ac3 b74ca5d1 b745798f
2b9ef948 f0f8a26d 37ce87bb aa62e3f4 30f42897
22806e14 6350f1f4 1b59e163 a09f6c25 412b6263
```

The cohort ID is
`146dd6947158d0042ee95f275d027a9902f388b7f7f83ad0c3a5220daf4f825a`.
The manifest SHA-256 is
`e136482415e3519155b3203c916810e23c0b9dadb72c197bf3a2c8497781757a`.

## Query-blind provider contract

- VARC source commit:
  `bd478ecf362e6499a988b05f33223e5c5fc6a6be`.
- Offline checkpoint SHA-256:
  `c52d719fe9a41066aa8e96548ae6f83a0abf3568facd5103e176d7005e8421b3`.
- Every provider-visible query output is an exact copy of that query input.
  Gold query files remain outside the provider root.
- The 1,040 provider JSON files contain 20 source tasks and 1,020 released
  perspective/color variants.  Every query in every file must satisfy the
  sentinel invariant; multiple-query tasks are explicitly supported.
- The released ARC-2 ViT settings are unchanged: 100 TTT epochs, one TTT
  replica, seed 42, 10 inference attempts, and nine color permutations plus
  the released basic geometric transformations.
- Predictions are frozen and hashed before the scoring process can read gold.
  The released script's internal score against sentinels is ignored.

## Comparison and endpoints

The exact same task IDs are run through a clean worktree at source commit
`8c855bf0c2a37c135c9ac0f68fbe2e3fd677bc7c` for the existing DSL/CA/scene
portfolio and object/code v0.3.  No controller parameters are trained.

The primary endpoint is visual-provider unique selectable coverage.  At least
1/20 advances to a fixed 100-task provider gate; zero stops immediate
integration.  Secondary endpoints are standalone raw coverage, pass@1/pass@2,
union coverage, output-shape agreement, closest same-shape Hamming distance,
candidate diversity, oracle rank, invalid-grid count, and GPU-minutes per task.

The run is invalid on any exposure-registry overlap, sentinel failure, source or
checkpoint hash mismatch, prediction freeze after gold access, repeated
OOM/non-finite loss, or more than 40 GPU-minutes for any task.

## Claim boundary

This is a Solver-track candidate-distribution experiment.  A positive result
does not support residual control or brain-like switching.  A valid zero-unique
result rejects this checkpoint/configuration as an immediately useful provider
on the pilot, not visual reasoning as a research direction.
