# NVARC TRM anchor eligibility audit v1

## Decision question

Can the released NVARC TRM `step_220708` checkpoint be used as a clean,
reference-scale candidate-generation anchor on either the published 120-task
evaluation bundle or our frozen 50-task development cohort?

This audit precedes GPU execution. A checkpoint that was trained on query
outputs may still be useful for software reproduction, but its candidates and
scores cannot be used as held-out solver evidence or as the parent pool of a
repair comparison on the exposed tasks.

## Frozen sources and assets

- NVARC repository commit: `846d0198efa752534594e321fc3289fc0a06c657`
- TinyRecursiveModels submodule commit:
  `e7b68717f0a6c4cbb4ce6fbef787b14f42083bd9`
- ARC-AGI-2 repository commit:
  `f3283f727488ad98fe575ea6a5ac981e4a188e49`
- checkpoint: Kaggle `cpmpml/arc-prize-trm-031`, version 1,
  `step_220708`
- evaluation tensors: Kaggle `cpmpml/arc-prize-trm-evaluation-data`, version 1
- pretraining identifier/index metadata: Kaggle
  `cpmpml/arc-prize-trm-training-data`, version 1
- frozen project cohort: the 50 tasks in
  `results/control_legend_dev_20260810/result.json`

The 39 GB pretraining tensor bundle is not downloaded. The audit downloads the
37.8 MB identifier map, the two 4.2 MB row-index arrays, and exact byte ranges
for three preregistered tasks. The byte ranges are derived from the immutable
NumPy header and puzzle-index arrays.

## Preregistered probes

The probes are `e4888269`, `5adee1b2`, and `984d8a3e`. They were selected before
reading their rows because they cover two newly solved development tasks and
one ordinary failure. For each probe, all unaugmented pretraining input/label
rows must be decoded and uniquely matched against the official ARC-AGI-2 task.

## Eligibility rule

The checkpoint is a clean anchor only if all of the following hold:

1. the published 120 evaluation IDs have zero overlap with pretraining IDs;
2. the project 50-task cohort has zero overlap with pretraining IDs;
3. none of the preregistered query outputs occurs in pretraining labels;
4. checkpoint structure, size, tensor finiteness, source revisions, and asset
   hashes match the frozen contract.

Any failure sets `clean_heldout_anchor_eligible=false`. No score from an
ineligible task may be called held-out, and no GPU run on those tasks is
authorized as clean evidence. A separately labelled contaminated software
replication remains permissible.

## Follow-up if ineligible

Freeze an instance-fresh and family-disjoint mechanism cohort before adapting
the bridge. The replacement cohort must not share identifiers with the NVARC
training set and must keep generator witnesses, reasoning chains, and query
outputs on the evaluator side. It is a mechanism benchmark, not a replacement
for ARC-AGI-2 private solver evaluation.
