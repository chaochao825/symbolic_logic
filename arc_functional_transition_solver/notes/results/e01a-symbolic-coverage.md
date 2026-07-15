# E01a symbolic coverage evidence checkpoint

## Status and identity

E01a is verified as an infrastructure and bounded-symbolic-coverage experiment. It
does not satisfy E01b or Gate 1, because diffusion, code-model, and independent rule
sources have not yet been frozen or compared.

This file records the historical v1 checkpoint. The failure-driven M03a/M05b
successor is verified separately in `notes/results/e01a-shape-expansion.md` and
`results/e01a_shape_v1`; it does not overwrite the identities below.

- Evidence root: `results/e01a_symbolic_v1`.
- Runtime source fingerprint:
  `3eb556bcfed1ea8a865a7bf154a465e449f53aeac0455e1bf019cc83629d82ad`.
- Test-source fingerprint:
  `c32377889b17d483d2ecf574c3fc66bc92f41384026ccf2bb9f43b064e937338`.
- Shared source-snapshot ZIP SHA-256:
  `8c00aaa620fc6238215597e82213a7c494d08cf5b043e6f1d7799d0817577210`.
- Full unit suite at freeze: 86/86 passing with warnings treated as errors.
- Both evaluation bundles passed `full_parent_replay_pass`.
- A separate read-only evidence process independently verified all eight artifact
  manifests, file hashes, JSONL row counts, parent bindings, and metric reductions.

## Generator-known synthetic control

- Case-set ID:
  `ba25a6f290114bcf17a950ce857b37619545ce40cb162f6a34d1d178f6e3b63b`.
- Pool-content ID:
  `45b8afb5e0fe36c0137ad3ba3979505e2fa8b6b4f475df0bb237a433adf8626c`.
- Eval-spec ID:
  `278a6ec2f4025befcb9589788bd9bcf2f959dec2ddca71a88783910b678b1844`.
- Evaluation artifact-manifest SHA-256:
  `05cbbdc200347f08393bfee82bfbc0ed7be99518fcde3685cf9d39732776c5ba`.
- 21 tasks, 42 query pairs, 35,380 expansions, 176,900 program executions,
  342 retained demo-exact programs, 674 candidate records, and 55 unique query
  outputs.
- Generator replay, generator evaluated, and generator retained rates are 1.0.
- Generator structural recall is 0.857143 at program rank 1 and 1.0 by rank 8.
- Semantic program coverage, task-first pair coverage, micro pair coverage, and
  strict task coverage are all 1.0 at output/program rank 1.
- Summed measured search wall time is 205.89 seconds; maximum per-task traced peak
  is 13,840,123 bytes.

This is a grammar-aligned positive control. It verifies generation, typing,
execution, bounded search, candidate emission, and evidence plumbing. It is not an
estimate of real ARC expressivity or compositional OOD generalization.
The three `crop_rotate90` tasks place the exact generator AST at structural rank 3,
while a semantically equivalent solver is already rank 1; this explains why AST
recall@1 is 18/21 but semantic/output coverage@1 is 21/21.

## Pinned ARC-AGI-2 public-training development smoke

- Dataset commit: `f3283f727488ad98fe575ea6a5ac981e4a188e49`.
- Case-set ID:
  `93fe4870982313fb840d27bc4082e079f41d8b58d4f679cea15b5e5b1381b177`.
- Pool-content ID:
  `62783de7c7180c88e87e3422fa52a995ffb643874950312503a1ac1ec2f1e9de`.
- Eval-spec ID:
  `f9ba18a92199f4a502b9874ed51e2f94df18c9997b1f4d191ab0d8d8eb7c4d9e`.
- Evaluation artifact-manifest SHA-256:
  `6ba38819e69a0d7137c84255b30bb3bbd96aa820f05348f39f5c3fba5235c92c`.
- 20 tasks and 21 query pairs, selected from the development bucket after fixed
  150-task holdout and 150-task validation reservations.
- 45,864 expansions, 209,061 program executions, 140 retained demo-exact
  programs, 93 candidate records, and 13 unique query outputs.
- Task-first output coverage is 0.10, micro pair coverage is 2/21 = 0.095238,
  strict task coverage is 0.10, and semantic solving-program coverage is 0.10.
- None of these coverage measures improves between rank 1 and rank 128.
- Summed measured search wall time is 1,139.91 seconds; maximum per-task traced
  peak is 25,462,241 bytes.

This is a repeatedly runnable public-training development smoke, not a holdout,
public-evaluation, pass@2, or release result. Exact semantic duplicates were
quarantined; D4/color-normalized near-duplicate auditing remains pending and is
recorded as a limitation.

## Interpretation and next action

The synthetic control passes while the real smoke saturates at 10%. Candidate
absence, not ranking, is therefore the immediate bottleneck. At this checkpoint,
the next slice was registered to:

1. classify the 18 unsolved smoke tasks by missing parse, shape, selector, primitive,
   binding, composition-depth, and search-pruning causes;
2. implement M02b lines, corners, frames, panels, motifs, and symmetry views;
3. extend the typed DSL with scale, tile, translation, concatenation, relational
   selectors, and explicit output-shape proposals;
4. compare an exhaustive small-cost search with beam search to measure false-prune;
5. rerun the same development smoke before adding a learned controller.

Ranker/controller training must not be used to hide this measured coverage ceiling.
The first registered shape/scale/tile subset subsequently raised the same post-hoc
development smoke to 20% task-first/strict coverage, and the independent M02b panel
overlay successor raises it to 25%. The subsequent indexed panel-sequence D4 slice
raises it to 30%. The subsequent periodic panel-lattice slice raises it to 35%,
and the additive M02c/M05f axis-ray bbox-contact slice raises it to 40%, leaving 12
tasks without a correct candidate while preserving all earlier solutions. The next
action remains broader candidate-source coverage rather than controller training. A
subsequent limited scan of the two cleanest diagnosed symbolic signatures found each
to be unique to its known task without excluding broader symbolic opportunities.
The current plan is therefore to pre-register M04a global masked-grid generation.
See `notes/results/e01a-bbox-contact-expansion.md` and
`notes/results/e01a-next-source-diagnostic.md`.
