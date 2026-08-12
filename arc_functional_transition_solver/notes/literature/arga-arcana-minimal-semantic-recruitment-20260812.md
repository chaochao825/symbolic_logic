# ARGA, ARCANA, and minimal semantic recruitment

## Evidence maturity

ARGA is a reproducible object-centric program-synthesis baseline.  Its paper
evaluates a manually selected set of 160 ARC-1 tasks covering object movement,
recoloring, and augmentation.  ARGA solves every demonstration for 63/160 tasks
and the held-out test input for 57/160.  The 2020 Kaggle winner reaches 64/160 on
the same subset.  ARGA explores about 7,505 unique search states on its solved
tasks, compared with roughly 2.25 million for that baseline, although wall-clock
time is higher because the public ARGA implementation is Python.  Removing
constraint acquisition changes held-out solves from 57 to 55 while increasing
the average explored states from 7,505 to 12,114.  Its strongest evidence is
therefore efficient search inside a fixed object-graph language, not general ARC
coverage or dynamic representation recruitment.

Sources:

- https://arxiv.org/abs/2210.09880
- https://github.com/khalil-research/ARGA-AAAI23

ARCANA describes a more directly related architecture: object-centric scene
graphs, a 47-primitive latent program decoder, exact symbolic execution, a dense
failure-refinement signal, a shared differentiable blackboard, and a learned
meta-controller.  The arXiv v1 reports 32.5% pass@2 accuracy on an unspecified
120-task ARC-AGI-2 semi-private split, versus 26.0% for its strongest listed
baseline.  On a separate 120-task public ablation set, removing reflective
refinement lowers 35.8% to 25.0%, while replacing the learned meta-controller by
a fixed schedule lowers it to 33.0%.

Those ARCANA numbers are not currently an auditable anchor for this project.
The seven-page paper does not identify the 120 task IDs, the construction of the
semi-private split, the training corpus, the source of the ground-truth programs
required by its CVAE objective, or a code/data artifact.  Its reported dollar
cost cannot be independently reconstructed from the stated 4xL4/12-hour setup.
The method is useful as a hypothesis about latent failure-conditioned search,
but its quantitative claims require an independent release or reproduction.

Source: https://arxiv.org/abs/2607.09059

## Mechanistic comparison

ARGA searches a fixed union of hand-designed graph abstractions.  Its
constraints say which operations should be pruned, and a Tabu list temporarily
deprioritizes abstractions whose search trajectories deteriorate.  It does not
derive a proof that the active representation is missing a particular typed
computation, and it does not insert that computation into an existing executable
program.

ARCANA uses a binary pixel error map and counterfactual step deletion to produce
a dense refinement vector.  That vector shifts the CVAE program prior in the
next turn.  This can be effective when a trained latent program manifold already
contains the right repair, but the feedback does not provide a discrete witness
that a representation-changing action is legal, minimal, or frontier-changing.

The current project is weaker in candidate generation but stronger in
auditability.  Its potential contribution is not another object graph or shared
blackboard.  It is **proof-carrying representation recruitment**:

1. retain one content-addressed executable interpretation;
2. compute a finite version space for the active representation;
3. use emptiness plus a typed failure core to authorize the smallest semantic
   extension;
4. insert the extension on declared dataflow edges;
5. replay the affected suffix and require a new output frontier;
6. compare unique recovery with an equal native-cost restart.

This is a counterexample-guided representation expansion rather than classic
CEGAR: the counterexample broadens the executable language by one typed function
instead of refining an abstract state partition.

## Minimal-semantic-recruitment principle

Let `L0` be the current bounded language and `L1, ..., Lk` be typed extensions.
For demonstrations `D`, define the exact version space

```text
V(L, D) = { p in L : execute(p, x_i) = y_i for every demonstration i }.
```

An extension is recruitable only if:

```text
V(L0, D) is empty,
V(Lj, D) is non-empty,
and no strictly smaller registered extension between L0 and Lj is non-empty.
```

For query input `x*`, recruitment is frontier-changing only if the image of the
new version space contains at least one output not already in the incumbent DAG:

```text
{ execute(p, x*) : p in V(Lj, D) } - incumbent_outputs != empty.
```

This separates three statements that must never be conflated:

- the system can change topology;
- the new language fits the demonstrations;
- the new language creates a useful natural-task frontier.

The first two are mechanistic prerequisites.  Only the third can improve a
solver.
