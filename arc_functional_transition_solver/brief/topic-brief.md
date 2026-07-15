# Topic Brief

## Working title

**Execution-Aware Functional Switching for Heterogeneous ARC Candidate Search and Repair**

## Research question

Under a fixed per-task compute budget, can an execution-aware controller improve
ARC exact-match pass@2 by dynamically switching among masked grid denoising,
typed program synthesis, open-ended code hypotheses, verification, and local
repair, compared with any single solver and with static ensembles?

## Core idea

The solver is organized as specialist functions around a shared blackboard,
loosely analogous to functional specialization and switching among brain regions.
The analogy motivates modularity only; it is not a claim of biological fidelity.
Each specialist has a strong inductive bias:

- masked discrete diffusion proposes diverse grids and repairs selected regions;
- a typed DSL provides executable and verifiable transformations;
- an LLM or code model proposes hypotheses outside the hand-built DSL;
- deterministic checks and learned rankers filter candidates;
- a controller selects the next specialist from demonstrations, execution state,
  residual structure, candidate provenance, and remaining budget.

## Primary objective

Maximize official exact-grid pass@2 on held-out ARC-AGI tasks subject to a declared
wall time, accelerator, memory, and candidate budget. The official scorer assigns a
task the fraction of its test pairs solved and then averages across tasks. We also
report the stricter rate at which every test pair in a task is solved. The first
diagnostic objective is oracle candidate coverage; ranking and control cannot recover
a candidate that was never generated.

## Scope

- ARC-AGI-1 and ARC-AGI-2 static grid tasks.
- Multiple training demonstrations and one or more test inputs per task.
- Grid-, object-, relation-, program-, and trace-level representations.
- Exact execution, explicit provenance, and per-task refinement.
- Controlled synthetic compositional OOD tests plus sealed real-task audits.

ARC-AGI-3 is out of scope for the first project because it changes the problem to
interactive environments and would confound the static-grid research question.

## Audience and deliverable

- **Audience:** researchers in program synthesis, neuro-symbolic reasoning,
  generative modeling, modular learning, and ARC.
- **Deliverable:** an empirical method paper plus an open, reproducible solver.
- **Assumed length:** 8-10 main-text pages plus appendices.

## Evaluation constraints

1. At most two candidate answers per test input, matching the official pass@2 policy.
2. Exact match only per test pair; pixel similarity is a search signal, not score.
3. The current official ARC-AGI-2 repository is the dataset authority at run time.
4. Public evaluation tasks remain sealed during routine development.
5. Development uses a fixed split of public training tasks and synthetic OOD tasks.
6. Every result records code revision, data revision, hardware, wall time, seeds,
   candidate budget, and whether external APIs or pretrained data were used.
7. Contest-constrained, public-evaluation, semi-private, private, and commercial
   API scores are never compared as if they used the same protocol.

## Known evidence at project start

- Official ARC Prize reporting identifies per-task refinement loops as a defining
  2025 trend and reports a 24.03% top contest score on the private ARC-AGI-2 set.
- The ARChitects report demonstrates a 2D-aware LLaDA-8B masked-diffusion solver
  with recursive sampling, a separate shape predictor, and a multiplicative shape
  bottleneck.
- Execution-guided GridCoder2 outperforms non-execution-guided synthesis on a
  controlled compositional OOD experiment, but its real-task DSL coverage is small.
- Object-centered neuro-symbolic work supports separating perception, proposal,
  exact execution, and cross-example consistency.
- Modular-learning studies warn that the routing function, not the existence of
  specialists alone, is often the OOD bottleneck.

These points support the direction but do not verify the proposed system's primary
claim.

## User-provided evidence requiring recovery

The initiating material reports a synthetic ARC-like experiment comparing hard
DLGN, MLP, Boolean tree, heuristics, and memory baselines. The linked files used
`sandbox:/mnt/data/...` paths that do not exist in this workspace. Until the code,
configuration, raw per-seed outputs, and task generator are recovered, every number
from that experiment remains `unverified_user_report` and cannot support a paper
claim.
