# arc_functional_transition_solver

Updated: 2026-08-12

## North star

Build an ARC system in which independently strong, heterogeneous specialist
systems maintain their own representations, while a shared workspace recruits
the next specialist under a native-compute budget from the current population
of executable hypotheses and failures.  Functional switching must change the
reachable candidate frontier or validated selection outcome; changing only the
order of weak modules is not sufficient.

## Primary falsifiable claim

`C-000`: Under a strict native-compute budget, query-gold-blind recruitment over
a content-addressed population of heterogeneous frozen providers recovers more
strict task-level oracle coverage and final pass@2 per unit compute than the
best fixed provider order and equal-cost restart.

## Scientific value

The project separates two uncertainties that earlier experiments conflated:
whether specialist candidate distributions are complementary, and whether an
online controller can exploit that complement cheaply.  A negative result can
therefore falsify budgeted functional recruitment without erasing the value of
strong provider ensembles or the reproducible candidate-audit infrastructure.

## Success envelope

- Evidence threshold: on a fresh, family-disjoint 100-task cohort, at least two
  provider families must each contribute strict task-level coverage, the added
  provider must contribute at least 3 exclusive tasks, and a query-blind 30%
  recruitment policy must recover at least half of the observed union gain and
  at least 2 strict tasks over the cheap anchor while beating deterministic
  random-allocation and equal-cost restart controls.
- Falsification threshold: zero or one exclusive task from the added provider,
  or a valid recruitment experiment that gains fewer than 2 strict tasks and
  does not beat equal-cost controls, sharply narrows or rejects `C-000` for the
  tested provider pair and cohort distribution.
- Reproducibility: candidate freezes precede solution access; every code, data,
  provider, checkpoint, configuration, cost receipt, and output identity is
  content-addressed; freeze and score replay byte-for-byte twice before a gate
  can close.
- Resource envelope: the current post-hoc normalization gate is CPU-only and
  capped at 30 minutes.  A new GPU cohort run requires a separate frozen
  protocol and may use at most two full provider runs per task, with no router
  training in this stage.

## Non-goals

- Do not optimize presentation before the central uncertainty is resolved.
- Do not promote exploratory signals into confirmatory claims.
- Do not claim ARC-AGI benchmark competitiveness from synthetic ARC-TGI data.
- Do not treat typed local repair as the mainline unless a frozen near-miss
  cluster predicts lower cost than provider recruitment or cold restart.
- Do not train GRU, diffusion, MLP, or XGBoost controllers before provider
  complement and query-blind recruitability pass their gates.
- Do not impose a common internal DSL on providers; only the evidence envelope,
  candidate identity, cost, and verification boundary are shared.

## Protected human decisions

The researcher owns changes to:

- the north star and primary claim;
- the project mainline and portfolio priority;
- protected architecture and irreversible design commitments;
- budget expansion, canonical-repository changes, and external release;
- interpretations that depend on tacit domain knowledge or scientific value judgments.

## Repository boundary

- Project root: repository containing `.research-control.json`
- Control root: `.research-control`
- Canonical implementation: `chaochao825/symbolic_logic`, branch
  `agent/arc-cognitive-workspace-v2-20260811`, project path
  `arc_functional_transition_solver` in the 210 worktree.
- Canonical data: ARC-TGI confirmatory cohort
  `b237836bbaf9340497292f632efb7c2208e140a877b75a5bd3783ff11b4e2a27`;
  its existing outcome is exposed and may be used only for development or
  post-hoc verification, never as a fresh confirmatory cohort.
