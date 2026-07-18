# Empirical Paper Outline Contract

No full paper prose should be drafted before the planned evidence is generated.
Citation quotas count distinct, directly relevant primary sources or official reports.

1. **Introduction**

   Define ARC as few-shot task induction, show the multiplicative generation-selection
   bottleneck, and state the matched-budget research question. Avoid biological claims.
   Target 6-8 citations and one overview figure. Contributions are phrased as tested
   findings only after result backfill.

2. **Related Work**

   Organize by technical gap: ARC benchmarks and refinement; masked discrete
   generation; program synthesis and execution guidance; object-centered
   neuro-symbolic reasoning; modular routing and logic-gate policies. Target 15-22
   citations. Explicitly distinguish contest-private, public, and semi-private results.

3. **Method**

   Specify the blackboard state, candidate IR, specialist interfaces, dynamic legality,
   controller, verifier ladder, and grid/program repair operators. Include equations
   and two structural figures. Cite prior components; claim novelty only for the tested
   coupling.

4. **Experimental Protocol**

   Declare pinned data revisions, sealed splits, exact pass@1/pass@2, budgets,
   provenance, seeds, baselines, and leakage policy. Target 3-5 methodological
   citations and one dataset/protocol table.

5. **Candidate Coverage and Verification**

   Report per-source oracle coverage, complementarity, duplicate rate, shape coverage,
   and hard-filter false-negative recall. One verified table and one coverage-cost
   figure are required.

6. **Ranking, Repair, and Functional Switching**

   Report frozen-pool rankers, grid/program repair versus restart, controller budget
   curves, and source allocation. Require at least two verified tables and two figures.
   Separate `verified`, `planned`, and `placeholder` content.

7. **Ablations and Failure Analysis**

   Include source drops, transition-mask variants, anti-overfit features, DLGN
   soft/hard behavior, parse/shape misses, false pruning, and timeouts. Require one
   ablation table and a task-level error taxonomy. Negative results remain visible.

8. **Limitations and Validity Threats**

   Cover DSL ceiling, candidate absence, public-task contamination, compute fairness,
   API dependence, verifier ambiguity, and the non-biological status of the functional
   analogy. Target 3-5 citations where external claims are made.

9. **Conclusion**

   State only verified findings with their dataset and budget scope. No broad AGI,
   neuroscience, state-of-the-art, or hardware claim without direct evidence.

10. **Appendices**

    Provide source manifests, candidate schema, DSL specification, prompts, full
    baselines, per-task outcomes, statistics, negative results, and provenance. Planned
    hardware mapping, if any, stays explicitly planned until synthesis artifacts exist.

## Citation and visual requirements

- Every factual prior-work or novelty-boundary claim has at least one direct primary
  or official source. Counts are diagnostics, not quotas; do not pad citations.
- The final map must still cover ARC benchmarking, masked/discrete diffusion,
  program synthesis/neuro-symbolic reasoning, algorithm portfolios/CEGIS/type
  constraints, and modular control.
- One architecture figure, one evaluation-flow figure, two result figures, and three
  result or ablation tables. Result visuals must be generated from real files.
