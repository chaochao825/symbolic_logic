# Executable Abductive Workspace v0.1 Gate

## Decision question

Can a partial scene program be represented as a content-addressed typed sketch,
compiled by demonstration-only abstract execution into replayable proof
obligations, and completed through a type-legal hole action without changing the
legacy scene executor or any existing solver result?

## Candidate and lane

- Lane: Explore.
- Active candidate: `TypedProgramSketch + abstract_execute + ProofObligation`.
- This gate does not test ARC accuracy, coverage, repair efficiency, learned
  routing, or a biological mechanism claim.

## Frozen baseline and evaluator

- Source baseline: commit `fb22e1d` on
  `agent/arc-cognitive-workspace-v2-20260811`.
- Legacy evaluator: `execute_scene_pipeline` and existing scene/stateful tests.
- Valid pre-change baseline:
  `PYTHONPATH=src python -m pytest -q tests/test_scene_graph.py tests/test_stateful_scene.py`
  produced `14 passed`.
- An earlier invocation without `PYTHONPATH=src` was invalid at collection and
  carries no code or method interpretation.

## Data and leakage guard

- Controlled synthetic grids in unit tests only.
- Abstract execution accepts `BlindTask` and reads training demonstrations only.
- Query outputs cannot enter the interface; `ARCTask` is rejected.
- No exposed development endpoint may be used to claim solver improvement.

## Mutable surface

- New executable-workspace module.
- New focused unit tests.
- This gate note.
- No changes to legacy program serialization, candidate hashes, provider search,
  router, budget accounting, checkpoint formats, result artifacts, or default
  solver wiring.

## Primary endpoint and hard guards

Primary endpoint: the controlled fill-hole loop changes
`incomplete -> exact` using a type-legal binding and the unchanged executor.

Hard guards:

1. complete sketch materialization is exactly equal to the source
   `ScenePipelineProgram`;
2. wrong-type bindings fail before execution;
3. complete shape, support, render, parse, and selection failures produce the
   expected typed obligation families;
4. sketch, obligation, and result identities are deterministic and content
   addressed;
5. the valid legacy baseline and full test suite do not regress;
6. no existing result artifact is modified.

## Cost cap and stop rule

- One implementation of the frozen interface and bounded correctness fixes.
- CPU unit tests only; no model training or dataset-scale search.
- Stop if satisfying the gate requires changing legacy execution semantics,
  widening a provider search, adding a learned controller, or inspecting query
  outputs.

## Outcome mapping

- Pass: all primary and guard tests pass without legacy changes.
- Boundary: typed holes work, but complete execution failures cannot be compiled
  beyond a generic executor obligation.
- Null: sketches serialize but cannot support a legal incomplete-to-exact loop.
- Adverse: legacy behavior, determinism, or leakage/fairness guards regress.
- Engineering failure: implementation or harness prevents testing the contract;
  no method belief update.
- Invalid: wrong source identity, modified evaluator, contaminated inputs, or
  missing provenance.

## Next decision if passed

Open exactly one new gate for a bounded abstract reachable-set domain and a
single topology hole (`insert_typed_node`).  Do not connect a learned router or
claim natural-task repair before that gate succeeds.

## Closure

- Classification: **boundary**. The primary engineering endpoint passed, while
  the unfiltered full-suite guard conflicted with the explicit no-training cost
  cap because the repository suite contains M04A and E01 training smoke paths.
- Valid targeted result: 23/23 tests passed locally and on server 210.
- Valid dependency result: 49/49 statically selected direct-dependency tests
  passed in 47.53 seconds on server 210.
- Integrity result: the SHA256 identities of `scene_graph.py`, hybrid
  `__init__.py`, and `pyproject.toml` remained exactly equal to the frozen
  baseline. No legacy source file, provider, router, budget contract, or result
  was modified.
- Invalid attempts are preserved separately. Missing `PYTHONPATH` invocations,
  SSH transport timeouts, the pre-refinement candidate, and suites that entered
  training paths carry no pass/fail interpretation for the final candidate.

### Supports

- A content-addressed partial scene program can expose typed holes without
  changing the legacy executor.
- An incomplete sketch compiles into a type-legal fill obligation, and filling
  it can produce `incomplete -> exact` under unchanged demonstration execution.
- Complete shape, support, render, parse, selection, and execution failures can
  be separated into typed obligation families in controlled examples.
- Wrong node classes and incompatible cross-node refinement contracts fail
  before execution.

### Does not support

- Any improvement in ARC accuracy, oracle coverage, selectable coverage,
  pass@2, or repair efficiency.
- Abstract reachable-set reasoning, AST topology insertion, natural near-miss
  recovery, visual-posterior causality, dynamic functional switching, or a
  biological mechanism claim.

### Unknown

- Whether the current obligation taxonomy identifies causal repair locations on
  fresh natural ARC failures.
- Whether a topology-changing action can open a novel semantic candidate region
  more efficiently than equal-cost cold restart.

### Decision

`revise-protocol`: the next gate must use the frozen direct-dependency regression
set as its mandatory engineering guard and keep training suites out of scope
unless a future candidate actually changes a training path. The single next
scientific candidate remains bounded abstract reachability plus one
`insert_typed_node` hole.
