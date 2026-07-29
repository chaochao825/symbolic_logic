# Object/code scene AST v0.3

## Decision boundary

The v0.2 result rejected a larger global-color grammar, not object-centric
reasoning in general.  v0.3 therefore changes the representation while keeping
the controller, neural providers, baseline task IDs, native budget accounting,
query-blind generation, and v0.1/v0.2 program execution frozen.

No router, code model, masked model, or learned proposal prior is trained in this
phase.

## Demonstration-only analysis

A read-only prototype inspected the frozen offset-100 task demonstrations and
query inputs, but not query outputs.  It found three task families outside the
baseline selectable set for which a bounded object/canvas grammar can fit every
demonstration:

- color-group role selection followed by crop (`0b148d64`);
- smallest-object selection followed by crop (`23b5c85d`); and
- object selection followed by an asymmetric padded crop (`3f7978a0`).

This is only a reachability hypothesis.  It is not oracle coverage evidence and
does not authorize task-specific rules.  Query outputs remain hidden until the
source commit and content-addressed candidate pools are frozen.

## Typed representation

The new `scene_pipeline` artifact uses object/code DSL v0.2 and scene AST v0.1.
Its fixed stages are:

1. `parse`: background, 4/8-connectivity, and monochrome-component,
   foreground-component, or color-group parsing;
2. `correspond`: optional equivalence classes using shape, size, topology, and
   relative-position features, with an explicit D4-invariance flag;
3. `select`: object roles such as smallest/largest, spatial extrema,
   hole-based roles, unique signature, and peer group;
4. `operate`: crop, copy, count, arrange, or compose, plus a typed D4 transform;
5. `canvas`: bbox, tight, input-sized, fixed, or count-line size rule with an
   explicit output background and bounded padding; and
6. `render`: source crop, selected-only, object composition, or solid count.

Each execution returns a node-level trace.  AST holes name flattened typed slots,
for example `ast.select.role` or `ast.canvas.padding`.  Old v0.1 `role_stamp` and
`d4_label_completion` payloads retain their exact schema and execution behavior.
The complete v0.2 grammar is preserved as an identical prefix before v0.3 scene
programs, so a 20,000-trial cap cannot silently change legacy trial order.

## Frontier-changing postcondition

A legal operator is not sufficient evidence that a search frontier changed.
Programs now have provenance-independent SHA256 content IDs.  Before a typed
repair is executed, programs already present in the frozen parent pool are
removed.  Diagnostics obey the invariant:

```text
frontier_changed == (novel_frontier_count > 0)
```

If no emitted candidate has a novel program ID, the provider abstains with
`no_novel_object_code_frontier`.  The audit records the raw frontier, novel
frontier, basis of novelty, and equal native costs against cold restart.

## Negative-result attribution

The development gate distinguishes the following cases:

- **implementation/protocol failure:** positive controls, source binding,
  replay, hidden-output isolation, legacy artifact parsing, or the novelty
  invariant fails;
- **search-budget failure:** the deterministic grammar exceeds the unchanged
  20,000-trial cap and no exact demonstration program is reached;
- **representation failure:** the full grammar is exhausted without an exact
  demonstration program;
- **inductive ambiguity/generalization failure:** an exact demonstration program
  exists, but frozen query scoring misses;
- **selectability failure:** an oracle candidate exists but hard verification or
  support eligibility excludes it; and
- **repair-theory failure:** novel typed actions execute correctly but do not
  beat an equal-native-cost cold restart.

The provider must add at least 3/100 unique selectable tasks before this phase can
be considered successful.  The broader selectable-union and natural-repair gates
remain unchanged, so passing the provider-only threshold does not by itself
authorize router training or a neural-provider claim.

## Reproducible test invocation on server 210

The active conda environment contains an unrelated `site-packages/tests`
package, while this repository historically uses both `tests.test_*` and bare
`test_*` imports.  The repository now marks its own `tests/` as a package, and
the complete server-side regression uses:

```bash
PYTHONPATH=tests:.:src:../src python -m pytest -q
```

This is a test-runner namespace fix only; it does not enter provider execution or
experiment artifacts.
