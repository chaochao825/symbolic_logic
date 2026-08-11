# Relational-delta v0.2 terminal failure matrix protocol

## Decision boundary

This is a read-only, outcome-exposed development audit of the frozen
`afts-relational-delta-dsl/v0.2` candidate artifact.  It does not change the
provider, candidate order, budgets, scores, or any published result.  It uses
demonstration inputs and outputs only; query outputs and ARC-AGI-2 evaluation
tasks remain unavailable to the classifier.

The audit asks one question:

> At which earliest typed execution boundary do the 48 tasks without a
> demonstration-exact v0.2 program terminate?

The matrix is diagnostic evidence for choosing one subsequent representation
family.  It is not evidence that a new family generalizes or improves pass@2.

## Frozen inputs

- candidate freeze ID:
  `85811f28942fc447d529be60e217d4cd4b5326c485ea901dda3d19b983fe32e5`;
- cohort ID:
  `dfee67cf54613ba721e1fe46723967c95020715dff1a40500fe498055805e3ac`;
- DSL version: `afts-relational-delta-dsl/v0.2`;
- task inclusion: exactly the records whose frozen
  `full.demo_exact_program_count == 0`;
- expected task count: 48;
- query-gold access: forbidden;
- public-evaluation access: forbidden.

Every blind-task hash, grammar size, program ID, and source-contract hash must
close against the frozen artifact.  The matrix is constructed twice and must
be byte-identical before it can be written.

## Mutually exclusive terminal categories

Categories are assigned in this precedence order.  A program reaches a stage
only when it reaches that stage on every demonstration, so a rule that works on
one example but not the others cannot mask an earlier cross-example failure.

1. `canvas_incompatible`: at least one demonstration changes grid height or
   width.  The v0.2 executor is same-canvas by construction.
2. `parse_failure`: no enumerated program assigns its declared actor/support
   roles on every demonstration.  The current concrete terminal reason is
   `ambiguous_roles`; this category includes the role layer because it is part
   of the frozen parse contract.
3. `relation_missing`: at least one program assigns roles on every
   demonstration, but no program instantiates its n-ary relation on every
   demonstration.  The concrete terminal reason is
   `no_unique_correspondence`.
4. `ast_insufficient`: at least one program instantiates a relation on every
   demonstration, but no program completes a legal typed mask and render on
   every demonstration.  Render/type failures include target-side, conflict,
   bounds, overwrite, topology, and empty-effect violations.
5. `legal_but_inexact`: at least one program executes legally on every
   demonstration, but every such program has a nonzero demonstration residual.

The five counts must sum to 48, and every task must have exactly one primary
category.  The artifact also records non-exclusive evidence: execution reason
histograms, cross-demo stage reachability, identity residual, best legal
program residual, delta geometry, palette changes, and connected components of
the demonstrated delta.

## Interpretation and next decision

The most frequent terminal category does not automatically authorize a code
change.  A subsequent representation family is eligible only when:

1. its semantics explain a coherent subcluster using demonstration evidence;
2. it is not a task-ID special case or a relaxation of hard verification;
3. its candidate IDs are absent from the frozen v0.2 prefix;
4. a bounded grammar or proposal policy can be enumerated query-blind; and
5. success would change selectable coverage or typed recovery, not only pixel
   similarity.

If the largest category is heterogeneous, the next family is selected from the
largest coherent structural subcluster, with the uncovered remainder retained
in the matrix.  Only one family may be active.  It must be implemented as an
opt-in version so v0.2 artifacts and numerical behavior remain immutable.
