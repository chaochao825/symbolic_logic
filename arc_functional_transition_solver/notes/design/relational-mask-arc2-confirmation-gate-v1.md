# Relational-mask ARC-AGI-2 training confirmation gate v1

## Decision question

After the legacy scene-AST existing-slot gate returned zero reachable repair
opportunities on a complete fresh ARC-GEN scan, does an opt-in relational-mask
representation add content-novel query coverage and create natural single-slot
repair opportunities on unseen ARC-AGI-2 public **training** tasks?

This is a representation and actuator gate. It is not an ARC-AGI-2 public-eval
score, a visual-posterior experiment, a learned-router experiment, or evidence
for biological brain switching.

## Frozen representation

The representation is `afts-relational-mask-dsl/v0.1`. A program identifies a
background center jointly supported by one color at every member of a typed
direction set (`cardinal4` or `diagonal4`) and one or two radii in `[1, 4]`.
It writes a learned demonstration-visible output color on either an input-copy
or blank canvas. Its six existing leaf slots are:

- relation background;
- relation support color;
- relation direction set;
- relation radii;
- render output color;
- render canvas mode.

The module is opt-in and is not appended to the default object/code provider.
Consequently, legacy program order, candidate IDs, numerical behavior,
checkpoints, and published results remain unchanged.

## Source and exclusions

- ARC-AGI-2 source commit:
  `f3283f727488ad98fe575ea6a5ac981e4a188e49`.
- Only `data/training` is eligible. The 120-task public evaluation split is not
  read or scored.
- ARC-AGI-1 identities come only from the 400+400 files in ARC-GEN's locked
  ARC-AGI submodule commit
  `399030444e0ab0cc8b4e199870fb20b863846f34`.
- Exclude every task ID discoverable in authored repository evidence and all
  139 IDs in fresh ARC-GEN feasibility scan
  `459b1667edbfb6d9af5aa180f1a402b8309890991b6fa22bcf32decd758a1440`.
- At preregistration this leaves 50 task IDs. No task is added if an exclusion
  later reduces that set.
- Order by
  `sha256("relational-mask-arc2-confirm-v1-20260809" + NUL + task_id)`.

The cohort builder first creates blind payloads containing demonstrations and
query inputs only. Selection and structural signatures receive those blind
payloads, never query outputs. Gold files are content-addressed by a data
custodian path and are opened only after the complete candidate artifact is
frozen and replayed.

ARC-AGI-2 does not publish semantic family labels. The protocol therefore does
not claim ground-truth family disjointness. It provides the auditable substitute
of task-ID disjointness plus one representative per blind structural signature
(demo/query shapes, color-count multisets, component-size multisets, and visible
demo delta contracts). The signature of the development-only ARC-GEN family
`9f5f939b` is excluded. This limitation must remain explicit in reporting.

## Frozen budgets

- Legacy comparison: first 512 programs in the unchanged object/code order.
- Relational provider: complete bounded v0.1 grammar, hard-capped at 1,280
  programs; no query labels enter enumeration or ranking.
- Natural repair parent: highest-ranked execution-valid, non-exact program in
  the first 64 relational programs with demonstration agreement at least 0.5.
- Typed action: first 16 content-novel single-existing-slot variants allowed by
  the demonstration failure certificate.
- Equal-cost restart: next 16 relational programs after the same 64-program
  prefix.
- Both action arms reserve exactly 16 program trials and execute every trial on
  every demonstration and query input. Missing variants are padded by replaying
  the parent and cannot emit candidates.
- `novel_frontier_count` must be positive before an action is called
  frontier-changing.
- Candidate ranking is fixed MDL then program ID. At most two attempts are
  selected. Oracle coverage is reported separately and never used for ranking.

## Failure certificate

The compiler only uses demonstration input, parent execution, and demonstration
gold:

- invalid/no-center execution -> `relation_geometry` and the four relation
  slots;
- blank-canvas destruction with otherwise visible residual -> `render_canvas`;
- identical changed-coordinate mask but wrong values -> `render_palette`;
- all other valid residuals -> `relation_geometry`.

Every resulting candidate must differ from the parent in exactly one declared
existing leaf slot, be absent from the initial prefix, replay successfully, and
be exact on all demonstrations before it can be selected.

## Gates

The next visual/LODO experiment is allowed only if all representation gates pass:

1. the relational provider has at least two query-exact recoveries that the
   512-program legacy baseline lacks, and at least 4% unique coverage on the
   frozen cohort;
2. at least three tasks expose a non-empty, content-novel natural typed-edit
   frontier;
3. typed repair uniquely recovers at least three tasks beyond its 64-program
   prefix and exceeds equal-cost cold restart by at least one task;
4. every pre-gold candidate artifact is byte-identical on replay and all cost
   ledgers close exactly.

If raw relational coverage passes but repair does not, the result supports a
new provider but not residual control. If repair passes but unique provider
coverage does not, the mechanism remains interesting but the solver track does
not advance. If either first two preconditions fails, VARC/LODO and all learned
controllers remain frozen.
