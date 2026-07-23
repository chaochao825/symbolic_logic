# Functional router implementation

This module implements the requested ARC control flow as an executable,
oracle-free pipeline:

```text
DSL/code/masked-diffusion/sparse-CA/DiffLogic providers
  -> demonstration-only representation router
  -> whole-task hypothesis bundles
  -> demo-exact replay + MDL + built-in hard verification / external attestation
  -> residual-directed color/local-transition repair
  -> re-verification and at most two semantically distinct outputs
```

This document describes the preserved v1 provider and verification substrate.
The stateful v2 controller that recompiles actions after every execution result
is specified in `online-residual-controller.md`; `FunctionalRouterSolver` remains
the eager static baseline and `OnlineFunctionalRouterSolver` is the new default
CLI path.

## Trust boundary

`FunctionalRouterSolver.solve()` accepts `BlindTask` only. A public ARC file is
converted with `BlindTask.from_task()` before routing; public test outputs are
therefore absent from features, provider calls, repair, MDL, and selection. The
selected whole-task hypothesis is expanded into legacy `CandidateRecord` values
only at the end.

Every rule candidate carries a deterministic content ID, replay function,
description length, serialized specification, and hard-verifier interface. The
built-in DSL/CA verifiers reconstruct executable rules from their serialized
specification and re-execute all demonstrations and queries. Sparse CA additionally requires
direct categorical execution, constructive binary4 execution, and constructive
one-hot execution to agree with valid color decodes. Sparse tables must also
pass overall and non-modal foreground query-support gates, so a large background
cannot hide one unsupported salient cell. A grid-only diffusion bundle may enter
diagnostics, but it cannot pass the hard-rule gate.

## Providers

- `DslProgramProvider` reuses the typed DSL and bounded search. Identity and D4
  seed programs remain available as near candidates for repair.
- `SparseCAProvider` reuses the root categorical CA and bounded CA-program
  library. It explicitly enumerates `none`, `d4`, `bgpad`, and `d4_bgpad`; every
  augmented rule is replayed on the original demonstrations before admission.
- `MaskedDiffusionProvider`, `CodeModelProvider`, and
  `DiffLogicHardProvider` are explicit callback boundaries. With no configured
  inference source they return an auditable abstention. The framework itself
  never starts training or imports the DiffLogic training module. A configured
  callback is an explicitly trusted integration boundary and may execute
  external code; it must supply a content hash, the correct route/mode, a
  deterministic replay artifact where applicable, and a verifier attestation.
  The report does not misrepresent that callback-owned verifier as a framework-
  independent proof. External source labels are canonicalized to the provider;
  their original declaration is retained only as audit metadata.

The orchestrator centrally enforces unique provider names, the registered route,
route budget, candidate route, and deterministic MDL/ID ordering. If two providers
submit the same artifact ID with different canonical payloads, replay bundles,
or verifier results, every instance of that ID is quarantined. Clean candidates
and unrelated specialists continue, so provider order cannot select a winner in
an identity conflict.

The archived DiffLogic result files do not contain a complete, independently
reloadable task preprocessing/configuration bundle. Consequently the default
DiffLogic provider safely reports `missing_hard_circuit`; an integration must
supply an inference-only callback whose candidate includes hard replay, a
verifier attestation, and a self-contained circuit manifest. The manifest binds
the blind-task content hash, source commit, feature schema, state/configuration,
augmentation policy, positive bounded horizon, acyclic gate payload, and output
wiring. Circuit inputs must equal the declared feature width, and every gate is
one of the frozen 16 two-input Boolean functions (canonical name or index). The
framework recomputes both the circuit digest and the digest of every manifest
field (including extensions) and
charges the serialized circuit in MDL. This makes the artifact content-addressed
and auditable; execution of callback-owned preprocessing/replay remains the
explicit external trust boundary. Soft predictions or an unverified checkpoint
are not accepted.

## Router

The router uses only demonstration shapes, color transitions, D4 consistency,
and connected-component changes:

- object/relation/shape-change evidence prioritizes DSL/program synthesis;
- otherwise unexplained tasks prioritize an optional code model;
- same-shape, local, high-support transitions prioritize sparse CA/D4/bgpad;
- locally compressible transitions place verified DiffLogic hard circuits next;
- masked diffusion remains a broad optional candidate/repair source.

All specialists still return a receipt. Unsupported shape-changing CA and
unconfigured optional sources abstain instead of silently disappearing.

## MDL and repair

MDL charges the representation route, a provider-specific prefix code,
demonstration residual, and every unsupported query cell. DSL and CA both use
grammar code lengths rather than comparing verbose DSL JSON bytes against CA
theory bits. Demo-exact, hard verification, and CA support are gates, not soft
score bonuses.

Near candidates with a valid hard parent may receive two bounded repairs:

1. an unambiguous global color map learned from all demonstration residuals;
2. a radius-one local transition table whose changed entries have repeated
   support and no context collision.

Each repair records its parent and added description bits, composes with the
parent replay, reconstructs its repair table from the serialized specification,
and must pass the full demo-exact/hard gate again. Ambiguous maps, shape
mismatches, one-off local exceptions, and non-replayable parents abstain. Thus a
masked-diffusion source intended for repair must expose deterministic
checkpoint+seed replay; an output-only sample bundle remains diagnostic.

Selected per-query records remain in `candidate_records`. The report also emits
`lineage_candidate_records`, a parent-first, content-ID-closed ledger. Embedded
parent JSON is parsed back through `CandidateRecord.from_json_dict()` and its IDs
must exactly match the selected child's formal parent list.

## Run

From the repository root:

```bash
python arc_functional_transition_solver/scripts/afts_arc_hybrid.py path/to/task.json
```

The JSON report contains features, route reasons/budgets, provider abstentions,
all verification/MDL receipts, repairs, selected bundle IDs, final submission
`CandidateRecord` outputs, and the closed lineage ledger. It intentionally
contains no evaluation against a test oracle.
