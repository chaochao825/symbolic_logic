# RDR: provenance-effect v0.4 implementation amendment

## Status

Accepted as a non-semantic implementation amendment after the first G1 run and
before the final reproducibility pair.  The frozen relation/effect language,
task allocation, cost reservation, thresholds, and gate order remain unchanged.

## Correctness repair

The first G1 implementation incorrectly rejected a mixture of base-exact and
single-delta demonstrations.  A legal effect may be a no-op on an example with
empty effective support.  Base-exact examples must therefore constrain the same
finite version space after the color delta is inferred from inexact examples.

This was the single implementation-only repair authorized by G0.  The invalid
artifact is retained with an explicit marker and excluded from all scientific
interpretation.  A controlled mixed exact/inexact test was added, and the
historical 12-task development audit remained 5/12 with no baseline successes.

## Serialization correction

The first corrected artifact was 386 MB because each crop cell was incorrectly
assigned every persistent relation incident to its entity; some summaries then
repeated more than 230,000 relation IDs.  These edges are not data producers for
crop pixels.  The corrected representation is:

- each cell records exact source coordinate/color and optional entity IDs;
- relation IDs on a cell are reserved for operations that actually consume a
  relation to produce that cell, so crop records none;
- `EffectSummary` content-addresses the full relation subgraph induced by its
  affected entities using a set hash and count;
- at most sixteen relation IDs are serialized as an audit sample;
- the complete relation set remains reproducible from the content-addressed
  persistent parse state.

This amendment changes provenance serialization IDs and artifact size, but not
rendered grids, effect candidates, strict LODO, baseline definitions, costs, or
gate outcomes.  The compact implementation must be run twice from the same
sealed cohort before it becomes the final artifact.
