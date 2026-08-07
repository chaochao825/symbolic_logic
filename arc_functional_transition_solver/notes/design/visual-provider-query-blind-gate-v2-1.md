# Query-blind visual-provider gate v2.1: invalid-candidate amendment

## Reason for the amendment

The exposure-audited v2 model run was inspected before gold scoring.  Some raw
VARC samples contained internal padding color `11`; one sample was an empty
grid.  These values are outside the ARC output domain.  The original v2
contract therefore invalidates that protocol version and receives no score.

This amendment is an interface-conformance repair, not an accuracy-driven
change.  It was fixed without reading query labels.  It reuses the exact v2 raw
prediction files and does not rerun or retune the model.

## Frozen normalization rule

For every task and query, process raw samples in their emitted order.  Reject a
sample if and only if any of the following holds:

- the grid or a row is not a non-empty list;
- rows have unequal widths;
- height or width exceeds 30;
- a cell is not an integer; or
- a cell lies outside `[0, 9]`.

Do not crop, pad, recolor, coerce, or otherwise repair an invalid sample.  Rank
the remaining samples by frequency, then first-emission position, then
canonical grid JSON.  The run fails if any query has no valid sample.

Before gold is opened, freeze:

- every raw prediction-file SHA-256;
- a canonical hash of the complete raw payload;
- a canonical hash of the validated payload;
- raw, valid, and rejected sample counts for every query; and
- rejection reasons and counts.

The scorer uses only the validated, already-frozen payload.  Rejected samples
are provider abstentions and cannot count toward raw or selectable coverage.

## Claim boundary

Results from v2.1 must be labeled as a pre-gold protocol amendment.  They may
test whether the fixed checkpoint supplies useful legal ARC candidates, but
they are not the originally preregistered v2 result and are not an exact
reproduction of the released VARC evaluation script.
