# Visual posterior consensus bridge v1

## Trigger

The frozen v2.1 posterior-localization audit passed its exploratory gate:
median pixel-error AUROC was 0.929 and the fixed 10% disagreement mask had
5.96x macro recall lift over a random mask.  This licenses one deterministic
bridge test.  It does not license tuning several repair operators on the same
gold-scored cohort.

## Typed action

For each query, take the frequency-ranked top-1 valid visual grid and all valid
samples with the same shape.  The query-blind certificate is the per-pixel
fraction of those samples that disagree with top-1.  If every pixel has zero
disagreement, abstain.  Otherwise execute `posterior_consensus_compose`:

1. choose each pixel's modal color across the same-shape samples;
2. break a color-count tie in favor of top-1's color, then by lower color ID;
3. emit exactly one composed grid; and
4. content-address it and subtract the complete raw visual pool.

No gold, demonstration output, baseline result, threshold search, cropping,
recoloring, or additional neural inference participates in candidate creation.
The bridge artifact is frozen before gold scoring.

For a deterministic pass@2 policy, retain visual top-1 first.  Use the consensus
grid second when it differs from top-1; otherwise retain the original visual
top-2 candidate if one exists.

## Gate and boundary

The bridge advances only if it produces at least one novel candidate and at
least one task-level unique selectable recovery over the frozen base plus
visual top-2 result.  Report task-level unique raw recovery separately.

This is a frontier-change test, not yet a matched-cost repair result.  Even a
positive outcome does not satisfy the required 5/100 repair gate, compare with
equal-cost cold restart, or support a learned controller.
