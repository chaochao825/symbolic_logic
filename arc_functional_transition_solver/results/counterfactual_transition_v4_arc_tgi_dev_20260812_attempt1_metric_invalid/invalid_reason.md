# Attempt 1 status: invalid diagnostic metric

Source commit `8ced4d7dd39192b48266d9f731228b97d9c1c6b0` passed the
17-test semantic selection, then started two query-blind freezes. The run was
terminated before either freeze completed because `improving_trial_count`
compared the complete ranking tuple, including description length and program
ID. A child with an unchanged execution residual could therefore be counted as
an improvement solely because its program was shorter.

This defect did not affect candidate generation, residual-beam ordering,
demonstration exactness, query-output novelty, or scoring. However, it could
overstate unit-compute repair evidence, so the incomplete artifact is retained
only as an engineering incident and supports no scientific conclusion. The
retry compares only exact count, shape agreement, mismatch count, and execution
validity when logging a residual improvement.
