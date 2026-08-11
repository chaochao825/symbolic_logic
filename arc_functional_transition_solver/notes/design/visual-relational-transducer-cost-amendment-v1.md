# Visual relational transducer cost amendment — 2026-08-11

The frozen v1 gate correctly fixes the same number of selected tasks and the
same 24 structural hypotheses per task, but its prose multiplies executions by
each selected task's own demonstration and query counts. Different visual and
cold task sets can therefore have different raw execution totals.

This arithmetic issue was found before implementing or running the transducer
gate. It is corrected without changing the representation, allocation score,
30% task count, cold seed, or hypothesis count:

- every selected task reserves and is charged 24 times the cohort-wide maximum
  demonstration count;
- every selected task reserves and is charged 24 times the cohort-wide maximum
  query count;
- missing executions are deterministic padding executions;
- visual and cold arms select the same number of tasks and therefore have
  identical reserved and charged native costs.

The static all-task family audit remains non-primary and reports its larger cost
separately.
