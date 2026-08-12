# EXP-004 pre-execution commitment

This directory contains the compact, query-gold-free commitment for the
collision-free prospective reserve experiment.

- Eligible cohort ID:
  `9706b3af57b19564e4579a20c29f7127271d253bab4e2e5c58657216681ffe15`
- Eligible seal ID:
  `d9c3c71ca6eb10289e3a64eeff3fedbec3107c6c8f947160ff50afc2edd9fe60`
- Eligible challenge file SHA-256:
  `60682d2ca38496987ddd85033ce9400cfb9a7838995310c1676a6e1156a6d864`
- Committed seal file SHA-256:
  `85af3256cf3b2e540238ac7252ce796e86766e11967300374b9f5027fc3ba3b5`
- Parent seal ID:
  `7c89b3e3cc3f41131b39b60b5bb5ad7d62a9032fabcf864407dc057e3b7e899b`
- Task count: 98
- Eligibility rule: `no-demo-query-input-collision/v1`
- Query gold written or opened: no
- Frozen protocol:
  `.research-control/experiments/protocols/EXP-004.md`

The two excluded tasks were selected solely because a query input exactly
equals a demonstration input.  Provider outcomes played no role.  `EXP-003`
remains an invalid protocol record and is not pooled with this experiment.

The primary ordering and gates are frozen before either provider outcome on
this cohort.  Recursive candidates are frozen before the visual run; the
visual provider cannot influence the recruitment plan.
