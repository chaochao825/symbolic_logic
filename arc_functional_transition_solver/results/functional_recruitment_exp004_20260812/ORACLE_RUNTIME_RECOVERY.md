# EXP-004 oracle-runtime recovery

The first closure attempt stopped before oracle authorization because the
closure script invoked the ARC-TGI generator with the frozen VARC environment,
which does not contain ARC-TGI's declared `shortuuid` dependency.

- Closure exit code: `1`
- Failure-log SHA-256:
  `1016d87e90cf13cf7c89af51cabc13d7ebd365466084c7cf6c6bcce2028bf47e`
- Oracle authorization created: no
- Solution artifact created: no
- Query gold read: no
- Frozen VARC candidate SHA-256:
  `afdec9c42aba3ab1db027cf5adcf155b974b1cf0c7f764cc9a91bcba5c2f987a`
- Frozen population SHA-256:
  `07d0e86503db5280ce8b798803522837f2d3ce76d3e950395046adfe3aec7`

The recovery changes only the interpreter used for deterministic ARC-TGI
oracle regeneration.  It uses the already recorded cohort-generation runtime
`/home/spco/sow_linear/.venvs/afts_arc_tgi_20260810/bin/python`, which is bound
in `results/arc_tgi_arcmini_cohort_v2_20260810/cohort_manifest.json`.  Provider
execution, candidates, population, recruitment plan, costs, random controls,
thresholds, and scoring semantics remain unchanged.

The failed closure products are preserved under the experiment-local ignored
`trash/` tree.  The closure is rerun from the frozen raw visual predictions;
all A/B receipt, candidate, population, authorization, and score comparisons
remain mandatory.
