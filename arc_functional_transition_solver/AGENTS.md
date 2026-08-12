# Research Control Plane

Use this map for research-management work in this repository.

## Start here

Read, in order:

1. `.research-control/PROJECT.md`
2. `.research-control/STATUS.md`
3. accepted records under `.research-control/docs/decisions/`
4. the single file under `.research-control/docs/plans/active/`
5. `.research-control/research/claims.csv`, `.research-control/research/candidates.csv`, and `.research-control/experiments/registry.csv`

Treat authority as:

`PROJECT.md` > accepted RDR > active plan > experiment protocol > registries and results > chat.

## Invariants

- Keep exactly one project mainline and at most two bounded side probes.
- Attach every active line and experiment to a falsifiable claim.
- Require a cost cap, stop rule, and decision effect before execution.
- Keep exploratory, confirmatory, and integration work in their declared lanes.
- Preserve valid null, adverse, boundary, and contradictory evidence even when code is reverted.
- Do not treat a plateau, failed run, or attractive new idea as permission to pivot.
- Ask the researcher before changing the primary claim, mainline, protected architecture, budget, canonical repository, or external release state.
- Update the result record, registries, `STATUS.md`, and active plan when a gate closes.

## Mechanical checks

Invoke the installed skill script with an available Python 3 interpreter. For example:

```powershell
python "C:\Users\Administrator\.codex\skills\research-control-plane\scripts\rcp.py" validate . --strict
python "C:\Users\Administrator\.codex\skills\research-control-plane\scripts\rcp.py" status .
```

Use `$research-control-plane` for audits, recovery, portfolio decisions, gate design, bounded auto-research loops, and control-plane gardening.
