# E00 Phase-0 evidence (current)

Evidence status: `verified_harness_smoke_not_research_result`.

These runs validate strict ARC loading, immutable task/candidate records, the D4
generator, official two-attempt pair-fraction score semantics, strict-task
diagnostics, and atomic provenance bundles. They do not test the proposed diffusion,
DSL, repair, or controller contributions.

## Pinned inputs

- ARC-AGI-1: `399030444e0ab0cc8b4e199870fb20b863846f34`
- ARC-AGI-2: `f3283f727488ad98fe575ea6a5ac981e4a188e49`
- ARC scorer: `688c9be4fb270ad7eede09fe8bba6f8187be3bed`
- scorer blob SHA256: `924f0aee825835c916ec48dbb4874e50e5d5ad134608b2ec9d0787bf719ba583`
- package-bootstrap and executing-source fingerprint: `eb594c4cd0cce5847d94b2d76067ced82f63ee385bffbf5df0ad5818dca14618`
- source loader: direct source launcher, isolated import path, no site startup,
  fresh nonexistent pycache prefix, and bytecode writes disabled

## Smoke results

| Dataset / split | Tasks | Candidate rows | pass@1 | pass@2 | pair oracle coverage | strict oracle tasks | Run ID |
|---|---:|---:|---:|---:|---:|---:|---|
| ARC-AGI-1 public training | 400 | 3,064 | 0 | 0.0025 | 0.0192307692 | 0.0175 | `d6e897ce8b23ad61f95a` |
| ARC-AGI-2 public training | 1,000 | 8,072 | 0 | 0.0010 | 0.0074349442 | 0.0070 | `0566a4a62b9f5d8cc652` |

Each dataset audit is an atomic directory containing `audit.json`, the exact source
snapshot, and an artifact manifest. Each run directory additionally contains the
summary, run manifest, task and candidate JSONL, source snapshot, and an artifact
manifest with SHA256, byte count, and JSONL row count.
