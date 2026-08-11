# VARC runtime-alias neutrality result — 2026-08-11

The first compatibility proposal is rejected as numerically non-neutral.

- v2 partial prediction SHA-256:
  `7d139cb8869c398d7bb6254c3d0159178a9cbbdaf893cc622a6c653d44ae4cc0`;
- runtime-alias replay SHA-256:
  `0e59e2e39d38cf962f93ad3fcf591a5181955c10887b4234780e11693352a338`;
- both contain 510 raw attempts for the same query;
- v2 contains nine unique grids, while the alias replay contains one;
- ordered JSON and candidate sets are not equal.

The replay completed all 101 epochs and exited zero, so the original diagnostic
crash was fixed. However, changing filenames also changes model-visible task
names. Training trajectories diverged after epoch zero, and the available
evidence cannot distinguish that effect from the known nondeterministic GPU
path. Under the predeclared byte-equality rule, either explanation is enough
to reject the implementation.

This is an implementation/protocol failure, not evidence against visual
posterior proposals. The replacement v3 compatibility layer leaves every
model-visible name and byte unchanged and adds only the file read by the
upstream diagnostic after predictions have already been serialized.
