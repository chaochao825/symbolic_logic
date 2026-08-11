# Object-Graph Rewrite v2 QA

This QA run is bound to source commit
`afded52f17cfd90f220953308c8e0c4449a5e402`.

The new cognitive-workspace, object-graph rewrite, gate, and reserve tests pass
(`12 passed` remotely).  The complete repository invocation
`PYTHONPATH=src:tests python -m pytest -q` finished with:

- 546 passed;
- 3 skipped; and
- 6 failed in 386.04 seconds.

All six failures are availability failures in pre-existing M04a evidence tests.
The local evidence tree declares but does not contain protected payloads
`public_train_dev_blind/source_snapshot.zip` and
`m04a_global_source_v0_1/rearc_train_cache/data.bin`.  No M04a source or test was
changed by the bound commit.  This is reported as an incomplete private-fixture
environment, not as a green full suite and not as an Object-Graph Rewrite
algorithm failure.

Raw stdout/stderr remain protected publication metadata.  The public artifact
manifest records their hashes and byte counts without copying machine-specific
logs into Git.
