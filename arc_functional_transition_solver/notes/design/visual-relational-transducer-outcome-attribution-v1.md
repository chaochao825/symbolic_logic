# Visual relational-transducer outcome attribution — 2026-08-11

This pre-gold clarification adds a stronger, secondary causal audit to the
frozen 100-task confirmation. It does not change the representation grammar,
candidate cap, visual score, allocation fraction, cold seed, native-cost
accounting, primary gate, or any existing artifact.

The query-blind allocation ablation already freezes three equal-size task sets:

1. the observed visual-posterior allocation;
2. the allocation after clearing only posterior preservation mass;
3. the allocation after clearing every certificate feature.

After all three sets, candidates, costs, and content identities are frozen, a
separate scorer applies the same two-candidate-per-task static-family frontier
to each set. It reports unique recovery beyond the frozen VARC/NVARC union.

The original confirmation gate remains unchanged. A stronger claim that the
visual posterior improves recovery is permitted only when the observed visual
allocation recovers strictly more tasks than the posterior-cleared allocation.
If the task set changes but recovery does not improve, the evidence supports a
causal action effect but not a causal performance contribution. If the observed
allocation beats cold but not the posterior-cleared allocation, any advantage
must be attributed to the composite static descriptors or their interaction,
not to posterior preservation mass alone.

This audit is secondary because it was added after launching blind visual
inference, but before candidate freezing or access to confirmation solutions.
Its protocol hash must be bound into the candidate freeze, and its outcome must
be replayed byte-for-byte before interpretation.
