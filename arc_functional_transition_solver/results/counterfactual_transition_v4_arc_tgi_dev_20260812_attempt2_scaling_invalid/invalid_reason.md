# Attempt 2 status: invalid incomplete scaling run

Source commit `ed15b8a78137ec9176836e6129e466d07abb6a76` passed all
18 semantic tests and corrected the residual-improvement metric. Two independent
eight-worker freezes then ran for more than 50 minutes. In each freeze, seven
workers completed their queues while the same high-cardinality grammar task
remained CPU-bound in one worker. Neither candidate freeze was materialized,
no solution was read, and the incomplete run supports no scientific conclusion.

The first profile identified repeated serialization and hashing during grammar
grouping and residual-beam extension. Later bounded probes found two additional
costs on high-object-count inputs: relation materialization during raster-only
first-stage scoring and repeated stateful suffix execution. The retry therefore
precomputes each grammar program's typed node tuple, program ID, and ordering key,
reuses the complete grammar once, and omits unused pairwise relations only from
grammar construction and raster-only scoring. Stateful replay still constructs
the complete persistent relation state. The authoritative serialized
node-difference check still runs for every selected transition, so these changes
alter neither candidate membership nor ordering and introduce no fallback or
wall-clock task omission.
