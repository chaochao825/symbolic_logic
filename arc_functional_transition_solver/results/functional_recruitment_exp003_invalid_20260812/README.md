# EXP-003 invalid pre-GPU cohort audit

`EXP-003` stopped before any provider or GPU execution.  The NVARC dataset
compiler was deterministic and used no reserve solution file, but the strict
boundary audit found two query inputs that exactly duplicate a demonstration
input from the same task:

- `arc_tgi_task2Kfzy4hxyK4Wm6wcQfYvCC_r02_28fb1ef140e7588c`, query 0;
- `arc_tgi_taskFCUY2yh6ka8QS6WWHYZ7Un_r00_e66ba8c91c155c30`, query 1.

Every other violation count is zero: all 100 tasks are represented, compiled
training pairs reconstruct only demonstrations, query inputs reconstruct only
queries, and train/test example counts agree.  The two collisions nevertheless
break the intended held-out reasoning boundary and could reward memorization.
The audit therefore remains failed; its criterion was not weakened.

The protocol also cited the earlier A/B seal ID `618c...` while the committed
compact seal used for compilation has ID `7c89...`.  Their challenge bytes and
cohort ID are identical, but this is still a provenance mismatch.  No result
from `EXP-003` is evidence for or against provider complement or recruitment.

The correction is query-gold-blind: derive a new content-addressed 98-task
subset by excluding exactly the two collision tasks, issue a new seal and
cohort ID, then preregister a separate experiment without relaxing any method
threshold.
