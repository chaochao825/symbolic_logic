# Object/code role reachability v0.2

## Trigger

The v0.1 gate exhaustively enumerated 3,252 programs on two disjoint 100-task
slices, but every program belonged to `d4_label_completion`. The executable
`role_stamp` family was unreachable because source and target roles were required
to disappear from every demonstration output before any canvas mode was tested.

That condition belongs to blank/erase rendering, not to object parsing. In
particular, it excludes a copy canvas that preserves its anchors.

## Bounded change

Provider version `afts-hybrid-object-code/v0.2` keeps the v0.1 program schema and
execution semantics. Only task-derived role enumeration changes:

- source candidates remain unique in every demonstration input;
- target candidates remain present in every demonstration input;
- payload candidates remain present in every demonstration input and output;
- disappearing source/target colors are ranked first, but persistent colors are
  no longer discarded before canvas replay;
- source, target, and payload lists are capped at 3, 4, and 4 respectively;
- demo-exact replay, hard verification, MDL ranking, content IDs, query blindness,
  and all repair budgets remain unchanged.

The change is opt-in through the new provider version. Legacy providers and v0.1
result artifacts are not rewritten.

## Evaluation boundary

The v0.1 offset-100 result was already observed and becomes the v0.2 development
slice. A blind, label-free enumeration preflight confirmed that the relaxed family
is reachable there. v0.2 confirmation uses the previously unscored offset-200
100-task slice and the same absolute hard gates. Its query labels are not used to
change v0.2 after this preflight.

No controller or neural provider is trained. Failure of unique coverage remains a
candidate-language result; success only authorizes the next representation phase,
not a dynamic-control claim.
