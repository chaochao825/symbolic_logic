# E01a next-source diagnostic after M02c/M05f

## Status and decision

This is a post-hoc, input-only source-selection diagnostic. It does not change the
frozen M02c/M05f result, does not score a new candidate pool, and is not holdout
evidence. Its purpose is to decide whether a fifth consecutive one-task symbolic
slice is more informative than freezing the first non-symbolic candidate source.

The decision is to pause task-by-task symbolic expansion and pre-register M04a, an
independent global masked-grid candidate source. Local residual-directed remasking
remains a later M11 experiment.

The evidence motivating this decision is cumulative:

- the fixed 20-task smoke now has 8/20 strict and 8/21 micro-pair coverage;
- all 12 remaining tasks have zero raw DSL candidates;
- the M02b/M05c, M05d, M05e, and M02c/M05f slices each added exactly one task;
- M02c/M05f added only one public proposal and one unique output at unchanged
  48,177 expansions and 218,637 ordinary program executions.

These observations do not show that the symbolic source is exhausted. They show
that another target-selected primitive would provide little new evidence about the
multi-source architecture requested by the project.

## Frozen scan inputs

- Repository: `external/ARC-AGI-2`.
- Commit: `f3283f727488ad98fe575ea6a5ac981e4a188e49`.
- Scan root: `external/ARC-AGI-2/data/training/*.json`.
- Files parsed: 1,000; repository status was clean.
- `0becf7df.json`: SHA-256
  `24983d27ad792b6a2efe3c696ec3503a3a3e44a0615474bc71fe8c4bc2c72c8d`.
- `fcc82909.json`: SHA-256
  `5c1504b43de9696ef5934bd4be2f5694a65be3996010385c06736723a8a2f080`.

The fixed 20 development tasks were not excluded. For every task, the scan visited
all demonstration inputs and all test inputs:

```text
grids = [pair["input"] for pair in task["train"] + task["test"]]
```

The JSON files necessarily contain outputs, but the predicates never accessed or
compared a `pair["output"]`. The result is therefore input-only but still post-hoc:
the signatures were designed after examining the diagnosed development tasks.

Both scans use the same simple diagnostic background rather than the project's
multi-hypothesis parser: the most frequent color, with the smaller color selected
on a frequency tie.

## Strict corner-legend signature

For each input grid, inspect the four named 2 by 2 corner windows. A window matches
when all four cells differ from the modal background, all four colors are distinct,
at least one non-background cell exists outside the window, and every outside
non-background color belongs to the four-color window. A task matches only when
every train and test input has exactly one matching corner; the corner need not be
the same across examples.

The root reproduction returned:

```text
files 1000
legend 1 ['0becf7df']
```

For `0becf7df`, all four inputs use background 0 and uniquely match the top-left
corner. The signature does not infer pair orientation inside the legend and does not
execute a color swap.

## Strict block-to-bar compatibility signature

For each input grid, form color-agnostic 4-connected non-background components. A
grid matches when it contains two to four components; every component fills a 2 by
2 box; every component has at least two distinct input colors; a two-column region
of height equal to that distinct-color count fits immediately below the component;
every such region is background in the input; and the destination regions do not
overlap. A task matches only when all train and test inputs match.

The root reproduction returned:

```text
files 1000
bars 1 ['fcc82909']
```

For `fcc82909`, the three train inputs have component/count summaries
`2:[3,2]`, `2:[2,3]`, and `3:[4,2,2]`; the test input has `3:[2,4,3]`.
The signature does not read the output, choose a render color, or prove that the
downward direction is a general rule.

The correct interpretation is narrow: these two strict, post-hoc input signatures
each identify only their diagnosed task in this pinned 1,000-task public-training
tree. This suggests that implementing either exact contract would directly target
one known task. It does not show that a broader legend or feature-to-render
representation would also be single-task.

## Remaining symbolic candidates

| Priority if symbolic work resumes | Task | Required representation/action | Reason not selected now |
|---:|---|---|---|
| 1 | `0becf7df` | D4-normalized corner legend plus an input-bound scoped color map | The strict signature is unique to this task; pair orientation still needs a target-derived contract. |
| 2 | `fcc82909` | 2 by 2 multicolor blocks, distinct-color feature, and feature-sized adjacent rendering | Direction and output color are not input-identified without a broader binding design. |
| 3 | `d2acf2cb` | aligned delimiter spans, max selector, region mask, and scoped color permutation | Span/mask semantics are not yet unambiguous; a historical upstream manual audit also reported demonstration 2 as input/output-swapped, while warning that audited versions may differ from later repository commits. |
| 4 | `1b8318e3` | marker-anchor bipartite relations, distance/contact slots, joint assignment, move and erase | This requires a substantially larger action and conflict policy than copy-only M02c. |

The historical `d2acf2cb` caveat is recorded in
[ARC-AGI issue #123](https://github.com/fchollet/ARC-AGI/issues/123); it is a reason
for caution, not proof that the pinned ARC-AGI-2 file retains the reported defect.

## M04a handoff boundary

The planned M04a contract would initialize a categorical 0-9 query grid from a full mask and condition on
demonstrations, the query input, and a deployable non-oracle shape hypothesis. It
must freeze the data split, checkpoint, tokenizer, seeds, mask schedule, denoising
steps, temperature, and candidate budget before oracle-stage scoring. Fixed-smoke
tasks and their augmentations are excluded from training, and query outputs, query
output shapes, oracle residuals, oracle-selected seeds, and oracle early stopping are
prohibited.

The first slice measures global generation and matched-call cold restarts only. It
records raw, format-valid, unique, duplicate, and DSL-novel candidates plus forward
calls, mask counts, GPU time, and peak memory. The public diagnostic gate is at least
one correct output, after canonical deduplication against DSL v0.6, among the 12
currently empty tasks. A replayed zero is an admissible negative result. M04a alone
cannot establish local-repair benefit, heterogeneous complementarity, learned
functional switching, holdout transfer, or public-evaluation performance.
