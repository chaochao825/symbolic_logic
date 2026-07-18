# Audit of Claims in the Initiating Attachments

This file prevents convenient but incorrect claims from entering future drafts.

## Confirmed with scope corrections

- **NVARC 24.03% and ARChitects 16.53%:** confirmed as 2025 contest-private
  scores. NVARC used autoregressive TTT plus TRM components, not the 2025 masked
  diffusion model.
- **ARChitects LLaDA-8B, 2D positional modification, recursive soft masking,
  separate shape model:** confirmed. The 2D change lacks an isolated ablation.
- **ARChitects 30.5% with known shape and 85% shape accuracy:** confirmed as
  author internal evaluation estimates. Their live/public best was 21.67%; final
  contest-private was 16.53%.
- **GridCoder2 80%, GridCoder1 42.86%, NN-only 10%:** confirmed for a controlled
  seven-task synthetic OOD experiment. The comparison changes more than execution
  guidance, so it cannot isolate a causal 37-point execution-feedback effect.
- **DLGN is fast and discrete:** confirmed on classification workloads. No source
  establishes ARC routing superiority.
- **Light DLGN scaling limitations:** confirmed.
- **Poetiq 54%:** confirmed on official semi-private verification with a commercial
  model and high per-task cost, not contest private.
- **Johan Land 72.9%:** confirmed on official semi-private; the author's 76.11% is
  public self-evaluation. It is no longer the current maximum across all official
  model verifications as of 2026-07-11.

## Misattributed or overstated

- **“ARChitects white-box filters remove 35%-55% of candidates.”** The result is
  from the LongT5 ARC-AGI-2 report (`arXiv:2603.06590`) on an internal 177-task
  holdout, not the ARChitects system.
- **“ARChitects performs residual-region local repair.”** Its reported core loop
  soft-masks all positions and feeds logits back. Residual-targeted regional repair
  is our planned extension.
- **“Compositional Neuro-Symbolic Reasoning is a clean fixed-DSL system that beats
  pure neural and pure symbolic baselines.”** Its deployed pipeline uses several
  closed models and structured hints; exact symbolic intersection is partly
  idealized, and the pure-symbolic comparison is not established.
- **“SOAR 52%, TRM 8%, and CompressARC 4% are comparable ARC-AGI-2 hidden scores.”**
  False or unsupported. SOAR's headline is ARC-AGI-1 public; TRM has distinct public
  and semi-private ARC-AGI-2 numbers; CompressARC's author paper centers ARC-AGI-1.
- **“Program diffusion with execution feedback is new.”** ICLR 2025 Tree Diffusion
  directly covers grammar-preserving tree diffusion, execution feedback, value
  guidance, and search.
- **“Brain-area functional switching is itself a novel or biologically supported ARC
  mechanism.”** Dynamic module selection and temporally extended options predate this
  project; fMRI evidence supports network reconfiguration only at an analogy level.

## Unverified user-provided experiment

The attachments report a synthetic ARC-like comparison with DLGN, MLP, tree,
heuristic, and memory baselines. The referenced `sandbox:/mnt/data/...` files are not
available in this workspace. The experiment may be useful, but its values must remain
`unverified_user_report` until the following are recovered:

- task generator and train/OOD split definitions;
- all code and dependency versions;
- model configurations and learned checkpoints or seeds;
- raw per-task and per-seed outputs;
- aggregation script and generated tables;
- an independent rerun.

No paper or project status report may call those values verified before recovery.
