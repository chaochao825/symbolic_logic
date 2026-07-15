# Verified Literature Map

Last audited: 2026-07-11. Sources are grouped by the claim they can support. An
official report verifies its own protocol and result; it does not make that result
comparable to a different dataset, attempt policy, compute limit, or API setting.

## 1. Benchmark and current result landscape

| Source | Verified use | Important boundary |
|---|---|---|
| [Official ARC-AGI-2 repository](https://github.com/arcprize/ARC-AGI-2) | Current public data contain 1,000 training and 120 evaluation tasks; grids use symbols 0-9 and sizes 1x1 to 30x30. | Historical papers can describe older task counts. Pin the repository commit used by an experiment. |
| [Official ARC guide](https://arcprize.org/guide/1) | Documents two attempts per test input. | Some prose describes task success more strictly than the current scorer implementation. |
| [Official benchmarking scorer](https://github.com/arcprize/arc-agi-benchmarking/blob/main/src/arc_agi_benchmarking/scoring/scoring.py) | A test pair is correct if any attempt exactly matches; task score is solved test pairs divided by number of test pairs; task scores are averaged. | The code does not itself cap list length, so our harness enforces the policy limit of two attempts. |
| [ARC Prize 2025 technical report](https://arxiv.org/abs/2601.10904) | Private contest scores: NVARC 24.03%, ARChitects 16.53%, MindsAI 12.64%; identifies per-task refinement as the defining 2025 theme. | Historical 2025 private contest protocol; not comparable to public or semi-private API results. |
| [NVARC repository](https://github.com/1ytic/NVARC) | Synthetic-data ensemble combining an improved 2024-style autoregressive TTT component and TRM components. | NVARC is not evidence that masked diffusion plus TRM achieved 24.03%. |
| [ReARC](https://github.com/michaelhodel/re-arc) | Provides task-specific procedural generators and verifier programs for the 400 ARC-AGI-1 training parents, plus 1,000 verified examples per parent. | Generated descendants inherit their parent task's exposure and split; fixed-smoke or holdout parents must be excluded before sampling. It is not an OOD family by itself. |
| [ARChitects 2025 report](https://lambdalabsml.github.io/ARC2025_Solution_by_the_ARChitects/) | LLaDA-8B, 2D-inspired positional encoding, random-mask training, per-task LoRA, recursive soft masking, visit-count selection, and a separate shape model. | 21.67% was a live/public leaderboard result; 16.53% was final private. The 2D encoding lacks an isolated causal ablation. |
| [Poetiq verified report](https://poetiq.ai/posts/arcagi_verified/) | Multi-round hypothesis, validation, and correction reached 54% on official semi-private verification at about USD 30.57 per task. | External commercial-model protocol, not offline contest private. |
| [Modality-Driven Search with Holistic Trace Judging](https://arxiv.org/abs/2606.31543) | Reports 72.9% on official semi-private verification using text/image/code candidates and holistic judges; reports 76.11% on author-run public evaluation. | Single run, multiple closed models, high cost; public tasks influenced development. |
| [ARC official GPT-5.6 results](https://arcprize.org/results/openai-gpt-5-6) | As of 2026-07-11, the official page reports 92.5% ARC-AGI-2 for GPT-5.6 Sol at max reasoning. | Closed direct-model verification. It supersedes claims that the 72.9% system is the current overall leaderboard maximum, but is not an open/offline baseline. |
| [Tiny Recursive Model](https://arxiv.org/abs/2510.04871) | About 7M parameters; recursive answer/latent refinement; public ARC-AGI-2 around 7.8% and official semi-private around 6.3%. | Do not call this 8% private. |
| [SOAR](https://arxiv.org/abs/2507.14172) | Self-improving evolutionary program synthesis reaches about 52% on ARC-AGI-1 public evaluation. | The 52% number is not an ARC-AGI-2 result. |
| [CompressARC](https://arxiv.org/abs/2512.06104) | A 76K-parameter, per-puzzle MDL method reports 20% on ARC-AGI-1 public evaluation without pretraining. | ARC-AGI-2 ~4% appears in organizer summaries, not as a complete author-paper experiment; label it secondary evidence. |

## 2. Masked discrete generation and repair

| Source | Mechanism relevant to this project | Limitation or required baseline |
|---|---|---|
| [D3PM](https://arxiv.org/abs/2107.03006) | Discrete-state diffusion, including absorbing-mask transitions. | Its corruption matrix is not a functional legality matrix. Keep these mathematical objects separate. |
| [Mask-Predict](https://arxiv.org/abs/1904.09324) | Parallel prediction followed by iterative masking of low-confidence tokens. | Direct baseline for confidence-guided ARC repair. |
| [MaskGIT](https://arxiv.org/abs/2202.04200) and [code](https://github.com/google-research/maskgit) | Iteratively commits high-confidence tokens and resamples uncertain tokens; supports inpainting. | Original objective targets perceptual image tokens rather than exact categorical grids. |
| [Masked Diffusion Language Models](https://arxiv.org/abs/2406.07524) | Principled absorbing-mask language-model objective. | Standard committed tokens are difficult to revise; remasking requires an extension. |
| [LLaDA](https://arxiv.org/abs/2502.09992) and [code](https://github.com/ML-GSAI/LLaDA) | Large masked-diffusion language model used by ARChitects. | Sampling can be slow, fixed-length handling and KV caching are difficult. |
| [ReMDM](https://arxiv.org/abs/2503.00307) | Principled remasking and inference-time scaling for discrete diffusion. | Strong baseline; a heuristic local mask is not novel by itself. |
| [ARChitects 2025 report](https://lambdalabsml.github.io/ARC2025_Solution_by_the_ARChitects/) | Full-grid soft mask, self-logit feedback, recursive refinement, two 51-step rounds with a cold restart, separate shape predictor. | It does not establish residual-region repair. The recursion was found late, was not trained end-to-end, and could be unstable. |
| [LongT5 ARC-AGI-2 report](https://arxiv.org/abs/2603.06590) | Decomposes candidate coverage, post-filter coverage, and final pass@2; uses color, dimension, ratio, and containment filters. | Its 35%-55% pruning and small coverage loss are on 177 internal held-out tasks, not an official hidden set, and not the ARChitects system. |
| [Diffusion on Syntax Trees for Program Synthesis](https://openreview.net/forum?id=wN3KaUXA5X) | ICLR 2025 Spotlight: type/grammar-preserving tree corruption and denoising, execution results, value model, and search for program debugging. | Direct prior art against a generic “program diffusion plus execution feedback” novelty claim; tested on narrow inverse graphics. |
| [Diffuser](https://arxiv.org/abs/2205.09991) | Denoises whole state-action trajectories and supports constraint inpainting. | Requires offline trajectories; a generated schedule must still replan after ARC execution evidence. |
| [Decision Diffuser](https://arxiv.org/abs/2211.15657) | Conditions trajectory diffusion on returns, skills, and constraints. | Supports a controller analogy, not proof that diffusion beats beam or a stepwise policy for ARC. |

Three clocks must remain distinct: denoising step, program execution step, and outer
repair round. Conflating them makes both training targets and attribution ambiguous.

## 3. ARC program synthesis and neuro-symbolic reasoning

| Source | Verified use | Important boundary |
|---|---|---|
| [GridCoder2 controlled OOD study](https://arxiv.org/html/2507.15877v2) | Predict one DSL instruction, execute it, encode accumulated states, and predict the next. On seven synthetic OOD compositions, reported medians are 80% GridCoder2, 42.86% GridCoder1, and 10% NN-only. | Not a clean execution-feedback ablation: GridCoder1 also has a different, higher-level DSL; NN-only removes search but retains state conditioning. Real-task DSL coverage is very limited. |
| [Compositional Neuro-Symbolic Reasoning](https://arxiv.org/html/2604.02434) | 8-connected objects, object features, 22 unit patterns, neural proposals, and cross-example consistency; reports 24.4% public pass@2 and 30.8% with ARC Lang Solver. | 2026 preprint; deployed pipeline also uses multiple closed models and structured hints. Exact symbolic intersection is partly idealized, and no strict pure-symbolic comparison is provided. |
| [ARGA](https://arxiv.org/abs/2210.09880) | Object graphs, constraints, and graph-space program search for ARC. | Establishes object graph + DSL + constraint search as prior art. |
| [Inductive Logic Programming for ARC](https://arxiv.org/abs/2405.06399) | Synthesizes object-centered logical programs from small I/O sets using a hand-built background DSL. | Evaluation selects DSL-coverable tasks, illustrating the coverage ceiling. |
| [Write, Execute, Assess](https://arxiv.org/abs/1906.04604) | Separates program proposal, exact REPL execution, value assessment, and sequential Monte Carlo search. | Execute-and-score loops are not novel to ARC. |
| [Dreaming with ARC](https://openreview.net/forum?id=-gjy2V1ko6t) | Neural-guided module selection and ordering for different ARC tasks. | “Functional switching” alone is established prior art. |
| [CodeIt](https://arxiv.org/abs/2402.04858) | Treats ARC as programming by example with program sampling, hindsight relabeling, and prioritized replay. | Reported 15% is ARC-AGI-1 evaluation under that work's setup. |
| [DreamCoder](https://arxiv.org/abs/2006.08381) | Wake-sleep learns reusable DSL abstractions and a search model. | Macro discovery and library growth are established ideas. |
| [LARC / Communicating Natural Programs](https://arxiv.org/abs/2106.07824) | Human natural-language rules communicate a broad open vocabulary of ARC abstractions. | Evidence that a fixed executable DSL will remain incomplete. |
| [BARC](https://arxiv.org/abs/2411.02272) | LLM-generated programs, execution filtering, and iterative code repair for ARC. | High scores primarily concern ARC-AGI-1; compare protocol and model exposure carefully. |
| [SOAR](https://arxiv.org/abs/2507.14172) | Evolutionary sampling/refinement plus hindsight fine-tuning from search traces. | Strong baseline for code/program refinement; not a fixed DSL method. |
| [CompressARC](https://arxiv.org/abs/2512.06104) | Per-task MDL optimization provides a useful complexity prior. | Better viewed here as verifier/ranker evidence than a standalone ARC-AGI-2 solution. |

## 4. Modular control, routing, and logic gates

| Source | Verified use | Important boundary |
|---|---|---|
| [Routing Networks](https://arxiv.org/abs/1711.01239) | A router recursively selects function blocks and receives their outputs. | Direct prior art for state-dependent functional composition. |
| [Neural Module Networks](https://arxiv.org/abs/1511.02799) | Dynamically composes reusable modules from problem structure. | Dynamic module assembly is not itself new. |
| [Recurrent Independent Mechanisms](https://arxiv.org/abs/1909.10893) | Specialist recurrent groups update selectively and communicate sparsely. | Engineering analogy, not an ARC or brain-area solver. |
| [Option-Critic](https://arxiv.org/abs/1609.05140) | Learns temporally extended options, termination, and the policy over options. | Useful when a specialist should run several internal steps before switching. |
| [Attention for Compositional Modularity](https://openreview.net/forum?id=3UrIn433-Ez) | Controlled evidence that routing quality can dominate modular OOD performance. | Workshop study on synthetic equations; not ARC evidence. |
| [Modular Deep Learning survey](https://openreview.net/forum?id=z9EkXfvxta) | Separates computation, routing, aggregation, and module-local updates across modular architectures. | Survey support for terminology, not a performance claim. |
| [Deep Differentiable Logic Gate Networks](https://arxiv.org/abs/2210.08277) | Continuous relaxation trains Boolean gates; discretized inference exceeds one million MNIST images/s on one CPU core. | Supports cheap Boolean decisions only, not ARC control accuracy. |
| [Convolutional DLGN](https://arxiv.org/abs/2411.04732) | Extends logic-gate networks to convolutional vision with large gate counts. | Vision classification, not object binding or program routing. |
| [Light DLGN](https://arxiv.org/abs/2510.03250) | Identifies vanishing gradients, discretization errors, and high training cost; proposes a smaller/faster reparameterization. | Requires soft-hard agreement and training-cost audits in this project. |

### Older systems and configuration precedents

| Source | Novelty boundary for this project |
|---|---|
| [Blackboard systems](https://ojs.aaai.org/aimagazine/index.php/aimagazine/article/view/550/0) | Shared state, heterogeneous knowledge sources, and a scheduler are classic AI architecture patterns. |
| [SATzilla algorithm portfolios](https://www.cs.ubc.ca/~kevinlb/pub.php?u=SATzilla-full.pdf) | Selecting among complementary solvers from instance features is established; a heterogeneous ARC portfolio is not novel by itself. |
| [Dynamic Algorithm Configuration](https://www.tnt.uni-hannover.de/papers/data/1432/20-ECAI-DAC.pdf) | Changing algorithm configuration during a run under state and budget is direct prior art for online functional switching. |
| [Counterexample-Guided Inductive Synthesis](https://digicoll.lib.berkeley.edu/record/134841?v=pdf) | Propose, verify, obtain a counterexample, and repair is an established synthesis loop. |
| [SyPet type-directed synthesis](https://doi.org/10.1145/3093333.3009851) | Type constraints and type-directed search already prune program spaces. |
| [Invalid action masking](https://arxiv.org/abs/2006.14171) | State-dependent legal-action masks have theoretical and empirical precedent. |

Consequently, the project cannot claim blackboards, portfolios, dynamic switching,
counterexample loops, type pruning, or action masks separately. Its candidate
contribution is the pre-registered ARC-specific coupling and evidence decomposition.

This literature favors a stepwise, state-conditioned option/controller with exact
execution feedback. Diffusion over complete functional traces is a proposal generator,
not a replacement for receding-horizon replanning.

## 5. Safe use of the brain-network analogy

- [Cole et al. 2013](https://www.colelab.org/pubs/2013_Cole_NatNeurosci.pdf)
  reports task-dependent changes in whole-brain connectivity involving flexible hubs.
- [Shine et al. 2016](https://arxiv.org/abs/1511.02976) reports movement between more
  segregated and more integrated network states in time-resolved fMRI.

These sources can motivate **context-dependent functional reconfiguration**. They do
not identify an artificial specialist with a brain region, establish a Boolean state
matrix, or support a biological diffusion mechanism. The project therefore uses
“functional switching” as an engineering abstraction and keeps neuroscience claims
out of the contribution map.

## 6. Literature-derived design decisions

1. Separate generation coverage, filter survival, ranking, and official pass@2.
2. Treat output shape as a first-class candidate dimension.
3. Use `parse beam x program beam`; never fix 8-connectivity as the only object view.
4. Reserve hard filters for necessary format/type constraints until true-candidate
   false-negative rates are measured.
5. Compare residual masks with Mask-Predict, ReMDM, random masks, global soft masks,
   and cold restart at equal model calls.
6. Compare syntax-tree diffusion with typed beam, mutation, and resynthesis.
7. Compare the functional controller with fixed schedules, bandits, XGBoost, MLP,
   trees, and DLGN on the same features and budgets.
8. Keep open-vocabulary code proposals because fixed DSL coverage is a hard ceiling.
9. Preserve candidate diversity; applying identical refinement to every source may
   collapse complementary hypotheses.
10. Claim novelty only for a verified coupling: counterexample-to-mask compilation
    across grid, typed AST, and function trace plus budget-aware repair-space routing.
