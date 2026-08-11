# Higher cognition after strong perception: a computational synthesis

## Thought experiment

Assume visual, linguistic, motor, code, and symbolic specialists are already
strong and callable.  What remains to support long-term memory and planning is
not a single additional faculty.  It is a recurrent control organization that
makes specialist outputs persist, interact, acquire credit, and change future
computation.

The most defensible synthesis has six coupled parts:

1. **Sparse shared workspace.**  Specialized processors expose a small common
   state rather than exchanging opaque final answers.  Global-workspace models
   motivate selective access and broadcast; they do not imply a literal central
   homunculus.
2. **Gated working memory and goals.**  A goal stack must remain stable while
   subgoals are opened and closed.  Updating it is an action with opportunity
   cost, not an automatic overwrite by every new observation.
3. **Complementary long-term memory.**  Fast episodic traces preserve individual
   events and failures; slower semantic/procedural memory admits abstractions
   only after validation.  Replay is useful for consolidation and sometimes
   planning, but neuroscience does not support treating every replay event as a
   planned future.
4. **Relational world/task model.**  Planning requires a state on which legal
   interventions can be simulated.  For ARC this should be an object/program
   graph with uncertainty and typed holes, not only a pixel distribution.
5. **Hierarchical options and metacontrol.**  Skills are temporally extended
   actions with initiation conditions, termination conditions, cost, and an
   expected state change.  Metacontrol decides whether to retrieve, simulate,
   execute, re-represent, or stop according to expected value and information
   per cost.
6. **Verification and credit assignment.**  Execution failure must identify the
   state assumption or program node that failed.  Otherwise feedback cannot
   teach which memory, representation, or capability deserved credit.

The core is therefore **controlled recurrent state construction**: maintaining
a compact task model across time, routing verified information into it, using
it for counterfactual rollouts, and assigning delayed credit to the operations
that changed the model.  A large context window, vector database, planner, or
router alone implements only one part.

## What remains distinctly difficult after perception is solved

The thought experiment removes recognition, language parsing, and individual
skills as bottlenecks.  It does not remove four harder problems:

1. **State construction:** decide which specialist outputs refer to the same
   entity, event, variable, or causal role, and bind them into a compact state
   that survives across time.
2. **Counterfactual composition:** reuse known relations and skills in a novel
   configuration, simulate their consequences, and distinguish an imagined
   transition from an observed one.
3. **Long-horizon credit:** identify which representation choice, retrieval, or
   intermediate action caused a delayed success or failure.
4. **Resource-rational control:** decide when another retrieval, reparse, or
   rollout is worth its time and uncertainty, and when to stop.

These are more specific than "executive function" and more useful than naming
one anatomical region.  The proposed computational core is a closed loop:

```text
goal and uncertainty
  -> construct/retrieve a typed shared state
  -> choose a legal information-changing option
  -> simulate or execute it
  -> verify the predicted state change
  -> store an exposure-scoped episode and assign credit
  -> update the goal, confidence, and next computation
```

Language and symbolic thought are important because they compress, compose,
and externally transmit such states and procedures.  They amplify the loop and
support cultural accumulation; they are not substitutes for grounded state,
memory admission, simulation, or control.

There is no settled single-factor account of human cognitive uniqueness.  One
recent continuity account attributes apparently qualitative differences to a
quantitative expansion in global information capacity and sharing among
memory, attention, and learning.  Other accounts emphasize symbols, hierarchy,
social learning, or open-ended culture.  The architecture below therefore
treats increased shared capacity and compositional depth as testable scaling
variables rather than declaring one uniquely human primitive.

A contrasting mental-program account proposes several domain-specific internal
languages that discretize experience into symbols and recursively compose them
under an MDL-like pressure.  For ARC, this supports retaining multiple typed
representations and executable programs, but it does not show that the current
DSL is expressive enough or that program search alone explains human reasoning.

## Mapping to this ARC project

| Cognitive function | Auditable implementation |
|---|---|
| specialized cortex-like processing | diffusion, visual, code, DSL, CA, and circuit providers |
| working/global workspace | typed object graph, AST, goal stack, uncertainty, and active certificate |
| episodic memory | immutable content-addressed execution/failure episodes |
| semantic/procedural memory | demo-validated schemas and replayable programs with family provenance |
| cognitive map/world model | object-relation/program state with executable transitions |
| hierarchical action | typed provider option with initiation predicate and termination evidence |
| prefrontal-like metacontrol | budgeted retrieve/simulate/execute/re-represent/stop decision |
| error monitoring | demo residual and node-level failure certificate |

This mapping deliberately uses “brain-inspired” as a functional analogy.  The
software modules are not asserted to be homologous to anatomical brain regions.

## Implications from adjacent work

- Global-workspace and modular-network work supports shared latent/state access
  between specialists, but the useful unit here must remain typed and replayable.
- Complementary-learning-systems work argues against storing everything in one
  memory: episodic and consolidated schema stores need different update rules.
- Options and hierarchical-control work suggests that “switching” should be
  defined by legal initiation/termination and goal effects, not provider labels.
- MuZero, Dreamer, and meta-RL planning show that internal rollout can be learned
  without reconstructing every observation detail.  ARC still needs hard
  verification, so the rollout state should terminate in an executable program
  or candidate grid.
- External-memory agents and lifelong ARC systems show the value of persistent
  skills and retrieved abstractions.  Their main danger for evaluation is
  outcome contamination; every reusable record therefore needs task/family and
  exposure provenance.
- ArcMemo reports a relative improvement from concept-level natural-language
  memory over a strong no-memory baseline.  The complementary test here is
  whether executable, family-disjoint schemas retain that benefit after
  outcome-scoped records are quarantined and random-retrieval controls are
  matched.
- Recent object-centric ARC models strengthen candidate distributions through
  latent slots and recurrent transitions.  The distinct question here is
  whether an observed failure can cause a legal, budgeted, cross-representation
  frontier change under intervention.
- Recent hippocampal models make a sharper prediction than "memory helps":
  replay can construct a new relational state space from reusable cortical
  building blocks, while a metacontroller can learn when a rollout is worth its
  opportunity cost.  This motivates compositional graph rewrites and explicit
  rollout cost, not an unconstrained text reflection loop.
- The first ARC-AGI-3 milestone systems provide an engineering warning.  Strong
  multimodal base capabilities, compact running memory, short plan queues, and
  legal-action constraints were useful, while one winning team reported that
  hand-built scaffolding could reduce performance.  Added control structure
  therefore needs selective ablations and an equal-cost baseline.
- ARCANA is a close 2026 preprint-level comparison: it combines scene graphs,
  latent DSL proposals, execution feedback, a blackboard, and a learned
  metacontroller.  Its broad architecture cannot establish novelty for a generic
  "multi-agent reflection" claim.  The differentiating hypothesis here is the
  causal and audit contract: typed legal frontier changes, content-addressed
  replay, exposure-scoped memory, native-cost matching, and prospective oracle
  gating.  ARCANA's reported semi-private score is not used as an empirical
  anchor without independent reproduction.

## Minimal implementation and what it does not yet prove

The v2 implementation intentionally supplies only an auditable baseline:

- immutable, content-addressed goals, memories, failures, options, plans, and
  episodes;
- separate task-local, family-disjoint validated, outcome-exposed diagnostic,
  and sealed-oracle memory scopes;
- deterministic typed option planning under native multi-dimensional budgets;
- two-stage object/program composition in which a first-stage execution trace
  exposes a typed hole and a second stage must produce a demo-exact, novel
  candidate;
- residual injection, bridge lesion, cost intervention, replay, and
  equal-native-cost cold-restart controls.

This is not a learned human-like planner, a model of consciousness, or evidence
that the software modules correspond to brain anatomy.  It tests the prerequisite
claim that an explicit state bridge can create a candidate region that isolated
specialists and matched cold restart do not reach.  Learned metacontrol remains
frozen until that prerequisite passes on a prospective, family-disjoint gate.

## Research ladder

1. **Representation gate:** two-stage graph rewrite must create at least five
   development opportunities and at least two prospective reserve recoveries.
2. **Memory gate:** family-disjoint validated schemas must improve prospective
   results over no-memory and random-retrieval controls without outcome access.
3. **Planning gate:** bounded counterfactual rollouts must outperform equal-cost
   myopic execution, with benefit concentrated on tasks requiring composition.
4. **Metacontrol gate:** injected certificates and cost changes must cause the
   predicted selective option changes and a net prospective gain.
5. **Continual gate:** across ARC-AGI-3-style episodes, learned abstractions must
   transfer while task-specific outcome traces remain quarantined.

Only gates 1--4 together support a brain-inspired functional-switching claim.
None of them alone supports a biological brain-region claim.

## Benchmark implication

Static ARC-AGI-1/2 remains useful for candidate coverage, abstraction, and exact
verification, but it is a weak direct assay of long-horizon planning: each task
contains a few demonstrations and one or two final queries.  Cross-task memory
can also become benchmark contamination unless the family and outcome boundary
is explicit.

The two project tracks should therefore diverge after the representation gate:

- **Solver track:** ARC-AGI-2-style pass@2, selectable oracle, native cost, and
  hidden-family generalization with strong providers.
- **Mechanism track:** generated prospective families first, then interactive
  ARC-AGI-3-style environments for goal acquisition, belief update, exploration,
  memory compression, hierarchical plans, and value-of-computation stopping.

This is not abandoning ARC.  It assigns each ARC variant to the claim it can
actually test and prevents a static one-query repair from being mislabeled as
human-like long-term planning.

## Falsifiable predictions

1. Stronger providers without shared executable state improve oracle coverage
   but need not improve failure-directed control.
2. A workspace without memory provenance can appear better through leakage;
   family-disjoint admission should remove that gain.
3. A memory store without a world/task model improves retrieval but not
   counterfactual planning.
4. A world model without metacontrol wastes budget on rollouts whose expected
   frontier gain is zero.
5. A controller cannot learn meaningful switching until certificates map to
   distinct legal option sets.
6. If the proposed graph-rewrite bridge is real, certificate injection, bridge
   lesion, and cost intervention will have selective—not global—effects.

## Empirical update: persistent state is necessary but not sufficient

Stateful Object-Graph Rewrite v3 implemented the first bounded version of the
shared-state hypothesis: persistent object/relation identities, executable
node traces, typed failure certificates, counterfactual node rewrites, and
dependency-local replay. Controlled interventions verify those semantics.

The natural development gate is nevertheless null: 0/50 query-blind novel
outputs. More importantly, a demonstration-only intervention audit finds that
the compiled node improves 1/196 parents while some legal node improves 94/196;
even the all-node oracle produces zero demo-exact tasks. This narrows the
higher-cognition hypothesis in two ways:

1. a workspace needs causal state variables and calibrated credit assignment,
   not merely persistent records; and
2. counterfactual planning needs transitions that can coordinate several
   dependent state changes, not just enumerate one local field mutation.

The result does not justify returning to a larger router. A learned controller
would currently learn from incorrect intervention labels and an action set
with zero exact natural successes. The next mechanism evidence must come from
fresh typed-fault data and bounded multi-node transition plans that beat
equal-cost restart. Long-term memory remains downstream of that gate: storing
failed traces cannot compensate for a state/action ontology that cannot express
the successful counterfactual.

## Primary references

- Dehaene & Changeux (2011), *Experimental and Theoretical Approaches to
  Conscious Processing*: https://pubmed.ncbi.nlm.nih.gov/21521609/
- Mashour et al. (2020), *Conscious Processing and the Global Neuronal
  Workspace Hypothesis*: https://pubmed.ncbi.nlm.nih.gov/32135090/
- VanRullen & Kanai (2021), *Deep Learning and the Global Workspace Theory*:
  https://arxiv.org/abs/2012.10390
- O'Reilly & Frank (2006), PFC/basal-ganglia working-memory gating:
  https://pubmed.ncbi.nlm.nih.gov/16378516/
- McClelland, McNaughton & O'Reilly (1995), complementary learning systems:
  https://web.stanford.edu/~jlmcc/papers/McCMcNaughtonOReilly95.pdf
- Kumaran, Hassabis & McClelland (2016), updated complementary learning
  systems: https://pubmed.ncbi.nlm.nih.gov/27315762/
- Shenhav, Botvinick & Cohen (2013), expected value of control:
  https://pubmed.ncbi.nlm.nih.gov/23889930/
- Sutton, Precup & Singh (1999), the options framework:
  https://www.sciencedirect.com/science/article/pii/S0004370299000521
- Badre (2008), hierarchical cognitive control:
  https://pubmed.ncbi.nlm.nih.gov/18403252/
- Dehaene et al. (2022), symbols and recursive mental programs:
  https://pubmed.ncbi.nlm.nih.gov/35933289/
- Graves et al. (2014), Neural Turing Machines: https://arxiv.org/abs/1410.5401
- Schrittwieser et al. (2020), MuZero:
  https://www.nature.com/articles/s41586-020-03051-4
- Hafner et al. (2023), DreamerV3: https://arxiv.org/abs/2301.04104
- Goyal et al. (2019), Recurrent Independent Mechanisms:
  https://arxiv.org/abs/1909.10893
- Wang et al. (2025), Titans: Learning to Memorize at Test Time:
  https://arxiv.org/abs/2501.00663
- Park et al. (2023), Generative Agents: https://arxiv.org/abs/2304.03442
- Wang et al. (2023), Voyager: https://arxiv.org/abs/2305.16291
- Jensen, Hennequin & Mattar (2024), recurrent planning and adaptive replay:
  https://www.nature.com/articles/s41593-024-01675-7
- Bakermans et al. (2025), compositional state construction and replay:
  https://www.nature.com/articles/s41593-025-01908-3
- Cantlon & Piantadosi (2024), expanded cross-system information capacity as a
  continuity account of human cognitive uniqueness:
  https://www.nature.com/articles/s44159-024-00283-3
- ARC Prize 2025 official analysis:
  https://arcprize.org/blog/arc-prize-2025-results-analysis
- ARC Prize 2026 ARC-AGI-3 milestone 1:
  https://arcprize.org/blog/arc-prize-2026-milestone-1
- Xu et al. (2026), ARC-AGI-3: https://arxiv.org/abs/2603.24621
- ARC-TGI (2026): https://arxiv.org/abs/2603.05099
- *Slots, Transitions, Loops* (2026): https://arxiv.org/abs/2606.12316
- ArcMemo (2025): https://arxiv.org/abs/2509.04439
- ARCANA (2026 preprint): https://arxiv.org/abs/2607.09059
