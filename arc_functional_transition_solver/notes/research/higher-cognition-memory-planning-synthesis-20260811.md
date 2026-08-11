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
- Recent object-centric ARC models strengthen candidate distributions through
  latent slots and recurrent transitions.  The distinct question here is
  whether an observed failure can cause a legal, budgeted, cross-representation
  frontier change under intervention.

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
- ARC Prize 2025 official analysis:
  https://arcprize.org/blog/arc-prize-2025-results-analysis
- Xu et al. (2026), ARC-AGI-3: https://arxiv.org/abs/2603.24621
- ARC-TGI (2026): https://arxiv.org/abs/2603.05099
- *Slots, Transitions, Loops* (2026): https://arxiv.org/abs/2606.12316
- ArcMemo (2025): https://arxiv.org/abs/2509.04439
