# Literature map and source links

The report uses sources for field context; they are not used to manufacture any experimental number. “Current” below means checked through 2026-07-16. Preprints are marked as such.

## Knowledge compilation and tractable circuits

1. [Darwiche, *Decomposable Negation Normal Form* (2001)](https://doi.org/10.1145/502090.502091). DNNF gives a structural condition under which some Boolean queries are tractable.
2. [Darwiche and Marquis, *A Knowledge Compilation Map* (2002)](https://doi.org/10.1613/jair.989). The standard taxonomy of compiled propositional representations and query trade-offs.
3. [Chavira and Darwiche, *On Probabilistic Inference by Weighted Model Counting* (2008)](https://doi.org/10.1016/j.artint.2007.11.002). Connects WMC with compiled logical structure.
4. [Darwiche, *Tractable Boolean and Arithmetic Circuits* (2022)](https://arxiv.org/abs/2202.02942). Modern review of structural requirements and the difference between an arbitrary Boolean circuit and a tractable one.

## Differentiable and neuro-symbolic logic

5. [Badreddine et al., *Logic Tensor Networks* (2022 version)](https://arxiv.org/abs/2012.13635). Fuzzy real-valued logic for combining data and first-order constraints.
6. [Xu et al., *Semantic Loss* (2018)](https://proceedings.mlr.press/v80/xu18h.html). Compiles Boolean constraints to a circuit so loss and gradient are linear in compiled-circuit size; compilation itself remains a cost.
7. [Manhaeve et al., *DeepProbLog* (2021 AIJ version)](https://doi.org/10.1016/j.artint.2021.103504). Neural probabilistic facts paired with probabilistic logic-program inference.
8. [Huang et al., *Scallop* (NeurIPS 2021)](https://proceedings.neurips.cc/paper/2021/hash/d367eef13f90793bd8121e2f675f0dc2-Abstract.html). Differentiable Datalog-style reasoning through provenance semirings.
9. [Dong et al., *Neural Logic Machines* (2019)](https://arxiv.org/abs/1904.11694). Relational neural-symbolic reasoning; object grounding tensors still scale with predicate arity.

## Logic-gate networks and deployment

10. [Petersen et al., *Deep Differentiable Logic Gate Networks* (2022)](https://doi.org/10.52202/068431-0146). Uses continuous mixtures of two-input gates in training and discrete gates at deployment.
11. [Mielke et al., *Convolutional Differentiable Logic Gate Networks* (2024)](https://doi.org/10.52202/079017-3851). Extends the approach with convolutional logic trees and reports training/depth challenges alongside fast discrete inference.
12. [WARP Logic Neural Networks (2026 preprint)](https://arxiv.org/abs/2602.03527). Recent work explicitly targeting high fan-in, redundancy, and soft-to-hard parameterization difficulties.

## LLM/formal-system hybrid direction

13. [Pan et al., *Logic-LM* (2023)](https://doi.org/10.18653/v1/2023.findings-emnlp.248). LLM formalization followed by symbolic solving and error feedback.
14. [Trieu et al., *Solving Olympiad Geometry without Human Demonstrations* (2024)](https://doi.org/10.1038/s41586-023-06747-5). AlphaGeometry combines neural auxiliary-construction proposals with symbolic deduction.
15. [Gold-medalist Performance in Solving Olympiad Geometry with AlphaGeometry2 (2025 preprint)](https://arxiv.org/abs/2502.03544). A recent example of neural guidance coupled to formal/symbolic verification rather than a single homogeneous reasoner.

## Complexity and boundary references

16. [Cook, *The Complexity of Theorem-Proving Procedures* (1971)](https://doi.org/10.1145/800157.805047). SAT NP-completeness.
17. [Valiant, *The Complexity of Enumeration and Reliability Problems* (1979)](https://doi.org/10.1137/0208032). #P complexity, relevant to weighted model counting.
18. [Shannon, *The Synthesis of Two-Terminal Switching Circuits* (1949)](https://doi.org/10.1002/j.1538-7305.1949.tb03624.x). Counting arguments behind the fact that most Boolean functions do not admit small circuits.

## Coding rate reduction and white-box networks

19. [Yu et al., *Learning Diverse and Discriminative Representations via the Principle of Maximal Coding Rate Reduction* (2020)](https://arxiv.org/abs/2006.08558). Defines the finite-sample log-det MCR² objective used in the controlled reproduction.
20. [Chan et al., *ReduNet: A White-box Deep Network from the Principle of Maximizing Rate Reduction* (2021)](https://arxiv.org/abs/2105.10446). Unrolls rate-reduction optimization and derives linear/nonlinear and shift-invariant convolutional operators.
21. [Baek et al., *Efficient Maximal Coding Rate Reduction by Variational Forms* (CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/html/Baek_Efficient_Maximal_Coding_Rate_Reduction_by_Variational_Forms_CVPR_2022_paper.html). Shows that direct log-det MCR² has material computational cost and gives scalable variational forms.
22. [Yu et al., *White-Box Transformers via Sparse Rate Reduction: Compression Is All There Is?* (2023)](https://arxiv.org/abs/2311.13110). Derives CRATE-style subspace attention and an ISTA sparsification block from a sparse rate-reduction objective.
23. [Wang et al., *A Global Geometric Analysis of Maximal Coding Rate Reduction* (2024)](https://arxiv.org/abs/2406.01909). Characterizes the controlled objective landscape while also noting that earlier theory did not provide a complete global justification.
24. [Shannon, *Coding Theorems for a Discrete Source With a Fidelity Criterion* (1959)](https://ieeexplore.ieee.org/document/5311476). Primary source for lossy source coding under an explicit distortion/fidelity constraint.

## Discrete universal coding, MDL, and exact logic synthesis

25. [Krichevsky and Trofimov, *The Performance of Universal Encoding* (1981)](https://ieeexplore.ieee.org/document/1056331). Original add-one-half universal mixture used for operational Boolean sequence lengths.
26. [Rissanen, *Modeling by Shortest Data Description* (1978)](https://doi.org/10.1016/0005-1098(78)90005-5). Primary two-part minimum-description-length formulation.
27. [Blumer et al., *Occam's Razor* (1987)](https://doi.org/10.1016/0020-0190(87)90114-1). Classic connection between short consistent hypotheses and PAC learnability.
28. [McAllester, *Some PAC-Bayesian Theorems* (1998)](https://doi.org/10.1145/279943.279989). Establishes generalization control through prior/posterior KL; this becomes a description term when the prior is induced by a prefix code.
29. [Bryant, *Graph-Based Algorithms for Boolean Function Manipulation* (1986)](https://people.eecs.berkeley.edu/~russell/classes/cs289/f04/readings/Bryant%3A1986.pdf). Canonical reduced ordered BDDs under a fixed variable order.
30. [Bollig and Wegener, *Improving the Variable Ordering of OBDDs Is NP-Complete* (1996)](https://doi.org/10.1109/12.537122). Shows the computational difficulty of optimizing OBDD variable order; Bryant's restricted representation model remains essential when interpreting ROBDD size.
31. [Haaswijk et al., *SAT-Based Exact Synthesis* (2020)](https://si2.epfl.ch/demichel/publications/archive/2020/winston-exact.pdf). Size-optimum Boolean-chain synthesis under a fixed computation model and operator basis; topology families constrain the SAT search.
32. [Kojevnikov et al., *Finding Efficient Circuits Using SAT-Solvers* (2009)](https://doi.org/10.1007/978-3-642-02777-2_5). SAT-based circuit search and exact small-circuit reasoning.
## Basis-aware circuit representations and synthesis

33. [Amarù, Gaillardon, and De Micheli, *Majority-Inverter Graph: A Novel Data-Structure and Algorithms for Efficient Logic Optimization* (DAC 2014)](https://infoscience.epfl.ch/entities/publication/10dae280-8d2c-42d6-8b6d-6eb51c7a0eb1). Defines MIGs as DAGs of three-input majority nodes with regular or complemented edges; this is the basis convention used by the bounded oracle.
34. [Meuli, Soeken, and De Micheli, *Xor-And-Inverter Graphs for Quantum Compilation* (2022)](https://infoscience.epfl.ch/entities/publication/a4a09216-24cc-4535-b1a8-9e13632fd6df). Primary XAG reference; it motivates XOR/AND/inverter graphs and basis-specific cost rather than a language-independent gate count.
35. [Soeken et al., *The EPFL Logic Synthesis Libraries* (2018)](https://arxiv.org/abs/1805.05121). Open modular implementations spanning classical and emerging logic-network representations; useful context for basis-aware synthesis and conversion.
36. [Brayton and Mishchenko, *ABC: An Academic Industrial-Strength Verification Tool* (CAV 2010)](https://people.eecs.berkeley.edu/~alanmi/publications/2010/cav10_abc.pdf). Primary overview of AIG-based synthesis and SAT-backed verification in ABC; synthesis statistics and equivalence checks are separate obligations.

## Cellular automata, recurrent logic, and distributed tasks

37. [Miotti et al., *Differentiable Logic Cellular Automata: From Game of Life to Pattern Generation* (ALIFE 2025)](https://doi.org/10.1162/isal.a.882), with the [Google project page](https://google-research.github.io/self-organising-systems/difflogic-ca/). Trains soft mixtures over all 16 two-input Boolean functions on fixed wiring, then deploys hard recurrent circuits for Game of Life and pattern generation.
38. [Google Research, official `diffLogic_CA.ipynb`](https://github.com/google-research/self-organising-systems/blob/3d5547ca48b60ecac459834e2c05c9ff5df87991/notebooks/diffLogic_CA.ipynb). Apache-2.0 reference implementation tested by the authors with JAX/JAXLIB 0.4.33; the commit-pinned source and released DigitalJS artifact hashes are recorded in this repository.
39. [Mitchell, Crutchfield, and Hraber, *Evolving Cellular Automata to Perform Computations: Mechanisms and Impediments* (1994)](https://doi.org/10.1016/0167-2789(94)90293-3). Primary density-classification study of evolved binary radius-3 cellular automata.
40. [Das et al., *Evolving Globally Synchronized Cellular Automata* (1995), author PDF](https://melaniemitchell.me/PapersContent/EGSCA.pdf). Defines the no-partial-credit global synchronization task and publishes the `phi_sync` radius-3 rule used here.
41. [Land and Belew, *No Perfect Two-State Cellular Automata for Density Classification Exists* (1995)](https://doi.org/10.1103/PhysRevLett.74.5148). Rules out a perfect fixed-radius two-state solution to the original density task over all lattice sizes.
42. [Cook, *Universality in Elementary Cellular Automata* (2004)](https://www.complex-systems.com/abstracts/v15_i01_a01/). Proves Rule 110 universality via a cyclic-tag-system construction; reproducing its eight-entry LUT alone is not a universality reproduction.
43. [Gilpin, *Cellular Automata as Convolutional Neural Networks* (2019)](https://arxiv.org/abs/1809.02942), with [author code](https://github.com/williamgilpin/convoca). Shows exact and learned convolutional representations of cellular-automaton transitions.
44. [Mordvintsev et al., *Growing Neural Cellular Automata* (2020)](https://distill.pub/2020/growing-ca/). Continuous multi-channel NCA for growth, persistence, and regeneration from local updates.
45. [Randazzo et al., *Self-Classifying MNIST Digits* (2020)](https://distill.pub/2020/selforg/mnist/). Uses locally communicating cells to reach a spatial classification consensus and documents disconnected-component limitations.
46. [Earle et al., *Pathfinding Neural Cellular Automata* (2023 preprint)](https://arxiv.org/abs/2301.06820), with [author code](https://github.com/smearle/pathfinding-nca). Hand-codes and learns local recurrent BFS/DFS-style updates and evaluates grid-size generalization.
47. [Sandler et al., *Image Segmentation via Cellular Automata* (2020 preprint)](https://arxiv.org/abs/2008.04965). Applies a learned local CA update repeatedly for image segmentation with a small shared parameter count.

## ARC and cellular-automaton reasoning

48. [Chollet, *On the Measure of Intelligence* (2019)](https://arxiv.org/abs/1911.01547). Defines the skill-acquisition-efficiency motivation and the original Abstraction and Reasoning Corpus protocol.
49. [ARC Prize Foundation, official ARC-AGI-2 repository](https://github.com/arcprize/ARC-AGI-2/tree/f3283f727488ad98fe575ea6a5ac981e4a188e49). The commit-pinned Apache-2.0 source for the 1,000 public-training and 120 public-evaluation tasks used by this extension; upstream corrections are tracked in its changelog.
50. [Chollet et al., *ARC-AGI-2: A New Challenge for Frontier AI Reasoning Systems* (2026 revision)](https://arxiv.org/abs/2505.11831). Current technical report for ARC-AGI-2; this project keeps it distinct from the interactive ARC-AGI-3 environment.
51. [Xu and Miikkulainen, *Neural Cellular Automata for ARC-AGI* (2025)](https://arxiv.org/abs/2506.15746). Trains a task-specific continuous NCA on ARC-AGI-1: 23 of 172 filtered feasible public-training tasks are exact, with resize, unseen-color, global-coordination, and overfitting limits explicitly reported.
52. [Guichard et al., *ARC-NCA: Towards Developmental Solutions to the Abstraction and Reasoning Corpus* (2025)](https://arxiv.org/abs/2505.08778), with [author code](https://github.com/etimush/ARC_NCA). Its main task-specific NCA/EngramNCA comparison uses 262 mostly non-resize ARC-AGI-1 public-evaluation tasks and separates single-model from top-2 unions; the paper also reports a maximal-padding experiment over all problems, so filtered denominators are not its only protocol.
53. [Faldor and Cully, *CAX: Cellular Automata Accelerated in JAX* (2024)](https://arxiv.org/abs/2410.02651), with [author code](https://github.com/maxencefaldor/cax). Includes a 60.12% result on the simplified one-dimensional 1D-ARC benchmark; it is not evidence on 2D ARC-AGI-1 or ARC-AGI-2.
54. [Grattarola et al., *Learning Graph Cellular Automata* (NeurIPS 2021)](https://proceedings.neurips.cc/paper/2021/hash/af87f7cdcda223c41c3f3ef05a3aaeea-Abstract.html). Extends learned local transitions to given arbitrary graph topologies; the topology is supplied rather than inferred.
55. [Endo and Yasuoka, *Neural Cellular Maze Solver* (2021)](https://umu1729.github.io/pages-neural-cellular-maze-solver/). Demonstrates recurrent local path propagation and also documents hysteresis under changed inputs, motivating explicit state/reset controls.
