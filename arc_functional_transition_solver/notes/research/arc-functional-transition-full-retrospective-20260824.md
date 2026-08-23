# ARC 异构候选、残差修复与功能转换研究：完整理论与实验回顾

日期：2026-08-24

项目：ARC Functional Transition Solver

性质：历史研究综合报告，不新增实验结果

## 摘要

本项目最初希望构造一种面向 ARC 的“类脑功能转换”系统：masked diffusion、LLM/code、DSL/program synthesis、sparse CA、DiffLogic 等专门模块产生候选；任务表示与执行残差驱动模块切换；demo-exact、MDL 和硬规则负责验证；局部 repair 在不重新搜索全部空间的前提下修复 near miss。经过多轮理论收缩、工程实现、受控实验、公开训练集开发审计、合成 ARC-TGI 强 provider 实验和一次严格 prospective confirmation，结论已经比最初设想清楚得多。

当前最准确的总体判断是：

1. 项目已经形成一套较可信的异构候选、内容寻址、重放、预算记账、数据边界和负结果归因基础设施。
2. 异构归纳偏置确实存在互补性；视觉 provider、递归 neural provider、DSL/CA 和小型程序族都曾进入其他模块没有覆盖的候选区域。
3. 当前实现尚未证明 residual-driven functional switching。早期 residual 清空或打乱不改变动作；后续多个 typed repair 在自然任务上没有达到预注册的独有恢复门槛。
4. 失败的主要原因不是 controller 不够复杂，而是候选语言、父候选可修复性、结构化诊断和动作语言之间没有形成足够强的闭环。
5. 强 provider 主线带来了目前最有希望的结果：在合成 ARC-TGI 上，NVARC/VARC 的候选并集很高，静态的 query-gold-blind recruitment 也能用较少 visual 计算找回边缘任务；但严格 prospective EXP-004 中只有 2 个 visual-exclusive 任务，且来自同一个 family，未通过“至少 3 个独立 family”的 complement gate。
6. 因此当前系统既不是已经有竞争力的 ARC-AGI solver，也不足以支持生物学脑区机制主张。它支持继续研究一种更窄、可证伪的命题：强异构 specialist 保留各自表示，共享 workspace 根据可观测不确定性或结构化失败，以原生成本约束选择性招募能够改变候选前沿的 specialist。

一句话概括整个研究史：

> 项目从“用更复杂的 router 调度许多弱模块”，经过一系列有效负结果，转向“先建立真正互补的强候选群体，再检验共享状态、失败证书和预算化招募是否能有效利用这种互补性”。

## 1. 报告范围与证据等级

本报告综合以下来源：项目控制面、项目总 README、结果状态表、各实验结果摘要、研究 gate 文档，以及最初的理论分析和审稿式复盘。为避免把探索性观察写成正式论文结论，全文使用五种证据等级。

| 等级 | 含义 | 可支持的结论 |
|---|---|---|
| 正式 confirmatory | 预注册、边界合格、结果有效的 EXP | 仅支持该 protocol、cohort、provider 和阈值下的结论 |
| 验证过的开发实验 | 可重放但使用公开或 outcome-exposed 数据 | 可选下一候选，不能证明泛化 |
| 受控机制实验 | 人工构造、单元测试或合成 fault | 证明执行器/类型/干预语义，不证明自然任务效用 |
| 无效或工程失败 | 数据边界、依赖、运行或 protocol 不合格 | 不产生方法层面的正负更新 |
| 理论提案/历史分析 | 形式化、相关工作和尚未完整执行的路线 | 作为设计依据，不当作实验事实 |

最初的 LGN 受控组合实验来自附带的历史实验报告，不在当前仓库的正式 EXP registry 中；本报告保留其结果，但明确标为“历史受控证据”。ARC-TGI 是合成、generator-family-disjoint 的研究基准，不等同 ARC-AGI-1/2 成绩。

## 2. 原始动机与理论形式化

### 2.1 原始系统设想

最初的数据流是：

```text
masked diffusion / code / DSL 生成大量候选
  -> 任务与表示 router
     -> 对象、关系、shape change：DSL / program synthesis
     -> 开放式异常假设：LLM / code model
     -> 同形、局部、高 support：sparse CA / D4 / bgpad
     -> 可压缩局部 transition：DiffLogic hard circuit
  -> demo-exact + MDL + hard-rule verification
  -> residual-directed local repair
```

“类脑”直觉不是把软件模块等同于真实脑区，而是强调四件事：功能专门化、共享工作空间、执行后的误差监控、以及受资源约束的功能招募。

### 2.2 从任务分类转向执行感知控制

早期理论分析已经指出，仅根据 query 输入预测任务类别在一般情况下不可辨识。若两个任务具有相同 query 输入、不同 demonstrations 和不同正确输出，任何 query-only router 都不可能同时正确。更合理的任务局部程序是

\[
p_\tau=((z_1,a_1),\ldots,(z_T,a_T)),\qquad
s_{t+1}=F_{z_t}(s_t;a_t),
\]

而 controller 应依赖 demonstrations、当前可执行状态和残差：

\[
q(z_{t+1}\mid D_\tau,S_t,r_t,z_t).
\]

这带来三个一直保留至今的原则：

- controller 只选功能或 option，高基数对象、颜色和几何参数由约束求解或小规模枚举完成；
- 类型和前后条件生成动态合法转移，而不是依赖固定任务类别；
- top-k 路由、精确执行和 demonstration verifier 比一次性 top-1 分类更重要。

### 2.3 版本空间、验证与 MDL

demo-exact 只说明程序属于

\[
\mathcal V(D_\tau)=\{p:\operatorname{Exec}(p,x_i)=y_i,\forall i\},
\]

并不保证不同一致程序在 query 上给出相同结果。因此项目一直使用或讨论 MDL、颜色置换等变性、对象连续性、较少例外和程序复用作为欠定问题的先验。后续实验进一步证明，应区分：

- 程序 ID 新颖；
- query 输出内容新颖；
- demo-exact；
- 对 oracle 真正有独有贡献。

仅有新程序而没有新输出，不算改变候选前沿。

### 2.4 布尔/LGN 控制的原始定位

布尔转移矩阵可以把长度受限的模块序列从指数搜索压缩到合法序列，但若矩阵剪掉真程序，成功概率直接归零。DLGN/LGN 因此最适合处理离散控制谓词、合法性、停止和候选筛选，不适合单独承担可变尺寸、对象绑定、计数、递归和输出生成。

项目对 LGN 的最终定位一直是：可解释、离散、低成本和硬件友好的控制器候选，而不是默认最强的 ARC solver。事实上，后续所有真实任务实验都表明，在候选互补性与可执行 repair 尚未成立之前，训练更复杂 controller 没有意义。

## 3. 方法谱系

### 3.1 候选生成模块

项目尝试或实现过以下候选源。

| 候选源 | 主要表示与归纳偏置 | 实际状态 |
|---|---|---|
| typed grid DSL | scale、tile、panel、lattice、ray、bbox 等离散程序 | 在 20 题 smoke 上逐步扩展，最终 8/20 |
| sparse CA / D4 / bgpad | 同尺寸、局部、高支持 transition 与几何等变 | 形成少量互补候选，但覆盖有限 |
| scene/object DSL | connected components、角色、canvas、crop、关系和多阶段 scene pipeline | v3 首次带来 3/100 独有 selectable，但成本高且 repair 为零 |
| M04a Grid-CMLM | 8.73M 参数 masked grid completion + MaskGIT | 训练未完整结束；只证明条件补全信号，未形成正式 ARC candidate pool |
| VARC visual provider | query-blind visual hypotheses、posterior/disagreement | 在多个 cohort 上提供明显视觉独有候选 |
| NVARC/TRM provider | reference-scale recursive/neural 候选 population | ARC-TGI 强 anchor，top-10 oracle 很高 |
| LLM/code | 开放式程序假设与 Python 候选 | 早期主要是显式 callback/abstention，未形成稳定强 provider |
| DiffLogic/DLGN circuit | 可部署硬逻辑电路或 callback | 接口已建，但历史 artifact 缺完整预处理/config，不能视为可重放 checkpoint |
| USRL-style neural loop | understanding/solving recurrent internal loop | 仅完成严重欠规模 pilot，不是论文复现 |

### 3.2 控制与路由

控制路线依次经历：

1. 受控 ARC-like 环境中的 LGN/MLP/布尔树 next-operation routing；
2. demonstration-only 静态 representation router；
3. content-addressed online residual controller，包含 typed action、预算 reservation 和 STOP；
4. coverage-aware 静态/规则控制，在相同候选池上减少无效执行成本；
5. visual-posterior 和结构特征驱动的 task allocation；
6. 强 provider population 上的 query-gold-blind functional recruitment。

主线的重要变化是：router 的目标从“预测任务类别”改成“在原生成本下选择能增加边缘覆盖的 specialist”。

### 3.3 共享 workspace 与可执行状态

工程上先后实现了：

- immutable content-addressed blackboard；
- candidate provenance DAG、whole-task hypothesis 和冲突 quarantine；
- Object–Program Workspace；
- Cognitive Workspace + 两阶段 object-graph rewrite；
- persistent entity/relation identities；
- node-level failure certificate；
- affected-subtree replay；
- TypedProgramSketch、typed hole、abstract execution 和 ProofObligation。

这些机制提高了可审计性，并在受控任务上证明类型、插入、重放和干预语义正确；自然任务实验则表明，持久状态本身并不会自动产生正确诊断或足够强的动作语言。

### 3.4 Repair 与前沿改变动作

Repair 路线按表达能力逐步扩展：

1. global color mapping 与 radius-one local transition；
2. parent-conditioned DSL suffix/shape resynthesis；
3. sparse CA policy repair；
4. visual pixel consensus 和局部 remask 思想；
5. object rematch、canvas reinfer、AST-hole 和 relation mask；
6. 24 个 D4/color relational transducer；
7. selector-only recolor/erase；
8. 两阶段 scene rewrite；
9. persistent-state single-node rewrite；
10. `Grid -> RecolorGridNode -> Output` topology insertion；
11. `Grid x Grid x Scene -> Grid` anchor-rasterized delta；
12. provenance-aligned relational effect。

研究过程中逐渐形成一条硬定义：

> 动作只有在产生 content-new、demo-exact 的候选输出，或改变经过验证的最终选择时，才算真正的 functional switch；仅更换 provider 调用顺序或产生新的程序字符串不算。

### 3.5 长期记忆与规划理论

在“假设感知和各种 specialist 已经很强”的思想实验中，剩余的高等能力被总结为 controlled recurrent state construction，而不是一个更大的 router。其必要组成包括：

- 稀疏共享 workspace；
- gated working memory 和稳定 goal stack；
- episodic 与 semantic/procedural 的互补长期记忆；
- 能做合法干预的 relational world/task model；
- hierarchical options 和 value-of-computation metacontrol；
- verification、credit assignment 与 replay。

映射到 ARC 的最小闭环是：

```text
executable trace
  -> persistent object/relation identities
  -> node-level typed failure
  -> counterfactual state rewrite
  -> affected-subtree re-execution
  -> exact verification
  -> exposure-scoped episodic memory / validated procedural memory
```

后续 v2/v3 实验实现了这一闭环的一部分，却在自然任务上失败。这说明持久记录和局部 counterfactual actuator 是必要基础，但还缺少正确的因果状态变量、多效应动作和长期 credit assignment。

## 4. 实验结果总表

### 4.1 起始受控实验：LGN 是否适合模块控制

历史实验使用 8 个参数化原语、3 个 demonstrations、训练只见一至两步程序、OOD 使用未见过的三至四步组合。它不是 ARC-AGI 分数，但验证了最初机制的一部分。

| 方法 | ID exact | 未见组合 OOD exact | OOD 中位展开节点 |
|---|---:|---:|---:|
| Hard DLGN + transition mask | 97.3% ± 1.5% | 80.0% ± 12.8% | 44.3 |
| Context MLP + mask | 97.7% ± 0.6% | 90.7% ± 5.1% | 48.2 |
| Context MLP，无 mask | 97.7% ± 0.6% | 92.0% ± 5.6% | 120.0 |
| MLP，无 execution predicates | 98.0% ± 1.0% | 84.3% ± 1.2% | 49.0 |
| Boolean tree + mask | 93.3% ± 0.6% | 63.3% ± 11.9% | 51.3 |
| Execution heuristic | 91.7% ± 3.5% | 55.7% ± 3.5% | 33.2 |
| Input-only MLP + mask | 57.3% ± 3.5% | 57.0% ± 19.1% | 43.7 |
| KNN whole-program memory | 83.7% ± 1.5% | 0.0% | 3.0 |
| Greedy verifier | 66.0% | 1.7% ± 1.5% | 9.0 |
| absolute pixel-patch memory | 0.3% ± 0.6% | 0.3% ± 0.6% | 0 |

关键观察：

- DLGN 可以学习可组合控制，但没有优于 MLP；其优势是离散、可检查和低成本。
- mask 把 MLP 的中位搜索从 120 降到 48.2，主要贡献是压缩搜索，不是提高精度。
- execution predicates 带来约 6.4 个百分点的 OOD 增益，支持“执行—观察—再路由”。
- Hard DLGN 的 next-module top-1 约 70.3%，top-3 约 93.2%，说明 beam + verifier 比一次决定全部程序更合理。
- 移除必需 primitive 后，对应任务为 0，预示真实 ARC 的首要上限是 representation/library coverage。
- 一个 demonstration 污染一个像素时，严格 verifier 降到 0；允许一个带 MDL 代价的异常后，DLGN/MLP 分别恢复到约 74.7%/84.3%。任务内记忆应保存参数和压缩例外，而不是绝对坐标补丁。

### 4.2 基础设施与早期符号候选

| 阶段 | 数据/设置 | 结果 | 结论 |
|---|---|---|---|
| Phase 0 harness | source/dataset/scorer 绑定、two-attempt scoring、atomic bundle | 工程 gate 通过 | D4 分数仅为 harness smoke |
| DSL v0.1 | 固定 20 题 public-training smoke | 2/20 | 基础 DSL 可重放 |
| DSL v0.2 scale/tile | 同一 20 题 | 4/20 | 加法覆盖增长 |
| DSL v0.3 panel-overlay | 同一 20 题 | 5/20 | 新 panel family 有贡献 |
| DSL v0.4 indexed panel-sequence D4 | 同一 20 题 | 6/20 | D4 + sequence 扩展 |
| DSL v0.5 periodic panel-lattice | 同一 20 题 | 7/20 | 周期结构扩展 |
| DSL v0.6 axis-ray bbox-contact | 同一 20 题 | 8/20 | 保留全部旧解；仍有 12/20 无正确候选 |

剩余任务的干净输入 signature 往往只对应一个已知题目，继续增加单题 primitive 会造成 outcome-driven DSL 膨胀，因此主线转向全局 neural 候选。

### 4.3 Masked grid、USRL 与未完成 neural 路线

M04a 实现了精确 8,733,706 参数 Grid-CMLM、MaskGIT sampler 和闭环训练基础设施。生产训练计划 20,000 step，在 step 13,150 因观测 stdout 的 `BrokenPipeError` 中止，保留到 step 12,000 的六个 checkpoint，但没有原子发布 selected checkpoint、candidate pool 或 ARC 结果。只读重建显示 validation masked-cell CE 从 step 2,000 的 1.2439 降至 step 12,000 的 0.8713；full-mask one-pass exact 仅 1/664。这证明模型学到条件补全，不证明 ARC solving。

USRL 路线重建了公开的 tensor contract、strict unseen 与 transductive protocol，但没有官方代码/checkpoint，论文训练细节也不完整。pilot 只有约 177K 参数、3,000 updates，而 paper-shape 约 7.1M 参数、训练量估计高两个数量级；strict 0/87、transductive 0/419 只能称 under-scaled smoke，不能作为对论文报告 pass@2 的复现或反证。

DiffLogic 和 code/LLM 在早期系统中主要停留在 callback boundary。DiffLogic 历史 artifact 缺完整 preprocessing/config bundle；LLM/code 没有成为有稳定独有覆盖的 active provider。因此早期“完整五脑区系统”的实验事实上只测试了几个窄符号 provider，而非最初架构的全部模块。

### 4.4 在线 residual controller：工程成功，机制失败

v1/v2 建立了 content-addressed blackboard、typed action、原生预算 reservation、failure/residual compiler、parent-conditioned DSL/CA 以及 frozen heterogeneous pools。工程审计表现良好：20/20 原生预算可比、零 reservation violation、提交前后 replay 一致、专项测试 32/32 通过。

但关键方法假设未成立：

- `structured_deliberation`、`summary_grounded` 和去掉 phase/residual 的静态版本在 20/20 任务上产生完全相同的动作轨迹、成本和结果，均为 4/20。
- 100-task option audit 中，dynamic、residual cleared、residual shuffled 均为 13/100；第二动作改变 0/100；条件互信息约 0.0107 bit，置换检验 `p=1.0`。
- 误差近似分解为 83 个 candidate-language gap、1 个 selectable gap、0 control gap、0 final-selection gap；raw oracle 17/100，selectable 16/100。
- 历史 76 次 repair 和新增 14 次 repair 均无独有恢复。

这个结果不意味着“覆盖低必然使 residual 无信息”。更精确的结论是：当前池以 out-of-language failure 为主，residual compiler 又不能把失败变成可执行的表示变化，所以 residual 对当前动作集没有区分度。

后续 disjoint 100-task v2/v3 replay 中，exact-first/action-aware 路径达到全部 14 个 selectable tasks；coverage-aware 策略在结果仍为 14 时把成本从 945 降到 630 NCU。控制器可以少做无效工作，却不能创建缺失假设。

### 4.5 Object/code provider 与完整 failure matrix

| 版本 | 主要扩展 | 结果 | 判断 |
|---|---|---|---|
| object/code v0.1 | 初步对象/代码候选 | 两个 disjoint 100-task 审计仅 1/200 unique selectable；0/65 natural repair | provider 与 repair 均未过门 |
| v0.2 | role reachability | 新增 2,784 trials、40 个 role near miss；0 exact/unique，0/76 repair | 表示与父候选不足 |
| v0.3 scene AST | roles、canvas、crop、scene pipeline | frozen offset-100 上 3/100 unique selectable，union 12/100；repair 0/100 | provider-only gate 窄幅通过，约为 v0.2 的 333× trials |

随后对 48 个终止失败做完整 query-blind matrix：18 parse failure、14 missing relation、12 canvas incompatible、3 legal-but-inexact、1 insufficient AST。它是冻结语言的分解，不是 ARC 的一般因果 taxonomy；但它明确显示，局部 recolor 无法修复解析、canvas、全局 correspondence 和 AST 缺失。

### 4.6 Visual posterior：候选互补成立，像素共识 repair 失败

query-blind VARC 在两个 disjoint、exposure-audited ARC-AGI-2 training cohort 上增加 8/31 unique selectable 和 13/31 unique raw。对 27 个错误且 query 同尺寸的案例，median pixel-error AUROC 为 0.942，top-10% disagreement mask 的 recall lift 为 5.87×。视觉 posterior 确实知道“哪里可能错”。

但直接逐像素共识失败：26/26 query 触发，11 个产生真实新 frontier，exact candidate 为 0，unique recovery 为 0。内容哈希、replay 和 novelty 都通过，因此这是算法语义失败：像素边缘分布破坏对象、拓扑和程序节点之间的联合相关性。

在 12 个 frozen ARC-GEN families 的 visual-structure bridge 中，VARC raw 8/12、selectable 7/12，legacy portfolio 0/12；demo-derived 结构 second-view selector 最终仍为 7/12，recovery/regression 均为 0。whole-grid visual hypotheses 有互补性，但粗粒度 4-connected transition marginals 无法挑出有用 minority。

更严格的 relational-mask 路线先扫描 139 个 ARC-GEN family，最终 0/139 满足已有 scene-AST 单槽 neighbor 条件；随后有限 mask DSL 在冻结 ARC-AGI-2 public-training 50 题上 exact/unique 均为 0，typed repair 与 equal-cost restart 均为 0。三个“像素一致 near miss”实际是 background-dominated false near miss。

### 4.7 强 provider anchor 与 relational transducer

reference-scale NVARC/TRM 在 frozen ARC-TGI confirmation-100 上达到 pass@1 87/100、top-10 oracle 92/100。这是项目第一次拥有真正强的 candidate anchor，但它是合成 ARC-TGI，不是 ARC-AGI。

在 collision-free、outcome-exposed ARC-TGI development-49 上：

- VARC/NVARC raw union 为 46/49；
- frozen 24-program D4/color transducer recover 剩余 3 题；
- visual composite allocation 选 15/49，使用 360 program trials，recover 3；equal-charged cold 同样选 15/49，recover 0；
- 但清空 visual posterior 项会替换 7/15 动作，最终仍 recover 3；清空全部 certificate features 则 recover 1。

因此这个实验支持小型候选语言与 composite allocation 的开发集效用，却不能把性能增益因果归因给 visual posterior。

### 4.8 Object–Program/Cognitive Workspace 系列

| 实验 | 受控结果 | 自然开发结果 | 因果判断 |
|---|---|---|---|
| Object–Program Workspace v1 | 12 个 selector-fault 上 visual/cleared/shuffled 均 12/12，cold 5/12 | exposed 100 上 3 个新 program ID、0 新 output、0 unique；94 题无 demo-exact child | actuator 可用，sensor attribution 不成立 |
| Cognitive Workspace + Rewrite v2 | typed two-stage bridge 可执行 | dev50 上 1 个 novel output，且 exact/unique；门槛为 5/50 | 存在单个自然成功，但表示覆盖不足 |
| Stateful Object-Graph Rewrite v3 | persistent IDs、typed node rewrite、suffix replay 通过 | dev50 0/50 novel；compiler 节点改善 1/196，all-node oracle 改善 94/196，但 7,446 个 legal single-node trial 无一 demo-exact | 同时是 diagnosis 和 single-node action-language null |

v2 中 49/50 有执行合法父候选、48/50 无 demo-exact second stage；最佳父候选 demonstration residual 的中位数约 103 pixels，只有 4 题在 8 pixels 内。大量失败不是 near miss，而是需要重新解析或协调多处状态变化。

### 4.9 TypedProgramSketch、abstract execution 与 topology insertion

Executable Abductive Workspace v0.1 实现 typed sketch、proof obligation、fill action、deterministic content identity 和 legacy replay。23/23 targeted tests 与 49/49 direct-dependency tests 通过。完整 repository suite 包含不属于该 gate 的训练路径，因此早期无 `PYTHONPATH`、进入 training fixture 或 SSH 超时的运行均不能解释为方法负结果。v0.1 只是受控基础设施 boundary。

Recolor topology v0.2 冻结唯一 topology hole：

```text
Render: Grid -> RecolorGridNode: Grid -> Grid -> Output
```

受控 gate 中，3 个正例全部闭环，5 个负例全部拒绝，81/81 exhaustive abstract/concrete comparison 一致。自然 dev50 审计却只有 1 个 task/4 个 parent 可达；唯一 exact output 已在 incumbent 中，novel frontier 和 unique recovery 均为 0/50。192 个不可达 parent 中，188 不能由单一 global recolor 修复，4 个 shape 不兼容。

Anchor-rasterized delta v0.3 再加入有限 `Grid x Grid x Scene -> Grid` node，支持 12 个 anchor effect。75 个 dependency tests 与双重 replay 通过，但 strict LODO 仅 1/50，低于 3/50 sensor gate；该一题也可由 global/component recolor 解释，novel/unique 仍为 0/50。结论是独立 anchor-color raster marginals 太弱。

### 4.10 Provenance-aligned Relational Effect v0.4

v0.4 加入 PersistentEntityId、PersistentRelationId、crop/D4 exact cell provenance、EffectSummary 和 RelationalFailureCore，同时冻结原有 12-effect language，不在看到结果后扩 DSL。

G0 controlled/replay gate 通过；fresh sealed reserve-v2 的 G1 strict LODO 只有 2/100，且来自同一 family，与 global 和 whole-component recolor baseline 完全打平。因为 G1 失败，query gold 保持未读，G2 frontier、G3 utility 和 G4 residual causality 均未运行。

failure matrix 为：54 delta semantics inconsistent、13 cross-demo delta inconsistent、24 effect-language unreachable、5 no scene parent、2 shape incompatible、2 strict-but-baseline-redundant。source-cell provenance 覆盖 97.76% residual cells，而 entity-backed coverage 为 63.32%。这说明“找不到原始 cell”已不是主瓶颈；下一种表示若继续，应能表达多效应 obligation 和跨 demonstration relational role binding。

### 4.11 Heterogeneous population 与 functional recruitment：当前主线

正式 EXP registry 的五个实验把问题从 repair 转到强候选群体的互补性和按需招募。

| EXP | 数据与性质 | 候选结果 | Recruitment/结论 | 状态 |
|---|---|---|---|---|
| EXP-000 | exposed ARC-TGI 100，post-hoc audit | NVARC 92、VARC 89、union 95；5 recursive-only、2 visual-only、1 query-composed | visual-exclusive 未到 3；只支持继续诊断 | boundary |
| EXP-001 | 同 cohort，retrospective | 3 个 marginal-value task | 前 30% visual 成本 recover 2/3，random median 1；前 10% 也为 2/3 | valid-positive exploratory |
| EXP-002 | disjoint historical development-50 | 两边各 45、union 47，各自 2 exclusive | 前 30% recover 2/2，random upper median 1；含 abstention 与 genuine disagreement 各一 | valid-positive exploratory |
| EXP-003 | prospective reserve-100，pre-GPU | 未运行 provider | 发现 2 个 demo/query input collision 和 seal provenance mismatch | invalid，无方法更新 |
| EXP-004 | collision-free prospective-98 | NVARC raw 93；VARC raw 88；union 96；7 recursive-only、2 visual-only、3 neither、1 query-composed | 预注册要求至少 3 个 visual-exclusive，实际 2 且三个 marginal task 均来自一个 family；策略前 10/30 tasks 找回 3/3，仅作描述 | valid null |

EXP-004 中 NVARC pass@1/pass@2 为 88/89，VARC 为 75/80；总共 6,514 个 provider-independent unique hypotheses，只有 161 个跨 provider shared。视觉计算约 20,112 GPU seconds；前 10 个 recruitment task 使用约 1,910 seconds，前 30 个约 6,076 seconds。策略局部化信号是有意义的，但不能挽救 provider complement gate：有效独立 family 数只有 1。

EXP-004 的首次 closure 因错误 runtime 缺 `shortuuid` 而在 oracle 授权前停止；改用 cohort 已记录 runtime 后完成。该事件是工程失败，不改变数值和科学语义。EXP-003 的 collision 则使整个实验无效，不能与 EXP-004 合并。

## 5. 贯穿所有实验的核心因果链

早期弱池的失败可概括为：

\[
\text{候选语言覆盖低}
\rightarrow
\text{失败多为 out-of-language 而非 near miss}
\rightarrow
\text{粗 residual 无法定位合法表示变化}
\rightarrow
\text{router 退化为静态排序}
\rightarrow
\text{局部 repair 无法产生 exact 新前沿}.
\]

后续实验把这条链进一步细化为六个可分离层次：

1. **Provider complement**：正确输出是否存在于某个 provider population。
2. **Selectable complement**：正确候选是否能通过 demo-only admissibility 并进入可选池。
3. **Diagnosis**：失败证书是否定位到真正需要改变的表示、节点或角色。
4. **Actuation**：合法动作语言是否能表达所需的一个或多个状态改变。
5. **Frontier novelty**：动作是否产生 content-new output，而不只是新程序 ID。
6. **Selection**：在 pass@2 和原生成本约束下，是否选出边缘正确候选。

不同实验失败在不同层：

- 早期 DSL/CA 主要失败在 1；
- residual controller 同时受 1、3、4 限制；
- visual pixel consensus 的定位信号较好，但失败在联合结构和 4；
- Workspace v1 产生新 program、不产生新 output，失败在 5；
- Rewrite v3 中 oracle node 能改善 residual，但单节点无 demo-exact，3 和 4 都失败；
- provenance v0.4 已解决大量 cell lineage，仍失败在跨 demo role consistency 和 multi-effect language；
- EXP-004 的 provider union 很强，失败则集中在 family-diverse complement 门槛，尚不能进入 confirmatory recruitment/selection 主张。

## 6. 最重要的新 insight

### 6.1 Coverage 是上限，但不是 residual 信息性的充分解释

“覆盖低导致 residual 无信息”过于绝对。即使正确答案不在当前语言里，一个足够好的 certificate 仍可能识别“缺 canvas 变化”“需要 object rematch”或“缺一个 AST stage”。当前真正被证伪的是这个联合命题：out-of-language 失败占主导，并且现有 compiler 无法把它们变成当前动作集中的可执行表示转换。

### 6.2 像素 posterior 适合当 sensor，不适合直接当 joint repair

视觉 disagreement 的 AUROC 和 recall lift 表明 posterior 能定位风险区域；逐像素共识 0 recovery 则表明正确修复需要保留对象、拓扑、关系和程序节点之间的联合约束。最有潜力的桥不是 posterior -> color，而是：

```text
leave-one-demo-out visual posterior
  -> canvas/object/mask/AST typed certificate
  -> candidate-conditioned object graph or AST holes
  -> legal structural proposal
  -> exact replay
```

### 6.3 Persistent state 必要但不充分

PersistentEntityId、trace 和 suffix replay 解决“状态丢失”和“无法局部重放”，但 v3 的 1/196 compiled-node improvement 与 0 demo-exact 说明 workspace 必须保存可干预的因果变量，且 action 往往需要协调多个依赖变化。记录更多状态不会自动带来正确 credit assignment。

### 6.4 Program novelty 不等于 candidate-frontier novelty

Workspace v1 和 recolor topology 都产生过合法新程序，但输出与 incumbent 重复。`novel_frontier_count > 0` 是必要 guard；最终仍需 unique exact recovery 与 equal-cost restart 对比。

### 6.5 任务数必须再按 family 多样性折算

EXP-004 的 3 个 marginal task 全部来自一个 family。若只看 task count，容易把一个局部模式的重复成功误认为广泛功能互补。正式 gate 应同时要求 task 数和 distinct family 数。

### 6.6 Controller 的正标签来自边缘互补，而不是普通正确任务

当两个 provider 都正确或都错误时，controller 的选择不会改变 oracle coverage。真正有监督价值的是 provider-exclusive、abstention、query-composed 和成本敏感任务。EXP-001/002 说明静态 uncertainty/abstention policy 有信号；EXP-004 说明当前独立边缘 family 太少，尚不足以训练 GRU、MLP 或 diffusion controller。

### 6.7 “功能切换”应由可达语义定义

更换模块标签不构成切换。可检验定义应是：动作改变当前 workspace 的合法候选可达集合，或在相同候选池上改变经验证的选择结果，并在 residual injection、bridge lesion、cost intervention 或 module insertion 下产生预测中的选择性效应。

## 7. 与相关工作的关系

| 相关路线 | 主要机制 | 对本项目的启发 | 本项目仍需证明的差异 |
|---|---|---|---|
| USRL | 单一神经网络内部 understanding-solving recurrence 与 adaptive halt | candidate-conditioned re-understanding、可变计算深度、缓存规则表示 | 我们是系统级异构 population/verification；欠规模 pilot 不构成复现 |
| ARGA | object graph abstraction + DSL transformation | 对象级表示和图变换是必要基线 | 对象图本身不新；创新必须来自跨表示 certificate/action 和因果控制 |
| ARCANA | scene graph、latent DSL、blackboard、execution feedback、metacontrol | 说明“多模块反思 + workspace”已拥挤 | 需以 typed legal frontier change、成本匹配和 prospective intervention 区分 |
| NVARC | 大规模合成数据、强 code/neural model、TTFT/DFS/ensemble | 最重要的是强候选分布、逐题适配和独有覆盖 | 我们的控制层需在不重跑全部 provider 时利用互补性 |
| ARChitects / masked diffusion | 2D mask、递归 remask、候选访问频率与 shape model | 适合作为 neural provider，输出 shape、entropy、trajectory | 不应把像素 marginals 直接当 typed repair；需结构桥 |
| MindsAI | leave-one-demo-out TTFT、D4/颜色增强、投票 | LODO 与等变 voting 是强 neural anchor 技术 | direct grid provider 缺可执行 failure location |
| TRM | 小模型内部递归 refinement | 递归和强训练分布能形成有效 provider | 内部 latent recurrence 不等于模块功能切换 |
| SOAR | LLM/Python generation、execution feedback、evolution/hindsight | 最接近所需的强 code provider；可借 AST mutation 与执行轨迹 | free-form mutation 应被收缩为 typed、可计费、可重放 action |
| CompressARC / MDL | 单题优化与压缩先验 | 任务内归纳、ranking 与异常代价 | 不解决异构 provider recruitment 本身 |
| Hilbert-Geo、Step-CoT、Structured CoT、Visual Thoughts、Agile Deliberation | structured intermediate state、stepwise verification、concept deliberation | 推理过程需显式结构、局部反馈和分阶段决策 | 这些工作不直接证明 ARC 中跨表示 repair 的独有恢复 |
| Global Workspace、options、CLS、MuZero/Dreamer、meta-RL | 共享广播、分层 option、双记忆、rollout 和 value of control | 支持功能类比下的 state/memory/planning 设计 | 不能据此声称软件模块与生物脑区同源或等价 |

与前沿 ARC 系统相比，本项目目前的主要特点不是分数，而是：候选/动作/成本可重放，能区分 oracle、selectable、observed 与 selected coverage，能做 residual intervention 和 equal-cost restart 对照。其潜在优势只有在强 provider 已经提供 family-diverse complement 后、functional recruitment 或 typed repair 仍能带来净增益时才成立。

## 8. 为什么与高精度相关工作差距很大

### 8.1 强方法先解决候选分布，本项目长期先优化了控制层

高精度方法通常依靠大规模 synthetic tasks、reference-scale pretrained/code model、test-time fine-tuning、递归采样、D4/颜色增强、Python program search 或多模型 ensemble。项目早期真实 active pool 主要是窄 DSL、CA 和 scene DSL，masked model 未完成，LLM/code 与 DiffLogic 未成为有效 provider。控制器实际上是在选择多个错误求解器。

### 8.2 初始架构中的模块并未全部落地

“diffusion + code + DSL + CA + DiffLogic”的完整系统没有被同一次公平实验检验过。早期正结果主要来自人工扩 DSL，后期强结果主要来自外部 reference-scale NVARC/VARC。最初动机与实现之间存在明显 capability gap。

### 8.3 Natural failure 远离局部 repair manifold

父候选常有错误 parse、错误 canvas、缺全局 correspondence、缺阶段或多处 conditional mapping。v2 的 median demo residual 约 103 pixels；v3 单节点 oracle 虽能改善许多 parent，却无 demo-exact。局部颜色、单 mask 或一个 topology node 不是自然任务所需的完整 counterfactual。

### 8.4 视觉边缘分布缺联合结构

pixel entropy/disagreement 能识别不确定位置，却无法决定哪些 cell 属于同一对象、哪个关系是因果角色、多个变化如何联合发生。直接共识会把多个完整但互斥的 hypothesis 混成一个不合法 output。

### 8.5 硬验证提高可信度，也限制 exploratory neural lane

要求 demo-exact executable program 很适合 certified lane，却会把只能给出可重放 grid 的 neural candidate 变成死通路。理论上曾提出 certified/exploratory dual lane：一个 pass@2 slot 优先 proof-carrying program，另一个允许 source-frozen、确定性可重放但没有程序证明的 neural grid。该方案尚未作为正式主线完成检验。

### 8.6 Benchmark 与结论层级不一致

20-task public-training smoke、公开训练集开发审计、ARC-GEN controlled families、ARC-TGI synthetic family-disjoint cohort 和 ARC-AGI evaluation 是不同证据层级。ARC-TGI 的 96/98 union 不能外推成 ARC-AGI 分数；公开榜或论文报告也不能与本项目 13/100、16/100 直接比较。

### 8.7 工程质量先于算法能力成熟

内容寻址、replay、预算 ledger、candidate provenance、data boundary 和 negative-result protocol 已经很强，但这些设施不会自行增加正确候选。项目的工程可信度一度显著领先于算法能力，这也是“代码越来越好、分数却不涨”的根本原因。

## 9. 已证实、已证伪与仍未知

### 9.1 当前证据支持

- 异构表示具有局部互补覆盖。
- 执行反馈在受控组合任务中比 input-only routing 更有效。
- 类型、内容寻址、exact replay 和原生预算可以稳定实现。
- 静态 coverage-aware/recruitment policy 可以在部分 cohort 上节省计算并定位边缘任务。
- 视觉 posterior 的 disagreement 对 pixel error 有定位信息。
- topology insertion、persistent identity、suffix replay 和 typed intervention 在受控语义上可行。
- 强 provider population 是继续研究 solver 和 mechanism 两条线的必要 anchor。

### 9.2 当前版本已被证伪或关闭

- 早期 residual 实际参与动态动作选择。
- 当前 global/local color repair 能产生独有恢复。
- 逐像素 visual consensus 能保持结构并修复自然 near miss。
- selector-only recolor/erase workspace 能扩展 output frontier。
- 单一 global recolor topology 在自然 parent 上有新增效用。
- frozen anchor-rasterized 12-effect language 能通过 strict LODO。
- provenance-aligned 同一 12-effect language 能在 fresh reserve 上超过简单 baseline。
- EXP-004 的 NVARC/VARC complement 达到至少 3 个独立 visual-exclusive family。

### 9.3 尚未被证伪

- 更强、结构不同的 code/program provider 能带来 family-diverse 独有覆盖。
- 达到参考规模的另一类 masked-neural provider 能补足 NVARC 的失误 family。
- visual posterior 通过对象/AST joint proposal，而非 raster marginal，能产生独有恢复。
- 多节点、多效应、关系角色绑定的 typed rewrite 能超过 equal-cost restart。
- 在互补 provider 足够多后，learned metacontrol 能提高 pass@2 per compute。
- 带严格 provenance 的 episodic/procedural memory 能在 family-disjoint 或交互式任务上产生长期收益。

## 10. 当前最合理的统一算法

经过全部结果后，最简洁的统一形式不是“大 router + 大量手写 repair”，而是三层结构。

### 第一层：强异构 hypothesis population

每个 specialist 保留自身最合适的表示和训练方式：masked visual、recursive neural、code/program、object DSL、local CA。系统只要求它们输出统一的 evidence envelope：candidate content、demo trace、confidence/uncertainty、source provenance 和 native cost。

### 第二层：persistent executable workspace

Workspace 保存当前目标、候选 DAG、对象/关系身份、程序 trace、失败证书、预算和 exposure-scoped memory。它不是把所有表示压成一个弱公共 ontology，而是在必要处建立 typed bridge。

### 第三层：budgeted functional recruitment

Option 的价值应定义为单位原生成本下的预期边缘前沿增益：

\[
\operatorname{VOF}(o\mid W_t)
=
\frac{\mathbb E[\Delta\operatorname{Reach}_{\text{validated}}(W_t,o)]
+\lambda\,\mathbb E[\Delta\operatorname{Selection}(W_t,o)]}
{\operatorname{NativeCost}(o)}.
\]

功能切换只有在 option 产生新的 validated candidate region、解除明确 proof obligation，或改变最终 verified selection 时才成立。控制器可以从静态规则和 calibrated bandit 开始；只有边缘正例足够多、residual intervention 确实改变动作后，才依次考虑 XGBoost、MLP/GRU、最后 diffusion controller。

## 11. 两条研究轨道与下一决策

### Solver track

目标是提高真实 ARC 的 pass@2、oracle coverage、成本和 hidden-family 泛化。当前唯一主瓶颈是 family-diverse candidate complement。下一步应冻结 controller，只接入一个结构上真正不同、达到参考规模的强 code/program 或 masked-neural provider，并要求在新 prospective cohort 上至少产生 3 个 recruited-provider-exclusive sealed families。通过后才复用当前静态 recruitment policy，再单独研究 pass@2 selector。

### Mechanism track

目标是检验功能类比：residual injection、bridge lesion、成本干预、新模块插入和 procedural memory 是否产生选择性效应。该轨道不能再用“调用了不同 provider”作为证据，而必须证明 reachable semantics 或 validated selection 发生变化，并超过等成本 restart。

建议保留的硬门槛是：

| Gate | 进入下一阶段的条件 |
|---|---|
| Provider complement | 至少 3 个 recruited-provider-exclusive sealed families；同时报告 task 与 family concentration |
| Repair opportunity | natural near miss 优于 identity/global/component baseline，且父候选确实在动作语言可达范围内 |
| Frontier | `novel_frontier_count > 0`，按 output content 去重 |
| Repair utility | 至少 5/100 unique recovery，并严格超过 equal-native-cost cold restart |
| Residual causality | clear/shuffle/inject residual 后动作和成功任务按预测选择性变化 |
| Learned controller | 仅在上述门槛通过后依次测试 tree/XGBoost、bandit、GRU，最后 diffusion |
| Brain-inspired claim | specialization lesion、bridge lesion、cost intervention、module insertion 和 memory admission 均有预测中的选择性效应 |

如果新强 provider 仍没有 family-diverse complement，应更换候选分布而不是训练 router。如果 complement 成立但控制无增益，项目可成为强、可审计的 ensemble solver；如果 typed control 有独有增益但总覆盖一般，可形成机制论文；只有两者同时成立，才接近最初的完整目标。

## 12. 最终评价

从 solver 角度看，项目目前仍未展示 ARC-AGI 竞争力。最强的 92/100、95/100、96/98 等数字来自合成 ARC-TGI，不可与 ARC-AGI 排行榜直接比较；弱符号池上的 13/100、16/100 则准确暴露了候选语言上限。

从机制角度看，早期“residual 驱动脑区切换”已被明确否定；后续 workspace、typed topology 和 provenance bridge 依次证明了执行器的必要子机制，却没有达到自然 unique recovery 门槛。它们不是白做：这些负结果逐步排除了静态排序、像素共识、program-ID novelty、单节点 rewrite、单 delta mask 和单 effect language，把真正问题收缩到 family-diverse specialization、joint relational state、multi-effect action 和边缘 credit assignment。

从研究设计角度看，项目最大的既有优势是能够区分实现失败、有效 null、boundary 和 confirmatory failure，并且不会用更大搜索、公开数据反复调参或程序字符串 novelty 冒充算法进步。这正是下一阶段可以可信检验强方法的基础。

最终保留的核心命题是：

> 在内容寻址的强异构候选群体上，以持久可执行状态承载目标、关系、失败和预算；用 query-gold-blind 的不确定性或 failure certificate 招募能够改变合法候选前沿的专门功能；再通过精确执行、原生成本和因果干预检验这种招募是否优于静态全运行与等成本重启。

这仍然与“类似大脑功能分工与转换”的原始直觉一致，但主张已经从生物学类比收缩为可操作、可反驳的计算机制。

## 13. 仓库证据入口

- [项目总览](../../README.md)
- [结果状态图](../../results/README.md)
- [研究控制面](../../.research-control/PROJECT.md)
- [接受的主线决策](../../.research-control/docs/decisions/RDR-0001-heterogeneous-population-mainline.md)
- [EXP-000 结果](../../results/hypothesis_population_exp000_20260812/README.md)
- [EXP-001 结果](../../results/functional_recruitment_exp001_20260812/README.md)
- [EXP-002 结果](../../results/functional_recruitment_exp002_20260812/README.md)
- [EXP-003 无效运行](../../results/functional_recruitment_exp003_invalid_20260812/README.md)
- [EXP-004 prospective 结果](../../results/functional_recruitment_exp004_20260812/README.md)
- [Higher cognition 综合](higher-cognition-memory-planning-synthesis-20260811.md)
- [TypedProgramSketch v0.1 gate](executable-abductive-workspace-v0.1-gate-20260812.md)
- [Recolor topology v0.2 gate](executable-abductive-workspace-recolor-topology-v0.2-gate-20260812.md)
- [Recolor natural audit](executable-abductive-workspace-recolor-natural-audit-v0.2-20260812.md)
- [Anchor-rasterized delta v0.3 gate](anchor-rasterized-delta-v0.3-gate-20260812.md)
- [Provenance relational effect v0.4 gate](provenance-relational-effect-v0.4-gate-20260812.md)

本报告没有重新训练模型、重新打开 sealed query oracle 或改变任何既有实验设置和数值结果。
