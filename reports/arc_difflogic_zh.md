# 真正的 DiffLogic-CA 能否修复 ARC-CA 的归纳缺口？

## 结论

本轮实现并训练了真正的固定 wiring、16 算子可微逻辑门 CA，而不是回放官方硬电路，也不是把已经选好的 categorical lookup table 再编译成门。结论是：

1. **逻辑门可训练、可硬化、可递归。** 局部 NOT 上三个种子的硬电路均全精确；所有正式 DiffLogic 候选的 PyTorch hard rollout 与独立 NumPy 门电路逐步一致。两步 shift 任务上，`DLR2` 的 demo-selected 硬电路达到全精确，而相同局部半径的一步 `DL1` 失败。
2. **这没有转化为真实 ARC 的 DiffLogic 归纳增益。** 在 8-task 开发 cohort 上，`DL1`、宽容量 `DL1-wide`、全局上下文 `DLC1` 和递归 `DLR` 的 demo-consistent hard exact 都是 `0/8`。一次性冻结的 12-task 确认 cohort 上，`DL1` 与 `DL1-wide` 都是 `0/12`。
3. **便宜而明确的结构先验更有效。** D4 对称增强在确认 cohort 上得到 `1/12`；D4+modal-background padding 在开发集得到 `1/8`、确认集得到 `1/12`。确认集中两者命中同一个 `b60334d2`，因此 union 仍是 `1/12`，不能相加成 `2/12`。
4. **画布接口确实能解除一类瓶颈，但门不是唯一原因。** demo-only constant-shape router 使 `DLF` 和 interface-matched MLP 都在 constant-canvas synthetic 上达到 3/3 种子全精确；两者在 scale×2 上又都失败或只记住 demonstrations。收益来自输出接口，尚未来自通用空间程序。
5. **“同效果下有优势”只在很窄的受控条件成立。** constant-canvas 上，DLF 的训练参数 payload 为 `190,464` bits、声明式硬电路为 `7,176` bits；MLP 权重 payload 为 `1,852,800` bits，分别相差约 `9.73×` 和 `258×`。但 DLF 的所选种子训练时间反而约为 MLP 的 `3.10×`，且 Boolean gate evaluation 与 dense MAC 不是同一种硬件代价单位。本轮没有得到真实 ARC 精度不降条件下的速度或 PPA 优势。

因此，已有证据仍支持原来的瓶颈顺序：

```text
任务语义 / 对象与关系表示
  -> 输出画布、边界与坐标拓扑
  -> 隐藏状态和多步传播
  -> 少样本下的规则与不变性选择
  -> 最后才是门编译与门级执行
```

这里的顺序是工程诊断，不是对所有 ARC 方法的数学全序。尤其是，本轮 engineered object raster 没有解决对象任务；它说明“需要更好的对象/程序表示”，而不是说明当前对象特征已经充分。

## 与冻结的 464/464 结果是什么关系

冻结 categorical ARC-CA 已证明：四个 ARC split 中 464 个被选中的局部规则实例，在 direct categorical、binary4 和 one-hot10 三路执行上 `464/464` 等价，非法解码率为零。这回答的是：

> 给定一个有限共享局部规则，能否不改语义地门化？

本轮回答另一个问题：

> 能否从少量 demonstrations 直接训练出正确的门规则、状态和迭代控制？

前者是编译问题，后者是归纳与表示问题。确认实验的 `0/12` 表明，编译兼容性不能推出规则可识别性。冻结的 ARC-AGI-2 evaluation `0/120` 没有被重跑，也不能用这里的 ARC-AGI-2 training 机制 cohort 外推成新 benchmark 分数。

## 模型与协议

### 真正的可微门 CA

每个 gate 对公开顺序的 16 个二输入 Boolean 函数维护 logits。温度为 `tau` 时：

```text
p_g = softmax(theta_g / tau)
y_g(a,b) = sum_k p_g[k] f_k(a,b)
```

训练前段使用 soft mixture，后段使用 straight-through argmax；部署只保存 gate ID、两条固定输入线和二值状态。ARC 颜色用 binary4，10--15 记为非法码并计数。固定 wiring 使用覆盖平衡采样，并为每个动态 state bit 保留贯穿所有层的 center-state backbone；这个修复避免了“每层覆盖输入、但输出端没有完整因果路径”的假训练。

straight-through 阶段的 0/1 forward value 采用 affine epsilon map 进入 BCE 和非法码惩罚，避免 `clamp` 在边界把梯度截断。正式 hard export 会在全部 demonstrations 和 test inputs、全部所选步上与独立 NumPy 门执行逐状态比较；不一致即中止。

### 分层变体

- `DL1`：binary4、一步、无 hidden state；
- `DL1-wide`：同接口的容量控制；
- `DLR/DLR2`：8 hidden bits、共享递归更新，horizon 只从 `1,2,4,8` 的可用前缀中由 demonstrations 选择；
- `DLC1`：加入 demonstration/task 与 test-input 全局 context；
- `DLO`：再加入原始颜色、坐标、边界、mask 和 4-connected component raster；
- `DLF`：加入 demo-only shape program 与 padded workspace；
- `MLP/MLPO1`：同 state、context、horizon、decoder 和训练预算的连续 NCA 对照。它是**接口匹配**，不是参数量匹配。

shape program 只允许 same、constant output shape、整数 scale 和 foreground bounding-box。modal-background padding 在 demonstrations 与 test input 上完全同构地加一圈，输出按公开的一格 margin 裁剪；若裁剪后不能全精确复现原 demonstrations，候选必须 abstain。

### 标签隔离与确认冻结

solver-facing `ArcProblem` 只含 demonstrations 和 test inputs；test outputs 位于独立 `ArcLabels`。runner 先为**所有任务、种子、horizon 和硬电路**生成预测并完成 demo selection，之后 evaluator 才收到任何 labels。

开发过程允许查看 `DEV-LOW-MDL` 和 synthetic。确认配置在读取 `CONF-INDUCTION-HASH` 结果前冻结，绑定：

- source commit `4f54aa45d1e6328d96352028375718b7506d926a`；
- 12 个 task IDs、ID digest 和内容 digest；
- `DL1,DL1-wide`，seeds `0,1,2`，480 epochs；
- 全部温度、loss、optimizer、augmentation 和 device 参数；
- 主指标 `strict_hard_task_exact = hard_demo_task_exact * hard_task_exact`。

配置 SHA-256 为 `0817bad7b056fc3ca39a9802c54637499373a8b31c5f9ce598b83eee0bd90e26`。未完整解释 demonstrations 却偶然命中 test 的候选只保留为 raw diagnostic，不能支持“学到规则”。

## Synthetic 机制结果

| 任务 | 关键对照 | 正式结果 | 能支持什么 |
|---|---|---|---|
| local NOT | `DL1` | 3/3 seeds hard demo/test exact；soft→hard drop 0 | 门 logits 确实被优化并可独立硬化 |
| hidden shift×2 | `DL1` vs `DLR2` | `DL1` 0；`DLR2` 1/3 seed 成功，demo selector 正确选中 horizon 2；sparse radius-2 也能解 | recurrence 可扩大相同半径模型的因果锥，但不优于更宽显式邻域 |
| hidden dilation×2 | `DL1` vs `DLR2` | 两者均失败 | 有两步状态不等于会学到正确迭代算法 |
| global marker | `DL1` vs `DLC1` | 两者 strict 0；`DL1` 有一次未拟合 demos 的 raw test 偶然命中 | 全局 context 通道本身没有解决任务条件学习 |
| object anchor | `DL1/DLO1/MLPO1` | 全部 test 失败；MLP 3/3 拟合 demos 后仍 OOD 失败 | engineered object bits 不等于对象级程序归纳 |
| constant canvas | `DLF/MLP` | 两者 3/3 demo/test exact | demo-only 输出接口有效，门类型不是唯一解释 |
| scale×2 | `DLF/MLP` | DLF 不拟合；MLP 3/3 拟合 demos 但 0/3 test | shape 尺寸正确仍缺少坐标复制/空间程序 |

`DLR2` 的成功种子使用 horizon 2、12 state bits、每 cell-step 260 个声明式 gate；测试共计 `14,040` 次 gate evaluation。所选 `DL1` 对照为 `4,428` 次，递归成功以约 `3.17×` 动态门评估和 `6×` state-bit update 为代价，而且只有 1/3 种子恢复正确算法。

## ARC 开发与一次性确认结果

下表使用严格的 demo-consistent hard exact；括号内仅在必要时给 raw exact。

| cohort | tasks | sparse | D4 sparse | D4+bgpad | DL1 | DL1-wide | DLC1 | DLR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `DEV-LOW-MDL` | 8 | 0 | 0 | **1** | 0 (raw 1) | 0 (raw 1) | 0 | 0 |
| `CONF-INDUCTION-HASH` | 12 | 0 | **1** | **1** | 0 | 0 | 未运行 | 未运行 |

开发集中 `DL1/DL1-wide` 的 raw 命中都是 `25d8a9c8`，但所选硬电路没有全精确解释 demonstrations，所以严格分数为零。D4+bgpad 的 `3aa6fb7a` 命中满足 demo exact。确认集中 D4 与 D4+bgpad 都命中 `b60334d2` 且满足 demo exact；它们不是两个不同任务。

这否定了本轮预设的 `ARC-DL-C02` 强命题：当前 DiffLogic prior 没有改善冻结 induction-gap cohort。它同时给出更具体的正结果：少样本 ARC 中，公开而正确的不变性/边界假设比无任务语义的通用 gate relaxation 更有效。

### 真实 ARC 分层 probes

| probe | tasks | 主要方法 | strict exact | 诊断 |
|---|---:|---|---:|---|
| positive local-rule | 4 | sparse / D4 | 4/4 | 冻结局部表在其正例上可靠 |
| positive local-rule | 4 | DL1 / DL1-wide | 1/4 | 两者只解出 `c8f0f002`；说明真实任务可训练，但远不如精确表稳定 |
| representation-gap | 4 | DLO / DLF / MLP | 0/4 | MLP 在 2/4 拟合 demos 后 test 仍全失败 |
| local-conflict | 4 | DLR / DLO / MLP | 0/4 | MLP 仅 1/4 拟合 demos；hidden recurrence 没有覆盖真实冲突 |
| shape-change | 4 | DLF / MLP | 0/4 | MLP 在 2/4 拟合 demos；shape router 没有学到内容映射 |

这些 probe 使结论比单纯 `0/12` 更明确：实现不是完全不能在真实 ARC 上训练——`c8f0f002` 是严格正例；失败集中在稳定性、表示、冲突与空间程序，而不是 hard exporter 的语义错误。

## 压缩、效果和计算

### 同效果时有没有优势

constant-canvas 是当前唯一能做近似同效果比较的正例：DLF 与 MLP 均为 3/3 种子 exact。

| 指标（demo-selected seed） | DLF | MLP | 解释 |
|---|---:|---:|---|
| trainable parameter payload | 190,464 bits | 1,852,800 bits | DLF 小约 9.73× |
| declared deployment payload | 7,176 bits | 1,852,800 bits | 固定宽度 gate/wire code 对 dense 权重 payload，小约 258×；不是最小码 |
| 抽象动态运算 | 2,976 gate eval | 460,800 dense MAC | 单位不同，不能直接换成 PPA |
| 所选 seed 训练时间 | 18.21 s | 5.87 s | 当前 GPU 原型 DLF 反而慢约 3.10× |
| hard task exact | 1 | 1 | 仅此受控任务成立 |

这支持“硬门可能压低部署存储与 dense 化”的窄命题，但不支持“已经在真实 neuro-symbolic/ARC 上效果几乎不变且更快”。先前 464-rule NumPy 编译器的 binary4/one-hot 路径中位数还比 direct categorical 慢约 4.64×；软件 interpreter 和硬件综合必须分开。

### 为什么门编译不是当前瓶颈

把完整 solver 写成：

```text
y_hat = Decode( Rollout_H( F_theta, Topology(x,D), Represent(x,D) ), Canvas(x,D) )
```

那么 gate optimization 只改变 `F_theta`。如果 `Canvas` 给错输出形状、`Represent` 没有对象/任务变量，或 horizon 的因果锥到不了所需位置，任何 `F_theta` 都无法输出正确答案。

对 radius-1 CA，H 步后的 cell 最多依赖 Chebyshev 距离 H 内的信息；远距离 marker 需要足够 H 或显式全局 context。固定输入/输出 lattice 不能凭局部门创造新的输出坐标。少量 demonstrations 对未见 patch 的输出通常不唯一，soft relaxation 只改变优化几何，不增加识别正确规则所需的信息。D4 和 background padding 有效，正是因为它们加入了可验证的不变性，而不是因为它们有更多连续参数。

从 MDL 角度，门化只有在选定 basis 中存在短 circuit 时才压缩。一个 layer gate ID 至少要说明 16 选 1，wiring 还要付 source index；随机 LUT 或错误 basis 可以比 raw table 更长。当前 fixed-width hard payload 是可审计上界，不是 Kolmogorov complexity、ABC 最小网表或技术映射后面积。

## 是否浮现了 DiffLogic-CA 论文中的结论

部分浮现：

- 学习阶段可以使用连续 gate mixture，部署阶段可以变成纯二值 recurrent circuit；
- successful synthetic 的 soft/hard exact 相同，独立 hard executor 一致；
- recurrence 可以让局部规则跨多步传播；
- 与 dense MLP 相比，声明式参数与部署 payload 可以显著更小。

没有浮现或尚未验证：

- 多任务 ARC 上的稳定训练与 damage recovery；
- 只靠 fixed random wiring 自动获得对象级、全局和空间程序；
- 真实硬件 PPA 优势；
- 拓扑学习或最小门电路；
- 从少量 demonstrations 可靠选择正确规则。

本实现是 ARC-adapted architecture，不是 Google notebook topology 的逐层训练复现。官方 hard GoL/checkerboard artifact 的语义回放仍在单独结果中，不能拿来替代本轮训练证据。

## 当前 ARC 成功率低的具体原因

1. **训练信号极少且规则不唯一。** 2--4 个 demonstrations 无法覆盖大量局部 patch；固定 wiring 与 Boolean circuit prior 不能自动选择人类预期的不变性。
2. **binary4 是编码，不是对象语义。** 四个 color bits 让颜色变换可门化，却没有“最大对象、内外、对齐、复制、计数、角色绑定”等变量。
3. **当前 object raster 是静态提示，不是对象程序。** 它没有对象 slot、关系图、对应匹配、选择和写回控制；MLP 对 object-anchor 的 demo 记忆后 OOD 失败直接显示了这一点。
4. **shape router 只解决尺寸，不解决内容生成。** constant shape 成功，scale×2 失败，说明还缺输入坐标到输出坐标的可执行映射。
5. **递归优化不稳定。** shift×2 只有 1/3 seed 学到硬算法；更简单的 dilation×2 仍失败。BPTT、straight-through 和 fixed wiring 的联合优化高度非凸。
6. **任务条件不足。** broadcast global bits 告诉网络 palette/background 等事实，却没有从 demonstrations 归纳出可组合控制程序。
7. **候选选择比门表达更关键。** 确认集中 D4 的 1/12 与 DiffLogic 的 0/12 表明，正确 prior 的选择优先于门层容量。

## 这些适用前提在神经符号系统里常见吗

“规则可压缩、grounding 高置信、数据布局稳定”三项各自并不少见，但三者同时成立只覆盖神经符号系统中的一个重要子域，而不是默认状态。

| 前提 | 较常见的场景 | 容易失效的场景 | 本轮含义 |
|---|---|---|---|
| 规则可压缩 | 类型检查、有限约束、知识图谱关系过滤、棋盘/CA 更新、协议和控制状态机 | open-world 常识、长证明搜索、随机/实例特定 LUT、需要大外部记忆的任务 | gate/MDL 只应接管能被短 circuit 描述的 transition |
| grounding 高置信 | ARC 这类原生离散符号、传感器后有可靠校准与 abstention、人工结构化数据库 | 遮挡、类别混淆、分布外视觉、连续概率事实 | hard gate 不应吞掉尚未解决的感知不确定性 |
| 布局/拓扑稳定 | 固定网格、给定图、数据库 schema、已知对象槽和邻接关系 | 动态对象发现、可变画布、关系随任务改变、需要学习读写地址 | fixed wiring 很省，但前提是 topology 已经正确 |

因此 LGN/DiffLogic 最合理的位置是一个**条件化的编译后端**：上游表示与 router 先确认这三项近似成立，再把局部 dense/continuous 模块硬化；若不成立则保留 soft、搜索或 raw route。本轮 D4/bgpad 胜过通用门训练，正好说明 router 必须选择语言和边界假设，不能只选择 gate ID。

## 下一步最值得做什么

当前结果不支持继续单纯加宽 gate layers。更合理的下一版是：

1. 用对象 slots/scene graph 表示输入，并学习 demo-conditioned 对象对应与关系；
2. 把 shape、crop、tile、reflect、scale、copy-path 写成可执行 canvas/topology program，由 MDL 或 validation 选择；
3. 让 gate CA 只承担选定 program 内部的局部可编译 transition，而不是独自承担任务解释；
4. 为 recurrence 加 curriculum、逐步监督或可验证中间 invariant，并在多距离任务上选择 horizon；
5. 将 D4、background、颜色置换、方向性等先验作为可 abstain 的候选语言路由，而不是无条件 augmentation；
6. 只有在真实任务 strict exact 不降后，再导出 Verilog/ABC 并测量综合 PPA。

## 复现与原始证据

- 协议：`notes/design/difflogic-arc-v1-contract.md`
- 核心：`src/trainable_difflogic.py`、`src/arc_difflogic_model.py`
- ARC adapter/training：`src/arc_difflogic_features.py`、`src/arc_difflogic_train.py`
- runner：`src/run_arc_difflogic.py`
- 汇总与 hash 检查：`src/summarize_arc_difflogic_results.py`
- 合并结果：`results/arc_difflogic_v1_selected.csv`、`results/arc_difflogic_v1_summary.csv`、`results/arc_difflogic_v1_metadata.json`
- 正式 run：`results/runs/diffarc_formal_*_4f54aa4/`

正式 run 均从 clean source commit `4f54aa4` 运行，metadata 记录源码 SHA-256、环境、task content digest、训练配置和每个 artifact 的 SHA-256。公开来源见 `reports/references.md`：Google DiffLogic-CA、ARC-NCA、ARC-AGI-2。所有 CPU/GPU 时间都只属于当前软件栈，不是 ASIC/FPGA PPA。
