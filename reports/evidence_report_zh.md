# 逻辑门作为 Neuro-Symbolic AI 局部编译层：证据报告

## 结论先行

当前证据支持一个有边界的结论：逻辑门适合放在

```text
encoder / LLM proposal
    -> named predicates (+ uncertainty)
    -> learned or compiled Boolean gate DAG
    -> BFS / Datalog / SAT / planner / verifier
```

中，承担重复调用、局部组合、候选过滤和可编译的 bit-level 计算。它不适合替代递归、proof search、规划、概率边缘化或感知模型。

本轮补做了两个此前缺失的闭环实验：

1. `SoftGateCircuit` 是一个固定 wiring 的可微门混合代理。它先学习 AND/OR/XOR/NAND 的 softmax 权重，再把每个节点 argmax 编译成硬门，测量 soft→hard 的端到端掉点。
2. 将同一个 learned gate 接入分层图 BFS，分别测量 `LearnedSoftGate+BFS` 和 `LearnedHardenedGate+BFS`，而不是只报告 oracle 局部谓词。

这些结果证明“接口可以兼容”，但只在局部规则可压缩、谓词已经二值化或带可靠置信度、以及后端 solver 保留为独立模块时成立。

## 1. 局部规则学习与 soft→hard 闭环

原有 GateBeam 是透明的 bit-packed 结构搜索代理，不是某个发表的 DLGN/Conv-DLGN 复现。在 8-bit 可组合规则、50% 训练覆盖、10 个划分上，GateBeam 的 OOD 平衡准确率为 `1.000 ± 0.000`，小 MLP 为 `0.823 ± 0.104`；随机 LUT 对照中 GateBeam 只有 `0.518 ± 0.170`。因此优势来自可压缩结构，而不是“门天然泛化”。GateBeam 的小真值表拟合约 `0.7–0.8 s`，高于小 MLP 的约 `0.20–0.22 s`，动态规则或频繁更新时这项编译成本可能抵消部署收益。

新增的 learned soft-gate 结果（`results/learned_gate_results.csv`）如下：

| 训练覆盖 | 方法 | 全表平衡准确率 | OOD 平衡准确率 | OOD Brier | soft→hard OOD 掉点 | 拟合时间 |
|---:|---|---:|---:|---:|---:|---:|
| 50% | LearnedSoftGate | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000015 ± 0.000003 | — | 0.425 ± 0.018 s |
| 50% | LearnedHardenedGate | 1.000 ± 0.000 | 1.000 ± 0.000 | 0 | **0.000** | 0.425 ± 0.018 s |
| 75% | LearnedSoftGate | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000017 ± 0.000002 | — | 0.494 ± 0.015 s |
| 75% | LearnedHardenedGate | 1.000 ± 0.000 | 1.000 ± 0.000 | 0 | **0.000** | 0.494 ± 0.015 s |

10 个 seed 中硬化电路都保持了 soft 输出的 OOD 分类结果；不同 seed 可能学到等价的门表达式，例如 `NAND NAND XOR NAND OR NAND AND`，所以不能把某一组门名当成唯一真值结构。这个实验是**固定 wiring 的 controlled proxy**：它验证 operator learning 和 argmax hardening 的接口，不等价于对任意 learned-LGN 结构搜索的证明。

## 2. learned-Gate+BFS：输入输出是否能融合

每个图边仍以 8 个 named predicates 输入；gate 输出边有效性的概率/bit；BFS 只负责全局递归。图的宽度为 8、16、32，路径长度为 4、8、12，每个条件 60 个图、每个训练覆盖 10 个 seed。结果在 `results/learned_gate_bfs_results.csv`：

| 训练覆盖 | 方法 | 平衡准确率 | 正例准确率 | 负例准确率 | 查询中位数 |
|---:|---|---:|---:|---:|---:|
| 50% | LearnedSoftGate+BFS | 1.000 ± 0.000 | 1.000 | 1.000 | 0.050 ms* |
| 50% | LearnedHardenedGate+BFS | 1.000 ± 0.000 | 1.000 | 1.000 | 0.043 ms* |
| 75% | LearnedSoftGate+BFS | 1.000 ± 0.000 | 1.000 | 1.000 | 0.049 ms* |
| 75% | LearnedHardenedGate+BFS | 1.000 ± 0.000 | 1.000 | 1.000 | 0.042 ms* |

`*` 只计时预计算边之后的 BFS 查询；门推理、输入布局和 pack/unpack 不在该列内。

![learned gate hardening and BFS](../figures/learned_gate_integration.png)

在这个合成分布中，soft gate 先以 `p>=0.5` 生成边，hardened gate 直接用编译后的 Boolean DAG；两者都在 4/8/12 层保持完美图级结果。它补足了此前“oracle gate + BFS”证据，但仍不能推出真实感知误差下的端到端无损：图边特征来自已知的 8-bit 合成谓词，soft BFS 也使用了阈值化边，而不是概率路径边缘化。

固定深度反例仍然成立：`FixedK=4GateCircuit` 在长度大于 4 时平衡准确率为 0.5、正例准确率为 0；oracle 局部谓词加 BFS 在长度 2–12、宽度 8–32 的 30 个独立条件上为 1.0。因此门层和全局 solver 的职责必须分开。

## 3. 计算效率与真正瓶颈

在 `2^20 = 1,048,576` 个样本上，CPU/NumPy 测量为：

| 路径 | 中位时间 | 显式输入/活动 buffer | 相对 float t-norm |
|---|---:|---:|---:|
| float t-norm | 33.38 ms | 33.55 / 34.60 MB | 1.0× |
| NumPy bool | 7.04 ms | 8.39 / 9.44 MB | 4.74× |
| 预打包门内核 | 0.430 ms | 1.05 / 1.18 MB | 77.6× |
| pack + kernel + unpack | 10.02 ms | 8.39 / 10.62 MB | 3.33× |

因此“离散、稀疏、可编译”在重复查询内核上成立，但端到端优势只有约 3.33×；pack/unpack、缓存驻留和调用次数是接口成本。关系过滤也说明只把每个 pair 换成门并不会消除组合爆炸：`N=2048` 时索引候选只检查 `0.0373%` 的 pair，约比 dense 快 6.9×；小 `N` 时 Python 索引开销反而可能更慢。真正需要和门一起设计的是候选生成、数据布局和内存驻留。

训练/编译与部署的瓶颈是不同的：GateBeam 在小表上的结构搜索约 0.7–0.8 s，learned soft-gate 代理约 0.42–0.49 s；部署后硬门只需微秒级局部查询。若规则频繁更新，编译时间可能成为主导；若规则固定且重复调用，编译摊销后 bit-pack 和候选生成才是主要接口成本；递归、proof search、规划和概率推理仍然是全局复杂度来源。

## 4. grounding 风险与 hardening 条件

独立 bit-flip 率为 0.20 时，原有局部实验的分类准确率 hard/soft 都约为 `0.694`，但 Brier score 分别为 `0.306 ± 0.002` 和 `0.204 ± 0.001`。硬门放大 grounding 错误并丢弃不确定性；soft 概率电路保留了后验信息。因而 hardening 只有在谓词已高置信二值化、误差代价可接受，或系统能把低置信输入回退给概率模块/solver 时才合适。

## 5. 回答“兼容吗、能否近乎无损、是否是瓶颈”

- **兼容性：** 兼容的是 `named predicates → local gate → solver` 接口，而不是把整个 neuro-symbolic 系统硬化成一个电路。新增 learned-Gate+BFS 在合成 clean 输入上验证了这条接口。
- **近乎无损：** 在有限、可压缩、高置信的局部规则上，本轮 soft→hard OOD 掉点为 0，图级 soft/hard BFS 也均为 1.0；这不能外推到随机 LUT、噪声 grounding、概率推理或动态规则。
- **瓶颈：** 不是单一瓶颈。静态部署的瓶颈转向输入 pack/unpack、索引/候选生成和全局 solver；动态系统的瓶颈可能回到 GateBeam/soft-gate 的训练与编译；不确定 grounding 下最明显的系统风险是 hard gate 的错误传播。

## 6. 证据边界与复现

本项目仍是合成 CPU/NumPy 研究原型：没有视觉/语言端到端 grounding、FPGA/ASIC PPA、真实 DLGN 训练复现或概率路径边缘化。learned-Gate+BFS 的计时不包含门推理和数据布局；soft BFS 也先以 0.5 阈值把边概率离散化。新增 `SoftGateCircuit` 明确标为固定 wiring 的代理，避免把它误称为通用 LGN。原始输入哈希、环境、CSV 和图表见 [provenance.md](provenance.md)、`results/` 和 `figures/`；当前回归测试为 10/10 通过；完整工程见 [README](../README.md)。
