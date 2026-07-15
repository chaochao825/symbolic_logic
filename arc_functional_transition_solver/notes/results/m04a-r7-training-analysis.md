# M04a r7 训练模型与效果审计

## 结论

M04a r7 训练得到的不是完整的 ARC 混合求解器，而是一个从零训练的
8,733,706 参数 Grid-CMLM：它用 Transformer encoder 读取 demonstrations 与
query input，用非因果 decoder 对给定目标形状中的 10 种颜色做条件掩码预测。
12 步、64 lane 的 MaskGIT 风格过程是外部离散重掩码调度；模型没有连续噪声、
score matching、扩散时间步 embedding 或学习到的功能状态转移矩阵。

训练证明模型学到了有意义的条件补全能力，但没有证明它能独立求解 ARC。
现存最优 checkpoint 是 step 12,000：冻结选择指标重算为 0.8713449，整体
masked-cell accuracy 为 71.20%。在 664 个完全遮挡验证 episode 上，单步格子
准确率为 62.13%，完整网格 exact 只有 1 个（0.15%）。实际 12 步、64 lane
Oracle@64、DSL union 增益、pass@2 和官方 ARC solve rate 均未测得。

## 训练与失败边界

- 训练计划：20,000 optimizer updates，每步 16 个单 episode microbatch。
- 每步来源：8 个 ARC-AGI-2 episode 与 8 个 RE-ARC episode。
- 数据增强：D4 变换、颜色置换和 demonstration 顺序置换。
- corruption：10% 完全遮挡；其余为非空、非全遮挡的随机比例掩码。
- 目标：只在被遮挡格子上计算 FP32 cross entropy。
- 优化：AdamW，峰值学习率 3e-4，2,000-step warmup，随后 cosine-to-zero。
- 实际进度：进程完成 step 13,150（210,400 个训练 episode），但最后一次完整
  checkpoint 是 step 12,000（192,000 个训练 episode）。
- 失败回执记录的直接终止异常：进度输出写入已关闭的 stdout，触发
  `BrokenPipeError`。本次只读远端审计未在已保存 checkpoint 中观察到 NaN/Inf
  或 state 损坏；这不等于公开仓库可以独立排除训练过程中所有潜在数值问题。
- 正式状态：`CAMPAIGN_INCOMPLETE`，没有原子发布的 training artifact、验证
  指标文件或 selected-checkpoint manifest。

本次 2026-07-15 只读远端审计观察到：六个 checkpoint 均包含模型权重、AdamW
moments 和 RNG state，157 个模型 tensor 全部为有限值。checkpoint 二进制未在
普通 Git 中公开，因此这些观察不能只依靠本仓库独立复核。step 13,150 的内存
权重没有保存，不能恢复。

## 只读重算结果

重算严格使用原冻结 validation manifest：2,656 个 episode、817 个 target
group、145 个 semantic parent。评估使用 `inference_mode`，禁用参数梯度，未构造
optimizer，未执行 backward，也没有继续训练。

| checkpoint step | parent-grouped masked CE ↓ | masked-cell accuracy | masked-region exact | full-mask cell accuracy | full-mask exact |
|---:|---:|---:|---:|---:|---:|
| 2,000 | 1.2439249 | 58.69% | 3.77% | 53.16% | 2/664 |
| 4,000 | 1.0610075 | 64.19% | 4.03% | 57.70% | 4/664 |
| 6,000 | 0.9763545 | 67.76% | 4.44% | 60.29% | 1/664 |
| 8,000 | 0.9211581 | 69.62% | 4.89% | 61.18% | 1/664 |
| 10,000 | 0.9020414 | 70.29% | 5.42% | 61.03% | 2/664 |
| 12,000 | **0.8713449** | **71.20%** | **5.46%** | **62.13%** | **1/664** |

step 12,000 按遮挡比例进一步分解：

| mask fraction | masked-cell CE | masked-cell accuracy | masked-region exact |
|---:|---:|---:|---:|
| 0.15 | 0.6839950 | 81.68% | 14.31% |
| 0.35 | 0.7153853 | 81.45% | 5.42% |
| 0.65 | 0.8335189 | 77.34% | 1.96% |
| 1.00 | 1.2747535 | 62.13% | 0.15% |

这说明模型更擅长“已有大部分目标结构时补局部缺口”，而不是从全 MASK 冷启动
构造完整程序化输出。整体 masked-region exact 包含部分遮挡 episode，不能当作
ARC task exact-match。

## 与原始混合架构的差距

当前 checkpoint 只实现 M04a 神经候选源。输出形状由外部 M03a/DSL 规则提出，
最多接受四个形状；checkpoint 本身不预测形状。训练和 checkpoint 中不存在：

- LLM/code model 的开放式假设生成；
- DSL 程序的联合训练或执行验证；
- LGN/MLP/XGBoost candidate ranker 或 repair policy；
- residual-directed local repair；
- 学习到的功能路由或脑区状态切换。

冻结 sampler 会永久提交高置信格子，只重掩码低置信格子，因此高置信错误不能被
后续步骤修订。它可以作为局部补全或候选先验的起点，但现有证据不足以升级为
“大规模候选生成已经提高 ARC coverage”的结论。

## 证据身份

- 模型语义：`afts-grid-cmlm/v0.1`
- sampler 语义：`afts-maskgit-cosine/v0.1`
- validation summary ID：
  `38dde5553291edcd133dfebbc3f39e77cfce6cf3fdcd640ae565996eb85c20f5`
- validation JSONL SHA-256：
  `030dd768a84683c81bdd85244827783a354c7a377837d4c53ddab37d5690ebf7`
- failure receipt SHA-256：
  `4c7780eaf66ddd42f23eab15f14fff15677129145f379efc94242a0d1f210efe`

机器可读的逐 checkpoint 指标、文件大小和 checkpoint SHA-256 见
`results/m04a_r7_posthoc_evaluation_20260715.json`。
