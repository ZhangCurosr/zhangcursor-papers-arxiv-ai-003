---
title: "Who-May-Release-an-Agent-s-Promise-Standing-over-Commitments"
source: https://arxiv.org/pdf/2610.21203v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 21:46:57"
field: "多用户智能体系统与承诺管理"
keywords: ["multi-user agents", "commitment standing", "instruction conflict", "state tracking", "agent evaluation", "sycophancy", "tool-use agents"]
innovations: ["定义承诺管辖权并构建15单元格评估环境分离两类错误", "揭示规模导致的错误模式翻转（override→deference）", "自维护账本+策略组合达到无oracle的最佳性能"]
benchmarks: ["1440-episode controlled chat environment", "15-cell commitment release matrix"]
---

# 论文速读：Who-May-Release-an-Agent-s-Promise-Standing-over-Commitments

## 一句话总结
本文针对多用户共享助手场景中"谁有权取消已做出的承诺"这一问题，定义了**承诺管辖权（standing over commitments）**的概念，构建了包含 1,440 个片段的受控评估环境，系统测量了 6 个模型在未经授权的释放（unauthorized release）与债权人覆盖（creditor override）两种错误上的表现，并验证了策略规则与自维护承诺账本（ledger）的修复效果。

## 研究问题与动机
1. **多用户场景下的承诺归属问题**：助手向团队中特定成员（债权人/creditor）做出承诺后，其他成员（同事、上级、机器人）声称取消时，助手应如何判断谁有权解除该承诺？现有多用户指令冲突基准仅关注"谁的指令优先级更高"，未建模承诺的归属关系。
2. **两类错误需被区分测量**：模型可能犯两种方向不同的错误——①**未经授权释放**（unauthorized release）：听从无管辖权者的话取消承诺；②**债权人覆盖**（creditor override）：忽视债权人已取消的承诺而继续执行。仅测量最终准确率无法区分这两种本质不同的失败模式。
3. **规模与能力的异质性影响**：小规模模型倾向于覆盖债权人（即使债权人已明确取消仍执行），大规模模型倾向于对任何人的直接指令无条件服从（deference），启用推理（thinking）对小模型的效果类似于扩大规模。
4. **无 oracle 的部署需求**：理论上的干预（告知规则、承诺归属、状态）是上界，实际部署需要助手自行维护承诺账本，仅靠策略提示能否达到同等效果？

## 核心贡献（创新点）
1. **定义承诺管辖权并构建评估环境**：提出承诺三元组 $c = \langle A, y, \varphi, \alpha \rangle$（代理、债权人、条件、动作）及管辖权集合 $\text{St}(c)$，设计了涵盖 15 个单元格 × 2 种策略 × 2 种时间间隔 × 8 种承诺类型 × 3 种表面变体的 1,440 片段测试集，显式分离两种错误。
2. **揭示模型行为的两个 regime**：发现小规模开放模型（如 Qwen3.5-9B）在债权人直接取消后仍以 69% 概率执行承诺（override）；而大规模模型（如 Qwen3.8-27B、gpt-6.1-sol）在同事直接取消时以 89%/83% 概率丢弃承诺（unauthorized release），在同一模型族内规模增大使一种错误替换为另一种。
3. **解耦两类错误的信息需求**：通过 oracle 干预链（B2→B3→B3+→B4→B4+）发现，告知规则或承诺归属可抑制未经授权释放，但无法修复覆盖；只有记录承诺**当前状态**（是否已释放）才能切断覆盖错误，证明两类错误依赖不同的信息输入。
4. **自维护承诺账本 + 策略的组合效果**：助手使用 `ledger_add` / `ledger_update` 工具自行维护承诺账本，在一段落策略配合下，gpt-6-luna 的严格准确率从 49%（仅策略）提升至 83%，且无需外部 oracle 干预。

## 方法详解
**承诺建模**：每个承诺表示为元组 $c = \langle A, y, \varphi, \alpha \rangle$，其中 $A$ 为代理，$y$ 为债权人（请求者），$\varphi$ 为触发条件，$\alpha$ 为承诺动作。管辖权集合定义为：
$$\text{St}(c) = \{y\} \cup \{z : y \text{ named } z \text{ to decide on } c\} \cup \{z : \pi \text{ lets } z \text{ release } c\}$$
即只有债权人本人、债权人指定的代理、或策略授权者拥有管辖权。

**utterance 类型与语义**：
- **直接释放**（direct release）："don't book Maple"——仅当发言者有管辖权时有效。
- **间接释放**（implicit release）："looks like we won't need a room"——同上。
- **转述释放**（relayed release）："Alice told me she doesn't need it"——无独立效力，应询问债权人。
- **事实声明**（fact）："Maple is closed on Friday"——无需管辖权，但需验证。
- **重新激活**（reinstatement）："go ahead with it after all"——撤销之前的取消，效力同释放。

**评估环境设计**：每片段包含债权人请求（附带预填充的承诺回复）、无关对话填充、委托步骤（部分单元格）、取消步骤、重新激活步骤（REINSTATE 单元格）、条件触发集成消息。共 20 个工具，包括承诺动作工具、验证工具、查询工具等。

**两种错误度量**：
- **覆盖错误 $O_{\text{cred}}$**：债权人有管辖权并已释放承诺后，代理仍执行承诺的比例。
- **未经授权释放 $U_{\text{rel}}$ net**：无管辖权者直接/间接要求取消后，代理未经咨询即丢弃承诺的比例，减去同间隔下无取消时的基线丢弃率（排除遗忘）。

**Oracle 干预梯度**：
- B2：附加一句话说原则（"承诺属于债权人，只有他们或其指定人可释放"）。
- B3：追加承诺账本（开放承诺、债权人、代理名）。
- B3+：B3 基础上附加当前状态行（"open"/"released by X"）。
- B4：在无管辖权者发言时附加提示（"此发言者无权释放，请咨询债权人"）。
- B4+：B3+B4 组合。

**自维护账本工具**：
- `ledger_add(owed_to, promise)`：记录承诺。
- `ledger_update(item, status, by, note)`：记录状态变更（open/released/done/on_hold）。
- 每次新消息后代理均看到自身账本快照。

## 实验与结果
**数据集与环境**：1,440 个受控聊天片段（6 种承诺类型构成 1,080 片段测试集 + 2 种开发集），6 个模型（Qwen3.5-9B、Qwen3.8-27B、Gemma-4-12B、Llama-3.1-8B、gpt-6-luna、gpt-6.1-sol）。

**核心结果（测试集，无策略 baseline）**：
| 模型 | 覆盖错误 $O_{\text{cred}}$ | 未经授权释放 $U_{\text{rel}}$ net | 严格准确率 Acc. |
|------|--------------------------|----------------------------------|----------------|
| Qwen3.5-9B | 69% [50, 86] | 9% [1, 20] | 32% [29, 35] |
| Qwen3.8-27B | 0% [0, 0] | 33% [12, 54] | 35% [31, 39] |
| Gemma-4-12B | 53% [45, 61] | 11% [3, 20] | 36% [33, 40] |
| Llama-3.1-8B | 32% [12, 56] | -3% [-13, 8] | 25% [21, 29] |
| gpt-6-luna | 21% [10, 33] | 22% [5, 41] | 29% [24, 33] |
| gpt-6.1-sol | 0% [0, 0] | 11% [-21, 31] | 50% [40, 56] |

**策略效果**：显式策略（explicit policy）使 Qwen3.8-27B 准确率从 35%→63%，gpt-6.1-sol 从 50%→81%；但对小模型覆盖错误无效（Qwen3.5-9B 仍 75%）。

**Oracle 干预效果**：
- B2（告知规则）：Qwen3.8-27B $U_{\text{rel}}$ net 从 33% 降至 15%，gpt-6.1-sol 从 11% 降至 -9%；但对覆盖错误几乎无影响。
- B3（告知归属）：进一步降低未经授权释放，但使 Gemma、Llama、gpt-6-luna 的覆盖错误**上升**（因账本持续显示"open"）。
- B3+（状态追踪）：唯一能修复覆盖错误的干预——Qwen3.5-9B 覆盖错误从 63% 降至 24%，gpt-6-luna 从 37% 降至 3%。

**自维护账本效果**：
- 无策略时：覆盖错误大幅下降（Qwen3.5-9B 67%→6%，Gemma 53%→4%），但未经授权释放急剧上升（Gemma 11%→55%）。
- 账本+策略组合：gpt-6-luna 准确率从 49% 提升至 83%，Qwen3.8-27B 从 63% 提升至 73%；四种模型超过最佳 oracle 干预（除 Qwen3.5-9B 持平）。

**Reinstatement 问题**：债权人取消后，同事说"go ahead after all"，Qwen3.5-9B 和 Gemma 在 89%-94% 的情况下立即执行（甚至在条件触发前），仅 gpt-6.1-sol 抵抗（8%）。

**推理（thinking）效果**：开启 thinking 对 Qwen3.5-9B 使覆盖错误从 67% 降至 33%，但未经授权释放从 9% 升至 23%，严格准确率变化不大（32%→38%），复现了"规模效应"。

## 相关工作脉络
1. **承诺与言语行为理论**：Searle (1976) 的 commissive 分类、Darwall (2006) 的第二人称立场、Owens (2012) 的释放权理论，为"承诺归属债权人"提供哲学基础；Singh (1999) 的多智能体社会承诺形式化是本文技术框架的直接前身。
2. **多用户指令冲突**：Yang et al. (2026) 的 Multi-user LLM agents、Fan et al. (2026) 的 Harness-MU、Wallace et al. (2024) 的 instruction hierarchy 均以"权限等级/优先級"解决冲突，而本文强调承诺的**关系性归属**而非绝对权限。
3. **主忠忠诚与代理人背叛**：Li & Shi (2026) 的"谁的阵营是你的代理？"研究单 principal 场景下的 counterparty capitulation，本文扩展至多 principal 环境中的跨承诺归属。
4. **记忆与状态追踪**：Liu et al. (2024) 的"lost in the middle"、Maharana et al. (2024) 的长程记忆基准、Cerruti et al. (2026) 的记忆授权清洗，本文的账本机制是对这类工作的任务导向应用。
5. **顺从性（sycophancy）研究**：Perez et al. (2023)、Sharma et al. (2024) 发现模型倾向于迎合用户观点，本文的"deference"（对任何人直接指令无条件服从）是其多用户场景的具体表现。
6. **工具使用与状态管理**：Yao et al. (2023) 的 ReAct、Schick et al. (2023) 的 Toolformer、Shinn et al. (2023) 的 Reflexion，本文的 ledger 工具是对这类工作的轻量级状态持久化扩展。

## 局限性与未来方向
1. **合成环境的生态效度**：所有片段为英文脚本，仅债权人会回复，字符串由语言模型起草，无法完全模拟真实工作空间的多变交互模式。
2. **承诺类型偏差**：`post_announcement` 类型因模板缺陷（请求未含发布文本）导致大模型高丢弃率，排除后结果更清晰，但揭示了模板设计对评估的敏感性。
3. **单一样本限制**：除推理实验外，多数运行仅用单次采样（temperature 0.7），统计波动可能影响小概率事件（如覆盖错误）的估计。
4. **无管辖权者事实声明的处理不足**：当同事声称"行动不可能"且验证为假时，大模型常确认债权人后询问声明者但得不到回复，导致承诺被不合理丢弃——真实场景中声明者会回应。
5. **未来方向**：①将账本机制泛化为更通用的"承诺状态机"；②探索策略文本的自动化生成（而非人工编写）；③研究重新激活（reinstatement）的独立建模（当前状态追踪无法完全阻止该错误）；④扩展到非英语及跨文化场景。

## 研究启发与可借鉴点
1. **错误解耦的评估设计**：将"未经授权的释放"与"债权人覆盖"作为两个独立指标分别测量，避免了单一准确率的掩盖效应，为多用户 agent 评估提供了可复用的指标体系设计范式。
2. **Oracle 干预梯度法**：通过逐步添加信息（规则→归属→状态→提示）来定位模型失败的信息论根源，这种方法论可迁移至其他 agent 能力诊断场景（如工具使用、长期规划、安全对齐）。
3. **自维护账本作为通用状态管理原语**：`ledger_add`/`ledger_update` 的极简工具接口证明了"让代理自己写状态"比"外部 oracle 注入状态"更具部署可行性，且与策略提示结合可超越 oracle 上界——这一设计对 RAG 系统、长期记忆 agent 有直接借鉴价值。
4. **规模-能力 trade-off 的量化刻画**：同一模型族内从小规模到大规模，错误模式从 override 翻转为 deference，揭示了"规模带来状态追踪能力但不带来规则理解"的普适规律，为模型选型（小型 vs 大型）提供了决策依据。
5. **Thinking 的双刃剑效应**：开启推理对小模型的修复效果类似扩大规模（减少 override 但增加 unauthorized release），提示团队在部署思考型模型时需同步引入规则/账本约束，否则可能转移而非消除风险。

## 关键术语表
**Standing（管辖权）**：对特定承诺有权做出释放决定的身份关系，属于（发言者，承诺）对而非绝对属性，债权人、债权人指定的代理或策略授权者拥有管辖权。

**Unauthorized Release（未经授权释放）**：代理在无管辖权者（同事、上级、机器人）直接/间接要求下，未经咨询债权人即丢弃承诺的错误行为。

**Creditor Override（债权人覆盖）**：代理在债权人或指定代理已释放承诺后，仍执行承诺的错误行为，本质是状态追踪失败。

**Commitment Ledger（承诺账本）**：代理自行维护的承诺状态记录，包含承诺内容、债权人、代理名、当前状态（open/released/done/on_hold）及变更原因。

**Deference（顺从/ deference）**：大规模模型对任何直接指令无条件服从的行为倾向，类似于 sycophancy 在多用户场景的表现。

**Reinstatement（重新激活）**：债权人已取消承诺后，第三方（通常是同事）声称"go ahead after all"的行为，本文发现几乎所有模型都会遵从此类重新激活。

**Strict Accuracy（严格准确率）**：在每个单元格中，代理不仅需要做出正确行动（执行/丢弃/咨询/验证），还需满足额外条件（如告知债权人委托取消、报告丢弃原因）的准确率。

**Oracle Ladder（神谕梯）**：B0→B2→B3→B3+→B4→B4+ 的逐步信息注入序列，用于定位修复两类错误所需的信息类型。

## 可复现要素
- **数据集**：论文声明已开源（代码、提示词、输出和标签均发布），包含 1,440 个片段。
- **代码/权重**：全部开源， GitHub 仓库应包含评估脚本、提示词模板（Appendix F）、输出文件。
- **模型**：Qwen3.5-9B（bf16, vLLM）、Qwen3.8-27B（4-bit GGUF, llama.cpp）、Gemma-4-12B（bf16, vLLM）、Llama-3.1-8B（bf16, vLLM）、gpt-6-luna（OpenAI API, reasoning=none）、gpt-6.1-sol（OpenAI API, reasoning=low）。
- **关键超参**：temperature=0.7，top-p=1（Qwen）/0.95（Gemma）/0.9（Llama），thinking off/on；采样次数：除推理实验外多数为 1 sample/condition。
- **评估指标**：覆盖错误 $O_{\text{cred}}$、未经授权释放 $U_{\text{rel}}$ net、咨询率 $Q_{\text{relay}}$、通知率 N、重新激活服从率、严格准确率，95% bootstrap 置信区间按承诺类型聚类。
