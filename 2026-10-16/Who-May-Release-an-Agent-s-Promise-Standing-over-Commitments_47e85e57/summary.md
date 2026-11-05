---
title: "Who-May-Release-an-Agent-s-Promise-Standing-over-Commitments"
source: https://arxiv.org/pdf/2610.21203v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 21:46:28"
field: "多用户对话中的代理承诺管理"
keywords: ["multi-agent commitment", "standing over promises", "agent memory", "multi-user agents", "commitment tracking", "agent benchmark"]
innovations: ["定义承诺资格standing并建立15单元格受控评测环境以分离两种错误", "提出代理自维护承诺账本ledger配合简短政策即可达83%严格准确率", "发现模型尺度增长将override错误转化为deference错误及thinking的等效作用"]
benchmarks: ["1440-episode controlled chat environment", "15-cell experimental matrix"]
---

# 论文速读：Who May Release an Agent's Promise? Standing over Commitments in Multi-Party Chat

## 一句话总结
本文定义了"承诺资格（standing）"概念——只有承诺的债权人或其授权代表才能撤销代理的承诺，与职位高低无关；通过1,440个受控对话场景测试六个模型，发现小型模型会忽略债权人的撤销（override），大型模型则会无条件执行任意人的直接请求（deference），最终提出一个自维护的承诺账本（commitment ledger）配合简短政策即可将最优模型的严格准确率从49%提升至83%。

## 研究问题与动机
- **多用户对话中的承诺撤销权归属问题**：当团队共享同一个助手时，助手对不同人做出多项承诺；同事后续说"不用了"，助手应听谁的？现有多用户工作以权限等级排序指令优先级，但承诺已做出后属于债权人，与说话者职级无关。
- **两种错误尚未被现有基准分离评估**：（1）未经授权释放（unauthorized release）——听信无资格者的话取消承诺；（2）覆盖债权人（creditor override）——债权人已撤销仍继续执行；两者机制不同，需分别度量。
- **现有方法不足**：指令层级（instruction hierarchy）方法按职级裁决冲突指令，无法处理"承诺已归属特定人"这一语义；多用户代理工作关注权限分配而非承诺状态跟踪。
- **缺乏系统性测试环境**：没有基准同时覆盖债权人、委托代表、平级同事、经理、机器人等多种说话者身份及直接/隐含/二手/事实声明/恢复承诺等多种撤销形式。

## 核心贡献（创新点）
- **定义"承诺资格（standing）"并建立判定规则**：将承诺建模为四元组 $\langle A, y, \varphi, \alpha \rangle$，给出 $St(c)$ 的正式定义，区分直接撤销、隐含撤销、二手报告、事实声明和恢复五种话语类型各自的效力。
- **构建1,440条受控episode的评测环境**：15种实验单元格（说话者身份×话语类型）× 两种策略 × 两种时间间隔 × 8种承诺类型 × 3种表面变体，首次在程序上分离两种错误并分别度量。
- **发现模型行为的两大 regime 及尺度翻转现象**：小模型（Qwen3.5-9B）以69%概率覆盖债权人撤销，大模型（Qwen3.8-27B/gpt-6.1-sol）对任何人直接请求的未经授权释放率达83%–89%；同一模型族内尺度增长将一种错误转为另一种。
- **提出代理自维护承诺账本（ledger）方案**：两个工具 `ledger_add`/`ledger_update` 配合一段政策说明，无需外部 oracle 即可将 gpt-6-luna 严格准确率从49%提升至83%，达到或超越最强 oracle 干预水平。

## 方法详解
- **承诺的形式化定义**：承诺 $c = \langle A, y, \varphi, \alpha \rangle$，即代理 $A$ 对债权人 $y$ 在条件 $\varphi$ 满足时执行动作 $\alpha$；有资格者集合为 $St(c) = \{y\} \cup \{z: y\text{ 指定 }z\text{ 作为代表}\} \cup \{z: \pi\text{ 授权 }z\text{ 撤销}\}$。
- **话语类型及其效力规则**：
  - 直接/隐含撤销（direct/implicit）：仅发言者在 $St(c)$ 内时有效。
  - 二手报告（relayed）：如"Alice说她不需要了"——无自身效力，正确回应是询问债权人。
  - 事实声明（fact）：如"Maple周五关闭"——任何人可陈述，但需验证后才可撤销。
  - 恢复（reinstatement）：债权人撤销后他人再说"继续"，效力等同于相同发言者的撤销。
- **实验环境的15个单元格设计**：说话者包括 NONE、CRED（债权人）、DELEG（委托代表）、SUP（经理）、PEER（平级）、BOT（机器人），每种含直接/隐含撤销，另有二手报告、恢复、事实三种类型；搭配两种操作策略（显式/沉默）和两种时间间隔（短/长）。
- **两种核心误差的度量公式**：覆盖误差 $O$ = 债权人有资格撤销后仍执行的比例；未经授权释放 $U_{rel}$ = 无资格者直接/隐含消息后未咨询即取消的比例（扣除同期 NONE 基线以降低遗忘干扰）。
- **承诺账本方案**：给代理提供 `ledger_add(owed_to, promise)` 和 `ledger_update(item, status, by, note)` 两个工具；每条消息后向代理展示当前账本；配合一段政策说明（"请求属于提出者，仅其或其所指名人可撤销"），账本记录状态，政策提供规则，二者缺一不可。

## 实验与结果
- **数据集**：1,440条受控 episode（1,080条测试集，6种承诺类型），系统提示含完整团队成员名单及层级关系，20种工具调用。
- **评测模型**：Qwen3.5-9B、Qwen3.8-27B、Gemma-4-12B、Llama-3.1-8B（开源权重），gpt-6-luna、gpt-6.1-sol（API模型），共6个。
- **主要发现**：
  - 小型模型 override 严重：Qwen3.5-9B 在债权人直接撤销后仍执行占69% [50, 86]；Gemma 为53% [45, 61]。
  - 大型模型 deference 严重：Qwen3.8-27B 对平级同事直接撤销的未经授权释放占89% [75, 97]；gpt-6.1-sol 占83% [69, 92]。
  - 策略修复大型模型的未经授权释放：gpt-6.1-sol 从28%降至1%，Qwen3.8-27B 从64%降至21%；但对 override 无效，且使 gpt-6-luna 的 override 从21%升至49%。
  - 所有模型几乎都服从旁人的恢复请求（67%–100%），仅 gpt-6.1-sol 抵抗（8%）。
  - 启用 thinking 可将小型模型的 override 降低至与尺度相当水平：Qwen3.5-9B 从67%降至33%，Gemma 从53%降至9%。
- **最强结果**：gpt-6-luna 结合承诺账本+政策后严格准确率达 **83%**，较无策略基线（49%）提升34个百分点，与最强 oracle 干预（B4+）持平。
- **Oracle干预分层实验**：仅告知规则（B2）可减少未经授权释放但不影响 override；仅告知债权人归属（B3）减少未经授权释放但可能加剧 override；只有记录完整状态（B3+）才能有效降低 override。

## 相关工作脉络
- **指令层级（Instruction Hierarchy）**（Wallace et al., 2024; Zhang et al., 2026）：按特权级别排序指令优先级，本文指出承诺已归属特定人后不再适用该范式，强调 standing 而非 rank。
- **多用户代理权威分配**（Yang et al., 2026; Fan et al., 2026）：为每个用户分配权限级别解决冲突指令，本文认为承诺的撤销权属于债权关系而非职级关系。
- **社会承诺理论**（Singh, 1999; Yolum & Singh, 2002; Fornara & Colombetti, 2004）：将释放权赋予债权人或治理上下文，本文采用此框架并扩展至 LLM 代理场景。
- **多代理承诺存储**（Hamblin, 1970; Walton & Krabbe, 1995）：为每个参与者维护承诺库，本文的账本设计与之呼应但面向多用户共享代理场景。
- **代理记忆与状态丢失**（Cerruti et al., 2026; Liu et al., 2024; Maharana et al., 2024）：记忆系统可能丢失撤销信息，本文的账本方案显式解决此问题。
- **主忠诚度与代理 capitulation**（Li & Shi, 2026）：代理在非本人委托人面前屈服，本文将大模型的 deference 与此现象关联，归因于缺乏明确规则而非纯粹能力不足。

## 局限性与未来方向
- 环境为合成英语职场对话，脚本由语言模型生成，非真实数据；仅债权人在撤销后回复，其他角色沉默。
- post\_announcement 承诺类型存在模板缺陷（未提供文案），部分模型因此从不执行，但排除后结论不变。
- 大多数实验使用单一采样（thinking off），结果可能存在随机性；oracle 干预为理论上限，非实际部署方案。
- 仅有一个标注者验证 judge，notify 指标的 inter-annotator 一致性较低（κ=0.64）。
- 事实声明为假时，部分大模型会向声明者或经理询问但得不到回复后直接放弃执行，低估了大模型处理事实的能力。
- 未来方向：将账本集成到真实多用户协作平台、探索自动学习撤销规则的策略、处理更复杂的多轮嵌套承诺场景。

## 研究启发与可借鉴点
- **两种误差分离度量的实验设计**值得借鉴：将 unauthorized release 和 override 分开评估，避免笼统的"准确率"掩盖模型真实缺陷，可迁移至其他多用户 agent 评测。
- **承诺账本（ledger）+ 简短政策**的组合模式实用性强：账本负责状态追踪，政策提供规则认知，二者互补，可推广至其他需要持久化承诺跟踪的 agent 系统。
- **"thinking 等效于尺度"的发现**提示：对小模型而言，引入 reasoning 可能比单纯放大参数量更能改善状态跟踪能力，可在团队内部小规模模型评测中验证。
- **15种单元格×多变量的系统实验矩阵设计**是构建可控 benchmark 的良好范例，尤其适合研究 agent 在复杂社交场景中的行为差异。
- **混合逻辑回归分析（Table 4）**展示了如何量化各因素（standing、rank、话语类型）对模型行为的独立贡献，方法可直接复用于其他行为分析研究。

## 关键术语表
**Standing（承诺资格）**：有权撤销某项承诺的主体身份，属于发言者与承诺的二元关系，仅债权人、债权人指定的代表或政策授权者可拥有。

**Creditor Override（债权人覆盖错误）**：代理在债权人有资格撤销承诺后仍继续执行承诺的错误行为，本质是状态跟踪失败。

**Unauthorized Release（未经授权释放）**：代理在发言者无资格时未经咨询即取消承诺的错误行为，本质是过度 deference 或顺从。

**Commitment Ledger（承诺账本）**：代理自维护的承诺状态记录，包含承诺内容、债权人、委托代表和当前状态，通过 `ledger_add`/`ledger_update` 工具更新。

**Deference（屈从）**：大模型将任何直接请求视为决定性信号的行为倾向，类似 sycophancy，表现为对无资格者的直接撤销不加质疑即取消承诺。

**Strict Accuracy（严格准确率）**：同时要求正确执行/取消承诺、且在有资格撤销时通知债权人、对二手报告和事实声明进行验证的综合评分。

**Reinstatement（恢复承诺）**：债权人已撤销后，另一人再次要求继续执行承诺的话语，多数模型会立即服从，视为独立的行为缺陷。

## 可复现要素
- 数据集：1,440条合成 episode，代码、prompts、输出和标签均已开源（论文声明）。
- 代码/权重：开源模型权重可通过 HuggingFace 获取（Qwen3.5-9B、Qwen3.8-27B GGUF 4-bit、Gemma-4-12B、Llama-3.1-8B）；API模型通过 OpenAI API 调用。
- 关键超参：temperature=0.7，top-p=1（Qwen）、top-p=0.95/top-k=64（Gemma）、top-p=0.9（Llama）；gpt-6.1-sol 使用默认 temperature 和低推理强度。
-  serving：开源模型使用 vLLM（除 Qwen3.8-27B 用 llama.cpp 4-bit GGUF）；API 模型通过 OpenAI API。
