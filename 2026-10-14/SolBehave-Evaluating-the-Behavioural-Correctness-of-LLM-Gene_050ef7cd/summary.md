---
title: "SolBehave-Evaluating-the-Behavioural-Correctness-of-LLM-Gene"
source: https://arxiv.org/pdf/2610.18745v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-02 09:04:38"
field: "智能合约代码生成评估"
keywords: ["smart contracts", "large language models", "code generation", "differential testing", "dynamic condition response graphs", "Solidity", "Ethereum", "behavioural fidelity"]
innovations: ["提出三类互补的行为保真度预言机（差分模糊测试、历史交易回放、DCR图引导合成回放）统一评估LLM生成合约的行为正确性", "设计神经符号交易序列生成器：LLM提取DCR图约束、确定性图遍历结合on-chain guard求值生成合法调用序", "实证揭示编译成功率与行为等价性的巨大鸿沟，并证明提取DCR图与生成Solidity是两项独立能力"]
benchmarks: ["SolBehave", "Ethereum mainnet verified contracts", "SolBench", "SolContractEval"]
---

# 论文速读：SolBehave-Evaluating-the-Behavioural-Correctness-of-LLM-Generated-Smart-Contracts

## 一句话总结
本文提出了 SolBehave，一个基于三个互补行为保真度预言机（差分模糊测试、历史交易回放、DCR图引导的合成交易回放）的基准，用于系统评估 LLM 生成的 Solidity 智能合约的行为正确性；研究发现编译成功率（26.4%–92.7%）远不能代表行为等价性，仅 31.4% 的编译通过函数体在模糊测试下与原始合约等价。

## 研究问题与动机
- **编译≠正确**：现有评估范式主要衡量生成代码是否能编译（compile@k），但已部署的智能合约无法修补，单一潜伏缺陷会导致永久可被利用的安全漏洞。
- **现有行为评测的覆盖盲区**：差分模糊测试（如 SolBench）逐个函数测试，无法捕捉多函数调用序列产生的状态依赖缺陷；历史交易回放（如 SolContractEval）受限于链上真实用户行为，无法覆盖从未发生过的合法调用顺序（如所有者专属设置操作）。
- **缺乏面向 LLM 生成合约的行为基准**：现有测试工作针对手写合约设计，依赖开发者显式声明的不变量或规范，而生成的合约以原始合约为唯一规格参考，需要一个以原始合约为 ground truth 的评估框架。
- **DCR 图提取的可靠性存疑**：用 LLM 自动提取动态条件响应图（DCR graph）作为测试序列生成依据，其精度和召回率是否足以驱动 oracle 验证，尚需实证。

## 核心贡献（创新点）
1. **三类互补的行为保真度预言机（O1–O3）**：将差分模糊测试、历史交易回放与 DCR 图引导的合成交易回放统一于同一框架，三者分别覆盖单函数输入空间、链上历史轨迹和合法调用序列表达空间。
2. **神经符号交易的序列生成器**：LLM 从合约源码中提取 DCR 图（规则挖掘实例），由确定性图遍历算法枚举合法事件排序并通过对原始合约的执行验证，正确性由算法保证而非模型输出保证。
3. **开放行为基准与复现包**：包含 250 个经新鲜部署后验证能复现自身链上历史的合约、每个合约的 DCR 图、交易序列及再生成/评估脚本，以 MIT 许可证开源。
4. **量化编译与行为等价的巨大鸿沟**：实证发现编译通过率不能作为行为正确性的代理，六个模型中有五个的至少 1/8 编译合约在回放预言机下与原始合约发散。
5. **揭示提取 DCR 图与生成 Solidity 是两项独立能力**：最佳生成器（Gemma-4-31B）同时也是最佳提取器，但其他模型在两任务上的排名明显不一致，说明二者不存在天然映射。

## 方法详解
- **基准构建（§IV-A）**：从以太坊主网块高 16,308,190（2023-01-01）起抓取候选合约，筛除代理合约并保留 Etherscan 验证源码且至少有 200 条历史交易的合约；对候选合约进行新鲜部署复现性检查（需复现 ≥80% 历史交易），最终按非平凡函数数（排除 constructor 和 view/pure）过滤到 1–20 个函数的 250 个合约。
- **代码生成（§IV-B）**：采用 stub-and-complete 流水线（遵循 SolBench），去除原始注释并为每个公开函数重新生成 NatSpec；删除所有函数体保留签名、可见性与修饰符，按源码顺序逐个让模型补全函数体；每合约生成 k=10 个独立样本。
- **O1 差分模糊测试（§IV-C）**：对每个可模糊函数（状态修改函数且参数均为基元类型），构造一个部署 C* 和 Ĉ 的 harness 合约并暴露包装函数；用 Echidna 在 10,000 次调用、序列长度 20、120 秒超时下搜索使两者返回/状态分化的输入；等价判据为未找到违反断言的输入。
- **O2 历史交易回放（§IV-D）**：取原始合约主网上前 1,000 条历史交易，以原顺序和原发送者重放至新鲜部署的 C* 和 Ĉ；比较每次调用的 status（成功/ revert）、gas 消耗和事件发射。
- **O3 DCR 引导的合成回放（§IV-E）**：
  - **DCR 形式化**：DCR 图由事件（每公共状态修改函数分支对应一事件，标注调用者角色）和四种关系构成——Condition (A →• B，B 在 A 执行前不可启用)、Response (A •→ B，执行 A 使 B 待处理)、Inclusion (A →+ B，执行 A 包含 B)、Exclusion (A →% B，执行 A 排除 B)；本文扩展允许关系携带 guard（基于合约状态变量的 Solidity 表达式）。
  - **图提取**：用 LLM 读取合约源码与 ABI，以 JSON 格式输出事件与关系，并为每事件建议一组参数值及编译时常量。
  - **验证序列生成（Algorithm 1）**：从 C* 的新鲜部署与初始 marking 出发，每步评估 guard、选出允许事件，按优先队列选取：obligation（响应关系强制）→ prerequisite（条件前置）→ UCB 探索； ownership-transferring 事件延迟到最后；对原合约执行所选事件，成功则追加到序列，失败则丢弃；达到预算 b 后输出一个验证序列。
- **比较轴心（§IV-F）**：主要判据为 status match rate（成功/ revert 一致）；次要观察 gas 偏差（>±10% 报警）与事件 signature+indexed args 一致；合约级发散定义为至少一个样本在至少一次调用上与 C* 的 status 不一致；聚合取均值与最小值。

## 实验与结果
- **模型与设置**：七款开源模型（Gemma-4-31B、Gemma-4-12B、Qwen3.6-27B、Qwen3-Coder-30B、Ornith-35B、Codestral-22B、Devstral-24B），4-bit GGUF、60k context、temperature 0.7、每函数最多 2,048 tokens；编译失败样本给一次同模型自修复（temperature 0.2）。
- **编译率（Table II）**：修复后 Compile@10 区间为 26.4%（Devstral-24B）至 92.7%（Gemma-4-31B），各模型排名保持不变。
- **O1 差分模糊（Table V）**： pooled 跨模型与样本，仅 31.4% 编译通过的函数体与原始等价；合同级（所有可模糊函数均等价）最高仅 35.5%（Qwen3.6-27B）；最强生成器 Gemma-4-31B 仍有 2/3 函数不等价。
- **O2 历史回放（Table VI, Fig.4a）**：六款模型的 13%–31% 编译合约在历史回放下与原始发散；仅 Gemma-4-31B 保持 <3%。
- **O3 DCR 回放**：发散率 13%–44%（Gemma-4-31B 同样 <3%）；matched vs. fixed 提取器设置下发散合约数变化 ≤2，证明互补性来源于交易源差异而非提取质量。
- **覆盖率对比（Table IV）**：O2 平均覆盖 55.7% 的状态修改函数；O3 覆盖 33.6%–47.3%；两者非嵌套——O3 多覆盖 246–311 个函数（1/6 的状态修改函数，多为 owner-only 设置/管理操作），O2 多覆盖 511–619 个函数。
- **互补性（Table VI）**：O1 单独标识 38%–69% 的发散合约；O1 在 N 内严格强于 O2/O3（七个模型下仅 1 个合约仅被回放预言机捕获，可能源于多交易序列依赖）；O2 与 O3 的互补性显著：每个模型的 O3-only 段非零（Gemma-4-31B 为 3，Codestral-22B 为 17）。
- **DCR 图验证（Table III）**：Gemma-4-31B 在 30 合约 200 个关系上 human-validated，整体 precision 0.72、recall 0.88；condition 关系 precision 达 0.96、recall 0.84，保障调用序的正确性。
- **最强结果**：Gemma-4-31B 在 Compile@10（92.7%）、O1 函数等价率（33.4%）、O2 status match（99.6%）、O3 status match（98.4%）和 DCR 图 coverage（47.3%）上均为最优。

## 相关工作脉络
- **SolBench [10]**：对 7,604 个合约做差分模糊测试，但每函数在原合约其余部分不变的情况下独立 fuzz；本文在完整重生成合约内 fuzz，允许一个函数的缺陷污染另一函数的状态，更贴近真实部署场景。
- **SolContractEval [11]**：仅回放历史主网交易，受限于真实调用顺序；本文的 O3 能触达链上从未发生过的合法前件序列（如 owner-only 设置）。
- **SolEval [9]**：以 pass@k 和 compile@k 为主，辅以 gas 与静态漏洞率，不测试行为等价；本文强调行为语义而非成本/漏洞模式。
- **Sequence-aware fuzzers (IR-Fuzz [29], MuFuzz [30])**：针对手写合约，从代码结构推断调用序以扩大分支覆盖；本文目标不同——比较两个合约，需要的是原始合约视为合法的调用序（来自 DCR 规范），而非探索代码路径。
- **LLM4FUZZ [28]**：用 LLM 推断正确性条件并引导 fuzzer，仍面向手写合约的显式属性检查；本文的 DCR 图从合约源码自动提取，无需人工声明不变量。
- **HighGuard [34]**：使用手写 DCR 图在运行时监控合约；本文首次将自动提取的 DCR 图用于主动测试序列生成，并应用于 LLM 生成合约评估。

## 局限性与未来方向
- **仅以 status 为发散判据**：两次 revert 即便原因不同也算"一致"，miss 部分语义差异；return value 因用户账户不可见未纳入 O2/O3。
- **O3 仅回放原合约接受的调用**：未对原合约拒绝的 pruned 调用做负向测试（即不能捕获生成合约意外接受非法调用的缺陷）。
- **DCR 图提取的系统误差**：inclusion 与 exclusion 易被混淆（一次性子条件误报为 inclusion/exclusion、可恢复的零值误报为排除），导致图过约束；当前依赖 DCR 的宽松语义与 on-chain guard 评估缓解，但未根本消除。
- **参数搜索缺失**：pruned 调用中 0.9% 回落到历史交易解码的参数，其余均由 LLM 建议；固定参数+图序组合可能导致大量调用失败，未来需结合 fuzzer 联合搜索参数。
- **基准规模与领域局限**：250 合约以 token 类为主，规模有限；未覆盖复杂 DeFi、桥接、治理合约，这些场景中顺序依赖缺陷更为突出。
- **单一标注者**：DCR 图人工验证仅一位熟悉 DCR 形式的作者完成（15 小时专家时间），尽管抽样可复现，但与多标注者交叉验证的稳健性待提升。

## 研究启发与可借鉴点
- **三方预言机互补的评估范式**：单一测试源（历史或 fuzzing）均有结构性盲区，将历史回放、函数级差分 fuzz 与规范引导合成序列结合，可显著提升行为缺陷检出率；该方法可迁移至其他代码生成任务（如编译器 IR 生成、协议栈实现生成）。
- **神经符号分离设计**：LLM 负责提取约束（DCR 图），确定性图遍历负责序列枚举与 on-chain guard 求值，将 LLM 的不确定性与符号算法的可靠性解耦；这一分工原则可复用于需要"从非结构化规范生成可达路径"的任务。
- **guard-on-chain 求值替代静态推导**：直接从部署合约读取状态变量求值关系 guard，避免了对状态转移的预测误差；该技巧可推广至任何需在执行期动态判定先决条件的测试生成场景。
- **self-repair 保留模型间可比性**：让模型自行修复自身编译错误而非引入外部修复器，维持模型能力边界的一致定义；在大规模模型对比中可作为标准化预处理环节。
- **以"原始部署复现率 ≥80%"作为合约入选门槛**：确保 ground truth 本身的可复现性，过滤掉因 EVM 版本漂移或数据损坏导致的不可靠基线；可作为区块链代码生成基准构建的通用质量门控。

## 关键术语表
- **SolBehave**：本文提出的以太坊智能合约行为正确性评估基准，包含 250 个经复现验证的合约与三预言机评估框架。
- **行为保真度预言机（Behavioural Fidelity Oracle）**：以原始合约 C* 为 ground truth，通过比较生成合约 Ĉ 在相同输入下的行为输出（status、返回、事件、存储）判定是否等效的测试模块。
- **Differential Fuzzing（差分模糊测试）**：在同一 sandboxed chain 上并行部署 C* 与 Ĉ，用 fuzzer 生成输入并断言两者行为一致，任何分化均标记为缺陷。
- **Historical Transaction Replay（历史交易回放）**：将合约主网上真实用户交易按原序重放至新鲜部署的 C* 和 Ĉ，比较 receipt 判定行为一致性。
- **Dynamic Condition Response (DCR) Graph**：一种声明式事件基过程模型，通过 condition/response/inclusion/exclusion 四种关系刻画事件间的调用序约束，本文扩展支持 state-dependent guard。
- **Validate Sequence Generation（验证序列生成）**：基于 DCR 图与 on-chain guard 求值，以 obligation→prerequisite→UCB 探索的优先级策略构建经原合约接受的成功调用序列。
- **Compile@k**：在 k 个生成样本中通过编译的比例，本文取 k=10。
- **Status Match Rate**：重放交易中 C* 与 Ĉ status 一致的比例，均值作为合约级指标；任一调用 diverge 即判定合约发散。

## 可复现要素
- **数据集**：250 个以太坊主网合约（block ≥16,308,190），经新鲜部署复现性校验（≥80% 历史交易复现），来源于 Etherscan 验证源码；**开源**，链接 https://github.com/disi-unibo-nlp/sol-behave（MIT 许可证）。
- **代码/脚本**：合约再生成流水线、DCR 提取 prompt 与 JSON schema、三预言机评估脚本均随 replication package 开源。
- **模型**：七款模型以 4-bit GGUF 形式通过 llama.cpp 服务，context 60k、temperature 0.7（生成）/0.2（修复）、每函数 max 2,048 tokens；硬件为单卡 RTX 5090。
- **关键超参**：k=10 样本、每合约 1,000 条回放交易（O2/O3）、O1 Echidna 配置（10,000 调用、序列长 20、120s 超时）、O3 预算 b=1,000、guard 阈值 ±10% gas、β=0.7（obligation 概率）、α=0.8（prerequisite 概率）。
- **随机种子**：seed=42。
