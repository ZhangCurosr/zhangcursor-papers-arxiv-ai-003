---
title: "SAFE-PERSISTENT-AND-EVOLVING-AGENT-HAR-NESS-FOR-UNDERSTANDIN"
source: https://arxiv.org/pdf/2610.11552v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 10:22:36"
field: "企业智能体与安全执行"
keywords: ["enterprise agent", "partial observability", "agent harness", "world model", "policy compliance", "abductive reasoning", "LLM agents"]
innovations: ["提出E-LEDGER多智能体安全持久harness，分离运行时控制与领域知识，代码化策略门防绕过", "提出WorldAbduct四视角诊断+溯因主动验证框架，发现并验证工具接口外的隐藏业务规则", "在WOW基准四种LLM backbone上均取得最高STCR，较最强基线MemoHarness提升5-15个百分点"]
benchmarks: ["World of Workflows (WOW)", "ScienceWorld", "DiscoveryWorld"]
---

# 论文速读：SAFE, PERSISTENT, AND EVOLVING AGENT HARNESS FOR UNDERSTANDING PARTIALLY OBSERVABLE WORLDS

## 一句话总结
本文提出了 **E-LEDGER** 多智能体框架和 **WorldAbduct** 演化方法，用于解决企业在部分可观测环境下的安全智能体执行问题；通过代码化策略审批层保障合规性，并结合基于溯因推理的世界模型演化机制发现并验证隐藏业务规则，在 WOW 基准上相比最强演化基线提升 5–15 个百分点的安全任务完成率。

## 研究问题与动机
1. **企业工作流需要严格的策略合规**：智能体的操作必须遵守组织政策、权限和审批要求，局部合理（locally plausible）的动作仍可能违反合规约束。
2. **环境高度部分可观测**：工具响应仅暴露交互表层，隐藏了跨记录联动效应、延迟副作用和状态转换规则，成功响应不代表最终状态已正确达成。
3. **长跨度任务需要持久状态追踪**：跨多个对象、读写的长期任务中，丢失已确认事实或未完成子目标会导致重复查询、错误动作或过早终止。
4. **已有 harness 方案在隐藏规则面前失效**：即使策略检查正确实现，若底层状态因未观测到的副作用而过时，检查本身也形同虚设，harness 必须学会理解工具接口背后的"世界模型"。

## 核心贡献（创新点）
1. **提出 E-LEDGER 安全持久多智能体 harness**：将运行时控制与领域知识分离，通过代码化审批层将组织策略编译为可执行程序，配合 world ledger 维护静态隐藏规则和动态任务状态——与已有工作的本质区别在于把"策略执行"从模型内部推理剥离到外部代码网关，防止模型绕过已知约束。
2. **提出 WorldAbduct 溯因驱动的世界模型演化框架**：通过状态一致性、世界-观测差距、策略门正确性、目标判定四个互补视角诊断执行轨迹，生成假设后由专属溯因智能体在独立沙盒中进行针对性交互验证，仅当证据充分时才将规则写入静态世界层——与已有工作的本质区别在于引入"主动假设检验"环节，避免直接从不完整的轨迹归纳经验。
3. **在 WOW 基准上显著优于现有演化方法**：在四种 LLM backbone 下均取得最高安全任务完成率（STCR），较最强基线 MemoHarness 提升 5–15 个百分点，同时在 ScienceWorld 和 DiscoveryWorld 验证了方法的泛化性。

## 方法详解

**整体架构（E-LEDGER）**：由四个角色构成闭环——Task Agent 提出动作与状态预测；Policy-Context Agent 提取策略上下文绑定；Code Approval Layer 将文本策略编译为可执行代码进行前置审批（四态决策：execute/query/probe/block）；State-Tracking Agent 从工具响应中提取任务相关状态增量，更新动态 world state $\mathcal{W}_D$。

**World Ledger 双层结构**：
- 静态世界知识 $\mathcal{W}_S$：存储隐藏规则 $r : (a, s, c) \longrightarrow \Delta s$，包括工具响应中不可见的跨记录联动、延迟副作用等。
- 动态世界状态 $\mathcal{W}_D$：以 schema-aligned 的全局结构存储任务相关实体属性与关系，逐交互增量更新，保留证据支撑。

**Task Agent 动作选择（公式 3）**：
$$
(a_t, \hat{\Delta s}_t) = \mathcal{A}_\theta(U, P, A, \mathcal{W}_S, \mathcal{W}_D^t, h_t)
$$
在输入中显式预测状态变化 $\hat{\Delta s}_t$，用于后续与观测比对以驱动演化。

**Approval Code Gate 四态输出**：
- execute：通过审批，执行；
- query/probe：证据不足，需要补充信息后再评估；
- block：违反政策，必须替换动作；
该审批为不可绕过（non-bypassable）的确定性代码执行。

**动态状态更新（公式 4）**：
$$
\mathcal{W}_D^{t+1} = \mathcal{R}_\theta(\mathcal{W}_D^t, a_t, o_{t+1}, \mathcal{W}_S)
$$
可利用已验证的静态规则解释间接效应，冲突和未决字段记录以供后续验证。

**WorldAbduct 四视角诊断（公式 5–9）**：
1. **State Consistency**（公式 5）：评估 $\hat{\Delta s}_t$ 与观测 $o_{t+1}$ 及 $\mathcal{W}_D^{t+1}$ 的一致性，聚焦可直接观测部分；
2. **World Observation Gap**（公式 6）：分析完整轨迹，识别工具响应中未出现但可能影响策略决策的隐藏状态变迁，生成 hidden-rule 假设；
3. **Policy-Gate Correctness**（公式 7）：审计未通过审批的动作及其上下文，诊断策略门失败原因并提出 prompt 或规则改进；
4. **Goal Judgment**（公式 8）：基于轨迹和评估反馈诊断提前终止、目标未达成或未发现的违规。

四视角输出分为两类：直接修改 harness 组件的 $\Delta H_{\setminus \mathcal{W}_S}$，以及需验证的 hidden-rule 假设 $h$。

**Hidden-Rule Abduction Agent（公式 10）**：
$$
B_\phi(U, P, A, \tau, h, \tau^{\mathrm{val}}) \longrightarrow (d, \mathrm{payload}), \quad d \in \{\text{act, revise, verified, unresolved}\}
$$
在新沙盒中自适应设计验证动作序列，可接受（verified）、修订（revise）、继续探测（act）或判定不可解决（unresolved）；仅 verified 规则进入 $\mathcal{W}_S$。

## 实验与结果

**数据集**：
- 主基准：**World of Workflows (WOW)**（Gupta et al., 2026），50 个企业任务，含 10 种类型×5 变体，20/10/20 训练/验证/测试划分；
- 泛化基准：**ScienceWorld**（Wang et al., 2022，150 任务）和 **DiscoveryWorld**（Jansen et al., 2024，40 episodes）。

**评估指标**：STCR（安全任务完成率）、TCR（任务完成率）、PSR（策略安全率）、交互步数、测试时推理成本（USD）。

**基线**：ReAct（无 harness）、MemoHarness（6 维 prompt 演化）、WorldEvolver（基于可观测轨迹的 world model 库）。

**WOW 主要结果（Table 1）**：
| Backbone | 方法 | STCR | TCR | PSR |
|---|---|---|---|---|
| GPT-5.6-Terra | WorldAbduct | **65.0** | 65.0 | 100.0 |
| Kimi-K3 | WorldAbduct | **70.0** | 75.0 | 95.0 |
| GLM-5.3-Flash | WorldAbduct | **75.0** | 80.0 | 95.0 |
| DeepSeek-V4-Pro | WorldAbduct | **80.0** | 85.0 | 95.0 |

- 相比最强演化基线 MemoHarness，WorldAbduct 在四种 backbone 上 STCR 均领先 **5–15 个百分点**；
- DeepSeek-V4-Pro 上 ReAct 的 TCR=50%，E-LEDGER 保持相似 TCR 但 STCR 从 25→50（翻倍）；
- GPT-5.6-Terra 较保守，E-LEDGER 维持 100% PSR 但 TCR 提升有限。

**消融实验（GLM-5.3-Flash，Table 2）**：
- w/o $\mathcal{W}_S$：STCR 从 75 降至 40（最显著下降）；
- w/o $\mathcal{W}_D$：STCR 降至 65；
- w/o Code Approval：STCR 降至 70，PSR 从 95 降至 90。

**演化视角消融（Table 5）**：
- Only Goal Judgment：STCR=65，但 PSR 降至 80（引入不安全动作）；
- Only World Observation Gap：PSR=95，STCR=65；
- Without Abduction：STCR=65，PSR=85（未经验证的假设导致过窄/过宽的规则）；
- Full WorldAbduct：STCR=**75**，PSR=**95**，最优综合。

**隐藏规则质量（Figure 3b + Appendix C）**：
- 演化后的 $\mathcal{W}_S$ 在 50 个动作序列的 policy-violation localization 任务上，各 backbone 均提升 0.10–0.22 绝对值；
- 人工审计 4 个 backbone 的 54 条冻结规则：49 条完全正确，5 条部分正确（机制对但适用条件偏窄/宽），**无错误规则**；
- Abduction 共发起 64 次检查，50 次直接接受，8 次修订后接受，6 次未解决；平均每次检查 13.3–19.1 步交互。

**ScienceWorld / DiscoveryWorld（Table 3）**：
- ScienceWorld：WorldAbduct Success=**71.7**（最优），Score=**75.5**（最优）；
- DiscoveryWorld：WorldAbduct Success=**93.8**（最优），Steps=**22.7**（最少）。

## 相关工作脉络
1. **WorkArena (Drouin et al., 2024)**：评估 ServiceNow 知识库任务上的 web agent 能力，但缺少跨记录隐藏联动；WOW 在此基础上显式建模了部分可观测的企业工作流挑战。
2. **MemoHarness (Huang et al., 2026)**：从失败轨迹蒸馏全局模式以优化 6 个 harness 组件；定位差异——MemoHarness 不显式建模隐藏规则，也不进行主动假设检验，易在高策略约束场景下产生 unsafe completion。
3. **WorldEvolver (Zhang et al., 2026)**：将 evolution 视为基于可观测轨迹的动作级状态变化库；定位差异——WorldEvolver 仅记录工具反馈中可见的变更，无法发现工具响应外的隐藏副作用。
4. **Metaharness (Lee et al., 2026)**：端到端优化 harness 结构；定位差异——本文方法与结构优化正交，聚焦于 world knowledge 的主动演化。
5. **Test-Time Harness Evolution (TTHE, Nie et al., 2026)**：在线修正当前任务的控制策略；定位差异——TTHE 侧重单任务适应，WorldAbduct 将验证过的规则沉淀为可跨任务复用的静态世界知识。
6. **Reflexion (Shinn et al., 2023) / TextGrad (Yuksekgonul et al., 2024)**：使用自然语言反馈优化系统；定位差异——本文采用"四视角诊断 + 主动环境验证"的混合方式，而非纯文本梯度/反馈。

## 局限性与未来方向
1. **实验为单次运行**：因交互式环境演化与评估成本较高，各配置仅报告单次结果（Appendix G），统计方差与重复稳定性未充分讨论。
2. **演化成本随任务规模增长**：WorldAbduct 在演化阶段额外发起 11–23 次 abductive 验证，每次平均 13–19 步交互，对长 horizon 任务的总成本仍有优化空间。
3. **未覆盖的隐式规则仍存在**：部分 partially correct 规则适用条件偏窄或过宽（如遗漏 manager-reassignment 例外），说明自动发现的规则粒度控制仍需改进。
4. **未来方向**：可扩展到更多企业平台（非仅 ServiceNow）；可探索更高效的 rule pruning 机制；可与测试时适应（test-time adaptation）方法结合以提升单次任务弹性。

## 研究启发与可借鉴点
1. **"四视角诊断"设计值得迁移**：状态一致性 + 观测差距 + 策略门 + 目标判定的组合覆盖了"可见世界/不可见规则/执行合规/任务完成"四个层次，可复用于其他需要持久状态和策略约束的 agent 场景（如自动化客服、数据管道编排）。
2. **溯因验证作为"假设过滤器"的价值**：直接从不完整轨迹归纳规则易引入噪声甚至错误依赖；引入独立沙盒的主动验证环节显著提升规则质量（0 条错误规则，审计通过率极高），这一"诊断→假设→验证"三段式流程具有通用方法论意义。
3. **Code-as-Gate 思想**：将文本策略编译为可执行代码并由独立策略上下文 agent 绑定动态证据，避免模型"自行推理绕过约束"——对任何需要严格合规保证的 agent 部署（金融、医疗、政务）均有参考价值。
4. **与团队方向的结合机会**：团队在长期 multi-step reasoning 和工具使用方面已有积累，可将 E-LEDGER 的 world ledger 结构与团队现有的 memory/knowledge 模块结合；同时，abductive validation 的思想也可迁移到科学发现类任务（与 ScienceWorld/DiscoveryWorld 高度契合）。

## 关键术语表
**E-LEDGER**：本文提出的多智能体安全持久 harness，由 Task Agent、Policy-Context Agent、Code Approval Layer 和 State-Tracking Agent 构成闭环。
**WorldAbduct**：基于世界模型的溯因演化方法，通过四视角诊断轨迹并主动验证隐藏规则假设。
**World Ledger**：双层知识层，静态层 $\mathcal{W}_S$ 存储已验证隐藏规则，动态层 $\mathcal{W}_D$ 存储证据支撑的任务相关状态。
**Code Approval Layer**：将文本策略编译为确定性可执行代码的前置审批网关，输出 execute/query/probe/block 四态决策。
**STCR (Safe Task Completion Rate)**：安全任务完成率，同时要求任务成功完成且全程无策略违规。
**World Observation Gap**：演化视角之一，识别工具响应中未出现但可能影响策略决策的隐藏状态变迁。
**Abduction Agent**：负责设计验证动作序列以检验 hidden-rule 假设的智能体，输出 act/revise/verified/unresolved。
**WOW (World of Workflows)**：本文主基准，模拟 ServiceNow 企业环境的有隐藏联动的部分可观测工作流基准。

## 可复现要素
- **数据集**：WOW（公开）、ScienceWorld（公开）、DiscoveryWorld（公开）；划分方式 2:1:2（训练/验证/测试），论文提供了详细 split 说明。
- **代码**：开源，地址 https://github.com/HKUST-KnowComp/E-LEDGER-WorldAbduct。
- **模型/权重**：实验使用 GPT-5.6-Terra、Kimi-K3、GLM-5.3-Flash、DeepSeek-V4-Pro 四种商业 backbone，论文未提供自训练权重。
- **关键超参**：任务 horizon 50 步（WOW/ScienceWorld）、100 步（DiscoveryWorld）；演化轮数最多 3 轮；abduction 每假设动作预算与 horizon 一致（50/100）；修订预算 3 次；演化过程中最多 3 轮迭代（其余细节见 Appendix A.2）。
