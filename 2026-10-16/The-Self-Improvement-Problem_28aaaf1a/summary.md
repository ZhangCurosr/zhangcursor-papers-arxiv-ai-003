---
title: "The-Self-Improvement-Problem"
source: https://arxiv.org/pdf/2610.21646v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-04 10:32:22"
field: "AI 安全与可计算性理论交叉"
keywords: ["self-improvement", "computability theory", "evaluator hacking", "reward hacking", "agent safety", "formal verification", "Kleene recursion theorem"]
innovations: ["证明验证门控自改进在自指设定下不存在同时可靠且非保守的终止门控", "在有界评估与有限提案预算下恢复可判定性", "揭示有界域内多轮更新可导致真实评分漂移的结构性现象"]
---

# 论文速读：The-Self-Improvement-Problem

## 一句话总结
本文形式化了验证门控自改进系统中"评估器欺骗"的结构性困境：当提案机制可自适应于门控决策时，任何总可计算门控（certifier 或 searcher）无法同时保证可靠性（不接纳非改进）与非保守性（不错失可用改进）；但在有界评估与有限提案预算的可判定子情形下，穷举搜索可恢复相关保证。

## 研究问题与动机
- **核心问题**：自治 Agent 系统通过自动化门控（reward model、测试套件、自动审稿人）决定是否接受自身提出的更新，但"语义改进"依赖于可能无界的程序行为，导致门控评估与真实改进之间存在结构性不匹配。
- **现有方法不足**：实践中常见的 reward model 或自动化审稿人作为终止可计算过程，无法桥接代理评估与语义改进之间的鸿沟；提案者若能预测/适应门控，便会"优化评估器"而非真实目标，产生 evaluator hacking。
- **自指构造的普遍威胁**：无论是确定性门控、随机化门控（有界误差）、隐藏门控还是状态门控，自指对角构造均可强制其失效。
- **有界可恢复性**：实际系统普遍使用有界评估与有限预算，本文明确刻画了这种限制下可恢复保证的精确边界及其剩余盲区（proxy drift）。

## 核心贡献（创新点）
1. **验证门控自改进的最小化形式化**：将 Agent 编码为程序索引、提案生成建模为可计算枚举器、门控建模为必须终止的总可计算过程，使得可靠性与进步性保证可精确表述——与已有工作相比，本文的novelty在于将其转化为面向 ML 的形式语言并明确分离不可能与可恢复边界。
2. **自指驱动的不可能性边界（Theorem 1 & 2）**：在完全一般（无界）设定下，证明不存在同时满足可靠性与非保守性的总可计算 certifier（Theorem 1）或 proposal-respecting total searcher（Theorem 2）——与 Rice 定理相比，本文提供了构造性对角论证，刻画了"被拒绝则好、被接受则坏"的对抗性候选的精确机制。
3. **随机化门控不可逃逸（Proposition 1）**：有界误差的随机门控可通过 König 引理推导随机化为确定性门控，从而同样受 Theorem 1 约束——本质区别在于此前文献通常将随机化视为增强门控能力的途径。
4. **有界条件下的可判定性恢复（Theorem 3）**：在固定时间界 T、提案预算 B 与生成界 $T_E$ 下，穷举模拟给出一个终止 searcher，相对于有界评分 $J_T$ 满足 soundness 与弱完备性——这是对实践常用限制的精确刻画。
5. **有界代理漂移构造（Proposition 2）**：即使在不涉及自指的有界可判定区域内，一系列被接受的更新仍可每轮提升有界分而降低真实分——揭示了有界恢复的内在局限，此前 Goodhart 文献仅从统计代理偏差角度讨论此现象。

## 方法详解
- **Agent 模型**：Agent 编码为程序索引 $p \in \mathbb{N}$（Gödel 数），其行为由偏可计算函数 $\varphi_p$ 描述。更新为另一索引 $q$。
- **语义评分泛函**：$J: \mathbb{N} \to S$（如 $\{0,1\}$ 或 $\mathbb{R}$），可一般不可计算；扩展性要求 $J(p)$ 仅依赖 $\varphi_p$ 而非程序文本。
- **Certifier**：总可计算函数 $V: \mathbb{N} \times \mathbb{N} \to \{0,1\}$，$V(p,q)=1$ 表示批准从 $p$ 更新到 $q$。
- **严格改进定义**：$q$ 是 $p$ 的严格改进当且仅当 $J(q) > J(p)$。
- **Soundness**：$V(p,q)=1 \Rightarrow J(q) > J(p)$（无错误接纳）。
- **Completeness**：$J(q) > J(p) \Rightarrow V(p,q)=1$（不错失改进）。
- **Theorem 1 证明核心**：取简单评分 $J(p)=1$ 当且仅当 $\varphi_p(0)$ 停机且输出 1，否则 0。构造 $f(e)$ 使程序 $f(e)$ 在输入 0 上先计算 $V(p,e)$，若为 1 则输出 0，否则输出 1。由 Kleene 递归定理，存在 $q$ 使 $\varphi_q = \varphi_{f(q)}$，产生对角矛盾。
- **Searcher 形式化**：$E_e$ 为提案枚举器，$S(p,e)$ 为 total searcher，要求 proposal-respecting：若 $S(p,e)=q \neq \bot$ 则 $q=E_e(p,i)$ 对某 $i$。Theorem 2 用类似对角构造证明即使如此也无法同时满足 soundness 与弱完备性。
- **有界评分**：$J_T(p)=1$ 当且仅当 $\varphi_p(0)$ 在 $T$ 步内停机并输出 1。
- **Theorem 3  searcher 构造**：模拟 $\varphi_p(0)$ 至多 $T$ 步得 $J_T(p)$；依次模拟 $E_e(p,i)$ 至多 $T_E$ 步（$i<B$），对每个停机的候选再模拟至多 $T$ 步计算 $J_T(q_i)$，返回首个使 $J_T$ 提升的 $q_i$。

## 实验与结果
- 本文属理论形式化论文，无传统意义上的数据集与基准实验。
- **主要结果数字**：Theorem 1–2 证明在无界一般设定下 impossibility；Theorem 3 在有界设定（$T, B, T_E$）下证明存在 sound + weakly complete searcher；Proposition 2 构造了 $n=2k$ 任务下有界分从 0 升至 $k$ 而真实分从 $2k$ 降至 $k$ 的漂移链。
- **最强保证**：Theorem 3 给出的 exhaustive searcher $S_{T,B,T_E}$ 在有界域内是 sound 且弱完备的，这是本文唯一的正向存在性结果。
- **漂移幅度**（Proposition 2）：经过 $k$ 轮更新，有界分提升 $k$ 单位，真实分下降 $k$ 单位，相对初始值各变化 $50\%$（从 $2k$ 到 $k$）。
- Lean 4 形式化验证已开源：https://github.com/Brian-Moser/SelfImprovementProof

## 相关工作脉络
1. **Rice 定理（Rice, 1953）**：本文 Theorem 1 在关系形式下的推广；不同之处在于本文提供构造性对角论证并应用于现代 ML 评估管线语境。
2. **Gödel Machine（Schmidhuber, 2003）**：proof-carrying gate 的经典设计，本文定位其为 sound 但 inherently incomplete（错过无限多真实改进），给出其保守性的精确计算理论解释。
3. **Reward Hacking / Goodhart 效应（Amodei et al., 2016; Manheim & Garrabrant, 2018; Skalse et al., 2022; Pan et al., 2022）**：统计视角下代理偏差的实证与概念研究；本文从结构/计算理论角度补充，指出即使完美指定的代理也无法避免自指导致的失效。
4. **自我修改智能体（Everitt et al., 2016）**：决策论视角研究理性 agent 是否会保留或颠覆其目标；本文关注计算视角（评估本身是否可终止判定），两者互补。
5. **Löbian obstacle / Tiling Agents / Vingean Reflection（Soares & Fallenstein, 2015; Yudkowsky & Herreshof, 2013）**：self-referential agent 推理的一致性障碍；本文将此类计算理论现象转化为 ML 自动化管线语言。
6. **AI Scientist 管线（Lu et al., 2024; Yamada et al., 2025）**：端到端自动科研管线匹配本文模型；本文划定此类系统在高自治度下的结构性极限。

## 局限性与未来方向
- **单步分析局限**：Theorems 1–3 针对单次更新步骤；Proposition 2 开启多轮分析但仅给出存在性构造，未刻画漂移速率的上界。
- **未覆盖中间机制**：作者承认在不可判定全域与完全有界域之间存在大量中间限制（如 proposal 无法条件化于 gate、更新语言受限、gate 拥有更多计算资源等），哪些恢复哪些保证尚未系统映射。
- **自适应漂移率**：Proposition 2 后的开放问题——当提案者适应重评估调度时，漂移以多快速率增长、何种额外约束可在保留有用进步的同时控制漂移。
- **实值评分的扩展**：当前形式化主要使用 $\{0,1\}$ 二元评分；扩展到实值奖励时需处理可计算实数相等性不可判定的额外困难（Appendix B 仅在有理值假设下扩展）。

## 研究启发与可借鉴点
1. **有界性作为设计原则**：自动改进管线应显式声明时间预算、提案预算与生成预算，而非追求"通用正确门控"；这一设计洞察可直接迁移至 Agentic Workflow 与自动实验管线的架构决策。
2. **周期性重评估策略**：Proposition 2 的 drift 分析支持"固定每轮预算保速度、定期扩大预算重评估"的设计模式，可用于长期运行的 agent loop。
3. **对角论证的迁移应用**：本文的 Kleene 递归定理构造范式可复用于分析其他自指场景（如 multi-agent pipeline 的安全性证明、autoformalization 中的 verifier 欺骗），Appendix A 已示范向多智能体的推广。
4. **One-sided guarantee 实用化**：proof-carrying update 或接受仅当近似certify足够大gap（$\geq$ margin）的设计，是将 soundness 优先于 completeness 的实用策略，适合安全关键场景。
5. **Formal verification 辅助**：本文配套 Lean 4 形式化验证为后续工作提供了可复用的证明基础设施模板，可应用于更复杂的 agent 形式化研究。

## 关键术语表
**Semantic Score Functional ($J$)**：衡量 Agent 真实性能的函数，可能不可计算，仅依赖于程序的扩展语义（输出行为）而非源码文本。
**Certifier ($V$)**：总可计算的门控过程，接收当前 agent 与候选更新，输出 1（接受）或 0（拒绝），必须在有限步内终止。
**Proposal Enumerator ($E_e$)**：总可计算函数，给定当前 agent 索引和提案序号 $i$，输出第 $i$ 个候选更新。
**Soundness**：门控的可靠性保证——若接受某更新，则该更新确为严格改进（$J(q) > J(p)$）。
**Weak Completeness**：门控的非保守性保证——若候选枚举器中存在至少一个改进，则门控不会返回 $\bot$（能找到一个）。
**Kleene's Recursion Theorem**：对任意总可计算函数 $f$，存在索引 $e$ 使 $\varphi_e = \varphi_{f(e)}$；保障自指程序的构造可行性。
**Bounded-Horizon Score ($J_T$)**：在 $T$ 步内模拟程序行为所得的可计算代理评分，用于有界可判定情形。
**Proxy Drift**：在反复接受有界评分改进的更新后，真实语义评分持续下降的现象（Proposition 2）。

## 可复现要素
- **代码/形式化**：Lean 4 形式化证明已开源于 https://github.com/Brian-Moser/SelfImprovementProof（论文声明）。
- **数据集**：本文为纯理论论文，无数据集；Proposition 2 的构造性证明可通过直接编码复现。
- **关键超参**：时间界 $T$、提案预算 $B$、生成界 $T_E$——均为参数化变量，非固定数值；具体实验数值论文未提供（理论工作）。
- **模型/权重**：不涉及。
