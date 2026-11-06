---
title: "Verifiable-LLM-Guided-Focal-SMT-Solving-for-Quantified-Array"
source: https://arxiv.org/pdf/2610.39435v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-06 17:14:14"
field: "SMT求解与形式化验证"
keywords: ["SMT求解", "量化数组", "大语言模型", "形式验证", "量词实例化", "混合推理"]
innovations: ["提出语义焦点集（IFS/CFS）作为LLM与SMT求解器之间的可验证窄接口", "双分支求解：UNSAT用超近似实例化、SAT用下近似约束强化并配蕴含检查", "反馈驱动的迭代精炼与优雅回退机制，保留基线覆盖度"]
benchmarks: ["SMT-LIB / SMT-COMP 2025", "ABV", "ABVFP", "ABVFPLRA", "ALIA", "ANIA"]
---

# 论文速读：Verifiable-LLM-Guided-Focal-SMT-Solving-for-Quantified-Array

## 一句话总结
论文提出 LINGUAARRAY 框架，利用大语言模型（LLM）识别量化数组 SMT 公式中的"语义焦点集"（Semantic Focus Sets），并通过 SMT 求解器进行严格验证，使 Z3/CVC5 的求解成功率从约 32% 提升至约 70%。

## 研究问题与动机
- 量化数组公式（结合量词与数组理论）在现代 SMT 求解中极具挑战：关键实例化项常被辅助变量、公理守卫或嵌套数组操作遮蔽，导致基于触发器的 E-matching 和 MBQI 等实例化技术失败。
- 现有方法依赖固定语法触发或人工工程难以系统性恢复缺失的语义洞察；这类关键"焦点"往往只由少量决定性项或约束驱动。
- LLM 已能在结构化符号输入中识别语义模式，但其输出不可作为形式证明直接使用。
- 如何在保留形式可验证性的前提下，让 LLM 为求解器提供安全、可验证的语义引导？

## 核心贡献（创新点）
- **提出"语义焦点集"接口**：将 LLM 产出限制为实例化焦点集（IFS）或约束焦点集（CFS），两者均由求解器严格判定，保证单调性与可证性。
- **双分支求解架构**：UNSAT 分支通过 IFS 对全称量词做有界实例化实现超近似；SAT 分支通过 CFS 做附加假设强化实现下近似；强化后的 UNSAT 还需额外的蕴含检查以避免伪反例。
- **反馈驱动迭代与优雅回退**：在有限预算 R 内循环迭代，从失败求解中提取反馈（未约束守卫、任意赋值索引等），引导 LLM 在下轮提出新建议；预算耗尽时回退到基线求解器 $S(F)$，防止覆盖度退化。

## 方法详解
- **预处理**：将输入 SMT-LIB 公式化为否定范式（NNF）以显式量词极性，并做常量传播、平凡子公式消除与可逆标识符重命名 $F'$。
- **语义焦点集定义**：
  - **IFS**（Instantiation Focus Set）：对 NNF 中正出现的全称子公式 $\forall \vec{x}.\varphi(\vec{x})$，给出有限基项集合 $\{t\}$，导出有界合取实例 $\bigwedge_{\vec{t}} \varphi[\vec{x}\mapsto\vec{t}]$。
  - **CFS**（Constraint Focus Set）：由若干基原子公式 $\{\psi_i\}$ 组成，作为附加假设加入，缩小模型搜索空间。
- **两条支路的正确性保障**：
  - **UNSAT 分支（Theorem 1）**：$F' \Rightarrow F_{IFS}^{\downarrow}$，若 $S(F_{IFS}^{\downarrow}) = \mathsf{unsat}$ 则 $F$ 不可满足。
  - **SAT 分支（Theorem 2）**：$F_{CFS}^{\uparrow} = F' \wedge \bigwedge c$，若 $S(F_{CFS}^{\uparrow}) = \mathsf{sat}$ 则 $F$ 可满足。
  - **强化后 UNSAT 的验证**：若 $F_{CFS}^{\uparrow}$ 被判 UNSAT，需对每个 $c \in CFS$ 检查 $S(F' \wedge \neg c) = \mathsf{unsat}$ 以确保 $F' \Rightarrow c$，否则不能直接传递 UNSAT；若某次检查为 SAT，可立即得到一个 $F'$ 的模型。
- **迭代流程与 LLM 交互**：每轮 LLM 根据当前累积反馈 $fb$ 生成 IFS/CFS；无效项（类型不符、引入签名外符号等）被丢弃而非改写；最终在剩余时间内调用 $S(F)$ 兜底。

## 实验与结果
- **数据集**：从 SMT-COMP 2025 / SMT-LIB 抽取 1015 个量化数组实例，覆盖 5 个逻辑：ABV(400)、ABVFP(60)、ABVFPLRA(77)、ALIA(400)、ANIA(78)。
- **评估设置**：端到端超时 1200s（含 LLM 延迟）；后端求解器 Z3 v4.15.0 与 CVC5 v1.3.2；LLM 使用 DeepSeek-V3.2 与 Gemini-3-Flash-Preview（temperature=0）；默认迭代预算 R=10。
- **主要结果**：
  - LINGUAARRAY(Z3)：解决 701/1015（69.06%），相比 Z3 基线 141/1015（13.89%）提升 **397%**。
  - LINGUAARRAY(CVC5)：解决 659/1015（64.93%），相比 CVC5 基线 321/1015（31.63%）提升 **105%**。
  - 超越 SMT-COMP 2025 组态 CVC5-COMP（38.23%）与虚拟最佳组合 Z3∪CVC5（38.4%）。
  - LLM 互补性：DeepSeek 与 Gemini 的虚拟最佳分别在 Z3/CVC5 上多解 74/80 题。
  - 消融：harness-only（空焦点集）与随机提议者分别仅解 143/436 和 322/445，显著低于 LLM 语义引导。
  - ABVFP/ABVFPLRA 中 Z3 内部错误，CVC5 仍显著增益。
- **效率**：LINGUAARRAY 解决实例平均耗时 ~76s，虽高于基线，但整体在 1200s 窗口内完成更多求解；1 次迭代即可将 Z3 成功率从 13.9% 提至 50.1%，3 次迭代后收益显著下降。

## 相关工作脉络
- **CEGQI / SyQI / QSMA**：基于模型反例或抽象细化引导实例化；LINGUAARRAY 不替代这些机制，而是以 LLM 提出的焦点集在其之外额外注入语义线索。
- **Learning for SMT（ML 特征+GBDT/LightGBM）**：从公式特征学习触发/实例化排序；本文不训练模型，直接使用现成 LLM 作为零样本/少样本提示生成器。
- **AquaForte（LLM+UFX）**：LLM 合成完整函数定义用于验证，主要面向 SAT；本文提出更局部的焦点集（IFS/CFS）并支持 UNSAT/SAT 双向验证。
- **Array Property Fragment**：通过限制量词形式获得可判定性与有限实例化完备性；本文面向该片段之外的一般量化数组公式。
- **E-matching / MBQI**：传统求解器的核心实例化机制；本文在其之上叠加一层 LLM 语义引导，仍复用并增强而非替换这些机制。

## 局限性与未来方向
- 当前仅针对量化数组理论，扩展到更广泛的量词理论（含未解释函数、更复杂理论组合）需要统一化焦点集规范。
- 性能高度依赖 LLM 质量与推理速度；强推理模型（如 DeepSeek Reasoner）输出长但延迟高，导致可迭代轮数受限。
- 对极端复杂的理论交互场景，公式本身复杂度成为瓶颈，单纯焦点引导的提升空间有限。
- 重复提案比例较高（CFS 约 59.5%，IFS 约 65.1% 跨轮重复），说明探索多样性仍待改善。
- 未提供统计显著性检验/误差条，结果稳定性仅在少量重复实验中给出。

## 研究启发与可借鉴点
- **"可验证窄接口"设计范式**：LLM 仅输出求解器可直接消费的结构性对象（项/约束），所有决定由符号引擎裁定——这一模式可迁移到其他需要形式保证的 LLM+符号混合系统。
- **超近似/下近似双分支思路**：UNSAT 用有界实例化的超近似、SAT 用附加假设的下近似，并配合蕴含检查消除伪 UNSAT，这种对称结构可复用于其他量词消去/实例化场景。
- **反馈驱动精炼循环**：从求解器模型中提取"未约束守卫/任意赋值位置"作为自然语言化反馈，引导 LLM 下一轮聚焦，机制通用且实现轻量。
- **虚拟最佳（virtual-best）评测与融合潜力**：不同 LLM 在不同子集上互补，提示可构建轻量多模型集成或路由策略以获得更高覆盖。
- **实验设计亮点**：设置 harness-only 与随机提议者两个非 LLM 对照，分离出"语义引导"相对于"接口与预算"本身的贡献，论证更严谨。

## 关键术语表
- **SMT（Satisfiability Modulo Theories）**：在命题可满足性基础上加入背景理论（如数组、算术）的一阶逻辑可满足性判定。
- **量化数组（Quantified Arrays）**：含全称/存在量词并对数组进行跨索引断言的逻辑片段，广泛用于程序不变量与内存安全建模。
- **Semantic Focus Set**：LLM 提出的、由求解器验证的有限"焦点"集合，分为 IFS（实例化候选）与 CFS（约束假设）。
- **IFS（Instantiation Focus Set）**：对全称量词给出有限基项集合，构造超近似 $F_{IFS}^{\downarrow}$ 用于 UNSAT 证明。
- **CFS（Constraint Focus Set）**：附加的基原子约束集合，构造强化公式 $F_{CFS}^{\uparrow}$ 用于 SAT 证明及蕴含检查。
- **E-matching**：基于触发项的等价图匹配技术，用于量词实例化。
- **MBQI（Model-Based Quantifier Instantiation）**：基于当前候选模型的反例提取来自动生成实例。
- **Virtual-best portfolio**：将多个求解器/配置所解实例取并集的理想上限基准，用于衡量方法的上界潜力。

## 可复现要素
- **数据集**：SMT-LIB / SMT-COMP 2025 采样；论文使用随机种子 42 抽样，代码与实例列表已开源。
- **代码**：MIT 协议公开于 https://github.com/kylelv2000/LinguaArray（含 prompt 模板、求解器配置与脚本）。
- **LLM**：DeepSeek-V3.2、Gemini-3-Flash-Preview；temperature=0；模型为黑盒调用。
- **求解器配置**：Z3 使用默认配置；CVC5 基线为 `--finite-model-find --arrays-exp --fp-exp --mbqi --enum-inst --no-cegqi --no-sygus-inst`；COMP 配置为 `-L smt2.6 --no-incremental --no-type-checking --no-interactive --fp-exp --use-portfolio`（详见附录 B）。
- **超时与迭代**：端到端 1200s，默认 R=10 轮；单轮超时控制未在正文给出细粒度切分，需参考仓库。
- **稳定性补充实验**：以 DeepSeek-V4-Flash 做 3 次独立运行，均值 689.7±4.0，602 题稳定解决。
