---
title: "Verifiable-LLM-Guided-Focal-SMT-Solving-for-Quantified-Array"
source: https://arxiv.org/pdf/2610.39435v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-06 17:14:11"
field: "SMT与形式化验证中的LLM引导求解"
keywords: ["SMT solving", "quantified arrays", "large language models", "formal verification", "quantifier instantiation", "hybrid reasoning"]
innovations: ["提出Semantic Focus Sets作为LLM与SMT求解器的可验证窄接口", "设计双分支架构（IFS过近似证UNSAT、CFS欠近似证SAT）并辅以蕴含认证", "迭代反馈驱动精炼与预算耗尽回退基线的工程化机制"]
benchmarks: ["SMT-LIB quantified array instances", "SMT-COMP 2025 benchmarks"]
---

# 论文速读：Verifiable-LLM-Guided-Focal-SMT-Solving-for-Quantified-Array

## 一句话总结
论文提出 LINGUAARRAY 框架，利用大语言模型（LLM）为量化数组 SMT 公式生成语义焦点集（实例化焦点集 IFS 与约束焦点集 CFS），由传统 SMT 求解器独立完成所有正确性验证；在保持形式化完备性的同时，求解成功率从不到 32% 提升至约 70%，最大提升幅度达 397%。

## 研究问题与动机
- **量化数组 SMT 求解瓶颈**：现代求解器依赖基于触发器的 E-matching 和 MBQI，当关键 witness 项被辅助变量、嵌套数组操作或蕴含前件遮蔽时，难以高效定位实例化候选项，导致超时或放弃。
- **传统规则引擎缺乏语义洞察**：仅依靠固定语法触发器与硬编码规则无法系统性地捕捉真实公式中的深层语义关联，需要引入外部语义引导。
- **LLM 输出不可直接信任**：LLM 能在形式化工件中识别有用语义结构，但其输出不能直接作为证明或求解器决策依据，需限定在可验证接口内使用。
- **焦点现象类比 SAT 后门集**：作者观察到许多量化数组实例主要由少量决定性 ground term 或约束驱动，类似于 SAT 中的 backdoor set，暗示可通过 LLM 聚焦关键子结构加速求解。

## 核心贡献（创新点）
1. **提出 Semantic Focus Sets 作为 LLM-求解器可验证接口**：通过 IFS（对全称量词的过近似）和 CFS（对模型的子空间约束）限定 LLM 影响范围，所有结论均由后端求解器独立验证，保真 soundness。
2. **设计双分支推理架构**：UNSAT 分支通过 IFS 生成有限实例化合取，SAT 分支通过 CFS 施加辅助假设以指导模型构造；针对 CFS 强化后产生的虚假 UNSAT，引入显式蕴含检查进行认证。
3. **实现迭代反馈驱动的 refine 机制**：每轮根据求解器反馈（数组索引任意赋值、未约束 guard 等）引导 LLM 追加或替换焦点集，预算耗尽时回退到原始公式由基线求解器处理，保证覆盖率不降。
4. **在 SMT-LIB 五个量化数组逻辑上验证显著增益**：LINGUAARRAY(Z3) 解决率 69.06%，相对 Z3 基线提升 397%；LINGUAARRAY(CVC5) 解决率 64.93%，相对 CVC5 提升 105%，且结果跨 LLM、跨轮次保持稳定。

## 方法详解
- **预处理**：将 SMT-LIB 输入解析为否定范式（NNF）使量词极性显式化；执行常量传播、平凡子公式消除，并通过可逆映射 $\Sigma$ 重命名冗长标识符，得到等可满足的简洁公式 $F'$ 供 LLM 解析。
- **Semantic Focus Sets 定义**：
  - **IFS（Instantiation Focus Set）**：对 NNF 中正出现的全称量词 $\forall\vec{x}.\varphi(\vec{x})$，LLM 提供一个有限 ground 项集合 $\{\vec{t}\}$，导出过近似 $F_{IFS}^{\downarrow} = \bigwedge_{\vec{t}} \varphi[\vec{x}\mapsto\vec{t}]$。定理 1 保证 $F' \Rightarrow F_{IFS}^{\downarrow}$，故若 $S(F_{IFS}^{\downarrow})=\text{unsat}$ 则原公式亦 UNSAT。
  - **CFS（Constraint Focus Set）**：LLM 提供一组 quantifier-free ground 公式 $\{\psi_i\}$，构造强化公式 $F_{CFS}^{\uparrow} = F' \wedge \bigwedge_i \psi_i$。定理 2 保证 $F_{CFS}^{\uparrow} \Rightarrow F'$，故若 $S(F_{CFS}^{\uparrow})=\text{sat}$ 则原公式亦 SAT。
- **CFS 虚假 UNSAT 认证**：当 $S(F_{CFS}^{\uparrow})=\text{unsat}$ 时，对每个 $c\in\text{CFS}$ 检查 $S(F'\wedge\neg c)=\text{unsat}$；全部通过则 $F'\Rightarrow c$ 成立，UNSAT 结论转移至原公式；任一检查返回 SAT 则直接获得 $F'$ 的满足模型。
- **反馈驱动迭代**：第 $r$ 轮 LLM 以累积反馈状态 $fb$ 为条件生成新 IFS/CFS。UNSAT 分支在返回 SAT 时提取模型中任意赋值索引与未约束 guard 作为反馈；SAT 分支在超时或认证失败时将失败约束加入反馈。
- **LLM 交互协议**：每次查询含任务说明、预处理公式 $F'$、当前焦点集与历史反馈；返回 JSON（IFS 含 `witness_map`，CFS 含 `constraint`），经签名合法性、类型检查与去重后传入求解器。
- **退让机制**：迭代预算 $R$ 耗尽或未产生确定性结论时，调用 $S(F)$ 处理原始公式剩余超时，防止覆盖率回退。

## 实验与结果
- **数据集**：SMT-COMP 2025 / SMT-LIB 的 1015 个量化数组实例，覆盖 ALIA(400)、ABV(400)、ABVFP(60)、ABVFPLRA(77)、ANIA(78) 五个逻辑；采用固定随机种子 (42) 对实例数超 400 的逻辑均匀采样。
- **基线与配置**：Z3 v4.15.0（默认）、CVC5 v1.3.2（tuned baseline: `fmf_arr_fp_mbqi_enum`）、CVC5-COMP（SMT-COMP 2025 竞赛配置）；LLM 使用 DeepSeek-V3.2-chat、Gemini-3-Flash-Preview、DeepSeek-V3.2-reasoner（temperature=0）。
- **端到端超时**：1200 秒/实例（含 LLM 调用延迟）；默认 10 轮迭代。
- **主要结果**：
  - LINGUAARRAY(Z3) 解决 701 实例，成功率 69.06%，相对 Z3 基线 141 实例（13.89%）提升 **397%**。
  - LINGUAARRAY(CVC5) 解决 659 实例，成功率 64.93%，相对 CVC5 基线 321 实例（31.63%）提升 **105%**。
  - 优于 CVC5-COMP 的 38.23%；即使对比 Z3+CVC5 虚拟最优组合（390 实例，38.4%），LINGUAARRAY(Z3) 仍额外多出 256 实例。
  - ABVFP/ABVFPLRA 中 Z3 处理量词+数组+浮点组合时报内部错误（解 0 例），CVC5 基线亦仅 31%/59%，LINGUAARRAY(CVC5) 提升至 71.67%/70.13%。
- **消融与 ablation**：
  - Harness-only（空焦点集）几乎与基线持平（Z3: 143 vs 141；CVC5: 322 vs 321）。
  - 随机启发式提议者（从 ground 子项抽样）分别解 436（Z3）/445（CVC5）实例，显著低于 LLM 引导版本。
- **LLM 差异**：DeepSeek-Reasoner 单次输出约 8439 tokens、延迟常逼近 600s，预算内仅能迭代 2 轮；Gemini-Flash 平均 553 tokens 支持更多轮次；多模型虚拟最优组合（DeepSeek-Chat + Gemini + Reasoner）分别比最佳单模型再增加 74（Z3）/80（CVC5）实例。
- **迭代收益**：首轮即带来显著提升（Z3 从 13.9%→50.1%），前 3 轮收获大部分增益，之后收益递减。
- **稳定性**：DeepSeek-V4-Flash × Z3 三次独立运行解决 689.7±4.0 实例，602 实例在所有运行中一致解决。

## 相关工作脉络
- **CEGQI（Reynolds et al., 2015）与 SyQI（Niemetz et al., 2021）**：基于反例驱动的量化实例化，在语法/语义层面枚举候选项并对照模型修正；LINGUAARRAY 与其本质区别是不依赖反例模型提取，而是由 LLM 一次性提出语义焦点集。
- **QSMA（Bonacina et al., 2023）**：通过抽象-细化循环计算量词的 under-/over-approximation；LINGUAARRAY 借鉴过/欠近似思想但接口更窄，仅用 IFS/CFS 聚焦局部关键子句。
- **Array Property Fragment（Bradley et al., 2006）**：限制量词至算术索引守卫与值约束以实现判定性；LINGUAARRAY 面向该片段之外的任意量化数组公式，靠 LLM 语义洞察绕过严格的语法限制。
- **ML 引导量化选择（Jakubův et al., 2025；Janota et al., 2022）**：使用梯度提升树/LightGBM 学习实例选择或排序；LINGUAARRAY 使用 LLM 替代统计模型，并提供可解释的结构化焦点而非黑箱打分。
- **AquaForte（作者先前工作，AAAI 2026）**：LLM 合成完整未解释函数定义后由求解器验证，主要面向 SAT 且改进局限于可满足实例；LINGUAARRAY 针对 UNSAT/SAT 双方向，提出局部焦点集而非完整解。

## 局限性与未来方向
- **理论覆盖有限**：当前仅针对量化数组理论，尚未扩展到包含未解释函数、更复杂理论组合或一般一阶逻辑的广泛场景。
- **LLM 质量敏感**：框架效果高度依赖 LLM 推理能力；低质量提议会增加无效迭代开销，推理型模型因耗时长反而轮次受限。
- **复杂公式仍是瓶颈**：当公式自身结构复杂度（而非实例化位置选择）成为主要障碍时，单纯焦点引导难以突破。
- **未来方向**：扩展至更广泛的量化 SMT 理论（如未解释函数、混合理论）；开发更统一的焦点集生成规范；加深 LLM 与求解器内部（如 E-graph、DPLL(T) 循环）的耦合；探索多 LLM 集成/投票机制与学习式细化策略。

## 研究启发与可借鉴点
1. **"窄接口+强验证"的混合范式**：用 LLM 提供局部、结构化的语义假设，再由形式化求解器完成全局验证，既利用 LLM 的洞察力又规避其幻觉风险——该模式可迁移至定理证明、程序分析、约束求解等领域。
2. **双重 over-/under-approximation 策略**：UNSAT 用 IFS 过近似、SAT 用 CFS 欠近似，配合蕴含认证桥接两者，构成一套对称且可合成的双分支框架，设计优雅且易于推广到其他理论。
3. **反馈驱动迭代与预算退让机制**：每轮将求解器返回的反例特征（未约束 guard、任意赋值索引等）注入下一轮提示，并在预算耗尽时无缝回退到基线——对任何 LLM-符号混合系统具有通用参考价值。
4. **LLM 多样性互补**：不同模型在不同实例子集上各有所长，虚拟最优组合带来额外增益，提示多模型集成/路由策略是提升鲁棒性的可行路径。
5. **可复现性工程实践**：详细的 prompt schema、解析修复流程、签名/类型校验、非法提议丢弃机制等，为 LLM 介入硬核形式化系统的工程化落地提供完整参考。

## 关键术语表
- **Semantic Focus Set**：LLM 提出的有限结构化集合（IFS 或 CFS），用于在可验证接口内引导 SMT 求解器的推理方向。
- **Instantiation Focus Set (IFS)**：针对全称量词正出现位置的一组 ground 实例项，导出过近似公式 $F_{IFS}^{\downarrow}$，用于 UNSAT 分支。
- **Constraint Focus Set (CFS)**：一组 quantifier-free ground 约束，作为辅助假设加入原公式形成 $F_{CFS}^{\uparrow}$，用于 SAT 分支与可能的 UNSAT 认证。
- **E-matching**：基于触发器模式匹配的量化实例化技术，通过在 E-graph 中检索与触发项等价ground 项并生成对应实例。
- **MBQI（Model-Based Quantifier Instantiation）**：为当前 ground 约束构造候选模型，对违反量词语句提取 counterexample 并生成实例。
- **DPLL(T)**：现代 SMT 求解器的基础架构，将命题 DPLL 搜索与背景理论判定器组合，通过 unit propagation 与 theory reasoning 协同决策。
- **Array Property Fragment**：限制量化数组公式中量词至算术索引守卫与值约束的可判定子片段，超出该片段需启发式实例化。
- **Virtual-best portfolio**：多个求解器/配置所解实例集的并集，用作理想化比较基准以衡量新方法是否突破了现有组合的覆盖边界。

## 可复现要素
- **数据集**：SMT-LIB / SMT-COMP 2025 量化数组实例；采样策略与随机种子（42）已在论文中声明。
- **代码与提示模板**：完整开源，GitHub https://github.com/kylelv2000/LinguaArray，MIT 许可证；prompt 模板、输出 schema、解析修复、验证与跨轮反馈均公开。
- **SMT 求解器版本与配置**：Z3 v4.15.0（默认）；CVC5 v1.3.2（tuned baseline: `–finite-model-find –arrays-exp –fp-exp –mbqi –enum-inst –no-cegqi –no-sygus-inst`）；附录 B 给出完整命令行。
- **LLM 配置**：DeepSeek-V3.2-chat / DeepSeek-V3.2-reasoner / Gemini-3-Flash-Preview，temperature=0；附录 F 含完整 prompt 格式与解析细节。
- **关键超参**：端到端超时 1200 秒；默认迭代预算 $R=10$；附录 D.1 报告深度Seek-V4-Flash 稳定性实验的 3 次独立运行参数。
- **补充材料**：Appendix A 给出完整算法伪代码；Appendix C 报告 CVC5 参数消融；Appendix E 提供非 LLM 启发式基线细节；Appendix F.5 统计 LLM 提议的合法性比例。
