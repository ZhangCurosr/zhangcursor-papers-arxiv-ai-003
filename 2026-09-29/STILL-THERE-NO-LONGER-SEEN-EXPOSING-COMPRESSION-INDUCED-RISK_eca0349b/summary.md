---
title: "STILL-THERE-NO-LONGER-SEEN-EXPOSING-COMPRESSION-INDUCED-RISK"
source: https://arxiv.org/pdf/2609.35002v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 22:09:49"
field: "视觉语言模型安全与鲁棒性"
keywords: ["压缩诱导风险", "视觉语言模型", "对抗攻击", "Token 压缩", "压缩特有失败", "鲁棒性评估"]
innovations: ["提出压缩特有失败(CSF)的配对归因框架并验证保留集分配的因果效应", "设计CIRA编码器侧攻击，在无下游访问条件下实现跨压缩器的高CSFR", "提出TCS跨视图选择稳定化防御，显著抑制压缩选择性攻击"]
benchmarks: ["POPE", "TextVQA", "MME"]
---

# 论文速读：STILL THERE, NO LONGER SEEN: EXPOSING COMPRESSION-INDUCED RISK IN LARGE VISION-LANGUAGE MODELS

## 一句话总结
论文定义了"压缩特有失败"（CSF），揭示了视觉 Token 压缩在保留模型全 Token 正确性的同时可在压缩路径上诱导对抗失败的风险，并据此提出 **CIRA** 攻击框架——仅需视觉编码器白盒访问，即可在不了解具体压缩器与压缩预算的条件下，实现对多种压缩规则的高迁移性压缩特异攻击。

## 研究问题与动机
1. **评估盲区**：现有 LVLM 鲁棒性评估多在完整 token 推理下完成，当部署模型使用压缩变体时，压缩路径特有的失败无法被该评估捕获。
2. **因果归因缺失**：聚合准确率指标掩盖了由视觉加速（压缩）引入的实例级预测变化，无法将失败归因于压缩本身还是基础模型已有漏洞。
3. **部署不确定性下的攻击难题**：在未知具体部署压缩器类型和精确压缩预算的情况下，能否通过单一图像扰动在保持全 Token 推理正确的同时，选择性地在压缩路径上诱导失败？
4. **风险量化需求**：压缩不仅降低推理开销，还改变了压缩后可用的视觉证据集合，需建立配对评估框架以量化压缩引入的安全风险。

## 核心贡献（创新点）
1. **压缩风险归因框架**：提出基于干净条件配对判定的 CSF 形式化定义，并通过受控反事实干预证明保留集分配对压缩正确性具有因果影响，且恢复效果与位移证据的表征漂移呈负相关。与先前工作不同，本文首次将压缩风险作为独立的因果变量进行建模与归因，而非仅报告聚合 accuracy 下降。
2. **CIRA 攻击（仅编码器侧目标）**：设计了全局选择劫持（GSH）与隐藏证据保留（HEP）两个编码器端目标的联合优化，在仅访问视觉编码器、不依赖下游问题/标签/压缩器/预算的前提下实现跨压缩规则的 CSF 诱导。与 VEAttack/CAGE 等下游无关方法的本质区别在于，CIRA 通过优先级重分配而非单纯扰动表征来选择性地在压缩路径制造失败。
3. **跨视图选择稳定化防御 TCS**：提出 Translation-Consensus Selection，利用干净高优先级证据在平移视图下具有优先级稳定性的特性来抑制 CIRA，相对现有防御工作在对抗未感知压缩器的攻击上实现了 81.4% 的相对降幅。

## 方法详解
- **压缩特有失败（CSF）定义**：对于干净条件下全 Token 和压缩推理均正确的样本集合 $\mathcal{S}_K$，CSF 为对抗样本 $x^{\text{adv}}$ 满足 $c_i(x^{\\text{adv}})=1$（全 Token 正确）但 $c_{i,K}(x^{\text{adv}})=0$（压缩推理失败）的实例集合。
- **全局选择劫持（GSH）**：利用编码器代理优先级分数向量 $\mathbf{s}$，通过最大化对抗优先级与清洁逆优先级排名的标准化对齐度 $\mathcal{L}_{\text{GSH}}$，实现全局优先级倒置——使清洁高优先级 token 被降级、低优先级 token 被晋升，从而改变压缩器的保留集合。
- **隐藏证据保留（HEP）**：对跨越候选保留边界的清洁高优先级 token（即在未知预算区间内可能被移出的 token），以预算边际权重 $w_i$ 聚合其表征漂移距离（半余弦距离），最小化 $\mathcal{L}_{\text{HEP}}$，确保被移位的清洁高优先级 token 的特征方向变化有限，从而维持全 Token 正确性。
- **联合优化**：$\mathcal{L}_{\text{CIRA}} = \mathcal{L}_{\text{GSH}} + \lambda \cdot \mathcal{L}_{\text{HEP}}$，使用投影符号梯度上升，$\epsilon = 4/255$，100 步，$\lambda=0.8$。排序操作不参与梯度传播，离散权重通过 stop-gradient 处理。
- **TCS 防御**：构建四个平移视图（7 像素偏移），对各视图的优先级排名做秩分位数对齐与平均，取 consensus Top-K 作为压缩器输入，从而过滤掉 CIRA 诱导的视角敏感替换 token。

## 实验与结果
- **数据集**：POPE（1,000 对）、TextVQA（1,000 对）、MME（1,000 对）。
- **模型**：主模型 LLaVA-v1.5-7B，附加 Qwen3-VL-8B-Instruct 和 InternVL3.5-8B。
- **压缩器**：VisionZip、VisPruner、PruMerge、FastV；预算 $K \in \{32, 64, 128, 192\}$，共 12 个数据集–压缩器组合。
- **基线**：VEAttack、CAGE（下游无关）；CAA†（强访问参考，使用下游问题与 LM）。
- **主要结果（LLaVA-v1.5-7B）**：CIRA 平均 CSFR = **20.35%**，Full ASR = **6.92%**；对比 VEAttack（CSFR 5.49%，ASR 45.95%）和 CAGE（CSFR 5.92%，ASR 50.20%）；CAA† 虽 Full ASR 更低（3.32%），但 CSFR 仅 3.24%，远低于 CIRA。
- **跨模型迁移**：在 Qwen3-VL 和 InternVL 上 CIRA 同样有效，TextVQA 上表现最强。
- **机制分析**：GSH 驱动压缩特有失败诱导（移除 GSH 后 CSFR 从 18.22% 降至 3.68%）；HEP 显著降低 Full ASR（移除后从 6.92% 升至 23.10%）。
- **防御效果**：TCS 使标准 CIRA 平均 CSFR 从 18.22% 降至 3.39%（相对降幅 81.4%）；Adaptive CIRA 在 TCS 下 CSFR 回升至 12.70%（达到无防御值的 74.5%）。

## 相关工作脉络
1. **VEAttack**（Mei et al., 2026b）：下游无关的视觉编码器攻击，针对编码器表征质量而非压缩选择机制，本文在其基础上引入了压缩选择性维度。
2. **CAGE**（Zhang et al., 2026c）：面向未知压缩设置的对抗攻击，但目标是破坏压缩后性能而非保持全 Token 正确性，本文与之对比展示了 CSF 诱导与通用鲁棒性攻击的本质差异。
3. **CAA**（Zhang et al., 2026a）：可直接操纵 token 选择排名的白盒攻击，依赖下游问题和 LM 访问；本文在更受限的编码器白盒设定下实现了更高的 CSFR。
4. **SAP**（Wang et al., 2026a）和 **robustness-oriented pruning**（Gu et al., 2026）：面向剪枝鲁棒性的防御工作，本文从攻击侧补充了压缩引入风险的度量框架。
5. **视觉 token 压缩方法**（VisionZip、VisPruner、PruMerge、FastV 等）：本文在四种主流压缩器上统一评估了 CSF 风险，揭示了压缩机制多样性下的普遍脆弱性。

## 局限性与未来方向
1. **仅评估单图推理**，未涉及视频、多图或多轮对话等含时间/上下文依赖的场景，其中证据保留还受时序冗余影响。
2. **未覆盖非选择型压缩机制**，如自适应剪枝（跨输入/层变化）、学习性摘要（表征变换）、可恢复路由等，这些机制下的 CSF 归因需额外诊断工具。
3. **CSF 定义基于任务正确性**，未涉及响应安全性或定向 jailbreak；全 Token 正确性不等于安全保留，压缩路径失败也不一定构成安全对齐失效。
4. **TCS 防御针对确定性公开压缩器**，在自适应攻击和部分未知的部署场景下防御效果有限。

## 研究启发与借鉴点
1. **配对评估范式**（全 Token vs 压缩路径）可有效隔离压缩引入的独立风险，该思路可迁移至其他模型加速技术（如 quantization、sparsification）的鲁棒性评估中。
2. **保留集反事实干预**（exchange retained/omitted tokens 并观测正确性变化）是一种简洁有力的因果诊断工具，可用于分析任意 token 选择模块的脆弱性来源。
3. **表征漂移与恢复效果的负相关发现**提示：在对抗攻击设计中对被移出 token 做表征保护，是同时维持目标路径失败和非目标路径正确性的关键技巧。
4. **跨视图优先级稳定性**作为防御思路可推广到其他依赖 token 排名的模块（如检索增强、多尺度特征聚合），用于增强部署阶段的鲁棒性。
5. **encoder-only 白盒设定下的优先级重分配目标**具有通用性，可与本团队在 efficient VLM 推理安全评估的方向结合，探索更多压缩策略下的 paired robustness 评测基准。

## 关键术语表
**Compression-Specific Failure (CSF)**：全 Token 推理正确但压缩推理失败的对抗样本，用于将压缩引入的风险与基础模型已有漏洞分离。
**Global Selection Hijacking (GSH)**：通过编码器端代理优先级对齐目标，全局重排 token 优先级以实现压缩器保留集合的劫持。
**Hidden-Evidence Preservation (HEP)**：对跨越压缩边界的清洁高优先级 token 限制其表征漂移，以维持全 Token 推理的正确性。
**Translation-Consensus Selection (TCS)**：利用多平移视图下的优先级排名共识来稳定 token 选择，抵御 CIRA 类攻击。
**Representation Drift**：清洁与对抗 token 表征间的半余弦距离，衡量对抗扰动对 token 语义内容的改变程度。
**Selective Gap**：Avg. CSFR 与 Full ASR 之差，用于量化攻击在压缩路径与全 Token 路径上的选择性差异。

## 可复现要素
- **数据集**：POPE、TextVQA、MME（均为公开数据集），各采样 1,000 对 image-question。
- **模型**：LLaVA-v1.5-7B、Qwen3-VL-8B-Instruct、InternVL3.5-8B（公开预训练模型）。
- **代码**：已开源至 Github（论文 Reproducibility Statement 明确声明）。
- **关键超参**：$\epsilon = 4/255$，优化步数 100，$\lambda = 0.8$，步长 $1/255$，候选预算区间 $[32, 192]$，TCS 平移距离 $d = 7$ 像素。
- **设备**：单张 NVIDIA GeForce RTX 4090。
