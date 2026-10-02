---
title: "RETRIEVAL-AUGMENTED-SKILL-OPTIMIZATION-VIA-CROSS-HARNESS-ADA"
source: https://arxiv.org/pdf/2609.38024v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 08:18:33"
field: "Agent技能优化与知识检索"
keywords: ["Agent Skill", "Skill Optimization", "Retrieval-Augmented", "Cross-Harness Adaptation", "LLM Agent"]
innovations: ["提出RASO框架，首次将外部技能语料库同时用于技能初始化和迭代更新", "设计Cross-Harness Adaptation机制，解决跨领域/跨工具的技能迁移适配问题"]
benchmarks: ["OfficeQA", "SpreadsheetBench", "ALFWorld", "WebShop"]
---

# 论文速读：RETRIEVAL-AUGMENTED-SKILL-OPTIMIZATION-VIA-CROSS-HARNESS-ADA

## 一句话总结
本文提出检索增强技能优化框架（RASO），通过外部技能语料库作为先验知识，利用跨执行框架适配（Cross-Harness Adaptation）将异构来源的程序性知识迁移至目标任务与环境，实现无需大量Rollout的高质量初始技能构建与迭代优化。

## 研究问题与动机
- **核心问题**：现有Agent技能优化方法（如TextGrad、GEPA、SkillOpt等）仅依赖目标Agent自身Rollout生成的执行轨迹进行迭代优化，忽视了已公开积累的海量外部技能知识库（如GitSkills中数百万条技能）。
- **检索直接复用的缺陷**：即使从外部语料检索到相关技能，由于来源技能的领域、工具集、执行框架（harness）与目标环境存在显著差异（如SpreadsheetBench中92.9%的相似技能来自不同harness），直接复用会引入不匹配的程序知识甚至降低性能。
- **计算成本瓶颈**：传统优化方法需大量昂贵的agent rollout来收集经验，导致优化过程成本高、探索空间受限。

## 核心贡献（创新点）
1. **提出RASO框架**：首次将外部技能语料库同时用于技能初始化和迭代更新两个阶段，打破传统方法仅依赖自身Rollout经验的局限。
2. **设计Cross-Harness Adaptation机制**：通过适配智能体将检索到的源领域技能转化为目标harness词汇表下的可执行指导（lessons），解决跨领域、跨工具的语义鸿沟。
3. **RASI无需Rollout构建高质量初始技能**：仅需任务描述T和harness描述H即可生成知识grounded的初始技能，在零Rollout条件下超越直接检索和无检索LLM初始化。
4. **RASU基于执行反馈检索缺失知识**：在迭代优化阶段通过分析失败轨迹生成文本梯度与检索查询，动态补充外部知识，持续提升技能质量。

## 方法详解

### 问题设定
- 冻结语言模型 $\mathcal{M}$ 在固定执行框架 $h$ 下作为Agent，技能 $s$ 为自然语言形式的上下文输入。
- 优化目标：在训练集 $\mathcal{D}_{\text{train}}$ 上构造技能，在验证集 $\mathcal{D}_{\text{val}}$ 上评估选择，最终在测试集 $\mathcal{D}_{\text{test}}$ 上报分：
$$s^{\star} = \arg\max_s \mathbb{E}_{x \sim \mathcal{D}_{\text{val}}}[r(h(\mathcal{M}, x, s))]$$

### 共享模块：Section-level Retrieval + Cross-Harness Adaptation
1. **段落级检索**：将外部技能文档按标题切分为段落级别子库 $\mathcal{C}$，通过BM25检索Top-K最相关段落 $\mathcal{S}_q = \text{BM25}(q, \mathcal{C}, K)$，提升信噪比。
2. **适配公式**：给定需求 $c_i$、任务描述 $T$、harness描述 $H$ 和检索段落 $\mathcal{S}_{q_i}$，适配智能体生成可执行lesson：
$$\ell_i = \mathcal{A}_{\text{adaptation}}(c_i, T, H, \mathcal{S}_{q_i})$$
3. **适配三原则**：(1) 移除源领域特有名词、剔除目标harness无对应实现的程序；(2) 仅聚焦当前需求 $c_i$；(3) 保留的工具/参数行为声明必须有 $H$ 佐证。

### RASI（检索增强技能初始化）
四步流程，零Rollout：
1. **需求与查询生成**：$\{(c_i, q_i)\}_{i=1}^{M} = \mathcal{A}_{\text{query\_init}}(T, H)$
2. **段落检索**：对每个 $q_i$ 检索 $\mathcal{S}_{q_i}$
3. **Cross-Harness Adaptation**：生成 grounded lessons $\mathcal{L}_{\text{init}} = \{\ell_1, \ldots, \ell_M\}$
4. **技能合成**：$s_0 = \mathcal{A}_{\text{skill\_init}}(T, H, \{c_i\}, \mathcal{L}_{\text{init}})$

### RASU（检索增强技能更新）
在RASI初始技能 $s_0$ 基础上迭代优化：
1. **文本梯度与查询生成**：在 $\mathcal{D}_{\text{train}}$ 上采样mini-batch执行rollout，分析失败轨迹生成 $(q_i, \delta_i)$：
$$\{(q_i, \delta_i)\}_{i=1}^{N} = \mathcal{A}_{\text{query\_update}}(T, H, s_t, \{\tau_j\})$$
2. **检索与适配**：同上生成 $\mathcal{L}_{\text{update}}$
3. **技能更新**：$s_{t+1} = \mathcal{A}_{\text{skill\_update}}(s_t, \{\delta_i\}, \mathcal{L}_{\text{update}})$
4. **接受条件**：仅在 $\mathcal{D}_{\text{val}}$ 上优于 $s_t$ 时接受 $s_{t+1}$

## 实验与结果

### 实验设置
- **模型**：GPT-5.6-Luna、Qwen-3.5-9B
- **基准**：OfficeQA（企业文档推理）、SpreadsheetBench（电子表格操作）、ALFWorld（具身交互）、WebShop（网页购物）
- **外部语料**：GitSkills（跨领域技能库）
- **基线**：No Skill、SkillRouter、RFSI（检索-free初始化）、TextGrad、GEPA、SkillOpt、WikiSkill

### 主要结果（GPT-5.6-Luna）
**初始化阶段（零Rollout）**：
- OfficeQA：RASI **45.74** vs RFSI 40.11（+5.63）vs SkillRouter 11.44
- SpreadsheetBench：RASI **49.17** vs RFSI 44.40（+4.77）
- ALFWorld：RASI **72.64** vs RFSI 69.40（+3.24）
- WebShop：RASI **45.06** vs RFSI 43.89（+1.17）

**迭代更新阶段**：
- OfficeQA：RASO **49.03**（vs SkillOpt 45.54，+3.49）
- SpreadsheetBench：RASO **63.33**（vs SkillOpt 57.02，+6.31）
- ALFWorld：RASO **74.13**（vs SkillOpt 72.64，+1.49）
- WebShop：RASO **46.61**（vs SkillOpt 45.54，+1.07）

### Qwen-3.5-9B结果
- OfficeQA：RASO **42.25**（vs WikiSkill 37.40，+4.85）
- SpreadsheetBench：RASO **31.55**（vs WikiSkill 29.76，+1.79）
- ALFWorld：RASO **51.00**（vs GEPA 43.53，+7.47）
- WebShop：RASO **24.73**（vs WikiSkill 13.43，+11.30）

### 关键消融结论
- **RASU贡献**：RFSI+RFSU → RFSI+RASU，OfficeQA +6.86，Spreadsheet +9.40
- **RASI贡献**：RASI+RFSU vs RFSI+RFSU，OfficeQA +5.23，Spreadsheet +6.78
- **联合增益**：RASI+RASU相比基线分别提升 +8.33（OfficeQA）和 +11.66（Spreadsheet）
- **Cross-Harness Adaptation必要性**：RASI无适配仅41.86（OfficeQA），有适配达45.74（+3.88）；SkillRouter直接复用甚至低于No Skill基线
- **最优K值**：K=5时性能最佳，K=10略有下降（噪声增加）
- **语料规模**：仅用1%语料即显著优于无检索；性能随语料增大持续提升
- **成本优势**：RASO在3/4基准上API成本最低（如WebShop $10.28 vs TextGrad $41.03，节省75%）

## 相关工作脉络

1. **TextGrad**（Yuksekgonul et al., 2025）：将技能视为文本变量，通过自然语言反馈反向传播优化；本文定位差异：TextGrad仅依赖模型自身parametric知识生成梯度，RASO额外检索外部技能知识弥补经验盲区。
2. **GEPA**（Agrawal et al., 2026）：通过反思执行轨迹进行prompt演化，维护Pareto前沿；本文定位差异：GEPA纯内源性优化，RASO通过外部检索扩展探索空间。
3. **SkillOpt**（Yang et al., 2026）：将rollout得分转化为有界编辑操作（add/delete/replace）；本文定位差异：SkillOpt局限于本地轨迹编辑，RASO引入跨域知识注入。
4. **WikiSkill**（Tang et al., 2026）：构建持久化wiki整合经验；本文定位差异：WikiSkill是"自我经验积累"，RASO是"外部知识检索+适配"。
5. **SkillRouter**（Zheng et al., 2026）：两级检索-重排选取外部技能；本文定位差异：SkillRouter直接复用未适配技能，实验证明会导致性能下降（如SpreadsheetBench上低于No Skill），RASO的Cross-Harness Adaptation是关键改进。
6. **GitSkills**（Destefanis et al., 2026）：大规模公开技能数据集；本文定位：将其作为外部知识源，首次系统性用于skill optimization的初始化和迭代两个阶段。

## 局限性与未来方向

- **检索粒度限制**：当前采用段落级检索，对于需要跨多个段落组合知识的复杂任务可能覆盖不足。
- **适配质量依赖LLM能力**：Cross-Harness Adaptation的效果受适配智能体自身能力制约，可能在极端异构场景下产生信息丢失或误适配。
- **语料偏见风险**：GitSkills等公开语料可能存在领域分布不均，对特定垂直领域（如法律、医疗）的支持可能有限。
- **固定轮次优化**：当前采用固定epoch数（2 epochs）和mini-batch size（40），缺乏动态停止准则。
- **未探索多技能协同**：仅考虑单技能优化，未研究多个互补技能的联合检索与组合策略。

## 研究启发与可借鉴点

1. **检索粒度设计**：段落级检索（而非文档级）显著提升信噪比，这一设计可迁移至其他基于检索的知识增强场景（如代码生成、数学推理）。
2. **适配优先于直接复用**：实验证明SkillRouter直接复用反而劣于无检索，强烈支持"检索+适配"范式而非"检索+拼接"，这一原则适用于任何跨域知识迁移任务。
3. **双阶段设计解耦**：RASI（离线知识注入）与RASU（在线反馈调整）的职责分离清晰，初始化质量直接影响后续优化上限，可在其他代理学习任务中复用此架构。
4. **成本-性能权衡量化**：本文系统报告了API成本与rollout次数的对比，RASO在保持最优性能的同时成本最低，这一评估维度值得在后续工作中延续。
5. **文本梯度与检索结合**：RASU将失败分析生成的文本梯度与外部检索query并行生成，可启发研究如何将"模型内生梯度"与"外生知识检索"统一建模。

## 关键术语表

**Agent Skill**：以自然语言形式表达的可复用程序性知识 artifact，指导Agent在特定harness下完成任务，区别于模型参数中的隐性知识。
**Execution Harness（h）**：定义Agent可用工具、文件访问权限和评估流程的执行环境接口。
**Cross-Harness Adaptation**：将检索到的源领域技能抽象掉领域专有表述，重新表述为目标harness词汇表下的可执行指导（lesson）的适配机制。
**RASI（Retrieval-Augmented Skill Initialization）**：基于外部技能语料库和任务/harness描述，无需任何rollout即可生成高质量初始技能的方法。
**RASU（Retrieval-Augmented Skill Update）**：基于执行反馈识别失败模式，检索补充缺失知识并迭代更新技能的优化方法。
**Textual Gradient（文本梯度）**：由LLM分析失败轨迹后生成的自然语言形式的优化建议，类比于参数空间的梯度信号。
**GitSkills**：包含数百万公开Agent技能的GitHub技能数据集，作为本文外部知识源。
**BM25**：基于概率检索模型的段落级文档排序算法，用于从技能库中检索最相关段落。

## 可复现要素

- **数据集**：OfficeQA、SpreadsheetBench、ALFWorld、WebShop（均为公开benchmark）
- **代码/权重**：论文未提及开源计划
- **关键超参**：检索段落数K=5；训练轮数2 epochs；mini-batch size=40；chunk size=8；温度Qwen: rollout=0.0, optimizer=0.7
- **外部语料**：GitSkills（需自行下载构建索引）
- **实验硬件**：单张NVIDIA RTX A6000 GPU（Qwen-3.5-9B）；GPT-5.6-Luna通过API调用
