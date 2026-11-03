---
title: "What-and-Where-in-Identifying-and-Quantifying-Conventions-in"
source: https://arxiv.org/pdf/2610.20205v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-03 09:00:54"
field: "多智能体协作与零样本协调"
keywords: ["convention gap", "off-belief learning", "zero-shot coordination", "Hanabi", "implicit communication", "multi-agent interpretability"]
innovations: ["提出基于精确字面基线的惯例差距度量，实现隐性通信的直接定位与量化", "建立人类与OBL智能体可比框架，揭示间接引用等人类特有惯例", "设计频率匹配干预实验，证明惯例贡献度不等于因果重要性"]
benchmarks: ["hanab.live公开对局数据集", "AH2AC2基准"]
---

# 论文速读：What-and-Where-in-Identifying-and-Quantifying-Conventions-in

## 一句话总结
论文提出"惯例差距"（convention gap）度量，在Hanabi卡牌游戏中将人类与Off-belief learning（OBL）智能体置于同一标尺，直接定位并量化双方隐性通信惯例的异同；发现人类整体优势主要来自间接引用等稀有情境，共享惯例上人类仅略优于OBL Level 2。

## 研究问题与动机
1. **惯例只能间接观测**：现有ZSC研究通过换伙伴后的分数下降推断惯例存在，但无法说明惯例"是什么"、"在哪里使用"及其价值。
2. **人类与AI惯例是否共享**：已教授惯例系统的人类玩家与学习型智能体在哪些情境下收敛于相同/不同的隐性通信规则？
3. **惯例贡献能否预测因果重要性**：观测排名靠前的惯例，其被阻断后是否必然造成更高分数损失？
4. **惯例使用的时序与特征载体**：惯例集中在游戏中哪些阶段、由状态中哪些特征驱动？

## 核心贡献（创新点）
1. **提出精确字面基线度量惯例差距**：以规则保证信息计算每步的P(life loss)，与实际失败率相减得到signed gap（pp），实现惯例的直接定位而非间接推断。
2. **建立人类-OBL同一尺度可比框架**：将hanab.live公开对局与AH2AC2基准通过完全相同的后验计算与抽取管线重放，验证跨语料结构一致性（r=0.97）。
3. **揭示人类优势的结构性来源**：共享高频情境中人类≈Level 2；优势集中于间接引用（人类+61.3 pp vs Level 5仅+13.7 pp）等智能体极少进入的稀有cell。
4. **设计频率匹配干预范式证明贡献≠重要性**：阻断单个高贡献惯例仅造成-0.12 pp超额成本（意义重路由）；阻断Top-8惯例联合产生-1.08 pp超加性成本。

## 方法详解
1. **惯例差距公式**：Δ = mean(P(life loss)) − actual life-loss rate，其中P(life loss)由literal observer基于全手牌联合分布（正负提示约束+可见区卡牌计数）精确计算；区间采用整局bootstrap（人类按pair聚类）。
2. **OBL层次结构**：Level k的信念模型由Level k−1策略诱导；Level 1以均匀随机π₀为基，仅读取行动字面内容，后续层级逐步允许解释更多隐含信息。
3. **双分类体系**：①手工五类（fresh color/rank hint、hinted earlier、unhinted newest/older）；②六分量诱导taxonomy（前驱动作、棋盘关系、提示历史、滞后lag、手牌位置、令牌库存），50/50 discovery/estimation split，n_min=52–139（±5pp精度目标推导），缺失类型按固定backoff合并，Depth-4 CART树作紧凑摘要。
4. **干预实验**：当greedy action落入目标category时将其从合法集移除、取次优；对照为frequency-matched random mask（按目标触发率随机屏蔽同等频率行动），超额成本=配对score变化−控制曲线在同触发率下的预测值。

## 实验与结果
- **数据集**：hanab.live公开日志425局/9017次出牌（155玩家/95对）；AH2AC2独立采集1858局（已完成局过滤后1752局）。
- **主要数字**：人类gap +26.16 [+24.05, +28.28] pp，高于OBL1–5（+1.57→+21.68 pp）；AH2AC2 +27.62 [+27.27, +27.97] pp，类别结构r=0.97复现。
- **共享cell距离**：D₁=15.9 pp → D₂=4.1 pp → D₅=6.5 pp；排除11个"人类>所有级"cell后D₂−D₅=+2.4 [−0.6, +3.5] pp（不显著）。
- **间接引用**：人类+61.3 [+54.4, +66.0] pp（color）、+41.1 [+31.9, +48.8] pp（rank）；Level 5仅+13.7 / +10.3 pp；finesse子cell人类+63.9 vs Level 5 +6.1 pp。
- **时序**：Level 2–5呈倒U型，峰值中局Level 5 +36.7 pp；人类峰值+39.2 pp且开局/终局显著高于Level 5。
- **载体迁移**：Level 2–3由hand position主导；Level 4转为hint history；Level 5由color-only knowledge主导；延迟color cell（lag 2–3/4+）gap达+62.2 pp，远超fresh-hint。
- **干预**：阻塞fresh color-hint immediacy（双侧）得分−2.03点；单次mask超额成本多在[−0.02, +0.03]点；Top-8联合mask得分−1.84点，显著超加性（−1.08 [−1.47, −0.72]）。

## 相关工作脉络
1. **Other-Play [Hu et al., 2020]**：去除对称依赖惯例；本文不提出ZSC算法，而是定位这些算法所作用的惯例。
2. **Off-belief learning [Hu et al., 2021]**：提供1–5级信念层次；本文将其作为惯例量化标尺，验证gap随level单调上升。
3. **Rational Speech Act [Frank & Goodman, 2012]**：将提示建模为递归语用推理；本文无需拟合听众，直接比较观测与字面预测。
4. **AH2AC2基准 [Dizdarevic et al., 2025]**：基于人类对局克隆代理评估ZSC；本文验证克隆保留共享惯例但丢失人类特有indirect-reference cell。
5. **K-level推理 [Cui et al., 2021]**：递归深度路径；本文预注册混合拟合发现τ不可识别，深度不足以解释人机差异。
6. **Any-Play [Lucas & Allen, 2022]**：内蕴增强抑制惯例；本文提供惯例可解释性替代：精确基线+联合消融。

## 局限性与未来方向
1. 仅验证于Hanabi单游戏；方法需精确字面基线，推广至更复杂协作环境待检验。
2. 人类数据来自教授惯例的社区（hanab.live），描述的是"教学化"而非典型人类玩家。
3. Indirect-reference因智能体进入频率过低（Level 5仅0.28% vs 人类2.64%）无法进行因果定价。
4. 克隆实验仅在1.7k–18.3k局开放分割下评估，未覆盖AH2AC2实际部署的147k局+RL微调代理。

## 研究启发与可借鉴点
1. **精确字面基线可迁移至其他协作领域**：只要存在可由规则/状态完全推导的失败概率基线，即可 Decompose gap 定位隐性通信。
2. **频率匹配随机对照分离"惯例价值"与"通用干扰"**：适用于多智能体可解释性中的ablation设计，避免把任意行动屏蔽的成本归因于惯例。
3. **Discovery/Estimation split + CART摘要避免过拟合**：诱导taxonomy可冻结并迁移到另一人群（本文frozen level-5 taxonomy在人类R²=0.28），适合跨population对比。
4. **人类-AI惯例对比框架**：可直接用于评估机器人团队、human-robot协作系统中的隐性协议质量。

## 关键术语表
- **Convention gap（惯例差距）**：实际life-loss率与字面预测失败率之差（pp），正值表示伙伴协作使失败率低于纯规则预期。
- **Off-belief learning（离信念学习）**：以虚构信念模型（固定基策略）生成Q-target的训练范式；Level k的信念由Level k−1策略诱导，递增允许解释过去行动。
- **Indirect reference（间接引用）**：队友提示触碰与当前行动无关的其他卡牌时，当前行动所承载的约定含义（如finesse）。
- **Literal observer（字面观察者）**：仅基于提示约束与可见区计数计算每张牌身份后验的分析器，不含任何 learned component。
- **Zero-shot coordination（零样本协调）**：策略与未训练过的新伙伴配对时的协作绩效。
- **Behavioral clone（行为克隆）**：以监督学习从人类对局数据拟合的代理，用于替代真人评估ZSC算法。

## 可复现要素
- **数据集**：hanab.live公开对局日志（经filter与hash脱敏）；AH2AC2开源基准（Dizdarevic et al., 2025）。
- **代码/权重**：度量引擎随前作公开发布（pinned commit）；完整分析管道、audit JSON、生成脚本将在出版版本发布；Agent checkpoint哈希记录于生成服务器。
- **关键超参**：最小单元格n_min=52–139（由s与±5pp目标推导）；FDR控制q=0.05；50/50 discovery/estimation split；CART深度4；每级5种子×1000局。
