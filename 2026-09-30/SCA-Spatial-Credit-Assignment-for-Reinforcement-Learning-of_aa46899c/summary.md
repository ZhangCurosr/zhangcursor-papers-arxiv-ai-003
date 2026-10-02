---
title: "SCA-Spatial-Credit-Assignment-for-Reinforcement-Learning-of"
source: https://arxiv.org/pdf/2609.36939v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 19:56:53"
field: "GUI agent reinforcement fine-tuning"
keywords: ["GUI agents", "reinforcement learning", "spatial credit assignment", "group-relative RL", "GUI grounding", "policy optimization", "Theil-Sen estimator"]
innovations: ["提出组内空间信用分配框架，用坐标-奖励趋势修正混合命中组信用", "设计距离排序分支应对全失败组零优势退化", "在合成设定下验证更新方向与精确返回梯度余弦相似度0.99997"]
benchmarks: ["ScreenSpot-Pro", "ScreenSpot", "GUI-Act-Web", "OmniAct-Web", "OmniAct-Desktop", "GUI-Odyssey"]
---

# 论文速读：SCA-Spatial-Credit-Assignment-for-Reinforcement-Learning-of-GUI-Agents

## 一句话总结
本文提出空间信用分配（Spatial Credit Assignment, SCA）方法，通过在组内利用点击坐标构建空间参考信号，改进GUI代理强化学习中的奖励信用分配，解决二元奖励导致的"全失败零优势"和"多失败同信用"问题，在ScreenSpot-Pro和OmniAct等基准上均取得RFT模型中的最优或接近最优结果。

## 研究问题与动机
- **二元奖励忽视空间结构**：现有组相对强化学习（如GRPO）仅用二值奖励比较同状态采样响应，导致空间位置不同的失败点击被赋予相同信用，丢失空间位置信息。
- **全失败组零优势困境**：当一组采样点击全部失败时，二元奖励的组内方差为零，传统组相对信用退化为零，无法提供任何相对学习信号。
- **混合命中组的信用平权问题**：在同时包含成功与失败的组中，所有失败点击获得相同的负向优势，但距离目标远近不同的失败点击应得到不同程度的修正信号。
- **推理部署无额外开销**：所需的空间修正仅在训练时构造，部署策略保持不变，无需引入额外预测器。

## 核心贡献（创新点）
1. **提出组内空间信用分配框架**：将屏幕坐标信息引入强化学习信用计算，区分混合命中与全失败两种失败模式，这是首个针对GUI动作空间结构系统性改进组相对信用的工作。
2. **设计SCA-Residual分支**：通过留一交叉拟合坐标-奖励趋势，对混合命中组内的失败点击进行残差修正，区别于已有工作仅依赖二值奖励的信用分配。
3. **设计SCA-Prox分支**：在全失败组中基于目标距离指数排序提供信用信号，与已有基于高斯奖励塑形（GUI-G²）的方法相比，仅在奖励方差退化时才激活。
4. **建立精确梯度对照分析**：在合成设定下验证SCA更新方向与精确返回梯度的余弦相似度接近1、MSE降低约8.5%，从理论层面支撑方法有效性。
5. **实现RFT模型中最强结果**：在ScreenSpot-Pro十二子列均值26.1%、OmniAct-Desktop接地精度80.5%、GUI-Odyssey成功率66.0%，均为已报告RFT结果最高。

## 方法详解
**基础框架**：给定GUI状态x，策略πθ采样N个响应形成组，二元评估器给出奖励r_i∈{0,1}。SCA在每个组内计算空间信用A_i后detach并共享给该响应所有token。

**Residual分支（混合命中组）**：
- 将坐标投影到四个方向（水平、垂直、两条对角线），对每个响应i，用其余N-1个响应的坐标-奖励对拟合Theil-Sen直线（中位数斜率+中位数截距）
- 内层留一验证评估预测质量s_{i,d}，加权融合得预测奖励r̂_i
- 预测质量q_i∈[0.80,0.99]线性映射到门控权重w_i
- 残差信用A^{res}_i=z_V(r_V-r̂_V)_i与GRPO优势混合：A^{Residual}_i=[z(w⊙A^{res}+(1-w)⊙A^{GRPO})]_i

**Prox分支（全失败组）**：
- 计算点击到目标中心c_T的归一化距离， proximity p_i=exp(-||a_i-c_T||/ℓ_T)
- 信用A^{Prox}_i=α_T·z_{V_T}(p_{V_T})_i，α_T缩放proximity分数的标准差
- 有效坐标外信用为零

**Regime路由**：混合命中用Residual，全失败用Prox，全命中保留原始GRPO优势；无效坐标/退化拟合时回退到GRPO。

**训练目标**：Clipped actor loss L_actor=-∑M_{it}ℓ_{it}/∑M_{it}+λ_KL·D̂_KL，clip比值ε_c=0.2，KL系数λ_KL=10^{-2}。

## 实验与结果
**基线与设置**：
- 主干模型：Qwen2.5-VL-3B-Instruct，每组N=5采样，使用EasyR1实现
- 对比：GUI-R1-3B、UI-R1-3B等已报告RFT结果；SFT基线包括SeeClick、OS-Atlas、UGround等
- 评估：ScreenSpot-Pro（12个子列）、ScreenSpot、GUI-Act-Web、OmniAct-Web、OmniAct-Desktop、GUI-Odyssey

**核心结果**：
- ScreenSpot-Pro均值：SCA 26.1% vs GUI-R1 25.2%，提升0.9个百分点
- ScreenSpot Web Text/Icon：90.5±0.30 / 73.5±0.42；Desktop Text/Icon：94.8±0.28 / 66.5±0.45
- OmniAct-Desktop GR：80.5±0.41，超GUI-R1 2.13pp；OmniAct-Web GR：77.0±0.45；GUI-Act-Web GR：89.0±0.42
- GUI-Odyssey SR：66.0±0.52，超GUI-R1 1.59pp
- 低层任务11项指标中占优10项

**精确梯度分析**（合成设定）：
- Binary GRPO MSE=0.16201，Full SCA MSE=0.14754（↓8.5%），余弦相似度0.99997
- OLS-LOO对照MSE虽低（0.10082）但余弦=-0.572，方向错误

**消融诊断**：
- Text目标改进1.23pp，Icon目标改进0.48pp（感知瓶颈而非信用信号不足）
- 六大全专业领域均提升0.70-1.10pp

## 相关工作脉络
1. **Group-relative RL（GRPO/RLOO）**：Shao et al. 2024; Guo et al. 2025；本文定位为在GUI grounding场景中，用坐标信息弥补二值奖励组内方差的不足。
2. **Action-dependent baselines/critics**：Gu et al. 2017 (Q-Prop); Liu et al. 2018; Wu et al. 2018；本文不引入额外critic网络，完全在组内采样点间构造空间参考。
3. **Spatial reward shaping（GUI-G²）**：Tang et al. 2026 使用高斯奖励建模；本文只在二值奖励方差退化的全失败组中才启用距离信号，保留原评估器不变。
4. **GUI grounding agents（SeeClick/OS-Atlas/ShowUI/UGround）**：Cheng et al. 2024; Wu et al. 2025; Lin et al. 2025; Gou et al. 2025；本文在这些SFT基线之上进行RFT优化。
5. **GUI benchmarks（ScreenSpot/OmniAct/GUI-Odyssey）**：Cheng et al. 2024; Kapoor et al. 2024; Lu et al. 2025；本文使用同一套评测协议保证可比性。
6. **Distance-based credit control**：已有一些工作直接用目标距离作为稠密奖励；本文通过group-relative路由机制仅在必要时激活距离信号，避免干扰正常信号。

## 局限性与未来方向
- **评估仅限离线固定状态**：未测试OSWorld、WebArena、MiniWoB等在线交互环境，无法验证动作改变后续观察后的恢复能力。
- **单尺度主干**：仅验证3B模型，未扩展到7B/更大规模，泛化性待验证。
- **分支归因不清**：Table结果测量整体规则，Residual与Prox各自贡献需匹配消融实验才能分离。
- **门控校准依赖合成数据**：q band [0.80, 0.99]从17种合成场景选定，与真实分布可能存在gap。
- **N=5固定组大小**：未探索组大小敏感性与扩展至N=8/16的效果。
- **奖励塑形交互未研究**：与稠密奖励（高斯+格式）的配合方式待探索。

## 研究启发与可借鉴点
1. **空间参考信号可在组内原位构造**：无需额外网络即可利用坐标-奖励关系，可迁移到其他具离散空间动作（如机器人抓取、屏幕阅读顺序）的RL任务。
2. **交叉拟合+留一验证防泄漏**：Theil-Sen估计+双层留一结构避免自身奖励参与预测，这一设计对任何小样本回归信用分配任务都有参考价值。
3. **精确梯度对照分析的说服力**：在合成设定下与精确返回梯度比较MSE/余弦/范数比三指标，可从理论层面验证信用构造的有效性，建议后续类似工作复用此验证范式。
4. **Regime-aware路由机制**：根据奖励方差是否退化自动选择不同修正策略（回归拟合 vs 距离排序），可作为通用"退化检测+备选信号"模板。
5. **推理零开销保证**：空间修正仅影响训练信用分配，部署保持原策略，这种"训练增强、推理无损"设计易于落地。

## 关键术语表
**Spatial Credit Assignment (SCA)**：利用屏幕坐标构建空间参考信号，修正组相对强化学习中二元奖励导致的信用分配缺陷。
**Residual branch**：SCA在混合命中组中的分支，通过留一拟合坐标-奖励趋势计算残差信用。
**Prox branch**：SCA在全失败组中的分支，按点击到目标距离的指数衰减分配相对信用。
**Theil-Sen estimator**：基于中位数 pairwise slope 的鲁棒线性回归估计量，用于坐标-奖励趋势拟合。
**Leave-one-out cross-fitting**：对每个响应i，用除i外的其余响应拟合预测模型，防止自身奖励参与自身预测。
**Gate weight w_i**：基于嵌套留一验证预测质量q_i线性映射得到的空间修正混合权重。
**Regime routing**：根据组内奖励分布（全命中/混合/全失败）选择不同信用分配路径的机制。
**Exact return gradient**：目标函数关于策略参数的真实梯度，用作合成验证的参考基准。

## 可复现要素
- **数据集**：ScreenSpot-Pro、ScreenSpot、GUI-Act-Web、OmniAct-Web、OmniAct-Desktop、GUI-Odyssey（论文未声明代码/权重开源）
- **代码/权重**：论文未提及开源
- **关键超参**：每组响应数N=5；clip比值ε_c=0.2；KL系数λ_KL=10^{-2}；门控q band [0.80, 0.99]；方向数D=4；Prox距离缩放ℓ_T=max(0.5·max(d_T,1)+50, ε)
- **训练主干**：Qwen2.5-VL-3B-Instruct
- **实现框架**：EasyR1 clipped PPO
