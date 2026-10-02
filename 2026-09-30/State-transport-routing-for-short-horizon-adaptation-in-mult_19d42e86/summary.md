---
title: "State-transport-routing-for-short-horizon-adaptation-in-mult"
source: https://arxiv.org/pdf/2609.36926v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 19:58:39"
---

# 论文速读：State transport routing for short-horizon adaptation in multi-horizon photovoltaic forecasting

## 一句话总结
本文提出一种名为状态传输路由（State Transport Routing, STR）的轻量级输出适配器，通过融合冻结预言模型的原预测、最新实测功率水平与近期线性趋势三条轨迹，动态路由混合权重以修正前120分钟预测，在无需重训练底层模型的前提下显著提升光伏多步预测的短期精度，并严格保留长时预测输出。

## 研究问题与动机
- **核心问题**：如何将最近实测的功率状态信息有效融入已训练好的多步光伏预测模型，同时避免简单外推短期波动导致的长时预测误差放大。
- **现有方法不足**：传统持续性（Persistence）与局部斜率外推假设单一，无法应对辐照/出力的快速突变；通用残差修正缺乏对“当前水平”与“近期趋势”等状态假设的显式区分，易将瞬态爬坡错误放大至长步。
- **设计动机**：构造一个仅作用于预测输出层、不触碰骨干网络内部参数、且对短时与长时预测采取差异化处理的即插即用适配机制。

## 核心贡献（创新点）
1. **显式状态轨迹路由设计**：将原预测、最新水平保持与线性趋势外推三条轨迹作为独立候选路径，由 horizon-conditioned router 动态混合；与黑盒残差修正的本质区别在于引入了可解释的物理先验假设，使路由决策具备明确的短期状态语义。
2. **Exact Bypass 机制**：适配器仅在首个120分钟（K=8步）内生效，此后直接原样返回骨干网络输出，从架构层面杜绝了短视适配对长时预测可靠性的污染。
3. **跨冻结神经骨干的迁移验证**：在统一 PVDAQ 协议下，独立训练 STR 适配器成功复用于 TimeMixer、N-HiTS、TSMixer、DLinear 与 iTransformer 五种异构架构，证明其作为后处理模块的架构无关性。
4. **参数匹配的严格对照实验**：构建与 STR 参数量一致的纯残差适配器作为 control，通过配对95%置信区间验证，证实显式轨迹设计带来的精度提升具有统计显著性。

## 方法详解
- **候选轨迹构造**：对预报原点 $t$ 的步长 $h$，定义三条轨迹：
  - $p_{t,h}^{(1)} = b_{t,h}$（冻结骨干模型的原始多步预测）
  - $p_{t,h}^{(2)} = x_t$（最新实测功率，即持续性假设）
  - $p_{t,h}^{(3)} = x_t + h \cdot \frac{x_t - x_{t-3}}{3}$（基于最近3个采样点的线性斜率外推）
- **状态描述符与路由**：提取最近12步功率测量、一阶差分与局部波动构成局部状态描述符 $\mathbf{u}_t$，结合预报步长嵌入 $\mathbf{e}_h$，经线性投影与 GELU 激活得隐藏表征 $\mathbf{z}_{t,h} = \mathrm{GELU}(W_u \mathbf{u}_t + \mathbf{e}_h)$。
- **权重与残差生成**：线性层输出3个路由 logits $\mathbf{a}_{t,h}$ 与加性残差 $d_{t,h}$，软路由权重 $\pmb{\alpha}_{t,h} = \mathrm{softmax}(\mathbf{a}_{t,h})$。
- **分阶段预测公式**：
  - 短时适应窗口（$h \le K$，$K$ 对应120分钟）：$\hat{y}_{t,h} = \sum_{j=1}^3 \alpha_{t,h}^{(j)} p_{t,h}^{(j)} + d_{t,h}$
  - 长时精确旁路（$h > K$）：$\widehat{y}_{t,h} = b_{t,h}$
- **对照设计**：参数量匹配的残差控制模型仅学习 $\hat{y}^{\text{control}} = b_{t,h} + d_{t,h}$，路由 logits 实际不起作用，用于隔离显式轨迹的贡献。
- **训练配置**：骨干参数全程冻结；AdamW 优化，lr=$10^{-3}$，weight decay=$10^{-4}$，batch size=256，24 epoch，梯度裁剪 norm=1.0；每2 epoch 在选择集评估并保留最低 MAE Checkpoint；使用
