---
title: "VOMMI-Collecting-and-Leveraging-Portable-Demonstrations-for"
source: https://arxiv.org/pdf/2610.08220v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-07 22:29:15"
field: "具身智能-移动操作"
keywords: ["mobile manipulation", "portable demonstration", "visual odometry", "VLA", "R2-VO", "residual adapter"]
innovations: ["低成本双视角RGB便携式演示采集协议，无需机器人或IMU硬件", "R2-VO双模态运动重建：在线因果token+离线VGGT锚点漂移校正", "仅作用于基座动作的VO条件化残差适配器，解耦操控残差"]
benchmarks: ["Barriers-fruit", "Room-bottle", "Cabinet-box"]
---

# 论文速读：VOMMI-Collecting-and-Leveraging-Portable-Demonstrations-for-Mobile-Manipulation

## 一句话总结
VOMMI 提出了一种基于双目 RGB 摄像头的低成本便携式移动操作演示采集框架，通过离线轨迹重建（R2-VO）与在线视觉里程条件化机制，将无机器人依赖的演示数据有效迁移到 VLA 基座策略的后训练中，显著提升长时程移动操作性能。

## 研究问题与动机
1. **移动操作数据稀缺**：全身 VLA/WAM 模型需要长时程、多阶段移动操作演示，但标注与采集成本高、可扩展性差。
2. **现有便携式方案硬件门槛高**：UMI、iPhUMI、Mobile UMI 等方法依赖 GoPro、iPhone、IMU、VR 头显等专用设备，成本和参与门槛限制了大规模数据采集。
3. **纯 VO 轨迹存在漂移与不一致性**：直接使用视觉里程计估计轨迹会累积漂移，尤其在大旋转场景下会导致数据质量下降甚至不可用。
4. **缺乏对可观测运动的有效利用**：已有方法通常假设已知 metric pose 或 IMU，而非从纯 RGB 观测中恢复可迁移的局部视觉运动先验。

## 核心贡献（创新点）
1. **低成本无机器人双视角采集协议**：仅需两个鱼眼 RGB 摄像头（胸挂+机械臂式手柄），无需目标机器人或人机运动学标定，采集成本约 $100-200。
2. **R2-VO 实时双模态运动重建模块**：离线分支融合冻结的 VGGT 几何锚点校正漂移，在线分支以 15 Hz 生成多预测步长的因果局部运动 token，延迟约 10 ms。
3. **仅作用于基座动作的残差适配器设计**：将 VO 运动 token 通过 Gate 仅注入 base-action 残差分支，manipulation 残差独立于身体运动，避免错误耦合。
4. **面向策略学习的当前帧相对动作表征**：定义 Body 平面 SE(2) 与 Hand TCP-relative SE(3) 的本地坐标系动作空间，消除全局位姿偏移敏感性。

## 方法详解
- **数据采集流**：示范者佩戴胸前鱼眼 RGB 摄像头（Body 视角）并手持另一台鱼眼 RGB 摄像头于夹爪（Hand 视角），15 Hz 同步录制。存储内容包括双视角视频、相机内参、TCP/夹爪信号及有效性掩码。
- **当前帧相对动作表征**：Body 动作定义为平面 SE(2) 增量 $\xi^B_{t,k} = [\Delta x, \Delta y, \Delta\psi]$，Hand 动作定义为 TCP 相对 SE(3) 增量 $\xi^E_{t,k} = \log(T_E(t)^{-1}T_E(t+k+1))$，两个流独立计算、互不依赖全局坐标系。
- **R2-VO 在线分支**：提取光流、几何线索和跟踪质量统计构成 206 维特征向量，结合 7 帧时序窗口得 317 维，经极端随机树回归器预测单步运动；在 $k \in \{2,3,6,12\}$ 步长上积分得 29 维运动 token $\nu_t$。
- **R2-VO 离线分支**：将因果轨迹与 1 Hz 的 VGGT-1B 稀疏几何锚点对齐，通过加权 Ridge 回归校准器映射到度量初始相机帧，补偿公式 $\hat{p}^{\text{off}}_{t,xy} = g_{t,xy} + c_{t,xy} - L(c_{xy})_t$ 保留短期运动同时修正累积漂移。
- **平面 Body 补偿**：将 Body 轨迹投影至 SE(2) 平面后提升至 SE(3)，用于对 Hand 轨迹进行 Body-Hand 坐标系对齐：$\bar{T}^{\|}_{E|B}(t) = (T^W_{B,\|}(t))^{-1}\hat{T}^W_E(t)$。
- **VO 条件化动作组残差适配**：运动 token 经 LayerNorm-MLP 投影后经有效性门控生成 $z_t$，base 残差为 $r^b_{t,k} = R_b([h_{t,k} \odot W_z z_t, z_t]) - R_b(0)$，manipulation 残差 $r^m_{t,k} = R_m(h_{t,k})$ 不含运动条件；最终输出 $\hat{u}_{t,k} = u^0_{t,k} + g^b_{t,k} r^b_{t,k} M_b + g^m_{t,k} r^m_{t,k} M_m$，损失函数 $\mathcal{L} = \frac{1}{B}\sum_i \mathcal{L}^{(i)}_{\text{flow}} + 0.1\mathcal{L}_{\text{router}}$。
- **部署推理**：策略输出 base 变换后，按公式 $T^E_{E_{t+k},\text{cmd}} = (T^W_E(t))^{-1}T^W_{B,\text{cmd}}(t+k)\bar{T}^{\text{des}}_{E|B}(t+k)$ 合成六维 EE 指令，全程仅需 RGB、内参、时间戳和机器人本体感知。

## 实验与结果
- **数据集**：每个任务采集 500 条便携式演示，其中 75 条用于 RGB-VO 评估（不参与任何拟合），剩余 425 条用于开发，另设 200 条机器人演示作为参考。
- **任务**：Barriers-fruit（跨越障碍取水果）、Room-bottle（进出房间操作瓶子）、Cabinet-box（开柜放置折叠盒子）。
- **VO 精度**：在线 15 Hz R2-VO 的 Body 平移误差 mean/P90 为 0.26/0.54 cm，较 DPVO 降低 80.4%、较 DROID-SLAM 降低 83.8%；离线 VGGT+anchor 版本 Body ATE 从 15.65 cm 降至 5.97 cm。
- **轨迹重建**：离线重建平均降低 Body/Hand 绝对轨迹误差 24.6%（相对于各流最佳基线）。
- **整体成功率**：VOMMI 在三个任务上的平均成功率为 58.3%，对比 OpenPI 0.5 的 50.0% 提升 8.3 个百分点；对比 GR00T N1.6（平均 18.3%）和 X-VLA（36.7%）优势显著。
- **动作预测**：VOMMI-R 的 base MAE 为 0.036 m/s，较 M20-trained（0.044 m/s）降低 18.2%；EE 平移/旋转误差为 0.170 cm / 0.453°。
- **基线对比**：VO 评估涉及 DROID-SLAM、DPVO、DROID-W、MASt3R-SLAM、CUT3R、VGGT-SLAM 2.0；策略对比涉及 OpenPI 0.5、GR00T N1.6、X-VLA。
- **消融**：Base-VO 残差设计优于 All-Action-VO（后者将 VO 注入全部动作，flow loss 为 3.83 vs VOMMI 3.79）；Parent（无残差）流损失为 3.86。

## 相关工作脉络
1. **UMI 系列（Chi et al., 2024）**：便携式手持夹爪抓取相机采集操作演示，VOMMI 继承其裸设备理念，但进一步降至仅 RGB 双鱼眼，去掉了 IMU/VR 等硬件依赖。
2. **Mobile UMI（Huang et al., 2026）与 HoMMI（Xu et al., 2026）**：均扩展至移动操作，使用多视角+解耦表示；VOMMI 的不同之处在于完全不用 metric pose，改用 VO 条件化 local motion token。
3. **DECOWAM（Ma et al., 2026）**：分离 base/arm latent 并 condition 于 base velocity；VOMMI 与其思路相近，但采用残差 adapter 而非 latent 分离，且直接对接 VLA 后训练接口。
4. **WholeBodyVLA / OpenHLM**：强调 loco-manipulation 的统一建模；VOMMI 聚焦于如何将低成本 portable 数据融入此类预训练 VLA。
5. **DROID-SLAM / DPVO / VGGT-SLAM 2.0**：经典/最新的 VO/SLAM 方法；本文将其作为 motion estimation 基线对比，证明 R2-VO 在局部增量精度上的优势。
6. **EgoHumanoid（Shi et al., 2026）**：使用第一人称人体演示；VOMMI 与其互补，后者需特定设备而 VOMMI 只需通用鱼眼相机。

## 局限性与未来方向
1. **跨 embodiment 泛化尚未充分验证**：论文仅在 M20S 平台上验证，不同构型机器人的运动学差异未系统评估。
2. **导航主导任务增益更显著**：对 manipulation-dominant 任务提升有限，说明运动条件化对操作子任务的影响机制有待深入分析。
3. **VGGT 锚点依赖离线预计算**：离线重建需要冻结的 VGGT-1B，增加了数据处理的算力门槛。
4. **未探索更多视角组合**：当前仅用 Body+Hand 双视角，未来可研究多视角或引入深度信息进一步提升重建鲁棒性。

## 研究启发与可借鉴点
1. **当前帧相对动作表征的设计**：将全局 SE(3)/SE(2) 转换为当前帧相对增量，天然消除 VO 漂移的全局一致性错误，值得在其他视觉里程计+策略结合的场景中复用。
2. **仅对部分动作组施加条件化的残差隔离思想**：base 与 manipulation 残差解耦可避免错误信号污染精细操作，对多自由度机器人策略微调具有通用参考价值。
3. **离线融合预训练几何先验校正 VO 漂移**：用冻结 VGGT 作为稀疏锚点、轻量回归器做坐标映射，兼顾精度与训练成本，可用于其他需要 metric 监督的 VO 任务。
4. **数据划分与审计分离的设计**：明确区分 portable-data 监督与 robot-data 监督，使得数据来源贡献可量化审计，是数据集构建的可借鉴范式。
5. **多预测步长运动 token 的结构**：在 2/3/6/12 帧上分别积分并拼接 uncertainty 估计，为策略提供了多时间尺度的运动先验，比单步 prediction 更丰富。

## 关键术语表
**VOMMI**：Visual-Odometry-Conditioned Mobile Manipulation Interface，一种面向移动操作的便携式演示采集与 VLA 后训练框架。
**R2-VO**：Real-time and Refined Visual Odometry，论文提出的双模态运动重建模块，支持在线推理与离线轨迹校正。
**VLA（Vision-Language-Action）**：视觉-语言-动作模型，结合大语言模型与多模态感知的端到端机器人策略网络。
**SE(2)/SE(3)**：二维/三维特殊欧氏群，分别表示平面位姿（平移+偏航）与三维刚体位姿（平移+旋转）。
**VGGT-1B**：Visual Geometry Grounded Transformer，10 亿参数级的前馈几何重建模型，本文用作离线稀疏锚点来源。
**Flow Loss**：基于流匹配（flow matching）的动作预测损失，论文中结合动作掩码在有效维度上归一化计算。
**Residual Adapter**：在预训练策略旁附加的轻量残差分支，用于注入额外条件（如 VO token）而不破坏原有参数。
**Planar Compensation**：将 Body 轨迹投影到地面平面（SE(2)）后反向补偿 Hand 轨迹，实现 Body-Hand 视角的坐标系对齐。

## 可复现要素
- **数据集**：论文声明每个任务 500 条便携式演示（75 条独立 held-out），200 条机器人参考数据；代码/数据公开状态论文未明确声明。
- **代码**：论文未提及开源声明。
- **关键超参**：策略 chunk 长度 H=50，时钟频率 15 Hz；运动 token 预测步长 k∈{2,3,6,12}；Ridge 回归正则化系数 λ；router 权重 0.1。
- **硬件**：两个鱼眼 RGB 摄像头，估测成本 $100-200；评估平台为 DeepRobotics M20S + CM1 机械臂。
- **评估平台**：Intel Core i5-9600KF CPU，在线推理平均延迟 10.02 ms（P95: 13.76 ms）。
