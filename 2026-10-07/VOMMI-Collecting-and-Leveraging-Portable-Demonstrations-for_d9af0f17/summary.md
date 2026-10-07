---
title: "VOMMI-Collecting-and-Leveraging-Portable-Demonstrations-for"
source: https://arxiv.org/pdf/2610.08220v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-07 10:56:14"
field: "移动操作策略学习"
keywords: ["mobile manipulation", "portable demonstration", "visual odometry", "VLA", "residual adapter", "low-cost robotics"]
innovations: ["无标定双视角便携数据采集协议", "R2-VO在线-离线协同运动重建", "动作组残差适配器实现导航-操作解耦条件"]
benchmarks: ["OpenPI 0.5", "GR00T N1.6", "X-VLA", "DROID-SLAM", "DPVO"]
---

# 论文速读：VOMMI-Collecting-and-Leveraging-Portable-Demonstrations-for

## 一句话总结
本文提出VOMMI（Visual-Odometry-Conditioned Mobile Manipulation Interface）框架，仅需两个廉价鱼眼RGB摄像头即可采集便携演示数据，并通过R2-VO模块从RGB重建局部运动，将其作为视觉-语言-动作（VLA）策略的条件输入，从而在零机器人标定条件下实现长时程移动操作策略训练。

## 研究问题与动机
- 移动操作（Mobile Manipulation）需要协调大范围导航与精细末端操作，但高质量示范数据稀缺。
- 现有便携数据采集方法依赖昂贵专用硬件（如IMU、VR头显、GoPro/iPhone），且常需繁琐的人-机运动学标定，难以低成本扩展。
- 直接使用视觉里程计（VO）轨迹会因累积漂移和旋转误差导致数据质量下降，尤其对长时程移动操作任务不利。
- 如何将仅含RGB的便携演示无缝接入VLA训练，并在部署时不依赖额外传感器（如IMU、全局定位），仍是开放挑战。

## 核心贡献（创新点）
- **提出无标定的双视角便携数据采集协议**：仅需两个廉价鱼眼RGB摄像头（胸前+手持）同步采集Body与Hand视图，无需目标机器人或人-机运动学对应标定。
- **设计R2-VO双分支运动重建模块**：在线因果分支提供低延迟局部运动token用于实时推理；离线分支结合VGGT稀疏几何锚点校正VO漂移，提升轨迹重建精度。
- **定义动作组残差适配器接口**：将视觉运动token仅注入基础（Base）动作残差路径，操作（Manipulation）残差保持独立，实现导航与操作的解耦条件控制。
- **仅用便携数据训练即超越机器人演示训练的效果**：在三个真实机器人任务上，仅用便携演示训练的VLA策略比使用机器人演示训练的OpenPI 0.5平均成功率提升8.3个百分点，基础速度误差降低18.2%。

## 方法详解
- **双视图采集与时序对齐**：演示者佩戴胸前鱼眼相机（Body视角）和手持夹爪相机（Hand视角），以15 Hz同步记录RGB视频，存储语言指令、内参、TCP信号等，形成与机器人兼容的episode。
- **局部跨本体动作表示**：Base动作以当前Body帧的平面速度表示，EE目标以当前TCP帧的相对SE(3)增量表示，避免全局坐标系漂移影响。
- **Body-Hand平面对齐补偿**：将Body轨迹投影到SE(2)平面，补偿Hand的TCP运动，得到当前Body相对目标，使双视图在统一参考系下对齐。
- **R2-VO运动估计**：
  - 在线分支：从光流、几何线索等提取206维描述子，经7帧时序窗口聚合为317维，由极端随机树回归器预测单步运动，累积得到多视野（2/3/6/12帧）局部运动token（29维）。
  - 离线分支：将因果VO轨迹与1Hz VGGT-1B稀疏锚点对齐，通过低容量Ridge映射校正累积漂移，提升轨迹精度。
- **VO条件动作组适配**：将运动token经LayerNorm-MLP投影后，通过有效性门控加入Base残差路径；Manipulation残差仅依赖主体隐藏状态，两路径独立门控融合，零初始化保证训练稳定。

## 实验与结果
- **数据集**：每个任务500条便携演示轨迹，其中75条作为固定RGB-VO评估集，425条用于模型开发；同时准备200条机器人演示作为参考。
- **评估基线**：VO方法对比DROID-SLAM、DPVO、MASt3R-SLAM、CUT3R、VGGT-SLAM 2.0；策略对比OpenPI 0.5、GR00T N1.6、X-VLA。
- **主要结果**：
  - R2-VO离线分支较各流最优基线平均降低绝对轨迹误差24.6%。
  - 在线分支在15 Hz下Body平移误差0.26 cm（均值），延迟10.02 ms，满足实时要求。
  - 真实机器人任务（跨越障碍物取水果、进出房间开瓶、开柜放盒）中，VOMMI平均成功率58.3%，较OpenPI 0.5的50.0%提升8.3个百分点。
  - 基础速度MAE为0.036 m/s，较机器人演示训练的M20政策降低18.2%；末端平移误差0.170 cm，旋转误差0.453°。
- **最强提升**：仅在便携数据上微调的策略，在导航密集任务上超越使用机器人演示的基线，验证了便携数据的有效性。

## 相关工作脉络
- **UMI系列**（UMI、UMI-on-Legs、Mobile UMI）：使用手持夹爪+腕部相机采集操作演示，但未解决移动基座的VO条件融合，且依赖专用硬件。
- **HoMMI/EgoHumanoid**：利用人类第一人称演示学习全身移动操作，但需要昂贵的VR头显或头部穿戴设备。
- **DECOWAM**：通过分离基座与手臂潜在空间并条件基座速度提升 loco-manipulation，但未利用便携RGB演示。
- **VLA微调方法**（OpenPI、GR00T、X-VLA）：聚焦于模型架构扩展，缺乏针对便携演示的专门运动重建与条件接口设计。
- **视觉SLAM/VO**（DROID-SLAM、DPVO、VGGT）：提供全局一致轨迹重建，但计算开销大、延迟高，不适合直接作为策略实时输入。

## 局限性与未来方向
- 仅在DeepRobotics M20S平台验证，跨本体泛化能力待进一步评估。
- 便携式数据在操作主导任务中提升有限，导航主导任务收益更显著。
- 离线漂移校正依赖VGGT等预训练几何模型，需额外存储与计算资源。
- 未讨论复杂动态场景或光照剧烈变化下的鲁棒性。

## 研究启发与可借鉴点
- **解耦条件设计**：将外部感知信号（如VO）仅注入特定动作组残差，避免干扰无关分支，值得在全身体控制任务中复用。
- **在线-离线协同重建**：实时推断与离线校准分离，兼顾低延迟与高精度，为其他策略学习中的数据管道设计提供参考。
- **低成本硬件协议**：仅需双RGB相机与时间戳同步，大幅降低数据采集门槛，可推广至更多机器人平台。
- **局部相对表示**：使用当前帧相对增量而非绝对位姿，天然抑制全局漂移，适用于长时程任务。

## 关键术语表
- **VOMMI**：Visual-Odometry-Conditioned Mobile Manipulation Interface，一种基于视觉里程计条件的移动操作便携演示收集与学习框架。
- **R2-VO**：Real-time and Refined RGB motion reconstruction module，提供在线运动token与离线漂移校正轨迹的双分支重建模块。
- **VLA**：Vision-Language-Action model，结合视觉、语言与动作输出的多模态机器人策略模型。
- **Visual Odometry (VO)**：视觉里程计，通过连续图像估计相机运动的技术，此处指从RGB视频恢复平面运动。
- **Residual Adapter**：残差适配器，在预训练VLA中添加轻量级适配层，仅修改特定动作维度。
- **Planar Compensation**：平面对齐补偿，将Body运动投影到SE(2)平面并补偿Hand TCP运动，实现双视图统一。
- **Current-Relative Action**：当前相对动作，以当前帧为参考的局部运动增量，减少对全局坐标系的依赖。
- **Action-Group Residual**：动作组残差，将残差路径按Base和Manipulation分组，分别注入不同条件信号。

## 可复现要素
- **数据集**：论文未明确说明是否公开，但提到500轨迹/任务的演示数据。
- **代码/权重**：论文未提及开源信息。
- **关键超参**：策略时钟15 Hz，action chunk长度50步，R2-VO树回归器32棵树，VGGT锚点频率1 Hz，残差适配器门控学习率未明确。
