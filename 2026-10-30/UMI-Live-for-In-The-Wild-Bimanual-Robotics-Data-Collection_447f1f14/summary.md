---
title: "UMI-Live-for-In-The-Wild-Bimanual-Robotics-Data-Collection"
source: https://arxiv.org/pdf/2610.39385v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-06 22:24:08"
field: "机器人模仿学习与数据采集"
keywords: ["UMI", "bimanual robotics", "in-the-wild data collection", "VLA policy", "LiDAR-inertial SLAM", "flow matching"]
innovations: ["自包含双臂手持UMI硬件与异步并行处理流水线", "UMI手持演示与机器人遥操作数据的统一TCP表示与混合微调框架"]
benchmarks: ["Orange-to-bowl dual YAM arm task", "UMI-Live public dataset (3,255 episodes, ~40 hours)"]
---

# 论文速读：UMI-Live for In-The-Wild Bimanual Robotics Data Collection

## 一句话总结
本文提出了UMI-Live，一套完整的开源双臂手持式演示数据采集系统，将无固定跟踪设备的野外采集与自动化异步数据处理流水线结合，最终生成可直接用于VLA策略微调训练的LeRobot格式数据集。

## 研究问题与动机
1. **现有UMI系统的复现门槛高**：后续UMI变体（如FastUMI、UMI-3D等）缺乏清晰的硬件构建与校准指南，以及配套的自动化数据处理软件。
2. **数据采集与处理串行化导致效率低下**：每次录制后需人工执行重建、格式转换等步骤，操作员无法在后台处理进行中时继续采集新演示。
3. **双臂协调数据采集缺乏统一方案**：多数现有UMI系统仅支持单臂，或需外接定位基站（如YUBI的固定 tracking rig、ActiveUMI的背包式计算机），难以在野外灵活部署。
4. **跨环境演示数据的策略迁移验证不足**：手持演示与目标机器人评估场景存在光照、位置差异，如何有效融合两类数据仍需系统验证。

## 核心贡献（创新点）
1. **自包含双臂手持硬件设计**：每个设备集成广角全局快门相机、RoboSense Airy LiDAR-IMU、Raspberry Pi 5及可拆卸电池，无需外部跟踪设备或环境标定即可在野外采集双臂演示。
   - *本质区别*：不同于HiFi-UMI（头戴式双目惯性传感）或ActiveUMI（VR头显+背包计算机），UMI-Live将所有传感、计算、供电单元置于手柄内，实现真正的移动化采集。
2. **异步并行处理流水线**：操作员单键启停录制，后台流水线自动完成时间同步校验、GLIM SLAM轨迹重建、ArUco机械爪开合估计、LeRobot格式组装，支持中断续传。
   - *本质区别*：区别于传统串行处理流程，本系统允许操作员连续采集，已完成片段并行重建，大幅提升数据产出效率。
3. **统一的观测与动作表示兼容UMI与机器人遥操作**：将UMI手持演示（腕部图像+重建TCP轨迹）与YAM双臂遥操作数据（关节角+前向运动学推导TCP）映射到同一14维双状态表示与查询相对动作chunk格式。
   - *本质区别*：打破UMI数据与真实机器人遥操作数据的格式壁垒，使两种来源能无缝混合微调。
4. **公开40小时双臂数据集与端到端可复现系统**：发布3,255个episodes、跨越橙子放置/倒水/折毛巾三个任务的完整数据集，附带硬件设计文件、软件代码及复现指南。
   - *本质区别*：相比DROID、ABC-130K等纯遥操作数据集，本数据集源自无机器人环境的手持演示，填补了"in-the-wild"双臂数据的空白。

## 方法详解
**硬件架构**：每个UMI-Live单元包含bearings-supported trigger与gear pivots驱动的平行jaw机械爪、PLA+刚性基底+TPU柔性接触层的手指、ArUco标签（用于爪开合估计）、RoboSense Airy LiDAR-IMU、800×600@30fps全局快门广角相机、Raspberry Pi 5、LCD状态屏及可拆卸电池模块（续航1.5小时，单体重1.42 kg）。

**传感器同步**：Chrony同步Pi时钟，gPTP同步LiDAR与Pi，相机曝光时间戳映射至主机单调时钟；服务器端通过仿射映射校正LiDAR嵌入时间与Pi包到达时间之间的偏移，筛选跨设备偏移>5 ms的无效episode。

**轨迹重建与爪开合估计**：各设备独立运行GLIM进行LiDAR-惯性SLAM，固定传感器到TCP偏移量将LiDAR本体运动转换为TCP轨迹；ArUco标签中心经鱼眼相机标定模型反投影后，结合固定相机-标签几何关系估算物理爪宽，缺失值线性插值，经校准闭合/张开端点映射为归一化$g \in [0,1]$。

**数据组装**：以主相机时间线为基准，线性插值位置、四元数球形插值旋转；从站视频取对应时刻最近帧；单手臂状态$\mathbf{s}_t^i = [\mathbf{p}_t^i; \boldsymbol{\phi}_t^i; g_t^i] \in \mathbb{R}^7$（位置m、旋转向量rad、归一化爪开度），双臂拼接为$\mathbf{s}_t^{\mathrm{bi}} \in \mathbb{R}^{14}$，每帧观测$o_t = (I_t^L, I_t^R, g_t^L, g_t^R, \ell)$，任务指令$\ell$。

**动作表示与训练目标**：动作目标为查询相对TCP pose chunk与绝对爪开度，即$\Delta T_{t,k}^i = (T_t^i)^{-1}T_{t+k}^i$，旋转编码为6D向量（取旋转矩阵前两行），每个爪动作$\mathbf{a}_{t,k}^i \in \mathbb{R}^{10}$，双爪拼接为$A_t \in \mathbb{R}^{H \times 20}$；采用flow matching损失$\mathcal{L}_{\mathrm{flow}} = \mathbb{E}\left[\frac{1}{20H}\|f_\theta(o_t, X_{t,\tau}, \tau) - (\epsilon - A_t)\|_F^2\right]$。

**推理部署**：策略服务端返回action chunk，本地dual-YAM客户端通过I2RT执行前/逆运动学，将相对预测映射至机器人目标pose，经IK解算为关节指令，Gripper命令按校准值映射。

## 实验与结果
**数据集**：发布3,255个双臂episodes、约39.7小时（30 Hz），涵盖orange placement、water pouring、towel folding三个任务；评估任务为双i2RT YAM机械臂的六橙入碗任务（随机初始位姿，成功标准：全部六颗橙子入碗）。

**基线与对比**：
- 发布版MolmoAct2-BimanualYAM checkpoint（零样本，0/10成功）
- $\pi_{0.5}$ fine-tuned with 95% UMI / 5% teleoperation → 9/10
- $\pi_{0.5}$ fine-tuned with 90% UMI / 10% teleoperation → **10/10**
- MolmoAct2 fine-tuned with 75% UMI / 25% teleoperation → 8/10

**训练设置**：12,000次优化步、global batch size=32、AdamW（$\pi_{0.5}$全参微调，MolmoAct2仅训VLM LoRA rank=64+全action expert）、学习率$2.5\times10^{-5}$（$\pi_{0.5}$）/ $5\times10^{-5}$（MolmoAct2 adapter）/$10^{-4}$（action expert）、cosine decay至峰值0.1、gradient clipping norm=1、action horizon H=24@30Hz。

**结论**：手持UMI演示（与评估场景不同位置、不同光照）配合少量目标环境遥操作数据，可使$\pi_{0.5}$与MolmoAct2在物理双臂任务上达到高成功率，UMI数据占比越高表现越稳定（90/10 > 95/5）。

## 相关工作脉络
1. **原始UMI（Chi et al., RSS 2024）**：提出手持机械爪+腕部视觉+VI轨迹重建的范式；UMI-Live在此基础上扩展为双臂、集成LiDAR、实现异步流水线。
2. **FastUMI（CoRL 2025）**：硬件与追踪工具扩展；UMI-Live更强调野外免标定、自包含供电与并行处理。
3. **UMI-3D（2026）**：引入LiDAR为中心的3D重建；UMI-Live同样使用LiDAR-惯性但聚焦双臂协同与端到端数据集发布。
4. **HiFi-UMI（2026）**：头戴式双目惯性传感+硬件同步+直接相对位姿估计；UMI-Live选择手柄集成方案，牺牲部分姿态保真换取便携性。
5. **YUBI（2026）**：强调指尖对齐与双臂灵巧操作，但依赖 stationary/portable配置；UMI-Live专注无外部跟踪的完全移动采集。
6. **DROID / ABC-130K（2024-2026）**：大规模遥操作数据集；UMI-Live填补了"in-the-wild手持双臂演示"的数据空白，二者可互补混合。

## 局限性与未来方向
- **仍需一次性标定**：相机内参、LiDAR-IMU到TCP的固定外参、ArUco标签几何需预先标定。
- **LiDAR-惯性重建质量依赖场景几何与运动**：弱纹理、动态场景、低速运动可能导致SLAM退化。
- **设备重量较大**（1.42 kg/单元），长时间操作可能影响舒适性。
- **实验验证有限**：仅单任务、单seed、固定UMI/teleop混合比例，未隔离UMI数据与遥操作数据的独立贡献。
- **未来方向**：测量轨迹重建精度、处理吞吐率与延迟、操作效率；扩展至更多任务、多seed、多样化数据混合比例；支持UMI-only与teleop-only对照实验。

## 研究启发与可借鉴点
1. **异步并行处理流水线设计**：操作员录制与后台重建/格式转换解耦，支持中断续传，可复用于其他数据采集系统以提升吞吐量。
2. **UMI与遥操作数据的统一表示**：将手持VI轨迹与真实机器人关节空间通过FK/IK映射到同一TCP-action表示，使异构演示可混合微调，为跨源数据融合提供范式。
3. **ArUco标签用于机械爪开合估计**：轻量级视觉标记方案替代力矩/编码器测距，成本低、易实现，可推广至其他手持设备。
4. **广角全局快门相机适配无顶视场景**：单腕部超广角镜头捕获workspace全局上下文，减少对固定监控摄像头的依赖。
5. **LeRobot格式直接对接策略训练**：从原始录像到可训练数据集的全自动转换，降低数据工程门槛，可与本团队的数据 pipelines 结合。

## 关键术语表
- **UMI (Universal Manipulation Interface)**：无需真实机器人即可通过手持机械爪采集演示数据的接口范式。
- **LiDAR–Inertial SLAM**：结合激光雷达点云与惯性测量单元进行三维定位与建图，本文使用GLIM算法。
- **TCP (Tool Center Point)**：机械爪末端工具中心点，作为轨迹表示的参考位姿。
- **Flow Matching**：基于连续扩散过程的生成建模方法，本文用于策略的动作chunk预测训练。
- **LeRobot**：开源机器人学习库，提供统一数据集表示与策略训练/推理工具链。
- **VLA (Vision-Language-Action) Model**：联合视觉、语言与动作输入的端到端机器人策略模型，如$\pi_{0.5}$、MolmoAct2。
- **Query-Relative Action**：相对于当前时刻位姿的未来动作增量表示，提升策略对初始状态的泛化性。
- **YAM (Your Arm Model)**：六自由度轻量机械臂平台，本文使用dual i2RT YAM进行物理评估。

## 可复现要素
- **数据集**：3,255 episodes、~39.7小时，已发布于Hugging Face。
- **代码/硬件设计**：项目网站提供完整硬件CAD、构建与标定指南、软件（录制、异步处理、LeRobot导出、双YAM遥操作、策略训练与推理）。
- **关键超参**：global batch size=32、12,000更新步、warmup=600、cosine lr decay、$\pi_{0.5}$全参微调（peak lr=$2.5\times10^{-5}$）、MolmoAct2 LoRA rank=64/alpha=16/dropout=0.05（VLM adapter lr=$5\times10^{-5}$，action expert lr=$10^{-4}$）、action horizon H=24@30Hz、FP32/BF16精度。
- **传感器**：RoboSense Airy LiDAR-IMU、800×600全局快门广角相机、Raspberry Pi 5。
- **算法**：GLIM（SLAM）、Chrony/gPTP（时间同步）、ArUco（爪开合估计）。
