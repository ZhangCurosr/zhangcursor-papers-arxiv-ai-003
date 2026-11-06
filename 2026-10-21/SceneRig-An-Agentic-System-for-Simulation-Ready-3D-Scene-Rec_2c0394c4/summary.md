---
title: "SceneRig-An-Agentic-System-for-Simulation-Ready-3D-Scene-Rec"
source: https://arxiv.org/pdf/2610.28585v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-06 08:43:28"
field: "机器人仿真与3D视觉重建"
keywords: ["single-image 3D reconstruction", "agentic system", "simulation-ready scene", "physics-grounded pose refinement", "robot policy evaluation", "real-to-sim"]
innovations: ["物理锚定编辑-验证循环：Move/Edit工具结合PhysX仿真实现姿态精化闭环", "支持有序稳定化：按场景图关系逐层稳定化避免全局漂移", "模块化生成器-验证器管线：几何规则门+视觉验证器双保险确保物理可信"]
benchmarks: ["RoboDojo/RoboLab 44张仿真渲染场景", "80张桌面图像基准（含真实工作区/AI生成/手机拍摄）"]
---

# 论文速读：SceneRig - An Agentic System for Simulation-Ready 3D Scene Reconstruction from Single Images

## 一句话总结
SceneRig 提出了一种基于视觉语言模型（VLM）的智能体系统，能够从单张图像重建出**几何精确、外观逼真且物理稳定**的仿真就绪3D场景；其核心是通过"编辑-验证"循环，将物理模拟反馈直接融入姿态精化过程。

## 研究问题与动机
- **机器人仿真环境规模化瓶颈**：真实机器人评估依赖昂贵的硬件和大量人工操作，仿真虽能大规模测试策略，但构建高保真、物理可信的场景需要大量手工工作。
- **单图重建存在"外观对、物理错"的问题**：现有方法（如 VIGA、REST3D、SimFoundry）在视觉相似度上表现不错，但事后物理稳定化（post-hoc settling）会导致物体偏离观测位置，产生"看起来正确但行为错误"的场景。
- **仅靠提升 VLM 能力无法解决物理稳定性**：实验表明，即使使用更强的 GPT-6 Astra 作为骨干，没有物理工具辅助的通用代理仍然无法实现稳定的物体放置。
- **需要端到端的物理反馈闭环**：单纯的几何/视觉优化不足以保证仿真就绪，需要智能体具备物理仿真工具调用能力，通过迭代式验证修正姿态。

## 核心贡献（创新点）
1. **SceneRig 智能体框架**：提出首个将视觉推理、几何约束和物理仿真紧密结合的单图3D场景重建系统，强调物理落地（physical grounding）而非仅视觉相似。
2. **编辑-验证循环中的物理工具集**：设计了 Move 和 Edit 两类工具——Move 工具通过粗到细的有界搜索（bounded coarse-to-fine search）在物理模拟中测试候选变换，Edit 工具允许直接场景代码编辑；两者均配合物理验收检查（防倾倒、碰撞等）。
3. **支持有序物理稳定化（support-ordered settling）**：按照场景图关系逐层稳定化物体，避免全局稳定导致的物体漂移和碰撞重排。
4. **模块化 Agent 场景构建管线**：几何/材料/光照构建由独立生成器-验证器配对完成，引入确定性规则检查门（check_rules gate）保证几何约束先于视觉评估。
5. **在机器人应用上显著优于基线**：与 SimFoundry 相比，开环轨迹回放成功率从 36% 提升至 80%，episode 级别策略一致性从 54% 提升至 69%（McNemar p=0.017）。

## 方法详解
SceneRig 分为四个主要阶段：

**阶段一：关系解析与度量初始化**
- 使用 VLM（Claude Opus 5）识别可移动物体 $\mathcal{O}$ 和根表面 $\mathcal{R}$，并建立支撑关系图 $G=(V, E)$。
- MolmoPoint-8B 定位每个实体，SAM3 提供实例掩码，恢复被遮挡/合并/分裂的物体。
- 使用 MoGe-2 估计度量深度和相机内参，拟合主支撑面作为重力对齐坐标系（$z=0$）。
- SAM3D 为每个物体重建带纹理网格，通过度量点云对齐初始变换，并估计物理属性（质量、摩擦系数等）。
- **支持有序稳定化**：按父-子关系依次放置，每个物体在 PhysX/Isaac Sim 中释放后运行 ICP 微调平移、旋转（yaw）和缩放。

**阶段二：Agent 场景构建（几何→材料→光照）**
- 三阶段共享同一 Blender 场景，每阶段配独立生成器-验证器。
- 几何阶段：构建根表面（table/floor/wall），使用 RANSAC 拟合平面确定位置和朝向，并添加确定性规则检查门（root placement、object support、interpenetration）。
- 材料与光照阶段：修改根表面材质/纹理和照明设置，不触碰物体姿态。

**阶段三：物理锚定的物体姿态精化**
- **Investigate 工具**：返回参考图/渲染图配对裁剪、几何测量（遮挡、大小、偏航角）、支撑信息。
- **Move 工具**：支持水平平移、均匀缩放、yaw 调整三种有界变换，通过粗到细搜索评估候选变换，评分函数（Eq.5）融合 IoU、深度误差、投影面积误差和 DINOv2 外观相似度。
- **物理验收**：变换后在 PhysX 中仿真，检查倾倒（capsize >45°）、过度漂移（>50mm）、新增穿透等；通过后更新得分需不低于基线。
- **Edit 工具**：当 Move 无法解决时，通过直接编辑场景代码执行协调变换，由智能体决定是否保留。
- **上下文管理**：压缩历史交互轮次（每4轮压缩一次），保留目标图像和阶段快照，**84.1% 的输入 token 命中 prompt cache**。

**阶段四：导出到 USD**
- 联合重力仿真确定最终变换，烘焙程序材质，导出 USD 文件（含碰撞几何、物理属性）。

## 实验与结果
- **数据集**：80 张桌面场景图像，其中 44 张为 RoboDojo/RoboLab 仿真器渲染（含 ground truth 物体身份和3D位置），其余包括真实机器人工作区、AI 生成图、手机拍摄等。
- **评估指标**：源视角 PSNR/SSIM/LPIPS/nCLIP；实例指标（ARI、FG-IoU、Count MAE）；3D布局（OSPA@5/@10、F1@10）；物理稳定性（5s内位移<50mm且速度归零）。
- **重建质量**：SceneRig 在所有源视角指标（PSNR=13.277、LPIPS=0.518、nCLIP=0.122）和 VLM 偏好评分（Completeness +0.895、Layout +2.597、Appearance +3.139）上均最优，且 80/80 场景全部通过稳定性测试。
- **实例与布局**：ARI=0.878（↑0.202 vs SimFoundry）、FG-IoU=0.819、Count MAE=0.27、OSPA@5=2.4cm、F1@10=0.93。
- **消融实验**：移除 Move 工具仅保留 Edit（code edits only），轨迹回放成功率从 80% 降至 44%；直接 SAM3D 原始姿态+事后5s稳定化，OSPA@5 为 3.0cm 且仅 2/80 场景保持原位。
- **不同 VLM 骨干**：GPT-6 Astra 在外观上最强（PSNR*=14.535），但 SceneRig 框架在最强骨干下仍超越 VIGA/Codex 在布局稳定性和实例指标上的表现。
- **机器人应用**：
  - 开环轨迹回放：SceneRig 20/25 vs SimFoundry 9/25（p=0.001）。
  - Real-to-sim 策略评估（π₀.₅ + LingBot-VLA 2.0 共100 episode）：SceneRig 一致性 69% vs SimFoundry 54%（McNemar p=0.017），Pearson r=0.92 vs 0.84。

## 相关工作脉络
1. **单图场景重建**：VIGA [52] 采用生成器-验证器循环重建3D场景；SAM3D [6] 预测物体几何、纹理和相机相对布局；REST3D [26] 用交叉熵优化位姿；CAST [51] 结合遮挡补全与姿态精化。本文区别：引入物理仿真验证工具进行迭代修正。
2. **Agent 3D 场景创建**：VIGA [52]、SEIG [12]、SceneConductor [18] 均使用 VLM 智能体；Scenethesis [22]、SAGE [47] 结合语言规划与几何约束。本文区别：不仅生成场景，还确保物理稳定性，工具具有度量感知的搜索能力。
3. **物理锚定 real-to-sim**：SimFoundry [35] 是最近邻基线，从视频重建可编辑数字孪生；Agentic Real2Sim [4] 从录制交互创建可仿真孪生；TableVerse [41]、ϕ-Scene [20] 用几何平移或 clearance 优化处理碰撞。本文区别：Agent 直接调用物理仿真工具迭代修正而非仅事后稳定化。
4. **机器人仿真评估**：PolaRiS [16]、REALM [37] 评估 sim-real 相关性；SIM-PLER [21] 校准控制器与匹配背景/纹理。本文贡献：提供更高保真度的场景重建以支持更可靠的政策评估。

## 局限性与未来方向
- 复杂房间场景（非桌面）仍存在挑战：遮挡严重、深度歧义、频繁接触关系。
- SAM3D 缺乏 PBR 材质，限制 photorealism。
- 当前表示将每个物体视为单一刚体，未恢复关节结构或柔性变形。
- 未来方向：改进深度估计模型、提升 image-to-3D 模型、探索可形变物体和关节连接。

## 研究启发与可借鉴点
1. **物理工具作为代理能力扩展**：将 Move/Edit 工具与物理仿真闭环结合，使 VLM 能"感知"物理后果而非仅视觉相似，对任何需要物理可信度的 3D 生成任务具有迁移价值。
2. **支持有序稳定化策略**：按 parent-before-child 顺序逐个稳定物体，避免全局仿真导致的场景漂移，可借鉴于其他需保持空间关系的 3D 重建任务。
3. **Prompt 缓存设计模式**：通过分层上下文管理（近期完整日志 + 旧轮压缩 + 物体状态表），实现 84.1% token 缓存命中率，对长会话多工具调用的 agent 系统有参考价值。
4. **几何规则门 + 视觉验证器双保险**：先通过确定性 check_rules 过滤违反几何约束的候选，再由 VLM 做视觉评估，兼顾效率与质量。
5. **轨迹回放作为端到端评估**：不仅用 PSNR/LPIPS 等视觉指标，还通过 open-loop trajectory replay 验证物理场景对机器人行为的保真度，是更实用的评测范式。

## 关键术语表
- **Simulation-ready 3D scene**：不仅外观逼真，且物体姿态物理稳定、碰撞几何正确、可直接导入物理引擎用于机器人策略评估的3D场景。
- **Edit-and-validate loop**：智能体基于物理仿真反馈（如倾倒、穿透检测）迭代调整物体姿态并验证的闭环流程。
- **Support-ordered settling**：按照场景图中父-子支撑关系顺序（先父后子）依次稳定化物体的方法，避免同时仿真导致的位置漂移。
- **Bounded coarse-to-fine search**：在预设边界内从粗到细搜索候选变换（平移/缩放/yaw），通过评分函数评估并物理仿真验证的方法。
- **Physical acceptance check**：仿真验证步骤，检查候选变换是否导致倾倒（>45°）、过度漂移（>50mm）、新穿透（>8mm）等不稳定情况。
- **DINOv2 patch feature matching**：利用 DINOv2 ViT 提取的前背景特征比对渲染图与参考图，用于判断物体朝向（front-back）和 yaw 方向。
- **Open-loop trajectory replay**：将真实机器人录制的关节轨迹直接导入仿真环境回放，无需策略干预，验证场景是否支持复现真实行为。
- **Metric 3D point map**：由 MoGe-2 等模型输出的带真实尺度信息的稠密3D点云，用于度量对齐和初始姿态估计。

## 可复现要素
- **数据集**：80张桌面图像基准，44张含 ground truth（RoboDojo/RoboLab）；论文未提及单独公开数据集，但提供了基准描述和评估协议。
- **代码**：GitHub 开源 — https://github.com/Red-Fairy/SceneRig
- **模型权重**：MoGe-2、SAM3、SAM3D、MolmoPoint、DINOv2、CoACD 等模型依赖预训练权重；SHARP 和 GPT-Image-2 为开源/商用 API。
- **关键超参**：RANSAC 200次/≤5000点/0.01内点距离；ICP 最多60次迭代/yaw ±30°/缩放0.85–1.18；物理仿真600步/60Hz；Move 搜索步长±{0.05,0.15,0.4}×物体尺寸/yaw ±{5,15,40}°/缩放 {0.8,0.9,1.1,1.25}；阈值 ε_translation=0.01/ε_scale=0.01/ε_yaw=0.012；倾倒阈值45°/漂移阈值50mm/穿透阈值8mm（恶化>5mm则拒绝）。
- **API 成本**：SceneRig 每场景约 $4.22（Claude Opus 5），GPT-6 Astra 版本约 $6.24。
