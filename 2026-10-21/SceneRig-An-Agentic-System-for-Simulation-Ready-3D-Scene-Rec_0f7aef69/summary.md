---
title: "SceneRig-An-Agentic-System-for-Simulation-Ready-3D-Scene-Rec"
source: https://arxiv.org/pdf/2610.28585v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 17:23:53"
field: "机器人仿真与 3D 场景重建"
keywords: ["real-to-sim", "single-image 3D reconstruction", "agentic system", "robot simulation", "physics-grounded pose refinement", "open-loop trajectory replay"]
innovations: ["提出 SceneRig 智能体系统，将 VLM 视觉推理与几何测量、PhysX 物理验证耦合，从单图重建 simulation-ready 3D 场景", "设计 investigate-edit-validate 循环 + 有界 Move 工具（IoU/深度/DINOv2 多信号搜索 + 物理稳定验收），显著提升 3D 摆放精度", "引入确定性 check_rules gate 与 per-object state prompt caching，使 84.1% 输入 token 命中缓存并减少无效迭代"]
benchmarks: ["80-scene tabletop benchmark (44 simulator-grounded + 36 diverse)", "π₀.₅ / LingBot-VLA 2.0 real-to-sim policy evaluation (50 episodes/policy)", "Open-loop trajectory replay (25 teleoperated demonstrations)"]
---

# 论文速读：SceneRig: An Agentic System for Simulation-Ready 3D Scene Reconstruction from Single Images

## 一句话总结
SceneRig 是一个智能体驱动的"真实到仿真"（real-to-sim）系统，能从单张 RGB 图像重建出**几何精确、物理稳定**的可编辑 3D 场景，直接支持机器人策略评估与开环轨迹回放。

## 研究问题与动机
- 机器人策略评估依赖大规模仿真，但手动搭建逼真场景成本极高；从单图重建是 Scaling 仿真环境的有前景路线。
- 既有"单图→3D"方法（如 VIGA、REST3D、SceneConductor、SimFoundry）通常**追求视觉相似性**，但重建出的场景经物理 settling 后对象会漂移/碰撞，导致仿真中行为失真，无法满足机器人任务需求。
- 现有 VLM 智能体（如 GPT-6 Astra via VIGA 或通用 Codex harness）在缺乏显式几何/物理工具时，难以通过试错获得稳定且位置正确的摆放。
- 目标：构建一个能将**视觉推理、几何测量、物理仿真**耦合的智能体系统，使重建场景兼具外观高保真与物理可用。

## 核心贡献（创新点）
1. **提出 SceneRig，首个面向机器人仿真就绪的"单图→可编辑 3D 场景"智能体系统**：与现有方法仅追求视觉相似不同，本文把物理稳定性作为第一等优化目标，在 80 张 diverse 图像上全部通过稳定性测试（100%），同时提升 instance segmentation、object-centric appearance、3D placement 指标。
2. **设计了模块化 harness + 四阶段工作流（关系图→对象创建→环境构建→pose refinement）**：区别于 VIGA/SEIG 等把验证全交给 verifier 的做法，本文在初始化阶段引入确定性 `check_rules` gate，强制通过几何/支撑/穿透规则后才进入视觉审查。
3. **发明 "investigate–edit" 物理地面 pose refinement 循环**：Agent 不是直接让 VLM 猜测坐标偏移，而是调用 Move（有界粗→细搜索 + 物理验证）和 Edit（场景代码直接编辑 + PhysX 回传结果）工具，配合 Eq.(5)–(7) 定义的 IoU/深度/DINOv2 对齐得分 + 物理检查，实现闭环修正；移除 Move 工具的 ablation 使 trajectory replay 成功率从 80% 跌至 44%。
4. **在下游机器人任务上建立严格评测**：在 π₀.₅ 与 LingBot-VLA 2.0 两个策略上，SceneRig 的 open-loop trajectory replay 成功率 80% vs SimFoundry 36%，episode 级 agreement 69% vs 54%（McNemar p=0.017），Pearson r 达 0.92。

## 方法详解
系统流程分四阶段（图 2）：

### 3.1 关系解析与度量初始化
- **Relational parsing**：VLM（Claude Opus 5）给出可移动对象集合 O、根表面 R 及支撑关系 E_sup、结构关系 E_surf；MolmoPoint-8B 对每个实例做定位，SAM3 提供候选 mask；通过 synonym pool、点提示选择、重叠/包含检查、LanPaint + Qwen-Image-Edit 去遮挡、LingBot-Depth 完成深度等方式恢复 mask。
- **Metric initialization**：MoGe-2 估计单目度量深度与相机内参；RANSAC 拟合主支撑平面得到重力对齐坐标系（Eq.3）；SAM3D 重建每个对象 mesh，平移+均匀缩放对齐到 3D 点云；SceneSmith 范式估计物理属性（质量、摩擦）。
- **Support-ordered physical settling**：按支撑树 parent-before-child 顺序逐个 drop 到 PhysX/Isaac Sim，再用 ICP（平移 + 尺度 + yaw）贴合 masked 3D 点；带 guarded fallback（平化碰撞底、调整质心、增大摩擦阻尼）解决倾斜。

### 3.2 智能体场景构建（共享 Blender 场景，固定阶段）
- 三个阶段（geometry → material → lighting）各自 pairing 一个 generator agent 和一个 verifier agent，均输入 scene graph + 当前 scene snapshot + 允许编辑范围；stage-entry snapshot 跨轮保留，避免重复加载。
- **几何 agent**：从 fitted planes 构造 root surfaces；新增确定性 `check_rules` gate（结构/pose/接触三层），在送 verifier 前必须通过。
- **材质/光照 agent**：只改根表面纹理/roughness 与灯光曝光，不触碰对象摆放，保证 pose refinement 阶段背景稳定。
- 每阶段设预算上限，超预算回退 best-effort 产物进入下一阶段。

### 3.3 Physics-Grounded Object Pose Refinement
- **Investigate tool**：返回参考图/渲染图 paired crop、几何测量（silhouette overlap、投影尺寸、yaw、前后证据 via DINOv2 patch cosine）。
- **Move tool（有界搜索）**：六种动作（±x/y/xy 平移、uniform scale、yaw、180° flip）；粗→细候选集，候选排列使用 Eq.(5) 得分：

  $$
  A_o^{(a)}(X)=\begin{cases}
  \mathrm{IoU}_o(X)-0.4D_o(X)-0.3Q_o(X), & a=\text{translation/scale}\\
  \mathrm{cIoU}_o(X)+0.2F_o(X), & a=\text{yaw}
  \end{cases}
  $$

  其中 D 为相对深度误差，Q 为投影面积误差，F 为 DINOv2 patch 余弦相似度。
- **物理验证与接受（Eq.7）**：改进超过阈值 ε_a 且通过穿透/倾覆/漂移检查后，再做 post-settling 重评分；维持或回滚前一状态；支持父对象被移动时传播重设子对象。
- **Edit tool**：当有界移动不足时，允许 agent 直接写 scene code 做协调修改；工具自动 resolve overlap、support-ordered settle 后返回渲染 + 物理诊断，由 agent 决定是否保留。
- **Context management & prompt caching**：旧轮压缩为日志块，保留最近 5–8 轮完整历史；每对象 state table 紧跟 cache breakpoint 之后；评测中 84.1% 输入 token 命中 cache（图 9）。
- **USD export**：最终联合 gravity simulation 烘焙 pose，导出含碰撞体与物理属性的 USD，供 Isaac Sim 直接使用。

## 实验与结果
- **数据集**：80 张 tabletop 图像（44 张带 ground-truth 的 RoboDojo/RoboLab 仿真渲染 + 18 张真实机器人工作台 + 6 手机照 + 8 互联网图 + 4 AI 生成图）。
- **基线**：VIGA、REST3D、SceneConductor、SimFoundry、SAM3D*（共享分割/相机/对象但保留 SAM3D 原始位姿）、Codex harness（GPT-6 Astra + 通用 Blender 工具）。
- **主要量化结果**：

  | 指标 | SceneRig | 最佳基线（提升）|
  |---|---|---|
  | Source-view PSNR ↑ | **13.277** | SimFoundry 12.060 (+1.217) |
  | LPIPS ↓ | **0.518** | VIGA 0.653 (-0.135) |
  | nCLIP ↓ | **0.122** | SAM3D* 0.078 (差)；SimFoundry 0.194 (+0.072) |
  | 稳定性 Scene% ↑ | **100.0%** | SimFoundry 98.8% |
  | FG-IoU ↑ | **0.819** | SimFoundry 0.589 (+0.230) |
  | OSPA@5 ↓ | **2.4 cm** | SimFoundry 3.3 cm (-0.9) |
  | F1@10 ↑ | **0.93** | SimFoundry 0.82 (+0.11) |

- **机器人下游**（π₀.₅ + LingBot-VLA 2.0，50 episodes/policy）：
  - Episode-level agreement：SceneRig **69%** vs SimFoundry 54%（McNemar p=0.017）；Cohen's κ 0.38 vs 0.09（Δ=0.29，CI [0.08, 0.50]）。
  - Pearson r（5 任务 success rate）：**0.92**（SceneRig）vs 0.84（SimFoundry）。
  - Open-loop trajectory replay（25 演示）：**20/25（80%）** vs SimFoundry 9/25（36%），全部 Task 均更强（McNemar p=0.001）。
- **Ablation**：
  - Remove Move（code edits only）：replay 成功率跌至 44%（Table 11）。
  - 换更强 VLM（GPT-6 Astra）仅改善外观，3D layout 提升有限（Table 12）；SceneRig 仍显著优于同 backbone 的 VIGA/Codex（Table 4/14）。

## 相关工作脉络
1. **VIGA（ECCV 2026）/ SEIG（arXiv 2026）**：VLM agent + generator-verifier 的 3D 场景重建；差异在于二者无物理工具与 metric 落地，仅靠迭代渲染反馈，SceneRig 引入确定性质心物理验证 + 有界搜索工具。
2. **REST3D（SIGGRAPH Asia 2026）**：用 cross-entropy optimization 优化 pose；SceneRig 用 agent 驱动 tool-use loop 并在 PhysX 中实时验证。
3. **SimFoundry（CoRL 2026）**：视频到 real-to-sim，支持 policy evaluation；差异在 SceneRig 专注单图输入并加入 pose refinement 阶段，后者缺失物理化对象级修正。
4. **SceneConductor（arXiv 2026）/ SAGE（CVPR 2026）/ Scenethesis（ICLR 2026）**：面向室内生成/控制；共同点是 agent orchestration，但前者目标为"好看室内"而非"机器人可用仿真"。
5. **ϕ-Scene（arXiv 2026）/ SimuScene（arXiv 2026）/ TableVerse（arXiv 2026）**：通过几何平移、simulated displacement、collision resolution 做物理校正，但均为单步/离线优化，缺乏 agent 的 inspect-edit-validate 闭环。
6. **SAM3D（CVPR 2026）/ MolmoPoint（arXiv 2026）/ MoGe-2（NeurIPS 2025）**：提供 segmentation、point grounding、metric depth 等基础组件；SceneRig 将其整合进 workflow，并在此基础上加 physics settling 与 pose refinement。

## 局限性与未来方向
- **复杂房间场景**：遮挡、深度歧义、频繁接触使重建困难（当前针对 tabletop 优化）。
- **PBR 材质缺失**：SAM3D 输出不含 PBR，限制 photorealism；需未来 image-to-3D 模型支持。
- **刚性单一对象假设**：未恢复 joint 结构与可变形体，不适用于软体/连杆机构。
- **VLM 依赖强**：性能高度依赖 Claude Opus 5 / GPT-6 Astra 等商用大模型的成本与可用性；换弱模型会拉低外观质量。
- 作者展望未来工作包括改进 depth estimation 和 image-to-3D foundation models。

## 研究启发与可借鉴点
1. **"规则 gate + verifier" 双保险设计**：初始化阶段先过确定性 check_rules（支撑/穿透/结构约束），再送 VLM verifier 做视觉审查；比纯视觉 feedback loop 更节省 token、避免无效迭代——可移植到任何 Blender/USD 生成管线。
2. **有界 search + 物理验收的 pose refinement 范式**：Move 工具在候选生成时融合 IoU/深度/DINOv2 多信号，再用 PhysX settle 后二次评分（Eq.7），这种"工具式搜索→仿真验算→阈值接受"的结构可作为 agentic manipulation planning 的通用模块。
3. **Prompt caching with per-object state table**：用块级压缩 + 断点后的可变态表实现 84.1% token 缓存命中，在保证上下文新鲜度的同时大幅降成本——对长对话 agent 系统有直接参考价值。
4. **Nearby-view 合成 + 图像修复作为可视化辅助**：SHARP 生成高斯新视角 + GPT-Image-2 修复 artifact，作为 agent 的"多视图证据"而非 ground truth，降低了单视 ambiguities；类似思路可用于 3D 场景质量诊断。
5. **实例一致性追踪（entity ID 贯穿全程）**：从关系解析、mask recovery、到 USD export 均保留 persistent identity，使下游 trajectory replay 能一对一映射真实→仿真对象——在 multi-object 任务评估中至关重要。

## 关键术语表
- **Real-to-sim（真实到仿真）**：把现实 workspace 的重建映射到物理仿真环境，以低成本、可扩展地评估机器人策略。
- **Physical stability test**：在 PhysX 中运行 5 s，若所有对象位移 < 50 mm 且末速度为零则通过。
- **Support-ordered settling**：按支撑图 parent-before-child 顺序释放对象到模拟器，避免整体同时 drop 导致的漂移。
- **Investigate–edit loop**：pose refinement 的核心循环——Investigate 工具诊断错位证据，Agent 选择 Move/Edit 工具修正，物理验证后回传结果。
- **OSPA（Optimal Subpattern Assignment）**：多目标定位误差度量，这里用 5/10 cm cutoff 惩罚错位与未匹配对象。
- **Open-loop trajectory replay**：把真实机器人记录的时间戳关节轨迹与 gripper 命令直接注入仿真控制器，不做策略查询或轨迹自适应，检验场景几何/物理与真实是否一致。
- **Check_rules gate**：初始化阶段前的确定性规则检查（结构/pose/接触三层），失败即拒绝入 verifier。
- **nCLIP**：1 − cos_sim(CLIP_ViT-B/32(reference), CLIP_ViT-B/32(render))，衡量语义级外观一致性。

## 可复现要素
- **数据集**：80 张图像 benchmark（44 simulator-grounded + 36 diverse）；具体来源含 RoboDojo / RoboLab renderings、真实机器人工作站照、手机照、网络图与 AI 生成图。**论文未明确公开**原始图像集，但项目页与 GitHub 仓库应含评估脚本。
- **代码**：已开源，地址 https://github.com/Red-Fairy/SceneRig（Appendix A 声明）。
- **权重**：使用预训练模型 MoGe-2、SAM3、SAM3D、MolmoPoint-8B、LanPaint、Qwen-Image-Edit-2509、LingBot-Depth、SHARP、DINOv2 ViT-S/14，均为公开模型；物理仿真用 PhysX / Isaac Sim。
- **关键超参**（Table 8）：输入长边 1536 px；RANSAC 200 次/5000 点/0.01 inlier；occluder 去遮 20 diffusion steps / guidance 4；CoACD 普通 resolution 50、支撑面 resolution clip([ℓ_max/0.003], 100, 200)；Physics 120 Hz / 600 步 settle；ICP 最多 60 次/±30° yaw/0.85–1.18 scale；Move 粗探针 {±0.05, ±0.15, ±0.4}×物体尺寸 / yaw ±{5,15,40}° / scale {0.8,0.9,1.1,1.25}；ε_translation=0.01、ε_scale=0.01（有 size hint 时 0.005）、ε_yaw=0.012；穿深阈值 8 mm、恶化 > 5 mm 拒；倾倒阈值 45°、可滚物体水平漂移 50 mm；每对象 budget 3 investigations + 5 moves，Edit 软上限 3 次；整体 pose refinement 会话 min(70, 8+5|O|) 轮。
- **VLM 默认**：Claude Opus 5；对比实验用 GPT-6 Astra。
- **模拟器**：Isaac Sim（PhysX 后端）；渲染主要用 Blender/Cycles，部分基线用 OmniGibson。
