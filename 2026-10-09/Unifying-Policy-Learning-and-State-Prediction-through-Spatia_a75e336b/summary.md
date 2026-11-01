---
title: "Unifying-Policy-Learning-and-State-Prediction-through-Spatia"
source: https://arxiv.org/pdf/2610.12172v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-01 18:56:41"
field: "机器人操作与视觉-语言-动作建模"
keywords: ["spatial language modeling", "policy learning", "world model", "Push-T", "imitation learning", "action-state unification"]
innovations: ["用语法化坐标-语义 token 统一动作与状态预测", "random-play 过渡预训练与专家联合训练共享参数", "grammar 驱动的 action-only 闭环解码"]
benchmarks: ["Push-T simulation", "real-robot Push-T"]
---

# 论文速读：Unifying-Policy-Learning-and-State-Prediction-through-Spatial-Language-Modeling

## 一句话总结
论文提出 **Spatial Language Modeling（空间语言建模）**，通过一套共享词汇的坐标-语义token序列，让单个自回归 Transformer 同时学习动作生成与动作条件状态预测；在 Push-T 仿真与真实机器人上，取得优于 ACT、Diffusion Policy 等基线的成功率与目标覆盖率。

## 研究问题与动机
- 现有策略学习方法（如 ACT、Diffusion Policy）仅从专家示范中学习动作分布，难以利用非专家交互数据中的场景几何变化信号。
- 世界模型（如 DreamGen、LingBot-VA）虽然预测未来观察，但与动作生成往往使用不同接口或架构，难以用同一套参数同时优化动作与状态转移。
- 需要一个统一的离散空间表示，使得"动作目标"和"预测的未来状态"在同一 token 词表中可被同一个自回归模型以 next-token 目标学习，从而实现动作生成与状态预测的参数共享。
- 希望将随机交互（random-play）的过渡信息与专家示范的动作监督统一训练，提升闭环控制的鲁棒性与目标覆盖率。

## 核心贡献（创新点）
1. **语法结构化的空间语言表示（SLS）**：将场景几何、目标、动作目标和未来状态统一编码为坐标与语义 token 序列，使动作生成与状态预测共享输出词表与 next-token 目标。
   - 区别：不同于以往用视觉/潜变量空间做预测，本文在离散坐标层面建立统一接口，动作坐标与状态坐标同义同序。

2. **两阶段联合训练方案**：先用随机播放数据做动作条件化的过渡预训练（掩码动作坐标loss），再用专家示范进行动作与状态联合训练，两段更新同一组参数。
   - 区别：不同于"先训世界模型再训策略"的两段独立流程，本文全程共享模型与损失，实现过渡监督与动作监督的联合收敛。

3. **动作-only 闭环解码**：推理时通过语法状态机排除未来状态块，仅自回归采样动作目标，新观测刷新历史即可闭环控制。
   - 区别：与需要联合解码动作+状态的策略相比，执行时只输出动作 token，兼顾训练时的几何监督与推理时的控制效率。

4. **统一模型支持额外预测设定**：给定初始场景与已提供动作轨迹，同一模型可预测连续场景状态，验证动作-几何关系的联合学习有效性。

## 方法详解
- **词表设计**：将平面工作区离散为规则网格，每个格点为一个坐标 token（`COORD ∈ Q_coord`）；语义 token（`Q_semantic = {state, pusher, tblock, goal, move-to}`）标识元素类型与命令。
- **序列格式（SLS）**：由上下文无关文法（CFG）组织，核心生产规则：
  - `STATE → state pusher CB⁺ tblock CB⁺ goal CB⁺`
  - `MOVE_TO → move-to CB [STATE]`
  - `SEQ → STATE MOVE_TO*`
  其中 `CB⁺` 是按固定行列优先排序的坐标序列，表示实体轮廓。
- **训练目标**：自回归 next-token 交叉熵 `L(θ) = -Σ m_i log p_θ(s_i | s_<i)`。
  - 专家示范阶段：所有位置参与损失。
  - 随机播放预训练阶段：仅掩码紧跟 `move-to` 的动作坐标 token；后续 STATE 以 30% 概率保留，作为状态转移目标。
- **闭环控制**：历史前缀包含三段连续的 `STATE–MOVE_TO`，末尾以 `move-to` 提示下一动作坐标；语法状态机在解码时剔除 STATE 块，生成动作后由机器人执行，新观测更新历史。
- **推理效率**：在 A100 上 prefilling 约 7000 tokens/s、decoding 约 800 tokens/s；一次 100 步动作生成约需 0.25s。

## 实验与结果
- **数据集/设置**：仿真使用公开 200 条 Diffusion Policy 专家演示；真实机器人采集 40 条成功专家演示与 20 条 random-play（共 5229 帧），工作区离散为 128×128 网格，模型为 462M 参数 Qwen2，4×A100 训练 5 小时。
- **仿真结果**：Max coverage 达到 0.95，与 DP-C/DP-T 并列最高；Avg（最后 10 个 checkpoint 均值）达到 0.93，优于 DP-C 的 0.91 和 DP-T 的 0.79。
- **真实机器人结果**：成功率为 0.80（对比 DP-Coord 0.45、DP-Tag 0.50、ACT 0.65）；Max coverage 为 0.95（对比 DP 变体 0.89、ACT 0.86）。
- **消融**：加入未来状态块使成功率从 0.05 提升至 0.65、Max 从 0.67 提升至 0.92；加入 random-play 预训练进一步使成功率从 0.65 提升至 0.80、Max 从 0.92 提升至 0.95。
- **状态预测**：给定手工标注轨迹，10 步后 coverage 为 0.92、IoU 为 0.85，误差随步数累积但仍保持较大几何重叠。

## 相关工作脉络
- **Action Chunking Transformers / Diffusion Policy**：从示范学习动作分布；本文在相同 Push-T 任务上以离散坐标序列替代连续/扩散动作输出，并通过联合状态监督提升成功率。
- **DreamGen / LingBot-VA / LeWorldModel**：世界模型通过视频或潜变量预测做控制；本文在坐标级别显式建模场景转换，并与动作共享单一自回归参数。
- **Decision Transformer / Behavior Transformers**：将序列建模引入决策；本文沿用自回归思路，但用任务 grammar 统一结构化和离散化的几何/动作 token。
- **PolyGen / MeshGPT / Pix2Seq**：用语言模型处理几何/检测序列；本文将该思路迁移到机器人操作，以 coordinate token 作为动作与状态的公共接口。
- **PointLLM / 3D-LLM / Scene-LLM**：语言模型用于 3D 场景理解；本文聚焦 2D 平面操作并强调"动作-几何联合训练"，而非纯理解。

## 局限性与未来方向
- 真实系统依赖校准的 AprilTag 定位与已知物体几何，未评估端到端从原始传感数据学习。
- 语法为手工设计，grid 分辨率在精度、词表大小与序列长度之间存在权衡；扩展到新实体/动作类型需修改表示。
- 仅在平面 Push-T 上评估，泛化到不同物体几何、其他操作任务和三维场景仍需验证。
- 自回归状态 rollout 存在误差累积，长时域预测仍需改进。

## 研究启发与可借鉴点
1. **共享坐标词表统一动作与状态**：将动作目标与未来状态放在同一 token 体系，并以 mask 区分 loss 贡献，可在其他离散化操作任务中复用。
2. **两阶段联合训练（random-play 过渡预训练 + 专家联合监督）**：利用非专家交互数据的几何转移信号辅助策略训练，值得在数据稀缺场景下推广。
3. **grammar 驱动的控制解码**：推理时通过语法状态机构造 action-only 序列，既能享受训练时的联合监督，又能保持执行时的简洁输出，可作为"训练/推理不一致但安全"的设计范式。
4. **低维坐标替代图像输入的轻量化策略**：用轮廓坐标序列作为输入，减少了对高分辨率视觉骨干的依赖，适合计算受限的真实机器人部署。

## 关键术语表
- **Spatial Language Modeling (SLS)**：将场景几何、目标、动作与未来状态编码为坐标与语义 token 序列的语言建模框架。
- **COORD / Q_coord**：表示离散网格位置的坐标 token，构成模型输出的公共词表部分。
- **Q_semantic**：标识状态块、实体类型与运动命令的语义 token 集合（如 state、pusher、move-to 等）。
- **CFG / Grammar**：规定 SLS 中各元素排列与组合规则的上下文无关文法，支撑序列化与解析。
- **Random-play pretraining**：用随机交互数据训练模型预测动作后的状态变化，动作坐标被掩码不参与 loss。
- **Next-token objective**：自回归模型的标准交叉熵目标，统一用于动作与状态预测的训练。
- **Coverage**：目标区域内物体占据比例，用于衡量 Push-T 完成任务的程度（阈值 0.95 判定成功）。

## 可复现要素
- **数据集**：仿真使用公开 200 条 Diffusion Policy 专家演示；真实机器人由作者自行采集（40 条专家、20 条 random-play）。
- **代码/权重**：论文未明确声明开源链接。
- **关键超参**：128×128 网格、462M 参数 Qwen2（7 层、隐藏维度 1536、24 头）、RoPE 位置编码、最大上下文长度 13000、随机状态保留概率 30%、训练 200 epoch、每 5 epoch 评估一次、real-robot 使用 3 帧历史、baseline 使用 1–2 帧。
