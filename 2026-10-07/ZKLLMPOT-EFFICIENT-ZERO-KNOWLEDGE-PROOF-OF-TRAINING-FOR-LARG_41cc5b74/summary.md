---
title: "ZKLLMPOT-EFFICIENT-ZERO-KNOWLEDGE-PROOF-OF-TRAINING-FOR-LARG"
source: https://arxiv.org/pdf/2610.08258v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-08 02:26:23"
field: "可验证与隐私保护机器学习"
keywords: ["zero-knowledge proof", "proof of training", "large language model", "commit-then-challenge", "sumcheck", "lookup argument", "Transformer circuit"]
innovations: ["将zkPoT关系从训练轨迹验证弱化为前向评估，证明成本独立于训练步数", "commit-then-challenge两阶段协议防止针对挑战数据的后验过拟合", "算子级融合与mixed-radix softmax/lookup编译使1.1-13B Transformer可在数十秒内出证"]
benchmarks: ["OPT-1.3B", "Llama-1.1B", "Qwen2.5-1.5B", "DeepSeek-Coder-1.3B", "13B scale extrapolation"]
---

# 论文速读：ZKLLMPOT-EFFICIENT-ZERO-KNOWLEDGE-PROOF-OF-TRAINING-FOR-LARG

## 一句话总结
论文提出 zkLLMPoT，一种面向大语言模型的零知识训练结果认证框架，通过"提交后挑战"(commit-then-challenge)协议将认证计算简化为对已提交权重的单次前向评估，证明成本与训练迭代次数无关；在 1.1–13B 参数模型上实现 41–131 秒证明时间、<0.5 秒验证时间。

## 研究问题与动机
- **隐私与可信审计的矛盾**：LLM 训练数据与模型权重往往涉及隐私或商业机密，机构通常拒绝向外部审计方公开；审计方仅能拿到 model card 或 API 端点，无法独立验证声称的训练结果。
- **已有 zkPoT 方案在 Transformer 规模不可行**：现有证明训练轨迹的方案需编码完整反向传播与优化器更新，如 Kaizen 在 VGG-11 上每迭代约 15 分钟，Confidential-DPproof 在 CIFAR-10 上约 100 小时；在 LLM 规模下完全不可承受。
- **推理完整性证明不覆盖训练结果**：zkLLM、zkGPT 等工作仅保证单次推理查询的完整性（检测"降级服务"），无法证明 checkpoint 在审计方选定数据上达到声明的能力或政策指标。
- **挑战数据提前暴露可导致过拟合**：若审计序列在权重提交前已知，训练方可针对挑战数据调整权重，使评估失去对真实训练效果的表征意义。

## 核心贡献（创新点）
1. **面向训练结果的"前向评估"认证形式化**：将 zkPoT 的关系从"运行指定学习算法于指定数据集"替换为"已提交权重在审计方目标函数下的损失值"，显式排除反向传播与优化器，使证明代价独立于训练步数。
2. **commit-then-challenge 两阶段协议**：训练方先固定架构并将权重 commitments 广播，随后审计方才公布挑战序列；训练方无法在看到挑战后修改权重，确保评估反映真实训练效果。
3. **算子级 Transformer 电路合成与融合**：基于 sumcheck 与 lookup 论证组合成完整 Decoder-only Transformer 前向电路；线性层用超立方体内积 sumcheck，非线性层（Norm、FFN激活、softmax）通过公共查找表+对数导数成员资格论证实现，避免浮点约束编码。
4. **可定制的审计目标接口**：默认 next-token NLL；审计方可按需更换为交叉熵、组公平差距、成员推断信号或安全评分，仅改变末端目标 gadget，前向电路与权重 commitment 复用。
5. **跨四族模型的经验可扩展性证明**：在 OPT / Llama / Qwen2.5 / DeepSeek-Coder 上 1.1–1.5B 参数模型证明 41–59 秒、13B 模型 131 秒、验证 <0.5 秒；证明规模较 Prior 最大系统（VGG-11 / MobileNet-v2）提升约两个数量级参数量，且代价不随训练步数增长。

## 方法详解
- **统一架构形式化**：将 OPT、Llama、Qwen2.5、DeepSeek-Coder 视为同一 Decoder-only Transformer 家族 A 的实例，共享骨架公式：
  - 前向：$h^{(0)} = \text{Embed}_\theta(x)$，$h^{(\ell+1)} = \text{Block}_\theta^{(\ell)}(h^{(\ell)})$，$z = h^{(L)}W_{\text{out}}$。
  - Block：预归一化残差结构 $u = h + \text{Attn}_\theta(\text{Norm}(h);\text{Pos})$，$h' = u + \text{FFN}_\theta^\Phi(\text{Norm}(u))$。
  - 目标：masked next-token NLL $\mathcal{L}_{\text{LM}}(\theta;x,m) = -\sum_{i=1}^{n-1} m_{i+1}\log p_\theta(x_{i+1}|x_{\le i})$，覆盖继续预训练与 SFT。
- **声明关系**：$v = \mathcal{L}(f_{A,\theta}(S))$，其中 $\mathcal{A}$、$\mathcal{L}$、$S$ 为公共输入，$\theta$ 与中间激活 $H$ 为见证，$v$ 为声明标量。
- **承诺设计**：使用 Hyrax 风格 Pedersen 多项式承诺 [[·]]；权重 commitment 一次性计算、跨审计复用；激活 commitment 每次审计重算；张量以固定定点量化后模 $p$ 展平为 $\{0,1\}^n \to \mathbb{F}_p$ 的表。
- **约束电路 $C_{A,\mathcal{L}}$ 构造**：
  - **线性层/注意力乘积**：$Y=XW$ 写成 $\sum_b X(u_B,b)W(b,u_O)$，作为 $\mu=2$ 的 multilinear sumcheck。
  - **残差连接**：线性，直接拆分 claim。
  - **元素级非线性 (GELU/SiLU/ReLU)**：通过 folding + logup 对公共双列表 $T_{\text{in}},T_{\text{out}}$ 做成员资格 lookup。
  - **Norm (LayerNorm / RMSNorm)**：行统计内积（$\mu=3$）+ range check + 查找 $1/\sqrt{\cdot}$。
  - **softmax**：逐行平移后按 mixed-radix 分解，每位查表 $T_k[d]=\lfloor\theta_k\exp(\cdot)\rceil$，最后 Hadamard 积 $\mu=K-L+1=4$，为全协议最高多项式次数。
  - **FFN / 输出头**：复用线性 matmul sumcheck。
- **协议交互流程 (Algorithm 1)**：
  1. 审计方固定 $\mathcal{L}$ 与接受区间 $[\underline{v},\overline{v}]$，训练方编译电路 $C_{A,\mathcal{L}}$。
  2. 训练方发送 [[θ]]。
  3. 审计方发送挑战序列 $S$（公开输入）。
  4. 训练方评估 $v=C_{A,\mathcal{L}}(\theta,S)$，发送 [[v]] 与各层激活 commitments [[H]]。
  5. 从 $\ell=L$ 到 $1$，训练方通过 gadget 将输出 claim 归约为输入 claim 与权重 claim，发送 sumcheck 转录。
  6. 审计方逐层校验链式一致性 $(u_{\ell-1},y_{\ell-1})$。
  7. 审计方 open 检查 θ、H，并在公共表 $S$ 上直接求值残余 claim；重写计算各非线性查找表 $T$。
  8. 审计方接受当且仅当所有检验通过且 $v\in[\underline{v},\overline{v}]$。
- **安全性**：归纳反拓扑序自输出向输入传播 false claim，最终落在 committed table 的 opening 或公共 $S$ 的直接校验上；理论安全约 127 bit（保守 union bound）。

## 实验与结果
- **评测规模**：OPT-1.3B、Llama-1.1B、Qwen2.5-1.5B、DeepSeek-Coder-1.3B，外加 0.125B–13B 的族系外推。
- **基线对比**：
  - zkPoT (logistic regression, 1K 参数): commit 3690s, prove 600s。
  - Kaizen (VGG-11, 10.1M): commit 218s, prove 882s / 迭代。
  - ZKAudit (MobileNet v2, 3.5M): prove 328s。
  - DPproof (logistic, 41K): prove 100h（交互、无 proof 对象）。
- **zkLLMPoT 主结果 (T=512)**：
  - OPT-1.3B: commit 175s, prove 58.8s, verify 428ms, proof 17.0 MiB。
  - Llama-1.1B: commit 138s, prove 53.2s, verify 423ms, proof 17.3 MiB。
  - Qwen2.5-1.5B: commit 206s, prove 42.4s, verify 416ms, proof 19.9 MiB。
  - DeepSeek-Coder-1.3B: commit 171s, prove 41.0s, verify 406ms, proof 16.9 MiB。
- **算子分解 (Figure 2)**：线性投影 / FFN / 输出头融合 sumcheck 合计 4.1–6.0s；Norm 2.6–3.1s；elementwise 3.0–5.0s；attention score matmul 4.6–10.3s；softmax 占证明时间 56–64%，为最大开销。
- **扩展性 (Figure 3)**：commit 约 124–142s / 十亿参数；covered proving 随参数 104× 增长仅 8.3× (15.7s→130.6s, 0.125B→13B)；verify 始终 <0.5s。
- **目标灵活性 (§4.6)**：生物医学/CS 微调下 held-out loss 下降约 0.3 nats，远大于定点表示分辨率；output head 只占 prove 时间 0.1–1.5%。
- **最强结果**：DeepSeek-Coder-1.3B 在 41.0s 证明时间、<0.5s 验证，相对 prior 最大系统（VGG-11 / MobileNet-v2）参数量提升约 2–3 个数量级且代价显著更低。

## 相关工作脉络
- **zkPoT 系列 (Garg et al. 2023; Abbaszadeh et al. 2024; Shamsabadi et al. 2024)**：目标是证明"checkpoint 来自指定学习算法在指定数据上的执行"，需编码反向传播与优化器，代价随训练步数与模型尺寸双重增长；本文将其关系弱化为单步前向评估，彻底切断与训练步数的依赖。
- **zkML 推理完整性 (zkCNN / Mystique / zkLLM / zkGPT)**：关注 per-query serving integrity（检测下线降级服务），不处理审计方事后选定挑战、不绑定训练结果；本文与之互补，关注 checkpoint 级 outcome attestation。
- **乐观 / 第三方审计 (ZKAudit, Srivastava et al. 2024)**：允许审计方重跑训练或使用确定性硬件以保证可复现性，代价是暴露权重 / 数据或依赖外部复刻；本文在纯零知识下达成可验证，隐私更强但仅覆盖前向性质。
- **Privacy-preserving ML audit (Waiwitlikhit et al. 2024)**：面向成员推断 / 公平性审计，但其框架假设审计方能观测或重执行训练过程；本文在同一审计目标下无需重执行与可见训练数据。
- **SNARK 编译技术 (Thaler 2022; Wahby et al. 2018 Hyrax; Habock 2022 logup)**：本文复用 sumcheck、Hyrax PCS、logup lookup 等基础原语，创新在于将其以算子级 fusion 与 commit-then-challenge 整合到 LLM Transformer 尺度。

## 局限性与未来方向
- **仅覆盖 Decoder-only Transformer 前向**：未处理 MoE、编码器-解码器架构、长上下文滑动窗口 attention 等变体；Attention softmax 占主导开销，仍是性能瓶颈。
- **证明不绑定训练轨迹**：仅验证 checkpoint 在给定目标下的数值，不证明该权重是否由声明的算法 / 数据产生；可能存在对抗性初始化或蒸馏产出达到同样目标的情形。
- **定点量化引入表示误差**：虽实验中 0.3 nats 目标差距远大于精度，但在高精度审计任务下仍需评估量化误差对 soundness 的实际影响。
- **当前为两方协议**：不支持多训练方联合 checkpoint、多审计方挑战或下游验证者复用的复杂场景。
- **代码尚未开源**：Reproducibility Statement 声明审稿后开源，当前评估依赖作者自研 CUDA/C++ 原型。

## 研究启发与可借鉴点
- **"前向评估替代轨迹验证"的思路**：对于任何"声称训练结果满足某事后指标"的场景，均可考虑以 commit-then-challenge 把证明负担从 $O(\text{steps} \times \text{model})$ 降到 $O(\text{model})$，值得迁移到 RLHF / DPO / GRPO 等偏好对齐的 checkpoint 认证。
- **算子级融合 + 查找表分解的非线性编译策略**：将 softmax 按 mixed-radix 分解至多位查表、Norm 用内积 + 单点逆平方根查找，是 Transformer 在 SNARK 友好的有限域上高效实现的通用模板，可复用于 ViT / Mamba / MoE 等架构的编译。
- **Hyrax-style PCS + logup lookup 的组合**：对大规模张量 commitment 使用矩阵折叠降低 opening 成本，并用对数导数统一表达 elementwise 非线性，构成一套可插拔的 "linear=sumcheck, nonlinear=lookup" 电路合成范式。
- **审计目标的接口化设计**：将 custom loss / fairness gap / membership / safety score 全部映射到同一前向电路的末端 gadget，只需更换一个可插拔模块即可复用整条 prove 链，便于后续扩展新审计协议。
- **性能分解视角指导优化优先级**：本文算子分解揭示 softmax 占 56–64% 证明时间，提示未来工作应优先改进 attention 的 SNARK-friendly 实现（如低秩近似 / 稀疏 attention / kernelized softmax）以获得更大收益。

## 关键术语表
- **zk-SNARK**：零知识简洁非交互知识论证，证明方可使验证方相信某 NP 关系成立而不泄露见证，且验证代价远低于重执行。
- **Commit-then-challenge**：协议顺序让承诺方先绑定电路与权重，再由挑战方公布输入，防止承诺方针对挑战数据做后验调整。
- **Sumcheck**：Thaler 等人的代数协议，将高维表之和的证明降维为单点求值，开销与表维度成对数关系。
- **Logup / Lookup 论证**：基于对数导数恒等式的成员资格证明，用于在 ZK 中高效约束 elementwise 非线性查找表（GELU、softmax 等）。
- **Hyrax-style PCS**：将长向量按矩阵布局折叠为行承诺，再以内积论证完成点开放，使 opening 代价从线性降为对数。
- **Mixed-radix softmax**：将 logits 按混合进制位分解后逐位查表相乘，以低次数 lookup 逼近指数与非线性归一化。
- **Outcome attestation**：仅声明 checkpoint 对审计目标的数值满足性，区别于 proving training trajectory 的强关系。
- **Next-token NLL**：标准 LLM 预训练/SFT 损失，masked 求和仅计算 response 位置，用于衡量生成质量与记忆风险。

## 可复现要素
- **数据集**：未使用私有数据集；模型配置来自公开 OPT / Llama / Qwen2.5 / DeepSeek-Coder ；实验使用合成 fixed-point 张量与公开 benchmark。
- **代码/权重**：论文声明审稿后开源实现（CUDA/C++ prover + C++ verifier），包含协议代码与复现实验脚本；当前版本未提供。
- **关键超参**：
  - 序列长度 T=512、batch=1。
  - 有限域 BLS12-381 scalar field（$|\mathbb{F}_p|\approx 2^{255}$）。
  - 定点缩放因子 $s$（公开，具体值未在主文给出）。
  - Softmax mixed-radix 位宽 $d=4$；Hyrax 折叠平衡 $n_1\approx n_2$。
  - 安全参数保守 union bound 给出约 127 bit 安全。
