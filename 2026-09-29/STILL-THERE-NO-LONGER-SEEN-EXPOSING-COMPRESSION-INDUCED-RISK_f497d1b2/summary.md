---
title: "STILL-THERE-NO-LONGER-SEEN-EXPOSING-COMPRESSION-INDUCED-RISK"
source: https://arxiv.org/pdf/2609.35002v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 22:09:49"
field: "多模态大模型安全与鲁棒性"
keywords: ["Vision-Language Models", "Adversarial Attacks", "Token Compression", "Robustness", "Compression-Specific Failure"]
innovations: ["提出CSF配对定义量化压缩诱导风险", "设计仅依赖encoder的CIRA攻击实现跨配置压缩选择性攻击", "提出TCS跨视图选择稳定化防御机制"]
benchmarks: ["POPE", "TextVQA", "MME"]
---

# 论文速读：STILL THERE, NO LONGER SEEN: EXPOSING COMPRESSION-INDUCED RISK IN LARGE VISION-LANGUAGE MODELS

## 一句话总结
本文定义了视觉token压缩特有的对抗性失败（CSF），并提出CIRA攻击框架，仅通过vision encoder白盒访问即可在未知压缩器和压缩预算下诱导压缩路径失败，同时保持完整token推理基本正确。

## 研究问题与动机
- **压缩引入的风险归属问题**：现有鲁棒性评估仅测试完整token推理，无法区分哪些对抗性失败是由压缩本身引入的，哪些是底层模型固有的漏洞。
- **压缩特定失败（CSF）的量化缺失**：缺乏配对评估准则——即同一输入在完整推理正确但压缩后失败——来精确归因压缩带来的安全风险。
- **实际部署中的攻击约束**：部署方通常不了解压缩器类型和具体压缩预算，攻击方法需要在黑盒压缩条件下仍能有效诱导压缩失败。
- **压缩改变证据可见性**：视觉token压缩通过选择、聚合和紧凑表示改变推理过程中可用的视觉证据，可能导致关键证据被错误丢弃。

## 核心贡献（创新点）
- **压缩风险归因框架**：提出干净的配对CSF定义，通过控制反事实实验证明保留集分配因果影响压缩正确性，且恢复效果与被置换证据的表示漂移呈负相关。
- **CIRA攻击方法**：设计仅依赖vision encoder的压缩诱导风险攻击，结合全局选择劫持（GSH）重新分配token优先级，并通过隐藏证据保留（HEP）限制表示漂移，实现压缩选择性攻击。
- **跨配置迁移能力**：同一扰动图像无需重新优化即可在不同压缩器（选择型、剪枝型、合并型）和压缩预算下诱导压缩特定失败，平均CSFR达20.35%。
- **TCS防御机制**：提出跨视图选择稳定化防御，利用平移不变性抑制攻击诱导的优先级重分配，使标准CIRA的CSFR降低81.4%。

## 方法详解
- **CSF定义**：对于清洁可评估集合$\mathcal{S}_K=\{i:c_i(x_i)=1, c_{i,K}(x_i)=1\}$，压缩特定失败定义为$\mathcal{F}_K^{\mathrm{CSF}}=\{i\in\mathcal{S}_K:c_i(x_i^{\mathrm{adv}})=1, c_{i,K}(x_i^{\mathrm{adv}})=0\}$。
- **GSH（全局选择劫持）**：基于encoder侧代理优先级分数$\mathbf{s}(x)$，最大化对抗优先级与清洁逆优先级的标准化对齐：$\mathcal{L}_{\mathrm{GSH}}(\delta)=\mathrm{Align}_{\epsilon_s}(\mathbf{s}^a, \mathcal{R}(\mathbf{s}^c))$，促使清洁高优先级token降序、低优先级token升序。
- **HEP（隐藏证据保留）**：计算跨越候选保留边界的clean Top-K token的表示漂移$d_i=[1-\cos(\tilde{\mathbf{h}}_i^c,\tilde{\mathbf{h}}_i^a)]/2$，通过预算边际权重$w_i$加权：$\mathcal{L}_{\mathrm{HEP}}(\delta)=-\sum_i \tilde{w}_i d_i$，限制被置换证据的语义偏移。
- **联合优化**：$\max_\delta \mathcal{L}_{\mathrm{CIRA}}(\delta)=\mathcal{L}_{\mathrm{GSH}}(\delta)+\lambda\cdot\mathcal{L}_{\mathrm{HEP}}(\delta)$，约束$\|\delta\|_\infty\leq\epsilon$，使用投影符号梯度上升法，步长$1/255$，$\lambda=0.8$。
- **TCS防御**：在4个平移视图$\mathcal{V}=\{T_{0,0},T_{d,0},T_{0,d},T_{d,d}\}$上计算对齐的优先级分数，取跨视图rank quantile共识$\bar{q}_i$的Top-K作为选择集合。

## 实验与结果
- **实验设置**：LLaVA-v1.5-7B为主模型，评测POPE、TextVQA、MME各1000样本；压缩器VisionZip/VisPruner/PruMerge/FastV；预算$K\in\{32,64,128,192\}$。
- **最强结果**：CIRA平均CSFR达**20.35%**，而VEAttack为5.49%、CAGE为5.92%；Full ASR仅为**6.92%**，远低于VEAttack的45.95%和CAGE的50.20%。
- **跨模型泛化**：在Qwen3-VL-8B和InternVL3.5-8B上同样有效，TextVQA上CSFR提升最显著（如VisionZip@32达51.65%）。
- **机制验证**：GSH驱动压缩特定失败（移除GSH后CSFR从18.22%降至3.68%）；HEP限制表示漂移并大幅降低Full ASR（移除后从6.92%升至23.10%）。
- **防御效果**：TCS使标准CIRA平均CSFR从18.22%降至3.39%（相对降低81.4%）；Adaptive CIRA可部分恢复至12.70%。

## 相关工作脉络
- **VEAttack**：下游无关的vision encoder攻击，目标是最小化编码器输出与参考答案的对齐，不关注压缩选择性，Full ASR较高（~46%）。
- **CAGE**：针对未知压缩设置的攻击，通过目标token survival概率优化，但缺乏对压缩特定失败的显式建模。
- **CAA**：利用下游问题和语言模型白盒访问的灰盒攻击，虽然Full ASR更低（3.32%），但CSFR仅3.24%，远低于CIRA。
- **SAP/Robust Pruning**：防御侧工作，通过安全感知剪枝或鲁棒性导向剪枝缓解压缩漏洞，未涉及攻击视角的系统分析。
- **CIRA定位**：唯一在encoder白盒、无压缩器/预算知识条件下，通过优先级重分配+证据保留实现压缩选择性攻击的方法。

## 局限性与未来方向
- **单图推理限制**：当前评估仅限静态图像，视频、多图和多轮对话场景下的证据跨帧/跨轮次分布未研究。
- **压缩机制覆盖不足**：未涵盖自适应剪枝、学习摘要、可恢复路由、空间扰乱等复杂压缩机制的归因分析。
- **任务正确性≠安全**：CSF基于任务正确性定义，未涉及安全对齐破坏或多模态越狱场景。
- **防御对抗性局限**：TCS为确定性公开方法，Adaptive CIRA可部分恢复攻击效果，实际部署需更强防御。

## 研究启发与可借鉴点
- **配对评估范式**：CSF的配对定义（完整vs压缩）为压缩加速模块的安全评估提供了可复用的归因框架。
- **优先级重分配策略**：GSH的逆优先级对齐设计可作为通用技术迁移至其他token选择/聚合架构的安全分析。
- **表示漂移约束**：HEP的分层证据保留思想可应用于设计抗压缩攻击的鲁棒压缩器训练策略。
- **跨视图共识防御**：TCS利用空间不变性稳定token选择，为多视图一致性防御提供了可借鉴的设计模式。
- **开源可复现**：代码已开源，可直接用于后续压缩安全基准测试和防御方法对比。

## 关键术语表
- **CSF（Compression-Specific Failure）**：压缩特定失败，指在完整token推理正确但在压缩推理中失败的对抗样本。
- **CIRA（Compression-Induced Risk Attack）**：压缩诱导风险攻击，仅通过vision encoder优化图像扰动以诱导压缩路径失败。
- **GSH（Global Selection Hijacking）**：全局选择劫持，通过最大化对抗优先级与清洁逆优先级的对齐来重新分配token重要性排序。
- **HEP（Hidden-Evidence Preservation）**：隐藏证据保留，限制被压缩边界置换的高优先级token的表示漂移。
- **TCS（Translation-Consensus Selection）**：平移共识选择，利用多平移视图的优先级rank共识稳定token选择过程。
- **Retained-set allocation**：保留集分配，指压缩后实际保留的token集合，其组成直接影响压缩推理正确性。
- **Representation drift**：表示漂移，清洁token在对抗扰动后编码器输出的特征方向变化程度。

## 可复现要素
- **数据集**：POPE、TextVQA、MME，均为公开基准，使用随机采样的1000个样本。
- **代码**：已在GitHub开源（论文声明"Code is provided in the Github"）。
- **权重**：使用公开预训练模型LLaVA-v1.5-7B、Qwen3-VL-8B-Instruct、InternVL3.5-8B及对应压缩器。
- **关键超参**：$\epsilon=4/255$，优化步数100，步长$1/255$，$\lambda=0.8$，候选预算区间$[32,192]$，TCS平移量$d=7$像素。
