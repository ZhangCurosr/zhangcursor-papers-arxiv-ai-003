---
title: "STAMP-as-a-topology-aware-molecular-language-for-fragment-le"
source: https://arxiv.org/pdf/2610.18438v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-01 22:48:55"
field: "分子表征与生成"
keywords: ["molecular language", "fragment representation", "molecular generation", "molecular optimization", "molecular editing", "STAMP", "activity cliff"]
innovations: ["提出STAMP片段树语言，将motif身份、joint连接位点与branch树拓扑解耦为独立token族，使同一片段可在不同附着上下文中复用且子树对应连续token段", "构建确定性栈机编译与解码器，实现STAMP序列与分子图的严格可逆映射并提供容错规则", "在同一表示上打通分子生成、性质预测、片段树搜索优化与冻结LLM指令引导编辑四类任务并给出统一基准对比"]
benchmarks: ["ZINC-250K", "MoleculeNet", "MoleculeACE", "PMO", "S²-Bench", "ChEBI-20", "GenMol generation protocol"]
---

# 论文速读：STAMP-as-a-topology-aware-molecular-language-for-fragment-le

## 一句话总结
STAMP（Sequential Tree Attachment-based Motif-Preserving representation）提出了一种拓扑感知的分子片段树语言，将分子结构分解为可寻址的片段节点与子树，使同一套表示可同时支持分子生成、性质预测、片段树搜索优化与冻结语言模型的指令引导编辑。

## 研究问题与动机
- **分子语言的"可编辑性"缺失**：SMILES 能精确序列化分子图，但功能团替换等操作需要修改字符串中非连续部分，难以定位和局部修改；SELFIES 保障了合法性，但未显式暴露片段边界与连接位点。
- **片段级表示缺乏直接可寻址性**：现有片段化表示（如 fragSMILES、Group-SELFIES、SAFE、t-SMILES、Molcano）在生成或学习中暴露了化学片段，但片段的连接位点、树拓扑与身份耦合或隐式，导致结构编辑仍需外部图操作或重新生成完整字符串。
- **生成、学习与编辑三个任务缺乏统一可操作状态**：当前方法往往将分子表示作为单一序列，难以同时满足"学习时保留化学单元语义"和"编辑时精确定位操作对象"两种需求。
- **活动悬崖（activity cliff）预测是评价表征质量的关键难点**：微小结构变化可能导致活性剧变，现有 SMILES-BPE 编码器在该任务上表现受限，亟需能更好分离"连接位置变化"与"片段身份变化"的表示。

## 核心贡献（创新点）
1. **提出 STAMP 片段-树分子语言**：通过 motif（片段）、joint（连接原子）与 branch（子树范围）三类独立 token 分别编码片段身份、连接位点和树拓扑，使同一片段可在不同上下文中复用，且任意子树对应连续 token 段。与 t-SMILES 等显式编码片段连接但不分离附着位点的表示相比，STAMP 将"连接位置"与"片段身份"解耦。
2. **构建确定性编译与解码机制**：基于栈机器的解码器可将 STAMP 序列严格还原为分子图；对于越界 joint index 采用取模、价键饱和则顺延至下一可用原子，从而既保证生成有效性又提供容错性。与 fragSMILES 等依赖外部图拼接的方法不同，STAMP 的编译规则完全内置于语言自身。
3. **在同一表示上打通四类任务**：生成（de novo 与约束填充）、性质预测（MLM/MTR 预训练 + MoleculeNet 微调）、片段树搜索优化（TreeCrossover + 捐赠片段池）以及冻结 LLM 指令引导编辑（STAMP-Edit），证明该表示具备跨任务通用性。
4. **揭示 STAMP 嵌入的多尺度化学组织结构**：在 motif 相似度、附着位点变化敏感度、重复转化一致性以及整体分子距离对齐四个层面均显著优于匹配规模的 ChemBERTa-2 10M SMILES-BPE 基线。
5. **证明可寻址子树操作对编辑/优化的结构性增益**：TreeCrossover 兼容性替换在严格合法后代占比（98.5%）和 Bemis–Murcko 骨架保留率（48.5%）上大幅领先 SAFE splicing、任意 token 段替换和 SMILES 子串替换，并在 23 项 PMO 目标中有 15 项取得最优 AUC-Top10。

## 方法详解
- **Token 族设计**：$\mathcal{V} = \mathcal{V}_{\mathrm{mot}} \cup \mathcal{V}_{\mathrm{J}} \cup \mathcal{V}_{\mathrm{B}} \cup \mathcal{V}_{\mathrm{ctrl}}$，其中 motif token 编码 RDKit 规范 SMILES（不含 dummy 原子和 atom-map），joint token $[J_i]$ 引用前一 motif 的第 $i$ 个重原子，branch token $[B_n]$ 声明后续子树包含的 motif 组数。
- **片段化与树构建**：默认采用外链单键切断策略（同 fragSMILES），环系和稠环保留在片段内；以最高度数片段为根，按子树大小降序排列子节点，主链最后一个子节点作为"主延续"，其余子树作为分支。深度优先遍历保证每个子树占据连续 token 段。
- **编码顺序**：根节点 emit motif → joint → stereo；非根节点第一个 joint 为输入端口，其余为输出端口，分支端口先于主延续端口排列；在每个分支子树前 emit 对应的 $[B_n]$。
- **栈机解码**：从左到右处理序列，维护部分分子图 $G$、开放端口栈 $S$ 和分支帧栈。遇到 motif 时弹出父端口连接输入 joint 并压入剩余输出端口；遇到 $[B_n]$ 时保存当前栈并创建局部帧，完成后由内向外恢复栈。
- **容错规则**：joint index 越界时取模；端口价键饱和时顺延到下一个可用原子；超长分支截断；空栈时的 motif 启动新组分；未闭合输出端口以隐式氢补齐。
- **预训练与微调**：
  - **MLM**：15% token 被 mask（80% [MASK]、10% 随机替换、10% 不变），损失函数 $\mathcal{L}_{\mathrm{MLM}}(\theta) = \mathbb{E}_{\mathcal{M}} \sum_{i \in \mathcal{M}} -\log p_\theta(x^{(i)} | x \llcorner \mathcal{M})$。
  - **MTR**：对 pooled 表征经预测头映射到 208 个 RDKit 描述符（z-score 归一化），损失为 $\mathcal{L}_{\mathrm{MTR}}(\theta,\phi) = \frac{1}{208}\|\hat{\mathbf{z}} - \tilde{\mathbf{z}}\|_2^2$。
- **AO-GPT 任意顺序生成**：采样位置置换 $\pi$，目标位置集 $\mathcal{I}_\pi$，损失 $\mathcal{L}(\theta) = \mathbb{E}_\pi[-\frac{1}{|\mathcal{I}_\pi|}\sum_{j \in \mathcal{I}_\pi} \log(\mathrm{softmax}\, z_{\theta,j}(x;\pi))_{y_{\pi(j)}}]$；90% 样本随机置换，10% 保持原序，首尾非 padding 位置固定。
- **TreeCrossover 片段树交换**：子树 $u$ 的端口签名 $\sigma(u)=(b_u,p_u)$，供体 $b$ 替换受体 $a$ 需满足 $\sigma(a)=\sigma(b)$ 且 $|z(a)-z(b)|\leq\delta$（默认 $\delta=4$）；优化循环中高分分子贡献兼容子树进入签名索引的捐赠池（上限 50/签名、5000 总量）。
- **STAMP-Edit 冻结 LLM 接口**：提供 SMILES + motif 节点表（含功能团标签），模型选择 delete/replace/grow 动作作用于指定节点，编译器根据节点索引、端口签名和替换片段重建分子；支持两阶段转化迁移和独立变化识别任务。

## 实验与结果
- **精确重构与紧凑性（ZINC-250K，20,000 样本）**：STAMP 100% 精确重构，平均长度 15.31 tokens（P95=24），显著短于 SMILES/SELFIES/Group-SELFIES（均 >2 倍）；fragSMILES 16.66 tokens 但仅 93.345% 精确重构。
- **ChEMBL 编辑局部性**：STAMP 平均影响 2.24 个连续段（SMILES 4.07，fragSMILES 2.31），最大段平均 5.60 tokens（SMILES 10.10）。
- **受控生成（8 种表示，同构 causal Transformer，100K 训练，10K 请求）**：STAMP quality yield 75.5%，次优 fragSMILES 66.8%，SMILES 55.2%，Molcano 31.4%；STAMP validity 97.38%。
- **大规模生成（AO-GPT，ZINC20 4.76 亿条）**：STAMP quality 88.0±1.1%，diversity 0.861±0.001，validity 99.7%，uniqueness 99.4%，接近 GenMol 最高配置。
- **片段约束生成（GenMol 五任务协议）**：超结构生成 STAMP 58.0% vs GenMol 34.8%；motif 扩展 55.1% vs 30.1%；linker 设计 36.6% vs 21.9%；scaffold decoration 30.1% vs 31.8%；所有设置中 uniqueness 更高、validity ≥98.2%。
- **文本到分子（STAMP-MolT5-Base，ChEBI-20）**：validity 97.6%（MolT5-Base 77.2%），canonical SMILES exact match 10.6%（8.1%），三组指纹相似度均提升。
- **MoleculeNet 性质预测（6/8 MLM 任务、5/8 MTR 任务优于 ChemBERTa-2 10M）**：例如 BBBP-MTR AUROC 0.749 vs 0.733，ClinTox-MTR 0.890 vs 0.601，Lipophilicity-MLM RMSE 0.848 vs 1.009。
- **MoleculeACE 活动悬崖（30 靶标，MTR）**：STAMP-MTR 在 23/30 靶标上 activity-cliff RMSE 低于匹配 SMILES-BPE，均值差 -0.040（Wilcoxon p=0.0040）；STAMP 全集/活动悬崖 RMSE 分别为 0.719/0.825，低于所有深度学习基线，仅 SVM-ECFP4（0.671/0.751）更强。
- **嵌入化学组织结构**：motif 邻域 ECFP4 Tanimoto 相似度约为随机对的 5 倍；附着位点变化 vs 片段身份变化的中位余弦距离比 STAMP 为 0.119（ChemBERTa-2 为 0.439）；重复转化的 leave-one-context-out 一致性 STAMP 中位余弦相似 0.808（ChemBERTa-2 0.520）；90,620 结构匹配对的 Spearman 相关 STAMP 0.671（ChemBERTa-2 0.386）。
- **PMO 优化（23 目标，10,000 次在线 oracle）**：STAMP aggregate AUC-Top10 18.668，GenMol 18.362，f-RAG 16.928；STAMP 在 15/23 目标上位列六方法最优，且超过 GenMol 的目标达 17 个。
- **Docking 先导优化（5 靶标 × 3 起始分子 × 2 阈值）**：δ=0.4 时 STAMP 15/15 成功、GenMol 14/15、Graph GA 10/15；δ=0.6 时 STAMP 12/15、GenMol 12/15、Graph GA 9/15；在 27 个至少一方法成功的 cell 中，STAMP 在 23 个取得最佳 QuickVina2 分数。
- **STAMP-Edit 冻结 LLM 编辑（S²-Bench，Qwen-2.5-7B-Instruct / DeepSeek-V4-Pro）**：MolEdit 平均成功率 Qwen 29.3% → 62.0%（+32.7pp）、DeepSeek 76.2% → 85.4%（+9.2pp）；MolOpt 33.8% → 63.9%（+30.1pp）和 53.6% → 75.3%（+21.7pp）；QED 定向编辑仍最具挑战（Qwen 40.8%、DeepSeek 56.4%）。
- **转化迁移与独立变化识别**：两阶段迁移 STAMP node replacement Qwen 30.8% vs SMILES 13.3%、DeepSeek 87.5% vs 55.8%；独立识别 exact group accuracy Qwen 71% vs 15%、DeepSeek 97% vs 71%。
- **兼容性子树替换有效性**：严格合法后代 98.5%（STAMP 兼容）vs SAFE 97.7%、任意 STAMP 段 23.6%、SMILES 子串 2.7%；骨架保留率 48.5%（STAMP 兼容）vs SAFE 16.8%、任意 STAMP 8.1%、SMILES 0.6%。

## 相关工作脉络
- **t-SMILES**：通过广度优先遍历二叉树序列化片段 SMILES，以 TSSA/TSDY/TSID 编码连接；区别在于 t-SMILES 片段连接通过共享原子或未编号 dummy 原子，STAMP 将 motif 身份、joint 索引与 branch 计数解耦并用深度优先保证子树连续性。
- **Molcano**：使用 junction tags 表达面-对-面与子结构级连接；区别在于 Molcano 的连接关系偏图谱层面，STAMP 使每个子树成为可直接寻址和替换的连续 token 段。
- **fragSMILES / Group-SELFIES / SAFE**：均显式暴露片段或连接，但片段边界与附着位点信息耦合或隐式；区别在于 STAMP 的 joint token 精确指向 motif 内特定重原子，允许同一片段在不同附着位置复用。
- **Junction-tree VAE / Hierarchical generation**：以树结构生成分子；区别在于这些方法侧重生成概率建模，STAMP 将树结构作为可编辑的共享表示接口，支持搜索与 LLM 动作。
- **ChemBERTa-2 / SMILES-BPE**：以字符/子词为单位预训练分子编码器；区别在于 STAMP-MTR 在 motif 级别学习，嵌入能更好分离"连接位置变化"和"片段身份变化"，且在 MoleculeACE 活动悬崖任务上显著更低 RMSE。
- **GenMol / f-RAG / Graph GA / REINVENT**：PMO 与先导优化基准的主要对比基线；区别在于 STAMP 以片段树为搜索空间并通过 TreeCrossover 直接操作子树，而非原子级扩散或图流操作。

## 局限性与未来方向
- **片段化策略依赖默认外链单键切割**：其他方案（BRICS、MMPA、rotatable-bond、PS-VAE 主图）虽已支持但并非默认，针对特定化学空间的最优切割策略仍需探索。
- **QED 定向编辑仍是难点**：无论 Qwen 还是 DeepSeek，QED-directed MolOpt 成功率仅为 40.8% 和 56.4%，反映性质梯度与片段离散操作之间的鸿沟。
- **原子级动作作为补充而非替代**：论文表明质子化状态、形式电荷等变化需原子级操作辅助，当前框架中两者需串联使用，尚未深度融合。
- **大模型能力差异显著影响编辑增益**：DeepSeek 在 substitution 等子任务上 SMILES 改写与 STAMP 动作表现接近甚至反超，说明表示优势部分依赖于底层模型能力。
- **经典 SVM–ECFP4 在 MoleculeACE 上仍最强**：STAMP-MTR 虽超越所有深度学习基线，但未能超越 ECFP4+SVM，提示结构嵌入与经典指纹之间仍有性能 gap。

## 研究启发与可借鉴点
- **"表示即接口"的设计范式**：将分子语言同时作为学习表征与操作接口，使生成器、预测器、搜索器和 LLM 共享同一套可寻址化学单元，这一思路可迁移至聚合物、肽、材料等其他离散化学实体。
- **连续子树 span 与确定性编译器结合**：用 depth-first 遍历保证子结构连续、用栈机解码保证可逆重建，这种"语法约束 + 结构化地址"对其它树形/图形序列化工具具有参考值。
- **控制变量分析嵌入组织方式**：通过固定 parent/probe 分别扰动 attachment position 与 motif identity，定量验证表征对两类结构变化的敏感性分离，该方法可作为评估分子语言"可编辑性"的标准协议。
- **捐赠片段池 + 签名索引的重用机制**：TreeCrossover 将高 reward 分子子树按端口签名归类复用，兼具多样性与结构保持，可推广至其他基于子图交换的优化算法。
- **冻结 LLM 通过动作编译器执行化学编辑**：无需微调即可将指令转化为分子结构变更，接口层只需提供节点表与端口签名；该范式可复制到反应预测、合成规划等场景。

## 关键术语表
- **STAMP**：Sequential Tree Attachment-based Motif-Preserving representation，一种将分子表示为有根片段树的拓扑感知分子语言。
- **Motif token**：编码一个规范片段化学结构的 token，化学等同片段在不同上下文中共享同一 motif 标识。
- **Joint token**：$[J_i]$ 形式的 token，指向 preceding motif 中第 $i$ 个重原子，显式标识跨片段键的附着位点。
- **Branch token**：$[B_n]$ 形式的 token，声明其后跟随的子树包含 $n$ 个 motif 组，用于划分树拓扑边界。
- **TreeCrossover**：将供体子树替换到受体子树位置的搜索算子，要求端口签名匹配且重原子数偏差在 $\delta$ 内。
- **MoleculeACE**：含 30 个靶标的活动悬崖基准，用于评估分子机器学习模型在微小结构变化下预测活性跃迁的能力。
- **PMO（Practical Molecular Optimization）**：含 23 个黑盒目标的分子优化基准，以 AUC-Top10 衡量在有限 oracle 调用下找到高质量分子的能力。
- **S²-Bench**：Speak-to-Structure Benchmark，评估 LLM 在自然语言指令下进行分子编辑与优化的能力，包含 MolEdit 与 MolOpt 子任务。

## 可复现要素
- **数据集与开源**：ZINC-250K、ZINC15、ZINC20、PubChem-10M、ChEMBL、C4+ZINC15 混合语料、ChEBI-20、MoleculeNet、MoleculeACE、S²-Bench 等均来自公开来源；主数据集已托管于 https://huggingface.co/datasets/bioS/STAMP_V1-data。
- **代码与权重**：STAMP 框架源码开源于 https://github.com/YuX-Ren/STAMP_V1；ChemBERTa-2 10M 与 MolT5-Base 等基线权重来自各自公开发布。
- **关键超参**：
  - 受控 Transformer：4 层、dim 256、8 heads、FFN 1024、max_len 256、dropout 0.1、lr 3e-4、weight_decay 0.01、batch 256、10 epochs。
  - AO-GPT：12 层、dim 768、12 heads、FFN 3072、max_len 256；de novo 300K steps、batch 1024、lr 3e-4；infill fine-tune 3 epochs、batch 256、lr 1e-4；scaffold decoration 加 10 epochs、batch 256、lr 5e-5。
  - STAMP-MolT5-Base：12+12 层、dim 768、FFN 2048、12 heads；C4+ZINC15 交替预训练 500K steps、batch 1024、lr 1.4e-4、5K warmup；下游 fine-tune lr 5e-4、batch 32、最多 30 epochs、6 次无改进早停；beam width 5、max new tokens 256。
  - MLM 预训练：10 epochs、batch 512、lr 5e-4；MTR 预训练：5 epochs、batch 512、lr 3e-4。
  - Fine-tune：batch 32、lr 2e-5、20 epochs。
  - TreeCrossover 默认 $\delta=4$；PMO 在线 oracle 预算 10,000 次；冻结 LLM 评估 temperature 0、max output 8192、关闭 reasoning。
