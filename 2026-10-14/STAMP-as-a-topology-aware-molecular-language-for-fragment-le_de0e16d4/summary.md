---
title: "STAMP-as-a-topology-aware-molecular-language-for-fragment-le"
source: https://arxiv.org/pdf/2610.18438v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-01 22:49:14"
field: "分子语言与可编辑表征"
keywords: ["molecular language", "fragment tree", "STAMP", "molecular generation", "property prediction", "molecular optimization", "LLM-driven editing"]
innovations: ["提出 STAMP 四 token 类（motif/joint/branch/ctrl）将片段身份、连接位点与子树拓扑分离表示，实现确定性编译解码", "构建可寻址子树空间与 TreeCrossover 算子，支持兼容替换与骨架保留的片段搜索", "在生成、性质预测、PMO 优化与冻结 LLM 编辑四项任务上统一验证，编辑成功率提升 9–56 pp"]
benchmarks: ["ZINC-250K", "MoleculeNet", "MoleculeACE", "PMO", "S²-Bench", "ChEBI-20"]
---

# 论文速读：STAMP-as-a-topology-aware-molecular-language-for-fragment-le

## 一句话总结
本文提出 STAMP（Sequential Tree Attachment-based Motif-Preserving representation），一种拓扑感知的分子语言，将分子表示为以化学片段为节点的根树序列；通过将基序标识、连接位点与子树拓扑显式分离，使同一化学单元可在不同上下文中复用，并支持基于片段树的精确结构编辑、性质预测与生成优化。

## 研究问题与动机
- 现有分子语言（如 SMILES、SELFIES）虽能忠实序列化分子图，但功能团替换等编辑操作常需在非连续字符串位置进行协调修改，难以直接定位与替换完整化学单元。
- 已有片段/分组表示（如 SAFE、fragSMILES、Group-SELFIES、t-SMILES、Molcano、Junction-tree 模型）使片段或连接显式化，但多将片段身份与连接位点耦合或未明确定义子树边界，导致“学习与可编辑性”脱节。
- 分子设计需要三要素同时显式：目标位置（attachment）、替换边界（subtree span）与连接要求（bond/open-port），现有编码常隐式或缺失其中一项。
- 研究希望建立一种“可执行分子语言”，使预训练学到的化学单元在学习阶段保持语义，并在搜索/编辑阶段仍可直接寻址与操作。

## 核心贡献（创新点）
- 提出 STAMP 树形片段语言：用 Motif/Joint/Branch/Ctrl 四类 token 分别编码片段身份、跨片段键合原子与有序树拓扑，使同一基序可在不同取代/连接上下文中复用。
- 设计确定性编译解码器与栈机组装过程：每个子树映射为连续 token 区间，支持将替换、生长、删除等操作直接编译为合法分子图。
- 统一共享状态打通学习与编辑：编码器学到的 motif 相似度、连接变化与重复变换一致性，在同一表示上同时服务于属性预测、片段树搜索与冻结 LLM 指令编辑。
- 在多项任务上取得显著增益：ZINC-250K 精确重建 100% 且平均序列最短；8 种表示对比中生成质量产量最高；MoleculeACE 在 30 靶点中 23 个降低活性悬崖预测误差；PMO 基准在 23 项目标中 15 项第一；对冻结 LLM 的编辑成功率提升 9–56 个百分点。
- 开源代码、数据与模型基础设施：提供 codec、词表挖掘、AO-GPT 生成/填充、TreeCrossover 优化、representation learning、molecule–text 与 STAMP-Edit 接口。

## 方法详解
- **Token 体系**：词汇分为 $\mathcal{V}_\text{mot}$（基序）、$\mathcal{V}_\text{J}=\{[J_i]\}$（joint，指向 preceding motif 重原子索引 $i$）、$\mathcal{V}_\text{B}=\{[B_n]\}$（branch，记录后续子树 motif 组数 $n$）与 $\mathcal{V}_\text{ctrl}$（控制/立体化学符号）。
- **片段化**：默认策略为断环外所有单键（沿用 fragSMILES 约定），环与稠环保留在 motif 内；每个 motif 转为无 dummy 原子、无 atom-map 的 RDKit 规范 SMILES，保证化学等价片段跨上下文共享同一 motif token。亦支持 BRICS、MMPA、rotatable-bond、principal-subgraph 等切割。
- **根树构建与遍历**：以度最大片段为根（平局按重原子数），子节点按子树大小降序排列；最大子树为主干继续，其余为分支。采用深度优先遍历，使每个子树占据连续 token 区间；训练时可作根与子节点顺序增强。
- **编码语法**：每节点 emits 一组 “motif token + joint tokens + 立体化学符号”。根节点的 joints 均为输出端口；非根节点首 joint 为来自父节点的输入端口，其余为输出端口；分支端口先于主干端口列出。每个分支子树前插入 $[B_n]$，其中 $n$ 为该分支内 motif 组总数（不含 joint/branch/立体控制 token）。
- **栈机解码**：从左到右维护部分分子图 $G$、开放端口栈 $S$ 与分支帧栈。每次 motif 组弹出父端口、连入本组输入 joint，并将剩余输出端口逆序压栈；遇 $[B_n]$ 时弹出下一连接端口、保存当前栈并开启分支局部栈，完成 $n$ 组后由内向外闭合。对畸形序列提供回退：joint 索引越界取模、饱和位点顺延至下一可用价、过长分支截断、空端口栈时开启新连通分量、尾部开放端口补 implicit H。
- **词表构造**：受约束的 ZINC-250K 比较用 6,783 总 token；更大规模 AO-GPT 保留 12,000 motifs 并加 atom-level fallback，共 12,659 token；分子编码器使用 PS-VAE 主成分子图挖掘得到 9,904 motifs，频率筛选后 4,666 motifs，总计 4,761 token，累计覆盖 99%（motif-only 96.85%）。
- **生成模型**：受控比较用 4 层 causal Transformer（d=256，8 头，FFN=1024，maxlen=256，dropout=0.1，AdamW lr=3e-4，wd=0.01）。大规模使用 12 层 AO-GPT（d=768，12 头，FFN=3072，maxlen=256，条件化自适应 LayerNorm），按任意顺序采样（90% 随机置换、10% 恒等，首末非 padding 位置固定）。生成以左到右解码为主；片段填充以 group-aligned holes 作为 prompt，孔内预测并 splice 回原树。
- **优化算子 TreeCrossover**：子树 $u$ 的端口签名 $\sigma(u)=(b_u,p_u)$（进入键级与根 motif 开放端口数）。捐赠者 $b$ 可替换受体 $a$ 当且仅当 $\sigma(a)=\sigma(b)$ 且 $|z(a)-z(b)|\le \delta$（默认 $\delta=4$ 重原子容差）。后代解码与 sanitize 失败则拒绝。
- **预训练与下游**：STAMP-MTR/MLM 复用 ChemBERTa-2 10M 配置（6 层，d=768，12 头，FFN=1536，len=128），MLM 掩码 15%，其内 80/10/10 为 masked/random/保持；MTR 目标为 208 个 z-scored RDKit 描述符（剔除 ipc、Kappa3），损失为均方。下游微调 lr=2e-5、batch=32、20 epochs。
- **冻结 LLM 编辑接口 STAMP-Edit**：输入 canonical SMILES 与带功能团标签的 motif-node 表；模型冻结，输出 `delete_subtree`、`replace_subtree` 或 `grow_at_port` 三类操作，由编译器按 port-preserving 规则执行并返回分子。

## 实验与结果
- **重建与紧凑性**：ZINC-250K 训练划分随机 20,000 样本，STAMP 精确重建率 100%，平均序列长度 15.31 token（P95=24），为 8 种表示中最短；SMILES/SELFIES/Group-SELFIES 虽 100% 重建但平均长度超两倍。
- **编辑局部性**：ChEMBL 1,497 对匹配分子对中，STAMP 编辑平均影响 2.24 个连续区间，低于 SMILES 的 4.07；最大区间平均 5.60 vs SMILES 10.10。
- **生成质量产量**（ZINC-250K，同 backbone、同输入、10k 请求）：STAMP 75.5% 最高；fragSMILES 66.8%，t-SMILES TSDY 58.4%，SMILES 55.2%，Molcano 31.4%。STAMP 有效率为 97.38%。
- **大规模 AO-GPT**：STAMP AO-GPT 12 层，quality 88.0±1.1%、diversity 0.861±0.001，有效性 $99.7\%$、唯一性 $99.4\%$，接近 GenMol 最高配置（quality 84.6%、diversity 0.818）。
- **片段约束任务**：STAMP 在 superstructure（58.0% vs 34.8%）、motif extension（55.1% vs 30.1%）、linker design（36.6% vs 21.9%）优于 GenMol；scaffold decoration 30.1% vs 31.8%。
- **文本↔分子**：STAMP-MolT5（Base，62.6M 词表）在 ChEBI-20 上 7 项指标中 5 项优于 MolT5-Base；有效性 97.6%（vs 77.2%），canonical SMILES 精确匹配 10.6%（vs 8.1%），指纹相似度全面提升。
- **性质预测**（MoleculeNet，8 任务，匹配 ChemBERTa-2 10M）：MLM 下 STAMP 在 6/8 任务占优；MTR 下 5/8 占优。
- **MoleculeACE 30 靶点**：STAMP-MTR 相对 SMILES-BPE 在 23/30 靶点降低 activity-cliff RMSE，配对差均值 −0.040（Wilcoxon p=0.0040）；全集合 RMSE 0.719、AC RMSE 0.825，为深度方法最低；经典 SVM–ECFP4 为 0.671/0.751。
- **嵌入化学语义**（STAMP-MTR frozen）：motif 邻域 10-近邻平均 ECFP4 Tanimoto 约 5 倍随机； attachment 变化 vs motif 替换的 median cosine 分别为 0.021 vs 0.179（ChemBERTa-2 为 0.061 vs 0.138），位置/身份比值 0.119 vs 0.439，显示更清晰区分；1,485 次重复变换在 ≥6 上下文中 leave-one-out 一致性中位数 0.808（ChemBERTa-2 0.520）；314,873 对 Spearman 相关 STAMP 0.671 vs ChemBERTa-2 0.386，90,620 结构匹配子集 30 靶点全部更高。
- **替换算子兼容性**（1,000  trial/算子）：STAMP 兼容子树严格有效 98.5%（SAFE 97.7%、任意 STAMP 23.6%、SMILES 子串 2.7%）；同时保留 Bemis–Murcko 骨架的比例 48.5%（SAFE 16.8%、任意 STAMP 8.1%、SMILES 0.6%）；在严格有效后代中骨架保留率 49.2%（SAFE 17.3%、任意 STAMP 34.5%、SMILES 22.2%）。
- **PMO 23 目标 AUC-Top10**：STAMP 18.668 > GenMol 18.362 > f-RAG 16.928 > Gen. GFN 16.213 > Mol GA 14.708 > REINVENT 14.196；STAMP 在 15/23 目标上均值最高，超过 GenMol 17 项。
- **Docking 优化**（5 靶点×3 起点×2 δ）：δ=0.4 时 STAMP 15/15 返回改善候选，δ=0.6 时 12/15（GenMol 同为 15/15 与 12/15，Graph GA 10/15 与 9/15）；27 个至少一方成功的 cell 中 STAMP 最优 23 个、GenMol 3 个、Graph GA 1 个；pairwise 胜 GenMol 24 个。
- **冻结 LLM 编辑 S²-Bench**：Qwen-2.5-7B-Instruct MolEdit 成功率 62.0% vs complete-SMILES 29.3%；DeepSeek-V4-Pro 85.4% vs 76.2%。MolOpt Qwen 63.9% vs 33.8%，DeepSeek 75.3% vs 53.6%。两阶段变换迁移（120 例）：Qwen 30.8% vs 13.3%，DeepSeek 87.5% vs 55.8%。独立变化识别（100 对）：STAMP 71%/97%，SMILES 15%/71%。整体提升 9–56 pp。

## 相关工作脉络
- **SMILES/SELFIES/Group-SELFIES**：字符串序列化；前者有效性依赖语法，后者 100% 有效但功能团替换需跨段协调修改；STAMP 在片段层面显式边界与端口。
- **SAFE / fragSMILES**：使片段/分组显式；fragSMILES 重建率与局部性较好，但未把“节点-区间-端口签名”统一到可编译操作界面；STAMP 在此基础上增加可寻址子树与确定性编译器。
- **t-SMILES（TSSA/TSDY/TSID）**：基于二叉树遍历序列化片段 SMILES；连接通过共享原子或无号/标号虚拟原子表达；STAMP 分离 motif 身份与 attachment 索引，并用深度优先让子树连续。
- **Molcano**：用 junction tag 表达面面与子结构级连接；STAMP 用有序树使子树可直接寻址与 splice，更适合基于片段的操作。
- **Junction-tree VAE / hierarchical motif 生成**：以图/树结构生成分子；多面向生成建模，缺少面向冻结 LLM 的可编译动作接口；STAMP 将表征与编辑统一于同态。
- **GenMol / f-RAG / Graph GA / REINVENT**：PMO 与 docking 优化基线；STAMP 以相同的 target-aware prescreening 协议可比，体现片段树搜索对目标局部修改的高效性。
- **ChemBERTa-2 / MolT5**：分子语言预训练基座；本文在匹配配置下展示 STAMP encoder 的表征优势，并把文本↔分子翻译扩展到 fragment 序列。

## 局限性与未来方向
- 默认 exocyclic 单键切割未必对所有反应/合成场景最优；其他切割（BRICS/MMPA/rotatable/learned）在紧凑性与任务适配上的系统比较未充分展开。
- Scaffold-decoration 约束质量略低于 GenMol，提示纯片段替换在“装饰既定 scaffold"任务中可能存在表达力瓶颈。
- 活性悬崖预测虽整体优于 SMILES-BPE，但 SVM–ECFP4 仍是最强参考；冻结距离对活动悬崖判别的区分度接近随机，说明表征组织与下游监督区分之间存在差距。
- 冻结 LLM 编辑效果高度依赖模型能力与标注接口（DeepSeek 在 substitution 上较平稳，Qwen 在 QED 定向编辑上仍困难），通用迁移仍需进一步对齐化学先验与指令空间。
- 原子级操作作为补充动作存在，但本文以片段操作为主；更细粒度的混合原子-片段搜索与多步规划未系统评估。

## 研究启发与可借鉴点
- **“语言即接口”**：将分子序列从单纯的数据载体升级为可执行操作接口（motif+joint+branch+compiler），可为其他化学/材料序列语言提供统一范式。
- **子树连续性 + 端口签名**：深度优先展平使任意子结构对应连续区间，并以外露键级与端口数定义兼容条件，便于将图编辑问题转化为区间 splice，值得复用到图结构序列学习。
- **嵌入分析的化学可解释协议**：attachment vs motif 替换的对照对、跨上下文的重复变换一致性、结构与嵌入距离的 Spearman 对齐，构成一套可迁移的“表征质量审计”流程。
- **同配置公平对比**：与 SMILES-BPE 在层数/dim/head/pretrain/corpus/fine-tune 上完全对齐，使增益归因更可信；多任务/多基线共用 prescreening 协议也值得沿用。
- **冻结 LLM + 化学编译器**：不改模型权重而靠结构化动作界面提升编辑成功率，为通用 LLM 的化学应用提供低成本落地方向。

## 关键术语表
- **STAMP**：Sequential Tree Attachment-based Motif-Preserving representation，一种以片段为节点、以 joint/branch token 表达连接与树拓扑的分子序列语言。
- **Motif token**：编码规范化学片段身份的 token，相同化学片段跨上下文共享同一 token。
- **Joint token [J_i]**：指向前一个 motif 规范原子序 $i$ 的连接符号，显式标注跨片段键合位点。
- **Branch token [B_n]**：标记其后紧跟的子树包含 $n$ 个 motif 组，用于栈机解码恢复树形嵌套。
- **TreeCrossover**：基于端口签名与重原子容差 $\delta$ 的兼容性子树替换算子，用于片段树搜索优化。
- **AO-GPT**：Any-order autoregressive GPT，以任意顺序采样位置进行自回归训练的 Transformer 生成器。
- **MTR**：Multi-task regression 预训练目标，以 208 个 z-scored RDKit 描述符为监督信号。
- **S²-Bench**：Speak-to-structure benchmark，评估冻结 LLM 在 MolEdit/MolOpt 等自然语言驱动分子编辑与优化任务上的能力。

## 可复现要素
- 代码：https://github.com/YuX-Ren/STAMP_V1
- 数据：https://huggingface.co/datasets/bi0s/STAMP_V1-data
- 公开数据集：ZINC-250K、ZINC15、ZINC20、PubChem-10M（cleaned）、ChEMBL matched pairs、MoleculeNet、MoleculeACE、C4+ZINC15、ChEBI-20、S²-Bench
- 关键超参（代表性）：Causal Transformer 4 层/d=256/8 头/FFN=1024/maxlen=256/lr=3e-4/wd=0.01/batch=256；AO-GPT 12 层/d=768/12 头/FFN=3072/maxlen=256；MolT5 Base 12+12 层/d=768/12 头/FFN=2048；MTR/MLM head d=768→208；TreeCrossover 默认 δ=4；生成采样常见 temp=0.7–1.3/top-p=0.95/1.0、top-k=50 或无截断。
- 评估协议与复现细节：见 Methods 与 Supplementary，PMO 使用与 GenMol 相同的 target-aware prescreening；MoleculeACE 使用配对 Wilcoxon 检验。
