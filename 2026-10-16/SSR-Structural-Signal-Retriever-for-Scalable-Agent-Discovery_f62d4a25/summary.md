---
title: "SSR-Structural-Signal-Retriever-for-Scalable-Agent-Discovery"
source: https://arxiv.org/pdf/2610.21113v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-04 03:58:01"
field: "多智能体系统中的检索与路由"
keywords: ["agent discovery", "candidate-context scalability", "structural retrieval", "SSR", "LLM orchestration", "inverted index", "MS MARCO", "multi-agent systems"]
innovations: ["提出候选上下文可缩放性的系统级建模与分离度量", "SSR: 从 Agent Card 字段提取结构词法信号并用 IDF+ISI 做确定性无训练检索", "ISI 等价性保证: 仅排除零分候选，保留正分候选的原始排名与分数"]
benchmarks: ["SSR-AgentBench", "MS MARCO Passage Ranking"]
---

# 论文速读：SSR-Structural-Signal-Retriever-for-Scalable-Agent-Discovery

## 一句话总结
本文针对 LLM 编排的多智能体系统中**候选上下文可缩放性（candidate-context scalability）**问题，提出了一种确定性、无训练的 Agent Card 检索方法 SSR——从 Agent Card 能力字段提取结构词法信号，用 IDF 加权并在倒排信号索引（ISI）上做选择性打分，将最多 K 个候选 Agent Card 呈现给 LLM 进行最终仲裁。在 1000 个 Agent 规模下，SSR 将 LLM 输入缩减 98.8%，编排延迟降低 70.9%；在 MS MARCO 适配实验中，ISI 保留检索有效性的同时减少 96.8% 的显式打分文档。

## 研究问题与动机
1. **完整目录编排不可缩放**：直接把全量 Agent Card 注入 LLM prompt 时，随着注册表增大，context 消耗、推理成本、延迟线性增长，并最终超出模型或 API 上下文窗口限制。
2. **已有检索工作未显式分离"注册表规模检索"与"候选上下文预算"**：现有工具/Agent 检索研究关注召回质量或学习路由，但未把 candidate-context scalability 作为系统级约束建模与分离度量。
3. **Agent Card 特有结构未被利用**：A2A 协议定义的 Agent Card 包含 name、description、skills、capability tags 等机器可读字段，但通用检索方案并未直接基于这些结构字段设计检索表示。
4. **可追溯性与确定性需求**：在实际部署中，检索分数需要能追溯到具体匹配的 Agent Card 证据信号，而非黑盒向量相似度。

## 核心贡献（创新点）
1. **提出候选上下文可缩放性的系统级建模**：将注册表规模检索、检索计算与呈现给 LLM 的紧凑候选上下文三者分离，并用最大候选预算 K 显式约束 |P_q| ≤ K。
2. **SSR 检索方法**：从 Agent Card 能力字段确定性提取 unigram/bigram 结构信号，用 IDF 加权匹配分数，并通过 ISI 仅对结构共现候选打分，避免全量扫描。
3. **ISI 的等价性保证**：理论证明 ISI 仅排除零分候选，保留所有正分候选的原始排名顺序与分数；实验验证 top-K 与全扫完全一致。
4. **SSR-AgentBench 基准**：引入包含 1000 张 Agent Card、250 条自然语言查询、24 个应用域与 144 项技能的合成基准，支持可控注册表缩放实验。
5. **系统级评测框架**：通过 RQ1/RQ2/RQ3 分别度量缩放开销、检索有效性/效率与端到端 LLM 仲裁行为，并以 BM25 作为对照分离"预选择机制的一般效果"与"SSR 特定效果"。

## 方法详解
**信号提取与表示**
- 从 Agent Card 的 name、description、skills、capability tags、examples、tool descriptions 等能力承载字段，按 `[a-z0-9]+` 正则提取词法单元。
- 小写、去停词与单字符 token 后生成 unigram 及连续 bigram 信号；去重后使用二进制存在（非 TF）。
- 每份表示最多保留 64 个信号：先按 token 频率降序保留 unigram，再按字典序考虑 bigram。

**IDF 加权打分**
- 匹配分数：Score(q,a) = Σ_{s ∈ S_q ∩ S_a} idf(s)，其中 idf(s) = log((N+1)/(df(s)+1)) + 1。
- 稀有信号贡献更大；所有 IDF 权重为正，确保 ISI 排除零分候选不改变正分候选的相对顺序。

**倒排信号索引（ISI）**
- ISI(s) = {a ∈ A : s ∈ S_a}，候选集合 C_q = ∪_{s ∈ S_q} ISI(s)。
- 仅对 C_q 内候选计算 Score；若 C_q = ∅ 则返回空池 P_q = ∅，不填充零分候选。

**在线复杂度**
- 候选构造：O(Σ_{s ∈ S_q} p_s)，p_s 为 posting list 长度。
- 打分：O(|C_q|·|S_q|)；排序：O(|C_q| log |C_q|)。
- 总复杂度：O(Σ p_s + |C_q||S_q| + |C_q| log |C_q|)，与注册表大小 N 解耦。

**候选预算 K**
- P_q = TopK(C_q, K)，|P_q| = min(K, |C_q|) ≤ K。
- K=20 经独立 1000 查询验证（within-0.02 recall rule）被选定为稳健操作点。

## 实验与结果
**数据集**
- SSR-AgentBench：1000 Agent Card（850 base + 150 capability twins）、250 查询、24 域、144 技能，seed=2026。
- MS MARCO Passage Ranking：8,841,823 passage，6980 开发查询用于有效性评测。

**RQ1 缩放开销（SSR-AgentBench, K=10）**
- N=1000 时，Full-catalog 平均 prompt 63,645 tokens，SSR 仅 767 tokens；输入减少 98.8%。
- 成本从 $0.009551 → $0.000119/查询；延迟从 2.647s → 0.769s（减少 70.9%，3.44×）。
- SSR 候选召回率与最终选择准确率均为 1.000（全规模）。

**RQ2 检索有效性（MS MARCO, K=10）**
- BM25 最强原始基线：R@10=0.3890, MRR@10=0.1829；SSR 为 R@10=0.3378, MRR@10=0.1544。
- IDF 相对 Structural Overlap+ISI 提升：R@10 +5.3%，MRR@10 +5.6%。
- ISI 验证：200 查询子集上，全扫 69,848.3 ms/查询（8.8M passage） vs SSR 941.5 ms/查询（~286K passage）；显式打分 passage 减少 96.8%，74.2× 加速，top-K 完全一致。

**RQ3 端到端发现（MS MARCO, K=20）**
- N≤1000 时 SSR/BM25 候选召回均为 0.980；N=5000 时 BM25 0.970 vs SSR 0.950。
- 全目录在 N=5000 时全部 200 次 LLM 调用失败（prompt 估计 447,761 tokens）；预选择方案均正常运行。
- McNemar 检验显示 SSR 与 BM25 在端到端正确率上无统计学显著差异（Holm 校正后 p=1.000）。
- SSR 端到端准确率 95% Wilson 区间：N=100→[0.950,0.992], N=500→[0.936,0.986], N=1000→[0.923,0.980], N=5000→[0.833,0.922]。

## 相关工作脉络
1. **ToolRet [5]**：大规模工具检索信息检索建模，SSR 与其区别在于面向 Agent Card 结构字段而非通用工具描述。
2. **PORTS [6]**：偏好优化检索器用于工具选择，SSR 无训练、确定性强。
3. **ToolScope [7]**：工具合并+上下文感知过滤，与 SSR 共享"减少 LLM 候选上下文"目标但机制不同。
4. **Tool-to-Agent Retrieval [8] / Agent-as-a-Graph [9]**：共享向量空间或图结构检索，属于稠密/关系方法；SSR 为稀疏词法方法。
5. **ACE-Router [10] / AgentRouter [36]**：学习型路由，依赖历史/知识图谱；SSR 训练免费、可追溯。
6. **DTDR [15] / ToolQP [16]**：动态/多步检索，SSR 为单次确定性检索，未来可结合。
7. **Adaptive shortlist depth [18]**：将候选数量本身作为决策变量，与 SSR 的固定 K 形成互补方向。

## 局限性与未来方向
1. **MS MARCO 是文档派生 Agent Card 抽象**，缺失真实 Agent Card 的结构化技能、接口、认证要求等元数据，结论需在原生 Agent Card 集合上验证。
2. **SSR-AgentBench 为合成基准**，单目标每查询，无法反映运营生态中的异构性、多有效 Agent 并存、能力演化等。
3. **SSR 仅依赖显式词法 unigram/bigram**，词汇不匹配会降低召回；更 expressive 的字段派生/语义/混合检索可能提升覆盖。
4. **K=20 为回顾性选定**，非前瞻性 held-out 协议；需在未来工作中先冻结 K 再在独立测试集上评测。
5. **RQ3 仅用 200 查询与单一 LLM/API 配置**，BM25 与 SSR 分开执行；延迟/成本/失败点不可泛化至其他模型或 provider。
6. **离线索引构建成本未计入在线延迟**，预处理、存储与维护开销未评估。
7. 未来方向：查询自适应候选深度 K(q)、SSR 生成廉价候选后接语义 reranking、执行感知动态检索、原生公共 Agent Card 集合、多相关性判标、分布式增量索引。

## 研究启发与可借鉴点
1. **candidate-context scalability 的系统分离视角**：将检索覆盖、检索计算与 LLM 候选上下文作为独立可测组件，为多智能体编排的模块化设计提供新框架。
2. **ISI 等价性证明范式**：用理论证明+实验验证双保险确保倒排索引不损失排名质量，对稀疏检索工程化有直接参考价值。
3. **BM25 对照分离"预选择一般效果"与"方法特定效果"**：通过相同 K 预算下的公平对照，凸显系统贡献与方法贡献，值得在后续评测中沿用。
4. **SSR-AgentBench 的可控生成流水线**（taxonomy→base cards→twins→queries→targets）为 Agent 发现领域的标准化基准提供了可复现模板。
5. **与动态/语义检索的串行组合策略**：SSR 可作廉价第一层生成候选，再经语义 rerank 补充，契合 ToolDreamer 思路与 DTDR 动态更新框架。

## 关键术语表
- **Candidate-context scalability**：注册表规模检索与呈现给 LLM 的紧凑候选上下文解耦的系统可缩放性质。
- **Agent Card**：A2A 协议定义的机器可读 Agent 描述对象，含身份、能力、技能、接口与元数据。
- **Structural Signal**：从 Agent Card 能力字段确定性提取的 unigram/bigram 词法单元。
- **IDF (Inverse Document Frequency)**：稀有信号赋予更高权重的词项权重，这里定义为 log((N+1)/(df+1))+1。
- **ISI (Inverted Signal Index)**：按结构信号建倒排索引，ISI(s) 返回含信号 s 的所有 Agent Card 集合。
- **Maximum candidate budget K**：上限约束 |P_q| ≤ K，控制呈现给 LLM 的候选数量。
- **SSR-AgentBench**：包含 1000 Agent Card 与 250 查询的合成基准，用于可控注册表缩放评测。
- **End-to-end accuracy**：检索命中且 LLM 最终选出相关 Agent 的比例，含检索缺失、空池、API 失败等全部失败模式。

## 可复现要素
- **数据集**：SSR-AgentBench（合成，seed=2026）；MS MARCO Passage Ranking（公开）。
- **代码/权重**：GitHub https://github.com/gmaldonadoc/sscr-agent-routing（benchmark 生成与评测代码开源）；SSR 无训练权重。
- **关键超参**：最大信号数 64；K=10（RQ1）、K=10（RQ2）、K=20（RQ3）；BM25 参数 k₁=1.2, b=0.75。
- **LLM**：gpt-4o-mini, temperature=0, JSON-mode；API 定价 $0.15/M input, $0.60/M output。
- **环境**：Amazon SageMaker ml.m5d.8xlarge, 32 CPU, ~124 GiB RAM, Xeon 8259CL, Python 3.12.9, SQLite 3.48.0。
- **随机种子**：RQ1/SSR-AgentBench 用 seed=2026；RQ2 时序/等价与 RQ3 样本用 seed=42；K 独立验证用 seed=2027。
