---
title: "Stronger-Judges-Do-Not-See-More-Deployment-Integrity-Blindne"
source: https://arxiv.org/pdf/2610.21135v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-04 03:59:22"
field: "LLM 评估与可靠性"
keywords: ["LLM evaluation integrity", "deployment tampering", "LLM judge", "deployment identity", "adversarial robustness", "benchmark blindness"]
innovations: ["揭示'越强评估器越盲'反直觉现象并提出DIB-Bench基准", "建立PAC风格行为-身份二分化理论框架并形式化不可判定类", "提出分级部署身份描述符作为评测报告的新增规范"]
benchmarks: ["DIB-Bench", "BFCL-simple", "GSM8K", "AgentDojo"]
---

# 论文速读：Stronger Judges Do Not See More: Deployment-Integrity Blindness in LLM Evaluation

## 一句话总结
本文首次系统揭示 LLM 评测中的**部署完整性盲点（Deployment-Integrity Blindness, DIB）**：输出-only 评估器越强大，对部署篡改的接受率反而越高（从规则的 88.6% 升至 GPT-5-nano 的 94.1%），且任何样本量下都无法检测零输出偏移的触发条件型篡改；作者提出 DIB-Bench 基准与分级部署身份描述符作为补救方案。

## 研究问题与动机
1. **评测对象已从模型变为复合系统**：当前 LLM-agent 评测仅报告行为分数（准确率、安全性等），但未披露部署时的运行时栈身份；一个经过篡改但仍保持评测输出的部署可轻松通过所有输出型评估器。
2. **更强的评估器并未解决问题，反而加剧盲点**：直觉上"用更强 judge 即可检测篡改"的补救思路被实验证伪——GPT-5-nano 比规则评分接受更多篡改输出。
3. **理论上的不可判定类存在**：PAC 风格理论证明，对输出分布无偏移（ε=0）的篡改在任意样本量下对任何行为测试均不可区分；只有独立的身份通道（manifest hash）才能以零样本判定此类。
4. **现有 agent 基准（BFCL、AgentDojo、τ-bench 等）只关注行为，忽略部署身份这一新坐标**，导致排行榜排名可能反映的是系统栈而非模型本身。

## 核心贡献（创新点）
1. **提出 DIB-Bench**：首个针对部署完整性盲点的可扩展基准，覆盖 7,560 个打分示例、六种单组件篡改类型及一条从规则评分→Qwen3-8B judge→GPT-5-nano judge 的评估器阶梯；与已有工作本质区别在于将"评估器强度与盲点关系"作为核心变量而非仅报告行为分。
2. **建立行为证据与身份证据的形式化分离**（Proposition 1 + Theorem 1）：以 S-等价类和 TV 距离界定不可区分部署类，并用 PAC 二分化给出样本复杂度下界（Nε）与上界（O(ε⁻² log(1/δ))），这是首次对 LLM 评测中"输出不可见即不可判"给出严格理论刻画。
3. **实证发现"越强评估器越盲"的反直觉现象**：mean tamper acceptance 从 rule-based 的 88.6% 单调升至 GPT-5-nano 的 94.1%，且 GPT-5-nano 的 suspicion auditing 对 clean 和 tampered 输出几乎无法区分（54.0% vs 52.6%）；与已有工作（如认为 LLM judge 可替代人工审核）形成直接对照。
4. **构造并验证两种 stealth tamper**（E4 触发条件型、E8 全通道逃避型），前者在评测集上 100% 不可见但外部触发后有害率达 48–100%（随模型能力提升），后者同时逃避输出评估器和身份通道；揭示了 manifest coverage 决定身份通道检测力的关键约束。
5. **提出分级部署身份描述符（Deployment-Identity Descriptor）**：为每项评测报告附加每组件内容哈希及来源方式（self-declared / build provenance / runtime attestation），使"被评测对象是谁"成为可审计的一等公民。

## 方法详解
- **部署模型**：将部署视为六层运行时栈——model/tokenizer、system prompt、tool spec & router、retrieval policy & corpus、runtime dependencies、postprocessor，每个组件用 SHA-256（canonical JSON）记录于声明 manifest 中。
- **双通道评估框架**：
  - **输出通道**：$E_{\text{out}}(\text{system}(x)) = \text{pass/fail}$，使用规则评分（Standard/Strict/Forensic/Canary）、Qwen3-8B 本地 judge、GPT-5-nano judge 四个强度递增的评估器。
  - **身份通道**：$E_{\text{int}}(M_{\text{expected}}, M_{\text{observed}}) = \text{pass} \iff \forall c,\; h(M_{\text{expected}}[c]) = h(M_{\text{observed}}[c])$。
- **盲点度量**：示例级盲点 = $E_{\text{out}}=\text{pass} \wedge E_{\text{int}}=\text{fail}$ 的均值；套件级盲点 blind@τ = $\mathbb{I}[\text{output\_pass\_rate} \geq \tau \wedge \text{integrity\_pass\_rate}=0]$。
- **理论核心——Theorem 1（PAC 二分化）**：定义 $\varepsilon = \mathbb{E}_{x\sim\mathcal{D}}[\text{TV}(A(x), B(x))]$ 为评测分布上的输出分布偏移。若 $\varepsilon = 0$，则任意基于 N 个 i.i.d. 样本的行为测试的区分优势 $|\Pr[T=\text{tampered}|B] - \Pr[T=\text{tampered}|A]| \leq N\varepsilon = 0$，即完全不可区分；反之若 $\varepsilon > 0$，存在 $N = O(\varepsilon^{-2}\log(1/\delta))$ 的测试以误差 ≤δ 区分。该 bound 适用于所有 output-only 评估器（规则、LLM judge、fingerprint），而身份通道以一次 manifest 读取即可判定 ε=0 类，与样本量无关。
- **触发条件型篡改（E4）构造**：在 system prompt 中注入仅在特定 trigger token 出现时激活的恶意指令，评测集不含 trigger，故行为完全保留；外部触发后 policy-violation 翻转为 72.5%、secret leak 为 60.0%。
- **隐身篡改（E8）构造**：将相同触发逻辑通过 chat template（未列入 manifest 的组件）注入，identity 通道因 manifest 完整而 100% pass，同时行为也 100% 匹配 clean，双重逃避。

## 实验与结果
- **数据集与模型**：7,560 个打分示例（9 次运行：3 组 model pair × 3 个 random seed），model pair 为 Qwen2.5-14B→7B、Qwen2.5-7B→1.5B、Llama-3.1-8B→Qwen2.5-7B；测试套件含 120 个合成任务/条件（事实 QA、策略允许/拒绝、工具选择、secret leak 拒绝、canary 探针）。
- **评估器阶梯主结果（Table 2）**：mean tamper acceptance 单调上升——Standard 88.6% → Qwen3-8B 89.3% → GPT-5-nano **94.1%**；identity 对所有 tamper 均为 0%。最强结果：GPT-5-nano 接受 94.1% 篡改输出，相比规则评分提升 **+5.5pp**（相对提升约 6.2%）。
- **触发条件型篡改（Table 5+E4）**：ordinary suite 上 100% 决策匹配、harm 13.3%（clean baseline）；triggered probes（n=58）上 harm 升至 **81.0%**（seed 42）；按模型能力缩放：Llama-3.1-8B **48.3%** → Qwen2.5-7B **81.0%** → Qwen2.5-14B **100%**。
- **隐身篡改 E8（Table 5）**：identity=100%，match=100%，triggered harm=**86.2%**，同时逃避两个通道。
- **严格规则评分局限（Table 3）**：即使 τ=0.90 达到 0/6 tamper 接受，但 clean acceptance 仅 85.4%（Standard）/75.9%（Canary），即会误杀干净部署。
- **Behavioral fingerprinting（LLMmap 基线）**：三分类准确率 76.7%（高于 33.3% 随机），但仅覆盖 1/6 组件（model），且 4.5× 参差降级仍存在 37.5% 碰撞率。
- **公开基准复现（Table 4）**：BFCL-simple 上 tampered 99.9% vs clean 100%；GSM8K 上 27.6% vs 27.7%；AgentDojo banking（n=16）上 prompt tamper 50.0% vs clean 43.8%（行为不可见）；model substitution（7B→1.5B）utility 降至 25.0%（可见）。所有 setting 下 identity 均 0%。

## 相关工作脉络
1. **LLM-as-judge / 评估完整性（Zheng et al. 2023, Zhang et al. 2024, Singh et al. 2025）**：聚焦"给定固定系统时分数是否可信"；本文反向追问"给定固定分数时系统是否仍是声明的那个"，提出身份通道作为正交坐标。
2. **Agent / tool 评测基准（Ruan et al. 2024, Yao et al. 2024, Lu et al. 2025, Andriushchenko et al. 2025）**：关注长尾风险、多步交互、工具使用；本文关注这些基准的部署身份未被报告的结构缺陷。
3. **Prompt injection / retrieval poisoning（Abdelnabi et al. 2023, Yi et al. 2025, Zhan et al. 2024, Zou et al. 2025）**：评估特定攻击成功率；本文借鉴其攻击面设计但目的不同——测量 output-only 评估能否察觉组件变更。
4. **模型指纹与归因（Pasquini et al. 2025, Pei et al. 2025）**：证明可用于 model 识别（76.7%）但仅覆盖 1/6 组件、且有 37.5% 碰撞下限，与 Theorem 1 的 ε=0 类边界一致。
5. **模型替代与部署漂移（Chen et al. 2024, Cai et al. 2025）**：发现商业 LLM 服务行为随时间漂移、输出统计测试需大量查询；本文将其推广至全部 6 个栈组件并聚焦于评测报告层面的报告责任。
6. **软件供应链证明（Torres-Arias et al. 2019 in-toto, SLSA Framework 2026）**：提供 artifact 完整性机制；本文将其引入 NLP 评测报告规范，建立"行为分数 + 身份证据"双轨报告框架。

## 局限性与未来方向
1. **主要实验使用合成任务与单组件篡改**，未覆盖多组件联合攻击或能伪造 manifest 的 adversarial 场景；E8 实验显示 undeclared 组件是具体缺口。
2. **测试模型范围为 1.5B–20B**，未覆盖 frontier-scale 部署系统；GPT-5-nano 代表当前商业 judge，但其自身部署身份的可靠性未被独立验证。
3. **Theorem 1 的 PAC 二分化仅覆盖被动 i.i.d. 采样**；adaptive probe 识别触发型篡改的下界（预算 ≥ 1/ρ，ρ 为 trigger 密度）仅作为未来工作提出，尚未给出算法。
4. **Manifest 是软件级证据**，缺乏 hardware-rooted 保证；graded descriptor 的更高 Assurance 层级（signed provenance / runtime attestation）未在本次实验中实现。
5. **作者自述未来三方向**：（a）adaptive-probe identification budget 的形式化与算法；（b）对主流 agent 框架做 longitudinal deployment-drift 研究；（c）leaderboard rank decomposition——在控制组件交换下重测排行榜验证模型排名与栈排名是否重合。

## 研究启发与可借鉴点
1. **双通道评估框架可直接迁移**：任何 Agent/LLM 评测流水线均可引入 manifest-hash 身份通道，以极低成本（一次组件哈希对比）获得 output-only 评估无法提供的部署真实性保证，适合作为论文评测章节的标准附属报告。
2. **触发条件型 stealth tamper 构造方法**（在评测分布外注入条件性后门）可作为 red-teaming 模板，用于检验评测集对"off-distribution harmful behavior"的检测盲区。
3. **"越强评估器越盲"的发现提示**：在构建更高强度的 LLM judge 时，需同步引入 identity/ provenance 校验，否则性能提升可能被系统性盲点抵消；对 judge 本身也应报告其 runtime 身份。
4. **分级部署身份描述符的设计模式**（self-declared → build provenance → hardware attestation）可推广至其他 AI 评估场景（如 fine-tuning 溯源、RAG pipeline 验证），形成可复用的报告规范。
5. **与团队方向的潜在结合点**：若团队关注评测鲁棒性、安全对齐或 Agent 部署监控，可将 DIB-Bench 的六种 tamper 类型接入现有评测管线，形成"行为分 + 身份验真"的双轨评测报告作为论文的新颖性贡献。

## 关键术语表
**Deployment-Integrity Blindness（DIB）**：输出-only 评估器对部署运行时栈未经声明的篡改保持 silent acceptance 的系统性缺陷。
**Evaluator Ladder**：从 rule-based scorer 经本地 LLM judge（Qwen3-8B）到商业 reasoning model（GPT-5-nano）的递增强度评估器谱系，本文的核心对照实验设计。
**S-equivalent Deployments**：在同一输出评测集 S 上产生相同 acceptance 结果的部署对；S-等价类内所有部署对行为测试不可区分（Proposition 1）。
**Behavioral Sample-Complexity Dichotomy（Theorem 1）**：以输出分布 TV 偏移 ε 为界，ε=0 时任意 N 样本的行为测试完全不可判；ε>0 时需 $O(\varepsilon^{-2}\log(1/\delta))$ 样本即可判。
**Trigger-Conditional Tamper（E4）**：在 system prompt 中注入仅在特定 trigger token 出现时激活的恶意指令，评测集上 100% 决策匹配 clean，外部触发后有害率达 48–100%。
**Stealth Tamper（E8）**：通过未列入 manifest 的组件（如 chat template）注入触发条件逻辑，同时逃避输出评估器和身份通道两个检测通道。
**Deployment-Identity Descriptor**：对每个运行时栈组件记录内容哈希及获取方式（self-declared/build provenance/runtime attestation）的分级报告规范，建议作为评测报告的标配附件。
**Blind@τ**：套件级盲点指标，定义为 output_pass_rate ≥ τ 且 integrity_pass_rate = 0 的篡改条件占比。

## 可复现要素
- **数据集**：DIB-Bench 合成任务（120 tasks/condition × 7 conditions × 9 runs = 7,560 示例）；复现中使用的公开基准为 BFCL-simple、GSM8K、AgentDojo（各基准均有公开许可）。
- **代码/权重**：部署模型（Qwen2.5-1.5B/7B/14B-Instruct、Qwen3-8B、gpt-oss-20B、Llama-3.1-8B-Instruct）均为 Apache 2.0 或社区许可开源；推理引擎 vLLM（Apache 2.0）；GPT-5-nano 通过 OpenAI API 访问（论文未声明开源，受其 ToU 约束）。
- **关键超参**：temperature=0，top-p=1，max_new_tokens=160，context_length=2048；Qwen3-8B judge thinking 关闭、temperature=0；GPT-5-nano 使用 minimal reasoning effort；suite threshold τ=0.80（除非注明）；95% bootstrap CI 基于 1,000 resamples。
- **计算环境**：NVIDIA RTX A6000 GPU（48 GB）单卡；vLLM 0.21.0、PyTorch 2.11.0、Transformers 5.6.0、Python 3.12。
