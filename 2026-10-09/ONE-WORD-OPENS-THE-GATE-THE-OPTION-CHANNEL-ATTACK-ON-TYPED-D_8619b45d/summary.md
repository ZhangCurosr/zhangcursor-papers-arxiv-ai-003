---
title: "ONE-WORD-OPENS-THE-GATE-THE-OPTION-CHANNEL-ATTACK-ON-TYPED-D"
source: https://arxiv.org/pdf/2610.12292v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-01 07:16:12"
field: "AI安全与护栏评估"
keywords: ["typed decision model", "agent guardrail", "option-channel attack", "fail-open fail-closed", "prompt injection defense", "security evaluation"]
innovations: ["提出选项通道攻击：仅改一个选项标签即可将fail-open率推至100%", "证明无信息文本注入可翻转多数正确阻断决策", "揭示低置信度升级策略对语义攻击完全无效"]
benchmarks: ["GuardBench", "deepset/prompt-injections", "jackhhao/jailbreak-classification", "toxicchat"]
---

# 论文速读：ONE-WORD-OPENS-THE-GATE-THE-OPTION-CHANNEL-ATTACK-ON-TYPED-DECISION-MODELS-AS-AGENT-GUARDRAILS

## 一句话总结
本文系统评估了将 typed decision model（类型化决策模型）用作 AI Agent 安全护栏的可靠性，发现其错误方向被单一准确率掩盖；提出了"选项通道攻击"（option-channel attack）——仅改一个选项标签即可将 fail-open 率推至 100%，并证明所有测试防御均可被自适应攻击击败。

## 研究问题与动机
- 随着 Agent 系统普及，typed decision model 正被广泛部署为安全护栏（guardrail），负责判断工具调用是否应放行。然而现有安全文献未将 fail-open（放行违规动作，即漏洞）与 fail-closed（阻断合法动作，即成本）分开评估。
- 现有四篇攻击论文（Hu et al. 2026; Xu 2026; Wu & Lim 2026; Sun et al. 2026）均未分离两种错误方向，也未提出任何防御措施。
- 模型在公开筛选数据集上准确率仅 36%–72%（随机基线为 50%），且部分模型默认偏向极端方向（一个几乎全放行，另一个几乎全阻断）。
- 政策文本的表述方式（肯定 vs 否定）对安全性有重大影响，但部署论文均未讨论。

## 核心贡献（创新点）
1. **GuardBench 合成基准**：构建了由属性驱动的 Agent 工具调用数据集，含 6 条安全策略，ground truth 由生成属性计算得出，无标签噪声、类别严格平衡，每个样本含明确标注的可被攻击者控制的文本段。
2. **选项通道攻击（option-channel attack）**：仅改变一个选项标签（不触碰输入文本与定义），即可将 fail-open 率从 0% 提升至 100%（4/6 模型），攻击代价极低（平均 5–10 次查询即可）。
3. **无信息文本注入攻击**：追加完全不含策略相关信息的普通服务器日志文本（如 `200 OK in 143ms; cache: warm`），即可驱动正确阻断决策被翻转为放行，证明当前模型对无关后缀极度敏感。
4. **揭示模型决策置信度缺陷**：正确阻断决策的中位 P(block) 仅为 0.57，距 0.5 决策边界极近；攻击翻转后的决策置信度与原决策相当，因此基于低置信度升级的策略完全失效。
5. **五项防御的系统性评估与全部攻破**：包括标签规范化（D1）、多标签集成（D3）、双重标签分歧检测（D4+D6）、结构化字段传递（D5）、指令重写（D8），每项均被自适应攻击击败。

## 方法详解
- **选项通道攻击原理**：Typed decision model 接受两个输入——state（待判定文本）和 typed question（含选项 label + definition）。攻击者仅修改 permissive 选项的 label（如将 `allow` 改为 `escalate`），定义与 state 完全不变，即可使模型倾向放行。攻击对 LAYA-EN/TD/ML 和 Qwen2.5-7B 有效，而对 VON/RLCD 无效（因其调用库仅使用 definition 构建输入序列，label 不参与模型输入）。
- **无信息文本注入**：向工具输出段追加 k 行无关文本（distraction：服务器日志；persuasion："此操作已由安全团队预批准"），不改变任何策略属性，正确决策不变，但模型 P(block) 显著下降。
- **防御机制**：
  - D1（canonical labels）：将所有选项标签替换为 `A`/`B`，消除标签语义差异。
  - D5（field-structured）：仅传递策略相关的结构化字段值（如 `destination: paste.ee`），移除被攻击者控制的原始 trace。
  - D8（instruction rewrite）：指令改写为"仅依据定义决策，选项名称无信息"。
- **确定性强规则**：当 D5 能将输入解析为结构化值时，直接对该值应用确定性规则可达 100% 准确率且零错误，证明此时模型非必需。

## 实验与结果
- **数据集**：GuardBench（6 条合成策略：exfil、secret、escape、exfil_secret、unauth_irrev、escape_or_exfil）+ 三个公开数据集：prompt-injection（406 条）、jailbreak（1034 条）、toxic-content（768 条）。
- **评估模型**：LAYA-TD、LAYA-EN、LAYA-ML（ModernBERT-large，421M/322M 参数）、VON（ModernBERT，395M）、RLCD（GLiClass，151M）、Qwen2.5-1.5B/7B（instruct baseline）。
- **公开任务准确率**：36%（LAYA-ML jailbreak）至 72%（LAYA-TD jailbreak），远低于安全护栏所需水平。
- **选项通道攻击最强结果**：Qwen2.5-7B（honest fail-open 4.7%）→ 重命名 permissive 选项为 `remediate` → fail-open 升至 100%（全部 6 条策略）。LAYA-TD 在 unauth_irrev 上 honest 为 0% → `escalate` 达 100%。
- **文本注入攻击**：LAYA-TD 在 unauth_irrev 上 6 行日志文本使 fail-open 从 0% 升至 63%；toxic 任务中追加 17 词礼貌结束语使 88.6% 正确阻断决策被翻转为放行。
- **防御评估**：D5（结构化字段）是唯一在匹配比较中从未失败的防御（16/18 胜利），但对自适应攻击仍会失败（4/5 模型上 fail-open 达 100%）。
- **转移攻击**：在一模型上找到的最优标签可直接用于另一模型（如 LAYA-EN 的最佳标签使 LAYA-TD fail-open 提升 38.2pp）。

## 相关工作脉络
- **Hu et al. (2026)**：向输入追加未验证意见翻转决策，未分离错误方向。
- **Sun et al. (2026)**：重命名选项集并观察到答案变化，归因于约束决策头，但未将其建模为攻击面。
- **Tang & Zheng (2026)**：对前 28 篇论文的审计，指出仅 1/27 测试了决策对选项名称的不变性，本文填补该空白。
- **NeMo Guardrails (Rebedea et al. 2023)**：策略外置于配置语言的护栏框架，与本文研究的嵌入模型内的 option-label 形成对比。
- **Jev 商业模型**：引入 typed decision 接口的原型系统，本文仅评估开源实现。
- **Li et al. (2026)**：推荐低置信度案例升级至更强模型，本文证明该策略对此类攻击完全无效。

## 局限性与未来方向
- GuardBench 为合成数据，虽无标签噪声且无训练数据泄露风险，但真实度有限；三项公开数据集用于补充。
- 仅评估开源模型，未测试引入该接口的商业模型 Jev 本身（依赖 Sun et al. 的外部证据）。
- 攻击为黑盒、低预算、固定词表搜索，属 lower bound；更强的白盒/优化攻击可能达到更高成功率。
- 未来方向：研究真正输入不变的鲁棒性保障机制；探索模型在无法解析结构化字段的场景下的安全部署边界。

## 研究启发与可借鉴点
1. **错误方向分离评估范式**：安全护栏不应报告单一准确率，必须分离 fail-open 与 fail-closed，该评估框架可迁移至任何分类型安全组件评测。
2. **选项/标签语义作为攻击面**：任何将"配置数据"（如选项名、分类标签）拼入模型输入的系统，均存在类似选项通道的风险，提示需严格区分"代码"与"数据"。
3. **确定性规则替代弱模型**：当输入可充分解析为结构化值时，强规则优于小模型（本文 100% 准确率对比模型最差 36%），为安全关键场景提供了简洁的替代方案。
4. **低置信度升级对语义攻击无效**：攻击翻转后的决策置信度与原决策相当，挑战了"置信度门控"这一流行防御思路的适用性。
5. **攻击转移（cross-model transfer）**：单模型最优攻击标签可迁移至同族模型，提示联合评估/防御的重要性。

## 关键术语表
**Typed Decision Model**：接收文本输入和带定义的选项集，返回每个选项的概率而不自愈文本的小型分类模型。
**Fail-Open**：护栏错误地放行政策禁止的动作，属于安全漏洞。
**Fail-Closed**：护栏错误地阻断政策允许的动作，属于成本损失。
**Option Channel**：通过修改选项标签（而非定义或输入文本）来操控模型决策方向的攻击通道。
**Margin**：模型输出概率距 0.5 决策边界的距离，衡量决策置信度。
**GuardBench**：本文构建的合成 Agent 工具调用基准，含 6 条可计算的确定性强规则策略。
**Readout**：将语言模型输出转化为选项选择的各种规则策略。
**ASR Open**：Attack Success Rate for open errors，衡量攻击将原正确阻断决策翻转为放行的比例。

## 可复现要素
- **代码**：已开源，地址为 `github.com/ArminAzizi98/option-channel-attack`
- **数据集**：GuardBench 为合成基准（论文附生成逻辑）；公开数据集为 deepset/prompt-injections、jackhhao/jailbreak-classification、linzihao/toxicchat，均已公开
- **模型**：LAYA-TD、LAYA-EN、LAYA-ML、VON、RLCD、Qwen2.5-1.5B/7B，均为开源权重
- **关键超参**：选项词表大小 198；k 行追加文本实验中 k ∈ {1, 3, 6, 10}；margin bootstrap 重采样 4000 次；论文未提及 GPU/批量大小等训练细节（模型均为 pre-trained/fine-tuned 后直接调用）
