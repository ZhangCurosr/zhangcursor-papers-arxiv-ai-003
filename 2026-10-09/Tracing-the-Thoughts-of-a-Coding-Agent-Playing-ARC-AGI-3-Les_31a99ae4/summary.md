---
title: "Tracing-the-Thoughts-of-a-Coding-Agent-Playing-ARC-AGI-3-Les"
source: https://arxiv.org/pdf/2610.11450v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-01 14:29:59"
field: "agent持续学习与可解释性"
keywords: ["持续学习", "coding agent", "ARC-AGI-3", "程序化记忆", "负迁移", "白盒分析", "知识转移"]
innovations: ["提出跨任务思想追踪测量协议，直接计数保留与转移", "发现跨任务知识转移以重新推导而非跨任务引用为主要机制", "提出将硬编码常量转为参数的负迁移修复方法"]
benchmarks: ["ARC-AGI-3"]
---

# 论文速读：Tracing-the-Thoughts-of-a-Coding-Agent-Playing-ARC-AGI-3

## 一句话总结
本文研究一个基于冻结基础模型的编程智能体在ARC-AGI-3抽象推理任务序列上的持续学习方式：智能体通过读写文件和执行脚本积累知识，论文提出一种测量协议直接追踪每条"思想"（信念、规则、计划）从形成到修正/废弃的全生命周期，揭示其学习本质是**重新推导**而非**跨任务引用**，并发现最昂贵的错误是硬编码值被带入不再成立的新任务。

## 研究问题与动机
1. **核心问题**：在冻结模型权重的前提下，编程智能体如何在一系列任务边界上决定保留什么知识、重建什么、放弃什么？这与参数化模型的持续学习（continual learning）问题同源，但知识存储在文件而非权重中。
2. **现有方法不足**：传统持续学习方法归因于稳定性-可塑性权衡，依赖参数正则化、重放或参数隔离；而文件系统的智能体没有缓冲区预算压力，丢弃决策更自由，但此前缺乏直接读取记忆artifact来度量保留/转移的方法。
3. **可解释性缺口**：现有可解释性研究（如电路追踪）代价高昂且每prompt一次；本文的智能体将状态写在文件中，可提供从第一个任务到最后一条操作的全部可追踪轨迹。
4. **负迁移风险**：ARC-AGI-3的每个关卡共享同一接口但规则可能突变，上一关的正确理论在新关可能直接失效，现有benchmark缺乏对"携带错误信念跨任务"的细粒度分析。

## 核心贡献（创新点）
1. **提出测量协议**：将每条脚本分配给写入时的任务，追踪其在任务边界处的四种结局（被引用复用、被复制复用、被重写、被废弃），使保留与转移次数可直接计数——此前只能通过任务表现间接推断。
2. **实证刻画跨任务知识转移机制**：发现智能体跨任务转移知识的方式是**重新推导**（re-derivation）而非**跨任务引用**——智能体携带学到的规则，但丢弃建立该规则的脚本；重写新版本是分离"环境恒定规律"与"已离开任务特有细节"的时刻。
3. **发现负迁移来源及修复方法**：最昂贵的错误是硬编码值被带入不再成立的任务；唯一有效的修复是将常量变为参数（parameterization），使新任务必须重新供给该值——这一修复仅在脚本中可行，notes无法表达。
4. **白盒分析持续学习过程**：在三个backbone、两个模型家族、七次运行、689个任务边界、4,786条脚本上应用协议，无需访问模型本身即可完整追踪每条思想的生命周期。

## 方法详解
**智能体架构**：冻结基础模型运行在固定harness内，通过编写和执行Python/shell脚本行动。记忆由两部分组成：① **scripts**（可执行代码，Python文件）；② **notes**（纯文本信念记录）。harness向logs.txt追加每次交互记录，智能体只能读取该文件路径。

**评估环境**：ARC-AGI-3提供25个interactive reasoning games，每个game是level序列。每个level = 一个任务，边界无提示到达（仅通过score变化感知）。接口（64×64网格、ACTION1-ACTION7、RESET）跨关卡不变，但布局、对象、获胜条件变化。

**测量协议**（Section 3.4，Table 2）：
- 每条脚本分配给写入时的任务；记录689个边界处的四种结局：
  - **reused by reference**：边界后脚本import/exec前脚本
  - **reused by copying**：边界后不同脚本包含与原脚本相同函数体
  - **rewritten**：边界后写入新版本脚本
  - **abandoned**：既未重写也未复用

**工具链**：六工具sandbox内运行（无网络），ReAct循环约10次tool call：搜索logs.txt→写parser→运行→检查输出→修订→发出计划。每次action消耗score，运行脚本查logs.txt免费，激励智能体用脚本回放历史而非重复探测。

**损失/评估**：RHAE（Relative Human Action Efficiency）——比人类上中位数首次通关更少action得分；共183个可用levels。

## 实验与结果
**数据集**：ARC-AGI-3公开demo dataset，25个games，共183个levels。

**实验设置**：七次运行（Table 1），三种backbone（Anthropic Opus 4.7/Opus 5 × 4次 + Qwen3.8-27B-FP8 × 3次），跨两个模型家族。总计689个边界，4,786条脚本（排除Qwen日志库）。

**主要结果**：

| 指标 | 数值 |
|------|------|
| 跨任务引用存活率 | 33/630（5.2%），大部分脚本嵌入当前level状态，关卡变化即失效 |
| 废弃率 | 74%（1,726/2,319）——多数脚本在边界后被完全放弃 |
| 重写比例 | 523/2,319（22.6%） |
| 新版本函数保留率 | R4丢弃56%，R2/R3丢弃62%/61%，R1丢弃38%，Qwen丢弃38%-74% |
| 跨边界复制复用 | 72对（vs 引用复用仅33对） |
| Opus 5跨边界复用 | 仅R2-R4有复用，R1和Qwen三次运行均无 |
| Notes增长模式 | 只增不改，修正以append方式添加，矛盾共存 |

**最强结果**：Run R4（Opus 5，去除prompt先验）以99.95% RHAE通过全部183个levels，生成734条脚本、1,357 kB notes。

**关键发现**：
- 重写的本质：保留编码游戏规则的功能，替换编码当前关卡布局的功能（如su15游戏L5→L6，5个功能保留、4个替换）
- Notes中的矛盾：R4游戏wa30中，两格player模型→一格修正→80 actions后又恢复两格，两者并存65行，无撤消标记——由外部logs.txt裁决
- 负迁移案例（m0r0游戏）：水平控制方向在L4→L5→L6三次翻转，Opus 5三次运行均携带L4方向进入L5；修复方式：将固定值改为参数`def solve(polarity, ...)`或`S3 = int(sys.argv[1])`
- Qwen反向案例（tu93游戏）：L2用参数`def move(r, c, mv, n=2)`，L3重写变回固定表`DIRS={'ACTION1':(-2,0),...}`——参数化被逆转

## 相关工作脉络
1. **Parisi et al. (2019), De Lange et al. (2021), Wang et al. (2023b)**：持续学习综述，按参数正则化/重放/参数隔离分类——本文继承"保留多少/获取多少/先前学习是否帮助后续"的问题，但移除了缓冲区预算约束。
2. **Aljundi et al. (2019a, 2018, 2019b)**：task-free持续学习、正则化保护、梯度样本选择重放——本文无预先告知的任务边界，智能体自主决定何时变更模型。
3. **Asawa et al. (2026) CL-Bench**：发现在context window内使用完整上下文优于多个专用记忆系统——本文memory system同样验证了"外部记录+可执行代码"路线的有效性。
4. **Park et al. (2023) Generative Agents, Packer et al. (2023) MemGPT**：记忆系统以外部记录替代参数遗忘，但引入检索瓶颈和整合错误——本文的notes保留相同"仅append不删除"模式，scripts提供可执行校验。
5. **Wang et al. (2023a) Voyager, Tang et al. (2026) WikiSkill**：技能库假设collection只增长——本文边界计数反驳此前提，智能体大部分artifact被废弃。
6. **Fox et al. (2026) PRO-LONG**：程序化记忆设计基础——本文loop取自PRO-LONG，仅更换backbone/prompt/operating mode。
7. **Ameisen et al. (2025), Lindsey et al. (2025)**：电路追踪以高cost解析模型内部计算——本文以文件trace提供同等可解释性但零额外cost。
8. **Yao et al. (2023) ReAct**：推理-行动循环基础——本文每个turn为约10次tool call的ReAct循环。

## 局限性与未来方向
1. **样本量有限**：仅七次运行，横跨三个backbone，未做统计显著性检验（作者明确不排名、不报告置信区间）。
2. **单向评估**：已清除的level不再回访，backward transfer未定义，所计"遗忘"仅为artifact废弃而非competence损失。
3. **Qwen日志缺失**：Qwen运行的console logs未保留，边界从level number推断，94%脚本可分配，部分不可追溯。
4. **单一benchmark**：仅在ARC-AGI-3上验证，其他interactive reasoning环境（如CL-Bench）未知。
5. **Notes矛盾累积**：notes中错误信念不被删除，仅靠外部log裁决——长期运行下notes可能无限膨胀且自我矛盾。
6. **未来方向**：可探索notes的自我清理机制；可将参数化修复泛化为"常量→参数"的模式匹配；可扩展到多agent协作场景下的知识转移分析。

## 研究启发与可借鉴点
1. **白盒持续学习分析框架**：文件系统的memory设计使保留/转移/废弃可直接计数——任何将知识写入可执行代码的智能体系统均可复用此测量协议，无需侵入模型内部。
2. **参数化作为负迁移修复**："将硬编码值变为运行时参数"是本文发现的最有效修复——可迁移至任何涉及跨任务知识迁移的agent系统，作为防止stale memory的正则化策略。
3. **选择性遗忘优于灾难性遗忘**：智能体只丢弃belief不丢弃evidence（logs.txt永远保留），使错误丢弃仅导致re-derivation cost而非competence loss——此设计原则可用于构建鲁棒的持续学习系统。
4. **Notes-Ground Truth分离架构**：notes作为dated record of past beliefs而非current statement，由外部log裁决矛盾——可借鉴于需要长期记忆累积且允许信念修正的agent设计。
5. **重写而非编辑的memory更新策略**：智能体新版本的script与旧版本仅62-73%函数重叠，本质是"重新推导"而非"in-place修改"——提示我们在设计agent memory时，鼓励重写可能比编辑更安全。

## 关键术语表
**ARC-AGI-3**：ARC Prize Foundation设计的interactive reasoning benchmark，25个games每个含sequence of levels，测试agent在新任务上skill acquisition效率。
**Task boundary / Level boundary**：连续两level之间的转换瞬间，即上一level清除、下一level加载的时刻——在单episode内无提示到达，是本Paper的task单元。
**Continual learning（持续学习）**：在任务序列上学习而不灾难性遗忘先前能力的研究范式，本文以非参数化agent视角重新审视。
**Catastrophic forgetting（灾难性遗忘）**：新增任务导致先前能力骤降的现象——本文agent因只丢弃artifact不丢弃evidence而避免此问题。
**Stability-plasticity dilemma（稳定性-可塑性权衡）**：保留已有知识 vs 获取新知识的内在张力——本文agent以隔离策略（isolation strategy）解决，不依赖参数调节。
**Negative transfer（负迁移）**：先前任务知识损害新任务表现的现象——本文发现硬编码常量跨任务携带是最常见来源。
**RHAE（Relative Human Action Efficiency）**：ARC Prize计分方法，以相对人类上中位数首次通关action数越少得分越高。
**Programmatic memory（程序化记忆）**：知识以可执行代码（scripts）形式存储于文件系统的设计，源自PRO-LONG。

## 可复现要素
- **数据集**：ARC-AGI-3公开demo dataset（25 games），来源 https://arcprize.org/arc-agi/3
- **代码**：Loop取自PRO-LONG（Fox et al., 2026）未修改；协议代码论文未明确开源声明，"论文未提及"
- **模型权重**：Anthropic Opus 4.7、Opus 5；本地Qwen3.8-27B-FP8 on 8 H100s
- **关键超参**：action cap（R1-R4为20，R5-R7为50）；R4去除prompt priors
- **运行配置**：详见Table 1（七次运行差异在backbone、prompt、action cap、operating mode）
- **记忆文件结构**：每game一个folder含Python scripts + notes*.md；logs.txt由harness写入
