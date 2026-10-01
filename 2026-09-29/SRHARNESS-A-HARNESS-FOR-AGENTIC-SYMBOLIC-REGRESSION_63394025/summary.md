---
title: "SRHARNESS-A-HARNESS-FOR-AGENTIC-SYMBOLIC-REGRESSION"
source: https://arxiv.org/pdf/2609.35501v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 22:09:48"
field: "符号回归与科学发现"
keywords: ["Symbolic Regression", "Agent Harness", "Scientific Discovery", "LLM Agents", "Expression-Based Views", "Persistent State"]
innovations: ["提出独立于搜索策略的Agent Harness层，支持可组合科学操作、持久化状态和轨迹生命周期管理", "通过表达式视图实现异构科学操作在原始/变换/候选衍生数据间的统一接口", "构建匿名化评估设置分离语义先验与数据驱动的符号恢复能力"]
benchmarks: ["LLM-SRBench", "LSR-Synth", "LSR-Transform", "LSR-Transform-Anon"]
---

# 论文速读：SRHARNESS-A-HARNESS-FOR-AGENTIC-SYMBOLIC-REGRESSION

## 一句话总结
论文提出了SRHarness，一个面向Agent符号回归的领域特定运行框架，通过可组合科学操作、持久化科学状态和轨迹生命周期管理三大机制，在不改变底层模型和搜索策略的前提下，显著提升符号回归的数值泛化能力和符号恢复准确率。

## 研究问题与动机
- **现有方法的局限**：当前Agent符号回归方法将LLM从方程生成器提升为自主控制器，但性能差异混杂了搜索策略差异与运行时支撑差异，缺乏独立于策略的运行时层抽象。
- **科学探索的特殊需求**：符号回归需要在长时程交互中维护候选假设集合、支持对原始/变换/残差等多视图数据的操作、协调继续/分支/重启等轨迹转换。
- **工具不等于优势**：将相同科学工具直接暴露给Codex并未复现SRHarness的增益，说明价值在于运行时对决策、状态和轨迹的结构化管理。
- **语义先验的可迁移性**：现有方法依赖变量语义和科学描述，需要验证纯数据驱动的符号恢复能力。

## 核心贡献（创新点）
- **提出独立的Agent Harness层**：将运行时支撑从搜索策略中解耦，使harness成为可复用的一等公民组件，区别于SR-Scientist等方法围绕特定策略定制支持机制。
- **可组合科学操作接口**：通过表达式视图（expression-based views）让同一分析/拟合过程作用于原始变量、变换变量和候选残差，无需每次重建数据集引用。
- **持久化科学状态与紧凑视图**：将候选方程及其证据存储于对话上下文之外，并通过Pareto前沿和Top-k投影策略向模型暴露有界视图。
- **轨迹生命周期管理**：支持继续、分支、重启和终止的协调， preserving科学信息跨长时程搜索的一致性。

## 方法详解
- **可组合科学操作**：科学操作表示为 $r = \mathcal{A}(q; \kappa)$，其中$q$为Agent指定的科学参数和选项，$\kappa$为harness管理的执行上下文（数据绑定、训练-验证划分、运行时配置）。操作支持基于表达式的视图，同一过程可作用于原始变量$x_i$、变换视图$\log x_i$或候选残差$y - f(x)$。候选产出操作遵循统一的"候选-证据"契约。
- **持久化科学状态**：状态$S_t$存储已评估的候选方程、量化评估、复杂度、支持证据和溯源信息，跨对话轨迹持久化。模型-facing视图$v_t = P_\rho(S_t)$通过投影策略$\rho$生成，实现为fit-complexity Pareto视图和Top-k视图。
- **轨迹生命周期管理**：支持continuation（推进现有对话）、branching（从共同前缀创建替代路径）、restart（从持久状态的模型视图初始化新对话）。参考实现使用可配置的$R \times C \times L \times K$调度器（重启轮次×独立分支×精炼深度×本地响应采样），主实验配置为R1-C1-L30-K1。

## 实验与结果
- **数据集**：LLM-SRBench包含两个互补赛道，LSR-Synth（128个问题，含物理/化学/生物/材料科学，评估ID和OOD数值泛化）和LSR-Transform（111个变换方程问题，强调精确符号恢复）；另构建LSR-Transform-Anon（移除科学描述和变量语义的匿名化版本）。
- **基线**：PySR（传统符号回归）、LLM-SR、IGSR、SR-Scientist（三种不同粒度的LLM方法），均使用官方开源实现匹配 backbone 复现。
- **主要结果**：
  - LSR-Synth（DeepSeek-v4-flash-0731）：物理ID 84.46%/OOD 78.00%，化学ID 95.24%/OOD 78.87%，生物ID 95.14%/OOD 87.09%，材料ID 95.06%/OOD 96.00%，SA 6.20%，平均搜索时间11.11分钟。
  - LSR-Transform：SRHarness达到93.69% SA，expression complexity仅14.64，远超SR-Scientist的62.16% SA和17.50复杂度；LSR-Transform-Anon上SRHarness保持72.97% SA，SR-Scientist仅39.64%。
  - Harness独立性验证：相同DeepSeek-v4-flash-0731下，SRHarness SA 72.97% vs Codex仅20.72%；SRHarness with DeepSeek接近Codex with GPT-5.5（68.47%），而将相同工具暴露给GPT-5.5的Codex SA反而降至64.86%。
- **效率**：SRHarness在LSR-Synth上比LLM-SR快5倍（11.11 vs 57.67分钟），比IGSR快2倍；API成本\$0.03/问题，优于多数基线。

## 相关工作脉络
- **LLM-guided SR（LLM-SR、LaSR、SGA等）**：LLM作为候选生成或修改组件，嵌入预定义搜索流程；SRHarness将LLM提升为自主控制器。
- **Agentic SR（SR-Scientist、KeplerAgent、MOT-SR等）**：这些方法主要开发搜索策略，支持机制围绕各自策略定制；SRHarness提供独立于策略的可复用运行时支撑。
- **Agent Harnesses for Scientific Search（Beaver、多模态报告生成harness等）**：面向证据合成或环境交互的harness；SRHarness专门支持异构分析/拟合/搜索操作共享科学量。
- **传统符号回归（PySR）**：进化搜索算法；提供非LLM参照点，验证Agent方法优势。

## 局限性与未来方向
- **主干模型选择的非单调性**：更大参数规模模型并非总是更强（如DeepSeek-v4.1-flash优于DeepSeek-v4-pro），与长时程Agent基准的相关性高于静态科学QA能力，模型选择仍需谨慎。
- **Material Science领域接近饱和**：该领域问题因变量范围覆盖平滑区域，低次多项式即可逼近，可能掩盖方法真实能力。
- **轨迹预算分配权衡**：消融显示增加重启/分支不如保持精炼深度有效，但未探索更复杂的调度策略。
- **科学工具扩展性**：添加拟合操作提升SA但增加复杂度，需权衡工具丰富性与搜索效率。

## 研究启发与可借鉴点
- **H层抽象价值**：将运行时支撑作为独立于模型和策略的一等公民，为Agent科学发现提供通用范式，可迁移至其他科学Agent任务。
- **表达式视图机制**：通过共享操作接口支持原始/变换/候选衍生视图，避免策略特定接口，可推广至多模态Agent工作流。
- **结构化状态管理**：持久化状态+紧凑投影的策略（Pareto视图优先于单一最佳公式）对长时程Agent设计有借鉴意义。
- **匿名化评估设置**：LSR-Transform-Anon有效分离语义先验与数据驱动发现，可作为验证Agent真正推理能力的标准协议。
- **团队结合点**：可在团队Agent系统设计中引入类似harness层，提升长时程科学发现的稳定性与可复现性。

## 关键术语表
- **Symbolic Regression**：从观测数据中发现可解释数学表达式而非拟合数值的回归方法。
- **Agent Harness**：支持Agent在长时程交互中执行的运行时基础设施层，分离执行、状态管理和治理关注点。
- **Composable Scientific Actions**：通过统一接口和表达式视图支持异构科学操作在不同数据视图间组合使用的机制。
- **Persistent Scientific State**：跨对话轨迹存储候选方程、评估证据和溯源信息的结构化状态。
- **Trajectory Lifecycle Management**：协调Agent搜索轨迹的继续、分支、重启和终止的生命周期管理。
- **LSR-Transform-Anon**：移除变量语义和科学描述的匿名化评估变体，用于分离数据驱动发现与语义回忆。
- **Pareto View**：从持久状态投影到模型-facing视图的一种策略，展示候选方程在拟合复杂度上的权衡前沿。
- **Symbolic Accuracy (SA)**：衡量发现的表达式与参考方程在结构上等价的比例。

## 可复现要素
- **数据集**：LLM-SRBench（LSR-Synth 128问题、LSR-Transform 111问题、LSR-Transform-Anon匿名变体）。
- **代码开源**：是，代码公开于 https://github.com/tsinghua-fib-lab/SRHarness。
- **权重/模型**：使用DeepSeek-v4-flash-0731、GLM-5.3-flash等OpenRouter API，未提供本地权重。
- **关键超参**：R1-C1-L30-K1调度配置（1重启轮次、1分支、30步精炼深度、1本地采样）；80/20训练-验证划分，split seed 42；输出token限制4096（DeepSeek）/8192（GLM）。
