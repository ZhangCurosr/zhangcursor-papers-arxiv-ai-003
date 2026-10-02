---
title: "What-Drives-Citations-in-Production-Large-Language-Models-An"
source: https://arxiv.org/pdf/2609.35077v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 08:15:59"
field: "大语言模型引用行为分析"
keywords: ["LLM citation", "observational study", "double machine learning", "answer engine optimization", "prompt-content alignment", "domain fixed effects", "Simpson's paradox"]
innovations: ["九方法共识框架声明LLM引用驱动因子", "揭示AEO清单效应的Simpson悖论", "证实prompt-content alignment是唯一稳健强页级预测因子"]
benchmarks: ["Discovered Labs AEO benchmarking platform", "Four production engines (ChatGPT, Claude, Google AI, Gemini)"]
---

# 论文速读：What-Drives-Citations-in-Production-Large-Language-Models-An

## 一句话总结
本研究通过对四个商业LLM引擎（ChatGPT、Claude、Google AI、Gemini）在六个月内的约200万条Web引用进行大规模观察性分析，发现**prompt-content alignment（提示词-内容对齐）**是唯一稳健且主导性的页级预测因子（β=+0.37, q≈10⁻⁷³），而传统AEO清单中的结构化信号（FAQ、Schema、Core Web Vitals等）在控制domain固定效应后效应反转或归零，揭示了AEO文献中普遍存在的Simpson悖论。

## 研究问题与动机
- **核心问题**：生产级LLM在生成答案时引用哪些Web页面，其引用频率由哪些页级特征驱动？
- **现有方法不足**：
  - 现有LLM引用研究多基于合成提示词、单一引擎或极小样本，缺乏规模与泛化性
  - 实证GEO/AEO研究使用pooled estimates（无域级固定效应），无法控制品牌混淆
  -  practitioners的AEO建议（结构化数据、FAQ、Core Web Vitals等）多来自小规模案例研究，缺乏confound control
  - 现有observational分析缺少多方法共识框架来声明因果效应

## 核心贡献（创新点）
1. **首个多引擎大规模LLM引用观察性研究**：覆盖4个生产引擎、19个B2B SaaS工作区、约200万引用，规模远超 prior work（Kumar & Palkhouski, 2025的1,100 URL；Zhang et al., 2025的55,936查询）。
   - 与已有工作的本质区别：首次引入domain-level fixed effects控制品牌混淆，并采用九方法共识协议声明效应。

2. **揭示prompt-content alignment的主导性作用**：Jaccard重叠（针对完整workspace prompt corpus计算，避免循环）是唯一的强域内预测因子（β=+0.37, Π̂=1.0, q≈10⁻⁷³）。
   - 与已有工作的本质区别：prior work（如Yang, 2025; Zhang et al., 2026）关注citation selection vs. absorption的描述性模式，本文首次通过confound-controlled多方法框架量化alignment的独立效应。

3. **发现AEO清单效应的Simpson悖论**：传统页级信号（Core Web Vitals、FAQ、Schema）在pooled数据中显示正向效应，但在domain fixed effects控制下反转或归零。
   - 与已有工作的本质区别：这是首次在生产LLM引用数据中系统展示Simpson悖论的方法学警示，挑战了practitioner consensus。

4. **提出九方法共识框架**：结合mixed-effects regression、FDR、stability-selection Lasso、DML、GAM等九种方法，制定严格的consensus protocol（五条件验证）。
   - 与已有工作的本质区别：将信息检索领域的observational分析方法正式扩展到黑盒LLM引用行为研究，提供可复用的方法论贡献。

## 方法详解
- **数据收集**：从Discovered Labs AEO基准平台采集≈2.1×10⁶条⟨prompt, engine, cited URL, position⟩元组，跨越六个月；对引用≥2次的URL抽样至10,042个独特页面，headless browser爬取并提取60+特征。
- **特征工程**：五大类特征：
  - Alignment：lexical Jaccard（vs. full workspace prompt corpus）、semantic cosine（title/intro/best-paragraph）
  - Structural：word count、outbound links、FAQ/TLDR、author bio、schema markup
  - Recency：page age、publication date flag
  - Infrastructure：real-user CWV（LCP, INP, CLS）、synthetic Lighthouse scores
  - Page type：七类标签（article, comparison, how-to, listicle等）
- **目标变量**：y = log₂(1 + n_c)，所有连续预测变量z-score标准化。
- **九方法框架**：
  1. **Mixed-effects regression with domain fixed effects**：y_i = α_d(i) + x_i^T β + p_i^T γ + ε_i，OLS + HC1稳健标准误（N=4,015, D=297）
  2. **FDR correction**：Benjamini-Hochberg q < 0.05
  3. **Stability-selection Lasso**：B=200 bootstrap，λ由10-fold CV选择，阈值Π*=1.0
  4. **Double machine learning**：正交化偏效应，交叉拟合残差化（5-fold），HC1稳健SE
  5. **Factor analysis on speed metrics**：promax旋转处理LCP/INP/CLS等共线性
  6. **Generalised additive models**：惩罚三次样条检测非线性
  7. **SHAP feature importance**：梯度提升树 + target-encoded domain（k=5 shrinkage）
  8. **Sensitivity analysis**：五种子集重新拟合
  9. **Leave-one-domain-out replication**：八次迭代
- **Consensus protocol**：效应需同时满足五项：(i) q<0.05, (ii) Π̂≥0.60, (iii) DML后非零, (iv) GAM单调/单峰, (v) 5/5敏感性子集符号保持。

## 实验与结果
- **数据集**：≈2.1×10⁶ citations（四个引擎）；N_p=10,042 pages（品牌控制子集N=4,015, D=297 domains）；第三方形分析N=1.27×10⁶ true third-party citations。
- **主要结果**：
  - **Prompt-content alignment**：β̂=+0.37 [0.33, 0.41], q≈10⁻⁷³，通过所有九方法验证；1 SD增加对应~30% citation count增长
  - **次强特征**：page length (β=+0.13, q<10⁻⁷)、title-prompt similarity (β=+0.09, q<10⁻⁴)、page age (β=+0.05, q<10⁻³)
  - **AEO清单效应**：FAQ (β=+0.07)、TLDR (β=+0.05)、author bio (β=+0.02) 在domain control后不显著；Core Web Vitals pooled负相关但domain-residualized归零
  - **Simpson悖论**：speed metrics pooled rank-biserial负关联（快页被引更少），但domain控制后反转——大 incumbent品牌有慢页但高引用
  - **Domain authority主导**：mean absolute SHAP=0.381 vs. 最强非alignment页级特征0.060（6倍差距）；domain-only模型R²=0.320，全特征R²=0.450（ΔR²=0.130）
  - **Cross-engine异质性**：median citation age从Claude 5.1月到ChatGPT 8.0月；brand share-of-voice ChatGPT 39.3% vs. Gemini 13.8%；LinkedIn 96.6%第三方引用来自Google AI Overviews，Gemini为零
  - **Page format**：pricing页面残差效应最大(β=+0.39, q<10⁻³)；on-page信号在funnel deeper时更可预测
  - **Temporal hold-out**：alignment系数从训练0.605衰减至hold 0.342，CI仍严格>0；cross-period r²=0.018

## 相关工作脉络
1. **Lewis et al. (2020) RAG**：LLM引用系统基础，但本研究逆向提问——不问"引用是否支持声明"而是问"哪些页级特征预测引用频率"。
2. **Aggarwal et al. (2023) GEO**：首次系统测试GEO干预，但仅单引擎小规模；本文扩展到四引擎大规模+confound control。
3. **Kumar & Palkhouski (2025)**：logistic regression on 1,100 URLs，发现metadata freshness最强，但无domain fixed effects；本文揭示其估计可能受Simpson悖论污染。
4. **Zhang et al. (2025, 2026)**：classification框架区分citation selection vs. absorption，发现长/模块化/语义对齐页面吸收率高；本文进一步量化alignment的独立效应并检验causal direction。
5. **Yang (2025)**：document平台specific concentration across 366K citations；本文扩展到engine-specific heterogeneity（如LinkedIn仅Google AI引用）。
6. **Kumar & Lakkaraju (2024)**：lexical insertion manipulates LLM recommendation；本文聚焦自然观测而非manipulation，更贴近production场景。
7. **Ali et al. (2019); Chaney et al. (2018)**：observational IR分析先例；本文将formal multivariate methods扩展到黑盒LLM引用。

## 局限性与未来方向
- **Observational design**：clients未randomise干预，系数为within-domain partial association而非treatment effect；DML控制measured confounders但无法排除unmeasured（content quality, editorial investment）
- **Scope限制**：19个workspace均为B2B SaaS单一benchmarking平台；结论不直接transfer至news/e-commerce/B2C；仅四引擎（缺Perplexity、Bing Chat）；六个月窗口可能错过engine pipeline季度变化
- **Alignment metric**：lexical Jaccard仅捕获unigram/bigram语义对齐；semantic cosine against citing prompts应为upper bound
- **Feature gaps**：schema extraction在brand-controlled pages因CMS JSON-LD渲染失败；PageSpeed覆盖率95.6%缺失部分imputed at median
- **未来方向**：controlled prompt manipulation做因果识别；longitudinal rolling windows检验跨季度稳定性；扩展到非B2B-SaaS域；用contemporary embedding models构建 richer semantic alignment metrics

## 研究启发与可借鉴点
1. **Nine-method consensus framework可迁移**：将mixed-effects + DML + stability-selection + GAM + temporal hold-out结合，用于任何observational LLM behavior研究（如hallucination rate、toxicity correlation），可替代单一方法的脆弱性。
2. **Domain fixed effects作为baseline**：任何跨publisher/跨品牌的LLM引用分析必须默认加入domain-level固定效应，否则pooled估计系统性混淆brand-correlated covariates；这对团队后续做multi-domain对比实验有直接指导价值。
3. **Temporal hold-out复制设计**：保持URL集合固定、按日期partition citations训练/验证，隔离coefficient stability与exposure-time confounding——可复用至评估LLM behavior随时间漂移的研究。
4. **Jaccard alignment computation against non-citing prompts**：避免circularity的关键技巧——将alignment metric定义在full workspace prompt corpus（含未产生引用的提示）而非仅citing prompts，可直接借鉴到content positioning工作中。
5. **Page-position depth analysis**：median best-paragraph depth=0.36（top-third preferred）为content structuring提供实证依据；团队可进一步探索"answer-in-first-paragraph"对absorption的影响。

## 关键术语表
**Prompt-content alignment**：页面tokens与workspace prompt corpus（含非引用提示）的lexical Jaccard重叠及semantic cosine相似度，是本文发现的最强页级预测因子。
**Double machine learning (DML)**：通过交叉拟合残差化正交化部分效应，用于在high-dimensional confounders下估计target predictor的causal参数。
**Simpson's paradox（in this context）**：pooled data中AEO signals显示正向效应，但conditioning on domain fixed effects后效应反转或归零，因大 incumbent品牌同时拥有高引用和特定页级特征分布。
**Stability-selection Lasso**：200次bootstrap重采样下L1正则化回归的非零系数出现频率，阈值Π*=1.0表示确定性选择。
**R² decomposition**：domain-only模型解释32%方差，full feature set解释45%，页级增量ΔR²=0.13为page-level optimisation的理论上限。
**Brand-controlled URL share**：引用中属于自家或竞争对手品牌的URL比例，跨引擎从Gemini 13.8%到ChatGPT 39.3%差异显著。
**Funnel-stage heterogeneity**：on-page signal效应在transactional（bottom-funnel）页面显著大于informational（top-funnel）页面。
**Third-party concentration**：top-10 domains贡献<20%引用，~1,000 domains覆盖80%，无单一source占dominant share。

## 可复现要素
- **数据集**：citation corpus（≈2.1×10⁶）和page corpus（N_p=10,042）未公开（客户匿名保护）；third-party domain分布可复现（见Figure 9）
- **代码/权重**：analytic pipeline已开源发布（论文声明"released with the analytic pipeline"），具体仓库未列出需查证
- **关键超参**：
  - Stability-selection: B=200, λ由10-fold CV, threshold Π*=1.0
  - DML: 5-fold交叉拟合, gradient-boosted nuisance models
  - GAM: penalised cubic splines
  - Mixed-effects: HC1 robust SE, domain fixed effects (D=297)
  - Target variable: y=log₂(1+n_c), all predictors z-scored
