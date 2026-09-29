# Textual User Taste: Natural-Language User Context for Foundation-Model Recommender System at Scale

GHAZAL FAZELNIA, PAUL GIGIOLI, ELIZA KLYCE, SHARON ZHENG, KATIE ZELVIN, YE MYAT THEIN, ANURAG DESHPANDE, SEDA DAVTYAN, KATE REMEIKA, MAYA HRISTAKEVA, ERIK FRANCO, KAREN BANZON, PENG GE, JACQUELINE WOOD, NANDINI SINGH, DAVID MURGATROYD, MOUNIA LALMAS, YVES RAIMOND, and ANDREAS DAMIANOU, Spotify, USA, UK, Sweden

Foundation model recommender systems require user context that can be consumed by large language models, reasoned over, and refined through natural-language interaction. Traditional behavioral embedding vectors remain highly efective for retrieval and ranking, but they are opaque to users and not natively expressed for language model workflows. We present Textual User Taste, a system that generates structured natural-language taste profiles from listening behavior, interaction signals, content metadata, and optional user feedback, and deploys them to millions of Spotify users. We describe the end-to-end production lifecycle required to generate, evaluate, optimize, and maintain these representations at industrial scale, including prompt development and compression, user steering, and integration with downstream personalization systems.

Because no unique ground-truth taste profile exists, we introduce a multi-faceted evaluation framework to evaluate taste profiles as a production representation: they carry user-specific predictive signal independently, and when integrated with behavioral embeddings, improve MRR by 0.6% for future-track prediction and NDCG@7 by 2.2% for search ranking. Our evaluation also reveals that taste profiles support positive natural-language steering, while exposing important limitations, including challenges with negation and short-term temporal adaptation. These findings position taste profiles not as replacements for behavioral embeddings, but as an interpretable and steerable interface between evolving user context and foundation-model recommender systems.

CCS Concepts: • Information systems → Recommender systems; • Computing methodologies → Natural language processing.

## 1 Introduction

User modeling in recommender systems has traditionally relied on latent representations learned from interaction data, from matrix factorization [10] to neural collaborative filtering [6] and large-scale sequential models [12, 19]. These representations are compact, eficient, and highly predictive. However, they are also opaque: users cannot inspect a behavioral embedding, correct an incorrect assumption about their preferences, or express a new contextual preference.

Foundation models change the role of user representations in recommender systems. Conversational recommenders, natural-language search, and agentic content discovery consume textual context, reason about intent, and generate language as part ofthe recommendation process. Passing raw interaction history to every downstream model is expensive and often exceeds practical context budgets. A textual taste profile provides a diferent abstraction: a cached, semantically organized representation of a user’s longer-term preferences and consumption patterns that language models can readily consume and that users can inspect and refine. More broadly, it extends context-aware recommendation [1] by expressing behavioral, temporal, and explicit user context as reusable natural-language representations.

Recent research has demonstrated the value of language-based user profiles for recommendation, transparency, and control [14, 16, 21]. More recent work has optimized profile generation using downstream or self-supervised behavioral objectives [2, 4, 8], while industrial systems have demonstrated real-time persona generation at very large scale [18]. These advances establish textual user profiles as a viable user representation for recommender systems. However, they also raise a broader lifecycle question: how should such representations be generated, evaluated, refreshed, and governed once they become shared infrastructure across multiple products?

Our central argument is that taste profiles should be evaluated as operational user representations rather than simply as pieces of text. High-quality language is necessary but not suficient: a useful profile must remain grounded in observed behavior, distinguish one user from another, influence downstream systems in intended ways, and respond predictably to explicit user feedback. We therefore treat generation and evaluation as a closed operational loop in which evaluation continuously informs how taste profiles are generated, optimized, deployed, and monitored.

This paper studies that lifecycle through Textual User Taste, Spotify’s production system for generating and serving textual taste profiles, surfaced in a user-facing experience and consumed by both foundation-model and conventional personalization systems. The taste profiles summarize listening patterns across content types from behavioral signals and optional user-authored feedback. Operating this representation at scale exposes challenges that extend beyond taste profile generation: there is no single reference taste profile; fluent text may be insuficiently grounded in observed behavior; prompt changes propagate across multiple downstream consumers; the same inputs may produce diferent profiles across regenerations; and taste profiles may expose unsupported or sensitive inferences about users.

Our contributions are:

• We present an end-to-end production framework for generating, evaluating, maintaining, and serving structured textual user representations at scale.

• We introduce a comprehensive evaluation framework for taste profiles that combines intrinsic and functional evaluation to address the absence of a unique ground-truth profile.

• We provide an empirical study of the strengths and limitations of taste profiles in production recommender systems, identifying where they complement behavioral embeddings and where important challenges remain.

## 2 Related Work

Textual user profiles. The Language-based Factorization Model (LFM) generates compact profiles from rating histories for downstream prediction, showing benefits in cold-start settings [21]. Natural-language profiles have also been shown to support scrutable recommendation, allowing users to inspect and edit their profiles to influence recommendations [14]. TEARS aligns textual and collaborative representations while exposing an editable profile [13]. PURE incrementally updates profiles from reviews, while PersonaX maintains multi-interest profiles for recommendation agents [3, 17]. In the music domain, recent work has examined whether listeners identify with LLM-generated taste profiles and how profile quality varies across users and items [16]. Collectively, these studies establish the value of textual user profiles, but primarily study profile generation or use within a single domain, task, or experimental setting rather than persistent context shared across production systems. Our work addresses this gap by studying taste profiles as production infrastructure, where the absence of a unique reference profile makes evaluation, maintenance, and deployment central challenges.

Industrial scale and complementary representations. Recent work has demonstrated scalable generation of real-time user personas for industrial recommendation systems using distillation, asynchronous inference, and clustered input representations [18]. While this establishes the feasibility of large-scale persona generation, it does not consider a persistent, user-visible, multi-content representation that can be explicitly refined through natural-language feedback and reused across multiple downstream systems. A related line of work injects collaborative embeddings into language models or aligns semantic and collaborative representations [11, 15, 20]. These approaches propose new mechanisms for combining textual and behavioral representations, but largely leave open how such representations should be evaluated, maintained, and operated as production context. Rather than proposing a new fusion architecture, we treat taste profiles and behavioral embeddings as complementary production representations and study where taste profiles provide value beyond behavioral embeddings alone.

## 3 Textual User Taste System

This section describes the Textual User Taste system and the production lifecycle of its taste profiles. We present the key components of the system, from aggregating behavioral signals and generating structured profiles to serving them across downstream consumers and continuously refining them through evaluation and user feedback.

## 3.1 Inputs and Profile Structure

For each user, the system generates a structured taste profile from aggregated behavioral signals rather than raw interaction histories. Inputs span multiple temporal horizons and include artists, tracks, genres, podcasts, audiobooks, topical interests, follows, negative feedback (e.g., hides), discovery patterns (e.g., exploration versus repeat listening), and time-of-week consumption. Content metadata, including titles, descriptions, categories, and creator names, provide additional semantic grounding. Using aggregated signals reduces prompt size, lowers generation cost, and eliminates the need to pass complete interaction histories to the generator, while also capturing rich semantic information across multiple levels of granularity. An optional user-authored note (e.g., “more jazz”) provides explicit feedback; notes are filtered for relevance and safety and kept separate from observed behavior so the generator can distinguish stated from inferred preferences.

The generator returns a structured taste profile with four fields:

• Music Taste: enduring musical preferences, afinities, and representative content;

• Podcast Taste: preferred topics, formats, and engaged shows;

• Audiobook Taste: preferred genres, authors, and engagement patterns;

• Listening Rhythms: recurring temporal and cross-content consumption patterns.

This schema separates enduring content preferences from consumption context, allowing downstream systems to select only the fields relevant to a given task while avoiding unsupported inferences when behavioral evidence is sparse.

## 3.2 Generation, Guardrails, and Serving

The structured input is supplied to a cost-eficient instruction-tuned LLM that generates the taste profile. The prompt explains the meaning of each input feature, specifies the output schema and style, prioritizes recent evidence while retaining durable preferences, and instructs the model to avoid unsupported entities and sensitive inferences. Guardrails are applied both before and after generation. Input safeguards filter irrelevant, unsafe, or adversarial user notes, while output safeguards monitor baseline safety together with profile-specific risks, including protected-attribute inference, personally identifiable information, emotional overreach, and unsupported claims about creators.

The model and prompt are versioned together, allowing candidate versions to be evaluated before deployment and monitored after rollout. Taste profiles are stored and served to both user-facing and backend consumers rather than regenerated for every request, reducing inference cost while providing downstream systems with a stable representation. When a user submits an accepted note, it is incorporated during a subsequent regeneration, making profile updates part of a controlled feedback-and-refresh loop rather than unrestricted editing of stored text.

![](images/d599c5aae82d9a51a6dc4bc61c662c7651173f2d0651bbcb160878a98f94c2a8.jpg)  
Fig. 1. Textual User Taste lifecycle. Aggregated behavior, metadata, and optional user feedback are converted into a structured prompt. The generated fields are evaluated and served to user-facing and downstream personalization systems; feedback and evaluation results inform later generations.

Table 1. Instruction compression trade-ofs. Factuality is scored from 1 to 10.
<table><tr><td>Condition</td><td>Tokens</td><td>Factuality</td><td>Valid output</td></tr><tr><td>Uncompressed</td><td>~6,800</td><td>8.66</td><td>100%</td></tr><tr><td>LLMLingua 2×</td><td>3,428</td><td>8.61</td><td>100%</td></tr><tr><td>LLMLingua 4×</td><td>1,803</td><td>7.80</td><td>73.1%</td></tr><tr><td>LLMLingua 8×</td><td>1,202</td><td>3.31</td><td>16.5%</td></tr></table>

Figure 1 summarizes the complete lifecycle. Evaluation is an integral part of the system rather than a final validation step: signals from LLM judges, predictive tasks, robustness tests, and downstream evaluations continuously inform prompt refinement, model selection, and future profile generations.

## 3.3 Prompt Development and Optimization

Prompt development was treated as an iterative optimization process balancing profile quality and production cost. We iteratively refined seven prompt versions difering in instruction specificity, examples, tone, and safety guidance, assessing each with LLM judges on factuality, coherence, and specificity. Across iterations, factuality improved from 7.70 to 8.57 and coherence from 6.46 to 7.98 on a ten-point scale. The longest prompt achieved the highest factuality and coherence, whereas a shorter variant produced the best specificity, showing that prompt length alone is a poor proxy for quality.

We explored two complementary optimization approaches. First, LLMLingua [7] compressed prompts by removing lower-information instruction tokens while preserving the schema and safety instructions. Table 1 shows a clea trade-of: two-fold compression preserved factuality and structured validity, whereas more aggressive compression substantially degraded both. Second, we formulated prompt construction as a program optimization problem using DSPy MIPROv2 and BootstrapFewShot [9]. The resulting prompt achieved an 80.2% composite objective (average of the evaluation metrics) using approximately 2,000 tokens (i.e., one-third of the length of the best hand-authored prompt)

These approaches generate candidate prompts. Whether a prompt is deployed depends on the broader evaluation framework described in the next section, ensuring that optimization is guided by downstream utility rather than intrinsic quality alone.

## 4 Evaluation Methodology

Because taste profiles have no unique ground-truth representation, we evaluate them through complementary intrinsic and functional assessments.

## 4.1 Intrinsic Quality and Robustness

Because taste profiles have no unique ground-truth representation, intrinsic evaluation must assess multiple aspects of quality rather than rely on a single score. We therefore use seven LLM judges to evaluate factuality, coherence, coverage, specificity, diversity, relevance, and safety. These dimensions capture whether a profile is grounded, informative, well formed, and appropriate for downstream use. Each judge follows a task-specific rubric and receives only the evidence required for its assessment, allowing unsupported inferences to be distinguished from poor writing.

We use this evaluation suite to compare prompt variants, generator models, and prompt compression strategies, while factuality and safety assessments also support production monitoring. Intrinsic quality alone, however, cannot establish whether a profile benefits downstream systems. We therefore complement these measures with the functional evaluations described in the following subsection. LLM-judge scores are used as diagnostic signals rather than definitive measures of profile quality because they may inherit model biases and exhibit inconsistent calibration across languages.

Because users can explicitly steer their taste profiles, we also evaluate how the system handles challenging feedback scenarios. We therefore constructed approximately 50 stress scenarios spanning nine categories, including contradictory behavior and explicit feedback, prompt injection, personally identifiable information, unsafe content, multilingual inputs, shared-account behavior, empty profiles, and requests that cannot be faithfully represented. These scenarios evaluate both whether generated profiles remain safe and whether conflicts between observed behavior and explicit user feedback are handled transparently.

## 4.2 Behavioral and Downstream Tests

We complement intrinsic evaluation with four functional evaluations, each designed to assess a diferent aspect of taste profiles: predictive value, user specificity, downstream utility, and steerability.

Future-track prediction. We evaluate whether taste profiles provide complementary predictive signal beyond behavioral embeddings. We compare four user representations in the same downstream ranking task, training the same ranking model for each one: a production behavioral embedding [5], a taste-profile embedding, their concatenation, and a gated fusion of the two. Taste-profile embeddings are set to 120 dimensions using a Matryoshka text encoder to match the dimensionality of the behavioral embeddings. The evaluation uses 400,000 user–track examples, split 80/20 by user. Results are averaged over five runs and reported using MRR, ROC-AUC, and NDCG@10.

User specificity. We evaluate whether taste profiles capture user-specific information that generalizes beyond observed listening behavior. For each of the users in the sample set, an LLM ranks five candidate tracks consisting of one track the user consumed in a held-out future period and four distractors. We compare three conditions: no profile, the user’s own profile, and a randomly assigned profile serving as a control.

Search ranking. We evaluate whether taste profiles improve downstream search personalization. A language model is fine-tuned for search-result ranking under four user-context configurations: no user representation, taste profile only, behavioral user embedding only, and both representations together. The evaluation contains 10,000 query sessions and approximately 5.6 million query–candidate pairs. Performance is measured using SessionNDCG@7 on two temporal train–test partitions.

Steering. We evaluate whether natural-language feedback steers recommendations in the requested direction. For a sample of users and candidate tracks, we augment each taste profile with requests such as “more jazz”, recompute the profile embedding, and measure whether resulting recommendations shift toward the requested genre or mood. We quantify this shift using semantic similarity and Top-20 precision with respect to the requested genre or mood. We additionally evaluate negative requests $( \mathrm { e . g . }$ , “less electronic music”), situational requests $( \mathbf { e . g . }$ , “music for deep focus”), and multi-turn refinements in which users iteratively refine their preferences. This is a controlled representation study that evaluates whether textual feedback produces the intended changes in an embedding-based recommender rather than measuring user satisfaction

## 4.3 Representation Complementarity and Temporal Stability

Representation complementarity. We evaluate the extent to which taste profiles relate to behavioral embeddings. For a sample of users, we compare pairwise similarity rankings, nearest-neighbor relationships, and cluster assignments between the two representation spaces. Structural agreement is quantified using Representational Similarity Analysis (RSA) with Spearman correlation, nearest-neighbor overlap at $( k = 5 )$ , and Adjusted Rand Index (ARI). These metrics characterize the structural relationship between the two representation spaces rather than quantifying the amount of information contained in either representation.

Temporal stability. Temporal stability is measured by regenerating taste profiles after three and seven days under largely unchanged behavioral inputs. For a sample of users, profiles are regenerated using data from a reference day and again after three and seven days. We compare changes in the inputs and seven output measures, including embedding cosine similarity and surface-form similarity. Because the profiles summarize long-term listening behavior, this experiment primarily measures generation stability under nearly unchanged user behavior rather than responsiveness to genuine preference change

## 5 Results

This section evaluates taste profiles from four complementary perspectives, corresponding to the evaluation framework introduced in Section 4. Intrinsic evaluation confirms that profile quality is inherently multidimensional. Principal component analysis of the seven intrinsic quality scores identifies two dominant components explaining 93.7% of the variance: one associated with semantic richness and another with factual quality. Moreover, factuality and relevance are only weakly negatively correlated (� = −0.18), indicating that no single intrinsic metric adequately characterizes profile quality. This finding supports our use of multiple complementary quality dimensions throughout the evaluation.

## 5.1 Text Adds Complementary Predictive Signal

Table 2 shows that behavioral embeddings remain the strongest standalone representation for future-track prediction. Taste profiles alone achieve competitive performance, but consistently underperform behavioral embeddings across al three metrics. Combining the two representations yields the best overall performance: concatenation improves MRR from 0.956 to 0.962, ROC-AUC from 0.957 to 0.963, and NDCG@10 from 0.932 to 0.941. The ROC-AUC improvement over the behavioral baseline is statistically significant, whereas gated fusion performs similarly but ofers no further improvement. These results indicate that taste profiles provide complementary predictive signal rather than replacing behavioral embeddings. The benefit is not universal, however: approximately 30% of users achieve better predictions than with the behavioral embedding alone

Table 2. Future-track ranking (five-run mean).
<table><tr><td>Representation</td><td>MRR</td><td>AUC</td><td>NDCG@10</td></tr><tr><td>Behavioral</td><td>0.956</td><td>0.957</td><td>0.932</td></tr><tr><td>Taste profile</td><td>0.937</td><td>0.946</td><td>0.912</td></tr><tr><td>Taste profile + behavioral</td><td>0.962</td><td>0.963</td><td>0.941</td></tr><tr><td>Gated fusion</td><td>0.961</td><td>0.962</td><td>0.940</td></tr></table>

Table 3. Search ranking; uplift is relative to no context.
<table><tr><td>Context</td><td>NDCG @7 Uplift</td></tr><tr><td>None</td><td>baseline</td></tr><tr><td>Taste profile</td><td>+0.9%</td></tr><tr><td>Behavioral</td><td>+1.7%</td></tr><tr><td>Combined</td><td>+2.2%</td></tr></table>

The user-specificity evaluation tests whether the gains observed above arise from information specific to an individual user rather than simply from providing additional prompt context. Using the user’s own taste profile increases MRR from 0.656 to 0.776 (+18.3%; � = 0.0003, � = 0.26), whereas replacing it with another user’s taste profile reduces MRR to 0.477. This demonstrates that the LLM benefits from the user-specific information encoded in the profile rather than from the presence of additional text alone. While this does not imply that every statement in the generated profile is factually correct, it does show that the profile captures user-specific information that is predictive of future listening.

## 5.2 Downstream Context Works Best in Combination

Table 3 evaluates whether taste profiles improve a real downstream personalization task. A taste profile alone improves SessionNDCG@7 by 0.9% over the no-user-context baseline, whereas the behavioral embedding improves performance by 1.7%. Combining the two representations yields the largest improvement (2.2%). Across a second temporal partition, the standalone profile and behavioral representations exhibit inconsistent gains, whereas their combination remains consistently positive. These results reinforce the central finding of the previous subsection: taste profiles complement rather than replace behavioral embeddings. Their greatest value lies in providing complementary context for downstream personalization rather than serving as standalone user representations.

## 5.3 Text Is Steerable, but Not by Simple Negation

Positive natural-language requests consistently steer recommendations in the intended direction. Across all eight evaluated genres and moods, requests such as “more jazz” move the top-ranked recommendations toward the requested genre or mood $( \boldsymbol { p } < 0 . 0 0 0 1$ for every direction), increasing mean Top-20 precision from 56% to 74%. We compared appending user feedback to the existing taste profile against regenerating the taste profile with an LLM. Appending the request produced recommendations that were semantically closer to the requested genre or mood (0.031 versus 0.022), suggesting that preserving the existing profile is more efective than regenerating it.

This limitation is equally informative. Requests such as “no electronic music” move recommendations toward, rather than away from, the mentioned genre. This occurs because the embedding primarily captures the semantic concept “electronic music” rather than the logical meaning introduced by the negation. Consequently, simply re-embedding a negated request increases its similarity to the very concept the user intends to avoid. Handling negative preferences therefore requires explicit reasoning or filtering mechanisms rather than embedding manipulation alone.

More complex forms ofsteering are also efective. Situational requests such as “music for falling asleep” or “deep focus” produce larger shifts than genre- or mood-based requests (mean shift 0.069), and multi-turn interactions progressively refine recommendations toward increasingly specific user preferences. These findings reinforce the role of taste profiles as an interactive interface between users and foundation-model recommender systems, although confirming their impact on user satisfaction will require evaluation in deployed user-facing systems.

## 5.4 Structural and Temporal Diagnostics

Taste profiles and behavioral embeddings exhibit only modest structural agreement, indicating that they organize user preferences diferently in representation space. Representational Similarity Analysis yields a Spearman correlation of $\rho = 0 . 2 8 6$ , with 17.5% nearest-neighbor overlap at $k = 5$ and an Adjusted Rand Index of 0.08. These metrics characterize the structural relationship between the two representation spaces rather than quantifying the information captured by either representation. The stronger evidence for complementarity comes from the downstream improvements observed when both representations are used together.

Taste profiles also remain highly stable over short time horizons. Full-profile embedding cosine similarity is 0.954 after three days and 0.955 after seven days, with no significant diferences across seven output measures (all Wilcoxon $p > 0 . 0 5 )$ . Approximately 95.0% and 94.4% of the corresponding inputs remain unchanged over the two intervals, indicating that the observed stability reflects consistent profile generation under largely unchanged inputs. Evaluating responsiveness to genuine preference change requires a diferent experimental setting and remains future work.

## 6 Discussion and Conclusion

Our results suggest that taste profiles should be treated as contextual interfaces rather than standalone substitutes for behavioral representations. They must be evaluated through their consumers: because intrinsic dimensions are not interchangeable and profile versions produce diferent downstream trade-ofs, no single judge score is suficien for system selection. Behavioral embeddings remain stronger for compact prediction, whereas text provides semantic organization, inspectability, and an interface for explicit context; accordingly, our most consistent gains come from combining the two. Although both representations originate from the same behavioral signals, text introduces a diferent inductive bias through language-model knowledge and prompt design. This complementarity may diminish as behavioral encoders incorporate richer metadata or language-model reasoning. Successful deployment requires more than text similarity: positive requests can steer the representation, but negation, conflicts between stated and observed preferences, and changing intent require explicit policies and signal provenance. Similarly, refresh strategies mus balance stability under unchanged inputs with responsiveness to meaningful events, while safety must be enforced at the representation layer because profile inferences are visible to users and reused across systems.

These findings should be interpreted in light of several limitations. Taste profile generation is an unsupervised summarization problem with no canonical ground-truth profile. Although our evaluation suite measures factuality, coverage, specificity, relevance, safety, behavioral signal, and downstream utility, the absence of gold labels makes it better suited to comparing system variants than establishing absolute profile quality. Taste profiles are also a new product and representation surface, so mature interaction patterns and long-term measures of user value are not yet available, while proprietary data and infrastructure limit exact reproduction. Deployment to a large population demonstrates operational feasibility, but does not imply that every profile is equally accurate or that every consumer benefits; each application still requires controlled evaluation and monitoring. Within these bounds, Textual User Taste demonstrates how natural-language user context can be generated, evaluated, refreshed, safeguarded, and served at scale. Rather than superseding behavioral embeddings, taste profiles provide a complementary, interpretable, and steerable interface between evolving user context and foundation-model recommender systems

## References

[1] Gediminas Adomavicius, Bamshad Mobasher, Francesco Ricci, and Alexander Tuzhilin. 2011. Context-Aware Recommender Systems. AI Magazine 32, 3 (2011), 67–80. doi:10.1609/aimag.v32i3.236

[2] Rajat Arora, Ye Tao, Jianqiang Shen, Ping Liu, Muchen Wu, Qianqi Shen, Benjamin Le, Fedor Borisyuk, Jingwei Wu, and Wenjing Zhang. 2026. High Fidelity Textual User Representation over Heterogeneous Sources via Reinforcement Learning. arXiv preprint arXiv:2602.07333 (2026). https://arxiv.org/abs/2602.07333

[3] Seunghwan Bang and Hwanjun Song. 2025. LLM-based User Profile Management for Recommender System. arXiv preprint arXiv:2502.14541 (2025) https://arxiv.org/abs/2502.14541

[4] Yue Chen, Yifei Sun, Lu Wang, Fangkai Yang, Pu Zhao, et al. 2026. DUET: Joint Exploration of User–Item Profiles in Recommendation System. In Findings ofthe Association for Computational Linguistics: ACL 2026. 21392–21406. doi:10.18653/v1/2026.findings-acl.1076

[5] Ghazal Fazelnia, Sanket Gupta, Claire Keum, Mark Koh, Timothy Christopher Heath, Guillermo Carrasco Hernández, Stephen Xie, Nandini Singh, Ian Anderson, Maya Hristakeva, Petter Pehrson Skidén, and Mounia Lalmas. 2025. Generalized User Representations for Large-Scale Recommendations and Downstream Tasks. In Proceedings ofthe 19th ACM Conference on Recommender Systems. 962–966. doi:10.1145/3705328.3748132

[6] Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu, and Tat-Seng Chua. 2017. Neural Collaborative Filtering. In Proceedings of the 26th International Conference on World Wide Web. 173–182. doi:10.1145/3038912.3052569

[7] Huiqiang Jiang, Qianhui Wu, Chin-Yew Lin, Yuqing Yang, and Lili Qiu. 2023. LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models. arXiv preprint arXiv:2310.05736 (2023). https://arxiv.org/abs/2310.05736

[8] Clark Mingxuan Ju, Yuwei Qiu, Tong Zhao, and Neil Shah. 2026. Self-supervised User Profile Generation for Personalization. arXiv preprint arXiv:2606.05336 (2026). https://arxiv.org/abs/2606.0533

[9] Omar Khattab, Arnav Singhvi, Paridhi Maheshwari, Zhiyuan Zhang, Keshav Santhanam, Sri Vardhamanan, Saiful Haq, Ashutosh Sharma, Martin Josifoski, Divyansh Gopinadhan, Matei Zaharia, and Christopher Potts. 2024. DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines. arXiv preprint arXiv:2310.03714 (2024). https://arxiv.org/abs/2310.03714

[10] Yehuda Koren, Robert Bell, and Chris Volinsky. 2009. Matrix Factorization Techniques for Recommender Systems. Computer 42, 8 (2009), 30–37. doi:10.1109/MC.2009.263

[11] Qidong Liu, Xian Wu, Yejing Wang, Zijian Zhang, Feng Tian, Yefeng Zheng, and Xiangyu Zhao. 2024. LLM-ESR: Large Language Models Enhancement for Long-tailed Sequential Recommendation. In Advances in Neural Information Processing Systems. https://arxiv.org/abs/2405.20646

[12] Nikil Pancha, Andrew Zhai, Jure Leskovec, and Charles Rosenberg. 2022. PinnerFormer: Sequence Modeling for User Representation at Pinterest. In Proceedings ofthe 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 3702–3712. doi:10.1145/3534678.3539156

[13] Emiliano Penaloza, Olivier Gouvert, Haolun Wu, and Laurent Charlin. 2025. TEARS: Textual Representations for Scrutable Recommendations. In Proceedings ofthe ACM Web Conference 2025. 4949–4968. https://arxiv.org/abs/2410.19302

[14] Jerome Ramos, Hossein A. Rahmani, Xi Wang, Xiao Fu, and Aldo Lipani. 2024. Transparent and Scrutable Recommendations Using Natural Language User Profiles. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 13971–13984. doi:10.18653/v1/2024.acl-long.753

[15] Xubin Ren, Wei Wei, Lianghao Xia, Lixin Su, Suqi Cheng, Junfeng Wang, Dawei Yin, and Chao Huang. 2024. Representation Learning with Large Language Models for Recommendation. In Proceedings ofthe ACM Web Conference 2024. doi:10.1145/3589334.3645458

[16] Bruno Sguerra, Elena V. Epure, Harin Lee, and Manuel Moussallam. 2025. Biases in LLM-Generated Musical Taste Profiles for Recommendation. In Proceedings ofthe 19th ACM Conference on Recommender Systems. doi:10.1145/3705328.3748030

[17] Yunxiao Shi, Wujiang Xu, Zhang Zeqi, Xing Zi, Qiang Wu, and Min Xu. 2025. PersonaX: A Recommendation Agent-Oriented User Modeling Framework for Long Behavior Sequence. In Findings ofthe Association for Computational Linguistics: ACL 2025. 5764–5787. doi:10.18653/v1/2025. findings-acl.300

[18] Haoting Wang, Haokai Lu, Zheyun Feng, et al. 2026. LLM-Based User Personas for Recommendations at Scale. arXiv preprint arXiv:2606.12198 (2026). https://arxiv.org/abs/2606.12198

[19] Jiaqi Zhai et al. 2024. Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations. In Proceedings of the 41st International Conference on Machine Learning. https://arxiv.org/abs/2402.17152

[20] Yang Zhang, Fuli Feng, Jizhi Zhang, Keqin Bao, Qifan Wang, and Xiangnan He. 2025. CoLLM: Integrating Collaborative Embeddings into Large Language Models for Recommendation. IEEE Transactions on Knowledge and Data Engineering (2025). https://arxiv.org/abs/2310.19488

[21] Joyce Zhou, Yijia Dai, and Thorsten Joachims. 2024. Language-Based User Profiles for Recommendation. arXiv preprint arXiv:2402.15623 (2024) https://arxiv.org/abs/2402.15623