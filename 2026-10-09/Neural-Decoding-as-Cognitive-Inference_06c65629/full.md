# Neural Decoding as Cognitive Inference

Yi Guo<sup>1,2</sup>, Changhong Jing<sup>1,3</sup>, Yong Hu<sup>1</sup>, Yan Liu<sup>3</sup>, Michael K. P. Ng<sup>4</sup>, Shanshan Wang<sup>1,2</sup>, Shuqiang Wang<sup>1,2</sup>

<sup>1</sup>Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences, Shenzhen, China.

<sup>2</sup>University of Chinese Academy of Sciences, Beijing, China.

<sup>3</sup>Department of Computing, The Hong Kong Polytechnic University, Hong Kong. <sup>4</sup>Department of Mathematics, Hong Kong Baptist University, Hong Kong.

## Abstract

The brain maintains stable cognition despite continuously changing neural activity. How to extract stable cognitive states from variable neural observations remains a central problem in neural decoding. Existing neural decoding methods map neural observations to predefined external labels based on the stimulus-response principle, often capturing recording-specific spurious correlations. Inspired by how the brain infers the world, and specifically by Bayesian brain theory, we recast neural decoding as cognitive inference constrained by brain-intrinsic priors, yielding high-level meta-neural semantic representations. In decoding experiments spanning five neural recording modalities and three cognitive domains (motor, perception and internal mentation), our cognitive inference method reorganized the geometry of neural observation representations, yielding meta-neural semantic representations that exhibited consistent geometric relationships across cognitive tasks and enabled the recovery of stable cognitive states from variable neural observations. Our work provides an account of how the brain maintains relatively stable cognition despite continual changes in the external environment. Cognitive stability is sustained through cognitive inference from changing neural activity, without requiring fixed neural activity patterns.

## Introduction

The brain maintains relatively consistent cognitive states despite continuous changes in neural activity <sup>1,2</sup>. Neural observations vary with task context, individual diferences and recording sessions, whereas cognitive states remain stable. Similarly, audition and vision depend on diferent sensory inputs and evoke distinct patterns of neural activity, yet the high-level neural representations derived from this activity show systematic semantic consistency <sup>3</sup>. A central problem in neural decoding is therefore how to recover stable cognitive states from changing neural observations <sup>4–6</sup>. Most existing neural decoding methods remain based on the stimulus-response principle. These methods map neural observations onto predefined labels and often capture recording-specific spurious correlations <sup>7,8</sup>. If the stability of cognitive states does not depend on fixed patterns of neural activity, rigid stimulus-response mappings are insuficient for decoding stable cognition.

To overcome this limitation, neural decoding needs to reconsider how the brain understands the external world <sup>9,10</sup>. According to Bayesian brain theory, the brain does not passively respond to sensory evidence but actively infers the external world from sensory evidence, with brain-intrinsic priors constraining the inference <sup>11–15</sup>. The brain’s cognitive states are therefore not directly determined by sensory evidence. Even when sensory evidence difers, the brain may infer consistent cognitive states. For neural decoding, this implies that recovering stable cognitive states from changing neural observations requires inference constrained by brain-intrinsic priors.

Building on this view, we recast neural decoding as cognitive inference. Neural observations provide evidence about cognitive states, while brainintrinsic priors constrain the inference of those states from neural observations. On this basis, we proposed a hierarchical decoding framework inspired by Bayesian brain theory. The framework infers cognitive states from changing neural observations and thereby yields high-level representations that we term meta-neural semantic representations (Fig. 1). Within this framework, brain-intrinsic priors may arise from structural constraints of the brain. Previous studies have shown that cortical geometry systematically constrains neural activity throughout the cerebral cortex and that cortical geometric eigenmodes accurately reconstruct the cortex-wide spatial distribution of this activity<sup>16–18</sup>. Accordingly, we introduced cortical geometry as a brain-intrinsic prior into cognitive inference. The cortical geometry prior is constructed independently of the cognitive states to be inferred and does not directly encode their specific content. This independence allows us to test whether consistent cognitive states can still be inferred from variable neural observations despite changes in the underlying patterns of neural activity.

We systematically evaluated the framework using five neural modalities: EEG, MEG, ECoG, fNIRS and fMRI. The experiments covered three functional domains, Motor, Perception and Internal mentation, and assessed whether cognitive inference could recover consistent cognitive states from variable neural observations. (1) Compared with neural observation representations, metaneural semantic representations were more broadly distributed throughout the cortex and exhibited more consistent geometric relations under diferent cognitive conditions. (2) Meta-neural semantic representations preserved consistent cognitive state relations among diferent tasks, and decoding performance remained stable despite neural drift over days to months. (3) Neural observations difered markedly between auditory and visual perception, whereas the inferred meta-neural semantic representations remained aligned with the semantic structure of cognitive content. Cognitive state consistency was preserved despite diferences in sensory input. (4) Cognitive inference in the pro posed framework recovered internally generated cognitive content from neural observations. The framework extended neural decoding from externally evoked content to internal mentation. Overall, cognitive inference in the proposed framework recovered stable cognitive states from variable neural observations. These findings support the view that cognitive stability does not depend on fixed patterns of neural activity but may instead be maintained through continual cognitive inference from variable neural activity.

a  
![](images/3902bb56ef33c24efc1974eb643f25bed6def3098ef6c094b295197518fda025.jpg)

b  
![](images/8da2a390911a8294265a5949c623f747aa37c6bc9b2868587472022ab5f93f99.jpg)

c  
![](images/8960b0ec23ba462a69da30463e9e2b03512b59438f15d754eabc20e2d25efe9c.jpg)

d Brain-wide semantic representation  
![](images/49c332b8dd72c0cc3d5289e166fab55df970dd80aec080c2348b852d93bea79b.jpg)

e Reorganization of representational geometry  
![](images/70d24dc018bca9f029717779db7fd3f3c9a37e6977fbc729494d6d128cc4916a.jpg)

f  
![](images/be7b95c12ed5f4d31c35edba9505f75ce3a3db4922838074e29e5d415f655a60.jpg)

g <sub>Generative</sub> <sub>decoding</sub> <sub>of</sub> <sub>language</sub> <sub>and</sub> <sub>vision</sub>  
![](images/03587a193759eec1ad4b3148f735625bc3952466d02a53b5a1021ea99f6853a4.jpg)  
Fig. 1 Overview of the Bayesian brain hierarchical decoding framework. a, Bayesian brain account linking speech and scene information to latent causes and neural activity measured with EEG, MEG, fMRI, fNIRS and ECoG. b, Hierarchical decoder comprising low-level neural observation o, mid-level prior-constrained u and high-level meta-neural semantic z representations, with z supplied to a generative AI model to produce decoded output y. c, Task domains: perception (vision and audition), motor (motor imagery) and internal mentation (memory and judgment). d, Cortical semantic-attribution maps showing localized low-level and brain-wide highlevel representations. e, Representational geometry of low-level and high-level representations across the three task domains. Open and filled circles distinguish cognitive states; blue and orange distinguish cognitive conditions. Arrows connect states within each condition; greater parallelism across conditions indicates greater cognitive stability. f, Prediction–target correlations for self-generated thought and subjective interpretation, comparing NeuroSTORM with the proposed framework. Boxes show medians and interquartile ranges (IQRs); whiskers extend to the most extreme values within 1.5 × IQR of the box limits, with outliers shown individually. \*\*\*\*P < 0.0001. g, Representative open-ended language and visual decoding outputs. Green words mark shared key semantic con tent in reference and decoded text.

## Results Brain-intrinsic priors constrain cognitive inference to yield brain-wide meta-neural semantic representations

The Bayesian brain hierarchical decoding framework used cortical geometric eigenmodes to instantiate brain-intrinsic structural priors and inferred meta-neural semantic representations constrained by these priors (Fig. 2a).

Cognitive inference-based encoding used the cognitive state representation constrained by cortical geometric eigenmodes to predict cortical activity, whereas stimulus-response-based encoding mapped stimulus semantics to evoked cortical responses. Cognitive inference-based encoding produced significant predictions over a broader cortical range and achieved higher noise-ceiling-normalized squared prediction correlation than stimulusresponse-based encoding (Fig. 2b–c). These results demonstrate the advantage of the framework in encoding broadly distributed cortical responses constrained by brain-intrinsic priors. This validation was conducted on the Natural Scenes Dataset (NSD; n = 8 participants).

In NSD, cortical searchlight RSA showed broader cortical correspondence for high-level meta-neural semantic representations than for low-level image features (51.59% versus 28.28%;

Fig. 2d). In SMN4Lang, cortical semantic attribution was more broadly distributed for the highlevel meta-neural semantic representation than for the low-level neural observation representation (Fig. 2e; n = 12 participants). From a neural-field perspective, attribution of the meta-neural semantic representation concentrated in long-wavelength cortical geometric eigenmodes, peaked at approximately 84 mm and decreased with eigenmode order (Spearman ρ = −0.820; Fig. 2f). This scale dependence related the broader semantic range of the representation to large-scale cortical fields.

The whole-cortex semantic–connectivity coefficient was higher for the proposed framework than for CogReader (0.33 versus 0.05), with the same pattern observed within each of the seven Yeo networks (Fig. 2g). The corresponding chord diagrams showed denser relations between network pairs and more long-range connections, as reflected by greater cortical geodesic distances (Fig. 2h). These results indicate that the metaneural semantic representation was more closely aligned with distributed brain-network structure than the CogReader representation.

Cognitive inference constrained by cortical geometry yielded brain-wide meta-neural semantic representations with large-scale spatial relationships. Cognitive state information was thus organized under cortical structural constraints extending beyond local neural activity. These findings support brain-intrinsic structure as a relatively stable constraint on cognitive inference from limited neural observations.

a  
![](images/1533251b11afc0fcbb36857cf4ccd5769bb8fed7e03b01d833810a6eac0a2878.jpg)

![](images/0562ec26713727117de2da1d8dd9a7ca9dadc9a0b6672562683ee80d2eca9ebd.jpg)  
b  
Cognitive-inference based encoding

![](images/d55fcf70efbb696487f856f3add047ea64199b4b9bd2c81fda25813a6a62d9de.jpg)  
c Encoding performance  
d Cortical map of representational geometry

![](images/036c018ab3691ea0507930100ea8d389da92bfc11022134aef8ac7ef29ce864c.jpg)

![](images/230c372a9a413aa2acbf46707ee1897eef34c817900bd8e7970ac2394cf164d7.jpg)

![](images/5b8dd90348bcc4eea5853cad46213a233f188b50340d935620b405b7c15f9683.jpg)

![](images/9b0151c2852f0aebe64f26ca388b3aadf68fe687b91742b4a6c81e418ab5e002.jpg)

![](images/94e72506e0c482e099818c106155723ae9a06dec36d3c3a52dea4de8fbde1ead.jpg)

![](images/041418a0e45d7a4e41fe1f2bcfd5405b832cbab21450be56a8d05ecfbb65aebd.jpg)

![](images/54038c9c59d0d2005499faa7c987c11432b7716866f2f1e19603fcc72903876f.jpg)

![](images/99fb46f43ccb6ad4ceba12d65681c438dd782e1f7e6611802845f11682d3c434.jpg)

h  
![](images/573d1ac06d6413274ddce7ea8cbcfea504b93df1398f871e35625dc7cc939971.jpg)

![](images/a4c02ba0c727cb0bb5a27ba60c0b3299c9ea9c1d1a0927031ce4c90a06a479fb.jpg)  
Fig. 2 Brain-intrinsic priors shape brain-wide meta-neural semantic representations. a, Cortical geometric eigenmodes instantiate brain-intrinsic priors in the hierarchical decoding framework and constrain inference from neural observations to meta-neural semantic representations. b, Vertex-wise correlations between predicted and measured cortical activity for cognitive inference-based encoding and stimulus-response encoding in NSD (n = 8); evaluation stimuli were held out during model fitting. c, Noise-ceiling-normalized squared prediction correlation for the two encoding models. Boxes show medians and quartiles; whiskers extend to the most extreme values within 1.5 × IQR of the box limits, with outliers shown individually. d, Cortical searchlight RSA maps in NSD show broader cortical correspondence for high-level meta-neural semantic representations than for low-level image features. Color denotes mean Spearman correlation. e–h, Results from SMN4Lang $( n = 1 2 )$ . e, Semanticattribution maps for three representative participants (Participants A–C). Maps for all 12 participants are shown in Extended Data Fig. 1. f, Semantic attribution across cortical geometric eigenmodes; the line and shading show the participant mean and s.e.m. The association with eigenmode order was assessed using Spearman’s rank correlation. $\mathbf { g } ,$ Semantic–connectivity coeficients across the whole cortex and seven Yeo networks for CogReader and the proposed framework; values show the participant mean ± s.e.m. h, Thresholded interregional relations for the two methods, grouped by Yeo network. Edge color denotes cortical geodesic length. A two-sided paired t-test was used in c. Asterisks denote statistical significance $( ^ { * } P < 0 . 0 5 ,$ , \*\*P < 0.01, \*\*\*P < 0.001 and $^ { * * * * } P < 0 . 0 0 0 1 )$ .

## Cognitive inference enables generalizable decoding in diverse tasks

We evaluated the generalizability of discriminative decoding based on meta-neural semantic representations across tasks spanning motor, perception and internal mentation. The evaluation considered representational geometry, within-participant decoding accuracy and zero-shot decoding in unseen participants.

Three-dimensional MDS projections showed that state-diference directions were more consistent across cognitive conditions at the high level across motor, perception and internal mentation (Fig. 3a). Quantitative analyses further showed greater cross-condition generalization performance and stronger parallelism of factor directions at the high level across the three domains (Fig. 3b). Complementary analyses showed lower within-state dispersion, greater between-state centroid distance and higher category-structure RSA at the high level (Extended Data Fig. 3). These results reveal a consistent geometric reorganization of the meta-neural semantic representation, in which task-relevant structure was preserved more consistently across cognitive conditions.

Within-participant decoding in BCIC, FACED, SEED-V and Motor Imagery covered motorimagery, afective-state and movement-state decoding <sup>19–22</sup>. Among the four evaluated methods, the proposed framework achieved the highest participant-averaged accuracy in all four datasets compared with LaBraM, BrainOmni and CBraMod (Fig. 3c) <sup>23–25</sup>. The box plots showed consistently higher median accuracies for the proposed framework, with participant-level distributions shifted above those of the three comparison methods. The corresponding confusion matrices were dominated by diagonal entries in each dataset (Fig. 3d). Thus, the geometric reorganization of the metaneural semantic representation was accompanied by higher decoding accuracy across multiple tasks.

The proposed framework also achieved the highest mean zero-shot decoding accuracy in each dataset compared with LaBraM, BrainOmni and CBraMod (Fig. 3e). All four methods were fitted using only the training participants and applied to unseen participants without participant-specific adaptation. This result extends the performance of the meta-neural semantic representation from within-participant decoding to zero-shot decoding in unseen participants.

Across motor, perception and internal mentation, cognitive inference yielded more consistent relationships among cognitive states despite diferences in tasks and neural observations. This geometric consistency was accompanied by improved within-participant and zero-shot decoding. The results suggest that distinct neural activity patterns can support common relationships among cognitive states without requiring identical neural representations across tasks.

## Cognitive inference enables stable decoding across time despite neural drift

Meta-neural semantic representations preserved cognitive state relationships and supported crosssession decoding despite neural drift. Repeated recordings spanned 208 days in ECoG Speech and 4 days in AJILE12, both invasive ECoG datasets, and 8 days in non-invasive SHU-MI EEG. These datasets covered speech-command production, natural upper-limb movement and motor imagery.

a  
![](images/8dbec99fadfdf08ebc02d0eae960c1edf432126cda52538760a3e74845b4df64.jpg)

![](images/03abfe4eca46aaefee1374245cdd67c695abe0b384cc0eed3abb54b0967c74e2.jpg)

![](images/d2b39b4e43738d47fc18228201e45feacff87941948e4b233458dcfd92cc3da0.jpg)

b  
![](images/3b48b6d245c75086198d318340d7123e5e41245690dd21e972ddf21e80c51b48.jpg)

![](images/9e6b965def99f2d6d917856ee430ef153eda7ac6ad4ce8239b2fc625cfc9af6f.jpg)

c  
![](images/80f634a1e718d88a9010155ea4a80d48729cf7e52df9709ba8a378001d8ff57a.jpg)

![](images/be53b5877d8cbf7cc2d776753155fae211322600622386c3d04b3672936d87f9.jpg)

![](images/ceb71bb08b93d9b61f77164e4a272d5fb904863735907be3c7cbcb78893553cc.jpg)

![](images/3afb6c9edec1c3dadb1c4e4c9bc057490ad1eb480ec5d98621eb005d449cff7b.jpg)

![](images/fcf507ff033b20ddfef7dabb4e192f5a363e3685bd53399247a44f472ec7bf3e.jpg)

![](images/7adf2d731131a667192e01ea9c7260d628fa9b1d4bbdb48afb5efe8a4e39a117.jpg)

![](images/c00c22cc80cd846250dd98773e0154718455d33db5f005a62e322f17dbd16acf.jpg)

![](images/a55358fe42ffeb3582ddea1454851d4d451ca35da7935eace5f36d91401a9079.jpg)

![](images/0ddd0bedf1a27639067e17c221a7fb4bbe8ae5a0a7aab9500b7e4ede271a0ea8.jpg)

![](images/933f6bedb48b1bf0774cf532069cfcbaf4fc2fbf01a38cc563a4390c41d305f6.jpg)

![](images/5b3d6ac13d08a297ba403449b5025417c3301ede5464e4fdc4440b6866dc43e9.jpg)

![](images/868daeb4d513c304b8c8f7b0bd9d8b7cfc2ae6ead67e671c50cd82d7e0f5cb17.jpg)  
Fig. 3 Meta-neural semantic representations exhibit consistent geometric relationships and improved decoding performance across diverse tasks. a, Three-dimensional MDS projections of the low level neural observation and high-level meta-neural semantic representations in motor, perception and internal mentation, using the Motor Imagery, NSD and Spacetop datasets, respectively. Open and filled circles distinguish cognitive states; blue and orange distinguish cognitive conditions. Arrows connect cognitive states within each condition, and pale lines connect corresponding states across conditions. Greater parallelism across conditions indicates greater cognitive stability. b, Cross-condition generalization performance (CCGP) and parallelism score (PS) for the low-level and high-level representations. CCGP measures the generalization of cognitive state decoding across conditions, whereas PS measures the parallelism of the corresponding state-diference directions. Violin plots show the distributions of both metrics. $\mathbf { c } ,$ Within-participant decoding accuracy for LaBraM, BrainOmni, CBraMod and the proposed framework across BCIC, FACED, SEED-V and Motor Imagery $( n = 9 , 1 2 3 , 1 6$ and 10 participants, respectively). The datasets comprised four motor-imagery classes, three afective states, five afective states and two movement states, respectively. d, Within-participant confusion matrices for the proposed frame work; rows denote true classes, columns predicted classes and cells row-normalized trial percentages. $\mathbf { e } ,$ Zero-shot decoding accuracy for participants excluded from model training. In c and $\mathbf { e } ,$ boxes show medians and quartiles; whiskers extend to the most extreme values within 1.5 × IQR of the box limits, with outliers shown individually. Points denote individual participants. One-way repeated-measures ANOVA with Greenhouse–Geisser correction followed by Sid´ak-adjusted multiple comparisons was used in<sup>ˇ</sup> c and e. Asterisks denote statistical significance $( ^ { * } P < 0 . 0 5 , ^ { * * } P < 0 . 0 1 , ^ { * * * } P < 0 . 0 0 1$ and \*\*\*\* $P < 0 . 0 0 0 1 )$ .

Normalized amplitude profiles changed markedly between recording sessions in the low-level neural observation representation but remained more consistent in the high-level metaneural semantic representation (Fig. 4a, c, e). The corresponding cross-session similarity matrices showed higher between-session similarity at the high level in ECoG Speech, AJILE12 and SHU-MI (Fig. 4b, d, f). Mean similarity increased from 0.393 to 0.803 in ECoG Speech, from 0.459 to 0.819 in AJILE12, and from 0.214 to 0.758 in SHU-MI. The high-level representation therefore preserved condition-specific amplitude patterns as low-level neural observations drifted across sessions.

In ECoG Speech, the high-level representation retained clear state separation in both the reference recording and the 42-day follow-up, whereas the low-level representation showed greater withinclass dispersion (Fig. 4h). This diference was also reflected in higher silhouette coeficients at the high level (Fig. 4i). Moreover, the meta-neural semantic representation preserved more consistent representational trajectory geometry across sessions, whereas the corresponding low-level trajectories changed more substantially between D0 and D42 (Fig. 4g). Representational displacement relative to the reference recording remained lower at the high level throughout the 208-day period, including follow-ups more than 200 days after day 0 (shaded interval in Fig. 4j). The respective lowlevel and high-level displacement values were 17.5 and 3.7 in ECoG Speech, 12.7 and 2.5 in AJILE12, and 17.9 and 5.3 in SHU-MI. These values corresponded to reductions by factors of three to five across the three datasets (Fig. 4k). This stability was accompanied by higher mean cross-session decoding accuracy. Across 20 matched random seeds, the framework showed higher mean crosssession accuracy than SPaRCNet at every ECoG Speech follow-up and than both SPaRCNet and BrainOmni across all three datasets (Fig. 4l–m).

Over days to months of neural drift, metaneural semantic representations preserved relatively stable relationships among cognitive states and supported cross-session decoding. Cognitive states could therefore be recovered consistently from time-varying neural observations, indicating that stable recovery does not require fixed neural activity patterns.

## Cognitive states exhibit semantic consistency across perceptual domains

Meta-neural semantic representations supported open-ended decoding of cognitive content across language and vision.

Language decoding was evaluated in Alice, LPPC-fMRI, StudyForrest and LPPC-HK, covering English, Mandarin Chinese, Cantonese, French and German<sup>26–29</sup>. BGEScore measured the sentence-level semantic similarity between decoded and reference text<sup>30</sup>. The median BGEScore was 0.588 in Alice, 0.638 in LPPC-fMRI, 0.559 in

a  
![](images/ba47f25d07f10cdb4b3fa2712109eb44017059a8f1d8ce7b510aacfec3821b8b.jpg)

![](images/695b4a0a151cbf4dff7d14b82d8f17b9da86b2ece3eca89b59bebcedaacd2802.jpg)  
c

![](images/b4882c7c1513fc5c05028a65068ba0a158d8b3d0fe821ed8844298947bb859ce.jpg)

![](images/e4a4e70d9cb7d0d65a1baa0a7f41ce872ee7c0529561308dfa6a6f1e418869ce.jpg)

![](images/ba954d69bf74aa462c86143c521bde59d1420eaa3b628ed617267a243f6b636e.jpg)

d  
![](images/e143d5289c5c7bad6ed0d627ad455d710ed3a4b9494fb404bf10907a50d8572c.jpg)

![](images/5500cf782e3b328b11339537d5eaef364154f785585ed4c414d274b7b9f424d9.jpg)

![](images/dcc70c8678dab6b765bb00f76eb8284861ca7e75b075e0d4e3d6f9a5d28bd481.jpg)

e  
![](images/bc46b8e1178dab48cbafb90ed47fff9cdc6d22645d279378e79f0134bc0bd95f.jpg)

![](images/942bd3b95249760f27c107133b2784bd20aefa5efd349291b701ee874fc0474f.jpg)

f  
![](images/5f5fb7159819b3cd7873b56f3bcabceef9c5fce9b08c342ffaa2a8232b5a4ca8.jpg)

![](images/3766f6b393cc97a52cd00a093fce207b3d8d252590c0a8fda8d86b496acb82ae.jpg)

g  
![](images/4ffc82ce06064366abe75902e4ccbb31c93427a9a2d96ff0a875b1ec9ce5be52.jpg)

![](images/d5ce28689b5799275088b2aeba7f45e45ac7c11c52ff27b1a3c88c29b9e0e438.jpg)

![](images/8f6470e01b000e95a4fed9f3359fd70fd4e76442ae1a69b5575adf025655e028.jpg)

![](images/eb0a10ff322fbb2d8731d4b8de7de528cf3ae1f74c1a49a7b46e54409d6ea2d5.jpg)

![](images/420ec57d2fab7bd926230d629f017bea6b58e877333bbd83759e40b3691cb62b.jpg)

![](images/1993c97963cba4caa1bbae1de6499a2a8c612fa94cf35ddf915443deead7536f.jpg)

![](images/ac895724fc9ae6499c57fe452c52c62f676b71038f14b182652c1eca0c1a4f8c.jpg)

h  
![](images/c8ae354b57067812edebfa4240190b3ec585a2f1ed4eeedc89a61b78c8755445.jpg)

![](images/d5298dc47095e8aabb53fb8dd706044d5a504525d346c001e91c4b521d1367e4.jpg)

![](images/a208227cbac1ea7c2fbf4ec8923d24b1c3da15aeb86fdc51289d02951305aa66.jpg)

i  
![](images/f5d85f0612001b79ca548ff05ebdc129d20ed7ae784dd3d007be388bb62ab229.jpg)

![](images/30d8e186094de59dfc9019383fb5cf9b45d9a9a31c550e2b85b21dcbe8054ca4.jpg)

![](images/8aaeaeaa77e06d8c50fe17fe392dc0dd8088556c070f25ec35dcc2bb34d5dd8a.jpg)

![](images/245c7ba7c7f10908275656c6ccc7e8101e948cdede329af69b30e4aecee2a662.jpg)

l  
![](images/35492d4dbfa66aaeab4264c118c5e7916842975ba5420e7b78f5a5969887fc7e.jpg)

m  
![](images/6eaee3d63ee0ba9cee5adc27fd5cfabfe3cd2e5db9c029632c729b0d93d33395.jpg)  
Fig. 4 Meta-neural semantic representations support stable decoding across days to months of neural drift. ${ \mathbf { a } } , { \mathbf { c } } , { \mathbf { e } } .$ Normalized amplitude profiles of the low-level neural observation and high-level meta-neural semantic representations across sessions in ECoG Speech, AJILE12 and SHU-MI, respectively. b, d, f, Corre sponding cross-session similarity matrices shown on a common scale. g, Low-level and high-level representational trajectories at D0 and D42 in ECoG Speech, projected onto the first three principal components; temporal profiles and Pearson correlations between D0 and D42 are shown for each component. h, t-SNE projections at D0 and D42. $\mathbf { i } ,$ Silhouette coeficients at D0 and D42 across 50 matched initializations. Boxes show medians and quartiles; whiskers extend to the most extreme values within 1.5 × IQR of the box limits, with outliers shown individually. j, Representational displacement from D0 across 208 days in ECoG Speech; shading marks follow-ups beyond 200 days. $\mathbf { k } ,$ Low-level and high-level representational displacement across the three datasets. l, Mean ECoG Speech cross-session decoding accuracy for SPaRCNet and the proposed framework at each follow-up; points and error bars show mean $\pm \ \mathrm { s . d . }$ . across 20 matched random seeds. m, Mean cross-session decoding accuracy for SPaRCNet, BrainOmni and the proposed framework across ECoG Speech (six classes), AJILE12 (two classes) and SHU-MI EEG (two classes). Violin plots show distributions across the same 20 matched random seeds; internal boxes indicate medians and interquartile ranges.

StudyForrest and 0.631 in LPPC-HK, and the proposed framework outperformed CogReader and the random-output control in all four datasets (all $P < 0 . 0 0 0 1$ ; Fig. 5a) <sup>31</sup>. This advantage was sustained throughout entire narratives, as illustrated by the Chinese and French stories in LPPCfMRI (Fig. 5b–c). Representative decoded passages retained central entities, actions and event relations from the reference narratives (Fig. 5d). These results show that the framework reconstructed text semantics more accurately across multiple languages.

Image decoding from meta-neural semantic representations was evaluated within participants in NSD (n = 8 participants)<sup>32</sup>. Representational similarity analysis (RSA) showed a stronger correspondence between the high-level meta-neural semantic representation and image-description semantics than between the low-level neural observation representation and image-description semantics $( P <$ 0.01; Fig. 5e). Compared with Neural Code Conversion $\mathrm { ( N C C ) ^ { 3 3 } }$ , the proposed framework achieved higher top-1, top-5 and top-10 image-retrieval accuracy $( P ~ < ~ 0 . 0 0 1$ for top-1 and $P ~ < ~ 0 . 0 0 0 1$ for top-5 and top-10; Fig. 5f), higher CLIP similarity for reconstruction of viewed images (visual perception) $( P < 0 . 0 1 ; \mathrm { F i g . } 5 \mathrm { g } )$ and mental images (visual imagery) $( P < 0 . 0 5 ; \mathrm { F i g . 5 h } )$ <sup>33</sup>. Representative retrieval results identified images that matched the target semantics, and representative reconstructions retained the principal objects, scenes and visual content of viewed and imagined targets (Fig. 5i–j).

In language and vision, cognitive inference recovered semantic content consistent with the target cognitive states despite diferences in sensory inputs and neural activity patterns. This semantic consistency in both perceptual domains supported open-ended decoding, showing that cognitive content could be recovered through cognitive inference across diferent forms of sensory input.

## Cognitive inference extends neural decoding from externally evoked content to internal mentation

Internal cognitive states can arise endogenously during processes such as imagined speech, memory recall and self-generated thought, and can vary between individuals even in response to the same external stimuli. We therefore asked whether these states could be inferred from neural observations using the Bayesian brain hierarchical decoding framework.

In the Imagined Speech task, we used an optically pumped magnetometer magnetoencephalography (OPM-MEG) dataset collected for this study $( n ~ = ~ 6$ participants). Participants read a text prompt and then imagined its content during a 4-s imagery phase; only neural observations from this phase were used (Fig. 6a). The framework achieved a median semantic correlation of 0.567, which was higher than that obtained with BrainOmni $( P <$ 0.05; Fig. 6b). Median three-class accuracy reached 48%, compared with 33% for BrainOmni $( P \ <$ 0.05; Fig. 6c). These results show that meta-neural semantic representations preserved the semantic content of internally generated cognitive states during imagined speech.

StudyForrest (German)

Alice (English)  
![](images/26106da7ef95d1a5368d121538de513b8458f17b9f83a0c746bb2322ebae3c5a.jpg)

![](images/7e3b073b4ed04f1a12f6e1b31ccc58f4c0b89c2cd5bc0a43948b7152a9ca1d0e.jpg)  
b

![](images/639f2c2a9f53a2caee92d8df438825f694830c5357f8c56fef79ec6438f015f7.jpg)  
c

![](images/5f0454e4b83393cb62ea11e7cb464d43646ca9c11da1b81a7aa32fb0584d58f6.jpg)

![](images/da93245d2a7f9586d060fd1f64950156097a25e5ae3331e9ada2216e5e57725a.jpg)  
d

星星就是他们探讨的学问，对我所遇见的那个实业家来说，星星是金钱。

星星的事儿我也对它们感兴趣，可是我不是喜欢金钱，那些实业家却是金钱爱好者。

王子的事儿我也对它们感兴趣，可是他们却只喜欢编号。

![](images/b35f19eb0cea537c974505028b54034d3b6cc30707b0e9179e92735925117395.jpg)  
LPPC-fMRI (French)

Le petit prince enferme sa fleur toutes les nuits sous son globe de verre.

Le petit prince achevait de regard er, il ne renonçait pas à la fleur

Jenny sieht ihm in die Augen, lächelt traurig und nimmt seine Hand.

Translation: The little prince had just finished looking, he would not give up the flower.

Je ne savais pas que c'était un serpent, il y avait des serpents.

Jenny nimmt seine Hand und lächelt, denn er ist traurig.

She tried the little golden key in the lock, and Alice opened the door.

Translation: Jenny takes his hand and smiles, because he is sad.

Er steht am Fenster und schaut auf die Straße hinaus.

e  
i  
![](images/32289c91ebb315ce7e406a76fc3ca21a0da456c577fb85b0f29078f8ddb35340.jpg)

f  
![](images/2326848823bf24da3c392f0024728f4067522fb9ad9f5319573f427c9d038d3d.jpg)  
Image retrieval

g  
![](images/1b45c46be908eb067c992cbbb13297f1d113756d0f744cc1d126b2425f62ea89.jpg)

h  
![](images/02cf007e9ac7d3110a85a65607b89da36a46d4cadcba6f1fb3f993f795d1e294.jpg)

![](images/e31be71430ab5d478057b955d92a8709c5c8f60842dfa013734a20ee062ad75a.jpg)

![](images/a0d1bad1461f7b149a308329ab084e6a2589f481c3fc6c8b969524e5bd6da647.jpg)  
Fig. 5 Cognitive inference enables open-ended decoding of cognitive states across language and visual perceptual domains. a, Sentence-level BGEScore between decoded and reference text for the proposed framework, CogReader and a random-output control across Alice, LPPC-fMRI, StudyForrest and LPPC-HK $( n = 2 6 ,$ , 112, 20 and 52 participants, respectively). $\mathbf { b - c , }$ Segment-wise BGEScore for Chinese $( n = 3 5 )$ and French $( n = 2 8 )$ LPPC-fMRI participants, respectively; lines and error bars show mean ± s.e.m. d, Representative decoded passages in English, Chinese, French and German. Green highlighting denotes content semantically consistent with the reference, whereas red highlighting denotes divergent content. e–h, Within-participant visual decoding in NSD $( n = 8 )$ . e, RSA between image-description semantics and the low-level neural observation or high-level meta-neural semantic representation. $\mathbf { f - h } ,$ Comparison with NCC for top-k image retrieval $( k = 1$ 5 and 10) (f), viewed-image reconstruction (g) and mental-imagery reconstruction (h). For each retrieval trial, the candidate set comprised the target image and 99 randomly selected non-target images from the 1,000-image test set. i, Representative image-retrieval results. j, Representative image reconstructions. Boxes in a and f–h show medians and quartiles; whiskers extend to the most extreme values within $1 . 5 \times \mathrm { I Q R }$ of the box limits, with outliers shown individually. The violin plot in e shows participant-level distributions with medians and quartiles. Two-tailed paired t-tests were used in a against CogReader and in e–h; two-tailed Welch’s t-tests were used in a against the random-output control. Asterisks denote statistical significance $( ^ { * } P < 0 . 0 5$ $^ { * * } P < 0 . 0 1$ $^ { * * * } P < 0 . 0 0 1$ and $^ { * * * * P } _ { } < 0 . 0 0 0 1 )$ .

a  
![](images/507a5b4558d226c293c003fd8170e5b1aaeac20b468b5c0bb96cfe229f2593a1.jpg)

b  
![](images/80ae3454775cf6abc7303ffd6795fd025a22b9bbbadef23ef4323bbea4f11408.jpg)

c  
![](images/c3db426fe4d7e5188a9b19c5b2002f4c2d76b2852739f01ad69f29d141ee4a3b.jpg)

d eSelf-generated thought  
![](images/f3aba7e46e0330dab75212a7388d0f67014d471a71d153b51f2c6cfa1b56cdec.jpg)

![](images/3c1b5f4d5e78ee231cb593f5a9bec6ece4b44f71f9d3921630a394e223145065.jpg)  
f

![](images/0b0f990781c40dc0770bff2ec480fc5028e971ce6413d7c8da778e8f96cee53f.jpg)

![](images/1175727be5d0b956443a132cc3ea0824b69fdedad1ccc8c3c30c48eac9d32af7.jpg)

h  
![](images/f5b391e2fc68beab005dac38184b8cbdebf44c23b76a60ffb340bbfb47900194.jpg)

![](images/c01696fac463a4c9a57cac0e5825a962a999223235cb7df82ba93fccb5c8d11f.jpg)

![](images/216e0e06dd54842d896b02454a94a8d5e9eba7d15c489b346db3f99a1fe6724d.jpg)

k  
![](images/aa3ec37f7a64d19b13b3aa0d9a3e08d40c310f7911a59e90c330c42bfc56b6d3.jpg)

![](images/1a46623f0884b70263d2dcd2863edffa97c75278bd8d5d6696858731fafeb73e.jpg)

l  
![](images/9d53db19e7bf2a8168373449b28b14b3a34cfc6e7ab8a1a18724de1fde7a0722.jpg)

m  
![](images/71dbe45fe4befb9878acf4b2739b908903f832d03680a235d73bd4198506eee4.jpg)

![](images/aeb79eba366707d4bb2dcd87d233cc3ce62334a10586bef928d5d813442620e2.jpg)

n  
![](images/9a340daf91ad65ba3b96e41170545f6b9ae0b7f1cdf68cbb38f6d1d403ae6f44.jpg)  
Fig. 6 Meta-neural semantic representations capture cognitive states in internal mentation. a– c, Imagined Speech OPM-MEG experiment $( n = 6 )$ . a, Reading, imagery and rest phases with corresponding event-related fields. $\mathbf { b } ,$ Semantic correlation during imagery for BrainOmni and the proposed framework. c, Threeclass semantic-category decoding accuracy for imagined content. d–g, Self-generated thought in two public fMRI datasets, comparing NeuroSTORM with the proposed framework. d, State-classification tasks. e, Three-class decoding accuracy in DuPre2016 $( n = 3 1 )$ ). f, Two-class decoding accuracy in Lee2021 $( n = 2 6 )$ . g, Correlations between predicted and reported clarity, vividness and valence. h–j, Participant-specific subjective ratings in Spacetop Alignvideo $( n = 3 0 )$ , comparing NeuroSTORM with the proposed framework. h, Shared films and subjective ratings. i, Prediction–rating correlations across five emotional dimensions. j, Residual prediction–rating correlations after removal of the film-specific component shared across participants. k, Representational geometry of happiness for four selected participants, comparing $\mathrm { N e u r o S T O R M }$ with the proposed framework. Arrows represent participants and point towards higher happiness ratings; greater parallelism indicates more consistent representational relationships between happiness states across participants. l, Mean cross-participant parallelism of representational changes associated with happiness ratings $( n = 3 0 )$ $^ { * * * * P } < 0 . 0 0 0 1$ . m, Pearson correlations between neural representational distance and subjective-rating distance across 136 pairs of 17 participants. $\mathbf { n } ,$ Participant-level RSA correlations between neural representations and subjective ratings in the same participants. Boxes in b, c, e–g and i show medians and quartiles; whiskers extend to the most extreme values within 1.5×IQR of the box limits, with outliers shown individually. In l, violin plots show distributions of participant-level mean parallelism, with individual participants shown as black points. Violin plots in j and n show participantlevel distributions, with internal lines indicating medians and quartiles. Two-tailed paired t-tests were used in $\mathbf { b } ,$ c, e, f and n. For l, the between-representation diference in mean parallelism was tested using a participant-level leave-one-out jackknife standard error and a two-sided t approximation (29 d.f.), with Holm correction across five emotional dimensions. RSA coeficients in n were Fisher z-transformed before testing. Two-way repeated-measures ANOVA followed by Sid´ak-adjusted multiple comparisons was used in <sup>ˇ</sup> $\mathbf { g } ,$ i and j. Asterisks denote statistical significance $( ^ { * } P < 0 . 0 5 , ^ { * * } P < 0 . 0 1 , ^ { * * * } P < 0 . 0 0 1$ and \*\*\*\* $P < 0 . 0 0 0 1 )$

Self-generated thought was assessed in the DuPre2016<sup>34</sup> and Lee2021 fMRI datasets<sup>35</sup>. The classification tasks distinguished among selfgenerated thought states, including autobiographical recollection and imagined future events (Fig. 6d). The framework achieved higher stateclassification accuracy than NeuroSTORM in DuPre2016 $\mathit { \Pi } ^ { ' } P \ < \ 0 . 0 0 0 1$ ; Fig. 6e) and Lee2021 $( P ~ < ~ 0 . 0 1$ ; Fig. 6f) <sup>36</sup>. Predictions also tracked the clarity of autobiographical recollection and the vividness and valence of imagined future events, with higher prediction–experience correlations than those obtained with NeuroSTORM for all three measures $( P < 0 . 0 0 0 1$ for all comparisons; Fig. 6g). Thus, the framework distinguished selfgenerated thought states and predicted continuous variation in their reported experience.

Subjective feelings elicited by shared films were assessed in Spacetop Alignvideo. The 30 participants with the most complete film ratings viewed the same videos and rated fear, engagement, happiness, sadness and disgust <sup>37</sup> (Fig. 6h). Prediction–rating correlations were higher for the proposed framework than for NeuroSTORM in all five dimensions $( P ~ < ~ 0 . 0 0 0 1$ for all comparisons; Fig. 6i). To determine whether this performance captured participant-specific experience rather than only film-related responses shared across participants, the shared film-specific component was removed separately from the observed ratings and model predictions. Correlations between the residual components remained higher than those obtained with NeuroSTORM in all five dimensions $( P _ { \mathrm { ~ \normalfont ~  ~ } } < \mathrm { ~ \normalfont ~ 0 . 0 0 0 1 ~ }$ for all comparisons;

Fig. 6j). The residual correlations therefore captured participant-specific subjective variation after accounting for responses to each film that were shared across participants.

Participant-specific decoding was accompa nied by consistent representational relationships between subjective states across individuals. For happiness, these relationships were more parallel across individuals in the proposed framework than in NeuroSTORM (mean parallelism, 0.59 versus 0.01; n = 30; $P ~ < ~ 0 . 0 0 0 1$ ; Fig. 6k–l). Similar increases were observed for sadness, fear, disgust and engagement (0.37–0.52 versus −0.01–0.08; all $P ~ < ~ 0 . 0 0 1 )$ . These results indicate that metaneural semantic representations captured consis tent relationships between subjective states across individuals. Across participant pairs, neural rep resentational distance was related to subjectiverating distance for the 17 participants with complete ratings in all five dimensions. The association reached $r = 0 . 3 1 4$ for the proposed framework and $r ~ = ~ 0 . 1 4 5$ for NeuroSTORM (Fig. 6m), indicat ing that participants with more similar meta-neural semantic representations reported more similar subjective ratings. Within-participant RSA pro vided a complementary test of the correspondence between representational and rating dissimilarity. Participant-level RSA correlations were higher for the proposed framework than for NeuroSTORM $( P < 0 . 0 0 0 1$ ; Fig. 6n). Figs. 2 and 4 show corre sponding results in additional rating tasks. These results demonstrate that the similarity structure of meta-neural semantic representations aligned with that of subjective evaluations both across and within participants.

Meta-neural semantic representations captured both internally generated cognitive content and distinct participant-specific subjective states elicited by the same stimuli. These findings extend neural decoding to cognitive content generated internally and shaped by individual subjective interpretation, beyond what external stimuli alone specify. Within the framework, such cognitive states were inferred from neural observations under brain-intrinsic priors, rather than identified through a direct mapping between stimuli and neural responses.

## Discussion

Changes in neural activity patterns do not imply changes in cognitive state <sup>2</sup>. A key question in neural decoding is therefore how to resolve the cognitive states that remain stable in the brain despite continuously changing neural population activity <sup>4,5</sup>. Existing neural decoding methods, which are largely grounded in the stimulusresponse principle, treat predefined external labels as decoding targets and learn statistical correspondences between neural observations and these labels <sup>7,8</sup>. When neural activity changes continuously, reliance on such statistical correspondences tends to map changes in neural observations onto changes in cognitive state. These methods thus have dificulty resolving stable cognitive states from changing neural observations. Here, we redefine neural decoding as the process of inferring the cognitive state of the brain from neural observations, and we propose a Bayesian brain hierarchical decoding framework as a computational implementation of this inference. The proposed framework recovers relatively stable cognitive states from changing neural observations, thereby distinguishing changes in neural observations from changes in cognitive states. Accordingly, the framework not only extends the objective of neural decoding but also provides an analytical tool for examining the relationship between changes in neural activity patterns and the stability of cognitive states.

From a neural manifold perspective, cognitive states are embedded within the manifold structure formed by whole-brain neural population activity <sup>2</sup>. Existing neural recording technologies capture only partial observations of this whole-brain activity <sup>6</sup>. Neural observations are therefore projections of whole-brain neural population activity onto a specific observation space, and they reveal the relationships between cognitive states only incompletely. Our Bayesian brain hierarchical decoding framework treats limited neural observations as evidence and constrains cognitive inference with brain-intrinsic priors. This inference yields meta neural semantic representations that carry the semantic content of cognitive states. Representational geometry analyses further revealed that cognitive inference was accompanied by a systematic geometric reorganization of the meta-neural semantic representation manifolds (Fig. 3a–b; Extended Data Fig. 3). Cognitive inference altered the relative configuration of the manifolds corresponding to diferent cognitive states in representational space, such that geometric relationships susceptible to variation in finite neural observations became more stable in the high-level representation. Consequently, the high-level representation preserved the relative relationships among cognitive states even as neural observations changed. At the level of representational geometry, cognitive stability may therefore be understood as the preservation of the relational structure among cognitive states.

Stable cross-session neural decoding does not require neural activity patterns to remain unchanged <sup>2,38,39</sup>. Neural drift substantially alters the neural observations associated with a given cognitive state across sessions, even when the cognitive state itself remains stable. The free-energy principle provides a theoretical perspective on how changes in neural activity can coexist with cognitive stability. Within this framework, free-energy minimization characterizes how the brain maintains relative stability in a changing environment without requiring fixed patterns of neural activity<sup>13</sup>. From this perspective, neural drift can be understood as cross-session variation in neural activity that is compatible with stable cognitive states, rather than as evidence that the cognitive states themselves have changed. Our results are consistent with this view. Despite neural drift over days to months, cognitive inference with our method yielded meta-neural semantic representations whose representational geometry remained consistent across sessions (Fig. 4). On this basis, the proposed framework maintained stable decoding performance across sessions. Together, these results indicate that stable cross-session neural decoding need not rely on aligning neural observations across sessions, but can instead be achieved by inferring stable cognitive states from changing neural observations.

The cognitive state of the brain is not uniquely determined by external stimuli. Cognitive states also arise from internal processes such as imag ination and spontaneous thought, and the same external stimulus gives rise to diferent subjective interpretations. Decoding based on the stimulusresponse principle takes predefined external labels as targets and thereby prescribes stimulus content as the decoding outcome. Such decoding mainly reproduces stimulus information rather than inferring the cognitive state of the brain. In contrast, results from the internal mentation experiments (Fig. 6) showed that the meta-neural semantic rep resentations inferred by the proposed framework are not restricted to stimulus content but also reflect internally generated cognitive states and subjective interpretations. Our method therefore extends the scope of neural decoding to internal cognitive processes, including memory, imagination, judgment and subjective interpretation, and provides a computational approach for investigating the cognitive states generated by such processes. This extension broadens the range of information accessible to human-machine interaction and provides a computational foundation for cognitive brain-computer interfaces <sup>40</sup>.

Human brain structure exhibits pronounced inter-individual variation <sup>41</sup>. Because brainintrinsic priors are shaped by brain structure, this variation implies that cognitive inference in diferent individuals is constrained by different priors. From this perspective, individual diferences in brain structure become a source of individuality in cognitive inference itself. The individualized cortical geometric priors in the proposed framework capture this individuality, and the decoding improvement they bring supports this view (Extended Data Fig. 6). At the same time, the cognitive states of diferent individuals are inferred through the same process, and individual diferences enter this process through the constraints imposed by each brain’s prior. Our method thereby achieves personalized decoding under a common inference principle, constraining cognitive inference with each brain’s own structure to recover stable cognitive states from individual neural activity.

In this study, we incorporated constraints imposed by brain structure on neural activity into cognitive inference as brain-intrinsic priors, and instantiated these structural priors using cortical geometric eigenmodes <sup>16,18</sup>. Future studies could also consider priors defined by neural dynamics and examine how diferent brain-intrinsic priors shape cognitive stability <sup>42,43</sup>. For example, temporal priors derived from neural dynamics could allow cognitive inference to directly exploit regularities in the evolution of cognitive states, enabling their formation to be tracked over longer timescales. Overall, our experimental results indicate that, in a continuously changing external environment, the brain need not maintain fixed patterns of neural activity to sustain relatively stable cognitive states, but can instead achieve stability through cognitive inference from changing neural activity.

## Methods

## Bayesian brain hierarchical decoding framework

Bayesian brain theory motivates hierarchical inference from current neural evidence under constraints imposed by brain-intrinsic priors (Fig. 1). We denote the neural observation representation by $^ { o , }$ the prior-constrained representation by u, and the inferred high-level meta-neural semantic representation by z. The inference is formulated as

$$
p ( z \mid o , u ) \propto p ( o \mid z ) p ( z \mid u ) .\tag{1}
$$

Here, $p ( o \mid z )$ quantifies the compatibility of a candidate representation z with the current neural observation representation, whereas $p ( z \mid u )$ constrains the candidate representation according to the brain-intrinsic prior. Their product defines the inference over z.

Neural observation encoder. Modalityspecific pretrained encoders transformed neural recordings from EEG, MEG, ECoG, fNIRS and fMRI into the neural observation representation o. Grouped spatial-coordinate embeddings and factorized attention accommodated diferences in recording length and channel configuration. The resulting o represented the evidence provided by the current neural observation.

Brain-intrinsic prior encoder. The prior encoder incorporated brain-intrinsic prior information into the current neural representation to produce the prior-constrained representation u. In the present implementation, the fixed cortical geometric eigenmode basis instantiated the brain-intrinsic structural prior, and u represented the current neural observation under this structural constraint. Their projection coeficients quantified the expression of the current neural observation within the fixed eigenmode basis, whereas the corresponding eigenvalues indexed spatial scale. These quantities were encoded across spatial scale and time to obtain $u ,$ which imposed the brain-intrinsic prior constraint during inference of z. We denote the eigenmode projection operator by $P _ { \Psi }$ , such that $\mathbf { a } = { P } _ { \Psi } ( o )$ gives the eigenmode coeficient vector associated with neural observation representation o.

Inference of meta-neural semantic representations. To infer the meta-neural semantic representation $z ,$ we formulated inference under two complementary constraints derived within each recording modality from the same neural observation. The neural observation representation o retained evidence from the current recording, whereas the prior-constrained representation u encoded brain-intrinsic structural constraints. We implemented this Bayesian formulation using a temperature-scaled product of experts <sup>44</sup>:

$$
p _ { T } ( z \mid o , u ) \propto p ( o \mid z ) ^ { 1 / T _ { o } } p ( z \mid u ) ^ { 1 / T _ { u } } .\tag{2}
$$

The positive temperatures $T _ { o }$ and $T _ { u }$ controlled the relative contributions of the observation and prior-constrained factors, respectively. At the representation level, the two factors were implemented as isotropic Gaussian experts with a common scale and means centered at o and $u ,$ respectively, yielding the normalized inverse-temperature weights

$$
\omega _ { o } = \frac { T _ { o } ^ { - 1 } } { T _ { o } ^ { - 1 } + T _ { u } ^ { - 1 } } , \qquad \omega _ { u } = \frac { T _ { u } ^ { - 1 } } { T _ { o } ^ { - 1 } + T _ { u } ^ { - 1 } } ,\tag{3}
$$

and the inferred representation

$$
z = \omega _ { o } o + \omega _ { u } u .\tag{4}
$$

The resulting z constituted the meta-neural semantic representation used for subsequent decoding.

Masked-autoencoding pretraining. The neural observation encoder and brain-intrinsic prior encoder were pretrained independently as masked autoencoders, without access to task labels. Fifty percent of the temporal and spatial patches were randomly masked. A reconstruction decoder predicted the masked patches from the corresponding encoder representation, with reconstruction loss evaluated only over the masked index set Ω:

$$
\mathcal { L } _ { \mathrm { M A E } } = \frac { 1 } { \left| \Omega \right| } \sum _ { p \in \Omega } \left\| \hat { \boldsymbol { x } } _ { p } - \boldsymbol { x } _ { p } \right\| _ { 2 } ^ { 2 } .\tag{5}
$$

Here, $x _ { p }$ denotes the target patch at masked position $p ,$ and $\hat { x } _ { p }$ denotes its reconstruction. Pretraining was performed independently for each neural recording modality using only unlabeled neural recordings from the corresponding training partition.

## Cortical geometric eigenmodes

The current implementation instantiates the brainintrinsic prior through intrinsic cortical geometry, represented by a fixed eigenmode basis. This geometry constrains the spatial propagation of largescale neural activity. Its eigenfunctions form an orthogonal basis ordered by spatial scale and fixed across experimental tasks. Modality-specific mappings express this basis in each measurement space. Previous work has shown that these eigenmodes account for the large-scale organization of task activity and resting-state networks <sup>16,17</sup>.

Cortical eigenmode construction. The Laplace–Beltrami operator $\Delta$ was defined on the midthickness triangular cortical surface. Its eigenfunctions satisfy

$$
\Delta \phi _ { k } = - \lambda _ { k } \phi _ { k } .\tag{6}
$$

The non-negative eigenvalues were ordered from low to high. Smaller eigenvalues correspond to larger spatial scales. The constant zeroth mode was removed. The remaining K modes were assembled as $\Phi = [ \phi _ { 1 } , \ldots , \phi _ { K } ]$

Modality-specific eigenmode mapping. Cortical eigenmodes were used as a common geometric basis across neural recording modalities. Because diferent modalities sample neural activity in distinct observation spaces, the cortical eigenmode basis was mapped to each modality according to its spatial sampling geometry, and the resulting modality-specific basis was used to obtain eigenmode coeficient time series from the observed neural signals.

The fMRI eigenproblem was solved separately on both hemispheres. The analysis used the standard 32k fs LR midthickness surface from the Human Connectome Project <sup>45</sup>. Each hemisphere contains 32,492 vertices. The cortical atlas region of interest mask removed the medial wall. A finiteelement solver computed the eigenmodes. The analysis retained 1,000 modes from each hemisphere, giving 2,000 modes in total. Cortical time series were projected by least squares to obtain eigenmode coeficient time courses.

For EEG and MEG, dipole sources were constrained to a common cortical source space, and a three-layer boundary-element model was used to compute the forward solution for each modality. For each modality, the resulting lead-field matrix mapped the cortical basis to sensor space.

The forward operator mixes cortical modes into correlated sensor patterns. Singular value decomposition was used to process the mapped basis, and selected components formed a sensor-space basis for estimating coeficient time series. Cortical spatial-scale information was incorporated into the sensor-space representation.

For ECoG, the cortical eigenmodes were evaluated at the contact locations. Dataset-specific contact coordinates were mapped to the standard surface. Contact j was assigned to its nearest cortical vertex $v _ { j } .$ , giving $\widetilde { \Phi } _ { j k } = \phi _ { k } ( \boldsymbol { v } _ { j } )$ . Tikhonovregularized least squares estimated the coeficients from the observed contact signals.

For fNIRS, source-detector midpoints projected the surface eigenmodes into channel space. The components were orthogonalized before separate projection of HbO and HbR.

Participant-specific eigenmodes were constructed analogously using each participant’s cortical surface. Individual surfaces were registered to the standard 32k fs LR mesh to maintain vertex correspondence. The resulting participant-specific basis replaced the standard cortical basis in the modality-specific mapping procedures described above, with all subsequent steps unchanged.

Decoding and encoding with meta-neural semantic representations Decoding from meta-neural semantic representations. A task-specific linear classification head was applied to the inferred meta-neural semantic representation. For trial $i ,$ its categorical estimate was

$$
q _ { i } ( c \mid z _ { i } ) = \mathrm { s o f t m a x } ( W _ { \mathrm { c l s } } z _ { i } + b _ { \mathrm { c l s } } ) _ { c } .\tag{7}
$$

For target class $c _ { i } ,$ the classification head was fitted using cross-entropy:

$$
\mathcal { L } _ { D } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathrm { C E } ( c _ { i } , q _ { i } ( \cdot \mid z _ { i } ) ) .\tag{8}
$$

Here, $W _ { \mathrm { c l s } }$ and $b _ { \mathrm { c l s } }$ are the classification weights and bias, respectively, N is the number of training trials and CE denotes cross-entropy. All discriminative accuracy estimates were computed from $q _ { i } ( c \mid z _ { i } )$ . The pretrained encoders and temperature parameters remained fixed, and only $W _ { \mathrm { c l s } }$ and $b _ { \mathrm { c l s } }$ were fitted on the corresponding training partition.

Open-ended language decoding comprised semantic alignment followed by conditional generation (Fig. 5). BGE-M3 produced token-level text states that were resampled to a fixed-length semantic sequence <sup>30</sup>. A query-based cross-attention transformer (Q-Former) aligned the meta-neural semantic representation with this sequence <sup>46</sup>. Let κ¯ denote their mean tokenwise cosine similarity.

The objective combined reconstruction, cosinedistance and symmetric within-batch contrastive losses:

$$
\begin{array} { c } { { \mathcal { L } _ { A } = \alpha _ { 1 } \mathcal { L } _ { 2 } + \alpha _ { 2 } \left( 1 - \bar { \kappa } \right) } } \\ { { { } } } \\ { { + \alpha _ { 3 } \mathcal { L } _ { \mathrm { c o n } } . } } \end{array}\tag{9}
$$

The coeficients $\alpha _ { 1 }$ , $\alpha _ { 2 }$ and $\alpha _ { 3 }$ are loss weights. The term $\mathcal { L } _ { 2 }$ denotes reconstruction loss, and ${ \mathcal { L } } _ { \mathrm { c o n } }$ denotes the mean of the two cross-modal retrieval cross-entropies computed between meanpooled neural and text representations within each batch. Parameters of the alignment model and language model were shared across participants. The same story-level 9:1 split was applied to every participant. Semantic alignment was fitted before the language adapters, with both stages using the same training stories. Test stories were excluded from all model-fitting stages and used only for final generation and evaluation. The aligned sequence was projected into the hidden space of the 3.8- billion-parameter Phi-4-mini-instruct model and provided as a soft prompt<sup>47</sup>. PiSSA initialized the LoRA adapters on the attention projections <sup>48</sup>. Causal language-model loss was evaluated only at answer tokens. Beam-search decoding included penalties for sequence repetition and repeated $n -$ grams. Language-specific text normalization was applied to generated sequences before evaluation.

SD Image Variations v2.0 provided normalized $\mathrm { C L I P \ V i T { - } L / 1 4 }$ image embeddings <sup>49</sup>. The same cross-attention alignment model mapped the meta-neural semantic representation into this conditioning space and was fitted using the reconstruction, cosine-distance and symmetric within-batch contrastive losses defined above. A frozen latentdifusion decoder generated images from Gaussian latent noise <sup>50</sup>. Sampling used DDIM and classifier-free guidance <sup>51</sup>. Difusion-model parameters remained fixed. Sampling settings and random seeds were matched within each comparison. NOD trials were used exclusively for visual pretraining. All image-decoding evaluations were conducted within individual NSD participants. Participantspecific models were fitted on each participant’s NSD-core training data. Held-out NSD-core trials were used to evaluate viewed-image reconstruction, and the NSD imagery condition was used to evaluate mental-imagery reconstruction.

Encoding brain activity through metaneural semantic representations. Let $e _ { i } \in \Sigma$ $\mathbb { R } ^ { d }$ denote the external semantic representation of stimulus i and $b _ { i } \in \mathbb { R } ^ { J }$ its cortical GLM beta pattern. A semantic bridge $G$ produced $z _ { i } ^ { \mathrm { s e m } } = G ( e _ { i } )$ For training stimuli, the frozen hierarchical model determined the target representation $z _ { i }$ from $b _ { i }$ and its projection constrained by brain-intrinsic priors. Predictions for independent test stimuli were based exclusively on $e _ { i } .$

The target response was decomposed as $b _ { i } =$ $\Phi a _ { i } + \epsilon _ { i }$ , where $a _ { i } = \Phi ^ { + } b _ { i }$ and $\epsilon _ { i } = b _ { i } - \Phi { a } _ { i }$ . Two decoders predicted these components from $z _ { i } ^ { \mathrm { s e m } }$ ， and the predicted components were combined using positive weights:

$$
\hat { b } _ { i } = \gamma _ { \mathrm { m o d e } } \Phi D _ { \mathrm { m o d e } } ( z _ { i } ^ { \mathrm { s e m } } ) + \gamma _ { \mathrm { r e s } } D _ { \mathrm { r e s } } ( z _ { i } ^ { \mathrm { s e m } } ) .\tag{10}
$$

Both $\gamma _ { \mathrm { m o d e } }$ and $\gamma _ { \mathrm { r e s } }$ were constrained to be positive. The decoders $D _ { \mathrm { m o d e } }$ and $D _ { \mathrm { r e s } }$ predicted $a _ { i }$

and $\epsilon _ { i } ,$ respectively. The encoding objective was

$$
\begin{array} { r l } & { { \mathcal { L } _ { E } } = \beta _ { 0 } \left[ 1 - \cos ( z _ { i } ^ { \mathrm { s e m } } , z _ { i } ) \right] } \\ & { \qquad + \beta _ { 1 } \left\| D _ { \mathrm { m o d e } } ( z _ { i } ^ { \mathrm { s e m } } ) - a _ { i } \right\| _ { 2 } ^ { 2 } } \\ & { \qquad + \beta _ { 2 } \left\| D _ { \mathrm { r e s } } ( z _ { i } ^ { \mathrm { s e m } } ) - \epsilon _ { i } \right\| _ { 2 } ^ { 2 } } \\ & { \qquad + \beta _ { 3 } \left[ 1 - \mathrm { c o r r } \left( \hat { b } _ { i } , b _ { i } \right) \right] . } \end{array}\tag{11}
$$

Here, $\cos ( \cdot , \cdot )$ denotes cosine similarity, $\operatorname { c o r r } ( \cdot , \cdot )$ denotes Pearson correlation and $\beta _ { 0 } , \ldots , \beta _ { 3 }$ are loss weights. Caption embeddings were the primary semantic input. Category embeddings and image features served as controls. The bridge, response decoders and positive weights were fitted only on training stimuli.

The stimulus-response encoding baseline directly mapped the external semantic representation $e _ { i }$ to the cortical response $b _ { i }$ using an MLP, without hierarchical inference or cortical-geometric prior constraints. It used the same response targets, training and test stimuli, data partition and optimization protocol as the cognitive-inferential encoding model.

## Training and implementation details

The neural observation and brain-intrinsic prior encoders were independently pretrained as masked autoencoders with a reconstruction objective (masking ratio, 0.5) using AdamW (learning rate, $3 \times 1 0 ^ { - 4 } )$ for 60 epochs, with batch sizes of 32 and 64, respectively, and cosine learning-rate decay. The checkpoint with the lowest validation reconstruction loss was retained, after which both encoders were frozen. The positive temperatures $T _ { o }$ and $T _ { u }$ were calibrated separately for each recording modality using the validation partition of the corresponding training data. They controlled the relative strength of the neural observation and prior-constrained factors in the temperature-scaled PoE inference. The temperature pair with the lowest validation loss was retained and fixed for all held-out evaluations. The outputs of the two encoders entered the temperature-scaled product-of-experts (PoE) inference to obtain the meta-neural semantic representation. For language decoding, this representation was aligned with BGE-M3 embeddings through subject-specific layers and a Q-Former using a weighted combination of mean-squarederror, token-wise cosine and contrastive losses for 200 epochs (batch size, 256; AdamW; learning rate, $3 \times 1 0 ^ { - 4 }$ ; cosine learning-rate decay). For text generation, we then jointly optimized a linear projector and PiSSA-LoRA adapters (rank, 16; dropout, 0.05) on frozen Phi-4-mini using crossentropy with label smoothing of 0.1 for 15 epochs (batch size, 16; AdamW; learning rates, $3 \times 1 0 ^ { - 4 }$ for the projector and $5 \times 1 0 ^ { - 5 }$ for LoRA; cosine learning-rate decay).

For image decoding, the same representation was aligned with CLIP image embeddings through a visual alignment head using a weighted combination of mean-squared-error, cosine and contrastive losses for 200 epochs (batch size, 64; AdamW; learning rate, $2 \times 1 0 ^ { - 4 } )$ . The aligned embeddings were decoded into images using a frozen difusion generator without updating its parameters. All models were trained on two NVIDIA A800 GPUs with 80 GB memory each.

## Datasets

The internally acquired Imagined Speech cohort comprised adults aged 25 to 29 years. Each trial consisted of 5 s of reading, 4 s of speech imagery and 4 s of fixation, with only the imagery epoch included in the main analysis. Neural activity was recorded using a 64-channel single-axis OPM-MEG system (Pyramag Epoch 64, Quanmag Healthcare) in a semi-enclosed magnetically shielded cylinder. A T1-weighted anatomical image was acquired for each participant using a 3.0 T uMR 790 scanner (United Imaging Healthcare). The protocol was approved by the Ethics Committee of the Shenzhen Institute of Advanced Technology (approval no. SIAT-IRB-251015-H1027), and all participants provided written informed consent. All public datasets used in this study, together with their cohort sizes, recording modalities and analysis assignments, are summarized in Extended Data Table 1. Analysis-specific sample sizes are reported in the corresponding figure legends.

## Data preprocessing

T1-weighted images were processed with FreeSurfer through the ABCD-HCP workflow, which reconstructed native white and pial surfaces and derived the midthickness surface <sup>52</sup>. Each surface was registered and resampled to the standard 32k fs LR mesh. The participant-specific cortical region of interest mask was retained. Participant surfaces supported individual eigenmode construction. The group surface was used for datasets without participant anatomical images.

fMRI data were processed with the ABCD-HCP BIDS pipeline <sup>52</sup>, an extension of the Human

Connectome Project minimal workflow. Processing comprised denoising, spatial registration, cortical reconstruction and CIFTI generation. Motion and distortion correction followed the information supplied by each dataset. Datasets without field maps entered the workflow without fieldmap correction. Nuisance regression and motion censoring were performed before cortical projection. The final model input was the cortical CIFTI time series on the standard 32k fs LR mesh. Event windows followed the source annotations. A 6 s hemodynamic ofset aligned task events to the blood oxygen level dependent response. Image-encoding analyses used GLM beta maps estimated from the preprocessed time series.

EEG datasets released only as preprocessed epochs were analyzed using the preprocessing provided with the original release. For datasets distributed as continuous recordings, signals were notch filtered at the dataset-specific mains frequency, band-pass filtered between 0.1 and 50 Hz, and resampled to 256 Hz. Independent component analysis was applied where continuous recordings and suficient channel coverage were available to remove ocular, cardiac and stereotyped myogenic components. Task epochs were defined from the event annotations provided with each dataset. Electrode coordinates were retained for forward modeling.

MEG recordings were notch filtered at the dataset-specific mains frequency, band-pass filtered between 0.1 and 50 Hz, and resampled to 256 Hz. Sensor geometry was retained for forward modeling. For OPM-MEG, sensor positions and orientations were co-registered to the individual anatomical image before source modeling.

fNIRS recordings were processed according to the form in which each dataset was released. Raw intensity data were screened for invalid measurements, converted to optical density, corrected for motion artifacts and band-pass filtered before conversion to hemoglobin concentration changes using the modified Beer–Lambert law. Trial-wise baseline correction was applied where required. Datasets released as processed hemoglobin time series retained the provided preprocessing. Channel correspondence was harmonized across runs before analysis.

ECoG recordings were screened for discontinuities and non-finite samples before filtering. Line noise at the dataset-specific mains frequency and its harmonics was removed, after which signals were band-pass filtered and resampled to 256 Hz. The reference scheme provided with the source release was retained where applicable. Electrode coordinates supplied with each dataset were used for cortical-surface sampling.

## Model comparisons

EEG and MEG comparisons used LaBraM, Brain-Omni and CBraMod with their released pretrained weights <sup>23–25</sup>. A task-specific linear head was attached to each backbone. Within-participant comparisons used identical dataset partitions. Zero-shot cross-participant training excluded every sample from the evaluated participant. Crossparticipant evaluation froze each pretrained backbone and fitted only its classification head.

Cross-session ECoG and EEG comparisons also included SPaRCNet <sup>53</sup>. Every method was trained on the same reference session and evaluated on the same subsequent-session trials. Internal-state fMRI analyses compared the proposed framework with NeuroSTORM<sup>36</sup>.

Language decoding was compared with CogReader using the same story boundaries <sup>31</sup>. A random control replaced the neural soft prompt with independent standard-normal tokens. The frozen language model and decoding procedure remained unchanged. Image retrieval was compared with NCC on identical NSD candidate sets and test trials within each participant <sup>33</sup>. The fNIRS Transformer comparison used the same within-participant fivefold partitions as the proposed framework<sup>54</sup>.

Shared cohorts used identical trial definitions, label vocabularies and test samples across methods. Each released model retained its native backbone and input representation. Data partitions, optimization budgets and evaluation metrics were fixed before test evaluation.

## Quantitative analyses

Evaluation protocol. Training and test data were separated at the analysis-specific grouping level. Within-participant classification used runblocked folds for datasets with multiple runs and trial-blocked folds otherwise. Zero-shot crossparticipant evaluation used disjoint participant sets. Language datasets used a 9:1 story-level split shared across participants, with nine parts for training and one for testing. Held-out stories were excluded from model fitting and used only for final generation and scoring. Visual decoding analyses used separate stimuli for model fitting and evaluation. For cross-session evaluation, models were trained on the reference session and applied to all subsequent sessions without refitting.

Cortical semantic attribution. The semantic-attribution maps in Fig. 2e were computed on independent SMN4Lang test data. Attribution for the neural observation representation (low-level) was defined as the square root of non-negative cross-validated vertex-tosemantic-embedding $R ^ { 2 }$ after surface smoothing. Attribution for the meta-neural semantic representation (high-level) was obtained from cross-validated eigenmode-to-semantic-embedding $R ^ { 2 }$ , with squared eigenmode amplitudes used to project these values onto the cortical surface. The two attribution maps were independently rank-normalized for visualization. The displayed percentages indicate the proportion of cortical surface vertices represented in each attribution map.

Surface-searchlight RSA compared cortical coverage of low-level and high-level reference geometries in eight NSD participants, using 907–1,000 Shared1000 images per participant and repetitionaveraged fMRI responses at 59,412 fsLR32k cortical vertices. Searchlights included vertices within 10 mm along template midthickness mesh edges, within the same hemisphere and excluding the medial wall. Local brain RDMs contained pairwise correlation distances (1 − r, Pearson correlation) between image-response patterns.

Reference RDMs used 768-dimensional CLIP ViT-L/14 embeddings or 1,024-dimensional metaneural semantic representations predicted by a participant-specific image-to-representation bridge. Its 2,944 inputs combined CLIP ViT-L/14 embeddings, mean-pooled ViT-bigG/14 tokens and 512 training-fitted ConvNeXt principal components. A 2,944–512–1,024 network with GELU and dropout 0.1 predicted fused posterior means inferred from non-test fMRI. Training used mean-squared error, AdamW (learning rate $3 \times 1 0 ^ { - 4 }$ , weight decay 0.01, batch size 512) and a 90:10 fitting–validation split of non-shared images. Validation RDM correlation selected the checkpoint over at most 120 epochs, with checks every five epochs and stopping after 25 epochs without improvement. The configuration was selected on sub-02 validation data and fixed across participants; Shared1000 data were excluded from fitting and validation.

Searchlight RSA correlated the upper triangles of brain and reference RDMs using Spearman correlation. Exact two-sided sign-flip tests across participants (256 assignments) were followed by Benjamini–Hochberg correction jointly over both references and all vertices. Coverage comprised vertices with mean $\rho > 0 . 0 1$ and $q < 0 . 0 5$ , weighted by template vertex area. Paired high-minus-low RSA diferences were tested similarly, with correction across vertices as a separate family.

The contribution of each eigenmode band or latent token was quantified by the decrease in decoding performance after occlusion. For eigenmode band $g$ and latent token $m ,$ the occlusion

efects were

$$
\delta _ { g } = s _ { \mathrm { f u l l } } - s _ { g } ^ { \mathrm { o c c } } , \qquad \delta _ { m } = s _ { \mathrm { f u l l } } - s _ { m } ^ { \mathrm { o c c } } ,\tag{12}
$$

where $s _ { \mathrm { f u l l } }$ denotes the score obtained with the complete representation, and $s _ { g } ^ { \mathrm { o c c } }$ and $s _ { m } ^ { \mathrm { o c c } }$ denote the corresponding scores after occluding eigenmode band g and latent token m, respectively. The score was BGEScore for language decoding and CLIP cosine similarity for visual decoding. Modes were grouped into non-overlapping bands in ascending eigenvalue order. Their representative wavelength was

$$
\ell _ { g } = \frac { 2 \pi } { \sqrt { \bar { \lambda } _ { g } } } .\tag{13}
$$

The value $\bar { \lambda } _ { g }$ is the mean eigenvalue in band $g .$ The final spatial-attention layer supplied the normalized token-to-mode weight $A _ { m k }$ . This weight was projected through the cortical basis as $\widetilde { A } _ { m } ( v ) =$ $\textstyle \sum _ { k } A _ { m k } \phi _ { k } ( v )$ . The participant-level cortical attribution map was

$$
\mathcal { M } ( v ) = \sum _ { m } \operatorname* { m a x } ( \delta _ { m } , 0 ) | \widetilde { A } _ { m } ( v ) | .\tag{14}
$$

The resulting attribution map was non-negative by construction and was normalized before group aggregation.

Cortical attribution maps were averaged within the Schaefer atlas <sup>55</sup>. Parcels were assigned to the seven Yeo networks <sup>56</sup>. Inspired by activityflow mapping <sup>57</sup>, we estimated participant-specific functional-connectivity weights from SMN4Lang task-fMRI using pairwise Pearson correlations between parcel time series. Each attribution map was standardized across parcels, and the diagonal of the resulting connectivity matrix W was set to zero. For target parcel $j ,$ , connectivity-weighted semantic attribution was

$$
\hat { s } _ { j } = \sum _ { \ell \neq j } W _ { \ell j } s _ { \ell } .\tag{15}
$$

Here, $s _ { \ell }$ denotes semantic attribution at source parcel ℓ. The semantic–connectivity alignment coeficient was

$$
\rho = \mathrm { c o r r } _ { j } \left( \hat { s } _ { j } , s _ { j } \right) .\tag{16}
$$

Here, corr<sub>j</sub> denotes Pearson correlation across target parcels $j .$ The same calculation was repeated within each Yeo network. For Fig. 2g, alignment coeficients for the proposed framework and CogReader were compared descriptively. For Fig. 2h, interregional relations were displayed for both methods. Cross-network relations joined parcels assigned to diferent Yeo networks. Relations were classified as long-distance when their parcel-centroid geodesic distance exceeded the 75th percentile.

Representational geometry analyses. Analyses crossed two cognitive states with two cognitive conditions: rest versus right-hand imagery with versus without EEG neurofeedback; inanimate versus animate images across small versus large real-world sizes; and low versus high engagement across low versus high sadness, defined by participant-specific rating medians.

Three-dimensional MDS visualizations used centroids of the four state–condition combinations from centered, isotropically scaled representations. Projection onto an orthonormal basis spanning their state- and condition-diference vectors preserved pairwise Euclidean distances; coordinates were normalized by mean inter-centroid distance. Each representation was embedded separately, with quantitative metrics computed in full feature space.

CCGP used L<sub>2</sub>-regularized logistic regression $( C = 0 . 0 1 )$ to classify states in one cognitive condition after training in the other. Feature standardization was fitted on the training condition. Accuracy was averaged over both transfer directions and 20 resamples, balancing all four state–condition combinations without replacement. PS was the cosine similarity between the two condition-specific state-diference vectors, each obtained by subtracting one state centroid from the other in a consistent order. High-minus-low diferences in participantlevel CCGP and PS were tested using exact twosided paired sign flips of the mean diference.

Representational structure and crosssession stability. Frozen representations were standardized before t-SNE. Principal-component reduction was performed before the twodimensional embedding. For Fig. 4h–i, t-SNE was performed with 50 random initializations. A silhouette coeficient was calculated for each trial and averaged across trials. For Fig. 4i, the same coeficient was calculated after balanced sampling of ten trials per class within each ECoG Speech session. Linear-probe accuracy was used to quantify class separability.

Cosine similarity measured direct alignment between neural and semantic features. Representational similarity analysis used $d _ { i j } = 1 \mathrm { - c o r r } ( h _ { i } , h _ { j } )$ 2 where $h _ { i }$ is the representation of item i and corr denotes Pearson correlation. The RSA coefficient was the Spearman correlation between the upper triangles of two dissimilarity matrices. Group analyses used Fisher-transformed coeficients. Normalized representational-amplitude profiles, crosssession similarity matrices and population-vector displacement quantified cross-session stability. ECoG Speech models were trained on the reference session and applied to every follow-up session. AJILE12 ECoG and SHU-MI used the same reference-session protocol.

Internal cognitive state analyses. Imagined Speech analyses used only the 4 s OPM-MEG imagery epoch. Semantic correlation was quantified using category-level RSA. Trial-level neural representations were averaged within each category to obtain neural centroids. BGE embeddings of the corresponding sentences were averaged within the same categories to obtain semantic centroids. Pairwise distances were calculated separately among neural and semantic centroids, and the Spearman correlation between the corresponding uppertriangular distance values yielded the RSA coeficient. A classifier fitted to imagery epochs yielded content accuracy. DuPre2016 and Lee2021 used their released task labels for self-generated condition decoding. Their trial-level clarity and vividness scores supplied continuous targets. Valence supplied a further continuous target.

Spacetop Alignvideo analyses in Fig. 6h–j included participants with complete ratings across all five dimensions for the common set of videos. A separate regression model was fitted for each dimension. Direct performance was the withinparticipant correlation between predicted and observed ratings. Video-specific consensus was estimated from training participants and subtracted from both the test-participant rating and the model prediction. The correlation between these residuals quantified participant-specific interpretation. Participant-level correlations were Fisher transformed for inference.

The RSA analyses in Fig. 6m–n used participants with complete ratings across all five dimensions for the same set of videos. The dimensions were happy, sad, afraid, disgusted and engaged. The item index i identifies a video in this analysis. Rating vectors were standardized using parameters estimated from the training participants and are denoted $\widetilde { y } _ { s i }$ . The analyses compared timepooled representations from NeuroSTORM and the proposed framework.

For the between-participant analysis, each method yielded a representation matrix for each video, with one row per participant. Neural distance between participants $s _ { 1 }$ and $s _ { 2 }$ for video i was

$$
d _ { i } ^ { ( z ) } ( s _ { 1 } , s _ { 2 } ) = 1 - \operatorname { c o r r } ( z _ { s _ { 1 } i } , z _ { s _ { 2 } i } ) .\tag{17}
$$

Rating distance for the same participant pair was

$$
d _ { i } ^ { ( y ) } ( s _ { 1 } , s _ { 2 } ) = \lVert \widetilde { y } _ { s _ { 1 } i } - \widetilde { y } _ { s _ { 2 } i } \rVert _ { 2 } .\tag{18}
$$

Each video yielded participant-pair distances of each type. Fractional ranks were calculated over participant pairs within each video and averaged over videos:

$$
\begin{array} { c } { \displaystyle \overline { { \mathcal { R } } } ^ { ( \xi ) } ( s _ { 1 } , s _ { 2 } ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathcal { R } _ { i } \Big ( d _ { i } ^ { ( \xi ) } ( s _ { 1 } , s _ { 2 } ) \Big ) , } \\ { \xi \in \{ z , y \} . } \end{array}\tag{19}
$$

Here, N denotes the number of videos in this analysis. The operator $\mathcal { R } _ { i }$ maps the participant-pair distances for video i to fractional ranks between zero and one. The reported coeficient was the Pearson correlation between the upper-triangular entries of the two participant-pair fractional-rank matrices. This analysis quantified the association between neural representational distance and rating distance.

For the within-participant analysis, each participant yielded neural and rating dissimilarity matrices across videos. Their upper-triangular entries yielded one Pearson coeficient $r _ { s } .$ . Coeficients were Fisher transformed as $\zeta _ { s } = \mathrm { a r c t a n h } ( r _ { s } )$ . The transformed values were averaged across participants. The hyperbolic tangent returned the group coeficient to the correlation scale. The participant was the analysis unit. Group diferences used a twotailed paired t-test on Fisher-transformed values.

For both RSA analyses, scaling parameters and model selection were estimated from the training data and held fixed during evaluation. Reported coeficients were then calculated on the corresponding held-out observations.

For Extended Data Fig. 4, Spacetop participants who had ratings for both tasks were included. The tasks were Narratives and Shortvideo. RSA compared meta-neural semantic dissimilarity with rating dissimilarity across the shared stimuli. The participant was the unit of analysis.

Structural controls and robustness. Equaldimensional Fourier and spherical-harmonic bases served as structural controls. Graph-Laplacian and cortical-eigenmode bases completed the comparison. Every basis used the same encoder and classifier. Data partitions were also fixed. Mode-number analyses changed only the number of retained basis functions. Group-template and participant-specific bases were evaluated on identical trials.

Robustness was assessed by adding zero-mean Gaussian noise scaled by the training-fold channel standard deviation, masking nested proportions of channels and restricting the training set to nested fractions of trials. At each perturbation level, BrainOmni and the proposed model received the same perturbation and sample subset. A separate cross-participant analysis progressively added training data from the target participant while holding the test set fixed. These analyses are reported in Extended Data Fig. 8.

Performance metrics and statistical analysis. Discriminative decoding was summarized by accuracy. Language generation was evaluated using sentence-level BGE-M3 embedding cosine similarity <sup>30</sup>, termed BGEScore. Image reconstruction used CLIP image-embedding cosine similarity. Retrieval reported top-1, top-5 and top-10 accuracy in a 100-way task. For each test image, the candidate set comprised the target image and 99 randomly selected non-target images from the 1,000-image test set.

Semantic encoding of brain activity was evaluated on independent test stimuli. Vertex-wise prediction correlation was $r _ { v } = \mathrm { c o r r } ( \lbrace \hat { b } _ { i , v } \rbrace _ { i } , \lbrace b _ { i , v } \rbrace _ { i } )$ Let $n _ { v }$ denote the repeated-response noise ceiling at vertex v. The noise-ceiling-normalized squared prediction correlation was

$$
r _ { \mathrm { n o r m } , v } ^ { 2 } = { \frac { r _ { v } ^ { 2 } } { n _ { v } } } .\tag{20}
$$

This measure uses squared test-set Pearson correlation, rather than residual-sum-of-squares $R ^ { 2 }$ Vertices with $n _ { v } \leq 0$ were excluded. Encoding coverage was the proportion of vertices with positive prediction correlation and a Benjamini–Hochberg adjusted $\begin{array} { r l r } { P } & { { } < } & { 0 . 0 5 } \end{array}$ . Vertex-wise null distributions used $1 0 ^ { 4 }$ permutations of stimulus identity. Semantic-attribution maps and network edges used the same number of permutations. Spatial correspondence, network concentration and coverage diferences used $1 0 ^ { 4 }$ spherical rotations. Rotations were performed separately in each hemisphere and preserved the medial wall. Empirical probabilities used the standard add-one correction.

All hypothesis tests were two-sided. Comparisons among matched methods or structural bases used one-way repeated-measures ANOVA. Each participant provided one matched row of measurements. Analyses involving method and a second repeated factor used two-way repeatedmeasures ANOVA. Sphericity was not assumed, and the Greenhouse–Geisser correction was applied where appropriate. Pairwise contrasts used Sid´ak- <sup>ˇ</sup> adjusted multiple comparisons. Direct comparisons between two matched conditions used two-tailed paired t-tests. Comparisons between languagedecoding outputs and the independent Random control used two-tailed Welch’s t-tests.

Line plots show the mean and s.e.m. when specified. The participant was the biological replicate. Session-level analyses treated the session as the experimental unit only when explicitly stated. Repeated t-SNE initializations and matched random-seed model fits quantified optimization stability. Inferential tests used the participant, session, initialization or random-seed sampling units specified in the corresponding figure legends.

## References

[1] Gonzalez, W. G., Zhang, H., Harutyunyan, A. & Lois, C. Persistence of neuronal representations through time and damage in the hippocampus. Science 365, 821–825 (2019).

[2] Gallego, J. A., Perich, M. G., Chowdhury, R. H., Solla, S. A. & Miller, L. E. Longterm stability of cortical population dynamics underlying consistent behavior. Nat. Neurosci. 23, 260–270 (2020).

[3] Tyree, T. J., Metke, M. & Miller, C. T. Cross-modal representation of identity in the primate hippocampus. Science 382, 417–423 (2023).

[4] Haynes, J.-D. & Rees, G. Decoding mental states from brain activity in humans. Nat. Rev. Neurosci. 7, 523–534 (2006).

[5] Poldrack, R. A. Inferring mental states from neuroimaging data: From reverse inference to large-scale decoding. Neuron 72, 692–697 (2011).

[6] Kragel, P. A., Koban, L., Barrett, L. F. & Wager, T. D. Representation, pattern information, and brain signatures: from neurons to neuroimaging. Neuron 99, 257–273 (2018).

[7] Mathis, M. W., Perez Rotondo, A., Chang, E. F., Tolias, A. S. & Mathis, A. Decoding the brain: From neural representations to mechanistic models. Cell 187, 5814–5832 (2024).

[8] Peelen, M. V. & Downing, P. E. Testing cognitive theories with multivariate pattern analysis of neuroimaging data. Nat. Hum. Behav. 7, 1430–1441 (2023).

[9] Dijksterhuis, D. E. et al. Pronouns reactivate conceptual representations in human hippocampal neurons. Science 385, 1478–1484 (2024).

[10] Liu, C., Todorova, R., Tang, W., Oliva, A. & Fernandez-Ruiz, A. Associative and predictive hippocampal codes support memory-guided behaviors. Science 382, eadi8237 (2023).

[11] Knill, D. C. & Pouget, A. The Bayesian brain: the role of uncertainty in neural coding and computation. Trends Neurosci. 27, 712–719 (2004).

[12] K¨ording, K. P. & Wolpert, D. M. Bayesian integration in sensorimotor learning. Nature 427, 244–247 (2004).

[13] Friston, K. The free-energy principle: a unified brain theory? Nat. Rev. Neurosci. 11, 127–138 (2010).

[14] Berkes, P., Orb´an, G., Lengyel, M. & Fiser, J. Spontaneous cortical activity reveals hallmarks of an optimal internal model of the environment. Science 331, 83–87 (2011).

[15] Lange, R. D., Shivkumar, S., Chattoraj, A. & Haefner, R. M. Bayesian encoding and decoding as distinct perspectives on neural coding. Nat. Neurosci. 26, 2063–2072 (2023).

[16] Pang, J. C. et al. Geometric constraints on human brain function. Nature 618, 566–574 (2023).

[17] Normand, F. et al. Geometric constraints on the architecture of mammalian cortical connectomes. Cell 189, 5283–5303.e17 (2026).

[18] Atasoy, S., Donnelly, I. & Pearson, J. Human brain networks function in connectomespecific harmonic waves. Nat. Commun. 7, 10340 (2016).

[19] Brunner, C., Leeb, R., M¨uller-Putz, G. R., Schl¨ogl, A. & Pfurtscheller, G. BCI competition 2008—Graz data set A (2008). https: //www.bbci.de/competition/iv/desc 2a.pdf.

[20] Chen, J. et al. A large finer-grained afective computing EEG dataset. Sci. Data 10, 740 (2023).

[21] Liu, W., Qiu, J.-L., Zheng, W.-L. & Lu, B.- L. Comparing recognition performance and robustness of multimodal deep learning models for multimodal emotion recognition. IEEE Trans. Cogn. Dev. Syst. 14, 715–729 (2022).

[22] Lioi, G. et al. Simultaneous EEG-fMRI during a neurofeedback task, a brain imaging dataset for multimodal data integration. Sci. Data 7, 173 (2020).

[23] Jiang, W.-B., Zhao, L.-M. & Lu, B.-L. Large brain model for learning generic representations with tremendous EEG data in BCI. The Twelfth International Conference on Learning Representations 35429– 35450 (2024). URL https://openreview.net/ forum?id=QzTpTRVtrP.

[24] Xiao, Q. et al. BrainOmni: A brain foundation model for unified EEG and MEG signals. Advances in Neural Information Processing Systems 38, 46081–46114 (2025).

[25] Wang, J. et al. CBraMod: A criss-cross brain foundation model for EEG decoding. The Thirteenth International Conference on Learning Representations 62056– 62092 (2025). URL https://openreview.net/ forum?id=NPNUHgHF2w.

[26] Bhattasali, S., Brennan, J., Luh, W.-M., Franzluebbers, B. & Hale, J. The Alice Datasets: fMRI & EEG observations of natural language comprehension. Proceedings of the Twelfth Language Resources and Evaluation Conference 120–125 (2020). URL https:// aclanthology.org/2020.lrec-1.15/.

[27] Li, J. et al. Le Petit Prince multilingual naturalistic fMRI corpus. Sci. Data 9, 530 (2022).

[28] Hanke, M. et al. A high-resolution 7-Tesla fMRI dataset from complex natural stimulation with an audio movie. Sci. Data 1, 140003 (2014).

[29] Momenian, M. et al. Le Petit Prince Hong Kong (LPPHK): Naturalistic fMRI and EEG data from older Cantonese speakers. Sci. Data 11, 992 (2024).

[30] Chen, J. et al. M3-Embedding: Multi-linguality, multi-functionality, multigranularity text embeddings through self-knowledge distillation. Findings of the Association for Computational Linguistics: ACL 2024 2318–2335 (2024). URL https: //aclanthology.org/2024.findings-acl.137/.

[31] Lu, W. et al. Brain-inspired fMRI-to-text decoding via incremental and wrap-up language modeling. Advances in Neural Information Processing Systems 38, 166540–166563 (2025).

[32] Allen, E. J. et al. A massive 7T fMRI dataset to bridge cognitive neuroscience and artificial intelligence. Nat. Neurosci. 25, 116–126 (2022).

[33] Wang, H. et al. Inter-individual and inter-site neural code conversion without shared stimuli. Nat. Comput. Sci. 5, 534–546 (2025).

[34] DuPre, E., Luh, W.-M. & Spreng, R. N. Multi-echo fMRI replication sample of autobiographical memory, prospection and theory of mind reasoning tasks. Sci. Data 3, 160116 (2016).

[35] Lee, S., Parthasarathi, T. & Kable, J. W. The ventral and dorsal default mode networks are dissociably modulated by the vividness and valence of imagined events. J. Neurosci. 41, 5243–5250 (2021).

[36] Wang, C. et al. Towards a general-purpose foundation model for functional MRI analysis. Nat. Biomed. Eng. (2026). https://doi.org/10. 1038/s41551-026-01666-y.

[37] Jung, H. et al. Spacetop: A multimodal fMRI dataset unifying naturalistic processes with a rich array of experimental tasks. Sci. Data 12, 1465 (2025).

[38] Degenhart, A. D. et al. Stabilization of a brain–computer interface via the alignment of low-dimensional spaces of neural activity. Nat. Biomed. Eng. 4, 672–685 (2020).

[39] Sussillo, D., Stavisky, S. D., Kao, J. C., Ryu, S. I. & Shenoy, K. V. Making brain–machine interfaces robust to future neural variability. Nat. Commun. 7, 13749 (2016).

[40] Saez, I. The emerging field of cognitive brain– computer interfaces. Trends Cogn. Sci. (2026). https://doi.org/10.1016/j.tics.2026.06.012.

[41] Bethlehem, R. A. et al. Brain charts for the human lifespan. Nature 604, 525–533 (2022).

[42] Breakspear, M. Dynamic models of large-scale brain activity. Nat. Neurosci. 20, 340–352 (2017).

[43] Demirta¸s, M. et al. Hierarchical heterogeneity across human cortex shapes large-scale neural

dynamics. Neuron 101, 1181–1194.e13 (2019).

[44] Hinton, G. E. Training products of experts by minimizing contrastive divergence. Neural Computation 14, 1771–1800 (2002).

[45] Glasser, M. F. et al. The minimal preprocessing pipelines for the human connectome project. NeuroImage 80, 105–124 (2013).

[46] Li, J., Li, D., Savarese, S. & Hoi, S. BLIP-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. Proceedings of the 40th International Conference on Machine Learning 202, 19730– 19742 (2023). URL https://proceedings.mlr. press/v202/li23q.html.

[47] Abouelenin, A. et al. Phi-4-Mini technical report: Compact yet powerful multimodal language models via mixture-of-LoRAs. arXiv:2503.01743 [cs.CL] (2025).

[48] Meng, F., Wang, Z. & Zhang, M. PiSSA: Principal singular values and singular vectors adaptation of large language models. Advances in Neural Information Processing Systems 37, 121038–121072 (2024).

[49] Radford, A. et al. Learning transferable visual models from natural language supervision. Proceedings of the 38th International Conference on Machine Learning 139, 8748–8763 (2021). URL https://proceedings.mlr.press/ v139/radford21a.html.

[50] Rombach, R., Blattmann, A., Lorenz, D., Esser, P. & Ommer, B. High-resolution image

synthesis with latent difusion models. 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition 10674–10685 (2022).

[51] Song, J., Meng, C. & Ermon, S. Denoising difusion implicit models. The Ninth International Conference on Learning Representations (2021). URL https://openreview.net/ forum?id=St1giarCHLP.

[52] Feczko, E. et al. Adolescent brain cognitive development (ABCD) community MRI collection and utilities. bioRxiv 2021.07.09.451638 [Preprint] (2021). https://doi.org/10.1101/ 2021.07.09.451638.

[53] Jing, J. et al. Development of expert-level classification of seizures and rhythmic and periodic patterns during EEG interpretation. Neurology 100, e1750–e1762 (2023).

[54] Wang, Z., Zhang, J., Zhang, X., Chen, P. & Wang, B. Transformer model for functional near-infrared spectroscopy classification. IEEE J. Biomed. Health Inform. 26, 2559–2569 (2022).

[55] Schaefer, A. et al. Local–global parcellation of the human cerebral cortex from intrinsic functional connectivity MRI. Cereb. Cortex 28, 3095–3114 (2018).

[56] Yeo, B. T. T. et al. The organization of the human cerebral cortex estimated by intrinsic functional connectivity. J. Neurophysiol. 106, 1125–1165 (2011).

[57] Cole, M. W., Ito, T., Bassett, D. S. & Schultz, D. H. Activity flow over resting-state networks shapes cognitive task activations. Nat. Neurosci. 19, 1718–1726 (2016).

[58] Simistira Liwicki, F. et al. Bimodal electroencephalography–functional magnetic resonance imaging dataset for inner-speech recognition. Sci. Data 10, 378 (2023).

[59] Dubois, J. et al. Reliability of brain metrics derived from a time-domain functional nearinfrared spectroscopy system. Sci. Rep. 14, 17500 (2024).

[60] Duwadi, S. et al. Decoding spatial attention in the cocktail party problem using wearable whole-head high-density fNIRS. bioRxiv 2026.05.06.722322 [Preprint] (2026). https:// doi.org/10.64898/2026.05.06.722322.

[61] Ma, J. et al. A large EEG dataset for studying cross-session variability in motor imagery brain–computer interface. Sci. Data 9, 531 (2022).

[62] Golmohammadi, M., Harati Nejad Torbati, A. H., Lopez de Diego, S., Obeid, I. & Picone, J. Automatic analysis of EEGs using big data and hybrid deep learning architectures. Front. Hum. Neurosci. 13, 76 (2019).

[63] Lin, F.-H. et al. somatomotor. OpenNeuro [Data set] (2025). URL https://doi.org/10. 18112/openneuro.ds006035.v1.0.0.

[64] Wang, S., Zhang, X., Zhang, J. & Zong, C. A synchronized multimodal neuroimaging

dataset for studying brain language processing. Sci. Data 9, 590 (2022).

[65] Nastase, S. A. et al. The “Narratives” fMRI dataset for evaluating models of naturalistic language comprehension. Sci. Data 8, 250 (2021).

[66] Gong, Z. et al. A large-scale fMRI dataset for the visual processing of naturalistic scenes. Sci. Data 10, 559 (2023).

[67] Peterson, S. M. et al. AJILE12: Long-term naturalistic human intracranial neural recordings and pose. Sci. Data 9, 184 (2022).

[68] Angrick, M. et al. Online speech synthesis using a chronically implanted brain–computer interface in an individual with ALS. Sci. Rep. 14, 9617 (2024).

LOW

![](images/fd929f77b5cc9ef74eee505567b1cba104cd021c0b00236360c63fcdcb2062cb.jpg)

sub-01

HIGH

![](images/ac1fe99fc3f82a3cf5f2f33a3d3447f4977b4d12ec8fae8c49a20019c41314bf.jpg)

LOW

![](images/10244ec4d289e03baba72a8d74ad20af316f7435b5addd42fd446a644c955c20.jpg)

sub-02

HIGH

![](images/3d232f7c7e754ae9eaee9204200d933da79519a82317f8dc78a1879b8e3fcefc.jpg)

![](images/819c1124e14332b33485038a297d14607f8734249423fe07299ceed74e6551b0.jpg)

![](images/6398e41515c322726772a79ff662b1146068592456c88ab344807f05ef544ddc.jpg)

sub-04

![](images/2b1d71c036e32945e261f02c9142b87dd6d010106183be6659258701b380b486.jpg)

![](images/398bbca2ce32595fc228f4f5e02cbf452d09a940705752642bad3d689b863f91.jpg)

![](images/59d1d385c4285a3d12205ca9b01c111fecb8d9725b82220e1f26ac6acd69f0cb.jpg)

![](images/fb82a6e0e4cd9259d6aad03010048fd1664017c6b8f9f9c0a2b59f4c7bd271e8.jpg)

![](images/d322db715c69cacd97303918909a098233aa86f84731fcf9b288b0199c00723e.jpg)

sub-06

![](images/326478eb6b337b42e985917384af80dac34072581b5a0249f22634c13017c5b6.jpg)

![](images/f1098d4437912373a75dc61a131583aecc3636038ae695036c7e43eeb8fde428.jpg)

![](images/2a2898e0539f4ebcc96439da50ccc74f5d66cc576e731199b9756b2f5c75d860.jpg)

![](images/6b59af2220e4e36f9d062b0bd9bf61648293c3a5d6c17562ee48b3f916856e59.jpg)

![](images/7961cc4f6772c7e895277d327ce80787d4f4e86c42c8c64c432cd9a4ca7774cf.jpg)

![](images/2ffd5d8ec44fc1a65061ec02e868a613a12fe2ae534d1caf82ea1e211494a71e.jpg)

![](images/ded8bacce9e863b88d24df22e115c97d6bba275e02d3ffa8f6aef4cce03e9082.jpg)

![](images/deb6aed9aa3d2e76ff6f4db1db5dce6c97bafc398ca5bd4f9903714481d1d56b.jpg)

![](images/32c55f17a5c2be191911aff14c6c3405b48e552b66726ea6e4e20645e8df926a.jpg)

sub-11

![](images/e0ba1dbd64e0d9037c36e4f0b089a2d87b76f60f688b5510dafe1bb5829b8d3e.jpg)

![](images/5d0ec40b5efd708475d16cfd7d469765a92b817ae7bea1c47fb38ab6df0e81a7.jpg)

sub-12

![](images/2fbd80b0a109d96aeb0384fbb29e882f64f3fc3841df9cef43bce2db77b20a57.jpg)

![](images/c324cf0747dce0c1d5fdd7c1c7b05759700ad0b1cc7146acb9cd1a8751324a90.jpg)

Extended Data Fig. 1 Semantic-attribution maps across participants. Maps of the low-level neural observation representation and high-level meta-neural semantic representation are shown for 12 SMN4Lang participants. Percentages indicate cortical coverage.

![](images/29f5bc5e384fa01522110b0de736f6346eeecc0b72d04ec8f3319167dc6c5e7a.jpg)

![](images/35a9f474810202192e2b33f8c1c7260e018a9c764f4e9ed5cda79b806dea8808.jpg)

b  
![](images/69109fe906fdcebc9aeaccd33edac3613e8c9adbf3adfc5ccd3ab16e07bbaae0.jpg)

![](images/e561dde8ce2d063ec11a649862528b0faf1bcc211164e002c96732264f194827.jpg)  
Extended Data Fig. 2 Subjective-state prediction across Spacetop tasks. a, Prediction–rating correlations for expectation intensity, expectation valence, feeling intensity and feeling valence in Spacetop Narratives, and for likeability, mentalizing and similarity in Spacetop Shortvideo. Each target included $n = 3 0$ participants. b, Prediction of participant-specific rating residuals after removal of the shared stimulus-locked component. Green denotes NeuroSTORM and red denotes the proposed method. Box plots show the median (center line), 25th and 75th percentiles (box limits), and $1 . 5 \times \mathrm { I Q R }$ whiskers; values outside the whiskers are shown individually. Violin plots show participant-level distributions. Correlation coeficients were Fisher z-transformed before statistical testing; raw Pearson r values are shown. Two-way repeated-measures ANOVA followed by Sid´ak-adjusted multi-<sup>ˇ</sup> ple comparisons was used in each task, with comparisons between methods at each rating dimension. Asterisks denote statistical significance $( ^ { * } P < 0 . 0 5$ , \*\*P < 0.01, $^ { * * * } P < 0 . 0 0 1$ and $^ { * * * * P } _ { } < 0 . 0 0 0 1 )$

a  
![](images/8a9e77efd51f4b26cbd4450afa9ce4d7aa7dbca9a6d70ad151f2ae7b49cb59b1.jpg)

![](images/72f0d90b7fb4d729559b7752b0ac3d71b185bfc0a19732e827e4bafa5d8ae294.jpg)

![](images/f7dc0dd17718f609970c9c7e25f9af77ac2704af2e683c0b85fefb09801e353f.jpg)

b  
![](images/7a35d21e76d224f37e8ef09c825325499da75dd4de9d53ff5dc99ceee0d0936b.jpg)

![](images/c6739b449d0f516bfb2f60114f763ede12fe32398eba999b3c295a82551be405.jpg)

![](images/bbd8f48034916ceb91d40f730f120e9d3dfda7fb059b6a6a88fddf83f44ffda5.jpg)

![](images/ce1c099321ddf25163c94cdf1be1a46bbeb50602dbd88c4a03bebdf8a6d815d3.jpg)

![](images/503726e5f1783d2f6d36630a3a3babf4ff8d54db279d1bddf58aafddfe02e6b8.jpg)

![](images/9cc281a1597764f1bc92c7a370d587365787a913f29b13a57a9e207c0601f2b3.jpg)  
Extended Data Fig. 3 Representational geometry of low-level and high-level representations across cognitive categories. ${ \mathbf { a } } ,$ Multidimensional-scaling visualizations for motor, perception and internalmentation categories in the low-level neural observation and high-level meta-neural semantic representations. $\mathbf { b } ,$ Within-class dispersion, between-class separation and category-structure representational similarity analysis (RSA) for the two representations. Violin plots show distributions, with internal lines indicating medians and quartiles. Asterisks denote statistical significance $( ^ { * } P < 0 . 0 5 , ^ { * * } P < 0 . 0 1 , ^ { * * * } P < 0 . 0 0 1$ and $^ { * * * * } P < 0 . 0 0 0 1 )$ .

a  
![](images/96e0fe8d672315d53a48e18ff6af46377153d819e5d33d7ae8a6b7906d2c0330.jpg)

![](images/47c174c1970a8925488565c92dd2f8abf91ae06f64f60d70d6c6c7bf8312813f.jpg)

b  
![](images/84ed9a31eaa1d9d3946e306bd26a59341f5bb01595efd533dd1a2b39093f5bbb.jpg)

c  
![](images/b1e88d076b5f33b10a3604b9d78c6bd2dbfaba3fa29d2cffda16ea74404b4774.jpg)

![](images/22fd945ae8e6512746f404499e07f04d5caadbb0312a698c9418627c50fded14.jpg)

d  
![](images/cf0413f2886776f51bce7a3f5302f93256e5dae2d628f50270deceb627682265.jpg)  
Extended Data Fig. 4 Neural representations relate to subjective ratings in Spacetop. a, Association between neural representational distance and rating distance across 136 participant pairs formed from $n = 1 7$ participants in Narratives. b, Participant-level RSA correlations between neural representations and Narratives ratings. c, Corresponding participant-pair analysis for Shortvideo ratings. d, Participant-level RSA correlations for Shortvideo ratings. Green denotes NeuroSTORM and red denotes the proposed framework. Points in a and c denote participant pairs, and lines show linear fits with Pearson correlation coeficients. Violin plots in b and d show participant-level distributions; internal lines indicate the median and quartiles. Two-tailed paired t-tests on Fisher z-transformed participant-level RSA coeficients were used in b and d. Asterisks denote statistical significance $( ^ { * } P < 0 . 0 5 , ^ { * * } P < 0 . 0 1 , ^ { * * * } P < 0 . 0 0 1$ and $^ { * * * * P } _ { } < 0 . 0 0 0 1 )$ .

a  
![](images/5b768528591e193365511f4f3c763fddef621d180661d7f4ffdb3d293bfa3896.jpg)  
b

![](images/c6606c8a007868fd63aedcf89db4fb67aa1e187a8cba7701124e28ad8cf498e8.jpg)

![](images/0793a486bb9f7504e3aa8edbb5bbc8c86991f4c2d4f0f237f589b52aa348e792.jpg)

![](images/9d69f4966b14347a063484832575a441a8c009c3139db1a82c576bfc5304982e.jpg)

![](images/d51acfa15a32376550de983900f204ff37beed560e12c95e6f167b4a3522c824.jpg)

![](images/88c7f6a9a30d4ccc3bc6ccfef87f6259e4acf6f41c467da45fb305963325bbcb.jpg)

![](images/0bcad38fa7893ffae08caa3308778e9f76df09fb7408507b4c2269693d80250e.jpg)

![](images/5ee4837827fbbe01027d3d88e0bad4e72d5709dcc3b4d8f5a1fe1f782fc0d150.jpg)

c  
![](images/83527e4c3e609ff37febbe2373d55f19498dbdf64c2327bbcfaa5b55264a5095.jpg)

![](images/ab0440f79a3ba1397feb50a053c236c45f07f995ec0e59e9dd707e668282e5b9.jpg)

![](images/4d4154185993c1c0bafd4a022a58aadf85474798fe008980a4b81fad77d410c7.jpg)

![](images/231a308672936762aeeab6c0514d2608fd936b0b9dcc862ef0f066b65448c4ff.jpg)  
Extended Data Fig. 5 Ablation analyses of cortical geometric eigenmodes. a, Within-participant decoding accuracy using Fourier, spherical-harmonic, graph-Laplacian and cortical geometric eigenmode bases. b, Cross-participant decoding accuracy for the same bases. BCIC included $n = 9$ participants, FACED $n = 1 2 3 ,$ SEED-V $n = 1 6$ and Motor Imagery $n = 1 0 . \ \mathbf { c } ,$ Decoding accuracy as a function of the number of retained modes. Red boxes mark the dimensionalities used in the main analyses: 14, 20, 30 and 28 modes. Box plots show the median (center line), 25th and 75th percentiles (box limits), and $1 . 5 \ \times \ \mathrm { I Q R }$ whiskers; values outside the whiskers are shown individually. One-way repeated-measures ANOVA with Greenhouse–Geisser correction followed by Sid´ak-adjusted multiple comparisons was used in<sup>ˇ</sup> a and b. Asterisks denote statistical significance $( ^ { * } P < 0 . 0 5 , ^ { * * } P < 0 . 0 1 , ^ { * * * } P < 0 . 0 0 1$ and $^ { * * * * } P < 0 . 0 0 0 1 )$

![](images/86d59b84a725807ddd6dd833f5462fa9867ea9f84b85d442a57b8cd2639ec6bc.jpg)

b  
![](images/cda7af06037e1ce16e01e2019e0ac46e94c211ceba2d7eadbe236e01e4e80ced.jpg)

c  
![](images/95d85cde4f9a3e956c56e03a638770bca4efef50ce48af12ad3d7fbc535f497e.jpg)

d  
e  
![](images/4c80a305b33f96c34b9b8ac35774749a695badd01f74ee60183823d01ac7ade8.jpg)

f Imagined Speech  
![](images/49a73990617463fe0addf1b6338c51dd96f24ffffd2eb64e9267b3cc55a90e04.jpg)

![](images/09865939bf158282c884b90ef9bc91d0811153c14303fcd82be5100039b18d29.jpg)

g  
![](images/96e7fed9fe0c0836960efbcfe51a10a7b66730cdff0303662a583740a0bdeeef.jpg)

h  
![](images/d986c6076b6ffc679e15ee9447145560168f65fcbfe74a18709150c5103742b4.jpg)

i  
![](images/112b5fe4d0386c61769e997498f0a60eeec70f1ead7391672f08a49efdf3a9f3.jpg)

![](images/d1ae3577d3b035359a1cc9b65097e7288e972058404586946fad9cea0772f1b2.jpg)

k  
![](images/80dfe010970fbf85d924435f9c019311c2c5ab89291edfe9e76b1d04f77c1073.jpg)  
Extended Data Fig. 6 Individual cortical geometry and decoded features. a, Cortical surfaces and geometric eigenmodes 1, 5, 20, 40, 100, 200, 500 and 1,000 from the HCP S1200 group reference and three representative participants. b–c, Group–individual correspondence across cortical eigenmode ranks. b, Absolute same-rank spatial correlations |r| between HCP S1200 group eigenmodes and individual eigenmodes for one participant from the Alice dataset. c, Group–individual deviations in cortical-area-normalized eigenvalues across four eigenmode-rank bands in 26 Alice participants. For each participant, deviations were averaged within each band, and box plots summarize the participant distributions. Higher-rank eigenmodes showed weaker spatial correspondence but increasingly consistent eigenvalue scales. d–f, Decoding accuracy using group-level and participant-specific cortical geometric eigenmodes in SomatoMotor (two-class; $n = 5 )$ , Motor Imagery (four-class; $n = 1 0 )$ and the Imagined Speech OPM-MEG dataset collected for this study (three-class; $n = 6 )$ , respectively. g–k, BGEScore using group-level and participant-specific cortical geometric eigenmodes in Narratives, Alice, LPPC-fMRI, StudyForrest and LPPC-HK, with n = 37, 26, 112, 20 and 52 participants, respectively. Box plots in c–k show the median (center line), the 25th and 75th percentiles (box limits), and $1 . 5 \times \mathrm { I Q R }$ whiskers; values outside the whiskers are shown individually. Two-tailed paired t-tests were used in d–k. Asterisks denote statis tical significance $( ^ { * } P < 0 . 0 5 , ^ { * * } P < 0 . 0 1 , ^ { * * * } P < 0 . 0 0 1$ and $^ { * * * * } P < 0 . 0 0 0 1 )$

a  
![](images/355ca2e1a7334e5d46dc312d5e26f7fa76f7cb1ecd18464a42f261f9144b3c71.jpg)

b  
![](images/8d9590e2780bb3ca2e008258d22b251da2bc362d70f0700f74c6dcd1381bcce9.jpg)

c  
![](images/31f086976935b7195fa51dc270861a64607da55f78e8ebc0f2defbb2efcea74f.jpg)

d  
![](images/5c10980c4807aa9870fe9325795f2f5cf7da8a263ce3d96433251c7c71697469.jpg)  
Extended Data Fig. 7 Additional discriminative decoding results. a, Inner-speech decoding accuracy across eight classes in Bimodal Inner Speech $( n = 4$ participants) for LaBraM, BrainOmni, CBraMod and the proposed framework <sup>58</sup>. b, Somatomotor decoding accuracy across two classes in SomatoMotor $( n = 5 )$ for the same methods. c, Decoding accuracy across three classes in Dubois2024 $\left( n = 4 9 \right) ^ { 5 9 }$ for the fNIRS Transformer <sup>54</sup> and the proposed framework. d, Two-class decoding accuracy for cocktail-party fNIRS; 30 of the 38 source participants had valid recordings for analysis <sup>60</sup>. Box plots show the median (center line), 25th and 75th percentiles (box limits), and $1 . 5 \times \mathrm { I Q R }$ whiskers; values outside the whiskers are shown individually. Brackets indicate pairwise comparisons. One-way repeated-measures ANOVA with Greenhouse–Geisser correction followed by Sid´ak-adjusted<sup>ˇ</sup> multiple comparisons was used in a and b. Two-sided paired t-tests were used in c and d. Asterisks denote statistical significance $( ^ { * } P < 0 . 0 5 , ^ { * * } P < 0 . 0 1 , ^ { * * * } P < 0 . 0 0 1$ and $^ { * * * * P } < 0 . 0 0 0 1 )$

![](images/4c711448d8f0c6eca969c4a8d62f15703a370452b4440698e1fe7f8a0eaf530d.jpg)  
Extended Data Fig. 8 Robustness of decoding performance. a, Decoding accuracy after adding increasing levels of noise relative to the training-fold channel standard deviation. b, Accuracy after dropping increasing proportions of channels. c, Accuracy as progressively smaller fractions of within-participant training data were used. d, Accuracy in the cross-participant setting as increasing amounts of training data from the target participant were introduced. Each row compares BrainOmni with the proposed framework; columns show BCIC, FACED, SEED-V and Motor Imagery, with $n = 9 ,$ 123, 16 and 10 participants, respectively. Box plots show the median (center line), 25th and 75th percentiles (box limits), and 1.5 × IQR whiskers; values outside the whiskers are shown individually. Brackets indicate pairwise comparisons. Two-way repeated-measures ANOVA followed by Sid´ak-adjusted multiple comparisons was used for each dataset and analysis, with methods compared at each<sup>ˇ</sup> level. Asterisks denote statistical significance (\*P < 0.05, \*\*P < 0.01, \*\*\*P < 0.001 and $^ { * * * * P } _ { } < 0 . 0 0 0 1 )$ .

Extended Data Table 1 Datasets used in this study. N denotes the reported cohort size (participants).
<table><tr><td> $\overline { { \mathrm { N o . } } }$ </td><td>Dataset</td><td>Modality</td><td>N</td><td>Task</td><td>Ref.</td></tr><tr><td>1</td><td>FACED</td><td>EEG</td><td>123</td><td>Affective video viewing</td><td>20</td></tr><tr><td>2</td><td>SEED-V</td><td>EEG</td><td>16</td><td>Affective video viewing</td><td>21</td></tr><tr><td>3</td><td>BCIC-IV-2a</td><td>EEG</td><td>9</td><td>Motor imagery</td><td>19</td></tr><tr><td>4</td><td>SHU-MI</td><td>EEG</td><td>25</td><td>Motor imagery</td><td>61</td></tr><tr><td>5</td><td>TUEV</td><td>EEG</td><td>518</td><td>Clinical EEG event classification</td><td>62</td></tr><tr><td>6</td><td>SomatoMotor</td><td>MEG + EEG</td><td>5</td><td>Somatosensory stimulation</td><td>63</td></tr><tr><td>7</td><td>Duwadi2026</td><td>fNIRS</td><td>38</td><td>Auditory spatial attention</td><td>60</td></tr><tr><td>8</td><td>Dubois2024</td><td>fNIRS</td><td>49</td><td>Auditory and cognitive tasks</td><td>59</td></tr><tr><td>9</td><td>Motor Imagery</td><td>fMRI + EEG</td><td>10</td><td>Motor imagery with neurofeedback</td><td>22</td></tr><tr><td>10</td><td>Bimodal Inner Speech</td><td>fMRI + EEG</td><td>4</td><td>Inner speech production</td><td>58</td></tr><tr><td>11</td><td>SMN4Lang</td><td>fMRI + MEG</td><td>12</td><td>Narrative listening</td><td>64</td></tr><tr><td>12</td><td>Narratives</td><td>fMRI</td><td>345</td><td>Narrative listening</td><td>65</td></tr><tr><td>13</td><td>Alice</td><td>fMRI</td><td>26</td><td>Narrative listening</td><td>26</td></tr><tr><td>14</td><td>LPPC-fMRI</td><td>fMRI</td><td>112</td><td>Narrative listening</td><td>27</td></tr><tr><td>15</td><td>LPPC-HK</td><td>fMRI + EEG</td><td>52</td><td>Narrative listening</td><td>29</td></tr><tr><td>16</td><td>StudyForrest</td><td>fMRI</td><td>20</td><td>Naturalistic movie viewing</td><td>28</td></tr><tr><td>17</td><td>NOD</td><td>fMRI</td><td>30</td><td>Natural image viewing</td><td>66</td></tr><tr><td>18</td><td>NSD</td><td>fMRI</td><td>8</td><td>Natural image viewing</td><td>32</td></tr><tr><td>19</td><td>Spacetop</td><td>fMRI</td><td>101</td><td>Affective and cognitive tasks</td><td>37</td></tr><tr><td>20</td><td>DuPre2016</td><td>fMRI</td><td>31</td><td>Autobiographical cognition</td><td>34</td></tr><tr><td>21</td><td>Lee2021</td><td>fMRI</td><td>26</td><td>Future event imagination</td><td>35</td></tr><tr><td>22</td><td>AJILE12</td><td>ECoG</td><td>12</td><td>Naturalistic upper-limb movement</td><td>67</td></tr><tr><td>23</td><td>ECoG Speech</td><td>ECoG</td><td>1</td><td>Speech production</td><td>68</td></tr></table>