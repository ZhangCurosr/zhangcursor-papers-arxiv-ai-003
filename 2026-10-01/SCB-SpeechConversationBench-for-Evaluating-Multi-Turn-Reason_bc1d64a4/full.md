# SCB: SpeechConversationBench for Evaluating Multi-Turn Reasoning in Speech-to-Speech Model

Kanpat Vesessook<sup>∗</sup> SCBX R&D t\_kanpat.v@scbx.com

Saksorn Ruangtanusak SCB DataX, SCBX Group saksorn.ruangtanusak@data-x.ai

## Abstract

Speech-to-speech systems must solve tasks whose requirements emerge across conversational turns. We introduce SpeechConversationBench (SCB), a focused evaluation of spoken mathematical reasoning using a pool of 103 sharded GSM8K problems. The framework compares the original problem delivered in one turn (FULL), its concatenated information shards delivered together (CONCAT), and incremental spoken disclosure across turns (SHARDED). We report final-answer accuracies for four commercial speech systems and LEGO, a proprietary speech pipeline developed internally by the SCBX Innovation Lab team, with explicit conversational context management. Relative to CONCAT, SHARDED accuracy decreases by 5.0-25.3 percentage points across the four commercial systems. LEGO attains 77.5% in all three conditions, compared with 76.6% SHARDED accuracy for GPT-4o Realtime. Both single-turn baselines are needed to distinguish sensitivity to reformulation from the challenges of incremental interaction.

## 1 Introduction and Related Work

Solving a complete spoken request does not establish that a system can solve the same task when relevant facts arrive over several turns. Mathematical word problems provide a focused test: success requires integrating incrementally disclosed quantities and constraints into a verifiable final answer.

Speech and dialogue evaluation. VoiceBench evaluates voice assistants across content, speaker, and environmental variations [1], while Dynamic-SUPERB organizes diverse instruction-based speech tasks [6]. Multi-turn speech evaluation is also an established research direction. MTalk-Bench combines pairwise and rubric-based assessment across semantic, paralinguistic, and ambient-sound dimensions [4]. Audio MultiChallenge evaluates natural spoken interactions involving memory, instruction retention, self-coherence, and spoken repairs [5]. SpeechConversationBench complements these broader evaluations with a narrower comparison: numerical task completion under three presentations of corresponding problem information. It does not measure their full range of acoustic or interactional capabilities.

Incremental reasoning and memory. We adapt the FULL/CONCAT/SHARDED protocol of Laban et al. [7] to spoken mathematical tasks. LoCoMo instead evaluates memory over extended, multi-session conversations [9]; our setting concerns integrating the facts of one problem within an interaction.

Architectural context. Moshi models full-duplex speech-text dialogue [3], while MemGPT manages information across memory tiers [10]. Context-aware decoding strengthens adherence to spoken history [8]. These approaches motivate our comparison of five speech systems, including the context-managing cascade LEGO; they do not validate LEGO itself.

## 2 Evaluation Framework

## 2.1 Task and Input Conditions

The problem pool contains 103 mathematical word problems from the sharded GSM8K resource of Laban et al. [7]. GSM8K contains grade-school problems requiring multi-step numerical reasoning [2]. A shard is a partial statement of a problem’s information; the shards jointly specify the task. Spoken prompts provide the input, and correctness of the final spoken answer is the evaluation target. Figure 1 summarizes the conditions; Figure 2 shows the original simulator.

In FULL, the original complete problem is presented in one spoken turn. In CONCAT, the corresponding shards are combined and presented together in a single turn. In SHARDED, information is disclosed incrementally through a multi-turn spoken exchange. Thus, FULL and CONCAT differ in formulation, while CONCAT and SHARDED differ in how the shard content is distributed through interaction. This follows the conceptual controls of the original text protocol [7] without assuming that its model settings transfer to speech.

The CONCAT baseline is especially useful because reformulation can itself change difficulty. Comparing SHARDED only with FULL can conflate that change with the effect of incremental disclosure. Nevertheless, CONCAT versus SHARDED is not an isolation of memory alone: assistant responses, speech processing, and the evolving conversational trajectory may also contribute to the observed difference.

![](images/7ea9652d523bcbb339f4189677be9c0b13db87274aeee5bd1192a1addbba1fc2.jpg)  
Figure 1: Spoken evaluation conditions. FULL and CONCAT present information together; SHARDED distributes it across an exchange. Final spoken answers are assessed for numerical correctness.

![](images/49db46a831d014cff85be00cba24f465f81d59bbd7274f3c0c7e9c2146be6067.jpg)  
Figure 2: Sharded-conversation simulator reproduced from Laban et al. [7]. This reference architecture motivates the spoken adaptation; it does not specify our audio-processing or scoring components.

## 2.2 Systems and Outcome Measures

We evaluate GPT-4o Realtime, GPT-4o Mini Realtime, Gemini 2.5 Flash Live, and Gemini 2.5 Flash Preview Native Audio Dialog, alongside LEGO. These names identify the systems in the reported comparison; they do not establish exact API snapshots. LEGO is a proprietary pipeline developed internally by the SCBX Innovation Lab team. It combines automatic speech recognition (ASR), a large language model (LLM) with explicit context management, and text-to-speech (TTS) synthesis, with all component models self-hosted internally. The pipeline incorporates the Thai semantic end-of-turn detection method described by Popit et al. [11]. Its context mechanism summarizes conversational history and reintroduces relevant information at successive turns. This is a system-level comparison rather than a controlled test of any individual component.

Let $A _ { F } , A _ { C }$ , and $A _ { S }$ denote the reported percentages of correct final answers under FULL, CONCAT, and SHARDED. We summarize the multi-turn contrast by an absolute change and a normalized retention ratio:

$$
\Delta _ { S - C } = A _ { S } - A _ { C } , \qquad R _ { S / C } = 1 0 0 { \frac { A _ { S } } { A _ { C } } } .\tag{1}
$$

The change is measured in percentage points (pp); the ratio expresses SHARDED accuracy as a percentage of CONCAT accuracy. A retention value of 100% means equal aggregate accuracies, not that every problem has the same outcome. All derived quantities use the displayed accuracies and are rounded to one decimal place. The 103-problem pool is distinct from the per-condition trial denominators and repeat counts, which are not specified in the aggregate results. Consequently, we do not infer correct-answer counts or uncertainty intervals from that pool size.

## 3 Results

Table 1 reports all five systems. Among the commercial systems, SHARDED accuracy ranges from 46.6% to 76.6%, and every system scores lower in SHARDED than in CONCAT. The absolute decreases are 5.0 pp for GPT-4o Realtime, 25.3 pp for GPT-4o Mini Realtime, 15.5 pp for Gemini 2.5 Flash Live, and 25.0 pp for Gemini 2.5 Flash Preview Native Audio Dialog. These correspond to retention ratios of 93.9%, 67.4%, 75.0%, and 68.8%, respectively.

Table 1: Final-answer accuracy (%) under each input condition. $\Delta _ { S - C }$ is the SHARDED-minus-CONCAT change in percentage points; $R _ { S / C }$ is aggregate accuracy retention (%). Derived columns use the displayed accuracies. Bold marks the highest SHARDED accuracy without implying significance.
<table><tr><td>System</td><td>FULL</td><td>CONCAT</td><td>SHARDED</td><td> $\Delta { s } _ { - C }$ </td><td> $R _ { S / C }$ </td></tr><tr><td>GPT-4o Realtime</td><td>70.0</td><td>81.6</td><td>76.6</td><td>-5.0</td><td>93.9</td></tr><tr><td>GPT-4o Mini Realtime</td><td>75.5</td><td>77.7</td><td>52.4</td><td>-25.3</td><td>67.4</td></tr><tr><td>Gemini 2.5 Flash Live</td><td>54.4</td><td>62.1</td><td>46.6</td><td>-15.5</td><td>75.0</td></tr><tr><td>Gemini 2.5 Flash Preview†</td><td>85.0</td><td>80.0</td><td>55.0</td><td>-25.0</td><td>68.8</td></tr><tr><td>LEGO</td><td>77.5</td><td>77.5</td><td>77.5</td><td>0.0</td><td>100.0</td></tr></table>

<sup>†</sup>Gemini 2.5 Flash Preview Native Audio Dialog.

The baseline changes the interpretation. GPT-4o Realtime improves from 70.0% in FULL to 76.6% in SHARDED, even though it falls from 81.6% in CONCAT. A FULL-only comparison would therefore conceal its disadvantage relative to the concatenated-shard baseline. More generally, the CONCAT-minus-FULL changes are +11.6, +2.2, +7.7, and −5.0 pp for the four commercial systems, respectively. Both single-turn controls are therefore informative; degradation is not universal across baseline choices.

Single-turn rankings do not carry over. Gemini 2.5 Flash Preview Native Audio Dialog has the highest FULL accuracy, 85.0%, but achieves 55.0% in SHARDED. GPT-4o Realtime has lower FULL accuracy yet the strongest SHARDED result among the commercial systems. Selecting a system by single-turn accuracy can therefore favor a different system than selecting by multi-turn accuracy.

LEGO preserves aggregate accuracy. LEGO obtains 77.5% in all three conditions and the highest observed SHARDED score, exceeding GPT-4o Realtime by 0.9 pp. Its zero aggregate change makes explicit context management worth investigating, but does not show that summarization caused the result. The systems differ in their components, and no memory ablation is available. Equal aggregate scores could also conceal problems that become correct and others that become incorrect. We therefore interpret LEGO as a promising system-level observation, not proof that a cascade is generally superior to native audio modeling.

## 4 Discussion, Limitations, and Conclusion

What the comparison measures. The observed multi-turn gaps do not identify a failure mechanism. A wrong numerical answer can arise from misperception, loss or misuse of earlier information, arithmetic error, or answer rendering. Without intermediate transcripts and matched interventions, these explanations cannot be separated. The table establishes neither forgetting nor monotonic degradation with turn count.

Scope and reproducibility. This small mathematical evaluation does not measure open-domain or multilingual dialogue, interruptions, prosody, noise robustness, latency, speech naturalness, or memory-management cost. The aggregate record does not specify exact model snapshots, audiogeneration settings, sampling parameters, per-condition denominators, repeat counts, or answerextraction and scoring details. LEGO’s component identities and memory-update implementation are also unspecified. These omissions limit reproduction and statistical interpretation.

Architectural interpretation. A matched text-only baseline is needed to quantify an audio-specific penalty. To test whether LEGO’s context mechanism is responsible for its observed stability, the informative comparison would hold recognition, reasoning, and synthesis components fixed while varying context summarization. Problem-level outcomes and repeated runs would then support paired comparisons and uncertainty estimates. Such evidence would distinguish improvements in remembering information from improvements in using it, a distinction also raised by work on context-aware decoding [8]. Broader spoken benchmarks remain necessary to assess whether gains in numerical correctness extend to natural interaction.

Conclusion. SpeechConversationBench compares spoken mathematical task completion across original, consolidated, and incrementally disclosed inputs. All four commercial systems lose accuracy from CONCAT to SHARDED, whereas LEGO preserves its reported aggregate accuracy. The strongest supported lesson is methodological: retain both single-turn controls and report absolute task accuracy alongside the multi-turn change.

## Acknowledgments and Disclosure of Funding

We thank Monthol Charattrakool, Peerawat Rojratchadakorn, and Natthapath Rungseesiripak for their engineering contributions to LEGO. We also thank Weerin Chantaroje, Head of Innovation Lab, SCBX, and Tutanon Sinthuprasith, Head of SCBX R&D, for their leadership and support.

## References

[1] Yiming Chen, Xianghu Yue, Chen Zhang, Xiaoxue Gao, Robby T. Tan, and Haizhou Li. VoiceBench: Benchmarking LLM-based voice assistants. arXiv:2410.17196, 2024. URL https://arxiv.org/abs/2410.17196.

[2] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. arXiv:2110.14168, 2021. URL https://arxiv.org/abs/2110.14168.

[3] Alexandre Défossez, Laurent Mazaré, Manu Orsini, Amélie Royer, Patrick Pérez, Hervé Jégou, Edouard Grave, and Neil Zeghidour. Moshi: a speech-text foundation model for real-time dialogue. arXiv:2410.00037, 2024. URL https://arxiv.org/abs/2410.00037.

[4] Yuhao Du, Qianwei Huang, Guo Zhu, Zhanchen Dai, Shunian Chen, Qiming Zhu, Le Pan, Minghao Chen, Yuhao Zhang, Li Zhou, Benyou Wang, and Haizhou Li. MTalk-Bench: Evaluating speech-to-speech models in multi-turn dialogues via arena-style and rubrics protocols. arXiv:2508.18240, 2025. URL https://arxiv.org/abs/2508.18240.

[5] Advait Gosai, Tyler Vuong, Utkarsh Tyagi, Steven Li, Wenjia You, Miheer Bavare, Arda Uçar, Zhongwang Fang, Brian Jang, Bing Liu, and Yunzhong He. Audio MultiChallenge: A multi-turn evaluation of spoken dialogue systems on natural human interaction. arXiv:2512.14865, 2025. URL https://arxiv.org/abs/2512.14865.

[6] Chien-yu Huang, Ke-Han Lu, Shih-Heng Wang, Chi-Yuan Hsiao, Chun-Yi Kuan, Haibin Wu, Siddhant Arora, Kai-Wei Chang, Jiatong Shi, Yifan Peng, Roshan Sharma, Shinji Watanabe, Bhiksha Ramakrishnan, Shady Shehata, and Hung-yi Lee. Dynamic-SUPERB: Towards a dynamic, collaborative, and comprehensive instruction-tuning benchmark for speech. arXiv:2309.09510, 2023. URL https://arxiv.org/abs/2309.09510.

[7] Philippe Laban, Hiroaki Hayashi, Yingbo Zhou, and Jennifer Neville. LLMs get lost in multiturn conversation. arXiv:2505.06120, 2025. URL https://arxiv.org/abs/2505.06120.

[8] Che Hyun Lee, Heeseung Kim, and Sungroh Yoon. From awareness to adherence: Bridging the context gap in spoken dialogue systems via context-aware decoding. arXiv:2606.16472, 2026. URL https://arxiv.org/abs/2606.16472.

[9] Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Barbieri, and Yuwei Fang. Evaluating very long-term conversational memory of LLM agents. arXiv:2402.17753, 2024. URL https://arxiv.org/abs/2402.17753.

[10] Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang, Shishir G. Patil, Ion Stoica, and Joseph E. Gonzalez. MemGPT: Towards LLMs as operating systems. arXiv:2310.08560, 2023. URL https://arxiv.org/abs/2310.08560.

[11] Thanapol Popit, Natthapath Rungseesiripak, Monthol Charattrakool, and Saksorn Ruangtanusak. Thai semantic end-of-turn detection for real-time voice agents. arXiv:2510.04016, 2025. URL https://arxiv.org/abs/2510.04016.