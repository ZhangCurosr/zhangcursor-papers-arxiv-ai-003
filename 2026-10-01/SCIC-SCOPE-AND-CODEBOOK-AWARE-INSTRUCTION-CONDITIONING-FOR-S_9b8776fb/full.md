# SCIC: SCOPE- AND CODEBOOK-AWARE INSTRUCTION CONDITIONING FOR SPEAKER-ADAPTED EXPRESSIVE TTS

Longyu Lu, Zongwei Du, Mengtao Xing, Zhuoqun Liu, Zifan Guan, Meiguang Jin<sup>\*</sup>, Junfeng Ma

TaoLive-AIGC Team Taobao & Tmall Group of Alibaba

## ABSTRACT

Long-form live-streaming TTS requires context-dependent prosody and paragraph-level coherence. However, many existing instructionbased TTS systems use global or uniform conditions, providing limited explicit control over clause-level relative prosodic changes. We introduce Speaker-Relative Inline Prosody Control, where each Pitch, Energy, or Speed instruction targets a clause relative to the preceding clause from the same speaker, while Pause uses an absolute duration interval. In codec-based TTS, Speed and Pause affect sequence length, whereas Pitch and Energy rely on residual codebooks. By analyzing Qwen3-TTS RVQ codebooks, we find that Energy concentrates in early residual codebooks, whereas Pitch accumulates across a deeper prefix. We therefore propose Scopeand Codebook-Aware Instruction Conditioning (SCIC), combining a Temporal Instruction Router for frame-level tag activation with Tag-Specific Codebook Weighting over residual codebooks. SCIC improves speaker-relative Pitch and Energy control over standard instruction fine-tuning using text-token tags. We further apply multi-reward GDPO post-training to jointly optimize control and quality, improving control accuracy while preserving CER and speaker similarity. In long-form synthesis, SCIC produces a more distinct paragraph-level expressive hierarchy than speakeradapted SFT without instructions. Audio demos are available at: https://taoliveaigc.github.io/SCIC/.

Index Terms— text-to-speech, speaker adaptation, relative prosody control, residual vector quantization

## 1. INTRODUCTION

In long-form live-streaming scenarios, speech exhibits local variations in pitch, energy, speaking rate, and pauses. TTS systems for this setting require fine-grained instruction control while maintaining paragraph-level coherence. Recent systems support naturallanguage instructions [1–11], categorical, continuous, or relative attribute control [12–14], and fine-grained local control through inline or hierarchical annotations [15–20]. Preference or policy optimization has also improved instruction adherence and controllability [8, 18]. However, existing methods do not jointly formulate inline prosody as a clause-level change relative to the preceding samespeaker clause and model its temporal scope and RVQ codebook allocation. This gap limits coherent prosodic progression in long-form synthesis.

We address this problem through Speaker-Relative Inline Prosody Control. An inline tag placed before a target clause specifies a local Pitch, Energy, Speed, or Pause change. For Pitch, Energy, and Speed, the preceding clause from the same speaker serves as the reference anchor for the controlled clause; Pause is specified by an absolute duration interval. This formulation avoids imposing one absolute acoustic target across speakers and instead preserves speaker-specific prosodic variation. It also provides a direct interface for constructing paragraph-level expressive progression from a sequence of locally controlled clauses. Our setting is speaker-adapted rather than zero-shot speaker generalization: all controlled systems are adapted to the target voices during supervised fine-tuning and evaluated on held-out utterances from those speakers.

A remaining challenge is that different prosodic attributes are realized through different parts of a codec-based autoregressive generator. In Qwen3-TTS [21], the Main Talker predicts the first codebook q<sub>0</sub> and advances the acoustic-frame sequence, while Multi-Token Prediction (MTP) predicts residual codebooks q<sub>1</sub>–q<sub>15</sub> within each frame. Speed and Pause primarily affect temporal progression and sequence length and can therefore be controlled through the existing q<sub>0</sub> path. Pitch and Energy, by contrast, depend on acoustic information distributed across the residual codebooks. Standard instruction fine-tuning represents inline tags only as text tokens and does not explicitly determine either when a Pitch/Energy tag should be active or how strongly it should affect each residual codebook.

To identify the relevant codec structure, we analyze Qwen3-TTS through RVQ Codebook Diagnosis, exchanging projected codebook contributions between original and attribute-transformed speech. The resulting cumulative-transfer trajectories differ substantially by attribute: Energy effects saturate within the first few residual codebooks, whereas Pitch transfer accumulates across a substantially deeper codebook prefix. Motivated by this observation, we propose Scope- and Codebook-Aware Instruction Conditioning (SCIC). Its Temporal Instruction Router predicts frame-level activation for each inline Pitch/Energy tag, and its Tag-Specific Codebook Weighting modulates the signed conditioning strength across q<sub>1</sub>–q<sub>15</sub>. The two components jointly inject a localized SCIC conditioning signal into MTP, while Speed and Pause remain on the autoregressive q<sub>0</sub> path. We further post-train SCIC-SFT with multi-reward GDPO to jointly optimize prosody control, Pause accuracy, intelligibility, and speaker similarity. Experiments confirm improved local control while preserving synthesis quality and long-form coherence.

Our key contributions are as follows. First, we formulate Speaker-Relative Inline Prosody Control with absolute Pause intervals and relative Pitch, Energy, and Speed Up/Down instructions. Second, RVQ Codebook Diagnosis reveals attribute-dependent behavior: Energy transfer saturates within the first few residual codebooks, whereas Pitch transfer accumulates across a substantially deeper codebook prefix. Third, we propose SCIC, which combines a Temporal Instruction Router with Tag-Specific Codebook Weighting, and further improve SCIC-SFT through multi-reward GDPO post-training with explicit control and quality objectives.

## 2. METHODOLOGY

## 2.1. Overall framework

Figure 1 illustrates the generation pipeline. The Main Talker predicts q and provides the frame-level state $h _ { t }$ to SCIC. For each Pitch/Energy tag τ , its tag embedding $e _ { \tau } .$ presence indicator $\chi _ { \tau }$ , Router output $\rho _ { t , \tau }$ , and codebook weight $\omega _ { \tau , k }$ form $s _ { t , k }$ after summation over tags. At residual step $k ,$ this signal is added to the preceding codebook-token embedding, yielding $u _ { t , k } = E _ { k - 1 } ( q _ { t , k - 1 } ) + s _ { t , k }$ . MTP uses $h _ { t }$ and the conditioned inputs to predict q<sub>1</sub>–q<sub>15</sub>, whose projected contributions are summed and decoded. Multi-reward GDPO further optimizes control, intelligibility, and speaker similarity.

## 2.2. Speaker-relative inline control formulation

We first formalize how inline instructions specify local prosodic targets. Each instruction associates a target clause with either a speaker-relative change in Pitch, Energy, or Speed, measured against the immediately preceding clause from the same speaker, or an absolute Pause interval at the instructed boundary. Let $c _ { i - 1 }$ and $c _ { i }$ denote the reference and target clauses, respectively, with the inline tag placed immediately before $c _ { i }$ . For voiced-median pitch $F _ { i }$ , shared-voiced Top-80% RMS energy $E _ { i }$ , and aligned-character speaking rate $r _ { i } .$ , the adjacent-clause changes are

$$
\begin{array} { r l r } & { \Delta _ { i } ^ { \mathrm { p i t } } = 1 2 \log _ { 2 } ( F _ { i } / F _ { i - 1 } ) , ~ } & { \Delta _ { i } ^ { \mathrm { e n g } } = 2 0 \log _ { 1 0 } ( E _ { i } / E _ { i - 1 } ) , } \\ & { \Delta _ { i } ^ { \mathrm { s p d } } = \log ( r _ { i } / r _ { i - 1 } ) , ~ } & { m _ { i } ^ { a } = \eta _ { i } \Delta _ { i } ^ { a } , } \end{array}\tag{1}
$$

where $a ~ \in ~ \{ \mathrm { p i t , e n g , s p d } \}$ indexes the controlled attribute, and $\eta _ { i } ~ = ~ + 1$ for an Up instruction and −1 for a Down instruction. Thus $m _ { i } ^ { a }$ is the direction-normalized change, with a positive value always indicating movement in the requested direction. The logarithmic definitions provide symmetric representations of reciprocal Up and Down changes and avoid imposing one absolute acoustic target across speakers.

Pause is treated as an absolute-duration task with three reported levels: L1 [0.50, 0.75], L2 [0.75, 1.00], and L3 [1.00, 1.25] seconds.

## 2.3. RVQ codebook diagnosis

Given an original waveform x, we construct a duration-matched counterpar $\cdot x ^ { \bar { \prime } }$ by applying either an upward or downward Pitch shift, or an increase or decrease in Energy. Codec encoding produces $C , C ^ { \prime } \in \mathbb { Z } ^ { T \times 1 6 }$ , where $T$ is the number of codec frames and column k contains codebook-k tokens. Let $\ell _ { k }$ and $\ell _ { k } ^ { \prime }$ denote the projected contribution of codebook k to the decoder input, giving $z =$ $\textstyle \sum _ { k = 0 } ^ { 1 5 } \ell _ { k }$ and $\begin{array} { r } { z ^ { \prime } = \sum _ { k = 0 } ^ { 1 5 } \ell _ { k } ^ { \prime } } \end{array}$ . We exchange these projected contributions rather than discrete token columns. For a codebook set $S \subseteq \{ 0 , \ldots , 1 5 \}$

$$
\delta _ { S } = \sum _ { k \in S } ( \ell _ { k } ^ { \prime } - \ell _ { k } ) , \quad \widetilde { z _ { S } } ^ { \ast } = z + \delta _ { S } , \quad \widetilde { z _ { S } } ^ { \ast } = z ^ { \prime } - \delta _ { S } .\tag{2}
$$

Here $\delta _ { S }$ is the transformed-minus-original contribution from $S ; { \tilde { z } } _ { S } ^ {  }$ inserts it into $z ,$ whereas $\widetilde { z } _ { S } ^ {  }$ removes it from $z ^ { \prime } .$ . Let $D$ denote codec decoding and $\phi _ { a }$ the measurement of attribute a $\in \{ \mathrm { p i t } , \mathrm { e n g } \}$ Transfer relative to the complete codec-reconstructed change is

$$
\begin{array} { r l } & { \mathrm { T r } _ { \vec { S _ { a } } } ^ { \right. } = \frac { \phi _ { a } \left( D ( \vec { z } _ { \vec { S } } ^ {  } ) \right) - \phi _ { a } \left( D ( z ) \right) } { \ph\right.i _ { a } \left( D ( z ^ { \prime } ) \right) - \phi _ { a } \left( D ( z ) \right) } , } \\ & { \mathrm { T r } _ { \vec { S _ { a } } } ^ { \left. } = \frac { \phi _ { a } \left( D ( z ^ { \prime } ) \right) - \phi _ { a } \left( D ( \vec { z } _ { \vec { S } } ^ { \left. \right)} )  } { \phi _ { a } \left( D ( z ^ { \prime } ) \right) - \phi _ { a } \left( D ( z ) \right) } . } \end{array}\tag{3}
$$

Forward intervention inserts the transformed contributions into the original reconstruction, while reverse intervention removes them from the transformed reconstruction. Negative transfer indicates movement opposite to the transformation, and values above 100% indicate overshoot. These ratios describe attribute transfer under latent intervention rather than additive information percentages.

Energy rapidly saturates within $q _ { 1 } { - } q _ { 3 }$ , whereas Pitch requires a deeper interacting prefix and approaches saturation around $q _ { 1 } .$ –q<sub>10</sub>. These different trajectories motivate tag-specific codebook weights rather than uniform injection or one fixed profile.

## 2.4. Scope- and codebook-aware instruction conditioning

SCIC factorizes local instruction conditioning along the temporal and codebook axes. Let τ index the four inline Pitch/Energy instruction-tag types, $\chi _ { \tau } \in \{ 0 , 1 \}$ indicate whether tag τ occurs in the input, and $e _ { \tau } = P _ { \mathrm { t e x t } } \mathrm { E m b } _ { \mathrm { t e x t } } ( \tau )$ be its projected text representation. Here $\operatorname { E m b } _ { \mathrm { t e x t } } ( \tau )$ is the embedding of tag token $\tau ,$ and $P _ { \mathrm { t e x t } }$ is the existing Qwen3-TTS text projection that maps it to the hidden dimension shared by the Main Talker and MTP inputs. Given the Main Talker state $h _ { t }$ associated with codec frame $t ,$ the Temporal Instruction Router predicts

$$
\rho _ { t , \tau } = \sigma ( [ W \mathrm { L N } ( h _ { t } ) + b ] _ { \tau } ) \chi _ { \tau } .\tag{4}
$$

Here $W$ and b are learned Router parameters, LN is layer normalization, and σ is the logistic sigmoid. Thus $\rho _ { t , \tau } \in [ 0 , 1 ]$ specifies the frame-level activation strength of tag τ; multiplication by $\chi \tau$ makes absent tags exactly inactive. Aligned clause spans provide binary frame-level supervision. The Router loss balances positive and negative frames separately for each supervised tag type so that long inactive regions do not dominate training.

For residual codebook $k \in \{ 1 , \ldots , 1 5 \}$ , Tag-Specific Codebook Weighting assigns each tag a zero-initialized parameter $\omega _ { \tau , k }$ . The localized conditioning signal and conditioned input are

$$
\begin{array} { l } { { s _ { t , k } = \displaystyle \sum _ { \tau } \rho _ { t , \tau } \omega _ { \tau , k } e _ { \tau } } , } \\ { { } } \\ { { u _ { t , k } = \mathrm { E m b } _ { k - 1 } \big ( q _ { t , k - 1 } \big ) + s _ { t , k } . } } \end{array}\tag{5}
$$

MTP predicts $\left( \widehat { q } _ { t , 1 } , \dots , \widehat { q } _ { t , 1 5 } \right)$ from $h _ { t }$ and $( u _ { t , 1 } , \ldots , u _ { t , 1 5 } )$ . Here $\operatorname { E m b } _ { k - 1 }$ is the codec-token embedding used when predicting $q _ { t , k } .$ The Router controls temporal scope, while $\omega _ { \tau , k }$ controls signed strength across $q \mathrm { 1 } { - } q \mathrm { 1 5 }$ . Pitch and Energy use this explicit SCIC conditioning path. Training combines generation and balanced framelevel Router losses, $\mathcal { L } = \mathcal { L } _ { \mathrm { g e n } } + \lambda _ { r } \mathcal { L } _ { \mathrm { r o u t e r } }$ , where $\lambda _ { r }$ weights the Router objective.

## 2.5. Multi-reward GDPO post-training

SCIC-SFT is post-trained with bounded control and quality rewards. For Pitch, Energy, or Speed, let m be the direction-normalized change, $T = T _ { v , a } = 1 . 8 \sigma _ { v , a }$ the frozen speaker-specific threshold, and $P = P _ { v , a , d }$ the direction-specific training-set 99th-percentile cap, with $0 < T < P$ . The control reward is

$$
\begin{array} { r } { R _ { \mathrm { c t r l } } ( m ) = \left\{ \begin{array} { l l } { \mathrm { m a x } ( - 1 , m / T ) , } & { m \leq 0 , } \\ { m / T , } & { 0 < m < T , } \\ { 1 , } & { T \leq m \leq P , } \\ { \mathrm { m a x } \left( - 1 , 1 - \displaystyle \frac { 2 ( m - P ) } { P - T } \right) , } & { m > P . } \end{array} \right. } \end{array}\tag{6}
$$

![](images/1cffa5379b8502638cfb73ae7d30b863e6f8c8fb75874464ac235fccdb0a7d21.jpg)  
Fig. 1. SCIC in Qwen3-TTS. The Main Talker predicts $q _ { 0 }$ and provides $h _ { t } .$ For each Pitch/Energy tag, its tag embedding is scaled by the Temporal Instruction Router output and the tag-specific codebook weight to form $s _ { t , k }$ , which conditions MTP prediction of $q \mathrm { _ { 1 } } \mathrm { - } q \mathrm { _ { 1 5 } }$

This rewards requested-direction changes above T and penalizes incorrect or excessive responses. For pause duration p and target interval $[ l , h ]$ , let dist $( p , \bar { [ \imath , \ : h ] } ) = \operatorname* { m a x } ( l - p , 0 , p - h )$ . The Pause reward is

$$
R _ { \mathrm { p a u } } ( p ) = \left\{ \begin{array} { l l } { 1 , } & { p \in [ l , h ] , } \\ { \operatorname* { m a x } \left( - 1 , 1 - \frac { 2 \mathrm { d i s t } ( p , [ l , h ] ) } { h - l } \right) , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{7}
$$

Bounded CER and speaker-similarity rewards preserve intelligibility and speaker identity. Following GDPO [22], each objective o is standardized across the G responses j to input i before weighted aggregation:

$$
A _ { o } ^ { ( i , j ) } = \frac { R _ { o } ^ { ( i , j ) } - \mu _ { i , o } } { \sigma _ { i , o } + \epsilon } , ~ A ^ { ( i , j ) } = \sum _ { o } w _ { o } A _ { o } ^ { ( i , j ) } .\tag{8}
$$

Here ϵ ensures numerical stability and $w _ { o }$ weights objective o, preventing reward-scale differences from dominating optimization.

## 3. EXPERIMENTS

## 3.1. Experimental setup

Dataset. Our supervised data comprise approximately 1,000 hours of natural Chinese speech from 30 speakers. Each recording is at most 90 seconds long, and the processed corpus contains 132,018 instruction tag. All evaluation utterances are disjoint from the supervised and post-training data.

Implementation details. We initialize all systems from a live-streaming Qwen3-TTS CPT checkpoint and then perform supervised fine-tuning on our instruction data. Instruction SFT uses the same inline tags and supervised data but represents the instructions only through text-token conditioning. SCIC-SFT additionally introduces the Temporal Instruction Router and Tag-Specific Codebook Weighting, and SCIC-GDPO further applies multi-reward post-training. Supervised fine-tuning uses rank-32 LoRA [23] with α = 64 on attention and MLP projections. AdamW uses learning rates of $1 0 ^ { - 5 }$ for LoRA and $\mathrm { i 0 ^ { - 4 } }$ for control embeddings, the Router, and tag-specific codebook weights. We train on eight NVIDIA RTX PRO 5000 72GB GPUs with a per-GPU batch size of 8. SCIC-GDPO samples eight responses per input and uses a learning rate of $5 \times 1 0 ^ { - 6 }$

Evaluation metrics. We evaluate 1,800 Pitch/Energy/Speed samples using control-direction accuracy and median directionnormalized change, and 1,200 samples per Pause level using interval-hit accuracy. Qwen3-ForcedAligner-0.6B [24] provides alignment, Qwen3-ASR [24] measures CER on 1,800 texts, and WeSpeaker [25] measures similarity on 1,200 samples. Router localization is reported as mean aligned boundary error in milliseconds. Long-form HMOS uses 200 texts, outputs longer than 60 seconds, and five listeners.

## 3.2. Main results

Relative to Instruction SFT, SCIC-SFT improves Pitch, Energy, Speed, and Pause by 7.31, 10.53, 0.51, and 0.67 percentage points, respectively, with the largest gains on the explicitly conditioned Pitch and Energy attributes. Multi-reward post-training extends the corresponding SCIC-GDPO gains to 9.65, 11.30, 2.97, and 3.87 percentage points. CER remains within a narrow 1.63–1.68% range, while speaker similarity remains essentially unchanged.

Pause ACC measures whether the detected silence falls inside the requested duration interval. SCIC-GDPO achieves the best result at all three levels, and the mean detected durations of all systems lie inside their corresponding target intervals.

Cumulative transfer  
![](images/c281022c1382ad446b8004652e33b2798731b9c84d30b9b8d855ebb94428c072.jpg)  
Fig. 2. Median speaker-relative change magnitudes after direction normalization. Down values are sign-flipped so positive values indicate the requested direction; black lines and diamonds denote frozen training-reference magnitudes.

Table 1. RVQ transfer (%). Left: selected single-codebook results with speaker-bootstrap 95% confidence intervals. Right: cumulative transfer over residual-codebook prefixes.  
Single-codebook transfer
<table><tr><td>Codebook</td><td>Pitch Fwd.</td><td>Pitch Rev.</td><td>Energy Fwd.</td><td>Energy Rev.</td></tr><tr><td>q0</td><td>0.0 [-0.0,0.1]</td><td>0.5 [-0.2,1.7]</td><td>-0.0 [-0.1,0.0]</td><td>-0.0 [-0.1,0.0]</td></tr><tr><td>q1</td><td></td><td></td><td>-5.0 [-7.4,-2.8] 18.3 [15.7,21.0] 83.5 [82.7,84.3]85.3 [84.5,86.1]</td><td></td></tr><tr><td>q2</td><td></td><td>-0.1 [-1.6,1.2] 12.7 [10.0,16.1]</td><td>6.6 [6.1,7.0]</td><td>8.1 [7.7,8.6]</td></tr><tr><td>q3</td><td>3.7 [2.6,4.7]</td><td>10.2 [7.7,12.9]</td><td>2.5 [2.2,2.8]</td><td>3.1 [2.7,3.5]</td></tr><tr><td>q4</td><td>2.1 [1.4,2.9]</td><td>10.2 [7.3,13.0]</td><td>1.6 [1.3,2.0]</td><td>1.9 [1.6,2.1]</td></tr><tr><td>q5</td><td>0.8 [0.2,1.2]</td><td>2.9 [-0.9,5.5]</td><td>1.0 [0.8,1.2]</td><td>1.2 [0.9,1.5]</td></tr></table>

<table><tr><td colspan="2">Attribute Codebooks Forward Reverse</td></tr><tr><td>Energy</td><td>q1−q2 91.0 92.1</td></tr><tr><td>Energy</td><td>q1−q3 94.0 94.7</td></tr><tr><td>Pitch q1−q4</td><td>52.9 68.1</td></tr><tr><td>Pitch q1−q6</td><td>83.4 88.7</td></tr><tr><td>Pitch q1-q8</td><td>91.0 96.5</td></tr><tr><td>Pitch q1−q10</td><td>95.4 99.6</td></tr></table>

Table 2. Control and quality results. Pitch, Energy, Speed, Pause ACC, and CER are percentages; SIM is cosine similarity, and mean Pause durations are in seconds.
<table><tr><td>Model</td><td>Pitch Energy Speed</td><td></td><td></td><td colspan="4">Pause ACC</td><td colspan="4">Mean Pause</td><td>CER↓ SIM↑</td></tr><tr><td></td><td></td><td></td><td></td><td>L1</td><td>L2</td><td>L3</td><td>Macro</td><td>L1</td><td>L2</td><td>L3</td><td></td><td></td></tr><tr><td>Instruction SFT</td><td>82.83</td><td>85.86</td><td>90.89</td><td>92.41</td><td>92.55</td><td>88.81</td><td>91.26</td><td>0.615</td><td>0.863</td><td>1.110</td><td>1.68</td><td>0.8721</td></tr><tr><td>SCIC-SFT</td><td>90.14</td><td>96.39</td><td>91.40</td><td>94.34</td><td>92.79</td><td>88.65</td><td>91.93</td><td>0.618</td><td>0.856</td><td>1.101</td><td>1.63</td><td>0.8719</td></tr><tr><td>SCIC-GDPO</td><td>92.48</td><td>97.16</td><td>93.86</td><td>95.82</td><td>95.90</td><td>93.68</td><td>95.13</td><td>0.635</td><td>0.8761.136</td><td></td><td>1.64</td><td>0.8718</td></tr></table>

On 200 long-form texts, HMOS increases from 3.244 for Zero-Shot and 3.504 for SFT without instructions to 4.042 with paragraphlevel instructions, indicating a clearer expressive hierarchy for the complete system.

The control-magnitude results further show that the controlaccuracy gains correspond to stronger direction-normalized changes in each attribute’s native domain. Relative to SCIC-SFT, SCIC-GDPO increases the median change magnitude for both directions of Pitch, Energy, and Speed while remaining below the frozen training references.

## 3.3. Temporal and codebook-axis ablations

Replacing the soft Router with a binary Router reduces Pitch accuracy and increases Boundary MAE from 295 to 375 ms, supporting graded temporal activation. Removing Tag-Specific Codebook Weighting lowers Pitch and Energy accuracy, while its 292-ms Boundary MAE remains close to SCIC-SFT. These results indicate that the Temporal Instruction Router primarily controls instruction scope, whereas Tag-Specific Codebook Weighting mainly controls attribute allocation across residual predictions.

Table 3. Ablations: control-direction accuracy (%) and Router Boundary MAE.
<table><tr><td>Variant</td><td>Pitch</td><td>Energy</td><td>MAE↓</td></tr><tr><td>SCIC-SFT</td><td>90.14</td><td>96.39</td><td>295 ms</td></tr><tr><td>w/ Hard Router</td><td>88.41</td><td>96.44</td><td>375 ms</td></tr><tr><td>w/o Tag-Specific Codebook Weighting</td><td>89.43</td><td>94.90</td><td>292 ms</td></tr></table>

## 4. CONCLUSIONS

We introduced SCIC for speaker-relative inline prosody control. RVQ Codebook Diagnosis shows that Energy concentrates in early residual codebooks whereas Pitch requires a deeper prefix, motivating temporal routing and tag-specific codebook weighting. Multireward GDPO improves Pitch, Energy, Speed, and Pause control while maintaining comparable intelligibility and speaker similarity. Long-form listening indicates a clearer paragraph-level expressive hierarchy for the complete instruction-conditioned system.

## 5. REFERENCES

[1] Dongchao Yang, Songxiang Liu, Rongjie Huang, Chao Weng, and Helen Meng, “InstructTTS: Modelling expressive TTS in discrete latent space with natural language style prompt,” IEEE/ACM Trans. Audio, Speech, Language Process., vol. 32, pp. 2913–2925, 2024.

[2] Zhifang Guo, Yichong Leng, Yihan Wu, Sheng Zhao, and Xu Tan, “PromptTTS: Controllable text-to-speech with text descriptions,” in Proc. IEEE ICASSP, pp. 1–5, 2023.

[3] Yichong Leng, Zhifang Guo, Kai Shen, Xu Tan, Zeqian Ju, Yanqing Liu, Yufei Liu, Dongchao Yang, Leying Zhang, Kaitao Song, et al., “PromptTTS 2: Describing and generating voices with text prompt,” arXiv preprint arXiv:2309.02285, 2023.

[4] Guanghou Liu, Yongmao Zhang, Yi Lei, Yunlin Chen, Rui Wang, Zhifei Li, and Lei Xie, “PromptStyle: Controllable style transfer for text-to-speech with natural language descriptions,” in Proc. Interspeech, pp. 4888–4892, 2023.

[5] Shengpeng Ji, Qian Chen, Wen Wang, Jialong Zuo, Minghui Fang, Ziyue Jiang, Hai Huang, Zehan Wang, Xize Cheng, Siqi Zheng, et al., “ControlSpeech: Towards simultaneous zeroshot speaker cloning and zero-shot language style control with decoupled codec,” arXiv preprint arXiv:2406.01205, 2024.

[6] Hanzhao Li, Yuke Li, Xinsheng Wang, Jingbin Hu, Qicong Xie, Shan Yang, and Lei Xie, “FleSpeech: Flexibly controllable speech generation with various prompts,” arXiv preprint arXiv:2501.04644, 2025.

[7] Yong Ren, Jiangyan Yi, Jianhua Tao, Haiyang Sun, Zhengqi Wen, Hao Gu, Le Xu, and Ye Bai, “OV-InstructTTS: Towards open-vocabulary instruct text-to-speech,” arXiv preprint arXiv:2601.01459, 2026.

[8] Dekun Chen, Xueyao Zhang, Yuancheng Wang, Kenan Dai, Li Ma, and Zhizheng Wu, “FlexiVoice: Enabling flexible style control in zero-shot TTS with natural language instructions,” arXiv preprint arXiv:2601.04656, 2026.

[9] Jinchuan Tian, Haoran Wang, Siddhant Arora, Takashi Maekaku, Keita Goto, Jin Sakuma, Yusuke Shinohara, Chao-Han Huck Yang, and Shinji Watanabe, “Bagpiper-TTS: Natural language guided universal speech synthesis,” arXiv preprint arXiv:2606.22811, 2026.

[10] Zhihao Du, Changfeng Gao, Yuxuan Wang, Fan Yu, Tianyu Zhao, Hao Wang, Xiang Lv, Hui Wang, Chongjia Ni, Xian Shi, et al., “CosyVoice 3: Towards in-the-wild speech generation via scaling-up and post-training,” arXiv preprint arXiv:2505.17589, 2025.

[11] Sihang Nie, Xiaofen Xing, Jingyuan Xing, Baiji Liu, and Xiangmin Xu, “HD-PPT: Hierarchical decoding of content- and prompt-preference tokens for instruction-based TTS,” in Proc. IEEE ICASSP, pp. 16487–16491, 2026.

[12] Xinsheng Wang, Mingqi Jiang, Ziyang Ma, Ziyu Zhang, Songxiang Liu, Linqin Li, Zheng Liang, Qixi Zheng, Rui Wang, Xiaoqin Feng, et al., “Spark-TTS: An efficient LLMbased text-to-speech model with single-stream decoupled speech tokens,” arXiv preprint arXiv:2503.01710, 2025.

[13] Nikita Torgashov, Gustav Eje Henter, and Gabriel Skantze, “VoXtream2: Full-stream TTS with dynamic speaking rate control,” arXiv preprint arXiv:2603.13518, 2026.

[14] Haitao Li, Chunxiang Jin, Chenglin Li, Wenhao Guan, Zhengxing Huang, and Xie Chen, “ReStyle-TTS: Relative and continuous style control for zero-shot speech synthesis,” arXiv preprint arXiv:2601.03632, 2026.

[15] Jialong Mai, Xiaofen Xing, and Xiangmin Xu, “MAGIC-TTS: Fine-grained controllable speech synthesis with explicit local duration and pause control,” arXiv preprint arXiv:2604.21164, 2026.

[16] Sihang Nie, Jinxin Ji, Xiaofen Xing, Deyi Tuo, Chengbin Jin, Jialong Mai, and Xiangmin Xu, “WordVoice: Explicit and decoupled multi-dimensional word-level control for LLM-based TTS,” arXiv preprint arXiv:2607.06461, 2026.

[17] Bajian Xiang, Cheng Wen, Han Zhao, Hao Wang, Haoxu Wang, Jiawei Jin, Jiayan Cui, Jie Chen, Mengxi Nie, Tianyu Zhao, et al., “Qwen-Audio-3.0-TTS: Freely controllable and highly robust speech synthesis with multi-stage training paradigm,” arXiv preprint arXiv:2607.23938, 2026.

[18] Shijia Liao, Yuxuan Wang, Songting Liu, Yifan Cheng, Ruoyi Zhang, Tianyu Li, Shidong Li, Yisheng Zheng, Xingwei Liu, Qingzheng Wang, et al., “Fish Audio S2 technical report,” arXiv preprint arXiv:2603.08823, 2026.

[19] Xingchen Song, Di Wu, Dinghao Zhou, Pengyu Cheng, Hongwu Ding, Yunchao He, Jie Wang, Shuai Wang, Shengfan Shen, Sixiang Lv, et al., “Any2Speech: Borderless long audio synthesis,” arXiv preprint arXiv:2603.19798, 2026.

[20] Tianchi Liu, Zeyang Song, Tianrui Wang, Zhipeng Li, Chenglin Xu, and Yiwen Guo, “EmoTra-TTS: Smooth intrautterance emotion transitions for speech synthesis,” arXiv preprint arXiv:2608.23791, 2026.

[21] Hangrui Hu, Xinfa Zhu, Ting He, Dake Guo, Bin Zhang, Xiong Wang, Zhifang Guo, Ziyue Jiang, Hongkun Hao, Zishan Guo, et al., “Qwen3-TTS technical report,” arXiv preprint arXiv:2601.15621, 2026.

[22] Shih-Yang Liu, Xin Dong, Ximing Lu, Shizhe Diao, Peter Belcak, Mingjie Liu, Min-Hung Chen, Hongxu Yin, Yu-Chiang Frank Wang, Kwang-Ting Cheng, et al., “GDPO: Group reward-decoupled normalization policy optimization for multireward RL optimization,” arXiv preprint arXiv:2601.05242, 2026.

[23] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen, “LoRA: Low-rank adaptation of large language models,” in Proc. ICLR, 2022.

[24] Xian Shi, Xiong Wang, Zhifang Guo, Yongqi Wang, Pei Zhang, Xinyu Zhang, Zishan Guo, Hongkun Hao, Yu Xi, Baosong Yang, et al., “Qwen3-ASR technical report,” arXiv preprint arXiv:2601.21337, 2026.

[25] Hongji Wang, Chengdong Liang, Shuai Wang, Zhengyang Chen, Binbin Zhang, Xu Xiang, Yanlei Deng, and Yanmin Qian, “WeSpeaker: A research and production oriented speaker embedding learning toolkit,” arXiv preprint arXiv:2210.17016, 2022.