# V-Engram: Trigger-Indexed External Memory for Modular Text-to-Image Personalization

Haoran He Runyuan Cai Yiming Wang Yu Lin Xiaodong Zeng

AutoArk-AI

![](images/24526f283b982f7f5104092bdb01d9838d0469c6077e9e874f26d9dc2070888a.jpg)  
Figure 1: Compositional generation with V-Engram. A few reference images bind the trigger <rc-car> to a specific toy car; three are shown for illustration. The same memory composes with unseen environments, motion, lighting, and object interactions while preserving the characteristic driver, antenna, colors, and vehicle shape.

## Abstract

Pretrained text-to-image models contain broad visual knowledge, yet they cannot reliably acquire or refine a specific visual identity from only a few references while preserving compositional control. Token-embedding methods are compact but often underfit identity, whereas adapter-based methods improve fidelity through persistent weight updates that can be costly to store and interfere when concepts are composed. We introduce V-Engram, a trigger-indexed external memory mechanism for Stable Difusion 3.5. Each concept is assigned an explicit trigger that retrieves concept-specific memory, whose gated directions enter frozen text-encoder and MMDiT context states as relative residuals. Separating this memory from backbone adaptation enables prompt-selective and multi-concept access without merging model updates. Experiments show that V-Engram broadly matches DreamBooth-LoRA in overall subject fidelity while showing advantages in settings such as contextual subject preservation. Prompt-matched loading retrieves only matched entries, reducing most additional adaptation-state loading for a single-concept query. Qualitative results further demonstrate paired-trigger composition and same-class separation, while prompts without registered entries retain the frozen model’s base behavior. Together, these results establish trigger-indexed memory as a modular interface for adding targeted visual evidence without rewriting the generator.

## 1 Introduction

Large text-to-image generators encode broad visual and compositional knowledge [Rombach et al., 2022, Esser et al., 2024], yet their coverage is neither complete nor uniformly precise. A model may miss a newly observed subject, confuse similar instances, or render a familiar identity only approximately. Personalization thus spans both acquiring new concepts and selectively refining imprecise prior knowledge. This raises a broader question: where should targeted visual evidence be stored, how should a prompt retrieve it, and how can it enter generation without overwriting useful priors?

Few-shot personalization is a natural testbed for this question [Gal et al., 2023, Ruiz et al., 2023, Wu et al., 2025, Chen et al., 2025]. From a handful of references, a method must bind a concept to a textual handle, recover its fine-grained appearance, and follow new contexts, attributes, and interactions. The evidence should also be selectively addressable: inactive when its handle is absent and composable without identity omission or attribute leakage [Huang et al., 2025, Yao et al., 2025, Peng et al., 2026]. Figure 1 illustrates this objective for one toy car across diferent environments and actions.

Existing methods expose a capacity–modularity trade-of. Textual Inversion stores a concept in a small token embedding but often lacks capacity for detailed identity [Gal et al., 2023, Wu et al., 2025]. DreamBooth and LoRA increase capacity by adapting generator pathways, yielding concept-coupled state whose composition can introduce interference [Ruiz et al., 2023, Hu et al., 2022, Peng et al., 2026]. Encoder-based systems instead encode references at inference time for instant general-subject or face conditioning [Ye et al., 2023, Zhang et al., 2024, Li et al., 2024, Wang et al., 2024]. Together, these methods leave room for memory richer than a single token embedding but still explicitly addressed by text and inactive outside that address. This is particularly relevant to Stable Difusion 3.5, where text and image tokens interact throughout MMDiT blocks rather than through a fixed U-Net cross-attention interface [Esser et al., 2024, Peebles and Xie, 2023, Feng et al., 2026].

We propose V-Engram, a trigger-indexed external memory for Stable Difusion 3.5. Each concept receives an explicit trigger such as <dog>, which registers tokenizer-specific �-gram entries. Exact matches activate the entries present in a prompt; their directions and tanh gates enter mounted text-encoder and MMDiT context states as residuals scaled to the local hidden-state norm. Concept evidence resides outside the backbone, and all pretrained components remain frozen.

Unmatched prompts receive no Engram residual; prompts containing registered triggers activate and sum all exactly matched entries. The interface can acquire a novel subject, separate similar instances, or refine an imprecise familiar concept. Storage grows with the registry, while inference activates only prompt-matched entries.

Experiments compare V-Engram with zero-shot Stable Difusion 3.5, embedding-only learning, and joint DreamBooth-LoRA. V-Engram and LoRA show broadly comparable fidelity with complementary CLIP-I and DINOv2 outcomes. In held-out contexts, V-Engram preserves identity better, while LoRA favors prompt alignment. Qualitative studies show joint expression of separately addressed memories, inactive-route preservation, and same-class separation. Ablations support gated residual injection and distributed MMDiT access, while showing that denser mounting is not uniformly beneficial. Overall, V-Engram ofers a diferent balance of fidelity, modularity, and compositional access rather than a universally stronger replacement for weight adaptation.

Our contributions are:

• We formulate few-shot acquisition and refinement as exact trigger-indexed memory retrieval over a frozen difusion transformer.

• We introduce concept-specific Engram directions and gated relative residual injection for explicit, prompt-selective memory access in SD3.5.

![](images/07d5f164fcab32b0e978a8e058be5c612e780c5133a9c69ceafbc211097fabcc.jpg)  
Figure 2: Overview of V-Engram. Few-shot references and an explicit trigger construct tokenizer-specific, layer-wise Engram entries. Exact token-id matches retrieve entries whose residuals are injected into selected frozen Stable Difusion 3.5 layers.

• We characterize its trade-of against embedding-only learning and joint LoRA across fidelity, addressable loading, and compositional access.

## 2 Related Work

Conditional Memory. Parameter-eficient adaptation limits trainable state through adapters, prompts, or low-rank updates, but does not by itself specify concept-level activation [Houlsby et al., 2019, Li and Liang, 2021, Lester et al., 2021, Hu et al., 2022, Liu et al., 2022]. External memory adds an addressing mechanism: Product-Key Memory performs structured content lookup, Memorizing Transformers use approximate �NN retrieval, and DeepSeek Engram deterministically maps matched token �-grams to learned entries [Lample et al., 2019, Wu et al., 2022, Cheng et al., 2026]. V-Engram shares deterministic lexical addressing, but changes both the memory content and target: few-shot visual references train concept-specific residual directions and gates for a frozen difusion model, without semantic or approximate retrieval.

Optimization-Based Personalization. Learned concepts can reside in token space or model updates. Textual Inversion learns one embedding, CoRe regularizes its contextual interaction, and NeTI and ProSpect enrich frozen-backbone representations across layers or timesteps [Gal et al., 2023, Wu et al., 2025, Alaluf et al., 2023, Zhang et al., 2023]. DreamBooth fine-tunes with prior preservation, while compact alternatives update attention, low-rank or rank-one factors, singular values, or sparse concept neurons [Ruiz et al., 2023, Kumari et al., 2023, Hu et al., 2022, Tewel et al., 2023, Han et al., 2023, Liu et al., 2023]. CustomContrast and AttnDreamBooth further target subject discrimination or text alignment [Chen et al., 2025, Pang et al., 2024]. V-Engram occupies a diferent point: the backbone remains frozen as in embedding learning, but its external, layer-wise memory is not confined to one input embedding.

Reference-Guided Methods. Image references can produce conditioning or compact adaptation state. BLIP-Difusion, ELITE, IP-Adapter, and SSR-Encoder learn general subject features [Li et al., 2023, Wei et al., 2023, Ye et al., 2023, Zhang et al., 2024]; PhotoMaker and InstantID specialize in human identity [Li et al., 2024, Wang et al., 2024]; and HyperDreamBooth and AnyDoor predict compact weights or object features [Ruiz et al., 2024, Chen et al., 2024]. These methods make a reference encoder or reference image part of the personalization interface. V-Engram instead distills references once into persistent, text-addressed memory, so generation is subsequently invoked by the trigger alone.

Multi-Concept Methods. Prior work jointly learns concepts, extracts several handles, or fuses adapters [Kumari et al., 2023, Avrahami et al., 2023, Gu et al., 2023]. FastComposer and DynASyn target identity and action control, while condition merging and isolated sampling paths reduce multi-condition confusion and leakage [Xiao et al., 2025, Choi et al., 2025, Huang et al., 2025, Yao et al., 2025]. TARA reduces interference in LoRA composition through rare-token masking and token–region alignment [Peng et al., 2026]. V-Engram instead stores trigger-indexed direction-and-gate entries and injects them as exact-match hidden-state residuals, shifting the focus from spatially aligned LoRA composition to deterministic retrieval and selective activation of persistent concept memory.

Difusion Transformers and MMDiT. Latent difusion commonly uses a text-conditioned U-Net, whereas DiT replaces it with transformer blocks [Rombach et al., 2022, Peebles and Xie, 2023]. Stable Difusion 3.5 combines CLIP/OpenCLIP and T5 streams in MMDiT, where text and image tokens interact jointly [Radford et al., 2021, Ilharco et al., 2021, Rafel et al., 2020, Esser et al., 2024]. Recent work uses DiT features for training-free subject injection, but MMDiT can still neglect or mix similar subjects [Feng et al., 2026, Wei et al., 2024]. V-Engram therefore targets concept-specific text and context states inside SD3.5, beyond an initial embedding or U-Net cross-attention interface.

## 3 Method

## 3.1 Overview

Figure 2 shows the workflow. A few references train external entries registered from a trigger such as <dog>. At inference, exact tokenizer-specific span matches retrieve these entries; unmatched prompts follow the frozen path. Retrieved entries enter selected hidden-state sites as gated residuals.

Let $G _ { \theta }$ be a pretrained Stable Difusion 3.5 pipeline with frozen text encoders, VAE, and MMDiT transformer. A concept � is specified by a few reference images $\mathcal { D } _ { c } ~ = ~ \{ x _ { i } ^ { c } \} _ { i = 1 } ^ { N _ { c } }$ and a trigger string $s _ { c }$ that appears in training prompts, e.g., “a photo of <dog>”. The trainable concept memory $\phi _ { c }$ consists of Engram directions and tanh gates attached to a set of mounted sites Q. Each site $q \ = \ ( g , \ell )$ denotes a tokenizer/conditioning stream $g$ and a hidden-state location ℓ, with hidden width $d _ { q } ;$ ; we write $g ( q )$ for the stream associated with site �. The backbone parameters � are never updated.

## 3.2 Trigger-Indexed Memory Registry

Stable Difusion 3.5 uses three tokenizer streams, CLIP-L, OpenCLIP-G, and T5-XXL; denote this set by $\mathcal { G }$ Since the same trigger can tokenize diferently across streams, V-Engram builds a separate registry for each tokenizer. The registry for stream $g \in { \mathcal { G } }$ contains token sequences and the memory entries they address. For a prompt �, exact token-id scanning returns

$$
\mathcal { A } _ { g } ( p ) = \{ ( m , i ) : \tau _ { g } ( p ) _ { i : i + | \kappa _ { m } | } = \kappa _ { m } , \kappa _ { m } \in \mathcal { R } _ { g } \} ,\tag{1}
$$

where $\kappa _ { m }$ is a registered span, � its Engram entry, and � its start position. The same surface trigger may tokenize diferently across streams. The main configuration registers contiguous 2–4-grams and always retains the full trigger span; single-token triggers use exact spans. All exact matches, including overlaps, are active and summed below. This decomposition is intended to preserve local relations among constituent tokens in a multi-token trigger, a property not separately evaluated here. Lookup uses token ids rather than semantic or approximate retrieval, attention-based detection, or prompt rewriting. With no match, the conditioning path equals the frozen base model.

## 3.3 Gated Relative Residual Injection

For each registered entry � and mounted site $\boldsymbol { q } = ( \boldsymbol { g } , \ell )$ , V-Engram stores a trainable direction $e _ { m , q } \in \mathbb { R } ^ { d _ { q } }$ and a per-channel gate $a _ { m , q } \in \mathbb { R } ^ { d _ { q } }$ . Routing is determined by the trigger lookup; the gate modulates the signed residual strength and is not a separate routing network. Since naive additive memories can over-perturb frozen hidden states, especially when mounted at several transformer blocks, V-Engram injects a gated relative residual. Given the hidden-state sequence $H ^ { q }$ of length $T _ { q }$ at site $q ,$ , the residual produced by entry � is

$$
\begin{array} { l } { { \Delta _ { m , q } ( H ^ { q } ) = \rho ( H ^ { q } ) \left( \displaystyle \frac { e _ { m , q } } { \| e _ { m , q } \| _ { 2 } + \varepsilon } \odot \mathrm { t a n h } ( a _ { m , q } ) \right) , } } \\ { { \rho ( H ^ { q } ) = \mathrm { s g } \left( \displaystyle \frac { 1 } { T _ { q } } \sum _ { j = 1 } ^ { T _ { q } } \| H _ { j } ^ { q } \| _ { 2 } \right) . } } \end{array}\tag{2}
$$

The Engram direction supplies residual direction, the tanh gate provides bounded signed per-channel modulation, and the detached norm $\rho ( H ^ { q } )$ calibrates the current activation scale. Our single-vector directembedding baseline disables this entire path and injects the learned embedding without a gate, direction normalization, or relative scaling:

$$
\Delta _ { m , q } ^ { \mathrm { u n g a t e d } } = e _ { m , q } .\tag{3}
$$

For prompt $p ,$ let $C _ { j } ^ { q } ( \boldsymbol { p } )$ collect the matched entries whose trigger spans cover position $j$ at site $q .$ . An entry � belongs to $C _ { i } ^ { q } ( p )$ if its matched span in $\mathcal { A } _ { g ( q ) } ( p )$ covers token position $j .$ . Omitting the repeated argument of $\Delta _ { m , q } ( H ^ { q } )$ , the update is

$$
\tilde { H } _ { j } ^ { q } = H _ { j } ^ { q } + \sum _ { m \in C _ { j } ^ { q } ( p ) } \Delta _ { m , q } .\tag{4}
$$

This form covers both the full trigger span and the case where a trigger such as <dog> activates several registered �-gram entries over the same token region. For CLIP-L and OpenCLIP-G, the injection is mounted before the penultimate transformer block, so downstream CLIP states used by Stable Difusion 3.5 are Engram-conditioned. For T5-XXL, it is mounted before the final encoder block and afects the final T5 hidden states.

## 3.4 Memory Mounting in Stable Difusion 3.5

V-Engram first attaches memory to the three text-encoder streams and also mounts memory inside selected MMDiT text/context blocks. We denote the selected MMDiT blocks by $\mathcal { L } _ { \mathrm { m m d i t } }$ . Unless otherwise specified, our default configuration uses seven evenly distributed blocks, $\mathcal { L } _ { \mathrm { m m d i t } } = \{ 0 , 6 , 1 2 , 1 9 , 2 5 , 3 1 , 3 7 \}$ . Encoderonly mounting and diferent MMDiT mounting densities are evaluated in the ablation study. After SD3.5 concatenates CLIP and T5 contexts, matched CLIP spans map to the CLIP context region and matched T5 spans map to the T5 region. We maintain tokenizer-specific ofsets when constructing this concatenated context sequence, so each trigger match is injected only into the corresponding CLIP or T5 token region. The pooled projection path is left unchanged. This gives a simple way to vary how deeply concept memory enters the multimodal denoising computation without modifying the pretrained transformer weights.

## 3.5 Training Objective and Storage

V-Engram stores external memory outside the pretrained model as small state dictionaries: three text-encoder Engram banks and their gate banks, plus selected MMDiT text/context banks. The trigger configuration records the trigger strings, tokenizer-specific registered targets, mounting sites, and gate setting. At inference time, the registry identifies exact prompt-matched entries, and only these entries are activated. Multiple concepts are composed by matching multiple triggers and summing all matched residuals; no LoRA adapters are merged and no backbone weights are modified.

For a concept �, let $K _ { c , g } = | \mathcal { M } _ { c , g } |$ be the number of entries registered in stream �. The trainable storage scales with registered entries and mounted sites:

$$
| \phi _ { c } | _ { \mathrm { p a r a m s } } = 2 \sum _ { q \in Q } K _ { c , g ( q ) } d _ { q } ,\tag{5}
$$

where $\mathcal { M } _ { c , g }$ is the set of entries registered for concept � in stream � and the factor 2 corresponds to one direction vector and one gate vector. The ungated direct-embedding variant stores only the embedding vectors, giving $\begin{array} { r } { \sum _ { q \in Q } K _ { c , g ( q ) } d _ { q } } \end{array}$ parameters.

Training freezes all pretrained components and updates only Engram directions and gates. For a clean latent $z _ { 0 } .$ , Gaussian noise �, and flow time �, define $z _ { t } = ( 1 - t ) z _ { 0 } + t \epsilon$ and the SD3.5 rectified-flow target $\nu = \epsilon - z _ { 0 }$ . The objective is

$$
\mathcal { L } _ { \mathrm { f l o w } } = \mathbb { E } _ { z _ { 0 } , \epsilon , t , p } \left[ \left\| F _ { \theta } ( z _ { t } , t , p ; \phi _ { \mathcal { R } ( p ) } ) - \nu \right\| _ { 2 } ^ { 2 } \right] ,\tag{6}
$$

where $\begin{array} { r } { \mathcal { A } ( p ) = \bigcup _ { g \in \mathcal { G } } \mathcal { A } _ { g } ( p ) } \end{array}$ denotes the active prompt-matched Engram entries across the three tokenizer streams. Prompts without active triggers use the unmodified frozen conditioning path.

## 4 Experiments

## 4.1 Experimental Setup

Benchmarks and references. The main benchmark contains 15 subjects and 80 Google DreamBooth references [Ruiz et al., 2023]. Same-class disambiguation uses seven dogs and 37 references from the same dataset. For familiar-identity refinement, we select 15 Celebrity-1000 identities [tonyassi, 2026] and form, for each identity, a five-image training set and a five-image-disjoint test set with visibly varied pose, hairstyle, illumination, and setting.

Baselines and training. We compare frozen Stable Difusion 3.5, TI-style embedding learning [Gal et al., 2023], joint DreamBooth-LoRA, and V-Engram. All learned methods use the same references and joint 15-subject training. TI-style, both LoRA ranks, and V-Engram use 25k, 8k, and 20k steps at learning rates $1 0 ^ { - 3 } , 1 0 ^ { - 4 }$ , and $5 \times 1 0 ^ { - 4 }$ , respectively. All use batch size 4, AdamW, a constant schedule, bfloat16, and seed 42. By default, V-Engram uses gated memories in the text encoders and seven MMDiT blocks {0, 6, 12, 19, 25, 31, 37}. Rank 4 (5.895M parameters) is the fixed LoRA configuration for controlled analyses; rank 8 (11.790M) is additionally evaluated in the main and contextual comparisons as a capacity check near the 10.477M full V-Engram registry. All qualitative figures containing LoRA outputs use rank 4.

Inference and metrics. To isolate subject fidelity, the main comparison uses “a photo of <trigger>”, with the subject-specific address substituted for <trigger>. Seeds 42–46 yield five $7 6 8 \times 7 6 8$ images per subject using 20 steps, guidance 4.5, the default scheduler, no negative prompt, and bfloat16. CLIP-I and DINOv2 use CLIP-L/14 and DINOv2-base cosine similarity [Radford et al., 2021, Oquab et al., 2024]. Best-reference takes each generation’s maximum over its references; all-pairs averages the complete generation–reference matrix. Both are averaged within subject and then macro-averaged, respectively tolerating viewpoint variation and measuring full-reference consistency. CLIP-T encodes the exact contextual prompt, including its anglebracketed trigger. For familiar identities, we compute cosine similarity between ArcFace-style face embeddings [Deng et al., 2019] extracted with InsightFace’s buffalo\_l model pack, comparing generations only against held-out references and reporting valid-face coverage separately. Further details are supplementary.

## 4.2 Main Comparison

Table 1 asks whether trigger-indexed memory can match weight adaptation while improving over input-level embedding learning.
<table><tr><td></td><td>Avg. best-ref</td><td></td><td>Avg. all-pairs</td></tr><tr><td>Method</td><td colspan="3">DINOv2 ↑ CLIP-I↑ DINOv2 ↑</td></tr><tr><td>Zero-shot SD3.5</td><td>0.4274</td><td>0.6765</td><td>0.3501 0.6361</td></tr><tr><td>TI-style</td><td>0.4919</td><td>0.7188</td><td>0.4071 0.6813</td></tr><tr><td>DB-LoRA (r = 4)</td><td>0.8084</td><td>0.8682</td><td>0.6911 0.8290</td></tr><tr><td>DB-LoRA (r = 8)</td><td>0.8167</td><td>0.8875</td><td>0.6930 0.8439</td></tr><tr><td>V-Engram</td><td>0.7896</td><td>0.8902</td><td>0.6809 0.8463</td></tr></table>

Table 1: Main comparison over 15 subjects. Both reference aggregations are averaged over five generations and then macro-averaged over subjects.

Figure 3 illustrates the rank-4 qualitative comparison for three representative subjects.  
![](images/47610ef23d01eaf114e20ec2c285415bdf2cf1ad8f364f139aa385a757982703.jpg)  
Figure 3: Qualitative main comparison under the common prompt template “a photo of <trigger>”. Rows show the personalized clock, fancy boot, and gray sloth plushie; columns show one reference image, zero-shot SD3.5, TI-style / embedding-only personalization, rank-4 DreamBooth-LoRA, and V-Engram.

The base and TI-style outputs frequently drift at the instance level, whereas rank-4 LoRA and V-Engram more faithfully recover distinguishing appearance, including the clock’s twin-bell silhouette and yellow numeral motif, the boot’s fringe structure, and the gray sloth plushie’s shape and facial details. Increasing LoRA from rank 4 to 8 raises best-reference CLIP-I by 0.0192 and DINOv2 by 0.0082. Against the parameter-near rank-8 adapter, V-Engram is within 0.0028 on both CLIP-I aggregations, while rank-8 LoRA leads DINOv2 by 0.0271 best-reference and 0.0121 all-pairs. Taken together, the two methods achieve broadly comparable subject fidelity under the simple prompt: V-Engram essentially matches rank-8 LoRA in CLIP-I, whereas LoRA is higher in DINOv2, so neither dominates across visual representations.

Resource scaling. We measure addressability as the registry grows. Joint rank-4 LoRA remains fixed at 5.895M parameters; nested V-Engram registries with 1, 7, and 9 triggers contain 0.639M, 4.985M, and 6.221M, crossing LoRA at nine. In a controlled A40 profile (10 warmup and 100 measured steps), V-Engram uses 0.940–1.004 GiB less peak allocated training memory but takes 14.9–29.0% longer per step. For a single-trigger query to a 15-trigger checkpoint, matched loading activates 1.061M rather than 10.477M Engram parameters, avoiding 89.87% of full-registry adaptation-state loading. This enables selective adaptation-state management without loading unrelated memories. Per-setting measurements are supplementary.

## 4.3 Ablation Study

We isolate injection depth and the gated residual. The density study jointly trains five concepts (28 references); all variants use 10k steps, five seeds per concept, and fixed settings. Encoder-only retains three text-encoder sites; the others additionally use � = 3–9 uniformly distributed MMDiT blocks.

N=5
<table><tr><td>Mounting setting</td><td>DINOv2↑</td><td>CLIP-I ↑</td></tr><tr><td>Encoder only</td><td>0.6132</td><td>0.7991</td></tr><tr><td>N = 3</td><td>0.7612</td><td>0.8499</td></tr><tr><td>N = 4</td><td>0.7814</td><td>0.8719</td></tr><tr><td>N = 5</td><td>0.7595</td><td>0.8820</td></tr><tr><td>N = 6</td><td>0.7780</td><td>0.8894</td></tr><tr><td>N = 7</td><td>0.8068</td><td>0.9007</td></tr><tr><td>N = 8</td><td>0.7345</td><td>0.8692</td></tr><tr><td>N = 9</td><td>0.8058</td><td>0.8798</td></tr></table>

Table 2: Mounting-density ablation. � is the number of MMDiT blocks in addition to encoder sites.

Figure 4 shows the corresponding qualitative variation.

![](images/4c3b3330d3bebaa244b25f2fdabaeba53f676a560faf20f327a1ebe97af47cad.jpg)

![](images/52fef4fdb61362b8e1503ac14aef0c907238424c41472f0f75e168de5cfc7c2c.jpg)  
Encoder-only

![](images/955f3858c96c7eeb35aa937b5316e847bf319fe4cd6acf3c53b59d5b839cb7e4.jpg)  
N=3

![](images/07b62e0d5526bb630e975f9b86d4cfdf1a2264767f0c75619420bd7b0f5d10bb.jpg)  
N=4

![](images/759932c052cd8ad25b357456c9a410e74dc81e5b1efa57b676b40a4beb10e923.jpg)  
Reference

![](images/b40b57c1a91bfa3e13478d3586514cbc46caad81b4d8daa759f58c902eba5b88.jpg)  
N=6

![](images/bbbcacc4160d4ee40fd6acb7c11f446744032b50114057ce9187e8311d99c914.jpg)  
N=7

![](images/f609c4cb3a274b54731ac411d1b10b883be7a359ea6008030c60a258824222cf.jpg)  
N=8

![](images/8d0a075c28628818721be77476cf6ef90913e6fc266df25662898132269b3025.jpg)  
N=9  
Figure 4: Qualitative mounting-density ablation. � denotes the number of selected MMDiT blocks. Compared with encoder-only injection, mounted variants more consistently preserve the subject’s distinctive shape and appearance, with visible variation across densities.

![](images/3845545ce751dea03328a9eaa9259d0452e37f747b7e33a2dce3e533264d2e93.jpg)

All mounted variants beat encoder-only on both metrics, while density is non-monotonic. Seven blocks perform best on this validation subset and define our default, without implying an architecture-independent optimum. A matched rerun compares TI-like direct single-vector injection with the complete V-Engram residual—normalized directions, hidden-state-relative scaling, and per-channel gates—under fixed data, schedule, and sites. The full design raises DINOv2 from 0.7065 to 0.8113 and CLIP-I from 0.8262 to 0.8892.

## 4.4 Additional Analysis

We next test held-out contexts, paired memories, unmatched routing, same-class instances, and refinement of imprecise prior knowledge.

Contextual Composition. We apply five held-out, category-agnostic templates—close-up, room, snow, garden, and beside a red car—to all 15 subjects at seed 42. Shared wording enables controlled comparison across objects and animals without category-specific prompt engineering. Reusing the main checkpoints gives 75 images per method; Figure 1 separately shows broader composition for one toy car.

<table><tr><td rowspan="2">Method</td><td>CLIP-T</td><td colspan="2">CLIP-I</td></tr><tr><td>Score ↑</td><td>Best-ref ↑</td><td>All-pairs ↑</td></tr><tr><td>DB-LoRA (r = 4)</td><td>0.2558</td><td>0.8264</td><td>0.7888</td></tr><tr><td>DB-LoRA (r = 8)</td><td>0.2658</td><td>0.8105</td><td>0.7706</td></tr><tr><td>V-Engram</td><td>0.2334</td><td>0.8717</td><td>0.8261</td></tr></table>

Table 3: Contextual composition under five category-agnostic templates shared by all 15 subjects. Scores are macroaveraged over subjects.  
V-Engram  
Figure 5: Additional contextual prompts for a personalized cat, shown separately from the five-template protocol in Table 3. The reference is at left; the upper and lower rows show rank-4 DreamBooth-LoRA and V-Engram, respectively. Exact prompts and additional examples are provided in the supplementary material.

Table 3 shows that V-Engram better preserves personalized subjects in complex contexts, whereas rank-8 LoRA aligns text better. V-Engram leads rank 8 by 0.0612/0.0554 in best-reference/all-pairs CLIP-I (14/15 subjects), while rank 8 leads CLIP-T by 0.0324 (13/15). Raising LoRA from rank 4 to 8 improves CLIP-T by 0.0101 but reduces the two CLIP-I scores by 0.0158/0.0182; more capacity thus does not improve contextual fidelity. Figure 5 illustrates this trade-of.

Multi-Concept Composition. Figure 6 shows three of 20 matched seed-42 paired-trigger prompts against rank-4 LoRA. V-Engram more consistently depicts both concepts, while LoRA sometimes omits or blends one. We keep this comparison qualitative because full-image similarity cannot attribute retention to each subject; both methods still exhibit failures.

![](images/37a80d93d61eadb7a98ce39e7ea1c18fbf2f7835e5c30bc81818132241b3feb3.jpg)  
Figure 6: Paired-trigger examples: teapot+vase, cat+toy, and sunglasses+plushie. Columns show two references, V-Engram, and rank-4 DreamBooth-LoRA.

Non-Triggered Base-Behavior Preservation. Eight ordinary categories under seven templates yield 56 prompts without registered triggers. Exact-match logs show zero active entries in all three text streams; no residual is injected, so V-Engram exactly reduces to the frozen base under identical inference settings. Joint LoRA remains globally active and can shift outputs toward training instances. Supplementary examples visualize this diference.

Same-Class Trigger Disambiguation. Seven dogs (37 references) receive indexed triggers (<dog>, <dog2>, . . . ) or arbitrary names (<Oscar>, <Ryan>, . . . ). V-Engram trains for 10k steps and rank-4 LoRA for 4k. Seed-42 results cover a simple photo, running on grass, and sitting beside a red car; Figure 7 shows the red-car slice for four identities. Because its exact-match registry accepts user-defined triggers, V-Engram is not tied to one naming convention; indexed and name-based triggers are the two forms tested here, and both better preserve dog-specific cues. LoRA tends toward similar dogs with indexed triggers and can invoke human priors with names. Without a calibrated dog-identity recognizer, this remains qualitative.

Refining Familiar Concepts. Finally, we test calibration of imprecise prior knowledge on 15 identities. Each condition generates five images per identity, compared only with five held-out references. The image-disjoint splits contain photographs from diferent times and capture conditions, rewarding recurring identity cues rather than training-image reconstruction.

![](images/f5dca4f8826e9e3ebc258bbb882bd8bc272d465ca42d97a25bdc4a1d0c7daa93.jpg)

Figure 7: Same-class disambiguation under indexed and name-based triggers for the red-car prompt. Four of the seven dog identities are shown; rows compare V-Engram and rank-4 DreamBooth-LoRA under the two naming schemes.
<table><tr><td rowspan="2">Method</td><td colspan="2">InsightFace</td><td rowspan="2">Valid</td></tr><tr><td>Best-ref ↑</td><td>All-pairs ↑</td></tr><tr><td>Zero-shot Stable Diffusion 3.5</td><td>0.2344</td><td>0.1824</td><td>75/75</td></tr><tr><td>DB-LoRA (r = 4)</td><td>0.4618</td><td>0.4014</td><td>75/75</td></tr><tr><td>V-Engram</td><td>0.6084</td><td>0.5415</td><td>75/75</td></tr></table>

Table 4: InsightFace identity similarity on 15 familiar identities. Scores are averaged over generations and macroaveraged over identities.

V-Engram exceeds the tested rank-4 LoRA by 0.1466 best-reference and 0.1401 all-pairs similarity, winning 13/15 identities; both adaptations beat the base on all 15. All 75 generations per method are valid detections. The image-disjoint split supports identity-level refinement beyond single-image memorization, with three held-out visual examples shown in Figure 8.

![](images/a5d3d5d72fe76654733082807509c472a62157ca4e8683a81d3f8d051d736d49.jpg)  
Figure 8: Qualitative familiar-identity refinement for three of the 15 evaluated identities. Columns show a held-out reference, zero-shot Stable Difusion 3.5, rank-4 DreamBooth-LoRA, and V-Engram.

## 5 Conclusion

We introduced V-Engram, a trigger-indexed external-memory interface that provides a frozen Stable Difusion 3.5 backbone with prompt-addressed Engram entries. Under simple subject prompts, V-Engram is broadly competitive with LoRA, with comparable CLIP-I but lower DINOv2; in held-out contexts, it preserves identity better, while LoRA favors prompt alignment. Exact �-gram routing leaves unmatched prompts on the frozen base path and supports selective memory activation and loading; further analyses support familiar-identity refinement, paired-trigger composition, and same-class separation. Overall, these findings show a diferent balance among fidelity, modularity, and compositional access rather than universal superiority over weight adaptation. More broadly, exact token matching may provide an addressing mechanism for external memory in other token-conditioned generative or multimodal architectures, although cross-architecture validation remains future work.

## References

Yuval Alaluf, Elad Richardson, Gal Metzer, and Daniel Cohen-Or. A neural space-time representation for text-to-image personalization. ACM Transactions on Graphics, 42(6):243:1–243:10, 2023. doi: 10.1145/3618322.

Omri Avrahami, Kfir Aberman, Ohad Fried, Daniel Cohen-Or, and Dani Lischinski. Break-a-scene: Extracting multiple concepts from a single image. In SIGGRAPH Asia 2023 Conference Papers, pages 96:1–96:12, 2023. doi: 10.1145/3610548.3618154.

Nan Chen, Mengqi Huang, Zhuowei Chen, Yang Zheng, Lei Zhang, and Zhendong Mao. Customcontrast: A multilevel contrastive perspective for subject-driven text-to-image customization. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 39, pages 2123–2131, 2025. doi: 10.1609/aaai.v39i2.32210.

Xi Chen, Lianghua Huang, Yu Liu, Yujun Shen, Deli Zhao, and Hengshuang Zhao. Anydoor: Zero-shot object-level image customization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6593–6602, 2024.

Xin Cheng, Rui Tian, Wangding Zeng, Damai Dai, Qinyu Chen, Bingxuan Wang, Zhenda Xie, Kezhao Huang, Xingkai Yu, Chengqi Deng, Shangyan Zhou, Chenggang Zhao, Zhewen Hao, Yukun Li, Han Zhang, Zhengyan Zhang, Yixu Wei, M. Y. Xu, Huishuai Zhang, Dongyan Zhao, and Wenfeng Liang. Conditional memory via scalable lookup: A new axis of sparsity for large language models. arXiv preprint arXiv:2601.07372, 2026.

Yongjin Choi, Chanhun Park, and Seung Jun Baek. DynASyn: Multi-subject personalization enabling dynamic action synthesis. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 2564–2572, 2025. doi: 10.1609/aaai.v39i3.32259.

Jiankang Deng, Jia Guo, Niannan Xue, and Stefanos Zafeiriou. ArcFace: Additive angular margin loss for deep face recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019.

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, and Robin Rombach. Scaling rectified flow transformers for high-resolution image synthesis. In Proceedings ofthe 41st International Conference on Machine Learning, volume 235, pages 12606–12633, 2024.

Haoran Feng, Zehuan Huang, Lin Li, and Lu Sheng. Personalize anything for free with difusion transformer. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 40, pages 3921–3929, 2026. doi: 10.1609/aaai.v40i5.37394.

Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H. Bermano, Gal Chechik, and Daniel Cohen-Or. An image is worth one word: Personalizing text-to-image generation using textual inversion. In International Conference on Learning Representations, 2023.

Yuchao Gu, Xintao Wang, Jay Zhangjie Wu, Yujun Shi, Yunpeng Chen, Zihan Fan, Wuyou Xiao, Rui Zhao, Shuning Chang, Weijia Wu, Yixiao Ge, Ying Shan, and Mike Zheng Shou. Mix-of-show: Decentralized low-rank adaptation for multi-concept customization of difusion models. In Advances in Neural Information Processing Systems, volume 36, 2023.

Ligong Han, Yinxiao Li, Han Zhang, Peyman Milanfar, Dimitris Metaxas, and Feng Yang. SVDif: Compact parameter space for difusion fine-tuning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7323–7334, 2023.

Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin de Laroussilhe, Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly. Parameter-eficient transfer learning for NLP. In Proceedings ofthe International Conference on Machine Learning, 2019.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022.

Qihan Huang, Siming Fu, Jinlong Liu, Hao Jiang, Yipeng Yu, and Jie Song. Resolving multi-condition confusion for finetuning-free personalized image generation. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 39, pages 3707–3714, 2025. doi: 10.1609/aaai.v39i4.32386.

Gabriel Ilharco, Mitchell Wortsman, Ross Wightman, Cade Gordon, Nicholas Carlini, Rohan Taori, Achal Dave, Vaishaal Shankar, Hongseok Namkoong, John Miller, Hannaneh Hajishirzi, Ali Farhadi, and Ludwig Schmidt. OpenCLIP. https://github.com/mlfoundations/open\_clip, 2021.

Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu. Multi-concept customization of text-to-image difusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023.

Guillaume Lample, Alexandre Sablayrolles, Marc’Aurelio Ranzato, Ludovic Denoyer, and Hervé Jégou. Large memory layers with product keys. In Advances in Neural Information Processing Systems, volume 32, 2019.

Brian Lester, Rami Al-Rfou, and Noah Constant. The power of scale for parameter-eficient prompt tuning. In Proceedings ofthe Conference on Empirical Methods in Natural Language Processing, 2021.

Dongxu Li, Junnan Li, and Steven C. H. Hoi. BLIP-difusion: Pre-trained subject representation for controllable text-to-image generation and editing. In Advances in Neural Information Processing Systems, volume 36, 2023.

Xiang Lisa Li and Percy Liang. Prefix-tuning: Optimizing continuous prompts for generation. In Proceedings of the Annual Meeting of the Association for Computational Linguistics, 2021.

Zhen Li, Mingdeng Cao, Xintao Wang, Zhongang Qi, Ming-Ming Cheng, and Ying Shan. Photomaker: Customizing realistic human photos via stacked ID embedding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8640–8650, 2024.

Haokun Liu, Derek Tam, Mohammed Muqeeth, Jay Mohta, Tenghao Huang, Mohit Bansal, and Colin Rafel. Few-shot parameter-eficient fine-tuning is better and cheaper than in-context learning. In Advances in Neural Information Processing Systems, 2022.

Zhiheng Liu, Ruili Feng, Kai Zhu, Yifei Zhang, Kecheng Zheng, Yu Liu, Deli Zhao, Jingren Zhou, and Yang Cao. Cones: Concept neurons in difusion models for customized generation. In Proceedings ofthe International Conference on Machine Learning, pages 21548–21566, 2023.

Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel HAZIZA, Francisco Massa, Alaaeldin El-Nouby, Mahmoud Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Herve Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. DINOv2: Learning robust visual features without supervision. Transactions on Machine Learning Research, 2024.

Lianyu Pang, Jian Yin, Baoquan Zhao, Feize Wu, Fu Lee Wang, Qing Li, and Xudong Mao. Attndreambooth: Towards text-aligned personalized text-to-image generation. In Advances in Neural Information Processing Systems, volume 37, 2024. doi: 10.52202/079017-1259.

William Peebles and Saining Xie. Scalable difusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023.

Yuqi Peng, Lingtao Zheng, Yufeng Yang, Yi Huang, Mingfu Yan, Jianzhuang Liu, and Shifeng Chen. TARA: Token-aware LoRA for composable personalization in difusion models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 8385–8393, 2026. doi: 10.1609/aaai.v40i10.37788.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In Proceedings of the International Conference on Machine Learning, 2021.

Colin Rafel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal ofMachine Learning Research, 21(140):1–67, 2020.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High-resolution image synthesis with latent difusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022.

Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image difusion models for subject-driven generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023.

Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Wei Wei, Tingbo Hou, Yael Pritch, Neal Wadhwa, Michael Rubinstein, and Kfir Aberman. Hyperdreambooth: Hypernetworks for fast personalization of text-to-image models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6527–6536, 2024.

Yoad Tewel, Rinon Gal, Gal Chechik, and Yuval Atzmon. Key-locked rank one editing for text-toimage personalization. In ACM SIGGRAPH 2023 Conference Proceedings, pages 1–11, 2023. doi: 10.1145/3588432.3591506.

tonyassi. Celebrity-1000. https://huggingface.co/datasets/tonyassi/celebrity-1000, 2026. Hugging Face dataset, accessed July 2026.

Qixun Wang, Xu Bai, Haofan Wang, Zekui Qin, Anthony Chen, Huaxia Li, Xu Tang, and Yao Hu. InstantID: Zero-shot identity-preserving generation in seconds. arXiv preprint arXiv:2401.07519, 2024.

Tianyi Wei, Dongdong Chen, Yifan Zhou, and Xingang Pan. Enhancing MMDiT-based text-to-image models for similar subject generation. arXiv preprint arXiv:2411.18301, 2024.

Yuxiang Wei, Yabo Zhang, Zhilong Ji, Jinfeng Bai, Lei Zhang, and Wangmeng Zuo. ELITE: Encoding visual concepts into textual embeddings for customized text-to-image generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15943–15953, 2023.

Feize Wu, Yun Pang, Junyi Zhang, Lianyu Pang, Jian Yin, Baoquan Zhao, Qing Li, and Xudong Mao. CoRe: Context-regularized text embedding learning for text-to-image personalization. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 8377–8385, 2025. doi: 10.1609/aaai.v39i8.32904.

Yuhuai Wu, Markus N. Rabe, DeLesley Hutchins, and Christian Szegedy. Memorizing transformers. In International Conference on Learning Representations, 2022.

Guangxuan Xiao, Tianwei Yin, William T. Freeman, Frédo Durand, and Song Han. Fastcomposer: Tuning-free multi-subject image generation with localized attention. International Journal ofComputer Vision, 133: 1175–1194, 2025. doi: 10.1007/s11263-024-02227-z.

Zebin Yao, Fangxiang Feng, Ruifan Li, and Xiaojie Wang. Concept conductor: Orchestrating multiple personalized concepts in text-to-image synthesis. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 9427–9435, 2025. doi: 10.1609/aaai.v39i9.33021.

Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. IP-adapter: Text compatible image prompt adapter for text-to-image difusion models. arXiv preprint arXiv:2308.06721, 2023.

Yuxin Zhang, Weiming Dong, Fan Tang, Nisha Huang, Haibin Huang, Chongyang Ma, Tong-Yee Lee, Oliver Deussen, and Changsheng Xu. Prospect: Prompt spectrum for attribute-aware personalization of difusion models. ACM Transactions on Graphics, 42(6):244:1–244:14, 2023. doi: 10.1145/3618342.

Yuxuan Zhang, Yiren Song, Jiaming Liu, Rui Wang, Jinpeng Yu, Hao Tang, Huaxia Li, Xu Tang, Yao Hu, Han Pan, and Zhongliang Jing. SSR-Encoder: Encoding selective subject representation for subject-driven generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8069–8078, 2024.