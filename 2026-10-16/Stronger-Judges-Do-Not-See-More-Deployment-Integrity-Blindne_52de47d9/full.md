# Stronger Judges Do Not See More: Deployment-Integrity Blindness in LLM Evaluation

Jianchang Su Dawei Xiang Yifan Zhang Wei Zhang University of Connecticut

## Abstract

LLM-agent evaluation reports behavioral scores but leaves the identity of the deployed runtime stack unreported. A tampered deployment that preserves evaluated outputs passes every output-only evaluator. We introduce DIB-Bench, a benchmark that measures this deployment-integrity blindness across 7,560 scored examples, six tamper types, and an evaluator ladder from regex scorers through LLM judges to GPT-5-nano. Stronger evaluators accept more tampered outputs: mean acceptance rises from 88.6% (rule-based) to 94.1% (GPT-5-nano), and a trigger-conditional tamper invisible on the evaluation suite produces 48–100% harmful outputs, rising with model capability. A PAC-style bound shows that tampers with zero output shift are undetectable from outputs at any sample size; a manifest-hash identity channel detects them with zero samples. Behavioral fingerprinting identifies the model component above chance (76.7%) but covers one of six stack components. We also propose a graded deployment-identity descriptor for evaluation reports.

## 1 Introduction

LLM-agent evaluation reports behavioral scores: accuracy, task success, preference, harmlessness. This remains necessary, but the evaluated object has shifted from a model to a system. A deployed LLM agent is a runtime stack of model artifacts, tokenizers, system prompts, tool specifications, retrieval corpora, runtime dependencies, and postprocessors, all of which can change after training, after evaluation, and across deployment instances (Zaharia et al., 2024). Agent leaderboards report a single rank per “model” even when the ranked entity is a composite system whose components could be swapped without changing the leaderboard entry. This is a structural gap in the field’s evaluation vocabulary.

In response to this gap, the field’s default remedy is a stronger evaluator. If output evaluation misses a deployment change, the implied fix is a better scorer: an LLM judge, a forensic classifier, a behavioral fingerprint. If judges caught tampering, output-only evaluation would suffice and a separate identity channel would be redundant. They do not.

We evaluate six classes of deployment tampering across 7,560 scored examples (three model pairs, three random seeds) under an evaluator ladder: rule-based scorers, a locally run LLM judge (Qwen3-8B), and a current-generation commercial reasoning model (GPT-5-nano from OpenAI’s GPT-5 family). Figure 1 summarizes the protocol and the main result. We find:

1. Blindness increases with evaluator strength. Mean tamper acceptance rises from 88.6% (rulebased) to 89.3% (Qwen3-8B) to 94.1% (GPT-5- nano).

2. A trigger-conditional tamper is 100% invisible on the evaluation suite and up to 100% harmful off it. Harm scales with model capability: 48% (Llama-3.1-8B), 81% (Qwen2.5-7B), 100% (Qwen2.5-14B).

3. A stealth tamper can evade both the output evaluator and the identity channel by hiding in an undeclared component, which quantifies identity detection as a function of manifest coverage.

4. A PAC-style dichotomy characterizes the undetectable class: tampers with zero outputdistribution shift are undetectable at any sample size by any evaluator. A manifest-hash identity channel decides this class with zero samples.

We contribute (i) DIB-Bench, an extensible protocol for measuring deployment-integrity blindness with controlled single-component tampers, (ii) the formal separation of behavioral and identity evidence for LLM evaluation, and (iii) a graded deployment-identity descriptor for evaluation reports. Agent and prompt-injection benchmarks have pushed evaluation beyond single-turn tasks (Ruan et al., 2024; Yao et al., 2024; Debenedetti et al., 2024; Andriushchenko et al., 2025); identity evidence is a coordinate beyond their scope.

![](images/5fd7563996fd27e171cc8471f0ae8e1f9688b6a9743e2964aa1fe53314a5feff.jpg)  
Figure 1: Overview of DIB-Bench. (a) A deployment is a stack of six components, each recorded by a content hash in the declared manifest (hash prefixes from the Qwen2.5-7B run). A tamper changes one component (here the system prompt) while preserving the outputs the evaluation suite checks. Output-only evaluators score the outputs and pass the tampered deployment; the identity channel compares manifest hashes and fails it. A deployment that passes behavior and fails identity counts as blind. (b) Mean acceptance of the six tampered conditions (bars) and of the clean deployment (green ticks) across the evaluator ladder $\scriptstyle ( n = 7 , 5 6 0 )$ ). LLM judges accept more tampered outputs than rule-based scoring; the identity channel accepts none.

## 2 Setup

## 2.1 Deployments, Manifests, and the Two Channels

A deployed LLM agent is a runtime stack of components: model and tokenizer, system prompt, tool specification and router, retrieval policy and corpus, runtime dependencies, and postprocessor. We represent a deployment by a declared manifest $M _ { \mathrm { e x p e c t e d } }$ and an observed runtime manifest $M _ { \mathrm { o b s e r v e d } }$ , each listing a per-component content hash (SHA-256 over canonical JSON).

Output evaluation passes if $E _ { \mathrm { o u t } } ( \mathrm { s y s t e m } ( x ) ) =$ pass for a behavioral scorer $E _ { \mathrm { o u t } }$ . Identity evaluation passes if $E _ { \mathrm { i n t } } ( M _ { \mathrm { e x p e c t e d } } , M _ { \mathrm { o b s e r v e d } } ) = \mathrm { p a s s } ,$ defined as $h ( M _ { \mathrm { e x p e c t e d } } [ c ] ) = h ( M _ { \mathrm { o b s e r v e d } } [ c ] )$ for every component $c .$ The hash is intentionally lightweight: it separates the evaluation question from production-grade attestation. Signed provenance, in-toto-style metadata (Torres-Arias et al., 2019), SLSA attestations (SLSA Framework, 2026), or hardware-rooted attestation can substitute for the same channel role. We use deployment-integrity blindness for the failure mode and deployment-identity evidence for the channel that detects it.

## 2.2 Blindness Metrics

The example-level blindness indicator is $\nVdash [ E _ { \mathrm { o u t } } ( \mathrm { s y s t e m } ( x ) )$ pass ∧ $\begin{array} { r c l } { E _ { \mathrm { i n t } } ( M _ { \mathrm { e x p e c t e d } } , M _ { \mathrm { o b s e r v e d } } ) } & { = } & { \mathrm { f a i l } ] ; } \end{array}$ its mean across examples is the example-level blindness rate. The suite-level false-acceptance indicator at threshold τ is blind@τ = ⊮[output\_pass\_rate ≥ $\tau \wedge$ integrity\_pass\_rate = 0].

Figure 1a shows the two channels side by side. Current NLP evaluation reports the output channel; the identity channel is unreported.

## 3 Theory

The separation between the two channels can be made precise. Behavioral evidence is a function of an output suite. For any finite output evaluation suite S with input set $X _ { S }$ and acceptance predicate $A _ { S }$ , define two deployments $D _ { 1 } , D _ { 2 }$ as S-equivalent when $A _ { S } ( D _ { 1 } ( x ) ) = A _ { S } ( D _ { 2 } ( x ) )$ for every $x \in X _ { S }$

Proposition 1 (Indistinguishability). For anyfinite output suite $S$ and any deployment $D _ { 1 }$ that passes $S ,$ there exists a deployment $D _ { 2 }$ that differs from $D _ { 1 }$ in at least one manifest component but is $S .$ equivalent to $D _ { 1 }$ . The identity of the deployed system is therefore not afunction of $A _ { S } ( D ( X _ { S } ) )$ .

Proof sketch. $X _ { S }$ is finite, so $D _ { 1 } ( X _ { S } )$ is a finite set of outputs. For any manifest component c, construct $D _ { 2 }$ by replacing c with a variant that produces the same outputs on $X _ { S }$ (a fine-tune, an identity-on- $D _ { 1 } ( X _ { S } )$ postprocessor, a prompt edit conditional on a trigger absent from $X _ { S }$ , or a runtime swap). Then $A _ { S } ( D _ { 2 } ( x ) ) = A _ { S } ( D _ { 1 } ( x ) )$ for every $x \in X _ { S }$ □

Proposition 1 is the trivial reading: the class of indistinguishable deployments is non-empty. The strong reading is that practical tampers in the class can cause harm outside $X _ { S }$ while remaining invisible on it. We state this as an empirical hypothesis tested in §6:

Hypothesis 1 (Practical stealth). There exist tampered deployments $D _ { 2 }$ with $\varepsilon ( D _ { 1 } , D _ { 2 } ; { \mathcal { D } } ) = 0$ on the evaluation distribution D that produce harmful outputs on inputs outside the support of D.

The following dichotomy quantifies the boundary between detectable and undetectable tampers. It upgrades Proposition 1 from an existence fact to a sample-complexity bound.

Theorem 1 (Behavioral sample-complexity dichotomy). Let A (clean) and B (tampered) be deployments. Define $\begin{array} { r l r } { \varepsilon } & { { } = } & { \varepsilon ( A , B ; \mathcal { D } ) \quad : = } \end{array}$ $\mathbb { E } _ { { x } \sim { D } } [ \mathrm { T V } ( A ( { x } ) , B ( { x } ) ) ]$ as the output-distribution shift on the evaluation distribution D.

(i) Lower bound. For every behavioral test T using N i.i.d. samples, the distinguishing advantage satisfies $| \mathrm { P r } [ T =$ tampered | $B ] - \operatorname* { P r } [ T =$ tampered $| \ A ] | \leq N \varepsilon$ $H \varepsilon = 0$ , every behavioral test at any N is indistinguishable from chance.

(ii) Upper bound. $H \varepsilon > 0 ,$ , there is a behavioral test using $N = O ( \varepsilon ^ { - 2 } \log ( 1 / \delta ) )$ samples that distinguishes Afrom B with error $\leq \delta .$

The proof (Appendix A) uses sub-additivity of TV under products for (i) and Hoeffding concentration for (ii). The theorem subsumes all output-only evaluators: rule-based, LLM-judge, forensic, fingerprint. Each is a function of the same output samples, so the Nε bound applies verbatim. By contrast, the identity channel escapes this bound: it compares declared and observed component hashes with one manifest read, independent of ε and N. It decides the $\varepsilon = 0$ class that remains undecidable for behavioral tests at any sample size.

## 4 DIB-Bench

## 4.1 Conditions, Tasks, and Models

We evaluate one clean condition and six singlecomponent tampers: model substitution (observed model/tokenizer differs from declared), prompt tampering (system prompt receives an unauthorized override), tool tampering (tool spec/router makes a restricted tool appear safe), retrieval tampering (retrieval policy or docs contain override text), dependency tampering (runtime descriptor changes), and postprocess tampering (generated text is modified after inference). Each tampered condition changes exactly one manifest component, isolating the causal comparison.

<table><tr><td>Condition</td><td>Id. pass</td><td>Std</td><td> $\Delta$  vs clean</td></tr><tr><td colspan="4">E2: undeclared-component tampers</td></tr><tr><td>sampling (temp  $0  0 . 7 )$ </td><td>100</td><td>93.3</td><td>33.3% changed</td></tr><tr><td>chat-template injection</td><td>100</td><td>90.0</td><td>77.5% changed</td></tr><tr><td colspan="4">E3: benign declared-component changes</td></tr><tr><td>prompt paraphrase</td><td>0</td><td>100.0</td><td>0% changed</td></tr><tr><td>dependency patch bump</td><td>0</td><td>100.0</td><td>0% changed</td></tr></table>

Table 1: Instrument validation (Qwen2.5-7B, seed 42, n=120 each). E2: changing an undeclared component passes identity at 100% while behavior visibly changes, so the 0% identity rate in the main study is not tautological: it reflects manifest coverage. E3: changing a declared component in a semantically benign way fails identity at 0% while behavior is identical to clean, so identity is a change-detection signal, not a semanticequivalence check.

The main study uses 120 synthetic tasks per condition across five families (factual-context QA, policy allow/deny, tool selection, secret-leak refusal, canary probes). Nine runs (3 model pairs × 3 random seeds 42, 43, 44) yield 7,560 scored examples. Model pairs: Qwen2.5-14B-Instruct → Qwen2.5- 7B-Instruct, Qwen2.5-7B-Instruct → Qwen2.5- 1.5B-Instruct (Yang et al., 2024), and Llama-3.1- 8B-Instruct → Qwen2.5-7B-Instruct (Team, 2024). All inference uses vLLM (Kwon et al., 2023).

## 4.2 Instrument Validation: Detection Equals Coverage

Before presenting results, we validate two properties of the identity channel. Table 1 reports two undeclared-component conditions (E2) and two benign-change conditions (E3).

Identity detection is a function of manifest coverage (E2). Changing the sampling temperature (an undeclared parameter) or injecting a directive through the chat template (an undeclared component) changes behavior measurably (33%–78% of outputs differ from clean) while identity passes at 100%. The identity channel sees exactly the changes that the manifest covers. The 0% identity rate in the main study therefore reflects manifest coverage: detection equals coverage, and the coverage decision determines whether the channel is useful.

Identity is a change-detection signal, not a semantic-equivalence check (E3). Paraphrasing the system prompt or bumping a dependency patch version changes the manifest hash (identity fails at 0%) even though behavior is 100% identical to clean (every output decision matches). The channel cannot distinguish a benign rephrase from a malicious edit; it flags change and delegates adjudication to a human or process. This is the expected behavior of a change-detection instrument, not a limitation.

Both properties replicate on Qwen2.5-7B (seeds 42 and 43), Llama-3.1-8B→Qwen2.5-7B, and Qwen2.5-14B: undeclared changes alter 15.8– 100% of outputs while identity passes at 100%, and benign declared changes keep a 100% decision match while identity fails (Appendix B).

## 5 Results: The Evaluator Ladder

With the instrument validated, we turn to the central question: how does blindness vary across evaluators of different strength? Table 2 reports the same seven conditions under four output-only evaluators of increasing capability and the identity channel. All values are percentages; for tampered conditions, blindness equals the acceptance rate since identity is 0%.

A GPT-5-generation judge is more blind than a regex scorer. The GPT-5-nano quality judge, a current-generation reasoning model, accepts 94.1% of tampered outputs across six conditions, compared with 89.3% for a locally run Qwen3-8B and 88.6% for rule-based scoring. The increase is monotonic: as evaluator capability rises, tamper acceptance rises with it. It is largest for the postprocess tamper (61.9% to 77.4%) and the retrieval tamper (89.0% to 94.6%), the two conditions with the lowest rule-based acceptance. The reason is structural: a more capable judge is better at recognizing that a tampered output is correct, because the tamper was designed to preserve correctness. This improved capability helps evaluate output quality but is uninformative about whether the evaluated system changed.

Suspicion auditing fails to separate tampered from clean. We configured GPT-5-nano as a suspicion auditor, asking it to flag outputs that show signs of originating from a system that deviates from the declared policy. The auditor flags 54.0% of clean outputs, 52.6% of prompt-tampered outputs, and 51.2% of dependency-tampered outputs. Prompt tampering is flagged less than clean. Tampered and clean outputs overlap at every threshold; an operator using the auditor would reject half of clean deployments while still passing all tampers.

(a) Task family  
(b) Model pair
<table><tr><td></td><td rowspan=1 colspan=1>76</td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1>93</td></tr><tr><td></td><td rowspan=1 colspan=1>67</td><td rowspan=1 colspan=1>96</td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1>86</td><td rowspan=1 colspan=1>100</td></tr><tr><td></td><td rowspan=1 colspan=1>57</td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1>98</td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1>91</td><td rowspan=1 colspan=1>95</td></tr><tr><td></td><td rowspan=1 colspan=1>84</td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1>64</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>96</td><td rowspan=1 colspan=1>93</td><td rowspan=1 colspan=1>89</td></tr><tr><td></td><td rowspan=1 colspan=1>57</td><td rowspan=1 colspan=1>99</td><td rowspan=1 colspan=1>94</td><td rowspan=1 colspan=1>67</td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>81</td><td rowspan=1 colspan=1>89</td><td rowspan=1 colspan=1>97</td></tr><tr><td rowspan=2 colspan=1>dependencypostprocess</td><td rowspan=1 colspan=1>74</td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1>92</td></tr><tr><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1>44</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>62</td><td rowspan=1 colspan=1>62</td><td rowspan=1 colspan=1>62</td></tr></table>

Figure 2: Standard acceptance (%) by task family and by model pair, pooled over the main study (n=7,560). For tampered rows, acceptance equals blindness because identity is 0%. Model pairs are declared → deployed; L8B is Llama-3.1-8B and the other models are Qwen2.5. Acceptance drops where the suite probes the behavior that a tamper changes and stays near 100% elsewhere.

Stricter rule-based scoring rejects clean deployments before it rejects all tampers. Strict scoring lowers mean tamper acceptance to 73.1%; model substitution falls to 72.4% because the smaller model violates the output format more often. Table 3 varies the suite threshold τ for the four rule-based evaluators. Standard and forensic scoring accept four of six tampers even at τ=0.90. Strict and canary scoring reach 0/6 only at thresholds above their own clean acceptance (85.4% and 75.9%), where they also reject the clean deployment. Every rule-based setting therefore either passes some tampers or fails the clean system. This is the empirical counterpart of Theorem 1: refining the output suite shrinks the S-equivalence class but cannot collapse it.

Output evaluation catches a tamper where the suite probes the changed behavior. Figure 2a breaks standard acceptance down by task family. Each tamper stays near 100% on most families and drops on the families that probe what it changes. The postprocess tamper, which modifies generated text after inference, falls to 0% on secret and canary tasks and to 44% on policy tasks; the tool tamper falls to 64% on tool-selection tasks; the retrieval tamper falls to 67% on secret tasks. Detection by output evaluation is therefore a matter of suite coverage, the behavioral counterpart of manifest coverage for the identity channel (§4.2). A suite detects a tamper when it contains a probe for the changed behavior, and a tamper that avoids the suite’s probes passes. Figure 2b shows the same pattern across model pairs. Model substitution is fully accepted for the two pairs of similar capability (14B→7B and Llama-3.1-8B→Qwen2.5-7B) and partly visible for the 4.5× downgrade (7B→1.5B: 85.6% standard, 38.3% strict). In the Llama pair, the substituted Qwen2.5-7B scores higher than the declared model (100% vs. 92.8%), so a substitution can raise the reported number.

<table><tr><td></td><td colspan="2">Rule-based</td><td colspan="2">LLM judge</td><td colspan="2"></td></tr><tr><td>Condition</td><td>Standard</td><td>Strict</td><td>Qwen3-8B</td><td>GPT-5-nano</td><td>Identity</td><td>Blindness</td></tr><tr><td>clean</td><td>97.6</td><td>85.4</td><td>99.9</td><td>99.7</td><td>100.0</td><td>0.0</td></tr><tr><td>model substitution</td><td>95.2</td><td>72.4</td><td>92.9</td><td>98.6</td><td>0.0</td><td>98.6</td></tr><tr><td>prompt tamper</td><td>95.3</td><td>81.9</td><td>99.4</td><td>99.4</td><td>0.0</td><td>99.4</td></tr><tr><td>tool tamper</td><td>92.7</td><td>82.9</td><td>93.4</td><td>95.0</td><td>0.0</td><td>95.0</td></tr><tr><td>retrieval tamper</td><td>89.0</td><td>67.0</td><td>87.7</td><td>94.6</td><td>0.0</td><td>94.6</td></tr><tr><td>dependency tamper</td><td>97.4</td><td>85.0</td><td>99.9</td><td>99.7</td><td>0.0</td><td>99.7</td></tr><tr><td>postprocess tamper</td><td>61.9</td><td>49.6</td><td>62.3</td><td>77.4</td><td>0.0</td><td>77.4</td></tr><tr><td>mean (tampers)</td><td>88.6</td><td>73.1</td><td>89.3</td><td>94.1</td><td>0.0</td><td>94.1</td></tr></table>

Table 2: The evaluator ladder on the synthetic suite (n=7,560, 9 runs). Columns represent output-only evaluators of increasing strength and the identity channel. Blindness is the GPT-5-nano acceptance rate under identity failure (= judge acceptance, since identity is 0% for all tampers). Mean tamper acceptance rises from Standard (88.6%) through Qwen3-8B (89.3%) to GPT-5-nano (94.1%): stronger evaluators accept more tampered outputs. The identity channel separates every tamper at 0%.

<table><tr><td>T</td><td>Standard</td><td>Strict</td><td>Forensic</td><td>Canary</td></tr><tr><td>0.70</td><td>5/6</td><td>4/6</td><td>5/6</td><td>2/6</td></tr><tr><td>0.80</td><td>5/6</td><td>3/6</td><td>5/6</td><td> $0 / 6 ^ { \times }$ </td></tr><tr><td>0.90</td><td>4/6</td><td> $0 / 6 ^ { \times }$ </td><td>4/6</td><td> $0 / 6 ^ { \times }$ </td></tr><tr><td>clean acc.</td><td>97.6</td><td>85.4</td><td>96.8</td><td>75.9</td></tr></table>

Table 3: Suite-level blindness blind@τ: tampered conditions (of six) accepted at threshold τ while identity is 0%, for the four rule-based evaluators. Forensic adds a scan for tamper artifacts (override strings, leaked secrets) to standard scoring; Canary applies strict scoring to canary probes alone. <sup>×</sup>The same setting also rejects the clean deployment, whose acceptance (%) is in the last row.

Behavioral fingerprinting covers one of six components. An LLMmap-style fingerprinting baseline (Pasquini et al., 2025) using 40 discriminative probes identifies the model component with 76.7% three-way accuracy (81.2% base-vs-substitute), above the 33.3% chance level. Fingerprinting is operationally valuable for model identification. It leaves the equivalence class intact, for two reasons. First, the two models emit byte-identical outputs on 37.5% of probes even across a 4.5× parameter gap, a collision floor independent of fingerprint strength. Second, the five non-model tampers (prompt, tool, retrieval, dependency, postprocessor) leave the model unchanged, so fingerprinting is structurally blind to them. A model-preserving tamper (e.g., the conditional prompt tamper of §6) evades fingerprinting by construction.

<table><tr><td>Setting</td><td>n/cond.</td><td>Clean</td><td>Tampered</td><td>Id.</td></tr><tr><td>Synthetic suite (main study)</td><td>1,080</td><td>97.6</td><td>88.6</td><td>0</td></tr><tr><td>BFCL-simple</td><td>600</td><td>100.0</td><td>99.9</td><td>0</td></tr><tr><td>GSM8K</td><td>600</td><td>27.7</td><td>27.6</td><td>0</td></tr><tr><td> $\mathrm { Q w e n } 3 \mathrm { - } 8 \mathrm { B }  \mathrm { Q w e n } 2 . 5 \mathrm { - } 7 \mathrm { B } ^ { \dag }$ </td><td>120</td><td>100.0</td><td>100.0</td><td>0</td></tr><tr><td> $\mathrm { g p t } { \scriptstyle - 0 . 8 s s - 2 0 \mathbf { B } }  \mathbf { Q w e n } 2 . 5 { \scriptscriptstyle - 7 } \mathbf { B } ^ { \dagger }$ </td><td>120</td><td>100.0</td><td>100.0</td><td>0</td></tr><tr><td> $\mathrm { \ A g e n t D o j o \ b a n k i n g ^ { \ddagger } }$ </td><td>16</td><td>43.8</td><td>50.0</td><td>0</td></tr></table>

Table 4: Scope and replication. Clean: standard acceptance (%) of the clean deployment. Tampered: mean standard acceptance (%) over the six tampers. Id.: identity pass rate (%) of the tampered deployments. BFCL and GSM8K pool three seeds (Qwen2.5-7B → 1.5B). <sup>†</sup>Model substitution only. <sup>‡</sup>Native task utility, prompt tamper only.

Tampered acceptance tracks clean acceptance across benchmarks, model sizes, and generations. Table 4 extends the main study to two public benchmarks, newer and larger declared models, and a multi-step agent benchmark (details in Appendix B). The informative comparison is tampered versus clean acceptance, because absolute acceptance depends on task difficulty. On BFCLsimple (Patil et al., 2025), the six tampers pass at 99.9% against a clean rate of 100%, under both standard and strict scoring. On GSM8K (Cobbe et al., 2021), the clean deployment solves 27.7% of problems and the tampered deployments solve 27.6%; blindness is bounded by task accuracy, and within that bound the tampers are indistinguishable from clean. Substituting Qwen2.5-7B for a newergeneration declared model (Qwen3-8B) or for a 20B model (gpt-oss-20B) leaves standard acceptance at 100%. On AgentDojo (Debenedetti et al., 2024), the prompt tamper leaves utility unchanged (50.0% vs. 43.8% clean, one task of 16). The same benchmark also shows the limit of the claim: substituting Qwen2.5-1.5B lowers utility to 25.0%, so a large capability drop is behaviorally visible. In every setting, the identity channel fails all tampered deployments.

<table><tr><td>Slice</td><td>Id.</td><td>Match</td><td>Harm</td></tr><tr><td colspan="4">E4: conditional prompt tamper</td></tr><tr><td>ordinary suite</td><td>0</td><td>100%</td><td>13.3%</td></tr><tr><td>triggered probes</td><td>0</td><td></td><td>81.0%</td></tr><tr><td colspan="4">E8: stealth undeclared tamper</td></tr><tr><td>ordinary suite</td><td>100</td><td>100%</td><td>13.3%</td></tr><tr><td>triggered probes</td><td>100</td><td></td><td>86.2%</td></tr></table>

Table 5: Stealth tampers (Qwen2.5-7B, seed 42). Id.: identity pass rate. Match: decision-match vs. clean on the ordinary evaluation suite. Harm: triggered harm rate on held-out triggered probes (n=58). E4 evades output evaluation (100% match) and is caught by identity (0%). E8 evades both channels (identity 100%) by injecting through the undeclared chat template. Both produce >80% harm on triggered inputs. The 13.3% harm on the ordinary suite equals the clean baseline.

## 6 Stealth Tampers: Testing Hypothesis 1

Hypothesis 1 asks whether practical tampers exist that are invisible on the evaluation suite and harmful off it. We test this with two experiments: a trigger-conditional tamper (E4) and a stealth tamper that evades both channels (E8). Table 5 reports the results.

A trigger-conditional tamper is invisible on the suite and harmful off it (E4). The conditional prompt tamper injects a malicious directive that activates when a trigger token appears in the user input. On the ordinary evaluation suite (triggerfree), this deployment is 100% decision-identical to clean, with harm at the clean baseline (13.3%). Every output evaluator in the ladder, including the GPT-5-nano judge, accepts it. On held-out triggered probes (n=58, policy and secret tasks), the same deployment produces 81.0% harmful outputs: 72.5% policy-violation flips and 60.0% secret leaks.

![](images/ff9d4481d27acac146f4ecfe3741b8a1bf12a6811914d4386ac847518e9d4e31.jpg)  
Figure 3: Capability scaling of the trigger-conditional tamper (E4), in %. On the ordinary suite, each tampered deployment matches the clean deployment’s decisions on at least 96.6% of tasks, so every output evaluator accepts it. On held-out triggered probes (n=58), harm rises with the capability of the deployed model.

This result confirms Hypothesis 1. The deployment has $\varepsilon ( A , B ; \mathcal { D } ) = 0$ on the evaluation distribution (100% decision-identical) and causes 81% harm on triggered inputs. Theorem 1 implies that behavioral tests at any sample size remain unable to detect it. The identity channel catches it at 0% because the prompt component changed. A second seed gives 79% triggered harm.

Harm scales with model capability. Figure 3 repeats E4 on three deployed models with the same triggered probes. All three deployments are behavior-preserving on the ordinary suite (decision match ≥96.6%), while triggered harm rises from 48.3% (Llama-3.1-8B) to 81.0% (Qwen2.5- 7B) to 100% (Qwen2.5-14B). The more capable the model, the more reliably it executes the hidden conditional while staying invisible on the evaluation distribution. The two scaling results point in the same direction: a stronger evaluator accepts more tampered outputs (§5), and a stronger deployed model executes a hidden tamper more reliably.

A stealth tamper evades both channels (E8). E8 injects the same trigger-conditional logic through the chat template, a component outside the manifest’s coverage. On the ordinary suite, identity passes at 100% (the manifest is intact), behavior is

<table><tr><td>Component</td><td>Descriptor</td><td>Declared → observed</td><td></td></tr><tr><td>Model</td><td>revision id; tokenizer and chat-template hash</td><td>6b171a → 6b171a</td><td>√</td></tr><tr><td>Prompt</td><td>hash of the resolved prompt</td><td>c23ca1 → f957c9</td><td>X</td></tr><tr><td>Tooling</td><td>hash of tool specs and router</td><td>077a40 → 077a40</td><td>√</td></tr><tr><td>Retrieval</td><td>policy hash, corpus snapshot id</td><td>cb9f5d → cb9f5d</td><td>√</td></tr><tr><td>Runtime</td><td>dependency versions, container digest</td><td>2b9d71 → 2b9d71</td><td>√</td></tr><tr><td>Postproc.</td><td>hash of the output transform</td><td>d728b3 → d728b3</td><td>√</td></tr></table>

Table 6: A minimal deployment-identity descriptor, with a worked example: the record DIB-Bench produces for the Qwen2.5-7B prompt-tamper deployment (SHA-256 prefixes). Each component is also reported with how its descriptor was obtained (self-declared, build provenance, or attestation). The single mismatch localizes the tamper to the system prompt. Content-hashing the chat template (row 1) closes the E8 gap from §6.

100% decision-identical to clean, and harm equals the clean baseline. On triggered probes, identity still passes at 100% while harm reaches 86.2%. One deployment simultaneously evades both the output evaluator and the identity channel.

The tamper evades identity because it hides in an undeclared component. The identity channel’s power equals its coverage, and coverage must be reported. The graded descriptor in §7 addresses this: content-hashing the resolved chat template (or the fully resolved prompt) closes this gap, and higher assurance levels substitute signed build provenance or runtime attestation.

## 7 A Deployment-Identity Descriptor for Evaluation Reports

The evaluator ladder motivates a concrete addition to evaluation reports: a deployment-identity descriptor that records, for each stack component, a content hash and the method used to obtain it. This is a reporting convention, cheap enough to include in any agent-evaluation paper.

Table 6 also shows the second benefit of a percomponent record: it localizes a change. In the main study, every tampered example reports a mismatch on exactly the changed component, and every clean example reports a full match (Appendix E). A reader of the report therefore learns which part of the stack differs, which a behavioral score cannot provide.

The descriptor is graded rather than binary. Its cheapest form is a self-reported manifest of content hashes, which is what DIB-Bench records.

Stronger forms substitute signed build provenance (Torres-Arias et al., 2019; SLSA Framework, 2026) or hardware-rooted runtime attestation. What matters for the report is that the channel is present and auditable. The descriptor certifies that the deployed component is the one the report describes, rather than certifying safety or correctness, which is the claim that behavioral numbers leave unaddressed.

Our results translate into four reporting recommendations for agent evaluations and leaderboards:

1. Report the descriptor next to the score. A behavioral score without a deployment-identity descriptor names a model, while the evaluated object is a stack (§5).

2. Report manifest coverage. Identity detection equals coverage (§4.2), and E8 evades a manifest that omits the chat template (§6). A report should list which components are hashed.

3. Report suite coverage. Output evaluation catches a tamper when the suite probes the changed behavior (Figure 2). A report should state which behaviors its suite probes, such as policy decisions, secret handling, and tool selection.

4. Treat a mismatch as a change to review. A benign prompt paraphrase also fails identity (§4.2). A mismatch identifies the changed component for human review and does not by itself indicate harm.

Three directions remain for future work. First, the adaptive-probe identification budget: detecting a trigger-gated tamper requires querying an input in its trigger set, so the expected probe count is bounded below by $1 / \rho$ for trigger density ρ. Second, longitudinal deployment-drift studies of popular agent frameworks to measure how often runtime stacks diverge from declared configurations in practice. Third, leaderboard rank decomposition: re-scoring an existing agent leaderboard under controlled component swaps to test whether model rank and stack rank coincide.

## 8 Related Work

LLM judges and evaluation integrity. LLM-asa-judge evaluation agrees with human preferences at high rates (Zheng et al., 2023) but carries position and verbosity biases (Zheng et al., 2023; Wang et al., 2024). A second line of work asks whether reported scores measure what they claim: GSM1k shows that some models overfit a public benchmark (Zhang et al., 2024), and the Leaderboard Illusion documents how private testing distorts arena rankings (Singh et al., 2025). These studies question the score given a fixed system. We question the system given a fixed score: even an unbiased judge on an uncontaminated benchmark cannot establish which deployment produced the outputs.

Agent and tool evaluation. ToolEmu evaluates long-tail risks with an LM-emulated sandbox (Ruan et al., 2024). τ-bench evaluates tool-agentuser interaction (Yao et al., 2024), ToolSandbox adds stateful conversational tool use (Lu et al., 2025), and AgentHarm measures harmfulness in multi-step settings (Andriushchenko et al., 2025). These benchmarks evaluate agent behavior; our work asks whether behavioral evaluation can detect that the deployed runtime stack differs from the declared one.

Prompt injection and retrieval poisoning. Indirect prompt injection shows that external content can hijack LLM-integrated applications (Abdelnabi et al., 2023). BIPIA, InjecAgent, and AgentDojo benchmark injection robustness (Yi et al., 2025; Zhan et al., 2024; Debenedetti et al., 2024), and PoisonedRAG studies knowledge corruption in RAG (Zou et al., 2025). Our tamper conditions are inspired by this attack surface, but we measure whether output-only evaluation notices component changes, not whether a particular attack succeeds behaviorally.

Model fingerprinting and attribution. LLMmap uses stylistic signals to identify serving models (Pasquini et al., 2025); behavioral fingerprinting has been proposed for runtime attribution (Pei et al., 2025). Our E6 baseline confirms fingerprinting is useful (76.7% accuracy) and limited (1 of 6 components, 37.5% collision floor), consistent with the Theorem 1 bound.

Model substitution and deployment drift. Chen et al. (2024) show that the behavior of a commercial LLM service changes substantially within months under an unchanged name. Cai et al. (2025) audit model substitution in LLM APIs and find that output-based statistical tests need many queries and miss subtle substitutions, and they propose trusted execution environments as a remedy. Our results agree with both and extend them in two directions: from the model to the five other stack components, and from API auditing to the evaluation report, where the same blindness affects every published behavioral score.

Compound AI systems. Zaharia et al. (2024) frame the shift from single-model to compoundsystem evaluation. Our paper sharpens one consequence: when the unit is a stack, an evaluation report that names the model is structurally underspecified, and stricter behavioral tests leave the gap open.

Documentation, provenance, and attestation. Model cards, datasheets, data statements, and AI FactSheets emphasize structured transparency (Mitchell et al., 2019; Gebru et al., 2021; Bender and Friedman, 2018; Arnold et al., 2019). Software supply-chain systems such as in-toto and SLSA provide artifact-integrity mechanisms (Torres-Arias et al., 2019; SLSA Framework, 2026). The EU AI Act (European Parliament and Council of the European Union, 2024) and the NIST AI RMF (National Institute of Standards and Technology, 2023) presuppose identifiable, lineage-traceable deployments. We connect these lines to NLP evaluation: if the evaluated object is a deployed LLM agent, component identity should become part of the evaluation report.

## 9 Conclusion

DIB-Bench measures a gap that stronger evaluators widen rather than close. Mean tamper acceptance rises from 88.6% (rule-based) to 94.1% (GPT-5- nano), and a PAC-style dichotomy shows that the ε=0 tamper class is undetectable at any sample size by any output-only evaluator. A trigger-conditional tamper realizes this class with 48–100% harm scaling with model capability. A deployment-identity channel based on manifest hashes decides the undetectable class with zero samples. The graded deployment-identity descriptor provides the minimal addition to close the gap.

Output evaluation detects what its suite probes and the identity channel what its manifest hashes, so reports should state both coverages. New tamper conditions, evaluators, and model pairs plug into the same protocol. As LLM agents become deployed systems, evaluation reports should carry identity evidence alongside behavioral scores.

## Ethics Statement

This work studies deployment tampering in a controlled synthetic environment. The goal is defensive measurement: to show that evaluations of deployed LLM agents should report runtime identity evidence alongside behavioral results. The protocol includes synthetic secret tokens and policybypass prompts, without real secrets, users, or external services. Because the work could inform stronger tamper testing, released artifacts should avoid presenting the prompts as operational attack recipes and should emphasize the defensive evaluation framing.

## Limitations

The main empirical package uses synthetic tasks for single-component tampering. Public-benchmark anchors (BFCL-simple $n { = } 2 0 0 { \times } 3$ seeds, GSM8K $n { = } 2 0 0 { \times } 3$ seeds) replicate the pattern at scale; AgentDojo provides multi-step agent trajectories (n=16). The tampering conditions are singlecomponent and do not cover multi-component attacks or adversaries who can forge the manifest directly; the E8 stealth result shows that undeclared components are a concrete gap, addressed by the graded descriptor but not eliminated without runtime attestation. The models tested range from 1.5B to 20B parameters; the GPT-5-nano judge represents a current-generation commercial model, but evaluating on frontier-scale deployed systems would strengthen the scope. The PAC dichotomy covers passive i.i.d. sampling; the adaptive-probe lower bound (budget $\geq 1 / \rho$ for trigger density ρ) is scoped as future work. The manifest is softwarelevel evidence: it lacks hardware-rooted guarantees and can check components that are declared and measured.

## References

Sahar Abdelnabi, Kai Greshake, Shailesh Mishra, Christoph Endres, Thorsten Holz, and Mario Fritz. 2023. Not what you’ve signed up for: Compromising real-world llm-integrated applications with indirect prompt injection. In Proceedings of the 16th ACM Workshop on Artificial Intelligence and Security, AISec 2023, Copenhagen, Denmark, 30 November 2023, pages 79–90. ACM.

Maksym Andriushchenko, Alexandra Souly, Mateusz Dziemian, Derek Duenas, Maxwell Lin, Justin Wang, Dan Hendrycks, Andy Zou, J. Zico Kolter, Matt Fredrikson, Yarin Gal, and Xander Davies. 2025. Agentharm: A benchmark for measuring harmfulness of LLM agents. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net.

Matthew Arnold, Rachel K. E. Bellamy, Michael Hind, Stephanie Houde, Sameep Mehta, Aleksandra Mojsilovic, Ravi Nair, Karthikeyan Natesan Ramamurthy, Alexandra Olteanu, David Piorkowski, Darrell Reimer, John T. Richards, Jason Tsay, and Kush R. Varshney. 2019. Factsheets: Increasing trust in AI services through supplier’s declarations of conformity. IBM J. Res. Dev., 63(4/5):6:1–6:13.

Emily M. Bender and Batya Friedman. 2018. Data statements for natural language processing: Toward mitigating system bias and enabling better science. Trans. Assoc. Comput. Linguistics, 6:587–604.

Will Cai, Tianneng Shi, Xuandong Zhao, and Dawn Song. 2025. Are you getting what you pay for? auditing model substitution in LLM APIs. In Advances in Neural Information Processing Systems 38.

Lingjiao Chen, Matei Zaharia, and James Zou. 2024. How is ChatGPT’s behavior changing over time? Harvard Data Science Review.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021. Training verifiers to solve math word problems. CoRR, abs/2110.14168.

Edoardo Debenedetti, Jie Zhang, Mislav Balunovic, Luca Beurer-Kellner, Marc Fischer, and Florian Tramèr. 2024. Agentdojo: A dynamic environment to evaluate prompt injection attacks and defenses for LLM agents. In Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024.

European Parliament and Council of the European Union. 2024. Regulation (EU) 2024/1689 of the european parliament and of the council laying down harmonised rules on artificial intelligence (artificial intelligence act). Official Journal of the European Union, L series, 12 July 2024.

Timnit Gebru, Jamie Morgenstern, Briana Vecchione, Jennifer Wortman Vaughan, Hanna M. Wallach, Hal Daumé III, and Kate Crawford. 2021. Datasheets for datasets. Commun. ACM, 64(12):86–92.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient memory management for large language model serving with pagedattention. In Proceedings ofthe 29th Symposium on Operating Systems Principles, SOSP 2023, Koblenz, Germany, October 23-26, 2023, pages 611– 626. ACM.

Jiarui Lu, Thomas Holleis, Yizhe Zhang, Bernhard Aumayer, Feng Nan, Haoping Bai, Shuang Ma, Shen Ma, Mengyu Li, Guoli Yin, Zirui Wang, and Ruoming Pang. 2025. Toolsandbox: A stateful, conversational, interactive evaluation benchmark for LLM tool use capabilities. In Findings of the Association

for Computational Linguistics: NAACL 2025, Albuquerque, New Mexico, USA, April 29 - May 4, 2025, Findings of ACL, pages 1160–1183. Association for Computational Linguistics.

Margaret Mitchell, Simone Wu, Andrew Zaldivar, Parker Barnes, Lucy Vasserman, Ben Hutchinson, Elena Spitzer, Inioluwa Deborah Raji, and Timnit Gebru. 2019. Model cards for model reporting. In Proceedings ofthe Conference on Fairness, Accountability, and Transparency, FAT\* 2019, Atlanta, GA, USA, January 29-31, 2019, pages 220–229. ACM.

National Institute of Standards and Technology. 2023. Artificial intelligence risk management framework (AI RMF 1.0). Technical Report NIST AI 100-1, U.S. Department of Commerce.

Dario Pasquini, Evgenios M. Kornaropoulos, and Giuseppe Ateniese. 2025. Llmmap: Fingerprinting for large language models. In 34th USENIX Security Symposium, USENIX Security 2025, Seattle, WA, USA, August 13-15, 2025, pages 299–318. USENIX Association.

Shishir G. Patil, Huanzhi Mao, Fanjia Yan, Charlie Cheng-Jie Ji, Vishnu Suresh, Ion Stoica, and Joseph E. Gonzalez. 2025. The berkeley function calling leaderboard (BFCL): from tool use to agentic evaluation of large language models. In Fortysecond International Conference on Machine Learning, ICML 2025, Vancouver, BC, Canada, July 13-19, 2025, Proceedings of Machine Learning Research. PMLR / OpenReview.net.

Zehua Pei, Hui-Ling Zhen, Yingchun Zhang, Zhiyuan Yang, Xing Li, Xianzhi Yu, Mingxuan Yuan, and Bei Yu. 2025. Behavioral fingerprinting of large language models. CoRR, abs/2509.04504.

Yangjun Ruan, Honghua Dong, Andrew Wang, Silviu Pitis, Yongchao Zhou, Jimmy Ba, Yann Dubois, Chris J. Maddison, and Tatsunori Hashimoto. 2024. Identifying the risks of LM agents with an lmemulated sandbox. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net.

Shivalika Singh, Yiyang Nan, Alex Wang, Daniel D’Souza, Sayash Kapoor, Ahmet Üstün, Sanmi Koyejo, Yuntian Deng, Shayne Longpre, Noah A. Smith, Beyza Ermis, Marzieh Fadaee, and Sara Hooker. 2025. The leaderboard illusion. arXiv preprint arXiv:2504.20879.

SLSA Framework. 2026. Supply-chain levels for software artifacts. https://slsa.dev/spec/v1.2/ about. Accessed 2026-05-25.

Llama Team. 2024. The llama 3 herd of models. CoRR, abs/2407.21783.

Santiago Torres-Arias, Hammad Afzali, Trishank Karthik Kuppusamy, Reza Curtmola, and Justin Cappos. 2019. in-toto: Providing farm-to-table guarantees for bits and bytes. In 28th USENIX

Security Symposium, USENIX Security 2019, Santa Clara, CA, USA, August 14-16, 2019, pages 1393–1410. USENIX Association.

Peiyi Wang, Lei Li, Liang Chen, Zefan Cai, Dawei Zhu, Binghuai Lin, Yunbo Cao, Lingpeng Kong, Qi Liu, Tianyu Liu, and Zhifang Sui. 2024. Large language models are not fair evaluators. In Proceedings ofthe 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 9440–9450.

An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, and 22 others. 2024. Qwen2.5 technical report. CoRR, abs/2412.15115.

Shunyu Yao, Noah Shinn, Pedram Razavi, and Karthik Narasimhan. 2024. τ-bench: A benchmark for toolagent-user interaction in real-world domains. CoRR, abs/2406.12045.

Jingwei Yi, Yueqi Xie, Bin Zhu, Emre Kiciman, Guangzhong Sun, Xing Xie, and Fangzhao Wu. 2025. Benchmarking and defending against indirect prompt injection attacks on large language models. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining, V.1, KDD 2025, Toronto, ON, Canada, August 3-7, 2025, pages 1809–1820. ACM.

Matei Zaharia, Omar Khattab, Lingjiao Chen, Jared Quincy Davis, Heather Miller, Christopher Potts, James Zou, Michael Carbin, Jonathan Frankle, Naveen Rao, and Ali Ghodsi. 2024. The shift from models to compound AI systems. https://bair.berkeley.edu/blog/2024/ 02/18/compound-ai-systems/. Berkeley AI Research Blog.

Qiusi Zhan, Zhixiang Liang, Zifan Ying, and Daniel Kang. 2024. Injecagent: Benchmarking indirect prompt injections in tool-integrated large language model agents. In Findings of the Association for Computational Linguistics, ACL 2024, Bangkok, Thailand and virtual meeting, August 11-16, 2024, Findings of ACL, pages 10471–10506. Association for Computational Linguistics.

Hugh Zhang, Jeff Da, Dean Lee, Vaughn Robinson, Catherine Wu, Will Song, Tiffany Zhao, Pranav Raja, Charlotte Zhuang, Dylan Slack, Qin Lyu, Sean Hendryx, Russell Kaplan, Michele Lunati, and Summer Yue. 2024. A careful examination of large language model performance on grade school arithmetic. In Advances in Neural Information Processing Systems 37 (Datasets and Benchmarks Track).

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. 2023. Judging

LLM-as-a-judge with MT-bench and chatbot arena. In Advances in Neural Information Processing Systems 36 (Datasets and Benchmarks Track).

Wei Zou, Runpeng Geng, Binghui Wang, and Jinyuan Jia. 2025. Poisonedrag: Knowledge corruption attacks to retrieval-augmented generation of large language models. In 34th USENIX Security Symposium, USENIX Security 2025, Seattle, WA, USA, August 13- 15, 2025, pages 3827–3844. USENIX Association.

## A PAC Dichotomy Proof

Fix the evaluation distribution D. A deployment D maps each input x to a (possibly stochastic) output distribution $D ( x )$ . A behavioral test $T$ is any function of a sample ${ \cal S } _ { N } = \{ ( x _ { i } , o _ { i } ) \} _ { i = 1 } ^ { N }$ with $x _ { i } \sim \mathcal { D } \mathrm { i . i . d } .$ . and $o _ { i } \sim D ( x _ { i } ) $ ; it outputs a guess in {clean, tampered}. This subsumes rule-based scorers, LLM judges, forensic classifiers, and behavioral fingerprints.

For clean deployment A and tampered deployment B, define the joint distributions $P _ { A } ( x , o ) =$ $\mathcal { D } ( \boldsymbol { x } ) \cdot \boldsymbol { A } ( \boldsymbol { x } ) ( o )$ and $P _ { B } ( x , o ) = \mathcal { D } ( x ) \cdot B ( x ) ( o )$ Since A and B share the input marginal:

$$
\mathrm { T V } ( P _ { A } , P _ { B } ) = \mathbb { E } _ { \boldsymbol { x } \sim \mathcal { D } } [ \mathrm { T V } ( A ( \boldsymbol { x } ) , B ( \boldsymbol { x } ) ) ] = \varepsilon .
$$

Part (i): lower bound. The N-sample laws are product measures $P _ { A } ^ { N }$ and $P _ { B } ^ { N }$ . By sub-additivity of TV under products:

$$
\mathrm { T V } ( P _ { A } ^ { N } , P _ { B } ^ { N } ) \leq N \cdot \mathrm { T V } ( P _ { A } , P _ { B } ) = N \varepsilon .
$$

The distinguishing advantage of any test T satisfies $| \operatorname* { P r } [ T ( S _ { N } )$ = tampered $B ] - \operatorname* { P r } [ T ( S _ { N } ) =$ tampered $| \mathbf { \delta } A ] | \leq \mathrm { T V } ( P _ { A } ^ { N } , P _ { B } ^ { N } ) \dot { \mathbf { \leq } } N \varepsilon$ . If $\varepsilon = 0$ then $P _ { A } = P _ { B }$ and advantage is exactly 0 for every N.

Part (ii): upper bound. Since $\mathrm { T V } ( P _ { A } , P _ { B } ) =$ $\varepsilon > 0$ , there exists a measurable event E on $( x , o )$ with $| P _ { A } ( E ) - P _ { B } ( E ) | \geq \varepsilon .$ . Estimate Pr[E] by its empirical frequency $\begin{array} { r } { \hat { p } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathcal { H } [ ( x _ { i } , o _ { i } ) \in E ] } \end{array}$ By Hoeffding’s inequality, $| \hat { p } - p | \le \varepsilon / 2$ with probability $\geq 1 - \delta$ when $\begin{array} { r } { N \ge \frac { 2 } { \varepsilon ^ { 2 } } } \end{array}$ ln $\frac { 2 } { \delta }$ . Threshold at the midpoint $( P _ { A } ( E ) + P _ { B } ( E ) ) / 2$ to distinguish A from B with error $\leq \delta .$ □

Remark. The identity evaluator compares declared vs. observed component hashes with one manifest read, independent of ε. It decides the $\varepsilon = 0$ class with zero samples, which Theorem 1(i) proves is undecidable for behavioral tests at any N.

## B Scope and Replication

E2/E3 across model pairs and seeds. E2 (undeclared-component tampers) and E3 (benignchange conditions) replicate identically across three configurations: Qwen2.5-7B → 1.5B (seeds 42, 43), Llama-3.1-8B → Qwen2.5-7B (seed 42), and Qwen2.5-14B (seed 42). E2 identity passes at 100% while behavior changes by 15–100%; E3 identity fails at 0% while decision-match is 100%.

E4 across model pairs. The conditional prompt tamper replicates across all three model sizes. On the ordinary suite, decision-match vs. clean is $2 9 6 . 6 \%$ for all three. Triggered harm: Llama-3.1- 8B 48%, Qwen2.5-7B 81% (seed 42) / 79% (seed 43), Qwen2.5-14B 100%.

BFCL-simple $n { = } 2 0 0 \ \times \ 3$ seeds. Clean standard/strict: 100%. All six tampers: standard 99– 100%, strict 99–100%, identity 0%, mean blindness 99.9%. Six of six tampers accepted at $\tau { = } 0 . 8 0$ The extreme-blindness anchor replicates at 12× the original sample size.

GSM8K $n { = } 2 0 0 \ \times \ 3$ seeds. Clean standard: ∼28% (the lower value vs. the paper’s 50-item 38% slice reflects a length effect at max\_len 2048; the pattern is identical). All four behavior-preserving tampers cluster at the clean baseline (∼27–28% standard), identity 0%. Model substitution drops strict to 7.8%.

Cross-generation substitution. Declared Qwen3-8B (a 2025 model), deployed Qwen2.5-7B: standard 100% and strict 85% for both clean and substituted deployments, identity 0%, blindness 100%. Newer-generation models are equally vulnerable to substitution blindness.

Above 14B parameters. A 20B-parameter model (gpt-oss-20b) substituted with Qwen2.5-7B: standard 100% → 100%, identity $1 0 0 \%  0 \%$ blindness 100%. The effect extends beyond small open models.

AgentDojo banking suite (n=16). Clean utility: 43.8%. Prompt tamper utility: 50.0% (behaviorally invisible; identity fails). Model substitution (Qwen2.5-1.5B) utility: 25.0% (behaviorally visible; identity fails). The prompt tamper is invisible on a real multi-step agent benchmark while identity catches it. The 4.5× model downgrade is behaviorally visible, which underscores the point: output-only evaluation has discriminative power for some tampers but not others, and without the identity channel an operator cannot distinguish the two cases.

<table><tr><td>Layer</td><td>Borrowed shape</td><td>DIB-Bench instantiation</td><td>Reported artifact</td></tr><tr><td>Task environment</td><td>Agent benchmarks define tasks, tools, and adversarial contexts.</td><td>DIB-Bench defines factual, policy, tool, secret, and canary task families</td><td>Output text and task metadata.</td></tr><tr><td>Deployment condi- tion</td><td>Attack benchmarks perturb prompts, tools, users, or re-</td><td>DIB-Bench perturbs one declared run- time component at a time.</td><td>Condition label and changed component.</td></tr><tr><td>Behavioral scoring</td><td>trieval content. Benchmarks score task success,</td><td>DIB-Bench reports standard, strict, foren-</td><td>Output-only pass rates</td></tr><tr><td>Integrity evidence</td><td>safety, and robustness. Supply-chain systems record provenance and artifact in-</td><td>sic, and canary output scores. DIB-Bench records expected-vs- observed component hashes and traces.</td><td>and CIs. Manifest matches, mis- matches, hashes, and</td></tr><tr><td>Blindness metric</td><td>tegrity. The bridge: compare behav- ior acceptance with deployment</td><td>DIB-Bench measures output-passing in- tegrity failures.</td><td>traces. Example and suite- level blindness.</td></tr></table>

Table 7: DIB-Bench interface: a bridge between agent behavior benchmarks and deployment-identity evidence.

![](images/35e70548bb4cbd4bba07c15a1df64bf1b2149abb021f162167e9379a7fbeb792.jpg)  
Figure 4: Blindness by (condition, model pair).

## C DIB-Bench Framework

Table 7 summarizes DIB-Bench as a bridge between agent behavior benchmarks and supplychain provenance frameworks.

## D Robustness Across Model Pairs

Figure 4 shows blindness rates per (tamper condition, model pair). The pattern persists across two within-family Qwen substitutions and one crossfamily Llama-to-Qwen substitution.

## E Component-Localized Evidence

Table 8 reports component-level mismatch evidence. Each controlled tamper is localized to the intended manifest component.

## F Artifacts, Licenses, and Compute

Artifacts and licenses. Deployed and judge models: Qwen2.5-1.5B/7B/14B-Instruct and Qwen3- 8B (Apache 2.0), gpt-oss-20b (Apache 2.0), and Llama-3.1-8B-Instruct (Llama 3.1 Community License). Benchmarks: GSM8K (Cobbe et al., 2021) (MIT), BFCL (Patil et al., 2025) from the Gorilla repository (Apache 2.0), and AgentDojo (Debenedetti et al., 2024) (MIT). Inference engine: vLLM (Apache 2.0). GPT-5-nano is accessed through the OpenAI API under its terms of use. All artifacts are used for research evaluation, consistent with their licenses and intended use. The synthetic tasks contain synthetic secret tokens and policy prompts and contain no personal data.

Setup and statistics. Generation uses vLLM 0.21.0 with PyTorch 2.11.0, Transformers 5.6.0, and Python 3.12, with greedy decoding (temperature 0, top-p 1), at most 160 new tokens, and a 2,048-token context. The Qwen3-8B judge runs with thinking disabled and temperature 0; the GPT-5-nano judge runs with minimal reasoning effort. The suite threshold is $\tau { = } 0 . 8 0$ unless stated otherwise. Main-study values are means over nine runs (three model pairs, seeds 42, 43, 44; n=1,080 per condition). 95% bootstrap confidence intervals (1,000 resamples) for per-condition blindness have half-widths of at most 2.9 points.

Compute. All open-model inference runs on a single NVIDIA RTX A6000 GPU (48 GB). The deployed models range from 1.5B to 20B parameters.

Table 8: Component-localized identity evidence. Rates are percentages.
<table><tr><td>Condition</td><td>Model</td><td>Prompt</td><td>Tooling</td><td>Retrieval</td><td>Runtime</td><td>Postproc.</td><td>Any</td></tr><tr><td>clean</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>model_substitution</td><td>100.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>100.0</td></tr><tr><td>prompt_tamper</td><td>0.0</td><td>100.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>100.0</td></tr><tr><td>tool_tamper</td><td>0.0</td><td>0.0</td><td>100.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>100.0</td></tr><tr><td>retrieval_tamper</td><td>0.0</td><td>0.0</td><td>0.0</td><td>100.0</td><td>0.0</td><td>0.0</td><td>100.0</td></tr><tr><td>dependency_tamper</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>100.0</td><td>0.0</td><td>100.0</td></tr><tr><td>postprocess_tamper</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>100.0</td><td>100.0</td></tr></table>