# RESERVE-AWARE CONTRAST CERTIFICATES FOR CONSERVATIVE BANDITS WITH UNCERTAIN BASELINES

Qinchuan Cheng

School of Automation Science and Engineering, Xi’an Jiaotong University, Xi’an, China 2310820636@qq.com ORCID: 0009-0005-9325-7554

## ABSTRACT

Conservative bandits must improve an incumbent policy with out exhausting a prescribed performance budget. When the incumbent is uncertain, separately bounding candidate and baseline rewards can charge twice for shared estimation er ror. We develop Reserve-C4B around the baseline-relative contrast itself. A shared confidence set yields an exact expression for this avoidable penalty and a tighter admissibility test at every fixed history. A reserve ledger separates statistical evidence from permitted performance deficit; a prefix-refresh extension recertifies accumulated decisions under the current confidence set without discarding previously certified credit. For linear rewards, self-normalized confidence sets provide simultaneous validity over time and adaptively generated candidates, and the resulting policy satisfies a conditional-mean performance constraint with high probability. Reproducible experiments isolate certificate coupling, prefix refresh, and historical information, showing large reductions in baseline fallback while exposing the limitations of frozen certificates.

Index Terms— Conservative bandits, sequential decision making, confidence sets, baseline uncertainty, safe exploration

## 1. INTRODUCTION

Online adaptation in recommendation, sensing, and resource allocation must often retain the reliability of an existing policy. Conservative bandits formalize this requirement by constraining cumulative reward relative to an incumbent while learning from sequential feedback [1, 2]. With an uncertain incumbent, the safety decision is comparative: can an action’s reward cover a specified fraction of the baseline’s reward? Estimating these two quantities independently can obscure how much of their uncertainty is shared.

Consider two identical actions with a common reward interval [ℓ, �]. A separate lower–upper comparison gives $\ell -$ $( 1 - \alpha ) U$ , which may be negative even when $\ell > 0$ . The actual contrast is $\alpha \mu ,$ , certified by $\alpha \ell .$ . The difference, $( 1 - \alpha ) ( U - \ell )$ is an uncertainty penalty created by the comparison rather than by the action. Nonidentical actions inherit the same problem when their estimation errors are aligned. A larger initial reserve can postpone rejection, but does not remove this penalty or validate an incorrect certificate.

We study this distinction at the certificate and budget levels. Reserve-C4B uses one confidence set to bound the candidate–baseline contrast, quantifies the price of decoupling that set, and spends only certified credit. Its prefixrefresh extension also pools uncertainty across executed rounds: historical decisions can be recertified as information improves, while a carry-forward bound preserves previously established credit. The analysis makes the source of every budget increment explicit.

Relation to prior work. Unknown baselines are already treated by conservative linear and combinatorial bandits [2, 3]; improved exploration rules, budget reductions, and nonlinear regression oracles broaden that literature [4, 5, 6]. Robust baseline regret also motivates evaluating two policies under a shared model [7]. Building on these formulations, we provide a contrast-first certificate analysis, an explicit carry/refresh ledger with an anytime safety proof, and matched ablations that identify where shared uncertainty and historical information change admissibility. Time-uniform off-policy evaluation offers a complementary route to gated deployment [8]; here the target is the realized action sequence’s cumulative conditional means.

## 2. DECISION SETTING AND STATISTICAL EVIDENCE

At round �, the learner observes a finite candidate set $\mathcal { A } _ { t } .$ , baseline $b _ { t }$ , and feature vectors $x _ { t } ( a ) \in \mathbb { R } ^ { d }$ . These are measurable before the current reward, conditional on the history and any candidate-generation randomness. Rewards follow

$$
Y _ { t } = \mu _ { t } ( a _ { t } ) + \eta _ { t } , \qquad \mu _ { t } ( a ) = x _ { t } ( a ) ^ { \top } \theta _ { \ast } ,\tag{1}
$$

where $\theta _ { * }$ is fixed, $\| \theta _ { * } \| _ { 2 } \leq S ,$ features are bounded and predictable, and $\eta _ { t }$ is conditionally �-sub-Gaussian with zero mean. Only the selected reward is observed. The baseline mean is unknown; its nonnegativity is known.

Let $c = 1 - \alpha , 0 < \alpha < 1$ , and

$$
z _ { t } ( a ) = x _ { t } ( a ) - c x _ { t } ( b _ { t } ) , \quad \Delta _ { t } ( a ) = z _ { t } ( a ) ^ { \top } \theta _ { * } .\tag{2}
$$

For an externally specified reserve $R _ { 0 } \geq 0$ , the requirement is

$$
D _ { t } : = R _ { 0 } + \sum _ { s = 1 } ^ { t } \Delta _ { s } ( a _ { s } ) \geq 0 \quad \mathrm { f o r ~ e v e r y ~ } t .\tag{3}
$$

$R _ { 0 } = 0$ recovers the strict cumulative conservative constraint. Positive $R _ { 0 }$ permits an additive deficit; it is not additional statistical confidence. Historical training observations improve estimation but receive no deployment budget credit. Equation (3) concerns conditional means, not every noisy realizedreward path.

Let $\mathcal { T } _ { t }$ index all observations available before round $t ,$ including the historical sample. Form

$$
V _ { t } = \lambda I + \sum _ { i \in J _ { t } } x _ { i } x _ { i } ^ { \top } ,\tag{4}
$$

$$
\widehat { \theta _ { t } } = V _ { t } ^ { - 1 } \sum _ { i \in J _ { t } } x _ { i } Y _ { i } ,\tag{5}
$$

$$
\beta _ { t } = \sigma \sqrt { \log \frac { \operatorname* { d e t } V _ { t } } { \lambda ^ { d } } + 2 \log \frac { 1 } { \delta } } + \sqrt { \lambda } S .\tag{6}
$$

The self-normalized result of [9] gives

$$
\begin{array} { r } { \operatorname* { P r } \{ \theta _ { * } \in C _ { t } \ \forall t \} \geq 1 - \delta , \quad C _ { t } = \{ \theta : \left\| \theta - \widehat \theta _ { t } \right\| _ { V _ { t } } \leq \beta _ { t } \} . } \end{array}\tag{7}
$$

This event covers every predictable candidate, even if the generator depends on past observations. No union bound over pool size is required. Marginal prediction coverage for future outcomes [10] is not a substitute for this simultaneous conditional-mean guarantee.

## 3. CONTRAST CERTIFICATES AND RESERVE LEDGERS

## 3.1. The cost of independent uncertainty bounds

A shared-set certificate is $L _ { t } ^ { \mathrm { J } } ( a ) = \operatorname* { i n f } _ { \theta \in C _ { t } } z _ { t } ( a ) ^ { \top } \theta$ . Writing $\| \nu \| _ { t } = \sqrt { \nu ^ { \top } V _ { t } ^ { - 1 } \nu }$ , its closed form and the separate alternative are

$$
L _ { t } ^ { \mathrm { J } } ( a ) = z _ { t } ( a ) ^ { \top } \widehat { \theta } _ { t } - \beta _ { t } \| z _ { t } ( a ) \| _ { t } ,\tag{8}
$$

$$
L _ { t } ^ { \mathrm { S } } ( a ) = z _ { t } ( a ) ^ { \top } \widehat { \theta _ { t } } - \beta _ { t } \big ( \left\| x _ { t } ( a ) \right\| _ { t } + c \left\| x _ { t } ( b _ { t } ) \right\| _ { t } \big ) .\tag{9}
$$

Proposition 1 (Decoupling penalty). For the same history and action,

$$
L _ { t } ^ { \mathrm { J } } ( a ) - L _ { t } ^ { \mathrm { S } } ( a ) = \beta _ { t } \big ( \| x _ { t } ( a ) \| _ { t } + c \| x _ { t } ( b _ { t } ) \| _ { t } - \| z _ { t } ( a ) \| _ { t } \big ) \geq 0 .\tag{10}
$$

At $a = b _ { t } ,$ , the penalty equals $c ( U _ { t } ( b _ { t } ) - \ell _ { t } ( b _ { t } ) )$ , where $\ell _ { t } , U _ { t }$ are the ellipsoid’s reward bounds.

Proof. Minimizing a linear functional over the ellipsoid gives (9). Subtraction gives (10), and the triangle inequality establishes nonnegativity. When $a = b _ { t } , z _ { t } = \alpha x _ { t } ( b _ { t } )$ and $U _ { t } - \ell _ { t } = 2 \beta _ { t } \left. \boldsymbol { x } _ { t } ( \boldsymbol { b } _ { t } ) \right. _ { t }$ □

Thus a shared certificate admits every action admitted by a separate certificate at a fixed history and bank. This does not imply reward dominance between policies that subsequently collect different data. For the baseline itself, both methods should exploit its identity and known nonnegativity:

$$
L _ { t } ( b _ { t } ) = \alpha \operatorname* { m a x } \{ \ell _ { t } ( b _ { t } ) , 0 \} \geq 0 .\tag{11}
$$

Rejecting the baseline using the separate formula would be an avoidable implementation error, not an intrinsic obstacle to conservative learning.

## 3.2. Carry-forward and prefix refresh

Initialize $B _ { 0 } ~ = ~ R _ { 0 }$ . The frozen Reserve-C4B ledger tests $G _ { t } ^ { \mathrm { c a r r y } } ( a ) ~ = ~ B _ { t - 1 } + L _ { t } ^ { \mathrm { J } } ( a )$ , using (11) for $b _ { t }$ . Accept the highest-UCB nonbaseline candidate with $G _ { t } ^ { \mathrm { c a r r y } } ( a ) \geq 0 ,$ or execute the baseline if none qualifies. The score is $x _ { t } ( a ) ^ { \top } \widehat { \theta } _ { t } +$ $\beta _ { t } \parallel x _ { t } ( a ) \parallel _ { t }$ . Set $\boldsymbol { B } _ { t } = \boldsymbol { G } _ { t } ^ { \mathrm { c a r r y } } ( \boldsymbol { a } _ { t } )$ before observing $Y _ { t }$

Frozen certificates cannot recover credit lost to early uncertainty. Under the shared parameter in (1), keep $Z _ { t - 1 } ~ =$ $\begin{array} { r } { \sum _ { s < t } z _ { s } ( a _ { s } ) } \end{array}$ and recertify the complete candidate prefix:

$$
\begin{array} { r } { Q _ { t } ( a ) = R _ { 0 } + ( Z _ { t - 1 } + z _ { t } ( a ) ) ^ { \top } \widehat { \theta } _ { t } - \beta _ { t } \| Z _ { t - 1 } + z _ { t } ( a ) \| _ { t } } \end{array}\tag{12}
$$

The refresh extension uses

$$
\begin{array} { r } { G _ { t } ( { a } ) = \operatorname* { m a x } \{ B _ { t - 1 } + L _ { t } ( { a } ) , Q _ { t } ( { a } ) \} , \qquad B _ { t } = G _ { t } ( \boldsymbol { a } _ { t } ) , } \end{array}\tag{13}
$$

with the same UCB selection rule, then updates $Z _ { t }$ , observes $Y _ { t }$ , and updates the estimator. Refreshing uses newly available evidence, not unearned reward credit. The maximum is essential: a current confidence set need not be nested inside yesterday’s set, so a recomputed bound can be worse than the carried certificate.

Theorem 1 (Anytime conditional-mean safety). Under (1)– (7), both ledgers satisfy $D _ { t } \ \geq \ B _ { t } \ \geq \ 0$ at every deployment round with probability at least $1 - \delta$

Proof. On the common event (7), $L _ { t } ( a _ { t } ) \leq \Delta _ { t } ( a _ { t } )$ . If $B _ { t - 1 } \leq$ $D _ { t - 1 }$ , the carry bound is at most $D _ { t }$ . Also, $Q _ { t } ( a _ { t } )$ lowerbounds $R _ { 0 } + Z _ { t } ^ { \top } \theta _ { * } = D _ { t }$ . Their maximum remains a lower bound. The gate ensures nonnegativity, and the baseline is always feasible by (11). Induction starts at $B _ { 0 } = D _ { 0 } = R _ { 0 }$ . □

The proof remains valid for adaptively chosen candidates because a single parameter-containment event covers all of them. Without a certified nonnegative fallback, the policy must stop when no action qualifies. With dense covariance matrices, candidate evaluation costs $O ( | \mathcal { R } _ { t } | d ^ { 2 } )$ per round; rank-one updates cost $O ( d ^ { 2 } )$ . Refresh adds only the �-vector $Z _ { t }$ , not storage of the full trajectory.

## 3.3. How much reserve does decoupling consume?

For a fixed action path and its frozen certificates $L _ { 1 } , \dots , L _ { T } ,$ define the smallest reserve that makes every certified prefix

Algorithm 1 Reserve-C4B with optional prefix refresh   
1: Initialize estimator from history; $\overline { { B  R _ { 0 } , Z  0 } } .$   
2: for deployment rounds $t = 1 , \dots , T$ do   
3: Observe candidates and baseline; form $C _ { t }$   
4: Compute UCB scores and contrast certificates.   
5: Set $G ( a ) \gets B + L _ { t } ( a )$ for every action.   
6: if refresh is enabled then   
7: $G ( a ) \gets \operatorname* { m a x } \{ G ( a ) , \mathcal { Q } _ { t } ( a ) \}$ using $Z .$   
8: end if   
9: Choose highest-UCB candidate with $G ( a ) \geq 0 ;$   
use baseline when no candidate qualifies.   
10: Log $B  G ( a _ { t } ) ;$ set $Z \gets Z + z _ { t } ( a _ { t } )$   
11: Execute $\textstyle a _ { t } ,$ observe $Y _ { t }$ , update estimator.   
12: end for

nonnegative. It has the exact form

$$
R _ { \operatorname* { m i n } } ( L ) = \operatorname* { m a x } \left\{ 0 , \operatorname* { m a x } _ { t \leq T } \left[ - \sum _ { s \leq t } L _ { s } \right] \right\} .\tag{14}
$$

Proposition 2 (Fixed-path reserve cost). Let $\Pi _ { s } = L _ { s } ^ { \mathrm { J } } - L _ { s } ^ { \mathrm { S } } \ge$ 0 along the same path and confidence sequence, using the same structural baseline bound in both ledgers. Then

$$
0 \leq R _ { \operatorname* { m i n } } ( L ^ { \mathrm { S } } ) - R _ { \operatorname* { m i n } } ( L ^ { \mathrm { J } } ) \leq \sum _ { s = 1 } ^ { T } \Pi _ { s } .\tag{15}
$$

Proof. Every separate-bound deficit equals the corresponding joint-bound deficit plus $\textstyle \sum _ { s \leq t } \Pi _ { s }$ . Taking prefix maxima proves the lower inequality; bounding every prefix penalty by its total proves the upper inequality. Equation (14) follows directly from the prefix constraints. □

This is a certificate-level reserve requirement, not the reserve an online policy can know in advance. It explains why decoupling can consume an entire exploration allowance even when each reward interval is individually valid. Prefix refresh addresses the remaining cost of freezing early certificates; it changes the ledger, rather than simply increasing $R _ { 0 }$

## 4. EXPERIMENTS

## 4.1. Reproducible protocol and comparisons

We use a controlled linear study to separate uncertainty geometry from model misspecification. Set $d \ = \ 5 , \ \theta _ { * } \ =$ $( 1 , 0 . 6 , 0 , 0 , 0 ) , x ( b ) = ( 1 , 0 , 0 , 0 , 0 )$ , and draw 32 candidates $x ( a ) = ( 1 , \rho u )$ per round, with � uniform on the unit sphere in $\mathbb { R } ^ { 4 }$ . Gaussian noise has standard deviation $\sigma .$ . The learner knows $S = 1 . 5$ and $\sigma .$ , but not $\theta _ { * }$ or the baseline mean. With $\rho \leq 0 . 8 $ , all means are nonnegative.

Each episode has 20 historical observations followed by 200 deployment rounds. Histories are either diverse, with features (1, �), or baseline-only. We test every combination of $\rho \in \{ 0 . 1 5 , 0 . 4 , 0 . 8 \} , \sigma \in \{ 0 . 1 , 0 . 3 \}$ , and $R _ { 0 } \in \{ 0 , 0 . 5 , 2 \}$ under both histories: 36 settings, 256 independent episodes per setting, with $\alpha = \delta = 0 . 0 5$ and $\lambda = 0 . 1$ . Methods share historical samples, candidate pools, and potential noise within each episode. Scenario reuse is not treated as independent replication. Intervals use Student’s � over episodes, pairing method differences within episodes.

Table 1. Mean reward ratio / fallback percentage at $\rho = 0 . 1 5 ,$ $\sigma = 0 . 3 , R _ { 0 } = 0$ (256 episodes). Reward is normalized by the true baseline mean, used only by the evaluator.
<table><tr><td>Method</td><td>Diverse history</td><td>Baseline-only</td></tr><tr><td>Separate</td><td>1.0084 / 88.2</td><td>1.0007 / 94.1</td></tr><tr><td>Contrast</td><td>1.0557 / 19.4</td><td>1.0011 / 91.8</td></tr><tr><td>Refresh</td><td>1.0606 / 0.5</td><td>1.0190 / 12.1</td></tr><tr><td>Revalue</td><td>1.0124 / 81.6</td><td>1.0025 / 83.8</td></tr><tr><td>Revalue-F</td><td>1.0121 /81.1</td><td>1.0011 / 80.0</td></tr><tr><td>LinUCB</td><td>1.0613 / 0.0</td><td>1.0272 / 0.0</td></tr></table>

![](images/75f17edbf8f6021b8895280b44110c5e7b883093d27c503a9f002b4023176fb2.jpg)  
Fig. 1. Cumulative fallback under diverse history $( \rho = 0 . 1 5$ $\sigma = 0 . 3 , R _ { 0 } = 0 )$ . Curves average 256 episodes; error bars mark pointwise 95% intervals at selected rounds.

Separate and Contrast use the frozen ledger with $L ^ { S }$ and $L ^ { \mathrm { J } }$ , respectively. Refresh adds (13). Revalue applies the unknown-baseline cumulative separate-bound geometry of [2], checking only the optimistic proposal; Revalue-F instead selects the best admissible proposal. Both use the current ellipsoid and structural fallback, and are explicit ablations rather than reproductions of the published nested-set algorithm. LinUCB has no safety gate. Revalue-F controls for action filtering when comparing refresh strategies.

![](images/15e743dd98e05096cb150584075cc8f9971c1517822b81d0fe1661580e30f81c.jpg)  
Fig. 2. Reserve sensitivity with baseline-only history $( \rho ~ =$ $0 . 4 , \sigma = 0 . 3 )$ . Points are means with pointwise 95% intervals over 256 episodes; lines connect the tested reserve values.  
Table 2. Zero-reserve reward ratios across noise levels and candidate radii (256 episodes per setting). C: Contrast; C+: Refresh; R-F: Revalue-F; U: LinUCB.

## 4.2. What coupling and refresh change

Table 1 isolates the two effects. With diverse history, replacing Separate by Contrast reduces fallback from 88.2% to 19.4% and increases the reward ratio by 0.0473 (paired 95% interval: 0.0456–0.0490). Refresh reduces fallback further to 0.5%. Figure 1 shows that this is sustained throughout deployment rather than confined to initialization.

<table><tr><td rowspan=1 colspan=1>ρ   σ      C    C+   R-F      U</td></tr><tr><td rowspan=1 colspan=1>Diverse history</td></tr><tr><td rowspan=1 colspan=1>0.15 0.1  1.077 1.077 1.051 1.077</td></tr><tr><td rowspan=1 colspan=1>0.15 0.3  1.056 1.061  1.012  1.061</td></tr><tr><td rowspan=1 colspan=1>0.4 0.1 1.205 1.205  1.183  1.2050.4 0.3 1.099 1.164 1.038 1.173</td></tr><tr><td rowspan=1 colspan=1>0.8 0.1  1.414 1.414 1.401  1.415</td></tr><tr><td rowspan=1 colspan=1>0.8 0.3  1.218  1.358  1.175  1.377</td></tr><tr><td rowspan=1 colspan=1>Baseline-only history0.15 0.1 1.008 1.047 1.015 1.048</td></tr><tr><td rowspan=1 colspan=1>0.15  0.3  1.001  1.019  1.001  1.027</td></tr><tr><td rowspan=1 colspan=1>0.4 0.1  1.018 1.163  1.121  1.184</td></tr><tr><td rowspan=1 colspan=1>0.4 0.3 1.002 1.091  1.007  1.139</td></tr><tr><td rowspan=1 colspan=1>0.8 0.1  1.045  1.311  1.297  1.397</td></tr><tr><td rowspan=1 colspan=1>0.8 0.3  1.003  1.198  1.065  1.350</td></tr></table>

Table 2 varies candidate radius and observation noise without changing the reserve. Refresh improves on the frozen contrast ledger in every displayed setting. Its gap to unconstrained LinUCB is larger with baseline-only history and wider candidate variation, reflecting the cost of learning un familiar directions under a prefix constraint. The revalued comparison is also informative: increasing the admissible choice set can change the data collected, so Revalue-F need not outperform the optimistic-proposal Revalue policy on reward. Fixed-history inclusion is a geometric statement, not a ranking of complete learning trajectories.

Baseline-only history exposes the limitation of a frozen ledger. Contrast still falls back on 91.8% of rounds: repeatedly observing the incumbent does not identify all action directions, and previous pessimistic charges remain in the ledger. Refresh reduces fallback to 12.1%, with reward ratio 1.0190 versus 1.0011 for Revalue-F. Its paired reward gain over Revalue-F is 0.0179 (95% interval: 0.0169–0.0190). This comparison separates shared-prefix certification from merely choosing a different admissible action.

Increasing reserve relaxes the performance requirement and can improve exploration (Fig. 2), but does not eliminate frozen-certificate conservatism. The baseline-only setting shows why reserve and historical information must be varied

separately.

Safety and scope. All five gated variants had zero conditional-mean prefix violations in all tested settings, and all evaluated confidence events held. Unconstrained LinUCB violated the zero-reserve constraint in 16.4% of episodes in Table 1’s baseline-only setting. These observations check implementation behavior; the guarantee comes from Theorem 1, not from empirical coverage alone. Zero failures in 256 episodes imply a one-sided 95% binomial upper bound of 1.16% for one fixed setting, not a simultaneous guarantee across the grid. The study does not establish robustness to nonlinear rewards, drifting parameters, misspecified noise bounds, or performance on live retail traffic. The comparisons isolate certificate mechanisms at a fixed horizon.

## 5. CONCLUSION

Conservative decisions should be certified in the same comparative coordinates as their performance constraint. Shared contrast certificates remove a precisely quantifiable uncertainty penalty; prefix refresh prevents early pessimism from becoming permanent budget debt. Together, they give a transparent reserve-aware filter with an anytime conditional-mean guarantee and measurable reductions in fallback under a fully reproducible linear model.

## ACKNOWLEDGMENTS

ChatGPT (OpenAI) assisted with writing and idea refinement throughout the paper, and with analysis, code, and figures in Sections 2–4.

## 6. REFERENCES

[1] Yifan Wu, Roshan Shariff, Tor Lattimore, and Csaba Szepesvári, “Conservative bandits,” in Proceedings of the 33rd International Conference on Machine Learn ing, 2016, vol. 48 of Proceedings of Machine Learning Research, pp. 1254–1262.

[2] Abbas Kazerouni, Mohammad Ghavamzadeh, Yasin Abbasi-Yadkori, and Benjamin Van Roy, “Conservative contextual linear bandits,” in Advances in Neural Information Processing Systems, 2017, vol. 30.

[3] Xiaojin Zhang, Shuai Li, Weiwen Liu, and Shengyu Zhang, “Contextual combinatorial conservative bandits,” arXiv preprint arXiv:1911.11337, 2019.

[4] Evrard Garcelon, Mohammad Ghavamzadeh, Alessandro Lazaric, and Matteo Pirotta, “Improved algorithms for conservative exploration in bandits,” arXiv preprint arXiv:2002.03221, 2020.

[5] Yunchang Yang, Tianhao Wu, Han Zhong, Evrard Garcelon, Matteo Pirotta, Alessandro Lazaric, Liwei Wang, and Simon S. Du, “A reduction-based framework for conservative bandits and reinforcement learning,” arXiv preprint arXiv:2106.11692, 2021.

[6] Rohan Deb, Mohammad Ghavamzadeh, and Arindam Banerjee, “Conservative contextual bandits: Beyond linear representations,” arXiv preprint arXiv:2412.06165, 2024.

[7] Marek Petrik, Yinlam Chow, and Mohammad Ghavamzadeh, “Safe policy improvement by minimizing robust baseline regret,” in Advances in Neural Information Processing Systems, 2016, vol. 29.

[8] Nikos Karampatziakis, Paul Mineiro, and Aaditya Ramdas, “Off-policy confidence sequences,” arXiv preprint arXiv:2102.09540, 2021.

[9] Yasin Abbasi-Yadkori, Dávid Pál, and Csaba Szepesvári, “Improved algorithms for linear stochastic bandits,” in Advances in Neural Information Processing Systems, 2011, vol. 24.

[10] Jing Lei, Max G’Sell, Alessandro Rinaldo, Ryan J. Tibshirani, and Larry Wasserman, “Distribution-free predictive inference for regression,” Journal of the American Statistical Association, vol. 113, no. 523, pp. 1094– 1111, 2018.