# The Singularity Has a Distribution

Exact Timing Laws and Detection Limits for a Stochastic Intelligence Explosion

Xiaoyu Li<sup>1</sup> · Zheng Gao<sup>1</sup> · Chiwun Yang<sup>2</sup> · Zhizhou Sha<sup>3</sup>

<sup>1</sup>University of New South Wales <sup>2</sup>City University of Hong Kong <sup>3</sup>University of Texas at Austin

Correspondence: xiaoyulics@gmail.com

## ABSTRACT

Automating AI research creates a positive feedback loop between software capability and research efort. We study a stochastic realisation of this loop in which independent, positive, multiplicative improvements arrive at a rate proportional to a power of current capability. Within this model, randomness leaves the deterministic explosion threshold unchanged but replaces the blow-up date by an explicit distribution. For equal-sized improvements the explosion time is a geometrically weighted sum of independent exponentials; its �-series representation gives its transforms, moments and tail behaviour, including a two-term small-time asymptotic. For Lomax-distributed improvement sizes the explosion time is exactly Gamma distributed. Self-similarity implies that, with parameters known, the relative uncertainty about the remaining time does not vanish as explosion approaches. We formulate a fixed-sample test between constant and geometrically increasing idea-arrival rates. If the baseline rate and the alternative log-rate increment ℓ are known, the minimum sample size at error level � is asymptotic to $( 1 2 z _ { \alpha } ^ { 2 } ) ^ { 1 / 3 } \ell ^ { - 2 / 3 }$ , where $z _ { \alpha }$ is the upper-� standard normal quantile. Uncertainty about the growth exponent near its critical value produces a $1 / t$ tail with an explicit constant, and hence an infinite conditional mean waiting time, although explosion is certain given a positive exponent. Separate extensions show that a fixed deployment delay prevents explosion in the specified delayed-feedback models, that power-law diversion of efort shifts the growth threshold, and that in a deterministic oversight race a fixed allocation share does not change the asymptotic regime, except at a critical equality of exponents. Numerical illustrations show how timing and detectability depend on the granularity of improvements. They are not empirical forecasts of AI development.

KEYWORDS intelligence explosion; pure birth process; explosion time; �-series; exponential functionals; perpetuities; hypothesis testing; delay differential equations   
MSC 2020 60J27, 60G51, 33D15, 62F03, 34K25, 91B62

![](images/86e3852ba98ace93e4972d3855016704308f0dd08ff12eb5e441b887ce4329f1.jpg)  
Figure 1 The singularity has a distribution. Left: 1 400 simulated histories, coloured by explosion time, rise as staircases of ideas and leave the sky at their own explosion times; the fluid ODE (dashed) has a single date. The horizon glows with the exact law of � (here Gamma, Corollary 3.9), and its ticks mark $\mathbb { E } T _ { n } \to \mathbb { E } \zeta .$ . Right: the same histories with radius � � and angle log � , each a stepped logarithmic spiral into the singularity (Section 4). Details in Section A.

## Contents

1 Introduction 2   
2 The model and its calibration 5   
3 Explosion criterion and exact timing law 7   
4 Residual-time forecasting and parameter uncertainty 11   
5 Fixed-sample detection and the cube-root law 14   
6 Model extensions: delay, diversion and an oversight race 17   
7 Numerical illustrations 20   
8 Limitations and open problems 20   
A Numerical specification and reproducibility 21

## 1 Introduction

## 1.1 A feedback law and what it leaves out

In 1960, von Foerster, Mora and Amiot fitted a hyperbola $N ( t ) \propto ( t ^ { * } - t ) ^ { - 1 }$ to two millennia of world population and dated its singularity to Friday, 13 November, A.D. 2026 [37]. That date is now a month away, and the world population remains finite. Hyperbolic growth is back all the same, this time as a hypothesis about software.

Chan et al. [12] argue that the automation of AI research and development could trigger an intelligence explosion, in which AI progress, driven by AI itself, speeds up so much that years of advances happen within months. The mechanism is a feedback loop: AI systems enlarge the efective research workforce, and that workforce produces better AI systems. The quantitative core of their supplementary materials is the idea production function of semi-endogenous growth theory [22, 9], $\dot { A } = A ^ { 1 - \beta } E ^ { \lambda }$ . Here � is software quality, � is efective research labour, � measures returns to scale in labour and $\beta$ measures how fast ideas get harder to find. Under full automation $E = k A ,$ , and

$$
\begin{array} { r } { \dot { A } = c A ^ { 1 + \varepsilon } , \qquad \varepsilon : = \lambda - \beta = \beta ( r - 1 ) , \qquad r : = \lambda / \beta . } \end{array}\tag{1.1}
$$

When $r > 1 , ( 1 . 1 )$ reaches infinity at the finite time $T ^ { * } = ( \varepsilon c A _ { 0 } ^ { \varepsilon } ) ^ { - 1 }$ . The authors read this “infinite progress in finite time” as the point where the model breaks down, not as a prediction [12, SM]. Their central estimates are $\lambda = 1 . 4 0$ and $\beta = 1 . 0 1$ , so that $\varepsilon = 0 . 3 9$ and $r = \lambda / \beta = 1 . 3 8 6 ;$ we use these values for illustration only.

The deterministic law cannot say how uncertain the blow-up date is, how early an observer could tell acceleration from steady exponential growth, or what a deployment lag does to the singularity. These are the questions of this paper. We keep the drift of (1.1), replace its continuum of progress by discrete, randomly timed improvements, and ask what can be computed exactly.

## 1.2 AI-generated mathematics and the research feedback loop

Recent developments in AI-assisted mathematical research make the feedback mechanism in (1.1) more concrete, while showing how dificult it is to measure. In September 2026, OpenAI reported an AI-generated analytical argument, with a Lean formalisation, for finite-time singularity formation in the three-dimensional Navier–Stokes equations with a smooth external force [28]. The Clay Mathematics Institute described the problem as apparently settled, while noting that its evaluation and credit-assignment procedures would take time [13]. The forcing assumption matters: this is not a claimed resolution of the corresponding unforced regularity question. It is also a mathematical singularity of a diferent kind from the one studied here.

On 6 October 2026, OpenAI released a repository of mathematical manuscripts produced by an internal model [30, 31]. As released it held 722 manuscripts grouped into 372 families of related results, with computer-checkable Lean formalisations for many, but not all, of them; its maintainers caution that some unformalised results could have issues [31]. A day later three manuscripts were withdrawn after a sign error was found, and the repository now lists 719 [31]. These counts are of manuscripts, not of independent, externally certified breakthroughs, and the withdrawal shows that validation continues after release. Such distinctions matter whenever a “new idea” is treated as an observable event.

The connection to AI research automation is more direct, but still incomplete. In September 2026, OpenAI reported increasing use of coding agents within its research organisation: by mid-August it measured 3.1 agent-workdays of runtime per human workday, together with more experiment activity and delegation of more complex tasks [29]. These are operational indicators, not estimates of efective research labour $E ,$ of the returns ratio $r = \lambda / \beta$ , or of the exponent $\varepsilon = \beta ( r - 1 )$ . The same report notes confounding from increased compute and the continued importance of human steering and bottlenecks.

These observations motivate a mathematical study of research acceleration; they do not establish that a supercritical feedback loop exists. They also sharpen the distinction between producing a candidate result and validating, understanding and incorporating it. The ideal observer of Section 5 sees every efective idea and its arrival time, whereas actual observers see manuscripts, experiments, model releases and incomplete verification records. This gap is a reason to study statistical detectability, deployment lags and the limits of extrapolation, and not to read the calibrated explosion dates below as empirical forecasts.

## 1.3 Model, assumptions and contributions

The model. In state �, the next improvement (an “idea”) arrives after an exponential time with rate $\kappa A ^ { \varepsilon }$ and multiplies � by $1 + H$ , where the sizes � are i.i.d. and strictly positive. We call this a Good process, after I.J. Good’s ultraintelligent machine [18]. When all ideas have the same size $h ,$ the idea count is a pure birth process with geometric rates $q _ { n } = q _ { 0 } \rho ^ { n } , \rho = ( 1 + h ) ^ { \iota }$ , and its explosion time is

$$
\zeta = \sum _ { n \geq 0 } { \frac { e _ { n } } { q _ { 0 } \rho ^ { n } } } , \qquad e _ { n } { \mathrm { ~ i . i . d . ~ } } \operatorname { E x p } ( 1 ) .
$$

Throughout, for $\varepsilon > 0 , q : = 1 / \rho \in ( 0 , 1 )$ , and $\ell : = \log \rho = \varepsilon \log ( 1 + h )$ is the increment of the log-rate per idea. The model is one tractable stochastic realisation of $( 1 . 1 )$ , not the unique one, and Remark 2.2 lists which assumptions support which result.

Results. Write $\bar { \zeta } : = q _ { 0 } \zeta$ for the standardised explosion time.

1. Explosion criterion (Theorem 3.1). For every i.i.d. positive idea-size law in this model, explosion is almost sure $\mathrm { i f } r > 1$ and impossible $\mathrm { i f } \ r \leq 1$ . For equal sizes and a monotone rate function, explosion of the chain is equivalent to blow-up of its fluid equation.

2. Exact timing law (Theorems 3.4 and 3.7 and Corollary 3.9). For equal sizes, $\bar { \zeta }$ has Laplace transform $1 / ( - s ; q ) _ { \infty } ,$ , moments $n ! / ( q ; q ) _ { n }$ , Mellin transform $\Gamma ( 1 + s ) ( q ^ { 1 + s } ; q ) _ { \infty } / ( q ; q ) _ { \infty }$ and an alternating �-series distribution function. Its slow tail is $\begin{array} { r } { \mathbb { P } ( \bar { \zeta } > t ) \sim e ^ { - t } \sum _ { m } p ( m ) q ^ { m } } \end{array}$ , and at fixed $q$ it has the two-term small-time asymptotic log $\begin{array} { r } { \mathbb { P } ( \bar { \zeta } \leq t ) = - \frac { L ^ { 2 } } { 2 \ell } - \frac { L \log L } { \ell } + \overleftarrow { O } ( L ) , L = \log ( 1 / t ) } \end{array}$ . For random sizes $\bar { \zeta }$ is a perpetuity with explicit moments, and when $\log ( 1 + H )$ is exponential (Lomax sizes), $\bar { \zeta }$ is exactly Gamma distributed. Most of these formulas are known (Table 1); we give self-contained proofs.

3. Residual-time self-similarity (Theorem 4.1 and Propositions 4.5 and 4.6). With parameters fixed and known, at every stopping time � the remaining time $\zeta - \tau _ { \mathrm { { ; } } }$ , multiplied by the current idea rate, has the law of $\bar { \zeta } ,$ independent of the past. Relative predictive uncertainty therefore does not vanish as explosion approaches. In coordinates centred on the realised explosion time, the mean density of ideas is log-periodic with harmonics $\Gamma ( 1 - 2 \pi i k / \ell )$ . If the growth exponent is uncertain with positive density at the critical value, then $t \mathbb { P } ( \zeta > t \mid \varepsilon > 0 )$ converges to an explicit constant.

4. Fixed-sample detection (Theorem 5.1 and Corollary 5.3). For the binary experiment with known baseline rate and known increment ℓ, decided once after the first � waiting times, the minimum sample size at level � satisfies $n ^ { * } \sim ( 1 2 z _ { \alpha } ^ { 2 } ) ^ { 1 / 3 } \ell ^ { - 2 / 3 }$ , with non-asymptotic bounds on both sides.

5. Model extensions (Theorems 6.1 and 6.3 and Proposition 6.2). In a deterministic delayed-feedback model any fixed delay $D > 0$ prevents blow-up and gives log log $A ( t ) \sim t \log r / D ;$ the corresponding stochastic model does not explode. Power-law diversion of efort changes the explosion threshold. In a deterministic race with a fixed allocation share, the asymptotic regime of the oversight-to-capability ratio is determined by exponents, except at a critical equality where the allocation enters the limit.

Table 1 Contribution map. “Known” means that the formula appears in the cited literature; we give self-contained proofs because the application reading is the point. Novelty claims are limited to what we checked (Section 1.4).
<table><tr><td>Component</td><td>Where</td><td>Status</td></tr><tr><td>Explosion criterion; exponential-functional representation</td><td>Thm 3.1; Rem. 2.3</td><td>classical structure applied to (1.1)</td></tr><tr><td>Laplace transform, density, integer and fractional moments; random-size moments</td><td>Thm 3.4(i)–(iii), (vi); Thm 3.7</td><td>known [11, 15, 5, 19]</td></tr><tr><td>Slow tail from the first exponential mode</td><td>Theorem 3.4(iv)</td><td>explicit consequence of the known density</td></tr><tr><td>Two-term small-time asymptotic</td><td>Theorem 3.4(v)</td><td>known in sharper form [4, 7]; self-contained proof</td></tr><tr><td>Lomax sizes give a Gamma explosion time</td><td>Corollary 3.9</td><td>classical [36, 8]; derived from Theorem 3.7</td></tr><tr><td>Residual-time self-similarity as a forecasting statement</td><td>Theorem 4.1</td><td>direct consequence of self-similarity</td></tr><tr><td>Γ-coefficients of the explosion-centred intensity</td><td>Proposition 4.5</td><td>exact specialisation</td></tr><tr><td>Exact 1/t tail constant under parameter uncertainty</td><td>Proposition 4.6</td><td>routine Abelian argument; not found stated for this model</td></tr><tr><td>Sharp fixed-sample detection threshold</td><td>Theorem 5.1</td><td>statistical centrepiece of the paper</td></tr><tr><td>Delay, diversion and oversight race</td><td>Section 6</td><td>separate model extensions with their own scopes</td></tr></table>

Table 1 separates what is classical from what may be new.

## 1.4 Related work and organisation

Finite-time singularities in growth models go back at least to [37]. In economics they appear in Kremer’s model of population and technology [23] and in discussions of AI and growth [1]. The modelling of a software intelligence explosion that we start from is surveyed in [12], which builds on estimates of � by Ho and Whitfill [21], on the feedback-loop analysis of Davidson et al. [14], and on Ord’s discussion of the time between rounds of improvement [32].

On the probabilistic side, the explosion time of a Good process is the exponential functional of a compound Poisson subordinator in Lamperti’s representation [24], and such functionals are well studied [11, 6]. The Poisson case, which is our lattice case, arose in the analysis of AIMD congestion-control algorithms [15, 19]. Guillemin, Robert and Zwart [19] give its density as an alternating series, its Laplace transform as a �-series, and the moment formulas of Theorem 3.4(ii) and Theorem 3.7. Bertoin, Biane and Yor [5] connected it to �-integrals and the log-normal moment problem and obtained its fractional moments, which give the Mellin formula (vi). Perpetuities of the form in Theorem 3.7 go back at least to Vervaat [36]. The Gamma example of Corollary 3.9 is classical; it is stated explicitly, with a reference to [36], in [8, Example 2.1]. The small-time asymptotics of geometrically weighted sums of i.i.d. positive variables are known in sharper form. Aurzada [4] gives the leading term, and Borovkov and Ruzankin [7, Theorem 1.1] give the second term together with an explicit � � constant. Theorem 3.4(v) is a weaker form of their result, included with a short self-contained proof. We did not find Proposition 4.6 stated for this model; it follows by a routine Abelian-type argument. We are not aware of earlier treatments of the residual-time invariance as a forecasting statement, of the Γ-coeficients of the explosion-centred intensity, or of the detection, delay and race results in this context. These statements are limited to the sources we checked.

Organisation. Section 2 defines Good processes, states the modelling assumptions and fixes the calibration. Section 3 proves the explosion criterion and the exact timing law. Section 4 treats forecasting from inside the explosion and parameter uncertainty, and Section 5 the fixed-sample detection problem. Section 6 collects the separate model extensions. Section 7 gives numerical illustrations, and Section 8 lists limitations and open problems. Section A specifies the numerical methods and how to reproduce every number.

## 2 The model and its calibration

Definition 2.1 Good process Let $A _ { 0 } > 0 , \mathnormal { x } > 0 , \mathnormal { \varepsilon } \in \mathbb { R }$ , and let $\mu$ be a probability measure on $( 0 , \infty )$ , the idea-size law. Let $\textstyle ( e _ { n } ) _ { n \geq 0 }$ be i.i.d. Exp 1 and $( H _ { n } ) _ { n \geq 1 }$ i.i.d. with law $\mu ,$ all independent. Put

$$
A ^ { ( 0 ) } : = A _ { 0 } , \qquad A ^ { ( n ) } : = A ^ { ( n - 1 ) } ( 1 + H _ { n } ) , \qquad \tau _ { n } : = \frac { e _ { n } } { \tau ( A ^ { ( n ) } ) ^ { \varepsilon } } , \qquad T _ { n } : = \sum _ { k < n } \tau _ { k } ,
$$

and $\zeta : = \mathrm { l i m } _ { n } T _ { n } \in ( 0 , \infty ]$ . The Good process is $A _ { t } : = A ^ { ( n ) }$ for $T _ { n } \leq t < T _ { n + 1 }$ , with $A _ { t } : = \infty$ for $t \geq \zeta .$ . Its idea count is $N _ { t } : = n$ on $[ T _ { n } , T _ { n + 1 } ) , ( { \mathcal F } _ { t } )$ is the usual augmentation of the filtration generated by $\left( A _ { t } \right)$ and � is its explosion time $( \zeta = \infty$ means no explosion).

In state � the process waits an exponential time with rate ��<sup>�</sup> and then jumps to $a ( 1 + H )$ . It is a pure-jump Markov process, minimal in the sense that nothing is defined after the first explosion. We call � the explosion time or the singularity time of the model; it is not a separately defined capability threshold.

Remark 2.2 Scope of the model We study a tractable stochastic realisation of the feedback law (1.1), not a unique stochastic model implied by that law. The assumptions of independent positive multiplicative improvements and exponential holding times make the clock explicit. Our robustness statements concern the specified family of processes; they do not establish robustness to arbitrary dependence, negative shocks, time-varying returns or alternative observation schemes. The assumptions are used as follows.

• Geometric rates together with exponential holding times produce the �-products of Theorem 3.4.

• The memoryless property of the holding times is what makes the residual-time statement of Theorem 4.1 hold at arbitrary stopping times, not just at idea times.

• The explosion criterion of Theorem 3.1(b) needs i.i.d. strictly positive sizes but no moment condition on $H ;$ matching a finite level drift through �� (Box 1, convention A) does need $\mathbb { E } H < \infty$

• What counts as one “idea” is a modelling choice. Grouping several exponential waiting stages into one idea generally produces a non-exponential holding time, so a coarser idea definition is not automatically another Good process.

The lattice case. When $H \equiv h$ we write

$$
\begin{array} { r } { q _ { n } : = \kappa A _ { 0 } ^ { \varepsilon } ( 1 + h ) ^ { \varepsilon n } = q _ { 0 } \rho ^ { n } , \qquad \rho : = ( 1 + h ) ^ { \varepsilon } , \qquad q : = \rho ^ { - 1 } , \qquad \ell : = \log \rho = \varepsilon \log ( 1 + h ) . } \end{array}
$$

The waiting time in state � is $\textstyle { e _ { n } } / q _ { n } ,$ , and

$$
\zeta = \sum _ { n \geq 0 } { \frac { e _ { n } } { q _ { 0 } \rho ^ { n } } } , \qquad { \bar { \zeta } } : = q _ { 0 } \zeta = \sum _ { n \geq 0 } e _ { n } q ^ { n } .\tag{2.1}
$$

The law of <sup>¯</sup>� depends on � alone; we denote it $\Lambda _ { q } .$ . Thus $\bar { \zeta }$ measures time in units of the current expected waiting time $1 / q _ { 0 }$

Remark 2.3 Lamperti representation Let � be a unit-rate Poisson process independent of $( H _ { i } )$ , and let $\xi _ { s } : =$ $\textstyle \sum _ { i \leq M _ { s } } \log ( 1 + H _ { i } )$ , a compound Poisson subordinator. Measure time in “ideas” � and convert it to calendar time through $\mathrm { d } t = \mathrm { d } s / ( \kappa A ^ { \varepsilon } )$ . Then $A _ { t } = A _ { 0 } \exp ( \xi _ { \sigma ( t ) } )$ , where � is the inverse of $s \mapsto ( \kappa A _ { 0 } ^ { \varepsilon } ) ^ { - 1 } \int _ { 0 } ^ { s } e ^ { - \varepsilon \xi _ { u } }$ d�, and

$$
\zeta = ( \kappa A _ { 0 } ^ { \varepsilon } ) ^ { - 1 } \int _ { 0 } ^ { \infty } e ^ { - \varepsilon \xi _ { s } } \mathrm { d } s .
$$

Thus � is a positive self-similar Markov process in the sense of Lamperti [24], with a negative index: if � starts at $^ { a , }$ then $( c A _ { c ^ { \varepsilon } t } ) _ { t \geq 0 }$ has the law of � started at ��. Moreover $\zeta$ is an exponential functional of the subordinator �� [11, 6].

## 2.1 Calibration: what is held fixed

A drift law does not determine a jump process, and the phrase “doubling time” has at least three meanings for one. Box 1 separates them. All calibrated numbers in this paper use convention B.

## BOX 1 Three meanings of doubling time

<sup>Write</sup> L <sup>for</sup> <sup>the</sup> <sup>generator</sup> <sup>of</sup> <sup>the</sup> <sup>lattice</sup> <sup>Good</sup> <sup>process:</sup> ${ \mathcal { L } } f ( a ) = x a ^ { \varepsilon } \left( f ( a ( 1 + h ) ) - f ( a ) \right)$ .

A. Local level drift. $\mathcal { L } \operatorname { i d } ( a ) = \kappa h a ^ { 1 + \varepsilon }$ . Matching (1.1) requires $\textstyle { \boldsymbol { x } } = { \boldsymbol { c } } / h _ { : }$ so $q _ { 0 } = c A _ { 0 } ^ { \varepsilon } / h$ . (For random sizes, � �� replaces �ℎ, which needs $\mathbb { E } H < \infty . )$

B. Local log drift (used throughout). $g : = \mathcal { L } \log ( A _ { 0 } ) = q _ { 0 } \log ( 1 + h )$ . We set $g = \log 2 / d _ { 0 }$ with $d _ { 0 } = 4 . 5$ months, so $g = 0 . 1 5 4 0$ per month and

$$
q _ { 0 } = \frac { g } { \log ( 1 + h ) } .
$$

Here �<sub>0</sub> is a log-drift-equivalent doubling-time proxy; it is not in general the expected time to reach $2 A _ { 0 }$ . We say “local” because unconditional expectations such as � log �<sub>�</sub> are infinite for every $t > 0$ once explosion before � has positive probability.

C. First-passage doubling. For the ODE with $c A _ { 0 } ^ { \varepsilon } = g _ { 0 }$ , the actual time to reach $2 A _ { 0 }$ is $( 1 - 2 ^ { - \varepsilon } ) / ( \varepsilon g _ { 0 } )$ , which equals log $2 / g _ { 0 }$ only in the limit $\varepsilon  0 .$ For the lattice process,

$$
\mathbb { E } T ( 2 A _ { 0 } ) = \frac { 1 - q ^ { m } } { q _ { 0 } ( 1 - q ) } , \qquad m = \bigg \lceil \frac { \log 2 } { \log ( 1 + h ) } \bigg \rceil .
$$

Numbers at $\varepsilon = 0 . 3 9 , d _ { 0 } = 4 . 5$ . With $g _ { 0 } = g ,$ the ODE blows up at $T ^ { * } = 1 / ( \varepsilon g ) = 1 6 . 6 5$ months, and its actual first doubling takes 3.94 months. If instead the ODE’s actual first doubling is fixed at 4.5 months, it blows up at 19.00 months. Under convention B the mean first doubling of the lattice process is 3.97, 4.17, 4.36, 4.50 and 14.95 months for $h = 0 . 0 1 , 0 . 0 5 , 0 . 1 , 1 , 9$ . Only $h = 1$ gives exactly $d _ { 0 } ;$ the $h = 9$ process overshoots $2 A _ { 0 }$ at its first tenfold jump, so its first-doubling time is the whole initial wait.

A versus B. If the fluid coeficient $c A _ { 0 } ^ { \varepsilon } = g$ is held fixed, $\mathrm { A g i v e s } \ q _ { 0 } = g / h$ and B gives $q _ { 0 } = g / \log ( 1 + h )$ ; they difer for every $h > 0 .$ If instead $c A _ { 0 } ^ { \varepsilon } = 1 / d _ { 0 }$ they coincide at $h = 1$ , but then the level-drift ODE blows up at $d _ { 0 } / \varepsilon = 1 1 .$ 54 months.

The ODE (1.1) is the small-idea limit of Good processes in the following sense.

Proposition 2.4 Fluid limit of hitting times Consider the lattice case with $\tau = c / h$ (convention A) for fixed $c > 0 , \varepsilon \in \mathbb { R }$ and $A _ { 0 } > 0$ . For $a > A _ { 0 }$ let $T ^ { h } ( a ) : = \operatorname* { i n f } \{ t : A _ { t } \geq a \}$ and $\textstyle t ( a ) : = \int _ { A _ { 0 } } ^ { a } \mathrm { d } x / ( c x ^ { 1 + \varepsilon } )$ the time the solution of (1.1) takes to reach �. Then $T ^ { h } ( a )  t ( a )$ in $L ^ { 2 }$ as ℎ 0. If $\varepsilon > 0 ,$ then also $\zeta ^ { h }  T ^ { * } = ( \varepsilon c A _ { 0 } ^ { \varepsilon } ) ^ { - 1 } \mathrm { i n } L ^ { 2 }$

Proof. Let $x _ { k } : = A _ { 0 } ( 1 + h ) ^ { k }$ and $n : = \operatorname* { m i n } \{ k : x _ { k } \geq a \}$ , so that $\begin{array} { r } { T ^ { h } ( a ) = T _ { n } = \sum _ { k < n } e _ { k } / q _ { k } } \end{array}$ with $q _ { k } = c x _ { k } ^ { \varepsilon } / h$ Since $x _ { k + 1 } - x _ { k } = h x _ { k }$

$$
\mathbb { E } T _ { n } = \sum _ { k < n } { \frac { h } { c x _ { k } ^ { \varepsilon } } } = \sum _ { k < n } { \frac { x _ { k + 1 } - x _ { k } } { c x _ { k } ^ { 1 + \varepsilon } } }
$$

is a Riemann sum for $\int _ { A _ { 0 } } ^ { x _ { n } } \mathrm { d } x / ( c x ^ { 1 + \varepsilon } )$ with mesh at most ℎ� $\left( 1 + h \right)$ . Since $x _ { n } \ \in \ \left[ a , a ( 1 + h ) \right)$ , we get $\mathbb { E } T _ { n } \to t ( a )$ . Moreover Var $\begin{array} { r } { T _ { n } = \sum _ { k < n } q _ { k } ^ { - 2 } \le ( \operatorname* { m a x } _ { k < n } q _ { k } ^ { - 1 } ) \mathbb { E } T _ { n } } \end{array}$ , and max<sub>�<�</sub> $q _ { k } ^ { - 1 } \leq$ ℎ max $\left( A _ { 0 } ^ { - \varepsilon } , a ^ { - \varepsilon } \right) / c \to 0$ Hence $\mathbb { E } | T _ { n } - t ( a ) | ^ { 2 } = \operatorname { V a r } T _ { n } + ( \mathbb { E } T _ { n } - t ( a ) ) ^ { 2 } \to 0 . \operatorname { F o r } \varepsilon > 0 ,$ (2.1) gives

$$
\mathbb { E } \zeta ^ { h } = \frac { h } { c A _ { 0 } ^ { \varepsilon } ( 1 - ( 1 + h ) ^ { - \varepsilon } ) } \to \frac { 1 } { \varepsilon c A _ { 0 } ^ { \varepsilon } } , \quad \quad \mathrm { V a r } \zeta ^ { h } = \frac { h ^ { 2 } } { c ^ { 2 } A _ { 0 } ^ { 2 \varepsilon } ( 1 - ( 1 + h ) ^ { - 2 \varepsilon } ) } \to 0 .
$$

The proposition concerns hitting times; it is not a path-space limit theorem. Under convention B the same limit holds, because $h / \log ( 1 + h ) \to 1$

![](images/5045d4411ed667529597718611388ad85d8118b1a401a726acd697f61427b007.jpg)

![](images/d41237cda76968f3081497adccff2e216093ffcbc7d23276fdd5136a902f40c2.jpg)  
Figure 2 (a) Sample paths of log<sub>2</sub> $A _ { t } \mathrm { a t } \varepsilon = 0 . 3 9$ under convention B (local log drift $g = \log 2 / 4 . 5$ per month held fixed), for doubling-sized ideas $( h = 1 ;$ , whose first doubling is expected after exactly 4.5 months) and small ideas $( h = 0 . 0 5$ whose first doubling is expected after 4.17 months). The dashed curve is the fluid ODE with $c A _ { 0 } ^ { \varepsilon } = g ,$ the small-idea limit, which blows up at $T ^ { * } = 1 6 . 6 5$ months. (b) Exact density of the explosion time (Theorem 3.4(iii)) for three idea sizes under convention B, with a Monte Carlo histogram for $h = 1$

## 3 Explosion criterion and exact timing law

Theorem 3.1 Explosion criterion

(a) Lattice sizes, arbitrary rates. Let $h > 0 ,$ let $\nu : ( 0 , \infty ) \to ( 0 , \infty )$ be any function, and consider the pure birth chain with rates $q _ { n } = \nu \big ( A _ { 0 } \big ( 1 + h \big ) ^ { n } \big )$ , i.e. the lattice process with rate function � in place of $\boldsymbol { \kappa } \boldsymbol { a } ^ { \varepsilon } .$ . Then $\mathbb { P } ( \zeta < \infty ) = 1$ if $\textstyle \sum _ { n } 1 / q _ { n } < \infty ,$ and $\mathbb { P } ( \zeta = \infty ) = 1$ otherwise. If � is monotone, then $\textstyle \sum _ { n } { 1 / q _ { n } } < \alpha$ if and only if

$$
\int _ { A _ { 0 } } ^ { \infty } \frac { \mathrm { d } x } { x \nu ( x ) } < \infty ,
$$

which is the Osgood condition for finite-time blow-up of the fluid equation ${ \dot { a } } = h a \nu ( a )$

(b) Power rates, arbitrary sizes. For the Good process with any i.i.d. strictly positive idea-size law �: if $\varepsilon > 0$ then $\mathbb { E } [ \zeta ^ { m } ] < \infty$ for every � $\geq 1 ,$ so in particular $\zeta < \infty \mathrm { a . s . ; i f } \varepsilon \leq 0$ then $\zeta = \infty \mathtt { a } . s .$ No moment condition on � is needed.

Proof. (a) Here $\zeta = \textstyle \sum _ { n } e _ { n } / q _ { n }$ with deterministic $q _ { n } . \mathrm { ~ I f ~ } \sum 1 / q _ { n } <$ then $\mathbb { E } \zeta < \infty$ . If $\textstyle \sum 1 / q _ { n } = \infty ,$ then $\textstyle \sum _ { n } \log ( 1 + 1 / q _ { n } ) = \infty$ <sup>(use</sup> <sup>log</sup>( $1 + x ) \geq x / 2$ for $x \leq 1 ) ,$ so $\begin{array} { r } { \mathbb { E } e ^ { - \zeta } = \prod _ { n } \bigl ( 1 + 1 / q _ { n } \bigr ) ^ { - 1 } = 0 } \end{array}$ and $\zeta = \infty \mathsf { a } . s .$ For the integral test, suppose � is nondecreasing. Then $u \mapsto 1 / \nu \big ( A _ { 0 } \big ( 1 + h \big ) ^ { u } \big )$ is nonincreasing, so its sum over $u \in \mathbb { N }$ and its integral over $[ 0 , \infty )$ converge together. The substitution $x = A _ { 0 } ( 1 + h ) ^ { u }$ turns the integral into $\begin{array} { r } { ( \log ( 1 + h ) ) ^ { - 1 } \int _ { A _ { 0 } } ^ { \infty } \mathrm { d } x / ( x \nu ( x ) ) } \end{array}$ . If � is nonincreasing, both the sum and the integral diverge.

(b) Let $Y _ { i } : = ( 1 + H _ { i } ) ^ { - \varepsilon }$ . Then $\begin{array} { r } { \bar { \zeta } : = \kappa A _ { 0 } ^ { \varepsilon } \zeta = \sum _ { n } e _ { n } \prod _ { i \leq n } Y _ { i \cdot } \operatorname { I f } \varepsilon > 0 , } \end{array}$ then $Y _ { i } \in ( 0 , 1 ) \mathrm { a } . s . , s 0 y _ { m } : = \mathbb { E } Y ^ { m } < 1$ for every �. By Minkowski’s inequality in $L ^ { m }$ and independence,

$$
\| \bar { \zeta } \| _ { m } \leq \sum _ { n \geq 0 } \| e _ { n } \| _ { m } \bigg \| \prod _ { i \leq n } Y _ { i } \bigg \| _ { m } = ( m ! ) ^ { 1 / m } \sum _ { n \geq 0 } y _ { m } ^ { n / m } < \infty .
$$

If $\varepsilon \leq 0 ,$ then $Y _ { i } \geq 1$ and $\begin{array} { r } { \bar { \zeta } \ge \sum _ { n } e _ { n } = \infty \mathrm { a } . s } \end{array}$

Corollary 3.2 The threshold is $r = 1$ In the parametrisation of (1.1), with $\beta > 0 ,$ a Good process explodes almost surely i $\mathrm { \Delta } \mathrm { f } \ r > 1$ and never explodes if $r \leq 1$ , for every i.i.d. strictly positive idea-size law and every �. Within this model, randomness changes when the explosion happens, but not whether it does.

Remark 3.3 Saturation Part (a) also covers the working paper’s observation that � must eventually fall below 1 because of physical limits [12, note 5]. Any monotone rate function with $\begin{array} { r } { \int ^ { \infty } \mathrm { d } x / ( x \nu ( x ) ) = \overset { \cdot } { \infty } , } \end{array}$ for

example one that saturates, prevents explosion almost surely. The singularity is a property of the idealised power law.

We use the �-Pochhammer symbols $\begin{array} { r } { ( z ; q ) _ { n } : = \prod _ { j = 0 } ^ { n - 1 } ( 1 - z q ^ { j } ) \mathrm { a n d } ( z ; q ) _ { \infty } : = \prod _ { j \geq 0 } ( 1 - z q ^ { j } ) } \end{array}$ , and Euler’s two identities [2, Ch. 2], [17, §1.3]:

$$
\sum _ { n \geq 0 } { \frac { z ^ { n } } { ( q ; q ) _ { n } } } = { \frac { 1 } { ( z ; q ) _ { \infty } } } \quad ( | z | < 1 ) ,\tag{3.1}
$$

$$
\sum _ { n \geq 0 } { \frac { ( - 1 ) ^ { n } q ^ { n ( n - 1 ) / 2 } z ^ { n } } { ( q ; q ) _ { n } } } = ( z ; q ) _ { \infty } \quad ( z \in \mathbb { C } ) .\tag{3.2}
$$

Recall also that $\begin{array} { r } { 1 / ( q ; q ) _ { n } = \sum _ { m \ge 0 } p _ { \leq n } ( m ) q ^ { m } } \end{array}$ , where $p _ { \leq n } ( m )$ is the number of partitions of � into parts of size at most �, and that $\begin{array} { r } { 1 / ( q ; q ) _ { \infty } = \sum _ { m } p ( m ) q ^ { m } = : \mathcal { P } ( q ) } \end{array}$ is the partition generating function [2, Ch. 1].

Theorem 3.4 Exact law, lattice case Let $q \in ( 0 , 1 ) , \ell = \log ( 1 / q )$ , and $\begin{array} { r } { \bar { \zeta } = \sum _ { n \geq 0 } e _ { n } q ^ { n } } \end{array}$ as in (2.1).

(i) Laplace transform. $\mathbb { E } e ^ { - s \bar { \zeta } } = 1 / ( - s ; q ) _ { \infty } { \mathrm { ~ f o r ~ } } s > - 1$

(ii) Moments count partitions. For $n \in \mathbb { N } ,$

$$
\mathbb { E } \bar { \zeta } ^ { n } = \frac { n ! } { ( q ; q ) _ { n } } , \qquad \mathrm { e q u i v a l e n t l y } \qquad \frac { \mathbb { E } \bar { \zeta } ^ { n } } { n ! } = \sum _ { m \ge 0 } p _ { \le n } ( m ) q ^ { m } .
$$

(iii) Distribution. For $t \geq 0 ,$

$$
\mathbb { P } ( \bar { \zeta } > t ) = \frac { 1 } { ( q ; q ) _ { \infty } } \sum _ { k \ge 0 } \frac { ( - 1 ) ^ { k } q ^ { k ( k + 1 ) / 2 } } { ( q ; q ) _ { k } } e ^ { - t / q ^ { k } } , \qquad f _ { \bar { \zeta } } ( t ) = \frac { 1 } { ( q ; q ) _ { \infty } } \sum _ { k \ge 0 } \frac { ( - 1 ) ^ { k } q ^ { k ( k - 1 ) / 2 } } { ( q ; q ) _ { k } } e ^ { - t / q ^ { k } } .
$$

(iv) Slow tail. For $t \geq 0 ,$

$$
\left| \mathbb { P } ( \zeta > t ) - \mathcal { P } ( q ) e ^ { - t } \right| \leq \frac { ( - q ; q ) _ { \infty } } { ( q ; q ) _ { \infty } } e ^ { - t / q } .
$$

(v) Small-time asymptotic. At fixed �, as � 0, with $L : = \log ( 1 / t )$

$$
\log \mathbb { P } ( \bar { \zeta } \leq t ) = - \frac { L ^ { 2 } } { 2 \ell } - \frac { L \log L } { \ell } + O ( L ) .
$$

(vi) Mellin transform. The function $s \mapsto \mathbb { E } \bar { \zeta } ^ { s }$ is entire, and for all $s \in \mathbb { C }$

$$
\mathbb { E } \bar { \zeta } ^ { s } = \frac { \Gamma ( 1 + s ) ( q ^ { 1 + s } ; q ) _ { \infty } } { ( q ; q ) _ { \infty } } ,
$$

where the right-hand side has removable singularities at $s = - 1 , - 2 , . . .$ . In particular, for $n \geq 1$

$$
\mathbb { E } \big [ \bar { \zeta } ^ { - n } \big ] = \frac { \ell } { ( n - 1 ) ! } \prod _ { m = 1 } ^ { n - 1 } ( \rho ^ { m } - 1 ) , \qquad \mathrm { s o } \qquad \mathbb { E } [ 1 / \bar { \zeta } ] = \ell .
$$

The leading term in (v) is of log-normal type, but the law is not log-normal: the � log � term is part of the statement. The asymptotic is at fixed � and is not claimed to be uniform in the separate small-idea limit $q \uparrow 1$ . Item (v) is a weaker form of a known result. For exponential summands, Borovkov and Ruzankin [7, Theorem 1.1] prove

$$
\log \mathbb { P } ( \bar { \zeta } \leq t ) = - \frac { L ^ { 2 } } { 2 \ell } - \frac { L \log L } { \ell } - \frac { ( \ell - 2 \log \ell - 2 ) L } { 2 \ell } + o ( L ) ,
$$

and our high-precision evaluations at $\begin{array} { r } { q = \frac { 1 } { 2 } } \end{array}$ are consistent with this (Section A).

Proof. (i) By independence, $\begin{array} { r } { \Sigma e ^ { - s \bar { \zeta } } = \prod _ { k } \mathbb { E } e ^ { - s q ^ { k } e _ { k } } = \prod _ { k } ( 1 + s q ^ { k } ) ^ { - 1 } } \end{array}$ for $s > - 1$

(ii) By (3.1) with $\begin{array} { r } { z = - s , \mathbb { E } e ^ { - s \bar { \zeta } } = \sum _ { n } ( - s ) ^ { n } / ( q ; q ) _ { n } \mathrm { ~ f o r ~ } | s | < 1 } \end{array}$ . Since $\bar { \zeta }$ has exponential moments of small order, this expansion equal $\begin{array} { r } { \sum _ { n } ( - s ) ^ { n } \mathbb { E } \bar { \zeta } ^ { n } / n ! } \end{array}$ . The partition statement is the expansion $\textstyle \prod _ { j = 1 } ^ { n } ( 1 - q ^ { j } ) ^ { - 1 } =$ $\textstyle \prod _ { j = 1 } ^ { n } \sum _ { a _ { j } \geq 0 } q ^ { j a _ { j } }$

(iii) Let $\begin{array} { r } { \bar { \zeta } _ { n } : = \sum _ { k < n } e _ { k } q ^ { k } } \end{array}$ . Its summands are independent exponentials with distinct rates $q ^ { - k }$ , so by the classical formula for the hypoexponential law,

$$
\mathbb { P } ( \bar { \zeta } _ { n } > t ) = \sum _ { k < n } c _ { k } ^ { ( n ) } e ^ { - t / q ^ { k } } , \qquad c _ { k } ^ { ( n ) } : = \prod _ { \substack { j < n } } \frac { 1 } { 1 - q ^ { j - k } } .
$$

The factors with $j < k$ equal $( 1 - q ^ { - m } ) ^ { - 1 } = - q ^ { m } / ( 1 - q ^ { m } )$ with $m = k - j \in \{ 1 , \ldots , k \}$ , and those with $j > k$ equal $( 1 - q ^ { m } ) ^ { - 1 }$ with $m = j - k \in \{ 1 , \ldots , n - 1 - k \}$ . Hence

$$
c _ { k } ^ { ( n ) } = \frac { ( - 1 ) ^ { k } q ^ { k ( k + 1 ) / 2 } } { ( q ; q ) _ { k } ( q ; q ) _ { n - 1 - k } } \xrightarrow [ n  \infty ] { } \frac { ( - 1 ) ^ { k } q ^ { k ( k + 1 ) / 2 } } { ( q ; q ) _ { k } ( q ; q ) _ { \infty } } ,
$$

and $| c _ { k } ^ { ( n ) } | \leq q ^ { k ( k + 1 ) / 2 } / ( q ; q ) _ { \infty } ^ { 2 }$ , which is summable. Since $\bar { \zeta } _ { n } \uparrow \bar { \zeta } , \mathbb { P } ( \bar { \zeta } _ { n } > t ) \uparrow \mathbb { P } ( \bar { \zeta } > t )$ , and dominated convergence gives the series. As a check, at $t = 0$ the series equals 1 by $( 3 . 2 )$ with $z = q .$ . Diferentiating term by term gives the density; this is justified because, uniformly in $t \geq 0 ,$ , the diferentiated series is dominated by the convergent series $\textstyle \sum _ { k } q ^ { k ( k - 1 ) / 2 } / ( q ; q ) _ { k }$

(iv) The $k = 0$ term of the series is $e ^ { - t } / ( q ; q ) _ { \infty }$ . For $k \geq 1$ we have $e ^ { - t / q ^ { k } } \leq e ^ { - t / q } ,$ , and by (3.2) with $z = - q$

$$
\sum _ { k \geq 1 } { \frac { q ^ { k ( k + 1 ) / 2 } } { ( q ; q ) _ { k } } } \leq ( - q ; q ) _ { \infty } .
$$

(Probabilistically, $\bar { \zeta } = e _ { 0 } + X$ with � independent of $e _ { 0 }$ and $\begin{array} { r } { \mathbb E e ^ { X } = \prod _ { k \geq 1 } ( 1 - q ^ { k } ) ^ { - 1 } = \mathcal P ( q ) ; } \end{array}$ : a slow takeof needs only a slowfirst idea.)

(v) Upper bound. For $s \ \geq \ 1$ let $m : = \lfloor \log s / \ell \rfloor$ and write log $s = \ell ( m + \theta )$ with $\theta ~ \in ~ [ 0 , 1 )$ . Since log $\left( 1 + x \right) \geq \log x$ and $s q ^ { k } \geq 1$ for $k \leq m$

$$
\log ( - s ; q ) _ { \infty } \geq \sum _ { k = 0 } ^ { m } ( \log s - k \ell ) = \ell ( m + 1 ) { \Big ( } { \frac { m } { 2 } } + \theta { \Big ) } \geq { \frac { \ell ( m + \theta ) ^ { 2 } } { 2 } } = { \frac { ( \log s ) ^ { 2 } } { 2 \ell } } ,
$$

because $( m + 1 ) ( m + 2 \theta ) - ( m + \theta ) ^ { 2 } = m + 2 \theta - \theta ^ { 2 } \geq 0$ . By Chernof’s bound and (i), log $\mathbb { P } \big ( \bar { \zeta } \ \leq \ t \big ) \ \leq$ $s t - ( \log s ) ^ { 2 } / ( 2 \ell )$ . Choose log $s = L + \log ( L / \ell )$ , which is admissible once $L \geq \ell ,$ so that $s t = L / \ell .$ Then

$$
\log \mathbb { P } ( \bar { \zeta } \leq t ) \leq \frac { L } { \ell } - \frac { ( L + \log ( L / \ell ) ) ^ { 2 } } { 2 \ell } \leq - \frac { L ^ { 2 } } { 2 \ell } - \frac { L \log L } { \ell } + \frac { ( 1 + \log \ell ) L } { \ell } .
$$

Lower bound. Let $n : = \lceil ( L + \log \frac { 4 } { 1 - q } ) / \ell \rceil$ . Let $E _ { 1 } : = \{ e _ { k } \ \leq \ x _ { k } \ \forall k \ < \ n \}$ with $x _ { k } : = t q ^ { - k } / ( 2 n )$ , and $\begin{array} { r } { E _ { 2 } : = \{ \sum _ { k \geq n } e _ { k } q ^ { k } \leq t / 2 \} } \end{array}$ . These events are independent and $E _ { 1 } \cap E _ { 2 } \subseteq \{ \bar { \zeta } \leq t \}$ . By Markov’s inequality, $\mathbb { P } ( E _ { 2 } ^ { c } ) \le 2 q ^ { n } / ( ( 1 - q ) t ) \le \frac { 1 } { 2 }$ . Since $\begin{array} { r } { 1 - e ^ { - x } \ge \frac { 1 } { \mathit { \Omega } _ { 2 } } } \end{array}$ <sup>min</sup>(<sup>�,</sup> <sup>1</sup>)<sup>,</sup>

$$
\log \mathbb { P } ( E _ { 1 } ) \ge - n \log 2 + \sum _ { k < n } \operatorname* { m i n } ( 0 , k \ell - M ) , \qquad M : = L + \log ( 2 n ) .
$$

The function $k \mapsto M - k \ell$ is decreasing, so the sum over $0 \leq k \leq M / t$ of $M - k \ell$ is at most $M + M ^ { 2 } / ( 2 \ell )$ Hence

$$
\log \mathbb { P } ( \bar { \zeta } \leq t ) \ \geq \ - \log 2 - n \log 2 - M - \frac { M ^ { 2 } } { 2 \ell } .
$$

Now $n = L / \ell + O ( 1 )$ and $\log ( 2 n ) = \log L + O ( 1 )$ , so $M ^ { 2 } = L ^ { 2 } + 2 L \log L + O ( L )$ , and the lower bound matches the upper bound up to $O ( L )$

(vi) For $\Re s > - 1$ , integrate the density in (iii) against $t ^ { s }$ term by term, using

$$
\int _ { 0 } ^ { \infty } t ^ { s } e ^ { - t / q ^ { k } } \mathrm { d } t = \Gamma ( 1 + s ) q ^ { k ( 1 + s ) } .
$$

The interchange is justified by absolute convergence, since the series $\textstyle \sum _ { k } q ^ { k ( k - 1 ) / 2 + k ( 1 + \mathfrak { R e } ) } / ( q ; q ) _ { k }$ <sub>�</sub> converges. Then

$$
\mathbb { E } \bar { \zeta } ^ { s } = \frac { \Gamma ( 1 + s ) } { ( q ; q ) _ { \infty } } \sum _ { k \ge 0 } \frac { ( - 1 ) ^ { k } q ^ { k ( k - 1 ) / 2 } ( q ^ { 1 + s } ) ^ { k } } { ( q ; q ) _ { k } } = \frac { \Gamma ( 1 + s ) ( q ^ { 1 + s } ; q ) _ { \infty } } { ( q ; q ) _ { \infty } }
$$

by (3.2). By (v) and $( \mathrm { i v } ) , \mathbb { E } \bar { \zeta } ^ { x } < \infty$ for every real �. So $s \mapsto \mathbb { E } e ^ { s \log \bar { \zeta } }$ is the moment generating function of log <sup>¯</sup>�, finite on all of ℝ, and therefore entire. On the right-hand side, the pole of $\Gamma ( 1 + s )$ at $s = - n$ is cancelled by the zero of the factor $1 - q ^ { s + n }$ in $( q ^ { 1 + s } ; q ) _ { \infty }$ . The identity theorem extends the formula to ℂ. Near $s = - n$ we have $\Gamma ( 1 + s ) \sim ( - 1 ) ^ { n - 1 } / ( ( n - 1 ) ! ( s + n ) )$ and $1 - q ^ { s + n } \sim \ell ( s + n )$ . The remaining factors give $\begin{array} { r } { \prod _ { m = 1 } ^ { n - 1 } ( 1 - q ^ { - m } ) \cdot ( q ; q ) _ { \infty } } \end{array}$ , and combining these yields the stated negative moments.

Corollary 3.5 $\mathbb { E } \bar { \zeta } = \frac { 1 } { 1 - q } , \operatorname { V a r } \bar { \zeta } = \frac { 1 } { 1 - q ^ { 2 } }$ , and $\operatorname { C V } ( \bar { \zeta } ) = \sqrt { \frac { 1 - q } { 1 + q } } . \operatorname { A s } q \uparrow 1$ (small ideas), $\mathrm { C V } \sim \sqrt { \ell / 2 } \to$ 0, consistent with Proposition 2.4. As �  0 (one giant idea), $\mathrm { C V } \stackrel { - } {  } 1$

Remark 3.6 The working paper’s computation is a mean Chan et al. assume that each doubling of � takes $2 ^ { - \varepsilon } \approx 0 . 7 6$ times as long as the previous one, starting from 4.5 months, and sum nine terms of the geometric series [12, SM]. For a Good process with doubling-sized ideas $( h = 1 )$ under convention $\mathrm { B } , q _ { 0 } = 1 / d _ { 0 }$ and the expected duration of the �-th doubling is exactly $d _ { 0 } q ^ { k }$ with $q = 2 ^ { - \varepsilon }$ . So their seventeen months is, up to rounding, $\mathbb { E } T _ { 9 } = 1 7 . 3$ months, and the full series $d _ { 0 } / ( 1 - 2 ^ { - \varepsilon } ) = 1 9 . 0 0$ months is $\mathbb { E } { \zeta } .$ Theorem 3.4 supplies the rest of the distribution (Table 2).

Ideas need not all be the same size. For general sizes the explosion time is a perpetuity $[ 3 6 ] ,$ and its moments remain explicit.

Theorem 3.7 Random idea sizes Let $\varepsilon > 0 ,$ let � be any i.i.d. strictly positive idea-size law, $Y : = ( 1 + H ) ^ { - \varepsilon } ,$ $y _ { j } : = \mathbb { E } Y ^ { j } \in ( 0 , 1 )$ , and $\bar { \zeta } : = \kappa A _ { 0 } ^ { \varepsilon } \zeta$ . Then

(a) $\bar { \zeta } \overset { d } { = } e + Y \bar { \zeta } ^ { \prime }$ with $e \sim \mathrm { E x p } ( 1 )$ , � and $\bar { \zeta } ^ { \prime } \overset { d } { = } \bar { \zeta }$ independent;

(b) $\begin{array} { r } { \mathbb { E } \bar { \zeta } ^ { m } = m ! \big / \prod _ { j = 1 } ^ { m } \bigl ( 1 - y _ { j } \bigr ) \mathrm { f o r } m \in \mathbb { N } ; } \end{array}$

(c) $\mathrm { C V } ( \bar { \zeta } ) ^ { 2 } = \frac { 2 \big ( 1 - y _ { 1 } \big ) } { 1 - y _ { 2 } } - 1 .$

Proof. (a) $\begin{array} { r } { \bar { \zeta } = e _ { 0 } + Y _ { 1 } \sum _ { n \geq 1 } e _ { n } \prod _ { 2 \leq i \leq n } Y _ { i } , } \end{array}$ , and the last sum is a copy of $\bar { \zeta }$ independent of $( e _ { 0 } , Y _ { 1 } )$ . (b) All moments are finite by Theorem 3.1(b). Expanding $( e + Y \bar { \zeta } ^ { \prime } ) ^ { m }$ and isolating the term $j = m$ gives

$$
( 1 - y _ { m } ) \mathbb { E } \bar { \zeta } ^ { m } = \sum _ { j = 0 } ^ { m - 1 } \frac { m ! } { j ! } y _ { j } \mathbb { E } \bar { \zeta } ^ { j } \qquad ( y _ { 0 } : = 1 ) .
$$

The claim follows by induction from the telescoping identity $\textstyle \sum _ { j = 0 } ^ { m - 1 } y _ { j } \prod _ { i = 1 } ^ { j } ( 1 - y _ { i } ) ^ { - 1 } = \prod _ { i = 1 } ^ { m - 1 } ( 1 - y _ { i } ) ^ { - 1 }$ ; to verify it, add the term $j = m$ to both sides and use $\left( 1 - y _ { m } \right) + y _ { m } = 1 . \left( \mathrm { c } \right)$ follows from (b) with $m = 1 , 2$

Corollary 3.8 Equal ideas are the most predictable Among all idea-size laws with the same standardised mean explosion time $\mathbb { E } \bar { \zeta } = 1 / ( 1 - y _ { 1 } )$ (equivalently, the same mean at a common current rate $\kappa A _ { 0 } ^ { \varepsilon } )$ , the coeficient of variation of $\bar { \boldsymbol { \zeta } }$ is minimised exactly when � is a.s. constant, i.e. in the lattice case.

Proof. At fixed $y _ { 1 } , \mathrm { C V } ^ { 2 }$ is increasing in $y _ { 2 } ,$ and $y _ { 2 } \geq y _ { 1 } ^ { 2 } ;$ , with equality if and only if � is a.s. constant.

The perpetuity is explicit for at least one natural family of sizes.

Corollary 3.9 Lomax ideas give a Gamma explosion time Let $\varepsilon > 0 , b > 0 ,$ and suppose $\mathbb { P } ( H > x ) =$ $\begin{array} { r l } { \models } & { { } ( 1 + x ) ^ { - b \varepsilon } } \end{array}$ for $x \geq 0 ;$ equivalently, $\log ( 1 + H )$ is exponential with rate ��, or $Y = ( 1 + H ) ^ { - \varepsilon } \sim \mathrm { B e t a } ( b , 1 )$

Then

$$
\bar { \zeta } \sim \mathrm { G a m m a } ( b + 1 , 1 ) , \qquad \mathbb { E } \bar { \zeta } = b + 1 , \qquad \mathrm { C V } ( \bar { \zeta } ) = \frac { 1 } { \sqrt { b + 1 } } .
$$

At the same standardised mean the lattice case has $\mathrm { C V } = 1 / { \sqrt { 2 b + 1 } }$ , in line with Corollary 3.8. The improvement size has finite mean if and only if $b \varepsilon > 1 _ { \scriptscriptstyle \mathscr { Z } }$ the corollary needs no such condition.

Proof. $\mathbb { P } ( Y \le y ) = \mathbb { P } \big ( 1 + H \ge y ^ { - 1 / \varepsilon } \big ) = y ^ { b } ,$ so � Beta �, 1 and $y _ { j } = b / ( b + j )$ . By Theorem 3.7(b),

$$
\mathbb { E } \bar { \zeta } ^ { m } = m ! \prod _ { j = 1 } ^ { m } \frac { b + j } { j } = \frac { \Gamma ( b + m + 1 ) } { \Gamma ( b + 1 ) } ,
$$

which are the moments of Gamma $( b + 1 , 1 )$ . The Gamma law is determined by its moments (its moment generating function is finite near 0), so $\bar { \zeta } \sim \mathrm { G a m m a } ( b + 1 , 1 )$ . As a direct check, $F ( s ) = ( 1 + s ) ^ { - ( b + 1 ) }$ satisfies the transform form of Theorem 3.7(a), $F ( s ) = ( 1 + s ) ^ { - 1 } \mathbb { E } F ( s Y )$ , because $\begin{array} { r } { \frac { \mathrm { d } } { \mathrm { d } y } \left[ y ^ { b } ( 1 + s y ) ^ { - b } \right] = } \end{array}$ $b y ^ { b - 1 } ( 1 + s y ) ^ { - b - 1 }$ gives $\begin{array} { r } { \int _ { 0 } ^ { 1 } b y ^ { b - 1 } ( 1 + s y ) ^ { - ( b + 1 ) } \mathrm { d } y = ( 1 + s ) ^ { - b } } \end{array}$ . For the lattice comparison, $\mathbb { E } \bar { \zeta } = b + 1$ forces $q = b / ( b + 1 )$ , and $( \bar { 1 } - q ) / ( 1 + q ) = 1 / ( 2 b + 1 )$ .

Remark 3.10A Ramanujan integral For $0 < u < 1$ , Tonelli’s theorem applied to $\begin{array} { r } { \bar { \zeta } ^ { - u } = \Gamma ( u ) ^ { - 1 } \int _ { 0 } ^ { \infty } s ^ { u - 1 } e ^ { - s \bar { \zeta } } \mathrm { d } s } \end{array}$ together with Theorem 3.4(i) and (vi) at $s = - u ,$ gives

$$
\int _ { 0 } ^ { \infty } \frac { t ^ { u - 1 } \mathrm { d } t } { ( - t ; q ) _ { \infty } } = \frac { \pi } { \sin \pi u } \frac { ( q ^ { 1 - u } ; q ) _ { \infty } } { ( q ; q ) _ { \infty } } ,
$$

an evaluation in the circle of Ramanujan’s �-beta integrals $\left[ 3 , 1 7 \right]$

## 4 Residual-time forecasting and parameter uncertainty

## 4.1 Self-similarity of the residual time

Theorem 4.1 Residual-time self-similarity In the lattice case with $\varepsilon > 0 ,$ , let � be any $( \mathcal { F } _ { t } )$ -stopping time. On $\{ \tau < \zeta \}$ , conditionally on $\mathcal { F } _ { \tau }$ , the random variable $q _ { N _ { \tau } } ( \zeta - \tau )$ has law $\Lambda _ { q } .$ . That ${ \mathrm { i } } s ,$ for every bounded measurable $\varphi ,$

$$
{ \mathbb E } \left[ \varphi \big ( q _ { N _ { \tau } } ( \zeta - \tau ) \big ) \mathbf { 1 } _ { \{ \tau < \zeta \} } \big | { \mathcal F } _ { \tau } \right] = \mathbf { 1 } _ { \{ \tau < \zeta \} } { \mathbb E } \varphi ( \bar { \zeta } ) ,
$$

where expressions multiplied by $\mathbf { 1 } _ { \{ \tau < \zeta \} }$ are read as zero outside that event.

Proof. $( N _ { t } )$ is the minimal pure birth chain with rates $\left( q _ { n } \right)$ . By the strong Markov property of minimal continuous-time Markov chains [27], on $\{ \tau < \zeta , \ : N _ { \tau } = n \}$ and conditionally on $\mathcal { F } _ { \tau }$ , the process $( N _ { \tau + s } - n ) _ { s \geq 0 }$ is a minimal pure birth chain with rates $( q _ { n + j } ) _ { j \geq 0 } = ( \rho ^ { n } q _ { j } ) _ { j \geq 0 }$ , and its explosion time is $\zeta - \tau .$ . Multiplying all rates by $\rho ^ { n }$ divides all holding times by $\rho ^ { n }$ . So $\zeta - \tau \stackrel { d } { = } \rho ^ { - n } \zeta = \bar { \zeta } / q _ { n }$ conditionally, as claimed.

What the theorem does and does not say. With model parameters fixed and known, the remaining time has a scale-invariant conditional distribution, so its relative predictive uncertainty does not vanish as the process approaches explosion. Absolute uncertainty, measured in calendar time, does shrink, in proportion to the current expected waiting time. The theorem does not rule out learning unknown parameters, which is a diferent source of uncertainty from the irreducible randomness of future holding times, and it is not a statement about intelligence-explosion models outside this class. Let $m _ { t } : = 1 / q _ { N _ { t } }$ be the current expected waiting time for the next idea.

Corollary 4.2 On $\{ t < \zeta \}$ , with parameters known:

(a) $\mathbb { E } \left[ \zeta - t ~ \vert ~ \mathcal { F } _ { t } \right] = m _ { t } / ( 1 - q ) ~ \mathrm { a n d } ~ \mathrm { C V } ( \zeta - t ~ \vert ~ \mathcal { F } _ { t } ) = \sqrt { ( 1 - q ) / ( 1 + q ) } .$ , whatever � and whatever has happened so far;

(a) forecasting with known parameters  
![](images/65fba7431316a81ec200772eb6f45a55f1b83ae0747cfbcbb0ad3a626abd1c03.jpg)

(b) explosion-centred mean intensity  
![](images/e31f9af3fabca73fea2b5cb3c09c9cf299502f3d5eb8d0b863b6b16533118691.jpg)  
log � �<sub>�</sub> in units of ℓ (mod 2)  
Figure 3 (a) Forecasting along one simulated path $( h = 1 , \varepsilon = 0 . 3 9 ,$ , convention B) with known parameters. The forecast date $t + \mathbb { E } [ \zeta - t \mid \mathcal { F } _ { t } ]$ jumps down at each idea and drifts up in between. The shaded region is the conditional 90% predictive band given <sub>�</sub> under the model; it is not a confidence band for an unknown deterministic date. (b) Mean intensity of ideas in coordinates centred on the realised explosion time, log $\left( \zeta - T _ { n } \right)$ , folded modulo ℓ (Proposition 4.5). The curves are the exact Fourier series with Γ-coeficients; the circles are Monte Carlo for $\ell = \log 1 0 .$ . For $\ell = 1 . 2$ the oscillation is about 0.3%

(b) every $\mathcal { F } _ { t }$ -measurable forecast $\hat { Z }$ satisfies the conditional mean-squared-error bound

$$
\mathbb { E } [ ( \zeta - \hat { Z } ) ^ { 2 } | \mathcal { F } _ { t } ] \ge \frac { 1 - q } { 1 + q } \big ( \mathbb { E } [ \zeta - t | \mathcal { F } _ { t } ] \big ) ^ { 2 } ,
$$

with equality at $\hat { Z } = \mathbb { E } [ \zeta \mid \mathcal { F } _ { t } ]$

(c) if $[ a , b ]$ is an $\mathcal { F } _ { t } .$ -measurable interval with $\mathbb { P } ( \zeta \in [ a , b ] \mid \mathcal { F } _ { t } ) \ge 1 - \delta ,$ , then

$$
b - a \geq ( 1 - \hat { \sigma } ) ( 1 - q ) \mathbb { E } [ \zeta - t \mid \mathcal { F } _ { t } ] .
$$

Proof. (a) follows from Theorem 4.1 and Corollary 3.5. (b) holds because $\mathbb { E } [ ( \zeta - \hat { Z } ) ^ { 2 } \mid \mathcal { F } _ { t } ] = \operatorname { V a r } ( \zeta \mid$ $\mathcal { F } _ { t } ) + ( \mathbb { E } [ \zeta ~ | ~ \mathcal { F } _ { t } ] - \hat { Z } ) ^ { 2 }$ . For $( \mathrm { c } ) ,$ conditionally on $\mathcal { F } _ { t }$ the variable $q _ { N _ { t } } ( \zeta - t )$ has law $\Lambda _ { q } ,$ whose density is bounded by 1 (it is the convolution of the $\mathrm { E x p } ( 1 )$ density with another law). An interval carrying mass $1 - \delta$ therefore has length at least $1 - \delta$ in these units, and undoing the scaling with (a) gives (c). The bound in (c) is valid but generally not sharp; it does not identify the shortest predictive interval.

Remark 4.3 Lead times in current units Let $z _ { p }$ be the �-quantile of $\Lambda _ { q } .$ . By Theorem 4.1, $\mathbb { P } ( \zeta - t \geq m _ { t } z _ { p } \ |$ $\mathcal { F } _ { t } ) = 1 - p$ on $\{ t < \zeta \}$ . A response that needs lead time Δ and must be in place with probability at least $1 - p$ therefore has to be launched while $m _ { t } \geq \Delta / z _ { p }$ . For doubling-sized ideas at $\varepsilon = 0 . 3 9 , m _ { i }$ is the current expected doubling time, $\mathbb { E } [ \zeta - t \ | \ \mathcal { F } _ { t } ] = 4 . 2 2 m _ { t }$ , and the 5% and 95% quantiles are $z _ { 0 . 0 5 } ~ = ~ 2 . 1 4 8$ and $z _ { 0 . 9 5 } = 7 . 0 9 8 . \mathrm { ~ A ~ }$ twelve-month response that must be in place with probability 0.95 has to start while $m _ { t } \ge 1 2 / z _ { 0 . 0 5 } = 5 . 5 9$ months. All of this is conditional on the model and its parameters being known.

Remark 4.4 Random sizes For non-lattice sizes, Theorem 4.1 holds at the idea times $\tau = T _ { n }$ directly from Definition $2 . 1 \colon x ( A ^ { ( n ) } ) ^ { \varepsilon } ( \zeta - T _ { n } )$ is independent of $\mathcal { F } _ { T _ { n } }$ with the law of $\bar { \zeta }$ in Theorem 3.7. We do not use the extension to arbitrary stopping times.

## 4.2 Log-periodic structure in explosion-centred time

Finite-time singularities with discrete scale invariance are known to produce log-periodic oscillations as the critical time is approached [33, 34]. In the lattice Good process the discrete scale invariance is exact: one more idea multiplies all future rates by $\rho .$ The amplitude of the resulting oscillation is given exactly by the Gamma function.

Proposition 4.5 Explosion-centred mean intensity In the lattice case with $\varepsilon \ > \ 0 ,$ the ideas arrive at $T _ { 1 } , T _ { 2 } , \dots$ . Let $\begin{array} { r } { \nu : = \sum _ { n \geq 1 } \hat { \partial } _ { \log } ( q _ { 0 } ( \zeta - T _ { n } ) ) } \end{array}$ be the point process of their log-residual times, and let $\phi$ be the density of log �. Then $\mathbb { E } \nu ( \mathrm { d } x ) = m ( x )$ d� with

$$
m ( x ) = \sum _ { n \geq 1 } \phi ( x + n \ell ) , \qquad | m ( x ) - p ( x ) | \leq { \frac { e ^ { x } } { 1 - q } } ,
$$

where $p$ is the ℓ-periodic function

$$
p ( x ) = \frac { 1 } { \ell } \bigg [ 1 + 2 \sum _ { k > 1 } \Re \left( \Gamma \left( 1 - \frac { 2 \pi i k } { \ell } \right) e ^ { 2 \pi i k x / \ell } \right) \bigg ] , \qquad \left| \Gamma \left( 1 - \frac { 2 \pi i k } { \ell } \right) \right| ^ { 2 } = \frac { 2 \pi ^ { 2 } k / \ell } { \sinh ( 2 \pi ^ { 2 } k / \ell ) } .
$$

In particular, as $x \ \to \ - \infty$ the relative amplitude of the fundamental harmonic of $\ell m ( x )$ tends to $2 | \bar { \Gamma ( 1 + 2 \pi i / \ell ) } | \sim \left( { 4 \pi / \sqrt { \ell } } \right) e ^ { - \pi ^ { 2 } / \ell }$ as $\ell \to 0$

Proof. From $( 2 . 1 ) , \zeta - T _ { n } = q ^ { n } \zeta ^ { ( n ) }$ with $\begin{array} { r } { \zeta ^ { ( n ) } : = \sum _ { j \ge 0 } e _ { n + j } q ^ { j } / q _ { 0 } \overset { d } { = } \zeta } \end{array}$ . Hence log $\begin{array} { r } { ( q _ { 0 } ( \zeta - T _ { n } ) ) = - n \ell + \log \bar { \zeta } ^ { ( n ) } } \end{array}$ where each $\bar { \zeta } ^ { ( n ) } : = q _ { 0 } \zeta ^ { ( n ) }$ has law $\Lambda _ { q } ,$ and linearity of expectation gives �. The density of $\bar { \zeta }$ is bounded by 1, so $\phi ( y ) = e ^ { y } f _ { \bar { \zeta } } ( e ^ { y } ) \leq e ^ { y }$ , and

$$
| m ( x ) - p ( x ) | = \sum _ { j \geq 0 } \phi ( x - j \ell ) \leq e ^ { x } \sum _ { j \geq 0 } q ^ { j } ,
$$

where $\begin{array} { r } { p ( x ) : = \sum _ { n \in \mathbb { Z } } \phi ( x + n \ell ) } \end{array}$ . The periodisation $p$ of the integrable function $\phi$ has Fourier coeficients

$$
{ \frac { 1 } { \ell } } \int _ { \mathbb { R } } \phi ( y ) e ^ { - 2 \pi i k y / \ell } { \mathrm { d } } y = { \frac { 1 } { \ell } } \mathbb { E } \left[ \tilde { \zeta } ^ { - 2 \pi i k / \ell } \right] = { \frac { 1 } { \ell } } \Gamma { \bigl ( } 1 - { \frac { 2 \pi i k } { \ell } } { \bigr ) } { \frac { ( q ^ { 1 - 2 \pi i k / \ell } ; q ) _ { \infty } } { ( q ; q ) _ { \infty } } } = { \frac { 1 } { \ell } } \Gamma { \bigl ( } 1 - { \frac { 2 \pi i k } { \ell } } { \bigr ) } ,
$$

by Theorem $3 . 4 ( \mathrm { v i } )$ and the identity $q ^ { - 2 \pi i k / \ell } = e ^ { 2 \pi i k } = 1$ . The classical formula $| \Gamma ( 1 + i y ) | ^ { 2 } = \pi y / \mathrm { s i n h } ( \pi y )$ shows that the coeficients are absolutely summable. So the Fourier series converges uniformly to $p ,$ which is continuous because the periodisation converges locally uniformly. Stirling’s formula gives the asymptotics.

This is an ensemble statement in coordinates centred on the realised explosion time. Since that time is not known to an online observer, the proposition is not itself a prospective detection procedure, and it says nothing about how visible the oscillation is along a single path; substituting an estimated explosion time for $\zeta$ would require a separate treatment of the estimation error. What it does identify is an exact discrete-scaleinvariant structure, together with the size of its harmonics. That size is nonzero but exponentially small in $1 / \ell \colon$ the fundamental amplitude $2 | \Gamma ( 1 + 2 \pi i / \ell ) |$ is about $3 . 4 \times 1 0 ^ { - 1 5 }$ for doubling-sized ideas at $\varepsilon = 0 . 3 9$ about $3 . 1 \times 1 0 ^ { - 3 } \mathrm { a t } \ell = 1 . 2$ , and 11% only when each idea multiplies the rate of progress tenfold $( \ell = \log 1 0 ) ;$ see Fig. 3(b).

## 4.3 Uncertain growth exponent: an explicit $1 / t$ tail

The working paper stresses that the returns parameter is uncertain; one estimate has a 90% credible interval for � of 0.727, 2.094 [12, note $6 ]$ . Uncertainty near the critical value has a sharp consequence.

Proposition 4.6 Exact Pareto tail under parameter uncertainty In the lattice case with fixed ℎ and fixed current idea rate $q _ { 0 } ,$ let � be random, independent of $( e _ { n } )$ , and suppose its law has a density � on a neighbourhood of 0 that is continuous at 0 with $\pi ( 0 ) > 0$ . Then $\mathbb { E } [ \zeta \mid \varepsilon > 0 ] = \infty$ and

$$
\operatorname* { l i m } _ { t \to \infty } t \mathbb { P } ( \zeta > t \mid \varepsilon > 0 ) = \frac { \pi ( 0 ) } { q _ { 0 } \log ( 1 + h ) \mathbb { P } ( \varepsilon > 0 ) } .
$$

Since $q _ { 0 } \log ( 1 + h ) = g$ is the local log drift (Box 1, B), the constant equals $\pi ( 0 ) / ( g \mathbb { P } ( \varepsilon > 0 ) )$ , which is also the limit of $t \mathbb { P } ( T ^ { * } > t \mid \varepsilon > 0 )$ for the fluid waiting time $T ^ { * } = 1 / ( \varepsilon g )$

![](images/04620398389a10ad1c71742b5035269fa91aa4ff2154a6a2dbcc5a9d821320cc.jpg)  
Figure 4 Survival functions conditional on explosion. Blue: the fluid waiting time $T ^ { * } = 1 / ( \varepsilon g )$ mixed over an illustrative log-normal prior on � that matches the interval 0.727, 2.094 of [12, note 6]; this is not a fitted predictive distribution for actual AI development. Dashed: the limit � � of Proposition 4.6. Orange: intrinsic randomness only, with $r = 1 . 3 8 6$ known and doubling-sized ideas (Theorem 3.4(iii)).

Proof. Write $a : = \log ( 1 + h ) , C : = ( q _ { 0 } a ) ^ { - 1 }$ , and $\zeta _ { \varepsilon }$ for the explosion time given $\varepsilon \ > \ 0 ,$ Infinite mean. $\begin{array} { r } { \mathbb { E } \zeta _ { \varepsilon } = 1 / ( q _ { 0 } ( 1 - e ^ { - a \varepsilon } ) ) \geq 1 / ( q _ { 0 } a \varepsilon ) , \mathrm { a n d } \int _ { \Omega } \pi ( \varepsilon ) \varepsilon ^ { - 1 } \mathrm { d } \varepsilon = \infty } \end{array}$

Large �. Chernof’s bound with $\begin{array} { r } { s = \frac { \top } { 2 } } \end{array}$ <sup>and</sup> <sup>the</sup> <sup>inequality</sup> − <sup>log</sup> $1 - y / 2 ) \leq y$ <sup>on</sup> [<sup>0,</sup> <sup>1</sup>] <sup>give</sup> $\mathbb { P } ( \bar { \zeta } >$ $x ) \leq e ^ { - x / 2 } \mathbb { E } e ^ { \hat { \zeta } / 2 } \leq \exp \{ - x / 2 + 1 / ( 1 - q ) \}$ . Since $1 / ( 1 - q ) \leq 1 + 1 / ( a \varepsilon )$ , every $\varepsilon \geq 4 C / t$ satisfies $\mathbb { P } ( \zeta _ { \varepsilon } > t ) \le e ^ { 1 - q _ { 0 } t / 4 }$ , uniformly. The contribution of $\{ \varepsilon \geq 4 C / t \} \mathrm { t o } t \mathbb { P } ( \zeta > t , \varepsilon > 0 )$ is therefore at most $t e ^ { 1 - q _ { 0 } t / 4 } \to 0 .$

Small �. For � large, $4 C / t$ lies in the neighbourhood where � is a density, and the substitution $\varepsilon = u / t$ gives

$$
t \mathbb { P } ( \zeta > t , 0 < \varepsilon < 4 C / t ) = \int _ { 0 } ^ { 4 C } \pi ( u / t ) \mathbb { P } ( \zeta _ { u / t } > t ) \mathrm { d } u .
$$

For fixed $u > 0 ,$ Corollary 3.5 gives $\mathbb { E } [ \zeta _ { u / t } / t ] = 1 / ( q _ { 0 } t ( 1 - e ^ { - a u / t } ) ) \to C / u$ and $\mathrm { V a r } ( \zeta _ { u / t } / t ) = 1 / ( q _ { 0 } ^ { 2 } t ^ { 2 } ( 1 -$ $e ^ { - 2 a u / t } ) ) \to 0 ,$ so $\mathbb { P } ( \zeta _ { u / t } > t ) \to \mathbf { 1 } _ { \{ u < C \} }$ for � $\neq C .$ . The integrand is bounded, since � is bounded near 0 by continuity, so dominated convergence gives the limit $\begin{array} { r } { \int _ { 0 } ^ { 4 C } \pi ( 0 ) { \mathbf { 1 } } _ { \{ u < C \} } \mathrm { d } u = \pi ( 0 ) C } \end{array}$ . Dividing by $\mathbb { P } ( \varepsilon > 0 )$ proves the claim. The fluid statement is $\mathbb { P } ( 0 < \varepsilon < 1 / ( \hat { g t } ) ) \sim \pi ( 0 ) / ( g t )$

For a log-normal prior on � matching the interval above, with $\beta = 1 . 0 1$ and convention $\mathbf { B } , \mathbb { P } ( r > 1 ) =$ $0 . 7 4 3 \cdot$ . Conditional on $r > 1$ , the median fluid waiting time is 17.3 months, $\mathbb { P } ( T ^ { * } > 5 \mathrm { y e a r s } ) = 0 . 1 5 0$ $\mathbb { P } ( T ^ { * } > 1 0 \mathrm { y e a r s } ) = 0 . 0 7 4$ , and the tail constant of Proposition 4.6 is $8 . 6 7$ months $\mathrm { ( F i g . 4 ) } .$ . These values validate the calculation under the selected prior; they do not validate the prior or the underlying interval. The intrinsic randomness of Theorem 3.4 has an exponential tail, so it is parameter uncertainty that makes the expected waiting time infinite, even though explosion is certain given $\varepsilon > 0$

## 5 Fixed-sample detection and the cube-root law

The first priority in [12] is visibility, and one item on it is detecting the onset of an explosion. We study a specific, well-defined binary experiment, chosen to be as favourable to the observer as possible.

The experiment. Both candidate processes start at a known origin. The monitor observes the first � complete waiting times $\left( \tau _ { 0 } , \dots , \tau _ { n - 1 } \right)$ , so it waits until the �-th idea; the elapsed sampling time $T _ { n }$ is itself random. The baseline rate �<sub>0</sub> and the alternative increment $\ell > 0$ are known design parameters. After the �-th idea the monitor makes one decision between

$$
\begin{array} { l } { { H _ { 0 } : \ q _ { k } \equiv q _ { 0 } } } \\ { { \ } } \\ { { H _ { 1 } : \ q _ { k } = q _ { 0 } e ^ { k \ell } } } \end{array}
$$

$$
( { \mathrm { s t e a d y } } { \mathrm { e x p o n e n t i a l ~ p r o g r e s s , ~ } } r = 1 ) , \quad
$$

accelerating progress from the origin .

Let $P _ { 0 } ^ { n } , P _ { 1 } ^ { n }$ be the two laws: independent Exp �<sub>0</sub> , respectively independent $\mathrm { E x p } ( q _ { 0 } e ^ { k \ell } )$ . Define the minimax error and the minimum fixed-sample size for this experiment,

$$
\begin{array} { r } { \operatorname { e r r } _ { n } ( \ell ) : = \underset { \psi } { \operatorname* { i n f } } \operatorname* { m a x } \big \{ P _ { 0 } ^ { n } ( \psi = 1 ) , P _ { 1 } ^ { n } ( \psi = 0 ) \big \} , \qquad n ^ { * } ( \alpha , \ell ) : = \operatorname* { m i n } \{ n : \operatorname { e r r } _ { n } ( \ell ) \leq \alpha \} , } \end{array}
$$

where the infimum runs over all (possibly randomised) tests based on $\left( \tau _ { 0 } , \dots , \tau _ { n - 1 } \right)$ , and write $z _ { \alpha } : = \Phi ^ { - 1 } ( 1 -$ � . This is not an anytime alarm and not the detection of an unknown change point: the alternative accelerates from the beginning of the record, and the error bounds below must not be reused for repeated testing after every idea, which needs its own false-alarm criterion.

Theorem 5.1 Cube-root law for thefixed-sample experiment

(i) For all $\ell > 0$ and $\begin{array} { r } { \alpha \in ( 0 , \frac { 1 } { 4 } ) \colon n ^ { * } ( \alpha , \ell ) \ \geq \ \big ( 6 \log \frac { 1 } { 4 \alpha } \big ) ^ { 1 / 3 } \ell ^ { - 2 / 3 } . } \end{array}$

(ii) For all $\ell > 0$ and $n \geq 1 ! \quad \mathrm { e r r } _ { n } ( \ell ) \leq \exp \bigg ( - \frac { \ell ^ { 2 } ( n - 1 ) ^ { 3 } } { 2 4 ( 1 + n \ell / 2 ) } \bigg )$ . Hence $n ^ { * } ( \alpha , \ell ) \le n$ for every � with $\begin{array} { r } { \ell ^ { 2 } ( n - 1 ) ^ { 3 } \geq 2 4 ( 1 + n \ell / 2 ) \log \frac { 1 } { \alpha } . } \end{array}$

(iii) For every $\alpha \in ( 0 , { \frac { 1 } { 2 } } ) ;$ lim ℓ<sup>2</sup>/<sup>3</sup> �∗ �, ℓ = 12�<sup>2 1</sup>/<sup>3</sup>. ℓ 0

The exponent $2 / 3$ has a one-line explanation: the log-rate drifts by ℓ per idea, so after � ideas the Kullback–Leibler divergence is $\begin{array} { r } { \asymp \ell ^ { 2 } \sum _ { k < n } k ^ { 2 } \asymp \ell ^ { 2 } n ^ { 3 } } \end{array}$ , and reliable discrimination needs this to be of order one.

L Lemma 5.2 For $x \geq 0 ,$ , tanh $x \ge x / ( 1 + x )$ , and consequently log cosh � $\geq x ^ { 2 } / ( 2 ( 1 + x ) )$ .

Proof. Let $g ( x ) : = ( 1 + x )$ tanh $x \mathrm { ~ - ~ } x .$ . Then $g ( 0 ) = 0$ and $g ^ { \prime } ( x ) = ( 1 + x ) \operatorname { s e c h } ^ { 2 } x - ( 1 - \operatorname { t a n h } x ) =$ $2 \big ( ( 1 + 2 x ) e ^ { 2 x } - 1 \big ) / ( e ^ { 2 x } + 1 ) ^ { 2 } \geq 0$ . Next let $f ( x ) : =$ log cosh $x - x ^ { 2 } / ( 2 ( 1 + x ) )$ . Then $f ( 0 ) = 0$ and $\begin{array} { r } { f ^ { \prime } ( x ) = \operatorname { t a n h } x - \frac { x ( x + 2 ) } { 2 ( 1 + x ) ^ { 2 } } \geq \frac { x } { 1 + x } - \frac { x ( x + 2 ) } { 2 ( 1 + x ) ^ { 2 } } = \frac { x ^ { 2 } } { 2 ( 1 + x ) ^ { 2 } } \geq 0 . } \end{array}$

Proof of Theorem 5.1. The log-likelihood ratio is

$$
\mathcal { L } _ { n } : = \log \frac { \mathrm { d } P _ { 1 } ^ { n } } { \mathrm { d } P _ { 0 } ^ { n } } = \sum _ { k < n } \big ( k \ell - ( e ^ { k \ell } - 1 ) q _ { 0 } \tau _ { k } \big ) .
$$

Under $P _ { 0 } , q _ { 0 } \tau _ { k } = E _ { k }$ , and under $P _ { 1 } , q _ { 0 } \tau _ { k } = e ^ { - k \ell } E _ { k }$ , with $( E _ { k } ) \mathrm { i . i . d . } \mathrm { E x p } ( 1 )$

(i) By the Bretagnolle–Huber inequality [10], [35, Lemma 2.6], every test satisfies $P _ { 0 } ^ { n } ( \psi = 1 ) + P _ { 1 } ^ { n } ( \psi =$ $\begin{array} { r } { 0 ) \ge \frac { 1 } { 2 } \exp ( - \mathrm { K L } ( P _ { 1 } ^ { n } | | P _ { 0 } ^ { n } ) ) } \end{array}$ . Moreover

$$
\mathrm { K L } ( P _ { 1 } ^ { n } \| P _ { 0 } ^ { n } ) = \sum _ { k < n } \left( k \ell - 1 + e ^ { - k \ell } \right) \le \frac { \ell ^ { 2 } } { 2 } \sum _ { k < n } k ^ { 2 } < \frac { \ell ^ { 2 } n ^ { 3 } } { 6 } .
$$

If both errors are at most �, then $\textstyle 2 \alpha \geq { \frac { 1 } { 2 } } e ^ { - \ell ^ { 2 } n ^ { 3 } / 6 } ;$ , which rearranges to the claim.

(ii) Take $\psi : = \mathbf { 1 } \{ \mathcal { L } _ { n } > 0 \}$ . By Markov’s inequality, $P _ { 0 } ^ { n } ( \mathcal { L } _ { n } > 0 ) \leq \mathbb { E } _ { 0 } e ^ { \mathcal { L } _ { n } / 2 }$ and $P _ { 1 } ^ { n } ( \mathcal { L } _ { n } \leq 0 ) \leq \mathbb { E } _ { 1 } e ^ { - \mathcal { L } _ { n } / 2 }$ Both equal the Bhattacharyya coeficient $\textstyle \prod _ { k < n } \int { \sqrt { p _ { 0 , k } p _ { 1 , k } } }$ . For exponentials with rates �<sub>0</sub> and $q _ { 0 } e ^ { k \ell }$ each factor is $2 e ^ { \boldsymbol { k } \ell / 2 } / ( 1 + e ^ { \boldsymbol { k } \ell } ) = \mathrm { s e c h } ( \boldsymbol { k } \ell / 2 )$ . By Lemma 5.2,

$$
- \sum _ { k < n } \log \operatorname { s e c h } \frac { k \ell } { 2 } \geq \sum _ { k < n } \frac { ( k \ell / 2 ) ^ { 2 } } { 2 ( 1 + k \ell / 2 ) } \geq \frac { \ell ^ { 2 } } { 8 ( 1 + n \ell / 2 ) } \sum _ { k < n } k ^ { 2 } \geq \frac { \ell ^ { 2 } ( n - 1 ) ^ { 3 } } { 2 4 ( 1 + n \ell / 2 ) } ,
$$

using $\textstyle \sum _ { k < n } k ^ { 2 } = ( n - 1 ) n ( 2 n - 1 ) / 6 \geq ( n - 1 ) ^ { 3 } / 3 .$

(iii) Step 1: reduction to the likelihood-ratio test. For $n \geq 2 , { \mathcal { L } } _ { n }$ has a continuous law under both hypotheses, as a nondegenerate linear combination of independent exponentials. By the Neyman–Pearson lemma [25], for every test � there is a threshold � with $P _ { 0 } ( \mathcal { L } _ { n } > t ) = P _ { 0 } ( \psi = 1 )$ and $P _ { 1 } ( \mathcal { L } _ { n } \leq t ) \leq P _ { 1 } ( \psi = 0 )$ . Hence $\operatorname { e r r } _ { n } = \operatorname { i n f } _ { t }$ max $\{ P _ { 0 } ( \mathcal { L } _ { n } > t ) , P _ { 1 } ( \mathcal { L } _ { n } \leq t ) \}$ . Also $\operatorname { e r r } _ { n }$ is nonincreasing in �, since a test may ignore data.

Step 2: Gaussian limit. Fix $c > 0$ , let $n = n ( \ell ) : = \lfloor c \ell ^ { - 2 / 3 } \rfloor$ and $\delta : = n \ell ,$ so $\delta  0$ and $\ell ^ { 2 } n ^ { 3 } \to c ^ { 3 }$ as $\ell \downarrow 0 .$ . For $x \geq 0 ,$ Taylor’s theorem gives $\begin{array} { r } { \frac { x ^ { 2 } } { \gamma } \leq e ^ { x } - 1 - x \leq \frac { x ^ { 2 } } { \gamma } e ^ { x } } \end{array}$ and $\begin{array} { r } { \frac { x ^ { 2 } } { \gamma } e ^ { - x } \leq x - 1 + e ^ { - x } \leq \frac { x ^ { 2 } } { \gamma } } \end{array}$ , and also $x \leq e ^ { x } - 1 \leq x e ^ { x }$ and $x e ^ { - x } \leq 1 - e ^ { - x } \leq x .$ . With $\begin{array} { r } { S _ { 2 } : = \sum _ { k < n } ( \stackrel { \sim } { k \ell } ) ^ { 2 }  c ^ { 3 } \stackrel { \sim } { / } 3 } \end{array}$ , these give the following. Under $\begin{array} { r } { P _ { 0 } , \mathbb { E } _ { 0 } \mathcal { L } _ { n } = - \sum _ { k < n } ( e ^ { k \ell } - 1 - k \ell ) \in [ - e ^ { \hat { o } } S _ { 2 } / 2 , - S _ { 2 } / 2 ] } \end{array}$ and $\begin{array} { r } { \operatorname { V a r } _ { 0 } \mathscr { L } _ { n } = \sum _ { k < n } ( e ^ { k \ell } - 1 ) ^ { 2 } \in [ S _ { 2 } , e ^ { 2 \hat { o } } S _ { 2 } ] } \end{array}$ . Under $\begin{array} { r } { P _ { 1 } , \mathbb { E } _ { 1 } \mathcal { L } _ { n } = \sum _ { k < n } ( k \ell - 1 + e ^ { - k \ell } ) \in [ e ^ { - \hat { o } } S _ { 2 } / 2 , S _ { 2 } / 2 ] } \end{array}$ and Var<sub>1</sub> $\begin{array} { r } { \mathcal { L } _ { n } = \sum _ { k < n } ( 1 - e ^ { - k \ell } ) ^ { 2 } \in [ e ^ { - 2 \hat { \sigma } } S _ { 2 } , S _ { 2 } ] } \end{array}$ . The Lyapunov ratio is at most $\begin{array} { r } { \mathbb { E } | E _ { 1 } - 1 | ^ { 3 } e ^ { 3 \hat { \sigma } } \sum _ { k < n } ( k \ell ) ^ { 3 } / ( e ^ { - 3 \hat { \sigma } } S _ { 2 } ^ { 3 / 2 } ) \le \mathbb { E } | E _ { 1 } - 1 | ^ { 3 } e ^ { 6 \hat { \sigma } } \hat { \sigma } S _ { 2 } ^ { - 1 / 2 } \to 0 } \end{array}$ . By Lyapunov’s central limit theorem, $\mathcal { L } _ { n } \Rightarrow N ( \mp c ^ { 3 } / 6 , c ^ { 3 } / 3 )$ under $P _ { 0 }$ and $P _ { 1 }$ respectively.

Step 3: conclusion. The limits are continuous, so by Pólya’s theorem the distribution functions converge uniformly, and therefore

$$
\operatorname { e r r } _ { n ( \ell ) } ( \ell ) \to \operatorname* { i n f } _ { t } \operatorname* { m a x } \left\{ \Phi \bigg ( \frac { t + \mu } { \sigma } \bigg ) , \Phi \bigg ( \frac { t - \mu } { \sigma } \bigg ) \right\} = \Phi \bigg ( - \frac { \mu } { \sigma } \bigg ) = \Phi \bigg ( - \frac { c ^ { 3 / 2 } } { \sqrt { 1 2 } } \bigg ) ,
$$

with $\mu = c ^ { 3 } / 6$ and $\sigma ^ { 2 } = c ^ { 3 } / 3 ;$ ; the infimum is attained at $t = 0$ by symmetry. Let $c ^ { * } : = ( 1 2 z _ { \alpha } ^ { 2 } ) ^ { 1 / 3 } . \mathrm { I f } c < c ^ { * } ;$ the limit exceeds $\alpha ,$ so er $\mathbf { \hat { \rho } } _ { n } ( \ell ) > \alpha$ for small ℓ, and by monotonicity $n ^ { * } > c \ell ^ { - 2 / 3 } - 1 . \mathrm { I f } c > c ^ { * }$ , the limit is below �, so $n ^ { * } \leq c \ell ^ { - 2 / 3 }$ for small ℓ. Since � was arbitrary, $\ell ^ { 2 / 3 } n ^ { * } \to c ^ { * }$

How much time remains at the decision? The next statement answers this with a ratio of expectations and with predictive quantiles.

Corollary 5.3 Remaining duration at the decision Under $H _ { 1 }$ , at the arrival time $T _ { n }$ of the �-th idea, $\zeta - T _ { n }$ is independent of $\mathcal { F } _ { T _ { n } }$ and distributed as $q ^ { n } \zeta$ , where $q = e ^ { - \ell }$ . Hence its quantiles are $q ^ { n }$ times the quantiles of $\zeta ,$ and

$$
W _ { n } : = \frac { \mathbb { E } _ { 1 } [ \zeta - T _ { n } ] } { \mathbb { E } _ { 1 } \zeta } = q ^ { n } .
$$

Let $W ^ { * } ( \alpha , \ell ) : = W _ { n ^ { * } ( \alpha , \ell ) }$ . Then $\begin{array} { r } { W ^ { * } \leq \exp \lbrace - ( 6 \log \frac { 1 } { 4 \alpha } ) ^ { 1 / 3 } \ell ^ { 1 / 3 } \rbrace } \end{array}$ for all $\ell > 0$ and $\alpha \in ( 0 , \frac { 1 } { 4 } )$ , and

$$
\log W ^ { * } ( \alpha , \ell ) = - ( 1 2 z _ { \alpha } ^ { 2 } \ell ) ^ { 1 / 3 } ( 1 + o ( 1 ) ) \qquad ( \ell \downarrow 0 ) .
$$

Proof. The first statement is Theorem 4.1 at $\tau = T _ { n } ,$ , or (2.1) directly. The bounds follow from $W ^ { * } = e ^ { - \ell n ^ { * } }$ and Theorem 5.1.

$W ^ { * }$ is the remaining expected duration normalised by the initial expected explosion time. It is not the expectation of the fraction remaining along a realised trajectory, $\mathbb { E } _ { 1 } \left[ { \cal ( \zeta - T _ { n } ) / \zeta } \right]$ , whose denominator is random and contains the remaining time. For doubling-sized ideas and $n = 9$ , a Monte Carlo estimate of the latter is 0.0988 0.00011, whereas $W _ { 9 } = 0 . 0 8 7 8$ (Section A).

Reading the threshold. Since $W ^ { * } \to 1 { \mathrm { a s } } \ell \to 0 ,$ , an explosion made of small ideas becomes distinguishable from steady growth, in this experiment, while most of the expected remaining duration is still ahead. The convergence is slow, at rate $\ell ^ { \bar { 1 / 3 } }$ , and for lumpy progress $W ^ { * }$ is small. Table 3 gives the calibrated values, including predictive quantiles of the remaining time. The data-processing inequality shows that any statistic of the same first-� record, for instance a coarsened version of it, cannot do better than the threshold of Theorem 5.1. This says nothing about a real monitor that has auxiliary information absent from the model.

Remark 5.4 No uniformfixed-sample test over unseparated alternatives If ℓ is unknown and only $\ell > 0$ is specified, no fixed-� test has a nontrivial uniform error bound. Indeed $\begin{array} { r } { \mathrm { \mathop { U L } } ( P _ { \ell } ^ { n } \| P _ { 0 } ^ { n } ) = \sum _ { k < n } ( k \ell - 1 + e ^ { - k \ell } ) \to 0 } \end{array}$ as $\ell \downarrow 0 ,$ so $P _ { \ell } ^ { n } \to P _ { 0 } ^ { n }$ in total variation by Pinsker’s inequality. If $P _ { 0 } ^ { n } ( \psi = 1 ) \leq \alpha$ , then $P _ { \ell } ^ { n } ( \psi = 1 )  P _ { 0 } ^ { n } ( \psi =$ $1 ) \leq \alpha$ , and the worst-case type-II error over $\ell > 0$ is at least $1 - \alpha$ . Meaningful formulations use a separated alternative $\ell \ge \ell _ { \mathrm { m i n } } > 0 ,$ , a local asymptotic power criterion, or a sequential design with an explicit criterion.

Remark 5.5 Unknown baseline: an invariant reduction Suppose ℓ is specified but $q _ { 0 }$ is unknown. The lower bound (i) still holds, since knowing �<sub>0</sub> can only help. The scale-invariant statistic $P _ { k } : = \tau _ { k } / \sum _ { j < n } \tau _ { j }$ removes $q _ { 0 }$ exactly: integrating out $\begin{array} { r } { S = \sum _ { j } \tau _ { j } ( \operatorname { J a c o b i a n } S ^ { n - 1 } ) } \end{array}$ shows that � is uniform on the simplex under $H _ { 0 }$ and

![](images/ca400df9f2a927c70eeaa33bc309d1ddef5e702f71f214e8daffdb697b71d55c.jpg)

(b) normalised expected remaining duration  
![](images/6aadcdaf99c8658abd4833100fe61a3867dcb8bb8d290a822c3c84b95b544821.jpg)  
log-rate increment $\ell = \varepsilon \log ( 1 + h )$  
Figure 5 The fixed-sample experiment of Section 5: known baseline rate, known increment $\ell ,$ one decision after <sub>1</sub>the first � waiting times. (a) The minimum fixed-sample size $n ^ { * }$ at $\alpha ~ = ~ 0 . 0 5$ , rescaled by $\ell ^ { 2 / 3 }$ . Circles are exact integer thresholds from the hypoexponential law of the likelihood ratio (Section $\mathrm { A } ) ;$ the horizontal line is the limit $( 1 2 z _ { \alpha } ^ { 2 } ) ^ { 1 / 3 } \approx 3 . 1 9$ ; the grey curves are the non-asymptotic bounds of Theorem 5.1. (b) The normalised expected remaining duration $W ^ { * } = q ^ { n ^ { * } }$ (remaining mean divided by initial mean). Vertical lines mark idea sizes ℎ at $\varepsilon = 0 . 3 9$

that, under rates $q _ { 0 } e ^ { k \ell } ,$

$$
\frac { f _ { \ell } ( P ) } { f _ { 0 } ( P ) } = \frac { \exp \{ \ell \sum _ { k < n } k \} } { \big ( \sum _ { k < n } e ^ { k \ell } P _ { k } \big ) ^ { n } } .
$$

The score at $\ell = 0$ is $\begin{array} { r } { U _ { n } = \sum _ { k < n } k - n \sum _ { k < n } k P _ { k } } \end{array}$ , whose null variance, by the Dirichlet covariance formula, is $\begin{array} { r } { { \frac { n } { n + 1 } } \sum _ { k < n } ( k - \bar { k } ) ^ { 2 } \sim n ^ { 3 } / 1 2 } \end{array}$ , against $\textstyle \sum _ { k } k ^ { 2 } \sim n ^ { 3 } / 3$ for known $q _ { 0 }$ . This suggests the constant $( 4 8 z _ { \alpha } ^ { 2 } ) ^ { 1 / 3 }$ about 1.59 times as many ideas. We have not proved this: it requires a composite minimax formulation, triangular-array limits under both hypotheses and a matching lower bound (Section 8).

## 6 Model extensions: delay, diversion and an oversight race

The three results of this section concern diferent models with diferent assumptions. They are not properties of one unified controlled stochastic system.

## 6.1 A deterministic delayed-feedback model

Training runs take months, and an idea improves the research workforce only once it has been deployed. Suppose deployment takes time $D ,$ so that efective labour depends on $A ( t - D )$ while the dificulty of finding new ideas depends on the current frontier � � :

$$
\dot { A } ( t ) = c A ( t ) ^ { 1 - \beta } \big ( k A ( t - D ) \big ) ^ { \lambda } , \qquad t > 0 , \qquad A = \varphi \mathrm { o n } [ - D , 0 ] .\tag{6.1}
$$

Theorem 6.1 Delayed feedback Let $\beta , \lambda , c , k , D > 0$ and let $\varphi : [ - D , 0 ]  ( 0 , \infty )$ be continuous.

(i) In the deterministic delayed model (6.1), there is a unique solution on $[ - D , \infty )$ ; it is finite and increasing on 0, , so there is no finite-time blow-up, for any �.

(ii) If $r = \lambda / \beta > 1$ , then in the deterministic delayed model

$$
\operatorname* { l i m } _ { t \to \infty } { \frac { \log \log A ( t ) } { t } } = { \frac { \log r } { D } } .
$$

(iii) With � = 0 (and $r > 1 )$ the solution blows up at $T ^ { * } = ( \varepsilon c k ^ { \lambda } A ( 0 ) ^ { \varepsilon } ) ^ { - 1 }$

(iv) Stochastic delayed model. Let ideas of size ℎ arrive at intensity $( c / h ) k ^ { \lambda } A _ { t - } ^ { - \beta } A _ { t - D } ^ { \lambda }$ , with $A _ { s } = A _ { 0 }$ for $s \leq 0 .$ Then the process does not explode: $N _ { t } < \infty$ for all $t , \mathsf { a } . s .$

Proof. Put $B : = A ^ { \beta }$ and $C : = \beta c k ^ { \lambda }$ . Then (6.1) becomes

$$
\begin{array} { r } { \dot { B } ( t ) = C B ( t - D ) ^ { r } . } \end{array}\tag{6.2}
$$

The factor $A ( t ) ^ { 1 - \beta }$ has cancelled: diminishing returns at the frontier exactly ofset the frontier’s own growth. (i) On $[ j D , ( j + 1 ) D ]$ the right-hand side of (6.2) is a known continuous function, so � is obtained by integration, step by step (the method of steps [20]). Conversely, any positive $C ^ { 1 }$ solution of (6.1) gives a solution of $( 6 . 2 )$ , which proves uniqueness. Since $\dot { B } > 0 , B$ is increasing on $[ 0 , \infty )$ , and $B ( t ) \geq B ( 0 ) +$ $C ( \operatorname* { m i n } \varphi ) ^ { \beta r } t \to \infty$

(ii) Upper bound. For $\begin{array} { r } { t \ge D , B ( t + D ) = B ( t ) + C \int _ { t - D } ^ { t } B ( \nu ) ^ { r } \mathrm { d } \nu \le B ( t ) + C D B ( t ) ^ { r } } \end{array}$ . Once $B \geq 1 _ { \cdot }$ , this gives log $B ( t + D ) \leq$ � log � � log 1 �� . So � := log $\mathbf { \bar { \boldsymbol { B } } } + \log ( 1 + C \boldsymbol { D } ) / ( r - 1 )$ satisfies $u ( t + D ) \leq r u ( t )$ and therefore lim sup $t ^ { - 1 }$ log log $B ( t ) \leq ( \log r ) / D$

Lower bound. Fix $\delta \in ( 0 , 1 )$ . For $t \geq D$

$$
B ( t + D ) \ge C \int _ { t + D - \hat { \sigma } D } ^ { t + D } B ( u - D ) ^ { r } \mathrm { d } u \ge C \hat { \sigma } D B ( t - \hat { \sigma } D ) ^ { r } .
$$

So � := log $B { + } \mathrm { l o g } ( C \delta D ) / ( r { - } 1 )$ satisfies $\nu \big ( s + \big ( 1 + \delta \big ) D \big ) \geq r \nu \big ( s \big )$ for $; \ge D$ . Since $B \to \infty$ , � is bounded below by some $\nu _ { \mathrm { m i n } } > 0$ on a window of length $( 1 + \delta ) D$ . Iterating gives lim inf $\dot { \boldsymbol { t } } ^ { - 1 }$ log log $B ( t ) \geq ( \log r ) / ( ( 1 + \delta ) D )$ and we let $\delta \downarrow 0$ . Finally log log $A = \log \log B - \log \beta$

(iii) This is (1.1) with � replaced by ��<sup>�</sup>.

(iv) On $( j D , ( j + 1 ) D ] , A _ { t - } \geq A _ { j D }$ and $A _ { t - D } \leq A _ { j D }$ . So the intensity is bounded by $Q _ { j } : = ( c / h ) k ^ { \lambda } A _ { i D } ^ { \lambda - \beta }$ which is $\mathcal { F } _ { j D }$ -measurable. By a thinning construction, conditionally on $\mathcal { F } _ { j D }$ the number of ideas in that window is dominated by a Poisson variable with mean $Q _ { j } D ,$ , which is finite a.s. Induction on � completes the proof.

Theorem 6.1 makes precise, for this delayed-feedback model, the working paper’s remark that some research steps are inherently sequential [12, SM]. In the deterministic delayed model any fixed $D > 0$ removes the singularity, and what remains is a tower-like growth law: the exponent of � is multiplied by � every �. With $D = 3$ months and $r = 1 . 3 8 6$ , log � doubles about every $6 . 3 7$ months $\left( \mathrm { F i g . 6 ( a ) } \right)$ . The proof uses the particular placement of the delayed labour term in $( 6 . 1 ) ;$ ; it is not a statement about systems with an additional instantaneous feedback channel. For the stochastic delayed model we prove non-explosion only; whether $t ^ { - 1 }$ log log $A _ { t } \to ( \log r ) / D$ almost surely is open (Section 8).

## 6.2 Progressive diversion

Suppose a share � of the automated workforce is diverted from capabilities research to other uses, for example safety, oversight or beneficial applications.

Proposition 6.2 Progressive diversion Let $\dot { A } = c A ^ { 1 - \beta } \big ( ( 1 - \theta ( A ) ) k A \big ) ^ { \lambda }$ , where the share retained for capabilities research is $1 - \theta ( a ) = ( a / A _ { 0 } ) ^ { - \gamma }$ for $a \geq A _ { 0 }$ , with $\gamma \geq 0$ and $r > 1$ . Then � explodes in finite time if and only if $\gamma < 1 - 1 / r . \mathrm { A t } \gamma = 1 - 1 / r$ growth is exponential, and for $\gamma > 1 - 1 / r$ it is polynomial. The same threshold holds for the Good process with rate $a ^ { \varepsilon - \gamma \lambda }$ , by Theorem 3.1.

Proof. The equation reads $\dot { A } = c k ^ { \lambda } A _ { 0 } ^ { \gamma \lambda } A ^ { 1 + \varepsilon _ { \gamma } }$ with $\varepsilon _ { \gamma } : = \lambda ( 1 - \gamma ) - \beta$ . It blows up if and only if $\varepsilon _ { \gamma } > 0 ,$ , i.e.   
$\gamma < 1 - \beta / \lambda . \operatorname { I f } \varepsilon _ { \gamma } = 0$ the growth is exponential, and if $\varepsilon _ { \gamma } < 0$ then $A ^ { \vert \varepsilon _ { \gamma } \vert }$ grows linearly.

At the working paper’s central values $1 - 1 / r = 0 . 2 7 9 \colon$ a diverted share of 47% at ten times today’s capability and 85% at a thousand times turns the singularity into exponential growth. A constant share $\theta < 1$ only delays the blow-up, since $T ^ { * }$ scales like $( 1 - \theta ) ^ { - \lambda }$ . Proposition 6.2 is a statement about the capability equation alone. Non-explosion is not a statement about whether oversight keeps pace, which depends on a separate equation (Example 6.4).

## 6.3 Fixed-allocation race between capabilities and oversight

Let � measure oversight capability (alignment, control, monitoring), produced by the diverted labour under its own idea production function. With a constant share $\theta \in ( 0 , 1 )$ :

$$
\dot { A } = c A ^ { 1 - \beta } \big ( ( 1 - \theta ) k A \big ) ^ { \lambda } , \qquad \dot { S } = c _ { S } S ^ { 1 - \beta _ { S } } \big ( \theta k A \big ) ^ { \lambda _ { S } } , \qquad A ( 0 ) = A _ { 0 } , \ S ( 0 ) = S _ { 0 } > 0 ,\tag{6.3}
$$

with $\varepsilon = \lambda - \beta > 0 , \beta _ { S } , \lambda _ { S } > 0 ,$ , and $\varepsilon _ { S } : = \lambda _ { S } - \beta _ { S }$ , the exponent with which oversight would explode if it were fed by its own output. Let $T ^ { * }$ be the blow-up time of $A .$

Theorem 6.3 Fixed-allocation asymptotic race regimes As $t \uparrow T ^ { * }$

(i) i $\therefore \lambda _ { S } < \varepsilon ,$ then $S ( t ) \to S ( T ^ { * } ) < \infty ;$

(ii) if $\lambda _ { S } \geq \varepsilon ,$ then $S ( t )  \infty ,$ and

$$
\frac { S ( t ) } { A ( t ) } \longrightarrow \left\{ \begin{array} { l l } { 0 , } & { \varepsilon _ { S } < \varepsilon , } \\ { \mathcal { R } ( \theta ) : = \Big ( \frac { c _ { S } ( \theta k ) ^ { \lambda _ { S } } } { c ( ( 1 - \theta ) k ) ^ { \lambda } } \Big ) ^ { 1 / \beta _ { S } } , } & { \varepsilon _ { S } = \varepsilon , } \\ { \infty , } & { \varepsilon _ { S } > \varepsilon . } \end{array} \right.
$$

Away from the critical equality $\varepsilon _ { S } = \varepsilon _ { \mathrm { ~ } }$ , changing a fixed allocation share $\theta \in ( 0 , 1 )$ changes constants and dates but not the asymptotic regime of the ratio $S / A .$ At the critical equality the limit $\mathcal { R } ( \theta )$ depends on $\theta .$

Proof. Write $c ^ { \prime } : = c ( ( 1 - \theta ) k ) ^ { \lambda }$ . Then $A ( t ) = ( \varepsilon c ^ { \prime } ( T ^ { * } - t ) ) ^ { - 1 / \varepsilon }$ , and ${ \textstyle \frac { \mathrm { d } } { \mathrm { d } t } } S ^ { \beta _ { S } } = \beta _ { S } c _ { S } ( \theta k A ) ^ { \lambda _ { S } }$ . Hence

$$
S ( t ) ^ { \beta _ { S } } = S _ { 0 } ^ { \beta _ { S } } + \beta _ { S } c _ { S } ( \theta k ) ^ { \lambda _ { S } } ( \varepsilon c ^ { \prime } ) ^ { - p } \int _ { 0 } ^ { t } ( T ^ { * } - u ) ^ { - p } \mathrm { d } u , \qquad p : = \lambda _ { S } / \varepsilon .
$$

If $p \ < \ 1$ the integral converges as $t \uparrow T ^ { * }$ , which is (i). If $p = 1$ it diverges logarithmically, so $S  \infty$ while $S / A \ \to \ 0 ;$ this case has $\varepsilon _ { S } ~ = ~ \varepsilon ~ - ~ \beta _ { S } ~ < ~ \varepsilon \quad$ . If $p ~ > ~ 1$ , then $S ^ { \beta _ { S } } = K ( T ^ { * } - t ) ^ { 1 - p } ( 1 + o ( 1 ) )$ with $K = \beta _ { S } c _ { S } ( \theta k ) ^ { \lambda _ { S } } ( \varepsilon c ^ { \prime } ) ^ { - p } / ( p - 1 )$ , and so

$$
\frac { S ( t ) } { A ( t ) } = K ^ { 1 / \beta _ { S } } ( \varepsilon c ^ { \prime } ) ^ { 1 / \varepsilon } ( T ^ { * } - t ) ^ { \frac { \varepsilon - \varepsilon _ { S } } { \varepsilon \beta _ { S } } } ( 1 + o ( 1 ) ) .
$$

The exponent is positive, zero or negative according as $\varepsilon _ { S } < \varepsilon , \varepsilon _ { S } = \varepsilon \mathrm { o r } \varepsilon _ { S } > \varepsilon ,$ . In the case of equality, $p - 1 = \beta _ { S } / \varepsilon$ , and a short computation gives $K ^ { 1 / \beta _ { S } } ( \varepsilon c ^ { \prime } ) ^ { 1 / \varepsilon } = ( c _ { S } ( \theta k ) ^ { \lambda _ { S } } / c ^ { \prime } ) ^ { 1 / \beta _ { S } } = \mathcal { R } ( \theta )$

Example 6.4 Removing the singularity does not make oversight keep pace Take $\lambda = 2 , \beta = 1 , c = k = c _ { S } = 1$ $A _ { 0 } = 1$ and $1 - \theta ( A ) = A ^ { - 1 / 2 }$ , which is exactly the threshold $\begin{array} { r } { \gamma = 1 - 1 / r = \frac { 1 } { \gamma } } \end{array}$ of Proposition 6.2. Then ${ \dot { A } } = A , s o A ( t ) = e ^ { t }$ and there is no blow-up. Let oversight follow the time-varying-allocation version of (6.3) with $\lambda _ { S } = 1$ and $\beta _ { S } = 2$ , namely ${ \dot { S } } = S ^ { - 1 } \theta ( A ) A .$ Then $\begin{array} { r } { \frac { \mathrm { d } } { \mathrm { d } t } S ^ { 2 } = 2 ( e ^ { t } - e ^ { t / 2 } ) } \end{array}$ , so

$$
S ( t ) ^ { 2 } = S _ { 0 } ^ { 2 } + 2 ( e ^ { t } - 1 ) - 4 ( e ^ { t / 2 } - 1 ) , \qquad { \frac { S ( t ) } { A ( t ) } } \sim \sqrt 2 e ^ { - t / 2 }  0 ,
$$

although the diverted share tends to one.

Reading the extensions. Three qualifications belong next to any policy reading of this section. (a) The asymptotic regime of $S / A$ is insensitive to a fixed allocation share away from the critical equality of exponents; at that equality the share sets the limiting ratio. (b) Non-explosion of �, as in Proposition 6.2, is not the same as oversight keeping pace (Example $6 . 4 ) ;$ the coupled problem with time-varying allocation is not solved here. (c) A policy can materially change finite-horizon performance, the time available for intervention and threshold crossings even when it does not change an asymptotic exponent. Treating $S / A$ as a measure of the adequacy of oversight is a modelling assumption, not a theorem about alignment. In particular, nothing here shows that fixed safety spending is useless or that diversion alone establishes safety. What the model suggests is that the exponents of oversight research, namely its returns to labour $\lambda _ { S }$ and its dificulty growth $\beta _ { S } ,$ deserve measurement alongside its budget.

(b) asymptotic regimes, fixed �

![](images/564fff14c1f8962e3a25293d82bf628377e4c0a72e012bd936efff729c463bd6.jpg)

![](images/46b009a2b6ee9caeefd5033aeddff4eb16cc5c8e6b2901b857aeea87e9b1212f.jpg)  
Figure 6 (a) Solutions of the deterministic delayed model (6.1) with $D = 1$ and 3 months $( \lambda = 1 . 4 0 , \beta = 1 . 0 1 , c k ^ { \lambda } = g ,$ $A = 1 \mathrm { o n } [ - D , 0 ] ) \mathrm { . }$ <sub>1</sub>: log � grows exponentially instead of blowing up. The stochastic delayed model is only proved not to explode. (b) Asymptotic regimes of the ratio $S / A$ in Theorem 6.3 at $\varepsilon = 0 . 3 9$ for a fixed allocation share �. On the boundary line the limit exists and depends on �.

Table 2 Exact laws of the explosion time (Theorem 3.4) at � = 0.39 under convention B. Times in months. The value $\mathbb { P } ( \zeta \leq 1 2 ) \approx 1 . 2 \times 1 0 ^ { - 1 3 }$ for ℎ = 0.01 is a high-precision evaluation, stable under precision doubling; it is far below Monte Carlo resolution.
<table><tr><td>idea size h</td><td>e</td><td>Eζ</td><td>CV</td><td>5% quantile</td><td>95% quantile</td><td> $\mathbb { P } ( \zeta \leq 1 2 )$ </td></tr><tr><td>0.01</td><td>0.0039</td><td>16.7</td><td>0.044</td><td>15.5</td><td>17.9</td><td> $1 . 2 \times 1 0 ^ { - 1 3 }$ </td></tr><tr><td>0.1</td><td>0.037</td><td>17.0</td><td>0.136</td><td>13.4</td><td>21.0</td><td>0.007</td></tr><tr><td>1</td><td>0.270</td><td>19.0</td><td>0.367</td><td>9.7</td><td>31.9</td><td>0.140</td></tr><tr><td>9</td><td>0.898</td><td>25.2</td><td>0.649</td><td>6.9</td><td>57.0</td><td>0.196</td></tr></table>

## 7 Numerical illustrations

We use $\lambda = 1 . 4 0 , \beta = 1 . 0 1 ( \varepsilon = 0 . 3 9 , r = \lambda / \beta$ unrounded) and convention B of Box 1 with $d _ { 0 } = 4 . 5$ months, so that $g = \log 2 / d _ { 0 }$ and $q _ { 0 } = g / \log ( 1 + h )$ for each idea size ℎ. The fluid blow-up time is $T ^ { * } = 1 / ( \varepsilon g ) = 1 6 . 6 5$ months. Under convention B the columns ��, the quantiles, $\mathbb { P } ( \zeta \leq 1 2 )$ and all durations in months depend on the convention; CV, $n ^ { * }$ and $W ^ { * }$ do not. These are illustrations of the theorems, not forecasts of AI development.

Reading the tables. The deterministic 16.65 months is the small-idea limit of a distribution, which is tight if progress consists of many small ideas and wide and right-skewed if it consists of a few large ones (Table 2). At equal local log drift, lumpy progress also puts more mass on early arrival, 14–20% within a year; unlike the CV, this comparison depends on the calibration convention. With parameters known, relative uncertainty about the remaining time does not shrink (Theorem 4.1), so any plan keyed to this model is naturally indexed to the current expected doubling time rather than to a calendar date. In the fixed-sample experiment (Table 3) the attainable lead time is governed by the granularity ℓ. For 1% ideas the decision comes after about 6.6 months in expectation, with a remaining duration whose 5–95% range is 9.36, 10.82 months. For doubling-sized ideas the decision comes after the working paper’s tenfold acceleration $( \mathbb { E } T _ { 9 } = 1 7 . 3$ months), with 0.85, 2.80 months remaining. These are predictive quantiles within the model, not guarantees about the world.

## 8 Limitations and open problems

The Good process is deliberately minimal (Remark 2.2). Ideas are conditionally independent and strictly positive; the exponent � is constant; compute and data, which the working paper lists as separate frictions, are absent; and the idea size ℎ is a free parameter without a direct empirical counterpart. The numerical illustrations depend on an explicit calibration convention and are not forecasts. Within these limits, the following problems seem natural.

Table 3 The fixed-sample experiment of Section 5 at � = 5% (known $q _ { 0 }$ and $\ell ) . n ^ { * }$ is the exact minimum fixed-sample size; the boundary pair e $\operatorname { r r } _ { n ^ { * } - 1 } > \alpha \geq \operatorname { e r r } _ { n ^ { * } }$ certifies it. $W ^ { * } = q ^ { n ^ { * } }$ is the remaining mean divided by the initial mean. The last three columns describe $\zeta - T _ { n ^ { * } }$ , which is distributed as $q ^ { n ^ { * } } \zeta$ (Corollary 5.3). Times in months.
<table><tr><td colspan="7"></td><td colspan="3">remaining time  $\zeta - T _ { n ^ { * } }$ </td></tr><tr><td>h</td><td> $n ^ { * }$ </td><td> $\mathrm { e r r } _ { n ^ { * } - 1 }$ </td><td> $\operatorname { e r r } _ { n ^ { * } }$ </td><td> $W ^ { * }$ </td><td> $\mathbb { E } T _ { n ^ { * } }$ </td><td>mean</td><td>5% quantile</td><td></td><td>95% quantile</td></tr><tr><td>0.01</td><td>130</td><td>0.0519</td><td>0.0499</td><td>0.604</td><td>6.6</td><td>10.07</td><td>9.36</td><td></td><td>10.82</td></tr><tr><td>0.1</td><td>30</td><td>0.0538</td><td>0.0451</td><td>0.328</td><td>11.4</td><td></td><td>5.56</td><td>4.40</td><td>6.88</td></tr><tr><td>1</td><td>9</td><td>0.0641</td><td>0.0332</td><td>0.088</td><td>17.3</td><td>1.67</td><td></td><td>0.85</td><td>2.80</td></tr><tr><td>9</td><td>5</td><td>0.0644</td><td>0.0136</td><td>0.011</td><td>24.9</td><td>0.28</td><td></td><td>0.08</td><td>0.64</td></tr></table>

1. Composite and sequential detection. With ℓ specified and $q _ { 0 }$ unknown, prove the cube-root law through the invariant reduction of Remark 5.5, presumably with the constant $( 4 8 z _ { \alpha } ^ { 2 } ) ^ { 1 / 3 }$ . With ℓ unknown, use a separated alternative $\ell \geq \ell _ { \mathrm { m i n } }$ or a local power criterion (Remark 5.4). Separately, determine optimal anytime alarms with explicit false-alarm control and treat an unknown onset (change point); neither is addressed by Theorem 5.1.

2. Multitype Good processes. Software and hardware improve each other [14]. For multitype pure-jump processes with rates $\textstyle { \boldsymbol { x } } _ { i } \prod _ { j } A _ { j } ^ { \varepsilon _ { i j } }$ , find the explosion criterion and determine whether the explosion time is governed by a matrix analogue of $( z ; q ) _ { \infty }$

3. Stochastic delays. For the stochastic delayed model of Theorem 6.1(iv), decide whether $t ^ { - 1 } \log \log A _ { t } \to$ log � � almost surely, and describe the fluctuations.

4. Coupled allocation under uncertainty. Choose a time-varying share $\theta _ { t }$ when � is unknown, with the coupled oversight equation of Example 6.4 in the loop. This is a control problem with learning and an absorbing catastrophe.

5. Exact laws beyond the lattice. Corollary 3.9 gives one explicit non-lattice law. Characterise broader classes of size laws for which the law of $\bar { \zeta }$ in Theorem 3.7 is explicit [11, 6].

6. Saturation. With physical limits, � eventually becomes negative $[ 1 2 ,$ note 5]. For logistic-type rate functions, describe the law of the time to reach a given fraction of the ceiling, and find what replaces the �-series.

Acknowledgements. We are grateful to Chan et al. [12] for a clear and carefully argued working paper; its explicit statement of the feedback model is what made the present analysis possible.

Use of AI tools. AI-based assistants supported parts of this work, including the drafting and editing of the text, the development and cross-checking of the numerical code, and an additional technical reading of the manuscript. The authors reviewed this material and take full responsibility for the content of the paper.

## A Numerical specification and reproducibility

Every number in this paper is produced by the scripts in the code/ folder of the accompanying repository [26],

$$
\mathsf { h t t p s : / / g i t h u b . c o m / x i a o y u l a c s / s i n g u l a r i t y - d i s t r i b u t i o n , }
$$

which also contains all generated outputs and figures; no number is typed by hand. make\_numbers.py computes all quantities and writes generated/numbers.tex, which the manuscript reads as LAT<sub>E</sub>X macros. run\_checks.py runs the checks listed below and writes generated/checks\_ report.txt; make\_figures.py regenerates Figs. 2 to $^ { 6 , }$ and make\_hero.py with make\_hero\_ labels.py regenerates Fig. 1. The script reproduce.sh (or reproduce.ps1) runs all of them in order. Environment: Python 3.13.14 with numpy 2.4.6, scipy 1.17.1, mpmath 1.3.0 and matplotlib 3.11.0 (PGF backend, so that figure text is set by LAT<sub>E</sub>X in the paper’s fonts); MiKTeX pdfT<sub>E</sub>X 1.40.29. Random seeds are fixed in each script.

Exact law. The alternating series of Theorem 3.4(iii) is evaluated in mpmath. The working precision is set per evaluation from a floating-point estimate of the largest term relative to $( q ; q ) _ { \infty }$ (the number of digits lost to cancellation) plus 40–60 guard digits. For example, this gives 172 digits for the 5% quantile at $h = 0 . 0 1$ $\left( q = 0 . 9 9 6 1 3 \right)$ and 90 digits for $h \geq 0 . 1$ . Truncation is certified: the term magnitudes decrease monotonically once $q ^ { k + 1 } < \frac { 1 } { 2 }$ , i.e. for $k > \log 2 / \ell ,$ , and from that point the remainder of the alternating series is bounded by the first omitted term, which is required to be below $1 0 ^ { - ( \mathrm { d i g i t s } - 5 ) }$ relative to the partial sum. Quantiles are found by bisection on the distribution function to relative tolerance $1 0 ^ { - 1 1 }$ . Lower-tail probabilities are computed as $1 - \mathbb { P } ( \bar { \zeta } > t )$ with additional guard digits, never subtracted at inadequate precision.

Checks run (results in generated/checks\_report.txt).

• Regression targets. All entries of Tables 2 and 3 agree with an independent recomputation to relative accuracy $2 \times 1 0 ^ { - 6 }$ , and the four detection boundary pairs agree to $1 0 ^ { - 9 }$

• Precision doubling. Each quantile and each $\mathbb { P } ( \zeta \leq 1 2 )$ is recomputed at twice the working precision; the largest change is below $1 0 ^ { - 4 0 }$ in absolute terms, and the relative change of $\mathbb { P } ( \zeta \leq 1 2 ) \approx 1 . 1 6 \times 1 0 ^ { - 1 3 }$ at $h = 0 . 0 1$ is below $1 0 ^ { - 8 0 }$ . In the near-one regime $q = 0 . 9 9 6 1 3 ,$ , the normalisation $\mathbb { P } ( \bar { \zeta } > 0 ) = 1$ holds to $1 0 ^ { - 3 9 }$ at 312 digits and to $1 0 ^ { - 3 5 1 }$ at 624 digits.

• Structure. On 400-point grids the distribution function is monotone and the density is nonnegative within $1 0 ^ { - 1 4 }$ ; for $h \in \{ 0 . 1 , 1 , 9 \}$ , moments 0–3 obtained by quadrature of the density agree with $n ! / ( q ; q ) _ { \ i }$ <sub>�</sub> to relative accuracy $1 0 ^ { - 1 5 }$

• Monte Carlo cross-checks with standard errors. For $h \in \{ 0 . 1 , 1 , 9 \} \left( 1 0 ^ { 5 } - 2 \times 1 0 ^ { 5 } \right.$ samples), the mean, the distribution function at the two quantiles and $\mathbb { P } ( \zeta \leq 1 2 )$ agree with the exact values within 4.5 standard errors (largest � about 2.3). At $h = 0 . 0 1 , \mathbb { P } ( \zeta \leq 1 2 )$ is not Monte Carlo-validated: it is far below simulation resolution. The two error probabilities at the exact balancing threshold of the likelihood-ratio test are simulated $( 4 \times 1 0 ^ { 5 }$ replicates per hypothesis) and agree with the exact minimax error within 4.5 standard errors. The pathwise fraction quoted after Corollary 5.3 uses $2 \times 1 0 ^ { 5 }$ paths truncated after 200 ideas (the omitted tail has relative size below $1 0 ^ { - 2 3 } )$ .

• Detection thresholds. For $k = 1 , \ldots , n - 1$ let $w _ { 0 k } = e ^ { k \ell } - 1 , w _ { 1 k } = 1 - e ^ { - k \ell }$ and $\begin{array} { r } { Z _ { j } = \sum _ { k } w _ { j k } E _ { k } } \end{array}$ ; the zero-weight term $k = 0$ is omitted. Under $\begin{array} { r } { H _ { j } , \mathcal { L } _ { n } = \ell \sum _ { k } k - Z _ { j } } \end{array}$ . With distinct weights, $\begin{array} { r } { \mathbb { P } ( Z _ { j } > x ) = \sum _ { k } c _ { j k } e ^ { - x / w _ { j k } } } \end{array}$ with $\begin{array} { r } { c _ { j k } = \prod _ { i \neq k } w _ { j k } / \big ( w _ { j k } - w _ { j i } \big ) } \end{array}$ , evaluated at 300 digits after checking $\textstyle { \big | } \sum _ { k } c _ { j k } - 1 { \big | } < 1 0 ^ { - 3 0 }$ . The balancing threshold solves $\mathbb { P } ( Z _ { 0 } > x ) + \mathbb { P } ( Z _ { 1 } > x ) = 1$ , and the common error is $\mathbb { P } ( Z _ { 1 } > x )$ . We report the boundary pair $( { \mathrm { e r r } } _ { n ^ { * } - 1 } , { \mathrm { e r r } } _ { n ^ { * } } ) ; { \mathrm { a t } } h = 0 . 0 1 , { \mathrm { e r r } } _ { n }$ lies only $8 \times 1 0 ^ { - 5 }$ below $\alpha ,$ so a Monte Carlo search could not certify this threshold. The non-asymptotic bounds of Theorem 5.1 bracket every computed $n ^ { * }$

• Proposition 4.6 and Corollary 3.9. For $\varepsilon \sim \mathrm { U n i f o r m } ( 0 , 1 ) , h = 1 , q _ { 0 } = 1$ , the quantity $t \mathbb { P } ( \zeta > t \mid \varepsilon > 0 )$ computed by Gauss–Legendre quadrature over � with the exact survival function, equals 1.489, 1.465, 1.454 and 1.448 at $t = 2 5 , 5 0 , 1 0 0 , 2 0 0 ,$ , approaching $1 / \log 2 = 1 . 4 4 2 7$ . For ${ \mathrm { B e t a } } ( b , 1 )$ marks, $b \in \{ 0 . 5 , 2 , 5 \}$ simulated perpetuities pass a Kolmogorov–Smirnov test against Gamma $\left( b + 1 , 1 \right) \left( 1 0 ^ { 5 } \right.$ samples each), and the transform identity in the proof of Corollary 3.9 holds to $1 0 ^ { - 1 0 }$

• Small-time asymptotic. check\_small\_time.py evaluates log $\mathbb { P } ( \bar { \zeta } \ \leq \ t ) \ \mathrm { a t } \ q \ = \ \frac { 1 } { 2 }$ from the exact series at 600–18,000 digits for $t = 1 0 ^ { - 8 } , 1 0 ^ { - 1 6 } , 1 0 ^ { - 3 2 } , 1 0 ^ { - 6 4 }$ . Relative to the three-term expansion of [7, Theorem 1.1] quoted after Theorem 3.4, the residual divided by $L { \mathrm { i s } } - 0 . 8 1 , - 0 . 5 2 , - 0 . 3 3$ and 0.20, consistent with an $o ( L )$ remainder; relative to the two terms of Theorem 3.4(v) it stays bounded ( 0.40 to 0.21), as stated.

• Gamma coeficient. At ℓ = log 10, quadrature of $= \bar { \zeta } ^ { - 2 \pi i / \ell }$ against the exact density agrees with $\Gamma ( 1 - 2 \pi i / \ell )$ to ten digits.

• Figure 1. make\_hero.py simulates 1 400 histories of a Good process with Lomax idea sizes, $\log ( 1 { + } H ) ~ \sim$ Exp �� with $b = 6 \mathrm { a n d } \varepsilon = 0 . 3 9$ , at local log drift log $2 / 4 . 5$ per month (so $q _ { 0 } = g b \varepsilon$ and <sup>¯</sup>� Gamma 7 by Corollary 3.9). Histories are rasterised as additive light with bilinear splatting, bloom and a filmic tone map, and coloured by the rank of their explosion time. The histories at the 10%, 50% and 90% quantiles of � are traced brightly in both panels. Left panel: capability in doublings against calendar time (0–36 months, 0–13 doublings); the dashed curve is the fluid ODE with $c A _ { 0 } ^ { \varepsilon } = g ;$ the horizon glow and silhouette are the exact density of $\zeta = \bar { \zeta } / q _ { 0 } ;$ the ticks are at $\mathbb { E } T _ { n } = { \bigl ( } 1 - y _ { 1 } ^ { n } { \bigr ) } / { \bigl ( } q _ { 0 } { \bigl ( } 1 - y _ { 1 } { \bigr ) } { \bigr ) }$ with $y _ { 1 } = b / ( b + 1 )$ , which converge to ��. Right panel: radius $\zeta - t \left( \mathrm { u p } \right)$ to 27 months) and angle 2� $\log _ { 2 } ( A _ { t } / A _ { 0 } ) / 5$ , so that between ideas a history moves radially inwards and at each idea it turns through an arc. By the self-similarity of Section 4 (at idea times, as in the remark on random sizes), the part of a spiral after any idea is, in law, a rotated and rescaled copy of a whole spiral. The stars, the halo and the reflection in the water are decorative. make\_hero\_labels.py sets the labels in the paper’s fonts.

• Delay equation. (6.2) is integrated on the logarithmic scale by the explicit Euler method; the figure uses step $1 0 ^ { - 3 }$ months. Halving the step from $2 \times 1 0 ^ { - 3 }$ to 10−<sup>3</sup> and to $5 \times 1 0 ^ { - 4 }$ changes log � � at $t = 1 0 , 2 0 , 3 0 ,$ , 40 by at most $2 . 8 \times 1 0 ^ { - 4 }$ and $1 . 4 \times 1 0 ^ { - 4 }$ (relative) for $D = 1 _ { : }$ , and by $2 . 2 \times 1 0 ^ { - 5 }$ and $1 . 1 \times 1 0 ^ { - 5 }$ for $D = 3 ;$ consistent with first-order convergence.

## References

1 P. Aghion, B. F. Jones, and C. I. Jones. Artificial intelligence and economic growth. In A. Agrawal, J. Gans, and A. Goldfarb, editors, The Economics ofArtificial Intelligence: An Agenda, pages 237–282. University of Chicago Press, 2019.

2 G. E. Andrews. The Theory of Partitions. Encyclopedia of Mathematics and its Applications, vol. 2. Addison-Wesley, 1976.

3 R. Askey. Ramanujan’s extensions of the gamma and beta functions. American Mathematical Monthly, 87(5):346–359, 1980.

4 F. Aurzada. A short note on small deviations of sequences of i.i.d. random variables with exponentially decreasing weights. Statistics & Probability Letters, 78(15):2300–2307, 2008.

5 J. Bertoin, P. Biane, and M. Yor. Poissonian exponential functionals, �-series, �-integrals, and the moment problem for log-normal distributions. In R. C. Dalang, M. Dozzi, and F. Russo, editors, Seminar on Stochastic Analysis, Random Fields and Applications IV, Progress in Probability 58, pages 45–56. Birkhäuser, 2004.

6 J. Bertoin and M. Yor. Exponential functionals of Lévy processes. Probability Surveys, 2:191–212, 2005.

7 A. A. Borovkov and P. S. Ruzankin. Small deviations of series of independent positive random variables with weights close to exponential. Matematicheskie Trudy, 11(1):49–67, 2008. English translation: Siberian Advances in Mathematics, 18(3):163–175, 2008.

8 D. Buraczewski, P. Dyszewski, A. Iksanov, and A. Marynych. On perpetuities with gamma-like tails.Journal ofApplied Probability, 55(2):368–389, 2018.

9 N. Bloom, C. I. Jones, J. Van Reenen, and M. Webb. Are ideas getting harder to find? American Economic Review, 110(4):1104–1144, 2020.

10 J. Bretagnolle and C. Huber. Estimation des densités: risque minimax. Zeitschrift für Wahrscheinlichkeitstheorie und verwandte Gebiete, 47:119–137, 1979.

11 P. Carmona, F. Petit, and M. Yor. On the distribution and asymptotic results for exponential functionals of Lévy processes. In M. Yor, editor, Exponential Functionals and Principal Values Related to Brownian Motion, pages 73–130. Biblioteca de la Revista Matemática Iberoamericana, 1997.

12 A. Chan, C. Winter, A. Barto, J. Pachocki, G. Hinton, E. Horvitz, Y. Bengio, D. Song, J. Clark, H. Greaves, A. Korinek, S. Hammond, T. Graepel, B. Bariach, P. H. S. Torr, S. A. McIlraith, J. Clune, S. Manning, G. Sastry, T. Davidson, D. Eth, and S. Mindermann. What if automating AI R&D triggers an intelligence explosion? Frontier AI Working Paper

Series No. 2/2026, Cambridge Programme on AI Science & Policy (CASP), University of Cambridge, September 2026.

13 Clay Mathematics Institute. Navier–Stokes announcement. 11 September 2026. https://www.claymath. org/news/navier-stokes-announcement/.

14 T. Davidson, B. Halperin, T. Houlden, and A. Korinek. When does automating AI research produce explosive growth? Feedback loops in innovation networks. NBER Working Paper No. 35155, 2026.

15 V. Dumas, F. Guillemin, and Ph. Robert. A Markovian analysis of additive-increase multiplicative-decrease algorithms. Advances in Applied Probability, 34(1):85–111, 2002.

16 W. Feller. An Introduction to Probability Theory and Its Applications, Vol. I, 3rd edition. Wiley, 1968.

17 G. Gasper and M. Rahman. Basic Hypergeometric Series, 2nd edition. Cambridge University Press, 2004.

18 I. J. Good. Speculations concerning the first ultraintelligent machine. In F. L. Alt and M. Rubinof, editors, Advances in Computers, vol. 6, pages 31–88. Academic Press, 1965 (volume dated 1966 in some indexes).

19 F. Guillemin, Ph. Robert, and B. Zwart. AIMD algorithms and exponential functionals. Annals of Applied Probability, 14(1):90–117, 2004.

20 J. K. Hale and S. M. Verduyn Lunel. Introduction to Functional Diferential Equations. Applied Mathematical Sciences 99. Springer, 1993.

21 A. Ho and P. Whitfill. The software intelligence explosion debate needs experiments. Epoch AI, Gradient Updates, November 2025.

22 C. I. Jones. R&D-based models of economic growth. Journal of Political Economy, 103(4):759–784, 1995.

23 M. Kremer. Population growth and technological change: One million B.C. to 1990. Quarterly Journal of Economics, 108(3):681–716, 1993.

24 J. Lamperti. Semi-stable Markov processes. I. Zeitschrift für Wahrscheinlichkeitstheorie und verwandte Gebiete, 22:205– 225, 1972.

25 E. L. Lehmann and J. P. Romano. Testing Statistical Hypotheses, 3rd edition. Springer, 2005.

26 X. Li, Z. Gao, C. Yang, and Z. Sha. Code and numerical outputs for “The Singularity Has a Distribution”. GitHub repository, 2026. https://github.com/xiaoyulics/singularity-distribution.

27 J. R. Norris. Markov Chains. Cambridge University Press, 1997.

28 OpenAI. On the Navier–Stokes Millennium Prize Problem. 8 September 2026. https://openai.com/index/ navier-stokes-solution/.

29 OpenAI. Research acceleration: The view inside OpenAI. 6 September 2026. https://openai.com/index/ research-acceleration-view-inside-openai/.

30 OpenAI. Sharing AI progress in mathematics. 6 October 2026. https://openai.com/index/ sharing-ai-progress-in-mathematics/.

31 OpenAI. openai/math: mathematical manuscripts and supporting proof artifacts. GitHub repository and README, initial release 6 October 2026, updated 8 October 2026. https://github.com/openai/math.

32 T. Ord. The dynamics of intelligence explosions. arXiv:2608.14426, 2026.

33 D. Sornette. Discrete-scale invariance and complex dimensions. Physics Reports, 297(5):239–270, 1998.

34 D. Sornette. Why Stock Markets Crash: Critical Events in Complex Financial Systems. Princeton University Press, 2003.

35 A. B. Tsybakov. Introduction to Nonparametric Estimation. Springer, 2009.

36 W. Vervaat. On a stochastic diference equation and a representation of non-negative infinitely divisible random variables. Advances in Applied Probability, 11(4):750–783, 1979.

37 H. von Foerster, P. M. Mora, and L. W. Amiot. Doomsday: Friday, 13 November, A.D. 2026. Science, 132(3436):1291– 1295, 1960.