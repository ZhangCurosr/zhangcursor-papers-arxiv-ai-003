# When Does Muon Actually Win? A Paired Data-, Time-, and Tuning-Budget Benchmark

Mushir Akhtar mushirakhtar.ml@gmail.com

Hazem N. Nounou

Majdi Mansouri m.mansouri@squ.edu.om

hnounou@hbku.edu.qa

Mohamed N. Nounou mnounou@hbku.edu.qa

## Abstract

Muon’s gains in data eficiency motivate evaluating its performance under explicit data, time, and tuning budgets. On a single A100 GPU, we compare a reference single-device Muon implementation with AdamW across 17.7M- and 33.6M-parameter causal language models and 2.70M-parameter vision transformers. A common 23-candidate development budget selects configurations evaluated with five matched seeds, full validation trajectories, optimizer-step timing, and a frozen one-shot test. Muon reduces validation-loss area under the curve by 3.25–22.44% at equal data and lowers test loss in all 20 primary pairs. In this configuration, Muon improves time-normalized loss area by 19.68–22.31% on vision, while AdamW is better by 3.38–6.96% on language. The full Muon optimizer call, including auxiliary AdamW updates, consumes 56.1–69.6% of training time, compared with 6.1– 9.1% for the AdamW baseline. Muon also has higher expected regret under restricted access to these candidate pools. These results characterize small models with reference kernels; their wall-clock ordering does not establish a ranking for larger models or optimized implementations.

## 1. Introduction

AdamW combines coordinate-wise adaptation with decoupled weight decay and mature implementations (Kingma and Ba, 2015; Loshchilov and Hutter, 2019). Muon instead forms a matrix momentum update and approximately orthogonalizes it through Newton–Schulz iteration (Jordan et al., 2024; Bernstein and Newhouse, 2024). Large language-model experiments report favorable scaling and lower training loss per unit of compute (Liu et al., 2025; Khona et al., 2026).

Eficiency can mean fewer training tokens, shorter runtime, or less tuning. These rankings can difer when optimizers have diferent update costs and tuning sensitivity. The original Muon implementation discussion distinguishes sample eficiency from wall-clock speed (Jordan et al., 2024); distributed work targets orthogonalization and communication costs (Chen et al., 2026; Gruntkowska et al., 2026).

We pair AdamW and Muon from matched initializations on two FineWeb-Edu language models and two vision transformers trained on balanced and 100:1 long-tailed CIFAR-100. A one-seed, 23-candidate development study selects configurations for five-seed full-horizon runs. We report equal-data and equal-time validation trajectories, GPU event timing, and a frozen one-shot test evaluation. This controlled benchmark characterizes small dense models and reference kernels on one accelerator. Large-scale and distributed wall-clock rankings lie outside its scope.

Three findings organize the paper. Muon wins every primary equal-data comparison and all 20 paired test-loss comparisons. Under the measured training clock, Muon leads on vision and AdamW leads on language; the reference Muon optimizer call accounts for over half of training time. With one randomly available candidate from each frozen pool, expected validation-AUC regret is 13.66% for Muon and 7.52% for AdamW.

Relation to optimizer benchmarks. Optimizer rankings depend on search spaces, workloads, and measurement rules (Choi et al., 2020; Schmidt et al., 2021). AlgoPerf addresses this dependence through fixed hardware, multiple workloads, time-to-result measurement, and explicit tuning protocols (Dahl et al., 2023). We apply these principles to AdamW and Muon, adding paired crossover curves, optimizer-time attribution, and postselection test evaluation.

## 2. A budget-aware comparison

## 2.1. Workloads and optimizer routing

The language workloads use decoder-only transformers with context length 256 on a deterministic 100M-token subset of FineWeb-Edu (Penedo et al., 2024). The micro model has 6 layers, width 256, and 17.7M parameters; the small model has 8 layers, width 384, and 33.6M parameters. Both train for 5,000 optimizer steps with an efective batch of 16 sequences. The vision workloads use a 6-layer, width-192 ViT with 4×4 patches and 2.70M parameters (Dosovitskiy et al., 2021). One uses the 45,000-example balanced CIFAR-100 training split; the other applies exponential class-frequency decay with a 100:1 imbalance ratio and retains 9,754 examples (Krizhevsky, 2009). Both run for 100 epochs. Each workload holds out 5,000 validation examples or 5M validation tokens during development.

For a matrix gradient $G _ { t }$ , Muon maintains momentum and applies five steps of the quintic Newton–Schulz map to a Frobenius-normalized matrix. This is a practical inexact linear minimization oracle; its degenerate case has recently been analyzed (Qian and Richt´arik, 2026). Writing $M _ { t }$ for the Nesterov momentum candidate and $M _ { t } = U \Sigma V ^ { \top }$ , the resulting direction approximates $U V ^ { \top }$

$$
X _ { k + 1 } = a X _ { k } + b ( X _ { k } X _ { k } ^ { \top } ) X _ { k } + c ( X _ { k } X _ { k } ^ { \top } ) ^ { 2 } X _ { k } , \quad ( a , b , c ) = ( 3 . 4 4 4 5 , - 4 . 7 7 5 0 , 2 . 0 3 1 5 ) .\tag{1}
$$

We apply Muon to hidden attention and feed-forward matrices, including the fused QKV projection. Embeddings, normalization parameters, biases, and output heads use AdamW, following the reference routing (Jordan et al., 2024). Muon updates approximately 26.7% and 42.2% of the parameters in language-micro and language-small; auxiliary AdamW updates the remaining 73.3% and 57.8%. The baseline applies AdamW to every parameter. Both methods use decoupled weight decay, gradient clipping at 1.0, 5% warmup, cosine decay, bfloat16 arithmetic, and the same deterministic data order within a seed. Appendix A records the selected hyperparameters and parameter counts.

## 2.2. Selection, estimands, and test isolation

Development uses one seed and 23 candidates per workload and strategy, accumulated through an initial grid, boundary extension, and five-point audit. Selection minimizes datanormalized validation-loss AUC, breaking ties by final loss and run identifier. AdamW and

Muon select interior learning rates on all workloads. Full-horizon runs use seeds 101–105; they estimate performance conditional on the selected configurations. Both data- and timebudget comparisons use these data-AUC-selected configurations. Selection stability across development seeds remains unmeasured.

For optimizer $^ { O , }$ let $L _ { o } ( x )$ denote interpolated validation loss against processed data or measured training seconds. For a paired common horizon H,

$$
\mathrm { A U C } _ { o } ( H ) = \frac { 1 } { H } \int _ { 0 } ^ { H } L _ { o } ( x ) d x , \qquad \mathrm { g a i n } = 1 - \frac { \mathrm { A U C } _ { M u o n } } { \mathrm { A U C } _ { A d a m W } } .\tag{2}
$$

We also report loss at equal data and equal time. Throughout the paper, gains are defined so that positive values favor the named alternative. A stable crossover is the first recorded point after which Muon remains at least 1% below AdamW through the common data horizon. Training time excludes validation and checkpoint $\mathrm { I / O ; }$ CUDA events measure optimizer-step time inside that interval. The seed is the replication unit. We use 10,000 paired bootstrap resamples for within-workload intervals. The pooled test-loss interval averages workloads equally and resamples seeds within workload, so its scope is the four fixed workloads rather than a population of tasks.

We froze training, selection, and all trajectory analyses before accessing the oficial test sets. We evaluated each selected checkpoint once under the frozen evaluation protocol. The final set contains 40 primary evaluations (two optimizers, four workloads, and five seeds) and 24 prespecified secondary evaluations. Test outcomes do not alter any hyperparameter, checkpoint, threshold, or analysis rule.

## 3. The winner depends on the budget

## 3.1. Data eficiency is consistent; time eficiency is not

Muon lowers validation-loss AUC at equal data on every workload (Figure 1). The gain is modest for language (3.25% and 4.46%) and larger for vision (16.98% and 22.44%). Every one of the 20 paired trajectories reaches a stable 1%-margin crossover. Mean crossover occurs at 4.8% and 3.0% of the language horizons. Both vision crossovers occur at the first 1% evaluation point, which gives an upper bound rather than a more precise crossing time.

Wall-clock accounting divides the workloads. Muon reduces time-normalized AUC by 19.68% on balanced vision and 22.31% on long-tailed vision. AdamW leads by 6.96% on language-micro and 3.38% on language-small. Equal-time endpoint loss has the same pattern: Muon is better by 39.71% and 41.09% on vision, while AdamW is better by 3.56% and 0.73% on language. The small-language endpoint interval narrowly includes zero (Appendix B).

## 3.2. Optimizer-step cost in the reference implementation

Table 1 reports the costs of the tested implementation. Muon’s throughput is 34–47% of AdamW’s with the reference kernels; optimizer steps consume 56.1–69.6% of training time, versus 6.1–9.1% for AdamW. The timed region includes matrix updates and auxiliary AdamW; separate Newton–Schulz timing and kernel-utilization measurements are unavailable. Peak allocated memory is 0.6–2.4% lower for Muon, but this metric alone does not identify the cause of the slowdown. Larger token batches and optimized kernels can change the relative cost of an optimizer step (Khaled et al., 2025).

![](images/15c048a383fd281ea3cfff05ca71440d3d85d158c27050dc99128d638f90bcb4.jpg)  
Figure 1: Mean validation-loss AUC reduction relative to AdamW over five matched seed pairs. Positive values favor Muon. Green bars compare equal data; hatched orange bars compare equal training time. With these reference kernels on one A100, the time-budget ranking reverses on both language models.

Table 1: Mean systems measurements over five seeds. The throughput ratio is Muon/AdamW. Optimizer shares cover the full optimizer call, including auxiliary AdamW in the reference Muon implementation.
<table><tr><td>Workload</td><td>Throughput ratio</td><td>AdamW opt. share</td><td>Muon opt. share</td></tr><tr><td>Language-micro</td><td>0.377</td><td>8.8%</td><td>63.1%</td></tr><tr><td>Language-small</td><td>0.468</td><td>9.1%</td><td>56.1%</td></tr><tr><td>Vision-balanced</td><td>0.339</td><td>7.4%</td><td>69.6%</td></tr><tr><td>Vision-long-tail</td><td>0.414</td><td>6.1%</td><td>63.1%</td></tr></table>

## 3.3. The data-budget advantage transfers to the test set

The frozen test evaluation confirms the validation result (Figure 2). Muon lowers test loss in every paired seed, with mean reductions of 4.77% and 7.01% on the language models and 11.52% and 26.03% on balanced and long-tailed vision. Language perplexity falls by 23.40% and 32.04%. Vision accuracy rises by 6.76 percentage points on the balanced split and 7.39 points on the long-tailed split; tail and worst-group accuracy on the long-tailed split each improve by 2.08 points. Across the four equally weighted workloads, mean relative testloss reduction is 12.33% (stratified bootstrap 95% interval [11.84, 12.83]%). This interval conditions on the chosen workloads and should not be read as uncertainty over the broader task distribution.

![](images/41c47547d4292434b3c1b0a0aaad4575bbf8a5d9c1ed4899f67c4ebdb7062697.jpg)  
Figure 2: Paired test-loss reduction for Muon relative to AdamW. Yellow circles denote matched seeds; orange diamonds and horizontal intervals show within-workload means and 95% bootstrap intervals. We froze all checkpoints and evaluation rules before test access.

## 4. What changes the conclusion?

Tuning allowance. We randomly permute each 23-candidate development set 10,000 times to emulate restricted access to the frozen pools. Expected relative AUC regret with one candidate is 7.52% for AdamW and 13.66% for Muon; with five it is 0.42% and 1.69%, and with ten, 0.09% and 0.67%. These estimates condition on the candidate ranges, pool construction, development seed, and uniform sampling rule (Figure 3). Equal candidate counts do not equalize search-space coverage or tuning time. Muon’s higher regret therefore describes these pools; adaptive step scaling ofers a separate approach to tuning sensitivity (Demidovich et al., 2026).

Variants and approximation. Switching from AdamW to Muon after 25% of training reduces test loss by 3.97–23.40% relative to AdamW, but full-horizon Muon achieves lower loss on all four workloads. An exploratory three-seed Variance-Muon control improves both language models and balanced vision, then increases long-tailed test loss by 11.71%. This control isolates post-orthogonalization second-moment adaptation; AdaMuon and NorMuon include additional mechanisms (Si et al., 2025; Li et al., 2025). In one-seed language-small audits, three rather than five Newton–Schulz steps save 21.87% of training time with a 0.74% AUC penalty. Restricted matrix routing saves 28–45% with penalties of 1.13–2.76%. Appendices D and E report these descriptive comparisons.

## 5. Discussion and limitations

A break-even interpretation. Let $q _ { o }$ denote measured throughput for optimizer o. At time T, its trajectory has processed approximately $q _ { o } T$ data units, so a time-budget com parison couples the loss curve to the implementation that produces it. Here $q _ { M u o n } / q _ { A d a m W }$ ranges from 0.339 to 0.468. The larger vision gains overcome that throughput gap; the smaller language gains do not. Faster orthogonalization raises $q _ { M u o n }$ and moves the break even point even if the loss trajectory per token is unchanged. Within the frozen candidate pools, restricting the tuning budget also increases Muon’s expected regret more than AdamW’s.

![](images/88ddf2c7c6532755ffbd769e5d2fd344a7f3b80a3d25eae0a3e24de205b56a90.jpg)

<table><tr><td>Candidates</td><td>AdamW</td><td>Muon</td><td>Variance-Muon</td></tr><tr><td>1</td><td>7.52%</td><td>13.66%</td><td>29.46%</td></tr><tr><td>5</td><td>0.42%</td><td>1.69%</td><td>7.70%</td></tr><tr><td>10</td><td>0.09%</td><td>0.67%</td><td>1.73%</td></tr><tr><td>23</td><td>0.00%</td><td>0.00%</td><td>0.00%</td></tr></table>

Figure 3: Expected relative validation-AUC regret as the candidate budget grows. Lower is better. The table gives prespecified checkpoints from the curves. Estimates average 10,000 random orderings within each workload–strategy pool, each containing 23 candidates. This audit measures restricted access to a frozen pool, not adaptive search.

Scale and implementation. The language models use widths 256/384 and 4,096 tokens per optimizer update on one A100. The relative cost of orthogonalization depends on matrix dimensions, token batch, and kernel implementation; our measurements do not isolate GPU underutilization or kernel-launch overhead. Auxiliary AdamW also covers most languagemodel parameters, which limits comparisons with architectures dominated by Muon-eligible matrices. Parameter count alone does not determine the resulting loss gain.

Large-scale studies report favorable Muon compute eficiency (Liu et al., 2025; Khona et al., 2026). In an 8B model with tensor parallelism, Khaled et al. (2025) report a much smaller throughput penalty and an 8% throughput increase for MuonBP over Muon. DMuon and EF21-Muon address distributed execution and communication costs (Chen et al., 2026; Gruntkowska et al., 2026). These systems operate in diferent regimes; the present benchmark supports no extrapolation of its wall-clock ordering to them.

Further limits include one-seed selection, short training horizons, crossovers resolved only at evaluation checkpoints, and fixed-pool tuning. The largest test efect, 26.03%, comes from one 100:1 CIFAR-100 construction and cannot establish a general long-tailed result.

Strength of evidence. The primary comparison uses five paired seeds. The three-seed variants and one-seed mechanism audits are descriptive, with less replication and narrower coverage. Broader evidence requires selection across multiple seeds, additional architectures and imbalance constructions, and optimized kernels on multiple accelerators.

Credible comparisons should pair initializations, equalize tuning opportunities, report data- and time-normalized curves, attribute optimizer cost, and freeze test evaluation. In the four tested workloads, Muon improves data-budget performance, while the equal-time ranking remains conditional on model size, parameter routing, token batch, and implementation.

## References

Jeremy Bernstein and Laker Newhouse. Old optimizer, new norm: An anthology. arXiv preprint arXiv:2409.20325, 2024. URL https://arxiv.org/abs/2409.20325.

Vincent Chen et al. DMuon: Eficient distributed muon training with near-Adam overhead. arXiv preprint arXiv:2606.27153, 2026. URL https://arxiv.org/abs/2606.27153.

Dami Choi et al. On empirical comparisons of optimizers for deep learning. arXiv preprint arXiv:1910.05446, 2020. URL https://arxiv.org/abs/1910.05446.

George E. Dahl et al. Benchmarking neural network training algorithms. arXiv preprint arXiv:2306.07179, 2023. URL https://arxiv.org/abs/2306.07179.

Yury Demidovich, Abhishek Chakraborty, Grigory Malinovsky, Angelia Nedi´c, and Peter Richt´arik. Distance-aware Muon: Adaptive step scaling for normalized optimization. arXiv preprint arXiv:2605.18999, 2026. URL https://arxiv.org/abs/2605.18999.

Alexey Dosovitskiy et al. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=YicbFdNTTy.

Kaja Gruntkowska, Alexander Gaponov, Zhirayr Tovmasyan, and Peter Richt´arik. Error feedback for Muon and friends. In International Conference on Learning Representations, 2026. URL https://iclr.cc/virtual/2026/poster/10007128.

Keller Jordan et al. Muon: An optimizer for hidden layers in neural networks. Technical note, 2024. URL https://kellerjordan.github.io/posts/muon/.

Ahmed Khaled, Kaan Ozkara, Tao Yu, Mingyi Hong, and Youngsuk Park. MuonBP: Faster Muon via block-periodic orthogonalization. arXiv preprint arXiv:2510.16981, 2025. URL https://arxiv.org/abs/2510.16981.

Mikail Khona et al. SOAP, Muon, and beyond: Pushing LLM pretraining scales. arXiv preprint arXiv:2607.20548, 2026. URL https://arxiv.org/abs/2607.20548.

Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. International Conference on Learning Representations, 2015. URL https://arxiv.org/abs/ 1412.6980.

Alex Krizhevsky. Learning multiple layers of features from tiny images. Technical report, University of Toronto, 2009.

Zichong Li, Liming Liu, Chen Liang, Weizhu Chen, and Tuo Zhao. NorMuon: Making muon more eficient and scalable. arXiv preprint arXiv:2510.05491, 2025. URL https: //arxiv.org/abs/2510.05491.

Jingyuan Liu et al. Muon is scalable for LLM training. arXiv preprint arXiv:2502.16982, 2025. URL https://arxiv.org/abs/2502.16982.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019. URL https://openreview.net/forum? id=Bkg6RiCqY7.

Guilherme Penedo et al. The FineWeb datasets: Decanting the web for the finest text data at scale. arXiv preprint arXiv:2406.17557, 2024. URL https://arxiv.org/abs/2406. 17557.

Xun Qian and Peter Richt´arik. Convergence analysis of Muon-type methods with inexact LMO in the degenerate case. arXiv preprint arXiv:2606.21581, 2026. URL https: //arxiv.org/abs/2606.21581.

Robin M. Schmidt, Frank Schneider, and Philipp Hennig. Descending through a crowded valley—benchmarking deep learning optimizers. In Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pages 9367–9376, 2021. URL https://proceedings.mlr.press/v139/ schmidt21a.html.

Chongjie Si, Debing Zhang, and Wei Shen. AdaMuon: Adaptive Muon optimizer. arXiv preprint arXiv:2507.11005, 2025. URL https://arxiv.org/abs/2507.11005.

## Appendix A. Experimental protocol

## A.1. Data construction

The language corpus is drawn from FineWeb-Edu, configuration sample-10BT, at frozen revision

## 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9.

Documents are assigned deterministically by a hash of the revision and stable source identifier. GPT-2 tokenization produces disjoint splits with 100M training tokens, 5M validation tokens, and 5M test tokens. The final language evaluator reads 50 deterministic batches for each checkpoint, using its frozen batch size and sequence length.

CIFAR-100 uses the original 50,000-example training split and 10,000-example test split. Fifty examples per class, selected with seed 2026, form a 5,000-example validation set. The balanced workload trains on the remaining 45,000 examples. The long-tailed workload applies exponential class-frequency decay to those training indices, from 450 examples in the largest class to 4 in the smallest, producing 9,754 examples and a 100:1 nominal imbalance ratio. The oficial test split is evaluated in full. Training augmentation uses random crops and horizontal flips; validation and test preprocessing are fixed.

## A.2. Models and horizons

Table 2 places the four architectures and training horizons on a common scale. The two language workloads vary depth, width, and parameter count; the two vision workloads hold the architecture fixed and change the training distribution.

Table 2: Workload definitions. The language batch is the efective number of sequences after accumulation.
<table><tr><td>Workload</td><td>Layers</td><td>Width</td><td>Heads</td><td>Parameters</td><td>Horizon</td></tr><tr><td>Language-micro</td><td>6</td><td>256</td><td>8</td><td>17.66M</td><td>5,000 steps</td></tr><tr><td>Language-small</td><td>8</td><td>384</td><td>8</td><td>33.57M</td><td>5,000 steps</td></tr><tr><td>Vision-balanced</td><td>6</td><td>192</td><td>3</td><td>2.70M</td><td>100 epochs</td></tr><tr><td>Vision-long-tail</td><td>6</td><td>192</td><td>3</td><td>2.70M</td><td>100 epochs</td></tr></table>

Both language models use context length 256, tied input/output embeddings, prenormalization blocks, GELU feed-forward layers of width 4d, and no dropout. Languagemicro uses batch 16 with no accumulation; language-small uses batch 8 and two accumulation steps. The ViT uses 4×4 patches, six pre-normalization blocks, width 192, three attention heads, MLP ratio 4, and no dropout. Vision batch size is 128. Evaluations occur every 50 language steps and every vision epoch.

## A.3. Selected hyperparameters

Table 3 records the configurations selected before the five-seed confirmation stage. Reporting the matrix and auxiliary learning rates separately makes Muon’s parameter routing explicit.

Table 3: Frozen primary configurations. AdamW learning rate and weight decay apply to all parameters in the baseline. For Muon, $\eta _ { M }$ and $\lambda _ { M }$ apply to eligible matrices; auxiliary parameters use AdamW.
<table><tr><td>Workload</td><td>Strategy</td><td> $\eta \mathrm { ~ o r ~ } \eta _ { M }$ </td><td>Weight decay</td><td> $\beta _ { 2 }$  or momentum</td><td>Aux. LR</td></tr><tr><td rowspan="2">Language-micro</td><td>AdamW</td><td>0.001</td><td>0.00</td><td>0.95</td><td></td></tr><tr><td>Muon</td><td>0.010</td><td>0.10</td><td>0.90</td><td>0.0003</td></tr><tr><td rowspan="2">Language-small</td><td>AdamW</td><td>0.0003</td><td>0.10</td><td>0.99</td><td></td></tr><tr><td>Muon</td><td>0.010</td><td>0.10</td><td>0.90</td><td>0.0003</td></tr><tr><td rowspan="2">Vision-balanced</td><td>AdamW</td><td>0.0003</td><td>0.45</td><td>0.999</td><td></td></tr><tr><td>Muon</td><td>0.010</td><td>0.05</td><td>0.85</td><td>0.0003</td></tr><tr><td rowspan="2">Vision-long-tail</td><td>AdamW</td><td>0.0003</td><td>0.45</td><td>0.90</td><td></td></tr><tr><td>Muon</td><td>0.010</td><td>0.45</td><td>0.95</td><td>0.0003</td></tr></table>

Muon auxiliary weight decay is 0.10 for language and 0.05 for vision; auxiliary AdamW uses $\beta _ { 2 } = 0 . 9 5$ . Muon has 4.72M eligible matrix parameters in language-micro, 14.16M in language-small, and 2.65M in each ViT. The corresponding auxiliary counts are 12.94M, 19.41M, and 46,180. Learning-rate schedules use 5% linear warmup followed by cosine decay to 0.1 of the initial rate in language and 0.01 in vision.

## A.4. Tuning protocol

The development horizon is 1,000 steps for language and 20 epochs for vision. Each of the 12 workload–strategy cells for AdamW, Muon, and the variance-adapted control contains 23 candidates. The first stage contributed 12 candidates per cell. A declared boundary extension added six, and a final audit added five low/high learning-rate and high-decay candidates. No candidate failed. The selection rule is lexicographic: normalized validationloss AUC against data, final validation loss, then run identifier. AdamW and Muon select an interior learning rate on all four workloads. The variance-adapted control selects the lower audited learning-rate boundary and the upper weight-decay boundary in all four workloads, which motivates caution in its interpretation.

## A.5. Timing and reproducibility

The training clock covers forward, backward, clipping, scheduling, and optimizer steps. It pauses during validation, serialization, and logging. CUDA events surround the optimizer call, and synchronization occurs before the accumulated event durations are read. For Muon, the call includes auxiliary AdamW, matrix momentum, Newton–Schulz iterations, scaling, and decay. The reported optimizer-time share measures their combined cost. Throughput divides cumulative tokens or examples by this training clock. Peak allocated GPU memory is the maximum value reported by $\mathrm { P y }$ Torch over the run.

All confirmation runs use deterministic algorithms and bfloat16 on an NVIDIA A100- SXM4-40GB. The software stack is Python 3.10.12, PyTorch 2.5.1+cu121, CUDA 12.1, and cuDNN 9.1. Checkpoint integrity is verified before each test evaluation. Test metrics are never used during training, selection, or trajectory analysis.

## Appendix B. Full primary results

Table 4 aligns validation and final-test efects under the frozen primary comparison. Table 5 gives the corresponding absolute test losses and paired intervals. The systems measurements in Table 6 identify the cost behind the time-budget reversal; Figure 4 makes the optimizertime disparity visible across workloads.

Table 4: Primary five-seed Muon gains relative to AdamW. Positive values favor Muon. Data/time columns are validation-loss AUC gains; equal-time is the endpoint loss gain at the common time horizon.
<table><tr><td>Workload</td><td>Data AUC</td><td>Time AUC</td><td>Equal-time</td><td>Test loss</td></tr><tr><td>Language-micro</td><td>3.25%</td><td>-6.96%</td><td>-3.56%</td><td>4.77%</td></tr><tr><td>Language-small</td><td>4.46%</td><td>-3.38%</td><td>-0.73%</td><td>7.01%</td></tr><tr><td>Vision-balanced</td><td>16.98%</td><td>19.68%</td><td>39.71%</td><td>11.52%</td></tr><tr><td>Vision-long-tail</td><td>22.44%</td><td>22.31%</td><td>41.09%</td><td>26.03%</td></tr></table>

The language-small equal-time endpoint gain is −0.73%, with a paired bootstrap 95% interval of $[ - 1 . 6 0 , 0 . 2 2 ] \%$ . Its time-AUC ordering therefore does not imply a resolved endpoint advantage.

Table 5: Frozen final-test endpoints, mean ± sample standard deviation over five paired seeds. Intervals cover the paired relative loss reduction within each workload.
<table><tr><td>Workload</td><td>AdamW loss</td><td>Muon loss</td><td>Relative reduction [95% CI]</td></tr><tr><td>Language-micro</td><td> $5 . 5 9 4 3 \pm 0 . 0 0 4 4$ </td><td> $5 . 3 2 7 7 \pm 0 . 0 0 5 7$ </td><td>4.77% [4.67, 4.86]</td></tr><tr><td>Language-small</td><td> $5 . 5 1 1 2 \pm 0 . 0 0 8 5$ </td><td> $5 . 1 2 5 0 \pm 0 . 0 0 5 2$ </td><td>7.01% [6.95, 7.06]</td></tr><tr><td>Vision-balanced</td><td> $2 . 8 1 3 1 \pm 0 . 0 6 7 8$ </td><td> $2 . 4 8 7 9 \pm 0 . 0 3 4 7$ </td><td>11.52% [9.71, 13.34]</td></tr><tr><td>Vision-long-tail</td><td> $5 . 7 7 1 5 \pm 0 . 0 3 3 0$ </td><td> $4 . 2 6 8 8 \pm 0 . 0 3 6 0$ </td><td>26.03% [25.24, 26.82]</td></tr></table>

Table 6: Absolute systems metrics, averaged across five seeds.
<table><tr><td rowspan="2">Workload</td><td colspan="2">Throughput</td><td rowspan="2">Optimizer-time share AdamW</td><td rowspan="2">Muon</td></tr><tr><td>AdamW</td><td>Muon</td></tr><tr><td>Language-micro</td><td>116,300</td><td>43,879</td><td>8.84%</td><td>63.07%</td></tr><tr><td>Language-small</td><td>62,884</td><td>29,456</td><td>9.09%</td><td>56.07%</td></tr><tr><td>Vision-balanced</td><td>4,215</td><td>1,430</td><td>7.35%</td><td>69.56%</td></tr><tr><td>Vision-long-tail</td><td>3,137</td><td>1,298</td><td>6.09%</td><td>63.10%</td></tr></table>

## Appendix C. Tuning-budget audit

For each workload–strategy cell, a random permutation of its 23 candidates represents an order in which configurations become available. The incumbent after budget b is the minimum AUC among the first b candidates. Relative regret divides its gap from the best of all 23 by the absolute best AUC. We average 10,000 permutations within a cell, then average the 12 cell curves by strategy. This estimates performance under uniform random access to the frozen pool. Pool construction includes extensions informed by earlier development outcomes, so random permutations summarize this completed pool rather than a prospective adaptive search. Diferent candidate ranges, sampling distributions, or development seeds can change the regret ordering.

![](images/3afe2fb8e785f59f8b4627119d94b4a211c52b024522dbb43688b114e3c77422.jpg)  
Figure 4: Optimizer-step time as a fraction of measured training time, averaged over five seeds. Green bars denote AdamW; hatched orange bars denote Muon. Validation and checkpoint I/O are excluded from the training clock.

Figure 3 in the main paper shows the complete budget curve and the four prespecified checkpoints. The common endpoint at 23 candidates verifies that regret is measured relative to the best configuration in the same frozen pool.

## Appendix D. Secondary strategies

The switch strategy runs AdamW on every parameter through 25% of the horizon, then applies Muon to eligible matrices while AdamW continues on auxiliary parameters. Development validation AUC selects the switch point from 25%, 50%, and 75%. The varianceadapted control applies an elementwise, bias-corrected second moment to the orthogonalized matrix update while preserving the same momentum, Newton–Schulz, scaling, decay, and auxiliary routing as the reference Muon implementation.

Figure 5 displays the paired uncertainty for both variants; Table 7 provides the corresponding means. These three-seed comparisons are exploratory and are not power-matched to the five-seed primary study. The long-tailed workload separates the strategies sharply and motivates treating the variance-adapted result as a controlled ablation rather than a uniformly stronger optimizer.

![](images/01549f1e0340566beebe5374e52eb0aa862967264e94b3bf6fbd6222b19c1bd5.jpg)  
Figure 5: Three-seed final-test loss reduction relative to AdamW. Positive values favor the alternative. Green bars denote the 25% AdamW→Muon switch; hatched orange bars denote Variance-Muon. Error bars are within-workload bootstrap intervals.

Table 7: Secondary final-test loss reduction relative to AdamW, three paired seeds.
<table><tr><td>Workload</td><td>25% AdamW→Muon</td><td>Variance-Muon</td></tr><tr><td>Language-micro</td><td>3.97%</td><td>5.16%</td></tr><tr><td>Language-small</td><td>5.92%</td><td>7.45%</td></tr><tr><td>Vision-balanced</td><td>8.65%</td><td>5.89%</td></tr><tr><td>Vision-long-tail</td><td>23.40%</td><td>-11.71%</td></tr></table>

## Appendix E. Descriptive mechanism audit

The mechanism study uses the language-small development workload and seed 11. Seven learning-rate candidates are evaluated for each Newton–Schulz depth; three candidates are evaluated for each eligible-role setting. All selections use the same normalized validation-AUC rule. The role labels are: all eligible matrices; all attention projections; feed-forward up/down projections; attention output projections; and fused QKV projections. Because the model stores Q, K, and V in one matrix, the audit cannot distinguish their individual efects.

Figure 6 reports the quality–time trade-of from restricting eligible matrices. Table 8 isolates a second approximation axis by comparing the selected three- and five-step Newton– Schulz configurations.

Table 8: Selected Newton–Schulz configurations in the mechanism audit.
<table><tr><td>NS steps</td><td>Matrix LR</td><td>Validation AUC</td><td>Training seconds</td><td>Orthogonality error</td></tr><tr><td>3</td><td>0.01</td><td>6.6162</td><td>100.59</td><td>0.851</td></tr><tr><td>5</td><td>0.01</td><td>6.5676</td><td>128.75</td><td>0.513</td></tr></table>

![](images/3d16b9857bdbee583cc0f11cdf0c5f52e32fde01038cbec1fd5027c3e96bdd9f.jpg)  
Figure 6: Validation-AUC penalty and training time saved when Muon is restricted to a subset of eligible matrices. Yellow bars show the AUC penalty; hatched orange bars show time saved relative to routing all eligible matrices through Muon. The audit uses one workload and seed and is descriptive.

## Appendix F. Statistical scope and reporting conventions

All primary contrasts are paired by workload and seed. Evaluation checkpoints within a trajectory are used for interpolation and integration, not as independent replicates. AUC uses trapezoidal integration after sorting and de-duplicating the budget axis; interpolation inserts a common-horizon endpoint when needed. The initial observed loss is carried to budget zero. Equal-budget horizons are the minimum attained by the paired optimizers.

Within-workload intervals resample the five paired seed efects with replacement 10,000 times using random seed 2026. The pooled final-test interval first resamples seeds separately inside each workload and then averages the four workload estimates with equal weight. It is conditional on this fixed workload set. Secondary intervals use three paired seeds. Crossover observations are interval-resolved at recorded validation checkpoints. No multiplicity correction is applied because the workload efects and resource definitions are reported as a structured profile rather than a family of interchangeable null-hypothesis tests.

All gains are direction-normalized: positive values indicate lower loss or AUC for the named alternative. Accuracy efects are percentage-point diferences. Time metrics use validation curves because test evaluation occurs only after training.