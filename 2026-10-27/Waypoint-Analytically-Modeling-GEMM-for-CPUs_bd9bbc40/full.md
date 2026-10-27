# Waypoint: Analytically Modeling GEMM for CPUs

Srinidhi N   
Advanced Micro Devices Inc   
India   
Srinidhi.N@amd.com   
Eashan Dash   
Advanced Micro Devices Inc   
India   
Eashan.Dash@amd.com

Sunanda Somwase Advanced Micro Devices Inc India SomwaseSunanda.Vishnu@amd.com

Varun Agrawal   
Advanced Micro Devices Inc India   
Varun.Agrawal@amd.com   
Hemant Kumar Chandrakant   
Chandavar   
Advanced Micro Devices Inc   
India   
hkcc375@gmail.com

Prakash Raghavendra Advanced Micro Devices Inc India Prakash.Raghavendra@amd.com

## Abstract

High-performance General Matrix Multiplication (GEMM) implementations rely on breaking down GEMM into carefully selected tiles to achieve peak performance. We present Waypoint, a new analytical model to automatically identify the optimal tile size for a given hardware and input matrix shape. Waypoint estimates the achievable performance for a given tile choice by capturing all the architectural efects including interactions with memory hierarchy, instructionlevel parallelism, and multi-threading. Using the estimates as a proxy, we search through the space of possible tile sizes to identify the one that maximizes performance. Addition ally, Waypoint helps hardware designers estimate realistic performance on experimental hardware optimizations. We build a new GEMM library libzenmm for CPUs that uses Waypoint to automatically break down GEMM and achieve performance that matches state-of-the-art GEMM libraries and helps accelerate LLM inference on CPUs achieving 1.36 − 1.98× speedups.

## 1 Introduction

Matrix Multiplication operations make up for a large part of the core computation in several modern machine learning based applications such as image processing, recommendation systems, and more recently, Large Language Model (LLM) inference. To ensure that these applications run eficiently, it is crucial to optimize the performance of General Matrix Multiplication (GEMM) operations on the underlying hardware. While the industry has moved towards developing specialized hardware accelerators such as GPUs, CPUs remain important for cost-sensitive use cases and small-scale inference workloads. Hardware vendors continue to enhance CPU architectures with new features aimed at improving GEMM performance.

On CPUs, improving GEMM performance involves careful coordination between the hardware and the software that implements GEMM operations. Hardware designers continue to innovate and introduce features that push the peak performance achievable. At the same time, software implementations optimize the use of these hardware features to achieve high performance.

A naive 3-loop implementation of the matrix multiplication algorithm is far from achieving peak performance. Instead, the libraries use specialized hardware features (e.g. vector/matrix instructions), and carefully organize the input matrices in memory so as to reduce the cost of accessing them for the computation. They do so by breaking down the GEMM operation into smaller units called tiles. For each tile, the libraries use vector/matrix instructions to compute the output, and between successive tiles, they ensure that memory accesses are eficient.

The tile sizes are critical to achieving high performance and need to be chosen carefully. Selecting suboptimal tile sizes can lead to ineficient memory accesses and underutilization of the hardware’s computational capabilities. The right tile depends both on the hardware and the specific input matrix size that the application invokes [31]. Therefore, an ideal solution would adapt the tile sizes not only to the hardware but also to specific input sizes.

Existing state-of-the-art GEMM libraries [16, 30] identify the tile sizes that are either optimal for a given hardware, or the specific class of shapes (e.g., tall and skinny [23], irregular [31]) but do not adapt them to the application’s needs. Many techniques are used to identify the tile sizes for a given hardware such as - manual expert-driven tuning resulting in handwritten assembly, automated empirical analysis [7, 10], and analytical modeling [24, 26, 31].

In this paper we propose a new analytical model to derive optimal tile sizes that are tailored not just for the hardware but also for the specific input matrix sizes that the application uses. Our model addresses another crucial challenge that the hardware designers face when innovating to accelerate performance on future hardware. They need to be able to estimate the realistic performance achievable on experi mental hardware designs. They routinely propose new ISA extensions (e.g. AMX [21], ACE [33], APX [20]) aimed at optimizing GEMM performance.

Existing analytical models address the challenge partially. They provide a Roofline [32] estimate of the maximum achievable performance. But depending on the matrix sizes, the realistic achievable performance is far from the Roofline.

Building a realistic performance model requires accounting for diferent architectural efects that are accompanied with a GEMM operation such as cache efects, pipelining, multi-threaded execution, and memory bandwidth limitations. The traditional Roofline model makes simplistic assumptions to over-approximate the peak performance achievable. It assumes that either the cost of computation dominates the cost of accessing data (i.e. compute-bound) or vice versa (i.e. memory-bound), which rarely holds true in practice. Therefore, a more nuanced model that captures both computation and memory access costs accurately is necessary to predict realistic GEMM performance.

In what follows, we describe our contribution - a new analytical model Waypoint that is parametric both in the hardware architecture, and the application’s needs (input matrix sizes). Waypoint provides accurate estimates of the realistic performance achievable, aiding hardware designers, and identifies the optimal tile sizes, enabling automatic and eficient GEMM implementations.

We build our model in two stages. First, we develop Waypoint-Strict, to provide realistic estimates ofGEMM performance on a given hardware independent of the software implementation. We prove theoretically that Waypoint-Strict is tighter than the Roofline model and show empirically that state-of-the-art GEMM libraries realize these estimates. In compute-bound scenario, Waypoint-Strict accounts for the cost of additional instructions that are always issued with the main compute instruction. Similarly for the memory-bound scenario, Waypoint-Strict uses more accurate bandwidth values to calculate the time it takes to fetch data. As a result, in both scenarios, Waypoint provides estimates that are tighter than the Roofline.

Second, we refine Waypoint-Strict for a tile based GEMM implementation resulting in Waypoint-Tile that estimates GEMM performance for a specific tile size. Using Waypoint-Tile’s estimates, we search through possible tile sizes to select the optimal one. Building Waypoint-Tile is challenging because it requires accurately modeling the interaction between the tile size and the memory hierarchy, including cache levels, cache line sizes, and memory bandwidth lim itations. Waypoint-Tile accurately estimates the number of cache misses that are imposed by a tile size and adds the associated miss costs. In building Waypoint-Tile, we identify several insights that allow us to rule out certain tile sizes and narrow down the search for an optimal one.

Empirically, we evaluate the accuracy of both Waypoint-Strict and Waypoint-Tile against state-of-the-art GEMM libraries. We find that Waypoint-Strict estimates are consistently closer to the observed performance (a geomean of 1.096×). Similarly, we compare the estimate of the best tile size identified by Waypoint-Tile and find that the model is accurate in predicting realistic performance (a geomean of 1.012×). These accuracy results allow us to confidently use the models for identifying the optimal tile size and estimating the performance on future hardware.

We develop a GEMM library libzenmm that uses Waypoint-Tile to select the best tile size for each input matrix size and hardware configuration. libzenmm performs comparable to state-of-the-art GEMM libraries across diferent hardware configurations and input sizes. Further on new hardware configurations, libzenmm is able to quickly adapt and achieve near-peak performance without any manual tuning. Finally, we use libzenmm to accelerate LLM inference on CPUs achieving 1.36 − 1.98× speedups on new hardware.

To summarize, our contributions are,

• An accurate implementation agnostic analytical model Waypoint-Strict, to estimate GEMM performance that is parametric in both the hardware architecture and the application’s needs. A more refined model, Waypoint-Tile that is specific to a tile-based GEMM implementation and used in practice to identify the optimal tile-size.

• A GEMM library libzenmm that uses Waypoint-Tile to automatically select optimal tiling parameters for a given GEMM shape and hardware configuration, achieving near-peak performance without extensive manual or automated tuning.

• Using libzenmm, we accelerate LLM inference on current and future CPUs.

## 2 Analytical Model

The Roofline model [32] distinguishes a computation between a compute-bound scenario, where performance is limited by the processing capacity of the CPU, and a memorybound scenario, where performance is limited by the memory bandwidth. It provides upper bounds in either case based on the peak computational throughput and memory bandwidth of the hardware.

Building on the Roofline model specifically for GEMM computations, we first develop an implementation-agnostic Waypoint-Strict that refines the cost in both computebound and memory-bound scenarios. Then, we further refine the compute cost assuming a tiled GEMM implementation to obtain a Waypoint-Tile that captures accurately the number of cache misses and the associated costs. We prove theoretically that the Waypoint-Strict is tighter than the Roofline model and empirically show (in Section 4) that the Waypoint-Tile is accurate.

## 2.1 Waypoint-Strict

The Waypoint-Strict is parametric in the hardware configuration, including cache sizes, memory bandwidth, and compute instructions and provides an estimate ofthe number of cycles a GEMM operation will take on the given hardware.

Formally a hardware configuration � consists of the following constants:

• The instruction reciprocal throughput for the core compute instruction $\tau _ { f m a }$ and the dimensions of the instruction $\delta _ { f m a }$ . The instruction is defined for each data-type and instruction set architecture (ISA) of the CPU. For example, in an x86-ISA based CPU using AVX512 family of SIMD instructions, the instruction reciprocal throughput for a single precision FMA instruction in AMD ‘Zen $5 '$ architecture is 0.5 cycles/instruction and contains $1 6 \times 1$ elements.

• Register file information such as the number of registers available to issue FMA instructions $N _ { r e g }$ and width of the registers in bits $W _ { r e g }$

• Cache sizes $( S _ { L 1 } , S _ { L 2 } , S _ { L 3 } )$ and access reciprocal throughput $( T _ { L 1 } , T _ { L 2 } , T _ { L 3 } )$ for the diferent levels of the memory hierarchy. For example, on AMD ‘Zen $5 '$ cores, the L1 data cache is 48 KB with an access reciprocal throughput of 0.5 cycles.

• Memory bandwidth information, including the peak bandwidth per memory channel $D _ { b w }$ and the number of channels $D _ { n }$

• Maximum number of cores $C _ { m a x }$ and the frequency $f$ of the CPU. The frequency is used to convert the number of cycles to time.

We define a GEMM operation �� as one that computes an output matrix $C = a ( A \cdot B ) + b C$ of size $M \cdot N$ where the input matrices are of size � · � and � · � using � threads on the data-type � of width $W _ { d }$ bits, and scalars � and �. We assume for simplicity that $a = 1$ and $b = 0$

Given the hardware and GEMM operation, the Waypoint-Strict computes the latency $L _ { o p }$ of the GEMM operation as follows:

$$
L ( o p ) = \operatorname* { m a x } \left( \frac { L _ { c } ( o p ) } { o p . T } , L _ { m } ( o p ) \right) + L _ { m t } ( o p )
$$

where $L _ { c }$ is the compute latency, $L _ { m }$ is the memory latency, and $L _ { m t }$ is the threading overhead for the GEMM operation $\boldsymbol { o p } .$ . In what follows, we describe each of the components in detail.

2.1.1 Compute latency. The Waypoint-Strict assumes a deterministic compute instruction to be used for a given data-type and hardware architecture e.g. AVX512 VNNI is used for Int8 GEMM on x86-ISA based CPUs. The exact mapping can be provided as input. We first compute the number of compute instructions $N _ { f m a }$ required to perform the GEMM:

$$
N _ { f m a } = M \cdot N \cdot K / \delta _ { f m a }
$$

Where $\delta _ { f m a }$ is used to denote the number of elements in an instruction $\mathrm { e . g . }$ In an AVX512 VNNI instruction that computes a column-wise dot product of two $4 \times 1 6$ matrices of 8-bit integers, $\delta _ { f m a } = 4 \times 1 6 = 6 4$

Instruction Modularity. By allowing the user to define the dimensions of the core compute instruction, we accommodate new advances in the x86-ISA architecture that enable matrix instructions such as AMX, ACE, Arm SME. A matrix instruction can compute a matrix product of two matrices of size $m _ { f m a } \cdot k _ { f m a }$ and $k _ { f m a } \cdot n _ { f m a }$ in a single instruction. Therefore, the number of compute instructions is $N _ { c } = M \cdot N \cdot K / ( m _ { f m a } \cdot k _ { f m a } \cdot n _ { f m a } )$

The latency of the compute instructions is computed as $N _ { f m a } * \tau _ { f m a }$ where $\tau _ { f m a }$ is the instruction reciprocal throughput for the core computation.

In addition to the latency of the compute instruction, the total compute latency $L _ { c }$ includes the latency of support instructions for zeroing the registers $L _ { z }$ , loading the input matrices $L _ { d i n }$ into the registers (assuming perfect cache behavior), and storing the output matrix back to memory $L _ { d o u t } .$ We assume that stores hit the nearest cache and thus incur the lowest latency.

2.1.2 Memory latency. The memory latency $L _ { m }$ is computed using the peak memory bandwidth applicable for the GEMM operation.

$$
L _ { m } ( o p ) = { \frac { ( M \cdot K + K \cdot N + M \cdot N ) \cdot W _ { d } \cdot f } { D _ { b w } \cdot d _ { n } } }
$$

where $d _ { n } \leq D _ { n }$ is the number of memory channels accessible to the GEMM operation.

When running multithreaded GEMM programs, the bandwidth available to each thread may be lower than the peak bandwidth because of contention among threads accessing memory. We capture the efect of this contention using a model that distributes work across diferent memory channels, allowing us to predict data-access latency more accurately.

For example, on AMD ‘Zen 5’ processors, if a GEMM operation uses fewer than eight threads (the size of a single Core Complex Die, or CCD), it can utilize the memory bandwidth of a single CCD. If more threads are used, the operation may need to access multiple CCDs, potentially increasing efective memory latency because of contention across memory channels.

2.1.3 Threading overhead. To compute the threading overhead, we assume that there is no data contention between threads. This assumption is grounded in practice, as it reflects the behavior of well-optimized GEMM implementations. When the assumption does not hold, the threads must synchronize to read and write data, incurring additional overhead.

It follows from our assumption that each thread consumes the entire � dimension to compute the result and the input matrices are split only in the � and � dimensions. Let $M _ { t }$ and $N _ { t }$ be the splits across the � and � dimensions of the output matrix for each thread.

The data needed by each thread is $S _ { A _ { t } } = M _ { t } \cdot K \cdot W _ { d }$ and $S _ { B _ { t } } = K \cdot N _ { t } \cdot W _ { d }$ . The data that does not fit in the caches, $\Delta = S _ { A _ { t } } + S _ { B _ { t } } - \left( S _ { L 3 } + S _ { L 2 } + S _ { L 1 } \right)$ , is repeatedly accessed from memory by multiple threads, incurring significant overhead. We define the threading overhead as follows:

$$
L _ { m t } = \operatorname* { m i n } ( \Delta \cdot M _ { t } , \Delta \cdot N _ { t } ) \cdot \frac { 1 } { D _ { b w } \cdot d _ { n } }
$$

To capture the scenario in which the � dimension is also split, we account for the cost of the reduction operation that follows the partial computations across diferent threads. This reduction typically involves summing the partial results along the � dimension, which introduces additional memory accesses and synchronization overhead.

2.1.4 Theoretical results. The Roofline model estimates peak performance for a given GEMM operation as,

$$
P = \operatorname* { m i n } ( C _ { p e a k } , B _ { p e a k } \cdot I ( o p ) ) = \frac { 2 \cdot M \cdot N \cdot K \cdot f } { L ^ { r } ( o p ) }
$$

where $C _ { p e a k }$ is the peak compute performance ofthe CPUs, $\boldsymbol { B } _ { p e a k }$ is the peak memory bandwidth of the CPU, �(��) is the operational intensity of the GEMM operation, and $L ^ { r } ( o p )$ is the Roofline latency. Formally,

$$
C _ { p e a k } = \frac { 2 \cdot \delta _ { f m a } \cdot o p . T \cdot f } { \tau _ { f m a } } B _ { p e a k } = \operatorname* { m a x } _ { D } ( D _ { b w } \cdot D _ { n } )
$$

$$
I ( o p ) = \frac { 2 \cdot M \cdot N \cdot K } { ( M \cdot K + K \cdot N + M \cdot N ) \cdot W d }
$$

Substituting the definitions, the resulting latency estimate $L ^ { r } ( o p )$ of the Roofline model is:

$$
L ^ { r } ( o p ) = \left\{ \begin{array} { l l } { \displaystyle \frac { M \cdot N \cdot K \cdot \tau _ { f m a } } { \delta _ { f m a } \cdot T } } & { \mathrm { i f ~ } I > \displaystyle \frac { C _ { p e a k } } { B _ { p e a k } } } \\ { \displaystyle \frac { ( M \cdot N + M \cdot K + K \cdot N ) \cdot W _ { d } \cdot f } { D _ { b w } ^ { m a x } \cdot D _ { n } ^ { m a x } } } & { \mathrm { o t h e r w i s e } } \end{array} \right.
$$

To compare, the Waypoint-Strict is designed to provide more realistic estimates of latency in comparison to the Roofline model. We capture the result in the following theorem,

Theorem 2.1. For any GEMM operation �� on a given hardware configuration $H , L ^ { r } ( o p ) \leq L ( o p )$

The theorem follows trivially from the definitions of $L ( o p )$ and $L ^ { r } ( o p )$ . In the compute bound case, the Waypoint-Strict accounts for additional cost in the compute latency $L _ { c }$ such as the cost ofzeroing registers, loading, and storing data. Similarly, in the memory bound case, the efective bandwidth is at most $D _ { b w } \cdot D _ { n }$ . In both cases, the additional threading overhead $L _ { m t }$ might be non-zero. Therefore, the Waypoint-Strict is tighter than Roofline.

![](images/e0d79255895440facba4a5c631438e1b424569d23b04d9908d04e2443c18a435.jpg)  
Extra latency: Waypoint-Strict vs. Roofline  
Figure 1. Additional latency captured by the Waypoint-Strict relative to Roofline across GEMM shapes, separated by hardware and data type. Boxes show the interquartile range and median; points show individual shapes.

2.1.5 Soundness of the assumptions. Figure 1 reinforces the theoretical result empirically. We compare the latency estimates of the Waypoint-Strict and the Roofline model on two diferent hardware configurations for many diferent shapes. The modest improvement over Roofline (upto 3%) is significant since performance of optimized GEMM libraries match the estimates (with 0.03% gap) for specific input matrix sizes.

## 2.2 Waypoint-Tile

While the Waypoint-Strict provides a tighter-than-roofline estimate, it does not provide a mechanism to realize the estimates. Therefore, we refine the Waypoint-Strict to account for a tiling-based GEMM implementation. The Waypoint-Tile helps identify the best tiling parameters for a given GEMM operation on a given hardware configuration. Using the tiling parameters, we construct a GEMM implementation (Section 3) that demonstrates the realizability of the estimates. To identify the best tiling parameters, we augment the GEMM operation with three levels of tiling parameters: the register tiles $( M _ { r } , N _ { r } )$ that form the two inner loops of Algorithm 1, the cache-tile parameters $( M _ { c } , N _ { c } , K _ { c } )$ and the multithreading-tile parameters $( M _ { t } , N _ { t } )$ . The Waypoint-Tile computes the latency of the GEMM operation for a given tiling option. Note that Algorithm 1 is one possible instantiation of the tiled implementation. In our implementation, we select the loop ordering that works best for a given tile parameter set.

Given the Waypoint-Tile, we then search the space of possible tiling options to identify the best. While theoretically, the search space sufers from combinatorial explosion, we describe heuristics to prune the search. In Section 4, we also show that the best tiling parameters identified using the Waypoint-Tile improve over the state-of-the-art GEMM libraries across many diferent matrix shapes and hardware configurations. Waypoint-Tile makes certain assumptions of the behavior of the underlying hardware and implementation such as strided data accesses, fixed frequency workload, and using the best compute instruction available (e.g. AVX512 on x86 Systems). For the sake of brevity, we defer the details to Appendix C

Algorithm 1 Tiled GEMM: � += � · �   
Require: $A \in \mathbb { R } ^ { M \times K } , B \in \mathbb { R } ^ { K \times N } , C \in \mathbb { R } ^ { M \times N }$   
Require: Thread tile $( M _ { t } , N _ { t } )$ , Macro tile $( M _ { c } , N _ { c } , K _ { c } )$ , Mi  
cro tile $( M _ { r } , N _ { r } , K _ { r } )$   
1: for $j _ { d } \gets 0$ to $N - 1$ step $N _ { t }$   
2: for $i _ { d } \gets 0$ to $M - 1$ step $M _ { t }$   
3: for $j _ { c } \gets j _ { d }$ to $j _ { d } + N _ { t } - 1$ step $N _ { c }$   
4: for $i _ { c } \gets i _ { d }$ to $i _ { d } + M _ { t } - 1$ step $M _ { c }$   
5: for $k _ { c } \gets 0$ to $K - 1$ step $K _ { c }$   
6: for $j _ { r }  j _ { c }$ to $j _ { c } + N _ { c } - 1$ step $N _ { r }$   
7: for $i _ { r }  i _ { c }$ to $i _ { c } + M _ { c } - 1$ step $M _ { r }$   
8: $C _ { [ i _ { r } , j _ { r } ] } \mathrel { + } = A _ { [ i _ { r } , k _ { c } ] } \cdot B _ { [ k _ { c } , j _ { r } ] }$   
9: end for   
10: end for   
11: end for   
12: end for   
13: end for   
14: end for   
15: end for

Apart from providing accurate estimates of GEMM operation for a given tiling option, the Waypoint-Tile also ranks the diferent tiling options accurately, based on their expected performance. We show empirically that the estimates of the best-tiling options are accurate within 5% for existing hardware configurations.

The Waypoint-Tile reformulates the compute cost $L _ { c }$ taking the tiling parameters into account. Specifically, we first break up the computation into micro-kernels – the inner most loop that iterates over the $( M _ { r } , N _ { r } , K _ { c } )$ dimensions. Then, for each micro-kernel, we account for additional costs such as cache access overheads when the tiling parameters are not optimal. Further, we add overheads for data movement costs at the macro-kernel level – the outer $( M _ { c } , N _ { c } )$ loops. We redefine the compute cost $L _ { c }$ as follows,

$$
{ \cal L } _ { c } = { \cal L } _ { \mu } * { \cal N } _ { \mu } + { \cal L } _ { \omega } * { \cal N } _ { \omega }
$$

where $L _ { \mu }$ and $N _ { \mu }$ are the latency and number of microkernels, and $L _ { \omega }$ and $N _ { \omega }$ are the latency and number ofmacrokernels.

We identify several analytical insights that capture the architectural behavior of the CPU and help prune the search space for the best tiling parameters. We describe the Waypoint-Tile in detail below.

2.2.1 Micro-kernel cost. The micro-kernel loop computes an output matrix of size $M _ { r } \times N _ { r }$ from input matrices of sizes $M _ { r } \times K _ { c }$ and $K _ { c } \times N _ { r }$ . The micro-kernel is the innermost loop of the GEMM operation, where the bulk of the computation occurs, and is typically implemented in handwritten assembly code optimized for the specific hardware. The data needed for the micro-kernel is fetched from the L1 cache (closest to the CPU).

The handwritten assembly code for the micro-kernel has the following segments: (1) zero/load instructions to initialize the registers used to store the result; (2) prefetch instructions to fetch future data into the cache; (3) load instructions to load the input matrices into registers; (4) FMA instructions to perform the core computation; and (5) store instructions to store the result back to memory.

We associate a cost with each of the above segments and compute their sum as the total cost of the micro-kernel. All components depend on the number of registers used in the hand-written assembly in relation to the number of registers available in the hardware.

Reuse Overhead. A key component of the micro-kernel latency cost is accurately modeling the cost of loading data. Optimal kernels under a perfect cache incur no cost to load data because of out-of-order execution. However, depending on the size of the data loaded (in terms of cache lines), a fraction of the load instructions will miss the cache and incur additional costs. We refer to this cost as reuse overhead. The reuse overhead is computed as the number of cache lines that incur a cache miss multiplied by the efective access latency from the L2 cache.

Let $A _ { r e g }$ and $B _ { r e g }$ to be the number of registers used to store data from the input matrices � and � (derived using $M _ { r }$ and $N _ { r }$ respectively). Then, then number of registers used to store the result, $C _ { r e g }$ is $C _ { r e g } = A _ { r e g } \cdot B _ { r e g } .$ . Together, the total number of registers used in one loop iteration of the micro-kernel is $R _ { u s e d } = A _ { r e g } + B _ { r e g } + C _ { r e g }$

Two equations govern the computation of reuse-overhead and, in efect, determine the optimal values of $M _ { r }$ and $N _ { r }$ First, a lower bound on the register pressure, defined as the ratio of input and output registers. Second, an upper bound on the load-to-fma latency ratio, defined as the ratio of the load instruction latency to the FMA instruction latency. Formally,

Lemma 2.2. Register pressure equation: The ratio ofinput and output registers is lower bounded by an architecturedependent constant �.

$$
\frac { A _ { r e g } + B _ { r e g } } { C _ { r e g } } \geq \alpha
$$

![](images/715521f3e4ad2d0c8412acb6e75a5b6479deb34e9f8ade560248e5d77c4f9b20.jpg)  
Figure 2. Candidate register tile sizes $( M _ { r } , N _ { r } )$ for AMD ‘Zen 5’ architecture. The candidates are generated by enumerating all possible values of $M _ { r }$ and $N _ { r }$ that satisfy the register pressure and load-to-fma latency ratio bounds.

For a given hardware architecture �, we calculate the bound � based on integer solutions to the equation,

$$
A _ { r e g } + B _ { r e g } + C _ { r e g } \leq N _ { r e g }
$$

Lemma 2.3. Load-to-fma latency ratio: The ratio of the load instruction latency to the FMA instruction latency is upper bounded by 1.

$$
\frac { \left( A _ { r e g } + B _ { r e g } \right) \cdot \tau _ { l o a d } } { C _ { r e g } \cdot \tau _ { f m a } } \leq 1
$$

When either of the conditions are violated, the microkernel stalls to load the data, incurring a non-zero reuse cost. When register pressure bound is violated, the micro-kernel is using more registers than available and hence leads to register spilling. Similarly, when load-to-fma latency ratio bound is violated, some of the loads are not hidden due to pipelining. The additional loads incur a cost of loading from the higher level cache.

The equations, along with the assumption that $N _ { r e g }$ is small (typically ≤ 32), efectively reduce the search space for optimal � and � (Figure 2 highlights the possible candidates for ‘Zen 5’). Therefore, we are able to enumerate all possibilities and find the optimal one. To further narrow the list, we order them based on the number of bytes loaded from memory in one iteration. The candidate that loads fewer bytes is preferred since it minimizes the number of total memory accesses.

Algorithm 2 Macro-kernel cost $L _ { \omega }$   
Require: Working-set size �, L2 size $S _ { L 2 } ,$ , L3 size $S _ { L 3 } ,$ thread   
count � for a given hardware configuration �   
1: procedure CapacityMissCost $\cdot ( H , T , S )$   
2: if $S \le S _ { L 2 }$ then   
3: return 0   
4: end if   
5: $S _ { L 3 - r e s i d e n t } \gets \operatorname* { m i n } ( S , S _ { L 3 } ) - S _ { L 2 }$   
6: � ← MoveFromL3Latency $( H , S _ { L 3 - r e s i d e n t } )$   
7: if $S > S _ { L 3 }$ then   
8: � ← �+FetchFromMemLatency $( H , S - S _ { L 3 } , T )$   
9: end if   
10: return �   
11: end procedure

2.2.2 Macro-kernel cost. The micro-kernel cost described above accounts for all the compute cost of the GEMM operation. In contrast, the macro-kernel cost we describe below accounts for the cost of moving data between the primary memory and the L2 cache. The macro-kernel computes an output matrix of size $M _ { c } \times N _ { c }$ using inputs of size $M _ { c } \times K _ { c }$ and $K _ { c } \times N _ { c } ,$ , and is where the bulk of data movement happens.

When computing the output, the macro-kernel loop streams through either the � or � matrix. The number ofcache misses depends on which matrix is streamed. Without loss of generality, we describe the macro-kernel cost assuming � is streamed. We define the macro-kernel cost $L _ { \omega }$ as,

$$
\begin{array} { r } { L _ { \omega } = \mathrm { C } _ { \mathrm { A P A C I T Y M I S S C o s T } } ( H , T , S _ { A _ { c } } + S _ { C _ { c } } ) } \\ { + \mathrm { ~ } \beta \cdot \mathrm { C } _ { \mathrm { A P A C I T Y M I S S C o s T } } ( H , T , S _ { B _ { c } } ) } \end{array}\tag{1}
$$

where $S _ { A _ { c } } = M _ { c } * K _ { c } * W _ { d } , S _ { B _ { c } } = K _ { c } * N _ { c } * W _ { d } ,$ and $S _ { C _ { c } } =$ $M _ { c } * N _ { c } * W _ { d }$ are the sizes of the input matrices A and B and output matrix C respectively. CapacityMissCost(�,�, �) denotes the cost of a capacity miss for a working set of size � and is defined in Algorithm 2. Essentially, if the data set overflows L2, then we compute the cost of moving data from L3 to L2 and similarly if the data overflows L3, we compute the cost of fetching the overflowed data from memory to L3. We compute the capacity cost independently for the B matrix, and the A and C matrices together due to their diferences in access patterns. Additionally, when $K _ { c } < K$ , we account for the cost of accumulating the partial results by including the cost of loading the result matrix.

The parameter $\beta \in \left[ 0 , 1 \right]$ in (1) is an architecture-dependent cost determined by the aggressiveness of the prefetcher. The prefetcher performs well when the data access pattern follows a fixed stride length, as it does when we stream through the B matrix in the macro-tile. Therefore, only a fraction of the B matrix incurs a capacity-miss cost. However, the A and C matrices are not streamed, so we do not expect the prefetcher to perform well.

## 2.3 Packing

GEMM libraries expect that the input matrices are pre-packed into memory a specific layout that depends on the tiling parameters. They expose routines to pack the matrices into the required format before performing the matrix multiplication. Without packing the inputs, the performance of the resulting GEMM operation degrades. The instruction used to load data in the innermost kernel difers from the optimal one, leading to ineficient memory accesses and suboptimal utilization of the cache hierarchy. Since the packing format depends on the tiling parameters, adopting a shape-dependent tiling strategy incurs additional costs.

However, with Waypoint-Tile, we identify a common packing strategy that works across diferent tiling parameters. Specifically, the packed layout should ensure two invari ants: each access loads an entire cache line, and successive access locations follow a fixed stride length. We find that additional packing optimizations do not yield significant performance improvements and only lead to higher packing costs. Therefore, in libzenmm, we pack the input matrices into blocks of $( M _ { p } \times K _ { p } )$ , where $M _ { p }$ and $K _ { p }$ are the least common multiples of all possible values of $M _ { r }$ and $K _ { r }$ respectively.

## 3 Model based GEMM library

To demonstrate that the parameters obtained from Waypoint are realizable and achieve the expected performance, we implement a GEMM library libzenmm that uses the parameters derived from Waypoint-Tile for a given GEMM operation and hardware.

libzenmm is implemented in C++. In the library, we first use the tiling parameters generated by Waypoint to break down the large GEMM operation and then dispatch the micro-kernel to a JIT based small GEMM library included in libzenmm. To break down the large GEMM, we use Parlooper[15], a library that dynamically generates loops with support for OMP based parallelization. Then, in the inner most loop, the JIT based small GEMM library generates the assembly instructions to compute the micro-kernel and caches it for subsequent uses. As an alternative to the JIT based GEMM library, we evaluated Libxsmm [17] and found it lacking the flexibility needed to generate kernels based on analytical-model-derived parameters. Therefore, we use our own JIT based small GEMM library. In Section 4 we report the GEMM performance comparison with state-of-the art GEMM libraries and show that libzenmm is competitive.

libzenmm supports two methods for finding the best tiling parameters: exhaustive search and fast heuristic-driven search. During exhaustive search, we evaluate all possible tiling parameters after pruning the search space based on insights from Waypoint-Tile. Specifically, we search only optimal $M _ { r }$ and $N _ { r }$ values and bound the $M _ { c } , N _ { c } ,$ , and $K _ { c }$ values based on the cache sizes. However, for latency-sensitive applications such as LLM inference, the fast heuristic search ofers a quick alternative.

![](images/31466ad093cdc751293fbc0588c053ac28e45fec470b571b8072ca456aeb374f.jpg)  
Figure 3. High-level workflow for evolving a Waypoint CPU split-factor heuristic. Seed heuristics and LLM-generated mutations are evaluated against pinned exhaustive-search optima; the resulting metrics drive selection of the best heuristic.

The fast-heuristic search is developed using recent advances in algorithmic evolution [8, 9, 27]. We use OpenEvolve [28], an open-source implementation of AlphaEvolve [27], to discover the heuristics. Our evolution process is focused on obtaining fast heuristics for a curated set of GEMM operations (or shapes) that modern LLM inference workloads invoke. However, the methodology can be used to discover heuristics that are tailored for any set of GEMM operations that the workload demands. The fast-heuristic generation is a one-time cost and the resulting heuristics can be used for all future GEMM operations.

Figure 3 demonstrates the flow to obtain the heuristic search algorithm. OpenEvolve starts with a set of seed programs as candidates for heuristics search. Our seeds are derived using insights from the tiling parameters generated by Waypoint-Tile. Then, OpenEvolve runs for many iterations, generating candidate heuristics in a genetic-mutation based algorithm. In each iteration we check the correctness of the tiling parameters and use Waypoint-Tile is used to score the candidates. A candidate is good if the estimate of the selected tile parameter is not far from the best known. After running for 100 iterations, OpenEvolve generates a fast-heuristic search algorithm that can be used to very quickly select near-optimal tiling parameters for a given set of GEMM operations and hardware. We defer the details of the OpenEvolve configuration to Appendix A.

Table 1. Evaluated GEMM shapes
<table><tr><td>Category</td><td>Count</td></tr><tr><td>Small GEMM (&lt;512)</td><td>4</td></tr><tr><td>Medium Square (512-4096)</td><td>9</td></tr><tr><td>Large GEMM (&gt;4096)</td><td>8</td></tr><tr><td>Tall skinny (M»N)</td><td>4</td></tr><tr><td>Fat Short (M«N)</td><td>2</td></tr><tr><td>Odd Shapes</td><td>4</td></tr><tr><td>LLM Shapes</td><td>20</td></tr></table>

The resulting fast-heuristic search is 18851× faster than the exhaustive search and discovers tiling parameters whose estimated latency is within 1.94% (maximum) of the best tiling parameters derived by Waypoint<sup>1</sup>. In our evaluation we show that the fast-heuristic search accelerates LLM performance on both current and future hardware. Our results include the cost of the fast heuristic search.

## 4 Evaluation

We evaluate the accuracy of our analytical model and the performance of libzenmm relative to three existing state-ofthe-art libraries, $\mathrm { A O C L _ { T M } \mathrm { - B L A S } }$ [3], $\mathrm { A O C L } _ { \mathrm { T M } ^ { - } } \mathrm { D L P }$ [2], and SFC [13]. AOCL-BLAS and AOCL-DLP are state-of-the-art tiling-based GEMM libraries that are optimized for AMD CPUs and support low-precision GEMM kernels used in machine learning workloads. While AOCL-BLAS uses handwritten micro-kernels, AOCL-DLP contains a JIT engine that dynamically generates the micro-kernel. In contrast, SFC is applicable to all x86 processors and implements a novel traversal mechanism (using space-filling curves) to minimize data access costs.

Our evaluation is focused on the AMD Zen architecture. Specifically, we evaluate on ‘Zen 5’ (Turin) [11] and ‘Zen 6’ (Venice). Across both architectures, all our evaluations are performed for 3 diferent data types, FP32, BF16, and INT8. We evaluate the performance of the libzenmm for a set of 51 unique GEMM shapes (summarized in Table 5) (306 comparisons in total) that are commonly used in modern LLM inference workloads. A detailed list of the shapes used to evaluate can be found in Appendix B.

With our evaluation we aim to answer the following research questions:

RQ1 Is the model accurate for diferent data types and architectures?

RQ2 How well does libzenmm compare to state-of-the-art implementations?

RQ3 Does using the analytical model based libzenmm improve the performance of end-to-end LLM inference?

The Waypoint-Strict and Waypoint-Tile models are evaluated against the Roofline model and the actual performance of GEMM operations implemented in the three libraries. Waypoint-Strict is tighter than Roofline and predicts actual performance to within 9.6%. Waypoint-Tile is accurate both in ranking diferent tiling options and in estimating the performance of the best tiling option compared with the libraries (geometric mean of 1.2%).

libzenmm achieves performance comparable to that of the three libraries across all configurations. When including packing costs, libzenmm beats AOCL-BLAS by 16.7%, AOCL-DLP by 13.6%, and SFC by 21.0%. Further, when comparing pure GEMM computation, libzenmm is 10.8% faster than AOCL-BLAS and achieves performance similar to AOCL-DLP (0.85% slower) and SFC (0.8% faster).

Finally, we evaluate end-to-end LLM inference performance using PACE<sub>TM</sub> [4], a vLLM-compatible framework with optimized inference kernels for AMD CPUs. We embed the fast-heuristic search in PACE and evaluate it against a baseline implementation that does not use Waypoint. We find that the shape-adaptive tiling strategy of Waypoint yields average time-to-first-token (TTFT) speedups of 1.36× to 1.98× across diferent batch sizes, input lengths, models, and hardware architectures.

For the remainder of the section, we first describe the experimental setup and then elaborate on the results for each of the research questions.

## 4.1 Experimental Setup

The models are compared against two hardware architectures. One, AMD ‘Zen 5’ (Turin) EPYC 9755 128-Core Processor with 512MB of L3 cache running at a fixed frequency of 2.7GHz. The other, AMD ‘Zen 6’ (Venice) EPYC 9996 256- Core Processor with 1GB of L3 cache running at a fixed frequency of 2.5GHz. We were provided advanced access to ‘Zen 6’ processor by AMD. While production systems are run with variable frequency, we fix the frequency to ensure consistent performance measurements and to isolate the performance improvements of our approach. We present comparison numbers to AOCL-BLAS and AOCL-DLP for all 3 data types (Int8, BFloat16, and FP32), and SFC for only 2 data types (BFloat16 and FP32) since SFC does not support Int8. The libzenmm is implemented in C++ and compiled using GCC 13.3.0. All comparisons are done by measuring cycles using the rdtsc instruction. To mitigate the efects of randomness when measuring the performance of GEMM libraries, for each shape, we run 10 warm-up trials and 100 measurement trials and report the best latency numbers. All numbers are reported using geometric mean.

Table 2. Roofline vs Waypoint-Strict comparison for different data types across architectures. We report the median, mean, and max ratios of Waypoint-Strict to Roofline estimates across diferent shapes. A value of ‘1.000x‘ means Waypoint-Strict and Roofline estimate match.
<table><tr><td rowspan=1 colspan=1>Datatype</td><td rowspan=1 colspan=1>Threads</td><td rowspan=1 colspan=1>n</td><td rowspan=1 colspan=1>Median</td><td rowspan=1 colspan=1>Mean</td><td rowspan=1 colspan=1>Max</td></tr><tr><td rowspan=1 colspan=1>Int8</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>58</td><td rowspan=1 colspan=1>1.001x</td><td rowspan=1 colspan=1>1.005x</td><td rowspan=1 colspan=1>1.031x</td></tr><tr><td rowspan=1 colspan=1>Int8</td><td rowspan=1 colspan=1>2-8</td><td rowspan=1 colspan=1>24</td><td rowspan=1 colspan=1>1.000x</td><td rowspan=1 colspan=1>1.001x</td><td rowspan=1 colspan=1>1.011x</td></tr><tr><td rowspan=1 colspan=1>Int8</td><td rowspan=1 colspan=1>&gt;8</td><td rowspan=1 colspan=1>20</td><td rowspan=1 colspan=1>1.000x</td><td rowspan=1 colspan=1>1.001x</td><td rowspan=1 colspan=1>1.002x</td></tr><tr><td rowspan=1 colspan=1>BFloat16</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>58</td><td rowspan=1 colspan=1>1.001x</td><td rowspan=1 colspan=1>1.004x</td><td rowspan=1 colspan=1>1.031x</td></tr><tr><td rowspan=1 colspan=1>BFloat16</td><td rowspan=1 colspan=1>2-8</td><td rowspan=1 colspan=1>24</td><td rowspan=1 colspan=1>1.000x</td><td rowspan=1 colspan=1>1.002x</td><td rowspan=1 colspan=1>1.013x</td></tr><tr><td rowspan=1 colspan=1>BFloat16</td><td rowspan=1 colspan=1>&gt;8</td><td rowspan=1 colspan=1>20</td><td rowspan=1 colspan=1>1.001x</td><td rowspan=1 colspan=1>1.001x</td><td rowspan=1 colspan=1>1.005x</td></tr><tr><td rowspan=1 colspan=1>Float32</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>58</td><td rowspan=1 colspan=1>1.001x</td><td rowspan=1 colspan=1>1.004x</td><td rowspan=1 colspan=1>1.031x</td></tr><tr><td rowspan=1 colspan=1>Float32</td><td rowspan=1 colspan=1>2-8</td><td rowspan=1 colspan=1>24</td><td rowspan=1 colspan=1>1.001x</td><td rowspan=1 colspan=1>1.003x</td><td rowspan=1 colspan=1>1.014x</td></tr><tr><td rowspan=1 colspan=1>Float32</td><td rowspan=1 colspan=1>&gt;8</td><td rowspan=1 colspan=1>20</td><td rowspan=1 colspan=1>1.002x</td><td rowspan=1 colspan=1>1.002x</td><td rowspan=1 colspan=1>1.006x</td></tr></table>

For measuring end-to-end LLM inference times, we integrate libzenmm into AMD-PACE[4]. Our measurement compares against a baseline version that uses TPP [14] to dispatch GEMM calls. Our inference runs on 128 threads.

## 4.2 RQ1: Model accuracy

Table 2 shows the comparison of Waypoint-Strict against the Roofline model across diferent data types and architectures. As shown, Waypoint-Strict is tighter than the Roofline model. Specifically when the data needed for compu tation is large, Waypoint-Strict deviates from the Roofline (up to 1.031x) accounting for the storage, and multi-threaded execution costs.

Waypoint-Tile predicts the performance of a given tiling solution more accurately than Waypoint-Strict but does not guarantee a tight upper bound. Figure 4 compares the predictions of Waypoint-Tile and Waypoint-Strict with the measured performance of the three implementations. As shown, Waypoint-Strict serves as a tight upper bound on GEMM performance. The depicted AOCL-BLAS performance measurements form a bimodal distribution, with the two modes corresponding to the diferent architectures (‘Zen 5’ and ‘Zen 6’).

Further, the Waypoint-Tile is able to accurately rank different tiling options for a given GEMM shape. To ensure the correctness of the ranking, we evaluate the 1st, 5th, and 10th tiling options for each GEMM shape and compare the predicted performance against the measured performance. We find that the Waypoint-Tile is able to accurately rank the top tiling option for 88.7% (within 5% of the peak performance) of the evaluated GEMM shapes and strictly the best in 62.7% of the cases.

Future hardware predictions. A key contribution of Waypoint is its ability to predict the performance of future proposed hardware. To demonstrate this, we evaluate the performance of the GEMM operation on a hypothetical future hardware that implements the proposed AI compute instructions [33] where one Tile operation computes a 16 × 16 × 4 (for Int8) and $1 6 \times 1 6 \times 2$ (for BFloat16). Table 3 shows the predicted performance gap of the GEMM averaged across all shapes for diferent throughput configurations of the AI compute instructions relative to ‘Zen 5’ hardware. In ‘Zen 5’, the VNNI instruction can be issued at the rate of 2/cycle and the tile op does the work equivalent to 16 VNNI instruction. As shown in the table, the model tempers the expected performance improvements of 8× (1 tile op/cycle),16×, and 32×, accounting for the data movement costs when compute is accelerated. The result shows that adding many compute units will not automatically scale the performance linearly because the kernels will become memory bound.

Figure 4. Waypoint-Tile vs Waypoint-Strict vs Implementations (AOCL-BLAS, SFC, and AOCL-DLP). We report the ratios and distribution for all the shapes.  
![](images/6cd497ed264189ff493c9d540f7ae03491e0de9ebcbdd9405245f47b182147e0.jpg)

Table 3. Predicted geometric-mean GEMM speedup from the proposed AI Compute Extensions (ACE) across 44 shapes per datatype vs ‘Zen 5’ hardware. ACE throughput is efective number of Tile operations that can be issued per cycle.
<table><tr><td rowspan=1 colspan=1>ACE throughput |</td><td rowspan=1 colspan=1>BFloat16</td><td rowspan=1 colspan=1>Int8</td></tr><tr><td rowspan=3 colspan=1>124</td><td rowspan=1 colspan=1>7.21x</td><td rowspan=1 colspan=1>7.45x</td></tr><tr><td rowspan=1 colspan=1>12.84x</td><td rowspan=1 colspan=1>13.48x</td></tr><tr><td rowspan=1 colspan=1>21.47x</td><td rowspan=1 colspan=1>23.00x</td></tr></table>

## 4.3 RQ2: libzenmm performance comparison

Figure 5 compares the performance of the libzenmm relative to three state-of-the-art implementations and the Waypoint-Tile estimates for the same set of GEMM shapes. We observe that the libzenmm closely follows the performance predicted by the Waypoint-Tile, demonstrating the accuracy and practical utility of our approach. On ‘Zen 5’, libzenmm meets the performance of existing, highly optimized GEMM libraries. However on ‘Zen 6’ the libraries are yet to be tuned (especially for the Int 8 datatype) as the hardware is not readily available, but libzenmm demonstrates a significant performance advantage over the existing implementations.

Packing cost diferences The diferent implementations also difer in their packing strategies. As mentioned in 2.3, the implementations extract peak performance by assuming that the data is pre-packed into contiguous memory regions in a specific format. For example, SFC requires that the data is laid out in the sequence of the space filling curve. This additional transformation increases the packing cost compared to more straightforward packing strategies used in AOCL-BLAS, AOCL-DLP, and libzenmm.

When comparing GEMM performance, SFC often achieves higher throughput due to its optimized data layout and tiling strategies. However, this comes at the cost of increased packing overhead. Comparing the packing costs, we find that libzenmm achieves a 1.40× speedup over AOCL-DLP, 1.29× over AOCL-BLAS, and 2.44× over SFC. We include a detailed table in Appendix B. The higher packing cost can ofset the performance gains in scenarios where the packing overhead is significant relative to the time taken to compute the result.

Discussion Including both the packing cost and GEMM latency, we observe that libzenmm is overall 1.168× faster than AOCL-BLAS, 1.210× faster than SFC, and 1.136× faster than AOCL-DLP. Conceptually, the gains of libzenmm over AOCL-BLAS and AOCL-DLP are primarily due to its shapeadaptive tiling strategies and comparable packing cost. In contrast, the gains over SFC are primarily driven by significantly lower packing cost and comparable GEMM performance matching the data reuse achieved by the optimized SFC algorithm.

## 4.4 RQ3: End-to-end LLM inference performance

We evaluate the impact of Waypoint on end-to-end inference performance of Large Language models using, AMD-PACE [4]. Figure 6 shows the average end-to-end inference speedup on prefill heavy inference workloads that primarily rely on the performance of GEMM calls. Other workloads that involve more GEMV operations retain existing performance levels. As shown, Waypoint consistently improves inference performance across diferent hardware and model configurations.

The graph depicts speedups across batch sizes of 8, 16, 32, 64, and 128 for input lengths of1024 and 2048. With higher batch sizes, the cost of inference heavily skews towards GEMM operation latency. Therefore, by integrating Waypoint, we see the highest gains in higher batch sizes (e.g. 3.6× for batch size 128 on ‘Zen 5’ with OPT 6 7B and 9.15× for batch size 128 on ‘Zen 6’ with Qwen 3 8B). The gains are attributed primarily to improvements in GEMM computation times of the odd shapes encountered in LLM inference, driven by the shape-adaptive tiling strategy of Waypoint. The graph also shows the ability of Waypoint to accelerate inference on new hardware. The range of speed up is larger in ‘Zen 6’ compared to ‘Zen 5’.

## 4.5 Extending to other architectures

While our focus in the evaluation is on AMD x86 processor family, the principles used in both Waypoint-Strict and Waypoint-Tile are general and can be easily extended to other architectures. The estimates of performance depend primarily on a layered memory hierarchy and a pipelined vector/matrix processing unit.

The model immediately translates to Intel x86 processors given their similarity in ISA and hardware architecture to AMD x86 processors. Further, the model can be adopted to ARM processors. ARM processors also feature a layered memory hierarchy and pipelined vector/matrix processing units, although the specific details of the cache sizes, memory bandwidth, and vector instruction sets difer. Specifically, the variable vector length in ARM’s Scalable Vector Instruction and Scalable Matrix Extension fit well with the parametric nature of Waypoint-Strict and Waypoint-Tile. The length and number of lanes of the registers are parametric in the model and can be configured based on the hardware architecture.

## 5 Related work

There has been extensive work on modeling and optimizing the performance of General Matrix Multiplication (GEMM) operations on various hardware architectures.

GotoBLAS [16] introduced a layered approach to implement GEMM operations in the BLAS standard library [1]. They identified the optimization that most libraries [2, 3, 30] today adopt. Namely, that breaking down the large GEMM operation into smaller blocks (tiles) significantly improves cache utilization and overall performance. Subsequent works [6, 7, 10] developed empirical methods to identify the optimal tiling parameters for a given hardware architecture and study the efects of diferent tiling choices on performance [5].

However, to overcome the time-consuming and resource intensive process of empirically identifying the best tiling parameters, analytical model approaches [24, 34, 35] have been proposed. These models [26] adapt to diferent hardware and to specific classes of shapes [31]. Waypoint extends prior works to identify optimal tiling parameters by being parametric and general not just in the hardware architecture but also in the specific input matrix shape.

![](images/c22e87c3cffbf43dd5936fd5ab1f9afbb43d5aaf64b0f94c7128e9edb386938b.jpg)  
Figure 5. Relative performance of AOCL-BLAS, SFC, AOCL-DLP, and the Waypoint-Tile vs libzenmm, separated by hardware architecture.

![](images/351313919a8fe663ac033da4c0fe6a8bbc5b1cdf5d5d94e017ee24be9d635e0f.jpg)  
Figure 6. End-to-end inference speedup of Waypoint on prefill heavy LLM workloads across diferent hardware and models. Bars show the geometric mean; whiskers show the full range across batch sizes and input lengths.

The performance of tile-based GEMM implementations are sensitive to the data-layout of the input matrices in memory. To alleviate this, recent works have borrowed ideas from communication-avoiding algorithms [12, 13] to reduce data movement and improve cache utilization. They propose and show that by carefully organizing the input matrices in the order of a Space Filling Curve reduces cache misses and helps extract peak performance. Waypoint models are general enough to accurately predict the performance of such alternative implementations while at the same time, matching their performance with libzenmm.

While the traditional Roofline model [32] has been widely adopted to predict peak performance on proposed hardware designs, it fails to provide realistic estimates. As a result, the software never ceases to be optimized on existing hardware and hardware designers are unable to predict the realistic achievable performance on proposed hardware. Execution-Cache-Memory (ECM) [18, 19], instruction-level models [22], and GPU specific models [25] have extended the Roofline model to capture diferent architectural efects and provide more realistic estimates. However, these models are either heavily tailored to a specific architecture, or require execution traces to be collected from the target hardware. Waypoint furthers the research by being both accurate and parametric in the hardware architecture, allowing hardware designers to get reliable estimates of the realistic performance achievable.

Our work is closely motivated by a similar efort focused on GPUs [29] which predicts performance of GEMM operations on GPUs by accounting for cache-miss rates, memory access overheads and other architectural efects. Our work Waypoint, generalizes the approach while demonstrating its applicability to CPUs.

## 6 Discussion and Conclusion

Under multi-threaded GEMM operations, Waypoint makes simplistic assumptions on the strategy to split work among diferent threads such as equal distribution of tiles or static scheduling, which may not always reflect the optimal parallel execution strategy. We observe that for certain input matrix sizes, using lower than the prescribed number of threads can yield better performance, highlighting the need for more sophisticated thread management strategies in future work.

We present Waypoint, a two part analytical model which first provides a tighter than Roofline estimate for GEMM performance and second, helps select the optimal tiling parameters to realize the estimates. The model derives several insights into estimating, and narrowing the search for opti mal tiling parameters, ultimately enabling high-performance GEMM across diverse hardware architectures. We implement a GEMM library libzenmm that uses the analytical model of Waypoint to guide its tiling decisions, and show that it achieves performance close to the theoretical estimates. libzenmm matches the performance of state-of-theart GEMM libraries on existing hardware while showing considerable gains on future hardware architectures. Finally, we show that using libzenmm and Waypoint speeds up endto-end CPU inference tasks significantly, demonstrating the practical impact of our approach.

## References

[1] 2002. An updated set of basic linear algebra subprograms (BLAS). ACM Trans. Math. Softw. 28, 2 (June 2002), 135–151. htps://doi.org/ 10.1145/567806.567807

[2] 2026. amd/aocl-dlp. htps://github.com/amd/aocl-dlp original-date: 2025-09-02T10:58:34Z.

[3] Advanced Micro Devices, Inc. 2025. AMD AOCL-BLAS: BLAS-like Library Instantiation Software Framework. htps://github.com/amd/ blis. Accessed: 2026-07-06.

[4] Advanced Micro Devices, Inc. 2026, 1.2.0. AMD PACE: AMD Platform Aware Compute Engine. htps: //github.com/amd/amd-pace

[5] Guillermo Alaejos, Adrián Castelló, Pedro Alonso-Jordá, Francisco D. Igual, Héctor Martínez, and Enrique S. Quintana-Ortí. 2024. Algorithm 1039: Automatic Generators for a Family of Matrix Multiplication Routines with Apache TVM. ACM Trans. Math. Softw. 50, 1, Article 6 (March 2024), 34 pages. htps://doi.org/10.1145/3638532

[6] Jason Ansel, Shoaib Kamil, Kalyan Veeramachaneni, Jonathan Ragan Kelley, Jefrey Bosboom, Una-May O’Reilly, and Saman Amarasinghe. 2014. OpenTuner: An Extensible Framework for Program Autotuning. In Proceedings of the 23rd International Conference on Parallel Architectures and Compilation (PACT ’14). Association for Computing Machinery, 303–316. htps://doi.org/10.1145/2628071.2628092

[7] Jef Bilmes, Krste Asanovic, Chee-Whye Chin, and Jim Demmel. 1997. Optimizing matrix multiply using PHiPAC: a portable, highperformance, ANSI C coding methodology. In Proceedings of the 11th International Conference on Supercomputing (Vienna, Austria) (ICS ’97). Association for Computing Machinery, New York, NY, USA, 340–347. htps://doi.org/10.1145/263580.263662

[8] Audrey Cheng, Shu Liu, Melissa Pan, Zhifei Li, Shubham Agarwal, Mert Cemri, Bowen Wang, Alexander Krentsel, Tian Xia, Jongseok Park, Shuo Yang, Jef Chen, Lakshya Agrawal, Ashwin Naren, Shulu Li, Ruiying Ma, Aditya Desai, Jiarong Xing, Koushik Sen, Matei Zaharia, and Ion Stoica. 2025. Let the Barbarians In: How AI Can Accelerate Systems Performance Research. arXiv:2512.14806 [cs.SE] htps: //arxiv.org/abs/2512.14806

[9] Audrey Cheng, Shu Liu, Melissa Pan, Zhifei Li, Bowen Wang, Alex Krentsel, Tian Xia, Mert Cemri, Jongseok Park, Shuo Yang, Jef Chen, Lakshya Agrawal, Aditya Desai, Jiarong Xing, Koushik Sen, Matei Zaharia, and Ion Stoica. 2025. Barbarians at the Gate: How AI is Upending Systems Research. arXiv:2510.06189 [cs.AI] htps://arxiv. org/abs/2510.06189

[10] R Clint Whaley, Antoine Petitet, and Jack J Dongarra. 2001. Automated empirical optimizations of software and the ATLAS project. Parallel Comput. 27, 1-2 (Jan. 2001), 3–35.

[11] Brad Cohen, Mahesh Subramony, and Mike Clark. 2024. Next Generation "Zen 5" Core. In 36th IEEE Hot Chips Symposium, HCS 2024, Stanford, CA, USA, August 25-27, 2024. IEEE, 1–27. htps: //doi.org/10.1109/HCS61935.2024.10665102

[12] James Demmel, David Eliahu, Armando Fox, Shoaib Kamil, Benjamin Lipshitz, Oded Schwartz, and Omer Spillinger. 2013. Communication-Optimal Parallel Recursive Rectangular Matrix Multiplication. In 2013 IEEE 27th International Symposium on Parallel and Distributed Processing. IEEE, 261–272. htps://doi.org/10.1109/IPDPS.2013.80

[13] Evangelos Georganas, Alexander Heinecke, and Pradeep Dubey. 2026. Space Filling Curves is All You Need: Communication-Avoiding Matrix Multiplication Made Simple. arXiv:2601.16294 [cs.DC] htps://arxiv. org/abs/2601.16294

[14] Evangelos Georganas, Dhiraj Kalamkar, Sasikanth Avancha, Menachem Adelman, Cristina Anderson, Alexander Breuer, Jeremy Bruestle, Narendra Chaudhary, Abhisek Kundu, Denise Kutnick, Frank Laub, Vasimuddin Md, Sanchit Misra, Ramanarayan Mohanty, Hans Pabst, Barukh Ziv, and Alexander Heinecke. 2021. Tensor processing

primitives: a programming abstraction for eficiency and portabil ity in deep learning workloads. In Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (St. Louis, Missouri) (SC ’21). Association for Comput ing Machinery, New York, NY, USA, Article 14, 14 pages. htps: //doi.org/10.1145/3458817.3476206

[15] Evangelos Georganas, Dhiraj Kalamkar, Kirill Voronin, Abhisek Kundu, Antonio Noack, Hans Pabst, Alexander Breuer, and Alexander Heinecke. 2024. Harnessing Deep Learning and HPC Kernels via High Level Loop and Tensor Abstractions on CPU Architectures. In 2024 IEEE International Parallel and Distributed Processing Symposium (IPDPS). IEEE, 950–963. htps://doi.org/10.1109/IPDPS57955.2024. 00089

[16] Kazushige Goto and Robert A. van de Geijn. 2008. Anatomy of High Performance Matrix Multiplication. ACM Trans. Math. Software 34, 3 (May 2008), 1–25. htps://doi.org/10.1145/1356052.1356053

[17] Alexander Heinecke, Greg Henry, Maxwell Hutchinson, and Hans Pabst. 2016. LIBXSMM: accelerating small matrix multiplications by runtime code generation. In Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (Salt Lake City, Utah) (SC ’16). IEEE Press, Article 84, 11 pages. htps://doi.org/10.1109/SC.2016.83

[18] Johannes Hofmann, Dietmar Fey, Jan Eitzinger, Georg Hager, and Gerhard Wellein. 2016. Analysis of Intel’s Haswell Microarchitec ture Using the ECM Model and Microbenchmarks. In Architecture of Computing Systems – ARCS 2016, Frank Hannig, João M. P. Cardoso, Thilo Pionteck, Dietmar Fey, Wolfgang Schröder-Preikschat, and Jürgen Teich (Eds.). Springer International Publishing, Cham, 210–222.

[19] Aleksandar Ilic, Frederico Pratas, and Leonel Sousa. 2014. Cache-aware Roofline model: Upgrading the loft. IEEE Computer Architecture Letters 13, 1 (2014), 21–24. htps://doi.org/10.1109/L-CA.2013.6

[20] Intel. 2026. Introducing Intel® Advanced Performance Extensions (Intel® APX) — intel.com. htps://www.intel.com/content/www/us/ en/developer/articles/technical/advanced-performance-extensionsapx.html [Accessed 10-09-2026].

[21] Intel Corporation. 2022. Accelerate Artificial Intelligence (AI) Workloads with Intel Advanced Matrix Extensions (Intel AMX). Solution Brief. Intel Corporation. htps://www.intel.com/content dam/www/central-libraries/us/en/documents/2022-12/accelerateai-with-amx-sb.pdf

[22] Jan Laukemann, Julian Hammer, Johannes Hofmann, Georg Hager, and Gerhard Wellein. 2018. Automated Instruction Stream Through put Prediction for Intel and AMD Microarchitectures. In 2018 IEEE/ACM Performance Modeling, Benchmarking and Simulation of High Performance Computer Systems (PMBS). 121–131. htps: //doi.org/10.1109/PMBS.2018.8641578

[23] Chendi Li, Haipeng Jia, Hang Cao, Jianyu Yao, Boqian Shi, Chunyang Xiang, Jinbo Sun, Pengqi Lu, and Yunquan Zhang. 2021. AutoTSMM: An Auto-tuning Framework for Building High Performance Tall-and-Skinny Matrix-Matrix Multiplication on CPUs. In 2021 IEEE Intl Conf on Parallel and Distributed Processing with Applications, Big Data and Cloud Computing, Sustainable Computing and Communications, Social Computing and Networking (ISPA/BDCloud/SocialCom/SustainCom). 159–166. htps://doi.org/10. 1109/ISPA-BDCloud-SocialCom-SustainCom52081.2021.00034

[24] Tze Meng Low, Francisco D. Igual, Tyler M. Smith, and Enrique S. Quintana-Orti. 2016. Analytical Modeling Is Enough for High-Performance BLIS. ACM Trans. Math. Softw. 43, 2, Article 12 (Aug. 2016), 18 pages. htps://doi.org/10.1145/2925987

[25] Sangkug Lym, Donghyuk Lee, Mike O’Connor, Niladrish Chatterjee, and Mattan Erez. 2019. DeLTA: GPU Performance Model for Deep Learning Applications with In-Depth Memory System Trafic Analysis. In 2019 IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS). 293–303. htps://doi.org/10.1109/

ISPASS.2019.00041

[26] Héctor Martínez, Sandra Catalán, Francisco D Igual, José R Herrero, Rafael Rodríguez-Sánchez, and Enrique S Quintana-Ortí. 2025. Cacheaware optimization of matrix multiplication and matrix factorizations on multicore processors. Cluster Comput. 28, 12 (Nov. 2025).

[27] Alexander Novikov, Ngân Vu, Marvin Eisenberger, Emilien Dupont,˜ Po-Sen Huang, Adam Zsolt Wagner, Sergey Shirobokov, Borislav Kozlovskii, Francisco J. R. Ruiz, Abbas Mehrabian, M. Pawan Kumar, Abigail See, Swarat Chaudhuri, George Holland, Alex Davies, Sebastian Nowozin, Pushmeet Kohli, and Matej Balog. 2025. AlphaEvolve: A coding agent for scientific and algorithmic discovery. arXiv:2506.13131 [cs.AI] htps://arxiv.org/abs/2506.13131

[28] Asankhaya Sharma. 2025. OpenEvolve: an open-source evolutionary coding agent. htps://github.com/algorithmicsuperintelligence/ openevolve

[29] Ryan Swann, Muhammad Osama, Xiaohu Guo, Bryant Nelson, Lixun Zhang, Alex Brown, Yen Ong, Ali Yazdani, Sean Siddens, Ganesh Dasika, and Alex Underwood. 2025. tritonBLAS: Tritonbased Analytical Approach for GEMM Kernel Parameter Selection. arXiv:2512.04226 [cs.DC] htps://arxiv.org/abs/2512.04226

[30] Field G. Van Zee and Robert A. van de Geijn. 2015. BLIS: A Framework for Rapidly Instantiating BLAS Functionality. ACM Trans. Math. Software 41, 3 (June 2015), 14:1–14:33. htps://doi.org/10.1145/2764454

[31] Cunyang Wei, HaipengJia, Yunquan Zhang,Jianyu Yao, Chendi Li, and Wenxuan Cao. 2024. IrGEMM: An Input-Aware Tuning Framework for Irregular GEMM on ARM and X86 CPUs. IEEE Transactions on Parallel and Distributed Systems 35, 9 (2024), 1672–1689. htps://doi. org/10.1109/TPDS.2024.3432579

[32] Samuel Williams, Andrew Waterman, and David A. Patterson. 2009. Roofline: an insightful visual performance model for multicore archi tectures. Commun. ACM 52, 4 (2009), 65–76. htps://doi.org/10.1145/ 1498765.1498785

[33] x86 Ecosystem Advisory Group. 2026, 1.16.2. AI Compute Extensions (ACE) Specification. Technical Report. x86 Ecosystem Advisory Group. htps://x86ecosystem.org/resource/aicompute-extensions-ace-specification Specification Release.

[34] Kamen Yotov, Xiaoming Li, Gang Ren, Michael Cibulskis, Gerald De-Jong, Maria Garzaran, David Padua, Keshav Pingali, Paul Stodghill, and Peng Wu. 2003. A comparison of empirical and model-driven optimization. SIGPLAN Not. 38, 5 (May 2003), 63–76. htps://doi.org/ 10.1145/780822.781140

[35] K. Yotov, Xiaoming Li, Gang Ren, M.J.S. Garzaran, D. Padua, K. Pingali, and P. Stodghill. 2005. Is Search Really Necessary to Generate High-Performance BLAS? Proc. IEEE 93, 2 (2005), 358–386. htps://doi.org/ 10.1109/JPROC.2004.840444

## A Appendix A: Heuristic Search Details

We run OpenEvolve to discover fast-search heuristics that ofer a fast alternative to the exhaustive search of Waypoint. To discover the heuristics, we first curate a set of GEMM operations (or shapes) that modern LLM inference workloads invoke. Then, we build a baseline of best tiling parameters for each shape using Waypoint.

Starting with a predefined set of seed programs that contain initial heuristics, OpenEvolve runs many iterations, generating candidate heuristics that are evaluated against the baseline. Below we elaborate on the details of the seed programs and the evaluation process.

## A.1 Seed programs

We generate 6 seed programs each containing independent set of heuristics that target a specific class of the shapes. The goal with the seeds is to provide a diverse set of possible heuristics that could be used to arrive at the final program and allow OpenEvolve to mutate, combine, and evolve them over many iterations to discover efective fast-search heuristics. Each seed program corresponds to a single island of evolution in OpenEvolve. The seeds are,

1. Tiling based primarily on data type and ratio of matrix dimensions.

2. Uses L1/L2 cache sizes to bound the tiling options.

3. Targets optimal multi-threading splits based on the shape of the GEMM operation.

4. Use arithmetic intensity to guide the inner most tiling decisions.

5. Target specifically the AI Compute Instructions / AMX instructions.

6. Initially baseline or the previous best program.

## A.2 Fast heuristic search results

We plot below the average diference in expected latency between the fast heuristic search algorithm and the exhaustive search algorithm.

## B Appendix B: GEMM evaluation shapes

## B.1 Evaluation shapes

Table 5 lists the 51 unique GEMM shapes used in our evaluation. Each shape is evaluated on both hardware configurations and for all three data types described in Section 4.

## B.2 Packing cost

Table 6 contains the packing cost speedup of libzenmm. Values are geometric-mean ratios of implementation packing cycles to libzenmm packing cycles for all the shapes.

## C Model Assumptions

The estimates derived using Waypoint (both the Simple and Waypoint-Tiles) rely on certain assumptions about the behavior of the underlying hardware. We list below the assumptions along with the reasoning.

Fixed frequency workloads: Waypoint estimates the performance of a GEMM operation assuming that the CPU frequency is fixed. We calculate the number of cycles required to access data from memory using the frequency of the machine. Under a variable frequency scenario, one can extend the model by incorporating the dynamic changes in frequency to adjust the cycle estimates accordingly.

Best instruction for compute: Due to the highly optimized nature of GEMM kernels, Waypoint assumes that the best instruction i.e. the one with the highest throughput and lowest latency, is used for the compute operations. For example, in x86-ISA based CPU, Waypoint assumes that the

AVX512 FMA/VNNI instruction is used compute operations. We do not explore the possibility of a mixed instruction use for a specific workload.

Strided Data access: Waypoint assumes that the data needed for compute is loaded into the registers in strides of contiguous memory locations. Waypoint accounts for the strided access by assuming the lowest cost for loading data into registers. For GEMM workloads, the assumption is valid since data is pre-packed to align accesses with cache lines and stride lengths.

Cache memory model: Waypoint assumes that the different level of caches provide diferent access latencies and that lower levels of caches have faster access times. We do not make any assumptions on the inclusivity, write-back policies, or replacement strategies of the caches. The model only considers the access latency diferences between cache levels. A more complex model of the caches will help in determining the access latencies better.

Partial bandwidth when loading data: For multi-core (thread) GEMM operations, Waypoint assumes that the memory bandwidth is shared across all threads. Hence, each thread only receives a fraction of the total bandwidth avail able. The fractional cost is accounted for when computing the data accesses from memory.

Peak store throughput: We assume that stores occur at the peak throughput of the memory subsystem. Since the result-matrix locations targeted by store instructions typically reside in caches rather than main memory, the store operations are not expected to be bottlenecked by main memory and can achieve peak throughput.

Table 4. Average diference in expected latency between the fast heuristic search algorithm and the exhaustive search algorithm across diferent shape categories and architectures.
<table><tr><td>Shape category</td><td>Z5 Float</td><td>Z5 BF16</td><td>Z5 Int8</td><td>Z6 Float</td><td>Z6 BF16</td><td>Z6 Int8</td></tr><tr><td>Small Square</td><td>0.002%</td><td>0.000%</td><td>0.443%</td><td>0.010%</td><td>0.018%</td><td>0.000%</td></tr><tr><td>Medium Square</td><td>0.144%</td><td>0.060%</td><td>0.059%</td><td>0.242%</td><td>0.251%</td><td>0.426%</td></tr><tr><td>Large Square</td><td>0.089%</td><td>0.059%</td><td>0.055%</td><td>0.234%</td><td>0.295%</td><td>0.514%</td></tr><tr><td>Tall Skinny</td><td>0.692%</td><td>0.145%</td><td>0.473%</td><td>0.111%</td><td>0.093%</td><td>0.006%</td></tr><tr><td>Fat short</td><td>0.110%</td><td>0.121%</td><td>0.127%</td><td>0.106%</td><td>0.114%</td><td>0.209%</td></tr><tr><td>Odd Shapes</td><td>0.698%</td><td>0.651%</td><td>0.838%</td><td>0.710%</td><td>0.698%</td><td>1.590%</td></tr><tr><td>LLM Shapes</td><td>0.250%</td><td>0.200%</td><td>0.235%</td><td>0.081%</td><td>0.074%</td><td>0.292%</td></tr><tr><td>Overall</td><td>0.247%</td><td>0.157%</td><td>0.253%</td><td>0.200%</td><td>0.211%</td><td>0.427%</td></tr></table>

Table 5. GEMM evaluation shapes. �, �, and � are the matrix dimensions, and � is the number of threads.
<table><tr><td>M</td><td>N</td><td>K</td><td>T|</td><td>M</td><td>N</td><td>K</td><td>T</td></tr><tr><td>64</td><td>64</td><td>64</td><td>1</td><td>4096</td><td>64</td><td>4096</td><td>1</td></tr><tr><td>64</td><td>2048</td><td>64</td><td>1</td><td>4096</td><td>128</td><td>4096</td><td>1</td></tr><tr><td>96</td><td>96</td><td>96</td><td>1</td><td>4096</td><td>4096</td><td>4096</td><td>1</td></tr><tr><td>128</td><td>32</td><td>2880</td><td>1</td><td>128</td><td>14336</td><td>4096</td><td>4</td></tr><tr><td>128</td><td>128</td><td>128</td><td>1</td><td>128</td><td>28672</td><td>4096</td><td>4</td></tr><tr><td>128</td><td>2880</td><td>2880</td><td>1</td><td>128</td><td>128256</td><td>4096</td><td>4</td></tr><tr><td>128</td><td>2880</td><td>4096</td><td>1</td><td>2048</td><td>2048</td><td>2048</td><td>4</td></tr><tr><td>128</td><td>2880</td><td>5760</td><td>1</td><td>4096</td><td>4096</td><td>4096</td><td>4</td></tr><tr><td>128</td><td>4096</td><td>128</td><td>1</td><td>8192</td><td>8192</td><td>8192</td><td>4</td></tr><tr><td>128</td><td>4096</td><td>4096</td><td>1</td><td>128</td><td>14336</td><td>4096</td><td>8</td></tr><tr><td>128</td><td>4096</td><td>14336</td><td>1</td><td>128</td><td>28672</td><td>4096</td><td>8</td></tr><tr><td>128</td><td>5120</td><td>2880</td><td>1</td><td>128</td><td>128256</td><td>4096</td><td>8</td></tr><tr><td>128</td><td>5760</td><td>2880</td><td>1</td><td>2048</td><td>2048</td><td>2048</td><td>8</td></tr><tr><td>128</td><td>6144</td><td>4096</td><td>1</td><td>4096</td><td>4096</td><td>4096</td><td>8</td></tr><tr><td>128</td><td>14336</td><td>4096</td><td>1</td><td>8192</td><td>8192</td><td>8192</td><td>8</td></tr><tr><td>128</td><td>28672</td><td>4096</td><td>1</td><td>128</td><td>14336</td><td>4096</td><td>16</td></tr><tr><td>128</td><td>128256</td><td>4096</td><td>1</td><td>128</td><td>128256</td><td>4096</td><td>16</td></tr><tr><td>256</td><td>256</td><td>256</td><td>1</td><td>2048</td><td>2048</td><td>2048</td><td>16</td></tr><tr><td>288</td><td>192</td><td>384</td><td>1</td><td>4096</td><td>4096</td><td>4096</td><td>16</td></tr><tr><td>512</td><td>512</td><td>512</td><td>1</td><td>8192</td><td>8192</td><td>8192</td><td>16</td></tr><tr><td>768</td><td>512</td><td>256</td><td>1</td><td>16384</td><td>16384</td><td>16384</td><td>16</td></tr><tr><td>992</td><td>768</td><td>1024</td><td>1</td><td>8192</td><td>8192</td><td>8192</td><td>32</td></tr><tr><td>1024</td><td>1024</td><td>1024</td><td>1</td><td>16384</td><td>16384</td><td>16384</td><td>32</td></tr><tr><td>2048</td><td>64</td><td>2048</td><td>1</td><td>16384</td><td>16384</td><td>16384</td><td>64</td></tr><tr><td>2048</td><td>128</td><td>2048</td><td>1</td><td>16384</td><td>16384</td><td>16384</td><td>128</td></tr><tr><td>2048</td><td>2048</td><td>2048</td><td>1</td><td></td><td></td><td></td><td></td></tr></table>

Table 6. Packing cost relative to libzenmm. Values are geometric-mean ratios of implementation packing cycles to libzenmm packing cycles for diferent thread counts.
<table><tr><td rowspan=1 colspan=1>HardwareImpl</td><td rowspan=1 colspan=1>Type</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>2-8</td><td rowspan=1 colspan=1>&gt; 8</td></tr><tr><td rowspan=1 colspan=5>‘Zen 5’</td></tr><tr><td rowspan=1 colspan=1>AOCL-BLAS</td><td rowspan=1 colspan=1>INT8BFloat16Float32</td><td rowspan=1 colspan=1>2.02×0.40×1.11×</td><td rowspan=1 colspan=1>3.01×1.37×1.78×</td><td rowspan=1 colspan=1>10.34×4.40×5.05×</td></tr><tr><td rowspan=1 colspan=1>AOCL-DLP</td><td rowspan=1 colspan=1>INT8BFloat16Float32</td><td rowspan=1 colspan=1>2.11×0.44×1.17×</td><td rowspan=1 colspan=1>3.75×1.42×1.89×</td><td rowspan=1 colspan=1>10.23×4.48×5.05×</td></tr><tr><td rowspan=1 colspan=1>SFC</td><td rowspan=1 colspan=1>BFloat16Float32</td><td rowspan=1 colspan=1>2.05×3.92×</td><td rowspan=1 colspan=1>1.64×2.26×</td><td rowspan=1 colspan=1>1.73×3.66×</td></tr><tr><td rowspan=1 colspan=5>‘Zen 6’</td></tr><tr><td rowspan=1 colspan=1>AOCL-BLAS</td><td rowspan=1 colspan=1>INT8BFloat16Float32</td><td rowspan=1 colspan=1>1.62×0.50×1.01×</td><td rowspan=1 colspan=1>1.51×0.65×0.79×</td><td rowspan=1 colspan=1>1.94×0.97×1.52×</td></tr><tr><td rowspan=1 colspan=1>AOCL-DLP</td><td rowspan=1 colspan=1>INT8BFloat16Float32</td><td rowspan=1 colspan=1>1.60×0.55×1.13×</td><td rowspan=1 colspan=1>2.93×0.61×0.86×</td><td rowspan=1 colspan=1>2.35×0.98×1.57×</td></tr><tr><td rowspan=1 colspan=1>SFC</td><td rowspan=1 colspan=1>BFloat16Float32</td><td rowspan=1 colspan=1>2.45×3.49×</td><td rowspan=1 colspan=1>1.15×1.89×</td><td rowspan=1 colspan=1>1.36×3.55×</td></tr></table>