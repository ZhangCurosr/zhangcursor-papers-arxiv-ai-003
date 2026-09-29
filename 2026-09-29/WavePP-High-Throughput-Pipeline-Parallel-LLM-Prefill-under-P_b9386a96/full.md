# WavePP: High-Throughput Pipeline Parallel LLM Prefill under Prefix Reuse

Aaryam Sharma Baseten<sup>1</sup> aaryam.sharma@baseten.co

## Abstract

Pipeline parallelism can improve prefill throughput by processing multiple request chunks concurrently across different stages of the model. However, keeping the pipeline fully utilized requires efficient scheduling and request preparation. In systems where stages retain and evict cache state independently, a local cache hit does not guarantee that the same prefix can be reused across the pipeline. Here, coordination overhead can impede request admission cadence and thus reduce overall throughput. In this paper, we present WavePP, a prefill runtime built on top of TensorRT-LLM that addresses these challenges by overlapping request admission with pipeline execution. WavePP asynchronously finds a prefix that can be reused across all stages, protects the cached state, and reserves space for the remaining input while earlier requests continue to execute. It subsequently plans the chunk sizes of each request dynamically to maximize pipeline fill. Each stage then completes the local preparation before executing the request. In the same system and pipeline topology, WavePP improves TensorRT-LLM’s prefill throughput in 37 of 40 tested settings on GLM 5.2 and MiniMax M2.7. At concurrency 128 with high cache reuse, these changes increase throughput by factors of 2.91 and 2.02, respectively. Across 28 Kimi K3 settings, WavePP also has the highest measured throughput in all 18 settings at concurrency eight or higher, compared with tensor/expert-parallel and pipeline-parallel baselines from TRT-LLM, SGLang, and vLLM.

## 1. Introduction

Large Language Models (LLMs) have become ubiquitous in our daily lives. As agentic workloads have gained popularity, LLMs increasingly process larger amounts of text in the conversation, documents, and outputs from tools. Even if only a few tokens are generated, serving such requests can be computationally expensive [1]. Therefore, improving inference performance has become a key focus of recent serving systems [2–4].

LLM inference is often split into a prefill phase and a decode phase. The prefill phase processes the initial input tokens and fills the cache with the necessary state for generation, while the decode phase autoregressively generates new output tokens, continuously updating the same cache. In many enterprise systems, this work is often split between separate worker types using disaggregated serving. This is especially effective when multiple long context requests need to be processed so that the prefill computation remains decoupled from the decode work.

In addition to increasing input sizes, however, LLMs themselves have also substantially grown in size. The largest frontier models often do not fit on a single GPU. Including space for caches, models are often split over even more GPUs. Several parallelism strategies exist to achieve this, such as tensor parallelism (TP) which divides individual layers across GPUs, pipeline parallelism (PP) splits the model into contiguous groups of layers, and context parallelism (CP) divides work along the request input length. Each strategy has different costs in terms of communication, but also the concurrent amount of work that is done [5–7]. In particular, each parallelism strategy not only has specific advantages for different workloads, but also for different inference phases.

PP has often been used for throughput efficiency in both training and inference. Since several microbatches are passed through successive groups of layers in parallel, it allows several parts of the model to execute concurrently, also increasing the number of concurrent inputs processed [6, 8, 9]. Many training systems hence also support PP for general neural networks. For inference, many popular libraries support pipeline parallelism for prefill [10, 11] and several works optimize scheduling when prefill and decode share a pipeline. These optimizations include improving scheduling to reduce bubbles between prefill and decode batches [4], dynamically chunking requests to reduce imbalance between pipeline stages [9], and overlapping host-side work with GPU work efficiently for prefill or decode [12, 13].

Due to the autoregressive nature of most modern LLMs, the cached states of a shared prefix can be reused across requests which greatly reduces computation in online serving, agentic workflows, and batch inference with common shared prompts [2, 3, 14–16]. For PP, each stage typically contains a local cache for its layers, making request preparation a distributed operation. This coordination can be simplified with a shared cache view, or by replicated schedulers operating on compatible logical states in practical inference systems. Instead, we consider an alternative system, where KV blocks and recurrent-state snapshots may be retained, offloaded, and evicted independently, both within a stage and also across pipeline stages. This enables each stage to conduct host side cache management independently, while ranks agree on reuse and follow a common execution schedule. Each cache manager can respond to local memory pressure and prepare requests around local execution. When pipeline stages are not balanced, one stage can prepare a request while another is still computing.

However, this means that in our setting, a local cache hit does not guarantee that this point can be reused globally. The ranks hence must agree on a common reuse boundary, protect the agreed upon state, and reserve sufficient capacity for the uncached input suffix. Performing this work on the main execution path can delay the next forward pass, meaning that our runtime must establish these guarantees sufficiently early to maintain pipeline occupancy.

We thus present WavePP, a runtime design that establishes this admission agreement efficiently while earlier requests continue through the pipeline. Each stage protects the reusable state and reserves space for the rest of the input before the request is scheduled. The stages then finish preparing their local caches independently, allowing stage 0 to begin without waiting for every stage to finish. The executor sees a sequence of waves, each containing one or more request chunks, hence leading to our name.

Our contributions are:

• We design WavePP, an asynchronous prefill runtime that overlaps request admission with pipeline execution while allowing stages to manage their caches independently.

• We develop a protocol that agrees on reusable state and reserves space for the remaining input before allocating the uncached suffix’s blocks. This lets stages prepare requests independently and delays eviction until the reserved space is needed.

• We show that faster admission, more waves in flight, and adaptive wave sizing together improve throughput on the same kernels. Within TensorRT-LLM PP4, these changes improve throughput in 37 of 40 GLM 5.2 and MiniMax M2.7 settings, with 2.91× and 2.02× throughput for short-suffix requests under cache pressure at concurrency 128.

• We compare WavePP with TP and PP baselines across cold, cached, and mixed Kimi K3 workloads. It has the highest measured throughput in all 18 settings at concurrency eight or higher.

## 2. Related Works

## 2.1. Pipeline Parallelism

Pipeline parallelism has been extensively studied for training. GPipe partitions a model across devices, and accumulates gradients before each training update [6]. PipeDream also overlaps work from different minibatches, using weight versions to support pipelined forward and backward execution [8]. More recently, LLMs also have several frameworks for efficient PP training [17, 18]. TeraPipe further uses a dynamic programming-based algorithm to optimize the pipeline execution in training [19].

While there is no backward pass in inference, pipeline parallelism can still be beneficial for throughput. Sarathi-Serve combines chunked prefill with stall-free batching to schedule prefill alongside concurrent decode passes and reduce variation between microbatches [4]. gLLM separately regulates prefill and decode token counts, where its prefill budget accounts for both pending work and KV-cache availability [12]. Revisiting Pipeline Parallelism dynamically adjusts chunk sizes and delays selected decode requests to reduce imbalance [9]. WavePP also adjusts the wave token budget, while focusing on preparing requests early enough to keep the executor supplied with work.

Several works focus on improving execution around each forward pass, for both decode and prefill. gLLM sends metadata ahead of activations so workers can prepare inputs during computation [12]. TD-Pipe temporally separates prefill and decode for offline inference, balances decode batches, and separates centralized scheduling from distributed execution [20]. SiPipe moves sampling to CPUs and overlaps input preparation with GPU work [13]. VPP uses a virtual-stage arrangement for long-context prefill, reducing imbalance while keeping chunk sizes fixed [21].

Many practical inference libraries also support PP. In vLLM’s V1 PP runtime, an engine-core scheduler and cache manager coordinate the distributed workers [11]. SGLang uses replicated schedulers over an ordered request stream [22]. Both support overlapping pipeline execution, including successive prefill chunks of the same request in flight [22, 23]. WavePP focuses on distributed admission where it agrees on reusable state, protects that state, and reserves suffix capacity outside the executor loop, while each stage manages its local cache, reducing the effect of admission on the execution cadence.

Another common direction of improvements to PP is in the allocation of layers across pipeline stages. Some approaches enable live redistribution of layers in real time, including migration of cache across stages such as in DynaPipe and PipeLive [24, 25]. Other approaches such as Helix optimize static model placement and request scheduling across heterogeneous devices as an optimization problem [26]. WavePP studies admission within a fixed pipeline configuration.

## 2.2. Cache Reuse and Serving Runtimes

Prompt caching and reuse have been established as powerful accelerators across various works [27]. Preble jointly considers prefix reuse and computation load when assigning requests across workers [1]. Most serving runtimes support cross-request cache reuse. PagedAttention in vLLM manages KV state in blocks [2], while RadixAttention in SGLang organizes reusable prefixes in a radix tree [3]. TensorRT-LLM also builds a radix tree in order to coordinate prefix reuse efficiently [28]. WavePP is implemented on top of TensorRT-LLM, adopting its caching strategy.

These caches are often also distributed across many layers of memory, including on-device, on host, disk, and remote storage. Tools such as Mooncake and LMCache enable this distributed storage, while also supporting cache transfers [16, 29]. Modern LLMs also combine attention types with different cache mechanisms, including recurrent layers whose prefix caches retain selected state snapshots. Marconi selects cache entries using their expected reuse and compute savings relative to memory cost [14]. Jenga supports heterogeneous cache allocation and caching policies tailored to the dependencies of each layer [15]. Hybrid cache management is also supported in vLLM, SGLang, and TRT-LLM [30–32].

Enterprise systems also employ disaggregated serving, separating prefill and decode work across different worker types to reduce interference and meet different latency objectives [33, 34]. Mooncake builds a disaggregated serving architecture around distributed KV storage and reuse [29]. Baseten’s discussion of the inference efficiency frontier describes how dedicated workers allow the two phases to use different configurations and scale with the workload [35]. Disaggregation is also implemented in TensorRT-LLM, vLLM, and SGLang, which provide mechanisms for transferring cache state between prefill and decode workers [36–38]. NVIDIA Dynamo coordinates disaggregated worker pools and KV-aware routing across these inference backends [39]. WavePP thus focuses on a prefill-only worker.

The terms lease and escrow have a longer history in distributed systems. Gray and Cheriton’s leases give time-limited rights over cached data [40]. WavePP uses the term for a local reference that protects a cached prefix, without time-based expiry. O’Neil’s escrow method reserves quantities for concurrent transactions [41]. Our capacity escrow similarly reserves a shared resource before use, but does not implement database transaction or recovery semantics.

## 2.3. Other Parallelism Strategies

PP can be combined with other ways of dividing model execution. Megatron-LM develops tensor parallelism, while DeepSpeed Ulysses, Ring Attention, and USP distribute long-sequence attention work across devices [5, 7, 42, 43]. Expert parallelism distributes the experts of mixtureof-experts models [44]. Helix Parallelism combines KV sharding during attention with tensor parallelism for dense feed-forward layers, or tensor and expert parallelism for MoE layers [45]. These strategies change how work and state are distributed within layers and can complement a pipeline across layer groups. We compare PP against tensor parallelism in this paper.

## 3. Preliminaries

## 3.1. Transformer Architectures

Autoregressive attention typically produces various intermediate representations that are reusable across subsequent generation steps of a request. In conventional multi-head attention (MHA), each layer retains the keys (K) and values (V) associated with its KV heads, which allows subsequent tokens to attend to the previous tokens without recomputing them. Attention mechanisms such as grouped-query and multi-query attention reduce KV storage by sharing keys and values between query heads [46, 47]. Multi-head latent attention (MLA) instead stores a lower-dimensional latent representation from which the attention content keys and values can be reconstructed [48]. The storage mechanisms for these representations are collectively referred to as the KV Cache.

Typical attention mechanisms such as MHA or MLA are $O ( n )$ in computation for the $n ^ { \mathrm { t h } }$ token in the sequence, and have a growing KV cache that scales with the sequence length. Linear attention architectures replace this growing KV cache with a recurrent state whose dimensions are independent of sequence length. A broad family of linear attention mechanisms can be written as

$$
S _ { t } = A _ { t } S _ { t - 1 } + U _ { t } , \qquad o _ { t } = f ( q _ { t } , S _ { t } ) ,\tag{1}
$$

where $S _ { t }$ captures the prefix $x _ { 1 : t }$ . This recurrent expression is a central property of linear attention [49, 50].

Kimi Delta Attention (KDA), used in Kimi K3, is one such architecture [51, 52]. KDA builds on Gated DeltaNet, combining delta-rule state updates with finer-grained gating [50, 51]. A simplified single-head KDA update can be written as

$$
\begin{array} { r } { S _ { t } = \left( I - \beta _ { t } k _ { t } k _ { t } ^ { \top } \right) \mathrm { D i a g } ( \alpha _ { t } ) S _ { t - 1 } + \beta _ { t } k _ { t } \nu _ { t } ^ { \top } , } \end{array}\tag{2}
$$

followed by

$$
\widetilde { o } _ { t } = S _ { t } ^ { \top } q _ { t } .\tag{3}
$$

## 3.2. Reuse

Two requests with the same common prefix can share the computed cache for that prefix. Cache reuse can speed up request prefill by several times. For example, a 100k context request with 90% cached only requires processing the last 10k token suffix. Hence, most popular inference serving systems store the model’s cache for reuse across requests [3, 28, 53].

While the KV cache saves computation by reusing previous intermediate results, it can become expensive on memory to store it during long-context serving. Instead of storing a single contiguous tensor for each request’s KV cache, modern inference engines typically allocate the cache in smaller, fixed-size pages, as in PagedAttention [2], commonly also referred to as KV-cache blocks. A request containing � cached tokens and a block size of � tokens occupies approximately $\begin{array} { r } { N _ { \mathrm { b l o c k s } } ( n ) = \left\lceil \frac { n } { B } \right\rceil } \end{array}$ logical blocks, whose physical storage need not be contiguous. Paging allows blocks to be allocated on demand, reduces memory fragmentation, and enables block-level sharing between requests.

Prefix caching is the mechanism that allows requests to identify and reuse these blocks across requests. SGLang’s RadixAttention, for example, organizes shared token prefixes using a radix tree [3]. Other implementations, such as in vLLM, identify cached blocks using hashes derived from the tokens within a block together with the preceding prefix [53]. An implementation may therefore index token prefixes directly, index block-sized token sequences, or use a chained block hash.

While GPU memory is the fastest location for cached blocks, inference systems often extend this KV storage into CPU DRAM, local SSDs, or even remote memory. Mooncake, for example, uses CPU DRAM and SSD capacity as part of a distributed KV-cache hierarchy [29], while LMCache provides an external cache layer capable of moving KV state among GPU memory,

CPU memory, disk, and remote storage [16]. Such hierarchical caching increases the cache capacity, and transporting the cache can be much faster than recomputing it. The logical existence of a cached prefix, hence, differs from its physical residency in the system at any given moment.

Linear attention changes prefix reuse fundamentally. To resume a full-attention layer after token $r ,$ the runtime needs all KV cache blocks corresponding to the previous tokens. On the other hand, to resume a recurrent layer, the recurrent state after exactly that prefix, $S _ { r } ,$ is required. In general, a later state $S _ { r ^ { \prime } }$ for $r ^ { \prime } > r$ cannot be rewound to recover �<sub>�</sub> [14].

While a single recurrent state is constant with respect to sequence length, an individual state snapshot can be significantly large. Ignoring smaller auxiliary states, the storage required by one checkpoint is approximately

$$
M _ { \mathrm { s n a p s h o t } } \propto b \sum _ { \ell \in \mathcal { L } _ { \mathrm { l i n e a r } } } H _ { \ell } d _ { k , \ell } d _ { \nu , \ell } ,\tag{4}
$$

where � is the number of bytes per state element, $H _ { \ell }$ is the number of state heads, and $d _ { k , \ell }$ and $d _ { \nu , \ell }$ are the K, V dimensions. Hence, materializing a snapshot at every token or at frequent intervals such as KV block boundaries becomes infeasible for long context serving. Practical recurrent-state caches therefore store snapshots only at selected boundaries, such as sparse periodic intervals or request boundaries [14].

Models such as Kimi K3 contain MLA layers as well as linear attention layers. These heterogeneous state types can be managed in a single unified cache, or they can be managed independently. The former design allows a consistent view across caches, while the latter allows each cache to use retention and storage policies appropriate for its state type. Current vLLM coordinates multiple cache groups, including recurrent or Mamba and full-attention groups, through a common cache coordinator and shared block pool [30].

When these caches evolve independently, the reusable prefix boundaries may not be consistent, especially because the linear attention checkpoints are sparse. Thus, the reusable prefix boundary for a request must belong to the intersection of the reusable boundaries across the two caches. Figure 1 illustrates this sparsity on two identical prefix trees.

![](images/c40c6f37f8240707870928142fd2bb3129d58785e6405fdb20b93074420ae00e.jpg)  
Figure 1: Dense KV storage and sparse recurrent checkpoints on the same prefix tree. The recurrent cache retains four snapshots (�): one at the fork, two on branch A, and one at the end of branch B. Dashed nodes mark boundaries without a stored snapshot.

## 3.3. Disaggregated Serving

During the prefill phase of inference, the uncached input tokens are processed and the model materializes the state needed to continue generation. During the decode phase, the model

autoregressively processes the newest token while reading and updating the state created by the preceding prefix until a stopping condition is met, thereby generating the output.

Disaggregation allows these two phases to use different parallelism strategies and worker allocations, since their computation and memory requirements differ. The state produced by prefill is transferred to a decode worker before generation continues [33–35].

Our focus is on prefill workers in a dedicated disaggregated serving system that we run using pipeline parallelism. Our primary metrics are token throughput and prefill completion time (latency), as these are key factors in service level agreements (SLAs).

## 3.4. Parallelism Strategies

Suppose that the model has � layers and we execute on � GPUs. Figure 2 illustrates how TP and PP divide the model across them.

Under tensor parallelism (TP), each layer is divided among the � ranks. Each rank processes the same token batch and exchanges partial layer results through collective communication. This can lead to fast execution of individual layers, but requires collective communication between devices for each sharded layer, which also adds overhead [5].

Under pipeline parallelism (PP), the layers of the model are partitioned among � ordered stages. Stage � owns a contiguous fixed subset $L _ { s }$ of the model. Each stage processes its own assigned layers and on completion of the forward pass, it sends its output activations to stage � + 1. Different microbatches of requests can occupy different stages concurrently. Each PP stage owns the cache state belonging to its layers.

Since each stage must wait for the previous stage to finish processing a microbatch before it can start, the overall throughput is limited by the slowest stage. This can lead to underutilization of resources if the workload is not balanced across stages. Online serving further complicates the problem because the cost of processing a microbatch can vary significantly due to factors such as cache hits, prompt lengths, and chunk tails. Therefore, a PP scheduler must expose enough independent work to fill the stages while keeping the work in successive microbatches balanced.

PP can also enable more cache space when TP replicates caches across ranks. However, PP requires consistency in cache reuse across stages. The host-side and scheduler work increases as more requests are in-flight. Moreover, the system should be designed carefully to prevent gaps across the pipeline due to slow admission rates. Thus, despite the numerous advantages of PP, achieving consistent and efficient execution is challenging, especially in an online inference setting where requests arrive asynchronously and have varying lengths.

## 4. Design of WavePP

In this section we present the design of WavePP. We first describe the steps that must be taken before a request can execute, then explain how this work is overlapped with earlier requests. Finally, we describe how the scheduler divides the admitted work to keep the pipeline occupied.

Waves and lanes. After admission, a request’s uncached suffix is divided into contiguous chunks for processing as in chunked prefill. We call a microbatch of one or more request chunks that move through the pipeline together a wave. Each request keeps a logical place, or lane, across consecutive waves until all its chunks are scheduled.

![](images/7507b3b11078640f9d0be67e8c2c7febf40525d659221fdb874f078ed6a8161b.jpg)

![](images/299a0e9fad7470b9e4069c4011f4da855d7e65c540b1c149f682fe3e3b177c18.jpg)  
Figure 2: Two ways to divide an eight-layer model across four GPUs. TP splits each layer among the devices, combining partial results through collectives. PP assigns complete layer groups to successive stages and passes activations between them.

## 4.1. A Minimum Implementation

A straightforward implementation in our setting can establish consistent prefix reuse and allocate space for the remaining context through the following steps:

1. Receive and preprocess the request (such as tokenization), then transport it to every rank so each stage can prepare its local state.

2. Establish a reuse agreement across all cache types and ranks.

3. Pin the local cache for reuse, including recurrent snapshots, and allocate sufficient memory for the uncached suffix for the request.

4. Build the request execution metadata including materialization of the request’s uncached suffix blocks in the cache and references to cache memory.

5. Chunk and execute the requests using chunked prefill. Each stage receives activations from its predecessor, which are used for the current layer’s computation and then transferred to the successor.

6. Publish the completed cache state for reuse for future requests, and transfer the cache to a decode worker, or continue decoding locally.

In this implementation, every rank receives the request tokens to update its local cache tree. The ranks need a common reuse boundary so that each stage has the context required for the selected suffix. A simple implementation can serialize admission to keep requests from claiming or evicting each other’s cache. To admit several requests concurrently, it must protect both reusable state and suffix capacity. If requests pin most of the cache and then wait for more space, none may be able to finish without releasing state or being preempted. Each stage also needs the activations from the previous stage before computing, and cache publication makes the completed work reusable by future requests.

Furthermore, many of these steps require mutual exclusion, since concurrent modifications to the cache could lead to inconsistencies or race conditions. In a simple two-walk implementation, each rank first traverses its radix trees and communicates its maximum reuse endpoint. Once a common endpoint is established, the ranks traverse their trees again to validate and pin the cache for reuse, protecting those branches from eviction. The tree mutex serializes changes such as eviction and pinning, and prevents two requests from claiming the same free block or eviction victim. This protects the allocation itself, but does not ensure that enough capacity remains for every admitted request to finish.

However, under pressure of several long context requests, doing such a traversal becomes costly, and since ranks may diverge, a second walk post-communication to rectify the current cache lease adds even more cost. Under first-come, first-served scheduling this can be even more cumbersome when the request must requeue to access the radix tree mutex, or other requests must block on the first one completing its communications.

Another consideration is that we want waves of newer requests to schedule while previous waves are computing. Alongside decoupling reuse estimation from the main execution loop, we allow each rank to materialize its cache locally. This lets stage 0 begin once it is ready while later ranks still prepare the request.

## 4.2. Decoupling Admission

We first decouple admission from the main execution of the model. Each rank has two main threads. The WAVE thread transports requests and handles cache reuse agreement and capacity. The LOOP thread runs the executor, including the materialization of request cache blocks. Materialization uses the local allocator to claim blocks and build the sequence metadata, including radix-tree attachments. Rank 0 is the leader rank, and does the chunk planning on the executor thread.

We also split communication across two planes. The data plane carries wave schedules and activations across stages to support the model forward, while the rounds plane handles requests and admission results using a rank 0 broadcast followed by an all-gather. Communications rounds contain multiple requests at once to amortize overhead. Admission rounds run more frequently while waiting for another rank’s response, and repeat installation records only for requests still being admitted. Though these collectives may block the admission thread (WAVE), the LOOP thread can continue executing earlier admitted work. Data-plane sends and receives are asynchronous, allowing transfers to overlap work that does not depend on their completion. Each stage still waits for the inputs needed by its next forward pass.

Rank 0 receives the request, and announces it once it enqueues. The other ranks register this and prepare the request locally, which allows hashing and reuse negotiation to begin while the executors continue previous work. The cache agreement is separate from physically materializing the state in the tree. Before we schedule, every rank must only promise that it has pinned the agreed upon cache (lease), and claimed sufficient space for the uncached suffix (escrow), i.e., it does not need to add the blocks to the respective cache trees. Later, each rank completes its own materialization before the wave is executed. This means that stage 0 can execute the request once locally ready, without waiting for future ranks to complete installation. A later rank that is not ready when the wave arrives can still introduce a local only gap. On follower ranks, kernel launches can block while holding the Python interpreter lock. We therefore wait for a receive event with that lock released before enqueueing the next forward. The executor also yields when there are requests awaiting admission without runnable work.

Figure 3 illustrates this overlap.

a Naive: admission after pipeline drain  
![](images/c58f7056cfa567dab5663ecc3724d275b88d75a0004d1fcfe0f9510e87964963.jpg)  
Figure 3: A new request arrives while stage 0 executes C. The naive policy waits for the pipeline to drain. Lockstep admission pauses at a common scheduling boundary. WavePP overlaps preparation with earlier waves and materializes locally before E. Orange includes all CPU preparation, including materialization. Durations are schematic.

## 4.3. Fast Reuse Hints

The naive implementation first walks each cache tree to find reusable state, then walks it again after agreement to pin the common prefix. Both walks require a mutex due to concurrent modifications. We could instead pin blocks during the first walk, but each rank would claim its local prefix before knowing how much the other ranks can reuse. Ranks with more cached state would then need to release blocks beyond the common endpoint. For recurrent caches, lowering the endpoint may also require acquiring an earlier snapshot and releasing the one just claimed. Across many waiting requests, these early claims reduce what the cache can evict to make room for requests ready to execute. Similarly, reserving space for suffixes early would tie up more memory, including for requests that may wait or be cancelled. Claiming only for the request at the head of the queue avoids much of this pressure, but leaves reuse discovery and agreement to be done when that request is ready for admission.

In order to reduce the overhead of repeated traversals and mutex contention, we build an index that provides a reuse hint to each rank. This index does not contain any blocks and is only updated on cache tree changes. Queued requests can therefore discover reuse and agree on a candidate without pinning cache or reserving suffix capacity. Only requests selected for admission then perform the authoritative walk, validating and pinning no more than the candidate prefix. This lets candidate agreement proceed before the request reaches the head of the queue, while bounding the later walk on each rank.

The common hint also limits how far each rank must search. Consider a surge of 200K-token requests where most stages retain nearly the whole prefix, but one stage can resume only at 32K for one of the requests. If 32K is usable on every stage, other cache rich stages need to walk and pin only 32K tokens of the prefix. Requests outside the admission limit can still obtain hints without taking ownership. If they are cancelled while waiting, there are no cache references or capacity reservations to undo.

![](images/7a09cc8a74483ea0ebe808c7e65a012f13decc54a0ebb3c31b41dae4ee7043aa.jpg)  
Attach: count +1. Detach: count -1.

![](images/c0757b991d6919fcd64fe793670076c14df50ab68d1c5e53228af844310b3a41.jpg)  
Attention hint: 7 blocks. Recurrent hint: 6 blocks. Local candidate: 6 blocks.

Figure 4: The sharded hint index and an example lookup. Attention probes gallop and bisect to seven blocks. The reverse recurrent scan lowers the local candidate to six. The later lease validates and pins the actual state.

For each full token block $B _ { i } ,$ we compute a chained prefix hash

$$
H _ { - 1 } = 0 , \qquad H _ { i } = \mathrm { h a s h } ( B _ { i } , \mathrm { e x t r a K e y s } _ { i } , \mathrm { s a l t } , H _ { i - 1 } ) .\tag{5}
$$

The parent hash thus makes the key dependent on the prefix and the current block. Each rank maintains an index keyed by the cache window and $H _ { i } ,$ with a counter for each key. When a block is added to the tree, we increment its key’s counter, and once it is detached from the tree, we reduce the counter by one, evicting the index entry if the count reaches zero. Offloading a block to a different storage method does not affect this index since offloading still preserves reuse.

Reading the hint index still requires acquiring the mutex of each queried shard. For dense attention prefixes, we reduce the number of probes from �(�) to approximately �(log �) for � eligible blocks by probing depths at increasing powers of two until the first miss, then bisecting between the last match and miss. Request hashing still requires �(�) steps.

Recurrent snapshots are sparse, so we instead scan backward from the attention-limited endpoint. This scan can require �(�) probes of the index, but typically requires far fewer. We take the minimum of the required local cache endpoints, then the all-rank minimum, to obtain the reuse candidate. This is still a hint, since lowering the endpoint may require finding an earlier recurrent snapshot.

Note that the index can be changed during lookup as the lock is not constantly held, so an authoritative final walk is still required (next section). Hence, the index provides a useful, but occasionally stale, view of the cache state to quickly enable ranks to agree upon a candidate reuse endpoint.

In a 200K-token, concurrency-64 stress experiment, the optimized sharded lookup reduced median probe time from 64.4 ms to 24 �s, and hint discovery reached a p99 of 2.35 ms.

## 4.4. Local Leases

After a candidate reuse endpoint is agreed upon, the cache must be pinned so that it cannot be evicted while the request awaits materialization. A local lease maintains these references without physically installing the sequence (Figure 5). The admission thread builds the exact block keys outside the tree mutex, then walks the tree under mutex, and pins the required blocks based on reuse agreement. Other requests can also share the pinned state, but cannot evict it until the lease is released. The mutex is released before communicating the lease acquisition to reduce contention.

Once the lease is acquired, each rank reports the reuse endpoint it actually pinned. If endpoints differ, richer ranks trim their claims to a common endpoint before committing, or all ranks release the attempt and prepare again at the lower endpoint if any rank has already committed. In our regular experiments, we did not observe this discrepancy, but encountered it once under extreme load with frequent request cancellations when lease creation and abort churn was high.

For attention blocks, lowering the endpoint simply unpins the excess blocks, however, the next usable recurrent endpoint may once again diverge across the ranks. For example, suppose rank 0 has snapshots at 64K, 72K, and 90K, while rank 1 has them at 64K and 80K. At a candidate of 80K, rank 0 falls back to 72K. Rank 1 has no snapshot at 72K and must then fall back to 64K, which both ranks can use. Each adjustment acquires the earlier recurrent state before releasing the later one. Further reconciliation can only lower the endpoint. This process is bounded since every reconciliation monotonically decreases the claimed endpoint, ensuring eventual convergence.

Note that pinning reduces space for other requests to claim space for uncached suffixes. We therefore acquire leases in first-come, first-served (FCFS) ready order, and limit the number of requests that can hold leases simultaneously to prevent too many requests pinning cache and stalling the pipeline due to no eviction space. This is based on the number of admitted tokens required as well as remaining token capacity in the cache. We first cap the request count at the maximum batch size, since each wave may contain several requests. Secondly, we leave a token headroom proportional to 2��, where � is the number of pipeline stages, and � is the maximum number of tokens per wave. The exception to this rule is allowing a single oversized request to ignore the token headroom limit. This avoids blocking a single large request while still protecting the cache from being overwhelmed by many smaller requests.

The lease has no expiry based on time and is held through materialization until the sequence acquires its own references. This leaves no gap in ownership.

![](images/faa18d27408f9e57b367d0e7f8e93ac03a07466a74ea7e0fb91d92ea6c045f10.jpg)  
6 reserved blocks = 3 free + 2 offload victims + 1 eviction victim

Figure 5: The lease protects the reused prefix and its final recurrent snapshot. Six blocks of suffix capacity are backed by three free blocks, two offload victims with host destinations, and one eviction victim. The dashed suffix has not yet been allocated.

## 4.5. Capacity Escrow

A leased prefix guarantees reuse, but the uncached suffix also needs space. If we admit requests on prefix hits alone, they can reserve most of the cache but leave too little room for any of them to finish. We therefore reserve cache space for each layer (a capacity escrow) immediately after each local lease. If this fails, then the lease is released and the request waits until sufficient capacity is available.

The reservation is first prepared using the rank’s local lease endpoint. An escrow commit finalizes the reservation at a fixed reuse endpoint. Commit rechecks the required suffix capacity and the reserved victims, acquires any additional or replacement capacity, and marks the reservation as committed. It does not yet allocate the request’s blocks or evict the victims. Those operations remain part of local materialization.

Available capacity includes both free blocks and blocks that can be reclaimed by offloading or evicting cache. We follow the allocator’s existing eviction order. When we require offloading a victim, we reserve space for the destination before acquiring the primary block. The escrow holds concrete references to victim blocks, but leaves the contents attached to the tree without evicting or copying them. The actual eviction or offloading steps are deferred to the local executor during materialization, allowing the reservation work to overlap GPU execution. Recurrent placeholders are interchangeable and hence reserved by count rather than by holding individual snapshot destinations. Materialization uses the production allocator while the escrow remains held, rather than allocating directly from its reserved block list.

Since we leave victim blocks in the tree, other requests can still claim them for reuse. At commit, the escrow detects the additional owner through the block’s reference count and selects a replacement from the same cache window. It acquires the replacement, including any required host destination, before releasing its reference to the original victim. The claiming request keeps the original content alive, while the replacement backs the reserved suffix capacity. The admission limits and capacity floors retain headroom for active requests and restrict how much cache pending leases can pin. This headroom is intended to accommodate victim substitution while earlier requests finish. Commit rechecks the capacity floors before publishing the substitutions.

Committing a capacity escrow also recomputes demand if the reuse endpoint has reduced. For example, reducing reuse from 112K to 96K in a 200K-token prompt increases the uncached suffix from 88K to 104K. The rank must reserve additional capacity before committing the lower endpoint. The same capacity checks apply when replacing a victim. Commit gathers additional capacity and victim replacements under the tree mutex, then publishes the revised reservation only if all checks succeed. A failure releases only the new acquisitions, leaving the original reservation intact for distributed cleanup.

Commit can run immediately after preparation when the exact lease preserves the hint candidate. Each rank reports the result with its lease vote, avoiding a separate commit round if all ranks succeed at that endpoint. Once all ranks report successful commits at the same endpoint, rank 0 can schedule after its own materialization without another round of enqueue acknowledgements. If the preparation, commit, or enqueue fails, every rank releases the escrow first, then the lease. Otherwise, each rank holds its escrow and lease until its sequence owns the real cache blocks, then releases the escrow followed by the lease.

## 4.6. Pipeline Scheduling

Even when admission maintains a fast cadence, a purely greedy scheduling approach often leads to suboptimal pipeline utilization. Consider a set of requests with a combined 2� tokens, where � is the pipeline per-wave budget. If packed into full waves, this creates only two waves in total, and at any given time, at most two ranks process requests, leaving other ranks idle and the pipeline underutilized. Although many existing approaches for dynamically chunking requests exist, we adopt a very simple strategy by noticing that halving the budget in this case leads to 4 waves, as Figure 6 shows. Smaller waves also add overhead due to additional forward launches and activation transfers, so we only shrink the budget of waves when little admitted work remains. We use Maximum Pipeline Utilization (MPU), a variant of prefill throttling [12] tailored to our PP design. MPU decides the wave budget to keep the pipeline full, accounting for admitted work and chunks already in flight. Cache capacity is handled separately during admission.

The executor maintains an execution ring of $R = 2 P$ waves, allowing one wave to execute and another to wait at each stage. Rank 0 can therefore schedule further work before the result of every earlier wave returns. To keep this ring supplied, the planner targets $S = R + ( F - 1 ) P$ waves, where � − 1 adds waiting pipeline fills. The matched PP4 experiments use $F = 2$

The planner chooses from a sequence that repeatedly halves the maximum wave budget � down to a specified floor $b _ { \mathrm { m i n } } \mathrm { . }$

$$
b _ { 0 } = M , \qquad b _ { j + 1 } = \operatorname* { m a x } \left( b _ { \mathrm { m i n } } , \mathrm { a l i g n D o w n } _ { B } ( b _ { j } / 2 ) \right) .\tag{6}
$$

Here � is the block size, and � and $b _ { \mathrm { m i n } }$ are block-aligned with $B \leq b _ { \operatorname* { m i n } } \leq M$ . We stop halving at $b _ { \mathrm { m i n } }$ . Let � include the not-yet-scheduled suffix tokens of admitted requests and their most recently committed chunks still occupying the ring. Counting this in-flight work avoids shrinking the budget simply because earlier chunks have left the planner.

a Fixed budget M: two larger waves  
![](images/8f98fd4fe0fc997e2a068a4991e2416b27dc19ff6d0b5ba94a59d0c95cdbcc2e.jpg)

b MPU budget M/2: four smaller waves  
![](images/525f0521e7877fa98932f2cad5abb4ba640e731eb84347574f1cb3613131b837.jpg)  
Figure 6: Wave sizing with the same 2� tokens on four stages. A budget of � gives two waves, while $M / 2$ gives four. Stage time is assumed proportional to token count and fixed overheads are omitted, so this illustrates the scheduling effect rather than a measured speedup.

Token count alone can underestimate how many waves are available. Two request pieces larger than half a wave cannot share that wave. For a candidate budget $b ,$ let $N _ { > b / 2 }$ count such pieces among the remaining suffixes and in-flight chunks. We estimate the available waves as

$$
n ( b ) = \mathrm { m a x } \big ( W / b , N _ { > b / 2 } \big ) .\tag{7}
$$

We select the largest budget with $n ( b ) \geq S ,$ falling back to $b _ { \mathrm { m i n } }$ if none qualifies. Once a larger budget already supplies � waves, further thinning is limited when it would mainly split requests that fit the larger budget. In particular, pieces that would be newly split must account for at most 12.5% of �. This avoids adding forward launches and transfers just to create more chunks of the same requests.

Large bursts can increase the budget immediately. A one-step increase must persist for $R / 2$ planning decisions, while a decrease must persist for � decisions and pass a hysteresis check. The number of lanes depends on the maximum batch size, so changing the wave budget does not remove an existing request’s lane. During packing, we also avoid filling a small leftover space with a fragment of a larger request when that fragment is smaller than max $\left( b _ { \mathrm { m i n } } , b / 2 \right)$ . The matched PP4 comparison uses � = 16,384 and $b _ { \mathrm { m i n } } = 8 , 1 9 2$ . The Kimi K3 scheduling settings are given in Section 5.3.

Rank 0 is responsible for packing the requests in FCFS order. While the ring is busy, a lone wave below half capacity may wait one iteration for another request in admission so they can share a wave. The scheduler still checks resources and may accept less work, so the planner advances its token positions only by the spans actually accepted. Followers cannot infer the selected chunks from their local cache views. Therefore, rank 0 sends each scheduled chunk’s request ID, starting token, and length to other ranks. Followers apply those spans after local materialization while preserving both request order and token boundaries. If a position mismatch remains after materialization, execution fails rather than processing activations against the wrong cache state. This keeps scheduling consistent, while cache management proceeds independently across all stages.

Successive context chunks can enter while earlier chunks remain in downstream stages. Their order is preserved on each stage’s execution stream.

## 5. Evaluation

We evaluate WavePP on cold prefill, requests with a shared cached prefix, and mixtures of the two. We first measure how WavePP improves prefill throughput within TensorRT-LLM PP4 on GLM 5.2 and MiniMax M2.7, using the same kernels as concurrency and cache pressure increase. We then compare PP8 and TP8 across TRT-LLM, SGLang, and vLLM on 28 Kimi K3 workload and concurrency settings. Finally, we examine prefix retention and admission and scheduling ablations.

## 5.1. Metrics

We report aggregate prompt throughput, measured as the total input tokens of completed requests divided by the run’s wall time. This includes cached tokens, so it measures the input served rather than only the tokens computed. We use � for request concurrency. We also report prefill completion time, the time taken to complete the entire prefill of a request, including admission and queueing. For the Kimi K3 comparisons, we report p50 and p95 across requests.

Table conventions. Green shading and an underline mark the best overall value. Underlining alone marks the second-best PP value and the best TP value, or the second-best TP value when TP has the best overall value. Ties at the reported precision share the same marking.

In the Kimi K3 tables, $\Delta _ { \mathrm { T P } }$ and Δ compare WavePP with the best TP and external PP result, respectively, for each metric. Both use (WavePP/baseline − 1) × 100%. Positive throughput deltas indicate higher throughput for WavePP, and negative latency deltas indicate lower completion time. Latency deltas use the rounded times shown in the tables. Retention differences are in percentage points.

## 5.2. Pipeline Parallelism in TensorRT-LLM

We first examine how WavePP’s admission and scheduling changes compare to TensorRT-LLM’s original PP runtime. We compare TensorRT-LLM PP4 and WavePP4 on GLM 5.2 [54, 55] and MiniMax M2.7 [56, 57], using the same TensorRT-LLM 1.3.0rc26 kernels and model configurations. This measures the combined runtime changes within a four-stage pipeline.

Setup. Both models use NVFP4 weights and four B300 GPUs per system. We tune the TensorRT-LLM PP4 configuration to a 16,384-token microbatch limit and use the same maximum wave budget for WavePP. Both systems allow up to 128 requests per batch. WavePP uses its standard configuration, including the 2� execution ring, $F = 2 ,$ , and an 8,192-token minimum wave budget. Each system keeps one configuration across all workloads, with diagnostic instrumentation disabled. We report medians from three independent server boots per system and model, with the runs interleaved on the same node. The rest of the configuration is identical across systems including the pipeline partition.

Cold workloads run first within each boot. Subsequent reuse workloads therefore encounter a cache pool that has already filled and evicted earlier prompts. This tests admission during continued use of the cache. We include short and long cold inputs, 90% prefix reuse at both lengths, and mixed traffic with 75% warm short requests and 25% cold long requests. The recorded short/long lengths are approximately 128K/260K for GLM and 129K/195K for MiniMax. A further workload uses a 4,096-token uncached suffix behind a shared prefix of about 125K tokens and extends concurrency to 128. Throughput counts the full input, including the cached prefix. Table 1 reports all 40 settings. The goal of the short suffix workload is to stress the runtime’s admission mechanisms under high cache pressure and rapid request turnover.

![](images/7cc98566047c54129bba5afd4f2fae093d30de0fcfa1118ab9121b635dcf9495.jpg)  
Figure 7: Prefill throughput with TensorRT-LLM PP4 and WavePP4 on the same kernels. Cold and mixed workloads use short/long inputs of approximately 128K/260K tokens for GLM and 129K/195K for MiniMax. Mixed traffic is 75% warm short requests and 25% cold long requests. Points are median aggregate prompt throughput across three server boots.

Throughput under sustained load. The admission and scheduling changes improve throughput relative to TensorRT-LLM PP4 in 37 of 40 settings, including every cold and mixed setting. On cold and mixed traffic, WavePP improves throughput by 1.9–9.1% for GLM and 7.5–10.7% for MiniMax (Figure 7). With 90% reuse of the shorter inputs, WavePP improves throughput at � = 16 and � = 32 by 11.5% and 17.8% for GLM, and 9.1% and 10.4% for MiniMax.

The largest throughput improvements appear when many requests reuse a long prefix and compute only a short suffix (Figure 8). These requests complete quickly, increasing the rate at which the runtime must prepare and schedule new work. At � = 64, WavePP improves throughput by 39.0% on GLM and 24.8% on MiniMax. At � = 128, it sustains 1,212,572 and

1,113,196 input tok $\mathbf { \nabla } \cdot / \mathbf { s } ,$ compared with 416,294 and 551,143 for TensorRT-LLM PP4. WavePP provides 2.91× and 2.02× the throughput of TensorRT-LLM PP4 after the cache has been filled by earlier cold traffic.

![](images/ddd0245a2e0f7e048a91ad3dab179849e11ca3ce73172d254517ab665d07ea71.jpg)  
Figure 8: Reuse throughput with the same kernels and PP4 topology. The first two columns reuse 90% of the short and long inputs. The last column uses a 4,096-token suffix and extends to � = 128. Reuse follows cold workloads within each server boot, so these measurements include cache pressure from earlier traffic. Throughput includes cached input tokens.

For the 4K-suffix workload at $c = 1 2 8 .$ , median prefill completion time also falls by approximately 71% on GLM and 55% on MiniMax. However, throughput is lower in three settings: long-prefix reuse at $c = 8$ for both models and the 4K-suffix MiniMax workload at $c = 1 6 ,$ as shown in the figure and table.

The smaller throughput improvements on cold inputs in Figure 7 are consistent with the greater amount of GPU computation per request. Each admission supplies many chunks, giving both runtimes more work to execute before another request must be prepared. Reducing admission and scheduling delays therefore affects a smaller fraction of the total processing time.

The 4K-suffix results show a different trend. From � = 32 to $c = 1 2 8$ , TensorRT-LLM PP4 throughput decreases for both models, while WavePP throughput increases (Table 1). More concurrent requests compete for space in the filled cache pool, while each admission supplies little computation. Preparing reuse and reserving suffix capacity during earlier execution helps WavePP keep new requests ready as others finish. The ablation in Section 5.9 separately examines admission and scheduling on Kimi K3.

Table 1: PP4 comparison on matched kernels. Aggregate prompt throughput (tokens/s, including cached tokens) is the median of three boots per system and model. Each system uses one configuration throughout. Throughput change Δ is (WavePP/TRT-LLM − 1) × 100%. Green shading marks the higher throughput. Short and long reuse use 90% cached prefixes. The 4K-suffix cells follow cold traffic and retain its cache-pool eviction history.
<table><tr><td colspan="2"></td><td colspan="3">GLM 5.2</td><td colspan="3">MiniMax M2.7</td></tr><tr><td>Family</td><td>C</td><td>TRT-LLM</td><td>WavePP</td><td>∆</td><td>TRT-LLM</td><td>WavePP</td><td>Δ</td></tr><tr><td>Short cold</td><td>8</td><td>46,658</td><td>47,552</td><td>+1.9%</td><td>50,677</td><td>54,457</td><td>+7.5%</td></tr><tr><td></td><td>16</td><td>46,207</td><td>48,071</td><td>+4.0%</td><td>50,127</td><td>54,642</td><td>+9.0%</td></tr><tr><td></td><td>32</td><td>45,730</td><td>47,864</td><td>+4.7%</td><td>49,596</td><td>54,615</td><td>+10.1%</td></tr><tr><td>Long cold</td><td>8</td><td>41,839</td><td>43,456</td><td>+3.9%</td><td>37,146</td><td>40,388</td><td>+8.7%</td></tr><tr><td></td><td>16</td><td>41,610</td><td>43,310</td><td>+4.1%</td><td>37,051</td><td>40,363</td><td>+8.9%</td></tr><tr><td></td><td>32</td><td>40,844</td><td>43,357</td><td>+6.2%</td><td>36,789</td><td>40,165</td><td>+9.2%</td></tr><tr><td>Short reuse</td><td>8</td><td>309,301</td><td>309,762</td><td>+0.1%</td><td>283,199</td><td>303,921</td><td>+7.3%</td></tr><tr><td></td><td>16</td><td>351,119</td><td>391,338</td><td>+11.5%</td><td>344,174</td><td>375,368</td><td>+9.1%</td></tr><tr><td></td><td>32</td><td>321,750</td><td>378,951</td><td>+17.8%</td><td>315,446</td><td>348,107</td><td>+10.4%</td></tr><tr><td>Long reuse</td><td>8</td><td>316,516</td><td>278,418</td><td>-12.0%</td><td>250,815</td><td>206,548</td><td>-17.6%</td></tr><tr><td></td><td>16</td><td>297,822</td><td>310,786</td><td>+4.4%</td><td>239,657</td><td>241,150</td><td>+0.6%</td></tr><tr><td></td><td>32</td><td>264,520</td><td>306,497</td><td>+15.9%</td><td>227,220</td><td>236,459</td><td>+4.1%</td></tr><tr><td>Mixed</td><td>8</td><td>89,844</td><td>92,759</td><td>+3.2%</td><td>91,270</td><td>99,810</td><td>+9.4%</td></tr><tr><td></td><td>16</td><td>87,974</td><td>93,515</td><td>+6.3%</td><td>91,314</td><td>100,117</td><td>+9.6%</td></tr><tr><td></td><td>32</td><td>84,387</td><td>92,080</td><td>+9.1%</td><td>89,427</td><td>99,039</td><td>+10.7%</td></tr><tr><td>4K suffix</td><td>8</td><td>497,230</td><td>503,184</td><td>+1.2%</td><td>435,088</td><td>441,314</td><td>+1.4%</td></tr><tr><td></td><td>16</td><td>658,662</td><td>661,230</td><td>+0.4%</td><td>642,896</td><td>606,764</td><td>-5.6%</td></tr><tr><td></td><td>32</td><td>884,810</td><td>895,482</td><td>+1.2%</td><td>872,496</td><td>886,305</td><td>+1.6%</td></tr><tr><td></td><td>64</td><td>835,235</td><td>1,160,931</td><td>+39.0%</td><td>832,645</td><td>1,039,376</td><td>+24.8%</td></tr><tr><td></td><td>128</td><td>416,294</td><td>1,212,572</td><td>+191.3%</td><td>551,143</td><td>1,113,196</td><td>+102.0%</td></tr></table>

## 5.3. Kimi K3 Experimental Setup

Model and hardware. We next evaluate WavePP across serving libraries on Kimi K3, a 2.8- trillion-parameter hybrid KDA/MLA MoE model with both recurrent state and attention caches [52]. These cross-library experiments use the same eight GB300 GPUs arranged as two four-GPU compute nodes. WavePP and the pipeline baselines use TP1×PP8 and the tensor baselines use TP8/EP8.

Systems. WavePP is implemented in TensorRT-LLM [32]. We compare it with TRT-LLM TP8/EP8, and with the TP8/EP8 and PP8 implementations in SGLang and vLLM (Table 2). WavePP and the TRT-LLM TP8/EP8 reference share the same Kimi K3 kernels and library. Both SGLang configurations use v0.5.18 at commit 71de97b264b0 and the same hybrid host-cache configuration [22].

WavePP reserves 128 GiB of host cache per rank. Its scheduler uses a 16,384-token limit for both total work and context chunks. SGLang uses a 16K prefill budget, and vLLM PP8 uses a 16K packed-token budget (Table 2). For Kimi K3, MPU uses the remaining unscheduled tokens to choose between two budgets, � and �/2, targeting one pipeline fill.

As a separate baseline check, our SGLang PP8 run with 8K cold inputs reaches approximately 6,300 input tokens/s per GPU on two four-GPU GB300 nodes. The SGLang day-0 report highlights 5,958 input tokens/s per GPU for 8K prefill at � = 64 on the same topology [58]. Our result is approximately 5.7% higher.

Table 2: Configurations of the Kimi K3 comparison baselines.
<table><tr><td>Baseline</td><td>Parallel configuration</td><td>Benchmark configuration</td></tr><tr><td>vLLM TP8/EP8</td><td>TP8/EP8 on eight GPUs</td><td>v0.28.0 with the FlashInfer TensorRT-LLM MoE path</td></tr><tr><td>SGLang TP8/EP8</td><td>TP8/EP8 on eight GPUs</td><td>Same SGLang build, 16K prefill budget, and hybrid host-cache con- figuration as the SGLang PP8 arm</td></tr><tr><td>TRT-LLM TP8/EP8</td><td>TP8/EP8 on eight GPUs</td><td>Same Kimi K3 engine family and shared kernel changes as WavePP; valid two-node tensor-parallel placement</td></tr><tr><td>vLLM PP8</td><td>Eight-stage pipeline parallelism</td><td>v0.28.0 with a 16K packed-token budget and prefix caching enabled</td></tr><tr><td>SGLang PP8</td><td>Eight-stage pipeline parallelism</td><td>16K prefill chunks; 128 GiB/rank hybrid MLA+Mamba host cache; write-through offload, kernel I/O, and page-first loading</td></tr></table>

Workloads. Table 3 lists the workloads and request concurrency �. We use 65.5K, 131K, and 262K for 65,536, 131,072, and 262,144 input tokens, respectively. Seeded-reuse experiments first send standalone prefix requests to populate the cache. Mixed traffic contains 75% warm 131K requests and 25% cold 262K requests by request count. For 262K inputs, vLLM PP8 uses 262,080 tokens to leave room for the output token within its context limit, while the other systems use 262,144. Throughput uses the actual input token count.

All systems receive the same request sequence, subject to the context-length adjustment above. Request bodies are prepared before timing, and all requests in these Kimi K3 experiments completed successfully.

Table 3: Kimi K3 workloads and offered concurrency. Input lengths are abbreviated in decimal thousands; the vLLM PP8 262K exception is described in the text.
<table><tr><td>Family</td><td>Input tokens</td><td></td><td>Shared prefix Request composition</td><td>Concurrency</td></tr><tr><td>Cold 65.5K</td><td>65,536</td><td>0% cold</td><td></td><td>1, 4, 8, 16, 32</td></tr><tr><td>Cold 131K</td><td>131,072</td><td>0% cold</td><td></td><td>1, 4, 8, 16, 32</td></tr><tr><td>Cold 262K</td><td>262,144</td><td>0% cold</td><td></td><td>1, 4, 8, 16, 32</td></tr><tr><td>Reuse 131K</td><td>131,072</td><td></td><td>90% seeded shared prefix</td><td>1, 4, 8, 16, 32</td></tr><tr><td>Reuse 262K</td><td>262,144</td><td></td><td>90% seeded shared prefix</td><td>1, 4, 8, 16, 32</td></tr><tr><td>Mixed</td><td>一</td><td></td><td>75/25 warm-131K/cold-262K</td><td>8,16,32</td></tr></table>

## 5.4. Kimi K3 Overall Performance

WavePP has the highest measured throughput in 21 of the 28 settings (Figure 9), including all 18 settings at � ≥ 8. WavePP has lower throughput than some baselines in a few low concurrency settings when there is less independent work to fill the pipeline, though these gaps are also present for other PP methods as well.

For Kimi K3, the PP topology appears more favorable for prefill than TP. For example, SGLang PP8 improves geometric-mean throughput by 26.8% over SGLang TP8/EP8. WavePP further improves geometric-mean throughput by 9.2% over SGLang PP8 and 15.7% over vLLM PP8 across the full grid.

Kimi K3 throughput comparison across 28 settings TP / PP deltas: WavePP vs. the best TP8/EP8 / external PP8 result.  
![](images/6ede6c4a0680d51cae2828d7730ef9ba78241d26147521a6c9736f7794e7c0bb.jpg)  
Figure 9: The system with the highest measured aggregate throughput in each Kimi K3 cell. Each tile names the library with the highest throughput and reports WavePP’s throughput difference relative to the highestthroughput TP8/EP8 and external PP8 results in that cell. Positive values indicate higher throughput for WavePP. Color identifies only the library with the highest measured throughput.

## 5.5. Cold Long-Context Prefill

Cold requests compute the full input without reusing earlier cache state. WavePP improves geometric-mean throughput over TRT-LLM TP8/EP8 by 58.8%, 49.1%, and 37.8% for 65.5K, 131K, and 262K inputs, respectively. At � = 32, it sustains 44,643, 35,878, and 28,031 input tok/s.

The differences between PP implementations are smaller on these cold inputs, which require more GPU computation per request. Across the same three input lengths, WavePP improves geometric-mean throughput by 2.2%, 4.2%, and 8.0% over vLLM PP8, and by 9.4%, 4.5%, and 1.9% over SGLang PP8. WavePP benefits most at higher concurrencies. At � = 1, WavePP’s throughput is below the best baseline result at every input length, and its throughput remains below vLLM PP8 for cold 65.5K at � = 4.

![](images/c8034f385d9d9bdb41e80dda3505ff505f1978cbd3de907fc4f2f793dc09fcbe.jpg)  
Figure 10: Cold long-context performance. The top row shows throughput by concurrency. The bottom row shows throughput p50 completion-time operating points. The arrow in each operating-point panel points toward higher throughput and lower completion time.

Table 4: Cold aggregate prompt throughput (input tok/s). Positive deltas indicate higher throughput for WavePP.
<table><tr><td>Context, c</td><td>vLLM TP8/EP8</td><td>SGLang TP8/EP8</td><td>TRT-LLM TP8/EP8</td><td>vLLM PP8</td><td>SGLang PP8</td><td>WavePP</td><td>ΔTP</td><td>∆PP</td></tr><tr><td>65.5K, 1</td><td>23,330</td><td>20,378</td><td>22,976</td><td>20,361</td><td>16,691</td><td>22,089</td><td>-5.3%</td><td>+8.5%</td></tr><tr><td>65.5K, 4</td><td>23,922</td><td>22,215</td><td>23,524</td><td>41,554</td><td>39,926</td><td>37,467</td><td>+56.6%</td><td>-9.8%</td></tr><tr><td>65.5K, 8</td><td>23,917</td><td>22,194</td><td>23,490</td><td>41,980</td><td>40,458</td><td>43,007</td><td>+79.8%</td><td>+2.4%</td></tr><tr><td>65.5K, 16</td><td>23,925</td><td>22,215</td><td>23,537</td><td>42,242</td><td>40,852</td><td>44,746</td><td>+87.0%</td><td>+5.9%</td></tr><tr><td>65.5K, 32</td><td>23,925</td><td>22,213</td><td>23,536</td><td>42,456</td><td>41,180</td><td>44,643</td><td>+86.6%</td><td>+5.2%</td></tr><tr><td>131K,1</td><td>21,430</td><td>19,810</td><td>21,380</td><td>23,030</td><td>21,014</td><td>20,699</td><td>-3.4%</td><td>-10.1%</td></tr><tr><td>131K,4</td><td>22,059</td><td>20,673</td><td>21,619</td><td>33,089</td><td>33,662</td><td>35,765</td><td>+62.1%</td><td>+6.2%</td></tr><tr><td>131K, 8</td><td>22,058</td><td>20,662</td><td>21,554</td><td>33,242</td><td>33,873</td><td>36,030</td><td>+63.3%</td><td>+6.4%</td></tr><tr><td>131K, 16</td><td>22,052</td><td>20,670</td><td>21,677</td><td>33,322</td><td>33,984</td><td>36,129</td><td>+63.8%</td><td>+6.3%</td></tr><tr><td>131K, 32</td><td>22,070</td><td>20,669</td><td>21,691</td><td>33,410</td><td>34,102</td><td>35,878</td><td>+62.6%</td><td>+5.2%</td></tr><tr><td>262K,1</td><td>18,772</td><td>17,785</td><td>18,938</td><td>21,479</td><td>21,754</td><td>21,065</td><td>+11.2%</td><td>-3.2%</td></tr><tr><td>262K, 4</td><td>18,989</td><td>18,138</td><td>19,220</td><td>25,224</td><td>27,041</td><td>27,924</td><td>+45.3%</td><td>+3.3%</td></tr><tr><td>262K,8</td><td>18,987</td><td>18,133</td><td>19,230</td><td>25,271</td><td>27,110</td><td>27,988</td><td>+45.5%</td><td>+3.2%</td></tr><tr><td>262K, 16</td><td>18,990</td><td>18,138</td><td>19,249</td><td>25,296</td><td>27,144</td><td>27,971</td><td>+45.3%</td><td>+3.0%</td></tr><tr><td>262K, 32</td><td>18,995</td><td>18,118</td><td>19,254</td><td>25,321</td><td>27,179</td><td>28,031</td><td>+45.6%</td><td>+3.1%</td></tr></table>

Table 5: Queue-inclusive completion time for cold prefill, in seconds, with WavePP deltas against the lowest-latency TP8/EP8 and PP8 results in each cell. Negative deltas indicate lower completion time for WavePP.
<table><tr><td></td><td colspan="2">vLLM TP8/EP8</td><td colspan="2">SGLang TP8/EP8</td><td colspan="2">TRT-LLM TP8/EP8</td><td colspan="2">vLLM PP8</td><td colspan="2">SGLang PP8</td><td colspan="2">WavePP</td><td colspan="2"> $\Delta _ { \mathrm { T P } }$ </td><td colspan="2"> $\Delta _ { \mathrm { P P } }$ </td></tr><tr><td>Context, c</td><td>p50</td><td>p95</td><td>p50</td><td>p95</td><td>p50</td><td>p95|</td><td>p50</td><td>p95</td><td>p50</td><td>p95</td><td>p50</td><td>p95</td><td>p50</td><td>p95</td><td>p50</td><td>p95</td></tr><tr><td>65.5K, 1</td><td>2.8</td><td>2.8</td><td>3.2</td><td>3.2</td><td>2.9</td><td>2.9</td><td>3.2</td><td>3.2</td><td>3.9</td><td>3.9</td><td>3.0</td><td>3.0+7.1%</td><td></td><td>+7.1%</td><td>-6.2%</td><td>-6.2%</td></tr><tr><td>65.5K, 4</td><td>10.8</td><td>11.5</td><td>11.8</td><td>11.8</td><td>11.1</td><td>11.1</td><td>6.1</td><td>6.1</td><td>6.3</td><td>6.3</td><td>6.8</td><td>7.1 -37.0%</td><td></td><td>-36.0% +11.5% +16.4%</td><td></td><td></td></tr><tr><td>65.5K, 8</td><td>21.6</td><td>22.3</td><td>23.5</td><td>23.8</td><td>22.3</td><td>22.5</td><td>12.2</td><td>12.2</td><td>12.6</td><td>12.6</td><td>11.8</td><td>12.4 -45.4%</td><td></td><td>-44.4%</td><td>-3.3%</td><td>+1.6%</td></tr><tr><td>65.5K, 16</td><td>43.9</td><td>44.0</td><td>47.1</td><td>47.3</td><td>44.5</td><td>44.5</td><td>24.4</td><td>24.4</td><td>25.1</td><td>25.1</td><td>22.8</td><td>23.2</td><td>-48.1% -47.3%</td><td></td><td>-6.6%</td><td>-4.9%</td></tr><tr><td>65.5K, 32</td><td>87.3</td><td>87.9</td><td>94.4</td><td>94.4</td><td>89.0</td><td>89.1</td><td>48.8</td><td>48.8</td><td>50.1</td><td>50.2</td><td>45.4</td><td>45.9</td><td>-48.0%</td><td>-47.8%</td><td>-7.0%</td><td>-5.9%</td></tr><tr><td>131K, 1</td><td>6.1</td><td>6.2</td><td>6.6</td><td>6.6</td><td>6.1</td><td>6.2</td><td>5.7</td><td>5.7</td><td>6.2</td><td>6.2</td><td>6.3</td><td>6.4 +3.3%</td><td></td><td>+3.2% +10.5%+12.3%</td><td></td><td></td></tr><tr><td>131K, 4</td><td>24.0</td><td>24.1</td><td>25.3</td><td>25.3</td><td>24.2</td><td>24.2</td><td>15.6</td><td>15.6</td><td>15.3</td><td>15.3</td><td>14.2</td><td>14.8</td><td>-40.8% -38.6%</td><td></td><td>-7.2%</td><td>-3.3%</td></tr><tr><td>131K, 8</td><td>47.4</td><td>48.1</td><td>50.7</td><td>50.9</td><td>48.4</td><td>49.8</td><td>31.2</td><td>31.3</td><td>30.6</td><td>30.6</td><td>28.5</td><td>29.1</td><td>-39.9%-39.5%</td><td></td><td>-6.9%</td><td>-4.9%</td></tr><tr><td>131K, 16</td><td>94.8</td><td>95.5 101.3</td><td></td><td>101.6</td><td>96.7</td><td>96.8</td><td>62.4</td><td>62.5</td><td>61.1</td><td>61.1</td><td>56.8</td><td>57.6</td><td>-40.1%-39.7%</td><td></td><td>-7.0%</td><td>-5.7%</td></tr><tr><td>131K, 32</td><td>189.6</td><td>190.3 202.9</td><td></td><td>202.9193.2 193.5</td><td></td><td></td><td></td><td>124.9 124.9 122.2</td><td></td><td>122.2</td><td>114.4</td><td>116.3</td><td>-39.7%-38.9%</td><td></td><td>-6.4%</td><td>-4.8%</td></tr><tr><td>262K, 1</td><td>14.0</td><td>14.0</td><td>14.7</td><td>14.7</td><td>13.8</td><td>14.0</td><td>12.2</td><td>12.2</td><td>12.1</td><td>12.1</td><td>12.4</td><td>12.8 -10.1%</td><td></td><td>-8.6%</td><td>+2.5%</td><td>+5.8%</td></tr><tr><td>262K, 4</td><td>55.2</td><td>55.2</td><td>57.8</td><td>57.8</td><td>54.6</td><td>54.6</td><td>41.3</td><td>41.3</td><td>38.5</td><td>38.5</td><td>37.1</td><td>37.7 -32.1%</td><td>-31.0%</td><td></td><td>-3.6%</td><td>-2.1%</td></tr><tr><td>262K, 8</td><td>110.4</td><td>110.5 115.6</td><td></td><td>115.8 109.0</td><td></td><td>109.1</td><td>82.6</td><td>82.7</td><td>77.0</td><td>77.0</td><td>74.2</td><td>74.8</td><td>-31.9%-31.4%</td><td></td><td>-3.6%</td><td>-2.9%</td></tr><tr><td>262K, 16</td><td>220.8220.9 231.1</td><td></td><td></td><td>231.4 217.7</td><td></td><td>218.2</td><td></td><td>165.3 165.3 153.9</td><td></td><td>153.9</td><td>149.0</td><td>149.7</td><td>-31.6%-31.4%</td><td></td><td>-3.2%</td><td>-2.7%</td></tr><tr><td>262K, 32</td><td>441.0441.7 462.5</td><td></td><td></td><td>462.7</td><td>435.4</td><td>435.9</td><td></td><td>330.5 330.6 307.8</td><td></td><td>307.9</td><td>297.8</td><td>298.9</td><td>-31.6%-31.4%</td><td></td><td>-3.2%</td><td>-2.9%</td></tr></table>

The higher throughput at $c \geq 8$ is accompanied by lower completion time. WavePP’s p50 is 31.6 to 49.0% lower than TRT-LLM TP8/EP8 and lower than both PP baselines in every such setting. Its p95 is also lower than both PP baselines, except for cold 65.5K at $c = 8 ,$ where it is close to vLLM PP8 (1.6% higher).

## 5.6. Seeded Prefix Reuse

We next consider requests with 90% of the input cached. Each request then has much less prefill work left. This makes the dependence on concurrency more pronounced. Short suffixes provide few chunks per request, so several concurrent requests are needed to keep all stages busy. At � = 1, all PP baselines have lower throughput than the TP baselines as most of the pipeline remains idle. Nevertheless, WavePP has the highest throughput among PP methods for both 131K and 262K requests, by about 31% over vLLM PP8. Its throughput is still 63.3% and 51.4% below vLLM TP8/EP8, respectively. $\mathrm { A t } \mathrm { ~ } c = 4$ , WavePP’s throughput remains below the TP baselines for 131K inputs, but for 262K inputs it is 15.3% higher than TRT-LLM TP8/EP8 and 25.8% higher than SGLang PP8. From � = 8 onward, it has the highest throughput at both input lengths. Thus, the amount of work remaining after reuse is very important for pipeline occupancy.

For 131K reuse at $c \geq 8 ,$ WavePP improves throughput by 36.8 to 55.4% over TRT-LLM TP8/EP8. $\mathrm { A t } c = 3 2 ,$ , it reaches 289,096 input tok/s with p50/p95 completion times of 13.2/17.0 s. Across all five concurrencies, including the low-concurrency settings, its geometric-mean throughput is 4.1% below TRT-LLM TP8/EP8 but 11.6% and 24.9% above SGLang PP8 and vLLM PP8.

For 262K reuse, WavePP improves geometric-mean throughput by 11.1% over TRT-LLM TP8/EP8, 22.2% over SGLang PP8, and 45.3% over vLLM PP8. At � = 32, it serves 218,743 input tok/s with p50/p95 completion times of 35.8/39.7 s, compared with 184,753 input tok/s and 44.1/44.9 s for SGLang PP8.

![](images/3caedc45e6897077010908e9da6f35f953f077931e422f606257b74f104c5fad.jpg)  
Figure 11: Seeded reuse with a 90% shared prefix. The top row shows throughput by concurrency. The bottom row shows throughput p50 operating points. The arrow in each operating-point panel points toward higher throughput and lower completion time.

Table 6: Seeded-reuse aggregate prompt throughput (input tok/s). Positive deltas indicate higher throughput for WavePP.
<table><tr><td>Context, c</td><td>vLLM TP8/EP8</td><td>SGLang TP8/EP8</td><td>TRT-LLM TP8/EP8</td><td>vLLM PP8</td><td>SGLang PP8</td><td>WavePP</td><td> $\Delta _ { \mathrm { T P } }$ </td><td>ΔPP</td></tr><tr><td>131K, 1</td><td>155,002</td><td>127,746</td><td>142,607</td><td>43,197</td><td>41,022</td><td>56,810</td><td>-63.3%</td><td>+31.5%</td></tr><tr><td>131K, 4</td><td>167,165</td><td>172,924</td><td>183,856</td><td>130,420</td><td>134,161</td><td>124,682</td><td>-32.2%</td><td>-7.1%</td></tr><tr><td>131K, 8</td><td>173,271</td><td>173,266</td><td>186,690</td><td>171,714</td><td>228,012</td><td>255,406</td><td>+36.8%</td><td>+12.0%</td></tr><tr><td>131K, 16</td><td>173,510</td><td>172,581</td><td>185,931</td><td>212,477</td><td>249,227</td><td>262,740</td><td>+41.3%</td><td>+5.4%</td></tr><tr><td>131K, 32</td><td>173,780</td><td>172,847</td><td>186,064</td><td>219,649</td><td>253,844</td><td>289,096</td><td>+55.4%</td><td>+13.9%</td></tr><tr><td>262K, 1</td><td>133,140</td><td>122,545</td><td>131,382</td><td>49,238</td><td>45,127</td><td>64,674</td><td>-51.4%</td><td>+31.3%</td></tr><tr><td>262K, 4</td><td>142,686</td><td>142,359</td><td>153,418</td><td>118,899</td><td>140,713</td><td>176,960</td><td>+15.3%</td><td>+25.8%</td></tr><tr><td>262K, 8</td><td>142,617</td><td>142,854</td><td>152,211</td><td>139,579</td><td>180,199</td><td>190,842</td><td>+25.4%</td><td>+5.9%</td></tr><tr><td>262K, 16</td><td>142,661</td><td>141,581</td><td>151,376</td><td>141,134</td><td>183,182</td><td>220,899</td><td>+45.9%</td><td>+20.6%</td></tr><tr><td>262K, 32</td><td>142,715</td><td>141,497</td><td>133,965</td><td>141,248</td><td>184,753</td><td>218,743</td><td>+53.3%</td><td>+18.4%</td></tr></table>

At � ≥ 8, WavePP reduces p50 completion time by 19.1–42.2% relative to TRT-LLM TP8/EP8 and by 5.4–19.2% relative to the PP baseline with the lower p50. Its p95 is lower than both PP baselines in three of the six settings and 2.4–3.2% higher than the lower PP baseline p95 in the other three. WavePP therefore improves throughput and median completion time more consistently than p95 completion time.

Table 7: Queue-inclusive completion time for seeded reuse, in seconds, with WavePP deltas against the lowest-latency TP8/EP8 and PP8 results in each cell. Negative deltas indicate lower completion time for WavePP.
<table><tr><td></td><td colspan="2">vLLM TP8/EP8</td><td colspan="2">SGLang TP8/EP8</td><td colspan="2">TRT-LLM TP8/EP8</td><td colspan="2">vLLM PP8</td><td colspan="2">SGLang PP8</td><td colspan="2">WavePP</td><td colspan="2">∆TP</td><td colspan="2">∆PP</td></tr><tr><td>Context, c p50</td><td></td><td></td><td>p95 p50</td><td></td><td>p95 p50</td><td>p95</td><td>p50</td><td>p95</td><td>p50</td><td>p95</td><td>p50</td><td>p95</td><td>p50</td><td>p95</td><td>p50</td><td>p95</td></tr><tr><td>131K,1</td><td>0.8</td><td>0.9</td><td>1.0</td><td>1.0</td><td>0.9</td><td>0.9</td><td>3.0</td><td>3.0</td><td>3.2</td><td>3.2</td><td>2.3</td><td>2.3</td><td>+187.5%</td><td>+155.6%</td><td>-23.3%</td><td>-23.3%</td></tr><tr><td>131K, 4</td><td>3.1</td><td>3.7</td><td>3.1</td><td>3.8</td><td>2.8</td><td>3.0</td><td>3.6</td><td>4.6</td><td>3.8</td><td>4.7</td><td>4.2</td><td>4.5</td><td>+50.0%</td><td>+50.0%</td><td>+16.7%</td><td>-2.2%</td></tr><tr><td>131K, 8</td><td>6.2</td><td>6.3</td><td>5.6</td><td>7.0</td><td>5.2</td><td>6.9</td><td>5.7</td><td>7.9</td><td>4.1</td><td>6.2</td><td>3.7</td><td>5.5</td><td>-28.8%</td><td>-12.7%</td><td>-9.8%</td><td>-11.3%</td></tr><tr><td>131K, 16</td><td>12.4</td><td></td><td>12.5 12.0</td><td>12.7 10.5</td><td></td><td>12.3</td><td>8.9</td><td>12.4</td><td>8.0</td><td>9.3</td><td>7.3</td><td>9.6</td><td>-30.5%</td><td>-22.0%</td><td>-8.8%</td><td>+3.2%</td></tr><tr><td>131K, 32</td><td>23.9</td><td></td><td>24.8 23.8</td><td>24.7 22.6</td><td></td><td>22.8</td><td>17.9</td><td>20.3 15.9</td><td></td><td>16.6</td><td>13.2</td><td>17.0</td><td>-41.6%</td><td>-25.4%</td><td>-17.0%</td><td>+2.4%</td></tr><tr><td>262K,1</td><td>2.0</td><td>2.0</td><td>2.1</td><td>2.1</td><td>2.0</td><td>2.0</td><td>5.3</td><td>5.3</td><td>5.8</td><td>5.8</td><td>4.1</td><td>4.1</td><td>+105.0%</td><td>+105.0%</td><td>-22.6%</td><td>-22.6%</td></tr><tr><td>262K, 4</td><td>6.8</td><td>8.0</td><td>6.9</td><td>8.1</td><td>7.7</td><td>8.5</td><td>8.5</td><td>10.2</td><td>7.1</td><td>8.4</td><td>5.7</td><td>6.2</td><td>-16.2%</td><td>-22.5%</td><td>-19.7%</td><td>-26.2%</td></tr><tr><td>262K, 8</td><td>14.7</td><td></td><td>14.7 14.8</td><td>15.0 13.1</td><td></td><td>15.2</td><td>14.5</td><td>16.4 11.2</td><td></td><td>12.4</td><td>10.6</td><td>12.7</td><td>-19.1%</td><td>-13.6%</td><td>-5.4%</td><td>+2.4%</td></tr><tr><td>262K, 16</td><td>29.4</td><td></td><td>29.5 29.5</td><td>30.9</td><td>27.9</td><td>28.4</td><td>28.9</td><td>29.0</td><td>22.4</td><td>22.4</td><td>18.1</td><td>19.3</td><td>-35.1%</td><td>-32.0%</td><td>-19.2%</td><td>-13.8%</td></tr><tr><td>262K, 32</td><td>58.9</td><td></td><td>58.9 59.3</td><td>60.6 61.9</td><td></td><td>66.0</td><td>57.9</td><td>58.0</td><td>44.1</td><td>44.9</td><td>35.8</td><td>39.7</td><td>-39.2%</td><td>-32.6%</td><td>-18.8%</td><td>-11.6%</td></tr></table>

## 5.7. Mixed Traffic and Completion Latency

Mixed traffic places short uncached suffixes alongside long cold requests. WavePP has the highest throughput at all three concurrencies. It improves geometric-mean throughput by 46.1% over TRT-LLM TP8/EP8, 56.8% over SGLang TP8/EP8, and 50.0% over vLLM TP8/EP8. Relative to the PP baselines, WavePP improves geometric-mean throughput by 5.0% over SGLang and 14.6% over vLLM. At � = 32, WavePP serves 61,190 input tok/s with p50/p95 completion times of 83.8/109.7 s.

![](images/5bd90a30d583ab7bb0e50b923af2aa0b3d1f021e55030fb5ee84628234f52872.jpg)  
Figure 12: Mixed 75/25 warm-131K/cold-262K traffic. The left panel reports aggregate throughput and the right reports queue-inclusive p50 completion time.

WavePP reduces p50 completion time by 9.5–33.1% and p95 by 22.5–43.4% relative to the lowest corresponding TP baseline values. Its p50 is also 5.4–5.6% lower than the lowest PP baseline p50. Relative to the lowest PP baseline p95, WavePP’s p95 is 3.0% lower at � = 8, but 14.3% and 11.8% higher at � = 16 and � = 32. In this mixed workload, WavePP improves throughput and reduces median completion time, while p95 completion time is higher at the two larger concurrencies.

Table 8: Mixed-workload aggregate prompt throughput (input tok/s). Positive deltas indicate higher throughput for WavePP.
<table><tr><td>C</td><td>vLLM TP8/EP8</td><td>SGLang TP8/EP8</td><td>TRT-LLM TP8/EP8</td><td>vLLM PP8</td><td>SGLang PP8</td><td>WavePP</td><td> $\Delta _ { \mathrm { T P } }$ </td><td>∆PP</td></tr><tr><td>8</td><td>40,797</td><td>38,986</td><td>41,945</td><td>53,129</td><td>58,079</td><td>61,063</td><td>+45.6%</td><td>+5.1%</td></tr><tr><td>16</td><td>40,790</td><td>39,062</td><td>42,064</td><td>53,551</td><td>58,468</td><td>61,436</td><td>+46.1%</td><td>+5.1%</td></tr><tr><td>32</td><td>40,849</td><td>39,074</td><td>41,695</td><td>53,678</td><td>58,348</td><td>61,190</td><td>+46.8%</td><td>+4.9%</td></tr></table>

Table 9: Queue-inclusive completion time for mixed traffic, in seconds, with WavePP deltas against the lowest-latency TP8/EP8 and PP8 results in each cell. Negative deltas indicate lower completion time for WavePP.
<table><tr><td rowspan="2"></td><td colspan="2">vLLM TP8/EP8</td><td colspan="2">SGLang TP8/EP8</td><td colspan="2">TRT-LLM TP8/EP8</td><td colspan="2">vLLM PP8</td><td colspan="2">SGLang PP8</td><td colspan="2">WavePP</td><td colspan="2">∆TP</td><td colspan="2">∆PP</td></tr><tr><td>p50</td><td>p95</td><td>p50</td><td>p95</td><td>p50</td><td>p95</td><td>p50</td><td>p95</td><td>p50</td><td>p95</td><td>p50</td><td>p95</td><td>p50</td><td>p95</td><td>p50</td><td>p95</td></tr><tr><td>8</td><td>32.3</td><td>45.2</td><td>33.6</td><td>47.5</td><td>23.2</td><td>46.6</td><td>24.1</td><td>26.4</td><td>22.2</td><td>31.3</td><td>21.0</td><td>25.6</td><td>-9.5%</td><td>-43.4%</td><td>-5.4%</td><td>-3.0%</td></tr><tr><td>16</td><td>64.0</td><td>77.6</td><td>67.2</td><td>67.9</td><td>62.1</td><td>66.5</td><td>48.1</td><td>50.7</td><td>44.3</td><td>44.7</td><td>41.9</td><td>51.1</td><td>-32.5%</td><td>-23.2%</td><td>-5.4%</td><td>+14.3%</td></tr><tr><td>32</td><td>128.0</td><td>141.6</td><td>133.9</td><td>148.7</td><td>125.2</td><td>142.4</td><td>96.3</td><td>115.9</td><td>88.8</td><td>98.1</td><td>83.8</td><td>109.7</td><td>-33.1%</td><td>-22.5%</td><td>-5.6%</td><td>+11.8%</td></tr></table>

## 5.8. Prefix Retention

To measure how well reusable state survives as new requests arrive, we compare WavePP with TRT-LLM TP8/EP8 at five arrival rates between 0.50 and 1.00 requests/s. A good hit reuses at least 80% of the expected prefix. WavePP achieves a good-hit rate of 88–100%, compared with 73–83% for the tensor-parallel baseline. Its good-hit rate is higher at every tested rate.

![](images/6fda51e6c62c36c2642bd65c4b91b38db2e5ba4a24660e07216e5739732e4ef5.jpg)  
Figure 13: Good-hit rate in the prefix-retention sweep. Both systems receive identical requests at each offered-load setting.

## 5.9. Ablation Study

We evaluate the admission and scheduling mechanisms on one eight-GPU B300 node using the same code with different mechanisms enabled. Each configuration ran the six workloads in one server boot. Table 11 lists the five configurations. Starting from our naive PP8 re-implementation, we add wave admission and transport, then leases and capacity escrows, and finally MPU. The

Table 10: Prefix-retention good-hit rate. The five offered-load settings span 0.50–1.00 requests/s; positive differences indicate a higher good-hit rate for WavePP.
<table><tr><td>Offered rate (req/s)</td><td>TRT-LLM TP8/EP8</td><td>WavePP</td><td>Difference (pp) (↑ better)</td></tr><tr><td>0.500</td><td>82%</td><td>100%</td><td>+18</td></tr><tr><td>0.625</td><td>76%</td><td>92%</td><td>+16</td></tr><tr><td>0.750</td><td>73%</td><td>91%</td><td>+18</td></tr><tr><td>0.875</td><td>80%</td><td>88%</td><td>+8</td></tr><tr><td>1.000</td><td>83%</td><td>97%</td><td>+14</td></tr></table>

MPU phase includes planner-driven packing, adaptive wave sizing, and the progress deadline, so its results measure these scheduling changes together.

The TP8/EP8 reference uses a 32,768-token budget and a batch limit of 32, which we found to be the best setting we tested. All PP configurations use a 16,384-token budget and a batch limit of 16. The comparison with TP therefore includes these configuration differences, while the successive PP comparisons keep them fixed. Figure 14 shows cold 131K at � = 16 and reuse 131K at � = 8.

Table 11: Cumulative ablation configurations.
<table><tr><td>Rung</td><td>Configuration</td><td>Added mechanism</td></tr><tr><td>R0</td><td>TRT-LLM TP8/EP8</td><td>Tensor-parallel reference</td></tr><tr><td>R1</td><td>Naive re-implementation</td><td>Lockstep request preparation</td></tr><tr><td>R2</td><td>+ wave admission/transport</td><td>Independent context-wave execution</td></tr><tr><td>R3</td><td>+ lease admission</td><td>Exact reuse and capacity ownership</td></tr><tr><td>R4</td><td>+ MPU</td><td>Planner-driven packing, adaptive wave sizing, and progress deadline</td></tr></table>

Mechanism ablation  
![](images/5236236013a4cfd0b5cf9bd8283fbf30f475e51e22945e0e5f85d496c02b5530.jpg)  
Figure 14: Cumulative B300 ablation in two representative cells. Bars report aggregate throughput for the tensor-parallel reference and each cumulative configuration. R1 is the naive re-implementation. R4 combines plannerdriven packing, adaptive wave sizing, and a progress deadline as MPU.

On cold 131K at � = 16, wave admission and transport increase throughput from 8,826 to 29,692 input tok/s, a 236.4% increase. Later changes have little effect on throughput. MPU reduces p50 completion time from 69.69 to 43.03 s, but mean completion time changes only from 63.18 to 62.33 s and throughput rises by just 1.4%.

Admission and scheduling both matter more for cached requests. On reuse 131K at � = 8, adding leases and capacity escrows increases throughput from 96,796 to 149,021 input tok/s, a 54.0% increase. MPU further increases throughput to 211,214, an additional 41.7%, while reducing p50 from 6.51 to 4.20 s. This is because, without these mechanisms, stages can wait for reusable state causing gaps between GPU work. On the other hand, adaptive wave sizing spreads the limited suffix work across more waves so that more stages can execute concurrently. Both mechanisms matter here because requests have little computation left after reuse.

On reuse 262K at � = 32, the combined MPU phase reduces throughput from 175,305 to 158,238 input tok/s, a 9.7% decrease, although throughput remains 24.5% above TP8/EP8. Thus, in some circumstances, planner-driven packing can limit performance gains as compared to the optimal packing strategy (as we see a static packing performs better) for a specific workload, though the adaptibility is still beneficial for mixed and ever-changing workloads.

## 6. Limitations and Further Directions

Fully dynamic chunking. MPU adjusts the token budget in discrete steps based on the amount of admitted work. It does not estimate the execution time of each chunk. Yet equally sized chunks can have different costs because they attend to different prefix lengths or encounter different stage bottlenecks. A more flexible policy could choose chunk sizes from estimated execution times and observed stage progress. The challenge is to improve pipeline balance and completion latency while keeping planning inexpensive enough to stay ahead of execution. We further observed that adding MPU decreases performance in certain experiments. Improving the MPU policy to be more adaptive could mitigate these regressions.

Other pipeline topologies. The current evaluation uses TP1×PP4 and TP1×PP8. Configurations such as TP2×PP4 and PP16 would help establish how admission overhead and overlap change with stage count, tensor-parallel collectives, and inter-node bandwidth, and how other topologies respond to WavePP. Uneven layer placement is another useful case, since keeping requests ready cannot by itself remove a persistent imbalance in stage execution time.

Scheduling and cache policies. WavePP currently packs requests in FCFS order and reserves capacity before scheduling their chunks. Further work could explore alternative scheduling strategies that consider request priorities, deadlines, and the dynamic state of the cache to optimize both latency and throughput.

Multiple replicas. Combining multiple replicas with cache-aware routing across prefill workers could improve tail latency. Evaluating WavePP and the baselines in this setting under realistic traffic patterns would also test the benefits of a larger cache size for a long-running deployment. This could also include producing detailed SLO curves to better understand the latency behavior under different load conditions.

## 7. Conclusion

Pipeline parallelism requires more than keeping the GPUs busy with forward passes. New requests must also find reusable state and obtain enough cache capacity while earlier requests are still running. This becomes a distributed operation when stages retain, offload, and evict their caches independently. Performing this preparation on the execution path can delay the next wave even when the GPUs have work available.

We presented WavePP, which moves reuse discovery and admission ahead of execution. Hints reduce the cache-tree walks needed to find a common reusable prefix. Leases protect the chosen prefix, while capacity escrows reserve space for the suffix without immediately evicting reusable blocks. Each stage then materializes the request before executing its scheduled chunks. These mechanisms allow cache management to proceed independently across stages while the pipeline follows a common schedule.

With the same kernels and PP4 topology, WavePP improves TensorRT-LLM’s prefill throughput in 37 of 40 GLM 5.2 and MiniMax M2.7 settings. At concurrency 128, throughput for short-suffix requests under cache pressure increases by factors of 2.91 and 2.02. WavePP also achieves the highest measured throughput in all 18 Kimi K3 settings at concurrency eight or higher. The ablations show that admission and transport improve cold prefill, while coordinated cache admission and scheduling provide additional gains under reuse. Together, these results show the value of preparing requests early and keeping enough work available to sustain pipeline execution.

## 8. Acknowledgements

The author thanks Joyjit Daw and Philip Howes for their tremendous support and encouragement throughout the project, and Joyjit for the helpful discussions.

The author used AI tools to improve the clarity and presentation of the manuscript and to assist with generation of figures and tables. The research ideas, methodology, and experimental results are the author’s own work. The author reviewed the manuscript and takes full responsibility for its content.

## References

[1] V. Srivatsa, Z. He, R. Abhyankar, D. Li, and Y. Zhang. Preble: Efficient Distributed Prompt Scheduling for LLM Serving. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=meKEKDhdnx.

[2] W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, et al. Efficient Memory Management for Large Language Model Serving with PagedAttention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles, pages 611–626. ACM, 2023. URL https://dl.acm.org/doi /10.1145/3600006.3613165.

[3] L. Zheng, L. Yin, Z. Xie, C. Sun, J. Huang, C. H. Yu, et al. SGLang: Efficient Execution of Structured Language Model Programs. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, et al., editors, Advances in Neural Information Processing Systems, volume 37, pages 62557–62583. Curran Associates, Inc., 2024. doi: 10.52202/079017-2000.

[4] A. Agrawal, N. Kedia, A. Panwar, J. Mohan, N. Kwatra, B. Gulavani, et al. Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve. In 18th USENIX Symposium on Operating Systems Design and Implementation (OSDI 24), pages 117–134, Santa Clara, CA, July 2024. USENIX Association. URL https://www.usenix.org/conference/osdi24/presentation/agrawal.

[5] M. Shoeybi, M. Patwary, R. Puri, P. LeGresley, J. Casper, and B. Catanzaro. Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism. arXiv preprint arXiv:1909.08053, 2019. doi: 10.48550/arXiv.1909.08053.

[6] Y. Huang, Y. Cheng, A. Bapna, O. Firat, M. X. Chen, D. Chen, et al. GPipe: efficient training of giant neural networks using pipeline parallelism. In Proceedings of the 33rd International Conference on Neural Information Processing Systems, Red Hook, NY, USA, 2019. Curran Associates Inc. URL https://papers.neurips.cc/paper/8305-gpipe-efficient-training-of-giant-neural-networks-using -pipeline-parallelism.pdf.

[7] H. Liu, M. Zaharia, and P. Abbeel. RingAttention with Blockwise Transformers for Near-Infinite Context. In The Twelfth International Conference on Learning Representations, 2024. URL https://open review.net/forum?id=WsRHpHH4s0.

[8] D. Narayanan, A. Harlap, A. Phanishayee, V. Seshadri, N. R. Devanur, G. R. Ganger, et al. PipeDream: generalized pipeline parallelism for DNN training. In Proceedings of the 27th ACM Symposium on Operating Systems Principles, pages 1–15, Huntsville Ontario Canada, October 2019. ACM. doi: 10.1145/3341301.3359646.

[9] S. Hwang and J. Ahn. Revisiting Pipeline Parallelism for LLM Serving. In 20th USENIX Symposium on Operating Systems Design and Implementation (OSDI 26), pages 1803–1819, Seattle, WA, July 2026. USENIX Association. URL https://www.usenix.org/conference/osdi26/presentation/hwang.

[10] SGLang Contributors. Pipeline Parallelism for Long Context. SGLang documentation, n.d. URL https://docs.sglang.io/docs/advanced\_features/pipeline\_parallelism. Accessed September 11, 2026.

[11] vLLM Contributors. Architecture Overview. vLLM documentation, n.d. URL https://docs.vllm.ai /en/latest/design/arch\_overview/. Accessed September 11, 2026.

[12] T. Guo, X. Zhang, J. Du, Z. Chen, N. Xiao, and Y. Lu. gLLM: Global Balanced Pipeline Parallelism Systems for Distributed LLMs Serving with Token Throttling. In Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis, pages 1725–1741, St. Louis MO USA, November 2025. ACM. doi: 10.1145/3712285.3759823.

[13] Y. He, B. Zhao, and Z. Cao. SiPipe: Bridging the CPU-GPU Utilization Gap for Efficient Pipeline-Parallel LLM Inference. arXiv preprint arXiv:2506.22033, June 2025. doi: 10.48550/arXiv.2506.22033.

[14] R. Pan, Z. Wang, Z. Jia, C. Karakus, L. Zancato, T. Dao, et al. Marconi: Prefix Caching for the Era of Hybrid LLMs. arXiv preprint arXiv:2411.19379, April 2025. doi: 10.48550/arXiv.2411.19379.

[15] C. Zhang, K. Du, S. Liu, W. Kwon, X. Mo, Y. Wang, et al. Jenga: Effective Memory Management for Serving LLM with Heterogeneity. In Proceedings of the ACM SIGOPS 31st Symposium on Operating Systems Principles, pages 446–461, Lotte Hotel World Seoul Republic of Korea, October 2025. ACM. doi: 10.1145/3731569.3764823.

[16] Y. Liu, Y. Cheng, J. Yao, Y. An, X. Chen, S. Feng, et al. LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference. arXiv preprint arXiv:2510.09665, December 2025. doi: 10.48550/arX iv.2510.09665.

[17] D. Narayanan, M. Shoeybi, J. Casper, P. LeGresley, M. Patwary, V. Korthikanti, et al. Efficient largescale language model training on GPU clusters using megatron-LM. In Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis, pages 1–15, St. Louis Missouri, November 2021. ACM. doi: 10.1145/3458817.3476209.

[18] P. Qi, X. Wan, G. Huang, and M. Lin. Zero Bubble Pipeline Parallelism. arXiv preprint arXiv:2401.10241, November 2023. doi: 10.48550/arXiv.2401.10241.

[19] Z. Li, S. Zhuang, S. Guo, D. Zhuo, H. Zhang, D. Song, et al. TeraPipe: Token-Level Pipeline Parallelism for Training Large-Scale Language Models. In M. Meila and T. Zhang, editors, Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings ofMachine Learning Research, pages 6543–6552. PMLR, 18–24 Jul 2021. URL https://proceedings.mlr.press/v139/li21y.h tml.

[20] H. Zhang, T. Wei, Z. Zheng, J. Du, Z. Chen, and Y. Lu. TD-Pipe: Temporally-Disaggregated Pipeline Parallelism Architecture for High-Throughput LLM Inference. In Proceedings of the 54th International Conference on Parallel Processing, pages 689–698, San Diego CA USA, September 2025. ACM. doi: 10.1145/3754598.3754621.

[21] Y. Shi, X. Wang, J. Gao, J. Luo, X. Zhou, F. Liu, et al. VPP: Virtual Pipeline Parallelism for Efficient Chunked Prefill in Long-Context LLM Inference. arXiv preprint arXiv:2608.26523, August 2026. doi: 10.48550/arXiv.2608.26523

[22] SGLang Contributors. SGLang. Software repository, 2026. URL https://github.com/sgl-project/s glang/tree/71de97b264b04dcd514cf904003028aefe9775c8. Version 0.5.18, revision 71de97b264b0. Inspected September 11, 2026.

[23] vLLM Contributors. vLLM. Software repository, 2026. URL https://github.com/vllm-project/vll m/tree/v0.28.0. Version 0.28.0. Inspected September 11, 2026.

[24] H. Xu, T. Guo, and X. Zhang. DynaPipe: Dynamic Layer Redistribution for Efficient Serving of LLMs with Pipeline Parallelism. In Advances in Neural Information Processing Systems, volume 38, pages 151789–151811. Neural Information Processing Systems Foundation, Inc. (NeurIPS), 2025. doi: 10.52202/085713-4569.

[25] X. Bai, M. T. Islam, C. Wang, and A. N. Toosi. PipeLive: Efficient Live In-place Pipeline Parallelism Reconfiguration for Dynamic LLM Serving. arXiv preprint arXiv:2604.12171, April 2026. doi: 10.48550/arXiv.2604.12171.

[26] Y. Mei, Y. Zhuang, X. Miao, J. Yang, Z. Jia, and R. Vinayak. Helix: Serving Large Language Models over Heterogeneous GPUs and Network via Max-Flow. In Proceedings ofthe 30th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 1, pages 586–602, Rotterdam Netherlands, March 2025. ACM. doi: 10.1145/3669940.3707215.

[27] I. Gim, G. Chen, S.-s. Lee, N. Sarda, A. Khandelwal, and L. Zhong. Prompt Cache: Modular Attention Reuse for Low-Latency Inference. arXiv preprint arXiv:2311.04934, April 2024. doi: 10.48550/arXiv.2311.04934.

[28] NVIDIA. KV Cache System. TensorRT-LLM 1.1.0 documentation, n.d. URL https://nvidia.github. io/TensorRT-LLM/1.1.0/features/kvcache.html. Accessed September 11, 2026.

[29] R. Qin, Z. Li, W. He, J. Cui, F. Ren, M. Zhang, et al. Mooncake: Trading More Storage for Less Computation — A KVCache-centric Architecture for Serving LLM Chatbot. In 23rd USENIX Conference on File and Storage Technologies (FAST 25), pages 155–170, Santa Clara, CA, February 2025. USENIX Association. URL https://www.usenix.org/conference/fast25/presentation/qin.

[30] vLLM Contributors. Hybrid KV Cache Manager. vLLM design documentation, n.d. URL https: //docs.vllm.ai/en/latest/design/hybrid\_kv\_cache\_manager/. Accessed September 11, 2026.

[31] Z. Huang, K. Bao, Y. Zhang, J. Ouyang, and S. Pan. Unified Radix Cache: One Tree for Hybrid Model Prefix Caching. LMSYS technical blog, August 2026. URL https://www.lmsys.org/blog/2026-08-1 1-unified-radix-cache. Published August 11, 2026.

[32] NVIDIA. TensorRT-LLM. Software repository, n.d. URL https://github.com/NVIDIA/TensorRT-L LM. Accessed September 11, 2026.

[33] Y. Zhong, S. Liu, J. Chen, J. Hu, Y. Zhu, X. Liu, et al. DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving. In 18th USENIX Symposium on Operating Systems Design and Implementation (OSDI 24), pages 193–210, Santa Clara, CA, July 2024. USENIX Association. URL https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin.

[34] P. Patel, E. Choukse, C. Zhang, A. Shah, Í. Goiri, S. Maleki, et al. Splitwise: Efficient Generative LLM Inference Using Phase Splitting. In 2024 ACM/IEEE 51st Annual International Symposium on Computer Architecture (ISCA), pages 118–132, 2024. doi: 10.1109/ISCA59077.2024.00019.

[35] P. Kiely. The Efficient Frontier of LLM Inference. Baseten technical blog, September 2026. URL https://www.baseten.co/blog/the-efficient-frontier-of-llm-inference/. Updated September 1, 2026.

[36] NVIDIA. Disaggregated Serving. TensorRT-LLM documentation, n.d. URL https://nvidia.github. io/TensorRT-LLM/features/disagg-serving.html. Accessed September 11, 2026.

[37] vLLM Contributors. Disaggregated Prefilling (Experimental). vLLM documentation, n.d. URL https://docs.vllm.ai/en/stable/features/disagg\_prefill/. Accessed September 11, 2026.

[38] SGLang Contributors. PD Disaggregation. SGLang documentation, n.d. URL https://docs.sglang. io/docs/advanced\_features/pd\_disaggregation. Accessed September 11, 2026.

[39] NVIDIA. Architecture Flow. NVIDIA Dynamo documentation, n.d. URL https://docs.dynamo.nv idia.com/dynamo/knowledge-base/concepts/system-architecture/architecture-flow. Accessed September 11, 2026.

[40] C. Gray and D. Cheriton. Leases: an efficient fault-tolerant mechanism for distributed file cache consistency. In Proceedings of the twelfth ACM symposium on Operating systems principles, pages 202–210. ACM, November 1989. doi: 10.1145/74850.74870.

[41] P. E. O’Neil. The Escrow transactional method. ACM Transactions on Database Systems, 11(4):405–430, December 1986. doi: 10.1145/7239.7265.

[42] S. A. Jacobs, M. Tanaka, C. Zhang, M. Zhang, S. L. Song, S. Rajbhandari, et al. DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models. arXiv preprint arXiv:2309.14509, October 2023. doi: 10.48550/arXiv.2309.14509.

[43] J. Fang and S. Zhao. USP: A Unified Sequence Parallelism Approach for Long Context Generative AI. arXiv preprint arXiv:2405.07719, July 2024. doi: 10.48550/arXiv.2405.07719.

[44] D. Lepikhin, H. Lee, Y. Xu, D. Chen, O. Firat, Y. Huang, et al. GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=qrwe7XHTmYb.

[45] N. Bhatia, A. More, R. Borkar, T. Mitra, R. Matas, R. Zhao, et al. Helix Parallelism: Rethinking Sharding Strategies for Interactive Multi-Million-Token LLM Decoding. arXiv preprint arXiv:2507.07120, July 2025. doi: 10.48550/arXiv.2507.07120.

[46] J. Ainslie, J. Lee-Thorp, M. de Jong, Y. Zemlyanskiy, F. Lebron, and S. Sanghai. GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 4895–4901, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main.298.

[47] N. Shazeer. Fast Transformer Decoding: One Write-Head is All You Need. arXiv preprint arXiv:1911.02150, 2019. doi: 10.48550/ARXIV.1911.02150.

[48] DeepSeek-AI, A. Liu, B. Feng, B. Wang, B. Wang, B. Liu, et al. DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model. arXiv preprint arXiv:2405.04434, 2024. doi: 10.48550/ARXIV.2405.04434.

[49] A. Katharopoulos, A. Vyas, N. Pappas, and F. Fleuret. Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention. In Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pages 5156–5165. PMLR, 13–18 Jul 2020. URL https://proceedings.mlr.press/v119/katharopoulos20a.html.

[50] S. Yang, J. Kautz, and A. Hatamizadeh. Gated Delta Networks: Improving Mamba2 with Delta Rule. arXiv preprint arXiv:2412.06464, March 2025. doi: 10.48550/arXiv.2412.06464.

[51] Kimi Team, Y. Zhang, Z. Lin, X. Yao, J. Hu, F. Meng, et al. Kimi Linear: An Expressive, Efficient Attention Architecture. arXiv preprint arXiv:2510.26692, November 2025. doi: 10.48550/arXiv.2510. 26692.

[52] Kimi Team, T. Bai, Y. Bai, Y. Bao, M. C, J. Cai, et al. Kimi K3: Open Frontier Intelligence. arXiv preprint arXiv:2607.24653, August 2026. doi: 10.48550/arXiv.2607.24653.

[53] vLLM Contributors. Automatic Prefix Caching. vLLM design documentation, n.d. URL https: //docs.vllm.ai/en/stable/design/prefix\_caching/. Accessed September 11, 2026.

[54] Z.ai. GLM-5.2. Hugging Face model card, 2026. URL https://huggingface.co/zai-org/GLM-5.2. Accessed September 11, 2026.

[55] GLM-5-Team, A. Zeng, X. Lv, Z. Hou, Z. Du, Q. Zheng, et al. GLM-5: from Vibe Coding to Agentic Engineering. arXiv preprint arXiv:2602.15763, February 2026. doi: 10.48550/arXiv.2602.15763.

[56] MiniMax. MiniMax-M2.7. Hugging Face model card, 2026. URL https://huggingface.co/MiniMax AI/MiniMax-M2.7. Accessed September 11, 2026.

[57] A. Chen, A. Li, B. Zhou, B. Gong, B. Jiang, B. Dan, et al. The MiniMax-M2 Series: Mini Activations Unleashing Max Real-World Intelligence. arXiv preprint arXiv:2605.26494, July 2026. doi: 10.48550 /arXiv.2605.26494.

[58] SGLang Team. SGLang and Miles Add Day-0 Support for Kimi K3. LMSYS technical blog, July 2026. URL https://www.lmsys.org/blog/2026-07-27-kimi-k3-day0-support/. Published July 27, 2026.