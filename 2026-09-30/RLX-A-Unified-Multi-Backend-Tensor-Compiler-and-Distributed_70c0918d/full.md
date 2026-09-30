# RLX: A Unified Multi-Backend Tensor Compiler and Distributed Runtime in Rust

Eugene Hauptmann

Massachusetts Institute of Technology

Cambridge, MA, USA

eugenehp@mit.edu

Nataliya Kosmyna

Massachusetts Institute of Technology

Cambridge, MA, USA

nkosmyna@mit.edu

Abstract—Production machine learning (ML) stacks often split graph compilation and kernel execution across different layers and languages, making backend behavior, deployment guarantees, and performance fallbacks hard to reason about endto-end. RLX addresses this gap with a single Rust codebase that combines compiler and runtime roles around one primitive-level, three-level intermediate representation (IR), plus a transparent dispatch contract that resolves each operator to native, common-IR, or rewritten lowering and fails compilation when legalization is not possible. The same IR targets fourteen runtime devices (cpu, metal, mlx, ane, cuda, rocm, oneapi, tpu, hexagon, gpu, vulkan, opengl, directx, webgpu) and two specialty codegen paths (Cortex-M INT8 and FPGA), ingests safetensors, GGUF, ONNX, and rten formats, supports F16/BF16/F64/C64 and quantized INT4/INT8 flows with AMP/PTQ/QAT, and scales via tensor-/pipeline-parallel collectives over TCP and RDMA transports. Beyond neural workloads, RLX also extends to scientific/physics-style domains through sparse and dense linear algebra extensions (e.g., CSR LU/CG/matvec and LAPACKbacked factorizations) and 3D Gaussian splatting operators. We evaluate RLX against PyTorch, TensorFlow, JAX, candle, burn, tch, rten, MLX, CoreML, IREE, Glow, TensorRT, and tinygrad under identical input generation and p50 measurement methodology on one host. On all-MiniLM-L6-v2, RLX-Metal is fastest at every batch (e.g., 16.6 ms at batch 32 vs. PyTorch-MPS 26.7 ms). In the MNIST training table, RLX also has the top-throughput entry (graph-fused MLP: 946,487 img/s), above NumPy+BLAS (787,349 img/s), while retaining 100% top-1 parity on reference checks (e.g., Qwen3).

Index Terms—machine learning systems, compilers, intermediate representation, automatic differentiation, heterogeneous hardware, Rust, quantization, distributed computing

## I. INTRODUCTION

The progress of machine learning has been coupled to the ease with which developers can turn high-level model descriptions into efficient code on diverse hardware. Two strategies dominate. Graph compilers such as XLA [6], TVM [4], and MLIR [5] capture a whole-program representation and apply fusion, layout, and scheduling transformations before emitting code. Kernel runtimes such as cuDNN [7], Apple’s Metal Performance Shaders, and MLX [8] instead provide a curated set of hand-optimized primitives that an eager front end dispatches at run time. The first strategy maximizes cross-operator optimization; the second minimizes the distance between a call and a vendor-tuned kernel.

In practice the two are combined together with adapter layers, foreign-function boundaries, and a host language, typically Python, that is separate from the systems language in which the kernels are written. Python’s global interpreter lock, per-object reference counting, and dynamic dispatch impose a latency cost that adds up at small batch sizes and limits threading. It further complicates deployment to limited or nontraditional targets, such as microcontrollers, web browsers, custom silicon, and makes it hard to verify end-to-end if the code will actually run.

We introduce RLX a single, memory-safe system language which can host both a graph compiler and a kernel runtime, with one IR spanning the full range of targets from a server GPU down to an INT8 microcontroller. RLX is implemented as a Rust workspace of layered crates. Its IR is a small, composable set of primitives: structured control flow (Op::Scan), dense linear solves (Op::DenseSolve), quantization (Op::FakeQuantize, Op::DequantMatMul), masked attention with an explicit MaskKind, and a first-class custom-op extension surface. The optimizer performs fusion, automatic-mixed-precision policy, reverse- and forward-mode automatic differentiation, batching (vmap), broadcast legalization, and post-training-quantization insertion. The runtime performs backend dispatch, compile caching, and cost-based device selection across heterogeneous hardware.

This paper contributes a unified IR, optimizer, and runtime in one Rust workspace (Section III); a transparent dispatch that fails and does not silently degrade (Section IV) the graph ops and performance; portability across fourteen runtime and two codegen targets (Section V); extensibility, including distributed TP/PP (Section VI, Section III-D); and a cross-framework evaluation with runnable benchmark scripts (Section VIII).

## II. BACKGROUND AND MOTIVATION

## A. Requirements

We designed RLX using four requirements: (i) heterogeneous targets – one system reaching multicore CPUs, Apple unified-memory GPUs, discrete NVIDIA/AMD GPUs, TPUs, neural accelerators, and edge microcontrollers and custom logic, without a rewrite per device and responsive to perdeployment objectives (throughput, precision, IO latency, and power); (ii) one language without host/kernel split – the IR, optimizer, runtime, and kernels in one memory-safe language, so a model and its lowering are inspectable from a single tree;

(iii) transparency over silent fallback – the system reports, per operator, natively, or if it is lowered, or rewritten path and fails when an operator cannot be legalized for the chosen device; and (iv) extensibility from primitives – quantization schemes, custom operators, and downstream domains (sparse and dense linear algebra, Gaussian splatting) expressed against a stable extension surface.

## B. Motivation behind choosing Rust

Rust gives memory and thread safety without garbage collection, a build system (Cargo) that scales to a large workspace, and zero-cost FFI to vendor libraries (cuBLAS, cuDNN, MPSGraph, MLX, libtpu) - so RLX keeps the IR, optimizer, and backend kernels in one tree while still calling hand-tuned vendor code where it wins. The Rust frameworks Burn [9] and Candle [10] show community interest in this direction; RLX differs in coupling a primitive-level compiler IR to a transparent dispatch contract and in targeting MCU and FPGA from the same IR.

## C. Related work

RLX draws its representation from JAX [3] and XLA [6] (composable array primitives, differentiation and batching as transformations), its whole-program ethos from TVM [4], MLIR [5], IREE [19], and Glow [20], and its ergonomics from PyTorch [2]. The Rust frameworks candle [10], burn [9], tch [11] and the engines rten [12] and ONNX Runtime [13] are the closest neighbors; retargetable stacks (Mojo [21], TensorRT [22], tinygrad [23], Luminal [24], ZML [25]) share the compile-an-IR philosophy. MLIR is infrastructure for building compilers – no runtime, kernels, or autodiff of its own (IREE supplies those) – whereas RLX is a complete system with all three layers. Section IX makes the axis-by-axis comparison.

## III. SYSTEM DESIGN

RLX is organized as three layers in a IR crate, shown in Figure 1: authoring front ends and weight-format ingestion trace into a single IR, which an optimizer and runtime lower, dispatch, and distribute across a wide set of hardware backends and deployment platforms.

## A. Intermediate representation

The crate rlx-ir defines the tensor IR: an Op enum, Shape and DType types with a promotion table, a graph builder, shape inference, and a verifier. The IR is deliberately primitive-level. It contains roughly 113+ operators, including structured control flow (Scan), reductions, masked attention parameterized by MaskKind ∈ {None, Causal, SlidingWindow(w), Custom}, the recurrent family (Gru, Rnn, Mamba2), Fourier transforms (Op::Fft, native ondevice down to discrete Vulkan) over which finite impulse response (FIR) or infinite impulse response (IIR) and convolution-reverb filters compose with no new kernel, and quantized matrix multiplication (Op::DequantMatMul). An Op::Custom/Op::CustomFn pair provides an extension surface so that downstream crates register new semantics without modifying the core.

a) Three IR levels: The operator set is expressed at three progressively lower levels that form the compile pipeline of Figure 1. high-level IR (HIR) is the block vocabulary model builders emit – units such as Linear and SwiGLU – so a transformer layer is a few nodes, not dozens of primitives. HIR lowers to mid-level IR (MIR), the fused primitive tensor directed acyclic graph (DAG) that is the optimizer’s input: every fusion, autodiff, and legalization pass reads and writes MIR, as does the common-IR dispatch path of Section IV. Memory planning then annotates MIR with an arena buffer assignment to produce low-level IR (LIR), the input each backend lowers to device code. Authoring can begin at either level – block builders start at HIR; the lazy Tensor DSL traces directly to MIR – and keeping all three as views of one operator set is what lets autodiff, fusion, and the dispatch ladder compose without translation layers.

Two authoring surfaces target this IR: an explicit graph builder (Graph::new(...), g.matmul(...), Session::new(device).compile(g)) and a higherlevel NumPy-style Tensor DSL (rlx-tensor) of lazy, operator-overloaded handles – expressions such as (&a + &b).relu() trace into the same IR are not eagerly executed, and are fused and memory-planned only when a result is materialized.

## B. Optimizer

The optimization layer comprises three crates exposed through a single facade (rlx-opt).

Fusion (rlx-fusion) runs region-fusion passes over the mid-level IR and an inverse unfuse step that re-exposes primitives for differentiation and for backends that do not implement a fused form. Representative fused regions include a fused attention block, fused QK-RoPE, and a strided “Narrow×3 →Attention” pattern that provides the intermediate query/key/value writes.

Automatic differentiation (rlx-autodiff) enables reverse-mode grad, forward-mode jvp, Hessian-vector products hvp, nth-order scalar derivatives, and leading-axis batching vmap, all as transformations on the mid-level IR. Every training-specific backward operator decomposes to primitives so that derivatives can be stacked to defined order.

Compilation (rlx-compile) runs legalization, memory planning, automatic-mixed-precision policy, and post-trainingquantization insertion, producing a device-specific schedule. View-producing operators (Reshape, same-dtype Cast, axis-0 Narrow) are recognized as pure views and aliased to a root buffer with an offset, so backends emit a no-op rather than a copy.

a) Numeric precision: Precision is handled at the IR level. DType spans floating point (F32, F16, BF16, F64, complex C64) and integer types, with a promotion table fixing the result type of every mixed-type operation. An automatic mixed-precision (AMP) policy assigns per-operator compute and accumulation types such as F32 matmul accumulation over F16/BF16 inputs and a cast-elimination pass removes the redundant conversions the AMP rewrite introduces. It also tracks arithmetic intensity: F16 compute leaves a small, dispatchbound convolutional net unchanged but speeds a wide 16khidden MLP matmul by 16%, accuracy unmoved since storage and the optimizer stay F32. Below floating point, quantization flows through the same pipeline: a post-training-quantization (PTQ) pass inserts FakeQuantize nodes, quantizationaware training (QAT) uses straight-through estimators and learned step sizes, and weight-only quantization spans the GGUF schemes from INT8 to INT4 and block-scaled tensorcore minifloats – the FP8/FP6/FP4 grids and arbitrary fNeXmY formats, selectable per matmul (Section VI). Since precision is tracked in the IR, autodiff and fusion preserve the precision, and low-precision graphs are parity-checked against their F32 reference.

![](images/e7b1d520fb1d9dd4c28874f77b66a599faf4675f74ef949f5412e07d4d50ed54.jpg)  
Fig. 1. RLX architecture. Front ends and weight-format loaders (safetensors, GGUF, ONNX) trace into one primitive-level IR defined in rlx-ir over a single operator set (113 kinds). IR has three levels, named on the vertical edges by the representation crossing them: HIR (high-level block ops, e.g. Linear, SwiGLU) lowers to MIR (the fused primitive tensor DAG the optimizer consumes), which is buffer-planned into LIR (the input each backend lowers). The same IR reaches fourteen runtime targets (with GPU variants exposed as gpu/vulkan/opengl/directx/webgpu) plus two specialty codegen targets and every major platform.

## C. Runtime

The runtime (rlx-runtime) exposes the user-facing API: Session::new(device).compile(graph) returns a CompiledGraph that is executed with bound inputs. Once a graph is pruned, placed, and scheduled, its per-device schedule is cached so that repeated execution avoids recompilation. Multi-backend helpers (GraphDevices, DeviceRouter, DevicePolicy, FlexibleSession) implement costbased and environment-driven device selection, including fallback chains across devices.

## D. Distributed execution

RLX scales beyond one device with two forms of parallelism over a transport abstraction in rlx-driver. Pipeline parallelism gives each node a block of layers; only the hidden-state tensor crosses the wire, host-side between graph runs, so no compiler change is needed and a model too large for one device fits across several. Tensor parallelism shards every layer as column-sharded (gate/up) and rowsharded (down/output) and inserts an in-graph all-reduce (collective.all\_reduce); the same all-reduce synchronizes gradients for data-parallel training. Collectives and point-to-point send/receive sit on a SymmetricTransport trait with an in-process LocalTransport and a network NetTransport (TCP); on Apple Silicon the all-reduce also runs device-resident through MLX over its ring/jaccl transports (RDMA over Thunderbolt, fiber-optic, and PCIe). We evaluate for correctness: a column-sharded and rowsharded layer reproduces the single-node result bit-for-bit at 2- and 4-way shards, and pipeline relaying is numerically exact. Single-box multi-process TP/PP (including GPU-resident allreduce) is tested; a multi-machine throughput study is future work, as the cross-machine transport is not yet hardened.

## IV. TRANSPARENT KERNEL DISPATCH

A distinguishing element of RLX is how it resolves each operator to code. Rather than silently routing unsupported operators to a CPU fallback, the runtime classifies every operator into one of four paths: native (the operator is in the backend’s supported\_ops, so a native MSL, CUDA, CPU, . . . ), and a common IR (a registered logical kernel not claimed by the backend, lowered to portable primitive MIR); rewritten – a structural unfuse/lower yields the same semantics in a different graph shape, and unsupported which are still illegal after rewriting, so the compile fails with a diagnostic report.

The default policy, PreferNative, takes the native path when a backend declares an operator in its supported\_ops claim set and otherwise lowers to the common-IR path. The policy can be overridden globally (RLX\_KERNEL\_DISPATCH=common|native) or per compile, including per-operator force lists. Crucially, the same supported\_ops claim set that the system uses to decide whether to fail a compile is also what device selection uses, so the graph will be routed to the device only if it is guaranteed to run successfully. Setting RLX\_DISPATCH\_REPORT=1 emits a per-operator summary (native / common-IR / rewritten / missing) so that a performance engineer can see exactly which operators are not in the fast path, and the path to acceleration is open: implement the native kernels, add the operator to that backend’s supported\_ops, and re-run until the operator moves from common-IR to native.

## V. BACKEND PORTABILITY

RLX targets fourteen runtime device variants behind one Device enum, all shown in Figure 1: cpu, metal, mlx, ane, cuda, rocm, oneapi, tpu, hexagon, gpu, vulkan, opengl, directx, and webgpu. Two more are specialty codegen targets that consume a narrower op set and are not in the runtime registry: rlx-cortexm (no\_std ARMv7E-M INT8) and rlx-fpga (IR → Verilog → bitstream).

Backends share kernel sources where the hardware allows: ROCm reuses the CUDA .cu sources dispatched through HIP, and the Metal and CPU paths mirror each other’s fusion patterns (e.g. stride-aware Narrow→RoPE). Every backend publishes a supported\_ops contract stating how many of the 113+ operator kinds it claims. For the eight production backends with published coverage tables, CPU lowers 104+ as the reference, then the GPU backends – MLX 84, Metal 76, wgpu 75, CUDA 71, ROCm 68, ANE 57 – and the INT8-first TPU path claims 50 with exclusive quantized matmul/conv operators. The Backend trait is open: additional runtime targets (Cerebras wafers, mobile NPUs, Qualcomm QNN) register by implementing the claim-set contract and providing operator thunks. Because this is the same claim set the legalizer checks, a graph routed to a backend is guaranteed to contain only operators that backend will compile, or the compile fails with a report.

## VI. EXTENSIBILITY CASE STUDIES

Like prior unified systems [1], RLX builds advanced features from its own primitives rather than as core special cases. Higher-order differentiation: because every backward operator decomposes to primitives, derivatives compose – RLX provides first- through third-order scalar derivatives with interlayer CSE, and third-order parity holds between the CPU reference and CUDA/Metal/MLX/wgpu on $x ^ { 3 }$ , ReLU, tanh, GELU, and SiLU. Quantized inference: Op::DequantMatMul covers the full llama.cpp scheme set (Q4 0–Q8 0, K-quants, IQ/TQ, MX) with GPU dequant on Metal/CUDA/ROCm/wgpu and a fused INT4 GEMV on Metal; a ScaledMatMul path adds block-scaled minifloats (FP8/FP6/FP4 and parameterized fNeXmY), bit-exact against the F32 oracle on

Apple Metal and an NVIDIA RTX through both CUDA and native Vulkan. Microcontroller INT8: rlx-cortexm emits no\_std ARMv7E-M kernels (with a companion native trainer for the INT8 weights), outside the runtime Backend model and validated against reference implementations. Hardware synthesis: rlx-fpga lowers the IR to Verilog and a bitstream with integer-only Q0.31 requant – the same IR that drives a server GPU also drives custom logic.

## VII. IMPLEMENTATION AND FORMATS

RLX is a Cargo workspace of layered crates: the rlx-ir has no backend dependencies, the optimizer crates depend only on the IR, each backend depends on the driver and IR, and a prelude crate (rlx) re-exports the common surface. Backends are feature-gated and mutually compatible, so one build can carry several and select at run time; Python bindings (pyrlx, via PyO3) expose the same surface. RLX ingests four open-model formats, all lowering to the same IR: safetensors (the native path for full/mixed-precision checkpoints), GGUF v1–v3 (the standalone rlx-gguf crate covers every llama.cpp scheme – Q4 0–Q8 0, K-quants, IQ/TQ, MX/NVFP4 – and can also write and convert GGUF), ONNX (imported and lowered by rlx-onnx with a conformance suite, so a model targeting ONNX Runtime compiles unchanged to any RLX backend), and rten [12] (.rten flatbuffer models produced by the rten toolchain, sharing import infrastructure with the ONNX path). A checkpoint from any format is fused, quantized, memory-planned, and dispatched by the identical pipeline.

## VIII. EVALUATION

We evaluate RLX along several axes: inference latency across RLX’s backends and other frameworks, the breadth of differentiation it can compile, model coverage, and end-to-end correctness; we also explain what in the architecture lets the same IR train, not just infer. (Operator coverage was quantified in Section V).

## A. Setup

All numbers use latest RLX on a single Apple Silicon host. Latency is the median (p50) of repeated trials of the encoder forward pass; the embedding runners (RLX via rlx-embed, PyTorch via transformers, and ONNX Runtime over the model’s ONNX export) all take identical synthetic seq-128 inputs and report p50 on the CPU. The differentiation benchmark uses the cycle counter (CNTVCT\_EL0) and is gated on thermal state (the harness refuses to run under throttling, a real source of silent slowdowns). Model weights come from Hugging Face. All benchmark scripts and result CSVs behind Fig. 2 and the differentiation numbers accompany the paper.

## B. Inference: RLX backends vs. other frameworks

We run a single common model – all-MiniLM-L6-v2 (384-d, 6 layers) – through every engine identically on one Apple Silicon host (seq 128, batch 1-32, synthetic inputs, p50 of the encoder forward). The same compiled RLX graph runs unchanged on each RLX backend (CPU, Metal, MLX, and GPU paths via wgpu/Vulkan on this host; CUDA/ROCm/oneAPI/T-PU/Hexagon/ANE/ DirectX/WebGPU/OpenGL on matching hardware); PyTorch (transformers) and ONNX Runtime each run on CPU and the Apple GPU/ANE (MPS, CoreML). Figure 2 plots all of them. RLX-Metal – the router’s pick – is fastest at every batch (16.6 ms at batch 32, ∼3.3× over its own CPU path at 55.2 ms), ahead of the best external, PyTorch-MPS (26.7 ms); RLX’s CPU path matches or beats PyTorch-CPU (67.9 ms at batch 32) and leads ONNX Runtime (118.6 ms at batch 32) from batch 4, while ONNX-CoreML trails. Scripts also wire candle as a parity reference and the llama.cpp embedding path.

![](images/c8f266ee6a51ee807ea4470f42e8050f4b18c86ed35ccbbcd5a313ad963ab25e.jpg)  
Fig. 2. One model, every engine: p50 latency (lower is better) of the five RLX backends (solid) against PyTorch and ONNX Runtime – each on CPU and the Apple GPU/ANE (dashed) – on all-MiniLM-L6-v2, seq-128 inputs across batch, one Apple Silicon host. The same compiled RLX graph runs unchanged on every backend; RLX-Metal leads at all batches, with PyTorch-MPS the fastest external. RLX-Vulkan (MoltenVK) and ONNX-CoreML (whose provider partitions the graph 55 ways) run off the top above batch 8. Scripts and CSVs accompany the paper at https://github.com/MIT-RLX/rlx-paper.

## C. Training: what the architecture enables

RLX trains, not only infers, and without a separate engine: gradients come from transforming the same IR. Three choices carry this. rlx-autodiff rewrites the HIR graph in place, so reverse-, forward-, and higher-order gradients (§VIII-D) compile through the same fusion, legalization, and devicerouting pipeline as the forward pass; the optimizer is part of the graph (an SGD/Adam step fuses into the schedule with weights resident via JIT); and the fusion levers that speed inference (chain fusion, an im2col+BLAS convolution) minimize the backward pass identically. The payoff is empirical (Fig. 3): on an identical LeNet CNN, graph-fusing the convolution and the SGD step lifts RLX’s CPU training to the throughput of the XLA/MLX stacks. In Fig. 3, the top-throughput order is RLX-Metal (67,476 img/s), RLX-Vk MLP (63,967 img/s), and RLX g-fused (56,507 img/s), directly from the plotted values in Fig. 3. The fastest RLX path is ∼16× its own scalar path, while every framework lands in the same 95– 98% accuracy band. The fused path carries no overhead even versus hand-written code: an MLP reaches 946,487 img/s (top row in the full table), ahead of NumPy+BLAS (787,349 img/s) at higher accuracy. Because backward is just more IR, the trained model then serves inference on every RLX backend at identical accuracy – up to 161,660 img/s on the Neural Engine – and emits the INT8 microcontroller deployment of §VI. On-device ANE training stays efficient by binding trainable weights as graph inputs (compile once, predict per step). The gradients match the CPU bit-for-bit, CPU and Metal are the most exercised, and some larger convolutional backward graphs remain bounded by Core ML lowering.

![](images/fee00a0ae11d585dc200ee4e94dd37fdc21dd60e45a464c1401528ff73298b47.jpg)  
Fig. 3. MNIST training throughput on an identical LeNet CNN (SGD batch 128, 2 epochs); RLX in purple, each bar labelled with its time-to-finish. Every bar is an Apple M4 Pro except green-edged RLX-CUDA (NVIDIA RTX) – the same IR, no rewrite, once its conv and max-pool backward run on GPU. All reach 95–98% accuracy; RLX trains fastest on Apple Metal (1.8 s), and the top-throughput order is RLX-Metal (67,476 img/s), RLX-Vk MLP (63,967 img/s), and RLX g-fused (56,507 img/s). Graph-fusing the convolution and SGD step lifts RLX’s CPU path into the XLA/MLX range (∼13.5× its scalar path), while RLX-Metal reaches ∼16× its own scalar path. The lighter RLX-Vk MLP bar is the same MLP on native Vulkan (MoltenVK), the only non-CNN entry. The chart shows portability and parity, not a claim that RLX beats a vendor-tuned cuDNN/cuBLAS pipeline; the full training CSV (not all plotted) also has RLX graph-fused MLP as the global top entry at 946,487 img/s. Trained models then serve inference on every RLX backend at identical 97.2% accuracy.

## D. Higher-order differentiation

To exercise the autodiff transforms beyond inference, we benchmark third-order scalar differentiation of a cubic-sum objective (batch 200) on the Apple Silicon host (median of 50 runs, via the accompanying bench/ harness). At these sizes the CPU is fastest (median time grows ∼ O(N<sup>2</sup>), 0.5 µs at N=64 to 12.7 µs at N=4096) while the launch-bound Metal and MLX paths stay nearly flat, overtaking only at much larger N. The point is that the same composable transforms differentiate to third order across CPU, Metal, and MLX (and, elsewhere, wgpu and CUDA) at on-device parity with the CPU.

## E. Supported models

A companion repository https://github.com/MIT-RLX/ rlx-models provides roughly 100+ model-family crates – LLMs (Qwen3, LLaMA 3.2, Gemma, Phi, GLM), state-space models (Mamba, LFM), vision and diffusion (SAM, DINOv2, BioCLIP-2, FLUX.2), embeddings (BERT, NomicBERT), and automatic speech recognition (ASR) and text-to-speech (TTS) models (Whisper, TinyTTS, Orpheus, Voxtral) – all at reference parity. Qwen3 uses safetensors and GGUF at

![](images/79302e2b825b7988ee28acfceedd279d8f6b7064255eb7552754e4b73ab35ac4.jpg)  
Fig. 4. Deployment reach (twelve platforms) versus capability depth (four: native-Rust author+compile, fused compiler IR, training-grade autodiff, and MCU/FPGA codegen). RLX occupies the top-right corner; CoreML and MLX cover the Apple family but stop there.

100% top-1 parity with Hugging Face; BioCLIP-2 reaches 100% parity on CPU/Metal/MLX/wgpu; the Cortex-M INT8 path reaches 96.6% on nRF52840; and the TPU path matches MiniLM-L6 through PJRT – one IR from a microcontroller to a server accelerator.

## IX. COMPARISON WITH ML FRAMEWORKS, APPLE LIBRARIES, AND INFERENCE ENGINES

We place RLX within the field (Fig. 4): Python frameworks (PyTorch [2], TensorFlow [1], JAX [3]); eager kernel libraries (candle [10], tch [11], MLX [8]); inference engines (rten [12], ONNX Runtime [13], [14], CoreML [15]); LLM stacks (ggml [16], llama.cpp [17], mistral.rs [18]); burn [9]; and compiler stacks (IREE [19], Glow [20], MAX [21], TensorRT [22], tinygrad [23], Luminal/ZML [24], [25]).

Each covers part of the design space but does not implement it end-to-end as RLX does. The majority of libraries and LLM stacks carry no whole-program compiler IR with crossbackend lowering; the inference engines and most compiler stacks are inference-only, without first-class differentiation; the Apple libraries are Apple-only; tch and TensorRT bind a heavy or single-vendor runtime; and all but candle, burn, rten, mistral.rs, and Luminal are non-Rust. Among the compilers only JAX and TensorFlow match RLX’s transform-based, fused-IR view of differentiation – but neither is native Rust or have lower overhead or better performance, nor reaches the Neural Engine, microcontrollers, or FPGAs from one IR.

RLX alone combines a native-Rust author-and-compile path, a primitive-level IR with reverse/forward/higher-order differentiation plus batching, transparent multi-backend dispatch, and edge codegen with full platform reach. MLX and CoreML are themselves RLX backends, so a model authored once targets Apple Silicon and NVIDIA, AMD, TPU, web, and MCU – no rewrite. Tensor-/pipeline-parallel distributed execution (Section III-D) extends the advantage beyond what the Rust neighbors provide.

## X. CONCLUSION AND FUTURE WORK

We presented RLX, a tensor compiler and runtime that unifies the graph-compiler and kernel-runtime roles in one Rust workspace. A primitive-level, three-level IR feeds a fusion, autodiff, and precision optimizer; a runtime resolves every operator through a transparent dispatch ladder and routes it across fourteen runtime targets plus microcontroller and FPGA codegen, with parallel distributed execution over TCP and RDMA. Across a comparison spanning the Python frameworks, Apple libraries, inference engines, LLM stacks, and retargetable compilers, RLX is distinguished by uniting a primitive-level compiler IR with training-grade differentiation, transparent dispatch, and edge codegen in one native-Rust tree – and, in the reported benchmarks, posts the top throughput results (fastest inference from batch 4 up on a common embedding model, and the top training-table entry at 946k img/s). Future work will include GPU and TPU optimizations, better training, and better multi-machine transport.

## ACKNOWLEDGMENTS

The authors thank the maintainers of the open-source Rust and ML ecosystems on which RLX builds, and the reviewers for their comments. Kimi-K3 LoRA pretrained on rlx and rlxmodels codebases running on rlx inference engine was used to produce an initial draft of the paper that was subsequently significantly revised by human authors.

## REFERENCES

[1] M. Abadi et al., “TensorFlow: A system for large-scale machine learning,” in OSDI, 2016, pp. 265–283.

[2] A. Paszke et al., “PyTorch: An imperative style, high-performance deep learning library,” in NeurIPS, 2019.

[3] R. Frostig, M. J. Johnson, and C. Leary, “Compiling machine learning programs via high-level tracing,” in SysML, 2018.

[4] T. Chen et al., “TVM: An automated end-to-end optimizing compiler for deep learning,” in OSDI, 2018, pp. 578–594.

[5] C. Lattner et al., “MLIR: Scaling compiler infrastructure for domain-specific computation,” in CGO, 2021, pp. 2–14.

[6] XLA and TensorFlow teams, “XLA: Optimizing compiler for machine learning,” 2017. https://www.tensorflow.org/xla

[7] S. Chetlur et al., “cuDNN: Efficient primitives for deep learning,” arXiv:1410.0759, 2014.

[8] A. Hannun et al., “MLX: machine learning on Apple Silicon,” 2023. https://github. com/ml-explore/mlx

[9] “Burn: a deep-learning framework in Rust,” 2022. https://burn.dev

[10] “Candle: minimalist ML for Rust,” 2023. https://github.com/huggingface/candle

[11] L. Mazare et al., “tch-rs: Rust bindings for libtorch,” 2019. https://github.com/ LaurentMazare/tch-rs

[12] R. Knight et al., “rten: an ML runtime for Rust,” 2023. https://github.com/ robertknight/rten

[13] “ONNX Runtime,” 2018. https://onnxruntime.ai

[14] “ONNX: Open Neural Network Exchange,” 2017. https://onnx.a

[15] Apple Inc., “Core ML,” 2017. https://developer.apple.com/documentation/coreml

[16] ggml-org, “ggml: tensor library for machine learning,” 2022. https://github.com/ ggml-org/ggml

[17] ggml-org, “llama.cpp: LLM inference in C/C++,” 2023. https://github.com/ ggml-org/llama.cpp

[18] E. Buehler, “mistral.rs: an LLM inference engine in Rust,” 2024. https://github. com/EricLBuehler/mistral.rs

[19] The IREE Authors, “IREE: an MLIR-based end-to-end compiler and runtime,” 2019. https://iree.dev

[20] N. Rotem et al., “Glow: graph lowering compiler techniques for neural networks,” arXiv:1805.00907, 2018.

[21] Modular, “MAX and the Mojo programming language,” 2023. https://www. modular.com

[22] NVIDIA, “TensorRT: a high-performance deep-learning inference SDK,” 2016. https://developer.nvidia.com/tensorrt

[23] tinygrad authors, “tinygrad: a simple deep-learning framework,” 2020. https: //github.com/tinygrad/tinygrad

[24] J. Fioti et al., “Luminal: ahead-of-time deep-learning compilation in Rust,” 2024. https://github.com/luminal-ai/luminal

[25] “ZML: AI inference in Zig,” 2024. https://github.com/zml/zml