# SceneRig: An Agentic System for Simulation-Ready 3D Scene Reconstruction from Single Images

Rundong Luo<sup>1,2,∗</sup>, Yunong Liu<sup>1,†</sup>, Di Cao<sup>1,†</sup>, Matthew Tancik<sup>1</sup>, Shyamal Buch<sup>1</sup>, Colton Stearns<sup>1</sup>, Wenqi Xian<sup>1</sup>

<sup>1</sup>Luma AI, <sup>2</sup>Cornell University

## Abstract

Simulation is a powerful tool for robotics, but building diverse, simulation-ready scenes at scale remains dificult. Reconstructing scenes from single images is a promising alternative, but visual similarity alone is insuficient: objects must also be recovered, accurately placed, and physically grounded. Existing methods that rely on post-hoc physics settling often misplace objects, producing scenes that look correct but behave poorly under manipulation. We present SceneRig, an agentic system that turns a single image into an editable, simulation-ready scene. SceneRig combines geometric, visual, and physical tools to reconstruct a faithful 3D scene. An agent then performs edit-and-validate loops inside a physics simulator to achieve physical stability. Across diverse inputs, SceneRig improves instance segmentation, object-centric appearance, and 3D placement while keeping the reconstructed scenes physically stable. Furthermore, SceneRig is efective for downstream robotics: its scenes predict real-robot policy outcomes more accurately than the baseline method and more reliably support replay of recorded manipulations without trajectory adaptation.

Project Page: https://red-fairy.github.io/SceneRig/

## 1 Introduction

Evaluating the quality and robustness of action policies remains a fundamental bottleneck in robotics. While physical execution is the gold standard, on-robot evaluations are notoriously dificult to scale: they demand expensive hardware, human supervision, and repeated trials across tasks and conditions [21]. Consequently, simulation is a powerful alternative that enables large-scale policy evaluation [48]. However, scaling these environments remains dificult, as constructing realistic scenes requires substantial manual efort [16].

Recent advances in agentic reconstruction and 3D generation have made real-to-sim scene reconstruction a promising route to scaling simulation environments [6, 35, 52]. In fact, real-to-sim digital twins go beyond mere scaling: they capture the objects and arrangements of physical workspaces and let anyone turn a single photo into a simulation environment for policy evaluation. Yet, despite recent progress in real-to-sim, balancing high-quality 3D reconstruction with faithful, robot-ready simulation physics remains challenging [4, 26]. Even when a reconstruction is physically settled, objects can move away from their observed positions, producing a stable scene that no longer matches the robot’s workspace. To bridge this gap, we propose SceneRig, an agentic system for simulation-ready 3D scene reconstruction from a single image. SceneRig addresses this problem by coupling image-guided pose refinement with physical validation for faithful trajectory replay and policy evaluation.

![](images/4330e1b47e63fa7769c8c7c803be2d4475ccdb491b00bd6e891d1e190632085a.jpg)

![](images/fd2f9c51d83046550242436ca16277e43ac5f5500632d0184a40c457ab60daf9.jpg)

![](images/a543583effe4d84e2d5bdec0b4fd6f8941e30ca92beaa3ad4600d9b4bcf4940d.jpg)

![](images/25116235e6acf0cc55a0dd3de7da0a910d6b5e571253f4829eb8dc72eaa634b9.jpg)  
Real-to-sim policy evaluation <sub>Task:</sub> <sub>put</sub> <sub>mustard</sub> <sub>into</sub> <sub>the</sub> <sub>left</sub> <sub>bin</sub>

![](images/4a839cfdb8f15c4ac53c6f22dbd7e89ca77d866c4262ecf85787327a97e82026.jpg)

![](images/ea5a448b9c6d3fa4cb0df3527c89524f6a05b95d04e3667d1020958771d47f88.jpg)

![](images/028cab665aec37ff234399fe1358bbab1a6702978f9436a49982d36e14df308d.jpg)

![](images/298ef8c1f7aeff35c1a51fbc94b718eda93905fa758d660b4b4d8d21c2b5ab4d.jpg)  
Open-loop trajectory replay <sub>Task:</sub> <sub>put</sub> <sub>everything</sub> <sub>into</sub> <sub>the</sub> <sub>bin</sub>

![](images/37fa74e3cc27fac892ccf8e9dd11e0bbe7c7055420a3c81080f22c8a9798c82a.jpg)

![](images/be18c963fe44c2ba0e65de9f07cbc4a37e4b12a6532abb795bfdac866bff092f.jpg)

![](images/a01c04589ca0a75a882f0ad50628a29eaf29068a3dbb150c03d7479589f898fc.jpg)

![](images/881c26953e781e9e62d2141d0dfe2bb6c62f9e1a941ca1f5ab8efea4bdf4adca.jpg)  
Figure 1 Single-image simulation-ready scene reconstruction for robotics. SceneRig combines vision tools, agentic reasoning, and physical simulation to create an editable, simulation-ready scene from a single image (top). Its reconstructions support real-to-sim policy evaluation and open-loop trajectory replay, reproducing real-world outcomes more faithfully than SimFoundry (bottom).

Making this coupling efective requires the agent to inspect the efects of physical simulation and use that feedback to guide further corrections. Even GPT-6 Astra struggles to achieve accurate, physically stable object placement through VIGA or a general-purpose coding agent when the reconstructed scenes are evaluated after physical settling (Table 4). SceneRig’s core contribution is therefore a harness that connects the agent’s visual reasoning with geometric and physical tools for accurate, stable scene reconstruction. The agent uses the updated scene and feedback from the edit to decide what to investigate or correct next.

More concretely, SceneRig uses a flexible, modular harness with four stages. First, SceneRig builds a relational scene graph, identifying objects and their support relationships. Next, scene object creation uses 2D and 3D vision models to estimate metric geometry and reconstruct object meshes. We then assign physical properties and initialize object placement through geometric alignment and support-ordered settling. Third, specialized agents construct the surrounding geometry, materials, and lighting, with strict rule-based verification gates for geometry and visual verification of all three. Finally, and most importantly, SceneRig executes a physical edit-and-validate loop. The agent selects which object to adjust and whether to change its position, scale, or orientation; geometric and physical tools search for an adjustment and test its efects in simulation. Although the current implementation leverages specific state-of-the-art models, SceneRig is designed to incorporate future foundation models.

We evaluate SceneRig on diverse images, spanning synthetic renderings, real robot workspaces, AI-generated images, and in-the-wild smartphone captures. We demonstrate that SceneRig improves instance segmentation, object-centric appearance, and 3D object placement over existing methods, with all evaluated scenes passing physical stability checks (Tables 1, 2). Beyond geometric reconstruction, we show that SceneRig yields robot-ready environments (Figure 1), supporting both real-to-sim policy evaluation and open-loop trajectory replay. In summary, our contributions are:

• We introduce SceneRig, an agentic system that reconstructs simulation-ready digital twins from a single image, prioritizing physical grounding in addition to visual similarity.

• We build a modular harness that equips agents with geometric and physical tools. Through iterative edit-and-validate loops, SceneRig directly incorporates physics into pose refinement.

• SceneRig improves reconstruction quality and robotics performance over single-image real-to-sim systems. Compared with the baseline, it increases open-loop trajectory replay success from 36% to 80% and episode-level agreement with real-world policy result from 54% to 69%.

## 2 Related Work

Single-Image Scene Reconstruction. Object-centric reconstruction recovers individual objects and their spatial arrangement. Supervised methods predict geometry or retrieve assets: Total3D [28] predicts indoor layout and meshes, while ROCA [9] retrieves and aligns CAD models. Without object annotations, uORF [53] and uOCF [25] learn compositional neural fields from multi-view images to separate objects and backgrounds. For explicit meshes, generative methods reconstruct instances separately and align them [1, 10, 56, 58], while MIDI [15] jointly generates multiple objects. CAST [51] combines occlusion-completed object generation with pose refinement using scene relations and signed-distance constraints. SAM3D [6] predicts object geometry, texture, and camera-relative layout. However, visual fidelity alone does not ensure physical stability or readiness for simulation.

Agentic 3D Scene Creation. VLM agents construct 3D scenes through tool use and iterative feedback [13, 14]. To ground generation and editing in scene structure, Liu et al. [24] maintain labeled point clouds and scene relations. SceneWeaver [49] adaptively selects synthesis and editing tools, while Scenethesis [22] combines language-based planning with generated-image guidance and geometric constraints. Beyond planning and tool selection, SAGE [47] and SceneSmith [32] use editor-critic loops for indoor scene generation. For precise placement, agents also use numerical solvers [38, 43, 50].

Agentic workflows also support reconstruction from visual references. VIGA [52] reconstructs 3D scenes with a generator–verifier loop, while SEIG [12] extends this approach to refine geometry, materials, and lighting sequentially. SceneConductor [18] initializes layouts, builds environments from point maps, and delegates local corrections to specialist agents. For evaluation, BlenderGym [8] benchmarks graphics editing, while IR3D-Bench [23] evaluates scene understanding through agentic inverse rendering. In contrast, SceneRig’s agent corrects objects using geometry- and physics-grounded tools.

Physically Grounded Real-to-Sim. Real-to-sim aims at building interactive environments for robot learning and evaluation. RialTo [40], Re3Sim [11], and HoloScene [46] recover interactive assets from video or scans. Agentic Real2Sim [4] turns a recorded robot–object interaction into a simulatable twin. Our closest baseline, SimFoundry [35], reconstructs editable digital twins from video and generates digital cousins for real-tosim policy evaluation and sim-to-real policy training. For data generation without dynamics simulation, Real2Render2Real [54] renders robot training images from scans.

Concurrent single-image methods use several correction strategies. TableVerse [41] resolves collisions through geometric translations before settling; ϕ-Scene [20] optimizes clearance and settles objects in support order. REST3D [26] optimizes poses with cross-entropy optimization, while RoboSnap [55] alternates signed-distance optimization with settling. SimuScene [19] uses simulated displacement to trigger shape stretching or regeneration. In SceneRig, an agent identifies misplaced objects and calls geometry- and physics-grounded tools to correct them. These tools search for pose adjustments and test edits in simulation, returning feedback to guide further inspection and correction.

![](images/6bf2d653b193c588c6a5ea2b04ee722dbf51c5905155c1d1088fd3a05c09bbab.jpg)  
Figure 2 Overview of SceneRig. From a single image, SceneRig recovers objects, supporting surfaces, and their relations, and initializes a gravity-aligned metric scene through physical settling. Geometry, material, and lighting agents construct the surrounding scene. Finally, a pose-refinement agent inspects paired reference/render crops and invokes geometry- and physics-grounded tools to correct object placement.

Simulation for Policy Evaluation. Kadian et al. [17] tune simulation to improve sim–real correlation. SIM-PLER [21] calibrates controllers from trajectories and matches backgrounds and textures. PolaRiS [16] combines Gaussian splatting with policy co-training on real and simulated data; REALM [37] calibrates robot dynamics through replay and tests visual, semantic, and behavioral generalization. Following these, we assess our system through real-sim policy rollout agreement, task success-rate correlation, and open-loop trajectory replay.

## 3 Method

Given a single RGB image, we reconstruct an editable, simulation-ready 3D scene that matches its appearance and object arrangement. The system primarily targets tabletop scenes for robotics and can also process room scenes by treating the floor as the main support surface.

Let O denote movable objects and R denote root surfaces, such as tables, floors, and walls. Each entity has a persistent identity and a state defined by

$$
x _ { v } = \left( M _ { v } , \tau _ { v } , \theta _ { v } ^ { \mathrm { p h y s } } , X _ { v } \right) , \qquad v \in V = \mathcal { O } \cup \mathcal { R } .\tag{1}
$$

Here, M<sub>v</sub> is geometry, $\tau _ { v }$ appearance, $\theta _ { v } ^ { \mathrm { p h y s } }$ collision and physical properties, and $X _ { v }$ a metric transform with translation, rotation, and scale. Illumination is L, and the scene graph is

$$
G = ( V , E ) , \qquad E = E _ { \mathrm { s u p } } \cup E _ { \mathrm { s u r f } } ,\tag{2}
$$

where $E _ { \mathrm { s u p } }$ links objects to direct supports and $E _ { \mathrm { s u r f } }$ records structural relations among roots.

Figure 2 summarizes the pipeline. We describe relational scene graph construction and scene object creation under Section 3.1, followed by Agentic scene construction (Section 3.2) and physics-grounded object pose refinement (Section 3.3).

## 3.1 Relational and Metric Initialization

We first recover which entities are present, how they support one another, and where they lie in metric 3D.   
This relational and geometric evidence guides object reconstruction and physical settling.

Relational parsing. A VLM proposes movable objects O, root surfaces ${ \mathcal { R } } ,$ , and their support and structural relations, with a distinct description for each instance. MolmoPoint [7] localizes each entity from its description, and its predicted points select instance masks from SAM3 [3] candidates for that category. Subsequent checks recover objects omitted from the initial inventory, repair split or merged masks, and revise root surfaces and their relations. For occluded objects, we optionally remove foreground occluders with LanPaint [57] and re-segment the target. Entity identities persist throughout reconstruction (Appendix B.1).

Metric initialization. The scene graph tells us what is present and how entities are related, but not their metric placement in 3D. We use MoGe-2 [42] to estimate camera intrinsics and a dense metric 3D point map. When metric depth and calibrated intrinsics are available, such as from a robot’s RGB-D camera, they replace these estimates. A plane fitted to the main support defines a gravity-aligned coordinate frame in which the support lies $\mathrm { a t } ~ z = 0$ . Fitted planes and estimated contact geometry also provide geometric evidence for validating the proposed scene relations. SAM3D [6] reconstructs each movable object’s mesh and appearance from the image and its instance mask. We initialize its transform by translating and uniformly scaling the mesh to align with the corresponding metric 3D points. Finally, we estimate the physical properties for simulation following SceneSmith [32].

Support-ordered physical settling. With the support graph and metric object estimates in place, we settle the reconstructed objects while preserving their observed arrangement. Settling the full scene at once can cause objects to drift, topple, or resolve collisions in ways that difer from the image evidence. Thus, we settle objects in parent-before-child order under G. Each object is placed against its support and previously settled objects and simulated with PhysX in Isaac Sim. We then use ICP (adjusting translation, scale, and yaw) to improve alignment with its masked 3D observations and simulate it again. Guarded fallbacks may adjust collision geometry and physical properties to obtain a resting configuration; Appendix B.2 details these adjustments and the validation procedure.

## 3.2 Agentic Scene Construction

With objects settled, we reconstruct root geometry $( M _ { r } , X _ { r } )$ , materials $\tau _ { r }$ , and lighting L in separate stages sharing a persistent Blender scene. Following SEIG [12], each stage edits one variable group. It pairs an editing agent with an independent verifier, both of which receive the scene graph, the current scene snapshot, and the permitted edits. The editor modifies scene code and renders the result, and the verifier compares the render with the reference image, then approves it or requests corrections within the same scope. Each revision continues from the updated scene, preserving earlier work across iterations and stages.

Scene structure. The geometry agent constructs root surfaces in R as static colliders, positioning and orienting them with RANSAC-fitted planes and inferring their shapes and sizes from the input image; they must satisfy the required relations in $E _ { \mathrm { s u r f } }$ . Imported object geometry and appearance remain fixed; objects move only to correct measured contact defects or clear position mismatches.

VLM agents can produce visually plausible geometry that violates support or contact constraints, such as tables intersecting walls or objects floating above surfaces. Whereas VIGA and SEIG leave all checking to the verifier, we add a deterministic check\_rules gate that the editing agent must pass before verifier review. The gate checks root placement, object support, and interpenetration, and reports the first failing level with the afected entities. It enforces a minimum geometric standard, leaving the verifier to assess visual alignment, including object placement and background geometry.

Material and lighting. Once the structure is fixed, the material agent updates only root-surface materials and textures, including color, roughness, and surface patterns. The lighting agent then adjusts illumination and exposure to match the shadows, color, and contrast of the input image. Both stages leave background geometry, object placement, and object appearance unchanged, so pose refinement operates on fixed background geometry and rendering conditions.

## 3.3 Physics-Grounded Object Pose Refinement

After reconstructing the surrounding scene, we refine the object transforms $X _ { o }$ to correct placement errors that remain after initialization. A naive approach asks the VLM agent to adjust coordinates directly, e.g., by translating an object 0.05 units in scene code. However, lacking intrinsic metric grounding, agents often hallucinate these magnitudes, causing collisions or instability. We therefore use an investigation and editing loop in which the agent diagnoses placement errors and calls tools to search for corrections and test them in simulation.

Investigate--edit loop. A single agent session runs this loop in the persistent scene. Every object in O must be investigated at least once, and the agent may revisit objects as their surroundings change. The Investigate tool returns paired crops from the input image and current render, geometric measurements, and support information. Using this information, the agent decides which object to adjust and whether to change its position, scale, or orientation. For example, if an object looks too small in the render, the agent checks whether to move it closer to the camera or enlarge its mesh.

The agent passes the selected object and edit type to the Move tool, which supports horizontal translation, uniform scaling, and yaw adjustment. The tool determines the adjustment magnitude through bounded coarse-to-fine search, using the original pose as a baseline. Translation and scale scores combine silhouette overlap with depth; yaw combines centered silhouette overlap with DINOv2 features [31] from object crops. Appendix B.4 formalizes these scores in Eq. (5).

Physical validation. For routine bounded moves, the Move tool then simulates the selected candidate only if it improves alignment. Physical checks reject capsize or excessive drift, and substantial new or worsened penetration. An update is retained only if these checks pass and its post-settling acceptance score matches or exceeds the baseline (Eq. (7)); otherwise, the previous state is restored. If a support moves or is resized, its dependents in $E _ { \mathrm { s u p } }$ are repositioned and resettled as needed.

When bounded moves are insuficient, the agent can specify pose changes directly through the scene-code editing tool Edit. The tool settles afected objects in support order and returns the simulated render and physical diagnostics. The agent then decides whether to retain or undo the result. Appendix B.4 details search objectives and special cases.

Context management and prompt caching. We summarize older interaction rounds in blocks, retaining a log of actions, scores, and outcomes and replacing outdated images with references to saved artifacts. We retain the target image, stage-entry snapshot, and recent evidence. A per-object state table sits after the cache breakpoint, so measurements, pending checks, and budgets can change without invalidating the reusable prompt prefix. In our evaluation benchmark (Sec. 4), 84.1% of pose-refinement input tokens are served from cache (Appendix B.4; Figure 9).

Scene delivery and USD export. After refinement, a joint gravity simulation sets the final object transforms. We bake procedural materials and export meshes, textures, lights, and camera state to USD, preserving object identities, collision geometry, and physical properties for simulation. Appendix B.5 details conversion and simulator-side validation.

## 4 Experiments

We evaluate SceneRig’s reconstruction fidelity, object recovery, 3D layout, and physical stability (Sec. 4.2). We then examine physics-grounded placement and the efects of the VLM backbone and reconstruction harness (Sec. 4.3). Section 5 evaluates robotics applications; Appendix D provides additional analyses and results.

## 4.1 Experimental Setup

Baselines. We compare with four baselines: VIGA [52], REST3D [26], SceneConductor [18], and Sim-Foundry [35]. To support object-centric evaluation and simulation, we augment VIGA’s system prompt with a simple object-grouping contract: all parts of each movable object share a single empty Blender parent. We also evaluate SAM3D<sup>∗</sup>, which shares SceneRig’s segmentation, camera, objects, and scene context but retains SAM3D’s predicted poses without our metric grounding, physical settling, or pose refinement. We use Claude Opus 5 as the default VLM wherever applicable. With GPT-6 Astra, we also compare against Codex, a general-purpose harness that reconstructs the scene directly in Blender without external models. Appendices C.1 and D.2 provide implementation details and full harness results.

![](images/dd3595d7e9497f325f0c526c0ffc0f9d389c6e713587524892fa11f8056dabf7.jpg)  
Figure 3 Qualitative scene reconstruction results. Compared with the baselines, SceneRig more consistently preserves object identity, scale, and relative placement while reconstructing the surrounding scene.

Data. Our benchmark contains 80 tabletop images, including 44 simulator renderings from RoboDojo [5] and RoboLab [48] with ground-truth object identities and 3D positions. The remainder comprise real robot workspace images, Internet or AI-generated images, and smartphone photos. We evaluate source-view fidelity, perceptual preference, and physical stability on all scenes, and object-centric reconstruction and 3D layout on the simulator subset.

Metrics. We report full-frame PSNR, SSIM, LPIPS, and nCLIP. Object-centric PSNR<sup>∗</sup>/SSIM<sup>∗</sup> use object regions only (exclude background and table), while LPIPS<sup>∗</sup>/nCLIP<sup>∗</sup> average the per-object crop scores. Instance metrics are adjusted Rand index (ARI), foreground IoU, and object-count MAE. For 3D layout, optimal subpattern assignment (OSPA) [36] at 5/10 cm cutofs penalizes position errors and unmatched objects; F1@10 uses one-to-one 3D matches within 10 cm. To account for diferences in scale conventions across methods, we align each reconstructed layout to ground truth using a single global scale factor before computing OSPA and F1. GPT-5.6-terra and Qwen3.5-122B-A10B compare reconstruction pairs for completeness, layout, and appearance using the input image and both candidates’ source- and nearby-view renders. We summarize preferences with mean-centered Bradley–Terry scores (inter-judge Cohen’s κ = 0.87). An object passes the stability test if its displacement after 5 s is below 50 mm and its final velocity is zero; a scene passes only if every object passes. Check details in Appendices C.2 and C.3.

## 4.2 Reconstruction Quality

Reconstruction fidelity and stability. SceneRig achieves the best source-view PSNR, LPIPS, and nCLIP and the highest VLM preference scores across all three criteria (Table 1). VIGA and SceneConductor often fail the stability test, while REST3D’s higher SSIM does not translate into accurate object recovery or placement. SimFoundry achieves high stability but trails in object appearance and 3D layout. SceneRig leads all reported instance-recovery, object-appearance, and 3D-layout metrics (Table 2), with every reconstructed scene passing the stability test.

Qualitative results. Figure 3 compares reconstructions across diverse scenes. For visualization, REST3D and

Table 1 Scene-level reconstruction, pairwise preference, and physical stability. VLM columns report meancentered Bradley–Terry (BT) scores from pairwise judgments of source and nearby views. Within each criterion, a score advantage of ∆ implies a modeled pairwise win probability of $1 / ( 1 + e ^ { - \Delta } )$ . An object passes the stability test if its displacement after 5 s is below 50 mm and its final velocity is zero; a scene passes only if every object does. All methods use Claude Opus 5 as the underlying VLM.
<table><tr><td rowspan="2">Method</td><td colspan="4">Source-view reconstruction</td><td colspan="3">Pairwise VLM evaluation</td><td colspan="2">Physical stability</td></tr><tr><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td><td>nCLIP ↓</td><td>Completeness ↑</td><td>Layout ↑</td><td>Appearance ↑</td><td>Scene (%) ↑</td><td>Object (%) ↑</td></tr><tr><td>VIGA</td><td>11.414</td><td>0.529</td><td>0.653</td><td>0.180</td><td>0.225</td><td>-0.237</td><td>-0.249</td><td>10.0</td><td>27.1</td></tr><tr><td>REST3D</td><td>11.092</td><td>0.590</td><td>0.729</td><td>0.214</td><td>-1.074</td><td>-0.840</td><td>-1.625</td><td>31.3</td><td>76.6</td></tr><tr><td>SceneConductor</td><td>10.140</td><td>0.503</td><td>0.727</td><td>0.172</td><td>0.385</td><td>-0.837</td><td>-0.473</td><td>8.8</td><td>36.5</td></tr><tr><td>SimFoundry</td><td>12.060</td><td>0.582</td><td>0.720</td><td>0.194</td><td>-0.432</td><td>-0.683</td><td>-0.792</td><td>98.8</td><td>99.9</td></tr><tr><td>SceneRig (ours)</td><td>13.277</td><td>0.544</td><td>0.518</td><td>0.122</td><td>0.895</td><td>2.597</td><td>3.139</td><td>100.0</td><td>100.0</td></tr></table>

Table 2 Object recovery and 3D layout. Results on 44 simulator-grounded scenes. PSNR<sup>∗</sup>/SSIM<sup>∗</sup> use object pixels (table excluded); LPIPS<sup>∗</sup>/nCLIP<sup>∗</sup> average per-object crop scores with non-object pixels white. Layout metrics use global scale alignment; OSPA distances and cutofs are in centimeters.
<table><tr><td>Method</td><td colspan="7">Object-centric metrics</td><td colspan="3">3D object layout</td></tr><tr><td></td><td>ARI↑</td><td>FG-IoU ↑</td><td>Count MAE↓</td><td>PSNR*↑</td><td>SSIM* ↑</td><td>LPIPS*↓</td><td>nCLIP*↓</td><td>OSPA@5↓</td><td>OSPA@10 ↓</td><td>F1@10 ↑</td></tr><tr><td>SAM3D*</td><td>0.784</td><td>0.705</td><td></td><td>12.810</td><td>0.403</td><td>0.195</td><td>0.078</td><td>2.7</td><td>3.7</td><td>0.88</td></tr><tr><td>VIGA</td><td>0.283</td><td>0.260</td><td>3.36</td><td>8.987</td><td>0.315</td><td>0.325</td><td>0.149</td><td>4.9</td><td>8.9</td><td>0.29</td></tr><tr><td>REST3D</td><td>0.210</td><td>0.204</td><td>3.20</td><td>9.636</td><td>0.378</td><td>0.296</td><td>0.148</td><td>4.7</td><td>8.7</td><td>0.29</td></tr><tr><td>SceneConductor</td><td>0.197</td><td>0.187</td><td>1.14</td><td>9.493</td><td>0.329</td><td>0.317</td><td>0.152</td><td>5.0</td><td>9.5</td><td>0.14</td></tr><tr><td>SimFoundry</td><td>0.676</td><td>0.589</td><td>1.05</td><td>11.497</td><td>0.379</td><td>0.240</td><td>0.103</td><td>3.3</td><td>4.8</td><td>0.82</td></tr><tr><td>SCENERIG (ours)</td><td>0.878</td><td>0.819</td><td>0.27</td><td>13.609</td><td>0.440</td><td>0.170</td><td>0.062</td><td>2.4</td><td>3.1</td><td>0.93</td></tr></table>

SimFoundry are shown with white support surfaces. In these examples, VIGA alters object geometry, while REST3D and SceneConductor misestimate object scale or placement. SimFoundry also introduces objects absent from the input. SceneRig more faithfully preserves individual objects and their relative positions, including small utensils in cluttered scenes, while reconstructing the surrounding scene. Additional examples are provided in Appendix D.3.

## 4.3 Ablation Studies

In this section, we evaluate our placement pipeline and the value of the bounded move tool within a physics-enabled agent and compare the influence of VLM backbones and diferent harnesses.

Physics-grounded object placement. All variants in Table 3 share the same components except object placement. The SAM3D variants retain raw poses, bypassing our metric initialization, settling, and pose refinement. These poses often penetrate or float above supports; settling them for 5 s restores stability but reduces segmentation overlap and 3D placement accuracy. SceneRig’s support-ordered settling and pose refinement improve both while preserving stability. The code-edits-only variant removes Move while retaining Edit, which simulates agent-specified pose changes and returns feedback for agent review. This variant obtains worse object-appearance and 3D-layout scores (Tables 3 and 13). Removing Move reduces robot trajectory replay success from 80% to 44% (Appendix C.5, Table 11). These results highlight the efectiveness of our physics-grounded tool design.

VLM backbones and reconstruction harnesses. Within SceneRig, replacing Opus 5 with GPT-6 Astra substantially improves appearance but only modestly improves 3D layout (Appendix D.1). With a shared GPT-6 Astra backbone, SceneRig still achieves better results on most object-centric metrics, more accurate placement, and more stable scenes than VIGA and Codex (Tables 4 and 14). These results show that a stronger VLM alone does not ensure accurate and stable object placement: even when all methods use the same stronger backbone, SceneRig still achieves more accurate placement and better physical stability.

Table 3 Physics-grounded object placement. All components except placement and physical checks are shared. Post-hoc PhysX settles raw SAM3D poses for 5s; with code edits only, the agent sets move sizes.  
Table 4 Reconstruction harness comparison. All methods use GPT-6 Astra. VIGA and Codex are evaluated after 5 s of PhysX settling. Stability counts scenes that pass the stability test during this settle.
<table><tr><td></td><td colspan="2">Instance structure</td><td>3D object layout</td><td></td></tr><tr><td>Placement SAM3D raw poses</td><td>ARI↑ 0.784</td><td>FG-IoU ↑</td><td>OSPA@5 ↓ F1@10 ↑</td><td></td></tr><tr><td>+ post-hoc PhysX</td><td>0.748</td><td>0.705 0.664</td><td>2.7</td><td>0.88</td></tr><tr><td></td><td>0.858</td><td></td><td>3.0 2.4</td><td>0.84</td></tr><tr><td>ScENERIG, code edits only</td><td>0.878</td><td>0.792 0.819</td><td></td><td>0.92</td></tr><tr><td>SCENERIG</td><td></td><td></td><td>2.4</td><td>0.93</td></tr></table>

<table><tr><td></td><td colspan="3">Object appearance</td><td>3D object layout Stability</td></tr><tr><td>Method</td><td>|LPIPS*↓ nCLIP*</td><td></td><td>↓|OSPA@5 ↓ F1@10 ↑|Scenes ↑</td><td></td></tr><tr><td>VIGA</td><td>0.187</td><td>0.082</td><td>4.6</td><td>24/80</td></tr><tr><td>Codex</td><td>0.178</td><td>0.073</td><td>4.2 0.95</td><td>16/80</td></tr><tr><td>SCENERIG</td><td>0.151</td><td>0.060</td><td>2.3</td><td>80/80</td></tr></table>

Table 5 Open-loop trajectory replay. Five successful real demonstrations per task, with shared commands and simulation settings.

Figure 4 Real-to-sim policy evaluation. Each point is one policy–task pair (ten episodes); nested markers denote coincident pairs, and r pools all ten pairs.  
![](images/6bc312367977e1c4a66c1d1c5172d7c0ea84b7c5a260938fef78d8ff829037f5.jpg)

<table><tr><td>Task SCENERIG SimFoundry</td></tr><tr><td>Everything→bin 3/5 0/5</td></tr><tr><td>Cup→bowl 4/5 3/5</td></tr><tr><td>Fruits→bowl 4/5 2/5</td></tr><tr><td>Marker→cup 4/5 0/5</td></tr><tr><td>Mustard→bin 5/5 4/5</td></tr><tr><td>All 20/25 9/25</td></tr></table>

## 5 Robotics Applications

We assess SceneRig with real-to-sim policy evaluation and open-loop trajectory replay. We compare with SimFoundry, which reconstructs the source-view camera and accounts for physical stability; REST3D’s exported scenes do not retain a calibrated source camera, while VIGA and SceneConductor omit physical simulation. For each episode, both methods reconstruct the scene from an armless head-camera image using its metric depth and calibrated intrinsics instead of monocular estimates. Scenes are registered to the robot using the calibrated head-camera pose.

## 5.1 Real-to-Sim Policy Evaluation

Setup. We evaluate finetuned $\pi _ { 0 . 5 }$ [2] and LingBot-VLA 2.0 [45] on five tabletop tasks, using separate sets of 50 real episodes per policy (10 episodes per task, 100 episodes in total). Both reconstructions of each episode run the policy checkpoint used on the real robot, sharing simulator, calibration, and global simulation settings. We report success rates, episode-level agreement, and Cohen’s κ for each policy. Pearson’s r is computed both within each policy across five tasks and across all ten policy–task pairs. Details are provided in Appendix C.4.

SceneRig better predicts real-robot outcomes. Simulated outcomes match real ones in 69% of episodes with SceneRig versus 54% with SimFoundry (Tables 6 and 9), a significant paired diference (McNemar p = 0.017), and SceneRig’s Cohen’s κ exceeds SimFoundry’s by 0.29 (95% CI [0.08, 0.50]). Across ten policy–task pairs, real and simulated success rates correlate at $r = 0 . 9 2$ with SceneRig and 0.84 with SimFoundry, with mean maximum rank violations (MMRV) [21] of 0.04 and 0.11 (Figure 4; Table 10).

## 5.2 Open-Loop Trajectory Replay

Setup. We collect and evaluate 25 successful teleoperated demonstrations, five per task. Both SceneRig’s and SimFoundry’s reconstructions of each episode are imported into Isaac Sim with the calibrated bimanual robot. Controllers track the measured joint trajectory and gripper commands from the recorded initial configuration, without policy queries or trajectory adaptation. Commands, calibration, gains, and physics settings are shared

Table 6 Real-to-sim policy evaluation. Results for $\pi _ { 0 . 5 }$ on 50 real episodes and their reconstructions. Entries count successful episodes (success) or matching real and simulated outcomes (agreement). Cohen’s κ uses these 50 episodes; Pearson’s r correlates the five real and simulated task success rates.
<table><tr><td colspan="2">Environment</td><td>Everything→bin</td><td>Fruits→plate</td><td>Fruits→bowl</td><td>Marker→cup</td><td>Mustard→bin</td><td>All</td><td>| Cohen&#x27;s κ ↑ Pearson r ↑</td><td></td></tr><tr><td>Real-world</td><td>| success</td><td>7/10</td><td>7/10</td><td>7/10</td><td>5/10</td><td>7/10</td><td>33/50</td><td>1 一</td><td></td></tr><tr><td rowspan="2">SimFoundry</td><td>success</td><td>4/10</td><td>6/10</td><td>5/10</td><td>6/10</td><td>6/10</td><td>27/50</td><td></td><td>-0.38</td></tr><tr><td>agreement</td><td>5/10</td><td>7/10</td><td>2/10</td><td>5/10</td><td>5/10</td><td>24/50</td><td>-0.07</td><td></td></tr><tr><td rowspan="2">SceneRig</td><td>| success</td><td>9/10</td><td>8/10</td><td>5/10</td><td>4/10</td><td>7/10</td><td>33/50</td><td></td><td>0.70</td></tr><tr><td>agreement</td><td>8/10</td><td>7/10</td><td>6/10</td><td>7/10</td><td>8/10</td><td>36/50</td><td>0.38</td><td></td></tr></table>

Figure 5 Open-loop trajectory replay in reconstructed scenes. Both methods reconstruct the same image. Each pair shows the initial reconstruction and the final result of replaying identical recorded joint and gripper commands. SceneRig transfers the fruits separately; SimFoundry’s fused apple–plate mesh causes the plate to be lifted with the apple. View the embedded animation in Adobe Acrobat Reader or visit our project page.

(Appendix C.5).

SceneRig more reliably supports open-loop trajectory replay. SceneRig successfully replays 20/25 demonstrations, versus 9/25 for SimFoundry, with higher success on every task (Table 5); it succeeds on every demonstration that SimFoundry replays successfully (exact McNemar p = 0.001). SimFoundry fails due to missing objects, misplaced containers, and geometry that obstructs the robot. In Figure 5, its fused apple–plate mesh lifts the plate when the apple is grasped, while SceneRig preserves separate objects and supports the recorded manipulation.

## 6 Discussion

Conclusions. We present SceneRig, an agentic system that reconstructs simulation-ready scenes from single images. Its modular harness equips agents with geometric and physical tools to refine object placement through edit-and-validate loops. Across diverse tabletop scenes, SceneRig improves visual fidelity and placement accuracy while maintaining physical stability. Its scenes better predict real-world policy outcomes and more reliably reproduce recorded manipulations. These results ofer a path to scaling robot policy evaluation by bringing real workspaces into simulation.

Limitations and future work. Although SceneRig accurately places objects in tabletop scenes, complex room scenes remain challenging due to occlusion, depth ambiguity, and frequent contacts. SAM3D’s lack of PBR materials limits photorealism, and our current representation treats each reconstructed object as a single rigid body and does not recover joint structures or deformation. Future work will explore improved depth estimation and image-to-3D models.

## References

[1] Andreea Ardelean, Mert Özer, and Bernhard Egger. Gen3DSR: Generalizable 3D scene reconstruction via divide and conquer from a single view. In 3DV, 2025.

[2] Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Manuel Y. Galliker, Dibya Ghosh, Lachy Groom, Karol Hausman, Brian Ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke, Devin LeBlanc, Sergey Levine, Adrian Li-Bell, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Allen Z. Ren, Lucy Xiaoyang Shi, Laura Smith, Jost Tobias Springenberg, Kyle Stachowicz, James Tanner, Quan Vuong, Homer Walke, Anna Walling, Haohuan Wang, Lili Yu, and Ury Zhilinsky. π<sub>0.5</sub>: a vision-language-action model with open-world generalization. In CoRL, 2025.

[3] Nicolas Carion, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris, Chaitanya Ryali, Kalyan Vasudev Alwala, Haitham Khedr, Andrew Huang, Jie Lei, Tengyu Ma, Baishan Guo, Arpit Kalla, Markus Marks, Joseph Greer, Meng Wang, Peize Sun, Roman Rädle, Triantafyllos Afouras, Efrosyni Mavroudi, Katherine Xu, Tsung-Han Wu, Yu Zhou, Liliane Momeni, Rishi Hazra, Shuangrui Ding, Sagar Vaze, Francois Porcher, Feng Li, Siyuan Li, Aishwarya Kamath, Ho Kei Cheng, Piotr Dollár, Nikhila Ravi, Kate Saenko, Pengchuan Zhang, and Christoph Feichtenhofer. SAM 3: Segment anything with concepts. In ICLR, 2026.

[4] Guanxiong Chen, Qianjun Xia, Jiawei Peng, Heng Zhang, Pengyu Jing, Bole Ma, Justin Qian, Yixian Cheng, Ziyi Jiao, Bingyang Zhou, Yiduo Qu, Luoxin Ye, Kaifeng Zhang, Kunyi Wang, Weijia Zeng, Yunuo Chen, Pengzhi Yang, Ziqiu Zeng, Siyuan Luo, Huamin Wang, Chao Liu, Alan Yuille, Fan Shi, Changxi Zheng, Yunzhu Li, Chenfanfu Jiang, and Peter Yichen Chen. Agentic Real2Sim: Physics-based world modeling with vision-language agents. arXiv:2607.19190, 2026.

[5] Tianxing Chen, Yue Chen, Zixuan Li, Junyuan Tang, Kailun Su, Haoran Lu, Weijie Wan, Baijun Chen, Songling Liu, Haowen Yan, Honghao Su, Zhiyang Dou, Kaixuan Wang, Dandan Zhang, Yunze Liu, Yan Qin, Qiwei Liang, Qiwei Wu, Zijian Lin, Wenwei Lin, Yuran Wang, Minghua He, Tianshu Wu, Ruihai Wu, Jingquan Zhou, Kai-Chong Lei, Haibao Yu, Yuanfeng Ji, Weiyang Jin, Guanyu Lin, Xiaofan Li, Qi Xiong, Renjing Xu, Zhongyu Li, Wenhao Chai, Enze Xie, Ziwei Wang, Yao Mu, Hao Dong, Wojciech Matusik, Mingyu Ding, Wenbo Ding, Ping Luo, and Masayoshi Tomizuka. RoboDojo: A unified sim-and-real benchmark for comprehensive evaluation of generalist robot manipulation policies. arXiv:2607.04434, 2026.

[6] Xingyu Chen, Fu-Jen Chu, Pierre Gleize, Kevin J. Liang, Alexander Sax, Hao Tang, Weiyao Wang, Michelle Guo, Thibaut Hardin, Xiang Li, Aohan Lin, Jia-Wei Liu, Ziqi Ma, Anushka Sagar, Bowen Song, Xiaodong Wang, Jianing Yang, Bowen Zhang, Piotr Dollár, Georgia Gkioxari, Matt Feiszli, and Jitendra Malik. SAM 3D: 3Dfy anything in images. In CVPR, 2026.

[7] Christopher Clark, Yue Yang, Jae Sung Park, Zixian Ma, Jieyu Zhang, Rohun Tripathi, Mohammadreza Salehi, Sangho Lee, Taira Anderson, Winson Han, and Ranjay Krishna. MolmoPoint: Better pointing for VLMs with grounding tokens. arXiv:2603.28069, 2026.

[8] Yunqi Gu, Ian Huang, Jihyeon Je, Guandao Yang, and Leonidas Guibas. BlenderGym: Benchmarking foundational model systems for graphics editing. In CVPR, 2025.

[9] Can Gümeli, Angela Dai, and Matthias Nießner. ROCA: Robust CAD model retrieval and alignment from a single image. In CVPR, 2022.

[10] Haonan Han, Rui Yang, Huan Liao, Jiankai Xing, Zunnan Xu, Xiaoming Yu, Junwei Zha, Xiu Li, and Wanhua Li. REPARO: Compositional 3D assets generation with diferentiable 3D layout alignment. In ICCV, 2025.

[11] Xiaoshen Han, Junqiu Yu, Minghuan Liu, Yilun Chen, Xiaoyang Lyu, Yang Tian, Bolun Wang, Weinan Zhang, and Jiangmiao Pang. Re<sup>3</sup>Sim: Generating high-fidelity simulation data via 3D-photorealistic real-to-sim for robotic manipulation. In ICRA, 2026.

[12] Guangzhao He, Rundong Luo, Wei-Chiu Ma, and Hadar Averbuch-Elor. Thinking in Blender: Staged executable inverse graphics with vision-language models. arXiv:2606.02580, 2026.

[13] Ziniu Hu, Ahmet Iscen, Aashi Jain, Thomas Kipf, Yisong Yue, David A. Ross, Cordelia Schmid, and Alireza Fathi. SceneCraft: An LLM agent for synthesizing 3D scenes as Blender code. In ICML, 2024.

[14] Ian Huang, Guandao Yang, and Leonidas Guibas. BlenderAlchemy: Editing 3D graphics with vision-language models. In ECCV, 2024.

[15] Zehuan Huang, Yuan-Chen Guo, Xingqiao An, Yunhan Yang, Yangguang Li, Zi-Xin Zou, Ding Liang, Xihui Liu, Yan-Pei Cao, and Lu Sheng. MIDI: Multi-instance difusion for single image to 3D scene generation. In CVPR, 2025.

[16] Arhan Jain, Mingtong Zhang, Kanav Arora, William Chen, Marcel Torne, Muhammad Zubair Irshad, Sergey Zakharov, Yue Wang, Sergey Levine, Chelsea Finn, Wei-Chiu Ma, Dhruv Shah, Abhishek Gupta, and Karl Pertsch. PolaRiS: Scalable real-to-sim evaluations for generalist robot policies. In RSS, 2026.

[17] Abhishek Kadian, Joanne Truong, Aaron Gokaslan, Alexander Clegg, Erik Wijmans, Stefan Lee, Manolis Savva, Sonia Chernova, and Dhruv Batra. Sim2Real predictivity: Does evaluation in simulation predict real-world performance? IEEE RA-L, 2020.

[18] Jeonghwan Kim, Yushi Lan, Yongwei Chen, Hieu Trung Nguyen, Chuanyu Pan, and Xingang Pan. SceneConductor: 3D scene generation from a single image with multi-agent orchestration. arXiv:2606.08402, 2026.

[19] Inhee Lee, Sangwon Baik, Sungjoo Kim, Hyeonwoo Kim, Hyunsoo Cha, and Hanbyul Joo. SimuScene: Simulationready compositional 3D scene reconstruction from a single image. arXiv:2606.03994, 2026.

[20] Haodong Li, Lulu Shao, Haolin Lu, Yu Fu, Yen-Ru Chen, Seemandhar Jain, and Manmohan Chandraker. ϕ-Scene: Physically grounded image-to-3D scene reconstruction. arXiv:2606.21596, 2026.

[21] Xuanlin Li, Kyle Hsu, Jiayuan Gu, Karl Pertsch, Oier Mees, Homer Rich Walke, Chuyuan Fu, Ishikaa Lunawat, Isabel Sieh, Sean Kirmani, Sergey Levine, Jiajun Wu, Chelsea Finn, Hao Su, Quan Vuong, and Ted Xiao. Evaluating real-world robot manipulation policies in simulation. In CoRL, 2024.

[22] Lu Ling, Chen-Hsuan Lin, Tsung-Yi Lin, Yifan Ding, Yu Zeng, Yichen Sheng, Yunhao Ge, Ming-Yu Liu, Aniket Bera, and Zhaoshuo Li. Scenethesis: A language and vision agentic framework for 3D scene generation. In ICLR, 2026.

[23] Hengyu Liu, Chenxin Li, Zhengxin Li, Yipeng Wu, Wuyang Li, Zhiqin Yang, Zhenyuan Zhang, Yunlong Lin, Sirui Han, and Brandon Feng. IR3D-Bench: Evaluating vision-language model scene understanding as agentic inverse rendering. In NeurIPS, 2025.

[24] Xinhang Liu, Yu-Wing Tai, and Chi-Keung Tang. Agentic 3D scene generation with spatially contextualized VLMs. arXiv:2505.20129, 2025.

[25] Rundong Luo, Hong-Xing Yu, and Jiajun Wu. Unsupervised discovery of object-centric neural fields. TMLR, 2025.

[26] Xiaoxuan Ma, Jiashun Wang, Nicolás Ugrinovic, Yehonathan Litman, and Kris Kitani. REST3D: Reconstructing physically stable 3D scenes from a single image. In SIGGRAPH Asia, 2026.

[27] Lars Mescheder, Wei Dong, Shiwei Li, Xuyang Bai, Marcel Santos, Peiyun Hu, Bruno Lecouat, Mingmin Zhen, Amaël Delaunoy, Tian Fang, Yanghai Tsin, Stephan R. Richter, and Vladlen Koltun. Sharp monocular view synthesis in less than a second. In ICLR, 2026.

[28] Yinyu Nie, Xiaoguang Han, Shihui Guo, Yujian Zheng, Jian Chang, and Jian Jun Zhang. Total3DUnderstanding: Joint layout, object pose and mesh reconstruction for indoor scenes from a single image. In CVPR, 2020.

[29] OpenAI. Codex CLI: A lightweight coding agent that runs in your terminal. https://github.com/openai/codex, 2025.

[30] OpenAI. GPT-Image-2, 2026. URL https://developers.openai.com/api/docs/models/gpt-image-2.

[31] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mahmoud Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Hervé Jégou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. DINOv2: Learning robust visual features without supervision. TMLR, 2024.

[32] Nicholas Pfaf, Thomas Cohn, Sergey Zakharov, Rick Cory, and Russ Tedrake. SceneSmith: Agentic generation of simulation-ready indoor scenes. In ICML, 2026.

[33] Qwen Team. Qwen-Image-Edit-2509, 2025. URL https://huggingface.co/Qwen/Qwen-Image-Edit-2509.

[34] Qwen Team. Qwen3.5-122B-A10B. Hugging Face, 2026. URL https://huggingface.co/Qwen/Qwen3. 5-122B-A10B.

[35] Nadun Ranawaka, Josiah Wong, Wei-Lin Pai, Wei-Teng Chu, Tianyuan Dai, Masoud Moghani, Hang Yin, Yunfan Jiang, Wesley Durbano, Brandon Huynh, Yu Fang, Danfei Xu, Ruohan Zhang, Li Fei-Fei, Linxi Fan, Bowen Wen, Ajay Mandlekar, and Yuke Zhu. SimFoundry: Modular and automated scene generation for policy learning and evaluation. In CoRL, 2026.

[36] Dominic Schuhmacher, Ba-Tuong Vo, and Ba-Ngu Vo. A consistent metric for performance evaluation of multi-object filters. IEEE TSP, 2008.

[37] Martin Sedlacek, Pavlo Yefanov, Georgy Ponimatkin, Jai Bardhan, Simon Pilc, Mederic Fourmy, Evangelos Kazakos, Cees G. M. Snoek, Josef Sivic, and Vladimir Petrik. REALM: A real-to-sim validated benchmark for generalization in robotic manipulation. IEEE RA-L, 2026.

[38] Fan-Yun Sun, Weiyu Liu, Siyi Gu, Dylan Lim, Goutam Bhat, Federico Tombari, Manling Li, Nick Haber, and Jiajun Wu. LayoutVLM: Diferentiable optimization of 3D layout via vision-language models. In CVPR, 2025.

[39] Bin Tan, Changjiang Sun, Xiage Qin, Hanat Adai, Zelin Fu, Tianxiang Zhou, Han Zhang, Yinghao Xu, Xing Zhu, Yujun Shen, and Nan Xue. Masked depth modeling for spatial perception. arXiv:2601.17895, 2026.

[40] Marcel Torne, Anthony Simeonov, Zechu Li, April Chan, Tao Chen, Abhishek Gupta, and Pulkit Agrawal. Reconciling reality through simulation: A real-to-sim-to-real approach for robust manipulation. In RSS, 2024.

[41] Boyuan Wang, Yue Zhang, Xutao Xue, Xueyu Song, and Yu Sun. TableVerse: A large-scale tabletop dataset with real-world grounded layouts for generalizable manipulation. arXiv:2607.21017, 2026.

[42] Ruicheng Wang, Sicheng Xu, Yue Dong, Yu Deng, Jianfeng Xiang, Zelong Lv, Guangzhong Sun, Xin Tong, and Jiaolong Yang. MoGe-2: Accurate monocular geometry with metric scale and sharp details. In NeurIPS, 2025.

[43] Yian Wang, Han Yang, Minghao Guo, Xiaowen Qiu, Tsun-Hsuan Wang, Wojciech Matusik, Joshua B. Tenenbaum, and Chuang Gan. PhyScensis: Physics-augmented LLM agents for complex physical scene arrangement. In ICLR, 2026.

[44] Xinyue Wei, Minghua Liu, Zhan Ling, and Hao Su. Approximate convex decomposition for 3D meshes with collision-aware concavity and tree search. ACM TOG, 2022.

[45] Wei Wu, Fangjing Wang, Fan Lu, He Sun, Shi Liu, Yunnan Wang, Yibin Yan, Yong Wang, Shuailei Ma, Xinyang Wang, Yibin Liu, Shuai Yang, Tianxiang Zhou, Kejia Zhang, Lei Zhou, Cheng Su, Nan Xue, Bin Tan, Han Zhang, Youchao Zhang, Fei Liao, Xing Zhu, Yujun Shen, and Kecheng Zheng. From foundation to application: Improving VLA models in practice. arXiv:2607.06403, 2026.

[46] Hongchi Xia, Chih-Hao Lin, Hao-Yu Hsu, Quentin Leboutet, Katelyn Gao, Michael Paulitsch, Benjamin Ummenhofer, and Shenlong Wang. HoloScene: Simulation-ready interactive 3D worlds from a single video. In NeurIPS, 2025.

[47] Hongchi Xia, Xuan Li, Zhaoshuo Li, Qianli Ma, Jiashu Xu, Ming-Yu Liu, Yin Cui, Tsung-Yi Lin, Wei-Chiu Ma, Shenlong Wang, Shuran Song, and Fangyin Wei. SAGE: Scalable agentic 3D scene generation for embodied AI. In CVPR, 2026.

[48] Jenai Xuning Yang, Rishit Dagli, Alex Zook, Hugo Hadfield, Ankit Goyal, Stan Birchfield, Fabio Ramos, and Jonathan Tremblay. RoboLab: A high-fidelity simulation benchmark for analysis of task generalist policies. In RSS, 2026.

[49] Yandan Yang, Baoxiong Jia, Shujie Zhang, and Siyuan Huang. SceneWeaver: All-in-one 3D scene synthesis with an extensible and self-reflective agent. In NeurIPS, 2025.

[50] Yue Yang, Fan-Yun Sun, Luca Weihs, Eli VanderBilt, Alvaro Herrasti, Winson Han, Jiajun Wu, Nick Haber, Ranjay Krishna, Lingjie Liu, Chris Callison-Burch, Mark Yatskar, Aniruddha Kembhavi, and Christopher Clark. Holodeck: Language guided generation of 3D embodied AI environments. In CVPR, 2024.

[51] Kaixin Yao, Longwen Zhang, Xinhao Yan, Yan Zeng, Qixuan Zhang, Wei Yang, Lan Xu, Jiayuan Gu, and Jingyi Yu. CAST: Component-aligned 3D scene reconstruction from an RGB image. ACM TOG, 2025.

[52] Shaofeng Yin, Jiaxin Ge, Zora Zhiruo Wang, Chenyang Wang, Xiuyu Li, Michael J. Black, Trevor Darrell, Angjoo Kanazawa, and Haiwen Feng. Vision-as-inverse-graphics agent via interleaved multimodal reasoning. In ECCV, 2026.

[53] Hong-Xing Yu, Leonidas J. Guibas, and Jiajun Wu. Unsupervised discovery of object radiance fields. In ICLR, 2022.

[54] Justin Yu, Letian Fu, Huang Huang, Karim El-Refai, Rares Andrei Ambrus, Richard Cheng, Muhammad Zubair Irshad, and Ken Goldberg. Real2Render2Real: Scaling robot data without dynamics simulation or robot hardware. In CoRL, 2025.

[55] Shujie Zhang, Jingkun Yi, Weipeng Zhong, Zirui Zhou, Yangkun Zhu, Hanqing Wang, Xudong Xu, Weinan Zhang, and Chunhua Shen. RoboSnap: One-shot real-to-sim scene generation for generalizable robot learning and evaluation. arXiv:2607.06699, 2026.

[56] Qingcheng Zhao, Xiang Zhang, Haiyang Xu, Zeyuan Chen, Jianwen Xie, Yuan Gao, and Zhuowen Tu. DepR: Depth guided single-view scene reconstruction with instance-level difusion. In ICCV, 2025.

[57] Candi Zheng, Yuan Lan, and Yang Wang. LanPaint: Training-free difusion inpainting with asymptotically exact and fast conditional sampling. TMLR, 2025.

[58] Junsheng Zhou, Yu-Shen Liu, and Zhizhong Han. Zero-shot scene reconstruction from single images with deep prior assembly. In NeurIPS, 2024.

## A Supplementary Material Overview

This supplementary material provides implementation details, evaluation protocols, and additional results for SceneRig. Section B describes relational perception, object initialization and support-ordered settling, agentic scene construction, object pose refinement, implementation settings, and simulator export. Section C details baseline implementation and metric definitions, as well as the protocols for real-to-sim policy evaluation and open-loop trajectory replay. Section D presents ablation studies on pose/physics modules, VLM backbones, and harness comparisons, including the direct Codex baseline with GPT-6 Astra, and additional qualitative examples. Code is available at https://github.com/Red-Fairy/SceneRig.

## B Additional Implementation Details

This appendix expands the perception, initialization, and refinement procedures in Sections 3.1–3.3. Figures 6, 7, and 8 summarize relational perception, agentic scene construction, and the pose-refinement loop, respectively. We use the entity IDs and graph notation defined in Section 3 throughout. The descriptions below distinguish model predictions, deterministic tests, and fallbacks. Tables 7 and 8 collect the model roles and implementation defaults.

## B.1 Relational perception and mask recovery

Scene routing and entity parsing. A routing VLM first classifies the image as a room, a close-up showing a table, or a tabletop close-up. The efective route selects the proposer/verifier prompt pair for support construction and connectivity filtering.

A VLM call then groups entities by category. Each category record contains a short detector-friendly noun, up to three synonyms, and a list of instances. For example, a single mug record can contain a white mug at the front and a blue mug behind it. Each instance has its own description, type (movable object or root surface), and proposed direct support. Attributes and relative locations belong in the instance description, not in the category name. Category names are normalized and duplicate category records are consolidated. The parsing pipeline assigns category-local IDs such as mug#0 and mug#1; later recovery steps preserve surviving IDs and allocate new ones for additional instances. Root-surface categories are recognized by their head noun, such as table, wall, floor, or counter. The proposer also emits relations among these surfaces, such as a wall behind a table.

The routing distinction between a tabletop close-up and a view of the table’s body or base determines whether the main support needs only a slab or connected structure beneath it. The ordinary close-up path focuses on the main support and its objects, excluding people and objects on unrelated surfaces. An optional room-scale mode instead treats the floor as a root and furniture as objects. Containers remain objects, with their contents linked to the container as their direct support. Bounding walls receive separate root IDs. Following the proposed support chains identifies the root holding the most objects as the main support. Category grouping shares detection work; it does not assume that all instances have equal physical size.

Point-guided mask selection. MolmoPoint [7] localizes each instance from its description, including root surfaces. SAM3 [3] produces a full-image mask pool for each category; instances of that category share the pool but select masks using their own points. Selection starts with the smallest mask containing the point, then prefers a larger candidate if it contains at least 80% of that mask. Partially overlapping instances remain separate. Synonym pools are computed as needed and cached. The recovery ladder proceeds from full-image text prompts, to text prompts on a point-centered crop, to point-prompted SAM3 with all of the instance’s localized points as foreground prompts. Crops initially span one third of the image width and height. A mask cut by an internal crop boundary triggers a larger, two-thirds crop. Root surfaces skip cropping because a local crop can lose their full extent.

Selection is instance-aware. A candidate covering three or more distinct object points is rejected as a mergedobject mask; declared support children are excluded from this test. For a singleton category, compatible fragments near the localization point can be joined, and a substantially larger synonym mask can replace a detected part with the whole object. Point prompting normally requires SAM3’s predicted IoU to reach 0.6. If single-point prompting returns plausible masks with more than a twofold area diference, a VLM selects the intended object extent instead of comparing confidence scores across these scales; this scale-adjudicated choice bypasses the confidence floor. Roots have a more permissive final fallback: a nearby point-to-mask match is allowed, and a largest-mask choice is used only when localization returns no point. Root point-prompt masks are not subject to the object confidence floor. Each accepted mask records its recovery tier. Instances that exhaust the ladder without a usable mask enter the dropped list with a reason.

![](images/973ccf64691681d03aeb33060d30c4883b6789fbfbc420f35f8ced6e7a3bd394.jpg)  
Figure 6 Relational perception and mask recovery. A VLM identifies entities and proposes relations. MolmoPoint locations select SAM3 masks; subsequent checks recover missed objects, merge duplicate or split masks, complete occluded objects, and revise the root-surface records.

Inventory and overlap checks. Object-level checks target missed instances in repeated categories, such as masking only two of three mugs, and objects split across masks, such as a monitor separated into its base and screen. These checks reconcile duplicate assignments and overlaps.

Two instance descriptions can localize the same object. When this causes a duplicate or failed assignment, MolmoPoint gets one retry on an image with already claimed instances of that category crossed out. SAM3 still segments the original image. For categories initially containing multiple objects, unused masks in the category/synonym pools trigger a plural localization query, such as “all the mugs.” A new instance is accepted only when a point from the plural query falls inside a distinct unclaimed SAM3 mask and the mask passes the overlap checks against claimed objects; a VLM supplies its description. Roots are excluded. An initially singleton category is recounted only when its mask combines multiple instances: distinct SAM3 members must each occupy 8–92% of its area, lie at least 80% inside it, and together cover at least 70% of it. MolmoPoint must localize at least two members with a plural query. One member retains the original ID; the others enter the same unclaimed-mask acceptance procedure.

High-overlap masks are first deduplicated by entity type (IoU > 0.7 or containment > 0.8). Cross-category containment is deferred because it may represent a vessel and its contents. For touching object masks, the merge cascade first joins high-overlap duplicates at IoU ≥ 0.45, except deferred containment pairs, which require VLM review. Below this threshold, same-category pairs are never merged. Cross-category pairs with IoU 0.25–0.45 are sent to a VLM; other touching pairs are skipped. Mere contact does not imply one object. The VLM sees an unmarked crop and a red/blue mask overlay. Accepted merges union the masks, preserve the survivor’s ID, and redirect support references. Containment is the fraction of the smaller mask covered by the larger. When over 80% of a smaller, cross-category object is embedded in a larger mask, its pixels are removed from the larger mask unless this would remove over half of that mask. This carving runs before occlusion completion; legitimate overlap between completed masks is not carved away afterward.

Object components containing their localization points are retained. A fragment of at least 50 pixels can be restored if it reconnects through an occluder’s mask, contains no other object’s point, and overlaps other objects by at most 10% of its area.

Root-surface verification. A separate VLM receives the original image, a color-coded root-mask overlay labeled with IDs, and the root descriptions and proposed relations. It corrects the relations and describes missing surfaces. It can add a root by description without rerunning segmentation, but cannot add, remove, or modify movable objects. Initially unsegmentable roots are dropped; this review can reintroduce missing surfaces as description-only roots in R for visual construction.

Occlusion completion. Preprocessing invokes generative resegmentation after segmentation; its per-instance checks determine whether an image edit or mask recovery is needed. Mask pairs with at least 20 overlapping pixels after three-pixel dilation are possible occluder–target pairs. A VLM uses their image crops and depthorder evidence to identify the foreground object. A separate crop check for each instance estimates visibility and frame truncation and identifies missing structural parts, such as a handle or base. These judgments feed a deterministic recovery policy. Self-occlusion is not a missing external part. A supporting ancestor will not be treated as an occluder of its descendant. Completion is considered for a missing structural part, visibility below 75%, or an overlap-separation loss of at least 150 pixels or 3% of the target mask.

We apply LanPaint [57] with Qwen-Image-Edit-2509 [33] to remove the occluder and its support descendants. The edit mask is their union, dilated by 12 pixels; the prompt names the objects to remove and the target to reveal. SAM3 resegments the edited image using a part-aware description, the category, the original localization point, and finally synonyms. Candidate masks must have IoU at least 0.25 with the original target or reconnect a missing part within a 10-pixel dilation of it; adjacent-part unions cannot exceed eight times the original area. Sibling masks are excluded. If a candidate covers over 80% of a removed foreign object’s footprint, that footprint is clipped outside the original target mask. Removed objects in the target’s support subtree are exempt. The chosen mask must grow by more than 2%. When overlap carving alone triggers recovery, failed resegmentation can instead use the union of the original target and removed footprints. Other failures retain the original evidence. If only a structural part was missed, resegmentation uses the original image without an edit.

A VLM audits newly generated candidates using aligned before/after RGB images and masked cutouts. It checks target identity and count and mask fidelity; edited images additionally require preserved observed geometry and plausible completion, with occluder removal checked when applicable. All applicable checks must pass; rejection, uncertainty, or an unavailable judgment retains the original evidence. This audit also applies to the union fallback and border completion. Older cached completions without an audit may be reused, but are explicitly marked unverified.

RGB completion alone would leave the removed object’s depth in the newly revealed region. We therefore invalidate depth in the edit hole and pass the edited RGB image, remaining MoGe-2 depth, and camera intrinsics to LingBot-Depth [39]. Its completed metric points support mesh reconstruction and placement. Original masks, depth, and camera records are retained separately, and failed image or depth completion falls

back to the available original evidence.

A separate border-completion branch handles masks reaching within three pixels of exactly one left or right image edge. Completion requires either a VLM estimate of at least 10% beyond-frame extent or edge contact in at least 85% of the rows within the mask’s bounding box. Eligible cases receive a 25% canvas extension on that side, followed by LanPaint/Qwen outpainting and LingBot depth completion. Occluder removal, when needed, is a separate edit. Padding shifts the camera’s principal point but does not change its physical viewpoint. The extended image helps reconstruct the complete asset; source-view registration still uses the original image domain. This branch does not extrapolate arbitrary top, bottom, or two-sided truncations.

Post-completion cleanup. After completion, a further VLM pass checks whether nearby point-prompted detections are parts of one physical assembly, such as a stand and its supported device. Candidate masks must touch or lie within a small dilation of one another; proximity alone does not justify merging them. The VLM must identify them as parts of the same object. Merges update support references as well as masks. Objects without a usable mask are removed before the final inventory is fixed, with their records and reasons retained for inspection.

After completion and component merging, a connectivity check filters objects assigned directly to masked roots. An object must connect to its support, either directly or through accepted objects sharing that support, within approximately 3% of the image’s shorter dimension. Disconnected objects are removed before fixing the inventory. The check uses completed masks when available, except for border completion, whose extended canvas requires the original in-frame mask. Objects supported by other objects or unmasked roots are exempt from this check.

## B.2 Metric initialization and support-ordered settling

Metric frame and relation validation. MoGe-2 [42] estimates metric depth, camera intrinsics, normals, and a dense point map. A RANSAC fit to the main support estimates the upward direction. The fit starts from the localized support component and includes compatible coplanar fragments. A direction-consensus step also considers planar support components and planes on directly supported objects, reducing the efect of a contaminated support mask. The mean predicted support normal is a fallback when a point-based fit is unreliable. Small estimated camera rolls below $2 . 5 ^ { \circ }$ are removed in the monocular path; calibrated depth inputs retain their measured roll.

For an upward unit vector u and a support-plane point $p _ { 0 }$ , the canonical transform satisfies

$$
R _ { g } u = e _ { z } , \qquad p ^ { \mathrm { c a n } } = R _ { g } p + t _ { g } , \qquad t _ { g } = ( 0 , 0 , - e _ { z } ^ { \top } R _ { g } p _ { 0 } ) ^ { \top } .\tag{3}
$$

Thus gravity is along −z and the main support lies at $z = 0$ . The same transform is applied to the camera. If the declared main support has no usable mask, another supported horizontal root is considered; if none is measurable, a fallback level ground is placed below the observed object bases. These fallback choices are recorded rather than treated as measured support planes.

Root-plane reliability depends on both RANSAC inliers and image coverage, so a narrow wall sliver cannot provide a strong orientation constraint. Reliable near-horizontal normals are leveled and near-vertical walls are made plumb. Ambiguous or unmeasurable planes remain visual construction tasks, not grounds for deleting a root.

Root relations distinguish flush support–wall contact (against), a joined horizontal/vertical pair (perpendicular), two walls meeting at a vertical edge (corner), and a lower surface supporting the full base of an upper structure (under). We then verify VLM-estimated relationships via monocular geometry. Confirmed relations become required construction constraints; uncertain or conflicting relations remain advisory, and rejected ones are not enforced. The graph stores this status and its evidence along with $E _ { \mathrm { s u p } }$ and $E _ { \mathrm { s u r f } } .$ . Every support ID must resolve to an entity, parent and child records must agree, and support cycles are forbidden.

Object assets and physical properties. SAM3D [6] reconstructs a textured mesh from the full-frame image and the instance mask. Its predicted orientation is transformed into the gravity-aligned frame without reflection. The mesh is centered at the metric position and uniformly scaled by the depth-projected 3D points of that object after outlier filtering. Both the posed mesh and an unposed canonical copy are saved; the latter supplies an alternative upright orientation during settling. Materials are rendered double-sided to avoid transparent-looking open mesh shells.

Following SceneSmith [32], a VLM estimates material and mass from metric dimensions, sealed mesh volume, six mesh renders (four sides, top, and bottom), and a marked crop showing the object in the input image. Friction comes from a material lookup table, not an unconstrained numerical VLM prediction. An implied density above three times the material density triggers one reconsideration; a still-excessive estimate is clamped. Low implied density is only flagged because hollow objects are legitimate. Missing estimates use the documented density and friction defaults. An estimate made at box dimensions $b _ { o } ^ { 0 }$ is rescaled to current dimensions $b _ { o }$ by

$$
m _ { o } ^ { \mathrm { p h y s } } = m _ { o } ^ { 0 } \prod _ { j = 1 } ^ { 3 } \frac { b _ { o , j } } { b _ { o , j } ^ { 0 } } .\tag{4}
$$

This gives the cubic law under uniform scaling. Physical-property overrides retain consistent mass, center-ofmass, and inertia records.

Nearby-view references. We combine SHARP [27] with GPT-Image-2 [30] to render nearby views. SHARP predicts a 3D Gaussian representation once from the input image, using the estimated focal length. We render it at four source-camera ofsets: $( + 3 0 ^ { \circ } , 0 ) , ( - 3 0 ^ { \circ } , 0 ) , ( 0 , + 1 5 ^ { \circ } )$ , and (0, −15<sup>◦</sup>) in azimuth and elevation. The orbit uses the estimated gravity direction and normally pivots about the source viewing axis’s intersection with the main-support plane. A grazing or implausibly distant intersection falls back to the centroid of all objects’ positions projected onto the support plane.

SHARP renders can contain holes, haze, or stretched edges. Thus, we use GPT-Image-2 to edit each rendered image with instructions to repair these artifacts while preserving clear content and viewpoint. This is a mask-free edit of the rendered view, not a text-only request for a new angle and not a paired input-photo edit. We do not composite original pixels back into the generated result. To detect content drift, pixels with SHARP alpha below 0.8, dilated by 1% of the image’s long edge, are excluded from a comparison of raw and edited RGB. The edit is rejected if the mean absolute diference over the remaining observed pixels exceeds 25/255, after resizing it to the raw render’s resolution. Up to three edit attempts are allowed. An unsuccessful view may retain its raw render for diagnosis but is not exposed as a trusted reference.

If SHARP fails, the fallback splats MoGe-2 points into the same cameras and uses masked GPT-Image-2 editing to fill disocclusions. Generated images are visual guidance, not measured ground truth, and their hidden content may be inaccurate. Agents receive the requested render/reference pair on demand; the source-view reference is always the original photograph. The lighting stage uses only that real source view, since generated illumination is not reliable evidence for lighting estimation.

Support-ordered settling. CoACD [44] converts each mesh into compound convex collision shapes. Supports and containers receive finer decomposition to preserve interior cavities. The validated decomposition is reused during pose refinement and export when the mesh difers only by a recoverable transform; geometry changes trigger rebuilding. A persistent PhysX scene processes objects in parent-before-child order under $E _ { \mathrm { s u p } } ,$ with larger footprints prioritized where dependencies permit. This order applies to initial placement, not the pose agent’s investigation sequence. Temporary root slabs provide support until the initialization agent builds root geometry.

Each object is first dropped against its support context, then against the already placed scene. It is released 2 mm above collision-free contact and simulated with all six rigid-body degrees of freedom. For a non-rollable object, excessive tilt triggers a recovery ladder: try the canonical upright mesh orientation, then a guarded physical-property adjustment. The adjustment can flatten a narrow collider base band, move or lower the center of mass, and increase friction and damping, with matching inertia. A sliver-like contact footprint disables this adjustment to avoid manufacturing an implausible balancing pose. If no upright candidate works, the actual fallen resting pose is retained and flagged. Rollable objects instead use horizontal displacement to detect instability; rolling in place is allowed, and center-of-mass stabilization is skipped. Small lateral searches address collisions with neighbors, and a bounded inward nudge can move an object of its support’s rim.

After settling, trimmed iterative closest point (ICP) alignment fits the mesh to the masked metric points. It optimizes horizontal translation, yaw, and uniform scale about the object’s bottom center; the fitted vertical ofset is discarded so simulation determines resting height. The correction must reduce point-to-mesh RMS error by at least 20%. Scale is fixed for equal-size or otherwise scale-locked assets. ICP is skipped after a successful rim or wrong-support escape, where fitting the original points would undo the rescue. A correction needing more than 3 cm of collision clearance is rejected. Otherwise, another drop tests convergence and class-appropriate stability; failure restores the exact pre-ICP pose. An already fallen object is not rejected merely for retaining that same attitude. Accepted collision shapes remain in the scene for subsequent objects. After all objects are placed, a final joint pass releases every object at once in a single free simulation run to rest; the resulting small drift is baked into the saved poses and recorded per object before the scene is imported into Blender.

## B.3 Agentic scene construction

Generator--verifier interaction. Each stage pairs a VLM generator, which edits and renders the Blender scene, with a VLM verifier, which compares the render with the reference images. Both receive the allowed edits for that stage, the object and root-surface records O and R, the relation graph G, and the current Blender scene through a stage-entry snapshot and scene-inspection tools. The snapshot exposes object names, transforms, dimensions, materials, lights, and camera state; the graph supplies stable IDs, supports, descriptions, and root-plane evidence. A stage-specific prompt states which of these quantities may change. The original image remains the visual anchor.

The generator edits scene code, renders the result, and can undo its latest edit. The read-only verifier sees a current render and the generator’s relevant actions and rule evidence. It either approves the stage or returns concrete corrections within the same scope. A retry continues from the shared scene, not a reset scene; initialization also carries forward its construction plan. Feedback from unrelated stages is not repeatedly inserted into the prompt. Budgets bound this loop. Reaching a budget without approval produces a best-efort artifact sent to the subsequent stage. Figure 7 shows the shared-scene progression.

![](images/21f15d9171631b16c21701bf1f62114e510ac76fbef1968f4a50fd932e1e7435.jpg)  
Figure 7 Agentic scene construction. Scene structure, root material, and lighting are refined in a shared Blender scene before object-pose refinement. Each stage pairs a generator with a verifier. The initialization stage’s generator agent additionally has a rules check tool that is required to pass before sending the results to the verifier.

Stage-specific edits. The initialization generator constructs root geometry and placement from reliable fitted planes, the reference image, and $E _ { \mathrm { s u r f } }$ . Plane points and normals constrain position and orientation; the photograph determines shape and finite extent. A frame-clipped mask supplies only a lower bound beyond the clipped edge. The main support is level with its top at $z = 0$ , but its horizontal position, footprint, and yaw remain adjustable. Walls follow their measured horizontal normal and vertical direction through an orthogonal basis. Roots lacking reliable planes are built from their descriptions and the photograph. A required floor-under-table relation calls for connected legs or a base reaching the floor, rather than placing the floor just beneath the tabletop slab.

Every registered root must be represented. Openings can be cut into a wall, with small frame or glass details attached to that wall. A dedicated root-creation tool can register a genuinely missing architectural wall, and only such runtime-added roots can be removed through the corresponding tool. The agent cannot silently delete a preprocessed root or create duplicate unregistered walls. One auxiliary level ground is allowed when needed. Root collision geometry is static, with predefined friction. Coarse materials and lighting are initialized once before later appearance refinement.

Imported object poses are the trusted starting point. Initialization may translate an object or support stack only through a controlled correction tool, in response to a measured contact defect or a clear position mismatch in paired images. It cannot rotate, resize, remesh, or retexture the object. The correction tool tests object integrity, support, and new or worsened collisions, rolls back a failed correction, and requires another rule check after success. Moving an object merely to compensate for a wrongly built support is disallowed.

The appearance stage changes root materials and textures τ<sub>r</sub>, including procedural shader detail, roughness, and bump patterns, while keeping geometry and object assets fixed. The illumination stage changes lights, world illumination, exposure, and view-transform settings, not materials or layout. It compares only sourceview renders with the input image. The source camera stays fixed; alternate views use rendering tools rather than camera edits.

Rule checks and visual feedback. As a special tool for the initialization stage only, the read-only check\_rules tool evaluates three levels in sequence. Structure checks verify imported-object integrity, registered-root coverage, main-support geometry, top height, and required structural relations. Pose checks compare root silhouettes with the reference masks and verify that all objects rest on their supports. Contact checks detect penetration, floating objects, and violated surface relations. Advisory relations do not acquire the force of required constraints. The report identifies the first failing level and its afected entities; later levels are not evaluated and must not be assumed to pass.

The initialization generator revises the scene and reruns the tool after edits. Its voluntary completion requires a current pass and verifier approval. For ambiguous or tiny wall masks, an explicit visibility exception can waive an unreliable mask comparison, but not the requirement to build the root. Passing numerical rules establishes a minimum geometric standard, not visual faithfulness: the verifier still judges table edges, wall placement, gaps, and visible structure. Appearance and lighting have their own visual verifiers without this initialization rule ladder. Their feedback targets root-surface appearance and source-view lighting, respectively.

## B.4 Physics-Grounded Pose Refinement

Object investigation. Pose refinement uses one continuous VLM session, without an external verifier or a fresh-memory retry. The agent interprets visual evidence and selects tools for geometric measurement, pose search, and simulation. Scene-rendering tools provide source, nearby, and bird’s-eye views. The investigation tool returns paired, labeled photo/render crops for one object or a small spatially adjacent group. Small objects are inspected alone or in pairs so that their directional features remain visible. Investigation makes only the listed IDs eligible for bounded moves; the next investigation replaces that set, while code edits and undo clear it.

The Investigate tool reports silhouette overlap and separate evidence for displacement, projected size, fine yaw, and front–back orientation. Comparing raw and centroid-aligned overlap helps detect displacement. Projected size compares radial mask extents, or major-axis extents for elongated objects; a large position error or incompatible aspect ratios suppress an unreliable size suggestion. Yaw is measured from silhouette principal axes only when the axis is suficiently distinctive. Missing evidence means unmeasured, not correct. The agent checks the paired crops and distinguishes a depth error from a true size error before resizing.

Front–back evidence compares the current crop and a 180<sup>◦</sup> yaw-flipped render with the photograph using DINOv2 [31] patch features. Weak and strong reversal margins are reported separately; nearly tied or unavailable features do not establish an orientation. For elongated objects, a significant fine-yaw error postpones the flip suggestion until comparable crops are available. When an accepted occlusion-completed reference exists, a known foreground object resting on the target can be hidden in the render and paired with its removal-edited reference. Uncertain neighbors and objects being investigated together remain visible.

![](images/6eaf1dd64c0909f14c667047bf81b9d98c738f7ab2d33a284a53f85b28360b8c.jpg)  
Figure 8 Pose refinement and export. The agent investigates objects and requests edits. The Move tool searches and physically validates bounded moves; the Edit tool simulates direct edits for agent review. After a final gravity pass, object poses and physical properties are exported to a simulation-ready USD file.

Bounded candidate search. The agent selects an object and a move aspect a from six choices: horizontal translation along x, along y, or jointly in xy; uniform scaling; yaw adjustment; or a $1 8 0 ^ { \circ }$ yaw flip. The flip follows the special procedure described below. For routine bounded moves, the agent does not specify the adjustment magnitude or search direction. The Move tool evaluates a coarse candidate set and then finer candidates around the winner. Translation step sizes are proportional to object size, so larger objects are tested with larger shifts. For joint x–y moves, the tool compares the rendered object mask with the reference mask and uses their ofset to propose a translation. Scale and yaw searches can similarly use measured size and orientation diferences to propose candidates. Before rendering each candidate, the tool resolves overlap with the support and adjusts the object’s height to place it on the geometry below. These geometric placement operations do not run a physics simulation. Candidates are ranked using the alignment scores below; only the selected candidate proceeds to physical settling and acceptance checks.

Aspect-specific alignment scores. Let $m _ { o }$ be the reference mask and $\widehat { m } _ { o } ( X )$ the rendered silhouette of object o at transform X. We write $\mathrm { I o U } _ { o } ( X ) = \mathrm { I o U } ( \widehat { m } _ { o } ( X ) , m _ { o } ) ; \mathrm { c I o U } _ { o } ( X )$ aligns the masks’ area-weighted centroids before comparison. Both use coverage-weighted masks, preserving fractional alpha values and thin structures. The score for aspect a is

$$
A _ { o } ^ { ( a ) } ( X ) = \left\{ \begin{array} { l l } { \mathrm { I o U } _ { o } ( X ) - 0 . 4 D _ { o } ( X ) - 0 . 3 Q _ { o } ( X ) , } & { a = \mathrm { t r a n s l a t i o n } , } \\ { \mathrm { c I o U } _ { o } ( X ) - 0 . 4 D _ { o } ( X ) - 0 . 3 Q _ { o } ( X ) , } & { a = \mathrm { s c a l e } , } \\ { \mathrm { c I o U } _ { o } ( X ) + 0 . 2 F _ { o } ( X ) , } & { a = \mathrm { y a w } . } \end{array} \right.\tag{5}
$$

Here, $D _ { o } ( X ) = \vert d _ { o } ( X ) - d _ { o } ^ { \mathrm { r e f } } \vert / \vert d _ { o } ^ { \mathrm { r e f } } \vert$ measures relative depth error. $d _ { o } ( X )$ is the depth of the object’s center after the candidate translation, and $d _ { o } ^ { \mathrm { r e f } }$ is the median depth of the masked points of that object. The size penalty $Q _ { o } ( X ) = \left| \sqrt { | \widehat { m } _ { o } ( X ) | / | m _ { o } | } - 1 \right|$ compares projected mask areas. Centering the masks for scale and yaw reduces interference from translation errors; depth and projected size constrain translation and scale. Yaw additionally uses appearance to distinguish orientations with similar silhouettes. Specifically,

$$
F _ { o } ( X ) = \frac { 1 } { | \mathcal { P } _ { o } | } \sum _ { p \in \mathcal { P } _ { o } } \frac { f _ { p } ^ { \mathrm { r e n d e r } } ( X ) ^ { \top } f _ { p } ^ { \mathrm { p h o t o } } } { \| f _ { p } ^ { \mathrm { r e n d e r } } ( X ) \| _ { 2 } \| f _ { p } ^ { \mathrm { p h o t o } } \| _ { 2 } } ,\tag{6}
$$

where $f _ { p }$ is a DINOv2 ViT-S/14 patch token from an RGB object crop and $\mathcal { P } _ { o }$ indexes reference-foreground patches in matched-size crops. If no foreground patch survives downsampling, the full patch grid is used. The scoring window is the reference mask’s box with 25% padding, and the isolated render is composited on neutral gray. The front–back comparison crops each object to its own box on white and resizes to 224 × 224, reducing sensitivity to position and scale. Without DINOv2 features, yaw uses silhouette scoring and supplies no appearance evidence for orientation.

Simulation and acceptance. Let $X _ { o }$ be the pre-edit transform, $X _ { o } ^ { \prime }$ the proposal tested for acceptance after any reduction to a feasible fraction, and $\widehat { X } _ { o }$ its simulated resting pose. The post-settling score $\widetilde { A } _ { o } ^ { ( a ) } ( \widehat { X } _ { o } )$ recomputes rendered terms while holding the tested proposal’s depth penalty fixed. Let $P _ { o }$ indicate that the applicable proposal-level yaw check and post-settling physical checks pass. For routine bounded moves, the update is

$$
X _ { o } \gets \left\{ \begin{array} { l l } { \widehat { X } _ { o } , } & { A _ { o } ^ { ( a ) } ( X _ { o } ^ { \prime } ) - A _ { o } ^ { ( a ) } ( X _ { o } ) \geq \epsilon _ { a } , } \\ & { P _ { o } = 1 , \quad \widetilde { A } _ { o } ^ { ( a ) } ( \widehat { X } _ { o } ) \geq A _ { o } ^ { ( a ) } ( X _ { o } ) , } \\ { X _ { o } , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{7}
$$

The aspect-specific threshold $\epsilon _ { a }$ suppresses small proposed gains; after settling, retaining the baseline score is suficient. Table 8 lists the gain and physical thresholds. Collision checks compare pre- and post-edit penetration, rejecting substantial new or worsened overlap; pre-existing overlap is still subject to the final whole-scene rule gate. Non-rollable objects are checked for capsize, while rollable objects are checked for excessive horizontal drift.

A failed candidate can be reduced to a smaller feasible fraction, which must again clear the gain threshold before simulation. Reliable yaw evidence adds an improvement test on the selected proposal, with a small noise tolerance. If appearance features become unavailable during rescoring, their term is removed from both scores in the post-settling comparison. Rejection restores all afected transforms, including those of supported objects.

Support propagation and exceptional edits. Equation (7) applies to routine bounded searches. The 180<sup>◦</sup> flip is a special semantic correction: it need not improve IoU, but is physically tested and must be followed immediately by a fresh paired investigation or undo. A retained or undone applied flip consumes its one-shot allowance; one physically rejected attempt can be retried.

When moving a support parent, the Move tool compares leaving children in place with carrying them. Children follow if the free-child simulation would topple them or if carrying improves the group’s mean overlap. A resize instead settles the resized parent and then reseats its children and contact-dependent neighbors.

When bounded moves repeatedly fail or cannot express a correction, the code-edit tool Edit allows direct layout changes, including coordinated edits. The scene-code editing tool detects changed objects, resolves initial overlaps, and settles afected supports before their dependents in simulation. It then returns the post-simulation render and physical diagnostics for the agent to decide whether to retain or undo the simulated result. Automatic alignment and physical acceptance checks govern bounded moves; the agent decides whether to retain simulated code edits. If the required simulation fails to execute, the raw edit is rolled back. Undo restores the pre-edit scene snapshot rather than using approximate inverse transforms.

Context management and prompt caching. The agent retains the initial instructions, target image, and stage-entry scene snapshot. Recent rounds retain their full text, while older rounds are compressed in blocks of four into a chronological log, normally leaving five to eight recent rounds. The log preserves actions, scores, acceptance/rejection, and undo outcomes, but removes old images and stale pose suggestions. A compression boundary is moved backward if it would separate an edit from its undo, so the recent window can occasionally grow beyond eight rounds.

![](images/a72e8a8266a99e42edf9d64bc0484da72605ca9989cb445702a19f8492cbfba9.jpg)  
Figure 9 Prompt reuse with fresh scene evidence. In this example, tool call #11 reuses the unchanged prefix of tool call #10 (green), appends the cup/saucer investigation results (amber), and refreshes the object state and budget after the cache boundary (purple). The crops come from the recorded run; the layout illustrates the context policy.

Images are also aged within the visible history. Superseded full-scene renders, alternate views, and crop messages become text references to their saved artifacts; paired crop/reference messages are retained or removed together. Duplicate source photographs need not be reattached because the initial target remains protected. Aging occurs at the same block boundaries, not on every round, so it does not continually rewrite the prompt prefix. The full stored history remains intact; compression changes only what is sent to the VLM. The shorter initialization, appearance, and lighting conversations use append-only history rather than this pose-specific sliding window.

A per-object state table is rebuilt from the full interaction history. It records the latest measured scores, completed or pending follow-up checks, tool-use counts, rejected aspects, and exhausted budgets. The table and remaining-round budget sit after the cache breakpoint, at the changing end of the prompt. Their updates therefore preserve the reusable prefix. Block-aligned compression and image aging allow prompt-cache hits across consecutive calls without freezing the current object state. Figure 9 illustrates this mechanism.

Across all 80 benchmark scenes, 84.1% of pose-refinement input tokens are cache reads, measured over 2,112 logged calls. We divide summed cache-read tokens by total input tokens, including uncached inputs and cache writes in the denominator and excluding output tokens.

Completion and export. The completion tool, End, first checks that every retained object has been investigated.

A coverage failure stops this check before the expensive penetration pass, so it does not imply collision clearance. Once coverage passes, unresolved penetration must be corrected, and any post-flip follow-up must be complete. Voluntary completion and a current rule pass are both required. A budget-limited session is explicitly marked incomplete.

After refinement, one joint gravity simulation releases all objects together. Its resulting transforms are baked into the saved Blender scene; no object is pinned merely to conceal a final topple. Non-convergence, large drift, or toppling is recorded as a warning, and does not by itself establish dynamic simulation readiness. Appendix B.5 details USD conversion and its separate readiness check.

## B.5 Implementation settings and simulator export

Model roles and execution settings. Table 7 lists the models used in our system and their roles. In the backbone comparison (Table 12), we switch all configurable VLM roles to the selected backbone, including preprocessing, physical-property estimation, scene-construction generators and verifiers, and pose refinement. Table 8 reports implementation defaults, which can be overridden by a run configuration.

Table 7 Models and their roles. The two image-editing paths have diferent inputs and purposes; generated references never replace the real source photograph.
<table><tr><td>Component</td><td>Role and conditioning</td></tr><tr><td>Configurable VLM</td><td>Category/instance proposals, semantic reviews, physical-property esti- mates, and stage agents; launcher default: claude-opus-5.</td></tr><tr><td>MolmoPoint-8B / SAM3</td><td>Instance-description localization / category, synonym, crop, and point- conditioned masks.</td></tr><tr><td>MoGe-2 / SAM3D</td><td>Metric camera and point map / textured per-object mesh.</td></tr><tr><td>LanPaint + Qwen-Image-Edit-2509</td><td>Masked occluder removal and one-sided outpainting.</td></tr><tr><td>LingBot-Depth</td><td>Edited RGB, surviving metric depth, and intrinsics; ViT-L/14 v0.5 pretrained checkpoint.</td></tr><tr><td>SHARP + GPT-Image-2</td><td>Gaussian novel-view rendering followed by image repair; point- splat/masked-edit fallback.</td></tr><tr><td>DINOv2 ViT-S/14</td><td>Foreground patch similarity for rotation and front-back evidence.</td></tr></table>

Simulator Export. USD conversion is a separate step after Blender reconstruction. It bakes procedural materials and transfers meshes, textures, lights, and camera state, then assigns collision geometry and physical properties. An identity map links each pipeline object to its USD primitive, accounting for name sanitization. Missing, ambiguous, or nested object identities fail conversion rather than losing bodies. Mass and inertia are adjusted consistently for the exported geometry. A simulator-side verification checks the exact object inventory, dynamic versus kinematic state, support contact, convergence, translation and rotation drift, and final velocities. Reconstruction completion, conversion success, and dynamic readiness are recorded separately.

## C Additional Experimental Details

## C.1 Baseline implementation details

We adapt the baseline implementations to the same single-image benchmark. Each method retains its reconstruction pipeline and scene representation; method-specific adapters expose the reconstructed objects and evaluation renders. Below we describe these adaptations and, in particular, how a source camera and a nearby evaluation camera are obtained.

VIGA. We use VIGA’s static-scene configuration with SAM-based segmentation and SAM3D asset initialization, followed by iterative Blender code generation, rendering, and verifier feedback. To support object-centric evaluation and simulation, we augment the generator’s system prompt with a simple object-grouping contract (one parent per movable object with every part beneath it, one parent for the static environment, no loose top-level meshes), and after every code execution the tool result lists the meshes that are not yet grouped, so the agent can repair them before it stops. We use a 20-round generator budget with medium reasoning efort for every backbone; the same VLM also names the segmented objects. Grouping changes neither the renderer nor the assets, and the agent complied in every scene. Intermediate renders preserve the input aspect ratio with a 512-pixel long edge. For evaluation, we select the latest numbered render directory containing a rendered frame and load its saved Blender scene. We select the main camera by name, falling back to the camera named Camera and then an available camera, and retain its saved pose and intrinsics. This avoids using a diagnostic camera left active. Evaluation renders use the input-image dimensions.

Table 8 Key implementation defaults. Physical lengths are in meters unless otherwise stated.
<table><tr><td>Setting</td><td>Default</td></tr><tr><td>Input / image-edit resolution</td><td>RGB input long edge 1536 pixels; GPT image-edit input long edge at most 2048 pixels.</td></tr><tr><td>Root plane fitting</td><td>200 RANSAC trials, at most 5000 points, 0.01 inlier distance; reliable root planes require inlier fraction ≥ 0.6 and image coverage ≥ 2%.</td></tr><tr><td>Occluder image editing</td><td>20 diffusion steps, guidance 4, seed 0; one LanPaint Langevin step per diffusion step. Concavity 0.05 / 0.02; at most 32 / 64 parts; a capped support can be retried with</td></tr><tr><td>Ordinary / support CoACD</td><td>128 parts. Ordinary resolution 50; support resolution  $\mathrm { c l i p } ( \lceil \ell _ { \mathrm { m a x } } / 0 . 0 0 3 \rceil , 1 0 0 , 2 0 0 )$  where  $\ell _ { \mathrm { m a x } }$  is mesh maximum extent.</td></tr><tr><td>Physics material fallback Settlement</td><td>Density 250 kg/m³; contact offset 0.002, rest offset 0. 120 Hz; 600-step ordinary or 240-step local budget, with a further 600 steps for an unconverged drop.</td></tr><tr><td>Rest detection</td><td>After at least 30 steps, 20 consecutive steps with translation  $< 5 \times 1 0 ^ { - 5 }$  m and rotation &lt; 0.0005 radians per step.</td></tr><tr><td>ICP</td><td>At most 60 iterations, ±30° yaw, scale 0.85–1.18, at least 150 target points; scale locked for equal-size groups.</td></tr><tr><td>Move coarse probes</td><td>Axis translation ±{0.05, 0.15, 0.4} times object size; yaw  $\pm \{ 5 , 1 5 , 4 0 \} ^ { \circ }$  scale {0.8, 0.9, 1.1, 1.25}, plus measured proposals and fine probes.</td></tr><tr><td>Move gain thresholds</td><td>Translation 0.01; scale 0.01, or 0.005 with an active size hint; centered rotation score 0.012.</td></tr><tr><td>Move stability tests</td><td>Ordinary capsize threshold 45°; rollable horizontal drift 50 mm. Penetration fails</td></tr><tr><td>Agent round budgets</td><td>when depth exceeds 8 mm and worsens by more than 5 mm. Initialization 20, appearance 10, lighting 10; verifiers 8 each; up to 2 attempts per</td></tr><tr><td>Per-object pose budgets</td><td>paired stage. Pose refinement: one session of min  $( 7 0 , 8 + 5 | \mathcal { O } | )$  rounds. 3 investigations and 5 moves; code relocations have a soft cap of 3.</td></tr></table>

SceneConductor. We run the three-stage pipeline: mask review and merging, SAM3D asset generation and GALP layout initialization; environment construction; and relation-guided scene refinement, including refinement of selected object groups in isolated subscenes. Its source camera uses GALP’s predicted focal length and orientation in the camera-centered reconstruction frame, with the pipeline’s coordinate-convention correction. We render the final scene at the input-image dimensions, preserving its aspect ratio instead of using the default fixed-size diagnostic views. Walls that obstruct the evaluation camera are hidden from camera rays. The nearby-view adapter also uses a gray world background with unit-strength ambient illumination.

SimFoundry. We use the pipeline’s single-image mode: the input is supplied as one RGB frame, with a repeated-frame video only to satisfy the video input interface. Reconstruction uses DA3 depth and camera estimates, support-plane alignment, object extraction and image cleanup, Hunyuan3D asset generation, FoundationPose refinement, and physics preparation and settling before OmniGibson scene assembly. The source camera combines the estimated intrinsics with the camera-to-world transform from support-plane alignment. The single-image configuration uses the simulator’s planar support surface. Pairwise evaluation renders use OmniGibson at a 1536-pixel long edge with the input aspect ratio; the simulator’s added robot is hidden.

REST3D and source-camera recovery. We evaluate REST3D’s final object reconstruction after its scene assembly and stabilization stages. Its exported scene does not retain a calibrated source camera, so our evaluation adapter recovers an approximate camera from the input image and the reconstruction intermediates. We rerun the same MoGe model used by its asset pipeline to estimate image intrinsics. To recover the pose, we fit the reconstructed raw floor/carpet geometry with a plane, use its normal to recover the pipeline’s up-axis alignment, and propagate the raw camera origin, forward direction, and up direction through that alignment. We then apply the support-plane height shift and the conversion from the pipeline’s Y-up frame to the evaluation Z-up frame. This recovery is approximate because REST3D’s subsequent per-object grounding and stabilization are not a single rigid scene transform. The camera is derived geometrically from these intermediates, without selecting it by render similarity or using a ground-truth simulator camera. The final meshes are converted to USD with vertex colors retained and rendered in Isaac Sim at the input resolution, using the recovered horizontal focal length, square pixels, and a centered principal point. A synthetic support slab is added because the raw floor reference is omitted from the final object export. For the shared stability test, REST3D’s reconstructed tables and desks and its one wall-mounted object are static colliders; every other REST3D mesh is a dynamic rigid body on the support slab.

SAM3D object baseline. This comparison shares SceneRig’s segmentation and generated object assets. We retain SAM3D’s predicted rotation, translation, and scale, applying only the camera-coordinate conversion and the shared gravity alignment to place them in the scene frame. We bypass SceneRig’s depth-based recentering and rescaling, registration, settling, and subsequent object-pose refinement. For rendering, these raw-pose assets replace the objects in SceneRig’s final scene while retaining its source camera, support surfaces, walls, and lighting. Thus this is an object-reconstruction and placement baseline with shared scene context.

Source and nearby view selection. Source-view metrics use the camera supplied or recovered for each method as described above. For pairwise judging, we render a nearby view by rotating the source camera position c by $+ 3 0 ^ { \circ }$ around a pivot p and the reconstruction’s vertical axis:

$$
c _ { \mathrm { n e a r } } = p + R _ { z } ( 3 0 ^ { \circ } ) ( c - p ) .\tag{8}
$$

The focal length and the camera’s height and distance relative to p are preserved. Each nearby view is thus a corresponding viewpoint change in the reconstruction’s own frame, not a shared ground-truth extrinsic, and is used only for evaluation.

The pivot p is where the source optical axis meets a horizontal reference plane. REST3D and SimFoundry use the support plane $z = 0 ;$ SceneConductor uses its estimated floor height; VIGA uses the median height of visible mesh bounding-box centers. If the intersection is invalid, REST3D falls back to its scene bounding-box center, SceneConductor to the placed-object centroid, and VIGA to the component-wise median of the visible mesh bounding-box centers (also used when the intersection lies too far outside the scene); the REST3D and SceneConductor fallbacks are projected onto their reference planes. Because each camera is aimed at its pivot, a fixed orbital elevation does not guarantee identical roll, or identical pitch when a fallback pivot is used.

The source and nearby images are presented to the judge at common tile sizes, preserving aspect ratio, together with the original input photograph (Appendix C.3). Rendering remains method-specific: VIGA and SceneConductor use Blender/Cycles, REST3D uses Isaac Sim, and SimFoundry uses OmniGibson. The scored REST3D and SimFoundry renders use their brown synthetic support surfaces.

## C.2 Benchmark and metric details

Benchmark composition. The 80-scene benchmark comprises 44 simulator-grounded scenes and 36 additional scenes whose input images consist of 18 real robot workspace images, 6 phone captures, 8 internet images, and 4 AI-generated images.

Image and instance metrics. Full-frame metrics compare the source-camera render with the input image. LPIPS uses AlexNet features; nCLIP is one minus the cosine similarity of CLIP ViT-B/32 image embeddings. The reported PSNR uses raw RGB errors, without fitting an exposure or color correction. Object-only PSNR pools squared errors over ground-truth object pixels, excluding the support table. Object-only SSIM averages the spatial SSIM map over the same pixels.

For object-centric LPIPS and nCLIP, ground-truth instance masks define crop boxes shared by the reference and render. Boxes have 10% padding and a minimum side of 64 pixels; objects with fewer than 200 visible pixels are excluded from crop metrics. Within each box, pixels outside the object’s ground-truth mask are set to white in both images, so these metrics score the object without its surroundings. Crop scores are averaged over objects within a scene.

Ground-truth mesh parts are grouped into logical objects before instance evaluation. ARI compares the two instance partitions over all pixels, with background, including the table, treated as one cluster. FG-IoU is the intersection over union between the predicted and ground-truth object foregrounds, each the union of its object pixels with the table excluded. Thus ARI measures partitioning while FG-IoU measures detection coverage: missed, hallucinated, or misplaced objects lower FG-IoU even when the remaining pixels are partitioned consistently. Count MAE is the mean absolute diference between the predicted and ground-truth object counts in the evaluated instance masks. For final-scene evaluation, the predicted count includes only objects visible in the source-view render, rather than all objects stored in the reconstructed scene.

Canonical 3D layout evaluation. Ground-truth and predicted positions are expressed in corresponding canonical frames whose origins are the source viewing axis’s intersections with the support plane. The vertical axis is gravity-up, and the horizontal orientation is fixed by the source-camera direction. If either camera cannot define this intersection, each set instead uses the centroid of all objects’ positions projected onto the support plane. For both predictions and ground truth, object positions are defined by world-space axis-aligned bounding-box centers and then transformed into the canonical frame. All predicted objects contribute to the metric, including unmatched extras. We evaluate object arrangement up to a global scale factor, allowing comparison across reconstruction methods with diferent scale conventions.

We fit only a global scale. Candidate scales include one and ratios of the two sets’ pairwise-distance quantiles at 25, 50, 75, and 90%. Hungarian matching at a 20 cm cutof ranks candidates first by match count and then by mean matched distance. Up to two least-squares refinements use $\begin{array} { r } { s = \sum _ { i } p _ { i } ^ { \top } g _ { i } / \sum _ { i } \| p _ { i } \| _ { 2 } ^ { 2 } } \end{array}$ for matched predicted and true positions $( p _ { i } , g _ { i } )$ . Refinement requires at least three matches and cannot reduce their count. This scale is shared across all reported cutofs.

For point sets P and Q with $m \leq n$ elements and $n > 0$ , OSPA of order one is

$$
\mathrm { O S P A } _ { c } ( P , Q ) = \frac { 1 } { n } \left[ \operatorname* { m i n } _ { \pi } \sum _ { i = 1 } ^ { m } \operatorname* { m i n } \big ( c , \| p _ { i } - q _ { \pi ( i ) } \| _ { 2 } \big ) + c ( n - m ) \right] ,\tag{9}
$$

where π assigns the smaller set to distinct members of the larger. Two empty sets have zero distance. It penalizes localization error and unmatched objects [36]. We report 5 and 10 cm cutofs and F1 from a separate 10 cm-threshold assignment on these aligned positions.

## C.3 Pairwise VLM and physics protocols

For pairwise visual evaluation, GPT-5.6-terra and Qwen3.5-122B-A10B [34] compare reconstruction pairs for completeness, layout, and appearance. The nearby evaluation view uses the $3 0 ^ { \circ }$ orbit and method-specific camera recovery described in Appendix C.1, keeping the camera’s orbital elevation and distance fixed. Each candidate panel shows its source and nearby render, with anonymous labels and uniform tile sizes; the judge also sees the real input photograph. Each pair is presented in both candidate orders. We remove orderinconsistent judgments, give ties half credit, and fit Bradley–Terry scores to summarize pairwise preferences. Half a pseudo-win is added in each direction for every compared method pair to avoid infinite strength estimates for undefeated methods. Scores are centered to have zero mean across methods for each criterion. After excluding comparisons for which either judge is order-inconsistent, the two judges achieve Cohen’s $\kappa = 0 . 8 7$ . Order-inconsistent verdicts, discarded as above, make up 13–23% of paired judgments per criterion.

Physical stability is evaluated separately by running each reconstructed scene for 300 PhysX steps at 60 Hz. Displacement is the Euclidean distance between an object’s initial and final position, not accumulated travel along its path. An object passes the stability test if its displacement after 5 s is below 50 mm and its final velocity is zero; a scene passes only if every object passes. The scene-level rate is the fraction of passing scenes, while the object-level rate pools passing objects across scenes.

## C.4 Real-to-sim policy evaluation protocol

This section details the two policy evaluations in Section 5.1. Table 6 reports $\pi _ { 0 . 5 }$ and Table 9 reports LingBot-VLA 2.0; both policies are fine-tuned on the same in-house teleoperation data and evaluated on separate sets of 50 real episodes across the same five tasks: ten episodes per policy–task pair, totaling 100 real episodes. Each episode is evaluated in both reconstruction methods with its corresponding policy checkpoint. The five tasks evaluated are: everything into the bin, fruit on plate, fruits from plate to bowl, marker in cup, and mustard in bin.

Scene export. Each scene is reconstructed by SceneRig (or baseline) from an armless head-camera image and exported to Isaac Sim together with the calibrated camera rig of the station, the episode’s initial arm configuration (reset pose), and the episode’s text instruction. Unlike the RGB-only benchmark of Section 4, these station scenes provide metric depth: both SceneRig and SimFoundry receive the head camera’s color-aligned depth map, hole-completed with LingBot-Depth [39], together with the calibrated intrinsics, and use it in place of monocular depth estimation. Both methods use their own mass estimates for objects, and the friction coeficient is derived from the estimated material; contacts use the larger friction coeficient of the two bodies. The gripper fingertips use signed-distance-field collision shapes with a friction coeficient of 8 (static) and 7 (dynamic). We choose these friction values following manipulation simulation benchmarks such as RoboDojo [5] and RoboLab [48], so that objects can be reliably grasped in simulation.

Closed-loop rollout. Physics is stepped at 240 Hz, and control runs at 30 Hz. Each action chunk is executed for 50 control steps (1.67 s) before the next policy query; the episode is therefore closed loop, unlike the open-loop trajectory replay in Appendix C.5. An episode ends after 360 s, once both arms have been stationary for 10 s (fingertips and gripper joints moving less than 5 mm), or once no object has moved more than 5 mm for 60 s, counted from the first object motion. The same settings are used for the SceneRig and the SimFoundry rows.

Scoring. Simulated success is a task-specific geometric predicate on the final object centroids relative to the container at its final position. Plates and bowls use a circle fitted to the outer rim, cups and mugs a circle fitted to the wall (which excludes a mug’s handle), and bins the axis-aligned box of the bin mesh. Required objects are defined by the real scene: the fruit tasks require every fruit, mustard in bin requires the mustard bottle, and everything into the bin requires every object that did not start inside the bin. A required object missing from the reconstruction counts as a failure, and for everything into the bin, hallucinated objects must also be placed in the bin. A human scorer verifies the verdicts generated by the script. Real success is the operator’s verdict.

Metric aggregation. For each policy, agreement and Cohen’s κ use its 50 episodes, and Pearson’s r correlates its five real and simulated task success rates. Combined agreement pools all 100 episodes. The pooled correlation is computed directly from all ten policy–task success-rate pairs. MMRV [21] averages, over policy–task pairs, the largest real success-rate gap to any pair whose simulated order contradicts the real order; lower is better. Confidence intervals come from 10,000 paired bootstrap resamples of episodes within each policy–task cell, and McNemar’s exact test compares the two methods’ per-episode agreement with the real verdicts. With five tasks per policy, the rate-level metrics (r, MMRV, and rate error) are reported without intervals.

MMRV. We follow SIMPLER’s published formula [21], using strict < comparisons. For K paired real and simulated success rates $r _ { i }$ and $s _ { i }$

$$
V _ { i j } = | r _ { i } - r _ { j } | \mathbf 1 [ ( s _ { i } < s _ { j } ) \neq ( r _ { i } < r _ { j } ) ] , \qquad \mathrm { M M R V } = \frac 1 K \sum _ { i = 1 } ^ { K } \operatorname* { m a x } _ { 1 \leq j \leq K } V _ { i j } .
$$

Across the ten pooled policy–task pairs, MMRV is 0.04 for SceneRig and 0.11 for SimFoundry (Table 10);   
lower is better.

SimFoundry inference. For the SimFoundry rows, the reconstructed objects are replaced by SimFoundry’s meshes, textures, poses, and masses, and registered into the station frame through the shared head camera. Since SimFoundry does not reconstruct table mesh, we borrow it from our method and paint the table with SimFoundry’s inpainted table image.

Table 9 Real-to-sim evaluation with LingBot-VLA 2.0. The same fine-tuned checkpoint is evaluated on 50 real episodes and both reconstructions. Success counts successful episodes; agreement counts matching real and simulated verdicts. Cohen’s κ uses all 50 episodes; Pearson’s r uses the five task-level success rates.
<table><tr><td>Environment</td><td></td><td>Everything→bin</td><td>Fruits→plate</td><td>Fruits→bowl</td><td>Marker→cup</td><td>Mustard→bin</td><td>All</td><td>| Cohen&#x27;s κ ↑ Pearson r ↑</td><td></td></tr><tr><td>Real-world</td><td>success</td><td>5/10</td><td>7/10</td><td>2/10</td><td>3/10</td><td>3/10</td><td>20/50</td><td>1 一</td><td></td></tr><tr><td rowspan="2">SimFoundry</td><td>success</td><td>4/10</td><td>4/10</td><td>0/10</td><td>0/10</td><td>2/10</td><td>10/50</td><td></td><td>0.88</td></tr><tr><td>agreement</td><td>5/10</td><td>5/10</td><td>8/10</td><td>7/10</td><td>5/10</td><td>30/50</td><td>0.09</td><td></td></tr><tr><td rowspan="2">SceneRig</td><td>success</td><td>5/10</td><td>6/10</td><td>1/10</td><td>1/10</td><td>0/10</td><td>13/50</td><td></td><td>0.93</td></tr><tr><td>agreement</td><td>6/10</td><td>7/10</td><td>7/10</td><td>6/10</td><td>7/10</td><td>33/50</td><td>0.25</td><td></td></tr></table>

Table 10 Statistical analysis of real-to-sim agreement. Agreement and Cohen’s κ use per-episode verdicts, with 95% paired-bootstrap intervals. McNemar counts the episodes on which only SceneRig or only SimFoundry agrees with the real verdict. Pearson’s r, MMRV, and rate error (the mean absolute diference between real and simulated task success rates) use task-level success rates and are descriptive.
<table><tr><td>Policy</td><td>Method</td><td>Agreement ↑</td><td>Cohen&#x27;s κ ↑</td><td></td><td>Pearson r ↑ MMRV ↓</td><td>Rate error ↓ |</td><td>McNemar</td></tr><tr><td>π0.5</td><td>SimFoundry SCENERIG</td><td>48% [36, 62] 72% [60, 84]</td><td>−0.07 [−0.32, 0.18] 0.38 [0.11, 0.63]</td><td>-0.38 0.70</td><td>0.12 0.00</td><td>0.16 0.12</td><td>18 vs. 6, p = 0.023</td></tr><tr><td>LingBot-VLA 2.0</td><td>SimFoundry SCENERIG</td><td>60% [46, 74] 66% [52, 78]</td><td>0.09 [−0.16, 0.35] 0.25 [0.00, 0.50]</td><td>0.88 0.93</td><td>0.06 0.04</td><td>0.20 0.14</td><td>7 vs. 4, p = 0.55</td></tr><tr><td>Pooled</td><td>SimFoundry SCENERIG</td><td>54% [45, 63] 69% [60, 78]</td><td>0.09 [−0.08, 0.27] 0.38 [0.20, 0.56]</td><td>0.84 0.92</td><td>0.11 0.04</td><td>0.18 0.13</td><td>25 vs. 10, p = 0.017</td></tr></table>

Agreement across policies. For LingBot-VLA 2.0, SceneRig achieves 66% episode agreement (κ = 0.25) versus 60% $( \kappa = 0 . 0 9 )$ for SimFoundry, with task-level correlations of r = 0.93 and r = 0.88, respectively. Both underestimate real-world success: SceneRig succeeds on 13/50 episodes and SimFoundry on 10/50, versus 20/50 on the real robot. The agreement advantage varies by task; SimFoundry agrees more often on fruits-to-bowl and marker-to-cup. Across both policies, SceneRig matches 69/100 real outcomes, versus 54/100 for SimFoundry. Figure 4 combines their ten policy–task success-rate pairs, yielding pooled correlations of 0.92 and 0.84. These correlations capture variation across both policies and tasks; the tables report each policy separately.

Statistical analysis. Table 10 adds uncertainty estimates and paired tests. Pooled over both policies, SceneRig’s agreement advantage is significant (25 vs. 10 discordant episodes, McNemar $p = 0 . 0 1 7 )$ , and its κ exceeds SimFoundry’s by 0.29 (95% CI [0.08, 0.50]).

## C.5 Open-loop trajectory replay protocol

This section details the open-loop trajectory replay experiment in Section 5.2. Unlike the evaluation in Section 5.1, which queries the policy between action chunks, this experiment executes an entire command sequence recorded on the real robot without additional policy inference.

Episodes. We evaluate 25 successful teleoperated demonstrations recorded on a bimanual station of the same design, with arm-base and head-camera poses calibrated using a ChArUco board before recording. There are five demonstrations per task: everything into the bin, cup into the bowl, fruits from the plate into the bowl, marker into the cup, and mustard bottle into the bin. The demonstrations involve 1–6 evaluated objects each. We replay the recorded follower’s joint trajectory, logged at 30–60 Hz depending on the session, rather than the leader commands, so that the simulated arm follows the motion observed in the recording.

Scene construction and registration. Both methods receive the same armless head-camera image and metric depth input as in Appendix C.4. For each episode, we import the reconstructed textured geometry and estimated physical properties into Isaac Sim. We align the reconstructed scene to the station frame using the calibrated head-camera pose. The simulated rig uses the measured robot-base poses and the camera’s measured intrinsics. Collision geometry is derived from the reconstructed object meshes, the tabletop is a static prism collider at the measured height, and the robot is initialized from the episode’s recorded joint configuration. For SimFoundry, we register its objects into the same rig frame through the head camera, as in the object-replacement protocol above, keep its estimated object masses, assign friction from our material table by object category, and replay the same command stream in the same calibrated robot setup. SimFoundry scenes are rendered with its inpainted table texture and SceneRig scenes with the reconstruction’s own tabletop.

Command replay. Physics is stepped at 240 Hz. After a 0.5 s settling period, we execute the time-stamped joint targets and gripper commands through the simulated controllers according to their recorded timing. The arm controller uses hardware-matched joint gains with full gravity feed-forward. After initialization, the robot follows the commands through simulated controllers, while objects move through the resulting contacts and dynamics. No policy queries, trajectory adaptation, or human intervention occur after replay begins.

Scoring. We score final object placements with the criteria in Appendix C.4, including treating a missing required object as a failure, with automated verdicts reviewed against the real demonstrations by a human scorer. For everything-into-the-bin replay, objects already inside the bin are excluded from the placement count. We also exclude objects that the real demonstration leaves untouched. These exclusions apply to both methods; hallucinated objects are not exempt, so an episode fails if one remains outside the bin.

Replays succeed in 20 of 25 SceneRig scenes and 9 of 25 SimFoundry scenes (95% Wilson intervals [61, 91]% and [20, 55]%). SceneRig succeeds on all nine demonstrations that SimFoundry replays successfully and on eleven more (exact McNemar p = 0.001). Across the SceneRig scenes, 39 of 45 evaluated objects reach their targets. Of the five unsuccessful replays, four leave one object unplaced and one leaves two. SimFoundry’s automatic object counts include extra instances and can omit missing or fused objects, so they do not provide a directly comparable object-level success fraction. We therefore compare the methods using the reviewed episode-level outcomes in Table 5.

Placement ablation. We repeat the same 25 demonstrations with the code-edits-only variant, which removes the bounded move tool while retaining direct scene-code edits and physical settling. Recorded commands, calibration, controllers, simulation settings, and scoring remain unchanged. This variant succeeds on 11/25 replays (44%), compared with 20/25 (80%) for full SceneRig (Table 11). The full method improves success on four tasks and ties on mustard into the bin, supporting the value of the bounded move tool for placement fidelity in robot manipulation.

Table 11 Object placement ablation for open-loop replay. Successful replays out of five real demonstrations per task. Code edits only removes Move while retaining Edit; both variants use the same replay protocol.
<table><tr><td>Task</td><td>Code edits only</td><td>Full SCENERIG</td></tr><tr><td>Everything→bin</td><td>1/5</td><td>3/5</td></tr><tr><td>Cup→bowl</td><td>1/5</td><td>4/5</td></tr><tr><td>Fruits→bowl</td><td>2/5</td><td>4/5</td></tr><tr><td>Marker→cup</td><td>2/5</td><td>4/5</td></tr><tr><td>Mustard→bin</td><td>5/5</td><td>5/5</td></tr><tr><td>All</td><td>11/25 (44%)</td><td>20/25 (80%)</td></tr></table>

## D Additional Experiments

## D.1 Ablation Studies

Backbone comparison. Table 12 evaluates SceneRig with diferent VLM backbones (main experiments use Claude Opus 5). GPT-6 Astra achieves the best results across all reported metrics, with the clearest gains in appearance. Object placement remains similar across backbones, suggesting that improved rendered similarity does not necessarily translate into comparable gains in 3D layout. Figure 10 provides qualitative comparisons.

Physics-grounded object placement. Table 13 compares raw SAM3D poses, the same poses after 5 s of PhysX settling, and SceneRig’s combined initialization and pose-refinement pipeline. All variants share segmentation, assets, camera, root surfaces, materials, and lighting. The SAM3D variants retain the shared gravity-aligned frame but bypass our metric pose initialization and refinement, gravity-based stability checks, and physical settling during reconstruction. The post-hoc variant applies its 5 s settling pass only after the raw-pose scene is assembled. Raw poses often violate support relations: 69% of objects lie below their support surface (median 17 mm), and 15% float above it. Only 2 of 80 scenes keep every object within 10 mm of its initial position after settling, and 46 of 80 within the 50 mm stability threshold. Post-hoc settling allows objects to reach resting configurations but can reduce segmentation overlap and 3D placement accuracy. SceneRig improves both alignment and placement while all 80 scenes pass the stability test. This comparison evaluates the combined placement pipeline, rather than isolating initialization and refinement individually. To assess the bounded move tool, the code-edits-only variant removes it from pose refinement while retaining the scene-code editing tool, which settles and renders each edit for agent review. It obtains lower instance overlap (ARI 0.858 vs. 0.878; FG-IoU 0.792 vs. 0.819) and object appearance (LPIPS<sup>∗</sup> 0.178 vs. 0.170), while all 80 scenes remain stable. The replay ablation reveals a larger downstream gap: success falls from 80% to 44% without the bounded move tool (Table 11). This suggests that aggregate reconstruction metrics can miss placement diferences that matter for executing recorded robot motions.

Table 12 Reconstruction with different VLM backbones. Every row uses SceneRig with the listed backbone for all configurable VLM roles across preprocessing and the agent stages.
<table><tr><td></td><td colspan="4">Source-view reconstruction</td><td colspan="4">Object-centric appearance</td><td colspan="2">Instance structure</td><td colspan="2">3D object layout</td></tr><tr><td>VLM backbone</td><td>|PSNR ↑</td><td>SSIM ↑</td><td>LPIPS ↓ nCLIP ↓</td><td></td><td>PSNR* ↑</td><td>SSIM*↑</td><td>LPIPS*</td><td>nCLIP*</td><td>ARI ↑</td><td>FG-IoU ↑</td><td>OSPA@5 ↓ F1@10 ↑</td><td></td></tr><tr><td>Claude Opus 5</td><td>13.277</td><td>0.544</td><td>0.518</td><td>0.122</td><td>13.609</td><td>0.440</td><td>0.170</td><td>0.062</td><td>0.878</td><td>0.819</td><td>2.4</td><td>0.93</td></tr><tr><td>Claude Fable 5</td><td>12.837</td><td>0.535</td><td>0.535</td><td>0.126</td><td>13.082</td><td>0.432</td><td>0.170</td><td>0.064</td><td>0.879</td><td>0.820</td><td>2.4</td><td>0.94</td></tr><tr><td>GPT-5.6-terra</td><td>13.918</td><td>0.577</td><td>0.527</td><td>0.128</td><td>13.338</td><td>0.437</td><td>0.178</td><td>0.067</td><td>0.850</td><td>0.781</td><td>2.6</td><td>0.92</td></tr><tr><td>Kimi K3</td><td>12.830</td><td>0.542</td><td>0.557</td><td>0.121</td><td>13.196</td><td>0.439</td><td>0.184</td><td>0.066</td><td>0.865</td><td>0.802</td><td>2.5</td><td>0.93</td></tr><tr><td>GPT-6 Astra</td><td>16.512</td><td>0.592</td><td>0.369</td><td>0.100</td><td>14.535</td><td>0.462</td><td>0.151</td><td>0.060</td><td>0.891</td><td>0.838</td><td>2.3</td><td>0.95</td></tr></table>

Table 13 Object placement ablation. SAM3D’s raw poses share every other component (segmentation, assets, camera, root surfaces) with SceneRig. “Code edits only” removes Move; the agent specifies pose changes through Edit, which still runs physical simulation on the edited scene.
<table><tr><td></td><td colspan="3">Source-view reconstruction</td><td colspan="3">Object-centric appearance</td><td colspan="2">Instance structure</td><td colspan="2">3D object layout</td></tr><tr><td>Placement</td><td>PSNR ↑</td><td>LPIPS ↓ nCLIP ↓</td><td></td><td>|PSNR* ↑</td><td>LPIPS*</td><td>nCLIP*</td><td>|ARI ↑</td><td>FG-IoU ↑</td><td>OSPA@5↓</td><td>F1@10 ↑</td></tr><tr><td>SAM3D raw poses</td><td>12.915</td><td>0.538</td><td>0.131</td><td>12.810</td><td>0.195</td><td>0.078</td><td>0.784</td><td>0.705</td><td>2.7</td><td>0.88</td></tr><tr><td>SAM3D raw poses + physics settle</td><td>12.768</td><td>0.546</td><td>0.129</td><td>12.408</td><td>0.198</td><td>0.079</td><td>0.748</td><td>0.664</td><td>3.0</td><td>0.84</td></tr><tr><td>SCENERIG, code edits only</td><td>13.272</td><td>0.523</td><td>0.124</td><td>13.415</td><td>0.178</td><td>0.066</td><td>0.858</td><td>0.792</td><td>2.4</td><td>0.92</td></tr><tr><td>SCENERIG</td><td>13.277</td><td>0.518</td><td>0.122</td><td>13.609</td><td>0.170</td><td>0.062</td><td>0.878</td><td>0.819</td><td>2.4</td><td>0.93</td></tr></table>

## D.2 Additional Quantitative Results

Direct Codex harness with GPT-6 Astra. Codex [29] is OpenAI’s general-purpose coding-agent meta-harness: it gives the model shell and file tools but no task-specific structure. For the direct Codex baseline in Table 14, we attach each benchmark image to an isolated GPT-6 Astra session with ultra reasoning efort. We automate the runs through the non-interactive codex exec interface, using the prompt shown in Listing 1. The agent writes Blender Python, runs Blender headlessly, inspects rendered images, and iteratively refines the camera, geometry, materials, lighting, and composition. Each worker uses Blender 4.5.3 with Cycles on one assigned CUDA GPU, with a six-hour timeout per attempt and at most two attempts per image. Filesystem isolation prevents access to other scene workspaces and reconstruction results. The prompt prohibits external models, textures, images, and assets, and requires a final-render inspection followed by at least one refinement pass. It also requires one semantically labeled top-level Empty per independently movable object, with all of that object’s parts grouped beneath it, so that each object can be simulated as a single rigid body; environment geometry, cameras, and lights have separate roots. Required outputs are a procedural scene.py, a self-contained final.blend, its rendered final.png, and a report.json describing reconstruction decisions and known mismatches.

Reconstruction harnesses with shared VLM backbones. Table 14 compares reconstruction harnesses with a shared VLM backbone. Both VIGA and the direct Codex harness receive an object-grouping instruction: all parts of each movable object share a parent Empty, allowing it to be simulated as a single rigid body. We run VIGA with medium reasoning efort and a 20-round budget for both backbones. With Claude Opus 5, SceneRig outperforms VIGA across appearance, instance structure, object placement, and physical stability. With GPT-6 Astra we compare against two baselines that share the backbone but not our harness. Neither VIGA nor the direct Codex harness has a physics stage, so we score their scenes after a 5 s PhysX settle with every object as a rigid body with its parts (primitives) being convex-hull colliders, letting floating and interpenetrating objects come to rest before the metrics are computed. Astra substantially improves VIGA’s appearance, and the Codex harness achieves the highest full-scene PSNR. SceneRig nevertheless achieves better instance segmentation, more accurate placement, and a higher stability pass rate: all 80 scenes pass, compared with 24 for VIGA and 16 for Codex, while OSPA@5 is 2.3 cm compared with 4.6 and 4.2 cm, respectively. These results indicate that improved image-space fidelity alone does not ensure accurate and stable object placement. SceneRig complements VLM reasoning with explicit scene structure, metric grounding, and physics-checked pose corrections; this comparison evaluates the complete harnesses rather than individual components. Figure 11 illustrates the comparison.

Input  
GPT-6  
Opus 5 (main experiment)  
GPT-5.6-Terra  
Kimi K3  
Fable 5  
![](images/e4a6e7b4d10ffe2091942df3c5d273492dc69f78b085c80dbe2973fbc038d099.jpg)  
Figure 10 Qualitative comparison across VLM backbones.

Reconstruct the attached target image as a procedural Blender scene.

The same reference image is available at ‘{{TARGET\_FILE}}‘ inside your current   
workspace. Work autonomously   
and iteratively: write Blender Python, run Blender headlessly, render candidate   
images, inspect those renders with the image-viewing tool, and improve the   
camera, geometry, materials, lighting, and composition until the result is a   
strong visual match.

Constraints:

Table 14 Reconstruction with different harnesses and shared VLM backbones. We compare VIGA, SceneRig, and the direct Codex harness. VIGA and Codex appearance and layout metrics are computed after an additional 5 s settling pass. Note that results from Tables 1 and 2 evaluate each method’s native reconstruction output. Stability is measured over the settling pass. API cost includes both VLM and image generation model costs.
<table><tr><td colspan="2"></td><td colspan="3">Full-scene appearance</td><td colspan="3">Object-centric appearance</td><td colspan="2">Instance structure</td><td colspan="2">3D object layout</td><td>Physics</td><td>API cost</td></tr><tr><td rowspan="2">VLM backbone | Method</td><td></td><td></td><td>|PSNR ↑ LPIPS ↓ nCLIP ↓</td><td></td><td>|PSNR* ↑ LPIPS* ↓ nCLIP*</td><td></td><td></td><td>↓ARI↑</td><td>FG-IoU ↑</td><td>|OSPA@5 ↓ F1@10 ↑|</td><td></td><td>|Stable scenes ↑|</td><td>|($/scene) ↓</td></tr><tr><td>VIGA Claude Opus 5</td><td>11.042</td><td>0.671</td><td>0.191</td><td>8.725</td><td>0.337</td><td>0.154</td><td>0.261</td><td>0.239</td><td>4.9</td><td>0.20</td><td>8/80</td><td>16.13</td></tr><tr><td rowspan="3">GPT-6 Astra</td><td>SCENERIG</td><td>13.277</td><td>0.518</td><td>0.122</td><td>13.609</td><td>0.170</td><td>0.062</td><td>0.878</td><td>0.819</td><td>2.4</td><td>0.93</td><td>80/80</td><td>4.22</td></tr><tr><td>VIGA</td><td>15.316</td><td>0.433</td><td>0.125</td><td>14.077</td><td>0.187</td><td>0.082</td><td>0.721</td><td>0.648</td><td>4.6</td><td>0.37</td><td>24/80</td><td>7.74</td></tr><tr><td>Codex harness SCENERIG</td><td>17.007 16.512</td><td>0.370 0.369</td><td>0.100 0.100</td><td>14.773 14.535</td><td>0.178 0.151</td><td>0.073 0.060</td><td>0.653 0.891</td><td>0.594 0.838</td><td>4.2 2.3</td><td>0.60 0.95</td><td>16/80 80/80</td><td>3.99 6.24</td></tr></table>

```lua
Work only inside the current workspace; do not search for other files or solutions.
Do not download or use external models, textures, images, or other assets.
Blender and Blender’s embedded Python are the only application/runtime you
should use for scene creation and rendering.
Prefer a procedural, reproducible ‘scene.py‘ over manual-only edits.
Use headless Blender and render with Cycles on the assigned CUDA GPU. CPU
rendering is not an acceptable fallback in this benchmark. In every script
that renders, initialize Cycles with the Blender 4.5-compatible sequence:
‘‘‘python
scene.render.engine = "CYCLES"
preferences = bpy.context.preferences.addons["cycles"].preferences
preferences.compute_device_type = "CUDA"
preferences.refresh_devices()
cuda_devices = [device for device in preferences.devices if device.type == "CUDA"]
if not cuda_devices:
raise RuntimeError("Assigned CUDA GPU is unavailable")
for device in preferences.devices:
device.use = device.type == "CUDA"
scene.cycles.device = "GPU"

Do not use ‘get_devices()‘ in place of ‘refresh_devices()‘, and do not catch
CUDA initialization errors by falling back to CPU. Confirm the final render
log identifies a CUDA device and record ‘Cycles CUDA‘ in ‘report.json‘.
Match the reference image’s framing and aspect ratio.
一 The final ‘.blend‘ must render without external file dependencies.
- Do not stop at an explanation or plan; create and validate the artifacts.
Object grouping requirement:
- Create one top-level Empty for each independently movable physical object.
Parent every mesh, curve, text object, and decorative part belonging to that
object under its Empty.
If several parts would normally move together as one rigid object, they
belong to the same Empty. For example, a mug body, handle, liquid, and logo
should share one parent.
- Do not combine separate physical objects under one Empty, even when they
touch or overlap. Do not create a separate Empty for every mesh part.
Name obiect roots '0BJ <semantic label> <index>. such as ‘0BJ mug 01'. and
set these custom properties on each root:
‘‘‘python
root["is_object_asset"] = True
root["semantic_label"] = "mug"
ccc
- Put non-obiect geometry under the reserved top-level Empty ‘ENVIRONMENT'.
- Put cameras and lights under ‘CAMERA_RIG‘ and ‘LIGHTING_RIG‘.
Only Empty objects may exist at the top level. Every renderable object must
ultimately descend from one of these top-level groups.
Before saving ‘final.blend‘, validate the grouping:
‘‘‘python
renderable_types = {"MESH", "CURVE", "FONT", "SURFACE", "META"}
top_level = [obj for obj in bpy.context.scene.objects if obj.parent is None]
assert all(obj.type == "EMPTY" for obj in top_level), [
obj.name for obj in top_level if obj.type != "EMPTY"
]
```

for obj in bpy.context.scene.objects:   
if obj.type not in renderable\_types:   
continue   
root = obj   
while root.parent is not None:   
root = root.parent   
assert root.type == "EMPTY", obj.name   
assert (   
root.get("is\_object\_asset", False)   
or root.name in {"ENVIRONMENT", "CAMERA\_RIG", "LIGHTING\_RIG"}   
), f"{obj.name} is not assigned to a recognized group"   
  
Required deliverables:   
‘output/scene.py‘   
‘output/final.blend‘   
‘output/final.png‘   
‘output/report.json‘   
‘report.json‘ must contain the scene id ‘{{SCENE\_ID}}‘, the chosen renderer,   
image dimensions, the main reconstruction decisions, and any known mismatch.   
Before finishing, open ‘output/final.png‘ with the image-viewing tool   
and make at least one refinement pass based on that inspection.  
Listing 1 Task prompt for the direct Codex baseline.

Segmentation before and after 3D reconstruction. Table 15 scores each method’s own 2D instance segmentation of the input and the instance masks rendered from its final scene, against the simulator ground truth of the 44 grounded scenes. SceneRig achieves high input-mask agreement with the ground truth (ARI 0.981 and FG-IoU 0.970). In contrast, although REST3D and SceneConductor also segment well, their results degrade notably once objects are placed in 3D. The decline from input segmentation to rendered instance masks highlights errors introduced during lifting and reconstruction, including geometry, camera alignment, scale, and object placement.

Table 15 Instance segmentation at the segmentation stage and in the final scene. Results on the 44 simulator-grounded scenes. Stage-1 metrics use visible instance masks on the original input image, whereas final-scene counts include only objects visible in the source-view render. Changes in visibility can therefore change the count even without a change in the reconstructed object inventory.
<table><tr><td></td><td colspan="3">Stage-1 segmentation</td><td colspan="3">Final scene</td></tr><tr><td>Method</td><td>ARI ↑</td><td>FG-IoU ↑</td><td>Count MAE↓</td><td>ARI↑</td><td>FG-IoU ↑</td><td>Count MAE↓</td></tr><tr><td>VIGA</td><td>0.450</td><td>0.394</td><td>4.43</td><td>0.283</td><td>0.260</td><td>3.36</td></tr><tr><td>REST3D</td><td>0.864</td><td>0.809</td><td>3.18</td><td>0.210</td><td>0.204</td><td>3.20</td></tr><tr><td>SceneConductor</td><td>0.824</td><td>0.800</td><td>1.11</td><td>0.197</td><td>0.187</td><td>1.14</td></tr><tr><td>SimFoundry</td><td>0.780</td><td>0.732</td><td>1.84</td><td>0.676</td><td>0.589</td><td>1.05</td></tr><tr><td>SCENERIG (ours)</td><td>0.981</td><td>0.970</td><td>0.27</td><td>0.878</td><td>0.819</td><td>0.27</td></tr></table>

## D.3 Additional Qualitative Results

Figure 12 extends the comparison to more scenes. Across these examples, SceneRig more closely preserves object arrangement, small items, and surrounding surfaces. The novel views reveal missing geometry and incorrect depth placement that can be less apparent from the input viewpoint, highlighting the importance of reconstructing individual objects and their spatial relationships.

![](images/bf965e1350458ccb131b69dba115921016081e11b3849f27520240b5ae4cc9de.jpg)  
Figure 11 Scene reconstruction with a shared GPT-6 Astra backbone.

![](images/439e7a72659da72e52f486fed44bbea457a7959604214ad288440e59326595ac.jpg)  
Figure 12 Additional qualitative reconstruction results.