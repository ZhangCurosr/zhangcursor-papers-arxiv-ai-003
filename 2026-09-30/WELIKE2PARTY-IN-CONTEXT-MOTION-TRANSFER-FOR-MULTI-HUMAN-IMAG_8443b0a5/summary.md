---
title: "WELIKE2PARTY-IN-CONTEXT-MOTION-TRANSFER-FOR-MULTI-HUMAN-IMAG"
source: https://arxiv.org/pdf/2609.36937v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 23:29:37"
---

# 论文速读：WELIKE2PARTY-IN-CONTEXT-MOTION-TRANSFER-FOR-MULTI-HUMAN-IMAG

## 一句话总结
提出 WeLike2Party，一种无需推理时显式姿态/网格估计的多人图像动画框架，通过直接将参考图、驱动视频与目标视频拼接为 in-context 序列进行联合去噪；结合高分辨率参考图的分数 RoPE 编码与身份绑定监督损失，有效解决多主体遮挡下的外观细节丢失与身份-运动绑定错乱问题。

## 研究问题与动机
- 现有方法依赖 2D 骨架或参数化 3D 人体网格等显式运动表示，但在多人交互与强遮挡场景下，估计器易产生关节缺失、错误分配或帧级时序抖动，误差会直接级联至生成视频并引发视觉伪影。
- 显式表示本身信息容量有限，难以保留细粒度几何与交互线索；而直接采用 in-context 视频 conditioning 虽能保留 richer cues，但 naive 方案因 3D VAE 下采样与 patch embedding 会严重丢失面部、手部等细粒度外观细节。
- 缺乏主体级的显式身份-运动对应监督，多主体动画中身份易发生漂移、交叉重叠时出现身份交换（identity swap）或与相邻主体特征融合。
- 现有跨身份配对数据极度匮乏，且多数合成数据依赖姿态驱动动画模型生成，继承其 motion correspondence 误差；真实采集几乎无法获得严格时间同步的跨身份运动对。

## 核心贡献（创新点）
1. **WeLike2Party in-context 运动迁移框架**：推理阶段完全跳过显式姿态/网格提取，将参考图、驱动视频与加噪目标视频的 latent token 直接拼接为单一序列输入 DiT。与已有方法的本质区别在于绕过了易产生误差的中间表征管线，让自注意力直接学习身份、运动与交互的联合映射。
2. **Reference Asymmetric RoPE Conditioning (RARC)**：将参考图以更高空间分辨率编码，并通过分数坐标缩放将其映射到目标视频的预训练 RoPE 有效区间，使目标 token 能通过自注意力可靠地检索细粒度外观细节。与直接放大参考分辨率的做法不同，RARC 避免了位置外推导致的重复纹理与空间理解崩溃。
3. **Identity Binding Supervision (IBS)**：引入仅作用于训练阶段的辅助损失，利用真实实例掩码对目标到参考的注意力分布施加交叉熵监督，强制每个参考身份全程绑定到预期运动轨迹。与隐式数据驱动关联不同，IBS 提供强可解释的主体级绑定约束，且不增加任何推理参数或延迟。
4. **MotionTwin 大规模合成数据集与 MotionTwin-Bench 评测基准**：基于 Unreal Engine
