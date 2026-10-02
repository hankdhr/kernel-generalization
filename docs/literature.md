# 相关工作与定位

检索截止：2026-10-02。下面用于建立研究边界，不是穷尽文献综述。不同语言、硬件、模型、搜索次数、统计分母的成绩不能直接排总榜。主方案定稿前仍需逐篇检查训练样本、划分和消融细节。

| 工作 | 与本项目的关系 | 本项目需要超过的简单组合 |
|---|---|---|
| [CUDA Agent](https://arxiv.org/abs/2602.24286)；[代码](https://github.com/BytedTsinghua-SIA/CUDA-Agent) | 大规模 agentic RL、数据合成、验证/profiling 环境 | “可执行反馈 + RL”已有；不能直接把其系统换模型当新算法 |
| [DRTriton](https://arxiv.org/abs/2603.21465) | 合成计算图、课程 RL 和 test-time search；目标语言为 Triton | 组合任务生成本身不足以构成创新 |
| [KernelPro](https://arxiv.org/abs/2606.26453) | 语义化 profiler 工具、分层瓶颈判断、搜索和记忆 | 单纯给原始指标加解释、增加搜索工具已被覆盖 |
| [KernelBench](https://arxiv.org/abs/2502.10517)；[代码](https://github.com/ScalingIntelligence/KernelBench) | 公开任务与 fast_p 指标 | 必须明确基线、失败分母和测试预算 |
| [AccelOpt](https://arxiv.org/abs/2511.15915) | 核优化搜索与经验积累的相关参照 | 经验检索与权重学习需要拆开对照 |
| [Kevin](https://arxiv.org/abs/2507.11948) | 通过执行反馈训练 kernel 生成/优化的相关参照 | 需要比较只训练最佳代码与学习迭代过程 |
| [Dr. Kernel](https://arxiv.org/abs/2602.05885) | kernel RL 与执行评测基础设施的相关参照 | 验证奖励可靠性和失败轨迹处理 |
| [FlashInferBench](https://arxiv.org/abs/2601.00227)；[代码](https://github.com/flashinfer-ai/flashinfer-bench) | 后续真实推理算子评测候选 | 真实场景和公开合成任务分层报告 |

## 拟研究的差异

目前的工作假设是：**把相同代码起点下的可复测变换效果，以及效果随条件改变的证据，作为训练对象；用冻结工具与预算的留出实验验证它是否变成模型参数中的迁移能力。**

这需要证明的不只是最终代码更快，还包括条件预测有效、负例/效果反转有增益、关闭检索仍有效、对未见任务有效。以上仍是待查重、待验证的研究设计；不能写成“首次提出”。

## 模型和真实算子来源

- [Qwen3-8B 官方模型卡](https://huggingface.co/Qwen/Qwen3-8B)：首轮初始化，固定推理模式。
- [Qwen2.5-Coder-7B-Instruct 官方模型卡](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct)：模型家族对照。
- [Qwen3.5-9B 官方模型卡](https://huggingface.co/Qwen/Qwen3.5-9B)：后续较新模型验证。
- [FLA 仓库](https://github.com/fla-org/flash-linear-attention)：真实 kernel 候选，固定 commit，先测性能空间。

精确 revision 和查询时间见 `configs/sources.lock.json`。模型权重未下载，上游训练结果未在本项目复现。
