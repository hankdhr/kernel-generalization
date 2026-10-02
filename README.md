# Kernel Generalization

**研究目标：让模型在未见 CUDA 任务上，以更少的生成、编译和测量次数，写出正确且更快的 kernel。**

本仓库是研究启动版本。已实现任务清单、预算统计、环境检查、手写 CUDA 测量探针和单次模型生成入口。**尚无 GPU 实验结果，尚未实现完整 agent、训练循环或公共 benchmark 适配器。** 不宣称 SOTA，也不预设论文结论。

## 已做出的选择

| 项目 | 首轮选择 | 原因 |
|---|---|---|
| 主模型 | Qwen/Qwen3-8B，固定 revision | 8B 级可训练模型，先验证训练机制；推理模式明确固定 |
| 模型对照 | Qwen/Qwen2.5-Coder-7B-Instruct | 检查机制能否迁移到另一个代码模型 |
| 后续较新模型对照 | Qwen/Qwen3.5-9B | 核心机制成立后再投入，加载及训练兼容性另验 |
| 开发任务 | 自建 pilot-v0，108 个 FP32 行操作配置 | 验证 correctness、stride、计时、失败记录；不足以证明广泛泛化 |
| 公共评测 | KernelBench，先 L1/L2，完整任务表另行冻结 | 与文献对齐；不把公共任务当作无污染 OOD |
| 真实算子 | FLA 中的 Gated DeltaNet 候选，待测 headroom | 先查可优化空间，再决定是否作为案例 |
| GPU | 优先一张独占 H800，待确认实例和预算 | 分开生成与测速，训练配置以实测显存为准 |

来源和精确版本见 [sources.lock.json](configs/sources.lock.json)。硬件尚未租赁，没有启动付费运行。

## 核心假设

从**相同代码起点下的成对优化实验**提取“在何种 shape、layout、硬件和实现条件下有效或失效”，将这些证据用于模型训练，可能比只学习最佳代码，更能提高未见组合上的优化效率。

关键对照是相同初始化、相近训练 token/计算预算、相同测试工具和搜索预算下的 best-code SFT。还必须比较完整轨迹 SFT、仅检索经验、更多搜索。详见 [研究设计](docs/research-plan.md)、[实验矩阵](docs/experiments.md) 和 [算法候选](docs/ideas.md)。

## 本地可立即运行

Python 3.10+；下面的核心检查没有第三方依赖。在仓库根目录执行：

```bash
python -m unittest discover -s tests -v
python -m kernel_generalization check data/pilot-v0.jsonl
python -m kernel_generalization preflight --out runs/local-preflight.json
python scripts/run_probe.py --plan-only --out runs/probe-plan-001
```

输出路径必须不存在，防止覆盖历史。重新运行时使用新目录。生成任务清单：

```bash
python -m kernel_generalization suite --out runs/rebuilt-pilot.jsonl
```

GPU 测量入口（需要 CUDA 版 PyTorch、兼容 nvcc、C++ 编译器和 ninja；GPU 环境尚未验证）：

```bash
python scripts/run_probe.py --split train --limit 2 --out runs/h800-smoke-001
```

它测试两个手写 block-size 变体与 eager/torch.compile，保存逐次测量。它不执行模型生成代码，不提供模型成绩。完整 GPU 设置步骤见 [运行手册](docs/runbook.md)。

## 阅读顺序

1. [研究设计与模型选择](docs/research-plan.md)
2. [实验矩阵与停止条件](docs/experiments.md)
3. [新想法、算法和消融](docs/ideas.md)
4. [评测协议](docs/protocol.md)
5. [相关工作](docs/literature.md)
6. [全流程开源约定](docs/open-science.md)
7. [论文提纲](paper/outline.md)

## 状态与下一步

- [x] 固定模型和上游仓库 revision；生成带哈希的预实验任务。
- [x] 编写并本地测试预算指标、失败分母与清单完整性。
- [x] 提供单次生成和 GPU 探针入口；只验证 CPU 路径。
- [ ] 确认首轮预算、H800 SSH 入口、GitHub owner/repository。
- [ ] 在 GPU 上编译探针，验证 stream、数值、测量稳定性及 profiler 权限。
- [ ] 实现隔离 candidate worker、累计成本记录、多轮 agent 和 KernelBench 适配。
- [ ] 收集先导轨迹；完成条件学习与对照训练；锁定完整评测集。
- [ ] 运行正式实验和消融；写出依据实测结果的论文；发布可复现材料。

原创代码按 Apache-2.0 发布，第三方材料保留原许可。见 [LICENSE](LICENSE) 与 [THIRD_PARTY.md](THIRD_PARTY.md)。
