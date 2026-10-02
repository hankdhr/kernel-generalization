# 首轮运行手册

## A. 本地准备（已具备）

在仓库根目录使用 Python 3.10+，运行 README 的 CPU 检查。核心包无需 torch。数据清单可重建并比较哈希。`scripts/generate_one.py --plan-only` 只产生可审查 prompt，不下载模型。

```bash
python -c 'import json; print(json.loads(open("data/pilot-v0.jsonl").readline())["task_id"])'
python scripts/generate_one.py --task-id TASK_ID_FROM_ABOVE --plan-only --out runs/generation-plan-001
```

## B. GPU 环境，待接入

1. 选择独占 H800。先用实际机器执行 `python -m kernel_generalization preflight --out runs/h800-inventory-001.json`。
2. 在项目专用环境安装与驱动、nvcc 匹配的 CUDA PyTorch；需要 ninja、transformers 和 safetensors。具体版本在机器可用后选定并做兼容测试，当前不提供伪装成已验证的 GPU requirements.lock。
3. 保存 `python --version`、`nvcc --version`、`nvidia-smi`、主要包版本、编译命令；不要直接公开含私有索引凭证的完整环境变量或 pip URL。
4. 无其他 GPU 工作时运行 `scripts/run_probe.py --limit 2`。inspect 每个任务各 method 的 correctness 和 timing；completion 文件只表示流程结束，不意味着全部方法通过。
5. 在不同会话重复，检查 host-launch、温度、缓存和时钟漂移；有需要再建立 CUDA Graph 对照。对照范围必须在正式实验前固定。
6. 独立运行 ncu 对探针采集，检查实际 counter 访问权限。再用 compute-sanitizer 检查越界和 race；这些尚未在本地验证。

## C. 模型单次生成，待 GPU 验证

使用 `scripts/generate_one.py`，去掉 `--plan-only`。脚本使用锁定模型 revision、固定 non-thinking 模式和 seed，保存实际 chat template、原始 token、文本、版本和耗时。首次运行会下载模型，需要足够磁盘和内存；没有自动租机或调用付费 API。

单次生成脚本不是 agent；它没有编译反馈、策略搜索和训练。也不把生成文本自动执行。先完成下一节的 worker，再采集可用于 E01 的成绩。

## D. 下一项实现：隔离评测 worker

输入：候选源文件 hash、冻结任务合同、预算、独立测试输入种子。输出：结构化状态、编译日志、正确性、原始计时、累计资源成本。

采用独立受限进程/容器，候选无法读参考答案、模型凭证或测试器代码；关闭网络，限制写入目录，设置 compile/runtime/watchdog 超时。CUDA context 在任务之间隔离，device error 后重启 worker。明确架构可执行文件和缓存路径的隔离规则。

agent 接口：实现 → 编译 → correctness → 测量 → 根据允许反馈修改。每步保留完整日志；失败消耗预算。最终计时与模型生成串行。先实现一个固定策略作为所有训练模型的公共评测器，再测试可学习策略。

## E. 外部 benchmark

从 `configs/sources.lock.json` 指定的 revision 拉取 KernelBench 到 `third_party/`。在首轮 GPU 上锁定可运行任务及排除原因，预先冻结名单；排除不兼容任务不能根据模型结果决定。保留上游许可证。适配器尚未实现，不能把自建 pilot 分数写成 KernelBench 分数。

FLA 和 FlashInferBench 同理。在 FLA 候选上先与当前人工优化实现做测量；若没有稳定空间，就更换案例并记录理由，研究主假设不依赖某个 kernel 必须被击败。
