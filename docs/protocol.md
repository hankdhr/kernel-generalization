# 评测协议 v0

该文档是先导协议；确认性实验前应冻结为带 Git commit 的新版本。当前工具只实现清单、基础预算聚合和手写探针，bootstrap、隔离 worker、完整 agent 与公共 benchmark 适配尚待实现。

## 1. 数据与留出

- 先按规范化计算图/语义族分 train/dev/test，再采 shape、dtype、layout。仅改变变量名不算新任务。
- 原始图、生成 seed、依赖版本和任务哈希均保存。相同或近似结构跨 split 的审查需要语义检查，当前 family 检查并不等于完整语义去重。
- 单独记录公开 benchmark 是否进入训练/提示检索。公开预训练语料不可完全审计，因此不声称零污染。
- 公开测试定义与 agent 可读取的环境不同：worker 只暴露本轮任务、开发输入和允许的反馈；最终 correctness 输入由评测器保管。开源不意味着运行时可读取隐藏测试答案。
- 当前 pilot 的输入 seeds 用于流程复现且全部公开，不承担强防泄漏评测职责；正式评测的开发反馈和最终隐藏输入必须分开。

## 2. 正确性先于性能

核对 shape、dtype、layout 约定、数值误差、输入是否被修改、返回 tensor 的设备和 CUDA stream。正式测试加入边界长度、对齐差异、stride、极端有限值，并定义 NaN/Inf、空张量等是否在合同中。pilot 只覆盖有限 FP32 输入、正行列和正 stride。

当前探针每配置检查 3 seeds × 5 分布：normal、uniform、zero、impulse、alternating；对比 float64 计算再 cast 到 FP32 的结果，rtol=1e-4/atol=2e-4。此容差仅属于这组预实验，不能直接沿用到所有算子。最终容差以算子数值分析和上游合同为准。

禁止从输出缓存、输入特定随机 seed、固定答案、参考实现调用或改写评测器获得结果。正式候选需隔离运行、限定文件和网络访问、设置编译/运行超时；同时审查源代码并用独立输入和 profiler 确认实际计算。当前生成脚本只保存文本，不执行模型代码。

## 3. 性能计时

- 使用独占 GPU；生成模型、训练、profiler 与计时不并发。记录 GPU 型号、驱动、CUDA、编译器、包版本、功耗/时钟设置、系统负载和代码 hash。
- correctness 全通过后再计时。预热后交错、随机化运行候选和参考实现，保存每次样本，跨会话复测。不能仅保留最快一次。
- 报告计时范围：kernel-only / wrapper、是否含分配和 host launch、CUDA Graph、warm/cold cache 等。pilot 使用 CUDA events，warm cache，wrapper 分配包含在调用路径，非 CUDA Graph；小 kernel 的事件间隔可能受到主机调度影响，需要独立校验。
- eager 和 torch.compile 均须正确，compile 完成后才测稳态延迟。正式比较基线应在独立阶段固定并复测，避免用同批噪声选择最快基线；上游默认 baseline 另报以支持横向比较。
- profiler 单独采集，不以被 profile 的延迟当性能结果。`ncu --version` 成功不代表计数器权限可用。
- 训练/数据采集/模型加载/编译/搜索耗时不计入稳态 kernel 延迟，但全部计入相应总成本。缓存命中和编译冷启动分别记录。

## 4. 预算指标

固定最大候选数 K、总生成 token B 和墙钟 W；任一耗尽即停止。所有模型生成的思考和最终答案 token 都计入 B；单独报告 prompt token、GPU 秒、API 费用。所有失败尝试计数，不允许把“修语法”视为免费。

对任务 i，只有通过全部数值检查且在预算内的候选，才进入其最佳速度比 s_i。主要指标：

\[
\mathrm{fast}_{p}(K,B,W)=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[\text{有正确候选且 }s_i>p].
\]

主要 p=1.05，辅报 1.0 和 1.2。总分母包含错误、超时、编译失败和没有完成的任务。这个 5% 是实用门槛，不能代替不确定性分析。correct-only 几何均值只能作为附带指标，并在名称中明确限定。

报告 K∈{1,4,8} 下的曲线，以及到第一个正确解/达到性能门槛所需的尝试、token 和时间。未达标任务是删失数据，不能删除后只比较成功样本的平均时间。后续实现预算曲线面积或限制平均达标时间，并公开定义。

### 当前聚合器输入

每个 JSONL 文件只能包含一种 method、一个 run、一个生成 seed。按任务的尝试编号从 1 连续增长，必须保留失败行。累计成本单调。`baseline_ms` 为独立校准后固定的正确基线。

```json
{"run_id":"example-only","method":"initial-model","task_id":"<manifest-id>","seed":101,"attempt":1,"status":"compile_error","tokens_total":1024,"wall_seconds_total":6.2}
```

若 status 为 `ok`，还必须有正的 `baseline_ms` 和 `candidate_ms`。这里只是格式示例，不是实测数据。

```bash
python -m kernel_generalization report --manifest data/pilot-v0.jsonl --records runs/experiment/candidates.jsonl --split train --seed 101 --attempts 4 --max-tokens 32768 --max-wall 900 --out runs/experiment/report-k4.json
```

当前聚合器输出是开发阶段的 best-correct-attempt 指标。正式主表还需实现候选选择与独立最终测试：agent 只按开发反馈选一个最终产物，最终测试不能替 agent 从多个候选中择优。这两种指标应分列报告。

## 5. 统计与归因

使用同任务、同预算的配对比较；按计算图/任务族聚类 bootstrap，shape 和输入 seed 不当独立语义样本。报告每族分数、置信区间、正确率退化和全部失败类型。正式统计分析代码尚未实现。

评测种子与训练种子分开；对照固定工具、提示模板、模型模式、上下文长度和最大候选数。分别匹配训练 token、优化器步数和计算量，若不能同时严格相等，报告偏差并增加单因素控制。

若使用外部教师，公开教师身份和生成成本，并单列教师带来的知识迁移。相同在线预算和包含离线训练的数据/算力总预算是两个不同比较，论文需同时给出。
