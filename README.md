# RTC implementation reference

给另一个 agent 参考 **如何在已有 flow-matching 动作模型中实现 training-time RTC** 的精简代码集。

先读 [IMPLEMENTATION.md](IMPLEMENTATION.md)，再读下面的核心文件。这里包含可运行的小型算法与队列测试，以及原 OpenPI 接入片段；训练完整模型仍需目标项目自己的网络、数据管线和训练器。

## 阅读顺序

| 顺序 | 文件 | 要迁移的逻辑 |
| --- | --- | --- |
| 1 | [training.py](rtc_reference/training.py) | 抽 delay、干净前缀、逐 action 时间、有效 token loss |
| 2 | [sampling.py](rtc_reference/sampling.py) | 每一步去噪固定前缀，前缀时间为 0 |
| 3 | [coordinates.py](rtc_reference/coordinates.py) | 绝对命令 → 当前 state 的 delta → 原训练归一化 |
| 4 | [queue.py](rtc_reference/queue.py) / [controller.py](rtc_reference/controller.py) | 观测 tick、承诺前缀、异步响应合并与超时拒绝 |
| 5 | [OpenPI 模型片段](reference/openpi_model_methods.py) | 原模型中 loss、attention mask 与 KV-cache 采样的具体接法 |
| 6 | [时间条件与冻结片段](reference/openpi_time_and_freeze.py) / [policy 片段](reference/openpi_rtc_policy.py) | adaRMS 的逐 token 条件、可选微调策略、协议和坐标转换 |

`rtc_reference/` 是可执行的模型无关参考实现。`reference/` 是从原实现保留的代码片段，用于阅读和迁移，依赖原 OpenPI 定义，不能单独导入执行。它们不构成完整模型。

## 核心约定

- 时间方向：`t=1` 为噪声，`t=0` 为干净动作。
- `simulated_delay=S` 的训练支持集为 `0..S-1`，默认 `P(d) ∝ exp(-d)`。
- 训练时，前 `d` 个动作干净、时间为 0；后缀使用同一个采样时间。
- 前缀与 episode padding 不计算 loss；padding 也必须从 attention 中屏蔽。
- 默认 loss 对动作维求和，再除以全 batch 的有效 action token 数。
- 推理时，**每一步**去噪都将前缀固定为已承诺动作，前缀时间设为 0。
- 线上队列保存绝对命令；每次观测后用当前 state 重新编码前缀。
- 返回动作从观测时刻索引。超过承诺前缀截止 tick 的响应会被拒绝。

这是 training-time hard-prefix RTC；没有 inference-time VJP / PiGDM guidance。

## 运行参考测试

Python 3.11，CPU 即可。只需要 NumPy、JAX 和 pytest，无需 OpenPI、Flax、PyTorch、模型权重或数据集。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
JAX_PLATFORMS=cpu python -m pytest -q
```

测试验证数学与梯度、硬前缀条件化、坐标转换和异步 tick 合并。完整网络的 attention padding 隔离、逐 token adaRMS 和端到端训练，需要在目标项目中另外验证。

2026-10-28 精简后验证：独立目录中 **13 项测试通过**；source distribution 和 wheel 构建通过。与旧完整实现的 53 项测试是两套不同范围的测试，不能用这个结果代替完整模型集成验证。

## 给另一个 agent 的入口

```text
请阅读这个仓库的 README.md 和 IMPLEMENTATION.md，再参考 rtc_reference/ 的核心逻辑，
将 training-time hard-prefix RTC 接入当前项目已有的 flow-matching 动作模型。
reference/ 保留原 OpenPI 接法，只作为源代码参考。
重点核对时间方向、逐 action timestep、padding attention mask、全局 loss 归约、
前缀的当前 state 坐标转换，以及观测 tick 到异步队列的索引。
保留目标项目自己的模型、训练器和数据接口，并补齐集成测试。
```

## 来源

从已检查的 `openpi-training-rtc` 实现中精简。完整代码提取版本保留在 Git 历史中；当前默认分支专注实现参考。

- OpenPI 基线：[Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)，`215abfb217dbac7d5f1273282331b9b1866c0479`。
- RTC 算法参考：[real-time-chunking-kinetix](https://github.com/Physical-Intelligence/real-time-chunking-kinetix)，`9296f31d62d5bfeb5779dcb2f9bcf71ca37f448b`。
- `queue.py` / `controller.py` 来自原实现，仅调整模块导入；模型与 policy 片段保留原方法体。
- `training.py` / `sampling.py` / `coordinates.py` 为便于迁移而抽出的模型无关接口，不是额外一套已训练模型。
- [Apache 2.0 许可证](LICENSE)，保留的 [Gemma 条款](LICENSE_GEMMA.txt)。
