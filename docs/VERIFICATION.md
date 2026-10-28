# 独立提取验证

日期：2026-10-28。

提取后的源码复制到独立临时目录，使用已有 Python 3.11 依赖环境运行；`PYTHONPATH` 仅显式加入提取版本的 `src` 和 `packages/openpi-client/src`，不使用原 checkout 的源码。

| 检查 | 结果 |
| --- | --- |
| RTC 回归测试 + fixed noise policy | **53 passed**，27.12 秒 |
| 运行设备 | CPU，`XLA_FLAGS=--xla_force_host_platform_device_count=2` |
| Python 3.11 语法检查 | 102 个 Python 文件通过 |
| JSON / TOML 解析 | 通过 |
| 训练 CLI | `scripts/train.py pi05_piper_banana_training_rtc_lora --help` 正常；确认路径覆盖参数可用 |
| 发布内容 | 无数据、权重、checkpoint、统计资产、日志、缓存或环境目录 |

测试输出包含 632 条上游依赖警告，主要为 Beartype / Flax 弃用提示和 Orbax 分片元数据提示，没有测试失败。

验证覆盖小模型 flow loss、JIT 训练反向传播、冻结策略、双学习率优化器、逐 action 时间条件、硬前缀采样、padding 隔离、episode 划分、验证聚合、异步队列和双虚拟 CPU 设备上的 checkpoint 恢复。

本次没有重新训练完整 π0.5、执行 H20 GPU smoke 或操作机器人。首次 wheel 构建尝试因源机器访问 PyPI 的 TLS 连接失败而中止；依赖已存在的环境中，源码回归测试通过。GitHub Actions 提供从锁文件安装依赖和运行同一测试套件的流程。
