> 迁移说明：本文保留原始实现说明与历史验证记录；旧机器路径及训练产物不随仓库发布。当前安装、路径与运行命令以根目录 [README](../README.md) 为准。

# Training RTC 全量微调发布检查

本次检查对应配置 `pi05_piper_banana_training_rtc_full`，基于 OpenPI
`215abfb217dbac7d5f1273282331b9b1866c0479`。运行步骤见
[H20 全量微调指南](training_rtc_full_h20.md)，算法与动作约定见
[TRAINING_RTC.md](TRAINING_RTC.md)。

## 检查结果与修复

- 全量模式使用 `gemma_2b` 和 `gemma_300m`，不创建 LoRA adapters，冻结过滤器为
  `nnx.Nothing`。抽象参数检查得到总参数和可训练参数均为 3,353,433,872。
  CLI 覆盖后重新解析训练模式；全量模式使用统一 AdamW，避免被 LoRA 分组优化器覆盖。
- 修复 checkpoint 恢复边界：目录中只有已提交的 step 0 时，原逻辑会将其误判为空目录。
  现在仅在确实没有 checkpoint 时从头初始化；回归测试覆盖 step 0 和 step 1。
- 使用两个虚拟 CPU 设备检查 FSDP 保存/恢复：参数、AdamW 一阶/二阶矩、优化器计数、
  训练 step 及恢复后的分片均保持一致。
- 归一化脚本增加 `--repo-id`，新机器可以显式指定 banana 数据位置。
  全量与 LoRA 模式共享同一训练集划分、delta 定义和归一化资产。
- 检查固定 delay 0–4 验证、有效 token 的全局 loss 归约、绝对动作前缀到当前观测
  delta/归一化坐标的转换，以及客户端绝对时间索引和队列合并。相应回归测试已包含在
  `tests/rtc/` 中。

## 本机验证记录

| 检查 | 结果 |
| --- | --- |
| 前置 smoke | 20 passed，1 deselected |
| 完整 RTC 测试 | 49 passed，75.67 秒；两个虚拟 CPU 设备 |
| 变更 Python 文件 Ruff | All checks passed |
| 发布差异空白检查 | `git diff --check` 通过 |
| banana 实际数据预检 | 45 个训练 episode，6 个验证 episode；动作 shape `[2,30,32]` |
| 指定数据路径的统计 smoke | episode 0：558 个 state，16,305 个有效动作 token |

完整测试命令（先安装依赖并进入仓库）：

```bash
source scripts/rtc_env.sh
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' \
  XLA_FLAGS=--xla_force_host_platform_device_count=2 \
  .venv/bin/python -m pytest tests/rtc -q
```

测试运行产生 632 条上游依赖警告，包括 JAX/Flax/Beartype 弃用提示和 Orbax 分片元数据提示，
没有测试失败。模型前向、反向、JIT 和采样测试使用缩小的模型；两设备保存/恢复测试使用
小型参数树。完整 π0.5 参数数量通过抽象形状检查获得。

预检使用验证 episode `[7,14,21,28,35,42]`，完整训练集归一化的内容摘要为
`2add21f82ac548f6fad5f2ba6a474e9a2d08100d9616723be84670b05b3fcf7d`。
本机原始日志位于 `outputs/verification/github_review/`，由 `.gitignore` 排除。

## 验证边界

尚未在两张 H20 96 GB 上运行完整 π0.5 GPU smoke 或全量训练，实际显存、编译耗时和吞吐
需按运行指南确认。当前测试也不证明实机动作平滑度或任务成功率。

默认 RTC loss 是对动作维求和、再按有效目标 token 数归一化的 flow 速度误差，与原 OpenPI
逐元素均值存在动作维数对应的尺度差异。固定 delay 验证能够检查前缀条件下的预测误差，
平滑度仍需比较物理动作空间中 chunk 切换处的增量、速度、加速度和 jerk，并报告任务成功率。

`CHANGES.json` 记录本分支相对固定 OpenPI 基线的源代码、配置和文档变更及 SHA256；
不包含该清单自身和 `provenance/` 生成物。`provenance/training-rtc.patch` 对应同一组变更，
`UPSTREAM.json` 记录基线归档及其摘要。
