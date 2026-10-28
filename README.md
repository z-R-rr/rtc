# RTC: π0.5 training-time real-time chunking

独立的 JAX π0.5 Training RTC 仓库，包含硬前缀条件训练、Piper 数据适配、LoRA / 全量微调、验证、checkpoint 和异步推理队列。基于 OpenPI 源码内置必要运行模块，保留 `openpi` / `openpi_client` Python 命名空间；运行时不需要另一个 OpenPI checkout。

本仓库仅发布代码、测试、文档和 episode 划分。数据、底座权重、归一化统计、checkpoint、日志和原始 provenance 归档均需另行准备。原始训练机器上的 checkpoint 不随仓库上传。

## 实现

- 每个样本抽取延迟 `d`，前 `d` 个动作保持干净，逐 action timestep 为 0；其余动作按 flow matching 加噪。
- 默认 `H=30`、`D=32`、`simulated_delay=5`，即 `d ∈ {0,1,2,3,4}`，概率正比于 `exp(-d)`。
- Loss 只计算有效的非前缀 action token；episode padding 同时从 loss 和 attention 中排除。
- Piper 原始 7 维绝对命令转换为 `action[t+i] - state[t]`，包括夹爪；归一化后补到 32 维。线上前缀使用相同转换。
- 默认 loss 为动作维求和后除以有效 token 数。`raw_mse` 是前 7 维归一化速度场误差，不是物理位置误差或平滑度指标。
- LoRA 模式冻结 VLM，训练 action expert LoRA、adaRMS、时间网络和动作投影；全量模式训练全部参数。

| 路径 | 内容 |
| --- | --- |
| `src/openpi/models/pi0.py` | RTC flow loss、逐 action timestep、硬前缀采样 |
| `src/openpi/models/training_rtc.py` | 延迟分布和有效 token loss 归约 |
| `src/openpi/training/` | 数据、归一化、优化器、验证、checkpoint 与 manifest |
| `src/openpi/policies/rtc_policy.py` | 绝对动作前缀协议与模型坐标转换 |
| `packages/openpi-client/src/openpi_client/rtc_*.py` | 客户端队列和异步控制器 |
| `scripts/` | 统计、预检、训练、评估、服务入口 |
| `tests/rtc/` | RTC 数学、模型、队列、验证和恢复测试 |

保留部分上游兼容模块以满足配置和模型的导入依赖；Training RTC 支持 JAX π0.5，PyTorch RTC 配置会被拒绝。

## 安装

建议 Linux + NVIDIA GPU，Python 3.11，使用 [uv](https://docs.astral.sh/uv/)。依赖由 `uv.lock` 固定，包括 JAX 0.5.3、Flax 0.10.2，以及固定 revision 的 LeRobot。

```bash
git clone https://github.com/z-R-rr/rtc.git
cd rtc
GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --group dev
source scripts/rtc_env.sh
RTC_PYTHON="$PWD/.venv/bin/python"
```

GPU 依赖沿用上游 `jax[cuda12]`。`source scripts/rtc_env.sh` 将本 checkout 的源码和客户端加入 `PYTHONPATH`，避免意外导入其他安装的 OpenPI。

## 输入与配置

默认输入路径已改为本仓库下的相对路径，支持 CLI 覆盖：

```text
data/banana/                 # LeRobot v2.1 数据
data/pi05_base/params/       # 官方 π0.5 底座的 params 目录
assets/                     # 本地计算的归一化统计
checkpoints/                # 本次训练输出
```

提供三种训练配置：

| 配置 | 用途 |
| --- | --- |
| `pi05_piper_banana_training_rtc_lora` | Banana + 官方 π0.5 底座，action expert 侧适配 |
| `pi05_piper_banana_training_rtc_full` | 同一 Banana 划分，全参数微调；默认 FSDP=2 |
| `pi05_piper_training_rtc_lora` | 原四本书任务底座适配，输入另放在 `data/` 下 |

Banana 配置固定 51 条 episode 中的 45 条训练和 6 条验证；验证 ID 为 `[7,14,21,28,35,42]`。划分快照见 `configs/banana_train_validation.json`。**自有数据集需修改配置中的 episode 列表并重算统计**，不能直接沿用 Banana 划分。

## 计算统计与预检

```bash
RTC_DATASET=/absolute/path/to/banana
RTC_BASE_PARAMS=/absolute/path/to/pi05_base/params
RTC_CONFIG=pi05_piper_banana_training_rtc_lora

JAX_PLATFORMS=cpu "$RTC_PYTHON" scripts/compute_rtc_norm_stats.py \
  --config-name "$RTC_CONFIG" --repo-id "$RTC_DATASET"

JAX_PLATFORMS=cpu "$RTC_PYTHON" scripts/rtc_preflight.py \
  --config "$RTC_CONFIG" --repo-id "$RTC_DATASET" \
  --base-params "$RTC_BASE_PARAMS" --output outputs/preflight.json
```

统计只使用训练 episode 的有效 delta action window，排除越界 padding。不要用数据集的原始绝对动作 `meta/stats.json` 替代。

## 训练

LoRA 示例（默认 batch=16、10,000 次更新、EMA 关闭）：

```bash
"$RTC_PYTHON" scripts/train.py pi05_piper_banana_training_rtc_lora \
  --data.repo-id "$RTC_DATASET" \
  --weight-loader.params-path "$RTC_BASE_PARAMS" \
  --exp-name banana_rtc_lora
```

全量微调示例（默认面向两张 H20 96 GB；先完成目标硬件 smoke）：

```bash
CUDA_VISIBLE_DEVICES=0,1 "$RTC_PYTHON" scripts/train.py \
  pi05_piper_banana_training_rtc_full \
  --data.repo-id "$RTC_DATASET" \
  --weight-loader.params-path "$RTC_BASE_PARAMS" \
  --exp-name banana_rtc_full
```

短 smoke 可加 `--num-train-steps 2 --rtc-validation-interval 2 --log-interval 1`，使用独立 `--exp-name`。正式训练不要复用 smoke 的实验名。恢复训练时使用原配置、相同总步数、相同实验名并加 `--resume`；需要保留完整实验根目录和 `rtc_manifest.json`，只复制单个 step 目录不足以恢复当前入口。

## 离线评估与服务

```bash
RTC_CHECKPOINT=checkpoints/pi05_piper_banana_training_rtc_lora/banana_rtc_lora/9999

"$RTC_PYTHON" scripts/eval_training_rtc.py \
  --config pi05_piper_banana_training_rtc_lora \
  --checkpoint "$RTC_CHECKPOINT" --repo-id "$RTC_DATASET" \
  --output outputs/rtc_eval.json

"$RTC_PYTHON" scripts/serve_training_rtc.py \
  --config pi05_piper_banana_training_rtc_lora \
  --checkpoint "$RTC_CHECKPOINT" --port 8000
```

推理请求中的 `rtc.prefix` 是从观测 tick 起已经承诺执行的**绝对命令**，形状 `[delay,7]`；不是响应返回时重新截取的队列，也不是 delta。服务端每次去噪固定此前缀，客户端负责绝对 tick、截止时间和队列合并。协议和接入示例见 [算法与协议说明](docs/TRAINING_RTC.md)。

## 测试

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' \
  XLA_FLAGS=--xla_force_host_platform_device_count=2 \
  "$RTC_PYTHON" -m pytest tests/rtc tests/test_fixed_noise_policy.py -q
```

这套测试使用小模型和两个虚拟 CPU 设备，验证数学、反向传播、冻结策略、硬前缀条件化、padding、队列、验证聚合和分片 checkpoint 恢复。它们不等于完整 π0.5 GPU 训练或机器人任务成功率验证。提取前原目录的 RTC 测试为 49 passed；独立版本的验证结果由 GitHub Actions 和 `docs/VERIFICATION.md` 记录。

## 来源与许可证

- OpenPI: [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)，基线 `215abfb217dbac7d5f1273282331b9b1866c0479`。
- RTC 算法参考：[real-time-chunking-kinetix](https://github.com/Physical-Intelligence/real-time-chunking-kinetix)，revision `9296f31d62d5bfeb5779dcb2f9bcf71ca37f448b`。
- 本仓库从 `openpi-training-rtc` 独立提取，保留实现和测试；只调整默认本地路径、仓库文档与测试入口，不改变训练目标。
- 代码保留 [Apache 2.0 LICENSE](LICENSE) 与上游 [Gemma 许可证](LICENSE_GEMMA.txt)。下载和使用外部模型还需遵循其许可证。
- [来源说明](docs/UPSTREAM.json)；[原始改动清单](docs/original_changes.json) 是迁移前的历史快照，其中列出的归档、资产和实验输出未随代码发布。
