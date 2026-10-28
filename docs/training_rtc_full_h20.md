> 迁移说明：本文保留原始实现说明与历史验证记录；旧机器路径及训练产物不随仓库发布。当前安装、路径与运行命令以根目录 [README](../README.md) 为准。

# Banana training RTC，全量微调：2 × H20 96 GB

本分支基于官方 OpenPI `215abfb217dbac7d5f1273282331b9b1866c0479`，配置名为
`pi05_piper_banana_training_rtc_full`。底座是官方 `pi05_base`，全部
3,353,433,872 个参数可训练，包括 SigLIP、PaliGemma、Action expert、adaRMS 和动作头。
完整算法和坐标约定见 [TRAINING_RTC.md](TRAINING_RTC.md)，本机测试结果及验证边界见
[发布检查记录](training_rtc_review.md)。

默认配置：全局 batch 16，每卡 8；FSDP=2；10,000 次更新；统一 AdamW，峰值学习率
1e-5，warmup 500 步，cosine 衰减至 1e-6；EMA 关闭；W&B 默认关闭。
每 1,000 次更新在 64 个固定验证起始帧上验证 delay 0–4。

## 新机器准备

GitHub 包含代码、测试、划分列表和文档。Banana 数据、底座权重、归一化资产、训练输出
分别在本机准备。不要把 LeRobot `meta/stats.json` 直接当成 delta `norm_stats.json`。

```bash
git clone git@github.com:z-R-rr/rtc.git
cd rtc
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
source scripts/rtc_env.sh
RTC_PYTHON="$PWD/.venv/bin/python"
RTC_DATASET=/actual/path/to/banana
RTC_BASE_PARAMS=/actual/path/to/pi05_base/params
```

确认目标机器的两张 H20 编号。后文以 `CUDA_VISIBLE_DEVICES=0,1` 为例。
先用 CPU 做单轨迹统计试算，然后计算正式统计。它们都从配置中的明确训练列表取数据：
训练集 45 条轨迹；验证 episode ID 为 `[7,14,21,28,35,42]`，不参与统计。

```bash
JAX_PLATFORMS=cpu "$RTC_PYTHON" scripts/compute_rtc_norm_stats.py \
  --config-name pi05_piper_banana_training_rtc_full --repo-id "$RTC_DATASET" \
  --max-episodes 1 --output-dir outputs/full_h20/norm_smoke
JAX_PLATFORMS=cpu "$RTC_PYTHON" scripts/compute_rtc_norm_stats.py \
  --config-name pi05_piper_banana_training_rtc_full --repo-id "$RTC_DATASET"
JAX_PLATFORMS=cpu "$RTC_PYTHON" scripts/rtc_preflight.py \
  --config pi05_piper_banana_training_rtc_full \
  --repo-id "$RTC_DATASET" --base-params "$RTC_BASE_PARAMS" \
  --output outputs/full_h20/preflight.json
```

若复制了这次已验证的完整归一化资产目录，则无需重算；应同时复制
`norm_stats.json` 和 `norm_stats_manifest.json`。共享资产目录为：

```text
assets/pi05_piper_banana_training_rtc_lora/piper_banana_all_delta_h30_val_7_14_21_28_35_42/
```

目录名包含 `lora` 是因为两个配置共享同一份训练集统计量。其动作定义均为
`action[t+i] - state[t]`，7 维全 delta（含夹爪），H=30，排除 episode 尾部 padding。
在新机器重新计算同一份 banana 数据，应得到 24,413 个 state、712,815 个有效动作 token。

## 先做 GPU smoke

长任务在 tmux 中运行。新开 tmux 后，重新设置上面的 `RTC_PYTHON`、`RTC_DATASET`、
`RTC_BASE_PARAMS` 变量并 `source scripts/rtc_env.sh`。

```bash
tmux new -s banana_rtc_full_smoke
# 在 tmux 内进入仓库，并完成上面的环境变量设置。
mkdir -p outputs/full_h20
CUDA_VISIBLE_DEVICES=0,1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
  "$RTC_PYTHON" scripts/train.py pi05_piper_banana_training_rtc_full \
  --data.repo-id "$RTC_DATASET" --weight-loader.params-path "$RTC_BASE_PARAMS" \
  --exp-name banana_full_h20_smoke --num-train-steps 2 --log-interval 1 \
  --rtc-validation-interval 2 --rtc-validation-batches 4 \
  > outputs/full_h20/smoke.log 2>&1
```

检查 loss/梯度有限、delay 0–4 验证输出正常，以及
`checkpoints/pi05_piper_banana_training_rtc_full/banana_full_h20_smoke/1/` 完成保存。
配置的实际 H20 显存和通信需要这一步确认；小模型 CPU 测试不代表 H20 实机验证。

## 正式训练和恢复

在单独的 tmux 会话中，保留相同环境变量和数据/权重路径，启动新实验：

```bash
CUDA_VISIBLE_DEVICES=0,1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
  "$RTC_PYTHON" scripts/train.py pi05_piper_banana_training_rtc_full \
  --data.repo-id "$RTC_DATASET" --weight-loader.params-path "$RTC_BASE_PARAMS" \
  --exp-name banana_full_h20_v1 > outputs/full_h20/train.log 2>&1
```

需要 W&B 时，先在目标机器执行 `wandb login`，并在首次启动命令加 `--wandb-enabled`。
同一实验续训时，在原命令追加 `--resume`。恢复模型参数、Adam 状态和 step；未保存的更新
会重跑，数据加载迭代位置未保存。代码、数据、归一化和训练语义需与原实验一致。
不要用这版无 LoRA adapters 的模型直接加载之前训练出的 LoRA checkpoint。

最终输出目录为
`checkpoints/pi05_piper_banana_training_rtc_full/banana_full_h20_v1/9999/`。
RTC 客户端、服务端、delta 前缀转换和队列合并代码已包含在分支内。服务命令：

```bash
CUDA_VISIBLE_DEVICES=0 "$RTC_PYTHON" scripts/serve_training_rtc.py \
  --config pi05_piper_banana_training_rtc_full \
  --checkpoint checkpoints/pi05_piper_banana_training_rtc_full/banana_full_h20_v1/9999 \
  --port 8000 --num-steps 10
```

离线 RTC loss 是归一化空间中的 flow 速度误差。动作平滑度仍需比较 chunk 切换处的物理
动作增量、速度/加速度/jerk，并同时报告任务成功率和完成时间。
