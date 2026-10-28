> 迁移说明：本文保留原始实现说明与历史验证记录；旧机器路径及训练产物不随仓库发布。当前安装、路径与运行命令以根目录 [README](../README.md) 为准。

# π0.5 Training-Time RTC：JAX + delta，动作侧 LoRA / 全量微调

本目录是独立的官方 OpenPI 源码副本，不依赖原 `/home/ubuntu/Workspace/openpi` 的源码。OpenPI 基线固定为 `215abfb217dbac7d5f1273282331b9b1866c0479`；RTC 算法对照 `Physical-Intelligence/real-time-chunking-kinetix` 的 `9296f31d62d5bfeb5779dcb2f9bcf71ca37f448b`。上游许可证保留在本目录，来源和修改文件见 `UPSTREAM.json`、`CHANGES.json`。

范围：JAX 的 π0.5、相对于 chunk 观测 state 的 delta 动作、training-time 硬前缀条件化。默认适配 Piper 单臂 7 维任务；不包含 inference-time PiGDM/VJP guidance。PyTorch 使用 RTC 配置会明确报错。数据加载仍使用官方的 PyTorch/LeRobot 依赖。

## 冻结策略与实际参数量

配置名：`pi05_piper_training_rtc_lora`，位于 `src/openpi/training/config.py`。

| 部分 | 更新策略 |
|---|---|
| SigLIP、视觉投影、PaliGemma 语言 Transformer | 全部冻结 |
| 已存在的 PaliGemma LoRA（如果选择相应模型 variant） | 冻结 |
| Action expert 原始 attention / FFN 权重 | 冻结 |
| Action expert 全部 18 层 attention / FFN LoRA，rank=32 | 训练 |
| Action expert 每层 pre-attention / pre-FFN adaRMS、final adaRMS 的 Dense kernel/bias | 训练 |
| `time_mlp_in/out`、`action_in_proj/out_proj` 的 kernel/bias | 训练 |

默认从已有任务全量微调 checkpoint 开始，因此 VLM 选择 `gemma_2b`，不创建无用的随机 VLM adapters。若底座包含已有 VLM LoRA，改用匹配的 `gemma_2b_lora` 保留它们；冻结规则仍会冻结它们。专用权重加载器拒绝静默丢弃底座中的额外 LoRA 参数。

实际完整模型形状检查：总参数 **3,375,552,272**；可训练 **140,789,792（约 4.17%）**：

- Expert LoRA：22,118,400。
- Expert adaRMS：116,505,600。
- 时间 MLP 与动作输入/输出投影：2,165,792。

adaRMS 含多个 `1024 -> 3072` 稠密映射，解冻后不是几万参数的小改动。完整路径、形状见 `outputs/verification/parameters.json`。`RTCAdamW` 对 LoRA 使用基准 LR，对继承的 adaRMS/时间网络/动作投影使用 0.2 倍 LR；不是只缩放原始梯度。默认 LoRA peak LR=2.5e-5，继承参数 peak LR=5e-6。数值是可调整起点，不是已验证的真机最优值。

## 训练目标和官方差异

OpenPI 时间方向是 `t=1` 噪声、`t=0` 干净动作，与 Kinetix 相反。

1. 每个样本独立采样 `d`；`simulated_delay=S` 的支持集为 **0..S-1**。
2. 前 `d` 个动作保持干净，其逐 token 时间为 0；后续动作使用同一个采样时间。
3. 前缀不计算 loss；LeRobot 的 episode 尾部 `action_is_pad` 也不计算 loss。
4. `train` 只控制图像增强，**`train=False` 不关闭 RTC**。固定 delay 可以通过 `compute_loss(..., rtc_delay=d)` 或验证脚本选择。

默认 `rtc_loss_reduction="official"`：

```
L = sum_{batch, time, action_dim}(active * squared_velocity_error)
    / sum_{batch, time}(active)
```

与公开 RTC 代码一样，分母没有动作维数 D，也没有每个样本各自归一化。`compute_loss` 返回一个 [B,H] 网格，训练器对其 mean 后恰好得到上式。JAX 全局数组上的归约覆盖全 batch，不能在每块设备上分别归约再简单平均。

提供显式选项 `rtc_loss_reduction="per_element"`，在上式基础上再除以 D。这是尺度变体，必须作为不同实验记录。与原 OpenPI 的逐元素 mean 比较时，默认 official loss 存在 D 的尺度差异；也会影响梯度裁剪和学习率调参。

默认保留官方 π0.5 的 `Beta(1.5,1)*0.999+0.001` 时间采样。公开 Kinetix 使用 uniform；要对齐这一项可以设 `rtc_time_distribution="uniform"`。这份代码复用官方 RTC 条件化/归约，**不宣称复现了 Kinetix 的网络、数据和训练预算**。

默认 delay PMF 为官方的 `P(d) ∝ exp(-d)`；S=5 时为 `[0.6364,0.2341,0.0861,0.0317,0.0117]`。可用 `simulated_delay_weights` 明确配置重平衡分布，例如 `(0.1,0.1,0.2,0.4,0.2)`。服务端拒绝训练支持集以外的前缀长度。

## 验证能力和数据划分

默认按 episode ID 确定性划分 90% / 10%，seed=42，训练和验证互斥。固定版本 LeRobot 对不连续 episode ID 的 delta query 边界存在索引问题，本目录通过 `RTCLeRobotDataset` 修正，不修改外部安装包。

每 1000 次更新及训练结束时验证，默认每次最多 4 个完整 batch，关闭图像增强，逐一评价 d=0..4。验证集不足 requested batches 时不循环重复；最后一个不足 batch_size 的 batch 被丢弃。训练集和验证集都不足一个 batch 时会报错，需要降低 batch_size。

指标包括每个 d 的：

- `official_token_loss`：官方维度求和的 flow loss。
- `loss`：配置所选归约。
- `raw_mse`：仅真实动作前 7 维的**归一化速度场 MSE**，不是反归一化位置误差，也不是平滑度。
- `active_tokens`：有效目标数量。没有有效目标时报告 null，不把它当零误差。

先跨 batch 累加误差分子和有效 token 数，再相除，避免不同有效长度带来的 mean-of-means 偏差。日志保存在 checkpoint 实验根目录 `rtc_validation.jsonl`。

验证集只保证未参与本次 RTC 适配；原任务底座和继承的 norm stats 可能已经使用这些 episode。若要衡量完全独立的泛化，需要一个从底座训练阶段就留出的数据集。

Checkpoint 的 `assets/rtc_manifest.json` 保存模型设置、PMF、loss、时间分布、delta mask、统计量 SHA256、训练/验证 episode ID、底座路径、优化器、源码 SHA256。服务端以 checkpoint 中的模型设置为准，并检查归一化统计和动作约定。RTC resume 要求原配置和记录一致；更改设置应创建新实验。

## delta 前缀转换

数据集的 `action` 和线上协议中的 `rtc.prefix` 都必须是**绝对目标命令，使用机器人的原始单位**。这不等于模型用绝对动作：模型内部仍是 delta。

对观测时刻 t 的 state，整个 chunk 的转换为：

```
delta[i] = absolute_command[t+i] - state[t]  # 7 维全 delta，包括夹爪
model_action = Pad32(Normalize(delta))
```

它不是 `command[i]-command[i-1]`，也不是相对于旧 chunk 的 state。政策包装器把前缀放入同一套训练输入 transforms，执行 DeltaActions -> checkpoint Normalize -> Pad；不另写一套近似归一化。π0.5 默认 quantile normalization，无额外 clipping。输出反向转换为绝对目标后，再精确恢复已承诺前缀，避免浮点 roundtrip 误差改变旧命令。

更换机器人时需要修改数据适配器及明确的 delta mask，不能仅更改 raw_action_dim。特别核对夹爪单位和其 delta/absolute 约定是否与底座和 norm stats 一致。

## 推理协议与队列合并

请求直接兼容官方 websocket 的 `infer(obs)`，RTC 字段位于 obs 内，包装器在输入 transforms 前提取：

```python
obs = {
    "images": {"top_head": head_rgb, "hand_right": wrist_rgb},
    "state": current_state_7d,
    "prompt": instruction,
    "rtc": {
        "prefix": committed_absolute_commands,  # [d,7]，从观测 tick 开始
        "delay": d,
        "start_index": observation_tick,
        "request_id": unique_request_id,
    },
}
```

每次去噪都固定前缀并设置其时间为 0，Euler 更新后再固定一次。模型只做前向推理，不使用 VJP guidance。首个 chunk 用 d=0 同步生成，控制循环启动前完成首次 JIT。

队列和控制器在：

- `packages/openpi-client/src/openpi_client/rtc_queue.py`
- `packages/openpi-client/src/openpi_client/rtc_controller.py`

队列始终保存绝对命令和整数控制 tick。`begin_request(capture_observation, delay=d)` 在队列锁内采集观测并快照前缀，避免使用响应返回时的队列来构造前缀。观测设备仍需自行提供同步/足够新的图像与 state；软件锁无法消除传感器本身的延迟。

若请求起点为 n，返回时已执行 e 个动作：

- e <= d：保留尚未执行的旧前缀 `[e:d]`，接上新 chunk 的 `[d:H]`。
- e > d：拒绝该响应，继续现有旧队列；记录 `missed_prefix_deadline`。不将错误条件化结果强行拼接。
- 重复、取消、上一 episode 的响应：拒绝。
- 队列耗尽：明确报错，由机器人控制端选择安全的 hold/stop/rebootstrap，不隐式重复命令。

提前返回也不会提前替换已承诺的前缀。因此 d 是承诺执行长度，可取测量延迟的保守上界；范围仍必须在训练覆盖内。AsyncRTCController 每个 execution_horizon=s tick 发起一次请求，要求 `1 <= d <= s <= H-d`，保证正常情况下有可用旧前缀。必须把采图、网络、推理、反序列化和控制调度都计入实际延迟。

最小接入示例（调用者负责控制频率和机器人 I/O）：

```python
from openpi_client.websocket_client_policy import WebsocketClientPolicy
from openpi_client.rtc_controller import AsyncRTCController

client = WebsocketClientPolicy(host="127.0.0.1", port=8000)
controller = AsyncRTCController(
    client, horizon=30, action_dim=7, max_delay=4,
    delay=3, execution_horizon=10,  # 示例，需按实测延迟调整
)
controller.bootstrap(capture_observation)  # 控制暂停时同步生成首个 chunk
# 之后每个固定控制周期调用一次：
absolute_target = controller.next_action(capture_observation)
# 在该 tick 下发 absolute_target。若另做 clipping/filtering/interpolation，
# 必须让 queue 保存实际承诺执行的命令；否则训练前缀与执行前缀不再一致。
# 退出时 controller.close()
```

## Banana 数据集与新统计量（当前选定方案）

用户当前选择：从官方 `/home/ubuntu/Data/pi05_base/params` 开始，使用 `/home/ubuntu/Data/banana`，配置名 **`pi05_piper_banana_training_rtc_lora`**。本配置继承上文冻结方案和训练目标，替换数据、权重与资产路径；之前的四本书任务配置仍独立保留。

Banana 为 LeRobot v2.1：51 episodes、27,856 frames、30 Hz，两路视频、7 维 state/action。采集 contract 确认 `action` 是绝对 leader command，关节 radian、夹爪 meter，时间 offset 从 0 开始。任务文本为 `Put this carrot and this banana on the plate`。

归一化统计必须在转 delta 后计算，不能把 LeRobot 的 `meta/stats.json` 改名使用。命令：

```bash
cd /home/ubuntu/Workspace/openpi-training-rtc
source scripts/rtc_env.sh
RTC_PYTHON=/home/ubuntu/Workspace/openpi/.venv/bin/python
JAX_PLATFORMS=cpu "$RTC_PYTHON" scripts/compute_norm_stats.py \
  --config-name pi05_piper_banana_training_rtc_lora
```

对于 Piper RTC，官方入口转发到新增的 numeric-only 统计函数：读取 Parquet 数值列，应用相同 `DeltaActions`，不解码视频、不调用模型，也不需要 GPU。当前划分的单 episode smoke 输出在 `outputs/banana_norm_stats/resplit_val_7_14_21_28_35_42/smoke/`，不能用于正式训练。

正式输出：

```
assets/pi05_piper_banana_training_rtc_lora/piper_banana_all_delta_h30_val_7_14_21_28_35_42/norm_stats.json
assets/pi05_piper_banana_training_rtc_lora/piper_banana_all_delta_h30_val_7_14_21_28_35_42/norm_stats_manifest.json
```

划分来自 `/home/ubuntu/Data/banana/splits/manifest.json` 及同目录的 `train_episodes.json`、`validation_episodes.json`。本项目在 `configs/banana_train_validation.json` 保存原始 episode ID、帧数与来源 SHA256；注册配置固定同一组 ID。训练集 45 条轨迹、24,413 帧；验证集为 `[7,14,21,28,35,42]`，6 条轨迹、3,443 帧。显式列表优先于 `split_seed` / `validation_fraction`，不使用 `meta/info.json` 中原始存储划分重新分配。后续更改划分时必须同步注册配置与快照并重算统计。

统计逻辑：state 每个训练帧计一次；动作从该帧开始取 H=30，做 `action[t+i]-state[t]`（7 维全 delta）。排除 episode 尾部越界 token，不将重复尾帧计入统计；只读取上述 45 条训练轨迹，共 712,815 个有效动作 token。q01/q99 使用官方 `RunningStats` 的直方图近似分位数，pi05 使用 quantile 归一化。统计本身不按随机 RTC delay 再次加权，是训练、验证和推理共用的固定坐标变换；不为验证集单独拟合统计量。

旧 `piper_banana_all_delta_h30_train/` 资产仍保留，来自旧随机划分，当前 banana 配置已切换到新目录。训练启动时如存在统计 manifest，会校验训练/验证 episode 列表、horizon、统计摘要以及非 smoke 标记；checkpoint 记录准确的 episode 列表，后续验证从 checkpoint 恢复该列表。

manifest 记录 horizon、mask、split、原始 Parquet SHA256 和统计量 SHA256。更改 horizon、动作定义、数据内容或训练划分时，应新计算并保存对应资产，避免在旧实验使用的文件上原地替换；仅修改 delay PMF 不改变动作坐标变换。

实际数据 preflight：

```bash
JAX_PLATFORMS=cpu "$RTC_PYTHON" scripts/rtc_preflight.py \
  --config pi05_piper_banana_training_rtc_lora \
  --output outputs/banana_norm_stats/resplit_val_7_14_21_28_35_42/preflight.json
```

后续训练、服务、验证均使用 `pi05_piper_banana_training_rtc_lora`。checkpoint 会复制这份 stats，服务端从 checkpoint 读取，不使用线上观测重新估计统计量。此统计步骤没有启动 GPU 训练，也不证明从通用底座直接进行动作侧 RTC 适配的任务成功率。

## Banana 全量微调：2 × H20 96 GB

配置名 **`pi05_piper_banana_training_rtc_full`**。从官方 `/home/ubuntu/Data/pi05_base/params` 开始，面向两张 H20 96 GB；与 LoRA 实验使用同一数据划分、同一 delta norm stats、H=30、delay 支持集 0–4、Beta 时间采样和 official loss 归约。归一化由数据与动作定义决定，切换全量微调不需要重算。

`model.rtc_finetune_mode="full"`，模型为 `gemma_2b` + `gemma_300m`，不创建 LoRA adapters。SigLIP、视觉投影、PaliGemma 语言 Transformer、Action expert 原始 attention/FFN、adaRMS、时间 MLP、动作投影全部可训练，共 **3,353,433,872** 个参数；冻结过滤器为 `nnx.Nothing`，未增加 stop-gradient。训练入口在 CLI 覆盖后按 full/LoRA 模式重建冻结过滤器和校验优化器，避免 full 被原来的 LoRA 规则覆盖。

| 参数 | 全量微调设置 |
|---|---|
| GPU / FSDP | 2 × H20 96 GB；`fsdp_devices=2` |
| 全局 batch size | 16，每卡 8 |
| 优化器 | 普通 AdamW；全部参数使用相同学习率，无 0.2 倍继承参数分组 |
| 学习率 | warmup 500 步；峰值 1e-5；cosine 衰减至 1e-6 |
| Adam / 裁剪 | b1=0.9，b2=0.95，eps=1e-8；global norm clip=1.0 |
| 更新预算 | 10,000 次，全局 batch 与原 LoRA 实验一致 |
| 验证 | 每 1,000 次更新，4 × 16=64 个固定起始帧，逐 delay 验证 |
| 保存 | 每 1,000 步及最终 step 9999；保留最近与 5000 倍数 |
| EMA / W&B | EMA=None；W&B 默认 disabled |
| 参数精度 | 继承 OpenPI：可训练参数与 Adam 状态 FP32，模型计算使用 bfloat16 |

以下命令在 H20 机器上执行。先按上游依赖安装环境，并确保数据、`pi05_base` 和本项目 assets 可访问。`RTC_PYTHON` 应指向该机器安装好依赖的 Python；此机器可复用 `/home/ubuntu/Workspace/openpi/.venv/bin/python`。`CUDA_VISIBLE_DEVICES=0,1` 仅为 H20 机器示例，按实际物理卡号设置。

先在 tmux 中运行与正式配置同 batch 的 2-step smoke，检验实际显存、双卡通信、RTC 验证与 checkpoint 保存：

```bash
cd /home/ubuntu/Workspace/openpi-training-rtc
tmux new -s banana_rtc_full_smoke
source scripts/rtc_env.sh
RTC_PYTHON=/path/to/venv/bin/python
mkdir -p outputs/train/banana_rtc_full_h20
CUDA_VISIBLE_DEVICES=0,1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
  "$RTC_PYTHON" scripts/train.py pi05_piper_banana_training_rtc_full \
  --exp-name banana_full_h20_smoke --num-train-steps 2 --log-interval 1 \
  --rtc-validation-interval 2 --rtc-validation-batches 4 \
  > outputs/train/banana_rtc_full_h20/smoke.log 2>&1
```

smoke 成功后在另一个 tmux 会话中启动正式训练；正式配置仍从官方底座开始：

```bash
cd /home/ubuntu/Workspace/openpi-training-rtc
tmux new -s banana_rtc_full_h20
source scripts/rtc_env.sh
RTC_PYTHON=/path/to/venv/bin/python
CUDA_VISIBLE_DEVICES=0,1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
  "$RTC_PYTHON" scripts/train.py pi05_piper_banana_training_rtc_full \
  --exp-name banana_full_h20_v1 \
  > outputs/train/banana_rtc_full_h20/train.log 2>&1
```

默认输出 `checkpoints/pi05_piper_banana_training_rtc_full/banana_full_h20_v1/`。需要恢复同一 full 实验时追加 `--resume`；旧 LoRA checkpoint 含有 adapters，不能直接作为这版无 LoRA 模型的 base，也不能用 full 配置 resume 旧 LoRA 实验。权重加载器会拒绝静默丢弃 adapters。

已有服务和验证脚本支持 full checkpoint。例如：

```bash
CUDA_VISIBLE_DEVICES=0 "$RTC_PYTHON" scripts/serve_training_rtc.py \
  --config pi05_piper_banana_training_rtc_full \
  --checkpoint checkpoints/pi05_piper_banana_training_rtc_full/banana_full_h20_v1/9999 \
  --port 8000 --num-steps 10
```

checkpoint 的 model manifest 保存 full/LoRA 模式；推理与验证按 checkpoint 恢复，旧版缺少此字段的 RTC LoRA checkpoint 仍按 LoRA 模式加载。两张 H20 的实际 GPU smoke 与吞吐尚待运行；CPU 上的小模型检查和参数形状检查不能代替实机显存验证。本次仅编写配置，没有启动全量训练。

## 安装、检查和运行

可按上游 README 使用本目录 `uv.lock` 安装独立环境。现有机器代码验证使用了原 OpenPI 的 Python 环境，但通过 PYTHONPATH 强制导入本目录源码；没有复制或修改原环境。

```bash
cd /home/ubuntu/Workspace/openpi-training-rtc
source scripts/rtc_env.sh
RTC_PYTHON=/home/ubuntu/Workspace/openpi/.venv/bin/python
# 新装本目录环境时改为 RTC_PYTHON="$RTC_ROOT/.venv/bin/python"
```

先核对输入。本次实现时以下旧路径在当前机器上不存在，需要修改注册配置，或在 preflight 和训练命令中传入一致的路径：

- `/home/ubuntu/dataset/recognize_four_book_lerobot_v21_processed`
- `/home/ubuntu/Model/pi05-piper-recognize-fourbook-fullft-bs64-5000/params`
- 同一底座的 `assets/piper_recognize_four_book_all_delta/norm_stats.json`

```bash
JAX_PLATFORMS=cpu "$RTC_PYTHON" scripts/rtc_preflight.py \
  --repo-id /actual/dataset \
  --base-params /actual/base/params \
  --assets-dir /actual/base/assets \
  --asset-id piper_recognize_four_book_all_delta
```

preflight 检查路径、确定性 split、读取一个小 batch、归一化后的有限数值和 episode padding mask。它不验证底座对机器人单位的语义，也不加载整个模型。

先做 2-step smoke；如果使用 CLI 指定路径，在下面命令也追加 `--data.repo-id`、`--data.assets.assets-dir`、`--data.assets.asset-id`、`--weight-loader.params-path`：

```bash
CUDA_VISIBLE_DEVICES=0 "$RTC_PYTHON" scripts/train.py pi05_piper_training_rtc_lora \
  --exp-name rtc_smoke --num-train-steps 2 --batch-size 2 --num-workers 0 \
  --rtc-validation-interval 1 --rtc-validation-batches 1
```

正式训练前检查 smoke 的数据、loss、验证和 checkpoint。长任务请放 tmux 中，并显式保存日志，例如在 tmux 内执行：

```bash
mkdir -p outputs/train
CUDA_VISIBLE_DEVICES=0 "$RTC_PYTHON" scripts/train.py pi05_piper_training_rtc_lora \
  --exp-name rtc_action_lora_v1 > outputs/train/rtc_action_lora_v1.log 2>&1
```

默认 checkpoint 目录为 `checkpoints/pi05_piper_training_rtc_lora/<exp_name>/`。沿用上游零起始 step 命名，10,000 次更新的最终目录通常是 `9999`，2 次更新为 `1`。没有自动覆盖已有实验目录。

服务端：

```bash
"$RTC_PYTHON" scripts/serve_training_rtc.py \
  --checkpoint checkpoints/pi05_piper_training_rtc_lora/rtc_action_lora_v1/9999 \
  --port 8000 --num-steps 10
```

固定 delay 的离线验证：

```bash
"$RTC_PYTHON" scripts/eval_training_rtc.py \
  --checkpoint checkpoints/pi05_piper_training_rtc_lora/rtc_action_lora_v1/9999 \
  --batches 20 --batch-size 8 --output outputs/eval/rtc_action_lora_v1.json
```

验证不是动作平滑度测试。平滑度应另行记录**实际反归一化执行命令和实测关节轨迹**，比较切换点的命令增量、速度/加速度/jerk 分位数，关节和夹爪分开统计，并同时报告任务成功率和完成时间。模型没有额外的平滑正则或在线滤波，不保证只靠 RTC 就能获得指定的物理平滑度。

## 代码验证

```bash
RTC_PYTHON="$RTC_PYTHON" bash scripts/verify_rtc.sh
JAX_PLATFORMS=cpu "$RTC_PYTHON" scripts/inspect_rtc_parameters.py
```

长的完整测试已在 tmux 后台运行；日志在 `outputs/verification/`。测试使用缩小的 Gemma 和轻量视觉前端来验证真实 JAX 前向、反向、JIT、冻结参数和采样路径；完整 π0.5 只做抽象参数形状检查，不分配大模型权重。已核对 quantile 和 z-score 的绝对前缀转换、msgpack 协议、队列索引、固定 delay 验证与官方 loss 归约。

未执行真实 Piper 数据训练、真实 checkpoint 推理、GPU 多卡运行或真机评测；测试通过不等价于已证明实际动作更平滑。
