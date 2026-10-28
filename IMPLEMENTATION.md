# 在已有模型中接入 Training RTC

## 1. 训练目标

假设原始动作 chunk 为 `a ∈ [B,H,D]`，已经完成目标项目的动作变换和归一化。独立采样标准高斯 `ε`、每个样本的时间 `t ∈ [B]` 和整数延迟 `d ∈ [B]`。

```text
prefix[b,i]   = i < d[b]
time[b,i]     = 0 if prefix[b,i] else t[b]
x[b,i]        = time[b,i] * ε[b,i] + (1-time[b,i]) * a[b,i]
target[b,i]   = ε[b,i] - a[b,i]
active[b,i]   = not prefix[b,i] and not action_is_pad[b,i]
```

前缀是该训练样本的真实未来动作，模拟推理中已承诺的旧命令。训练时不需要启动异步控制器。参考实现：`training.prepare_training`。

模型输出 `vθ(observation, x, time)`。默认损失为：

```text
L = Σ[b,i,k] active[b,i] * (vθ[b,i,k] - target[b,i,k])²
    / max(Σ[b,i] active[b,i], 1)
```

分母没有 `D`，也没有逐样本归一化。`flow_loss` 返回标量；若原训练器只接受 `[B,H]` loss 网格，则用 `loss_grid` 后统一取 mean。`per_element` 模式额外除以 D，需要作为不同损失尺度记录。多设备训练应按全局误差分子和全局有效 token 数归约，不能简单平均各设备独立归一化的 loss。

最小接入形状如下（`model` 与 `observation` 来自目标项目）：

```python
noise_rng, time_rng, delay_rng = jax.random.split(rng, 3)
noise = jax.random.normal(noise_rng, actions.shape)
time = sample_time(time_rng, actions.shape[0])
delay = draw_delay(delay_rng, (actions.shape[0],), upper_bound=5)
rtc_batch = prepare_training(actions, noise, time, delay, action_is_pad)
velocity = model(
    observation, rtc_batch['noisy_actions'], rtc_batch['model_time'],
    action_valid=rtc_batch['action_valid'],
)
loss = flow_loss(velocity, rtc_batch)
```

这里的 `action_valid` 是模型需要新增或接入的接口；仅调用 `prepare_training` 不会自动修改 attention。

## 2. 模型的关键改动

原模型若只支持 `time: [B]`，需要扩展为 `[B,H]`。在原 π0.5 实现中，sin/cos 时间编码、time MLP 与 action expert 的 adaRMS 都传递逐 action 时间。

```text
time [B,H]
  → sin/cos [B,H,E]
  → time MLP [B,H,E]
  → 每层 adaRMS 的 scale/shift/gate [B,H,E]
```

当 adaRMS 的条件仍为 `[B,E]` 时才增加序列轴广播；已有 `[B,H,E]` 条件不能再增加一个轴。原实现见 `reference/openpi_time_and_freeze.py` 的 `RMSNorm`。

保留原 prefix-LM 的 attention 结构：视觉/语言作为观测前缀，action block 内支持动作间注意力。RTC 干净前缀必须能影响待生成后缀。将 `action_is_pad` 从 action 输入的有效 token mask 中剔除，再构造完整 attention mask 和位置索引。**只屏蔽 padding loss 会让重复的 episode 尾部目标泄漏进有效预测。**

原模型接入方法见 `reference/openpi_model_methods.py`。片段保留 `embed_suffix`、`flow_errors`、`compute_loss`、`rtc_validation_sums` 和 `sample_actions`，省略了原网络初始化、图像/语言编码与辅助函数，因此不能单独运行。

## 3. 推理采样

给定当前观测和已承诺的前 d 个动作，将它们转换到当前模型坐标。使用 Euler 从噪声时间 1 积分到 0，每一步执行：

```text
x[:, :d]    ← committed_prefix
time[:, :d] ← 0
time[:, d:] ← 当前去噪时间
v           ← model(observation, x, time)
x           ← x - v / num_steps
x[:, :d]    ← committed_prefix
```

`sampling.hard_prefix_sample` 演示通用接口。`velocity(x, time)` 回调应捕获当前 observation 和可选的 KV cache。原 π0.5 具体 KV-cache 接法见模型片段。

调用者需验证：`S <= H`、delay 是整数、训练 `0 <= d < S`，服务默认只接受训练支持集。`num_steps` 必须为正整数。把长于训练支持的前缀作为部署扩展，需要单独评估。

首个 chunk 使用 `d=0` 同步生成。后续异步请求通常 `d>=1`。训练和推理都用同一时间方向；若目标模型使用相反时间约定，需要同步改插值、速度符号和积分方向。

## 4. 动作坐标

原 Piper 接法中的 wire action 和队列元素是 7 维绝对目标，关节 radian、夹爪 meter。模型动作是相对 **当前 chunk 观测 state** 的 delta：

```text
delta[i] = absolute_command[i] - current_observation_state
model_action = PadToModelDim(TrainingNormalize(delta))
```

整个 chunk 共用当前 state；不是相邻命令差，也不是上一次请求的 state。原 Piper 的夹爪也使用 delta。其他机器人的 mask、单位和归一化必须跟其训练数据保持一致。

`coordinates.encode_absolute_prefix` 通过 `encode_chunk` 回调复用目标模型的实际训练变换。不要重新实现近似的 normalize codec。模型输出经过原来的逆变换回到绝对命令后，调用 `restore_committed_prefix` 精确覆盖已承诺前缀，避免归一化 roundtrip 的浮点误差改变旧命令。

`reference/openpi_rtc_policy.py` 保留了原服务器的完整包装方式，包括 shape / finite 校验、当前 state 转换、JAX 调用和响应元数据。

## 5. 异步 tick 与队列合并

请求协议：

```python
obs['rtc'] = {
    'request_id': unique_integer,
    'start_index': observation_control_tick,
    'delay': d,
    'prefix': committed_absolute_commands,  # [d,D_raw]，从 start_index 起
}
```

响应中 `actions: [H,D_raw]` 同样从 `start_index` 起算，并回传 request_id、start_index、delay、`action_space='absolute'`。

设返回时从观测开始已执行 `e` 个动作：

| 条件 | 合并规则 |
| --- | --- |
| `e <= d` | 保留尚未执行的旧前缀 `[e:d]`，拼接新动作 `[d:H]` |
| `e > d` | 拒绝这次响应，旧队列继续；此输出未对当前已经执行的动作作条件化 |
| 重复、取消或上一 episode 的响应 | 拒绝 |

不能直接用 `new[e:]` 覆盖所有旧动作，也不能对 delay 再偏移一次。提前返回仍应兑现完整承诺前缀。

`RTCQueue.begin_request` 在锁内捕获 observation 并快照前缀，让观测 tick、队列索引和前缀一致。传感器同步和数据新鲜度仍由机器人侧负责。`pop()` 必须与实际下发控制命令一一对应；若额外 clipping / interpolation 改了命令，队列也必须保存实际承诺的值。

```python
controller = AsyncRTCController(
    policy, horizon=30, action_dim=7, max_delay=4,
    delay=3, execution_horizon=10,
)
try:
    controller.bootstrap(capture_observation)  # 控制循环开始前同步完成
    # 每个真实控制 tick 调用一次：
    absolute_command = controller.next_action(capture_observation)
finally:
    controller.close()
```

`execution_horizon=s` 是两次请求观测起点之间计划执行的 tick 数；源控制器要求 `1 <= d <= s <= H-d`。队列耗尽会明确报错，机器人层负责 hold / stop / rebootstrap。该控制器每次只保留一个在途请求，传输层需自行设置超时。

## 6. 微调、统计与验证

这些是原实现的实验选择，RTC 核心目标不要求使用 LoRA：

- 原 LoRA 配方：冻结 VLM，训练 action expert LoRA、adaRMS、time MLP、action 输入/输出投影。继承参数学习率为 LoRA 的 0.2 倍，使用真正的参数组学习率；仅缩放梯度在 Adam 下不等价。
- 全量微调：训练所有参数，使用统一 AdamW。冻结过滤器片段只适用于原 π0.5 的参数路径，迁移后需枚举实际参数验证。
- 时间分布：源实现默认 `Beta(1.5,1)*0.999+0.001`，可选 uniform。默认 delay 为 `exp(-d)`，不是均匀分布。
- 归一化统计只从训练轨迹计算，在转 delta 后、同一 horizon 的有效窗口上估计；不要将绝对动作统计量直接用于 delta。
- 验证固定多个 delay，关闭图像增强；先跨 batch 累加误差分子和有效 token 数，再相除。flow 速度场误差不能直接当作物理动作位置误差或平滑度。
- checkpoint 应保存时间/动作约定、delay 支持与分布、loss 归约、归一化、数据划分和可训练参数策略。

## 7. 目标项目必须补齐的集成验证

本仓库测试核心算法与队列；完成模型迁移后还需验证：

1. 修改干净前缀能改变生成后缀；只在最终输出中覆盖前缀不算实现条件化。
2. 改变 padding 目标值不影响有效 token 的预测。
3. 逐 action timestep 真实进入每层时间条件，不被错误广播成单一时间。
4. 混合 delay / 有效长度、跨设备情况下的 loss 与梯度归约正确。
5. 数据、在线前缀、模型输出的动作单位和 delta/absolute 变换一致。
6. early / on-time / late 响应、重置后旧响应、重复响应都符合 tick 语义。
7. 完整模型能前向、反向、保存与恢复，再在目标硬件上做真实延迟和任务评估。
