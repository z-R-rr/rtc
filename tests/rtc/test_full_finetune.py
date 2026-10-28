import dataclasses
import functools
import importlib.util
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import gemma
from openpi.training import config
from openpi.training import optimizer
from openpi.training import rtc_manifest
from openpi.training.utils import TrainState


def test_full_config_keeps_data_and_uses_unscaled_adamw():
    lora = config.get_config("pi05_piper_banana_training_rtc_lora")
    full = config.get_config("pi05_piper_banana_training_rtc_full")
    # Training startup must override a stale freeze filter in either direction.
    full = config.resolve_rtc_training_config(dataclasses.replace(full, freeze_filter=nnx.Everything))
    assert full.freeze_filter is nnx.Nothing
    assert full.model.paligemma_variant == "gemma_2b"
    assert full.model.action_expert_variant == "gemma_300m"
    assert full.data is lora.data
    assert full.weight_loader.params_path == "./data/pi05_base/params"
    assert full.model.action_horizon == lora.model.action_horizon == 30
    assert full.model.simulated_delay == lora.model.simulated_delay == 5
    assert full.model.rtc_loss_reduction == lora.model.rtc_loss_reduction == "official"
    assert full.model.rtc_time_distribution == lora.model.rtc_time_distribution == "beta"
    assert full.batch_size * full.rtc_validation_batches == 64
    assert full.batch_size == lora.batch_size == 16
    assert full.fsdp_devices == 2
    assert type(full.optimizer) is optimizer.AdamW
    schedule = full.lr_schedule.create()
    assert float(schedule(500)) == pytest.approx(1e-5)
    assert float(schedule(10_000)) == pytest.approx(1e-6)
    tx = full.optimizer.create(1e-5)
    params = {"vlm": jnp.ones(2), "expert": jnp.ones(2), "time": jnp.ones(2)}
    changes, _ = tx.update(params, tx.init(params), params)
    np.testing.assert_allclose(changes["vlm"], -1e-5, rtol=1e-4)
    np.testing.assert_array_equal(changes["vlm"], changes["time"])


def test_invalid_full_mode_and_optimizer_combinations_are_rejected():
    lora = config.get_config("pi05_piper_banana_training_rtc_lora")
    full = config.get_config("pi05_piper_banana_training_rtc_full")
    with pytest.raises(ValueError, match="without LoRA"):
        dataclasses.replace(lora.model, rtc_finetune_mode="full")
    with pytest.raises(ValueError, match="one learning rate"):
        config.resolve_rtc_training_config(dataclasses.replace(full, optimizer=lora.optimizer))
    with pytest.raises(ValueError, match="two-rate"):
        config.resolve_rtc_training_config(dataclasses.replace(lora, optimizer=full.optimizer))
    with pytest.raises(ValueError, match="ema_decay"):
        config.resolve_rtc_training_config(dataclasses.replace(full, ema_decay=0.99))


def test_full_jitted_step_updates_vision_language_expert_and_heads(tiny_model, monkeypatch):
    small_config, _, observation = tiny_model
    monkeypatch.setattr(gemma, "PALIGEMMA_VOCAB_SIZE", 128)
    small_config = dataclasses.replace(small_config, action_expert_variant="gemma_300m", rtc_finetune_mode="full")
    model = small_config.create(jax.random.key(3))
    # Pretrained adaRMS has nonzero gates AND kernels; zero kernels would block
    # the time MLP's gradient on the first randomly initialized model step.
    params = nnx.state(model)
    for path, value in params.flat_state().items():
        if "_norm_1/Dense_0/bias" in "/".join(path):
            value.value = jnp.ones_like(value.value) * 0.1
        if "_norm_1/Dense_0/kernel" in "/".join(path):
            value.value = jax.random.normal(jax.random.key(7), value.value.shape) * 0.01
    nnx.update(model, params)
    observation = dataclasses.replace(
        observation,
        tokenized_prompt=observation.tokenized_prompt % 128,
        action_is_pad=jnp.asarray([[False] * 5 + [True], [False] * 6]),
    )
    cfg = config.resolve_rtc_training_config(
        dataclasses.replace(
            config.get_config("pi05_piper_banana_training_rtc_full"),
            model=small_config,
        )
    )
    assert not params.filter(cfg.freeze_filter).flat_state()
    assert set(params.filter(cfg.trainable_filter).flat_state()) == set(params.flat_state())
    assert not any("lora" in "/".join(p) for p in params.flat_state())
    path = Path(__file__).resolve().parents[2] / "scripts/train.py"
    spec = importlib.util.spec_from_file_location("rtc_full_train_script", path)
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    tx = cfg.optimizer.create(1e-4)
    state = TrainState(
        step=0,
        params=params,
        model_def=nnx.graphdef(model),
        tx=tx,
        opt_state=tx.init(params),
        ema_decay=None,
        ema_params=None,
    )
    actions = jax.random.normal(jax.random.key(4), (2, 6, 4))
    new_state, info = jax.jit(functools.partial(train.train_step, cfg))(
        jax.random.key(5),
        state,
        (observation, actions),
    )
    assert int(new_state.step) == 1
    assert np.isfinite(float(info["loss"]))
    assert float(info["grad_norm"]) > 0
    before = params.flat_state()
    changed = {
        "/".join(p) for p, v in new_state.params.flat_state().items() if not np.array_equal(v.value, before[p].value)
    }
    for prefix in (
        "PaliGemma/img/",
        "PaliGemma/llm/embedder/",
        "time_mlp_in/",
        "time_mlp_out/",
        "action_in_proj/",
        "action_out_proj/",
    ):
        assert any(p.startswith(prefix) for p in changed), (prefix, changed)
    assert any("/attn/" in p and "_1/" not in p for p in changed)
    assert any("/mlp/" in p for p in changed)
    assert any("/attn/" in p and "_1/" in p for p in changed)
    assert any("/mlp_1/" in p for p in changed)
    assert any("_norm_1/Dense_0/" in p for p in changed)


@pytest.mark.parametrize("legacy_lora", [False, True])
def test_checkpoint_restores_full_or_legacy_lora_mode(tmp_path, legacy_lora):
    from test_manifest import DataStub

    full = config.get_config("pi05_piper_banana_training_rtc_full")
    lora = config.get_config("pi05_piper_banana_training_rtc_lora")
    saved = lora if legacy_lora else full
    template = full if legacy_lora else lora
    model_values = dataclasses.asdict(saved.model)
    if legacy_lora:
        model_values.pop("rtc_finetune_mode")
    manifest = {
        "model": model_values,
        "asset_id": "test",
        "split_seed": 42,
        "validation_fraction": 0.1,
        "train_episodes": [0, 2],
        "validation_episodes": [1],
        "use_quantile_norm": True,
        "raw_action_dim": 7,
    }
    restored = rtc_manifest.restore_config(dataclasses.replace(template, data=DataStub()), manifest, tmp_path)
    assert restored.model.rtc_finetune_mode == ("lora" if legacy_lora else "full")
    assert restored.model.action_expert_variant == saved.model.action_expert_variant
    if not legacy_lora:
        assert restored.freeze_filter is nnx.Nothing
