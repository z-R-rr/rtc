# ruff: noqa: SLF001
import dataclasses

import flax.linen as nn
import jax
import pytest

from openpi.models import gemma
from openpi.models import pi0
from openpi.models import pi0_config


class TinyVision(nn.Module):
    num_classes: int
    variant: str
    pool_type: str
    scan: bool
    dtype_mm: str

    @nn.compact
    def __call__(self, image, *, train=False):
        x = image.mean(axis=(1, 2))[:, None, :]
        return nn.Dense(self.num_classes, dtype=self.dtype_mm)(x), {}


@pytest.fixture
def tiny_model(monkeypatch):
    original = gemma.get_config

    def small(variant):
        c = original(variant)
        return dataclasses.replace(c, width=32, depth=2, mlp_dim=64, num_heads=2, num_kv_heads=1, head_dim=16)

    monkeypatch.setattr(gemma, "get_config", small)
    monkeypatch.setattr(pi0._siglip, "Module", TinyVision)
    cfg = pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m_lora",
        dtype="float32",
        action_dim=4,
        action_horizon=6,
        max_token_len=4,
        simulated_delay=5,
    )
    model = cfg.create(jax.random.key(0))
    obs = cfg.fake_obs(batch_size=2)
    return cfg, model, obs
