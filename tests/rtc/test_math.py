import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models.pi0_config import Pi0Config
from openpi.models.training_rtc import draw_delay
from openpi.models.training_rtc import loss_grid
from openpi.training.config import rtc_episode_split
from openpi.training.optimizer import RTCAdamW


def test_official_reduction_and_gradient_with_mixed_lengths():
    errors = jnp.stack([jnp.ones((30, 32)), jnp.full((30, 32), 9.0)])
    active = jnp.arange(30)[None] >= jnp.asarray([0, 4])[:, None]
    expected = (30 + 26 * 9) * 32 / 56
    assert float(loss_grid(errors, active).mean()) == pytest.approx(expected)
    assert float(loss_grid(errors, active, "per_element").mean()) == pytest.approx(expected / 32)
    grad = jax.grad(lambda e: loss_grid(e, active).mean())(errors)
    np.testing.assert_allclose(grad[0], 1 / 56)
    np.testing.assert_allclose(grad[1, :4], 0)
    np.testing.assert_allclose(grad[1, 4:], 1 / 56)
    assert float(loss_grid(errors, jnp.zeros_like(active)).mean()) == 0


def test_delay_distribution_and_validation():
    values = np.asarray(draw_delay(jax.random.key(2), (100_000,), 5))
    expected = np.exp(-np.arange(5))
    expected /= expected.sum()
    np.testing.assert_allclose(np.bincount(values, minlength=5) / len(values), expected, atol=0.004)
    assert np.all(np.asarray(draw_delay(jax.random.key(2), (20,), 5, (0, 0, 0, 1, 0))) == 3)
    for weights in [(0,) * 5, (-1, 1, 1, 1, 1), (float("nan"), 1, 1, 1, 1), (1, 2)]:
        with pytest.raises(ValueError, match="RTC weights"):
            Pi0Config(pi05=True, simulated_delay=5, simulated_delay_weights=weights)


def test_heldout_split():
    train, val = rtc_episode_split(23, 0.1, 42)
    assert len(val) == 3
    assert set(train).isdisjoint(val)
    assert sorted(train + val) == list(range(23))
    assert (train, val) == rtc_episode_split(23, 0.1, 42)


def test_inherited_parameters_really_use_smaller_lr():
    params = {"lora_a": jnp.ones((2,)), "time_mlp": jnp.ones((2,))}
    tx = RTCAdamW(inherited_lr_scale=0.2, weight_decay=0).create(1e-3)
    updates, _ = tx.update(jax.tree.map(jnp.ones_like, params), tx.init(params), params)
    np.testing.assert_allclose(updates["time_mlp"], updates["lora_a"] * 0.2, rtol=1e-5)
