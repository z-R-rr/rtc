import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rtc_reference.sampling import hard_prefix_sample
from rtc_reference.training import draw_delay
from rtc_reference.training import flow_loss
from rtc_reference.training import prepare_training


def test_prepare_training_makes_clean_prefix_and_masks_episode_padding():
    actions = jnp.ones((2, 5, 2)) * 2
    noise = jnp.ones_like(actions) * 8
    batch = prepare_training(
        actions, noise, jnp.asarray([0.25, 0.5]), jnp.asarray([2, 1]),
        jnp.asarray([[False, False, False, False, True], [False, False, False, False, False]]),
    )
    np.testing.assert_allclose(batch["noisy_actions"][0, :2], 2)
    np.testing.assert_allclose(batch["noisy_actions"][0, 2:], 3.5)
    np.testing.assert_array_equal(batch["model_time"][0], [0, 0, 0.25, 0.25, 0.25])
    np.testing.assert_array_equal(batch["active"][0], [False, False, True, True, False])
    np.testing.assert_array_equal(batch["action_valid"][1], True)


def test_loss_global_active_token_reduction_and_gradients():
    errors = jnp.stack([jnp.ones((5, 2)), jnp.full((5, 2), 9.0)])
    active = jnp.arange(5)[None, :] >= jnp.asarray([0, 2])[:, None]
    loss = flow_loss(jnp.sqrt(errors), {"target_velocity": jnp.zeros_like(errors), "active": active})
    # 5 tokens * 2 dims * 1 + 3 tokens * 2 dims * 9, divided by 8 tokens.
    assert float(loss) == pytest.approx((10 + 54) / 8)
    batch = {"target_velocity": jnp.zeros_like(errors), "active": active}
    assert float(flow_loss(jnp.sqrt(errors), batch, "per_element")) == pytest.approx(float(loss) / 2)
    assert float(flow_loss(jnp.sqrt(errors), {**batch, "active": jnp.zeros_like(active)})) == 0
    grad = jax.grad(lambda prediction: flow_loss(
        prediction, {"target_velocity": jnp.zeros_like(errors), "active": active}
    ))(jnp.sqrt(errors))
    np.testing.assert_allclose(grad[0], 2 / 8)
    np.testing.assert_array_equal(grad[1, :2], 0)
    np.testing.assert_allclose(grad[1, 2:], 6 / 8)


def test_delay_distribution_and_explicit_weights():
    values = np.asarray(draw_delay(jax.random.key(3), (50_000,), 5))
    expected = np.exp(-np.arange(5))
    expected /= expected.sum()
    np.testing.assert_allclose(np.bincount(values, minlength=5) / len(values), expected, atol=0.006)
    assert np.all(np.asarray(draw_delay(jax.random.key(4), (20,), 5, (0, 0, 0, 1, 0))) == 3)
    for weights in ((0,) * 5, (-1, 1, 1, 1, 1), (float("nan"), 1, 1, 1, 1), (1, 2)):
        with pytest.raises(ValueError, match="RTC weights"):
            draw_delay(jax.random.key(0), (2,), 5, weights)


def test_prefix_stays_exact_through_all_euler_steps_and_conditions_continuation():
    noise = jnp.ones((1, 6, 2))
    prefix = jnp.zeros_like(noise).at[:, :2].set(7)
    delay = jnp.asarray([2])
    result = hard_prefix_sample(lambda x, t: x, noise, prefix, delay, num_steps=4)
    np.testing.assert_array_equal(result[:, :2], 7)
    np.testing.assert_allclose(result[:, 2:], (1 - 1 / 4) ** 4, atol=1e-6)
    changed_prefix = prefix.at[:, :2].set(-7)
    def conditioned_velocity(x, t):
        # The suffix depends on the conditioned commands AND the per-token time.
        # Nonzero prefix time would change the continuation and fail this oracle.
        return jnp.broadcast_to(x[:, :1] + t[:, 2:3, None] + jnp.sum(t[:, :2]), x.shape)
    a = jax.jit(lambda p: hard_prefix_sample(conditioned_velocity, noise, p, delay, num_steps=4))(prefix)
    b = hard_prefix_sample(conditioned_velocity, noise, changed_prefix, delay, num_steps=4)
    np.testing.assert_array_equal(a[:, :2], 7)
    np.testing.assert_allclose(a[:, 2:], -6.625)
    np.testing.assert_allclose(b[:, 2:], 7.375)


def test_invalid_rtc_shapes_and_sampler_steps_fail_early():
    with pytest.raises(ValueError, match="matching"):
        prepare_training(jnp.ones((1, 3, 2)), jnp.ones((1, 4, 2)), jnp.ones(1), jnp.zeros(1))
    with pytest.raises(ValueError, match="positive static integer"):
        hard_prefix_sample(lambda x, t: x, jnp.ones((1, 3, 2)), jnp.ones((1, 3, 2)), jnp.zeros(1), num_steps=0)
