"""RTC flow matching: t=1 is noise, t=0 is clean, delay support is [0, S)."""

import math

import jax
import jax.numpy as jnp


def draw_delay(rng, batch_shape, upper_bound, weights=None):
    """Default P(d) ∝ exp(-d). Weights are a static tuple of S nonnegative values."""
    if upper_bound < 1:
        raise ValueError("Delay support must be nonempty")
    logits = -jnp.arange(upper_bound, dtype=jnp.float32)
    if weights is not None:
        if (len(weights) != upper_bound or any(not math.isfinite(v) or v < 0 for v in weights)
                or sum(weights) <= 0):
            raise ValueError("RTC weights must be finite, nonnegative, length S, and have positive sum")
        logits = jnp.log(jnp.asarray(weights, dtype=jnp.float32))
    return jax.random.categorical(rng, logits, shape=batch_shape)


def sample_time(rng, batch_size, distribution="beta"):
    if distribution == "beta":
        return jax.random.beta(rng, 1.5, 1, (batch_size,)) * 0.999 + 0.001
    if distribution == "uniform":
        return jax.random.uniform(rng, (batch_size,))
    raise ValueError("Unknown time distribution")


def prepare_training(actions, noise, time, delay, action_is_pad=None):
    """Construct one RTC batch in normalized model action coordinates.

    actions/noise: [B,H,D]; time/delay: [B]; padding: [B,H].
    Caller samples integer delay in [0,S), with S <= H. Arrays returned here
    are JAX pytrees; this function and flow_loss can be used inside jax.jit.
    action_valid MUST also mask action keys/queries in the model's attention.
    """
    actions, noise = jnp.asarray(actions), jnp.asarray(noise)
    time, delay = jnp.asarray(time), jnp.asarray(delay)
    if actions.ndim != 3 or actions.shape != noise.shape:
        raise ValueError("actions and noise must have matching [B,H,D] shapes")
    b, h, _ = actions.shape
    if time.shape != (b,) or delay.shape != (b,):
        raise ValueError("time and delay must have shape [B]")
    valid = jnp.ones((b, h), dtype=jnp.bool_)
    if action_is_pad is not None:
        if jnp.shape(action_is_pad) != (b, h):
            raise ValueError("action_is_pad must have shape [B,H]")
        valid = ~jnp.asarray(action_is_pad, dtype=jnp.bool_)
    prefix = jnp.arange(h)[None, :] < delay[:, None]
    model_time = jnp.where(prefix, 0.0, time[:, None])
    noisy_actions = model_time[..., None] * noise + (1 - model_time[..., None]) * actions
    return {
        "noisy_actions": noisy_actions,
        "model_time": model_time,
        "target_velocity": noise - actions,
        "active": valid & ~prefix,
        "action_valid": valid,
    }


def loss_grid(errors, active, reduction="official"):
    """Original OpenPI RTC reduction: mean(grid) = sum(active error)/count(tokens).

    The default SUMS over action dimensions; per_element additionally divides
    by D. Normalize over all active batch/time tokens, not per sample/device.
    """
    numerator = jnp.sum(jnp.where(active[..., None], errors, 0.0), axis=-1)
    denominator = jnp.maximum(jnp.sum(active), 1)
    if reduction == "per_element":
        denominator = denominator * errors.shape[-1]
    elif reduction != "official":
        raise ValueError(f"Unknown reduction: {reduction}")
    return numerator * (active.size / denominator)


def flow_loss(predicted_velocity, batch, reduction="official"):
    errors = jnp.square(predicted_velocity - batch["target_velocity"])
    return jnp.mean(loss_grid(errors, batch["active"], reduction))
