"""Model-independent Euler sampler illustrating training-time RTC hard prefixes."""

import jax
import jax.numpy as jnp


def hard_prefix_sample(velocity, noise, prefix, delay, *, num_steps=10):
    """velocity(x, time) receives [B,H,D] actions and [B,H] per-action time.

    Observation encoding and KV caches belong in the velocity closure. Prefix
    is [B,H,D] in the SAME normalized coordinates as model actions; only the
    first delay[b] entries are used. Require 0 <= delay < H before calling.
    num_steps is a positive static integer. No inference-time guidance/VJP.
    """
    noise, prefix, delay = jnp.asarray(noise), jnp.asarray(prefix), jnp.asarray(delay)
    if noise.ndim != 3 or prefix.shape != noise.shape or delay.shape != noise.shape[:1]:
        raise ValueError("Expected noise/prefix [B,H,D] and delay [B]")
    if not isinstance(num_steps, int) or isinstance(num_steps, bool) or num_steps <= 0:
        raise ValueError("num_steps must be a positive static integer")
    mask = jnp.arange(noise.shape[1])[None, :] < delay[:, None]
    dt = -1.0 / num_steps

    def step(i, x):
        x = jnp.where(mask[..., None], prefix, x)
        time = jnp.where(mask, 0.0, 1.0 + i * dt)
        updated = x + dt * velocity(x, time)
        return jnp.where(mask[..., None], prefix, updated)

    return jax.lax.fori_loop(0, num_steps, step, noise)
