import dataclasses
import itertools

import jax
import jax.numpy as jnp
import numpy as np

from openpi.training.data_loader import rtc_validation_indices


def test_validation_budget_covers_all_episodes_and_both_time_ends():
    lengths = [500, 400, 600, 300, 700, 800]
    indices = rtc_validation_indices(lengths, 64)
    bounds = np.cumsum([0, *lengths])
    assert len(indices) == len(set(indices)) == 64
    for lo, hi in itertools.pairwise(bounds):
        assert lo in indices
        assert hi - 1 in indices
        assert sum(lo <= i < hi for i in indices) >= 10
    assert indices == rtc_validation_indices(lengths, 64)
    assert sorted(rtc_validation_indices([1, 2, 4], 7)) == list(range(7))


def test_padded_targets_cannot_influence_valid_predictions(tiny_model):
    _, model, obs = tiny_model
    # Nonzero gates exercise attention; random zero-gated blocks hide leakage.
    import flax.nnx as nnx

    params = nnx.state(model)
    for path, value in params.flat_state().items():
        if "_norm_1/Dense_0/bias" in "/".join(path):
            value.value = jnp.ones_like(value.value) * 0.2
    nnx.update(model, params)
    padding = jnp.array([[False, False, False, True, True, True]] * 2)
    obs = dataclasses.replace(obs, action_is_pad=padding)
    actions = jnp.ones((2, 6, 4))
    noise = jax.random.normal(jax.random.key(7), actions.shape)
    kwargs = {"train": False, "rtc_delay": jnp.array([1, 1]), "noise": noise, "time": jnp.array([0.4, 0.7])}
    errors, mask = model.flow_errors(jax.random.key(2), obs, actions, **kwargs)
    changed = actions.at[:, 3:].set(1000)
    errors2, mask2 = model.flow_errors(jax.random.key(2), obs, changed, **kwargs)
    np.testing.assert_array_equal(mask, mask2)
    np.testing.assert_allclose(np.asarray(errors)[np.asarray(mask)], np.asarray(errors2)[np.asarray(mask)], atol=1e-6)
