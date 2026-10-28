# ruff: noqa: SLF001
import copy

import flax.nnx as nnx
import jax.numpy as jnp
import numpy as np
from openpi_client import msgpack_numpy
from openpi_client.rtc_queue import RTCQueue
import pytest

from openpi import transforms
from openpi.policies.piper_policy import PiperInputs
from openpi.policies.piper_policy import PiperOutputs
from openpi.policies.rtc_policy import TrainingRTCPolicy
from openpi.shared.normalize import NormStats


class DummyModel(nnx.Module):
    action_horizon = 6
    action_dim = 32
    simulated_delay = 5

    def sample_actions(self, rng, observation, **kwargs):
        return jnp.zeros((1, self.action_horizon, self.action_dim))


@pytest.mark.parametrize("quantile", [False, True])
def test_absolute_prefix_is_rebased_normalized_padded_once(quantile):
    stats = {
        key: NormStats(
            mean=np.arange(7) * 0.1,
            std=np.arange(7) * 0.1 + 0.2,
            q01=np.arange(7) * -0.2 - 1,
            q99=np.arange(7) * 0.2 + 1,
        )
        for key in ("state", "actions")
    }
    policy = TrainingRTCPolicy(
        DummyModel(),
        raw_action_dim=7,
        transforms=[
            PiperInputs(),
            transforms.DeltaActions((True,) * 7),
            transforms.Normalize(stats, use_quantiles=quantile),
            transforms.PadStatesAndActions(32),
        ],
        output_transforms=[
            transforms.Unnormalize(stats, use_quantiles=quantile),
            transforms.AbsoluteActions((True,) * 7),
            PiperOutputs(),
        ],
    )
    calls = []

    def sample(rng, observation, **kwargs):
        calls.append((observation, kwargs))
        return kwargs["rtc_prefix"]

    policy._sample_actions = sample
    obs = {
        "images": {"top_head": np.zeros((8, 8, 3), dtype=np.uint8), "hand_right": np.zeros((8, 8, 3), dtype=np.uint8)},
        "state": np.arange(7, dtype=np.float32),
    }
    absolute = np.stack([np.arange(7) + 0.3, np.arange(7) + 0.5]).astype(np.float32)
    obs["rtc"] = {"prefix": absolute, "delay": 2, "start_index": 12, "request_id": 8}
    before = copy.deepcopy(obs)
    response = policy.infer(obs)
    expected = transforms.Normalize({"actions": stats["actions"]}, use_quantiles=quantile)(
        {"actions": absolute - obs["state"]}
    )["actions"]
    np.testing.assert_allclose(calls[-1][1]["rtc_prefix"][0, :2, :7], expected)
    np.testing.assert_array_equal(calls[-1][1]["rtc_prefix"][0, :, 7:], 0)
    np.testing.assert_array_equal(response["actions"][:2], absolute)
    np.testing.assert_array_equal(obs["rtc"]["prefix"], before["rtc"]["prefix"])
    np.testing.assert_array_equal(obs["state"], before["state"])
    # Reusing the SAME absolute commands with a new observation must change model coordinates.
    obs["state"] += 0.2
    policy.infer(obs)
    assert not np.allclose(calls[-1][1]["rtc_prefix"][0, :2, :7], expected)
    obs["rtc"]["delay"] = 5
    with pytest.raises(ValueError, match="trained support"):
        policy.infer(obs)
    policy.set_inference_max_delay(5)
    obs["rtc"]["prefix"] = np.tile(obs["state"], (5, 1))
    response = policy.infer(obs)
    np.testing.assert_array_equal(response["actions"][:5], obs["rtc"]["prefix"])


def test_wire_protocol_reaches_policy_and_queue_through_obs_dictionary():
    stats = {key: NormStats(mean=np.zeros(7), std=np.ones(7)) for key in ("state", "actions")}
    policy = TrainingRTCPolicy(
        DummyModel(),
        raw_action_dim=7,
        transforms=[
            PiperInputs(),
            transforms.DeltaActions((True,) * 7),
            transforms.Normalize(stats),
            transforms.PadStatesAndActions(32),
        ],
        output_transforms=[transforms.Unnormalize(stats), transforms.AbsoluteActions((True,) * 7), PiperOutputs()],
    )
    obs = {
        "state": np.ones(7),
        "images": {"top_head": np.zeros((8, 8, 3), dtype=np.uint8), "hand_right": np.zeros((8, 8, 3), dtype=np.uint8)},
    }
    q = RTCQueue(6, 7, 4)
    packer = msgpack_numpy.Packer()

    def wire(request):
        # Same request/response serialization and infer(obs) call used by the official websocket server.
        received = msgpack_numpy.unpackb(packer.pack(request))
        return msgpack_numpy.unpackb(packer.pack(policy.infer(received)))

    assert q.merge(wire(q.begin_request(lambda: obs, delay=0))).accepted
    req = q.begin_request(lambda: obs, delay=2)
    q.pop()
    assert q.merge(wire(req)).accepted
    np.testing.assert_array_equal(q.pop(), np.ones(7))
