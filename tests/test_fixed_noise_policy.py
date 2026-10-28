import numpy as np
from openpi_client.fixed_noise_policy import FixedNoisePolicy


class RecordingPolicy:
    def infer(self, obs, *, noise):
        result = {"obs": obs, "noise": noise.copy()}
        noise[:] = 0
        return result

    def reset(self):
        pass


def test_noise_repeats_across_requests_resets_and_instances():
    policy = FixedNoisePolicy(RecordingPolicy(), seed=42, shape=(30, 32))
    first = policy.infer({"rtc": {"delay": 0}})
    policy.reset()
    second = policy.infer({"rtc": {"delay": 4}})
    np.testing.assert_array_equal(first["noise"], second["noise"])
    assert second["obs"]["rtc"]["delay"] == 4
    assert first["noise"].shape == (30, 32)
    assert first["noise"].dtype == np.float32
    np.testing.assert_array_equal(first["noise"], FixedNoisePolicy(
        RecordingPolicy(), seed=42, shape=(30, 32)).infer({})["noise"])
    assert not np.array_equal(first["noise"], FixedNoisePolicy(
        RecordingPolicy(), seed=43, shape=(30, 32)).infer({})["noise"])


def test_half_variance_scales_standard_deviation_and_keeps_fixed_noise():
    standard = FixedNoisePolicy(RecordingPolicy(), seed=42, shape=(30, 32)).infer({})["noise"]
    policy = FixedNoisePolicy(RecordingPolicy(), seed=42, shape=(30, 32), variance=0.5)
    scaled = policy.infer({})["noise"]
    np.testing.assert_array_equal(scaled, standard * np.float32(np.sqrt(0.5)))
    np.testing.assert_array_equal(scaled, policy.infer({})["noise"])


def test_random_mode_resamples_with_requested_variance():
    policy = FixedNoisePolicy(RecordingPolicy(), seed=None, shape=(1000, 1000), variance=0.5)
    first = policy.infer({})["noise"]
    assert not np.array_equal(first, policy.infer({})["noise"])
    assert abs(float(first.mean())) < 0.01
    assert abs(float(first.var()) - 0.5) < 0.01


def test_invalid_variance_rejected():
    import pytest
    for variance in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            FixedNoisePolicy(RecordingPolicy(), seed=42, shape=(30, 32), variance=variance)
