"""Optional fixed initial noise for policies accepting infer(obs, noise=...)."""
import numpy as np
from openpi_client.base_policy import BasePolicy


class FixedNoisePolicy(BasePolicy):
    """Sample N(0, variance I); reuse noise when seed is set, otherwise resample."""

    def __init__(self, policy, *, seed: int | None, shape: tuple[int, int], variance: float = 1.0):
        self._policy = policy
        if not np.isfinite(variance) or variance <= 0:
            raise ValueError("Noise variance must be finite and positive")
        self._rng = np.random.default_rng(seed)
        self._scale = np.float32(np.sqrt(variance))
        self.noise_shape = shape
        self._noise = self._sample() if seed is not None else None

    def _sample(self):
        return self._rng.standard_normal(self.noise_shape).astype(np.float32) * self._scale

    def infer(self, obs):
        # Isolate the template from policies that mutate their noise argument.
        return self._policy.infer(obs, noise=self._noise.copy() if self._noise is not None else self._sample())

    def reset(self):
        self._policy.reset()
