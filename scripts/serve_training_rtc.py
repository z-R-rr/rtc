"""Serve a JAX Training RTC checkpoint using the official OpenPI websocket protocol."""

import dataclasses
import logging
import math

from openpi_client.fixed_noise_policy import FixedNoisePolicy
import tyro

from openpi.policies import policy_config
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.training import config
from openpi.training import rtc_manifest


@dataclasses.dataclass
class Args:
    checkpoint: str
    config: str = "pi05_piper_training_rtc_lora"
    port: int = 8000
    host: str = "0.0.0.0"
    num_steps: int = 10
    default_prompt: str | None = None
    # Reuse the same action-expert initial Gaussian noise on every request.
    fixed_noise_seed: int | None = None
    # Optional serving override for the maximum committed prefix. Values above the
    # checkpoint's trained support are inference extrapolation and should be validated
    # on the target hardware.
    max_delay: int | None = None
    # Initial action noise covariance is noise_variance * I (not standard deviation).
    noise_variance: float = 1.0


def main(args: Args):
    if not math.isfinite(args.noise_variance) or args.noise_variance <= 0:
        raise ValueError("noise_variance must be finite and positive")
    cfg = config.get_config(args.config)
    if getattr(cfg.model, "simulated_delay", None) is None:
        raise ValueError("Select an RTC config")
    policy = policy_config.create_trained_policy(
        cfg, args.checkpoint, sample_kwargs={"num_steps": args.num_steps}, default_prompt=args.default_prompt
    )
    metadata = dict(policy.metadata)
    if args.max_delay is not None:
        setter = getattr(policy, "set_inference_max_delay", None)
        if setter is None:
            raise ValueError("--max-delay requires an RTC policy")
        setter(args.max_delay)
        metadata["rtc_max_delay"] = args.max_delay
        logging.info("RTC serving max delay override: %s", args.max_delay)
    if args.fixed_noise_seed is not None or args.noise_variance != 1.0:
        manifest = rtc_manifest.read(args.checkpoint)
        model = manifest["model"]
        policy = FixedNoisePolicy(
            policy, seed=args.fixed_noise_seed, variance=args.noise_variance,
            shape=(model["action_horizon"], model["action_dim"]),
        )
        metadata["fixed_noise_seed"] = args.fixed_noise_seed
        logging.info("Expert sampling noise: fixed_seed=%s, variance=%s, shape=%s",
                     args.fixed_noise_seed, args.noise_variance, policy.noise_shape)
    metadata["noise_variance"] = args.noise_variance
    logging.info("RTC metadata: %s", metadata)
    WebsocketPolicyServer(policy, host=args.host, port=args.port, metadata=metadata).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))
