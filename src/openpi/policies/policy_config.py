import logging
import os
import pathlib
from typing import Any

import jax.numpy as jnp

import openpi.models.model as _model
import openpi.policies.policy as _policy
import openpi.shared.download as download
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
import openpi.transforms as transforms


def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
    pytorch_device: str | None = None,
) -> _policy.Policy:
    """Create a policy from a trained checkpoint.

    Args:
        train_config: The training config to use to create the model.
        checkpoint_dir: The directory to load the model from.
        repack_transforms: Optional transforms that will be applied before any other transforms.
        sample_kwargs: The kwargs to pass to the `sample_actions` method. If not provided, the default
            kwargs will be used.
        default_prompt: The default prompt to use for the policy. Will inject the prompt into the input
            data if it doesn't already exist.
        norm_stats: The norm stats to use for the policy. If not provided, the norm stats will be loaded
            from the checkpoint directory.
        pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda", "cuda:0").
                      If None and is_pytorch=True, will use "cuda" if available, otherwise "cpu".

    Note:
        The function automatically detects whether the model is PyTorch-based by checking for the
        presence of "model.safensors" in the checkpoint directory.
    """
    repack_transforms = repack_transforms or transforms.Group()
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))

    rtc_provenance = None
    if getattr(train_config.model, "simulated_delay", None) is not None:
        from openpi.training import rtc_manifest

        rtc_provenance = rtc_manifest.read(checkpoint_dir)
        train_config = rtc_manifest.restore_config(train_config, rtc_provenance, checkpoint_dir)

    # Check if this is a PyTorch model by looking for model.safetensors
    weight_path = os.path.join(checkpoint_dir, "model.safetensors")
    is_pytorch = os.path.exists(weight_path)

    rtc_enabled = getattr(train_config.model, "simulated_delay", None) is not None
    if rtc_enabled and is_pytorch:
        raise ValueError("Training RTC only supports JAX checkpoints")
    logging.info("Loading model...")
    if is_pytorch:
        model = train_config.model.load_pytorch(train_config, weight_path)
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    else:
        # Preserve trainable FP32 RTC parameters for consistent online/offline validation.
        model = train_config.model.load(
            _model.restore_params(checkpoint_dir / "params", dtype=None if rtc_enabled else jnp.bfloat16)
        )
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if norm_stats is None:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)

    # Determine the device to use for PyTorch models
    if is_pytorch and pytorch_device is None:
        try:
            import torch

            pytorch_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            pytorch_device = "cpu"

    policy_cls = _policy.Policy
    extra_kwargs = {}
    if rtc_enabled:
        from openpi.policies.rtc_policy import TrainingRTCPolicy

        if data_config.rtc_raw_action_dim is None or data_config.rtc_delta_mask is None:
            raise ValueError("RTC requires an explicit delta data configuration")
        if norm_stats is None or "actions" not in norm_stats or "state" not in norm_stats:
            raise ValueError("RTC requires checkpoint action and state normalization statistics")
        if repack_transforms.inputs or repack_transforms.outputs:
            raise ValueError("RTC wire observations must already use canonical state/images keys")
        if rtc_manifest.stats_digest(norm_stats) != rtc_provenance["norm_stats_sha256"]:
            raise ValueError("RTC normalization statistics do not match checkpoint provenance")
        if list(data_config.rtc_delta_mask) != rtc_provenance["delta_mask"]:
            raise ValueError("RTC delta convention differs from training")
        policy_cls = TrainingRTCPolicy
        extra_kwargs["raw_action_dim"] = data_config.rtc_raw_action_dim

    return policy_cls(
        model,
        **extra_kwargs,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata={
            **train_config.policy_metadata,
            **(
                {
                    "action_horizon": train_config.model.action_horizon,
                    "simulated_delay": train_config.model.simulated_delay,
                    "raw_action_dim": data_config.rtc_raw_action_dim,
                }
                if rtc_enabled
                else {}
            ),
        },
        is_pytorch=is_pytorch,
        pytorch_device=pytorch_device if is_pytorch else None,
    )
