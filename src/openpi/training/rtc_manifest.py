"""Checkpoint provenance for model semantics, delta convention and episode split."""

import dataclasses
import hashlib
import json
import pathlib

import numpy as np

OPENPI_REVISION = "215abfb217dbac7d5f1273282331b9b1866c0479"
RTC_REVISION = "9296f31d62d5bfeb5779dcb2f9bcf71ca37f448b"


def stats_digest(stats):
    digest = hashlib.sha256()
    for name in sorted(stats):
        for field in ("mean", "std", "q01", "q99"):
            value = getattr(stats[name], field)
            if value is not None:
                array = np.asarray(value, dtype=np.float64)
                digest.update(f"{name}/{field}/{array.shape}".encode())
                digest.update(array.tobytes())
    return digest.hexdigest()


def source_digest():
    root = pathlib.Path(__file__).resolve().parents[3]
    digest = hashlib.sha256()
    for folder in ("src/openpi", "scripts", "packages/openpi-client/src"):
        for path in sorted((root / folder).rglob("*.py")):
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def build(config, data_config):
    if data_config.norm_stats is None:
        raise ValueError("RTC requires normalization statistics")
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

    from openpi.training import config as configs

    train, val = configs.rtc_episode_split(
        LeRobotDatasetMetadata(data_config.repo_id).total_episodes,
        data_config.rtc_validation_fraction,
        data_config.rtc_split_seed,
        train_episodes=data_config.rtc_train_episodes,
        validation_episodes=data_config.rtc_validation_episodes,
    )
    if tuple(data_config.episodes) != train:
        raise ValueError("Training loader does not use the expected train episode split")
    stats_source = (
        pathlib.Path(config.data.assets.assets_dir or config.assets_dirs)
        / data_config.asset_id
        / "norm_stats_manifest.json"
    )
    if stats_source.exists():
        stats_info = json.loads(stats_source.read_text())
        if (
            stats_info.get("smoke_only")
            or stats_info["train_episodes"] != list(train)
            or stats_info["validation_episodes"] != list(val)
            or stats_info["action_horizon"] != config.model.action_horizon
            or stats_info["norm_stats_sha256"] != stats_digest(data_config.norm_stats)
        ):
            raise ValueError("RTC statistics do not match the active episode split/horizon; recompute normalization")
    result = {
        "schema": 1,
        "source_sha256": source_digest(),
        "upstream_openpi": OPENPI_REVISION,
        "upstream_rtc": RTC_REVISION,
        "config_name": config.name,
        "model": dataclasses.asdict(config.model),
        "optimizer": dataclasses.asdict(config.optimizer),
        "lr_schedule": dataclasses.asdict(config.lr_schedule),
        "base_checkpoint": dataclasses.asdict(config.weight_loader),
        "batch_size": config.batch_size,
        "seed": config.seed,
        "num_train_steps": config.num_train_steps,
        "repo_id": data_config.repo_id,
        "asset_id": data_config.asset_id,
        "raw_action_dim": data_config.rtc_raw_action_dim,
        "delta_mask": list(data_config.rtc_delta_mask),
        "use_quantile_norm": data_config.use_quantile_norm,
        "norm_stats_sha256": stats_digest(data_config.norm_stats),
        "split_method": "explicit" if data_config.rtc_train_episodes is not None else "seeded_random",
        "split_seed": data_config.rtc_split_seed,
        "validation_fraction": data_config.rtc_validation_fraction,
        "train_episodes": list(train),
        "validation_episodes": list(val),
        "freeze_strategy": (
            "Full fine-tuning: all VLM, action expert, adaRMS, time MLP and action projection parameters trainable"
            if config.model.rtc_finetune_mode == "full"
            else "VLM frozen; expert LoRA + adaRMS + time MLP + action input/output projections trainable"
        ),
    }
    return json.loads(json.dumps(result))  # Stable tuple/list representation for resume comparison.


def read(checkpoint_dir):
    path = pathlib.Path(checkpoint_dir) / "assets" / "rtc_manifest.json"
    if not path.exists():
        raise ValueError(f"RTC checkpoint lacks provenance: {path}")
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != 1:
        raise ValueError("Unsupported RTC manifest schema")
    return manifest


def restore_config(config, manifest, checkpoint_dir):
    """Inference/validation follow checkpoint model values, including nondefault delay PMFs."""
    from openpi.training.config import AssetsConfig

    model_values = dict(manifest["model"])
    # Checkpoints created before this field existed always used the RTC LoRA recipe.
    model_values.setdefault("rtc_finetune_mode", "lora")
    if model_values.get("simulated_delay_weights") is not None:
        model_values["simulated_delay_weights"] = tuple(model_values["simulated_delay_weights"])
    model = dataclasses.replace(config.model, **model_values)
    data = dataclasses.replace(
        config.data,
        assets=AssetsConfig(assets_dir=str(pathlib.Path(checkpoint_dir) / "assets"), asset_id=manifest["asset_id"]),
        split_seed=manifest["split_seed"],
        validation_fraction=manifest["validation_fraction"],
        train_episodes=tuple(manifest["train_episodes"]) if "train_episodes" in manifest else None,
        validation_episodes=tuple(manifest["validation_episodes"]) if "validation_episodes" in manifest else None,
    )
    restored = dataclasses.replace(config, model=model, data=data, freeze_filter=model.get_rtc_freeze_filter())
    data_config = data.create(restored.assets_dirs, model)
    if data_config.use_quantile_norm != manifest["use_quantile_norm"]:
        raise ValueError("Normalization mode differs from checkpoint")
    if data_config.rtc_raw_action_dim != manifest["raw_action_dim"]:
        raise ValueError("Raw action dimension differs from checkpoint")
    return restored
