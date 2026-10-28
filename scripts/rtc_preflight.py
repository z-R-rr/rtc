"""Validate local RTC inputs and one transformed batch before starting a long training job."""

import argparse
import dataclasses
import json
import pathlib

import numpy as np

from openpi.training import config
from openpi.training import data_loader
from openpi.training import rtc_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="pi05_piper_training_rtc_lora")
    parser.add_argument("--repo-id")
    parser.add_argument("--base-params")
    parser.add_argument("--assets-dir")
    parser.add_argument("--asset-id")
    parser.add_argument("--output", default="outputs/preflight/result.json")
    args = parser.parse_args()
    cfg = config.get_config(args.config)
    data_kwargs = {}
    if args.repo_id:
        data_kwargs["repo_id"] = args.repo_id
    if args.assets_dir or args.asset_id:
        data_kwargs["assets"] = dataclasses.replace(
            cfg.data.assets,
            **({"assets_dir": args.assets_dir} if args.assets_dir else {}),
            **({"asset_id": args.asset_id} if args.asset_id else {}),
        )
    cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, **data_kwargs), batch_size=2, num_workers=0)
    if args.base_params:
        cfg = dataclasses.replace(
            cfg, weight_loader=dataclasses.replace(cfg.weight_loader, params_path=args.base_params)
        )
    paths = {
        "dataset": pathlib.Path(cfg.data.repo_id),
        "base_params": pathlib.Path(cfg.weight_loader.params_path),
        "norm_stats": pathlib.Path(cfg.data.assets.assets_dir) / cfg.data.assets.asset_id / "norm_stats.json",
    }
    missing = {k: str(v) for k, v in paths.items() if not v.exists()}
    if missing:
        raise FileNotFoundError(f"Local RTC inputs missing: {json.dumps(missing, indent=2)}")
    loader = data_loader.create_data_loader(cfg, shuffle=False, num_batches=1)
    obs, actions = next(iter(loader))
    if obs.action_is_pad is None:
        raise ValueError("Training data lost action_is_pad")
    if not np.isfinite(actions).all():
        raise ValueError("Nonfinite transformed actions")
    manifest = rtc_manifest.build(cfg, loader.data_config())
    result = {
        "paths": {k: str(v) for k, v in paths.items()},
        "actions_shape": list(actions.shape),
        "padded_tokens": int(np.asarray(obs.action_is_pad).sum()),
        "train_episodes": manifest["train_episodes"],
        "validation_episodes": manifest["validation_episodes"],
        "norm_stats_sha256": manifest["norm_stats_sha256"],
    }
    path = pathlib.Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
