"""Compute train-only delta norm stats from local numeric Parquet columns, without video decoding."""

import argparse
import dataclasses

from openpi.training import config
from openpi.training.rtc_norm_stats import compute


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="pi05_piper_banana_training_rtc_lora")
    parser.add_argument("--repo-id", help="Relocated local dataset with the same episode IDs and contents")
    parser.add_argument("--output-dir", help="Override output assets directory (required for a limited smoke)")
    parser.add_argument("--max-episodes", type=int, help="Limited smoke run; never use these statistics for training")
    args = parser.parse_args()
    cfg = config.get_config(args.config_name)
    if args.repo_id:
        cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, repo_id=args.repo_id))
    compute(cfg, output_dir=args.output_dir, max_episodes=args.max_episodes)


if __name__ == "__main__":
    main()
