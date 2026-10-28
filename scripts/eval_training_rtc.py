"""Validate a checkpoint at every trained delay, with no image augmentation."""

import argparse
import dataclasses
import json
import pathlib

import jax

from openpi.policies import policy_config
from openpi.shared import nnx_utils
from openpi.training import config
from openpi.training import data_loader
from openpi.training import rtc_manifest
from openpi.training import rtc_validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="pi05_piper_training_rtc_lora")
    parser.add_argument("--repo-id", help="Optional relocated dataset with identical episode IDs/content")
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = rtc_manifest.read(args.checkpoint)
    cfg = rtc_manifest.restore_config(config.get_config(args.config), manifest, args.checkpoint)
    cfg = dataclasses.replace(
        cfg,
        batch_size=args.batch_size,
        num_workers=0,
        data=dataclasses.replace(cfg.data, split="validation", repo_id=args.repo_id or manifest["repo_id"]),
    )
    policy = policy_config.create_trained_policy(cfg, args.checkpoint)
    dataset_config = data_loader.resolve_rtc_episodes(cfg.data.create(cfg.assets_dirs, cfg.model))
    if list(dataset_config.episodes) != manifest["validation_episodes"]:
        raise ValueError("Validation episode IDs changed from checkpoint manifest")
    loader = data_loader.create_data_loader(cfg, shuffle=False, num_batches=args.batches)
    policy._model.eval()  # noqa: SLF001 - diagnostics use the restored model, not action sampling.
    # module_jit captures this checkpoint once. Static args define the diagnostic's raw dimensions.
    evaluate_delay = nnx_utils.module_jit(policy._model.rtc_validation_sums, static_argnames=("raw_action_dim",))  # noqa: SLF001

    def evaluate(rng, batch):
        return jax.tree.map(
            lambda *xs: jax.numpy.stack(xs),
            *[
                evaluate_delay(rng, *batch, delay=jax.numpy.asarray(d), raw_action_dim=manifest["raw_action_dim"])
                for d in range(cfg.model.simulated_delay)
            ],
        )

    metrics = rtc_validation.evaluate_batches(
        evaluate,
        loader,
        seed=manifest["seed"],
        num_batches=args.batches,
        raw_action_dim=manifest["raw_action_dim"],
        model_action_dim=cfg.model.action_dim,
        reduction=cfg.model.rtc_loss_reduction,
    )
    result = {
        "checkpoint": args.checkpoint,
        "validation_episodes": manifest["validation_episodes"],
        "note": "Held out from RTC adaptation; the task base checkpoint may have seen these episodes.",
        "metrics": metrics,
    }
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
