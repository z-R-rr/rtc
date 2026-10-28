"""Inspect full pi0.5 parameter shapes without allocating model weights or loading a checkpoint."""

import argparse
import json
import math
import pathlib

import flax.nnx as nnx
import jax

from openpi.training import config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="pi05_piper_training_rtc_lora")
    parser.add_argument("--output", default="outputs/verification/parameters.json")
    args = parser.parse_args()
    cfg = config.resolve_rtc_training_config(config.get_config(args.config))
    full = cfg.model.rtc_finetune_mode == "full"
    model = nnx.eval_shape(cfg.model.create, jax.random.key(0))
    all_params = nnx.state(model, nnx.Param).flat_state()
    trainable = nnx.state(model, cfg.trainable_filter).flat_state()
    rows = [
        {
            "path": "/".join(p),
            "shape": list(v.value.shape),
            "count": math.prod(v.value.shape),
            "lr_group": "all" if full else ("lora" if "lora" in "/".join(p) else "inherited"),
        }
        for p, v in trainable.items()
    ]
    assert rows
    if full:
        assert set(trainable) == set(all_params)
        assert any("/img/" in r["path"] for r in rows)
        assert all("lora" not in r["path"] for r in rows)
    else:
        assert all("/img/" not in r["path"] for r in rows)
        assert all("_1/" in r["path"] for r in rows if r["path"].startswith("PaliGemma/llm"))
    assert any("final_norm_1/Dense_0/kernel" in r["path"] for r in rows)
    result = {
        "config": cfg.name,
        "finetune_mode": cfg.model.rtc_finetune_mode,
        "total_parameters": sum(math.prod(v.value.shape) for v in all_params.values()),
        "trainable_parameters": sum(r["count"] for r in rows),
        "trainable": rows,
    }
    path = pathlib.Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "trainable"}, indent=2))
    print("Parameter paths:", path)


if __name__ == "__main__":
    main()
