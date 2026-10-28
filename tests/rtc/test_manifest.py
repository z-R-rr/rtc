# ruff: noqa: SLF001
import dataclasses
import json
from types import SimpleNamespace

import numpy as np
import pytest

from openpi.shared.normalize import NormStats
from openpi.training import config
from openpi.training import rtc_manifest


@dataclasses.dataclass(frozen=True)
class DataStub:
    assets: config.AssetsConfig = dataclasses.field(default_factory=config.AssetsConfig)
    train_episodes: tuple[int, ...] | None = None
    validation_episodes: tuple[int, ...] | None = None
    split_seed: int = 0
    validation_fraction: float = 0.1

    def create(self, assets_dirs, model):
        return SimpleNamespace(use_quantile_norm=True, rtc_raw_action_dim=7)


def test_checkpoint_restores_rtc_model_settings_and_assets(tmp_path):
    cfg = dataclasses.replace(config.get_config("pi05_piper_training_rtc_lora"), data=DataStub())
    model = dataclasses.replace(
        cfg.model,
        action_horizon=40,
        simulated_delay=6,
        simulated_delay_weights=(0.1, 0.1, 0.1, 0.1, 0.2, 0.4),
        rtc_time_distribution="uniform",
    )
    manifest = {
        "schema": 1,
        "model": dataclasses.asdict(model),
        "asset_id": "piper",
        "split_seed": 12,
        "train_episodes": [0, 2, 4],
        "validation_episodes": [1, 3],
        "validation_fraction": 0.2,
        "use_quantile_norm": True,
        "raw_action_dim": 7,
    }
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "rtc_manifest.json").write_text(json.dumps(manifest))
    restored = rtc_manifest.restore_config(cfg, rtc_manifest.read(tmp_path), tmp_path)
    assert restored.model == model
    assert restored.data.split_seed == 12
    assert restored.data.train_episodes == (0, 2, 4)
    assert restored.data.validation_episodes == (1, 3)
    assert restored.data.assets.assets_dir == str(tmp_path / "assets")
    manifest["use_quantile_norm"] = False
    with pytest.raises(ValueError, match="Normalization mode"):
        rtc_manifest.restore_config(cfg, manifest, tmp_path)


def test_normalization_digest_detects_changed_statistics():
    stats = {"actions": NormStats(mean=np.zeros(7), std=np.ones(7))}
    first = rtc_manifest.stats_digest(stats)
    stats["actions"].std[0] = 2
    assert rtc_manifest.stats_digest(stats) != first


@pytest.mark.parametrize(
    "mismatch",
    [
        None,
        {"smoke_only": True},
        {"train_episodes": [0, 2]},
        {"validation_episodes": [0]},
        {"action_horizon": 40},
        {"norm_stats_sha256": "wrong"},
    ],
)
def test_training_rejects_statistics_from_another_split_or_smoke(tmp_path, monkeypatch, mismatch):
    from lerobot.common.datasets import lerobot_dataset

    monkeypatch.setattr(lerobot_dataset, "LeRobotDatasetMetadata", lambda *_: SimpleNamespace(total_episodes=3))
    monkeypatch.setattr(rtc_manifest, "source_digest", lambda: "test-source")
    cfg = config.get_config("pi05_piper_banana_training_rtc_lora")
    cfg = dataclasses.replace(
        cfg,
        data=dataclasses.replace(
            cfg.data,
            assets=config.AssetsConfig(assets_dir=str(tmp_path), asset_id="stats"),
            train_episodes=(0, 1),
            validation_episodes=(2,),
        ),
    )
    stats = {"actions": NormStats(mean=np.zeros(7), std=np.ones(7))}
    data = config.DataConfig(
        repo_id="unused",
        asset_id="stats",
        episodes=(0, 1),
        rtc_train_episodes=(0, 1),
        rtc_validation_episodes=(2,),
        rtc_raw_action_dim=7,
        rtc_delta_mask=(True,) * 7,
        norm_stats=stats,
    )
    manifest = {
        "smoke_only": False,
        "train_episodes": [0, 1],
        "validation_episodes": [2],
        "action_horizon": cfg.model.action_horizon,
        "norm_stats_sha256": rtc_manifest.stats_digest(stats),
    }
    manifest.update(mismatch or {})
    (tmp_path / "stats").mkdir()
    (tmp_path / "stats/norm_stats_manifest.json").write_text(json.dumps(manifest))
    if mismatch:
        with pytest.raises(ValueError, match="statistics do not match"):
            rtc_manifest.build(cfg, data)
    else:
        result = rtc_manifest.build(cfg, data)
        assert result["train_episodes"] == [0, 1]
        assert result["validation_episodes"] == [2]
        assert result["split_method"] == "explicit"


def test_base_checkpoint_vlm_lora_cannot_be_silently_discarded(monkeypatch):
    from openpi.training import weight_loaders

    loaded = {"PaliGemma": {"llm": {"attn": {"lora_a": np.ones(2)}}}}
    monkeypatch.setattr(weight_loaders._model, "restore_params", lambda *args, **kwargs: loaded)
    monkeypatch.setattr(weight_loaders.download, "maybe_download", lambda x: x)
    with pytest.raises(ValueError, match="adapters absent"):
        weight_loaders.RTCCheckpointWeightLoader("unused").load({"x": np.zeros(2)})
