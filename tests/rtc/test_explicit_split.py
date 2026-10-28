import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from openpi.training import config
from openpi.training import data_loader


def test_banana_config_loader_and_snapshot_use_same_explicit_split(monkeypatch):
    cfg = config.get_config("pi05_piper_banana_training_rtc_lora")
    snapshot = json.loads((Path(__file__).resolve().parents[2] / "configs/banana_train_validation.json").read_text())
    train, val = config.rtc_episode_split(
        51, 0.5, 999, train_episodes=cfg.data.train_episodes, validation_episodes=cfg.data.validation_episodes
    )
    assert list(train) == snapshot["train"]
    assert list(val) == snapshot["validation"] == [7, 14, 21, 28, 35, 42]
    monkeypatch.setattr(
        data_loader.lerobot_dataset, "LeRobotDatasetMetadata", lambda *_: SimpleNamespace(total_episodes=51)
    )
    base = config.DataConfig(repo_id="unused", rtc_split="train", rtc_train_episodes=train, rtc_validation_episodes=val)
    assert data_loader.resolve_rtc_episodes(base).episodes == train
    assert data_loader.resolve_rtc_episodes(dataclasses.replace(base, rtc_split="validation")).episodes == val
    # Existing freeze strategy remains selected: no trainable VLM adapters.
    assert cfg.model.paligemma_variant == "gemma_2b"
    assert cfg.model.action_expert_variant == "gemma_300m_lora"


@pytest.mark.parametrize(
    ("train", "val", "match"),
    [
        ((0, 1), (1, 2), "overlap"),
        ((0, 0), (1, 2), "Duplicate"),
        ((0,), (1,), "cover"),
        ((0, 1), (3,), "range"),
        ((0, 1), None, "both"),
        ((0, 1), (True,), "integer"),
    ],
)
def test_invalid_explicit_partition_is_rejected(train, val, match):
    with pytest.raises(ValueError, match=match):
        config.rtc_episode_split(3, 0.1, 42, train_episodes=train, validation_episodes=val)
