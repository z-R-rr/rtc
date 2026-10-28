import dataclasses

import numpy as np

from openpi import transforms


def _image_uint8(image):
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
        image = image.transpose(1, 2, 0)
    if np.issubdtype(image.dtype, np.floating) and image.max(initial=0) <= 1.0:
        image = image * 255.0
    return np.clip(image, 0, 255).astype(np.uint8)


@dataclasses.dataclass(frozen=True)
class PiperInputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        images = data["images"]
        for name in ("top_head", "hand_right"):
            if name not in images:
                raise ValueError(f"Missing Piper image: {name}")
        base = _image_uint8(images["top_head"])
        wrist = _image_uint8(images["hand_right"])
        out = {
            "image": {
                "base_0_rgb": base,
                "right_wrist_0_rgb": wrist,
                "left_wrist_0_rgb": np.zeros_like(base),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
                "left_wrist_0_rgb": np.False_,
            },
            "state": np.asarray(data["state"], dtype=np.float32),
        }
        if "actions" in data:
            out["actions"] = np.asarray(data["actions"], dtype=np.float32)
        if "action_is_pad" in data:
            out["action_is_pad"] = np.asarray(data["action_is_pad"], dtype=bool)
        if "prompt" in data:
            out["prompt"] = data["prompt"]
        return out


@dataclasses.dataclass(frozen=True)
class PiperOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])[..., :7]}
