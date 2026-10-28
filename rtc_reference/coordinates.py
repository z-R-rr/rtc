"""Keep wire commands absolute; reuse the model's existing training transforms."""

import numpy as np


def encode_absolute_prefix(prefix, state, horizon, encode_chunk):
    """encode_chunk(absolute_chunk, current_state) -> normalized [H,D_model].

    Reuse the actual training delta/normalization/padding transforms in the
    callback. For the source Piper adapter delta[i] = command[i] - state NOW,
    including the gripper. Do not reuse a previous chunk's normalized prefix.
    """
    state = np.asarray(state, dtype=np.float32)
    prefix = np.asarray(prefix, dtype=np.float32)
    if (state.ndim != 1 or prefix.ndim != 2 or prefix.shape[1:] != state.shape
            or not 0 <= len(prefix) < horizon):
        raise ValueError("Expected state [D_raw], prefix [delay,D_raw], delay < H")
    if not np.isfinite(state).all() or not np.isfinite(prefix).all():
        raise ValueError("State and prefix must be finite")
    absolute = np.broadcast_to(state, (horizon, len(state))).copy()
    absolute[:len(prefix)] = prefix
    encoded = np.asarray(encode_chunk(absolute, state.copy()), dtype=np.float32)
    if encoded.ndim != 2 or encoded.shape[0] != horizon or not np.isfinite(encoded).all():
        raise ValueError("Training transforms must return finite [H,D_model] actions")
    return encoded


def restore_committed_prefix(decoded_absolute, prefix):
    """Restore exact original commands after decoding/unnormalizing model output."""
    actions = np.asarray(decoded_absolute, dtype=np.float32).copy()
    prefix = np.asarray(prefix, dtype=np.float32)
    if actions.ndim != 2 or prefix.ndim != 2 or prefix.shape[1:] != actions.shape[1:]:
        raise ValueError("Decoded chunk and prefix must share raw action dimensions")
    if len(prefix) >= len(actions) or not np.isfinite(actions).all() or not np.isfinite(prefix).all():
        raise ValueError("Invalid decoded actions or committed prefix")
    actions[:len(prefix)] = prefix
    return actions
