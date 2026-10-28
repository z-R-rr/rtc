import numpy as np

from rtc_reference.coordinates import encode_absolute_prefix
from rtc_reference.coordinates import restore_committed_prefix


def test_prefix_uses_current_state_and_training_encoder_and_restores_exact_wire_values():
    state = np.asarray([10, 20], dtype=np.float32)
    prefix = np.asarray([[11, 19], [12, 18]], dtype=np.float32)
    seen = {}

    def training_delta_encoder(absolute, current_state):
        seen["state"] = current_state
        return absolute - current_state

    encoded = encode_absolute_prefix(prefix, state, 4, training_delta_encoder)
    np.testing.assert_array_equal(encoded[:2], [[1, -1], [2, -2]])
    np.testing.assert_array_equal(encoded[2:], 0)
    np.testing.assert_array_equal(seen["state"], state)

    decoded = restore_committed_prefix(np.full((4, 2), 99, dtype=np.float32), prefix)
    np.testing.assert_array_equal(decoded[:2], prefix)
    np.testing.assert_array_equal(decoded[2:], 99)
