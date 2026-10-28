import numpy as np
from openpi_client.rtc_queue import RTCQueue
import pytest


def reply(request, actions):
    m = {k: request["rtc"][k] for k in ("request_id", "start_index", "delay")}
    m["action_space"] = "absolute"
    actions = np.asarray(actions, dtype=np.float32).reshape(6, 1)
    actions[: m["delay"]] = request["rtc"]["prefix"]
    return {"rtc": m, "actions": actions}


def bootstrap():
    q = RTCQueue(6, 1, 3)
    req = q.begin_request(dict, delay=0)
    assert q.merge(reply(req, np.arange(6))).accepted
    return q


@pytest.mark.parametrize("elapsed", [0, 1, 2])
def test_early_and_on_time_preserve_old_prefix_without_double_shift(elapsed):
    q = bootstrap()
    assert q.pop()[0] == 0
    req = q.begin_request(lambda: {"state": np.array([100.0])}, delay=2)
    assert req["rtc"]["start_index"] == 1
    np.testing.assert_array_equal(req["rtc"]["prefix"][:, 0], [1, 2])
    for i in range(elapsed):
        assert q.pop()[0] == i + 1
    result = q.merge(reply(req, [1, 2, 30, 40, 50, 60]))
    assert result.accepted
    assert result.switch_index == 3
    np.testing.assert_array_equal(q.snapshot()[1][:, 0], ([1, 2][elapsed:] + [30, 40, 50, 60]))


def test_late_and_stale_replies_do_not_overwrite_queue():
    q = bootstrap()
    req = q.begin_request(dict, delay=1)
    q.pop()
    q.pop()
    before = q.snapshot()[1]
    result = q.merge(reply(req, np.arange(6) + 100))
    assert not result.accepted
    assert result.reason == "missed_prefix_deadline"
    np.testing.assert_array_equal(q.snapshot()[1], before)
    assert q.merge(reply(req, np.arange(6))).reason == "stale_request"
    req = q.begin_request(dict, delay=1)
    q.reset()
    boot = q.begin_request(dict, delay=0)
    assert q.merge(reply(req, np.arange(6))).reason == "stale_request"
    assert q.merge(reply(boot, np.arange(6))).accepted


def test_prefix_mutation_and_insufficient_tail_rejected():
    q = bootstrap()
    with pytest.raises(ValueError, match="Delay outside"):
        q.begin_request(dict, delay=4)
    req = q.begin_request(dict, delay=3)
    response = reply(req, np.arange(6))
    response["actions"][0] = 99
    with pytest.raises(ValueError, match="committed prefix"):
        q.merge(response)
    q.cancel()
    for _ in range(5):
        q.pop()
    with pytest.raises(ValueError, match="Delay outside"):
        q.begin_request(dict, delay=2)


def test_official_execution_horizon_semantics():
    q = bootstrap()
    # After s=2 actions, old tail aligns with observation at index 2.
    np.testing.assert_array_equal([q.pop()[0], q.pop()[0]], [0, 1])
    req = q.begin_request(dict, delay=1)
    assert q.pop()[0] == 2  # one old action executes during inference
    assert q.merge(reply(req, [2, 31, 32, 33, 34, 35])).accepted
    assert q.pop()[0] == 31  # new[d:s], not new[0] and not new[2*d]
    req2 = q.begin_request(dict, delay=1)
    assert req2["rtc"]["start_index"] == 4
    assert req2["rtc"]["prefix"][0, 0] == 32
