"""Absolute-command queue indexed by control ticks, for asynchronous Training RTC.

The controller owns tick advancement. Capture an observation inside begin_request's
callback so its start_index and command prefix are atomic with respect to pop().
Only one request is outstanding. Late results are rejected rather than spliced
against a prefix different from the one the model conditioned on.
"""

import copy
import dataclasses
import operator
import threading

import numpy as np


def _integer(value, name):
    if isinstance(value, bool | np.bool_):
        raise ValueError(f"{name} must be an integer")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc


@dataclasses.dataclass(frozen=True)
class MergeResult:
    accepted: bool
    reason: str
    elapsed: int
    start_index: int
    switch_index: int


class RTCQueue:
    def __init__(self, horizon: int, action_dim: int, max_delay: int):
        self.horizon = _integer(horizon, "horizon")
        self.action_dim = _integer(action_dim, "action_dim")
        self.max_delay = _integer(max_delay, "max_delay")
        if self.horizon < 1 or self.action_dim < 1 or not 0 <= self.max_delay < self.horizon:
            raise ValueError("Invalid queue dimensions or maximum delay")
        self._lock = threading.RLock()
        self._next_request_id = 0
        self.reset()

    def reset(self, *, start_index=0):
        with self._lock:
            start_index = _integer(start_index, "start_index")
            if start_index < 0:
                raise ValueError("start_index must be nonnegative")
            self._index = start_index
            self._actions = np.empty((0, self.action_dim), dtype=np.float32)
            self._pending = None
            # Do not reset request IDs: replies from a previous episode stay stale.

    @property
    def index(self):
        with self._lock:
            return self._index

    def snapshot(self):
        with self._lock:
            return self._index, self._actions.copy()

    def pop(self):
        """Call exactly once per command tick; apply returned absolute target without modification."""
        with self._lock:
            if not len(self._actions):
                raise RuntimeError("RTC queue exhausted: controller must explicitly hold/stop/rebootstrap")
            action = self._actions[0].copy()
            self._actions = self._actions[1:]
            self._index += 1
            return action

    def begin_request(self, capture_observation, *, delay):
        """delay is a committed prefix length, measured in command ticks (0..max_delay).

        An early reply preserves all committed commands until start+delay. A late
        reply (index > start+delay) is rejected. Choose delay from measured latency
        including camera capture, serialization, network, inference and scheduling.
        """
        with self._lock:
            delay = _integer(delay, "delay")
            if not 0 <= delay <= self.max_delay or delay > len(self._actions):
                raise ValueError("Delay outside training support or insufficient queued prefix")
            if self._pending is not None:
                raise RuntimeError("A request is already pending")
            obs = copy.deepcopy(capture_observation())
            if "rtc" in obs:
                raise ValueError("capture_observation must not supply RTC metadata")
            request = {
                "request_id": self._next_request_id,
                "start_index": self._index,
                "delay": delay,
                "prefix": self._actions[:delay].copy(),
            }
            self._next_request_id += 1
            self._pending = copy.deepcopy(request)
            obs["rtc"] = request
            return obs

    def cancel(self):
        """Invalidate pending work while retaining the old command queue."""
        with self._lock:
            self._pending = None

    def merge(self, response):
        with self._lock:
            metadata = response["rtc"]
            request_id = _integer(metadata["request_id"], "request_id")
            if self._pending is None or request_id != self._pending["request_id"]:
                return MergeResult(False, "stale_request", 0, self._index, self._index)
            pending = self._pending
            start, delay = pending["start_index"], pending["delay"]
            elapsed = self._index - start
            result_args = (elapsed, start, start + delay)
            if metadata.get("action_space") != "absolute":
                raise ValueError("Queue accepts absolute commands only")
            if (
                _integer(metadata["start_index"], "start_index") != start
                or _integer(metadata["delay"], "delay") != delay
            ):
                raise ValueError("Response indices do not match request")
            actions = np.asarray(response["actions"], dtype=np.float32)
            if actions.shape != (self.horizon, self.action_dim) or not np.isfinite(actions).all():
                raise ValueError("Invalid response action chunk")
            if not np.array_equal(actions[:delay], pending["prefix"]):
                raise ValueError("Response modified committed prefix")
            self._pending = None
            if elapsed > delay:
                return MergeResult(False, "missed_prefix_deadline", *result_args)
            remaining = delay - elapsed
            if not np.array_equal(self._actions[:remaining], pending["prefix"][elapsed:]):
                raise RuntimeError("Command queue no longer matches the conditioned prefix")
            # No double delay shift: actions are indexed from observation time.
            self._actions = np.concatenate([self._actions[:remaining], actions[delay:]], axis=0)
            return MergeResult(True, "accepted", *result_args)
