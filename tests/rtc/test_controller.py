# ruff: noqa: SLF001
from concurrent.futures import Future

import numpy as np
from openpi_client.rtc_controller import AsyncRTCController


class Policy:
    def infer(self, obs):
        req = obs["rtc"]
        actions = np.arange(6, dtype=np.float32)[:, None] + req["start_index"] * 10
        actions[: req["delay"]] = req["prefix"]
        return {"actions": actions, "rtc": {**req, "action_space": "absolute"}}


class Executor:
    def submit(self, fn, request):
        self.future = Future()
        self.response = fn(request)
        return self.future

    def shutdown(self, **kwargs):
        pass


def test_controller_requests_at_observation_tick_and_preserves_committed_prefix():
    ctl = AsyncRTCController(Policy(), horizon=6, action_dim=1, max_delay=3, delay=1, execution_horizon=2)
    ctl._executor.shutdown()
    ctl._executor = Executor()

    def capture():
        return {"state": np.array([0.0])}

    try:
        ctl.bootstrap(capture)
        assert ctl.next_action(capture)[0] == 0
        assert ctl.next_action(capture)[0] == 1
        assert ctl.next_action(capture)[0] == 2  # observation/request tick 2, old prefix executes
        ctl._executor.future.set_result(ctl._executor.response)
        assert ctl.next_action(capture)[0] == 21  # new[delay], indexed from observation tick 2
    finally:
        ctl.close()
