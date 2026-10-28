"""One background inference request with explicit control-tick queue ownership."""

from concurrent.futures import ThreadPoolExecutor
import dataclasses
import logging

from openpi_client.rtc_queue import RTCQueue, _integer

logger = logging.getLogger(__name__)


class AsyncRTCController:
    """Transport-agnostic controller; policy can be WebsocketClientPolicy.

    Call bootstrap while control is paused, then next_action exactly once per
    control tick. Apply its absolute command unchanged. capture_observation must
    return synchronized current state/images in the model's robot units.
    No robot I/O, sleeps, interpolation, clipping, or implicit holds occur here.
    """

    def __init__(self, policy, *, horizon, action_dim, max_delay, delay, execution_horizon):
        self.delay = _integer(delay, "delay")
        self.execution_horizon = _integer(execution_horizon, "execution_horizon")
        if not 1 <= self.delay <= max_delay:
            raise ValueError("Async delay must be in [1,max_delay]; d=0 is used only for synchronous bootstrap")
        if not self.delay <= self.execution_horizon <= horizon - self.delay:
            raise ValueError("Require delay <= execution_horizon <= horizon-delay")
        self.queue = RTCQueue(horizon, action_dim, max_delay)
        self.policy = policy
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="training-rtc")
        self._future = None
        self._last_request_index = None

    def bootstrap(self, capture_observation):
        if self._future is not None:
            raise RuntimeError("Cannot bootstrap with an outstanding inference")
        self.queue.reset()
        request = self.queue.begin_request(capture_observation, delay=0)
        try:
            response = self.policy.infer(request)
            result = self.queue.merge(response)
        except Exception:
            self.queue.cancel()
            raise
        if not result.accepted:
            raise RuntimeError(f"Bootstrap rejected: {result.reason}")
        self._last_request_index = self.queue.index

    def next_action(self, capture_observation):
        if self._last_request_index is None:
            raise RuntimeError("Call bootstrap before control starts")
        if self._future is not None and self._future.done():
            future, self._future = self._future, None
            try:
                result = self.queue.merge(future.result())
            except Exception:
                self.queue.cancel()
                raise
            logger.info("RTC merge %s", dataclasses.asdict(result))
        index = self.queue.index
        if self._future is None and index - self._last_request_index >= self.execution_horizon:
            request = self.queue.begin_request(capture_observation, delay=self.delay)
            self._last_request_index = index
            self._future = self._executor.submit(self.policy.infer, request)
        return self.queue.pop()

    def close(self):
        # Transport should define its own network timeout. shutdown does not stop a running request.
        self.queue.cancel()
        self._executor.shutdown(wait=False, cancel_futures=True)
