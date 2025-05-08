# SPDX-FileCopyrightText: 2024 Ferenc Nandor Janky <ferenj@effective-range.com>
# SPDX-FileCopyrightText: 2024 Attila Gombos <attila.gombos@effective-range.com>
# SPDX-License-Identifier: MIT

import time

from context_logger import get_logger

log = get_logger('RateLimiter')


class IRateLimiter(object):

    def acquire(self, size: int = 1) -> bool:
        raise NotImplementedError()


class TokenBucketLimiter(IRateLimiter):

    def __init__(self, fill_rate: int, time_base: int = 1, burst_factor: int = 3) -> None:
        self._scale_factor = time_base
        self._fill_rate = fill_rate * time_base
        self._capacity = self._fill_rate * burst_factor
        self._tokens = self._capacity
        self._last_time = self._get_time()

    def acquire(self, size: int = 1) -> bool:
        current_time = self._get_time()
        elapsed_time = current_time - self._last_time
        self._last_time = current_time

        self._tokens += round(elapsed_time * self._fill_rate)
        self._tokens = min(self._tokens, self._capacity)

        required_size = size * self._scale_factor

        if required_size > self._tokens:
            log.debug("Rate limit exceeded", tokens=self._tokens, required=required_size)
            return False
        else:
            log.debug("Acquired", tokens=self._tokens, required=required_size)
            self._tokens -= required_size
            return True

    def _get_time(self) -> float:
        return time.monotonic() / self._scale_factor
