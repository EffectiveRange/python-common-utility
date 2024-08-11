# SPDX-FileCopyrightText: 2024 Ferenc Nandor Janky <ferenj@effective-range.com>
# SPDX-FileCopyrightText: 2024 Attila Gombos <attila.gombos@effective-range.com>
# SPDX-License-Identifier: MIT

from threading import Timer, Lock
from typing import Any, Iterable, Optional, Mapping


class IReusableTimer(object):

    def start(
        self, interval_sec: int, function: Any, *args: Optional[Iterable[Any]], **kwargs: Optional[Mapping[str, Any]]
    ) -> 'IReusableTimer':
        raise NotImplementedError()

    def restart(self) -> None:
        raise NotImplementedError()

    def cancel(self) -> None:
        raise NotImplementedError()

    def is_alive(self) -> bool:
        raise NotImplementedError()


class ReusableTimer(IReusableTimer):

    def __init__(self) -> None:
        self._timer_lock = Lock()
        self._timer: Optional[Timer] = None

    def start(
        self, interval_sec: int, function: Any, *args: Optional[Iterable[Any]], **kwargs: Optional[Mapping[str, Any]]
    ) -> IReusableTimer:
        with self._timer_lock:
            if self._timer:
                self._timer.cancel()
            self._timer = Timer(interval_sec, function, args, kwargs)
            self._timer.start()
        return self

    def restart(self) -> None:
        with self._timer_lock:
            if self._timer:
                self._timer.cancel()
                self._timer = Timer(self._timer.interval, self._timer.function, self._timer.args, self._timer.kwargs)
                self._timer.start()

    def cancel(self) -> None:
        with self._timer_lock:
            if self._timer:
                self._timer.cancel()
                self._timer = None

    def is_alive(self) -> bool:
        with self._timer_lock:
            return self._timer is not None and self._timer.is_alive()
