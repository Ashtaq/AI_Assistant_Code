"""In-process response cache and per-organization daily budget for the top tier.

Both live in memory: they reset on restart and are per instance. That is fine for one container;
move them to Redis when running several replicas.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Generic, TypeVar

T = TypeVar("T")


class TTLCache(Generic[T]):
    def __init__(self, ttl_seconds: int, max_entries: int, clock: Callable[[], float] = time.monotonic):
        self.ttl = ttl_seconds
        self.max = max_entries
        self._clock = clock
        self._data: OrderedDict[str, tuple[float, T]] = OrderedDict()

    @property
    def enabled(self) -> bool:
        return self.ttl > 0 and self.max > 0

    def get(self, key: str) -> T | None:
        if not self.enabled:
            return None
        hit = self._data.get(key)
        if not hit:
            return None
        expires, value = hit
        if expires < self._clock():
            del self._data[key]
            return None
        self._data.move_to_end(key)
        return value

    def set(self, key: str, value: T) -> None:
        if not self.enabled:
            return
        self._data[key] = (self._clock() + self.ttl, value)
        self._data.move_to_end(key)
        while len(self._data) > self.max:
            self._data.popitem(last=False)


class DailyBudget:
    """Counts top-tier calls per organization per UTC day. limit=0 means unlimited."""

    def __init__(self, limit: int, today: Callable[[], str] | None = None):
        self.limit = limit
        self._today = today or (lambda: datetime.now(UTC).date().isoformat())
        self._day = ""
        self._used: dict[str, int] = {}

    def _roll(self) -> None:
        day = self._today()
        if day != self._day:
            self._day = day
            self._used = {}

    def available(self, org_id: str) -> bool:
        self._roll()
        return self.limit == 0 or self._used.get(org_id, 0) < self.limit

    def spend(self, org_id: str) -> None:
        self._roll()
        self._used[org_id] = self._used.get(org_id, 0) + 1
