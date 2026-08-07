"""Work units and the in-memory lease queue for parallel batch execution.

A **work unit** is one simulation attempt: ``(run_id, task_id, trial, seed)``.
Units are in-memory only and never persisted — the checkpoint (results.json)
is the single source of truth, and the queue is recomputed from it on resume.
See docs/designs/parallel-runner.md.

The queue is transport-agnostic: the local batch loop consumes it directly,
and the HTTP controller serves it to worker processes. Policy lives at lease
time — per-provider caps, a global in-flight cap — which is what lets one
controller interleave many runs.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from enum import Enum
from typing import Callable, Optional, Protocol

from pydantic import BaseModel

# One requeue after a failed attempt. Sim-level retries already happen inside
# the attempt (run_with_retry); this cap bounds whole-unit retries after a
# worker death or an infrastructure_error result.
DEFAULT_UNIT_ATTEMPTS = 2

# Leases are kept alive by heartbeats (a dedicated worker thread beats every
# ~30s), so the TTL only needs to outlive a heartbeat gap, not a simulation.
# 300s = 10 missed beats: on a loaded host running many voice sims, a few
# slow beats must not expire a live lease — the first field run showed a
# requeued-while-alive sim costs a full duplicate execution.
DEFAULT_LEASE_TTL_SECONDS = 300.0


def make_unit_id(run_id: str, task_id: str, trial: int) -> str:
    return f"{run_id}/{task_id}/t{trial}"


def parse_provider_limits(spec: Optional[str]) -> Optional[dict[str, int]]:
    """Parse a CLI cap spec like "openai=40,gemini=20"."""
    if not spec:
        return None
    limits: dict[str, int] = {}
    for part in spec.split(","):
        name, sep, value = part.partition("=")
        name = name.strip()
        if not sep or not name or not value.strip().isdigit():
            raise ValueError(
                f"Invalid provider limit {part!r}; expected e.g. 'openai=40,gemini=20'"
            )
        limits[name] = int(value)
    return limits


class WorkUnit(BaseModel):
    """One simulation attempt. Fully determines the simulation: any worker
    executing it produces the same run (seeds are computed by the producer)."""

    unit_id: str
    run_id: str
    task_id: str
    trial: int
    seed: int
    provider: Optional[str] = None
    attempt: int = 0
    progress_str: str = ""


class FailOutcome(str, Enum):
    REQUEUED = "requeued"
    DEAD = "dead"
    STALE = "stale"


class TaskSource(Protocol):
    """What a consumer needs from a queue of work units.

    ``WorkQueue`` implements it in-process; ``ControllerClient`` implements it
    over HTTP for worker processes.
    """

    def lease(self, worker_id: str) -> Optional[WorkUnit]: ...

    def complete(self, unit_id: str, worker_id: Optional[str] = None) -> bool: ...

    def fail(
        self, unit_id: str, error: str = "", worker_id: Optional[str] = None
    ) -> FailOutcome: ...

    def heartbeat(self, unit_id: str, worker_id: Optional[str] = None) -> bool: ...


class _Lease:
    __slots__ = ("unit", "worker_id", "expires_at")

    def __init__(self, unit: WorkUnit, worker_id: str, expires_at: float):
        self.unit = unit
        self.worker_id = worker_id
        self.expires_at = expires_at


class WorkQueue:
    """Thread-safe in-memory queue with leases, TTL expiry, and lease-time caps.

    Every mutation happens under one lock; expired leases are reaped lazily on
    each public call, so no background thread is needed. ``clock`` is
    injectable for tests (monotonic seconds).
    """

    def __init__(
        self,
        units: list[WorkUnit],
        *,
        max_attempts: int = DEFAULT_UNIT_ATTEMPTS,
        lease_ttl_seconds: float = DEFAULT_LEASE_TTL_SECONDS,
        provider_limits: Optional[dict[str, int]] = None,
        global_limit: Optional[int] = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self._pending: deque[WorkUnit] = deque(units)
        self._leases: dict[str, _Lease] = {}
        self._completed: set[str] = set()
        self.dead_units: list[tuple[WorkUnit, str]] = []
        self._max_attempts = max_attempts
        self._lease_ttl_seconds = lease_ttl_seconds
        self._provider_limits = provider_limits or {}
        self._global_limit = global_limit
        self._clock = clock
        self._lock = threading.Lock()

    # ---- Internal helpers (call with lock held) ----

    def _reap_expired(self) -> None:
        now = self._clock()
        for unit_id in [
            uid for uid, lease in self._leases.items() if lease.expires_at <= now
        ]:
            lease = self._leases.pop(unit_id)
            self._retire_or_requeue(lease.unit, "lease expired")

    def _retire_or_requeue(self, unit: WorkUnit, error: str) -> FailOutcome:
        if unit.attempt + 1 >= self._max_attempts:
            self.dead_units.append((unit, error))
            return FailOutcome.DEAD
        retry = unit.model_copy(update={"attempt": unit.attempt + 1})
        self._pending.append(retry)
        return FailOutcome.REQUEUED

    def _provider_inflight(self, provider: Optional[str]) -> int:
        return sum(
            1 for lease in self._leases.values() if lease.unit.provider == provider
        )

    def _leasable_index(self) -> Optional[int]:
        if self._global_limit is not None and len(self._leases) >= self._global_limit:
            return None
        for i, unit in enumerate(self._pending):
            limit = self._provider_limits.get(unit.provider)
            if limit is None or self._provider_inflight(unit.provider) < limit:
                return i
        return None

    def _lease_matches(self, unit_id: str, worker_id: Optional[str]) -> bool:
        lease = self._leases.get(unit_id)
        if lease is None:
            return False
        if worker_id is not None and lease.worker_id != worker_id:
            return False
        return True

    # ---- TaskSource interface ----

    def lease(self, worker_id: str) -> Optional[WorkUnit]:
        """The first pending unit whose provider has capacity; None if nothing
        is leasable right now (distinct from done — see all_resolved())."""
        with self._lock:
            self._reap_expired()
            index = self._leasable_index()
            if index is None:
                return None
            self._pending.rotate(-index)
            unit = self._pending.popleft()
            self._pending.rotate(index)
            self._leases[unit.unit_id] = _Lease(
                unit, worker_id, self._clock() + self._lease_ttl_seconds
            )
            return unit

    def complete(self, unit_id: str, worker_id: Optional[str] = None) -> bool:
        """Resolve a leased unit. False means stale: the lease expired (and the
        unit was requeued to another worker) or was already resolved."""
        with self._lock:
            self._reap_expired()
            if not self._lease_matches(unit_id, worker_id):
                return False
            del self._leases[unit_id]
            self._completed.add(unit_id)
            return True

    def fail(
        self, unit_id: str, error: str = "", worker_id: Optional[str] = None
    ) -> FailOutcome:
        with self._lock:
            self._reap_expired()
            if not self._lease_matches(unit_id, worker_id):
                return FailOutcome.STALE
            lease = self._leases.pop(unit_id)
            return self._retire_or_requeue(lease.unit, error)

    def heartbeat(self, unit_id: str, worker_id: Optional[str] = None) -> bool:
        with self._lock:
            self._reap_expired()
            if not self._lease_matches(unit_id, worker_id):
                return False
            self._leases[unit_id].expires_at = self._clock() + self._lease_ttl_seconds
            return True

    # ---- Introspection ----

    def all_resolved(self) -> bool:
        """Every unit either completed or dead — nothing pending, nothing leased."""
        with self._lock:
            self._reap_expired()
            return not self._pending and not self._leases

    def counts(self) -> dict[str, int]:
        with self._lock:
            self._reap_expired()
            return {
                "pending": len(self._pending),
                "leased": len(self._leases),
                "completed": len(self._completed),
                "dead": len(self.dead_units),
            }
