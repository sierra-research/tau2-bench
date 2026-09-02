"""Tests for the in-memory work queue (lease/complete/fail/heartbeat semantics)."""

from tau2.runner.work import (
    FailOutcome,
    WorkQueue,
    WorkUnit,
    make_unit_id,
)


def _make_units(
    n: int, provider: str | None = None, run_id: str = "run"
) -> list[WorkUnit]:
    return [
        WorkUnit(
            unit_id=make_unit_id(run_id, f"task_{i}", 0),
            run_id=run_id,
            task_id=f"task_{i}",
            trial=0,
            seed=42,
            provider=provider,
        )
        for i in range(n)
    ]


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float):
        self.now += seconds


class TestLeaseAndComplete:
    def test_lease_is_fifo_and_complete_resolves(self):
        queue = WorkQueue(_make_units(2))
        first = queue.lease("w1")
        second = queue.lease("w1")
        assert first.task_id == "task_0"
        assert second.task_id == "task_1"
        assert queue.lease("w1") is None
        assert not queue.all_resolved()

        assert queue.complete(first.unit_id)
        assert queue.complete(second.unit_id)
        assert queue.all_resolved()

    def test_complete_unknown_or_duplicate_is_stale(self):
        queue = WorkQueue(_make_units(1))
        unit = queue.lease("w1")
        assert queue.complete(unit.unit_id)
        assert not queue.complete(unit.unit_id)
        assert not queue.complete("nonexistent")

    def test_counts_snapshot(self):
        queue = WorkQueue(_make_units(3))
        unit = queue.lease("w1")
        queue.complete(unit.unit_id)
        counts = queue.counts()
        assert counts == {"pending": 2, "leased": 0, "completed": 1, "dead": 0}


class TestProviderAndGlobalLimits:
    def test_provider_limit_gates_leasing(self):
        units = _make_units(2, provider="openai")
        queue = WorkQueue(units, provider_limits={"openai": 1})
        first = queue.lease("w1")
        assert first is not None
        assert queue.lease("w2") is None  # provider at capacity
        queue.complete(first.unit_id)
        assert queue.lease("w2") is not None  # capacity released

    def test_unlimited_provider_not_gated(self):
        units = _make_units(2, provider="gemini")
        queue = WorkQueue(units, provider_limits={"openai": 1})
        assert queue.lease("w1") is not None
        assert queue.lease("w1") is not None

    def test_capped_provider_skipped_in_favor_of_uncapped(self):
        capped = _make_units(2, provider="openai")
        free = _make_units(1, provider="gemini", run_id="other")
        queue = WorkQueue(capped + free, provider_limits={"openai": 1})
        assert queue.lease("w1").provider == "openai"
        # openai is at capacity, so the gemini unit is leased even though it
        # sits behind an openai unit in FIFO order.
        assert queue.lease("w1").provider == "gemini"
        assert queue.lease("w1") is None

    def test_global_limit(self):
        queue = WorkQueue(_make_units(3), global_limit=2)
        assert queue.lease("w1") is not None
        assert queue.lease("w1") is not None
        assert queue.lease("w1") is None


class TestFailureAndRetry:
    def test_fail_requeues_with_incremented_attempt(self):
        queue = WorkQueue(_make_units(1), max_attempts=2)
        unit = queue.lease("w1")
        assert unit.attempt == 0
        assert queue.fail(unit.unit_id, "boom") == FailOutcome.REQUEUED
        retried = queue.lease("w1")
        assert retried.task_id == unit.task_id
        assert retried.attempt == 1

    def test_fail_exhausts_attempts_to_dead(self):
        queue = WorkQueue(_make_units(1), max_attempts=2)
        unit = queue.lease("w1")
        queue.fail(unit.unit_id, "boom")
        retried = queue.lease("w1")
        assert queue.fail(retried.unit_id, "boom again") == FailOutcome.DEAD
        assert queue.lease("w1") is None
        assert queue.all_resolved()
        assert len(queue.dead_units) == 1
        dead_unit, error = queue.dead_units[0]
        assert dead_unit.task_id == "task_0"
        assert error == "boom again"

    def test_fail_unknown_is_stale(self):
        queue = WorkQueue(_make_units(1))
        assert queue.fail("nonexistent", "boom") == FailOutcome.STALE


class TestLeaseExpiry:
    def test_expired_lease_requeues(self):
        clock = FakeClock()
        queue = WorkQueue(
            _make_units(1), lease_ttl_seconds=10, max_attempts=3, clock=clock
        )
        unit = queue.lease("w1")
        assert queue.lease("w2") is None
        clock.advance(11)
        retried = queue.lease("w2")
        assert retried is not None
        assert retried.attempt == unit.attempt + 1

    def test_heartbeat_extends_lease(self):
        clock = FakeClock()
        queue = WorkQueue(_make_units(1), lease_ttl_seconds=10, clock=clock)
        unit = queue.lease("w1")
        clock.advance(8)
        assert queue.heartbeat(unit.unit_id)
        clock.advance(8)
        assert queue.lease("w2") is None  # still leased thanks to heartbeat

    def test_expiry_exhausts_attempts_to_dead(self):
        clock = FakeClock()
        queue = WorkQueue(
            _make_units(1), lease_ttl_seconds=10, max_attempts=1, clock=clock
        )
        queue.lease("w1")
        clock.advance(11)
        assert queue.lease("w2") is None
        assert queue.all_resolved()
        assert len(queue.dead_units) == 1

    def test_stale_complete_after_expiry_requeue_is_ignored(self):
        clock = FakeClock()
        queue = WorkQueue(
            _make_units(1), lease_ttl_seconds=10, max_attempts=3, clock=clock
        )
        unit = queue.lease("w1")
        clock.advance(11)
        retried = queue.lease("w2")
        # Original worker finally reports; its lease was reaped, and the
        # unit_id now identifies the requeued copy held by w2, so the
        # late complete must not resolve w2's live lease.
        assert not queue.complete(unit.unit_id, worker_id="w1")
        assert queue.complete(retried.unit_id, worker_id="w2")
        assert queue.all_resolved()

    def test_heartbeat_on_expired_lease_fails(self):
        clock = FakeClock()
        queue = WorkQueue(_make_units(1), lease_ttl_seconds=10, clock=clock)
        unit = queue.lease("w1")
        clock.advance(11)
        assert not queue.heartbeat(unit.unit_id)
