from server.session_broker import (
    SessionBroker,
    SessionCapacityError,
    SessionWorkerError,
)


class FakeClock:
    def __init__(self):
        self.value = 1000.0

    def now(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds

    def advance(self, seconds):
        self.value += seconds


def make_broker(max_sessions=2, session_lifetime_sec=60):
    registry = {}
    worker_status = {}
    spawned = []
    clock = FakeClock()

    def spawn_worker(session_id, user_id):
        spawned.append((session_id, user_id))
        worker_status[session_id] = {
            "session_id": session_id,
            "status": "ready",
            "url": f"https://{session_id}.modal.test",
        }

    broker = SessionBroker(
        registry,
        worker_status,
        spawn_worker,
        max_sessions=max_sessions,
        session_lifetime_sec=session_lifetime_sec,
        endpoint_wait_sec=1,
        poll_interval_sec=0.1,
        now=clock.now,
        sleep=clock.sleep,
    )
    return broker, registry, worker_status, spawned, clock


def test_same_user_reuses_single_worker_endpoint():
    broker, _, _, spawned, _ = make_broker()

    first = broker.allocate("user-1")
    second = broker.allocate("user-1")

    assert first["session_id"] == second["session_id"]
    assert first["url"] == second["url"]
    assert len(spawned) == 1


def test_new_user_is_rejected_at_capacity_without_spawning():
    broker, _, _, spawned, _ = make_broker(max_sessions=1)
    broker.allocate("user-1")

    try:
        broker.allocate("user-2")
    except SessionCapacityError as error:
        assert str(error) == "at_capacity"
    else:
        raise AssertionError("expected capacity error")

    assert [user_id for _, user_id in spawned] == ["user-1"]


def test_stopped_worker_frees_slot_for_new_user():
    broker, _, worker_status, spawned, _ = make_broker(max_sessions=1)
    first = broker.allocate("user-1")
    worker_status[first["session_id"]] = {"status": "stopped"}

    second = broker.allocate("user-2")

    assert second["session_id"] != first["session_id"]
    assert [user_id for _, user_id in spawned] == ["user-1", "user-2"]


def test_expired_worker_record_self_heals():
    broker, _, _, spawned, clock = make_broker(max_sessions=1, session_lifetime_sec=10)
    broker.allocate("user-1")
    clock.advance(11)

    second = broker.allocate("user-2")

    assert second["url"]
    assert [user_id for _, user_id in spawned] == ["user-1", "user-2"]


def test_worker_spawn_failure_frees_slot():
    registry: dict = {}
    worker_status: dict = {}
    clock = FakeClock()

    def spawn_worker(session_id, user_id):
        raise RuntimeError("modal spawn boom")

    broker = SessionBroker(
        registry,
        worker_status,
        spawn_worker,
        max_sessions=1,
        session_lifetime_sec=60,
        endpoint_wait_sec=1,
        poll_interval_sec=0.1,
        now=clock.now,
        sleep=clock.sleep,
    )

    try:
        broker.allocate("user-1")
    except SessionWorkerError:
        pass
    else:
        raise AssertionError("expected worker spawn failure")

    # A failed spawn must not leak the reserved slot.
    assert registry.get("user-1") is None
    assert worker_status == {}


def test_pending_worker_returns_without_url_but_holds_slot():
    registry: dict = {}
    worker_status: dict = {}
    clock = FakeClock()
    spawned = []

    def spawn_worker(session_id, user_id):
        spawned.append((session_id, user_id))
        # Cold start: reserved but no tunnel URL published yet.
        worker_status[session_id] = {"session_id": session_id, "status": "starting"}

    broker = SessionBroker(
        registry,
        worker_status,
        spawn_worker,
        max_sessions=1,
        session_lifetime_sec=60,
        endpoint_wait_sec=1,
        poll_interval_sec=0.5,
        now=clock.now,
        sleep=clock.sleep,
    )

    first = broker.allocate("user-1")
    assert first["url"] is None
    assert first["status"] == "starting"

    # Same user polling again reuses the pending session (no second spawn).
    second = broker.allocate("user-1")
    assert second["session_id"] == first["session_id"]
    assert len(spawned) == 1

    # A pending (URL-less) worker still consumes the single capacity slot.
    try:
        broker.allocate("user-2")
    except SessionCapacityError:
        pass
    else:
        raise AssertionError("pending worker should hold the slot")
