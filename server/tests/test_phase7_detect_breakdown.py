import time

from conftest import load_streaming_slam_module
from test_phase6_recompute_gating import make_detection_slam


BREAKDOWN_KEYS = {"sam_ms", "bbox_ms", "recompute_ms", "dedup_ms"}


def test_reconcile_detection_state_returns_breakdown_for_empty_queries(monkeypatch):
    streaming_slam_mod = load_streaming_slam_module(monkeypatch)
    slam = make_detection_slam(streaming_slam_mod)

    breakdown = slam._reconcile_detection_state([])

    assert set(breakdown) == BREAKDOWN_KEYS
    assert breakdown["sam_ms"] == 0.0
    assert breakdown["bbox_ms"] == 0.0
    assert breakdown["recompute_ms"] == 0.0
    assert isinstance(breakdown["dedup_ms"], float)
    assert breakdown["dedup_ms"] >= 0.0


def test_reconcile_detection_state_returns_non_negative_breakdown(monkeypatch):
    streaming_slam_mod = load_streaming_slam_module(monkeypatch)
    slam = make_detection_slam(streaming_slam_mod)
    slam._run_clip_sam_on_submap = lambda submap, queries: [(submap.get_id(), 0, queries[0])]
    slam._compute_bboxes_for_keys = lambda keys: time.sleep(0.001)
    slam._recompute_all_bboxes = lambda: time.sleep(0.001)

    t0 = time.perf_counter()
    breakdown = slam._reconcile_detection_state(["chair"], recompute_all_bboxes=True)
    detect_time_ms = (time.perf_counter() - t0) * 1000

    assert set(breakdown) == BREAKDOWN_KEYS
    assert all(isinstance(value, float) for value in breakdown.values())
    assert all(value >= 0.0 for value in breakdown.values())
    assert sum(breakdown.values()) <= detect_time_ms + 50.0
    assert breakdown["bbox_ms"] > 0.0
    assert breakdown["recompute_ms"] > 0.0


def test_process_submap_latency_log_includes_detect_breakdown(tmp_path, capsys, monkeypatch):
    streaming_slam_mod = load_streaming_slam_module(monkeypatch)
    slam = make_detection_slam(streaming_slam_mod)
    image_path = tmp_path / "frame_000001.jpg"
    image_path.write_bytes(b"jpeg")
    slam.image_names_subset = [str(image_path)]
    slam.image_io_times_subset = [0.001]
    slam._run_clip_sam_on_submap = lambda submap, queries: []

    slam.process_submap()

    out = capsys.readouterr().out
    assert "[latency] submap=" in out
    assert "sam=" in out
    assert "bbox=" in out
    assert "recompute=" in out
    assert "dedup=" in out


def test_detect_after_submap_update_keeps_recompute_zero_without_loop_closure(monkeypatch):
    streaming_slam_mod = load_streaming_slam_module(monkeypatch)
    slam = make_detection_slam(streaming_slam_mod)
    slam._run_clip_sam_on_submap = lambda submap, queries: []
    slam._recompute_all_bboxes = lambda: (_ for _ in ()).throw(AssertionError("unexpected recompute"))

    breakdown = slam._detect_after_submap_update(loop_closure_detected=False)

    assert set(breakdown) == BREAKDOWN_KEYS
    assert breakdown["recompute_ms"] == 0.0
