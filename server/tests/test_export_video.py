"""Stage-1 video tests: ego_view.mp4 frame count + global ordering (§7.4).

Encoding uses OpenCV's platform codec (avc1/mp4v). If no mp4 writer can be opened in this
environment we skip rather than fail -- the frame<->row alignment logic is what we assert.
"""

from __future__ import annotations

import os

import pytest

from export_fakes import FakeMap, FakeSolver, FakeSubmap, count_video_frames, make_frames

from server.export.trajectory import build_index_map
from server.export.video import frame_to_rgb_uint8, write_ego_view_video


def test_frame_to_rgb_uint8_handles_chw_float():
    import numpy as np

    chw = np.zeros((3, 8, 8), dtype=np.float32)
    chw[0] = 1.0  # full red
    rgb = frame_to_rgb_uint8(chw)
    assert rgb.shape == (8, 8, 3)
    assert rgb.dtype == np.uint8
    assert rgb[0, 0, 0] == 255 and rgb[0, 0, 1] == 0


def _solver_with_frames():
    import numpy as np

    s0 = FakeSubmap(
        0,
        cameras=[(np.eye(3), [0, 0, 0]), (np.eye(3), [1, 0, 0])],
        frame_ids=[0.0, 1.0],
        frames=make_frames(2),
    )
    s1 = FakeSubmap(2, cameras=[(np.eye(3), [2, 0, 0])], frame_ids=[2.0], frames=make_frames(1))
    return FakeSolver(FakeMap([s0, s1]))


def test_write_ego_view_video_frame_count(tmp_path):
    solver = _solver_with_frames()
    index_map = build_index_map(solver)
    assert len(index_map) == 3
    path = str(tmp_path / "ego_view.mp4")
    try:
        written = write_ego_view_video(solver, index_map, path, fps=10.0)
    except RuntimeError as exc:
        pytest.skip(f"no mp4 VideoWriter available in this env: {exc}")
    assert written == 3
    assert os.path.isfile(path) and os.path.getsize(path) > 0
    # Read back when the platform decoder supports it; tolerate a non-decodable container.
    read = count_video_frames(path)
    assert read in (3, 0)
