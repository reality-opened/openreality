"""Shared duck-typed fakes for the dataset-export tests (no GPU / model weights).

Mirrors the ``tests/test_scene_features.py`` fake style. Builds *real* camera projection
matrices so the production ``decompose_camera`` round-trips, exercising the true geometry
path in ``server/export/trajectory.py`` without torch/VGGT.

``vggt_slam/slam_utils.py`` imports ``matplotlib`` at module top (used only by
``overlay_masks``, which we never call). In environments where matplotlib is broken (e.g. a
numpy-2 ABI mismatch) bare ``import matplotlib`` fails, so we stub it -- but only when it is
genuinely unimportable, so a healthy env keeps the real module.

⚠️ LEAK HAZARD: the stub is installed into ``sys.modules`` at *import* time and is never
removed, so it is visible to **every** test module collected after this one -- there is no
fixture teardown and no monkeypatch scoping. Any production module that does
``import matplotlib`` and then touches an attribute will see this stub, not the real
library. That is exactly how a missing matplotlib turned into an ``AttributeError`` inside
``server/oreos/recordings/report.py`` (``matplotlib.use("Agg")``) and took CI's whole collection down.
Keep the stub's surface = the attributes the suite actually touches (below), and prefer
lazy/guarded matplotlib imports in production code over relying on this.
"""

from __future__ import annotations

import sys
import types

try:  # keep the real matplotlib when it works; stub only a broken one.
    import matplotlib  # noqa: F401
except Exception:  # pragma: no cover - environment dependent
    _stub = types.ModuleType("matplotlib")
    # Minimal surface: `use` is the backend selector every headless caller invokes
    # first (report.py, slam_utils consumers). No-op is correct -- nothing that
    # reaches this stub actually renders.
    _stub.use = lambda *_a, **_kw: None
    # `colormaps` is touched by vggt_slam.slam_utils.overlay_masks, which the fakes
    # never call; present only so an accidental attribute probe fails loudly at use
    # time rather than at import time.
    _stub.colormaps = None
    _stub.__all__ = ["use", "colormaps"]
    sys.modules["matplotlib"] = _stub

import numpy as np
from scipy.spatial.transform import Rotation

DEFAULT_K = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])


def make_projection_4x4(K, R_cw, C) -> np.ndarray:
    """Camera projection 4×4 from intrinsics ``K``, cam->world rotation ``R_cw`` and world
    camera center ``C`` -- the inverse of what ``decompose_camera`` recovers (t=C, R=R_cw)."""
    R_wc = np.asarray(R_cw, dtype=np.float64).T
    C = np.asarray(C, dtype=np.float64).reshape(3)
    P3 = np.asarray(K, dtype=np.float64) @ np.hstack([R_wc, (-R_wc @ C).reshape(3, 1)])
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :] = P3
    return pose


def rot(rx=0.0, ry=0.0, rz=0.0) -> np.ndarray:
    return Rotation.from_euler("xyz", [rx, ry, rz]).as_matrix()


class FakeSubmap:
    def __init__(
        self,
        sid,
        cameras,
        frame_ids=None,
        frames=None,
        points=None,
        colors=None,
        lc=False,
        K=None,
    ):
        """``cameras``: list of ``(R_cw (3,3), C (3,))`` tuples (or precomputed 4×4 proj mats)."""
        self._id = sid
        self._lc = lc
        self._K = DEFAULT_K if K is None else np.asarray(K, dtype=np.float64)
        self._proj = []
        for cam in cameras:
            arr = np.asarray(cam, dtype=np.float64) if not isinstance(cam, tuple) else None
            if arr is not None and arr.shape == (4, 4):
                self._proj.append(arr)
            else:
                R_cw, C = cam
                self._proj.append(make_projection_4x4(self._K, R_cw, C))
        n = len(self._proj)
        self._frame_ids = (
            list(frame_ids) if frame_ids is not None else [float(i) for i in range(n)]
        )
        self._frames = None if frames is None else np.asarray(frames, dtype=np.float32)
        self._points = None if points is None else np.asarray(points, dtype=np.float64)
        self._colors = None if colors is None else np.asarray(colors)

    def get_id(self):
        return self._id

    def get_lc_status(self):
        return self._lc

    def get_frame_ids(self):
        return list(self._frame_ids)

    def get_all_poses_world(self, graph, give_camera_mat=False):
        arr = np.stack(self._proj, axis=0)
        if give_camera_mat:
            return arr
        from vggt_slam.slam_utils import decompose_camera

        out = []
        for P in arr:
            _K, R, t, _s = decompose_camera(P)
            pose = np.eye(4)
            pose[:3, :3] = R
            pose[:3, 3] = t
            out.append(pose)
        return np.stack(out, axis=0)

    def get_frame_at_index(self, i):
        if self._frames is None:
            raise ValueError("fake submap has no frames")
        return self._frames[i]

    def get_points_in_world_frame(self, graph):
        return self._points

    def get_points_colors(self):
        return self._colors

    def get_points_list_in_world_frame(self, graph):
        pts = self._points if self._points is not None else np.zeros((0, 3))
        masks = [np.ones(len(pts), dtype=bool) for _ in self._proj]
        return [pts for _ in self._proj], list(self._frame_ids), masks


class FakeMap:
    def __init__(self, submaps):
        self._submaps = {s.get_id(): s for s in submaps}

    def ordered_submaps_by_key(self):
        for k in sorted(self._submaps):
            yield self._submaps[k]

    def get_submap(self, sid):
        return self._submaps.get(sid)

    def get_submaps(self):
        return list(self._submaps.values())


class FakeSolver:
    def __init__(self, smap):
        self.map = smap
        self.graph = object()


def make_frames(n, h=16, w=16) -> np.ndarray:
    """``(n, 3, h, w)`` float frames in [0,1] with a distinct per-frame tint (CHW, like Submap)."""
    frames = np.zeros((n, 3, h, w), dtype=np.float32)
    for i in range(n):
        frames[i, i % 3, :, :] = (i + 1) / float(n + 1)
    return frames


def count_video_frames(path) -> int:
    import cv2

    cap = cv2.VideoCapture(path)
    count = 0
    while True:
        ok, _frame = cap.read()
        if not ok:
            break
        count += 1
    cap.release()
    return count
