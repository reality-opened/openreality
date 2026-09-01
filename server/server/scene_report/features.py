"""Derive structured 3D facts from a VGGT-SLAM reconstruction.

Operates on duck-typed objects so it stays testable without GPU/model weights:
``solver`` needs ``.graph`` and ``.map``; ``.map`` needs ``ordered_submaps_by_key()``;
each submap needs ``get_lc_status()``, ``get_id()``, ``get_points_in_world_frame(graph)``
and ``get_all_poses_world(graph)``.

Coordinate frame: geometry is read in the SLAM *world* frame (via the world-frame
accessors). Detection bounding boxes are stored recentered around
``latest_scene_center`` (see ``StreamingSLAM._compute_bboxes_for_keys``), so we
re-anchor their centers back to world by adding ``scene_center``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from server.scene_report.schemas import (
    EvidenceRef,
    ObjectDescription,
    ObjectInstance,
    SceneFacts,
    SceneMetrics,
    SpatialRelation,
)


class SceneFeatureExtractor:
    def __init__(
        self,
        near_threshold: float = 1.5,
        medium_threshold: float = 3.0,
        max_relation_objects: int = 24,
        max_relations: int = 40,
        max_points_for_metrics: int = 200_000,
        percentile: float = 2.0,
    ):
        self.near_threshold = float(near_threshold)
        self.medium_threshold = float(medium_threshold)
        self.max_relation_objects = int(max_relation_objects)
        self.max_relations = int(max_relations)
        self.max_points_for_metrics = int(max_points_for_metrics)
        self.percentile = float(percentile)

    def extract(self, solver, detections: list[dict[str, Any]], scene_center) -> SceneFacts:
        center = self._as_vec3(scene_center)
        metrics = self._compute_metrics(solver)
        objects = self._build_objects(detections or [], center)
        counts: dict[str, int] = {}
        for obj in objects:
            counts[obj.query] = counts.get(obj.query, 0) + 1
        relations = self._build_relations(objects)
        return SceneFacts(
            metrics=metrics,
            objects=objects,
            object_counts=counts,
            relations=relations,
            coverage_estimate=self._coverage_estimate(metrics),
            scene_center=center.tolist(),
        )

    # -- geometry -----------------------------------------------------

    def _compute_metrics(self, solver) -> SceneMetrics:
        graph = getattr(solver, "graph", None)
        smap = getattr(solver, "map", None)
        if smap is None:
            return SceneMetrics()

        pts_chunks: list[np.ndarray] = []
        cam_positions: list[np.ndarray] = []
        num_submaps = 0
        num_keyframes = 0

        try:
            submaps = list(smap.ordered_submaps_by_key())
        except Exception:
            submaps = []

        for submap in submaps:
            try:
                if submap.get_lc_status():
                    continue
            except Exception:
                pass
            num_submaps += 1
            try:
                p = submap.get_points_in_world_frame(graph)
                if p is not None:
                    p = np.asarray(p, dtype=np.float64).reshape(-1, 3)
                    if p.size:
                        pts_chunks.append(p)
            except Exception:
                pass
            try:
                poses = np.asarray(submap.get_all_poses_world(graph), dtype=np.float64)
                if poses.ndim == 3 and poses.shape[0] > 0:
                    num_keyframes += int(poses.shape[0])
                    cam_positions.append(poses[:, :3, 3])
            except Exception:
                pass

        metrics = SceneMetrics(num_submaps=num_submaps, num_keyframes=num_keyframes)

        if pts_chunks:
            pts = np.concatenate(pts_chunks, axis=0)
            pts = pts[np.isfinite(pts).all(axis=1)]
            if pts.shape[0] > self.max_points_for_metrics:
                idx = np.random.default_rng(0).choice(
                    pts.shape[0], self.max_points_for_metrics, replace=False
                )
                pts = pts[idx]
            if pts.shape[0] > 0:
                lo = np.percentile(pts, self.percentile, axis=0)
                hi = np.percentile(pts, 100.0 - self.percentile, axis=0)
                dims = np.abs(hi - lo)
                metrics.bbox_min = lo.tolist()
                metrics.bbox_max = hi.tolist()
                metrics.dimensions = sorted((float(d) for d in dims), reverse=True)
                metrics.point_count = int(pts.shape[0])

        if cam_positions:
            traj = np.concatenate(cam_positions, axis=0)
            if traj.shape[0] >= 2:
                seg = np.linalg.norm(np.diff(traj, axis=0), axis=1)
                seg = seg[np.isfinite(seg)]
                metrics.trajectory_length = float(np.sum(seg))

        return metrics

    # -- objects ------------------------------------------------------

    def _build_objects(self, detections, center) -> list[ObjectInstance]:
        objects: list[ObjectInstance] = []
        for det in detections:
            if not isinstance(det, dict):
                continue
            bbox = det.get("bounding_box")
            if not isinstance(bbox, dict):
                continue
            raw_center = bbox.get("center")
            if raw_center is None:
                continue
            try:
                c = np.asarray(raw_center, dtype=np.float64).reshape(3)
            except Exception:
                continue
            query = str(det.get("query", "")).strip().lower()
            if not query:
                continue
            try:
                extent = np.asarray(bbox.get("extent") or [], dtype=np.float64).reshape(-1).tolist()
            except Exception:
                extent = []
            objects.append(
                ObjectInstance(
                    query=query,
                    center=(c + center).tolist(),  # re-anchor to world frame
                    extent=extent,
                    confidence=float(det.get("confidence", 0.0) or 0.0),
                    evidence=[
                        EvidenceRef(
                            submap_id=int(det.get("matched_submap", -1)),
                            frame_idx=int(det.get("matched_frame", -1)),
                            # additive: carry the real per-frame SAM detection geometry when
                            # the streaming pipeline persisted it (guarded; None otherwise).
                            box_2d=self._as_box2d(det.get("box_2d")),
                            mask_rle=det.get("mask_rle")
                            if isinstance(det.get("mask_rle"), dict)
                            else None,
                        )
                    ],
                    description=self._as_description(det.get("description")),
                )
            )
        return objects

    @staticmethod
    def _as_box2d(value) -> list[float] | None:
        """Coerce a detection's optional 2D pixel box to ``[x0,y0,x1,y1]`` floats, or ``None``.
        Best-effort: anything that is not a length-4 numeric sequence yields ``None``."""
        if value is None:
            return None
        try:
            box = [float(v) for v in value]
        except (TypeError, ValueError):
            return None
        return box if len(box) == 4 else None

    @staticmethod
    def _as_description(value) -> ObjectDescription | None:
        """Coerce a detection's optional fine-grained ``description`` (dict or model) into
        an ``ObjectDescription`` so it persists into the scene record. Best-effort."""
        if value is None:
            return None
        if isinstance(value, ObjectDescription):
            return value
        if isinstance(value, dict):
            try:
                return ObjectDescription.model_validate(value)
            except Exception:
                return None
        return None

    def _build_relations(self, objects) -> list[SpatialRelation]:
        ranked = sorted(objects, key=lambda o: o.confidence, reverse=True)[
            : self.max_relation_objects
        ]
        relations: list[SpatialRelation] = []
        for i in range(len(ranked)):
            for j in range(i + 1, len(ranked)):
                a, b = ranked[i], ranked[j]
                if a.query == b.query:
                    continue
                ca = np.asarray(a.center, dtype=np.float64)
                cb = np.asarray(b.center, dtype=np.float64)
                dist = float(np.linalg.norm(ca - cb))
                relations.append(
                    SpatialRelation(a=a.query, b=b.query, distance=dist, relation=self._qualify(dist))
                )
        relations.sort(key=lambda r: r.distance)
        return relations[: self.max_relations]

    def _qualify(self, dist: float) -> str:
        if dist < self.near_threshold:
            return "near"
        if dist < self.medium_threshold:
            return "medium"
        return "far"

    @staticmethod
    def _coverage_estimate(metrics: SceneMetrics) -> float:
        # Coarse v1 heuristic; replaced by real coverage mapping in a later phase.
        return float(min(1.0, metrics.num_submaps * 0.08))

    @staticmethod
    def _as_vec3(value) -> np.ndarray:
        try:
            v = np.asarray(value, dtype=np.float64).reshape(-1)
            if v.shape[0] >= 3:
                return v[:3]
        except Exception:
            pass
        return np.zeros(3, dtype=np.float64)
