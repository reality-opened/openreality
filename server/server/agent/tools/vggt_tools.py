"""VGGT-backed tool implementations for spatial agent runtime."""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
from PIL import Image

from vggt_slam.object_detector import ObjectDetector


def _normalize_queries(raw: list[str], max_count: int = 16) -> list[str]:
    out: list[str] = []
    for item in raw:
        q = str(item).strip().lower()
        if not q:
            continue
        if len(q) > 120:
            q = q[:120]
        if q not in out:
            out.append(q)
        if len(out) >= max_count:
            break
    return out


class VGGTTools:
    def __init__(self, streaming_slam, on_query_added=None):
        self.slam = streaming_slam
        self.on_query_added = on_query_added

    def get_scene_snapshot(self, args) -> dict[str, Any]:
        num_submaps = int(self.slam.solver.map.get_num_submaps())
        num_loops = int(self.slam.solver.graph.get_num_loops())
        with self.slam._detection_lock:
            detections = list(self.slam.accumulated_detections)
            active_queries = list(self.slam.active_queries)

        payload: dict[str, Any] = {
            "frame_id": int(self.slam.frame_count),
            "num_submaps": num_submaps,
            "num_loops": num_loops,
            "active_queries": active_queries,
            "scene_center": self.slam.latest_scene_center.tolist(),
        }
        if bool(args.include_detections):
            payload["detections"] = detections
            payload["detected_queries"] = sorted(
                {str(d.get("query", "")).strip().lower() for d in detections if d.get("query")}
            )
        return payload

    def search_objects(self, args) -> dict[str, Any]:
        queries = _normalize_queries(args.queries)
        if not queries:
            return {"queries": [], "detections": [], "count": 0}

        t0 = time.time()

        with self.slam._detection_lock:
            already_active = set(self.slam.active_queries)

        for query in queries:
            if query not in already_active:
                for _ in self.slam.add_query_progressive(query):
                    pass
                if self.on_query_added:
                    self.on_query_added(query)

        query_set = set(queries)
        with self.slam._detection_lock:
            detections = [d for d in self.slam.accumulated_detections if d.get("query") in query_set]
        detections = detections[: int(args.max_results)]

        return {
            "queries": queries,
            "count": len(detections),
            "detections": detections,
            "is_final": True,
            "latency_ms": int((time.time() - t0) * 1000),
        }

    def inspect_detection(self, args) -> dict[str, Any]:
        submap = self.slam.solver.map.get_submap(int(args.submap_id))
        if submap is None:
            return {
                "ok": False,
                "error": f"Submap {args.submap_id} not found",
            }

        frame_tensor = submap.get_frame_at_index(int(args.frame_idx))
        frame_np = (frame_tensor.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        frame_pil = Image.fromarray(frame_np)

        keyframe_image = ObjectDetector.image_to_base64(frame_np)
        best_mask = None
        best_score = -math.inf
        for mask_2d, _box, score in self.slam.object_detector.segment_all(frame_pil, str(args.query)):
            if score > best_score:
                best_score = float(score)
                best_mask = mask_2d

        mask_image = None
        bbox_3d = None
        if best_mask is not None:
            mask_image = ObjectDetector.mask_overlay_to_base64(frame_np, best_mask)
            bbox_3d = self.slam.object_detector.compute_3d_bbox(
                submap,
                int(args.frame_idx),
                best_mask,
                self.slam.solver.graph,
                self.slam.latest_scene_center,
            )

        return {
            "ok": True,
            "query": str(args.query).strip().lower(),
            "submap_id": int(args.submap_id),
            "frame_idx": int(args.frame_idx),
            "keyframe_image": keyframe_image,
            "mask_image": mask_image,
            "bbox_3d": bbox_3d,
            "sam_score": None if best_score == -math.inf else best_score,
        }

    def locate_object_3d(self, args) -> dict[str, Any]:
        query = str(args.query).strip().lower()
        with self.slam._detection_lock:
            candidates = [
                d for d in self.slam.accumulated_detections
                if str(d.get("query", "")).strip().lower() == query and d.get("bounding_box")
            ]

        if not candidates:
            return {"query": query, "found": False}

        best = max(candidates, key=lambda d: float(d.get("confidence", 0.0) or 0.0))
        bb = best.get("bounding_box", {})
        return {
            "query": query,
            "found": True,
            "center": bb.get("center"),
            "extent": bb.get("extent"),
            "rotation": bb.get("rotation"),
            "matched_submap": best.get("matched_submap"),
            "matched_frame": best.get("matched_frame"),
            "confidence": float(best.get("confidence", 0.0) or 0.0),
        }

    def infer_spatial_relations(self, args) -> dict[str, Any]:
        wanted = set(_normalize_queries(args.queries))
        with self.slam._detection_lock:
            detections = [d for d in self.slam.accumulated_detections if d.get("bounding_box")]

        best_by_query: dict[str, dict[str, Any]] = {}
        for det in detections:
            q = str(det.get("query", "")).strip().lower()
            if not q:
                continue
            if wanted and q not in wanted:
                continue
            cur = best_by_query.get(q)
            score = float(det.get("confidence", 0.0) or 0.0)
            if cur is None or score > float(cur.get("confidence", 0.0) or 0.0):
                best_by_query[q] = det

        keys = sorted(best_by_query.keys())
        relations: list[dict[str, Any]] = []
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                qa, qb = keys[i], keys[j]
                ca = np.array(best_by_query[qa]["bounding_box"]["center"], dtype=np.float32)
                cb = np.array(best_by_query[qb]["bounding_box"]["center"], dtype=np.float32)
                dist = float(np.linalg.norm(ca - cb))
                relations.append(
                    {
                        "a": qa,
                        "b": qb,
                        "distance": dist,
                        "relation": "near" if dist < 1.5 else ("medium" if dist < 3.0 else "far"),
                    }
                )

        relations.sort(key=lambda x: x["distance"])
        return {
            "objects": keys,
            "relations": relations[:20],
        }

    def add_detection_object(self, args) -> dict[str, Any]:
        """Add a single object to active detection queries and persist it in the viewer."""
        query = str(args.query).strip().lower()
        if not query:
            return {"success": False, "error": "empty query"}
        with self.slam._detection_lock:
            already_active = query in self.slam.active_queries
        if already_active:
            return {"success": True, "query": query, "status": "already_active"}
        if self.on_query_added:
            self.on_query_added(query)
        return {"success": True, "query": query, "status": "queued"}

    def remove_detection_object(self, args) -> dict[str, Any]:
        """Remove a single object from active detection queries and clear its cached detections."""
        query = str(args.query).strip().lower()
        if not query:
            return {"success": False, "error": "empty query"}
        self.slam.remove_query(query)
        return {"success": True, "query": query, "removed": True}

    def propose_next_scan_focus(self, args) -> dict[str, Any]:
        goal = str(args.goal).strip() if args.goal else ""
        max_queries = int(args.max_queries)

        with self.slam._detection_lock:
            active = [str(q).strip().lower() for q in self.slam.active_queries if str(q).strip()]
            detections = list(self.slam.accumulated_detections)

        found_queries = {
            str(d.get("query", "")).strip().lower()
            for d in detections
            if str(d.get("query", "")).strip()
        }

        missing = [q for q in active if q not in found_queries]
        suggestions: list[str] = []
        for q in missing:
            if q not in suggestions:
                suggestions.append(q)
            if len(suggestions) >= max_queries:
                break

        if not suggestions:
            low_conf = sorted(
                [
                    (
                        str(d.get("query", "")).strip().lower(),
                        float(d.get("confidence", 0.0) or 0.0),
                    )
                    for d in detections
                    if str(d.get("query", "")).strip()
                ],
                key=lambda x: x[1],
            )
            for q, _ in low_conf:
                if q and q not in suggestions:
                    suggestions.append(q)
                if len(suggestions) >= max_queries:
                    break

        return {
            "goal": goal or None,
            "suggested_queries": suggestions[:max_queries],
            "reason": "prioritize unresolved targets" if missing else "recheck low-confidence detections",
        }
