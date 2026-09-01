"""Lightweight in-memory scene index used by the spatial agent."""

from __future__ import annotations

import threading
import time
from typing import Any


def _norm_query(value: Any) -> str:
    return str(value or "").strip().lower()


class SceneIndex:
    """Session-local index over deduped detections for low-latency lookup."""

    def __init__(self, max_per_query: int = 32):
        self.max_per_query = max(1, int(max_per_query))
        self._lock = threading.Lock()
        self._records_by_query: dict[str, list[dict[str, Any]]] = {}
        self._records_by_key: dict[tuple[str, int, int], dict[str, Any]] = {}

    def clear(self) -> None:
        with self._lock:
            self._records_by_query.clear()
            self._records_by_key.clear()

    def ingest(self, detections: list[dict[str, Any]]) -> int:
        """Upsert detections and return number of updated records."""
        updated = 0
        now = time.time()
        with self._lock:
            for det in detections:
                query = _norm_query(det.get("query"))
                if not query:
                    continue
                submap = int(det.get("matched_submap", -1))
                frame = int(det.get("matched_frame", -1))
                key = (query, submap, frame)
                confidence = float(det.get("confidence", 0.0) or 0.0)

                rec = {
                    "query": query,
                    "confidence": confidence,
                    "matched_submap": submap,
                    "matched_frame": frame,
                    "bounding_box": det.get("bounding_box"),
                    "last_seen_ts": now,
                    "raw": det,
                }
                prev = self._records_by_key.get(key)
                if prev is None or confidence >= float(prev.get("confidence", 0.0) or 0.0):
                    self._records_by_key[key] = rec
                updated += 1

            rebuilt: dict[str, list[dict[str, Any]]] = {}
            for rec in self._records_by_key.values():
                q = rec["query"]
                rebuilt.setdefault(q, []).append(rec)

            for q, rows in rebuilt.items():
                rows.sort(
                    key=lambda x: (
                        float(x.get("confidence", 0.0) or 0.0),
                        float(x.get("last_seen_ts", 0.0) or 0.0),
                    ),
                    reverse=True,
                )
                rebuilt[q] = rows[: self.max_per_query]

            allowed_keys: set[tuple[str, int, int]] = set()
            for q, rows in rebuilt.items():
                for row in rows:
                    allowed_keys.add((q, int(row["matched_submap"]), int(row["matched_frame"])))
            self._records_by_key = {
                k: v
                for k, v in self._records_by_key.items()
                if k in allowed_keys
            }
            self._records_by_query = rebuilt

        return updated

    def search(self, query: str, max_results: int = 8) -> list[dict[str, Any]]:
        q = _norm_query(query)
        if not q:
            return []

        max_results = max(1, int(max_results))
        with self._lock:
            out: list[dict[str, Any]] = []
            seen_keys: set[tuple[str, int, int]] = set()

            exact = self._records_by_query.get(q, [])
            for rec in exact:
                key = (rec["query"], int(rec["matched_submap"]), int(rec["matched_frame"]))
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                out.append(self._format_match(rec, match_type="exact"))
                if len(out) >= max_results:
                    return out

            terms = [t for t in q.split() if t]
            for rec_q, rows in self._records_by_query.items():
                if rec_q == q:
                    continue
                if q in rec_q or rec_q in q or any(t in rec_q for t in terms):
                    for rec in rows:
                        key = (rec["query"], int(rec["matched_submap"]), int(rec["matched_frame"]))
                        if key in seen_keys:
                            continue
                        seen_keys.add(key)
                        out.append(self._format_match(rec, match_type="partial"))
                        if len(out) >= max_results:
                            return out

            return out

    def summary(self, max_queries: int = 12) -> dict[str, Any]:
        max_queries = max(1, int(max_queries))
        with self._lock:
            query_rows = sorted(
                (
                    {
                        "query": q,
                        "count": len(rows),
                        "top_confidence": float(rows[0].get("confidence", 0.0) or 0.0) if rows else 0.0,
                    }
                    for q, rows in self._records_by_query.items()
                ),
                key=lambda x: (x["top_confidence"], x["count"]),
                reverse=True,
            )
            return {
                "indexed_query_count": len(self._records_by_query),
                "indexed_detection_count": len(self._records_by_key),
                "top_queries": query_rows[:max_queries],
            }

    @staticmethod
    def _format_match(rec: dict[str, Any], match_type: str) -> dict[str, Any]:
        return {
            "query": rec.get("query"),
            "confidence": float(rec.get("confidence", 0.0) or 0.0),
            "matched_submap": int(rec.get("matched_submap", -1)),
            "matched_frame": int(rec.get("matched_frame", -1)),
            "bounding_box": rec.get("bounding_box"),
            "match_type": match_type,
        }
