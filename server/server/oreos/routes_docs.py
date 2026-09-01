"""Demo doc routes: the manifest read + the guarded generic doc writer.

The demo surface keeps ALL of its mutations under the scene's
``derived/demo/...`` namespace (BUILD-PLAN "Derived-key namespace") via the
existing non-destructive ``save_derived_artifact`` path — originals are never
touched, and the existing derived GET route serves everything back with zero
new read plumbing.

``derived/demo/manifest.json`` is the shell's single index of what demo
artifacts exist for a scene. It is maintained ONLY by this module, under a
``threading.Lock`` read-modify-write (safe: the broker deploys with
``max_containers=1`` and the founder is the single operator — one process, many
threads).

Routes:

  GET /api/scenes/<scan_id>/demo/manifest
      The manifest, or the empty default when none has been written yet.

  PUT /api/scenes/<scan_id>/demo/doc   {"key_suffix": "...", "json": <doc>}
      Generic guarded JSON-doc writer so feature workstreams (W4/W5/W6) never
      need their own server merges. Writes ``derived/demo/<key_suffix>`` and
      indexes the key into the manifest. The guard REJECTS (422) anything that
      could land outside ``derived/demo/`` — traversal segments, absolute
      paths, empty/oversized/charset-violating segments — and the reserved
      ``manifest.json`` (writer-maintained only). Docs must be ``.json`` keys;
      binary artifacts (npz/png/ply) are written server-side by their owning
      routes, not through this writer.
"""

from __future__ import annotations

import json
import re
import threading
import time
from datetime import datetime, timezone
from typing import Any, Optional

from flask import jsonify, request

from server.oreos import auth_user_id, oreos_bp, scene_persistence

# One lock for every manifest read-modify-write in this process (see module
# docstring for why process-local locking is sufficient here).
_MANIFEST_LOCK = threading.Lock()

# The manifest's key relative to the scene's derived/ root, and its wire key.
_MANIFEST_RELATIVE_KEY = "demo/manifest.json"
MANIFEST_KEY = "derived/demo/manifest.json"

# Guard rules for PUT key_suffix values. Mirrors (strictly: is a subset of) the
# store's ``_safe_derived_segment`` charset so an accepted suffix round-trips
# through ``save_derived_artifact`` byte-identical — anything the store would
# have to rewrite is rejected here instead of silently mutated.
_SEGMENT_RE = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
_MAX_KEY_SUFFIX_LEN = 200
_MAX_DOC_BYTES = 5 * 1024 * 1024  # metadata-class docs only; the broker has 4 GB total

# first path segment of key_suffix → manifest collection it indexes into.
# Everything (known or not) also lands in the generic ``docs`` list, so the
# manifest is always a complete index even for future namespaces.
_COLLECTIONS = {
    "agent_runs": "agent_runs",
    "edits": "edits",
    "paths": "paths",
    "nav": "paths",  # derived/demo/nav/paths/<pid>.json — F4's path docs
    "measurements": "measurements",
    "objects": "variations",  # objects/<uid>/… completions + variants — F3/F6
}


def _empty_manifest() -> dict[str, Any]:
    """The empty-default manifest (design/shell.md §3 shape + additive docs[])."""
    return {
        "version": 1,
        "updated_at": None,
        "annotations_key": None,
        "agent_runs": [],
        "edits": [],
        "paths": [],
        "variations": [],
        "measurements": [],
        "docs": [],
    }


def _validate_key_suffix(key_suffix: Any) -> Optional[str]:
    """Return the validated suffix, or ``None`` when it must be rejected.

    Accepts only relative, slash-separated ``.json`` paths whose every segment
    is non-empty, charset-clean, and not a traversal special — so the final key
    ``demo/<suffix>`` can never resolve outside the scene's ``derived/demo/``
    dir, and never collides with the writer-owned ``manifest.json``."""
    if not isinstance(key_suffix, str):
        return None
    suffix = key_suffix
    if not suffix or len(suffix) > _MAX_KEY_SUFFIX_LEN:
        return None
    if suffix.startswith("/") or suffix.endswith("/") or "\\" in suffix:
        return None
    segments = suffix.split("/")
    for segment in segments:
        if not segment or segment in (".", "..") or not _SEGMENT_RE.match(segment):
            return None
    if not segments[-1].endswith(".json") or segments[-1] == ".json":
        return None
    if suffix == "manifest.json":  # reserved: maintained by this module only
        return None
    return suffix


def _read_manifest(store: Any, user_id: str, scan_id: str) -> dict[str, Any]:
    """Load the scene's manifest, else the empty default (also on unreadable /
    non-dict content — the manifest is an index, not a source of truth, so a
    corrupt blob degrades to 'nothing indexed yet' rather than a 500)."""
    raw = store.get_derived_artifact(user_id, scan_id, MANIFEST_KEY)
    if not raw:
        return _empty_manifest()
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        print(f"[demo.docs] unreadable manifest for {scan_id}: {exc}")
        return _empty_manifest()
    if not isinstance(parsed, dict):
        return _empty_manifest()
    # Merge over the default so older manifests always expose every collection.
    manifest = _empty_manifest()
    manifest.update(parsed)
    return manifest


def _agent_run_ref(key_suffix: str, derived_key: str, doc: Any) -> dict[str, Any]:
    """A rich ``manifest.agent_runs`` entry (protocol demo.ts DemoManifestRunRef)
    for a doc written through the generic PUT route — the same shape runlog's
    flusher indexes, so the manifest never mixes plain strings in from this
    writer. Fields the doc carries (W2's events-doc shape: ``mode``,
    ``created_at``, ``replay``) are threaded through; ``created_at`` falls back
    to now (epoch seconds, per protocol)."""
    segments = key_suffix.split("/")
    if len(segments) >= 3 and segments[-1] == "events.json":
        run_id = segments[-2]  # agent_runs/<run_id>/events.json
    else:
        run_id = segments[-1][: -len(".json")]
    ref: dict[str, Any] = {"run_id": run_id, "events_key": derived_key}
    if isinstance(doc, dict):
        kind = doc.get("mode") or doc.get("kind")
        if kind:
            ref["kind"] = str(kind)
        if doc.get("created_at") is not None:
            ref["created_at"] = doc["created_at"]
        if "replay" in doc:
            ref["replay"] = bool(doc.get("replay"))
    ref.setdefault("created_at", time.time())
    return ref


def _index_into_manifest(
    manifest: dict[str, Any], key_suffix: str, derived_key: str, doc: Any = None
) -> None:
    """Record ``derived_key`` in the manifest: the matching typed collection
    (first path segment, see ``_COLLECTIONS``; ``annotations/…`` sets
    ``annotations_key``) plus the complete generic ``docs`` list.
    ``agent_runs/…`` keys index as rich run refs (see ``_agent_run_ref``),
    deduped by run_id/events_key against refs the runlog flusher already wrote."""
    first = key_suffix.split("/", 1)[0]
    if first == "annotations":
        manifest["annotations_key"] = derived_key
    elif first == "agent_runs":
        ref = _agent_run_ref(key_suffix, derived_key, doc)
        entries = manifest.get("agent_runs")
        if not isinstance(entries, list):
            entries = []
        already = any(
            (
                isinstance(e, dict)
                and (e.get("run_id") == ref["run_id"] or e.get("events_key") == derived_key)
            )
            or e == derived_key  # legacy plain-string entry
            for e in entries
        )
        if not already:
            entries.append(ref)
        manifest["agent_runs"] = entries
    else:
        collection = _COLLECTIONS.get(first)
        if collection is not None:
            entries = manifest.get(collection)
            if not isinstance(entries, list):
                entries = []
            if derived_key not in entries:
                entries.append(derived_key)
            manifest[collection] = entries
    docs = manifest.get("docs")
    if not isinstance(docs, list):
        docs = []
    if derived_key not in docs:
        docs.append(derived_key)
    manifest["docs"] = docs
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")


@oreos_bp.route("/api/scenes/<scan_id>/demo/manifest", methods=["GET"])
def get_demo_manifest(scan_id: str):
    """The scene's demo manifest (empty default when nothing was written yet)."""
    user_id = auth_user_id()
    if not user_id:
        return jsonify({"error": "invalid_token"}), 401
    store = scene_persistence()
    if store is None or not store.get_scene(user_id, scan_id):
        return jsonify({"error": "not_found"}), 404
    with _MANIFEST_LOCK:
        manifest = _read_manifest(store, user_id, scan_id)
    return jsonify(manifest)


@oreos_bp.route("/api/scenes/<scan_id>/demo/doc", methods=["PUT"])
def put_demo_doc(scan_id: str):
    """Guarded generic writer: persist one JSON doc under ``derived/demo/`` and
    index it into the manifest. See the module docstring for the guard rules."""
    user_id = auth_user_id()
    if not user_id:
        return jsonify({"error": "invalid_token"}), 401
    store = scene_persistence()
    if store is None or not store.get_scene(user_id, scan_id):
        return jsonify({"error": "not_found"}), 404

    body = request.get_json(silent=True)
    if not isinstance(body, dict) or "json" not in body:
        return jsonify({"error": "bad_request", "detail": "expected {key_suffix, json}"}), 400
    suffix = _validate_key_suffix(body.get("key_suffix"))
    if suffix is None:
        return (
            jsonify(
                {
                    "error": "invalid_key",
                    "detail": (
                        "key_suffix must be a relative .json path inside derived/demo/ "
                        "(charset [A-Za-z0-9_.-], no traversal, manifest.json reserved)"
                    ),
                }
            ),
            422,
        )
    try:
        doc_bytes = json.dumps(body["json"], ensure_ascii=False, indent=2).encode("utf-8")
    except (TypeError, ValueError):
        return jsonify({"error": "bad_request", "detail": "json member is not serializable"}), 400
    if len(doc_bytes) > _MAX_DOC_BYTES:
        return jsonify({"error": "doc_too_large", "max_bytes": _MAX_DOC_BYTES}), 413

    with _MANIFEST_LOCK:
        derived_key = store.save_derived_artifact(
            user_id, scan_id, f"demo/{suffix}", doc_bytes
        )
        manifest = _read_manifest(store, user_id, scan_id)
        _index_into_manifest(manifest, suffix, derived_key, doc=body["json"])
        store.save_derived_artifact(
            user_id,
            scan_id,
            _MANIFEST_RELATIVE_KEY,
            json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"),
        )
    return jsonify({"key": derived_key, "manifest": manifest})
