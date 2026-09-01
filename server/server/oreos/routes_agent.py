"""Scene-agent routes (W2) — REST run + 1 s polling over the persisted-scene agent.

Transport (BUILD-PLAN decision 6, shell.md §6): ``POST …/annotate`` → 202 ``{run_id}``
(409 ``agent_run_active`` while the scene already has a live run) → the client polls
``GET …/runs/<run_id>/events?after=<seq>`` at ~1 Hz → ``{events, status, next}``. The
run thread appends to an in-memory ring and flushes the whole log to
``derived/demo/agent_runs/<run_id>/events.json`` (see ``runlog.py``), so every live
run is automatically a deterministic, $0 replay artifact.

Replay surfaces (REPLAY invariant — the stream ALWAYS begins with
``run_meta{replay: true, recorded_at, source_run_id}``; no off switch):
  * ``POST …/annotate {replay_of: <run_id>}`` — server-paced playback run (clamped
    original deltas / ``speed``); the unmodified polling client renders it live-like.
  * ``GET …/runs/<run_id>/events?replay=1[&speed=S]`` — the stored log in one
    response, each event annotated ``delay_ms`` for client-side pacing.

Also here (deliverable F7): ``POST /api/scenes/<scan_id>/measure`` — the deterministic
measurement route sharing ``demo/measure.py`` with the agent's ``measure.*`` tools.
(Kept in this module so W2 touches no W0-owned files; the blueprint route table in
``__init__``'s docstring is the map.)

Auth: every route owner-only (module docstring of ``server.oreos``).
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Optional

from flask import jsonify, request

from server.oreos import auth_user_id, oreos_bp, scene_persistence
from server.oreos import measure as measure_mod
from server.oreos import runlog
from server.oreos.runlog import REGISTRY, ActiveRunError

_ANNOTATE_MODES = ("full", "fact_only")
_MAX_CHAT_HISTORY = 6


def _gate(scan_id: str):
    """Shared owner+scene gate → ``(user_id, store, record, error_response)``."""
    user_id = auth_user_id()
    if not user_id:
        return None, None, None, (jsonify({"error": "invalid_token"}), 401)
    store = scene_persistence()
    if store is None:
        return None, None, None, (jsonify({"error": "not_found"}), 404)
    record = store.get_scene(user_id, scan_id)
    if not record:
        return None, None, None, (jsonify({"error": "not_found"}), 404)
    return user_id, store, record, None


def _busy_response(exc: ActiveRunError):
    return (
        jsonify({"error": "agent_run_active", "active_run_id": exc.active_run_id}),
        409,
    )


# ── run-row normalization (protocol demo.ts AgentRunListItem) ──
#
# The registry/manifest speak the server's internal vocabulary (``mode``,
# ``created_at``, ``event_count``); the contract in web/packages/protocol
# (``AgentRunListItem``) speaks ``kind``/``started_at``/``n_events``/``cost_usd``.
# Rows carry BOTH — the protocol keys are what clients read, the internal keys stay
# for existing readers (additive-only doctrine).


def _run_done_facts(doc: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Spend/outcome facts a finished run recorded in its ``run_done`` event.

    These are *recorded* values (never estimated): ``cost_usd`` is ``None`` unless the
    provider actually priced the calls, which is also exactly what a $0 replay reports.
    """
    out: dict[str, Any] = {"cost_usd": None, "llm_calls": None, "findings_emitted": None}
    if not doc:
        return out
    for event in reversed(doc.get("events") or []):
        if not isinstance(event, dict) or event.get("type") != "run_done":
            continue
        payload = event.get("payload") or {}
        if not isinstance(payload, dict):
            break
        cost = payload.get("cost_usd")
        out["cost_usd"] = float(cost) if isinstance(cost, (int, float)) else None
        calls = payload.get("llm_calls")
        out["llm_calls"] = int(calls) if isinstance(calls, (int, float)) else None
        found = payload.get("findings_emitted")
        out["findings_emitted"] = int(found) if isinstance(found, (int, float)) else None
        break
    return out


def _protocol_row(row: dict[str, Any]) -> dict[str, Any]:
    """Stamp the protocol aliases onto a run row (internal keys preserved)."""
    row.setdefault("kind", row.get("mode"))
    if row.get("kind") is None:
        row["kind"] = row.get("mode")
    row["started_at"] = row.get("created_at")
    row["n_events"] = row.get("event_count")
    row.setdefault("cost_usd", None)
    row.setdefault("error", None)
    return row


# ── annotate (F2) ──


@oreos_bp.route("/api/scenes/<scan_id>/demo/agent/annotate", methods=["POST"])
def agent_annotate(scan_id: str):
    """Start an annotation run — or, with ``replay_of``, a $0 replay of a stored one.

    Body: ``{mode?: "full"|"fact_only", replay_of?: <run_id>, speed?: number,
    attach_if_active?: bool}``.
    202 ``{run_id}`` (+``replay: true`` for replays) | 409 ``agent_run_active`` |
    404 ``unknown_run`` for a ``replay_of`` with no persisted log.

    ``attach_if_active`` (additive, opt-in) makes the start **idempotent for one
    operator**: a second click while this scene already has a live run returns 202
    ``{run_id: <the active run>, attached: true}`` — the caller ends up watching the
    run it asked for instead of a dead-end 409. Callers that do NOT opt in keep the
    honest 409 with ``active_run_id`` (genuine concurrency: another tab/operator must
    learn that someone else's run owns the scene, not silently join it). No new run is
    ever started while one is active either way — the one-active-run guarantee and its
    spend control are untouched.
    """
    user_id, store, record, err = _gate(scan_id)
    if err:
        return err
    body = request.get_json(silent=True) or {}
    if not isinstance(body, dict):
        return jsonify({"error": "bad_request", "detail": "expected a JSON object"}), 400

    from server.oreos import persisted_agent

    attach_if_active = bool(body.get("attach_if_active"))

    def _busy_or_attach(exc: ActiveRunError):
        if attach_if_active:
            return jsonify({"run_id": exc.active_run_id, "attached": True}), 202
        return _busy_response(exc)

    replay_of = body.get("replay_of")
    if replay_of:
        doc = runlog.load_run_doc(store, user_id, scan_id, str(replay_of))
        if doc is None:
            return (
                jsonify({"error": "unknown_run", "detail": f"no stored events for run {replay_of!r}"}),
                404,
            )
        try:
            speed = float(body.get("speed", 1.0) or 1.0)
        except (TypeError, ValueError):
            speed = 1.0
        try:
            run = persisted_agent.start_replay_run(store, user_id, scan_id, doc, speed=speed)
        except ActiveRunError as exc:
            return _busy_or_attach(exc)
        return (
            jsonify({"run_id": run.run_id, "replay": True, "source_run_id": str(replay_of)}),
            202,
        )

    mode = str(body.get("mode") or "full").strip() or "full"
    if mode not in _ANNOTATE_MODES:
        return (
            jsonify({"error": "invalid_mode", "detail": f"mode must be one of {list(_ANNOTATE_MODES)}"}),
            400,
        )
    try:
        run = persisted_agent.start_annotate_run(store, user_id, scan_id, record, mode=mode)
    except ActiveRunError as exc:
        return _busy_or_attach(exc)
    return jsonify({"run_id": run.run_id}), 202


# ── runs listing ──


@oreos_bp.route("/api/scenes/<scan_id>/demo/agent/runs", methods=["GET"])
def agent_runs(scan_id: str):
    """All known runs for this scene: in-process (live status) merged over the
    persisted manifest refs (survivors of a broker restart)."""
    user_id, store, _record, err = _gate(scan_id)
    if err:
        return err
    rows: dict[str, dict[str, Any]] = {}
    for ref in runlog.list_persisted_runs(store, user_id, scan_id):
        rid = str(ref.get("run_id"))
        rows[rid] = {
            "run_id": rid,
            "mode": ref.get("kind"),
            "status": "done",  # persisted-only default; live rows below override
            "created_at": ref.get("created_at"),
            "events_key": ref.get("events_key"),
            "replay": bool(ref.get("replay", False)),
            "persisted": True,
        }
        doc = runlog.load_run_doc(store, user_id, scan_id, rid)
        if doc:
            rows[rid]["status"] = (
                doc.get("status") if doc.get("status") in ("done", "error") else "done"
            )
            rows[rid]["event_count"] = len(doc.get("events") or [])
            rows[rid]["finished_at"] = doc.get("finished_at")
            rows[rid]["source_run_id"] = doc.get("source_run_id")
            # mode/created_at live in the doc too — the manifest ref can predate them.
            rows[rid]["mode"] = rows[rid].get("mode") or doc.get("mode")
            if rows[rid].get("created_at") is None:
                rows[rid]["created_at"] = doc.get("created_at")
            # A restart loses the registry's in-memory error string; the doc keeps it.
            rows[rid]["error"] = doc.get("error")
            rows[rid].update(_run_done_facts(doc))
    for summary in REGISTRY.list_for_scene(user_id, scan_id):
        rid = summary["run_id"]
        merged = rows.get(rid, {})
        recorded = {k: v for k, v in merged.items() if k in ("cost_usd", "llm_calls", "findings_emitted")}
        merged.update(summary)
        # summary() has no spend fields — keep the ones read off the persisted run_done.
        for key, value in recorded.items():
            if merged.get(key) is None:
                merged[key] = value
        merged["persisted"] = bool(merged.get("events_key")) or merged.get("persisted", False)
        rows[rid] = merged
    ordered = sorted(rows.values(), key=lambda r: float(r.get("created_at") or 0.0))
    active = REGISTRY.active_run(user_id, scan_id)
    return jsonify(
        {
            "runs": [_protocol_row(r) for r in ordered],
            "active_run_id": active.run_id if active else None,
        }
    )


# ── events poll ──


@oreos_bp.route("/api/scenes/<scan_id>/demo/agent/runs/<run_id>/events", methods=["GET"])
def agent_run_events(scan_id: str, run_id: str):
    """Poll a run's feed. ``?after=<seq>`` (default −1 = everything), ``?replay=1``
    to re-serve the persisted log as a replay stream (``delay_ms``-annotated, replay
    run_meta first), ``?speed=S`` to scale replay pacing."""
    user_id, store, _record, err = _gate(scan_id)
    if err:
        return err
    try:
        after = int(request.args.get("after", "-1"))
    except (TypeError, ValueError):
        after = -1
    replay_flag = str(request.args.get("replay", "")).strip() in ("1", "true", "yes")
    try:
        speed = float(request.args.get("speed", "1") or 1.0)
    except (TypeError, ValueError):
        speed = 1.0

    if replay_flag:
        doc = runlog.load_run_doc(store, user_id, scan_id, run_id)
        if doc is None:
            return jsonify({"error": "unknown_run"}), 404
        events, status, next_cursor = runlog.replay_view(doc, after=after, speed=speed)
        return jsonify(
            {
                "events": events,
                "status": status,
                "next": next_cursor,
                "replay": True,
                "error": doc.get("error"),
            }
        )

    run = REGISTRY.get_for_scene(user_id, scan_id, run_id)
    if run is not None:
        events, status, next_cursor = run.events_after(after)
        # ``error`` is additive and always present (null while healthy) so a client can
        # render the REASON a run failed instead of a bare "errored" status.
        return jsonify(
            {"events": events, "status": status, "next": next_cursor, "error": run.error}
        )

    # Not in-process (broker restarted, or a historical run): serve the stored log.
    doc = runlog.load_run_doc(store, user_id, scan_id, run_id)
    if doc is None:
        return jsonify({"error": "unknown_run"}), 404
    all_events = [e for e in (doc.get("events") or []) if isinstance(e, dict)]
    events = [e for e in all_events if int(e.get("seq", -1)) > after]
    status = doc.get("status") if doc.get("status") in ("done", "error") else "done"
    # next = highest stored seq (pass-back-verbatim cursor; see runlog.events_after).
    next_cursor = max((int(e.get("seq", -1)) for e in all_events), default=-1)
    return jsonify(
        {
            "events": events,
            "status": status,
            "next": next_cursor,
            "error": doc.get("error"),
        }
    )


# ── chat ──


def _chat_history_from_doc(doc: Optional[dict[str, Any]]) -> list[dict[str, str]]:
    """Prior chat turns of a stored run → LLM history (user/assistant strings)."""
    if not doc:
        return []
    history: list[dict[str, str]] = []
    for event in doc.get("events") or []:
        if not isinstance(event, dict) or event.get("type") != "agent_thought":
            continue
        payload = event.get("payload") or {}
        content = str(payload.get("content") or "").strip()
        if not content:
            continue
        if payload.get("author") == "user":
            history.append({"role": "user", "content": content})
        elif payload.get("type") == "chat_response":
            history.append({"role": "assistant", "content": content})
    return history[-_MAX_CHAT_HISTORY:]


@oreos_bp.route("/api/scenes/<scan_id>/demo/agent/chat", methods=["POST"])
def agent_chat(scan_id: str):
    """One chat turn, streamed as its own run on the scene's feed.

    Body ``{message, run_id?}``. ``run_id`` continues a FINISHED run's conversation
    (its prior turns become LLM history; ``run_meta.continues_run`` links them for the
    timeline). While any run is still active → 409 (turns are serialized per scene).
    """
    user_id, store, record, err = _gate(scan_id)
    if err:
        return err
    body = request.get_json(silent=True) or {}
    message = str(body.get("message") or "").strip() if isinstance(body, dict) else ""
    if not message:
        return jsonify({"error": "bad_request", "detail": "expected {message}"}), 400
    if len(message) > 2000:
        return jsonify({"error": "bad_request", "detail": "message too long (>2000 chars)"}), 400

    history: list[dict[str, str]] = []
    continues: Optional[str] = None
    prior_run_id = body.get("run_id") if isinstance(body, dict) else None
    if prior_run_id:
        prior = REGISTRY.get_for_scene(user_id, scan_id, str(prior_run_id))
        if prior is not None and prior.status == runlog.STATUS_RUNNING:
            return _busy_response(ActiveRunError(prior.run_id))
        doc = runlog.load_run_doc(store, user_id, scan_id, str(prior_run_id))
        if doc is None and prior is not None:
            doc = prior.to_doc()
        history = _chat_history_from_doc(doc)
        continues = str(prior_run_id)

    from server.oreos import persisted_agent

    try:
        run = persisted_agent.start_chat_run(
            store, user_id, scan_id, record, message, history=history
        )
    except ActiveRunError as exc:
        return _busy_response(exc)
    return jsonify({"run_id": run.run_id, "continues_run": continues}), 202


# ── pilot (F4 agent-drive) ──


@oreos_bp.route("/api/scenes/<scan_id>/demo/agent/pilot", methods=["POST"])
def agent_pilot(scan_id: str):
    """Agent-drive instruction → bounded tool loop over
    {list_scene_objects, plan_path, measure_distance} (plan_path is the W4 seam —
    an honest tool error until the planner lands). 202 ``{run_id}``."""
    user_id, store, record, err = _gate(scan_id)
    if err:
        return err
    body = request.get_json(silent=True) or {}
    instruction = (
        str(body.get("instruction") or "").strip() if isinstance(body, dict) else ""
    )
    if not instruction:
        return jsonify({"error": "bad_request", "detail": "expected {instruction}"}), 400
    if len(instruction) > 2000:
        return (
            jsonify({"error": "bad_request", "detail": "instruction too long (>2000 chars)"}),
            400,
        )

    from server.oreos import persisted_agent

    try:
        run = persisted_agent.start_pilot_run(store, user_id, scan_id, record, instruction)
    except ActiveRunError as exc:
        return _busy_response(exc)
    return jsonify({"run_id": run.run_id}), 202


# ── measurement (F7) ──


@oreos_bp.route("/api/scenes/<scan_id>/measure", methods=["POST"])
def scene_measure(scan_id: str):
    """Deterministic measurement over world points (features.md F7 server side).

    Body ``{kind: "distance"|"angle", points_world: [[x,y,z],…]}`` (2 points for
    distance, 3 for angle — vertex is the 2nd). Response = protocol ``MeasureResponse``
    ``{kind, value, units, scale_factor?, scale_source, degrees?}`` (+ additive
    ``value_slam``/``metric_state``). Metres ONLY when the scene's ``derived_latest``
    anchor pointer provides a scale; a measurement audit doc is persisted under
    ``derived/demo/measurements/<stamp>/measure.json`` (non-destructive).
    """
    user_id, store, record, err = _gate(scan_id)
    if err:
        return err
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"error": "bad_request", "detail": "expected {kind, points_world}"}), 400

    depth_provenance = None
    try:
        raw = store.get_derived_artifact(
            user_id, scan_id, "derived/demo/metric/provenance.json"
        )
        if raw:
            parsed = json.loads(raw.decode("utf-8"))
            depth_provenance = parsed if isinstance(parsed, dict) else None
    except Exception:
        depth_provenance = None

    metric = measure_mod.resolve_metric(record, depth_provenance)
    try:
        result = measure_mod.measure(body.get("kind"), body.get("points_world"), metric)
    except measure_mod.MeasureError as exc:
        status = 400 if exc.code in ("invalid_kind", "invalid_points") else 422
        return jsonify({"error": exc.code, "detail": str(exc)}), status

    # Audit trail (best-effort, never blocks the response): the measurement doc.
    try:
        stamp = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
        audit = {
            "scan_id": scan_id,
            "created_at": time.time(),
            "request": {"kind": body.get("kind"), "points_world": body.get("points_world")},
            "result": result,
            "parent_artifact": (record.get("derived_latest") or {}).get("source_key")
            or record.get("splat_key")
            or record.get("points_key"),
        }
        store.save_derived_artifact(
            user_id,
            scan_id,
            f"demo/measurements/{stamp}/measure.json",
            json.dumps(audit, ensure_ascii=False, indent=2).encode("utf-8"),
        )
    except Exception as exc:
        print(f"[demo.measure] audit persist failed for {scan_id}: {exc}")

    return jsonify(result)
