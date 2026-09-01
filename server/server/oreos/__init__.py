"""OpenReality demo-surface blueprint (docs/demo-2026-07 — design/shell.md §4).

All end-of-July demo REST routes live in this package; exactly ONE line in
``server/app.py`` registers ``oreos_bp``. Handlers reach app state (the auth
helpers, the scene persistence) by importing ``server.app`` lazily INSIDE the
handler (the ``server/reconstruct_pilot.py:59`` pattern), so this package never
participates in ``server.app``'s import cycle and stays importable standalone —
tests register the blueprint on a bare Flask app with a stubbed ``server.app``.

Route map (BUILD-PLAN "Routes"):

  routes_docs    GET  /api/scenes/<scan_id>/demo/manifest              (live)
                 PUT  /api/scenes/<scan_id>/demo/doc                   (live, guarded writer)
  jobs           GET  /api/workspace/jobs/<job_id>                     (live)
  routes_ingest  POST /api/workspace/ingest/{video,splat}              (501 stubs — W1)
  routes_agent   /api/scenes/<id>/demo/agent/*                         (501 stubs — W2)
  routes_sam3d   POST /api/scenes/<id>/segment, .../objects/<uid>/*    (501 stubs — W3)

Auth: every route is owner-only. The app-level ``before_request``
(``require_modal_auth``) has already verified the bearer/session token for all
``/api/*`` paths before a handler runs, and share/project tokens are
endpoint-allowlisted in ``server/app.py`` so they can never reach a demo
endpoint. Handlers re-check by requiring ``_auth_user_id()`` to resolve (401
otherwise) — belt and braces, and it keeps the blueprint safe even if it were
ever mounted on an app without the hook.
"""

from flask import Blueprint

oreos_bp = Blueprint("demo", __name__)


def app_module():
    """The live ``server.app`` module, imported lazily at call time (the
    ``reconstruct_pilot.py:59`` pattern) so importing ``server.oreos`` never
    triggers the heavy ``server.app`` import chain, and ``server.app`` can in
    turn import this package at its own module level without a cycle. Tests
    stub ``sys.modules['server.app']`` and this indirection picks the stub up."""
    from server import app as app_module

    return app_module


def auth_user_id():
    """The authenticated owner's user id, or ``None``. See the module docstring
    for why a handler treats ``None`` as 401."""
    return app_module()._auth_user_id()


def scene_persistence():
    """The configured ``ModalScenePersistence`` (or ``None`` when the app runs
    without a durable store, e.g. a bare local dev server)."""
    return getattr(app_module(), "_scene_persistence", None)


# Route modules attach their rules to ``oreos_bp`` at import time. Imported at the
# bottom so they can ``from server.oreos import oreos_bp`` — the partially
# initialized package already carries the attribute (standard Flask blueprint
# package layout).
from server.oreos import (  # noqa: E402,F401  (registration side effects)
    jobs,
    routes_agent,
    routes_docs,
    routes_ingest,
    routes_sam3d,
)

# W4: nav routes -------------------------------------------------------------
#   routes_nav   POST /api/scenes/<scan_id>/nav/plan                    (live)
# (append-only block; imported after the W0 skeleton so merges stay friction-free.
# routes_nav additionally imports routes_docs' manifest lock/helpers — it must
# come after routes_docs above.)
from server.oreos import routes_nav  # noqa: E402,F401  (registration side effects)

# W5: planes — POST /api/scenes/<id>/planes (F5 surface candidates) +
# POST /api/scenes/<id>/objects/<uid>/floor_patch (F6 hole-fill).
from server.oreos import routes_planes  # noqa: E402,F401  (registration side effects)

# W6: export surface — GET /api/scenes/<id>/demo/export/manifest (separate import block
# so parallel workstream merges never conflict on the tuple above).
from server.oreos import routes_export  # noqa: E402,F401  (registration side effects)

# Prepared export: POST /api/scenes/<id>/demo/export/prepare + GET .../prepared — build
# the zip as a background job and download a finished file (the zip bytes themselves ride
# the existing derived GET route). Must come after routes_export: it reuses that module's
# source-key precedence helper.
from server.oreos import routes_export_job  # noqa: E402,F401  (registration side effects)

# LOD: GET/POST /api/scenes/<id>/lod — level-of-detail splat index + generation.
# The LOD binaries themselves ride the existing derived GET route.
from server.oreos import routes_lod  # noqa: E402,F401  (registration side effects)

# Imported-splat parity: synthetic views (POST/GET /api/demo/scene/<id>/synthetic_views)
# + ground frame (POST/GET /api/scenes/<id>/ground_frame). Own import block so the two
# imported-splat workstreams never conflict on the tuple above.
from server.oreos import routes_imported  # noqa: E402,F401  (registration side effects)

# Robot recordings (merged Oreos pipeline): POST /api/workspace/ingest/recording — .db
# upload → oreos_recording_job (recon on the separate oreos-recon app + QC gate) →
# persisted source="robot_recording" scene. Must come after routes_ingest (reuses its
# upload staging helpers).
from server.oreos import routes_recordings  # noqa: E402,F401  (registration side effects)

# LiDAR capture-session ingest (EXP-45 metric grounding): POST
# /api/workspace/ingest/capture/{init,chunk/<index>,finalize} — zipped iPhone
# CaptureSession upload → the standalone lidar-recon Modal app (VGGT-SLAM with the
# LiDAR metric anchor + metric gauge layer) → persisted source="recon_lidar" scene.
# Must come after routes_ingest (reuses its upload staging helpers).
from server.oreos import routes_capture  # noqa: E402,F401  (registration side effects)
