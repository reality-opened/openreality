"""
Flask + python-socketio ASGI streaming server for VGGT-SLAM 2.0.

Handles:
  - WebSocket frame streaming from phone/browser
  - SLAM update broadcasting to viewer clients
  - Object detection queries (CLIP + SAM3)
  - Session-scoped spatial agents with shared SLAM core
  - Video file testing mode
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hmac
import concurrent.futures
import contextlib
import io
import traceback
import json
import os
import queue
import secrets
import shutil
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from urllib.parse import quote
import base64
import argparse
import asyncio
import mimetypes
from http.cookies import SimpleCookie
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import cv2
import numpy as np
import torch
import json
import re

from flask import Flask, after_this_request, jsonify, request, send_file
from asgiref.wsgi import WsgiToAsgi
from flask_cors import CORS
import jwt
from PIL import Image
import requests
import socketio as socketio_pkg
from socketio.exceptions import ConnectionRefusedError as SocketIOConnectionRefused

from server.billing.usage import named_tally
from server.share_access import ShareAccessLog, access_id_for_token
from server.oreos.measure import resolve_metric as _resolve_metric_state
from server.llm import OpenRouterClient
from server.session_broker import SessionCapacityError, SessionWorkerError
from server.streaming_slam import StreamingSLAM
from server.scene_report import (
    ObjectEnricher,
    SceneFeatureExtractor,
    SceneReport,
    SceneReportBuilder,
    SerpApiLensProvider,
)
from server.scene_report import anchor as _anchor_impl
from server.scene_report.cloud_io import build_ply_bytes, downsample_cloud
from server.scene_report.keyframes import choose_frame_refs, encode_frames
from server.scene_report.splat_io import (
    DEFAULT_SCALE_CLAMP_PERCENTILE,
    clamp_splat_fields,
    read_splat_ply,
    serialize_splat_ply,
)
from server.export.record import EXPORT_SOURCE_ORIGINAL, load_record_from_store
# The export tree/zip builders moved to server.export.zip_builder so the background
# export job can reach them without importing this module. See _build_export_zip_file.
from server.export.zip_builder import build_export_zip_file
from vggt_slam.object_detector import ObjectDetector

# ------------------------------
# Clerk Auth Setup
# ------------------------------
AUTH_COOKIE_NAME = "slam_session"
AUTH_QUERY_PARAM = "auth_token"
APPROVED_TIER = "approved"
SCANS_CLAIM = "scansRemaining"
DEFAULT_SCANS = 2
SESSION_TIMEOUT_SEC = 10 * 60
AUTH_CLAIMS_ENV_KEY = "open_reality.auth_claims"
SHARE_TOKEN_ENV_KEY = "open_reality.share_token"  # raw share token of the current request (share-authed only)

# Durable broker session token (HS256, broker-signed) — bootstrapped from a verified
# Clerk JWT but with an INDEPENDENT expiry, so revisit/history reads + grounded Q&A keep
# authenticating after the short-lived Clerk JWT in the page's URL hash expires (the
# static-hash-token trap; see docs/gotchas.md). Identity only: it re-states the caller's
# `sub` + a snapshot of their scan-quota claims — it never grants scans. Set
# SESSION_TOKEN_SECRET (a Modal secret) to keep tokens valid across broker restarts /
# redeploys; otherwise a per-process key is used (fine for the single-instance broker,
# resets on redeploy).
SESSION_TOKEN_ALG = "HS256"
SESSION_TOKEN_ISSUER = "open-reality-broker"
SESSION_TOKEN_TTL_SEC = int(os.environ.get("SESSION_TOKEN_TTL_SEC", str(12 * 3600)))
SESSION_TOKEN_SECRET = (
    os.environ.get("SESSION_TOKEN_SECRET", "").strip() or secrets.token_urlsafe(48)
)
# Share/building links are external + long-lived (up to 30 days), so a random per-process
# secret means they silently break on restart / across replicas. Warn always when unset;
# hard-fail in production so a deploy can't ship ephemeral signing for external links.
if not os.environ.get("SESSION_TOKEN_SECRET", "").strip():
    _ephemeral_secret_msg = (
        "SESSION_TOKEN_SECRET is unset — signing session/share/building tokens with a "
        "random per-process key. External share + building links will break on restart "
        "or across replicas. Set SESSION_TOKEN_SECRET for any shared/multi-instance deploy."
    )
    if os.environ.get("OPEN_REALITY_ENV", "").strip().lower() in ("production", "prod"):
        raise RuntimeError(_ephemeral_secret_msg)
    print(f"[auth][warning] {_ephemeral_secret_msg}")

# Deploy-time escape hatch for throwaway test deployments (e.g. backbone/model
# validation apps): bypasses ALL token verification on both the HTTP and the
# Socket.IO paths, granting synthetic approved (unlimited-scan) claims to every
# caller. The deployment is then open to anyone holding the URL. Never set this
# in production — it is only injected when the deploying shell exports
# DANGEROUSLY_DISABLE_AUTH=1 (see _SESSION_RUNTIME_ENV in modal_streaming.py).
DANGEROUSLY_DISABLE_AUTH = os.environ.get("DANGEROUSLY_DISABLE_AUTH", "").strip() == "1"
if DANGEROUSLY_DISABLE_AUTH:
    print("=" * 72)
    print("!!  DANGEROUSLY_DISABLE_AUTH=1 — AUTH IS BYPASSED ON EVERY ROUTE.  !!")
    print("!!  Anyone with the URL has full, unlimited access. TEST ONLY.     !!")
    print("=" * 72)
    # With billing enforced this flag is not just an access hole, it is free
    # money: every caller gets synthetic unlimited claims. Refuse to boot rather
    # than serve, because the failure is otherwise completely silent.
    if os.environ.get("BILLING_ENFORCE", "").strip().lower() in {"1", "true", "on", "yes"}:
        raise RuntimeError(
            "DANGEROUSLY_DISABLE_AUTH=1 with BILLING_ENFORCE set — that grants "
            "every anonymous caller unlimited paid work. Unset one of them."
        )


def _bypass_claims() -> dict[str, Any]:
    """Synthetic claims handed out when DANGEROUSLY_DISABLE_AUTH is on.

    Shaped like a verified Clerk JWT: ``tier=approved`` sails through
    _require_scan_access (unlimited), and ``exp`` satisfies the cookie/QR-token
    routes that read an expiry off the claims.
    """
    return {
        "sub": "dev-bypass",
        "tier": APPROVED_TIER,
        "exp": int(time.time()) + SESSION_TOKEN_TTL_SEC,
    }


class AuthError(Exception):
    def __init__(self, message: str, status_code: int = 401):
        super().__init__(message)
        self.status_code = status_code


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} must be set")
    return value


def _parse_allowed_azp(raw: str) -> frozenset[str]:
    origins = frozenset(
        origin.strip().rstrip("/")
        for origin in raw.split(",")
        if origin.strip()
    )
    if not origins:
        raise RuntimeError("CLERK_JWT_AZP must contain at least one origin")
    return origins


# --- Auth mode --------------------------------------------------------------------
# OPENREALITY_AUTH selects the identity bootstrap:
#   clerk (default) — Clerk JWTs bootstrap identity (the hosted deploys).
#   local           — self-host mode, no Clerk account needed: the single static
#                     bearer in OPENREALITY_LOCAL_TOKEN is the bootstrap
#                     credential. Broker session tokens and ork_ API keys still
#                     work on top of it, so `openreality-mcp login --token` and
#                     the /api/keys mint/list/revoke routes behave unchanged.
#                     Single-user by construction (every caller is sub="local").
# Unlike DANGEROUSLY_DISABLE_AUTH this is NOT a bypass: unauthenticated requests
# still 401, and the token comparison is constant-time.
AUTH_MODE = os.environ.get("OPENREALITY_AUTH", "").strip().lower() or "clerk"
if AUTH_MODE not in ("clerk", "local"):
    raise RuntimeError(f"OPENREALITY_AUTH must be 'clerk' or 'local' (got {AUTH_MODE!r})")

LOCAL_AUTH_TOKEN = os.environ.get("OPENREALITY_LOCAL_TOKEN", "").strip()
LOCAL_AUTH_ISSUER = "open-reality-local"
if AUTH_MODE == "local":
    if len(LOCAL_AUTH_TOKEN) < 16:
        raise RuntimeError(
            "OPENREALITY_AUTH=local requires OPENREALITY_LOCAL_TOKEN to be set to a "
            "secret of at least 16 characters. Generate one with: "
            "python3 -c 'import secrets; print(secrets.token_urlsafe(32))'"
        )
    CLERK_JWKS_URL = None
    CLERK_JWT_ISSUER = None
    CLERK_JWT_ALLOWED_AZP: frozenset[str] = frozenset()
else:
    CLERK_JWKS_URL = _require_env("CLERK_JWKS_URL")
    CLERK_JWT_ISSUER = _require_env("CLERK_JWT_ISSUER")
    CLERK_JWT_ALLOWED_AZP = _parse_allowed_azp(_require_env("CLERK_JWT_AZP"))


def _load_signing_keys(jwks_url: str) -> dict[str, Any]:
    response = requests.get(jwks_url, timeout=10)
    response.raise_for_status()
    jwks = response.json()
    keys: dict[str, Any] = {}
    for jwk in jwks.get("keys", []):
        kid = jwk.get("kid")
        if not kid:
            continue
        keys[kid] = jwt.algorithms.RSAAlgorithm.from_jwk(json.dumps(jwk))
    if not keys:
        raise RuntimeError("CLERK_JWKS_URL did not return any signing keys")
    return keys


# Clerk rotates its JWKS signing keys periodically. Caching them once at import means a
# token signed with a freshly rotated `kid` is rejected as invalid_token on a warm
# container until it restarts. Cache with a TTL and refetch-once on an unknown kid
# (throttled so a flood of bogus kids can't hammer the JWKS endpoint). See fixes.md #3.
_JWKS_TTL_S = 3600.0
_JWKS_MIN_REFRESH_INTERVAL_S = 60.0
_jwks_lock = threading.Lock()
_jwks_state: dict[str, Any] = {"keys": {}, "fetched_at": 0.0, "last_attempt": 0.0}


def _refresh_jwks(force: bool = False) -> None:
    if CLERK_JWKS_URL is None:  # local auth mode — there are no Clerk keys to fetch
        return
    now = time.time()
    with _jwks_lock:
        if not force and now - _jwks_state["fetched_at"] < _JWKS_TTL_S:
            return
        if now - _jwks_state["last_attempt"] < _JWKS_MIN_REFRESH_INTERVAL_S:
            return
        _jwks_state["last_attempt"] = now
        try:
            _jwks_state["keys"] = _load_signing_keys(CLERK_JWKS_URL)
            _jwks_state["fetched_at"] = now
        except Exception as exc:
            # Keep serving the existing keys; a transient JWKS outage shouldn't
            # break every login on an otherwise-healthy warm container.
            print(f"[auth] JWKS refresh failed: {exc}")


def _get_signing_key(kid: str):
    key = _jwks_state["keys"].get(kid)
    if key is None:
        _refresh_jwks(force=True)  # unknown kid → Clerk may have rotated keys
        key = _jwks_state["keys"].get(kid)
    return key


# Eager initial load: fail fast at boot if Clerk is misconfigured (see docs/gotchas.md).
# Local auth mode has no Clerk to load — _refresh_jwks no-ops there.
if AUTH_MODE == "clerk":
    _refresh_jwks(force=True)
    if not _jwks_state["keys"]:
        raise RuntimeError("CLERK_JWKS_URL did not return any signing keys")


def _verify_clerk_token(token: str) -> dict[str, Any]:
    if AUTH_MODE != "clerk":  # no Clerk trust anchor exists in local mode
        raise AuthError("invalid_token")
    try:
        header = jwt.get_unverified_header(token)
    except jwt.PyJWTError as exc:
        raise AuthError("invalid_token") from exc

    kid = header.get("kid")
    alg = header.get("alg")
    if alg != "RS256" or not kid:
        raise AuthError("invalid_token")

    _refresh_jwks()  # TTL-gated no-op on the hot path; picks up rotations within the TTL
    signing_key = _get_signing_key(kid)
    if signing_key is None:
        raise AuthError("invalid_token")

    try:
        claims = jwt.decode(
            token,
            signing_key,
            algorithms=["RS256"],
            issuer=CLERK_JWT_ISSUER,
            options={
                "verify_aud": False,
                "require": ["exp", "iss", "sub"],
            },
        )
    except jwt.PyJWTError as exc:
        raise AuthError("invalid_token") from exc

    # Native mobile tokens carry no `azp` at all: Clerk derives it from the browser
    # Origin and refuses it as a custom template claim, so requiring it 401'd every
    # native call. Absent azp is accepted; a present azp must still be allow-listed,
    # so browser tokens behave exactly as before.
    azp = claims.get("azp")
    if azp is not None and (
        not isinstance(azp, str) or azp.rstrip("/") not in CLERK_JWT_ALLOWED_AZP
    ):
        raise AuthError("invalid_token")

    return claims


def _issue_session_token(claims: dict[str, Any]) -> tuple[str, int]:
    """Mint a durable broker session token from already-verified Clerk claims. Returns
    ``(token, expires_at_epoch)``. Carries the caller's identity (``sub``), with the
    broker's own ``exp`` decoupled from the Clerk JWT's. Bootstrap-only: the caller must
    already hold a valid Clerk JWT (enforced by ``require_modal_auth``) to obtain one.

    It carries IDENTITY, never a quota. It used to snapshot ``scansRemaining``,
    which made a 12-hour token into a 12-hour grant: a user at zero balance
    holding a fresh one could keep allocating GPU sessions all day. Balance is
    now read live from the ledger by ``_require_scan_access``, and the real gate
    is the hold taken at dispatch.

    ``tier`` stays: it changes about never, and it is the break-glass path if the
    ledger is unreachable."""
    now = int(time.time())
    expires_at = now + SESSION_TOKEN_TTL_SEC
    payload = {
        "iss": SESSION_TOKEN_ISSUER,
        "sub": str(claims.get("sub", "")).strip(),
        "iat": now,
        "exp": expires_at,
        "tier": claims.get("tier"),
    }
    token = jwt.encode(payload, SESSION_TOKEN_SECRET, algorithm=SESSION_TOKEN_ALG)
    return token, expires_at


def _verify_session_token(token: str) -> dict[str, Any]:
    """Verify a broker-issued session token (HS256), enforcing our issuer + ``exp``."""
    try:
        claims = jwt.decode(
            token,
            SESSION_TOKEN_SECRET,
            algorithms=[SESSION_TOKEN_ALG],
            issuer=SESSION_TOKEN_ISSUER,
            options={"require": ["exp", "iss", "sub"]},
        )
    except jwt.PyJWTError as exc:
        raise AuthError("invalid_token") from exc
    if not str(claims.get("sub", "")).strip():
        raise AuthError("invalid_token")
    return claims


# --- First-class API keys (Bearer ork_...) -----------------------------------------
# A durable, revocable secret for programmatic /api/* access (MCP clients, CI). Unlike
# the signed session JWT it is registry-backed (a modal.Dict shared by the broker and
# GPU workers, injected via configure_api_key_registry — mirrors the
# configure_scene_persistence pattern), so a key survives restarts and revocation is
# immediate. Only the key's SHA-256 lives at rest (see server/api_keys.py); the claims
# it yields snapshot tier + scan quota from mint time, exactly the fields
# _issue_session_token snapshots — identity only, never a scan grant.
API_KEY_PREFIX = "ork_"
API_KEY_ISSUER = "open-reality-api-key"

_api_key_registry: Any = None
_api_key_registry_warned = False


def configure_api_key_registry(registry: Any) -> None:
    """Inject the durable API-key registry (mirrors configure_scene_persistence)."""
    global _api_key_registry
    _api_key_registry = registry


def _verify_api_key(token: str) -> dict[str, Any]:
    """Verify a Bearer ``ork_`` API key against the injected registry, returning
    session-token-shaped claims plus an ``auth_kind`` marker (so the mint route can
    refuse key-mints-key) and the ``key_id`` (for auditability). Raises
    ``AuthError("invalid_token")`` for unknown/revoked keys — and when no registry is
    configured, so deploys that never inject one simply don't accept API keys."""
    global _api_key_registry_warned
    if _api_key_registry is None:
        if not _api_key_registry_warned:
            _api_key_registry_warned = True
            print(
                "[auth][warning] API key presented but no API-key registry is "
                "configured — rejecting. Inject one via configure_api_key_registry()."
            )
        raise AuthError("invalid_token")
    record = _api_key_registry.verify(token)
    if not record:
        raise AuthError("invalid_token")
    sub = str(record.get("user_id", "")).strip()
    if not sub:
        raise AuthError("invalid_token")
    return {
        "iss": API_KEY_ISSUER,
        "sub": sub,
        "tier": record.get("tier"),
        SCANS_CLAIM: record.get(SCANS_CLAIM),
        "auth_kind": "api_key",
        "key_id": record.get("key_id"),
    }


def _local_auth_claims() -> dict[str, Any]:
    """Claims for the self-host static bearer (OPENREALITY_AUTH=local): the single
    local operator, approved tier, and a session-token-length ``exp`` so the
    refresh/mint routes that read an expiry off the claims behave normally."""
    return {
        "iss": LOCAL_AUTH_ISSUER,
        "sub": "local",
        "tier": APPROVED_TIER,
        SCANS_CLAIM: None,
        "exp": int(time.time()) + SESSION_TOKEN_TTL_SEC,
    }


def _verify_any_token(token: str) -> dict[str, Any]:
    """Accept an ``ork_`` API key, a Clerk JWT (RS256, the bootstrap), or a durable
    broker session token (HS256). In local auth mode the static
    ``OPENREALITY_LOCAL_TOKEN`` matches first (constant-time). API keys are matched
    by their literal prefix BEFORE any JWT header parsing (they are not JWTs); the
    two JWT kinds are then routed by the ``alg`` header so they can't be confused."""
    if AUTH_MODE == "local" and hmac.compare_digest(token, LOCAL_AUTH_TOKEN):
        return _local_auth_claims()
    if token.startswith(API_KEY_PREFIX):
        return _verify_api_key(token)
    try:
        alg = jwt.get_unverified_header(token).get("alg")
    except jwt.PyJWTError as exc:
        raise AuthError("invalid_token") from exc
    if alg == SESSION_TOKEN_ALG:
        return _verify_session_token(token)
    return _verify_clerk_token(token)


# --- Read-only scene share tokens -------------------------------------------------
# A share token is a single-scan, read-only capability (no Clerk login) used to embed a
# finished scan in a third party's dashboard (the real-estate embed pilots). It is an
# HS256 JWT signed with the SAME SESSION_TOKEN_SECRET as the durable session token, but
# carries a distinct issuer + a ``scope``/``scan_id`` so it can never be confused with a
# session token (issuer verification) and is authorized only for the scene-read endpoints
# of its one scan. The owner ``sub`` is carried so the user-scoped store lookups resolve
# to the owner's scenes. See platform/contracts/embed-delivery.md.
SHARE_TOKEN_ISSUER = "open-reality-share"
SHARE_TOKEN_SCOPE = "scene:read"
SHARE_TOKEN_TTL_SEC_DEFAULT = int(os.environ.get("SHARE_TOKEN_TTL_SEC", str(30 * 24 * 3600)))
# Hard upper bound on any minted share/project token lifetime, so an owner (or a stolen
# owner session) can't mint an effectively-permanent external link. Default 90 days.
SHARE_TOKEN_TTL_MAX = int(os.environ.get("SHARE_TOKEN_TTL_MAX", str(90 * 24 * 3600)))
SHARE_QUERY_PARAM = "share_token"
# The scene-read endpoints (view-function names) a share token may reach. Anything else —
# the scenes list, DELETE, the mint route itself — is rejected for a share token.
SHARE_TOKEN_ALLOWED_ENDPOINTS = frozenset({
    "get_scene_route",
    "get_scene_points_route",
    "get_scene_splat_ply_route",
    "get_scene_cloud_ply_route",
    "get_scene_keyframe_route",
    "get_scene_object_asset_route",
    "scene_qa_route",
    # Fast-preview (LOD) delivery for the share embed. A real reconstruction's splat.ply
    # is 1-5 GB, so the embed loads the decimated `.spz` level the OS viewer uses instead.
    # GET only: the index read + the artifact bytes. POST /lod (the build, a billable
    # Modal job) stays owner-only — a scraped link must never be able to spawn compute.
    #
    # The LOD route lives on the oreos Blueprint, whose name is the persisted ``demo``
    # token (server/oreos/__init__.py), so Flask reports its endpoint as
    # ``demo.get_scene_lod_route`` — the bare function name never matches and 403s
    # (shipped that way on 2026-08-24; tests/test_oreos_lod.py pins the real name).
    "demo.get_scene_lod_route",
    "get_scene_derived_route",
})
# Derived-artifact keys (relative to ``derived/``) a share token may read through
# ``get_scene_derived_route``. Only the LOD previews: every other derived artifact
# (anchor/clamp splats, OS manifests, agent runs, the share-access log…) stays owner-only.
SHARE_TOKEN_DERIVED_KEY_PREFIXES = ("demo/lod/",)


def _share_token_derived_key_allowed(derived_key: Any) -> bool:
    """May a share token read this ``derived/<key>``? True only for the LOD preview tree.
    A leading ``derived/`` is tolerated (the route tolerates it too); any ``..`` or empty
    segment is refused here regardless of what the store's resolver would do with it."""
    rel = str(derived_key or "").strip("/")
    if rel.startswith("derived/"):
        rel = rel[len("derived/"):]
    if not rel or any(part in ("..", "") for part in rel.split("/")):
        return False
    return any(rel.startswith(prefix) for prefix in SHARE_TOKEN_DERIVED_KEY_PREFIXES)


def _issue_share_token(
    scan_id: str, owner_sub: str, ttl_seconds: Optional[int] = None,
    max_exp: Optional[int] = None,
) -> tuple[str, int]:
    """Mint a read-only share token for one scan. Returns ``(token, expires_at_epoch)``.
    ``ttl_seconds`` is clamped to ``[1, SHARE_TOKEN_TTL_MAX]``; ``max_exp`` (epoch, optional)
    further caps the absolute expiry — used so a manifest-minted scene token never outlives
    the project token that minted it."""
    now = int(time.time())
    ttl = SHARE_TOKEN_TTL_SEC_DEFAULT
    if ttl_seconds is not None:
        try:
            ttl = int(ttl_seconds)
        except (TypeError, ValueError):
            ttl = SHARE_TOKEN_TTL_SEC_DEFAULT
    ttl = max(1, min(ttl, SHARE_TOKEN_TTL_MAX))
    expires_at = now + ttl
    if max_exp is not None:
        expires_at = min(expires_at, int(max_exp))
        if expires_at <= now:
            expires_at = now + 1  # never mint an already-expired token
    payload = {
        "iss": SHARE_TOKEN_ISSUER,
        "scope": SHARE_TOKEN_SCOPE,
        "scan_id": str(scan_id),
        "sub": str(owner_sub).strip(),
        "iat": now,
        "exp": expires_at,
    }
    token = jwt.encode(payload, SESSION_TOKEN_SECRET, algorithm=SESSION_TOKEN_ALG)
    return token, expires_at


def _verify_share_token(token: str) -> dict[str, Any]:
    """Verify a share token (HS256 + our share issuer). Raises ``AuthError`` if it is not
    a valid share token — including a Clerk JWT, a session token (issuer mismatch), or an
    expired/tampered share token — so the caller can fall through to the normal auth path."""
    try:
        claims = jwt.decode(
            token,
            SESSION_TOKEN_SECRET,
            algorithms=[SESSION_TOKEN_ALG],
            issuer=SHARE_TOKEN_ISSUER,
            options={"require": ["exp", "iss", "sub", "scan_id"]},
        )
    except jwt.PyJWTError as exc:
        raise AuthError("invalid_token") from exc
    if claims.get("scope") != SHARE_TOKEN_SCOPE:
        raise AuthError("invalid_token")
    if not str(claims.get("scan_id", "")).strip() or not str(claims.get("sub", "")).strip():
        raise AuthError("invalid_token")
    return claims


def _extract_share_token() -> Optional[str]:
    """A share token arrives as a Bearer header (fetch/POST) or a ``?share_token=`` query
    param (``<img>``/blob/``.ply`` GETs that can't set headers)."""
    query_token = request.args.get(SHARE_QUERY_PARAM, "")
    return _extract_bearer_token() or (query_token.strip() if query_token.strip() else None)


# --- Building (project) tokens ----------------------------------------------------
# A project token shares a *building* (a whole client/project) as one unit. It is an HS256
# JWT signed with the SAME SESSION_TOKEN_SECRET as the session + share tokens, but under a
# distinct issuer + a ``project:read`` scope and ``client``/``project`` claims, so it can
# never be confused with a session token (issuer) or a single-scene share token (scope).
# It authorizes EXACTLY ONE endpoint — the building manifest (GET /api/projects/manifest) —
# and only for its own client+project. The manifest then mints fresh per-scene share tokens
# so the per-scene read security model is unchanged (a project token never reads a scene
# directly). The owner ``sub`` is carried so the user-scoped store lookups resolve to the
# owner's scenes. See platform/contracts/embed-delivery.md.
PROJECT_TOKEN_ISSUER = "open-reality-project"
PROJECT_TOKEN_SCOPE = "project:read"
PROJECT_TOKEN_TTL_SEC_DEFAULT = int(
    os.environ.get("PROJECT_TOKEN_TTL_SEC", str(SHARE_TOKEN_TTL_SEC_DEFAULT))
)
PROJECT_QUERY_PARAM = "project_token"
# The single endpoint (view-function name) a project token may reach.
PROJECT_TOKEN_ENDPOINT = "project_manifest_route"


def _issue_project_token(
    client: str, project: str, owner_sub: str, ttl_seconds: Optional[int] = None
) -> tuple[str, int]:
    """Mint a read-only building token for one ``client``/``project``. Returns
    ``(token, expires_at_epoch)``. Mirrors ``_issue_share_token`` but scoped to a project."""
    now = int(time.time())
    ttl = PROJECT_TOKEN_TTL_SEC_DEFAULT
    if ttl_seconds is not None:
        try:
            ttl = int(ttl_seconds)
        except (TypeError, ValueError):
            ttl = PROJECT_TOKEN_TTL_SEC_DEFAULT
    ttl = max(1, min(ttl, SHARE_TOKEN_TTL_MAX))
    expires_at = now + ttl
    payload = {
        "iss": PROJECT_TOKEN_ISSUER,
        "scope": PROJECT_TOKEN_SCOPE,
        "client": str(client),
        "project": str(project),
        "sub": str(owner_sub).strip(),
        "iat": now,
        "exp": expires_at,
    }
    token = jwt.encode(payload, SESSION_TOKEN_SECRET, algorithm=SESSION_TOKEN_ALG)
    return token, expires_at


def _verify_project_token(token: str) -> dict[str, Any]:
    """Verify a building token (HS256 + our project issuer). Raises ``AuthError`` if it is
    not a valid project token — including a Clerk JWT, a session token, or a single-scene
    share token (issuer/scope mismatch) — so the caller can fall through to the normal auth
    path."""
    try:
        claims = jwt.decode(
            token,
            SESSION_TOKEN_SECRET,
            algorithms=[SESSION_TOKEN_ALG],
            issuer=PROJECT_TOKEN_ISSUER,
            options={"require": ["exp", "iss", "sub"]},
        )
    except jwt.PyJWTError as exc:
        raise AuthError("invalid_token") from exc
    if claims.get("scope") != PROJECT_TOKEN_SCOPE:
        raise AuthError("invalid_token")
    if (
        not str(claims.get("sub", "")).strip()
        or not str(claims.get("client", "")).strip()
        or not str(claims.get("project", "")).strip()
    ):
        raise AuthError("invalid_token")
    return claims


def _extract_project_token() -> Optional[str]:
    """A project token arrives as a Bearer header (the building page's fetch) or a
    ``?project_token=`` query param."""
    query_token = request.args.get(PROJECT_QUERY_PARAM, "")
    return _extract_bearer_token() or (query_token.strip() if query_token.strip() else None)


def _scans_remaining(claims: dict[str, Any]) -> Optional[int]:
    """Remaining scans for the caller, or None when access is unlimited.

    Legacy ``tier == "approved"`` accounts act as an admin bypass (unlimited).
    A missing/blank claim is treated as the just-in-time default so a brand-new
    user (never written to Clerk) still gets DEFAULT_SCANS before their first
    scan is recorded. The authoritative balance lives in Clerk and is spent by
    the landing /api/scans/consume route; this claim is the worker's
    best-effort, defence-in-depth gate (it can lag the live balance by the Clerk
    token cache TTL, so the launch flow always spends a scan *after* the worker
    session is created, never before).
    """
    if claims.get("tier") == APPROVED_TIER:
        return None
    raw = claims.get(SCANS_CLAIM)
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return DEFAULT_SCANS
    try:
        return int(raw)
    except (TypeError, ValueError):
        return DEFAULT_SCANS


def _require_scan_access(claims: dict[str, Any]) -> None:
    """Advisory access gate.

    Once billing is enforced the AUTHORITATIVE gate is ``credit_hold`` at each
    dispatch point, which is transactional. This stays as defence in depth and
    deliberately FAILS OPEN on any ledger problem: a database blip must not stop
    people scanning, and the hold will refuse the actual work a moment later
    anyway.

    With enforcement off the behaviour is unchanged — the lagging JWT claim.
    """
    from server.billing import meter

    if meter.enforcing():
        credits = meter.balance(str(claims.get("sub", "")).strip())
        if credits is not None:
            if credits <= 0 and claims.get("tier") != APPROVED_TIER:
                raise AuthError("no_scans_remaining", 403)
            return  # the ledger answered; the stale claim is irrelevant

    remaining = _scans_remaining(claims)
    if remaining is not None and remaining <= 0:
        raise AuthError("no_scans_remaining", 403)


def _extract_bearer_token() -> Optional[str]:
    auth_header = request.headers.get("Authorization", "")
    scheme, _, token = auth_header.partition(" ")
    if scheme.lower() == "bearer" and token.strip():
        return token.strip()
    return None


def _extract_http_token() -> Optional[str]:
    query_token = request.args.get(AUTH_QUERY_PARAM, "")
    return (
        _extract_bearer_token()
        or request.cookies.get(AUTH_COOKIE_NAME)
        or (query_token.strip() if query_token.strip() else None)
    )


def _verify_http_token() -> dict[str, Any]:
    if DANGEROUSLY_DISABLE_AUTH:
        return _bypass_claims()
    token = _extract_http_token()
    if not token:
        raise AuthError("missing_token")
    # HTTP accepts the durable broker session token in addition to the Clerk JWT, so
    # revisit/history reads + Q&A survive the hash JWT's expiry. The live socket path
    # (_verify_socket_auth) stays Clerk-only.
    claims = _verify_any_token(token)
    _require_scan_access(claims)
    return claims


def _extract_socket_cookie_token(environ: dict[str, Any]) -> Optional[str]:
    cookie_header = str(environ.get("HTTP_COOKIE", "")).strip()
    if not cookie_header:
        return None
    try:
        cookie = SimpleCookie()
        cookie.load(cookie_header)
    except Exception:
        return None
    morsel = cookie.get(AUTH_COOKIE_NAME)
    if morsel is None:
        return None
    token = str(morsel.value).strip()
    return token or None


def _auth_response(error: AuthError):
    return jsonify({"error": str(error)}), error.status_code


def _jwt_expiry_datetime(claims: dict[str, Any]) -> datetime:
    return datetime.fromtimestamp(int(claims["exp"]), tz=timezone.utc)


def _verify_socket_auth(auth: Any, environ: dict[str, Any]) -> dict[str, Any]:
    if DANGEROUSLY_DISABLE_AUTH:
        return _bypass_claims()
    token = auth.get("token") if isinstance(auth, dict) else None
    if not isinstance(token, str) or not token.strip():
        token = _extract_socket_cookie_token(environ)
    if not isinstance(token, str) or not token.strip():
        raise AuthError("missing_token")
    claims = _verify_clerk_token(token.strip())
    _require_scan_access(claims)
    return claims

# ------------------------------
# Flask + python-socketio Setup
# ------------------------------
app = Flask(__name__)
_cors_origins = sorted(CLERK_JWT_ALLOWED_AZP)
CORS(app, origins=_cors_origins, supports_credentials=True)

sio = socketio_pkg.AsyncServer(
    async_mode="asgi",
    # Socket.IO still requires a valid Clerk token in handle_connect().
    # Allowing origins here lets Modal-hosted static pages reach that auth check.
    cors_allowed_origins="*",
    max_http_buffer_size=10_000_000,
    ping_timeout=120,
    ping_interval=25,
)
_socketio_asgi_application = socketio_pkg.ASGIApp(sio, WsgiToAsgi(app))


async def asgi_application(scope, receive, send):
    if scope["type"] == "lifespan":
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                await send({"type": "lifespan.startup.complete"})
            elif message["type"] == "lifespan.shutdown":
                await send({"type": "lifespan.shutdown.complete"})
                return
        return

    # The Flask half of the socket.io ASGI app is a WSGI wrapper (WsgiToAsgi), which RAISES
    # ("received a non-HTTP-request message") on any websocket scope it gets handed — and that
    # uncaught raise crashes the whole broker/worker container (observed on Modal: "Runner
    # disappeared", intermittent 404s). socket.io's own transport lives under /socket.io/ and is
    # handled by the engineio ASGI layer; any OTHER websocket upgrade (a stray client, a probe,
    # a viewer page pointed at the wrong path) must be closed cleanly here so it can never reach
    # the WSGI wrapper. HTTP scopes pass straight through unchanged.
    if scope["type"] == "websocket" and not scope.get("path", "").startswith("/socket.io"):
        try:
            await receive()  # drain the initial websocket.connect
        except Exception:
            pass
        await send({"type": "websocket.close", "code": 1000})
        return

    await _socketio_asgi_application(scope, receive, send)

# Queues for streaming pipeline
frame_queue = queue.Queue(maxsize=30)
result_queue = queue.Queue(maxsize=10)

# Global SLAM processor (initialized in initialize() or start_server())
slam_processor: Optional[StreamingSLAM] = None
client_connected = threading.Event()

# Track connected socket IDs so we only stop SLAM when the last client leaves
_connected_sids: set[str] = set()
_sids_lock = threading.Lock()

# Background streaming task handle — started once when the first client connects
_stream_task: Optional[asyncio.Task] = None

# Event loop for scheduling cross-thread async tasks
_event_loop: Optional[asyncio.AbstractEventLoop] = None
_result_ready: Optional[asyncio.Event] = None

# Single-threaded executor for blocking GPU ops (segment_all / detection pipeline)
_gpu_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

# Demo mode (pre-recorded local videos)
_DEMO_VIDEO_EXTENSIONS = {'.mp4', '.mov', '.m4v', '.avi', '.mkv', '.webm'}

_ROTATION_MAP = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def _apply_video_rotation(frame, angle):
    """Rotate a frame to correct for orientation metadata (e.g. iPhone portrait MOV)."""
    code = _ROTATION_MAP.get(int(angle) % 360)
    if code is not None:
        return cv2.rotate(frame, code)
    return frame
_demo_lock = threading.Lock()
_demo_video_feeder = None
_demo_active_video_id = None
_demo_active_video_path = None
_demo_started_at = None
_demo_target_fps = None
_demo_thumbnail_cache = {}
_demo_catalog_cache = None
# Agent executor (model calls for multiple sessions)
_agent_executor = concurrent.futures.ThreadPoolExecutor(max_workers=6)

# Per-query detection task registry: (sid, query) -> asyncio.Task
_query_tasks: dict[tuple[str, str], asyncio.Task] = {}

# Query update lock (lazy-initialized under async context)
_query_update_lock: Optional[asyncio.Lock] = None

# OpenRouter clients for HTTP APIs
_plan_client: Optional[OpenRouterClient] = None
_assistant_client: Optional[OpenRouterClient] = None
_object_enricher: Optional[ObjectEnricher] = None

# OpenRouter key cached during initialize()
_openrouter_api_key: str = ""

# Fine-grained object enrichment (turns "laptop" -> "MacBook Pro 14\", Space Black,
# likely M3"). A VLM captions a crop of the object's best keyframe, optionally grounded by
# a reverse image search. Runs on click (on-demand) and over the top-N at finalize.
OBJECT_ENRICH_ENABLED = os.environ.get("OBJECT_ENRICH_ENABLED", "1").strip().lower() not in {
    "0", "false", "no", "",
}
OBJECT_ENRICH_MODEL = os.environ.get("OBJECT_ENRICH_MODEL", "google/gemini-3-flash-preview")
OBJECT_ENRICH_FALLBACKS = [
    m.strip() for m in os.environ.get("OBJECT_ENRICH_FALLBACKS", "openai/gpt-4o-mini").split(",")
    if m.strip()
]
OBJECT_ENRICH_FINALIZE_TOP_N = int(os.environ.get("OBJECT_ENRICH_FINALIZE_TOP_N", "8"))
# Reverse image search grounding (Phase B) — OFF by default. Needs a public crop URL the
# scraper can fetch (served by /api/crop/<token>.jpg below) + a SerpApi key.
OBJECT_ENRICH_REVERSE_SEARCH = os.environ.get(
    "OBJECT_ENRICH_REVERSE_SEARCH", "0"
).strip().lower() in {"1", "true", "yes", "on"}
OBJECT_ENRICH_CROP_TTL_S = float(os.environ.get("OBJECT_ENRICH_CROP_TTL_S", "600"))

# End-of-scan scene report (Phase 1+2). The extractor is stateless; the builder
# is lazily constructed once the OpenRouter key is known and falls back to a
# fact-only report when no key/LLM is available.
_scene_feature_extractor = SceneFeatureExtractor()
_scene_report_builder: Optional[SceneReportBuilder] = None
# Last generated report, cached so a separately-connected summary page (a
# different sid than the scanning viewer) can fetch it via get_scene_report
# after the scan has stopped. Safe to keep at module scope: each worker serves a
# single Clerk user.
_last_scene_report: Optional[dict[str, Any]] = None
_scene_report_lock = threading.Lock()
_scene_report_last_trigger: float = 0.0

# Durable per-user scene persistence (Phase 4). Injected by modal_streaming.py
# (a ModalScenePersistence over a modal.Dict + Volume) on both the GPU worker (which
# writes finished scans) and the CPU broker (which serves them with no GPU). None in
# plain local runs unless configured. Scans are keyed by (user_id, scan_id); a fresh
# scan_id is minted whenever a new scan starts and persisted when the scan ends.
_scene_persistence: Any = None
_current_scan_id: Optional[str] = None
# Optional caller-supplied scan id used for the NEXT scan start. The offline pilot harness sets
# this so a clip persists under a chosen id (e.g. "chorus-1") instead of a random hash: the
# video feed path auto-starts SLAM via _begin_new_scan(), which would otherwise overwrite the
# id the harness set. The live server never sets this (stays None → a fresh hash is minted, as
# before). It is consumed (cleared) on use so it can't leak into a later, unrelated scan.
_pilot_scan_id: Optional[str] = None
# Optional caller-supplied human label for the NEXT finalized scan (e.g. "Amenity Floor"). The
# offline pilot harness sets this per clip (from the captures manifest) so the scan persists with
# its space name and the building manifest shows it directly — no separate label post-step. Like
# _pilot_scan_id it is a per-scan one-shot: consumed (cleared) at persist so it can't leak into a
# later, unlabeled scan. The live server never sets it (stays None → label None, as before).
_pilot_label: Optional[str] = None
# The full-fidelity cloud is stored UNcapped (the durable artifact). For *display*
# only, the broker downsamples the cloud to this many points on read so the revisit
# /points payload stays snappy (~1M ≈ 20MB base64) without a sparse "skeleton". This
# never touches the stored artifact (downloadable in full as PLY); 0 = no downsample.
SCENE_DISPLAY_MAX_POINTS = int(os.environ.get("SCENE_DISPLAY_MAX_POINTS", "1000000"))


_share_access: Optional[ShareAccessLog] = None


def configure_scene_persistence(persistence: Any) -> None:
    """Inject the durable scene store (mirrors configure_session_store)."""
    global _scene_persistence, _share_access
    _scene_persistence = persistence
    # Share-link access log rides on the same store (derived/share_access/events.json).
    _share_access = ShareAccessLog(persistence) if persistence is not None else None


def _record_share_access(claims: dict[str, Any], token: str, *, question: Optional[str] = None) -> None:
    """Append a share-link access event (see ``server/share_access.py``). Never raises."""
    log = _share_access
    if log is None:
        return
    try:
        endpoint = request.url_rule.endpoint if request.url_rule else ""
        log.record(
            user_id=str(claims.get("sub", "")),
            scan_id=str(claims.get("scan_id", "")),
            token=token,
            token_iat=claims.get("iat"),
            token_exp=claims.get("exp"),
            endpoint=endpoint or "",
            method=getattr(request, "method", "GET") or "GET",
            question=question,
            remote_addr=(request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
                         or getattr(request, "remote_addr", None)),
            user_agent=request.headers.get("User-Agent"),
        )
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[share_access] hook failed: {exc}")


# -- pilot reconstruction config (multi-video harness) ---------------------------
# Optional grouping tags + a 3DGS-splat-export toggle that the end-of-scan persist
# (``_persist_scene_report``) threads into ``save_scene(...)``. Off / None by default
# so ordinary scans (live sender, demo, single ``--video``) are completely unaffected.
# Driven by ``server/reconstruct_pilot.py`` (or env: PILOT_CLIENT / PILOT_PROJECT /
# EXPORT_SPLAT).
_persist_client: Optional[str] = os.environ.get("PILOT_CLIENT") or None
_persist_project: Optional[str] = os.environ.get("PILOT_PROJECT") or None
_persist_export_splat: bool = os.environ.get("EXPORT_SPLAT", "").strip().lower() in (
    "1", "true", "yes", "on",
)
# When set (offline pilot harness), the finalize path runs an agentic object-detection pass
# before building the report: the LLM auto-discovers the objects visible in the scan's
# keyframes, then the SAM3/CLIP detector locates each in 3D, so facts.objects (the source of
# the viewer's 3D bounding boxes) is populated. Off by default — the live flow runs detection
# interactively via the spatial agent, so this never changes live behavior.
_pilot_detect_objects: bool = False
# Cap on auto-discovered object queries per scan (each is a detector pass over all submaps).
# Already env-configurable -- the detection-recall knobs added alongside this comment
# (DETECTION_CLIP_THRESHOLD / DETECTION_SAM_THRESHOLD / DETECTION_FRAMES_PER_WINDOW, see
# server/streaming_slam.py's _detection_env_defaults) intentionally don't duplicate this one.
# Raised 12 -> 20 for detection recall (EXP-46 PILOT-LOG L8/L9): with 12, scene0663's
# inventory came out as 10 labels / 34 detections while the questions asked about it
# referenced whiteboard, shelf, window, trash can, telephone and cabinet -- none of which
# had ever been queried, so grounded answers lost to blind guessing on those. The cap, not
# the segmentation model, was the binding constraint. Each extra query costs one detector
# pass over all submaps, which is why this is a budget and not "everything".
PILOT_DETECT_MAX_QUERIES = int(os.environ.get("PILOT_DETECT_MAX_QUERIES", "20"))

# The pilot harness mutates the module-global scan-id + persistence config above and drives
# the shared SLAM session. It is an OFFLINE batch tool (a dedicated process, not a live
# serving worker). This non-reentrant lock guards against two harness runs interleaving
# their global mutations in one process; the harness acquires it for the whole run.
_pilot_lock = threading.Lock()


def configure_pilot_persistence(
    client: Optional[str] = None,
    project: Optional[str] = None,
    export_splat: Optional[bool] = None,
    detect_objects: Optional[bool] = None,
) -> None:
    """Tag the next finalized scan with ``client``/``project`` and (optionally) attach a
    3DGS ``splat.ply`` exported from the live solver. ``detect_objects`` enables the offline
    agentic object-detection pass at finalize (see ``_pilot_detect_objects``). Used by the
    multi-video pilot harness; ``None`` arguments leave the current value (incl. the env
    defaults) untouched so callers can set just one knob."""
    global _persist_client, _persist_project, _persist_export_splat, _pilot_detect_objects
    if client is not None:
        _persist_client = str(client) or None
    if project is not None:
        _persist_project = str(project) or None
    if export_splat is not None:
        _persist_export_splat = bool(export_splat)
    if detect_objects is not None:
        _pilot_detect_objects = bool(detect_objects)


def _export_splat_bytes(solver: Any, refine: bool = False) -> Optional[bytes]:
    """Best-effort: export a 3DGS ``splat.ply`` from a finished ``Solver`` and return
    its bytes, or ``None`` on any failure (missing core/Open3D, no GPU, empty map).

    The exporter lives in **core** (``vggt_slam.splat_export.export_splat``) and is
    GPU/Open3D-adjacent, so the import is lazy and fully guarded — in CI ``vggt_slam``
    may be absent, and a failure here must degrade (scene still persists, just without a
    splat), never crash the persist path."""
    if solver is None:
        return None
    try:
        from vggt_slam.splat_export import export_splat  # lazy: GPU/Open3D-adjacent
    except Exception as exc:  # ImportError in CI, or transitive heavy-import failure
        print(f"[pilot] splat export unavailable (import failed): {exc}")
        return None
    tmp_dir = tempfile.mkdtemp(prefix="pilot_splat_")
    out_path = os.path.join(tmp_dir, "splat.ply")
    try:
        result = export_splat(solver, out_path, refine=refine)
        # export_splat returns the path on success, None on a no-op (empty map).
        read_path = result if isinstance(result, str) and os.path.isfile(result) else out_path
        if not os.path.isfile(read_path):
            print("[pilot] splat export produced no file (empty map?)")
            return None
        with open(read_path, "rb") as fh:
            data = fh.read()
        return data or None
    except Exception as exc:
        print(f"[pilot] splat export failed: {exc}")
        return None
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# Progressive (in-scan) scene report (Phase 3). Refreshed every N submaps while the
# scan runs and broadcast as ``scene_report_update``, then superseded by the final
# ``scene_report_ready`` once the scan stops. Decoupled from the agent, so it also
# runs in reconstruction-only mode.
SCENE_REPORT_PROGRESSIVE = os.environ.get("SCENE_REPORT_PROGRESSIVE", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "",
}
SCENE_REPORT_PROGRESSIVE_EVERY_N = max(
    1, int(os.environ.get("SCENE_REPORT_PROGRESSIVE_EVERY_N_SUBMAPS", "2"))
)
SCENE_REPORT_PROGRESSIVE_MIN_INTERVAL_S = float(
    os.environ.get("SCENE_REPORT_PROGRESSIVE_MIN_INTERVAL_S", "12")
)
SCENE_REPORT_PROGRESSIVE_KEYFRAMES = max(
    1, int(os.environ.get("SCENE_REPORT_PROGRESSIVE_KEYFRAMES", "4"))
)
# Running progressive report (kept as the `previous` for the next incremental fold).
_progressive_report: Optional[SceneReport] = None
_progressive_report_last_submaps: int = 0
_progressive_report_last_ts: float = 0.0
_progressive_report_lock = threading.Lock()
# Guards against overlapping builds: the LLM call can outlast the throttle interval.
_progressive_report_inflight: bool = False
# Set once the final report builds, so any in-flight progressive refresh stops emitting.
_scene_report_finalized: bool = False

# Salvage of abandoned scans: when the last client vanishes without stop_slam (dead
# transport, killed app, locked phone), persist the scan anyway after a reconnect grace
# window — handle_disconnect's cleanup path never builds a report, so without this the
# whole map is dropped at worker shutdown. The generation counter lets any thread (a
# socket connect on the loop, /reset on a route thread) void a pending salvage without
# cross-thread task cancellation.
SCAN_SALVAGE_GRACE_S = float(os.environ.get("SCAN_SALVAGE_GRACE_S", "15"))
_scan_salvage_task: Optional[asyncio.Task] = None
_scan_salvage_generation: int = 0
_scan_salvage_lock = threading.Lock()

# Input guardrails (no-auth deployment still needs abuse protection)
MAX_QUERY_COUNT = int(os.environ.get("MAX_QUERY_COUNT", "16"))
MAX_QUERY_LEN = int(os.environ.get("MAX_QUERY_LEN", "120"))
MAX_FRAME_B64_LEN = int(os.environ.get("MAX_FRAME_B64_LEN", "10000000"))
FRAME_RATE_WINDOW_S = float(os.environ.get("FRAME_RATE_WINDOW_S", "1.0"))
FRAME_RATE_LIMIT = int(os.environ.get("FRAME_RATE_LIMIT", "45"))

_frame_rate_state: dict[str, list[float]] = {}


@dataclass
class SessionState:
    sid: str
    user_id: str
    claims: dict[str, Any] = field(default_factory=dict)
    manual_queries: set[str] = field(default_factory=set)
    agent_queries: set[str] = field(default_factory=set)
    reconstruction_only: bool = False
    agent: Any = None
    connected_at: float = field(default_factory=time.time)
    ui_results: list[dict[str, Any]] = field(default_factory=list)


_sessions: dict[str, SessionState] = {}
_sessions_lock = threading.Lock()
_session_timeout_tasks: dict[str, asyncio.Task] = {}

_active_session_store: Any = None
_local_active_sessions: dict[str, dict[str, Any]] = {}
_active_sessions_lock = threading.Lock()
_gpu_session_broker: Any = None
_worker_expected_user_id: Optional[str] = None
_worker_activity_callback: Optional[Callable[[], None]] = None


def configure_session_store(store: Any) -> None:
    global _active_session_store
    _active_session_store = store


def configure_gpu_session_broker(broker: Any) -> None:
    global _gpu_session_broker
    _gpu_session_broker = broker


def configure_worker_runtime(
    user_id: Optional[str],
    activity_callback: Optional[Callable[[], None]] = None,
) -> None:
    """Bind this single-session worker to one user and track live activity."""
    global _worker_expected_user_id, _worker_activity_callback
    _worker_expected_user_id = str(user_id).strip() if user_id else None
    _worker_activity_callback = activity_callback


def has_connected_clients() -> bool:
    with _sids_lock:
        return bool(_connected_sids)


def _note_worker_activity() -> None:
    if _worker_activity_callback is not None:
        _worker_activity_callback()


def _require_worker_owner(claims: dict[str, Any]) -> None:
    if _worker_expected_user_id is None:
        return
    if str(claims.get("sub", "")).strip() != _worker_expected_user_id:
        raise AuthError("session_busy", 403)


def _get_active_session_store():
    return _active_session_store if _active_session_store is not None else _local_active_sessions


def _entry_has_live_sid(entry: dict[str, Any], now: float) -> bool:
    """True if a session-store entry still holds at least one non-expired sid."""
    raw_sids = entry.get("sids")
    if isinstance(raw_sids, dict):
        for expires_at in raw_sids.values():
            try:
                if float(expires_at) > now:
                    return True
            except (TypeError, ValueError):
                continue
        return False
    # Legacy single-sid entry shape.
    legacy_sid = entry.get("sid")
    try:
        return bool(legacy_sid) and float(entry.get("expires_at", 0)) > now
    except (TypeError, ValueError):
        return False


def _claim_user_session_sync(user_id: str, sid: str) -> bool:
    """Claim the single-occupancy GPU slot for ``user_id``.

    The container holds one global SLAM map (``slam_processor``) and Modal runs
    at most one container (``max_containers=1``), so two *different* users would
    otherwise share one reconstruction — a cross-user scene/detection leak. We
    reject a claim while another user has a live session. Multiple sids for the
    *same* user (e.g. laptop viewer + phone sender) are allowed.

    A stale entry left by an unclean disconnect self-heals once its sids pass
    ``SESSION_TIMEOUT_SEC``; worst case a new user waits out that window.

    Returns False if a different user currently holds the slot.
    """
    now = time.time()
    expires_at = now + SESSION_TIMEOUT_SEC
    store = _get_active_session_store()
    with _active_sessions_lock:
        for other_id, other_entry in list(store.items()):
            if str(other_id) == user_id or not isinstance(other_entry, dict):
                continue
            if _entry_has_live_sid(other_entry, now):
                return False

        existing = store.get(user_id)
        sids: dict[str, float] = {}
        if isinstance(existing, dict):
            raw_sids = existing.get("sids")
            if isinstance(raw_sids, dict):
                for existing_sid, existing_expires_at in raw_sids.items():
                    try:
                        parsed_expires_at = float(existing_expires_at)
                    except (TypeError, ValueError):
                        continue
                    if parsed_expires_at > now:
                        sids[str(existing_sid)] = parsed_expires_at
            else:
                existing_sid = existing.get("sid")
                try:
                    existing_expires_at = float(existing.get("expires_at", 0))
                except (TypeError, ValueError):
                    existing_expires_at = 0
                if existing_sid and existing_expires_at > now:
                    sids[str(existing_sid)] = existing_expires_at

        sids[sid] = expires_at
        store[user_id] = {"sids": sids, "expires_at": max(sids.values())}
    return True


def _release_user_session_sync(user_id: str, sid: str) -> None:
    store = _get_active_session_store()
    with _active_sessions_lock:
        existing = store.get(user_id)
        if not isinstance(existing, dict):
            return
        raw_sids = existing.get("sids")
        if isinstance(raw_sids, dict):
            remaining_sids: dict[str, float] = {}
            for existing_sid, existing_expires_at in raw_sids.items():
                if existing_sid == sid:
                    continue
                try:
                    parsed_expires_at = float(existing_expires_at)
                except (TypeError, ValueError):
                    continue
                remaining_sids[str(existing_sid)] = parsed_expires_at
            if remaining_sids:
                store[user_id] = {
                    "sids": remaining_sids,
                    "expires_at": max(remaining_sids.values()),
                }
                return
        elif existing.get("sid") != sid:
            return

        try:
            store.pop(user_id, None)
        except TypeError:
            try:
                store.pop(user_id)
            except KeyError:
                pass


async def _claim_user_session(user_id: str, sid: str) -> bool:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _claim_user_session_sync, user_id, sid)


async def _release_user_session(user_id: str, sid: str) -> None:
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _release_user_session_sync, user_id, sid)


# ------------------------------
# Helpers: Query/session management
# ------------------------------
def _ensure_query_lock() -> asyncio.Lock:
    global _query_update_lock
    if _query_update_lock is None:
        _query_update_lock = asyncio.Lock()
    return _query_update_lock


def _normalize_query_list(queries: list[Any]) -> list[str]:
    norm: list[str] = []
    for item in queries:
        q = str(item).strip().lower()
        if len(q) > MAX_QUERY_LEN:
            q = q[:MAX_QUERY_LEN]
        if q and q not in norm:
            norm.append(q)
        if len(norm) >= MAX_QUERY_COUNT:
            break
    return norm


def _session_active_queries(sid: str) -> list[str]:
    with _sessions_lock:
        state = _sessions.get(sid)
        if state is None:
            return []
        if state.reconstruction_only:
            return []
        merged = sorted(state.manual_queries | state.agent_queries)
    return merged


def _filter_detections_by_queries(detections: list[dict[str, Any]], queries: list[str]) -> list[dict[str, Any]]:
    if not queries:
        return []
    query_set = set(queries)
    return [
        det for det in detections
        if str(det.get("query", "")).strip().lower() in query_set
    ]


# Pacing for intermediate detection_partial emits, per sid. Finals always go out.
_detection_partial_last_emit: dict[str, float] = {}
DETECTION_PARTIAL_MIN_INTERVAL_S = float(
    os.environ.get("DETECTION_PARTIAL_MIN_INTERVAL_S", "0.3")
)


def _wire_detections(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy detections for a live socket emit, dropping ``mask_rle``. The RLE dominates
    the wire size (O(2·H) run-length ints per detection) and no live client decodes it:
    the web viewer's overlay decode is unimplemented (protocol ``types.ts`` says so) and
    the iOS model marks the field optional. Exports and persistence read the stored
    dicts, which keep it. Measured 2026-08-13: 14 query workers × 11 submaps each
    re-emitting all 14 masked detections serialized ~2k detection payloads across 154
    emits in ~2s, all inline on the event loop — starving socket.io pings until live
    phones timed out mid-scan."""
    return [
        {k: v for k, v in det.items() if k != "mask_rle"}
        if det.get("mask_rle") is not None
        else det
        for det in detections
    ]


def _allow_frame_for_sid(sid: str) -> bool:
    now = time.time()
    with _sessions_lock:
        ts = _frame_rate_state.setdefault(sid, [])
        ts.append(now)
        cutoff = now - FRAME_RATE_WINDOW_S
        while ts and ts[0] < cutoff:
            ts.pop(0)
        return len(ts) <= FRAME_RATE_LIMIT


def _is_sid_connected(sid: str) -> bool:
    with _sids_lock:
        return sid in _connected_sids


def _emit_to_sid_threadsafe(sid: str, event: str, data: dict[str, Any]):
    loop = _event_loop
    if loop is None or loop.is_closed():
        return
    if not _is_sid_connected(sid):
        return

    def _on_done(fut: concurrent.futures.Future):
        try:
            fut.result()
        except Exception as emit_err:
            print(f"Emit error sid={sid} event={event}: {emit_err}")

    try:
        fut = asyncio.run_coroutine_threadsafe(sio.emit(event, data, to=sid), loop)
        fut.add_done_callback(_on_done)
    except Exception as e:
        print(f"Emit error sid={sid} event={event}: {e}")


def _get_scene_report_builder() -> SceneReportBuilder:
    """Lazily build the end-of-scan report builder.

    Uses the OpenRouter key cached in initialize(); if absent, the builder is
    constructed with no client and emits a fact-only (degraded) report.
    """
    global _scene_report_builder
    if _scene_report_builder is None:
        client = None
        if _openrouter_api_key:
            try:
                model = os.environ.get(
                    "SCENE_REPORT_MODEL",
                    os.environ.get("SPATIAL_ORCH_MODEL", "google/gemini-3-flash-preview"),
                )
                fallbacks = [
                    m.strip()
                    for m in os.environ.get("SCENE_REPORT_FALLBACKS", "openai/gpt-4o-mini").split(",")
                    if m.strip()
                ]
                client = OpenRouterClient(
                    api_key=_openrouter_api_key,
                    primary_model=model,
                    fallback_models=fallbacks,
                    timeout=float(os.environ.get("SCENE_REPORT_TIMEOUT_S", "30")),
                    max_retries=int(os.environ.get("SCENE_REPORT_RETRIES", "1")),
                    usage_sink=named_tally("scene_report"),
                )
            except Exception as e:
                print(f"Scene report LLM client init failed: {e}")
                client = None
        _scene_report_builder = SceneReportBuilder(
            client, max_keyframes=int(os.environ.get("SCENE_REPORT_MAX_KEYFRAMES", "12"))
        )
    return _scene_report_builder


def _broadcast_scene_report(payload: dict[str, Any], event: str = "scene_report_ready") -> None:
    """Emit a scene report to every connected sid on this (single-user) worker.

    Broadcasts (not per-sid) because the summary page is a separate socket from the
    scanning viewer — a per-sid emit would miss it."""
    with _sids_lock:
        sids = list(_connected_sids)
    for s in sids:
        _emit_to_sid_threadsafe(s, event, payload)


def _clear_scene_report() -> None:
    global _last_scene_report, _progressive_report, _progressive_report_inflight
    global _progressive_report_last_submaps, _progressive_report_last_ts, _scene_report_finalized
    _last_scene_report = None
    with _progressive_report_lock:
        _progressive_report = None
        _progressive_report_last_submaps = 0
        _progressive_report_last_ts = 0.0
        _progressive_report_inflight = False
        _scene_report_finalized = False


def _begin_new_scan() -> None:
    """Start of a fresh scan: drop any prior report/progressive state and mint a new
    scan_id so the end-of-scan report is persisted under its own key. If a caller
    pre-set ``_pilot_scan_id`` (the offline pilot harness), adopt that id for this scan
    instead of a random hash, and consume it so it can't leak into a later scan."""
    global _current_scan_id, _pilot_scan_id
    _clear_scene_report()
    _current_scan_id = _pilot_scan_id or uuid.uuid4().hex
    _pilot_scan_id = None


def _persist_scene_report(report: SceneReport, facts, scene_center) -> None:
    """Best-effort durable save of a finished scan (Phase 4). Runs on the GPU worker
    only (slam_processor present); the broker reads these back with no GPU. Never
    raises into the emit path — a persistence failure must not break the live report."""
    global _current_scan_id, _pilot_label
    if _scene_persistence is None or slam_processor is None:
        return
    try:
        user_id = _worker_expected_user_id or "local"
        scan_id = _current_scan_id or uuid.uuid4().hex
        _current_scan_id = scan_id
        # Consume the per-scan label one-shot (set by the pilot harness from the captures
        # manifest); clear it so the next, unlabeled scan in this run can't inherit it.
        label = _pilot_label
        _pilot_label = None
        refs = choose_frame_refs(
            facts.objects, slam_processor.solver,
            int(os.environ.get("SCENE_REPORT_MAX_KEYFRAMES", "12")),
        )
        keyframes_b64 = encode_frames(slam_processor.solver, refs)
        # The COMPLETE world-frame cloud (no stride/cap) — the durable artifact. The
        # broker downsamples it for display on read and serves the full thing as PLY.
        points = slam_processor.gather_world_point_cloud()
        # Optional 3DGS splat + pilot grouping tags (multi-video harness). Best-effort:
        # a splat-export failure leaves splat_bytes None so the scene still persists,
        # mirroring the store's best-effort blob writes. Off by default → ordinary scans
        # call save_scene exactly as before. The helper already swallows its own errors,
        # but the call is guarded here too so an unexpected raise can never abort persist.
        splat_bytes = None
        if _persist_export_splat:
            try:
                splat_bytes = _export_splat_bytes(slam_processor.solver)
            except Exception as e:
                print(f"[pilot] splat export raised (degrading): {e}")
                splat_bytes = None
        _scene_persistence.save_scene(
            user_id, scan_id, report, facts,
            keyframes_b64=keyframes_b64, points=points,
            splat_bytes=splat_bytes,
            client=_persist_client, project=_persist_project,
            label=label,
        )
        print(f"[scene_store] saved scan {scan_id} for user {user_id} "
              f"({len(keyframes_b64)} keyframes, {points[0].shape[0]} points, "
              f"splat={'yes' if splat_bytes else 'no'}, "
              f"client={_persist_client}, project={_persist_project}, label={label!r})")
    except Exception as e:
        print(f"[scene_store] persist failed: {e}")


def _build_and_emit_scene_report(sid: Optional[str] = None) -> None:
    """Extract 3D facts from the finished scan, then cache + broadcast a report.

    Runs on a worker thread (dispatched from handle_stop, the demo-stop route, or
    end-of-demo-video). The report is cached in ``_last_scene_report`` so a
    separately-connected summary page can fetch it via ``get_scene_report``, and
    broadcast to all of this user's sids. Best-effort: failures broadcast an error
    payload rather than raising.
    """
    global _last_scene_report, _scene_report_finalized
    if slam_processor is None:
        return
    # The scan is over: stop any further progressive refreshes so a late one cannot
    # clobber the final report on the summary page.
    with _progressive_report_lock:
        _scene_report_finalized = True
    try:
        with slam_processor._detection_lock:
            detections = list(slam_processor.accumulated_detections)
        # Fine-grained enrichment of the most prominent objects so the saved report +
        # revisit carry rich labels even for objects nobody clicked. The GPU loop is idle
        # at finalize, so crop + VLM run here on the agent thread. Best-effort.
        if getattr(slam_processor, "object_enricher", None) is not None:
            try:
                slam_processor.enrich_top_n(detections, OBJECT_ENRICH_FINALIZE_TOP_N)
            except Exception as e:
                print(f"[enrich] finalize top-N failed: {e}")
        scene_center = slam_processor.latest_scene_center
        solver = slam_processor.solver
        facts = _scene_feature_extractor.extract(solver, detections, scene_center)
        if facts.metrics.num_submaps == 0:
            return  # nothing scanned yet; don't emit an empty report
        report = _get_scene_report_builder().build_final(solver, facts, scene_center)
        payload = report.model_dump()
        _last_scene_report = payload
        # Durable save BEFORE the live emit: the native app holds its "Saving" state
        # until scene_report_ready and then immediately lists the library, so the scene
        # row must exist by the time the event lands (emit-first raced the save and the
        # fresh scan was missing from the first library fetch). Costs the summary page
        # a few seconds of blob writes; the app's save deadline (120s) absorbs it.
        _persist_scene_report(report, facts, scene_center)
        _broadcast_scene_report(payload)
    except Exception as e:
        print(f"Scene report build error sid={sid}: {e}")
        err = {"error": str(e)}
        if sid:
            _emit_to_sid_threadsafe(sid, "scene_report_ready", err)
        else:
            _broadcast_scene_report(err)


def _wait_for_frame_queue_drain(timeout_s: float = 30.0, grace_s: float = 1.5) -> None:
    """Block until the SLAM frame queue drains, plus a short grace for the in-flight
    submap to finish — used before an end-of-video report so the last submap is
    included."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if frame_queue.empty():
            time.sleep(grace_s)
            return
        time.sleep(0.5)


async def _run_end_of_scan_report(wait_for_drain: bool = False) -> None:
    """Finalize detections, then build + broadcast the scene report. Shared by the
    demo-stop route and end-of-demo-video completion (handle_stop has its own inline
    path for the live sender flow)."""
    if slam_processor is None:
        return
    loop = asyncio.get_event_loop()
    if wait_for_drain:
        await loop.run_in_executor(_agent_executor, _wait_for_frame_queue_drain)
    try:
        # flush=True builds a final submap from any buffered keyframes that never
        # filled a full batch (short/low-motion clip). Runs on the GPU executor so
        # the VGGT pass stays off the event loop and serializes with finalize below.
        await loop.run_in_executor(_gpu_executor, lambda: slam_processor.stop(flush=True))
    except Exception as e:
        print(f"stop() before scene report failed: {e}")
    try:
        await loop.run_in_executor(_gpu_executor, slam_processor.finalize_detection_state)
    except Exception as e:
        print(f"Final detection reconciliation error: {e}")
    await loop.run_in_executor(_agent_executor, _build_and_emit_scene_report, None)


_DISCOVERY_SYS_PROMPT = (
    "You are a spatial-scene analyst looking at several keyframes from a 3D scan of one "
    "physical space. List the distinct, individually-locatable PHYSICAL OBJECTS visible "
    "across the frames (furniture, fixtures, equipment, appliances, structural features). "
    "Use short, concrete open-vocabulary search phrases a detector can find (e.g. 'office "
    "chair', 'electrical panel', 'staircase', 'kitchen island'). Skip whole-room nouns "
    "(room, floor, wall, ceiling), people, and vague abstractions. Deduplicate. "
    'Respond ONLY as JSON: {"objects": ["...", "..."]}.'
)


# General indoor STRUCTURAL categories that the discovery LLM reliably under-proposes: it
# lists the salient furniture and equipment in the keyframes and skips the fabric of the
# room. EXP-46's C2-real probe measured the cost (PILOT-LOG L8/L9) -- questions referenced
# whiteboard, shelf, window, trash can and cabinet, none of them ever queried, so the
# grounded agent honestly answered "not in scene" and scored below blind guessing. These
# are therefore ALWAYS in the discovery list, including when the LLM proposal fails: they
# need no vision call, and an operator who asked for object detection is better served by
# the fabric of the room than by zero boxes.
# Deliberately NOT a benchmark-derived prompt list (that would leak the eval into the
# product): generic indoor structure only. The real fix for referent coverage remains
# question-conditioned open-vocabulary re-query -- specced in L9, not built here.
_STRUCTURAL_DISCOVERY_QUERIES = (
    "door",
    "window",
    "cabinet",
    "shelf",
    "whiteboard",
    "trash can",
)


_QUERY_PLURAL_EXCEPTIONS = {"shelves": "shelf", "boxes": "box"}


def _query_dedupe_key(query: str) -> str:
    """Loose "did we already ask for this category" key: lowercase, whitespace-collapsed,
    naive singular ("Office Chairs" -> "office chair", "shelves" -> "shelf").

    Mirrors core's ``normalize_object_label`` rather than importing it: ``vggt_slam`` is a
    GPU-adjacent import this module keeps lazy and guarded, so pulling it in at module scope
    would break the CI/web-serving import path. Drift between the two folds is tolerable --
    this key only decides whether to spend a second detector pass on the same category,
    while fusing the resulting INVENTORY is core's ``deduplicate_detections`` job -- but the
    irregular-plural map is kept in sync because "shelf"/"shelves" is one of the structural
    categories below and would otherwise cost a duplicate pass on every scan.
    """
    text = " ".join(str(query or "").lower().split())
    if text in _QUERY_PLURAL_EXCEPTIONS:
        return _QUERY_PLURAL_EXCEPTIONS[text]
    if len(text) > 3 and text.endswith("s") and not text.endswith("ss"):
        return text[:-1]
    return text


def _assemble_discovery_queries(proposed: list[str], max_queries: int) -> list[str]:
    """LLM-proposed queries plus the always-on structural categories, deduped and capped.

    The structural categories get RESERVED slots -- they are subtracted from the budget
    before the proposed list is truncated -- so a chatty LLM cannot crowd out the door and
    the window. They never reserve more than half the budget, so a pathologically small cap
    degrades instead of silently dropping every proposed object. Any slack left over (a
    structural category the LLM already proposed under its own spelling) is handed back to
    the next-most-salient proposals, and proposed order is otherwise preserved because the
    LLM ranks by salience.
    """
    limit = max(1, int(max_queries))
    # Two caps exist here and they are NOT the same knob. MAX_QUERY_COUNT is an input
    # guardrail on CLIENT- and agent-supplied query lists (see _normalize_query_list); this
    # is the trusted offline pilot harness, whose budget is PILOT_DETECT_MAX_QUERIES. The
    # pilot budget governs, and when it is the wider of the two we say so in the log rather
    # than silently clamping: a silent clamp would make a recall measurement report a query
    # count it never actually ran (EXP-46 pre-gate P3 reads this list).
    if limit > MAX_QUERY_COUNT:
        print(
            f"[pilot] discovery budget {limit} exceeds the client-input guardrail "
            f"MAX_QUERY_COUNT={MAX_QUERY_COUNT}; offline pilot input is trusted, so the "
            f"budget governs"
        )

    structural: list[str] = []
    structural_keys: set[str] = set()
    for query in _STRUCTURAL_DISCOVERY_QUERIES:
        key = _query_dedupe_key(query)
        if key and key not in structural_keys:
            structural_keys.add(key)
            structural.append(query)

    clean: list[str] = []
    clean_keys: set[str] = set()
    for item in proposed:
        query = " ".join(str(item).strip().lower().split())
        if not query:
            continue
        key = _query_dedupe_key(query)
        if key in clean_keys:
            continue
        clean_keys.add(key)
        clean.append(query)

    out: list[str] = []
    used: set[str] = set()

    def _append(query: str) -> None:
        key = _query_dedupe_key(query)
        if key in used or len(out) >= limit:
            return
        used.add(key)
        out.append(query)

    reserved = min(len(structural), max(1, limit // 2))
    for query in clean[: max(0, limit - reserved)]:
        _append(query)
    for query in structural:
        _append(query)
    for query in clean:  # hand any reserved-but-unneeded slot back to the LLM's ranking
        _append(query)
    return out


def _propose_object_queries_via_llm(max_queries: int) -> list[str]:
    """Ask the report LLM (vision) which distinct objects are visible across the finished
    scan's keyframes, as detector search phrases. Best-effort: [] on any failure."""
    builder = _get_scene_report_builder()
    llm = getattr(builder, "llm", None)
    if llm is None:
        print("[pilot] object proposal skipped — no LLM client (structural queries only)")
        return []
    try:
        # No facts yet → choose_frame_refs falls back to spreading frames across submaps.
        refs = choose_frame_refs([], slam_processor.solver, max(6, min(10, max_queries)))
        frames = encode_frames(slam_processor.solver, refs)
        images_b64 = [f["image_b64"] for f in frames if f.get("image_b64")]
        if not images_b64:
            return []
        parsed, _response = llm.chat_json(
            system_prompt=_DISCOVERY_SYS_PROMPT,
            user_prompt="List the distinct physical objects you can see across these frames.",
            images_b64=images_b64,
            temperature=0.2,
            max_tokens=400,
        )
        raw = (parsed or {}).get("objects") if isinstance(parsed, dict) else None
        if not isinstance(raw, list):
            return []
        seen: set[str] = set()
        queries: list[str] = []
        for item in raw:
            q = str(item).strip().lower()
            if q and q not in seen:
                seen.add(q)
                queries.append(q)
        return queries[:max_queries]
    except Exception as e:
        print(f"[pilot] object proposal failed (falling back to structural queries): {e}")
        return []


def _discover_object_queries(max_queries: int) -> list[str]:
    """The detector search phrases for one finished scan: what the report LLM saw in the
    keyframes, plus the always-on structural categories, deduped and capped at
    ``max_queries``. Best-effort — an LLM failure degrades to the structural list, not to
    an empty one (see ``_STRUCTURAL_DISCOVERY_QUERIES``)."""
    if slam_processor is None:
        return []
    return _assemble_discovery_queries(_propose_object_queries_via_llm(max_queries), max_queries)


def _run_offline_object_detection(max_queries: int) -> int:
    """Offline agentic detection pass (pilot): auto-discover objects, then run the SAM3/CLIP
    detector for each over all submaps so ``accumulated_detections`` (→ facts.objects → the
    viewer's 3D boxes) is populated. Best-effort; returns the number of queries detected.
    Call AFTER stop(flush=True) (all submaps built) and BEFORE finalize_detection_state()."""
    if slam_processor is None:
        return 0
    queries = _discover_object_queries(max_queries)
    if not queries:
        print("[pilot] no objects discovered — scan will have no 3D boxes")
        return 0
    print(f"[pilot] detecting {len(queries)} discovered object(s): {queries}")
    detected = 0
    for q in queries:
        try:
            # add_query_progressive is a generator that scans every submap for the query;
            # exhaust it so detection completes (it accumulates into accumulated_detections).
            for _partial in slam_processor.add_query_progressive(q):
                pass
            detected += 1
        except Exception as e:
            print(f"[pilot] detection failed for {q!r} (skipping): {e}")
    return detected


def finalize_scan_blocking(wait_for_drain: bool = True) -> None:
    """Synchronously finalize the current scan: drain the queue, flush the last partial
    submap, reconcile detections, then build + persist the scene report **once**.

    The same stop→finalize→report sequence as ``_run_end_of_scan_report`` but with no
    asyncio loop or socket sids — used by the offline multi-video pilot harness, which
    feeds every video into one ``StreamingSLAM`` session and then calls this exactly once
    at the end (so the persist + splat export happen for one accumulated world frame, not
    per video). Best-effort throughout; ``_build_and_emit_scene_report`` runs the persist
    (with pilot tags / splat) via ``_persist_scene_report``."""
    if slam_processor is None:
        return
    if wait_for_drain:
        _wait_for_frame_queue_drain()
    # Make sure no submap pass is still running before we flush-stop + finalize. The live
    # path's stop() uses a 2s join; here (offline batch) we wait generously so a long VGGT
    # pass fully finishes and the final submap/cloud reflects every fed frame.
    if not slam_processor.wait_until_idle():
        print("[pilot] WARNING: processing not idle before finalize; finalizing anyway")
    try:
        slam_processor.stop(flush=True, join_timeout=120.0)
    except Exception as e:
        print(f"stop() before scene report failed: {e}")
    # Offline agentic object detection (pilot only): now that every submap is built, auto-
    # discover the objects in the scan and locate them in 3D, so facts.objects (→ the viewer's
    # bounding boxes) is populated. The live flow does this interactively via the spatial agent;
    # this is the batch equivalent. Best-effort — a failure leaves the scan boxless, not broken.
    if _pilot_detect_objects:
        try:
            _run_offline_object_detection(PILOT_DETECT_MAX_QUERIES)
        except Exception as e:
            print(f"[pilot] offline object detection pass failed (degrading): {e}")
    try:
        slam_processor.finalize_detection_state()
    except Exception as e:
        print(f"Final detection reconciliation error: {e}")
    _build_and_emit_scene_report(None)


def reset_for_next_scan() -> None:
    """Reset SLAM + scene-report state between scans WITHOUT reloading the model — the same
    soft-reset path as the ``/reset`` route (minus the sockets/agent fan-out), exposed for
    the per-scene pilot harness. Lets one worker reconstruct several non-overlapping clips as
    SEPARATE scans (one persisted scan per clip) while paying the VGGT model load only once.
    ``slam_processor.soft_reset()`` calls ``solver.reset()`` so the world frame is wiped
    between scenes (no cross-clip bleed)."""
    global _current_scan_id
    _clear_queues()
    if slam_processor is not None:
        # Wait for any in-flight submap pass to finish before wiping the solver, so
        # soft_reset()/solver.reset() can't race a running VGGT pass on shared state.
        if not slam_processor.wait_until_idle():
            print("[pilot] WARNING: processing not idle before reset; resetting anyway")
        slam_processor.soft_reset()
        slam_processor.set_detection_queries([])
    _clear_scene_report()
    _current_scan_id = None


def _trigger_scene_report_threadsafe(wait_for_drain: bool = False) -> None:
    """Schedule end-of-scan report generation from any thread (a Flask route or the
    VideoFeeder thread). Debounced so a near-simultaneous demo-stop + end-of-video
    does not build the report twice."""
    global _scene_report_last_trigger
    loop = _event_loop
    if loop is None or loop.is_closed():
        return
    with _scene_report_lock:
        now = time.time()
        if now - _scene_report_last_trigger < 5.0:
            return
        _scene_report_last_trigger = now
    asyncio.run_coroutine_threadsafe(_run_end_of_scan_report(wait_for_drain=wait_for_drain), loop)


def _invalidate_pending_salvage() -> None:
    """Void any scheduled abandoned-scan salvage (thread-safe): a reconnect continues
    the scan and a /reset starts a new one — either way persisting the old map now
    would be wrong."""
    global _scan_salvage_generation
    with _scan_salvage_lock:
        _scan_salvage_generation += 1


async def _salvage_abandoned_scan(generation: int) -> None:
    """The last client vanished mid-scan without stop_slam — dead transport, killed
    app, locked phone. The disconnect cleanup path never builds or persists, so
    without this the whole map is silently dropped at worker shutdown (measured
    2026-08-13: three consecutive native scans lost this way — detection emit storms
    starved socket pings, the client timed out, stop_slam never arrived). After a
    grace window for reconnects, finalize + persist exactly as a clean stop would.
    The partial-batch tail was already dropped by the plain stop() on disconnect;
    every completed submap survives."""
    global _scan_salvage_task
    try:
        await asyncio.sleep(SCAN_SALVAGE_GRACE_S)
        with _scan_salvage_lock:
            if generation != _scan_salvage_generation:
                return
        with _sids_lock:
            if _connected_sids:
                return
        if slam_processor is None or _scene_report_finalized:
            return
        if slam_processor.solver.map.get_num_submaps() == 0:
            return
        print(f"[salvage] no reconnect within {SCAN_SALVAGE_GRACE_S:.0f}s of last "
              "disconnect — persisting the abandoned scan")
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: finalize_scan_blocking(wait_for_drain=False))
    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"[salvage] abandoned-scan persist failed: {e}")
    finally:
        _scan_salvage_task = None


def _latest_submap_id() -> Optional[int]:
    """Newest non-loop-closure submap id, used to bias the live report's keyframes."""
    if slam_processor is None:
        return None
    try:
        return int(slam_processor.solver.map.get_largest_key(ignore_loop_closure_submaps=True))
    except Exception:
        return None


def _build_and_emit_progressive_report() -> None:
    """Refresh + broadcast the in-scan progressive report (Phase 3).

    Runs on a worker thread, dispatched (throttled) from the result stream. Reads
    only the reconstruction, so it works with the agent disabled. Best-effort:
    failures are logged and skipped — the final report still covers the whole scan."""
    global _progressive_report, _progressive_report_inflight
    if slam_processor is None:
        return
    try:
        with _progressive_report_lock:
            if _scene_report_finalized:
                return
            previous = _progressive_report
        with slam_processor._detection_lock:
            detections = list(slam_processor.accumulated_detections)
        scene_center = slam_processor.latest_scene_center
        solver = slam_processor.solver
        facts = _scene_feature_extractor.extract(solver, detections, scene_center)
        if facts.metrics.num_submaps == 0:
            return
        latest_id = _latest_submap_id()
        prefer = [latest_id] if latest_id is not None else None
        report = _get_scene_report_builder().build_incremental(
            solver,
            facts,
            scene_center,
            previous=previous,
            max_keyframes=SCENE_REPORT_PROGRESSIVE_KEYFRAMES,
            prefer_submap_ids=prefer,
        )
        payload = report.model_dump()
        # Decide + emit under the lock so a final report that lands mid-build can't be
        # clobbered by a late progressive update (final sets _scene_report_finalized).
        with _progressive_report_lock:
            if _scene_report_finalized:
                return
            _progressive_report = report
            _broadcast_scene_report(payload, event="scene_report_update")
    except Exception as e:
        print(f"Progressive scene report error: {e}")
    finally:
        with _progressive_report_lock:
            _progressive_report_inflight = False


def _maybe_dispatch_progressive_report(result: dict[str, Any]) -> None:
    """Throttled trigger for the in-scan progressive report, called from the result
    stream. Cheap and non-blocking: the throttle check runs inline; the heavy build is
    dispatched to a worker thread. Caps LLM calls at one per ``EVERY_N`` new submaps
    *and* no more often than ``MIN_INTERVAL_S``."""
    global _progressive_report_last_submaps, _progressive_report_last_ts
    global _progressive_report_inflight
    if not SCENE_REPORT_PROGRESSIVE or slam_processor is None:
        return
    try:
        num_submaps = int(result.get("num_submaps", 0))
    except (TypeError, ValueError):
        return
    if num_submaps < 1:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    now = time.time()
    with _progressive_report_lock:
        if _scene_report_finalized or _progressive_report_inflight:
            return
        if num_submaps - _progressive_report_last_submaps < SCENE_REPORT_PROGRESSIVE_EVERY_N:
            return
        if now - _progressive_report_last_ts < SCENE_REPORT_PROGRESSIVE_MIN_INTERVAL_S:
            return
        _progressive_report_last_submaps = num_submaps
        _progressive_report_last_ts = now
        _progressive_report_inflight = True
    try:
        loop.run_in_executor(_agent_executor, _build_and_emit_progressive_report)
    except Exception:
        with _progressive_report_lock:
            _progressive_report_inflight = False


def _build_agent_state_payload(sid: str) -> dict[str, Any]:
    with _sessions_lock:
        state = _sessions.get(sid)

    if state is None:
        return {"enabled": False, "active_queries": []}

    active_queries = [] if state.reconstruction_only else sorted(state.manual_queries | state.agent_queries)

    if state.agent is None:
        return {
            "enabled": False,
            "scene_description": "",
            "room_type": "unknown",
            "missions": [],
            "active_queries": active_queries,
            "discovered_objects": [],
            "current_goal": None,
            "submaps_processed": 0,
            "coverage_estimate": 0.0,
            "health": "disabled",
            "degraded_mode": False,
            "active_tasks": [],
            "pending_jobs": [],
            "running_jobs": [],
            "last_job_errors": [],
            "orchestrator_busy": False,
        }

    payload = state.agent.get_state()
    if state.reconstruction_only:
        payload["enabled"] = False
        payload["reconstruction_only"] = True
    payload["active_queries"] = active_queries
    return payload


def _collect_global_query_union() -> list[str]:
    with _sessions_lock:
        query_set: set[str] = set()
        for state in _sessions.values():
            if state.reconstruction_only:
                continue
            query_set.update(state.manual_queries)
            query_set.update(state.agent_queries)
    return sorted(query_set)


def _agent_persist_query(sid: str, query: str):
    """Called from agent tools when they scan for a new object.

    Adds the query to manual_queries so it persists as a chip + bounding boxes
    until the user explicitly removes it via the UI. Also spawns a detection worker
    if one isn't already running for this query.
    """
    query = query.strip().lower()
    if not query or _event_loop is None or _event_loop.is_closed():
        return
    with _sessions_lock:
        state = _sessions.get(sid)
        if state is None:
            return
        if state.reconstruction_only:
            print(f"[agent_persist_query] ignored in reconstruction-only mode: {query}")
            return
        if query in state.manual_queries:
            return  # Already visible, nothing to do
        state.manual_queries.add(query)

    _emit_to_sid_threadsafe(sid, "agent_ui_command", {
        "id": str(uuid.uuid4())[:12],
        "name": "add_detection_query",
        "args": {"query": query},
        "mission_id": None,
        "ttl_ms": None,
        "timestamp": time.time(),
    })
    asyncio.run_coroutine_threadsafe(
        _spawn_query_worker_if_needed(query, sid),
        _event_loop,
    )


async def _spawn_query_worker_if_needed(query: str, sid: str):
    existing = _query_tasks.get((sid, query))
    if existing and not existing.done():
        return
    task = asyncio.create_task(_run_single_query_detection(query, sid))
    _query_tasks[(sid, query)] = task


async def _clear_session_detection_state(sid: str):
    if slam_processor is None:
        return

    with _sessions_lock:
        state = _sessions.get(sid)
        if state is None:
            queries: set[str] = set()
        else:
            queries = set(state.manual_queries | state.agent_queries)
            state.manual_queries.clear()
            state.agent_queries.clear()

    for q in queries:
        task = _query_tasks.pop((sid, q), None)
        if task and not task.done():
            task.cancel()
        slam_processor.remove_query(q)

    await sio.emit(
        "detection_partial",
        {
            "detections": [],
            "active_queries": [],
            "is_final": True,
        },
        to=sid,
    )


def _on_agent_queries_changed(sid: str, queries: list[str]):
    """Called from agent threads; spawns/cancels per-query detection workers."""
    with _sessions_lock:
        state = _sessions.get(sid)
        if state is None:
            return
        if state.reconstruction_only:
            print(f"[agent_query_diff] ignored in reconstruction-only mode: {queries}")
            return
        old = set(state.agent_queries)
        state.agent_queries = set(_normalize_query_list(queries))
        new = set(state.agent_queries)

    if _event_loop is None or _event_loop.is_closed():
        return

    removed = old - new
    added = new - old

    if not removed and not added:
        return

    asyncio.run_coroutine_threadsafe(
        _apply_agent_query_diff(sid, added, removed),
        _event_loop,
    )


async def _apply_agent_query_diff(sid: str, added: set[str], removed: set[str]):
    """Cancel tasks for removed agent queries and spawn tasks for new ones."""
    if slam_processor is None:
        return
    with _sessions_lock:
        state = _sessions.get(sid)
        if state is not None and state.reconstruction_only:
            print(f"[agent_query_diff] skipped in reconstruction-only mode sid={sid[:8]}")
            return

    print(
        f"[agent_query_diff] sid={sid[:8]} "
        f"+{sorted(added)} -{sorted(removed)} "
        f"slam_active={slam_processor.active_queries}"
    )

    for q in removed:
        task = _query_tasks.pop((sid, q), None)
        if task and not task.done():
            task.cancel()
            print(f"  [agent_query_diff] cancelled task for '{q}'")

        with _sessions_lock:
            state = _sessions.get(sid)
            pinned = state is not None and q in state.manual_queries

        if pinned:
            # Query was added by an agent tool and is visible in the UI — keep it
            print(f"  [agent_query_diff] keeping '{q}' — pinned in manual_queries")
        else:
            slam_processor.remove_query(q)
            print(f"  [agent_query_diff] removed '{q}' — slam active_queries now: {slam_processor.active_queries}")
            await sio.emit(
                "agent_ui_command",
                {
                    "id": str(uuid.uuid4())[:12],
                    "name": "remove_detection_query",
                    "args": {"query": q},
                    "mission_id": None,
                    "ttl_ms": None,
                    "timestamp": time.time(),
                },
                to=sid,
            )

    if removed:
        active = _session_active_queries(sid)
        accumulated = list(slam_processor.accumulated_detections)
        filtered = _filter_detections_by_queries(accumulated, active)
        print(f"  [agent_query_diff] immediate update: active={active} filtered={len(filtered)}")
        await sio.emit(
            "detection_partial",
            {
                "detections": _wire_detections(filtered),
                "active_queries": active,
                "is_final": True,
            },
            to=sid,
        )

    for q in sorted(added):
        print(f"  [agent_query_diff] spawning task for '{q}'")
        # Tell the frontend to add a chip for this agent query immediately
        await sio.emit(
            "agent_ui_command",
            {
                "id": str(uuid.uuid4())[:12],
                "name": "add_detection_query",
                "args": {"query": q},
                "mission_id": None,
                "ttl_ms": None,
                "timestamp": time.time(),
            },
            to=sid,
        )
        task = asyncio.create_task(_run_single_query_detection(q, sid))
        _query_tasks[(sid, q)] = task


def _create_session_agent(sid: str):
    if not _openrouter_api_key:
        return None

    from server.spatial_agent import SpatialAgent

    return SpatialAgent(
        streaming_slam=slam_processor,
        emit_fn=lambda event, data: _emit_to_sid_threadsafe(sid, event, data),
        openrouter_api_key=_openrouter_api_key,
        session_id=sid,
        on_queries_changed=_on_agent_queries_changed,
        on_query_persisted=lambda q: _agent_persist_query(sid, q),
    )


def _ensure_session(
    sid: str,
    user_id: Optional[str] = None,
    claims: Optional[dict[str, Any]] = None,
) -> SessionState:
    with _sessions_lock:
        state = _sessions.get(sid)
        if state is not None:
            return state

        state = SessionState(
            sid=sid,
            user_id=user_id or sid,
            claims=claims or {},
        )
        if slam_processor is not None:
            state.agent = _create_session_agent(sid)
        _sessions[sid] = state
        return state


async def _run_single_query_detection(query: str, trigger_sid: str):
    """Run detection for a single query, streaming partials to the triggering session."""
    if slam_processor is None:
        return

    print(f"[query_worker:{query}] started sid={trigger_sid[:8]}")

    loop = asyncio.get_event_loop()
    partial_q: asyncio.Queue = asyncio.Queue()

    def run():
        try:
            for partial in slam_processor.add_query_progressive(query):
                loop.call_soon_threadsafe(partial_q.put_nowait, partial)
        except Exception as e:
            loop.call_soon_threadsafe(
                partial_q.put_nowait,
                {"detections": [], "is_final": True, "error": str(e)},
            )

    _gpu_executor.submit(run)

    partial_count = 0
    try:
        while True:
            partial = await partial_q.get()
            partial_count += 1
            active = _session_active_queries(trigger_sid)
            all_dets = partial.get("detections", [])
            filtered = _filter_detections_by_queries(all_dets, active)
            is_final = bool(partial.get("is_final", False))

            print(
                f"[query_worker:{query}] partial #{partial_count} "
                f"total_dets={len(all_dets)} filtered={len(filtered)} "
                f"active={active} is_final={is_final}"
                + (f" error={partial['error']}" if partial.get("error") else "")
            )

            # Pace intermediates: concurrent per-query workers all emit the same
            # accumulated superset, so a burst of N workers × M submaps collapses to
            # a few progressive refreshes per second. Every worker's final still goes
            # out (it may carry that query's error field).
            emit_now = _is_sid_connected(trigger_sid)
            if emit_now and not is_final:
                last = _detection_partial_last_emit.get(trigger_sid, 0.0)
                emit_now = (time.monotonic() - last) >= DETECTION_PARTIAL_MIN_INTERVAL_S
            if emit_now:
                _detection_partial_last_emit[trigger_sid] = time.monotonic()
                await sio.emit(
                    "detection_partial",
                    {
                        "detections": _wire_detections(filtered),
                        "active_queries": active,
                        "is_final": is_final,
                    },
                    to=trigger_sid,
                )
            if is_final:
                break
    except asyncio.CancelledError:
        print(f"[query_worker:{query}] cancelled after {partial_count} partials")
        raise

    print(f"[query_worker:{query}] done — {partial_count} partials emitted")
    _query_tasks.pop((trigger_sid, query), None)


async def _refresh_global_detection_queries(trigger_sid: Optional[str], emit_progress: bool):
    """Recompute shared detection cache from session query union.

    Compatibility behavior:
      - Emits detection_partial only to the triggering session (when requested).
      - Global detector still runs on union of all session queries.
    """
    if slam_processor is None:
        return

    lock = _ensure_query_lock()
    async with lock:
        global_queries = _collect_global_query_union()

        loop = asyncio.get_event_loop()
        partial_q: asyncio.Queue = asyncio.Queue()

        def run_gen():
            try:
                for partial in slam_processor.run_detection_progressive(global_queries):
                    loop.call_soon_threadsafe(partial_q.put_nowait, partial)
            except Exception as e:
                loop.call_soon_threadsafe(
                    partial_q.put_nowait,
                    {"detections": [], "is_final": True, "error": str(e)},
                )

        _gpu_executor.submit(run_gen)

        while True:
            partial = await partial_q.get()
            if emit_progress and trigger_sid is not None:
                active = _session_active_queries(trigger_sid)
                filtered = _filter_detections_by_queries(partial.get("detections", []), active)
                await sio.emit(
                    "detection_partial",
                    {
                        "detections": _wire_detections(filtered),
                        "active_queries": active,
                        "is_final": bool(partial.get("is_final", False)),
                    },
                    to=trigger_sid,
                )
            if partial.get("is_final", False):
                break

        if trigger_sid is not None:
            await sio.emit("agent_state", _build_agent_state_payload(trigger_sid), to=trigger_sid)


# ------------------------------
# Video File Feeder (testing mode)
# ------------------------------
class VideoFeeder:
    """Reads frames from a video file and pushes them into frame_queue."""

    def __init__(self, video_path, fast=False, target_fps=2.0, on_complete=None, realtime=True):
        self.video_path = video_path
        self.fast = fast
        self.target_fps = target_fps
        self.on_complete = on_complete
        # realtime=True (default): a live camera or a viewer watching the feed in real
        # time is on the other end, so a lagging consumer must never backpressure the
        # producer — drop the frame instead (existing behavior, unchanged).
        # realtime=False: offline/batch feeding (the pilot harness — see
        # reconstruct_pilot.py) with no live camera/viewer to protect. There, a dropped
        # frame is a silent, nondeterministic quality hit on long captures, so block for
        # room instead of dropping. Independent of `fast` (which only controls pacing):
        # `--realtime` on the pilot CLI paces the feed but is still offline batch work.
        self.realtime = realtime
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._feed_loop, daemon=True)
        self._thread.start()
        print(
            f"VideoFeeder started: {self.video_path} "
            f"(target_fps={self.target_fps}, fast={self.fast}, realtime={self.realtime})"
        )

    def stop(self):
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _feed_loop(self):
        if _is_lfs_pointer(self.video_path):
            print(f"Failed to open video: {self.video_path}"
                  " (file is a Git LFS pointer, not actual video content)")
            return
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Failed to open video: {self.video_path}")
            return

        rotation = cap.get(cv2.CAP_PROP_ORIENTATION_META)
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        skip = max(1, int(round(video_fps / self.target_fps)))
        effective_fps = video_fps / skip
        delay = 0.0 if self.fast else 1.0 / effective_fps

        frames_to_feed = total_frames // skip
        print(f"Video: {total_frames} frames @ {video_fps:.1f} FPS")
        print(
            f"  Feeding every {skip} frame(s) -> ~{effective_fps:.1f} effective FPS "
            f"(~{frames_to_feed} frames, delay={delay * 1000:.0f}ms)"
        )

        raw_idx = 0
        fed_count = 0
        t0 = time.time()

        while not self._stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break

            raw_idx += 1
            if (raw_idx - 1) % skip != 0:
                continue

            if rotation:
                frame = _apply_video_rotation(frame, rotation)

            ok, jpeg_buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                continue
            b64 = base64.b64encode(jpeg_buf.tobytes()).decode("ascii")
            data = {"image": b64, "timestamp": time.time()}

            if self.realtime:
                try:
                    frame_queue.put(data, timeout=10)
                except queue.Full:
                    print(f"frame_queue full, dropping frame {fed_count}")
                    continue
            else:
                # Offline/batch mode: no live camera/viewer to backpressure, so block
                # for room instead of losing a keyframe. Retry in short bounded waits
                # (rather than one long blocking put) so stop() can still interrupt us
                # promptly; the consumer (StreamingSLAM.process_loop, its own thread)
                # keeps draining the queue the whole time, so this cannot deadlock.
                wait_start = time.time()
                logged_slow = False
                put_ok = False
                while not self._stop_event.is_set():
                    try:
                        frame_queue.put(data, timeout=1.0)
                        put_ok = True
                        break
                    except queue.Full:
                        waited = time.time() - wait_start
                        if waited > 5.0 and not logged_slow:
                            print(
                                f"frame_queue full, blocking to feed frame {fed_count} "
                                f"({waited:.1f}s and counting — SLAM loop is behind)"
                            )
                            logged_slow = True
                if not put_ok:
                    # stop() fired while we were blocked waiting for room.
                    break

            if fed_count == 0 and slam_processor is not None and not slam_processor.is_running:
                print("Auto-starting SLAM processing (video mode)...")
                _begin_new_scan()
                slam_processor.start()

            fed_count += 1
            if fed_count % 50 == 0:
                elapsed = time.time() - t0
                print(f"Fed {fed_count}/{frames_to_feed} frames ({elapsed:.1f}s elapsed)")

            if delay > 0:
                time.sleep(delay)

        cap.release()
        elapsed = time.time() - t0
        print(f"VideoFeeder finished: {fed_count} frames fed in {elapsed:.1f}s")
        # On natural completion (not an explicit stop), generate the scene report.
        if not self._stop_event.is_set() and self.on_complete is not None:
            try:
                self.on_complete()
            except Exception as e:
                print(f"VideoFeeder on_complete error: {e}")


def _get_demo_video_dir():
    return os.environ.get(
        'DEMO_VIDEO_DIR',
        os.path.join(os.path.dirname(__file__), 'demo_videos'),
    )


def _clear_queues():
    while not frame_queue.empty():
        try:
            frame_queue.get_nowait()
        except Exception:
            break
    while not result_queue.empty():
        try:
            result_queue.get_nowait()
        except Exception:
            break


def _stop_demo_feeder():
    global _demo_video_feeder, _demo_active_video_id, _demo_active_video_path
    global _demo_started_at, _demo_target_fps
    with _demo_lock:
        if _demo_video_feeder is not None:
            print("Stopping active demo feeder...")
            _demo_video_feeder.stop()
            _demo_video_feeder = None
            _demo_active_video_id = None
            _demo_active_video_path = None
            _demo_started_at = None
            _demo_target_fps = None


def _is_lfs_pointer(path):
    """Return True if the file is a Git LFS pointer (not actual video content)."""
    try:
        if os.path.getsize(path) > 1024:
            return False
        with open(path, "r") as f:
            return "git-lfs" in f.read(50)
    except Exception:
        return False


def _is_supported_video_file(path):
    ext = os.path.splitext(path)[1].lower()
    return ext in _DEMO_VIDEO_EXTENSIONS


def _safe_demo_path(video_id):
    demo_dir = os.path.abspath(_get_demo_video_dir())
    if not video_id:
        raise ValueError('video_id is required')

    normalized_id = os.path.normpath(str(video_id).replace('\\', '/')).lstrip('/')
    full_path = os.path.abspath(os.path.join(demo_dir, normalized_id))

    if os.path.commonpath([demo_dir, full_path]) != demo_dir:
        raise ValueError('Invalid video_id path')
    if not os.path.isfile(full_path):
        requested_id = normalized_id.lower()
        for candidate_path in _collect_demo_video_files():
            candidate_id = os.path.relpath(candidate_path, demo_dir).replace(os.sep, '/')
            if candidate_id.lower() == requested_id:
                full_path = os.path.abspath(candidate_path)
                break
        else:
            raise FileNotFoundError(f'Video not found: {video_id}')
    if not _is_supported_video_file(full_path):
        raise ValueError('Unsupported video format')
    return full_path


def _collect_demo_video_files():
    demo_dir = os.path.abspath(_get_demo_video_dir())
    if not os.path.isdir(demo_dir):
        return []
    candidates = []
    for root, _, files in os.walk(demo_dir):
        for file_name in files:
            full_path = os.path.join(root, file_name)
            if not _is_supported_video_file(full_path):
                continue
            if _is_lfs_pointer(full_path):
                print(f"Skipping Git LFS pointer: {full_path}")
                continue
            candidates.append(full_path)
    candidates.sort()
    return candidates


def _build_thumbnail_data_url(video_path):
    cache_key = (video_path, os.path.getmtime(video_path))
    cached = _demo_thumbnail_cache.get(cache_key)
    if cached:
        return cached

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    rotation = cap.get(cv2.CAP_PROP_ORIENTATION_META)
    thumbnail = None
    for _ in range(10):
        ok, frame = cap.read()
        if not ok:
            break
        if frame is None or frame.size == 0:
            continue
        if rotation:
            frame = _apply_video_rotation(frame, rotation)
        h, w = frame.shape[:2]
        if w > 320:
            scale = 320.0 / float(w)
            frame = cv2.resize(frame, (320, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        ok, jpeg_buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            b64 = base64.b64encode(jpeg_buf.tobytes()).decode('ascii')
            thumbnail = f'data:image/jpeg;base64,{b64}'
            break
    cap.release()

    if thumbnail:
        _demo_thumbnail_cache[cache_key] = thumbnail
    return thumbnail


def _build_demo_catalog(force_refresh=False):
    global _demo_catalog_cache
    if _demo_catalog_cache is not None and not force_refresh:
        return _demo_catalog_cache

    demo_dir = os.path.abspath(_get_demo_video_dir())
    videos = []
    for video_path in _collect_demo_video_files():
        rel_path = os.path.relpath(video_path, demo_dir).replace(os.sep, '/')
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()
        duration_sec = float(frame_count / fps) if fps > 0 else 0.0
        videos.append({
            'video_id': rel_path,
            'name': os.path.splitext(os.path.basename(video_path))[0],
            'filename': os.path.basename(video_path),
            'mime_type': mimetypes.guess_type(video_path)[0] or 'video/mp4',
            'thumbnail': _build_thumbnail_data_url(video_path),
            'fps': round(float(fps), 3) if fps > 0 else None,
            'duration_sec': round(duration_sec, 2) if duration_sec > 0 else None,
            'width': width or None,
            'height': height or None,
        })
    _demo_catalog_cache = videos
    return videos


# ------------------------------
# Flask Routes
# ------------------------------
PROTECTED_HTTP_PATHS = frozenset({"/health", "/reset", "/session", "/session/status"})
PROTECTED_HTTP_PREFIXES = ("/api/",)


def _requires_http_auth(path: str) -> bool:
    return path in PROTECTED_HTTP_PATHS or any(
        path.startswith(prefix) for prefix in PROTECTED_HTTP_PREFIXES
    )


def _request_origin() -> str:
    """The public origin (scheme://host) the request arrived on, honoring the proxy
    headers Modal sets, so a minted embed URL points at the real broker host."""
    proto = (request.headers.get("X-Forwarded-Proto", "").split(",")[0].strip()
             or request.scheme)
    host = (request.headers.get("X-Forwarded-Host", "").split(",")[0].strip()
            or request.host)
    return f"{proto}://{host}"


def _try_share_token_auth() -> bool:
    """If the request carries a valid read-only share token, authorize it for the allowed
    scene-read endpoints (scoped to the token's scan_id) and return ``True`` so the caller
    lets the request proceed. Returns ``False`` when there is no share token (fall through
    to the normal Clerk/session auth). Raises ``AuthError`` when the share token is valid
    but the route or scan is not permitted (so a share token can read its one scan and
    nothing else — not the list, not DELETE, not another scan, not the mint route)."""
    query_token = request.args.get(SHARE_QUERY_PARAM, "").strip()
    token = query_token or _extract_bearer_token()
    if not token:
        return False
    try:
        claims = _verify_share_token(token)
    except AuthError:
        # An explicit ?share_token= that fails verification is a dead/forged share link —
        # reject it (401) rather than silently falling back to cookie/session auth (which
        # would let an expired share URL keep working for a logged-in owner, masking expiry).
        # A Bearer that isn't a share token is ambiguous (could be Clerk/session) → fall through.
        if query_token:
            raise AuthError("invalid_token", 401)
        return False
    endpoint = request.url_rule.endpoint if request.url_rule else None
    if endpoint not in SHARE_TOKEN_ALLOWED_ENDPOINTS:
        raise AuthError("forbidden", 403)
    path_scan_id = (request.view_args or {}).get("scan_id")
    if not path_scan_id or str(path_scan_id) != str(claims.get("scan_id")):
        raise AuthError("forbidden", 403)
    # The derived route serves EVERY derived artifact; a share token gets the LOD tree only.
    if endpoint == "get_scene_derived_route" and not _share_token_derived_key_allowed(
        (request.view_args or {}).get("derived_key")
    ):
        raise AuthError("forbidden", 403)
    # Authorized. Carry the OWNER's id so the user-scoped store lookups resolve to the
    # owner's scene; mark it a share grant (read-only) for any downstream check.
    request.environ[AUTH_CLAIMS_ENV_KEY] = {
        "sub": str(claims.get("sub")),
        "scope": SHARE_TOKEN_SCOPE,
        "scan_id": str(claims.get("scan_id")),
        "share": True,
        "iat": claims.get("iat"),
        "exp": claims.get("exp"),
        "access_id": access_id_for_token(token),
    }
    request.environ[SHARE_TOKEN_ENV_KEY] = token
    # Measurement: the Q&A route logs its own event (with the question); every other
    # share-authorised read is logged here.
    if endpoint != "scene_qa_route":
        _record_share_access(claims, token)
    _note_worker_activity()
    return True


def _try_project_token_auth() -> bool:
    """If the request carries a valid building (project) token, authorize it for the ONE
    manifest endpoint — and only when the query ``client``+``project`` match the token —
    returning ``True`` so the caller lets the request proceed. Returns ``False`` when there
    is no project token (fall through to the share-token / Clerk / session path). Raises
    ``AuthError`` (403) when the token is a valid project token but is presented on any other
    endpoint, or with a mismatched client/project — so a project token can ONLY read its own
    building manifest and nothing else (not a scene, not the list, not another project)."""
    query_token = request.args.get(PROJECT_QUERY_PARAM, "").strip()
    token = query_token or _extract_bearer_token()
    if not token:
        return False
    try:
        claims = _verify_project_token(token)
    except AuthError:
        # As with share tokens, an explicit ?project_token= that fails verification is
        # rejected (not fallen through). A Bearer is ambiguous → fall through.
        if query_token:
            raise AuthError("invalid_token", 401)
        return False
    endpoint = request.url_rule.endpoint if request.url_rule else None
    if endpoint != PROJECT_TOKEN_ENDPOINT:
        raise AuthError("forbidden", 403)
    q_client = str(request.args.get("client", "")).strip()
    q_project = str(request.args.get("project", "")).strip()
    if q_client != str(claims.get("client")) or q_project != str(claims.get("project")):
        raise AuthError("forbidden", 403)
    # Authorized. Carry the OWNER's id (user-scoped store lookups resolve to the owner's
    # scenes) + the project scope so the manifest route can confirm it was project-authed.
    request.environ[AUTH_CLAIMS_ENV_KEY] = {
        "sub": str(claims.get("sub")),
        "scope": PROJECT_TOKEN_SCOPE,
        "client": str(claims.get("client")),
        "project": str(claims.get("project")),
        "project_token": True,
        # Carried so the manifest caps each minted per-scene token to the project token's
        # remaining lifetime (a scene link can't outlive the building link that spawned it).
        "exp": claims.get("exp"),
    }
    _note_worker_activity()
    return True


@app.before_request
def require_modal_auth():
    if request.method == "OPTIONS":
        return None
    if request.path == "/auth/session":
        return None
    if not _requires_http_auth(request.path):
        return None
    try:
        # Read-only share token first (single-scan capability, no Clerk login). Falls
        # through to the normal path when the request carries no share token.
        if _try_share_token_auth():
            return None
        # Building (project) token next: authorizes ONLY the manifest route for its own
        # client+project; rejects (403) on any other endpoint; falls through when absent.
        if _try_project_token_auth():
            return None
        claims = _verify_http_token()
        _require_worker_owner(claims)
        _note_worker_activity()
        request.environ[AUTH_CLAIMS_ENV_KEY] = claims
    except AuthError as error:
        return _auth_response(error)
    return None


@app.route("/auth/session", methods=["POST"])
def create_auth_session():
    token = _extract_bearer_token()
    if not token:
        return _auth_response(AuthError("missing_token"))

    try:
        claims = _verify_clerk_token(token)
        _require_scan_access(claims)
        _require_worker_owner(claims)
        _note_worker_activity()
    except AuthError as error:
        return _auth_response(error)

    response = app.response_class(status=204)
    response.set_cookie(
        AUTH_COOKIE_NAME,
        token,
        expires=_jwt_expiry_datetime(claims),
        httponly=True,
        secure=True,
        samesite="None",
        path="/",
    )
    return response


@app.route("/auth/qr-token", methods=["GET"])
def get_qr_token():
    """Return the current Modal access token so a logged-in laptop can mint a phone QR."""
    token = _extract_http_token()
    if not token:
        return _auth_response(AuthError("missing_token"))

    try:
        claims = _verify_clerk_token(token)
        _require_scan_access(claims)
        _require_worker_owner(claims)
        _note_worker_activity()
    except AuthError as error:
        return _auth_response(error)

    return jsonify({"token": token, "expires_at": int(claims["exp"])})


@app.route("/session", methods=["POST"])
def allocate_gpu_session():
    if _gpu_session_broker is None:
        return jsonify({"error": "session_allocator_unavailable"}), 503
    claims = request.environ.get(AUTH_CLAIMS_ENV_KEY, {})
    user_id = str(claims.get("sub", "")).strip()
    if not user_id:
        return _auth_response(AuthError("invalid_token"))
    # Cheap pre-check before we spawn a GPU. The authoritative gate is the hold
    # the worker opens and tops up per minute — this only avoids starting an
    # A100 for someone who plainly cannot pay for the first block. Never refuses
    # in shadow mode.
    from server.billing import meter as _meter
    from server.billing import prices as _prices

    if _meter.enforcing():
        _meter.ensure_account(user_id)
        _balance = _meter.balance(user_id)
        if _balance is not None and _balance < _prices.CREDITS_SESSION_OPENING_BLOCK:
            return jsonify({"error": "no_credits", "credits": _balance}), 402

    try:
        allocation = _gpu_session_broker.allocate(user_id)
    except SessionCapacityError:
        return jsonify({"error": "at_capacity"}), 503
    except SessionWorkerError:
        return jsonify({"error": "worker_unavailable"}), 503
    return jsonify(allocation), (200 if allocation.get("url") else 202)


@app.route("/session/status", methods=["GET"])
def gpu_session_status():
    if _gpu_session_broker is None:
        return jsonify({"error": "session_allocator_unavailable"}), 503
    claims = request.environ.get(AUTH_CLAIMS_ENV_KEY, {})
    allocation = _gpu_session_broker.status(str(claims.get("sub", "")).strip())
    if allocation is None:
        return jsonify({"status": "none"}), 404
    return jsonify(allocation), (200 if allocation.get("url") else 202)


@app.route("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "gpu": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        }
    )


@app.route("/reset", methods=["POST"])
def reset():
    """Soft reset: clear SLAM data, keep models loaded."""
    _invalidate_pending_salvage()
    _stop_demo_feeder()
    _clear_queues()
    if slam_processor is None:
        return jsonify({"status": "no_processor"}), 503

    while not frame_queue.empty():
        try:
            frame_queue.get_nowait()
        except Exception:
            break

    while not result_queue.empty():
        try:
            result_queue.get_nowait()
        except Exception:
            break

    # Cancel all per-query detection tasks
    for key in list(_query_tasks.keys()):
        task = _query_tasks.pop(key, None)
        if task and not task.done():
            task.cancel()

    agents_to_reset = []
    with _sessions_lock:
        _frame_rate_state.clear()
        for state in _sessions.values():
            state.manual_queries.clear()
            state.agent_queries.clear()
            if state.agent is not None:
                agents_to_reset.append(state.agent)

    for agent in agents_to_reset:
        try:
            agent.reset()
        except Exception:
            pass

    slam_processor.soft_reset()
    slam_processor.set_detection_queries([])
    global _current_scan_id
    _clear_scene_report()
    _current_scan_id = None  # no active scan after a reset

    if _event_loop is not None and not _event_loop.is_closed():
        asyncio.run_coroutine_threadsafe(sio.emit("slam_reset", {"status": "reset"}), _event_loop)

    return jsonify({"status": "reset_complete", "message": "SLAM and session state cleared"})


# @app.route("/api/export_ply", methods=["GET"])
# def export_ply():
#     """Generates the 3D map as a .ply file and triggers a direct file download."""
#     if slam_processor is None:
#         return jsonify({"status": "error", "message": "SLAM processor not initialized"}), 503
    
#     import os
#     from flask import send_file
    
#     tmp_path = "/tmp/slam_export.ply"
    
#     try:
#         # Ask Open3D to generate the PLY file over to /tmp memory
#         slam_processor.solver.map.write_points_to_file(slam_processor.solver.graph, tmp_path)
        
#         # Check if the map is empty and generated nothing
#         if not os.path.exists(tmp_path):
#             return jsonify({"status": "error", "message": "Point cloud map is empty or failed to save"}), 500
            
#         # Return it straight to the browser/client to download
#         return send_file(
#             tmp_path,
#             as_attachment=True,
#             download_name="reality_opened_map.ply",
#             mimetype="application/octet-stream"
#         )
#     except Exception as e:
#         return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/demo/videos', methods=['GET'])
def list_demo_videos():
    """Return local demo videos with base64 thumbnails for selection UI."""
    refresh = request.args.get('refresh', '').lower() in {'1', 'true', 'yes'}
    videos = _build_demo_catalog(force_refresh=refresh)
    return jsonify({
        'videos': videos,
        'demo_dir': _get_demo_video_dir(),
        'active_video_id': _demo_active_video_id,
    })


@app.route('/api/demo/status', methods=['GET'])
def demo_status():
    """Return current demo feeder state for sender-side preview sync."""
    return jsonify({
        'running': _demo_video_feeder is not None,
        'video_id': _demo_active_video_id,
        'started_at': _demo_started_at,
        'target_fps': _demo_target_fps,
    })


@app.route('/api/demo/video', methods=['GET'])
def get_demo_video():
    """Serve a selected demo video file for browser-side preview playback."""
    video_id = request.args.get('video_id')
    try:
        video_path = _safe_demo_path(video_id)
    except (ValueError, FileNotFoundError) as e:
        return jsonify({'error': str(e)}), 400
    return send_file(video_path, conditional=True)


@app.route('/api/demo/start', methods=['POST'])
def start_demo():
    """Start feeding a selected local demo video into frame_queue."""
    global _demo_video_feeder, _demo_active_video_id, _demo_active_video_path
    global _demo_started_at, _demo_target_fps
    data = request.get_json() or {}
    video_id = data.get('video_id')
    target_fps = float(data.get('fps', 10.0))

    if slam_processor is None:
        return jsonify({'error': 'SLAM processor not initialized'}), 503

    try:
        video_path = _safe_demo_path(video_id)
    except (ValueError, FileNotFoundError) as e:
        return jsonify({'error': str(e)}), 400

    target_fps = max(0.5, min(30.0, target_fps))

    with _demo_lock:
        if _demo_video_feeder is not None:
            _demo_video_feeder.stop()
            _demo_video_feeder = None

        _clear_queues()
        slam_processor.soft_reset()
        _begin_new_scan()  # fresh scan: drop any prior report + progressive state, mint scan_id

        _demo_video_feeder = VideoFeeder(
            video_path,
            fast=False,
            target_fps=target_fps,
            on_complete=lambda: _trigger_scene_report_threadsafe(wait_for_drain=True),
        )
        _demo_video_feeder.start()
        _demo_active_video_id = video_id
        _demo_active_video_path = video_path
        _demo_started_at = time.time()
        _demo_target_fps = target_fps

    return jsonify({
        'status': 'demo_started',
        'video_id': video_id,
        'fps': target_fps,
    })


@app.route('/api/demo/stop', methods=['POST'])
def stop_demo():
    """Stop active demo video feeder and generate the end-of-scan report."""
    _stop_demo_feeder()
    _trigger_scene_report_threadsafe(wait_for_drain=True)
    return jsonify({'status': 'demo_stopped'})
# Lazy-initialized OpenRouter client for the /api/plan route
_plan_client = None

_PLAN_STOPWORDS = {
    "i",
    "a",
    "an",
    "the",
    "in",
    "on",
    "at",
    "to",
    "for",
    "and",
    "or",
    "my",
    "me",
    "we",
    "is",
    "are",
    "was",
    "want",
    "need",
    "track",
    "find",
    "locate",
    "detect",
    "identify",
    "watch",
    "follow",
    "using",
    "with",
    "this",
    "that",
    "please",
    "help",
    "looking",
    "look",
    "search",
    "explore",
    "navigate",
    "moving",
    "move",
    "walk",
    "walking",
    "around",
    "inside",
    "outside",
    "toward",
    "towards",
    "front",
    "back",
    "left",
    "right",
    "from",
    "into",
    "through",
    "near",
    "scene",
    "room",
    "area",
    "object",
    "objects",
    "item",
    "items",
    "stuff",
    "things",
    "video",
    "camera",
    "demo",
    "live",
}

_GENERIC_OBJECT_WORDS = {
    "object",
    "objects",
    "item",
    "items",
    "thing",
    "things",
    "target",
    "targets",
}

_PLAN_OBJECT_HINTS: list[tuple[set[str], list[str]]] = [
    ({"office", "workspace", "desk"}, ["chair", "table", "laptop", "monitor", "keyboard", "bottle"]),
    ({"classroom", "school"}, ["chair", "desk", "backpack", "laptop", "whiteboard"]),
    ({"kitchen"}, ["refrigerator", "microwave", "sink", "cup", "bottle"]),
    ({"living", "livingroom", "sofa"}, ["couch", "coffee table", "tv", "lamp", "remote"]),
    ({"hallway", "corridor"}, ["door", "sign", "chair", "trash can"]),
    ({"garage", "workshop"}, ["toolbox", "drill", "ladder", "box"]),
    ({"crime", "evidence"}, ["phone", "wallet", "bag", "bottle"]),
    ({"disaster", "rescue"}, ["person", "backpack", "bottle", "helmet"]),
]

_PLAN_DEFAULT_OBJECTS = [
    "person",
    "chair",
    "table",
    "bottle",
    "backpack",
    "door",
]


def _clean_plan_object_name(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return ""
    cleaned = re.sub(r"[^a-z0-9\s\-]", " ", raw)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) > 36:
        cleaned = cleaned[:36].strip()
    return cleaned


def _extract_prompt_object_candidates(prompt: str) -> list[str]:
    text = str(prompt or "").lower()
    words = re.findall(r"[a-z][a-z0-9\-]{1,}", text)
    word_set = set(words)
    candidates: list[str] = []

    def add(name: str):
        obj = _clean_plan_object_name(name)
        if not obj or obj in candidates:
            return
        if obj in _PLAN_STOPWORDS or obj in _GENERIC_OBJECT_WORDS:
            return
        candidates.append(obj)

    # Domain hints improve relevance for common scene types.
    for keys, objs in _PLAN_OBJECT_HINTS:
        if keys & word_set:
            for obj in objs:
                add(obj)

    # Lightweight keyword extraction from prompt text.
    for word in words:
        if word in _PLAN_STOPWORDS or word in _GENERIC_OBJECT_WORDS:
            continue
        if len(word) < 3 or len(word) > 24:
            continue
        add(word)

    return candidates


def _finalize_plan_objects(
    llm_objects: Any,
    prompt: str,
    min_count: int = 5,
    max_count: int = 8,
) -> list[str]:
    merged: list[str] = []

    def add(value: Any):
        obj = _clean_plan_object_name(value)
        if not obj:
            return
        if obj in _PLAN_STOPWORDS or obj in _GENERIC_OBJECT_WORDS:
            return
        if obj not in merged:
            merged.append(obj)

    if isinstance(llm_objects, str):
        for part in re.split(r"[,;]\s*", llm_objects):
            add(part)
    elif isinstance(llm_objects, list):
        for item in llm_objects:
            add(item)

    for candidate in _extract_prompt_object_candidates(prompt):
        add(candidate)

    for fallback in _PLAN_DEFAULT_OBJECTS:
        add(fallback)

    if len(merged) < min_count:
        for fallback in ["phone", "bag", "laptop", "cup"]:
            add(fallback)
            if len(merged) >= min_count:
                break

    return merged[:max_count]

def _get_plan_client() -> OpenRouterClient:
    global _plan_client
    if _plan_client is None:
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set")

        _plan_client = OpenRouterClient(
            api_key=api_key,
            primary_model=os.environ.get(
                "PLAN_MODEL", "google/gemini-3-flash-preview"
            ),
            fallback_models=[
                os.environ.get("PLAN_FALLBACK_MODEL", "openai/gpt-4o-mini")
            ],
            timeout=15.0,
            usage_sink=named_tally("plan"),
            app_name="Real-Eyes Plan API",
            max_retries=2,
        )
    return _plan_client


def _plan_response_from_result(result):
    return jsonify(
        {
            "objects": result.get("objects", []),
            "waypoints": {
                "enabled": True,
                "justification": result.get(
                    "waypoints_justification", "Waypoints mark key locations."
                ),
            },
            "pathfinding": {
                "enabled": True,
                "justification": result.get(
                    "pathfinding_justification",
                    "Pathfinding visualizes your traversed route.",
                ),
            },
        }
    )


@app.route("/api/plan", methods=["POST"])
def generate_plan():
    """Generate a tracking plan from a natural language prompt via OpenRouter."""
    data = request.get_json() or {}
    prompt = str(data.get("prompt", "")).strip()

    try:
        client = _get_plan_client()
        system_prompt = (
            "You extract concrete visible physical objects from user scenarios for "
            "3D spatial tracking. Output strict JSON only."
        )
        user_prompt = (
            f'Given this scenario: "{prompt}"\n'
            "Return JSON with exact keys: "
            '{"objects": ["obj1", "obj2", "obj3", "obj4", "obj5"], '
            '"waypoints_justification": "1-2 sentences", '
            '"pathfinding_justification": "1-2 sentences", '
            '"agent_intro": "1-2 sentence first-person statement (starting with I will...) describing what you will do as the spatial intelligence agent for this mission"}. '
            "Objects must be concrete physical items trackable in 3D space. "
            "Return 5-8 unique objects, prioritized by relevance."
        )
        result, _ = client.chat_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.2,
            max_tokens=400,
        )
        finalized_objects = _finalize_plan_objects(result.get("objects", []), prompt)

        return jsonify(
            {
                "objects": finalized_objects,
                "waypoints": {
                    "enabled": True,
                    "justification": result.get(
                        "waypoints_justification", "Waypoints mark key locations."
                    ),
                },
                "pathfinding": {
                    "enabled": True,
                    "justification": result.get(
                        "pathfinding_justification",
                        "Pathfinding visualizes your traversed route.",
                    ),
                },
                "agent_intro": result.get(
                    "agent_intro",
                    "I will scan the scene and lock onto all high-value targets in the environment.",
                ),
            }
        )
    except Exception as e:
        print(f"Plan generation error: {e}; falling back to keyword extraction")
        objects = _finalize_plan_objects([], prompt)
        return jsonify(
            {
                "objects": objects,
                "waypoints": {
                    "enabled": True,
                    "justification": "Waypoints help mark key locations.",
                },
                "pathfinding": {
                    "enabled": True,
                    "justification": "Pathfinding visualizes your traversed route.",
                },
                "agent_intro": "I will scan the scene and lock onto all high-value targets in the environment.",
            }
        )


def _get_assistant_client() -> OpenRouterClient:
    global _assistant_client
    if _assistant_client is None:
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set")

        _assistant_client = OpenRouterClient(
            api_key=api_key,
            primary_model=os.environ.get(
                "ASSISTANT_MODEL", "google/gemini-3-flash-preview"
            ),
            fallback_models=[
                os.environ.get("ASSISTANT_FALLBACK_MODEL", "openai/gpt-4o-mini")
            ],
            timeout=15.0,
            app_name="Real-Eyes Summary Assistant",
            max_retries=2,
            usage_sink=named_tally("summary_assistant"),
        )
    return _assistant_client


def _get_object_enricher() -> Optional[ObjectEnricher]:
    """Lazily build the fine-grained object enricher; None when disabled or no key.

    Mirrors ``_get_assistant_client``. Reverse image search (Phase B) is attached only when
    ``OBJECT_ENRICH_REVERSE_SEARCH`` is on AND a ``SERPAPI_API_KEY`` is present; otherwise
    the enricher is VLM-only.
    """
    global _object_enricher
    if not OBJECT_ENRICH_ENABLED:
        return None
    if _object_enricher is None:
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            return None
        try:
            client = OpenRouterClient(
                api_key=api_key,
                primary_model=OBJECT_ENRICH_MODEL,
                fallback_models=OBJECT_ENRICH_FALLBACKS,
                timeout=float(os.environ.get("OBJECT_ENRICH_TIMEOUT_S", "20")),
                app_name="Real-Eyes Object Enricher",
                max_retries=2,
                usage_sink=named_tally("object_enricher"),
            )
        except Exception as e:
            print(f"Object enricher LLM client init failed: {e}")
            return None
        reverse = None
        if OBJECT_ENRICH_REVERSE_SEARCH:
            serp_key = os.environ.get("SERPAPI_API_KEY", "")
            if serp_key:
                reverse = SerpApiLensProvider(serp_key)
            else:
                print("OBJECT_ENRICH_REVERSE_SEARCH on but SERPAPI_API_KEY missing; VLM-only")
        _object_enricher = ObjectEnricher(
            client, model=OBJECT_ENRICH_MODEL, reverse_search=reverse
        )
    return _object_enricher


# --- Phase B: ephemeral public crop URLs for reverse image search -------------------
# SerpApi's Google Lens takes a public image URL (no raw upload), so when reverse search
# is enabled we stash the crop bytes under an unguessable token and serve them, briefly, at
# /api/crop/<token>.jpg on the worker's public tunnel. Short TTL, no Clerk auth (the token
# is the capability). In-memory + best-effort; fine for single-user workers.
_crop_blobs: dict[str, tuple[bytes, float]] = {}
_crop_blobs_lock = threading.Lock()


def _register_crop_blob(data: bytes) -> str:
    token = secrets.token_urlsafe(24)
    now = time.time()
    with _crop_blobs_lock:
        for t in [t for t, (_, exp) in _crop_blobs.items() if exp < now]:
            _crop_blobs.pop(t, None)
        _crop_blobs[token] = (data, now + OBJECT_ENRICH_CROP_TTL_S)
    return token


def _get_crop_blob(token: str) -> Optional[bytes]:
    now = time.time()
    with _crop_blobs_lock:
        entry = _crop_blobs.get(token)
        if entry is None:
            return None
        data, exp = entry
        if exp < now:
            _crop_blobs.pop(token, None)
            return None
        return data


def _make_crop_url(crop_bytes: Optional[bytes]) -> Optional[str]:
    """Register crop bytes and return a public URL for the scraper, or None if reverse
    search is off / no public base URL is configured."""
    if not (OBJECT_ENRICH_REVERSE_SEARCH and crop_bytes):
        return None
    base = os.environ.get("WORKER_PUBLIC_URL", "").rstrip("/")
    if not base:
        return None
    return f"{base}/api/crop/{_register_crop_blob(crop_bytes)}.jpg"


def _current_scene_report_dict() -> Optional[dict[str, Any]]:
    """The report the summary page is currently showing, as a plain dict: the final
    end-of-scan report once the scan stops, otherwise the latest in-scan progressive
    report. Mirrors ``handle_get_scene_report`` so the chat grounds on exactly what the
    user sees on screen."""
    if isinstance(_last_scene_report, dict):
        return _last_scene_report
    with _progressive_report_lock:
        progressive = _progressive_report
    return progressive.model_dump() if progressive is not None else None


@app.route("/api/assistant/chat", methods=["POST"])
def assistant_chat():
    """Live "talk to your scan" chat for the summary page. Grounds in the worker's
    cached scene report (final or in-scan progressive) through the SAME core as the
    persisted revisit Q&A — so the two surfaces can't drift — instead of the thin,
    lossy object-name list the client used to send. Degrades gracefully until a report
    exists. Returns the canonical {answer, model, degraded, focus, evidence} shape."""
    data = request.get_json() or {}
    user_message = str(data.get("message", "")).strip()
    if not user_message:
        return jsonify({"error": "message is required"}), 400

    history = data.get("history")
    if not isinstance(history, list):
        history = []

    report = _current_scene_report_dict()
    if not isinstance(report, dict):
        return jsonify(
            {
                "answer": (
                    "I don't have a scan report to ground on yet — once the scan has "
                    "mapped a bit of the space (or finished), I can answer questions "
                    "about where things are."
                ),
                "model": "",
                "degraded": True,
                "focus": None,
                "evidence": [],
            }
        )

    facts = report.get("facts") if isinstance(report.get("facts"), dict) else {}
    try:
        # The live cache holds no keyframe blobs, so this path is text-grounded
        # (no image_loader); the persisted revisit path attaches cited keyframes.
        return jsonify(
            _grounded_scene_answer(facts, report, user_message, history, image_loader=None)
        )
    except Exception as e:
        print(f"Assistant chat error: {e}")
        return jsonify({"error": str(e)}), 500


# ------------------------------
# Durable scene history (Phase 4) + grounded Q&A (Phase 5) HTTP routes
# ------------------------------
# Served by both the GPU worker and the always-on CPU broker (both mount
# asgi_application); the dashboard hits the broker so browsing history never boots
# an A100. All routes live under /api/ (auto-authed by require_modal_auth) and are
# hard-scoped to the requesting Clerk user. On the broker _require_worker_owner is a
# no-op (no assigned owner), so any approved user may read their own scenes.

# Input bounds for the grounded Q&A path (defence against a leaked share token driving spend).
SCENE_QA_MAX_QUESTION_CHARS = int(os.environ.get("SCENE_QA_MAX_QUESTION_CHARS", "1000"))
SCENE_QA_MAX_HISTORY = int(os.environ.get("SCENE_QA_MAX_HISTORY", "12"))
SCENE_QA_MAX_OBJECTS = int(os.environ.get("SCENE_QA_MAX_OBJECTS", "8"))
SCENE_QA_MAX_IMAGES = int(os.environ.get("SCENE_QA_MAX_IMAGES", "3"))
# Keyframes offered to the model as a labeled menu it cites evidence from (> shown,
# so it has a real choice). The shown evidence stays capped at SCENE_QA_MAX_IMAGES.
SCENE_QA_CANDIDATE_IMAGES = int(os.environ.get("SCENE_QA_CANDIDATE_IMAGES", "6"))
# Opt-in semantic tie-break: when relevance ties (notably the common no-lexical-match
# case) break by question↔object embedding similarity instead of static confidence.
# Off by default — it adds one batched embeddings call per QA; degrades to confidence
# if disabled, keyless, or the call fails. Reuses OPENROUTER_API_KEY (OpenRouter serves
# an OpenAI-compatible /embeddings endpoint).
SCENE_QA_EMBED_TIEBREAK = os.environ.get("SCENE_QA_EMBED_TIEBREAK", "0").strip().lower() not in (
    "0", "", "false", "no", "off",
)
SCENE_QA_EMBED_MODEL = os.environ.get("SCENE_QA_EMBED_MODEL", "openai/text-embedding-3-small")
# Skip embeddings past this many objects (one batched call, but bound the cost/latency).
SCENE_QA_EMBED_MAX_OBJECTS = int(os.environ.get("SCENE_QA_EMBED_MAX_OBJECTS", "50"))


def _auth_user_id() -> Optional[str]:
    claims = request.environ.get(AUTH_CLAIMS_ENV_KEY, {})
    user_id = str(claims.get("sub", "")).strip()
    return user_id or None


@app.route("/api/session/refresh", methods=["POST"])
def refresh_session_token():
    """Exchange a still-valid Clerk JWT for a durable broker session token. The page's
    URL-hash Clerk JWT is short-lived and unrefreshable, so the revisit page calls this
    on load and uses the returned token thereafter — keeping its history reads + grounded
    Q&A authenticated long after the hash JWT would have expired. ``require_modal_auth``
    has already verified the bootstrap token (Clerk *or* an existing session token) and
    scan access before we get here."""
    claims = request.environ.get(AUTH_CLAIMS_ENV_KEY, {})
    user_id = str(claims.get("sub", "")).strip()
    if not user_id:
        return _auth_response(AuthError("invalid_token"))
    token, expires_at = _issue_session_token(claims)
    return jsonify({"token": token, "expires_at": expires_at})


# --- /api/keys — first-class API keys (wire contract v1) --------------------------
# Under /api/ so require_modal_auth already ran: the caller's verified claims are in
# request.environ[AUTH_CLAIMS_ENV_KEY]. Share/project tokens can never reach these
# routes (their endpoint allow-lists 403 first). Registry-less deploys mirror the
# _scene_persistence posture: list serves empty, delete 404s, mint reports unavailable.


@app.route("/api/keys", methods=["POST"])
def create_api_key_route():
    """Mint an API key. The raw ``ork_`` secret is returned ONLY here — at rest we keep
    its SHA-256, so it can never be shown again. Bearer auth must be a Clerk JWT or a
    broker session token: an API key cannot mint another API key (403
    ``api_key_cannot_mint``), so a leaked key can't self-propagate past revocation."""
    claims = request.environ.get(AUTH_CLAIMS_ENV_KEY, {})
    user_id = str(claims.get("sub", "")).strip()
    if not user_id:
        return _auth_response(AuthError("invalid_token"))
    if claims.get("auth_kind") == "api_key":
        return jsonify({"error": "api_key_cannot_mint"}), 403
    if _api_key_registry is None:
        return jsonify({"error": "api_keys_unavailable"}), 503
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        payload = {}
    full_key, record = _api_key_registry.mint(
        user_id,
        payload.get("name"),
        # The same fields _issue_session_token snapshots: identity only, plus tier
        # (break-glass if the ledger is unreachable). Deliberately NOT scansRemaining:
        # an API key outlives any session token, and a quota snapshot inside a durable
        # credential is a standing grant — the same defect the billing line removed
        # from session tokens. Balance is read live from the ledger at dispatch.
        {"tier": claims.get("tier")},
    )
    return jsonify(
        {
            "key": full_key,
            "key_id": record["key_id"],
            "name": record["name"],
            "prefix": record["prefix"],
            "created_at": record["created_at"],
        }
    )


@app.route("/api/keys", methods=["GET"])
def list_api_keys_route():
    """The caller's API keys, newest first, revoked included — metadata only (the raw
    secret is unrecoverable by design; ``prefix`` is the display handle)."""
    claims = request.environ.get(AUTH_CLAIMS_ENV_KEY, {})
    user_id = str(claims.get("sub", "")).strip()
    if not user_id:
        return _auth_response(AuthError("invalid_token"))
    if _api_key_registry is None:
        return jsonify({"keys": []})
    keys = [
        {
            "key_id": record.get("key_id"),
            "name": record.get("name"),
            "prefix": record.get("prefix"),
            "created_at": record.get("created_at"),
            "last_used_at": record.get("last_used_at"),
            "revoked_at": record.get("revoked_at"),
        }
        for record in _api_key_registry.list_for_user(user_id)
    ]
    return jsonify({"keys": keys})


@app.route("/api/keys/<key_id>", methods=["DELETE"])
def revoke_api_key_route(key_id):
    """Revoke a key. Idempotent — a repeat revoke returns the ORIGINAL ``revoked_at``.
    404 for an unknown key_id or another user's key (indistinguishable on purpose, so
    key_ids can't be probed across accounts)."""
    claims = request.environ.get(AUTH_CLAIMS_ENV_KEY, {})
    user_id = str(claims.get("sub", "")).strip()
    if not user_id:
        return _auth_response(AuthError("invalid_token"))
    if _api_key_registry is None:
        return jsonify({"error": "not_found"}), 404
    record = _api_key_registry.revoke(user_id, key_id)
    if record is None:
        return jsonify({"error": "not_found"}), 404
    return jsonify({"key_id": record["key_id"], "revoked_at": record["revoked_at"]})


@app.route("/api/scenes", methods=["GET"])
def list_scenes_route():
    if _scene_persistence is None:
        return jsonify({"scenes": []})
    user_id = _auth_user_id()
    if not user_id:
        return _auth_response(AuthError("invalid_token"))
    return jsonify({"scenes": _scene_persistence.list_scenes(user_id)})


@app.route("/api/scenes/<scan_id>", methods=["GET"])
def get_scene_route(scan_id):
    if _scene_persistence is None:
        return jsonify({"error": "not_found"}), 404
    user_id = _auth_user_id()
    if not user_id:
        return _auth_response(AuthError("invalid_token"))
    record = _scene_persistence.get_scene(user_id, scan_id)
    if not record:
        return jsonify({"error": "not_found"}), 404
    # Expose what the revisit/embed pages need; never leak the internal owner id.
    return jsonify(
        {
            "scan_id": record.get("scan_id"),
            "created_at": record.get("created_at"),
            "report": record.get("report"),
            "facts": record.get("facts"),
            "keyframes": record.get("keyframes"),
            "points_key": record.get("points_key"),
            "point_count": record.get("point_count"),
            "scene_center": record.get("scene_center", [0.0, 0.0, 0.0]),
            "has_splat": bool(record.get("splat_key")),
            # "Latest calibrated" derived-artifact pointer (set by the anchor/clamp Refine routes),
            # so the workflow Export/Refine stages can offer the calibrated/clamped result and fetch
            # it back for a visual diff. ``None`` until a Refine action runs. Never leaks a path.
            "derived_latest": record.get("derived_latest"),
            "client": record.get("client"),
            "project": record.get("project"),
            # Additive (demo build, W0): how this scene came to exist — "recon_video" |
            # "imported_splat" | "recon_gemini2" | "recon_depthcam" | "robot_recording" |
            # "recon_lidar" | None for pre-demo scans (see docs/demo-2026-07 +
            # store.save_scene's source kwarg).
            "source": record.get("source"),
            # Additive: the AI-completed object layer (ObjectLayerManifest) rides along on
            # SCENE_DETAIL when the scan has one, so the embed viewer / product-workflow Detect
            # stage picks it up with no extra round-trip (mirrors the optional
            # PersistedScene.object_layer field in web/packages/protocol). OMITTED when absent
            # (dict-spread of {}), so layer-less embeds are byte-for-byte unaffected. Placed as
            # the LAST key (a single spread line) so it never overlaps the derived_latest key the
            # sibling export/derived branch adds mid-dict — the two merge cleanly. Pure read;
            # never mutates the record. See platform/contracts + scene_report/object_layer.py.
            **({"object_layer": record["object_layer"]} if record.get("object_layer") else {}),
            # Additive: the SYNTHETIC-VIEW manifest (metadata only — the PNGs are derived
            # blobs). A separate key from ``keyframes`` on purpose and forever: a render of
            # an imported splat and a photograph from a capture are both usable evidence
            # and are NOT equally strong, so nothing downstream may confuse them. Omitted
            # when absent, in its own spread line, for the same clean-merge reason as
            # object_layer above. See server/oreos/synthetic_views.py.
            **(
                {"synthetic_views": record["synthetic_views"]}
                if record.get("synthetic_views")
                else {}
            ),
        }
    )


@app.route("/api/scenes/<scan_id>/share", methods=["POST"])
def share_scene_route(scan_id):
    """Mint a read-only, single-scan share token + embed URL for a scan the caller owns.
    Owner-authed (Clerk/session only — a share token cannot reach this endpoint). The
    returned ``embed_url`` is dropped into a third party's dashboard as an <iframe>; the
    token grants read + Q&A on this one scan until ``expires_at``. See
    platform/contracts/embed-delivery.md."""
    if _scene_persistence is None:
        return jsonify({"error": "not_found"}), 404
    user_id = _auth_user_id()
    if not user_id:
        return _auth_response(AuthError("invalid_token"))
    record = _scene_persistence.get_scene(user_id, scan_id)
    if not record:
        return jsonify({"error": "not_found"}), 404
    data = request.get_json(silent=True) or {}
    token, expires_at = _issue_share_token(scan_id, user_id, data.get("ttl_seconds"))
    # Encode scan_id (the user-controllable value, e.g. a pilot --scan-id); the token is a
    # JWT (already URL-safe) so it goes in raw, as is standard for a JWT in a URL.
    embed_url = (
        f"{_request_origin()}/embed.html#scan_id={quote(str(scan_id), safe='')}"
        f"&share_token={token}"
    )
    return jsonify(
        {
            "share_token": token,
            "embed_url": embed_url,
            "scan_id": scan_id,
            "expires_at": expires_at,
            # Ties this link to the person it is sent to: record it next to their name in
            # the discovery log, then read GET …/share/access to see what they did.
            "access_id": access_id_for_token(token),
        }
    )


@app.route("/api/scenes/<scan_id>/share/access", methods=["GET"])
def share_access_route(scan_id):
    """Owner-only: what every minted share link for this scan was used for — per link
    (``access_id``): opened, request count, visits, first/last seen, returned (a visit
    >= 24 h after the first), and the Q&A questions asked. A share token cannot reach
    this route (not in ``SHARE_TOKEN_ALLOWED_ENDPOINTS``). ``?events=1`` appends the raw
    event list. Definitions live in ``server/share_access.py``."""
    if _scene_persistence is None or _share_access is None:
        return jsonify({"error": "not_found"}), 404
    user_id = _auth_user_id()
    if not user_id:
        return _auth_response(AuthError("invalid_token"))
    record = _scene_persistence.get_scene(user_id, scan_id)
    if not record:
        return jsonify({"error": "not_found"}), 404
    payload = _share_access.summary(user_id, scan_id)
    if str(request.args.get("events", "")).strip() in ("1", "true", "yes"):
        payload["events"] = _share_access.events(user_id, scan_id)
    return jsonify(payload)


def _scenes_matching_project(user_id: str, client: str, project: str) -> list[dict[str, Any]]:
    """The owner's scene summaries tagged with this ``client``+``project`` (newest first, as
    ``list_scenes`` returns them). Empty when no store is configured."""
    if _scene_persistence is None:
        return []
    return [
        s
        for s in _scene_persistence.list_scenes(user_id)
        if str(s.get("client") or "") == client and str(s.get("project") or "") == project
    ]


@app.route("/api/projects/share", methods=["POST"])
def share_project_route():
    """Mint a building (project) token + building embed URL for a ``client``/``project`` the
    caller owns. Owner-authed (Clerk/session only — a project token cannot reach this
    endpoint, it is rejected by ``_try_project_token_auth``). 404 unless the owner has ≥1
    scene tagged with this client+project. The returned ``building_embed_url`` is dropped
    into a third party's dashboard as an <iframe>; the project token authorizes ONLY the
    building manifest (which hands out per-scene share tokens). See
    platform/contracts/embed-delivery.md."""
    user_id = _auth_user_id()
    if not user_id:
        return _auth_response(AuthError("invalid_token"))
    data = request.get_json(silent=True) or {}
    client = str(data.get("client") or "").strip()
    project = str(data.get("project") or "").strip()
    if not client or not project:
        return jsonify({"error": "missing_client_or_project"}), 400
    if not _scenes_matching_project(user_id, client, project):
        return jsonify({"error": "not_found"}), 404
    token, expires_at = _issue_project_token(client, project, user_id, data.get("ttl_seconds"))
    building_embed_url = (
        f"{_request_origin()}/building.html#client={quote(client, safe='')}"
        f"&project={quote(project, safe='')}&token={quote(token, safe='')}"
    )
    return jsonify(
        {
            "project_token": token,
            "client": client,
            "project": project,
            "expires_at": expires_at,
            "building_embed_url": building_embed_url,
        }
    )


@app.route("/api/projects/manifest", methods=["GET"])
def project_manifest_route():
    """Building-tour manifest — every scene in one ``client``/``project``, each carrying a
    FRESH per-scene read-only share token + embed URL. Authed by a PROJECT token ONLY
    (``require_modal_auth`` → ``_try_project_token_auth`` set the project scope + the owner
    sub/client/project in the claims env; a Clerk/session/share token that reaches here is
    rejected because it lacks that scope). The per-scene reads use the per-scene share tokens
    returned here — the project token never authorizes a scene read — so the per-scene
    security model is intact. See platform/contracts/embed-delivery.md."""
    claims = request.environ.get(AUTH_CLAIMS_ENV_KEY, {})
    if claims.get("scope") != PROJECT_TOKEN_SCOPE or not claims.get("project_token"):
        return _auth_response(AuthError("forbidden", 403))
    user_id = str(claims.get("sub", "")).strip()
    client = str(claims.get("client", "")).strip()
    project = str(claims.get("project", "")).strip()
    if not user_id or not client or not project:
        return _auth_response(AuthError("invalid_token"))
    # Cap each minted per-scene token to the project token's remaining lifetime, so a
    # building link fetched near expiry can't spawn scene links that outlive it.
    project_exp = claims.get("exp")
    origin = _request_origin()
    scenes: list[dict[str, Any]] = []
    for i, summary in enumerate(_scenes_matching_project(user_id, client, project), 1):
        scan_id = summary.get("scan_id")
        # An explicit per-scene label (set by the operator who knows what each space is) wins.
        custom_label = str(summary.get("label") or "").strip()
        room_type = str(summary.get("room_type") or "").strip()
        # Else use the capitalized room type — but "unknown"/"other" aren't real classifications,
        # so fall back to a positional "Scene N" for those rather than showing a bare "Other".
        label = custom_label or (
            room_type.capitalize()
            if room_type and room_type.lower() not in ("unknown", "other")
            else f"Scene {i}"
        )
        token, _ = _issue_share_token(scan_id, user_id, max_exp=project_exp)
        embed_url = (
            f"{origin}/embed.html#scan_id={quote(str(scan_id), safe='')}"
            f"&share_token={token}"
        )
        scenes.append(
            {
                "scan_id": scan_id,
                "label": label,
                "room_type": summary.get("room_type"),
                "created_at": summary.get("created_at"),
                "has_splat": bool(summary.get("has_splat")),
                "share_token": token,
                "embed_url": embed_url,
            }
        )
    return jsonify({"client": client, "project": project, "scenes": scenes})


@app.route("/api/scenes/<scan_id>/keyframes/<blob_key>", methods=["GET"])
def get_scene_keyframe_route(scan_id, blob_key):
    if _scene_persistence is None:
        return jsonify({"error": "not_found"}), 404
    user_id = _auth_user_id()
    if not user_id:
        return _auth_response(AuthError("invalid_token"))
    data = _scene_persistence.get_keyframe(user_id, scan_id, blob_key)
    if data is None:
        return jsonify({"error": "not_found"}), 404
    resp = app.response_class(data, mimetype="image/jpeg")
    resp.headers["Cache-Control"] = "private, max-age=3600"
    return resp


@app.route("/api/crop/<token>.jpg", methods=["GET"])
def get_crop_route(token):
    """Serve an ephemeral object crop by capability token (Phase B reverse image search).

    No Clerk auth — the unguessable, short-TTL token IS the capability, so an external
    reverse-image-search service (SerpApi/Google Lens) can fetch it. Disabled in practice
    unless OBJECT_ENRICH_REVERSE_SEARCH is on and WORKER_PUBLIC_URL is set."""
    data = _get_crop_blob(token)
    if data is None:
        return jsonify({"error": "not_found"}), 404
    resp = app.response_class(data, mimetype="image/jpeg")
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/api/scenes/<scan_id>/points", methods=["GET"])
def get_scene_points_route(scan_id):
    """Return a DISPLAY view of the saved cloud as a ready-to-render SLAMUpdate payload
    (the same base64 float32/uint8 shape the live viewer decodes) so the revisit page
    can call SceneManager.updateVisualization() unchanged. The stored artifact is the
    full world-frame cloud; here we recenter it (display frame, matching the live
    viewer) and downsample to SCENE_DISPLAY_MAX_POINTS so the payload stays snappy. The
    complete cloud is downloadable in full via /cloud.ply."""
    if _scene_persistence is None:
        return jsonify({"error": "not_found"}), 404
    user_id = _auth_user_id()
    if not user_id:
        return _auth_response(AuthError("invalid_token"))
    record = _scene_persistence.get_scene(user_id, scan_id)
    if not record:
        return jsonify({"error": "not_found"}), 404
    empty = {"type": "full", "n_points": 0, "points_b64": "", "colors_b64": ""}
    cloud = _scene_persistence.get_cloud(user_id, scan_id)
    if cloud is None or cloud[0].shape[0] == 0:
        return jsonify(empty)
    positions, colors = cloud
    center = np.asarray(record.get("scene_center") or [0.0, 0.0, 0.0], dtype=np.float32)
    disp_pos, disp_col = downsample_cloud(positions, colors, SCENE_DISPLAY_MAX_POINTS)
    disp_pos = np.ascontiguousarray(disp_pos - center, dtype=np.float32)  # world → display
    disp_col = np.ascontiguousarray(disp_col, dtype=np.uint8).reshape(-1, 3)
    return jsonify(
        {
            "type": "full",
            "n_points": int(disp_pos.shape[0]),
            "n_cameras": 0,
            "points_b64": base64.b64encode(disp_pos.tobytes()).decode("ascii"),
            "colors_b64": base64.b64encode(disp_col.tobytes()).decode("ascii"),
            "camera_positions": [],
            "camera_rotations": [],
            "scene_center": [float(v) for v in center],
        }
    )


@app.route("/api/scenes/<scan_id>/cloud.ply", methods=["GET"])
def get_scene_cloud_ply_route(scan_id):
    """Download the FULL-fidelity point cloud as a binary PLY (world frame, every stored
    point) — the durable artifact, e.g. for evidence/export; opens in
    CloudCompare/MeshLab/etc. Auth via header or ?auth_token= so a plain <a download>
    works. The map is up-to-scale and not gravity-aligned (see scene-report docs)."""
    if _scene_persistence is None:
        return jsonify({"error": "not_found"}), 404
    user_id = _auth_user_id()
    if not user_id:
        return _auth_response(AuthError("invalid_token"))
    record = _scene_persistence.get_scene(user_id, scan_id)
    if not record:
        return jsonify({"error": "not_found"}), 404
    cloud = _scene_persistence.get_cloud(user_id, scan_id)
    if cloud is None or cloud[0].shape[0] == 0:
        return jsonify({"error": "no_points"}), 404
    ply = build_ply_bytes(cloud[0], cloud[1])
    return send_file(
        io.BytesIO(ply),
        mimetype="application/octet-stream",
        as_attachment=True,
        download_name=f"scan-{scan_id}.ply",
    )


@app.route("/api/scenes/<scan_id>/splat.ply", methods=["GET"])
def get_scene_splat_ply_route(scan_id):
    """Download the scan's 3DGS ``splat.ply`` artifact (produced by core's --export_splat).
    Auth via header or ``?share_token=``/``?auth_token=`` so a plain <a download> or the
    embed's splat loader works. 404 (no_splat) for scans with no splat — older scans or
    reconstruction-only runs have only a point cloud (/cloud.ply), so callers degrade."""
    if _scene_persistence is None:
        return jsonify({"error": "not_found"}), 404
    user_id = _auth_user_id()
    if not user_id:
        return _auth_response(AuthError("invalid_token"))
    record = _scene_persistence.get_scene(user_id, scan_id)
    if not record:
        return jsonify({"error": "not_found"}), 404
    # Prefer STREAMING the splat from disk: artifacts can be hundreds of MB, and reading one
    # fully into a BytesIO (the get_splat path) would OOM the small-memory broker (and balloon
    # under concurrency). Fall back to in-memory bytes for stores with no on-disk path (tests).
    get_path = getattr(_scene_persistence, "get_splat_path", None)
    splat_path = get_path(user_id, scan_id) if callable(get_path) else None
    if splat_path:
        return send_file(
            splat_path,
            mimetype="application/octet-stream",
            as_attachment=True,
            download_name=f"scan-{scan_id}.splat.ply",
            conditional=True,  # honor Range/If-Range so a big splat resumes + seeks
        )
    splat = _scene_persistence.get_splat(user_id, scan_id)
    if not splat:
        return jsonify({"error": "no_splat", "message": "scan has no 3DGS splat artifact"}), 404
    return send_file(
        io.BytesIO(splat),
        mimetype="application/octet-stream",
        as_attachment=True,
        download_name=f"scan-{scan_id}.splat.ply",
    )


@app.route("/api/scenes/<scan_id>/objects/<object_id>.glb", methods=["GET"])
def get_scene_object_asset_route(scan_id, object_id):
    """Serve one object-layer asset — the world-baked ``.glb`` the embed viewer / product-workflow
    Detect stage places at the object's measured pose. Auth via header or ``?share_token=`` (the
    ``<model>``/GLTFLoader fetch can't set an Authorization header — same trick as ``/splat.ply``);
    a share token can reach ONLY this scan's own assets (enforced in ``_try_share_token_auth`` +
    ``get_object_asset_path`` gating the id to the scan's manifest). 404 for scans with no layer,
    an unknown/box-only object, or an id the manifest doesn't declare. The path segment is a Flask
    ``<object_id>`` (no slashes) and is re-validated against a strict allow-list on the store side,
    so no directory traversal is possible.

    Kept adjacent to ``/splat.ply`` (before ``delete_scene_route``) so it does not overlap the
    sibling product-workflow routes (objects PATCH is added after ``delete_scene_route``; the
    derived/anchor/clamp routes are further down) — all three branches merge cleanly."""
    if _scene_persistence is None:
        return jsonify({"error": "not_found"}), 404
    user_id = _auth_user_id()
    if not user_id:
        return _auth_response(AuthError("invalid_token"))
    record = _scene_persistence.get_scene(user_id, scan_id)
    if not record:
        return jsonify({"error": "not_found"}), 404
    # Prefer STREAMING from disk (GLBs are single-digit MB but can be tens for a big layer);
    # fall back to in-memory bytes for stores with no on-disk path (tests).
    get_path = getattr(_scene_persistence, "get_object_asset_path", None)
    asset_path = get_path(user_id, scan_id, object_id) if callable(get_path) else None
    if asset_path:
        resp = send_file(
            asset_path,
            mimetype="model/gltf-binary",
            conditional=True,  # honor Range/If-Range so the loader can resume/seek
            max_age=86400,
        )
    else:
        get_bytes = getattr(_scene_persistence, "get_object_asset", None)
        data = get_bytes(user_id, scan_id, object_id) if callable(get_bytes) else None
        if not data:
            return jsonify({"error": "not_found"}), 404
        resp = send_file(io.BytesIO(data), mimetype="model/gltf-binary", max_age=86400)
    # Per-scan asset behind a share token → keep it out of shared caches.
    try:
        resp.headers["Cache-Control"] = "private, max-age=86400"
    except (AttributeError, TypeError):
        pass  # test double for send_file returns a plain dict; nothing to set
    return resp


@app.route("/api/scenes/<scan_id>", methods=["DELETE"])
def delete_scene_route(scan_id):
    if _scene_persistence is None:
        return jsonify({"error": "not_found"}), 404
    user_id = _auth_user_id()
    if not user_id:
        return _auth_response(AuthError("invalid_token"))
    _scene_persistence.delete_scene(user_id, scan_id)
    return jsonify({"status": "deleted"})


def _coerce_object_edit(data: Any) -> dict[str, Any]:
    """Validate + normalize a PATCH /api/scenes/<id>/objects/<object_id> body into the
    keyword args ``update_object_edit`` accepts.

    Body is ``{label?: string, dismissed?: boolean}``; at least one field must be present.
    Only keys that are actually present are forwarded, so a relabel and a dismiss are
    independent (one never clobbers the other). Raises ``ValueError`` (→ 400) on a
    non-object body, a wrong-typed field, or an empty edit.
    """
    if not isinstance(data, dict):
        raise ValueError("request body must be a JSON object")
    edit: dict[str, Any] = {}
    if "label" in data:
        label = data["label"]
        if not isinstance(label, str):
            raise ValueError("'label' must be a string")
        edit["label"] = label
    if "dismissed" in data:
        dismissed = data["dismissed"]
        if not isinstance(dismissed, bool):
            raise ValueError("'dismissed' must be a boolean")
        edit["dismissed"] = dismissed
    if not edit:
        raise ValueError("provide at least one of 'label' or 'dismissed'")
    return edit


@app.route("/api/scenes/<scan_id>/objects/<int:object_id>", methods=["PATCH"])
def patch_scene_object_route(scan_id, object_id):
    """Persist a human review edit onto one detected object of a persisted scan (Stage 3,
    Detect). Owner-authed (Clerk/session only — mirrors the other product-workflow routes;
    a share token cannot reach this endpoint).

    ``object_id`` is the 0-based index into this scan's ``facts.objects`` inventory (the
    inventory is written once at finalize and never reordered, so the index is a stable
    per-object handle). Body ``{label?: string, dismissed?: boolean}`` — at least one field.

    NON-DESTRUCTIVE: a relabel records the operator's ``label`` in the object's
    ``human_label`` field ALONGSIDE the model's original ``query`` (never overwritten);
    ``dismissed: true`` flags the instance as hidden from the reviewed inventory without
    deleting the detection. Returns the updated object. See
    ``server/scene_report/store.py::ModalScenePersistence.update_object_edit``.
    """
    if _scene_persistence is None:
        return jsonify({"error": "not_found"}), 404
    user_id = _auth_user_id()
    if not user_id:
        return _auth_response(AuthError("invalid_token"))
    record = _scene_persistence.get_scene(user_id, scan_id)
    if not record:
        return jsonify({"error": "not_found"}), 404

    try:
        edit = _coerce_object_edit(request.get_json(silent=True))
    except ValueError as exc:
        return jsonify({"error": "invalid_request", "message": str(exc)}), 400

    try:
        obj = _scene_persistence.update_object_edit(user_id, scan_id, int(object_id), **edit)
    except KeyError:
        # get_scene passed above, so this only fires on a genuine race (scan deleted between
        # the read and the write); surface it as the same not_found the reader returns.
        return jsonify({"error": "not_found"}), 404
    except IndexError:
        return (
            jsonify({"error": "unknown_object", "message": f"no object with id {object_id} in this scan"}),
            404,
        )
    return jsonify({"scan_id": scan_id, "object_id": int(object_id), "object": obj})


@app.route("/api/scenes/<scan_id>/qa", methods=["POST"])
def scene_qa_route(scan_id):
    """Grounded face-to-face Q&A over a persisted scene (Phase 5). No GPU/SLAM —
    answers are grounded in the saved 3D facts + report + a few cited keyframes."""
    if _scene_persistence is None:
        return jsonify({"error": "not_found"}), 404
    user_id = _auth_user_id()
    if not user_id:
        return _auth_response(AuthError("invalid_token"))
    record = _scene_persistence.get_scene(user_id, scan_id)
    if not record:
        return jsonify({"error": "not_found"}), 404
    data = request.get_json() or {}
    question = str(data.get("question", "")).strip()
    if not question:
        return jsonify({"error": "question is required"}), 400
    # Bound the inputs before the LLM path: a leaked share token can drive Q&A, so cap the
    # question length and the history depth so it can't be used to run up unbounded model spend.
    if len(question) > SCENE_QA_MAX_QUESTION_CHARS:
        question = question[:SCENE_QA_MAX_QUESTION_CHARS]
    history = data.get("history") if isinstance(data.get("history"), list) else []
    history = history[-SCENE_QA_MAX_HISTORY:]
    auth_claims = request.environ.get(AUTH_CLAIMS_ENV_KEY, {}) or {}
    if auth_claims.get("share") and request.environ.get(SHARE_TOKEN_ENV_KEY):
        # What a prospect asks the scene is the job, in their words — record it.
        _record_share_access(auth_claims, request.environ[SHARE_TOKEN_ENV_KEY], question=question)
    try:
        return jsonify(_answer_scene_question(user_id, scan_id, record, question, history))
    except Exception as e:
        print(f"Scene QA error: {e}")
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Product workflow: GPU-free dataset export, metric anchor, auto-clamp.
#
# All three read a scan a user already owns (broker-only, no live Solver/GPU) and are
# NON-DESTRUCTIVE: the anchor + clamp routes never touch a scene's original persisted
# ``cloud.npz`` / ``splat.ply`` (live pilot embeds ship from those) -- they write
# calibrated/clamped copies under NEW ``derived/...`` keys via
# ``ModalScenePersistence.save_derived_artifact``. The metric-anchor and scale-clamp math
# mirror unmerged ``core`` (reality-opened/core) branches -- read there as a SPEC, never
# imported at runtime (the deployed ``server`` pins a released core tag that predates that
# work): ``vggt_slam.metric_absolute.scale_from_known_distance`` and
# ``vggt_slam.splat_export.clamp_scales_percentile`` / ``read_splat_ply`` /
# ``write_splat_ply``. See ``server/scene_report/splat_io.py``.
# ---------------------------------------------------------------------------

_EXPORT_FORMATS = ("openreality", "groot_lerobot_v2")

# Matches vggt_slam.metric_absolute.MIN_DISTANCE_UNITS -- reject a degenerate
# (near-coincident) operator-picked point pair; no scale can be inferred from it.
_MIN_ANCHOR_DISTANCE_UNITS = _anchor_impl.MIN_ANCHOR_DISTANCE_UNITS


def _build_export_zip_file(record: dict[str, Any], export_format: str) -> str:
    """The export-zip builder — see ``server.export.zip_builder.build_export_zip_file``.

    The implementation moved to that module so the background export job
    (``modal_oreos_export.py``) can call the SAME builder without importing this one
    (which drags in torch, cv2 and the whole Flask/SocketIO surface). Kept here as the
    name every existing caller and test already knows."""
    return build_export_zip_file(record, export_format)


def _build_export_zip(record: dict[str, Any], export_format: str) -> bytes:
    """``_build_export_zip_file`` convenience wrapper returning the zip BYTES
    (tests inspect the archive tree through it). The route must NOT use this —
    it holds the whole archive in RAM (see ``_build_export_zip_file``)."""
    path = _build_export_zip_file(record, export_format)
    try:
        with open(path, "rb") as fh:
            return fh.read()
    finally:
        with contextlib.suppress(OSError):
            os.remove(path)


@app.route("/api/scenes/<scan_id>/export", methods=["GET"])
def export_scene_route(scan_id):
    """GPU-free robot-training dataset export from a persisted scan (Stage 4), zipped.

    ``?format=openreality`` (default) — the native OpenReality export tree.
    ``?format=groot_lerobot_v2`` — that tree transcoded to GR00T-LeRobot v2.
    ``?format=isaac_usd`` — an Isaac Sim/Lab USD scene (``scene.usd`` + ``trajectory.usd`` +
    ``manifest.json``), metric + gravity-aligned (``server/export/isaac/``, see
    ``docs/isaac-export.md``). GATED on a metric scale: pass ``?scale=<metres-per-SLAM-unit>``
    OR run the Refine **Metric anchor** first (its ``derived_latest`` of kind ``"anchor"`` is
    consumed — the calibrated geometry is already metric). Without either -> 409
    ``metric_scale_required`` (never a silent up-to-scale USD). Needs ``usd-core``/``open3d`` in
    the runtime image; if absent -> 501 ``isaac_unavailable``. See ``_export_isaac_usd``.

    ``?source=<derived_key>`` — OPTIONAL geometry selector. Export the calibrated/clamped
    result of a Refine action instead of the original geometry: pass a ``derived/...`` key
    returned by ``POST .../anchor`` or ``.../clamp`` (e.g. ``derived/anchor/<stamp>/cloud.ply``);
    the export reads that key's whole derived group (its sibling ``cloud.ply`` / ``trajectory.npz``
    / ``splat.ply``), falling back to the ORIGINAL for any artifact type the group lacks. Precedence
    is EXPLICIT ``source`` > the scan's persisted "latest calibrated" pointer (set by anchor/clamp)
    > original. Pass ``?source=original`` to force the original even when a pointer exists. A
    ``source`` key that isn't a real derived artifact of THIS scan is a 404 (``unknown_derived_key``)
    — never a silent fallback and never a path traversal. NON-DESTRUCTIVE: only reads derived
    artifacts; a scan's original ``cloud.npz`` / ``splat.ply`` / ``trajectory.npz`` are never touched.
    """
    if _scene_persistence is None:
        return jsonify({"error": "not_found"}), 404
    user_id = _auth_user_id()
    if not user_id:
        return _auth_response(AuthError("invalid_token"))
    record = _scene_persistence.get_scene(user_id, scan_id)
    if not record:
        return jsonify({"error": "not_found"}), 404

    # Validate an EXPLICIT derived selector up front: a bogus/foreign key is a hard 404, not a
    # silent degrade to the original. ``original``/empty (force-original) skips the check.
    source = request.args.get("source")
    if source is not None and str(source).strip() not in ("", EXPORT_SOURCE_ORIGINAL):
        get_derived_path = getattr(_scene_persistence, "get_derived_artifact_path", None)
        resolved = (
            get_derived_path(user_id, scan_id, str(source).strip())
            if callable(get_derived_path)
            else None
        )
        if not resolved:
            return (
                jsonify(
                    {
                        "error": "unknown_derived_key",
                        "message": (
                            f"no derived artifact {str(source).strip()!r} for this scan; pass a "
                            "derived key returned by the anchor/clamp routes, or omit ?source="
                        ),
                    }
                ),
                404,
            )

    export_format = (request.args.get("format") or "openreality").strip()
    if export_format == "isaac_usd":
        # Own builder + gates (metric scale, heavy USD deps) — not a ``_build_export_zip`` format.
        return _export_isaac_usd(user_id, scan_id, record, source)
    if export_format not in _EXPORT_FORMATS:
        return (
            jsonify(
                {
                    "error": "invalid_format",
                    "message": (
                        f"unknown format {export_format!r}; expected one of "
                        f"{list(_EXPORT_FORMATS)} (or isaac_usd)"
                    ),
                }
            ),
            400,
        )

    normalized = load_record_from_store(_scene_persistence, user_id, scan_id, source=source)
    if normalized is None:
        return (
            jsonify(
                {
                    "error": "no_trajectory",
                    "message": (
                        "scan has no persisted per-keyframe trajectory (pre-Stage-4 scan); "
                        "cannot export GPU-free"
                    ),
                }
            ),
            404,
        )

    try:
        zip_file_path = _build_export_zip_file(normalized, export_format)
    except Exception as exc:
        print(f"[export] scan {scan_id} export failed ({export_format}): {exc}")
        return jsonify({"error": "export_failed", "message": str(exc)}), 500

    # Stream the archive FROM DISK (GB-class on a 4 GB broker — never via
    # BytesIO). POSIX unlink-while-open: the callback runs at response
    # finalization, before the WSGI layer streams the wrapped file handle out,
    # and the already-open fd keeps the data alive until the response closes.
    @after_this_request
    def _cleanup_export_zip(response):
        with contextlib.suppress(OSError):
            os.remove(zip_file_path)
        return response

    try:
        return send_file(
            zip_file_path,
            mimetype="application/zip",
            as_attachment=True,
            download_name=f"scan-{scan_id}-{export_format}.zip",
        )
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(zip_file_path)
        raise


# ---------------------------------------------------------------------------
# Isaac Sim / Isaac Lab USD export (``?format=isaac_usd``)
#
# The isaac writer (``server/export/isaac/``) already turns the SAME persisted export geometry
# the GR00T/LeRobot export consumes (world-frame dense cloud + keyframe poses/intrinsics) into a
# metric, gravity-aligned, Z-up USD scene. Two things the offline CLI got from flags, the route
# must supply from the persisted record:
#   * gravity alignment — the writer estimates it itself from the floor plane (no input needed).
#   * a METRIC SCALE — REQUIRED. The route GATES: either ``?scale=<metres-per-SLAM-unit>`` or a
#     persisted Metric-anchor calibration (``derived_latest`` kind ``"anchor"``). Never emit an
#     up-to-scale USD.
# The heavy deps (``pxr``/``usd-core`` to author USD, ``open3d`` for the Poisson collider mesh)
# were an offline concern; they may not be in the broker/export runtime image. They import LAZILY
# and are pre-checked so a missing dep is an honest 501, never a server crash.
# ---------------------------------------------------------------------------

def _is_anchor_derived_key(key: Any) -> bool:
    """True iff ``key`` names a Metric-anchor derived artifact (``derived/anchor/<stamp>/...``) —
    i.e. geometry that is ALREADY metric (the anchor route scaled it), so an Isaac export reading
    it must apply scale ``1.0`` (never re-scale). ``_DERIVED_KEY_PREFIX`` is referenced at call
    time (it is defined further down the module)."""
    return bool(key) and str(key).startswith(f"{_DERIVED_KEY_PREFIX}/anchor/")


# Ceiling on the point count the Isaac USD exporter will attempt, because the broker has
# 4 GB and Modal caps a web request at 150 s. The founder's 63.3M-point scene exhausted
# memory ~5 s into loading and returned a bare 500; Poisson meshing it ran past 900 s even
# on 8 GB. Above this we refuse with the number rather than attempt it. Both real scenes we
# ship (18.8M canonical, 17.2M TUM) sit comfortably under the default.
ISAAC_MAX_POINTS = int(os.environ.get("ISAAC_MAX_POINTS", str(25_000_000)))


def _isaac_dependency_status() -> list[str]:
    """Return the pip package names the Isaac USD writer needs but that are NOT importable in this
    deployment (empty list = all present). ``usd-core`` (``pxr``) authors the USD stages;
    ``open3d`` builds the Poisson collider mesh. Uses ``find_spec`` so nothing heavy is imported.
    Single chokepoint: the route 501s ``isaac_unavailable`` when this is non-empty, and tests
    monkeypatch it to simulate a bare runtime image."""
    import importlib.util

    missing: list[str] = []
    if importlib.util.find_spec("pxr") is None:
        missing.append("usd-core")
    if importlib.util.find_spec("open3d") is None:
        missing.append("open3d")
    return missing


def _parse_isaac_scale_arg(raw: Any) -> Optional[float]:
    """Parse ``?scale=`` — a user-supplied SLAM-units->metres factor. ``None`` when absent/blank.
    Raises ``ValueError`` on a non-numeric or non-positive value (a bad gate input -> 400)."""
    if raw is None or str(raw).strip() == "":
        return None
    val = float(raw)  # ValueError propagates -> 400 at the route
    if not (np.isfinite(val) and val > 0):
        raise ValueError(f"'scale' must be a positive finite number, got {raw!r}")
    return val


def _resolve_isaac_scale(
    record: dict[str, Any], source: Any, scale_arg: Optional[float]
) -> Optional[tuple]:
    """Resolve (geometry source, scale, scale_source label, anchor scale_factor) for an Isaac
    export so the geometry fed to the writer * scale == metres, NEVER double-scaled; or ``None``
    when no metric scale is available (-> the route returns 409 ``metric_scale_required``).

    Precedence:
      * explicit ``?scale`` -> apply it to the ORIGINAL (up-to-scale) geometry (``user:factor``).
      * else a Metric anchor -> read its ALREADY-METRIC calibrated geometry, scale ``1.0``
        (``anchor:prescaled``). The anchor is either an explicit ``?source=derived/anchor/...`` or
        the scan's ``derived_latest`` pointer of kind ``"anchor"``.
      * else -> ``None`` (gate fail).
    """
    if scale_arg is not None:
        return (EXPORT_SOURCE_ORIGINAL, scale_arg, "user:factor", None)

    s = str(source).strip() if source is not None else ""
    if s and s != EXPORT_SOURCE_ORIGINAL and _is_anchor_derived_key(s):
        # Explicit anchor-calibrated source: geometry already metric.
        return (s, 1.0, "anchor:prescaled", None)

    pointer = record.get("derived_latest") if isinstance(record, dict) else None
    if (
        isinstance(pointer, dict)
        and pointer.get("kind") == "anchor"
        and pointer.get("source_key")
    ):
        return (str(pointer["source_key"]), 1.0, "anchor:prescaled", pointer.get("scale_factor"))

    return None


def _build_isaac_zip(
    record: dict[str, Any],
    scan_id: str,
    *,
    scale: float,
    scale_source: str,
    anchor_scale: Optional[float] = None,
) -> bytes:
    """Author the Isaac USD tree from a normalized Stage-4 record and return it zipped, in memory
    (``<scan_id>/isaac/{scene.usd,trajectory.usd,manifest.json}``, like the CLI writes). GPU-free;
    ``export_isaac_from_record`` lazy-imports ``pxr``/``open3d`` (a missing one raises
    ``ImportError`` -> mapped to 501 at the route). ``require_metric=True`` refuses to write an
    up-to-scale scene as defense-in-depth behind the route's gate."""
    from server.export.isaac.writer import export_isaac_from_record

    with tempfile.TemporaryDirectory(prefix="openreality_isaac_") as tmp:
        isaac_dir = export_isaac_from_record(
            record,
            tmp,
            scan_id=scan_id,
            scale=scale,
            scale_source=scale_source,
            anchor_scale=anchor_scale,
            require_metric=True,
        )
        if not isaac_dir:
            raise RuntimeError("record had no geometry to author an Isaac scene from")
        # Archive the <scan_id>/ subtree so the zip's top-level entry is the scan id
        # (entries: <scan_id>/isaac/{scene.usd,trajectory.usd,manifest.json}).
        zip_path = shutil.make_archive(
            os.path.join(tmp, "archive"), "zip", root_dir=tmp, base_dir=scan_id
        )
        with open(zip_path, "rb") as fh:
            return fh.read()


def _export_isaac_usd(user_id: str, scan_id: str, record: dict[str, Any], source: Any):
    """Handle ``GET /api/scenes/<id>/export?format=isaac_usd``. Gates on a metric scale (409
    ``metric_scale_required``) then on the heavy USD deps (501 ``isaac_unavailable``), then streams
    the zip. The metric gate runs FIRST so it is decidable (and testable) without ``pxr``/``open3d``
    installed. NON-DESTRUCTIVE: only reads the persisted / derived geometry."""
    # 1) metric-scale gate (dep-free). Bad ``?scale`` value -> 400.
    try:
        scale_arg = _parse_isaac_scale_arg(request.args.get("scale"))
    except (ValueError, TypeError) as exc:
        return jsonify({"error": "invalid_request", "message": str(exc)}), 400

    resolved = _resolve_isaac_scale(record, source, scale_arg)
    if resolved is None:
        return (
            jsonify(
                {
                    "error": "metric_scale_required",
                    "message": (
                        "Run Metric anchor first — Isaac needs a metric scale. "
                        "Alternatively pass ?scale=<metres-per-SLAM-unit>."
                    ),
                }
            ),
            409,
        )
    isaac_source, isaac_scale, scale_source_label, anchor_scale = resolved

    # 2) size gate. The broker runs with 4 GB and Modal caps a web request at 150 s, so a
    # huge cloud cannot succeed here no matter what: loading the founder's 63,308,835-point
    # scene exhausted memory ~5 s in, and because the load sits OUTSIDE the try/except below
    # that surfaced as a bare 500 with no message and no `[isaac]` log line. Poisson meshing
    # afterwards ran past 900 s even on an 8 GB container. Refuse with the number, the same
    # doctrine as the viewer's DIRECT_LOAD_MAX_BYTES: a request that cannot possibly succeed
    # is answered honestly, never attempted.
    point_count = record.get("point_count")
    if isinstance(point_count, int) and point_count > ISAAC_MAX_POINTS:
        return (
            jsonify(
                {
                    "error": "scene_too_large",
                    "message": (
                        f"this scene has {point_count:,} points — above the "
                        f"{ISAAC_MAX_POINTS:,} the Isaac exporter can mesh inside the "
                        "broker's memory and request budget. Export a decimated scene, or "
                        "run the export offline."
                    ),
                    "point_count": point_count,
                    "limit": ISAAC_MAX_POINTS,
                }
            ),
            413,
        )

    # 3) heavy-dependency gate. A bare runtime image can't author USD -> honest 501, never a crash.
    missing = _isaac_dependency_status()
    if missing:
        return (
            jsonify(
                {
                    "error": "isaac_unavailable",
                    "message": (
                        f"Isaac USD export needs {', '.join(missing)} which "
                        f"{'is' if len(missing) == 1 else 'are'} not installed in this deployment."
                    ),
                }
            ),
            501,
        )

    # 4) load the export geometry (record path, honoring the resolved source), then author + zip.
    # Guarded: this used to be bare, so any failure here became a 500 with nothing to act on.
    try:
        normalized = load_record_from_store(_scene_persistence, user_id, scan_id, source=isaac_source)
    except MemoryError:
        print(f"[isaac] scan {scan_id} ran out of memory loading {point_count} points")
        return (
            jsonify(
                {
                    "error": "scene_too_large",
                    "message": (
                        "ran out of memory loading this scene's geometry for export. "
                        "Export a decimated scene, or run the export offline."
                    ),
                }
            ),
            413,
        )
    except Exception as exc:
        traceback.print_exc()
        print(f"[isaac] scan {scan_id} failed loading export geometry: {exc}")
        return jsonify({"error": "export_failed", "message": str(exc)}), 500
    if normalized is None:
        return (
            jsonify(
                {
                    "error": "no_trajectory",
                    "message": (
                        "scan has no persisted per-keyframe trajectory (pre-Stage-4 scan); "
                        "cannot export Isaac USD GPU-free"
                    ),
                }
            ),
            404,
        )
    cloud = normalized.get("cloud")
    if not cloud or cloud[0] is None or len(cloud[0]) == 0:
        return (
            jsonify(
                {
                    "error": "no_points",
                    "message": "scan has no stored point cloud to build an Isaac scene from",
                }
            ),
            404,
        )

    try:
        zip_bytes = _build_isaac_zip(
            normalized,
            scan_id,
            scale=isaac_scale,
            scale_source=scale_source_label,
            anchor_scale=anchor_scale,
        )
    except ImportError as exc:
        # A lazy dep vanished between the precheck and the write — degrade to the same honest 501.
        print(f"[isaac] scan {scan_id} export dep import failed: {exc}")
        return (
            jsonify(
                {
                    "error": "isaac_unavailable",
                    "message": f"Isaac USD dependency unavailable at runtime: {exc}",
                }
            ),
            501,
        )
    except Exception as exc:
        print(f"[isaac] scan {scan_id} isaac_usd export failed: {exc}")
        return jsonify({"error": "export_failed", "message": str(exc)}), 500

    return send_file(
        io.BytesIO(zip_bytes),
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"scan-{scan_id}-isaac_usd.zip",
    )


# --- metric anchor -------------------------------------------------------------------
# The implementation lives in ``server/scene_report/anchor.py`` so a Modal container with no
# Flask (and no way to mint a 60-second Clerk token) can apply an anchor through the SAME
# code the route runs -- see ``modal_tum_depth_anchor.py``. Aliased under the historical
# private names: the route bodies below, and the tests that call them, are unchanged.
_validate_anchor_point = _anchor_impl.validate_anchor_point
_scale_factor_from_known_distance = _anchor_impl.scale_factor_from_known_distance
_cloud_extent = _anchor_impl.cloud_extent
_derived_pointer = _anchor_impl.derived_pointer
_persist_derived_pointer = _anchor_impl.persist_derived_pointer
_apply_metric_anchor = _anchor_impl.apply_metric_anchor

# Injectable spawner for the anchor materialization job (routes_lod.py pattern —
# tests pass a recording fake; ``None`` restores the real deployed-function lookup).
_anchor_job_spawner = None


def configure_anchor_job_spawner(spawner) -> None:
    global _anchor_job_spawner
    _anchor_job_spawner = spawner


def _spawn_anchor_job(**kwargs) -> None:
    """Spawn ``demo_anchor_job`` on the deployed app (fire-and-forget; status flows
    through the shared demo-jobs Dict)."""
    if _anchor_job_spawner is not None:
        _anchor_job_spawner(**kwargs)
        return
    import modal

    modal.Function.from_name("vggt-slam-streaming", "demo_anchor_job").spawn(**kwargs)


@app.route("/api/scenes/<scan_id>/anchor", methods=["POST"])
def anchor_scene_route(scan_id):
    """Metric-anchor calibration (CAL-2 — see ``vggt_slam.metric_absolute``, core,
    read as a spec): body ``{point_a:[x,y,z], point_b:[x,y,z], distance_m:number}`` —
    two points already picked in the scene's current (up-to-scale) world frame, plus the
    real-world distance between them. ``scale_factor = distance_m / measured``.

    NON-DESTRUCTIVE: writes calibrated copies of the cloud (and, when present, the
    trajectory / splat) under NEW ``derived/anchor/...`` keys. ``cloud.npz`` / ``splat.ply``
    are never touched — live pilot embeds keep shipping from the originals.
    """
    if _scene_persistence is None:
        return jsonify({"error": "not_found"}), 404
    user_id = _auth_user_id()
    if not user_id:
        return _auth_response(AuthError("invalid_token"))
    record = _scene_persistence.get_scene(user_id, scan_id)
    if not record:
        return jsonify({"error": "not_found"}), 404

    data = request.get_json(silent=True) or {}

    # Async form (``{"materialize": "job"}``): the scale factor is pure arithmetic, and
    # metres are display math — the viewer keeps original-gauge geometry and overlays
    # (see modal_oreos_lod.py's gauge note). Only the calibrated derived COPIES are heavy
    # (GB-class splat rewrite), so those move to ``demo_anchor_job`` while the response
    # returns instantly with the factor and a pollable job id. The pending
    # ``derived_latest`` pointer flips the UI to metres now; the Isaac gate stays closed
    # until the job upgrades the pointer with real keys (it requires a truthy source_key).
    if str(data.get("materialize") or "").strip() == "job":
        try:
            if data.get("distance_m") is None:
                raise ValueError("'distance_m' is required")
            distance_m = float(data.get("distance_m"))
            measured, scale_factor = _anchor_impl.compute_anchor_scale(
                data.get("point_a"), data.get("point_b"), distance_m
            )
        except (ValueError, TypeError) as exc:
            return jsonify({"error": "invalid_request", "message": str(exc)}), 400
        point_count = record.get("point_count")
        if isinstance(point_count, (int, float)) and point_count <= 0:
            return (
                jsonify({"error": "no_points", "message": "scan has no stored point cloud to calibrate"}),
                404,
            )
        applied_at = datetime.now(timezone.utc).isoformat()
        job_id = uuid.uuid4().hex[:12]
        try:
            from server.oreos.jobs import _get_jobs_store, job_status_record

            store = _get_jobs_store()
            if store is not None:
                store[job_id] = job_status_record(
                    job_id, user_id, status="queued", stage="queued", scan_id=scan_id, kind="anchor"
                )
        except Exception as exc:  # status record is best-effort; the job re-publishes
            print(f"[anchor] job pre-record failed for {scan_id}: {exc}")
        _persist_derived_pointer(
            _scene_persistence,
            user_id,
            scan_id,
            _anchor_impl.pending_anchor_pointer(scale_factor, applied_at, job_id),
        )
        try:
            _spawn_anchor_job(
                job_id=job_id,
                user_id=user_id,
                scan_id=scan_id,
                scale_factor=scale_factor,
                applied_at=applied_at,
                measured=measured,
                distance_m=distance_m,
            )
        except Exception as exc:
            # The pointer is already honest (pending, factor recorded); tell the client
            # the materialization could not start rather than pretending it did.
            print(f"[anchor] job spawn failed for {scan_id}: {exc}")
            return (
                jsonify({"error": "spawn_failed", "message": "could not start the anchor job"}),
                503,
            )
        return (
            jsonify(
                {
                    "scale_factor": scale_factor,
                    "measured_distance": measured,
                    "distance_m": distance_m,
                    "applied_at": applied_at,
                    "pending": True,
                    "job_id": job_id,
                    "gauge_span_before": measured,
                    "gauge_span_after_m": measured * scale_factor,
                }
            ),
            202,
        )

    try:
        if data.get("distance_m") is None:
            raise ValueError("'distance_m' is required")
        result = _apply_metric_anchor(
            _scene_persistence,
            user_id,
            scan_id,
            data.get("point_a"),
            data.get("point_b"),
            float(data.get("distance_m")),
        )
    except (ValueError, TypeError) as exc:
        return jsonify({"error": "invalid_request", "message": str(exc)}), 400
    except KeyError as exc:
        if str(exc).strip("'\"") == "no_geometry":
            return (
                jsonify({"error": "no_points", "message": "scan has no stored point cloud to calibrate"}),
                404,
            )
        raise
    # Record this as the scan's latest calibrated artifact so a later default (no-selector) export
    # picks it up and the workflow UI can offer/preview it. Metadata-only; originals untouched.
    _persist_derived_pointer(
        _scene_persistence,
        user_id,
        scan_id,
        _derived_pointer(
            "anchor",
            cloud_key=result.get("calibrated_cloud_key"),
            trajectory_key=result.get("calibrated_trajectory_key"),
            splat_key=result.get("calibrated_splat_key"),
            applied_at=result.get("applied_at"),
            scale_factor=result.get("scale_factor"),
        ),
    )
    return jsonify(result)


@app.route("/api/scenes/<scan_id>/clamp", methods=["POST"])
def clamp_scene_route(scan_id):
    """Auto-clamp: pull the top-percentile per-gaussian scale tail down (default p99) to
    remove 3DGS render spikes — see ``vggt_slam.splat_export.clamp_scales_percentile``
    (core, read as a spec). Optional JSON body ``{"percentile": number}`` overrides the
    default (0, 100) exclusive.

    NON-DESTRUCTIVE: writes a clamped copy of the splat under a NEW
    ``derived/clamp/...`` key. ``splat.ply`` is never touched.
    """
    if _scene_persistence is None:
        return jsonify({"error": "not_found"}), 404
    user_id = _auth_user_id()
    if not user_id:
        return _auth_response(AuthError("invalid_token"))
    record = _scene_persistence.get_scene(user_id, scan_id)
    if not record:
        return jsonify({"error": "not_found"}), 404

    get_splat_path = getattr(_scene_persistence, "get_splat_path", None)
    splat_path = get_splat_path(user_id, scan_id) if callable(get_splat_path) else None
    if not splat_path:
        return jsonify({"error": "no_splat", "message": "scan has no 3DGS splat artifact"}), 404

    data = request.get_json(silent=True) or {}
    percentile = data.get("percentile", DEFAULT_SCALE_CLAMP_PERCENTILE)
    try:
        percentile = float(percentile)
    except (TypeError, ValueError):
        return (
            jsonify({"error": "invalid_request", "message": f"'percentile' must be a number, got {percentile!r}"}),
            400,
        )

    try:
        clamped_fields, info = clamp_splat_fields(read_splat_ply(splat_path), percentile)
    except ValueError as exc:
        return jsonify({"error": "invalid_splat", "message": str(exc)}), 500

    stamp = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
    derived_key = _scene_persistence.save_derived_artifact(
        user_id, scan_id, f"clamp/{stamp}/splat.ply", serialize_splat_ply(clamped_fields)
    )

    # Latest calibrated pointer (metadata-only; see the anchor route). A clamp group carries only
    # a splat -> a later default export uses this clamped splat + the ORIGINAL cloud/trajectory.
    _persist_derived_pointer(
        _scene_persistence,
        user_id,
        scan_id,
        _derived_pointer("clamp", splat_key=derived_key),
    )

    return jsonify(
        {
            "clamped_gaussian_count": info["clamped_gaussian_count"],
            "scale_clamp_percentile": info["scale_clamp_percentile"],
            # NOTE: named to match the platform-agreed contract for this route; it holds the
            # DERIVED SPLAT's key (not a point cloud) -- see docs/dataset-export.md / the route
            # docstring above. `derived_splat_key` below is the same value under a clearer name.
            "cloud_key": derived_key,
            "derived_splat_key": derived_key,
            "max_scale_before": info["max_scale_before"],
            "max_scale_after": info["max_scale_after"],
            "gaussian_count": info["gaussian_count"],
        }
    )


# The store namespaces every derived (non-destructive) artifact under this key prefix
# (``ModalScenePersistence._DERIVED_ROOT``). The read route below rebuilds the full key from it.
_DERIVED_KEY_PREFIX = "derived"


@app.route("/api/scenes/<scan_id>/derived/<path:derived_key>", methods=["GET"])
def get_scene_derived_route(scan_id, derived_key):
    """Stream a NON-DESTRUCTIVE derived artifact back to the owner — the calibrated/clamped
    ``derived/...`` cloud or splat that the anchor/clamp Refine routes produced — so the workflow
    viewer can render a true before/after. Read-only: never writes, never mutates an original.

    ``<derived_key>`` is the artifact key WITHOUT the ``derived/`` prefix (e.g.
    ``anchor/<stamp>/cloud.ply``); a leading ``derived/`` is tolerated too. It's validated against
    THIS scan's derived dir via the store's path resolver — a foreign/unknown key, or any
    ``..`` traversal attempt, resolves to nothing and is a 404 (same guard as the export
    ``?source=`` selector). Owner-authed via header or ``?auth_token=`` (so the Gaussian-splat URL
    loader, which can't set headers, can fetch a clamped splat). A read-only share token may also
    reach it, but ONLY for keys under ``demo/lod/`` (the fast-preview splats the share embed
    renders) — enforced in ``_try_share_token_auth`` before this body runs."""
    if _scene_persistence is None:
        return jsonify({"error": "not_found"}), 404
    user_id = _auth_user_id()
    if not user_id:
        return _auth_response(AuthError("invalid_token"))
    record = _scene_persistence.get_scene(user_id, scan_id)
    if not record:
        return jsonify({"error": "not_found"}), 404

    get_derived_path = getattr(_scene_persistence, "get_derived_artifact_path", None)
    if not callable(get_derived_path):
        return jsonify({"error": "not_found"}), 404
    rel = str(derived_key).strip("/")
    if rel.startswith(_DERIVED_KEY_PREFIX + "/"):
        rel = rel[len(_DERIVED_KEY_PREFIX) + 1:]
    full_key = f"{_DERIVED_KEY_PREFIX}/{rel}"
    path = get_derived_path(user_id, scan_id, full_key)
    if not path:
        return (
            jsonify({"error": "not_found", "message": "no such derived artifact for this scan"}),
            404,
        )
    return send_file(
        path,
        mimetype="application/octet-stream",
        as_attachment=False,
        download_name=os.path.basename(path),
        conditional=True,  # honor Range/If-Range so a big derived splat resumes + seeks
    )


def _stem(word: str) -> str:
    """Very light suffix stripping so plurals/gerunds collapse ('couches'→'couch',
    'plants'→'plant') without pulling in a real stemmer. Intentionally crude."""
    w = word.lower()
    if w.endswith("ss"):  # 'glass', 'class' — don't strip the trailing s
        return w
    for suf, repl in (("ies", "y"), ("ing", ""), ("es", ""), ("ed", ""), ("s", "")):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: len(w) - len(suf)] + repl
    return w


def _terms(text: Any, min_len: int = 3) -> set[str]:
    """Tokenize to a set of stemmed terms, dropping short stopword-ish tokens."""
    return {
        _stem(t)
        for t in re.split(r"[^a-z0-9]+", str(text).lower())
        if len(t) >= min_len
    }


def _substr_match(a: str, b: str) -> bool:
    """Directional substring overlap with a length guard, so 'whiteboard'~'board' or
    'tabletop'~'table' match but trivial fragments don't."""
    if len(a) >= 4 and a in b:
        return True
    return len(b) >= 4 and b in a


def _relation_terms_by_object(relations: Optional[list[dict[str, Any]]]) -> dict[str, set[str]]:
    """For each object name (lowercased), the stemmed terms drawn from relations it
    appears in — its partner's name + the qualifier — so 'what's next to the couch' can
    re-rank the couch's neighbours."""
    out: dict[str, set[str]] = {}
    for r in relations or []:
        if not isinstance(r, dict):
            continue
        a = str(r.get("a", "")).strip().lower()
        b = str(r.get("b", "")).strip().lower()
        qual = _stem(str(r.get("relation", "")).strip()) if r.get("relation") else None
        for name, partner in ((a, b), (b, a)):
            if not name:
                continue
            terms = out.setdefault(name, set())
            terms |= _terms(partner)
            if qual:
                terms.add(qual)
    return out


def _report_terms_by_object(
    report: Optional[dict[str, Any]], objects: list[dict[str, Any]]
) -> dict[str, set[str]]:
    """For each object name, the stemmed terms of any report summary/observation/
    location text that mentions it — so the LLM's own prose feeds back into ranking."""
    if not isinstance(report, dict):
        return {}
    sentences: list[str] = []
    if report.get("summary"):
        sentences.append(str(report["summary"]))
    sentences.extend(str(o) for o in (report.get("observations") or []))
    for o in report.get("objects") or []:
        if isinstance(o, dict) and o.get("location"):
            sentences.append(f"{o.get('name', '')} {o.get('location')}")
    names = [str(o.get("query", "")).strip().lower() for o in objects]
    out: dict[str, set[str]] = {}
    for sent in sentences:
        low = sent.lower()
        sent_terms = _terms(sent)
        for name in names:
            if name and name in low:
                out.setdefault(name, set()).update(sent_terms)
    return out


_NAME_WEIGHT = 3.0
_NAME_SUBSTR_WEIGHT = 1.5
_CONTEXT_WEIGHT = 1.0
_CONTEXT_SUBSTR_WEIGHT = 0.5


def _lexical_relevance_score(
    q_terms: set[str], name_terms: set[str], context_terms: set[str]
) -> float:
    """Weighted, stem/substring-aware overlap of the question against an object's NAME
    (strongest) and its relation/report CONTEXT (weaker). Replaces the old exact
    set-intersection so non-exact phrasings still score."""
    score = 0.0
    for qt in q_terms:
        if qt in name_terms:
            score += _NAME_WEIGHT
        elif any(_substr_match(qt, nt) for nt in name_terms):
            score += _NAME_SUBSTR_WEIGHT
        elif qt in context_terms:
            score += _CONTEXT_WEIGHT
        elif any(_substr_match(qt, ct) for ct in context_terms):
            score += _CONTEXT_SUBSTR_WEIGHT
    return score


def _object_embed_text(
    obj: dict[str, Any], rel_terms: dict[str, set[str]], report_terms: dict[str, set[str]]
) -> str:
    """A short description of an object for semantic similarity: its name plus any
    related/report context terms we already gathered."""
    name = str(obj.get("query", "")).strip()
    key = name.lower()
    extra = sorted((rel_terms.get(key, set()) | report_terms.get(key, set())) - {key})
    return name if not extra else f"{name} ({', '.join(extra)})"


def _cosine_sim(a: Any, b: Any) -> float:
    va = np.asarray(a, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64)
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def _embedding_similarities(
    question: str, objects: list[dict[str, Any]], texts: list[str]
) -> Optional[dict[int, float]]:
    """Question↔object cosine similarities for the ranking tie-break, in ONE batched
    embeddings request (reuses the assistant's OpenRouter key). Keyed by ``id(obj)``.
    Returns None on any failure / unexpected shape so the caller degrades to the
    static-confidence tie-break."""
    if not objects or len(objects) > SCENE_QA_EMBED_MAX_OBJECTS:
        return None
    try:
        vecs = _get_assistant_client().embed([question] + texts, SCENE_QA_EMBED_MODEL)
    except Exception as e:
        print(f"Scene QA embedding tie-break unavailable ({e}); using confidence.")
        return None
    if not vecs or len(vecs) != len(objects) + 1:
        return None
    qv = vecs[0]
    return {id(obj): _cosine_sim(qv, vecs[i + 1]) for i, obj in enumerate(objects)}


def _relevant_scene_objects(
    objects: list[dict[str, Any]],
    question: str,
    relations: Optional[list[dict[str, Any]]] = None,
    report: Optional[dict[str, Any]] = None,
    sim_by_id: Optional[dict[int, float]] = None,
) -> list[dict[str, Any]]:
    """Rank persisted objects by how well they match the question — a weighted,
    stem/substring-aware overlap against each object's NAME and its relation/report
    CONTEXT, so non-exact phrasings still re-rank (not just exact name words). Ties
    (notably the common all-zero, no-lexical-match case) break by question↔object
    embedding similarity when available (``sim_by_id``), else by confidence, so we
    always have a sensible order to ground on."""
    q_terms = _terms(question)
    rel_terms = _relation_terms_by_object(relations)
    report_terms = _report_terms_by_object(report, objects)

    def lexical(obj: dict[str, Any]) -> float:
        key = str(obj.get("query", "")).strip().lower()
        context = rel_terms.get(key, set()) | report_terms.get(key, set())
        return _lexical_relevance_score(q_terms, _terms(obj.get("query", "")), context)

    def tiebreak(obj: dict[str, Any]) -> float:
        if sim_by_id is not None and id(obj) in sim_by_id:
            return sim_by_id[id(obj)]
        return float(obj.get("confidence", 0.0) or 0.0)

    ranked = sorted(objects, key=lambda o: (lexical(o), tiebreak(o)), reverse=True)
    return ranked[:SCENE_QA_MAX_OBJECTS]


def _validate_focus(
    parsed: dict[str, Any], objects: list[dict[str, Any]]
) -> Optional[dict[str, Any]]:
    """Map the LLM's chosen ``focus`` name back to a real inventory object and return
    that object's *stored* world-frame center — we trust our own geometry, never a
    coordinate the model might have invented. Returns None for an empty/unknown name."""
    focus_raw = parsed.get("focus") if isinstance(parsed.get("focus"), dict) else {}
    focus_name = str((focus_raw or {}).get("name", "")).strip().lower()
    if not focus_name:
        return None
    for o in objects:
        if str(o.get("query", "")).strip().lower() == focus_name and o.get("center"):
            return {"name": o.get("query"), "center": o.get("center")}
    return None


def _ref_key(ref: dict[str, Any]) -> tuple[int, int]:
    """Coerce a keyframe ref to a hashable ``(submap_id, frame_idx)`` key, mapping
    anything malformed to the sentinel ``(-1, -1)`` so it can't match a real frame."""
    try:
        return int(ref.get("submap_id", -1)), int(ref.get("frame_idx", -1))
    except (TypeError, ValueError):
        return (-1, -1)


def _candidate_keyframes(relevant: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten the question-relevant objects' evidence frames into a deduped, bounded
    menu the model picks its cited keyframes from. Each entry notes which object it
    ``shows`` so the model — which reasons over the structured inventory — can line a
    frame up with the thing it depicts."""
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for obj in relevant:
        for ev in obj.get("evidence", []) or []:
            if not isinstance(ev, dict):
                continue
            key = _ref_key(ev)
            if key == (-1, -1) or key in seen:
                continue
            seen.add(key)
            candidates.append(
                {"submap_id": key[0], "frame_idx": key[1], "shows": obj.get("query")}
            )
            if len(candidates) >= SCENE_QA_CANDIDATE_IMAGES:
                return candidates
    return candidates


def _select_evidence(
    model_evidence: Any, resolved: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Choose which resolved keyframes to cite back to the UI, honoring the model's own
    choice: keep the cited ``(submap_id, frame_idx)`` refs we actually have bytes for,
    in the model's order, so the shown images track the answer instead of a static
    confidence-ranked top-3. Fall back to the resolved order so a model that cites
    nothing still shows evidence when keyframes exist."""
    by_key = {_ref_key(r): r for r in resolved}
    chosen: list[dict[str, Any]] = []
    taken: set[tuple[int, int]] = set()
    for ev in model_evidence if isinstance(model_evidence, list) else []:
        if not isinstance(ev, dict):
            continue
        key = _ref_key(ev)
        ref = by_key.get(key)
        if ref is not None and key not in taken:
            taken.add(key)
            chosen.append(ref)
        if len(chosen) >= SCENE_QA_MAX_IMAGES:
            break
    if not chosen:
        chosen = resolved[:SCENE_QA_MAX_IMAGES]
    return [
        {"submap_id": r["submap_id"], "frame_idx": r["frame_idx"], "blob_key": r["blob_key"]}
        for r in chosen
    ]


def _qa_metric_grounding(record: Optional[dict[str, Any]], metrics: dict[str, Any]) -> dict[str, Any]:
    """What the Q&A model may say about lengths, derived from the scene's metric state
    (``oreos.measure.resolve_metric``) and its fitted ground frame. Returns ``scale``
    (metres per SLAM unit, 1.0 when relative), the system-prompt clause, the UNITS line
    and the ROOM SIZE line. Honesty doctrine: metres exist ONLY via the metric anchor;
    a floor-to-ceiling height exists ONLY when the floor frame was fitted and a ceiling
    was found; everything else is relative and must be described that way."""
    anchored = False
    scale = 1.0
    wording = "relative units (uncalibrated)"
    scale_source = "none"
    if record is not None:
        try:
            state = _resolve_metric_state(record)
            anchored = bool(state.anchored and state.scale_factor)
            if anchored:
                scale = float(state.scale_factor)
                wording = state.wording
                scale_source = state.scale_source
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[scene_qa] metric state unavailable: {exc}")

    def _m(v: Any) -> Optional[float]:
        return round(float(v) * scale, 2) if isinstance(v, (int, float)) else None

    unit = "m" if anchored else "relative units"
    dims = [_m(d) for d in (metrics.get("dimensions") or []) if isinstance(d, (int, float))]
    parts: list[str] = []
    if dims:
        parts.append(f"bounding box {json.dumps(dims)} {unit}")
    vertical_known = bool(metrics.get("vertical_axis_known"))
    if vertical_known:
        fe = metrics.get("floor_extent")
        if isinstance(fe, (list, tuple)) and len(fe) == 2:
            parts.append(f"floor extent {json.dumps([_m(fe[0]), _m(fe[1])])} {unit}")
        fa = metrics.get("floor_area")
        if isinstance(fa, (int, float)):
            # area scales with the SQUARE of the length factor
            parts.append(f"floor area {round(float(fa) * scale * scale, 2)} {'m²' if anchored else 'relative units²'}")
        rh = _m(metrics.get("room_height"))
        if rh is not None:
            parts.append(f"floor-to-ceiling {rh} {unit}")
        else:
            parts.append("ceiling not captured (no floor-to-ceiling height available)")
    else:
        parts.append("floor frame not fitted (no floor extent or ceiling height available)")
    room_size = "; ".join(parts) if parts else "unknown"

    if anchored:
        system_clause = (
            f"A METRIC ANCHOR is applied: every length in the facts is in METRES "
            f"({wording}; scale_source={scale_source}). You MAY state sizes and distances "
            "in metres (and feet/inches alongside) when they follow from the given "
            "centres, extents, pairwise distances or room size — say 'about' and round to "
            "the nearest 0.1 m, and never quote more precision than the inputs carry. "
            "The world frame is not gravity-aligned unless ROOM SIZE gives a floor "
            "extent; describe left/right/near relatively. "
            + ("Only state a floor-to-ceiling height if ROOM SIZE lists one. " if vertical_known else
               "Do not assert a floor-to-ceiling height: the floor frame was not fitted. ")
            + "If the anchor wording says 'estimate', carry that caveat into the answer."
        )
        units_line = f"metres — metric anchor applied ({wording}; scale_source={scale_source})"
    else:
        system_clause = (
            "Coordinates/sizes are RELATIVE UNITS (uncalibrated, not metric) and the world "
            "frame is not gravity-aligned, so describe locations relatively (e.g. 'near "
            "the desk', 'left side of the room'), never assert metric distances or a "
            "floor-to-ceiling height, and if asked for a measurement say plainly that "
            "the scene has no metric anchor yet (one known real-world distance unlocks "
            "metres)."
        )
        units_line = "relative units (uncalibrated) — no metric anchor; do not state metres"
    return {
        "anchored": anchored,
        "scale": scale,
        "system_clause": system_clause,
        "units_line": units_line,
        "room_size": room_size,
    }


def _grounded_scene_answer(
    facts: dict[str, Any],
    report: dict[str, Any],
    question: str,
    history: list[dict[str, Any]],
    image_loader: Optional[
        Callable[[list[dict[str, Any]]], tuple[list[str], list[dict[str, Any]]]]
    ] = None,
    record: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Single grounded-Q&A core, shared by the live summary chat and the persisted
    revisit Q&A so the two surfaces can never drift.

    ``record`` (persisted path only) carries the scene's METRIC STATE: when a metric
    anchor is applied (``derived_latest.kind == "anchor"``, see ``oreos/measure.py``)
    every length handed to the model is converted to metres and the model is told it
    may state sizes/distances in metres; when the floor frame has been fitted
    (``metrics.vertical_axis_known``) the floor extent / room height are given too.
    Without an anchor the model is told lengths are relative units and must not assert
    metric numbers — the same honesty doctrine the measure tools follow.

    Grounds the LLM in the scan's STRUCTURED 3D facts — object inventory with
    world-frame coordinates, pairwise relations, room type + extents — plus the report
    prose and (optionally) a labeled menu of candidate keyframes. ``image_loader`` is
    the *only* surface-specific piece: given the candidate frame refs (built here from
    the relevant objects' evidence) it fetches ``(images_b64, resolved_refs)`` aligned
    in order; pass ``None`` when no keyframes are available (the live summary path,
    whose cache holds no blobs). The model **cites its own evidence** from that menu and
    we surface exactly those frames (``_select_evidence``), so the shown thumbnails
    track the question rather than a fixed top-3. The model's fly-to ``focus`` is
    validated against the real inventory. Returns the canonical
    ``{answer, model, degraded, focus, evidence}`` shape both frontends consume."""
    objects = [o for o in (facts.get("objects") or []) if isinstance(o, dict)]
    relations = facts.get("relations") if isinstance(facts.get("relations"), list) else []

    # Optional semantic tie-break (off by default): embed the question + each object's
    # name/context once and let cosine similarity order the objects relevance ties —
    # most useful for questions with no exact word match. Degrades to confidence on any
    # failure (None similarities).
    sim_by_id = None
    if SCENE_QA_EMBED_TIEBREAK and objects:
        rel_terms = _relation_terms_by_object(relations)
        report_terms = _report_terms_by_object(report, objects)
        texts = [_object_embed_text(o, rel_terms, report_terms) for o in objects]
        sim_by_id = _embedding_similarities(question, objects, texts)

    relevant = _relevant_scene_objects(objects, question, relations, report, sim_by_id)

    # Hand the model a bounded, labeled menu of the relevant objects' keyframes and let
    # it cite the ones that support its answer, so the surfaced images follow the
    # question instead of a static confidence-ranked top-3 (see scene-report.md).
    candidates = _candidate_keyframes(relevant)
    images_b64: list[str] = []
    resolved: list[dict[str, Any]] = []
    if image_loader is not None and candidates:
        images_b64, resolved = image_loader(candidates)
    shows_by_key = {_ref_key(c): c.get("shows") for c in candidates}
    catalog = [
        {
            "submap_id": r["submap_id"],
            "frame_idx": r["frame_idx"],
            "shows": shows_by_key.get(_ref_key(r)),
        }
        for r in resolved
    ]

    metrics = facts.get("metrics") if isinstance(facts.get("metrics"), dict) else {}
    grounding = _qa_metric_grounding(record, metrics)
    scale = grounding["scale"]  # metres per SLAM unit when anchored, else 1.0 (relative)

    def _len(v: Any) -> Any:
        if isinstance(v, (list, tuple)):
            return [round(float(x) * scale, 2) for x in v if isinstance(x, (int, float))]
        if isinstance(v, (int, float)):
            return round(float(v) * scale, 2)
        return v

    inventory = [
        {
            "name": o.get("query"),
            "center": _len(o.get("center")),
            "extent": _len(o.get("extent")),
            "confidence": round(float(o.get("confidence", 0.0) or 0.0), 3),
        }
        for o in relevant
    ]
    relations_out = [
        {**r, "distance": _len(r.get("distance"))} if isinstance(r, dict) and "distance" in r else r
        for r in relations
    ]
    system_prompt = (
        "You answer questions about a scanned 3D space, grounded in STRUCTURED 3D FACTS "
        "from the reconstruction (object inventory with world-frame coordinates and "
        "extents, pairwise distances, room extents) plus any attached keyframes. "
        + grounding["system_clause"]
        + " If the question is about an object present in the inventory, set \"focus\" to "
        "that exact object name so the 3D view can fly to it. Cite the keyframes that "
        "actually support your answer in \"evidence\", chosen ONLY from the CANDIDATE "
        "KEYFRAMES list (by their submap_id/frame_idx). Never invent objects, "
        "coordinates, or keyframes not given. Output strict JSON only."
    )
    user_prompt = (
        f"UNITS: {grounding['units_line']}\n"
        f"OBJECT INVENTORY (JSON): {json.dumps(inventory)}\n"
        f"ROOM TYPE: {report.get('room_type', 'unknown')}\n"
        f"ROOM SIZE: {grounding['room_size']}\n"
        f"ROOM SUMMARY: {report.get('summary', '')}\n"
        f"OBSERVATIONS: {json.dumps(report.get('observations', []))}\n"
        f"RELATIONS: {json.dumps(relations_out)}\n"
        f"CANDIDATE KEYFRAMES (attached images, in this order; each notes what it shows): "
        f"{json.dumps(catalog)}\n\n"
        f"QUESTION: {question}\n\n"
        "Return JSON exactly: {"
        '"answer":"grounded reply (2-4 sentences)",'
        '"focus":{"name":"<object name from inventory or empty>"},'
        '"evidence":[{"submap_id":0,"frame_idx":0}]'
        "}\n"
        "Pick \"evidence\" only from CANDIDATE KEYFRAMES — the 1-3 that best support your "
        "answer, most relevant first; use [] if none apply."
    )

    client = _get_assistant_client()
    parsed, response = client.chat_json(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        history=history[-6:] if isinstance(history, list) else None,
        images_b64=images_b64 or None,
        temperature=0.3,
        max_tokens=512,
    )

    answer = str(parsed.get("answer", "")).strip()
    return {
        "answer": answer or "I could not find that in the scan.",
        "model": getattr(response, "model", ""),
        "degraded": bool(getattr(response, "degraded", False)),
        "focus": _validate_focus(parsed, objects),
        "evidence": _select_evidence(parsed.get("evidence"), resolved),
    }


def _answer_scene_question(
    user_id: str,
    scan_id: str,
    record: dict[str, Any],
    question: str,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    """Revisit-page adapter (Phase 5): ground in the *persisted* record and resolve the
    candidate keyframes (chosen by the shared core) from its blob store, then defer to
    that core. The loader just fetches bytes — the core builds the candidate menu and
    selects which of them the model actually cited."""
    facts = record.get("facts") or {}
    report = record.get("report") or {}
    manifest = {
        (int(k.get("submap_id", -1)), int(k.get("frame_idx", -1))): k.get("blob_key")
        for k in (record.get("keyframes") or [])
        if isinstance(k, dict)
    }

    def load_images(
        candidates: list[dict[str, Any]],
    ) -> tuple[list[str], list[dict[str, Any]]]:
        """Resolve candidate frame refs to ``(images_b64, resolved_refs)`` kept in
        lockstep order, dropping any frame missing from the manifest or blob store."""
        images: list[str] = []
        refs: list[dict[str, Any]] = []
        for cand in candidates:
            key = _ref_key(cand)
            blob_key = manifest.get(key)
            if not blob_key:
                continue
            blob = _scene_persistence.get_keyframe(user_id, scan_id, blob_key)
            if not blob:
                continue
            images.append(base64.b64encode(blob).decode("ascii"))
            refs.append(
                {
                    "submap_id": key[0],
                    "frame_idx": key[1],
                    "blob_key": blob_key,
                    "shows": cand.get("shows"),
                }
            )
            if len(refs) >= SCENE_QA_CANDIDATE_IMAGES:
                break
        return images, refs

    return _grounded_scene_answer(facts, report, question, history, load_images, record=record)


# ------------------------------
# SocketIO Events
# ------------------------------
async def _enforce_session_timeout(sid: str):
    await asyncio.sleep(SESSION_TIMEOUT_SEC)
    if not _is_sid_connected(sid):
        return
    await sio.emit(
        "session_timeout",
        {"timeout_sec": SESSION_TIMEOUT_SEC},
        to=sid,
    )
    await sio.disconnect(sid)


@sio.on("connect")
async def handle_connect(sid, environ, auth):
    global _stream_task, _event_loop, _result_ready
    if slam_processor is None:
        return False

    try:
        claims = _verify_socket_auth(auth, environ)
    except AuthError:
        return False

    user_id = str(claims.get("sub", "")).strip()
    if not user_id:
        return False

    if _worker_expected_user_id is not None and user_id != _worker_expected_user_id:
        raise SocketIOConnectionRefused(
            {
                "error": "session_busy",
                "message": "This GPU session belongs to another user.",
            }
        )

    if not await _claim_user_session(user_id, sid):
        # Another user already holds this container's single SLAM session.
        # Reject with a reason so the client can surface "GPU busy" instead of
        # silently sharing one global map (cross-user scene leak).
        raise SocketIOConnectionRefused(
            {
                "error": "session_busy",
                "message": "Another user is currently using this GPU session.",
            }
        )

    _note_worker_activity()
    # A reconnect during the salvage grace window continues the scan — void the
    # pending salvage so it can't persist a half-scan under the client's feet.
    _invalidate_pending_salvage()
    if _scan_salvage_task is not None and not _scan_salvage_task.done():
        _scan_salvage_task.cancel()
    _event_loop = asyncio.get_running_loop()
    if _result_ready is None:
        _result_ready = asyncio.Event()
    slam_processor.event_loop = _event_loop
    slam_processor.result_ready_event = _result_ready
    if not result_queue.empty():
        _result_ready.set()

    with _sids_lock:
        _connected_sids.add(sid)

    _ensure_session(sid, user_id=user_id, claims=claims)
    _session_timeout_tasks[sid] = asyncio.create_task(_enforce_session_timeout(sid))

    print(f"Client connected ({len(_connected_sids)} total)")
    client_connected.set()

    if _stream_task is None or _stream_task.done():
        _stream_task = asyncio.ensure_future(stream_results())

    await sio.emit("connected", {"status": "ready"}, to=sid)
    await sio.emit("agent_state", _build_agent_state_payload(sid), to=sid)


@sio.on("disconnect")
async def handle_disconnect(sid):
    global _event_loop, _result_ready, _scan_salvage_task
    if slam_processor is None:
        return

    timeout_task = _session_timeout_tasks.pop(sid, None)
    if timeout_task and not timeout_task.done():
        timeout_task.cancel()

    with _sids_lock:
        _connected_sids.discard(sid)
        remaining = len(_connected_sids)

    with _sessions_lock:
        state = _sessions.pop(sid, None)
        _frame_rate_state.pop(sid, None)
    if state is not None:
        await _release_user_session(state.user_id, sid)
    if state is not None and state.agent is not None:
        try:
            state.agent.shutdown()
        except Exception:
            pass

    # Cancel all per-query detection tasks for this session
    for key in [k for k in _query_tasks if k[0] == sid]:
        task = _query_tasks.pop(key, None)
        if task and not task.done():
            task.cancel()

    print(f"Client disconnected ({remaining} remaining)")
    _note_worker_activity()
    _detection_partial_last_emit.pop(sid, None)

    # Only re-run detection for sessions that still exist. With zero sessions the
    # empty-union refresh used to wipe accumulated_detections AND the CLIP/SAM cache,
    # which (a) emptied the object list the abandoned-scan salvage persists, and
    # (b) made every reconnect re-scan all submaps cold — the exact query-worker
    # storm that starves pings. A fresh claim still wipes via /reset.
    if remaining > 0:
        await _refresh_global_detection_queries(trigger_sid=None, emit_progress=False)

    if remaining == 0:
        with _demo_lock:
            demo_was_active = _demo_video_feeder is not None
        _event_loop = None
        _result_ready = None
        slam_processor.event_loop = None
        slam_processor.result_ready_event = None
        client_connected.clear()
        slam_processor.stop()
        _stop_demo_feeder()
        _clear_queues()

        while not frame_queue.empty():
            try:
                frame_queue.get_nowait()
            except Exception:
                break
        while not result_queue.empty():
            try:
                result_queue.get_nowait()
            except Exception:
                break

        # A vanished last client means stop_slam is never coming — schedule a salvage
        # persist so a dead transport can't lose the scan. Demo playback keeps its
        # existing semantics (closing the tab abandons the demo scan).
        if not demo_was_active and not _scene_report_finalized:
            with _scan_salvage_lock:
                generation = _scan_salvage_generation
            if _scan_salvage_task is not None and not _scan_salvage_task.done():
                _scan_salvage_task.cancel()
            _scan_salvage_task = asyncio.create_task(_salvage_abandoned_scan(generation))


@sio.on("frame")
async def handle_frame(sid, data):
    if slam_processor is None:
        return
    if not isinstance(data, dict):
        return
    img_b64 = data.get("image")
    if not isinstance(img_b64, str) or not img_b64:
        return
    if len(img_b64) > MAX_FRAME_B64_LEN:
        await sio.emit("error", {"error": "frame_too_large"}, to=sid)
        return
    if not _allow_frame_for_sid(sid):
        return
    if not frame_queue.full():
        frame_queue.put(data)
    if not slam_processor.is_running:
        print("Auto-starting SLAM processing...")
        # A new scan after a previous stop: drop the prior report so its finalized
        # flag doesn't suppress this scan's progressive updates (and so get_scene_report
        # doesn't serve the old scan's report), and mint a fresh scan_id for persistence.
        # First-ever start: a no-op beyond minting the id.
        _begin_new_scan()
        slam_processor.start()


@sio.on("stop_slam")
async def handle_stop(sid, data=None):
    if slam_processor is None:
        return
    _stop_demo_feeder()
    # flush=True so a short live scan that never filled a submap still produces one.
    # Runs on the GPU executor to keep the VGGT pass off the event loop; flushing
    # before emitting slam_stopped ensures the final points reach the client first.
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(_gpu_executor, lambda: slam_processor.stop(flush=True))
    await sio.emit('slam_stopped', {'status': 'stopped'}, to=sid)
    # Finalize + build + broadcast the end-of-scan report via the shared, debounced
    # path. Debounce coalesces a demo's stop_slam + /api/demo/stop into one build.
    # wait_for_drain=True, same as both demo stop paths: a sustained live sender
    # (the native app at 10fps with detection on) leaves a deep frame backlog at
    # stop, and building the report while the processing thread still races it
    # kills the build inside its best-effort catch, so nothing ever persists
    # (measured 2026-08-12: every native scan was lost; web scans survived only
    # because a light backlog keeps the race window tiny).
    _trigger_scene_report_threadsafe(wait_for_drain=True)


@sio.on("set_detection_queries")
async def handle_set_detection_queries(sid, data):
    if slam_processor is None:
        return
    if not isinstance(data, dict):
        data = {}
    raw_queries = data.get("queries", [])
    if not isinstance(raw_queries, list):
        raw_queries = []
    queries = _normalize_query_list(raw_queries)
    state = _ensure_session(sid)
    if state.reconstruction_only:
        print(f"[set_detection_queries] ignored in reconstruction-only mode sid={sid[:8]}")
        await _clear_session_detection_state(sid)
        return

    with _sessions_lock:
        old = set(state.manual_queries)
    new = set(queries)

    removed = old - new
    added = new - old

    print(
        f"[set_detection_queries] sid={sid[:8]} "
        f"old={sorted(old)} new={sorted(new)} "
        f"+{sorted(added)} -{sorted(removed)} "
        f"active_tasks={[k[1] for k in _query_tasks if k[0] == sid]}"
    )

    # Cancel tasks and remove detections for removed queries
    for q in removed:
        task = _query_tasks.pop((sid, q), None)
        if task and not task.done():
            task.cancel()
            print(f"  [set_detection_queries] cancelled task for '{q}'")
        slam_processor.remove_query(q)
        print(f"  [set_detection_queries] removed query '{q}' — slam active_queries now: {slam_processor.active_queries}")

    with _sessions_lock:
        state.manual_queries = new

    # Emit immediate update if anything was removed
    if removed:
        active = _session_active_queries(sid)
        accumulated = list(slam_processor.accumulated_detections)
        filtered = _filter_detections_by_queries(accumulated, active)
        print(
            f"  [set_detection_queries] immediate update: "
            f"accumulated={len(accumulated)} filtered={len(filtered)} "
            f"active_queries={active}"
        )
        await sio.emit(
            "detection_partial",
            {
                "detections": _wire_detections(filtered),
                "active_queries": active,
                "is_final": True,
            },
            to=sid,
        )

    # Spawn tasks for newly added queries
    for q in sorted(added):
        print(f"  [set_detection_queries] spawning detection task for '{q}'")
        task = asyncio.create_task(_run_single_query_detection(q, sid))
        _query_tasks[(sid, q)] = task


@sio.on("get_detection_preview")
async def handle_get_detection_preview(sid, data):
    if slam_processor is None:
        return

    if not isinstance(data, dict):
        data = {}
    try:
        submap_id = int(data.get("submap_id"))
        frame_idx = int(data.get("frame_idx"))
    except Exception:
        await sio.emit("detection_preview", {"error": "Invalid submap/frame"}, to=sid)
        return
    query = str(data.get("query", "")).strip()[:MAX_QUERY_LEN]

    try:
        submap = slam_processor.solver.map.get_submap(submap_id)
        if submap is None:
            await sio.emit(
                "detection_preview",
                {"error": f"Submap {submap_id} not found"},
                to=sid,
            )
            return

        frame_tensor = submap.get_frame_at_index(frame_idx)
        frame_np = (frame_tensor.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        frame_pil = Image.fromarray(frame_np)

        keyframe_image = ObjectDetector.image_to_base64(frame_np)

        mask_image = None
        if query:
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                _gpu_executor,
                slam_processor.object_detector.segment_all,
                frame_pil,
                query,
            )
            if results:
                best_mask, _, _ = max(results, key=lambda r: r[2])
                mask_image = ObjectDetector.mask_overlay_to_base64(frame_np, best_mask)

        await sio.emit(
            "detection_preview",
            {
                "query": query,
                "submap_id": submap_id,
                "frame_idx": frame_idx,
                "keyframe_image": keyframe_image,
                "mask_image": mask_image,
            },
            to=sid,
        )

    except Exception as e:
        print(f"Error generating preview: {e}")
        await sio.emit("detection_preview", {"error": str(e)}, to=sid)


@sio.on("describe_detection")
async def handle_describe_detection(sid, data):
    """Fine-grained 'click-in' description of a detected object.

    Crops the object's best keyframe (GPU executor), then runs the VLM enricher off the
    GPU loop (agent executor), and emits ``detection_description``. Cached per (submap,
    frame, query) so re-clicks are free. Best-effort: emits a ``{error}`` payload rather
    than raising."""
    if slam_processor is None:
        return
    enricher = _get_object_enricher()
    if enricher is None:
        await sio.emit("detection_description", {"error": "enrichment_disabled"}, to=sid)
        return
    if getattr(slam_processor, "object_enricher", None) is None:
        slam_processor.set_object_enricher(enricher)

    if not isinstance(data, dict):
        data = {}
    try:
        submap_id = int(data.get("submap_id"))
        frame_idx = int(data.get("frame_idx"))
    except Exception:
        await sio.emit("detection_description", {"error": "Invalid submap/frame"}, to=sid)
        return
    query = str(data.get("query", "")).strip()[:MAX_QUERY_LEN]

    try:
        cached = slam_processor.get_cached_description(submap_id, frame_idx, query)
        if cached is not None:
            await sio.emit(
                "detection_description",
                {"submap_id": submap_id, "frame_idx": frame_idx, "query": query,
                 "description": cached},
                to=sid,
            )
            return

        loop = asyncio.get_event_loop()
        crop_b64, crop_bytes = await loop.run_in_executor(
            _gpu_executor, slam_processor.make_detection_crop, submap_id, frame_idx, query
        )
        if not crop_b64:
            await sio.emit("detection_description", {"error": "crop_failed"}, to=sid)
            return
        crop_url = _make_crop_url(crop_bytes)
        desc = await loop.run_in_executor(
            _agent_executor,
            slam_processor.enrich_from_crop,
            submap_id, frame_idx, query, crop_b64, crop_url,
        )
        if desc is None:
            await sio.emit("detection_description", {"error": "enrich_failed"}, to=sid)
            return
        await sio.emit(
            "detection_description",
            {"submap_id": submap_id, "frame_idx": frame_idx, "query": query,
             "description": desc},
            to=sid,
        )
    except Exception as e:
        print(f"Error describing detection: {e}")
        await sio.emit("detection_description", {"error": str(e)}, to=sid)


@sio.on("place_beacon")
async def handle_place_beacon(sid, data):
    if slam_processor is None:
        return

    beacon_id = data.get("beacon_id")
    frame_number = data.get("frame_number", 0)
    slam_processor.pending_beacons.append({"beacon_id": beacon_id, "frame_number": frame_number})
    print(f"Beacon {beacon_id} queued at frame {frame_number}")
    await sio.emit("beacon_queued", {"beacon_id": beacon_id}, to=sid)


@sio.on("clear_beacons")
async def handle_clear_beacons(sid, data=None):
    if slam_processor is None:
        return
    slam_processor.pending_beacons.clear()
    slam_processor.resolved_beacons.clear()
    print("All beacons cleared")


@sio.on("debug_detect")
async def handle_debug_detect(sid, data):
    if slam_processor is None:
        await sio.emit("debug_detect_results", {"error": "SLAM not initialized"}, to=sid)
        return

    queries = data.get("queries", [])
    clip_thresholds = data.get("clip_thresholds", {})
    sam_thresholds = data.get("sam_thresholds", {})
    top_k = data.get("top_k", None)

    if not queries:
        await sio.emit("debug_detect_results", {"error": "No queries provided"}, to=sid)
        return

    print(f"Debug detect: {queries} (CLIP={clip_thresholds}, SAM={sam_thresholds}, top_k={top_k})")
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            _gpu_executor,
            lambda: slam_processor.debug_detect_full(queries, clip_thresholds, sam_thresholds, top_k),
        )
        await sio.emit("debug_detect_results", result, to=sid)
    except Exception as e:
        print(f"Debug detect error: {e}")
        await sio.emit("debug_detect_results", {"error": str(e)}, to=sid)


@sio.on("get_global_map")
async def handle_get_global_map(sid, data=None):
    if slam_processor is None:
        return

    print("Client requested global map")
    try:
        if slam_processor.solver.map.get_num_submaps() > 0:
            if slam_processor._last_stream_data and slam_processor._last_stream_data.get("type") == "full":
                stream_data = dict(slam_processor._last_stream_data)
            else:
                stream_data = slam_processor.extract_stream_data_full()

            with slam_processor._detection_lock:
                all_detections = list(slam_processor.accumulated_detections)

            active_queries = _session_active_queries(sid)
            stream_data["active_queries"] = active_queries
            stream_data["detections"] = _filter_detections_by_queries(all_detections, active_queries)

            if stream_data and stream_data.get("n_points", 0) > 0:
                await sio.emit("global_map", stream_data, to=sid)
            else:
                empty = slam_processor._empty_data()
                empty["active_queries"] = active_queries
                empty["detections"] = []
                await sio.emit("global_map", empty, to=sid)
        else:
            empty = slam_processor._empty_data()
            empty["active_queries"] = _session_active_queries(sid)
            empty["detections"] = []
            await sio.emit("global_map", empty, to=sid)
    except Exception as e:
        print(f"Error fetching global map: {e}")


# ------------------------------
# Spatial Agent SocketIO Events
# ------------------------------
@sio.on("agent_chat")
async def handle_agent_chat(sid, data):
    state = _ensure_session(sid)
    if state.agent is None:
        return

    message = str(data.get("message", "")).strip()
    if not message:
        return

    await sio.emit(
        "agent_action",
        {
            "id": str(uuid.uuid4())[:8],
            "timestamp": time.time(),
            "action": "request_received",
            "details": "Chat request received; executing.",
        },
        to=sid,
    )
    loop = asyncio.get_event_loop()
    try:
        await asyncio.wait_for(
            loop.run_in_executor(_agent_executor, state.agent.handle_user_message, message),
            timeout=float(os.environ.get("SPATIAL_CHAT_TIMEOUT_S", "25")),
        )
    except asyncio.TimeoutError:
        await sio.emit(
            "agent_task_event",
            {
                "id": str(uuid.uuid4())[:12],
                "timestamp": time.time(),
                "task_type": "orchestrator",
                "name": "chat_request",
                "status": "timed_out",
                "error": "chat request timed out",
            },
            to=sid,
        )
        await sio.emit("agent_state", _build_agent_state_payload(sid), to=sid)


@sio.on("agent_set_goal")
async def handle_agent_set_goal(sid, data):
    state = _ensure_session(sid)
    if state.reconstruction_only:
        print(f"[agent_set_goal] ignored in reconstruction-only mode sid={sid[:8]}")
        await sio.emit("agent_state", _build_agent_state_payload(sid), to=sid)
        return
    if state.agent is None:
        return

    goal = str(data.get("goal", "")).strip()
    initial_queries = [str(q) for q in data.get("initial_queries", []) if q]
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        _agent_executor,
        state.agent.set_initial_context,
        goal,
        initial_queries,
    )


@sio.on("agent_toggle")
async def handle_agent_toggle(sid, data):
    state = _ensure_session(sid)
    if state.agent is None:
        await sio.emit("agent_state", _build_agent_state_payload(sid), to=sid)
        return

    enabled = bool(data.get("enabled", True))
    if state.reconstruction_only:
        enabled = False
    state.agent.enabled = enabled
    await sio.emit("agent_state", _build_agent_state_payload(sid), to=sid)


@sio.on("set_reconstruction_only")
async def handle_set_reconstruction_only(sid, data):
    state = _ensure_session(sid)
    enabled = bool(data.get("enabled", True)) if isinstance(data, dict) else True
    state.reconstruction_only = enabled
    if enabled and state.agent is not None:
        state.agent.enabled = False
        await _clear_session_detection_state(sid)
    await sio.emit("agent_state", _build_agent_state_payload(sid), to=sid)


@sio.on("get_agent_state")
async def handle_get_agent_state(sid, data=None):
    _ensure_session(sid)
    await sio.emit("agent_state", _build_agent_state_payload(sid), to=sid)


@sio.on("get_scene_report")
async def handle_get_scene_report(sid, data=None):
    # Summary page fetches the report on load. Prefer the final end-of-scan report;
    # if the scan is still running, hand back the latest progressive report so a
    # late-opening summary page populates immediately instead of waiting a cycle.
    _ensure_session(sid)
    if _last_scene_report is not None:
        await sio.emit("scene_report_ready", _last_scene_report, to=sid)
        return
    with _progressive_report_lock:
        progressive = _progressive_report
    if progressive is not None:
        await sio.emit("scene_report_update", progressive.model_dump(), to=sid)


@sio.on("agent_ui_result")
async def handle_agent_ui_result(sid, data):
    if not isinstance(data, dict):
        return
    cmd_id = str(data.get("id", "")).strip()
    status = str(data.get("status", "")).strip().lower()
    if not cmd_id or status not in {"ok", "error", "ignored", "timeout"}:
        return

    state = _ensure_session(sid)
    result = {
        "id": cmd_id,
        "status": status,
        "result": data.get("result"),
        "error": data.get("error"),
        "timestamp": time.time(),
    }
    with _sessions_lock:
        state.ui_results.append(result)
        if len(state.ui_results) > 128:
            state.ui_results = state.ui_results[-128:]


# ------------------------------
# Background Streaming Task
# ------------------------------
async def _broadcast_slam_update(result: dict[str, Any]):
    with _sids_lock:
        target_sids = list(_connected_sids)

    detections = result.get("detections", [])

    for sid in target_sids:
        active_queries = _session_active_queries(sid)
        payload = dict(result)
        payload["active_queries"] = active_queries
        payload["detections"] = _wire_detections(
            _filter_detections_by_queries(detections, active_queries)
        )
        await sio.emit("slam_update", payload, to=sid)


def _resolve_agent_submap_id(result: dict[str, Any]) -> Optional[int]:
    if slam_processor is None:
        return None

    graph_map = slam_processor.solver.map
    raw_submap_id = result.get("submap_id")
    if raw_submap_id is not None:
        try:
            submap_id = int(raw_submap_id)
            graph_map.get_submap(submap_id)
            return submap_id
        except Exception:
            pass

    candidates: list[Optional[int]] = []
    try:
        candidates.append(graph_map.get_largest_key(ignore_loop_closure_submaps=True))
    except Exception:
        candidates.append(None)
    try:
        candidates.append(graph_map.get_largest_key())
    except Exception:
        candidates.append(None)

    for candidate in candidates:
        if candidate is None:
            continue
        try:
            graph_map.get_submap(candidate)
            return int(candidate)
        except Exception:
            continue

    return None


def _schedule_agent_cycles_for_result(result: dict[str, Any]):
    with _sessions_lock:
        states = list(_sessions.values())

    if not states:
        return

    submap_id = _resolve_agent_submap_id(result)
    if submap_id is None:
        return

    detections_all = result.get("detections", [])
    loop = asyncio.get_running_loop()

    for state in states:
        if state.reconstruction_only or state.agent is None or not state.agent.enabled:
            continue
        active = sorted(state.manual_queries | state.agent_queries)
        filtered = _filter_detections_by_queries(detections_all, active)
        loop.run_in_executor(_agent_executor, state.agent.on_submap_processed, submap_id, filtered)


async def stream_results():
    """Stream results to connected clients."""
    while True:
        try:
            if not client_connected.is_set():
                await asyncio.sleep(5.0)
                continue

            result_ready = _result_ready
            if result_ready is None:
                await asyncio.sleep(5.0)
                continue

            await result_ready.wait()
            result_ready.clear()

            while client_connected.is_set():
                try:
                    result = result_queue.get_nowait()
                    put_time = result.pop("_put_time", None)
                    if put_time is not None:
                        lag_ms = (time.perf_counter() - put_time) * 1000.0
                        print(f"[latency] broadcast_lag={lag_ms:.0f}ms")
                    await _broadcast_slam_update(result)
                    _schedule_agent_cycles_for_result(result)
                    _maybe_dispatch_progressive_report(result)
                    print(
                        f"Sent update: {result['n_points']} points, "
                        f"{result['n_cameras']} cameras, "
                        f"{result['num_submaps']} submaps"
                    )
                except queue.Empty:
                    break
        except Exception as e:
            print(f"Stream emit error: {e}")
            await asyncio.sleep(1.0)


# Demo-surface blueprint (docs/demo-2026-07/design/shell.md §4): exactly ONE line lands in
# app.py — all demo routes/logic live in server/server/oreos/ (register only; owned by W0).
from server.oreos import oreos_bp; app.register_blueprint(oreos_bp)  # noqa: E402,E702


# ------------------------------
# Server Startup
# ------------------------------
_frontend_served = False


def serve_frontend(static_dir: Optional[str]) -> None:
    """Register the static SPA routes (idempotent). Called by initialize() on the GPU
    worker and directly by the broker web() so it can serve the revisit page + its
    assets with no GPU. The ``/<path:path>`` catch-all is the least-specific rule, so
    it never shadows ``/api/*`` or ``/session``."""
    global _frontend_served
    if _frontend_served or not static_dir:
        return
    _frontend_served = True
    from flask import send_from_directory

    @app.route("/")
    def serve_index():
        return send_from_directory(static_dir, "viewer.html")

    @app.route("/os")
    def serve_os():
        # Product alias: the Open Reality OS surface ships as demo.html (the file name
        # is a persisted contract — derived/demo/* keys and route paths keep the token;
        # see docs/demo-2026-07 in the platform repo for the 2026-08 Scan/OS promotion).
        return send_from_directory(static_dir, "demo.html")

    @app.route("/<path:path>")
    def serve_static(path):
        return send_from_directory(static_dir, path)


def initialize(
    submap_size=8,
    min_disparity=30.0,
    conf_threshold=25.0,
    vis_stride=4,
    serve_static_dir=None,
):
    """Initialize SLAM processor and static routes."""
    global slam_processor, _openrouter_api_key

    serve_frontend(serve_static_dir)

    slam_processor = StreamingSLAM(
        submap_size=submap_size,
        min_disparity=min_disparity,
        conf_threshold=conf_threshold,
        vis_stride=vis_stride,
    )
    slam_processor.frame_queue = frame_queue
    slam_processor.result_queue = result_queue
    slam_processor.event_loop = _event_loop
    slam_processor.result_ready_event = _result_ready

    # Session-scoped agent architecture keeps this None to avoid global per-submap callback.
    slam_processor.spatial_agent = None

    _openrouter_api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if _openrouter_api_key:
        print("Spatial Agent runtime enabled (OPENROUTER_API_KEY found)")
    else:
        print("Spatial Agent runtime disabled (no OPENROUTER_API_KEY)")

    # Fine-grained object enrichment (VLM captioner; None when disabled / no key).
    enricher = _get_object_enricher()
    slam_processor.set_object_enricher(enricher)
    if enricher is not None:
        rs = "on" if OBJECT_ENRICH_REVERSE_SEARCH else "off"
        print(f"Object enrichment enabled (model={OBJECT_ENRICH_MODEL}, reverse_search={rs})")

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print("=" * 60)
    print("VGGT-SLAM 2.0 Streaming Server — initialized")
    print(f"GPU: {gpu}  |  submap_size={submap_size}  |  vis_stride={vis_stride}")
    if serve_static_dir:
        print(f"Serving frontend from: {serve_static_dir}")
    print("=" * 60)


def start_server(
    port=5000,
    submap_size=8,
    min_disparity=30.0,
    conf_threshold=25.0,
    vis_stride=4,
    video=None,
    fast=False,
    video_fps=2.0,
    serve_static_dir=None,
):
    """Start the streaming SLAM server locally using uvicorn."""
    initialize(
        submap_size=submap_size,
        min_disparity=min_disparity,
        conf_threshold=conf_threshold,
        vis_stride=vis_stride,
        serve_static_dir=serve_static_dir,
    )

    video_feeder = None
    if video:
        video_feeder = VideoFeeder(
            video,
            fast=fast,
            target_fps=video_fps,
            on_complete=lambda: _trigger_scene_report_threadsafe(wait_for_drain=True),
        )
        video_feeder.start()

    ssl_certfile = None
    ssl_keyfile = None
    if not serve_static_dir:
        cert_path = os.path.join(os.path.dirname(__file__), "webserver", "server.cert")
        key_path = os.path.join(os.path.dirname(__file__), "webserver", "server.key")
        if os.path.exists(cert_path) and os.path.exists(key_path):
            ssl_certfile = cert_path
            ssl_keyfile = key_path
            print(f"SSL enabled: {cert_path}")
        else:
            print(
                "Warning: SSL certs not found. HTTPS disabled. "
                "Phone camera streaming requires HTTPS."
            )

    print("=" * 60)
    print("VGGT-SLAM 2.0 Streaming Server")
    print("=" * 60)
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    if video:
        print(f"Video input: {video} ({video_fps} fps, {'fast' if fast else 'real-time'})")
    else:
        print("Input: live WebSocket feed")
    print(f"Submap size: {submap_size}")
    print(f"Temp directory: {slam_processor.temp_dir}")
    if serve_static_dir:
        print(f"Serving frontend from: {serve_static_dir}")
    proto = "https" if ssl_certfile else "http"
    print(f"Server: {proto}://0.0.0.0:{port}")
    print("=" * 60)

    import uvicorn
    try:
        import uvloop
        loop = 'uvloop'
    except ImportError:
        loop = 'asyncio'

    uvicorn.run(
        asgi_application,
        host="0.0.0.0",
        port=port,
        ssl_certfile=ssl_certfile,
        ssl_keyfile=ssl_keyfile,
        loop=loop,
    )


# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VGGT-SLAM 2.0 Streaming Server")
    parser.add_argument("--port", type=int, default=5000, help="Server port")
    parser.add_argument("--video", type=str, default=None, help="Path to a video file for offline testing")
    parser.add_argument("--fast", action="store_true", help="Feed video frames as fast as possible")
    parser.add_argument("--video-fps", type=float, default=2.0, help="Effective FPS to extract from video")
    parser.add_argument("--submap-size", type=int, default=8, help="Frames per submap")
    parser.add_argument(
        "--min-disparity", type=float, default=30.0, help="Minimum disparity for keyframe selection"
    )
    parser.add_argument(
        "--conf-threshold", type=float, default=25.0, help="Confidence threshold percentage"
    )
    parser.add_argument("--vis-stride", type=int, default=4, help="Visualization stride")
    args = parser.parse_args()

    if args.video and not os.path.isfile(args.video):
        print(f"Video file not found: {args.video}")
        raise SystemExit(1)

    start_server(
        port=args.port,
        submap_size=args.submap_size,
        min_disparity=args.min_disparity,
        conf_threshold=args.conf_threshold,
        vis_stride=args.vis_stride,
        video=args.video,
        fast=args.fast,
        video_fps=args.video_fps,
    )
