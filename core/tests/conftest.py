import base64
import importlib
import json
import sys
import time
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_streaming_slam_module(monkeypatch):
    sys.modules.pop("server.streaming_slam", None)

    solver_mod = types.ModuleType("vggt_slam.solver")
    solver_mod.Solver = object
    monkeypatch.setitem(sys.modules, "vggt_slam.solver", solver_mod)

    object_detector_mod = types.ModuleType("vggt_slam.object_detector")

    class FakeObjectDetector:
        @staticmethod
        def deduplicate_detections(detections):
            return detections

    object_detector_mod.ObjectDetector = FakeObjectDetector
    monkeypatch.setitem(sys.modules, "vggt_slam.object_detector", object_detector_mod)

    slam_utils_mod = types.ModuleType("vggt_slam.slam_utils")
    slam_utils_mod.compute_image_embeddings = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "vggt_slam.slam_utils", slam_utils_mod)

    vggt_mod = types.ModuleType("vggt")
    vggt_models_mod = types.ModuleType("vggt.models")
    vggt_models_vggt_mod = types.ModuleType("vggt.models.vggt")
    vggt_models_vggt_mod.VGGT = object
    monkeypatch.setitem(sys.modules, "vggt", vggt_mod)
    monkeypatch.setitem(sys.modules, "vggt.models", vggt_models_mod)
    monkeypatch.setitem(sys.modules, "vggt.models.vggt", vggt_models_vggt_mod)

    return importlib.import_module("server.streaming_slam")


def load_app_module(monkeypatch):
    sys.modules.pop("server.app", None)

    flask_mod = types.ModuleType("flask")

    class FakeFlask:
        def __init__(self, *args, **kwargs):
            self.routes = []

        def route(self, *args, **kwargs):
            def decorator(fn):
                self.routes.append((args, kwargs, fn))
                return fn

            return decorator

        def before_request(self, fn):
            return fn

    flask_mod.Flask = FakeFlask
    flask_mod.jsonify = lambda *args, **kwargs: {"args": args, "kwargs": kwargs}
    flask_mod.request = types.SimpleNamespace(args={}, get_json=lambda: {})
    flask_mod.send_file = lambda *args, **kwargs: {"file": args, "kwargs": kwargs}
    monkeypatch.setitem(sys.modules, "flask", flask_mod)

    flask_cors_mod = types.ModuleType("flask_cors")
    flask_cors_mod.CORS = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "flask_cors", flask_cors_mod)

    asgiref_mod = types.ModuleType("asgiref")
    asgiref_wsgi_mod = types.ModuleType("asgiref.wsgi")
    asgiref_wsgi_mod.WsgiToAsgi = lambda app: app
    monkeypatch.setitem(sys.modules, "asgiref", asgiref_mod)
    monkeypatch.setitem(sys.modules, "asgiref.wsgi", asgiref_wsgi_mod)

    socketio_mod = types.ModuleType("socketio")

    class FakeAsyncServer:
        def __init__(self, *args, **kwargs):
            self.emits = []

        def on(self, *args, **kwargs):
            def decorator(fn):
                return fn

            return decorator

        async def emit(self, *args, **kwargs):
            self.emits.append((args, kwargs))

    socketio_mod.AsyncServer = FakeAsyncServer
    socketio_mod.ASGIApp = lambda sio, app: app

    socketio_exceptions_mod = types.ModuleType("socketio.exceptions")
    socketio_exceptions_mod.ConnectionRefusedError = type(
        "ConnectionRefusedError", (Exception,), {}
    )
    socketio_mod.exceptions = socketio_exceptions_mod
    monkeypatch.setitem(sys.modules, "socketio", socketio_mod)
    monkeypatch.setitem(sys.modules, "socketio.exceptions", socketio_exceptions_mod)

    llm_mod = types.ModuleType("server.llm")
    llm_mod.OpenRouterClient = object
    monkeypatch.setitem(sys.modules, "server.llm", llm_mod)

    streaming_slam_mod = types.ModuleType("server.streaming_slam")
    streaming_slam_mod.StreamingSLAM = object
    monkeypatch.setitem(sys.modules, "server.streaming_slam", streaming_slam_mod)

    object_detector_mod = types.ModuleType("vggt_slam.object_detector")
    object_detector_mod.ObjectDetector = object
    monkeypatch.setitem(sys.modules, "vggt_slam.object_detector", object_detector_mod)

    # Clerk auth is verified at import time (env requirements + JWKS fetch).
    # Provide fake env and stub `requests`/`jwt` so the module imports without
    # network access or real signing keys.
    monkeypatch.setenv("CLERK_JWKS_URL", "https://clerk.test/jwks")
    monkeypatch.setenv("CLERK_JWT_ISSUER", "https://clerk.test")
    monkeypatch.setenv("CLERK_JWT_AZP", "https://app.test")

    requests_mod = types.ModuleType("requests")

    class _FakeJWKSResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"keys": [{"kid": "test-kid", "kty": "RSA"}]}

    requests_mod.get = lambda *args, **kwargs: _FakeJWKSResponse()
    monkeypatch.setitem(sys.modules, "requests", requests_mod)

    jwt_mod = types.ModuleType("jwt")
    jwt_mod.PyJWTError = type("PyJWTError", (Exception,), {})

    # Clerk tokens (RS256) get fixed verified claims; the broker's own HS256 session
    # tokens are faked as "hs256.<b64(json payload)>" so the HS256 mint/verify/dispatch
    # logic is exercised (issuer + exp enforced) without real crypto.
    _HS_PREFIX = "hs256."

    def _fake_encode(payload, key, algorithm=None, **kwargs):
        blob = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()
        return _HS_PREFIX + blob

    def _fake_get_unverified_header(token):
        if isinstance(token, str) and token.startswith(_HS_PREFIX):
            return {"alg": "HS256"}
        return {"kid": "test-kid", "alg": "RS256"}

    def _fake_decode(token, key, **kwargs):
        if isinstance(token, str) and token.startswith(_HS_PREFIX):
            payload = json.loads(base64.urlsafe_b64decode(token[len(_HS_PREFIX):]).decode())
            issuer = kwargs.get("issuer")
            if issuer is not None and payload.get("iss") != issuer:
                raise jwt_mod.PyJWTError("issuer mismatch")
            if float(payload.get("exp", 0)) < time.time():
                raise jwt_mod.PyJWTError("expired")
            return payload
        return {
            "sub": "test-user",
            "azp": "https://app.test",
            "tier": "approved",
            "exp": 9_999_999_999,
        }

    jwt_mod.encode = _fake_encode
    jwt_mod.get_unverified_header = _fake_get_unverified_header
    jwt_mod.decode = _fake_decode
    jwt_mod.algorithms = types.SimpleNamespace(
        RSAAlgorithm=types.SimpleNamespace(from_jwk=lambda raw: "fake-key")
    )
    monkeypatch.setitem(sys.modules, "jwt", jwt_mod)

    return importlib.import_module("server.app")
