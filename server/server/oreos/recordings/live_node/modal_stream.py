"""Oreos live node (Modal): persistent-stream VGGT-SLAM over one WebSocket.

The production follow-up to ``server/oreos/recordings/live_probe`` (Task #5 of the
Oreos-on-DimOS build-out). The probe proved on go2_short @7.14 fps that a warm
A100 runs a 17-frame chunk in ~1.2 s (0.53x realtime, 2x headroom) but the
spawn-per-chunk transport loses ~1.15 s/chunk to Modal per-call overhead — its
verdict: "a production live node should stream chunks over one persistent
connection into the warm container." This is exactly that:

  * ``@app.cls`` with ``max_containers=1``: ONE container holds VGGT + the core
    ``Solver``'s SLAM state, loaded once in ``@modal.enter`` (recipe copied from
    ``live_probe/modal_live.py`` — image layers identical, so cache-hot).
  * ``@modal.asgi_app()`` on the cls serves a raw ASGI app (no framework deps):
    ``GET /health``, ``POST /reset`` (HTTP), and a full-duplex WebSocket at
    ``/ws``. Modal web endpoints natively support WebSockets (one function call
    per connection; messages capped at 2 MiB — the wire protocol in
    ``protocol.py`` fragments oversized chunks under that cap).
  * Per connection, a receiver task drains the socket into a bounded
    ``asyncio.Queue`` while a single worker task runs the GPU cycle via
    ``asyncio.to_thread`` and replies the moment each chunk completes — uplink of
    chunk k+1 overlaps GPU on chunk k, and there is NO per-chunk RPC: the warm
    Solver is one queue-pop away. ``@modal.concurrent`` lets /health answer while
    a stream is live.
  * One streaming session at a time (a threading.Lock guards the Solver); a
    second connection gets ``busy`` and is closed.

This file owns only the Modal + GPU half. The connection lifecycle lives in
``asgi_app.py`` and the session bookkeeping in ``session.py`` — both modal-free
and unit-tested (``tests/test_live_node_session.py``), because the 2026-07-24
audit found eight lifecycle defects in code that had no tests at all.

**Session semantics (behaviour change, 2026-07-24).** Solver state still
survives a dropped connection — that is what makes reconnect-and-resume work,
and a resent chunk is served idempotently from the results cache. But a NEW
stream (first chunk of a connection has ``chunk_id == 0``, not in the cache)
arriving while the node still holds state is now REFUSED with a
``session_state_present`` protocol error telling the operator to ``POST /reset``.
Previously those frames were merely flagged ``out_of_order`` and APPENDED to the
previous client's map: two recordings share no gauge, so the merged map is
silently wrong geometry — a worse outcome than a loud refusal.

**Stall watchdog.** ``client.py`` deliberately disables WebSocket keepalive (a
ping queues behind megabytes of chunk bytes on a saturated uplink), so a
half-open TCP connection produces no disconnect event at all. The worker's queue
wait is therefore bounded by ``OREOS_LIVE_IDLE_TIMEOUT_S`` (default 120 s):
on a stall the socket is closed and the session released, instead of pinning the
only container (``max_containers=1``, ``timeout=3600``) so that every later
connection gets ``busy`` and every ``/reset`` 409s for an hour.

Env knobs:

    OREOS_LIVE_IDLE_TIMEOUT_S     stall watchdog seconds (default 120; <=0 off)
    OREOS_LIVE_RESULTS_CACHE_MAX  resume-cache LRU entries (default 64)
    OREOS_LIVE_UPLINK_QUEUE_MAX   chunks buffered ahead of the GPU (default 2)
    OREOS_CORE_SOURCE             at DEPLOY time: `tag` (pinned core, default) or
                                  `local` (the operator's sibling checkout) — see
                                  server/oreos/recordings/modal_image.py. The choice is
                                  stamped into `hello` and `/health` as
                                  `core_source`, and lands in the client's
                                  live_node_results.json.

Timing split reported per chunk (all on the container clock — never difference
client and server timestamps, the clocks are unrelated):

  * ``gpu_ms``                       pure model+solver time (as the probe).
  * ``extract_ms``                   pose/cloud payload extraction (post-GPU).
  * ``server_arrive_ts``             chunk fully received off the socket.
  * ``server_dispatch_overhead_s``   start - max(arrive, previous done): dead
    time between "chunk available AND GPU free" and "GPU starts". This is THE
    architecture number — the probe's spawn path shows p50 1.94 s (0.9-3.6 s)
    of it per chunk (server_recv[k+1] - server_done[k] in its results, all
    chunks pre-queued); the persistent stream measures ~0.1 ms. It is
    independent of the client's uplink, so the verdict survives a slow network.

Deploy (from the server repo root; ``OREOS_CORE_SOURCE=local`` if you want your
sibling ``core`` checkout instead of the pinned tag):

    modal deploy server/oreos/recordings/live_node/modal_stream.py

Then drive it with ``server/oreos/recordings/live_node/client.py``. Stop the app when
measurements finish (``modal app stop oreos-live-node``) — no idle GPU burn.
"""

import os
import time
from pathlib import Path

import modal

try:  # local import (deploy-time, tests): the server repo package
    from server.oreos.recordings import modal_image as oreos_image, pose_qc
    from server.oreos.recordings.live_node import asgi_app, protocol as proto, session as sess
except ImportError:  # remote container: mounted at /root/oreos_*
    import oreos_modal_image as oreos_image  # type: ignore[no-redef]
    import oreos_pose_qc as pose_qc  # type: ignore[no-redef]
    from oreos_live_node import asgi_app, protocol as proto, session as sess  # type: ignore

APP_NAME = "oreos-live-node"
app = modal.App(APP_NAME)

_LIVE_NODE_DIR = Path(__file__).resolve().parent

# Same recipe as live_probe/modal_live.py (itself modal_recon's) — literally the
# same builder now, so the layers are identical and cache-hot after any oreos
# deploy. OREOS_CORE_SOURCE picks tag (pinned, default) vs local sibling
# checkout. Only addition: the live_node package (protocol + session + asgi_app
# run unchanged in the container), mounted last.
image = oreos_image.build_oreos_image(
    local_dirs=((_LIVE_NODE_DIR, "/root/oreos_live_node"),),
    local_files=((Path(pose_qc.__file__).resolve(), "/root/oreos_pose_qc.py"),),
)

model_cache = modal.Volume.from_name("vggt-slam-models", create_if_missing=True)

CACHE_PATH = "/root/.cache/torch/hub"


def _ensure_models() -> None:
    import torch

    ckpt_dir = os.path.join(torch.hub.get_dir(), "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    if not os.path.exists(os.path.join(ckpt_dir, "model.pt")):
        torch.hub.download_url_to_file(
            "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt",
            os.path.join(ckpt_dir, "model.pt"),
        )
    if not os.path.exists(os.path.join(ckpt_dir, "dino_salad.ckpt")):
        torch.hub.download_url_to_file(
            "https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt",
            os.path.join(ckpt_dir, "dino_salad.ckpt"),
        )
    if not os.path.exists(os.path.join(torch.hub.get_dir(), "facebookresearch_dinov2_main")):
        torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    model_cache.commit()


@app.cls(
    image=image,
    gpu="A100-80GB",
    volumes={CACHE_PATH: model_cache},
    timeout=3600,           # max lifetime of ONE WebSocket session (one input)
    startup_timeout=600,
    scaledown_window=240,   # stay warm between back-to-back measurement runs
    max_containers=1,       # SLAM state lives in this one container
    retries=0,              # a retry would silently hide latency — never retry
)
@modal.concurrent(max_inputs=8)  # /health + /reset answer while a stream is live
class LiveNode:
    """One warm A100 holding VGGT + a core Solver behind a persistent WebSocket."""

    # Match live_probe / modal_recon batch config so gpu_ms stays comparable.
    CONF_THRESHOLD = 25.0
    LC_THRES = 0.95
    MAX_LOOPS = 1

    @modal.enter()
    def load(self) -> None:
        import threading

        import torch

        from vggt.models.vggt import VGGT
        from vggt_slam.solver import Solver

        t0 = time.time()
        _ensure_models()
        self.solver = Solver(
            init_conf_threshold=self.CONF_THRESHOLD,
            lc_thres=self.LC_THRES,
            vis_voxel_size=None,
            skip_viewer=True,
        )
        model = VGGT()
        model.load_state_dict(
            torch.hub.load_state_dict_from_url(
                "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
            )
        )
        self.model = model.eval().to(torch.bfloat16).to("cuda")
        self._boot_ts = time.time()
        self.boot_s = self._boot_ts - t0
        # All chunk bookkeeping + the /health snapshot; results survive a dropped
        # connection so a resuming client's resent chunks are served without GPU
        # (LRU-bounded — see session.py).
        self.state = sess.LiveSessionState()
        self.session_lock = threading.Lock()  # one streaming session / reset at a time
        self._core_source = oreos_image.core_provenance_from_env()
        print(f"[live-node] container warm: models+solver loaded in {self.boot_s:.1f}s "
              f"(core: {oreos_image.describe_core_source(self._core_source)})")

    def core_source(self) -> dict:
        """Which core this container runs — build-time stamp, into run artifacts."""
        return getattr(self, "_core_source", None) or oreos_image.core_provenance_from_env()

    # -- GPU cycle (sync; runs on a worker thread via asyncio.to_thread) -------

    def _solver_counts(self) -> tuple:
        """(n_submaps, n_loops) read on the WORKER thread only.

        Called between chunks, never concurrently with add_points/optimize —
        which is exactly why /health may not call it (audit finding L5): it reads
        live Solver containers, and the snapshot it feeds is what /health serves.
        """
        try:
            return (len(list(self.solver.map.ordered_submaps_by_key())),
                    int(self.solver.graph.get_num_loops()))
        except Exception as e:  # noqa: BLE001 — bookkeeping must not kill a chunk
            print(f"[live-node] could not read solver counts: {type(e).__name__}: {e}")
            return (None, None)

    def process_chunk_sync(self, chunk_id: int, names: list, jpegs: list) -> dict:
        """One submap cycle on a chunk of jpeg frames + result-payload extraction.

        Mirrors the probe's ``process_chunk`` exactly for the GPU portion (so
        ``gpu_ms`` is comparable), then extracts the live payload: TUM pose
        lines for the latest submap (sign-repaired like modal_recon) + a cloud
        summary. Extraction is separately timed and non-fatal.
        """
        import shutil
        import traceback

        out_of_order = self.state.begin_chunk(chunk_id)
        workdir = f"/tmp/live_chunk_{chunk_id:05d}"
        os.makedirs(workdir, exist_ok=True)
        paths = []
        for name, blob in zip(names, jpegs):
            p = os.path.join(workdir, os.path.basename(name))
            with open(p, "wb") as f:
                f.write(blob)
            paths.append(p)

        result = {
            "type": proto.T_RESULT,
            "chunk_id": chunk_id,
            "n_frames": len(paths),
            "out_of_order": out_of_order,
            "container_boot_s": round(self.boot_s, 2),
        }
        error = None
        try:
            gpu_t0 = time.perf_counter()
            predictions = self.solver.run_predictions(
                paths, self.model, self.MAX_LOOPS, None, None
            )
            self.solver.add_points(predictions)
            self.solver.graph.optimize()
            gpu_ms = (time.perf_counter() - gpu_t0) * 1000.0

            ext_t0 = time.perf_counter()
            payload = self._latest_submap_payload()
            extract_ms = (time.perf_counter() - ext_t0) * 1000.0

            n_submaps, n_loops = self._solver_counts()
            result.update(
                gpu_ms=round(gpu_ms, 1),
                extract_ms=round(extract_ms, 1),
                n_submaps_total=n_submaps,
                loop_closures_total=n_loops,
                **payload,
            )
        except Exception as e:  # noqa: BLE001 — report; the client decides to abort
            error = f"{type(e).__name__}: {e}"
            result["error"] = error
            result["traceback"] = traceback.format_exc(limit=8)
            n_submaps, n_loops = self._solver_counts()
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
            # Advance on BOTH paths (audit finding L7): one failed chunk used to
            # mark every later chunk out_of_order forever. Publishing here is
            # also what keeps /health snapshot-only (L5).
            self.state.finish_chunk(
                chunk_id, ok=error is None, n_submaps=n_submaps, n_loops=n_loops,
                error=error, ts=time.time(),
            )
        return result

    def _latest_submap_payload(self) -> dict:
        """TUM pose lines + cloud summary of the latest submap (non-fatal)."""
        from scipy.spatial.transform import Rotation

        try:
            latest = self.solver.map.get_latest_submap()
            if latest is None:
                return {"n_poses": 0, "poses_tum_lines": [], "cloud_summary": None}
            # Count the improper-rotation EVENT from the raw camera matrices —
            # core repairs det(R) = -1 internally now, so counting our own repair
            # would count zero forever (see server/oreos/recordings/pose_qc.py). The local
            # repair is gone; proper_rigid_poses is the tripwire on core's
            # guarantee, and its AssertionError lands in extract_error below.
            raw = pose_qc.submap_improper_stats(latest, self.solver.graph)
            fixed = pose_qc.proper_rigid_poses(
                latest.get_all_poses_world(self.solver.graph), where="latest submap"
            )
            tum_lines = []
            for fid, Tn in zip(latest.get_frame_ids(), fixed):
                q = Rotation.from_matrix(Tn[:3, :3]).as_quat()
                t_ = Tn[:3, 3]
                tum_lines.append(
                    f"{float(fid):.6f} {t_[0]:.6f} {t_[1]:.6f} {t_[2]:.6f} "
                    f"{q[0]:.8f} {q[1]:.8f} {q[2]:.8f} {q[3]:.8f}"
                )
            cloud = None
            try:
                pts = latest.get_points_in_world_frame(self.solver.graph).reshape(-1, 3)
                if len(pts):
                    cloud = {
                        "n_points": int(len(pts)),
                        "bbox_min": [round(float(v), 3) for v in pts.min(axis=0)],
                        "bbox_max": [round(float(v), 3) for v in pts.max(axis=0)],
                    }
            except Exception as e:  # noqa: BLE001
                cloud = {"error": f"{type(e).__name__}: {e}"}
            return {
                "n_poses": len(tum_lines),
                "poses_tum_lines": tum_lines,
                # Same manifest key + QC semantics as modal_recon: raw improper
                # rotations, independent of which layer repairs them. None when
                # the raw matrices could not be read — never a fake 0.
                "n_pose_sign_repairs": raw["n_improper"] if raw["available"] else None,
                "n_pose_nonfinite_dets": raw["n_nonfinite"] if raw["available"] else None,
                "submap_lc_status": bool(latest.get_lc_status()),
                "cloud_summary": cloud,
            }
        except Exception as e:  # noqa: BLE001
            return {
                "n_poses": 0,
                "poses_tum_lines": [],
                "cloud_summary": None,
                "extract_error": f"{type(e).__name__}: {e}",
            }

    def reset_sync(self) -> dict:
        self.solver.reset()
        self.state.reset(ts=time.time())
        return {"type": proto.T_RESET_OK, "ts": time.time(), "container_boot_ts": self._boot_ts}

    def status(self) -> dict:
        """/health payload — SNAPSHOT ONLY.

        ``@modal.concurrent(max_inputs=8)`` means this runs on the event loop
        while the worker thread may be inside ``add_points``/``optimize``.
        Reading ``ordered_submaps_by_key()`` / ``get_num_loops()`` from here is a
        genuine data race (audit finding L5), and the old blanket ``except`` hid
        it behind a flaky ``-1``. The worker publishes those counts after each
        chunk instead; this only ever reads the published snapshot.
        """
        return {
            "ok": True,
            "app": APP_NAME,
            "server_ts": time.time(),
            "container_boot_s": round(self.boot_s, 2),
            "session_active": self.session_lock.locked(),
            "idle_timeout_s": sess.idle_timeout_s(),
            "core_source": self.core_source(),
            **self.state.snapshot(),
        }

    # -- ASGI app: /health, /reset (HTTP) + /ws (WebSocket) --------------------

    @modal.asgi_app()
    def web(self):
        return asgi_app.build_asgi(self)
