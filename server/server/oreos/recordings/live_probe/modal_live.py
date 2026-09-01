"""Oreos live probe worker (Modal): warm VGGT-SLAM container for per-chunk latency.

Phase-2 measurement of the Oreos-on-DimOS plan
(platform/experiments/research/2026-07-24-oreos-on-dimos-plan.md): the number the
round-1 red-team demanded — per-submap-chunk round-trip latency of remote-GPU
reconstruction vs a robot's real keyframe rate.

Design (deliberately different from the batch worker ``server/oreos/recordings/modal_recon.py``,
whose image recipe + Solver usage this copies):

  * ``@app.cls`` with ``max_containers=1``: ONE container holds the Solver's SLAM
    state (pose graph, submaps, retrieval DB) across chunks. With the default
    serial input concurrency, queued inputs execute FIFO — chunk order is
    preserved as long as the driver spawns them in order (the server verifies via
    the ``out_of_order`` flag in each result).
  * ``@modal.enter`` loads VGGT + constructs the Solver ONCE. A cold first chunk
    therefore pays container boot + model load in its round trip — that is the
    honest cold number; the driver runs a warm-up pass first and reports cold
    separately.
  * ``process_chunk`` takes the jpegs over the wire (no volume round trip — a live
    robot has no volume), writes them to container tmp under their original
    timestamp names, and runs exactly one batch-worker cycle:
    ``run_predictions`` + ``add_points`` + ``graph.optimize``.
  * ``gpu_ms`` is pure model+solver time (excludes tmp-file writes and transport).

Deploy (from the server repo root; the image + core source come from
``server/oreos/recordings/modal_image.py``, same as modal_recon — ``OREOS_CORE_SOURCE=local``
if you want your sibling ``core`` checkout instead of the pinned tag):

    modal deploy server/oreos/recordings/live_probe/modal_live.py

Then drive it with ``server/oreos/recordings/live_probe/driver.py`` (uses
``modal.Cls.from_name("oreos-live-probe", "LiveProbe")``).
"""

import os
import time

import modal

try:  # local import (deploy-time, tests): the server repo package
    from server.oreos.recordings import modal_image as oreos_image
except ImportError:  # remote container: mounted flat at /root/oreos_modal_image.py
    import oreos_modal_image as oreos_image  # type: ignore[no-redef]

app = modal.App("oreos-live-probe")

# Same recipe as server/oreos/recordings/modal_recon.py — literally the same builder now, so
# the layers are identical and the build is cache-hot after any oreos deploy.
# OREOS_CORE_SOURCE picks tag (pinned, default) vs local sibling checkout.
image = oreos_image.build_oreos_image()

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
    timeout=600,
    startup_timeout=600,
    scaledown_window=300,   # stay warm between back-to-back probe runs
    max_containers=1,       # SLAM state lives in this one container; serial FIFO inputs
    retries=0,              # a retry would silently hide latency — never retry
)
class LiveProbe:
    """One warm A100 holding VGGT + a core Solver; chunks stream in over the wire."""

    # Match the EXP-36 / modal_recon batch config so gpu_ms is comparable.
    CONF_THRESHOLD = 25.0
    LC_THRES = 0.95
    MAX_LOOPS = 1

    @modal.enter()
    def load(self) -> None:
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
        self._boot_s = self._boot_ts - t0
        self._next_chunk_id = 0
        print(f"[live-probe] container warm: models+solver loaded in {self._boot_s:.1f}s")

    @modal.method()
    def process_chunk(self, jpegs: list, names: list, chunk_id: int) -> dict:
        """Run one submap cycle on a chunk of jpeg frames.

        ``jpegs``/``names`` are parallel lists in replay order; names are the
        original timestamp-ish filenames (e.g. ``1778055838.694958.jpg``) so the
        Solver's frame ids stay the recorded timestamps. Returns timing split so
        the driver can separate transport from compute.
        """
        import shutil
        import traceback

        server_recv_ts = time.time()
        out_of_order = chunk_id != self._next_chunk_id

        workdir = f"/tmp/live_chunk_{chunk_id:05d}"
        os.makedirs(workdir, exist_ok=True)
        paths = []
        for name, blob in zip(names, jpegs):
            p = os.path.join(workdir, os.path.basename(name))
            with open(p, "wb") as f:
                f.write(blob)
            paths.append(p)

        result = {
            "chunk_id": chunk_id,
            "n_frames": len(paths),
            "server_recv_ts": server_recv_ts,
            "out_of_order": out_of_order,
            "container_boot_s": round(self._boot_s, 2),
            "core_source": oreos_image.core_provenance_from_env(),
        }
        try:
            gpu_t0 = time.perf_counter()
            predictions = self.solver.run_predictions(
                paths, self.model, self.MAX_LOOPS, None, None
            )
            self.solver.add_points(predictions)
            self.solver.graph.optimize()
            gpu_ms = (time.perf_counter() - gpu_t0) * 1000.0

            latest = self.solver.map.get_latest_submap()
            result.update(
                n_poses=len(latest.get_frame_ids()) if latest is not None else 0,
                gpu_ms=round(gpu_ms, 1),
                n_submaps_total=len(list(self.solver.map.ordered_submaps_by_key())),
                loop_closures_total=int(self.solver.graph.get_num_loops()),
            )
            self._next_chunk_id = chunk_id + 1
        except Exception as e:  # noqa: BLE001 — report; driver decides to abort
            result["error"] = f"{type(e).__name__}: {e}"
            result["traceback"] = traceback.format_exc(limit=8)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

        result["server_done_ts"] = time.time()
        return result

    @modal.method()
    def reset(self) -> dict:
        """Reset SLAM state without reloading models (fresh session, warm GPU)."""
        self.solver.reset()
        self._next_chunk_id = 0
        return {"reset": True, "ts": time.time(), "container_boot_ts": self._boot_ts,
                "core_source": oreos_image.core_provenance_from_env()}

    @modal.method()
    def ping(self) -> dict:
        """Cheap warmness check; does not touch solver state."""
        return {"ok": True, "ts": time.time(), "container_boot_ts": self._boot_ts,
                "core_source": oreos_image.core_provenance_from_env()}
