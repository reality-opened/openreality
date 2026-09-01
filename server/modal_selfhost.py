"""Self-host deploy for YOUR OWN Modal account. Public dependencies only.

    modal secret create openreality-selfhost \
        OPENREALITY_LOCAL_TOKEN=$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))') \
        OPENROUTER_API_KEY="" GEMINI_API_KEY="" FAL_API_KEY=""
    modal deploy modal_selfhost.py
    modal run modal_selfhost.py::download_models

Then connect the MCP client with OPENREALITY_URL=<the printed web URL> and
OPENREALITY_TOKEN=<your OPENREALITY_LOCAL_TOKEN>.

This is the two-function twin of the hosted deploy, stripped to what a
self-hoster needs: one always-addressable CPU broker (OPENREALITY_AUTH=local,
headless: no SPA, no Clerk, no billing) plus one GPU worker that runs the
workspace jobs through ``server.selfhost.run_*`` (a single job implementation
shared with the bare-GPU path; see server/selfhost.py and docs/self-hosting.md).

Backbone: MIT-SPARK VGGT_SPARK (pinned commit) + the facebook/VGGT-1B weights.
Both are CC BY-NC 4.0 — a self-hosted stack is for NON-COMMERCIAL use unless
you hold your own license from Meta. The optional detection stack adds
facebookresearch/perception_models (Apache-2.0 code) and facebookresearch/sam3
(SAM License: commercial use allowed, ITAR/acceptable-use restrictions).

Costs land on your Modal account: the broker is a small CPU container that
scales to zero; each reconstruction runs on OPENREALITY_GPU (default A10G).
"""

import os

import modal

APP_NAME = "openreality-selfhost"

# The core library from this monorepo (BSD-2-Clause) and the pinned NC backbone.
CORE_PIN = "openreality-core @ git+https://github.com/reality-opened/openreality@core-v0.1.0#subdirectory=core"
VGGT_SPARK_COMMIT = "6e6e16107b88e8e76c751826af10d4295d87ecd2"
VGGT_1B_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"

GPU = os.environ.get("OPENREALITY_GPU", "A10G")
RECON_MEMORY_MB = int(os.environ.get("OPENREALITY_RECON_MEMORY_MB", "49152"))

app = modal.App(APP_NAME)

scene_vol = modal.Volume.from_name(f"{APP_NAME}-scenes", create_if_missing=True)
model_cache = modal.Volume.from_name(f"{APP_NAME}-models", create_if_missing=True)
SCENES_PATH = "/root/scenes"
CACHE_PATH = "/root/.cache/torch/hub"

scene_dict = modal.Dict.from_name(f"{APP_NAME}-scenes-kv", create_if_missing=True)
jobs_dict = modal.Dict.from_name(f"{APP_NAME}-jobs", create_if_missing=True)
api_keys_dict = modal.Dict.from_name(f"{APP_NAME}-api-keys", create_if_missing=True)

# One user-created secret carries the local bearer plus the optional BYO API
# keys (scene agents via OpenRouter, SAM 3 / TRELLIS routes via fal.ai, Gemini
# lanes). Empty values are fine; the routes that need a missing key fail
# honestly instead of at deploy time.
selfhost_secret = modal.Secret.from_name(
    "openreality-selfhost", required_keys=["OPENREALITY_LOCAL_TOKEN"]
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "git", "cmake", "build-essential", "curl")
    .pip_install(
        # Mirrors the hosted image's dependency list (modal_streaming.py) minus
        # the frontend build, billing, and Isaac authoring extras.
        "torch==2.3.1",
        "torchvision==0.18.1",
        "numpy",
        "Pillow",
        "open3d",
        "pyarrow",
        "opencv-python",
        "trimesh",
        "huggingface_hub",
        "einops",
        "safetensors",
        "gtsam-develop",
        "scipy",
        "pytorch_metric_learning",
        "pytorch-lightning",
        "viser==0.2.23",
        "matplotlib",
        "gradio",
        "flask",
        "flask-cors",
        "python-socketio",
        "asgiref",
        "uvicorn",
        "google-generativeai",
        "openai",
        "anthropic",
        "PyJWT[crypto]",
        "termcolor",
        "tqdm",
        "omegaconf",
        "requests",
        "lz4",
        "ftfy",
        "regex",
        "uvloop",
    )
    # Loop detection (SALAD) — same third-party clone the hosted image uses.
    .run_commands(
        "git clone https://github.com/Dominic101/salad.git /root/third_party/salad"
        " && pip install -e /root/third_party/salad",
    )
    # VGGT backbone: the MIT-SPARK fork at a pinned commit provides the `vggt`
    # import backed by VGGT-1B. CC BY-NC 4.0 — cloned at build time from
    # upstream, never redistributed by this repo.
    .run_commands(
        "git clone https://github.com/MIT-SPARK/VGGT_SPARK.git /root/third_party/vggt"
        f" && git -C /root/third_party/vggt checkout {VGGT_SPARK_COMMIT}"
        " && pip install -e /root/third_party/vggt",
    )
    # Open-set detection stack (optional feature, on by default for recon).
    .run_commands(
        "git clone https://github.com/facebookresearch/perception_models.git"
        " /root/third_party/perception_models"
        " && pip install -e /root/third_party/perception_models --no-deps",
    )
    .run_commands(
        # sam3's training-data import chain pulls in decord + pycocotools which we
        # don't need for inference. Stub out decord (no Python 3.11 wheel exists),
        # install pycocotools, then install sam3. Mirrors the hosted image.
        "python -c \""
        "import site, os; p=site.getsitepackages()[0]+'/decord';"
        "os.makedirs(p,exist_ok=True);"
        "open(p+'/__init__.py','w').write('cpu=None\\nVideoReader=None')"
        "\""
        " && pip install pycocotools"
        " && git clone https://github.com/facebookresearch/sam3.git /root/third_party/sam3"
        " && pip install -e /root/third_party/sam3",
    )
    # The SLAM library (public mirror, BSD-2-Clause).
    .pip_install(CORE_PIN)
    .add_local_dir("server", remote_path="/root/project/server", copy=True)
)


@app.function(
    image=image,
    volumes={CACHE_PATH: model_cache},
    timeout=1800,
)
def download_models() -> None:
    """Cache the VGGT-1B weights (ungated, CC BY-NC 4.0) into the model volume.
    Idempotent; run once after the first deploy. Auxiliary models (SALAD's
    backbone, the metric-anchor depth model, detection checkpoints) download
    lazily on first use into the same mounted cache."""
    import urllib.request

    ckpt_dir = os.path.join(CACHE_PATH, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    dest = os.path.join(ckpt_dir, "model.pt")
    if os.path.exists(dest):
        print("VGGT-1B: already cached")
    else:
        print(f"Downloading VGGT-1B from {VGGT_1B_URL} ...")
        urllib.request.urlretrieve(VGGT_1B_URL, dest)
        print(f"VGGT-1B cached at {dest} ({os.path.getsize(dest) / 1e9:.2f} GB)")
    model_cache.commit()


@app.function(
    image=image,
    gpu=GPU,
    memory=RECON_MEMORY_MB,
    volumes={CACHE_PATH: model_cache, SCENES_PATH: scene_vol},
    secrets=[selfhost_secret],
    timeout=3600,
    max_containers=1,  # serialize jobs: spend guard + no duplicate-artifact races
)
def selfhost_job(kind: str, payload: dict) -> None:
    """One GPU worker for every workspace job kind, dispatching to the SHARED
    job bodies in server/selfhost.py (the bare-GPU path runs the same code as
    in-process threads). CPU-only kinds (splat/lod/anchor/export) ride the same
    function for simplicity; the GPU sits idle for those minutes."""
    import sys

    sys.path.insert(0, "/root/project")

    from server import selfhost as sh
    from server.scene_report.store import ModalScenePersistence

    scene_vol.reload()
    persistence = ModalScenePersistence(
        scene_dict, SCENES_PATH, commit_fn=scene_vol.commit, reload_fn=scene_vol.reload
    )
    handlers = {
        "recon": sh.run_recon_job,
        "splat": sh.run_splat_job,
        "lod": sh.run_lod_job,
        "anchor": sh.run_anchor_job,
        "export": sh.run_export_job,
    }
    handlers[kind](persistence, jobs_dict, **payload)


@app.function(
    image=image,
    secrets=[selfhost_secret],
    volumes={SCENES_PATH: scene_vol},
    cpu=1.0,
    memory=4096,
    timeout=86400,
    max_containers=1,
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def web():
    import sys

    sys.path.insert(0, "/root/project")
    os.environ.setdefault("OPENREALITY_AUTH", "local")

    from server import app as server_app
    from server import selfhost as sh
    from server.api_keys import ApiKeyRegistry
    from server.oreos import jobs as demo_jobs
    from server.oreos import routes_export_job, routes_ingest, routes_lod, routes_recordings
    from server.scene_report.store import ModalScenePersistence

    server_app.configure_scene_persistence(
        ModalScenePersistence(
            scene_dict, SCENES_PATH, commit_fn=scene_vol.commit, reload_fn=scene_vol.reload
        )
    )
    server_app.configure_api_key_registry(ApiKeyRegistry(api_keys_dict))
    demo_jobs.configure_jobs_store(jobs_dict)

    def _spawner(kind: str):
        return lambda **kw: selfhost_job.spawn(kind, kw)

    routes_ingest.configure_recon_spawner(_spawner("recon"))
    routes_ingest.configure_splat_spawner(_spawner("splat"))
    routes_lod.configure_lod_spawner(_spawner("lod"))
    routes_export_job.configure_export_spawner(_spawner("export"))
    server_app.configure_anchor_job_spawner(_spawner("anchor"))
    routes_recordings.configure_recording_spawner(
        lambda **kw: sh.run_recording_job(jobs_dict, **kw)  # instant honest refusal
    )
    # Deliberately NOT wired: serve_frontend (headless broker) and
    # configure_gpu_session_broker (live phone streaming is a hosted concern).

    return server_app.asgi_application
