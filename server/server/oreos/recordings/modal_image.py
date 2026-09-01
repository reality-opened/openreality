"""One Modal image recipe for all three Oreos GPU workers + core-source provenance.

``modal_recon.py`` (batch), ``live_probe/modal_live.py`` (latency probe) and
``live_node/modal_stream.py`` (persistent stream) used to carry three copies of
the same image chain, each ending in::

    .add_local_dir(_SERVER.parent / "core" / "vggt_slam", copy=True)
    .add_local_file(_SERVER.parent / "core" / "setup.py", copy=True)
    .run_commands("cd /root/project && pip install -e .")

i.e. every Oreos image shipped **whatever was in the operator's sibling `core`
checkout at deploy time**. That is a reproducibility hole, not a convenience:
uncommitted edits ship silently, two operators deploying the same server commit
get two different images, a standalone `server` clone cannot build at all, and
nothing in the run artifacts records which core actually ran. EXP-36's whole
value is that its numbers are attributable.

So the core source is now explicit, via ``OREOS_CORE_SOURCE``:

  ``tag``   (default) — pip-install ``openreality-core`` from the pinned tag
            (:data:`CORE_TAG`) out of the private repo, exactly like
            ``modal_streaming.py`` does. Needs the ``github-token`` Modal secret
            (a PAT with read access to ``reality-opened/core``); the build guards
            for it and fails loudly rather than baking an empty credential.
  ``local`` — today's behaviour: mount + editable-install the sibling checkout.
            The dev loop (edit core, redeploy, measure) and the only mode that
            works while the tag is unpublished.

⚠️ **CAVEAT (2026-07-24): core ``v2.2.0`` is a LOCAL tag.** It was cut today and
has NOT been pushed to ``reality-opened/core``. ``tag`` mode therefore fails at
image build until the founder pushes core with ``--follow-tags``; until then
deploy with ``OREOS_CORE_SOURCE=local``. The build error says so too.

Provenance: BOTH modes stamp the image env with ``OREOS_CORE_SOURCE`` and
``OREOS_CORE_PROVENANCE`` (JSON) — for ``tag`` the literal tag, for ``local`` the
sibling checkout's HEAD sha plus a dirty flag from ``git status --porcelain``.
The workers copy that stamp into their run artifacts
(``results/recon_summary.json`` -> report.html, the live node's ``hello`` /
``/health``), so a number can always be traced back to the core that produced it.

**Deliberate divergence from ``modal_streaming.py``**: that image also installs
``vggt @ git+…#subdirectory=vggt-omega-slam`` (the VGGT-Omega shim). The Oreos
workers must NOT — they load VGGT-1B weights
(``facebook/VGGT-1B/resolve/main/model.pt``) into the MIT-SPARK ``VGGT_SPARK``
fork, which is what EXP-36 and the live probe measured. The Omega shim is a
different architecture backed by a separate ``vggt_omega`` package, both forks
claim the ``vggt`` import name, and only one may be installed. Swapping it in
here would break the weight load and silently invalidate every gpu_ms comparison
in the Oreos line. Core (``vggt_slam``) is the shared dependency; the backbone is
not.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

# -- knobs ---------------------------------------------------------------------

ENV_CORE_SOURCE = "OREOS_CORE_SOURCE"
ENV_CORE_PROVENANCE = "OREOS_CORE_PROVENANCE"

CORE_SOURCE_TAG = "tag"
CORE_SOURCE_LOCAL = "local"
CORE_SOURCES = (CORE_SOURCE_TAG, CORE_SOURCE_LOCAL)

CORE_REPO = "https://github.com/reality-opened/core"
#: Pinned core tag for ``tag`` mode. Bump together with requirements.txt,
#: .github/workflows/ci.yml and modal_streaming.py — see server/CLAUDE.md.
CORE_TAG = "v2.2.2"

_THIS_FILE = Path(__file__).resolve()
_REMOTE_SELF = "/root/oreos_modal_image.py"

try:  # server repo root (…/server); remote container: this file sits at /root/
    _SERVER_ROOT = _THIS_FILE.parents[2]
except IndexError:  # pragma: no cover - container layout
    _SERVER_ROOT = Path("/nonexistent")
DEFAULT_CORE_DIR = _SERVER_ROOT.parent / "core"

_TAG_HINT = (
    f"core {CORE_TAG} could not be installed from {CORE_REPO}. Two likely causes: "
    f"(1) the tag is not on origin yet — {CORE_TAG} was cut locally on 2026-07-24 and "
    f"needs 'git push --follow-tags'; (2) the 'github-token' Modal secret is missing or "
    f"lacks read access to reality-opened/core. To deploy from your sibling checkout "
    f"instead, re-run with {ENV_CORE_SOURCE}={CORE_SOURCE_LOCAL}."
)


def core_source_mode(environ: dict | None = None) -> str:
    """The configured core source; defaults to :data:`CORE_SOURCE_TAG`."""
    env = os.environ if environ is None else environ
    mode = (env.get(ENV_CORE_SOURCE) or CORE_SOURCE_TAG).strip().lower()
    if mode not in CORE_SOURCES:
        raise ValueError(
            f"{ENV_CORE_SOURCE}={mode!r} is not one of {CORE_SOURCES}. "
            f"Use {CORE_SOURCE_TAG!r} (pinned {CORE_TAG}, reproducible) or "
            f"{CORE_SOURCE_LOCAL!r} (your sibling core checkout, for dev)."
        )
    return mode


def _git(core_dir: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(core_dir), *args],
            capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout


def core_provenance(mode: str | None = None, core_dir: Path | None = None) -> dict:
    """What core this image ships — resolved at BUILD time, on the operator's box.

    Never raises: a build must not die because git is unavailable, and this is
    also evaluated on module import inside the container (where the sibling
    checkout does not exist). An unresolvable local checkout is reported as
    ``sha: None``, which is itself the honest answer.
    """
    mode = mode or core_source_mode()
    if mode == CORE_SOURCE_TAG:
        return {
            "mode": CORE_SOURCE_TAG,
            "repo": CORE_REPO,
            "tag": CORE_TAG,
            "pip_spec": f"openreality-core @ git+{CORE_REPO}@{CORE_TAG}",
        }
    core_dir = Path(core_dir or DEFAULT_CORE_DIR)
    sha = _git(core_dir, "rev-parse", "HEAD")
    porcelain = _git(core_dir, "status", "--porcelain")
    prov = {
        "mode": CORE_SOURCE_LOCAL,
        "path": str(core_dir),
        "sha": sha.strip() if sha else None,
        "dirty": None if porcelain is None else bool(porcelain.strip()),
        "dirty_files": None if porcelain is None else len(porcelain.strip().splitlines() or []),
    }
    if prov["sha"] is None:
        prov["note"] = (
            f"could not read a git sha from {core_dir} — the image ships whatever is "
            f"in that directory, unattributably. Prefer {ENV_CORE_SOURCE}="
            f"{CORE_SOURCE_TAG}."
        )
    return prov


def provenance_env(mode: str | None = None, core_dir: Path | None = None) -> dict:
    """The two env vars stamped into the image (and read back by the workers)."""
    mode = mode or core_source_mode()
    return {
        ENV_CORE_SOURCE: mode,
        ENV_CORE_PROVENANCE: json.dumps(core_provenance(mode, core_dir), sort_keys=True),
    }


def core_provenance_from_env(environ: dict | None = None) -> dict:
    """Read the build-time stamp inside the container (for run artifacts).

    Returns a dict that is always safe to serialize into a manifest; a missing or
    corrupt stamp is reported as such rather than dropped, because "we don't know
    which core ran" is exactly the fact the manifest must not hide.
    """
    env = os.environ if environ is None else environ
    raw = env.get(ENV_CORE_PROVENANCE)
    if not raw:
        return {"mode": env.get(ENV_CORE_SOURCE) or "unknown",
                "note": f"{ENV_CORE_PROVENANCE} not set in the image env"}
    try:
        prov = json.loads(raw)
    except (TypeError, ValueError) as e:
        return {"mode": "unknown", "note": f"{ENV_CORE_PROVENANCE} is not valid JSON: {e}"}
    if not isinstance(prov, dict):
        return {"mode": "unknown", "note": f"{ENV_CORE_PROVENANCE} is not a JSON object"}
    return prov


def describe_core_source(prov: dict | None) -> str:
    """One-line human rendering for reports/logs."""
    if not prov:
        return "unknown"
    mode = prov.get("mode", "unknown")
    if mode == CORE_SOURCE_TAG:
        return f"tag {prov.get('tag')}"
    if mode == CORE_SOURCE_LOCAL:
        sha = prov.get("sha")
        short = sha[:9] if isinstance(sha, str) else "unknown-sha"
        dirty = prov.get("dirty")
        suffix = " (dirty)" if dirty else "" if dirty is False else " (dirty: unknown)"
        return f"local {short}{suffix}"
    return str(mode)


# -- the image -----------------------------------------------------------------

# Shared with live_probe/modal_recon/live_node: keep these three chains
# byte-identical across the workers so their image layers stay cache-hot.
_APT = ("libgl1-mesa-glx", "libglib2.0-0", "git", "cmake", "build-essential")
_PIP = (
    "torch==2.3.1", "torchvision==0.18.1", "numpy", "Pillow", "open3d",
    "opencv-python", "trimesh", "huggingface_hub", "einops", "safetensors",
    "gtsam-develop", "scipy", "pytorch_metric_learning", "pytorch-lightning",
    "viser==0.2.23", "matplotlib", "gradio", "termcolor", "tqdm", "omegaconf",
    "requests", "lz4", "ftfy", "regex", "imageio", "pyyaml",
)
_SALAD = (
    "git clone https://github.com/Dominic101/salad.git /root/third_party/salad"
    " && pip install -e /root/third_party/salad"
)
# The VGGT-1B backbone EXP-36 / the live probe measured. NOT the Omega shim —
# see the module docstring.
_VGGT_SPARK = (
    "git clone https://github.com/MIT-SPARK/VGGT_SPARK.git /root/third_party/vggt"
    " && pip install -e /root/third_party/vggt"
)


def build_oreos_image(
    *,
    local_dirs: tuple = (),
    local_files: tuple = (),
    core_dir: Path | None = None,
    environ: dict | None = None,
):
    """The shared Oreos GPU image, with core installed per ``OREOS_CORE_SOURCE``.

    ``local_dirs`` / ``local_files`` are ``(local_path, remote_path)`` pairs
    mounted (no copy) at the very end — the caller's own worker code. This module
    mounts itself first, so the workers can import it in the container.
    """
    import modal  # lazy: importable (and testable) without the modal client

    mode = core_source_mode(environ)
    core_dir = Path(core_dir or DEFAULT_CORE_DIR)

    image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install(*_APT)
        .pip_install(*_PIP)
        .run_commands(_SALAD)
        .run_commands(_VGGT_SPARK)
    )

    if mode == CORE_SOURCE_TAG:
        secret = modal.Secret.from_name("github-token")
        image = image.run_commands(
            # Same guard as modal_streaming.py: an empty/misnamed secret otherwise
            # bakes an EMPTY credential into ~/.gitconfig and only surfaces later
            # as GitHub's opaque "Invalid username or token".
            'test -n "${GITHUB_TOKEN}" || { echo "FATAL: github-token secret has no'
            ' GITHUB_TOKEN value"; exit 1; }',
            'git config --global url."https://x-access-token:${GITHUB_TOKEN}@github.com/"'
            '.insteadOf "https://github.com/"',
            secrets=[secret],
        ).run_commands(
            # run_commands (not pip_install) so the failure can carry the hint.
            f"pip install 'openreality-core @ git+{CORE_REPO}@{CORE_TAG}'"
            f' || {{ echo "FATAL: {_TAG_HINT}"; exit 1; }}',
            secrets=[secret],
        )
    else:
        if not (core_dir / "vggt_slam").is_dir():
            # Printed, not raised: this module is also imported inside the
            # container, where the sibling checkout does not exist and the image
            # is never rebuilt. A real build stops on the missing mount anyway.
            print(
                f"[oreos-image] {ENV_CORE_SOURCE}={CORE_SOURCE_LOCAL} but no core "
                f"checkout at {core_dir} — a build from here will fail. Use "
                f"{ENV_CORE_SOURCE}={CORE_SOURCE_TAG} for a standalone clone."
            )
        image = (
            image
            .add_local_dir(str(core_dir / "vggt_slam"),
                           remote_path="/root/project/vggt_slam", copy=True)
            .add_local_file(str(core_dir / "setup.py"),
                            remote_path="/root/project/setup.py", copy=True)
            .run_commands("cd /root/project && pip install -e .")
        )

    # Provenance stamp: cheap layer, and the ONLY way a run artifact can name the
    # core that produced it (the workers read these back at runtime).
    image = image.env(provenance_env(mode, core_dir))

    # Mounts last: add_local_* without copy must be the final layers.
    image = image.add_local_file(str(_THIS_FILE), remote_path=_REMOTE_SELF)
    for local, remote in local_files:
        image = image.add_local_file(str(local), remote_path=remote)
    for local, remote in local_dirs:
        image = image.add_local_dir(str(local), remote_path=remote)
    return image
