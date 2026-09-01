"""Generative-model access for the demo surface (design/features.md §0.4).

``fal_client`` wraps the fal.ai queue API used by SAM 3 click-segmentation
(F3 paths b/c) and TRELLIS variations (F3/F6). Its per-model payload and
response shapes were **verified live against fal on 2026-07-31** — read that
module's docstring before changing them; several differ sharply from what the
model pages imply (subpath endpoints, polling URLs that drop the subpath, a
default text prompt of "wheel", and a GLB — not gaussian — TRELLIS output).

``mesh_to_splat`` is the landing pad for that GLB: it resamples the textured
mesh into the 3DGS PLY layout every downstream consumer already speaks.

Access is gated on the ``FAL_API_KEY`` env, mounted from the ``fal-secret``
Modal secret, attached to every deploy unconditionally. Without it callers
convert :class:`fal_client.FalKeyMissing` into the honest 503
``fal_key_missing`` JSON — never a silent downgrade, never a mock result.
"""
