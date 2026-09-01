"""Improper-rotation accounting for Oreos poses — counted where the EVENT happens.

Why this module exists (a QC signal that was about to die silently):

  * EXP-36 measured a real failure mode: SL(4) submap poses whose decomposition
    yields ``det(R) = -1`` (319/2074 poses on one recording, 15% on the collapsed
    cell, 0 elsewhere). ``server/qc/confidence.py`` turns that into
    ``sign_repair_rate`` with thresholds 0.01 (review) / 0.05 (fail),
    **calibrated on EXP-36 against the then-unfixed core**.
  * ``modal_recon.py`` and ``live_node/modal_stream.py`` each carried their own
    copy of the repair (``if det(R) < 0: R, t = -R, -t``) and counted how often
    THEIR OWN copy fired — i.e. they measured a reaction, not the event.
  * core ``3ea5a03`` ("decompose_camera always returns a proper rotation") moved
    the repair inside ``decompose_camera``. Every Oreos image now builds from
    core at or past that commit, so the local copy can never fire again: the
    count goes vacuously 0, ``sign_repair_rate`` is always 0.0, and the gate
    reports "clean" on precisely the recordings it was calibrated to catch.
    Nothing fails, nothing warns — a dead gate that reports success.

The fix is to count the event fix-independently, from the RAW camera matrix,
before anyone repairs anything:

    P = K [R | t]   with K upper-triangular and, after core's positive-diagonal
                    normalisation, det(K) = k00*k11*k22 > 0
    => sign(det(P[:3, :3])) = sign(det(K)) * sign(det(R)) = sign(det(R))

``Submap.get_all_poses_world(graph, give_camera_mat=True)`` hands back exactly
that raw projection matrix (already dehomogenised by ``P[3,3]``, which is what
``decompose_camera`` then decomposes), so ``det(P[:3, :3]) < 0`` is the same
predicate the old local repair fired on — no matter which layer repairs it. The
manifest key stays ``n_pose_sign_repairs`` and the thresholds stay unchanged, so
the EXP-36 calibration keeps its meaning; only the semantics note changes:
**raw improper-rotation rate, independent of which layer repairs**.

The other half is :func:`proper_rigid_poses`: core now guarantees a proper
rotation, so the local repair is deleted and replaced by a cheap ``det > 0``
tripwire. That is not paranoia — ``scipy``'s ``Rotation.from_matrix`` silently
orthogonalises an improper matrix into a *wrong* rotation, so a core regression
would otherwise ship plausible, incorrect quaternions.

Pure numpy: no modal, no torch, no core import. Mounted into the GPU images
alongside the workers (``/root/oreos_pose_qc.py``) and unit-tested in CI.
"""

from __future__ import annotations

import numpy as np


def _as_stack(mats) -> np.ndarray:
    arr = np.asarray(mats, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3 or arr.shape[-1] < 3 or arr.shape[-2] < 3:
        raise ValueError(f"expected a stack of >=3x3 matrices, got shape {arr.shape}")
    return arr


def improper_rotation_stats(camera_mats) -> dict:
    """Count improper rotations in RAW camera matrices ``P`` (any of 3x4/4x4).

    Returns ``{n_poses, n_improper, n_nonfinite}``. ``n_improper`` counts
    ``det(P[:3, :3]) < 0`` — the improper-rotation event itself, whoever repairs
    it. ``n_nonfinite`` counts poses whose determinant is not finite: that is the
    projectively-singular SL(4) pose EXP-36 saw crash the decomposition, and
    lumping it in with "proper" would hide it.
    """
    if camera_mats is None:
        return {"n_poses": 0, "n_improper": 0, "n_nonfinite": 0}
    arr = _as_stack(camera_mats)
    if arr.shape[0] == 0:
        return {"n_poses": 0, "n_improper": 0, "n_nonfinite": 0}
    dets = np.linalg.det(arr[:, :3, :3])
    finite = np.isfinite(dets)
    return {
        "n_poses": int(arr.shape[0]),
        "n_improper": int(np.count_nonzero(dets[finite] < 0.0)),
        "n_nonfinite": int(np.count_nonzero(~finite)),
    }


def submap_improper_stats(submap, graph) -> dict:
    """:func:`improper_rotation_stats` for one core ``Submap``. Never raises.

    Adds ``available``: False means the raw camera matrices could not be read
    (an older core, or a submap that blew up), and the caller MUST report the
    count as unknown rather than as 0 — reporting 0 is how this signal died in
    the first place.
    """
    try:
        raw = submap.get_all_poses_world(graph, give_camera_mat=True)
    except Exception as e:  # noqa: BLE001 — QC accounting must not kill a run
        return {"available": False, "n_poses": 0, "n_improper": 0, "n_nonfinite": 0,
                "note": f"raw camera matrices unavailable: {type(e).__name__}: {e}"}
    try:
        stats = improper_rotation_stats(raw)
    except (TypeError, ValueError) as e:
        return {"available": False, "n_poses": 0, "n_improper": 0, "n_nonfinite": 0,
                "note": f"raw camera matrices unusable: {type(e).__name__}: {e}"}
    stats["available"] = True
    return stats


class ImproperRotationError(AssertionError):
    """A pose came back improper from a core that guarantees proper rotations."""


def proper_rigid_poses(poses, where: str = "") -> list:
    """Validate + orthonormalise cam->world poses from ``get_all_poses_world``.

    core (>= 3ea5a03) folds the ``det(R) = -1`` sign into ``(R, t)`` inside
    ``decompose_camera``, so the poses arrive proper and the workers no longer
    repair anything. This is the tripwire for that guarantee: a violation raises
    with a message that names the layer to look at, instead of letting
    ``Rotation.from_matrix`` silently turn an improper matrix into a plausible
    WRONG rotation.

    The SVD projection is kept: RQ output is only orthogonal to float precision,
    and it is the nearest-proper-rotation projection, not a sign repair.
    """
    if poses is None:
        return []
    arr = _as_stack(poses)
    out = []
    for i, T in enumerate(arr):
        R, t = T[:3, :3], T[:3, 3].copy()
        det = float(np.linalg.det(R))
        if not np.isfinite(det) or det <= 0.0:
            raise ImproperRotationError(
                f"improper or degenerate rotation (det={det:.6g}) at pose {i}"
                f"{' of ' + where if where else ''}. core's decompose_camera has "
                f"guaranteed det(R) > 0 since 3ea5a03, so this means the pinned core "
                f"regressed or an unrepaired pose reached us — refusing to emit a "
                f"quaternion that would look valid and be wrong."
            )
        U, _, Vt = np.linalg.svd(R)
        R = U @ np.diag([1.0, 1.0, float(np.sign(np.linalg.det(U @ Vt)))]) @ Vt
        Tn = np.eye(4)
        Tn[:3, :3], Tn[:3, 3] = R, t
        out.append(Tn)
    return out
