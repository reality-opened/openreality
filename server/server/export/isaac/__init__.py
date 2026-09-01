"""Isaac Sim / Isaac Lab USD scene export.

Turns a finished VGGT-SLAM scan into a **metric, gravity-aligned, Z-up USD** environment that
loads directly in NVIDIA Isaac Sim / Isaac Lab — alongside the LeRobot/GR00T trajectory export
(``server/export/``). See ``docs/isaac-export.md``.

Two facts make this net-new work on top of the existing geometry export (it is the
``docs/dataset-export.md`` §5 "metric + gravity alignment" item, plus a USD writer):

- The SLAM world frame is **up-to-scale** and **not gravity-aligned** (``docs/gotchas.md``).
  ``align.py`` estimates the gravity (up) vector from the dominant floor plane and resolves a
  metric scale from a user-supplied factor or a reference object, then composes the 4x4
  ``T_align`` mapping SLAM world -> Isaac world (metric, +Z up, floor at z=0).
- USD's *unauthored* defaults are wrong for robotics (``metersPerUnit`` falls back to 0.01 cm,
  ``upAxis`` to "Y"). ``usd_writer.py`` explicitly authors ``metersPerUnit=1.0`` + ``upAxis="Z"``.

Layout written per scan::

    export/<scan_id>/isaac/
      scene.usd        # environment: UsdGeom.Mesh (+ vertex colors) + UsdPhysics collision
      trajectory.usd   # animated UsdGeomCamera following the scan keyframes
      manifest.json    # T_align, scale provenance, metric / gravity_aligned flags

GPU-free and duck-typed like ``server/export/``: ``align.py`` is pure numpy/scipy (fully
unit-testable without open3d/pxr); the heavy/optional deps stay lazy inside functions
(``open3d`` for Poisson reconstruction in ``mesh.py``, ``pxr`` / ``usd-core`` in ``usd_writer.py``).
"""

from __future__ import annotations

from server.export.isaac.schema import ISAAC_SCHEMA_VERSION


def export_isaac(*args, **kwargs):
    """Lazy proxy for :func:`server.export.isaac.writer.export_isaac`."""
    from server.export.isaac.writer import export_isaac as _fn

    return _fn(*args, **kwargs)


__all__ = ["ISAAC_SCHEMA_VERSION", "export_isaac"]
