"""OpenReality dataset export — robot-training data (LeRobot / Isaac GR00T) + grounding sidecar.

Turns a finished VGGT-SLAM scan into the **OpenReality export** format (the format we own;
see ``docs/dataset-export.md``): a camera-trajectory dataset (RGB video + 6-DoF pose *state*
+ ego-motion *action*) plus a grounded semantic sidecar (scene graph, per-object 3D
grounding, temporally-localized language annotations). ``to_lerobot`` transcodes that tree
into the GR00T-LeRobot v2 layout for fine-tuning.

GPU-free and duck-typed (numpy/scipy/pydantic only), mirroring ``server/scene_report/`` so it
unit-tests with fakes and can run on the CPU broker. Heavy imports (torch, models) stay
inside functions.

Public entry points:
- ``export_scene(solver, detections, report, out_dir, ...)`` — full export from a live Solver.
- ``export_from_record(record, out_dir)`` — GPU-free export from a persisted scene record.
- ``convert_to_groot_lerobot(export_dir, out_dir)`` — transcode to GR00T-LeRobot v2.
- ``build_object_tracks(...)`` / ``write_object_tracks_jsonl(...)`` — optional dynamic-object
  track exporter (tracker-agnostic; default-off dynamics/ sidecar).
"""

from __future__ import annotations

from server.export.schema import SCHEMA_VERSION


def export_scene(*args, **kwargs):
    """Lazy proxy for :func:`server.export.writer.export_scene` (keeps import light)."""
    from server.export.writer import export_scene as _fn

    return _fn(*args, **kwargs)


def export_from_record(*args, **kwargs):
    """Lazy proxy for :func:`server.export.writer.export_from_record`."""
    from server.export.writer import export_from_record as _fn

    return _fn(*args, **kwargs)


def convert_to_groot_lerobot(*args, **kwargs):
    """Lazy proxy for :func:`server.export.to_lerobot.convert_to_groot_lerobot`."""
    from server.export.to_lerobot import convert_to_groot_lerobot as _fn

    return _fn(*args, **kwargs)


def write_human_sidecar(*args, **kwargs):
    """Lazy proxy for :func:`server.export.dynamics_human.write_human_sidecar`.

    Opt-in ``dynamics/human/`` world-motion sidecar (additive, default-off — never invoked by
    ``export_scene``/``export_from_record``). See ``docs/dynamics-human-export.md``.
    """
    from server.export.dynamics_human import write_human_sidecar as _fn

    return _fn(*args, **kwargs)


def build_object_tracks(*args, **kwargs):
    """Lazy proxy for :func:`server.export.dynamics.build_object_tracks`."""
    from server.export.dynamics import build_object_tracks as _fn

    return _fn(*args, **kwargs)


def write_object_tracks_jsonl(*args, **kwargs):
    """Lazy proxy for :func:`server.export.dynamics.write_object_tracks_jsonl`."""
    from server.export.dynamics import write_object_tracks_jsonl as _fn

    return _fn(*args, **kwargs)


__all__ = [
    "SCHEMA_VERSION",
    "export_scene",
    "export_from_record",
    "convert_to_groot_lerobot",
    "write_human_sidecar",
    "build_object_tracks",
    "write_object_tracks_jsonl",
]
