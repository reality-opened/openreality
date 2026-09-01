"""LiDAR capture-session ingest lane (EXP-45 metric grounding, merged into the OS).

An iPhone uploads a zipped ``CaptureSession`` (``server/oreos/routes_capture.py``:
chunked ``init``/``chunk``/``finalize``, mirroring the splat lane's staging
mechanics — see ``server/oreos/routes_ingest.py``). ``finalize`` validates the zip
(:mod:`server.oreos.capture.zip_validate`) and spawns the standalone ``lidar-recon``
Modal app (:mod:`server.oreos.capture.modal_recon`), which unzips the session,
runs VGGT-SLAM with the LiDAR metric anchor (core's ``vggt_slam.lidar_anchor`` /
``metric_layer``), and persists the result as a ``source="recon_lidar"`` scene via
:mod:`server.oreos.capture.persist_scene` — the SAME ``ModalScenePersistence`` path
every other ingest source uses, so the scene appears in the workspace library
identically.

Modules:
  ``zip_validate``     pure zip-slip + required-entry validation (no flask/modal)
  ``session_assembly``  safe zip -> session-dir extraction (no flask/modal)
  ``persist_scene``     numpy-arrays-in, ``save_scene``-out bridge (no flask/modal,
                        no core/gtsam import) — the parts of this lane that are
                        unit-testable without a GPU
  ``modal_recon``       the GPU job (imports modal + core; not unit-tested directly)
"""
