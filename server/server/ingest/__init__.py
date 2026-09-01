"""Robot-recording ingestion adapters (Oreos ingest).

Turns third-party robot recordings into the pipeline's standard inputs
(timestamp-named JPEG frames + optional TUM reference trajectory) with zero
vendor-SDK dependencies. First adapter: DimOS memory2 SQLite recordings
(EXP-36 provenance; see platform/experiments/exp36_oreos_dimos_spike/).
"""

from server.ingest.dimos_db import DimosDbReader, extract_frames
from server.ingest.video import extract_frames_from_video

__all__ = ["DimosDbReader", "extract_frames", "extract_frames_from_video"]
