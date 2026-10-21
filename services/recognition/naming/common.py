"""Helpers shared by the naming service and the day agent: logging, vectors, the appearance rule and archive media.

Archive media go through the recognition worker's own download and decode path (Worker.download + select_frames),
so detection boxes recorded by the worker line up with the pixels cut here.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

# The appearance rule: best memory object by cosine of the strong (DINOv2) feature, with a margin over the runner-up.
APPEARANCE_THRESHOLD, APPEARANCE_MARGIN = .65, .10


def log(**fields):
    print(json.dumps({'at': datetime.now(timezone.utc).isoformat(timespec='seconds'), **fields},
                     ensure_ascii=False), flush=True)


def unit(vector, dims=384):
    vector = np.asarray(json.loads(vector) if isinstance(vector, str) else vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if vector.shape != (dims,) or not math.isfinite(norm) or norm <= 0:
        raise ValueError('invalid_vector')
    return vector / norm


def media_ids(value):
    return json.loads(value or '[]') if isinstance(value, str) else list(value or [])


def archive_client():
    import httpx
    return httpx.Client(base_url=os.environ['ARCHIVE_URL'].rstrip('/'), timeout=httpx.Timeout(30, connect=3),
        headers={'Authorization': 'Bearer ' + os.environ['SCENE_ARCHIVE_TOKEN']}, trust_env=False)


def appearance_match(prototypes, feature):
    """Index and cosine of the best and runner-up memory object for a unit strong feature."""
    scores = prototypes @ np.asarray(feature, dtype=np.float32)
    order = np.argsort(-scores)
    best, runner = int(order[0]), (int(order[1]) if len(order) > 1 else None)
    runner_score = float(scores[runner]) if runner is not None else -1.
    return best, float(scores[best]), runner, runner_score, scores


def appearance_accepts(score, runner_score, threshold=APPEARANCE_THRESHOLD, margin=APPEARANCE_MARGIN):
    return score >= threshold and score - runner_score >= margin


class MediaSource:
    """Archive metadata (cached) and decoded frames exactly as the recognition worker produced them."""

    def __init__(self, archive):
        from entity_worker.worker import MAX_MEDIA_BYTES
        self.archive, self.max_bytes = archive, MAX_MEDIA_BYTES
        # Worker.download only touches self.archive; a stand-in avoids Worker.__init__ (it writes the entity store).
        self.downloader = SimpleNamespace(archive=archive)
        self.assets = {}

    def asset(self, media_id):
        """Metadata dict, or None when the archive no longer has the media."""
        if media_id not in self.assets:
            response = self.archive.get('/v1/archive/media/' + media_id)
            if response.status_code != 404:
                response.raise_for_status()
            self.assets[media_id] = None if response.status_code == 404 else response.json()
        return self.assets[media_id]

    def within_bounds(self, asset):
        return 0 < asset['duration_seconds'] <= 60 and 0 < asset['size_bytes'] <= self.max_bytes

    def frames(self, asset, directory, fps=1., max_frames=12):
        """[(offset_s, frame_number, BGR frame)]; Worker.download + select_frames unchanged."""
        from entity_worker.frames import select_frames
        from entity_worker.worker import Worker
        directory = Path(directory); source = directory / 'source.mov'
        Worker.download(self.downloader, asset, source)
        return select_frames(source, directory, asset['duration_seconds'], fps=fps, max_frames=max_frames)
