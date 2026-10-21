"""Let the user teach Eleagent objects and people by talking while recording ("这是我的蓝色水杯", "这是我同事林小雨").

Loop (every --interval s): read new archive `speech` events after NamingState's cursor (read-only SQLite, like
recognition's other readers). Each new turn goes to the local VLM's listener (vlm.listen) together with the transcript of the recording
so far (up to --context-turns turns of the last --context-seconds), notes of what this service did after earlier
turns (its NamingEvents), and the known memory objects (id + name). The model decides whether the user is teaching
Eleagent about an object in view, the label, and which known object is meant; no names or phrases are matched in code.
For name / rename / same_as with confidence >= --min-confidence:
  1. frame: the capture's strong object events (archive `object`, one per sampled frame) within +-2 s of the turn,
     waiting up to --wait-objects s because speech and recognition are independent pipelines; the frame is decoded
     with the recognition worker's own download/decode path (naming.common.MediaSource), so boxes line up;
  2. look: the VLM sees the frame, the transcript, the known objects and a sheet with one view of each known object
     (a tile cut from one of its views), and returns the object's box, its label (it can see the object, so it
     can undo similar-sounding recognition mistakes; NamingEvent keeps heard_name) and the known object it is, if
     any; its answer on the known object replaces the listener's. The strong detection with the highest IoU (>= .3)
     brings its own DINOv2 embedding; with no overlapping detection the VLM box itself is embedded (DINOv2-S on CPU,
     loaded lazily, same preprocessing as the worker); if the VLM finds nothing the status is no_target. More views
     of the same object come from the other sampled frames within +-SEED_WINDOW_S of the turn: in each, the strong
     detection that matches the grounded one by the appearance rule (up to MAX_SEEDS views in all);
  3. apply, in ONE Neo4j write transaction that first takes the graph-sync / sleep writer lock (SyncState 'cursors',
     LOCK_QUERY); a pending minute maintenance (maintenance_token) makes it retry; while the nightly sleep holds a
     snapshot claim (sleep_token) only brand-new entities are written, changes to existing memory objects wait.
     The target is the known object the model named (look, else listen), else the memory object matching the crop
     by the appearance rule (cosine >= .65, margin >= .10 vs object_vec), else a new memory object
     'object:named-<12 hex>' created from the crop. name / rename give the target the label; same_as adds the crop
     as evidence. Plus an (:ObjectView {source:'naming'}) per view, carrying the view's place in the archive and
     its DINOv2 vector: the look step cuts its known-object sheet from these views. A (:NamingEvent) for every non-none intent,
     and the NamingState cursor.
Test captures (graph-sync TEST_PREFIXES) take the whole path; every node written for them carries test_capture and
every changed entity keeps its previous properties in NamingEvent.prev_json (see naming.cleanup).
People (the listener says kind "person"): the faces the recognition worker found within +-2 s of the turn (archive
`face` events, SFace vectors), minus the owner's own face; one face cluster in view is the person, with several the
VLM picks from the frame with the faces numbered (vlm.pick_person). The target is the known person the listener named,
else the memory person whose face vectors match by the face rule (cosine >= .55, margin >= .08), else a new
'person:named-<12 hex>'. Up to MAX_SEEDS face views of that cluster from +-SEED_WINDOW_S go to the person's face_views
(JSON), so other angles of the face match the person too.
--dry-run reads production (archive, graph, VLM) but writes nothing to Neo4j.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import re
import os
from pathlib import Path
import sqlite3
import tempfile
import time

import numpy as np

from naming.common import (MediaSource, appearance_accepts, appearance_match, archive_client, log, media_ids,
                           unit)
from naming import vlm

STATE_ID = 'naming'
TEST_PREFIXES = ('archive-probe-', 'entity-acceptance-')
LOCK_QUERY = "MERGE (s:SyncState {id:'cursors'}) SET s.locked_at=$now RETURN s"
SCHEMA = [
    'CREATE CONSTRAINT naming_event_id IF NOT EXISTS FOR (n:NamingEvent) REQUIRE n.id IS UNIQUE',
    'CREATE INDEX naming_event_created IF NOT EXISTS FOR (n:NamingEvent) ON (n.created_at)',
    'CREATE INDEX naming_event_capture IF NOT EXISTS FOR (n:NamingEvent) ON (n.capture_id)',
    'CREATE CONSTRAINT naming_state_id IF NOT EXISTS FOR (s:NamingState) REQUIRE s.id IS UNIQUE',
    'CREATE CONSTRAINT object_view_id IF NOT EXISTS FOR (v:ObjectView) REQUIRE v.id IS UNIQUE',
    'CREATE INDEX object_view_memory IF NOT EXISTS FOR (v:ObjectView) ON (v.memory_id)',
]
ACTIONS = ('name', 'rename', 'same_as')
WINDOW_S, MIN_IOU = 2., .3
# More views of a just-named object; near-identical ones dropped. Views are compared by their DINOv2-S vectors: frames
# of an object that did not move or turn score >= ~.95 there, while other sides or lighting fall below.
SEED_WINDOW_S, MAX_SEEDS, SEED_DEDUP = 5., 12, .95
TILE, SHEET_COLUMNS, SHEET_MAX = 160, 6, 48           # the look step's sheet of known-object views
FACE_DIM, FACE_THRESHOLD, FACE_MARGIN, FACE_VIEW_MIN = 128, .55, .08, .5   # the face rule; views of one person
OWNER_ID = 'person:owner'
STRONG_MODEL = 'dinov2-small'   # model of a view whose detection did not say which DINOv2-S export made its vector
# Entity properties a naming action may change; their previous values go to NamingEvent.prev_json for cleanup.
TRACKED = ('name', 'name_source', 'named_at', 'named_utterance', 'naming_event_id', 'object_vec', 'object_vec_n',
           'media_ids', 'occurrences', 't_last', 't_end_us', 'updated_at')
PERSON_TRACKED = ('name', 'name_source', 'named_at', 'named_utterance', 'naming_event_id', 'face_vec', 'face_vec_n',
                  'face_views', 'media_ids', 'occurrences', 't_last', 't_end_us', 'updated_at')


class RetryLater(Exception):
    """Leave the cursor on this speech event and try again next cycle (lock busy, VLM down, ...)."""


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec='microseconds')


def iso_us(value):
    instant = datetime.fromisoformat(value.replace('Z', '+00:00'))
    return int(instant.timestamp() * 1_000_000)


def digest(*parts, size=32):
    return hashlib.sha256(json.dumps(parts, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()[:size]


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0., x2 - x1) * max(0., y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.


def prominence(box, width, height):
    """Large and central first: area fraction times (1 - normalised distance of the centre from the frame centre)."""
    area = (box[2] - box[0]) * (box[3] - box[1]) / float(width * height)
    cx, cy = (box[0] + box[2]) / 2 / width - .5, (box[1] + box[3]) / 2 / height - .5
    return area * max(0., 1. - math.hypot(cx, cy) / math.hypot(.5, .5))


def is_test(capture_id):
    return (capture_id or '').startswith(TEST_PREFIXES)


def ro(path):
    db = sqlite3.connect(f'file:{path}?mode=ro', uri=True, timeout=10)
    db.row_factory = sqlite3.Row
    return db


class StrongCrop:
    """The recognition worker's DINOv2-S appearance feature for an arbitrary box, on CPU (no YOLOE, no GPU)."""

    def __init__(self, weights, threads=2):
        import torch
        from transformers import AutoModel
        from entity_worker.objects import ObjectDetector
        weights = Path(weights)
        manifest = json.loads((weights / 'manifest-object-yoloe.json').read_text())
        for key in ('dinov2', 'dinov2_config', 'dinov2_preprocessor'):
            item = manifest['files'][key]
            if hashlib.sha256((weights / item['filename']).read_bytes()).hexdigest() != item['sha256']:
                raise ValueError('object_model_checksum_mismatch:' + key)
        self.version = 'dinov2-small-' + manifest['files']['dinov2']['sha256'][:16]
        torch.set_num_threads(threads)
        # ObjectDetector._appearance_inputs needs only these two attributes; skip its YOLOE load.
        self.stub = object.__new__(ObjectDetector)
        self.stub.preprocess_config = json.loads((weights / 'dinov2-small' / 'preprocessor_config.json').read_text())
        self.stub.device = 'cpu'
        self.inputs = ObjectDetector._appearance_inputs
        self.model = AutoModel.from_pretrained(weights / 'dinov2-small', local_files_only=True, trust_remote_code=False).eval()
        self.torch = torch

    def embed(self, rgb, box):
        from PIL import Image
        x1, y1, x2, y2 = box
        crop = rgb[int(y1):math.ceil(y2), int(x1):math.ceil(x2)]
        if crop.size == 0 or min(crop.shape[:2]) < 8:
            return None
        with self.torch.inference_mode():
            output = self.model(**self.inputs(self.stub, [Image.fromarray(np.ascontiguousarray(crop))]))
            feature = output.last_hidden_state[0, 0, :].float().cpu().numpy()
        norm = float(np.linalg.norm(feature))
        return feature / norm if math.isfinite(norm) and norm > 0 else None


class Graph:
    def __init__(self, driver, dry_run):
        self.driver, self.dry_run = driver, dry_run

    def read(self, query, **params):
        with self.driver.session() as session:
            return session.execute_read(lambda tx: tx.run(query, **params).data())

    def schema(self):
        if self.dry_run:
            return
        with self.driver.session() as session:
            for statement in SCHEMA:
                session.run(statement).consume()

    def state(self):
        rows = self.read('MATCH (s:NamingState {id:$id}) RETURN properties(s) AS s', id=STATE_ID)
        return rows[0]['s'] if rows else None

    def set_cursor(self, cursor, counters):
        if self.dry_run:
            return
        with self.driver.session() as session:
            session.execute_write(lambda tx: tx.run('MERGE (s:NamingState {id:$id}) SET s.event_cursor=$c, s.updated_at=$now, s += $n',
                id=STATE_ID, c=cursor, now=now_iso(), n=counters).consume())



def typical_first(views):
    """Views most like the object's other views first (they agree with each other), compared only among vectors of one
    size; views without such a vector follow, newest first."""
    sizes = {}
    for view in views:
        if isinstance(view.get('vector'), (list, tuple)) and view['vector']:
            sizes.setdefault(len(view['vector']), []).append(view)
    alike = max(sizes.values(), key=len) if sizes else []
    others = sorted((v for v in views if not any(v is a for a in alike)), key=lambda v: v.get('at') or '', reverse=True)
    if len(alike) > 1:
        vectors = np.stack([np.asarray(v['vector'], np.float64) for v in alike])
        vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
        alike = [alike[k] for k in np.argsort(-(vectors @ vectors.T).sum(1))]
    return alike + others


def memory_objects(tx):
    return tx.run("""MATCH (e:Entity) WHERE e.layer='memory' AND e.kind='object'
        RETURN e.id AS id, e.name AS name, e.name_source AS name_source, e.object_vec AS object_vec,
               """ + ', '.join(f'e.{key} AS `p_{key}`' for key in TRACKED if key != 'name') + ' ORDER BY e.id').data()


def decide(tx, item):
    """Pick the target memory object for one grounded action (read-only; runs inside the write transaction)."""
    memory = memory_objects(tx)
    feature = item.get('strong')
    with_vec = []
    for row in memory:
        try:
            if row['object_vec'] is not None:
                with_vec.append((row, unit(row['object_vec'])))
        except ValueError:
            pass
    rule = None
    if feature is not None and with_vec:
        best, score, _, runner_score, _ = appearance_match(np.stack([v for _, v in with_vec]), feature)
        item['strong_cosine'] = round(score, 4); item['runner_up'] = round(runner_score, 4)
        if appearance_accepts(score, runner_score):
            rule = with_vec[best][0]
    by_id = {row['id']: row for row in memory}
    if item.get('known_id') in by_id:
        return by_id[item['known_id']], 'model'
    if rule is not None:
        return rule, 'appearance'
    return (None, 'create') if item.get('name') else (None, 'no_target')


def face_vectors(row):
    """A person's face vectors: face_vec (sleep's running mean) and the face_views naming kept."""
    vectors = []
    for value in [row.get('face_vec')] + (json.loads(row['face_views']) if row.get('face_views') else []):
        try:
            vectors.append(unit(value, dims=FACE_DIM))
        except (TypeError, ValueError):
            pass
    return vectors


def memory_people(tx):
    return tx.run(f"""MATCH (e:Entity) WHERE e.layer='memory' AND e.kind='person' AND e.id <> '{OWNER_ID}'
        RETURN e.id AS id, e.name AS name, e.face_vec AS face_vec, e.face_views AS face_views,
               """ + ', '.join(f'e.{key} AS `{key}`' for key in PERSON_TRACKED if key not in ('name', 'face_vec', 'face_views'))
        + ' ORDER BY e.id').data()


def decide_person(tx, item):
    """The known person the listener named, else the one whose faces match the views (face rule), else create."""
    people = memory_people(tx)
    by_id = {row['id']: row for row in people}
    if item.get('known_id') in by_id:
        return by_id[item['known_id']], 'model'
    views = np.stack([np.asarray(v) for v in item['face_views']])
    scored = sorted(((float((np.stack(vectors) @ views.T).max()), row) for row in people if (vectors := face_vectors(row))),
                    key=lambda s: -s[0])
    if scored:
        item['strong_cosine'] = round(scored[0][0], 4)
        item['runner_up'] = round(scored[1][0], 4) if len(scored) > 1 else None
        if scored[0][0] >= FACE_THRESHOLD and (len(scored) < 2 or scored[0][0] - scored[1][0] >= FACE_MARGIN):
            return scored[0][1], 'appearance'
    return (None, 'create') if item.get('name') else (None, 'no_target')


def said_over_description(item, target):
    """same_as a known one that only carries Eleagent's own description, while the user said another name: the
    user's name wins (a description is a guess from looks and talk, not a name)."""
    return (item['intent'] == 'same_as' and target.get('name_source') != 'user' and bool(item.get('name'))
            and item['name'] != target.get('name'))


def shares_words(heard, said):
    """A repair of misheard words keeps some of them: a common CJK character, latin word or number."""
    def words(text):
        text = (text or '').lower()
        return set(re.findall(r'[\u3400-\u9fff]', text)) | set(re.findall(r'[a-z0-9]{2,}', text))
    return bool(words(heard) & words(said))


def note_of(event):
    """What this service did after a turn, in words for the model's transcript."""
    if not event:
        return None
    name, status = event.get('name'), event.get('status')
    noun = 'person' if event.get('kind') == 'person' else 'object'
    if status == 'named' and str(event.get('how') or '').startswith('created'):
        return f'remembered a new {noun} "{name}"'
    if status in ('named', 'renamed'):
        return f'labelled a {noun} "{name}"'
    if status == 'same_as':
        return f'added this view to "{name}"'
    if status == 'no_target':
        return f'understood teaching "{name}" but did not find the {noun} in the frame'
    return None


class Service:
    def __init__(self, graph, args, media=None):
        self.graph, self.args, self.media = graph, args, media
        self.cursor = None
        self.counters = {}
        self.strong = None
        self.cache = {}              # speech event_id -> analysed item (kept across lock retries)
        self.vlm_down = False
        self.failures = {}
        self.tiles = {}              # view id -> tile (BGR) or None when its frame could not be cut

    # --- archive -------------------------------------------------------------------------------------------------

    def archive_top(self, db):
        return int(db.execute('SELECT coalesce(max(rowid), 0) FROM archive_events').fetchone()[0])

    def context(self, db, capture_id, start_us):
        """Earlier turns of this recording (oldest first) with notes of what this service did after them."""
        since = start_us - int(self.args.context_seconds * 1e6)
        rows = db.execute("""SELECT event_id, body_json FROM archive_events WHERE capture_id=? AND kind='speech'
            AND start_us<? AND start_us>=? ORDER BY start_us DESC LIMIT ?""", (capture_id, start_us, since, self.args.context_turns)).fetchall()
        notes = {row['event_id']: row for row in self.graph.read("""MATCH (n:NamingEvent {capture_id:$c})
            RETURN n.event_id AS event_id, n.status AS status, n.name AS name, n.matched_by AS how, n.kind AS kind""", c=capture_id)}
        turns = []
        for row in reversed(rows):
            body = json.loads(row['body_json'])
            turns.append({'speaker': body.get('speaker_entity_id'), 'text': body.get('text') or '',
                          'note': note_of(notes.get(row['event_id']))})
        return turns

    def known(self):
        return self.graph.read("""MATCH (e:Entity) WHERE e.layer='memory' AND e.kind='object' AND e.name IS NOT NULL
            RETURN e.id AS id, e.name AS name, coalesce(e.name_source, 'eleagent') AS name_source
            ORDER BY coalesce(e.t_last, '') DESC LIMIT $n""", n=self.args.known_objects)

    def known_people(self):
        return self.graph.read(f"""MATCH (e:Entity) WHERE e.layer='memory' AND e.kind='person' AND e.name IS NOT NULL
            AND e.id <> '{OWNER_ID}' RETURN e.id AS id, e.name AS name, coalesce(e.name_source, 'eleagent') AS name_source
            ORDER BY coalesce(e.t_last, '') DESC LIMIT $n""",
            n=self.args.known_objects)

    def owner_faces(self):
        rows = self.graph.read(f"MATCH (e:Entity {{id:'{OWNER_ID}'}}) RETURN e.face_vec AS face_vec, e.face_views AS face_views")
        return face_vectors(rows[0]) if rows else []

    def face_frames(self, capture_id, start_us, end_us):
        """The recognition worker's face events of a capture: per sampled frame, the faces that carry a vector."""
        db = ro(self.args.archive_db)
        try:
            rows = db.execute("""SELECT start_us, body_json, refs_json FROM archive_events
                WHERE capture_id=? AND kind='face' AND start_us BETWEEN ? AND ? ORDER BY start_us""",
                (capture_id, start_us, end_us)).fetchall()
        finally:
            db.close()
        frames = []
        for row in rows:
            ref = (json.loads(row['refs_json'] or '[]') or [{}])[0]
            body = json.loads(row['body_json'])
            faces = []
            for o in body.get('observations') or []:
                try:
                    vec = unit(o['embedding'], dims=FACE_DIM)
                except (KeyError, TypeError, ValueError):
                    continue
                if o.get('bbox_xywh'):
                    x, y, w, h = (float(v) for v in o['bbox_xywh'])
                    faces.append({'entity_id': o.get('entity_id') or o.get('sample_id'), 'sample_id': o.get('sample_id'),
                                  'box': [x, y, x + w, y + h], 'vec': vec,
                                  'frame_width': o.get('frame_width'), 'frame_height': o.get('frame_height')})
            if ref.get('media_id') and ref.get('start_offset_seconds') is not None:
                frames.append({'start_us': row['start_us'], 'media_id': ref['media_id'], 'offset': float(ref['start_offset_seconds']),
                               'fps': float(body.get('sample_fps') or 1.), 'max_frames': int(body.get('max_frames_per_clip') or 12),
                               'faces': faces})
        return frames

    def object_frames(self, capture_id, start_us, end_us):
        db = ro(self.args.archive_db)
        try:
            rows = db.execute("""SELECT start_us, body_json, refs_json FROM archive_events
                WHERE capture_id=? AND kind='object' AND start_us BETWEEN ? AND ? ORDER BY start_us""",
                (capture_id, start_us, end_us)).fetchall()
        finally:
            db.close()
        frames = []
        for row in rows:
            ref = (json.loads(row['refs_json'] or '[]') or [{}])[0]
            body = json.loads(row['body_json'])
            if ref.get('media_id') and ref.get('start_offset_seconds') is not None:
                frames.append({'start_us': row['start_us'], 'media_id': ref['media_id'], 'offset': float(ref['start_offset_seconds']),
                               'observations': [o for o in body.get('observations') or [] if o.get('bbox')],
                               'fps': float(body.get('sample_fps') or 1.), 'max_frames': int(body.get('max_frames_per_clip') or 12)})
        return frames

    def decode(self, cache, media_id, fps, max_frames):
        """Frames of one media as the recognition worker sampled them, decoded once per naming."""
        key = (media_id, fps, max_frames)
        if key not in cache:
            cache[key] = []
            asset = self.media.asset(media_id)
            if asset is not None and self.media.within_bounds(asset):
                with tempfile.TemporaryDirectory(prefix='naming-') as directory:
                    cache[key] = self.media.frames(asset, directory, fps=fps, max_frames=max_frames)
        return cache[key]

    def tile(self, view, cache):
        """A square crop of one view's box (letterboxed), or None."""
        if view['id'] not in self.tiles:
            import cv2
            tile = None
            try:
                decoded = self.decode(cache, view['media_id'], 1., 20)
                near = min(decoded, key=lambda entry: abs(entry[0] - float(view['offset_s']))) if decoded else None
                if near is not None and abs(near[0] - float(view['offset_s'])) <= .6:
                    frame = near[2]; height, width = frame.shape[:2]
                    x1, y1, x2, y2 = (float(v) for v in view['box'])
                    pad = .08 * max(x2 - x1, y2 - y1)
                    x1, y1 = max(0, int(x1 - pad)), max(0, int(y1 - pad))
                    x2, y2 = min(width, math.ceil(x2 + pad)), min(height, math.ceil(y2 + pad))
                    if x2 - x1 >= 8 and y2 - y1 >= 8:
                        crop = frame[y1:y2, x1:x2]
                        scale = TILE / max(crop.shape[:2])
                        crop = cv2.resize(crop, (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale))))
                        tile = np.full((TILE, TILE, 3), 40, np.uint8)
                        oy, ox = (TILE - crop.shape[0]) // 2, (TILE - crop.shape[1]) // 2
                        tile[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop
            except Exception as error:
                log(event='tile_failed', view=view['id'], error=type(error).__name__)
            self.tiles[view['id']] = tile
        return self.tiles[view['id']]

    def sheet(self, known):
        """(JPEG, numbers shown) of one view per known object, numbered as in the known list; None without views."""
        import cv2
        # Views of any model: a tile needs only the view's place in the archive. curated=false marks views an earlier
        # curation left out (filed under the wrong object, or the less typical side of two look-alike objects).
        rows = self.graph.read("""MATCH (p:ObjectView) WHERE p.memory_id IN $ids
            AND p.media_id IS NOT NULL AND p.box IS NOT NULL AND p.offset_s IS NOT NULL AND coalesce(p.curated, true)
            RETURN p.id AS id, p.memory_id AS memory_id, p.source AS source, p.media_id AS media_id, p.offset_s AS offset_s,
                   p.box AS box, toString(p.created_at) AS at, p.vector AS vector""", ids=[o['id'] for o in known])
        by_object = {}
        for row in rows:
            by_object.setdefault(row['memory_id'], []).append(row)
        cache, tiles = {}, []
        for number, obj in enumerate(known, 1):
            # The user's own naming views first (oldest first), then the object's most typical views.
            views = by_object.get(obj['id'], [])
            mine = sorted((r for r in views if r['source'] == 'naming'), key=lambda r: r['at'] or '')
            views = mine + typical_first([r for r in views if r['source'] != 'naming'])
            for view in views[:3]:
                tile = self.tile(view, cache)
                if tile is not None:
                    tiles.append((number, tile)); break
            if len(tiles) >= SHEET_MAX:
                break
        if not tiles:
            return None
        rows_n = math.ceil(len(tiles) / SHEET_COLUMNS)
        sheet = np.full((rows_n * TILE, min(len(tiles), SHEET_COLUMNS) * TILE, 3), 255, np.uint8)
        for index, (number, tile) in enumerate(tiles):
            y, x = (index // SHEET_COLUMNS) * TILE, (index % SHEET_COLUMNS) * TILE
            tile = tile.copy()
            cv2.rectangle(tile, (0, 0), (TILE - 1, TILE - 1), (255, 255, 255), 2)
            cv2.rectangle(tile, (0, 0), (44, 30), (0, 0, 0), -1)
            cv2.putText(tile, str(number), (4, 23), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2, cv2.LINE_AA)
            sheet[y:y + TILE, x:x + TILE] = tile
        return cv2.imencode('.jpg', sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes(), [n for n, _ in tiles]

    def seed_views(self, item, cache):
        """The grounded view plus the same object in the other sampled frames around the turn (appearance rule)."""
        seeds = [{'media_id': item['media_id'], 'offset_s': item['offset_s'], 'box': item['box'], 'strong': item['strong'],
                  'strong_source': item.get('strong_source'), 'label': item.get('label'), 'sample_id': item.get('sample_id')}]
        lo, hi = item['start_us'] - int(SEED_WINDOW_S * 1e6), item['end_us'] + int(SEED_WINDOW_S * 1e6)
        candidates = []
        for frame in self.object_frames(item['capture_id'], lo, hi):
            if frame['media_id'] == item['media_id'] and abs(frame['offset'] - item['offset_s']) < 1e-3:
                continue
            observed = []
            for o in frame['observations']:
                try:
                    observed.append((o, unit(o['embedding'])))
                except (KeyError, TypeError, ValueError):
                    pass
            if not observed:
                continue
            best, score, _, runner, _ = appearance_match(np.stack([v for _, v in observed]), item['strong'])
            if appearance_accepts(score, runner):
                candidates.append((abs(frame['start_us'] - item['frame_us']), frame, observed[best][0], observed[best][1], score))
        for _, frame, observation, strong, score in sorted(candidates, key=lambda c: c[0]):
            if len(seeds) >= MAX_SEEDS:
                break
            try:
                decoded = self.decode(cache, frame['media_id'], frame['fps'], frame['max_frames'])
            except Exception as error:
                log(event='seed_decode_failed', media_id=frame['media_id'], error=type(error).__name__); continue
            exact = [entry for entry in decoded if abs(entry[0] - frame['offset']) < 1e-4]
            if not exact:
                continue
            offset, _, image = exact[0]
            # The tile is cut from this decoded frame later: the recorded box must belong to these pixels.
            if observation.get('frame_width') != image.shape[1] or observation.get('frame_height') != image.shape[0]:
                continue
            if any(float(np.dot(strong, seed['strong'])) >= SEED_DEDUP for seed in seeds):
                continue
            seeds.append({'media_id': frame['media_id'], 'offset_s': round(float(offset), 6), 'box': [round(float(v), 2) for v in observation['bbox']],
                          'strong': strong, 'strong_source': observation.get('feature_model'), 'label': observation.get('label'),
                          'sample_id': observation.get('sample_id'), 'strong_cosine': round(score, 4)})
        return seeds

    # --- analysis ------------------------------------------------------------------------------------------------

    def analyse(self, db, row):
        """Intent (+ frame, grounding and embeddings for actions) for one speech event; pure reads."""
        started = time.time()
        body, refs = json.loads(row['body_json']), json.loads(row['refs_json'] or '[]')
        text = (body.get('text') or '').strip()
        item = {'event_id': row['event_id'], 'capture_id': row['capture_id'], 'utterance': text,
                'speaker_entity_id': body.get('speaker_entity_id'), 'start_us': row['start_us'], 'end_us': row['end_us'],
                'started_at': row['started_at'], 'ended_at': row['ended_at'], 'analysed_at': started, 'test_capture': row['capture_id'] if is_test(row['capture_id']) else None,
                'timings': {'speech_event_age_ms': round((time.time() - iso_us(row['recorded_at']) / 1e6) * 1000),
                            'speech_lag_ms': round((iso_us(row['recorded_at']) - row['end_us']) / 1000)}}
        if not text:
            item.update(intent='none', name=None, referent=None, confidence=0.); return item
        turns, known, people = self.context(db, row['capture_id'], row['start_us']), self.known(), self.known_people()
        try:
            got, seconds = vlm.listen(self.args.vlm_url, self.args.vlm_model, text, turns, known, body.get('speaker_entity_id'),
                                      people=people)
        except Exception as error:
            raise RetryLater(f'intent: {type(error).__name__}: {error}') from error
        item.update(got); item['context'] = turns; item['known'] = known; item['heard'] = dict(got)
        item['timings']['intent_ms'] = round(seconds * 1000)
        if item['intent'] not in ACTIONS or item['confidence'] < self.args.min_confidence:
            return item
        if item.get('kind') == 'person' and self.ground_person(item, refs) == 'object':
            # The words sounded like an introduction, but no face in view is the one meant: teach it as an object.
            item['kind'], item['person_fallback'] = 'object', item.pop('reason', None)
            item.pop('status', None)
        if item.get('kind') != 'person':
            self.ground(item, refs)
        return item

    def ground_person(self, item, refs):
        """The face of the person introduced: the one face cluster in view besides the owner's, or the model's pick."""
        middle = (item['start_us'] + item['end_us']) / 2
        lo, hi = item['start_us'] - int(SEED_WINDOW_S * 1e6), item['end_us'] + int(SEED_WINDOW_S * 1e6)
        near = lambda f: item['start_us'] - WINDOW_S * 1e6 <= f['start_us'] <= item['end_us'] + WINDOW_S * 1e6
        turn_media = {ref.get('media_id') for ref in refs if ref.get('media_id')}
        deadline, waited = time.monotonic() + self.args.wait_objects, time.monotonic()
        while True:
            frames = self.face_frames(item['capture_id'], lo, hi)
            if any(frame['media_id'] in turn_media for frame in frames) or time.monotonic() >= deadline:
                break
            time.sleep(1.)
        item['timings']['wait_objects_ms'] = round((time.monotonic() - waited) * 1000)
        owner = self.owner_faces()
        others = lambda frame: [face for face in frame['faces']
                                if not (owner and max(float(np.dot(face['vec'], o)) for o in owner) >= FACE_THRESHOLD)]
        groups = {}
        for frame in frames:
            if near(frame):
                for face in others(frame):
                    groups.setdefault(face['entity_id'], []).append((frame, face))
        if not groups:
            item['reason'] = 'no_face'; return 'object'
        # Number the faces on the frame where most are seen and let the model pick (or see nobody is introduced).
        frame = max((f for f in frames if near(f)), key=lambda f: (len(others(f)), -abs(f['start_us'] - middle)))
        faces = others(frame)
        image = [e for e in self.decode({}, frame['media_id'], frame['fps'], frame['max_frames']) if abs(e[0] - frame['offset']) < 1e-4]
        if not image:
            item['status'] = 'no_target'; item['reason'] = 'no_frames'; return
        import cv2
        canvas = image[0][2].copy()
        for number, face in enumerate(faces, 1):
            x1, y1, x2, y2 = (int(v) for v in face['box'])
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(canvas, str(number), (x1 + 2, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)
        jpeg = cv2.imencode('.jpg', canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()
        try:
            number, reply, seconds = vlm.pick_person(self.args.vlm_url, self.args.vlm_model, jpeg, len(faces),
                item.get('context') or (), (item['speaker_entity_id'], item['utterance']), item['heard'])
        except Exception as error:
            raise RetryLater(f'pick: {type(error).__name__}: {error}') from error
        item['timings']['grounding_ms'] = round(seconds * 1000)
        item['vlm_reply'] = reply
        if number is None:
            item['reason'] = 'not_picked'; return 'object'
        key, grounding = faces[number - 1]['entity_id'], 'face+pick'
        chosen = groups[key]
        frame, face = min(chosen, key=lambda pair: abs(pair[0]['start_us'] - middle))
        reference = np.sum([f['vec'] for _, f in chosen], axis=0); reference = reference / np.linalg.norm(reference)
        # Views: that face over the whole window (the worker may split one person into clusters, so similar faces too).
        candidates = [(abs(f['start_us'] - middle), f, g) for f in frames for g in others(f)
                      if g['entity_id'] == key or float(np.dot(g['vec'], reference)) >= FACE_VIEW_MIN]
        views = []
        for _, f, g in sorted(candidates, key=lambda c: c[0]):
            if len(views) >= MAX_SEEDS:
                break
            if all(float(np.dot(g['vec'], v['vec'])) < SEED_DEDUP for v in views):
                views.append({'media_id': f['media_id'], 'vec': g['vec']})
        item.update(media_id=frame['media_id'], offset_s=round(frame['offset'], 6), frame_us=frame['start_us'],
                    frame_width=face.get('frame_width'), frame_height=face.get('frame_height'),
                    box=[round(float(v), 2) for v in face['box']], grounding=grounding, face_entity_id=key,
                    face_views=[[round(float(x), 4) for x in v['vec']] for v in views], face_media=[v['media_id'] for v in views])

    def ground(self, item, refs):
        # 1. frame: strong detections of this capture within +-2 s of the turn (wait for recognition to catch up).
        lo, hi = item['start_us'] - int(WINDOW_S * 1e6), item['end_us'] + int(WINDOW_S * 1e6)
        turn_media = {ref.get('media_id') for ref in refs if ref.get('media_id')}
        deadline, waited = time.monotonic() + self.args.wait_objects, time.monotonic()
        while True:
            frames = self.object_frames(item['capture_id'], lo, hi)
            if any(frame['media_id'] in turn_media for frame in frames) or time.monotonic() >= deadline:
                break
            time.sleep(1.)
        item['timings']['wait_objects_ms'] = round((time.monotonic() - waited) * 1000)
        middle = (item['start_us'] + item['end_us']) / 2
        usable = [f for f in frames if any(o.get('embedding') is not None for o in f['observations'])] or \
                 [f for f in frames if f['observations']] or frames
        if usable:
            chosen = min(usable, key=lambda f: abs(f['start_us'] - middle))
            media_id, offset, fps, max_frames, observations = chosen['media_id'], chosen['offset'], chosen['fps'], chosen['max_frames'], chosen['observations']
            frame_us = chosen['start_us']
        elif refs and refs[0].get('media_id'):
            # No strong detections at all: sample the turn's own media at the worker's default rate.
            ref = refs[0]
            media_id, fps, max_frames, observations = ref['media_id'], 1., 12, []
            offset = (float(ref.get('start_offset_seconds') or 0) + float(ref.get('end_offset_seconds') or 0)) / 2
            frame_us = int(middle)
        else:
            item['status'] = 'no_target'; item['reason'] = 'no_media'; return
        item.update(media_id=media_id, frame_us=frame_us)
        # 2. decode with the worker's own path, so recorded boxes line up with these pixels.
        t0 = time.perf_counter()
        try:
            asset = self.media.asset(media_id)
            if asset is None or not self.media.within_bounds(asset):
                item['status'] = 'no_target'; item['reason'] = 'media_missing'; return
            cache = {}
            decoded = self.decode(cache, media_id, fps, max_frames)
        except Exception as error:
            raise RetryLater(f'decode: {type(error).__name__}: {error}') from error
        if not decoded:
            item['status'] = 'no_target'; item['reason'] = 'no_frames'; return
        exact = [entry for entry in decoded if abs(entry[0] - offset) < 1e-4]
        frame_offset, _, frame = exact[0] if exact else min(decoded, key=lambda entry: abs(entry[0] - offset))
        if not exact:
            observations = []  # recorded boxes belong to another frame
        height, width = frame.shape[:2]
        observations = [o for o in observations if o.get('frame_width') == width and o.get('frame_height') == height]
        rgb = np.ascontiguousarray(frame[:, :, ::-1])
        import cv2
        jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()
        item.update(offset_s=round(float(frame_offset), 6), frame_width=int(width), frame_height=int(height),
                    frame_at=asset.get('captured_at'))
        item['timings']['frame_ms'] = round((time.perf_counter() - t0) * 1000)
        # 3. look: VLM box, label and known object, then the best-overlapping strong detection.
        t0 = time.perf_counter()
        try:
            sheet = self.sheet(item.get('known') or [])
        except Exception as error:
            sheet = None
            log(event='sheet_failed', error=type(error).__name__, detail=str(error)[:200])
        item['timings']['sheet_ms'] = round((time.perf_counter() - t0) * 1000)
        item['sheet_objects'] = len(sheet[1]) if sheet else 0
        try:
            box, said, known_id, reply, seconds = vlm.look(self.args.vlm_url, self.args.vlm_model, jpeg, width, height,
                item.get('context') or (), (item['speaker_entity_id'], item['utterance']), item['heard'], item.get('known') or (),
                sheet=sheet)
        except Exception as error:
            raise RetryLater(f'look: {type(error).__name__}: {error}') from error
        item['timings']['grounding_ms'] = round(seconds * 1000)
        item['vlm_box'] = [round(v, 2) for v in box] if box else None
        item['vlm_reply'] = reply
        # The label is the user's: the eyes may repair words the transcript misheard, not call the thing otherwise.
        if said and said != item.get('name') and shares_words(item.get('name'), said):
            item['heard_name'], item['name'] = item.get('name'), said
        if box is not None and not (item['intent'] == 'same_as' and item.get('known_id') and known_id is None):
            # The eyes saw the object and the known objects' views: their answer on which one it is stands, unless
            # the user said it is a known one and the eyes did not recognise any.
            item['known_id'] = known_id
        target = None
        if box is not None:
            scored = sorted(((iou(box, o['bbox']), prominence(o['bbox'], width, height), o) for o in observations),
                            key=lambda s: (-s[0], -s[1]))
            # Near-ties (within .05 IoU): prefer the more prominent (large, central) detection.
            top = [s for s in scored if s[0] >= MIN_IOU and s[0] >= scored[0][0] - .05] if scored else []
            if top:
                best = max(top, key=lambda s: (s[2].get('embedding') is not None, s[1]))
                target = {'box': best[2]['bbox'], 'observation': best[2], 'iou': round(best[0], 3), 'grounding': 'vlm+detection'}
            else:
                target = {'box': box, 'observation': None, 'iou': round(scored[0][0], 3) if scored else None, 'grounding': 'vlm'}
        if target is None:
            # No guessing: a "most prominent detection" fallback named a chair in a dry run. The user can repeat the name.
            item['status'] = 'no_target'; item['reason'] = 'not_grounded'; return
        t0 = time.perf_counter()
        observation = target['observation']
        strong = None
        if observation is not None and observation.get('embedding') is not None:
            try:
                strong = unit(observation['embedding'])
            except ValueError:
                strong = None
        if strong is None:
            if self.strong is None:
                started_load = time.perf_counter()
                self.strong = StrongCrop(self.args.weights, threads=self.args.threads)
                log(event='strong_loaded', version=self.strong.version, seconds=round(time.perf_counter() - started_load, 1))
            strong = self.strong.embed(rgb, target['box'])
            target['strong_source'] = self.strong.version + ':cpu-crop'
        else:
            target['strong_source'] = observation.get('feature_model')
        item['timings']['embed_ms'] = round((time.perf_counter() - t0) * 1000)
        if strong is None:
            item['status'] = 'no_target'; item['reason'] = 'crop_unusable'; return
        item.update(box=[round(float(v), 2) for v in target['box']], iou=target['iou'], grounding=target['grounding'],
                    strong_source=target['strong_source'], strong=strong,
                    label=(observation or {}).get('label'), sample_id=(observation or {}).get('sample_id'))
        t0 = time.perf_counter()
        item['seeds'] = self.seed_views(item, cache)
        item['timings']['seeds_ms'] = round((time.perf_counter() - t0) * 1000)

    # --- apply ---------------------------------------------------------------------------------------------------

    def event_row(self, item, status, memory_id=None, prev=None, matched_by=None):
        row = {'id': 'naming_' + digest(item['event_id']), 'event_id': item['event_id'], 'capture_id': item['capture_id'],
               'speaker_entity_id': item.get('speaker_entity_id'), 'utterance': item['utterance'], 'intent': item['intent'],
               'name': item.get('name'), 'heard_name': item.get('heard_name'), 'referent': item.get('referent'),
               'known_id': item.get('known_id'), 'heard_known_id': (item.get('heard') or {}).get('known_id'),
               'situation': item.get('situation'), 'kind': item.get('kind') or 'object', 'face_entity_id': item.get('face_entity_id'),
               'confidence': item.get('confidence'),
               'memory_id': memory_id, 'status': status, 'reason': item.get('reason'), 'matched_by': matched_by,
               'box': item.get('box'), 'vlm_box': item.get('vlm_box'), 'vlm_reply': item.get('vlm_reply'), 'iou': item.get('iou'), 'grounding': item.get('grounding'),
               'label': item.get('label'), 'media_id': item.get('media_id'), 'offset_s': item.get('offset_s'),
               'frame_width': item.get('frame_width'), 'frame_height': item.get('frame_height'),
               'strong_cosine': item.get('strong_cosine'), 'runner_up': item.get('runner_up'),
               'context_json': json.dumps(item.get('context') or [], ensure_ascii=False),
               'prev_json': json.dumps(prev, ensure_ascii=False) if prev is not None else None,
               'views': len(item.get('seeds') or item.get('face_views') or []) or None, 'sheet_objects': item.get('sheet_objects'),
               'created_at': now_iso(), 'test_capture': item.get('test_capture')}
        return {k: v for k, v in row.items() if v is not None}

    def view_rows(self, item, memory_id):
        """One view per seed; the grounded view keeps the id it always had. The vector is the view's DINOv2-S feature,
        model names where it came from (the recognition worker's detection or naming's own CPU crop)."""
        rows = []
        for index, seed in enumerate(item.get('seeds') or [{k: item.get(k) for k in ('media_id', 'offset_s', 'box', 'strong', 'strong_source', 'label', 'sample_id')}]):
            row = {'id': 'naming_' + (digest(item['event_id'], memory_id) if index == 0 else digest(item['event_id'], memory_id, index)),
                   'memory_id': memory_id, 'model': seed.get('strong_source') or STRONG_MODEL, 'naming_event_id': 'naming_' + digest(item['event_id']),
                   'vector': [round(float(x), 4) for x in seed['strong']], 'source': 'naming',
                   'matched_by': 'naming' if index == 0 else 'naming_view', 'media_id': seed['media_id'], 'offset_s': seed['offset_s'],
                   'box': seed['box'], 'label': seed.get('label'),
                   'strong_cosine': item.get('strong_cosine') if index == 0 else seed.get('strong_cosine'),
                   'sample_id': seed.get('sample_id'), 'created_at': now_iso(), 'test_capture': item.get('test_capture')}
            rows.append({k: v for k, v in row.items() if v is not None})
        return rows

    def views(self, item):
        return item.get('seeds') or [{'media_id': item['media_id'], 'strong': item['strong']}]

    def new_entity(self, item, now):
        frame_at = datetime.fromtimestamp(item['frame_us'] / 1e6, tz=timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')
        name, referent = item['name'], item.get('referent') or item.get('label') or 'object'
        text = f'用户命名的物体「{name}」（{referent}）。{frame_at} 说「{item["utterance"]}」时看到。'
        x1, y1, x2, y2 = item['box']
        views = self.views(item)
        mean_vec = np.sum([v['strong'] for v in views], axis=0); mean_vec = mean_vec / np.linalg.norm(mean_vec)
        props = {'id': 'object:named-' + digest(item['event_id'], size=12), 'layer': 'memory', 'kind': 'object', 'name': name,
                 'description': text, 'text': text, 'summary': name,
                 'object_vec': [float(x) for x in mean_vec], 'object_vec_n': len(views),
                 'media_ids': json.dumps(list(dict.fromkeys(v['media_id'] for v in views))), 'occurrences': 1,
                 'covers': json.dumps([{'media_id': item['media_id'], 'offset': item['offset_s'],
                                        'box': [round(x1, 1), round(y1, 1), round(x2 - x1, 1), round(y2 - y1, 1)],
                                        'w': item['frame_width'], 'h': item['frame_height']}]),
                 't_start': frame_at, 't_last': frame_at, 't_start_us': item['frame_us'], 't_end_us': item['frame_us'],
                 'created_at': now, 'updated_at': now, 'created_by': 'user-naming', 'nights': 0}
        if item.get('test_capture'):
            props['test_capture'] = item['test_capture']
        return props

    def apply(self, item, cursor):
        """One write transaction under the graph writer lock. Returns the NamingEvent row (also in dry-run)."""
        acting = item['intent'] in ACTIONS and item['confidence'] >= self.args.min_confidence and 'status' not in item
        if self.graph.dry_run:
            with self.graph.driver.session() as session:
                plan = session.execute_read(lambda tx: self._plan(tx, item, acting, sleeping=False))
            log(event='dry_run_would_write', timings=item['timings'], **{k: v for k, v in plan['event'].items() if k not in ('context_json',)},
                entity=({k: v for k, v in plan['entity'].items() if k not in ('object_vec', 'face_vec', 'face_views')} if plan.get('entity') else None),
                update=({k: v for k, v in plan['update'].items() if k not in ('object_vec', 'face_vec', 'face_views')} if plan.get('update') else None),
                views=len(plan.get('views') or ()))
            return plan['event']
        t0 = time.perf_counter()
        with self.graph.driver.session() as session:
            tx = session.begin_transaction()
            try:
                state = tx.run(LOCK_QUERY, now=now_iso()).single()['s']
                if state.get('maintenance_token'):
                    raise RetryLater('graph maintenance pending')
                expires = state.get('sleep_expires_at')
                sleeping = bool(state.get('sleep_token')) and bool(expires) and \
                    datetime.fromisoformat(str(expires).replace('Z', '+00:00')) > datetime.now(timezone.utc)
                plan = self._plan(tx, item, acting, sleeping)
                if plan.get('entity'):
                    tx.run('CREATE (e:Entity) SET e = $p', p=plan['entity']).consume()
                if plan.get('update'):
                    tx.run('MATCH (e:Entity {id:$id}) SET e += $p', id=plan['memory_id'], p=plan['update']).consume()
                for row in plan.get('views') or ():
                    tx.run('MERGE (v:ObjectView {id:$r.id}) ON CREATE SET v += $r', r=row).consume()
                event = plan['event']
                event.setdefault('timings', {})
                done = time.time()
                timings = {**item['timings'], 'write_ms': round((time.perf_counter() - t0) * 1000),
                           'naming_total_ms': round((done - item['analysed_at']) * 1000),
                           'utterance_end_to_write_ms': round((done - item['end_us'] / 1e6) * 1000)}
                event['timings'] = json.dumps(timings)
                tx.run('MERGE (n:NamingEvent {id:$r.id}) ON CREATE SET n += $r', r=event).consume()
                if cursor is not None:  # None: an explicit redo of old turns (--only-rowids) leaves the cursor alone
                    tx.run('MERGE (s:NamingState {id:$id}) SET s.event_cursor=$c, s.updated_at=$now, s += $n',
                           id=STATE_ID, c=cursor, now=now_iso(), n=self.counter_props(event['status'])).consume()
                tx.commit()
            except BaseException:
                if not tx.closed():
                    tx.rollback()
                raise
            finally:
                tx.close()
        self.bump(event['status'])
        log(event='named', status=event['status'], intent=item['intent'], name=item.get('name'), memory_id=event.get('memory_id'),
            matched_by=event.get('matched_by'), capture_id=item['capture_id'], grounding=item.get('grounding'), iou=item.get('iou'),
            strong_cosine=item.get('strong_cosine'), views=len(item.get('seeds') or ()), sheet_objects=item.get('sheet_objects'),
            timings=timings, test=bool(item.get('test_capture')))
        return event

    def _plan(self, tx, item, acting, sleeping):
        if not acting:
            status = item.get('status') or 'ignored'
            return {'event': self.event_row(item, status)}
        if item.get('kind') == 'person':
            return self._plan_person(tx, item, sleeping)
        target, how = decide(tx, item)
        now = now_iso()
        if target is None and how == 'create':
            entity = self.new_entity(item, now)
            entity.update(name_source='user', named_at=now, named_utterance=item['utterance'])
            event = self.event_row(item, 'named', entity['id'], matched_by='created' if item['intent'] == 'name' else 'created_' + item['intent'])
            entity['naming_event_id'] = event['id']
            return {'event': event, 'entity': entity, 'memory_id': entity['id'], 'views': self.view_rows(item, entity['id'])}
        if target is None:
            item['reason'] = 'no_known_object'
            return {'event': self.event_row(item, 'no_target', matched_by=how)}
        if sleeping:
            # The nightly sleep re-writes the memory objects it read; changing one now could be overwritten.
            raise RetryLater('sleep snapshot claim active')
        prev = {'name': target['name'], **{key: target.get('p_' + key) for key in TRACKED if key != 'name'}}
        prev = {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in prev.items()}
        event_id = 'naming_' + digest(item['event_id'])
        update = {'updated_at': now, 'naming_event_id': event_id}
        if item['intent'] in ('name', 'rename') or said_over_description(item, target):
            status = 'renamed' if item['intent'] == 'rename' else 'named'
            update.update(name=item['name'], name_source='user', named_at=now, named_utterance=item['utterance'])
        else:
            status = 'same_as'
            n, views = int(target.get('p_object_vec_n') or 0), self.views(item)
            added = np.sum([v['strong'] for v in views], axis=0)
            mean = unit(target['object_vec']) * n + added if target['object_vec'] is not None and n > 0 else added
            update['object_vec'] = [float(x) for x in mean / np.linalg.norm(mean)]
            update['object_vec_n'] = n + len(views)
            ids = media_ids(target.get('p_media_ids'))
            new_ids = [m for m in dict.fromkeys(v['media_id'] for v in views) if m not in ids]
            if new_ids:
                update['media_ids'] = json.dumps(ids + new_ids)
            update['occurrences'] = int(target.get('p_occurrences') or 0) + 1
            frame_at = datetime.fromtimestamp(item['frame_us'] / 1e6, tz=timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')
            if item['frame_us'] > float(target.get('p_t_end_us') or 0):
                update.update(t_last=frame_at, t_end_us=item['frame_us'])
        event = self.event_row(item, status, target['id'], prev=prev, matched_by=how)
        return {'event': event, 'update': update, 'memory_id': target['id'], 'views': self.view_rows(item, target['id'])}

    def _plan_person(self, tx, item, sleeping):
        target, how = decide_person(tx, item)
        now = now_iso()
        event_id = 'naming_' + digest(item['event_id'])
        frame_at = datetime.fromtimestamp(item['frame_us'] / 1e6, tz=timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')
        views = [np.asarray(v) for v in item['face_views']]
        mean = np.sum(views, axis=0); mean = mean / np.linalg.norm(mean)
        if target is None and how == 'create':
            x1, y1, x2, y2 = item['box']
            text = f'用户介绍的人「{item["name"]}」。{frame_at} 说「{item["utterance"]}」时看到。'
            entity = {'id': 'person:named-' + digest(item['event_id'], size=12), 'layer': 'memory', 'kind': 'person',
                      'name': item['name'], 'description': text, 'text': text, 'summary': item['name'],
                      'face_vec': [float(x) for x in mean], 'face_vec_n': len(views), 'face_views': json.dumps(item['face_views']),
                      'media_ids': json.dumps(list(dict.fromkeys(item['face_media']))), 'occurrences': 1,
                      'covers': json.dumps([{'media_id': item['media_id'], 'offset': item['offset_s'],
                                             'box': [round(x1, 1), round(y1, 1), round(x2 - x1, 1), round(y2 - y1, 1)],
                                             'w': item['frame_width'], 'h': item['frame_height']}]),
                      't_start': frame_at, 't_last': frame_at, 't_start_us': item['frame_us'], 't_end_us': item['frame_us'],
                      'created_at': now, 'updated_at': now, 'created_by': 'user-naming', 'nights': 0,
                      'name_source': 'user', 'named_at': now, 'named_utterance': item['utterance'], 'naming_event_id': event_id}
            if item.get('test_capture'):
                entity['test_capture'] = item['test_capture']
            event = self.event_row(item, 'named', entity['id'], matched_by='created' if item['intent'] == 'name' else 'created_' + item['intent'])
            return {'event': event, 'entity': entity, 'memory_id': entity['id']}
        if target is None:
            item['reason'] = 'no_known_person'
            return {'event': self.event_row(item, 'no_target', matched_by=how)}
        if sleeping:
            raise RetryLater('sleep snapshot claim active')
        prev = {key: (list(target[key]) if isinstance(target.get(key), (list, tuple)) else target.get(key)) for key in PERSON_TRACKED}
        # The newest views first (the user is looking at them now), then the ones kept before; near-duplicates dropped.
        kept = [v for v in item['face_views']]
        for old in json.loads(target['face_views']) if target.get('face_views') else []:
            if len(kept) >= MAX_SEEDS:
                break
            if all(float(np.dot(unit(old, dims=FACE_DIM), unit(v, dims=FACE_DIM))) < SEED_DEDUP for v in kept):
                kept.append(old)
        update = {'updated_at': now, 'naming_event_id': event_id, 'face_views': json.dumps(kept),
                  'occurrences': int(target.get('occurrences') or 0) + 1}
        if target.get('face_vec') is None:
            update.update(face_vec=[float(x) for x in mean], face_vec_n=len(views))
        ids = media_ids(target.get('media_ids'))
        new_ids = [m for m in dict.fromkeys(item['face_media']) if m not in ids]
        if new_ids:
            update['media_ids'] = json.dumps(ids + new_ids)
        if item['frame_us'] > float(target.get('t_end_us') or 0):
            update.update(t_last=frame_at, t_end_us=item['frame_us'])
        if item['intent'] in ('name', 'rename') or said_over_description(item, target):
            status = 'renamed' if item['intent'] == 'rename' else 'named'
            update.update(name=item['name'], name_source='user', named_at=now, named_utterance=item['utterance'])
        else:
            status = 'same_as'
        event = self.event_row(item, status, target['id'], prev=prev, matched_by=how)
        return {'event': event, 'update': update, 'memory_id': target['id']}

    def counter_props(self, status):
        counts = dict(self.counters); counts[status] = counts.get(status, 0) + 1
        return {'n_' + k: v for k, v in counts.items()}

    def bump(self, status):
        self.counters[status] = self.counters.get(status, 0) + 1

    # --- loop ----------------------------------------------------------------------------------------------------

    def start(self):
        self.graph.schema()
        state = self.graph.state() or {}
        self.counters = {k[2:]: int(v) for k, v in state.items() if k.startswith('n_')}
        db = ro(self.args.archive_db)
        try:
            top = self.archive_top(db)
        finally:
            db.close()
        first = state.get('event_cursor') is None
        if first:
            self.cursor = max(0, top - (self.args.replay_events if self.graph.dry_run else 0))
            if not self.graph.dry_run:
                self.graph.set_cursor(self.cursor, {})
        else:
            self.cursor = int(state['event_cursor'])
            if self.cursor > top:
                raise ValueError('archive cursor moved backwards; refuse implicit replay')
        log(event='start', dry_run=self.graph.dry_run, first_start=first, cursor=self.cursor, archive_top=top,
            vlm=self.args.vlm_model)
        t0 = time.perf_counter()
        try:  # cut the known objects' views now, so the first naming does not wait for it
            sheet = self.sheet(self.known())
            log(event='sheet_ready', objects=len(sheet[1]) if sheet else 0, seconds=round(time.perf_counter() - t0, 1))
        except Exception as error:
            log(event='sheet_failed', error=type(error).__name__, detail=str(error)[:200])

    def cycle(self):
        """Process new speech events in order; returns True when more are already waiting."""
        db = ro(self.args.archive_db)
        try:
            top = self.archive_top(db)
            if top < self.cursor:
                raise ValueError('archive cursor moved backwards; refuse implicit replay')
            columns = 'rowid AS r, event_id, capture_id, started_at, ended_at, start_us, end_us, body_json, refs_json, recorded_at'
            if self.args.only_rowids:  # dry-run diagnostics: exactly these speech events, cursor untouched
                rows = db.execute(f"SELECT {columns} FROM archive_events WHERE kind='speech' AND rowid IN ({','.join('?' * len(self.args.only_rowids))}) ORDER BY rowid",
                                  self.args.only_rowids).fetchall()
                top = through = self.cursor
            else:
                rows = db.execute(f"""SELECT {columns} FROM archive_events WHERE rowid>? AND rowid<=? AND kind='speech'
                    ORDER BY rowid LIMIT ?""", (self.cursor, top, self.args.max_events)).fetchall()
                through = rows[-1]['r'] if len(rows) == self.args.max_events else top
            if rows and not vlm.serving(self.args.vlm_url):
                # Every turn needs the VLM; while it is away, wait without spending the turns' attempts.
                if not self.vlm_down:
                    log(event='vlm_unavailable', pending=len(rows)); self.vlm_down = True
                return False
            if self.vlm_down:
                log(event='vlm_available'); self.vlm_down = False
            for row in rows:
                try:
                    item = self.cache.get(row['event_id']) or self.analyse(db, row)
                except Exception as error:  # RetryLater (VLM/lock/decode) or a bug: bounded retries, then skip
                    self.failures[row['event_id']] = self.failures.get(row['event_id'], 0) + 1
                    log(event='retry_later', event_id=row['event_id'], attempt=self.failures[row['event_id']],
                        error=type(error).__name__, detail=str(error)[:200])
                    if self.failures[row['event_id']] < self.args.max_attempts:
                        through = row['r'] - 1; break
                    item = None  # give up on this turn; it must not block later speech
                if item is None or item['intent'] == 'none':
                    self.bump('none' if item else 'failed')
                    continue
                self.cache[row['event_id']] = item
                try:
                    self.apply(item, None if self.args.only_rowids else row['r'])
                except RetryLater as error:
                    log(event='apply_waiting', event_id=row['event_id'], detail=str(error))
                    self.flush(row['r'] - 1); return False
                self.cache.pop(row['event_id'], None)
                if not self.args.only_rowids:
                    self.cursor = row['r']
        finally:
            db.close()
        self.flush(through)
        return through < top

    def flush(self, through):
        if through > self.cursor:
            self.graph.set_cursor(through, {'n_' + k: v for k, v in self.counters.items()})
            self.cursor = through

    def run(self):
        self.start()
        while True:
            started, busy = time.monotonic(), False
            try:
                busy = self.cycle()
            except Exception as error:
                log(event='retry_pending', error=type(error).__name__, detail=str(error)[:300])
            if self.args.once and not busy:
                log(event='once_done', cursor=self.cursor, counters=self.counters); return
            if not busy:
                time.sleep(max(0., self.args.interval - (time.monotonic() - started)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=Path, default=Path(os.environ.get('ENTITY_WEIGHTS_DIR', '/models/entities')))
    parser.add_argument('--archive-db', default=os.environ.get('ARCHIVE_DB', '/archive/archive.sqlite3'))
    parser.add_argument('--vlm-url', default=os.environ.get('VLM_URL', 'http://127.0.0.1:22410/v1'))
    parser.add_argument('--vlm-model', default=os.environ.get('VLM_MODEL', 'qwen3.6-35b-a3b'))
    parser.add_argument('--interval', type=float, default=2.)
    parser.add_argument('--min-confidence', type=float, default=.6)
    parser.add_argument('--context-turns', type=int, default=40, help='earlier turns of the recording the model sees')
    parser.add_argument('--context-seconds', type=float, default=600., help='... from at most this far back')
    parser.add_argument('--known-objects', type=int, default=300, help='known memory objects listed to the model')
    parser.add_argument('--wait-objects', type=float, default=20., help='seconds to wait for strong detections of the turn')
    parser.add_argument('--max-events', type=int, default=200, help='speech events read per cycle')
    parser.add_argument('--max-attempts', type=int, default=5, help='retries of one speech turn before it is skipped')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--dry-run', action='store_true', help='read production, write nothing to Neo4j')
    parser.add_argument('--replay-events', type=int, default=0, help='dry-run only: start this many event rowids back')
    parser.add_argument('--only-rowids', type=lambda v: [int(x) for x in v.split(',') if x], default=None,
                        help='analyse exactly these archive speech rowids (comma separated) and exit; the cursor stays. '
                             'Writes only with --redo (the user asked to redo those turns)')
    parser.add_argument('--redo', action='store_true', help='with --only-rowids: write the results')
    parser.add_argument('--once', action='store_true', help='exit once caught up')
    args = parser.parse_args()
    if (args.replay_events or (args.only_rowids and not args.redo)) and not args.dry_run:
        parser.error('--replay-events / --only-rowids are for --dry-run only (no historical replay), unless --only-rowids --redo')
    if args.redo and not args.only_rowids:
        parser.error('--redo needs --only-rowids')
    if args.only_rowids:
        args.once = True
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(os.environ['GRAPH_URL'], auth=tuple(os.environ['NEO4J_AUTH'].split('/', 1)))
    try:
        Service(Graph(driver, args.dry_run), args, MediaSource(archive_client())).run()
    finally:
        driver.close()


if __name__ == '__main__':
    main()
