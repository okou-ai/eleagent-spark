from __future__ import annotations
from datetime import datetime, timedelta, timezone
from contextlib import contextmanager
from . import retention
import hashlib
import json
import math
import sqlite3
import uuid
import numpy as np


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


ASSOCIATION_METHOD = {
    'clip_score': 'visual: sampled frames containing the entity / sampled frames; voice: speech seconds / clip seconds',
    'entity_score': 'weighted Jaccard over originals: sum(min(clip scores)) / sum(max(clip scores))',
    'basis': 'current assignments including manual corrections; recomputed on each request',
}


def normalized_box(data):
    width, height = data.get('frame_width'), data.get('frame_height')
    if not width or not height:
        return None
    if 'bbox' in data:
        x1, y1, x2, y2 = data['bbox']
    elif 'bbox_xywh' in data:
        x1, y1, w, h = data['bbox_xywh']; x2, y2 = x1 + w, y1 + h
    else:
        return None
    return (x1 / width, y1 / height, x2 / width, y2 / height)


def box_iou(a, b):
    w = max(0., min(a[2], b[2]) - max(a[0], b[0])); h = max(0., min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - w*h
    return w * h / union if union > 0 else 0.


def sample_cover(row):
    """Thumbnail reference for a face/object sample: x,y,w,h in the detector's decoded frame."""
    data = json.loads(row['data'])
    if 'bbox_xywh' in data:
        box = data['bbox_xywh']
    elif 'bbox' in data:
        x1, y1, x2, y2 = data['bbox']
        box = [x1, y1, x2 - x1, y2 - y1]
    else:
        return None
    width, height = int(data.get('frame_width') or 0), int(data.get('frame_height') or 0)
    if not width or not height:
        return None
    # Detector boxes can extend past the frame edge; clip to the visible region.
    left, top = max(0., float(box[0])), max(0., float(box[1]))
    right, bottom = min(float(width), float(box[0]) + float(box[2])), min(float(height), float(box[1]) + float(box[3]))
    if right - left < 2 or bottom - top < 2:
        return None
    return {'sample_id': row['id'], 'media_id': row['media_id'], 'offset': row['offset'],
            'box': [round(left, 2), round(top, 2), round(right - left, 2), round(bottom - top, 2)],
            'frame_width': width, 'frame_height': height}


class Store:
    def __init__(self, path, recover=False):
        self.db = sqlite3.connect(path, timeout=20, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS config (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, media_id TEXT NOT NULL,
          model TEXT NOT NULL, asset TEXT NOT NULL, state TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
          error TEXT, retry_at REAL NOT NULL DEFAULT 0);
        CREATE INDEX IF NOT EXISTS job_queue ON jobs(model,state,retry_at);
        CREATE TABLE IF NOT EXISTS entities (id TEXT PRIMARY KEY, model TEXT NOT NULL,
          name TEXT, prototypes TEXT NOT NULL, first_seen TEXT NOT NULL, last_seen TEXT NOT NULL,
          modality TEXT NOT NULL, scope_capture TEXT);
        CREATE TABLE IF NOT EXISTS samples (id TEXT PRIMARY KEY, entity_id TEXT REFERENCES entities(id),
          model TEXT NOT NULL, media_id TEXT NOT NULL, capture_id TEXT NOT NULL, offset REAL NOT NULL,
          started_at TEXT NOT NULL, data TEXT NOT NULL, assignment TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS sample_entity ON samples(entity_id);
        -- Short-term tracking looks up one capture's recent samples per detection; without this it scans all history.
        CREATE INDEX IF NOT EXISTS sample_track ON samples(model, capture_id, started_at);
        CREATE TABLE IF NOT EXISTS outbox (id TEXT PRIMARY KEY, payload TEXT NOT NULL, error TEXT);
        CREATE TABLE IF NOT EXISTS events (id TEXT PRIMARY KEY, payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS rejected_events (id TEXT PRIMARY KEY, payload TEXT NOT NULL, error TEXT NOT NULL, rejected_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS speech_pending (media_id TEXT PRIMARY KEY, capture_id TEXT NOT NULL,
          clip_seq INTEGER NOT NULL, asset TEXT NOT NULL, seen_at REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS speech_context (media_id TEXT PRIMARY KEY, capture_id TEXT NOT NULL,
          clip_seq INTEGER NOT NULL, asset TEXT NOT NULL, seen_at REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS speech_progress (capture_id TEXT PRIMARY KEY, published_until TEXT, holding INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS media_frames (media_id TEXT PRIMARY KEY, sampled_frames INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS corrections (id TEXT PRIMARY KEY, action TEXT NOT NULL,
          body TEXT NOT NULL, created_at TEXT NOT NULL);
        ''')
        if recover:
            self.db.execute("UPDATE jobs SET state='queued' WHERE state='processing'")
        self.db.commit()

    @contextmanager
    def write_transaction(self):
        """Serialize read/decide/write with other API or worker connections.

        SQLite's default deferred transaction starts at the first write, which
        leaves earlier sample/prototype reads outside its protection.
        """
        self.db.execute('BEGIN IMMEDIATE')
        try:
            yield
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise

    def config(self, key, default=None):
        row = self.db.execute('SELECT value FROM config WHERE key=?', (key,)).fetchone()
        return row[0] if row else default

    def set_config(self, key, value):
        with self.db:
            self.db.execute('INSERT OR REPLACE INTO config VALUES (?,?)', (key, value))

    def inherit_feed_cursor(self, key, family):
        """Resume an earlier worker generation, or read from zero on first installation."""
        if family not in {'speech_cursor', 'text_cursor'} or not key.startswith(family + ':'):
            raise ValueError('unsupported_feed_cursor_family')
        with self.write_transaction():
            row = self.db.execute('SELECT value FROM config WHERE key=?', (key,)).fetchone()
            if row is not None:
                return int(row['value'])
            previous = self.db.execute('SELECT value FROM config WHERE key=? OR key GLOB ?',
                                       (family, family + ':*')).fetchall()
            cursor = max((int(row['value']) for row in previous), default=0)
            self.db.execute('INSERT INTO config(key,value) VALUES(?,?)', (key, str(cursor)))
            return cursor

    def inherit_entity_pipeline(self, model):
        """Continue discovery and unfinished entity work after a model upgrade.

        Completed jobs are deleted, so the feed watermark owns progress. This
        transaction never reads or alters the speech/text cursors or outbox.
        """
        prefix = 'ingest_cursor:entities-v1-'
        if not model.startswith('entities-v1-'):
            raise ValueError('entity_pipeline_required')
        key = 'ingest_cursor:' + model
        with self.write_transaction():
            if self.db.execute('SELECT 1 FROM config WHERE key=?', (key,)).fetchone():
                return
            previous = self.db.execute("SELECT value FROM config WHERE key LIKE ?", (prefix + '%',)).fetchall()
            cursor = max((int(row['value']) for row in previous), default=0)
            pending = self.db.execute("""SELECT * FROM jobs WHERE model LIKE 'entities-v1-%'
                AND model!=? AND state IN ('queued','processing') ORDER BY rowid""", (model,)).fetchall()
            for job in pending:
                job_id = hashlib.sha256((job['media_id'] + ':' + model).encode()).hexdigest()
                self.db.execute("""INSERT OR IGNORE INTO jobs
                    (id,media_id,model,asset,state,attempts,error,retry_at)
                    VALUES(?,?,?,?,'queued',0,NULL,0)""", (job_id, job['media_id'], model, job['asset']))
                self.db.execute('DELETE FROM jobs WHERE id=?', (job['id'],))
            self.db.execute('INSERT INTO config(key,value) VALUES(?,?)', (key, str(cursor)))

    def enqueue(self, asset, model, max_pending=256):
        job_id = hashlib.sha256((asset['media_id'] + ':' + model).encode()).hexdigest()
        if self.db.execute('SELECT 1 FROM jobs WHERE id=?', (job_id,)).fetchone():
            return True
        count = self.db.execute("SELECT count(*) FROM jobs WHERE state IN ('queued','processing') AND model=?", (model,)).fetchone()[0]
        if count >= max_pending:
            return False
        with self.db:
            self.db.execute('INSERT INTO jobs(id,media_id,model,asset,state) VALUES(?,?,?,?,?)',
                (job_id, asset['media_id'], model, canonical(asset), 'queued'))
        return True

    def processed_elsewhere(self, media_id, model):
        return self.db.execute("SELECT 1 FROM jobs WHERE media_id=? AND model!=? AND state='done' LIMIT 1", (media_id, model)).fetchone() is not None

    def drop_superseded(self, model):
        """Pipeline changes apply to new media only: forget queued work for media an earlier pipeline finished."""
        with self.db:
            return self.db.execute('''DELETE FROM jobs WHERE model=? AND state='queued' AND media_id IN
                (SELECT media_id FROM jobs WHERE model!=? AND state='done')''', (model, model)).rowcount

    def take_job(self, now, model):
        row = self.db.execute("SELECT * FROM jobs WHERE state='queued' AND retry_at<=? AND model=? ORDER BY rowid LIMIT 1", (now,model)).fetchone()
        if not row:
            return None
        with self.db:
            self.db.execute("UPDATE jobs SET state='processing',attempts=attempts+1 WHERE id=?", (row['id'],))
        return dict(row)

    def fail_job(self, job_id, error, now, paused=False):
        with self.db:
            self.db.execute("UPDATE jobs SET state='queued',error=?,retry_at=? WHERE id=?", (error, now + (5 if paused else 60), job_id))

    def _track(self, model, asset, started_at, data, vector, excluded, window, min_iou=.3, min_similarity=.3):
        """Short-term spatial continuity inside one capture: the same class at an overlapping position
        within `window` seconds keeps its entity even when appearance drifts (blur, reflections, partial crops)."""
        box = normalized_box(data)
        if box is None:
            return None
        at = datetime.fromisoformat(started_at)
        low = (at - timedelta(seconds=window)).isoformat(timespec='microseconds')
        high = (at + timedelta(seconds=window)).isoformat(timespec='microseconds')
        best = None
        for row in self.db.execute('''SELECT s.entity_id,s.data,s.started_at FROM samples s JOIN entities e ON e.id=s.entity_id
              WHERE s.model=? AND s.capture_id=? AND s.started_at BETWEEN ? AND ? AND e.scope_capture IS NULL''',
              (model, asset['capture_id'], low, high)):
            if row['entity_id'] in excluded:
                continue
            other = json.loads(row['data'])
            other_box = normalized_box(other)
            if other_box is None:
                continue
            overlap = box_iou(box, other_box)
            if overlap < min_iou:
                continue
            if vector is not None and other.get('embedding'):
                if float(np.dot(vector, np.asarray(other['embedding'], dtype=np.float64))) < min_similarity:
                    continue
            gap = abs((datetime.fromisoformat(row['started_at']) - at).total_seconds())
            key = (overlap, -gap)
            if best is None or key > best[0]:
                best = (key, row['entity_id'], overlap)
        return best and {'entity_id': best[1], 'iou': round(best[2], 4)}

    def assign(self, sample_id, model, asset, offset, started_at, data, excluded=(), threshold=.55, margin=.08, modality="face", cluster_eligible=True, track_window=None):
        with self.write_transaction():
            existing = self.db.execute("SELECT entity_id,assignment,json_extract(data,'$._retention.retired_entity_id') AS retired_entity_id FROM samples WHERE id=?", (sample_id,)).fetchone()
            if existing:
                return {'entity_id': existing['entity_id'] or existing['retired_entity_id'], 'assignment': existing['assignment'], 'replayed': True}
            vector = data.get('embedding')
            if vector is not None:
                values = np.asarray(vector, dtype=np.float64)
                norm = float(np.linalg.norm(values))
                if values.ndim != 1 or not 0 < values.size <= 4096 or not np.isfinite(values).all() or not np.isfinite(norm) or norm <= 0:
                    raise ValueError('invalid_embedding')
                vector = (values / norm).tolist()
                data = {**data, 'embedding': vector}
            entity_id = None
            assignment = 'unknown_quality'
            similarity = None
            tracked = self._track(model, asset, started_at, data, vector, excluded, track_window) if track_window else None
            if tracked:
                entity_id, assignment = tracked['entity_id'], 'provisional_track'
                data = {**data, 'track': {'iou': tracked['iou'], 'window_seconds': track_window}}
                prototypes = json.loads(self.db.execute('SELECT prototypes FROM entities WHERE id=?', (entity_id,)).fetchone()[0])
                if vector is not None and prototypes:
                    similarity = max(float(np.dot(vector, prototype)) for prototype in prototypes)
            elif vector is not None and cluster_eligible:
                candidates = []
                for row in self.db.execute('SELECT id,prototypes,scope_capture FROM entities WHERE model=? AND modality=?', (model,modality)):
                    if row['scope_capture'] is not None and row['scope_capture'] != asset['capture_id']:
                        continue
                    if row['id'] in excluded:
                        continue
                    prototypes = json.loads(row['prototypes'])
                    if prototypes:
                        score = max(float(np.dot(vector, prototype)) for prototype in prototypes)
                        candidates.append((score, row['id']))
                candidates.sort(reverse=True)
                similarity = candidates[0][0] if candidates else None
                if candidates and candidates[0][0] >= threshold:
                    if len(candidates) < 2 or candidates[0][0] - candidates[1][0] >= margin:
                        entity_id, assignment = candidates[0][1], 'provisional_match'
                    else:
                        assignment = 'unknown_ambiguous_match'
                else:
                    entity_id, assignment = modality + '_' + uuid.uuid4().hex, 'provisional_new'
                    self.db.execute('INSERT INTO entities VALUES (?,?,?,?,?,?,?,?)',
                        (entity_id, model, None, '[]', started_at, started_at, modality, None))
            self.db.execute('INSERT INTO samples VALUES (?,?,?,?,?,?,?,?,?)',
                (sample_id, entity_id, model, asset['media_id'], asset['capture_id'], offset, started_at,
                 canonical({key: value for key, value in data.items() if key != retention.META_KEY}), assignment))
            if entity_id:
                self._rebuild(entity_id)
            return {'entity_id': entity_id, 'assignment': assignment, 'similarity': similarity}

    def match_entity(self, model, modality, vector, started_at, threshold=.55, margin=.05, excluded=()):
        """Match one diarized speaker centroid to a persistent entity of the same model, or create one.
        Prototypes come from the samples later recorded for the entity; a new entity starts empty."""
        values = np.asarray(vector, dtype=np.float64)
        norm = float(np.linalg.norm(values))
        if values.ndim != 1 or not 0 < values.size <= 4096 or not np.isfinite(values).all() or norm <= 0:
            raise ValueError('invalid_embedding')
        values = values / norm
        with self.write_transaction():
            candidates = []
            for row in self.db.execute('SELECT id,prototypes FROM entities WHERE model=? AND modality=? AND scope_capture IS NULL', (model, modality)):
                prototypes = json.loads(row['prototypes'])
                if row['id'] not in excluded and prototypes:
                    candidates.append((max(float(values @ np.asarray(p)) for p in prototypes), row['id']))
            candidates.sort(reverse=True)
            if candidates and candidates[0][0] >= threshold:
                ambiguous = len(candidates) > 1 and candidates[0][0] - candidates[1][0] < margin
                return {'entity_id': candidates[0][1], 'assignment': 'provisional_ambiguous_match' if ambiguous else 'provisional_match',
                        'similarity': candidates[0][0]}
            entity_id = modality + '_' + uuid.uuid4().hex
            self.db.execute('INSERT INTO entities VALUES (?,?,?,?,?,?,?,?)', (entity_id, model, None, '[]', started_at, started_at, modality, None))
            return {'entity_id': entity_id, 'assignment': 'provisional_new', 'similarity': candidates[0][0] if candidates else None}

    def prune_empty_entities(self, model, keep=()):
        """Drop entities of `model` that never received a sample (e.g. created by a batch that then failed)."""
        with self.write_transaction():
            ids = [row[0] for row in self.db.execute("""SELECT e.id FROM entities e WHERE e.model=? AND e.name IS NULL
                AND NOT EXISTS (SELECT 1 FROM samples s WHERE s.entity_id=e.id)
                AND NOT EXISTS (SELECT 1 FROM corrections c WHERE c.body LIKE '%' || e.id || '%')""", (model,)) if row[0] not in keep]
            self.db.executemany('DELETE FROM entities WHERE id=?', [(i,) for i in ids])
            return ids

    def recent_entities(self, model, modality, since, limit=8):
        """Recently active persistent entities with prototypes, newest first (speaker candidates for short turns)."""
        rows = self.db.execute("""SELECT id,prototypes FROM entities WHERE model=? AND modality=? AND last_seen>=?
            AND prototypes!='[]' ORDER BY last_seen DESC LIMIT ?""", (model, modality, since, limit)).fetchall()
        return [(row['id'], json.loads(row['prototypes'])) for row in rows]

    def record_speech(self, samples, events):
        """Atomically store diarized/transcribed turn samples and their timeline events (outbox)."""
        with self.write_transaction():
            touched = set()
            for item in samples:
                if self.db.execute('SELECT 1 FROM samples WHERE id=?', (item['sample_id'],)).fetchone():
                    continue
                data = dict(item['data'])
                if data.get('embedding') is not None:
                    values = np.asarray(data['embedding'], dtype=np.float64)
                    data['embedding'] = (values / np.linalg.norm(values)).tolist()
                self.db.execute('INSERT INTO samples VALUES (?,?,?,?,?,?,?,?,?)', (item['sample_id'], item['entity_id'], item['model'],
                    item['media_id'], item['capture_id'], item['offset'], item['started_at'], canonical(data), item['assignment']))
                if item['entity_id']:
                    touched.add(item['entity_id'])
            for entity_id in touched:
                self._rebuild(entity_id)
            for event in events:
                self.db.execute('INSERT OR IGNORE INTO events VALUES (?,?)', (event['event_id'], canonical(event)))
                stored = self.db.execute('SELECT payload FROM events WHERE id=?', (event['event_id'],)).fetchone()[0]
                self.db.execute('INSERT OR IGNORE INTO outbox VALUES (?,?,NULL)', (event['event_id'], stored))

    def _rebuild(self, entity_id):
        # Filter before LIMIT: recent manually assigned unknown samples must not
        # evict older usable representatives. Bounds still include every sample.
        rows = self.db.execute("""SELECT data FROM samples WHERE entity_id=?
            AND json_type(data,'$.embedding')='array'
            AND json_array_length(data,'$.embedding')>0
            ORDER BY started_at DESC,id DESC LIMIT 8""", (entity_id,)).fetchall()
        prototypes = []
        for row in rows:
            vector = json.loads(row['data'])['embedding']
            values = np.asarray(vector, dtype=np.float64)
            norm = float(np.linalg.norm(values))
            if values.ndim == 1 and np.isfinite(values).all() and np.isfinite(norm) and norm > 0:
                prototypes.append((values / norm).tolist())
        bounds = self.db.execute('SELECT min(started_at),max(started_at) FROM samples WHERE entity_id=?', (entity_id,)).fetchone()
        if bounds[0] is not None:
            self.db.execute('UPDATE entities SET prototypes=?,first_seen=?,last_seen=? WHERE id=?',
                (canonical(prototypes), bounds[0], bounds[1], entity_id))
        else:
            self.db.execute("UPDATE entities SET prototypes='[]' WHERE id=?", (entity_id,))

    def complete_job(self, job_id, events):
        with self.db:
            for event in events:
                self.db.execute('INSERT OR IGNORE INTO events VALUES (?,?)', (event['event_id'], canonical(event)))
                stored = self.db.execute('SELECT payload FROM events WHERE id=?', (event['event_id'],)).fetchone()[0]
                self.db.execute('INSERT OR IGNORE INTO outbox VALUES (?,?,NULL)', (event['event_id'], stored))
            # Finished work is forgotten: the ingest cursor never offers this media again.
            self.db.execute("DELETE FROM jobs WHERE id=?", (job_id,))

    def record_frames(self, media_id, sampled_frames):
        with self.db:
            self.db.execute('INSERT OR REPLACE INTO media_frames VALUES (?,?)', (media_id, int(sampled_frames)))

    def _presence(self):
        """Per-entity, per-original presence in [0,1] from current (corrected) assignments.
        Visual: share of sampled frames containing the entity. Voice: speech seconds / clip duration."""
        durations, frames = {}, dict(self.db.execute('SELECT media_id,sampled_frames FROM media_frames').fetchall())
        for media_id, asset in self.db.execute('SELECT media_id,asset FROM jobs'):
            durations.setdefault(media_id, float(json.loads(asset).get('duration_seconds') or 0))
        visual, speech, clips = {}, {}, {}
        for row in self.db.execute('''SELECT s.entity_id,s.media_id,s.capture_id,s.offset,s.started_at,s.data,e.modality
              FROM samples s JOIN entities e ON e.id=s.entity_id'''):
            clip = clips.setdefault(row['media_id'], {'media_id': row['media_id'], 'capture_id': row['capture_id'], 'started_at': row['started_at']})
            clip['started_at'] = min(clip['started_at'], row['started_at'])
            key = (row['entity_id'], row['media_id'])
            if row['modality'] == 'voice':
                data = json.loads(row['data'])
                speech[key] = speech.get(key, 0.) + max(0., float(data.get('end_seconds', 0)) - float(data.get('start_seconds', 0)))
            else:
                visual.setdefault(key, set()).add(round(row['offset'], 3))
        presence = {}
        for (entity_id, media_id), offsets in visual.items():
            duration = durations.get(media_id, 0.)
            # Historical clips predate media_frames; the deployment always sampled <=2 frames at 1 fps.
            total = frames.get(media_id) or (min(2, max(1, math.ceil(duration))) if duration else 2)
            presence.setdefault(entity_id, {})[media_id] = min(1., len(offsets) / max(1, total, len(offsets)))
        for (entity_id, media_id), seconds in speech.items():
            duration = durations.get(media_id, 0.)
            presence.setdefault(entity_id, {})[media_id] = min(1., seconds / duration) if duration > 0 else 1.
        return presence, clips

    def _entity_meta(self, ids):
        rows = self.db.execute('SELECT id,modality,name FROM entities WHERE id IN (%s)' % ','.join('?' * len(ids)), list(ids)).fetchall() if ids else []
        return {row['id']: {'modality': row['modality'], 'name': row['name']} for row in rows}

    def associations(self, entity_id, limit=20):
        if not self.db.execute('SELECT 1 FROM entities WHERE id=?', (entity_id,)).fetchone():
            raise KeyError('entity_not_found')
        presence, clips = self._presence()
        mine = presence.get(entity_id, {})
        scored = []
        for other, theirs in presence.items():
            if other == entity_id or not mine.keys() & theirs.keys():
                continue
            union = mine.keys() | theirs.keys()
            score = sum(min(mine.get(m, 0.), theirs.get(m, 0.)) for m in union) / sum(max(mine.get(m, 0.), theirs.get(m, 0.)) for m in union)
            scored.append((score, other, len(mine.keys() & theirs.keys())))
        scored.sort(key=lambda item: (-item[0], item[1]))
        scored = scored[:min(max(limit, 1), 100)]
        meta = self._entity_meta([item[1] for item in scored])
        ordered = sorted(mine.items(), key=lambda item: (-item[1], clips[item[0]]['started_at']))[:min(max(limit, 1), 100)]
        return {'entity_id': entity_id, 'method': ASSOCIATION_METHOD,
                'clips': [{**clips[m], 'score': round(v, 4), 'original_url': '/v1/archive/media/' + m + '/original'} for m, v in ordered],
                'entities': [{'entity_id': other, **meta.get(other, {}), 'score': round(score, 4), 'shared_clips': shared} for score, other, shared in scored]}

    def media_associations(self, media_id):
        presence, clips = self._presence()
        found = sorted(((values[media_id], entity_id) for entity_id, values in presence.items() if media_id in values), key=lambda item: (-item[0], item[1]))
        meta = self._entity_meta([entity_id for _, entity_id in found])
        return {'media_id': media_id, 'method': ASSOCIATION_METHOD, **({k: v for k, v in clips[media_id].items() if k != 'media_id'} if media_id in clips else {}),
                'entities': [{'entity_id': entity_id, **meta.get(entity_id, {}), 'score': round(score, 4)} for score, entity_id in found]}

    def pending_events(self, limit=32):
        return [dict(row) for row in self.db.execute('SELECT id,payload,error FROM outbox ORDER BY rowid LIMIT ?', (limit,))]

    def outbox_events(self, ids):
        rows = (self.db.execute('SELECT id,payload,error FROM outbox WHERE id=?', (i,)).fetchone() for i in ids)
        return [dict(row) for row in rows if row]

    def event_ack(self, event_id):
        # A persisted archive receipt confirms its exact observations. This is
        # separate from an empty delivery queue or merely having the source media.
        with self.write_transaction():
            row = self.db.execute('SELECT payload FROM outbox WHERE id=?', (event_id,)).fetchone()
            if row:
                event = json.loads(row['payload'])
                if event.get('kind') == 'object':
                    ids = [item['sample_id'] for item in event.get('body', {}).get('observations', ())
                           if isinstance(item.get('sample_id'), str)]
                    self.db.executemany("""UPDATE samples
                        SET data=json_set(data,'$._retention.archive_confirmed',1)
                        WHERE id=? AND json_extract(data,'$.modality')='object'""", [(sample_id,) for sample_id in ids])
            self.db.execute('DELETE FROM outbox WHERE id=?', (event_id,))
            self.db.execute('DELETE FROM events WHERE id=?', (event_id,))

    def event_reject(self, event_id, error):
        with self.write_transaction():
            row = self.db.execute('SELECT payload FROM outbox WHERE id=?', (event_id,)).fetchone()
            if row:
                self.db.execute('INSERT OR REPLACE INTO rejected_events VALUES (?,?,?,?)',
                    (event_id, row['payload'], error, datetime.now(timezone.utc).isoformat()))
                self.db.execute('DELETE FROM outbox WHERE id=?', (event_id,))

    def list_entities(self, limit=100):
        rows = self.db.execute('''SELECT e.id,e.model,e.name,e.first_seen,e.last_seen,e.modality,e.scope_capture,
          count(s.id) AS observations,count(DISTINCT s.media_id) AS media_count
          FROM entities e LEFT JOIN samples s ON s.entity_id=e.id GROUP BY e.id
          ORDER BY e.last_seen DESC LIMIT ?''', (min(max(limit, 1), 1000),)).fetchall()
        result = []
        for row in rows:
            cover = None
            if row['modality'] in ('face', 'object'):
                sample = self.db.execute('SELECT id,media_id,offset,data FROM samples WHERE entity_id=? ORDER BY started_at DESC LIMIT 1',
                                         (row['id'],)).fetchone()
                cover = sample and sample_cover(sample)
            result.append({**dict(row), 'status': 'provisional', 'cover': cover})
        return result

    def rename(self, entity_id, name):
        with self.write_transaction():
            if name is not None and (not isinstance(name, str) or not 1 <= len(name.strip()) <= 120):
                raise ValueError('invalid_display_name')
            row = self.db.execute('SELECT name FROM entities WHERE id=?', (entity_id,)).fetchone()
            if not row:
                raise KeyError('entity_not_found')
            correction = {'entity_id': entity_id, 'before': row['name'], 'after': name.strip() if name else None}
            self.db.execute('UPDATE entities SET name=? WHERE id=?', (correction['after'], entity_id))
            correction_id = self._correction('rename', correction)
            correction['correction_id'] = correction_id
            return correction

    def detail(self, entity_id, limit=100):
        entity = self.db.execute('SELECT id,model,name,modality,scope_capture,first_seen,last_seen FROM entities WHERE id=?', (entity_id,)).fetchone()
        if not entity:
            raise KeyError('entity_not_found')
        samples = self.db.execute('SELECT id,media_id,capture_id,offset,started_at,assignment,data FROM samples WHERE entity_id=? ORDER BY started_at DESC LIMIT ?', (entity_id,min(max(limit,1),100))).fetchall()
        return {**dict(entity), 'status': 'provisional', 'samples': [{**{key: row[key] for key in ('id','media_id','capture_id','offset','started_at','assignment')}, 'observation': {key:value for key,value in json.loads(row['data']).items() if key not in ('embedding', retention.META_KEY)}, 'original_url':'/v1/archive/media/' + row['media_id'] + '/original', 'cover': sample_cover(row)} for row in samples]}

    def reassign(self, sample_id, target=None):
        with self.write_transaction():
            sample = self.db.execute('SELECT * FROM samples WHERE id=?', (sample_id,)).fetchone()
            if not sample:
                raise KeyError('sample_not_found')
            if target:
                entity = self.db.execute('SELECT model FROM entities WHERE id=?', (target,)).fetchone()
                if not entity or entity['model'] != sample['model']:
                    raise ValueError('target_model_mismatch')
            if not target:
                modality = json.loads(sample['data']).get('modality', 'face')
                target = modality + '_' + uuid.uuid4().hex
                self.db.execute('INSERT INTO entities VALUES (?,?,?,?,?,?,?,?)',
                    (target, sample['model'], None, '[]', sample['started_at'], sample['started_at'], modality, None))
            before = sample['entity_id'] or json.loads(sample['data']).get(retention.META_KEY, {}).get('retired_entity_id')
            self.db.execute("UPDATE samples SET entity_id=?,assignment='user_corrected',data=json_remove(data,'$._retention.retired_entity_id') WHERE id=?", (target, sample_id))
            if sample['entity_id']:
                self._rebuild(sample['entity_id'])
            self._rebuild(target)
            correction = {'sample_id': sample_id, 'before': before, 'after': target,
                'media_id': sample['media_id'], 'capture_id': sample['capture_id'], 'offset': sample['offset']}
            correction_id = self._correction('reassign', correction)
            correction['correction_id'] = correction_id
            return correction

    def _correction(self, action, body):
        # Correction history is durable and separate from immutable original model observations.
        correction_id = 'correction_' + uuid.uuid4().hex
        created = datetime.now(timezone.utc).isoformat()
        self.db.execute('INSERT INTO corrections VALUES (?,?,?,?)', (correction_id, action, canonical(body), created))
        event = {'event_id': correction_id, 'kind': 'entity_correction', 'started_at': created,
            'ended_at': created, 'entity_ids': [value for value in (body.get('entity_id'), body.get('before') if action == 'reassign' else None, body.get('after') if action == 'reassign' else None) if value],
            'body': {'action': action, **body}, 'media_refs': []}
        if action == 'reassign':
            event['media_refs'] = [{'media_id': body['media_id'], 'start_offset_seconds': body['offset'],
                'end_offset_seconds': body['offset'], 'alignment': 'user_correction_of_sampled_frame'}]
        self.db.execute('INSERT INTO outbox VALUES (?,?,NULL)', (correction_id, canonical(event)))
        return correction_id

    def sample_detail(self, sample_id):
        row = self.db.execute('SELECT * FROM samples WHERE id=?', (sample_id,)).fetchone()
        if not row:
            raise KeyError('sample_not_found')
        data = json.loads(row['data'])
        return {**{key:row[key] for key in ('id','entity_id','model','media_id','capture_id','offset','started_at','assignment')}, 'observation':{key:value for key,value in data.items() if key not in ('embedding', retention.META_KEY)}, 'embedding_dimensions':len(data.get('embedding') or []), 'original_url':'/v1/archive/media/' + row['media_id'] + '/original'}

    def prepare_retention(self, **kwargs):
        return retention.prepare(self.db, **kwargs)

    def pending_retention(self):
        return retention.pending(self.db)

    def ack_retention(self, batch_id):
        return retention.ack(self.db, batch_id)

    def status(self):
        rejected = self.db.execute('SELECT id,error,rejected_at FROM rejected_events ORDER BY rowid DESC LIMIT 1').fetchone()
        return {'worker':json.loads(self.config('worker_status','{}')), 'model_versions':json.loads(self.config('model_versions','{}')), 'jobs': {row[0]: row[1] for row in self.db.execute('SELECT state,count(*) FROM jobs GROUP BY state')},
            'pending_events': self.db.execute('SELECT count(*) FROM outbox').fetchone()[0],
            'rejected_timeline_events': self.db.execute('SELECT count(*) FROM rejected_events').fetchone()[0],
            'last_timeline_rejection': dict(rejected) if rejected else None,
            'entities': self.db.execute('SELECT count(*) FROM entities').fetchone()[0],
            'models': {'face': 'cpu_baseline_provisional', 'voice': 'optional_provisional_no_overlap_detector', 'object': 'optional_appearance_candidates'}}
