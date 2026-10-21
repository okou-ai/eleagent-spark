"""Speech pipeline: CAM++ voiceprint diarization, then Qwen3-ASR per speaker turn.

Each archived clip is analysed as soon as it arrives, stitched (real gaps capped at 0.5 s) after
the clip(s) still held for its capture, split into speaker turns by voiceprint, matched to
persistent `voice_*` entities, and only the turns that have finished are transcribed and written
as `speech` timeline events plus entity samples; a turn still running at the end of the received
audio waits for the next clip. First installation processes the archive backlog; later worker
generations retain saved progress without replaying completed history.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, subprocess, tempfile, time
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
from .night_pause import NightPause
from .outbox import deliver

SR = 16000
PIPELINE = 'speech-v1'
DEFAULT_GLOSSARY = []  # Supply domain terms with --glossary-file when needed.
TEST_PREFIXES = ('archive-probe-', 'entity-acceptance-')
VAD = dict(threshold=.35, min_speech_duration_ms=250, min_silence_duration_ms=300, speech_pad_ms=100)
ANCHOR_SECONDS, ANCHOR_SIM = 2.5, .45      # long regions that define who is speaking in a batch
MATCH_SIM, MATCH_MARGIN = .55, .05          # batch speaker -> persistent voice entity
ASSIGN_SIM = .35                            # 2 s window -> nearest speaker, else unknown
CONTEXT_SECONDS = 2.5                       # shorter turns get no glossary (it can be echoed back)
EMBED_TURN_SECONDS = 2.0                    # only longer turns contribute prototypes
TURN_GAP_SECONDS = .8                       # a shorter pause continues the same speaker's turn
MIN_TURN_SECONDS = .25                      # shorter remnants of an already published turn are dropped
MAX_HOLD_CLIPS = 6                          # speech still running after this many clips is published anyway
POLL_SECONDS = .5


def sha16(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def fingerprint(*parts):
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


def windows_for(regions):
    """Embedding windows: whole region up to 3 s, else 2 s windows with a 1 s hop."""
    out = []
    for index, (start, end) in enumerate(regions):
        if end - start <= 3 * SR:
            out.append((index, start, end))
        else:
            out += [(index, a, min(a + 2 * SR, end)) for a in range(start, end - SR, SR)]
    return out


def build_turns(regions, windows, labels):
    """Half-second majority vote inside each region, isolated-flip smoothing, then merge
    same-speaker pieces across pauses < 0.8 s. Unknown (None) stays its own speaker."""
    pieces = []
    for index, (start, end) in enumerate(regions):
        own = [(a, b, label) for (r, a, b), label in zip(windows, labels) if r == index]
        for a in range(start, end, SR // 2):
            b = min(a + SR // 2, end)
            votes = [label for x, y, label in own if x < b and y > a] or [own[0][2]]
            pieces.append([index, a, b, max(votes, key=votes.count)])
    for i in range(1, len(pieces) - 1):
        p, c, n = pieces[i - 1], pieces[i], pieces[i + 1]
        if p[0] == c[0] == n[0] and p[3] == n[3] != c[3]:
            c[3] = p[3]
    turns = []
    for _, a, b, label in pieces:
        if turns and turns[-1]['speaker'] == label and a - turns[-1]['end'] < TURN_GAP_SECONDS * SR:
            turns[-1]['end'] = b
        else:
            turns.append({'speaker': label, 'start': a, 'end': b})
    return turns


def locate(spans, sample):
    """Stitched-sample -> (absolute time, media_id, media-relative seconds)."""
    for span in spans:
        if span['begin'] <= sample < span['end']:
            rel = (sample - span['begin']) / SR
            return span['t0'] + timedelta(seconds=rel), span['media_id'], span['origin'] + rel
    span = min(spans, key=lambda s: min(abs(sample - s['begin']), abs(sample - s['end'])))
    rel = (min(max(sample, span['begin']), span['end'] - 1) - span['begin']) / SR
    return span['t0'] + timedelta(seconds=rel), span['media_id'], span['origin'] + rel


def media_refs(spans, start, end):
    refs = []
    for span in spans:
        a, b = max(start, span['begin']), min(end, span['end'])
        if b > a:
            # Floor to whole milliseconds: the archive rejects references even 1 ms past the media duration.
            end_s = math.floor(min(span['duration'], span['origin'] + (b - span['begin']) / SR) * 1000 + 1e-6) / 1000
            start_s = min(end_s, math.floor((span['origin'] + (a - span['begin']) / SR) * 1000 + 1e-6) / 1000)
            refs.append({'media_id': span['media_id'], 'start_offset_seconds': start_s, 'end_offset_seconds': end_s,
                         'alignment': 'decoded_audio_stream_pts'})
    return refs


def parse_time(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00'))


def clip_end(asset):
    return parse_time(asset['captured_at']) + timedelta(seconds=float(asset['duration_seconds']))


def sample_at(spans, moment):
    """Absolute time -> stitched sample index, clamped to the stitched audio (None = its start)."""
    if moment is None:
        return 0
    for span in spans:
        rel = (moment - span['t0']).total_seconds()
        if rel < (span['end'] - span['begin']) / SR:
            return span['begin'] + max(0, int(round(rel * SR)))
    return spans[-1]['end']


def select_turns(turns, published, end, publish_all):
    """-> (turns to publish now, start sample of the first turn that must wait, or None).
    Audio before `published` was published by an earlier analysis of the same capture. A turn ending
    within TURN_GAP_SECONDS of the received audio's end may still continue in the next clip."""
    ready, hold = [], None
    for turn in turns:
        start = max(turn['start'], published)
        if turn['end'] - start < MIN_TURN_SECONDS * SR:
            continue
        if not publish_all and turn['end'] > end - TURN_GAP_SECONDS * SR:
            hold = start if hold is None else min(hold, start)
            continue
        ready.append({**turn, 'start': start})
    return ready, hold


def echoes_glossary(text, glossary):
    return sum(term.lower() in text.lower() for term in glossary) >= 4 and len(text) < 40 * 4


class Models:
    def __init__(self, weights, campplus, asr, device, language=None):
        self.language = language
        import torch
        from silero_vad import get_speech_timestamps
        from funasr import AutoModel
        from qwen_asr import Qwen3ASRModel
        self.torch, self.get_speech_timestamps = torch, get_speech_timestamps
        self.vad = torch.jit.load(str(Path(weights) / 'voice_silero.jit'), map_location='cpu').eval()
        self.campp = AutoModel(model=str(campplus), device='cpu', disable_update=True)
        self.asr = Qwen3ASRModel.from_pretrained(str(asr), dtype=torch.bfloat16 if device == 'cuda' else torch.float32,
            device_map='cuda:0' if device == 'cuda' else 'cpu', max_inference_batch_size=8, max_new_tokens=512)
        self.diarization_version = 'campplus-' + sha16(Path(campplus) / 'campplus_cn_common.bin') + ':silero-' + sha16(Path(weights) / 'voice_silero.jit')
        self.asr_version = 'qwen3-asr-1.7b-' + sha16(Path(asr) / 'model.safetensors.index.json')
        self.voice_model = self.diarization_version + ':spk-v1'

    def regions(self, audio):
        with self.torch.inference_mode():
            found = self.get_speech_timestamps(self.torch.from_numpy(audio.copy()), self.vad, sampling_rate=SR, return_seconds=False, **VAD)
        return [(r['start'], r['end']) for r in found]

    def embed(self, wave):
        value = self.campp.generate(input=wave, fs=SR)[0]['spk_embedding']
        value = np.asarray(value.cpu() if hasattr(value, 'cpu') else value, dtype=np.float64).reshape(-1)
        return value / np.linalg.norm(value)

    def transcribe(self, waves, contexts):
        if not waves:
            return []
        results = self.asr.transcribe(audio=[(w, SR) for w in waves], context=contexts, language=[self.language] * len(waves))
        return [(r.text or '').strip() for r in results]


class SpeechWorker:
    def __init__(self, store, archive_url, token, models, glossary=DEFAULT_GLOSSARY, window_clips=2, idle_seconds=8, reprocess_history=False):
        import httpx
        self.store, self.models, self.glossary = store, models, glossary
        self.window_clips, self.idle_seconds, self.reprocess_history = window_clips, idle_seconds, reprocess_history
        self.pipeline = PIPELINE + '-' + fingerprint(models.voice_model, models.asr_version, glossary, VAD, ANCHOR_SIM, MATCH_SIM, ASSIGN_SIM)[:16]
        self.cursor_key = 'speech_cursor:' + self.pipeline
        self.archive = httpx.Client(base_url=archive_url.rstrip('/'), timeout=httpx.Timeout(60, connect=3),
                                    headers={'Authorization': 'Bearer ' + token}, trust_env=False)

    def feed(self, after):
        response = self.archive.get('/v1/archive/feed', params={'after': after, 'limit': 100})
        response.raise_for_status()
        return response.json()

    def discover(self):
        cursor = self.store.config(self.cursor_key)
        if cursor is None and not self.reprocess_history:
            cursor = self.store.inherit_feed_cursor(self.cursor_key, 'speech_cursor')
        after = int(cursor or 0)
        for _ in range(5):
            page = self.feed(after)
            with self.store.db:
                for asset in page['items']:
                    if not (asset.get('metadata', {}).get('test_fixture') or asset['capture_id'].startswith(TEST_PREFIXES)):
                        self.store.db.execute('INSERT OR IGNORE INTO speech_pending VALUES (?,?,?,?,?)', (asset['media_id'],
                            asset['capture_id'], int(asset['clip_seq']), json.dumps(asset), time.time()))
                    after = asset['ingest_cursor']
                self.store.db.execute('INSERT OR REPLACE INTO config VALUES (?,?)', (self.cursor_key, str(after)))
            if not page['has_more']:
                break

    def ready(self, now=None):
        """Next unit of work for one capture, or None. New clips go first, each analysed right away after
        the clips still held for its capture (`stream`); a clip arriving after later ones were analysed is
        transcribed on its own (`late`); a capture idle for idle_seconds publishes what it still holds (`flush`)."""
        now = time.time() if now is None else now
        rows = lambda table: self.store.db.execute(
            f'SELECT media_id,capture_id,clip_seq,asset,seen_at FROM {table} ORDER BY capture_id,clip_seq').fetchall()
        pending, held = {}, {}
        for row in rows('speech_pending'):
            pending.setdefault(row['capture_id'], []).append(row)
        for row in rows('speech_context'):
            held.setdefault(row['capture_id'], []).append(row)
        for capture_id in sorted(pending, key=lambda c: min(r['seen_at'] for r in pending[c])):
            new, context = pending[capture_id], held.get(capture_id, [])
            late = [r for r in new if context and r['clip_seq'] <= context[-1]['clip_seq']]
            if late:
                return {'capture_id': capture_id, 'mode': 'late', 'context': [], 'new': late}
            return {'capture_id': capture_id, 'mode': 'stream', 'context': context, 'new': new[:self.window_clips]}
        for capture_id, context in held.items():
            if now - max(r['seen_at'] for r in context) >= self.idle_seconds:
                return {'capture_id': capture_id, 'mode': 'flush', 'context': context, 'new': []}
        return None

    def load_audio(self, asset, directory):
        path = Path(directory) / (asset['media_id'] + '.mov')
        digest = hashlib.sha256()
        with self.archive.stream('GET', '/v1/archive/media/' + asset['media_id'] + '/original') as response:
            response.raise_for_status()
            with open(path, 'wb') as handle:
                for chunk in response.iter_bytes():
                    handle.write(chunk); digest.update(chunk)
        if digest.hexdigest() != asset['sha256']:
            raise ValueError('original_checksum_mismatch')
        probe = json.loads(subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'a:0', '-show_entries',
            'stream=start_time:format=start_time', '-of', 'json', str(path)], capture_output=True, check=True, timeout=20).stdout)
        if not probe.get('streams'):
            return None, 0.
        origin = float(probe['streams'][0].get('start_time', 0)) - float(probe.get('format', {}).get('start_time', 0))
        pcm = subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(path), '-map', '0:a:0', '-ac', '1', '-ar', str(SR),
            '-f', 'f32le', '-'], capture_output=True, check=True, timeout=30).stdout
        return np.frombuffer(pcm, np.float32), max(0., origin)

    def analyze(self, batch, published_until=None, publish_all=True):
        """Return (samples, events, progress) for consecutive clips of a single capture. Only turns after
        `published_until` are transcribed; unless publish_all, a turn that may still continue waits.
        progress = {'published_until': datetime | None, 'hold_from': datetime | None}."""
        parts, spans, cursor, previous_end = [], [], 0, None
        with tempfile.TemporaryDirectory(prefix='speech-') as directory:
            for asset in sorted(batch, key=lambda a: a['clip_seq']):
                audio, origin = self.load_audio(asset, directory)
                if audio is None or not len(audio):
                    continue
                t0 = datetime.fromisoformat(asset['captured_at'].replace('Z', '+00:00')) + timedelta(seconds=origin)
                if previous_end is not None:
                    gap = int(max(0., min(.5, (t0 - previous_end).total_seconds())) * SR)
                    parts.append(np.zeros(gap, np.float32)); cursor += gap
                spans.append({'begin': cursor, 'end': cursor + len(audio), 't0': t0, 'media_id': asset['media_id'],
                              'origin': origin, 'duration': float(asset['duration_seconds'])})
                parts.append(audio); cursor += len(audio); previous_end = t0 + timedelta(seconds=len(audio) / SR)
        if not parts:
            return [], [], {'published_until': published_until, 'hold_from': None}
        audio = np.concatenate(parts)
        audio_end = locate(spans, len(audio) - 1)[0] + timedelta(seconds=1 / SR)
        regions = self.models.regions(audio)
        if not regions:
            return [], [], {'published_until': audio_end if publish_all else published_until, 'hold_from': None}
        capture_id = batch[0]['capture_id']
        batch_start = spans[0]['t0']
        windows = windows_for(regions)
        vectors = [self.models.embed(audio[a:b]) for _, a, b in windows]
        # Stage 1: long regions define this batch's speakers, matched to persistent voice entities.
        anchors = [i for i, (a, b) in enumerate(regions) if b - a >= ANCHOR_SECONDS * SR]
        candidates, matches, used = {}, {}, set()
        if anchors:
            A = np.array([self.models.embed(audio[regions[i][0]:regions[i][1]]) for i in anchors])
            if len(anchors) > 1:
                from sklearn.cluster import AgglomerativeClustering
                labels = AgglomerativeClustering(n_clusters=None, metric='cosine', linkage='average',
                                                 distance_threshold=1 - ANCHOR_SIM).fit_predict(A)
            else:
                labels = np.zeros(1, int)
            weight = {l: sum(regions[i][1] - regions[i][0] for i, x in zip(anchors, labels) if x == l) for l in set(labels)}
            for label in sorted(weight, key=weight.get, reverse=True):
                centroid = A[labels == label].mean(0); centroid /= np.linalg.norm(centroid)
                match = self.store.match_entity(self.models.voice_model, 'voice', centroid, batch_start.isoformat(),
                                                threshold=MATCH_SIM, margin=MATCH_MARGIN, excluded=used)
                used.add(match['entity_id']); matches[match['entity_id']] = match; candidates[match['entity_id']] = [centroid]
        # Recently active speakers cover batches with only short replies.
        since = (batch_start - timedelta(minutes=15)).isoformat()
        for entity_id, prototypes in self.store.recent_entities(self.models.voice_model, 'voice', since):
            candidates.setdefault(entity_id, [np.asarray(p) for p in prototypes])
        labels, sims = [], []
        for vector in vectors:
            scored = sorted(((max(float(vector @ p) for p in protos), eid) for eid, protos in candidates.items()), reverse=True)
            ok = scored and scored[0][0] >= ASSIGN_SIM
            labels.append(scored[0][1] if ok else None); sims.append(scored[0][0] if scored else None)
        turns, hold = select_turns(build_turns(regions, windows, labels), sample_at(spans, published_until), len(audio), publish_all)
        # Stage 2: transcribe each finished turn; glossary only for longer turns.
        waves = [audio[max(0, t['start'] - SR // 10):t['end'] + SR // 10] for t in turns]
        contexts = [('Words that may appear in the conversation: ' + ', '.join(self.glossary)) if (t['end'] - t['start']) >= CONTEXT_SECONDS * SR else '' for t in turns]
        texts = self.models.transcribe(waves, contexts)
        published = max([locate(spans, t['end'] - 1)[0] for t in turns] + [published_until] * bool(published_until), default=None)
        samples, events = [], []
        for turn, text, context in zip(turns, texts, contexts):
            if not text or echoes_glossary(text, self.glossary):
                continue
            seconds = (turn['end'] - turn['start']) / SR
            at, media_id, offset = locate(spans, turn['start'])
            end_at = at + timedelta(seconds=seconds)
            own = [v for (r, a, b), v, l in zip(windows, vectors, labels) if a < turn['end'] and b > turn['start'] and l == turn['speaker']]
            embedding = None
            if seconds >= EMBED_TURN_SECONDS and own:
                embedding = np.mean(own, 0); embedding = (embedding / np.linalg.norm(embedding)).tolist()
            entity_id = turn['speaker']
            similarity = max((float(np.asarray(embedding) @ p) for p in candidates.get(entity_id, [])), default=None) if embedding and entity_id else None
            assignment = (matches[entity_id]['assignment'] if entity_id in matches else 'provisional_recent_speaker') if entity_id else 'unknown_speaker'
            key = fingerprint(self.pipeline, capture_id, at.isoformat())[:48]
            sample_id, event_id = 'sample_' + key, 'speech_' + key
            samples.append({'sample_id': sample_id, 'entity_id': entity_id, 'model': self.models.voice_model, 'media_id': media_id,
                'capture_id': capture_id, 'offset': round(offset, 3), 'started_at': at.isoformat(timespec='microseconds'), 'assignment': assignment,
                'data': {'modality': 'voice', 'text': text, 'start_seconds': round(offset, 3), 'end_seconds': round(offset + seconds, 3),
                         'speech_seconds': round(seconds, 3), 'embedding': embedding, 'speaker_similarity': similarity,
                         'identity_status': 'provisional_voiceprint_cluster' if entity_id else 'unknown', 'pipeline': self.pipeline}})
            events.append({'event_id': event_id, 'kind': 'speech', 'capture_id': capture_id,
                'started_at': at.isoformat(timespec='microseconds'), 'ended_at': end_at.isoformat(timespec='microseconds'),
                'model': self.models.asr_version + '|' + self.models.voice_model, 'entity_ids': [entity_id] if entity_id else [],
                'media_refs': media_refs(spans, turn['start'], turn['end']),
                'body': {'text': text, 'speaker_entity_id': entity_id, 'sample_id': sample_id, 'assignment': assignment,
                         'embedding': [round(float(x), 4) for x in embedding] if entity_id and embedding else None,
                         'speaker_similarity': similarity, 'duration_seconds': round(seconds, 3), 'glossary_context': bool(context),
                         'pipeline': self.pipeline, 'asr_model': self.models.asr_version, 'diarization_model': self.models.voice_model,
                         'clock_source': 'client_wall_clock_plus_audio_pts',
                         'identity_semantics': 'voiceprint_cluster_not_verified_identity'}})
        return samples, events, {'published_until': audio_end if publish_all else published,
                                 'hold_from': None if hold is None else locate(spans, hold)[0]}

    def step(self, unit, record_events=True):
        """Analyse one unit from ready(), commit its turns and the capture's held clips; return its events."""
        db, capture_id, mode = self.store.db, unit['capture_id'], unit['mode']
        progress = db.execute('SELECT published_until,holding FROM speech_progress WHERE capture_id=?', (capture_id,)).fetchone()
        rows = unit['context'] + unit['new']
        window = [json.loads(r['asset']) for r in rows]
        samples, events, done = [], [], None
        if mode != 'flush' or (progress and progress['holding']):
            from . import metrics
            key = 'speech_failures:' + rows[-1]['media_id']
            started = time.perf_counter()
            published = parse_time(progress['published_until']) if mode != 'late' and progress and progress['published_until'] else None
            try:
                samples, events, done = self.analyze(window, published, mode != 'stream' or len(window) >= MAX_HOLD_CLIPS)
                metrics.job('speech_batch', 'ok', time.perf_counter() - started)
                metrics.EVENTS.labels(metrics.SERVICE, 'speech_turns', 'transcribed').inc(len(events))
            except Exception as error:
                metrics.job('speech_batch', 'error', time.perf_counter() - started)
                failures = int(self.store.config(key, '0')) + 1
                self.store.set_config(key, str(failures))
                if failures < 3:
                    raise
                # Clips that keep failing must not block later speech; the originals stay archived.
                self.store.set_config('speech_skipped:' + rows[-1]['media_id'], json.dumps({'error': type(error).__name__, 'detail': str(error)[:200]}))
                metrics.job('speech_batch', 'skipped')
            self.store.record_speech(samples, events if record_events else [])
            # Speakers are matched/created before transcription; anything left without a sample is noise.
            self.store.prune_empty_entities(self.models.voice_model)
        with db:
            db.executemany('DELETE FROM speech_pending WHERE media_id=?', [(r['media_id'],) for r in unit['new']])
            if mode == 'stream' and done:
                # Hold the clips from the first unfinished turn on (at least the newest clip) as the next context.
                keep_from = done['hold_from'] or clip_end(window[-1])
                keep = [r for r, a in zip(rows, window) if clip_end(a) > keep_from] or rows[-1:]
                db.execute('DELETE FROM speech_context WHERE capture_id=?', (capture_id,))
                db.executemany('INSERT INTO speech_context VALUES (?,?,?,?,?)', [tuple(r) for r in keep])
                db.execute('INSERT OR REPLACE INTO speech_progress VALUES (?,?,?)', (capture_id,
                    done['published_until'] and done['published_until'].isoformat(), int(done['hold_from'] is not None)))
            elif mode != 'late':
                db.execute('DELETE FROM speech_context WHERE capture_id=?', (capture_id,))
                db.execute('DELETE FROM speech_progress WHERE capture_id=?', (capture_id,))
        if record_events and events:
            self.publish(events)
        print(json.dumps({'status': 'batch_done', 'mode': mode, 'capture': capture_id[-8:], 'clips': len(rows),
                          'new': len(unit['new']), 'turns': len(events), 'holding': bool(done and done['hold_from'])}), flush=True)
        return events

    def publish(self, events):
        """Deliver this unit's events now instead of waiting for the recognition worker's outbox flush."""
        from . import metrics
        try:
            deliver(self.archive, self.store, self.store.outbox_events([e['event_id'] for e in events]))
        except Exception as error:
            print(json.dumps({'status': 'delivery_deferred', 'error_class': type(error).__name__}), flush=True)
            return
        for event in events:
            metrics.job('speech_turn_lag', 'published', max(0., time.time() - parse_time(event['ended_at']).timestamp()))

    def tick(self):
        self.discover()
        unit = self.ready()
        if not unit:
            return False
        self.step(unit)
        return True


def main():
    from .store import Store
    from . import metrics
    metrics.serve()
    p = argparse.ArgumentParser()
    p.add_argument('--state', default='/data/entities.sqlite'); p.add_argument('--weights', default=os.environ.get('ENTITY_WEIGHTS_DIR', '/models/entities'))
    p.add_argument('--campplus', default='/models/campplus'); p.add_argument('--asr', default='/models/qwen3-asr-1.7b')
    p.add_argument('--device', choices=('cuda', 'cpu'), default='cuda'); p.add_argument('--window-clips', type=int, default=2, help='new clips per analysis while catching up')
    p.add_argument('--language', default=os.environ.get('ASR_LANGUAGE') or None, help='ASR language name; omit for automatic detection')
    p.add_argument('--idle-seconds', type=float, default=8); p.add_argument('--glossary-file')
    p.add_argument('--reprocess-history', action='store_true')
    p.add_argument('--replay-capture', help='analyze one capture into --state (use a copy of the DB) and write events JSON to --dry-run-out')
    p.add_argument('--dry-run-out')
    p.add_argument('--start-cursor', type=int, help='testing: initial feed cursor when none is stored')
    p.add_argument('--exit-when-idle', action='store_true', help='testing: stop once nothing is ready')
    a = p.parse_args()
    import logging
    logging.getLogger('httpx').setLevel(logging.WARNING)  # the archive feed is polled twice a second
    glossary = [x.strip() for x in Path(a.glossary_file).read_text().splitlines() if x.strip()] if a.glossary_file else DEFAULT_GLOSSARY
    night_pause = NightPause('speech')
    models = Models(a.weights, a.campplus, a.asr, a.device, a.language)
    worker = SpeechWorker(Store(a.state), os.environ['ARCHIVE_URL'], os.environ['SCENE_ARCHIVE_TOKEN'], models, glossary,
                          a.window_clips, a.idle_seconds, a.reprocess_history)
    if a.replay_capture:
        items, after = [], 0
        while True:
            page = worker.feed(after); items += page['items']; after = page['next_cursor']
            if not page['has_more']:
                break
        clips = sorted((i for i in items if i['capture_id'] == a.replay_capture), key=lambda i: i['clip_seq'])
        # Feed clips in capture order as if each just arrived, then let the capture go idle.
        # Samples are recorded like the live loop so later units can match speakers; --state must be a DB copy.
        out = []
        def drain(now):
            while unit := worker.ready(now):
                t = time.perf_counter(); events = worker.step(unit, record_events=False)
                out.append({'mode': unit['mode'], 'arrived': now, 'clips': len(unit['context']) + len(unit['new']),
                            'seconds': round(time.perf_counter() - t, 2), 'events': events})
        for clip in clips:
            arrived = clip_end(clip).timestamp()
            with worker.store.db:
                worker.store.db.execute('INSERT OR IGNORE INTO speech_pending VALUES (?,?,?,?,?)',
                    (clip['media_id'], clip['capture_id'], int(clip['clip_seq']), json.dumps(clip), arrived))
            drain(arrived)
        drain(math.inf)
        Path(a.dry_run_out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
        return
    if a.start_cursor is not None and worker.store.config(worker.cursor_key) is None:
        worker.store.set_config(worker.cursor_key, str(a.start_cursor))
    print(json.dumps({'status': 'started', 'pipeline': worker.pipeline}), flush=True)
    night_pause.publish_ready()
    while True:
        try:
            # A unit commits before reaching this boundary. Pending and held clips stay intact.
            if night_pause.paused():
                if a.exit_when_idle:
                    return
                time.sleep(3)
                continue
            if not worker.tick():
                if a.exit_when_idle:
                    return
                time.sleep(POLL_SECONDS)
        except Exception as error:
            print(json.dumps({'status': 'retry_pending', 'error_class': type(error).__name__, 'detail': str(error)[:200]}), flush=True)
            time.sleep(15)


if __name__ == '__main__':
    main()
