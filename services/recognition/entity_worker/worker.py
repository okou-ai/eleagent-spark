from __future__ import annotations
import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import httpx

from .faces import FaceModel
from .frames import select_frames
from .store import Store, canonical
from . import metrics
from .night_pause import NightPause
from .outbox import deliver

MAX_MEDIA_BYTES = 16_000_000

OBJECT_TRACK_SECONDS = 8.0


class Paused(Exception):
    pass


def instant(asset, offset):
    start = datetime.fromisoformat(asset['captured_at'].replace('Z', '+00:00'))
    return (start + timedelta(seconds=offset)).astimezone(timezone.utc).isoformat(timespec='microseconds')


def fingerprint(*parts):
    return hashlib.sha256(canonical(parts).encode()).hexdigest()


class NoGate:
    """Run without a MiniCPM gate on the dedicated recognition host."""
    def check(self):
        return None


class GatewayGate:
    def __init__(self, url, worker_id=None, token=None, verify=True):
        self.worker_id = worker_id or os.environ.get('GATEWAY_WORKER_ID', 'recognition')
        self.client = httpx.Client(base_url=url.rstrip('/'), timeout=3, verify=verify, trust_env=False,
            headers={'Authorization': 'Bearer ' + token} if token else {})

    def check(self):
        try:
            response = self.client.get('/workers'); response.raise_for_status()
            payload = response.json()
            workers = payload if isinstance(payload, list) else payload.get('workers', [])
            target = next(item for item in workers if item.get('worker_id') == self.worker_id)
            if target.get('status') != 'idle':
                raise Paused('standby_minicpm_busy')
            stamp = datetime.fromisoformat(target['last_heartbeat'].replace('Z', '+00:00'))
            if stamp.tzinfo is None:
                stamp = stamp.replace(tzinfo=timezone.utc)
            age = (datetime.now(timezone.utc) - stamp).total_seconds()
            if not -5 <= age <= 10:
                raise Paused('gateway_heartbeat_stale')
        except (httpx.HTTPError, StopIteration, KeyError, ValueError, TypeError):
            raise Paused('gateway_state_unknown') from None


class Worker:
    def __init__(self, store, archive_url, token, faces, gate, objects=None, voices=None, sample_fps=1., max_frames=12, voice_threshold=.65, voice_margin=.10, reprocess_history=False):
        self.store, self.faces, self.objects, self.voices, self.gate = store, faces, objects, voices, gate
        self.reprocess_history = reprocess_history
        self.sample_fps, self.max_frames = sample_fps, max_frames
        self.voice_threshold, self.voice_margin = voice_threshold, voice_margin
        self.pipeline = 'entities-v1-' + fingerprint(faces.version if faces else None,
            objects.version if objects else None, voices.version if voices else None, sample_fps, max_frames, voice_threshold, voice_margin)[:32]
        self.cursor_key = 'ingest_cursor:' + self.pipeline
        if not reprocess_history:
            self.store.inherit_entity_pipeline(self.pipeline)
        self.store.set_config('model_versions', canonical({'face':faces.version if faces else None, 'object':objects.version if objects else None, 'voice':voices.version if voices else None}))
        self.archive = httpx.Client(base_url=archive_url.rstrip('/'), timeout=httpx.Timeout(30, connect=3),
            headers={'Authorization': 'Bearer ' + token}, trust_env=False)

    def discover(self):
        cursor = int(self.store.config(self.cursor_key, '0'))
        response = self.archive.get('/v1/archive/feed', params={'after': cursor, 'limit': 100})
        response.raise_for_status()
        page = response.json()
        if page.get('order') != 'durable_ingest_order_not_capture_time':
            raise ValueError('archive_ingest_feed_required')
        for asset in page['items']:
            # Public acceptance fixtures remain available in the archive, but
            # must never enroll people/objects into the production entity DB.
            if asset.get('metadata', {}).get('test_fixture') is True:
                self.store.set_config(self.cursor_key, str(asset['ingest_cursor']))
                continue
            # By default a new pipeline version does not redo media an earlier version already processed.
            if not self.reprocess_history and self.store.processed_elsewhere(asset['media_id'], self.pipeline):
                self.store.set_config(self.cursor_key, str(asset['ingest_cursor']))
                continue
            if not self.store.enqueue(asset, self.pipeline):
                break
            self.store.set_config(self.cursor_key, str(asset['ingest_cursor']))

    def flush(self):
        deliver(self.archive, self.store, self.store.pending_events())

    def download(self, asset, destination):
        if not 0 < asset['size_bytes'] <= MAX_MEDIA_BYTES:
            raise ValueError('media_size_outside_bounds')
        digest = hashlib.sha256(); size = 0
        # Construct the fixed endpoint ourselves; do not follow a server-supplied URL or redirects.
        with self.archive.stream('GET', '/v1/archive/media/' + asset['media_id'] + '/original') as response:
            response.raise_for_status()
            with destination.open('wb') as file:
                for block in response.iter_bytes(256 * 1024):
                    size += len(block)
                    if size > MAX_MEDIA_BYTES:
                        raise ValueError('media_download_exceeds_bound')
                    digest.update(block); file.write(block)
        if size != asset['size_bytes'] or digest.hexdigest() != asset['sha256']:
            raise ValueError('original_checksum_mismatch')

    def _event(self, asset, modality, model, start, end, observations, suffix):
        return {'event_id': 'entity_' + fingerprint(self.pipeline, asset['media_id'], modality, model, suffix)[:48],
            'kind': modality, 'capture_id': asset['capture_id'], 'model': model,
            'started_at': instant(asset, start), 'ended_at': instant(asset, end),
            'entity_ids': sorted({item['entity_id'] for item in observations if item.get('entity_id')}),
            'media_refs': [{'media_id': asset['media_id'], 'start_offset_seconds': start,
                'end_offset_seconds': end, 'alignment': 'decoded_frame_pts' if modality != 'voice' else 'decoded_audio_stream_pts'}],
            'body': {'observations': observations, 'pipeline': self.pipeline, 'sample_fps': self.sample_fps, 'max_frames_per_clip': self.max_frames,
                'identity_status': 'provisional_anonymous_clusters_not_verified_identity',
                'clock_source': 'client_wall_clock_plus_source_pts', 'original_sha256': asset['sha256']}}

    def _observation(self, asset, modality, model, offset, rank, data, excluded, eligible=True, threshold=.55, margin=.08, track_window=None):
        sample_id = 'sample_' + fingerprint(asset['media_id'], model, round(offset, 6), rank)[:48]
        data['modality'] = modality
        assigned = self.store.assign(sample_id, model, asset, offset, instant(asset, offset), data,
            excluded=excluded, cluster_eligible=eligible, modality=modality, threshold=threshold, margin=margin,
            track_window=track_window)
        if assigned['entity_id']:
            excluded.add(assigned['entity_id'])
        observation = {'sample_id': sample_id, **{key: value for key, value in data.items() if key != 'embedding'}, **assigned}
        # The archive is the single source for everything downstream: assigned detections carry their vector.
        if assigned['entity_id'] and data.get('embedding') is not None:
            observation['embedding'] = [round(float(x), 4) for x in data['embedding']]
        return observation

    def process(self, job):
        asset = json.loads(job['asset']); events = []
        with tempfile.TemporaryDirectory(prefix='entity-source-') as directory:
            directory = Path(directory); source = directory / 'source.mov'
            self.gate.check(); self.download(asset, source)
            frames = select_frames(source, directory, asset['duration_seconds'], fps=self.sample_fps, max_frames=self.max_frames)
            self.store.record_frames(asset['media_id'], len(frames))
            for offset, frame_number, frame in frames:
                self.gate.check()
                if self.faces:
                    used = set()
                    observations = [self._observation(asset, 'face', self.faces.version, offset, rank, face, used)
                        for rank, face in enumerate(self.faces.infer(frame))]
                    events.append(self._event(asset, 'face', self.faces.version, offset, offset, observations, frame_number))
                if self.objects:
                    self.gate.check(); used = set(); observations = []
                    for rank, item in enumerate(self.objects.extract(frame[:, :, ::-1].copy())):
                        data = dict(item); data['embedding'] = data.pop('feature', None)
                        # Match category-scoped appearance across sessions, never claim this proves
                        # the same physical item. Same-frame exclusions prevent duplicate assignment.
                        data['identity_semantics'] = 'appearance_cluster_not_verified_physical_instance'
                        matching_model = self.objects.version + ':class-' + str(data.get('class_id', data.get('label')))
                        observations.append(self._observation(asset, 'object', matching_model, offset, rank, data, used,
                            eligible=data['embedding'] is not None, threshold=.85, track_window=OBJECT_TRACK_SECONDS))
                    events.append(self._event(asset, 'object', self.objects.version, offset, offset, observations, frame_number))
            if self.voices:
                self.gate.check()
                probe = subprocess.run(['ffprobe','-v','error','-select_streams','a:0','-show_entries',
                    'stream=start_time:format=start_time','-of','json',str(source)],capture_output=True,check=True,timeout=20)
                metadata = json.loads(probe.stdout)
                streams = metadata.get('streams', [])
                if streams:
                    audio_origin = float(streams[0].get('start_time', 0)) - float(metadata.get('format', {}).get('start_time', 0))
                    audio = directory / 'audio.wav'
                    subprocess.run(['ffmpeg','-nostdin','-v','error','-threads','1','-i',str(source),'-map','0:a:0',
                        '-vn','-ac','1','-ar','16000','-c:a','pcm_s16le',str(audio)],check=True,capture_output=True,timeout=30)
                    for rank, segment in enumerate(self.voices.extract(audio)):
                        self.gate.check()
                        start = max(0.0, audio_origin + segment['start_seconds'])
                        end = min(asset['duration_seconds'], audio_origin + segment['end_seconds'])
                        if end <= start:
                            continue
                        data = dict(segment)
                        # VAD+embedding cannot certify one speaker or reject all overlap.
                        # Eligible embeddings may receive explicitly provisional anonymous IDs.
                        data.setdefault('identity_status', 'provisional_no_overlap_detector' if data.get('cluster_eligible') and data.get('candidate_for_clustering') else 'unknown')
                        observation = self._observation(asset, 'voice', self.voices.version, start, rank, data, set(),
                            eligible=bool(data.get('candidate_for_clustering')) and bool(data.get('cluster_eligible')), threshold=self.voice_threshold, margin=self.voice_margin)
                        events.append(self._event(asset, 'voice', self.voices.version, start, end, [observation], rank))
        self.store.complete_job(job['id'], events)

    def tick(self):
        self.flush(); self.discover(); self.gate.check()
        job = self.store.take_job(time.time(), self.pipeline)
        if job:
            started = time.perf_counter()
            try:
                self.process(job)
                metrics.job('entity_clip', 'ok', time.perf_counter() - started)
            except Paused:
                metrics.job('entity_clip', 'paused')
                self.store.fail_job(job['id'], 'standby_minicpm_busy_or_unknown', time.time(), paused=True)
                raise
            except Exception as error:
                metrics.job('entity_clip', 'error', time.perf_counter() - started)
                self.store.fail_job(job['id'], type(error).__name__, time.time())
                raise
        self.flush()
        return bool(job)


def main():
    metrics.serve()
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=Path, default=Path(os.environ.get('ENTITY_WEIGHTS_DIR', '/models/entities')))
    parser.add_argument('--state', type=Path, default=Path('/data/entities.sqlite'))
    parser.add_argument('--objects', action='store_true')
    parser.add_argument('--voices', action='store_true')
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--minicpm-gate', choices=('standby', 'off'), default='standby',
        help="standby: pause while the co-located MiniCPM worker is busy; off: run alongside it")
    parser.add_argument('--reprocess-history', action='store_true', help='let a changed pipeline redo media earlier pipelines finished')
    parser.add_argument('--device', choices=('cpu','cuda'), default='cpu')
    parser.add_argument('--object-device', choices=('cpu','cuda'), default='cpu')
    parser.add_argument('--sample-fps', type=float, default=1.)
    parser.add_argument('--max-frames', type=int, default=12)
    parser.add_argument('--voice-threshold', type=float, default=.65)
    parser.add_argument('--voice-margin', type=float, default=.10)
    args = parser.parse_args()
    if not 0 < args.sample_fps <= 2 or not 1 <= args.max_frames <= 121 or not 0 <= args.voice_threshold <= 1 or not 0 <= args.voice_margin <= 1:
        parser.error('sampling or matching limits out of range')
    args.state.parent.mkdir(parents=True, exist_ok=True)
    # Exactly one processing loop owns this DB; a second process cannot double-run GPU work.
    lock = args.state.with_suffix('.worker.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    night_pause = NightPause('entity')
    faces = FaceModel(args.weights)
    objects = voices = None
    if args.objects:
        from .objects import ObjectDetector
        objects = ObjectDetector(args.weights, device=args.object_device)
    if args.voices:
        from .voice import VoiceExtractor
        voices = VoiceExtractor(args.weights, device=args.device)
    gate = NoGate() if args.minicpm_gate == 'off' else GatewayGate(os.environ['GATEWAY_URL'], token=os.environ.get('GATEWAY_TOKEN'),
        verify=False if os.environ.get('GATEWAY_INSECURE_LOCAL') == '1' else os.environ.get('GATEWAY_CA_BUNDLE', True))
    worker = Worker(Store(args.state, recover=True), os.environ['ARCHIVE_URL'], os.environ['SCENE_ARCHIVE_TOKEN'], faces, gate, objects, voices, args.sample_fps, args.max_frames, args.voice_threshold, args.voice_margin, args.reprocess_history)
    if not args.reprocess_history:
        dropped = worker.store.drop_superseded(worker.pipeline)
        print(canonical({'status': 'superseded_history_skipped', 'dropped_jobs': dropped}), flush=True)
    worker.sample_fps, worker.max_frames = args.sample_fps, args.max_frames
    night_pause.publish_ready()
    while True:
        delay = 5
        try:
            # tick() owns a complete job and flush; the boundary has no GPU work in flight.
            if night_pause.paused():
                if args.once:
                    return
                time.sleep(3)
                continue
            processed = worker.tick()
            if processed: delay = 0
            worker.store.set_config('worker_status', canonical({'phase':'ready','at':datetime.now(timezone.utc).isoformat()}))
            print(canonical({'status': 'ok', **worker.store.status()}), flush=True)
        except Paused:
            worker.store.set_config('worker_status', canonical({'phase':'paused_for_minicpm_or_unknown_gateway','at':datetime.now(timezone.utc).isoformat()}))
            print(canonical({'status': 'paused_for_minicpm_or_unknown_gateway'}), flush=True)
        except Exception as error:
            # Never print response bodies, credentials, raw embeddings or private observations.
            worker.store.set_config('worker_status', canonical({'phase':'retry_pending','error_class':type(error).__name__,'at':datetime.now(timezone.utc).isoformat()}))
            print(canonical({'status': 'retry_pending', 'error_class': type(error).__name__}), flush=True)
            if args.once:
                raise SystemExit(1)
        if args.once:
            break
        if delay: time.sleep(delay)

if __name__ == '__main__':
    main()
