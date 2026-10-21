"""End-to-end check of the naming service with a synthetic test capture, timing every stage.

Run inside the naming container on dgx-spark-2 (it has ffmpeg, the archive token, the archive DB read-only and the
graph credentials):
  docker cp mug.jpg eleagent-naming:/tmp/naming-e2e/mug.jpg; docker cp test-cup.wav eleagent-naming:/tmp/naming-e2e/
  docker exec -w /app -e PYTHONPATH=/app eleagent-naming python naming/tests/inject_clip.py \
      --image /tmp/naming-e2e/mug.jpg --audio /tmp/naming-e2e/test-cup.wav --text 这是我的测试杯子

1. builds a 7 s clip: a still archived frame (e.g. a white mug) as H.264 video plus the spoken name (TTS WAV) at
   2.5 s as AAC, in a .mov like the phone's;
2. archives it with capture_id 'entity-acceptance-naming-<timestamp>' through POST /v1/archive/clips with the
   metadata/receipt the observer (behind graph-api's capture proxy) attaches after probing a phone clip;
3. speech: the deployed speech worker skips test captures (speech.py discover(): TEST_PREFIXES), so by default
   (--speech inject) this script posts the `speech` event the worker would write for the turn (same shape, same
   media reference) right after the clip is archived, i.e. a zero-batching speech worker; --speech wait polls for a
   real worker event instead (for a worker that accepts test captures);
4. polls the archive for the recognition worker's object events and Neo4j for the NamingEvent and the view
   (:ObjectView {source:'naming'}, DINOv2-S vector) of the named memory object, which carries the name;
5. prints a timeline (seconds after the clip POST). Clean up afterwards with
   python -m naming.cleanup --capture-id <capture_id>.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import tempfile
import time

PROBE = ['ffprobe', '-v', 'error', '-show_entries',
         'format=format_name,duration,start_time:stream=codec_type,codec_name,width,height,duration,start_time,time_base,sample_rate',
         '-of', 'json']
NAMED = """MATCH (p:ObjectView {source: 'naming'}) WHERE p.memory_id = $id AND p.vector IS NOT NULL
  MATCH (e:Entity {id: p.memory_id}) WHERE e.layer = 'memory' AND e.kind = 'object'
  RETURN e.id AS id, e.name AS name, count(p) AS views"""


def build_clip(image, audio, out, seconds, speech_at):
    common = ['ffmpeg', '-nostdin', '-v', 'error', '-y', '-loop', '1', '-framerate', '30', '-t', str(seconds), '-i', str(image),
              '-i', str(audio), '-filter_complex',
              f'[0:v]scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p[v];[1:a]adelay={int(speech_at * 1000)}:all=1,apad,atrim=0:{seconds}[a]',
              '-map', '[v]', '-map', '[a]', '-c:a', 'aac', '-ar', '48000', '-ac', '1', '-t', str(seconds)]
    for video in (['-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20'], ['-c:v', 'mpeg4', '-q:v', '3']):
        result = subprocess.run(common + video + ['-movflags', '+faststart', str(out)], capture_output=True, timeout=60)
        if result.returncode == 0:
            return video[1]
    raise RuntimeError('ffmpeg failed: ' + result.stderr.decode()[-400:])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--audio', required=True)
    parser.add_argument('--text', default='这是我的测试杯子', help='transcript used for the injected speech event')
    parser.add_argument('--seconds', type=float, default=7.)
    parser.add_argument('--speech-at', type=float, default=2.5)
    parser.add_argument('--speech', choices=('inject', 'wait'), default='inject')
    parser.add_argument('--capture-id', default='entity-acceptance-naming-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S'))
    parser.add_argument('--timeout', type=float, default=120.)
    parser.add_argument('--archive-db', default=os.environ.get('ARCHIVE_DB', '/archive/archive.sqlite3'))
    args = parser.parse_args()
    if not args.capture_id.startswith('entity-acceptance-'):
        parser.error('the capture id must be a test capture (entity-acceptance-...)')
    import httpx
    from neo4j import GraphDatabase
    archive = httpx.Client(base_url=os.environ['ARCHIVE_URL'].rstrip('/'), timeout=httpx.Timeout(60, connect=3),
                           headers={'Authorization': 'Bearer ' + os.environ['SCENE_ARCHIVE_TOKEN']}, trust_env=False)
    driver = GraphDatabase.driver(os.environ['GRAPH_URL'], auth=tuple(os.environ['NEO4J_AUTH'].split('/', 1)))
    marks, t0 = {}, None

    def mark(stage, **extra):
        marks[stage] = {'s': round(time.time() - t0, 2), **extra}
        print(json.dumps({'stage': stage, **marks[stage]}, ensure_ascii=False), flush=True)

    with tempfile.TemporaryDirectory(prefix='naming-e2e-') as directory:
        clip = Path(directory) / 'clip.mov'
        codec = build_clip(args.image, args.audio, clip, args.seconds, args.speech_at)
        metadata = json.loads(subprocess.run(PROBE + [str(clip)], capture_output=True, check=True, timeout=10).stdout)
        duration = float(metadata['format']['duration'])
        captured_at = datetime.now(timezone.utc) - timedelta(seconds=duration)  # the clip "ended" just now, like a live upload
        metadata.update(capture_clock='client_wall_clock_at_recording_start_not_hardware_sync',
                        captured_end_at=(captured_at + timedelta(seconds=duration)).isoformat(), source='native_camera',
                        camera_facing='back', transport='successive_movie_segments_with_possible_capture_gaps',
                        test_purpose='naming-e2e')
        receipt = {'status': 'archived', 'capture_id': args.capture_id, 'clip_seq': 0, 'packet_count': 0, 'audio_seconds': duration,
                   'video_frames': 0, 'first_seq': None, 'last_seq': None, 'duplicate': False, 'analysis_status': 'pending'}
        info = {'duration_seconds': duration, 'metadata': metadata, 'packet_spans': [], 'receipt': receipt}
        payload = clip.read_bytes()
        t0 = time.time()
        response = archive.post('/v1/archive/clips', params={'capture_id': args.capture_id, 'clip_seq': 0,
                                                             'captured_at': captured_at.isoformat()},
                                headers={'Content-Type': 'application/octet-stream', 'X-Archive-Info': json.dumps(info, separators=(',', ':'))},
                                content=payload)
        response.raise_for_status()
        media_id = response.json()['media_id']
        mark('clip_archived', capture_id=args.capture_id, media_id=media_id, duration=round(duration, 3), codec=codec,
             bytes=len(payload))

    turn_start = captured_at + timedelta(seconds=args.speech_at)
    audio_seconds = float(json.loads(subprocess.run(PROBE + [args.audio], capture_output=True, check=True, timeout=10).stdout)['format']['duration'])
    turn_end = turn_start + timedelta(seconds=audio_seconds)
    if args.speech == 'inject':
        key = hashlib.sha256(f'naming-e2e|{args.capture_id}|{turn_start.isoformat()}'.encode()).hexdigest()[:48]
        event = {'event_id': 'speech_' + key, 'kind': 'speech', 'capture_id': args.capture_id,
                 'started_at': turn_start.isoformat(timespec='microseconds'), 'ended_at': turn_end.isoformat(timespec='microseconds'),
                 'model': 'naming-e2e-injected', 'entity_ids': [],
                 'media_refs': [{'media_id': media_id, 'start_offset_seconds': round(args.speech_at, 3),
                                 'end_offset_seconds': round(min(duration, args.speech_at + audio_seconds), 3), 'alignment': 'decoded_audio_stream_pts'}],
                 'body': {'text': args.text, 'speaker_entity_id': None, 'sample_id': 'sample_' + key, 'assignment': 'unknown_speaker',
                          'embedding': None, 'speaker_similarity': None, 'duration_seconds': round(audio_seconds, 3),
                          'glossary_context': False, 'pipeline': 'naming-e2e-injected', 'asr_model': 'tts-script-transcript',
                          'diarization_model': None, 'clock_source': 'client_wall_clock_plus_audio_pts',
                          'identity_semantics': 'voiceprint_cluster_not_verified_identity', 'injected_by': 'naming/tests/inject_clip.py'}}
        archive.post('/v1/archive/events', json=event).raise_for_status()
        mark('speech_event_injected', event_id=event['event_id'])

    def archive_rows(kind):
        db = sqlite3.connect(f'file:{args.archive_db}?mode=ro', uri=True, timeout=10)
        try:
            return db.execute('SELECT recorded_at, body_json FROM archive_events WHERE capture_id=? AND kind=? ORDER BY rowid',
                              (args.capture_id, kind)).fetchall()
        finally:
            db.close()

    def recorded_s(value):
        return round(datetime.fromisoformat(value).timestamp() - t0, 2)

    naming_event = view = named = None
    deadline = time.time() + args.timeout
    while time.time() < deadline and not (naming_event and view and named):
        if 'objects_first' not in marks:
            rows = archive_rows('object')
            if rows:
                labels = sorted({o.get('label') for _, b in rows for o in json.loads(b).get('observations') or []})
                mark('objects_first', archive_recorded_s=recorded_s(rows[0][0]), events=len(rows), labels=labels)
        if 'speech_event' not in marks:
            rows = archive_rows('speech')
            if rows:
                mark('speech_event', archive_recorded_s=recorded_s(rows[0][0]), text=json.loads(rows[0][1]).get('text'))
        with driver.session() as session:
            if naming_event is None:
                found = session.run('MATCH (n:NamingEvent {capture_id:$c}) RETURN properties(n) AS n ORDER BY n.created_at LIMIT 1',
                                    c=args.capture_id).single()
                if found:
                    naming_event = found['n']
                    mark('naming_event', created_s=recorded_s(naming_event['created_at']), status=naming_event['status'],
                         intent=naming_event['intent'], name=naming_event.get('name'), memory_id=naming_event.get('memory_id'),
                         matched_by=naming_event.get('matched_by'), grounding=naming_event.get('grounding'), iou=naming_event.get('iou'),
                         strong_cosine=naming_event.get('strong_cosine'), timings=json.loads(naming_event.get('timings') or '{}'))
            if naming_event and naming_event.get('memory_id') and view is None:
                found = session.run("""MATCH (p:ObjectView {memory_id:$m, source:'naming'}) WHERE p.media_id = $media
                    RETURN p.id AS id, p.created_at AS created_at, p.model AS model ORDER BY p.created_at LIMIT 1""",
                                    m=naming_event['memory_id'], media=media_id).single()
                if found:
                    view = dict(found)
                    mark('view', created_s=recorded_s(view['created_at']), id=view['id'], model=view['model'])
            if view and named is None:
                found = session.run(NAMED, id=naming_event['memory_id']).single()
                if found and found['name'] == naming_event.get('name'):
                    named = dict(found)
                    mark('named_object', name=found['name'], views=found['views'])
        if naming_event and naming_event.get('status') in ('no_target', 'ignored'):
            break
        time.sleep(.5)
    driver.close()
    print(json.dumps({'capture_id': args.capture_id, 'media_id': media_id, 'captured_at': captured_at.isoformat(),
                      'utterance_end_s': round(turn_end.timestamp() - t0, 2), 'timeline': marks,
                      'ok': bool(naming_event and view and named)}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
