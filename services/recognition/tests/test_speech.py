import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import httpx
import numpy as np
from entity_worker.speech import SR, SpeechWorker, build_turns, echoes_glossary, locate, media_refs, select_turns, windows_for, DEFAULT_GLOSSARY
from entity_worker.store import Store


def test_windows_and_turns_split_speakers_at_votes_and_keep_unknown_separate():
    regions = [(0, 2 * SR), (int(2.3 * SR), int(8.3 * SR)), (9 * SR, int(9.6 * SR))]
    windows = windows_for(regions)
    assert windows[0] == (0, 0, 2 * SR) and windows[-1] == (2, 9 * SR, int(9.6 * SR))
    assert len(windows) == 7
    labels = ['a', 'a', 'a', 'a', 'b', 'b', None]
    turns = build_turns(regions, windows, labels)
    assert [t['speaker'] for t in turns] == ['a', 'b', None]
    assert turns[0]['start'] == 0 and turns[-1]['start'] == 9 * SR


def test_locate_and_media_refs_map_stitched_audio_back_to_originals():
    t0 = datetime(2026, 10, 10, 3, 0, tzinfo=timezone.utc)
    spans = [{'begin': 0, 'end': 5 * SR, 't0': t0, 'media_id': 'm1', 'origin': .02, 'duration': 5.03},
             {'begin': int(5.2 * SR), 'end': int(10.2 * SR), 't0': t0 + timedelta(seconds=5.2), 'media_id': 'm2', 'origin': 0., 'duration': 5.}]
    at, media, offset = locate(spans, int(6.2 * SR))
    assert media == 'm2' and abs(offset - 1) < 1e-6 and at == t0 + timedelta(seconds=6.2)
    refs = media_refs(spans, 4 * SR, int(6.2 * SR))
    assert [(r['media_id'], r['start_offset_seconds'], r['end_offset_seconds']) for r in refs] == [('m1', 4.02, 5.02), ('m2', 0.0, 1.0)]


def test_glossary_echo_is_detected():
    glossary = ['roadmap', 'milestone', 'planning', 'review']
    assert echoes_glossary('Roadmap milestone planning review.', glossary)
    assert not echoes_glossary('The roadmap review is scheduled for Thursday.', glossary)


def test_match_entity_reuses_speaker_after_samples_and_records_speech_atomically(tmp_path):
    db = Store(tmp_path / 'speech.sqlite')
    v = np.array([1., 0., 0.]); w = np.array([0., 1., 0.])
    first = db.match_entity('campp', 'voice', v, '2026-10-10T03:00:00+00:00')
    assert first['assignment'] == 'provisional_new'
    sample = {'sample_id': 's1', 'entity_id': first['entity_id'], 'model': 'campp', 'media_id': 'm1', 'capture_id': 'c',
              'offset': 1., 'started_at': '2026-10-10T03:00:01.000000+00:00', 'assignment': 'provisional_new',
              'data': {'modality': 'voice', 'text': 'Hello', 'start_seconds': 1., 'end_seconds': 4., 'embedding': [2., 0., 0.]}}
    event = {'event_id': 'speech_1', 'kind': 'speech', 'body': {'text': 'Hello'}}
    db.record_speech([sample], [event])
    db.record_speech([sample], [event])
    assert db.db.execute('SELECT count(*) FROM samples').fetchone()[0] == 1
    assert db.db.execute('SELECT count(*) FROM outbox').fetchone()[0] == 1
    again = db.match_entity('campp', 'voice', np.array([.9, .1, 0.]), '2026-10-10T03:01:00+00:00')
    assert again['entity_id'] == first['entity_id'] and again['assignment'] == 'provisional_match'
    other = db.match_entity('campp', 'voice', w, '2026-10-10T03:01:00+00:00')
    assert other['entity_id'] != first['entity_id']
    assert db.match_entity('campp', 'voice', v, '2026-10-10T03:02:00+00:00', excluded={first['entity_id']})['entity_id'] not in (first['entity_id'],)
    assert [e for e, _ in db.recent_entities('campp', 'voice', '2026-10-10T02:59:00+00:00')] == [first['entity_id']]
    detail = db.detail(first['entity_id'])
    assert detail['samples'][0]['observation']['text'] == 'Hello'


def test_media_refs_never_exceed_media_duration():
    t0 = datetime(2026, 10, 10, 3, 0, tzinfo=timezone.utc)
    spans = [{'begin': 0, 'end': int(5.0021 * SR), 't0': t0, 'media_id': 'm1', 'origin': 0., 'duration': 5.0016}]
    ref = media_refs(spans, int(4.5 * SR), int(5.0021 * SR))[0]
    assert ref['end_offset_seconds'] <= 5.0016 and ref['start_offset_seconds'] <= ref['end_offset_seconds']


def test_prune_empty_entities_keeps_entities_with_samples_or_names(tmp_path):
    db = Store(tmp_path / 'prune.sqlite')
    used = db.match_entity('campp', 'voice', np.array([1., 0., 0.]), '2026-10-10T03:00:00+00:00')['entity_id']
    db.record_speech([{'sample_id': 's1', 'entity_id': used, 'model': 'campp', 'media_id': 'm', 'capture_id': 'c', 'offset': 0.,
        'started_at': '2026-10-10T03:00:00+00:00', 'assignment': 'provisional_new', 'data': {'modality': 'voice', 'text': 'x'}}], [])
    empty = db.match_entity('campp', 'voice', np.array([0., 1., 0.]), '2026-10-10T03:00:00+00:00', excluded={used})['entity_id']
    named = db.match_entity('campp', 'voice', np.array([0., 0., 1.]), '2026-10-10T03:00:00+00:00', excluded={used, empty})['entity_id']
    db.rename(named, 'Morgan')
    assert db.prune_empty_entities('campp') == [empty]
    assert {e['id'] for e in db.list_entities()} == {used, named}


def test_select_turns_waits_for_a_turn_that_may_continue_and_skips_published_audio():
    turns = [{'speaker': 'a', 'start': 0, 'end': SR}, {'speaker': 'b', 'start': 2 * SR, 'end': 3 * SR},
             {'speaker': 'a', 'start': int(4.5 * SR), 'end': 5 * SR}]
    ready, hold = select_turns(turns, 0, 5 * SR, publish_all=False)
    assert [t['start'] for t in ready] == [0, 2 * SR] and hold == int(4.5 * SR)
    ready, hold = select_turns(turns, int(2.5 * SR), 5 * SR, publish_all=True)
    assert [t['start'] for t in ready] == [int(2.5 * SR), int(4.5 * SR)] and hold is None
    # A sliver left after an earlier analysis published most of a turn is not transcribed again.
    assert select_turns(turns, int(2.9 * SR), 5 * SR, publish_all=True)[0][0]['start'] == int(4.5 * SR)


class SyntheticSpeech:
    """Speech is wherever the signal is non-zero; one speaker; the transcript is the turn's padded length."""
    voice_model, asr_version = 'voice-test', 'asr-test'

    def regions(self, audio):
        on = np.flatnonzero(np.diff(np.r_[0, (audio != 0).astype(int), 0]))
        return list(zip(on[::2], on[1::2]))

    def embed(self, wave):
        return np.array([1., 0., 0.])

    def transcribe(self, waves, contexts):
        return [f'{len(w) / SR:.1f}s' for w in waves]


def test_each_clip_is_analysed_on_arrival_and_every_turn_is_published_once(tmp_path):
    t0 = datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc)
    speech = {0: [(4., 5.)], 1: [(0., 1.5), (2.5, 3.5)], 2: [(4.6, 5.)]}   # seconds within each 5 s clip
    clips = []
    for seq in range(3):
        audio = np.zeros(5 * SR, np.float32)
        for a, b in speech[seq]:
            audio[int(a * SR):int(b * SR)] = .1
        clips.append(({'media_id': f'm{seq}', 'capture_id': 'phone-1', 'clip_seq': seq, 'duration_seconds': 5.,
                       'captured_at': (t0 + timedelta(seconds=5 * seq)).isoformat()}, audio))
    posted = []
    def archive(request):
        event = json.loads(request.content); posted.append(event)
        return httpx.Response(200, json={'event_id': event['event_id'], 'persisted': True})
    store = Store(tmp_path / 'speech.sqlite')
    worker = SpeechWorker(store, 'http://archive.invalid', 'token', SyntheticSpeech())
    worker.archive = httpx.Client(base_url='http://archive.invalid', transport=httpx.MockTransport(archive))
    audio_of = {asset['media_id']: audio for asset, audio in clips}
    worker.load_audio = lambda asset, directory: (audio_of[asset['media_id']], 0.)
    def arrive(asset):
        arrived = (t0 + timedelta(seconds=5 * asset['clip_seq'] + 5)).timestamp()
        store.db.execute('INSERT INTO speech_pending VALUES (?,?,?,?,?)', (asset['media_id'], asset['capture_id'],
                         asset['clip_seq'], json.dumps(asset), arrived))
        store.db.commit()
        unit = worker.ready(arrived)
        assert unit['mode'] == 'stream' and [r['media_id'] for r in unit['new']] == [asset['media_id']]
        worker.step(unit)
        assert worker.ready(arrived) is None
        return [(e['body']['text'], e['started_at'][11:23]) for e in posted]

    # The turn running into the end of clip 0 waits for clip 1 instead of being cut at the clip boundary.
    assert arrive(clips[0][0]) == []
    assert arrive(clips[1][0]) == [('2.7s', '12:00:04.000'), ('1.2s', '12:00:07.500')]
    # Clip 1 is analysed again only as context: nothing from it is published twice.
    assert len(arrive(clips[2][0])) == 2
    assert [r['media_id'] for r in store.db.execute('SELECT media_id FROM speech_context')] == ['m2']
    # The last words go out once the capture has been idle for idle_seconds.
    last = (t0 + timedelta(seconds=15)).timestamp()
    assert worker.ready(last + 1) is None
    worker.step(worker.ready(last + worker.idle_seconds))
    assert [e['body']['text'] for e in posted] == ['2.7s', '1.2s', '0.5s']
    assert store.db.execute('SELECT count(*) FROM speech_context').fetchone()[0] == 0
    assert store.db.execute('SELECT count(*) FROM speech_progress').fetchone()[0] == 0
    assert store.db.execute('SELECT count(*) FROM outbox').fetchone()[0] == 0
    assert len({e['event_id'] for e in posted}) == 3
