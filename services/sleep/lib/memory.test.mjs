import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { Memory, pairKey } from './memory.mjs';
import { readTimeline } from './timeline.mjs';
import { zonedDateTime } from './util.mjs';

const at = (seconds) => new Date(Date.parse('2026-10-14T12:00:00Z') + seconds * 1000).toISOString();
const event = (id, seconds, kind = 'minicpm', capture = 'capture-1') => ({ event_id: id, kind, capture_id: capture,
  started_at: at(seconds), ended_at: at(seconds + 5), body: kind === 'minicpm' ? { generated_text: 'A device plays a tutorial' } : { text: 'Please open Settings' } });
const episode = (id, start, end) => ({ id, kind: 'episode', capture_id: 'capture-1', t_start: at(start), t_end_us: Date.parse(at(end)) * 1000,
  duration_s: end - start, observations: ['old unanchored summary'] });

function memory(t, nodes = {}, edges = {}) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'sleep-memory-test-'));
  fs.writeFileSync(path.join(dir, 'working.json'), JSON.stringify({ nodes, edges, day: '2026-10-14', captures: ['capture-1'] }));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  return new Memory(dir);
}

function archive(t, handler) {
  const prior = { url: process.env.ARCHIVE_URL, token: process.env.SCENE_ARCHIVE_TOKEN, fetch: globalThis.fetch };
  process.env.ARCHIVE_URL = 'http://archive.test'; process.env.SCENE_ARCHIVE_TOKEN = 'synthetic-test-token';
  globalThis.fetch = async (url) => ({ ok: true, json: async () => handler(new URL(url)) });
  t.after(() => {
    globalThis.fetch = prior.fetch;
    for (const [key, value] of [['ARCHIVE_URL', prior.url], ['SCENE_ARCHIVE_TOKEN', prior.token]]) {
      if (value === undefined) delete process.env[key]; else process.env[key] = value;
    }
  });
}

test('archive pagination retains timestamp ties and obeys the bounded capture query', async (t) => {
  const requests = [];
  archive(t, (url) => {
    requests.push(url);
    if (url.searchParams.get('kind') === 'speech') return { items: [event('s1', 5, 'speech')], has_more: false };
    return url.searchParams.has('after') ? { items: [event('m2', 0), event('m3', 10)], has_more: false }
      : { items: [event('m1', 0)], has_more: true, next_cursor: 'm1' };
  });
  assert.deepEqual((await readTimeline('capture-1', Date.parse(at(0)), Date.parse(at(15)))).map((e) => e.event_id), ['m1', 'm2', 's1', 'm3']);
  for (const url of requests) {
    assert.equal(url.searchParams.get('capture_id'), 'capture-1');
    assert.equal(url.searchParams.get('started_at'), at(0));
    assert.equal(url.searchParams.get('ended_at'), at(15));
  }
  assert.equal(requests.filter((url) => url.searchParams.get('after') === 'm1').length, 1);
});

test('non-advancing archive cursor fails instead of accepting an incomplete timeline', async (t) => {
  archive(t, (url) => url.searchParams.get('kind') === 'speech' ? { items: [], has_more: false }
    : { items: [event('m1', 0)], has_more: true, next_cursor: 'm1' });
  await assert.rejects(readTimeline('capture-1', Date.parse(at(0)), Date.parse(at(10))), /pagination did not advance/);
});

test('semantic event crosses physical episodes while preserving sources and only attaching overlapping clips', async (t) => {
  const ep1 = episode('episode:a', 0, 10), ep2 = episode('episode:b', 10, 20);
  const conv = { id: 'conversation:x', kind: 'conversation', layer: 'memory', t_start: at(0), t_last: at(20), transcript: [{ t: at(5), text: 'Open Settings', speaker: 'unknown' }] };
  const m = memory(t, { [ep1.id]: ep1, [ep2.id]: ep2, [conv.id]: conv });
  archive(t, (url) => {
    if (url.pathname.endsWith('/media')) return { items: [-5, 0, 5, 10, 15, 20].map((s) => ({ media_id: `clip-${s}`, capture_id: 'capture-1', captured_at: at(s), duration_seconds: 5 })), has_more: false };
    if (url.searchParams.get('kind') === 'speech') return { items: [event('speech', 5, 'speech')], has_more: false };
    return { items: [event('before', 0), event('first', 5), event('last', 10), event('after', 15)], has_more: false };
  });
  const viewed = await m.view_episode({ id: ep1.id });
  assert.match(viewed.text, /first \| 05:00:05–05:00:10/);
  assert.doesNotMatch(viewed.text, /- old unanchored summary/);
  await m.view_episode({ id: ep2.id });
  await m.remember_event({ first: 'first', last: 'last', title: 'Watching a phone tutorial', description: 'The topic changes to phone settings within the inspected windows.',
    evidence: [{ id: conv.id, text: 'The transcript contains the device tutorial.' }] });
  const saved = [...m.nodes.values()].find((n) => n.kind === 'event');
  assert.equal(saved.t_start, at(5)); assert.equal(saved.t_last, at(15));
  assert.deepEqual(saved.media_ids, ['clip-5', 'clip-10']);
  assert.deepEqual(m.nodes.get(ep1.id), ep1); assert.deepEqual(m.nodes.get(ep2.id), ep2);
  assert.deepEqual(m.nodes.get(conv.id), conv);
  assert.ok(m.edges.has(pairKey(saved.id, conv.id)));
  // New event records use only properties already used by existing memory entities.
  assert.deepEqual(Object.keys(saved).sort(), ['id', 'kind', 'layer', 'name', 'description', 'summary', 'text', 't_start', 't_last',
    't_start_us', 't_end_us', 'media_ids', 'occurrences', 'created_by', 'nights', '_touched'].sort());
  await assert.rejects(m.remember_event({ first: 'first', last: 'last', title: 'Duplicate', description: 'Duplicate' }), /already covers/);
  await m.write_diary({ title: 'Today', text: 'Watched a phone tutorial.', refs: [saved.id] });
  await m.finish();
  const result = JSON.parse(fs.readFileSync(path.join(m.dir, 'result.json'), 'utf8'));
  const persisted = result.upserts.find((n) => n.id === saved.id);
  assert.deepEqual(Object.keys(persisted).sort(), ['id', 'kind', 'layer', 'name', 'description', 'summary', 'text', 't_start', 't_last',
    't_start_us', 't_end_us', 'media_ids', 'occurrences', 'created_by', 'nights'].sort());
  assert.doesNotMatch(JSON.stringify(result), /"(?:timeline|first|last|event_id|started_at|ended_at)"/);
});

test('invalid or unavailable event evidence cannot mutate memory or consume an episode', async (t) => {
  const ep = episode('episode:a', 0, 10), m = memory(t, { [ep.id]: ep });
  const before = JSON.stringify([...m.nodes]);
  await assert.rejects(m.remember_event({ first: 'unseen', last: 'unseen', title: 'Test', description: 'Test' }), /Inspect/);
  await assert.rejects(m.consolidate({ ids: [ep.id], new: { kind: 'event', name: 'Test', description: 'Test' }, why: 'test' }), /remember_event/);
  m.timeline.set('a', event('a', 5)); m.timeline.set('b', event('b', 0)); m.timeline.set('c', event('c', 10, 'minicpm', 'other-capture'));
  for (const last of ['b', 'c']) await assert.rejects(m.remember_event({ first: 'a', last, title: 'Test', description: 'Test' }), /ordered windows/);
  archive(t, () => { throw new Error('offline'); });
  await assert.rejects(m.remember_event({ first: 'a', last: 'a', title: 'Test', description: 'Test' }), /offline/);
  assert.equal(JSON.stringify([...m.nodes]), before);
  const view = await m.view_episode({ id: ep.id });
  assert.match(view.text, /no per-observation timestamps/);
  assert.match(view.text, /old unanchored summary/);
});

test('merging events preserves the latest evidence end rather than the latest start', async (t) => {
  const a = { id: 'event:a', kind: 'event', layer: 'memory', name: 'Activity', t_start: at(0), t_last: at(90), occurrences: 0 };
  const b = { id: 'event:b', kind: 'event', layer: 'memory', name: 'Activity', t_start: at(30), t_last: at(60), occurrences: 0 };
  const m = memory(t, { [a.id]: a, [b.id]: b });
  await m.consolidate({ ids: [b.id], into: a.id, why: 'same occurrence' });
  assert.equal(m.nodes.get(a.id).t_start, at(0)); assert.equal(m.nodes.get(a.id).t_last, at(90));
  assert.equal(m.nodes.get(a.id).t_end_us, Date.parse(at(90)) * 1000);
  assert.equal(m.nodes.has(b.id), false);
});

test('memory evidence does not turn model references or missing dates into a use history', async (t) => {
  const entity = { id: 'object:a', kind: 'object', layer: 'memory', name: 'Object', occurrences: 5 };
  const diary = { id: 'diary:20261014', kind: 'diary', layer: 'memory', name: 'Diary' };
  const m = memory(t, { [entity.id]: entity, [diary.id]: diary });
  const before = m.evidenceLine(entity);
  await m.link({ a: diary.id, b: entity.id, text: 'The diary mentions the object.' });
  assert.equal(m.evidenceLine(entity), before);
  const view = await m.recall_day();
  assert.match(view.text, /source material count 5; evidence time envelope unknown – unknown/);
  assert.match(view.text, /not recall histories/);
  assert.equal(m.nodes.get(entity.id).occurrences, 5);
});

test('recall_day places memories on a day only by evidence time, never by the night that created them', async (t) => {
  const local = (day, hour, minute = 0) => zonedDateTime(day, hour, minute).toISOString();
  const lastNight = local('2026-10-14', 0, 38);
  const previous = { id: 'event:previous', kind: 'event', layer: 'memory', name: 'Previous standup', t_start: local('2026-10-13', 10, 41), t_last: local('2026-10-13', 10, 51), created_at: lastNight };
  const previousTalk = { id: 'conversation:previous', kind: 'conversation', layer: 'memory', name: 'Previous talk', t_start: local('2026-10-13', 10, 42), t_last: local('2026-10-13', 10, 49), created_at: lastNight };
  const today = { id: 'event:today', kind: 'event', layer: 'memory', name: 'Evening walk', t_start: local('2026-10-14', 19, 47), t_last: local('2026-10-14', 20, 3), created_at: local('2026-10-15', 0, 53) };
  const untimed = { id: 'object:untimed', kind: 'object', layer: 'memory', name: 'Untimed object', created_at: local('2026-10-14', 9) };
  const m = memory(t, { [previous.id]: previous, [previousTalk.id]: previousTalk, [today.id]: today, [untimed.id]: untimed });
  const { text } = await m.recall_day();
  const section = (title) => text.split(`## ${title}`)[1].split('\n## ')[0];
  assert.match(text, /This day already has 1 memory entities and 0 conversations/);
  assert.match(section('Memory entities of the day'), /event:today/);
  for (const id of ['event:previous', 'object:untimed']) {
    assert.doesNotMatch(section('Memory entities of the day'), new RegExp(id));
    assert.match(section('All long-term memory entities'), new RegExp(id));
  }
  assert.doesNotMatch(section("The day's conversations"), /conversation:previous/);
});

const owned = (id, name, extra = {}) => ({ id, kind: 'object', layer: 'memory', name, summary: name, name_source: 'user', created_by: 'user-naming', occurrences: 1, ...extra });
const plain = (id, name, extra = {}) => ({ id, kind: 'object', layer: 'memory', name, summary: name, description: `${name} seen on the desk`, occurrences: 2, ...extra });

test('revise keeps an owner-given name and still updates the description', async (t) => {
  const mug = owned('object:mug', 'Blue mug'), other = plain('object:cup', 'Cup');
  const m = memory(t, { [mug.id]: mug, [other.id]: other });
  const refused = await m.revise({ id: mug.id, name: 'Ceramic cup' });
  assert.match(refused.text, /Kept the name "Blue mug": the owner named object:mug, so only its description can be revised/);
  assert.equal(m.nodes.get(mug.id).name, 'Blue mug');
  assert.equal(m.nodes.get(mug.id).summary, 'Blue mug');
  assert.equal(m.nodes.get(mug.id)._touched, undefined);
  const both = await m.revise({ id: mug.id, name: 'Ceramic cup', description: 'A blue ceramic mug with a chipped rim' });
  assert.match(both.text, /^Updated the description\. Kept the name "Blue mug"/);
  assert.equal(m.nodes.get(mug.id).name, 'Blue mug');
  assert.equal(m.nodes.get(mug.id).description, 'A blue ceramic mug with a chipped rim');
  assert.equal((await m.revise({ id: mug.id, name: 'Blue mug', description: 'Same name is fine' })).text, 'Updated.');
  // Memories the owner did not name are revised as before.
  assert.equal((await m.revise({ id: other.id, name: 'Paper cup' })).text, 'Updated.');
  assert.equal(m.nodes.get(other.id).name, 'Paper cup');
});

test('forget never removes an owner-named memory', async (t) => {
  const mug = owned('object:mug', 'Blue mug'), other = plain('object:cup', 'Cup');
  const m = memory(t, { [mug.id]: mug, [other.id]: other });
  const result = await m.forget({ ids: [mug.id, other.id], why: 'clutter' });
  assert.equal(result.text, 'Forgot 1 fragment(s). Kept object:mug "Blue mug": named by the owner, so it cannot be forgotten.');
  assert.ok(m.nodes.has(mug.id));
  assert.equal(m.nodes.has(other.id), false);
});

test('consolidate keeps the owner-named memory as the survivor with its name', async (t) => {
  const link = (a, b, text) => ({ [pairKey(a, b)]: { a, b, text } });
  const person = { id: 'person:ann', kind: 'person', layer: 'memory', name: 'Ann' };
  const mug = owned('object:mug', 'Blue mug', { object_vec: [1, 0], object_vec_n: 1 });
  const cup = plain('object:cup', 'Ceramic cup', { object_vec: [0, 1], object_vec_n: 1 });
  const m = memory(t, { [mug.id]: mug, [cup.id]: cup, [person.id]: person }, link(cup.id, person.id, 'Ann drinks from the cup.'));
  // Asked to merge the owner-named mug into another memory: the other memory merges into the mug instead.
  const result = await m.consolidate({ ids: [mug.id], into: cup.id, why: 'same object' });
  assert.match(result.text, /^Consolidated 1 fragment\(s\) into object:mug \(Blue mug\)/);
  assert.match(result.text, /object:mug "Blue mug" was named by the owner, so it is the surviving memory and keeps its name; object:cup merged into it\./);
  assert.equal(m.nodes.has(cup.id), false);
  const kept = m.nodes.get(mug.id);
  assert.equal(kept.name, 'Blue mug');
  assert.equal(kept.name_source, 'user');
  // The surviving memory takes over the merged one's description (it had none), prototype and links.
  assert.equal(kept.description, 'Ceramic cup seen on the desk');
  assert.equal(kept.object_vec_n, 2);
  assert.ok(m.edges.has(pairKey(mug.id, person.id)));
  assert.deepEqual(m.mergedInto, { [cup.id]: mug.id });

  // A request for a new memory folds the others into the owner-named one instead.
  const lid = plain('object:lid', 'Lid');
  m.nodes.set(lid.id, lid);
  const fresh = await m.consolidate({ ids: [lid.id, mug.id], new: { kind: 'object', name: 'Mug set', description: 'A mug and its lid' }, why: 'one thing' });
  assert.match(fresh.text, /into object:mug \(Blue mug\).*no new memory was created\./);
  assert.equal(m.nodes.has(lid.id), false);
  assert.equal([...m.nodes.values()].filter((n) => n.name === 'Mug set').length, 0);
  assert.equal(m.nodes.get(mug.id).name, 'Blue mug');

  // Merging into the owner-named memory directly works as before.
  const saucer = plain('object:saucer', 'Saucer');
  m.nodes.set(saucer.id, saucer);
  await m.consolidate({ ids: [saucer.id], into: mug.id, why: 'same object' });
  assert.equal(m.nodes.has(saucer.id), false);
  assert.equal(m.nodes.get(mug.id).name, 'Blue mug');
});

test('consolidate refuses to merge two owner-named memories or to fold one into the owner', async (t) => {
  const mug = owned('object:mug', 'Blue mug'), watch = owned('object:watch', 'Grandpa watch'), cup = plain('object:cup', 'Cup');
  const owner = { id: 'person:owner', kind: 'person', layer: 'memory', name: 'Owner' };
  const m = memory(t, { [mug.id]: mug, [watch.id]: watch, [cup.id]: cup, [owner.id]: owner });
  const before = JSON.stringify([...m.nodes]);
  await assert.rejects(m.consolidate({ ids: [mug.id], into: watch.id, why: 'test' }), /object:watch and object:mug were each named by the owner/);
  await assert.rejects(m.consolidate({ ids: [mug.id, watch.id, cup.id], new: { kind: 'object', name: 'Things', description: 'Things' }, why: 'test' }), /each named by the owner/);
  await assert.rejects(m.consolidate({ ids: [mug.id], into: owner.id, why: 'test' }), /cannot be merged into person:owner/);
  await assert.rejects(m.consolidate({ ids: [mug.id], new: { kind: 'object', name: 'Cup', description: 'Renamed' }, why: 'test' }), /keeps its name; nothing else was given/);
  assert.equal(JSON.stringify([...m.nodes]), before);
});

test('recall_day and view_memory mark owner-named memories', async (t) => {
  const mug = owned('object:mug', 'Blue mug'), cup = plain('object:cup', 'Cup');
  const m = memory(t, { [mug.id]: mug, [cup.id]: cup });
  const { text } = await m.recall_day();
  assert.match(text, /object:mug \| object \| Blue mug \[named by the owner\] \|/);
  assert.match(text, /object:cup \| object \| Cup \|/);
  assert.match((await m.view_memory({ id: mug.id })).text, /^object:mug \| object \| Blue mug \[named by the owner\]/);
});
