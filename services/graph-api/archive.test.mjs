import test from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import { once } from 'node:events';
import { ArchiveClient, ArchiveError } from './archive.mjs';

const at = (seconds) => new Date(Date.UTC(2026, 0, 1) + seconds * 1000).toISOString().replace(/\.000Z$/, '.000000Z');
const us = (seconds) => Date.parse(at(seconds)) * 1000;
const token = 'private-archive-token';
const clip = (media_id, seconds = 0) => ({ media_id, capture_id: 'capture1', clip_seq: seconds / 10,
  captured_at: at(seconds), duration_seconds: 10, sha256: 'a'.repeat(64), size_bytes: 4096,
  archived_at: at(100), media_persisted: true, time_semantics: 'client_capture_wall_clock_plus_media_offset',
  metadata: { content_type: 'video/mp4', width: 1920, height: 1080, camera_facing: 'back',
    path: '/private/archive/movie.mp4', token, internal_url: 'http://internal-archive:8000',
    video_codec: '/private/codec', container: token },
  receipt: { local_path: '/private/movie.mp4', token }, original_url: 'http://internal-archive/original',
  path: '/private/movie.mp4' });
const storeFor = (properties) => ({ getEntity: async (id) => ({ id, properties }) });
async function fake(t, handler) {
  const requests = [];
  const server = http.createServer(async (req, res) => {
    requests.push({ method: req.method, url: req.url, headers: req.headers });
    try { await handler(req, res); } catch { if (!res.headersSent) res.writeHead(500); res.end(); }
  });
  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  t.after(() => { server.closeAllConnections(); server.close(); });
  const url = `http://127.0.0.1:${server.address().port}`;
  return { server, requests, url, client: new ArchiveClient({ url, token }) };
}
const json = (res, body, status = 200) => { res.writeHead(status, { 'Content-Type': 'application/json' }); res.end(JSON.stringify(body)); };
function sanitized(value) {
  const text = JSON.stringify(value);
  for (const secret of [token, '/private/', 'internal-archive', '127.0.0.1']) assert.ok(!text.includes(secret), text);
}

test('list maps archive filters, authentication and cursor; sanitizes payload and original URL', async (t) => {
  const { client, requests } = await fake(t, (_req, res) => json(res, { items: [clip('media_a')], has_more: true, next_cursor: 'media_a' }));
  const args = { capture_id: 'capture1', started_at: at(0), ended_at: at(30), limit: 3 };
  const first = await client.listVideoClips(args);
  assert.notEqual(first.next_cursor, 'media_a');
  const result = await client.listVideoClips({ ...args, cursor: first.next_cursor });
  const query = new URL(requests[1].url, 'http://fake').searchParams;
  assert.equal(query.get('after'), 'media_a');
  assert.equal(query.get('limit'), '3');
  assert.equal(query.get('capture_id'), 'capture1');
  assert.equal(query.get('started_at'), at(0));
  assert.equal(query.get('ended_at'), at(30));
  assert.equal(requests[0].headers.authorization, `Bearer ${token}`);
  assert.equal(requests[0].method, 'GET');
  assert.equal(result.next_cursor, first.next_cursor);
  assert.equal(result.items[0].original_url, '/v1/archive/media/media_a/original');
  assert.deepEqual(result.items[0].metadata, { width: 1920, height: 1080, camera_facing: 'back', content_type: 'video/mp4' });
  assert.equal(result.items[0].size_bytes, 4096);
  assert.equal(result.items[0].media_refs[0].kind, 'whole_clip');
  assert.equal(result.items[0].original_requires_auth, true);
  sanitized(result);
});

test('entity multi-clip IDs and crops are decoded, deduplicated and paginated including missing media', async (t) => {
  const { client, requests } = await fake(t, (req, res) => {
    const mediaId = req.url.split('/').at(-1);
    if (mediaId === 'media_b') return json(res, { path: '/private/not-found', token }, 404);
    json(res, clip(mediaId, mediaId === 'media_c' ? 10 : 0));
  });
  let calls = 0;
  const store = { getEntity: async (entityId) => {
    calls++; assert.equal(entityId, 'entity1');
    return { id: entityId, properties: { media_id: 'media_a', media_ids: '["media_a","media_b"]',
      offset_start: 2, crops: '[{"media_id":"media_c","offset":4,"path":"/private/crop"}]' } };
  }, listLinks: () => assert.fail('must not traverse graph') };
  const first = await client.listVideoClips({ entity_id: 'entity1', limit: 2 }, { store });
  assert.deepEqual(first.items.map((x) => x.media_id), ['media_a']);
  assert.deepEqual(first.missing_media_ids, ['media_b']);
  assert.equal(first.has_more, true);
  assert.deepEqual(first.items[0].media_refs, [{ media_id: 'media_a', kind: 'point', start_offset_seconds: 2, source: 'entity_offset' }]);
  const second = await client.listVideoClips({ entity_id: 'entity1', limit: 2, cursor: first.next_cursor }, { store });
  assert.equal(second.items[0].media_id, 'media_c');
  assert.equal(second.items[0].media_refs[0].start_offset_seconds, 4);
  assert.equal(second.has_more, false);
  assert.equal(second.next_cursor, null);
  assert.equal(requests.length, 3);
  assert.equal(calls, 2);
  sanitized([first, second]);
  await assert.rejects(client.listVideoClips({ entity_id: 'other', cursor: first.next_cursor }, { store: storeFor({ media_ids: ['media_a'] }) }), { code: 'invalid_archive_arguments' });
});

test('get validates direct entity membership without requesting unrelated media', async (t) => {
  const { client, requests } = await fake(t, (_req, res) => json(res, clip('media_a')));
  const store = storeFor({ media_ids: ['media_a'] });
  const result = await client.getVideoClip({ media_id: 'media_a', entity_id: 'e' }, { store });
  assert.equal(result.media_refs[0].kind, 'whole_clip');
  await assert.rejects(client.getVideoClip({ media_id: 'media_b', entity_id: 'e' }, { store }), { status: 404 });
  assert.equal(requests.length, 1);
});

test('capture interval evidence resolves multiple clips and clips offset ranges to each original', async (t) => {
  const { client, requests } = await fake(t, (req, res) => req.url.includes('/media?')
    ? json(res, { items: [clip('media_a'), clip('media_b', 10)], has_more: false })
    : json(res, clip(req.url.split('/').at(-1), req.url.endsWith('media_b') ? 10 : 0)));
  const store = storeFor({ kind: 'episode', capture_id: 'capture1', t_start_us: String(us(8)), t_end_us: String(us(13)) });
  const result = await client.listVideoClips({ entity_id: 'e' }, { store });
  const query = new URL(requests[0].url, 'http://fake').searchParams;
  assert.equal(query.get('started_at'), at(8));
  assert.equal(query.get('ended_at'), at(13));
  assert.deepEqual(result.items.map((x) => x.media_refs[0]), [
    { media_id: 'media_a', kind: 'interval', start_offset_seconds: 8, end_offset_seconds: 10, source: 'entity_capture_interval' },
    { media_id: 'media_b', kind: 'interval', start_offset_seconds: 0, end_offset_seconds: 3, source: 'entity_capture_interval' },
  ]);
  assert.equal((await client.getVideoClip({ media_id: 'media_b', entity_id: 'e' }, { store })).media_refs[0].end_offset_seconds, 3);
  await assert.rejects(client.getVideoClip({ media_id: 'media_a', entity_id: 'e' }, {
    store: storeFor({ capture_id: 'other', t_start_us: us(8), t_end_us: us(13) }),
  }), { code: 'entity_media_not_found' });
});

test('representative offset never fabricates exact end from event duration or start time', async (t) => {
  const { client } = await fake(t, (_req, res) => json(res, clip('media_a')));
  const result = await client.getVideoClip({ media_id: 'media_a', entity_id: 'e' }, {
    store: storeFor({ media_id: 'media_a', offset_start: '3', t_start_us: us(1), duration_s: 7 }),
  });
  assert.deepEqual(result.media_refs, [{ media_id: 'media_a', kind: 'point', start_offset_seconds: 3, source: 'entity_offset' }]);
});

test('sampled appearances preserve crop points and geometry, never their continuous envelope', async (t) => {
  const { client } = await fake(t, (_req, res) => json(res, clip('media_a')));
  for (const kind of ['face_appearance', 'object_appearance', 'text_appearance', 'unknown']) {
    const properties = { kind, media_id: 'media_a', offset_start: 2, capture_id: 'capture1',
      t_start_us: us(2), t_end_us: us(8), crops: [
        { media_id: 'media_a', offset: 2, box: [10, 20, 30, 40], w: 1920, h: 1080 },
        { media_id: 'media_a', offset: 8, box: [70, 80, 30, 40], w: 1920, h: 1080 },
        { media_id: 'media_a', offset: 9, box: [1900, 0, 40, 40], w: 1920, h: 1080 },
      ] };
    const store = storeFor(properties);
    const get = await client.getVideoClip({ media_id: 'media_a', entity_id: 'e' }, { store });
    const list = await client.listVideoClips({ entity_id: 'e' }, { store });
    assert.deepEqual(list.items[0].media_refs, get.media_refs);
    assert.deepEqual(get.media_refs.map((r) => r.kind), ['point', 'point', 'point', 'point']);
    assert.deepEqual(get.media_refs[1].box, [10, 20, 30, 40]);
    assert.equal(get.media_refs[1].w, 1920);
    assert.equal(get.media_refs[1].h, 1080);
    assert.deepEqual(get.media_refs[2].box, [70, 80, 30, 40]);
    assert.equal(get.media_refs[3].box, undefined);
  }
});

test('microsecond bounds, timezone offsets, original start and rounded durations remain precise', async (t) => {
  const original = { ...clip('micro'), captured_at: '2026-01-01T08:00:00.000500+08:00', duration_seconds: 0.0003996 };
  const { client, requests } = await fake(t, (req, res) => {
    if (req.url.includes('/media?')) {
      const q = new URL(req.url, 'http://fake').searchParams;
      // Reproduce the real archive upper-bound predicate (lexical UTC form).
      json(res, { items: q.get('ended_at') >= '2026-01-01T00:00:00.000500Z' ? [original] : [], has_more: false });
    } else json(res, original);
  });
  for (const kind of ['speech_segment', 'episode']) {
    const store = storeFor({ kind, capture_id: 'capture1', t_start_us: 1767225600000500, t_end_us: 1767225600000900 });
    const page = await client.listVideoClips({ entity_id: 'e' }, { store });
    assert.equal(page.items.length, 1);
    const query = new URL(requests.at(-1).url, 'http://fake').searchParams;
    assert.equal(query.get('started_at'), '2026-01-01T00:00:00.000500Z');
    assert.equal(query.get('ended_at'), '2026-01-01T00:00:00.000900Z');
    const get = await client.getVideoClip({ entity_id: 'e', media_id: 'micro' }, { store });
    assert.deepEqual(get.media_refs, [{ media_id: 'micro', kind: 'interval', start_offset_seconds: 0,
      end_offset_seconds: 0.0004, source: 'entity_capture_interval' }]);
  }
  await client.listVideoClips({ started_at: '2026-01-01T08:00:00.000900+08:00', ended_at: '2025-12-31T19:00:00.000901-05:00' });
  const query = new URL(requests.at(-1).url, 'http://fake').searchParams;
  assert.equal(query.get('started_at'), '2026-01-01T00:00:00.000900Z');
  assert.equal(query.get('ended_at'), '2026-01-01T00:00:00.000901Z');
  // Explicit-ID filtering must use rounded duration and microsecond comparisons too.
  assert.equal((await client.listVideoClips({ entity_id: 'e', started_at: '2026-01-01T00:00:00.000900Z' },
    { store: storeFor({ media_ids: ['micro'] }) })).items.length, 1);
  assert.equal((await client.listVideoClips({ entity_id: 'e', started_at: '2026-01-01T00:00:00.000901Z' },
    { store: storeFor({ media_ids: ['micro'] }) })).items.length, 0);
  for (const started_at of ['9999-01-01T00:00:00.000001Z', '2026-02-30T00:00:00Z', '2026-01-01T00:00:00.0000001Z']) {
    await assert.rejects(client.listVideoClips({ started_at }), { code: 'invalid_archive_arguments' });
  }
  await assert.rejects(client.listVideoClips({ entity_id: 'e' }, { store: storeFor({ t_start_us: Number.MAX_SAFE_INTEGER + 1 }) }), { status: 400 });
  client.request = async () => ({ ...original, duration_seconds: Number.MAX_SAFE_INTEGER });
  await assert.rejects(client.getVideoClip({ media_id: 'micro' }), { code: 'archive_invalid_response' });
});

test('archive-native cursors bind capture, filters, entity identity and evidence before upstream reads', async (t) => {
  const { client, requests } = await fake(t, (_req, res) => json(res, { items: [], has_more: true, next_cursor: 'a20' }));
  const args = { capture_id: 'capture1', started_at: at(0), ended_at: at(30) };
  const first = await client.listVideoClips(args);
  await client.listVideoClips({ ...args, cursor: first.next_cursor, limit: 1 });
  assert.equal(new URL(requests.at(-1).url, 'http://fake').searchParams.get('after'), 'a20');
  let count = requests.length;
  for (const changes of [{ capture_id: 'capture2' }, { started_at: at(1) }, { ended_at: at(29) }, { cursor: 'a20' }]) {
    await assert.rejects(client.listVideoClips({ ...args, cursor: first.next_cursor, ...changes }), { status: 400 });
  }
  assert.equal(requests.length, count);
  const properties = { kind: 'object_appearance', capture_id: 'capture1', t_start_us: us(0), t_end_us: us(30) };
  const store = storeFor(properties);
  const scoped = await client.listVideoClips({ entity_id: 'e' }, { store });
  await client.listVideoClips({ entity_id: 'e', cursor: scoped.next_cursor }, { store });
  count = requests.length;
  for (const [entity_id, p] of [['other', properties], ['e', { ...properties, t_end_us: us(29) }], ['e', { ...properties, kind: 'episode' }]]) {
    await assert.rejects(client.listVideoClips({ entity_id, cursor: scoped.next_cursor }, { store: storeFor(p) }), { status: 400 });
  }
  await assert.rejects(client.listVideoClips({ cursor: scoped.next_cursor }), { status: 400 });
  assert.equal(requests.length, count);
});

test('Unicode graph IDs are accepted but empty, oversized and control-containing IDs fail', async (t) => {
  const { client } = await fake(t, (_req, res) => json(res, clip('media_a')));
  const entityId = '人物/张三 空间 🐘';
  const store = { getEntity: async (value) => {
    assert.equal(value, entityId);
    return { id: value, properties: { media_id: 'media_a' } };
  } };
  assert.equal((await client.getVideoClip({ media_id: 'media_a', entity_id: entityId }, { store })).media_id, 'media_a');
  for (const invalidId of ['', 'x'.repeat(513), 'bad\nvalue', 'bad\u007fvalue']) {
    await assert.rejects(client.listVideoClips({ entity_id: invalidId }, { store }), { status: 400 });
  }
});

test('no direct evidence means empty list, no cluster/source or arbitrary link expansion', async (t) => {
  const { client, requests } = await fake(t, () => assert.fail('must not query archive'));
  const store = storeFor({ kind: 'face_cluster', source_id: 'sample1', episode_id: 'episode1', t_start_us: us(0), t_end_us: us(10) });
  const result = await client.listVideoClips({ entity_id: 'e' }, { store });
  assert.deepEqual(result.items, []);
  await assert.rejects(client.getVideoClip({ entity_id: 'e', media_id: 'media_a' }, { store }), { code: 'entity_media_not_found' });
  assert.equal(requests.length, 0);
});

test('filtered entity pages advance even when empty; bounds do not widen evidence', async (t) => {
  const { client, requests } = await fake(t, (req, res) => json(res, clip(req.url.split('/').at(-1))));
  const store = storeFor({ media_ids: ['media_a', 'media_b'] });
  const result = await client.listVideoClips({ entity_id: 'e', capture_id: 'other', limit: 1 }, { store });
  assert.equal(result.items.length, 0);
  assert.equal(result.has_more, true);
  const empty = await client.listVideoClips({ entity_id: 'e', started_at: at(50) }, {
    store: storeFor({ capture_id: 'capture1', t_start_us: us(0), t_end_us: us(10) }),
  });
  assert.equal(empty.items.length, 0);
  assert.equal(requests.length, 1);
});

test('invalid inputs bounded before archive calls; graph errors sanitized', async (t) => {
  const { client, requests } = await fake(t, () => assert.fail('unexpected archive read'));
  for (const args of [{ limit: 101 }, { limit: 0 }, { limit: 1.5 }, { cursor: 'x'.repeat(513) },
    { started_at: 'yesterday' }, { started_at: at(10), ended_at: at(1) }, { capture_id: 'bad/../path' }]) {
    await assert.rejects(client.listVideoClips(args), { status: 400 });
  }
  for (const media_id of ['..', '.', '../media_a', 'http://internal-archive']) await assert.rejects(client.getVideoClip({ media_id }), { status: 400 });
  await assert.rejects(client.listVideoClips({ entity_id: 'e' }, { store: storeFor({ media_ids: Array(10001).fill('media_a') }) }), { status: 400 });
  await assert.rejects(client.listVideoClips({ entity_id: 'e' }, { store: { getEntity: async () => { throw new Error(token + '/private/file'); } } }), (error) => {
    sanitized(error.message); return error.code === 'graph_lookup_failed';
  });
  assert.equal(requests.length, 0);
});

test('archive errors, malformed JSON, binary metadata and redirects are sanitized', async (t) => {
  const { client, requests } = await fake(t, (req, res) => {
    if (req.url.endsWith('missing')) return json(res, { detail: token + '/private/archive' }, 404);
    if (req.url.endsWith('redirect')) { res.writeHead(302, { Location: '/leak' }); res.end(); return; }
    if (req.url.endsWith('broken')) { res.writeHead(200, { 'Content-Type': 'application/json' }); res.end('{/private/' + token); return; }
    res.writeHead(200, { 'Content-Type': 'video/mp4' }); res.end('binary');
  });
  for (const media_id of ['missing', 'redirect', 'broken', 'binary']) {
    await assert.rejects(client.getVideoClip({ media_id }), (error) => {
      assert.ok(error instanceof ArchiveError); sanitized(error.message); return true;
    });
  }
  assert.equal(requests.length, 4);
  assert.ok(!requests.some((r) => r.url.endsWith('/original')));
});

test('oversized JSON response is bounded with and without Content-Length', async (t) => {
  const { client } = await fake(t, (req, res) => {
    res.writeHead(200, { 'Content-Type': 'application/json', ...(req.url.endsWith('length') ? { 'Content-Length': '2000000' } : {}) });
    res.end(' '.repeat(1024 * 1024 + 1));
  });
  for (const media_id of ['length', 'chunked']) await assert.rejects(client.getVideoClip({ media_id }), { code: 'archive_invalid_response' });
});

test('metadata fetch has bounded timeout', async (t) => {
  const { client } = await fake(t, () => {});
  const start = Date.now();
  await assert.rejects(client.getVideoClip({ media_id: 'media_a' }), { code: 'archive_request_failed' });
  assert.ok(Date.now() - start < 8000);
});

test('page deadline aborts in-flight metadata and never starts subsequent batches', async (t) => {
  let received = 0, aborted = 0;
  const timers = [];
  let resolveClosed;
  const closed = new Promise((resolve) => { resolveClosed = resolve; });
  const { client } = await fake(t, (req, res) => {
    received++;
    if (received <= 10) timers.push(setTimeout(() => json(res, clip(req.url.split('/').at(-1))), 3500));
    else res.on('close', () => { if (++aborted === 5) resolveClosed(); });
  });
  t.after(() => timers.forEach(clearTimeout));
  const start = Date.now();
  await assert.rejects(client.listVideoClips({ entity_id: 'e', limit: 100 }, {
    store: storeFor({ media_ids: Array.from({ length: 100 }, (_, i) => 'm' + String(i).padStart(3, '0')) }),
  }), { code: 'archive_request_failed' });
  await closed;
  assert.ok(Date.now() - start < 12000);
  assert.equal(received, 15);
  assert.equal(aborted, 5);
});

test('graph evidence wait observes caller cancellation and rejects pre-aborted get before lookup', async () => {
  const client = new ArchiveClient({ url: 'http://archive.invalid', token: 'synthetic-token' });
  let lookups = 0, requests = 0;
  const store = { getEntity() { lookups++; return new Promise(() => {}); } };
  client.request = async () => { requests++; throw new Error('Must not request metadata'); };
  const controller = new AbortController(); controller.abort();
  await assert.rejects(client.getVideoClip({ media_id: 'media_a', entity_id: 'e' }, { store, signal: controller.signal }));
  assert.equal(lookups, 0);
  const active = new AbortController();
  const waiting = client.listVideoClips({ entity_id: 'e' }, { store, signal: active.signal });
  active.abort();
  await assert.rejects(waiting);
  assert.equal(lookups, 1); assert.equal(requests, 0);
});

test('aggregate page deadline also bounds an unresolved graph evidence lookup', async () => {
  const client = new ArchiveClient({ url: 'http://archive.invalid', token: 'synthetic-token' });
  let requests = 0;
  client.request = async () => { requests++; throw new Error('Must not request metadata'); };
  const store = { getEntity: () => new Promise(() => {}) };
  const started = performance.now();
  await assert.rejects(client.listVideoClips({ entity_id: 'e' }, { store }), { code: 'archive_request_failed' });
  assert.ok(performance.now() - started < 15000); assert.equal(requests, 0);
});

test('caller cancellation aborts active batch and pre-aborted calls make no archive request', async (t) => {
  const controller = new AbortController();
  let received = 0, aborted = 0, resolveClosed;
  const closed = new Promise((resolve) => { resolveClosed = resolve; });
  const { client, requests } = await fake(t, (_req, res) => {
    res.on('close', () => { if (++aborted === 5) resolveClosed(); });
    if (++received === 5) controller.abort();
  });
  const store = storeFor({ media_ids: Array.from({ length: 20 }, (_, i) => 'm' + i) });
  await assert.rejects(client.listVideoClips({ entity_id: 'e', limit: 20 }, { store, signal: controller.signal }), { code: 'archive_request_failed' });
  await closed;
  assert.equal(requests.length, 5);
  await assert.rejects(client.listVideoClips({}, { signal: controller.signal }), { code: 'archive_request_failed' });
  await assert.rejects(client.getVideoClip({ media_id: 'media_a' }, { signal: controller.signal }), { code: 'archive_request_failed' });
  assert.equal(requests.length, 5);
});

async function proxy(t, client) {
  const server = http.createServer((req, res) => client.proxyOriginal(req, res, req.url.slice(1)));
  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  t.after(() => { server.closeAllConnections(); server.close(); });
  return `http://127.0.0.1:${server.address().port}`;
}

test('proxy streams original bytes, Range/If-Range, safe headers; does not forward caller credentials', async (t) => {
  const sha = 'a'.repeat(64);
  const { client, requests } = await fake(t, (_req, res) => {
    res.writeHead(206, { 'Content-Type': 'video/mp4', 'Content-Length': 4, 'Content-Range': 'bytes 2-5/10',
      'Accept-Ranges': 'bytes', ETag: `"${sha}"`, 'X-Archive-SHA256': sha, 'X-Archive-Media-ID': 'media_a',
      'Content-Disposition': 'attachment; filename="/private/movie.mp4"', 'X-Internal-Token': token });
    res.write(Buffer.from([2, 3])); res.end(Buffer.from([4, 5]));
  });
  const publicUrl = await proxy(t, client);
  const result = await fetch(publicUrl + '/media_a', { headers: { Range: 'bytes=2-5', 'If-Range': `"${sha}"`, Authorization: 'Bearer user-token', Cookie: 'private=user' } });
  assert.equal(result.status, 206);
  assert.deepEqual([...new Uint8Array(await result.arrayBuffer())], [2, 3, 4, 5]);
  assert.equal(result.headers.get('content-range'), 'bytes 2-5/10');
  assert.equal(result.headers.get('content-length'), '4');
  assert.equal(result.headers.get('etag'), `"${sha}"`);
  assert.equal(result.headers.get('x-archive-sha256'), sha);
  assert.equal(result.headers.get('content-disposition'), null);
  assert.equal(result.headers.get('x-internal-token'), null);
  assert.equal(requests[0].headers.authorization, `Bearer ${token}`);
  assert.equal(requests[0].headers.cookie, undefined);
  assert.equal(requests[0].headers.range, 'bytes=2-5');
  assert.equal(requests[0].headers['if-range'], `"${sha}"`);
  assert.equal(requests[0].url, '/v1/archive/media/media_a/original');
});

test('proxy HEAD preserves length without body and rejects unsafe methods, ranges and identifiers', async (t) => {
  const { client, requests } = await fake(t, (_req, res) => {
    res.writeHead(200, { 'Content-Type': 'video/quicktime', 'Content-Length': 100 }); res.end();
  });
  const publicUrl = await proxy(t, client);
  const head = await fetch(publicUrl + '/media_a', { method: 'HEAD' });
  assert.equal(head.status, 200); assert.equal(head.headers.get('content-length'), '100'); assert.equal(await head.text(), '');
  assert.equal(requests[0].method, 'HEAD');
  assert.equal((await fetch(publicUrl + '/media_a', { method: 'POST' })).status, 405);
  assert.equal((await fetch(publicUrl + '/media_a', { headers: { Range: 'bytes=0-1,3-4' } })).status, 400);
  assert.equal((await fetch(publicUrl + '/media_a', { headers: { 'If-Range': 'http://internal-archive' } })).status, 400);
  assert.equal((await fetch(publicUrl + '/bad%2Fpath')).status, 400);
  assert.equal(requests.length, 1);
});

test('proxy errors and redirect response bodies are never exposed; 416 keeps only range', async (t) => {
  const { client, requests } = await fake(t, (req, res) => {
    if (req.url.includes('redirect')) res.writeHead(302, { Location: '/private/leak' });
    else if (req.url.includes('unsatisfied')) res.writeHead(416, { 'Content-Range': 'bytes */10', 'Content-Type': 'text/html' });
    else res.writeHead(404);
    res.end('/private/file ' + token);
  });
  const publicUrl = await proxy(t, client);
  for (const [mediaId, expected] of [['missing', 404], ['redirect', 502], ['unsatisfied', 416]]) {
    const result = await fetch(publicUrl + '/' + mediaId);
    assert.equal(result.status, expected);
    sanitized(await result.text());
    if (expected === 416) assert.equal(result.headers.get('content-range'), 'bytes */10');
  }
  assert.equal(requests.length, 3);
});

test('publicBaseUrl uses public graph origin; config errors never expose credentials', async (t) => {
  const { url } = await fake(t, (_req, res) => json(res, clip('media_a')));
  const client = new ArchiveClient({ url, token, publicBaseUrl: 'https://graph.example' });
  assert.equal((await client.getVideoClip({ media_id: 'media_a' })).original_url, 'https://graph.example/v1/archive/media/media_a/original');
  for (const config of [{ url, token, publicBaseUrl: url }, { url: 'http://user:secret@internal-archive', token }, { url, token: '' }]) {
    assert.throws(() => new ArchiveClient(config), (error) => { sanitized(error.message); return error.status === 503; });
  }
});

test('frames by capture time resolve the covering clip and offset; by media id use the offsets given', async (t) => {
  const { client, requests } = await fake(t, (req, res) => {
    const url = new URL(req.url, 'http://fake');
    if (url.pathname === '/v1/archive/media') return json(res, { items: [clip('media_a', 0), clip('media_b', 10)], has_more: false, next_cursor: null });
    if (url.pathname.endsWith('/frame')) { res.writeHead(200, { 'Content-Type': 'image/jpeg' }); return res.end(Buffer.from([0xff, 0xd8, 0xff, 0xd9])); }
    res.writeHead(404); res.end();
  });
  const frames = await client.getFrames({ capture_id: 'capture1', at: [at(12.5)] });
  assert.equal(frames[0].media_id, 'media_b');
  assert.equal(frames[0].offset_s, 2.5);
  const frameRequest = requests.at(-1), query = new URL(frameRequest.url, 'http://fake').searchParams;
  assert.equal(new URL(frameRequest.url, 'http://fake').pathname, '/v1/archive/media/media_b/frame');
  assert.equal(query.get('t'), '2.500');
  assert.equal(query.get('size'), '768');
  assert.equal(frameRequest.headers.authorization, `Bearer ${token}`);
  assert.deepEqual([...frames[0].jpeg], [0xff, 0xd8, 0xff, 0xd9]);
  const [last] = await client.getFrames({ capture_id: 'capture1', at: [at(19.95)] });
  assert.deepEqual([last.media_id, last.offset_s], ['media_b', 9.75]);
  const direct = await client.getFrames({ media_id: 'media_a', offsets: [1, 4.25], size: 320 });
  assert.deepEqual(direct.map((f) => [f.media_id, f.offset_s, f.size]), [['media_a', 1, 320], ['media_a', 4.25, 320]]);
  await assert.rejects(client.getFrames({ capture_id: 'capture1', at: [at(500)] }), (e) => e instanceof ArchiveError && e.status === 404);
  await assert.rejects(client.getFrames({ media_id: 'media_a' }), (e) => e instanceof ArchiveError && e.status === 400);
  await assert.rejects(client.getFrames({ media_id: 'media_a', offsets: [1], capture_id: 'capture1' }), (e) => e.status === 400);
});
