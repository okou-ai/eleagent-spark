import assert from 'node:assert/strict';
import { once } from 'node:events';
import { after, before, test } from 'node:test';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StreamableHTTPClientTransport } from '@modelcontextprotocol/sdk/client/streamableHttp.js';
import { createApiServer } from './server.mjs';
import { ApiError } from './graph.mjs';
import { resourceAuth } from './test-resource-auth.mjs';

// Graph/protocol fixtures only; real OAuth is tested in auth.test.mjs and oauth-integration.test.mjs.
const token = 'mcp-test-token-not-for-production-123456789';
// Separate private signing fixture: production cursor keys are never exposed as auth credentials.
const cursorKey = 'mcp-test-private-cursor-key-123456789012345';
let server, origin, client, transport;
const calls = [];
const store = {
  async listEntities(args) {
    calls.push(['search', args]);
    return { entities: [{ id: args.after?.id || 'entity:one', properties: { text: 'Synthetic evidence' } }],
      after: args.after ? null : { id: 'entity:two', rank: 0, value: { type: 'null' } } };
  },
  async getEntity(id) {
    if (id === 'internal-error') throw new Error('secret driver message');
    if (id === 'missing') throw new ApiError(404, 'not_found', 'Entity not found');
    return { id, properties: { arbitrary: true } };
  },
  async listLinks(id, args) { calls.push(['links', args]); return { entities: [{ id, properties: {} }], links: [], after: null }; },
  async subgraph(args) { return { ...args, entities: [], links: [], truncated: true }; },
};
const archive = {
  async listVideoClips(args, deps) { assert.equal(deps.store, store); return { media: [], next_cursor: null, filters: args }; },
  async getVideoClip(args, deps) { assert.equal(deps.store, store); return { media_id: args.media_id, original_url: `/v1/archive/media/${args.media_id}/original` }; },
  async proxyOriginal(req, res) { res.writeHead(206, { 'Content-Type': 'video/mp4', 'Content-Range': 'bytes 0-2/3' }); res.end('abc'); },
  async getFrames(args, deps) { assert.equal(deps.store, store); return [{ media_id: args.media_id, offset_s: args.offsets[0], size: args.size, jpeg: Buffer.from('jpeg-bytes') }]; },
};
before(async () => {
  server = createApiServer({ store, cursorKey, archive, ...resourceAuth(token) });
  server.listen(0, '127.0.0.1'); await once(server, 'listening');
  origin = `http://127.0.0.1:${server.address().port}`;
  client = new Client({ name: 'eleagent-mcp-test', version: '1.0.0' });
  transport = new StreamableHTTPClientTransport(new URL(`${origin}/mcp`), { requestInit: { headers: { Authorization: `Bearer ${token}` } } });
  await client.connect(transport);
});
after(async () => { await client?.close(); await new Promise((resolve) => server.close(resolve)); });
const call = (name, args = {}) => client.callTool({ name, arguments: args });
const json = (r) => JSON.parse(r.content[0].text);

test('standard MCP SDK initializes and discovers exactly seven strictly read-only tools', async () => {
  const result = await client.listTools();
  assert.deepEqual(result.tools.map((x) => x.name), ['search_entities', 'get_entity', 'list_entity_links', 'get_subgraph', 'list_video_clips', 'get_video_clip', 'get_video_frames']);
  for (const tool of result.tools) { assert.equal(tool.annotations.readOnlyHint, true); assert.equal(tool.annotations.destructiveHint, false); assert.equal(tool.inputSchema.additionalProperties, false); }
});
test('graph tools reuse signed pagination bound to query and retain structured graph data', async () => {
  const first = json(await call('search_entities', { search: 'test', filter: { kind: 'event' }, limit: 1 }));
  assert.ok(first.next_cursor);
  const second = json(await call('search_entities', { search: 'test', filter: { kind: 'event' }, limit: 1, cursor: first.next_cursor }));
  assert.equal(second.entities[0].id, 'entity:two'); assert.equal(second.next_cursor, null);
  assert.equal((await call('search_entities', { search: 'different', cursor: first.next_cursor })).isError, true);
  assert.deepEqual(json(await call('get_entity', { id: 'entity:one' })), { id: 'entity:one', properties: { arbitrary: true } });
  assert.deepEqual(json(await call('list_entity_links', { id: 'entity:one' })).links, []);
  assert.equal(json(await call('get_subgraph', { ids: ['entity:one'], depth: 1 })).truncated, true);
});
test('invalid limits, fields, seeds, and unknown tools cannot reach backend operations', async () => {
  const n = calls.length;
  for (const args of [{ limit: 201 }, { query: 'MATCH (n) DELETE n' }, { filter: { x: {} } }, { filter: { x: 9007199254740992 } }]) {
    assert.equal((await call('search_entities', args)).isError, true);
  }
  assert.equal(calls.length, n);
  assert.equal((await call('get_subgraph', { ids: ['one', 'two'], max_entities: 1 })).isError, true);
  assert.equal((await call('execute_cypher', { query: 'RETURN 1' })).isError, true);
});
test('video tools provide same-origin original reference; invalid interval is rejected', async () => {
  assert.deepEqual(json(await call('list_video_clips', { entity_id: 'episode:one' })).media, []);
  const precise = await call('list_video_clips', { started_at: '2026-10-16T12:00:00.000500Z', ended_at: '2026-10-16T12:00:00.000900Z' });
  assert.ok(!precise.isError, 'Valid submillisecond evidence must not be rounded to an empty interval');
  const id = `media_${'a'.repeat(32)}`;
  const result = json(await call('get_video_clip', { media_id: id }));
  assert.equal(result.original_url, `/v1/archive/media/${id}/original`);
  assert.equal((await call('get_video_clip', { media_id: '../../etc/passwd' })).isError, true);
  assert.equal((await call('list_video_clips', { started_at: '2026-10-16T12:00:00Z', ended_at: '2026-10-16T11:00:00Z' })).isError, true);
});
test('tool errors do not expose driver messages', async () => {
  const result = await call('get_entity', { id: 'internal-error' });
  assert.equal(result.isError, true); assert.ok(!JSON.stringify(result).includes('secret driver'));
  assert.equal(json(await call('get_entity', { id: 'missing' })).error.code, 'not_found');
});
test('resource boundary requires fixture bearer for MCP, originals and REST; Origin remains guarded', async () => {
  const original = `/v1/archive/media/media_${'a'.repeat(32)}/original`;
  const denied = await fetch(`${origin}/mcp`, { method: 'POST', headers: { Origin: 'https://evil.invalid', 'Content-Type': 'application/json' }, body: '{}' });
  assert.equal(denied.status, 403);
  for (const path of ['/mcp', original, '/v1/graph/entities']) {
    for (const authorization of [undefined, 'Bearer incorrect', `Bearer ${cursorKey}`, `Basic ${token}`]) {
      const response = await fetch(origin + path, { headers: authorization ? { Authorization: authorization } : {} });
      assert.equal(response.status, 401);
    }
  }
  const video = await fetch(origin + original, { headers: { Authorization: `Bearer ${token}`, Range: 'bytes=0-2' } });
  assert.equal(video.status, 206); assert.equal(await video.text(), 'abc');
  assert.equal((await fetch(origin + '/v1/graph/entities')).status, 401);
});
test('body bounds and legacy REST graph endpoint remain intact', async () => {
  const response = await fetch(`${origin}/mcp`, { method: 'POST', headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' }, body: JSON.stringify({ padding: 'x'.repeat(33000) }) });
  assert.equal(response.status, 413);
  const legacy = await fetch(`${origin}/v1/graph/entities/entity:one`, { headers: { Authorization: `Bearer ${token}` } });
  assert.equal(legacy.status, 200); assert.equal((await legacy.json()).id, 'entity:one');
});

test('frames come back as MCP image content, so a client needs nothing beyond /mcp to see them', async () => {
  const id = `media_${'b'.repeat(32)}`;
  const result = await call('get_video_frames', { media_id: id, offsets: [2.5] });
  assert.ok(!result.isError);
  assert.deepEqual(JSON.parse(result.content[0].text).frames, [{ media_id: id, offset_s: 2.5, size: 768 }]);
  assert.equal(result.content[1].type, 'image');
  assert.equal(result.content[1].mimeType, 'image/jpeg');
  assert.equal(Buffer.from(result.content[1].data, 'base64').toString(), 'jpeg-bytes');
  assert.equal((await call('get_video_frames', { media_id: id, offsets: [1, 2, 3, 4, 5, 6, 7] })).isError, true);
});
