import test from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import { once } from 'node:events';
import { createApiServer } from './server.mjs';
import { createNoticesHandler } from './notices.mjs';

const at = second => new Date(Date.UTC(2026, 9, 20, 10, 0, second)).toISOString();

async function fixture(t, { now } = {}) {
  // NamingEvent and AgentEvent nodes as stored; the stub returns them all, so the handler's own window, status
  // mapping, order and cap are what the tests see.
  const namings = [], assists = [], queries = [];
  const store = { read: async callback => callback({ run: async (query, params) => {
    queries.push({ query, params });
    if (query.includes('NamingEvent')) return { records: namings.map(row => ({ get: key => row[key] })) };
    if (query.includes('AgentEvent')) return { records: assists.map(row => ({ get: key => row[key] })) };
    throw new Error('unexpected query ' + query);
  } }) };
  const listener = http.createServer(); listener.listen(0, '127.0.0.1'); await once(listener, 'listening');
  const origin = `http://127.0.0.1:${listener.address().port}`;
  const authService = { getOwnerSession: async headers => headers.get('cookie') === 'session=owner' ? { user: { id: 'owner' } } : null };
  const api = createApiServer({ store, cursorKey: 'k'.repeat(64), authService, authPublicUrl: origin,
    notices: createNoticesHandler({ store, ...(now && { now }) }) });
  listener.on('request', api.listeners('request')[0]);
  t.after(() => new Promise(r => listener.close(r)));
  const get = (path, { owner = true, headers = {} } = {}) => fetch(origin + path, { redirect: 'manual',
    headers: { ...(owner ? { cookie: 'session=owner' } : {}), ...headers } });
  return { get, queries, namings, assists, origin };
}

test('notices need the owner session and answer only GET', async t => {
  const f = await fixture(t);
  assert.equal((await f.get('/v1/notices', { owner: false })).status, 401);
  assert.equal((await fetch(f.origin + '/v1/notices', { method: 'POST', headers: { cookie: 'session=owner' } })).status, 405);
  const notices = createNoticesHandler({ store: {} });
  for (const path of ['/v1/live/notices', '/v1/live/diag']) assert.equal(notices.matches(path), false, path);
});

test('notices carry the last two minutes of namings and Okou handoffs, newest first', async t => {
  const now = Date.UTC(2026, 9, 20, 10, 2, 0);
  const f = await fixture(t, { now: () => now });
  f.namings.push(
    { id: 'naming:old', name: 'Old', status: 'named', memory_id: 'object:old', created_at: at(-1) },
    { id: 'naming:ignored', name: null, status: 'ignored', memory_id: null, created_at: at(30) },
    { id: 'naming:mug', name: 'Blue mug', status: 'named', memory_id: 'object:mug', created_at: at(40) },
    { id: 'naming:miss', name: 'Lamp', status: 'no_target', memory_id: null, created_at: { toString: () => at(50) } });
  f.assists.push({ id: 'assist_1', note: '约林小雨明天下午开会', status: 'sent', created_at: at(55) },
    { id: 'assist_2', note: null, status: 'not_connected', created_at: at(45) },
    { id: 'assist_3', note: 'x', status: 'failed', created_at: at(20) });
  const response = await f.get('/v1/notices');
  assert.equal(response.status, 200);
  assert.equal(response.headers.get('cache-control'), 'no-store');
  const body = await response.json();
  assert.deepEqual(body.namings.map(item => [item.id, item.status, item.name]), [['assist_1', 'assist', '约林小雨明天下午开会'],
    ['naming:miss', 'no_target', 'Lamp'], ['assist_2', 'assist_unconnected', null], ['naming:mug', 'named', 'Blue mug'],
    ['assist_3', 'assist_failed', 'x']]);
  assert.deepEqual(Object.keys(body.namings[0]).sort(), ['created_at', 'id', 'memory_id', 'name', 'status']);
  assert.equal(body.namings[3].memory_id, 'object:mug');
  assert.ok(f.queries.every(q => q.params.since === new Date(now - 120_000).toISOString()));
  assert.match(f.queries.find(q => q.query.includes('AgentEvent')).query, /'declined'/);
});

test('unchanged notices answer 304 to their ETag; a new one changes it', async t => {
  const f = await fixture(t, { now: () => Date.UTC(2026, 9, 20, 10, 1, 0) });
  const first = await f.get('/v1/notices'), tag = first.headers.get('etag');
  assert.ok(tag);
  assert.equal((await f.get('/v1/notices', { headers: { 'if-none-match': tag } })).status, 304);
  f.namings.push({ id: 'naming:mug', name: 'Blue mug', status: 'named', memory_id: 'object:mug', created_at: at(50) });
  const changed = await f.get('/v1/notices', { headers: { 'if-none-match': tag } });
  assert.equal(changed.status, 200);
  assert.notEqual(changed.headers.get('etag'), tag);
  assert.equal((await f.get('/v1/notices?x=1')).status, 400);
});
