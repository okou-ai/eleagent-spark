import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, rm, stat } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import http from 'node:http';
import { createOkouConnection } from './okou.mjs';

// A fake Okou resource + Clerk: protected-resource metadata, server metadata without registration_endpoint (with
// client ID metadata documents, like Okou's, or without), registration, and a token endpoint that counts refreshes.
function fakeOkou({ cimd = true, registration = true } = {}) {
  const calls = { register: [], token: [] };
  let issued = 0;
  const fetchImpl = async (url, init = {}) => {
    const u = new URL(url);
    const reply = (status, body) => new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
    if (u.href === 'https://api.okou.ai/.well-known/oauth-protected-resource/mcp') {
      return reply(200, { resource: 'https://api.okou.ai/mcp', authorization_servers: ['https://clerk.okou.ai'], scopes_supported: ['openid', 'okou:chat:send', 'offline_access'] });
    }
    if (u.href === 'https://clerk.okou.ai/.well-known/oauth-authorization-server') {
      return reply(200, { issuer: 'https://clerk.okou.ai', authorization_endpoint: 'https://clerk.okou.ai/oauth/authorize', token_endpoint: 'https://clerk.okou.ai/oauth/token',
        ...(cimd ? { client_id_metadata_document_supported: true } : {}) });
    }
    if (u.href === 'https://clerk.okou.ai/oauth/register') {
      calls.register.push(JSON.parse(init.body));
      if (!registration) return reply(422, { errors: [{ message: 'dynamic client registration is not enabled', long_message: 'Dynamic client registration is not enabled on this instance.' }] });
      return reply(201, { client_id: 'client_1', token_endpoint_auth_method: 'none' });
    }
    if (u.href === 'https://clerk.okou.ai/oauth/token') {
      const form = new URLSearchParams(init.body.toString());
      calls.token.push(Object.fromEntries(form));
      if (form.get('grant_type') === 'authorization_code' && form.get('code') !== 'code_1') return reply(400, { error: 'invalid_grant' });
      issued += 1;
      return reply(200, { access_token: 'access_' + issued, refresh_token: form.get('grant_type') === 'authorization_code' ? 'refresh_1' : undefined, expires_in: 3600, scope: 'openid okou:chat:send offline_access' });
    }
    return reply(404, { error: 'not_found' });
  };
  return { fetchImpl, calls };
}

test('connects with its client ID metadata document and PKCE, then refreshes for the agent', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'okou-'));
  try {
    const { fetchImpl, calls } = fakeOkou();
    let clock = Date.UTC(2026, 9, 21, 9);
    const okou = createOkouConnection({ publicUrl: 'https://eleagent.example', file: join(dir, 'okou.json'), fetchImpl, now: () => clock });
    assert.equal(await okou.accessToken(), null);

    // Connect: registered with the callback, sent to Clerk with PKCE, the resource and the advertised scopes.
    const res = { headers: {}, writeHead(status, headers) { this.status = status; Object.assign(this.headers, headers); }, end() {} };
    const req = { method: 'GET', resume() {} };
    const status = { headers: {}, writeHead(code, headers) { this.status = code; Object.assign(this.headers, headers); }, end(body) { this.body = body; } };
    await okou.handle(req, status, new URL('https://eleagent.example/okou/status'));
    assert.deepEqual(JSON.parse(status.body), { connected: false, connected_at: null });
    await okou.handle(req, res, new URL('https://eleagent.example/okou/connect'));
    assert.equal(res.status, 303);
    const authorize = new URL(res.headers.Location);
    assert.equal(authorize.origin + authorize.pathname, 'https://clerk.okou.ai/oauth/authorize');
    assert.equal(authorize.searchParams.get('client_id'), 'https://eleagent.example/okou/client.json');
    assert.equal(authorize.searchParams.get('code_challenge_method'), 'S256');
    assert.equal(authorize.searchParams.get('resource'), 'https://api.okou.ai/mcp');
    assert.equal(authorize.searchParams.get('scope'), 'openid okou:chat:send offline_access');
    assert.equal(calls.register.length, 0);
    // The document Clerk fetches: public, its client_id is its own URL, the callback is the only redirect.
    assert.equal(okou.isPublic('/okou/client.json'), true);
    assert.equal(okou.isPublic('/okou/status'), false);
    const doc = { headers: {}, writeHead(code, headers) { this.status = code; Object.assign(this.headers, headers); }, end(body) { this.body = body; } };
    await okou.handle({ method: 'GET', resume() {} }, doc, new URL('https://eleagent.example/okou/client.json'));
    const metadata = JSON.parse(doc.body);
    assert.equal(metadata.client_id, 'https://eleagent.example/okou/client.json');
    assert.deepEqual(metadata.redirect_uris, ['https://eleagent.example/okou/callback']);
    assert.equal(metadata.token_endpoint_auth_method, 'none');

    // A forged state is refused; the real callback exchanges the code with the verifier.
    const back = { headers: {}, writeHead(status, headers) { this.status = status; Object.assign(this.headers, headers); }, end(body) { this.body = body; } };
    await okou.handle({ method: 'GET', resume() {} }, back, new URL('https://eleagent.example/okou/callback?code=code_1&state=forged'));
    assert.match(back.body, /expired/);
    const state = authorize.searchParams.get('state');
    const done = { headers: {}, writeHead(status, headers) { this.status = status; Object.assign(this.headers, headers); }, end() {} };
    await okou.handle({ method: 'GET', resume() {} }, done, new URL(`https://eleagent.example/okou/callback?code=code_1&state=${state}`));
    assert.equal(done.status, 303);
    assert.equal(done.headers.Location, '/');
    const after = { headers: {}, writeHead(code, headers) { this.status = code; Object.assign(this.headers, headers); }, end(body) { this.body = body; } };
    await okou.handle({ method: 'GET', resume() {} }, after, new URL('https://eleagent.example/okou/status'));
    assert.equal(JSON.parse(after.body).connected, true);
    assert.ok(calls.token[0].code_verifier.length >= 43);
    assert.equal(calls.token[0].client_id, 'https://eleagent.example/okou/client.json');
    assert.equal(calls.token[0].resource, 'https://api.okou.ai/mcp');
    assert.equal((await stat(join(dir, 'okou.json'))).mode & 0o777, 0o600);
    assert.equal((await okou.accessToken()).access_token, 'access_1');

    // Near expiry the agent gets a refreshed token; the refresh token is kept.
    clock += 3600_000 - 60_000;
    assert.equal((await okou.accessToken()).access_token, 'access_2');
    assert.equal(calls.token[1].grant_type, 'refresh_token');
    assert.equal(calls.token[1].refresh_token, 'refresh_1');

    // The internal listener needs the shared secret and never returns the refresh token.
    const server = http.createServer(okou.internalHandler('s3cret'));
    await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
    const base = `http://127.0.0.1:${server.address().port}/internal/okou/token`;
    try {
      assert.equal((await fetch(base)).status, 401);
      assert.equal((await fetch(base, { headers: { Authorization: 'Bearer wrong!' } })).status, 401);
      const ok = await fetch(base, { headers: { Authorization: 'Bearer s3cret' } });
      assert.equal(ok.status, 200);
      const body = await ok.json();
      assert.equal(body.access_token, 'access_2');
      assert.equal(body.refresh_token, undefined);
    } finally { await new Promise((resolve) => server.close(resolve)); }
  } finally { await rm(dir, { recursive: true, force: true }); }
});

test('without metadata documents it registers itself, and shows the server\'s reason when that is refused', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'okou-'));
  try {
    const page = () => ({ headers: {}, writeHead(code, headers) { this.status = code; Object.assign(this.headers, headers); }, end(body) { this.body = body; } });
    const ok = fakeOkou({ cimd: false });
    const registered = createOkouConnection({ publicUrl: 'https://eleagent.example', file: join(dir, 'a.json'), fetchImpl: ok.fetchImpl });
    const res = page();
    await registered.handle({ method: 'GET', resume() {} }, res, new URL('https://eleagent.example/okou/connect'));
    assert.equal(new URL(res.headers.Location).searchParams.get('client_id'), 'client_1');
    assert.deepEqual(ok.calls.register[0].redirect_uris, ['https://eleagent.example/okou/callback']);

    const refused = createOkouConnection({ publicUrl: 'https://eleagent.example', file: join(dir, 'b.json'),
      fetchImpl: fakeOkou({ cimd: false, registration: false }).fetchImpl });
    const shown = page();
    await refused.handle({ method: 'GET', resume() {} }, shown, new URL('https://eleagent.example/okou/connect'));
    assert.match(shown.body, /HTTP 422: Dynamic client registration is not enabled/);
  } finally { await rm(dir, { recursive: true, force: true }); }
});
