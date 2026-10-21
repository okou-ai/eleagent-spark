import test from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import { once } from 'node:events';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createHash, randomBytes } from 'node:crypto';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { auth as sdkAuth } from '@modelcontextprotocol/sdk/client/auth.js';
import { StreamableHTTPClientTransport } from '@modelcontextprotocol/sdk/client/streamableHttp.js';
import { createAuthService } from './auth.mjs';
import { createApiServer } from './server.mjs';

// Entire HTTP boundary with actual Better Auth/SQLite and official MCP OAuth SDK.
// Graph/media data are synthetic here; live real-data validation is a separate deployment check.
test('official MCP OAuth SDK discovers, registers, authorizes, reads tools/media, and loses access on owner revocation', async (t) => {
  const dir = await mkdtemp(join(tmpdir(), 'eleagent-oauth-http-'));
  const listener = http.createServer(); listener.listen(0, '127.0.0.1'); await once(listener, 'listening');
  const origin = `http://127.0.0.1:${listener.address().port}`;
  const cursorKey = randomBytes(32).toString('hex');
  const service = await createAuthService({ publicUrl: origin, databasePath: join(dir, 'auth.sqlite'), secret: randomBytes(48).toString('base64url') });
  await service.setupOwner({ email: 'owner@example.test', password: 'synthetic-strong-owner-password' });
  const mediaId = `media_${'a'.repeat(32)}`; const original = Buffer.from('synthetic original video bytes');
  const store = {
    getEntity: async (id) => ({ id, properties: { kind: 'episode' } }),
    listEntities: async () => ({ entities: [{ id: 'episode:synthetic', properties: { kind: 'episode' } }], after: null }),
    listLinks: async (id) => ({ entities: [{ id, properties: {} }], links: [], after: null }),
    subgraph: async ({ ids }) => ({ entities: ids.map((id) => ({ id, properties: {} })), links: [], truncated: false }),
  };
  const metadata = { media_id: mediaId, original_url: `/v1/archive/media/${mediaId}/original`, original_requires_auth: true, size_bytes: original.length, sha256: createHash('sha256').update(original).digest('hex') };
  const api = createApiServer({ store, cursorKey, authService: service, authPublicUrl: origin, capture: {
    proxy(req, res) { req.resume(); res.writeHead(200, { 'Content-Type': 'text/plain' }); res.end('Owner-only capture'); },
  }, archive: {
    listVideoClips: async () => ({ items: [metadata], next_cursor: null, missing_media_ids: [], has_more: false }),
    getVideoClip: async () => metadata,
    proxyOriginal: async (req, res) => { const range = req.headers.range; res.writeHead(range ? 206 : 200, { 'Content-Type': 'video/mp4', 'Content-Length': range ? '3' : String(original.length) }); res.end(req.method === 'HEAD' ? undefined : range ? original.subarray(0, 3) : original); },
  } });
  listener.on('request', api.listeners('request')[0]);
  let client;
  t.after(async () => { await client?.close(); await new Promise((r) => listener.close(r)); await service.close(); await rm(dir, { recursive: true, force: true }); });
  const cookies = new Map();
  async function browser(path, body) {
    const response = await fetch(new URL(path, origin), { redirect: 'manual', method: body ? 'POST' : 'GET', headers: {
      ...(cookies.size ? { Cookie: [...cookies].map(([k, v]) => `${k}=${v}`).join('; ') } : {}),
      ...(body ? { Origin: origin, 'Content-Type': 'application/json' } : { Accept: 'text/html' }),
    }, ...(body ? { body: JSON.stringify(body) } : {}) });
    for (const cookie of response.headers.getSetCookie()) { const pair = cookie.split(';')[0], at = pair.indexOf('='); cookies.set(pair.slice(0, at), pair.slice(at + 1)); }
    return response;
  }
  assert.equal((await browser('/auth/setup', { email: 'intruder@example.test', password: 'synthetic-strong-owner-password' })).status, 404);
  const anonymousCapture = await browser('/'); assert.equal(anonymousCapture.status, 302);
  const before = await fetch(origin + '/mcp'); assert.equal(before.status, 401); assert.ok(before.headers.get('www-authenticate').includes('resource_metadata='));
  const state = randomBytes(24).toString('base64url');
  const provider = {
    redirectUrl: origin + '/synthetic-callback',
    clientMetadata: { client_name: 'Actual SDK integration', redirect_uris: [origin + '/synthetic-callback'], grant_types: ['authorization_code', 'refresh_token'], response_types: ['code'], token_endpoint_auth_method: 'none', scope: 'graph:read media:read' },
    state: () => state,
    clientInformation() { return this.info; }, saveClientInformation(value) { this.info = value; },
    tokens() { return this.savedTokens; }, saveTokens(value) { this.savedTokens = value; },
    redirectToAuthorization(value) { this.authorizationUrl = value; },
    saveCodeVerifier(value) { this.verifier = value; }, codeVerifier() { return this.verifier; },
  };
  assert.equal(await sdkAuth(provider, { serverUrl: origin + '/mcp', scope: 'graph:read media:read' }), 'REDIRECT');
  const login = await browser(provider.authorizationUrl); assert.ok([200, 302].includes(login.status));
  const loginUrl = new URL(login.status === 302 ? login.headers.get('location') : (await login.json()).url, origin); assert.equal(loginUrl.pathname, '/auth/sign-in');
  assert.equal((await browser(loginUrl)).status, 200);
  const signedQuery = loginUrl.search.slice(1);
  const signedIn = await browser('/api/auth/sign-in/email', { email: 'owner@example.test', password: 'synthetic-strong-owner-password', oauth_query: signedQuery });
  assert.equal(signedIn.status, 200); const continuation = await signedIn.json();
  assert.equal(continuation.redirect, true);
  const consentUrl = new URL(continuation.url, origin); assert.equal(consentUrl.pathname, '/auth/consent');
  const consentPage = await browser(consentUrl); assert.equal(consentPage.status, 200); assert.ok((await consentPage.text()).includes('Actual SDK integration'));
  const consent = await browser('/api/auth/oauth2/consent', { accept: true, oauth_query: consentUrl.search.slice(1) });
  assert.equal(consent.status, 200); const consentResult = await consent.json();
  const callback = new URL(consentResult.url || consentResult.redirect_uri); assert.equal(callback.searchParams.get('state'), state);
  assert.equal(callback.searchParams.get('iss'), origin + '/api/auth');
  assert.equal(await sdkAuth(provider, { serverUrl: origin + '/mcp', authorizationCode: callback.searchParams.get('code') }), 'AUTHORIZED');
  assert.ok(provider.savedTokens.access_token); assert.equal(provider.savedTokens.refresh_token, undefined);
  assert.ok(provider.savedTokens.expires_in > 364 * 24 * 3600, 'access tokens are long-lived');
  client = new Client({ name: 'authenticated-tools-integration', version: '1.0.0' });
  await client.connect(new StreamableHTTPClientTransport(new URL(origin + '/mcp'), { authProvider: provider }));
  assert.equal((await client.listTools()).tools.length, 7);
  for (const [name, args] of [['search_entities', {}], ['get_entity', { id: 'episode:synthetic' }], ['list_entity_links', { id: 'episode:synthetic' }], ['get_subgraph', { ids: ['episode:synthetic'] }], ['list_video_clips', { entity_id: 'episode:synthetic' }], ['get_video_clip', { media_id: mediaId }]]) {
    const r = await client.callTool({ name, arguments: args }); assert.ok(!r.isError, `${name} must be accessible with the approved scopes`);
  }
  const headers = { Authorization: `Bearer ${provider.savedTokens.access_token}` };
  const bytes = await fetch(origin + metadata.original_url, { headers }); assert.equal(bytes.status, 200); assert.deepEqual(Buffer.from(await bytes.arrayBuffer()), original);
  assert.equal((await fetch(origin + metadata.original_url)).status, 401);
  assert.equal((await fetch(origin + metadata.original_url, { headers: { ...headers, Range: 'bytes=0-2' } })).status, 206);
  assert.equal((await fetch(origin + '/mcp', { headers: { Cookie: [...cookies].map(([k, v]) => `${k}=${v}`).join('; ') } })).status, 401, 'Browser cookie is not a resource bearer token');
  assert.equal((await fetch(origin + '/mcp', { headers: { Authorization: `Bearer ${cursorKey}` } })).status, 401, 'Cursor signing material is not an access token');
  assert.equal((await browser('/')).status, 200, 'Owner session can open capture without an API key');
  const cookie = [...cookies].map(([k, v]) => `${k}=${v}`).join('; ');
  assert.equal((await fetch(origin + '/', { method: 'POST', headers: { Origin: origin, Cookie: cookie }, body: 'synthetic recording' })).status, 200);
  assert.equal((await fetch(origin + '/', { method: 'POST', headers: { Origin: 'https://untrusted.example', Cookie: cookie }, body: 'synthetic recording' })).status, 403);
  assert.equal((await fetch(origin + '/', { method: 'POST', headers: { Origin: origin, ...headers }, body: 'synthetic recording' })).status, 401, 'Read-only OAuth token cannot upload captures');
  const grants = await browser('/auth/grants'); const applications = await grants.json(); assert.equal(applications.length, 1);
  assert.equal((await browser('/auth/revoke', { clientId: applications[0].clientId })).status, 200);
  assert.equal((await fetch(origin + metadata.original_url, { headers })).status, 401);
  assert.equal((await fetch(origin + '/mcp', { headers })).status, 401);
});
