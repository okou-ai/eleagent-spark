// Connect Eleagent to Okou: Eleagent is an MCP OAuth client of https://api.okou.ai/mcp. The resource's metadata
// names the authorization server. Eleagent identifies itself with a client ID metadata document it publishes at
// /okou/client.json (the client_id is that URL; Okou's Clerk supports it and has dynamic registration switched off),
// or registers itself (RFC 7591) with servers that only offer that. The owner approves once in the browser
// (authorization code + PKCE, offline_access), and the tokens stay in one mode-0600 file next to the auth
// database. The day agent on Spark 2 asks the internal listener for a current access token and never sees the
// refresh token.
import { randomBytes, createHash, timingSafeEqual } from 'node:crypto';
import { readFileSync, writeFileSync, renameSync, rmSync } from 'node:fs';

const RESOURCE = 'https://api.okou.ai/mcp';
const PENDING_MS = 10 * 60_000, REFRESH_EARLY_MS = 2 * 60_000;
const b64url = (buffer) => buffer.toString('base64url');
const escapeHtml = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

export function createOkouConnection({ publicUrl, file, resource = RESOURCE, fetchImpl = fetch, now = Date.now }) {
  const callback = new URL('/okou/callback', publicUrl).href;
  const clientIdUrl = new URL('/okou/client.json', publicUrl).href;
  const metadataDocument = () => ({ client_id: clientIdUrl, client_name: 'Eleagent', client_uri: new URL('/', publicUrl).href,
    redirect_uris: [callback], grant_types: ['authorization_code', 'refresh_token'], response_types: ['code'],
    token_endpoint_auth_method: 'none' });
  const pending = new Map();   // state -> { verifier, at }
  let refreshing = null;

  const load = () => { try { return JSON.parse(readFileSync(file, 'utf8')); } catch { return {}; } };
  const save = (state) => {
    const tmp = file + '.tmp';
    writeFileSync(tmp, JSON.stringify(state), { mode: 0o600 });
    renameSync(tmp, file);
  };
  async function json(url, init) {
    const response = await fetchImpl(url, { ...init, signal: AbortSignal.timeout(15_000) });
    const text = await response.text();
    let body = {};
    try { body = text ? JSON.parse(text) : {}; } catch { /* not JSON */ }
    const reason = body.error_description || body.errors?.[0]?.long_message || body.errors?.[0]?.message || body.error;
    if (!response.ok) throw new Error(`${new URL(url).host}${new URL(url).pathname} HTTP ${response.status}${reason ? ': ' + reason : ''}`);
    return body;
  }

  // Discovery follows the MCP authorization spec: the resource's protected-resource metadata, then the server's.
  async function discover() {
    const metadataUrl = new URL('/.well-known/oauth-protected-resource' + new URL(resource).pathname, resource).href;
    const prm = await json(metadataUrl);
    const issuer = prm.authorization_servers?.[0];
    if (!issuer) throw new Error('Okou names no authorization server');
    let as;
    try { as = await json(new URL('/.well-known/oauth-authorization-server', issuer).href); }
    catch { as = await json(new URL('/.well-known/openid-configuration', issuer).href); }
    return {
      issuer, scopes: prm.scopes_supported || [],
      authorize: as.authorization_endpoint, token: as.token_endpoint, cimd: as.client_id_metadata_document_supported === true,
      // Okou's Clerk accepts registrations at /oauth/register without advertising it.
      register: as.registration_endpoint || new URL('/oauth/register', issuer).href,
    };
  }

  async function client(server) {
    const state = load();
    if (server.cimd) {
      const record = { issuer: server.issuer, redirect_uri: callback, client_id: clientIdUrl, client_secret: null, auth_method: 'none' };
      if (state.client?.client_id !== clientIdUrl || state.client.issuer !== server.issuer) save({ ...state, client: record });
      return record;
    }
    if (state.client?.client_id && state.client.issuer === server.issuer && state.client.redirect_uri === callback) return state.client;
    const registered = await json(server.register, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ client_name: 'Eleagent', redirect_uris: [callback], grant_types: ['authorization_code', 'refresh_token'],
        response_types: ['code'], token_endpoint_auth_method: 'none', scope: server.scopes.join(' ') }),
    });
    const record = { issuer: server.issuer, redirect_uri: callback, client_id: registered.client_id,
      client_secret: registered.client_secret || null, auth_method: registered.token_endpoint_auth_method || 'none' };
    save({ ...load(), client: record });
    return record;
  }

  function tokenRequest(server, record, params) {
    const headers = { 'Content-Type': 'application/x-www-form-urlencoded', Accept: 'application/json' };
    const form = new URLSearchParams({ ...params, resource });
    if (record.client_secret && record.auth_method === 'client_secret_basic') {
      headers.Authorization = 'Basic ' + Buffer.from(`${encodeURIComponent(record.client_id)}:${encodeURIComponent(record.client_secret)}`).toString('base64');
    } else {
      form.set('client_id', record.client_id);
      if (record.client_secret) form.set('client_secret', record.client_secret);
    }
    return json(server.token, { method: 'POST', headers, body: form });
  }

  const keep = (tokens, previous = {}) => ({
    access_token: tokens.access_token, refresh_token: tokens.refresh_token || previous.refresh_token || null,
    scope: tokens.scope || previous.scope || null, expires_at: now() + 1000 * Number(tokens.expires_in || 3600),
  });

  async function start() {
    const server = await discover();
    const record = await client(server);
    const state = b64url(randomBytes(24)), verifier = b64url(randomBytes(48));
    for (const [key, value] of pending) if (now() - value.at > PENDING_MS) pending.delete(key);
    pending.set(state, { verifier, at: now() });
    const url = new URL(server.authorize);
    url.search = new URLSearchParams({ response_type: 'code', client_id: record.client_id, redirect_uri: callback,
      scope: server.scopes.join(' '), state, code_challenge: b64url(createHash('sha256').update(verifier).digest()),
      code_challenge_method: 'S256', resource, prompt: 'consent' }).toString();
    return url.href;
  }

  async function finish(params) {
    const entry = pending.get(params.get('state') || '');
    pending.delete(params.get('state') || '');
    if (!entry || now() - entry.at > PENDING_MS) throw new Error('This sign-in link expired; connect again');
    if (params.get('error')) throw new Error('Okou did not approve: ' + params.get('error'));
    const server = await discover();
    const record = load().client;
    const tokens = await tokenRequest(server, record, { grant_type: 'authorization_code', code: params.get('code') || '',
      redirect_uri: callback, code_verifier: entry.verifier });
    save({ ...load(), tokens: keep(tokens), connected_at: new Date(now()).toISOString() });
  }

  // A current access token for the day agent; one refresh at a time.
  async function accessToken() {
    const state = load();
    if (!state.tokens?.access_token) return null;
    if (state.tokens.expires_at - now() > REFRESH_EARLY_MS) return state.tokens;
    if (!state.tokens.refresh_token) return null;
    refreshing ??= (async () => {
      try {
        const server = await discover();
        const tokens = await tokenRequest(server, state.client, { grant_type: 'refresh_token', refresh_token: state.tokens.refresh_token });
        const kept = keep(tokens, state.tokens);
        save({ ...load(), tokens: kept, refreshed_at: new Date(now()).toISOString() });
        return kept;
      } finally { refreshing = null; }
    })();
    return refreshing;
  }

  function page(res, message) {
    const state = load();
    const connected = Boolean(state.tokens?.refresh_token || state.tokens?.access_token);
    res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
    res.end(`<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Okou</title><link rel="stylesheet" href="/auth/style.css"><main><h1>Okou</h1>
${message ? `<p>${escapeHtml(message)}</p>` : ''}
<p>${connected ? `Connected since ${escapeHtml(state.connected_at)}. When you ask Okou for help while recording, Eleagent hands the task to Okou.` : 'Connect Okou so that Eleagent can hand tasks you ask for while recording to Okou.'}</p>
<p><a href="/okou/connect">${connected ? 'Reconnect Okou' : 'Connect Okou'}</a> · <a href="/">Back to recording</a></p>
${connected ? '<form method="post" action="/okou/disconnect"><button type="submit">Disconnect</button></form>' : ''}</main>`);
  }

  return {
    matches: (pathname) => pathname === '/okou' || pathname.startsWith('/okou/'),
    // Fetched by Okou's authorization server, so it is public (it holds no secret).
    isPublic: (pathname) => pathname === '/okou/client.json',
    accessToken,
    // The owner's session is checked by the server before this runs; POSTs are same-origin there too.
    async handle(req, res, url) {
      req.resume();
      try {
        if (url.pathname === '/okou' && req.method === 'GET') return page(res);
        if (url.pathname === '/okou/client.json' && ['GET', 'HEAD'].includes(req.method)) {
          res.writeHead(200, { 'Content-Type': 'application/json', 'Cache-Control': 'public, max-age=300' });
          return res.end(req.method === 'HEAD' ? undefined : JSON.stringify(metadataDocument()));
        }
        // The Okou mark on the recording page: grey until connected, in colour once connected.
        if (url.pathname === '/okou/status' && req.method === 'GET') {
          const state = load();
          res.writeHead(200, { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' });
          return res.end(JSON.stringify({ connected: Boolean(state.tokens?.refresh_token || state.tokens?.access_token),
            connected_at: state.connected_at ?? null }));
        }
        // GET too: the recording page navigates here (its CSP form-action would stop a form posting off-site).
        if (url.pathname === '/okou/connect' && ['GET', 'POST'].includes(req.method)) {
          res.writeHead(303, { Location: await start() }); return res.end();
        }
        if (url.pathname === '/okou/callback' && req.method === 'GET') {
          await finish(url.searchParams);
          res.writeHead(303, { Location: '/' }); return res.end();
        }
        if (url.pathname === '/okou/disconnect' && req.method === 'POST') {
          const state = load(); delete state.tokens; save(state);
          res.writeHead(303, { Location: '/okou' }); return res.end();
        }
        res.writeHead(404, { 'Content-Type': 'text/plain' }); return res.end('Not found');
      } catch (error) {
        return page(res, String(error.message || error).slice(0, 300));
      }
    },
    // Internal listener (Spark 1 <-> Spark 2 link only): GET /internal/okou/token with the shared bearer secret.
    internalHandler(secret) {
      const expected = Buffer.from('Bearer ' + secret);
      return async (req, res) => {
        const given = Buffer.from(req.headers.authorization || '');
        const send = (status, body) => { res.writeHead(status, { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' }); res.end(JSON.stringify(body)); };
        if (given.length !== expected.length || !timingSafeEqual(given, expected)) return send(401, { error: 'unauthorized' });
        if (req.method !== 'GET' || new URL(req.url, 'http://internal').pathname !== '/internal/okou/token') return send(404, { error: 'not_found' });
        try {
          const tokens = await accessToken();
          if (!tokens) return send(409, { error: 'okou_not_connected' });
          return send(200, { access_token: tokens.access_token, expires_at: tokens.expires_at, resource });
        } catch (error) {
          return send(502, { error: 'okou_token_failed', detail: String(error.message || error).slice(0, 200) });
        }
      };
    },
    forget: () => rmSync(file, { force: true }),
  };
}
