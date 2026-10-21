import http from 'node:http';
import { ApiError } from './graph.mjs';
import { cursors, parseFilter, integer, string, queryParams } from './validation.mjs';
import { createMcpHandler, TOOL_SCOPES } from './mcp.mjs';
import { authRequest, sendAuthResponse, requireSameOrigin } from './auth-http.mjs';
import { sendAuthPage, AUTH_CSS, AUTH_JS } from './auth-pages.mjs';

const MAX_BODY = 32 * 1024;
const bad = (message) => new ApiError(400, 'invalid_request', message);
const object = (x) => x !== null && typeof x === 'object' && !Array.isArray(x);
const canonical = (x) => JSON.stringify(object(x)
  ? Object.fromEntries(Object.keys(x).sort().map((k) => [k, x[k]])) : x);

async function jsonBody(req) {
  if (!/^application\/json(?:\s*;|$)/i.test(req.headers['content-type'] || '')) throw new ApiError(415, 'unsupported_media_type', 'Use application/json');
  const length = Number(req.headers['content-length']);
  if (length > MAX_BODY) { req.resume(); throw new ApiError(413, 'body_too_large', 'Request body exceeds 32 KiB'); }
  let size = 0;
  const chunks = [];
  for await (const chunk of req.iterator({ destroyOnReturn: false })) {
    size += chunk.length;
    if (size > MAX_BODY) { req.resume(); throw new ApiError(413, 'body_too_large', 'Request body exceeds 32 KiB'); }
    chunks.push(chunk);
  }
  let result;
  try { result = JSON.parse(Buffer.concat(chunks).toString('utf8')); } catch { throw bad('Body must be valid JSON'); }
  if (!object(result)) throw bad('Body must be a JSON object');
  return result;
}

export function createApiServer({ store, cursorKey, archive, capture, notices, okou, authService, authPublicUrl, onRequest = () => {} }) {
  if (typeof cursorKey !== 'string' || cursorKey.length < 32) throw new Error('A private cursor signing key is required');
  if (!authService || !authPublicUrl) throw new Error('OAuth and an explicit public URL are required');
  const cursor = cursors(cursorKey);
  const mcp = createMcpHandler({ store, cursorKey, archive });
  mcp.origins.add(new URL(authPublicUrl).origin);
  const authorize = async (req, res, scopes) => {
    const result = await authService.verify(await authRequest(req, authPublicUrl), { scopes });
    if (result instanceof Response) { await sendAuthResponse(res, result); return false; }
    return true;
  };
  const server = http.createServer({ maxHeaderSize: 64 * 1024 }, async (req, res) => {
    const started = performance.now();
    let route = 'unknown';
    res.on('finish', () => onRequest(route, res.statusCode, (performance.now() - started) / 1000));
    res.setHeader('Cache-Control', 'no-store');
    res.setHeader('X-Content-Type-Options', 'nosniff');
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Expose-Headers', 'WWW-Authenticate, X-Graph-Consistency, Content-Range, Accept-Ranges, ETag, X-Archive-SHA256, X-Archive-Media-ID');
    res.setHeader('X-Graph-Consistency', 'live');
    const send = (status, body) => {
      res.writeHead(status, { 'Content-Type': 'application/json; charset=utf-8' });
      res.end(JSON.stringify(body));
    };
    try {
      const url = new URL(req.url, 'http://graph.local');
      if (req.method === 'OPTIONS') {
        if (!(authService && (url.pathname.startsWith('/api/auth/') || url.pathname.startsWith('/.well-known/'))) && !url.pathname.startsWith('/v1/graph/') && url.pathname !== '/mcp' && !/^\/v1\/archive\/media\/media_[a-f0-9]{32}\/original$/.test(url.pathname)) throw new ApiError(404, 'not_found', 'Route not found');
        res.writeHead(204, {
          'Access-Control-Allow-Methods': 'GET, HEAD, POST, OPTIONS',
          'Access-Control-Allow-Headers': 'Authorization, Content-Type, MCP-Protocol-Version, MCP-Session-Id, Range, If-Range',
          'Access-Control-Max-Age': '600',
        });
        return res.end();
      }
      if (authService && (url.pathname.startsWith('/api/auth/') || url.pathname.startsWith('/.well-known/'))) {
        route = 'auth_api';
        // OAuth clients may omit Origin; browser calls must use configured trusted origins.
        if (req.headers.origin && req.headers.origin !== new URL(authPublicUrl).origin && !mcp.origins.has(req.headers.origin)) {
          throw new ApiError(403, 'forbidden_origin', 'Origin is not allowed');
        }
        const response = await authService.handler(await authRequest(req, authPublicUrl, { body: true }));
        return await sendAuthResponse(res, response);
      }
      if (authService && url.pathname.startsWith('/auth/')) {
        route = 'auth_page';
        if (req.method === 'GET' && ['/auth/style.css', '/auth/app.js'].includes(url.pathname)) {
          queryParams(url, []);
          res.writeHead(200, { 'Content-Type': url.pathname.endsWith('.css') ? 'text/css' : 'text/javascript', 'Cache-Control': 'no-store' });
          return res.end(url.pathname.endsWith('.css') ? AUTH_CSS : AUTH_JS);
        }
        if (url.pathname === '/auth/grants' && req.method === 'GET') {
          queryParams(url, []);
          try {
            const grants = await authService.listGrants((await authRequest(req, authPublicUrl)).headers);
            if (grants instanceof Response) return await sendAuthResponse(res, grants);
            return send(200, grants);
          } catch { throw new ApiError(401, 'unauthorized', 'Sign in to manage access'); }
        }
        if (url.pathname === '/auth/revoke' && req.method === 'POST') {
          requireSameOrigin(req, authPublicUrl); queryParams(url, []);
          const body = await jsonBody(req);
          if (Object.keys(body).some((key) => key !== 'clientId')) throw bad('Unknown revoke field');
          const clientId = string(body.clientId, 2048, 'clientId');
          try {
            const result = await authService.revokeGrant(clientId, (await authRequest(req, authPublicUrl)).headers);
            if (result instanceof Response) return await sendAuthResponse(res, result);
          } catch { throw new ApiError(403, 'revoke_denied', 'Sign in to revoke this application'); }
          return send(200, { ok: true });
        }
        if (req.method === 'GET' && ['/auth/sign-in', '/auth/account', '/auth/consent'].includes(url.pathname)) {
          if (url.pathname === '/auth/consent') {
            const details = await authService.consentDetails(await authRequest(req, authPublicUrl));
            if (details instanceof Response) return await sendAuthResponse(res, details);
            return sendAuthPage(res, 'consent', details);
          }
          if (url.pathname === '/auth/account') {
            const session = await authService.auth.api.getSession({ headers: (await authRequest(req, authPublicUrl)).headers });
            if (!session) { res.writeHead(302, { Location: '/auth/sign-in' }); return res.end(); }
          }
          if (url.pathname === '/auth/sign-in' && url.searchParams.has('return_to')) {
            queryParams(url, ['return_to']);
            const target = url.searchParams.get('return_to');
            if (!target?.startsWith('/') || target.startsWith('//') || /[\\\u0000-\u0020]/.test(target) || new URL(target, authPublicUrl).origin !== new URL(authPublicUrl).origin) throw bad('Invalid sign-in return path');
          }
          return sendAuthPage(res, url.pathname.slice('/auth/'.length));
        }
        throw new ApiError(404, 'not_found', 'Account route not found');
      }
      if (okou && authService && okou.matches(url.pathname)) {
        route = 'okou';
        if (okou.isPublic(url.pathname)) return await okou.handle(req, res, url);
        if (req.method === 'POST') requireSameOrigin(req, authPublicUrl);
        const session = await authService.getOwnerSession((await authRequest(req, authPublicUrl)).headers);
        if (!session) {
          req.resume();
          if (req.method === 'GET' && url.pathname !== '/okou/status') {
            res.writeHead(302, { Location: '/auth/sign-in?return_to=' + encodeURIComponent(url.pathname) }); return res.end();
          }
          return send(401, { error: { code: 'sign_in_required', message: 'Sign in to connect Okou' } });
        }
        return await okou.handle(req, res, url);
      }
      if (notices && authService && notices.matches(url.pathname)) {
        route = 'notices';
        const session = await authService.getOwnerSession((await authRequest(req, authPublicUrl)).headers);
        if (!session) {
          req.resume();
          return send(401, { error: { code: 'sign_in_required', message: 'Sign in to use the capture page' } });
        }
        return await notices.handle(req, res, url);
      }
      const capturePath = url.pathname === '/' || (['GET', 'HEAD'].includes(req.method) && (url.pathname === '/v1/observations/latest' || /^\/assets\/[a-zA-Z0-9_.-]+$/.test(url.pathname))) || (req.method === 'POST' && url.pathname === '/v1/clips');
      if (capture && capturePath && ['GET', 'HEAD', 'POST'].includes(req.method)) {
        route = 'capture';
        if (req.method === 'POST') requireSameOrigin(req, authPublicUrl);
        const session = await authService.getOwnerSession((await authRequest(req, authPublicUrl)).headers);
        if (!session) {
          if (['GET', 'HEAD'].includes(req.method)) {
            res.writeHead(302, { Location: '/auth/sign-in?return_to=' + encodeURIComponent(req.url) }); return res.end();
          }
          req.resume(); return send(401, { error: { code: 'sign_in_required', message: 'Sign in to upload recordings' } });
        }
        return await capture.proxy(req, res);
      }
      if (url.pathname === '/mcp') {
        route = 'mcp';
        queryParams(url, []);
        // Non-browser clients omit Origin; browsers must explicitly use a trusted origin.
        if (req.headers.origin && !mcp.origins.has(req.headers.origin)) {
          throw new ApiError(403, 'forbidden_origin', 'Origin is not allowed');
        }
        const body = req.method === 'POST' ? await jsonBody(req) : undefined;
        const tool = body?.method === 'tools/call' ? body?.params?.name : undefined;
        const scope = typeof tool === 'string' && Object.hasOwn(TOOL_SCOPES, tool) ? TOOL_SCOPES[tool] : undefined;
        if (!await authorize(req, res, scope ? [scope] : [])) return;
        return await mcp.handle(req, res, body);
      }
      const original = /^\/v1\/archive\/media\/(media_[a-f0-9]{32})\/original$/.exec(url.pathname);
      if (original) {
        route = 'video_original';
        queryParams(url, []);
        if (!['GET', 'HEAD'].includes(req.method)) throw new ApiError(405, 'method_not_allowed', 'Use GET or HEAD');
        if (!archive) throw new ApiError(503, 'archive_unavailable', 'Archive is not configured');
        if (!await authorize(req, res, ['media:read'])) return;
        return await archive.proxyOriginal(req, res, original[1]);
      }
      if (!await authorize(req, res, ['graph:read'])) return;
      const match = /^\/v1\/graph\/entities\/([^/]+)(\/links)?$/.exec(url.pathname);
      if (url.pathname === '/v1/graph/entities') {
        route = 'entities';
        if (req.method !== 'GET') throw new ApiError(405, 'method_not_allowed', 'Use GET');
        const p = queryParams(url, ['search', 'filter', 'sort', 'order', 'limit', 'cursor']);
        const search = p.search ?? '';
        if (search.length > 512) throw bad('search must be at most 512 characters');
        const filter = parseFilter(p.filter);
        const sort = string(p.sort ?? 'id', 256, 'sort');
        const order = p.order ?? 'asc';
        if (!['asc', 'desc'].includes(order)) throw bad('order must be asc or desc');
        const limit = integer(p.limit, 50, 1, 200, 'limit');
        const scope = ['entities', search, canonical(filter), sort, order];
        const result = await store.listEntities({ search, filter, sort, order, limit, after: cursor.decode(p.cursor, scope) });
        return send(200, { entities: result.entities, next_cursor: cursor.encode(result.after, scope) });
      }
      if (match) {
        let id;
        try { id = decodeURIComponent(match[1]); } catch { throw bad('Entity ID contains invalid encoding'); }
        string(id, 512, 'Entity ID');
        route = match[2] ? 'links' : 'entity';
        if (req.method !== 'GET') throw new ApiError(405, 'method_not_allowed', 'Use GET');
        if (!match[2]) {
          queryParams(url, []);
          return send(200, await store.getEntity(id));
        }
        const p = queryParams(url, ['limit', 'cursor']);
        const scope = ['links', id];
        const after = cursor.decode(p.cursor, scope);
        if (after !== null) string(after, 512, 'Cursor entity ID');
        const result = await store.listLinks(id, {
          limit: integer(p.limit, 100, 1, 500, 'limit'), after,
        });
        return send(200, { entities: result.entities, links: result.links, next_cursor: cursor.encode(result.after, scope) });
      }
      if (url.pathname === '/v1/graph/subgraph') {
        route = 'subgraph';
        if (req.method !== 'POST') throw new ApiError(405, 'method_not_allowed', 'Use POST');
        queryParams(url, []);
        const body = await jsonBody(req);
        if (Object.keys(body).some((k) => !['ids', 'depth', 'max_entities', 'max_links'].includes(k))) throw bad('Unknown body field');
        if (!Array.isArray(body.ids) || !body.ids.length || body.ids.length > 50) throw bad('ids must contain 1 to 50 entity IDs');
        const ids = [...new Set(body.ids.map((id) => string(id, 512, 'Entity ID')))];
        const depth = integer(body.depth, 1, 0, 3, 'depth');
        const max_entities = integer(body.max_entities, 100, 1, 500, 'max_entities');
        const max_links = integer(body.max_links, 200, 1, 2000, 'max_links');
        if (ids.length > max_entities) throw bad('max_entities must be at least the number of seed entities');
        return send(200, await store.subgraph({ ids, depth, max_entities, max_links }));
      }
      throw new ApiError(404, 'not_found', 'Route not found');
    } catch (error) {
      if (res.destroyed || res.writableEnded) return;
      if (res.headersSent) { res.destroy(); return; }
      if (error instanceof ApiError) return send(error.status, { error: { code: error.code, message: error.message } });
      // Never return or log driver messages, query text, tokens, or entity contents.
      const unavailable = typeof error?.code === 'string' && /Neo\.|ServiceUnavailable|SessionExpired/.test(error.code);
      send(unavailable ? 503 : 500, { error: {
        code: unavailable ? 'graph_unavailable' : 'internal_error',
        message: unavailable ? 'Graph query unavailable; try again later' : 'Request could not be completed',
      } });
    }
  });
  server.requestTimeout = 15_000;
  server.headersTimeout = 10_000;
  server.keepAliveTimeout = 5000;
  return server;
}
