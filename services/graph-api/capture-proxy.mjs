import http from 'node:http';
import https from 'node:https';
import { constants, readFileSync, realpathSync, statSync } from 'node:fs';
import { open } from 'node:fs/promises';
import { join } from 'node:path';
import { Transform, pipeline } from 'node:stream';

const FILES = new Map([
  ['/', ['index.html', 'text/html; charset=utf-8']],
  ['/assets/app.js', ['app.js', 'text/javascript; charset=utf-8']],
  ['/assets/capture-state.js', ['capture-state.js', 'text/javascript; charset=utf-8']],
  ['/assets/queue.js', ['queue.js', 'text/javascript; charset=utf-8']],
  ['/assets/style.css', ['style.css', 'text/css; charset=utf-8']],
  ['/assets/icon.svg', ['icon.svg', 'image/svg+xml']],
  ['/assets/manifest.webmanifest', ['manifest.webmanifest', 'application/manifest+json']],
  ['/assets/notices.js', ['notices.js', 'text/javascript; charset=utf-8']],
]);
// Matched to observer at main 301a796a23dca282ca94385b17899b1b11e137fd.
const BROWSER_HEADERS = {
  'cache-control': 'no-store', 'x-content-type-options': 'nosniff', 'referrer-policy': 'no-referrer',
  'content-security-policy': "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; media-src 'self' blob:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'",
  'permissions-policy': 'camera=(self), microphone=(self), geolocation=(self), screen-wake-lock=(self)',
};
const REQUEST_HEADERS = new Set(['accept', 'accept-encoding', 'accept-language', 'content-type', 'content-length', 'range', 'if-range', 'if-none-match', 'if-modified-since']);
const RESPONSE_HEADERS = new Set(['content-type', 'content-length', 'content-encoding', 'content-range', 'accept-ranges', 'etag', 'last-modified', 'content-security-policy', 'permissions-policy', 'x-content-type-options', 'referrer-policy']);
const HOP = new Set(['connection', 'keep-alive', 'proxy-authenticate', 'proxy-authorization', 'te', 'trailer', 'transfer-encoding', 'upgrade']);
function headersFrom(headers, allow) {
  const blocked = new Set([...HOP, ...String(headers.connection || '').toLowerCase().split(',').map(x => x.trim())]);
  return Object.fromEntries(Object.entries(headers).filter(([key]) => allow.has(key) && !blocked.has(key)));
}
function limiter(limit) {
  let bytes = 0;
  return new Transform({ transform(chunk, encoding, done) {
    bytes += chunk.length;
    if (bytes > limit) done(new Error('stream_limit'));
    else done(null, chunk);
  } });
}

/** Session/CSRF checks belong to the caller. No credentials from req reach observer.
 * HTTPS requires an explicit CA file and always verifies hostname and certificate.
 * No retries, redirects, buffering, logging, or upstream diagnostic forwarding.
 */
export function createCaptureProxy({ url, token, caFile, webRoot, maxUploadBytes = 102_000_000,
  maxResponseBytes = 8_000_000, timeoutMs = 120_000, idleTimeoutMs = 75_000 } = {}) {
  let target, ca, localRoot;
  try {
    target = new URL(url);
    if (!['http:', 'https:'].includes(target.protocol) || target.username || target.password ||
        target.pathname !== '/' || target.search || target.hash || typeof token !== 'string' || !token || /[\r\n]/.test(token)) throw new Error();
    if (target.protocol === 'https:') {
      if (!caFile) throw new Error();
      ca = readFileSync(caFile);
    }
    if (webRoot !== undefined) {
      if (typeof webRoot !== 'string' || !webRoot) throw new Error();
      localRoot = realpathSync(webRoot);
      if (!statSync(localRoot).isDirectory()) throw new Error();
    }
    for (const n of [maxUploadBytes, maxResponseBytes, timeoutMs, idleTimeoutMs]) {
      if (!Number.isSafeInteger(n) || n <= 0) throw new Error();
    }
  } catch { throw new Error('Invalid capture proxy configuration'); }
  const transport = target.protocol === 'https:' ? https : http;

  function proxy(req, res) {
    let upstream, response, timer, settled = false;
    function cleanup() {
      clearTimeout(timer);
      req.off('aborted', cancel);
      res.off('close', onClose);
    }
    function cancel() {
      if (settled) return;
      settled = true;
      cleanup();
      req.unpipe();
      upstream?.destroy();
      response?.destroy();
    }
    function onClose() { if (!res.writableFinished) cancel(); }
    function fail(status = 502) {
      if (settled) return;
      settled = true;
      cleanup();
      req.unpipe();
      upstream?.destroy();
      response?.destroy();
      if (res.headersSent) { res.destroy(); return; }
      // Fixed, bounded body. Never copy backend errors, redirects, cookies or URLs.
      const body = JSON.stringify({ error: status === 413 ? 'capture_upload_too_large' : status === 404 ? 'not_found' : 'capture_unavailable' });
      res.writeHead(status, { ...BROWSER_HEADERS, 'content-type': 'application/json', 'content-length': Buffer.byteLength(body) });
      res.end(req.method === 'HEAD' ? undefined : body);
      req.resume();
    }
    const rawPath = req.url || '';
    const path = rawPath.split('?')[0];
    const routeOK = ((req.method === 'GET' || req.method === 'HEAD') &&
      FILES.has(path)) || (req.method === 'GET' && path === '/v1/observations/latest') ||
      (req.method === 'POST' && (path === '/' || path === '/v1/clips'));
    if (!routeOK || !rawPath.startsWith('/') || rawPath.startsWith('//') || /[\r\n#]/.test(rawPath)) { fail(404); return; }
    if (req.headers['content-length'] !== undefined &&
      (!/^\d+$/.test(req.headers['content-length']) || Number(req.headers['content-length']) > maxUploadBytes)) { fail(413); return; }
    if (localRoot && (req.method === 'GET' || req.method === 'HEAD') && FILES.has(path)) {
      // Exact map only: no URL decoding, concatenated user paths, directory indexes or symlinks.
      // Bound by the opened inode's size; a file growing during a request cannot escape the limit.
      const [name, mime] = FILES.get(path);
      void (async () => {
        let file;
        try {
          file = await open(join(localRoot, name), constants.O_RDONLY | constants.O_NOFOLLOW | constants.O_NONBLOCK);
          const info = await file.stat();
          if (!info.isFile() || info.size > maxResponseBytes) { await file.close(); file = null; fail(); return; }
          if (res.destroyed) { await file.close(); return; }
          res.writeHead(200, { ...BROWSER_HEADERS, 'content-type': mime, 'content-length': info.size });
          if (req.method === 'HEAD' || info.size === 0) { await file.close(); res.end(); return; }
          const input = file.createReadStream({ start: 0, end: info.size - 1 });
          timer = setTimeout(() => { input.destroy(); fail(504); }, timeoutMs);
          timer.unref();
          const closed = () => input.destroy();
          res.once('close', closed);
          pipeline(input, limiter(maxResponseBytes), res, () => {
            clearTimeout(timer);
            res.off('close', closed);
          });
        } catch (error) {
          await file?.close().catch(() => {});
          fail(error.code === 'ENOENT' || error.code === 'ELOOP' ? 404 : 502);
        }
      })();
      return;
    }
    const headers = headersFrom(req.headers, REQUEST_HEADERS);
    headers.authorization = `Bearer ${token}`;
    req.once('aborted', cancel);
    res.once('close', onClose);
    timer = setTimeout(() => fail(504), timeoutMs);
    timer.unref();
    try {
      upstream = transport.request(target, { method: req.method, path: rawPath, headers,
        ...(ca ? { ca, rejectUnauthorized: true } : {}), agent: false }, incoming => {
        response = incoming;
        if (settled) { incoming.destroy(); return; }
        const status = incoming.statusCode || 502;
        if (status >= 300 && status !== 304) {
          fail(status >= 400 && status < 500 && status !== 401 && status !== 403 ? status : 502);
          return;
        }
        const outgoing = headersFrom(incoming.headers, RESPONSE_HEADERS);
        outgoing['cache-control'] = 'no-store';
        outgoing['x-content-type-options'] = 'nosniff';
        if (Number(incoming.headers['content-length']) > maxResponseBytes) { fail(); return; }
        res.writeHead(status, outgoing);
        pipeline(incoming, limiter(maxResponseBytes), res, error => {
          if (error) { fail(); return; }
          if (!settled) { settled = true; cleanup(); }
        });
      });
      upstream.on('error', () => fail());
      upstream.setTimeout(idleTimeoutMs, () => fail(504));
      const upload = limiter(maxUploadBytes);
      upload.on('error', () => fail(413));
      req.on('error', cancel);
      // pipe preserves backpressure. Do not pipeline req: destroying IncomingMessage
      // on an upload limit would also destroy the client socket before the 413.
      req.pipe(upload).pipe(upstream);
      upstream.once('close', () => { req.unpipe(upload); upload.unpipe(); upload.destroy(); });
    } catch { fail(); }
  }
  return { proxy };
}
