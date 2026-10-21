import test from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import https from 'node:https';
import { once } from 'node:events';
import { mkdtempSync, readFileSync, rmSync, writeFileSync, symlinkSync, mkdirSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { execFileSync } from 'node:child_process';
import { createCaptureProxy } from './capture-proxy.mjs';

const secret = 'server-only-observer-secret';
async function listen(t, handler, tls) {
  const server = tls ? https.createServer(tls, handler) : http.createServer(handler);
  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  t.after(() => { server.closeAllConnections(); server.close(); });
  return `${tls ? 'https' : 'http'}://127.0.0.1:${server.address().port}`;
}
async function setup(t, handler, options = {}) {
  const url = await listen(t, handler);
  const { proxy } = createCaptureProxy({ url, token: secret, ...options });
  return listen(t, proxy);
}
function request(url, options = {}, chunks = []) {
  return new Promise((resolve, reject) => {
    const req = http.request(url, options, res => {
      const data = [];
      res.on('data', chunk => data.push(chunk));
      res.on('error', reject);
      res.on('end', () => resolve({ status: res.statusCode, headers: res.headers, body: Buffer.concat(data).toString() }));
    });
    req.on('error', reject);
    for (const chunk of chunks) req.write(chunk);
    req.end();
  });
}

test('streams multipart bytes and preserves path, query and Range; isolates credentials in both directions', async t => {
  const payload = Buffer.from('multipart\0\xff\r\nraw bytes');
  const base = await setup(t, (req, res) => {
    assert.equal(req.url, '/v1/clips?capture=a%2Fb&x=1&x=2');
    assert.equal(req.headers.authorization, `Bearer ${secret}`);
    assert.equal(req.headers.cookie, undefined);
    assert.equal(req.headers['proxy-authorization'], undefined);
    assert.equal(req.headers.origin, undefined);
    assert.equal(req.headers['x-forwarded-host'], undefined);
    assert.equal(req.headers['x-private'], undefined);
    assert.equal(req.headers.range, 'bytes=0-99');
    assert.equal(req.headers['content-type'], 'multipart/form-data; boundary=abc');
    const bytes = [];
    req.on('data', b => bytes.push(b));
    req.on('end', () => {
      assert.deepEqual(Buffer.concat(bytes), payload);
      res.writeHead(206, { 'content-type': 'application/json', 'content-range': 'bytes 0-1/2',
        'set-cookie': `secret=${secret}`, authorization: secret, 'access-control-allow-origin': '*',
        'x-internal-url': 'http://observer.internal', connection: 'x-secret', 'x-secret': secret });
      res.end('{}');
    });
  });
  const result = await request(base + '/v1/clips?capture=a%2Fb&x=1&x=2', { method: 'POST', headers: {
    authorization: 'Bearer browser-secret', cookie: 'session=private', 'proxy-authorization': 'private',
    'x-forwarded-host': 'attacker', origin: 'https://evil.test', connection: 'x-private', 'x-private': 'private',
    range: 'bytes=0-99', 'content-type': 'multipart/form-data; boundary=abc' } }, [payload]);
  assert.equal(result.status, 206);
  assert.equal(result.body, '{}');
  assert.equal(result.headers['content-range'], 'bytes 0-1/2');
  for (const key of ['set-cookie', 'authorization', 'access-control-allow-origin', 'x-internal-url', 'x-secret']) assert.equal(result.headers[key], undefined);
});

test('upload reaches observer before browser finishes (no buffering), accepts a 100 MB stream', async t => {
  let notify;
  const first = new Promise(resolve => { notify = resolve; });
  const expected = 100_000_000;
  const base = await setup(t, (req, res) => {
    let bytes = 0;
    req.on('data', chunk => { bytes += chunk.length; notify(); });
    req.on('end', () => { assert.equal(bytes, expected); res.end('{}'); });
  });
  const req = http.request(base + '/', { method: 'POST', headers: { 'content-length': expected } });
  const result = new Promise((resolve, reject) => {
    req.on('response', res => { res.resume(); res.on('end', () => resolve(res.statusCode)); });
    req.on('error', reject);
  });
  req.write(Buffer.alloc(1024));
  await first;
  const chunk = Buffer.alloc(1_000_000, 123);
  for (let bytes = 1024; bytes < expected;) {
    const part = chunk.subarray(0, Math.min(chunk.length, expected - bytes));
    bytes += part.length;
    if (!req.write(part)) await once(req, 'drain');
  }
  req.end();
  assert.equal(await result, 200);
});

test('browser cancellation destroys upstream upload', async t => {
  let opened, closed;
  const first = new Promise(resolve => { opened = resolve; });
  const aborted = new Promise(resolve => { closed = resolve; });
  const base = await setup(t, (req, res) => {
    req.on('error', () => {});
    req.on('data', () => opened());
    req.on('aborted', () => closed());
  });
  const req = http.request(base, { method: 'POST' });
  req.on('error', () => {});
  req.write(Buffer.alloc(1024));
  await first;
  req.destroy();
  await aborted;
});

test('browser cancellation destroys upstream response stream', async t => {
  let closed;
  const aborted = new Promise(resolve => { closed = resolve; });
  const base = await setup(t, (req, res) => {
    res.write('first');
    res.on('close', () => closed());
  });
  await new Promise((resolve, reject) => {
    const req = http.get(base, res => { res.once('data', () => { res.destroy(); resolve(); }); });
    req.on('error', reject);
  });
  await aborted;
});

test('bounded errors: redirects, backend auth and huge diagnostic bodies never escape', async t => {
  for (const status of [302, 401, 403, 422, 500]) {
    const base = await setup(t, (req, res) => {
      res.writeHead(status, { location: 'https://observer.internal/secret', 'set-cookie': `token=${secret}`,
        'content-type': 'text/html', 'x-error': secret });
      res.end((secret + ' https://observer.internal/').repeat(20_000));
    });
    const result = await request(base);
    assert.equal(result.status, status === 422 ? 422 : 502);
    assert.ok(result.body.length < 100);
    assert.ok(!JSON.stringify(result).includes(secret));
    assert.ok(!JSON.stringify(result).includes('observer.internal'));
    assert.equal(result.headers.location, undefined);
  }
});

test('connection failure and deadline have fixed, bounded responses', async t => {
  const base = await setup(t, (req, res) => {}, { timeoutMs: 40 });
  assert.equal((await request(base)).status, 504);
  const { proxy } = createCaptureProxy({ url: 'http://127.0.0.1:1', token: secret });
  const unavailable = await listen(t, proxy);
  const result = await request(unavailable);
  assert.equal(result.status, 502);
  assert.equal(result.body, '{"error":"capture_unavailable"}');
});

test('length and chunked limits are enforced without buffering; unrelated paths do not reach observer', async t => {
  let calls = 0;
  const base = await setup(t, (req, res) => { calls++; req.resume(); }, { maxUploadBytes: 100 });
  assert.equal((await request(base, { method: 'POST', headers: { 'content-length': 101 } }, [Buffer.alloc(101)])).status, 413);
  assert.equal(calls, 0);
  assert.equal((await request(base, { method: 'POST' }, [Buffer.alloc(101)])).status, 413);
  for (const path of ['/admin', '//evil.test/', '/assets/../secret', '/assets/not-real.js']) assert.equal((await request(base + path)).status, 404);
});

test('only known static assets proxy, with HEAD and query preserved', async t => {
  const seen = [];
  const base = await setup(t, (req, res) => {
    seen.push([req.method, req.url]);
    res.writeHead(200, { 'content-type': 'text/javascript', 'content-length': '2' });
    res.end('{}');
  });
  const head = await request(base + '/assets/app.js?v=123', { method: 'HEAD' });
  assert.equal(head.status, 200);
  assert.equal(head.body, '');
  assert.equal((await request(base + '/assets/queue.js')).body, '{}');
  assert.deepEqual(seen, [['HEAD', '/assets/app.js?v=123'], ['GET', '/assets/queue.js']]);
});

test('oversize upstream content-length is rejected before any backend content reaches browser', async t => {
  const base = await setup(t, (req, res) => {
    res.writeHead(200, { 'content-length': '1000', 'set-cookie': `token=${secret}` });
    res.end(secret);
  }, { maxResponseBytes: 100 });
  const result = await request(base);
  assert.equal(result.status, 502);
  assert.equal(result.body, '{"error":"capture_unavailable"}');
  assert.equal(result.headers['set-cookie'], undefined);
});

test('local static exact map serves fresh UI, MIME and observer-equivalent security headers without backend calls', async t => {
  const webRoot = new URL('../../apps/web/', import.meta.url).pathname;
  let seen = 0;
  const base = await setup(t, (req, res) => { seen++; res.end('backend'); }, { webRoot });
  const files = [
    ['/', 'index.html', 'text/html; charset=utf-8'],
    ['/assets/app.js', 'app.js', 'text/javascript; charset=utf-8'],
    ['/assets/capture-state.js', 'capture-state.js', 'text/javascript; charset=utf-8'],
    ['/assets/queue.js', 'queue.js', 'text/javascript; charset=utf-8'],
    ['/assets/style.css', 'style.css', 'text/css; charset=utf-8'],
    ['/assets/icon.svg', 'icon.svg', 'image/svg+xml'],
    ['/assets/manifest.webmanifest', 'manifest.webmanifest', 'application/manifest+json'],
    ['/assets/notices.js', 'notices.js', 'text/javascript; charset=utf-8'],
  ];
  for (const [path, file, mime] of files) {
    const result = await request(base + path + '?v=latest');
    assert.equal(result.status, 200);
    assert.equal(result.body, readFileSync(join(webRoot, file), 'utf8'));
    assert.equal(result.headers['content-type'], mime);
    assert.equal(Number(result.headers['content-length']), Buffer.byteLength(result.body));
    assert.equal(result.headers['content-security-policy'], "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; media-src 'self' blob:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'");
    assert.equal(result.headers['permissions-policy'], 'camera=(self), microphone=(self), geolocation=(self), screen-wake-lock=(self)');
    assert.equal(result.headers['cache-control'], 'no-store');
    assert.equal(result.headers['referrer-policy'], 'no-referrer');
    assert.equal(result.headers['x-content-type-options'], 'nosniff');
    const head = await request(base + path, { method: 'HEAD' });
    assert.equal(head.status, 200);
    assert.equal(head.body, '');
    assert.equal(head.headers['content-length'], result.headers['content-length']);
  }
  assert.equal(seen, 0);
});

test('local files deny traversal, encoding, symlinks, directories and oversized files', async t => {
  const webRoot = mkdtempSync(join(tmpdir(), 'capture-static-'));
  t.after(() => rmSync(webRoot, { recursive: true, force: true }));
  writeFileSync(join(webRoot, 'index.html'), 'x'.repeat(33));
  writeFileSync(join(webRoot, 'secret'), 'private');
  symlinkSync(join(webRoot, 'secret'), join(webRoot, 'app.js'));
  mkdirSync(join(webRoot, 'queue.js'));
  let seen = 0;
  const base = await setup(t, (req, res) => { seen++; res.end(); }, { webRoot, maxResponseBytes: 32 });
  // Use raw HTTP path; URL parsers otherwise normalize dot segments before sending.
  for (const path of ['/assets/../secret', '/assets/%2e%2e/secret', '/assets/%2Fapp.js', '/assets/app.js/extra',
    '/assets/./app.js', '/assets/app.js%00', '/assets/app.js\\secret', '/assets/index.html', '/index.html',
    '/assets/.env', '/assets/tests/session.test.mjs', '//evil.test/', '/assets/%61pp.js']) {
    assert.equal((await request(base, { path })).status, 404, path);
  }
  assert.equal((await request(base + '/')).status, 502);
  assert.equal((await request(base + '/assets/app.js')).status, 404);
  assert.equal((await request(base + '/assets/queue.js')).status, 502);
  assert.equal((await request(base + '/assets/style.css')).status, 404);
  assert.equal((await request(base + '/', { method: 'PUT' })).status, 404);
  assert.equal(seen, 0);
  assert.throws(() => createCaptureProxy({ url: base, token: secret, webRoot: join(webRoot, 'secret') }), /Invalid capture proxy configuration/);
});

test('local UI still streams POST and only exact latest-observation GET; preserves conditional ETag and 204/304', async t => {
  const webRoot = new URL('../../apps/web/', import.meta.url).pathname;
  const seen = [];
  const base = await setup(t, (req, res) => {
    assert.equal(req.headers.authorization, `Bearer ${secret}`);
    assert.equal(req.headers.cookie, undefined);
    seen.push([req.method, req.url]);
    if (req.method === 'POST') {
      let body = '';
      req.on('data', chunk => { body += chunk; });
      req.on('end', () => { assert.equal(body, 'streamed-video'); res.end('{}'); });
    } else if (req.headers['if-none-match']) {
      assert.equal(req.headers['if-none-match'], '"latest-1"');
      res.writeHead(304, { etag: '"latest-1"' }); res.end();
    } else { res.writeHead(204); res.end(); }
  }, { webRoot });
  assert.equal((await request(base + '/', { method: 'POST' }, ['streamed-', 'video'])).status, 200);
  assert.equal((await request(base + '/v1/observations/latest')).status, 204);
  const result = await request(base + '/v1/observations/latest', { headers: { 'if-none-match': '"latest-1"', cookie: 'private' } });
  assert.equal(result.status, 304);
  assert.equal(result.headers.etag, '"latest-1"');
  for (const path of ['/v1/observations', '/v1/observations/latest/', '/v1/observations/latest/secret', '/v1/observations/other']) {
    assert.equal((await request(base + path)).status, 404);
  }
  assert.equal((await request(base + '/v1/observations/latest', { method: 'POST' })).status, 404);
  assert.equal((await request(base + '/v1/observations/latest', { method: 'HEAD' })).status, 404);
  assert.equal(seen.length, 3);
});

test('TLS requires an explicit trusted CA, validates hostname, rejects unknown certificates', async t => {
  const dir = mkdtempSync(join(tmpdir(), 'capture-tls-'));
  t.after(() => rmSync(dir, { recursive: true, force: true }));
  function cert(name, subject) {
    const key = join(dir, `${name}.key`), pem = join(dir, `${name}.pem`);
    execFileSync('openssl', ['req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-keyout', key, '-out', pem,
      '-days', '1', '-subj', `/CN=${subject}`, '-addext', `subjectAltName=${subject === '127.0.0.1' ? 'IP' : 'DNS'}:${subject}`], { stdio: 'ignore' });
    return { key: readFileSync(key), cert: readFileSync(pem), pem };
  }
  const trusted = cert('trusted', '127.0.0.1');
  const other = cert('other', '127.0.0.1');
  const mismatch = cert('mismatch', 'observer.invalid');
  const url = await listen(t, (req, res) => res.end('trusted'), trusted);
  assert.throws(() => createCaptureProxy({ url, token: secret }), /Invalid capture proxy configuration/);
  for (const [target, caFile, status] of [
    [url, trusted.pem, 200], [url, other.pem, 502],
    [await listen(t, (req, res) => res.end('wrong host'), mismatch), mismatch.pem, 502],
  ]) {
    const { proxy } = createCaptureProxy({ url: target, token: secret, caFile });
    const gateway = await listen(t, proxy);
    const result = await request(gateway);
    assert.equal(result.status, status);
    if (status === 502) assert.equal(result.body, '{"error":"capture_unavailable"}');
  }
});
