import { createHash } from 'node:crypto';
import { ApiError } from './graph.mjs';

// /v1/notices: the capture page's toasts (spoken namings and requests handed to Okou). Owner session required
// (server.mjs).
const NAMING_WINDOW_MS = 120_000, NAMING_LIMIT = 10;
const NAMINGS_QUERY = `
  MATCH (n:NamingEvent) WHERE coalesce(n.status, '') <> 'ignored'
    AND (n.created_at >= $since OR n.created_at >= datetime($since))
  RETURN n.id AS id, n.name AS name, n.status AS status, n.memory_id AS memory_id, n.created_at AS created_at
  ORDER BY n.created_at DESC LIMIT 50`;
// Requests the day agent handed to Okou ("已交给 Okou：…"), shown on the phone like namings; turns it decided were
// nothing new for Okou stay off the screen.
const ASSISTS_QUERY = `
  MATCH (a:AgentEvent) WHERE a.created_at >= $since AND coalesce(a.status, '') <> 'declined'
  RETURN a.id AS id, a.note AS note, a.status AS status, a.created_at AS created_at
  ORDER BY a.created_at DESC LIMIT 20`;
const ASSIST_STATUS = { sent: 'assist', not_connected: 'assist_unconnected' };

// created_at as epoch ms: ISO strings, Neo4j temporals (toString) or epoch numbers in s, ms or us.
function epochMs(value) {
  if (value == null) return null;
  if (typeof value.toNumber === 'function') value = value.toNumber();
  if (typeof value === 'bigint') value = Number(value);
  const ms = typeof value === 'number' ? (value < 1e11 ? value * 1e3 : value < 1e14 ? value : value / 1e3)
    : Date.parse(String(value).replace(/\[[^\]]*\]$/, ''));
  return Number.isFinite(ms) ? ms : null;
}

export function createNoticesHandler({ store, now = Date.now }) {
  // The capture page's toasts: spoken namings and requests handed to Okou in the last two minutes, newest first.
  async function notices() {
    const since = now() - NAMING_WINDOW_MS, iso = new Date(since).toISOString();
    const [namingRows, assistRows] = await store.read(async tx => [await tx.run(NAMINGS_QUERY, { since: iso }),
      await tx.run(ASSISTS_QUERY, { since: iso })]);
    const namings = [...namingRows.records.map(row => ({ id: row.get('id'), name: row.get('name') ?? null, status: row.get('status'),
      memory_id: row.get('memory_id') ?? null, at: epochMs(row.get('created_at')) })),
    ...assistRows.records.map(row => ({ id: row.get('id'), name: row.get('note') ?? null,
      status: ASSIST_STATUS[row.get('status')] ?? 'assist_failed', memory_id: null, at: epochMs(row.get('created_at')) }))]
      .filter(item => item.at != null && item.at >= since && item.status !== 'ignored')
      .sort((a, b) => b.at - a.at || String(b.id).localeCompare(String(a.id))).slice(0, NAMING_LIMIT)
      .map(({ at, ...item }) => ({ ...item, created_at: new Date(at).toISOString() }));
    return { generated_at: new Date(now()).toISOString(), namings };
  }

  return {
    matches: pathname => pathname === '/v1/notices',
    async handle(req, res, url) {
      if (!['GET', 'HEAD'].includes(req.method)) throw new ApiError(405, 'method_not_allowed', 'Use GET');
      if (url.search) throw new ApiError(400, 'invalid_request', 'Query parameters are not supported');
      // Polled every 5 s and mostly unchanged: the page sends back the ETag and gets 304 until something changes.
      const body = await notices(), { generated_at: _, ...stable } = body;
      const etag = `"${createHash('sha256').update(JSON.stringify(stable)).digest('base64url').slice(0, 27)}"`;
      if (req.headers['if-none-match'] === etag) {
        res.writeHead(304, { ETag: etag, 'Cache-Control': 'no-store' });
        return res.end();
      }
      res.writeHead(200, { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store', ETag: etag });
      return res.end(req.method === 'HEAD' ? undefined : JSON.stringify(body));
    },
  };
}
