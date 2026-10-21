import { createHash } from 'node:crypto';
import { Readable, Transform } from 'node:stream';
import { pipeline } from 'node:stream/promises';

const TIMEOUT_MS = 5000;
const MAX_BYTES = 1024 * 1024;
const MAX_FRAME_BYTES = 2 * 1024 * 1024, MAX_FRAMES = 6;
const MAX_REFS = 10000;

export class ArchiveError extends Error {
  constructor(status, code, message) {
    super(message);
    this.name = 'ArchiveError';
    this.status = status;
    this.code = code;
  }
}
const fail = (status, code, message) => { throw new ArchiveError(status, code, message); };
const invalid = () => fail(400, 'invalid_archive_arguments', 'Invalid video arguments.');
async function waitForEvidence(promise, signal) {
  if (!signal) return promise;
  let abort;
  try {
    return await Promise.race([promise, new Promise((_, reject) => {
      abort = () => reject(signal.reason);
      signal.addEventListener('abort', abort, { once: true });
      if (signal.aborted) abort();
    })]);
  } finally { if (abort) signal.removeEventListener('abort', abort); }
}
// Aborting the wait prevents further archive work. The graph driver independently
// bounds its read transaction/connection; this does not claim to cancel Cypher.
function id(value) {
  if (typeof value !== 'string' || value === '.' || value === '..' || !/^[a-zA-Z0-9_:.\-]{1,256}$/.test(value)) invalid();
  return value;
}
function number(value) {
  if (value === null || value === undefined || value === '') return null;
  const n = typeof value === 'number' || typeof value === 'string' ? Number(value) : NaN;
  return Number.isFinite(n) && n >= 0 ? n : null;
}
function array(value) {
  if (typeof value === 'string') {
    if (Buffer.byteLength(value) > MAX_BYTES) invalid();
    try { value = JSON.parse(value); } catch { invalid(); }
  }
  if (value === undefined || value === null) return [];
  if (!Array.isArray(value) || value.length > MAX_REFS) invalid();
  return value;
}
// Date is used only for whole-second calendar conversion, never fractions.
function timestampUs(value) {
  if (typeof value !== 'string' || value.length > 64) return null;
  const m = /^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d{1,6}))?(Z|([+-])(\d{2}):(\d{2}))$/.exec(value);
  if (!m) return null;
  const base = Date.parse(m[1] + 'Z');
  if (!Number.isFinite(base) || new Date(base).toISOString().slice(0, 19) !== m[1]
      || (m[3] !== 'Z' && (Number(m[5]) > 23 || Number(m[6]) > 59))) return null;
  const zone = m[3] === 'Z' ? 0 : (m[4] === '+' ? 1 : -1) * (Number(m[5]) * 60 + Number(m[6])) * 60 * 1e6;
  const result = base * 1000 - zone + Number((m[2] || '').padEnd(6, '0'));
  return Number.isSafeInteger(result) ? result : null;
}
function isoUs(value) {
  if (!Number.isSafeInteger(value)) invalid();
  const seconds = Math.floor(value / 1e6);
  return new Date(seconds * 1000).toISOString().replace(/\.000Z$/, '.' + String(value - seconds * 1e6).padStart(6, '0') + 'Z');
}
function date(value) {
  const micros = timestampUs(value);
  if (micros === null) invalid();
  return isoUs(micros);
}
function filters(args) {
  const out = {};
  if (args.capture_id != null) out.capture_id = id(args.capture_id);
  for (const key of ['started_at', 'ended_at']) if (args[key] != null) out[key] = date(args[key]);
  if (out.started_at && out.ended_at && out.started_at > out.ended_at) invalid();
  return out;
}
function interval(item) {
  const start = timestampUs(item.captured_at);
  const duration = number(item.duration_seconds);
  const durationUs = duration === null ? NaN : Math.round(duration * 1e6);
  return start !== null && Number.isSafeInteger(durationUs) && Number.isSafeInteger(start + durationUs)
    ? [start, start + durationUs] : null;
}
function matches(item, f) {
  if (f.capture_id && item.capture_id !== f.capture_id) return false;
  const span = interval(item);
  return (!f.started_at || (span && span[1] >= timestampUs(f.started_at)))
    && (!f.ended_at || (span && span[0] <= timestampUs(f.ended_at)));
}

// Interface: new ArchiveClient({ url, token, publicBaseUrl = '' }).
// listVideoClips(args = {}, { store } = {}) -> { items, has_more, next_cursor, missing_media_ids }.
// getVideoClip({ media_id, entity_id? }, { store } = {}) -> metadata with media_refs.
// store.getEntity(id) must return graph.mjs's { id, properties }. No graph traversal.
export class ArchiveClient {
  constructor({ url, token, publicBaseUrl = '' } = {}) {
    try {
      this.base = new URL(url);
      if (!['http:', 'https:'].includes(this.base.protocol) || this.base.username || this.base.password
          || this.base.search || this.base.hash) throw new Error();
      if (typeof token !== 'string' || !token || /[\r\n]/.test(token)) throw new Error();
      if (publicBaseUrl) {
        const publicUrl = new URL(publicBaseUrl);
        if (!['http:', 'https:'].includes(publicUrl.protocol) || publicUrl.username || publicUrl.password
            || publicUrl.search || publicUrl.hash || publicUrl.origin === this.base.origin) throw new Error();
      }
      this.publicBaseUrl = publicBaseUrl.replace(/\/$/, '');
      this.token = token;
    } catch {
      fail(503, 'archive_unavailable', 'Video archive is not configured.');
    }
  }

  // Authenticated graph-origin downloads use a separate private archive credential upstream.
  // This method owns the response, including sanitized failures; never JSON-wrap it.
  async proxyOriginal(req, res, mediaId) {
    const sendError = (status) => {
      if (res.headersSent) { res.destroy(); return; }
      for (const name of res.getHeaderNames()) res.removeHeader(name);
      res.writeHead(status, { 'Content-Type': 'application/json', 'Cache-Control': 'private, no-store' });
      res.end(req.method === 'HEAD' ? undefined : JSON.stringify({ error: 'Video original request failed.' }));
    };
    let controller, timer;
    const abort = () => controller?.abort();
    const resetIdle = () => { clearTimeout(timer); timer = setTimeout(abort, 30000); };
    try {
      id(mediaId);
      if (!['GET', 'HEAD'].includes(req.method)) { sendError(405); return; }
      const headers = { Authorization: `Bearer ${this.token}` };
      const range = req.headers?.range;
      if (range != null) {
        if (typeof range !== 'string' || range.length > 128 || !/^bytes=(?:\d{1,20}-\d{0,20}|-\d{1,20})$/.test(range)) {
          sendError(400); return;
        }
        headers.Range = range;
      }
      const ifRange = req.headers?.['if-range'];
      if (ifRange != null) {
        if (typeof ifRange !== 'string' || ifRange.length > 256 || /[\r\n]/.test(ifRange)
            || !( /^(?:W\/)?"[\x21\x23-\x7e]*"$/.test(ifRange)
              || /^\w{3}, \d{2} \w{3} \d{4} \d{2}:\d{2}:\d{2} GMT$/.test(ifRange))) {
          sendError(400); return;
        }
        headers['If-Range'] = ifRange;
      }
      controller = new AbortController();
      resetIdle();
      res.once('close', abort);
      const upstream = await fetch(new URL(this.base.href.replace(/\/$/, '')
        + '/v1/archive/media/' + encodeURIComponent(mediaId) + '/original'), {
        method: req.method, headers, redirect: 'error', signal: controller.signal,
      });
      if (![200, 206, 416, 304].includes(upstream.status)) {
        await upstream.body?.cancel();
        sendError(upstream.status === 404 ? 404 : 502); return;
      }
      const forwarded = { 'Cache-Control': 'private, no-store' };
      const safeHeaders = {
        'content-length': /^\d{1,20}$/,
        'content-type': /^(?:video\/(?:mp4|quicktime|webm)|application\/octet-stream)(?:;[^\r\n]{0,80})?$/i,
        'content-range': /^bytes (?:\d+-\d+|\*)\/\d+$/,
        'accept-ranges': /^bytes$/,
        'etag': /^(?:W\/)?"[a-fA-F0-9]{1,128}"$/,
        'x-archive-sha256': /^[a-fA-F0-9]{64}$/,
        'x-archive-media-id': /^[a-zA-Z0-9_:.\-]{1,256}$/,
        'last-modified': /^\w{3}, \d{2} \w{3} \d{4} \d{2}:\d{2}:\d{2} GMT$/,
      };
      for (const [name, pattern] of Object.entries(safeHeaders)) {
        const value = upstream.headers.get(name);
        if (value && pattern.test(value) && !value.includes(this.token) && !value.includes(this.base.hostname)) forwarded[name] = value;
      }
      // Do not forward arbitrary error bodies (often HTML containing paths).
      if (upstream.status === 416) {
        await upstream.body?.cancel();
        delete forwarded['content-length'];
        res.writeHead(416, forwarded); res.end(); return;
      }
      res.writeHead(upstream.status, forwarded);
      if (req.method === 'HEAD' || upstream.status === 304 || !upstream.body) {
        await upstream.body?.cancel(); res.end(); return;
      }
      resetIdle();
      const activity = new Transform({ transform(chunk, encoding, callback) {
        resetIdle(); callback(null, chunk);
      } });
      await pipeline(Readable.fromWeb(upstream.body), activity, res);
    } catch (error) { sendError(error instanceof ArchiveError && error.status === 400 ? 400 : 502); }
    finally {
      clearTimeout(timer);
      res.off('close', abort);
    }
  }

  async request(path, query = {}, { signal } = {}) {
    const controller = new AbortController();
    const requestSignal = signal ? AbortSignal.any([controller.signal, signal]) : controller.signal;
    const timeout = setTimeout(() => controller.abort(), TIMEOUT_MS);
    try {
      const url = new URL(this.base.href.replace(/\/$/, '') + '/v1/archive/' + path);
      for (const [key, value] of Object.entries(query)) if (value != null) url.searchParams.set(key, String(value));
      const response = await fetch(url, { headers: { Authorization: `Bearer ${this.token}`, Accept: 'application/json' },
        redirect: 'error', signal: requestSignal });
      if (!response.ok) {
        await response.body?.cancel();
        if (response.status === 404) fail(404, 'archive_media_not_found', 'Video clip not found.');
        fail(502, 'archive_request_failed', 'Video archive request failed.');
      }
      if (!/^application\/json\b/i.test(response.headers.get('content-type') || '')
          || Number(response.headers.get('content-length')) > MAX_BYTES) {
        await response.body?.cancel();
        fail(502, 'archive_invalid_response', 'Invalid video archive response.');
      }
      const chunks = [];
      let size = 0;
      for await (const chunk of response.body) {
        size += chunk.length;
        if (size > MAX_BYTES) {
          controller.abort();
          fail(502, 'archive_invalid_response', 'Invalid video archive response.');
        }
        chunks.push(chunk);
      }
      return JSON.parse(Buffer.concat(chunks).toString('utf8'));
    } catch (error) {
      if (error instanceof ArchiveError) throw error;
      fail(502, 'archive_request_failed', 'Video archive request failed.');
    } finally { clearTimeout(timeout); }
  }

  async evidence(entityId, store, signal) {
    if (signal?.aborted) fail(502, 'archive_request_failed', 'Video archive request failed.');
    if (entityId == null) return null;
    if (typeof entityId !== 'string' || !entityId.length || entityId.length > 512
        || /[\u0000-\u001f\u007f-\u009f]/u.test(entityId)) invalid();
    if (typeof store?.getEntity !== 'function') fail(503, 'graph_unavailable', 'Graph entity lookup is unavailable.');
    let entity;
    try { entity = await waitForEvidence(store.getEntity(entityId), signal); } catch (error) {
      if (error?.status === 404) fail(404, 'entity_not_found', 'Entity not found.');
      fail(502, 'graph_lookup_failed', 'Graph entity lookup failed.');
    }
    if (!entity) fail(404, 'entity_not_found', 'Entity not found.');
    const p = entity.properties;
    if (!p || typeof p !== 'object') fail(502, 'graph_invalid_entity', 'Invalid graph entity.');
    const ids = new Set(array(p.media_ids).map(id));
    if (p.media_id != null) ids.add(id(p.media_id));
    const refs = [];
    if (p.media_id && number(p.offset_start) !== null) refs.push({ media_id: p.media_id, offset: number(p.offset_start), source: 'entity_offset' });
    for (const crop of array(p.crops)) {
      if (!crop || typeof crop !== 'object' || !crop.media_id) continue;
      ids.add(id(crop.media_id));
      const offset = number(crop.offset ?? crop.start_offset_seconds);
      if (offset !== null) {
        const geometry = {};
        if (Array.isArray(crop.box) && crop.box.length === 4
            && crop.box.every((v) => typeof v === 'number' && Number.isFinite(v) && v >= 0)
            && crop.box[2] > 0 && crop.box[3] > 0
            && Number.isSafeInteger(crop.w) && crop.w > 0 && Number.isSafeInteger(crop.h) && crop.h > 0
            && crop.box[0] + crop.box[2] <= crop.w && crop.box[1] + crop.box[3] <= crop.h) {
          Object.assign(geometry, { box: [...crop.box], w: crop.w, h: crop.h });
        }
        refs.push({ media_id: crop.media_id, offset, source: 'entity_crop', ...geometry });
      }
    }
    if (ids.size > MAX_REFS) invalid();
    const start = number(p.t_start_us), end = number(p.t_end_us);
    if ((start !== null && !Number.isSafeInteger(start)) || (end !== null && !Number.isSafeInteger(end))) invalid();
    const capture = p.capture_id == null ? null : id(p.capture_id);
    // Kind affects evidence interpretation only, never media membership/access.
    return { ids: [...ids].sort(), refs, capture, continuous: ['speech_segment', 'episode'].includes(p.kind), start,
      end: start !== null && end !== null && end >= start ? end : null };
  }

  mediaRefs(item, evidence) {
    const refs = [];
    if (evidence) {
      for (const ref of evidence.refs.filter((r) => r.media_id === item.media_id)) {
        if (ref.offset <= item.duration_seconds) refs.push({ media_id: item.media_id, kind: 'point',
          start_offset_seconds: ref.offset, source: ref.source,
          ...(ref.box ? { box: ref.box, w: ref.w, h: ref.h } : {}) });
      }
      const span = interval(item);
      if (span && evidence.continuous && evidence.capture === item.capture_id && evidence.start !== null && evidence.end !== null) {
        const start = Math.max(span[0], evidence.start), end = Math.min(span[1], evidence.end);
        if (end >= start) refs.push({ media_id: item.media_id, kind: start === end ? 'point' : 'interval',
          start_offset_seconds: (start - span[0]) / 1e6, ...(end > start ? { end_offset_seconds: (end - span[0]) / 1e6 } : {}), source: 'entity_capture_interval' });
      }
    }
    return refs.length ? refs : [{ media_id: item.media_id, kind: 'whole_clip', source: evidence ? 'entity_media_membership' : 'archive_media' }];
  }

  metadata(raw, evidence) {
    if (!raw || typeof raw !== 'object' || typeof raw.media_id !== 'string'
        || raw.media_id === '.' || raw.media_id === '..'
        || !/^[a-zA-Z0-9_:.\-]{1,256}$/.test(raw.media_id) || !interval(raw)) {
      fail(502, 'archive_invalid_response', 'Invalid video archive response.');
    }
    // Deliberate allowlists: archive metadata/receipt can contain caller-supplied
    // arbitrary data. Never forward paths, URLs, command lines or opaque blobs.
    const cleanString = (v) => typeof v === 'string' && v.length <= 256 && !v.includes(this.token)
      && !v.includes(this.base.hostname) && !/(?:https?:|file:|[\\/]|bearer\s)/i.test(v);
    const result = { media_id: raw.media_id };
    for (const key of ['capture_id', 'captured_at', 'sha256', 'archived_at', 'time_semantics']) {
      if (cleanString(raw[key])) result[key] = raw[key];
    }
    for (const key of ['clip_seq', 'duration_seconds', 'size_bytes']) {
      if (typeof raw[key] === 'number' && Number.isFinite(raw[key]) && raw[key] >= 0) result[key] = raw[key];
    }
    if (typeof raw.media_persisted === 'boolean') result.media_persisted = raw.media_persisted;
    const metadata = {};
    for (const key of ['width', 'height', 'fps', 'frame_rate', 'audio_sample_rate', 'audio_channels', 'rotation']) {
      if (typeof raw.metadata?.[key] === 'number' && Number.isFinite(raw.metadata[key])) metadata[key] = raw.metadata[key];
    }
    for (const key of ['camera_facing', 'video_codec', 'audio_codec', 'container']) {
      if (cleanString(raw.metadata?.[key])) metadata[key] = raw.metadata[key];
    }
    const contentType = raw.metadata?.content_type;
    if (['video/mp4', 'video/quicktime', 'video/webm'].includes(contentType)) metadata.content_type = contentType;
    result.metadata = metadata;
    result.original_url = `${this.publicBaseUrl}/v1/archive/media/${encodeURIComponent(raw.media_id)}/original`;
    result.original_requires_auth = true;
    result.media_refs = this.mediaRefs(raw, evidence);
    return result;
  }

  // Still frames for get_video_frames: media_id with offsets (seconds from that video's start), or capture_id with
  // absolute times, each resolved to the clip covering it. The JPEG bytes go back to the MCP caller as image content,
  // so a client whose token only reaches /mcp still sees what the camera showed.
  async getFrames(args = {}, { store, signal } = {}) {
    const size = args.size ?? 768;
    if (!Number.isInteger(size) || size < 160 || size > 1280) invalid();
    const wanted = [];
    if (args.media_id != null) {
      if (args.capture_id != null || args.at != null || !Array.isArray(args.offsets) || !args.offsets.length || args.offsets.length > MAX_FRAMES) invalid();
      const mediaId = id(args.media_id);
      for (const offset of args.offsets) {
        if (typeof offset !== 'number' || !Number.isFinite(offset) || offset < 0 || offset > 3600) invalid();
        wanted.push({ media_id: mediaId, offset_s: offset });
      }
    } else {
      if (args.capture_id == null || args.offsets != null || !Array.isArray(args.at) || !args.at.length || args.at.length > MAX_FRAMES) invalid();
      for (const when of args.at) {
        const atUs = timestampUs(date(when));
        // Clips are seconds long: the one covering the time started at most a minute before it.
        const page = await this.listVideoClips({ capture_id: args.capture_id, started_at: isoUs(atUs - 60e6), ended_at: isoUs(atUs), limit: 20 },
          { store, signal });
        const covering = (page.items ?? []).find((item) => { const span = interval(item); return span && span[0] <= atUs && atUs < span[1]; });
        if (!covering) fail(404, 'archive_media_not_found', 'No video covers that time.');
        // The video track can end a little before the clip's nominal duration: stay a quarter second inside it.
        const [start, end] = interval(covering);
        wanted.push({ media_id: covering.media_id, offset_s: Math.max(0, Math.min(atUs, end - 250_000) - start) / 1e6, at: isoUs(atUs) });
      }
    }
    const frames = [];
    for (const item of wanted) frames.push({ ...item, size, jpeg: await this.frame(item.media_id, item.offset_s, size, signal) });
    return frames;
  }

  async frame(mediaId, seconds, size, signal) {
    const controller = new AbortController();
    const requestSignal = signal ? AbortSignal.any([controller.signal, signal]) : controller.signal;
    const timeout = setTimeout(() => controller.abort(), 25000);  // the archive decodes with ffmpeg (bounded at 20 s)
    try {
      const url = new URL(this.base.href.replace(/\/$/, '') + '/v1/archive/media/' + encodeURIComponent(mediaId) + '/frame');
      url.searchParams.set('t', seconds.toFixed(3));
      url.searchParams.set('size', String(size));
      const response = await fetch(url, { headers: { Authorization: `Bearer ${this.token}`, Accept: 'image/jpeg' }, redirect: 'error', signal: requestSignal });
      if (!response.ok) {
        await response.body?.cancel();
        if (response.status === 404) fail(404, 'archive_media_not_found', 'Video clip not found.');
        if (response.status === 422) fail(422, 'archive_frame_outside_media', 'That time is outside this video.');
        fail(502, 'archive_request_failed', 'Video archive request failed.');
      }
      if (!/^image\/jpeg\b/i.test(response.headers.get('content-type') || '') || Number(response.headers.get('content-length')) > MAX_FRAME_BYTES) {
        await response.body?.cancel();
        fail(502, 'archive_invalid_response', 'Invalid video archive response.');
      }
      const chunks = [];
      let total = 0;
      for await (const chunk of response.body) {
        total += chunk.length;
        if (total > MAX_FRAME_BYTES) { controller.abort(); fail(502, 'archive_invalid_response', 'Invalid video archive response.'); }
        chunks.push(chunk);
      }
      return Buffer.concat(chunks);
    } catch (error) {
      if (error instanceof ArchiveError) throw error;
      fail(502, 'archive_request_failed', 'Video archive request failed.');
    } finally { clearTimeout(timeout); }
  }

  async getVideoClip(args = {}, { store, signal } = {}) {
    const mediaId = id(args.media_id);
    const evidence = await this.evidence(args.entity_id, store, signal);
    if (evidence?.ids.length && !evidence.ids.includes(mediaId)) fail(404, 'entity_media_not_found', 'Video clip is not referenced by this entity.');
    if (evidence && !evidence.ids.length && !(evidence.capture && evidence.start !== null && evidence.end !== null)) {
      fail(404, 'entity_media_not_found', 'Video clip is not referenced by this entity.');
    }
    const raw = await this.request('media/' + encodeURIComponent(mediaId), {}, { signal });
    if (raw?.media_id !== mediaId) fail(502, 'archive_invalid_response', 'Invalid video archive response.');
    if (evidence && !evidence.ids.length && !matches(raw, { capture_id: evidence.capture,
      started_at: isoUs(evidence.start), ended_at: isoUs(evidence.end) })) {
      fail(404, 'entity_media_not_found', 'Video clip is not referenced by this entity.');
    }
    return this.metadata(raw, evidence);
  }

  async listVideoClips(args = {}, { store, signal } = {}) {
    const controller = new AbortController();
    const pageSignal = signal ? AbortSignal.any([controller.signal, signal]) : controller.signal;
    const timeout = setTimeout(() => controller.abort(), 10000);
    try {
      pageSignal.throwIfAborted();
      return await this.listPage(args, { store, signal: pageSignal });
    } catch (error) {
      if (pageSignal.aborted) fail(502, 'archive_request_failed', 'Video archive request failed.');
      throw error;
    } finally { clearTimeout(timeout); controller.abort(); }
  }

  async listPage(args, { store, signal }) {
    const f = filters(args), limit = args.limit ?? 20;
    if (!Number.isInteger(limit) || limit < 1 || limit > 100) invalid();
    if (args.cursor != null && (typeof args.cursor !== 'string' || args.cursor.length > 512)) invalid();
    const evidence = await this.evidence(args.entity_id, store, signal);
    const fingerprint = createHash('sha256').update(JSON.stringify([args.entity_id ?? null, evidence, f])).digest('hex').slice(0, 24);
    let after;
    if (args.cursor != null && !evidence?.ids.length) {
      const match = /^archive:([a-f0-9]{24}):([A-Za-z0-9_-]{1,342})$/.exec(args.cursor);
      if (!match || match[1] !== fingerprint) invalid();
      after = Buffer.from(match[2], 'base64url').toString('utf8');
      id(after);
      if (Buffer.from(after).toString('base64url') !== match[2]) invalid();
    }
    if (evidence?.ids.length) {
      let offset = 0;
      if (args.cursor != null) {
        const match = /^entity:([a-f0-9]{24}):(\d{1,5})$/.exec(args.cursor);
        if (!match || match[1] !== fingerprint) invalid();
        offset = Number(match[2]);
        if (offset >= evidence.ids.length) invalid();
      }
      const selected = evidence.ids.slice(offset, offset + limit), items = [], missing = [];
      // Five concurrent JSON reads maximum; at most limit IDs examined per page,
      // including missing/filtered clips, so sparse references cannot fan out.
      for (let i = 0; i < selected.length; i += 5) {
        signal.throwIfAborted();
        const batch = await Promise.all(selected.slice(i, i + 5).map(async (mediaId) => {
          try {
            const raw = await this.request('media/' + encodeURIComponent(mediaId), {}, { signal });
            if (raw?.media_id !== mediaId) fail(502, 'archive_invalid_response', 'Invalid video archive response.');
            const item = this.metadata(raw, evidence);
            return matches(raw, f) ? { item } : {};
          } catch (error) {
            if (error instanceof ArchiveError && error.status === 404) return { missing: mediaId };
            throw error;
          }
        }));
        for (const entry of batch) { if (entry.item) items.push(entry.item); if (entry.missing) missing.push(entry.missing); }
      }
      const more = offset + selected.length < evidence.ids.length;
      return { items, missing_media_ids: missing, has_more: more,
        next_cursor: more ? `entity:${fingerprint}:${offset + selected.length}` : null };
    }
    if (evidence) {
      if (!(evidence.capture && evidence.start !== null && evidence.end !== null)) {
        return { items: [], missing_media_ids: [], has_more: false, next_cursor: null };
      }
      if (f.capture_id && f.capture_id !== evidence.capture) return { items: [], missing_media_ids: [], has_more: false, next_cursor: null };
      f.capture_id = evidence.capture;
      const start = isoUs(evidence.start), end = isoUs(evidence.end);
      f.started_at = f.started_at && f.started_at > start ? f.started_at : start;
      f.ended_at = f.ended_at && f.ended_at < end ? f.ended_at : end;
      if (f.started_at > f.ended_at) return { items: [], missing_media_ids: [], has_more: false, next_cursor: null };
    }
    signal.throwIfAborted();
    const page = await this.request('media', { ...f, limit, after }, { signal });
    if (!Array.isArray(page?.items) || page.items.length > limit || typeof page.has_more !== 'boolean'
        || (page.has_more && (typeof page.next_cursor !== 'string' || !/^[a-zA-Z0-9_:.\-]{1,256}$/.test(page.next_cursor)))) {
      fail(502, 'archive_invalid_response', 'Invalid video archive response.');
    }
    return { items: page.items.map((raw) => this.metadata(raw, evidence)), missing_media_ids: [],
      has_more: page.has_more,
      next_cursor: page.has_more ? `archive:${fingerprint}:${Buffer.from(page.next_cursor).toString('base64url')}` : null };
  }
}
