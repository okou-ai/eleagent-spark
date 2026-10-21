import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StreamableHTTPServerTransport } from '@modelcontextprotocol/sdk/server/streamableHttp.js';
import { z } from 'zod/v3';
import { ApiError } from './graph.mjs';
import { ArchiveError } from './archive.mjs';
import { cursors, parseFilter } from './validation.mjs';

export const TOOL_SCOPES = Object.freeze({ search_entities: 'graph:read', get_entity: 'graph:read', list_entity_links: 'graph:read', get_subgraph: 'graph:read', list_video_clips: 'media:read', get_video_clip: 'media:read', get_video_frames: 'media:read' });
// A tool result that is MCP content itself (images), not JSON to wrap.
const CONTENT = Symbol('mcp content');

const id = z.string().min(1).max(512).regex(/^[^\u0000-\u001f]+$/);
const mediaId = z.string().regex(/^media_[a-f0-9]{32}$/);
const page = { cursor: z.string().max(12 * 1024).optional() };
const canonical = (x) => JSON.stringify(Object.fromEntries(Object.keys(x).sort().map((k) => [k, x[k]])));
const annotations = { readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: false };
const scalar = z.union([z.string().max(16384), z.number().finite().refine((n) => !Number.isInteger(n) || Number.isSafeInteger(n)), z.boolean(), z.null()]);
const filterSchema = z.record(z.union([scalar, z.array(scalar).max(256)])).refine((x) => Object.keys(x).length <= 20);

export function createMcpHandler({ store, cursorKey, archive, allowedOrigins = process.env.MCP_ALLOWED_ORIGINS || '' }) {
  const cursor = cursors(cursorKey);
  const origins = new Set(allowedOrigins.split(',').map((s) => s.trim()).filter(Boolean));
  const graphTools = [
    ['search_entities', 'Search Neo4j entities by case-insensitive property text and exact property filters. All kinds are ordinary properties. Results are live, not a snapshot.', {
      search: z.string().max(512).default(''), filter: filterSchema.default({}), sort: id.default('id'),
      order: z.enum(['asc', 'desc']).default('asc'), limit: z.number().int().min(1).max(200).default(50), ...page,
    }, async ({ search, filter, sort, order, limit, cursor: value }) => {
      filter = parseFilter(JSON.stringify(filter));
      const scope = ['entities', search, canonical(filter), sort, order];
      const result = await store.listEntities({ search, filter, sort, order, limit, after: cursor.decode(value, scope) });
      return { entities: result.entities, next_cursor: cursor.encode(result.after, scope) };
    }],
    ['get_entity', 'Read one graph entity by its stable ID. Returns id and properties without interpreting model claims as verified facts.', { id }, async ({ id }) => store.getEntity(id)],
    ['list_entity_links', 'Read an entity\'s LINK relationships and adjacent entities, with bounded cursor pagination.', {
      id, limit: z.number().int().min(1).max(500).default(100), ...page,
    }, async ({ id, limit, cursor: value }) => {
      const scope = ['links', id];
      const after = cursor.decode(value, scope);
      if (after !== null) idSchemaCheck(after);
      const result = await store.listLinks(id, { limit, after });
      return { entities: result.entities, links: result.links, next_cursor: cursor.encode(result.after, scope) };
    }],
    ['get_subgraph', 'Expand graph seed IDs through LINK relationships. Explicit truncation and limits; no arbitrary Cypher or writes.', {
      ids: z.array(id).min(1).max(50), depth: z.number().int().min(0).max(3).default(1),
      max_entities: z.number().int().min(1).max(500).default(100), max_links: z.number().int().min(1).max(2000).default(200),
    }, async (args) => {
      args.ids = [...new Set(args.ids)];
      if (args.ids.length > args.max_entities) throw new ApiError(400, 'invalid_request', 'max_entities must accommodate seed IDs');
      return store.subgraph(args);
    }],
  ];
  const videoTools = [
    ['list_video_clips', 'List original archived videos by capture/time or graph entity ID. References distinguish representative points, evidence intervals, and whole clips. No inferred end offset is presented as exact evidence.', {
      capture_id: z.string().min(1).max(80).regex(/^[a-zA-Z0-9_-]+$/).optional(),
      started_at: z.string().datetime({ offset: true }).optional(), ended_at: z.string().datetime({ offset: true }).optional(),
      entity_id: id.optional(), limit: z.number().int().min(1).max(100).default(50), ...page,
    }, async (args, extra) => {
      // Coarse precheck only; ArchiveClient compares full microsecond precision.
      if (args.started_at && args.ended_at && Date.parse(args.started_at) > Date.parse(args.ended_at)) {
        throw new ApiError(400, 'invalid_request', 'started_at must not follow ended_at');
      }
      if (!archive) throw new ApiError(503, 'archive_unavailable', 'Archive is not configured');
      return archive.listVideoClips(args, { store, signal: extra.signal });
    }],
    ['get_video_clip', 'Get original video metadata and a protected same-origin download URL by media_id. Optionally scope evidence references to entity_id. Downloads require OAuth access with media:read; video bytes are not embedded in MCP output.', {
      media_id: mediaId, entity_id: id.optional(),
    }, async (args, extra) => {
      if (!archive) throw new ApiError(503, 'archive_unavailable', 'Archive is not configured');
      return archive.getVideoClip(args, { store, signal: extra.signal });
    }],
    ['get_video_frames', 'Still frames (JPEG images) of original archived videos: what the camera showed. Give media_id with offsets in seconds from that video\'s start (list_video_clips returns media IDs and times), or capture_id with absolute times (RFC 3339 with offset). Up to 6 frames per call; size is the longer side in pixels.', {
      media_id: mediaId.optional(), offsets: z.array(z.number().min(0).max(3600)).min(1).max(6).optional(),
      capture_id: z.string().min(1).max(80).regex(/^[a-zA-Z0-9_-]+$/).optional(),
      at: z.array(z.string().datetime({ offset: true })).min(1).max(6).optional(),
      size: z.number().int().min(160).max(1280).default(768),
    }, async (args, extra) => {
      if (!archive) throw new ApiError(503, 'archive_unavailable', 'Archive is not configured');
      const frames = await archive.getFrames(args, { store, signal: extra.signal });
      const listed = frames.map(({ jpeg, ...frame }) => frame);
      return { [CONTENT]: { content: [{ type: 'text', text: JSON.stringify({ frames: listed }) },
        ...frames.map((frame) => ({ type: 'image', data: frame.jpeg.toString('base64'), mimeType: 'image/jpeg' }))],
        structuredContent: { frames: listed } } };
    }],
  ];
  function buildServer() {
    const server = new McpServer({ name: 'eleagent', version: '0.1.0' }, {
      instructions: 'Read-only access to the Eleagent graph and original video references. Graph contents include fallible model interpretations. Graph tools require graph:read; media tools and original downloads require media:read. Never put access tokens in URLs.',
    });
    for (const [name, description, shape, run] of [...graphTools, ...videoTools]) {
      server.registerTool(name, { description, inputSchema: z.object(shape).strict(), annotations }, async (args, extra) => {
        try {
          const result = await run(args, extra);
          if (result?.[CONTENT]) return result[CONTENT];
          return { content: [{ type: 'text', text: JSON.stringify(result) }], structuredContent: result };
        } catch (error) {
          const known = error instanceof ApiError || error instanceof ArchiveError;
          const result = { error: { code: known ? error.code : 'service_unavailable',
            message: known ? error.message : 'Read could not be completed' } };
          return { isError: true, content: [{ type: 'text', text: JSON.stringify(result) }], structuredContent: result };
        }
      });
    }
    return server;
  }
  return {
    origins,
    async handle(req, res, body) {
      // Stateless Streamable HTTP: no session registry or server-initiated SSE stream.
      if (req.method !== 'POST') {
        res.writeHead(405, { Allow: 'POST', 'Content-Type': 'application/json' });
        return res.end(JSON.stringify({ jsonrpc: '2.0', id: null, error: { code: -32000, message: 'Use POST for stateless MCP' } }));
      }
      const server = buildServer();
      const transport = new StreamableHTTPServerTransport({ sessionIdGenerator: undefined, enableJsonResponse: true });
      res.once('close', () => { void transport.close(); void server.close(); });
      await server.connect(transport);
      await transport.handleRequest(req, res, body);
    },
  };
}
function idSchemaCheck(value) { id.parse(value); }
