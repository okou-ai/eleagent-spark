import assert from 'node:assert/strict';
import { createHash, createHmac, randomUUID } from 'node:crypto';
import { once } from 'node:events';
import http from 'node:http';
import { after, before, describe, test } from 'node:test';
import neo4j from 'neo4j-driver';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StreamableHTTPClientTransport } from '@modelcontextprotocol/sdk/client/streamableHttp.js';
import { GraphStore } from './graph.mjs';
import { createApiServer } from './server.mjs';
import { resourceAuth } from './test-resource-auth.mjs';

// This suite may write only to the explicitly provisioned, empty test database.
// Never substitute a production URL or a default password here.
const testUrl = process.env.TEST_NEO4J_URL;
const allowedTestUrl = 'bolt://127.0.0.1:17687';
// These are graph/backend and protocol tests with a fake resource-auth boundary.
// Actual OAuth is covered by auth.test.mjs and oauth-integration.test.mjs.
const token = `integration-${randomUUID()}`;
// Separate private signing fixture; production never exposes cursor keys as auth credentials.
const cursorKey = `integration-cursor-${randomUUID()}`;
const runId = randomUUID();
const id = (suffix) => `integration-${runId}-${suffix}`;
const group = (name) => `${runId}-${name}`;
const prefix = '/v1/graph';

describe('generic graph API against an isolated real Neo4j', {
  skip: !testUrl && 'Set TEST_NEO4J_URL and TEST_NEO4J_PASSWORD for the isolated test database',
}, () => {
  let driver;
  let server;
  let origin;
  let seeded = false;
  let initialCounts;

  const nodes = [
    ...['a', 'b', 'c', 'd', 'e'].map((name) => ({
      id: id(name), group: group('graph'), name, kind: `arbitrary-kind-${name}`,
    })),
    { id: id('sort-bool-false'), group: group('sort'), rank: false },
    { id: id('sort-bool-true'), group: group('sort'), rank: true },
    { id: id('sort-number-2a'), group: group('sort'), rank: neo4j.int(2) },
    { id: id('sort-number-2b'), group: group('sort'), rank: 2 },
    { id: id('sort-number-10'), group: group('sort'), rank: neo4j.int(10) },
    { id: id('sort-string-02'), group: group('sort'), rank: '02' },
    { id: id('sort-string-10'), group: group('sort'), rank: '10' },
    { id: id('sort-missing'), group: group('sort') },
    { id: id('large-zlower'), group: group('large'), rank: neo4j.int('9007199254740993') },
    { id: id('large-ahigher'), group: group('large'), rank: neo4j.int('9007199254740994') },
    {
      id: id('unknown'), group: group('odd'), kind: 'a-kind-that-has-never-existed',
      arbitraryField: 'MagicValue is searchable', enabled: true,
      tags: ['one', 'two'], vector: [0.125, 0.5], 'odd.name': 'literal property',
      legacy: '{"text":"retain this JSON string","items":[1,2]}',
    },
    {
      id: id('injection'), group: group('odd'), kind: 'diary',
      text: "' MATCH (n) DETACH DELETE n //", tags: ['two', 'one'],
      legacy: '[{"nested":"still a string"}]',
    },
  ];
  // Deliberately mix stored directions; clients always receive canonical a <= b.
  const edges = [
    ['a', 'b'], ['b', 'c'], ['c', 'a'], ['a', 'd'], ['c', 'd'], ['d', 'e'],
  ].map(([a, b]) => ({ a: id(a), b: id(b), text: `${a} is related to ${b}` }));

  async function runQuery(query, params = {}) {
    const session = driver.session();
    try {
      return await session.run(query, params);
    } finally {
      await session.close();
    }
  }

  async function counts() {
    const result = await runQuery(`
      CALL () { MATCH (n) RETURN count(n) AS nodes }
      CALL () { MATCH ()-[r]->() RETURN count(r) AS links }
      RETURN nodes, links
    `);
    return {
      nodes: result.records[0].get('nodes').toNumber(),
      links: result.records[0].get('links').toNumber(),
    };
  }

  before(async () => {
    assert.equal(testUrl, allowedTestUrl, 'Refusing test writes outside the exact isolated test URL');
    assert.ok(process.env.TEST_NEO4J_PASSWORD, 'The isolated database password must be explicitly provided');
    driver = neo4j.driver(testUrl, neo4j.auth.basic('neo4j', process.env.TEST_NEO4J_PASSWORD));
    await driver.verifyConnectivity();
    assert.deepEqual(await counts(), { nodes: 0, links: 0 }, 'Refusing to seed a nonempty database');
    // Mark fixture creation before the first write so partial setup is also cleaned.
    seeded = true;
    await runQuery(`
      UNWIND $rows AS row
      CREATE (n:Entity:GraphApiIntegrationFixture)
      SET n = row, n.integrationRun = $runId
    `, { rows: nodes, runId });
    await runQuery(`
      UNWIND $rows AS row
      MATCH (a:Entity {id: row.a}), (b:Entity {id: row.b})
      CREATE (a)-[:LINK {text: row.text}]->(b)
    `, { rows: edges });
    await runQuery(`
      CREATE (:SleepState:GraphApiIntegrationFixture {
        id: $stateId, integrationRun: $runId, name: 'internal state'
      })
      CREATE (:Entity:GraphApiIntegrationFixture {integrationRun: $runId})
      CREATE (:Entity:GraphApiIntegrationFixture {id: 123, integrationRun: $runId})
      CREATE (:Entity:GraphApiIntegrationFixture {id: '', integrationRun: $runId})
    `, { stateId: id('internal'), runId });
    // Relationships other than LINK, or LINKs to internal nodes, are not public.
    await runQuery(`
      MATCH (a:Entity {id: $a}), (b:Entity {id: $e}), (s:SleepState {id: $state}), (empty:Entity {id: ''})
      CREATE (a)-[:INTERNAL]->(b)
      CREATE (a)-[:LINK]->(s)
      CREATE (a)-[:LINK]->(empty)
    `, { a: id('a'), e: id('e'), state: id('internal') });
    initialCounts = await counts();
    server = createApiServer({ store: new GraphStore(driver), cursorKey, ...resourceAuth(token) });
    server.listen(0, '127.0.0.1');
    await once(server, 'listening');
    origin = `http://127.0.0.1:${server.address().port}`;
  });

  after(async () => {
    let countFailure;
    try {
      if (initialCounts) assert.deepEqual(await counts(), initialCounts, 'HTTP reads and rejected writes must not mutate Neo4j');
    } catch (error) {
      countFailure = error;
    } finally {
      if (server?.listening) {
        await new Promise((resolve, reject) => server.close((error) => error ? reject(error) : resolve()));
      }
      try {
        // The exact-URL guard remains at the mutation boundary, not just in setup.
        if (seeded && driver && testUrl === allowedTestUrl) {
          await runQuery('MATCH (n:GraphApiIntegrationFixture {integrationRun: $runId}) DETACH DELETE n', { runId });
          assert.deepEqual(await counts(), { nodes: 0, links: 0 }, 'Fixture cleanup must leave the test database empty');
        }
      } finally {
        await driver?.close();
      }
    }
    if (countFailure) throw countFailure;
  });

  test('standard MCP client queries the real graph without mutations', async () => {
    const client = new Client({ name: 'real-neo4j-mcp-integration', version: '1.0.0' });
    try {
      await client.connect(new StreamableHTTPClientTransport(new URL(`${origin}/mcp`), {
        requestInit: { headers: { Authorization: `Bearer ${token}` } },
      }));
      assert.equal((await client.listTools()).tools.length, 7);
      const call = async (name, args) => {
        const result = await client.callTool({ name, arguments: args });
        assert.ok(!result.isError, JSON.stringify(result));
        return result.structuredContent;
      };
      const search = await call('search_entities', { filter: { group: group('graph') }, limit: 2 });
      assert.equal(search.entities.length, 2);
      assert.ok(search.next_cursor);
      const next = await call('search_entities', { filter: { group: group('graph') }, limit: 2, cursor: search.next_cursor });
      assert.equal(next.entities.length, 2);
      assert.ok(!next.entities.some((e) => search.entities.some((x) => x.id === e.id)));
      assert.equal((await call('get_entity', { id: id('unknown') })).properties.arbitraryField, 'MagicValue is searchable');
      assert.equal((await call('list_entity_links', { id: id('a') })).links.length, 3);
      const graph = await call('get_subgraph', { ids: [id('a')], depth: 1 });
      assert.equal(graph.entities.length, 4);
      assert.deepEqual(await counts(), initialCounts);
    } finally { await client.close(); }
  });

  async function request(path, { method = 'GET', body, authorization = `Bearer ${token}`, rawBody } = {}) {
    const headers = {};
    if (authorization !== null) headers.authorization = authorization;
    if (body !== undefined || rawBody !== undefined) headers['content-type'] = 'application/json';
    const response = await fetch(`${origin}${path}`, {
      method, headers,
      body: rawBody !== undefined ? rawBody : body === undefined ? undefined : JSON.stringify(body),
    });
    assert.match(response.headers.get('content-type') || '', /application\/json/);
    return { status: response.status, body: await response.json() };
  }

  function listPath(params = {}) {
    const search = new URLSearchParams();
    for (const [key, value] of Object.entries(params)) {
      search.set(key, key === 'filter' ? JSON.stringify(value) : String(value));
    }
    return `${prefix}/entities?${search}`;
  }

  function assertError(result, status) {
    assert.equal(result.status, status);
    assert.equal(typeof result.body.error?.code, 'string');
    assert.equal(typeof result.body.error?.message, 'string');
    assert.ok(result.body.error.message.length > 0);
    assert.equal(result.body.error.stack, undefined);
    assert.ok(!JSON.stringify(result.body).includes(token), 'Errors must not expose the API token');
    assert.ok(!JSON.stringify(result.body).includes(process.env.TEST_NEO4J_PASSWORD), 'Errors must not expose database credentials');
  }

  function assertGraph(body) {
    assert.ok(Array.isArray(body.entities));
    assert.ok(Array.isArray(body.links));
    const entityIds = new Set(body.entities.map((entity) => entity.id));
    assert.equal(entityIds.size, body.entities.length, 'Entity IDs must be unique');
    const pairs = new Set();
    for (const entity of body.entities) {
      assert.deepEqual(Object.keys(entity).sort(), ['id', 'properties']);
      assert.equal(entity.properties.id, undefined);
    }
    for (const link of body.links) {
      assert.deepEqual(Object.keys(link).sort(), ['a', 'b', 'properties']);
      assert.ok(link.a <= link.b, 'Undirected link endpoints must be canonical');
      assert.ok(entityIds.has(link.a) && entityIds.has(link.b), 'No dangling endpoints');
      const pair = JSON.stringify([link.a, link.b]);
      assert.ok(!pairs.has(pair), 'An undirected pair must occur at most once');
      pairs.add(pair);
    }
    return entityIds;
  }

  test('all four data routes require valid bearer authorization', async () => {
    const routes = [
      [`${prefix}/entities`, {}],
      [`${prefix}/entities/${id('a')}`, {}],
      [`${prefix}/entities/${id('a')}/links`, {}],
      [`${prefix}/subgraph`, { method: 'POST', body: { ids: [id('a')] } }],
    ];
    for (const [path, options] of routes) {
      assertError(await request(path, { ...options, authorization: null }), 401);
      assertError(await request(path, { ...options, authorization: 'Bearer incorrect' }), 401);
      assertError(await request(path, { ...options, authorization: `Bearer ${cursorKey}` }), 401);
    }
  });

  test('unknown kinds and every ordinary property use the same entity envelope', async () => {
    const result = await request(`${prefix}/entities/${id('unknown')}`);
    assert.equal(result.status, 200);
    assert.deepEqual(Object.keys(result.body).sort(), ['id', 'properties']);
    assert.equal(result.body.id, id('unknown'));
    assert.equal(result.body.properties.id, undefined);
    for (const [key, value] of Object.entries(nodes.find((node) => node.id === id('unknown')))) {
      if (key !== 'id') assert.deepEqual(result.body.properties[key], value);
    }
    const diary = await request(`${prefix}/entities/${id('injection')}`);
    assert.equal(diary.status, 200);
    assert.equal(typeof diary.body.properties.legacy, 'string', 'A kind must not trigger special JSON decoding');
  });

  test('internal nodes and malformed Entity IDs are excluded from public resources', async () => {
    const result = await request(listPath({ limit: 200 }));
    assert.equal(result.status, 200);
    assert.equal(result.body.entities.length, nodes.length);
    assert.deepEqual(new Set(result.body.entities.map((entity) => entity.id)), new Set(nodes.map((node) => node.id)));
    assert.equal(result.body.next_cursor, null);
    assertError(await request(`${prefix}/entities/${id('internal')}`), 404);
    assertError(await request(`${prefix}/entities/${id('internal')}/links`), 404);
  });

  test('search covers arbitrary string fields and filters support scalar, array, and missing values', async () => {
    const searched = await request(listPath({ search: 'magicVALUE' }));
    assert.equal(searched.status, 200);
    assert.deepEqual(searched.body.entities.map((entity) => entity.id), [id('unknown')]);
    for (const filter of [
      { group: group('odd'), enabled: true },
      { group: group('odd'), tags: ['one', 'two'] },
      { group: group('odd'), 'odd.name': 'literal property' },
    ]) {
      const result = await request(listPath({ filter }));
      assert.equal(result.status, 200);
      assert.deepEqual(result.body.entities.map((entity) => entity.id), [id('unknown')]);
    }
    const missing = await request(listPath({ filter: { group: group('sort'), rank: null } }));
    assert.equal(missing.status, 200);
    assert.deepEqual(missing.body.entities.map((entity) => entity.id), [id('sort-missing')]);
  });

  test('untrusted values and property names cannot change Cypher semantics', async () => {
    const literal = nodes.find((node) => node.id === id('injection')).text;
    const result = await request(listPath({ filter: { text: literal }, search: 'DETACH DELETE' }));
    assert.equal(result.status, 200);
    assert.deepEqual(result.body.entities.map((entity) => entity.id), [id('injection')]);
    const key = "name`) MATCH (n) DETACH DELETE n //";
    const noMatch = await request(listPath({ filter: { [key]: 'anything' } }));
    assert.equal(noMatch.status, 200);
    assert.deepEqual(noMatch.body.entities, []);
    assertError(await request(`${prefix}/entities/${encodeURIComponent(literal)}`), 404);
    assert.deepEqual(await counts(), initialCounts);
  });

  test('mixed-type sorting is numeric, stable across pages, and always places missing last', async () => {
    const expected = {
      asc: ['sort-bool-false', 'sort-bool-true', 'sort-number-2a', 'sort-number-2b', 'sort-number-10', 'sort-string-02', 'sort-string-10', 'sort-missing'],
      desc: ['sort-bool-true', 'sort-bool-false', 'sort-number-10', 'sort-number-2a', 'sort-number-2b', 'sort-string-10', 'sort-string-02', 'sort-missing'],
    };
    for (const order of ['asc', 'desc']) {
      const found = [];
      const cursors = new Set();
      let cursor;
      do {
        const result = await request(listPath({
          filter: { group: group('sort') }, sort: 'rank', order, limit: 2,
          ...(cursor ? { cursor } : {}),
        }));
        assert.equal(result.status, 200);
        assert.ok(result.body.entities.length <= 2);
        found.push(...result.body.entities.map((entity) => entity.id));
        assert.ok(Object.hasOwn(result.body, 'next_cursor'));
        cursor = result.body.next_cursor;
        if (cursor) {
          assert.equal(typeof cursor, 'string');
          assert.ok(!cursors.has(cursor), 'Pagination must advance');
          cursors.add(cursor);
        }
        assert.ok(cursors.size <= 4, 'Pagination must terminate');
      } while (cursor);
      assert.deepEqual(found, expected[order].map(id));
    }
    assertError(await request(listPath({ filter: { group: group('odd') }, sort: 'tags' })), 400);
  });

  test('opaque cursors reject tampering, malformed input, and different queries or endpoints', async () => {
    const base = { filter: { group: group('sort') }, sort: 'rank', order: 'asc', limit: 2 };
    const first = await request(listPath(base));
    assert.equal(first.status, 200);
    const cursor = first.body.next_cursor;
    assert.equal(typeof cursor, 'string');
    const midpoint = Math.floor(cursor.length / 2);
    const tampered = cursor.slice(0, midpoint) + (cursor[midpoint] === 'a' ? 'b' : 'a') + cursor.slice(midpoint + 1);
    for (const params of [
      { ...base, cursor: tampered },
      { ...base, cursor: 'definitely.not.a.cursor' },
      { ...base, cursor, search: 'changed' },
      { ...base, cursor, filter: { group: group('graph') } },
      { ...base, cursor, sort: 'id' },
      { ...base, cursor, order: 'desc' },
    ]) assertError(await request(listPath(params)), 400);
    assertError(await request(`${prefix}/entities/${id('a')}/links?cursor=${encodeURIComponent(cursor)}`), 400);
    const firstLink = await request(`${prefix}/entities/${id('a')}/links?limit=1`);
    assert.equal(firstLink.status, 200);
    const linkCursor = firstLink.body.next_cursor;
    assert.equal(typeof linkCursor, 'string');
    assertError(await request(listPath({ cursor: linkCursor })), 400);
    assertError(await request(`${prefix}/entities/${id('c')}/links?limit=1&cursor=${encodeURIComponent(linkCursor)}`), 400);
  });

  test('integer sort cursors preserve values beyond JavaScript safe integer precision', async () => {
    for (const order of ['asc', 'desc']) {
      const params = { filter: { group: group('large') }, sort: 'rank', order, limit: 1 };
      const first = await request(listPath(params));
      assert.equal(first.status, 200);
      assert.equal(typeof first.body.next_cursor, 'string');
      const second = await request(listPath({ ...params, cursor: first.body.next_cursor }));
      assert.equal(second.status, 200);
      assert.equal(second.body.next_cursor, null);
      const expectedIds = ['large-zlower', 'large-ahigher'];
      const expectedValues = ['9007199254740993', '9007199254740994'];
      if (order === 'desc') {
        expectedIds.reverse();
        expectedValues.reverse();
      }
      const actual = [...first.body.entities, ...second.body.entities];
      assert.deepEqual(actual.map((entity) => entity.id), expectedIds.map(id));
      assert.deepEqual(actual.map((entity) => entity.properties.rank), expectedValues);
    }
  });

  test('even correctly signed cursor payloads must have a valid shape', async () => {
    // Test-only access to the private signing fixture lets us exercise payload
    // validation independently of signatures. Authenticated production clients
    // cannot derive this key from their OAuth access tokens.
    const key = createHmac('sha256', cursorKey).update('scene-graph-api/cursor/v1').digest();
    const signPayload = (payload) => {
      const data = Buffer.from(JSON.stringify(payload)).toString('base64url');
      return `${data}.${createHmac('sha256', key).update(data).digest('base64url')}`;
    };
    assertError(await request(listPath({ cursor: signPayload(null) })), 400);
    const scope = createHash('sha256').update(JSON.stringify(['links', id('a')])).digest('base64url');
    const malformedAfter = signPayload({ v: 1, scope, after: { id: id('b') } });
    assertError(await request(`${prefix}/entities/${id('a')}/links?cursor=${encodeURIComponent(malformedAfter)}`), 400);
  });

  test('a chunked body over 32 KiB receives structured HTTP 413 instead of a reset socket', async () => {
    const result = await new Promise((resolve, reject) => {
      const outgoing = http.request(`${origin}${prefix}/subgraph`, {
        method: 'POST',
        headers: {
          authorization: `Bearer ${token}`,
          'content-type': 'application/json',
          'transfer-encoding': 'chunked',
          connection: 'close',
        },
      }, (response) => {
        const chunks = [];
        response.on('error', reject);
        response.on('data', (chunk) => chunks.push(chunk));
        response.on('end', () => {
          try {
            assert.match(response.headers['content-type'] || '', /application\/json/);
            resolve({ status: response.statusCode, body: JSON.parse(Buffer.concat(chunks).toString()) });
          } catch (error) {
            reject(error);
          }
        });
      });
      outgoing.on('error', reject);
      outgoing.setTimeout(5000, () => outgoing.destroy(new Error('Oversized chunked request did not receive a response')));
      assert.equal(outgoing.getHeader('content-length'), undefined);
      outgoing.write('{"ids":["');
      outgoing.write('x'.repeat(16 * 1024));
      outgoing.write('x'.repeat(16 * 1024));
      outgoing.end('"]}');
    });
    assertError(result, 413);
  });

  test('links paginate both stored directions consistently and include only each page endpoints', async () => {
    let cursor;
    const found = [];
    do {
      const query = new URLSearchParams({ limit: '1', ...(cursor ? { cursor } : {}) });
      const result = await request(`${prefix}/entities/${id('a')}/links?${query}`);
      assert.equal(result.status, 200);
      const entityIds = assertGraph(result.body);
      assert.ok(entityIds.has(id('a')));
      assert.equal(result.body.links.length, 1);
      assert.equal(result.body.entities.length, 2);
      found.push(...result.body.links);
      cursor = result.body.next_cursor;
      assert.ok(found.length <= 3, 'Link pagination must terminate');
    } while (cursor);
    assert.deepEqual(found.map((link) => link.a === id('a') ? link.b : link.a).sort(), ['b', 'c', 'd'].map(id));
    for (const link of found) {
      const neighbor = link.a === id('a') ? link.b : link.a;
      const reverse = await request(`${prefix}/entities/${neighbor}/links`);
      assert.equal(reverse.status, 200);
      assertGraph(reverse.body);
      assert.deepEqual(reverse.body.links.find((candidate) => candidate.a === link.a && candidate.b === link.b), link);
    }
  });

  test('subgraphs include cycles, same-layer edges, and the union of multiple seed neighborhoods', async () => {
    const result = await request(`${prefix}/subgraph`, { method: 'POST', body: { ids: [id('a')], depth: 1 } });
    assert.equal(result.status, 200);
    assert.deepEqual(assertGraph(result.body), new Set(['a', 'b', 'c', 'd'].map(id)));
    assert.equal(result.body.links.length, 5, 'The induced subgraph must include non-tree edges');
    assert.equal(result.body.truncated, false);
    assert.deepEqual(result.body.limits, { depth: 1, max_entities: 100, max_links: 200 });
    const multi = await request(`${prefix}/subgraph`, { method: 'POST', body: { ids: [id('b'), id('e')], depth: 1 } });
    assert.equal(multi.status, 200);
    assert.deepEqual(assertGraph(multi.body), new Set(['a', 'b', 'c', 'd', 'e'].map(id)));
    assert.equal(multi.body.links.length, 6);
    assert.equal(multi.body.truncated, false);
    const zero = await request(`${prefix}/subgraph`, { method: 'POST', body: { ids: [id('a'), id('b')], depth: 0 } });
    assert.equal(zero.status, 200);
    assert.deepEqual(assertGraph(zero.body), new Set(['a', 'b'].map(id)));
    assert.equal(zero.body.links.length, 1);
    assert.equal(zero.body.truncated, false);
  });

  test('node and edge truncation are explicit, retain seeds, and never create dangling links', async () => {
    const cases = [
      { ids: [id('a')], depth: 2, max_entities: 2, max_links: 200 },
      { ids: [id('a')], depth: 2, max_entities: 100, max_links: 2 },
      { ids: [id('b'), id('e')], depth: 2, max_entities: 2, max_links: 1 },
    ];
    for (const body of cases) {
      const result = await request(`${prefix}/subgraph`, { method: 'POST', body });
      assert.equal(result.status, 200);
      const returnedIds = assertGraph(result.body);
      for (const seed of body.ids) assert.ok(returnedIds.has(seed), 'Truncation must retain every seed');
      assert.ok(result.body.entities.length <= body.max_entities);
      assert.ok(result.body.links.length <= body.max_links);
      assert.equal(result.body.truncated, true);
      assert.deepEqual(result.body.limits, { depth: body.depth, max_entities: body.max_entities, max_links: body.max_links });
    }
    const exact = await request(`${prefix}/subgraph`, {
      method: 'POST', body: { ids: [id('a')], depth: 1, max_entities: 4, max_links: 5 },
    });
    assert.equal(exact.status, 200);
    assert.equal(exact.body.truncated, false, 'Reaching a limit exactly does not mean data was omitted');
  });

  test('missing IDs, invalid JSON, complex filters, and out-of-range limits return structured errors', async () => {
    assertError(await request(`${prefix}/entities/${id('missing')}`), 404);
    assertError(await request(`${prefix}/entities/${id('missing')}/links`), 404);
    assertError(await request(`${prefix}/subgraph`, { method: 'POST', body: { ids: [id('a'), id('missing')] } }), 404);
    assertError(await request(`${prefix}/subgraph`, { method: 'POST', rawBody: '{invalid JSON' }), 400);
    assertError(await request(`${prefix}/entities?filter=%7Bbroken`), 400);
    assertError(await request(listPath({ filter: { name: { arbitrary: 'object' } } })), 400);
    for (const limit of ['0', '-1', '1.5', 'no', '201']) {
      assertError(await request(listPath({ limit })), 400);
    }
    for (const limit of ['0', '-1', '1.5', 'no', '501']) {
      assertError(await request(`${prefix}/entities/${id('a')}/links?limit=${limit}`), 400);
    }
    const invalidBodies = [
      {}, { ids: [] }, { ids: 'a' }, { ids: [123] },
      { ids: Array.from({ length: 51 }, (_, index) => id(`too-many-${index}`)) },
      { ids: [id('a')], depth: -1 }, { ids: [id('a')], depth: 4 },
      { ids: [id('a')], depth: 1.5 },
      { ids: [id('a')], depth: null },
      { ids: [id('a')], max_entities: 0 }, { ids: [id('a')], max_entities: 501 },
      { ids: [id('a')], max_entities: 1.5 },
      { ids: [id('a')], max_entities: null },
      { ids: [id('a')], max_links: 0 }, { ids: [id('a')], max_links: 2001 },
      { ids: [id('a')], max_links: 1.5 },
      { ids: [id('a')], max_links: null },
      { ids: [id('a'), id('b')], max_entities: 1 },
    ];
    for (const body of invalidBodies) {
      assertError(await request(`${prefix}/subgraph`, { method: 'POST', body }), 400);
    }
  });

  test('the HTTP surface cannot mutate entities or relationships', async () => {
    for (const [path, method] of [
      [`${prefix}/entities`, 'POST'],
      [`${prefix}/entities/${id('a')}`, 'PATCH'],
      [`${prefix}/entities/${id('a')}`, 'DELETE'],
      [`${prefix}/entities/${id('a')}/links`, 'POST'],
      [`${prefix}/subgraph`, 'GET'],
    ]) {
      const result = await request(path, { method, ...(method !== 'GET' ? { body: { text: 'must not write' } } : {}) });
      assert.ok([403, 405].includes(result.status));
      assertError(result, result.status);
    }
    assert.deepEqual(await counts(), initialCounts);
  });
});
