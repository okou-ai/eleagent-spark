"""Remove everything the naming service wrote for test captures (capture ids 'entity-acceptance-' / 'archive-probe-').

  python -m naming.cleanup [--capture-id ID] [--dry-run]

One write transaction under the graph writer lock (SyncState 'cursors', like graph-sync / sleep):
  1. names and evidence changed on existing memory objects are reverted from NamingEvent.prev_json, newest event
     first, and only while the entity still carries that event's naming_event_id (a later real naming wins);
  2. memory objects created for test captures (Entity.test_capture) are deleted;
  3. their views (ObjectView nodes of any source) and every ObjectView / NamingEvent tagged test_capture are deleted.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os

from naming.service import LOCK_QUERY, TEST_PREFIXES


def clean(tx, capture_id=None):
    match = 'n.test_capture = $c' if capture_id else 'n.test_capture IS NOT NULL'
    events = tx.run(f"""MATCH (n:NamingEvent) WHERE {match}
        RETURN n.id AS id, n.memory_id AS memory_id, n.prev_json AS prev ORDER BY n.created_at DESC, n.id DESC""", c=capture_id).data()
    reverted, skipped = [], []
    for event in events:
        if not event['prev'] or not event['memory_id']:
            continue
        prev = json.loads(event['prev'])
        result = tx.run("""MATCH (e:Entity {id:$id}) WHERE e.naming_event_id = $event AND e.test_capture IS NULL
            SET e += $prev RETURN e.id AS id, e.name AS name""", id=event['memory_id'], event=event['id'], prev=prev).data()
        (reverted if result else skipped).append({'event': event['id'], 'memory_id': event['memory_id'],
                                                  'restored_name': prev.get('name') if result else None})
    match_e = 'e.test_capture = $c' if capture_id else 'e.test_capture IS NOT NULL'
    entities = [r['id'] for r in tx.run(f'MATCH (e:Entity) WHERE {match_e} RETURN e.id AS id', c=capture_id).data()]
    match_v = 'v.test_capture = $c' if capture_id else 'v.test_capture IS NOT NULL'
    views = tx.run(f"""MATCH (v:ObjectView) WHERE {match_v} OR v.memory_id IN $ids
        DETACH DELETE v RETURN count(*) AS n""", c=capture_id, ids=entities).single()['n']
    tx.run('UNWIND $ids AS id MATCH (e:Entity {id:id}) DETACH DELETE e', ids=entities).consume()
    naming_events = tx.run(f'MATCH (n:NamingEvent) WHERE {match} DETACH DELETE n RETURN count(*) AS n', c=capture_id).single()['n']
    return {'reverted': reverted, 'skipped_revert': skipped, 'entities_deleted': entities,
            'views_deleted': views, 'naming_events_deleted': naming_events}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture-id', help='only this test capture (default: every test capture)')
    parser.add_argument('--dry-run', action='store_true', help='run the cleanup and roll it back')
    args = parser.parse_args()
    if args.capture_id and not args.capture_id.startswith(TEST_PREFIXES):
        parser.error('only test captures (%s) can be cleaned up' % ', '.join(TEST_PREFIXES))
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(os.environ['GRAPH_URL'], auth=tuple(os.environ['NEO4J_AUTH'].split('/', 1)))
    try:
        with driver.session() as session:
            tx = session.begin_transaction()
            try:
                state = tx.run(LOCK_QUERY, now=datetime.now(timezone.utc).isoformat()).single()['s']
                if state.get('maintenance_token'):
                    raise SystemExit('graph maintenance pending; try again in a minute')
                result = clean(tx, args.capture_id)
                if args.dry_run:
                    tx.rollback()
                else:
                    tx.commit()
            finally:
                if not tx.closed():
                    tx.rollback()
                tx.close()
        print(json.dumps({'dry_run': args.dry_run, **result}, ensure_ascii=False, indent=1))
    finally:
        driver.close()


if __name__ == '__main__':
    main()
