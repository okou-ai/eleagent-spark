"""One-off migration (2026-10-21): the views naming keeps of memory objects move from :LivePrototype to :ObjectView.

The live overlay and its phone models are gone, so the graph loses what was named after them:
  - every (:LivePrototype) becomes (:ObjectView) with the same properties, except phone, which becomes curated (the
    same value: whether the retired live-pack's curation kept the view);
  - the (:LiveState) node of the retired live-pack service is deleted;
  - the live_prototype_id constraint and live_prototype_memory index are replaced by object_view_id / object_view_memory,
    and LiveState's live_state_id constraint is dropped.
Running it again changes nothing.
Stop the naming service first: it is the only writer of these nodes. The data change is one transaction under the
graph-sync / sleep writer lock. --backup writes every node it changes or deletes to a JSON file first.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os

from neo4j import GraphDatabase

from naming.service import LOCK_QUERY

COUNTS = """MATCH (p:LivePrototype) WITH count(p) AS old, count(CASE WHEN p.phone THEN 1 END) AS kept
    OPTIONAL MATCH (v:ObjectView) WITH old, kept, count(v) AS new, count(CASE WHEN v.curated THEN 1 END) AS curated
    OPTIONAL MATCH (s:LiveState) RETURN old, kept, new, curated, count(s) AS state"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backup', help='JSON file for the nodes changed or deleted (required unless --dry-run)')
    parser.add_argument('--dry-run', action='store_true', help='only count')
    args = parser.parse_args()
    if not args.dry_run and not args.backup:
        parser.error('--backup is required')
    user, password = os.environ['NEO4J_AUTH'].split('/', 1)
    driver = GraphDatabase.driver(os.environ['GRAPH_URL'], auth=(user, password))
    with driver.session() as session:
        before = session.run(COUNTS).single().data()
        print(json.dumps({'event': 'before', **before}))
        if args.dry_run:
            return
        nodes = session.run("""MATCH (n) WHERE n:LivePrototype OR n:LiveState
            RETURN labels(n) AS labels, properties(n) AS properties""").data()
        with open(args.backup, 'w', encoding='utf-8') as file:
            json.dump(nodes, file, ensure_ascii=False, default=str)
        print(json.dumps({'event': 'backup', 'file': args.backup, 'nodes': len(nodes)}))
        session.run('CREATE CONSTRAINT object_view_id IF NOT EXISTS FOR (v:ObjectView) REQUIRE v.id IS UNIQUE').consume()
        session.run('CREATE INDEX object_view_memory IF NOT EXISTS FOR (v:ObjectView) ON (v.memory_id)').consume()

        def move(tx):
            tx.run(LOCK_QUERY, now=datetime.now(timezone.utc).isoformat()).consume()
            moved = tx.run("""MATCH (p:LivePrototype) SET p:ObjectView, p.curated = p.phone
                REMOVE p:LivePrototype, p.phone RETURN count(p) AS n""").single()['n']
            state = tx.run('MATCH (s:LiveState) DETACH DELETE s RETURN count(*) AS n').single()['n']
            return moved, state
        moved, state = session.execute_write(move)
        print(json.dumps({'event': 'moved', 'views': moved, 'live_state_deleted': state}))
        session.run('DROP CONSTRAINT live_prototype_id IF EXISTS').consume()
        session.run('DROP INDEX live_prototype_memory IF EXISTS').consume()
        session.run('DROP CONSTRAINT live_state_id IF EXISTS').consume()
        after = session.run(COUNTS).single().data()
        leftover = session.run("""SHOW INDEXES YIELD name, labelsOrTypes WHERE 'LivePrototype' IN labelsOrTypes
            OR 'LiveState' IN labelsOrTypes RETURN collect(name) AS names""").single()['names']
        print(json.dumps({'event': 'after', **after, 'leftover_indexes': leftover}))
        ok = after['old'] == 0 and after['state'] == 0 and after['new'] == before['new'] + before['old'] \
            and after['curated'] == before['curated'] + before['kept'] and not leftover
        print(json.dumps({'event': 'done', 'ok': ok}))
        if not ok:
            raise SystemExit(1)
    driver.close()


if __name__ == '__main__':
    main()
