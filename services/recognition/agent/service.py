"""Eleagent's day agent: when the user asks Okou for help while recording, hand the task to Okou.

Loop (every --interval s): new archive `speech` turns after AgentState's cursor (read-only SQLite, like naming).
Each turn goes to the local VLM (vlm on Spark 2) with the recording's recent transcript: is the user asking Okou to
do something for them? If so, an OpenAI Agents SDK agent on the same model gets the moment — local time, place, the
transcript, who and what is recognised in view, what MiniCPM saw — and a memory search tool, and writes one message
that Okou can act on without the recording. hand_to_okou sends it with Okou MCP's send_chat_message into one standing
Okou thread (AgentState.okou_thread_id). Okou's access token comes from graph-api's internal listener on the Spark
link (--token-url, shared secret); the refresh token stays on Spark 1. An (:AgentEvent) per handled turn lets the
phone say "已交给 Okou：…". Okou does the work and asks the user whatever it needs in its own chat.

--text "…" --capture <id> tries one made-up newest turn against that recording's latest moment; with --dry-run
nothing goes to Okou or the graph (the message is printed).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import httpx
import numpy as np

from naming.common import appearance_accepts, appearance_match, log, unit
from naming.service import (FACE_DIM, FACE_MARGIN, FACE_THRESHOLD, OWNER_ID, digest, face_vectors, now_iso, ro)
from naming import vlm

STATE_ID = 'agent'
OKOU_MCP = 'https://api.okou.ai/mcp'
SCHEMA = ['CREATE CONSTRAINT agent_event_id IF NOT EXISTS FOR (a:AgentEvent) REQUIRE a.id IS UNIQUE',
          'CREATE INDEX agent_event_created IF NOT EXISTS FOR (a:AgentEvent) ON (a.created_at)']

SETTING = """Eleagent is a personal memory assistant. The user carries a phone camera that records their day. Speech \
recognition turns what is said into text one turn at a time: turns can be cut mid-sentence, can contain \
similar-sounding wrong words (names especially, and "Okou" itself can come out as something that sounds like it), \
and speaker labels are rough guesses.

What the user says is for one of several listeners. Mostly it is for the people around them, or for themselves. \
Some of it is for Eleagent: they teach it what people and things are called, so that it recognises them later. And \
some of it is for Okou, their AI assistant, which acts for them in their accounts (calendar, mail, messages, \
documents, the web): they ask Okou out loud to do something, the way they would ask a person, calling it by name, \
in any language (the transcript may show the name as a sound-alike, such as "OK" or "包括"). Eleagent hands those \
requests to Okou, which keeps them in one conversation, does the work and talks with the user there."""

GATE_SYSTEM = SETTING + """

You are the ear that notices requests for Okou. You receive the transcript of this recording so far and the newest \
turn. Decide whom the newest turn is for. Handing Okou talk that was not for it makes Okou act on things the user \
never asked for; missing a request for Okou makes the user repeat it.

Reply with JSON:
- situation: one sentence on what is happening with the newest turn.
- for: "Okou", "Eleagent", "people present" or "themselves".
- request: if it is for Okou, what the user asks Okou to do, in their words, with misheard words repaired where the \
conversation makes them clear; else null.
- confidence: 0..1, that the turn is for whom you said."""
GATE_SCHEMA = {'type': 'object', 'additionalProperties': False, 'required': ['situation', 'for', 'request', 'confidence'],
    'properties': {'situation': {'type': 'string'}, 'for': {'enum': ['Okou', 'Eleagent', 'people present', 'themselves']},
                   'request': {'type': ['string', 'null']}, 'confidence': {'type': 'number'}}}

AGENT_INSTRUCTIONS = SETTING + """

You are Eleagent's day agent. Eleagent's ear heard the user ask Okou for help. The ear hears one turn at a time and \
can be wrong; if, with the whole moment in front of you, the user was not asking Okou for anything, hand nothing \
over. Okou does not hear the recording and does not know \
where the user is or who is with them; it gets one message from you and starts from it. You receive what Eleagent \
knows about this moment: the time, the place, the conversation, who and what is recognised in view, and what the \
camera shows, and what was already handed to Okou during this recording (Okou has those; the transcript marks the \
turns they came from). search_memory looks up the people, places and things Eleagent remembers from earlier days.

Write the message Okou needs to understand the request on its own: what the user wants, in their words, and the \
facts you have that make it clear, leaving out what Okou already has, such as who the people are as Eleagent knows them, the date and time meant (from \
the current time), where the user is, and the few lines of conversation that matter. Okou knows how to do the work \
and can ask the user, so leave the how to Okou and say what is unclear rather than guessing. Write it in the user's \
language. Then call hand_to_okou once, with the message and a short line for the user's phone screen saying what \
was handed over; or, when there is nothing new for Okou (the turn was not a request for it, or Okou already has \
what it asks), call nothing_for_okou instead."""


class Graph:
    def __init__(self, driver, dry_run):
        self.driver, self.dry_run = driver, dry_run

    def read(self, query, **params):
        with self.driver.session() as session:
            return session.run(query, **params).data()

    def write(self, query, **params):
        if self.dry_run:
            return
        with self.driver.session() as session:
            session.execute_write(lambda tx: tx.run(query, **params).consume())


class Agent:
    def __init__(self, graph, args):
        self.graph, self.args = graph, args
        self.tz = ZoneInfo(args.timezone)
        self.cursor = None
        self.vlm_down = False
        self.state = {}
        self.replayed = None         # --replay-rowids: what this dry run handed over, instead of the graph's events

    # --- the moment -----------------------------------------------------------------------------------------------

    def events(self, db, capture_id, kind, lo_us, hi_us, limit=200):
        return db.execute("""SELECT rowid AS r, start_us, end_us, body_json FROM archive_events WHERE capture_id=? AND kind=?
            AND start_us BETWEEN ? AND ? ORDER BY start_us DESC LIMIT ?""", (capture_id, kind, lo_us, hi_us, limit)).fetchall()

    def history(self, capture_id):
        """What was handed to Okou during this recording: [{event_id, request, at_us}], oldest first."""
        if self.replayed is not None:
            return [h for h in self.replayed if h['capture_id'] == capture_id]
        rows = self.graph.read("""MATCH (a:AgentEvent {capture_id:$c}) WHERE a.status = 'sent'
            RETURN a.event_id AS event_id, a.request AS request, a.created_at AS created_at ORDER BY a.created_at""", c=capture_id)
        return [{'event_id': r['event_id'], 'request': r['request'], 'at': r['created_at']} for r in rows]

    def transcript(self, db, capture_id, start_us, newest_text, newest_speaker):
        rows = db.execute("""SELECT event_id, body_json FROM archive_events WHERE capture_id=? AND kind='speech' AND start_us<? AND start_us>=?
            ORDER BY start_us DESC LIMIT 40""", (capture_id, start_us, start_us - 600_000_000)).fetchall()
        handed = {h['event_id']: h['request'] for h in self.history(capture_id)}
        turns = [{'speaker': json.loads(r['body_json']).get('speaker_entity_id'), 'text': json.loads(r['body_json']).get('text') or '',
                  'note': f'handed to Okou: "{handed[r["event_id"]]}"' if r['event_id'] in handed else None} for r in reversed(rows)]
        return vlm.transcript(turns, (newest_speaker, newest_text))

    def in_view(self, db, capture_id, at_us):
        """Names of the people and memory objects recognised within a few seconds of the turn."""
        people = self.graph.read(f"""MATCH (e:Entity) WHERE e.layer='memory' AND e.kind='person'
            RETURN e.id AS id, e.name AS name, e.face_vec AS face_vec, e.face_views AS face_views""")
        faces = [(p, face_vectors(p)) for p in people]
        faces = [(p, np.stack(v)) for p, v in faces if v]
        seen_people, unknown_faces = {}, 0
        for row in self.events(db, capture_id, 'face', at_us - 5_000_000, at_us + 5_000_000):
            for o in json.loads(row['body_json']).get('observations') or []:
                try:
                    vec = unit(o['embedding'], dims=FACE_DIM)
                except (KeyError, TypeError, ValueError):
                    continue
                scored = sorted(((float((v @ vec).max()), p) for p, v in faces), key=lambda s: -s[0])
                if scored and scored[0][0] >= FACE_THRESHOLD and (len(scored) < 2 or scored[0][0] - scored[1][0] >= FACE_MARGIN):
                    seen_people[scored[0][1]['id']] = scored[0][1]['name'] or 'someone Eleagent knows'
                else:
                    unknown_faces += 1
        objects = self.graph.read("""MATCH (e:Entity) WHERE e.layer='memory' AND e.kind='object' AND e.object_vec IS NOT NULL
            AND e.name IS NOT NULL RETURN e.name AS name, e.object_vec AS object_vec""")
        vectors = []
        for o in objects:
            try:
                vectors.append((o['name'], unit(o['object_vec'])))
            except (TypeError, ValueError):
                pass
        seen_objects = []
        if vectors:
            stack = np.stack([v for _, v in vectors])
            for row in self.events(db, capture_id, 'object', at_us - 3_000_000, at_us + 3_000_000, limit=20):
                for o in json.loads(row['body_json']).get('observations') or []:
                    try:
                        best, score, _, runner, _ = appearance_match(stack, unit(o['embedding']))
                    except (KeyError, TypeError, ValueError):
                        continue
                    if appearance_accepts(score, runner) and vectors[best][0] not in seen_objects:
                        seen_objects.append(vectors[best][0])
        owner = seen_people.pop(OWNER_ID, None)
        return owner, list(seen_people.values()), unknown_faces, seen_objects[:12]

    def moment(self, db, capture_id, at_us, text, speaker):
        local = datetime.fromtimestamp(at_us / 1e6, tz=timezone.utc).astimezone(self.tz)
        owner, people, unknown, objects = self.in_view(db, capture_id, at_us)
        scene = self.events(db, capture_id, 'minicpm', at_us - 15_000_000, at_us + 5_000_000, limit=1)
        place = db.execute("""SELECT body_json FROM archive_events WHERE kind='location' AND start_us BETWEEN ? AND ?
            ORDER BY start_us DESC LIMIT 1""", (at_us - 1_800_000_000, at_us + 5_000_000)).fetchone()
        lines = [f"Now: {local.strftime('%Y-%m-%d %A %H:%M')} ({self.args.timezone}).",
                 f"The user: {self.state.get('owner_name') or 'the owner of this memory'}."]
        if place:
            p = json.loads(place['body_json'])
            lines.append(f"Where: phone location {p.get('latitude'):.5f}, {p.get('longitude'):.5f} (±{round(p.get('accuracy_m') or 0)} m).")
        lines.append('People recognised in view: ' + (', '.join(people) if people else 'none')
                     + (f'; {unknown} face(s) Eleagent does not know' if unknown else '') + ('; the user is in view' if owner else '') + '.')
        lines.append('Things recognised in view: ' + (', '.join(objects) if objects else 'none') + '.')
        known = [p['name'] for p in self.graph.read(f"""MATCH (e:Entity) WHERE e.layer='memory' AND e.kind='person'
            AND e.id <> '{OWNER_ID}' AND e.name IS NOT NULL RETURN e.name AS name ORDER BY coalesce(e.t_last, '') DESC LIMIT 30""")]
        lines.append('People Eleagent remembers (names as Eleagent keeps them; speech recognition may spell them '
                     'differently): ' + ('; '.join(known) if known else 'none yet') + '.')
        if scene:
            lines.append('What the camera shows (a quick automatic description): '
                         + (json.loads(scene[0]['body_json']).get('generated_text') or '').replace('\n', ' ')[:400])
        earlier = self.history(capture_id)
        lines.append('Already handed to Okou during this recording: ' + ('; '.join(f'"{h["request"]}"' for h in earlier) if earlier else 'nothing yet') + '.')
        lines.append('\nThe conversation (oldest first, the newest turn last):\n' + self.transcript(db, capture_id, at_us, text, speaker))
        return '\n'.join(lines)

    # --- Okou ------------------------------------------------------------------------------------------------------

    def okou_token(self):
        secret = open(self.args.token_secret_file).read().strip()
        response = httpx.get(self.args.token_url, headers={'Authorization': 'Bearer ' + secret}, timeout=20)
        if response.status_code == 409:
            raise RuntimeError('okou_not_connected')
        response.raise_for_status()
        return response.json()['access_token']

    async def okou_call(self, server, name, arguments):
        result = await server.call_tool(name, arguments)
        if getattr(result, 'isError', False):
            raise RuntimeError(f'{name}: ' + ' '.join(getattr(c, 'text', '') for c in result.content)[:300])
        if getattr(result, 'structuredContent', None):
            return result.structuredContent
        text = ''.join(getattr(c, 'text', '') for c in result.content)
        return json.loads(text) if text.strip().startswith(('{', '[')) else text

    async def okou_agent_id(self, server):
        if self.args.okou_agent_id:
            return self.args.okou_agent_id
        if self.state.get('okou_agent_id'):
            return self.state['okou_agent_id']
        agents = await self.okou_call(server, 'list_agents', {})
        agents = agents.get('result', agents) if isinstance(agents, dict) else agents
        agents = agents if isinstance(agents, list) else (agents.get('agents') or agents.get('items') or [])
        chosen = next((a for a in agents if str(a.get('name', '')).strip().lower() == 'okou'), agents[0] if agents else None)
        if not chosen:
            raise RuntimeError('okou_has_no_agent')
        return chosen.get('id') or chosen.get('agentId')

    # --- one request -----------------------------------------------------------------------------------------------

    async def handle(self, item, context):
        from agents import Agent as SdkAgent, ModelSettings, OpenAIChatCompletionsModel, Runner, function_tool, set_tracing_disabled
        from agents.mcp import MCPServerStreamableHttp
        from openai import AsyncOpenAI
        set_tracing_disabled(True)  # nothing leaves the Sparks except the message to Okou
        handed, used = {}, []

        @function_tool
        def search_memory(query: str) -> str:
            """Look up people, places, things and events Eleagent remembers whose name or description contains the
            query (a name, or a part of one: speech recognition may have misheard it)."""
            used.append('search_memory')
            return self.search(query)

        @function_tool
        async def hand_to_okou(message: str, screen_note: str) -> str:
            """Send the message to Okou, which starts working on it, and show screen_note on the user's phone."""
            used.append('hand_to_okou')
            if handed.get('message'):
                return 'already handed to Okou'
            handed.update(message=message.strip() + self.footer(item), note=screen_note.strip()[:80])
            if self.args.dry_run:
                return 'handed to Okou (dry run)'
            try:
                token = self.okou_token()
                async with MCPServerStreamableHttp(name='okou', params={'url': self.args.okou_mcp, 'timeout': 30,
                        'headers': {'Authorization': 'Bearer ' + token}}) as server:
                    agent_id = await self.okou_agent_id(server)
                    arguments = {'agentId': agent_id, 'prompt': handed['message']}
                    if self.state.get('okou_thread_id'):
                        arguments['threadId'] = self.state['okou_thread_id']
                    sent = await self.okou_call(server, 'send_chat_message', arguments)
            except Exception as exc:  # the SDK would hand this to the model; keep it for the event too
                handed['error'] = f'{type(exc).__name__}: {exc}'[:300]
                return 'could not reach Okou: ' + handed['error']
            handed.update(thread_id=sent.get('threadId'), okou_event_id=sent.get('eventId'), agent_id=agent_id)
            if handed['thread_id'] and handed['thread_id'] != self.state.get('okou_thread_id'):
                self.state.update(okou_thread_id=handed['thread_id'], okou_agent_id=agent_id)
                self.graph.write("MERGE (s:AgentState {id:$id}) SET s.okou_thread_id=$t, s.okou_agent_id=$a",
                                 id=STATE_ID, t=handed['thread_id'], a=agent_id)
            return 'handed to Okou'

        @function_tool
        def nothing_for_okou(reason: str) -> str:
            """Hand nothing over: the turn asks Okou for nothing new (not a request for Okou, or Okou already has it)."""
            used.append('nothing_for_okou')
            if not handed.get('message'):
                handed['declined'] = reason.strip()[:200]
            return 'nothing handed over'

        model = OpenAIChatCompletionsModel(model=self.args.vlm_model, openai_client=AsyncOpenAI(base_url=self.args.vlm_url, api_key='local'))
        agent = SdkAgent(name='Eleagent day agent', instructions=AGENT_INSTRUCTIONS, model=model, tools=[search_memory, hand_to_okou, nothing_for_okou],
                         model_settings=ModelSettings(temperature=0, max_tokens=1200))
        started = time.perf_counter()
        language = 'Chinese' if any('\u3400' <= ch <= '\u9fff' for ch in item['utterance']) else 'the language of the request'
        prompt = context + f"\n\nThe request as heard: {item['request']}\nThe user said it in {language}."
        try:
            result = await Runner.run(agent, prompt, max_turns=8)
            final = str(result.final_output or '')[:500]
            error = None
        except Exception as exc:
            final, error = '', f'{type(exc).__name__}: {exc}'[:300]
        handed['tools'] = used
        return handed, final, error, round(time.perf_counter() - started, 1)

    def footer(self, item):
        """Where the request was said, exactly: Okou can look at that moment through its Eleagent connector."""
        start = datetime.fromtimestamp(item['start_us'] / 1e6, tz=timezone.utc).astimezone(self.tz)
        end = datetime.fromtimestamp(item['end_us'] / 1e6, tz=timezone.utc).astimezone(self.tz)
        iso = lambda t: t.isoformat(timespec='seconds')
        # Frames come back through MCP itself; the original video's download URL needs a token on another path.
        moments = [start + (end - start) * k / 2 for k in range(3)]
        frames = f"get_video_frames(capture_id={item['capture_id']}, at=[{', '.join(iso(t) for t in moments)}])"
        clips = f"list_video_clips(capture_id={item['capture_id']}, started_at={iso(start)}, ended_at={iso(end)})"
        if any('\u3400' <= ch <= '\u9fff' for ch in item['utterance']):
            return (f"\n\n——\n这句话是 {start:%Y-%m-%d %H:%M:%S}–{end:%H:%M:%S}（{self.args.timezone}）在 Eleagent 录像 "
                    f"{item['capture_id']} 里说的。需要看当时的画面，可以用 Eleagent 连接器的 {frames}；"
                    f"前后的视频片段用 {clips}。")
        return (f"\n\n——\nSaid {start:%Y-%m-%d %H:%M:%S}–{end:%H:%M:%S} ({self.args.timezone}) in Eleagent recording "
                f"{item['capture_id']}. To see that moment, use the Eleagent connector: {frames}; the video clips around it: {clips}.")

    def search(self, query):
        """Memory entries containing the query; misheard names (林小宇 for 林小雨) still share two-character pieces,
        so with no exact hit the entries sharing the most pieces come back."""
        query = query.strip()[:80]
        found = self.graph.read("""MATCH (e:Entity) WHERE e.layer='memory' AND e.kind IN ['person','object','place','event','routine']
            WITH e, [p IN $pieces WHERE toLower(coalesce(e.name,'') + ' ' + coalesce(e.description,'')) CONTAINS p] AS hit
            WHERE size(hit) > 0
            RETURN e.kind AS kind, e.name AS name, left(coalesce(e.description, ''), 240) AS description, e.t_last AS last_seen,
                   size(hit) AS score, $whole IN hit AS exact
            ORDER BY exact DESC, score DESC, coalesce(e.t_last, '') DESC LIMIT 8""",
            pieces=list(dict.fromkeys([query.lower()] + [query[i:i + 2].lower() for i in range(len(query) - 1)
                                                         if len(query) > 2 and query[i:i + 2].strip()])), whole=query.lower())
        if any(r['exact'] for r in found):
            found = [r for r in found if r['exact']]
        return '\n'.join(f"- {r['kind']}: {r['name']} — {r['description']} (last seen {r['last_seen']})" for r in found) or 'nothing found'

    def gate(self, text, turns_text):
        system = GATE_SYSTEM
        user = 'Transcript so far:\n' + turns_text + '\n\nWhom is the newest turn for?'
        content, seconds = vlm.call(self.args.vlm_url, {'model': self.args.vlm_model, 'temperature': 0, 'max_tokens': 300, **vlm.NO_THINK,
            'response_format': {'type': 'json_schema', 'json_schema': {'name': 'ear', 'schema': GATE_SCHEMA}},
            'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]}, timeout=30)
        got = json.loads(content)
        return got, seconds

    def process(self, db, capture_id, event_id, start_us, end_us, text, speaker, recorded_at=None):
        turns_text = self.transcript(db, capture_id, start_us, text, speaker)
        got, seconds = self.gate(text, turns_text)
        asking = got.get('for') == 'Okou' and float(got.get('confidence') or 0) >= self.args.min_confidence and got.get('request')
        if not asking:
            return None
        item = {'event_id': event_id, 'capture_id': capture_id, 'utterance': text, 'request': got['request'], 'start_us': start_us, 'end_us': end_us,
                'situation': got.get('situation'), 'confidence': got.get('confidence'), 'gate_s': round(seconds, 1)}
        context = self.moment(db, capture_id, end_us, text, speaker)
        handed, final, error, agent_s = asyncio.run(self.handle(item, context))
        error = error or handed.get('error')
        status = ('sent' if handed.get('thread_id') or (handed.get('message') and self.args.dry_run) and not handed.get('error')
                  else 'not_connected' if error and 'okou_not_connected' in error
                  else 'declined' if handed.get('declined') and not handed.get('message')
                  else 'not_handed' if not handed.get('message') else 'failed')
        event = {'id': 'assist_' + digest(event_id), 'event_id': event_id, 'capture_id': capture_id, 'utterance': text,
                 'request': item['request'], 'situation': item['situation'], 'message': handed.get('message'), 'note': handed.get('note'),
                 'declined': handed.get('declined'),
                 'okou_thread_id': handed.get('thread_id'), 'okou_event_id': handed.get('okou_event_id'), 'status': status,
                 'tools': ','.join(handed.get('tools') or []) or None,
                 'error': error, 'final': final or None, 'gate_s': item['gate_s'], 'agent_s': agent_s, 'created_at': now_iso(),
                 'utterance_end_to_handoff_ms': round((time.time() - end_us / 1e6) * 1000)}
        event = {k: v for k, v in event.items() if v is not None}
        if self.replayed is not None and status == 'sent':
            self.replayed.append({'event_id': event_id, 'capture_id': capture_id, 'request': item['request']})
        self.graph.write('MERGE (a:AgentEvent {id:$e.id}) ON CREATE SET a += $e', e=event)
        log(event='assist', context=context if self.args.dry_run else None, **{k: v for k, v in event.items() if k != 'message'},
            message=event.get('message'))
        return event

    # --- loop ------------------------------------------------------------------------------------------------------

    def start(self):
        if not self.graph.dry_run:
            for query in SCHEMA:
                self.graph.write(query)
        rows = self.graph.read('MATCH (s:AgentState {id:$id}) RETURN properties(s) AS s', id=STATE_ID)
        self.state = dict(rows[0]['s']) if rows else {}
        owner = self.graph.read(f"MATCH (e:Entity {{id:'{OWNER_ID}'}}) RETURN e.name AS name")
        self.state['owner_name'] = owner[0]['name'] if owner else None
        db = ro(self.args.archive_db)
        try:
            top = int(db.execute('SELECT coalesce(max(rowid), 0) FROM archive_events').fetchone()[0])
        finally:
            db.close()
        if self.state.get('event_cursor') is None:  # the first start begins now; no replay of history
            self.cursor = top
            self.graph.write('MERGE (s:AgentState {id:$id}) SET s.event_cursor=$c', id=STATE_ID, c=top)
        else:
            self.cursor = int(self.state['event_cursor'])
        log(event='start', cursor=self.cursor, archive_top=top, okou_thread=self.state.get('okou_thread_id'), dry_run=self.args.dry_run)

    def cycle(self):
        db = ro(self.args.archive_db)
        try:
            rows = db.execute("""SELECT rowid AS r, event_id, capture_id, start_us, end_us, body_json FROM archive_events
                WHERE rowid>? AND kind='speech' ORDER BY rowid LIMIT 50""", (self.cursor,)).fetchall()
            if rows and not vlm.serving(self.args.vlm_url):
                if not self.vlm_down:
                    log(event='vlm_unavailable', pending=len(rows)); self.vlm_down = True
                return
            self.vlm_down = False
            for row in rows:
                text = (json.loads(row['body_json']).get('text') or '').strip()
                if text:
                    try:
                        self.process(db, row['capture_id'], row['event_id'], row['start_us'], row['end_us'], text,
                                     json.loads(row['body_json']).get('speaker_entity_id'))
                    except Exception as error:  # one turn must never stop the ear; it is logged and passed
                        log(event='turn_failed', event_id=row['event_id'], error=type(error).__name__, detail=str(error)[:300])
                self.cursor = row['r']
                self.graph.write('MERGE (s:AgentState {id:$id}) SET s.event_cursor=$c, s.updated_at=$now', id=STATE_ID, c=self.cursor, now=now_iso())
        finally:
            db.close()

    def run(self):
        self.start()
        while True:
            try:
                self.cycle()
            except Exception as error:
                log(event='cycle_failed', error=type(error).__name__, detail=str(error)[:300])
            time.sleep(self.args.interval)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive-db', default=os.environ.get('ARCHIVE_DB', '/archive/archive.sqlite3'))
    parser.add_argument('--vlm-url', default=os.environ.get('VLM_URL', 'http://127.0.0.1:22410/v1'))
    parser.add_argument('--vlm-model', default=os.environ.get('VLM_MODEL', 'qwen3.6-35b-a3b'))
    parser.add_argument('--timezone', default=os.environ.get('ELEAGENT_TIMEZONE', 'America/Los_Angeles'))
    parser.add_argument('--token-url', default=os.environ.get('OKOU_TOKEN_URL', 'http://10.100.24.1:22404/internal/okou/token'))
    parser.add_argument('--token-secret-file', default=os.environ.get('AGENT_INTERNAL_TOKEN_FILE', '/agent/internal.token'))
    parser.add_argument('--okou-mcp', default=os.environ.get('OKOU_MCP_URL', OKOU_MCP))
    parser.add_argument('--okou-agent-id', default=os.environ.get('OKOU_AGENT_ID') or None)
    parser.add_argument('--min-confidence', type=float, default=.8)
    parser.add_argument('--interval', type=float, default=2.)
    parser.add_argument('--dry-run', action='store_true', help='nothing goes to Okou or the graph')
    parser.add_argument('--text', help='try one made-up newest turn against --capture (latest moment) and exit')
    parser.add_argument('--capture', help='with --text: the recording whose latest moment is used')
    parser.add_argument('--replay-rowids', type=lambda v: [int(x) for x in v.split(',') if x], default=None,
                        help='dry run: these archive speech rowids in order, as if heard live')
    args = parser.parse_args()
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(os.environ['GRAPH_URL'], auth=tuple(os.environ['NEO4J_AUTH'].split('/', 1)))
    try:
        agent = Agent(Graph(driver, args.dry_run), args)
        if args.replay_rowids:
            if not args.dry_run:
                parser.error('--replay-rowids is for --dry-run only')
            agent.start(); agent.replayed = []
            db = ro(args.archive_db)
            try:
                for rowid in args.replay_rowids:
                    row = db.execute("SELECT event_id, capture_id, start_us, end_us, body_json FROM archive_events WHERE rowid=? AND kind='speech'", (rowid,)).fetchone()
                    body = json.loads(row['body_json'])
                    event = agent.process(db, row['capture_id'], row['event_id'], row['start_us'], row['end_us'], (body.get('text') or '').strip(), body.get('speaker_entity_id'))
                    print(json.dumps({'rowid': rowid, 'text': body.get('text'), 'status': (event or {}).get('status'), 'message': (event or {}).get('message'),
                                      'declined': (event or {}).get('declined')}, ensure_ascii=False), flush=True)
            finally:
                db.close()
            return
        if args.text:
            agent.start()  # in a dry run its writes are no-ops
            db = ro(args.archive_db)
            try:
                row = db.execute("""SELECT max(end_us) AS end_us FROM archive_events WHERE capture_id=? AND kind='speech'""", (args.capture,)).fetchone()
                at = int(row['end_us'] or time.time() * 1e6) + 1
                event = agent.process(db, args.capture, 'manual_' + digest(args.text, str(time.time())), at, at, args.text, None)
            finally:
                db.close()
            print(json.dumps(event or {'asking': False}, ensure_ascii=False, indent=1))
            return
        agent.run()
    finally:
        driver.close()


if __name__ == '__main__':
    main()
