"""The local VLM (Qwen3.6-35B-A3B via vLLM, OpenAI-compatible) for the naming service: listening and looking.

Standard library only, so tests/eval_intent.py runs on a bare host python3. Thinking is disabled per request.
The prompts describe the setting, the inputs and what the answer is used for, and leave the judgement to the model
(no phrase lists or case rules). Both replies are constrained to a JSON schema.

listen(): the recording's transcript so far (with notes of what Eleagent did during it), the known objects and people
  and the newest turn -> is the user teaching Eleagent about a person or an object in view; which label; which known
  one, if any.
look(): the camera frame of that moment, plus a sheet of views of the known objects -> the object's box (bbox_2d,
  relative [0, 1000]), its label (the model can see the object, so it can undo similar-sounding speech recognition
  mistakes) and which known object it is, if any (it can compare the object with how the known ones look).
pick_person(): the frame with the candidate faces numbered -> which person the user is introducing, if they are
  introducing one at all (the listener only heard the words).
"""
from __future__ import annotations

import base64
import json
import re
import time
import urllib.request

INTENTS = ('name', 'rename', 'same_as', 'none')
KINDS = ('object', 'person')
INTENT_SCHEMA = {'type': 'object', 'additionalProperties': False,
    'required': ['situation', 'intent', 'kind', 'name', 'known', 'referent', 'confidence'],
    'properties': {'situation': {'type': 'string'}, 'intent': {'enum': list(INTENTS)}, 'kind': {'enum': list(KINDS)},
                   'name': {'type': ['string', 'null']},
                   'known': {'type': ['integer', 'null']}, 'referent': {'type': ['string', 'null']},
                   'confidence': {'type': 'number'}}}

SETTING = """Eleagent is a personal memory assistant. The user carries a phone camera that records their day. Speech \
recognition turns what is said into text one turn at a time: turns can be cut mid-sentence, can contain \
similar-sounding wrong words, and speaker labels are rough guesses. Most of what is said is ordinary conversation, \
with other people or to oneself. People talk about the things around them all day: what they think of them, \
what is wrong with them, what is written on them, where they got them, what they cost. That talk is for the people \
around them or for themselves, not for Eleagent.

Eleagent remembers the physical objects in the user's life. It records what the camera sees on its own and gives \
recurring objects descriptive names, by which it tells the user about their day. The user can also teach it while recording, so that things carry the user's own names: they know Eleagent \
is listening, and when they want it to remember something they say so plainly, the way they would show a thing to \
a friend: saying what an object is or what they call it, correcting a name, or noting that something in view is a \
thing Eleagent already knows. There are no fixed commands and any language may be used. The rest of the time they \
are not thinking about Eleagent at all. People usually hold the \
thing they talk about or point the camera at it.

Eleagent also remembers the people the user spends time with, by their faces and names. The user can introduce someone the same way, saying who the person in view is or what they call \
them, usually with the camera turned towards them. People also talk to and about each other by name all day; that \
is not an introduction. Pets are not part of this yet."""

LISTEN_SYSTEM = SETTING + """

You are Eleagent's listener. You receive the transcript of this recording so far, notes of what Eleagent did \
during it, the objects and people Eleagent knows, and the newest turn. Decide whether, with the newest turn, the \
user is teaching Eleagent about an object or a person in view.

What you return is acted on at once: the camera frame of that moment is searched for the object you describe, and \
the label you give is confirmed on the user's screen and kept in their memory: from then on Eleagent calls the \
thing by it whenever it comes up, in every summary and every answer about their day. Missed teaching makes the user repeat themselves; talk taken for teaching puts a label they never asked for \
into their memory.

Reply with JSON:
- situation: one sentence on what the user is doing with the newest turn and who it is for.
- intent: "name" (says what an object in view is or what to call it, or who a person in view is, the first time or \
again), "rename" (corrects a name given before), "same_as" (says an object or person in view is one Eleagent \
already knows, keeping its name) or "none".
- kind: "person" when the teaching is about a human being in view, else "object".
- name: the object's label: a few words in the user's language, short enough for a one-line confirmation on a \
phone screen, that say what the user calls the thing, for a person the name they call them by; for same_as the known one's name. \
null for none.
- known: the number of the known object or person the user means, if they mean one. Eleagent applies your answer to that object: it \
gets the label, or the new view is added to it. A wrong number changes some other object in the user's memory. \
With null Eleagent makes a new object, unless the camera shows the thing is one it already knows.
- referent: a short English visual description of the object meant, to find it in the frame. null for none.
- confidence: 0..1."""

LOOK_PROMPT = SETTING + """

You are Eleagent's eyes. The user just taught Eleagent about an object; the first image is the camera frame from \
that moment.

What was said:
{transcript}
Eleagent's listener understood: {intent} "{name}"; the object looks like: {referent}.

{known}
{sheet}
Your answer is kept in the user's memory, and Eleagent calls the thing by it from then on:
- seen: one sentence on the object they mean as you see it: what it is, what it looks like, where it is.
- bbox_2d: the object they mean, [x1, y1, x2, y2] in relative 0-1000 coordinates.
- name: its label: what the user calls this thing, in a few words of their language. The user knows what their things are, while a thing can look like something it is not, and the names \
Eleagent gave objects on its own are guesses from their looks. The transcript may have turned what the user said \
into similar-sounding wrong words; you can see the object and the whole conversation, so give the label the user \
meant.
- known: if this is an object Eleagent already knows, its number; else null. Eleagent applies the answer to that \
object (a wrong number changes some other object in the user's memory); with null it makes a new one, unless the \
object's appearance matches one it knows.
Reply with JSON only."""
LOOK_SCHEMA = {'type': 'object', 'additionalProperties': False, 'required': ['seen', 'bbox_2d', 'name', 'known'],
    'properties': {'seen': {'type': 'string'}, 'bbox_2d': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 4, 'maxItems': 4},
                   'name': {'type': 'string'}, 'known': {'type': ['integer', 'null']}}}

NO_THINK = {'chat_template_kwargs': {'enable_thinking': False}}


def turn_of(item):
    """A transcript turn: {'speaker', 'text', 'note'} dict, or a (speaker, text) pair (test cases)."""
    if isinstance(item, dict):
        return item.get('speaker'), item.get('text') or '', item.get('note')
    speaker, text = item[0], item[1]
    return speaker, text, (item[2] if len(item) > 2 else None)


def transcript(turns, newest=None):
    """Speakers become 'speaker 1', 'speaker 2'... in order of appearance; Eleagent's notes follow their turn."""
    labels, lines = {}, []
    def label(speaker):
        if not speaker:
            return 'unknown speaker'
        return labels.setdefault(speaker, f'speaker {len(labels) + 1}')
    for item in turns:
        speaker, text, note = turn_of(item)
        lines.append(f'[{label(speaker)}] {text}')
        if note:
            lines.append(f'    (Eleagent: {note})')
    if newest is not None:
        speaker, text, _ = turn_of(newest)
        lines.append(f'NEWEST [{label(speaker)}] {text}')
    return '\n'.join(lines)


def known_block(known, people=None):
    """Numbered from 1; the model answers with the number (ids are long and copy badly), known_id() maps it back.
    People (listen only) continue the objects' numbering, so object numbers mean the same in look's sheet."""
    entry = lambda n, e: f"[{n}] {e['name']}"
    lines = ['Objects Eleagent knows:\n' + '\n'.join(entry(n, o) for n, o in enumerate(known, 1))
             if known else 'Objects Eleagent knows: none yet.']
    if people is not None:
        lines.append('People Eleagent knows:\n' + '\n'.join(entry(n, p) for n, p in enumerate(people, len(known) + 1))
                     if people else 'People Eleagent knows: none yet.')
    return '\n\n'.join(lines)


def known_id(number, known):
    return known[number - 1]['id'] if isinstance(number, int) and not isinstance(number, bool) and 1 <= number <= len(known) else None


def listen_messages(text, turns=(), known=(), speaker=None, people=()):
    """turns: earlier turns of this recording, oldest first; known / people: [{'id', 'name'}] of the known objects and
    people."""
    user = ('Transcript so far:\n' + transcript(turns) if turns else 'Transcript so far: (this is the first turn)')
    user += '\n\n' + known_block(known, list(people)) + '\n\n' + transcript((), (speaker, text))
    return [{'role': 'system', 'content': LISTEN_SYSTEM}, {'role': 'user', 'content': user}]


def serving(url, timeout=3):
    """True when the server answers; it does not while it is stopped for the night or still loading."""
    try:
        with urllib.request.urlopen(url.rstrip('/') + '/models', timeout=timeout) as response:
            return response.status == 200
    except OSError:
        return False


def call(url, body, timeout=60):
    request = urllib.request.Request(url.rstrip('/') + '/chat/completions', data=json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.load(response)
    return data['choices'][0]['message']['content'], time.perf_counter() - started


def normalise_intent(raw, known=(), people=()):
    got = json.loads(raw) if isinstance(raw, str) else dict(raw)
    intent = got.get('intent') if got.get('intent') in INTENTS else 'none'
    kind = got.get('kind') if got.get('kind') in KINDS else 'object'
    try:
        confidence = min(1., max(0., float(got.get('confidence') or 0)))
    except (TypeError, ValueError):
        confidence = 0.
    name = (got.get('name') or '').strip() or None
    referent = (got.get('referent') or '').strip() or None
    known_ref = known_id(got.get('known'), list(known) + list(people))
    if known_ref:  # the entry the number points at says what kind it is
        kind = 'person' if any(p['id'] == known_ref for p in people) else 'object'
    if intent != 'none' and not name and not known_ref:
        intent, confidence = 'none', 0.
    situation = (got.get('situation') or '').strip()[:300] or None
    return {'situation': situation, 'intent': intent, 'kind': kind, 'name': name[:120] if name else None, 'known_id': known_ref,
            'referent': referent[:200] if referent else None, 'confidence': round(confidence, 3)}


def listen(url, model, text, turns=(), known=(), speaker=None, timeout=30, people=()):
    """-> ({intent, kind, name, known_id, referent, confidence}, seconds)."""
    content, seconds = call(url, {'model': model, 'temperature': 0, 'max_tokens': 300, **NO_THINK,
        'response_format': {'type': 'json_schema', 'json_schema': {'name': 'teaching', 'schema': INTENT_SCHEMA}},
        'messages': listen_messages(text, turns, known, speaker, people)}, timeout=timeout)
    return normalise_intent(content, known, people), seconds


def parse_look(content, known=()):
    """-> (first bbox_2d in relative 0..1000 or None, name or None, known_id or None) from the look reply."""
    text = content.strip().strip('`').removeprefix('json').strip()
    name = number = None
    try:
        parsed = json.loads(text)
        item = parsed[0] if isinstance(parsed, list) else parsed
        box = item['bbox_2d']
        name = item.get('name') if isinstance(item.get('name'), str) else None
        number = item.get('known')
    except Exception:
        match = re.search(r'\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]', content)
        if not match:
            return None, None, None
        box = [float(v) for v in match.groups()]
    name = (name or '').strip()[:120] or None
    ref = known_id(number, known)
    if not (isinstance(box, list) and len(box) == 4):
        return None, name, ref
    x1, y1, x2, y2 = (min(1000., max(0., float(v))) for v in box)
    return ([x1, y1, x2, y2] if x2 > x1 and y2 > y1 else None), name, ref


SHEET_TEXT = """The second image shows what the known objects look like: each tile is one view of one known object, \
marked with its number in the list above ({count} of the {total} known objects have a view). A view shows the \
object as Eleagent saw it once; the same object can look different from another side or in other light."""


def look(url, model, jpeg, width, height, turns, newest, heard, known=(), timeout=30, sheet=None):
    """heard: the listener's reply; sheet: (jpeg, numbers shown) of known-object views or None.
    -> (pixel box or None, label or None, known_id or None, raw reply, seconds)."""
    image = base64.b64encode(jpeg).decode()
    sheet_text = SHEET_TEXT.format(count=len(sheet[1]), total=len(known)) + '\n' if sheet else ''
    prompt = LOOK_PROMPT.format(transcript=transcript(turns, newest), intent=heard['intent'], name=heard.get('name') or '',
                                referent=heard.get('referent') or 'the object they are talking about', known=known_block(known),
                                sheet=sheet_text)
    content = [{'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{image}'}}]
    if sheet:
        content.append({'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + base64.b64encode(sheet[0]).decode()}})
    content, seconds = call(url, {'model': model, 'temperature': 0, 'max_tokens': 300, **NO_THINK,
        'response_format': {'type': 'json_schema', 'json_schema': {'name': 'look', 'schema': LOOK_SCHEMA}},
        'messages': [{'role': 'user', 'content': content + [{'type': 'text', 'text': prompt}]}]}, timeout=timeout)
    box, name, ref = parse_look(content, known)
    if box is None:
        return None, name, ref, content[:500], seconds
    return [box[0] * width / 1000, box[1] * height / 1000, box[2] * width / 1000, box[3] * height / 1000], name, ref, content[:500], seconds


PICK_PROMPT = SETTING + """

You are Eleagent's eyes. Eleagent's listener heard the user introduce a person; it only heard the words, and you \
can see the moment. The image is the camera frame, with the faces Eleagent found outlined and numbered 1 to {count} \
(the user's own face, if it was in view, is not among them).

What was said:
{transcript}
Eleagent's listener understood: {intent} "{name}"; the person looks like: {referent}.

The face you pick gets this name in the user's memory, whenever it is seen from now on; picking the wrong face \
puts the name on someone else.
- seen: one sentence on the person the user means as you see them.
- face: the number of their face, or null if none of the numbered faces is them (or the user is showing a thing, \
not introducing anyone; then Eleagent treats it as an object).
Reply with JSON only."""
PICK_SCHEMA = {'type': 'object', 'additionalProperties': False, 'required': ['seen', 'face'],
    'properties': {'seen': {'type': 'string'}, 'face': {'type': ['integer', 'null']}}}


def pick_person(url, model, jpeg, count, turns, newest, heard, timeout=30):
    """jpeg: the frame with `count` numbered face boxes drawn. -> (face number 1..count or None, raw reply, seconds)."""
    prompt = PICK_PROMPT.format(count=count, transcript=transcript(turns, newest), intent=heard['intent'],
                                name=heard.get('name') or '', referent=heard.get('referent') or 'the person they introduce')
    content, seconds = call(url, {'model': model, 'temperature': 0, 'max_tokens': 200, **NO_THINK,
        'response_format': {'type': 'json_schema', 'json_schema': {'name': 'pick', 'schema': PICK_SCHEMA}},
        'messages': [{'role': 'user', 'content': [
            {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + base64.b64encode(jpeg).decode()}},
            {'type': 'text', 'text': prompt}]}]}, timeout=timeout)
    try:
        number = json.loads(content).get('face')
    except Exception:
        number = None
    ok = isinstance(number, int) and not isinstance(number, bool) and 1 <= number <= count
    return (number if ok else None), content[:500], seconds
