// The sleeping brain's working memory: an in-memory copy of tonight's subgraph plus the memory
// operations the agent may perform. Nothing touches Neo4j here; result() returns the diff that
// run.mjs applies in one transaction after the agent finishes.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { J, localDate, localTime, zonedDateTime, mean, normalize } from './util.mjs';
import { coverFile, crop } from './candidates.mjs';
import { interval, readTimeline, readMedia, timelineLine } from './timeline.mjs';

const PERCEPTUAL = new Set(['object_appearance', 'face_appearance', 'object_cluster', 'face_cluster', 'voice_cluster', 'speech_segment', 'text_appearance']);
const TEXT_CAP = 40;
const FADES = new Set([...PERCEPTUAL, 'episode', 'session']);
// Memory entities are first-class kinds (person, object, place, event, routine, conversation, diary...) marked layer=memory.
export const MEMORY_SUBKINDS = ['person', 'object', 'place', 'event', 'routine', 'reading', 'conversation', 'diary'];
const isMem = (n) => n?.layer === 'memory';
const VEC = { object_appearance: 'object_vec', face_appearance: 'face_vec', speech_segment: 'voice_vec' };
const VEC_KEYS = ['object_vec', 'face_vec', 'voice_vec'];
const ENTITY = (n) => isMem(n) && !['conversation', 'diary'].includes(n.kind);
// Named out loud by the owner (naming service): the name is theirs; the memory is never forgotten or merged away.
const userNamed = (n) => n?.name_source === 'user';
const NAMED = ' [named by the owner]';
const dayId = (day) => `diary:${day.replace(/-/g, '')}`;
const COVER_CAP = 12, MEDIA_CAP = 24;
// One link per pair of entities, carrying nothing but natural-language text.
export const pairKey = (a, b) => (a < b ? `${a}\u0000${b}` : `${b}\u0000${a}`);

export class Memory {
  constructor(runDir) {
    this.dir = runDir;
    const ws = JSON.parse(fs.readFileSync(path.join(runDir, 'working.json'), 'utf8'));
    this.nodes = new Map(Object.entries(ws.nodes));
    this.edges = new Map(Object.entries(ws.edges));
    this.loaded = new Set(this.nodes.keys());
    // Batched nights: each batch is a separate agent run on the same in-memory working set; the state the
    // previous batch left (state.json) is carried forward and the graph is written once, after the final run.
    const batchFile = path.join(runDir, 'batch.json');
    this.batch = fs.existsSync(batchFile) ? JSON.parse(fs.readFileSync(batchFile, 'utf8')) : null;
    this.batchEpisodes = new Set(this.batch?.episodes || []);
    this.captures = ws.captures || [];
    this.day = ws.day;              // the local date this sleep is about
    const candFile = path.join(runDir, 'candidates.json');
    this.cands = new Map(fs.existsSync(candFile) ? JSON.parse(fs.readFileSync(candFile, 'utf8')).map((c) => [c.id, c]) : []);
    // A speech fragment has two independent uses: transcript evidence and voice identity.
    // Candidates retain the transcript; snapshot its media before any identity consumption or
    // batch state reload. working.json remains the original source for every batch of this night.
    this.conversationMedia = new Map([...this.cands.values()].filter((c) => c.modality === 'conversation').map((c) => [c.id,
      Object.freeze([...new Set(c.fragments.flatMap((f) => {
        const n = ws.nodes[f];
        return n ? J(n.media_ids) || [n.media_id] : [];
      }).filter(Boolean))])
    ]));
    this.clusterVotes = new Map();
    this.candToMemory = new Map(); // candidate id -> memory node id (for speaker labels)
    this.remembered = new Set();
    this.mergedInto = {};
    this.unlinked = new Map(); // links removed tonight (pair key -> [a, b])
    this.finished = false;
    this.timeline = new Map(); // inspected archive windows; transient, never part of saved state or graph
    this.journal = path.join(runDir, 'journal.jsonl');
    const stateFile = path.join(runDir, 'state.json');
    if (fs.existsSync(stateFile)) this.loadState(JSON.parse(fs.readFileSync(stateFile, 'utf8')));
    else this.foldDiaries();
  }

  // ---- batches ------------------------------------------------------------------------------

  inBatch(n) { return !this.batch || this.batch.final || this.batchEpisodes.has(n?.episode_id) || this.batchEpisodes.has(n?.id); }

  candInBatch(c) {
    if (!this.batch || this.batch.final) return true;
    if (c.modality === 'conversation') return c.t_first >= this.batch.t_from && c.t_first < this.batch.t_to;
    return c.fragments.some((f) => this.batchEpisodes.has(this.nodes.get(f)?.episode_id));
  }

  saveState() {
    const plain = (m) => Object.fromEntries(m);
    fs.writeFileSync(path.join(this.dir, 'state.json'), JSON.stringify({
      nodes: plain(this.nodes), edges: plain(this.edges), loaded: [...this.loaded], remembered: [...this.remembered],
      candToMemory: plain(this.candToMemory), mergedInto: this.mergedInto, unlinked: plain(this.unlinked),
      clusterVotes: Object.fromEntries([...this.clusterVotes].map(([k, v]) => [k, plain(v)])) }));
  }

  loadState(st) {
    this.nodes = new Map(Object.entries(st.nodes));
    this.edges = new Map(Object.entries(st.edges));
    this.loaded = new Set(st.loaded);
    this.remembered = new Set(st.remembered);
    this.candToMemory = new Map(Object.entries(st.candToMemory));
    this.mergedInto = st.mergedInto || {};
    this.unlinked = new Map(Object.entries(st.unlinked || {}));
    this.clusterVotes = new Map(Object.entries(st.clusterVotes || {}).map(([k, v]) => [k, new Map(Object.entries(v))]));
  }

  /** Batch runs leave the day's diary material for one stretch of time; the final run writes the whole diary. */
  async note_period({ title, text }) {
    if (!this.batch || this.batch.final) throw new Error('note_period is only for batch runs; write the whole diary with write_diary');
    if (!title || !text) throw new Error('title and text are required');
    const gid = dayId(this.day);
    const d = this.nodes.get(gid) || { id: gid, kind: 'diary', layer: 'memory', occurrences: 0, t_start: zonedDateTime(this.day).toISOString(), created_by: 'sleep-agent' };
    const notes = (J(d.notes) || []).filter((x) => x.batch !== this.batch.index);
    notes.push({ written_at: new Date().toISOString(), title, text, batch: this.batch.index, from: this.batch.t_from, to: this.batch.t_to });
    d.notes = notes.sort((a, b) => String(a.from || a.written_at).localeCompare(String(b.from || b.written_at)));
    d.occurrences = d.notes.length;
    this.nodes.set(gid, d);
    this.log('note_period', { batch: this.batch.index, chars: text.length });
    return { text: `Noted the diary material for batch ${this.batch.index + 1} (${localTime(this.batch.t_from)}–${localTime(this.batch.t_to)}).` };
  }

  async finish_batch() {
    if (!this.batch || this.batch.final) throw new Error('finish_batch is only for batch runs; use finish');
    const pending = [...this.cands.values()].filter((c) => c.modality === 'conversation' && !this.remembered.has(c.id) && this.candInBatch(c));
    if (pending.length) throw new Error(`${pending.length} conversation(s) of this batch are not remembered yet: ${pending.map((c) => c.id).join(', ')}`);
    if (!(J(this.nodes.get(dayId(this.day))?.notes) || []).some((x) => x.batch === this.batch.index)) throw new Error('Write this batch\'s diary material with note_period first');
    this.saveState();
    fs.writeFileSync(path.join(this.dir, `batch-${this.batch.index}.done`), new Date().toISOString());
    this.log('finish_batch', { batch: this.batch.index });
    return { text: `Batch ${this.batch.index + 1}/${this.batch.total} is done; its state is handed to the next batch.` };
  }

  /** Older runs wrote one diary per run (diary:20261010b, …); fold them into the day's single diary as material. */
  foldDiaries() {
    const gid = dayId(this.day);
    const extras = [...this.nodes.values()].filter((n) => n.kind === 'diary' && n.id !== gid && n.id.startsWith(gid)).sort((a, b) => String(a.created_at || '').localeCompare(String(b.created_at || '')));
    const d = this.nodes.get(gid) || (extras.length ? { id: gid, kind: 'diary', layer: 'memory', occurrences: 0, t_start: zonedDateTime(this.day).toISOString(), created_by: 'sleep-agent' } : null);
    if (!d) return;
    const notes = J(d.notes) || [];
    if (!notes.length && d.description && !d.integrated_at) notes.push({ written_at: d.created_at || '', title: d.name, text: d.description });
    for (const x of extras) {
      notes.push({ written_at: x.created_at || '', title: x.name, text: x.description });
      for (const [, e] of this.incident(x.id)) this.say(gid, e.a === x.id ? e.b : e.a, e.text);
      this.drop(x.id);
    }
    d.notes = notes;
    d.occurrences = notes.length;
    this.nodes.set(gid, d);
    if (extras.length) this.log('fold_diaries', { into: gid, folded: extras.map((x) => x.id) });
  }

  log(op, data) { fs.appendFileSync(this.journal, JSON.stringify({ op, at: new Date().toISOString(), ...data }) + '\n'); }

  expand(ids) {
    const out = [], missing = [];
    for (const i of ids) {
      const c = this.cands.get(i);
      if (c) out.push(...c.fragments.filter((f) => this.nodes.has(f)));
      else if (this.nodes.has(i)) out.push(i);
      else missing.push(i);
    }
    if (missing.length) throw new Error(`unknown ids: ${missing.slice(0, 5).join(', ')}`);
    return [...new Set(out)];
  }

  incident(id) { return [...this.edges.entries()].filter(([, e]) => e.a === id || e.b === id); }

  drop(id) {
    for (const [k] of this.incident(id)) this.edges.delete(k);
    this.nodes.delete(id);
  }

  /** Add a sentence to the link between a and b (created if missing); a sentence the link already says is not repeated. */
  say(a, b, sentence) {
    const text = String(sentence || '').trim();
    if (!text || a === b) return;
    const k = pairKey(a, b), e = this.edges.get(k);
    this.unlinked.delete(k);
    if (!e) this.edges.set(k, { a, b, text, _dirty: true });
    else if (!String(e.text || '').includes(text)) { e.text = e.text ? `${e.text}\n${text}` : text; e._dirty = true; }
  }

  nameOf(id) { return this.nodes.get(id)?.name || id; }

  evidenceLine(n) {
    const media = new Set(J(n.media_ids) || []);
    const transcript = J(n.transcript) || [];
    const conversations = new Set(this.incident(n.id).map(([, e]) => e.a === n.id ? e.b : e.a)
      .filter((id) => this.nodes.get(id)?.kind === 'conversation'));
    return `source material count ${Number(n.occurrences || 0)}; evidence time envelope ${n.t_start || 'unknown'} – ${n.t_last || 'unknown'}; ` +
      `${media.size} retained media references; ${transcript.length} retained transcript lines; ${conversations.size} linked conversations (inspect their content)`;
  }

  convLocked(ids) {
    // Speech that belongs to a conversation candidate must be remembered through remember_conversation.
    const locked = new Set([...this.cands.values()].filter((c) => c.modality === 'conversation' && !this.remembered.has(c.id)).flatMap((c) => c.fragments));
    return ids.filter((i) => locked.has(i));
  }

  // ---- views ------------------------------------------------------------------------------

  epLine(e) {
    const obs = J(e.observations) || [];
    const first = String(obs[0] || '').replace(/\n/g, ' ').slice(0, 110);
    const acc = e.user_acceleration_rms_mps2;
    const motion = acc == null || acc === '' ? '' : Number(acc) > 0.6 ? ' moving' : ' still';
    const place = e.lat != null && e.lat !== '' ? ` @${Number(e.lat).toFixed(4)},${Number(e.lon).toFixed(4)}` : '';
    return `${e.id} | ${localTime(e.t_start)} ${(Number(e.duration_s || 0) / 60).toFixed(1)}min ${e.camera_facing || ''}${motion}${place} | speech ${e.speech_count || 0}${Number(e.text_count || 0) ? ` text ${e.text_count}` : ''} | ${first}`;
  }

  candLine(c) {
    const f = c.fast || {};
    const sim = (c.similar_memories || []).map((h) => `${h.id}「${h.name}」${h.sim}`).join('；');
    const known = c.recognized_as ? ` | recognized as memory ${c.recognized_as} "${this.nodes.get(c.recognized_as)?.name || ''}"` : '';
    const hint = sim ? ` | maybe an existing memory: ${sim}` : '';
    if (c.modality === 'object') {
      return `${c.id} | ${c.label} ×${c.fragments.length} in ${c.episodes.length} episode(s) ${localTime(c.t_first)}–${localTime(c.t_last)} focus frames ${c.frames}` +
        (c.fast ? ` | first opinion: is ${c.label}=${f.really_label ?? '?'} same object=${f.same_object ?? '?'} on screen=${f.on_screen ?? '?'} ${f.what || ''} / ${f.setting || ''}` : '') + known + hint;
    }
    if (c.modality === 'voice') return `${c.id} | voice ${c.seconds}s in ${c.episodes.length} episode(s) ${localTime(c.t_first)}–${localTime(c.t_last)} | sample lines: ${(c.sample_lines || []).slice(0, 2).join(' / ').slice(0, 120)}${known}${hint}`;
    if (c.modality === 'face') return `${c.id} | face ×${c.count} in ${c.episodes.length} episode(s) ${localTime(c.t_first)}–${localTime(c.t_last)}${known}${hint}`;
    const done = this.remembered.has(c.id) ? ' [remembered]' : '';
    return `${c.id} | conversation ${localTime(c.t_first)}–${localTime(c.t_last)} ${c.transcript.length} lines ${c.seconds}s speakers ${c.speakers.join(',')}${done} | ${c.transcript.slice(0, 3).map((l) => l.text).join(' / ').slice(0, 100)}`;
  }

  /** The day's not-yet-consolidated fragments (episodes, candidates, readings), if any. */
  recallFragments() {
    if (this.batch?.final) return this.recallLeftovers();
    const eps = [...this.nodes.values()].filter((n) => n.kind === 'episode' && this.inBatch(n)).sort((a, b) => a.t_start.localeCompare(b.t_start));
    const live = [...this.cands.values()].filter((c) => this.candInBatch(c) && (c.modality === 'conversation' ? !this.remembered.has(c.id) : c.fragments.some((f) => this.nodes.has(f))));
    if (!eps.length && !live.length) return ['', '## New perception fragments', '(none: all of this day\'s capture was already consolidated; this sleep only integrates memories)'];
    const lines = ['', `## New perception fragments (${this.captures.length} captures; ${eps.length} episodes; ${live.length} candidates)`, '', '### Episodes (in time order)', ...eps.map((e) => this.epLine(e))];
    for (const [title, mod] of [['### Conversations (remember every one with remember_conversation)', 'conversation'], ['### Face candidates', 'face'], ['### Voice candidates', 'voice'], ['### Object candidates (clustering + a fast model\'s first opinion, which may be wrong)', 'object']]) {
      const list = live.filter((c) => c.modality === mod).sort((a, b) => (mod === 'conversation' ? a.t_first.localeCompare(b.t_first) : b.fragments.length - a.fragments.length));
      lines.push('', title, ...list.map((c) => this.candLine(c)));
    }
    const texts = [...this.nodes.values()].filter((n) => n.kind === 'text_appearance' && this.inBatch(n)).sort((a, b) => Number(b.reads || 0) - Number(a.reads || 0));
    const top = this.batch ? 150 : 60;
    lines.push('', `### Text read (quick impressions, may be misread; ${texts.length} in total, top ${top} by read count; view_episode shows all text of an episode)`, ...texts.slice(0, top).map((t) => `${t.id} | ${localTime(t.t_start)} read ${t.reads}x | ${t.carrier} | ${String(t.text).replace(/\n/g, ' ').slice(0, 120)}`));
    return lines;
  }

  /** Final run of a batched night: what the batches left behind, without listing every fragment again. */
  recallLeftovers() {
    const convs = [...this.cands.values()].filter((c) => c.modality === 'conversation' && !this.remembered.has(c.id));
    const left = [...this.nodes.values()].filter((n) => FADES.has(n.kind));
    const kinds = left.reduce((m, n) => ((m[n.kind] = (m[n.kind] || 0) + 1), m), {});
    const objects = [...this.cands.values()].filter((c) => c.modality !== 'conversation' && c.fragments.filter((f) => this.nodes.has(f)).length >= 5)
      .sort((a, b) => b.fragments.length - a.fragments.length).slice(0, 40);
    return ['', `## Perception fragments left after the batches (${Object.entries(kinds).map(([k, v]) => `${k} ${v}`).join(', ') || 'none'}; whatever is not consolidated fades at finish)`,
      '', '### Conversations not remembered yet (remember them with remember_conversation)', ...convs.map((c) => this.candLine(c)),
      '', '### Candidates with many fragments that no batch handled (mostly background; consolidate only if meaningful)', ...objects.map((c) => this.candLine(c))];
  }

  async recall_day() {
    const all = [...this.nodes.values()].filter(isMem);
    const today = all.filter((n) => this.dayOf(n).includes(this.day));
    const diary = this.nodes.get(dayId(this.day));
    const notes = J(diary?.notes) || [];
    const convs = today.filter((n) => n.kind === 'conversation').sort((a, b) => String(a.t_start).localeCompare(String(b.t_start)));
    const ents = today.filter(ENTITY);
    const todayIds = new Set(today.filter((n) => n.kind !== 'diary').map((n) => n.id));
    const links = [...this.edges.values()].filter((e) => (todayIds.has(e.a) || todayIds.has(e.b)) && ![e.a, e.b].some((x) => this.nodes.get(x)?.kind === 'diary'));
    const earlier = diary?.integrated_at ? ['', '## The day\'s existing diary (written by an earlier sleep; material only, you rewrite it whole)', String(diary.description)] : [];
    const scope = !this.batch ? '' : this.batch.final
      ? `\nThis is tonight\'s final integration run: ${this.batch.total} batches have each handled the fragments of their stretch and left material in the diary notes.`
      : `\nThis is batch ${this.batch.index + 1}/${this.batch.total} of tonight; it covers only ${localTime(this.batch.t_from)}–${localTime(this.batch.t_to)} (the new fragments below list only that stretch).`;
    const lines = [`Sleep date: ${this.day} (configured local time). This day already has ${ents.length} memory entities and ${convs.length} conversations; long-term memory holds ${all.filter(ENTITY).length} entities.${scope}`,
      ...this.recallFragments(), ...earlier,
      ...(notes.length ? ['', '## Earlier diary notes (material)', ...notes.map((x, i) => `### Note ${i + 1} (${x.title || ''})\n${x.text}`)] : []),
      '', '## The day\'s conversations', ...convs.map((c) => `${c.id} | ${localTime(c.t_start)}–${localTime(c.t_last)} | ${c.name} | ${String(c.summary || '').slice(0, 160)}`),
      '', '## Memory entities of the day', ...ents.map((g) => `${g.id} | ${g.kind} | ${g.name}${userNamed(g) ? NAMED : ''} | ${String(g.description || '').slice(0, 140)} | ${this.evidenceLine(g)}`),
      '', '## Links of the day (each link is one passage; rewrite ones that grew long, repetitive or outdated into one short passage with link rewrite)', ...links.map((e) => `${e.a} "${this.nameOf(e.a)}" — ${e.b} "${this.nameOf(e.b)}": ${String(e.text || '').replace(/\n/g, ' / ')}`),
      '', '## Possible duplicates (same kind with similar features or names; your call)', ...this.duplicateHints(all.filter(ENTITY)),
      '', '## All long-term memory entities (for de-duplication and merging)',
      'Evidence envelopes and source counts are not recall histories, independent encounters, importance scores or last-use dates. Diary/model links are not new observations. Missing evidence is unknown, not evidence of low importance.',
      ...all.filter(ENTITY).map((g) => `${g.id} | ${g.kind} | ${g.name}${userNamed(g) ? NAMED : ''} | ${String(g.description || '').slice(0, 80)} | ${this.evidenceLine(g)}`)];
    return { text: lines.join('\n') };
  }

  /** Evidence time only: a sleep after midnight stamps the previous day's memories with the next date, so created_at never places a memory on a day. */
  dayOf(n) { return [n.t_start, n.t_last].filter(Boolean).map((t) => localDate(new Date(t))); }

  duplicateHints(list) {
    const out = [];
    const sim = (a, b) => { const A = new Set(a), B = new Set(b); let c = 0; for (const x of A) if (B.has(x)) c++; return (2 * c) / (A.size + B.size || 1); };
    for (let i = 0; i < list.length; i++) for (let j = i + 1; j < list.length; j++) {
      const a = list[i], b = list[j];
      if (a.kind !== b.kind) continue;
      let v = 0;
      for (const k of VEC_KEYS) if (a[k] && b[k]) { const x = normalize(J(a[k]).map(Number)), y = normalize(J(b[k]).map(Number)); v = Math.max(v, x.reduce((s, xi, idx) => s + xi * y[idx], 0)); }
      const n = sim(String(a.name || ''), String(b.name || ''));
      if (v >= 0.8 || n >= 0.6) out.push(`${a.id} "${a.name}" ≈ ${b.id} "${b.name}" (feature similarity ${v.toFixed(2)}, name similarity ${n.toFixed(2)})`);
    }
    return out;
  }

  async view_episode({ id }) {
    const e = this.nodes.get(id);
    if (!e || e.kind !== 'episode') throw new Error('not an episode id');
    const voiceOf = new Map([...this.cands.values()].filter((c) => c.modality === 'voice').flatMap((c) => c.fragments.map((f) => [f, c.id])));
    const speech = [...this.nodes.values()].filter((n) => n.kind === 'speech_segment' && n.episode_id === id).sort((a, b) => a.t_start.localeCompare(b.t_start));
    const present = [...this.cands.values()].filter((c) => c.episodes.includes(id)).map((c) => c.id);
    let timeline;
    try {
      const { start, end } = interval(e);
      const events = await readTimeline(e.capture_id, start, end);
      for (const event of events) this.timeline.set(event.event_id, event);
      timeline = ['Timestamped archive evidence (configured local time; copy archive_window_id values into remember_event first/last, not the physical episode id):', ...events.map(timelineLine)];
      if (!events.length) timeline.push('(no timestamped observations available; do not invent event boundaries)');
    } catch {
      timeline = ['Timestamped archive evidence unavailable. The following graph observations are deduplicated/thinned, with no per-observation timestamps. Do not infer 5-second spacing or precise event boundaries from their order.',
        ...(J(e.observations) || []).slice(0, 120).map((o) => `- ${String(o).replace(/\n/g, ' ')}`)];
    }
    return { text: [`Physical episode (not a remember_event boundary): ${this.epLine(e)}`, '', ...timeline,
      '', 'Text read (a local model\'s quick glance; characters, negations and numbers are often wrong or missing; id | reads | carrier | text):', ...[...this.nodes.values()].filter((n) => n.kind === 'text_appearance' && n.episode_id === id)
        .sort((a, b) => Number(b.reads || 0) - Number(a.reads || 0)).slice(0, 150).map((t) => `- ${t.id} | ${t.reads} | ${t.carrier} | ${String(t.text).replace(/\n/g, ' ').slice(0, 200)}`),
      '', 'Transcript:', ...speech.slice(0, 120).map((s) => `- ${localTime(s.t_start)} [${voiceOf.get(s.id) || 'unknown voice'}] ${s.text || ''}`), '', `Candidates present in this episode: ${present.join(', ')}`].join('\n') };
  }

  async view_candidate({ id }) {
    const c = this.cands.get(id);
    if (!c) {
      const node = this.nodes.get(id);
      const hint = node?.kind === 'episode' ? 'This is a physical episode id; inspect it with view_episode.'
        : isMem(node) ? 'This is a saved memory id; inspect it with view_memory.'
        : 'Copy an exact candidate id from recall_day or the candidate list returned by view_episode; do not construct an id from a timestamp.';
      throw new Error(`Unknown candidate id: ${id}. ${hint}`);
    }
    const lines = [this.candLine(c), `Episodes: ${c.episodes.slice(0, 30).join(', ')}`];
    if (c.modality === 'voice') {
      const segs = c.fragments.map((f) => this.nodes.get(f)).filter((n) => n?.kind === 'speech_segment').sort((a, b) => a.t_start.localeCompare(b.t_start));
      lines.push('Lines spoken by this voice:', ...segs.slice(0, 80).map((s) => `- ${localTime(s.t_start)} ${s.text || ''}`));
    }
    if (c.modality === 'conversation') lines.push('Full transcript:', ...c.transcript.map((l) => `- ${localTime(l.t)} [${l.speaker}] ${l.text}`));
    if (!c.crops && c.samples?.length) c.crops = (await Promise.all(c.samples.map((s) => crop(s, path.join(this.dir, 'img')).catch(() => null)))).filter(Boolean);
    return { text: lines.join('\n'), images: c.crops || [] };
  }

  async view_memory({ id }) {
    const g = this.nodes.get(id);
    if (!isMem(g)) throw new Error('not a memory node');
    const links = this.incident(id).map(([, e]) => { const o = e.a === id ? e.b : e.a; return `- with ${o} "${this.nameOf(o)}": ${String(e.text || '').replace(/\n/g, ' / ')}`; });
    const crops = (J(g.covers) || []).slice(0, 6).map((c) => path.join(this.dir, 'img', coverFile(c))).filter((f) => fs.existsSync(f));
    const transcript = g.kind === 'conversation' ? (J(g.transcript) || []) : [];
    const fields = g.kind === 'conversation' ? `Summary: ${g.summary}\nFull transcript (${transcript.length} lines):\n${transcript.map((l) => `- ${localTime(l.t)} [${l.speaker}] ${l.text}`).join('\n')}` : `${g.description}`;
    return { text: [`${g.id} | ${g.kind} | ${g.name}${userNamed(g) ? NAMED : ''}`, fields, this.evidenceLine(g), links.length ? 'Links:' : 'No links', ...links.slice(0, 40)].join('\n'), images: crops };
  }

  async search_memory({ query }) {
    const q = (query || '').toLowerCase();
    const mem = [...this.nodes.values()].filter(isMem);
    const hits = mem.filter((g) => !q || `${g.name} ${g.description || ''} ${g.summary || ''}`.toLowerCase().includes(q));
    return { text: (hits.length ? hits : mem).slice(0, 60).map((g) => `${g.id} | ${g.kind} | ${g.name} | ${String(g.description || g.summary || '').slice(0, 100)}`).join('\n') || 'No memories at all' };
  }

  // ---- operations ---------------------------------------------------------------------------

  absorb(gid, frags) {
    const g = this.nodes.get(gid);
    const fs_ = new Set(frags);
    const ranges = [g, ...frags.map((f) => this.nodes.get(f))].map(interval).filter((r) => Number.isFinite(r.start));
    g.occurrences = Number(g.occurrences || 0) + frags.filter((f) => !/_cluster$/.test(this.nodes.get(f).kind) && !isMem(this.nodes.get(f))).length;
    if (ranges.length) {
      g.t_start = new Date(Math.min(...ranges.map((r) => r.start))).toISOString();
      g.t_last = new Date(Math.max(...ranges.map((r) => r.end))).toISOString();
      g.t_start_us = new Date(g.t_start).getTime() * 1000;
      g.t_end_us = new Date(g.t_last).getTime() * 1000;
    }
    for (const [kind, vk] of Object.entries(VEC)) {
      const vs = frags.map((f) => this.nodes.get(f)).filter((n) => n.kind === kind && n[vk]).map((n) => normalize(J(n[vk]).map(Number)));
      if (!vs.length) continue;
      const prev = g[vk] ? J(g[vk]).map(Number) : null, n0 = Number(g[`${vk}_n`] || 0);
      const cur = mean(vs);
      g[vk] = prev ? normalize(prev.map((x, i) => x * n0 + cur[i] * vs.length)) : cur;
      g[`${vk}_n`] = n0 + vs.length;
    }
    // Merging memories: prototypes combine weighted by how much evidence each carried.
    for (const vk of VEC_KEYS) {
      for (const m of frags.map((f) => this.nodes.get(f)).filter((n) => isMem(n) && n[vk])) {
        const v = J(m[vk]).map(Number), w = Number(m[`${vk}_n`] || 1);
        const prev = g[vk] ? J(g[vk]).map(Number) : null, n0 = Number(g[`${vk}_n`] || 0);
        g[vk] = prev ? normalize(prev.map((x, i) => x * n0 + v[i] * w)) : normalize(v);
        g[`${vk}_n`] = n0 + w;
      }
    }
    g.occurrences = Number(g.occurrences || 0) + frags.map((f) => this.nodes.get(f)).filter(isMem).reduce((s, m) => s + Number(m.occurrences || 0), 0);
    // Covers: where in the archive to look to see this memory (crop references gathered from its fragments).
    const covers = new Map((J(g.covers) || []).map((c) => [JSON.stringify(c), c])), media = new Set(J(g.media_ids) || []);
    for (const f of frags) {
      const n = this.nodes.get(f);
      for (const c of [...(J(n.crops) || []).slice(0, 3), ...(J(n.covers) || [])]) if (covers.size < COVER_CAP) covers.set(JSON.stringify(c), c);
      for (const m of J(n.media_ids) || (n.media_id ? [n.media_id] : [])) if (media.size < MEDIA_CAP) media.add(m);
    }
    const texts = new Set(J(g.texts) || []);
    for (const f of frags) { const n = this.nodes.get(f); if (n.kind === 'text_appearance' && n.text && texts.size < TEXT_CAP) texts.add(String(n.text).slice(0, 300)); }
    if (texts.size) g.texts = [...texts];
    g.covers = [...covers.values()];
    g.media_ids = [...media];
    const vote = (cluster, n) => {
      const votes = this.clusterVotes.get(cluster) || new Map();
      votes.set(gid, (votes.get(gid) || 0) + n);
      this.clusterVotes.set(cluster, votes);
    };
    let rewired = 0;
    for (const f of frags) {
      const fn = this.nodes.get(f);
      // The fragment's cluster (a node property) is resolved at finish by majority of where its members went.
      if (fn.cluster_id && !fs_.has(fn.cluster_id) && /_cluster$/.test(this.nodes.get(fn.cluster_id)?.kind || '')) vote(fn.cluster_id, 1);
      if (/_cluster$/.test(fn.kind)) vote(f, 1000); // the agent chose the whole cluster
      for (const [, e] of this.incident(f)) {
        const other = e.a === f ? e.b : e.a;
        if (fs_.has(other) || other === gid || FADES.has(this.nodes.get(other)?.kind)) continue;
        this.say(gid, other, e.text); // a merged memory's links move to the memory it became
        rewired++;
      }
    }
    for (const f of frags) this.drop(f);
    return rewired;
  }

  async consolidate({ ids = [], evidence = [], into, new: fresh, why }) {
    // Consumed identity material and reusable supporting memories have different lifetimes.
    // Validate both completely before changing any node, link, candidate mapping or journal.
    if (!Array.isArray(ids) || ids.some((id) => typeof id !== 'string')) throw new Error('ids must be an array of fragment or memory entity ids');
    if (!Array.isArray(evidence)) throw new Error('evidence must be an array of {id, text}');
    for (const id of ids) {
      const n = this.nodes.get(this.candToMemory.get(id) || id);
      if (this.cands.get(id)?.modality === 'conversation' || ['conversation', 'diary'].includes(n?.kind)) throw new Error(`${id} cannot be consumed; use evidence to reference a saved memory instead`);
    }
    const expanded = this.expand(ids);
    // Identity consolidation can consume speech before its conversation is saved: the
    // candidate transcript and conversationMedia preserve that independent evidence.
    let frags = expanded.filter((f) => FADES.has(this.nodes.get(f).kind) || ENTITY(this.nodes.get(f)));
    if (frags.length !== expanded.length) throw new Error('ids contains a node that cannot be consumed');
    if (into && !ENTITY(this.nodes.get(into))) throw new Error('into must be an existing memory entity id');
    // A memory the owner named survives a merge with its name: the others (including a requested target) merge into it.
    const named = [...new Set([into, ...frags].filter((id) => id && userNamed(this.nodes.get(id))))];
    if (named.length > 1) throw new Error(`${named.join(' and ')} were each named by the owner; they stay separate memories`);
    let redirect = null;
    if (named.length && named[0] !== into) {
      if (into === 'person:owner') throw new Error(`${named[0]} was named by the owner and cannot be merged into person:owner`);
      redirect = { from: into || null, description: fresh?.description || this.nodes.get(into)?.description };
      frags = [...frags.filter((f) => f !== named[0]), ...(into ? [into] : [])];
      into = named[0]; fresh = undefined;
    }
    const targetKind = into ? this.nodes.get(into)?.kind : fresh?.kind;
    if (targetKind === 'event' && (!into || frags.some((f) => this.nodes.get(f).kind !== 'event'))) {
      throw new Error('Create timed events with remember_event; do not consume physical episodes or fragments as semantic events');
    }
    const refs = evidence.map((ref) => {
      if (!ref || typeof ref.id !== 'string' || typeof ref.text !== 'string' || !ref.text.trim()) throw new Error('each evidence reference needs id and non-empty text');
      const id = this.candToMemory.get(ref.id) || ref.id;
      if (!isMem(this.nodes.get(id))) throw new Error(`${ref.id} is not a saved memory; remember the conversation first`);
      if (id === into || frags.includes(id) || ids.includes(ref.id)) throw new Error(`${ref.id} cannot be both reused evidence and the target or consumed identity material`);
      return { id, text: ref.text.trim() };
    });
    if (redirect && !frags.length && !refs.length) throw new Error(`${into} was named by the owner and keeps its name; nothing else was given to merge into it (revise can update its description)`);
    if (!frags.length && !refs.length) throw new Error('nothing to consolidate');
    let gid;
    if (into) {
      if (!isMem(this.nodes.get(into)) || ['conversation', 'diary'].includes(this.nodes.get(into).kind)) throw new Error('into must be an existing memory entity id');
      gid = into;
    } else {
      if (!fresh?.name || !fresh?.description || !fresh?.kind) throw new Error('new needs kind, name, description');
      if (!MEMORY_SUBKINDS.includes(fresh.kind) || ['conversation', 'diary'].includes(fresh.kind)) throw new Error(`kind must be one of person, object, place, event, routine, reading`);
      gid = `${fresh.kind}:${crypto.randomUUID().slice(0, 12)}`;
      this.nodes.set(gid, { id: gid, kind: fresh.kind, layer: 'memory', name: fresh.name, description: fresh.description, summary: fresh.name, text: fresh.description, occurrences: 0, created_by: 'sleep-agent', nights: 0 });
    }
    const g = this.nodes.get(gid);
    if (redirect?.description && !g.description) { g.description = redirect.description; g.text = redirect.description; }
    g.nights = Number(g.nights || 0) + (g._touched ? 0 : 1);
    g._touched = true;
    for (const i of ids) if (this.cands.has(i)) this.candToMemory.set(i, gid);
    const merged = frags.filter((f) => isMem(this.nodes.get(f)) && f !== gid);
    for (const m of merged) this.mergedInto[m] = gid;
    const rewired = this.absorb(gid, frags.filter((f) => f !== gid));
    // A memory created solely from spoken evidence still records when it was mentioned;
    // no source occurrences, visual/voice prototypes, covers or transcript are absorbed.
    if (!g.t_start) {
      const starts = refs.map(({ id }) => this.nodes.get(id).t_start).filter(Boolean).sort();
      const ends = refs.map(({ id }) => this.nodes.get(id).t_last || this.nodes.get(id).t_start).filter(Boolean).sort();
      if (starts.length) {
        g.t_start = starts[0]; g.t_last = ends.at(-1);
        g.t_start_us = new Date(g.t_start).getTime() * 1000;
        g.t_end_us = new Date(g.t_last).getTime() * 1000;
      }
    }
    for (const { id, text } of refs) this.say(gid, id, text);
    this.log('consolidate', { gist: gid, name: g.name, fragments: frags.length, merged_memories: merged, rewired, evidence: refs, why, ...(redirect && { kept_owner_named: gid, requested_into: redirect.from }) });
    return { text: `Consolidated ${frags.length} fragment(s) into ${gid} (${g.name})` + (rewired ? `, moved ${rewired} link(s) over` : '') + (refs.length ? `, citing ${refs.length} kept memory evidence item(s)` : '') + '.'
      + (redirect ? ` ${gid} "${g.name}" was named by the owner, so it is the surviving memory and keeps its name` + (redirect.from ? `; ${redirect.from} merged into it.` : '; no new memory was created.') : '') };
  }

  async remember_conversation({ id, title, summary, participants = [], why }) {
    const c = this.cands.get(id);
    if (!c || c.modality !== 'conversation') throw new Error('not a conversation candidate');
    if (this.remembered.has(id)) throw new Error('already remembered');
    if (!title || !summary) throw new Error('title and summary are required');
    const cid = `conversation:${c.fragments[0].split(':').pop()}`;
    const media = [...this.conversationMedia.get(id)];
    this.nodes.set(cid, { id: cid, kind: 'conversation', layer: 'memory', name: title, summary, summary_level: 0, text: c.transcript.map((l) => l.text).join(' '),
      transcript: c.transcript, speakers: c.speakers, t_start: c.t_first, t_last: c.t_last, t_start_us: new Date(c.t_first).getTime() * 1000,
      t_end_us: new Date(c.t_last).getTime() * 1000, duration_s: c.seconds, media_ids: media, occurrences: c.transcript.length, created_by: 'sleep-agent' });
    this.candToMemory.set(id, cid);
    for (const p of participants) {
      const target = this.candToMemory.get(p) || p;
      if (isMem(this.nodes.get(target))) this.say(cid, target, `${this.nameOf(target)} participated in "${title}" on ${localDate(new Date(c.t_first))} at ${localTime(c.t_first)}`);
    }
    // Saving words does not consume voice identity evidence. A later consolidate absorbs
    // each remaining speech fragment once; otherwise it fades at finish like other perception.
    this.remembered.add(id);
    this.log('remember_conversation', { conversation: cid, title, lines: c.transcript.length, participants, why });
    return { text: `Remembered conversation ${cid} "${title}" (${c.transcript.length} lines, transcript saved).` };
  }

  async remember_event({ first, last, title, description, evidence = [] }) {
    const invalid = Object.entries({ first, last }).filter(([, id]) => !this.timeline.has(id));
    if (invalid.length) {
      const problems = invalid.map(([field, id]) => this.nodes.get(id)?.kind === 'episode' || String(id).startsWith('episode:')
        ? `${field} received physical episode id ${id}, not an archive window id`
        : `${field}=${id} is not an inspected archive window id`);
      throw new Error(`${problems.join('; ')}. Inspect the source episode with view_episode, then copy the archive_window_id values from its timestamped evidence rows into first and last. Choose the first and last windows supported by the event evidence; passing the episode id again will not work.`);
    }
    const a = this.timeline.get(first), b = this.timeline.get(last);
    if (!title?.trim() || !description?.trim()) throw new Error('title and description are required');
    const start = Date.parse(a.started_at), end = Date.parse(b.ended_at);
    if (a.capture_id !== b.capture_id || Date.parse(b.started_at) < start || end < Date.parse(a.ended_at) || end <= start) throw new Error('Event boundaries must be ordered windows in the same capture');
    // Re-read the full interval, including physical episode boundaries. Never consume any source node.
    const events = await readTimeline(a.capture_id, start, end);
    if (![first, last].every((id) => events.some((e) => e.event_id === id))) throw new Error('The selected archive evidence is no longer available');
    const media = await readMedia(a.capture_id, start, end);
    if (!media.length) throw new Error('The selected interval has no archived media references; cannot save a traceable event');
    if (!Array.isArray(evidence)) throw new Error('evidence must be an array of {id, text}');
    const refs = evidence.map((ref) => {
      const id = this.candToMemory.get(ref?.id) || ref?.id;
      if (!isMem(this.nodes.get(id)) || typeof ref?.text !== 'string' || !ref.text.trim()) throw new Error('Each evidence item must reference a saved memory and explain its support');
      return { id, text: ref.text.trim() };
    });
    const duplicate = [...this.nodes.values()].find((n) => n.kind === 'event' && isMem(n) && Date.parse(n.t_start) === start && Date.parse(n.t_last) === end
      && (J(n.media_ids) || []).some((id) => media.includes(id)));
    if (duplicate) throw new Error(`An event already covers these exact boundaries: ${duplicate.id}; inspect and revise it instead`);
    const id = `event:${crypto.randomUUID().slice(0, 12)}`;
    this.nodes.set(id, { id, kind: 'event', layer: 'memory', name: title, description, summary: title, text: description,
      t_start: new Date(start).toISOString(), t_last: new Date(end).toISOString(), t_start_us: start * 1000, t_end_us: end * 1000,
      media_ids: media, occurrences: 0, created_by: 'sleep-agent', nights: 1, _touched: true });
    for (const ref of refs) this.say(id, ref.id, ref.text);
    this.log('remember_event', { event: id, first, last });
    return { text: `Remembered ${id} (${title}), ${new Date(start).toISOString()} – ${new Date(end).toISOString()}, with ${media.length} archived media references. Boundaries have observation-window precision; source episodes and conversations are unchanged.` };
  }

  async forget({ ids, why }) {
    const frags = this.expand(ids);
    const locked = this.convLocked(frags);
    if (locked.length || ids.some((i) => this.cands.get(i)?.modality === 'conversation')) throw new Error('conversations cannot be forgotten tonight');
    const kept = frags.filter((f) => userNamed(this.nodes.get(f)));
    const doomed = frags.filter((f) => !kept.includes(f) && (FADES.has(this.nodes.get(f).kind) || (ENTITY(this.nodes.get(f)) && f !== 'person:owner')));
    for (const f of doomed) this.drop(f);
    this.log('forget', { fragments: doomed.length, kept_owner_named: kept, why });
    return { text: `Forgot ${doomed.length} fragment(s).` + (kept.length ? ` Kept ${kept.map((f) => `${f} "${this.nameOf(f)}"`).join(', ')}: named by the owner, so ${kept.length > 1 ? 'they' : 'it'} cannot be forgotten.` : '') };
  }

  async link({ a, b, text, rewrite = false }) {
    const [A, B] = [a, b].map((x) => this.candToMemory.get(x) || x);
    for (const [x, t] of [[a, A], [b, B]]) if (!isMem(this.nodes.get(t))) throw new Error(`${x} is not a memory entity`);
    if (A === B) throw new Error('a and b are the same entity');
    const k = pairKey(A, B), sentence = String(text || '').trim();
    if (!rewrite) {
      if (!sentence) throw new Error('text is required');
      this.say(A, B, sentence);
    } else if (!sentence) {
      if (this.edges.delete(k)) this.unlinked.set(k, [A, B]);
    } else {
      this.edges.set(k, { ...(this.edges.get(k) || { a: A, b: B }), text: sentence, _dirty: true });
      this.unlinked.delete(k);
    }
    this.log('link', { a: A, b: B, text: sentence, rewrite });
    const now = this.edges.get(k)?.text;
    return { text: now ? `The link between ${A} "${this.nameOf(A)}" and ${B} "${this.nameOf(B)}" is now:\n${now}` : `Removed the link between ${A} and ${B}.` };
  }

  async revise({ id, name, description }) {
    const g = this.nodes.get(id);
    if (!isMem(g) || g.kind === 'diary') throw new Error('only memory entities can be revised');
    // The owner's spoken name stays; the description is still the sleep's to improve.
    const refused = userNamed(g) && name && name !== g.name;
    const note = refused ? `Kept the name "${g.name}": the owner named ${id}, so only its description can be revised.` : '';
    if (refused && !description) { this.log('revise', { gist: id, name: g.name, refused_name: name }); return { text: note }; }
    if (name && !refused) { g.name = name; g.summary = name; }
    if (description) { g.description = description; g.text = description; }
    g._touched = true;
    this.log('revise', { gist: id, name: g.name, ...(refused && { refused_name: name }) });
    return { text: refused ? `Updated the description. ${note}` : 'Updated.' };
  }

  /** ⟦HH:MM⟧ / ⟦HH:MM:SS⟧ anchors (configured local time) -> the archived clip covering that moment, so the diary points at raw video/audio. */
  async resolveMoments(day, text) {
    const out = [];
    const base = process.env.ARCHIVE_URL, token = process.env.SCENE_ARCHIVE_TOKEN;
    for (const m of new Set([...text.matchAll(/⟦(\d{1,2}:\d{2}(?::\d{2})?)⟧/g)].map((x) => x[1]))) {
      const [h, mi, se = '0'] = m.split(':');
      const t = zonedDateTime(day, Number(h), Number(mi), Number(se));
      const moment = { anchor: m, t: t.toISOString() };
      try {
        // Exact cover first, then the nearest clip within ±2 minutes (captures have small gaps).
        for (const pad of [0, 120_000]) {
          const q = new URLSearchParams({ started_at: new Date(t.getTime() - pad).toISOString(), ended_at: new Date(t.getTime() + pad).toISOString(), limit: '100' });
          const r = await fetch(`${base}/v1/archive/media?${q}`, { headers: { Authorization: `Bearer ${token}` }, signal: AbortSignal.timeout(15_000) });
          const items = r.ok ? ((await r.json()).items || []) : [];
          const best = items.map((x) => ({ x, d: Math.abs(new Date(x.captured_at).getTime() + Number(x.duration_seconds) * 500 - t.getTime()) })).sort((a, b) => a.d - b.d)[0]?.x;
          if (best) {
            const offset = Math.max(0, Math.min(Number(best.duration_seconds) - 0.1, (t.getTime() - new Date(best.captured_at).getTime()) / 1000));
            Object.assign(moment, { media_id: best.media_id, capture_id: best.capture_id, offset: +offset.toFixed(2), clip_captured_at: best.captured_at, duration_seconds: best.duration_seconds });
            break;
          }
        }
      } catch (e) { moment.error = String(e.message || e).slice(0, 80); }
      out.push(moment);
    }
    return out;
  }

  /** Every sleep rewrites the whole day's diary (last writer wins); earlier versions were shown as material. */
  async write_diary({ title, text, refs = [] }) {
    if (this.batch && !this.batch.final) throw new Error('A batch run only writes its stretch\'s material (note_period); the whole day\'s diary is written in the final integration run');
    const day = this.day, gid = dayId(day);
    const d = this.nodes.get(gid) || { id: gid, kind: 'diary', layer: 'memory', occurrences: 0, t_start: zonedDateTime(day).toISOString(), created_by: 'sleep-agent' };
    Object.assign(d, { name: title, summary: title, description: text, text, integrated_at: new Date().toISOString() });
    d.occurrences = (J(d.notes) || []).length;
    const found = await this.resolveMoments(day, text);
    d.moments = found.sort((a, b) => a.t.localeCompare(b.t));
    d._touched = true;
    this.nodes.set(gid, d);
    for (const r of refs) {
      const t = this.candToMemory.get(r) || r;
      if (isMem(this.nodes.get(t)) && t !== gid) this.say(gid, t, `The diary for ${day} mentions "${this.nameOf(t)}"`);
    }
    fs.appendFileSync(path.join(this.dir, 'diary.md'), `# ${day} ${title}\n\n${text}\n\n`);
    this.log('write_diary', { diary: gid, chars: text.length });
    const linked = found.filter((x) => x.media_id).length;
    return { text: `Rewrote the diary of ${day}.` + ` ${found.length} time anchor(s), ${linked} matched to raw video/audio` + (found.length > linked ? `; unmatched: ${found.filter((x) => !x.media_id).map((x) => x.anchor).join(', ')} (nothing was captured at that moment; use real times from the material)` : '') };
  }

  async finish() {
    if (this.batch && !this.batch.final) throw new Error('This is a batch run: end it with finish_batch');
    const pending = [...this.cands.values()].filter((c) => c.modality === 'conversation' && !this.remembered.has(c.id));
    if (pending.length) throw new Error(`${pending.length} conversation(s) are not remembered yet: ${pending.map((c) => c.id).join(', ')}`);
    if (!this.nodes.get(dayId(this.day))?._touched) throw new Error('Write the diary first (write_diary)');
    const clusterMap = {};
    this.keep = new Set();
    for (const id of [...this.nodes.keys()]) {
      const n = this.nodes.get(id);
      if (/_cluster$/.test(n.kind)) {
        const votes = [...(this.clusterVotes.get(id) || new Map()).entries()].sort((x, y) => y[1] - x[1]);
        // A cluster still used by captures that are not consolidated yet stays, unless it became a memory.
        if (!votes.length && Number(n._outside_members || 0) > 0) { this.keep.add(id); continue; }
        clusterMap[id] = votes[0]?.[0] || '';
      }
    }
    for (const [id, votes] of this.clusterVotes) if (!(id in clusterMap)) clusterMap[id] = [...votes.entries()].sort((x, y) => y[1] - x[1])[0][0];
    const faded = [...this.nodes.values()].filter((n) => FADES.has(n.kind) && !this.keep.has(n.id)).map((n) => n.id);
    for (const f of faded) this.drop(f);
    // Speaker labels in transcripts: voice candidates resolve to the memory they became.
    for (const n of this.nodes.values()) {
      if (n.kind === 'conversation' && Array.isArray(n.transcript)) {
        for (const sp of new Set(n.transcript.map((l) => l.speaker))) {
          const who = this.candToMemory.get(sp);
          if (who && this.nodes.has(who) && !this.edges.has(pairKey(n.id, who))) this.say(n.id, who, `${this.nameOf(who)} spoke in "${n.name}" (identified by voice)`);
        }
        n.transcript = n.transcript.map((l) => ({ ...l, speaker: this.nodes.get(this.candToMemory.get(l.speaker))?.name || l.speaker }));
        n.speakers = [...new Set(n.transcript.map((l) => l.speaker))];
      }
    }
    const deleted = [...this.loaded].filter((id) => !this.nodes.has(id));
    for (const id of this.keep) { const n = this.nodes.get(id); if (n) delete n._outside_members; }
    const upserts = [...this.nodes.values()].filter((n) => isMem(n) && (n._touched || !this.loaded.has(n.id))).map(({ _touched, ...rest }) => rest);
    const memIds = new Set([...this.nodes.values()].filter(isMem).map((n) => n.id));
    // Only links written tonight; links of deleted nodes go with them.
    const links = [...this.edges.values()].filter((e) => e._dirty && memIds.has(e.a) && memIds.has(e.b)).map(({ a, b, text }) => ({ a, b, text }));
    const unlinked = [...this.unlinked.values()].filter(([a, b]) => this.nodes.has(a) && this.nodes.has(b));
    const kinds = (ids) => ids.reduce((m, id) => ((m[id.split(':')[0]] = (m[id.split(':')[0]] || 0) + 1), m), {});
    const stats = { nodes_loaded: this.loaded.size, nodes_after: this.nodes.size, deleted: deleted.length, deleted_kinds: kinds(deleted), faded_by_default: faded.length,
      memories_written: upserts.length, conversations: upserts.filter((n) => n.kind === 'conversation').length, links: links.length, unlinked: unlinked.length, clusters_mapped: Object.keys(clusterMap).length };
    fs.writeFileSync(path.join(this.dir, 'result.json'), JSON.stringify({ captures: this.captures, deleted, upserts, links, unlinked, clusterMap, mergedInto: this.mergedInto, stats }));
    this.finished = true;
    this.log('finish', stats);
    return { text: JSON.stringify(stats) };
  }
}

export const TOOL_NAMES = ['recall_day', 'view_episode', 'view_candidate', 'view_memory', 'search_memory', 'consolidate', 'remember_conversation', 'remember_event', 'forget', 'link', 'revise', 'write_diary', 'finish', 'note_period', 'finish_batch'];
