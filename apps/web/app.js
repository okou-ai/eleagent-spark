import { CaptureState, CLIP_MS, MAX_CLIP_BYTES, bufferHasRoom, bufferRecovered, captureCamera, formatAgo, formatElapsed,
  newCaptureId, recorderType, resumesAutomatically, uploadDisposition } from './capture-state.js';
import { openQueue } from './queue.js';
import { freshNamings, namingMessage, noticeTone } from './notices.js';

const $ = id => document.getElementById(id);
const CAMERA_KEY = 'eleagent-camera';
const STORAGE_FULL = 'Storage full';
const state = new CaptureState();
// Migrate devices that previously used private capture links. Never read the old secret.
try { localStorage.removeItem('eleagent-token'); } catch { /* Storage can be unavailable in private browsing. */ }
let queue, stream, recorder, wakeLock, clipTimer, locationWatch, locationFix, ticker, releaseTab, uploadTimer;
let session = null, ready = false, flipping = false, uploading = false, requestingWake = false;
let fatal = null, problem = null, blocked = null, uploadIssue = null, signInNavigationReady = false;
let pending = { count: 0, bytes: 0 };
let recordedMs = 0, recordingSince = 0;
let observation = null, observationTag = null, observationTimer, observing = false;
let camera = remembered(CAMERA_KEY) === 'front' ? 'front' : 'back';

function remembered(key) {
  try { return localStorage.getItem(key); } catch { return null; }
}
function remember(key, value) {
  try { localStorage.setItem(key, value); } catch { /* Private browsing can refuse storage; the value still lasts this visit. */ }
}

// Authentication belongs to the gateway; never receive or store observer credentials.
function signIn() {
  if (blocked === 'auth') return;
  blocked = 'auth';
  clearTimeout(uploadTimer);
  clearTimeout(observationTimer);
  pause('capture_interrupted');
  render();
  // Pause stops the recorder; its last segment must reach IndexedDB before navigation.
  void Promise.resolve(session).catch(() => {}).then(() => {
    signInNavigationReady = true;
    window.location.assign('/auth/sign-in?return_to=/');
  });
}
function sessionExpired(response) {
  return response.type === 'opaqueredirect' || response.status === 401 || response.status === 403 ||
    (response.status >= 300 && response.status < 400 && response.status !== 304);
}

function elapsed() {
  return recordedMs + (recordingSince ? performance.now() - recordingSince : 0);
}
function statusView() {
  if (fatal) return { text: fatal, tone: 'error' };
  if (!ready) return { text: '' };
  if (blocked === 'auth') return { text: 'Session expired · Sign in to resume. Saved clips remain on this device.', tone: 'error' };
  if (blocked === 'clip') return { text: 'Upload rejected', tone: 'error' };
  let view;
  if (state.phase === 'recording' && !flipping) view = { text: formatElapsed(elapsed()), tone: 'live' };
  else if (problem) view = { text: problem, tone: 'error' };
  else if (state.phase === 'paused' && !flipping) view = { text: 'Paused' };
  else view = { text: 'Starting' };
  if (uploadIssue && pending.count) view.text += ` · ${pending.count} waiting`;
  return view;
}
function render() {
  const view = statusView();
  $('status').textContent = view.text;
  $('status').dataset.tone = view.tone || '';
  const active = ['recording', 'acquiring'].includes(state.phase) || flipping;
  $('shutter').classList.toggle('recording', active);
  $('shutter').setAttribute('aria-label', active ? 'Pause' : 'Record');
  $('shutter').disabled = !ready || !!blocked || !!fatal;
  $('flip').disabled = !ready || !!fatal || flipping;
}
async function refreshQueue() {
  pending = await queue.stats();
  render();
}

async function requestWakeLock() {
  if (!navigator.wakeLock || state.phase !== 'recording' || document.hidden || wakeLock || requestingWake) return;
  requestingWake = true;
  try {
    const lock = await navigator.wakeLock.request('screen');
    if (state.phase !== 'recording' || document.hidden) { await lock.release(); return; }
    wakeLock = lock;
    lock.addEventListener('release', () => { if (wakeLock === lock) wakeLock = null; });
  } catch { /* Some browsers grant the lock only after a tap; the next tap retries. */ } finally { requestingWake = false; }
}
function stopDevices() {
  clearTimeout(clipTimer);
  clearInterval(ticker);
  if (recordingSince) recordedMs += performance.now() - recordingSince;
  recordingSince = 0;
  if (recorder?.state !== 'inactive') {
    try { recorder?.stop(); } catch { /* A track end may have already stopped the recorder. */ }
  }
  if (locationWatch !== undefined) navigator.geolocation.clearWatch(locationWatch);
  locationWatch = undefined;
  locationFix = null;
  stream?.getTracks().forEach(track => track.stop());
  stream = null;
  $('preview').srcObject = null;
  const lock = wakeLock;
  wakeLock = null;
  lock?.release().catch(() => {});
}
function pause(reason, interruption = 'Camera interrupted') {
  if (!['recording', 'acquiring'].includes(state.phase)) return;
  state.pause(reason);
  stopDevices();
  problem = reason === 'buffer_full' ? STORAGE_FULL : reason === 'capture_interrupted' ? (problem || interruption) : null;
  render();
}
function watchLocation() {
  if (!navigator.geolocation) return;
  locationWatch = navigator.geolocation.watchPosition(position => {
    const c = position.coords;
    locationFix = { captured_at: new Date(position.timestamp).toISOString(), latitude: c.latitude, longitude: c.longitude,
      accuracy_m: c.accuracy, altitude_m: c.altitude, altitude_accuracy_m: c.altitudeAccuracy,
      heading_deg: Number.isFinite(c.heading) ? c.heading : null, speed_mps: Number.isFinite(c.speed) && c.speed >= 0 ? c.speed : null };
  }, () => { locationFix = null; }, { enableHighAccuracy: false, maximumAge: 10_000, timeout: 15_000 });
}

function recordOne(activeStream, lens) {
  return new Promise((resolve, reject) => {
    const mimeType = recorderType(MediaRecorder);
    const current = new MediaRecorder(activeStream, { ...(mimeType ? { mimeType } : {}), videoBitsPerSecond: 1_200_000, audioBitsPerSecond: 64_000 });
    recorder = current;
    const parts = [];
    let meta, started = 0, failed = false;
    current.onstart = () => {
      if (state.phase !== 'recording') { current.stop(); return; }
      started = performance.now();
      meta = state.clip(lens);
      if (locationFix && Date.now() - Date.parse(locationFix.captured_at) < 30_000) meta.locations.push(locationFix);
      clipTimer = setTimeout(() => { if (current.state === 'recording') current.stop(); }, CLIP_MS);
    };
    current.ondataavailable = event => { if (event.data.size) parts.push(event.data); };
    current.onerror = () => { failed = true; };
    current.onstop = () => {
      clearTimeout(clipTimer);
      if (recorder === current) recorder = null;
      if (!meta) { resolve(null); return; }
      const duration = performance.now() - started;
      meta.captured_end_at = new Date().toISOString();
      if (failed) { reject(new Error('recording_failed')); return; }
      // Mobile browsers can defer timers while suspended. Never disguise a long segment as a five-second clip.
      if (duration > 6400) { reject(new Error('recording_suspended')); return; }
      const blob = new Blob(parts, { type: current.mimeType || parts[0]?.type || 'application/octet-stream' });
      if (duration < 200 || !blob.size) { resolve(null); return; }
      if (blob.size > MAX_CLIP_BYTES) { reject(new Error('clip_too_large')); return; }
      resolve({ meta, blob });
    };
    // Each recorder is new: every clip has its own container header and is independently decodable.
    current.start();
  });
}
async function captureLoop(activeStream, epoch, lens) {
  try {
    while (state.phase === 'recording' && state.epoch === epoch) {
      await refreshQueue();
      if (!bufferHasRoom(pending)) { pause('buffer_full'); break; }
      const clip = await recordOne(activeStream, lens);
      if (clip) {
        await queue.add(clip.meta, clip.blob);
        await refreshQueue();
        void upload();
      }
    }
  } catch (error) {
    if (error.message === 'buffer_full') pause('buffer_full');
    else pause('capture_interrupted', error.name === 'QuotaExceededError' ? STORAGE_FULL : 'Recording stopped');
  } finally {
    if (state.phase === 'recording' && state.epoch === epoch) pause('capture_interrupted');
  }
}

function start() {
  session ||= record().finally(() => {
    session = null;
    render();
    resumeIfAllowed();
  });
  return session;
}
async function record() {
  if (!ready || blocked || fatal || !['idle', 'paused'].includes(state.phase)) return;
  // A start requested in the background waits for the page to return.
  if (document.hidden) { state.pause('page_hidden'); return; }
  try { await refreshQueue(); } catch { fatal = 'Storage unavailable'; ready = false; return; }
  // A concurrent observation request may have expired the session while storage was loading.
  if (!ready || blocked || fatal) return;
  if (!bufferHasRoom(pending)) { problem = STORAGE_FULL; return; }
  // Resuming the same lens continues its capture; a different lens is a new capture.
  const resume = state.phase === 'paused' && captureCamera(state.captureId) === camera;
  const epoch = state.acquire(resume ? state.captureId : newCaptureId(camera));
  problem = null;
  render();
  try {
    const acquired = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: camera === 'front' ? 'user' : 'environment' },
      width: { ideal: 960 }, height: { ideal: 720 }, frameRate: { ideal: 15, max: 24 } }, audio: true });
    if (document.hidden) state.pause('page_hidden');
    if (!state.activate(epoch)) { acquired.getTracks().forEach(track => track.stop()); return; }
    stream = acquired;
    const facing = acquired.getVideoTracks()[0]?.getSettings().facingMode;
    const lens = facing === 'user' ? 'front' : facing === 'environment' ? 'back' : camera;
    if (lens !== captureCamera(state.captureId) && state.nextSeq === 0) {
      state.captureId = state.captureId.replace(/-(front|back)$/, `-${lens}`);
    }
    camera = lens;
    acquired.getTracks().forEach(track => track.addEventListener('ended', () => pause('capture_interrupted')));
    $('preview').classList.toggle('mirrored', lens === 'front');
    $('preview').srcObject = acquired;
    await $('preview').play().catch(() => {});
    if (state.phase !== 'recording' || state.epoch !== epoch) { stopDevices(); return; }
    recordingSince = performance.now();
    ticker = setInterval(render, 500);
    void requestWakeLock();
    watchLocation();
    render();
    await captureLoop(acquired, epoch, lens);
  } catch (error) {
    if (state.epoch === epoch) {
      state.pause('capture_interrupted');
      stopDevices();
      problem = error?.name === 'NotAllowedError' ? 'Allow camera access' : 'Camera unavailable';
    }
  }
}
function resumeIfAllowed() {
  if (session || flipping || !ready || blocked || fatal || document.hidden) return;
  if (problem === STORAGE_FULL || state.pauseReason === 'buffer_full') {
    if (bufferRecovered(pending)) void start();
  } else if (state.phase === 'idle' ? !problem : state.phase === 'paused' && resumesAutomatically(state.pauseReason)) {
    void start();
  }
}
function toggle() {
  if (['recording', 'acquiring'].includes(state.phase)) pause('user_pause');
  else void start();
}
async function flip() {
  if (!ready || flipping) return;
  camera = camera === 'back' ? 'front' : 'back';
  remember(CAMERA_KEY, camera);
  $('flip').classList.toggle('turned');
  // While paused, or before a pending start acquires the camera, the next start uses the new lens.
  if (!['recording', 'acquiring'].includes(state.phase)) { render(); return; }
  flipping = true;
  render();
  try {
    // Finish the current clip, then start a new capture on the other lens.
    pause('user_pause');
    await session;
  } finally {
    flipping = false;
  }
  void start();
}

async function upload() {
  if (!ready || uploading || blocked) return;
  if (!navigator.onLine) { uploadIssue = 'offline'; render(); return; }
  uploading = true;
  clearTimeout(uploadTimer);
  try {
    while (!blocked && navigator.onLine) {
      const item = await queue.first();
      if (!item) break;
      const body = new FormData();
      body.set('meta', JSON.stringify(item.meta));
      body.set('video', item.blob, item.blob.type.includes('webm') ? 'clip.webm' : 'clip.mp4');
      const abort = new AbortController();
      const timeout = setTimeout(() => abort.abort(), 70_000);
      let response;
      try {
        response = await fetch('/', { method: 'POST', body, signal: abort.signal, credentials: 'same-origin', redirect: 'manual' });
      } finally { clearTimeout(timeout); }
      if (sessionExpired(response)) { signIn(); break; }
      if (!response.ok) {
        const disposition = uploadDisposition(response.status);
        if (disposition === 'blocked') {
          // A rejected clip stays in the local buffer; reloading retries it.
          blocked = 'clip';
          pause('capture_interrupted');
          break;
        }
        throw new Error('retry_upload');
      }
      const receipt = await response.json();
      if (receipt.archived !== true || receipt.capture_id !== item.meta.capture_id || receipt.clip_seq !== item.meta.clip_seq || !receipt.media_id) {
        throw new Error('archive_receipt_missing');
      }
      await queue.remove(item);
      uploadIssue = null;
      await refreshQueue();
      resumeIfAllowed();
    }
    if (!navigator.onLine) uploadIssue = 'offline';
  } catch {
    uploadIssue = 'retry';
    if (navigator.onLine) uploadTimer = setTimeout(() => void upload(), 5000);
  } finally {
    uploading = false;
    await refreshQueue().catch(() => {});
  }
}

// The latest MiniCPM observation is polled separately from uploads, so it keeps updating while paused.
async function pollObservation() {
  clearTimeout(observationTimer);
  if (observing || document.hidden || blocked === 'auth') return;
  observing = true;
  try {
    const response = await fetch('/v1/observations/latest', { cache: 'no-store', credentials: 'same-origin', redirect: 'manual',
      headers: { ...(observationTag ? { 'If-None-Match': observationTag } : {}) } });
    if (sessionExpired(response)) { signIn(); return; }
    if (response.status === 200) {
      observation = await response.json();
      observationTag = response.headers.get('ETag');
    } else if (response.status === 204) {
      observation = observationTag = null;
    }
  } catch { /* Keep the last observation; the next poll retries. */ } finally { observing = false; }
  renderObservation();
  if (!document.hidden) observationTimer = setTimeout(() => void pollObservation(), 3000);
}
function renderObservation() {
  $('observation').hidden = !observation;
  if (!observation) return;
  const at = Date.parse(observation.captured_to || observation.observed_at);
  $('observation-age').textContent = Number.isFinite(at) ? formatAgo(at) : '';
  if ($('observation-text').textContent !== observation.text) $('observation-text').textContent = observation.text;
}
setInterval(renderObservation, 1000);

$('shutter').addEventListener('click', toggle);
$('flip').addEventListener('click', () => void flip());
document.addEventListener('pointerdown', () => void requestWakeLock());
document.addEventListener('visibilitychange', () => {
  if (document.hidden) { pause('page_hidden'); return; }
  void upload();
  void pollObservation();
  resumeIfAllowed();
  void requestWakeLock();
});
window.addEventListener('focus', () => void requestWakeLock());
window.addEventListener('online', () => { uploadIssue = null; render(); void upload(); });
window.addEventListener('offline', () => { uploadIssue = 'offline'; render(); });
window.addEventListener('pagehide', () => { pause('page_hidden'); ready = false; releaseTab?.(); });
window.addEventListener('pageshow', event => { if (event.persisted) window.location.reload(); });
window.addEventListener('beforeunload', event => {
  // The sign-in navigation already waits for the last segment to be durably queued.
  if (signInNavigationReady) return;
  if (session || pending.count || ['recording', 'acquiring'].includes(state.phase)) { event.preventDefault(); event.returnValue = ''; }
});

// The Okou mark: grey until Eleagent is connected to the owner's Okou (requests said to Okou go there), in colour
// once it is. Grey opens Okou's sign-in and comes back here; colour opens the connection page.
async function okouStatus() {
  const button = $('okou');
  if (!button) return;
  try {
    const response = await fetch('/okou/status', { cache: 'no-store' });
    const connected = response.ok && (await response.json()).connected === true;
    button.dataset.connected = String(connected);
    button.setAttribute('aria-label', connected ? 'Okou connected' : 'Connect Okou');
  } catch { /* offline: keep the last state */ }
}
$('okou')?.addEventListener('click', () => {
  location.href = $('okou').dataset.connected === 'true' ? '/okou' : '/okou/connect';
});

// What Eleagent did with what was said (a naming, a request handed to Okou), as a toast: GET /v1/notices every
// NOTICES_MS while the page is in front. The server answers 304 while nothing changed; the first list only seeds what
// counts as seen, so opening the page replays nothing.
const NOTICES_MS = 5_000, TOAST_MS = 3_500;
let noticesTag = null, noticesSeen = null, noticesTimer, noticing = false, toastTimer;
async function pollNotices() {
  clearTimeout(noticesTimer);
  if (noticing || document.hidden || blocked === 'auth') return;
  noticing = true;
  try {
    const response = await fetch('/v1/notices', { cache: 'no-store', credentials: 'same-origin', redirect: 'manual',
      headers: { ...(noticesTag ? { 'If-None-Match': noticesTag } : {}) } });
    if (response.status === 200) {
      const body = await response.json();
      noticesTag = response.headers.get('ETag');
      if (Array.isArray(body.namings)) announce(body.namings);
    }
  } catch { /* The next poll retries. */ } finally { noticing = false; }
  if (!document.hidden) noticesTimer = setTimeout(() => void pollNotices(), NOTICES_MS);
}
function announce(namings) {
  const { fresh, seen } = freshNamings(noticesSeen, namings);
  noticesSeen = seen;
  for (const item of fresh) {
    const text = namingMessage(item);
    if (!text) continue;
    const toast = $('toast');
    toast.textContent = text;
    toast.dataset.tone = noticeTone(item.status);
    toast.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { toast.hidden = true; }, TOAST_MS);
  }
}

async function initialize() {
  void okouStatus();
  void pollNotices();
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState !== 'visible') return;
    void okouStatus();
    void pollNotices();
  });
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia || !window.MediaRecorder || !window.indexedDB || !navigator.locks) {
    fatal = 'Open in Safari or Chrome over HTTPS';
    render();
    return;
  }
  // One tab owns capture and uploads. This prevents two pages from deleting or sending the same local queue.
  await navigator.locks.request('eleagent-capture-owner', { ifAvailable: true }, async lock => {
    if (!lock) { fatal = 'Already open in another tab'; render(); return; }
    try {
      queue = await openQueue();
      ready = true;
      await refreshQueue();
      void upload();
      resumeIfAllowed();
      await new Promise(resolve => { releaseTab = resolve; });
    } catch {
      ready = false;
      fatal = 'Storage unavailable';
      render();
    }
  });
}
render();
void pollObservation();
void initialize();
