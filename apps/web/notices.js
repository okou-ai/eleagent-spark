// Toasts for what Eleagent did with what was said: spoken namings (naming service) and requests handed to Okou (day
// agent), from GET /v1/notices. Each entry is { id, name, status, memory_id, created_at }, newest first.
const REMEMBERED = new Set(['named', 'renamed', 'same_as']);

// The toast for one notice, or null for none.
export function namingMessage({ status, name } = {}) {
  if (REMEMBERED.has(status)) return name ? `已记住：${name}` : '已记住';
  if (status === 'no_target') return '没找到要命名的物品';
  if (status === 'assist') return !name ? '已交给 Okou' : name.startsWith('已交给') ? name : `已交给 Okou：${name}`;
  if (status === 'assist_unconnected') return 'Okou 还没连接';
  if (status === 'assist_failed') return '没能交给 Okou';
  return null;
}

// Green for what worked, grey otherwise.
export function noticeTone(status) {
  return REMEMBERED.has(status) || status === 'assist' ? 'ok' : 'none';
}

// Notices not seen before, oldest first. The first list a page receives only seeds what counts as seen, so opening
// the page does not replay old notices. seen is null before that first list.
export function freshNamings(seen, namings) {
  const valid = namings.filter(item => item && item.id != null);
  const fresh = seen ? valid.filter(item => !seen.has(item.id)).reverse() : [];
  return { fresh, seen: new Set([...valid.map(item => item.id), ...(seen ?? [])].slice(0, 200)) };
}
