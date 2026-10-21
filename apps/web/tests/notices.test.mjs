import test from 'node:test';
import assert from 'node:assert/strict';
import { freshNamings, namingMessage, noticeTone } from '../notices.js';

test('notice toasts: remembered names, no target, requests handed to Okou, nothing for the rest', () => {
  for (const status of ['named', 'renamed', 'same_as']) assert.equal(namingMessage({ status, name: '蓝白牙刷' }), '已记住：蓝白牙刷');
  assert.equal(namingMessage({ status: 'named', name: null }), '已记住');
  assert.equal(namingMessage({ status: 'no_target', name: null }), '没找到要命名的物品');
  assert.equal(namingMessage({ status: 'assist', name: '约林小雨明天下午开会' }), '已交给 Okou：约林小雨明天下午开会');
  assert.equal(namingMessage({ status: 'assist', name: '已交给 Okou：约林小雨明天下午开会' }), '已交给 Okou：约林小雨明天下午开会');
  assert.equal(namingMessage({ status: 'assist', name: null }), '已交给 Okou');
  assert.equal(namingMessage({ status: 'assist_unconnected', name: null }), 'Okou 还没连接');
  assert.equal(namingMessage({ status: 'assist_failed', name: 'x' }), '没能交给 Okou');
  assert.equal(namingMessage({ status: 'ignored', name: 'x' }), null);
  assert.equal(namingMessage(), null);
  assert.deepEqual(['named', 'same_as', 'assist', 'no_target', 'assist_failed'].map(noticeTone), ['ok', 'ok', 'ok', 'none', 'none']);
});

test('the first notice list only seeds what was seen; later ones yield new notices oldest first', () => {
  const first = freshNamings(null, [{ id: 'n2' }, { id: 'n1' }]);
  assert.deepEqual(first.fresh, []);
  const next = freshNamings(first.seen, [{ id: 'n4', status: 'named' }, { id: 'n3', status: 'no_target' }, { id: 'n2' }, null]);
  assert.deepEqual(next.fresh.map(item => item.id), ['n3', 'n4']);
  assert.deepEqual(freshNamings(next.seen, [{ id: 'n4' }, { id: 'n3' }]).fresh, []);
  // Once an id leaves the server's window it is still remembered, so it never shows again.
  assert.deepEqual(freshNamings(next.seen, [{ id: 'n5' }]).fresh.map(item => item.id), ['n5']);
});
