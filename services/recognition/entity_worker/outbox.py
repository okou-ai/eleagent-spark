"""Deliver timeline events from the shared outbox to the archive (idempotent: the archive dedups by event id)."""
import re


def deliver(archive, store, events):
    for event in events:
        response = archive.post('/v1/archive/events', content=event['payload'],
            headers={'Content-Type': 'application/json'})
        if response.status_code in (400, 409, 413, 422):
            try:
                detail = response.json().get('detail', {})
                code = detail.get('code', '') if isinstance(detail, dict) else ''
            except (ValueError, AttributeError):
                code = ''
            if not isinstance(code, str) or not re.fullmatch(r'[a-z0-9_]{1,80}', code):
                code = 'archive_event_rejected'
            store.event_reject(event['id'], f'http_{response.status_code}:{code}')
            continue
        response.raise_for_status()
        receipt = response.json()
        if receipt.get('persisted') is not True or receipt.get('event_id') != event['id']:
            raise ValueError('timeline_receipt_mismatch')
        store.event_ack(event['id'])
