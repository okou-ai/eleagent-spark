#!/usr/bin/env python3
"""Host-local daytime/Atlas handoff; HTTP control never accepts shell commands."""
import argparse
import datetime as dt
import fcntl
import hmac
import http.client
import http.server
import ipaddress
import json
import os
from pathlib import Path
import re
import signal
import socket
import stat
import subprocess
import threading
import time
import uuid
from zoneinfo import ZoneInfo

GIB = 1024 ** 3
ACTIVE = {'preparing', 'prepared', 'starting', 'running'}


class Cancelled(Exception):
    pass


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w') as f:
        os.fchmod(f.fileno(), 0o600)
        json.dump(value, f, separators=(',', ':'))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def memory():
    rows = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    vm = dict(line.split() for line in Path('/proc/vmstat').read_text().splitlines())
    return int(rows['MemAvailable'].split()[0]) * 1024, int(vm['oom_kill'])


def command(argv, timeout=20, allowed=(0,)):
    result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    if result.returncode not in allowed:
        raise RuntimeError('command_failed')
    return result


def inspect(ident):
    return json.loads(command(['docker', 'inspect', ident]).stdout)[0]


def emergency_kill(ident):
    # Independent of the worker's blocking Docker CLI calls.
    if not re.fullmatch('[0-9a-f]{64}', ident):
        return
    conn = http.client.HTTPConnection('localhost', timeout=2)
    conn.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    conn.sock.settimeout(2)
    try:
        conn.sock.connect('/var/run/docker.sock')
        conn.request('POST', '/containers/' + ident + '/kill?signal=SIGKILL', body=b'')
        response = conn.getresponse()
        response.read(1024)
    finally:
        conn.close()


class Manager:
    def __init__(self, config):
        self.c = config
        self.lock = threading.RLock()
        self.wake = threading.Event()
        self.stopping = threading.Event()
        self.path = Path(config['state_file'])
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.process_lock = self.path.with_suffix('.lock').open('a')
        fcntl.flock(self.process_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        secret_path = Path(config['secret_file'])
        st = secret_path.stat()
        if not stat.S_ISREG(st.st_mode) or stat.S_IMODE(st.st_mode) != 0o600:
            raise RuntimeError('secret_permissions')
        self.secret = secret_path.read_text().strip()
        if len(self.secret) < 32:
            raise RuntimeError('secret_too_short')
        self.s = json.loads(self.path.read_text()) if self.path.exists() else {
            'version': 1, 'phase': 'idle', 'lease': None, 'completed_lease': None}
        if self.s.get('version') != 1:
            raise RuntimeError('state_version')
        self.heartbeat = time.monotonic()
        self.memory_seen = time.monotonic()
        self.available, self.oom = memory()
        # A restart always restores the captured daytime state, never resumes GLM.
        if self.s['phase'] != 'idle':
            self.s.update(phase='restoring', reason='manager_restart')
            self.save()
        else:
            self.assert_no_rank()

    def save(self):
        atomic(self.path, self.s)

    def status(self):
        with self.lock:
            return {key: self.s.get(key) for key in (
                'version', 'phase', 'lease', 'completed_lease', 'reason', 'error_type',
                'deadline_epoch', 'rank_id', 'restored_at')}

    def rank(self):
        label = self.c['rank_label']
        ids = command(['docker', 'ps', '-aq', '--filter',
                       'label=' + label['key'] + '=' + label['value']]).stdout.split()
        if len(ids) > 1:
            raise RuntimeError('multiple_owned_ranks')
        if not ids:
            return None
        c = inspect(ids[0])
        if (c['Name'].lstrip('/') != self.c['rank_name'] or
                c['Image'] != self.c['rank_image'] or
                c['Config']['Labels'].get(label['key']) != label['value']):
            raise RuntimeError('rank_identity_changed')
        return c

    def assert_no_rank(self):
        if self.rank() is not None:
            raise RuntimeError('rank_without_active_state')

    def request(self, action, lease):
        try:
            if str(uuid.UUID(lease)) != lease:
                raise ValueError()
        except (ValueError, TypeError, AttributeError):
            return 400, {'error_type': 'invalid_lease'}
        with self.lock:
            phase = self.s['phase']
            if action == 'acquire':
                if phase != 'idle':
                    return (200, self.status()) if lease == self.s.get('lease') else (409, {'error_type': 'lease_conflict'})
                if self.s.get('completed_lease') == lease:
                    return 409, {'error_type': 'lease_already_completed'}
                now = dt.datetime.now(ZoneInfo(self.c['timezone']))
                minute = now.hour * 60 + now.minute
                if not self.c['acquire_start_minute'] <= minute < self.c['acquire_end_minute']:
                    return 409, {'error_type': 'outside_acquire_window'}
                # Memory is checked in prepare(), once the daytime services have released theirs.
                _, oom = memory()
                stop_hour, stop_minute = divmod(self.c['hard_stop_minute'], 60)
                deadline = now.replace(hour=stop_hour, minute=stop_minute, second=0, microsecond=0).timestamp()
                self.s = {'version': 1, 'phase': 'preparing', 'lease': lease,
                          'completed_lease': self.s.get('completed_lease'),
                          'deadline_epoch': deadline, 'baseline_oom': oom,
                          'daytime': [], 'pauses': [], 'rank_id': None,
                          'acquired_at': time.time()}
                self.heartbeat = time.monotonic()
                self.save()
                self.wake.set()
                return 202, self.status()
            if action == 'release' and phase == 'idle' and self.s.get('completed_lease') == lease:
                return 200, self.status()
            if lease != self.s.get('lease'):
                return 409, {'error_type': 'lease_conflict'}
            if action == 'heartbeat':
                if phase not in ACTIVE:
                    return 409, self.status()
                self.heartbeat = time.monotonic()
                return 200, self.status()
            if action == 'start':
                if phase in {'starting', 'running'}:
                    return 200, self.status()
                if phase != 'prepared':
                    return 409, self.status()
                self.s['phase'] = 'starting'
                self.save()
                self.wake.set()
                return 202, self.status()
            if action == 'release':
                self.restore('released')
                return 202, self.status()
            return 404, {'error_type': 'unknown_action'}

    def restore(self, reason):
        with self.lock:
            if self.s['phase'] == 'idle':
                return
            if self.s['phase'] != 'restoring':
                self.s.update(phase='restoring', reason=reason)
                self.save()
            self.wake.set()

    def check(self, expected):
        with self.lock:
            if self.s['phase'] != expected:
                raise Cancelled()

    def prepare(self):
        # Record original IDs and policy before changing dispatch or Docker state.
        rows = []
        for name in self.c['daytime_containers']:
            self.check('preparing')
            c = inspect(name)
            if not c['State']['Running'] or c['State'].get('Paused'):
                raise RuntimeError('daytime_not_running')
            rows.append({'id': c['Id'], 'name': name, 'image': c['Image'],
                         'restart_policy': c['HostConfig']['RestartPolicy']})
        self.assert_no_rank()
        with self.lock:
            self.check('preparing')
            self.s['daytime'] = rows
            self.s['pauses'] = [{'path': p, 'existed': Path(p).exists()} for p in self.c.get('pause_files', [])]
            self.save()
        for pause in self.s['pauses']:
            self.check('preparing')
            if pause['existed']:
                raise RuntimeError('dispatch_already_paused')
            target = Path(pause['path'])
            target.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive creation preserves another controller's pause.
            with target.open('x') as f:
                os.fchmod(f.fileno(), 0o600)
                f.write(self.s['lease'] + '\n')
                f.flush()
                os.fsync(f.fileno())
        end = time.monotonic() + self.c.get('drain_timeout_seconds', 600)
        while True:
            self.check('preparing')
            drained = True
            for argv in self.c.get('drain_commands', []):
                self.check('preparing')
                result = command(argv, timeout=20, allowed=(0, 1, 75))
                drained = drained and result.returncode == 0
            if drained:
                break
            if time.monotonic() >= end:
                raise RuntimeError('daytime_drain_timeout')
            self.wake.wait(1)
            self.wake.clear()
        for row in rows:
            self.check('preparing')
            c = inspect(row['id'])
            if c['Name'].lstrip('/') != row['name'] or c['Image'] != row['image']:
                raise RuntimeError('daytime_identity_changed')
            command(['docker', 'update', '--restart=no', row['id']])
            self.check('preparing')
            command(['docker', 'stop', '-t', '30', row['id']], timeout=40)
            if inspect(row['id'])['State']['Running']:
                raise RuntimeError('daytime_still_running')
        end = time.monotonic() + 60
        while memory()[0] < self.c.get('recovery_available_gib', 64) * GIB:
            self.check('preparing')
            if time.monotonic() >= end:
                raise RuntimeError('insufficient_available_memory')
            self.wake.wait(1)
            self.wake.clear()
        with self.lock:
            self.check('preparing')
            self.s['phase'] = 'prepared'
            self.save()

    def start(self):
        self.check('starting')
        self.assert_no_rank()
        # The create program is pinned configuration, never supplied by HTTP.
        # Any uncertain create is resolved by labelled inspection during restore.
        command(self.c['rank_create_command'], timeout=60)
        c = self.rank()
        if c is None or c['State']['Status'] != 'created' or c['HostConfig']['RestartPolicy']['Name'] != 'no':
            raise RuntimeError('rank_create_invalid')
        with self.lock:
            self.s['rank_id'] = c['Id']
            self.save()
            self.check('starting')
        command(['docker', 'start', c['Id']], timeout=30)
        c = inspect(c['Id'])
        if not c['State']['Running']:
            raise RuntimeError('rank_not_running')
        with self.lock:
            self.check('starting')
            self.s['phase'] = 'running'
            self.save()

    def recover(self):
        # Each observed stage is safe to repeat after a failed/uncertain CLI call.
        c = self.rank()
        if c:
            with self.lock:
                known = self.s.get('rank_id')
                if known and known != c['Id']:
                    raise RuntimeError('rank_id_changed')
                self.s['rank_id'] = c['Id']
                self.save()
            if c['State']['Running']:
                command(['docker', 'kill', c['Id']], timeout=10, allowed=(0, 1))
                if inspect(c['Id'])['State']['Running']:
                    raise RuntimeError('rank_still_running')
            command(['docker', 'rm', c['Id']], timeout=20)
        # Wait for a rank's memory before daytime starts again. Without a rank nothing will release it, and once
        # daytime runs (the naming VLM alone holds ~46 GB) the floor no longer applies.
        stopped = any(not inspect(row['id'])['State']['Running'] for row in self.s.get('daytime', []))
        if stopped and (c or self.s.get('rank_id')) and memory()[0] < self.c.get('recovery_available_gib', 64) * GIB:
            raise RuntimeError('waiting_memory_recovery')
        for row in self.s.get('daytime', []):
            c = inspect(row['id'])
            if c['Name'].lstrip('/') != row['name'] or c['Image'] != row['image']:
                raise RuntimeError('daytime_identity_changed')
            policy = row['restart_policy']
            restart = policy['Name'] or 'no'
            if restart == 'on-failure' and policy.get('MaximumRetryCount'):
                restart += ':' + str(policy['MaximumRetryCount'])
            if not c['State']['Running']:
                command(['docker', 'start', row['id']], timeout=30)
            if not inspect(row['id'])['State']['Running']:
                raise RuntimeError('daytime_restore_not_running')
            if c['HostConfig']['RestartPolicy'] != policy:
                command(['docker', 'update', '--restart=' + restart, row['id']])
        for argv in self.c.get('ready_commands', []):
            if command(argv, timeout=25, allowed=(0, 1, 75)).returncode != 0:
                raise RuntimeError('waiting_daytime_ready')
        for pause in self.s.get('pauses', []):
            target = Path(pause['path'])
            if not pause['existed'] and target.exists():
                if target.read_text().strip() != self.s['lease']:
                    raise RuntimeError('pause_owner_changed')
                target.unlink()
        with self.lock:
            old = self.s
            self.s = {'version': 1, 'phase': 'idle', 'lease': None,
                      'completed_lease': old['lease'], 'reason': old.get('reason'),
                      'restored_at': time.time()}
            self.save()

    def worker(self):
        while not self.stopping.is_set():
            with self.lock:
                phase = self.s['phase']
            try:
                if phase == 'preparing':
                    self.prepare()
                elif phase == 'starting':
                    self.start()
                elif phase == 'restoring':
                    self.recover()
                elif phase == 'running':
                    c = self.rank()
                    if c is None or not c['State']['Running'] or c['State'].get('OOMKilled'):
                        self.restore('rank_exited')
            except Cancelled:
                pass
            except Exception as exc:
                with self.lock:
                    self.s['error_type'] = type(exc).__name__
                    # Exception messages are fixed tokens authored here. Never log
                    # subprocess output, secret values, paths or data payloads.
                    if isinstance(exc, RuntimeError) and re.fullmatch('[a-z_]+', str(exc)):
                        self.s['error_type'] = str(exc)
                    self.save()
                self.restore('operation_failed')
            self.wake.wait(1)
            self.wake.clear()

    def watchdog(self):
        last_restore_kill = 0
        while not self.stopping.wait(0.25):
            reason = None
            try:
                available, oom = memory()
                with self.lock:
                    self.available, self.oom = available, oom
                    self.memory_seen = time.monotonic()
                    phase = self.s['phase']
                    rank = self.s.get('rank_id')
                    if phase in ACTIVE:
                        if available < self.c.get('min_available_gib', 4) * GIB:
                            reason = 'low_memory'
                        elif oom > self.s['baseline_oom']:
                            reason = 'host_oom'
                        elif time.time() >= self.s['deadline_epoch']:
                            reason = 'hard_deadline'
                        elif time.monotonic() - self.heartbeat > self.c.get('lease_ttl_seconds', 120):
                            reason = 'lease_expired'
                    elif phase == 'restoring' and rank and time.monotonic() - last_restore_kill >= 1:
                        # Covers a docker-start request which was in flight when
                        # cancellation arrived; exact old IDs cannot be reused.
                        reason = 'restore_in_progress'
                        last_restore_kill = time.monotonic()
            except Exception:
                reason = 'memory_monitor_failed'
                with self.lock:
                    rank = self.s.get('rank_id')
            if reason:
                # Kill first; state fsync and worker Docker calls can wait.
                if rank:
                    try:
                        emergency_kill(rank)
                    except Exception:
                        pass
                self.restore(reason)


def handler(manager):
    class Handler(http.server.BaseHTTPRequestHandler):
        server_version = 'EleagentGPU/1'

        def setup(self):
            super().setup()
            self.connection.settimeout(5)

        def log_message(self, *_args):
            pass

        def reply(self, code, value):
            body = json.dumps(value, separators=(',', ':')).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def authorized(self):
            return hmac.compare_digest(self.headers.get('Authorization', ''), 'Bearer ' + manager.secret)

        def do_GET(self):
            if not self.authorized():
                self.reply(401, {'error_type': 'unauthorized'})
            elif self.path != '/status':
                self.reply(404, {'error_type': 'not_found'})
            else:
                self.reply(200, manager.status())

        def do_POST(self):
            if not self.authorized():
                return self.reply(401, {'error_type': 'unauthorized'})
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 256 or self.headers.get('Transfer-Encoding'):
                    raise ValueError()
                self.connection.settimeout(5)
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict) or set(body) != {'lease'}:
                    raise ValueError()
            except (ValueError, OSError):
                return self.reply(400, {'error_type': 'invalid_request'})
            if self.path not in {'/acquire', '/heartbeat', '/start', '/release'}:
                return self.reply(404, {'error_type': 'not_found'})
            try:
                code, value = manager.request(self.path[1:], body['lease'])
            except Exception:
                return self.reply(500, {'error_type': 'control_failed'})
            self.reply(code, value)
    return Handler


def validate(config):
    required = {'listen_ip', 'secret_file', 'state_file', 'daytime_containers', 'rank_name',
                'rank_image', 'rank_label', 'rank_create_command', 'pause_files', 'drain_commands', 'ready_commands',
                'allowed_bind_addresses', 'timezone', 'acquire_start_minute', 'acquire_end_minute', 'hard_stop_minute'}
    if not required <= config.keys():
        raise RuntimeError('missing_configuration')
    allowed = config['allowed_bind_addresses']
    if not isinstance(allowed, list) or not allowed or config['listen_ip'] not in allowed:
        raise RuntimeError('invalid_bind_address')
    for value in allowed:
        address = ipaddress.ip_address(value)
        if address.version != 4 or address.is_unspecified or address.is_multicast or not address.is_private:
            raise RuntimeError('unsafe_bind_address')
    ZoneInfo(config['timezone'])
    start, end, stop = (config[key] for key in
                        ('acquire_start_minute', 'acquire_end_minute', 'hard_stop_minute'))
    if not all(type(value) is int for value in (start, end, stop)) or not 0 <= start < end < stop < 1440:
        raise RuntimeError('invalid_acquire_window')
    if type(config.get('port', 9210)) is not int or not 1024 <= config.get('port', 9210) <= 65535:
        raise RuntimeError('invalid_port')
    if not re.fullmatch('sha256:[0-9a-f]{64}', config['rank_image']):
        raise RuntimeError('unpinned_rank_image')
    if not config['daytime_containers'] or not config['rank_create_command'] or not config['pause_files'] or not config['drain_commands'] or not config['ready_commands']:
        raise RuntimeError('empty_handoff_configuration')
    if config.get('lease_ttl_seconds', 120) != 120 or config.get('min_available_gib', 4) < 4 or config.get('recovery_available_gib', 64) < 64:
        raise RuntimeError('unsafe_limits')


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    validate(config)
    manager = Manager(config)
    server = http.server.ThreadingHTTPServer((config['listen_ip'], config.get('port', 9210)), handler(manager))
    server.daemon_threads = True
    threads = [threading.Thread(target=target, daemon=True) for target in
               (manager.worker, manager.watchdog, server.serve_forever)]
    for thread in threads:
        thread.start()
    exit_requested = threading.Event()
    def stop(_signal, _frame):
        manager.restore('manager_stop')
        exit_requested.set()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    while not exit_requested.wait(1):
        if not all(thread.is_alive() for thread in threads):
            manager.restore('manager_thread_failed')
            raise RuntimeError('manager_thread_failed')
    # systemd restart re-enters durable recovery even if graceful stop is cut off.
    end = time.monotonic() + 75
    while time.monotonic() < end and manager.status()['phase'] != 'idle':
        time.sleep(0.25)
    manager.stopping.set()
    server.shutdown()


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(json.dumps({'error_type': type(exc).__name__}), flush=True)
        raise SystemExit(1)
