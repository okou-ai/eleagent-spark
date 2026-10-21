#!/usr/bin/env python3
"""Check model readiness for current daytime processes before unpausing."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import urllib.request

from drain import minicpm_idle, running_containers


def answers(url):
    # Bypass ambient HTTP proxies for a host-configured internal health endpoint.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=5) as response:
        return response.status == 200


def ready(config):
    # Paused speech must not resume into a service that is still loading its model.
    if not all(answers(url) for url in config.get('ready_urls', [])):
        return False
    if config['host_role'] == 's1':
        running_containers(config['daytime_containers'])
        return minicpm_idle(config['minicpm_health_url'])
    workers = config['pause_workers']
    if not workers:
        return False
    containers = running_containers(list(workers.values()))
    for worker, name in workers.items():
        row = containers[name]
        # The same Docker ID can restart; previous-process markers are insufficient.
        marker = Path(config['data_dir']) / ('night-gpu.' + worker + '.ready')
        event = json.loads(marker.read_text())
        started = datetime.fromisoformat(row['State']['StartedAt'].replace('Z', '+00:00')).timestamp()
        if (set(event) != {'worker', 'pid'} or event['worker'] != worker
                or type(event['pid']) is not int or event['pid'] < 1
                or marker.stat().st_mtime < started):
            return False
    # An intervening restart invalidates readiness from the previous process.
    after = running_containers(list(workers.values()))
    return all(after[name]['Id'] == row['Id']
               and after[name]['State']['StartedAt'] == row['State']['StartedAt']
               for name, row in containers.items())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    try:
        config = json.loads(Path(args.config).read_text())
        loaded = config['host_role'] in {'s1', 's2'} and ready(config)
    except Exception:
        loaded = False
    raise SystemExit(0 if loaded else 1)


if __name__ == '__main__':
    main()
