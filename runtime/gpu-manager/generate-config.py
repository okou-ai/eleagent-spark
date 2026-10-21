#!/usr/bin/env python3
"""Generate one host's explicit manager configuration without touching Docker."""
import argparse
import importlib.util
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--role', choices=('s1', 's2'), required=True)
    parser.add_argument('--bind-address', required=True)
    parser.add_argument('--port', type=int, default=9210)
    parser.add_argument('--leader-address', required=True)
    parser.add_argument('--image-id', required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--model-root', type=Path, required=True)
    parser.add_argument('--repo-dir', type=Path, required=True)
    parser.add_argument('--state-dir', type=Path, required=True)
    parser.add_argument('--fabric-interface', required=True)
    parser.add_argument('--fabric-hca', required=True)
    parser.add_argument('--timezone', default='America/Los_Angeles')
    parser.add_argument('--minicpm-health-url', default='http://127.0.0.1:22400/health')
    # Any other GPU container must be handed over too, or the night cannot start.
    parser.add_argument('--daytime-container', action='append', default=[],
                        help='another GPU container to stop at night and restart afterwards (repeatable)')
    parser.add_argument('--ready-url', action='append', default=[],
                        help='endpoint that must answer 200 before daytime work resumes (repeatable)')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    for key in ('data_dir', 'model_root', 'repo_dir', 'state_dir', 'output'):
        if not getattr(args, key).is_absolute():
            parser.error(key.replace('_', '-') + ' must be absolute')
    rank = '0' if args.role == 's1' else '1'
    runtime = args.repo_dir / 'runtime/gpu-manager'
    atlas = args.repo_dir / 'deploy/atlas'
    config = {
        'listen_ip': args.bind_address,
        'allowed_bind_addresses': [args.bind_address],
        'port': args.port,
        'timezone': args.timezone,
        'acquire_start_minute': 30,
        'acquire_end_minute': 300,
        'hard_stop_minute': 330,
        'secret_file': str(args.state_dir / 'control.secret'),
        'state_file': str(args.state_dir / 'state.json'),
        'host_role': args.role,
        'data_dir': str(args.data_dir),
        'minicpm_health_url': args.minicpm_health_url,
        # The night belongs to Sleep: Spark 2 keeps only the archive (Sleep reads it, uploads land in it) and Graph
        # Sync (it hands Sleep the last fragments). Each stopped service resumes from its own cursor. Containers stop
        # and restart in this order, so naming and the day agent stop before the VLM they call.
        'daytime_containers': (['eleagent-minicpm'] if args.role == 's1' else
                               ['eleagent-recognition', 'eleagent-speech', 'eleagent-text', 'eleagent-naming',
                                'eleagent-agent', 'eleagent-vlm']) + args.daytime_container,
        'pause_workers': {'minicpm': 'eleagent-observer'} if args.role == 's1' else
                         {'entity': 'eleagent-recognition', 'speech': 'eleagent-speech'},
        'rank_name': 'eleagent-glm-rank' + rank,
        'rank_image': args.image_id,
        'rank_label': {'key': 'org.eleagent.gpu', 'value': 'sleep-glm'},
        'rank_create_command': [
            'bash', str(atlas / 'create-rank.sh'), '--rank', rank,
            '--leader-address', args.leader_address, '--model-root', str(args.model_root),
            '--image', args.image_id, '--fabric-interface', args.fabric_interface,
            '--fabric-hca', args.fabric_hca, '--profile', str(atlas / 'profile-4x1m.json'),
            '--kv-shard', '--cuda-cache', str(args.state_dir / 'cuda-cache'),
            '--name', 'eleagent-glm-rank' + rank,
        ],
        'pause_files': [str(args.data_dir / 'night-gpu.pause')],
        'drain_commands': [['python3', str(runtime / 'drain.py'), '--config', str(args.output)]],
        'ready_commands': [['python3', str(runtime / 'ready.py'), '--config', str(args.output)]],
        # Spark 2's VLM takes minutes to load after a restart; daytime work resumes once it answers.
        'ready_urls': ([] if args.role == 's1' else ['http://127.0.0.1:22410/health']) + args.ready_url,
        'drain_timeout_seconds': 600,
        'lease_ttl_seconds': 120,
        'min_available_gib': 4,
        'recovery_available_gib': 64,
    }
    module_spec = importlib.util.spec_from_file_location('gpu_manager', Path(__file__).with_name('gpu-manager.py'))
    manager_module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(manager_module)
    manager_module.validate(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Never silently replace an active manager's configuration.
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'w') as output:
        json.dump(config, output, indent=2)
        output.write('\n')
    print('Manager configuration generated; create the shared mode-0600 token separately.')


if __name__ == '__main__':
    main()
