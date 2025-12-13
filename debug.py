#!/usr/bin/env python3
"""
Unified Debug Tool
==================

One tool to investigate tasks, workers, and system health.
Uses system_logs as the primary data source.

Usage:
    debug.py task <task_id>             # Investigate specific task
    debug.py worker <worker_id>         # Investigate specific worker
    debug.py tasks                      # Analyze recent tasks
    debug.py workers                    # List recent workers
    debug.py health                     # System health check
    debug.py orchestrator               # Orchestrator status

Options:
    --json                              # Output as JSON
    --hours N                           # Time window in hours
    --limit N                           # Limit results
    --logs-only                         # Show only logs timeline
    --debug                             # Show debug info on errors

Examples:
    # Investigate why a task failed
    debug.py task 41345358-f3b5-418a-9805-b442aed30e18
    
    # Check worker health
    debug.py worker gpu-20251031_194121-74965f63
    
    # List recent workers with failures
    debug.py workers --hours 6
    
    # Get system health as JSON
    debug.py health --json
    
    # Check if orchestrator is running
    debug.py orchestrator
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from debug.client import DebugClient
from debug.commands import task, worker, tasks, workers, health, orchestrator, config, runpod, storage


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description='Unified debugging tool for investigating tasks, workers, and system health',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run', required=True)
    
    # Task command
    task_parser = subparsers.add_parser('task', help='Investigate specific task')
    task_parser.add_argument('task_id', help='Task ID to investigate')
    task_parser.add_argument('--json', action='store_true', help='Output as JSON')
    task_parser.add_argument('--logs-only', action='store_true', help='Show only logs timeline')
    task_parser.add_argument('--debug', action='store_true', help='Show debug info on errors')
    
    # Worker command
    worker_parser = subparsers.add_parser('worker', help='Investigate specific worker')
    worker_parser.add_argument('worker_id', help='Worker ID to investigate')
    worker_parser.add_argument('--hours', type=int, default=24, help='Hours of history (default: 24)')
    worker_parser.add_argument('--json', action='store_true', help='Output as JSON')
    worker_parser.add_argument('--logs-only', action='store_true', help='Show only logs timeline')
    worker_parser.add_argument('--startup', action='store_true', help='Show startup logs only')
    worker_parser.add_argument('--check-logging', action='store_true', help='Check if worker is logging')
    worker_parser.add_argument('--check-disk', action='store_true', help='Check disk space via SSH')
    worker_parser.add_argument('--debug', action='store_true', help='Show debug info on errors')
    
    # Tasks command
    tasks_parser = subparsers.add_parser('tasks', help='Analyze recent tasks')
    tasks_parser.add_argument('--limit', type=int, default=50, help='Number of tasks (default: 50)')
    tasks_parser.add_argument('--status', help='Filter by status')
    tasks_parser.add_argument('--type', help='Filter by task type')
    tasks_parser.add_argument('--worker', help='Filter by worker ID')
    tasks_parser.add_argument('--hours', type=int, help='Filter by hours')
    tasks_parser.add_argument('--json', action='store_true', help='Output as JSON')
    tasks_parser.add_argument('--debug', action='store_true', help='Show debug info on errors')
    
    # Workers command
    workers_parser = subparsers.add_parser('workers', help='List recent workers')
    workers_parser.add_argument('--hours', type=int, default=2, help='Hours of history (default: 2)')
    workers_parser.add_argument('--detailed', action='store_true', help='Show detailed analysis')
    workers_parser.add_argument('--json', action='store_true', help='Output as JSON')
    workers_parser.add_argument('--debug', action='store_true', help='Show debug info on errors')
    
    # Health command
    health_parser = subparsers.add_parser('health', help='System health check')
    health_parser.add_argument('--json', action='store_true', help='Output as JSON')
    health_parser.add_argument('--debug', action='store_true', help='Show debug info on errors')
    
    # Orchestrator command
    orch_parser = subparsers.add_parser('orchestrator', help='Orchestrator status')
    orch_parser.add_argument('--hours', type=int, default=1, help='Hours of history (default: 1)')
    orch_parser.add_argument('--json', action='store_true', help='Output as JSON')
    orch_parser.add_argument('--debug', action='store_true', help='Show debug info on errors')
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Show system configuration')
    config_parser.add_argument('--explain', action='store_true', help='Show detailed explanations')
    
    # RunPod command
    runpod_parser = subparsers.add_parser('runpod', help='Check RunPod sync status')
    runpod_parser.add_argument('--terminate', action='store_true', help='Terminate orphaned pods')
    runpod_parser.add_argument('--debug', action='store_true', help='Show debug info on errors')
    
    # Storage command
    storage_parser = subparsers.add_parser('storage', help='Check storage volume health')
    storage_parser.add_argument('--expand', type=str, help='Expand specified storage volume')
    storage_parser.add_argument('--debug', action='store_true', help='Show debug info on errors')
    
    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Create debug client
    try:
        client = DebugClient()
    except Exception as e:
        print(f"❌ Failed to initialize debug client: {e}")
        print("\n💡 Make sure your .env file is configured with:")
        print("   - SUPABASE_URL")
        print("   - SUPABASE_SERVICE_ROLE_KEY")
        sys.exit(1)
    
    # Convert args to options dict
    options = {
        'format': 'json' if hasattr(args, 'json') and args.json else 'text',
        'debug': args.debug if hasattr(args, 'debug') else False
    }
    
    # Add command-specific options
    if hasattr(args, 'hours'):
        options['hours'] = args.hours
    if hasattr(args, 'limit'):
        options['limit'] = args.limit
    if hasattr(args, 'status'):
        options['status'] = args.status
    if hasattr(args, 'type'):
        options['type'] = args.type
    if hasattr(args, 'worker'):
        options['worker'] = args.worker
    if hasattr(args, 'detailed'):
        options['detailed'] = args.detailed
    if hasattr(args, 'logs_only'):
        options['logs_only'] = args.logs_only
    if hasattr(args, 'startup'):
        options['startup'] = args.startup
    if hasattr(args, 'check_logging'):
        options['check_logging'] = args.check_logging
    if hasattr(args, 'check_disk'):
        options['check_disk'] = args.check_disk
    if hasattr(args, 'explain'):
        options['explain'] = args.explain
    if hasattr(args, 'terminate'):
        options['terminate'] = args.terminate
    if hasattr(args, 'expand'):
        options['expand'] = args.expand
    
    # Route to appropriate command handler
    try:
        if args.command == 'task':
            task.run(client, args.task_id, options)
        elif args.command == 'worker':
            worker.run(client, args.worker_id, options)
        elif args.command == 'tasks':
            tasks.run(client, options)
        elif args.command == 'workers':
            workers.run(client, options)
        elif args.command == 'health':
            health.run(client, options)
        elif args.command == 'orchestrator':
            orchestrator.run(client, options)
        elif args.command == 'config':
            config.run(client, options)
        elif args.command == 'runpod':
            runpod.run(client, options)
        elif args.command == 'storage':
            storage.run(client, options)
        else:
            parser.print_help()
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Command failed: {e}")
        if options.get('debug'):
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

