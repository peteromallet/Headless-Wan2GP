#!/usr/bin/env python3
"""
Heartbeat Guardian Process

This is a completely separate process that sends worker heartbeats.
It cannot be blocked by GIL, downloads, model loading, or any worker operations.

The guardian:
- Monitors worker process health via PID
- Receives logs from worker via multiprocessing.Queue
- Sends heartbeats with logs every 20 seconds using curl (no Python HTTP libraries)
- Never crashes or blocks, ensuring heartbeats are always sent
"""

import sys
import time
import json
import subprocess
import os
import signal
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

def check_process_alive(pid: int) -> bool:
    """Check if process with given PID is alive."""
    try:
        os.kill(pid, 0)  # Signal 0 doesn't kill, just checks if process exists
        return True
    except (OSError, ProcessLookupError):
        return False

def get_vram_info() -> tuple[int, int]:
    """Get VRAM info using nvidia-smi."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total,memory.used', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            total, used = result.stdout.strip().split(',')
            return int(float(total)), int(float(used))
    except Exception:
        pass
    return 0, 0

def collect_logs_from_queue(log_queue, max_count: int = 100) -> List[Dict[str, Any]]:
    """
    Collect available logs from queue (non-blocking).

    Args:
        log_queue: Multiprocessing queue containing log entries
        max_count: Maximum number of logs to collect

    Returns:
        List of log entries
    """
    logs = []
    while len(logs) < max_count:
        try:
            log_entry = log_queue.get_nowait()
            logs.append(log_entry)
        except:
            break  # Queue empty
    return logs

def send_heartbeat_with_logs(
    worker_id: str,
    vram_total: int,
    vram_used: int,
    logs: List[Dict[str, Any]],
    config: Dict[str, str]
) -> bool:
    """
    Send heartbeat with logs using curl subprocess.

    Uses curl instead of Python HTTP libraries to ensure complete isolation
    from worker process issues.
    """
    try:
        payload = json.dumps({
            'worker_id_param': worker_id,
            'vram_total_mb_param': vram_total,
            'vram_used_mb_param': vram_used,
            'logs_param': logs,
            'current_task_id_param': None  # Could be enhanced to track current task
        })

        result = subprocess.run([
            'curl', '-s', '-X', 'POST',
            '-m', '10',  # 10 second timeout
            f'{config["db_url"]}/rest/v1/rpc/func_worker_heartbeat_with_logs',
            '-H', f'apikey: {config["api_key"]}',
            '-H', 'Content-Type: application/json',
            '-H', 'Prefer: return=representation',
            '-d', payload
        ], capture_output=True, timeout=15)

        return result.returncode == 0

    except Exception as e:
        # Log error but never crash
        with open(f'/tmp/guardian_{worker_id}_error.log', 'a') as f:
            f.write(f"{datetime.now(timezone.utc).isoformat()}: Error sending with logs: {e}\n")
        return False

def send_heartbeat_simple(
    worker_id: str,
    status: str,
    config: Dict[str, str]
) -> bool:
    """
    Send simple heartbeat without logs using curl.
    Falls back to this when no logs are available.
    """
    try:
        timestamp = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        payload = json.dumps({
            "last_heartbeat": timestamp,
            "status": status
        })

        result = subprocess.run([
            'curl', '-s', '-X', 'PATCH',
            '-m', '10',
            f'{config["db_url"]}/rest/v1/workers?id=eq.{worker_id}',
            '-H', f'apikey: {config["api_key"]}',
            '-H', 'Content-Type: application/json',
            '-d', payload
        ], capture_output=True, timeout=15)

        return result.returncode == 0

    except Exception as e:
        with open(f'/tmp/guardian_{worker_id}_error.log', 'a') as f:
            f.write(f"{datetime.now(timezone.utc).isoformat()}: Error sending simple: {e}\n")
        return False

def guardian_main(worker_id: str, worker_pid: int, log_queue, config: Dict[str, str]):
    """
    Main guardian loop.

    This runs in a completely separate process and cannot be blocked by
    the worker process.
    """

    heartbeat_count = 0
    consecutive_failures = 0

    # Log startup
    startup_msg = f"{datetime.now(timezone.utc).isoformat()}: Guardian started for worker {worker_id} (PID {worker_pid})\n"
    with open(f'/tmp/guardian_{worker_id}.log', 'w') as f:
        f.write(startup_msg)

    print(f"[GUARDIAN] Started for worker {worker_id}, monitoring PID {worker_pid}")

    while True:
        try:
            # Check if worker process is alive
            worker_alive = check_process_alive(worker_pid)
            status = "active" if worker_alive else "crashed"

            # If worker crashed, send one final heartbeat and exit
            if not worker_alive:
                print(f"[GUARDIAN] Worker {worker_id} crashed (PID {worker_pid} no longer exists)")
                send_heartbeat_simple(worker_id, "crashed", config)
                break

            # Collect any available logs
            logs = collect_logs_from_queue(log_queue, max_count=100)

            # Get VRAM metrics
            vram_total, vram_used = get_vram_info()

            # Send heartbeat
            success = False
            if logs:
                success = send_heartbeat_with_logs(worker_id, vram_total, vram_used, logs, config)
            else:
                success = send_heartbeat_simple(worker_id, status, config)

            if success:
                heartbeat_count += 1
                consecutive_failures = 0

                # Log status every 5 minutes (15 heartbeats) to avoid spam
                if heartbeat_count % 15 == 1:
                    msg = f"{datetime.now(timezone.utc).isoformat()}: Heartbeat #{heartbeat_count} sent successfully"
                    if logs:
                        msg += f" with {len(logs)} logs"
                    msg += "\n"
                    with open(f'/tmp/guardian_{worker_id}.log', 'a') as f:
                        f.write(msg)
            else:
                consecutive_failures += 1
                if consecutive_failures >= 3:
                    # Log warning but continue trying
                    with open(f'/tmp/guardian_{worker_id}.log', 'a') as f:
                        f.write(f"{datetime.now(timezone.utc).isoformat()}: WARNING - {consecutive_failures} consecutive heartbeat failures\n")

        except Exception as e:
            # Never crash on any error
            with open(f'/tmp/guardian_{worker_id}_error.log', 'a') as f:
                f.write(f"{datetime.now(timezone.utc).isoformat()}: Exception in main loop: {e}\n")

        # Sleep for interval
        time.sleep(20)

    print(f"[GUARDIAN] Exiting for worker {worker_id}")

if __name__ == "__main__":
    # Ignore signals that might terminate us (except SIGKILL which can't be caught)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    if len(sys.argv) < 2:
        print("Usage: heartbeat_guardian.py <config_json_path> [log_queue_descriptor]")
        sys.exit(1)

    # Load configuration
    config_path = sys.argv[1]
    with open(config_path, 'r') as f:
        config_data = json.load(f)

    worker_id = config_data['worker_id']
    worker_pid = config_data['worker_pid']

    # The log_queue will be passed via multiprocessing spawn context
    # For now, we'll receive it differently based on how we launch
    log_queue = None  # Will be set by parent process

    guardian_main(worker_id, worker_pid, log_queue, config_data)
