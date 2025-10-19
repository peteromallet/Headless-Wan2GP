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

def check_process_alive(pid: int, start_time: float = None) -> bool:
    """
    Check if process with given PID is alive and is the same process.

    Args:
        pid: Process ID to check
        start_time: Expected process start time (from psutil) for verification

    Returns:
        True if process exists and matches expected start time (if provided)
    """
    try:
        # First check if PID exists at all
        os.kill(pid, 0)  # Signal 0 doesn't kill, just checks if process exists

        # If start_time provided, verify it's the SAME process (not a reused PID)
        if start_time is not None:
            try:
                import psutil
                proc = psutil.Process(pid)
                current_start_time = proc.create_time()

                # Allow small tolerance for floating point comparison
                if abs(current_start_time - start_time) > 0.1:
                    # PID was reused by a different process
                    return False
            except (psutil.NoSuchProcess, psutil.AccessDenied, ImportError):
                # If we can't verify, assume it's dead (fail-safe)
                return False

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
    config: Dict[str, str],
    status: str = "active"
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
            'status_param': status
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

        response_text = result.stdout.decode()

        # Parse and validate the response
        if result.returncode != 0:
            print(f"[GUARDIAN] ❌ Heartbeat curl failed with exit code {result.returncode}", flush=True)
            return False

        try:
            response_data = json.loads(response_text)
            success = response_data.get('success', False)

            if not success:
                print(f"[GUARDIAN] ⚠️ Database returned error: {response_data}", flush=True)

            return success
        except json.JSONDecodeError as e:
            print(f"[GUARDIAN] ⚠️ Could not parse response: {e}", flush=True)
            return True  # Assume success if we can't parse (backwards compatible)

    except Exception as e:
        # Log error but never crash
        with open(f'/tmp/guardian_{worker_id}_error.log', 'a') as f:
            f.write(f"{datetime.now(timezone.utc).isoformat()}: Error sending with logs: {e}\n")
        return False

def guardian_main(worker_id: str, worker_pid: int, log_queue, config: Dict[str, str]):
    """
    Main guardian loop.

    This runs in a completely separate process and cannot be blocked by
    the worker process.
    """

    import traceback

    worker_start_time = None

    try:
        heartbeat_count = 0

        # Capture the worker process start time for PID reuse detection
        try:
            import psutil
            proc = psutil.Process(worker_pid)
            worker_start_time = proc.create_time()
            print(f"[GUARDIAN] Captured worker start time: {worker_start_time}", flush=True)
        except Exception as e:
            print(f"[GUARDIAN] ⚠️ Could not capture worker start time: {e}", flush=True)
            print(f"[GUARDIAN] Will monitor PID {worker_pid} without start time verification", flush=True)

        # Log startup
        startup_msg = f"{datetime.now(timezone.utc).isoformat()}: Guardian started for worker {worker_id} (PID {worker_pid})\n"

        with open(f'/tmp/guardian_{worker_id}.log', 'w') as f:
            f.write(startup_msg)
            f.flush()

        print(f"[GUARDIAN] Started for worker {worker_id}, monitoring PID {worker_pid}", flush=True)

    except Exception as e:
        error_msg = f"[GUARDIAN CRASH] Exception during startup: {e}\n{traceback.format_exc()}"
        print(error_msg, flush=True)
        try:
            with open(f'/tmp/guardian_crash_{worker_id}.log', 'w') as f:
                f.write(error_msg)
        except:
            pass
        raise

    while True:
        try:
            # Check if worker process is alive (with start time verification to detect PID reuse)
            worker_alive = check_process_alive(worker_pid, worker_start_time)

            # If worker crashed, send one final heartbeat and exit
            if not worker_alive:
                print(f"[GUARDIAN] Worker {worker_id} crashed (PID {worker_pid} no longer exists)")
                send_heartbeat_with_logs(worker_id, 0, 0, [], config, status="crashed")
                break

            # Collect any available logs
            logs = collect_logs_from_queue(log_queue, max_count=100)

            # Get VRAM metrics
            vram_total, vram_used = get_vram_info()

            # Always use the same heartbeat function (just pass empty logs if none available)
            success = send_heartbeat_with_logs(worker_id, vram_total, vram_used, logs, config)

            if success:
                heartbeat_count += 1

                # Log status every 5 minutes (15 heartbeats) to avoid spam
                if heartbeat_count % 15 == 1:
                    msg = f"{datetime.now(timezone.utc).isoformat()}: Heartbeat #{heartbeat_count} sent successfully"
                    if logs:
                        msg += f" with {len(logs)} logs"
                    msg += "\n"
                    with open(f'/tmp/guardian_{worker_id}.log', 'a') as f:
                        f.write(msg)

        except Exception as e:
            # Never crash on any error
            import traceback
            error_msg = f"{datetime.now(timezone.utc).isoformat()}: Exception in main loop: {e}\n{traceback.format_exc()}\n"
            print(f"[GUARDIAN ERROR] {error_msg}", flush=True)
            with open(f'/tmp/guardian_{worker_id}_error.log', 'a') as f:
                f.write(error_msg)

        # Sleep for interval
        time.sleep(20)

    print(f"[GUARDIAN] Exiting for worker {worker_id}")

if __name__ == "__main__":
    # Ignore signals that might terminate us (except SIGKILL which can't be caught)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # NOTE: This script is designed to be launched via multiprocessing.Process()
    # from worker.py, which passes arguments directly to guardian_main().
    # If you need to run it standalone for testing, you would need to:
    # 1. Create a multiprocessing.Queue() object
    # 2. Pass it along with worker_id, worker_pid, and config dict

    print("ERROR: heartbeat_guardian.py should not be run directly.")
    print("It is launched automatically by worker.py via multiprocessing.Process()")
    print("with proper queue and config arguments.")
    sys.exit(1)
