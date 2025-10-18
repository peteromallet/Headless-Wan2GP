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
        # Show what we're about to send
        print(f"[GUARDIAN PAYLOAD] Sending {len(logs)} logs to database", flush=True)
        if logs:
            # Show first log in full detail to verify format
            first_log = logs[0]
            print(f"[GUARDIAN PAYLOAD] First log FULL FORMAT:", flush=True)
            print(f"  timestamp: {first_log.get('timestamp')}", flush=True)
            print(f"  level: {first_log.get('level')}", flush=True)
            print(f"  message: {first_log.get('message', '')[:100]}", flush=True)
            print(f"  task_id: {first_log.get('task_id')}", flush=True)
            print(f"  metadata: {first_log.get('metadata')}", flush=True)

            if len(logs) > 1:
                print(f"[GUARDIAN PAYLOAD] Last log: {logs[-1].get('level')} - {logs[-1].get('message', '')[:50]}", flush=True)

        payload = json.dumps({
            'worker_id_param': worker_id,
            'vram_total_mb_param': vram_total,
            'vram_used_mb_param': vram_used,
            'logs_param': logs,
            'current_task_id_param': None  # Could be enhanced to track current task
        })

        # Show payload size and structure
        print(f"[GUARDIAN PAYLOAD] Total payload size: {len(payload)} bytes", flush=True)
        print(f"[GUARDIAN PAYLOAD] Payload preview (first 300 chars): {payload[:300]}...", flush=True)

        result = subprocess.run([
            'curl', '-s', '-X', 'POST',
            '-m', '10',  # 10 second timeout
            f'{config["db_url"]}/rest/v1/rpc/func_worker_heartbeat_with_logs',
            '-H', f'apikey: {config["api_key"]}',
            '-H', 'Content-Type: application/json',
            '-H', 'Prefer: return=representation',
            '-d', payload
        ], capture_output=True, timeout=15)

        # Log the response for debugging
        print(f"[GUARDIAN CURL] Return code: {result.returncode}", flush=True)

        response_text = result.stdout.decode()
        print(f"[GUARDIAN CURL] Response stdout: {response_text[:500]}", flush=True)
        if result.stderr:
            print(f"[GUARDIAN CURL] Response stderr: {result.stderr.decode()[:200]}", flush=True)

        # Parse and validate the response
        if result.returncode != 0:
            print(f"[GUARDIAN CURL] ❌ curl failed with exit code {result.returncode}", flush=True)
            return False

        try:
            response_data = json.loads(response_text)
            success = response_data.get('success', False)
            logs_inserted = response_data.get('logs_inserted', 0)

            if success:
                print(f"[GUARDIAN CURL] ✅ Database confirmed: {logs_inserted} logs inserted", flush=True)
            else:
                print(f"[GUARDIAN CURL] ⚠️ Database returned success=false: {response_data}", flush=True)

            return success
        except json.JSONDecodeError as e:
            print(f"[GUARDIAN CURL] ⚠️ Could not parse JSON response: {e}", flush=True)
            print(f"[GUARDIAN CURL] ⚠️ Assuming success based on curl exit code", flush=True)
            return True  # Assume success if we can't parse (backwards compatible)

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

        print(f"[GUARDIAN SIMPLE] Sending simple heartbeat (no logs) for worker {worker_id}, status={status}", flush=True)

        result = subprocess.run([
            'curl', '-s', '-X', 'PATCH',
            '-m', '10',
            f'{config["db_url"]}/rest/v1/workers?id=eq.{worker_id}',
            '-H', f'apikey: {config["api_key"]}',
            '-H', 'Content-Type: application/json',
            '-d', payload
        ], capture_output=True, timeout=15)

        response_text = result.stdout.decode()
        print(f"[GUARDIAN SIMPLE] Return code: {result.returncode}", flush=True)
        print(f"[GUARDIAN SIMPLE] Response: {response_text[:500]}", flush=True)
        if result.stderr:
            print(f"[GUARDIAN SIMPLE] Error: {result.stderr.decode()[:200]}", flush=True)

        # For simple heartbeat, Supabase returns the updated row or empty array
        # Success is indicated by returncode 0 and non-error response
        if result.returncode != 0:
            print(f"[GUARDIAN SIMPLE] ❌ curl failed with exit code {result.returncode}", flush=True)
            return False

        # Check if response looks like an error
        if 'error' in response_text.lower() or 'message' in response_text.lower():
            try:
                response_data = json.loads(response_text)
                if 'error' in response_data:
                    print(f"[GUARDIAN SIMPLE] ❌ Database error: {response_data.get('error')}", flush=True)
                    return False
            except:
                pass

        print(f"[GUARDIAN SIMPLE] ✅ Simple heartbeat sent successfully", flush=True)
        return True

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

    import traceback

    try:
        print(f"[GUARDIAN DEBUG] Step 1: guardian_main called with worker_id={worker_id}, worker_pid={worker_pid}", flush=True)
        print(f"[GUARDIAN DEBUG] Step 2: log_queue type={type(log_queue)}, config keys={list(config.keys())}", flush=True)
        print(f"[GUARDIAN DEBUG] Step 2.5: API key (first 50 chars): {config.get('api_key', 'N/A')[:50]}...", flush=True)

        # Check if this is a service role key by looking for "service_role" in the JWT payload
        api_key = config.get('api_key', '')
        if 'eyJ' in api_key:  # Looks like a JWT
            try:
                import base64
                # JWT format: header.payload.signature
                payload_part = api_key.split('.')[1]
                # Add padding if needed
                payload_part += '=' * (4 - len(payload_part) % 4)
                decoded = base64.b64decode(payload_part).decode('utf-8')
                if 'service_role' in decoded:
                    print(f"[GUARDIAN DEBUG] ✅ Confirmed using SERVICE ROLE KEY (bypasses RLS)", flush=True)
                else:
                    print(f"[GUARDIAN DEBUG] ⚠️ WARNING: NOT using service_role key! Decoded: {decoded[:100]}", flush=True)
            except Exception as e:
                print(f"[GUARDIAN DEBUG] Could not decode JWT: {e}", flush=True)

        heartbeat_count = 0
        consecutive_failures = 0

        print(f"[GUARDIAN DEBUG] Step 3: Opening log file /tmp/guardian_{worker_id}.log", flush=True)

        # Log startup
        startup_msg = f"{datetime.now(timezone.utc).isoformat()}: Guardian started for worker {worker_id} (PID {worker_pid})\n"
        startup_msg += f"Queue type: {type(log_queue)}\n"
        startup_msg += f"Config: {config}\n"

        with open(f'/tmp/guardian_{worker_id}.log', 'w') as f:
            f.write(startup_msg)
            f.flush()

        print(f"[GUARDIAN DEBUG] Step 4: Log file written successfully", flush=True)
        print(f"[GUARDIAN] Started for worker {worker_id}, monitoring PID {worker_pid}", flush=True)
        print(f"[GUARDIAN DEBUG] Step 5: Entering main loop", flush=True)

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
            # Check if worker process is alive
            worker_alive = check_process_alive(worker_pid)
            status = "active" if worker_alive else "crashed"

            # If worker crashed, send one final heartbeat and exit
            if not worker_alive:
                print(f"[GUARDIAN] Worker {worker_id} crashed (PID {worker_pid} no longer exists)")
                send_heartbeat_simple(worker_id, "crashed", config)
                break

            # Collect any available logs
            print(f"[GUARDIAN DEBUG] Collecting logs from queue...", flush=True)
            logs = collect_logs_from_queue(log_queue, max_count=100)
            print(f"[GUARDIAN DEBUG] Collected {len(logs)} logs from queue", flush=True)

            # Get VRAM metrics
            vram_total, vram_used = get_vram_info()

            # Send heartbeat
            print(f"[GUARDIAN DEBUG] Sending heartbeat (with {len(logs)} logs)...", flush=True)
            success = False
            if logs:
                success = send_heartbeat_with_logs(worker_id, vram_total, vram_used, logs, config)
            else:
                success = send_heartbeat_simple(worker_id, status, config)

            if success:
                heartbeat_count += 1
                consecutive_failures = 0
                print(f"[GUARDIAN DEBUG] Heartbeat #{heartbeat_count} sent successfully", flush=True)

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
                print(f"[GUARDIAN DEBUG] Heartbeat failed (consecutive failures: {consecutive_failures})", flush=True)
                if consecutive_failures >= 3:
                    # Log warning but continue trying
                    with open(f'/tmp/guardian_{worker_id}.log', 'a') as f:
                        f.write(f"{datetime.now(timezone.utc).isoformat()}: WARNING - {consecutive_failures} consecutive heartbeat failures\n")

        except Exception as e:
            # Never crash on any error
            import traceback
            error_msg = f"{datetime.now(timezone.utc).isoformat()}: Exception in main loop: {e}\n{traceback.format_exc()}\n"
            print(f"[GUARDIAN ERROR] {error_msg}", flush=True)
            with open(f'/tmp/guardian_{worker_id}_error.log', 'a') as f:
                f.write(error_msg)

        # Sleep for interval
        print(f"[GUARDIAN DEBUG] Sleeping for 20 seconds...", flush=True)
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
