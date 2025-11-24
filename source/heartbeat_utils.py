import os
import threading
from multiprocessing import Process, Queue
from source.logging_utils import headless_logger
from heartbeat_guardian import guardian_main

def start_heartbeat_guardian_process(worker_id: str, supabase_url: str, supabase_key: str):
    """
    Start bulletproof heartbeat guardian as a separate process.
    """
    log_queue = Queue(maxsize=1000)

    config = {
        'worker_id': worker_id,
        'worker_pid': os.getpid(),
        'db_url': supabase_url,
        'api_key': supabase_key
    }

    guardian = Process(
        target=guardian_main,
        args=(worker_id, os.getpid(), log_queue, config),
        name=f'guardian-{worker_id}',
        daemon=True
    )
    guardian.start()

    headless_logger.essential(f"✅ Heartbeat guardian started: PID {guardian.pid} monitoring worker PID {os.getpid()}")
    
    return guardian, log_queue

def get_gpu_memory_usage():
    """
    Get GPU memory usage in MB.
    """
    try:
        import torch
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
            allocated = torch.cuda.memory_allocated(0) / (1024 * 1024)
            return int(total), int(allocated)
    except Exception:
        pass
    
    return None, None


