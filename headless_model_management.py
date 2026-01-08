#!/usr/bin/env python3
"""
WanGP Headless Task Queue Manager

A persistent service that maintains model state and processes generation tasks
via a queue system. This script keeps models loaded in memory and delegates
actual generation to headless_wgp.py while managing the queue, persistence,
and task scheduling.

Key Features:
- Persistent model state (models stay loaded until switched)
- Task queue with priority support
- Auto model switching and memory management
- Status monitoring and progress tracking
- Uses wgp.py's native queue and state management
- Hot-swappable task processing

Usage:
    # Start the headless service
    python headless_model_management.py --wan-dir /path/to/WanGP --port 8080
    
    # Submit tasks via API or queue files
    curl -X POST http://localhost:8080/generate \
         -H "Content-Type: application/json" \
         -d '{"model": "vace_14B", "prompt": "mystical forest", "video_guide": "input.mp4"}'
"""

import os
import sys
import time
import traceback

# Import debug print function from worker
try:
    from worker import dprint
except ImportError:
    def dprint(msg):
        if os.environ.get('DEBUG'):
            print(msg)
import json
import threading
import queue
import argparse
import signal
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from contextlib import contextmanager
import io
from source.lora_utils import cleanup_legacy_lora_collisions
from source.logging_utils import queue_logger
from source.params import TaskConfig

# Add WanGP to path for imports
def setup_wgp_path(wan_dir: str):
    """Setup WanGP path and imports."""
    wan_dir = os.path.abspath(wan_dir)
    if wan_dir not in sys.path:
        sys.path.insert(0, wan_dir)
    return wan_dir

# Task definitions
@dataclass
class GenerationTask:
    """Represents a single generation task."""
    id: str
    model: str
    prompt: str
    parameters: Dict[str, Any]
    priority: int = 0
    created_at: str = None
    status: str = "pending"  # pending, processing, completed, failed
    result_path: Optional[str] = None
    error_message: Optional[str] = None
    processing_time: Optional[float] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()

@dataclass 
class QueueStatus:
    """Current queue status information."""
    pending_tasks: int
    processing_task: Optional[str]
    completed_tasks: int
    failed_tasks: int
    current_model: Optional[str]
    uptime: float
    memory_usage: Dict[str, Any]


class HeadlessTaskQueue:
    """
    Main task queue manager that integrates with wgp.py's existing queue system.
    
    This class leverages wgp.py's built-in task management and state persistence
    while providing a clean API for headless operation.
    """
    
    def __init__(self, wan_dir: str, max_workers: int = 1, debug_mode: bool = False, main_output_dir: Optional[str] = None):
        """
        Initialize the headless task queue.

        Args:
            wan_dir: Path to WanGP directory
            max_workers: Number of concurrent generation workers (recommend 1 for GPU)
            debug_mode: Enable verbose debug logging (should match worker's --debug flag)
            main_output_dir: Optional path for output directory. If not provided, defaults to
                           'outputs' directory next to wan_dir (preserves backwards compatibility)
        """
        self.wan_dir = setup_wgp_path(wan_dir)
        self.max_workers = max_workers
        self.main_output_dir = main_output_dir
        self.running = False
        self.start_time = time.time()
        self.debug_mode = debug_mode  # Now controlled by caller
        
        # Import wgp after path setup (protect sys.argv to prevent argument conflicts)
        _saved_argv = sys.argv[:]
        sys.argv = ["headless_wgp.py"]
        # Headless stubs to avoid optional UI deps (tkinter/matanyone) during import
        try:
            import types
            # Stub tkinter if not available
            if 'tkinter' not in sys.modules:
                sys.modules['tkinter'] = types.ModuleType('tkinter')
            # Stub preprocessing.matanyone.app with minimal interface
            dummy_pkg = types.ModuleType('preprocessing')
            dummy_matanyone = types.ModuleType('preprocessing.matanyone')
            dummy_app = types.ModuleType('preprocessing.matanyone.app')
            def _noop_handler():
                class _Dummy:
                    def __getattr__(self, _):
                        return None
                return _Dummy()
            dummy_app.get_vmc_event_handler = _noop_handler  # type: ignore
            sys.modules['preprocessing'] = dummy_pkg
            sys.modules['preprocessing.matanyone'] = dummy_matanyone
            sys.modules['preprocessing.matanyone.app'] = dummy_app
        except Exception:
            pass
        # Don't import wgp during initialization to avoid CUDA/argparse conflicts
        # wgp will be imported lazily when needed (e.g., in _apply_sampler_cfg_preset)
        # This allows the queue to initialize even if CUDA isn't ready yet
        self.wgp = None
        
        # Restore sys.argv immediately (no wgp import, so no need for protection)
        try:
            sys.argv = _saved_argv
        except Exception:
            pass
        
        # Defer orchestrator initialization to avoid CUDA init during queue setup
        # Orchestrator imports wgp, which triggers deep imports that call torch.cuda
        # We'll initialize it lazily when first needed
        self.orchestrator = None
        self._orchestrator_init_attempted = False
        logging.getLogger('HeadlessQueue').info(f"HeadlessTaskQueue created (orchestrator will initialize on first use)")
        
        # Task management
        self.task_queue = queue.PriorityQueue()
        self.task_history: Dict[str, GenerationTask] = {}
        self.current_task: Optional[GenerationTask] = None
        self.current_model: Optional[str] = None
        
        # Threading
        self.worker_threads: List[threading.Thread] = []
        self.queue_lock = threading.RLock()
        self.shutdown_event = threading.Event()
        
        # Statistics
        self.stats = {
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "model_switches": 0,
            "total_generation_time": 0.0
        }
        
        # Setup logging
        self._setup_logging()
        
        # Initialize wgp state (reuse existing state management)
        self._init_wgp_integration()
        
        self.logger.info(f"HeadlessTaskQueue initialized with WanGP at {wan_dir}")
    
    def _setup_logging(self):
        """Setup structured logging that goes to Supabase via the log interceptor."""
        # Keep Python's basic logging for local file backup
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('headless.log')
            ]
        )
        self._file_logger = logging.getLogger('HeadlessQueue')
        
        # Use queue_logger (ComponentLogger) as main logger - this goes to Supabase
        # ComponentLogger has compatible interface: .info(), .error(), .warning(), .debug()
        self.logger = queue_logger
    
    def _ensure_orchestrator(self):
        """
        Lazily initialize orchestrator on first use to avoid CUDA init during queue setup.
        
        The orchestrator imports wgp, which triggers deep module imports (models/wan/modules/t5.py)
        that call torch.cuda.current_device() at class definition time. We defer this until
        the first task is actually processed, when CUDA is guaranteed to be ready.
        """
        if self.orchestrator is not None:
            return  # Already initialized
        
        if self._orchestrator_init_attempted:
            raise RuntimeError("Orchestrator initialization failed previously")
        
        self._orchestrator_init_attempted = True
        
        try:
            if self.debug_mode:
                self.logger.info("[LAZY_INIT] Initializing WanOrchestrator (first use)...")
                self.logger.info("[LAZY_INIT] Warming up CUDA before importing wgp...")

            # Warm up CUDA before importing wgp (upstream T5EncoderModel has torch.cuda.current_device()
            # as a default arg, which is evaluated at module import time)
            import torch

            # Detailed CUDA diagnostics
            if self.debug_mode:
                self.logger.info("[CUDA_DEBUG] ========== CUDA DIAGNOSTICS ==========")
                self.logger.info(f"[CUDA_DEBUG] torch.cuda.is_available(): {torch.cuda.is_available()}")

            if torch.cuda.is_available():
                if self.debug_mode:
                    try:
                        device_count = torch.cuda.device_count()
                        self.logger.info(f"[CUDA_DEBUG] Device count: {device_count}")

                        for i in range(device_count):
                            self.logger.info(f"[CUDA_DEBUG] Device {i}: {torch.cuda.get_device_name(i)}")
                            self.logger.info(f"[CUDA_DEBUG]   - Properties: {torch.cuda.get_device_properties(i)}")

                        # Try to get CUDA version info
                        try:
                            self.logger.info(f"[CUDA_DEBUG] CUDA version (torch): {torch.version.cuda}")
                        except:
                            pass

                        # Try to initialize current device
                        try:
                            current_dev = torch.cuda.current_device()
                            self.logger.info(f"[CUDA_DEBUG] Current device: {current_dev}")

                            # Try a simple tensor operation
                            test_tensor = torch.tensor([1.0], device='cuda')
                            self.logger.info(f"[CUDA_DEBUG] ✅ Successfully created tensor on CUDA: {test_tensor.device}")

                        except Exception as e:
                            self.logger.error(f"[CUDA_DEBUG] ❌ Failed to initialize current device: {e}")
                            raise

                    except Exception as e:
                        self.logger.error(f"[CUDA_DEBUG] ❌ Error during CUDA diagnostics: {e}\n{traceback.format_exc()}")
                        raise

            else:
                if self.debug_mode:
                    self.logger.warning("[CUDA_DEBUG] ⚠️  torch.cuda.is_available() returned False")
                    self.logger.warning("[CUDA_DEBUG] Checking why CUDA is not available...")

                    # Check if CUDA was built with torch
                    self.logger.info(f"[CUDA_DEBUG] torch.version.cuda: {torch.version.cuda}")
                    self.logger.info(f"[CUDA_DEBUG] torch.backends.cudnn.version(): {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}")

                    # Try to import pynvml for driver info
                    try:
                        import pynvml
                        pynvml.nvmlInit()
                        driver_version = pynvml.nvmlSystemGetDriverVersion()
                        self.logger.info(f"[CUDA_DEBUG] NVIDIA driver version: {driver_version}")
                        device_count = pynvml.nvmlDeviceGetCount()
                        self.logger.info(f"[CUDA_DEBUG] NVML device count: {device_count}")
                        for i in range(device_count):
                            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                            name = pynvml.nvmlDeviceGetName(handle)
                            self.logger.info(f"[CUDA_DEBUG] NVML Device {i}: {name}")
                    except Exception as e:
                        self.logger.warning(f"[CUDA_DEBUG] Could not get NVML info: {e}")

            if self.debug_mode:
                self.logger.info("[CUDA_DEBUG] ===========================================")

            if self.debug_mode:
                self.logger.info("[LAZY_INIT] Importing WanOrchestrator (this imports wgp and model modules)...")
            # Protect sys.argv and working directory before importing headless_wgp which imports wgp
            # wgp.py will try to parse sys.argv and will fail if it contains Supabase/database arguments
            # wgp.py also uses relative paths for model loading and needs to run from Wan2GP directory
            # CRITICAL: wgp.py loads models at MODULE-LEVEL (line 2260), so we MUST chdir BEFORE import
            _saved_argv_for_import = sys.argv[:]
            _saved_cwd = os.getcwd()
            try:
                sys.argv = ["headless_model_management.py"]  # Clean argv for wgp import

                # CRITICAL: Change to Wan2GP directory BEFORE importing/initializing WanOrchestrator
                # wgp.py uses relative paths (defaults/*.json) and expects to run from Wan2GP/
                if self.debug_mode:
                    self.logger.info(f"[LAZY_INIT] Changing to Wan2GP directory: {self.wan_dir}")
                    self.logger.info(f"[LAZY_INIT] Current directory before chdir: {os.getcwd()}")

                os.chdir(self.wan_dir)

                actual_cwd = os.getcwd()
                if self.debug_mode:
                    self.logger.info(f"[LAZY_INIT] Changed directory to: {actual_cwd}")

                # Verify the change worked
                if actual_cwd != self.wan_dir:
                    raise RuntimeError(
                        f"Directory change failed! Expected {self.wan_dir}, got {actual_cwd}"
                    )

                # Verify critical structure exists
                if not os.path.isdir("defaults"):
                    raise RuntimeError(
                        f"defaults/ directory not found in {actual_cwd}. "
                        f"Cannot proceed without model definitions!"
                    )

                if self.debug_mode:
                    self.logger.info(f"[LAZY_INIT] ✅ Now in Wan2GP directory, importing WanOrchestrator...")

                from headless_wgp import WanOrchestrator
                
                # Set mmgp verbose level for debug logging in any2video.py SVI path etc
                if self.debug_mode:
                    try:
                        from mmgp import offload
                        offload.default_verboseLevel = 2
                        self.logger.info("[LAZY_INIT] Set offload.default_verboseLevel=2 for debug logging")
                    except ImportError:
                        pass
                
                self.orchestrator = WanOrchestrator(self.wan_dir, main_output_dir=self.main_output_dir)
            finally:
                sys.argv = _saved_argv_for_import  # Restore original arguments
                # NOTE: We do NOT restore the working directory - WGP expects to stay in Wan2GP/
                # This ensures model downloads, file operations, etc. use correct paths

            if self.debug_mode:
                self.logger.info("[LAZY_INIT] ✅ WanOrchestrator initialized successfully")

            # Now that orchestrator exists, complete wgp integration
            self._init_wgp_integration()

        except Exception as e:
            # Always log orchestrator init failures - this is critical for debugging!
            self.logger.error(f"[LAZY_INIT] ❌ Failed to initialize WanOrchestrator: {e}")
            if self.debug_mode:
                self.logger.error(f"[LAZY_INIT] Traceback:\n{traceback.format_exc()}")
            raise
    
    def _init_wgp_integration(self):
        """
        Initialize integration with wgp.py's existing systems.
        
        This reuses wgp.py's state management, queue handling, and model persistence
        rather than reimplementing it.
        
        Called after orchestrator is lazily initialized.
        """
        if self.orchestrator is None:
            self.logger.warning("Skipping wgp integration - orchestrator not initialized yet")
            return
        
        # TODO: Integrate with wgp.py's existing queue system
        # Key integration points:
        
        # 1. Reuse wgp.py's state management
        # self.wgp_state = self.wgp.get_default_state()  # Use wgp's state object
        self.wgp_state = self.orchestrator.state  # Leverage orchestrator's state
        
        # 2. Hook into wgp.py's model loading/unloading
        # self.wgp.preload_model_policy = "S"  # Smart preloading
        
        # 3. Leverage wgp.py's queue persistence
        # if hasattr(self.wgp, 'load_queue_action'):
        #     self.wgp.load_queue_action("headless_queue.json", self.wgp_state)
        
        # 4. Use wgp.py's callback system for progress tracking
        # self.progress_callback = self._create_wgp_progress_callback()
        
        self.logger.info("WGP integration initialized")

    def _cleanup_memory_after_task(self, task_id: str):
        """
        Clean up memory after task completion WITHOUT unloading models.

        This clears PyTorch caches and Python garbage to prevent memory fragmentation
        that can slow down subsequent generations. Models remain loaded in VRAM.

        Args:
            task_id: ID of the completed task (for logging)
        """
        import torch
        import gc

        try:
            # Log memory BEFORE cleanup
            if torch.cuda.is_available():
                vram_allocated_before = torch.cuda.memory_allocated() / 1024**3
                vram_reserved_before = torch.cuda.memory_reserved() / 1024**3
                self.logger.info(
                    f"[MEMORY_CLEANUP] Task {task_id}: "
                    f"BEFORE - VRAM allocated: {vram_allocated_before:.2f}GB, "
                    f"reserved: {vram_reserved_before:.2f}GB"
                )

            # Clear PyTorch's CUDA cache (frees unused reserved memory, keeps models)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.logger.info(f"[MEMORY_CLEANUP] Task {task_id}: Cleared CUDA cache")

            # Run Python garbage collection to free CPU memory
            collected = gc.collect()
            self.logger.info(f"[MEMORY_CLEANUP] Task {task_id}: Garbage collected {collected} objects")

            # Log memory AFTER cleanup
            if torch.cuda.is_available():
                vram_allocated_after = torch.cuda.memory_allocated() / 1024**3
                vram_reserved_after = torch.cuda.memory_reserved() / 1024**3
                vram_freed = vram_reserved_before - vram_reserved_after

                self.logger.info(
                    f"[MEMORY_CLEANUP] Task {task_id}: "
                    f"AFTER - VRAM allocated: {vram_allocated_after:.2f}GB, "
                    f"reserved: {vram_reserved_after:.2f}GB"
                )

                if vram_freed > 0.01:  # Only log if freed >10MB
                    self.logger.info(
                        f"[MEMORY_CLEANUP] Task {task_id}: ✅ Freed {vram_freed:.2f}GB of reserved VRAM"
                    )
                else:
                    self.logger.info(
                        f"[MEMORY_CLEANUP] Task {task_id}: No significant VRAM freed (models still loaded)"
                    )

        except Exception as e:
            self.logger.warning(f"[MEMORY_CLEANUP] Task {task_id}: Failed to cleanup memory: {e}")
    
    def start(self, preload_model: Optional[str] = None):
        """
        Start the task queue processing service.

        Args:
            preload_model: Optional model to pre-load before processing tasks.
                          If specified, the model will be loaded immediately after
                          workers start, making the first task much faster.
                          Example: "wan_2_2_vace_lightning_baseline_2_2_2"
        """
        if self.running:
            self.logger.warning("Queue already running")
            return

        self.running = True
        self.shutdown_event.clear()

        # Start worker threads
        for i in range(self.max_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"GenerationWorker-{i}",
                daemon=True
            )
            worker.start()
            self.worker_threads.append(worker)

        # Start monitoring thread
        monitor = threading.Thread(
            target=self._monitor_loop,
            name="QueueMonitor",
            daemon=True
        )
        monitor.start()
        self.worker_threads.append(monitor)

        self.logger.info(f"Task queue started with {self.max_workers} workers")

        # Pre-load model if specified
        if preload_model:
            self.logger.info(f"Pre-loading model: {preload_model}")
            try:
                # Initialize orchestrator and load model in background
                self._ensure_orchestrator()
                self.orchestrator.load_model(preload_model)
                self.current_model = preload_model
                self.logger.info(f"✅ Model {preload_model} pre-loaded successfully")
            except Exception as e:
                # Log the full error with traceback
                self.logger.error(f"❌ FATAL: Failed to pre-load model {preload_model}: {e}\n{traceback.format_exc()}")
                
                # If orchestrator failed to initialize, this is fatal - worker cannot function
                if self.orchestrator is None:
                    self.logger.error("Orchestrator failed to initialize - worker cannot process tasks. Exiting.")
                    raise RuntimeError(f"Orchestrator initialization failed during preload: {e}") from e
                else:
                    # Orchestrator is OK but model load failed - this is recoverable
                    self.logger.warning(f"Model {preload_model} failed to load, but orchestrator is ready. Worker will continue.")
    
    def stop(self, timeout: float = 30.0):
        """Stop the task queue processing service."""
        if not self.running:
            return
        
        self.logger.info("Shutting down task queue...")
        self.running = False
        self.shutdown_event.set()
        
        # Wait for workers to finish
        for worker in self.worker_threads:
            worker.join(timeout=timeout)
        
        # Save queue state (integrate with wgp.py's save system)
        self._save_queue_state()
        
        # Optionally unload model to free VRAM
        # self._cleanup_models()
        
        self.logger.info("Task queue shutdown complete")
    
    def submit_task(self, task: GenerationTask) -> str:
        """
        Submit a new generation task to the queue.
        
        Args:
            task: Generation task to process
            
        Returns:
            Task ID for tracking
        """
        with self.queue_lock:
            # Integrate with wgp.py's task creation
            # TODO: Convert our task format to wgp.py's internal task format
            wgp_task = self._convert_to_wgp_task(task)
            
            # Add to queue with priority (higher priority = processed first)
            self.task_queue.put((-task.priority, time.time(), task))
            self.task_history[task.id] = task
            self.stats["tasks_submitted"] += 1
            
            self.logger.info(f"Task submitted: {task.id} (model: {task.model}, priority: {task.priority})")
            return task.id
    
    def get_task_status(self, task_id: str) -> Optional[GenerationTask]:
        """Get status of a specific task."""
        return self.task_history.get(task_id)
    
    def wait_for_completion(self, task_id: str, timeout: float = 300.0) -> Dict[str, Any]:
        """
        Wait for a task to complete and return the result.
        
        Args:
            task_id: ID of the task to wait for
            timeout: Maximum time to wait in seconds
            
        Returns:
            Dictionary with 'success', 'output_path', and optional 'error' keys
        """
        import time
        start_time = time.time()
        
        while (time.time() - start_time) < timeout:
            task_status = self.get_task_status(task_id)
            
            if task_status is None:
                return {
                    "success": False,
                    "error": f"Task {task_id} not found in queue"
                }
            
            if task_status.status == "completed":
                return {
                    "success": True,
                    "output_path": task_status.result_path
                }
            elif task_status.status == "failed":
                return {
                    "success": False,
                    "error": task_status.error_message or "Task failed with unknown error"
                }
            
            # Task is still pending or processing, wait a bit
            time.sleep(1.0)
        
        # Timeout reached
        return {
            "success": False,
            "error": f"Task {task_id} did not complete within {timeout} seconds"
        }
    
    def get_queue_status(self) -> QueueStatus:
        """Get current queue status."""
        with self.queue_lock:
            return QueueStatus(
                pending_tasks=self.task_queue.qsize(),
                processing_task=self.current_task.id if self.current_task else None,
                completed_tasks=self.stats["tasks_completed"],
                failed_tasks=self.stats["tasks_failed"],
                current_model=self.current_model,
                uptime=time.time() - self.start_time,
                memory_usage=self._get_memory_usage()
            )
    
    def _worker_loop(self):
        """Main worker loop for processing tasks."""
        worker_name = threading.current_thread().name
        self.logger.info(f"{worker_name} started")
        
        while self.running and not self.shutdown_event.is_set():
            try:
                # Get next task (blocks with timeout)
                try:
                    priority, timestamp, task = self.task_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Process the task
                self._process_task(task, worker_name)
                
            except Exception as e:
                self.logger.error(f"{worker_name} error: {e}\n{traceback.format_exc()}")
                time.sleep(1.0)
        
        self.logger.info(f"{worker_name} stopped")
    
    def _process_task(self, task: GenerationTask, worker_name: str):
        """
        Process a single generation task.
        
        This is where we delegate to headless_wgp.py while managing
        model persistence and state.
        """
        # Ensure logs emitted during this generation are attributed to this task.
        # This runs inside the GenerationWorker thread, which is where wgp/headless_wgp runs.
        try:
            from source.logging_utils import set_current_task_context  # local import to avoid cycles
            set_current_task_context(task.id)
        except Exception:
            pass

        with self.queue_lock:
            self.current_task = task
            task.status = "processing"
        
        self.logger.info(f"{worker_name} processing task {task.id}")
        start_time = time.time()
        
        try:
            # 1. Ensure correct model is loaded (orchestrator checks WGP's ground truth)
            self._switch_model(task.model, worker_name)
            
            # 2. Delegate actual generation to orchestrator
            # The orchestrator handles the heavy lifting while we manage the queue
            result_path = self._execute_generation(task, worker_name)

            # Verify we're still in Wan2GP directory after generation
            current_dir = os.getcwd()
            if "Wan2GP" not in current_dir:
                self.logger.warning(
                    f"[PATH_CHECK] After generation: Current directory changed!\n"
                    f"  Current: {current_dir}\n"
                    f"  Expected: Should contain 'Wan2GP'\n"
                    f"  This may cause issues for subsequent tasks!"
                )
            else:
                self.logger.debug(f"[PATH_CHECK] After generation: Still in Wan2GP ✓")

            # 3. Validate output and update task status
            processing_time = time.time() - start_time
            is_success = bool(result_path)
            try:
                if is_success:
                    # If a path was returned, check existence where possible
                    rp = Path(result_path)
                    is_success = rp.exists()

                    # Some environments (e.g. networked volumes) can briefly report a freshly-written file as missing.
                    # Do a short retry before failing the task.
                    if not is_success:
                        try:
                            # Note: os is imported at module level - don't re-import here as it causes
                            # "local variable 'os' referenced before assignment" due to Python scoping
                            import time as _time

                            retry_s = 2.0
                            interval_s = 0.2
                            attempts = max(1, int(retry_s / interval_s))

                            for _ in range(attempts):
                                _time.sleep(interval_s)
                                if rp.exists():
                                    is_success = True
                                    break

                            if not is_success:
                                # Tagged diagnostics to debug "phantom output path" failures
                                # Keep output bounded to avoid log spam.
                                tag = "[TravelNoOutputGenerated]"
                                cwd = os.getcwd()
                                parent = rp.parent
                                self.logger.error(f"{tag} Output path missing after generation: {result_path}")
                                self.logger.error(f"{tag} CWD: {cwd}")
                                try:
                                    self.logger.error(f"{tag} Parent exists: {parent} -> {parent.exists()}")
                                except Exception as _e:
                                    self.logger.error(f"{tag} Parent exists check failed: {type(_e).__name__}: {_e}")

                                try:
                                    if parent.exists():
                                        # Show a small sample of directory contents to spot mismatched output dirs.
                                        entries = sorted([p.name for p in parent.iterdir()])[:50]
                                        self.logger.error(f"{tag} Parent dir sample (first {len(entries)}): {entries}")
                                except Exception as _e:
                                    self.logger.error(f"{tag} Parent list failed: {type(_e).__name__}: {_e}")

                                # Common alternative location when running from Wan2GP/ with relative outputs
                                try:
                                    alt_parent = Path(cwd) / "outputs"
                                    if alt_parent != parent and alt_parent.exists():
                                        alt_entries = sorted([p.name for p in alt_parent.iterdir()])[:50]
                                        self.logger.error(f"{tag} Alt outputs dir: {alt_parent} sample (first {len(alt_entries)}): {alt_entries}")
                                except Exception as _e:
                                    self.logger.error(f"{tag} Alt outputs list failed: {type(_e).__name__}: {_e}")
                        except Exception:
                            # Never let diagnostics break the worker loop.
                            pass
            except Exception:
                # If any exception while checking, keep prior truthiness
                pass

            with self.queue_lock:
                task.processing_time = processing_time
                if is_success:
                    task.status = "completed"
                    task.result_path = result_path
                    self.stats["tasks_completed"] += 1
                    self.stats["total_generation_time"] += processing_time
                    self.logger.info(f"Task {task.id} completed in {processing_time:.1f}s: {result_path}")
                else:
                    task.status = "failed"
                    task.error_message = "No output generated"
                    self.stats["tasks_failed"] += 1
                    self.logger.error(f"Task {task.id} failed after {processing_time:.1f}s: No output generated")

            # Memory cleanup after each task (does NOT unload models)
            # This clears PyTorch's internal caches and Python garbage to prevent fragmentation
            self._cleanup_memory_after_task(task.id)

        except Exception as e:
            # Handle task failure
            processing_time = time.time() - start_time
            error_message_str = str(e)
            
            with self.queue_lock:
                task.status = "failed"
                task.error_message = error_message_str
                task.processing_time = processing_time
                self.stats["tasks_failed"] += 1
            
            self.logger.error(f"Task {task.id} failed after {processing_time:.1f}s: {e}")
            
            # Check if this is a fatal error that requires worker termination
            try:
                from source.fatal_error_handler import check_and_handle_fatal_error, FatalWorkerError
                check_and_handle_fatal_error(
                    error_message=error_message_str,
                    exception=e,
                    logger=self.logger,
                    worker_id=os.getenv("WORKER_ID"),
                    task_id=task.id
                )
            except FatalWorkerError:
                # Re-raise fatal errors to propagate to main worker loop
                raise
            except Exception as fatal_check_error:
                # If fatal error checking itself fails, log but don't crash
                self.logger.error(f"Error checking for fatal errors: {fatal_check_error}")
        
        finally:
            with self.queue_lock:
                self.current_task = None
            try:
                from source.logging_utils import set_current_task_context  # local import to avoid cycles
                set_current_task_context(None)
            except Exception:
                pass
    
    def _switch_model(self, model_key: str, worker_name: str) -> bool:
        """
        Ensure the correct model is loaded using wgp.py's model management.
        
        This leverages the orchestrator's model loading while tracking
        the change in our queue system. The orchestrator checks WGP's ground truth
        (wgp.transformer_type) to determine if a switch is actually needed.
        
        Returns:
            bool: True if a model switch actually occurred, False if already loaded
        """
        # Ensure orchestrator is initialized before switching models
        self._ensure_orchestrator()
        
        self.logger.debug(f"{worker_name} ensuring model {model_key} is loaded (current: {self.current_model})")
        switch_start = time.time()
        
        try:
            # Use orchestrator's model loading - it checks WGP's ground truth
            # and returns whether a switch actually occurred
            switched = self.orchestrator.load_model(model_key)
            
            if switched:
                # Only do switch-specific actions if a switch actually occurred
                self.logger.info(f"{worker_name} switched model: {self.current_model} → {model_key}")
                
                self.stats["model_switches"] += 1
                switch_time = time.time() - switch_start
                self.logger.info(f"Model switch completed in {switch_time:.1f}s")
            
            # Always sync our tracking with orchestrator's state
            self.current_model = model_key
            return switched
            
        except Exception as e:
            self.logger.error(f"Model switch failed: {e}")
            raise
    
    def _execute_generation(self, task: GenerationTask, worker_name: str) -> str:
        """
        Execute the actual generation using headless_wgp.py.
        
        This delegates to the orchestrator while providing progress tracking
        and integration with our queue system. Enhanced to support video guides,
        masks, image references, and other advanced features.
        """
        # Ensure orchestrator is initialized before generation
        self._ensure_orchestrator()
        
        self.logger.info(f"{worker_name} executing generation for task {task.id} (model: {task.model})")
        
        # Convert task parameters to WanOrchestrator format
        wgp_params = self._convert_to_wgp_task(task)

        # Remove model and prompt from params since they're passed separately to avoid duplication
        generation_params = {k: v for k, v in wgp_params.items() if k not in ("model", "prompt")}

        # DEBUG: Log all parameter keys to verify _parsed_phase_config is present
        self.logger.info(f"[PHASE_CONFIG_DEBUG] Task {task.id}: generation_params keys: {list(generation_params.keys())}")

        # CRITICAL: Apply phase_config patches NOW in the worker thread where wgp is imported
        # Store patch info for cleanup in finally block
        _patch_applied = False
        _parsed_phase_config_for_restore = None
        _model_name_for_restore = None

        if "_parsed_phase_config" in generation_params and "_phase_config_model_name" in generation_params:
            parsed_phase_config = generation_params.pop("_parsed_phase_config")
            model_name = generation_params.pop("_phase_config_model_name")

            # Save for restoration
            _parsed_phase_config_for_restore = parsed_phase_config
            _model_name_for_restore = model_name

            self.logger.info(f"[PHASE_CONFIG] Applying model patch in GenerationWorker for '{model_name}'")

            from source.phase_config import apply_phase_config_patch
            apply_phase_config_patch(parsed_phase_config, model_name, task.id)
            _patch_applied = True

        # Handle svi2pro: This is a model_def property, not a generate() parameter
        # We need to patch it into BOTH wgp.models_def AND wan_model.model_def
        # because the model captures model_def at load time, not at generation time
        _svi2pro_original = None
        _svi2pro_patched = False
        _wan_model_patched = False
        if generation_params.get("svi2pro"):
            try:
                import wgp
                model_key = task.model
                
                # Patch 1: wgp.models_def (for any new model loads)
                if model_key in wgp.models_def:
                    _svi2pro_original = wgp.models_def[model_key].get("svi2pro")
                    wgp.models_def[model_key]["svi2pro"] = True
                    _svi2pro_patched = True
                    self.logger.info(f"[SVI2PRO] Patched wgp.models_def['{model_key}']['svi2pro'] = True (was: {_svi2pro_original})", task_id=task.id)
                
                # Patch 2: wan_model.model_def DIRECTLY (the actual object used during generation)
                # This is critical because the model was loaded BEFORE we patched models_def
                if hasattr(wgp, 'wan_model') and wgp.wan_model is not None:
                    if hasattr(wgp.wan_model, 'model_def') and wgp.wan_model.model_def is not None:
                        # Diagnostic: check if they're the same object
                        models_def_obj = wgp.models_def.get(model_key)
                        wan_model_def_obj = wgp.wan_model.model_def
                        same_object = models_def_obj is wan_model_def_obj
                        self.logger.info(f"[SVI2PRO_DIAG] models_def['{model_key}'] id={id(models_def_obj)}, wan_model.model_def id={id(wan_model_def_obj)}, same_object={same_object}", task_id=task.id)
                        
                        # Patch svi2pro
                        wgp.wan_model.model_def["svi2pro"] = True
                        
                        # CRITICAL: Also patch sliding_window=True - required for video continuation
                        # Without this, reuse_frames=0 and video_source context is ignored
                        _sliding_window_original = wgp.wan_model.model_def.get("sliding_window")
                        wgp.wan_model.model_def["sliding_window"] = True
                        
                        # Also patch wgp.models_def for consistency
                        if model_key in wgp.models_def:
                            wgp.models_def[model_key]["sliding_window"] = True
                        
                        _wan_model_patched = True
                        
                        # Verify the patches took effect
                        verify_svi2pro = wgp.wan_model.model_def.get("svi2pro")
                        verify_sliding = wgp.wan_model.model_def.get("sliding_window")
                        self.logger.info(f"[SVI2PRO] ✅ Patched wan_model.model_def: svi2pro={verify_svi2pro}, sliding_window={verify_sliding} (was: {_sliding_window_original})", task_id=task.id)
                    else:
                        self.logger.warning(f"[SVI2PRO] ⚠️ wan_model exists but has no model_def", task_id=task.id)
                else:
                    self.logger.warning(f"[SVI2PRO] ⚠️ wgp.wan_model not found - model may not be loaded yet", task_id=task.id)
                    
            except Exception as e:
                self.logger.warning(f"[SVI2PRO] Failed to patch svi2pro: {e}", task_id=task.id)
            # Remove from generation_params since it's not a generate() parameter
            generation_params.pop("svi2pro", None)

        # Log generation parameters for debugging
        dprint(f"[GENERATION_DEBUG] Task {task.id}: Generation parameters:")
        for key, value in generation_params.items():
            if key in ["video_guide", "video_mask", "image_refs"]:
                dprint(f"[GENERATION_DEBUG]   {key}: {value}")
            elif key in ["video_length", "resolution", "num_inference_steps"]:
                dprint(f"[GENERATION_DEBUG]   {key}: {value}")

        # Determine generation type and delegate - wrap in try/finally for patch restoration
        try:
            # Check if model supports VACE features
            model_supports_vace = self._model_supports_vace(task.model)
            
            if model_supports_vace:
                dprint(f"[GENERATION_DEBUG] Task {task.id}: Using VACE generation path")
                
                # CRITICAL: VACE models require a video_guide parameter
                if "video_guide" in generation_params and generation_params["video_guide"]:
                    dprint(f"[GENERATION_DEBUG] Task {task.id}: Video guide provided: {generation_params['video_guide']}")
                else:
                    error_msg = f"VACE model '{task.model}' requires a video_guide parameter but none was provided. VACE models cannot perform pure text-to-video generation."
                    self.logger.error(f"[GENERATION_DEBUG] Task {task.id}: {error_msg}")
                    raise ValueError(error_msg)
                
                result = self.orchestrator.generate_vace(
                    prompt=task.prompt,
                    model_type=task.model,  # Pass model type for parameter resolution
                    **generation_params
                )
            elif self.orchestrator._is_flux():
                dprint(f"[GENERATION_DEBUG] Task {task.id}: Using Flux generation path")
                
                # For Flux, map video_length to num_images
                if "video_length" in generation_params:
                    generation_params["num_images"] = generation_params.pop("video_length")
                
                result = self.orchestrator.generate_flux(
                    prompt=task.prompt,
                    model_type=task.model,  # Pass model type for parameter resolution
                    **generation_params
                )
            else:
                dprint(f"[GENERATION_DEBUG] Task {task.id}: Using T2V generation path")
                
                # T2V or other models - pass model_type for proper parameter resolution
                # Note: WGP stdout is captured to svi_debug.txt file instead of logger
                # to avoid recursion issues
                result = self.orchestrator.generate_t2v(
                    prompt=task.prompt,
                    model_type=task.model,  # ← CRITICAL: Pass model type for parameter resolution
                    **generation_params
                )
            
            self.logger.info(f"{worker_name} generation completed for task {task.id}: {result}")

            # Post-process single frame videos to PNG for single_image tasks
            # BUT: Skip PNG conversion for travel segments (they must remain as videos for stitching)
            is_travel_segment = task.parameters.get("_source_task_type") == "travel_segment"
            if self._is_single_image_task(task) and not is_travel_segment:
                png_result = self._convert_single_frame_video_to_png(task, result, worker_name)
                if png_result:
                    self.logger.info(f"{worker_name} converted single frame video to PNG: {png_result}")
                    return png_result

            return result

        except Exception as e:
            self.logger.error(f"{worker_name} generation failed for task {task.id}: {e}")
            raise
        finally:
            # CRITICAL: Restore model patches to prevent contamination across tasks
            if _patch_applied and _parsed_phase_config_for_restore and _model_name_for_restore:
                try:
                    from source.phase_config import restore_model_patches
                    restore_model_patches(
                        _parsed_phase_config_for_restore,
                        _model_name_for_restore,
                        task.id
                    )
                    self.logger.info(f"[PHASE_CONFIG] Restored original model definition for '{_model_name_for_restore}' after task {task.id}")
                except Exception as restore_error:
                    self.logger.warning(f"[PHASE_CONFIG] Failed to restore model patches for task {task.id}: {restore_error}")
            
            # Restore svi2pro and sliding_window if we patched them
            if _svi2pro_patched or _wan_model_patched:
                try:
                    import wgp
                    model_key = task.model
                    
                    # Restore wgp.models_def
                    if _svi2pro_patched and model_key in wgp.models_def:
                        if _svi2pro_original is None:
                            wgp.models_def[model_key].pop("svi2pro", None)
                        else:
                            wgp.models_def[model_key]["svi2pro"] = _svi2pro_original
                        # Also restore sliding_window
                        wgp.models_def[model_key].pop("sliding_window", None)
                        self.logger.info(f"[SVI2PRO] Restored wgp.models_def['{model_key}']['svi2pro'] to {_svi2pro_original}", task_id=task.id)
                    
                    # Restore wan_model.model_def
                    if _wan_model_patched and hasattr(wgp, 'wan_model') and wgp.wan_model is not None:
                        if hasattr(wgp.wan_model, 'model_def') and wgp.wan_model.model_def is not None:
                            if _svi2pro_original is None:
                                wgp.wan_model.model_def.pop("svi2pro", None)
                            else:
                                wgp.wan_model.model_def["svi2pro"] = _svi2pro_original
                            # Also restore sliding_window
                            wgp.wan_model.model_def.pop("sliding_window", None)
                            self.logger.info(f"[SVI2PRO] Restored wan_model.model_def: svi2pro={_svi2pro_original}, sliding_window=removed", task_id=task.id)
                            
                except Exception as restore_error:
                    self.logger.warning(f"[SVI2PRO] Failed to restore svi2pro for task {task.id}: {restore_error}")

    def _model_supports_vace(self, model_key: str) -> bool:
        """
        Check if a model supports VACE features (video guides, masks, etc.).
        """
        # Ensure orchestrator is initialized before checking model support
        self._ensure_orchestrator()
        
        try:
            # Use orchestrator's VACE detection with model key
            if hasattr(self.orchestrator, 'is_model_vace'):
                return self.orchestrator.is_model_vace(model_key)
            elif hasattr(self.orchestrator, '_is_vace'):
                # Fallback: load model and check (less efficient)
                current_model = self.current_model
                if current_model != model_key:
                    # Would need to load model to check - use name-based detection as fallback
                    return "vace" in model_key.lower()
                return self.orchestrator._is_vace()
            else:
                # Ultimate fallback: name-based detection
                return "vace" in model_key.lower()
        except Exception as e:
            self.logger.warning(f"Could not determine VACE support for model '{model_key}': {e}")
            return "vace" in model_key.lower()
    
    def _is_single_image_task(self, task: GenerationTask) -> bool:
        """
        Check if this is a single image task that should be converted from video to PNG.
        """
        # Check if video_length is 1 (single frame) and this looks like an image task
        video_length = task.parameters.get("video_length", 0)
        return video_length == 1
    
    def _convert_single_frame_video_to_png(self, task: GenerationTask, video_path: str, worker_name: str) -> str:
        """
        Convert a single-frame video to PNG format for single image tasks.
        
        This restores the functionality that was in the original single_image.py handler
        where single-frame videos were converted to PNG files.
        """
        try:
            import cv2
            
            video_path_obj = Path(video_path)
            if not video_path_obj.exists():
                self.logger.error(f"Video file does not exist for PNG conversion: {video_path}")
                return video_path  # Return original path if conversion fails
            
            # Create PNG output path with sanitized filename to prevent upload issues
            original_filename = video_path_obj.stem
            
            # Sanitize the filename for storage compatibility
            try:
                # Try to import the existing sanitization function
                import sys
                source_dir = Path(__file__).parent / "source"
                if str(source_dir) not in sys.path:
                    sys.path.insert(0, str(source_dir))
                from common_utils import sanitize_filename_for_storage  # type: ignore
                
                sanitized_filename = sanitize_filename_for_storage(original_filename)
                if not sanitized_filename:
                    sanitized_filename = "generated_image"
                    
            except ImportError:
                # Fallback sanitization if import fails
                import re
                sanitized_filename = re.sub(r'[§®©™@·º½¾¿¡~\x00-\x1F\x7F-\x9F<>:"/\\|?*,]', '', original_filename)
                sanitized_filename = re.sub(r'\s+', '_', sanitized_filename.strip())
                if not sanitized_filename:
                    sanitized_filename = "generated_image"
            
            # Create PNG path with sanitized filename
            png_path = video_path_obj.parent / f"{sanitized_filename}.png"
            
            # Log sanitization if filename changed
            if sanitized_filename != original_filename:
                self.logger.info(f"[PNG_CONVERSION] Task {task.id}: Sanitized filename '{original_filename}' -> '{sanitized_filename}'")
            
            self.logger.info(f"[PNG_CONVERSION] Task {task.id}: Converting {video_path_obj.name} to {png_path.name}")
            
            # Extract the first frame using OpenCV
            cap = cv2.VideoCapture(str(video_path_obj))
            try:
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        # Save the frame as PNG
                        success = cv2.imwrite(str(png_path), frame)
                        if success and png_path.exists():
                            self.logger.info(f"[PNG_CONVERSION] Task {task.id}: Successfully saved PNG to {png_path}")
                            
                            # Clean up the original video file
                            try:
                                video_path_obj.unlink()
                                self.logger.info(f"[PNG_CONVERSION] Task {task.id}: Removed original video file")
                            except Exception as e_cleanup:
                                self.logger.warning(f"[PNG_CONVERSION] Task {task.id}: Could not remove original video: {e_cleanup}")
                            
                            return str(png_path)
                        else:
                            self.logger.error(f"[PNG_CONVERSION] Task {task.id}: Failed to save PNG to {png_path}")
                    else:
                        self.logger.error(f"[PNG_CONVERSION] Task {task.id}: Failed to read frame from video")
                else:
                    self.logger.error(f"[PNG_CONVERSION] Task {task.id}: Failed to open video file")
            finally:
                cap.release()
                
        except ImportError:
            self.logger.warning(f"[PNG_CONVERSION] Task {task.id}: OpenCV not available, keeping video format")
        except Exception as e:
            self.logger.error(f"[PNG_CONVERSION] Task {task.id}: Error during conversion: {e}")
        
        # Return original video path if conversion failed
        return video_path
    
    def _monitor_loop(self):
        """Background monitoring and maintenance loop."""
        self.logger.info("Queue monitor started")
        
        while self.running and not self.shutdown_event.is_set():
            try:
                # TODO: Implement monitoring features:
                
                # 1. Memory usage monitoring
                # memory_info = self._get_memory_usage()
                # if memory_info["gpu_usage"] > 0.9:
                #     self.logger.warning("High GPU memory usage detected")
                
                # 2. Queue health monitoring 
                # queue_size = self.task_queue.qsize()
                # if queue_size > 100:
                #     self.logger.warning(f"Large queue detected: {queue_size} tasks")
                
                # 3. Task timeout monitoring
                # self._check_task_timeouts()
                
                # 4. Auto-save queue state (integrate with wgp.py's autosave)
                # self._periodic_save()
                
                # 5. Model memory optimization
                # self._optimize_model_memory()
                
                time.sleep(10.0)  # Monitor every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Monitor error: {e}\n{traceback.format_exc()}")
                time.sleep(5.0)
        
        self.logger.info("Queue monitor stopped")
    
    def _convert_to_wgp_task(self, task: GenerationTask) -> Dict[str, Any]:
        """
        Convert task to WGP parameters using typed TaskConfig.
        
        This:
        1. Parses all params into TaskConfig at the boundary
        2. Handles LoRA downloads via LoRAConfig
        3. Converts to WGP format only at the end
        """
        # Parse into typed config
        config = TaskConfig.from_db_task(
            task.parameters,
            task_id=task.id,
            task_type=task.parameters.get('_source_task_type', ''),
            model=task.model,
            debug_mode=self.debug_mode
        )
        
        # Add prompt and model
        config.generation.prompt = task.prompt
        config.model = task.model
        
        # Log the parsed config
        if self.debug_mode:
            config.log_summary(self.logger.info)
        
        # Handle LoRA downloads if any are pending
        if config.lora.has_pending_downloads():
            self.logger.info(f"[LORA_PROCESS] Task {task.id}: {len(config.lora.get_pending_downloads())} LoRAs need downloading")
            
            # Ensure we're in Wan2GP directory for LoRA operations
            _saved_cwd = os.getcwd()
            if _saved_cwd != self.wan_dir:
                os.chdir(self.wan_dir)
            
            try:
                from source.lora_utils import _download_lora_from_url
                
                for url, mult in list(config.lora.get_pending_downloads().items()):
                    try:
                        local_path = _download_lora_from_url(url, task.id, dprint, model_type=task.model)
                        if local_path:
                            config.lora.mark_downloaded(url, local_path)
                            self.logger.info(f"[LORA_DOWNLOAD] Task {task.id}: Downloaded {os.path.basename(local_path)}")
                        else:
                            self.logger.warning(f"[LORA_DOWNLOAD] Task {task.id}: Failed to download {url}")
                    except Exception as e:
                        self.logger.warning(f"[LORA_DOWNLOAD] Task {task.id}: Error downloading {url}: {e}")
            finally:
                if _saved_cwd != self.wan_dir:
                    os.chdir(_saved_cwd)
        
        # Validate before conversion
        errors = config.validate()
        if errors:
            self.logger.warning(f"[TASK_CONFIG] Task {task.id}: Validation warnings: {errors}")
        
        # Convert to WGP format (single conversion point)
        wgp_params = config.to_wgp_format()
        
        # Ensure prompt and model are set
        wgp_params["prompt"] = task.prompt
        wgp_params["model"] = task.model
        
        # Filter out infrastructure params
        for param in ["supabase_url", "supabase_anon_key", "supabase_access_token"]:
            wgp_params.pop(param, None)
        
        if self.debug_mode:
            self.logger.info(f"[TASK_CONVERSION] Task {task.id}: Converted with {len(wgp_params)} params")
            self.logger.debug(f"[TASK_CONVERSION] Task {task.id}: LoRAs: {wgp_params.get('activated_loras', [])}")
        
        return wgp_params
    
    def _apply_sampler_cfg_preset(self, model_key: str, sample_solver: str, wgp_params: Dict[str, Any]):
        """Apply sampler-specific CFG and flow_shift settings from model configuration."""
        try:
            # Import WGP to get model definition
            # Protect sys.argv in case wgp hasn't been imported yet
            _saved_argv = sys.argv[:]
            try:
                sys.argv = ["headless_model_management.py"]
                import wgp
            finally:
                sys.argv = _saved_argv

            model_def = wgp.get_model_def(model_key)
            
            # Check if model has sampler-specific presets
            sampler_presets = model_def.get("sampler_cfg_presets", {})
            if sample_solver in sampler_presets:
                preset = sampler_presets[sample_solver]
                
                # Apply preset settings, but allow task parameters to override
                applied_params = {}
                for param, value in preset.items():
                    if param not in wgp_params:  # Only apply if not explicitly set in task
                        wgp_params[param] = value
                        applied_params[param] = value
                        
                self.logger.info(f"Applied sampler '{sample_solver}' CFG preset: {applied_params}")
            else:
                self.logger.debug(f"No CFG preset found for sampler '{sample_solver}' in model '{model_key}'")
                
        except Exception as e:
            self.logger.warning(f"Failed to apply sampler CFG preset: {e}")
    
    def _get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage statistics."""
        # TODO: Implement memory monitoring
        # - GPU memory (torch.cuda.memory_stats())
        # - System RAM 
        # - Model memory usage
        return {
            "gpu_memory_used": 0,
            "gpu_memory_total": 0,
            "system_memory_used": 0,
            "model_memory_usage": 0
        }
    
    def _save_queue_state(self):
        """Save current queue state for persistence."""
        # TODO: Integrate with wgp.py's save_queue_action()
        # This would allow restoring queue state on restart
        pass
    
    def _load_queue_state(self):
        """Load saved queue state on startup."""
        # TODO: Integrate with wgp.py's load_queue_action()
        pass


class HeadlessAPI:
    """
    Simple HTTP API for submitting tasks to the queue.
    
    This provides a REST interface for external systems to submit
    generation tasks without needing to interact with the queue directly.
    """
    
    def __init__(self, queue: HeadlessTaskQueue, port: int = 8080):
        self.queue = queue
        self.port = port
        self.app = None  # TODO: Add Flask/FastAPI app
    
    def start(self):
        """Start the HTTP API server."""
        # TODO: Implement REST API with endpoints:
        # POST /generate - Submit new task
        # GET /status/{task_id} - Get task status  
        # GET /queue - Get queue status
        # DELETE /tasks/{task_id} - Cancel task
        pass
    
    def stop(self):
        """Stop the HTTP API server."""
        pass


def create_sample_task(task_id: str, model: str, prompt: str, **params) -> GenerationTask:
    """Helper to create sample tasks for testing."""
    return GenerationTask(
        id=task_id,
        model=model,
        prompt=prompt,
        parameters=params
    )


def main():
    """Main entry point for the headless service."""
    parser = argparse.ArgumentParser(description="WanGP Headless Task Queue")
    parser.add_argument("--wan-dir", required=True, help="Path to WanGP directory")
    parser.add_argument("--port", type=int, default=8080, help="API server port")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker threads")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize queue
    task_queue = HeadlessTaskQueue(args.wan_dir, max_workers=args.workers)
    
    # Initialize API (optional)
    api = HeadlessAPI(task_queue, port=args.port)
    
    # Setup signal handlers
    def signal_handler(signum, frame):
        print("\nReceived shutdown signal, stopping...")
        task_queue.stop()
        api.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start services
        task_queue.start()
        # api.start()  # TODO: Uncomment when API is implemented
        
        print(f"🚀 Headless queue started on port {args.port}")
        print(f"📁 WanGP directory: {args.wan_dir}")
        print(f"👥 Workers: {args.workers}")
        print("Press Ctrl+C to stop...")
        
        # Example: Submit some test tasks
        if args.debug:
            print("\n🧪 Submitting test tasks...")
            
            # Test T2V task
            t2v_task = create_sample_task(
                "test-t2v-1",
                "t2v",
                "a mystical forest with glowing trees",
                resolution="1280x720",
                video_length=49,
                seed=42
            )
            task_queue.submit_task(t2v_task)
            
            # Test VACE task (would need actual video file)
            # vace_task = create_sample_task(
            #     "test-vace-1", 
            #     "vace_14B",
            #     "a cyberpunk dancer in neon lights",
            #     video_guide="inputs/dance.mp4",
            #     video_prompt_type="VP",
            #     resolution="1280x720"
            # )
            # task_queue.submit_task(vace_task)
        
        # Keep running until shutdown
        while task_queue.running:
            time.sleep(1.0)
            
            # Print periodic status
            if args.debug:
                status = task_queue.get_queue_status()
                print(f"📊 Queue: {status.pending_tasks} pending, "
                      f"{status.completed_tasks} completed, "
                      f"{status.failed_tasks} failed")
    
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"Fatal error: {e}")
        raise
    finally:
        task_queue.stop()
        api.stop()


if __name__ == "__main__":
    main()
