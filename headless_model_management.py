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
    
    def __init__(self, wan_dir: str, max_workers: int = 1):
        """
        Initialize the headless task queue.
        
        Args:
            wan_dir: Path to WanGP directory
            max_workers: Number of concurrent generation workers (recommend 1 for GPU)
        """
        self.wan_dir = setup_wgp_path(wan_dir)
        self.max_workers = max_workers
        self.running = False
        self.start_time = time.time()
        
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
        """Setup structured logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('headless.log')
            ]
        )
        self.logger = logging.getLogger('HeadlessQueue')
    
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
            self.logger.info("[LAZY_INIT] Initializing WanOrchestrator (first use)...")
            self.logger.info("[LAZY_INIT] Warming up CUDA before importing wgp...")
            
            # Warm up CUDA before importing wgp (upstream T5EncoderModel has torch.cuda.current_device() 
            # as a default arg, which is evaluated at module import time)
            import torch
            if torch.cuda.is_available():
                self.logger.info(f"[LAZY_INIT] CUDA is available: {torch.cuda.get_device_name(0)}")
                # Initialize CUDA by accessing a device
                _ = torch.cuda.current_device()
                self.logger.info(f"[LAZY_INIT] CUDA initialized on device {torch.cuda.current_device()}")
            else:
                self.logger.warning("[LAZY_INIT] ⚠️  No CUDA device available! Model loading may fail.")
            
            self.logger.info("[LAZY_INIT] Importing WanOrchestrator (this imports wgp and model modules)...")
            from headless_wgp import WanOrchestrator
            self.orchestrator = WanOrchestrator(self.wan_dir)
            
            self.logger.info("[LAZY_INIT] ✅ WanOrchestrator initialized successfully")
            
            # Now that orchestrator exists, complete wgp integration
            self._init_wgp_integration()
            
        except Exception as e:
            self.logger.error(f"[LAZY_INIT] ❌ Failed to initialize WanOrchestrator: {e}")
            self.logger.error("[LAZY_INIT] Traceback:", exc_info=True)
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
    
    def start(self):
        """Start the task queue processing service."""
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
                self.logger.error(f"{worker_name} error: {e}", exc_info=True)
                time.sleep(1.0)
        
        self.logger.info(f"{worker_name} stopped")
    
    def _process_task(self, task: GenerationTask, worker_name: str):
        """
        Process a single generation task.
        
        This is where we delegate to headless_wgp.py while managing
        model persistence and state.
        """
        with self.queue_lock:
            self.current_task = task
            task.status = "processing"
        
        self.logger.info(f"{worker_name} processing task {task.id}")
        start_time = time.time()
        
        try:
            # 1. Handle model switching (leverage wgp.py's model management)
            if task.model != self.current_model:
                self._switch_model(task.model, worker_name)
            
            # 2. Delegate actual generation to orchestrator
            # The orchestrator handles the heavy lifting while we manage the queue
            result_path = self._execute_generation(task, worker_name)

            # 3. Validate output and update task status
            processing_time = time.time() - start_time
            is_success = bool(result_path)
            try:
                if is_success:
                    # If a path was returned, check existence where possible
                    rp = Path(result_path)
                    is_success = rp.exists()
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
            
        except Exception as e:
            # Handle task failure
            processing_time = time.time() - start_time
            with self.queue_lock:
                task.status = "failed"
                task.error_message = str(e)
                task.processing_time = processing_time
                self.stats["tasks_failed"] += 1
            
            self.logger.error(f"Task {task.id} failed after {processing_time:.1f}s: {e}")
        
        finally:
            with self.queue_lock:
                self.current_task = None
    
    def _switch_model(self, model_key: str, worker_name: str):
        """
        Switch to a different model using wgp.py's model management.
        
        This leverages the orchestrator's model loading while tracking
        the change in our queue system.
        """
        # Ensure orchestrator is initialized before switching models
        self._ensure_orchestrator()
        
        if model_key == self.current_model:
            return
        
        self.logger.info(f"{worker_name} switching model: {self.current_model} → {model_key}")
        switch_start = time.time()
        
        try:
            # Use orchestrator's model loading (which uses wgp.py's persistence)
            self.orchestrator.load_model(model_key)
            self.current_model = model_key
            self.stats["model_switches"] += 1
            
            switch_time = time.time() - switch_start
            self.logger.info(f"Model switch completed in {switch_time:.1f}s")
            
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
        
        # Log generation parameters for debugging
        dprint(f"[GENERATION_DEBUG] Task {task.id}: Generation parameters:")
        for key, value in generation_params.items():
            if key in ["video_guide", "video_mask", "image_refs"]:
                dprint(f"[GENERATION_DEBUG]   {key}: {value}")
            elif key in ["video_length", "resolution", "num_inference_steps"]:
                dprint(f"[GENERATION_DEBUG]   {key}: {value}")
        
        # Determine generation type and delegate
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
                result = self.orchestrator.generate_t2v(
                    prompt=task.prompt,
                    model_type=task.model,  # ← CRITICAL: Pass model type for parameter resolution
                    **generation_params
                )
            
            self.logger.info(f"{worker_name} generation completed for task {task.id}: {result}")
            
            # Post-process single frame videos to PNG for single_image tasks
            if self._is_single_image_task(task):
                png_result = self._convert_single_frame_video_to_png(task, result, worker_name)
                if png_result:
                    self.logger.info(f"{worker_name} converted single frame video to PNG: {png_result}")
                    return png_result
            
            return result
            
        except Exception as e:
            self.logger.error(f"{worker_name} generation failed for task {task.id}: {e}")
            raise
    
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
                self.logger.error(f"Monitor error: {e}", exc_info=True)
                time.sleep(5.0)
        
        self.logger.info("Queue monitor stopped")
    
    def _convert_to_wgp_task(self, task: GenerationTask) -> Dict[str, Any]:
        """
        Convert our task format to WanOrchestrator-compatible parameters.
        
        This maps our GenerationTask structure to the parameters expected by
        the WanOrchestrator generate methods.
        """
        # Start with base parameters
        wgp_params = {
            "prompt": task.prompt,
            "model": task.model,
        }
        
        # Add all parameters from the task, with some parameter name mapping
        param_mapping = {
            # Map any parameter name differences between our format and WanOrchestrator
            "negative_prompt": "negative_prompt",
            "resolution": "resolution", 
            "video_length": "video_length",
            "num_inference_steps": "num_inference_steps",
            "guidance_scale": "guidance_scale",
            "flow_shift": "flow_shift",  # Important for CausVid/LightI2X settings
            "seed": "seed",
            "video_guide": "video_guide",
            "video_mask": "video_mask",
            "image_guide": "image_guide",
            "image_mask": "image_mask",
            "video_prompt_type": "video_prompt_type",
            "control_net_weight": "control_net_weight",
            "control_net_weight2": "control_net_weight2",
            "embedded_guidance_scale": "embedded_guidance_scale",
            "denoise_strength": "denoising_strength",  # Map LightI2X parameter name
            "guidance2_scale": "guidance2_scale",
            "guidance3_scale": "guidance3_scale",
            "switch_threshold": "switch_threshold",
            "switch_threshold2": "switch_threshold2",
            "guidance_phases": "guidance_phases",
            "model_switch_phase": "model_switch_phase",
            "image_refs_relative_size": "image_refs_relative_size",
            "override_profile": "override_profile",
            "sample_solver": "sample_solver",
            "lora_names": "lora_names",
            "lora_multipliers": "lora_multipliers",
        }
        

        
        # Helper: resolve media paths preferring repo root over Wan2GP cwd
        def _resolve_media_path(val: str) -> Optional[str]:
            if not val:
                return None
            p = Path(val)
            try:
                if p.is_absolute():
                    return str(p.resolve()) if p.exists() else None
                repo_root = Path(__file__).parent
                wan_root = Path.cwd()
                # Prefer repo root for relative paths (one level up from Wan2GP)
                candidate = repo_root / p
                if candidate.exists():
                    return str(candidate.resolve())
                candidate = wan_root / p
                if candidate.exists():
                    return str(candidate.resolve())
            except Exception:
                return None
            return None

        # Map parameters with proper defaults
        for our_param, wgp_param in param_mapping.items():
            if our_param in task.parameters:
                value = task.parameters[our_param]
                
                # Special handling for file path parameters
                if our_param in ["video_guide", "video_mask", "image_guide", "image_mask"] and value:
                    resolved = _resolve_media_path(str(value))
                    if resolved:
                        wgp_params[wgp_param] = resolved
                        self.logger.info(f"[PARAM_DEBUG] Task {task.id}: {our_param} path resolved: {resolved}")
                    else:
                        self.logger.warning(f"[PARAM_DEBUG] Task {task.id}: {our_param} not found at '{value}' (checked repo and Wan2GP); skipping")
                        continue
                
                # Special handling for image references
                elif our_param == "image_refs" and value:
                    if isinstance(value, list):
                        # Validate each image path
                        valid_images = []
                        for img_path in value:
                            try:
                                img_path_obj = Path(img_path)
                                if img_path_obj.exists():
                                    valid_images.append(str(img_path_obj.resolve()))
                                else:
                                    self.logger.warning(f"[PARAM_DEBUG] Task {task.id}: Image ref does not exist: {img_path}")
                            except Exception as e:
                                self.logger.warning(f"[PARAM_DEBUG] Task {task.id}: Error processing image ref '{img_path}': {e}")
                        
                        if valid_images:
                            wgp_params["image_refs"] = valid_images
                            self.logger.info(f"[PARAM_DEBUG] Task {task.id}: Validated {len(valid_images)} image references")
                    else:
                        # Single image reference
                        try:
                            img_path_obj = Path(value)
                            if img_path_obj.exists():
                                wgp_params["image_refs"] = [str(img_path_obj.resolve())]
                                self.logger.info(f"[PARAM_DEBUG] Task {task.id}: Validated single image reference")
                            else:
                                self.logger.warning(f"[PARAM_DEBUG] Task {task.id}: Single image ref does not exist: {value}")
                        except Exception as e:
                            self.logger.warning(f"[PARAM_DEBUG] Task {task.id}: Error processing single image ref '{value}': {e}")
                
                else:
                    # Normal parameter mapping
                    wgp_params[wgp_param] = value
        
        # Handle LoRA parameter format conversion
        if "activated_loras" in task.parameters:
            wgp_params["lora_names"] = task.parameters["activated_loras"]
        if "loras_multipliers" in task.parameters:
            # Convert from string format "1.0,0.8" to list
            if isinstance(task.parameters["loras_multipliers"], str):
                multipliers_str = task.parameters["loras_multipliers"]
                wgp_params["lora_multipliers"] = [float(x.strip()) for x in multipliers_str.split(",") if x.strip()]
            else:
                wgp_params["lora_multipliers"] = task.parameters["loras_multipliers"]

        # Ensure additional_loras is forwarded so LoRA processing can normalize/download them
        if "additional_loras" in task.parameters:
            wgp_params["additional_loras"] = task.parameters["additional_loras"]
            dprint(f"[LORA_PROCESS] Task {task.id}: Forwarded {len(task.parameters['additional_loras'])} additional LoRAs to processor")
        
        # Parameter resolution is now handled by WanOrchestrator._resolve_parameters()
        # This provides clean separation: HeadlessTaskQueue manages tasks, WanOrchestrator handles parameters
        self.logger.info(f"[TASK_CONVERSION] Converting task {task.id} for model '{task.model}' - parameter resolution delegated to orchestrator")
        
        # Apply sampler-specific CFG settings if available
        sample_solver = task.parameters.get("sample_solver", wgp_params.get("sample_solver", ""))
        if sample_solver:
            self._apply_sampler_cfg_preset(task.model, sample_solver, wgp_params)
        
        # Complete LoRA processing pipeline using centralized utilities
        import sys
        source_dir = Path(__file__).parent / "source"
        if str(source_dir) not in sys.path:
            sys.path.insert(0, str(source_dir))
        from lora_utils import process_all_loras
        
        # Use centralized LoRA processing pipeline
        dprint(f"[LORA_PROCESS] Task {task.id}: Starting centralized LoRA processing for model {task.model}")
        
        wgp_params = process_all_loras(
            params=wgp_params,
            task_params=task.parameters,
            model_name=task.model,
            orchestrator_payload=task.parameters.get("orchestrator_payload"),
            task_id=task.id,
            dprint=dprint
        )
        
        dprint(f"[LORA_PROCESS] Task {task.id}: Centralized LoRA processing complete")

        
        return wgp_params
    
    def _apply_sampler_cfg_preset(self, model_key: str, sample_solver: str, wgp_params: Dict[str, Any]):
        """Apply sampler-specific CFG and flow_shift settings from model configuration."""
        try:
            # Import WGP to get model definition
            import wgp
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
