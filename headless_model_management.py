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
    os.chdir(wan_dir)
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
        import wgp
        sys.argv = _saved_argv
        self.wgp = wgp
        
        # Import our orchestrator
        from headless_wgp import WanOrchestrator
        self.orchestrator = WanOrchestrator(wan_dir)
        
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
    
    def _init_wgp_integration(self):
        """
        Initialize integration with wgp.py's existing systems.
        
        This reuses wgp.py's state management, queue handling, and model persistence
        rather than reimplementing it.
        """
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
            
            # 3. Update task status
            processing_time = time.time() - start_time
            with self.queue_lock:
                task.status = "completed"
                task.result_path = result_path
                task.processing_time = processing_time
                self.stats["tasks_completed"] += 1
                self.stats["total_generation_time"] += processing_time
            
            self.logger.info(f"Task {task.id} completed in {processing_time:.1f}s: {result_path}")
            
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
        self.logger.info(f"{worker_name} executing generation for task {task.id} (model: {task.model})")
        
        # Convert task parameters to WanOrchestrator format
        wgp_params = self._convert_to_wgp_task(task)
        
        # Remove model and prompt from params since they're passed separately to avoid duplication
        generation_params = {k: v for k, v in wgp_params.items() if k not in ("model", "prompt")}
        
        # Log generation parameters for debugging
        self.logger.info(f"[GENERATION_DEBUG] Task {task.id}: Generation parameters:")
        for key, value in generation_params.items():
            if key in ["video_guide", "video_mask", "image_refs"]:
                self.logger.info(f"[GENERATION_DEBUG]   {key}: {value}")
            elif key in ["video_length", "resolution", "num_inference_steps"]:
                self.logger.info(f"[GENERATION_DEBUG]   {key}: {value}")
        
        # Determine generation type and delegate
        try:
            # Check if model supports VACE features
            model_supports_vace = self._model_supports_vace(task.model)
            
            if model_supports_vace:
                self.logger.info(f"[GENERATION_DEBUG] Task {task.id}: Using VACE generation path")
                
                # CRITICAL: VACE models require a video_guide parameter
                if "video_guide" in generation_params and generation_params["video_guide"]:
                    self.logger.info(f"[GENERATION_DEBUG] Task {task.id}: Video guide provided: {generation_params['video_guide']}")
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
                self.logger.info(f"[GENERATION_DEBUG] Task {task.id}: Using Flux generation path")
                
                # For Flux, map video_length to num_images
                if "video_length" in generation_params:
                    generation_params["num_images"] = generation_params.pop("video_length")
                
                result = self.orchestrator.generate_flux(
                    prompt=task.prompt,
                    model_type=task.model,  # Pass model type for parameter resolution
                    **generation_params
                )
            else:
                self.logger.info(f"[GENERATION_DEBUG] Task {task.id}: Using T2V generation path")
                
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
            
            # Create PNG output path (same directory, different extension)
            png_path = video_path_obj.with_suffix('.png')
            
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
            "video_guide2": "video_guide2",
            "video_mask2": "video_mask2", 
            "video_prompt_type": "video_prompt_type",
            "control_net_weight": "control_net_weight",
            "control_net_weight2": "control_net_weight2",
            "embedded_guidance_scale": "embedded_guidance_scale",
            "denoise_strength": "denoising_strength",  # Map LightI2X parameter name
            "guidance2_scale": "guidance2_scale",
            "sample_solver": "sample_solver",
            "lora_names": "lora_names",
            "lora_multipliers": "lora_multipliers",
        }
        

        
        # Map parameters with proper defaults
        for our_param, wgp_param in param_mapping.items():
            if our_param in task.parameters:
                value = task.parameters[our_param]
                
                # Special handling for file path parameters
                if our_param in ["video_guide", "video_mask", "video_guide2", "video_mask2"] and value:
                    # Ensure path exists and convert to absolute path
                    try:
                        path_obj = Path(value)
                        if path_obj.exists():
                            wgp_params[wgp_param] = str(path_obj.resolve())
                            self.logger.info(f"[PARAM_DEBUG] Task {task.id}: {our_param} path validated: {wgp_params[wgp_param]}")
                        else:
                            self.logger.warning(f"[PARAM_DEBUG] Task {task.id}: {our_param} path does not exist: {value}")
                            # Don't include invalid paths
                            continue
                    except Exception as e:
                        self.logger.warning(f"[PARAM_DEBUG] Task {task.id}: Error processing {our_param} path '{value}': {e}")
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
        
        # Parameter resolution is now handled by WanOrchestrator._resolve_parameters()
        # This provides clean separation: HeadlessTaskQueue manages tasks, WanOrchestrator handles parameters
        self.logger.info(f"[TASK_CONVERSION] Converting task {task.id} for model '{task.model}' - parameter resolution delegated to orchestrator")
        
        # Apply sampler-specific CFG settings if available
        sample_solver = task.parameters.get("sample_solver", wgp_params.get("sample_solver", ""))
        if sample_solver:
            self._apply_sampler_cfg_preset(task.model, sample_solver, wgp_params)
        
        # Apply special LoRA settings (CausVid, LightI2X) using shared utilities
        import sys
        source_dir = Path(__file__).parent / "source"
        if str(source_dir) not in sys.path:
            sys.path.insert(0, str(source_dir))
        from lora_utils import detect_lora_optimization_flags, apply_lora_parameter_optimization, ensure_lora_in_list
        
        # [DEBUG] Log task parameters for LoRA detection debugging
        self.logger.info(f"[LORA_DEBUG] Task {task.id}: task.parameters keys: {list(task.parameters.keys())}")
        self.logger.info(f"[LORA_DEBUG] Task {task.id}: use_causvid_lora in params: {'use_causvid_lora' in task.parameters}")
        self.logger.info(f"[LORA_DEBUG] Task {task.id}: use_lighti2x_lora in params: {'use_lighti2x_lora' in task.parameters}")
        if 'use_lighti2x_lora' in task.parameters:
            self.logger.info(f"[LORA_DEBUG] Task {task.id}: use_lighti2x_lora value: {task.parameters['use_lighti2x_lora']}")
        if 'use_causvid_lora' in task.parameters:
            self.logger.info(f"[LORA_DEBUG] Task {task.id}: use_causvid_lora value: {task.parameters['use_causvid_lora']}")
        
        # [DEEP_DEBUG] Log ALL parameters for this task to see everything
        self.logger.info(f"[DEEP_DEBUG] Task {task.id}: FULL task.parameters: {task.parameters}")
        
        # Detect LoRA optimization flags using shared logic
        use_causvid, use_lighti2x = detect_lora_optimization_flags(
            task_params=task.parameters,
            model_name=task.model,
            dprint=lambda msg: self.logger.info(msg)
        )
        
        # [CausVidDebugTrace] Enhanced parameter inspection
        self.logger.info(f"[CausVidDebugTrace] Task {task.id}: Pre-processing parameter analysis:")
        self.logger.info(f"[CausVidDebugTrace]   use_causvid: {use_causvid}")
        self.logger.info(f"[CausVidDebugTrace]   use_lighti2x: {use_lighti2x}")
        self.logger.info(f"[CausVidDebugTrace]   wgp_params keys before LoRA optimization: {list(wgp_params.keys())}")
        self.logger.info(f"[CausVidDebugTrace]   'num_inference_steps' in wgp_params: {'num_inference_steps' in wgp_params}")
        if "num_inference_steps" in wgp_params:
            self.logger.info(f"[CausVidDebugTrace]   existing num_inference_steps value: {wgp_params['num_inference_steps']}")
        
        # Apply LoRA parameter optimization using shared logic
        if use_causvid or use_lighti2x:
            wgp_params = apply_lora_parameter_optimization(
                params=wgp_params,
                causvid_enabled=use_causvid,
                lighti2x_enabled=use_lighti2x,
                model_name=task.model,
                task_params=task.parameters,
                task_id=task.id,
                dprint=lambda msg: self.logger.info(msg)
            )
        
        # Ensure required LoRAs are in the activated list
        if use_causvid:
            wgp_params = ensure_lora_in_list(
                params=wgp_params,
                lora_filename="Wan21_CausVid_14B_T2V_lora_rank32_v2.safetensors",
                lora_type="CausVid",
                task_id=task.id,
                dprint=lambda msg: self.logger.info(msg)
            )
            
            self.logger.info(f"[Task {task.id}] Applied CausVid LoRA settings via shared utilities")
        
        if use_lighti2x:
            wgp_params = ensure_lora_in_list(
                params=wgp_params,
                lora_filename="Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors",
                lora_type="LightI2X",
                task_id=task.id,
                dprint=lambda msg: self.logger.info(msg)
            )
            
            # LightI2X-specific additional settings
            wgp_params["sample_solver"] = "unipc"
            wgp_params["denoise_strength"] = 1.0
            
            self.logger.info(f"[Task {task.id}] Applied LightI2X LoRA settings via shared utilities")
        
        if not use_causvid and not use_lighti2x:
            self.logger.info(f"[CausVidDebugTrace] Task {task.id}: No LoRA optimizations enabled")
        
        # ADDITIONAL LORAS: Process additional LoRA names and multipliers from task parameters
        # Support both dict format ({"url": multiplier}) and list format ([names], [multipliers])
        additional_lora_names = task.parameters.get("additional_lora_names", [])
        additional_lora_multipliers = task.parameters.get("additional_lora_multipliers", [])
        
        # Check for dict format and convert if needed
        additional_loras_dict = task.parameters.get("additional_loras", {})
        self.logger.info(f"[PATH_TRACE] Task {task.id}: additional_loras_dict = {additional_loras_dict}")
        self.logger.info(f"[PATH_TRACE] Task {task.id}: additional_loras_dict type = {type(additional_loras_dict)}")
        self.logger.info(f"[PATH_TRACE] Task {task.id}: bool(additional_loras_dict) = {bool(additional_loras_dict)}")
        self.logger.info(f"[PATH_TRACE] Task {task.id}: isinstance(additional_loras_dict, dict) = {isinstance(additional_loras_dict, dict)}")
        
        if additional_loras_dict and isinstance(additional_loras_dict, dict):
            self.logger.info(f"[PATH_TRACE] Task {task.id}: ENTERING dict conversion branch")
            # Convert dict format to lists format and handle URL downloading
            dict_names = list(additional_loras_dict.keys())
            dict_multipliers = list(additional_loras_dict.values())
            
            self.logger.info(f"[PATH_TRACE] Task {task.id}: dict_names = {dict_names}")
            
            # Process URLs - download if needed and convert to local filenames
            processed_names = []
            for i, lora_name_or_url in enumerate(dict_names):
                self.logger.info(f"[PATH_TRACE] Task {task.id}: Processing LoRA {i}: '{lora_name_or_url}'")
                self.logger.info(f"[PATH_TRACE] Task {task.id}: lora_name_or_url.startswith('http') = {lora_name_or_url.startswith('http')}")
                
                if lora_name_or_url.startswith("http"):
                    self.logger.info(f"[PATH_TRACE] Task {task.id}: URL detected, starting download process")
                    # It's a URL - download it
                    try:
                        # Extract filename from URL
                        local_filename = lora_name_or_url.split("/")[-1]
                        self.logger.info(f"[PATH_TRACE] Task {task.id}: local_filename = '{local_filename}'")
                        
                        # Get LoRA directory for the current model
                        from Wan2GP.wgp import get_lora_dir
                        lora_dir = get_lora_dir(task.model)
                        local_path = os.path.join(lora_dir, local_filename)
                        self.logger.info(f"[PATH_TRACE] Task {task.id}: lora_dir = '{lora_dir}'")
                        self.logger.info(f"[PATH_TRACE] Task {task.id}: local_path = '{local_path}'")
                        self.logger.info(f"[PATH_TRACE] Task {task.id}: os.path.isfile(local_path) = {os.path.isfile(local_path)}")
                        
                        # Check if file already exists
                        if not os.path.isfile(local_path):
                            self.logger.info(f"[LORA_DOWNLOAD] Task {task.id}: Downloading LoRA from {lora_name_or_url}")
                            
                            # Use existing download logic pattern from WGP
                            
                            # Use WGP's download_file logic - need to import it properly
                            if lora_name_or_url.startswith("https://huggingface.co/") and "/resolve/main/" in lora_name_or_url:
                                from huggingface_hub import hf_hub_download
                                import shutil
                                
                                # Parse HuggingFace URL
                                url = lora_name_or_url[len("https://huggingface.co/"):]
                                url_parts = url.split("/resolve/main/")
                                repo_id = url_parts[0]
                                filename = os.path.basename(url_parts[-1])
                                
                                # Ensure LoRA directory exists
                                os.makedirs(lora_dir, exist_ok=True)
                                
                                # Download using HuggingFace hub
                                hf_hub_download(repo_id=repo_id, filename=filename, local_dir=lora_dir)
                                self.logger.info(f"[LORA_DOWNLOAD] Task {task.id}: Successfully downloaded {filename} to {lora_dir}")
                            else:
                                # Use urllib for other URLs
                                from urllib.request import urlretrieve
                                os.makedirs(lora_dir, exist_ok=True)
                                urlretrieve(lora_name_or_url, local_path)
                                self.logger.info(f"[LORA_DOWNLOAD] Task {task.id}: Successfully downloaded {local_filename} to {lora_dir}")
                        else:
                            self.logger.info(f"[LORA_DOWNLOAD] Task {task.id}: LoRA {local_filename} already exists locally")
                        
                        # Use local filename instead of URL
                        processed_names.append(local_filename)
                        
                    except Exception as e:
                        self.logger.error(f"[LORA_DOWNLOAD] Task {task.id}: Failed to download LoRA from {lora_name_or_url}: {e}")
                        # Keep original URL in case there's a fallback mechanism
                        processed_names.append(lora_name_or_url)
                else:
                    # It's already a local filename
                    self.logger.info(f"[PATH_TRACE] Task {task.id}: Local filename detected: '{lora_name_or_url}'")
                    processed_names.append(lora_name_or_url)
            
            # Merge with existing lists (dict format takes precedence)
            additional_lora_names.extend(processed_names)
            additional_lora_multipliers.extend(dict_multipliers)
            
            self.logger.info(f"[PATH_TRACE] Task {task.id}: processed_names = {processed_names}")
            self.logger.info(f"[PATH_TRACE] Task {task.id}: final additional_lora_names = {additional_lora_names}")
            
            self.logger.info(f"[CausVidDebugTrace] Task {task.id}: Converted additional_loras dict to lists: {len(processed_names)} LoRAs")
            for i, (name, mult) in enumerate(zip(processed_names, dict_multipliers)):
                self.logger.info(f"[CausVidDebugTrace]   {i+1}. {name} (multiplier: {mult})")
        else:
            self.logger.info(f"[PATH_TRACE] Task {task.id}: NOT entering dict conversion branch - using existing lists")
        
        if additional_lora_names:
            self.logger.info(f"[CausVidDebugTrace] Task {task.id}: Processing {len(additional_lora_names)} additional LoRAs")
            
            # Get current LoRA lists (may have been modified by CausVid/LightI2X logic above)
            current_loras = wgp_params.get("lora_names", [])
            current_multipliers = wgp_params.get("lora_multipliers", [])
            
            # Add additional LoRAs to the lists
            for i, lora_name in enumerate(additional_lora_names):
                if lora_name not in current_loras:
                    current_loras.append(lora_name)
                    multiplier = additional_lora_multipliers[i] if i < len(additional_lora_multipliers) else 1.0
                    current_multipliers.append(multiplier)
                    self.logger.info(f"[CausVidDebugTrace] Task {task.id}: Added additional LoRA: {lora_name} (multiplier: {multiplier})")
                else:
                    self.logger.debug(f"[CausVidDebugTrace] Task {task.id}: Additional LoRA {lora_name} already in list")
            
            # Update the wgp_params with combined LoRA lists
            wgp_params["lora_names"] = current_loras
            wgp_params["lora_multipliers"] = current_multipliers
            
            self.logger.info(f"[CausVidDebugTrace] Task {task.id}: Final combined LoRA list: {current_loras}")
            self.logger.info(f"[CausVidDebugTrace] Task {task.id}: Final combined multipliers: {current_multipliers}")
            self.logger.info(f"[PATH_TRACE] Task {task.id}: About to pass these LoRAs to WGP: {current_loras}")
        
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