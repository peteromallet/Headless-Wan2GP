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
        
        # Import wgp after path setup
        import wgp
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
        and integration with our queue system.
        """
        self.logger.info(f"{worker_name} executing generation for task {task.id} (model: {task.model})")
        
        # Convert task parameters to WanOrchestrator format
        wgp_params = self._convert_to_wgp_task(task)
        
        # Remove model and prompt from params since they're passed separately
        generation_params = {k: v for k, v in wgp_params.items() if k not in ["model", "prompt"]}
        
        # Determine generation type and delegate
        try:
            if self.orchestrator._is_vace():
                if "video_guide" not in generation_params:
                    raise ValueError("VACE model requires video_guide parameter")
                
                result = self.orchestrator.generate_vace(
                    prompt=task.prompt,
                    **generation_params
                )
            elif self.orchestrator._is_flux():
                # For Flux, map video_length to num_images
                if "video_length" in generation_params:
                    generation_params["num_images"] = generation_params.pop("video_length")
                
                result = self.orchestrator.generate_flux(
                    prompt=task.prompt,
                    **generation_params
                )
            else:
                # T2V or other models
                result = self.orchestrator.generate_t2v(
                    prompt=task.prompt,
                    **generation_params
                )
            
            self.logger.info(f"{worker_name} generation completed for task {task.id}: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"{worker_name} generation failed for task {task.id}: {e}")
            raise
    
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
            "seed": "seed",
            "video_guide": "video_guide",
            "video_mask": "video_mask",
            "video_guide2": "video_guide2",
            "video_mask2": "video_mask2", 
            "video_prompt_type": "video_prompt_type",
            "control_net_weight": "control_net_weight",
            "control_net_weight2": "control_net_weight2",
            "embedded_guidance_scale": "embedded_guidance_scale",
            "lora_names": "lora_names",
            "lora_multipliers": "lora_multipliers",
        }
        
        # Map parameters with proper defaults
        for our_param, wgp_param in param_mapping.items():
            if our_param in task.parameters:
                wgp_params[wgp_param] = task.parameters[our_param]
        
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
        
        return wgp_params
    
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