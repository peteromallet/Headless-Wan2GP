"""
Task Engine Router

Simple routing: tasks go to either ComfyUI or Wan2GP based on task_type.
No specialized handlers - just two engines.
"""

from typing import Dict, Any, Tuple, Optional
from pathlib import Path

from source.comfy_handler import handle_comfy_task
from source.worker_utils import make_task_dprint
from source.task_conversion import db_task_to_generation_task
import time
import traceback

# Define which task types use ComfyUI
COMFY_TASK_TYPES = {
    "comfy",
    # Add more ComfyUI task types here as needed
    # "comfy_image", "comfy_video", etc.
}

# All other task types use Wan2GP by default


def route_task(task_type: str, context: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Route task to appropriate engine: ComfyUI or Wan2GP.

    Args:
        task_type: The type of task (e.g., 'comfy', 'vace', 'flux')
        context: Task context with params, output dir, task_queue, etc.

    Returns:
        (success, output_path)
    """
    task_id = context["task_id"]
    params = context["task_params_dict"]
    dprint_func = make_task_dprint(task_id, context["debug_mode"])

    # Route to ComfyUI
    if task_type in COMFY_TASK_TYPES:
        dprint_func(f"Routing to ComfyUI engine")
        return handle_comfy_task(
            task_params_from_db=params,
            main_output_dir_base=context["main_output_dir_base"],
            task_id=task_id,
            dprint=dprint_func
        )

    # Route to Wan2GP (everything else)
    dprint_func(f"Routing to Wan2GP engine")
    return _handle_wan2gp_task(task_type, context)


def _handle_wan2gp_task(task_type: str, context: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Handle task using Wan2GP engine."""
    task_id = context["task_id"]
    task_queue = context["task_queue"]
    params = context["task_params_dict"]

    if not task_queue:
        return False, "Wan2GP task queue not available"

    try:
        # Convert DB task to generation task
        generation_task = db_task_to_generation_task(
            params, task_id, task_type,
            context["wan2gp_path"],
            context["debug_mode"]
        )

        # Apply video processing flags if set
        if context.get("colour_match_videos"):
            generation_task.parameters["colour_match_videos"] = True
        if context.get("mask_active_frames"):
            generation_task.parameters["mask_active_frames"] = True

        # Submit to Wan2GP queue
        task_queue.submit_task(generation_task)

        # Wait for completion
        max_wait_time = 3600  # 1 hour
        elapsed = 0

        while elapsed < max_wait_time:
            status = task_queue.get_task_status(task_id)

            if not status:
                return False, "Task status became None"

            if status.status == "completed":
                return True, status.result_path
            elif status.status == "failed":
                return False, status.error_message or "Failed without message"

            time.sleep(2)
            elapsed += 2

        return False, f"Task timeout after {max_wait_time}s"

    except Exception as e:
        traceback.print_exc()
        return False, f"Wan2GP error: {e}"
