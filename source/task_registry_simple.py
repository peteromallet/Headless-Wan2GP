"""
Simplified Task Registry - Two Engines Only

This is a cleaner alternative to task_registry.py that routes ALL tasks
to either ComfyUI or Wan2GP based on task_type.

To use this instead of the complex task_registry.py:
1. Rename task_registry.py to task_registry_legacy.py
2. Rename this file to task_registry.py
3. Test with existing tasks

WARNING: This removes all specialized handlers (travel_orchestrator, magic_edit, etc.)
Only use this if those handlers are not needed or can be refactored.
"""

from typing import Dict, Any, Tuple, Optional
from pathlib import Path

from source.logging_utils import headless_logger
from source.worker_utils import make_task_dprint
from source.task_engine_router import route_task, COMFY_TASK_TYPES


# Define Direct Queue Task Types (Wan2GP)
# These are all handled by the Wan2GP engine
DIRECT_QUEUE_TASK_TYPES = {
    "wan_2_2_t2i", "vace", "vace_21", "vace_22", "flux", "t2v", "t2v_22",
    "i2v", "i2v_22", "hunyuan", "ltxv", "generate_video",
    "qwen_image_edit", "qwen_image_style", "image_inpaint", "annotated_image_edit"
}


class TaskRegistry:
    """
    Simplified task registry with two engines:
    - ComfyUI: Workflow-based tasks
    - Wan2GP: All AI generation tasks
    """

    @staticmethod
    def dispatch(task_type: str, context: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Route task to appropriate engine: ComfyUI or Wan2GP.

        Routing logic:
        1. If task_type in COMFY_TASK_TYPES → ComfyUI engine
        2. Everything else → Wan2GP engine

        Args:
            task_type: The type of task to execute
            context: Dictionary containing all necessary context variables

        Returns:
            Tuple (success, output_location)
        """
        task_id = context["task_id"]

        # Log routing decision
        if task_type in COMFY_TASK_TYPES:
            headless_logger.debug(f"Routing {task_type} to ComfyUI engine", task_id=task_id)
        else:
            headless_logger.debug(f"Routing {task_type} to Wan2GP engine", task_id=task_id)

        # Route to appropriate engine
        return route_task(task_type, context)


# For backward compatibility, expose the same interface
def dispatch(task_type: str, context: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Convenience function that calls TaskRegistry.dispatch()"""
    return TaskRegistry.dispatch(task_type, context)
