"""Single image task handler - updated for queue-only system."""

import json
import tempfile
from pathlib import Path

from .. import db_operations as db_ops
from ..common_utils import (
    DEBUG_MODE, dprint, upload_and_get_final_output_location
)


def _handle_single_image_task(task_params_from_db: dict, main_output_dir_base: Path, task_id: str, image_download_dir: Path | str | None = None, *, dprint, task_queue=None):
    """
    DEPRECATED: Single image task handler - now handled via direct queue integration in worker.py
    
    This function should not be called anymore as worker.py routes single_image tasks 
    directly to HeadlessTaskQueue for better efficiency and model persistence.
    
    This is kept for backward compatibility but will return an error to indicate
    the architectural change.
    """
    error_msg = (
        f"Single image task {task_id}: DEPRECATED HANDLER CALLED. "
        f"Single image tasks are now processed via direct queue integration in worker.py. "
        f"This indicates a routing configuration issue - single_image tasks should be "
        f"handled by the direct_queue_task_types logic, not this legacy handler."
    )
    dprint(error_msg)
    return False, error_msg
                