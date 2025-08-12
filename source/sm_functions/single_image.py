"""Single image task handler - updated for queue-only system."""

import json
import tempfile
from pathlib import Path

from .. import db_operations as db_ops
from ..common_utils import (
    DEBUG_MODE, dprint, upload_and_get_final_output_location
)


def _handle_single_image_task(task_params_from_db: dict, main_output_dir_base: Path, task_id: str, image_download_dir: Path | str | None = None, apply_reward_lora: bool = False, *, dprint, task_queue=None):
    """
    Handles single image generation tasks using the new task_queue system.
    
    Args:
        task_params_from_db: Task parameters from the database
        main_output_dir_base: Base output directory
        task_id: Task ID for logging
        image_download_dir: Directory for downloading images if URLs are provided
        apply_reward_lora: Whether to apply reward LoRA
        task_queue: The queue system for processing tasks
    
    Returns:
        Tuple[bool, str]: (success, output_location_or_error_message)
    """
    output_dir = main_output_dir_base / "single_images"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Use task_queue system if available - this is the new queue-only approach
        if task_queue is None:
            error_msg = f"Single image task {task_id}: task_queue is required for queue-only system"
            dprint(error_msg)
            return False, error_msg
            
        model_name = task_params_from_db.get("model", "vace_14B_cocktail")
        dprint(f"Single image task {task_id}: Starting processing with model '{model_name}' via task_queue")

        # Convert the prompt
        prompt = task_params_from_db.get("prompt", "")
        negative_prompt = task_params_from_db.get("negative_prompt", "")
        
        if not prompt.strip():
            error_msg = f"Single image task {task_id}: No prompt provided"
            dprint(error_msg)
            return False, error_msg
        
        # Load model defaults from JSON config
        model_defaults = {}
        try:
            import sys
            from pathlib import Path
            wan_dir = Path(__file__).parent.parent.parent / "Wan2GP"
            if str(wan_dir) not in sys.path:
                sys.path.insert(0, str(wan_dir))
            
            import wgp
            model_defaults = wgp.get_default_settings(model_name)
            dprint(f"[SINGLE_IMAGE_DEBUG] Task {task_id}: Loaded model defaults for '{model_name}': {model_defaults}")
        except Exception as e:
            dprint(f"[SINGLE_IMAGE_DEBUG] Task {task_id}: Warning - could not load model defaults for '{model_name}': {e}")

        # Build parameters for the task queue system
        # Note: HeadlessTaskQueue will handle model defaults and task parameter overrides properly
        generation_params = {
            "task_id": task_id,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "model": model_name,
            "resolution": task_params_from_db.get("resolution", "512x512"),
            "video_length": 1,  # Single frame for images
            "seed": task_params_from_db.get("seed", -1),
        }
        
        # Add any task-specific parameter overrides
        overridable_params = ["num_inference_steps", "guidance_scale", "flow_shift", "switch_threshold"]
        for param in overridable_params:
            if param in task_params_from_db:
                generation_params[param] = task_params_from_db[param]
                dprint(f"[SINGLE_IMAGE_DEBUG] Task {task_id}: Task override {param}={task_params_from_db[param]}")
        
        # Add reference images if provided
        if task_params_from_db.get("image_refs_paths"):
            generation_params["image_refs"] = task_params_from_db["image_refs_paths"]
            dprint(f"Single image task {task_id}: Added reference images: {generation_params['image_refs']}")
        
        # Add LoRA settings if provided
        if apply_reward_lora or task_params_from_db.get("apply_reward_lora", False):
            # Add reward LoRA to the generation
            generation_params["apply_reward_lora"] = True
            dprint(f"Single image task {task_id}: Applying reward LoRA")

        # Add any additional LoRAs
        if task_params_from_db.get("activated_loras"):
            generation_params["activated_loras"] = task_params_from_db["activated_loras"]
            generation_params["loras_multipliers"] = task_params_from_db.get("loras_multipliers", [])
        
        dprint(f"Single image task {task_id}: Submitting generation task with parameters: {generation_params}")
        
        # Create a GenerationTask for the queue system
        from headless_model_management import GenerationTask
        
        generation_task = GenerationTask(
            task_id=task_id,
            model_name=model_name,
            **generation_params
        )
        
        # Execute the task using the queue system
        try:
            task_queue.submit_task(generation_task)
            result = task_queue.wait_for_completion(task_id, timeout=300)  # 5 minute timeout
            
            if result and result.get("success", False):
                output_path = result.get("output_path")
                if output_path and Path(output_path).exists():
                    # Handle upload if configured
                    final_output_location = upload_and_get_final_output_location(
                        local_path=output_path,
                        task_type="single_image",
                        task_id=task_id,
                        dprint=dprint
                    )
                    
                    dprint(f"Single image task {task_id}: Generation completed successfully. Output: {final_output_location}")
                    return True, final_output_location
                else:
                    error_msg = f"Single image task {task_id}: Generation completed but no output file found"
                    dprint(error_msg)
                    return False, error_msg
            else:
                error_msg = f"Single image task {task_id}: Generation failed: {result.get('error', 'Unknown error')}"
                dprint(error_msg)
                return False, error_msg
                
        except Exception as e:
            error_msg = f"Single image task {task_id}: Exception during generation: {e}"
            dprint(error_msg)
            return False, error_msg
    
    except Exception as e:
        error_msg = f"Single image task {task_id}: Unexpected error: {e}"
        dprint(error_msg)
        return False, error_msg
