"""Magic Edit functionality using Replicate API.

This module provides the magic_edit task handler that processes images through
the flux-kontext-apps/in-scene model on Replicate to generate scene variations.
"""

import os
import traceback
import tempfile
import shutil
from pathlib import Path
import requests
from typing import Tuple

try:
    import replicate
except ImportError:
    replicate = None

from .. import db_operations as db_ops
from ..common_utils import (
    sm_get_unique_target_path,
    download_image_if_url as sm_download_image_if_url,
    prepare_output_path_with_upload,
    upload_and_get_final_output_location,
    report_orchestrator_failure
)


def _handle_magic_edit_task(
    task_params_from_db: dict,
    main_output_dir_base: Path,
    task_id: str,
    *,
    dprint
) -> Tuple[bool, str]:
    """
    Handle a magic_edit task by processing an image through Replicate's black-forest-labs/flux-kontext-dev-lora model.
    
    Args:
        task_params_from_db: Task parameters containing orchestrator_details
        main_output_dir_base: Base directory for outputs
        task_id: Unique task identifier
        dprint: Debug print function
        
    Returns:
        Tuple of (success_bool, output_location_or_error_message)
    """
    print(f"[Task ID: {task_id}] Starting magic_edit task")
    
    try:
        # Check if replicate is available
        if replicate is None:
            msg = "Replicate library not installed. Run: pip install replicate"
            print(f"[ERROR Task ID: {task_id}] {msg}")
            report_orchestrator_failure(task_params_from_db, msg, dprint)
            return False, msg
            
        # Check for API token
        api_token = os.getenv("REPLICATE_API_TOKEN")
        if not api_token:
            msg = "REPLICATE_API_TOKEN environment variable not set"
            print(f"[ERROR Task ID: {task_id}] {msg}")
            report_orchestrator_failure(task_params_from_db, msg, dprint)
            return False, msg
            
        # Extract orchestrator details
        orchestrator_details = task_params_from_db.get("orchestrator_details", {})
        if not orchestrator_details:
            msg = "No orchestrator_details found in task parameters"
            print(f"[ERROR Task ID: {task_id}] {msg}")
            report_orchestrator_failure(task_params_from_db, msg, dprint)
            return False, msg
            
        # Required parameters
        image_url = orchestrator_details.get("image_url")
        prompt = orchestrator_details.get("prompt", "Make a shot in the same scene from a different angle")
        resolution = orchestrator_details.get("resolution", "768x576")
        seed = orchestrator_details.get("seed")
        in_scene = orchestrator_details.get("in_scene", False)  # Default to False
        
        if not image_url:
            msg = "image_url is required in orchestrator_details"
            print(f"[ERROR Task ID: {task_id}] {msg}")
            report_orchestrator_failure(task_params_from_db, msg, dprint)
            return False, msg
            
        dprint(f"Task {task_id}: Magic edit parameters - Image: {image_url}, Prompt: {prompt}, Resolution: {resolution}, In-Scene LoRA: {in_scene}")
        
        # Create temporary directory for processing
        temp_dir = Path(tempfile.mkdtemp(prefix=f"magic_edit_{task_id}_"))
        dprint(f"Task {task_id}: Created temp directory: {temp_dir}")
        
        try:
            # Validate that we have a URL (Replicate expects URLs, not local files)
            if not image_url.startswith(("http://", "https://")):
                msg = f"image_url must be a valid HTTP/HTTPS URL, got: {image_url}"
                print(f"[ERROR Task ID: {task_id}] {msg}")
                return False, msg
                    
            dprint(f"Task {task_id}: Using image URL directly with Replicate: {image_url}")
            
            # Prepare Replicate input
            replicate_input = {
                "prompt": prompt,
                "input_image": image_url,
                "aspect_ratio": "match_input_image",  # Let Replicate match the input image aspect ratio
                "lora_strength": 1,
                "output_format": "webp",
                "output_quality": 90
            }
            
            # Only add LoRA weights if in_scene is True
            if in_scene:
                replicate_input["lora_weights"] = "https://huggingface.co/peteromallet/Flux-Kontext-InScene/resolve/main/InScene-v1.0.safetensors"
                dprint(f"Task {task_id}: Using InScene LoRA for scene-consistent generation")
            else:
                dprint(f"Task {task_id}: Using base Flux model without InScene LoRA")
            
            # Add seed if specified
            if seed is not None:
                replicate_input["seed"] = int(seed)
                
            print(f"[Task ID: {task_id}] Running Replicate black-forest-labs/flux-kontext-dev-lora model...")
            dprint(f"Task {task_id}: Replicate input: {replicate_input}")
            
            # Run the model
            output = replicate.run(
                "black-forest-labs/flux-kontext-dev-lora",
                input=replicate_input
            )
            
            print(f"[Task ID: {task_id}] Replicate processing completed")
            dprint(f"Task {task_id}: Replicate output type: {type(output)}")
            
            # Download the result
            output_image_path = temp_dir / f"magic_edit_{task_id}.webp"
            
            if hasattr(output, 'read'):
                # Output is a file-like object
                with open(output_image_path, 'wb') as f:
                    f.write(output.read())
            elif hasattr(output, 'url'):
                # Output has a URL method
                response = requests.get(output.url(), timeout=60)
                response.raise_for_status()
                with open(output_image_path, 'wb') as f:
                    f.write(response.content)
            elif isinstance(output, str) and output.startswith(("http://", "https://")):
                # Output is a URL string
                response = requests.get(output, timeout=60)
                response.raise_for_status()
                with open(output_image_path, 'wb') as f:
                    f.write(response.content)
            else:
                msg = f"Unexpected output type from Replicate: {type(output)}"
                print(f"[ERROR Task ID: {task_id}] {msg}")
                return False, msg
                
            if not output_image_path.exists() or output_image_path.stat().st_size == 0:
                msg = "Failed to download result from Replicate"
                print(f"[ERROR Task ID: {task_id}] {msg}")
                return False, msg
                
            print(f"[Task ID: {task_id}] Downloaded result image: {output_image_path.name} ({output_image_path.stat().st_size} bytes)")
            
            # Prepare final output path
            final_output_path, initial_db_location = prepare_output_path_with_upload(
                task_id=task_id,
                filename=output_image_path.name,
                main_output_dir_base=main_output_dir_base,
                dprint=dprint
            )
            
            # Move to final location
            shutil.move(str(output_image_path), str(final_output_path))
            dprint(f"Task {task_id}: Moved result to final location: {final_output_path}")
            
            # Handle upload and get final DB location
            final_db_location = upload_and_get_final_output_location(
                final_output_path,
                task_id,
                initial_db_location,
                dprint=dprint
            )
            
            print(f"[Task ID: {task_id}] Magic edit completed successfully: {final_output_path.resolve()}")
            return True, final_db_location
            
        finally:
            # Cleanup temp directory
            try:
                shutil.rmtree(temp_dir)
                dprint(f"Task {task_id}: Cleaned up temp directory: {temp_dir}")
            except Exception as e_cleanup:
                print(f"[WARNING Task ID: {task_id}] Failed to cleanup temp directory {temp_dir}: {e_cleanup}")
                
    except Exception as e:
        print(f"[ERROR Task ID: {task_id}] Magic edit task failed: {e}")
        traceback.print_exc()
        msg = f"Magic edit exception: {e}"
        report_orchestrator_failure(task_params_from_db, msg, dprint)
        return False, msg 