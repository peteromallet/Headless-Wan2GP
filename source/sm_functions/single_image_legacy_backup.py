"""Single image generation task handler."""

import json
import tempfile
import traceback
from pathlib import Path

# Import from the restructured modules
from .. import db_operations as db_ops
from ..common_utils import (
    sm_get_unique_target_path,
    download_image_if_url as sm_download_image_if_url,
    load_pil_images as sm_load_pil_images,
    parse_resolution as sm_parse_resolution,
    build_task_state,
    prepare_output_path,
    process_additional_loras_shared,  # New shared function
    snap_resolution_to_model_grid,    # New shared function
    prepare_output_path_with_upload,  # New shared function
    upload_and_get_final_output_location  # New shared function
)
from ..wgp_utils import generate_single_video


def _handle_single_image_task(task_params_from_db: dict, main_output_dir_base: Path, task_id: str, image_download_dir: Path | str | None = None, apply_reward_lora: bool = False, *, dprint, task_queue=None):
    """
    Handles single image generation tasks.
    
    Args:
        wgp_mod: The WGP module for generation
        task_params_from_db: Task parameters from the database
        main_output_dir_base: Base output directory
        task_id: Task ID for logging
        image_download_dir: Directory for downloading images if URLs are provided
        apply_reward_lora: Whether to apply reward LoRA
        dprint: Debug print function
    
    Returns:
        Tuple[bool, str]: (success, output_location_or_error_message)
    """
    print(f"[FLOW_SHIFT_DEBUG] _handle_single_image_task: Starting for {task_id} - THIS PATH IS BEING USED")
    dprint(f"_handle_single_image_task: Starting for {task_id}")
    dprint(f"Single image task params: {json.dumps(task_params_from_db, default=str, indent=2)}")
    
    try:
        # -------------------------------------------------------------
        # Flatten orchestrator_details (if present) so that nested keys
        # like `use_causvid_lora` or `prompt` become first-class entries.
        # Top-level keys take precedence over nested ones in case of clash.
        # -------------------------------------------------------------
        if isinstance(task_params_from_db.get("orchestrator_details"), dict):
            task_params_from_db = {
                **task_params_from_db["orchestrator_details"],  # Nested first
                **{k: v for k, v in task_params_from_db.items() if k != "orchestrator_details"},  # Top-level override
            }

        # Extract required parameters with defaults
        prompt = task_params_from_db.get("prompt", " ").strip() or " "  # Default to space if empty
        model_name = task_params_from_db.get("model", "t2v")  # Default to t2v model
        resolution = task_params_from_db.get("resolution", "832x480")
        seed = task_params_from_db.get("seed", -1)
        negative_prompt = task_params_from_db.get("negative_prompt", "").strip() or " "  # Default to space if empty
        
        # Validate and parse resolution with model grid snapping
        try:
            parsed_res = sm_parse_resolution(resolution)
            if parsed_res is None:
                raise ValueError(f"Invalid resolution format: {resolution}")
            width, height = snap_resolution_to_model_grid(parsed_res)
            resolution = f"{width}x{height}"
            dprint(f"Single image task {task_id}: Adjusted resolution to {resolution}")
        except Exception as e_res:
            error_msg = f"Single image task {task_id}: Resolution parsing failed: {e_res}"
            print(f"[ERROR] {error_msg}")
            return False, error_msg
        
        # Handle reference images if provided
        image_refs_paths = []
        if task_params_from_db.get("image_refs_paths"):
            try:
                # With queue system, we'll pass image paths directly for processing
                dprint(f"Single image task {task_id}: Processing reference images via queue system")
                # loaded_refs = sm_load_pil_images(
                #     task_params_from_db["image_refs_paths"],
                #     wgp_mod.convert_image,  # No longer available
                #     image_download_dir,
                #     task_id,
                #     dprint
                # )
                if loaded_refs:
                    # Convert back to paths for the payload
                    image_refs_paths = task_params_from_db["image_refs_paths"]
            except Exception as e_refs:
                dprint(f"Single image task {task_id}: Warning - failed to load reference images: {e_refs}")
        
        # Use model_name directly - it should match JSON files in defaults/ directory
        actual_model_type = model_name
        print(f"[FLOW_SHIFT_DEBUG] Single image task {task_id}: model_name='{model_name}' → actual_model_type='{actual_model_type}'")
        dprint(f"Single image task {task_id}: using model_type='{actual_model_type}'")
        dprint(f"Single image task {task_id}: transformer_quantization='{wgp_mod.transformer_quantization}'")
        # Fix empty transformer_dtype_policy which causes get_model_filename to fail
        dtype_policy = wgp_mod.transformer_dtype_policy
        if not dtype_policy:
            dtype_policy = "auto"  # Use default value
            dprint(f"Single image task {task_id}: transformer_dtype_policy was empty, using default: '{dtype_policy}'")
        else:
            dprint(f"Single image task {task_id}: transformer_dtype_policy='{dtype_policy}'")
        
        # Determine model filename for LoRA handling
        model_filename_for_task = wgp_mod.get_model_filename(
            actual_model_type,
            wgp_mod.transformer_quantization,
            dtype_policy
        )
        dprint(f"Single image task {task_id}: model_filename_for_task='{model_filename_for_task}'")
        
        # Handle special LoRA downloads (CausVid, LightI2X) - same logic as in worker.py
        from ..common_utils import get_lora_dir_from_filename, download_file
        base_lora_dir_for_model = Path(get_lora_dir_from_filename(wgp_mod, model_filename_for_task))
        
        if task_params_from_db.get("use_causvid_lora", False):
            causvid_lora_name = "Wan21_CausVid_14B_T2V_lora_rank32_v2.safetensors"
            causvid_target_path = base_lora_dir_for_model / causvid_lora_name
            
            if not causvid_target_path.exists():
                dprint(f"Single image task {task_id}: {causvid_lora_name} not found. Attempting download...")
                causvid_url = "https://huggingface.co/DeepBeepMeep/Wan2.1/resolve/main/loras_accelerators/Wan21_CausVid_14B_T2V_lora_rank32_v2.safetensors"
                
                if not download_file(causvid_url, base_lora_dir_for_model, causvid_lora_name):
                    dprint(f"Single image task {task_id}: WARNING - Failed to download {causvid_lora_name}. Proceeding without CausVid LoRA.")
                    task_params_from_db["use_causvid_lora"] = False
                else:
                    dprint(f"Single image task {task_id}: Successfully downloaded {causvid_lora_name}")
            else:
                dprint(f"Single image task {task_id}: {causvid_lora_name} already exists")
                
        if task_params_from_db.get("use_lighti2x_lora", False):
            lighti2x_lora_name = "Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors"
            lighti2x_target_path = base_lora_dir_for_model / lighti2x_lora_name
            
            if not lighti2x_target_path.exists():
                dprint(f"Single image task {task_id}: {lighti2x_lora_name} not found. Attempting download...")
                lighti2x_url = "https://huggingface.co/DeepBeepMeep/Wan2.1/resolve/main/loras_accelerators/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors"
                
                if not download_file(lighti2x_url, base_lora_dir_for_model, lighti2x_lora_name):
                    dprint(f"Single image task {task_id}: WARNING - Failed to download {lighti2x_lora_name}. Proceeding without LightI2X LoRA.")
                    task_params_from_db["use_lighti2x_lora"] = False
                else:
                    dprint(f"Single image task {task_id}: Successfully downloaded {lighti2x_lora_name}")
            else:
                dprint(f"Single image task {task_id}: {lighti2x_lora_name} already exists")
        
        # Handle additional LoRAs using shared function
        processed_additional_loras = {}
        additional_loras = task_params_from_db.get("additional_loras", {})
        if additional_loras:
            dprint(f"Single image task {task_id}: Processing additional LoRAs: {additional_loras}")
            processed_additional_loras = process_additional_loras_shared(
                additional_loras, 
                wgp_mod, 
                model_filename_for_task, 
                task_id, 
                dprint
            )
        
        # Prepare the output path (with Supabase upload support)
        output_filename = f"single_image_{task_id}.png"
        local_output_path, initial_db_output_location = prepare_output_path_with_upload(
            task_id,
            output_filename,
            main_output_dir_base,
            dprint=dprint
        )
        
        # Create a temporary directory for WGP processing
        with tempfile.TemporaryDirectory(prefix=f"single_img_{task_id}_") as temp_dir:
            temp_dir_path = Path(temp_dir)
            temp_video_path = temp_dir_path / f"{task_id}_temp.mp4"
            
            dprint(f"Single image task {task_id}: Using temp directory {temp_dir_path}")
            
            # Set up WGP module state temporarily
            original_save_path = wgp_mod.save_path
            wgp_mod.save_path = str(temp_dir_path)
            
            try:
                dprint(f"Single image task {task_id}: Calling generate_single_video with new flexible API")
                
                # Get preset defaults from model definition FIRST
                try:
                    # Debug: Check what the model definition contains
                    model_def = wgp_mod.get_model_def(actual_model_type)
                    print(f"[FLOW_SHIFT_DEBUG] Single image task {task_id}: model_def for '{actual_model_type}': {model_def}")
                    if model_def:
                        settings = model_def.get("settings", None)
                        print(f"[FLOW_SHIFT_DEBUG] Single image task {task_id}: model_def['settings']: {settings}")
                    
                    preset_defaults = wgp_mod.get_default_settings(actual_model_type)
                    print(f"[FLOW_SHIFT_DEBUG] Single image task {task_id}: Loaded preset defaults for '{actual_model_type}': steps={preset_defaults.get('num_inference_steps')}, guidance={preset_defaults.get('guidance_scale')}, flow={preset_defaults.get('flow_shift')}")
                    print(f"[FLOW_SHIFT_DEBUG] Single image task {task_id}: Full preset_defaults dict: {preset_defaults}")
                    dprint(f"Single image task {task_id}: Loaded preset defaults for '{actual_model_type}': steps={preset_defaults.get('num_inference_steps')}, guidance={preset_defaults.get('guidance_scale')}, flow={preset_defaults.get('flow_shift')}")
                except Exception as e:
                    print(f"[FLOW_SHIFT_DEBUG] Single image task {task_id}: Warning - could not load preset defaults for '{actual_model_type}': {e}")
                    dprint(f"Single image task {task_id}: Warning - could not load preset defaults for '{actual_model_type}': {e}")
                    preset_defaults = {"num_inference_steps": 30, "guidance_scale": 5.0, "flow_shift": 3.0}
                

                # Use the new flexible keyword-style API
                use_causvid = task_params_from_db.get("use_causvid_lora", False)
                use_lighti2x = task_params_from_db.get("use_lighti2x_lora", False)

                # Priority: task params > preset defaults (no automatic LoRA optimizations)
                preset_steps = preset_defaults.get("num_inference_steps", 30)
                preset_guidance = preset_defaults.get("guidance_scale", 5.0) 
                preset_flow = preset_defaults.get("flow_shift", 3.0)
                
                dprint(f"Single image task {task_id}: Using preset/task values without automatic LoRA optimizations")

                # Parameter resolution: task params take priority over preset defaults
                num_inference_steps = (
                    task_params_from_db.get("steps")
                    or task_params_from_db.get("num_inference_steps")
                    or preset_steps
                )
                default_guidance = task_params_from_db.get("guidance_scale", preset_guidance)
                default_flow_shift = task_params_from_db.get("flow_shift", preset_flow)
                
                dprint(f"Single image task {task_id}: Final parameter resolution:")
                dprint(f"  num_inference_steps: {num_inference_steps} (preset: {preset_steps})")
                dprint(f"  guidance_scale: {default_guidance} (preset: {preset_guidance})")
                dprint(f"  flow_shift: {default_flow_shift} (preset: {preset_flow})")
                dprint(f"  task_params_from_db keys: {list(task_params_from_db.keys())}")
                dprint(f"  task_params_from_db.flow_shift: {task_params_from_db.get('flow_shift', 'NOT FOUND')}")
                
                generation_success, video_path_generated = generate_single_video(
                    wgp_mod=wgp_mod,
                    task_id=f"{task_id}_wgp_internal",
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    resolution=resolution,
                    video_length=1,  # Single frame
                    seed=seed,
                    model_filename=model_filename_for_task,
                    model_name=actual_model_type,  # Pass original model name for preset detection
                    use_causvid_lora=use_causvid,
                    use_lighti2x_lora=use_lighti2x,
                    apply_reward_lora=apply_reward_lora,
                    additional_loras=processed_additional_loras,
                    image_refs=image_refs_paths,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=default_guidance,
                    flow_shift=default_flow_shift,
                    cfg_star_switch=task_params_from_db.get("cfg_star_switch", 0),
                    cfg_zero_step=task_params_from_db.get("cfg_zero_step", -1),
                    prompt_enhancer=task_params_from_db.get("prompt_enhancer_mode", ""),
                    dprint=dprint
                )
                
                if not generation_success or not video_path_generated:
                    error_msg = f"Single image task {task_id}: WGP generation failed."
                    print(f"[ERROR] {error_msg}")
                    return False, error_msg
                
                # Convert video path to Path object if it's a string
                video_path_obj = Path(video_path_generated)
                if not video_path_obj.exists():
                    error_msg = f"Single image task {task_id}: WGP generation failed - no output video found at {video_path_generated}"
                    print(f"[ERROR] {error_msg}")
                    return False, error_msg
                
                print(f"[Single Image {task_id}] Extracting frame from generated video...")
                
                # Extract the first (and only) frame using cv2
                import cv2
                cap = cv2.VideoCapture(str(video_path_obj))
                try:
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret:
                            # Save the frame as PNG to the final location
                            success = cv2.imwrite(str(local_output_path), frame)
                            if success and local_output_path.exists():
                                print(f"[Single Image {task_id}] Successfully saved image to: {local_output_path}")

                                # Handle Supabase upload (if configured) and get final location for DB
                                final_db_location = upload_and_get_final_output_location(
                                    local_file_path=local_output_path,
                                    supabase_object_name=output_filename, # Pass only the filename
                                    initial_db_location=initial_db_output_location,
                                    dprint=dprint
                                )

                                return True, final_db_location
                            else:
                                error_msg = f"Single image task {task_id}: Failed to save extracted frame to {local_output_path}"
                                print(f"[ERROR] {error_msg}")
                                return False, error_msg
                        else:
                            error_msg = f"Single image task {task_id}: Failed to read frame from generated video"
                            print(f"[ERROR] {error_msg}")
                            return False, error_msg
                    else:
                        error_msg = f"Single image task {task_id}: Failed to open generated video {video_path_obj}"
                        print(f"[ERROR] {error_msg}")
                        return False, error_msg
                finally:
                    cap.release()
                    
            finally:
                # Restore original WGP save path
                wgp_mod.save_path = original_save_path
    
    except Exception as e:
        error_msg = f"Single image task {task_id}: Unexpected error: {e}"
        print(f"[ERROR] {error_msg}")
        traceback.print_exc()
        return False, error_msg 