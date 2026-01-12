import os
import sys
import json
import traceback
from pathlib import Path
import numpy as np

from source.logging_utils import headless_logger
from source.model_handlers.qwen_handler import QwenHandler
from source.common_utils import extract_orchestrator_parameters
from headless_model_management import GenerationTask
from source.worker_utils import dprint

def parse_phase_config(phase_config: dict, num_inference_steps: int, task_id: str = "unknown", model_name: str = None, debug_mode: bool = False) -> dict:
    """
    Parse phase_config override block and return computed parameters.
    """
    # Import the timestep_transform function from phase_distribution_simulator
    # ... (Logic from original parse_phase_config) ...
    # Since we are moving this, we need to be careful about imports.
    # We can duplicate the inline fallback for timestep_transform.
    
    def timestep_transform(t: float, shift: float = 5.0, num_timesteps: int = 1000) -> float:
        t = t / num_timesteps
        new_t = shift * t / (1 + (shift - 1) * t)
        new_t = new_t * num_timesteps
        return new_t

    # Validation with auto-correction
    num_phases = phase_config.get("num_phases", 3)
    steps_per_phase = phase_config.get("steps_per_phase", [2, 2, 2])
    flow_shift = phase_config.get("flow_shift", 5.0)
    phases_config = phase_config.get("phases", [])

    # Auto-correct num_phases
    if len(steps_per_phase) != num_phases or len(phases_config) != num_phases:
        inferred_phases = len(steps_per_phase)
        if len(phases_config) != inferred_phases:
            inferred_phases = len(phases_config)
            headless_logger.warning(
                f"phase_config mismatch: num_phases={num_phases}, steps_per_phase has {len(steps_per_phase)} values, "
                f"phases array has {len(phases_config)} entries. Using phases array length: {inferred_phases}",
                task_id=task_id
            )
        else:
            headless_logger.warning(
                f"phase_config mismatch: num_phases={num_phases} but steps_per_phase has {len(steps_per_phase)} values. "
                f"Auto-correcting num_phases to {inferred_phases}",
                task_id=task_id
            )
        num_phases = inferred_phases

    if num_phases not in [2, 3]:
        raise ValueError(f"Task {task_id}: num_phases must be 2 or 3, got {num_phases}")

    total_steps = sum(steps_per_phase)
    if total_steps != num_inference_steps:
        raise ValueError(f"Task {task_id}: steps_per_phase {steps_per_phase} sum to {total_steps}, but num_inference_steps is {num_inference_steps}")

    phases_config = phase_config.get("phases", [])
    if len(phases_config) != num_phases:
        raise ValueError(f"Task {task_id}: Expected {num_phases} phase configs, got {len(phases_config)}")

    # Generate timesteps
    sample_solver = phase_config.get("sample_solver", "euler")

    if sample_solver == "causvid":
        headless_logger.warning(f"phase_config with causvid solver is not recommended - causvid uses hardcoded timesteps", task_id=task_id)

    if sample_solver == "unipc" or sample_solver == "":
        sigma_max = 1.0
        sigma_min = 0.001
        sigmas = list(np.linspace(sigma_max, sigma_min, num_inference_steps + 1, dtype=np.float32))[:-1]
        sigmas = [flow_shift * s / (1 + (flow_shift - 1) * s) for s in sigmas]
        timesteps = [s * 1000 for s in sigmas]
    elif sample_solver in ["dpm++", "dpm++_sde"]:
        sigmas = list(np.linspace(1, 0, num_inference_steps + 1, dtype=np.float32))[:num_inference_steps]
        sigmas = [flow_shift * s / (1 + (flow_shift - 1) * s) for s in sigmas]
        timesteps = [s * 1000 for s in sigmas]
    elif sample_solver == "euler":
        timesteps = list(np.linspace(1000, 1, num_inference_steps, dtype=np.float32))
        timesteps.append(0.)
        timesteps = [timestep_transform(t, shift=flow_shift, num_timesteps=1000) for t in timesteps][:-1]
    else:
        timesteps = list(np.linspace(1000, 1, num_inference_steps, dtype=np.float32))

    headless_logger.debug(f"Generated timesteps for phase_config: {[f'{t:.1f}' for t in timesteps]}", task_id=task_id)

    # Calculate switch thresholds
    switch_step_1 = steps_per_phase[0]
    switch_threshold = None
    switch_threshold2 = None

    if switch_step_1 < num_inference_steps:
        last_phase1_t = timesteps[switch_step_1 - 1]
        first_phase2_t = timesteps[switch_step_1]
        switch_threshold = float((last_phase1_t + first_phase2_t) / 2)

    if num_phases >= 3:
        switch_step_2 = steps_per_phase[0] + steps_per_phase[1]
        if switch_step_2 < num_inference_steps:
            last_phase2_t = timesteps[switch_step_2 - 1]
            first_phase3_t = timesteps[switch_step_2]
            switch_threshold2 = float((last_phase2_t + first_phase3_t) / 2)

    result = {
        "guidance_phases": num_phases,
        "switch_threshold": switch_threshold,
        "switch_threshold2": switch_threshold2,
        "flow_shift": flow_shift,
        "sample_solver": sample_solver,
        "model_switch_phase": phase_config.get("model_switch_phase", 2),
    }

    if num_phases >= 1:
        result["guidance_scale"] = phases_config[0].get("guidance_scale", 3.0)
    if num_phases >= 2:
        result["guidance2_scale"] = phases_config[1].get("guidance_scale", 1.0)
    if num_phases >= 3:
        result["guidance3_scale"] = phases_config[2].get("guidance_scale", 1.0)

    # Process LoRAs
    all_lora_urls = []
    seen_urls = set()
    for phase_cfg in phases_config:
        for lora in phase_cfg.get("loras", []):
            url = lora["url"]
            if not url or not url.strip():
                continue
            if url not in seen_urls:
                all_lora_urls.append(url)
                seen_urls.add(url)

    lora_multipliers = []
    additional_loras = {}

    for lora_url in all_lora_urls:
        phase_mults = []
        for phase_idx, phase_cfg in enumerate(phases_config):
            lora_in_phase = None
            for lora in phase_cfg.get("loras", []):
                if lora["url"] == lora_url:
                    lora_in_phase = lora
                    break
            
            if lora_in_phase:
                multiplier_str = str(lora_in_phase["multiplier"])
                if "," in multiplier_str:
                    values = multiplier_str.split(",")
                    expected_count = steps_per_phase[phase_idx]
                    if len(values) != expected_count:
                        raise ValueError(f"Task {task_id}: Phase {phase_idx+1} LoRA multiplier has {len(values)} values, but phase has {expected_count} steps")
                    phase_mults.append(multiplier_str)
                else:
                    phase_mults.append(multiplier_str)
            else:
                phase_mults.append("0")

        if num_phases == 2:
            multiplier_string = f"{phase_mults[0]};{phase_mults[1]}"
        else:
            multiplier_string = f"{phase_mults[0]};{phase_mults[1]};{phase_mults[2]}"
        
        lora_multipliers.append(multiplier_string)

        try:
            first_val = float(phase_mults[0].split(",")[0])
            additional_loras[lora_url] = first_val
        except (ValueError, IndexError):
            additional_loras[lora_url] = 1.0

    # Prepare patch config (simplified for extracted module)
    if model_name:
        # Note: We need to locate the Wan2GP defaults directory.
        # Assuming this is running from source/task_conversion.py, we need to go up one level to find Wan2GP
        # But Wan2GP is usually at root level or alongside worker.py.
        # Let's assume standard structure relative to this file.
        # This file is in source/, so parent is root.
        wan_dir = Path(__file__).parent.parent / "Wan2GP"
        defaults_dir = wan_dir / "defaults"
        
        original_config_path = defaults_dir / f"{model_name}.json"
        if original_config_path.exists():
            try:
                with open(original_config_path, 'r') as f:
                    original_config = json.load(f)
                
                import copy
                temp_config = copy.deepcopy(original_config)
                temp_config["guidance_phases"] = num_phases
                
                guidance_scales = [p.get("guidance_scale", 1.0) for p in phases_config]
                if len(guidance_scales) >= 1: temp_config["guidance_scale"] = guidance_scales[0]
                if len(guidance_scales) >= 2: temp_config["guidance2_scale"] = guidance_scales[1]
                if len(guidance_scales) >= 3: temp_config["guidance3_scale"] = guidance_scales[2]

                temp_config["flow_shift"] = phase_config.get("flow_shift", temp_config.get("flow_shift", 5.0))
                temp_config["sample_solver"] = phase_config.get("sample_solver", temp_config.get("sample_solver", "euler"))

                if "model" in temp_config:
                    temp_config["model"]["loras"] = []
                    temp_config["model"]["loras_multipliers"] = []
                    # Enable SVI mode if phase_config has SVI LoRAs or svi2pro flag
                    svi_loras_present = any(
                        "SVI" in lora_url or "svi" in lora_url.lower() 
                        for lora_url in all_lora_urls
                    )
                    if svi_loras_present or phase_config.get("svi2pro", False):
                        temp_config["model"]["svi2pro"] = True
                        headless_logger.debug(f"[PATCH_CONFIG] Added svi2pro=True to model definition (SVI LoRAs detected)", task_id=task_id)

                result["_patch_config"] = temp_config
                result["_patch_model_name"] = model_name
            except Exception as e:
                headless_logger.warning(f"Could not prepare phase_config patch: {e}", task_id=task_id)

    # CRITICAL: lora_names must contain the URLs so LoRAConfig can:
    # 1. Detect URLs and mark them as PENDING for download
    # 2. Associate each URL with its corresponding phase-format multiplier
    # 3. Download via _download_lora_from_url() in the queue
    # Without this, activated_loras ends up empty and the multipliers aren't associated
    result["lora_names"] = all_lora_urls
    result["lora_multipliers"] = lora_multipliers
    result["additional_loras"] = additional_loras

    return result


def db_task_to_generation_task(db_task_params: dict, task_id: str, task_type: str, wan2gp_path: str, debug_mode: bool = False) -> GenerationTask:
    """
    Convert a database task row to a GenerationTask object for the queue system.
    """
    headless_logger.debug(f"Converting DB task to GenerationTask", task_id=task_id)

    prompt = db_task_params.get("prompt", "")

    # For img2img tasks, empty prompt is acceptable (will use minimal changes)
    # Provide a minimal default prompt to avoid errors
    img2img_task_types = {"z_image_turbo_i2i", "qwen_image_edit", "qwen_image_style", "image_inpaint"}
    if not prompt:
        if task_type in img2img_task_types:
            prompt = " "  # Minimal prompt for img2img
            headless_logger.debug(f"Task {task_id}: Using minimal prompt for img2img task", task_id=task_id)
        else:
            raise ValueError(f"Task {task_id}: prompt is required")
    
    model = db_task_params.get("model")
    if not model:
        task_type_to_model = {
            "generate_video": "t2v",
            "vace": "vace_14B_cocktail_2_2",
            "vace_21": "vace_14B",
            "vace_22": "vace_14B_cocktail_2_2",
            "wan_2_2_t2i": "t2v_2_2",
            "flux": "flux",
            "t2v": "t2v",
            "t2v_22": "t2v_2_2",
            "i2v": "i2v_14B",
            "i2v_22": "i2v_2_2",
            "hunyuan": "hunyuan",
            "ltxv": "ltxv_13B",
            "join_clips_segment": "wan_2_2_vace_lightning_baseline_2_2_2",
            "inpaint_frames": "wan_2_2_vace_lightning_baseline_2_2_2",
            "qwen_image_edit": "qwen_image_edit_20B",
            "qwen_image_hires": "qwen_image_edit_20B",
            "qwen_image_style": "qwen_image_edit_20B",
            "image_inpaint": "qwen_image_edit_20B",
            "annotated_image_edit": "qwen_image_edit_20B",
            # Text-to-image tasks
            "qwen_image": "qwen_image_edit_20B",
            "qwen_image_2512": "qwen_image_2512_20B",
            "z_image_turbo": "z_image",
            # Image-to-image tasks
            "z_image_turbo_i2i": "z_image_img2img"
        }
        model = task_type_to_model.get(task_type, "t2v")
    
    generation_params = {}
    
    param_whitelist = {
        "negative_prompt", "resolution", "video_length", "num_inference_steps",
        "guidance_scale", "seed", "embedded_guidance_scale", "flow_shift",
        "audio_guidance_scale", "repeat_generation", "multi_images_gen_type",
        "guidance2_scale", "guidance3_scale", "guidance_phases", "switch_threshold", "switch_threshold2", "model_switch_phase",
        "video_guide", "video_mask",
        "video_prompt_type", "control_net_weight", "control_net_weight2",
        "keep_frames_video_guide", "video_guide_outpainting", "mask_expand",
        "image_prompt_type", "image_start", "image_end", "image_refs",
        "frames_positions", "image_guide", "image_mask",
        "model_mode", "video_source", "keep_frames_video_source",
        "audio_guide", "audio_guide2", "audio_source", "audio_prompt_type", "speakers_locations",
        "activated_loras", "loras_multipliers", "additional_loras", "loras",
        "tea_cache_setting", "tea_cache_start_step_perc", "RIFLEx_setting",
        "slg_switch", "slg_layers", "slg_start_perc", "slg_end_perc",
        "cfg_star_switch", "cfg_zero_step", "prompt_enhancer",
        # Hires fix parameters
        "hires_scale", "hires_steps", "hires_denoise", "hires_upscale_method", "lightning_lora_strength",
        "sliding_window_size", "sliding_window_overlap", "sliding_window_overlap_noise",
        "sliding_window_discard_last_frames", "latent_noise_mask_strength",
        "vid2vid_init_video", "vid2vid_init_strength",
        "remove_background_images_ref", "temporal_upsampling", "spatial_upsampling",
        "film_grain_intensity", "film_grain_saturation",
        "image_refs_relative_size",
        "output_dir", "custom_output_dir",
        "override_profile",
        "image", "image_url", "mask_url",
        "style_reference_image", "subject_reference_image",
        "style_reference_strength", "subject_strength",
        "subject_description", "in_this_scene",
        "output_format", "enable_base64_output", "enable_sync_mode",
        # Uni3C motion guidance parameters
        "use_uni3c", "uni3c_guide_video", "uni3c_strength",
        "uni3c_start_percent", "uni3c_end_percent",
        "uni3c_keep_on_gpu", "uni3c_frame_policy",
        # Image-to-image parameters
        "denoising_strength",
    }
    
    for param in param_whitelist:
        if param in db_task_params:
            generation_params[param] = db_task_params[param]
    
    # Layer 1 Uni3C logging - detect whitelist failures early
    if "use_uni3c" in generation_params:
        headless_logger.info(
            f"[UNI3C] Task {task_id}: use_uni3c={generation_params.get('use_uni3c')}, "
            f"guide_video={generation_params.get('uni3c_guide_video', 'NOT_SET')}, "
            f"strength={generation_params.get('uni3c_strength', 'NOT_SET')}"
        )
    elif db_task_params.get("use_uni3c"):
        # CRITICAL: Detect when whitelist is missing the param
        headless_logger.warning(
            f"[UNI3C] Task {task_id}: ⚠️ use_uni3c was in db_task_params but NOT in generation_params! "
            f"Check param_whitelist in task_conversion.py"
        )
    
    # Extract orchestrator parameters
    # We use a lambda for dprint to match signature expected by extract_orchestrator_parameters if needed,
    # or just pass the function from worker_utils
    def dprint_wrapper(msg):
        dprint(msg, task_id=task_id, debug_mode=debug_mode)

    extracted_params = extract_orchestrator_parameters(db_task_params, task_id, dprint_wrapper)

    if "phase_config" in extracted_params:
        db_task_params["phase_config"] = extracted_params["phase_config"]

    # Copy additional_loras from extracted params if not already in generation_params
    if "additional_loras" in extracted_params and "additional_loras" not in generation_params:
        generation_params["additional_loras"] = extracted_params["additional_loras"]
    
    if "steps" in db_task_params and "num_inference_steps" not in generation_params:
        generation_params["num_inference_steps"] = db_task_params["steps"]
    
    # Note: LoRA resolution is handled centrally in HeadlessTaskQueue._convert_to_wgp_task()
    # via TaskConfig/LoRAConfig which detects URLs and downloads them automatically

    # Qwen Task Handlers
    qwen_handler = QwenHandler(
        wan_root=wan2gp_path,
        task_id=task_id,
        dprint_func=dprint_wrapper
    )

    if task_type == "qwen_image_edit":
        qwen_handler.handle_qwen_image_edit(db_task_params, generation_params)
    elif task_type == "qwen_image_hires":
        qwen_handler.handle_qwen_image_hires(db_task_params, generation_params)
    elif task_type == "image_inpaint":
        qwen_handler.handle_image_inpaint(db_task_params, generation_params)
    elif task_type == "annotated_image_edit":
        qwen_handler.handle_annotated_image_edit(db_task_params, generation_params)
    elif task_type == "qwen_image_style":
        qwen_handler.handle_qwen_image_style(db_task_params, generation_params)
        prompt = generation_params.get("prompt", prompt)
        model = "qwen_image_edit_20B"
    elif task_type == "qwen_image":
        qwen_handler.handle_qwen_image(db_task_params, generation_params)
        model = "qwen_image_edit_20B"
    elif task_type == "qwen_image_2512":
        qwen_handler.handle_qwen_image_2512(db_task_params, generation_params)
        model = "qwen_image_2512_20B"
    elif task_type == "z_image_turbo":
        # Z-Image turbo - fast text-to-image generation
        generation_params.setdefault("video_prompt_type", "")  # No input image
        generation_params.setdefault("video_length", 1)  # Single image output
        generation_params.setdefault("guidance_scale", 0)  # Z-Image uses guidance_scale=0
        generation_params.setdefault("num_inference_steps", int(db_task_params.get("num_inference_steps", 8)))

        # Resolution handling
        if "resolution" in db_task_params:
            generation_params["resolution"] = db_task_params["resolution"]

        # Override model to use Z-Image (user might pass "z-image" with hyphen)
        model = "z_image"

    elif task_type == "z_image_turbo_i2i":
        # Z-Image turbo img2img - fast image-to-image generation
        import tempfile
        from PIL import Image

        # Get image URL
        image_url = db_task_params.get("image") or db_task_params.get("image_url")
        if not image_url:
            raise ValueError(f"Task {task_id}: 'image' or 'image_url' required for z_image_turbo_i2i")

        # Download image to local file (required for WGP)
        local_image_path = None
        try:
            import requests
            import os
            headless_logger.debug(f"Downloading image for img2img: {image_url}", task_id=task_id)
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()

            # Save to temp file and keep it for WGP to use
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                tmp_file.write(response.content)
                local_image_path = tmp_file.name

            headless_logger.info(f"[Z_IMAGE_I2I] Downloaded image to: {local_image_path}", task_id=task_id)

            # Handle resolution - auto-detect from image or use provided
            image_size = db_task_params.get("image_size", "auto")
            if image_size == "auto" or not db_task_params.get("resolution"):
                with Image.open(local_image_path) as img:
                    width, height = img.size

                    # Scale to approximately 1 megapixel (1024x1024) while preserving aspect ratio
                    target_pixels = 1024 * 1024  # ~1 megapixel
                    current_pixels = width * height

                    if current_pixels > 0:
                        # Calculate scale factor to reach target pixels
                        import math
                        scale = math.sqrt(target_pixels / current_pixels)

                        # Apply scale to both dimensions
                        scaled_width = int(round(width * scale))
                        scaled_height = int(round(height * scale))

                        # Round to nearest multiple of 8 (Z-Image requirement)
                        scaled_width = (scaled_width // 8) * 8
                        scaled_height = (scaled_height // 8) * 8

                        # Ensure minimum size
                        scaled_width = max(8, scaled_width)
                        scaled_height = max(8, scaled_height)

                        generation_params["resolution"] = f"{scaled_width}x{scaled_height}"
                        headless_logger.info(
                            f"Scaled resolution: {width}x{height} ({current_pixels:,} px) → "
                            f"{scaled_width}x{scaled_height} ({scaled_width*scaled_height:,} px, scale={scale:.3f})",
                            task_id=task_id
                        )
                    else:
                        generation_params["resolution"] = "1024x1024"
            elif "resolution" in db_task_params:
                generation_params["resolution"] = db_task_params["resolution"]
            else:
                generation_params["resolution"] = "1024x1024"

        except Exception as e:
            # If download fails, clean up and raise error
            if local_image_path:
                import os
                try:
                    os.unlink(local_image_path)
                except:
                    pass
            raise ValueError(f"Task {task_id}: Failed to download image for img2img: {e}")

        # CRITICAL: Pass local file path (not URL) to WGP
        generation_params["image_start"] = local_image_path

        # Set img2img parameters
        generation_params.setdefault("video_prompt_type", "")  # Image input handled via image_start
        generation_params.setdefault("video_length", 1)  # Single image output
        generation_params.setdefault("guidance_scale", 0)  # Z-Image uses guidance_scale=0
        generation_params.setdefault("num_inference_steps", int(db_task_params.get("num_inference_steps", 12)))

        # CRITICAL: Add denoising_strength to generation_params so it gets passed through
        actual_strength = db_task_params.get("denoising_strength") or db_task_params.get("denoise_strength") or db_task_params.get("strength", 0.7)
        generation_params["denoising_strength"] = actual_strength

        headless_logger.info(
            f"[Z_IMAGE_I2I] Setup complete - local_image={local_image_path}, resolution={generation_params['resolution']}, "
            f"strength={actual_strength}, steps={generation_params['num_inference_steps']}",
            task_id=task_id
        )

        # Override model to use Z-Image img2img
        model = "z_image_img2img"

    # Defaults
    essential_defaults = {
        "seed": -1,
        "negative_prompt": "",
    }
    for param, default_value in essential_defaults.items():
        if param not in generation_params:
            generation_params[param] = default_value
    
    # Phase Config Override
    if "phase_config" in db_task_params:
        try:
            steps_per_phase = db_task_params["phase_config"].get("steps_per_phase", [2, 2, 2])
            phase_config_steps = sum(steps_per_phase)

            parsed_phase_config = parse_phase_config(
                phase_config=db_task_params["phase_config"],
                num_inference_steps=phase_config_steps,
                task_id=task_id,
                model_name=model,
                debug_mode=debug_mode
            )

            generation_params["num_inference_steps"] = phase_config_steps

            for key in ["guidance_phases", "switch_threshold", "switch_threshold2",
                       "guidance_scale", "guidance2_scale", "guidance3_scale",
                       "flow_shift", "sample_solver", "model_switch_phase",
                       "lora_names", "lora_multipliers", "additional_loras"]:
                if key in parsed_phase_config and parsed_phase_config[key] is not None:
                    generation_params[key] = parsed_phase_config[key]

            if "lora_names" in parsed_phase_config:
                generation_params["activated_loras"] = parsed_phase_config["lora_names"]
            if "lora_multipliers" in parsed_phase_config:
                generation_params["loras_multipliers"] = " ".join(str(m) for m in parsed_phase_config["lora_multipliers"])

            if "_patch_config" in parsed_phase_config:
                generation_params["_parsed_phase_config"] = parsed_phase_config
                generation_params["_phase_config_model_name"] = model
            
            # Note: LoRA URL resolution is handled centrally in HeadlessTaskQueue._convert_to_wgp_task()
            # URLs in activated_loras are detected by LoRAConfig.from_params() and downloaded there
        except Exception as e:
            raise ValueError(f"Task {task_id}: Invalid phase_config: {e}")

    priority = db_task_params.get("priority", 0)
    if task_type.endswith("_orchestrator"):
        priority = max(priority, 10)
    
    generation_task = GenerationTask(
        id=task_id,
        model=model,
        prompt=prompt,
        parameters=generation_params,
        priority=priority
    )
    
    return generation_task


