"""Utility helpers around Wan2GP.wgp API used by headless tasks."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Tuple, Dict, Optional, Union, Callable
from PIL import Image


def _ensure_lora_in_lists(lora_name: str, multiplier: str, activated_loras: list, loras_multipliers: Union[list, str]) -> Tuple[list, Union[list, str]]:
    """
    Helper function to ensure a LoRA is present in activated_loras with the correct multiplier.
    Returns updated (activated_loras, loras_multipliers) tuple.
    """
    if lora_name not in activated_loras:
        activated_loras.insert(0, lora_name)
        if isinstance(loras_multipliers, list):
            loras_multipliers.insert(0, multiplier)
        elif isinstance(loras_multipliers, str):
            mult_list = [m.strip() for m in loras_multipliers.split(",") if m.strip()] if loras_multipliers else []
            mult_list.insert(0, multiplier)
            loras_multipliers = ",".join(mult_list)
        else:
            loras_multipliers = [multiplier]
    return activated_loras, loras_multipliers


def _set_param_if_different(params: dict, key: str, target_value: Union[int, float], task_id: str, lora_name: str, dprint: Callable):
    """
    Helper function to set a parameter value if it differs from the target, with logging.
    """
    current_value = params.get(key)
    if isinstance(target_value, int):
        try:
            current_int = int(current_value) if current_value is not None else 0
        except (TypeError, ValueError):
            current_int = 0
        if current_int != target_value:
            dprint(f"{task_id}: {lora_name} active – overriding {key} {current_int} → {target_value}")
            params[key] = target_value
    elif isinstance(target_value, float):
        current_float = float(current_value) if current_value is not None else 0.0
        if current_float != target_value:
            dprint(f"{task_id}: {lora_name} active – setting {key} → {target_value}")
            params[key] = target_value


def _normalize_loras_multipliers_format(loras_multipliers: Union[list, str]) -> str:
    """
    Helper function to normalize loras_multipliers to string format expected by WGP.
    """
    if isinstance(loras_multipliers, list):
        return ",".join(loras_multipliers) if loras_multipliers else ""
    else:
        return loras_multipliers if loras_multipliers else ""


def generate_single_video(
    wgp_mod,
    task_id: str,
    prompt: str,
    negative_prompt: str = "",
    resolution: str = "512x512",
    video_length: int = 81,
    seed: int = 12345,
    num_inference_steps: int = 30,
    guidance_scale: float = 5.0,
    flow_shift: float = 3.0,
    model_filename: str = None,
    model_name: str = None,  # ← NEW: Original model name for proper type detection
    video_guide: str = None,
    video_mask: str = None,
    image_refs: list = None,
    use_causvid_lora: bool = False,
    use_lighti2x_lora: bool = False,
    apply_reward_lora: bool = False,
    additional_loras: dict = None,
    video_prompt_type: str = "T",
    dprint = None,
    **kwargs
) -> tuple[bool, str]:
    """
    Centralized wrapper for WGP video generation with comprehensive debugging.
    Returns (success_bool, output_path_or_error_message)
    """
    if dprint is None:
        dprint = lambda x: print(f"[DEBUG] {x}")
    
    print(f"[WGP_GENERATION_DEBUG] Starting generation for task {task_id}")
    print(f"[WGP_GENERATION_DEBUG] Parameters:")
    print(f"[WGP_GENERATION_DEBUG]   prompt: {prompt}")
    print(f"[WGP_GENERATION_DEBUG]   resolution: {resolution}")
    print(f"[WGP_GENERATION_DEBUG]   video_length: {video_length}")
    print(f"[WGP_GENERATION_DEBUG]   seed: {seed}")
    print(f"[WGP_GENERATION_DEBUG]   num_inference_steps: {num_inference_steps}")
    print(f"[WGP_GENERATION_DEBUG]   guidance_scale: {guidance_scale}")
    print(f"[WGP_GENERATION_DEBUG]   flow_shift: {flow_shift}")
    print(f"[WGP_GENERATION_DEBUG]   use_causvid_lora: {use_causvid_lora}")
    print(f"[WGP_GENERATION_DEBUG]   use_lighti2x_lora: {use_lighti2x_lora}")
    print(f"[WGP_GENERATION_DEBUG]   video_guide: {video_guide}")
    print(f"[WGP_GENERATION_DEBUG]   video_mask: {video_mask}")
    print(f"[WGP_GENERATION_DEBUG]   video_prompt_type: {video_prompt_type}")
    
    try:
        # Analyze input guide video if provided
        if video_guide and Path(video_guide).exists():
            from .common_utils import get_video_frame_count_and_fps
            guide_frames, guide_fps = get_video_frame_count_and_fps(video_guide)
            print(f"[WGP_GENERATION_DEBUG] Guide video analysis:")
            print(f"[WGP_GENERATION_DEBUG]   Path: {video_guide}")
            print(f"[WGP_GENERATION_DEBUG]   Frames: {guide_frames}")
            print(f"[WGP_GENERATION_DEBUG]   FPS: {guide_fps}")
            if guide_frames != video_length:
                print(f"[WGP_GENERATION_DEBUG]   ⚠️  GUIDE/TARGET MISMATCH! Guide has {guide_frames} frames, target is {video_length}")

        # Analyze input mask video if provided
        if video_mask and Path(video_mask).exists():
            from .common_utils import get_video_frame_count_and_fps
            mask_frames, mask_fps = get_video_frame_count_and_fps(video_mask)
            print(f"[WGP_GENERATION_DEBUG] Mask video analysis:")
            print(f"[WGP_GENERATION_DEBUG]   Path: {video_mask}")
            print(f"[WGP_GENERATION_DEBUG]   Frames: {mask_frames}")
            print(f"[WGP_GENERATION_DEBUG]   FPS: {mask_fps}")
            if mask_frames != video_length:
                print(f"[WGP_GENERATION_DEBUG]   ⚠️  MASK/TARGET MISMATCH! Mask has {mask_frames} frames, target is {video_length}")
        
        # Build task state for WGP
        from .common_utils import build_task_state
        
        task_params_dict = {
                "task_id": task_id,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "resolution": resolution,
            "video_length": video_length,
            "frames": video_length,  # Alternative key
            "seed": seed,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "flow_shift": flow_shift,
            "video_guide_path": video_guide,
            "video_mask": video_mask,
            "image_refs_paths": image_refs,
            "video_prompt_type": video_prompt_type,
            "use_causvid_lora": use_causvid_lora,
            "use_lighti2x_lora": use_lighti2x_lora,
            "apply_reward_lora": apply_reward_lora,
            "processed_additional_loras": additional_loras or {},
            **kwargs
        }
        
        print(f"[WGP_GENERATION_DEBUG] Built task_params_dict with {len(task_params_dict)} parameters")
        
        # Get LoRA directory and setup using centralized helper
        from .common_utils import get_lora_dir_from_filename
        lora_dir_for_active_model = get_lora_dir_from_filename(wgp_mod, model_filename)
        # Get model type for setup_loras (it needs model_type, not model_filename)
        # [VACE_FIX] Use original model_name for proper type detection, fallback to filename-based detection
        effective_model_name = model_name if model_name else model_filename
        model_type = effective_model_name if effective_model_name and not effective_model_name.endswith('.safetensors') else (wgp_mod.get_model_type(model_filename) if model_filename else "t2v")
        print(f"[WGP_VACE_DEBUG] Model type determination: effective_model_name='{effective_model_name}' → model_type='{model_type}'")
        all_loras_for_active_model, _, _, _, _, _, _ = wgp_mod.setup_loras(
            model_type, None, lora_dir_for_active_model, "", None
        )
        
        print(f"[WGP_GENERATION_DEBUG] LoRA setup complete. Available LoRAs: {len(all_loras_for_active_model) if all_loras_for_active_model else 0}")
        
        # Build state and UI params  
        # [VACE_FIX] Pass the correct model_type to ensure proper model definition lookup
        state, ui_params = build_task_state(
            wgp_mod, 
            model_filename, 
            task_params_dict, 
            all_loras_for_active_model, 
            None,  # image_download_dir
            apply_reward_lora=apply_reward_lora,
            model_type_override=model_type  # ← Use the corrected model type
        )
        
        # [VACE_FIX] Store the original model type for VACE detection 
        # while using the resolved base type for WGP configuration
        if model_name and model_name in ["vace_14B", "vace_1.3B", "vace_multitalk_14B"]:
            ui_params["original_model_type"] = model_name  # Keep for VACE detection
            print(f"[WGP_VACE_DEBUG] Preserved original_model_type='{model_name}' for VACE detection")
        
        print(f"[WGP_GENERATION_DEBUG] State and UI params built")
        print(f"[WGP_GENERATION_DEBUG] Final ui_params video_length: {ui_params.get('video_length', 'NOT_SET')}")
        print(f"[WGP_GENERATION_DEBUG] Final ui_params frames: {ui_params.get('frames', 'NOT_SET')}")
        
        # Create temporary output directory
        import tempfile
        temp_output_dir = tempfile.mkdtemp(prefix=f"wgp_single_{task_id}_")
        print(f"[WGP_GENERATION_DEBUG] Using temporary output directory: {temp_output_dir}")
        
        # Save original path and set temporary
        original_wgp_save_path = wgp_mod.save_path
        wgp_mod.save_path = str(temp_output_dir)
        
        try:
            # Create task and send_cmd objects
            gen_task_placeholder = {
                "id": 1, 
                "prompt": ui_params.get("prompt"), 
                "params": {
                    "model_filename_from_gui_state": model_filename, 
                    "model": kwargs.get("model", "t2v")
                }
            }
            
            def send_cmd_debug(cmd, data=None):
                if cmd == "progress":
                    if isinstance(data, list) and len(data) >= 2:
                        prog, txt = data[0], data[1]
                        if isinstance(prog, tuple) and len(prog) == 2:
                            step, total = prog
                            print(f"[WGP_PROGRESS] {step}/{total} – {txt}")
                        else:
                            print(f"[WGP_PROGRESS] {txt}")
                elif cmd == "status":
                    print(f"[WGP_STATUS] {data}")
                elif cmd == "info":
                    print(f"[WGP_INFO] {data}")
                elif cmd == "error":
                    print(f"[WGP_ERROR] {data}")
                    raise RuntimeError(f"WGP error for {task_id}: {data}")
                elif cmd == "output":
                    print(f"[WGP_OUTPUT] Video generation completed")
            
            print(f"[WGP_GENERATION_DEBUG] Calling wgp_mod.generate_video...")
            print(f"[WGP_GENERATION_DEBUG] Final parameters being passed:")
            print(f"[WGP_GENERATION_DEBUG]   video_length: {ui_params.get('video_length')}")
            print(f"[WGP_GENERATION_DEBUG]   resolution: {ui_params.get('resolution')}")
            print(f"[WGP_GENERATION_DEBUG]   seed: {ui_params.get('seed')}")
            print(f"[WGP_GENERATION_DEBUG]   num_inference_steps: {ui_params.get('num_inference_steps')}")
            print(f"[WGP_VACE_DEBUG] === VACE-Specific Parameters ===")
            print(f"[WGP_VACE_DEBUG]   video_prompt_type: '{ui_params.get('video_prompt_type')}'")
            print(f"[WGP_VACE_DEBUG]   video_guide: {ui_params.get('video_guide')}")
            print(f"[WGP_VACE_DEBUG]   video_mask: {ui_params.get('video_mask')}")
            print(f"[WGP_VACE_DEBUG]   control_net_weight: {ui_params.get('control_net_weight', 1.0)}")
            print(f"[WGP_VACE_DEBUG]   control_net_weight2: {ui_params.get('control_net_weight2', 1.0)}")
            print(f"[WGP_VACE_DEBUG]   denoising_strength: {ui_params.get('denoising_strength', 1.0)}")
            print(f"[WGP_VACE_DEBUG]   image_refs: {ui_params.get('image_refs')}")
            if ui_params.get('video_guide'):
                import os
                vg_path = ui_params.get('video_guide')
                print(f"[WGP_VACE_DEBUG]   video_guide exists: {os.path.exists(vg_path) if vg_path else 'None'}")
                if vg_path and os.path.exists(vg_path):
                    print(f"[WGP_VACE_DEBUG]   video_guide size: {os.path.getsize(vg_path)} bytes")
            if ui_params.get('video_mask'):
                import os
                vm_path = ui_params.get('video_mask')
                print(f"[WGP_VACE_DEBUG]   video_mask exists: {os.path.exists(vm_path) if vm_path else 'None'}")
                if vm_path and os.path.exists(vm_path):
                    print(f"[WGP_VACE_DEBUG]   video_mask size: {os.path.getsize(vm_path)} bytes")
            
            # Call the actual WGP generation with dynamic parameter filtering
            import inspect
            
            # Get the actual parameters that generate_video accepts
            generate_video_signature = inspect.signature(wgp_mod.generate_video)
            supported_params = set(generate_video_signature.parameters.keys())
            
            # Core parameters that we always want to pass
            call_params = {
                "task": gen_task_placeholder,
                "send_cmd": send_cmd_debug,
                "prompt": ui_params["prompt"],
                "negative_prompt": ui_params.get("negative_prompt", ""),
                "resolution": ui_params["resolution"],
                "video_length": ui_params.get("video_length", video_length),
                "seed": ui_params["seed"],
                "num_inference_steps": ui_params.get("num_inference_steps", num_inference_steps),
                "guidance_scale": ui_params.get("guidance_scale", guidance_scale),
                "flow_shift": ui_params.get("flow_shift", flow_shift),
                "video_guide": ui_params.get("video_guide"),
                "video_mask": ui_params.get("video_mask"),
                "image_refs": ui_params.get("image_refs"),
                "video_prompt_type": ui_params.get("video_prompt_type", video_prompt_type),
                "activated_loras": ui_params.get("activated_loras", []),
                "loras_multipliers": ui_params.get("loras_multipliers", ""),
                "state": state,
                "model_filename": model_filename,
                
                # Required parameters that were missing - adding defaults
                "image_mode": ui_params.get("image_mode", 0),
                "batch_size": ui_params.get("batch_size", 1),
                "force_fps": ui_params.get("force_fps", ""),
                "guidance2_scale": ui_params.get("guidance2_scale", 1.0),
                "switch_threshold": ui_params.get("switch_threshold", 0.5),
                "sample_solver": ui_params.get("sample_solver", "euler"),
                "multi_prompts_gen_type": ui_params.get("multi_prompts_gen_type", 0),
                "skip_steps_cache_type": ui_params.get("skip_steps_cache_type", ""),
                "skip_steps_multiplier": ui_params.get("skip_steps_multiplier", 1.0),
                "skip_steps_start_step_perc": ui_params.get("skip_steps_start_step_perc", 0),
                "frames_positions": ui_params.get("frames_positions", []),
                "image_guide": ui_params.get("image_guide"),
                "denoising_strength": ui_params.get("denoising_strength", 1.0),
                "video_guide_outpainting": ui_params.get("video_guide_outpainting", ""),
                "image_mask": ui_params.get("image_mask"),
                "control_net_weight": ui_params.get("control_net_weight", 1.0),
                "control_net_weight2": ui_params.get("control_net_weight2", 1.0),
                "mask_expand": ui_params.get("mask_expand", 0),
                "audio_guide2": ui_params.get("audio_guide2"),
                "audio_source": ui_params.get("audio_source", ""),
                "audio_prompt_type": ui_params.get("audio_prompt_type", "T"),
                "speakers_locations": ui_params.get("speakers_locations", []),
                "sliding_window_color_correction_strength": ui_params.get("sliding_window_color_correction_strength", 0.0),
                "film_grain_intensity": ui_params.get("film_grain_intensity", 0.0),
                "film_grain_saturation": ui_params.get("film_grain_saturation", 1.0),
                "MMAudio_setting": ui_params.get("MMAudio_setting", 0),
                "MMAudio_prompt": ui_params.get("MMAudio_prompt", ""),
                "MMAudio_neg_prompt": ui_params.get("MMAudio_neg_prompt", ""),
                "NAG_scale": ui_params.get("NAG_scale", 1.0),
                "NAG_tau": ui_params.get("NAG_tau", 1.0),
                "NAG_alpha": ui_params.get("NAG_alpha", 0.0),
                "apg_switch": ui_params.get("apg_switch", 0),
                "min_frames_if_references": ui_params.get("min_frames_if_references", 1),
                "model_type": ui_params.get("original_model_type", model_type),  # Use original for VACE detection
                "mode": ui_params.get("mode", "generation"),
            }
            
            # Log model type detection
            detected_model_type = wgp_mod.get_model_type(model_filename) or "t2v"
            print(f"[WGP_VACE_DEBUG]   model_filename: {model_filename}")
            print(f"[WGP_VACE_DEBUG]   detected_model_type: {detected_model_type}")
            
            # [VACE_FIX] This approach won't work cleanly - let's revert to the simpler solution
            # The real issue is that WGP's load_models() expects base_model_type to be the underlying type
            # Let's handle this in the state building instead
            
            # Optional parameters that may not be supported in all WGP versions
            optional_params = {
                "audio_guidance_scale": ui_params.get("audio_guidance_scale", 5.0),
                "embedded_guidance_scale": ui_params.get("embedded_guidance_scale", 6.0),
                "repeat_generation": ui_params.get("repeat_generation", 1),
                "multi_images_gen_type": ui_params.get("multi_images_gen_type", 0),
                "tea_cache_setting": ui_params.get("tea_cache_setting", 0.0),
                "tea_cache_start_step_perc": ui_params.get("tea_cache_start_step_perc", 0),
                "image_prompt_type": ui_params.get("image_prompt_type", "T"),
                "image_start": next((wgp_mod.convert_image(img) for img in ui_params.get("image_start", [])), None),
                "image_end": next((wgp_mod.convert_image(img) for img in ui_params.get("image_end", [])), None),
                "model_mode": ui_params.get("model_mode", 0),
                "video_source": ui_params.get("video_source"),
                "keep_frames_video_source": ui_params.get("keep_frames_video_source", ""),
                "keep_frames_video_guide": ui_params.get("keep_frames_video_guide", ""),
                "audio_guide": ui_params.get("audio_guide"),
                "sliding_window_size": ui_params.get("sliding_window_size", 81),
                "sliding_window_overlap": ui_params.get("sliding_window_overlap", 5),
                "sliding_window_overlap_noise": ui_params.get("sliding_window_overlap_noise", 20),
                "sliding_window_discard_last_frames": ui_params.get("sliding_window_discard_last_frames", 0),
                "remove_background_images_ref": ui_params.get("remove_background_images_ref", False),
                "temporal_upsampling": ui_params.get("temporal_upsampling", ""),
                "spatial_upsampling": ui_params.get("spatial_upsampling", ""),
                "RIFLEx_setting": ui_params.get("RIFLEx_setting", 0),
                "slg_switch": ui_params.get("slg_switch", 0),
                "slg_layers": ui_params.get("slg_layers", [9]),
                "slg_start_perc": ui_params.get("slg_start_perc", 10),
                "slg_end_perc": ui_params.get("slg_end_perc", 90),
                "cfg_star_switch": ui_params.get("cfg_star_switch", 0),
                "cfg_zero_step": ui_params.get("cfg_zero_step", -1),
                "prompt_enhancer": ui_params.get("prompt_enhancer", "")
            }
            
            # Only include optional parameters that are actually supported
            for param_name, param_value in optional_params.items():
                if param_name in supported_params:
                    call_params[param_name] = param_value
                    if param_name == "image_start":
                        print(f"[WGP_GENERATION_DEBUG] image_start type: {type(param_value)}, value: {param_value}")
                else:
                    print(f"[WGP_GENERATION_DEBUG] Skipping unsupported parameter: {param_name}")
            
            print(f"[WGP_GENERATION_DEBUG] Calling generate_video with {len(call_params)} parameters")
            print(f"[WGP_GENERATION_DEBUG] Required parameters added: model_type={call_params.get('model_type')}, batch_size={call_params.get('batch_size')}, sample_solver={call_params.get('sample_solver')}")
            
            # Log the actual VACE parameters being passed to WGP
            print(f"[WGP_VACE_DEBUG] === Final VACE Parameters to WGP ===")
            vace_params = ["video_prompt_type", "video_guide", "video_mask", "control_net_weight", "control_net_weight2", "denoising_strength", "image_refs"]
            for param in vace_params:
                if param in call_params:
                    print(f"[WGP_VACE_DEBUG]   {param}: {call_params[param]}")
            
            print(f"[WGP_VACE_DEBUG] ===============================")
            print(f"[WGP_GENERATION_DEBUG] === CALLING WGP.GENERATE_VIDEO ===")
            
            # [VACE_FIX] SURGICAL FIX: Temporarily override load_models() for VACE base type resolution
            # This preserves all VACE behavior but fixes the config.json lookup issue in load_wan_model()
            original_load_models = wgp_mod.load_models
            vace_model_type = call_params.get("model_type")
            
            if vace_model_type in ["vace_14B", "vace_1.3B", "vace_multitalk_14B"]:
                print(f"[WGP_VACE_DEBUG] Applying surgical fix for VACE config resolution")
                
                def vace_load_models_wrapper(model_type):
                    # For VACE models, resolve to base type for load_models() only
                    if model_type in ["vace_14B", "vace_1.3B", "vace_multitalk_14B"]:
                        try:
                            base_urls = wgp_mod.get_model_recursive_prop(model_type, "URLs", return_list=False)
                            if isinstance(base_urls, str) and base_urls in ["t2v", "i2v"]:
                                print(f"[WGP_VACE_DEBUG] load_models() override: '{model_type}' → base_type '{base_urls}' for config resolution")
                                # Get base model type using WGP's own logic
                                resolved_base_type = wgp_mod.get_base_model_type(base_urls) 
                                print(f"[WGP_VACE_DEBUG] Final base_model_type for load_wan_model: '{resolved_base_type}'")
                                # Call original load_models with base type for proper config resolution
                                return original_load_models(base_urls)
                        except Exception as e:
                            print(f"[WARNING] VACE base type resolution failed: {e}")
                    
                    # For non-VACE models, use normal behavior
                    return original_load_models(model_type)
                
                # Temporarily replace load_models function
                wgp_mod.load_models = vace_load_models_wrapper
            
            try:
                wgp_mod.generate_video(**call_params)
                print(f"[WGP_GENERATION_DEBUG] WGP generation call completed successfully")
            except Exception as e:
                print(f"[WGP_GENERATION_DEBUG] WGP generation call failed: {e}")
                raise
            finally:
                # [VACE_FIX] Always restore original load_models function
                if vace_model_type in ["vace_14B", "vace_1.3B", "vace_multitalk_14B"]:
                    wgp_mod.load_models = original_load_models
                    print(f"[WGP_VACE_DEBUG] Restored original load_models function")
            
            # Find generated video files
            generated_video_files = sorted([
                item for item in Path(temp_output_dir).iterdir()
                if item.is_file() and item.suffix.lower() == ".mp4"
            ])
            
            print(f"[WGP_GENERATION_DEBUG] Found {len(generated_video_files)} video files in output directory")
            
            if not generated_video_files:
                print(f"[WGP_GENERATION_DEBUG] ERROR: No .mp4 files found in {temp_output_dir}")
                return False, f"No video files generated in {temp_output_dir}"
            
            # Analyze each generated file
            for i, video_file in enumerate(generated_video_files):
                from .common_utils import get_video_frame_count_and_fps
                try:
                    frames, fps = get_video_frame_count_and_fps(str(video_file))
                    file_size = video_file.stat().st_size
                    print(f"[WGP_GENERATION_DEBUG] Generated file {i}: {video_file.name}")
                    print(f"[WGP_GENERATION_DEBUG]   Frames: {frames}")
                    print(f"[WGP_GENERATION_DEBUG]   FPS: {fps}")
                    print(f"[WGP_GENERATION_DEBUG]   Size: {file_size / (1024*1024):.2f} MB")
                    print(f"[WGP_GENERATION_DEBUG]   Expected frames: {video_length}")
                    if frames != video_length:
                        print(f"[WGP_GENERATION_DEBUG]   ⚠️  FRAME COUNT MISMATCH! Expected {video_length}, got {frames}")
                except Exception as e:
                    print(f"[WGP_GENERATION_DEBUG] ERROR analyzing {video_file}: {e}")
            
            # Return the first (or only) generated file
            final_output = str(generated_video_files[0].resolve())
            print(f"[WGP_GENERATION_DEBUG] Returning output: {final_output}")
            
            return True, final_output
            
        finally:
            # Restore original save path
            wgp_mod.save_path = original_wgp_save_path
            
    except Exception as e:
        print(f"[WGP_GENERATION_DEBUG] ERROR during generation: {e}")
        import traceback
        traceback.print_exc()
        return False, f"Generation failed: {str(e)}" 