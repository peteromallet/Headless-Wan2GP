"""
Task Registry Module

This module defines the TaskRegistry class and the TASK_HANDLERS dictionary.
It allows for a cleaner way to route tasks to their appropriate handlers
instead of using a massive if/elif block.
"""
from typing import Dict, Any, Callable, Optional, Tuple
from pathlib import Path
import time
import traceback
import json
import uuid

from source.logging_utils import headless_logger
from source.worker_utils import make_task_dprint, log_ram_usage, cleanup_generated_files, dprint
from source.task_conversion import db_task_to_generation_task, parse_phase_config
from source.phase_config import apply_phase_config_patch
from source.params.lora import LoRAConfig
from source.lora_utils import _download_lora_from_url

# Import task handlers
# These imports should be available from the environment where this module is used
from source.specialized_handlers import (
    handle_rife_interpolate_task,
    handle_extract_frame_task
)
from source.comfy_handler import handle_comfy_task
from source.task_engine_router import route_task, COMFY_TASK_TYPES
from source.sm_functions import travel_between_images as tbi
from source.sm_functions import magic_edit as me
from source.sm_functions.join_clips import _handle_join_clips_task
from source.sm_functions.join_clips_orchestrator import _handle_join_clips_orchestrator_task
from source.sm_functions.edit_video_orchestrator import _handle_edit_video_orchestrator_task
from source.sm_functions.inpaint_frames import _handle_inpaint_frames_task
from source.sm_functions.create_visualization import _handle_create_visualization_task
from source.travel_segment_processor import TravelSegmentProcessor, TravelSegmentContext
from source.common_utils import (
    parse_resolution as sm_parse_resolution,
    snap_resolution_to_model_grid,
    ensure_valid_prompt,
    ensure_valid_negative_prompt,
    download_image_if_url as sm_download_image_if_url
)
from source.video_utils import (
    prepare_vace_ref_for_segment as sm_prepare_vace_ref_for_segment,
    create_guide_video_for_travel_segment as sm_create_guide_video_for_travel_segment,
    extract_last_frame_as_image as sm_extract_last_frame_as_image
)
from source import db_operations as db_ops
from headless_model_management import HeadlessTaskQueue, GenerationTask

# Define Direct Queue Task Types
DIRECT_QUEUE_TASK_TYPES = {
    "wan_2_2_t2i", "vace", "vace_21", "vace_22", "flux", "t2v", "t2v_22",
    "i2v", "i2v_22", "hunyuan", "ltxv", "generate_video",
    "qwen_image_edit", "qwen_image_hires", "qwen_image_style", "image_inpaint", "annotated_image_edit",
    # Text-to-image tasks (no input image required)
    "qwen_image", "qwen_image_2512", "z_image_turbo"
}


def _handle_travel_segment_via_queue(task_params_dict, main_output_dir_base: Path, task_id: str, colour_match_videos: bool, mask_active_frames: bool, task_queue: HeadlessTaskQueue, dprint_func, is_standalone: bool = False):
    """
    Handle travel segment tasks via direct queue integration to eliminate blocking waits.
    This is moved from worker.py.
    
    Supports two modes:
    1. Orchestrator mode (is_standalone=False): Requires orchestrator_task_id_ref, orchestrator_run_id, segment_index
    2. Standalone mode (is_standalone=True): full_orchestrator_payload provided directly in params
    """
    headless_logger.debug(f"Starting travel segment queue processing (standalone={is_standalone})", task_id=task_id)
    log_ram_usage("Segment via queue - start", task_id=task_id)
    
    try:
        from source.sm_functions.travel_between_images import _handle_travel_chaining_after_wgp
        
        segment_params = task_params_dict
        orchestrator_task_id_ref = segment_params.get("orchestrator_task_id_ref")
        orchestrator_run_id = segment_params.get("orchestrator_run_id")
        segment_idx = segment_params.get("segment_index")
        
        # Check for orchestrator_details (standard field name) or full_orchestrator_payload (legacy)
        full_orchestrator_payload = segment_params.get("orchestrator_details") or segment_params.get("full_orchestrator_payload")
        
        if is_standalone:
            # Standalone mode: use provided payload, default segment_idx to 0
            if segment_idx is None:
                segment_idx = 0
            if not full_orchestrator_payload:
                return False, f"Individual travel segment {task_id} missing orchestrator_details"
            headless_logger.debug(f"Running in standalone mode (individual_travel_segment)", task_id=task_id)
        else:
            # Orchestrator mode: require segment_index and either inline orchestrator_details or ability to fetch
            if segment_idx is None:
                return False, f"Travel segment {task_id} missing segment_index"
            
            # If orchestrator_details not inline, try to fetch from parent task
            if not full_orchestrator_payload:
                if not orchestrator_task_id_ref:
                    return False, f"Travel segment {task_id} missing orchestrator_details and orchestrator_task_id_ref"
                orchestrator_task_raw_params_json = db_ops.get_task_params(orchestrator_task_id_ref)
                if orchestrator_task_raw_params_json:
                    fetched_params = json.loads(orchestrator_task_raw_params_json) if isinstance(orchestrator_task_raw_params_json, str) else orchestrator_task_raw_params_json
                    full_orchestrator_payload = fetched_params.get("orchestrator_details")
            
            if not full_orchestrator_payload:
                return False, f"Travel segment {task_id}: Could not retrieve orchestrator_details"
        
        # Check for individual_segment_params overrides (highest priority for segment-specific values)
        individual_params = segment_params.get("individual_segment_params", {})
        
        # Model name: top-level > orchestrator_details
        model_name = segment_params.get("model_name") or full_orchestrator_payload["model_name"]
        
        # Prompts: individual_segment_params > top-level > defaults
        prompt_for_wgp = ensure_valid_prompt(
            individual_params.get("base_prompt") or segment_params.get("base_prompt", " ")
        )
        negative_prompt_for_wgp = ensure_valid_negative_prompt(
            individual_params.get("negative_prompt") or segment_params.get("negative_prompt", " ")
        )
        
        # Resolution: top-level > orchestrator_details
        parsed_res_wh_str = segment_params.get("parsed_resolution_wh") or full_orchestrator_payload["parsed_resolution_wh"]
        parsed_res_raw = sm_parse_resolution(parsed_res_wh_str)
        if parsed_res_raw is None:
            return False, f"Travel segment {task_id}: Invalid resolution format {parsed_res_wh_str}"
        parsed_res_wh = snap_resolution_to_model_grid(parsed_res_raw)
        
        # Frame count: individual_segment_params.num_frames > top-level num_frames > segment_frames_target > segment_frames_expanded[idx]
        total_frames_for_segment = (
            individual_params.get("num_frames") or 
            segment_params.get("num_frames") or
            segment_params.get("segment_frames_target") or
            full_orchestrator_payload["segment_frames_expanded"][segment_idx]
        )
        
        current_run_base_output_dir_str = segment_params.get("current_run_base_output_dir")
        if not current_run_base_output_dir_str:
            current_run_base_output_dir_str = full_orchestrator_payload.get("main_output_dir_for_run", str(main_output_dir_base.resolve()))

        # Convert to Path and resolve relative paths against main_output_dir_base
        base_dir_path = Path(current_run_base_output_dir_str)
        if not base_dir_path.is_absolute():
            # Relative path - resolve against main_output_dir_base
            current_run_base_output_dir = main_output_dir_base / base_dir_path
        else:
            # Already absolute - use as is
            current_run_base_output_dir = base_dir_path

        segment_processing_dir = current_run_base_output_dir
        segment_processing_dir.mkdir(parents=True, exist_ok=True)
        
        debug_enabled = segment_params.get("debug_mode_enabled", full_orchestrator_payload.get("debug_mode_enabled", False))

        travel_mode = full_orchestrator_payload.get("model_type", "vace")

        start_ref_path = None
        end_ref_path = None
        
        # Image resolution priority:
        # 1. individual_segment_params.start_image_url / end_image_url
        # 2. individual_segment_params.input_image_paths_resolved (array)
        # 3. top-level input_image_paths_resolved
        # 4. orchestrator_details images (indexed by segment_idx)
        
        individual_images = individual_params.get("input_image_paths_resolved", [])
        top_level_images = segment_params.get("input_image_paths_resolved", [])
        orchestrator_images = full_orchestrator_payload.get("input_image_paths_resolved", [])
        
        dprint_func(f"[IMG_RESOLVE] Task {task_id}: segment_idx={segment_idx}, individual_images={len(individual_images)}, top_level_images={len(top_level_images)}, orchestrator_images={len(orchestrator_images)}")
        
        is_continuing = full_orchestrator_payload.get("continue_from_video_resolved_path") is not None
        use_svi = segment_params.get("use_svi", False) or full_orchestrator_payload.get("use_svi", False)
        svi_predecessor_video_url = segment_params.get("svi_predecessor_video_url") or full_orchestrator_payload.get("svi_predecessor_video_url")
        
        if use_svi:
            dprint_func(f"[SVI_MODE] Task {task_id}: SVI mode enabled for segment {segment_idx}")
        
        # =============================================================================
        # SVI MODE: Chain segments using predecessor video for overlapped_latents
        # SVI Pro uses the last ~9 frames (5 + sliding_window_overlap) from predecessor
        # to create temporal continuity via overlapped_latents in the VAE
        # =============================================================================
        svi_predecessor_video_for_source = None  # Will be set for SVI continuation
        
        if use_svi and (segment_idx > 0 or svi_predecessor_video_url):
            dprint_func(f"[SVI_CHAINING] Seg {segment_idx}: Using SVI end frame chaining mode")
            
            predecessor_output_url = None
            
            # Priority 1: Manually specified predecessor video URL
            if svi_predecessor_video_url:
                dprint_func(f"[SVI_CHAINING] Seg {segment_idx}: Using manually specified predecessor video: {svi_predecessor_video_url}")
                predecessor_output_url = svi_predecessor_video_url
            # Priority 2: Fetch from dependency chain (for segment_idx > 0)
            elif segment_idx > 0:
                task_dependency_id, predecessor_output_url = db_ops.get_predecessor_output_via_edge_function(task_id)
                if task_dependency_id and predecessor_output_url:
                    dprint_func(f"[SVI_CHAINING] Seg {segment_idx}: Found predecessor {task_dependency_id} with output: {predecessor_output_url}")
                else:
                    dprint_func(f"[SVI_CHAINING] Seg {segment_idx}: ERROR - Could not fetch predecessor output (dep_id={task_dependency_id})")
            
            if predecessor_output_url:
                # Download predecessor video if it's a URL
                predecessor_video_path = predecessor_output_url
                if predecessor_output_url.startswith("http"):
                    try:
                        from source.common_utils import download_file as sm_download_file
                        local_filename = Path(predecessor_output_url).name
                        local_download_path = segment_processing_dir / f"svi_predecessor_{segment_idx:02d}_{local_filename}"
                        
                        if not local_download_path.exists():
                            sm_download_file(predecessor_output_url, segment_processing_dir, local_download_path.name)
                            dprint_func(f"[SVI_CHAINING] Seg {segment_idx}: Downloaded predecessor video to {local_download_path}")
                        else:
                            dprint_func(f"[SVI_CHAINING] Seg {segment_idx}: Predecessor video already exists at {local_download_path}")
                        
                        predecessor_video_path = str(local_download_path)
                    except Exception as e_dl:
                        dprint_func(f"[SVI_CHAINING] Seg {segment_idx}: Failed to download predecessor video: {e_dl}")
                        predecessor_video_path = None
                
                # SVI CRITICAL: Extract only the last ~9 frames (5 + overlap_size) from predecessor
                # WGP uses prefix_video[:, -(5 + overlap_size):] to create overlapped_latents,
                # but then prepends the ENTIRE prefix_video to output. By extracting only the
                # last frames ourselves, we limit what gets prepended.
                if predecessor_video_path and Path(predecessor_video_path).exists():
                    from source.video_utils import (
                        get_video_frame_count_and_fps as sm_get_video_frame_count_and_fps,
                        extract_frame_range_to_video as sm_extract_frame_range_to_video
                    )
                    
                    # Get predecessor video frame count
                    pred_frames, pred_fps = sm_get_video_frame_count_and_fps(predecessor_video_path)
                    if pred_frames and pred_frames > 0:
                        # Extract last 9 frames (5 + overlap_size=4) for SVI overlapped_latents
                        overlap_size = 4  # SVI_STITCH_OVERLAP
                        frames_needed = 5 + overlap_size  # 9 frames total
                        start_frame = max(0, int(pred_frames) - frames_needed)
                        
                        trimmed_prefix_filename = f"svi_prefix_{segment_idx:02d}_last{frames_needed}frames_{uuid.uuid4().hex[:6]}.mp4"
                        trimmed_prefix_path = segment_processing_dir / trimmed_prefix_filename
                        
                        trimmed_result = sm_extract_frame_range_to_video(
                            source_video=predecessor_video_path,
                            output_path=str(trimmed_prefix_path),
                            start_frame=start_frame,
                            end_frame=None,  # To end
                            fps=float(pred_fps) if pred_fps and pred_fps > 0 else 16.0,
                            dprint_func=dprint_func
                        )
                        
                        if trimmed_result and Path(trimmed_result).exists():
                            svi_predecessor_video_for_source = str(trimmed_result)
                            dprint_func(f"[SVI_CHAINING] Seg {segment_idx}: Extracted last {frames_needed} frames from predecessor ({pred_frames} total) -> {trimmed_result}")
                        else:
                            # Fallback: use full video if extraction failed
                            svi_predecessor_video_for_source = predecessor_video_path
                            dprint_func(f"[SVI_CHAINING] Seg {segment_idx}: WARNING - Failed to extract last frames, using full predecessor video")
                    else:
                        # Fallback: use full video if we can't get frame count
                        svi_predecessor_video_for_source = predecessor_video_path
                        dprint_func(f"[SVI_CHAINING] Seg {segment_idx}: Could not get predecessor frame count, using full video")
                    
                    # Still extract last frame for image_refs (anchor reference)
                    start_ref_path = sm_extract_last_frame_as_image(
                        predecessor_video_path, 
                        segment_processing_dir, 
                        task_id
                    )
                    dprint_func(f"[SVI_CHAINING] Seg {segment_idx}: Extracted last frame as anchor start_ref: {start_ref_path}")
                else:
                    dprint_func(f"[SVI_CHAINING] Seg {segment_idx}: ERROR - Predecessor video not available at {predecessor_video_path}")
            
            # For SVI, end_ref is the target image from input array
            target_end_idx = segment_idx + 1 if segment_idx > 0 else 1
            if svi_predecessor_video_url and segment_idx == 0:
                target_end_idx = 1 if len(orchestrator_images) > 1 else 0
            
            if len(orchestrator_images) > target_end_idx:
                end_ref_path = orchestrator_images[target_end_idx]
                dprint_func(f"[SVI_CHAINING] Seg {segment_idx}: end_ref from input_images[{target_end_idx}]: {end_ref_path}")
            elif len(orchestrator_images) > 0:
                end_ref_path = orchestrator_images[-1]
                dprint_func(f"[SVI_CHAINING] Seg {segment_idx}: end_ref fallback to last input image: {end_ref_path}")
        
        # =============================================================================
        # SVI MODE: First segment uses input images normally (no manual predecessor)
        # =============================================================================
        elif use_svi and segment_idx == 0:
            dprint_func(f"[SVI_CHAINING] Seg {segment_idx}: First segment in SVI mode - using input images")
            if len(orchestrator_images) > 0:
                start_ref_path = orchestrator_images[0]
            if len(orchestrator_images) > 1:
                end_ref_path = orchestrator_images[1]
        
        # =============================================================================
        # NON-SVI MODES: Original logic
        # =============================================================================
        # Check individual_segment_params first (highest priority for standalone)
        elif individual_params.get("start_image_url") or individual_params.get("end_image_url"):
            start_ref_path = individual_params.get("start_image_url")
            end_ref_path = individual_params.get("end_image_url")
            dprint_func(f"[IMG_RESOLVE] Task {task_id}: Using individual_segment_params URLs: start={start_ref_path}")
        elif len(individual_images) >= 2:
            start_ref_path = individual_images[0]
            end_ref_path = individual_images[1]
            dprint_func(f"[IMG_RESOLVE] Task {task_id}: Using individual_segment_params array: start={start_ref_path}")
        elif is_standalone and len(top_level_images) >= 2:
            start_ref_path = top_level_images[0]
            end_ref_path = top_level_images[1]
            dprint_func(f"[IMG_RESOLVE] Task {task_id}: Using top-level images directly: start={start_ref_path}")
        elif is_continuing:
            if segment_idx == 0:
                continued_video_path = full_orchestrator_payload.get("continue_from_video_resolved_path")
                if continued_video_path and Path(continued_video_path).exists():
                    start_ref_path = sm_extract_last_frame_as_image(continued_video_path, segment_processing_dir, task_id)
                if orchestrator_images:
                    end_ref_path = orchestrator_images[0]
            else:
                if len(orchestrator_images) > segment_idx:
                    start_ref_path = orchestrator_images[segment_idx - 1]
                    end_ref_path = orchestrator_images[segment_idx]
        else:
            dprint_func(f"[IMG_RESOLVE] Task {task_id}: Using orchestrator images (from scratch)")
            if len(orchestrator_images) > segment_idx:
                start_ref_path = orchestrator_images[segment_idx]
            
            if len(orchestrator_images) > segment_idx + 1:
                end_ref_path = orchestrator_images[segment_idx + 1]
        dprint_func(f"[IMG_RESOLVE] Task {task_id}: start_ref_path after logic: {start_ref_path}")
        
        if start_ref_path:
            start_ref_path = sm_download_image_if_url(start_ref_path, segment_processing_dir, task_id, debug_mode=debug_enabled)
            dprint_func(f"[IMG_RESOLVE] Task {task_id}: start_ref_path AFTER DOWNLOAD: {start_ref_path}")
        if end_ref_path:
            end_ref_path = sm_download_image_if_url(end_ref_path, segment_processing_dir, task_id, debug_mode=debug_enabled)

        guide_video_path = None
        mask_video_path_for_wgp = None
        video_prompt_type_str = None

        if travel_mode == "vace":
            try:
                processor_context = TravelSegmentContext(
                    task_id=task_id,
                    segment_idx=segment_idx,
                    model_name=model_name,
                    total_frames_for_segment=total_frames_for_segment,
                    parsed_res_wh=parsed_res_wh,
                    segment_processing_dir=segment_processing_dir,
                    main_output_dir_base=main_output_dir_base,
                    full_orchestrator_payload=full_orchestrator_payload,
                    segment_params=segment_params,
                    mask_active_frames=mask_active_frames,
                    debug_enabled=debug_enabled,
                    dprint=dprint_func
                )
                processor = TravelSegmentProcessor(processor_context)
                segment_outputs = processor.process_segment()
                
                guide_video_path = segment_outputs.get("video_guide")
                mask_video_path_for_wgp = Path(segment_outputs["video_mask"]) if segment_outputs.get("video_mask") else None
                video_prompt_type_str = segment_outputs["video_prompt_type"]
                
            except Exception as e_shared_processor:
                traceback.print_exc()
                return False, f"Shared processor failed: {e_shared_processor}"
        
        # Seed: individual_segment_params > top-level > default
        seed_to_use = individual_params.get("seed_to_use") or segment_params.get("seed_to_use", 12345)
        
        generation_params = {
            "model_name": model_name,
            "negative_prompt": negative_prompt_for_wgp,
            "resolution": f"{parsed_res_wh[0]}x{parsed_res_wh[1]}",
            "video_length": total_frames_for_segment,
            "seed": seed_to_use,
        }
        
        # Always pass images if available, regardless of specific travel_mode string (hybrid models need them)
        if start_ref_path: 
            generation_params["image_start"] = str(Path(start_ref_path).resolve())
        if end_ref_path: 
            generation_params["image_end"] = str(Path(end_ref_path).resolve())
            
        dprint_func(f"[IMG_RESOLVE] Task {task_id}: generation_params image_start: {generation_params.get('image_start')}")
        
        # ═══════════════════════════════════════════════════════════════════════════
        # Parameter extraction with precedence: individual > segment > orchestrator
        # Note: Typed TaskConfig conversion happens in HeadlessTaskQueue (single point)
        # ═══════════════════════════════════════════════════════════════════════════
        
        # Additional LoRAs: individual_segment_params > top-level > orchestrator_details
        additional_loras = (
            individual_params.get("additional_loras") or 
            segment_params.get("additional_loras") or 
            full_orchestrator_payload.get("additional_loras", {})
        )
        if additional_loras:
            generation_params["additional_loras"] = additional_loras

        explicit_steps = (segment_params.get("num_inference_steps") or segment_params.get("steps") or 
                          full_orchestrator_payload.get("num_inference_steps") or full_orchestrator_payload.get("steps"))
        if explicit_steps: generation_params["num_inference_steps"] = explicit_steps
        
        explicit_guidance = (segment_params.get("guidance_scale") or full_orchestrator_payload.get("guidance_scale"))
        if explicit_guidance: generation_params["guidance_scale"] = explicit_guidance
        
        explicit_flow_shift = (segment_params.get("flow_shift") or full_orchestrator_payload.get("flow_shift"))
        if explicit_flow_shift: generation_params["flow_shift"] = explicit_flow_shift

        # Phase Config: individual_segment_params > top-level > orchestrator_details
        phase_config_source = (
            individual_params.get("phase_config") or 
            segment_params.get("phase_config") or 
            full_orchestrator_payload.get("phase_config")
        )
        if phase_config_source:
            try:
                steps_per_phase = phase_config_source.get("steps_per_phase", [2, 2, 2])
                phase_config_steps = sum(steps_per_phase)
                
                parsed_phase_config = parse_phase_config(
                    phase_config=phase_config_source,
                    num_inference_steps=phase_config_steps,
                    task_id=task_id,
                    model_name=generation_params.get("model_name"),
                    debug_mode=debug_enabled
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
                
                # CRITICAL: Run LoRA resolution after phase_config parsing
                # phase_config extracts LoRA URLs which need to be downloaded and resolved to absolute paths
                if any(key in generation_params for key in ["activated_loras", "loras_multipliers", "additional_loras"]):
                    lora_config = LoRAConfig.from_params(generation_params, task_id=task_id)
                    
                    # Download any pending LoRAs
                    if lora_config.has_pending_downloads():
                        for url in lora_config.get_pending_downloads().keys():
                            if url:
                                local_filename = _download_lora_from_url(
                                    url=url,
                                    task_id=task_id,
                                    dprint=dprint_func,
                                    model_type=model_name,
                                )
                                lora_config.mark_downloaded(url, local_filename)
                    
                    # Convert to WGP format
                    wgp_lora = lora_config.to_wgp_format()
                    if wgp_lora["activated_loras"]:
                        # DEBUG: Log exactly what LoRAConfig resolved
                        dprint_func(f"[LORA_CONFIG_OUTPUT] Task {task_id}: Resolved {len(wgp_lora['activated_loras'])} LoRAs:")
                        for i, lora_path in enumerate(wgp_lora["activated_loras"]):
                            dprint_func(f"[LORA_CONFIG_OUTPUT] Task {task_id}:   [{i}] '{lora_path}'")
                        
                        generation_params["activated_loras"] = wgp_lora["activated_loras"]
                        generation_params["loras_multipliers"] = wgp_lora["loras_multipliers"]
                        headless_logger.info(f"Resolved {len(wgp_lora['activated_loras'])} LoRAs from phase_config", task_id=task_id)
                
                if "_patch_config" in parsed_phase_config:
                    apply_phase_config_patch(parsed_phase_config, model_name, task_id)

            except Exception as e:
                raise ValueError(f"Task {task_id}: Invalid phase_config: {e}")

        if guide_video_path: generation_params["video_guide"] = str(guide_video_path)
        if mask_video_path_for_wgp: generation_params["video_mask"] = str(mask_video_path_for_wgp.resolve())
        generation_params["video_prompt_type"] = video_prompt_type_str
        
        # =============================================================================
        # SVI MODE: Add SVI-specific generation parameters
        # =============================================================================
        if use_svi:
            dprint_func(f"[SVI_PAYLOAD] Task {task_id}: Configuring WGP payload for SVI mode")
            
            # Enable SVI encoding mode
            generation_params["svi2pro"] = True
            
            # SVI requires video_prompt_type="I" to enable image_refs passthrough
            generation_params["video_prompt_type"] = "I"
            
            # Set image_refs to start image (anchor for SVI encoding)
            if start_ref_path:
                generation_params["image_refs_paths"] = [str(Path(start_ref_path).resolve())]
                dprint_func(f"[SVI_PAYLOAD] Task {task_id}: Set image_refs_paths to start image: {start_ref_path}")
            
            # CRITICAL: Pass predecessor video as video_source for SVI continuation
            # WGP uses prefix_video[:, -(5 + overlap_size):] to create overlapped_latents
            # This provides temporal continuity between segments (uses last ~9 frames)
            # IMPORTANT: Must NOT set image_start when video_source is set, otherwise
            # wgp.py prioritizes image_start and creates only a 1-frame prefix_video!
            if svi_predecessor_video_for_source:
                generation_params["video_source"] = str(Path(svi_predecessor_video_for_source).resolve())
                # Remove image_start so WGP uses video_source for multi-frame prefix_video
                # The anchor image is provided via image_refs instead
                if "image_start" in generation_params:
                    del generation_params["image_start"]
                    dprint_func(f"[SVI_PAYLOAD] Task {task_id}: Removed image_start (anchor provided via image_refs)")
                # CRITICAL: Set image_prompt_type to include "V" to enable video_source usage
                # SVI2Pro allows "SVL" - we use "SV" for start+video continuation
                generation_params["image_prompt_type"] = "SV"
                dprint_func(f"[SVI_PAYLOAD] Task {task_id}: Set image_prompt_type='SV' for video continuation")
                dprint_func(f"[SVI_PAYLOAD] Task {task_id}: Set video_source for SVI continuation: {svi_predecessor_video_for_source}")
            
            # SVI Pro sliding window overlap = 4 frames (standard for SVI)
            # This tells WGP how many frames to extract for overlapped_latents
            generation_params["sliding_window_overlap"] = 4
            dprint_func(f"[SVI_PAYLOAD] Task {task_id}: Set sliding_window_overlap=4 for SVI")
            
            # Add SVI generation parameters from segment_params (set by orchestrator)
            for key in ["guidance_phases", "num_inference_steps", "guidance_scale", "guidance2_scale",
                       "flow_shift", "switch_threshold", "model_switch_phase", "sample_solver"]:
                if key in segment_params and segment_params[key] is not None:
                    generation_params[key] = segment_params[key]
                    dprint_func(f"[SVI_PAYLOAD] Task {task_id}: Set {key}={segment_params[key]}")
            
            # Merge SVI LoRAs with existing additional_loras
            from source.sm_functions.travel_between_images import get_svi_additional_loras
            existing_payload_loras = generation_params.get("additional_loras", {})
            generation_params["additional_loras"] = get_svi_additional_loras(existing_payload_loras)
            dprint_func(f"[SVI_PAYLOAD] Task {task_id}: Merged SVI LoRAs into additional_loras")
        
        # === WGP SUBMISSION DIAGNOSTIC SUMMARY ===
        # Log key frame-related parameters before WGP submission
        dprint_func(f"[WGP_SUBMIT] Task {task_id}: ========== WGP GENERATION REQUEST ==========")
        dprint_func(f"[WGP_SUBMIT] Task {task_id}: video_length (target frames): {generation_params.get('video_length')}")
        is_valid_4n1 = (generation_params.get('video_length', 0) - 1) % 4 == 0
        dprint_func(f"[WGP_SUBMIT] Task {task_id}: Valid 4N+1: {is_valid_4n1} {'✓' if is_valid_4n1 else '✗ WARNING'}")
        dprint_func(f"[WGP_SUBMIT] Task {task_id}: video_guide: {generation_params.get('video_guide', 'None')}")
        dprint_func(f"[WGP_SUBMIT] Task {task_id}: video_mask: {generation_params.get('video_mask', 'None')}")
        dprint_func(f"[WGP_SUBMIT] Task {task_id}: model: {model_name}")
        dprint_func(f"[WGP_SUBMIT] Task {task_id}: resolution: {generation_params.get('resolution')}")
        dprint_func(f"[WGP_SUBMIT] Task {task_id}: =============================================")
        
        # IMPORTANT: Use the DB task_id as the queue task id.
        # This keeps logs, fatal error handling, and debug tooling consistent (no "travel_seg_" indirection).
        # We still include a hint in parameters so the queue can apply any task-type specific behavior.
        generation_params["_source_task_type"] = "travel_segment"
        generation_task = GenerationTask(
            id=task_id,
            model=model_name,
            prompt=prompt_for_wgp,
            parameters=generation_params
        )
        
        submitted_task_id = task_queue.submit_task(generation_task)
        
        max_wait_time = 1800
        wait_interval = 2
        elapsed_time = 0
        
        while elapsed_time < max_wait_time:
            status = task_queue.get_task_status(task_id)
            if status is None: return False, f"Travel segment {task_id}: Task status became None"
            
            if status.status == "completed":
                # In standalone mode, skip chaining (no orchestrator to coordinate with)
                if is_standalone:
                    headless_logger.debug(f"Standalone segment completed, skipping chaining", task_id=task_id)
                    return True, status.result_path
                
                # Orchestrator mode: run chaining
                chain_success, chain_message, final_chained_path = _handle_travel_chaining_after_wgp(
                    wgp_task_params={"travel_chain_details": {
                        "orchestrator_task_id_ref": orchestrator_task_id_ref,
                        "orchestrator_run_id": orchestrator_run_id,
                        "segment_index_completed": segment_idx,
                        "full_orchestrator_payload": full_orchestrator_payload,
                        "segment_processing_dir_for_saturation": str(segment_processing_dir),
                        "is_first_new_segment_after_continue": segment_params.get("is_first_segment", False) and full_orchestrator_payload.get("continue_from_video_resolved_path"),
                        "is_subsequent_segment": not segment_params.get("is_first_segment", True),
                        "colour_match_videos": colour_match_videos,
                        "cm_start_ref_path": None, "cm_end_ref_path": None, "show_input_images": False, "start_image_path": None, "end_image_path": None,
                    }},
                    actual_wgp_output_video_path=status.result_path,
                    image_download_dir=segment_processing_dir,
                    main_output_dir_base=main_output_dir_base,
                    dprint=dprint_func
                )
                if chain_success and final_chained_path:
                    return True, final_chained_path
                else:
                    return True, status.result_path
            elif status.status == "failed":
                return False, f"Travel segment {task_id}: Generation failed: {status.error_message}"
            
            time.sleep(wait_interval)
            elapsed_time += wait_interval
        
        return False, f"Travel segment {task_id}: Generation timeout"

    except Exception as e:
        traceback.print_exc()
        return False, f"Travel segment {task_id}: Exception: {str(e)}"


class TaskRegistry:
    """Registry for task handlers."""
    
    @staticmethod
    def dispatch(task_type: str, context: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Dispatch a task to the appropriate handler.
        
        Args:
            task_type: The type of task to execute.
            context: Dictionary containing all necessary context variables:
                - task_params_dict
                - main_output_dir_base
                - task_id
                - project_id
                - task_queue
                - colour_match_videos
                - mask_active_frames
                - debug_mode
                - wan2gp_path
        
        Returns:
            Tuple (success, output_location)
        """
        task_id = context["task_id"]
        params = context["task_params_dict"]
        dprint_func = make_task_dprint(task_id, context["debug_mode"])

        # 1. Direct Queue Tasks
        if task_type in DIRECT_QUEUE_TASK_TYPES and context["task_queue"]:
            return TaskRegistry._handle_direct_queue_task(task_type, context)

        # 2. Orchestrator & Specialized Handlers
        handlers = {
            "travel_orchestrator": lambda: tbi._handle_travel_orchestrator_task(
                task_params_from_db=params,
                main_output_dir_base=context["main_output_dir_base"],
                orchestrator_task_id_str=task_id,
                orchestrator_project_id=context["project_id"],
                dprint=dprint_func
            ),
            "travel_segment": lambda: _handle_travel_segment_via_queue(
                task_params_dict=params,
                main_output_dir_base=context["main_output_dir_base"],
                task_id=task_id,
                colour_match_videos=context["colour_match_videos"],
                mask_active_frames=context["mask_active_frames"],
                task_queue=context["task_queue"],
                dprint_func=dprint_func,
                is_standalone=False
            ),
            "individual_travel_segment": lambda: _handle_travel_segment_via_queue(
                task_params_dict=params,
                main_output_dir_base=context["main_output_dir_base"],
                task_id=task_id,
                colour_match_videos=context["colour_match_videos"],
                mask_active_frames=context["mask_active_frames"],
                task_queue=context["task_queue"],
                dprint_func=dprint_func,
                is_standalone=True
            ),
            "travel_stitch": lambda: tbi._handle_travel_stitch_task(
                task_params_from_db=params,
                main_output_dir_base=context["main_output_dir_base"],
                stitch_task_id_str=task_id,
                dprint=dprint_func
            ),
            "magic_edit": lambda: me._handle_magic_edit_task(
                task_params_from_db=params,
                main_output_dir_base=context["main_output_dir_base"],
                task_id=task_id,
                dprint=dprint_func
            ),
            "join_clips_orchestrator": lambda: _handle_join_clips_orchestrator_task(
                task_params_from_db=params,
                main_output_dir_base=context["main_output_dir_base"],
                orchestrator_task_id_str=task_id,
                orchestrator_project_id=context["project_id"],
                dprint=dprint_func
            ),
            "edit_video_orchestrator": lambda: _handle_edit_video_orchestrator_task(
                task_params_from_db=params,
                main_output_dir_base=context["main_output_dir_base"],
                orchestrator_task_id_str=task_id,
                orchestrator_project_id=context["project_id"],
                dprint=dprint_func
            ),
            "join_clips_segment": lambda: _handle_join_clips_task(
                task_params_from_db=params,
                main_output_dir_base=context["main_output_dir_base"],
                task_id=task_id,
                task_queue=context["task_queue"],
                dprint=dprint_func
            ),
            "inpaint_frames": lambda: _handle_inpaint_frames_task(
                task_params_from_db=params,
                main_output_dir_base=context["main_output_dir_base"],
                task_id=task_id,
                task_queue=context["task_queue"],
                dprint=dprint_func
            ),
            "create_visualization": lambda: _handle_create_visualization_task(
                task_params_from_db=params,
                main_output_dir_base=context["main_output_dir_base"],
                viz_task_id_str=task_id,
                dprint=dprint_func
            ),
            "extract_frame": lambda: handle_extract_frame_task(
                params, context["main_output_dir_base"], task_id, dprint_func
            ),
            "rife_interpolate_images": lambda: handle_rife_interpolate_task(
                params, context["main_output_dir_base"], task_id, dprint_func, task_queue=context["task_queue"]
            ),
            "comfy": lambda: handle_comfy_task(
                task_params_from_db=params,
                main_output_dir_base=context["main_output_dir_base"],
                task_id=task_id,
                dprint=dprint_func
            )
        }

        if task_type in handlers:
            # Orchestrator setup
            if task_type in ["travel_orchestrator", "join_clips_orchestrator", "edit_video_orchestrator"]:
                params["task_id"] = task_id
                if "orchestrator_details" in params:
                    params["orchestrator_details"]["orchestrator_task_id"] = task_id
            
            return handlers[task_type]()

        # Default fallthrough to queue
        if context["task_queue"]:
             return TaskRegistry._handle_direct_queue_task(task_type, context)
        
        raise ValueError(f"Unknown task type {task_type} and no queue available")

    @staticmethod
    def _handle_direct_queue_task(task_type: str, context: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        task_id = context["task_id"]
        task_queue = context["task_queue"]
        params = context["task_params_dict"]
        
        try:
            generation_task = db_task_to_generation_task(
                params, task_id, task_type, context["wan2gp_path"], context["debug_mode"]
            )
            
            if task_type == "wan_2_2_t2i":
                generation_task.parameters["video_length"] = 1
            
            if context["colour_match_videos"]:
                generation_task.parameters["colour_match_videos"] = True
            if context["mask_active_frames"]:
                generation_task.parameters["mask_active_frames"] = True
            
            task_queue.submit_task(generation_task)
            
            # Wait for completion
            max_wait_time = 3600
            elapsed = 0
            while elapsed < max_wait_time:
                status = task_queue.get_task_status(task_id)
                if not status: return False, "Task status became None"
                
                if status.status == "completed":
                    return True, status.result_path
                elif status.status == "failed":
                    return False, status.error_message or "Failed without message"
                
                time.sleep(2)
                elapsed += 2
            
            return False, "Timeout"
            
        except Exception as e:
            traceback.print_exc()
            return False, f"Queue error: {e}"

