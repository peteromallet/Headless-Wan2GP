import json
import math
import shutil
import traceback
from pathlib import Path
import time
import subprocess
import uuid
from datetime import datetime

# Import structured logging
from ..logging_utils import travel_logger

try:
    import cv2
    import numpy as np
    _COLOR_MATCH_DEPS_AVAILABLE = True
except ImportError:
    _COLOR_MATCH_DEPS_AVAILABLE = False

# --- SM_RESTRUCTURE: Import moved/new utilities ---
from .. import db_operations as db_ops
from ..common_utils import (
    generate_unique_task_id as sm_generate_unique_task_id,
    get_video_frame_count_and_fps as sm_get_video_frame_count_and_fps,
    sm_get_unique_target_path,
    create_color_frame as sm_create_color_frame,
    parse_resolution as sm_parse_resolution,    
    download_image_if_url as sm_download_image_if_url,
    prepare_output_path,
    prepare_output_path_with_upload,
    upload_and_get_final_output_location,
    snap_resolution_to_model_grid,
    ensure_valid_prompt,
    ensure_valid_negative_prompt,
    # process_additional_loras_shared,  # DEPRECATED - no longer used in queue-only system
    wait_for_file_stable as sm_wait_for_file_stable,
)
from ..video_utils import (
    extract_frames_from_video as sm_extract_frames_from_video,
    create_video_from_frames_list as sm_create_video_from_frames_list,
    cross_fade_overlap_frames as sm_cross_fade_overlap_frames,
    _apply_saturation_to_video_ffmpeg as sm_apply_saturation_to_video_ffmpeg,
    apply_brightness_to_video_frames,
    prepare_vace_ref_for_segment as sm_prepare_vace_ref_for_segment,
    create_guide_video_for_travel_segment as sm_create_guide_video_for_travel_segment,
    apply_color_matching_to_video as sm_apply_color_matching_to_video,
    extract_last_frame_as_image as sm_extract_last_frame_as_image,
    overlay_start_end_images_above_video as sm_overlay_start_end_images_above_video,
)
# Legacy wgp_utils import removed - now using task_queue system exclusively

# Add debugging helper function
def debug_video_analysis(video_path: str | Path, label: str, task_id: str = "unknown") -> dict:
    """Analyze a video file and return comprehensive debug info"""
    try:
        path_obj = Path(video_path)
        if not path_obj.exists():
            travel_logger.debug(f"{label}: FILE MISSING - {video_path}", task_id=task_id)
            return {"exists": False, "path": str(video_path)}
        
        frame_count, fps = sm_get_video_frame_count_and_fps(str(path_obj))
        file_size = path_obj.stat().st_size
        duration = frame_count / fps if fps and fps > 0 else 0
        
        debug_info = {
            "exists": True,
            "path": str(path_obj.resolve()),
            "frame_count": frame_count,
            "fps": fps,
            "duration_seconds": duration,
            "file_size_bytes": file_size,
            "file_size_mb": round(file_size / (1024*1024), 2)
        }
        
        travel_logger.debug(f"{label}: {debug_info['frame_count']} frames, {debug_info['fps']} fps, {debug_info['duration_seconds']:.2f}s, {debug_info['file_size_mb']} MB", task_id=task_id)
        
        return debug_info
        
    except Exception as e:
        travel_logger.debug(f"{label}: ERROR analyzing video - {e}", task_id=task_id)
        return {"exists": False, "error": str(e), "path": str(video_path)}

# --- SM_RESTRUCTURE: New Handler Functions for Travel Tasks ---
def _handle_travel_orchestrator_task(task_params_from_db: dict, main_output_dir_base: Path, orchestrator_task_id_str: str, orchestrator_project_id: str | None, *, dprint):
    travel_logger.essential("Starting travel orchestrator task", task_id=orchestrator_task_id_str)
    travel_logger.debug(f"Project ID: {orchestrator_project_id}", task_id=orchestrator_task_id_str)
    travel_logger.debug(f"Task params: {json.dumps(task_params_from_db, default=str, indent=2)[:1000]}...", task_id=orchestrator_task_id_str)
    generation_success = False # Represents success of orchestration step
    output_message_for_orchestrator_db = f"Orchestration for {orchestrator_task_id_str} initiated."

    try:
        if 'orchestrator_details' not in task_params_from_db:
            travel_logger.error("'orchestrator_details' not found in task_params_from_db", task_id=orchestrator_task_id_str)
            return False, "orchestrator_details missing"
        
        orchestrator_payload = task_params_from_db['orchestrator_details']
        travel_logger.debug(f"Orchestrator payload: {json.dumps(orchestrator_payload, indent=2, default=str)[:500]}...", task_id=orchestrator_task_id_str)

        run_id = orchestrator_payload.get("run_id", orchestrator_task_id_str)
        base_dir_for_this_run_str = orchestrator_payload.get("main_output_dir_for_run", str(main_output_dir_base.resolve()))
        
        # Use the base directory directly without creating run-specific subdirectories
        current_run_output_dir = Path(base_dir_for_this_run_str)
        current_run_output_dir.mkdir(parents=True, exist_ok=True)
        dprint(f"Orchestrator {orchestrator_task_id_str}: Base output directory for this run: {current_run_output_dir.resolve()}")

        num_segments = orchestrator_payload.get("num_new_segments_to_generate", 0)
        if num_segments <= 0:
            msg = f"[WARNING Task ID: {orchestrator_task_id_str}] No new segments to generate based on orchestrator payload. Orchestration complete (vacuously)."
            print(msg)
            return True, msg

        db_path_for_add = db_ops.SQLITE_DB_PATH if db_ops.DB_TYPE == "sqlite" else None
        previous_segment_task_id = None

        # --- Determine image download directory for this orchestrated run ---
        segment_image_download_dir_str : str | None = None
        if db_ops.DB_TYPE == "sqlite" and db_ops.SQLITE_DB_PATH: # SQLITE_DB_PATH is global
            try:
                sqlite_db_path_obj = Path(db_ops.SQLITE_DB_PATH).resolve()
                if sqlite_db_path_obj.is_file():
                    sqlite_db_parent_dir = sqlite_db_path_obj.parent
                    # Orchestrated downloads go into a subfolder named after the run_id
                    candidate_download_dir = sqlite_db_parent_dir / "public" / "data" / "image_downloads_orchestrated" / run_id
                    candidate_download_dir.mkdir(parents=True, exist_ok=True)
                    segment_image_download_dir_str = str(candidate_download_dir.resolve())
                    dprint(f"Orchestrator {orchestrator_task_id_str}: Determined segment_image_download_dir for run {run_id}: {segment_image_download_dir_str}")
                else:
                    dprint(f"Orchestrator {orchestrator_task_id_str}: SQLITE_DB_PATH '{db_ops.SQLITE_DB_PATH}' is not a file. Cannot determine parent for image_download_dir.")
            except Exception as e_idir_orch:
                dprint(f"Orchestrator {orchestrator_task_id_str}: Could not create image_download_dir for run {run_id}: {e_idir_orch}. Segments may not download URL images to specific dir.")
        # Add similar logic for Supabase if a writable shared path convention exists.

        # Expanded arrays from orchestrator payload
        expanded_base_prompts = orchestrator_payload["base_prompts_expanded"]
        expanded_negative_prompts = orchestrator_payload["negative_prompts_expanded"]
        expanded_segment_frames = orchestrator_payload["segment_frames_expanded"]
        expanded_frame_overlap = orchestrator_payload["frame_overlap_expanded"]
        vace_refs_instructions_all = orchestrator_payload.get("vace_image_refs_to_prepare_by_worker", [])

        # Preserve a copy of the original overlap list in case we need it later
        _orig_frame_overlap = list(expanded_frame_overlap)  # shallow copy

        # --- SM_QUANTIZE_FRAMES_AND_OVERLAPS ---
        # Adjust all segment lengths to match model constraints (4*N+1 format).
        # Then, adjust overlap values to be even and not exceed the length of the
        # smaller of the two segments they connect. This prevents errors downstream
        # in guide video creation, generation, and stitching.
        
        print(f"[FRAME_DEBUG] Orchestrator {orchestrator_task_id_str}: QUANTIZATION ANALYSIS")
        print(f"[FRAME_DEBUG] Original segment_frames_expanded: {expanded_segment_frames}")
        print(f"[FRAME_DEBUG] Original frame_overlap: {expanded_frame_overlap}")
        
        quantized_segment_frames = []
        dprint(f"Orchestrator: Quantizing frame counts. Original segment_frames_expanded: {expanded_segment_frames}")
        for i, frames in enumerate(expanded_segment_frames):
            # Quantize to 4*N+1 format to match model constraints, applied later in worker.py
            new_frames = (frames // 4) * 4 + 1
            print(f"[FRAME_DEBUG] Segment {i}: {frames} -> {new_frames} (4*N+1 quantization)")
            if new_frames != frames:
                dprint(f"Orchestrator: Quantized segment {i} length from {frames} to {new_frames} (4*N+1 format).")
            quantized_segment_frames.append(new_frames)
        
        print(f"[FRAME_DEBUG] Quantized segment_frames: {quantized_segment_frames}")
        dprint(f"Orchestrator: Finished quantizing frame counts. New quantized_segment_frames: {quantized_segment_frames}")
        
        quantized_frame_overlap = []
        # There are N-1 overlaps for N segments. The loop must not iterate more times than this.
        num_overlaps_to_process = len(quantized_segment_frames) - 1
        print(f"[FRAME_DEBUG] Processing {num_overlaps_to_process} overlap values")

        if num_overlaps_to_process > 0:
            for i in range(num_overlaps_to_process):
                # Gracefully handle if the original overlap array is longer than expected.
                if i < len(expanded_frame_overlap):
                    original_overlap = expanded_frame_overlap[i]
                else:
                    # This case should not happen if client is correct, but as a fallback.
                    dprint(f"Orchestrator: Overlap at index {i} missing. Defaulting to 0.")
                    original_overlap = 0
                
                # Overlap connects segment i and i+1.
                # It cannot be larger than the shorter of the two segments.
                max_possible_overlap = min(quantized_segment_frames[i], quantized_segment_frames[i+1])

                # Quantize original overlap to be even, then cap it.
                new_overlap = (original_overlap // 2) * 2
                new_overlap = min(new_overlap, max_possible_overlap)
                if new_overlap < 0: new_overlap = 0

                print(f"[FRAME_DEBUG] Overlap {i} (segments {i}->{i+1}): {original_overlap} -> {new_overlap}")
                print(f"[FRAME_DEBUG]   Segment lengths: {quantized_segment_frames[i]}, {quantized_segment_frames[i+1]}")
                print(f"[FRAME_DEBUG]   Max possible overlap: {max_possible_overlap}")
                
                if new_overlap != original_overlap:
                    dprint(f"Orchestrator: Adjusted overlap between segments {i}-{i+1} from {original_overlap} to {new_overlap}.")
                
                quantized_frame_overlap.append(new_overlap)
        
        print(f"[FRAME_DEBUG] Final quantized_frame_overlap: {quantized_frame_overlap}")
        
        # Persist quantised results back to orchestrator_payload so all downstream tasks see them
        orchestrator_payload["segment_frames_expanded"] = quantized_segment_frames
        orchestrator_payload["frame_overlap_expanded"] = quantized_frame_overlap
        
        # Calculate expected final length
        total_input_frames = sum(quantized_segment_frames)
        total_overlaps = sum(quantized_frame_overlap)
        expected_final_length = total_input_frames - total_overlaps
        print(f"[FRAME_DEBUG] EXPECTED FINAL VIDEO:")
        print(f"[FRAME_DEBUG]   Total input frames: {total_input_frames}")
        print(f"[FRAME_DEBUG]   Total overlaps: {total_overlaps}")
        print(f"[FRAME_DEBUG]   Expected final length: {expected_final_length} frames")
        print(f"[FRAME_DEBUG]   Expected duration: {expected_final_length / orchestrator_payload.get('fps_helpers', 16):.2f}s")
        
        # Replace original lists with the new quantized ones for all subsequent logic
        expanded_segment_frames = quantized_segment_frames
        expanded_frame_overlap = quantized_frame_overlap
        # --- END SM_QUANTIZE_FRAMES_AND_OVERLAPS ---

        # If quantisation resulted in an empty overlap list (e.g. single-segment run) but the
        # original payload DID contain an overlap value, restore that so the first segment
        # can still reuse frames from the previous/continued video.  This is crucial for
        # continue-video journeys where we expect `frame_overlap_from_previous` > 0.
        if (not expanded_frame_overlap) and _orig_frame_overlap:
            expanded_frame_overlap = _orig_frame_overlap

        for idx in range(num_segments):
            current_segment_task_id = sm_generate_unique_task_id(f"travel_seg_{run_id}_{idx:02d}_")
            
            # Note: segment handler now manages its own output paths using prepare_output_path()

            # Determine frame_overlap_from_previous for current segment `idx`
            current_frame_overlap_from_previous = 0
            if idx == 0 and orchestrator_payload.get("continue_from_video_resolved_path"):
                current_frame_overlap_from_previous = expanded_frame_overlap[0] if expanded_frame_overlap else 0
            elif idx > 0:
                # SM_RESTRUCTURE_FIX_OVERLAP_IDX: Use idx-1 for subsequent segments
                current_frame_overlap_from_previous = expanded_frame_overlap[idx-1] if len(expanded_frame_overlap) > (idx-1) else 0
            
            # VACE refs for this specific segment
            # Ensure vace_refs_instructions_all is a list, default to empty list if None
            vace_refs_safe = vace_refs_instructions_all if vace_refs_instructions_all is not None else []
            vace_refs_for_this_segment = [
                ref_instr for ref_instr in vace_refs_safe
                if ref_instr.get("segment_idx_for_naming") == idx
            ]

                    # [DEEP_DEBUG] Log orchestrator payload values BEFORE creating segment payload
        print(f"[ORCHESTRATOR_DEBUG] {orchestrator_task_id_str}: CREATING SEGMENT {idx} PAYLOAD")
        print(f"[ORCHESTRATOR_DEBUG]   orchestrator_payload.get('apply_causvid'): {orchestrator_payload.get('apply_causvid')}")
        print(f"[ORCHESTRATOR_DEBUG]   orchestrator_payload.get('use_lighti2x_lora'): {orchestrator_payload.get('use_lighti2x_lora')}")
        print(f"[ORCHESTRATOR_DEBUG]   orchestrator_payload.get('apply_reward_lora'): {orchestrator_payload.get('apply_reward_lora')}")
        print(f"[ORCHESTRATOR_DEBUG]   orchestrator_payload.get('additional_loras'): {orchestrator_payload.get('additional_loras')}")
        dprint(f"[DEEP_DEBUG] Orchestrator {orchestrator_task_id_str}: CREATING SEGMENT {idx} PAYLOAD")
        dprint(f"[DEEP_DEBUG]   orchestrator_payload.get('apply_causvid'): {orchestrator_payload.get('apply_causvid')}")
        dprint(f"[DEEP_DEBUG]   orchestrator_payload.get('use_lighti2x_lora'): {orchestrator_payload.get('use_lighti2x_lora')}")
        dprint(f"[DEEP_DEBUG]   orchestrator_payload.get('apply_reward_lora'): {orchestrator_payload.get('apply_reward_lora')}")
        dprint(f"[DEEP_DEBUG]   orchestrator_payload.get('additional_loras'): {orchestrator_payload.get('additional_loras')}")
        
        segment_payload = {
                "task_id": current_segment_task_id,
                "orchestrator_task_id_ref": orchestrator_task_id_str,
                "orchestrator_run_id": run_id,
                "project_id": orchestrator_project_id, # Added project_id
                "segment_index": idx,
                "is_first_segment": (idx == 0),
                "is_last_segment": (idx == num_segments - 1),
                
                "current_run_base_output_dir": str(current_run_output_dir.resolve()), # Base for segment's own output folder creation

                "base_prompt": expanded_base_prompts[idx],
                "negative_prompt": expanded_negative_prompts[idx],
                "segment_frames_target": expanded_segment_frames[idx],
                "frame_overlap_from_previous": current_frame_overlap_from_previous,
                "frame_overlap_with_next": expanded_frame_overlap[idx] if len(expanded_frame_overlap) > idx else 0,
                
                "vace_image_refs_to_prepare_by_worker": vace_refs_for_this_segment, # Already filtered for this segment

                "parsed_resolution_wh": orchestrator_payload["parsed_resolution_wh"],
                "model_name": orchestrator_payload["model_name"],
                "seed_to_use": orchestrator_payload.get("seed_base", 12345) + idx,
                "use_causvid_lora": orchestrator_payload.get("apply_causvid", False),
                "use_lighti2x_lora": orchestrator_payload.get("use_lighti2x_lora", False),
                "apply_reward_lora": orchestrator_payload.get("apply_reward_lora", False),
                "cfg_star_switch": orchestrator_payload.get("cfg_star_switch", 0),
                "cfg_zero_step": orchestrator_payload.get("cfg_zero_step", -1),
                "params_json_str_override": orchestrator_payload.get("params_json_str_override"),
                "fps_helpers": orchestrator_payload.get("fps_helpers", 16),
                "fade_in_params_json_str": orchestrator_payload["fade_in_params_json_str"],
                "fade_out_params_json_str": orchestrator_payload["fade_out_params_json_str"],
                "subsequent_starting_strength_adjustment": orchestrator_payload.get("subsequent_starting_strength_adjustment", 0.0),
                "desaturate_subsequent_starting_frames": orchestrator_payload.get("desaturate_subsequent_starting_frames", 0.0),
                "adjust_brightness_subsequent_starting_frames": orchestrator_payload.get("adjust_brightness_subsequent_starting_frames", 0.0),
                "after_first_post_generation_saturation": orchestrator_payload.get("after_first_post_generation_saturation"),
                "after_first_post_generation_brightness": orchestrator_payload.get("after_first_post_generation_brightness"),
                
                "segment_image_download_dir": segment_image_download_dir_str, # Add the download dir path string
                
                "debug_mode_enabled": orchestrator_payload.get("debug_mode_enabled", False),
                "skip_cleanup_enabled": orchestrator_payload.get("skip_cleanup_enabled", False),
                "continue_from_video_resolved_path_for_guide": orchestrator_payload.get("continue_from_video_resolved_path") if idx == 0 else None,
                "full_orchestrator_payload": orchestrator_payload, # Ensure full payload is passed to segment
                
                # Extract additional_loras to top level for HeadlessTaskQueue processing
                "additional_loras": orchestrator_payload.get("additional_loras", {}),
            }

            # [DEEP_DEBUG] Log segment payload values AFTER creation to verify they match
            print(f"[ORCHESTRATOR_DEBUG] {orchestrator_task_id_str}: SEGMENT {idx} PAYLOAD CREATED")
            print(f"[ORCHESTRATOR_DEBUG]   segment_payload['use_causvid_lora']: {segment_payload.get('use_causvid_lora')}")
            print(f"[ORCHESTRATOR_DEBUG]   segment_payload['use_lighti2x_lora']: {segment_payload.get('use_lighti2x_lora')}")
            print(f"[ORCHESTRATOR_DEBUG]   segment_payload['apply_reward_lora']: {segment_payload.get('apply_reward_lora')}")
            print(f"[ORCHESTRATOR_DEBUG]   segment_payload['additional_loras']: {segment_payload.get('additional_loras')}")
            print(f"[ORCHESTRATOR_DEBUG]   FULL segment_payload keys: {list(segment_payload.keys())}")
            
            # [DEEP_DEBUG] Also log what we're about to send to the Edge Function
            edge_function_payload = {
                "task_id": current_segment_task_id,
                "params": segment_payload,
                "task_type": "travel_segment",
                "project_id": orchestrator_project_id,
                "dependant_on": previous_segment_task_id
            }
            print(f"[ORCHESTRATOR_DEBUG] EDGE FUNCTION PAYLOAD keys: {list(edge_function_payload['params'].keys())}")
            
            dprint(f"[DEEP_DEBUG] Orchestrator {orchestrator_task_id_str}: SEGMENT {idx} PAYLOAD CREATED")
            dprint(f"[DEEP_DEBUG]   segment_payload['use_causvid_lora']: {segment_payload.get('use_causvid_lora')}")
            dprint(f"[DEEP_DEBUG]   segment_payload['use_lighti2x_lora']: {segment_payload.get('use_lighti2x_lora')}")
            dprint(f"[DEEP_DEBUG]   segment_payload['apply_reward_lora']: {segment_payload.get('apply_reward_lora')}")
            dprint(f"[DEEP_DEBUG]   segment_payload['additional_loras']: {segment_payload.get('additional_loras')}")
            dprint(f"[DEEP_DEBUG]   FULL segment_payload keys: {list(segment_payload.keys())}")
            
            # [DEEP_DEBUG] Also log what we're about to send to the Edge Function
            dprint(f"[DEEP_DEBUG] EDGE FUNCTION PAYLOAD keys: {list(edge_function_payload['params'].keys())}")

            dprint(f"Orchestrator: Enqueuing travel_segment {idx} (ID: {current_segment_task_id}) depends_on={previous_segment_task_id}")
            db_ops.add_task_to_db(
                task_payload=segment_payload, 
                task_type_str="travel_segment",
                dependant_on=previous_segment_task_id
            )
            previous_segment_task_id = current_segment_task_id
            dprint(f"Orchestrator {orchestrator_task_id_str}: Enqueued travel_segment {idx} (ID: {current_segment_task_id}) with payload (first 500 chars): {json.dumps(segment_payload, default=str)[:500]}... Depends on: {previous_segment_task_id}")
        
        # After loop, enqueue the stitch task
        stitch_task_id = sm_generate_unique_task_id(f"travel_stitch_{run_id}_")
        final_stitched_video_name = f"travel_final_stitched_{run_id}.mp4"
        # Stitcher saves its final primary output directly under main_output_dir (e.g., ./steerable_motion_output/)
        # NOT under current_run_output_dir (which is .../travel_run_XYZ/)
        # The main_output_dir_base is the one passed to worker.py (e.g. server's ./outputs or steerable_motion's ./steerable_motion_output)
        # The orchestrator_payload["main_output_dir_for_run"] is this main_output_dir_base.
        final_stitched_output_path = Path(orchestrator_payload.get("main_output_dir_for_run", str(main_output_dir_base.resolve()))) / final_stitched_video_name

        stitch_payload = {
            "task_id": stitch_task_id,
            "orchestrator_task_id_ref": orchestrator_task_id_str,
            "orchestrator_run_id": run_id,
            "project_id": orchestrator_project_id, # Added project_id
            "num_total_segments_generated": num_segments,
            "current_run_base_output_dir": str(current_run_output_dir.resolve()), # Stitcher needs this to find segment outputs
            "frame_overlap_settings_expanded": expanded_frame_overlap,
            "crossfade_sharp_amt": orchestrator_payload.get("crossfade_sharp_amt", 0.3),
            "parsed_resolution_wh": orchestrator_payload["parsed_resolution_wh"],
            "fps_final_video": orchestrator_payload.get("fps_helpers", 16),
            "upscale_factor": orchestrator_payload.get("upscale_factor", 0.0),
            "upscale_model_name": orchestrator_payload.get("upscale_model_name"),
            "seed_for_upscale": orchestrator_payload.get("seed_base", 12345) + 5000, # Consistent seed for upscale
            "debug_mode_enabled": orchestrator_payload.get("debug_mode_enabled", False),
            "skip_cleanup_enabled": orchestrator_payload.get("skip_cleanup_enabled", False),
            "initial_continued_video_path": orchestrator_payload.get("continue_from_video_resolved_path"),
            "final_stitched_output_path": str(final_stitched_output_path.resolve()),
             # For upscale polling, if stitcher enqueues an upscale sub-task
            "poll_interval_from_orchestrator": orchestrator_payload.get("original_common_args", {}).get("poll_interval", 15),
            "poll_timeout_from_orchestrator": orchestrator_payload.get("original_common_args", {}).get("poll_timeout", 1800),
            "full_orchestrator_payload": orchestrator_payload, # Added this line
        }
        
        dprint(f"Orchestrator: Enqueuing travel_stitch task (ID: {stitch_task_id}) depends_on={previous_segment_task_id}")
        db_ops.add_task_to_db(
            task_payload=stitch_payload, 
            task_type_str="travel_stitch",
            dependant_on=previous_segment_task_id
        )
        dprint(f"Orchestrator {orchestrator_task_id_str}: Enqueued travel_stitch task (ID: {stitch_task_id}) with payload (first 500 chars): {json.dumps(stitch_payload, default=str)[:500]}... Depends on: {previous_segment_task_id}")

        generation_success = True
        output_message_for_orchestrator_db = f"Successfully enqueued all {num_segments} segment tasks and 1 stitch task for run {run_id}."
        print(f"Orchestrator {orchestrator_task_id_str}: {output_message_for_orchestrator_db}")

    except Exception as e:
        msg = f"[ERROR Task ID: {orchestrator_task_id_str}] Failed during travel orchestration processing: {e}"
        print(msg)
        traceback.print_exc()
        generation_success = False
        output_message_for_orchestrator_db = msg
    
    return generation_success, output_message_for_orchestrator_db

def _handle_travel_segment_task(task_params_from_db: dict, main_output_dir_base: Path, segment_task_id_str: str, apply_reward_lora: bool = False, colour_match_videos: bool = False, mask_active_frames: bool = True, *, process_single_task, dprint, task_queue=None):
    dprint(f"_handle_travel_segment_task: Starting for {segment_task_id_str}")
    dprint(f"Segment task_params_from_db (first 1000 chars): {json.dumps(task_params_from_db, default=str, indent=2)[:1000]}...")
    # task_params_from_db contains what was enqueued for this specific segment,
    # including potentially 'full_orchestrator_payload'.
    segment_params = task_params_from_db 
    generation_success = False # Success of the WGP/Comfy sub-task for this segment
    final_segment_video_output_path_str = None # Output of the WGP sub-task
    output_message_for_segment_task = "Segment task initiated."

    try:
        # --- 1. Initialization & Parameter Extraction ---
        orchestrator_task_id_ref = segment_params.get("orchestrator_task_id_ref")
        orchestrator_run_id = segment_params.get("orchestrator_run_id")
        segment_idx = segment_params.get("segment_index")
        segment_image_download_dir_str = segment_params.get("segment_image_download_dir") # Get the passed dir
        segment_image_download_dir = Path(segment_image_download_dir_str) if segment_image_download_dir_str else None

        if orchestrator_task_id_ref is None or orchestrator_run_id is None or segment_idx is None:
            msg = f"Segment task {segment_task_id_str} missing critical orchestrator refs or segment_index."
            travel_logger.error(msg, task_id=segment_task_id_str)
            return False, msg

        full_orchestrator_payload = segment_params.get("full_orchestrator_payload")
        if not full_orchestrator_payload:
            dprint(f"Segment {segment_idx}: full_orchestrator_payload not in direct params. Querying orchestrator task {orchestrator_task_id_ref}")
            # This logic now uses the db_ops functions
            orchestrator_task_raw_params_json = db_ops.get_task_params(orchestrator_task_id_ref)
            
            if orchestrator_task_raw_params_json:
                try: 
                    fetched_params = json.loads(orchestrator_task_raw_params_json) if isinstance(orchestrator_task_raw_params_json, str) else orchestrator_task_raw_params_json
                    full_orchestrator_payload = fetched_params.get("orchestrator_details")
                    if not full_orchestrator_payload:
                        raise ValueError("'orchestrator_details' key missing in fetched orchestrator task params.")
                    dprint(f"Segment {segment_idx}: Successfully fetched orchestrator_details from DB.")
                except Exception as e_fetch_orc:
                    msg = f"Segment {segment_idx}: Failed to fetch/parse orchestrator_details from DB for task {orchestrator_task_id_ref}: {e_fetch_orc}"
                    print(f"[ERROR Task {segment_task_id_str}]: {msg}")
                    return False, msg
            else:
                msg = f"Segment {segment_idx}: Could not retrieve params for orchestrator task {orchestrator_task_id_ref}. Cannot proceed."
                print(f"[ERROR Task {segment_task_id_str}]: {msg}")
                return False, msg
        
        # Now full_orchestrator_payload is guaranteed to be populated or we've exited.
        # FIX: Prioritize job-specific settings from orchestrator payload over server-wide CLI flags.
        effective_colour_match_enabled = full_orchestrator_payload.get("colour_match_videos", colour_match_videos)
        effective_apply_reward_lora = full_orchestrator_payload.get("apply_reward_lora", apply_reward_lora)
        debug_enabled = segment_params.get("debug_mode_enabled", full_orchestrator_payload.get("debug_mode_enabled", False))

        # Use centralized extraction function (additional_loras already extracted from full_orchestrator_payload)
        additional_loras = full_orchestrator_payload.get("additional_loras", {})
        if additional_loras:
            dprint(f"Segment {segment_idx}: Found additional_loras in orchestrator payload: {additional_loras}")

        current_run_base_output_dir_str = segment_params.get("current_run_base_output_dir")
        if not current_run_base_output_dir_str: # Should be passed by orchestrator/prev segment
            current_run_base_output_dir_str = full_orchestrator_payload.get("main_output_dir_for_run", str(main_output_dir_base.resolve()))
            current_run_base_output_dir_str = str(Path(current_run_base_output_dir_str) / f"travel_run_{orchestrator_run_id}")

        from pathlib import Path  # Ensure Path is available in local scope
        current_run_base_output_dir = Path(current_run_base_output_dir_str)
        # Use the base directory directly without creating segment-specific subdirectories
        segment_processing_dir = current_run_base_output_dir
        segment_processing_dir.mkdir(parents=True, exist_ok=True)

        # ─── Use main processing directory for image downloads (no subfolders) ────────────
        if segment_image_download_dir is None:
            segment_image_download_dir = segment_processing_dir  # Save directly to main output dir
        dprint(f"Segment {segment_idx} (Task {segment_task_id_str}): Processing in {segment_processing_dir.resolve()} | image_download_dir={segment_image_download_dir}")

        # --- Color Match Reference Image Determination ---
        start_ref_path_for_cm, end_ref_path_for_cm = None, None
        if effective_colour_match_enabled:
            input_images_for_cm = full_orchestrator_payload.get("input_image_paths_resolved", [])
            is_continuing_for_cm = full_orchestrator_payload.get("continue_from_video_resolved_path") is not None
            
            if is_continuing_for_cm:
                if segment_idx == 0:
                    continued_video_path = full_orchestrator_payload.get("continue_from_video_resolved_path")
                    if continued_video_path and Path(continued_video_path).exists():
                        dprint(f"Seg {segment_idx} CM: Extracting last frame from {continued_video_path} as start ref.")
                        start_ref_path_for_cm = sm_extract_last_frame_as_image(continued_video_path, segment_processing_dir, segment_task_id_str)
                    if input_images_for_cm:
                        end_ref_path_for_cm = input_images_for_cm[0]
                else: # Subsequent segment when continuing
                    if len(input_images_for_cm) > segment_idx:
                        start_ref_path_for_cm = input_images_for_cm[segment_idx - 1]
                        end_ref_path_for_cm = input_images_for_cm[segment_idx]
            else: # From scratch
                if len(input_images_for_cm) > segment_idx + 1:
                    start_ref_path_for_cm = input_images_for_cm[segment_idx]
                    end_ref_path_for_cm = input_images_for_cm[segment_idx + 1]
            
            # Download images if they are URLs so they exist locally for the color matching function.
            if start_ref_path_for_cm:
                start_ref_path_for_cm = sm_download_image_if_url(
                    start_ref_path_for_cm, 
                    segment_processing_dir,  # Save directly to main output dir, no subfolder
                    segment_task_id_str, 
                    debug_mode=debug_enabled,
                    descriptive_name=f"seg{segment_idx:02d}_start_ref"
                )
            if end_ref_path_for_cm:
                end_ref_path_for_cm = sm_download_image_if_url(
                    end_ref_path_for_cm, 
                    segment_processing_dir,  # Save directly to main output dir, no subfolder
                    segment_task_id_str,
                    debug_mode=debug_enabled,
                    descriptive_name=f"seg{segment_idx:02d}_end_ref"
                )

            dprint(f"Seg {segment_idx} CM Refs: Start='{start_ref_path_for_cm}', End='{end_ref_path_for_cm}'")
        # --- End Color Match Reference Image Determination ---

        # --- Prepare VACE Refs for this Segment (moved to worker) ---
        actual_vace_image_ref_paths_for_wgp = []
        # Get the list of VACE ref instructions from the full orchestrator payload
        vace_ref_instructions_from_orchestrator = full_orchestrator_payload.get("", [])
        
        # Ensure vace_ref_instructions_from_orchestrator is a list, default to empty list if None
        if vace_ref_instructions_from_orchestrator is None:
            vace_ref_instructions_from_orchestrator = []
        
        # Filter instructions for the current segment_idx
        # The segment_idx_for_naming in the instruction should match the current segment_idx
        relevant_vace_instructions = [
            instr for instr in vace_ref_instructions_from_orchestrator
            if instr.get("segment_idx_for_naming") == segment_idx
        ]
        dprint(f"Segment {segment_idx}: Found {len(relevant_vace_instructions)} VACE ref instructions relevant to this segment.")

        if relevant_vace_instructions:
            # Ensure parsed_res_wh is available
            current_parsed_res_wh = full_orchestrator_payload.get("parsed_resolution_wh")
            # If resolution is provided as string (e.g., "512x512"), convert it to tuple[int, int]
            if isinstance(current_parsed_res_wh, str):
                try:
                    parsed_tuple = sm_parse_resolution(current_parsed_res_wh)
                    if parsed_tuple is not None:
                        current_parsed_res_wh = parsed_tuple
                    else:
                        dprint(f"[WARNING] Segment {segment_idx}: Failed to parse resolution string '{current_parsed_res_wh}'. Proceeding with original value.")
                except Exception as e_par:
                    dprint(f"[WARNING] Segment {segment_idx}: Error parsing resolution '{current_parsed_res_wh}': {e_par}. Proceeding with string value (may cause errors).")
            if not current_parsed_res_wh:
                # Fallback or error if resolution not found; for now, dprint and proceed (helper might handle None resolution)
                dprint(f"[WARNING] Segment {segment_idx}: parsed_resolution_wh not found in full_orchestrator_payload. VACE refs might not be resized correctly.")

            for ref_instr in relevant_vace_instructions:
                # Pass segment_image_download_dir to _prepare_vace_ref_for_segment_worker
                dprint(f"Segment {segment_idx}: Preparing VACE ref from instruction: {ref_instr}")
                processed_ref_path = sm_prepare_vace_ref_for_segment(
                    ref_instruction=ref_instr,
                    segment_processing_dir=segment_processing_dir,
                    target_resolution_wh=current_parsed_res_wh,
                    image_download_dir=segment_image_download_dir, # Pass it here
                    task_id_for_logging=segment_task_id_str
                )
                if processed_ref_path:
                    actual_vace_image_ref_paths_for_wgp.append(processed_ref_path)
                    dprint(f"Segment {segment_idx}: Successfully prepared VACE ref: {processed_ref_path}")
                else:
                    dprint(f"Segment {segment_idx}: Failed to prepare VACE ref from instruction: {ref_instr}. It will be omitted.")
        # --- End VACE Ref Preparation ---

        # --- 2. Guide Video Preparation ---
        actual_guide_video_path_for_wgp: Path | None = None
        path_to_previous_segment_video_output_for_guide: str | None = None
        
        is_first_segment = segment_params.get("is_first_segment", segment_idx == 0) # is_first_segment should be reliable
        is_first_segment_from_scratch = is_first_segment and not full_orchestrator_payload.get("continue_from_video_resolved_path")
        is_first_new_segment_after_continue = is_first_segment and full_orchestrator_payload.get("continue_from_video_resolved_path")
        is_subsequent_segment = not is_first_segment

        # Ensure parsed_res_wh is a tuple of integers with model grid snapping
        parsed_res_wh_str = full_orchestrator_payload["parsed_resolution_wh"]
        try:
            parsed_res_raw = sm_parse_resolution(parsed_res_wh_str)
            if parsed_res_raw is None:
                raise ValueError(f"sm_parse_resolution returned None for input: {parsed_res_wh_str}")
            parsed_res_wh = snap_resolution_to_model_grid(parsed_res_raw)
        except Exception as e_parse_res:
            msg = f"Seg {segment_idx}: Invalid format or error parsing parsed_resolution_wh '{parsed_res_wh_str}': {e_parse_res}"
            print(f"[ERROR Task {segment_task_id_str}]: {msg}"); return False, msg
        dprint(f"Segment {segment_idx}: Parsed resolution (w,h): {parsed_res_wh}")

        # --- Single Image Journey Detection ---
        input_images_for_cm_check = full_orchestrator_payload.get("input_image_paths_resolved", [])
        is_single_image_journey = (
            len(input_images_for_cm_check) == 1
            and full_orchestrator_payload.get("continue_from_video_resolved_path") is None
            and segment_params.get("is_first_segment")
            and segment_params.get("is_last_segment")
        )
        if is_single_image_journey:
            dprint(f"Seg {segment_idx}: Detected a single-image journey. Adjusting guide and mask generation.")

        # Calculate total frames for this segment once and reuse
        base_duration = segment_params.get("segment_frames_target", full_orchestrator_payload["segment_frames_expanded"][segment_idx])
        frame_overlap_from_previous = segment_params.get("frame_overlap_from_previous", 0)
        # The user-facing 'segment_frames_target' should represent the total length of the segment,
        # not just the new content. The overlap is handled internally for transition.
        total_frames_for_segment = base_duration

        print(f"[SEGMENT_DEBUG] Segment {segment_idx} (Task {segment_task_id_str}): FRAME ANALYSIS")
        print(f"[SEGMENT_DEBUG]   base_duration (segment_frames_target): {base_duration}")
        print(f"[SEGMENT_DEBUG]   frame_overlap_from_previous: {frame_overlap_from_previous}")
        print(f"[SEGMENT_DEBUG]   total_frames_for_segment: {total_frames_for_segment}")
        print(f"[SEGMENT_DEBUG]   is_first_segment: {segment_params.get('is_first_segment', False)}")
        print(f"[SEGMENT_DEBUG]   is_last_segment: {segment_params.get('is_last_segment', False)}")
        print(f"[SEGMENT_DEBUG]   use_causvid_lora: {full_orchestrator_payload.get('apply_causvid', False)}")

        fps_helpers = full_orchestrator_payload.get("fps_helpers", 16)
        fade_in_duration_str = full_orchestrator_payload["fade_in_params_json_str"]
        fade_out_duration_str = full_orchestrator_payload["fade_out_params_json_str"]
        
        # Define gray_frame_bgr here for use in subsequent segment strength adjustment
        gray_frame_bgr = sm_create_color_frame(parsed_res_wh, (128, 128, 128))



        try: # Parsing fade params
            fade_in_p = json.loads(fade_in_duration_str)
            fi_low, fi_high, fi_curve, fi_factor = float(fade_in_p.get("low_point",0)), float(fade_in_p.get("high_point",1)), str(fade_in_p.get("curve_type","ease_in_out")), float(fade_in_p.get("duration_factor",0))
        except Exception as e_fade_in:
            fi_low, fi_high, fi_curve, fi_factor = 0.0,1.0,"ease_in_out",0.0
            dprint(f"Seg {segment_idx} Warn: Using default fade-in params due to parse error on '{fade_in_duration_str}': {e_fade_in}")
        try:
            fade_out_p = json.loads(fade_out_duration_str)
            fo_low, fo_high, fo_curve, fo_factor = float(fade_out_p.get("low_point",0)), float(fade_out_p.get("high_point",1)), str(fade_out_p.get("curve_type","ease_in_out")), float(fade_out_p.get("duration_factor",0))
        except Exception as e_fade_out:
            fo_low, fo_high, fo_curve, fo_factor = 0.0,1.0,"ease_in_out",0.0
            dprint(f"Seg {segment_idx} Warn: Using default fade-out params due to parse error on '{fade_out_duration_str}': {e_fade_out}")

        if is_first_new_segment_after_continue:
            path_to_previous_segment_video_output_for_guide = full_orchestrator_payload.get("continue_from_video_resolved_path")
            if not path_to_previous_segment_video_output_for_guide or not Path(path_to_previous_segment_video_output_for_guide).exists():
                msg = f"Seg {segment_idx}: Continue video path {path_to_previous_segment_video_output_for_guide} invalid."
                print(f"[ERROR Task {segment_task_id_str}]: {msg}"); return False, msg
        elif is_subsequent_segment:
            # Get predecessor task ID and its output location in a single call using Edge Function (or fallback for SQLite)
            task_dependency_id, raw_path_from_db = db_ops.get_predecessor_output_via_edge_function(segment_task_id_str)
            
            if task_dependency_id and raw_path_from_db:
                dprint(f"Seg {segment_idx}: Task {segment_task_id_str} depends on {task_dependency_id} with output: {raw_path_from_db}")
                # path_to_previous_segment_video_output_for_guide will be relative ("files/...") if from SQLite and stored that way
                # or absolute if from Supabase or stored absolutely in SQLite.
                if db_ops.DB_TYPE == "sqlite" and db_ops.SQLITE_DB_PATH and raw_path_from_db.startswith("files/"):
                    sqlite_db_parent = Path(db_ops.SQLITE_DB_PATH).resolve().parent
                    path_to_previous_segment_video_output_for_guide = str((sqlite_db_parent / "public" / raw_path_from_db).resolve())
                    dprint(f"Seg {segment_idx}: Resolved SQLite relative path from DB '{raw_path_from_db}' to absolute path '{path_to_previous_segment_video_output_for_guide}'")
                else:
                    # Path from DB is already absolute (Supabase) or an old absolute SQLite path
                    path_to_previous_segment_video_output_for_guide = raw_path_from_db
            elif task_dependency_id and not raw_path_from_db:
                dprint(f"Seg {segment_idx}: Found dependency task {task_dependency_id} but no output_location available.")
                path_to_previous_segment_video_output_for_guide = None
            else:
                dprint(f"Seg {segment_idx}: No dependency found for task {segment_task_id_str}. Cannot create guide video based on predecessor.")
                path_to_previous_segment_video_output_for_guide = None
 
            # --- New: Handle Supabase public URLs by downloading them locally for guide processing ---
            if path_to_previous_segment_video_output_for_guide and path_to_previous_segment_video_output_for_guide.startswith("http"):
                try:
                    dprint(f"Seg {segment_idx}: Detected remote URL for previous segment: {path_to_previous_segment_video_output_for_guide}. Downloading...")
                    # Reuse download_file utility from common_utils
                    from ..common_utils import download_file as sm_download_file, sm_get_unique_target_path
                    remote_url = path_to_previous_segment_video_output_for_guide
                    local_filename = Path(remote_url).name
                    # Store under segment_processing_dir to keep things tidy
                    local_download_path = segment_processing_dir / f"prev_{segment_idx:02d}_{local_filename}"
                    # Ensure directory exists
                    segment_processing_dir.mkdir(parents=True, exist_ok=True)
                    # Perform download if file not already present
                    if not local_download_path.exists():
                        sm_download_file(remote_url, segment_processing_dir, local_download_path.name)
                        dprint(f"Seg {segment_idx}: Downloaded previous segment video to {local_download_path}")
                    else:
                        dprint(f"Seg {segment_idx}: Local copy of previous segment video already exists at {local_download_path}")
                    path_to_previous_segment_video_output_for_guide = str(local_download_path.resolve())
                except Exception as e_dl_prev:
                    dprint(f"[WARNING] Seg {segment_idx}: Failed to download remote previous segment video: {e_dl_prev}")
                    # Leave path unchanged – will trigger the existing invalid path error below

            if not path_to_previous_segment_video_output_for_guide or not Path(path_to_previous_segment_video_output_for_guide).exists():
                error_detail_path = raw_path_from_db if 'raw_path_from_db' in locals() and raw_path_from_db else path_to_previous_segment_video_output_for_guide
                msg = f"Seg {segment_idx}: Prev segment output for guide invalid/not found. Expected from prev task output. Path: {error_detail_path}"
                print(f"[ERROR Task {segment_task_id_str}]: {msg}"); return False, msg
        
        # ------------------------------------------------------------------
        #  Show-input-images (banner) – determine start/end images now so the
        #  information can be propagated downstream to the chaining stage.
        # ------------------------------------------------------------------
        show_input_images_enabled = bool(full_orchestrator_payload.get("show_input_images", False))
        start_image_for_banner = None
        end_image_for_banner = None
        if show_input_images_enabled:
            try:
                input_images_resolved_original = full_orchestrator_payload["input_image_paths_resolved"]
                # For banner overlay, always show the first and last images of the entire journey
                # This provides consistent context across all segments
                if len(input_images_resolved_original) > 0:
                    start_image_for_banner = input_images_resolved_original[0]  # Always first image

                if len(input_images_resolved_original) > 1:
                    end_image_for_banner = input_images_resolved_original[-1]  # Always last image
                elif len(input_images_resolved_original) == 1:
                    # Single image journey - use the same image for both
                    end_image_for_banner = input_images_resolved_original[0]

                # Ensure both banner images are local paths (download if URL)
                if start_image_for_banner:
                    start_image_for_banner = sm_download_image_if_url(
                        start_image_for_banner,
                        segment_processing_dir,
                        segment_task_id_str,
                        debug_mode=debug_enabled,
                        descriptive_name="journey_start_image"
                    )
                if end_image_for_banner:
                    end_image_for_banner = sm_download_image_if_url(
                        end_image_for_banner,
                        segment_processing_dir,
                        segment_task_id_str,
                        debug_mode=debug_enabled,
                        descriptive_name="journey_end_image"
                    )

            except Exception as e_banner_sel:
                dprint(f"Seg {segment_idx}: Error selecting banner images for show_input_images: {e_banner_sel}")
        # ------------------------------------------------------------------

        # Use the shared TravelSegmentProcessor to eliminate code duplication
        try:
            from ..travel_segment_processor import TravelSegmentProcessor, TravelSegmentContext
            
            # Create context for the shared processor
            processor_context = TravelSegmentContext(
                task_id=segment_task_id_str,
                segment_idx=segment_idx,
                model_name=full_orchestrator_payload["model_name"],
                total_frames_for_segment=total_frames_for_segment,
                parsed_res_wh=parsed_res_wh,
                segment_processing_dir=segment_processing_dir,
                full_orchestrator_payload=full_orchestrator_payload,
                segment_params=segment_params,
                mask_active_frames=mask_active_frames,
                debug_enabled=debug_enabled,
                dprint=dprint
            )
            
            # Create and use the shared processor
            processor = TravelSegmentProcessor(processor_context)
            segment_outputs = processor.process_segment()
            
            # Extract outputs
            actual_guide_video_path_for_wgp = Path(segment_outputs["video_guide"]) if segment_outputs["video_guide"] else None
            mask_video_path_for_wgp = Path(segment_outputs["video_mask"]) if segment_outputs["video_mask"] else None
            video_prompt_type_str = segment_outputs["video_prompt_type"]
            
            # Get VACE model detection result from processor
            is_vace_model = processor.is_vace_model
            
            dprint(f"[SHARED_PROCESSOR] Seg {segment_idx}: Guide video: {actual_guide_video_path_for_wgp}")
            dprint(f"[SHARED_PROCESSOR] Seg {segment_idx}: Mask video: {mask_video_path_for_wgp}")
            dprint(f"[SHARED_PROCESSOR] Seg {segment_idx}: Video prompt type: {video_prompt_type_str}")
            dprint(f"[SHARED_PROCESSOR] Seg {segment_idx}: Is VACE model: {is_vace_model}")
            
        except Exception as e_shared_processor:
            dprint(f"[ERROR] Seg {segment_idx}: Shared processor failed: {e_shared_processor}")
            traceback.print_exc()
            return False, f"Shared processor failed: {e_shared_processor}", None
        
        # [GUIDE_CONTENT_DEBUG] Log guide and mask video paths for content verification  
        dprint(f"[GUIDE_CONTENT_DEBUG] Seg {segment_idx}: Video guide path: {actual_guide_video_path_for_wgp}")
        dprint(f"[GUIDE_CONTENT_DEBUG] Seg {segment_idx}: Video mask path: {mask_video_path_for_wgp}")
        dprint(f"[GUIDE_CONTENT_DEBUG] Seg {segment_idx}: Empty prompt may cause generic generation - consider adding descriptive prompt")
        
        # --- Invoke WGP Generation directly ---
        if actual_guide_video_path_for_wgp is None and not is_first_segment_from_scratch:
            # If guide creation failed AND it was essential (i.e., for any segment except the very first one from scratch)
            msg = f"Task {segment_task_id_str}: Essential guide video failed to generate. Cannot proceed with WGP processing."
            print(f"[ERROR] {msg}")
            return False, msg
            
        final_frames_for_wgp_generation = total_frames_for_segment
        current_wgp_engine = "wgp" # Defaulting to WGP for travel segments
        
        print(f"[WGP_DEBUG] Segment {segment_idx}: GENERATION PARAMETERS")
        print(f"[WGP_DEBUG]   final_frames_for_wgp_generation: {final_frames_for_wgp_generation}")
        print(f"[WGP_DEBUG]   parsed_res_wh: {parsed_res_wh}")
        print(f"[WGP_DEBUG]   fps_helpers: {fps_helpers}")
        print(f"[WGP_DEBUG]   model_name: {full_orchestrator_payload['model_name']}")
        print(f"[WGP_DEBUG]   use_causvid_lora: {full_orchestrator_payload.get('apply_causvid', False)}")
        
        dprint(f"Task {segment_task_id_str}: Requesting WGP generation with {final_frames_for_wgp_generation} frames.")

        if final_frames_for_wgp_generation <= 0:
            msg = f"Task {segment_task_id_str}: Calculated WGP frames {final_frames_for_wgp_generation}. Cannot generate. Check segment_frames_target and overlap."
            print(f"[ERROR] {msg}")
            return False, msg

        # The WGP task will run with a unique ID, but it's processed in-line now
        wgp_inline_task_id = sm_generate_unique_task_id(f"wgp_inline_{segment_task_id_str[:8]}_")
        
        # Define the absolute final output path for the WGP generation by process_single_task.
        # If DB_TYPE is SQLite, process_single_task will ignore this and save to public/files, returning a relative path.
        # If not SQLite, process_single_task will use this path (or its default construction) and return an absolute path.
        wgp_video_filename = f"{orchestrator_run_id}_seg{segment_idx:02d}_output.mp4"
        # For non-SQLite, wgp_final_output_path_for_this_segment is a suggestion for process_single_task
        # For SQLite, this specific path isn't strictly used by process_single_task for its *final* save, but can be logged.
        wgp_final_output_path_for_this_segment = segment_processing_dir / wgp_video_filename 
        
        safe_vace_image_ref_paths_for_wgp = [str(p.resolve()) if p else None for p in actual_vace_image_ref_paths_for_wgp]
        safe_vace_image_ref_paths_for_wgp = [p for p in safe_vace_image_ref_paths_for_wgp if p is not None]
        
        # If no image refs, pass None instead of empty list to avoid WGP VAE encoder issues
        if not safe_vace_image_ref_paths_for_wgp:
            safe_vace_image_ref_paths_for_wgp = None

        current_segment_base_prompt = segment_params.get("base_prompt", " ")
        prompt_for_wgp = ensure_valid_prompt(current_segment_base_prompt)
        negative_prompt_for_wgp = ensure_valid_negative_prompt(segment_params.get("negative_prompt", " "))
        
        dprint(f"Seg {segment_idx} (Task {segment_task_id_str}): Effective prompt for WGP: '{prompt_for_wgp}'")

        # Get model name for JSON config loading
        model_name = full_orchestrator_payload["model_name"]
        
        # Load model defaults from JSON config BEFORE building wgp_payload
        model_defaults_from_config = {}
        try:
            import sys
            import os
            wan_dir = Path(__file__).parent.parent.parent / "Wan2GP"
            if str(wan_dir) not in sys.path:
                sys.path.insert(0, str(wan_dir))
            
            import wgp
            model_defaults_from_config = wgp.get_default_settings(model_name)
            dprint(f"[MODEL_CONFIG_DEBUG] Segment {segment_idx}: Loaded model defaults for '{model_name}': {model_defaults_from_config}")
        except Exception as e:
            dprint(f"[MODEL_CONFIG_DEBUG] Segment {segment_idx}: Warning - could not load model defaults for '{model_name}': {e}")

        wgp_payload = {
            "task_id": wgp_inline_task_id, # ID for this specific WGP generation operation
            "model": full_orchestrator_payload["model_name"],
            "prompt": prompt_for_wgp, # Use the processed prompt_for_wgp
            "negative_prompt": segment_params["negative_prompt"],
            "resolution": f"{parsed_res_wh[0]}x{parsed_res_wh[1]}", # Use parsed tuple here
            "frames": final_frames_for_wgp_generation,
            "seed": segment_params["seed_to_use"],
            # output_path for process_single_task: 
            # - If SQLite, it's ignored, output goes to public/files, and a relative path is returned.
            # - If not SQLite, this suggested path (or process_single_task's default) is used, and an absolute path is returned.
            "output_path": str(wgp_final_output_path_for_this_segment.resolve()), 
            "video_guide_path": str(actual_guide_video_path_for_wgp.resolve()) if actual_guide_video_path_for_wgp and actual_guide_video_path_for_wgp.exists() else None,
            "use_causvid_lora": full_orchestrator_payload.get("apply_causvid", False),
            "use_lighti2x_lora": full_orchestrator_payload.get("use_lighti2x_lora", False),
            "apply_reward_lora": full_orchestrator_payload.get("apply_reward_lora", False),
            "cfg_star_switch": full_orchestrator_payload.get("cfg_star_switch", 0),
            "cfg_zero_step": full_orchestrator_payload.get("cfg_zero_step", -1),
            "image_refs_paths": safe_vace_image_ref_paths_for_wgp,
            # Propagate video_prompt_type so VACE model correctly interprets guide and mask inputs
            "video_prompt_type": video_prompt_type_str,
            # Attach mask video if available
            **({"video_mask": str(mask_video_path_for_wgp.resolve())} if mask_video_path_for_wgp else {}),
        }
        
        # Add model config defaults to wgp_payload so they're available in parameter precedence chain
        # Note: HeadlessTaskQueue will properly handle task parameter overrides
        if model_defaults_from_config:
            for param in ["guidance_scale", "flow_shift", "num_inference_steps", "switch_threshold"]:
                if param in model_defaults_from_config:
                    wgp_payload[param] = model_defaults_from_config[param]
                    dprint(f"[MODEL_CONFIG_DEBUG] Segment {segment_idx}: Added {param}={model_defaults_from_config[param]} to wgp_payload from model config")
        if additional_loras:
            # Pass additional_loras in dict format - HeadlessTaskQueue handles conversion centrally
            wgp_payload["additional_loras"] = additional_loras
            dprint(f"Seg {segment_idx}: Added {len(additional_loras)} additional LoRAs to payload")

        if full_orchestrator_payload.get("params_json_str_override"):
            try:
                additional_p = json.loads(full_orchestrator_payload["params_json_str_override"])
                # Ensure override cannot change key params that indirectly control output length or resolution
                additional_p.pop("frames", None); additional_p.pop("video_length", None)
                additional_p.pop("steps", None); additional_p.pop("num_inference_steps", None)
                additional_p.pop("resolution", None); additional_p.pop("output_path", None)
                wgp_payload.update(additional_p)
            except Exception as e_json: dprint(f"Error merging override params for WGP payload: {e_json}")
        
        # Add travel_chain_details so process_single_task can call _handle_travel_chaining_after_wgp
        wgp_payload["travel_chain_details"] = {
            "orchestrator_task_id_ref": orchestrator_task_id_ref,
            "orchestrator_run_id": orchestrator_run_id,
            "segment_index_completed": segment_idx, 
            "is_last_segment_in_sequence": segment_params["is_last_segment"], 
            "current_run_base_output_dir": str(current_run_base_output_dir.resolve()),
            "full_orchestrator_payload": full_orchestrator_payload,
            "segment_processing_dir_for_saturation": str(segment_processing_dir.resolve()),
            "is_first_new_segment_after_continue": is_first_new_segment_after_continue,
            "is_subsequent_segment": is_subsequent_segment,
            "colour_match_videos": effective_colour_match_enabled,
            "cm_start_ref_path": start_ref_path_for_cm,
            "cm_end_ref_path": end_ref_path_for_cm,
            "show_input_images": show_input_images_enabled,
            "start_image_path": start_image_for_banner,
            "end_image_path": end_image_for_banner,
        }

        dprint(f"Seg {segment_idx} (Task {segment_task_id_str}): Invoking WGP generation via centralized wrapper (task_id for WGP op: {wgp_inline_task_id})")
        
        # Log VACE control weights if using VACE
        if is_vace_model:
            control_weight = full_orchestrator_payload.get("control_net_weight", 1.0)
            control_weight2 = full_orchestrator_payload.get("control_net_weight2", 1.0)
            dprint(f"[VACEWeights] Seg {segment_idx}: control_net_weight={control_weight}, control_net_weight2={control_weight2}")
        
        # Process additional LoRAs using shared function
        processed_additional_loras = {}
        if additional_loras:
            dprint(f"Seg {segment_idx}: Processing additional LoRAs using shared function")
            # Since legacy system is removed, use task_queue system for LoRA processing
            if task_queue is not None:
                # The new queue system handles LoRAs internally, so just pass them as-is
                processed_additional_loras = additional_loras
                dprint(f"Seg {segment_idx}: Using task_queue system for LoRA processing: {len(processed_additional_loras)} LoRAs")
            else:
                # Fallback: pass LoRAs as-is if no task_queue
                processed_additional_loras = additional_loras
                dprint(f"Seg {segment_idx}: Fallback LoRA processing: {len(processed_additional_loras)} LoRAs")

        # ------------------------------------------------------------------
        # Ensure sensible defaults for critical generation params using shared utilities
        # ------------------------------------------------------------------
        import sys
        from pathlib import Path
        source_dir = Path(__file__).parent.parent
        if str(source_dir) not in sys.path:
            sys.path.insert(0, str(source_dir))
        from lora_utils import detect_lora_optimization_flags, apply_lora_parameter_optimization
        
        model_name = full_orchestrator_payload["model_name"]
        
        # Detect LoRA optimization flags using shared logic
        causvid_enabled, lighti2x_enabled = detect_lora_optimization_flags(
            task_params=segment_params,
            orchestrator_payload=full_orchestrator_payload,
            model_name=model_name,
            dprint=dprint
        )

        # [CausVidDebugTrace] Add detailed parameter precedence logging
        dprint(f"[CausVidDebugTrace] Segment {segment_idx}: Parameter precedence analysis:")
        dprint(f"[CausVidDebugTrace]   causvid_enabled: {causvid_enabled}")
        dprint(f"[CausVidDebugTrace]   lighti2x_enabled: {lighti2x_enabled}")
        dprint(f"[CausVidDebugTrace]   segment_params.get('num_inference_steps'): {segment_params.get('num_inference_steps')}")
        dprint(f"[CausVidDebugTrace]   segment_params.get('steps'): {segment_params.get('steps')}")
        dprint(f"[CausVidDebugTrace]   full_orchestrator_payload.get('num_inference_steps'): {full_orchestrator_payload.get('num_inference_steps')}")
        dprint(f"[CausVidDebugTrace]   full_orchestrator_payload.get('steps'): {full_orchestrator_payload.get('steps')}")
        dprint(f"[CausVidDebugTrace]   wgp_payload.get('num_inference_steps'): {wgp_payload.get('num_inference_steps')}")
        dprint(f"[CausVidDebugTrace]   wgp_payload.get('steps'): {wgp_payload.get('steps')}")
        dprint(f"[CausVidDebugTrace]   default would be: {6 if lighti2x_enabled else (9 if causvid_enabled else 30)}")

        # Apply LoRA parameter optimization using shared logic
        optimized_params = apply_lora_parameter_optimization(
            params=wgp_payload.copy(),  # Work on a copy to avoid modifying original
            causvid_enabled=causvid_enabled,
            lighti2x_enabled=lighti2x_enabled,
            model_name=model_name,
            task_params=segment_params,
            orchestrator_payload=full_orchestrator_payload,
            task_id=segment_task_id_str,
            dprint=dprint
        )
        
        # Extract the optimized values
        num_inference_steps = optimized_params["num_inference_steps"]
        guidance_scale_default = optimized_params["guidance_scale"]
        flow_shift_default = optimized_params["flow_shift"]
        
        dprint(f"[CausVidDebugTrace] Segment {segment_idx}: FINAL SELECTED num_inference_steps = {num_inference_steps}")
        dprint(f"[CausVidDebugTrace] Segment {segment_idx}: FINAL SELECTED guidance_scale = {guidance_scale_default}")
        dprint(f"[CausVidDebugTrace] Segment {segment_idx}: FINAL SELECTED flow_shift = {flow_shift_default}")

        # Parameter defaults now handled in the shared optimization logic above

        # DEPRECATED: Legacy task queue system code - travel segments now processed via direct queue integration
        # Travel segments are now routed through worker.py's _handle_travel_segment_via_queue function
        # which eliminates the blocking wait pattern and provides better model persistence.
        generation_success = False
        wgp_output_path_or_msg = (
            f"DEPRECATED: Travel segment {segment_idx} should be processed via direct queue integration. "
            f"This indicates that worker.py is incorrectly routing travel_segment tasks to the legacy handler "
            f"instead of using _handle_travel_segment_via_queue. Check task routing configuration."
        )

        print(f"[WGP_DEBUG] Segment {segment_idx}: GENERATION RESULT")
        print(f"[WGP_DEBUG]   generation_success: {generation_success}")
        print(f"[WGP_DEBUG]   wgp_output_path_or_msg: {wgp_output_path_or_msg}")
        
        # Analyze the WGP output if successful
        if generation_success and wgp_output_path_or_msg:
            wgp_debug_info = debug_video_analysis(wgp_output_path_or_msg, f"WGP_RAW_OUTPUT_Seg{segment_idx}", segment_task_id_str)
            print(f"[WGP_DEBUG]   Expected frames: {final_frames_for_wgp_generation}")
            print(f"[WGP_DEBUG]   Actual frames: {wgp_debug_info.get('frame_count', 'ERROR')}")
            if wgp_debug_info.get('frame_count') != final_frames_for_wgp_generation:
                print(f"[WGP_DEBUG]   ⚠️  FRAME COUNT MISMATCH! Expected {final_frames_for_wgp_generation}, got {wgp_debug_info.get('frame_count')}")

        if generation_success:
            # Apply post-processing chain (saturation, brightness, color matching)
            chain_success, chain_message, final_chained_path = _handle_travel_chaining_after_wgp(
                wgp_task_params=wgp_payload,
                actual_wgp_output_video_path=wgp_output_path_or_msg,
                image_download_dir=segment_image_download_dir,
                dprint=dprint
            )
            
            if chain_success and final_chained_path:
                final_segment_video_output_path_str = final_chained_path
                output_message_for_segment_task = f"Segment {segment_idx} processing (WGP generation & chaining) completed. Final output: {final_segment_video_output_path_str}"
                
                # Analyze final chained output
                final_debug_info = debug_video_analysis(final_chained_path, f"FINAL_CHAINED_Seg{segment_idx}", segment_task_id_str)
                print(f"[CHAIN_DEBUG] Segment {segment_idx}: FINAL CHAINED OUTPUT ANALYSIS")
                print(f"[CHAIN_DEBUG]   Expected frames: {final_frames_for_wgp_generation}")
                print(f"[CHAIN_DEBUG]   Final frames: {final_debug_info.get('frame_count', 'ERROR')}")
                if final_debug_info.get('frame_count') != final_frames_for_wgp_generation:
                    print(f"[CHAIN_DEBUG]   ⚠️  CHAINING CHANGED FRAME COUNT! Expected {final_frames_for_wgp_generation}, got {final_debug_info.get('frame_count')}")
            else:
                # Use raw WGP output if chaining failed
                final_segment_video_output_path_str = wgp_output_path_or_msg
                output_message_for_segment_task = f"Segment {segment_idx} WGP completed but chaining failed: {chain_message}. Using raw output: {final_segment_video_output_path_str}"
                print(f"[WARNING] {output_message_for_segment_task}")
                
                # Analyze raw output being used as final
                if wgp_output_path_or_msg:
                    raw_debug_info = debug_video_analysis(wgp_output_path_or_msg, f"RAW_AS_FINAL_Seg{segment_idx}", segment_task_id_str)
            
            print(f"Seg {segment_idx} (Task {segment_task_id_str}): {output_message_for_segment_task}")
        else:
            # wgp_output_path_or_msg contains the error message if generation_success is False
            final_segment_video_output_path_str = None 
            output_message_for_segment_task = f"Segment {segment_idx} (Task {segment_task_id_str}) processing (WGP generation) failed. Error: {wgp_output_path_or_msg}"
            print(f"[ERROR] {output_message_for_segment_task}")
            
            # Notify orchestrator of segment failure
            try:
                db_ops.update_task_status(
                    orchestrator_task_id_ref,
                    db_ops.STATUS_FAILED,
                    output_message_for_segment_task[:500]  # Truncate to avoid DB overflow
                )
                dprint(f"Segment {segment_idx}: Marked orchestrator task {orchestrator_task_id_ref} as FAILED due to WGP generation failure")
            except Exception as e_orch:
                dprint(f"Segment {segment_idx}: Warning - could not update orchestrator status: {e_orch}")
        
        # The old polling logic is no longer needed as process_single_task is synchronous here.

        # The return value final_segment_video_output_path_str (if success) is the one that
        # process_single_task itself would have set as 'output_location' for the WGP task.
        # Now, it becomes the output_location for the parent travel_segment task.
        return generation_success, final_segment_video_output_path_str if generation_success else output_message_for_segment_task

    except Exception as e:
        print(f"ERROR Task {segment_task_id_str}: Unexpected error during segment processing: {e}")
        traceback.print_exc()
        
        # Notify orchestrator of segment failure
        if 'orchestrator_task_id_ref' in locals() and orchestrator_task_id_ref:
            try:
                error_msg = f"Segment {segment_idx if 'segment_idx' in locals() else 'unknown'} failed: {str(e)[:200]}"
                db_ops.update_task_status(
                    orchestrator_task_id_ref,
                    db_ops.STATUS_FAILED,
                    error_msg
                )
                dprint(f"Segment: Marked orchestrator task {orchestrator_task_id_ref} as FAILED due to exception")
            except Exception as e_orch:
                dprint(f"Segment: Warning - could not update orchestrator status: {e_orch}")
        
        # return False, f"Unexpected error: {str(e)[:200]}"
        return False, f"Segment {segment_idx if 'segment_idx' in locals() else 'unknown'} failed: {str(e)[:200]}"

# --- SM_RESTRUCTURE: New function to handle chaining after WGP/Comfy sub-task ---
def _handle_travel_chaining_after_wgp(wgp_task_params: dict, actual_wgp_output_video_path: str | None, image_download_dir: Path | str | None = None, *, dprint) -> tuple[bool, str, str | None]:
    """
    Handles the chaining logic after a WGP  sub-task for a travel segment completes.
    This includes post-generation saturation and enqueuing the next segment or stitch task.
    Returns: (success_bool, message_str, final_video_path_for_db_str_or_none)
    The third element is the path that should be considered the definitive output of the WGP task
    (e.g., path to saturated video if saturation was applied).
    """
    chain_details = wgp_task_params.get("travel_chain_details")
    wgp_task_id = wgp_task_params.get("task_id", "unknown_wgp_task")

    if not chain_details:
        return False, f"Task {wgp_task_id}: Missing travel_chain_details. Cannot proceed with chaining.", None
    
    # actual_wgp_output_video_path comes from process_single_task.
    # If DB_TYPE is sqlite, it will be like "files/wgp_output.mp4".
    # Otherwise, it's an absolute path.
    if not actual_wgp_output_video_path: # Check if it's None or empty string
        return False, f"Task {wgp_task_id}: WGP output video path is None or empty. Cannot chain.", None

    # This variable will track the absolute path of the video as it gets processed.
    video_to_process_abs_path: Path
    # This will hold the path to be stored in the DB (can be relative for SQLite).
    final_video_path_for_db = actual_wgp_output_video_path

    # Resolve initial absolute path
    if db_ops.DB_TYPE == "sqlite" and db_ops.SQLITE_DB_PATH and isinstance(actual_wgp_output_video_path, str) and actual_wgp_output_video_path.startswith("files/"):
        sqlite_db_parent = Path(db_ops.SQLITE_DB_PATH).resolve().parent
        video_to_process_abs_path = sqlite_db_parent / "public" / actual_wgp_output_video_path
    else:
        video_to_process_abs_path = Path(actual_wgp_output_video_path)

    if not video_to_process_abs_path.exists():
        return False, f"Task {wgp_task_id}: Source video for chaining '{video_to_process_abs_path}' (from '{actual_wgp_output_video_path}') does not exist.", actual_wgp_output_video_path

    try:
        orchestrator_task_id_ref = chain_details["orchestrator_task_id_ref"]
        orchestrator_run_id = chain_details["orchestrator_run_id"]
        segment_idx_completed = chain_details["segment_index_completed"]
        full_orchestrator_payload = chain_details["full_orchestrator_payload"]
        segment_processing_dir_for_saturation_str = chain_details["segment_processing_dir_for_saturation"]
        
        is_first_new_segment_after_continue = chain_details.get("is_first_new_segment_after_continue", False)
        is_subsequent_segment_val = chain_details.get("is_subsequent_segment", False)

        dprint(f"Chaining for WGP task {wgp_task_id} (segment {segment_idx_completed} of run {orchestrator_run_id}). Initial video: {video_to_process_abs_path}")

        # --- Always move WGP output to proper location first ---
        # Use consistent UUID-based naming and MOVE (not copy) to avoid duplicates
        timestamp_short = datetime.now().strftime("%H%M%S")
        unique_suffix = uuid.uuid4().hex[:6]
        moved_filename = f"seg{segment_idx_completed:02d}_output_{timestamp_short}_{unique_suffix}{video_to_process_abs_path.suffix}"
        
        # Use segment processing dir directly without task_id prefixing since UUID guarantees uniqueness
        moved_video_abs_path = Path(segment_processing_dir_for_saturation_str) / moved_filename
        moved_video_abs_path.parent.mkdir(parents=True, exist_ok=True)
        
        # MOVE (not copy) the WGP output to avoid creating duplicates
        try:
            # Ensure encoder has finished writing the source file
            sm_wait_for_file_stable(video_to_process_abs_path, checks=3, interval=1.0, dprint=dprint)

            shutil.move(str(video_to_process_abs_path), str(moved_video_abs_path))
            print(f"[CHAIN_DEBUG] Moved WGP output from {video_to_process_abs_path} to {moved_video_abs_path}")
            debug_video_analysis(moved_video_abs_path, f"MOVED_WGP_OUTPUT_Seg{segment_idx_completed}", wgp_task_id)
            dprint(f"Chain (Seg {segment_idx_completed}): Moved WGP output from {video_to_process_abs_path} to {moved_video_abs_path}")
            
            # Update paths for further processing
            video_to_process_abs_path = moved_video_abs_path
            final_video_path_for_db = str(moved_video_abs_path)  # Use absolute path as DB path
            
            # No cleanup needed since we moved (not copied) the file
            dprint(f"Chain (Seg {segment_idx_completed}): WGP output successfully moved to final location")
                    
        except Exception as e_move:
            dprint(f"Chain (Seg {segment_idx_completed}): Warning - could not move WGP output to proper location: {e_move}. Using original path.")
            # If move failed, keep original paths for further processing
            final_video_path_for_db = str(video_to_process_abs_path)

        # --- Post-generation Processing Chain ---
        # Saturation and Brightness are only applied to segments AFTER the first one.
        if is_subsequent_segment_val or is_first_new_segment_after_continue:

            # --- 1. Saturation ---
            sat_level = full_orchestrator_payload.get("after_first_post_generation_saturation")
            if sat_level is not None and isinstance(sat_level, (float, int)) and sat_level >= 0.0 and abs(sat_level - 1.0) > 1e-6:
                dprint(f"Chain (Seg {segment_idx_completed}): Applying post-gen saturation {sat_level} to {video_to_process_abs_path}")

                sat_filename = f"s{segment_idx_completed}_sat_{sat_level:.2f}{video_to_process_abs_path.suffix}"
                saturated_video_output_abs_path, new_db_path = prepare_output_path(
                    task_id=wgp_task_id,
                    filename=sat_filename,
                    main_output_dir_base=Path(segment_processing_dir_for_saturation_str)
                )
                
                if sm_apply_saturation_to_video_ffmpeg(str(video_to_process_abs_path), saturated_video_output_abs_path, sat_level):
                    print(f"[CHAIN_DEBUG] Saturation applied successfully to segment {segment_idx_completed}")
                    debug_video_analysis(saturated_video_output_abs_path, f"SATURATED_Seg{segment_idx_completed}", wgp_task_id)
                    dprint(f"Chain (Seg {segment_idx_completed}): Saturation successful. New path: {new_db_path}")
                    _cleanup_intermediate_video(full_orchestrator_payload, video_to_process_abs_path, segment_idx_completed, "raw", dprint)
                    
                    video_to_process_abs_path = saturated_video_output_abs_path
                    final_video_path_for_db = new_db_path
                else:
                    print(f"[CHAIN_DEBUG] WARNING: Saturation failed for segment {segment_idx_completed}")
                    dprint(f"[WARNING] Chain (Seg {segment_idx_completed}): Saturation failed. Continuing with unsaturated video.")
            
            # --- 2. Brightness ---
            brightness_adjust = full_orchestrator_payload.get("after_first_post_generation_brightness", 0.0)
            if isinstance(brightness_adjust, (float, int)) and abs(brightness_adjust) > 1e-6:
                dprint(f"Chain (Seg {segment_idx_completed}): Applying post-gen brightness {brightness_adjust} to {video_to_process_abs_path}")
                
                bright_filename = f"s{segment_idx_completed}_bright_{brightness_adjust:+.2f}{video_to_process_abs_path.suffix}"
                brightened_video_output_abs_path, new_db_path = prepare_output_path(
                    task_id=wgp_task_id,
                    filename=bright_filename,
                    main_output_dir_base=Path(segment_processing_dir_for_saturation_str)
                )
                
                processed_video = apply_brightness_to_video_frames(str(video_to_process_abs_path), brightened_video_output_abs_path, brightness_adjust, wgp_task_id)

                if processed_video and processed_video.exists():
                    print(f"[CHAIN_DEBUG] Brightness adjustment applied successfully to segment {segment_idx_completed}")
                    debug_video_analysis(brightened_video_output_abs_path, f"BRIGHTENED_Seg{segment_idx_completed}", wgp_task_id)
                    dprint(f"Chain (Seg {segment_idx_completed}): Brightness adjustment successful. New path: {new_db_path}")
                    _cleanup_intermediate_video(full_orchestrator_payload, video_to_process_abs_path, segment_idx_completed, "saturated", dprint)

                    video_to_process_abs_path = brightened_video_output_abs_path
                    final_video_path_for_db = new_db_path
                else:
                    print(f"[CHAIN_DEBUG] WARNING: Brightness adjustment failed for segment {segment_idx_completed}")
                    dprint(f"[WARNING] Chain (Seg {segment_idx_completed}): Brightness adjustment failed. Continuing with previous video version.")

        # --- 3. Color Matching (Applied to all segments if enabled) ---
        if chain_details.get("colour_match_videos"):
            start_ref = chain_details.get("cm_start_ref_path")
            end_ref = chain_details.get("cm_end_ref_path")
            print(f"[CHAIN_DEBUG] Color matching requested for segment {segment_idx_completed}")
            print(f"[CHAIN_DEBUG] Start ref: {start_ref}")
            print(f"[CHAIN_DEBUG] End ref: {end_ref}")
            dprint(f"Chain (Seg {segment_idx_completed}): Color matching requested. Start Ref: {start_ref}, End Ref: {end_ref}")

            if start_ref and end_ref and Path(start_ref).exists() and Path(end_ref).exists():
                cm_filename = f"s{segment_idx_completed}_colormatched{video_to_process_abs_path.suffix}"
                cm_video_output_abs_path, new_db_path = prepare_output_path(
                    task_id=wgp_task_id,
                    filename=cm_filename,
                    main_output_dir_base=Path(segment_processing_dir_for_saturation_str)
                )

                matched_video_path = sm_apply_color_matching_to_video(
                    str(video_to_process_abs_path),
                    start_ref,
                    end_ref,
                    str(cm_video_output_abs_path),
                    dprint
                )

                if matched_video_path and Path(matched_video_path).exists():
                    print(f"[CHAIN_DEBUG] Color matching applied successfully to segment {segment_idx_completed}")
                    debug_video_analysis(Path(matched_video_path), f"COLORMATCHED_Seg{segment_idx_completed}", wgp_task_id)
                    dprint(f"Chain (Seg {segment_idx_completed}): Color matching successful. New path: {new_db_path}")
                    _cleanup_intermediate_video(full_orchestrator_payload, video_to_process_abs_path, segment_idx_completed, "pre-colormatch", dprint)

                    video_to_process_abs_path = Path(matched_video_path)
                    final_video_path_for_db = new_db_path
                else:
                    print(f"[CHAIN_DEBUG] WARNING: Color matching failed for segment {segment_idx_completed}")
                    dprint(f"[WARNING] Chain (Seg {segment_idx_completed}): Color matching failed. Continuing with previous video version.")
            else:
                print(f"[CHAIN_DEBUG] WARNING: Color matching skipped - missing or invalid reference images")
                dprint(f"[WARNING] Chain (Seg {segment_idx_completed}): Skipping color matching due to missing or invalid reference image paths.")

        # --- 4. Optional: Overlay start/end images above the video ---
        if chain_details.get("show_input_images"):
            banner_start = chain_details.get("start_image_path")
            banner_end = chain_details.get("end_image_path")
            if banner_start and banner_end and Path(banner_start).exists() and Path(banner_end).exists():
                banner_filename = f"s{segment_idx_completed}_with_inputs{video_to_process_abs_path.suffix}"
                banner_video_abs_path, new_db_path = prepare_output_path(
                    task_id=wgp_task_id,
                    filename=banner_filename,
                    main_output_dir_base=Path(segment_processing_dir_for_saturation_str)
                )

                if sm_overlay_start_end_images_above_video(
                    start_image_path=banner_start,
                    end_image_path=banner_end,
                    input_video_path=str(video_to_process_abs_path),
                    output_video_path=str(banner_video_abs_path),
                    dprint=dprint,
                ):
                    print(f"[CHAIN_DEBUG] Banner overlay applied successfully to segment {segment_idx_completed}")
                    debug_video_analysis(banner_video_abs_path, f"BANNER_OVERLAY_Seg{segment_idx_completed}", wgp_task_id)
                    dprint(f"Chain (Seg {segment_idx_completed}): Banner overlay successful. New path: {new_db_path}")
                    _cleanup_intermediate_video(full_orchestrator_payload, video_to_process_abs_path, segment_idx_completed, "pre-banner", dprint)

                    video_to_process_abs_path = banner_video_abs_path
                    final_video_path_for_db = new_db_path
                else:
                    print(f"[CHAIN_DEBUG] WARNING: Banner overlay failed for segment {segment_idx_completed}")
                    dprint(f"[WARNING] Chain (Seg {segment_idx_completed}): Banner overlay failed. Keeping previous video version.")
            else:
                print(f"[CHAIN_DEBUG] WARNING: Banner overlay skipped - missing valid start/end images")
                dprint(f"[WARNING] Chain (Seg {segment_idx_completed}): show_input_images enabled but valid start/end images not found.")

        # The orchestrator has already enqueued all segment and stitch tasks.
        print(f"[CHAIN_DEBUG] Chaining complete for segment {segment_idx_completed}")
        print(f"[CHAIN_DEBUG] Final video path for DB: {final_video_path_for_db}")
        debug_video_analysis(video_to_process_abs_path, f"FINAL_CHAINED_Seg{segment_idx_completed}", wgp_task_id)
        msg = f"Chain (Seg {segment_idx_completed}): Post-WGP processing complete. Final path for this WGP task's output: {final_video_path_for_db}"
        dprint(msg)
        return True, msg, str(final_video_path_for_db)

    except Exception as e_chain:
        error_msg = f"Chain (Seg {chain_details.get('segment_index_completed', 'N/A')} for WGP {wgp_task_id}): Failed during chaining: {e_chain}"
        print(f"[ERROR] {error_msg}")
        traceback.print_exc()
        
        # Notify orchestrator of chaining failure
        orchestrator_task_id_ref = chain_details.get("orchestrator_task_id_ref") if chain_details else None
        if orchestrator_task_id_ref:
            try:
                db_ops.update_task_status(
                    orchestrator_task_id_ref,
                    db_ops.STATUS_FAILED,
                    error_msg[:500]  # Truncate to avoid DB overflow
                )
                dprint(f"Chain: Marked orchestrator task {orchestrator_task_id_ref} as FAILED due to chaining failure")
            except Exception as e_orch:
                dprint(f"Chain: Warning - could not update orchestrator status: {e_orch}")
        
        return False, error_msg, str(final_video_path_for_db) # Return path as it was before error



def _cleanup_intermediate_video(orchestrator_payload, video_path: Path, segment_idx: int, stage: str, dprint):
    """Helper to cleanup intermediate video files during chaining."""
    # Delete intermediates **only** when every cleanup-bypass flag is false.
    # That now includes the worker-server global debug flag (db_ops.debug_mode)
    # so that running the server with --debug automatically preserves files.
    if (
        not orchestrator_payload.get("skip_cleanup_enabled", False)
        and not orchestrator_payload.get("debug_mode_enabled", False)
        and not db_ops.debug_mode
        and video_path.exists()
    ):
        try:
            video_path.unlink()
            dprint(f"Chain (Seg {segment_idx}): Removed intermediate '{stage}' video {video_path}")
        except Exception as e_del:
            dprint(f"Chain (Seg {segment_idx}): Warning - could not remove intermediate video {video_path}: {e_del}")

def _handle_travel_stitch_task(task_params_from_db: dict, main_output_dir_base: Path, stitch_task_id_str: str, *, dprint):
    print(f"[IMMEDIATE DEBUG] _handle_travel_stitch_task: Starting for {stitch_task_id_str}")
    print(f"[IMMEDIATE DEBUG] task_params_from_db keys: {list(task_params_from_db.keys())}")
    print(f"[IMMEDIATE DEBUG] DB_TYPE: {db_ops.DB_TYPE}")
    
    dprint(f"_handle_travel_stitch_task: Starting for {stitch_task_id_str}")
    dprint(f"Stitch task_params_from_db (first 1000 chars): {json.dumps(task_params_from_db, default=str, indent=2)[:1000]}...")
    stitch_params = task_params_from_db # This now contains full_orchestrator_payload
    stitch_success = False
    final_video_location_for_db = None
    
    try:
        # --- 1. Initialization & Parameter Extraction --- 
        orchestrator_task_id_ref = stitch_params.get("orchestrator_task_id_ref")
        orchestrator_run_id = stitch_params.get("orchestrator_run_id")
        full_orchestrator_payload = stitch_params.get("full_orchestrator_payload")

        print(f"[IMMEDIATE DEBUG] orchestrator_run_id: {orchestrator_run_id}")
        print(f"[IMMEDIATE DEBUG] orchestrator_task_id_ref: {orchestrator_task_id_ref}")
        print(f"[IMMEDIATE DEBUG] full_orchestrator_payload present: {full_orchestrator_payload is not None}")

        if not all([orchestrator_task_id_ref, orchestrator_run_id, full_orchestrator_payload]):
            msg = f"Stitch task {stitch_task_id_str} missing critical orchestrator refs or full_orchestrator_payload."
            travel_logger.error(msg, task_id=stitch_task_id_str)
            return False, msg

        project_id_for_stitch = stitch_params.get("project_id")
        current_run_base_output_dir_str = stitch_params.get("current_run_base_output_dir", 
                                                            full_orchestrator_payload.get("main_output_dir_for_run", str(main_output_dir_base.resolve())))
        current_run_base_output_dir = Path(current_run_base_output_dir_str)
        
        # Use the base directory directly without creating stitch-specific subdirectories
        stitch_processing_dir = current_run_base_output_dir
        stitch_processing_dir.mkdir(parents=True, exist_ok=True)
        dprint(f"Stitch Task {stitch_task_id_str}: Processing in {stitch_processing_dir.resolve()}")

        num_expected_new_segments = full_orchestrator_payload["num_new_segments_to_generate"]
        print(f"[IMMEDIATE DEBUG] num_expected_new_segments: {num_expected_new_segments}")
        
        # Ensure parsed_res_wh is a tuple of integers for stitch task with model grid snapping
        parsed_res_wh_str = full_orchestrator_payload["parsed_resolution_wh"]
        try:
            parsed_res_raw = sm_parse_resolution(parsed_res_wh_str)
            if parsed_res_raw is None:
                raise ValueError(f"sm_parse_resolution returned None for input: {parsed_res_wh_str}")
            parsed_res_wh = snap_resolution_to_model_grid(parsed_res_raw)
        except Exception as e_parse_res_stitch:
            msg = f"Stitch Task {stitch_task_id_str}: Invalid format or error parsing parsed_resolution_wh '{parsed_res_wh_str}': {e_parse_res_stitch}"
            print(f"[ERROR Task {stitch_task_id_str}]: {msg}"); return False, msg
        dprint(f"Stitch Task {stitch_task_id_str}: Parsed resolution (w,h): {parsed_res_wh}")

        final_fps = full_orchestrator_payload.get("fps_helpers", 16)
        expanded_frame_overlaps = full_orchestrator_payload["frame_overlap_expanded"]
        crossfade_sharp_amt = full_orchestrator_payload.get("crossfade_sharp_amt", 0.3)
        initial_continued_video_path_str = full_orchestrator_payload.get("continue_from_video_resolved_path")

        # [OVERLAP DEBUG] Add detailed debug for overlap values
        print(f"[OVERLAP DEBUG] Stitch: expanded_frame_overlaps from payload: {expanded_frame_overlaps}")
        dprint(f"[OVERLAP DEBUG] Stitch: expanded_frame_overlaps from payload: {expanded_frame_overlaps}")

        # Extract upscale parameters
        upscale_factor = full_orchestrator_payload.get("upscale_factor", 0.0) # Default to 0.0 if not present
        upscale_model_name = full_orchestrator_payload.get("upscale_model_name") # Default to None if not present

        # --- 2. Collect Paths to All Segment Videos --- 
        segment_video_paths_for_stitch = []
        if initial_continued_video_path_str and Path(initial_continued_video_path_str).exists():
            dprint(f"Stitch: Prepending initial continued video: {initial_continued_video_path_str}")
            # Check the continue video properties
            cap = cv2.VideoCapture(str(initial_continued_video_path_str))
            if cap.isOpened():
                continue_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                continue_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                continue_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                dprint(f"Stitch: Continue video properties - Resolution: {continue_width}x{continue_height}, Frames: {continue_frame_count}")
                dprint(f"Stitch: Target resolution for stitching: {parsed_res_wh[0]}x{parsed_res_wh[1]}")
                if continue_width != parsed_res_wh[0] or continue_height != parsed_res_wh[1]:
                    dprint(f"Stitch: WARNING - Continue video resolution mismatch! Will need resizing during crossfade.")
            else:
                dprint(f"Stitch: ERROR - Could not open continue video for property check")
            segment_video_paths_for_stitch.append(str(Path(initial_continued_video_path_str).resolve()))
        
        # Fetch completed segments with a small retry loop to handle race conditions
        max_stitch_fetch_retries = 6  # Allow up to ~18s total wait
        completed_segment_outputs_from_db = []
        
        print(f"[IMMEDIATE DEBUG] About to start retry loop for run_id: {orchestrator_run_id}")
        
        for attempt in range(max_stitch_fetch_retries):
            print(f"[IMMEDIATE DEBUG] Stitch fetch attempt {attempt+1}/{max_stitch_fetch_retries} for run_id: {orchestrator_run_id}")
            dprint(f"[DEBUG] Stitch fetch attempt {attempt+1}/{max_stitch_fetch_retries} for run_id: {orchestrator_run_id}")
            
            try:
                completed_segment_outputs_from_db = db_ops.get_completed_segment_outputs_for_stitch(orchestrator_run_id, project_id=project_id_for_stitch) or []
                print(f"[IMMEDIATE DEBUG] DB query returned: {completed_segment_outputs_from_db}")
            except Exception as e_db_query:
                print(f"[IMMEDIATE DEBUG] DB query failed: {e_db_query}")
                completed_segment_outputs_from_db = []
            
            dprint(f"[DEBUG] Attempt {attempt+1} returned {len(completed_segment_outputs_from_db)} segments")
            print(f"[IMMEDIATE DEBUG] Attempt {attempt+1} returned {len(completed_segment_outputs_from_db)} segments")
            
            if len(completed_segment_outputs_from_db) >= num_expected_new_segments:
                dprint(f"[DEBUG] Expected {num_expected_new_segments} segment rows found on attempt {attempt+1}. Proceeding.")
                print(f"[IMMEDIATE DEBUG] Expected {num_expected_new_segments} segment rows found on attempt {attempt+1}. Proceeding.")
                break
            dprint(f"Stitch: No completed segment rows found (attempt {attempt+1}/{max_stitch_fetch_retries}). Waiting 3s and retrying...")
            print(f"[IMMEDIATE DEBUG] Insufficient segments found (attempt {attempt+1}/{max_stitch_fetch_retries}). Waiting 3s and retrying...")
            if attempt < max_stitch_fetch_retries - 1:  # Don't sleep after the last attempt
                time.sleep(3)
        dprint(f"Stitch Task {stitch_task_id_str}: Completed segments fetched: {completed_segment_outputs_from_db}")
        print(f"[IMMEDIATE DEBUG] Final completed_segment_outputs_from_db: {completed_segment_outputs_from_db}")

        # ------------------------------------------------------------------
        # 2b. Resolve each returned video path (local, SQLite-relative, or URL)
        # ------------------------------------------------------------------
        print(f"[STITCH_DEBUG] Starting path resolution for {len(completed_segment_outputs_from_db)} segments")
        print(f"[STITCH_DEBUG] Raw DB results: {completed_segment_outputs_from_db}")
        dprint(f"[DEBUG] Starting path resolution for {len(completed_segment_outputs_from_db)} segments")
        for seg_idx, video_path_str_from_db in completed_segment_outputs_from_db:
            print(f"[STITCH_DEBUG] Processing segment {seg_idx} with path: {video_path_str_from_db}")
            dprint(f"[DEBUG] Processing segment {seg_idx} with path: {video_path_str_from_db}")
            resolved_video_path_for_stitch: Path | None = None

            if not video_path_str_from_db:
                print(f"[STITCH_DEBUG] WARNING: Segment {seg_idx} has empty video_path in DB; skipping.")
                dprint(f"[WARNING] Stitch: Segment {seg_idx} has empty video_path in DB; skipping.")
                continue

            # Case A: Relative path that starts with files/ (works for both sqlite and supabase when worker has local access)
            if video_path_str_from_db.startswith("files/") or video_path_str_from_db.startswith("public/files/"):
                print(f"[STITCH_DEBUG] Case A: Relative path detected for segment {seg_idx}")
                sqlite_db_parent = None
                if db_ops.SQLITE_DB_PATH:
                    sqlite_db_parent = Path(db_ops.SQLITE_DB_PATH).resolve().parent
                else:
                    # Fall back: examine cwd and assume standard layout ../public
                    try:
                        sqlite_db_parent = Path.cwd()
                    except Exception:
                        sqlite_db_parent = Path(".")
                absolute_path_candidate = (sqlite_db_parent / "public" / video_path_str_from_db.lstrip("public/")).resolve()
                print(f"[STITCH_DEBUG] Resolved relative path '{video_path_str_from_db}' to '{absolute_path_candidate}' for segment {seg_idx}")
                dprint(f"Stitch: Resolved relative path '{video_path_str_from_db}' to '{absolute_path_candidate}' for segment {seg_idx}")
                if absolute_path_candidate.exists() and absolute_path_candidate.is_file():
                    resolved_video_path_for_stitch = absolute_path_candidate
                    print(f"[STITCH_DEBUG] ✅ File exists at resolved path")
                else:
                    print(f"[STITCH_DEBUG] ❌ File missing at resolved path")
                    dprint(f"[WARNING] Stitch: Resolved absolute path '{absolute_path_candidate}' for segment {seg_idx} is missing.")

            # Case B: Remote public URL (Supabase storage)
            elif video_path_str_from_db.startswith("http"):
                print(f"[STITCH_DEBUG] Case B: Remote URL detected for segment {seg_idx}")
                try:
                    from ..common_utils import download_file as sm_download_file
                    remote_url = video_path_str_from_db
                    local_filename = Path(remote_url).name
                    local_download_path = stitch_processing_dir / f"seg{seg_idx:02d}_{local_filename}"
                    print(f"[STITCH_DEBUG] Remote URL: {remote_url}")
                    print(f"[STITCH_DEBUG] Local download path: {local_download_path}")
                    dprint(f"[DEBUG] Remote URL detected, local download path: {local_download_path}")
                    
                    # Check if cached file exists and validate its frame count against orchestrator's expected values
                    need_download = True
                    if local_download_path.exists():
                        print(f"[STITCH_DEBUG] Local copy exists, validating frame count...")
                        try:
                            cached_frames, _ = sm_get_video_frame_count_and_fps(str(local_download_path))
                            expected_segment_frames = full_orchestrator_payload["segment_frames_expanded"]
                            expected_frames = expected_segment_frames[seg_idx] if seg_idx < len(expected_segment_frames) else None
                            print(f"[STITCH_DEBUG] Cached file has {cached_frames} frames (expected: {expected_frames})")
                            
                            if expected_frames and cached_frames == expected_frames:
                                print(f"[STITCH_DEBUG] ✅ Cached file frame count matches expected ({cached_frames} frames)")
                                need_download = False
                            elif expected_frames:
                                print(f"[STITCH_DEBUG] ❌ Cached file frame count mismatch! Expected {expected_frames}, got {cached_frames}, will re-download")
                            else:
                                print(f"[STITCH_DEBUG] ❌ No expected frame count available for segment {seg_idx}, will re-download")
                        except Exception as e_validate:
                            print(f"[STITCH_DEBUG] ❌ Could not validate cached file: {e_validate}, will re-download")
                    
                    if need_download:
                        print(f"[STITCH_DEBUG] Downloading remote segment {seg_idx}...")
                        dprint(f"Stitch: Downloading remote segment {seg_idx} from {remote_url} to {local_download_path}")
                        # Remove stale cached file if it exists
                        if local_download_path.exists():
                            local_download_path.unlink()
                        sm_download_file(remote_url, stitch_processing_dir, local_download_path.name)
                        print(f"[STITCH_DEBUG] ✅ Download completed for segment {seg_idx}")
                        dprint(f"[DEBUG] Download completed for segment {seg_idx}")
                    else:
                        print(f"[STITCH_DEBUG] ✅ Using validated cached file for segment {seg_idx}")
                        dprint(f"Stitch: Using validated cached file for segment {seg_idx} at {local_download_path}")
                    
                    resolved_video_path_for_stitch = local_download_path
                except Exception as e_dl:
                    print(f"[STITCH_DEBUG] ❌ Download failed for segment {seg_idx}: {e_dl}")
                    dprint(f"[WARNING] Stitch: Failed to download remote video for segment {seg_idx}: {e_dl}")

            # Case C: Provided absolute/local path
            else:
                print(f"[STITCH_DEBUG] Case C: Absolute/local path for segment {seg_idx}")
                absolute_path_candidate = Path(video_path_str_from_db).resolve()
                print(f"[STITCH_DEBUG] Treating as absolute path: {absolute_path_candidate}")
                dprint(f"[DEBUG] Treating as absolute path: {absolute_path_candidate}")
                if absolute_path_candidate.exists() and absolute_path_candidate.is_file():
                    resolved_video_path_for_stitch = absolute_path_candidate
                    print(f"[STITCH_DEBUG] ✅ Absolute path exists")
                    dprint(f"[DEBUG] Absolute path exists: {absolute_path_candidate}")
                else:
                    print(f"[STITCH_DEBUG] ❌ Absolute path missing or not a file")
                    dprint(f"[WARNING] Stitch: Absolute path '{absolute_path_candidate}' for segment {seg_idx} does not exist or is not a file.")

            if resolved_video_path_for_stitch is not None:
                segment_video_paths_for_stitch.append(str(resolved_video_path_for_stitch))
                print(f"[STITCH_DEBUG] ✅ Added video for segment {seg_idx}: {resolved_video_path_for_stitch}")
                dprint(f"Stitch: Added video for segment {seg_idx}: {resolved_video_path_for_stitch}")
                
                # Analyze the resolved video immediately
                debug_video_analysis(resolved_video_path_for_stitch, f"RESOLVED_Seg{seg_idx}", stitch_task_id_str)
            else: 
                print(f"[STITCH_DEBUG] ❌ Unable to resolve video for segment {seg_idx}; will be excluded from stitching.")
                dprint(f"[WARNING] Stitch: Unable to resolve video for segment {seg_idx}; will be excluded from stitching.")

        print(f"[STITCH_DEBUG] Path resolution complete")
        print(f"[STITCH_DEBUG] Final segment_video_paths_for_stitch: {segment_video_paths_for_stitch}")
        print(f"[STITCH_DEBUG] Total videos collected: {len(segment_video_paths_for_stitch)}")
        dprint(f"[DEBUG] Final segment_video_paths_for_stitch: {segment_video_paths_for_stitch}")
        dprint(f"[DEBUG] Total videos collected: {len(segment_video_paths_for_stitch)}")
        # [CRITICAL DEBUG] Log each video's frame count before stitching
        print(f"[CRITICAL DEBUG] About to stitch videos:")
        expected_segment_frames = full_orchestrator_payload["segment_frames_expanded"]
        for idx, video_path in enumerate(segment_video_paths_for_stitch):
            try:
                frame_count, fps = sm_get_video_frame_count_and_fps(video_path)
                expected_frames = expected_segment_frames[idx] if idx < len(expected_segment_frames) else "unknown"
                print(f"[CRITICAL DEBUG] Video {idx}: {video_path} -> {frame_count} frames @ {fps} FPS (expected: {expected_frames})")
                if expected_frames != "unknown" and frame_count != expected_frames:
                    print(f"[CRITICAL DEBUG] ⚠️  FRAME COUNT MISMATCH! Expected {expected_frames}, got {frame_count}")
            except Exception as e_debug:
                print(f"[CRITICAL DEBUG] Video {idx}: {video_path} -> ERROR: {e_debug}")

        total_videos_for_stitch = (1 if initial_continued_video_path_str and Path(initial_continued_video_path_str).exists() else 0) + num_expected_new_segments
        dprint(f"[DEBUG] Expected total videos: {total_videos_for_stitch}")
        if len(segment_video_paths_for_stitch) < total_videos_for_stitch:
            # This is a warning because some segments might have legitimately failed and been skipped by their handlers.
            # The stitcher should proceed with what it has, unless it has zero or one video when multiple were expected.
            dprint(f"[WARNING] Stitch: Expected {total_videos_for_stitch} videos for stitch, but found {len(segment_video_paths_for_stitch)}. Stitching with available videos.")
        
        if not segment_video_paths_for_stitch:
            dprint(f"[ERROR] Stitch: No valid segment videos found to stitch. DB returned {len(completed_segment_outputs_from_db)} segments, but none resolved to valid paths.")
            raise ValueError("Stitch: No valid segment videos found to stitch.")
        if len(segment_video_paths_for_stitch) == 1 and total_videos_for_stitch > 1:
            dprint(f"Stitch: Only one video segment found ({segment_video_paths_for_stitch[0]}) but {total_videos_for_stitch} were expected. Using this single video as the 'stitched' output.")
            # No actual stitching needed, just move/copy this single video to final dest.

        # --- 3. Stitching (Crossfade or Concatenate) --- 
        current_stitched_video_path: Path | None = None # This will hold the path to the current version of the stitched video


        if len(segment_video_paths_for_stitch) == 1:
            # If only one video, copy it directly using prepare_output_path
            source_single_video_path = Path(segment_video_paths_for_stitch[0])
            single_video_filename = f"{orchestrator_run_id}_final{source_single_video_path.suffix}"
            
            current_stitched_video_path, _ = prepare_output_path(
                task_id=stitch_task_id_str,
                filename=single_video_filename,
                main_output_dir_base=stitch_processing_dir
            )
            shutil.copy2(str(source_single_video_path), str(current_stitched_video_path))
            dprint(f"Stitch: Only one video found. Copied {source_single_video_path} to {current_stitched_video_path}")
        else: # More than one video, proceed with stitching logic
            num_stitch_points = len(segment_video_paths_for_stitch) - 1
            actual_overlaps_for_stitching = []
            if initial_continued_video_path_str: 
                actual_overlaps_for_stitching = expanded_frame_overlaps[:num_stitch_points] 
            else: 
                actual_overlaps_for_stitching = expanded_frame_overlaps[:num_stitch_points]
            
            # --- NEW OVERLAP DEBUG LOGGING ---
            print(f"[OVERLAP DEBUG] Number of videos: {len(segment_video_paths_for_stitch)} (expected stitch points: {num_stitch_points})")
            print(f"[OVERLAP DEBUG] actual_overlaps_for_stitching: {actual_overlaps_for_stitching}")
            if len(actual_overlaps_for_stitching) != num_stitch_points:
                print(f"[OVERLAP DEBUG] ⚠️  MISMATCH! We have {len(actual_overlaps_for_stitching)} overlaps for {num_stitch_points} joins")
            for join_idx, ov in enumerate(actual_overlaps_for_stitching):
                print(f"[OVERLAP DEBUG]   Join {join_idx} (video {join_idx} -> {join_idx+1}): overlap={ov}")
            # --- END NEW LOGGING ---
            
            any_positive_overlap = any(o > 0 for o in actual_overlaps_for_stitching)

            raw_stitched_video_filename = f"{orchestrator_run_id}_stitched.mp4"
            path_for_raw_stitched_video, _ = prepare_output_path(
                task_id=stitch_task_id_str,
                filename=raw_stitched_video_filename,
                main_output_dir_base=stitch_processing_dir
            )

            if any_positive_overlap:
                print(f"[CRITICAL DEBUG] Using cross-fade due to overlap values: {actual_overlaps_for_stitching}. Output to: {path_for_raw_stitched_video}")
                print(f"[STITCH_ANALYSIS] Cross-fade stitching analysis:")
                print(f"[STITCH_ANALYSIS]   Number of videos: {len(segment_video_paths_for_stitch)}")
                print(f"[STITCH_ANALYSIS]   Overlap values: {actual_overlaps_for_stitching}")
                print(f"[STITCH_ANALYSIS]   Expected stitch points: {num_stitch_points}")
                
                dprint(f"Stitch: Using cross-fade due to overlap values: {actual_overlaps_for_stitching}. Output to: {path_for_raw_stitched_video}")
                all_segment_frames_lists = [sm_extract_frames_from_video(p, dprint_func=dprint) for p in segment_video_paths_for_stitch]
                
                # [CRITICAL DEBUG] Log frame extraction results
                print(f"[CRITICAL DEBUG] Frame extraction results:")
                for idx, frame_list in enumerate(all_segment_frames_lists):
                    if frame_list is not None:
                        print(f"[CRITICAL DEBUG] Segment {idx}: {len(frame_list)} frames extracted")
                    else:
                        print(f"[CRITICAL DEBUG] Segment {idx}: FAILED to extract frames")
                
                if not all(f_list is not None and len(f_list)>0 for f_list in all_segment_frames_lists):
                    raise ValueError("Stitch: Frame extraction failed for one or more segments during cross-fade prep.")
                
                final_stitched_frames = []
                
                # Process each stitch point
                for i in range(num_stitch_points): 
                    frames_prev_segment = all_segment_frames_lists[i]
                    frames_curr_segment = all_segment_frames_lists[i+1]
                    current_overlap_val = actual_overlaps_for_stitching[i]

                    print(f"[CRITICAL DEBUG] Stitch point {i}: segments {i}->{i+1}, overlap={current_overlap_val}")
                    print(f"[CRITICAL DEBUG] Prev segment: {len(frames_prev_segment)} frames, Curr segment: {len(frames_curr_segment)} frames")

                    # --- NEW OVERLAP DETAIL LOG ---
                    if current_overlap_val > 0:
                        start_prev = len(frames_prev_segment) - current_overlap_val
                        end_prev = len(frames_prev_segment) - 1
                        start_curr = 0
                        end_curr = current_overlap_val - 1
                        print(
                            f"[OVERLAP_DETAIL] Join {i}: blending prev[{start_prev}:{end_prev}] with curr[{start_curr}:{end_curr}] (total {current_overlap_val} frames)"
                        )
                    # --- END OVERLAP DETAIL LOG ---

                    if i == 0:
                        # For the first stitch point, add frames from segment 0 up to the overlap
                        if current_overlap_val > 0:
                            # Add frames before the overlap region
                            frames_before_overlap = frames_prev_segment[:-current_overlap_val]
                            final_stitched_frames.extend(frames_before_overlap)
                            print(f"[CRITICAL DEBUG] Added {len(frames_before_overlap)} frames from segment 0 (before overlap)")
                        else:
                            # No overlap, add all frames from segment 0
                            final_stitched_frames.extend(frames_prev_segment)
                            print(f"[CRITICAL DEBUG] Added all {len(frames_prev_segment)} frames from segment 0 (no overlap)")
                    else:
                        pass

                    if current_overlap_val > 0:
                        # Remove the overlap frames already appended from the previous segment so that
                        # they can be replaced by the blended cross-fade frames for this stitch point.
                        if i > 0:
                            frames_to_remove = min(current_overlap_val, len(final_stitched_frames))
                            if frames_to_remove > 0:
                                del final_stitched_frames[-frames_to_remove:]
                                print(f"[CRITICAL DEBUG] Removed {frames_to_remove} duplicate overlap frames before cross-fade (stitch point {i})")
                        # Blend the overlapping frames
                        faded_frames = sm_cross_fade_overlap_frames(frames_prev_segment, frames_curr_segment, current_overlap_val, "linear_sharp", crossfade_sharp_amt)
                        final_stitched_frames.extend(faded_frames)
                        print(f"[CRITICAL DEBUG] Added {len(faded_frames)} cross-faded frames")
                    
                    # Add the non-overlapping part of the current segment
                    start_index_for_curr_tail = current_overlap_val
                    if len(frames_curr_segment) > start_index_for_curr_tail:
                        frames_to_add = frames_curr_segment[start_index_for_curr_tail:]
                        final_stitched_frames.extend(frames_to_add)
                        print(f"[CRITICAL DEBUG] Added {len(frames_to_add)} frames from segment {i+1} (after overlap)")
                    
                    print(f"[CRITICAL DEBUG] Running total after stitch point {i}: {len(final_stitched_frames)} frames")
                
                if not final_stitched_frames: raise ValueError("Stitch: No frames produced after cross-fade logic.")
                
                # [CRITICAL DEBUG] Final calculation summary
                # With proper cross-fade: output = sum(all frames) - sum(overlaps)
                # Because overlapped frames are blended, not duplicated
                total_input_frames = sum(len(frames) for frames in all_segment_frames_lists)
                total_overlaps = sum(actual_overlaps_for_stitching)
                expected_output_frames = total_input_frames - total_overlaps
                actual_output_frames = len(final_stitched_frames)
                print(f"[CRITICAL DEBUG] FINAL CROSS-FADE SUMMARY:")
                print(f"[CRITICAL DEBUG] Total input frames: {total_input_frames}")
                print(f"[CRITICAL DEBUG] Total overlaps: {total_overlaps}")
                print(f"[CRITICAL DEBUG] Expected output: {expected_output_frames}")
                print(f"[CRITICAL DEBUG] Actual output: {actual_output_frames}")
                print(f"[CRITICAL DEBUG] Match: {expected_output_frames == actual_output_frames}")
                
                created_video_path_obj = sm_create_video_from_frames_list(final_stitched_frames, path_for_raw_stitched_video, final_fps, parsed_res_wh)
                if created_video_path_obj and created_video_path_obj.exists():
                    current_stitched_video_path = created_video_path_obj
                else:
                    raise RuntimeError(f"Stitch: Cross-fade sm_create_video_from_frames_list failed to produce video at {path_for_raw_stitched_video}")

            else: 
                dprint(f"Stitch: Using simple FFmpeg concatenation. Output to: {path_for_raw_stitched_video}")
                try:
                    from ..common_utils import stitch_videos_ffmpeg as sm_stitch_videos_ffmpeg
                except ImportError:
                    print(f"[CRITICAL ERROR Task ID: {stitch_task_id_str}] Failed to import 'stitch_videos_ffmpeg'. Cannot proceed with stitching.")
                    raise

                if sm_stitch_videos_ffmpeg(segment_video_paths_for_stitch, str(path_for_raw_stitched_video)):
                    current_stitched_video_path = path_for_raw_stitched_video
                else: 
                    raise RuntimeError(f"Stitch: Simple FFmpeg concatenation failed for output {path_for_raw_stitched_video}.")

        if not current_stitched_video_path or not current_stitched_video_path.exists():
            raise RuntimeError(f"Stitch: Stitching process failed, output video not found at {current_stitched_video_path}")
        
        video_path_after_optional_upscale = current_stitched_video_path

        if isinstance(upscale_factor, (float, int)) and upscale_factor > 1.0 and upscale_model_name:
            print(f"[STITCH UPSCALE] Starting upscale process: {upscale_factor}x using model {upscale_model_name}")
            dprint(f"Stitch: Upscaling (x{upscale_factor}) video {current_stitched_video_path.name} using model {upscale_model_name}")
            
            original_frames_count, original_fps = sm_get_video_frame_count_and_fps(str(current_stitched_video_path))
            if original_frames_count is None or original_frames_count == 0:
                raise ValueError(f"Stitch: Cannot get frame count or 0 frames for video {current_stitched_video_path} before upscaling.")
            
            print(f"[STITCH UPSCALE] Input video: {original_frames_count} frames @ {original_fps} FPS")
            print(f"[STITCH UPSCALE] Target resolution: {int(parsed_res_wh[0] * upscale_factor)}x{int(parsed_res_wh[1] * upscale_factor)}")
            dprint(f"[DEBUG] Pre-upscale analysis: {original_frames_count} frames, {original_fps} FPS")

            target_width_upscaled = int(parsed_res_wh[0] * upscale_factor)
            target_height_upscaled = int(parsed_res_wh[1] * upscale_factor)
            
            upscale_sub_task_id = sm_generate_unique_task_id(f"upscale_stitch_{orchestrator_run_id}_")
            
            upscale_payload = {
                "task_id": upscale_sub_task_id,
                "project_id": stitch_params.get("project_id"),
                "model": upscale_model_name,
                "video_source_path": str(current_stitched_video_path.resolve()), 
                "resolution": f"{target_width_upscaled}x{target_height_upscaled}",
                "frames": original_frames_count,
                "prompt": full_orchestrator_payload.get("original_task_args",{}).get("upscale_prompt", "cinematic, masterpiece, high detail, 4k"), 
                "seed": full_orchestrator_payload.get("seed_for_upscale", full_orchestrator_payload.get("seed_base", 12345) + 5000),
            }
            
            db_path_for_upscale_add = db_ops.SQLITE_DB_PATH if db_ops.DB_TYPE == "sqlite" else None
            upscaler_engine_to_use = stitch_params.get("execution_engine_for_upscale", "wgp")
            
            db_ops.add_task_to_db(
                task_payload=upscale_payload, 
                task_type_str=upscaler_engine_to_use
            )
            print(f"[STITCH UPSCALE] Enqueued upscale sub-task {upscale_sub_task_id} ({upscaler_engine_to_use}). Waiting...")
            print(f"Stitch Task {stitch_task_id_str}: Enqueued upscale sub-task {upscale_sub_task_id} ({upscaler_engine_to_use}). Waiting...")
            
            poll_interval_ups = full_orchestrator_payload.get("poll_interval", 15)
            poll_timeout_ups = full_orchestrator_payload.get("poll_timeout_upscale", full_orchestrator_payload.get("poll_timeout", 30 * 60) * 2)
            
            print(f"[STITCH UPSCALE] Polling for completion (timeout: {poll_timeout_ups}s, interval: {poll_interval_ups}s)")
            
            upscaled_video_db_location = db_ops.poll_task_status(
                task_id=upscale_sub_task_id, 
                poll_interval_seconds=poll_interval_ups, 
                timeout_seconds=poll_timeout_ups
            )
            print(f"[STITCH UPSCALE] Poll result: {upscaled_video_db_location}")
            dprint(f"Stitch Task {stitch_task_id_str}: Upscale sub-task {upscale_sub_task_id} poll result: {upscaled_video_db_location}")

            if upscaled_video_db_location:
                upscaled_video_abs_path: Path
                if db_ops.DB_TYPE == "sqlite" and db_ops.SQLITE_DB_PATH and upscaled_video_db_location.startswith("files/"):
                    sqlite_db_parent = Path(db_ops.SQLITE_DB_PATH).resolve().parent
                    upscaled_video_abs_path = sqlite_db_parent / "public" / upscaled_video_db_location
                else: 
                    upscaled_video_abs_path = Path(upscaled_video_db_location)

                if upscaled_video_abs_path.exists():
                    print(f"[STITCH UPSCALE] Upscale completed successfully: {upscaled_video_abs_path}")
                    dprint(f"Stitch: Upscale sub-task {upscale_sub_task_id} completed. Output: {upscaled_video_abs_path}")
                    
                    # Analyze upscaled result
                    try:
                        upscaled_frame_count, upscaled_fps = sm_get_video_frame_count_and_fps(str(upscaled_video_abs_path))
                        print(f"[STITCH UPSCALE] Upscaled result: {upscaled_frame_count} frames @ {upscaled_fps} FPS")
                        dprint(f"[DEBUG] Post-upscale analysis: {upscaled_frame_count} frames, {upscaled_fps} FPS")
                        
                        # Compare frame counts
                        if upscaled_frame_count != original_frames_count:
                            print(f"[STITCH UPSCALE] Frame count changed during upscale: {original_frames_count} → {upscaled_frame_count}")
                    except Exception as e_post_upscale:
                        print(f"[WARNING] Could not analyze upscaled video: {e_post_upscale}")
                    
                    video_path_after_optional_upscale = upscaled_video_abs_path
                    
                    if not full_orchestrator_payload.get("skip_cleanup_enabled", False) and \
                       not full_orchestrator_payload.get("debug_mode_enabled", False) and \
                       current_stitched_video_path.exists() and current_stitched_video_path != video_path_after_optional_upscale:
                        try:
                            current_stitched_video_path.unlink()
                            dprint(f"Stitch: Removed non-upscaled video {current_stitched_video_path} after successful upscale.")
                        except Exception as e_del_non_upscaled:
                            dprint(f"Stitch: Warning - could not remove non-upscaled video {current_stitched_video_path}: {e_del_non_upscaled}")
                else: 
                    print(f"[STITCH UPSCALE] ERROR: Upscale output missing at {upscaled_video_abs_path}. Using non-upscaled video.")
                    print(f"[WARNING] Stitch Task {stitch_task_id_str}: Upscale sub-task {upscale_sub_task_id} output missing ({upscaled_video_abs_path}). Using non-upscaled video.")
            else: 
                print(f"[STITCH UPSCALE] ERROR: Upscale sub-task failed or timed out. Using non-upscaled video.")
                print(f"[WARNING] Stitch Task {stitch_task_id_str}: Upscale sub-task {upscale_sub_task_id} failed or timed out. Using non-upscaled video.")

        elif upscale_factor > 1.0 and not upscale_model_name:
            print(f"[STITCH UPSCALE] Upscale factor {upscale_factor} > 1.0 but no upscale_model_name provided. Skipping upscale.")
            dprint(f"Stitch: Upscale factor {upscale_factor} > 1.0 but no upscale_model_name provided. Skipping upscale.")
        else:
            print(f"[STITCH UPSCALE] No upscaling requested (factor: {upscale_factor})")
            dprint(f"Stitch: No upscaling (factor: {upscale_factor})")

        # Use consistent UUID-based naming for final video
        timestamp_short = datetime.now().strftime("%H%M%S")
        unique_suffix = uuid.uuid4().hex[:6]
        if upscale_factor > 1.0:
            final_video_filename = f"travel_final_upscaled_{upscale_factor:.1f}x_{timestamp_short}_{unique_suffix}{video_path_after_optional_upscale.suffix}"
        else:
            final_video_filename = f"travel_final_{timestamp_short}_{unique_suffix}{video_path_after_optional_upscale.suffix}"
        
        final_video_path, initial_db_location = prepare_output_path_with_upload(
            task_id=stitch_task_id_str,
            filename=final_video_filename,
            main_output_dir_base=stitch_processing_dir,
            dprint=dprint
        )
        
        # Move the video to final location if it's not already there
        if video_path_after_optional_upscale.resolve() != final_video_path.resolve():
            dprint(f"Stitch Task {stitch_task_id_str}: Moving {video_path_after_optional_upscale} to {final_video_path}")
            shutil.move(str(video_path_after_optional_upscale), str(final_video_path))
        else:
            dprint(f"Stitch Task {stitch_task_id_str}: Video already at final destination {final_video_path}")
        
        # Handle Supabase upload (if configured) and get final location for DB
        final_video_location_for_db = upload_and_get_final_output_location(
            final_video_path,
            final_video_filename,  # Pass only the filename to avoid redundant subfolder
            initial_db_location,
            dprint=dprint
        )
        
        print(f"Stitch Task {stitch_task_id_str}: Final video saved to: {final_video_path} (DB location: {final_video_location_for_db})")
        
        # Analyze final result
        try:
            final_frame_count, final_fps = sm_get_video_frame_count_and_fps(str(final_video_path))
            final_duration = final_frame_count / final_fps if final_fps > 0 else 0
            print(f"[STITCH FINAL] Final video: {final_frame_count} frames @ {final_fps} FPS = {final_duration:.2f}s")
            print(f"[STITCH_FINAL_ANALYSIS] Complete stitching analysis:")
            print(f"[STITCH_FINAL_ANALYSIS]   Input segments: {len(segment_video_paths_for_stitch)}")
            print(f"[STITCH_FINAL_ANALYSIS]   Overlap settings: {expanded_frame_overlaps}")
            # Calculate expected final length for analysis
            try:
                # Try to calculate expected final length from orchestrator data
                expected_segment_frames = full_orchestrator_payload.get("segment_frames_expanded", [])
                if expected_segment_frames:
                    total_input_frames = sum(expected_segment_frames)
                    total_overlaps = sum(expanded_frame_overlaps)
                    expected_final_length = total_input_frames - total_overlaps
                    print(f"[STITCH_FINAL_ANALYSIS]   Expected final frames: {expected_final_length}")
                    print(f"[STITCH_FINAL_ANALYSIS]   Actual final frames: {final_frame_count}")
                    if final_frame_count != expected_final_length:
                        print(f"[STITCH_FINAL_ANALYSIS]   ⚠️  FINAL LENGTH MISMATCH! Expected {expected_final_length}, got {final_frame_count}")
                else:
                    print(f"[STITCH_FINAL_ANALYSIS]   Expected final frames: Not available (no segment_frames_expanded)")
                    print(f"[STITCH_FINAL_ANALYSIS]   Actual final frames: {final_frame_count}")
            except Exception as e:
                print(f"[STITCH_FINAL_ANALYSIS]   Expected final frames: Not calculated ({e})")
                print(f"[STITCH_FINAL_ANALYSIS]   Actual final frames: {final_frame_count}")
            
            # Detailed analysis of the final video
            debug_video_analysis(final_video_path, "FINAL_STITCHED_VIDEO", stitch_task_id_str)
            
            dprint(f"[DEBUG] Final video analysis: {final_frame_count} frames, {final_fps} FPS, {final_duration:.2f}s duration")
        except Exception as e_final_analysis:
            print(f"[WARNING] Could not analyze final video: {e_final_analysis}")
        
        # Note: Individual segments already have banner overlays applied when show_input_images is enabled,
        # so the stitched video will automatically include them. No additional overlay needed here.
        
        stitch_success = True
        
        # Note: The orchestrator will be marked as complete by the Edge Function
        # when it processes the stitch task upload. This ensures atomic completion
        # with the final video upload.
        print(f"[ORCHESTRATOR_COMPLETION_DEBUG] Stitch task complete. Orchestrator {orchestrator_task_id_ref} will be marked complete by Edge Function.")
        dprint(f"Stitch: Task complete. Orchestrator completion will be handled by Edge Function.")
        
        # Return the final video path so the stitch task itself gets uploaded via Edge Function
        return stitch_success, str(final_video_path.resolve())

    except Exception as e:
        travel_logger.error(f"Stitch: Unexpected error during stitching: {e}", task_id=stitch_task_id_str)
        travel_logger.debug(traceback.format_exc(), task_id=stitch_task_id_str)
        
        # Notify orchestrator of stitch failure
        if 'orchestrator_task_id_ref' in locals() and orchestrator_task_id_ref:
            try:
                error_msg = f"Stitch task failed: {str(e)[:200]}"
                db_ops.update_task_status(
                    orchestrator_task_id_ref,
                    db_ops.STATUS_FAILED,
                    error_msg
                )
                dprint(f"Stitch: Marked orchestrator task {orchestrator_task_id_ref} as FAILED due to exception")
            except Exception as e_orch:
                dprint(f"Stitch: Warning - could not update orchestrator status: {e_orch}")
        
        return False, f"Stitch task failed: {str(e)[:200]}"
