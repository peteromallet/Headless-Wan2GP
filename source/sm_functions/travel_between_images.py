import json
import math
import shutil
import traceback
from pathlib import Path

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
)

# --- SM_RESTRUCTURE: New Handler Functions for Travel Tasks ---
def _handle_travel_orchestrator_task(task_params_from_db: dict, main_output_dir_base: Path, orchestrator_task_id_str: str, orchestrator_project_id: str | None, *, dprint):
    dprint(f"_handle_travel_orchestrator_task: Starting for {orchestrator_task_id_str}")
    dprint(f"Orchestrator Project ID: {orchestrator_project_id}") # Added dprint
    dprint(f"Orchestrator task_params_from_db (first 1000 chars): {json.dumps(task_params_from_db, default=str, indent=2)[:1000]}...")
    generation_success = False # Represents success of orchestration step
    output_message_for_orchestrator_db = f"Orchestration for {orchestrator_task_id_str} initiated."

    try:
        if 'orchestrator_details' not in task_params_from_db:
            msg = f"[ERROR Task ID: {orchestrator_task_id_str}] 'orchestrator_details' not found in task_params_from_db."
            print(msg)
            return False, msg
        
        orchestrator_payload = task_params_from_db['orchestrator_details']
        dprint(f"Orchestrator payload for {orchestrator_task_id_str} (first 500 chars): {json.dumps(orchestrator_payload, indent=2, default=str)[:500]}...")

        run_id = orchestrator_payload.get("run_id", orchestrator_task_id_str)
        base_dir_for_this_run_str = orchestrator_payload.get("main_output_dir_for_run", str(main_output_dir_base.resolve()))
        current_run_output_dir = Path(base_dir_for_this_run_str) / f"travel_run_{run_id}"
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
        vace_refs_instructions_all = orchestrator_payload.get("vace_image_refs_to_prepare_by_headless", [])

        # --- SM_QUANTIZE_FRAMES_AND_OVERLAPS ---
        # Adjust all segment lengths to match model constraints (4*N+1 format).
        # Then, adjust overlap values to be even and not exceed the length of the
        # smaller of the two segments they connect. This prevents errors downstream
        # in guide video creation, generation, and stitching.
        
        quantized_segment_frames = []
        dprint(f"Orchestrator: Quantizing frame counts. Original segment_frames_expanded: {expanded_segment_frames}")
        for i, frames in enumerate(expanded_segment_frames):
            # Quantize to 4*N+1 format to match model constraints, applied later in headless.py
            new_frames = (frames // 4) * 4 + 1
            if new_frames != frames:
                dprint(f"Orchestrator: Quantized segment {i} length from {frames} to {new_frames} (4*N+1 format).")
            quantized_segment_frames.append(new_frames)
        dprint(f"Orchestrator: Finished quantizing frame counts. New quantized_segment_frames: {quantized_segment_frames}")
        
        quantized_frame_overlap = []
        # There are N-1 overlaps for N segments. The loop must not iterate more times than this.
        num_overlaps_to_process = len(quantized_segment_frames) - 1

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

                if new_overlap != original_overlap:
                    dprint(f"Orchestrator: Adjusted overlap between segments {i}-{i+1} from {original_overlap} to {new_overlap}.")
                
                quantized_frame_overlap.append(new_overlap)
        
        # Replace original lists with the new quantized ones for all subsequent logic
        expanded_segment_frames = quantized_segment_frames
        expanded_frame_overlap = quantized_frame_overlap
        # --- END SM_QUANTIZE_FRAMES_AND_OVERLAPS ---

        for idx in range(num_segments):
            current_segment_task_id = sm_generate_unique_task_id(f"travel_seg_{run_id}_{idx:02d}_")
            
            # Define where this segment's specific processing assets will go and its final output
            # Segment handler will create this directory.
            segment_processing_dir_name = f"segment_{idx:02d}_{current_segment_task_id[:8]}"
            # Define where the segment's final output video should be placed by the segment handler itself
            # This path needs to be absolute for the segment handler.
            final_video_output_path_for_segment = current_run_output_dir / segment_processing_dir_name / f"s{idx}_final_output.mp4"

            # Determine frame_overlap_from_previous for current segment `idx`
            current_frame_overlap_from_previous = 0
            if idx == 0 and orchestrator_payload.get("continue_from_video_resolved_path"):
                current_frame_overlap_from_previous = expanded_frame_overlap[0] if expanded_frame_overlap else 0
            elif idx > 0:
                # SM_RESTRUCTURE_FIX_OVERLAP_IDX: Use idx-1 for subsequent segments
                current_frame_overlap_from_previous = expanded_frame_overlap[idx-1] if len(expanded_frame_overlap) > (idx-1) else 0
            
            # VACE refs for this specific segment
            vace_refs_for_this_segment = [
                ref_instr for ref_instr in vace_refs_instructions_all
                if ref_instr.get("segment_idx_for_naming") == idx
            ]

            segment_payload = {
                "task_id": current_segment_task_id,
                "orchestrator_task_id_ref": orchestrator_task_id_str,
                "orchestrator_run_id": run_id,
                "project_id": orchestrator_project_id, # Added project_id
                "segment_index": idx,
                "is_first_segment": (idx == 0),
                "is_last_segment": (idx == num_segments - 1),
                
                "current_run_base_output_dir": str(current_run_output_dir.resolve()), # Base for segment's own output folder creation
                "final_video_output_path_for_segment": str(final_video_output_path_for_segment.resolve()),

                "base_prompt": expanded_base_prompts[idx],
                "negative_prompt": expanded_negative_prompts[idx],
                "segment_frames_target": expanded_segment_frames[idx],
                "frame_overlap_from_previous": current_frame_overlap_from_previous,
                "frame_overlap_with_next": expanded_frame_overlap[idx] if len(expanded_frame_overlap) > idx else 0,
                
                "vace_image_refs_to_prepare_by_headless": vace_refs_for_this_segment, # Already filtered for this segment

                "parsed_resolution_wh": orchestrator_payload["parsed_resolution_wh"],
                "model_name": orchestrator_payload["model_name"],
                "seed_to_use": orchestrator_payload.get("seed_base", 12345) + idx,
                "use_causvid_lora": orchestrator_payload.get("use_causvid_lora", False),
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
            }

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
        # The main_output_dir_base is the one passed to headless.py (e.g. server's ./outputs or steerable_motion's ./steerable_motion_output)
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

def _handle_travel_segment_task(wgp_mod, task_params_from_db: dict, main_output_dir_base: Path, segment_task_id_str: str, apply_reward_lora: bool = False, colour_match_videos: bool = False, mask_active_frames: bool = True, *, process_single_task, dprint):
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
            print(f"[ERROR Task {segment_task_id_str}]: {msg}")
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

        additional_loras = full_orchestrator_payload.get("additional_loras", {})
        if additional_loras:
            dprint(f"Segment {segment_idx}: Found additional_loras in orchestrator payload: {additional_loras}")

        current_run_base_output_dir_str = segment_params.get("current_run_base_output_dir")
        if not current_run_base_output_dir_str: # Should be passed by orchestrator/prev segment
            current_run_base_output_dir_str = full_orchestrator_payload.get("main_output_dir_for_run", str(main_output_dir_base.resolve()))
            current_run_base_output_dir_str = str(Path(current_run_base_output_dir_str) / f"travel_run_{orchestrator_run_id}")

        current_run_base_output_dir = Path(current_run_base_output_dir_str)
        segment_processing_dir = current_run_base_output_dir / f"segment_{segment_idx:02d}_{segment_task_id_str[:8]}"
        segment_processing_dir.mkdir(parents=True, exist_ok=True)
        dprint(f"Segment {segment_idx} (Task {segment_task_id_str}): Processing in {segment_processing_dir.resolve()}")

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
                start_ref_path_for_cm = sm_download_image_if_url(start_ref_path_for_cm, segment_processing_dir, segment_task_id_str)
            if end_ref_path_for_cm:
                end_ref_path_for_cm = sm_download_image_if_url(end_ref_path_for_cm, segment_processing_dir, segment_task_id_str)

            dprint(f"Seg {segment_idx} CM Refs: Start='{start_ref_path_for_cm}', End='{end_ref_path_for_cm}'")
        # --- End Color Match Reference Image Determination ---

        # --- Prepare VACE Refs for this Segment (moved to headless) ---
        actual_vace_image_ref_paths_for_wgp = []
        # Get the list of VACE ref instructions from the full orchestrator payload
        vace_ref_instructions_from_orchestrator = full_orchestrator_payload.get("vace_image_refs_to_prepare_by_headless", [])
        
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
            if not current_parsed_res_wh:
                # Fallback or error if resolution not found; for now, dprint and proceed (helper might handle None resolution)
                dprint(f"[WARNING] Segment {segment_idx}: parsed_resolution_wh not found in full_orchestrator_payload. VACE refs might not be resized correctly.")

            for ref_instr in relevant_vace_instructions:
                # Pass segment_image_download_dir to _prepare_vace_ref_for_segment_headless
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

        # Ensure parsed_res_wh is a tuple of integers
        parsed_res_wh_str = full_orchestrator_payload["parsed_resolution_wh"]
        try:
            parsed_res_wh = sm_parse_resolution(parsed_res_wh_str)
            if parsed_res_wh is None:
                raise ValueError(f"sm_parse_resolution returned None for input: {parsed_res_wh_str}")
        except Exception as e_parse_res:
            msg = f"Seg {segment_idx}: Invalid format or error parsing parsed_resolution_wh '{parsed_res_wh_str}': {e_parse_res}"
            print(f"[ERROR Task {segment_task_id_str}]: {msg}"); return False, msg
        dprint(f"Segment {segment_idx}: Parsed resolution (w,h): {parsed_res_wh}")

        # Calculate total frames for this segment once and reuse
        base_duration = segment_params.get("segment_frames_target", full_orchestrator_payload["segment_frames_expanded"][segment_idx])
        frame_overlap_from_previous = segment_params.get("frame_overlap_from_previous", 0)
        # The user-facing 'segment_frames_target' should represent the total length of the segment,
        # not just the new content. The overlap is handled internally for transition.
        total_frames_for_segment = base_duration

        fps_helpers = full_orchestrator_payload.get("fps_helpers", 16)
        fade_in_duration_str = full_orchestrator_payload["fade_in_params_json_str"]
        fade_out_duration_str = full_orchestrator_payload["fade_out_params_json_str"]
        
        # Define gray_frame_bgr here for use in subsequent segment strength adjustment
        gray_frame_bgr = sm_create_color_frame(parsed_res_wh, (128, 128, 128))

        # -------------------------------------------------------------
        #   Generate mask video for active/inactive frames if enabled
        # -------------------------------------------------------------
        mask_video_path_for_wgp: Path | None = None  # default
        if mask_active_frames:
            try:
                # --- Determine which frame indices should be kept (inactive = black) ---
                inactive_indices: set[int] = set()

                # 1) Frames reused from the previous segment (overlap)
                overlap_count = max(0, int(frame_overlap_from_previous))
                inactive_indices.update(range(overlap_count))

                # 2) First frame when this is the very first segment from scratch
                is_first_segment_val = segment_params.get("is_first_segment", False)
                is_continue_scenario = full_orchestrator_payload.get("continue_from_video_resolved_path") is not None
                if is_first_segment_val and not is_continue_scenario:
                    inactive_indices.add(0)

                # 3) Last frame for ALL segments - each segment travels TO a target image
                # Every segment ends at its target image, which should be kept (inactive/black)
                inactive_indices.add(total_frames_for_segment - 1)

                # Debug: Show the conditions that determined inactive indices
                dprint(
                    f"Seg {segment_idx}: Mask conditions - is_first_segment={is_first_segment_val}, "
                    f"is_continue_scenario={is_continue_scenario}, is_last_segment={segment_params.get('is_last_segment', False)}, "
                    f"overlap_count={overlap_count}"
                )

                h_m, w_m = parsed_res_wh[1], parsed_res_wh[0]
                mask_frames_buf: list[np.ndarray] = [
                    np.full((h_m, w_m, 3), 0 if idx in inactive_indices else 255, dtype=np.uint8)
                    for idx in range(total_frames_for_segment)
                ]

                dprint(
                    f"Seg {segment_idx}: Building mask – inactive indices: {sorted(list(inactive_indices))[:10]}{'...' if len(inactive_indices) > 10 else ''}, "
                    f"total_frames={total_frames_for_segment}, overlap={overlap_count}"
                )
                 
                # Use the same target directory logic as guide videos for consistency
                mask_video_target_dir: Path
                if db_ops.DB_TYPE == "sqlite" and db_ops.SQLITE_DB_PATH:
                    sqlite_db_parent = Path(db_ops.SQLITE_DB_PATH).resolve().parent
                    mask_video_target_dir = sqlite_db_parent / "public" / "files"
                    mask_video_target_dir.mkdir(parents=True, exist_ok=True)
                    dprint(f"Seg {segment_idx} (Task {segment_task_id_str}): Mask video (SQLite mode) will target {mask_video_target_dir}")
                else:
                    mask_video_target_dir = segment_processing_dir # Default to segment_processing_dir
                    dprint(f"Seg {segment_idx} (Task {segment_task_id_str}): Mask video (Non-SQLite mode) will target {mask_video_target_dir}")

                mask_fname_tmp = f"s{segment_idx:02d}_mask_{segment_task_id_str[:8]}.mp4"
                mask_out_path_tmp = mask_video_target_dir / mask_fname_tmp
                created_mask_vid = sm_create_video_from_frames_list(mask_frames_buf, mask_out_path_tmp, fps_helpers, parsed_res_wh)
                if created_mask_vid and created_mask_vid.exists():
                    mask_video_path_for_wgp = created_mask_vid
                    dprint(f"Seg {segment_idx}: mask video generated at {mask_video_path_for_wgp}")
                else:
                    dprint(f"[WARNING] Seg {segment_idx}: Failed to generate mask video.")
            except Exception as e_mask_gen2:
                dprint(f"[WARNING] Seg {segment_idx}: Mask video generation error: {e_mask_gen2}")

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
            # Get predecessor task ID from current task's depends_on field using db_ops
            task_dependency_id = db_ops.get_task_dependency(segment_task_id_str)
            
            if task_dependency_id:
                dprint(f"Seg {segment_idx}: Task {segment_task_id_str} depends on {task_dependency_id}. Fetching its output for guide video.")
                # path_to_previous_segment_video_output_for_guide will be relative ("files/...") if from SQLite and stored that way
                # or absolute if from Supabase or stored absolutely in SQLite.
                raw_path_from_db = db_ops.get_task_output_location_from_db(task_dependency_id)
                
                if raw_path_from_db:
                    if db_ops.DB_TYPE == "sqlite" and db_ops.SQLITE_DB_PATH and raw_path_from_db.startswith("files/"):
                        sqlite_db_parent = Path(db_ops.SQLITE_DB_PATH).resolve().parent
                        path_to_previous_segment_video_output_for_guide = str((sqlite_db_parent / "public" / raw_path_from_db).resolve())
                        dprint(f"Seg {segment_idx}: Resolved SQLite relative path from DB '{raw_path_from_db}' to absolute path '{path_to_previous_segment_video_output_for_guide}'")
                    else:
                        # Path from DB is already absolute (Supabase) or an old absolute SQLite path
                        path_to_previous_segment_video_output_for_guide = raw_path_from_db
                else:
                    path_to_previous_segment_video_output_for_guide = None # Task found, but no output_location
            else:
                dprint(f"Seg {segment_idx}: Could not find a valid 'depends_on' task ID for {segment_task_id_str}. Cannot create guide video based on predecessor.")
                path_to_previous_segment_video_output_for_guide = None

            if not path_to_previous_segment_video_output_for_guide or not Path(path_to_previous_segment_video_output_for_guide).exists():
                error_detail_path = raw_path_from_db if 'raw_path_from_db' in locals() and raw_path_from_db else path_to_previous_segment_video_output_for_guide
                msg = f"Seg {segment_idx}: Prev segment output for guide invalid/not found. Expected from prev task output. Path: {error_detail_path}"
                print(f"[ERROR Task {segment_task_id_str}]: {msg}"); return False, msg
        
        try: # Guide Video Creation Block
            guide_video_base_name = f"s{segment_idx}_guide_vid"
            input_images_resolved_original = full_orchestrator_payload["input_image_paths_resolved"]
            
            guide_video_target_dir: Path
            if db_ops.DB_TYPE == "sqlite" and db_ops.SQLITE_DB_PATH:
                sqlite_db_parent = Path(db_ops.SQLITE_DB_PATH).resolve().parent
                guide_video_target_dir = sqlite_db_parent / "public" / "files"
                guide_video_target_dir.mkdir(parents=True, exist_ok=True)
                dprint(f"Seg {segment_idx} (Task {segment_task_id_str}): Guide video (SQLite mode) will target {guide_video_target_dir}")
            else:
                guide_video_target_dir = segment_processing_dir # Default to segment_processing_dir
                dprint(f"Seg {segment_idx} (Task {segment_task_id_str}): Guide video (Non-SQLite mode) will target {guide_video_target_dir}")

            # The download is now handled inside sm_create_guide_video_for_travel_segment (via sm_image_to_frame)
            # Just pass the original paths.
            input_images_resolved_for_guide = input_images_resolved_original

            end_anchor_img_path_str_idx = segment_idx + 1
            if full_orchestrator_payload.get("continue_from_video_resolved_path"):
                 end_anchor_img_path_str_idx = segment_idx

            actual_guide_video_path_for_wgp = sm_create_guide_video_for_travel_segment(
                segment_idx_for_logging=segment_idx,
                end_anchor_image_index=end_anchor_img_path_str_idx,
                is_first_segment_from_scratch=is_first_segment_from_scratch,
                total_frames_for_segment=total_frames_for_segment,
                parsed_res_wh=parsed_res_wh,
                fps_helpers=fps_helpers,
                input_images_resolved_for_guide=input_images_resolved_for_guide,
                path_to_previous_segment_video_output_for_guide=path_to_previous_segment_video_output_for_guide,
                output_target_dir=guide_video_target_dir,
                guide_video_base_name=guide_video_base_name,
                segment_image_download_dir=segment_image_download_dir,
                task_id_for_logging=segment_task_id_str, # Corrected keyword argument
                full_orchestrator_payload=full_orchestrator_payload,
                segment_params=segment_params,
                dprint=dprint
            )
        except Exception as e_guide:
            print(f"ERROR Task {segment_task_id_str} guide prep: {e_guide}")
            traceback.print_exc()
            actual_guide_video_path_for_wgp = None
        # --- Invoke WGP Generation directly ---
        if actual_guide_video_path_for_wgp is None and not (is_first_segment_from_scratch or path_to_previous_segment_video_output_for_guide is not None and Path(path_to_previous_segment_video_output_for_guide).exists()):
            # If guide creation failed AND it was essential (not first segment that could run guideless, or no prev video for subsequent)
            msg = f"Task {segment_task_id_str}: Essential guide video failed or not possible. Cannot proceed with WGP processing."
            print(f"[ERROR] {msg}")
            return False, msg
            
        final_frames_for_wgp_generation = total_frames_for_segment
        current_wgp_engine = "wgp" # Defaulting to WGP for travel segments
        
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
        wgp_video_filename = f"s{segment_idx}_{wgp_inline_task_id}_seg_output.mp4"
        # For non-SQLite, wgp_final_output_path_for_this_segment is a suggestion for process_single_task
        # For SQLite, this specific path isn't strictly used by process_single_task for its *final* save, but can be logged.
        wgp_final_output_path_for_this_segment = segment_processing_dir / wgp_video_filename 
        
        safe_vace_image_ref_paths_for_wgp = [str(p.resolve()) if p else None for p in actual_vace_image_ref_paths_for_wgp]
        safe_vace_image_ref_paths_for_wgp = [p for p in safe_vace_image_ref_paths_for_wgp if p is not None]

        current_segment_base_prompt = segment_params.get("base_prompt", "") # Default to empty string if key is missing
        
        prompt_for_wgp = current_segment_base_prompt
        if not current_segment_base_prompt or current_segment_base_prompt.strip() == "":
            dprint(f"Seg {segment_idx} (Task {segment_task_id_str}): Original base_prompt was '{current_segment_base_prompt}'. It is empty or whitespace. Using a default prompt for WGP: 'A beautiful landscape.'")
            prompt_for_wgp = "A beautiful landscape."
        
        dprint(f"Seg {segment_idx} (Task {segment_task_id_str}): Effective prompt for WGP payload will be: '{prompt_for_wgp}'")

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
            "use_causvid_lora": full_orchestrator_payload.get("use_causvid_lora", False),
            "apply_reward_lora": full_orchestrator_payload.get("apply_reward_lora", False),
            "cfg_star_switch": full_orchestrator_payload.get("cfg_star_switch", 0),
            "cfg_zero_step": full_orchestrator_payload.get("cfg_zero_step", -1),
            "image_refs_paths": safe_vace_image_ref_paths_for_wgp,
            # Attach mask video if available
            **({"video_mask": str(mask_video_path_for_wgp.resolve())} if mask_video_path_for_wgp else {}),
        }
        if additional_loras:
            wgp_payload["additional_loras"] = additional_loras

        if full_orchestrator_payload.get("params_json_str_override"):
            try:
                additional_p = json.loads(full_orchestrator_payload["params_json_str_override"])
                # Ensure critical calculated params are not accidentally overridden by generic JSON override
                additional_p.pop("frames", None); additional_p.pop("video_length", None) 
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
        }

        dprint(f"Seg {segment_idx} (Task {segment_task_id_str}): Directly invoking WGP processing (task_id for WGP op: {wgp_inline_task_id}). Payload (first 500 chars): {json.dumps(wgp_payload, default=str)[:500]}...")
        
        # process_single_task will handle the WGP generation. If travel_chain_details are present,
        # it will also call _handle_travel_chaining_after_wgp.
        # The main_output_dir_base is passed for context, but "output_path" in wgp_payload dictates the save location.
        generation_success, wgp_output_path_or_msg = process_single_task(
            wgp_mod,
            wgp_payload,                # The parameters for the WGP generation itself
            main_output_dir_base,       # Server's main output directory (context for process_single_task)
            current_wgp_engine,         # Task type for process_single_task (e.g., "wgp")
            project_id_for_task=segment_params.get("project_id"), # Added project_id
            image_download_dir=segment_image_download_dir, # Pass the determined download dir
            apply_reward_lora=effective_apply_reward_lora,
            mask_active_frames=mask_active_frames
        )

        if generation_success:
            # wgp_output_path_or_msg should be the path to the (potentially saturated) video
            final_segment_video_output_path_str = wgp_output_path_or_msg
            # The travel_segment task's record in the DB needs its output_location to be this final path.
            # This path is also what the stitcher will look for.
            # The _handle_travel_segment_task needs to return this path so the main loop can update its own DB record.
            output_message_for_segment_task = f"Segment {segment_idx} processing (WGP generation & chaining) completed. Final output for this segment: {final_segment_video_output_path_str}"
            print(f"Seg {segment_idx} (Task {segment_task_id_str}): {output_message_for_segment_task}")
        else:
            # wgp_output_path_or_msg contains the error message if generation_success is False
            final_segment_video_output_path_str = None 
            output_message_for_segment_task = f"Segment {segment_idx} (Task {segment_task_id_str}) processing (WGP generation & chaining) failed. Error: {wgp_output_path_or_msg}"
            print(f"[ERROR] {output_message_for_segment_task}")
        
        # The old polling logic is no longer needed as process_single_task is synchronous here.

        # The return value final_segment_video_output_path_str (if success) is the one that
        # process_single_task itself would have set as 'output_location' for the WGP task.
        # Now, it becomes the output_location for the parent travel_segment task.
        return generation_success, final_segment_video_output_path_str if generation_success else output_message_for_segment_task

    except Exception as e:
        print(f"ERROR Task {segment_task_id_str}: Unexpected error during segment processing: {e}")
        traceback.print_exc()
        # return False, f"Unexpected error: {str(e)[:200]}"
        return False, f"Segment {segment_idx} failed: {str(e)[:200]}"

# --- SM_RESTRUCTURE: New function to handle chaining after WGP/Comfy sub-task ---
def _handle_travel_chaining_after_wgp(wgp_task_params: dict, actual_wgp_output_video_path: str | None, wgp_mod, image_download_dir: Path | str | None = None, *, dprint) -> tuple[bool, str, str | None]:
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

        # --- Post-generation Processing Chain ---
        # Saturation and Brightness are only applied to segments AFTER the first one.
        if is_subsequent_segment_val or is_first_new_segment_after_continue:

            # --- 1. Saturation ---
            sat_level = full_orchestrator_payload.get("after_first_post_generation_saturation")
            if sat_level is not None and isinstance(sat_level, (float, int)) and sat_level >= 0.0:
                dprint(f"Chain (Seg {segment_idx_completed}): Applying post-gen saturation {sat_level} to {video_to_process_abs_path}")

                target_dir_for_sat, db_prefix_for_sat = _get_post_processing_target_dir(db_ops.DB_TYPE, db_ops.SQLITE_DB_PATH, segment_processing_dir_for_saturation_str)
                sat_out_base = f"s{segment_idx_completed}_sat_{sat_level:.2f}"
                saturated_video_output_abs_path = sm_get_unique_target_path(target_dir_for_sat, sat_out_base, video_to_process_abs_path.suffix)
                
                if sm_apply_saturation_to_video_ffmpeg(str(video_to_process_abs_path), saturated_video_output_abs_path, sat_level):
                    new_db_path = f"{db_prefix_for_sat}{saturated_video_output_abs_path.name}"
                    dprint(f"Chain (Seg {segment_idx_completed}): Saturation successful. New path: {new_db_path}")
                    _cleanup_intermediate_video(full_orchestrator_payload, video_to_process_abs_path, segment_idx_completed, "raw", dprint)
                    
                    video_to_process_abs_path = saturated_video_output_abs_path
                    final_video_path_for_db = new_db_path
                else:
                    dprint(f"[WARNING] Chain (Seg {segment_idx_completed}): Saturation failed. Continuing with unsaturated video.")
            
            # --- 2. Brightness ---
            brightness_adjust = full_orchestrator_payload.get("after_first_post_generation_brightness", 0.0)
            if isinstance(brightness_adjust, (float, int)) and abs(brightness_adjust) > 1e-6:
                dprint(f"Chain (Seg {segment_idx_completed}): Applying post-gen brightness {brightness_adjust} to {video_to_process_abs_path}")
                
                target_dir_for_bright, db_prefix_for_bright = _get_post_processing_target_dir(db_ops.DB_TYPE, db_ops.SQLITE_DB_PATH, segment_processing_dir_for_saturation_str)
                bright_out_base = f"s{segment_idx_completed}_bright_{brightness_adjust:+.2f}"
                brightened_video_output_abs_path = sm_get_unique_target_path(target_dir_for_bright, bright_out_base, video_to_process_abs_path.suffix)
                
                processed_video = apply_brightness_to_video_frames(str(video_to_process_abs_path), brightened_video_output_abs_path, brightness_adjust, wgp_task_id)

                if processed_video and processed_video.exists():
                    new_db_path = f"{db_prefix_for_bright}{brightened_video_output_abs_path.name}"
                    dprint(f"Chain (Seg {segment_idx_completed}): Brightness adjustment successful. New path: {new_db_path}")
                    _cleanup_intermediate_video(full_orchestrator_payload, video_to_process_abs_path, segment_idx_completed, "saturated", dprint)

                    video_to_process_abs_path = brightened_video_output_abs_path
                    final_video_path_for_db = new_db_path
                else:
                    dprint(f"[WARNING] Chain (Seg {segment_idx_completed}): Brightness adjustment failed. Continuing with previous video version.")

        # --- 3. Color Matching (Applied to all segments if enabled) ---
        if chain_details.get("colour_match_videos"):
            start_ref = chain_details.get("cm_start_ref_path")
            end_ref = chain_details.get("cm_end_ref_path")
            dprint(f"Chain (Seg {segment_idx_completed}): Color matching requested. Start Ref: {start_ref}, End Ref: {end_ref}")

            if start_ref and end_ref and Path(start_ref).exists() and Path(end_ref).exists():
                target_dir_for_cm, db_prefix_for_cm = _get_post_processing_target_dir(db_ops.DB_TYPE, db_ops.SQLITE_DB_PATH, segment_processing_dir_for_saturation_str)
                cm_out_base = f"s{segment_idx_completed}_colormatched"
                cm_video_output_abs_path = sm_get_unique_target_path(target_dir_for_cm, cm_out_base, video_to_process_abs_path.suffix)

                matched_video_path = sm_apply_color_matching_to_video(
                    str(video_to_process_abs_path),
                    start_ref,
                    end_ref,
                    str(cm_video_output_abs_path),
                    dprint
                )

                if matched_video_path and Path(matched_video_path).exists():
                    new_db_path = f"{db_prefix_for_cm}{Path(matched_video_path).name}"
                    dprint(f"Chain (Seg {segment_idx_completed}): Color matching successful. New path: {new_db_path}")
                    _cleanup_intermediate_video(full_orchestrator_payload, video_to_process_abs_path, segment_idx_completed, "pre-colormatch", dprint)

                    video_to_process_abs_path = Path(matched_video_path)
                    final_video_path_for_db = new_db_path
                else:
                    dprint(f"[WARNING] Chain (Seg {segment_idx_completed}): Color matching failed. Continuing with previous video version.")
            else:
                dprint(f"[WARNING] Chain (Seg {segment_idx_completed}): Skipping color matching due to missing or invalid reference image paths.")


        # The orchestrator has already enqueued all segment and stitch tasks.
        msg = f"Chain (Seg {segment_idx_completed}): Post-WGP processing complete. Final path for this WGP task's output: {final_video_path_for_db}"
        dprint(msg)
        return True, msg, str(final_video_path_for_db)

    except Exception as e_chain:
        error_msg = f"Chain (Seg {chain_details.get('segment_index_completed', 'N/A')} for WGP {wgp_task_id}): Failed during chaining: {e_chain}"
        print(f"[ERROR] {error_msg}")
        traceback.print_exc()
        return False, error_msg, str(final_video_path_for_db) # Return path as it was before error

def _get_post_processing_target_dir(db_type, sqlite_db_path, default_dir_str):
    """Helper to determine the correct output directory for post-processing steps."""
    if db_type == "sqlite" and sqlite_db_path:
        sqlite_db_parent = Path(sqlite_db_path).resolve().parent
        target_dir = sqlite_db_parent / "public" / "files"
        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir, "files/"
    else:
        return Path(default_dir_str), ""

def _cleanup_intermediate_video(orchestrator_payload, video_path: Path, segment_idx: int, stage: str, dprint):
    """Helper to cleanup intermediate video files during chaining."""
    if not orchestrator_payload.get("skip_cleanup_enabled", False) and \
       not orchestrator_payload.get("debug_mode_enabled", False) and \
       video_path.exists():
        try:
            video_path.unlink()
            dprint(f"Chain (Seg {segment_idx}): Removed intermediate '{stage}' video {video_path}")
        except Exception as e_del:
            dprint(f"Chain (Seg {segment_idx}): Warning - could not remove intermediate video {video_path}: {e_del}")

def _handle_travel_stitch_task(task_params_from_db: dict, main_output_dir_base: Path, stitch_task_id_str: str, *, dprint):
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

        if not all([orchestrator_task_id_ref, orchestrator_run_id, full_orchestrator_payload]):
            msg = f"Stitch task {stitch_task_id_str} missing critical orchestrator refs or full_orchestrator_payload."
            print(f"[ERROR Task {stitch_task_id_str}]: {msg}")
            return False, msg

        current_run_base_output_dir_str = stitch_params.get("current_run_base_output_dir", 
                                                            full_orchestrator_payload.get("main_output_dir_for_run", str(main_output_dir_base.resolve())))
        current_run_base_output_dir = Path(current_run_base_output_dir_str)
        # If current_run_base_output_dir was the generic one, ensure it includes the run_id subfolder.
        if not str(current_run_base_output_dir.name).endswith(orchestrator_run_id):
            current_run_base_output_dir = current_run_base_output_dir / f"travel_run_{orchestrator_run_id}"
        
        stitch_processing_dir = current_run_base_output_dir / f"stitch_final_output_{stitch_task_id_str[:8]}"
        stitch_processing_dir.mkdir(parents=True, exist_ok=True)
        dprint(f"Stitch Task {stitch_task_id_str}: Processing in {stitch_processing_dir.resolve()}")

        num_expected_new_segments = full_orchestrator_payload["num_new_segments_to_generate"]
        
        # Ensure parsed_res_wh is a tuple of integers for stitch task
        parsed_res_wh_str = full_orchestrator_payload["parsed_resolution_wh"]
        try:
            parsed_res_wh = sm_parse_resolution(parsed_res_wh_str)
            if parsed_res_wh is None:
                raise ValueError(f"sm_parse_resolution returned None for input: {parsed_res_wh_str}")
        except Exception as e_parse_res_stitch:
            msg = f"Stitch Task {stitch_task_id_str}: Invalid format or error parsing parsed_resolution_wh '{parsed_res_wh_str}': {e_parse_res_stitch}"
            print(f"[ERROR Task {stitch_task_id_str}]: {msg}"); return False, msg
        dprint(f"Stitch Task {stitch_task_id_str}: Parsed resolution (w,h): {parsed_res_wh}")

        final_fps = full_orchestrator_payload.get("fps_helpers", 16)
        expanded_frame_overlaps = full_orchestrator_payload["frame_overlap_expanded"]
        crossfade_sharp_amt = full_orchestrator_payload.get("crossfade_sharp_amt", 0.3)
        initial_continued_video_path_str = full_orchestrator_payload.get("continue_from_video_resolved_path")

        # Extract upscale parameters
        upscale_factor = full_orchestrator_payload.get("upscale_factor", 0.0) # Default to 0.0 if not present
        upscale_model_name = full_orchestrator_payload.get("upscale_model_name") # Default to None if not present

        # --- 2. Collect Paths to All Segment Videos --- 
        segment_video_paths_for_stitch = []
        if initial_continued_video_path_str and Path(initial_continued_video_path_str).exists():
            dprint(f"Stitch: Prepending initial continued video: {initial_continued_video_path_str}")
            segment_video_paths_for_stitch.append(str(Path(initial_continued_video_path_str).resolve()))
        
        # Query DB for all completed generation sub-tasks for this run_id
        completed_segment_outputs_from_db = db_ops.get_completed_segment_outputs_for_stitch(orchestrator_run_id)
        dprint(f"Stitch Task {stitch_task_id_str}: Raw completed_segment_outputs_from_db: {completed_segment_outputs_from_db}")

        # Filter and add valid paths from DB query results
        # `completed_segment_outputs_from_db` now contains (segment_idx, video_path_str) directly
        for seg_idx, video_path_str_from_db in completed_segment_outputs_from_db:
            resolved_video_path_for_stitch: Path | None = None
            if video_path_str_from_db:
                if db_ops.DB_TYPE == "sqlite" and db_ops.SQLITE_DB_PATH and video_path_str_from_db.startswith("files/"):
                    sqlite_db_parent = Path(db_ops.SQLITE_DB_PATH).resolve().parent
                    absolute_path_candidate = (sqlite_db_parent / "public" / video_path_str_from_db).resolve()
                    dprint(f"Stitch: Resolved SQLite relative path from DB '{video_path_str_from_db}' to absolute '{absolute_path_candidate}' for segment index {seg_idx}")
                    if absolute_path_candidate.exists() and absolute_path_candidate.is_file():
                        resolved_video_path_for_stitch = absolute_path_candidate
                    else:
                        dprint(f"[WARNING] Stitch: Resolved absolute path '{absolute_path_candidate}' for segment index {seg_idx} (DB path: '{video_path_str_from_db}') does not exist or is not a file.")
                else: # Path from DB is already absolute (Supabase) or an old absolute SQLite path, or non-standard
                    absolute_path_candidate = Path(video_path_str_from_db).resolve()
                    if absolute_path_candidate.exists() and absolute_path_candidate.is_file():
                        resolved_video_path_for_stitch = absolute_path_candidate
                    else:
                        dprint(f"[WARNING] Stitch: Absolute path from DB '{absolute_path_candidate}' for segment index {seg_idx} does not exist or is not a file.")

            if resolved_video_path_for_stitch:
                segment_video_paths_for_stitch.append(str(resolved_video_path_for_stitch))
                dprint(f"Stitch: Adding valid video for segment index {seg_idx}: {resolved_video_path_for_stitch}")
            else: 
                dprint(f"[WARNING] Stitch: Segment video (from DB, index {seg_idx}, original path '{video_path_str_from_db}') is missing or invalid after path resolution. It will be excluded.")

        total_videos_for_stitch = (1 if initial_continued_video_path_str and Path(initial_continued_video_path_str).exists() else 0) + num_expected_new_segments
        if len(segment_video_paths_for_stitch) < total_videos_for_stitch:
            # This is a warning because some segments might have legitimately failed and been skipped by their handlers.
            # The stitcher should proceed with what it has, unless it has zero or one video when multiple were expected.
            dprint(f"[WARNING] Stitch: Expected {total_videos_for_stitch} videos for stitch, but found {len(segment_video_paths_for_stitch)}. Stitching with available videos.")
        
        if not segment_video_paths_for_stitch:
            raise ValueError("Stitch: No valid segment videos found to stitch.")
        if len(segment_video_paths_for_stitch) == 1 and total_videos_for_stitch > 1:
            dprint(f"Stitch: Only one video segment found ({segment_video_paths_for_stitch[0]}) but {total_videos_for_stitch} were expected. Using this single video as the 'stitched' output.")
            # No actual stitching needed, just move/copy this single video to final dest.

        # --- 3. Stitching (Crossfade or Concatenate) --- 
        # temp_stitched_video_output_path will now be the direct final path if SQLite, or temp if not.
        current_stitched_video_path: Path | None = None # This will hold the path to the current version of the stitched video

        final_target_dir_for_stitch_outputs: Path
        db_path_prefix_for_stitch_outputs = ""

        if db_ops.DB_TYPE == "sqlite" and db_ops.SQLITE_DB_PATH:
            sqlite_db_parent = Path(db_ops.SQLITE_DB_PATH).resolve().parent
            final_target_dir_for_stitch_outputs = sqlite_db_parent / "public" / "files"
            final_target_dir_for_stitch_outputs.mkdir(parents=True, exist_ok=True)
            db_path_prefix_for_stitch_outputs = "files/"
            dprint(f"Stitch Task {stitch_task_id_str}: SQLite mode. Stitch outputs will target {final_target_dir_for_stitch_outputs} directly.")
        else:
            # Non-SQLite: Use stitch_processing_dir as the target for intermediate/final stitch products before any final move
            final_target_dir_for_stitch_outputs = stitch_processing_dir
            # db_path_prefix_for_stitch_outputs remains "" (paths will be absolute)
            dprint(f"Stitch Task {stitch_task_id_str}: Non-SQLite mode. Stitch outputs will target {final_target_dir_for_stitch_outputs}.")


        if len(segment_video_paths_for_stitch) == 1:
            # If only one video, copy it directly to the final target dir with a 'stitched' or 'final' name
            source_single_video_path = Path(segment_video_paths_for_stitch[0])
            single_video_name_stem = f"stitched_single_{orchestrator_run_id}" # Or use source_single_video_path.stem
            
            current_stitched_video_path = sm_get_unique_target_path(
                final_target_dir_for_stitch_outputs,
                single_video_name_stem,
                source_single_video_path.suffix
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
            any_positive_overlap = any(o > 0 for o in actual_overlaps_for_stitching)

            raw_stitched_video_name_stem = f"stitched_raw_{orchestrator_run_id}"
            path_for_raw_stitched_video = sm_get_unique_target_path(
                final_target_dir_for_stitch_outputs,
                raw_stitched_video_name_stem,
                ".mp4" 
            )

            if any_positive_overlap:
                dprint(f"Stitch: Using cross-fade due to overlap values: {actual_overlaps_for_stitching}. Output to: {path_for_raw_stitched_video}")
                all_segment_frames_lists = [sm_extract_frames_from_video(p, dprint_func=dprint) for p in segment_video_paths_for_stitch]
                if not all(f_list is not None and len(f_list)>0 for f_list in all_segment_frames_lists):
                    raise ValueError("Stitch: Frame extraction failed for one or more segments during cross-fade prep.")
                
                final_stitched_frames = []
                overlap_for_first_join = actual_overlaps_for_stitching[0] if actual_overlaps_for_stitching else 0
                if len(all_segment_frames_lists[0]) > overlap_for_first_join:
                    final_stitched_frames.extend(all_segment_frames_lists[0][:-overlap_for_first_join if overlap_for_first_join > 0 else len(all_segment_frames_lists[0])])
                else: 
                    final_stitched_frames.extend(all_segment_frames_lists[0])

                for i in range(num_stitch_points): 
                    frames_prev_segment = all_segment_frames_lists[i]
                    frames_curr_segment = all_segment_frames_lists[i+1]
                    current_overlap_val = actual_overlaps_for_stitching[i]

                    if current_overlap_val > 0:
                        faded_frames = sm_cross_fade_overlap_frames(frames_prev_segment, frames_curr_segment, current_overlap_val, "linear_sharp", crossfade_sharp_amt)
                        final_stitched_frames.extend(faded_frames)
                    else: 
                        pass 
                    
                    if (i + 1) < num_stitch_points: 
                        overlap_for_next_join_of_curr = actual_overlaps_for_stitching[i+1]
                        start_index_for_curr_tail = current_overlap_val 
                        end_index_for_curr_tail = len(frames_curr_segment) - (overlap_for_next_join_of_curr if overlap_for_next_join_of_curr > 0 else 0)
                        if end_index_for_curr_tail > start_index_for_curr_tail:
                             final_stitched_frames.extend(frames_curr_segment[start_index_for_curr_tail : end_index_for_curr_tail])
                    else: 
                        start_index_for_last_segment_tail = current_overlap_val
                        if len(frames_curr_segment) > start_index_for_last_segment_tail:
                            final_stitched_frames.extend(frames_curr_segment[start_index_for_last_segment_tail:])
                
                if not final_stitched_frames: raise ValueError("Stitch: No frames produced after cross-fade logic.")
                created_video_path_obj = sm_create_video_from_frames_list(final_stitched_frames, path_for_raw_stitched_video, final_fps, parsed_res_wh)
                if created_video_path_obj and created_video_path_obj.exists():
                    current_stitched_video_path = created_video_path_obj
                else:
                    raise RuntimeError(f"Stitch: Cross-fade sm_create_video_from_frames_list failed to produce video at {path_for_raw_stitched_video}")

            else: 
                dprint(f"Stitch: Using simple FFmpeg concatenation. Output to: {path_for_raw_stitched_video}")
                try:
                    from .common_utils import stitch_videos_ffmpeg as sm_stitch_videos_ffmpeg
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
            dprint(f"Stitch: Upscaling (x{upscale_factor}) video {current_stitched_video_path.name} using model {upscale_model_name}")
            
            original_frames_count, _ = sm_get_video_frame_count_and_fps(str(current_stitched_video_path))
            if original_frames_count is None or original_frames_count == 0:
                raise ValueError(f"Stitch: Cannot get frame count or 0 frames for video {current_stitched_video_path} before upscaling.")

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
            print(f"Stitch Task {stitch_task_id_str}: Enqueued upscale sub-task {upscale_sub_task_id} ({upscaler_engine_to_use}). Waiting...")
            
            poll_interval_ups = full_orchestrator_payload.get("poll_interval", 15)
            poll_timeout_ups = full_orchestrator_payload.get("poll_timeout_upscale", full_orchestrator_payload.get("poll_timeout", 30 * 60) * 2)
            
            upscaled_video_db_location = db_ops.poll_task_status(
                task_id=upscale_sub_task_id, 
                poll_interval_seconds=poll_interval_ups, 
                timeout_seconds=poll_timeout_ups
            )
            dprint(f"Stitch Task {stitch_task_id_str}: Upscale sub-task {upscale_sub_task_id} poll result: {upscaled_video_db_location}")

            if upscaled_video_db_location:
                upscaled_video_abs_path: Path
                if db_ops.DB_TYPE == "sqlite" and db_ops.SQLITE_DB_PATH and upscaled_video_db_location.startswith("files/"):
                    sqlite_db_parent = Path(db_ops.SQLITE_DB_PATH).resolve().parent
                    upscaled_video_abs_path = sqlite_db_parent / "public" / upscaled_video_db_location
                else: 
                    upscaled_video_abs_path = Path(upscaled_video_db_location)

                if upscaled_video_abs_path.exists():
                    dprint(f"Stitch: Upscale sub-task {upscale_sub_task_id} completed. Output: {upscaled_video_abs_path}")
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
                    print(f"[WARNING] Stitch Task {stitch_task_id_str}: Upscale sub-task {upscale_sub_task_id} output missing ({upscaled_video_abs_path}). Using non-upscaled video.")
            else: 
                print(f"[WARNING] Stitch Task {stitch_task_id_str}: Upscale sub-task {upscale_sub_task_id} failed or timed out. Using non-upscaled video.")

        elif upscale_factor > 1.0 and not upscale_model_name:
            dprint(f"Stitch: Upscale factor {upscale_factor} > 1.0 but no upscale_model_name provided. Skipping upscale.")

        if db_ops.DB_TYPE == "sqlite" and db_ops.SQLITE_DB_PATH:
            if video_path_after_optional_upscale.is_absolute():
                try:
                    sqlite_public_files_dir = (Path(db_ops.SQLITE_DB_PATH).resolve().parent / "public" / "files").resolve()
                    if video_path_after_optional_upscale.resolve().parent == sqlite_public_files_dir:
                        final_video_location_for_db = f"files/{video_path_after_optional_upscale.name}"
                    else: 
                        final_video_location_for_db = str(video_path_after_optional_upscale.resolve())
                        dprint(f"[WARNING] Stitch SQLite: final video {final_video_location_for_db} is absolute but not in public/files. This is unexpected.")
                except Exception as e_rel_path:
                     dprint(f"[WARNING] Stitch SQLite: Error ensuring relative path for {video_path_after_optional_upscale}: {e_rel_path}. Using absolute.")
                     final_video_location_for_db = str(video_path_after_optional_upscale.resolve())

            elif isinstance(video_path_after_optional_upscale, str) and video_path_after_optional_upscale.startswith("files/"):
                final_video_location_for_db = video_path_after_optional_upscale
            elif isinstance(video_path_after_optional_upscale, Path) and not video_path_after_optional_upscale.is_absolute():
                 final_video_location_for_db = str(video_path_after_optional_upscale)
            else: 
                final_video_location_for_db = str(video_path_after_optional_upscale.resolve())
                dprint(f"[WARNING] Stitch SQLite: Fallback for final_video_location_for_db using resolve() for: {video_path_after_optional_upscale}")

            print(f"Stitch Task {stitch_task_id_str}: Final SQLite Video is at: {video_path_after_optional_upscale.resolve()} (DB location: {final_video_location_for_db})")
        
        else: 
            user_specified_final_path_str = stitch_params.get("final_stitched_output_path")
            final_destination_if_move_needed: Path | None = None

            if db_ops.DB_TYPE != "sqlite": 
                if user_specified_final_path_str:
                    candidate_path = Path(user_specified_final_path_str)
                    candidate_path.parent.mkdir(parents=True, exist_ok=True)
                    if candidate_path.is_dir():
                        base_name = f"travel_final_{orchestrator_run_id}"
                        if upscale_factor > 1.0: base_name = f"travel_final_upscaled_{upscale_factor:.1f}x_{orchestrator_run_id}"
                        final_destination_if_move_needed = sm_get_unique_target_path(candidate_path, base_name, video_path_after_optional_upscale.suffix)
                    else:
                        final_destination_if_move_needed = candidate_path
                else:
                    base_name = f"travel_final_{orchestrator_run_id}"
                    if upscale_factor > 1.0: base_name = f"travel_final_upscaled_{upscale_factor:.1f}x_{orchestrator_run_id}"
                    final_destination_if_move_needed = sm_get_unique_target_path(current_run_base_output_dir, base_name, video_path_after_optional_upscale.suffix)

                if video_path_after_optional_upscale.resolve() != final_destination_if_move_needed.resolve():
                    dprint(f"Stitch Task {stitch_task_id_str} (Non-SQLite): Moving {video_path_after_optional_upscale} to {final_destination_if_move_needed}")
                    shutil.move(str(video_path_after_optional_upscale), str(final_destination_if_move_needed))
                    video_path_after_optional_upscale = final_destination_if_move_needed
                else:
                    dprint(f"Stitch Task {stitch_task_id_str} (Non-SQLite): Video already at final destination {video_path_after_optional_upscale}")
            
            final_video_location_for_db = str(video_path_after_optional_upscale.resolve())
            print(f"Stitch Task {stitch_task_id_str}: Final Video (Non-SQLite or user-specified path) saved to: {final_video_location_for_db}")

        stitch_success = True

    except Exception as e_stitch_main:
        msg = f"Stitch Task {stitch_task_id_str}: Main process failed: {e_stitch_main}"
        print(f"[ERROR Task {stitch_task_id_str}]: {msg}"); traceback.print_exc()
        stitch_success = False
        final_video_location_for_db = msg # Store truncated error for DB
    
    finally:
        # --- 6. Cleanup --- 
        debug_mode = False # Default
        skip_cleanup = False # Default
        if full_orchestrator_payload: # Check if it's not None
            debug_mode = full_orchestrator_payload.get("debug_mode_enabled", False)
            skip_cleanup = full_orchestrator_payload.get("skip_cleanup_enabled", False)
        
        # Condition for cleaning the stitch_processing_dir (the sub-workspace)
        # Cleaned if stitch was successful and skip_cleanup is not set.
        # debug_mode does not prevent cleanup of this specific intermediate directory.
        cleanup_stitch_sub_workspace = stitch_success and not skip_cleanup

        if cleanup_stitch_sub_workspace and 'stitch_processing_dir' in locals() and stitch_processing_dir.exists():
            dprint(f"Stitch: Cleaning up stitch processing sub-workspace: {stitch_processing_dir}")
            try: shutil.rmtree(stitch_processing_dir)
            except Exception as e_c1: dprint(f"Stitch: Error cleaning up {stitch_processing_dir}: {e_c1}")
        elif 'stitch_processing_dir' in locals() and stitch_processing_dir.exists(): # Not cleaning it up, state why
            dprint(f"Stitch: Skipping cleanup of stitch processing sub-workspace: {stitch_processing_dir} (stitch_success:{stitch_success}, skip_cleanup:{skip_cleanup})")

        # Condition for cleaning the main run directory (current_run_base_output_dir)
        # Cleaned if stitch was successful, skip_cleanup is not set, AND debug_mode is not set.
        cleanup_main_run_dir = stitch_success and not skip_cleanup and not debug_mode

        if cleanup_main_run_dir and 'current_run_base_output_dir' in locals() and current_run_base_output_dir.exists():
            dprint(f"Stitch: Full run successful, not in debug, and cleanup not skipped. Cleaning up main run directory: {current_run_base_output_dir}")
            try: shutil.rmtree(current_run_base_output_dir)
            except Exception as e_c2: dprint(f"Stitch: Error cleaning up main run directory {current_run_base_output_dir}: {e_c2}")
        elif 'current_run_base_output_dir' in locals() and current_run_base_output_dir.exists(): # Not cleaning main run dir, state why
            reasons_for_keeping_main_run_dir = []
            if not stitch_success: reasons_for_keeping_main_run_dir.append("stitch_failed")
            if skip_cleanup: reasons_for_keeping_main_run_dir.append("skip_cleanup_enabled")
            if debug_mode: reasons_for_keeping_main_run_dir.append("debug_mode_enabled")
            # Add a more specific reason if stitch_sub_workspace was kept due to skip_cleanup, which then prevents main dir cleanup
            if not cleanup_stitch_sub_workspace and stitch_success and skip_cleanup: 
                reasons_for_keeping_main_run_dir.append("stitch_sub_workspace_kept_due_to_skip_cleanup")
            
            # Filter out redundant "debug_mode_enabled" if it's already implied by other conditions not met for cleanup_main_run_dir
            if not (stitch_success and not skip_cleanup) and "debug_mode_enabled" in reasons_for_keeping_main_run_dir:
                 if not debug_mode: # if debug_mode is false, but it's listed as a reason, it's because other primary conditions failed
                     pass # Let it be listed if debug_mode is true
                 elif not (stitch_success and not skip_cleanup): # if other conditions failed, debug is secondary
                     # This part of logic for dprint is getting complex, simplify the message:
                     pass # Handled by the general list.

            final_reason_text = ", ".join(reasons_for_keeping_main_run_dir) if reasons_for_keeping_main_run_dir else "(e.g. stitch failed or conditions not met for cleanup)"
            dprint(f"Stitch: Skipping cleanup of main run directory: {current_run_base_output_dir} (Reasons: {final_reason_text})")
            
    return stitch_success, final_video_location_for_db
