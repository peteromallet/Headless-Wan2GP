"""Travel-between-images task handler."""

import json
import shutil
import traceback
from pathlib import Path
import datetime
import subprocess
# import math # No longer used directly here
import requests
from urllib.parse import urlparse
import uuid

import cv2
import numpy as np
from PIL import Image, ImageEnhance

# Import from the common_utils module
from .common_utils import (
    DEBUG_MODE, dprint, generate_unique_task_id, add_task_to_db, # poll_task_status will be removed from usage here
    # extract_video_segment_ffmpeg, # Not used directly by orchestrator
    stitch_videos_ffmpeg, # Will be used by a stitcher task in headless
    # create_pose_interpolated_guide_video, # Guide creation moves to headless
    generate_debug_summary_video, # Potentially used by a final debug task in headless
    # extract_specific_frame_ffmpeg, # Used by headless segment task
    # concatenate_videos_ffmpeg, # Alternative stitch, might be used by headless
    get_video_frame_count_and_fps, # Used by headless segment/stitch task
    _get_unique_target_path,
    image_to_frame,
    create_color_frame,
    _adjust_frame_brightness,
    _copy_to_folder_with_unique_name, # For downloading initial video
    # _apply_strength_to_image # This was the common_utils version, not the file one.
    # We are removing the file-based strength application from orchestrator.
)

# Import from the video_utils module (some might be used by headless now)
from .video_utils import (
    crossfade_ease,
    _blend_linear,
    _blend_linear_sharp,
    cross_fade_overlap_frames,
    extract_frames_from_video,
    create_video_from_frames_list,
    _apply_saturation_to_video_ffmpeg,
    color_match_video_to_reference
)

DEFAULT_SEGMENT_FRAMES = 81 # This constant might still be relevant for defaults before expansion


# --- Easing functions (for guide video timing, not cross-fade) ---
# These are kept if orchestrator prepares any initial guide data or passes easing params
def ease_linear(t: float) -> float:
    """Linear interpolation (0..1 -> 0..1)"""
    return t

def ease_in_quad(t: float) -> float:
    """Ease-in quadratic (0..1 -> 0..1)"""
    return t * t

def ease_out_quad(t: float) -> float:
    """Ease-out quadratic (0..1 -> 0..1)"""
    return t * (2 - t)

def ease_in_out_quad(t: float) -> float:
    """Ease-in-out quadratic (0..1 -> 0..1)"""
    return t * t * (3 - 2 * t) if t < 0.5 else (1 - ((-2 * t + 2) * (-2 * t + 2) / 2))


def get_easing_function(curve_type: str):
    if curve_type == "ease_in":
        return ease_in_quad
    elif curve_type == "ease_out":
        return ease_out_quad
    elif curve_type == "ease_in_out":
        return ease_in_out_quad
    elif curve_type == "linear":
        return ease_linear
    else: # Default or unknown
        dprint(f"Warning: Unknown curve_type '{curve_type}'. Defaulting to ease_in_out_quad.")
        return ease_in_out_quad



def run_travel_between_images_task(task_args, common_args, parsed_resolution, main_output_dir, db_file_path, executed_command_str: str | None = None):
    print("--- Queuing Task: Travel Between Images (Orchestrator) ---")
    dprint(f"Task Args: {task_args}")
    dprint(f"Common Args: {common_args}")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_id = timestamp # Unique ID for this entire travel operation

    # The orchestrator task itself might not need a deep processing folder,
    # as headless will manage folders for its child segment/stitch tasks.
    # However, a shallow folder for the orchestrator's own log/command might be useful.
    orchestrator_log_folder_name = f"travel_orchestrator_log_{run_id}"
    orchestrator_log_folder = main_output_dir / orchestrator_log_folder_name
    orchestrator_log_folder.mkdir(parents=True, exist_ok=True)
    dprint(f"Orchestrator logs/info for this run will be in: {orchestrator_log_folder}")

    if DEBUG_MODE and executed_command_str:
        try:
            command_file_path = orchestrator_log_folder / "executed_command.txt"
            with open(command_file_path, "w") as f:
                f.write(executed_command_str)
            dprint(f"Saved executed command to {command_file_path}")
        except Exception as e_cmd_save:
            dprint(f"Warning: Could not save executed command: {e_cmd_save}")

    # --- Helper function to download video if URL (used for continue_from_video) ---
    # This needs to run here to get the initial video path for the orchestrator payload.
    def _download_video_if_url(video_url_or_path: str, target_dir: Path, base_name: str) -> Path | None:
        parsed_url = urlparse(video_url_or_path)
        if parsed_url.scheme in ['http', 'https']:
            try:
                dprint(f"Downloading video from URL: {video_url_or_path}")
                response = requests.get(video_url_or_path, stream=True, timeout=300)
                response.raise_for_status()
                original_filename = Path(parsed_url.path).name
                original_suffix = Path(original_filename).suffix if Path(original_filename).suffix else ".mp4"
                if not original_suffix.startswith('.'):
                    original_suffix = '.' + original_suffix
                # Use common_utils' _get_unique_target_path
                temp_download_path = _get_unique_target_path(target_dir, base_name, original_suffix)
                with open(temp_download_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192): f.write(chunk)
                dprint(f"Video downloaded successfully to {temp_download_path}")
                return temp_download_path
            except Exception as e_req:
                print(f"Error downloading video {video_url_or_path}: {e_req}")
                return None
        else:
            local_path = Path(video_url_or_path)
            if not local_path.exists():
                print(f"Error: Local video file not found: {local_path}")
                return None
            # Use common_utils' _copy_to_folder_with_unique_name
            copied_local_video_path = _copy_to_folder_with_unique_name(
                source_path=local_path, target_dir=target_dir, base_name=base_name,
                extension=local_path.suffix if local_path.suffix else ".mp4"
            )
            if copied_local_video_path:
                dprint(f"Copied local video {local_path} to {copied_local_video_path}")
                return copied_local_video_path
            else:
                print(f"Error: Failed to copy local video {local_path} to orchestrator folder.")
                return None
    # --- End Download Helper ---

    # --- Determine number of segments ---
    if task_args.continue_from_video:
        if not task_args.input_images:  # Already validated by steerable_motion.py main
            print("Error: --input_images must be provided with --continue_from_video.")
            return 1  # Should not happen due to prior validation
        num_segments_to_generate = len(task_args.input_images)
    else:
        if len(task_args.input_images) < 2:  # Already validated
            print("Error: At least two input images are required without --continue_from_video.")
            return 1  # Should not happen
        num_segments_to_generate = len(task_args.input_images) - 1

    # --- Validate num_segments_to_generate ---
    if num_segments_to_generate <= 0:
        print(f"Error: Based on input_images ({len(task_args.input_images)}) and continue_from_video flag, no new segments would be generated.")
        return 1

    # Adjust parsed_resolution to be multiples of 16 for consistency downstream
    original_width, original_height = parsed_resolution
    adjusted_width = (original_width // 16) * 16
    adjusted_height = (original_height // 16) * 16
    final_parsed_resolution_wh = (adjusted_width, adjusted_height)

    if final_parsed_resolution_wh != parsed_resolution:
        dprint(f"Orchestrator: Adjusted parsed_resolution from {parsed_resolution} to {final_parsed_resolution_wh} (multiples of 16).")

    expanded_base_prompts = task_args.base_prompts * num_segments_to_generate if len(task_args.base_prompts) == 1 else task_args.base_prompts
    expanded_negative_prompts = task_args.negative_prompts * num_segments_to_generate if len(task_args.negative_prompts) == 1 else task_args.negative_prompts
    expanded_segment_frames = task_args.segment_frames * num_segments_to_generate if len(task_args.segment_frames) == 1 else task_args.segment_frames
    expanded_frame_overlap = task_args.frame_overlap * num_segments_to_generate if len(task_args.frame_overlap) == 1 else task_args.frame_overlap
    
    # --- VACE Image Refs Preparation (Simplified for Orchestrator) ---
    # Orchestrator now only records original paths and strengths.
    # Headless.py will handle the actual image processing for each segment.
    vace_image_refs_to_prepare_by_headless = [] 

    for i in range(num_segments_to_generate):
        # Initial Anchor for VACE
        initial_anchor_path_str_for_vace: str | None = None
        if i == 0 and not task_args.continue_from_video: # Very first segment from scratch
            initial_anchor_path_str_for_vace = task_args.input_images[0]
        elif i > 0 : # Subsequent new segment, initial anchor is previous new segment's end
            initial_anchor_path_str_for_vace = task_args.input_images[i] if not task_args.continue_from_video else task_args.input_images[i-1]

        if initial_anchor_path_str_for_vace and common_args.initial_image_strength > 0.0:
            original_path = Path(initial_anchor_path_str_for_vace)
            if original_path.exists():
                vace_image_refs_to_prepare_by_headless.append({
                    "type": "initial",
                    "original_path": str(original_path.resolve()),
                    "strength_to_apply": common_args.initial_image_strength,
                    "segment_idx_for_naming": i 
                })
                dprint(f"Orchestrator: Queued VACE Ref (initial for seg {i}): original {original_path}, strength {common_args.initial_image_strength}")
            else:
                dprint(f"Orchestrator: VACE Ref (initial for seg {i}): Original path {original_path} not found. Skipping.")
        
        # Final Anchor for VACE (end of current new segment i)
        final_anchor_path_str_for_vace = task_args.input_images[i] if task_args.continue_from_video else task_args.input_images[i+1]
        if common_args.final_image_strength > 0.0:
            original_path = Path(final_anchor_path_str_for_vace)
            if original_path.exists():
                vace_image_refs_to_prepare_by_headless.append({
                    "type": "final",
                    "original_path": str(original_path.resolve()),
                    "strength_to_apply": common_args.final_image_strength,
                    "segment_idx_for_naming": i 
                })
                dprint(f"Orchestrator: Queued VACE Ref (final for seg {i}): original {original_path}, strength {common_args.final_image_strength}")
            else:
                dprint(f"Orchestrator: VACE Ref (final for seg {i}): Original path {original_path} not found. Skipping.")
    # --- End VACE Image Refs ---

    orchestrator_task_id = generate_unique_task_id(f"sm_travel_orchestrator_{run_id[:8]}_")
    
    # Handle continue_from_video: download it now if it's a URL, get its path
    initial_video_path_for_headless: str | None = None
    if task_args.continue_from_video:
        dprint(f"Orchestrator: continue_from_video specified: {task_args.continue_from_video}")
        downloaded_continued_video_path = _download_video_if_url(
            task_args.continue_from_video,
            orchestrator_log_folder, 
            "continued_video_input"
        )
        if downloaded_continued_video_path and downloaded_continued_video_path.exists():
            initial_video_path_for_headless = str(downloaded_continued_video_path.resolve())
            dprint(f"Orchestrator: Continue video prepared at: {initial_video_path_for_headless}")
        else:
            print(f"Error: Could not load or download video from {task_args.continue_from_video}. Cannot proceed with orchestrator task.")
            return 1 # Error case

    orchestrator_payload = {
        "orchestrator_task_id": orchestrator_task_id,
        "run_id": run_id, # For grouping segment task outputs later in headless if needed
        "original_task_args": vars(task_args), # Store original CLI args for this travel task
        "original_common_args": vars(common_args), # Store original common args
        "parsed_resolution_wh": final_parsed_resolution_wh, # (width, height) tuple, adjusted to be multiple of 16
        "main_output_dir_for_run": str(main_output_dir.resolve()), # Base output dir for headless to use
        "orchestrator_log_folder": str(orchestrator_log_folder.resolve()), # For headless to potentially write logs or find assets

        "input_image_paths_resolved": [str(Path(p).resolve()) for p in task_args.input_images],
        "continue_from_video_resolved_path": initial_video_path_for_headless, # Path to downloaded/copied video if used

        "num_new_segments_to_generate": num_segments_to_generate,
        "base_prompts_expanded": expanded_base_prompts,
        "negative_prompts_expanded": expanded_negative_prompts,
        "segment_frames_expanded": expanded_segment_frames,
        "frame_overlap_expanded": expanded_frame_overlap,
        
        "vace_image_refs_to_prepare_by_headless": vace_image_refs_to_prepare_by_headless, # New structure

        # Pass fade params directly for headless to use when creating guides
        "fade_in_params_json_str": common_args.fade_in_duration,
        "fade_out_params_json_str": common_args.fade_out_duration,
        
        # Other common args that headless might need for segment tasks or final stitch/upscale
        "model_name": common_args.model_name,
        "seed_base": common_args.seed,
        "use_causvid_lora": common_args.use_causvid_lora,
        "cfg_star_switch": common_args.cfg_star_switch,
        "cfg_zero_step": common_args.cfg_zero_step,
        "params_json_str_override": common_args.params_json_str, # For headless to merge into segment tasks
        "fps_helpers": common_args.fps_helpers, # For guide/stitch tasks in headless
        "subsequent_starting_strength_adjustment": common_args.subsequent_starting_strength_adjustment,
        "desaturate_subsequent_starting_frames": common_args.desaturate_subsequent_starting_frames,
        "adjust_brightness_subsequent_starting_frames": common_args.adjust_brightness_subsequent_starting_frames,
        "after_first_post_generation_saturation": common_args.after_first_post_generation_saturation,
        "crossfade_sharp_amt": getattr(task_args, 'crossfade_sharp_amt', 0.3), # from travel_args default or value

        "upscale_factor": task_args.upscale_factor, # From travel_args
        "upscale_model_name": common_args.upscale_model_name,
        
        "debug_mode_enabled": DEBUG_MODE, # For headless to know if it should run in debug
        "skip_cleanup_enabled": common_args.skip_cleanup # For headless to respect
    }

    try:
        # Add the single orchestrator task to the DB
        add_task_to_db(
            task_payload={  # This is the `params` field in the DB
                "orchestrator_details": orchestrator_payload,
                "task_id": orchestrator_task_id
            },
            db_path=db_file_path,
            task_type_str="travel_orchestrator",
            depends_on=None # Orchestrator task itself has no dependency
        )
        print(f"Successfully enqueued 'travel_orchestrator' task (ID: {orchestrator_task_id}).")
        # Section 4.3: Report final stitch task ID
        final_stitch_task_id = f"travel_stitch_{run_id}" # run_id is the timestamp for the orchestrator run
        print(f"Orchestrator task ID: {orchestrator_task_id}")
        print(f"Final stitch task ID to monitor: {final_stitch_task_id}")

        dprint(f"Orchestrator payload submitted: {json.dumps(orchestrator_payload, indent=2, default=str)}")
        return 0 # Successful queuing
    except Exception as e_db_add:
        print(f"Failed to add travel_orchestrator task {orchestrator_task_id} to DB: {e_db_add}")
        traceback.print_exc()
        return 1 # Error during DB add
