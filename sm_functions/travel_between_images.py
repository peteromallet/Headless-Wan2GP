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
    DEBUG_MODE, dprint, generate_unique_task_id, add_task_to_db, poll_task_status, # poll_task_status will be removed from usage here
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
    _apply_strength_to_image # For preparing VACE refs if done by orchestrator
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


# --- Helper: Re-encode a video to H.264 using FFmpeg (ensures consistent codec) ---
# This might be used by headless if a continued video needs re-encoding.
def _reencode_to_h264_ffmpeg(
    input_video_path: str | Path,
    output_video_path: str | Path,
    fps: float | None = None,
    resolution: tuple[int, int] | None = None,
    crf: int = 23,
    preset: str = "veryfast"
):
    """Re-encodes the entire input video to H.264 using libx264."""
    inp = Path(input_video_path)
    outp = Path(output_video_path)
    outp.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(inp.resolve()),
        "-an",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", str(crf),
        "-preset", preset,
    ]
    if fps is not None and fps > 0:
        cmd.extend(["-r", str(fps)])
    if resolution is not None:
        w, h = resolution
        cmd.extend(["-vf", f"scale={w}:{h}"])
    cmd.append(str(outp.resolve()))

    dprint(f"REENCODE_TO_H264_FFMPEG: Running command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, encoding="utf-8")
        return outp.exists() and outp.stat().st_size > 0
    except subprocess.CalledProcessError as e:
        print(f"Error during FFmpeg re-encode of {inp} -> {outp}:\nstdout:\n{e.stdout}\nstderr:\n{e.stderr}")
        return False

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

    expanded_base_prompts = task_args.base_prompts * num_segments_to_generate if len(task_args.base_prompts) == 1 else task_args.base_prompts
    expanded_negative_prompts = task_args.negative_prompts * num_segments_to_generate if len(task_args.negative_prompts) == 1 else task_args.negative_prompts
    expanded_segment_frames = task_args.segment_frames * num_segments_to_generate if len(task_args.segment_frames) == 1 else task_args.segment_frames
    expanded_frame_overlap = task_args.frame_overlap * num_segments_to_generate if len(task_args.frame_overlap) == 1 else task_args.frame_overlap
    
    # --- Prepare VACE Image Refs (if strength > 0) ---
    # This logic might still live in the orchestrator if it needs to prepare these files upfront
    # and pass their paths to headless. Or, headless could do it. For now, keeping it here.
    # Paths will be absolute, resolved here.
    vace_image_refs_details = [] # List of dicts: {"type": "initial" or "final", "original_path": str, "processed_path": str, "strength": float}

    # Helper to prepare a VACE ref image
    def _prepare_vace_ref_image(original_path_str: str, strength: float, ref_type: str, segment_idx_for_naming: int, target_dir: Path) -> dict | None:
        original_path = Path(original_path_str).resolve()
        clamped_strength = max(0.0, min(1.0, strength))
        if not original_path.exists() or clamped_strength == 0.0:
            dprint(f"VACE Ref ({ref_type} for seg {segment_idx_for_naming}): Original path {original_path} not found or strength {clamped_strength} is 0. Skipping.")
            return None

        detail = {"type": ref_type, "original_path": str(original_path), "strength": clamped_strength}
        
        processed_name_base = f"s{segment_idx_for_naming}_{ref_type}_anchor_strength_{clamped_strength:.2f}"
        original_suffix = original_path.suffix if original_path.suffix else ".png"
        
        # Use common_utils
        path_for_processed_ref = _get_unique_target_path(target_dir, processed_name_base, original_suffix)

        if abs(clamped_strength - 1.0) < 1e-5:  # Strength is 1.0
            try:
                shutil.copy2(str(original_path), str(path_for_processed_ref))
                detail["processed_path"] = str(path_for_processed_ref.resolve())
                dprint(f"VACE Ref ({ref_type} for seg {segment_idx_for_naming}): Copied original due to strength 1.0: {path_for_processed_ref}")
                return detail
            except Exception as e_copy:
                dprint(f"VACE Ref ({ref_type} for seg {segment_idx_for_naming}): Failed to copy original: {e_copy}. Skipping.")
                return None
        else:  # Strength < 1.0
            processed_path = _apply_strength_to_image(
                image_path=original_path, strength=clamped_strength,
                output_path=path_for_processed_ref, target_resolution=parsed_resolution
            )
            if processed_path and processed_path.exists():
                detail["processed_path"] = str(processed_path.resolve())
                dprint(f"VACE Ref ({ref_type} for seg {segment_idx_for_naming}): Processed with strength {clamped_strength}: {processed_path}")
                return detail
                else:
                dprint(f"VACE Ref ({ref_type} for seg {segment_idx_for_naming}): Failed to process with strength {clamped_strength}. Skipping.")
                return None
    
    # Collect VACE refs per *newly generated* segment
    # The orchestrator_log_folder is a good temporary place for these processed VACE refs
    # if they are generated by the orchestrator itself.
    for i in range(num_segments_to_generate):
        # Initial Anchor for VACE
        initial_anchor_path_str_for_vace: str | None = None
        if i == 0 and not task_args.continue_from_video: # Very first segment from scratch
            initial_anchor_path_str_for_vace = task_args.input_images[0]
        elif i > 0 : # Subsequent new segment, initial anchor is previous new segment's end
             # If continue_from_video, input_images[i-1] is the end of (i-1)th *new* segment
             # If not continue_from_video, input_images[i] is the end of (i-1)th *new* segment (since input_images[0] was start of 0th new)
            initial_anchor_path_str_for_vace = task_args.input_images[i] if not task_args.continue_from_video else task_args.input_images[i-1]


        if initial_anchor_path_str_for_vace and common_args.initial_image_strength > 0.0:
            ref_detail = _prepare_vace_ref_image(initial_anchor_path_str_for_vace, common_args.initial_image_strength, "initial", i, orchestrator_log_folder)
            if ref_detail: vace_image_refs_details.append(ref_detail)
        
        # Final Anchor for VACE (end of current new segment i)
        # If continue_from_video, end anchor is input_images[i]
        # If not, end anchor is input_images[i+1]
        final_anchor_path_str_for_vace = task_args.input_images[i] if task_args.continue_from_video else task_args.input_images[i+1]
        if common_args.final_image_strength > 0.0:
            ref_detail = _prepare_vace_ref_image(final_anchor_path_str_for_vace, common_args.final_image_strength, "final", i, orchestrator_log_folder)
            if ref_detail: vace_image_refs_details.append(ref_detail)
    # --- End VACE Image Refs ---

    orchestrator_task_id = generate_unique_task_id(f"sm_travel_orchestrator_{run_id[:8]}_" )
    
    # Handle continue_from_video: download it now if it's a URL, get its path
    initial_video_path_for_headless: str | None = None
    if task_args.continue_from_video:
        dprint(f"Orchestrator: continue_from_video specified: {task_args.continue_from_video}")
        # Download to orchestrator_log_folder, headless will copy/use it from there or orchestrator will specify absolute path.
        # For simplicity, let orchestrator download it to a known sub-folder that headless can expect or is told about.
        # A sub-folder within orchestrator_log_folder is fine.
        downloaded_continued_video_path = _download_video_if_url(
            task_args.continue_from_video,
            orchestrator_log_folder, # Store it within the orchestrator's own log/asset folder
            "continued_video_input"
        )
        if downloaded_continued_video_path and downloaded_continued_video_path.exists():
            initial_video_path_for_headless = str(downloaded_continued_video_path.resolve())
            dprint(f"Orchestrator: Continue video prepared at: {initial_video_path_for_headless}")
            # Optionally, re-encode here if needed, though headless could also do this.
            # For now, assume headless can handle it or will re-encode if necessary.
                            else:
            print(f"Error: Could not load or download video from {task_args.continue_from_video}. Cannot proceed with orchestrator task.")
            return 1

    orchestrator_payload = {
        "orchestrator_task_id": orchestrator_task_id,
        "run_id": run_id, # For grouping segment task outputs later in headless if needed
        "original_task_args": vars(task_args), # Store original CLI args for this travel task
        "original_common_args": vars(common_args), # Store original common args
        "parsed_resolution_wh": parsed_resolution, # (width, height) tuple
        "main_output_dir_for_run": str(main_output_dir.resolve()), # Base output dir for headless to use
        "orchestrator_log_folder": str(orchestrator_log_folder.resolve()), # For headless to potentially write logs or find assets

        "input_image_paths_resolved": [str(Path(p).resolve()) for p in task_args.input_images],
        "continue_from_video_resolved_path": initial_video_path_for_headless, # Path to downloaded/copied video if used

        "num_new_segments_to_generate": num_segments_to_generate,
        "base_prompts_expanded": expanded_base_prompts,
        "negative_prompts_expanded": expanded_negative_prompts,
        "segment_frames_expanded": expanded_segment_frames,
        "frame_overlap_expanded": expanded_frame_overlap,
        
        "vace_image_refs_prepared": vace_image_refs_details, # List of dicts with paths to strength-adjusted images

        # Pass fade params directly for headless to use when creating guides
        "fade_in_params_json_str": common_args.fade_in_duration,
        "fade_out_params_json_str": common_args.fade_out_duration,
        
        # Other common args that headless might need for segment tasks or final stitch/upscale
        "model_name": common_args.model_name,
        "seed_base": common_args.seed,
        "execution_engine": common_args.execution_engine,
        "use_causvid_lora": common_args.use_causvid_lora,
        "cfg_star_switch": common_args.cfg_star_switch,
        "cfg_zero_step": common_args.cfg_zero_step,
        "params_json_str_override": common_args.params_json_str, # For headless to merge into segment tasks
        "fps_helpers": common_args.fps_helpers, # For guide/stitch tasks in headless
        "last_frame_duplication": common_args.last_frame_duplication,
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
                "orchestrator_details": orchestrator_payload 
            },
            db_path_str=db_file_path,
            task_type="travel_orchestrator",  # New task type
            task_id_override=orchestrator_task_id  # Use the generated ID
        )
        print(f"Successfully enqueued 'travel_orchestrator' task (ID: {orchestrator_task_id}).")
        print(f"Headless.py will now process this sequence.")
        dprint(f"Orchestrator payload submitted: {json.dumps(orchestrator_payload, indent=2, default=str)}")
        except Exception as e_db_add:
        print(f"Failed to add travel_orchestrator task {orchestrator_task_id} to DB: {e_db_add}")
                    traceback.print_exc()
        return 1

    return 0 # Success (orchestrator task queued)

# Note: The main loop, polling, segment processing, guide video creation,
# VACE ref application (if moved to headless), stitching, and cleanup
# are now expected to be handled by headless.py based on the orchestrator task.
# Functions like _get_unique_target_path, image_to_frame, create_color_frame,
# _adjust_frame_brightness etc. are now imported from common_utils.
# Video processing functions like extract_frames_from_video, create_video_from_frames_list,
# cross_fade_overlap_frames etc. are imported from video_utils.
# They will be called by headless.py.

