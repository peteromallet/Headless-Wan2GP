"""
Join Clips - Bridge two video clips using VACE generation

This module provides functionality to smoothly join two video clips by:
1. Optionally standardizing both videos to a target aspect ratio (via center-crop)
2. Extracting context frames from the end of the first clip
3. Extracting context frames from the beginning of the second clip
4. Generating transition frames between them using VACE
5. Using mask video to preserve the context frames and only generate the gap

This is a simplified, single-generation task that reuses the VACE continuation
infrastructure from travel_between_images without the complexity of orchestration.
"""

import json
import uuid
import time
from pathlib import Path
from typing import Tuple
from datetime import datetime

import cv2
import numpy as np

# Import shared utilities
from ..common_utils import (
    ensure_valid_prompt,
    ensure_valid_negative_prompt,
    get_video_frame_count_and_fps,
    download_video_if_url,
    save_frame_from_video
)
from ..video_utils import (
    extract_frames_from_video as sm_extract_frames_from_video,
    extract_frame_range_to_video,
    ensure_video_fps,
    standardize_video_aspect_ratio,
    stitch_videos_with_crossfade as sm_stitch_videos_with_crossfade,
)
from ..vace_frame_utils import (
    create_guide_and_mask_for_generation,
    prepare_vace_generation_params
)
from .. import db_operations as db_ops


def _calculate_vace_quantization(context_frame_count: int, gap_frame_count: int, replace_mode: bool) -> dict:
    """
    Calculate VACE quantization adjustments for frame counts.

    VACE requires frame counts to match the pattern 4n+1 (e.g., 45, 49, 53).
    When we request a frame count that doesn't match this pattern, VACE will
    quantize down to the nearest valid count.

    Args:
        context_frame_count: Number of frames from each clip's boundary
        gap_frame_count: Number of frames to generate in the gap
        replace_mode: Whether we're replacing a portion (True) or inserting (False)
                     Both modes now work similarly for VACE - context + gap + context

    Returns:
        dict with:
            - total_frames: Actual frame count VACE will generate
            - gap_for_guide: Adjusted gap to use in guide/mask creation
            - quantization_shift: Number of frames dropped by VACE (0 if no quantization)
    """
    # Both modes now use the same calculation: context + gap + context
    # The difference is in how the final video is stitched (replace removes original frames)
    desired_total = context_frame_count * 2 + gap_frame_count

    # Apply VACE quantization (4n + 1)
    actual_total = ((desired_total - 1) // 4) * 4 + 1
    quantization_shift = desired_total - actual_total

    # Adjust gap to account for dropped frames
    gap_for_guide = max(0, gap_frame_count - quantization_shift)

    return {
        'total_frames': actual_total,
        'gap_for_guide': gap_for_guide,
        'quantization_shift': quantization_shift,
    }


def _handle_join_clips_task(
    task_params_from_db: dict,
    main_output_dir_base: Path,
    task_id: str,
    task_queue = None,
    *, dprint
) -> Tuple[bool, str]:
    """
    Handle join_clips task: bridge two video clips using VACE generation.

    Args:
        task_params_from_db: Task parameters including:
            - starting_video_path: Path to first video clip
            - ending_video_path: Path to second video clip
            - context_frame_count: Number of frames to extract from each clip
            - gap_frame_count: Number of frames to generate between clips (INSERT mode) or replace (REPLACE mode)
            - replace_mode: Optional bool (default False). If True, gap frames REPLACE boundary frames instead of being inserted
            - prompt: Generation prompt for the transition
            - aspect_ratio: Optional aspect ratio (e.g., "16:9", "9:16", "1:1") to standardize both videos
            - model: Optional model override (defaults to wan_2_2_vace_lightning_baseline_2_2_2)
            - resolution: Optional [width, height] override
            - use_input_video_resolution: Optional bool (default False). If True, uses the detected resolution from input video instead of resolution override
            - fps: Optional FPS override (defaults to 16)
            - use_input_video_fps: Optional bool (default False). If True, uses input video's FPS. If False, downsamples to fps param (default 16)
            - max_wait_time: Optional timeout in seconds for generation (defaults to 1800s / 30 minutes)
            - additional_loras: Optional dict of additional LoRAs {name: weight}
            - Other standard VACE parameters (guidance_scale, flow_shift, etc.)
        main_output_dir_base: Base output directory
        task_id: Task ID for logging and status updates
        task_queue: HeadlessTaskQueue instance for generation
        dprint: Print function for logging

    Returns:
        Tuple of (success: bool, output_path_or_message: str)
    """
    dprint(f"[JOIN_CLIPS] Task {task_id}: Starting join_clips handler")

    try:
        # --- 1. Extract and Validate Parameters ---
        starting_video_path = task_params_from_db.get("starting_video_path")
        ending_video_path = task_params_from_db.get("ending_video_path")
        context_frame_count = task_params_from_db.get("context_frame_count", 8)
        gap_frame_count = task_params_from_db.get("gap_frame_count", 53)
        replace_mode = task_params_from_db.get("replace_mode", False)  # If True, gap REPLACES frames instead of inserting
        prompt = task_params_from_db.get("prompt", "")
        aspect_ratio = task_params_from_db.get("aspect_ratio")  # Optional: e.g., "16:9", "9:16", "1:1"
        
        # Extract keep_bridging_images param
        keep_bridging_images = task_params_from_db.get("keep_bridging_images", False)

        dprint(f"[JOIN_CLIPS] Task {task_id}: Mode: {'REPLACE' if replace_mode else 'INSERT'}")
        if replace_mode:
            dprint(f"[JOIN_CLIPS] Task {task_id}: gap_frame_count={gap_frame_count} frames will REPLACE boundary frames (no insertion)")

        # Check if this is part of an orchestrator and starting_video_path needs to be fetched
        orchestrator_task_id_ref = task_params_from_db.get("orchestrator_task_id_ref")
        is_first_join = task_params_from_db.get("is_first_join", False)

        if not starting_video_path:
            # Check if this is an orchestrator child task that needs to fetch predecessor output
            if orchestrator_task_id_ref:
                if is_first_join:
                    error_msg = "First join in orchestrator must have starting_video_path explicitly set"
                    dprint(f"[JOIN_CLIPS_ERROR] Task {task_id}: {error_msg}")
                    return False, error_msg

                dprint(f"[JOIN_CLIPS] Task {task_id}: Part of orchestrator {orchestrator_task_id_ref}, fetching predecessor output")

                # Fetch predecessor output using edge function
                predecessor_id, predecessor_output = db_ops.get_predecessor_output_via_edge_function(task_id)

                if not predecessor_output:
                    error_msg = f"Failed to fetch predecessor output (predecessor_id={predecessor_id})"
                    dprint(f"[JOIN_CLIPS_ERROR] Task {task_id}: {error_msg}")
                    return False, error_msg

                starting_video_path = predecessor_output
                dprint(f"[JOIN_CLIPS] Task {task_id}: Fetched predecessor output: {predecessor_output}")
            else:
                # Standalone join_clips task without starting_video_path
                error_msg = "starting_video_path is required"
                dprint(f"[JOIN_CLIPS_ERROR] Task {task_id}: {error_msg}")
                return False, error_msg

        if not ending_video_path:
            error_msg = "ending_video_path is required"
            dprint(f"[JOIN_CLIPS_ERROR] Task {task_id}: {error_msg}")
            return False, error_msg

        # Validate task queue
        if task_queue is None:
            error_msg = "task_queue is required for join_clips"
            dprint(f"[JOIN_CLIPS_ERROR] Task {task_id}: {error_msg}")
            return False, error_msg

        # Create working directory - use custom dir if provided by orchestrator
        if "join_output_dir" in task_params_from_db:
            join_clips_dir = Path(task_params_from_db["join_output_dir"])
            dprint(f"[JOIN_CLIPS] Task {task_id}: Using orchestrator output dir: {join_clips_dir}")
        else:
            join_clips_dir = main_output_dir_base / "join_clips" / task_id

        join_clips_dir.mkdir(parents=True, exist_ok=True)

        # Download videos if they are URLs (e.g., Supabase storage URLs)
        dprint(f"[JOIN_CLIPS] Task {task_id}: Checking if videos need to be downloaded...")
        starting_video_path = download_video_if_url(
            starting_video_path,
            download_target_dir=join_clips_dir,
            task_id_for_logging=task_id,
            descriptive_name="starting_video"
        )
        ending_video_path = download_video_if_url(
            ending_video_path,
            download_target_dir=join_clips_dir,
            task_id_for_logging=task_id,
            descriptive_name="ending_video"
        )

        # Convert to Path objects and validate existence
        starting_video = Path(starting_video_path)
        ending_video = Path(ending_video_path)

        if not starting_video.exists():
            error_msg = f"Starting video not found: {starting_video_path}"
            dprint(f"[JOIN_CLIPS_ERROR] Task {task_id}: {error_msg}")
            return False, error_msg

        if not ending_video.exists():
            error_msg = f"Ending video not found: {ending_video_path}"
            dprint(f"[JOIN_CLIPS_ERROR] Task {task_id}: {error_msg}")
            return False, error_msg

        dprint(f"[JOIN_CLIPS] Task {task_id}: Parameters validated")
        dprint(f"[JOIN_CLIPS]   Starting video: {starting_video}")
        dprint(f"[JOIN_CLIPS]   Ending video: {ending_video}")
        dprint(f"[JOIN_CLIPS]   Context frames: {context_frame_count}")
        dprint(f"[JOIN_CLIPS]   Gap frames: {gap_frame_count}")
        if aspect_ratio:
            dprint(f"[JOIN_CLIPS]   Target aspect ratio: {aspect_ratio}")

        # --- 2a. Ensure Videos are at Target FPS ---
        # use_input_video_fps: If True, keep original FPS. If False, downsample to fps param (default 16)
        use_input_video_fps = task_params_from_db.get("use_input_video_fps", False)
        
        if use_input_video_fps:
            dprint(f"[JOIN_CLIPS] Task {task_id}: use_input_video_fps=True, keeping original video FPS")
            # Don't convert - will use input video's FPS
        else:
            # Downsample to target FPS (default 16)
            # Note: Use 'or' to handle explicit None values (get() only returns default if key is missing)
            target_fps_param = task_params_from_db.get("fps") or 16
            dprint(f"[JOIN_CLIPS] Task {task_id}: use_input_video_fps=False, ensuring videos are at {target_fps_param} FPS...")
            
            starting_video_fps = ensure_video_fps(
                video_path=starting_video,
                target_fps=target_fps_param,
                output_dir=join_clips_dir,
                dprint_func=dprint
            )
            if not starting_video_fps:
                error_msg = f"Failed to ensure starting video is at {target_fps_param} fps"
                dprint(f"[JOIN_CLIPS_ERROR] Task {task_id}: {error_msg}")
                return False, error_msg
            starting_video = starting_video_fps
            
            ending_video_fps = ensure_video_fps(
                video_path=ending_video,
                target_fps=target_fps_param,
                output_dir=join_clips_dir,
                dprint_func=dprint
            )
            if not ending_video_fps:
                error_msg = f"Failed to ensure ending video is at {target_fps_param} fps"
                dprint(f"[JOIN_CLIPS_ERROR] Task {task_id}: {error_msg}")
                return False, error_msg
            ending_video = ending_video_fps

        # --- 2b. Standardize Videos to Target Aspect Ratio (if specified) ---
        if aspect_ratio:
            dprint(f"[JOIN_CLIPS] Task {task_id}: Standardizing videos to aspect ratio {aspect_ratio}...")

            # Standardize starting video
            standardized_start_path = join_clips_dir / f"start_standardized_{task_id}.mp4"
            result = standardize_video_aspect_ratio(
                input_video_path=starting_video,
                output_video_path=standardized_start_path,
                target_aspect_ratio=aspect_ratio,
                task_id_for_logging=task_id,
                dprint=dprint
            )
            if result is None:
                error_msg = f"Failed to standardize starting video to aspect ratio {aspect_ratio}"
                dprint(f"[JOIN_CLIPS_ERROR] Task {task_id}: {error_msg}")
                return False, error_msg
            starting_video = standardized_start_path
            dprint(f"[JOIN_CLIPS] Task {task_id}: Starting video standardized")

            # Standardize ending video
            standardized_end_path = join_clips_dir / f"end_standardized_{task_id}.mp4"
            result = standardize_video_aspect_ratio(
                input_video_path=ending_video,
                output_video_path=standardized_end_path,
                target_aspect_ratio=aspect_ratio,
                task_id_for_logging=task_id,
                dprint=dprint
            )
            if result is None:
                error_msg = f"Failed to standardize ending video to aspect ratio {aspect_ratio}"
                dprint(f"[JOIN_CLIPS_ERROR] Task {task_id}: {error_msg}")
                return False, error_msg
            ending_video = standardized_end_path
            dprint(f"[JOIN_CLIPS] Task {task_id}: Ending video standardized")
        else:
            # No explicit aspect ratio specified - auto-standardize ending video to match starting video
            dprint(f"[JOIN_CLIPS] Task {task_id}: Checking if videos have matching aspect ratios...")

            # Get dimensions of both videos
            try:
                import subprocess

                def get_video_dimensions(video_path):
                    probe_cmd = [
                        'ffprobe', '-v', 'error',
                        '-select_streams', 'v:0',
                        '-show_entries', 'stream=width,height',
                        '-of', 'csv=p=0',
                        str(video_path)
                    ]
                    result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10)
                    if result.returncode != 0:
                        return None, None
                    width_str, height_str = result.stdout.strip().split(',')
                    return int(width_str), int(height_str)

                start_w, start_h = get_video_dimensions(starting_video)
                end_w, end_h = get_video_dimensions(ending_video)

                if start_w and start_h and end_w and end_h:
                    start_aspect = start_w / start_h
                    end_aspect = end_w / end_h

                    dprint(f"[JOIN_CLIPS] Task {task_id}: Starting video: {start_w}x{start_h} (aspect: {start_aspect:.3f})")
                    dprint(f"[JOIN_CLIPS] Task {task_id}: Ending video: {end_w}x{end_h} (aspect: {end_aspect:.3f})")

                    # If aspect ratios differ by more than 1%, standardize ending video to match starting video
                    if abs(start_aspect - end_aspect) > 0.01:
                        dprint(f"[JOIN_CLIPS] Task {task_id}: Aspect ratios differ - standardizing ending video to match starting video")

                        # Calculate aspect ratio string from starting video
                        # Use common aspect ratios or create from dimensions
                        if abs(start_aspect - 16/9) < 0.01:
                            auto_aspect_ratio = "16:9"
                        elif abs(start_aspect - 9/16) < 0.01:
                            auto_aspect_ratio = "9:16"
                        elif abs(start_aspect - 1.0) < 0.01:
                            auto_aspect_ratio = "1:1"
                        elif abs(start_aspect - 4/3) < 0.01:
                            auto_aspect_ratio = "4:3"
                        elif abs(start_aspect - 21/9) < 0.01:
                            auto_aspect_ratio = "21:9"
                        else:
                            # Use exact dimensions as ratio
                            auto_aspect_ratio = f"{start_w}:{start_h}"

                        dprint(f"[JOIN_CLIPS] Task {task_id}: Auto-detected aspect ratio: {auto_aspect_ratio}")

                        standardized_end_path = join_clips_dir / f"end_standardized_{task_id}.mp4"
                        result = standardize_video_aspect_ratio(
                            input_video_path=ending_video,
                            output_video_path=standardized_end_path,
                            target_aspect_ratio=auto_aspect_ratio,
                            task_id_for_logging=task_id,
                            dprint=dprint
                        )
                        if result is None:
                            dprint(f"[JOIN_CLIPS_WARNING] Task {task_id}: Failed to auto-standardize ending video, proceeding with original")
                        else:
                            ending_video = standardized_end_path
                            dprint(f"[JOIN_CLIPS] Task {task_id}: Ending video standardized to match starting video")
                    else:
                        dprint(f"[JOIN_CLIPS] Task {task_id}: Videos have matching aspect ratios, no standardization needed")

            except Exception as e:
                dprint(f"[JOIN_CLIPS_WARNING] Task {task_id}: Could not check video dimensions: {e}")
                dprint(f"[JOIN_CLIPS] Task {task_id}: Proceeding with original videos")

        # --- 3. Extract Video Properties ---
        try:
            start_frame_count, start_fps = get_video_frame_count_and_fps(str(starting_video))
            end_frame_count, end_fps = get_video_frame_count_and_fps(str(ending_video))

            dprint(f"[JOIN_CLIPS] Task {task_id}: Starting video - {start_frame_count} frames @ {start_fps} fps")
            dprint(f"[JOIN_CLIPS] Task {task_id}: Ending video - {end_frame_count} frames @ {end_fps} fps")

        except Exception as e:
            error_msg = f"Failed to extract video properties: {e}"
            dprint(f"[JOIN_CLIPS_ERROR] Task {task_id}: {error_msg}")
            return False, error_msg

        # Validate that frame counts were detected (WebM and some codecs may fail)
        if start_frame_count is None:
            error_msg = f"Could not detect frame count for starting video: {starting_video}. The video may be corrupt, empty, or in an unsupported format."
            dprint(f"[JOIN_CLIPS_ERROR] Task {task_id}: {error_msg}")
            return False, error_msg

        if end_frame_count is None:
            error_msg = f"Could not detect frame count for ending video: {ending_video}. The video may be corrupt, empty, or in an unsupported format."
            dprint(f"[JOIN_CLIPS_ERROR] Task {task_id}: {error_msg}")
            return False, error_msg

        # Validate context frame counts
        if context_frame_count > start_frame_count:
            error_msg = f"context_frame_count ({context_frame_count}) exceeds starting video frame count ({start_frame_count})"
            dprint(f"[JOIN_CLIPS_ERROR] Task {task_id}: {error_msg}")
            return False, error_msg

        if context_frame_count > end_frame_count:
            error_msg = f"context_frame_count ({context_frame_count}) exceeds ending video frame count ({end_frame_count})"
            dprint(f"[JOIN_CLIPS_ERROR] Task {task_id}: {error_msg}")
            return False, error_msg

        # Set target FPS based on use_input_video_fps setting
        if use_input_video_fps:
            target_fps = start_fps
            dprint(f"[JOIN_CLIPS] Task {task_id}: Using input video FPS: {target_fps}")
        else:
            target_fps = task_params_from_db.get("fps") or 16
            dprint(f"[JOIN_CLIPS] Task {task_id}: Using target FPS: {target_fps}")

        # --- 4. Calculate gap sizes first (needed for REPLACE mode context extraction) ---
        # Calculate VACE quantization adjustments early so we know exact gap sizes
        quantization_result = _calculate_vace_quantization(
            context_frame_count=context_frame_count,
            gap_frame_count=gap_frame_count,
            replace_mode=replace_mode
        )
        gap_for_guide = quantization_result['gap_for_guide']
        quantization_shift = quantization_result['quantization_shift']
        
        # Calculate gap split for REPLACE mode
        gap_from_clip1 = gap_for_guide // 2 if replace_mode else 0
        gap_from_clip2 = (gap_for_guide - gap_from_clip1) if replace_mode else 0
        
        dprint(f"[JOIN_CLIPS] Task {task_id}: Mode={'REPLACE' if replace_mode else 'INSERT'}, gap_for_guide={gap_for_guide}")
        if replace_mode:
            dprint(f"[JOIN_CLIPS] Task {task_id}: REPLACE will remove {gap_from_clip1} from clip1, {gap_from_clip2} from clip2")

        # --- 5. Extract Context Frames ---
        dprint(f"[JOIN_CLIPS] Task {task_id}: Extracting context frames...")

        try:
            # Extract all frames from both videos
            start_all_frames = sm_extract_frames_from_video(str(starting_video), dprint_func=dprint)
            if not start_all_frames or len(start_all_frames) < context_frame_count:
                error_msg = f"Failed to extract {context_frame_count} frames from starting video"
                dprint(f"[JOIN_CLIPS_ERROR] Task {task_id}: {error_msg}")
                return False, error_msg

            end_all_frames = sm_extract_frames_from_video(str(ending_video), dprint_func=dprint)
            if not end_all_frames or len(end_all_frames) < context_frame_count:
                error_msg = f"Failed to extract {context_frame_count} frames from ending video"
                dprint(f"[JOIN_CLIPS_ERROR] Task {task_id}: {error_msg}")
                return False, error_msg

            if replace_mode:
                # REPLACE mode: Context comes from OUTSIDE the gap region
                # Gap is removed from boundary, context is adjacent to (but outside) the gap
                #
                # clip1: [...][context 8][gap N removed]
                # clip2: [gap M removed][context 8][...]
                #
                # Validate we have enough frames
                min_clip1_frames = context_frame_count + gap_from_clip1
                min_clip2_frames = context_frame_count + gap_from_clip2
                
                if len(start_all_frames) < min_clip1_frames:
                    error_msg = f"Starting video too short for REPLACE mode: need {min_clip1_frames} frames, have {len(start_all_frames)}"
                    dprint(f"[JOIN_CLIPS_ERROR] Task {task_id}: {error_msg}")
                    return False, error_msg
                
                if len(end_all_frames) < min_clip2_frames:
                    error_msg = f"Ending video too short for REPLACE mode: need {min_clip2_frames} frames, have {len(end_all_frames)}"
                    dprint(f"[JOIN_CLIPS_ERROR] Task {task_id}: {error_msg}")
                    return False, error_msg
                
                # Context from clip1: frames BEFORE the gap (not the last frames)
                # If removing last N frames, context is the 8 frames before that
                context_start_idx = len(start_all_frames) - gap_from_clip1 - context_frame_count
                context_end_idx = len(start_all_frames) - gap_from_clip1
                start_context_frames = start_all_frames[context_start_idx:context_end_idx]
                
                dprint(f"[JOIN_CLIPS] Task {task_id}: REPLACE mode - clip1 context from frames [{context_start_idx}:{context_end_idx}]")
                dprint(f"[JOIN_CLIPS] Task {task_id}:   (frames {context_end_idx} to {len(start_all_frames)-1} will be removed as gap)")
                
                # Context from clip2: frames AFTER the gap (not the first frames)
                # If removing first M frames, context is the 8 frames after that
                end_context_frames = end_all_frames[gap_from_clip2:gap_from_clip2 + context_frame_count]
                
                dprint(f"[JOIN_CLIPS] Task {task_id}: REPLACE mode - clip2 context from frames [{gap_from_clip2}:{gap_from_clip2 + context_frame_count}]")
                dprint(f"[JOIN_CLIPS] Task {task_id}:   (frames 0 to {gap_from_clip2-1} will be removed as gap)")
            else:
                # INSERT mode: Context is at the boundary (last/first frames)
                # No frames are removed, we're just inserting new frames between clips
                start_context_frames = start_all_frames[-context_frame_count:]
                end_context_frames = end_all_frames[:context_frame_count]
                
                dprint(f"[JOIN_CLIPS] Task {task_id}: INSERT mode - clip1 context from last {context_frame_count} frames")
                dprint(f"[JOIN_CLIPS] Task {task_id}: INSERT mode - clip2 context from first {context_frame_count} frames")

        except Exception as e:
            error_msg = f"Failed to extract context frames: {e}"
            dprint(f"[JOIN_CLIPS_ERROR] Task {task_id}: {error_msg}")
            import traceback
            traceback.print_exc()
            return False, error_msg

        # Get resolution from first frame or task params
        first_frame = start_context_frames[0]
        frame_height, frame_width = first_frame.shape[:2]
        detected_res_wh = (frame_width, frame_height)

        # Determine resolution: check use_input_video_resolution flag first, then explicit resolution param
        use_input_video_resolution = task_params_from_db.get("use_input_video_resolution", False)
        
        if use_input_video_resolution:
            parsed_res_wh = detected_res_wh
            dprint(f"[JOIN_CLIPS] Task {task_id}: use_input_video_resolution=True, using detected resolution: {parsed_res_wh}")
        elif "resolution" in task_params_from_db:
            resolution_list = task_params_from_db["resolution"]
            parsed_res_wh = (resolution_list[0], resolution_list[1])
            dprint(f"[JOIN_CLIPS] Task {task_id}: Using resolution override: {parsed_res_wh}")
        else:
            parsed_res_wh = detected_res_wh
            dprint(f"[JOIN_CLIPS] Task {task_id}: Using detected resolution: {parsed_res_wh}")

        # --- 6. Build Guide and Mask Videos (using shared helper) ---
        # (quantization already calculated above in step 4)
        quantized_total_frames = quantization_result['total_frames']
        
        dprint(f"[JOIN_CLIPS] Task {task_id}: Total VACE frames: {quantized_total_frames} (context + gap + context)")
        if quantization_shift > 0:
            dprint(f"[JOIN_CLIPS] Task {task_id}: VACE quantization applied: gap adjusted {gap_frame_count} → {gap_for_guide} frames")

        # Determine inserted frames for gap preservation (if enabled)
        # Both modes now use the same logic: insert boundary frames at 1/3 and 2/3 of gap
        gap_inserted_frames = {}
        
        if keep_bridging_images:
            if len(start_context_frames) > 0 and len(end_context_frames) > 0:
                # Anchor 1: End of first video (last frame of start context)
                anchor1 = start_context_frames[-1]
                idx1 = gap_for_guide // 3
                
                # Anchor 2: Start of second video (first frame of end context)
                anchor2 = end_context_frames[0]
                idx2 = (gap_for_guide * 2) // 3
                
                # Only insert if gap is large enough to separate them
                if gap_for_guide >= 3 and idx1 < idx2:
                    gap_inserted_frames[idx1] = anchor1
                    gap_inserted_frames[idx2] = anchor2
                    dprint(f"[JOIN_CLIPS] Task {task_id}: keep_bridging_images=True: Using start_clip[-1] at gap[{idx1}] and end_clip[0] at gap[{idx2}]")
                else:
                    dprint(f"[JOIN_CLIPS] Task {task_id}: Gap too small ({gap_for_guide}) for equidistant anchors, skipping")
            else:
                dprint(f"[JOIN_CLIPS_WARNING] Task {task_id}: keep_bridging_images=True but contexts empty")

        # Create guide/mask with adjusted gap
        try:
            created_guide_video, created_mask_video, guide_frame_count = create_guide_and_mask_for_generation(
                context_frames_before=start_context_frames,
                context_frames_after=end_context_frames,
                gap_frame_count=gap_for_guide,  # Use quantization-adjusted gap
                resolution_wh=parsed_res_wh,
                fps=target_fps,
                output_dir=join_clips_dir,
                task_id=task_id,
                filename_prefix="join",
                replace_mode=replace_mode,
                gap_inserted_frames=gap_inserted_frames,
                dprint=dprint
            )
        except Exception as e:
            error_msg = f"Failed to create guide/mask videos: {e}"
            dprint(f"[JOIN_CLIPS_ERROR] Task {task_id}: {error_msg}")
            import traceback
            traceback.print_exc()
            return False, error_msg

        if guide_frame_count != quantized_total_frames:
            dprint(f"[JOIN_CLIPS] Task {task_id}: Guide/mask total frame count ({guide_frame_count}) "
                   f"differs from quantized expectation ({quantized_total_frames}). Using actual count.")

        total_frames = guide_frame_count

        # --- 6. Prepare Generation Parameters (using shared helper) ---
        dprint(f"[JOIN_CLIPS] Task {task_id}: Preparing generation parameters...")

        # Determine model (default to Lightning baseline for fast generation)
        model = task_params_from_db.get("model", "wan_2_2_vace_lightning_baseline_2_2_2")

        # Ensure prompt is valid
        prompt = ensure_valid_prompt(prompt)
        negative_prompt = ensure_valid_negative_prompt(
            task_params_from_db.get("negative_prompt", "")
        )

        # Extract additional_loras for logging (if present)
        additional_loras = task_params_from_db.get("additional_loras", {})
        if additional_loras:
            dprint(f"[JOIN_CLIPS] Task {task_id}: Found {len(additional_loras)} additional LoRAs: {list(additional_loras.keys())}")

        # Extract phase_config for logging (if present)
        phase_config = task_params_from_db.get("phase_config")
        if phase_config:
            dprint(f"[JOIN_CLIPS] Task {task_id}: Found phase_config: {json.dumps(phase_config, default=str)[:100]}...")

        # Use shared helper to prepare standardized VACE parameters
        generation_params = prepare_vace_generation_params(
            guide_video_path=created_guide_video,
            mask_video_path=created_mask_video,
            total_frames=total_frames,
            resolution_wh=parsed_res_wh,
            prompt=prompt,
            negative_prompt=negative_prompt,
            model=model,
            seed=task_params_from_db.get("seed", -1),
            task_params=task_params_from_db  # Pass through for optional param merging (includes additional_loras)
        )

        dprint(f"[JOIN_CLIPS] Task {task_id}: Generation parameters prepared")
        dprint(f"[JOIN_CLIPS]   Model: {model}")
        dprint(f"[JOIN_CLIPS]   Video length: {total_frames} frames")
        dprint(f"[JOIN_CLIPS]   Resolution: {parsed_res_wh}")

        # Log LoRA settings
        if generation_params.get("additional_loras"):
            dprint(f"[JOIN_CLIPS]   Additional LoRAs: {len(generation_params['additional_loras'])} configured")

        # --- 7. Submit to Generation Queue ---
        dprint(f"[JOIN_CLIPS] Task {task_id}: Submitting to generation queue...")

        try:
            # Import GenerationTask from correct location
            from headless_model_management import GenerationTask

            generation_task = GenerationTask(
                id=task_id,
                model=model,
                prompt=prompt,
                parameters=generation_params,
                priority=task_params_from_db.get("priority", 0)
            )

            # Submit task using correct method
            submitted_task_id = task_queue.submit_task(generation_task)
            dprint(f"[JOIN_CLIPS] Task {task_id}: Submitted to generation queue as {submitted_task_id}")

            # Wait for completion using polling pattern (same as direct queue tasks)
            # Allow timeout override via task params (default: 30 minutes to handle slow model loading)
            max_wait_time = task_params_from_db.get("max_wait_time", 1800)  # 30 minute default timeout
            wait_interval = 2  # Check every 2 seconds
            elapsed_time = 0
            dprint(f"[JOIN_CLIPS] Task {task_id}: Waiting for generation (timeout: {max_wait_time}s)")

            while elapsed_time < max_wait_time:
                status = task_queue.get_task_status(task_id)

                if status is None:
                    error_msg = "Task status became None during processing"
                    dprint(f"[JOIN_CLIPS_ERROR] Task {task_id}: {error_msg}")
                    return False, error_msg

                if status.status == "completed":
                    transition_video_path = status.result_path
                    processing_time = status.processing_time or 0
                    dprint(f"[JOIN_CLIPS] Task {task_id}: Generation completed successfully in {processing_time:.1f}s")
                    dprint(f"[JOIN_CLIPS] Task {task_id}: Transition video: {transition_video_path}")

                    # IMPORTANT: Check actual frame count vs expected
                    # VACE may generate fewer frames than requested (e.g., 45 instead of 48)
                    actual_transition_frames, _ = get_video_frame_count_and_fps(transition_video_path)
                    
                    # Calculate safe blend_frames based on actual transition length
                    # Transition structure: [context_before][gap][context_after]
                    # We need at least blend_frames at each end for crossfading
                    expected_total = total_frames
                    
                    if actual_transition_frames != expected_total:
                        dprint(f"[JOIN_CLIPS] Task {task_id}: ⚠️  Frame count mismatch! Expected {expected_total}, got {actual_transition_frames}")
                        
                        # Calculate the difference
                        frame_diff = expected_total - actual_transition_frames
                        
                        if frame_diff > 0:
                            # VACE generated fewer frames than expected
                            # This could cause misalignment - we need to adjust blend_frames
                            dprint(f"[JOIN_CLIPS] Task {task_id}: VACE generated {frame_diff} fewer frames than expected")
                            
                            # Maximum safe blend = half of actual transition (to leave room for gap)
                            max_safe_blend = actual_transition_frames // 4  # Conservative: 1/4 of total at each end
                            
                            if context_frame_count > max_safe_blend:
                                dprint(f"[JOIN_CLIPS] Task {task_id}: ⚠️  Reducing blend_frames from {context_frame_count} to {max_safe_blend} for safety")
                        
                        total_frames = actual_transition_frames  # Use actual count
                    else:
                        max_safe_blend = context_frame_count

                    # --- 8. Concatenate Full Clips with Transition ---
                    dprint(f"[JOIN_CLIPS] Task {task_id}: Concatenating full clips with transition...")

                    try:
                        import subprocess
                        import tempfile

                        # Create trimmed versions of the original clips
                        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False, dir=join_clips_dir) as clip1_trimmed_file:
                            clip1_trimmed_path = Path(clip1_trimmed_file.name)

                        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False, dir=join_clips_dir) as clip2_trimmed_file:
                            clip2_trimmed_path = Path(clip2_trimmed_file.name)

                        # Trimming uses gap_from_clip1 and gap_from_clip2 calculated earlier
                        # REPLACE mode: Remove gap frames from boundary, context remains and blends
                        # INSERT mode: Don't remove any frames, just insert transition
                        
                        # For proper blending, blend over the full context region (or max safe if VACE returned fewer frames)
                        blend_frames = min(context_frame_count, max_safe_blend)
                        dprint(f"[JOIN_CLIPS] Task {task_id}: Blend frames: {blend_frames} (context={context_frame_count}, max_safe={max_safe_blend})")
                        
                        # Use pre-calculated gap sizes (gap_from_clip1, gap_from_clip2 from step 4)
                        frames_to_remove_clip1 = gap_from_clip1  # 0 for INSERT mode
                        frames_to_keep_clip1 = start_frame_count - frames_to_remove_clip1
                        
                        if replace_mode:
                            dprint(f"[JOIN_CLIPS] Task {task_id}: REPLACE mode - removing {gap_from_clip1} gap frames from clip1")
                        else:
                            dprint(f"[JOIN_CLIPS] Task {task_id}: INSERT mode - keeping all frames from clip1")
                        dprint(f"[JOIN_CLIPS] Task {task_id}:   Keeping {frames_to_keep_clip1}/{start_frame_count} frames from clip1")

                        # Use common frame extraction: frames 0 to (frames_to_keep_clip1 - 1)
                        trimmed_clip1 = extract_frame_range_to_video(
                            source_video=starting_video,
                            output_path=clip1_trimmed_path,
                            start_frame=0,
                            end_frame=frames_to_keep_clip1 - 1,
                            fps=start_fps,
                            dprint_func=dprint
                        )
                        if not trimmed_clip1:
                            raise ValueError(f"Failed to trim clip1 (frames 0-{frames_to_keep_clip1 - 1})")

                        # Clip2 trimming uses pre-calculated gap_from_clip2
                        frames_to_skip_clip2 = gap_from_clip2  # 0 for INSERT mode
                        
                        if replace_mode:
                            dprint(f"[JOIN_CLIPS] Task {task_id}: REPLACE mode - skipping {gap_from_clip2} gap frames from clip2")
                        else:
                            dprint(f"[JOIN_CLIPS] Task {task_id}: INSERT mode - keeping all frames from clip2")
                        
                        frames_remaining_clip2 = end_frame_count - frames_to_skip_clip2
                        dprint(f"[JOIN_CLIPS] Task {task_id}:   Keeping {frames_remaining_clip2}/{end_frame_count} frames from clip2")
                        
                        # Log net frame change summary
                        total_gap_removed = frames_to_remove_clip1 + frames_to_skip_clip2
                        # Transition = context + gap + context, but context regions overlap with clips via blend
                        # Effective new frames = gap_for_guide (the middle portion)
                        effective_frames_added = gap_for_guide
                        net_frame_change = effective_frames_added - total_gap_removed
                        dprint(f"[JOIN_CLIPS] Task {task_id}: === NET FRAME CHANGE ===")
                        dprint(f"[JOIN_CLIPS] Task {task_id}:   Gap frames removed from clips: {total_gap_removed}")
                        dprint(f"[JOIN_CLIPS] Task {task_id}:   Gap frames generated by VACE: {effective_frames_added}")
                        dprint(f"[JOIN_CLIPS] Task {task_id}:   Net change: {net_frame_change:+d} frames")

                        # Use common frame extraction: skip first frames_to_skip_clip2 frames
                        trimmed_clip2 = extract_frame_range_to_video(
                            source_video=ending_video,
                            output_path=clip2_trimmed_path,
                            start_frame=frames_to_skip_clip2,
                            end_frame=None,  # All remaining frames
                            fps=end_fps,
                            dprint_func=dprint
                        )
                        if not trimmed_clip2:
                            raise ValueError(f"Failed to trim clip2 (skip first {frames_to_skip_clip2} frames)")

                        # Final concatenated output
                        final_output_path = join_clips_dir / f"joined_{task_id}.mp4"

                        # Use generalized stitch function with frame-level crossfade blending
                        # This matches the approach used in travel_between_images.py
                        dprint(f"[JOIN_CLIPS] Task {task_id}: Stitching videos with {blend_frames}-frame crossfade at each boundary")
                        dprint(f"[JOIN_CLIPS] Task {task_id}: === FRAME ALIGNMENT DEBUG ===")
                        dprint(f"[JOIN_CLIPS] Task {task_id}: Transition video has {total_frames} frames")
                        dprint(f"[JOIN_CLIPS] Task {task_id}: Transition last {blend_frames} frames: [{total_frames - blend_frames}:{total_frames}]")
                        dprint(f"[JOIN_CLIPS] Task {task_id}: Clip2 trimmed starts at original frame {frames_to_skip_clip2}")
                        dprint(f"[JOIN_CLIPS] Task {task_id}: Clip2 trimmed first {blend_frames} frames = clip2[{frames_to_skip_clip2}:{frames_to_skip_clip2 + blend_frames}]")

                        video_paths = [
                            clip1_trimmed_path,
                            Path(transition_video_path),
                            clip2_trimmed_path
                        ]

                        # Blend between clip1→transition and transition→clip2
                        blend_frame_counts = [blend_frames, blend_frames]

                        try:
                            created_video = sm_stitch_videos_with_crossfade(
                                video_paths=video_paths,
                                blend_frame_counts=blend_frame_counts,
                                output_path=final_output_path,
                                fps=target_fps,
                                crossfade_mode="linear_sharp",
                                crossfade_sharp_amt=0.3,
                                dprint=dprint
                            )
                            
                            # Validate the stitch succeeded
                            if created_video is None:
                                raise ValueError("stitch_videos_with_crossfade returned None")
                            
                            dprint(f"[JOIN_CLIPS] Task {task_id}: Successfully stitched videos with crossfade blending")
                        except Exception as e:
                            raise ValueError(f"Failed to stitch videos with crossfade: {e}") from e

                        # Verify final output exists and is valid
                        if not final_output_path.exists():
                            raise ValueError(f"Final concatenated video does not exist: {final_output_path}")
                        
                        file_size = final_output_path.stat().st_size
                        if file_size == 0:
                            raise ValueError(f"Final concatenated video is empty (0 bytes)")
                        
                        dprint(f"[JOIN_CLIPS] Task {task_id}: Final video validated: {file_size} bytes")

                        # Extract poster image/thumbnail from the final video
                        poster_output_path = final_output_path.with_suffix('.jpg')
                        try:
                            # Extract first frame as poster
                            poster_frame_index = 0

                            dprint(f"[JOIN_CLIPS] Task {task_id}: Extracting poster image (first frame)")

                            if save_frame_from_video(
                                final_output_path,
                                poster_frame_index,
                                poster_output_path,
                                parsed_res_wh
                            ):
                                dprint(f"[JOIN_CLIPS] Task {task_id}: Poster image saved: {poster_output_path}")
                            else:
                                dprint(f"[JOIN_CLIPS] Task {task_id}: Warning: Failed to extract poster image")
                        except Exception as poster_error:
                            dprint(f"[JOIN_CLIPS] Task {task_id}: Warning: Poster extraction failed: {poster_error}")

                        # Clean up temporary files (unless debug mode is enabled)
                        debug_mode = task_params_from_db.get("debug", False)
                        if debug_mode:
                            dprint(f"[JOIN_CLIPS] Task {task_id}: Debug mode enabled - preserving intermediate files:")
                            dprint(f"[JOIN_CLIPS]   Clip1 trimmed: {clip1_trimmed_path}")
                            dprint(f"[JOIN_CLIPS]   Transition: {transition_video_path}")
                            dprint(f"[JOIN_CLIPS]   Clip2 trimmed: {clip2_trimmed_path}")
                        else:
                            try:
                                clip1_trimmed_path.unlink()
                                clip2_trimmed_path.unlink()
                                Path(transition_video_path).unlink()  # Remove transition-only video
                                dprint(f"[JOIN_CLIPS] Task {task_id}: Cleaned up temporary files")
                            except Exception as cleanup_error:
                                dprint(f"[JOIN_CLIPS] Warning: Cleanup failed: {cleanup_error}")

                        dprint(f"[JOIN_CLIPS] Task {task_id}: Successfully created final joined video")
                        dprint(f"[JOIN_CLIPS] Task {task_id}: Output: {final_output_path}")

                        return True, str(final_output_path)

                    except Exception as concat_error:
                        error_msg = f"Failed to concatenate full clips: {concat_error}"
                        dprint(f"[JOIN_CLIPS_ERROR] Task {task_id}: {error_msg}")
                        import traceback
                        traceback.print_exc()
                        # Return the transition video as fallback
                        dprint(f"[JOIN_CLIPS] Task {task_id}: Returning transition video as fallback")
                        return True, transition_video_path

                elif status.status == "failed":
                    error_msg = status.error_message or "Generation failed without specific error message"
                    dprint(f"[JOIN_CLIPS_ERROR] Task {task_id}: Generation failed - {error_msg}")
                    return False, error_msg

                else:
                    # Still processing
                    time.sleep(wait_interval)
                    elapsed_time += wait_interval

            # Timeout reached
            error_msg = f"Processing timeout after {max_wait_time} seconds"
            dprint(f"[JOIN_CLIPS_ERROR] Task {task_id}: {error_msg}")
            return False, error_msg

        except Exception as e:
            error_msg = f"Failed to submit/complete generation task: {e}"
            dprint(f"[JOIN_CLIPS_ERROR] Task {task_id}: {error_msg}")
            import traceback
            traceback.print_exc()
            return False, error_msg

    except Exception as e:
        error_msg = f"Unexpected error in join_clips handler: {e}"
        dprint(f"[JOIN_CLIPS_ERROR] Task {task_id}: {error_msg}")
        import traceback
        traceback.print_exc()
        return False, error_msg
