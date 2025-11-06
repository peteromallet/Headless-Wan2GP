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
        gap_frame_count: Number of frames to insert/replace
        replace_mode: Whether gap replaces frames (True) or inserts between (False)

    Returns:
        dict with:
            - total_frames: Actual frame count VACE will generate
            - gap_for_guide: Adjusted gap to use in guide/mask creation
            - quantization_shift: Number of frames dropped by VACE (0 if no quantization)
    """
    # Calculate desired total
    if replace_mode:
        desired_total = context_frame_count * 2
    else:
        desired_total = context_frame_count * 2 + gap_frame_count

    # Apply VACE quantization (4n + 1)
    actual_total = ((desired_total - 1) // 4) * 4 + 1
    quantization_shift = desired_total - actual_total

    # Adjust gap to account for dropped frames
    gap_for_guide = gap_frame_count - quantization_shift

    return {
        'total_frames': actual_total,
        'gap_for_guide': gap_for_guide,
        'quantization_shift': quantization_shift,
    }


def _calculate_replace_mode_clip2_skip(
    context_frame_count: int,
    gap_for_guide: int,
    blend_frames: int,
    quantization_shift: int
) -> int:
    """
    Calculate how many frames to skip from clip2 start in REPLACE mode.

    REPLACE mode preserves portions of both clip boundaries in the bridge.
    Due to VACE quantization, the preserved section gets SHIFTED earlier in
    the original clip by exactly quantization_shift frames.

    Example: context=24, gap=16, blend=3
        - Guide requests: 48 frames total (24 + 24)
        - VACE generates: 45 frames (quantization_shift = 3)
        - Guide specifies: preserve clip2[7:24] (17 frames)
        - VACE actually preserves: clip2[4:21] (shifted by -3)
        - To align crossfade, clip2 must start at frame 18 (not 21)

    Args:
        context_frame_count: Number of frames in original context extraction
        gap_for_guide: Adjusted gap used in guide/mask (after quantization)
        blend_frames: Number of frames to blend in crossfade
        quantization_shift: Frames dropped by VACE quantization

    Returns:
        Number of frames to skip from start of clip2
    """
    # How the gap is split between before/after contexts
    frames_replaced_from_after = gap_for_guide - (gap_for_guide // 2)

    # How many frames are preserved from the clip2 context
    preserved_frame_count = context_frame_count - frames_replaced_from_after

    # Base calculation: skip replaced frames + preserved frames - blend overlap
    base_skip = frames_replaced_from_after + (preserved_frame_count - blend_frames)

    # CRITICAL: VACE shifts the preserved section by quantization_shift
    # The bridge actually contains earlier frames than the guide specified
    actual_skip = base_skip - quantization_shift

    return actual_skip


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
            - model: Optional model override (defaults to lightning_baseline_2_2_2)
            - resolution: Optional [width, height] override
            - fps: Optional FPS override (defaults to 16)
            - max_wait_time: Optional timeout in seconds for generation (defaults to 1800s / 30 minutes)
            - use_causvid_lora: Optional bool to enable CausVid LoRA
            - use_lighti2x_lora: Optional bool to enable LightI2X LoRA
            - apply_reward_lora: Optional bool to enable Reward LoRA
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

        dprint(f"[JOIN_CLIPS] Task {task_id}: Mode: {'REPLACE' if replace_mode else 'INSERT'}")
        if replace_mode:
            dprint(f"[JOIN_CLIPS] Task {task_id}: gap_frame_count={gap_frame_count} frames will REPLACE boundary frames (no insertion)")

        # Validate required parameters
        if not starting_video_path:
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

        # Create working directory early so we can use it for downloads
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

        # --- 2. Standardize Videos to Target Aspect Ratio (if specified) ---
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

        # Validate context frame counts
        if context_frame_count > start_frame_count:
            error_msg = f"context_frame_count ({context_frame_count}) exceeds starting video frame count ({start_frame_count})"
            dprint(f"[JOIN_CLIPS_ERROR] Task {task_id}: {error_msg}")
            return False, error_msg

        if context_frame_count > end_frame_count:
            error_msg = f"context_frame_count ({context_frame_count}) exceeds ending video frame count ({end_frame_count})"
            dprint(f"[JOIN_CLIPS_ERROR] Task {task_id}: {error_msg}")
            return False, error_msg

        # Use FPS from task params, or default to starting video FPS
        target_fps = task_params_from_db.get("fps", start_fps)

        # --- 4. Extract Context Frames ---
        dprint(f"[JOIN_CLIPS] Task {task_id}: Extracting context frames...")

        try:
            # Extract all frames from starting video, take last N
            start_all_frames = sm_extract_frames_from_video(str(starting_video), dprint_func=dprint)
            if not start_all_frames or len(start_all_frames) < context_frame_count:
                error_msg = f"Failed to extract {context_frame_count} frames from starting video"
                dprint(f"[JOIN_CLIPS_ERROR] Task {task_id}: {error_msg}")
                return False, error_msg

            start_context_frames = start_all_frames[-context_frame_count:]
            dprint(f"[JOIN_CLIPS] Task {task_id}: Extracted {len(start_context_frames)} frames from end of starting video")
            dprint(f"[JOIN_CLIPS_ALIGNMENT] Start context frame indices in original video: [{start_frame_count - context_frame_count}:{start_frame_count}]")
            dprint(f"[JOIN_CLIPS_ALIGNMENT]   Example: frame index {start_frame_count - context_frame_count} → position 0 in context")
            dprint(f"[JOIN_CLIPS_ALIGNMENT]   Example: frame index {start_frame_count - 1} → position {context_frame_count - 1} in context")

            # Extract all frames from ending video, take first N
            end_all_frames = sm_extract_frames_from_video(str(ending_video), dprint_func=dprint)
            if not end_all_frames or len(end_all_frames) < context_frame_count:
                error_msg = f"Failed to extract {context_frame_count} frames from ending video"
                dprint(f"[JOIN_CLIPS_ERROR] Task {task_id}: {error_msg}")
                return False, error_msg

            end_context_frames = end_all_frames[:context_frame_count]
            dprint(f"[JOIN_CLIPS] Task {task_id}: Extracted {len(end_context_frames)} frames from beginning of ending video")
            dprint(f"[JOIN_CLIPS_ALIGNMENT] End context frame indices in original video: [0:{context_frame_count}]")
            dprint(f"[JOIN_CLIPS_ALIGNMENT]   Example: frame index 0 → position 0 in context")
            dprint(f"[JOIN_CLIPS_ALIGNMENT]   Example: frame index {context_frame_count - 1} → position {context_frame_count - 1} in context")

        except Exception as e:
            error_msg = f"Failed to extract context frames: {e}"
            dprint(f"[JOIN_CLIPS_ERROR] Task {task_id}: {error_msg}")
            import traceback
            traceback.print_exc()
            return False, error_msg

        # Get resolution from first frame or task params
        first_frame = start_context_frames[0]
        frame_height, frame_width = first_frame.shape[:2]

        # Allow resolution override from task params
        if "resolution" in task_params_from_db:
            resolution_list = task_params_from_db["resolution"]
            parsed_res_wh = (resolution_list[0], resolution_list[1])
            dprint(f"[JOIN_CLIPS] Task {task_id}: Using resolution override: {parsed_res_wh}")
        else:
            parsed_res_wh = (frame_width, frame_height)
            dprint(f"[JOIN_CLIPS] Task {task_id}: Using detected resolution: {parsed_res_wh}")

        # --- 5. Build Guide and Mask Videos (using shared helper) ---
        # Get regenerate_anchors setting (default True - regenerate anchor frames for smoother transitions)
        regenerate_anchors = task_params_from_db.get("regenerate_anchors", True)
        # Number of anchor frames to regenerate on each side (default 3 for smooth blending)
        num_anchor_frames = task_params_from_db.get("num_anchor_frames", 3)
        dprint(f"[JOIN_CLIPS] Task {task_id}: regenerate_anchors={regenerate_anchors}, num_anchor_frames={num_anchor_frames}")

        # Calculate VACE quantization adjustments
        quantization_result = _calculate_vace_quantization(
            context_frame_count=context_frame_count,
            gap_frame_count=gap_frame_count,
            replace_mode=replace_mode
        )

        total_frames = quantization_result['total_frames']
        gap_for_guide = quantization_result['gap_for_guide']
        quantization_shift = quantization_result['quantization_shift']

        if quantization_shift > 0:
            dprint(f"[JOIN_CLIPS] Task {task_id}: VACE quantization: {gap_frame_count + quantization_shift} → {total_frames} frames")
            dprint(f"[JOIN_CLIPS] Task {task_id}: Gap adjusted: {gap_frame_count} → {gap_for_guide} for guide/mask")

            if replace_mode:
                dprint(f"[JOIN_CLIPS] Task {task_id}: REPLACE mode - quantization will shift preserved section by -{quantization_shift} frames")
            else:
                dprint(f"[JOIN_CLIPS] Task {task_id}: INSERT mode - gap reduced to {gap_for_guide} frames")

        # Create guide/mask with adjusted gap
        try:
            created_guide_video, created_mask_video = create_guide_and_mask_for_generation(
                context_frames_before=start_context_frames,
                context_frames_after=end_context_frames,
                gap_frame_count=gap_for_guide,  # Use quantization-adjusted gap
                resolution_wh=parsed_res_wh,
                fps=target_fps,
                output_dir=join_clips_dir,
                task_id=task_id,
                filename_prefix="join",
                regenerate_anchors=regenerate_anchors,
                num_anchor_frames=num_anchor_frames,
                replace_mode=replace_mode,
                dprint=dprint
            )
        except Exception as e:
            error_msg = f"Failed to create guide/mask videos: {e}"
            dprint(f"[JOIN_CLIPS_ERROR] Task {task_id}: {error_msg}")
            import traceback
            traceback.print_exc()
            return False, error_msg

        # --- 6. Prepare Generation Parameters (using shared helper) ---
        dprint(f"[JOIN_CLIPS] Task {task_id}: Preparing generation parameters...")

        # Determine model (default to Lightning baseline for fast generation)
        model = task_params_from_db.get("model", "lightning_baseline_2_2_2")

        # Ensure prompt is valid
        prompt = ensure_valid_prompt(prompt)
        negative_prompt = ensure_valid_negative_prompt(
            task_params_from_db.get("negative_prompt", "")
        )

        # Extract additional_loras for logging (if present)
        additional_loras = task_params_from_db.get("additional_loras", {})
        if additional_loras:
            dprint(f"[JOIN_CLIPS] Task {task_id}: Found {len(additional_loras)} additional LoRAs: {list(additional_loras.keys())}")

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
        if generation_params.get("use_causvid_lora"):
            dprint(f"[JOIN_CLIPS]   CausVid LoRA: enabled")
        if generation_params.get("use_lighti2x_lora"):
            dprint(f"[JOIN_CLIPS]   LightI2X LoRA: enabled")
        if generation_params.get("apply_reward_lora"):
            dprint(f"[JOIN_CLIPS]   Reward LoRA: enabled")
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
                    if actual_transition_frames != total_frames:
                        dprint(f"[JOIN_CLIPS] Task {task_id}: ⚠️  Frame count mismatch! Expected {total_frames}, got {actual_transition_frames}")
                        dprint(f"[JOIN_CLIPS] Task {task_id}: Adjusting calculations for actual frame count")
                        total_frames = actual_transition_frames  # Use actual count for subsequent calculations

                    # --- 8. Concatenate Full Clips with Transition ---
                    dprint(f"[JOIN_CLIPS] Task {task_id}: Concatenating full clips with transition...")

                    try:
                        import subprocess
                        import tempfile

                        # --- Calculate blend_frames for crossfade transitions ---
                        # Default: num_anchor_frames (matches the regeneration strategy)
                        # Logic: We regenerate N anchor frames as critical transition zones,
                        # so we blend over those same N frames to complete the smoothing
                        default_blend_frames = num_anchor_frames if regenerate_anchors else (context_frame_count // 3)
                        blend_frames = task_params_from_db.get("blend_frames", default_blend_frames)

                        # Calculate maximum safe blend
                        if regenerate_anchors:
                            max_safe_blend = context_frame_count - num_anchor_frames
                        else:
                            max_safe_blend = context_frame_count

                        # Clamp blend_frames to safe range [0, max_safe_blend]
                        blend_frames = max(0, min(blend_frames, max_safe_blend))

                        dprint(f"[JOIN_CLIPS] Task {task_id}: Blend frames: {blend_frames} (default: {default_blend_frames}, max safe: {max_safe_blend})")

                        # Create trimmed versions of the original clips
                        # INSERT mode:
                        #   Clip1: Keep extra blend_frames for crossfade with transition
                        #   Clip2: Start earlier by blend_frames for crossfade with transition
                        # REPLACE mode:
                        #   Clip1: Remove more frames (context + replaced frames - blend)
                        #   Clip2: Skip more frames (context + replaced frames - blend)

                        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False, dir=join_clips_dir) as clip1_trimmed_file:
                            clip1_trimmed_path = Path(clip1_trimmed_file.name)

                        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False, dir=join_clips_dir) as clip2_trimmed_file:
                            clip2_trimmed_path = Path(clip2_trimmed_file.name)

                        if replace_mode:
                            # REPLACE MODE: Trim to blend with transition's preserved section
                            # Transition starts with preserved frames from Clip1 context
                            # Keep Clip1 up to (but not including) where transition's preserved section starts, plus blend overlap
                            # This ensures we blend Clip1[n] with Transition[0] which contains Clip1[n]' (VACE processed)
                            frames_to_remove_clip1 = context_frame_count - blend_frames
                            frames_to_keep_clip1 = start_frame_count - frames_to_remove_clip1
                            dprint(f"[JOIN_CLIPS] Task {task_id}: REPLACE mode trimming clip1 - removing last {frames_to_remove_clip1} frames, keeping {frames_to_keep_clip1}/{start_frame_count}")
                            dprint(f"[JOIN_CLIPS] Task {task_id}:   This keeps blend overlap with transition's preserved section")
                        else:
                            # INSERT MODE (original): keep all frames except last (context_frame_count - blend_frames)
                            # This leaves blend_frames overlap for crossfade with transition
                            frames_to_keep_clip1 = start_frame_count - (context_frame_count - blend_frames)
                            dprint(f"[JOIN_CLIPS] Task {task_id}: INSERT mode trimming clip1 - keeping {frames_to_keep_clip1}/{start_frame_count} frames")

                        # Use select filter for frame-accurate trimming: frames 0 to (frames_to_keep_clip1 - 1)
                        # setpts resets timestamps for smooth concatenation
                        trim_clip1_cmd = [
                            'ffmpeg', '-y', '-i', str(starting_video),
                            '-vf', f'select=between(n\\,0\\,{frames_to_keep_clip1 - 1}),setpts=N/FR/TB',
                            '-r', str(start_fps),
                            str(clip1_trimmed_path)
                        ]

                        result = subprocess.run(trim_clip1_cmd, capture_output=True, text=True, timeout=60)
                        if result.returncode != 0:
                            raise ValueError(f"Failed to trim clip1: {result.stderr}")

                        if replace_mode:
                            # REPLACE MODE: Calculate clip2 skip accounting for VACE quantization shift
                            frames_to_skip_clip2 = _calculate_replace_mode_clip2_skip(
                                context_frame_count=context_frame_count,
                                gap_for_guide=gap_for_guide,
                                blend_frames=blend_frames,
                                quantization_shift=quantization_shift
                            )
                            frames_remaining_clip2 = end_frame_count - frames_to_skip_clip2

                            dprint(f"[JOIN_CLIPS] Task {task_id}: REPLACE mode trimming clip2")
                            dprint(f"[JOIN_CLIPS] Task {task_id}:   Skipping first {frames_to_skip_clip2}/{end_frame_count} frames")
                            dprint(f"[JOIN_CLIPS] Task {task_id}:   Keeping {frames_remaining_clip2} frames")
                            dprint(f"[JOIN_CLIPS] Task {task_id}:   Quantization shift applied: -{quantization_shift} frames")
                            dprint(f"[JOIN_CLIPS] Task {task_id}:   Crossfade: {blend_frames} frames starting at clip2[{frames_to_skip_clip2}]")
                        else:
                            # INSERT MODE: Standard calculation
                            frames_to_skip_clip2 = context_frame_count - blend_frames
                            frames_remaining_clip2 = end_frame_count - frames_to_skip_clip2
                            dprint(f"[JOIN_CLIPS] Task {task_id}: INSERT mode trimming clip2 - skipping first {frames_to_skip_clip2}/{end_frame_count} frames, keeping {frames_remaining_clip2}")

                        # Select frames from context_frame_count onwards and reset PTS for smooth concatenation
                        trim_clip2_cmd = [
                            'ffmpeg', '-y', '-i', str(ending_video),
                            '-vf', f'select=gte(n\\,{frames_to_skip_clip2}),setpts=N/FR/TB',
                            '-r', str(end_fps),
                            str(clip2_trimmed_path)
                        ]

                        result = subprocess.run(trim_clip2_cmd, capture_output=True, text=True, timeout=60)
                        if result.returncode != 0:
                            raise ValueError(f"Failed to trim clip2: {result.stderr}")

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
                            dprint(f"[JOIN_CLIPS] Task {task_id}: Successfully stitched videos with crossfade blending")
                        except Exception as e:
                            raise ValueError(f"Failed to stitch videos with crossfade: {e}") from e

                        # Verify final output exists
                        if not final_output_path.exists() or final_output_path.stat().st_size == 0:
                            raise ValueError(f"Final concatenated video is missing or empty")

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


def _handle_join_clips_child_task(
    task_params_from_db: dict,
    main_output_dir_base: Path,
    task_id: str,
    task_queue = None,
    *, dprint
) -> Tuple[bool, str]:
    """
    Handle join_clips_child task - part of join_clips_orchestrator workflow.

    This is a wrapper around _handle_join_clips_task() that adds orchestrator-specific logic:
    - Fetches starting_video_path from predecessor task if not provided
    - Uses custom output directory structure for orchestrator runs

    Args:
        task_params_from_db: Task parameters (same as join_clips, but starting_video_path may be None)
        main_output_dir_base: Base output directory
        task_id: Task ID for logging
        task_queue: HeadlessTaskQueue instance
        dprint: Print function for logging

    Returns:
        Tuple of (success: bool, output_path_or_message: str)
    """
    dprint(f"[JOIN_CLIPS_CHILD] Task {task_id}: Starting child task handler")

    try:
        # Extract orchestrator context
        orchestrator_task_id_ref = task_params_from_db.get("orchestrator_task_id_ref")
        orchestrator_run_id = task_params_from_db.get("orchestrator_run_id")
        join_index = task_params_from_db.get("join_index", 0)
        is_first_join = task_params_from_db.get("is_first_join", False)

        dprint(f"[JOIN_CLIPS_CHILD] Task {task_id}: Orchestrator={orchestrator_task_id_ref}, Run={orchestrator_run_id}, Index={join_index}")

        # Check if starting_video_path needs to be fetched from predecessor
        starting_video_path = task_params_from_db.get("starting_video_path")

        if starting_video_path is None:
            if is_first_join:
                error_msg = "First join must have starting_video_path explicitly set"
                dprint(f"[JOIN_CLIPS_CHILD_ERROR] Task {task_id}: {error_msg}")
                return False, error_msg

            dprint(f"[JOIN_CLIPS_CHILD] Task {task_id}: starting_video_path is None, fetching from predecessor")

            # Fetch predecessor output
            predecessor_id, predecessor_output = db_ops.get_predecessor_output_via_edge_function(task_id)

            if not predecessor_output:
                error_msg = f"Failed to fetch predecessor output (predecessor_id={predecessor_id})"
                dprint(f"[JOIN_CLIPS_CHILD_ERROR] Task {task_id}: {error_msg}")
                return False, error_msg

            starting_video_path = predecessor_output
            dprint(f"[JOIN_CLIPS_CHILD] Task {task_id}: Fetched predecessor output: {predecessor_output}")

        # Update task params with resolved starting_video_path
        task_params_with_resolved_path = task_params_from_db.copy()
        task_params_with_resolved_path["starting_video_path"] = starting_video_path

        # Use custom output directory if specified by orchestrator
        if "join_output_dir" in task_params_from_db:
            custom_output_dir = Path(task_params_from_db["join_output_dir"])
            custom_output_dir.mkdir(parents=True, exist_ok=True)
            dprint(f"[JOIN_CLIPS_CHILD] Task {task_id}: Using custom output dir: {custom_output_dir}")
        else:
            custom_output_dir = main_output_dir_base

        # Delegate to existing join_clips handler
        dprint(f"[JOIN_CLIPS_CHILD] Task {task_id}: Delegating to _handle_join_clips_task()")
        success, result = _handle_join_clips_task(
            task_params_from_db=task_params_with_resolved_path,
            main_output_dir_base=custom_output_dir,
            task_id=task_id,
            task_queue=task_queue,
            dprint=dprint
        )

        if success:
            dprint(f"[JOIN_CLIPS_CHILD] Task {task_id}: Successfully completed join {join_index}")
        else:
            dprint(f"[JOIN_CLIPS_CHILD] Task {task_id}: Join {join_index} failed: {result}")

        return success, result

    except Exception as e:
        error_msg = f"Unexpected error in join_clips_child handler: {e}"
        dprint(f"[JOIN_CLIPS_CHILD_ERROR] Task {task_id}: {error_msg}")
        import traceback
        traceback.print_exc()
        return False, error_msg
