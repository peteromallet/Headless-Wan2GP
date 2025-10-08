"""
Join Clips - Bridge two video clips using VACE generation

This module provides functionality to smoothly join two video clips by:
1. Extracting context frames from the end of the first clip
2. Extracting context frames from the beginning of the second clip
3. Generating transition frames between them using VACE
4. Using mask video to preserve the context frames and only generate the gap

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
    get_video_frame_count_and_fps
)
from ..video_utils import (
    extract_frames_from_video as sm_extract_frames_from_video,
)
from ..vace_frame_utils import (
    create_guide_and_mask_for_generation,
    prepare_vace_generation_params
)
from .. import db_operations as db_ops


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
            - gap_frame_count: Number of frames to generate between clips
            - prompt: Generation prompt for the transition
            - model: Optional model override (defaults to lightning_baseline_2_2_2)
            - resolution: Optional [width, height] override
            - fps: Optional FPS override (defaults to 16)
            - Other standard VACE parameters
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
        prompt = task_params_from_db.get("prompt", "")

        # Validate required parameters
        if not starting_video_path:
            error_msg = "starting_video_path is required"
            dprint(f"[JOIN_CLIPS_ERROR] Task {task_id}: {error_msg}")
            return False, error_msg

        if not ending_video_path:
            error_msg = "ending_video_path is required"
            dprint(f"[JOIN_CLIPS_ERROR] Task {task_id}: {error_msg}")
            return False, error_msg

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

        # Validate task queue
        if task_queue is None:
            error_msg = "task_queue is required for join_clips"
            dprint(f"[JOIN_CLIPS_ERROR] Task {task_id}: {error_msg}")
            return False, error_msg

        dprint(f"[JOIN_CLIPS] Task {task_id}: Parameters validated")
        dprint(f"[JOIN_CLIPS]   Starting video: {starting_video}")
        dprint(f"[JOIN_CLIPS]   Ending video: {ending_video}")
        dprint(f"[JOIN_CLIPS]   Context frames: {context_frame_count}")
        dprint(f"[JOIN_CLIPS]   Gap frames: {gap_frame_count}")

        # --- 2. Extract Video Properties ---
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

        # --- 3. Extract Context Frames ---
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

            # Extract all frames from ending video, take first N
            end_all_frames = sm_extract_frames_from_video(str(ending_video), dprint_func=dprint)
            if not end_all_frames or len(end_all_frames) < context_frame_count:
                error_msg = f"Failed to extract {context_frame_count} frames from ending video"
                dprint(f"[JOIN_CLIPS_ERROR] Task {task_id}: {error_msg}")
                return False, error_msg

            end_context_frames = end_all_frames[:context_frame_count]
            dprint(f"[JOIN_CLIPS] Task {task_id}: Extracted {len(end_context_frames)} frames from beginning of ending video")

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

        # --- 4. Build Guide and Mask Videos (using shared helper) ---
        # Create working directory
        join_clips_dir = main_output_dir_base / "join_clips" / task_id
        join_clips_dir.mkdir(parents=True, exist_ok=True)

        try:
            created_guide_video, created_mask_video = create_guide_and_mask_for_generation(
                context_frames_before=start_context_frames,
                context_frames_after=end_context_frames,
                gap_frame_count=gap_frame_count,
                resolution_wh=parsed_res_wh,
                fps=target_fps,
                output_dir=join_clips_dir,
                task_id=task_id,
                filename_prefix="join",
                dprint=dprint
            )
        except Exception as e:
            error_msg = f"Failed to create guide/mask videos: {e}"
            dprint(f"[JOIN_CLIPS_ERROR] Task {task_id}: {error_msg}")
            import traceback
            traceback.print_exc()
            return False, error_msg

        total_frames = context_frame_count * 2 + gap_frame_count

        # --- 5. Prepare Generation Parameters (using shared helper) ---
        dprint(f"[JOIN_CLIPS] Task {task_id}: Preparing generation parameters...")

        # Determine model (default to Lightning baseline for fast generation)
        model = task_params_from_db.get("model", "lightning_baseline_2_2_2")

        # Ensure prompt is valid
        prompt = ensure_valid_prompt(prompt)
        negative_prompt = ensure_valid_negative_prompt(
            task_params_from_db.get("negative_prompt", "")
        )

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
            task_params=task_params_from_db  # Pass through for optional param merging
        )

        dprint(f"[JOIN_CLIPS] Task {task_id}: Generation parameters prepared")
        dprint(f"[JOIN_CLIPS]   Model: {model}")
        dprint(f"[JOIN_CLIPS]   Video length: {total_frames} frames")
        dprint(f"[JOIN_CLIPS]   Resolution: {parsed_res_wh}")

        # --- 6. Submit to Generation Queue ---
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
            max_wait_time = 600  # 10 minute timeout
            wait_interval = 2  # Check every 2 seconds
            elapsed_time = 0

            while elapsed_time < max_wait_time:
                status = task_queue.get_task_status(task_id)

                if status is None:
                    error_msg = "Task status became None during processing"
                    dprint(f"[JOIN_CLIPS_ERROR] Task {task_id}: {error_msg}")
                    return False, error_msg

                if status.status == "completed":
                    output_path = status.result_path
                    processing_time = status.processing_time or 0
                    dprint(f"[JOIN_CLIPS] Task {task_id}: Generation completed successfully in {processing_time:.1f}s")
                    dprint(f"[JOIN_CLIPS] Task {task_id}: Output: {output_path}")
                    return True, output_path

                elif status.status == "failed":
                    error_msg = status.error_message or "Generation failed without specific error message"
                    dprint(f"[JOIN_CLIPS_ERROR] Task {task_id}: Generation failed - {error_msg}")
                    return False, error_msg

                else:
                    # Still processing
                    dprint(f"[JOIN_CLIPS] Task {task_id}: Queue status: {status.status}, waiting...")
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
