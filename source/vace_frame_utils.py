"""
VACE Frame Utilities - Shared logic for frame-based VACE generation tasks

This module provides shared functionality for tasks that use VACE to generate
video frames with guide and mask videos. Used by:
- join_clips: Bridge two video clips with smooth transition
- inpaint_frames: Regenerate a range of frames within a single video

Key Features:
- Guide video creation with context frames + gray gap
- Mask video creation (black=preserve, white=generate)
- Consistent VACE parameter handling
"""

import uuid
from pathlib import Path
from typing import List, Tuple
from datetime import datetime
import numpy as np

from .common_utils import (
    create_mask_video_from_inactive_indices,
    create_color_frame as sm_create_color_frame
)
from .video_utils import (
    create_video_from_frames_list as sm_create_video_from_frames_list
)


def create_guide_and_mask_for_generation(
    context_frames_before: List[np.ndarray],
    context_frames_after: List[np.ndarray],
    gap_frame_count: int,
    resolution_wh: Tuple[int, int],
    fps: int,
    output_dir: Path,
    task_id: str,
    filename_prefix: str = "vace_gen",
    regenerate_anchors: bool = False,
    *,
    dprint=print
) -> Tuple[Path, Path]:
    """
    Create guide and mask videos for VACE generation.

    This shared function is used by both join_clips and inpaint_frames to create
    the guide video (context + gray gap + context) and mask video (black=keep, white=generate).

    Args:
        context_frames_before: Frames to preserve before the gap (numpy arrays)
        context_frames_after: Frames to preserve after the gap (numpy arrays)
        gap_frame_count: Number of frames to generate in the gap
        resolution_wh: (width, height) tuple for video resolution
        fps: Target frames per second
        output_dir: Directory to save guide and mask videos
        task_id: Task ID for logging
        filename_prefix: Prefix for output filenames (default: "vace_gen")
        regenerate_anchors: If True, exclude anchor frames from guide and regenerate them
        dprint: Print function for logging

    Returns:
        Tuple of (guide_video_path, mask_video_path)

    Raises:
        ValueError: If context frames are empty or resolution is invalid
        RuntimeError: If video creation fails
    """
    # Validate inputs
    if not context_frames_before and not context_frames_after:
        raise ValueError("At least one context frame set must be provided")

    if gap_frame_count < 0:
        raise ValueError(f"gap_frame_count must be non-negative, got {gap_frame_count}")

    if resolution_wh[0] <= 0 or resolution_wh[1] <= 0:
        raise ValueError(f"Invalid resolution: {resolution_wh}")

    # Calculate total frames accounting for regenerate_anchors
    num_context_before = len(context_frames_before)
    num_context_after = len(context_frames_after)

    # If regenerate_anchors, we'll exclude the last frame from before and first frame from after
    # and add them as gray placeholders to be generated
    num_anchors_to_regenerate = 0
    if regenerate_anchors:
        if num_context_before > 0:
            num_anchors_to_regenerate += 1
        if num_context_after > 0:
            num_anchors_to_regenerate += 1

    total_frames = num_context_before + gap_frame_count + num_context_after

    dprint(f"[VACE_UTILS] Task {task_id}: Creating guide and mask videos")
    dprint(f"[VACE_UTILS]   Context before: {num_context_before} frames")
    dprint(f"[VACE_UTILS]   Gap: {gap_frame_count} frames")
    dprint(f"[VACE_UTILS]   Context after: {num_context_after} frames")
    dprint(f"[VACE_UTILS]   Total: {total_frames} frames")
    if regenerate_anchors:
        dprint(f"[VACE_UTILS]   Regenerate anchors: {num_anchors_to_regenerate} anchor frames will be generated")

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate unique filenames
    timestamp_short = datetime.now().strftime("%H%M%S")
    unique_suffix = uuid.uuid4().hex[:6]
    guide_filename = f"{filename_prefix}_guide_{timestamp_short}_{unique_suffix}.mp4"
    mask_filename = f"{filename_prefix}_mask_{timestamp_short}_{unique_suffix}.mp4"

    guide_video_path = output_dir / guide_filename
    mask_video_path = output_dir / mask_filename

    # --- 1. Build Guide Video ---
    dprint(f"[VACE_UTILS] Task {task_id}: Building guide video...")

    guide_frames = []
    gray_frame = sm_create_color_frame(resolution_wh, (128, 128, 128))

    # Add context frames before gap
    if regenerate_anchors and num_context_before > 0:
        # Exclude the last frame (anchor) and add it as gray placeholder later
        guide_frames.extend(context_frames_before[:-1])
        dprint(f"[VACE_UTILS]   Added {len(context_frames_before) - 1} context frames (before, excluding anchor)")
    else:
        guide_frames.extend(context_frames_before)
        dprint(f"[VACE_UTILS]   Added {len(context_frames_before)} context frames (before)")

    # Add gray placeholder for regenerated anchor (last frame of before context)
    if regenerate_anchors and num_context_before > 0:
        guide_frames.append(gray_frame.copy())
        dprint(f"[VACE_UTILS]   Added 1 gray placeholder for regenerated anchor (end of before context)")

    # Add gray placeholder frames for the gap
    for _ in range(gap_frame_count):
        guide_frames.append(gray_frame.copy())
    dprint(f"[VACE_UTILS]   Added {gap_frame_count} gray placeholder frames (gap)")

    # Add gray placeholder for regenerated anchor (first frame of after context)
    if regenerate_anchors and num_context_after > 0:
        guide_frames.append(gray_frame.copy())
        dprint(f"[VACE_UTILS]   Added 1 gray placeholder for regenerated anchor (start of after context)")

    # Add context frames after gap
    if regenerate_anchors and num_context_after > 0:
        # Exclude the first frame (anchor) since we added it as gray placeholder above
        guide_frames.extend(context_frames_after[1:])
        dprint(f"[VACE_UTILS]   Added {len(context_frames_after) - 1} context frames (after, excluding anchor)")
    else:
        guide_frames.extend(context_frames_after)
        dprint(f"[VACE_UTILS]   Added {len(context_frames_after)} context frames (after)")

    # Create guide video
    try:
        created_guide_video = sm_create_video_from_frames_list(
            guide_frames,
            guide_video_path,
            fps,
            resolution_wh
        )

        if not created_guide_video or not created_guide_video.exists():
            raise RuntimeError("Guide video creation returned None or file doesn't exist")

        dprint(f"[VACE_UTILS] Task {task_id}: Guide video created: {created_guide_video}")

    except Exception as e:
        raise RuntimeError(f"Failed to create guide video: {e}") from e

    # --- 2. Build Mask Video ---
    dprint(f"[VACE_UTILS] Task {task_id}: Building mask video...")

    # Determine inactive (black) frame indices
    # Inactive = context frames we want to preserve
    # Active (white) = gap frames + anchor frames (if regenerate_anchors) we want to generate
    inactive_indices = set()

    # Mark context frames before gap as inactive (0 to num_context_before-1)
    if regenerate_anchors and num_context_before > 0:
        # Exclude the last frame (anchor) - it should be white/generate
        for i in range(num_context_before - 1):
            inactive_indices.add(i)
        dprint(f"[VACE_UTILS]   Marked {num_context_before - 1} frames as inactive (before context, excluding anchor)")
    else:
        for i in range(num_context_before):
            inactive_indices.add(i)
        dprint(f"[VACE_UTILS]   Marked {num_context_before} frames as inactive (before context)")

    # Mark context frames after gap as inactive
    start_of_after_context = num_context_before + gap_frame_count
    if regenerate_anchors and num_context_after > 0:
        # Exclude the first frame (anchor) - it should be white/generate
        # The anchor is at index start_of_after_context, so start from start_of_after_context + 1
        for i in range(start_of_after_context + 1, total_frames):
            inactive_indices.add(i)
        dprint(f"[VACE_UTILS]   Marked {total_frames - start_of_after_context - 1} frames as inactive (after context, excluding anchor)")
    else:
        for i in range(start_of_after_context, total_frames):
            inactive_indices.add(i)
        dprint(f"[VACE_UTILS]   Marked {total_frames - start_of_after_context} frames as inactive (after context)")

    # Active frames are everything not in inactive_indices
    active_indices = [i for i in range(total_frames) if i not in inactive_indices]

    dprint(f"[VACE_UTILS]   Inactive frame indices (black/keep): {sorted(inactive_indices)}")
    dprint(f"[VACE_UTILS]   Active frame indices (white/generate): {active_indices}")

    # Create mask video
    try:
        created_mask_video = create_mask_video_from_inactive_indices(
            total_frames=total_frames,
            resolution_wh=resolution_wh,
            inactive_frame_indices=inactive_indices,
            output_path=mask_video_path,
            fps=fps,
            task_id_for_logging=task_id,
            dprint=dprint
        )

        if not created_mask_video or not created_mask_video.exists():
            raise RuntimeError("Mask video creation returned None or file doesn't exist")

        dprint(f"[VACE_UTILS] Task {task_id}: Mask video created: {created_mask_video}")

    except Exception as e:
        raise RuntimeError(f"Failed to create mask video: {e}") from e

    return created_guide_video, created_mask_video


def validate_frame_range(
    total_frame_count: int,
    start_frame: int,
    end_frame: int,
    context_frame_count: int,
    task_id: str = "unknown",
    *,
    dprint=print
) -> Tuple[bool, str]:
    """
    Validate that a frame range has sufficient context frames on both sides.

    Used by inpaint_frames to ensure the requested range can be processed.

    Args:
        total_frame_count: Total frames in the source video
        start_frame: Start frame index (inclusive)
        end_frame: End frame index (exclusive)
        context_frame_count: Required context frames on each side
        task_id: Task ID for logging
        dprint: Print function for logging

    Returns:
        Tuple of (is_valid: bool, error_message: str or None)
    """
    dprint(f"[VACE_UTILS] Task {task_id}: Validating frame range")
    dprint(f"[VACE_UTILS]   Total frames: {total_frame_count}")
    dprint(f"[VACE_UTILS]   Range: [{start_frame}, {end_frame})")
    dprint(f"[VACE_UTILS]   Context required: {context_frame_count} frames on each side")

    # Check if range is valid
    if start_frame < 0:
        return False, f"start_frame ({start_frame}) must be non-negative"

    if end_frame > total_frame_count:
        return False, f"end_frame ({end_frame}) exceeds total frame count ({total_frame_count})"

    if start_frame >= end_frame:
        return False, f"start_frame ({start_frame}) must be less than end_frame ({end_frame})"

    # Check if there's enough context before
    if start_frame < context_frame_count:
        return False, f"Need {context_frame_count} context frames before start_frame ({start_frame}), but only {start_frame} available"

    # Check if there's enough context after
    frames_after = total_frame_count - end_frame
    if frames_after < context_frame_count:
        return False, f"Need {context_frame_count} context frames after end_frame ({end_frame}), but only {frames_after} available"

    dprint(f"[VACE_UTILS] Task {task_id}: Frame range validation passed")
    return True, None


def prepare_vace_generation_params(
    guide_video_path: Path,
    mask_video_path: Path,
    total_frames: int,
    resolution_wh: Tuple[int, int],
    prompt: str,
    negative_prompt: str,
    model: str = "lightning_baseline_2_2_2",
    seed: int = -1,
    task_params: dict = None
) -> dict:
    """
    Prepare standardized VACE generation parameters.

    Creates a consistent parameter dict for VACE generation with guide and mask.

    Args:
        guide_video_path: Path to guide video
        mask_video_path: Path to mask video
        total_frames: Total number of frames to generate
        resolution_wh: (width, height) tuple
        prompt: Generation prompt
        negative_prompt: Negative prompt
        model: Model name (default: lightning_baseline_2_2_2)
        seed: Random seed (-1 for random)
        task_params: Additional task-specific parameters to merge

    Returns:
        Dict of generation parameters ready for GenerationTask
    """
    generation_params = {
        "video_guide": str(guide_video_path.resolve()),
        "video_mask": str(mask_video_path.resolve()),
        "video_prompt_type": "VM",  # Video + Mask for VACE
        "video_length": total_frames,
        "resolution": f"{resolution_wh[0]}x{resolution_wh[1]}",
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": seed,
    }

    # Merge additional task-specific parameters if provided
    if task_params:
        # Add optional parameters only if explicitly provided
        optional_params = [
            "num_inference_steps", "guidance_scale", "embedded_guidance_scale",
            "flow_shift", "audio_guidance_scale", "cfg_zero_step",
            "guidance2_scale", "guidance3_scale", "guidance_phases",
            "switch_threshold", "switch_threshold2", "model_switch_phase",
            "sample_solver", "use_causvid_lora", "use_lighti2x_lora",
            "apply_reward_lora", "additional_loras"
        ]

        for param in optional_params:
            if param in task_params:
                generation_params[param] = task_params[param]

    return generation_params
