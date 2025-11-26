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
    num_anchor_frames: int = 3,
    replace_mode: bool = False,
    gap_inserted_frames: dict = None,
    total_frames: int = None,
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
        gap_frame_count: Number of frames to generate in the gap (INSERT mode) or replace (REPLACE mode)
        resolution_wh: (width, height) tuple for video resolution
        fps: Target frames per second
        output_dir: Directory to save guide and mask videos
        task_id: Task ID for logging
        filename_prefix: Prefix for output filenames (default: "vace_gen")
        regenerate_anchors: If True, exclude anchor frames from guide and regenerate them
        num_anchor_frames: Number of anchor frames to regenerate on each side (default: 3)
        replace_mode: If True, gap frames REPLACE boundary frames instead of being inserted (default: False)
        gap_inserted_frames: Optional dict mapping {relative_gap_index: image_array} to insert and preserve in the gap
        dprint: Print function for logging

    Returns:
        Tuple of (guide_video_path, mask_video_path, total_frame_count)

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

    # Calculate total frames accounting for regenerate_anchors and replace_mode
    num_context_before = len(context_frames_before)
    num_context_after = len(context_frames_after)

    # REPLACE MODE: gap frames REPLACE boundary frames instead of being inserted
    # Initialize preservation variables for use throughout function
    num_preserved_before = 0
    num_preserved_after = 0
    actual_gap_count = gap_frame_count
    max_replaceable_before = 0
    max_replaceable_after = 0
    
    if replace_mode:
        # In replace mode, gap_frame_count determines how many boundary frames to regenerate
        # Split the regeneration across the boundary
        frames_to_replace_from_before = gap_frame_count // 2
        frames_to_replace_from_after = gap_frame_count - frames_to_replace_from_before
        
        # Clamp replacement counts to available context to avoid negative preservation windows
        max_replaceable_before = min(frames_to_replace_from_before, num_context_before)
        max_replaceable_after = min(frames_to_replace_from_after, num_context_after)
        num_preserved_before = max(0, num_context_before - max_replaceable_before)
        num_preserved_after = max(0, num_context_after - max_replaceable_after)
        
        # If gap exceeds available context, clamp it to available boundary frames
        if frames_to_replace_from_before > num_context_before or frames_to_replace_from_after > num_context_after:
            actual_gap_count = max_replaceable_before + max_replaceable_after
            dprint(f"[VACE_UTILS]   Warning: Requested gap ({gap_frame_count}) exceeds available boundary frames "
                   f"({num_context_before} before, {num_context_after} after). Clamping to {actual_gap_count} frames.")

        # Total frames: use provided total_frames (for quantization) or calculate from preserved + gap
        if total_frames is None:
            total_frames = num_preserved_before + actual_gap_count + num_preserved_after

        dprint(f"[VACE_UTILS] Task {task_id}: Creating guide and mask videos (REPLACE MODE)")
        dprint(f"[VACE_UTILS]   Context before: {num_context_before} frames")
        dprint(f"[VACE_UTILS]   Context after: {num_context_after} frames")
        dprint(f"[VACE_UTILS]   Regenerating: {actual_gap_count} boundary frames (replacing up to {frames_to_replace_from_before} from before, {frames_to_replace_from_after} from after)")
        dprint(f"[VACE_UTILS]   Total: {total_frames} frames")

        # In replace mode, regenerate_anchors is effectively ignored - gap_frame_count controls regeneration
        num_anchor_frames_before = 0
        num_anchor_frames_after = 0
    else:
        # INSERT MODE (original behavior)
        # If regenerate_anchors, we'll exclude N anchor frames from each side
        # and add them as gray placeholders to be generated
        num_anchor_frames_before = 0
        num_anchor_frames_after = 0
        if regenerate_anchors:
            if num_context_before > 0:
                num_anchor_frames_before = min(num_anchor_frames, num_context_before)
            if num_context_after > 0:
                num_anchor_frames_after = min(num_anchor_frames, num_context_after)

        total_frames = num_context_before + gap_frame_count + num_context_after

        dprint(f"[VACE_UTILS] Task {task_id}: Creating guide and mask videos (INSERT MODE)")
        dprint(f"[VACE_UTILS]   Context before: {num_context_before} frames")
        dprint(f"[VACE_UTILS]   Gap: {gap_frame_count} frames")
        dprint(f"[VACE_UTILS]   Context after: {num_context_after} frames")
        dprint(f"[VACE_UTILS]   Total: {total_frames} frames")
        if regenerate_anchors:
            dprint(f"[VACE_UTILS]   Regenerate anchors: {num_anchor_frames_before} frames at end of before context, {num_anchor_frames_after} frames at start of after context")

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
    
    # Track indices of inserted frames (absolute index in guide_frames)
    inserted_frame_indices = []
    
    # Normalize gap_inserted_frames to empty dict if None
    gap_inserted_frames = gap_inserted_frames or {}

    if replace_mode:
        # REPLACE MODE: Build guide with boundary regeneration
        # (Preservation and gap counts already calculated above)
        
        # Add preserved frames from context_before
        guide_frames.extend(context_frames_before[:num_preserved_before])
        dprint(f"[VACE_UTILS]   Added {num_preserved_before} preserved frames from before context")

        # Add gray placeholders for regenerated boundary frames (or inserted frames)
        # Use actual_gap_count instead of gap_frame_count
        for i in range(actual_gap_count):
            if i in gap_inserted_frames:
                inserted_frame_indices.append(len(guide_frames))
                guide_frames.append(gap_inserted_frames[i])
            else:
                guide_frames.append(gray_frame.copy())
        
        if inserted_frame_indices:
            dprint(f"[VACE_UTILS]   Inserted {len(inserted_frame_indices)} frames into gap at relative indices: {list(gap_inserted_frames.keys())}")
        dprint(f"[VACE_UTILS]   Added {actual_gap_count} frames for boundary regeneration")

        # Add preserved frames from context_after (use max_replaceable_after for correct slicing)
        guide_frames.extend(context_frames_after[max_replaceable_after:])
        dprint(f"[VACE_UTILS]   Added {num_preserved_after} preserved frames from after context")

    else:
        # INSERT MODE (original behavior)
        # Add context frames before gap
        if regenerate_anchors and num_anchor_frames_before > 0:
            # Exclude the last N anchor frames and add them as gray placeholders later
            num_preserved_before = num_context_before - num_anchor_frames_before
            guide_frames.extend(context_frames_before[:num_preserved_before])
            dprint(f"[VACE_UTILS]   Added {num_preserved_before} context frames (before, excluding {num_anchor_frames_before} anchors)")
        else:
            guide_frames.extend(context_frames_before)
            dprint(f"[VACE_UTILS]   Added {len(context_frames_before)} context frames (before)")

        # Add gray placeholders for regenerated anchors (last N frames of before context)
        if regenerate_anchors and num_anchor_frames_before > 0:
            for _ in range(num_anchor_frames_before):
                guide_frames.append(gray_frame.copy())
            dprint(f"[VACE_UTILS]   Added {num_anchor_frames_before} gray placeholders for regenerated anchors (end of before context)")

        # Add gray placeholder frames for the gap (or inserted frames)
        for i in range(gap_frame_count):
            if i in gap_inserted_frames:
                inserted_frame_indices.append(len(guide_frames))
                guide_frames.append(gap_inserted_frames[i])
            else:
                guide_frames.append(gray_frame.copy())
        
        if inserted_frame_indices:
            dprint(f"[VACE_UTILS]   Inserted {len(inserted_frame_indices)} frames into gap at relative indices: {list(gap_inserted_frames.keys())}")
        dprint(f"[VACE_UTILS]   Added {gap_frame_count} frames for gap")

        # Add gray placeholders for regenerated anchors (first N frames of after context)
        if regenerate_anchors and num_anchor_frames_after > 0:
            for _ in range(num_anchor_frames_after):
                guide_frames.append(gray_frame.copy())
            dprint(f"[VACE_UTILS]   Added {num_anchor_frames_after} gray placeholders for regenerated anchors (start of after context)")

        # Add context frames after gap
        if regenerate_anchors and num_anchor_frames_after > 0:
            # Exclude the first N anchor frames since we added them as gray placeholders above
            guide_frames.extend(context_frames_after[num_anchor_frames_after:])
            dprint(f"[VACE_UTILS]   Added {len(context_frames_after) - num_anchor_frames_after} context frames (after, excluding {num_anchor_frames_after} anchors)")
        else:
            guide_frames.extend(context_frames_after)
            dprint(f"[VACE_UTILS]   Added {len(context_frames_after)} context frames (after)")

    # Determine final total frame count before writing videos/masks
    guide_frame_count = len(guide_frames)
    if guide_frame_count <= 0:
        raise ValueError("Guide video cannot be empty")

    if total_frames is None:
        total_frames = guide_frame_count
    elif total_frames != guide_frame_count:
        dprint(f"[VACE_UTILS] Task {task_id}: total_frames override ({total_frames}) "
               f"does not match constructed guide ({guide_frame_count}). Using guide frame count.")
        total_frames = guide_frame_count

    dprint(f"[VACE_UTILS]   Final guide frame count: {guide_frame_count}")

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

    if replace_mode:
        # REPLACE MODE: Mark preserved frames as inactive (black), boundary frames as active (white)
        # Calculate actual gap count used (may be clamped from requested gap_frame_count)
        actual_gap_in_mask = total_frames - (num_preserved_before + num_preserved_after)
        
        # BLACK: first num_preserved_before frames
        for i in range(num_preserved_before):
            inactive_indices.add(i)
        dprint(f"[VACE_UTILS]   Marked {num_preserved_before} frames as inactive (preserved from before context)")

        # WHITE: next actual_gap_in_mask frames (the boundary region to regenerate)
        # These are automatically active (not added to inactive_indices)

        # BLACK: last num_preserved_after frames
        start_of_preserved_after = num_preserved_before + actual_gap_in_mask
        for i in range(start_of_preserved_after, total_frames):
            inactive_indices.add(i)
        dprint(f"[VACE_UTILS]   Marked {num_preserved_after} frames as inactive (preserved from after context)")

    else:
        # INSERT MODE (original behavior)
        # Mark context frames before gap as inactive
        if regenerate_anchors and num_anchor_frames_before > 0:
            # Exclude the last N anchor frames - they should be white/generate
            num_inactive_before = num_context_before - num_anchor_frames_before
            for i in range(num_inactive_before):
                inactive_indices.add(i)
            dprint(f"[VACE_UTILS]   Marked {num_inactive_before} frames as inactive (before context, excluding {num_anchor_frames_before} anchors)")
        else:
            for i in range(num_context_before):
                inactive_indices.add(i)
            dprint(f"[VACE_UTILS]   Marked {num_context_before} frames as inactive (before context)")

        # Mark context frames after gap as inactive
        start_of_after_context = num_context_before + gap_frame_count
        if regenerate_anchors and num_anchor_frames_after > 0:
            # Exclude the first N anchor frames - they should be white/generate
            # The anchors start at index start_of_after_context, so start marking inactive from start_of_after_context + num_anchor_frames_after
            for i in range(start_of_after_context + num_anchor_frames_after, total_frames):
                inactive_indices.add(i)
            num_inactive_after = total_frames - (start_of_after_context + num_anchor_frames_after)
            dprint(f"[VACE_UTILS]   Marked {num_inactive_after} frames as inactive (after context, excluding {num_anchor_frames_after} anchors)")
        else:
            for i in range(start_of_after_context, total_frames):
                inactive_indices.add(i)
            dprint(f"[VACE_UTILS]   Marked {total_frames - start_of_after_context} frames as inactive (after context)")

    # Active frames are everything not in inactive_indices
    active_indices = [i for i in range(total_frames) if i not in inactive_indices]

    # If frames were inserted into the gap, ensure they are inactive (BLACK/KEEP)
    if inserted_frame_indices:
        for idx in inserted_frame_indices:
            inactive_indices.add(idx)
            if idx in active_indices:
                active_indices.remove(idx)
        dprint(f"[VACE_UTILS]   Marked {len(inserted_frame_indices)} inserted frames as inactive (black/keep) at indices: {inserted_frame_indices}")

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

    return created_guide_video, created_mask_video, total_frames


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
            "sample_solver", "additional_loras", "phase_config"
        ]

        for param in optional_params:
            if param in task_params:
                generation_params[param] = task_params[param]

    return generation_params
