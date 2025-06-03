"""Steerable Motion modular functions package.

This package aggregates task-specific handlers and common utilities
for `steerable_motion.py`.
"""

# Task Handlers
from .travel_between_images import run_travel_between_images_task
from .different_pose import run_different_pose_task

# Common Utilities
# These are re-exported here for convenience, allowing task modules to import
# them directly from `sm_functions.common_utils` or just `sm_functions` if preferred.
from .common_utils import (
    DEBUG_MODE, # Note: This is a global, set by steerable_motion.py
    DEFAULT_DB_TABLE_NAME,
    STATUS_QUEUED, 
    STATUS_IN_PROGRESS, 
    STATUS_COMPLETE, 
    STATUS_FAILED,
    dprint,
    parse_resolution,
    generate_unique_task_id,
    image_to_frame,
    create_color_frame,
    create_video_from_frames_list,
    add_task_to_db,
    poll_task_status,
    extract_video_segment_ffmpeg,
    stitch_videos_ffmpeg,
    save_frame_from_video,
    body_colors,
    face_color,
    hand_keypoint_color,
    hand_limb_colors,
    body_skeleton,
    face_skeleton,
    hand_skeleton,
    draw_keypoints_and_skeleton,
    gen_skeleton_with_face_hands,
    transform_all_keypoints,
    extract_pose_keypoints,
    create_pose_interpolated_guide_video,
    get_resized_frame,
    draw_multiline_text,
    generate_debug_summary_video,
    generate_different_pose_debug_video_summary,
    extract_specific_frame_ffmpeg,
    concatenate_videos_ffmpeg,
    get_video_frame_count_and_fps,
    get_image_dimensions_pil
)

# --- Make video_utils directly importable --- 
from .video_utils import (
    ease as crossfade_ease, # Renamed to avoid conflict if other eases are added
    _blend_linear,
    _blend_linear_sharp,
    cross_fade_overlap_frames,
    extract_frames_from_video,
    create_video_from_frames_list,
    _apply_saturation_to_video_ffmpeg,
    color_match_video_to_reference
)

__all__ = [
    # common_utils exports
    "DEBUG_MODE", "dprint", "generate_unique_task_id", "add_task_to_db", "poll_task_status",
    "extract_video_segment_ffmpeg", "stitch_videos_ffmpeg",
    "create_pose_interpolated_guide_video",
    "generate_debug_summary_video",
    "extract_specific_frame_ffmpeg",
    "concatenate_videos_ffmpeg",
    "get_video_frame_count_and_fps",
    "parse_resolution",
    "DEFAULT_DB_TABLE_NAME",
    "get_image_dimensions_pil", 
    "draw_multiline_text",
    "generate_different_pose_debug_video_summary",
    # travel_between_images exports
    "run_travel_between_images_task",
    # different_pose exports
    "run_different_pose_task",
    # video_utils exports
    "crossfade_ease",
    "_blend_linear",
    "_blend_linear_sharp",
    "cross_fade_overlap_frames",
    "extract_frames_from_video",
    "create_video_from_frames_list",
    "_apply_saturation_to_video_ffmpeg",
    "color_match_video_to_reference"
] 