"""
Edit Video Orchestrator - Regenerate selected portions of a video

Takes a source video and portions_to_regenerate, extracts the "keeper" 
segments (portions NOT being regenerated), then uses the join_clips 
infrastructure to regenerate transitions between keeper clips.

Example:
    Source video: 164 frames total (0-163)
    portions_to_regenerate: [
        {start_frame: 16, end_frame: 33},   # 18 frames to regenerate
        {start_frame: 49, end_frame: 66}    # 18 frames to regenerate
    ]

    Resulting keeper clips:
    - Clip 0: frames 0-15   (16 frames)
    - Clip 1: frames 34-48  (15 frames)  
    - Clip 2: frames 67-163 (97 frames)

    Join tasks created:
    - join_0: Clip 0 + Clip 1 → regenerates transition (was frames 16-33)
    - join_1: (Clip 0+1 result) + Clip 2 → regenerates transition (was frames 49-66)

The extracted keeper clips are passed to the shared _create_join_chain_tasks() 
function which creates the same join_clips_segment tasks as join_clips_orchestrator.
"""

import subprocess
import traceback
from pathlib import Path
from typing import Tuple, List, Optional, Dict

from .. import db_operations as db_ops
from ..common_utils import download_video_if_url, get_video_frame_count_and_fps
from ..video_utils import extract_frame_range_to_video, ensure_video_fps, get_video_frame_count_ffprobe

# Import shared functions from join_clips_orchestrator
from .join_clips_orchestrator import (
    _extract_join_settings_from_payload,
    _check_existing_join_tasks,
    _create_join_chain_tasks,
    _extract_boundary_frames_for_vlm,
    _generate_vlm_prompts_for_joins,
)


def _calculate_keeper_segments(
    portions: List[dict],
    total_frames: int,
    dprint,
    replace_mode: bool = False
) -> List[dict]:
    """
    Calculate keeper segments (parts NOT being regenerated).
    
    Args:
        portions: List of portions to regenerate with start_frame, end_frame
        total_frames: Total frames in source video
        dprint: Debug print function
        replace_mode: If True, the start and end frames of each portion become
                      ANCHORS that are preserved, and only content BETWEEN them
                      is regenerated. If False, the entire portion is regenerated.
        
    Returns:
        List of {start_frame, end_frame, frame_count} dicts for keeper segments
    """
    # Sort portions by start_frame
    sorted_portions = sorted(portions, key=lambda p: p["start_frame"])
    
    mode_desc = "REPLACE MODE (anchors preserved)" if replace_mode else "STANDARD MODE (full replacement)"
    dprint(f"[EDIT_VIDEO] === Keeper Segment Calculation ({mode_desc}) ===")
    dprint(f"[EDIT_VIDEO] Source video: {total_frames} total frames")
    dprint(f"[EDIT_VIDEO] Portions to regenerate ({len(sorted_portions)}):")
    for i, p in enumerate(sorted_portions):
        frame_count = p.get("frame_count") or (p["end_frame"] - p["start_frame"] + 1)
        if replace_mode:
            # In replace mode, anchors are kept, only middle is regenerated
            dprint(f"[EDIT_VIDEO]   Portion {i}: frames {p['start_frame']}-{p['end_frame']} "
                   f"(anchors {p['start_frame']} & {p['end_frame']} kept, regenerate {p['start_frame']+1}-{p['end_frame']-1})")
        else:
            dprint(f"[EDIT_VIDEO]   Portion {i}: frames {p['start_frame']}-{p['end_frame']} ({frame_count} frames)")
    
    # Validate portions don't overlap
    for i in range(len(sorted_portions) - 1):
        if sorted_portions[i]["end_frame"] >= sorted_portions[i + 1]["start_frame"]:
            raise ValueError(
                f"Portions overlap: portion {i} ends at {sorted_portions[i]['end_frame']}, "
                f"portion {i+1} starts at {sorted_portions[i + 1]['start_frame']}"
            )
    
    keepers = []
    
    # In replace_mode:
    #   - start_frame becomes the END of the previous keeper (anchor preserved)
    #   - end_frame becomes the START of the next keeper (anchor preserved)
    #   - Only frames BETWEEN anchors are regenerated
    #
    # Example: portion start_frame=417, end_frame=427 in replace_mode:
    #   - Keeper 0: frames 0-417 (includes anchor frame 417)
    #   - Regenerate: frames 418-426 (between anchors)
    #   - Keeper 1: frames 427-end (includes anchor frame 427)
    
    # Keeper before first portion
    first_start = sorted_portions[0]["start_frame"]
    if replace_mode:
        # In replace mode, include start_frame as anchor in keeper
        keeper_end = first_start  # Include the anchor
    else:
        # Standard mode: keeper ends before portion starts
        keeper_end = first_start - 1
    
    if keeper_end >= 0:
        keeper = {
            "start_frame": 0,
            "end_frame": keeper_end,
            "frame_count": keeper_end + 1
        }
        keepers.append(keeper)
        dprint(f"[EDIT_VIDEO] Keeper 0 (before first portion): frames 0-{keeper['end_frame']} ({keeper['frame_count']} frames)")
    
    # Keepers between portions
    for i in range(len(sorted_portions) - 1):
        if replace_mode:
            # Include end_frame of portion i as start (anchor)
            # Include start_frame of portion i+1 as end (anchor)
            gap_start = sorted_portions[i]["end_frame"]  # Anchor from portion i
            gap_end = sorted_portions[i + 1]["start_frame"]  # Anchor from portion i+1
        else:
            gap_start = sorted_portions[i]["end_frame"] + 1
            gap_end = sorted_portions[i + 1]["start_frame"] - 1
        
        if gap_end >= gap_start:
            keeper = {
                "start_frame": gap_start,
                "end_frame": gap_end,
                "frame_count": gap_end - gap_start + 1
            }
            keepers.append(keeper)
            dprint(f"[EDIT_VIDEO] Keeper {len(keepers)-1} (between portions): frames {gap_start}-{gap_end} ({keeper['frame_count']} frames)")
        else:
            # Adjacent portions with no gap - insert empty keeper marker
            dprint(f"[EDIT_VIDEO] No keeper between portions {i} and {i+1} (adjacent)")
    
    # Keeper after last portion
    last_end = sorted_portions[-1]["end_frame"]
    if replace_mode:
        # In replace mode, include end_frame as anchor in keeper
        keeper_start = last_end  # Include the anchor
    else:
        keeper_start = last_end + 1
    
    if keeper_start < total_frames:
        keeper = {
            "start_frame": keeper_start,
            "end_frame": total_frames - 1,
            "frame_count": total_frames - keeper_start
        }
        keepers.append(keeper)
        dprint(f"[EDIT_VIDEO] Keeper {len(keepers)-1} (after last portion): frames {keeper['start_frame']}-{keeper['end_frame']} ({keeper['frame_count']} frames)")
    
    dprint(f"[EDIT_VIDEO] Total keeper segments: {len(keepers)}")
    
    return keepers


def _extract_clip_frames(
    source_video: Path,
    output_path: Path,
    start_frame: int,
    end_frame: int,
    fps: float,
    dprint
) -> Path | None:
    """
    Extract a frame range from video using frame-accurate FFmpeg selection.
    
    Wrapper around extract_frame_range_to_video() from video_utils.
    
    Args:
        source_video: Path to source video
        output_path: Path for output clip
        start_frame: First frame to include (0-indexed, inclusive)
        end_frame: Last frame to include (0-indexed, inclusive)
        fps: Output framerate
        dprint: Debug print function
        
    Returns:
        Path to extracted clip, or None on failure
    """
    return extract_frame_range_to_video(
        source_video=source_video,
        output_path=output_path,
        start_frame=start_frame,
        end_frame=end_frame,
        fps=fps,
        dprint_func=dprint
    )


def _get_clip_url_or_path(
    local_path: Path,
    project_id: str | None,
    task_id: str,
    clip_name: str,
    dprint
) -> str:
    """
    Get the URL or path for an extracted clip.
    
    For now, returns the local file path. The join_clips_segment tasks
    will handle downloading if the path is a URL, and the final output
    will be uploaded by the edge function when the task completes.
    
    Note: If you need to upload intermediate clips to Supabase storage
    for distributed workers, you can extend this function to use the
    Supabase storage upload utilities.
    
    Returns:
        Local file path as string (or URL if upload is implemented)
    """
    # For local/distributed processing, the local path works since workers
    # share filesystem or can download URLs. The join_clips_segment handler
    # already supports both paths and URLs via download_video_if_url().
    
    # TODO: For fully distributed deployments without shared storage,
    # implement upload here and return the storage URL.
    
    dprint(f"[EDIT_VIDEO] Keeper clip {clip_name} at: {local_path}")
    return str(local_path)


def _preprocess_portions_to_regenerate(
    source_video_url: str,
    source_video_fps: float,
    source_video_total_frames: int,
    portions_to_regenerate: List[dict],
    per_join_settings: List[dict],
    work_dir: Path,
    orchestrator_task_id: str,
    orchestrator_project_id: str | None,
    dprint,
    replace_mode: bool = False
) -> Dict:
    """
    Pre-process portions_to_regenerate into keeper clips for join orchestration.

    Args:
        source_video_url: URL or path to source video
        source_video_fps: FPS of source video
        source_video_total_frames: Total frame count of source video
        portions_to_regenerate: List of portions to regenerate
        per_join_settings: Per-join settings from orchestrator payload
        work_dir: Working directory for extracted clips
        orchestrator_task_id: Task ID for logging
        orchestrator_project_id: Project ID for Supabase uploads
        dprint: Debug print function
        replace_mode: If True, keep start/end frames as anchors, regenerate between

    Returns:
        {
            "success": bool,
            "error": str | None,
            "clip_list": List[dict],
            "per_join_settings": List[dict]
        }
    """
    dprint(f"[EDIT_VIDEO] Preprocessing {len(portions_to_regenerate)} portions to regenerate")
    
    try:
        # Download source video if it's a URL
        source_video_path = download_video_if_url(
            source_video_url,
            download_target_dir=work_dir,
            task_id_for_logging=orchestrator_task_id,
            descriptive_name="source_video"
        )
        source_video = Path(source_video_path)
        
        if not source_video.exists():
            return {"success": False, "error": f"Source video not found: {source_video_path}"}
        
        # Ensure video is at the expected FPS (frame indices are calculated for this FPS)
        target_fps = source_video_fps or 16
        dprint(f"[EDIT_VIDEO] Expected (from payload): {source_video_total_frames} frames @ {target_fps} fps")
        
        # Use tight FPS tolerance (0.01) to force resampling when there's any meaningful FPS difference
        # This is critical because frame indices are calculated for target_fps - even a 0.3 FPS difference
        # causes ~0.5s drift over 400 frames, making extracted frames appear "too early" or "too late"
        source_video = ensure_video_fps(
            video_path=source_video,
            target_fps=target_fps,
            output_dir=work_dir,
            fps_tolerance=0.01,  # Tight tolerance to ensure frame-accurate extraction
            dprint_func=dprint
        )
        if not source_video:
            return {"success": False, "error": f"Failed to ensure video is at {target_fps} fps"}
        
        # Get actual frame count from the (potentially resampled) video
        # Use ffprobe for accuracy - OpenCV's CAP_PROP_FRAME_COUNT is unreliable
        actual_frames_ffprobe = get_video_frame_count_ffprobe(str(source_video), dprint=dprint)
        actual_frames_opencv, actual_fps = get_video_frame_count_and_fps(str(source_video))
        
        # Prefer ffprobe count, fall back to OpenCV
        actual_frames = actual_frames_ffprobe or actual_frames_opencv
        if not actual_frames:
            return {"success": False, "error": "Could not determine source video frame count"}
        
        # Log discrepancies for debugging
        if actual_frames_ffprobe and actual_frames_opencv and actual_frames_ffprobe != actual_frames_opencv:
            dprint(f"[EDIT_VIDEO] Frame count: ffprobe={actual_frames_ffprobe}, OpenCV={actual_frames_opencv} (using ffprobe)")
        
        # Trust payload's frame count if close to actual - it was calculated for this video
        # Only use actual if significantly different (e.g., after resampling changed frame count)
        if source_video_total_frames:
            diff = abs(source_video_total_frames - actual_frames)
            if diff <= 3:
                # Small difference - trust payload (frontend calculated correctly)
                dprint(f"[EDIT_VIDEO] Using payload frame count: {source_video_total_frames} (actual: {actual_frames}, diff: {diff})")
                actual_frames = source_video_total_frames
            else:
                dprint(f"[EDIT_VIDEO] Note: Payload frame count ({source_video_total_frames}) differs significantly from actual ({actual_frames})")
        
        source_video_total_frames = actual_frames
        source_video_fps = actual_fps or target_fps
        
        dprint(f"[EDIT_VIDEO] Final source video: {source_video_total_frames} frames @ {source_video_fps} fps")
        
        # Calculate keeper segments
        keeper_segments = _calculate_keeper_segments(
            portions=portions_to_regenerate,
            total_frames=source_video_total_frames,
            dprint=dprint,
            replace_mode=replace_mode
        )
        
        if len(keeper_segments) < 2:
            return {"success": False, "error": f"Need at least 2 keeper segments, got {len(keeper_segments)}"}
        
        # Create keeper clips directory
        keepers_dir = work_dir / "keeper_clips"
        keepers_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract each keeper clip
        extracted_clips = []
        for i, keeper in enumerate(keeper_segments):
            clip_name = f"keeper_{i}.mp4"
            local_path = keepers_dir / clip_name
            
            extracted = _extract_clip_frames(
                source_video=source_video,
                output_path=local_path,
                start_frame=keeper["start_frame"],
                end_frame=keeper["end_frame"],
                fps=source_video_fps,
                dprint=dprint
            )
            
            if not extracted:
                return {"success": False, "error": f"Failed to extract keeper clip {i}"}
            
            # Get URL or path for the clip
            clip_url = _get_clip_url_or_path(
                local_path=local_path,
                project_id=orchestrator_project_id,
                task_id=orchestrator_task_id,
                clip_name=clip_name,
                dprint=dprint
            )
            
            extracted_clips.append({
                "url": clip_url,
                "name": f"keeper_{i}",
                "start_frame": keeper["start_frame"],
                "end_frame": keeper["end_frame"],
                "frame_count": keeper["frame_count"]
            })
        
        dprint(f"[EDIT_VIDEO] ✅ Extracted {len(extracted_clips)} keeper clips")
        
        # Build per_join_settings from portions
        # Each join corresponds to a transition between keeper clips
        # 
        # gap_frame_count priority (highest to lowest):
        #   1. portion.gap_frame_count (per-portion override)
        #   2. per_join_settings[i].gap_frame_count (from orchestrator payload)
        #   3. orchestrator_payload.gap_frame_count (default for all joins)
        #
        sorted_portions = sorted(portions_to_regenerate, key=lambda p: p["start_frame"])
        new_per_join_settings = []
        
        for i, portion in enumerate(sorted_portions):
            # Start with existing per_join_settings if provided
            if i < len(per_join_settings):
                join_setting = per_join_settings[i].copy()
            else:
                join_setting = {}
            
            # Per-portion prompt override
            if "prompt" in portion:
                join_setting["prompt"] = portion["prompt"]
            
            # Per-portion gap_frame_count override
            # This allows setting different gaps for each transition
            if "gap_frame_count" in portion:
                join_setting["gap_frame_count"] = portion["gap_frame_count"]
                dprint(f"[EDIT_VIDEO] Join {i}: using per-portion gap_frame_count={portion['gap_frame_count']}")
            
            new_per_join_settings.append(join_setting)
            
            portion_frame_count = portion.get("frame_count") or (portion["end_frame"] - portion["start_frame"] + 1)
            gap_override = join_setting.get("gap_frame_count")
            gap_info = f", gap={gap_override}" if gap_override else ""
            dprint(f"[EDIT_VIDEO] Join {i}: chopped {portion_frame_count} frames (portion {portion['start_frame']}-{portion['end_frame']}){gap_info}")
        
        return {
            "success": True,
            "error": None,
            "clip_list": extracted_clips,
            "per_join_settings": new_per_join_settings
        }
        
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": str(e)}


def _handle_edit_video_orchestrator_task(
    task_params_from_db: dict,
    main_output_dir_base: Path,
    orchestrator_task_id_str: str,
    orchestrator_project_id: str | None,
    *,
    dprint
) -> Tuple[bool, str]:
    """
    Handle edit_video_orchestrator task - regenerate selected portions of a video.
    
    This orchestrator:
    1. Takes a source video and portions_to_regenerate
    2. Extracts "keeper" clips (portions NOT being regenerated)
    3. Creates join_clips_segment tasks to regenerate transitions between keepers
    
    Args:
        task_params_from_db: Task parameters containing orchestrator_details with:
            - source_video_url: URL or path to source video
            - source_video_fps: FPS of source video (default 16)
            - source_video_total_frames: Total frames in source video
            - portions_to_regenerate: List of {start_frame, end_frame, prompt?} dicts
            - All standard join_clips settings (context_frame_count, model, etc.)
        main_output_dir_base: Base output directory
        orchestrator_task_id_str: Orchestrator task ID
        orchestrator_project_id: Project ID for authorization
        dprint: Debug print function

    Returns:
        (success: bool, message: str)
    """
    dprint(f"[EDIT_VIDEO] Starting edit_video_orchestrator task {orchestrator_task_id_str}")
    
    try:
        # === 1. PARSE ORCHESTRATOR PAYLOAD ===
        if 'orchestrator_details' not in task_params_from_db:
            dprint("[EDIT_VIDEO] ERROR: orchestrator_details missing")
            return False, "orchestrator_details missing"
        
        orchestrator_payload = task_params_from_db['orchestrator_details']
        dprint(f"[EDIT_VIDEO] Orchestrator payload keys: {list(orchestrator_payload.keys())}")
        
        # Extract required fields for edit_video
        source_video_url = orchestrator_payload.get("source_video_url")
        portions_to_regenerate = orchestrator_payload.get("portions_to_regenerate", [])
        run_id = orchestrator_payload.get("run_id")
        
        if not source_video_url:
            return False, "source_video_url is required for edit_video_orchestrator"
        
        if not portions_to_regenerate or len(portions_to_regenerate) < 1:
            return False, "portions_to_regenerate must contain at least 1 portion"
        
        if not run_id:
            return False, "run_id is required"
        
        dprint(f"[EDIT_VIDEO] Source: {source_video_url[:80]}...")
        dprint(f"[EDIT_VIDEO] Portions to regenerate: {len(portions_to_regenerate)}")
        
        # Extract join settings (same as join_clips_orchestrator)
        join_settings = _extract_join_settings_from_payload(orchestrator_payload)
        per_join_settings = orchestrator_payload.get("per_join_settings", [])
        output_base_dir = orchestrator_payload.get("output_base_dir", str(main_output_dir_base.resolve()))
        
        # Create run-specific output directory
        current_run_output_dir = Path(output_base_dir) / f"edit_video_run_{run_id}"
        current_run_output_dir.mkdir(parents=True, exist_ok=True)
        dprint(f"[EDIT_VIDEO] Run output directory: {current_run_output_dir}")
        
        # === 2. PREPROCESS: Extract keeper clips ===
        replace_mode = orchestrator_payload.get("replace_mode", False)
        preprocess_result = _preprocess_portions_to_regenerate(
            source_video_url=source_video_url,
            source_video_fps=orchestrator_payload.get("source_video_fps", 16),
            source_video_total_frames=orchestrator_payload.get("source_video_total_frames"),
            portions_to_regenerate=portions_to_regenerate,
            per_join_settings=per_join_settings,
            work_dir=current_run_output_dir,
            orchestrator_task_id=orchestrator_task_id_str,
            orchestrator_project_id=orchestrator_project_id,
            dprint=dprint,
            replace_mode=replace_mode
        )
        
        if not preprocess_result["success"]:
            error_msg = preprocess_result.get("error", "Unknown preprocessing error")
            dprint(f"[EDIT_VIDEO] Preprocessing failed: {error_msg}")
            return False, f"Preprocessing failed: {error_msg}"
        
        clip_list = preprocess_result["clip_list"]
        per_join_settings = preprocess_result["per_join_settings"]
        
        num_joins = len(clip_list) - 1
        dprint(f"[EDIT_VIDEO] Preprocessed into {len(clip_list)} keeper clips = {num_joins} join tasks")
        
        # === 3. VLM PROMPT ENHANCEMENT (optional) ===
        enhance_prompt = orchestrator_payload.get("enhance_prompt", False)
        vlm_enhanced_prompts: List[Optional[str]] = [None] * num_joins
        
        if enhance_prompt:
            dprint(f"[EDIT_VIDEO] enhance_prompt=True, generating VLM-enhanced prompts")
            
            vlm_device = orchestrator_payload.get("vlm_device", "cuda")
            vlm_temp_dir = current_run_output_dir / "vlm_temp"
            vlm_temp_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                image_pairs = _extract_boundary_frames_for_vlm(
                    clip_list=clip_list,
                    temp_dir=vlm_temp_dir,
                    orchestrator_task_id=orchestrator_task_id_str,
                    dprint=dprint
                )
                
                base_prompt = join_settings.get("prompt", "")
                gap_frame_count = join_settings.get("gap_frame_count", 53)
                fps = orchestrator_payload.get("source_video_fps", 16)
                
                vlm_enhanced_prompts = _generate_vlm_prompts_for_joins(
                    image_pairs=image_pairs,
                    base_prompt=base_prompt,
                    gap_frame_count=gap_frame_count,
                    fps=fps,
                    vlm_device=vlm_device,
                    dprint=dprint
                )
                
                valid_count = sum(1 for p in vlm_enhanced_prompts if p is not None)
                dprint(f"[EDIT_VIDEO] VLM enhancement complete: {valid_count}/{num_joins} prompts generated")
                
            except Exception as vlm_error:
                dprint(f"[EDIT_VIDEO] VLM enhancement failed, using base prompts: {vlm_error}")
                traceback.print_exc()
                vlm_enhanced_prompts = [None] * num_joins
        
        # === 4. IDEMPOTENCY CHECK (using shared helper) ===
        idempotency_result, idempotency_message = _check_existing_join_tasks(
            orchestrator_task_id_str, num_joins, dprint
        )
        if idempotency_result is not None:
            return idempotency_result, idempotency_message
        
        # === 5. CREATE JOIN CHAIN (using shared core function) ===
        success, message = _create_join_chain_tasks(
            clip_list=clip_list,
            run_id=run_id,
            join_settings=join_settings,
            per_join_settings=per_join_settings,
            vlm_enhanced_prompts=vlm_enhanced_prompts,
            current_run_output_dir=current_run_output_dir,
            orchestrator_task_id_str=orchestrator_task_id_str,
            orchestrator_project_id=orchestrator_project_id,
            orchestrator_payload=orchestrator_payload,
            dprint=dprint
        )
        
        dprint(f"[EDIT_VIDEO] {message}")
        return success, message
        
    except Exception as e:
        msg = f"Failed during edit_video orchestration: {e}"
        dprint(f"[EDIT_VIDEO] ERROR: {msg}")
        traceback.print_exc()
        return False, msg

