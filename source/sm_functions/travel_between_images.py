import json
import math
import shutil
import traceback
from pathlib import Path
import time
import subprocess
import uuid
from datetime import datetime
import os

# Import structured logging
from ..logging_utils import travel_logger, safe_json_repr, safe_dict_repr

# RAM monitoring
try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False
    travel_logger.warning("psutil not available - RAM monitoring disabled")

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
    prepare_output_path,
    prepare_output_path_with_upload,
    upload_and_get_final_output_location,
    snap_resolution_to_model_grid,
    ensure_valid_prompt,
    ensure_valid_negative_prompt,
    # process_additional_loras_shared,  # DEPRECATED - no longer used in queue-only system
    wait_for_file_stable as sm_wait_for_file_stable,
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
    overlay_start_end_images_above_video as sm_overlay_start_end_images_above_video,
)
# Legacy wgp_utils import removed - now using task_queue system exclusively

# Add debugging helper function
def debug_video_analysis(video_path: str | Path, label: str, task_id: str = "unknown") -> dict:
    """Analyze a video file and return comprehensive debug info"""
    try:
        path_obj = Path(video_path)
        if not path_obj.exists():
            travel_logger.debug(f"{label}: FILE MISSING - {video_path}", task_id=task_id)
            return {"exists": False, "path": str(video_path)}

        frame_count, fps = sm_get_video_frame_count_and_fps(str(path_obj))
        file_size = path_obj.stat().st_size
        duration = frame_count / fps if fps and fps > 0 else 0

        debug_info = {
            "exists": True,
            "path": str(path_obj.resolve()),
            "frame_count": frame_count,
            "fps": fps,
            "duration_seconds": duration,
            "file_size_bytes": file_size,
            "file_size_mb": round(file_size / (1024*1024), 2)
        }

        travel_logger.debug(f"{label}: {debug_info['frame_count']} frames, {debug_info['fps']} fps, {debug_info['duration_seconds']:.2f}s, {debug_info['file_size_mb']} MB", task_id=task_id)

        return debug_info

    except Exception as e:
        travel_logger.debug(f"{label}: ERROR analyzing video - {e}", task_id=task_id)
        return {"exists": False, "error": str(e), "path": str(video_path)}

# RAM monitoring helper function
def log_ram_usage(label: str, task_id: str = "unknown", logger=None) -> dict:
    """
    Log current RAM usage with a descriptive label.
    Returns dict with RAM metrics for programmatic use.
    """
    if logger is None:
        logger = travel_logger

    if not _PSUTIL_AVAILABLE:
        return {"available": False}

    try:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        rss_mb = mem_info.rss / 1024**2
        rss_gb = rss_mb / 1024

        # Get system-wide memory stats
        sys_mem = psutil.virtual_memory()
        sys_total_gb = sys_mem.total / 1024**3
        sys_available_gb = sys_mem.available / 1024**3
        sys_used_percent = sys_mem.percent

        logger.info(
            f"[RAM] {label}: Process={rss_mb:.0f}MB ({rss_gb:.2f}GB) | "
            f"System={sys_used_percent:.1f}% used, {sys_available_gb:.1f}GB/{sys_total_gb:.1f}GB available",
            task_id=task_id
        )

        return {
            "available": True,
            "process_rss_mb": rss_mb,
            "process_rss_gb": rss_gb,
            "system_total_gb": sys_total_gb,
            "system_available_gb": sys_available_gb,
            "system_used_percent": sys_used_percent
        }

    except Exception as e:
        logger.warning(f"[RAM] Failed to get RAM usage: {e}", task_id=task_id)
        return {"available": False, "error": str(e)}

# --- SM_RESTRUCTURE: New Handler Functions for Travel Tasks ---
def _handle_travel_orchestrator_task(task_params_from_db: dict, main_output_dir_base: Path, orchestrator_task_id_str: str, orchestrator_project_id: str | None, *, dprint):
    travel_logger.essential("Starting travel orchestrator task", task_id=orchestrator_task_id_str)
    log_ram_usage("Orchestrator start", task_id=orchestrator_task_id_str)
    travel_logger.debug(f"Project ID: {orchestrator_project_id}", task_id=orchestrator_task_id_str)
    # Safe logging: Use safe_json_repr to prevent hangs on large nested structures
    travel_logger.debug(f"Task params: {safe_json_repr(task_params_from_db)}", task_id=orchestrator_task_id_str)
    generation_success = False # Represents success of orchestration step
    output_message_for_orchestrator_db = f"Orchestration for {orchestrator_task_id_str} initiated."

    try:
        if 'orchestrator_details' not in task_params_from_db:
            travel_logger.error("'orchestrator_details' not found in task_params_from_db", task_id=orchestrator_task_id_str)
            return False, "orchestrator_details missing"
        
        orchestrator_payload = task_params_from_db['orchestrator_details']
        # Safe logging: Use safe_dict_repr for better performance than JSON serialization
        travel_logger.debug(f"Orchestrator payload: {safe_dict_repr(orchestrator_payload)}", task_id=orchestrator_task_id_str)

        # Parse phase_config if present and add parsed values to orchestrator_payload
        if "phase_config" in orchestrator_payload:
            travel_logger.info(f"phase_config detected in orchestrator - parsing comprehensive phase configuration", task_id=orchestrator_task_id_str)

            try:
                # Import parse_phase_config
                import sys
                # Path is already imported at module level, don't shadow it
                worker_path = Path(__file__).parent.parent / "worker.py"
                worker_dir = worker_path.parent
                if str(worker_dir) not in sys.path:
                    sys.path.insert(0, str(worker_dir))

                from worker import parse_phase_config

                # Get total steps from phase_config
                phase_config = orchestrator_payload["phase_config"]
                steps_per_phase = phase_config.get("steps_per_phase", [2, 2, 2])
                total_steps = sum(steps_per_phase)

                # Parse phase_config to get all parameters
                parsed = parse_phase_config(
                    phase_config=phase_config,
                    num_inference_steps=total_steps,
                    task_id=orchestrator_task_id_str,
                    model_name=orchestrator_payload.get("model_name")
                )

                # Add parsed values to orchestrator_payload so segments can use them
                for key in ["guidance_phases", "switch_threshold", "switch_threshold2",
                           "guidance_scale", "guidance2_scale", "guidance3_scale",
                           "flow_shift", "sample_solver", "model_switch_phase",
                           "lora_names", "lora_multipliers", "additional_loras"]:
                    if key in parsed and parsed[key] is not None:
                        orchestrator_payload[key] = parsed[key]
                        dprint(f"[ORCHESTRATOR_PHASE_CONFIG] Added {key} to orchestrator_payload: {parsed[key]}")

                # Also update num_inference_steps
                orchestrator_payload["num_inference_steps"] = total_steps

                travel_logger.info(
                    f"phase_config parsed: {parsed['guidance_phases']} phases, "
                    f"steps={total_steps}, "
                    f"{len(parsed.get('additional_loras', {}))} LoRAs, "
                    f"lora_multipliers={parsed['lora_multipliers']}",
                    task_id=orchestrator_task_id_str
                )

            except Exception as e:
                travel_logger.error(f"Failed to parse phase_config: {e}", task_id=orchestrator_task_id_str)
                import traceback
                traceback.print_exc()
                return False, f"Failed to parse phase_config: {e}"

        # IDEMPOTENCY CHECK: Look for existing child tasks before creating new ones
        dprint(f"[IDEMPOTENCY] Checking for existing child tasks for orchestrator {orchestrator_task_id_str}")
        existing_child_tasks = db_ops.get_orchestrator_child_tasks(orchestrator_task_id_str)
        existing_segments = existing_child_tasks['segments']
        existing_stitch = existing_child_tasks['stitch']
        
        expected_segments = orchestrator_payload.get("num_new_segments_to_generate", 0)
        
        if existing_segments or existing_stitch:
            dprint(f"[IDEMPOTENCY] Found existing child tasks: {len(existing_segments)} segments, {len(existing_stitch)} stitch tasks")
            
            # Check if we have the expected number of tasks already
            if len(existing_segments) >= expected_segments and len(existing_stitch) >= 1:
                # Clean up any duplicates but don't create new tasks
                cleanup_summary = db_ops.cleanup_duplicate_child_tasks(orchestrator_task_id_str, expected_segments)

                if cleanup_summary['duplicate_segments_removed'] > 0 or cleanup_summary['duplicate_stitch_removed'] > 0:
                    travel_logger.info(f"Cleaned up duplicates: {cleanup_summary['duplicate_segments_removed']} segments, {cleanup_summary['duplicate_stitch_removed']} stitch tasks", task_id=orchestrator_task_id_str)

                # CHECK: Are all child tasks actually complete?
                # If they are, we should mark orchestrator as complete instead of leaving it IN_PROGRESS
                # Also check for terminal failure states (failed/cancelled) that should mark orchestrator as failed

                def is_complete(task):
                    return task.get('status') == 'complete'

                def is_terminal_failure(task):
                    """Check if task is in a terminal failure state (failed, cancelled, etc.)"""
                    status = task.get('status', '').lower()
                    return status in ('failed', 'cancelled', 'canceled', 'error')

                all_segments_complete = all(is_complete(seg) for seg in existing_segments) if existing_segments else False
                all_stitch_complete = all(is_complete(st) for st in existing_stitch) if existing_stitch else False

                any_segment_failed = any(is_terminal_failure(seg) for seg in existing_segments) if existing_segments else False
                any_stitch_failed = any(is_terminal_failure(st) for st in existing_stitch) if existing_stitch else False

                # Also ensure we have the minimum required tasks
                has_required_segments = len(existing_segments) >= expected_segments
                has_required_stitch = len(existing_stitch) >= 1

                # If any child task failed/cancelled, mark orchestrator as failed
                if (any_segment_failed or any_stitch_failed) and has_required_segments and has_required_stitch:
                    failed_segments = [seg for seg in existing_segments if is_terminal_failure(seg)]
                    failed_stitch = [st for st in existing_stitch if is_terminal_failure(st)]

                    error_details = []
                    if failed_segments:
                        error_details.append(f"{len(failed_segments)} segment(s) failed/cancelled")
                    if failed_stitch:
                        error_details.append(f"{len(failed_stitch)} stitch task(s) failed/cancelled")

                    dprint(f"[IDEMPOTENT_FAILED] Child tasks failed: {', '.join(error_details)}")
                    travel_logger.error(f"Child tasks in terminal failure state: {', '.join(error_details)}", task_id=orchestrator_task_id_str)

                    # Return failure so orchestrator is marked as failed
                    generation_success = False
                    output_message_for_orchestrator_db = f"[ORCHESTRATOR_FAILED] Child tasks failed: {', '.join(error_details)}"
                    return generation_success, output_message_for_orchestrator_db

                if all_segments_complete and all_stitch_complete and has_required_segments and has_required_stitch:
                    # All children are done! Return with special "COMPLETE" marker
                    dprint(f"[IDEMPOTENT_COMPLETE] All {len(existing_segments)} child segments and {len(existing_stitch)} stitch tasks are complete")
                    travel_logger.info(f"All child tasks complete, orchestrator should be marked as complete", task_id=orchestrator_task_id_str)

                    # Get the final output from the stitch task
                    final_output = existing_stitch[0].get('output_location', 'Completed via idempotency')

                    # Return with special marker so worker knows to mark as COMPLETE instead of IN_PROGRESS
                    # We use a tuple with the marker to signal completion
                    generation_success = True
                    output_message_for_orchestrator_db = f"[ORCHESTRATOR_COMPLETE]{final_output}"  # Special prefix
                    travel_logger.info(f"All child tasks complete. Returning final output: {final_output}", task_id=orchestrator_task_id_str)
                    return generation_success, output_message_for_orchestrator_db
                else:
                    # Some children still in progress - report status and let worker keep waiting
                    segments_complete_count = sum(1 for seg in existing_segments if seg['status'] == 'complete')
                    stitch_complete_count = sum(1 for st in existing_stitch if st['status'] == 'complete')

                    generation_success = True
                    output_message_for_orchestrator_db = f"[IDEMPOTENT] Child tasks already exist but not all complete: {segments_complete_count}/{len(existing_segments)} segments complete, {stitch_complete_count}/{len(existing_stitch)} stitch complete. Cleaned up {cleanup_summary['duplicate_segments_removed']} duplicate segments and {cleanup_summary['duplicate_stitch_removed']} duplicate stitch tasks."
                    travel_logger.info(output_message_for_orchestrator_db, task_id=orchestrator_task_id_str)
                    return generation_success, output_message_for_orchestrator_db
            else:
                # Partial completion - log and continue with missing tasks
                dprint(f"[IDEMPOTENCY] Partial child tasks found: {len(existing_segments)}/{expected_segments} segments, {len(existing_stitch)}/1 stitch. Will continue with orchestration.")
                travel_logger.warning(f"Partial child tasks found, continuing orchestration to create missing tasks", task_id=orchestrator_task_id_str)
        else:
            dprint(f"[IDEMPOTENCY] No existing child tasks found. Proceeding with normal orchestration.")

        run_id = orchestrator_payload.get("run_id", orchestrator_task_id_str)
        base_dir_for_this_run_str = orchestrator_payload.get("main_output_dir_for_run", str(main_output_dir_base.resolve()))
        
        # Use the base directory directly without creating run-specific subdirectories
        current_run_output_dir = Path(base_dir_for_this_run_str)
        current_run_output_dir.mkdir(parents=True, exist_ok=True)
        dprint(f"Orchestrator {orchestrator_task_id_str}: Base output directory for this run: {current_run_output_dir.resolve()}")

        num_segments = orchestrator_payload.get("num_new_segments_to_generate", 0)
        if num_segments <= 0:
            msg = f"No new segments to generate based on orchestrator payload. Orchestration complete (vacuously)."
            travel_logger.warning(msg, task_id=orchestrator_task_id_str)
            return True, msg

        # Track actual DB row IDs by segment index to avoid mixing logical IDs
        actual_segment_db_id_by_index: dict[int, str] = {}

        # Track which segments already exist to avoid re-creating them
        existing_segment_indices = set()
        existing_segment_task_ids = {}  # index -> task_id mapping
        
        for segment in existing_segments:
            segment_idx = segment['params'].get('segment_index', -1)
            if segment_idx >= 0:
                existing_segment_indices.add(segment_idx)
                existing_segment_task_ids[segment_idx] = segment['id']
                # CRITICAL FIX: Pre-populate actual_segment_db_id_by_index with existing segments
                # so that new segments can correctly depend on existing ones
                actual_segment_db_id_by_index[segment_idx] = segment['id']
                
        # Check if stitch task already exists
        stitch_already_exists = len(existing_stitch) > 0
        existing_stitch_task_id = existing_stitch[0]['id'] if stitch_already_exists else None
        
        dprint(f"[IDEMPOTENCY] Existing segment indices: {sorted(existing_segment_indices)}")
        dprint(f"[IDEMPOTENCY] Stitch task exists: {stitch_already_exists} (ID: {existing_stitch_task_id})")

        # Image download directory is not needed for Supabase - images are already uploaded
        segment_image_download_dir_str : str | None = None

        # Expanded arrays from orchestrator payload
        expanded_base_prompts = orchestrator_payload["base_prompts_expanded"]
        expanded_negative_prompts = orchestrator_payload["negative_prompts_expanded"]
        expanded_segment_frames = orchestrator_payload["segment_frames_expanded"]
        expanded_frame_overlap = orchestrator_payload["frame_overlap_expanded"]
        vace_refs_instructions_all = orchestrator_payload.get("vace_image_refs_to_prepare_by_worker", [])

        # Preserve a copy of the original overlap list in case we need it later
        _orig_frame_overlap = list(expanded_frame_overlap)  # shallow copy

        # --- IDENTICAL PARAMETER DETECTION AND FRAME CONSOLIDATION ---
        def detect_identical_parameters(orchestrator_payload, num_segments, dprint=None):
            """
            Detect if all segments will have identical generation parameters.
            Returns analysis that enables both model caching and frame optimization.
            """
            # Extract parameter arrays
            expanded_base_prompts = orchestrator_payload["base_prompts_expanded"]
            expanded_negative_prompts = orchestrator_payload["negative_prompts_expanded"]
            additional_loras = orchestrator_payload.get("additional_loras", [])

            # Check parameter identity
            prompts_identical = len(set(expanded_base_prompts)) == 1
            negative_prompts_identical = len(set(expanded_negative_prompts)) == 1

            is_identical = prompts_identical and negative_prompts_identical

            if dprint and is_identical:
                dprint(f"[IDENTICAL_DETECTION] All {num_segments} segments identical - enabling optimizations")
                dprint(f"  - Unique prompt: '{expanded_base_prompts[0][:50]}...'")
                dprint(f"  - LoRA count: {len(additional_loras)}")

            return {
                "is_identical": is_identical,
                "can_optimize_frames": is_identical,  # Key for frame allocation optimization
                "can_reuse_model": is_identical,      # Key for model caching
                "unique_prompt": expanded_base_prompts[0] if prompts_identical else None
            }

        def validate_consolidation_safety(orchestrator_payload, dprint=None):
            """
            Verify that frame consolidation is safe by checking parameter identity.
            """
            # Get parameter arrays
            prompts = orchestrator_payload["base_prompts_expanded"]
            neg_prompts = orchestrator_payload["negative_prompts_expanded"]
            additional_loras = orchestrator_payload.get("additional_loras", [])

            # Critical safety checks
            all_prompts_identical = len(set(prompts)) == 1
            all_neg_prompts_identical = len(set(neg_prompts)) == 1

            is_safe = all_prompts_identical and all_neg_prompts_identical

            if dprint:
                if is_safe:
                    dprint(f"[CONSOLIDATION_SAFETY] ✅ Safe to consolidate - all parameters identical")
                else:
                    dprint(f"[CONSOLIDATION_SAFETY] ❌ NOT safe to consolidate:")
                    if not all_prompts_identical:
                        dprint(f"  - Prompts differ: {len(set(prompts))} unique prompts")
                    if not all_neg_prompts_identical:
                        dprint(f"  - Negative prompts differ: {len(set(neg_prompts))} unique")

            return {
                "is_safe": is_safe,
                "prompts_identical": all_prompts_identical,
                "negative_prompts_identical": all_neg_prompts_identical,
                "can_consolidate": is_safe
            }

        def optimize_frame_allocation_for_identical_params(orchestrator_payload, max_frames_per_segment=65, dprint=None):
            """
            When all parameters are identical, consolidate keyframes into fewer segments.

            Args:
                orchestrator_payload: Original orchestrator data
                max_frames_per_segment: Maximum frames per segment (model technical limit)
                dprint: Debug logging function

            Returns:
                Updated orchestrator_payload with optimized frame allocation
            """
            original_segment_frames = orchestrator_payload["segment_frames_expanded"]
            original_frame_overlaps = orchestrator_payload["frame_overlap_expanded"]
            original_base_prompts = orchestrator_payload["base_prompts_expanded"]

            if dprint:
                dprint(f"[FRAME_CONSOLIDATION] Original allocation: {len(original_segment_frames)} segments")
                dprint(f"  - Segment frames: {original_segment_frames}")
                dprint(f"  - Frame overlaps: {original_frame_overlaps}")

            # Calculate keyframe positions based on raw segment durations (no overlaps for consolidated videos)
            keyframe_positions = [0]  # Start with frame 0
            cumulative_pos = 0

            for segment_frames in original_segment_frames:
                cumulative_pos += segment_frames
                keyframe_positions.append(cumulative_pos)

            if dprint:
                dprint(f"[FRAME_CONSOLIDATION] Keyframe positions: {keyframe_positions}")

            # Simple consolidation: group keyframes into videos respecting frame limit
            optimized_segments = []
            optimized_overlaps = []
            optimized_prompts = []

            video_start = 0
            video_keyframes = [0]  # Always include first keyframe

            for i in range(1, len(keyframe_positions)):
                kf_pos = keyframe_positions[i]
                video_length_if_included = kf_pos - video_start + 1

                if video_length_if_included <= max_frames_per_segment:
                    # Keyframe fits in current video
                    video_keyframes.append(kf_pos)
                    if dprint:
                        dprint(f"[CONSOLIDATION_LOGIC] Keyframe {kf_pos} fits in current video (length would be {video_length_if_included})")
                else:
                    # Current video is full, finalize it and start new one
                    final_frame = video_keyframes[-1]
                    raw_length = final_frame - video_start + 1
                    quantized_length = ((raw_length - 1) // 4) * 4 + 1
                    optimized_segments.append(quantized_length)
                    optimized_prompts.append(original_base_prompts[0])

                    if dprint:
                        dprint(f"[CONSOLIDATION_LOGIC] Video complete: frames {video_start}-{final_frame} (raw: {raw_length}, quantized: {quantized_length})")
                        dprint(f"[CONSOLIDATION_LOGIC] Video keyframes: {[kf - video_start for kf in video_keyframes]}")

                    # Add overlap for the next video if there are more keyframes to process
                    # When we finalize a video because the next keyframe doesn't fit,
                    # we need overlap for the next video
                    if i < len(keyframe_positions):  # Still have more keyframes = need next video
                        # Use original overlap value instead of calculating new one
                        if isinstance(original_frame_overlaps, list) and original_frame_overlaps:
                            overlap = original_frame_overlaps[0]  # Use first value from array
                        elif isinstance(original_frame_overlaps, int):
                            overlap = original_frame_overlaps  # Use int value directly
                        else:
                            overlap = 4  # Default fallback if no overlap specified

                        optimized_overlaps.append(overlap)
                        if dprint:
                            dprint(f"[CONSOLIDATION_LOGIC] Added overlap {overlap} frames for next video (from original settings)")

                    # Start new video
                    video_start = video_keyframes[-1]  # Start from last keyframe of previous video
                    video_keyframes = [video_start, kf_pos]
                    if dprint:
                        dprint(f"[CONSOLIDATION_LOGIC] Starting new video at frame {video_start}")

            # Finalize the last video
            final_frame = video_keyframes[-1]
            raw_length = final_frame - video_start + 1
            quantized_length = ((raw_length - 1) // 4) * 4 + 1
            optimized_segments.append(quantized_length)
            optimized_prompts.append(original_base_prompts[0])

            if dprint:
                dprint(f"[CONSOLIDATION_LOGIC] Final video: frames {video_start}-{final_frame} (raw: {raw_length}, quantized: {quantized_length})")
                dprint(f"[CONSOLIDATION_LOGIC] Final video keyframes: {[kf - video_start for kf in video_keyframes]}")

            # SANITY CHECK: Consolidation should NEVER increase segment count
            original_num_segments = len(original_segment_frames)
            new_num_segments = len(optimized_segments)

            if new_num_segments > original_num_segments:
                # This should never happen - consolidation split segments instead of combining them!
                if dprint:
                    dprint(f"[CONSOLIDATION_ERROR] ❌ Consolidation increased segments from {original_num_segments} to {new_num_segments} - ABORTING optimization")
                dprint(f"[FRAME_CONSOLIDATION] ❌ ERROR: Consolidation would increase segments ({original_num_segments} → {new_num_segments}) - keeping original allocation")
                # Return early without modifying the payload
                return orchestrator_payload

            # Update orchestrator payload
            orchestrator_payload["segment_frames_expanded"] = optimized_segments
            orchestrator_payload["frame_overlap_expanded"] = optimized_overlaps
            orchestrator_payload["base_prompts_expanded"] = optimized_prompts
            orchestrator_payload["negative_prompts_expanded"] = [orchestrator_payload["negative_prompts_expanded"][0]] * len(optimized_segments)
            orchestrator_payload["num_new_segments_to_generate"] = len(optimized_segments)

            # CRITICAL: Store end anchor image indices for consolidated segments
            # This tells each consolidated segment which image should be its end anchor
            consolidated_end_anchors = []
            original_num_segments = len(original_segment_frames)

            # For consolidated segments, calculate the correct end anchor indices
            # Each consolidated segment should use the final image of its range
            # Use the simplified approach: track which images each segment should end with
            consolidated_end_anchors = []

            # First segment ends with the image at the last keyframe it contains
            if len(optimized_segments) >= 1:
                # First segment: determine which keyframes it contains based on consolidation logic
                # Recreate the consolidation to find the correct end images
                video_start = 0
                video_keyframes = [0]  # Always include first keyframe
                current_image_idx = 0

                for i in range(1, len(keyframe_positions)):
                    kf_pos = keyframe_positions[i]
                    video_length_if_included = kf_pos - video_start + 1

                    if video_length_if_included <= max_frames_per_segment:
                        # Keyframe fits in current video
                        video_keyframes.append(kf_pos)
                        current_image_idx = i  # This image index goes in current video
                    else:
                        # Finalize current video - end with current_image_idx
                        consolidated_end_anchors.append(current_image_idx)
                        dprint(f"[FRAME_CONSOLIDATION] Segment {len(consolidated_end_anchors)-1}: end_anchor_image_index = {current_image_idx}")

                        # Start new video
                        video_start = video_keyframes[-1]
                        video_keyframes = [video_start, kf_pos]
                        current_image_idx = i  # Current keyframe goes in new video

                # Handle the final segment
                consolidated_end_anchors.append(current_image_idx)
                dprint(f"[FRAME_CONSOLIDATION] Segment {len(consolidated_end_anchors)-1}: end_anchor_image_index = {current_image_idx}")

            # Store the end anchor mapping for use during segment creation
            orchestrator_payload["_consolidated_end_anchors"] = consolidated_end_anchors

            # Calculate relative keyframe positions AND image indices for each consolidated segment
            consolidated_keyframe_segments = []
            consolidated_keyframe_image_indices = []

            # Recreate the same consolidation logic to properly assign keyframes
            video_start = 0
            video_keyframes = [0]  # Always include first keyframe (absolute positions)
            video_image_indices = [0]  # Track which input images correspond to keyframes
            current_video_idx = 0

            for i in range(1, len(keyframe_positions)):
                kf_pos = keyframe_positions[i]
                video_length_if_included = kf_pos - video_start + 1

                if video_length_if_included <= max_frames_per_segment:
                    # Keyframe fits in current video
                    video_keyframes.append(kf_pos)
                    video_image_indices.append(i)  # Input image index corresponds to keyframe index
                else:
                    # Finalize current video and start new one
                    final_frame = video_keyframes[-1]

                    # Convert absolute keyframe positions to relative positions for this video
                    # BUT: adjust for quantization - keyframes must fit within quantized segment bounds
                    raw_length = final_frame - video_start + 1
                    quantized_length = ((raw_length - 1) // 4) * 4 + 1

                    relative_keyframes = []
                    for kf_abs_pos in video_keyframes:
                        relative_pos = kf_abs_pos - video_start
                        # Ensure final keyframe fits within quantized bounds
                        if relative_pos >= quantized_length:
                            relative_pos = quantized_length - 1  # Last frame in quantized video
                        relative_keyframes.append(relative_pos)

                    consolidated_keyframe_segments.append(relative_keyframes)
                    consolidated_keyframe_image_indices.append(video_image_indices.copy())

                    if dprint:
                        dprint(f"[KEYFRAME_ASSIGNMENT] Video {current_video_idx}: absolute start {video_start}, final frame {final_frame}")
                        dprint(f"[KEYFRAME_ASSIGNMENT] Video {current_video_idx} keyframes: {video_keyframes} → relative: {relative_keyframes}")
                        dprint(f"[KEYFRAME_ASSIGNMENT] Video {current_video_idx} image indices: {video_image_indices}")

                    # Start new video
                    current_video_idx += 1
                    video_start = final_frame  # Start from last keyframe (overlap)
                    # The overlap keyframe uses the same image as the final keyframe of previous segment
                    last_image_idx = video_image_indices[-1]
                    video_keyframes = [final_frame, kf_pos]  # Include overlap and current keyframe
                    video_image_indices = [last_image_idx, i]  # Include overlap image and current image

            # Handle the last video (make sure it has the correct final keyframes)
            if len(video_keyframes) > 0:
                # Convert absolute keyframe positions to relative positions for the final video
                # Adjust for quantization like the consolidation logic does
                final_frame = video_keyframes[-1]
                raw_length = final_frame - video_start + 1
                quantized_length = ((raw_length - 1) // 4) * 4 + 1

                relative_keyframes = []
                for kf_abs_pos in video_keyframes:
                    relative_pos = kf_abs_pos - video_start
                    # Ensure final keyframe fits within quantized bounds
                    if relative_pos >= quantized_length:
                        relative_pos = quantized_length - 1  # Last frame in quantized video
                    relative_keyframes.append(relative_pos)

                consolidated_keyframe_segments.append(relative_keyframes)
                consolidated_keyframe_image_indices.append(video_image_indices.copy())

                if dprint:
                    dprint(f"[KEYFRAME_ASSIGNMENT] Video {current_video_idx}: absolute start {video_start}")
                    dprint(f"[KEYFRAME_ASSIGNMENT] Video {current_video_idx} keyframes: {video_keyframes} → relative: {relative_keyframes}")
                    dprint(f"[KEYFRAME_ASSIGNMENT] Video {current_video_idx} image indices: {video_image_indices}")

            # Store relative keyframe positions for guide video creation
            orchestrator_payload["_consolidated_keyframe_positions"] = consolidated_keyframe_segments

            if dprint:
                dprint(f"[FRAME_CONSOLIDATION] Optimized to {len(optimized_segments)} segments (was {len(original_segment_frames)})")
                dprint(f"  - New segment frames: {optimized_segments}")
                dprint(f"  - New overlaps: {optimized_overlaps}")
                dprint(f"  - Efficiency: {(len(original_segment_frames) - len(optimized_segments))} fewer segments")

            return orchestrator_payload


        # --- SM_QUANTIZE_FRAMES_AND_OVERLAPS ---
        # Adjust all segment lengths to match model constraints (4*N+1 format).
        # Then, adjust overlap values to be even and not exceed the length of the
        # smaller of the two segments they connect. This prevents errors downstream
        # in guide video creation, generation, and stitching.

        dprint(f"[FRAME_DEBUG] Orchestrator {orchestrator_task_id_str}: QUANTIZATION ANALYSIS")
        dprint(f"[FRAME_DEBUG] Original segment_frames_expanded: {expanded_segment_frames}")
        dprint(f"[FRAME_DEBUG] Original frame_overlap: {expanded_frame_overlap}")
        
        quantized_segment_frames = []
        dprint(f"Orchestrator: Quantizing frame counts. Original segment_frames_expanded: {expanded_segment_frames}")
        for i, frames in enumerate(expanded_segment_frames):
            # Quantize to 4*N+1 format to match model constraints, applied later in worker.py
            new_frames = (frames // 4) * 4 + 1
            dprint(f"[FRAME_DEBUG] Segment {i}: {frames} -> {new_frames} (4*N+1 quantization)")
            if new_frames != frames:
                dprint(f"Orchestrator: Quantized segment {i} length from {frames} to {new_frames} (4*N+1 format).")
            quantized_segment_frames.append(new_frames)
        
        dprint(f"[FRAME_DEBUG] Quantized segment_frames: {quantized_segment_frames}")
        dprint(f"Orchestrator: Finished quantizing frame counts. New quantized_segment_frames: {quantized_segment_frames}")
        
        quantized_frame_overlap = []
        # There are N-1 overlaps for N segments. The loop must not iterate more times than this.
        num_overlaps_to_process = len(quantized_segment_frames) - 1
        dprint(f"[FRAME_DEBUG] Processing {num_overlaps_to_process} overlap values")

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

                dprint(f"[FRAME_DEBUG] Overlap {i} (segments {i}->{i+1}): {original_overlap} -> {new_overlap}")
                dprint(f"[FRAME_DEBUG]   Segment lengths: {quantized_segment_frames[i]}, {quantized_segment_frames[i+1]}")
                dprint(f"[FRAME_DEBUG]   Max possible overlap: {max_possible_overlap}")
                
                if new_overlap != original_overlap:
                    dprint(f"Orchestrator: Adjusted overlap between segments {i}-{i+1} from {original_overlap} to {new_overlap}.")
                
                quantized_frame_overlap.append(new_overlap)
        
        dprint(f"[FRAME_DEBUG] Final quantized_frame_overlap: {quantized_frame_overlap}")
        
        # Persist quantised results back to orchestrator_payload so all downstream tasks see them
        orchestrator_payload["segment_frames_expanded"] = quantized_segment_frames
        orchestrator_payload["frame_overlap_expanded"] = quantized_frame_overlap
        
        # Calculate expected final length
        total_input_frames = sum(quantized_segment_frames)
        total_overlaps = sum(quantized_frame_overlap)
        expected_final_length = total_input_frames - total_overlaps
        dprint(f"[FRAME_DEBUG] EXPECTED FINAL VIDEO:")
        dprint(f"[FRAME_DEBUG]   Total input frames: {total_input_frames}")
        dprint(f"[FRAME_DEBUG]   Total overlaps: {total_overlaps}")
        dprint(f"[FRAME_DEBUG]   Expected final length: {expected_final_length} frames")
        dprint(f"[FRAME_DEBUG]   Expected duration: {expected_final_length / orchestrator_payload.get('fps_helpers', 16):.2f}s")
        
        # Replace original lists with the new quantized ones for all subsequent logic
        expanded_segment_frames = quantized_segment_frames
        expanded_frame_overlap = quantized_frame_overlap
        # --- END SM_QUANTIZE_FRAMES_AND_OVERLAPS ---

        # If quantisation resulted in an empty overlap list (e.g. single-segment run) but the
        # original payload DID contain an overlap value, restore that so the first segment
        # can still reuse frames from the previous/continued video.  This is crucial for
        # continue-video journeys where we expect `frame_overlap_from_previous` > 0.
        if (not expanded_frame_overlap) and _orig_frame_overlap:
            expanded_frame_overlap = _orig_frame_overlap

        # --- FRAME CONSOLIDATION OPTIMIZATION ---
        # Store original values for comparison logging
        original_num_segments = num_segments
        original_segment_frames = list(expanded_segment_frames)
        original_frame_overlap = list(expanded_frame_overlap)

        # Check if all prompts and LoRAs are identical to enable frame consolidation
        # IMPORTANT: Disable consolidation if enhance_prompt is enabled, because VLM will
        # generate different prompts for each segment AFTER consolidation would have run.
        # This would create consolidated segments with different prompts, breaking the
        # consolidation assumption that all segments have identical parameters.
        enhance_prompt_enabled = orchestrator_payload.get("enhance_prompt", False)
        if enhance_prompt_enabled:
            dprint(f"[FRAME_CONSOLIDATION] enhance_prompt=True detected - DISABLING frame consolidation")
            dprint(f"[FRAME_CONSOLIDATION] Reason: VLM will generate unique prompts per segment, breaking identity assumption")
            identity_analysis = {
                "can_optimize_frames": False,
                "can_reuse_model": False,
                "is_identical": False
            }
        else:
            identity_analysis = detect_identical_parameters(orchestrator_payload, num_segments, dprint)

        if identity_analysis["can_optimize_frames"]:
            # Only run consolidation if there are multiple segments to consolidate
            num_segments = len(orchestrator_payload["segment_frames_expanded"])
            if num_segments <= 1:
                dprint(f"[FRAME_CONSOLIDATION] ⏭️  Skipping optimization - only {num_segments} segment(s), nothing to consolidate")
                travel_logger.info(f"Frame consolidation: Only {num_segments} segment(s) - no consolidation needed", task_id=orchestrator_task_id_str)
            else:
                # Run safety validation before optimization
                safety_check = validate_consolidation_safety(orchestrator_payload, dprint)

                if safety_check["is_safe"]:
                    dprint(f"[FRAME_CONSOLIDATION] ✅ Triggering optimization for identical parameters")
                    travel_logger.info("Frame consolidation: All parameters identical - enabling optimization", task_id=orchestrator_task_id_str)

                    # Apply frame consolidation optimization
                    orchestrator_payload = optimize_frame_allocation_for_identical_params(
                        orchestrator_payload,
                        max_frames_per_segment=81,  # Max 81 frames per video
                        dprint=dprint
                    )

                    # Update variables with optimized values
                    expanded_segment_frames = orchestrator_payload["segment_frames_expanded"]
                    expanded_frame_overlap = orchestrator_payload["frame_overlap_expanded"]
                    expanded_base_prompts = orchestrator_payload["base_prompts_expanded"]
                    expanded_negative_prompts = orchestrator_payload["negative_prompts_expanded"]
                    num_segments = orchestrator_payload["num_new_segments_to_generate"]

                    # CRITICAL: Update VACE image references for consolidated segments
                    # When segments are consolidated, we need to reassign all VACE image refs
                    # to the new consolidated segments based on their new indices
                    if vace_refs_instructions_all:
                        dprint(f"[VACE_REFS_CONSOLIDATION] Updating VACE image refs for {num_segments} consolidated segments")
                        dprint(f"[VACE_REFS_CONSOLIDATION] Original VACE refs count: {len(vace_refs_instructions_all)}")

                        # For consolidated segments, reassign all VACE refs to the appropriate new segment
                        # based on the original keyframe positions and new segment boundaries
                        for ref_idx, ref_instr in enumerate(vace_refs_instructions_all):
                            original_segment_idx = ref_instr.get("segment_idx_for_naming", 0)

                            # For now, assign all refs to the first (and often only) consolidated segment
                            # This ensures the consolidated segment gets all the keyframe images
                            new_segment_idx = 0 if num_segments == 1 else min(original_segment_idx, num_segments - 1)

                            if original_segment_idx != new_segment_idx:
                                dprint(f"[VACE_REFS_CONSOLIDATION] VACE ref {ref_idx}: segment {original_segment_idx} → {new_segment_idx}")
                                ref_instr["segment_idx_for_naming"] = new_segment_idx

                        dprint(f"[VACE_REFS_CONSOLIDATION] VACE refs updated for consolidated segments")

                    # Summary logging for optimization results
                    segments_saved = original_num_segments - num_segments
                    travel_logger.info(f"Frame consolidation optimization: {original_num_segments} → {num_segments} segments (saved {segments_saved})", task_id=orchestrator_task_id_str)
                    travel_logger.debug(f"Original allocation: {original_segment_frames}, Optimized: {expanded_segment_frames}", task_id=orchestrator_task_id_str)
                    travel_logger.debug(f"Original overlaps: {original_frame_overlap}, Optimized: {expanded_frame_overlap}", task_id=orchestrator_task_id_str)

                    dprint(f"[FRAME_CONSOLIDATION] Successfully updated to {num_segments} optimized segments")
                else:
                    travel_logger.warning("Frame consolidation: Safety validation failed - parameters not identical enough", task_id=orchestrator_task_id_str)
                    dprint(f"[FRAME_CONSOLIDATION] Safety check failed - keeping original allocation")
        else:
            travel_logger.info("Frame consolidation: Parameters not identical - using standard allocation", task_id=orchestrator_task_id_str)
            dprint(f"[FRAME_CONSOLIDATION] Parameters not identical - keeping original allocation")
        # --- END FRAME CONSOLIDATION OPTIMIZATION ---

        # Extract and validate structure video parameters
        structure_video_path = orchestrator_payload.get("structure_video_path")
        structure_video_treatment = orchestrator_payload.get("structure_video_treatment", "adjust")
        structure_type = orchestrator_payload.get("structure_video_type", orchestrator_payload.get("structure_type", "flow"))  # Check both keys
        travel_logger.info(f"Structure video config: type={structure_type}, treatment={structure_video_treatment}, path={'YES' if structure_video_path else 'NO'}", task_id=orchestrator_task_id_str)

        # Extract strength parameters for each type
        motion_strength = orchestrator_payload.get("structure_video_motion_strength", 1.0)
        canny_intensity = orchestrator_payload.get("structure_canny_intensity", 1.0)
        depth_contrast = orchestrator_payload.get("structure_depth_contrast", 1.0)
        
        if structure_video_path:
            # Capture original URL if it's a remote path
            if isinstance(structure_video_path, str) and structure_video_path.startswith(("http://", "https://")):
                 orchestrator_payload["structure_original_video_url"] = structure_video_path

            # Download if URL
            from ..common_utils import download_video_if_url
            structure_video_path = download_video_if_url(
                structure_video_path,
                download_target_dir=current_run_output_dir,
                task_id_for_logging=orchestrator_task_id_str,
                descriptive_name="structure_video"
            )
            
            # Validate structure video exists (after potential download)
            if not Path(structure_video_path).exists():
                raise ValueError(f"Structure video not found: {structure_video_path}")
            
            # Validate treatment mode
            if structure_video_treatment not in ["adjust", "clip"]:
                raise ValueError(f"Invalid structure_video_treatment: {structure_video_treatment}. Must be 'adjust' or 'clip'")
            
            # Validate structure_type
            if structure_type not in ["flow", "canny", "depth"]:
                raise ValueError(f"Invalid structure_type: {structure_type}. Must be 'flow', 'canny', or 'depth'")
            
            # Warn about unused strength parameters
            if structure_type == "flow":
                if abs(canny_intensity - 1.0) > 1e-6:
                    dprint(f"[STRUCTURE_VIDEO] Warning: structure_canny_intensity={canny_intensity} ignored (structure_type is 'flow')")
                if abs(depth_contrast - 1.0) > 1e-6:
                    dprint(f"[STRUCTURE_VIDEO] Warning: structure_depth_contrast={depth_contrast} ignored (structure_type is 'flow')")
            elif structure_type == "canny":
                if abs(motion_strength - 1.0) > 1e-6:
                    dprint(f"[STRUCTURE_VIDEO] Warning: structure_video_motion_strength={motion_strength} ignored (structure_type is 'canny')")
                if abs(depth_contrast - 1.0) > 1e-6:
                    dprint(f"[STRUCTURE_VIDEO] Warning: structure_depth_contrast={depth_contrast} ignored (structure_type is 'canny')")
            elif structure_type == "depth":
                if abs(motion_strength - 1.0) > 1e-6:
                    dprint(f"[STRUCTURE_VIDEO] Warning: structure_video_motion_strength={motion_strength} ignored (structure_type is 'depth')")
                if abs(canny_intensity - 1.0) > 1e-6:
                    dprint(f"[STRUCTURE_VIDEO] Warning: structure_canny_intensity={canny_intensity} ignored (structure_type is 'depth')")
            
            # Log structure video configuration
            strength_param = {
                "flow": f"motion_strength={motion_strength}",
                "canny": f"canny_intensity={canny_intensity}",
                "depth": f"depth_contrast={depth_contrast}"
            }.get(structure_type, "unknown")

            travel_logger.info(
                f"Structure video enabled: type={structure_type}, treatment={structure_video_treatment}, {strength_param}",
                task_id=orchestrator_task_id_str
            )
            travel_logger.info(f"Structure video file: {structure_video_path}", task_id=orchestrator_task_id_str)

            # Calculate TOTAL flow frames needed across ALL segments
            # Flow visualizations are created for ALL frames INCLUDING anchors (even though anchors
            # will be replaced by keyframes), but EXCLUDING overlap frames (reused from previous segment)
            #
            # Example: 3 segments × 61 frames, 8 frame overlap
            # Segment 0: 61 flows (frames 0-60, including both anchor frames)
            # Segment 1: 53 flows (frames 8-60, excluding 8 overlap frames)
            # Segment 2: 53 flows (frames 8-60, excluding 8 overlap frames)
            # Total: 61 + 53 + 53 = 167 flows

            total_flow_frames = 0
            segment_flow_offsets = []  # Track where each segment's flows start

            for idx in range(num_segments):
                segment_total_frames = expanded_segment_frames[idx]

                if idx == 0 and not orchestrator_payload.get("continue_from_video_resolved_path"):
                    # First segment: all frames
                    segment_flow_offsets.append(0)
                    total_flow_frames = segment_total_frames
                    dprint(f"[STRUCTURE_VIDEO] Segment {idx}: offset=0, adds {segment_total_frames} frames, total={total_flow_frames}")
                else:
                    # Subsequent segments: Structure frames are REUSED in overlap region
                    # Segment generates (segment_total_frames + overlap) frames but reads structure
                    # starting from (current_position - overlap) to reuse overlap structure frames
                    overlap = expanded_frame_overlap[idx - 1] if idx > 0 else 0
                    segment_offset = total_flow_frames - overlap
                    segment_flow_offsets.append(segment_offset)
                    # Add only non-overlapping frames to total structure length
                    total_flow_frames += segment_total_frames
                    dprint(f"[STRUCTURE_VIDEO] Segment {idx}: offset={segment_offset}, overlap={overlap}, adds {segment_total_frames} new frames, total={total_flow_frames}")

            travel_logger.info(f"Structure video processing: {total_flow_frames} total frames needed across {num_segments} segments", task_id=orchestrator_task_id_str)

            # Create and upload pre-warped guidance video for segments to download
            travel_logger.info("Creating structure guidance video...", task_id=orchestrator_task_id_str)
            try:
                from source.structure_video_guidance import create_structure_guidance_video, create_trimmed_structure_video
                
                # Get resolution and FPS from orchestrator payload
                target_resolution_raw = orchestrator_payload["parsed_resolution_wh"]
                target_fps = orchestrator_payload.get("fps_helpers", 16)
                
                # Parse resolution if it's a string (e.g., "768x576" -> (768, 576))
                if isinstance(target_resolution_raw, str):
                    parsed_res = sm_parse_resolution(target_resolution_raw)
                    if parsed_res is None:
                        raise ValueError(f"Invalid resolution format: {target_resolution_raw}")
                    target_resolution = snap_resolution_to_model_grid(parsed_res)
                else:
                    target_resolution = target_resolution_raw
                
                # Generate unique filename to avoid race conditions and enable idempotency
                timestamp_short = datetime.now().strftime("%H%M%S")
                unique_suffix = uuid.uuid4().hex[:6]
                
                # 1. Create trimmed video (original content adjusted to timeline)
                travel_logger.info("Creating trimmed structure video (original content adjusted to timeline)...", task_id=orchestrator_task_id_str)
                trimmed_filename = f"structure_trimmed_{timestamp_short}_{unique_suffix}.mp4"
                
                trimmed_video_path = create_trimmed_structure_video(
                    structure_video_path=structure_video_path,
                    max_frames_needed=total_flow_frames,
                    target_resolution=target_resolution,
                    target_fps=target_fps,
                    output_path=current_run_output_dir / trimmed_filename,
                    treatment=structure_video_treatment,
                    dprint=dprint
                )
                
                # Upload trimmed video
                trimmed_video_url = upload_and_get_final_output_location(
                    local_file_path=trimmed_video_path,
                    supabase_object_name=trimmed_filename,
                    initial_db_location=str(trimmed_video_path),
                    dprint=dprint
                )
                orchestrator_payload["structure_trimmed_video_url"] = trimmed_video_url
                travel_logger.success(f"Trimmed structure video created: {trimmed_video_url}", task_id=orchestrator_task_id_str)

                # 2. Create pre-warped guidance video (processed with flow/canny/depth)
                structure_guidance_filename = f"structure_{structure_type}_{timestamp_short}_{unique_suffix}.mp4"

                structure_guidance_video_path = create_structure_guidance_video(
                    structure_video_path=structure_video_path,
                    max_frames_needed=total_flow_frames,
                    target_resolution=target_resolution,
                    target_fps=target_fps,
                    output_path=current_run_output_dir / structure_guidance_filename,
                    structure_type=structure_type,
                    motion_strength=motion_strength,
                    canny_intensity=canny_intensity,
                    depth_contrast=depth_contrast,
                    treatment=structure_video_treatment,
                    dprint=dprint
                )
                
                # Store path for segments (edge function will handle upload)
                structure_guidance_video_url = upload_and_get_final_output_location(
                    local_file_path=structure_guidance_video_path,
                    supabase_object_name=structure_guidance_filename,  # Unused but required for signature
                    initial_db_location=str(structure_guidance_video_path),
                    dprint=dprint
                )

                travel_logger.success(f"Structure guidance video created: {structure_guidance_video_url}", task_id=orchestrator_task_id_str)

                # Get frame count for logging
                guidance_frame_count, _ = sm_get_video_frame_count_and_fps(structure_guidance_video_path)
                travel_logger.info(f"Structure guidance frames: {guidance_frame_count} (treatment: {structure_video_treatment})", task_id=orchestrator_task_id_str)

                # Store URL and type in orchestrator payload for segments to use
                orchestrator_payload["structure_guidance_video_url"] = structure_guidance_video_url
                orchestrator_payload["structure_type"] = structure_type

            except Exception as e:
                travel_logger.error(f"Failed to create structure guidance video: {e}", task_id=orchestrator_task_id_str)
                import traceback
                traceback.print_exc()
                travel_logger.warning("Structure guidance will not be available for this generation", task_id=orchestrator_task_id_str)
                orchestrator_payload["structure_guidance_video_url"] = None
                orchestrator_payload["structure_type"] = structure_type  # Still pass the type

        # --- VLM BATCH PROCESSING (if enhance_prompt enabled) ---
        # Generate all transition prompts upfront to reuse the VLM model across segments
        vlm_enhanced_prompts = {}  # Dict: segment_idx -> enhanced_prompt
        if orchestrator_payload.get("enhance_prompt", False):
            dprint(f"[VLM_BATCH] enhance_prompt enabled - generating transition prompts for all segments...")
            log_ram_usage("Before VLM loading", task_id=orchestrator_task_id_str)
            try:
                # Import VLM helper
                from ..vlm_utils import generate_transition_prompts_batch
                from ..common_utils import download_image_if_url

                # Get input images
                input_images_resolved = orchestrator_payload.get("input_image_paths_resolved", [])
                vlm_device = orchestrator_payload.get("vlm_device", "cuda")

                # Get pre-existing enhanced prompts if available
                expanded_enhanced_prompts = orchestrator_payload.get("enhanced_prompts_expanded", [])
                base_prompt = orchestrator_payload.get("base_prompt", "")

                if base_prompt:
                    dprint(f"[VLM_BATCH] Base prompt from payload: '{base_prompt[:80]}...'")
                else:
                    dprint(f"[VLM_BATCH] No base_prompt found in payload")

                if expanded_enhanced_prompts:
                    dprint(f"[VLM_BATCH] Found {len(expanded_enhanced_prompts)} pre-existing enhanced prompts in payload")
                    for idx, prompt in enumerate(expanded_enhanced_prompts):
                        if prompt and prompt.strip():
                            dprint(f"[VLM_BATCH]   Segment {idx}: '{prompt[:80]}...'")
                else:
                    dprint(f"[VLM_BATCH] No pre-existing enhanced prompts found in payload")

                # Build lists of image pairs, base prompts, and frame counts
                image_pairs = []
                base_prompts_for_batch = []
                segment_frame_counts = []  # Frame count for each segment
                segment_indices = []  # Track which segment each pair belongs to

                for idx in range(num_segments):
                    # Check if enhanced prompt already exists for this segment
                    if idx < len(expanded_enhanced_prompts) and expanded_enhanced_prompts[idx] and expanded_enhanced_prompts[idx].strip():
                        # Use existing enhanced prompt, skip VLM enrichment
                        vlm_enhanced_prompts[idx] = expanded_enhanced_prompts[idx]
                        dprint(f"[VLM_BATCH] Segment {idx}: Using pre-existing enhanced prompt: {expanded_enhanced_prompts[idx][:80]}...")
                        continue

                    # Determine which images this segment transitions between
                    if orchestrator_payload.get("_consolidated_end_anchors"):
                        consolidated_end_anchors = orchestrator_payload["_consolidated_end_anchors"]
                        if idx < len(consolidated_end_anchors):
                            end_anchor_idx = consolidated_end_anchors[idx]
                            start_anchor_idx = 0 if idx == 0 else consolidated_end_anchors[idx - 1]
                        else:
                            start_anchor_idx = idx
                            end_anchor_idx = idx + 1
                    else:
                        start_anchor_idx = idx
                        end_anchor_idx = idx + 1

                    # Ensure indices are within bounds
                    if (start_anchor_idx < len(input_images_resolved) and
                        end_anchor_idx < len(input_images_resolved)):

                        start_image_path = input_images_resolved[start_anchor_idx]
                        end_image_path = input_images_resolved[end_anchor_idx]

                        # Download images if they're URLs
                        start_image_path = download_image_if_url(
                            start_image_path,
                            current_run_output_dir,
                            f"vlm_start_{idx}",
                            debug_mode=False,
                            descriptive_name=f"vlm_start_seg{idx}"
                        )
                        end_image_path = download_image_if_url(
                            end_image_path,
                            current_run_output_dir,
                            f"vlm_end_{idx}",
                            debug_mode=False,
                            descriptive_name=f"vlm_end_seg{idx}"
                        )

                        image_pairs.append((start_image_path, end_image_path))
                        # Use segment-specific base_prompt if available, otherwise use overall base_prompt
                        segment_base_prompt = expanded_base_prompts[idx] if expanded_base_prompts[idx] and expanded_base_prompts[idx].strip() else base_prompt
                        base_prompts_for_batch.append(segment_base_prompt)
                        # Get frame count for this segment for duration calculation
                        segment_frames = expanded_segment_frames[idx] if idx < len(expanded_segment_frames) else None
                        segment_frame_counts.append(segment_frames)
                        segment_indices.append(idx)
                    else:
                        dprint(f"[VLM_BATCH] Segment {idx}: Skipping - image indices out of bounds")

                # Generate all prompts in one batch (reuses VLM model)
                if image_pairs:
                    # Get FPS from orchestrator payload (default to 16)
                    fps_helpers = orchestrator_payload.get("fps_helpers", 16)

                    enhanced_prompts = generate_transition_prompts_batch(
                        image_pairs=image_pairs,
                        base_prompts=base_prompts_for_batch,
                        num_frames_list=segment_frame_counts,
                        fps=fps_helpers,
                        device=vlm_device,
                        dprint=dprint
                    )

                    # Map results back to segment indices
                    for idx, enhanced in zip(segment_indices, enhanced_prompts):
                        vlm_enhanced_prompts[idx] = enhanced
                        dprint(f"[VLM_BATCH] Segment {idx}: {enhanced[:80]}...")

                dprint(f"[VLM_BATCH] Generated {len(vlm_enhanced_prompts)} enhanced prompts")
                log_ram_usage("After VLM cleanup", task_id=orchestrator_task_id_str)

                # Call Supabase edge function to update shot_generations with newly enriched prompts
                try:
                    import httpx

                    # Build complete enhanced_prompts array (empty strings for non-enriched segments)
                    complete_enhanced_prompts = []
                    for idx in range(num_segments):
                        if idx in vlm_enhanced_prompts:
                            complete_enhanced_prompts.append(vlm_enhanced_prompts[idx])
                        else:
                            complete_enhanced_prompts.append("")

                    # Only call if we have SUPABASE configured and generated any new prompts
                    # Use SERVICE_KEY if available (admin), otherwise use ACCESS_TOKEN (user with ownership check)
                    auth_token = db_ops.SUPABASE_SERVICE_KEY or db_ops.SUPABASE_ACCESS_TOKEN
                    if db_ops.SUPABASE_URL and auth_token and len(complete_enhanced_prompts) > 0:
                        # Extract shot_id from orchestrator_payload
                        shot_id = orchestrator_payload.get("shot_id")
                        if not shot_id:
                            dprint(f"[VLM_BATCH] WARNING: No shot_id found in orchestrator_payload, skipping edge function call")
                        else:
                            # Call edge function to update shot_generations with enhanced prompts
                            edge_url = f"{db_ops.SUPABASE_URL.rstrip('/')}/functions/v1/update-shot-pair-prompts"
                            headers = {"Content-Type": "application/json"}
                            if auth_token:
                                headers["Authorization"] = f"Bearer {auth_token}"

                            payload = {
                                "shot_id": shot_id,
                                "enhanced_prompts": complete_enhanced_prompts
                            }

                            dprint(f"[VLM_BATCH] Calling edge function to update shot_generations with enhanced prompts...")
                            dprint(f"[VLM_BATCH] Payload: shot_id={shot_id}, enhanced_prompts={len(complete_enhanced_prompts)} items")
                            dprint(f"[VLM_BATCH] Using auth token: {'SERVICE_KEY' if db_ops.SUPABASE_SERVICE_KEY else ('ACCESS_TOKEN' if db_ops.SUPABASE_ACCESS_TOKEN else 'None')}")

                            resp = httpx.post(edge_url, json=payload, headers=headers, timeout=30)

                            if resp.status_code == 200:
                                dprint(f"[VLM_BATCH] Successfully updated shot_generations via edge function")
                                resp_json = resp.json()
                                dprint(f"[VLM_BATCH] Edge function response: {resp_json}")
                            else:
                                dprint(f"[VLM_BATCH] WARNING: Edge function call failed: {resp.status_code} - {resp.text}")
                    else:
                        dprint(f"[VLM_BATCH] Skipping edge function call (has_auth_token={bool(auth_token)}, has_supabase_url={bool(db_ops.SUPABASE_URL)}, generated={len(complete_enhanced_prompts)} prompts)")

                except Exception as e_edge:
                    dprint(f"[VLM_BATCH] WARNING: Failed to call edge function: {e_edge}")
                    import traceback
                    traceback.print_exc()
                    # Non-fatal - continue with task creation

            except Exception as e_vlm_batch:
                dprint(f"[VLM_BATCH] ERROR during batch VLM processing: {e_vlm_batch}")
                import traceback
                traceback.print_exc()
                dprint(f"[VLM_BATCH] Falling back to original prompts for all segments")
                vlm_enhanced_prompts = {}

        # Loop to queue all segment tasks (skip existing ones for idempotency)
        segments_created = 0
        for idx in range(num_segments):
            # Get travel mode for dependency logic
            travel_mode = orchestrator_payload.get("model_type", "vace")
            independent_segments = orchestrator_payload.get("independent_segments", False)
            
            # Determine dependency strictly from previously resolved actual DB IDs
            # I2V MODE: Independent segments (no dependency on previous task)
            if travel_mode == "i2v":
                previous_segment_task_id = None
                dprint(f"[DEBUG_DEPENDENCY_CHAIN] Segment {idx} (i2v mode): No dependency on previous segment")
            elif travel_mode == "vace" and independent_segments:
                previous_segment_task_id = None
                dprint(f"[DEBUG_DEPENDENCY_CHAIN] Segment {idx} (vace independent mode): No dependency on previous segment")
            else:
                # VACE MODE (Sequential): Dependent on previous segment
                previous_segment_task_id = actual_segment_db_id_by_index.get(idx - 1) if idx > 0 else None

                # Defensive fallback: if we somehow don't have the previous segment's DB ID yet,
                # try to resolve it from existing_segment_task_ids, and then from DB.
                if idx > 0 and not previous_segment_task_id:
                    fallback_prev = existing_segment_task_ids.get(idx - 1)
                    if fallback_prev:
                        dprint(f"[DEBUG_DEPENDENCY_CHAIN] Fallback resolved previous DB ID for seg {idx-1} from existing_segment_task_ids: {fallback_prev}")
                        actual_segment_db_id_by_index[idx - 1] = fallback_prev
                        previous_segment_task_id = fallback_prev
                    else:
                        try:
                            child_tasks = db_ops.get_orchestrator_child_tasks(orchestrator_task_id_str)
                            for seg in child_tasks.get('segments', []):
                                if seg.get('params', {}).get('segment_index') == idx - 1:
                                    prev_from_db = seg.get('id')
                                    if prev_from_db:
                                        dprint(f"[DEBUG_DEPENDENCY_CHAIN] DB fallback resolved previous DB ID for seg {idx-1}: {prev_from_db}")
                                        actual_segment_db_id_by_index[idx - 1] = prev_from_db
                                        previous_segment_task_id = prev_from_db
                                    break
                        except Exception as e_depdb:
                            dprint(f"[WARN][DEBUG_DEPENDENCY_CHAIN] Could not resolve previous DB ID for seg {idx-1} via DB fallback: {e_depdb}")

            # Skip if this segment already exists
            if idx in existing_segment_indices:
                existing_db_id = existing_segment_task_ids[idx]
                dprint(f"[IDEMPOTENCY] Skipping segment {idx} - already exists with ID {existing_db_id}")
                dprint(f"[DEBUG_DEPENDENCY_CHAIN] Using existing DB ID for segment {idx}: {existing_db_id}; next segment will depend on this")
                continue
                
            segments_created += 1
            
            # Note: segment handler now manages its own output paths using prepare_output_path()

            # Determine frame_overlap_from_previous for current segment `idx`
            current_frame_overlap_from_previous = 0
            if idx == 0 and orchestrator_payload.get("continue_from_video_resolved_path"):
                current_frame_overlap_from_previous = expanded_frame_overlap[0] if expanded_frame_overlap else 0
            elif idx > 0:
                # SM_RESTRUCTURE_FIX_OVERLAP_IDX: Use idx-1 for subsequent segments
                current_frame_overlap_from_previous = expanded_frame_overlap[idx-1] if len(expanded_frame_overlap) > (idx-1) else 0
            
            # VACE refs for this specific segment
            # Ensure vace_refs_instructions_all is a list, default to empty list if None
            vace_refs_safe = vace_refs_instructions_all if vace_refs_instructions_all is not None else []
            vace_refs_for_this_segment = [
                ref_instr for ref_instr in vace_refs_safe
                if ref_instr.get("segment_idx_for_naming") == idx
            ]

            # [DEEP_DEBUG] Log orchestrator payload values BEFORE creating segment payload
            dprint(f"[ORCHESTRATOR_DEBUG] {orchestrator_task_id_str}: CREATING SEGMENT {idx} PAYLOAD")
            
            # Use centralized extraction to get all parameters that should be at top level
            from ..common_utils import extract_orchestrator_parameters
            
            # Extract parameters using centralized function
            task_params_for_extraction = {
                "orchestrator_details": orchestrator_payload
            }
            extracted_params = extract_orchestrator_parameters(
                task_params_for_extraction,
                task_id=f"seg_{idx}_{orchestrator_task_id_str[:8]}",
                dprint=dprint
            )

            # VLM-enhanced prompt retrieval
            # Prompts were pre-generated in batch processing (lines 845-924) for performance.
            # This avoids reloading the VLM model for each segment.
            segment_base_prompt = expanded_base_prompts[idx]
            if idx in vlm_enhanced_prompts:
                segment_base_prompt = vlm_enhanced_prompts[idx]
                dprint(f"[VLM_ENHANCE] Segment {idx}: Using pre-generated enhanced prompt")
            
            # Fallback to orchestrator's base_prompt if segment prompt is empty
            if not segment_base_prompt or not segment_base_prompt.strip():
                segment_base_prompt = orchestrator_payload.get("base_prompt", "")
                if segment_base_prompt:
                    dprint(f"[PROMPT_FALLBACK] Segment {idx}: Using orchestrator base_prompt (segment prompt was empty)")

            # Apply text_before_prompts and text_after_prompts wrapping (after enrichment)
            text_before = orchestrator_payload.get("text_before_prompts", "").strip()
            text_after = orchestrator_payload.get("text_after_prompts", "").strip()

            if text_before or text_after:
                # Build the wrapped prompt, ensuring clean spacing
                parts = []
                if text_before:
                    parts.append(text_before)
                parts.append(segment_base_prompt)
                if text_after:
                    parts.append(text_after)
                segment_base_prompt = " ".join(parts)
                dprint(f"[TEXT_WRAP] Segment {idx}: Applied text_before/after wrapping")

            # Get negative prompt with fallback
            segment_negative_prompt = expanded_negative_prompts[idx] if idx < len(expanded_negative_prompts) else ""
            if not segment_negative_prompt or not segment_negative_prompt.strip():
                segment_negative_prompt = orchestrator_payload.get("negative_prompt", "")
                if segment_negative_prompt:
                    dprint(f"[PROMPT_FALLBACK] Segment {idx}: Using orchestrator negative_prompt (segment negative_prompt was empty)")

            # Calculate segment_frames_target with context frames for segments after the first
            base_segment_frames = expanded_segment_frames[idx]
            if idx > 0 and current_frame_overlap_from_previous > 0:
                # For segments after the first, add context frames from previous segment
                segment_frames_target_with_context = base_segment_frames + current_frame_overlap_from_previous
                dprint(f"[CONTEXT_FRAMES] Segment {idx}: base={base_segment_frames}, context={current_frame_overlap_from_previous}, total={segment_frames_target_with_context}")
            else:
                # First segment doesn't need context frames
                segment_frames_target_with_context = base_segment_frames
                dprint(f"[CONTEXT_FRAMES] Segment {idx}: base={base_segment_frames}, no context needed")

            segment_payload = {
                "orchestrator_task_id_ref": orchestrator_task_id_str,
                "orchestrator_run_id": run_id,
                "project_id": orchestrator_project_id, # Added project_id
                "segment_index": idx,
                "is_first_segment": (idx == 0),
                "is_last_segment": (idx == num_segments - 1),

                "current_run_base_output_dir": str(current_run_output_dir.resolve()), # Base for segment's own output folder creation

                "base_prompt": segment_base_prompt,
                "negative_prompt": segment_negative_prompt,
                "segment_frames_target": segment_frames_target_with_context,
                "frame_overlap_from_previous": current_frame_overlap_from_previous,
                "frame_overlap_with_next": expanded_frame_overlap[idx] if len(expanded_frame_overlap) > idx else 0,
                
                "vace_image_refs_to_prepare_by_worker": vace_refs_for_this_segment, # Already filtered for this segment

                "parsed_resolution_wh": orchestrator_payload["parsed_resolution_wh"],
                "model_name": orchestrator_payload["model_name"],
                "seed_to_use": orchestrator_payload.get("seed_base", 12345),
                "cfg_star_switch": orchestrator_payload.get("cfg_star_switch", 0),
                "cfg_zero_step": orchestrator_payload.get("cfg_zero_step", -1),
                "params_json_str_override": orchestrator_payload.get("params_json_str_override"),
                "fps_helpers": orchestrator_payload.get("fps_helpers", 16),
                "subsequent_starting_strength_adjustment": orchestrator_payload.get("subsequent_starting_strength_adjustment", 0.0),
                "desaturate_subsequent_starting_frames": orchestrator_payload.get("desaturate_subsequent_starting_frames", 0.0),
                "adjust_brightness_subsequent_starting_frames": orchestrator_payload.get("adjust_brightness_subsequent_starting_frames", 0.0),
                "after_first_post_generation_saturation": orchestrator_payload.get("after_first_post_generation_saturation"),
                "after_first_post_generation_brightness": orchestrator_payload.get("after_first_post_generation_brightness"),
                
                "segment_image_download_dir": segment_image_download_dir_str, # Add the download dir path string
                
                "debug_mode_enabled": orchestrator_payload.get("debug_mode_enabled", False),
                "skip_cleanup_enabled": orchestrator_payload.get("skip_cleanup_enabled", False),
                "continue_from_video_resolved_path_for_guide": orchestrator_payload.get("continue_from_video_resolved_path") if idx == 0 else None,
                "consolidated_end_anchor_idx": orchestrator_payload.get("_consolidated_end_anchors", [None] * num_segments)[idx] if orchestrator_payload.get("_consolidated_end_anchors") else None,
                "consolidated_keyframe_positions": orchestrator_payload.get("_consolidated_keyframe_positions", [None] * num_segments)[idx] if orchestrator_payload.get("_consolidated_end_anchors") else None,
                "full_orchestrator_payload": orchestrator_payload, # Ensure full payload is passed to segment
                
                # Structure video guidance parameters
                "structure_video_path": structure_video_path,
                "structure_video_treatment": structure_video_treatment,
                "structure_type": orchestrator_payload.get("structure_type", "flow"),  # Default to flow
                "structure_video_motion_strength": motion_strength if 'motion_strength' in locals() else orchestrator_payload.get("structure_video_motion_strength", 1.0),
                "structure_canny_intensity": canny_intensity if 'canny_intensity' in locals() else orchestrator_payload.get("structure_canny_intensity", 1.0),
                "structure_depth_contrast": depth_contrast if 'depth_contrast' in locals() else orchestrator_payload.get("structure_depth_contrast", 1.0),
                "structure_guidance_video_url": orchestrator_payload.get("structure_guidance_video_url"),  # Pre-warped video URL
                "structure_guidance_frame_offset": segment_flow_offsets[idx] if 'segment_flow_offsets' in locals() else 0,  # Starting frame in structure_guidance video
                
                # Original and trimmed structure videos for reference
                "structure_original_video_url": orchestrator_payload.get("structure_original_video_url"),
                "structure_trimmed_video_url": orchestrator_payload.get("structure_trimmed_video_url"),
            }
            
            # Add extracted parameters at top level for queue processing
            segment_payload.update(extracted_params)
            
            # Log any additional_loras found for debugging
            if extracted_params.get("additional_loras"):
                dprint(f"Orchestrator: Added additional_loras to segment {idx} payload: {extracted_params['additional_loras']}")

            # [DEEP_DEBUG] Log segment payload values AFTER creation
            dprint(f"[ORCHESTRATOR_DEBUG] {orchestrator_task_id_str}: SEGMENT {idx} PAYLOAD CREATED")
            dprint(f"[DEEP_DEBUG] Segment payload keys: {list(segment_payload.keys())}")

            dprint(f"[DEBUG_DEPENDENCY_CHAIN] Creating new segment {idx}, depends_on (prev idx {idx-1}): {previous_segment_task_id}")
            actual_db_row_id = db_ops.add_task_to_db(
                task_payload=segment_payload, 
                task_type_str="travel_segment",
                dependant_on=previous_segment_task_id
            )
            # Record the actual DB ID so subsequent segments depend on the real DB row ID
            actual_segment_db_id_by_index[idx] = actual_db_row_id
            dprint(f"[DEBUG_DEPENDENCY_CHAIN] New segment {idx} created with actual DB ID: {actual_db_row_id}; next segment will depend on this")
            # Post-insert verification of dependency from DB
            try:
                dep_saved = db_ops.get_task_dependency(actual_db_row_id)
                dprint(f"[DEBUG_DEPENDENCY_CHAIN][VERIFY] Segment {idx} saved dependant_on={dep_saved} (expected {previous_segment_task_id})")
            except Exception as e_ver:
                dprint(f"[WARN][DEBUG_DEPENDENCY_CHAIN] Could not verify dependant_on for seg {idx} ({actual_db_row_id}): {e_ver}")
        
        # After loop, enqueue the stitch task (check for idempotency)
        # SKIP if independent segments or I2V mode
        if independent_segments or travel_mode == "i2v":
             dprint(f"[STITCHING] Skipping stitch task creation for independent/i2v mode")
             stitch_created = 0
        else:
             stitch_created = 0
             if not stitch_already_exists:
                final_stitched_video_name = f"travel_final_stitched_{run_id}.mp4"
                # Stitcher saves its final primary output directly under main_output_dir (e.g., ./steerable_motion_output/)
                # NOT under current_run_output_dir (which is .../travel_run_XYZ/)
                # The main_output_dir_base is the one passed to worker.py (e.g. server's ./outputs or steerable_motion's ./steerable_motion_output)
                # The orchestrator_payload["main_output_dir_for_run"] is this main_output_dir_base.
                final_stitched_output_path = Path(orchestrator_payload.get("main_output_dir_for_run", str(main_output_dir_base.resolve()))) / final_stitched_video_name

                stitch_payload = {
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
                
                # Stitch should depend on the last segment's actual DB row ID
                previous_segment_task_id = actual_segment_db_id_by_index.get(num_segments - 1)
                dprint(f"[DEBUG_DEPENDENCY_CHAIN] Creating stitch task, depends_on (last seg idx {num_segments-1}): {previous_segment_task_id}")
                actual_stitch_db_row_id = db_ops.add_task_to_db(
                    task_payload=stitch_payload, 
                    task_type_str="travel_stitch",
                    dependant_on=previous_segment_task_id
                )
                dprint(f"[DEBUG_DEPENDENCY_CHAIN] Stitch task created with actual DB ID: {actual_stitch_db_row_id}")
                # Post-insert verification of dependency from DB
                try:
                    dep_saved = db_ops.get_task_dependency(actual_stitch_db_row_id)
                    dprint(f"[DEBUG_DEPENDENCY_CHAIN][VERIFY] Stitch saved dependant_on={dep_saved} (expected {previous_segment_task_id})")
                except Exception as e_ver2:
                    dprint(f"[WARN][DEBUG_DEPENDENCY_CHAIN] Could not verify dependant_on for stitch ({actual_stitch_db_row_id}): {e_ver2}")
                stitch_created = 1
             else:
                dprint(f"[IDEMPOTENCY] Skipping stitch task creation - already exists with ID {existing_stitch_task_id}")

        generation_success = True
        if segments_created > 0 or stitch_created > 0:
            output_message_for_orchestrator_db = f"Successfully enqueued {segments_created} new segment tasks and {stitch_created} new stitch task for run {run_id}. (Total expected: {num_segments} segments + 1 stitch)"
        else:
            output_message_for_orchestrator_db = f"[IDEMPOTENT] All child tasks already exist for run {run_id}. No new tasks created."
        travel_logger.info(output_message_for_orchestrator_db, task_id=orchestrator_task_id_str)
        log_ram_usage("Orchestrator end (success)", task_id=orchestrator_task_id_str)

    except Exception as e:
        msg = f"Failed during travel orchestration processing: {e}"
        travel_logger.error(msg, task_id=orchestrator_task_id_str)
        travel_logger.debug(traceback.format_exc(), task_id=orchestrator_task_id_str)
        traceback.print_exc()
        generation_success = False
        output_message_for_orchestrator_db = msg
        log_ram_usage("Orchestrator end (error)", task_id=orchestrator_task_id_str)

    return generation_success, output_message_for_orchestrator_db

def _handle_travel_segment_task(task_params_from_db: dict, main_output_dir_base: Path, segment_task_id_str: str, colour_match_videos: bool = False, mask_active_frames: bool = True, *, process_single_task, dprint, task_queue=None):
    travel_logger.essential(f"Starting travel segment task", task_id=segment_task_id_str)
    log_ram_usage("Segment start", task_id=segment_task_id_str)
    dprint(f"_handle_travel_segment_task: Starting for {segment_task_id_str}")
    # Safe logging: Use safe_json_repr to prevent hangs
    dprint(f"Segment task_params_from_db: {safe_json_repr(task_params_from_db)}")
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
            travel_logger.error(msg, task_id=segment_task_id_str)
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
        debug_enabled = segment_params.get("debug_mode_enabled", full_orchestrator_payload.get("debug_mode_enabled", False))

        # Use centralized extraction function for all orchestrator parameters
        from ..common_utils import extract_orchestrator_parameters
        
        # Convert task params to match the expected format for extraction
        task_params_for_extraction = {
            "orchestrator_details": full_orchestrator_payload
        }
        
        extracted_params = extract_orchestrator_parameters(
            task_params_for_extraction, 
            task_id=segment_task_id_str, 
            dprint=dprint
        )

        # Extract travel mode (defaults to vace)
        travel_mode = full_orchestrator_payload.get("model_type", "vace")
        dprint(f"Segment {segment_idx}: Travel mode: {travel_mode}")
        
        # Extract additional_loras from the centralized function result
        additional_loras = extracted_params.get("additional_loras", {})
        if additional_loras:
            dprint(f"Segment {segment_idx}: Found additional_loras via centralized extraction: {additional_loras}")

        current_run_base_output_dir_str = segment_params.get("current_run_base_output_dir")
        from pathlib import Path  # Ensure Path is available in local scope

        if not current_run_base_output_dir_str: # Should be passed by orchestrator/prev segment
            current_run_base_output_dir_str = full_orchestrator_payload.get("main_output_dir_for_run", str(main_output_dir_base.resolve()))
            current_run_base_output_dir_str = str(Path(current_run_base_output_dir_str) / f"travel_run_{orchestrator_run_id}")

        current_run_base_output_dir = Path(current_run_base_output_dir_str)
        # Use the base directory directly without creating segment-specific subdirectories
        segment_processing_dir = current_run_base_output_dir
        segment_processing_dir.mkdir(parents=True, exist_ok=True)

        # ─── Use main processing directory for image downloads (no subfolders) ────────────
        if segment_image_download_dir is None:
            segment_image_download_dir = segment_processing_dir  # Save directly to main output dir
        dprint(f"Segment {segment_idx} (Task {segment_task_id_str}): Processing in {segment_processing_dir.resolve()} | image_download_dir={segment_image_download_dir}")

        # --- Reference Image Determination (Start/End) ---
        # Used for both Color Matching (VACE) and I2V input generation
        start_ref_path, end_ref_path = None, None
        
        input_images_resolved = full_orchestrator_payload.get("input_image_paths_resolved", [])
        is_continuing = full_orchestrator_payload.get("continue_from_video_resolved_path") is not None
        
        if is_continuing:
            if segment_idx == 0:
                continued_video_path = full_orchestrator_payload.get("continue_from_video_resolved_path")
                if continued_video_path and Path(continued_video_path).exists():
                    dprint(f"Seg {segment_idx} Ref: Extracting last frame from {continued_video_path} as start ref.")
                    start_ref_path = sm_extract_last_frame_as_image(continued_video_path, segment_processing_dir, segment_task_id_str)
                if input_images_resolved:
                    end_ref_path = input_images_resolved[0]
            else: # Subsequent segment when continuing
                if len(input_images_resolved) > segment_idx:
                    start_ref_path = input_images_resolved[segment_idx - 1]
                    end_ref_path = input_images_resolved[segment_idx]
        else: # From scratch
            if len(input_images_resolved) > segment_idx + 1:
                start_ref_path = input_images_resolved[segment_idx]
                end_ref_path = input_images_resolved[segment_idx + 1]
        
        # Download images if they are URLs so they exist locally
        if start_ref_path:
            start_ref_path = sm_download_image_if_url(
                start_ref_path, 
                segment_processing_dir,  # Save directly to main output dir, no subfolder
                segment_task_id_str, 
                debug_mode=debug_enabled,
                descriptive_name=f"seg{segment_idx:02d}_start_ref"
            )
        if end_ref_path:
            end_ref_path = sm_download_image_if_url(
                end_ref_path, 
                segment_processing_dir,  # Save directly to main output dir, no subfolder
                segment_task_id_str,
                debug_mode=debug_enabled,
                descriptive_name=f"seg{segment_idx:02d}_end_ref"
            )

        dprint(f"Seg {segment_idx} Refs: Start='{start_ref_path}', End='{end_ref_path}'")
        
        # [DEBUG_REF_PATH] Trace start_ref_path
        if start_ref_path is None:
             print(f"[DEBUG_REF_PATH] start_ref_path is None after init logic. len(input_images_resolved)={len(input_images_resolved)}, segment_idx={segment_idx}, is_continuing={is_continuing}")
             # Fallback for segment 0 if not continuing
             if segment_idx == 0 and not is_continuing and len(input_images_resolved) > 0:
                 print(f"[DEBUG_REF_PATH] Attempting fallback assignment from input_images_resolved[0]: {input_images_resolved[0]}")
                 start_ref_path_raw = input_images_resolved[0]
                 start_ref_path = sm_download_image_if_url(
                    start_ref_path_raw, 
                    segment_processing_dir, 
                    segment_task_id_str, 
                    debug_mode=debug_enabled,
                    descriptive_name=f"seg{segment_idx:02d}_start_ref_fallback"
                )
                 print(f"[DEBUG_REF_PATH] Fallback result: {start_ref_path}")

        
        # Assign for backward compatibility / specific use
        start_ref_path_for_cm = start_ref_path
        end_ref_path_for_cm = end_ref_path
        # --- End Reference Image Determination ---

        # --- Prepare VACE Refs for this Segment (moved to worker) ---
        actual_vace_image_ref_paths_for_wgp = []
        # Get the list of VACE ref instructions from the full orchestrator payload
        vace_ref_instructions_from_orchestrator = full_orchestrator_payload.get("", [])
        
        # Ensure vace_ref_instructions_from_orchestrator is a list, default to empty list if None
        if vace_ref_instructions_from_orchestrator is None:
            vace_ref_instructions_from_orchestrator = []
        
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
            # If resolution is provided as string (e.g., "512x512"), convert it to tuple[int, int]
            if isinstance(current_parsed_res_wh, str):
                try:
                    parsed_tuple = sm_parse_resolution(current_parsed_res_wh)
                    if parsed_tuple is not None:
                        current_parsed_res_wh = parsed_tuple
                    else:
                        dprint(f"[WARNING] Segment {segment_idx}: Failed to parse resolution string '{current_parsed_res_wh}'. Proceeding with original value.")
                except Exception as e_par:
                    dprint(f"[WARNING] Segment {segment_idx}: Error parsing resolution '{current_parsed_res_wh}': {e_par}. Proceeding with string value (may cause errors).")
            if not current_parsed_res_wh:
                # Fallback or error if resolution not found; for now, dprint and proceed (helper might handle None resolution)
                dprint(f"[WARNING] Segment {segment_idx}: parsed_resolution_wh not found in full_orchestrator_payload. VACE refs might not be resized correctly.")

            for ref_instr in relevant_vace_instructions:
                # Pass segment_image_download_dir to _prepare_vace_ref_for_segment_worker
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

        # Ensure parsed_res_wh is a tuple of integers with model grid snapping
        parsed_res_wh_str = full_orchestrator_payload["parsed_resolution_wh"]
        try:
            parsed_res_raw = sm_parse_resolution(parsed_res_wh_str)
            if parsed_res_raw is None:
                raise ValueError(f"sm_parse_resolution returned None for input: {parsed_res_wh_str}")
            parsed_res_wh = snap_resolution_to_model_grid(parsed_res_raw)
        except Exception as e_parse_res:
            msg = f"Seg {segment_idx}: Invalid format or error parsing parsed_resolution_wh '{parsed_res_wh_str}': {e_parse_res}"
            print(f"[ERROR Task {segment_task_id_str}]: {msg}"); return False, msg
        dprint(f"Segment {segment_idx}: Parsed resolution (w,h): {parsed_res_wh}")

        # --- Single Image Journey Detection ---
        input_images_for_cm_check = full_orchestrator_payload.get("input_image_paths_resolved", [])
        is_single_image_journey = (
            len(input_images_for_cm_check) == 1
            and full_orchestrator_payload.get("continue_from_video_resolved_path") is None
            and segment_params.get("is_first_segment")
            and segment_params.get("is_last_segment")
        )
        if is_single_image_journey:
            dprint(f"Seg {segment_idx}: Detected a single-image journey. Adjusting guide and mask generation.")

        # Calculate total frames for this segment once and reuse
        base_duration = segment_params.get("segment_frames_target", full_orchestrator_payload["segment_frames_expanded"][segment_idx])
        frame_overlap_from_previous = segment_params.get("frame_overlap_from_previous", 0)

        # For segments after the first, add context frames from previous segment
        # This allows the model to see the end of the previous segment during generation
        # providing true temporal continuity rather than post-generation crossfading
        if frame_overlap_from_previous > 0:
            total_frames_for_segment = base_duration + frame_overlap_from_previous
        else:
            total_frames_for_segment = base_duration

        dprint(f"[SEGMENT_DEBUG] Segment {segment_idx} (Task {segment_task_id_str}): FRAME ANALYSIS")
        dprint(f"[SEGMENT_DEBUG]   base_duration (segment_frames_target): {base_duration}")
        dprint(f"[SEGMENT_DEBUG]   frame_overlap_from_previous: {frame_overlap_from_previous}")
        dprint(f"[SEGMENT_DEBUG]   total_frames_for_segment: {total_frames_for_segment}")
        dprint(f"[SEGMENT_DEBUG]   is_first_segment: {segment_params.get('is_first_segment', False)}")
        dprint(f"[SEGMENT_DEBUG]   is_last_segment: {segment_params.get('is_last_segment', False)}")

        fps_helpers = full_orchestrator_payload.get("fps_helpers", 16)

        # Define gray_frame_bgr here for use in subsequent segment strength adjustment
        gray_frame_bgr = sm_create_color_frame(parsed_res_wh, (128, 128, 128))

        # Hardcoded fade parameters (duration_factor=0.0 means no fading)
        fi_low, fi_high, fi_curve, fi_factor = 0.0, 1.0, "ease_in_out", 0.0
        fo_low, fo_high, fo_curve, fo_factor = 0.0, 1.0, "ease_in_out", 0.0

        if is_first_new_segment_after_continue:
            path_to_previous_segment_video_output_for_guide = full_orchestrator_payload.get("continue_from_video_resolved_path")
            if not path_to_previous_segment_video_output_for_guide or not Path(path_to_previous_segment_video_output_for_guide).exists():
                msg = f"Seg {segment_idx}: Continue video path {path_to_previous_segment_video_output_for_guide} invalid."
                print(f"[ERROR Task {segment_task_id_str}]: {msg}"); return False, msg
        elif is_subsequent_segment:
            # Get predecessor task ID and its output location in a single call using Edge Function
            task_dependency_id, raw_path_from_db = db_ops.get_predecessor_output_via_edge_function(segment_task_id_str)
            
            if task_dependency_id and raw_path_from_db:
                dprint(f"Seg {segment_idx}: Task {segment_task_id_str} depends on {task_dependency_id} with output: {raw_path_from_db}")
                # Path from DB is absolute (Supabase URL or absolute path)
                path_to_previous_segment_video_output_for_guide = raw_path_from_db
            elif task_dependency_id and not raw_path_from_db:
                dprint(f"Seg {segment_idx}: Found dependency task {task_dependency_id} but no output_location available.")
                path_to_previous_segment_video_output_for_guide = None
            else:
                dprint(f"Seg {segment_idx}: No dependency found for task {segment_task_id_str}. Cannot create guide video based on predecessor.")
                path_to_previous_segment_video_output_for_guide = None
 
            # --- New: Handle Supabase public URLs by downloading them locally for guide processing ---
            if path_to_previous_segment_video_output_for_guide and path_to_previous_segment_video_output_for_guide.startswith("http"):
                try:
                    dprint(f"Seg {segment_idx}: Detected remote URL for previous segment: {path_to_previous_segment_video_output_for_guide}. Downloading...")
                    # Reuse download_file utility from common_utils
                    from ..common_utils import download_file as sm_download_file, sm_get_unique_target_path
                    remote_url = path_to_previous_segment_video_output_for_guide
                    local_filename = Path(remote_url).name
                    # Store under segment_processing_dir to keep things tidy
                    local_download_path = segment_processing_dir / f"prev_{segment_idx:02d}_{local_filename}"
                    # Ensure directory exists
                    segment_processing_dir.mkdir(parents=True, exist_ok=True)
                    # Perform download if file not already present
                    if not local_download_path.exists():
                        sm_download_file(remote_url, segment_processing_dir, local_download_path.name)
                        dprint(f"Seg {segment_idx}: Downloaded previous segment video to {local_download_path}")
                    else:
                        dprint(f"Seg {segment_idx}: Local copy of previous segment video already exists at {local_download_path}")
                    path_to_previous_segment_video_output_for_guide = str(local_download_path.resolve())
                except Exception as e_dl_prev:
                    dprint(f"[WARNING] Seg {segment_idx}: Failed to download remote previous segment video: {e_dl_prev}")
                    # Leave path unchanged – will trigger the existing invalid path error below

            if not path_to_previous_segment_video_output_for_guide or not Path(path_to_previous_segment_video_output_for_guide).exists():
                error_detail_path = raw_path_from_db if 'raw_path_from_db' in locals() and raw_path_from_db else path_to_previous_segment_video_output_for_guide
                msg = f"Seg {segment_idx}: Prev segment output for guide invalid/not found. Expected from prev task output. Path: {error_detail_path}"
                print(f"[ERROR Task {segment_task_id_str}]: {msg}"); return False, msg
        
        # ------------------------------------------------------------------
        #  Show-input-images (banner) – determine start/end images now so the
        #  information can be propagated downstream to the chaining stage.
        # ------------------------------------------------------------------
        show_input_images_enabled = bool(full_orchestrator_payload.get("show_input_images", False))
        start_image_for_banner = None
        end_image_for_banner = None
        if show_input_images_enabled:
            try:
                input_images_resolved_original = full_orchestrator_payload["input_image_paths_resolved"]
                # For banner overlay, always show the first and last images of the entire journey
                # This provides consistent context across all segments
                if len(input_images_resolved_original) > 0:
                    start_image_for_banner = input_images_resolved_original[0]  # Always first image

                if len(input_images_resolved_original) > 1:
                    end_image_for_banner = input_images_resolved_original[-1]  # Always last image
                elif len(input_images_resolved_original) == 1:
                    # Single image journey - use the same image for both
                    end_image_for_banner = input_images_resolved_original[0]

                # Ensure both banner images are local paths (download if URL)
                if start_image_for_banner:
                    start_image_for_banner = sm_download_image_if_url(
                        start_image_for_banner,
                        segment_processing_dir,
                        segment_task_id_str,
                        debug_mode=debug_enabled,
                        descriptive_name="journey_start_image"
                    )
                if end_image_for_banner:
                    end_image_for_banner = sm_download_image_if_url(
                        end_image_for_banner,
                        segment_processing_dir,
                        segment_task_id_str,
                        debug_mode=debug_enabled,
                        descriptive_name="journey_end_image"
                    )

            except Exception as e_banner_sel:
                dprint(f"Seg {segment_idx}: Error selecting banner images for show_input_images: {e_banner_sel}")
        # ------------------------------------------------------------------

        # Use the shared TravelSegmentProcessor to eliminate code duplication
        # Initialize outputs
        actual_guide_video_path_for_wgp = None
        mask_video_path_for_wgp = None
        video_prompt_type_str = None
        is_vace_model = False

        if travel_mode == "vace":
            try:
                from ..travel_segment_processor import TravelSegmentProcessor, TravelSegmentContext
                
                # Create context for the shared processor
                processor_context = TravelSegmentContext(
                    task_id=segment_task_id_str,
                    segment_idx=segment_idx,
                    model_name=full_orchestrator_payload["model_name"],
                    total_frames_for_segment=total_frames_for_segment,
                    parsed_res_wh=parsed_res_wh,
                    segment_processing_dir=segment_processing_dir,
                    full_orchestrator_payload=full_orchestrator_payload,
                    segment_params=segment_params,
                    mask_active_frames=mask_active_frames,
                    debug_enabled=debug_enabled,
                    dprint=dprint
                )
                
                # Create and use the shared processor
                processor = TravelSegmentProcessor(processor_context)
                segment_outputs = processor.process_segment()
                
                # Extract outputs
                actual_guide_video_path_for_wgp = Path(segment_outputs["video_guide"]) if segment_outputs["video_guide"] else None
                mask_video_path_for_wgp = Path(segment_outputs["video_mask"]) if segment_outputs["video_mask"] else None
                video_prompt_type_str = segment_outputs["video_prompt_type"]
                
                # Get VACE model detection result from processor
                is_vace_model = processor.is_vace_model
                
                dprint(f"[SHARED_PROCESSOR] Seg {segment_idx}: Guide video: {actual_guide_video_path_for_wgp}")
                dprint(f"[SHARED_PROCESSOR] Seg {segment_idx}: Mask video: {mask_video_path_for_wgp}")
                dprint(f"[SHARED_PROCESSOR] Seg {segment_idx}: Video prompt type: {video_prompt_type_str}")
                dprint(f"[SHARED_PROCESSOR] Seg {segment_idx}: Is VACE model: {is_vace_model}")
                
            except Exception as e_shared_processor:
                dprint(f"[ERROR] Seg {segment_idx}: Shared processor failed: {e_shared_processor}")
                import traceback
                traceback.print_exc()
                return False, f"Shared processor failed: {e_shared_processor}", None
        
        elif travel_mode == "i2v":
            dprint(f"[I2V_MODE] Seg {segment_idx}: Using image-to-video generation mode")
            # For I2V, we need start and end images
            if not start_ref_path or not end_ref_path:
                # Special case: first segment from scratch might only have start image
                if is_first_segment_from_scratch and start_ref_path:
                    dprint(f"[I2V_MODE] First segment from scratch: Only start image provided")
                elif is_first_segment_from_scratch and not start_ref_path:
                     # Allow text-to-video for first segment if no images
                     dprint(f"[I2V_MODE] First segment from scratch: No images provided, falling back to text-to-video behavior")
                else:
                    msg = f"Seg {segment_idx}: I2V mode requires start and end reference images. Start: {start_ref_path}, End: {end_ref_path}"
                    travel_logger.error(msg, task_id=segment_task_id_str)
                    return False, msg
            
            # I2V doesn't use guide videos or VACE logic
            actual_guide_video_path_for_wgp = None
            mask_video_path_for_wgp = None
            video_prompt_type_str = None 
            is_vace_model = False

        # [GUIDE_CONTENT_DEBUG] Log guide and mask video paths for content verification  
        dprint(f"[GUIDE_CONTENT_DEBUG] Seg {segment_idx}: Video guide path: {actual_guide_video_path_for_wgp}")
        dprint(f"[GUIDE_CONTENT_DEBUG] Seg {segment_idx}: Video mask path: {mask_video_path_for_wgp}")
        dprint(f"[GUIDE_CONTENT_DEBUG] Seg {segment_idx}: Empty prompt may cause generic generation - consider adding descriptive prompt")
        
        # --- Invoke WGP Generation directly ---
        # Check guide video necessity based on mode
        if travel_mode == "vace" and actual_guide_video_path_for_wgp is None and not is_first_segment_from_scratch:
            # If guide creation failed AND it was essential (i.e., for any segment except the very first one from scratch)
            msg = f"Essential guide video failed to generate. Cannot proceed with WGP processing."
            travel_logger.error(msg, task_id=segment_task_id_str)
            return False, msg
            
        final_frames_for_wgp_generation = total_frames_for_segment
        current_wgp_engine = "wgp" # Defaulting to WGP for travel segments
        
        travel_logger.info(f"WGP generation: {final_frames_for_wgp_generation} frames, res={parsed_res_wh}, fps={fps_helpers}, model={full_orchestrator_payload['model_name']}", task_id=segment_task_id_str)
        dprint(f"[WGP_DEBUG] Segment {segment_idx}: GENERATION PARAMETERS")
        dprint(f"[WGP_DEBUG]   final_frames_for_wgp_generation: {final_frames_for_wgp_generation}")
        dprint(f"[WGP_DEBUG]   parsed_res_wh: {parsed_res_wh}")
        dprint(f"[WGP_DEBUG]   fps_helpers: {fps_helpers}")
        dprint(f"[WGP_DEBUG]   model_name: {full_orchestrator_payload['model_name']}")
        dprint(f"Task {segment_task_id_str}: Requesting WGP generation with {final_frames_for_wgp_generation} frames.")

        if final_frames_for_wgp_generation <= 0:
            msg = f"Calculated WGP frames {final_frames_for_wgp_generation}. Cannot generate. Check segment_frames_target and overlap."
            travel_logger.error(msg, task_id=segment_task_id_str)
            return False, msg

        # The WGP task will run with a unique ID, but it's processed in-line now
        wgp_inline_task_id = sm_generate_unique_task_id(f"wgp_inline_{segment_task_id_str[:8]}_")

        # Define the absolute final output path for the WGP generation by process_single_task.
        # wgp_final_output_path_for_this_segment is a suggestion for process_single_task.
        # With Supabase, process_single_task will use this path (or its default construction) and return an absolute path or URL.
        wgp_video_filename = f"{orchestrator_run_id}_seg{segment_idx:02d}_output.mp4"
        wgp_final_output_path_for_this_segment = segment_processing_dir / wgp_video_filename 
        
        safe_vace_image_ref_paths_for_wgp = [str(p.resolve()) if p else None for p in actual_vace_image_ref_paths_for_wgp]
        safe_vace_image_ref_paths_for_wgp = [p for p in safe_vace_image_ref_paths_for_wgp if p is not None]
        
        # If no image refs, pass None instead of empty list to avoid WGP VAE encoder issues
        if not safe_vace_image_ref_paths_for_wgp:
            safe_vace_image_ref_paths_for_wgp = None

        current_segment_base_prompt = segment_params.get("base_prompt", " ")
        prompt_for_wgp = ensure_valid_prompt(current_segment_base_prompt)

        # Apply text_before_prompts and text_after_prompts as fallback (if not already applied at orchestrator level)
        text_before = full_orchestrator_payload.get("text_before_prompts", "").strip()
        text_after = full_orchestrator_payload.get("text_after_prompts", "").strip()

        # Only apply if the prompt doesn't already contain the wrapped text (avoid double-wrapping)
        if (text_before or text_after) and not (text_before and prompt_for_wgp.startswith(text_before)):
            parts = []
            if text_before:
                parts.append(text_before)
            parts.append(prompt_for_wgp)
            if text_after:
                parts.append(text_after)
            prompt_for_wgp = " ".join(parts)
            dprint(f"Seg {segment_idx}: Applied text_before/after wrapping to prompt")

        negative_prompt_for_wgp = ensure_valid_negative_prompt(segment_params.get("negative_prompt", " "))

        dprint(f"Seg {segment_idx} (Task {segment_task_id_str}): Effective prompt for WGP: '{prompt_for_wgp}'")

        # Get model name for JSON config loading
        model_name = full_orchestrator_payload["model_name"]
        
        # Load model defaults from JSON config BEFORE building wgp_payload
        model_defaults_from_config = {}
        try:
            import sys
            import os
            wan_dir = Path(__file__).parent.parent.parent / "Wan2GP"
            if str(wan_dir) not in sys.path:
                sys.path.insert(0, str(wan_dir))

            # Protect sys.argv before importing wgp
            _saved_argv = sys.argv[:]
            try:
                sys.argv = ["travel_between_images.py"]
                import wgp
            finally:
                sys.argv = _saved_argv

            model_defaults_from_config = wgp.get_default_settings(model_name)
            dprint(f"[MODEL_CONFIG_DEBUG] Segment {segment_idx}: Loaded model defaults for '{model_name}': {model_defaults_from_config}")
        except Exception as e:
            dprint(f"[MODEL_CONFIG_DEBUG] Segment {segment_idx}: Warning - could not load model defaults for '{model_name}': {e}")

        # [DEBUG_REF_PATH] Check before payload
        print(f"[DEBUG_REF_PATH] Pre-payload check: start_ref_path={start_ref_path}, travel_mode={travel_mode}")

        wgp_payload = {
            "task_id": wgp_inline_task_id, # ID for this specific WGP generation operation
            "model": full_orchestrator_payload["model_name"],
            "prompt": prompt_for_wgp, # Use the processed prompt_for_wgp
            "negative_prompt": segment_params["negative_prompt"],
            "resolution": f"{parsed_res_wh[0]}x{parsed_res_wh[1]}", # Use parsed tuple here
            "frames": final_frames_for_wgp_generation,
            "seed": segment_params["seed_to_use"],
            # output_path for process_single_task: 
            # This suggested path (or process_single_task's default) is used, and an absolute path or URL is returned.
            "output_path": str(wgp_final_output_path_for_this_segment.resolve()), 
            
            # VACE specific inputs (pass None if not VACE)
            "video_guide_path": str(actual_guide_video_path_for_wgp.resolve()) if actual_guide_video_path_for_wgp and actual_guide_video_path_for_wgp.exists() else None,
            "image_refs_paths": safe_vace_image_ref_paths_for_wgp,
            
            # Always pass start/end images if available (required for I2V models even in VACE/Control mode)
            "image_start": str(Path(start_ref_path).resolve()) if start_ref_path else None,
            "image_end": str(Path(end_ref_path).resolve()) if end_ref_path else None,
            
            "cfg_star_switch": full_orchestrator_payload.get("cfg_star_switch", 0),
            "cfg_zero_step": full_orchestrator_payload.get("cfg_zero_step", -1),
            
            # Propagate video_prompt_type so VACE model correctly interprets guide and mask inputs
            "video_prompt_type": video_prompt_type_str,
            # Attach mask video if available
            **({"video_mask": str(mask_video_path_for_wgp.resolve())} if mask_video_path_for_wgp else {}),
        }
        
        # Add model config defaults to wgp_payload so they're available in parameter precedence chain
        # Note: HeadlessTaskQueue will properly handle task parameter overrides
        if model_defaults_from_config:
            for param in ["guidance_scale", "guidance2_scale", "guidance3_scale", "guidance_phases",
                         "flow_shift", "num_inference_steps", "switch_threshold", "switch_threshold2",
                         "model_switch_phase", "sample_solver", "video_prompt_type"]:
                if param in model_defaults_from_config:
                    wgp_payload[param] = model_defaults_from_config[param]
                    dprint(f"[MODEL_CONFIG_DEBUG] Segment {segment_idx}: Added {param}={model_defaults_from_config[param]} to wgp_payload from model config")
        if additional_loras:
            # Pass additional_loras in dict format - HeadlessTaskQueue handles conversion centrally
            wgp_payload["additional_loras"] = additional_loras
            dprint(f"Seg {segment_idx}: Added {len(additional_loras)} additional LoRAs to payload")

        if full_orchestrator_payload.get("params_json_str_override"):
            try:
                additional_p = json.loads(full_orchestrator_payload["params_json_str_override"])
                # Ensure override cannot change key params that indirectly control output length or resolution
                additional_p.pop("frames", None); additional_p.pop("video_length", None)
                additional_p.pop("steps", None); additional_p.pop("num_inference_steps", None)
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
            "show_input_images": show_input_images_enabled,
            "start_image_path": start_image_for_banner,
            "end_image_path": end_image_for_banner,
        }

        dprint(f"Seg {segment_idx} (Task {segment_task_id_str}): Invoking WGP generation via centralized wrapper (task_id for WGP op: {wgp_inline_task_id})")
        
        # Log VACE control weights if using VACE
        if is_vace_model:
            control_weight = full_orchestrator_payload.get("control_net_weight", 1.0)
            control_weight2 = full_orchestrator_payload.get("control_net_weight2", 1.0)
            dprint(f"[VACEWeights] Seg {segment_idx}: control_net_weight={control_weight}, control_net_weight2={control_weight2}")
        
        # Process additional LoRAs using shared function
        processed_additional_loras = {}
        if additional_loras:
            dprint(f"Seg {segment_idx}: Processing additional LoRAs using shared function")
            # Since legacy system is removed, use task_queue system for LoRA processing
            if task_queue is not None:
                # The new queue system handles LoRAs internally, so just pass them as-is
                processed_additional_loras = additional_loras
                dprint(f"Seg {segment_idx}: Using task_queue system for LoRA processing: {len(processed_additional_loras)} LoRAs")
            else:
                # Fallback: pass LoRAs as-is if no task_queue
                processed_additional_loras = additional_loras
                dprint(f"Seg {segment_idx}: Fallback LoRA processing: {len(processed_additional_loras)} LoRAs")

        # ------------------------------------------------------------------
        # Ensure sensible defaults for critical generation params using shared utilities
        # ------------------------------------------------------------------
        import sys
        # Path already imported at top of function
        # LoRA optimization removed - all parameters from model JSON config

        # DEPRECATED: Legacy task queue system code - travel segments now processed via direct queue integration
        # Travel segments are now routed through worker.py's _handle_travel_segment_via_queue function
        # which eliminates the blocking wait pattern and provides better model persistence.
        generation_success = False
        wgp_output_path_or_msg = (
            f"DEPRECATED: Travel segment {segment_idx} should be processed via direct queue integration. "
            f"This indicates that worker.py is incorrectly routing travel_segment tasks to the legacy handler "
            f"instead of using _handle_travel_segment_via_queue. Check task routing configuration."
        )

        travel_logger.debug(f"WGP generation result: success={generation_success}, output={wgp_output_path_or_msg}", task_id=segment_task_id_str)
        dprint(f"[WGP_DEBUG] Segment {segment_idx}: GENERATION RESULT")
        dprint(f"[WGP_DEBUG]   generation_success: {generation_success}")
        dprint(f"[WGP_DEBUG]   wgp_output_path_or_msg: {wgp_output_path_or_msg}")
        
        # Analyze the WGP output if successful
        if generation_success and wgp_output_path_or_msg:
            wgp_debug_info = debug_video_analysis(wgp_output_path_or_msg, f"WGP_RAW_OUTPUT_Seg{segment_idx}", segment_task_id_str)
            print(f"[WGP_DEBUG]   Expected frames: {final_frames_for_wgp_generation}")
            print(f"[WGP_DEBUG]   Actual frames: {wgp_debug_info.get('frame_count', 'ERROR')}")
            if wgp_debug_info.get('frame_count') != final_frames_for_wgp_generation:
                print(f"[WGP_DEBUG]   ⚠️  FRAME COUNT MISMATCH! Expected {final_frames_for_wgp_generation}, got {wgp_debug_info.get('frame_count')}")

        if generation_success:
            # Apply post-processing chain (saturation, brightness, color matching)
            chain_success, chain_message, final_chained_path = _handle_travel_chaining_after_wgp(
                wgp_task_params=wgp_payload,
                actual_wgp_output_video_path=wgp_output_path_or_msg,
                image_download_dir=segment_image_download_dir,
                dprint=dprint
            )
            
            if chain_success and final_chained_path:
                final_segment_video_output_path_str = final_chained_path
                output_message_for_segment_task = f"Segment {segment_idx} processing (WGP generation & chaining) completed. Final output: {final_segment_video_output_path_str}"
                travel_logger.info(f"Segment processing completed successfully", task_id=segment_task_id_str)
                
                # Analyze final chained output
                final_debug_info = debug_video_analysis(final_chained_path, f"FINAL_CHAINED_Seg{segment_idx}", segment_task_id_str)
                dprint(f"[CHAIN_DEBUG] Segment {segment_idx}: FINAL CHAINED OUTPUT ANALYSIS")
                dprint(f"[CHAIN_DEBUG]   Expected frames: {final_frames_for_wgp_generation}")
                dprint(f"[CHAIN_DEBUG]   Final frames: {final_debug_info.get('frame_count', 'ERROR')}")
                if final_debug_info.get('frame_count') != final_frames_for_wgp_generation:
                    dprint(f"[CHAIN_DEBUG]   ⚠️  CHAINING CHANGED FRAME COUNT! Expected {final_frames_for_wgp_generation}, got {final_debug_info.get('frame_count')}")
            else:
                # Use raw WGP output if chaining failed
                final_segment_video_output_path_str = wgp_output_path_or_msg
                output_message_for_segment_task = f"Segment {segment_idx} WGP completed but chaining failed: {chain_message}. Using raw output: {final_segment_video_output_path_str}"
                travel_logger.warning(f"Chaining failed: {chain_message}. Using raw WGP output", task_id=segment_task_id_str)
                
                # Analyze raw output being used as final
                if wgp_output_path_or_msg:
                    raw_debug_info = debug_video_analysis(wgp_output_path_or_msg, f"RAW_AS_FINAL_Seg{segment_idx}", segment_task_id_str)
            
            dprint(f"Seg {segment_idx} (Task {segment_task_id_str}): {output_message_for_segment_task}")
        else:
            # wgp_output_path_or_msg contains the error message if generation_success is False
            final_segment_video_output_path_str = None 
            output_message_for_segment_task = f"Segment {segment_idx} processing (WGP generation) failed. Error: {wgp_output_path_or_msg}"
            travel_logger.error(f"WGP generation failed: {wgp_output_path_or_msg}", task_id=segment_task_id_str)
            
            # Notify orchestrator of segment failure
            try:
                db_ops.update_task_status(
                    orchestrator_task_id_ref,
                    db_ops.STATUS_FAILED,
                    output_message_for_segment_task[:500]  # Truncate to avoid DB overflow
                )
                dprint(f"Segment {segment_idx}: Marked orchestrator task {orchestrator_task_id_ref} as FAILED due to WGP generation failure")
            except Exception as e_orch:
                dprint(f"Segment {segment_idx}: Warning - could not update orchestrator status: {e_orch}")
        
        # The old polling logic is no longer needed as process_single_task is synchronous here.

        # The return value final_segment_video_output_path_str (if success) is the one that
        # process_single_task itself would have set as 'output_location' for the WGP task.
        # Now, it becomes the output_location for the parent travel_segment task.
        log_ram_usage("Segment end (success)", task_id=segment_task_id_str)
        return generation_success, final_segment_video_output_path_str if generation_success else output_message_for_segment_task

    except Exception as e:
        travel_logger.error(f"Unexpected error during segment processing: {e}", task_id=segment_task_id_str)
        travel_logger.debug(traceback.format_exc(), task_id=segment_task_id_str)
        traceback.print_exc()
        
        # Notify orchestrator of segment failure
        if 'orchestrator_task_id_ref' in locals() and orchestrator_task_id_ref:
            try:
                error_msg = f"Segment {segment_idx if 'segment_idx' in locals() else 'unknown'} failed: {str(e)[:200]}"
                db_ops.update_task_status(
                    orchestrator_task_id_ref,
                    db_ops.STATUS_FAILED,
                    error_msg
                )
                dprint(f"Segment: Marked orchestrator task {orchestrator_task_id_ref} as FAILED due to exception")
            except Exception as e_orch:
                dprint(f"Segment: Warning - could not update orchestrator status: {e_orch}")

        # return False, f"Unexpected error: {str(e)[:200]}"
        log_ram_usage("Segment end (error)", task_id=segment_task_id_str)
        return False, f"Segment {segment_idx if 'segment_idx' in locals() else 'unknown'} failed: {str(e)[:200]}"

# --- SM_RESTRUCTURE: New function to handle chaining after WGP/Comfy sub-task ---
def _handle_travel_chaining_after_wgp(wgp_task_params: dict, actual_wgp_output_video_path: str | None, image_download_dir: Path | str | None = None, *, dprint) -> tuple[bool, str, str | None]:
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
    # Path is an absolute path or URL from Supabase.
    if not actual_wgp_output_video_path: # Check if it's None or empty string
        return False, f"Task {wgp_task_id}: WGP output video path is None or empty. Cannot chain.", None

    # This variable will track the absolute path of the video as it gets processed.
    video_to_process_abs_path: Path
    # This will hold the path to be stored in the DB (absolute path or URL)
    final_video_path_for_db = actual_wgp_output_video_path

    # Path is already absolute (Supabase URL or absolute path)
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

        # --- Always move WGP output to proper location first ---
        # Use consistent UUID-based naming and MOVE (not copy) to avoid duplicates
        timestamp_short = datetime.now().strftime("%H%M%S")
        unique_suffix = uuid.uuid4().hex[:6]
        moved_filename = f"seg{segment_idx_completed:02d}_output_{timestamp_short}_{unique_suffix}{video_to_process_abs_path.suffix}"
        
        # Use segment processing dir directly without task_id prefixing since UUID guarantees uniqueness
        moved_video_abs_path = Path(segment_processing_dir_for_saturation_str) / moved_filename
        moved_video_abs_path.parent.mkdir(parents=True, exist_ok=True)
        
        # MOVE (not copy) the WGP output to avoid creating duplicates
        try:
            # Ensure encoder has finished writing the source file
            sm_wait_for_file_stable(video_to_process_abs_path, checks=3, interval=1.0, dprint=dprint)

            shutil.move(str(video_to_process_abs_path), str(moved_video_abs_path))
            print(f"[CHAIN_DEBUG] Moved WGP output from {video_to_process_abs_path} to {moved_video_abs_path}")
            debug_video_analysis(moved_video_abs_path, f"MOVED_WGP_OUTPUT_Seg{segment_idx_completed}", wgp_task_id)
            dprint(f"Chain (Seg {segment_idx_completed}): Moved WGP output from {video_to_process_abs_path} to {moved_video_abs_path}")
            
            # Update paths for further processing
            video_to_process_abs_path = moved_video_abs_path
            final_video_path_for_db = str(moved_video_abs_path)  # Use absolute path as DB path
            
            # No cleanup needed since we moved (not copied) the file
            dprint(f"Chain (Seg {segment_idx_completed}): WGP output successfully moved to final location")
                    
        except Exception as e_move:
            dprint(f"Chain (Seg {segment_idx_completed}): Warning - could not move WGP output to proper location: {e_move}. Using original path.")
            # If move failed, keep original paths for further processing
            final_video_path_for_db = str(video_to_process_abs_path)

        # --- Post-generation Processing Chain ---
        # Saturation and Brightness are only applied to segments AFTER the first one.
        if is_subsequent_segment_val or is_first_new_segment_after_continue:

            # --- 1. Saturation ---
            sat_level = full_orchestrator_payload.get("after_first_post_generation_saturation")
            if sat_level is not None and isinstance(sat_level, (float, int)) and sat_level >= 0.0 and abs(sat_level - 1.0) > 1e-6:
                dprint(f"Chain (Seg {segment_idx_completed}): Applying post-gen saturation {sat_level} to {video_to_process_abs_path}")

                sat_filename = f"s{segment_idx_completed}_sat_{sat_level:.2f}{video_to_process_abs_path.suffix}"
                saturated_video_output_abs_path, new_db_path = prepare_output_path(
                    task_id=wgp_task_id,
                    filename=sat_filename,
                    main_output_dir_base=Path(segment_processing_dir_for_saturation_str)
                )
                
                if sm_apply_saturation_to_video_ffmpeg(str(video_to_process_abs_path), saturated_video_output_abs_path, sat_level):
                    print(f"[CHAIN_DEBUG] Saturation applied successfully to segment {segment_idx_completed}")
                    debug_video_analysis(saturated_video_output_abs_path, f"SATURATED_Seg{segment_idx_completed}", wgp_task_id)
                    dprint(f"Chain (Seg {segment_idx_completed}): Saturation successful. New path: {new_db_path}")
                    _cleanup_intermediate_video(full_orchestrator_payload, video_to_process_abs_path, segment_idx_completed, "raw", dprint)
                    
                    video_to_process_abs_path = saturated_video_output_abs_path
                    final_video_path_for_db = new_db_path
                else:
                    print(f"[CHAIN_DEBUG] WARNING: Saturation failed for segment {segment_idx_completed}")
                    dprint(f"[WARNING] Chain (Seg {segment_idx_completed}): Saturation failed. Continuing with unsaturated video.")
            
            # --- 2. Brightness ---
            brightness_adjust = full_orchestrator_payload.get("after_first_post_generation_brightness", 0.0)
            if isinstance(brightness_adjust, (float, int)) and abs(brightness_adjust) > 1e-6:
                dprint(f"Chain (Seg {segment_idx_completed}): Applying post-gen brightness {brightness_adjust} to {video_to_process_abs_path}")
                
                bright_filename = f"s{segment_idx_completed}_bright_{brightness_adjust:+.2f}{video_to_process_abs_path.suffix}"
                brightened_video_output_abs_path, new_db_path = prepare_output_path(
                    task_id=wgp_task_id,
                    filename=bright_filename,
                    main_output_dir_base=Path(segment_processing_dir_for_saturation_str)
                )
                
                processed_video = apply_brightness_to_video_frames(str(video_to_process_abs_path), brightened_video_output_abs_path, brightness_adjust, wgp_task_id)

                if processed_video and processed_video.exists():
                    print(f"[CHAIN_DEBUG] Brightness adjustment applied successfully to segment {segment_idx_completed}")
                    debug_video_analysis(brightened_video_output_abs_path, f"BRIGHTENED_Seg{segment_idx_completed}", wgp_task_id)
                    dprint(f"Chain (Seg {segment_idx_completed}): Brightness adjustment successful. New path: {new_db_path}")
                    _cleanup_intermediate_video(full_orchestrator_payload, video_to_process_abs_path, segment_idx_completed, "saturated", dprint)

                    video_to_process_abs_path = brightened_video_output_abs_path
                    final_video_path_for_db = new_db_path
                else:
                    print(f"[CHAIN_DEBUG] WARNING: Brightness adjustment failed for segment {segment_idx_completed}")
                    dprint(f"[WARNING] Chain (Seg {segment_idx_completed}): Brightness adjustment failed. Continuing with previous video version.")

        # --- 3. Color Matching (Applied to all segments if enabled) ---
        if chain_details.get("colour_match_videos"):
            start_ref = chain_details.get("cm_start_ref_path")
            end_ref = chain_details.get("cm_end_ref_path")
            print(f"[CHAIN_DEBUG] Color matching requested for segment {segment_idx_completed}")
            print(f"[CHAIN_DEBUG] Start ref: {start_ref}")
            print(f"[CHAIN_DEBUG] End ref: {end_ref}")
            dprint(f"Chain (Seg {segment_idx_completed}): Color matching requested. Start Ref: {start_ref}, End Ref: {end_ref}")

            if start_ref and end_ref and Path(start_ref).exists() and Path(end_ref).exists():
                cm_filename = f"s{segment_idx_completed}_colormatched{video_to_process_abs_path.suffix}"
                cm_video_output_abs_path, new_db_path = prepare_output_path(
                    task_id=wgp_task_id,
                    filename=cm_filename,
                    main_output_dir_base=Path(segment_processing_dir_for_saturation_str)
                )

                matched_video_path = sm_apply_color_matching_to_video(
                    str(video_to_process_abs_path),
                    start_ref,
                    end_ref,
                    str(cm_video_output_abs_path),
                    dprint
                )

                if matched_video_path and Path(matched_video_path).exists():
                    print(f"[CHAIN_DEBUG] Color matching applied successfully to segment {segment_idx_completed}")
                    debug_video_analysis(Path(matched_video_path), f"COLORMATCHED_Seg{segment_idx_completed}", wgp_task_id)
                    dprint(f"Chain (Seg {segment_idx_completed}): Color matching successful. New path: {new_db_path}")
                    _cleanup_intermediate_video(full_orchestrator_payload, video_to_process_abs_path, segment_idx_completed, "pre-colormatch", dprint)

                    video_to_process_abs_path = Path(matched_video_path)
                    final_video_path_for_db = new_db_path
                else:
                    print(f"[CHAIN_DEBUG] WARNING: Color matching failed for segment {segment_idx_completed}")
                    dprint(f"[WARNING] Chain (Seg {segment_idx_completed}): Color matching failed. Continuing with previous video version.")
            else:
                print(f"[CHAIN_DEBUG] WARNING: Color matching skipped - missing or invalid reference images")
                dprint(f"[WARNING] Chain (Seg {segment_idx_completed}): Skipping color matching due to missing or invalid reference image paths.")

        # --- 4. Optional: Overlay start/end images above the video ---
        if chain_details.get("show_input_images"):
            banner_start = chain_details.get("start_image_path")
            banner_end = chain_details.get("end_image_path")
            if banner_start and banner_end and Path(banner_start).exists() and Path(banner_end).exists():
                banner_filename = f"s{segment_idx_completed}_with_inputs{video_to_process_abs_path.suffix}"
                banner_video_abs_path, new_db_path = prepare_output_path(
                    task_id=wgp_task_id,
                    filename=banner_filename,
                    main_output_dir_base=Path(segment_processing_dir_for_saturation_str)
                )

                if sm_overlay_start_end_images_above_video(
                    start_image_path=banner_start,
                    end_image_path=banner_end,
                    input_video_path=str(video_to_process_abs_path),
                    output_video_path=str(banner_video_abs_path),
                    dprint=dprint,
                ):
                    print(f"[CHAIN_DEBUG] Banner overlay applied successfully to segment {segment_idx_completed}")
                    debug_video_analysis(banner_video_abs_path, f"BANNER_OVERLAY_Seg{segment_idx_completed}", wgp_task_id)
                    dprint(f"Chain (Seg {segment_idx_completed}): Banner overlay successful. New path: {new_db_path}")
                    _cleanup_intermediate_video(full_orchestrator_payload, video_to_process_abs_path, segment_idx_completed, "pre-banner", dprint)

                    video_to_process_abs_path = banner_video_abs_path
                    final_video_path_for_db = new_db_path
                else:
                    print(f"[CHAIN_DEBUG] WARNING: Banner overlay failed for segment {segment_idx_completed}")
                    dprint(f"[WARNING] Chain (Seg {segment_idx_completed}): Banner overlay failed. Keeping previous video version.")
            else:
                print(f"[CHAIN_DEBUG] WARNING: Banner overlay skipped - missing valid start/end images")
                dprint(f"[WARNING] Chain (Seg {segment_idx_completed}): show_input_images enabled but valid start/end images not found.")

        # The orchestrator has already enqueued all segment and stitch tasks.
        print(f"[CHAIN_DEBUG] Chaining complete for segment {segment_idx_completed}")
        print(f"[CHAIN_DEBUG] Final video path for DB: {final_video_path_for_db}")
        debug_video_analysis(video_to_process_abs_path, f"FINAL_CHAINED_Seg{segment_idx_completed}", wgp_task_id)
        msg = f"Chain (Seg {segment_idx_completed}): Post-WGP processing complete. Final path for this WGP task's output: {final_video_path_for_db}"
        dprint(msg)
        return True, msg, str(final_video_path_for_db)

    except Exception as e_chain:
        error_msg = f"Chain (Seg {chain_details.get('segment_index_completed', 'N/A')} for WGP {wgp_task_id}): Failed during chaining: {e_chain}"
        print(f"[ERROR] {error_msg}")
        traceback.print_exc()
        
        # Notify orchestrator of chaining failure
        orchestrator_task_id_ref = chain_details.get("orchestrator_task_id_ref") if chain_details else None
        if orchestrator_task_id_ref:
            try:
                db_ops.update_task_status(
                    orchestrator_task_id_ref,
                    db_ops.STATUS_FAILED,
                    error_msg[:500]  # Truncate to avoid DB overflow
                )
                dprint(f"Chain: Marked orchestrator task {orchestrator_task_id_ref} as FAILED due to chaining failure")
            except Exception as e_orch:
                dprint(f"Chain: Warning - could not update orchestrator status: {e_orch}")
        
        return False, error_msg, str(final_video_path_for_db) # Return path as it was before error



def _cleanup_intermediate_video(orchestrator_payload, video_path: Path, segment_idx: int, stage: str, dprint):
    """Helper to cleanup intermediate video files during chaining."""
    # Delete intermediates **only** when every cleanup-bypass flag is false.
    # That now includes the worker-server global debug flag (db_ops.debug_mode)
    # so that running the server with --debug automatically preserves files.
    if (
        not orchestrator_payload.get("skip_cleanup_enabled", False)
        and not orchestrator_payload.get("debug_mode_enabled", False)
        and not db_ops.debug_mode
        and video_path.exists()
    ):
        try:
            video_path.unlink()
            dprint(f"Chain (Seg {segment_idx}): Removed intermediate '{stage}' video {video_path}")
        except Exception as e_del:
            dprint(f"Chain (Seg {segment_idx}): Warning - could not remove intermediate video {video_path}: {e_del}")

def attempt_ffmpeg_crossfade_fallback(segment_video_paths: list[str], overlaps: list[int], output_path: Path, task_id: str, dprint) -> bool:
    """
    Fallback cross-fade implementation using FFmpeg's xfade filter.
    Achieves the same visual effect as frame-based cross-fade without frame extraction.

    Args:
        segment_video_paths: List of video file paths to stitch
        overlaps: List of overlap frame counts between segments
        output_path: Path for output video
        task_id: Task ID for logging
        dprint: Debug print function

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        import subprocess
        import cv2

        if len(segment_video_paths) < 2:
            dprint(f"FFmpeg Fallback: Not enough videos to cross-fade ({len(segment_video_paths)})")
            return False

        # Get video properties from first segment to calculate timing
        cap = cv2.VideoCapture(segment_video_paths[0])
        if not cap.isOpened():
            dprint(f"FFmpeg Fallback: Cannot read first video properties")
            return False

        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        if fps <= 0:
            dprint(f"FFmpeg Fallback: Invalid FPS ({fps})")
            return False

        print(f"[FFMPEG FALLBACK] Creating cross-fade with {len(segment_video_paths)} videos at {fps} FPS")

        # Build FFmpeg command for cross-fade stitching
        cmd = ["ffmpeg", "-y"]  # -y to overwrite output

        # Add all input videos
        for video_path in segment_video_paths:
            cmd.extend(["-i", str(video_path)])

        # Build complex filter for cross-fade transitions
        filter_parts = []
        current_label = "[0:v]"

        for i, overlap_frames in enumerate(overlaps):
            if i >= len(segment_video_paths) - 1:
                break

            next_input_idx = i + 1
            next_label = f"[{next_input_idx}:v]"
            output_label = f"[fade{i}]" if i < len(overlaps) - 1 else ""

            # Convert overlap frames to duration in seconds
            overlap_duration = overlap_frames / fps

            # Get duration of current video to calculate offset
            cap = cv2.VideoCapture(segment_video_paths[i])
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            video_duration = total_frames / fps
            offset_time = video_duration - overlap_duration

            # Create xfade filter
            xfade_filter = f"{current_label}{next_label}xfade=transition=fade:duration={overlap_duration:.3f}:offset={offset_time:.3f}{output_label}"
            filter_parts.append(xfade_filter)

            current_label = f"[fade{i}]"

        if filter_parts:
            filter_complex = ";".join(filter_parts)
            cmd.extend(["-filter_complex", filter_complex])

        # Add output parameters
        cmd.extend([
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            str(output_path)
        ])

        print(f"[FFMPEG FALLBACK] Running command: {' '.join(cmd[:10])}...")  # Don't log full command (too long)
        dprint(f"FFmpeg Fallback: Running cross-fade command")

        # Run FFmpeg
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
            print(f"[FFMPEG FALLBACK] ✅ Success! Output: {output_path} ({output_path.stat().st_size:,} bytes)")
            dprint(f"FFmpeg Fallback: Success - created {output_path}")
            return True
        else:
            print(f"[FFMPEG FALLBACK] ❌ Failed with return code {result.returncode}")
            if result.stderr:
                print(f"[FFMPEG FALLBACK] Error: {result.stderr[:200]}...")
            dprint(f"FFmpeg Fallback: Failed - {result.stderr[:100] if result.stderr else 'Unknown error'}")
            return False

    except subprocess.TimeoutExpired:
        print(f"[FFMPEG FALLBACK] ❌ Timeout after 300 seconds")
        dprint(f"FFmpeg Fallback: Timeout")
        return False
    except Exception as e:
        print(f"[FFMPEG FALLBACK] ❌ Exception: {e}")
        dprint(f"FFmpeg Fallback: Exception - {e}")
        return False


def _handle_travel_stitch_task(task_params_from_db: dict, main_output_dir_base: Path, stitch_task_id_str: str, *, dprint):
    travel_logger.essential(f"Starting travel stitch task", task_id=stitch_task_id_str)
    log_ram_usage("Stitch start", task_id=stitch_task_id_str)
    dprint(f"[IMMEDIATE DEBUG] _handle_travel_stitch_task: Starting for {stitch_task_id_str}")
    dprint(f"[IMMEDIATE DEBUG] task_params_from_db keys: {list(task_params_from_db.keys())}")
    dprint(f"_handle_travel_stitch_task: Starting for {stitch_task_id_str}")
    # Safe logging: Use safe_json_repr to prevent hangs
    dprint(f"Stitch task_params_from_db: {safe_json_repr(task_params_from_db)}")
    stitch_params = task_params_from_db # This now contains full_orchestrator_payload
    stitch_success = False
    final_video_location_for_db = None
    
    try:
        # --- 1. Initialization & Parameter Extraction --- 
        orchestrator_task_id_ref = stitch_params.get("orchestrator_task_id_ref")
        orchestrator_run_id = stitch_params.get("orchestrator_run_id")
        full_orchestrator_payload = stitch_params.get("full_orchestrator_payload")

        print(f"[IMMEDIATE DEBUG] orchestrator_run_id: {orchestrator_run_id}")
        print(f"[IMMEDIATE DEBUG] orchestrator_task_id_ref: {orchestrator_task_id_ref}")
        print(f"[IMMEDIATE DEBUG] full_orchestrator_payload present: {full_orchestrator_payload is not None}")

        if not all([orchestrator_task_id_ref, orchestrator_run_id, full_orchestrator_payload]):
            msg = f"Stitch task {stitch_task_id_str} missing critical orchestrator refs or full_orchestrator_payload."
            travel_logger.error(msg, task_id=stitch_task_id_str)
            return False, msg

        project_id_for_stitch = stitch_params.get("project_id")
        current_run_base_output_dir_str = stitch_params.get("current_run_base_output_dir", 
                                                            full_orchestrator_payload.get("main_output_dir_for_run", str(main_output_dir_base.resolve())))
        current_run_base_output_dir = Path(current_run_base_output_dir_str)
        
        # Use the base directory directly without creating stitch-specific subdirectories
        stitch_processing_dir = current_run_base_output_dir
        stitch_processing_dir.mkdir(parents=True, exist_ok=True)
        dprint(f"Stitch Task {stitch_task_id_str}: Processing in {stitch_processing_dir.resolve()}")

        num_expected_new_segments = full_orchestrator_payload["num_new_segments_to_generate"]
        print(f"[IMMEDIATE DEBUG] num_expected_new_segments: {num_expected_new_segments}")
        
        # Ensure parsed_res_wh is a tuple of integers for stitch task with model grid snapping
        parsed_res_wh_str = full_orchestrator_payload["parsed_resolution_wh"]
        try:
            parsed_res_raw = sm_parse_resolution(parsed_res_wh_str)
            if parsed_res_raw is None:
                raise ValueError(f"sm_parse_resolution returned None for input: {parsed_res_wh_str}")
            parsed_res_wh = snap_resolution_to_model_grid(parsed_res_raw)
        except Exception as e_parse_res_stitch:
            msg = f"Stitch Task {stitch_task_id_str}: Invalid format or error parsing parsed_resolution_wh '{parsed_res_wh_str}': {e_parse_res_stitch}"
            print(f"[ERROR Task {stitch_task_id_str}]: {msg}"); return False, msg
        dprint(f"Stitch Task {stitch_task_id_str}: Parsed resolution (w,h): {parsed_res_wh}")

        final_fps = full_orchestrator_payload.get("fps_helpers", 16)
        expanded_frame_overlaps = full_orchestrator_payload["frame_overlap_expanded"]
        crossfade_sharp_amt = full_orchestrator_payload.get("crossfade_sharp_amt", 0.3)
        initial_continued_video_path_str = full_orchestrator_payload.get("continue_from_video_resolved_path")

        # [OVERLAP DEBUG] Add detailed debug for overlap values
        dprint(f"[OVERLAP DEBUG] Stitch: expanded_frame_overlaps from payload: {expanded_frame_overlaps}")

        # Extract upscale parameters
        upscale_factor = full_orchestrator_payload.get("upscale_factor", 0.0) # Default to 0.0 if not present
        upscale_model_name = full_orchestrator_payload.get("upscale_model_name") # Default to None if not present

        # --- 2. Collect Paths to All Segment Videos --- 
        segment_video_paths_for_stitch = []
        if initial_continued_video_path_str and Path(initial_continued_video_path_str).exists():
            dprint(f"Stitch: Prepending initial continued video: {initial_continued_video_path_str}")
            # Check the continue video properties
            cap = cv2.VideoCapture(str(initial_continued_video_path_str))
            if cap.isOpened():
                continue_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                continue_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                continue_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                dprint(f"Stitch: Continue video properties - Resolution: {continue_width}x{continue_height}, Frames: {continue_frame_count}")
                dprint(f"Stitch: Target resolution for stitching: {parsed_res_wh[0]}x{parsed_res_wh[1]}")
                if continue_width != parsed_res_wh[0] or continue_height != parsed_res_wh[1]:
                    dprint(f"Stitch: WARNING - Continue video resolution mismatch! Will need resizing during crossfade.")
            else:
                dprint(f"Stitch: ERROR - Could not open continue video for property check")
            segment_video_paths_for_stitch.append(str(Path(initial_continued_video_path_str).resolve()))
        
        # Fetch completed segments with a small retry loop to handle race conditions
        max_stitch_fetch_retries = 6  # Allow up to ~18s total wait
        completed_segment_outputs_from_db = []
        
        print(f"[IMMEDIATE DEBUG] About to start retry loop for run_id: {orchestrator_run_id}")
        
        for attempt in range(max_stitch_fetch_retries):
            print(f"[IMMEDIATE DEBUG] Stitch fetch attempt {attempt+1}/{max_stitch_fetch_retries} for run_id: {orchestrator_run_id}")
            dprint(f"[DEBUG] Stitch fetch attempt {attempt+1}/{max_stitch_fetch_retries} for run_id: {orchestrator_run_id}")
            
            try:
                completed_segment_outputs_from_db = db_ops.get_completed_segment_outputs_for_stitch(orchestrator_run_id, project_id=project_id_for_stitch) or []
                print(f"[IMMEDIATE DEBUG] DB query returned: {completed_segment_outputs_from_db}")
            except Exception as e_db_query:
                print(f"[IMMEDIATE DEBUG] DB query failed: {e_db_query}")
                completed_segment_outputs_from_db = []
            
            dprint(f"[DEBUG] Attempt {attempt+1} returned {len(completed_segment_outputs_from_db)} segments")
            print(f"[IMMEDIATE DEBUG] Attempt {attempt+1} returned {len(completed_segment_outputs_from_db)} segments")
            
            if len(completed_segment_outputs_from_db) >= num_expected_new_segments:
                dprint(f"[DEBUG] Expected {num_expected_new_segments} segment rows found on attempt {attempt+1}. Proceeding.")
                print(f"[IMMEDIATE DEBUG] Expected {num_expected_new_segments} segment rows found on attempt {attempt+1}. Proceeding.")
                break
            dprint(f"Stitch: No completed segment rows found (attempt {attempt+1}/{max_stitch_fetch_retries}). Waiting 3s and retrying...")
            print(f"[IMMEDIATE DEBUG] Insufficient segments found (attempt {attempt+1}/{max_stitch_fetch_retries}). Waiting 3s and retrying...")
            if attempt < max_stitch_fetch_retries - 1:  # Don't sleep after the last attempt
                time.sleep(3)
        dprint(f"Stitch Task {stitch_task_id_str}: Completed segments fetched: {completed_segment_outputs_from_db}")
        print(f"[IMMEDIATE DEBUG] Final completed_segment_outputs_from_db: {completed_segment_outputs_from_db}")

        # ------------------------------------------------------------------
        # 2b. Resolve each returned video path (relative path or URL)
        # ------------------------------------------------------------------
        dprint(f"[STITCH_DEBUG] Starting path resolution for {len(completed_segment_outputs_from_db)} segments")
        dprint(f"[STITCH_DEBUG] Raw DB results: {completed_segment_outputs_from_db}")
        for seg_idx, video_path_str_from_db in completed_segment_outputs_from_db:
            dprint(f"[STITCH_DEBUG] Processing segment {seg_idx} with path: {video_path_str_from_db}")
            resolved_video_path_for_stitch: Path | None = None

            if not video_path_str_from_db:
                dprint(f"[STITCH_DEBUG] WARNING: Segment {seg_idx} has empty video_path in DB; skipping.")
                continue

            # Case A: Relative path that starts with files/ - resolve from current working directory
            if video_path_str_from_db.startswith("files/") or video_path_str_from_db.startswith("public/files/"):
                dprint(f"[STITCH_DEBUG] Case A: Relative path detected for segment {seg_idx}")
                # Resolve relative to current working directory
                base_dir = Path.cwd()
                absolute_path_candidate = (base_dir / "public" / video_path_str_from_db.lstrip("public/")).resolve()
                print(f"[STITCH_DEBUG] Resolved relative path '{video_path_str_from_db}' to '{absolute_path_candidate}' for segment {seg_idx}")
                dprint(f"Stitch: Resolved relative path '{video_path_str_from_db}' to '{absolute_path_candidate}' for segment {seg_idx}")
                if absolute_path_candidate.exists() and absolute_path_candidate.is_file():
                    resolved_video_path_for_stitch = absolute_path_candidate
                    print(f"[STITCH_DEBUG] ✅ File exists at resolved path")
                else:
                    print(f"[STITCH_DEBUG] ❌ File missing at resolved path")
                    dprint(f"[WARNING] Stitch: Resolved absolute path '{absolute_path_candidate}' for segment {seg_idx} is missing.")

            # Case B: Remote public URL (Supabase storage)
            elif video_path_str_from_db.startswith("http"):
                print(f"[STITCH_DEBUG] Case B: Remote URL detected for segment {seg_idx}")
                try:
                    from ..common_utils import download_file as sm_download_file
                    remote_url = video_path_str_from_db
                    local_filename = Path(remote_url).name
                    local_download_path = stitch_processing_dir / f"seg{seg_idx:02d}_{local_filename}"
                    print(f"[STITCH_DEBUG] Remote URL: {remote_url}")
                    print(f"[STITCH_DEBUG] Local download path: {local_download_path}")
                    dprint(f"[DEBUG] Remote URL detected, local download path: {local_download_path}")
                    
                    # Check if cached file exists and validate its frame count against orchestrator's expected values
                    need_download = True
                    if local_download_path.exists():
                        print(f"[STITCH_DEBUG] Local copy exists, validating frame count...")
                        try:
                            cached_frames, _ = sm_get_video_frame_count_and_fps(str(local_download_path))
                            expected_segment_frames = full_orchestrator_payload["segment_frames_expanded"]
                            expected_frames = expected_segment_frames[seg_idx] if seg_idx < len(expected_segment_frames) else None
                            print(f"[STITCH_DEBUG] Cached file has {cached_frames} frames (expected: {expected_frames})")
                            
                            if expected_frames and cached_frames == expected_frames:
                                print(f"[STITCH_DEBUG] ✅ Cached file frame count matches expected ({cached_frames} frames)")
                                need_download = False
                            elif expected_frames:
                                print(f"[STITCH_DEBUG] ❌ Cached file frame count mismatch! Expected {expected_frames}, got {cached_frames}, will re-download")
                            else:
                                print(f"[STITCH_DEBUG] ❌ No expected frame count available for segment {seg_idx}, will re-download")
                        except Exception as e_validate:
                            print(f"[STITCH_DEBUG] ❌ Could not validate cached file: {e_validate}, will re-download")
                    
                    if need_download:
                        print(f"[STITCH_DEBUG] Downloading remote segment {seg_idx}...")
                        dprint(f"Stitch: Downloading remote segment {seg_idx} from {remote_url} to {local_download_path}")
                        # Remove stale cached file if it exists
                        if local_download_path.exists():
                            local_download_path.unlink()
                        sm_download_file(remote_url, stitch_processing_dir, local_download_path.name)
                        print(f"[STITCH_DEBUG] ✅ Download completed for segment {seg_idx}")
                        dprint(f"[DEBUG] Download completed for segment {seg_idx}")
                    else:
                        print(f"[STITCH_DEBUG] ✅ Using validated cached file for segment {seg_idx}")
                        dprint(f"Stitch: Using validated cached file for segment {seg_idx} at {local_download_path}")
                    
                    resolved_video_path_for_stitch = local_download_path
                except Exception as e_dl:
                    print(f"[STITCH_DEBUG] ❌ Download failed for segment {seg_idx}: {e_dl}")
                    dprint(f"[WARNING] Stitch: Failed to download remote video for segment {seg_idx}: {e_dl}")

            # Case C: Provided absolute/local path
            else:
                print(f"[STITCH_DEBUG] Case C: Absolute/local path for segment {seg_idx}")
                absolute_path_candidate = Path(video_path_str_from_db).resolve()
                print(f"[STITCH_DEBUG] Treating as absolute path: {absolute_path_candidate}")
                dprint(f"[DEBUG] Treating as absolute path: {absolute_path_candidate}")
                if absolute_path_candidate.exists() and absolute_path_candidate.is_file():
                    resolved_video_path_for_stitch = absolute_path_candidate
                    print(f"[STITCH_DEBUG] ✅ Absolute path exists")
                    dprint(f"[DEBUG] Absolute path exists: {absolute_path_candidate}")
                else:
                    print(f"[STITCH_DEBUG] ❌ Absolute path missing or not a file")
                    dprint(f"[WARNING] Stitch: Absolute path '{absolute_path_candidate}' for segment {seg_idx} does not exist or is not a file.")

            if resolved_video_path_for_stitch is not None:
                segment_video_paths_for_stitch.append(str(resolved_video_path_for_stitch))
                print(f"[STITCH_DEBUG] ✅ Added video for segment {seg_idx}: {resolved_video_path_for_stitch}")
                dprint(f"Stitch: Added video for segment {seg_idx}: {resolved_video_path_for_stitch}")
                
                # Analyze the resolved video immediately
                debug_video_analysis(resolved_video_path_for_stitch, f"RESOLVED_Seg{seg_idx}", stitch_task_id_str)
            else: 
                print(f"[STITCH_DEBUG] ❌ Unable to resolve video for segment {seg_idx}; will be excluded from stitching.")
                dprint(f"[WARNING] Stitch: Unable to resolve video for segment {seg_idx}; will be excluded from stitching.")

        print(f"[STITCH_DEBUG] Path resolution complete")
        print(f"[STITCH_DEBUG] Final segment_video_paths_for_stitch: {segment_video_paths_for_stitch}")
        print(f"[STITCH_DEBUG] Total videos collected: {len(segment_video_paths_for_stitch)}")
        dprint(f"[DEBUG] Final segment_video_paths_for_stitch: {segment_video_paths_for_stitch}")
        dprint(f"[DEBUG] Total videos collected: {len(segment_video_paths_for_stitch)}")
        # [CRITICAL DEBUG] Log each video's frame count before stitching
        dprint(f"[CRITICAL DEBUG] About to stitch videos:")
        expected_segment_frames = full_orchestrator_payload["segment_frames_expanded"]
        for idx, video_path in enumerate(segment_video_paths_for_stitch):
            try:
                frame_count, fps = sm_get_video_frame_count_and_fps(video_path)
                expected_frames = expected_segment_frames[idx] if idx < len(expected_segment_frames) else "unknown"
                dprint(f"[CRITICAL DEBUG] Video {idx}: {video_path} -> {frame_count} frames @ {fps} FPS (expected: {expected_frames})")
                if expected_frames != "unknown" and frame_count != expected_frames:
                    dprint(f"[CRITICAL DEBUG] ⚠️  FRAME COUNT MISMATCH! Expected {expected_frames}, got {frame_count}")
            except Exception as e_debug:
                dprint(f"[CRITICAL DEBUG] Video {idx}: {video_path} -> ERROR: {e_debug}")

        total_videos_for_stitch = (1 if initial_continued_video_path_str and Path(initial_continued_video_path_str).exists() else 0) + num_expected_new_segments
        dprint(f"[DEBUG] Expected total videos: {total_videos_for_stitch}")
        if len(segment_video_paths_for_stitch) < total_videos_for_stitch:
            # This is a warning because some segments might have legitimately failed and been skipped by their handlers.
            # The stitcher should proceed with what it has, unless it has zero or one video when multiple were expected.
            dprint(f"[WARNING] Stitch: Expected {total_videos_for_stitch} videos for stitch, but found {len(segment_video_paths_for_stitch)}. Stitching with available videos.")
        
        if not segment_video_paths_for_stitch:
            dprint(f"[ERROR] Stitch: No valid segment videos found to stitch. DB returned {len(completed_segment_outputs_from_db)} segments, but none resolved to valid paths.")
            raise ValueError("Stitch: No valid segment videos found to stitch.")
        if len(segment_video_paths_for_stitch) == 1 and total_videos_for_stitch > 1:
            dprint(f"Stitch: Only one video segment found ({segment_video_paths_for_stitch[0]}) but {total_videos_for_stitch} were expected. Using this single video as the 'stitched' output.")
            # No actual stitching needed, just move/copy this single video to final dest.

        # --- 3. Stitching (Crossfade or Concatenate) --- 
        current_stitched_video_path: Path | None = None # This will hold the path to the current version of the stitched video


        if len(segment_video_paths_for_stitch) == 1:
            # If only one video, copy it directly using prepare_output_path
            source_single_video_path = Path(segment_video_paths_for_stitch[0])
            single_video_filename = f"{orchestrator_run_id}_final{source_single_video_path.suffix}"
            
            current_stitched_video_path, _ = prepare_output_path(
                task_id=stitch_task_id_str,
                filename=single_video_filename,
                main_output_dir_base=stitch_processing_dir
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
            
            # --- NEW OVERLAP DEBUG LOGGING ---
            dprint(f"[OVERLAP DEBUG] Number of videos: {len(segment_video_paths_for_stitch)} (expected stitch points: {num_stitch_points})")
            dprint(f"[OVERLAP DEBUG] actual_overlaps_for_stitching: {actual_overlaps_for_stitching}")
            if len(actual_overlaps_for_stitching) != num_stitch_points:
                dprint(f"[OVERLAP DEBUG] ⚠️  MISMATCH! We have {len(actual_overlaps_for_stitching)} overlaps for {num_stitch_points} joins")
            for join_idx, ov in enumerate(actual_overlaps_for_stitching):
                dprint(f"[OVERLAP DEBUG]   Join {join_idx} (video {join_idx} -> {join_idx+1}): overlap={ov}")
            # --- END NEW LOGGING ---
            
            any_positive_overlap = any(o > 0 for o in actual_overlaps_for_stitching)

            raw_stitched_video_filename = f"{orchestrator_run_id}_stitched.mp4"
            path_for_raw_stitched_video, _ = prepare_output_path(
                task_id=stitch_task_id_str,
                filename=raw_stitched_video_filename,
                main_output_dir_base=stitch_processing_dir
            )

            if any_positive_overlap:
                print(f"[CRITICAL DEBUG] Using cross-fade due to overlap values: {actual_overlaps_for_stitching}. Output to: {path_for_raw_stitched_video}")
                print(f"[STITCH_ANALYSIS] Cross-fade stitching analysis:")
                print(f"[STITCH_ANALYSIS]   Number of videos: {len(segment_video_paths_for_stitch)}")
                print(f"[STITCH_ANALYSIS]   Overlap values: {actual_overlaps_for_stitching}")
                print(f"[STITCH_ANALYSIS]   Expected stitch points: {num_stitch_points}")
                
                dprint(f"Stitch: Using cross-fade due to overlap values: {actual_overlaps_for_stitching}. Output to: {path_for_raw_stitched_video}")

                # Wait for all segment videos to be stable before extracting frames
                print(f"[CRITICAL DEBUG] Waiting for {len(segment_video_paths_for_stitch)} segment videos to be stable before frame extraction...")
                stable_paths = []
                for idx, video_path in enumerate(segment_video_paths_for_stitch):
                    dprint(f"Stitch: Checking file stability for segment {idx}: {video_path}")
                    if not Path(video_path).exists():
                        print(f"[CRITICAL DEBUG] Segment {idx} video file does not exist: {video_path}")
                        stable_paths.append(False)
                        continue

                    file_stable = sm_wait_for_file_stable(video_path, checks=5, interval=1.0, dprint=dprint)
                    if file_stable:
                        print(f"[CRITICAL DEBUG] Segment {idx} video file is stable: {video_path}")
                        stable_paths.append(True)
                    else:
                        print(f"[CRITICAL DEBUG] Segment {idx} video file is NOT stable after waiting: {video_path}")
                        stable_paths.append(False)

                if not all(stable_paths):
                    unstable_indices = [i for i, stable in enumerate(stable_paths) if not stable]
                    raise ValueError(f"Stitch: One or more segment videos are not stable or missing: indices {unstable_indices}")

                print(f"[CRITICAL DEBUG] All segment videos are stable. Proceeding with frame extraction...")

                # Retry frame extraction with backoff and re-download for corrupted videos
                max_extraction_attempts = 3
                all_segment_frames_lists = None
                retry_log = []  # Track all retry attempts for detailed error reporting

                for attempt in range(max_extraction_attempts):
                    print(f"[CRITICAL DEBUG] Frame extraction attempt {attempt + 1}/{max_extraction_attempts}")
                    attempt_start_time = time.time()
                    all_segment_frames_lists = [sm_extract_frames_from_video(p, dprint_func=dprint) for p in segment_video_paths_for_stitch]

                    # [CRITICAL DEBUG] Log frame extraction results
                    print(f"[CRITICAL DEBUG] Frame extraction results (attempt {attempt + 1}):")
                    failed_segments = []
                    successful_segments = []
                    for idx, frame_list in enumerate(all_segment_frames_lists):
                        if frame_list is not None and len(frame_list) > 0:
                            print(f"[CRITICAL DEBUG] Segment {idx}: {len(frame_list)} frames extracted")
                            successful_segments.append(idx)
                        else:
                            print(f"[CRITICAL DEBUG] Segment {idx}: FAILED to extract frames")
                            failed_segments.append(idx)

                    # Log this attempt
                    attempt_duration = time.time() - attempt_start_time
                    attempt_info = {
                        "attempt": attempt + 1,
                        "duration_seconds": round(attempt_duration, 2),
                        "successful_segments": successful_segments,
                        "failed_segments": failed_segments,
                        "redownloads": []
                    }

                    # Check if all extractions succeeded
                    if all(f_list is not None and len(f_list) > 0 for f_list in all_segment_frames_lists):
                        print(f"[CRITICAL DEBUG] All frame extractions successful on attempt {attempt + 1}")
                        retry_log.append(attempt_info)
                        break

                    # If not the last attempt, try to re-download corrupted videos before retry
                    if attempt < max_extraction_attempts - 1 and failed_segments:
                        wait_time = 3 + (attempt * 2)  # Progressive backoff: 3s, 5s, 7s
                        print(f"[CRITICAL DEBUG] Frame extraction failed for segments {failed_segments}. Attempting re-download and retry in {wait_time} seconds...")

                        # Try to re-download failed segments (only for remote URLs)
                        redownload_attempted = False
                        for failed_idx in failed_segments:
                            if failed_idx < len(completed_segment_outputs_from_db):
                                seg_output = completed_segment_outputs_from_db[failed_idx]
                                video_path_str_from_db = seg_output.get("video_file_path", "")

                                # Check if it's a remote URL that can be re-downloaded
                                if video_path_str_from_db.startswith("http"):
                                    try:
                                        failed_video_path = Path(segment_video_paths_for_stitch[failed_idx])
                                        print(f"[CRITICAL DEBUG] Re-downloading corrupted segment {failed_idx} from {video_path_str_from_db}")

                                        redownload_start = time.time()

                                        # Delete corrupted file
                                        if failed_video_path.exists():
                                            failed_video_path.unlink()
                                            print(f"[CRITICAL DEBUG] Deleted corrupted file: {failed_video_path}")

                                        # Re-download
                                        sm_download_file(video_path_str_from_db, stitch_processing_dir, failed_video_path.name)

                                        # Wait for stability
                                        if sm_wait_for_file_stable(failed_video_path, checks=5, interval=1.0, dprint=dprint):
                                            print(f"[CRITICAL DEBUG] Re-downloaded segment {failed_idx} successfully")
                                            redownload_duration = time.time() - redownload_start
                                            attempt_info["redownloads"].append({
                                                "segment_idx": failed_idx,
                                                "source_url": video_path_str_from_db,
                                                "duration_seconds": round(redownload_duration, 2),
                                                "success": True
                                            })
                                            redownload_attempted = True
                                        else:
                                            print(f"[CRITICAL DEBUG] Re-downloaded segment {failed_idx} not stable")
                                            redownload_duration = time.time() - redownload_start
                                            attempt_info["redownloads"].append({
                                                "segment_idx": failed_idx,
                                                "source_url": video_path_str_from_db,
                                                "duration_seconds": round(redownload_duration, 2),
                                                "success": False,
                                                "error": "File not stable after download"
                                            })

                                    except Exception as e_redownload:
                                        print(f"[CRITICAL DEBUG] Re-download failed for segment {failed_idx}: {e_redownload}")
                                        redownload_duration = time.time() - redownload_start
                                        attempt_info["redownloads"].append({
                                            "segment_idx": failed_idx,
                                            "source_url": video_path_str_from_db,
                                            "duration_seconds": round(redownload_duration, 2),
                                            "success": False,
                                            "error": str(e_redownload)
                                        })
                                else:
                                    print(f"[CRITICAL DEBUG] Segment {failed_idx} is not a remote URL, cannot re-download: {video_path_str_from_db}")
                                    attempt_info["redownloads"].append({
                                        "segment_idx": failed_idx,
                                        "source_url": video_path_str_from_db,
                                        "duration_seconds": 0,
                                        "success": False,
                                        "error": "Not a remote URL - cannot re-download"
                                    })

                        if redownload_attempted:
                            print(f"[CRITICAL DEBUG] Re-download completed. Waiting {wait_time} seconds before next extraction attempt...")
                        else:
                            print(f"[CRITICAL DEBUG] No re-downloads attempted. Waiting {wait_time} seconds before next extraction attempt...")

                        time.sleep(wait_time)

                    # Log this attempt (whether successful or failed)
                    retry_log.append(attempt_info)
                else:
                    # All attempts failed - generate detailed error report
                    failed_segments = [i for i, f_list in enumerate(all_segment_frames_lists) if not (f_list is not None and len(f_list) > 0)]

                    # Build detailed error message
                    error_details = []
                    error_details.append(f"Frame extraction failed for segments {failed_segments} after {max_extraction_attempts} attempts")
                    error_details.append(f"Total segments in stitch: {len(segment_video_paths_for_stitch)}")

                    # Add per-segment analysis
                    for idx, video_path in enumerate(segment_video_paths_for_stitch):
                        video_path_obj = Path(video_path)
                        status = "✅ SUCCESS" if idx not in failed_segments else "❌ FAILED"

                        if video_path_obj.exists():
                            try:
                                file_size = video_path_obj.stat().st_size
                                # Try to get basic video info
                                import cv2
                                cap = cv2.VideoCapture(str(video_path))
                                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else -1
                                fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else -1
                                cap.release()

                                error_details.append(f"  Segment {idx} [{status}]: {video_path_obj.name} ({file_size:,} bytes, {frame_count} frames, {fps:.1f} fps)")
                            except Exception:
                                file_size = video_path_obj.stat().st_size if video_path_obj.exists() else 0
                                error_details.append(f"  Segment {idx} [{status}]: {video_path_obj.name} ({file_size:,} bytes, properties unreadable)")
                        else:
                            error_details.append(f"  Segment {idx} [{status}]: {video_path} (FILE MISSING)")

                    # Add source information if available
                    if 'completed_segment_outputs_from_db' in locals():
                        error_details.append("Source URLs:")
                        for idx in failed_segments:
                            if idx < len(completed_segment_outputs_from_db):
                                seg_output = completed_segment_outputs_from_db[idx]
                                source_url = seg_output.get("video_file_path", "Unknown")
                                error_details.append(f"  Failed segment {idx} source: {source_url}")

                    # Add retry history
                    if retry_log:
                        error_details.append("Retry History:")
                        for log_entry in retry_log:
                            attempt_summary = f"  Attempt {log_entry['attempt']}: {log_entry['duration_seconds']}s, Success:{len(log_entry['successful_segments'])}, Failed:{len(log_entry['failed_segments'])}"
                            if log_entry['redownloads']:
                                redownload_summary = []
                                for rd in log_entry['redownloads']:
                                    status = "✅" if rd['success'] else "❌"
                                    redownload_summary.append(f"Seg{rd['segment_idx']}({status}{rd['duration_seconds']}s)")
                                attempt_summary += f", Redownloads:[{','.join(redownload_summary)}]"
                            error_details.append(attempt_summary)

                    # Before failing, try FFmpeg-based cross-fade as fallback
                    print(f"[CRITICAL DEBUG] Frame extraction failed completely. Attempting FFmpeg cross-fade fallback...")
                    try:
                        ffmpeg_result = attempt_ffmpeg_crossfade_fallback(
                            segment_video_paths_for_stitch,
                            actual_overlaps_for_stitching,
                            path_for_raw_stitched_video,
                            stitch_task_id_str,
                            dprint
                        )
                        if ffmpeg_result:
                            print(f"[CRITICAL DEBUG] FFmpeg cross-fade fallback succeeded!")
                            current_stitched_video_path = path_for_raw_stitched_video
                        else:
                            detailed_error = "Stitch: Both frame extraction and FFmpeg cross-fade fallback failed. " + " | ".join(error_details)
                            raise ValueError(detailed_error)
                    except Exception as e_ffmpeg:
                        detailed_error = f"Stitch: Frame extraction failed and FFmpeg fallback also failed ({str(e_ffmpeg)}). " + " | ".join(error_details)
                        raise ValueError(detailed_error)
                
                final_stitched_frames = []
                
                # Process each stitch point
                for i in range(num_stitch_points): 
                    frames_prev_segment = all_segment_frames_lists[i]
                    frames_curr_segment = all_segment_frames_lists[i+1]
                    current_overlap_val = actual_overlaps_for_stitching[i]

                    print(f"[CRITICAL DEBUG] Stitch point {i}: segments {i}->{i+1}, overlap={current_overlap_val}")
                    print(f"[CRITICAL DEBUG] Prev segment: {len(frames_prev_segment)} frames, Curr segment: {len(frames_curr_segment)} frames")

                    # --- NEW OVERLAP DETAIL LOG ---
                    if current_overlap_val > 0:
                        start_prev = len(frames_prev_segment) - current_overlap_val
                        end_prev = len(frames_prev_segment) - 1
                        start_curr = 0
                        end_curr = current_overlap_val - 1
                        print(
                            f"[OVERLAP_DETAIL] Join {i}: blending prev[{start_prev}:{end_prev}] with curr[{start_curr}:{end_curr}] (total {current_overlap_val} frames)"
                        )
                    # --- END OVERLAP DETAIL LOG ---

                    if i == 0:
                        # For the first stitch point, add frames from segment 0 up to the overlap
                        if current_overlap_val > 0:
                            # Add frames before the overlap region
                            frames_before_overlap = frames_prev_segment[:-current_overlap_val]
                            final_stitched_frames.extend(frames_before_overlap)
                            print(f"[CRITICAL DEBUG] Added {len(frames_before_overlap)} frames from segment 0 (before overlap)")
                        else:
                            # No overlap, add all frames from segment 0
                            final_stitched_frames.extend(frames_prev_segment)
                            print(f"[CRITICAL DEBUG] Added all {len(frames_prev_segment)} frames from segment 0 (no overlap)")
                    else:
                        pass

                    if current_overlap_val > 0:
                        # Check if we should regenerate anchor frames (skip blending the anchor)
                        regenerate_anchors = full_orchestrator_payload.get("regenerate_anchors", False)

                        if regenerate_anchors and current_overlap_val > 1:
                            # Regenerate anchor mode: crossfade all but the last frame (anchor)
                            # The anchor frame will be taken directly from current segment
                            crossfade_count = current_overlap_val - 1

                            # Remove the overlap frames (minus 1 for anchor) from accumulated
                            if i > 0:
                                frames_to_remove = min(crossfade_count, len(final_stitched_frames))
                                if frames_to_remove > 0:
                                    del final_stitched_frames[-frames_to_remove:]
                                    print(f"[CRITICAL DEBUG] [REGENERATE_ANCHORS] Removed {frames_to_remove} frames before cross-fade (keeping previous anchor)")

                            # Blend the non-anchor overlapping frames
                            frames_prev_for_fade = frames_prev_segment[-crossfade_count:] if crossfade_count > 0 else []
                            frames_curr_for_fade = frames_curr_segment[:crossfade_count]
                            faded_frames = sm_cross_fade_overlap_frames(frames_prev_for_fade, frames_curr_for_fade, crossfade_count, "linear_sharp", crossfade_sharp_amt)
                            final_stitched_frames.extend(faded_frames)
                            print(f"[CRITICAL DEBUG] [REGENERATE_ANCHORS] Added {len(faded_frames)} cross-faded frames (skipping anchor)")

                            # Add the regenerated anchor frame directly (no blend)
                            anchor_frame = frames_curr_segment[crossfade_count]
                            final_stitched_frames.append(anchor_frame)
                            print(f"[CRITICAL DEBUG] [REGENERATE_ANCHORS] Added regenerated anchor frame directly (no blend)")

                            # Adjust start index for remaining frames
                            start_index_for_curr_tail = current_overlap_val
                        else:
                            # Normal crossfade mode: blend all overlap frames
                            # Remove the overlap frames already appended from the previous segment so that
                            # they can be replaced by the blended cross-fade frames for this stitch point.
                            if i > 0:
                                frames_to_remove = min(current_overlap_val, len(final_stitched_frames))
                                if frames_to_remove > 0:
                                    del final_stitched_frames[-frames_to_remove:]
                                    print(f"[CRITICAL DEBUG] Removed {frames_to_remove} duplicate overlap frames before cross-fade (stitch point {i})")
                            # Blend the overlapping frames
                            faded_frames = sm_cross_fade_overlap_frames(frames_prev_segment, frames_curr_segment, current_overlap_val, "linear_sharp", crossfade_sharp_amt)
                            final_stitched_frames.extend(faded_frames)
                            print(f"[CRITICAL DEBUG] Added {len(faded_frames)} cross-faded frames")

                            # Normal start index for remaining frames
                            start_index_for_curr_tail = current_overlap_val
                    else:
                        start_index_for_curr_tail = 0

                    # Add the non-overlapping part of the current segment
                    if len(frames_curr_segment) > start_index_for_curr_tail:
                        frames_to_add = frames_curr_segment[start_index_for_curr_tail:]
                        final_stitched_frames.extend(frames_to_add)
                        print(f"[CRITICAL DEBUG] Added {len(frames_to_add)} frames from segment {i+1} (after overlap)")
                    
                    print(f"[CRITICAL DEBUG] Running total after stitch point {i}: {len(final_stitched_frames)} frames")
                
                if not final_stitched_frames: raise ValueError("Stitch: No frames produced after cross-fade logic.")
                
                # [CRITICAL DEBUG] Final calculation summary
                # With proper cross-fade: output = sum(all frames) - sum(overlaps)
                # Because overlapped frames are blended, not duplicated
                total_input_frames = sum(len(frames) for frames in all_segment_frames_lists)
                total_overlaps = sum(actual_overlaps_for_stitching)
                expected_output_frames = total_input_frames - total_overlaps
                actual_output_frames = len(final_stitched_frames)
                print(f"[CRITICAL DEBUG] FINAL CROSS-FADE SUMMARY:")
                print(f"[CRITICAL DEBUG] Total input frames: {total_input_frames}")
                print(f"[CRITICAL DEBUG] Total overlaps: {total_overlaps}")
                print(f"[CRITICAL DEBUG] Expected output: {expected_output_frames}")
                print(f"[CRITICAL DEBUG] Actual output: {actual_output_frames}")
                print(f"[CRITICAL DEBUG] Match: {expected_output_frames == actual_output_frames}")
                
                created_video_path_obj = sm_create_video_from_frames_list(final_stitched_frames, path_for_raw_stitched_video, final_fps, parsed_res_wh)
                if created_video_path_obj and created_video_path_obj.exists():
                    current_stitched_video_path = created_video_path_obj
                else:
                    raise RuntimeError(f"Stitch: Cross-fade sm_create_video_from_frames_list failed to produce video at {path_for_raw_stitched_video}")

            else: 
                dprint(f"Stitch: Using simple FFmpeg concatenation. Output to: {path_for_raw_stitched_video}")
                try:
                    from ..common_utils import stitch_videos_ffmpeg as sm_stitch_videos_ffmpeg
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
            print(f"[STITCH UPSCALE] Starting upscale process: {upscale_factor}x using model {upscale_model_name}")
            dprint(f"Stitch: Upscaling (x{upscale_factor}) video {current_stitched_video_path.name} using model {upscale_model_name}")
            
            original_frames_count, original_fps = sm_get_video_frame_count_and_fps(str(current_stitched_video_path))
            if original_frames_count is None or original_frames_count == 0:
                raise ValueError(f"Stitch: Cannot get frame count or 0 frames for video {current_stitched_video_path} before upscaling.")
            
            print(f"[STITCH UPSCALE] Input video: {original_frames_count} frames @ {original_fps} FPS")
            print(f"[STITCH UPSCALE] Target resolution: {int(parsed_res_wh[0] * upscale_factor)}x{int(parsed_res_wh[1] * upscale_factor)}")
            dprint(f"[DEBUG] Pre-upscale analysis: {original_frames_count} frames, {original_fps} FPS")

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

            upscaler_engine_to_use = stitch_params.get("execution_engine_for_upscale", "wgp")
            
            db_ops.add_task_to_db(
                task_payload=upscale_payload, 
                task_type_str=upscaler_engine_to_use
            )
            print(f"[STITCH UPSCALE] Enqueued upscale sub-task {upscale_sub_task_id} ({upscaler_engine_to_use}). Waiting...")
            print(f"Stitch Task {stitch_task_id_str}: Enqueued upscale sub-task {upscale_sub_task_id} ({upscaler_engine_to_use}). Waiting...")
            
            poll_interval_ups = full_orchestrator_payload.get("poll_interval", 15)
            poll_timeout_ups = full_orchestrator_payload.get("poll_timeout_upscale", full_orchestrator_payload.get("poll_timeout", 30 * 60) * 2)
            
            print(f"[STITCH UPSCALE] Polling for completion (timeout: {poll_timeout_ups}s, interval: {poll_interval_ups}s)")
            
            upscaled_video_db_location = db_ops.poll_task_status(
                task_id=upscale_sub_task_id, 
                poll_interval_seconds=poll_interval_ups, 
                timeout_seconds=poll_timeout_ups
            )
            print(f"[STITCH UPSCALE] Poll result: {upscaled_video_db_location}")
            dprint(f"Stitch Task {stitch_task_id_str}: Upscale sub-task {upscale_sub_task_id} poll result: {upscaled_video_db_location}")

            if upscaled_video_db_location:
                # Path is already absolute (Supabase URL or absolute path)
                upscaled_video_abs_path: Path = Path(upscaled_video_db_location)

                if upscaled_video_abs_path.exists():
                    print(f"[STITCH UPSCALE] Upscale completed successfully: {upscaled_video_abs_path}")
                    dprint(f"Stitch: Upscale sub-task {upscale_sub_task_id} completed. Output: {upscaled_video_abs_path}")
                    
                    # Analyze upscaled result
                    try:
                        upscaled_frame_count, upscaled_fps = sm_get_video_frame_count_and_fps(str(upscaled_video_abs_path))
                        print(f"[STITCH UPSCALE] Upscaled result: {upscaled_frame_count} frames @ {upscaled_fps} FPS")
                        dprint(f"[DEBUG] Post-upscale analysis: {upscaled_frame_count} frames, {upscaled_fps} FPS")
                        
                        # Compare frame counts
                        if upscaled_frame_count != original_frames_count:
                            print(f"[STITCH UPSCALE] Frame count changed during upscale: {original_frames_count} → {upscaled_frame_count}")
                    except Exception as e_post_upscale:
                        print(f"[WARNING] Could not analyze upscaled video: {e_post_upscale}")
                    
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
                    print(f"[STITCH UPSCALE] ERROR: Upscale output missing at {upscaled_video_abs_path}. Using non-upscaled video.")
                    print(f"[WARNING] Stitch Task {stitch_task_id_str}: Upscale sub-task {upscale_sub_task_id} output missing ({upscaled_video_abs_path}). Using non-upscaled video.")
            else: 
                print(f"[STITCH UPSCALE] ERROR: Upscale sub-task failed or timed out. Using non-upscaled video.")
                print(f"[WARNING] Stitch Task {stitch_task_id_str}: Upscale sub-task {upscale_sub_task_id} failed or timed out. Using non-upscaled video.")

        elif upscale_factor > 1.0 and not upscale_model_name:
            print(f"[STITCH UPSCALE] Upscale factor {upscale_factor} > 1.0 but no upscale_model_name provided. Skipping upscale.")
            dprint(f"Stitch: Upscale factor {upscale_factor} > 1.0 but no upscale_model_name provided. Skipping upscale.")
        else:
            print(f"[STITCH UPSCALE] No upscaling requested (factor: {upscale_factor})")
            dprint(f"Stitch: No upscaling (factor: {upscale_factor})")

        # Use consistent UUID-based naming for final video
        timestamp_short = datetime.now().strftime("%H%M%S")
        unique_suffix = uuid.uuid4().hex[:6]
        if upscale_factor > 1.0:
            final_video_filename = f"travel_final_upscaled_{upscale_factor:.1f}x_{timestamp_short}_{unique_suffix}{video_path_after_optional_upscale.suffix}"
        else:
            final_video_filename = f"travel_final_{timestamp_short}_{unique_suffix}{video_path_after_optional_upscale.suffix}"
        
        final_video_path, initial_db_location = prepare_output_path_with_upload(
            task_id=stitch_task_id_str,
            filename=final_video_filename,
            main_output_dir_base=stitch_processing_dir,
            dprint=dprint
        )
        
        # Move the video to final location if it's not already there
        if video_path_after_optional_upscale.resolve() != final_video_path.resolve():
            dprint(f"Stitch Task {stitch_task_id_str}: Moving {video_path_after_optional_upscale} to {final_video_path}")
            shutil.move(str(video_path_after_optional_upscale), str(final_video_path))
        else:
            dprint(f"Stitch Task {stitch_task_id_str}: Video already at final destination {final_video_path}")
        
        # Handle Supabase upload (if configured) and get final location for DB
        final_video_location_for_db = upload_and_get_final_output_location(
            final_video_path,
            final_video_filename,  # Pass only the filename to avoid redundant subfolder
            initial_db_location,
            dprint=dprint
        )
        
        travel_logger.info(f"Stitch complete: Final video saved to {final_video_path}", task_id=stitch_task_id_str)
        dprint(f"Stitch Task {stitch_task_id_str}: Final video saved to: {final_video_path} (DB location: {final_video_location_for_db})")
        
        # Analyze final result
        try:
            final_frame_count, final_fps = sm_get_video_frame_count_and_fps(str(final_video_path))
            final_duration = final_frame_count / final_fps if final_fps > 0 else 0
            print(f"[STITCH FINAL] Final video: {final_frame_count} frames @ {final_fps} FPS = {final_duration:.2f}s")
            print(f"[STITCH_FINAL_ANALYSIS] Complete stitching analysis:")
            print(f"[STITCH_FINAL_ANALYSIS]   Input segments: {len(segment_video_paths_for_stitch)}")
            print(f"[STITCH_FINAL_ANALYSIS]   Overlap settings: {expanded_frame_overlaps}")
            # Calculate expected final length for analysis
            try:
                # Try to calculate expected final length from orchestrator data
                expected_segment_frames = full_orchestrator_payload.get("segment_frames_expanded", [])
                if expected_segment_frames:
                    total_input_frames = sum(expected_segment_frames)
                    total_overlaps = sum(expanded_frame_overlaps)
                    expected_final_length = total_input_frames - total_overlaps
                    print(f"[STITCH_FINAL_ANALYSIS]   Expected final frames: {expected_final_length}")
                    print(f"[STITCH_FINAL_ANALYSIS]   Actual final frames: {final_frame_count}")
                    if final_frame_count != expected_final_length:
                        print(f"[STITCH_FINAL_ANALYSIS]   ⚠️  FINAL LENGTH MISMATCH! Expected {expected_final_length}, got {final_frame_count}")
                else:
                    print(f"[STITCH_FINAL_ANALYSIS]   Expected final frames: Not available (no segment_frames_expanded)")
                    print(f"[STITCH_FINAL_ANALYSIS]   Actual final frames: {final_frame_count}")
            except Exception as e:
                print(f"[STITCH_FINAL_ANALYSIS]   Expected final frames: Not calculated ({e})")
                print(f"[STITCH_FINAL_ANALYSIS]   Actual final frames: {final_frame_count}")
            
            # Detailed analysis of the final video
            debug_video_analysis(final_video_path, "FINAL_STITCHED_VIDEO", stitch_task_id_str)
            
            dprint(f"[DEBUG] Final video analysis: {final_frame_count} frames, {final_fps} FPS, {final_duration:.2f}s duration")
        except Exception as e_final_analysis:
            print(f"[WARNING] Could not analyze final video: {e_final_analysis}")
        
        # Note: Individual segments already have banner overlays applied when show_input_images is enabled,
        # so the stitched video will automatically include them. No additional overlay needed here.
        
        stitch_success = True

        # --- Cleanup Downloaded Segment Files ---
        cleanup_enabled = (
            not full_orchestrator_payload.get("skip_cleanup_enabled", False) and
            not full_orchestrator_payload.get("debug_mode_enabled", False) and
            not db_ops.debug_mode
        )

        if cleanup_enabled:
            files_cleaned = 0
            total_size_cleaned = 0

            for video_path_str in segment_video_paths_for_stitch:
                video_path = Path(video_path_str)

                # Skip the initial continued video (not downloaded)
                if (initial_continued_video_path_str and
                    str(video_path.resolve()) == str(Path(initial_continued_video_path_str).resolve())):
                    continue

                # Only delete files in our processing directory (downloaded files)
                if video_path.exists() and stitch_processing_dir in video_path.parents:
                    try:
                        file_size = video_path.stat().st_size
                        video_path.unlink()
                        files_cleaned += 1
                        total_size_cleaned += file_size
                        dprint(f"Stitch: Cleaned up downloaded segment {video_path.name} ({file_size:,} bytes)")
                    except Exception as e_cleanup:
                        dprint(f"Stitch: Failed to clean up {video_path}: {e_cleanup}")

            if files_cleaned > 0:
                print(f"[STITCH_CLEANUP] Removed {files_cleaned} downloaded files ({total_size_cleaned:,} bytes)")
        else:
            print(f"[STITCH_CLEANUP] Skipping cleanup (debug mode or cleanup disabled)")

        # Note: Orchestrator will be marked complete by worker.py after stitch upload completes
        # This ensures the orchestrator gets the final storage URL, not a local path

        # Return the final video path so the stitch task itself gets uploaded via Edge Function
        log_ram_usage("Stitch end (success)", task_id=stitch_task_id_str)
        return stitch_success, str(final_video_path.resolve())

    except Exception as e:
        travel_logger.error(f"Stitch: Unexpected error during stitching: {e}", task_id=stitch_task_id_str)
        travel_logger.debug(traceback.format_exc(), task_id=stitch_task_id_str)
        
        # Notify orchestrator of stitch failure
        if 'orchestrator_task_id_ref' in locals() and orchestrator_task_id_ref:
            try:
                error_msg = f"Stitch task failed: {str(e)[:200]}"
                db_ops.update_task_status(
                    orchestrator_task_id_ref,
                    db_ops.STATUS_FAILED,
                    error_msg
                )
                dprint(f"Stitch: Marked orchestrator task {orchestrator_task_id_ref} as FAILED due to exception")
            except Exception as e_orch:
                dprint(f"Stitch: Warning - could not update orchestrator status: {e_orch}")

        log_ram_usage("Stitch end (error)", task_id=stitch_task_id_str)
        return False, f"Stitch task failed: {str(e)[:200]}"
