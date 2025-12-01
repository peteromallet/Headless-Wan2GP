"""
Join Clips Orchestrator - Sequentially join multiple video clips

This orchestrator takes a list of video clips and creates a chain of join_clips_child
tasks to progressively build them into a single seamless video.

Pattern:
    Input: [clip_A, clip_B, clip_C, clip_D]

    Creates:
        join_0: clip_A + clip_B → AB.mp4 (no dependency)
        join_1: AB.mp4 + clip_C → ABC.mp4 (depends on join_0)
        join_2: ABC.mp4 + clip_D → ABCD.mp4 (depends on join_1)

    Each join task fetches the output of its predecessor via get_predecessor_output_via_edge_function()
"""

import traceback
from pathlib import Path
from typing import Tuple, List, Optional
import cv2

from .. import db_operations as db_ops
from ..common_utils import download_video_if_url
from ..video_utils import extract_frames_from_video


def _extract_boundary_frames_for_vlm(
    clip_list: List[dict],
    temp_dir: Path,
    orchestrator_task_id: str,
    dprint
) -> List[Tuple[str, str]]:
    """
    Extract boundary frames from clips for VLM prompt generation.
    
    For each join (clip[i] → clip[i+1]), extracts:
    - Last frame from clip[i]
    - First frame from clip[i+1]
    
    Args:
        clip_list: List of clip dicts with 'url' keys
        temp_dir: Directory to save temporary frame images
        orchestrator_task_id: Task ID for logging
        dprint: Debug print function
        
    Returns:
        List of (start_frame_path, end_frame_path) tuples for each join
    """
    image_pairs = []
    num_joins = len(clip_list) - 1
    
    # Cache downloaded videos and their frames to avoid re-downloading
    clip_frames_cache = {}  # url -> (first_frame_path, last_frame_path)
    
    for idx in range(num_joins):
        clip_start = clip_list[idx]
        clip_end = clip_list[idx + 1]
        
        start_url = clip_start.get("url")
        end_url = clip_end.get("url")
        
        if not start_url or not end_url:
            dprint(f"[VLM_EXTRACT] Join {idx}: Missing URL, skipping")
            image_pairs.append((None, None))
            continue
            
        try:
            # Get last frame from clip_start
            if start_url in clip_frames_cache:
                _, start_last_frame_path = clip_frames_cache[start_url]
            else:
                dprint(f"[VLM_EXTRACT] Join {idx}: Downloading clip_start: {start_url[:80]}...")
                local_start_path = download_video_if_url(
                    start_url,
                    download_target_dir=temp_dir,
                    task_id_for_logging=orchestrator_task_id,
                    descriptive_name=f"clip_{idx}_start"
                )
                
                # Extract all frames and get last one
                start_frames = extract_frames_from_video(local_start_path, dprint_func=dprint)
                if not start_frames:
                    dprint(f"[VLM_EXTRACT] Join {idx}: Failed to extract frames from start clip")
                    image_pairs.append((None, None))
                    continue
                    
                # Save first and last frames
                start_first_frame_path = temp_dir / f"vlm_clip{idx}_first.jpg"
                start_last_frame_path = temp_dir / f"vlm_clip{idx}_last.jpg"
                cv2.imwrite(str(start_first_frame_path), cv2.cvtColor(start_frames[0], cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(start_last_frame_path), cv2.cvtColor(start_frames[-1], cv2.COLOR_RGB2BGR))
                clip_frames_cache[start_url] = (str(start_first_frame_path), str(start_last_frame_path))
                start_last_frame_path = str(start_last_frame_path)
                dprint(f"[VLM_EXTRACT] Join {idx}: Extracted {len(start_frames)} frames from start clip")
            
            # Get first frame from clip_end
            if end_url in clip_frames_cache:
                end_first_frame_path, _ = clip_frames_cache[end_url]
            else:
                dprint(f"[VLM_EXTRACT] Join {idx}: Downloading clip_end: {end_url[:80]}...")
                local_end_path = download_video_if_url(
                    end_url,
                    download_target_dir=temp_dir,
                    task_id_for_logging=orchestrator_task_id,
                    descriptive_name=f"clip_{idx+1}_end"
                )
                
                # Extract all frames and get first one
                end_frames = extract_frames_from_video(local_end_path, dprint_func=dprint)
                if not end_frames:
                    dprint(f"[VLM_EXTRACT] Join {idx}: Failed to extract frames from end clip")
                    image_pairs.append((None, None))
                    continue
                    
                # Save first and last frames
                end_first_frame_path = temp_dir / f"vlm_clip{idx+1}_first.jpg"
                end_last_frame_path = temp_dir / f"vlm_clip{idx+1}_last.jpg"
                cv2.imwrite(str(end_first_frame_path), cv2.cvtColor(end_frames[0], cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(end_last_frame_path), cv2.cvtColor(end_frames[-1], cv2.COLOR_RGB2BGR))
                clip_frames_cache[end_url] = (str(end_first_frame_path), str(end_last_frame_path))
                end_first_frame_path = str(end_first_frame_path)
                dprint(f"[VLM_EXTRACT] Join {idx}: Extracted {len(end_frames)} frames from end clip")
            
            image_pairs.append((start_last_frame_path, end_first_frame_path))
            dprint(f"[VLM_EXTRACT] Join {idx}: Boundary frames ready")
            
        except Exception as e:
            dprint(f"[VLM_EXTRACT] Join {idx}: ERROR extracting frames: {e}")
            image_pairs.append((None, None))
            
    return image_pairs


def _generate_vlm_prompts_for_joins(
    image_pairs: List[Tuple[Optional[str], Optional[str]]],
    base_prompt: str,
    gap_frame_count: int,
    fps: int,
    vlm_device: str,
    dprint
) -> List[Optional[str]]:
    """
    Generate VLM-enhanced prompts for ALL joins using boundary frame pairs.
    
    When enhance_prompt=True, runs VLM for every join to generate motion-focused
    prompts based on the boundary frames.
    
    Args:
        image_pairs: List of (start_frame_path, end_frame_path) tuples
        base_prompt: Base prompt to use as VLM context (from orchestrator payload)
        gap_frame_count: Number of transition frames (for duration context)
        fps: Frames per second
        vlm_device: Device for VLM inference ('cuda' or 'cpu')
        dprint: Debug print function
        
    Returns:
        List of enhanced prompts (None for joins with missing image pairs)
    """
    from ..vlm_utils import generate_transition_prompts_batch
    
    num_joins = len(image_pairs)
    result = [None] * num_joins
    
    # Filter out invalid pairs, track their indices
    valid_pairs = []
    valid_indices = []
    
    for idx in range(num_joins):
        start_path, end_path = image_pairs[idx]
        if start_path and end_path:
            valid_pairs.append((start_path, end_path))
            valid_indices.append(idx)
        else:
            dprint(f"[VLM_PROMPTS] Join {idx}: Skipping - missing image pair")
    
    if not valid_pairs:
        dprint(f"[VLM_PROMPTS] No valid image pairs for VLM processing")
        return result
    
    dprint(f"[VLM_PROMPTS] Processing {len(valid_pairs)}/{num_joins} joins with VLM")
    dprint(f"[VLM_PROMPTS] Base prompt context: '{base_prompt[:80]}...'" if base_prompt else "[VLM_PROMPTS] No base prompt (VLM will infer from frames)")
    
    # Build lists - same base prompt and frame count for all joins
    base_prompts = [base_prompt] * len(valid_pairs)
    num_frames_list = [gap_frame_count] * len(valid_pairs)
    
    try:
        enhanced_prompts = generate_transition_prompts_batch(
            image_pairs=valid_pairs,
            base_prompts=base_prompts,
            num_frames_list=num_frames_list,
            fps=fps,
            device=vlm_device,
            dprint=dprint
        )
        
        # Map results back to original indices
        for i, enhanced in zip(valid_indices, enhanced_prompts):
            result[i] = enhanced
            dprint(f"[VLM_PROMPTS] Join {i}: {enhanced[:80]}...")
            
        return result
        
    except Exception as e:
        dprint(f"[VLM_PROMPTS] ERROR in batch VLM: {e}")
        traceback.print_exc()
        return result


def _handle_join_clips_orchestrator_task(
    task_params_from_db: dict,
    main_output_dir_base: Path,
    orchestrator_task_id_str: str,
    orchestrator_project_id: str | None,
    *,
    dprint
) -> Tuple[bool, str]:
    """
    Handle join_clips_orchestrator task - creates chained join_clips_child tasks.

    Args:
        task_params_from_db: Task parameters containing orchestrator_details
        main_output_dir_base: Base output directory
        orchestrator_task_id_str: Orchestrator task ID
        orchestrator_project_id: Project ID for authorization
        dprint: Debug print function

    Returns:
        (success: bool, message: str)
    """
    dprint(f"[JOIN_ORCHESTRATOR] Starting orchestrator task {orchestrator_task_id_str}")

    generation_success = False
    output_message = f"Join orchestration for {orchestrator_task_id_str} initiated."

    try:
        # === 1. PARSE ORCHESTRATOR PAYLOAD ===
        if 'orchestrator_details' not in task_params_from_db:
            dprint("[JOIN_ORCHESTRATOR] ERROR: orchestrator_details missing")
            return False, "orchestrator_details missing"

        orchestrator_payload = task_params_from_db['orchestrator_details']
        dprint(f"[JOIN_ORCHESTRATOR] Orchestrator payload keys: {list(orchestrator_payload.keys())}")

        # Extract required fields
        clip_list = orchestrator_payload.get("clip_list", [])
        run_id = orchestrator_payload.get("run_id")

        if not clip_list or len(clip_list) < 2:
            return False, "clip_list must contain at least 2 clips"

        if not run_id:
            return False, "run_id is required"

        num_joins = len(clip_list) - 1
        dprint(f"[JOIN_ORCHESTRATOR] Processing {len(clip_list)} clips = {num_joins} join tasks")

        # Extract join settings
        join_settings = {
            "context_frame_count": orchestrator_payload.get("context_frame_count", 8),
            "gap_frame_count": orchestrator_payload.get("gap_frame_count", 53),
            "replace_mode": orchestrator_payload.get("replace_mode", False),
            "blend_frames": orchestrator_payload.get("blend_frames", 3),
            "prompt": orchestrator_payload.get("prompt", "smooth transition"),
            "negative_prompt": orchestrator_payload.get("negative_prompt", ""),
            "model": orchestrator_payload.get("model", "lightning_baseline_2_2_2"),
            "regenerate_anchors": orchestrator_payload.get("regenerate_anchors", True),
            "num_anchor_frames": orchestrator_payload.get("num_anchor_frames", 3),
            "aspect_ratio": orchestrator_payload.get("aspect_ratio"),
            # LoRA parameters
            "additional_loras": orchestrator_payload.get("additional_loras", {}),
            # Keep bridging image param
            "keep_bridging_images": orchestrator_payload.get("keep_bridging_images", False),
        }

        per_join_settings = orchestrator_payload.get("per_join_settings", [])
        output_base_dir = orchestrator_payload.get("output_base_dir", str(main_output_dir_base.resolve()))

        # Create run-specific output directory
        current_run_output_dir = Path(output_base_dir) / f"join_clips_run_{run_id}"
        current_run_output_dir.mkdir(parents=True, exist_ok=True)
        dprint(f"[JOIN_ORCHESTRATOR] Run output directory: {current_run_output_dir}")

        # === VLM PROMPT ENHANCEMENT (optional) ===
        # Follows same pattern as travel_between_images.py orchestrator
        enhance_prompt = orchestrator_payload.get("enhance_prompt", False)
        vlm_enhanced_prompts: List[Optional[str]] = [None] * num_joins  # Default: no enhancements
        
        if enhance_prompt:
            dprint(f"[JOIN_ORCHESTRATOR] enhance_prompt=True, generating VLM-enhanced prompts for {num_joins} joins")
            
            vlm_device = orchestrator_payload.get("vlm_device", "cuda")
            vlm_temp_dir = current_run_output_dir / "vlm_temp"
            vlm_temp_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                # Step 1: Extract boundary frames from all clips
                dprint(f"[JOIN_ORCHESTRATOR] Extracting boundary frames from {len(clip_list)} clips...")
                image_pairs = _extract_boundary_frames_for_vlm(
                    clip_list=clip_list,
                    temp_dir=vlm_temp_dir,
                    orchestrator_task_id=orchestrator_task_id_str,
                    dprint=dprint
                )
                
                # Step 2: Generate VLM prompts for all joins
                base_prompt = join_settings.get("prompt", "")
                gap_frame_count = join_settings.get("gap_frame_count", 53)
                fps = orchestrator_payload.get("fps", 16)
                
                dprint(f"[JOIN_ORCHESTRATOR] Running VLM batch on {len(image_pairs)} pairs...")
                vlm_enhanced_prompts = _generate_vlm_prompts_for_joins(
                    image_pairs=image_pairs,
                    base_prompt=base_prompt,
                    gap_frame_count=gap_frame_count,
                    fps=fps,
                    vlm_device=vlm_device,
                    dprint=dprint
                )
                
                valid_count = sum(1 for p in vlm_enhanced_prompts if p is not None)
                dprint(f"[JOIN_ORCHESTRATOR] VLM enhancement complete: {valid_count}/{num_joins} prompts generated")
                
            except Exception as vlm_error:
                dprint(f"[JOIN_ORCHESTRATOR] VLM enhancement failed, using base prompts: {vlm_error}")
                traceback.print_exc()
                vlm_enhanced_prompts = [None] * num_joins
        else:
            dprint(f"[JOIN_ORCHESTRATOR] enhance_prompt=False, using base prompt for all joins")


        # === 2. IDEMPOTENCY CHECK ===
        dprint(f"[JOIN_ORCHESTRATOR] Checking for existing child tasks")
        existing_child_tasks = db_ops.get_orchestrator_child_tasks(orchestrator_task_id_str)
        existing_joins = existing_child_tasks.get('join_clips_segment', [])

        if existing_joins:
            dprint(f"[JOIN_ORCHESTRATOR] Found {len(existing_joins)} existing join tasks")

            # Check if we have the expected number
            if len(existing_joins) >= num_joins:
                dprint(f"[JOIN_ORCHESTRATOR] All {num_joins} join tasks already exist")

                # Check completion status
                def is_complete(task):
                    return task.get('status') == 'complete'

                def is_terminal_failure(task):
                    status = task.get('status', '').lower()
                    return status in ('failed', 'cancelled', 'canceled', 'error')

                all_joins_complete = all(is_complete(join) for join in existing_joins)
                any_join_failed = any(is_terminal_failure(join) for join in existing_joins)

                # If any failed, mark orchestrator as failed
                if any_join_failed:
                    failed_joins = [j for j in existing_joins if is_terminal_failure(j)]
                    error_msg = f"{len(failed_joins)} join task(s) failed/cancelled"
                    dprint(f"[JOIN_ORCHESTRATOR] FAILED: {error_msg}")
                    return False, f"[ORCHESTRATOR_FAILED] {error_msg}"

                # If all complete, return final output
                if all_joins_complete:
                    import json

                    # Sort by join_index to get the last one
                    # Parse task_params if it's a JSON string
                    def get_join_index(task):
                        params = task.get('task_params', {})
                        if isinstance(params, str):
                            try:
                                params = json.loads(params)
                            except (json.JSONDecodeError, ValueError):
                                return 0
                        return params.get('join_index', 0)

                    sorted_joins = sorted(existing_joins, key=get_join_index)
                    final_join = sorted_joins[-1]
                    final_output = final_join.get('output_location', 'Completed via idempotency')

                    # Extract thumbnail from final join's params (parse JSON string if needed)
                    final_params = final_join.get('task_params', {})
                    if isinstance(final_params, str):
                        try:
                            final_params = json.loads(final_params)
                        except (json.JSONDecodeError, ValueError):
                            final_params = {}

                    final_thumbnail = final_params.get('thumbnail_url', '')

                    dprint(f"[JOIN_ORCHESTRATOR] COMPLETE: All joins finished, final output: {final_output}")
                    dprint(f"[JOIN_ORCHESTRATOR] Final thumbnail: {final_thumbnail}")

                    # Include thumbnail in completion message using JSON format
                    completion_data = json.dumps({"output_location": final_output, "thumbnail_url": final_thumbnail})
                    return True, f"[ORCHESTRATOR_COMPLETE]{completion_data}"

                # Still in progress
                complete_count = sum(1 for j in existing_joins if is_complete(j))
                dprint(f"[JOIN_ORCHESTRATOR] IDEMPOTENT: {complete_count}/{num_joins} joins complete")
                return True, f"[IDEMPOTENT] Join tasks in progress: {complete_count}/{num_joins} complete"


        # === 3. CREATE JOIN_CLIPS_CHILD TASKS ===
        dprint(f"[JOIN_ORCHESTRATOR] Creating {num_joins} join tasks in dependency chain")

        previous_join_task_id = None
        joins_created = 0

        for idx in range(num_joins):
            clip_start = clip_list[idx]
            clip_end = clip_list[idx + 1]

            dprint(f"[JOIN_ORCHESTRATOR] Creating join {idx}: {clip_start.get('name', 'clip')} + {clip_end.get('name', 'clip')}")

            # Merge global settings with per-join overrides
            task_join_settings = join_settings.copy()
            if idx < len(per_join_settings):
                task_join_settings.update(per_join_settings[idx])
                dprint(f"[JOIN_ORCHESTRATOR] Applied per-join overrides for join {idx}")

            # Apply VLM-enhanced prompt if available (overrides base prompt)
            if vlm_enhanced_prompts[idx] is not None:
                task_join_settings["prompt"] = vlm_enhanced_prompts[idx]
                dprint(f"[JOIN_ORCHESTRATOR] Join {idx}: Using VLM-enhanced prompt")

            # Build join payload
            join_payload = {
                "orchestrator_task_id_ref": orchestrator_task_id_str,
                "orchestrator_run_id": run_id,
                "project_id": orchestrator_project_id,
                "join_index": idx,
                "is_first_join": (idx == 0),
                "is_last_join": (idx == num_joins - 1),

                # First join has explicit starting path, rest fetch from dependency
                "starting_video_path": clip_start.get("url") if idx == 0 else None,
                "ending_video_path": clip_end.get("url"),

                # Join settings
                **task_join_settings,

                # Output configuration
                "current_run_base_output_dir": str(current_run_output_dir.resolve()),
                "join_output_dir": str((current_run_output_dir / f"join_{idx}").resolve()),

                # Reference to full orchestrator payload
                "full_orchestrator_payload": orchestrator_payload,
            }

            dprint(f"[JOIN_ORCHESTRATOR] Submitting join {idx} to database, depends_on={previous_join_task_id}")

            # Create task with dependency chain
            actual_db_row_id = db_ops.add_task_to_db(
                task_payload=join_payload,
                task_type_str="join_clips_segment",
                dependant_on=previous_join_task_id
            )

            dprint(f"[JOIN_ORCHESTRATOR] Join {idx} created with DB ID: {actual_db_row_id}")

            # Update for next iteration
            previous_join_task_id = actual_db_row_id
            joins_created += 1


        # === 4. SUCCESS ===
        generation_success = True
        output_message = f"Successfully enqueued {joins_created} join tasks for run {run_id}"
        dprint(f"[JOIN_ORCHESTRATOR] {output_message}")

    except Exception as e:
        msg = f"Failed during join orchestration: {e}"
        dprint(f"[JOIN_ORCHESTRATOR] ERROR: {msg}")
        traceback.print_exc()
        generation_success = False
        output_message = msg

    return generation_success, output_message
