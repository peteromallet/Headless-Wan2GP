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

Shared Core Logic:
    The _create_join_chain_tasks() function is the shared core that creates the
    dependency chain of join tasks. It is used by:
    - join_clips_orchestrator: Takes a clip_list directly
    - edit_video_orchestrator: Preprocesses portions_to_regenerate into keeper clips first
"""

import traceback
import subprocess
from pathlib import Path
from typing import Tuple, List, Optional
import cv2

from .. import db_operations as db_ops
from ..common_utils import download_video_if_url, get_video_frame_count_and_fps, upload_intermediate_file_to_storage
from ..video_utils import extract_frames_from_video, reverse_video


def _extract_boundary_frames_for_vlm(
    clip_list: List[dict],
    temp_dir: Path,
    orchestrator_task_id: str,
    replace_mode: bool,
    gap_frame_count: int,
    dprint
) -> List[Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]]:
    """
    Extract boundary frames from clips for VLM prompt generation.
    
    For each join (clip[i] → clip[i+1]), extracts 4 frames:
    - First frame from clip[i] (scene context)
    - Boundary frame from clip[i] (last frame before gap, adjusted for REPLACE mode)
    - Boundary frame from clip[i+1] (first frame after gap, adjusted for REPLACE mode)
    - Last frame from clip[i+1] (scene context)
    
    Args:
        clip_list: List of clip dicts with 'url' keys
        temp_dir: Directory to save temporary frame images
        orchestrator_task_id: Task ID for logging
        replace_mode: Whether REPLACE mode is enabled (affects boundary frame selection)
        gap_frame_count: Number of gap frames (used for REPLACE mode boundary calculation)
        dprint: Debug print function
        
    Returns:
        List of (start_first, start_boundary, end_boundary, end_last) tuples for each join
    """
    image_quads = []
    num_joins = len(clip_list) - 1
    
    # Calculate gap split for REPLACE mode (same logic as join_clips.py)
    gap_from_clip1 = gap_frame_count // 2 if replace_mode else 0
    gap_from_clip2 = (gap_frame_count - gap_from_clip1) if replace_mode else 0
    
    if replace_mode:
        dprint(f"[VLM_EXTRACT] REPLACE mode: gap_from_clip1={gap_from_clip1}, gap_from_clip2={gap_from_clip2}")
    
    # Cache downloaded videos and their extracted frames to avoid re-downloading
    # url -> (frames_list, first_path, last_path)
    clip_data_cache = {}
    
    for idx in range(num_joins):
        clip_start = clip_list[idx]
        clip_end = clip_list[idx + 1]
        
        start_url = clip_start.get("url")
        end_url = clip_end.get("url")
        
        if not start_url or not end_url:
            dprint(f"[VLM_EXTRACT] Join {idx}: Missing URL, skipping")
            image_quads.append((None, None, None, None))
            continue
            
        try:
            # === Extract frames from clip_start ===
            if start_url in clip_data_cache:
                start_frames, start_first_path, start_last_path = clip_data_cache[start_url]
            else:
                dprint(f"[VLM_EXTRACT] Join {idx}: Downloading clip_start: {start_url[:80]}...")
                local_start_path = download_video_if_url(
                    start_url,
                    download_target_dir=temp_dir,
                    task_id_for_logging=orchestrator_task_id,
                    descriptive_name=f"clip_{idx}_start"
                )
                
                # Extract all frames
                start_frames = extract_frames_from_video(local_start_path, dprint_func=dprint)
                if not start_frames:
                    dprint(f"[VLM_EXTRACT] Join {idx}: Failed to extract frames from start clip")
                    image_quads.append((None, None, None, None))
                    continue
                    
                # Save first and absolute last frames (for context)
                start_first_path = temp_dir / f"vlm_clip{idx}_first.jpg"
                start_last_path = temp_dir / f"vlm_clip{idx}_last.jpg"
                cv2.imwrite(str(start_first_path), cv2.cvtColor(start_frames[0], cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(start_last_path), cv2.cvtColor(start_frames[-1], cv2.COLOR_RGB2BGR))
                
                clip_data_cache[start_url] = (start_frames, str(start_first_path), str(start_last_path))
                dprint(f"[VLM_EXTRACT] Join {idx}: Extracted {len(start_frames)} frames from start clip")
            
            # Calculate boundary frame index for clip_start
            # In REPLACE mode: boundary is the last frame BEFORE the gap region
            # In INSERT mode: boundary is the absolute last frame
            if replace_mode and gap_from_clip1 > 0:
                start_boundary_idx = max(0, len(start_frames) - gap_from_clip1 - 1)
                dprint(f"[VLM_EXTRACT] Join {idx}: REPLACE mode - start boundary at frame {start_boundary_idx} (not {len(start_frames)-1})")
            else:
                start_boundary_idx = len(start_frames) - 1
            
            # Save boundary frame
            start_boundary_path = temp_dir / f"vlm_clip{idx}_boundary.jpg"
            cv2.imwrite(str(start_boundary_path), cv2.cvtColor(start_frames[start_boundary_idx], cv2.COLOR_RGB2BGR))
            
            # === Extract frames from clip_end ===
            if end_url in clip_data_cache:
                end_frames, end_first_path, end_last_path = clip_data_cache[end_url]
            else:
                dprint(f"[VLM_EXTRACT] Join {idx}: Downloading clip_end: {end_url[:80]}...")
                local_end_path = download_video_if_url(
                    end_url,
                    download_target_dir=temp_dir,
                    task_id_for_logging=orchestrator_task_id,
                    descriptive_name=f"clip_{idx+1}_end"
                )
                
                # Extract all frames
                end_frames = extract_frames_from_video(local_end_path, dprint_func=dprint)
                if not end_frames:
                    dprint(f"[VLM_EXTRACT] Join {idx}: Failed to extract frames from end clip")
                    image_quads.append((None, None, None, None))
                    continue
                    
                # Save first and absolute last frames (for context)
                end_first_path = temp_dir / f"vlm_clip{idx+1}_first.jpg"
                end_last_path = temp_dir / f"vlm_clip{idx+1}_last.jpg"
                cv2.imwrite(str(end_first_path), cv2.cvtColor(end_frames[0], cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(end_last_path), cv2.cvtColor(end_frames[-1], cv2.COLOR_RGB2BGR))
                
                clip_data_cache[end_url] = (end_frames, str(end_first_path), str(end_last_path))
                dprint(f"[VLM_EXTRACT] Join {idx}: Extracted {len(end_frames)} frames from end clip")
            
            # Calculate boundary frame index for clip_end
            # In REPLACE mode: boundary is the first frame AFTER the gap region
            # In INSERT mode: boundary is the absolute first frame
            if replace_mode and gap_from_clip2 > 0:
                end_boundary_idx = min(gap_from_clip2, len(end_frames) - 1)
                dprint(f"[VLM_EXTRACT] Join {idx}: REPLACE mode - end boundary at frame {end_boundary_idx} (not 0)")
            else:
                end_boundary_idx = 0
            
            # Save boundary frame
            end_boundary_path = temp_dir / f"vlm_clip{idx+1}_boundary.jpg"
            cv2.imwrite(str(end_boundary_path), cv2.cvtColor(end_frames[end_boundary_idx], cv2.COLOR_RGB2BGR))
            
            # Return quad: (start_first, start_boundary, end_boundary, end_last)
            image_quads.append((
                str(clip_data_cache[start_url][1]),  # start first
                str(start_boundary_path),             # start boundary
                str(end_boundary_path),               # end boundary
                str(clip_data_cache[end_url][2])      # end last
            ))
            dprint(f"[VLM_EXTRACT] Join {idx}: 4 frames ready (first, boundary, boundary, last)")
            
        except Exception as e:
            dprint(f"[VLM_EXTRACT] Join {idx}: ERROR extracting frames: {e}")
            image_quads.append((None, None, None, None))
            
    return image_quads


def _generate_join_transition_prompt(
    start_first_path: str,
    start_boundary_path: str,
    end_boundary_path: str,
    end_last_path: str,
    base_prompt: str,
    num_frames: int,
    fps: int,
    extender,
    dprint
) -> str:
    """
    Generate a single transition prompt for join_clips using custom 3-sentence structure:
    1. Main motion (running, walking, camera movement)
    2. Visual style (anime, illustrated, photorealistic, etc.)
    3. Scene details (flowers, particles, lighting, environment)
    
    Uses 4 images side-by-side for full context:
    - First frame of clip A (scene context)
    - Boundary frame of clip A (transition start point)
    - Boundary frame of clip B (transition end point)  
    - Last frame of clip B (scene context)
    
    Args:
        start_first_path: Path to first frame of starting clip (scene context)
        start_boundary_path: Path to boundary frame of starting clip (transition start)
        end_boundary_path: Path to boundary frame of ending clip (transition end)
        end_last_path: Path to last frame of ending clip (scene context)
        base_prompt: Base prompt for context
        num_frames: Number of transition frames
        fps: Frames per second
        extender: QwenPromptExpander instance (reused for batch efficiency)
        dprint: Debug print function
        
    Returns:
        Generated 3-sentence prompt
    """
    from PIL import Image
    
    # Load all 4 images
    start_first = Image.open(start_first_path).convert("RGB")
    start_boundary = Image.open(start_boundary_path).convert("RGB")
    end_boundary = Image.open(end_boundary_path).convert("RGB")
    end_last = Image.open(end_last_path).convert("RGB")
    
    # Combine all 4 images side by side
    images = [start_first, start_boundary, end_boundary, end_last]
    combined_width = sum(img.width for img in images)
    combined_height = max(img.height for img in images)
    combined_img = Image.new('RGB', (combined_width, combined_height))
    
    x_offset = 0
    for img in images:
        combined_img.paste(img, (x_offset, 0))
        x_offset += img.width
    
    # Calculate duration
    duration_text = ""
    if num_frames and fps:
        duration_seconds = num_frames / fps
        duration_text = f"The transition between the two middle frames occurs over approximately {duration_seconds:.1f} seconds ({num_frames} frames at {fps} FPS)."
    
    base_prompt_text = base_prompt if base_prompt and base_prompt.strip() else "a video sequence"
    
    query = f"""Look at these 4 frames from a video, left to right in time order.

The MIDDLE TWO frames show where we need to generate a transition.

Context from user: {base_prompt_text}

Write a short video prompt describing what happens between the middle frames. Include:
1. The motion or action
2. The visual style
3. Key details in the scene

Keep it under 50 words total. Just write the prompt, nothing else.

Example: "A cat jumps from table to chair. Realistic home video style. Sunny kitchen with wooden furniture."

Example: "Camera pans right across city skyline. Cinematic drone footage at sunset. Skyscrapers with orange light reflections."

Your prompt:"""

    system_prompt = "Write a short video description prompt. Under 50 words. No explanations, just the prompt."
    
    result = extender.extend_with_img(
        prompt=query,
        system_prompt=system_prompt,
        image=combined_img
    )
    
    return result.prompt.strip()


def _generate_vlm_prompts_for_joins(
    image_quads: List[Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]],
    base_prompt: str,
    gap_frame_count: int,
    fps: int,
    vlm_device: str,
    dprint
) -> List[Optional[str]]:
    """
    Generate VLM-enhanced prompts for ALL joins using 4-image quads.
    
    Uses custom join_clips prompt structure:
    1. Main motion (running, walking, camera movement)
    2. Visual style (anime, illustrated, photorealistic)
    3. Scene details (flowers, particles, environment)
    
    Args:
        image_quads: List of (start_first, start_boundary, end_boundary, end_last) tuples
        base_prompt: Base prompt to use as VLM context (from orchestrator payload)
        gap_frame_count: Number of transition frames (for duration context)
        fps: Frames per second
        vlm_device: Device for VLM inference ('cuda' or 'cpu')
        dprint: Debug print function
        
    Returns:
        List of enhanced prompts (None for joins with missing image quads)
    """
    import sys
    from pathlib import Path
    import torch
    
    num_joins = len(image_quads)
    result = [None] * num_joins
    
    # Filter out invalid quads, track their indices
    valid_quads = []
    valid_indices = []
    
    for idx in range(num_joins):
        quad = image_quads[idx]
        # All 4 images must be present
        if all(path is not None for path in quad):
            valid_quads.append(quad)
            valid_indices.append(idx)
        else:
            dprint(f"[VLM_PROMPTS] Join {idx}: Skipping - missing image(s) in quad")
    
    if not valid_quads:
        dprint(f"[VLM_PROMPTS] No valid image quads for VLM processing")
        return result
    
    dprint(f"[VLM_PROMPTS] Processing {len(valid_quads)}/{num_joins} joins with VLM (4 images each)")
    dprint(f"[VLM_PROMPTS] Base prompt context: '{base_prompt[:80]}...'" if base_prompt else "[VLM_PROMPTS] No base prompt (VLM will infer from frames)")
    
    try:
        # Add Wan2GP to path for imports
        wan_dir = Path(__file__).parent.parent.parent / "Wan2GP"
        if str(wan_dir) not in sys.path:
            sys.path.insert(0, str(wan_dir))
        
        from Wan2GP.shared.utils.prompt_extend import QwenPromptExpander
        from ..vlm_utils import download_qwen_vlm_if_needed
        
        # Log memory before loading (same as vlm_utils.py)
        if torch.cuda.is_available():
            gpu_mem_before = torch.cuda.memory_allocated() / 1024**3
            dprint(f"[VLM_PROMPTS] GPU memory BEFORE VLM load: {gpu_mem_before:.2f} GB")
        
        # Initialize VLM ONCE for all quads (batch efficiency)
        local_model_path = wan_dir / "ckpts" / "Qwen2.5-VL-7B-Instruct"
        dprint(f"[VLM_PROMPTS] Checking for model at {local_model_path}...")
        download_qwen_vlm_if_needed(local_model_path)
        
        dprint(f"[VLM_PROMPTS] Initializing Qwen2.5-VL-7B-Instruct...")
        extender = QwenPromptExpander(
            model_name=str(local_model_path),
            device=vlm_device,
            is_vl=True
        )
        dprint(f"[VLM_PROMPTS] Model loaded (initially on CPU, moves to {vlm_device} for inference)")
        
        # Process each quad
        for i, quad in enumerate(valid_quads):
            idx = valid_indices[i]
            start_first, start_boundary, end_boundary, end_last = quad
            try:
                dprint(f"[VLM_PROMPTS] Processing join {idx} ({i+1}/{len(valid_quads)})...")
                
                enhanced = _generate_join_transition_prompt(
                    start_first_path=start_first,
                    start_boundary_path=start_boundary,
                    end_boundary_path=end_boundary,
                    end_last_path=end_last,
                    base_prompt=base_prompt,
                    num_frames=gap_frame_count,
                    fps=fps,
                    extender=extender,
                    dprint=dprint
                )
                
                result[idx] = enhanced
                dprint(f"[VLM_PROMPTS] Join {idx}: {enhanced[:100]}...")
                
            except Exception as e:
                dprint(f"[VLM_PROMPTS] Join {idx}: ERROR - {e}")
                # Continue with other quads
        
        # Cleanup VLM (same pattern as vlm_utils.py)
        # Log memory BEFORE cleanup
        if torch.cuda.is_available():
            gpu_mem_before_cleanup = torch.cuda.memory_allocated() / 1024**3
            dprint(f"[VLM_CLEANUP] GPU memory BEFORE cleanup: {gpu_mem_before_cleanup:.2f} GB")
        
        # Explicitly delete model and processor to free all memory
        dprint(f"[VLM_CLEANUP] Cleaning up VLM model and processor...")
        try:
            del extender.model
            del extender.processor
            del extender
            dprint(f"[VLM_CLEANUP] ✅ Successfully deleted VLM objects")
        except Exception as e:
            dprint(f"[VLM_CLEANUP] ⚠️  Error during deletion: {e}")
        
        # Force garbage collection
        import gc
        collected = gc.collect()
        dprint(f"[VLM_CLEANUP] Garbage collected {collected} objects")
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gpu_mem_after = torch.cuda.memory_allocated() / 1024**3
            gpu_freed = gpu_mem_before_cleanup - gpu_mem_after
            dprint(f"[VLM_CLEANUP] GPU memory AFTER cleanup: {gpu_mem_after:.2f} GB")
            dprint(f"[VLM_CLEANUP] GPU memory freed: {gpu_freed:.2f} GB")
        
        dprint(f"[VLM_CLEANUP] ✅ VLM cleanup complete")
        return result
        
    except Exception as e:
        dprint(f"[VLM_PROMPTS] ERROR in VLM processing: {e}")
        traceback.print_exc()
        return result


# =============================================================================
# SHARED CORE LOGIC - Used by both join_clips_orchestrator and edit_video_orchestrator
# =============================================================================

def _extract_join_settings_from_payload(orchestrator_payload: dict) -> dict:
    """
    Extract standardized join settings from an orchestrator payload.
    
    Used by both join_clips_orchestrator and edit_video_orchestrator.
    
    Args:
        orchestrator_payload: The orchestrator_details dict
        
    Returns:
        Dict of join settings for join_clips_segment tasks
    """
    return {
        "context_frame_count": orchestrator_payload.get("context_frame_count", 8),
        "gap_frame_count": orchestrator_payload.get("gap_frame_count", 53),
        "replace_mode": orchestrator_payload.get("replace_mode", False),
        "prompt": orchestrator_payload.get("prompt", "smooth transition"),
        "negative_prompt": orchestrator_payload.get("negative_prompt", ""),
        "model": orchestrator_payload.get("model", "wan_2_2_vace_lightning_baseline_2_2_2"),
        "aspect_ratio": orchestrator_payload.get("aspect_ratio"),
        "resolution": orchestrator_payload.get("resolution"),
        "use_input_video_resolution": orchestrator_payload.get("use_input_video_resolution", False),
        "fps": orchestrator_payload.get("fps"),
        "use_input_video_fps": orchestrator_payload.get("use_input_video_fps", False),
        "phase_config": orchestrator_payload.get("phase_config"),
        "num_inference_steps": orchestrator_payload.get("num_inference_steps"),
        "guidance_scale": orchestrator_payload.get("guidance_scale"),
        "seed": orchestrator_payload.get("seed", -1),
        # LoRA parameters
        "additional_loras": orchestrator_payload.get("additional_loras", {}),
        # Keep bridging image param
        "keep_bridging_images": orchestrator_payload.get("keep_bridging_images", False),
        # Vid2vid initialization for replace mode
        "vid2vid_init_strength": orchestrator_payload.get("vid2vid_init_strength"),
        # Audio to add to final output (only used by last join)
        "audio_url": orchestrator_payload.get("audio_url"),
    }


def _check_existing_join_tasks(
    orchestrator_task_id_str: str,
    num_joins: int,
    dprint
) -> Tuple[Optional[bool], Optional[str]]:
    """
    Check for existing child tasks (idempotency check).
    
    Returns:
        (None, None) if no existing tasks or should proceed with creation
        (success: bool, message: str) if should return early (complete/failed/in-progress)
    """
    import json
    
    dprint(f"[JOIN_CORE] Checking for existing child tasks")
    existing_child_tasks = db_ops.get_orchestrator_child_tasks(orchestrator_task_id_str)
    existing_joins = existing_child_tasks.get('join_clips_segment', [])

    if not existing_joins:
        return None, None
    
    dprint(f"[JOIN_CORE] Found {len(existing_joins)} existing join tasks")

    # Check if we have the expected number
    if len(existing_joins) < num_joins:
        return None, None

    dprint(f"[JOIN_CORE] All {num_joins} join tasks already exist")

    # Check completion status
    def is_complete(task):
        # DB stores statuses as "Complete" (capitalized). Compare case-insensitively.
        return (task.get('status', '') or '').lower() == 'complete'

    def is_terminal_failure(task):
        status = task.get('status', '').lower()
        return status in ('failed', 'cancelled', 'canceled', 'error')

    all_joins_complete = all(is_complete(join) for join in existing_joins)
    any_join_failed = any(is_terminal_failure(join) for join in existing_joins)

    # If any failed, mark orchestrator as failed
    if any_join_failed:
        failed_joins = [j for j in existing_joins if is_terminal_failure(j)]
        error_msg = f"{len(failed_joins)} join task(s) failed/cancelled"
        dprint(f"[JOIN_CORE] FAILED: {error_msg}")
        return False, f"[ORCHESTRATOR_FAILED] {error_msg}"

    # If all complete, return final output
    if all_joins_complete:
        # Sort by join_index to get the last one
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

        # Extract thumbnail from final join's params
        final_params = final_join.get('task_params', {})
        if isinstance(final_params, str):
            try:
                final_params = json.loads(final_params)
            except (json.JSONDecodeError, ValueError):
                final_params = {}

        final_thumbnail = final_params.get('thumbnail_url', '')

        dprint(f"[JOIN_CORE] COMPLETE: All joins finished, final output: {final_output}")
        dprint(f"[JOIN_CORE] Final thumbnail: {final_thumbnail}")

        completion_data = json.dumps({"output_location": final_output, "thumbnail_url": final_thumbnail})
        return True, f"[ORCHESTRATOR_COMPLETE]{completion_data}"

    # Still in progress
    complete_count = sum(1 for j in existing_joins if is_complete(j))
    dprint(f"[JOIN_CORE] IDEMPOTENT: {complete_count}/{num_joins} joins complete")
    return True, f"[IDEMPOTENT] Join tasks in progress: {complete_count}/{num_joins} complete"


def _create_join_chain_tasks(
    clip_list: List[dict],
    run_id: str,
    join_settings: dict,
    per_join_settings: List[dict],
    vlm_enhanced_prompts: List[Optional[str]],
    current_run_output_dir: Path,
    orchestrator_task_id_str: str,
    orchestrator_project_id: str | None,
    orchestrator_payload: dict,
    dprint
) -> Tuple[bool, str]:
    """
    Core logic: Create chained join_clips_segment tasks.
    
    This is the shared core function used by both:
    - join_clips_orchestrator: Provides clip_list directly
    - edit_video_orchestrator: Preprocesses source video into keeper clips first
    
    Args:
        clip_list: List of clip dicts with 'url' and optional 'name' keys
        run_id: Unique run identifier
        join_settings: Base settings for all join tasks
        per_join_settings: Per-join overrides (list, one per join)
        vlm_enhanced_prompts: VLM-generated prompts (or None for each join)
        current_run_output_dir: Output directory for this run
        orchestrator_task_id_str: Orchestrator task ID
        orchestrator_project_id: Project ID for authorization
        orchestrator_payload: Full orchestrator payload for reference
        dprint: Debug print function
        
    Returns:
        (success: bool, message: str)
    """
    num_joins = len(clip_list) - 1
    
    if num_joins < 1:
        return False, "clip_list must contain at least 2 clips"
    
    dprint(f"[JOIN_CORE] Creating {num_joins} join tasks in dependency chain")

    previous_join_task_id = None
    joins_created = 0

    for idx in range(num_joins):
        clip_start = clip_list[idx]
        clip_end = clip_list[idx + 1]

        dprint(f"[JOIN_CORE] Creating join {idx}: {clip_start.get('name', 'clip')} + {clip_end.get('name', 'clip')}")

        # Merge global settings with per-join overrides
        task_join_settings = join_settings.copy()
        if idx < len(per_join_settings):
            task_join_settings.update(per_join_settings[idx])
            dprint(f"[JOIN_CORE] Applied per-join overrides for join {idx}")

        # Apply VLM-enhanced prompt if available (overrides base prompt)
        if idx < len(vlm_enhanced_prompts) and vlm_enhanced_prompts[idx] is not None:
            task_join_settings["prompt"] = vlm_enhanced_prompts[idx]
            dprint(f"[JOIN_CORE] Join {idx}: Using VLM-enhanced prompt")

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

        dprint(f"[JOIN_CORE] Submitting join {idx} to database, depends_on={previous_join_task_id}")

        # Create task with dependency chain
        actual_db_row_id = db_ops.add_task_to_db(
            task_payload=join_payload,
            task_type_str="join_clips_segment",
            dependant_on=previous_join_task_id
        )

        dprint(f"[JOIN_CORE] Join {idx} created with DB ID: {actual_db_row_id}")

        # Update for next iteration
        previous_join_task_id = actual_db_row_id
        joins_created += 1

    return True, f"Successfully enqueued {joins_created} join tasks for run {run_id}"


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
        loop_first_clip = orchestrator_payload.get("loop_first_clip", False)

        if not run_id:
            return False, "run_id is required"

        # Handle loop_first_clip: reverse the first clip and use it as the second clip
        # This creates a "boomerang" effect where the clip plays forward then backward
        if loop_first_clip:
            dprint(f"[JOIN_ORCHESTRATOR] loop_first_clip=True - will reverse first clip to create looping effect")
            
            if not clip_list or len(clip_list) < 1:
                return False, "clip_list must contain at least 1 clip when loop_first_clip=True"
            
            # Create a temp directory for the reversed video
            loop_temp_dir = Path(main_output_dir_base) / f"join_clips_run_{run_id}" / "loop_temp"
            loop_temp_dir.mkdir(parents=True, exist_ok=True)
            
            first_clip = clip_list[0]
            first_clip_url = first_clip.get("url")
            first_clip_name = first_clip.get("name", "clip_0")
            
            if not first_clip_url:
                return False, "First clip in clip_list is missing 'url' field"
            
            # Download the first clip if it's a URL
            dprint(f"[JOIN_ORCHESTRATOR] Downloading first clip for reversal: {first_clip_url[:80]}...")
            local_first_clip_path = download_video_if_url(
                first_clip_url,
                download_target_dir=loop_temp_dir,
                task_id_for_logging=orchestrator_task_id_str,
                descriptive_name="first_clip_for_loop"
            )
            
            if not local_first_clip_path or not Path(local_first_clip_path).exists():
                return False, f"Failed to download first clip: {first_clip_url}"
            
            # Reverse the video
            reversed_clip_path = loop_temp_dir / f"{first_clip_name}_reversed.mp4"
            dprint(f"[JOIN_ORCHESTRATOR] Reversing first clip...")
            
            reversed_path = reverse_video(
                local_first_clip_path,
                reversed_clip_path,
                dprint=dprint
            )
            
            if not reversed_path or not reversed_path.exists():
                return False, f"Failed to reverse first clip: {local_first_clip_path}"
            
            dprint(f"[JOIN_ORCHESTRATOR] ✅ Created reversed clip locally: {reversed_path}")
            
            # Upload reversed clip to storage for cross-worker access
            reversed_filename = f"{first_clip_name}_reversed.mp4"
            reversed_url = upload_intermediate_file_to_storage(
                local_file_path=reversed_path,
                task_id=orchestrator_task_id_str,
                filename=reversed_filename,
                dprint=dprint
            )
            
            if not reversed_url:
                # Fallback to local path if upload fails (works for single-worker setups)
                dprint(f"[JOIN_ORCHESTRATOR] ⚠️  Upload failed, using local path (may fail on multi-worker)")
                reversed_url = str(reversed_path.resolve())
            else:
                dprint(f"[JOIN_ORCHESTRATOR] ✅ Reversed clip uploaded: {reversed_url}")
            
            # Build new clip_list: [original_first_clip, reversed_first_clip]
            # The reversed clip becomes the "second" clip
            reversed_clip_dict = {
                "url": reversed_url,
                "name": f"{first_clip_name}_reversed"
            }
            
            # Override clip_list with the two clips
            clip_list = [first_clip, reversed_clip_dict]
            dprint(f"[JOIN_ORCHESTRATOR] clip_list overridden: [{first_clip_name}, {reversed_clip_dict['name']}]")

        if not clip_list or len(clip_list) < 2:
            return False, "clip_list must contain at least 2 clips"

        num_joins = len(clip_list) - 1
        dprint(f"[JOIN_ORCHESTRATOR] Processing {len(clip_list)} clips = {num_joins} join tasks")

        # Extract join settings using shared helper
        join_settings = _extract_join_settings_from_payload(orchestrator_payload)
        per_join_settings = orchestrator_payload.get("per_join_settings", [])
        output_base_dir = orchestrator_payload.get("output_base_dir", str(main_output_dir_base.resolve()))

        # Create run-specific output directory
        current_run_output_dir = Path(output_base_dir) / f"join_clips_run_{run_id}"
        current_run_output_dir.mkdir(parents=True, exist_ok=True)
        dprint(f"[JOIN_ORCHESTRATOR] Run output directory: {current_run_output_dir}")

        # === VLM PROMPT ENHANCEMENT (optional) ===
        enhance_prompt = orchestrator_payload.get("enhance_prompt", False)
        vlm_enhanced_prompts: List[Optional[str]] = [None] * num_joins
        
        if enhance_prompt:
            dprint(f"[JOIN_ORCHESTRATOR] enhance_prompt=True, generating VLM-enhanced prompts for {num_joins} joins")
            
            vlm_device = orchestrator_payload.get("vlm_device", "cuda")
            vlm_temp_dir = current_run_output_dir / "vlm_temp"
            vlm_temp_dir.mkdir(parents=True, exist_ok=True)
            
            # Get settings needed for VLM frame extraction
            replace_mode = join_settings.get("replace_mode", False)
            gap_frame_count = join_settings.get("gap_frame_count", 53)
            base_prompt = join_settings.get("prompt", "")
            fps = orchestrator_payload.get("fps", 16)
            
            try:
                # Step 1: Extract boundary frames from all clips (4 images per join)
                dprint(f"[JOIN_ORCHESTRATOR] Extracting 4 frames per join from {len(clip_list)} clips...")
                dprint(f"[JOIN_ORCHESTRATOR]   replace_mode={replace_mode}, gap_frame_count={gap_frame_count}")
                image_quads = _extract_boundary_frames_for_vlm(
                    clip_list=clip_list,
                    temp_dir=vlm_temp_dir,
                    orchestrator_task_id=orchestrator_task_id_str,
                    replace_mode=replace_mode,
                    gap_frame_count=gap_frame_count,
                    dprint=dprint
                )
                
                # Step 2: Generate VLM prompts for all joins
                dprint(f"[JOIN_ORCHESTRATOR] Running VLM batch on {len(image_quads)} quads (4 images each)...")
                vlm_enhanced_prompts = _generate_vlm_prompts_for_joins(
                    image_quads=image_quads,
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

        # === 2. IDEMPOTENCY CHECK (using shared helper) ===
        idempotency_result, idempotency_message = _check_existing_join_tasks(
            orchestrator_task_id_str, num_joins, dprint
        )
        if idempotency_result is not None:
            return idempotency_result, idempotency_message

        # === 3. CREATE JOIN CHAIN (using shared core function) ===
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
        
        dprint(f"[JOIN_ORCHESTRATOR] {message}")
        return success, message

    except Exception as e:
        msg = f"Failed during join orchestration: {e}"
        dprint(f"[JOIN_ORCHESTRATOR] ERROR: {msg}")
        traceback.print_exc()
        return False, msg
