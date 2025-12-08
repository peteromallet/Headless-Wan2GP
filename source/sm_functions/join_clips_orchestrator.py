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
from ..common_utils import download_video_if_url, get_video_frame_count_and_fps
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


def _generate_join_transition_prompt(
    start_image_path: str,
    end_image_path: str,
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
    
    Args:
        start_image_path: Path to the starting frame
        end_image_path: Path to the ending frame
        base_prompt: Base prompt for context
        num_frames: Number of transition frames
        fps: Frames per second
        extender: QwenPromptExpander instance (reused for batch efficiency)
        dprint: Debug print function
        
    Returns:
        Generated 3-sentence prompt
    """
    from PIL import Image
    
    # Load and combine images side by side
    start_img = Image.open(start_image_path).convert("RGB")
    end_img = Image.open(end_image_path).convert("RGB")
    
    combined_width = start_img.width + end_img.width
    combined_height = max(start_img.height, end_img.height)
    combined_img = Image.new('RGB', (combined_width, combined_height))
    combined_img.paste(start_img, (0, 0))
    combined_img.paste(end_img, (start_img.width, 0))
    
    # Calculate duration
    duration_text = ""
    if num_frames and fps:
        duration_seconds = num_frames / fps
        duration_text = f"This transition occurs over approximately {duration_seconds:.1f} seconds ({num_frames} frames at {fps} FPS)."
    
    base_prompt_text = base_prompt if base_prompt and base_prompt.strip() else "a video sequence"
    
    query = f"""You are viewing two images side by side: the left image shows the starting frame, and the right image shows the ending frame of a video sequence.

{duration_text} Your goal is to create a THREE-SENTENCE prompt that describes this transition. Use the user's context if provided: '{base_prompt_text}'

YOUR RESPONSE MUST FOLLOW THIS EXACT STRUCTURE:

SENTENCE 1 (MAIN MOTION): Describe the primary action or movement happening - what are characters doing (running, walking, turning), what is the camera doing (panning, zooming, tracking), what major changes occur between the frames.

SENTENCE 2 (VISUAL STYLE): Describe the unique visual style of the footage - is it photorealistic, anime, illustrated, cel-shaded, watercolor, 3D rendered, vintage film, etc. Include the overall mood and color palette.

SENTENCE 3 (SCENE DETAILS): Describe the specific elements and details in the scene - objects, environment features, particles (dust, snow, petals), lighting effects, background elements, and any small details that make the scene rich.

Examples following this structure:

- "A woman runs through an open field, her arms pumping as the camera tracks alongside her moving figure. The scene has a dreamy, soft-focus aesthetic with warm golden hour lighting and muted pastel tones reminiscent of indie film photography. Tall grass sways in waves around her feet, dandelion seeds drift lazily through the air, and distant mountains are silhouetted against the glowing orange sky."

- "The camera slowly zooms into the character's face as their expression shifts from surprise to determination. The visual style is vibrant anime with bold outlines, cel-shading, and dramatic speed lines radiating outward. Cherry blossom petals swirl in the foreground, glowing magical particles orbit their raised hand, and detailed mechanical gears spin in the steampunk background."

- "A bat spreads its wings and takes flight from a gnarled tree branch while the camera follows its ascending path. The footage has a dark gothic illustration style with crosshatching textures and deep purple and black color schemes. A full moon glows behind thin clouds, bare twisted branches frame the composition, and tiny moths flutter near flickering lantern light below."

Now create your THREE-SENTENCE description (MOTION, STYLE, DETAILS) based on what you see in the frames."""

    system_prompt = "You are a video description assistant. Respond with EXACTLY THREE SENTENCES: 1) Main motion/action, 2) Visual style/aesthetic, 3) Scene details/elements. Be specific and descriptive."
    
    result = extender.extend_with_img(
        prompt=query,
        system_prompt=system_prompt,
        image=combined_img
    )
    
    return result.prompt.strip()


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
    
    Uses custom join_clips prompt structure:
    1. Main motion (running, walking, camera movement)
    2. Visual style (anime, illustrated, photorealistic)
    3. Scene details (flowers, particles, environment)
    
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
    import sys
    from pathlib import Path
    import torch
    
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
    
    try:
        # Add Wan2GP to path for imports
        wan_dir = Path(__file__).parent.parent.parent / "Wan2GP"
        if str(wan_dir) not in sys.path:
            sys.path.insert(0, str(wan_dir))
        
        from Wan2GP.wan.utils.prompt_extend import QwenPromptExpander
        from ..vlm_utils import download_qwen_vlm_if_needed
        
        # Log memory before loading (same as vlm_utils.py)
        if torch.cuda.is_available():
            gpu_mem_before = torch.cuda.memory_allocated() / 1024**3
            dprint(f"[VLM_PROMPTS] GPU memory BEFORE VLM load: {gpu_mem_before:.2f} GB")
        
        # Initialize VLM ONCE for all pairs (batch efficiency)
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
        
        # Process each pair
        for i, (start_path, end_path) in enumerate(valid_pairs):
            idx = valid_indices[i]
            try:
                dprint(f"[VLM_PROMPTS] Processing join {idx} ({i+1}/{len(valid_pairs)})...")
                
                enhanced = _generate_join_transition_prompt(
                    start_image_path=start_path,
                    end_image_path=end_path,
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
                # Continue with other pairs
        
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
        "blend_frames": orchestrator_payload.get("blend_frames", 3),
        "prompt": orchestrator_payload.get("prompt", "smooth transition"),
        "negative_prompt": orchestrator_payload.get("negative_prompt", ""),
        "model": orchestrator_payload.get("model", "wan_2_2_vace_lightning_baseline_2_2_2"),
        "regenerate_anchors": orchestrator_payload.get("regenerate_anchors", True),
        "num_anchor_frames": orchestrator_payload.get("num_anchor_frames", 3),
        "aspect_ratio": orchestrator_payload.get("aspect_ratio"),
        "resolution": orchestrator_payload.get("resolution"),
        "phase_config": orchestrator_payload.get("phase_config"),
        "num_inference_steps": orchestrator_payload.get("num_inference_steps"),
        "guidance_scale": orchestrator_payload.get("guidance_scale"),
        "seed": orchestrator_payload.get("seed", -1),
        # LoRA parameters
        "additional_loras": orchestrator_payload.get("additional_loras", {}),
        # Keep bridging image param
        "keep_bridging_images": orchestrator_payload.get("keep_bridging_images", False),
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

        if not clip_list or len(clip_list) < 2:
            return False, "clip_list must contain at least 2 clips"

        if not run_id:
            return False, "run_id is required"

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
