"""
Travel Batch Handler

Handles batched generation of multiple travel segments in a single operation for efficiency.
This reduces model loading overhead and improves GPU utilization for short multi-image journeys.

Key Features:
- Combines multiple segments into single generation task
- Creates composite guide videos spanning multiple target images
- Applies sophisticated masking for proper frame control
- Splits batch output back into individual segments for stitching
- Maintains full compatibility with existing pipeline
"""

import json
import math
import shutil
import traceback
from pathlib import Path
import time
import subprocess
import uuid
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional

# Import structured logging
from ..logging_utils import travel_logger

# Import shared utilities
from ..common_utils import (
    generate_unique_task_id as sm_generate_unique_task_id,
    get_video_frame_count_and_fps as sm_get_video_frame_count_and_fps,
    create_color_frame as sm_create_color_frame,
    parse_resolution as sm_parse_resolution,
    download_image_if_url as sm_download_image_if_url,
    prepare_output_path,
    prepare_output_path_with_upload,
    upload_and_get_final_output_location,
    snap_resolution_to_model_grid,
    ensure_valid_prompt,
    ensure_valid_negative_prompt,
    wait_for_file_stable as sm_wait_for_file_stable,
    create_mask_video_from_inactive_indices,
)
from ..video_utils import (
    extract_frames_from_video as sm_extract_frames_from_video,
    create_video_from_frames_list as sm_create_video_from_frames_list,
    create_guide_video_for_travel_segment as sm_create_guide_video_for_travel_segment,
)
from .. import db_operations as db_ops
from ..batch_optimizer import create_batch_mask_analysis, BatchGroup


def _handle_travel_batch_task(
    task_params_from_db: dict, 
    main_output_dir_base: Path, 
    batch_task_id_str: str, 
    *, 
    process_single_task, 
    dprint
) -> Tuple[bool, str]:
    """
    Handle batched generation of multiple travel segments.
    
    This function:
    1. Creates a composite guide video spanning all segments in the batch
    2. Generates appropriate mask video for frame control
    3. Runs single WGP generation for the entire batch
    4. Splits the generated video back into individual segments
    5. Saves segments with proper naming for stitcher compatibility
    
    Args:
        task_params_from_db: Batch task parameters from orchestrator
        main_output_dir_base: Base output directory
        batch_task_id_str: Unique batch task ID
        process_single_task: Function to process WGP generation
        dprint: Debug print function
    
    Returns:
        Tuple of (success_bool, message_str)
    """
    travel_logger.essential("Starting travel batch task", task_id=batch_task_id_str)
    travel_logger.debug(f"Batch params: {json.dumps(task_params_from_db, default=str, indent=2)[:1000]}...", task_id=batch_task_id_str)
    
    batch_params = task_params_from_db
    generation_success = False
    
    try:
        # --- 1. Extract batch parameters ---
        orchestrator_task_id_ref = batch_params.get("orchestrator_task_id_ref")
        orchestrator_run_id = batch_params.get("orchestrator_run_id")
        batch_index = batch_params.get("batch_index")
        segment_indices = batch_params.get("segment_indices_in_batch", [])
        total_batch_frames = batch_params.get("total_batch_frames")
        internal_overlaps = batch_params.get("internal_overlaps", [])
        
        travel_logger.debug(f"Batch {batch_index}: Processing segments {segment_indices} -> {total_batch_frames} frames", task_id=batch_task_id_str)
        
        if not all([orchestrator_task_id_ref, orchestrator_run_id, segment_indices, total_batch_frames]):
            msg = f"Batch task {batch_task_id_str} missing critical parameters"
            travel_logger.error(msg, task_id=batch_task_id_str)
            return False, msg
        
        # Get full orchestrator payload for context
        full_orchestrator_payload = batch_params.get("full_orchestrator_payload", {})
        batching_analysis = batch_params.get("batching_analysis")
        
        # Set up processing directory
        current_run_base_output_dir_str = batch_params.get("current_run_base_output_dir")
        current_run_base_output_dir = Path(current_run_base_output_dir_str)
        batch_processing_dir = current_run_base_output_dir
        batch_processing_dir.mkdir(parents=True, exist_ok=True)
        
        travel_logger.debug(f"Batch processing directory: {batch_processing_dir.resolve()}", task_id=batch_task_id_str)
        
        # --- 2. Create composite guide video ---
        parsed_res_wh_str = batch_params["parsed_resolution_wh"]
        parsed_res_wh = snap_resolution_to_model_grid(sm_parse_resolution(parsed_res_wh_str))
        fps_helpers = batch_params.get("fps_helpers", 16)
        
        composite_guide_path = _create_composite_guide_video_for_batch(
            batch_params, 
            parsed_res_wh, 
            fps_helpers,
            batch_processing_dir,
            task_id=batch_task_id_str,
            dprint=dprint
        )
        
        # --- 3. Create batch mask video ---
        batch_mask_path = _create_batch_mask_video(
            batch_params,
            batching_analysis,
            parsed_res_wh,
            fps_helpers,
            batch_processing_dir,
            task_id=batch_task_id_str,
            dprint=dprint
        )
        
        # --- 4. Generate batch video using WGP ---
        batch_video_path = _generate_batch_video(
            batch_params,
            composite_guide_path,
            batch_mask_path,
            total_batch_frames,
            parsed_res_wh,
            process_single_task,
            batch_processing_dir,
            task_id=batch_task_id_str,
            dprint=dprint
        )
        
        if not batch_video_path:
            msg = f"Batch video generation failed for task {batch_task_id_str}"
            travel_logger.error(msg, task_id=batch_task_id_str)
            return False, msg
        
        # --- 5. Split batch video into individual segments ---
        segment_paths = _split_batch_video_into_segments(
            batch_video_path,
            segment_indices,
            internal_overlaps,
            batch_params,
            batch_processing_dir,
            task_id=batch_task_id_str,
            dprint=dprint
        )
        
        if not segment_paths or len(segment_paths) != len(segment_indices):
            msg = f"Failed to split batch video into {len(segment_indices)} segments"
            travel_logger.error(msg, task_id=batch_task_id_str)
            return False, msg
        
        # --- 6. Save segments with proper naming for stitcher ---
        saved_segment_count = 0
        for seg_idx, segment_path in zip(segment_indices, segment_paths):
            if segment_path and Path(segment_path).exists():
                # Create final segment name matching stitcher expectations
                final_segment_name = f"{orchestrator_run_id}_seg{seg_idx:02d}_output.mp4"
                final_segment_path, db_location = prepare_output_path_with_upload(
                    task_id=batch_task_id_str,
                    filename=final_segment_name,
                    main_output_dir_base=batch_processing_dir,
                    dprint=dprint
                )
                
                # Move segment to final location
                shutil.move(str(segment_path), str(final_segment_path))
                
                # Upload and get final DB location
                final_db_location = upload_and_get_final_output_location(
                    final_segment_path,
                    final_segment_name,
                    db_location,
                    dprint=dprint
                )
                
                # Register segment completion in database
                # Note: This creates individual segment task records for stitcher compatibility
                segment_task_id = sm_generate_unique_task_id(f"batch_seg_{orchestrator_run_id}_{seg_idx:02d}_")
                db_ops.update_task_status(
                    task_id=segment_task_id,
                    status=db_ops.STATUS_COMPLETED,
                    output_location=final_db_location,
                    message=f"Segment {seg_idx} from batch {batch_index}"
                )
                
                saved_segment_count += 1
                travel_logger.debug(f"Saved segment {seg_idx} to {final_db_location}", task_id=batch_task_id_str)
        
        if saved_segment_count == len(segment_indices):
            generation_success = True
            msg = f"Batch {batch_index} completed: {saved_segment_count} segments generated and saved"
            travel_logger.essential(msg, task_id=batch_task_id_str)
        else:
            msg = f"Batch {batch_index} partially failed: {saved_segment_count}/{len(segment_indices)} segments saved"
            travel_logger.error(msg, task_id=batch_task_id_str)
        
        return generation_success, msg
        
    except Exception as e:
        error_msg = f"Batch task {batch_task_id_str} failed: {str(e)}"
        travel_logger.error(error_msg, task_id=batch_task_id_str)
        travel_logger.debug(traceback.format_exc(), task_id=batch_task_id_str)
        
        # Notify orchestrator of batch failure
        if 'orchestrator_task_id_ref' in locals() and orchestrator_task_id_ref:
            try:
                db_ops.update_task_status(
                    orchestrator_task_id_ref,
                    db_ops.STATUS_FAILED,
                    error_msg[:500]
                )
                travel_logger.debug(f"Marked orchestrator task {orchestrator_task_id_ref} as FAILED", task_id=batch_task_id_str)
            except Exception as e_orch:
                travel_logger.debug(f"Could not update orchestrator status: {e_orch}", task_id=batch_task_id_str)
        
        return False, error_msg


def _create_composite_guide_video_for_batch(
    batch_params: dict,
    parsed_res_wh: Tuple[int, int],
    fps_helpers: int,
    output_dir: Path,
    task_id: str,
    dprint
) -> Optional[Path]:
    """
    Create a composite guide video spanning all segments in the batch.
    
    This creates a single guide video that interpolates through all target images
    in the batch, providing proper guidance for the batched generation.
    """
    try:
        segment_indices = batch_params["segment_indices_in_batch"]
        total_batch_frames = batch_params["total_batch_frames"]
        full_orchestrator_payload = batch_params["full_orchestrator_payload"]
        
        travel_logger.debug(f"Creating composite guide for segments {segment_indices} -> {total_batch_frames} frames", task_id=task_id)
        
        # Get input images for the segments in this batch
        input_images_resolved = full_orchestrator_payload.get("input_image_paths_resolved", [])
        if not input_images_resolved:
            travel_logger.warning("No input images available for composite guide", task_id=task_id)
            return None
        
        # Create composite guide video filename
        guide_filename = f"batch_{batch_params['batch_index']}_composite_guide.mp4"
        guide_output_path, _ = prepare_output_path(
            task_id=task_id,
            filename=guide_filename,
            main_output_dir_base=output_dir
        )
        
        # Create frame sequence for composite guide
        gray_frame = sm_create_color_frame(parsed_res_wh, (128, 128, 128))
        guide_frames = [gray_frame.copy() for _ in range(total_batch_frames)]
        
        # Calculate frame positions for each segment's target image
        current_frame_pos = 0
        internal_overlaps = batch_params.get("internal_overlaps", [])
        
        for i, seg_idx in enumerate(segment_indices):
            # Get segment frame count from orchestrator
            segment_frames_expanded = full_orchestrator_payload.get("segment_frames_expanded", [])
            if seg_idx < len(segment_frames_expanded):
                segment_frame_count = segment_frames_expanded[seg_idx]
            else:
                travel_logger.warning(f"No frame count for segment {seg_idx}", task_id=task_id)
                continue
            
            # Calculate target image index
            target_image_idx = seg_idx + 1 if seg_idx + 1 < len(input_images_resolved) else seg_idx
            target_image_path = input_images_resolved[target_image_idx]
            
            # Download image if URL
            segment_image_download_dir_str = batch_params.get("segment_image_download_dir")
            if segment_image_download_dir_str:
                segment_image_download_dir = Path(segment_image_download_dir_str)
            else:
                segment_image_download_dir = output_dir
            
            target_image_path = sm_download_image_if_url(
                target_image_path,
                segment_image_download_dir,
                task_id,
                debug_mode=batch_params.get("debug_mode_enabled", False),
                descriptive_name=f"batch_target_{seg_idx}"
            )
            
            # Load and resize target image
            target_frame = _load_and_resize_image(target_image_path, parsed_res_wh, task_id)
            if target_frame is None:
                continue
            
            # Calculate where to place this target image in the composite guide
            if i == 0:
                # First segment: place target at the end of the segment
                target_frame_pos = current_frame_pos + segment_frame_count - 1
            else:
                # Subsequent segments: account for overlap
                overlap = internal_overlaps[i-1] if i-1 < len(internal_overlaps) else 0
                target_frame_pos = current_frame_pos + segment_frame_count - overlap - 1
            
            # Place target frame at calculated position
            if 0 <= target_frame_pos < len(guide_frames):
                guide_frames[target_frame_pos] = target_frame.copy()
                travel_logger.debug(f"Placed target image for segment {seg_idx} at frame {target_frame_pos}", task_id=task_id)
            
            # Advance frame position
            if i == 0:
                current_frame_pos += segment_frame_count
            else:
                overlap = internal_overlaps[i-1] if i-1 < len(internal_overlaps) else 0
                current_frame_pos += segment_frame_count - overlap
        
        # Create guide video from frames
        created_guide_path = sm_create_video_from_frames_list(
            guide_frames, 
            guide_output_path, 
            fps_helpers, 
            parsed_res_wh
        )
        
        if created_guide_path and created_guide_path.exists():
            travel_logger.debug(f"Composite guide video created: {created_guide_path}", task_id=task_id)
            return created_guide_path
        else:
            travel_logger.error("Failed to create composite guide video", task_id=task_id)
            return None
            
    except Exception as e:
        travel_logger.error(f"Error creating composite guide: {e}", task_id=task_id)
        return None


def _create_batch_mask_video(
    batch_params: dict,
    batching_analysis,
    parsed_res_wh: Tuple[int, int],
    fps_helpers: int,
    output_dir: Path,
    task_id: str,
    dprint
) -> Optional[Path]:
    """
    Create mask video for batched generation with proper frame control.
    
    This creates a sophisticated mask that:
    - Anchors key frames to input images (black/inactive)
    - Allows free generation in transition zones (white/active) 
    - Handles overlaps correctly between segments
    """
    try:
        total_batch_frames = batch_params["total_batch_frames"]
        batch_index = batch_params["batch_index"]
        
        travel_logger.debug(f"Creating batch mask for {total_batch_frames} frames", task_id=task_id)
        
        # Find the corresponding batch group from analysis
        batch_group = None
        if batching_analysis and hasattr(batching_analysis, 'batch_groups'):
            for bg in batching_analysis.batch_groups:
                if bg.batch_index == batch_index:
                    batch_group = bg
                    break
        
        if not batch_group:
            travel_logger.warning("No batch group found for mask analysis, using basic mask", task_id=task_id)
            # Create basic mask - mostly active with anchored endpoints
            inactive_indices = {0, total_batch_frames - 1}
        else:
            # Use sophisticated mask analysis
            full_orchestrator_payload = batch_params["full_orchestrator_payload"]
            segment_frames_expanded = full_orchestrator_payload.get("segment_frames_expanded", [])
            frame_overlap_expanded = full_orchestrator_payload.get("frame_overlap_expanded", [])
            
            mask_analysis = create_batch_mask_analysis(
                batch_group,
                segment_frames_expanded,
                frame_overlap_expanded,
                task_id=task_id
            )
            
            # Extract inactive indices from mask analysis
            inactive_indices = set()
            for frame_idx, mask_value in mask_analysis["frame_mask_map"].items():
                if mask_value == 0.0:  # Anchored frames
                    inactive_indices.add(frame_idx)
            
            travel_logger.debug(f"Mask analysis: {len(inactive_indices)} anchored frames out of {total_batch_frames}", task_id=task_id)
        
        # Create mask video filename
        mask_filename = f"batch_{batch_index}_mask.mp4"
        mask_output_path, _ = prepare_output_path(
            task_id=task_id,
            filename=mask_filename,
            main_output_dir_base=output_dir
        )
        
        # Create mask video using shared utility
        mask_video_path = create_mask_video_from_inactive_indices(
            inactive_indices=inactive_indices,
            total_frames=total_batch_frames,
            resolution_wh=parsed_res_wh,
            fps=fps_helpers,
            output_path=mask_output_path,
            task_id_for_logging=task_id
        )
        
        if mask_video_path and Path(mask_video_path).exists():
            travel_logger.debug(f"Batch mask video created: {mask_video_path}", task_id=task_id)
            return Path(mask_video_path)
        else:
            travel_logger.error("Failed to create batch mask video", task_id=task_id)
            return None
            
    except Exception as e:
        travel_logger.error(f"Error creating batch mask: {e}", task_id=task_id)
        return None


def _generate_batch_video(
    batch_params: dict,
    guide_path: Optional[Path],
    mask_path: Optional[Path],
    total_frames: int,
    parsed_res_wh: Tuple[int, int],
    process_single_task,
    output_dir: Path,
    task_id: str,
    dprint
) -> Optional[str]:
    """
    Generate the batch video using WGP with composite guide and mask.
    """
    try:
        # Create WGP payload for batch generation
        wgp_task_id = sm_generate_unique_task_id(f"wgp_batch_{task_id[:8]}_")
        
        wgp_payload = {
            "task_id": wgp_task_id,
            "model": batch_params["model_name"],
            "prompt": batch_params.get("combined_prompt", "cinematic travel sequence"),
            "negative_prompt": batch_params.get("combined_negative_prompt", "blurry, low quality"),
            "resolution": f"{parsed_res_wh[0]}x{parsed_res_wh[1]}",
            "frames": total_frames,
            "seed": batch_params.get("seed_to_use", 12345),
            "use_causvid_lora": batch_params.get("use_causvid_lora", False),
            "use_lighti2x_lora": batch_params.get("use_lighti2x_lora", False),
            "apply_reward_lora": batch_params.get("apply_reward_lora", False),
            "cfg_star_switch": batch_params.get("cfg_star_switch", 0),
            "cfg_zero_step": batch_params.get("cfg_zero_step", -1),
        }
        
        # Add guide video if available
        if guide_path and guide_path.exists():
            wgp_payload["video_guide_path"] = str(guide_path.resolve())
            travel_logger.debug(f"Using composite guide: {guide_path}", task_id=task_id)
        
        # Add mask video if available  
        if mask_path and mask_path.exists():
            wgp_payload["video_mask"] = str(mask_path.resolve())
            travel_logger.debug(f"Using batch mask: {mask_path}", task_id=task_id)
        
        # Apply parameter overrides if specified
        params_override = batch_params.get("params_json_str_override")
        if params_override:
            try:
                additional_params = json.loads(params_override)
                # Protect critical parameters
                for protected_param in ["frames", "resolution", "task_id"]:
                    additional_params.pop(protected_param, None)
                wgp_payload.update(additional_params)
            except Exception as e_override:
                travel_logger.warning(f"Failed to apply parameter override: {e_override}", task_id=task_id)
        
        travel_logger.debug(f"Invoking WGP for batch generation: {total_frames} frames", task_id=task_id)
        
        # Process the batch generation
        batch_video_path = process_single_task(wgp_payload, "wgp")
        
        if batch_video_path:
            travel_logger.debug(f"Batch video generated: {batch_video_path}", task_id=task_id)
            
            # Verify frame count
            actual_frames, _ = sm_get_video_frame_count_and_fps(batch_video_path)
            if actual_frames != total_frames:
                travel_logger.warning(f"Frame count mismatch: expected {total_frames}, got {actual_frames}", task_id=task_id)
            
            return batch_video_path
        else:
            travel_logger.error("WGP batch generation returned no output", task_id=task_id)
            return None
            
    except Exception as e:
        travel_logger.error(f"Error in batch video generation: {e}", task_id=task_id)
        return None


def _split_batch_video_into_segments(
    batch_video_path: str,
    segment_indices: List[int],
    internal_overlaps: List[int],
    batch_params: dict,
    output_dir: Path,
    task_id: str,
    dprint
) -> List[Optional[str]]:
    """
    Split the generated batch video back into individual segments.
    
    This carefully extracts the correct frame ranges for each segment,
    accounting for overlaps and maintaining frame count accuracy.
    """
    try:
        travel_logger.debug(f"Splitting batch video into {len(segment_indices)} segments", task_id=task_id)
        
        # Extract all frames from batch video
        all_batch_frames = sm_extract_frames_from_video(batch_video_path, dprint_func=dprint)
        if not all_batch_frames:
            travel_logger.error("Failed to extract frames from batch video", task_id=task_id)
            return [None] * len(segment_indices)
        
        travel_logger.debug(f"Extracted {len(all_batch_frames)} frames from batch video", task_id=task_id)
        
        # Get segment frame counts from orchestrator
        full_orchestrator_payload = batch_params["full_orchestrator_payload"]
        segment_frames_expanded = full_orchestrator_payload.get("segment_frames_expanded", [])
        
        segment_paths = []
        current_frame_idx = 0
        
        for i, seg_idx in enumerate(segment_indices):
            # Get expected frame count for this segment
            if seg_idx < len(segment_frames_expanded):
                segment_frame_count = segment_frames_expanded[seg_idx]
            else:
                travel_logger.error(f"No frame count available for segment {seg_idx}", task_id=task_id)
                segment_paths.append(None)
                continue
            
            # Calculate frame range for this segment
            if i == 0:
                # First segment: take all frames
                start_frame = current_frame_idx
                end_frame = current_frame_idx + segment_frame_count
            else:
                # Subsequent segments: account for overlap
                overlap = internal_overlaps[i-1] if i-1 < len(internal_overlaps) else 0
                start_frame = current_frame_idx - overlap
                end_frame = current_frame_idx + segment_frame_count - overlap
            
            # Extract frames for this segment
            if start_frame >= 0 and end_frame <= len(all_batch_frames):
                segment_frames = all_batch_frames[start_frame:end_frame]
                travel_logger.debug(f"Segment {seg_idx}: frames {start_frame}:{end_frame} ({len(segment_frames)} frames)", task_id=task_id)
            else:
                travel_logger.error(f"Invalid frame range for segment {seg_idx}: {start_frame}:{end_frame}", task_id=task_id)
                segment_paths.append(None)
                continue
            
            # Create segment video
            segment_filename = f"batch_segment_{seg_idx:02d}.mp4"
            segment_output_path, _ = prepare_output_path(
                task_id=task_id,
                filename=segment_filename,
                main_output_dir_base=output_dir
            )
            
            fps_helpers = batch_params.get("fps_helpers", 16)
            parsed_res_wh_str = batch_params["parsed_resolution_wh"]
            parsed_res_wh = snap_resolution_to_model_grid(sm_parse_resolution(parsed_res_wh_str))
            
            created_segment_path = sm_create_video_from_frames_list(
                segment_frames,
                segment_output_path,
                fps_helpers,
                parsed_res_wh
            )
            
            if created_segment_path and created_segment_path.exists():
                segment_paths.append(str(created_segment_path))
                travel_logger.debug(f"Created segment {seg_idx}: {created_segment_path}", task_id=task_id)
            else:
                travel_logger.error(f"Failed to create segment video for {seg_idx}", task_id=task_id)
                segment_paths.append(None)
            
            # Advance frame position
            if i == 0:
                current_frame_idx += segment_frame_count
            else:
                overlap = internal_overlaps[i-1] if i-1 < len(internal_overlaps) else 0
                current_frame_idx += segment_frame_count - overlap
        
        successful_segments = sum(1 for p in segment_paths if p is not None)
        travel_logger.debug(f"Successfully split {successful_segments}/{len(segment_indices)} segments", task_id=task_id)
        
        return segment_paths
        
    except Exception as e:
        travel_logger.error(f"Error splitting batch video: {e}", task_id=task_id)
        return [None] * len(segment_indices)


def _load_and_resize_image(image_path: str, target_resolution: Tuple[int, int], task_id: str):
    """Load and resize an image to target resolution."""
    try:
        import cv2
        import numpy as np
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            travel_logger.warning(f"Could not load image: {image_path}", task_id=task_id)
            return None
        
        # Resize to target resolution
        target_width, target_height = target_resolution
        resized_img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
        
        return resized_img
        
    except Exception as e:
        travel_logger.error(f"Error loading/resizing image {image_path}: {e}", task_id=task_id)
        return None
