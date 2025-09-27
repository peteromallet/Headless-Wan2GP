"""
Travel Segment Processor - Shared Logic for Travel Segment Generation

This module contains the shared logic for processing travel segments that was previously
duplicated between travel_between_images.py and worker.py. By extracting this common
functionality, we eliminate ~500 lines of code duplication and ensure consistent
behavior across both execution paths.

Key Components:
- Guide video creation for VACE models
- Mask video creation for frame control  
- Video prompt type construction for VACE compatibility
- Parameter precedence handling (user > model preset > system defaults)

This refactoring addresses the maintenance burden where every bug fix or feature
update had to be implemented twice in nearly identical code.
"""

import json
import uuid
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, List, Any, Tuple, Callable
import traceback
import uuid
from datetime import datetime

# Import shared utilities
from .common_utils import (
    ensure_valid_prompt,
    ensure_valid_negative_prompt,
    create_mask_video_from_inactive_indices,
    get_video_frame_count_and_fps
)
from .video_utils import create_guide_video_for_travel_segment as sm_create_guide_video_for_travel_segment
from . import db_operations as db_ops


@dataclass
class TravelSegmentContext:
    """Contains all parameters needed for travel segment processing."""
    task_id: str
    segment_idx: int
    model_name: str
    total_frames_for_segment: int
    parsed_res_wh: Tuple[int, int]
    segment_processing_dir: Path
    full_orchestrator_payload: Dict[str, Any]
    segment_params: Dict[str, Any]
    mask_active_frames: bool
    debug_enabled: bool
    dprint: Callable = print


class TravelSegmentProcessor:
    """
    Shared processor for travel segment generation logic.
    
    Eliminates code duplication between blocking and queue-based handlers
    by providing a single implementation of guide video creation, mask video 
    creation, and video_prompt_type construction.
    """
    
    def __init__(self, ctx: TravelSegmentContext):
        self.ctx = ctx
        self.is_vace_model = self._detect_vace_model()
    
    def _detect_vace_model(self) -> bool:
        """Detect if this is a VACE model that requires guide videos."""
        model_name = self.ctx.model_name.lower()
        
        # Standard VACE model detection logic
        vace_indicators = ["vace", "controlnet", "cocktail", "lightning"]
        is_vace = any(indicator in model_name for indicator in vace_indicators)
        
        self.ctx.dprint(f"[VACE_DEBUG] Seg {self.ctx.segment_idx}: Model '{self.ctx.model_name}' -> is_vace_model = {is_vace}")
        return is_vace
    
    def create_guide_video(self) -> Optional[Path]:
        """
        Create guide video for VACE models or debug mode.
        
        Returns:
            Path to created guide video, or None if not created/failed
        """
        ctx = self.ctx
        
        # Always create guide video for VACE models (required for functionality)
        # For non-VACE models, only create in debug mode
        if not ctx.debug_enabled and not self.is_vace_model:
            ctx.dprint(f"Task {ctx.task_id}: Debug mode disabled and non-VACE model, skipping guide video creation")
            return None
        
        if self.is_vace_model and not ctx.debug_enabled:
            ctx.dprint(f"Task {ctx.task_id}: VACE model detected, creating guide video (REQUIRED for VACE functionality)")
        
        try:
            # Generate unique guide video filename
            timestamp_short = datetime.now().strftime("%H%M%S")
            unique_suffix = uuid.uuid4().hex[:6]
            guide_video_filename = f"seg{ctx.segment_idx:02d}_vace_guide_{timestamp_short}_{unique_suffix}.mp4"
            guide_video_base_name = f"seg{ctx.segment_idx:02d}_vace_guide_{timestamp_short}_{unique_suffix}"
            guide_video_final_path = ctx.segment_processing_dir / guide_video_filename
            
            # Get previous segment video for guide creation
            path_to_previous_segment_video_output_for_guide = self._get_previous_segment_video()
            
            # Prepare input images for guide creation
            input_images_resolved_for_guide = self._prepare_input_images_for_guide()
            
            # Determine segment positioning
            is_first_segment = ctx.segment_params.get("is_first_segment", ctx.segment_idx == 0)
            is_first_segment_from_scratch = is_first_segment and not ctx.full_orchestrator_payload.get("continue_from_video_resolved_path")
            
            # Calculate end anchor image index - use consolidated end anchor if available
            consolidated_end_anchor = ctx.segment_params.get("consolidated_end_anchor_idx")
            if consolidated_end_anchor is not None:
                end_anchor_img_path_str_idx = consolidated_end_anchor
                ctx.dprint(f"[CONSOLIDATED_SEGMENT] Using consolidated end anchor index {consolidated_end_anchor} for segment {ctx.segment_idx}")
            else:
                end_anchor_img_path_str_idx = ctx.segment_idx + 1
            
            # Create guide video using shared function
            guide_video_path = sm_create_guide_video_for_travel_segment(
                segment_idx_for_logging=ctx.segment_idx,
                end_anchor_image_index=end_anchor_img_path_str_idx,
                is_first_segment_from_scratch=is_first_segment_from_scratch,
                total_frames_for_segment=ctx.total_frames_for_segment,
                parsed_res_wh=ctx.parsed_res_wh,
                fps_helpers=ctx.full_orchestrator_payload.get("fps_helpers", 16),
                input_images_resolved_for_guide=input_images_resolved_for_guide,
                path_to_previous_segment_video_output_for_guide=path_to_previous_segment_video_output_for_guide,
                output_target_dir=ctx.segment_processing_dir,
                guide_video_base_name=guide_video_base_name,
                segment_image_download_dir=ctx.segment_processing_dir,  # Use processing dir for downloads
                task_id_for_logging=ctx.task_id,
                full_orchestrator_payload=ctx.full_orchestrator_payload,
                segment_params=ctx.segment_params,
                single_image_journey=False,  # Travel segments are not single image journeys
                predefined_output_path=guide_video_final_path,
                dprint=ctx.dprint
            )
            
            if guide_video_path and Path(guide_video_path).exists():
                ctx.dprint(f"[GUIDE_DEBUG] Successfully created guide video: {guide_video_path}")
                return Path(guide_video_path)
            else:
                ctx.dprint(f"[GUIDE_ERROR] Guide video creation returned: {guide_video_path}")
                
                # For VACE models, guide video is essential
                if self.is_vace_model:
                    raise ValueError(f"VACE model '{ctx.model_name}' requires guide video but creation failed")
                
                return None
                
        except Exception as e_guide:
            ctx.dprint(f"[GUIDE_ERROR] Guide video creation failed: {e_guide}")
            traceback.print_exc()
            
            # For VACE models, if guide creation fails, we cannot proceed
            if self.is_vace_model:
                raise ValueError(f"VACE model '{ctx.model_name}' requires guide video but creation failed: {e_guide}")
            
            return None
    
    def create_mask_video(self) -> Optional[Path]:
        """
        Create mask video for frame control.
        
        Returns:
            Path to created mask video, or None if not created/failed
        """
        ctx = self.ctx
        
        if not ctx.mask_active_frames:
            ctx.dprint(f"Task {ctx.task_id}: mask_active_frames disabled, skipping mask video creation")
            return None
        
        try:
            # Determine which frame indices should be kept (inactive = black)
            inactive_indices = set()
            
            # Define overlap_count for consistent logging
            frame_overlap_from_previous = ctx.segment_params.get("frame_overlap_from_previous", 0)
            overlap_count = max(0, int(frame_overlap_from_previous))
            
            # 1) Frames reused from the previous segment (overlap)
            if overlap_count > 0:
                overlap_indices = set(range(overlap_count))
                inactive_indices.update(overlap_indices)
                ctx.dprint(f"Seg {ctx.segment_idx}: Adding {len(overlap_indices)} overlap frames to inactive set: {sorted(overlap_indices)}")
            else:
                ctx.dprint(f"Seg {ctx.segment_idx}: No overlap frames to mark as inactive")
            
            # 2) First frame when this is the very first segment from scratch
            is_first_segment_val = ctx.segment_params.get("is_first_segment", False)
            is_continue_scenario = ctx.full_orchestrator_payload.get("continue_from_video_resolved_path") is not None
            
            if is_first_segment_val and not is_continue_scenario:
                inactive_indices.add(0)
                ctx.dprint(f"Seg {ctx.segment_idx}: First segment from scratch - marking frame 0 as inactive")
            
            # 3) Last frame for multi-image segments - each segment travels TO a target image
            # For single image journeys, we don't anchor the end, let the model generate freely
            is_single_image_journey = self._detect_single_image_journey()
            if not is_single_image_journey:
                inactive_indices.add(ctx.total_frames_for_segment - 1)
                ctx.dprint(f"Seg {ctx.segment_idx}: Multi-image journey - marking last frame {ctx.total_frames_for_segment - 1} as inactive (target image)")
            else:
                ctx.dprint(f"Seg {ctx.segment_idx}: Single image journey - NOT marking last frame as inactive, letting model generate freely")

            # 4) Consolidated keyframe positions (frame consolidation optimization)
            consolidated_keyframe_positions = ctx.segment_params.get("consolidated_keyframe_positions")
            if consolidated_keyframe_positions:
                # Mark all keyframe positions as inactive since they should show exact keyframe images
                for frame_pos in consolidated_keyframe_positions:
                    if 0 <= frame_pos < ctx.total_frames_for_segment:
                        inactive_indices.add(frame_pos)
                ctx.dprint(f"Seg {ctx.segment_idx}: CONSOLIDATED SEGMENT - marking keyframe positions as inactive: {consolidated_keyframe_positions}")

            # --- DEBUG LOGGING (restored from original) ---
            ctx.dprint(f"[MASK_DEBUG] Segment {ctx.segment_idx}: frame_overlap_from_previous={frame_overlap_from_previous}")
            ctx.dprint(f"[MASK_DEBUG] Segment {ctx.segment_idx}: inactive (masked) frame indices: {sorted(list(inactive_indices))}")
            ctx.dprint(f"[MASK_DEBUG] Segment {ctx.segment_idx}: active (unmasked) frame indices: {[i for i in range(ctx.total_frames_for_segment) if i not in inactive_indices]}")
            # --- END DEBUG LOGGING ---
            
            # Create mask video output path
            timestamp_short = datetime.now().strftime("%H%M%S")
            unique_suffix = uuid.uuid4().hex[:6]
            mask_filename = f"seg{ctx.segment_idx:02d}_mask_{timestamp_short}_{unique_suffix}.mp4"
            mask_out_path_tmp = ctx.segment_processing_dir / mask_filename
            
            ctx.dprint(f"Seg {ctx.segment_idx}: Creating mask video with {len(inactive_indices)} inactive frames: {sorted(inactive_indices)}")
            
            # Always create mask video for VACE models (required for functionality)
            # For non-VACE models, only create in debug mode
            if not ctx.debug_enabled and not self.is_vace_model:
                ctx.dprint(f"Task {ctx.task_id}: Debug mode disabled and non-VACE model, skipping mask video creation")
                return None
            else:
                if self.is_vace_model and not ctx.debug_enabled:
                    ctx.dprint(f"Task {ctx.task_id}: VACE model detected, creating mask video (required for VACE functionality)")
                
                # Use the generalized mask creation function
                created_mask_vid = create_mask_video_from_inactive_indices(
                    total_frames=ctx.total_frames_for_segment,
                    resolution_wh=ctx.parsed_res_wh,
                    inactive_frame_indices=inactive_indices,
                    output_path=mask_out_path_tmp,
                    fps=ctx.full_orchestrator_payload.get("fps_helpers", 16),
                    task_id_for_logging=ctx.task_id,
                    dprint=ctx.dprint
                )
                
                if created_mask_vid and created_mask_vid.exists():
                    # Verify mask video properties match guide video
                    try:
                        mask_frames, mask_fps = get_video_frame_count_and_fps(str(created_mask_vid))
                        ctx.dprint(f"Seg {ctx.segment_idx}: Mask video generated - {mask_frames} frames @ {mask_fps}fps -> {created_mask_vid}")
                        
                        # Warn if frame count mismatch
                        if mask_frames != ctx.total_frames_for_segment:
                            ctx.dprint(f"[WARNING] Seg {ctx.segment_idx}: Mask frame count ({mask_frames}) != target ({ctx.total_frames_for_segment})")
                    except Exception as e_verify:
                        ctx.dprint(f"[WARNING] Seg {ctx.segment_idx}: Could not verify mask video properties: {e_verify}")
                    
                    return created_mask_vid
                else:
                    ctx.dprint(f"[MASK_ERROR] Seg {ctx.segment_idx}: Mask video creation failed")
                    return None
                    
        except Exception as e_mask:
            ctx.dprint(f"[MASK_ERROR] Seg {ctx.segment_idx}: Mask video creation failed: {e_mask}")
            traceback.print_exc()
            return None
    
    def create_video_prompt_type(self, mask_video_path: Optional[Path]) -> str:
        """
        Create video_prompt_type string for VACE compatibility.
        
        Args:
            mask_video_path: Path to mask video, if created
            
        Returns:
            video_prompt_type string (e.g., "VM", "VIM", "UM")
        """
        ctx = self.ctx
        
        ctx.dprint(f"[VPT_DEBUG] Seg {ctx.segment_idx}: Starting video_prompt_type construction")
        ctx.dprint(f"[VPT_DEBUG] Seg {ctx.segment_idx}: is_vace_model = {self.is_vace_model}")
        ctx.dprint(f"[VPT_DEBUG] Seg {ctx.segment_idx}: mask_video_path exists = {mask_video_path is not None}")
        
        if self.is_vace_model:
            ctx.dprint(f"[VPT_DEBUG] Seg {ctx.segment_idx}: ENTERING VACE MODEL PATH")
            vpt_components = []
            
            # Add video component (VACE always gets 'V' for video guide)
            vpt_components.append("V")
            ctx.dprint(f"[VPT_DEBUG] Seg {ctx.segment_idx}: Added video 'V', vpt_components = {vpt_components}")
            ctx.dprint(f"[VACEActivated] Seg {ctx.segment_idx}: Using VACE with raw video guide (no preprocessing)")
            
            # Add mask component if mask video exists
            if mask_video_path:
                vpt_components.append("M")
                ctx.dprint(f"[VPT_DEBUG] Seg {ctx.segment_idx}: Added mask 'M', vpt_components = {vpt_components}")
                ctx.dprint(f"[VACEActivated] Seg {ctx.segment_idx}: Adding mask control - mask video: {mask_video_path}")
            else:
                ctx.dprint(f"[VPT_DEBUG] Seg {ctx.segment_idx}: No mask video, vpt_components = {vpt_components}")
            
            video_prompt_type_str = "".join(vpt_components)
            ctx.dprint(f"[VPT_DEBUG] Seg {ctx.segment_idx}: VACE PATH RESULT: video_prompt_type = '{video_prompt_type_str}'")
            ctx.dprint(f"[VACEActivated] Seg {ctx.segment_idx}: Final video_prompt_type: '{video_prompt_type_str}'")
        else:
            ctx.dprint(f"[VPT_DEBUG] Seg {ctx.segment_idx}: ENTERING NON-VACE MODEL PATH")
            # Fallback for non-VACE models: use 'U' for unprocessed RGB to provide direct pixel-level control.
            u_component = "U"
            m_component = "M" if mask_video_path else ""
            
            ctx.dprint(f"[VPT_DEBUG] Seg {ctx.segment_idx}: Non-VACE components: U='{u_component}', M='{m_component}'")
            
            video_prompt_type_str = u_component + m_component
            ctx.dprint(f"[VPT_DEBUG] Seg {ctx.segment_idx}: NON-VACE PATH RESULT: video_prompt_type = '{video_prompt_type_str}'")
            ctx.dprint(f"[VACESkipped] Seg {ctx.segment_idx}: Using non-VACE model -> video_prompt_type: '{video_prompt_type_str}'")
        
        # Final debug logging
        ctx.dprint(f"[VPT_FINAL] Seg {ctx.segment_idx}: ===== FINAL VIDEO_PROMPT_TYPE SUMMARY =====")
        ctx.dprint(f"[VPT_FINAL] Seg {ctx.segment_idx}: Model: '{ctx.model_name}'")
        ctx.dprint(f"[VPT_FINAL] Seg {ctx.segment_idx}: Is VACE: {self.is_vace_model}")
        ctx.dprint(f"[VPT_FINAL] Seg {ctx.segment_idx}: Final video_prompt_type: '{video_prompt_type_str}'")
        ctx.dprint(f"[VPT_FINAL] Seg {ctx.segment_idx}: =========================================")
        
        return video_prompt_type_str
    
    def process_segment(self) -> Dict[str, Any]:
        """Main processing method that orchestrates all segment operations."""
        # Create guide video
        guide_video_path = self.create_guide_video()
        
        # Create mask video
        mask_video_path = self.create_mask_video()
        
        # Create video_prompt_type
        video_prompt_type = self.create_video_prompt_type(mask_video_path)
        
        return {
            "video_guide": str(guide_video_path) if guide_video_path else None,
            "video_mask": str(mask_video_path) if mask_video_path else None,
            "video_prompt_type": video_prompt_type
        }
    
    def _get_previous_segment_video(self) -> Optional[str]:
        """Get previous segment video output for guide creation."""
        ctx = self.ctx
        
        is_first_segment = ctx.segment_params.get("is_first_segment", ctx.segment_idx == 0)
        
        if is_first_segment and ctx.full_orchestrator_payload.get("continue_from_video_resolved_path"):
            # First segment continuing from video
            return ctx.full_orchestrator_payload.get("continue_from_video_resolved_path")
        elif not is_first_segment:
            # Subsequent segment - get predecessor output
            task_dependency_id, raw_path_from_db = db_ops.get_predecessor_output_via_edge_function(ctx.task_id)
            if task_dependency_id and raw_path_from_db:
                ctx.dprint(f"Seg {ctx.segment_idx}: Found predecessor output: {raw_path_from_db}")
                
                # Handle Supabase public URLs by downloading them locally for guide processing
                if raw_path_from_db.startswith("http"):
                    try:
                        ctx.dprint(f"Seg {ctx.segment_idx}: Detected remote URL for previous segment: {raw_path_from_db}. Downloading...")
                        
                        # Import download utilities
                        from . import common_utils
                        from pathlib import Path
                        
                        remote_url = raw_path_from_db
                        local_filename = Path(remote_url).name
                        # Store under segment_processing_dir to keep things tidy
                        local_download_path = ctx.segment_processing_dir / f"prev_{ctx.segment_idx:02d}_{local_filename}"
                        
                        # Ensure directory exists
                        ctx.segment_processing_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Perform download if file not already present
                        if not local_download_path.exists():
                            ctx.dprint(f"Seg {ctx.segment_idx}: Downloading from {remote_url}")
                            common_utils.download_file(remote_url, ctx.segment_processing_dir, local_download_path.name)
                            ctx.dprint(f"Seg {ctx.segment_idx}: Downloaded previous segment video to {local_download_path}")
                            
                            # Verify download was successful and file has content
                            if local_download_path.exists() and local_download_path.stat().st_size > 0:
                                ctx.dprint(f"Seg {ctx.segment_idx}: Download verified - file size: {local_download_path.stat().st_size:,} bytes")
                            else:
                                raise Exception(f"Download failed or resulted in empty file: {local_download_path}")
                        else:
                            ctx.dprint(f"Seg {ctx.segment_idx}: Local copy of previous segment video already exists at {local_download_path}")
                            # Verify cached file is still valid
                            if local_download_path.stat().st_size == 0:
                                ctx.dprint(f"Seg {ctx.segment_idx}: Cached file is empty, re-downloading...")
                                local_download_path.unlink()  # Remove empty file
                                common_utils.download_file(remote_url, ctx.segment_processing_dir, local_download_path.name)
                        
                        resolved_path = str(local_download_path.resolve())
                        ctx.dprint(f"Seg {ctx.segment_idx}: Returning local path for guide creation: {resolved_path}")
                        return resolved_path
                        
                    except Exception as e_dl_prev:
                        ctx.dprint(f"[WARNING] Seg {ctx.segment_idx}: Failed to download remote previous segment video: {e_dl_prev}")
                        # Return the original URL - this will likely cause an error downstream but preserves existing behavior
                        return raw_path_from_db
                else:
                    # Local path or relative path - handle SQLite path resolution
                    if db_ops.DB_TYPE == "sqlite" and db_ops.SQLITE_DB_PATH and raw_path_from_db.startswith("files/"):
                        from pathlib import Path
                        sqlite_db_parent = Path(db_ops.SQLITE_DB_PATH).resolve().parent
                        resolved_path = str((sqlite_db_parent / "public" / raw_path_from_db).resolve())
                        ctx.dprint(f"Seg {ctx.segment_idx}: Resolved SQLite relative path from DB '{raw_path_from_db}' to absolute path '{resolved_path}'")
                        return resolved_path
                    else:
                        # Path from DB is already absolute (Supabase) or an old absolute SQLite path
                        return raw_path_from_db
            else:
                ctx.dprint(f"[WARNING] Seg {ctx.segment_idx}: Could not retrieve predecessor output")
                return None
        else:
            # First segment from scratch - no previous video
            return None
    
    def _prepare_input_images_for_guide(self) -> List[str]:
        """Prepare input images for guide video creation."""
        ctx = self.ctx
        
        # Start with original input images
        input_images_resolved_original = ctx.full_orchestrator_payload["input_image_paths_resolved"]
        input_images_resolved_for_guide = input_images_resolved_original.copy()
        
        ctx.dprint(f"[GUIDE_INPUT_DEBUG] Seg {ctx.segment_idx}: Using {len(input_images_resolved_for_guide)} input images for guide creation")
        
        return input_images_resolved_for_guide
    
    def _detect_single_image_journey(self) -> bool:
        """
        Detect if this is a single image journey.
        
        A single image journey is when:
        - Only one input image is provided
        - Not continuing from a video
        - This segment is both first and last (entire journey in one segment)
        
        Returns:
            True if this is a single image journey, False otherwise
        """
        ctx = self.ctx
        
        input_images_for_check = ctx.full_orchestrator_payload.get("input_image_paths_resolved", [])
        is_single_image_journey = (
            len(input_images_for_check) == 1
            and ctx.full_orchestrator_payload.get("continue_from_video_resolved_path") is None
            and ctx.segment_params.get("is_first_segment")
            and ctx.segment_params.get("is_last_segment")
        )
        
        ctx.dprint(f"[SINGLE_IMAGE_DEBUG] Seg {ctx.segment_idx}: Single image journey detection:")
        ctx.dprint(f"[SINGLE_IMAGE_DEBUG]   Input images count: {len(input_images_for_check)}")
        ctx.dprint(f"[SINGLE_IMAGE_DEBUG]   Continue from video: {ctx.full_orchestrator_payload.get('continue_from_video_resolved_path') is not None}")
        ctx.dprint(f"[SINGLE_IMAGE_DEBUG]   Is first segment: {ctx.segment_params.get('is_first_segment')}")
        ctx.dprint(f"[SINGLE_IMAGE_DEBUG]   Is last segment: {ctx.segment_params.get('is_last_segment')}")
        ctx.dprint(f"[SINGLE_IMAGE_DEBUG]   Result: {is_single_image_journey}")
        
        return is_single_image_journey