#!/usr/bin/env python3
"""
Shared Travel Segment Processing Logic

This module contains the core logic for processing travel segments, eliminating
duplication between the full handler (travel_between_images.py) and the queue
handler (worker.py).

Key Components:
- TravelSegmentContext: Data class holding all segment parameters
- TravelSegmentProcessor: Main processor with shared logic
- ExecutionStrategy: Interface for different execution approaches
"""

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
    """Shared logic for processing travel segments."""
    
    def __init__(self, context: TravelSegmentContext):
        self.ctx = context
        self.is_vace_model = "vace" in context.model_name.lower()
        
    def create_guide_video(self) -> Optional[Path]:
        """Create guide video using shared logic."""
        ctx = self.ctx
        
        # Essential parameters for guide video creation
        fps_helpers = ctx.full_orchestrator_payload.get("fps_helpers", 16)
        segment_image_download_dir = ctx.segment_processing_dir  # Use processing dir for downloads
        
        # Determine if this is first segment from scratch
        is_first_segment = ctx.segment_params.get("is_first_segment", ctx.segment_idx == 0)
        is_first_segment_from_scratch = is_first_segment and not ctx.full_orchestrator_payload.get("continue_from_video_resolved_path")
        
        # Get input images and calculate end anchor index
        input_images_resolved_for_guide = ctx.full_orchestrator_payload.get("input_image_paths_resolved", [])
        end_anchor_img_path_str_idx = ctx.segment_idx + 1
        if ctx.full_orchestrator_payload.get("continue_from_video_resolved_path"):
            end_anchor_img_path_str_idx = ctx.segment_idx
        
        # Safety: if this is the last segment, no end anchor available
        if end_anchor_img_path_str_idx >= len(input_images_resolved_for_guide):
            end_anchor_img_path_str_idx = -1  # No end anchor image available
        
        # Get previous segment video output for guide creation
        path_to_previous_segment_video_output_for_guide = self._get_previous_segment_video()
        
        # Create guide video for VACE models (or debug mode)
        guide_video_path = None
        if ctx.debug_enabled or self.is_vace_model:
            if self.is_vace_model and not ctx.debug_enabled:
                ctx.dprint(f"[GUIDE_DEBUG] VACE model detected, creating guide video (REQUIRED for VACE functionality)")
            
            # Generate unique guide video filename
            timestamp_short = datetime.now().strftime("%H%M%S")
            unique_suffix = uuid.uuid4().hex[:6]
            guide_video_filename = f"seg{ctx.segment_idx:02d}_vace_guide_{timestamp_short}_{unique_suffix}.mp4"
            guide_video_base_name = f"seg{ctx.segment_idx:02d}_vace_guide_{timestamp_short}_{unique_suffix}"
            guide_video_final_path = ctx.segment_processing_dir / guide_video_filename
            
            try:
                # Call the shared guide video creation function
                guide_video_path = sm_create_guide_video_for_travel_segment(
                    segment_idx_for_logging=ctx.segment_idx,
                    end_anchor_image_index=end_anchor_img_path_str_idx,
                    is_first_segment_from_scratch=is_first_segment_from_scratch,
                    total_frames_for_segment=ctx.total_frames_for_segment,
                    parsed_res_wh=ctx.parsed_res_wh,
                    fps_helpers=fps_helpers,
                    input_images_resolved_for_guide=input_images_resolved_for_guide,
                    path_to_previous_segment_video_output_for_guide=path_to_previous_segment_video_output_for_guide,
                    output_target_dir=ctx.segment_processing_dir,
                    guide_video_base_name=guide_video_base_name,
                    segment_image_download_dir=segment_image_download_dir,
                    task_id_for_logging=ctx.task_id,
                    full_orchestrator_payload=ctx.full_orchestrator_payload,
                    segment_params=ctx.segment_params,
                    single_image_journey=False,  # Travel segments are not single image journeys
                    predefined_output_path=guide_video_final_path,
                    dprint=ctx.dprint
                )
                
                if guide_video_path and Path(guide_video_path).exists():
                    ctx.dprint(f"[GUIDE_DEBUG] Successfully created guide video: {guide_video_path}")
                else:
                    ctx.dprint(f"[GUIDE_ERROR] Guide video creation returned: {guide_video_path}")
                    
            except Exception as e_guide:
                ctx.dprint(f"[GUIDE_ERROR] Guide video creation failed: {e_guide}")
                traceback.print_exc()
                guide_video_path = None
        
        # CRITICAL: VACE models MUST have a guide video - fail if creation failed
        if self.is_vace_model and (guide_video_path is None or not Path(guide_video_path).exists()):
            error_msg = f"VACE model '{ctx.model_name}' requires a guide video but creation failed for segment {ctx.segment_idx}. VACE models cannot perform pure text-to-video generation."
            ctx.dprint(f"[VACE_ERROR] {error_msg}")
            raise ValueError(error_msg)
        
        return Path(guide_video_path) if guide_video_path else None
    
    def create_mask_video(self) -> Optional[Path]:
        """Create mask video using shared logic."""
        ctx = self.ctx
        
        if not ctx.mask_active_frames:
            return None
            
        try:
            # --- Determine which frame indices should be kept (inactive = black) ---
            inactive_indices = set()
    
            # Define overlap_count up front for consistent logging
            frame_overlap_from_previous = ctx.segment_params.get("frame_overlap_from_previous", 0)
            overlap_count = max(0, int(frame_overlap_from_previous))
    
            # Single image journey detection
            input_images_resolved_for_guide = ctx.full_orchestrator_payload.get("input_image_paths_resolved", [])
            is_single_image_journey = (
                len(input_images_resolved_for_guide) == 1
                and ctx.full_orchestrator_payload.get("continue_from_video_resolved_path") is None
                and ctx.segment_params.get("is_first_segment")
                and ctx.segment_params.get("is_last_segment")
            )
    
            if is_single_image_journey:
                # For a single image journey, only the first frame is kept from the guide.
                inactive_indices.add(0)
                ctx.dprint(f"Seg {ctx.segment_idx} Mask: Single image journey - keeping only frame 0 inactive.")
            else:
                # 1) Frames reused from the previous segment (overlap)
                inactive_indices.update(range(overlap_count))
    
            # 2) First frame when this is the very first segment from scratch
            is_first_segment_val = ctx.segment_params.get("is_first_segment", False)
            is_continue_scenario = ctx.full_orchestrator_payload.get("continue_from_video_resolved_path") is not None
            if is_first_segment_val and not is_continue_scenario:
                inactive_indices.add(0)
    
            # 3) Last frame for ALL segments - each segment travels TO a target image
            # Every segment ends at its target image, which should be kept (inactive/black)
            inactive_indices.add(ctx.total_frames_for_segment - 1)
    
            # --- DEBUG LOGGING ---
            print(f"[MASK_DEBUG] Segment {ctx.segment_idx}: frame_overlap_from_previous={frame_overlap_from_previous}")
            print(f"[MASK_DEBUG] Segment {ctx.segment_idx}: inactive (masked) frame indices: {sorted(list(inactive_indices))}")
            print(f"[MASK_DEBUG] Segment {ctx.segment_idx}: active (unmasked) frame indices: {[i for i in range(ctx.total_frames_for_segment) if i not in inactive_indices]}")
    
            # Generate mask video filename
            timestamp_short = datetime.now().strftime("%H%M%S")
            unique_suffix = uuid.uuid4().hex[:6]
            mask_filename = f"seg{ctx.segment_idx:02d}_vace_mask_{timestamp_short}_{unique_suffix}.mp4"
            mask_out_path_tmp = ctx.segment_processing_dir / mask_filename
            
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
                    ctx.dprint(f"[ERROR] Seg {ctx.segment_idx}: Mask video generation failed or file does not exist.")
                    return None
                    
        except Exception as e_mask_gen:
            ctx.dprint(f"[ERROR] Seg {ctx.segment_idx}: Mask video generation error: {e_mask_gen}")
            traceback.print_exc()
            return None
    
    def create_video_prompt_type(self, mask_video_path: Optional[Path]) -> str:
        """Create video_prompt_type string using shared logic."""
        ctx = self.ctx
        
        ctx.dprint(f"[VPT_DEBUG] Seg {ctx.segment_idx}: Starting video_prompt_type construction")
        ctx.dprint(f"[VPT_DEBUG] Seg {ctx.segment_idx}: is_vace_model = {self.is_vace_model}")
        ctx.dprint(f"[VPT_DEBUG] Seg {ctx.segment_idx}: mask_video_path exists = {mask_video_path is not None}")
        
        if self.is_vace_model:
            ctx.dprint(f"[VPT_DEBUG] Seg {ctx.segment_idx}: ENTERING VACE MODEL PATH")
            # For travel between images, default to frame masking rather than preprocessing
            # This lets VACE focus on the key frames while masking unused intermediate frames
            preprocessing_code = ctx.full_orchestrator_payload.get("vace_preprocessing", "M")  # Default to Mask-only
            ctx.dprint(f"[VPT_DEBUG] Seg {ctx.segment_idx}: vace_preprocessing from payload = '{preprocessing_code}'")
            
            # Build video_prompt_type based on available inputs and requested preprocessing
            vpt_components = ["V"]  # Always start with V for VACE
            ctx.dprint(f"[VPT_DEBUG] Seg {ctx.segment_idx}: Starting with vpt_components = {vpt_components}")
            
            if preprocessing_code != "M":
                # Explicit preprocessing requested (P=Pose, D=Depth, L=Flow, etc.)
                vpt_components.append(preprocessing_code)
                ctx.dprint(f"[VPT_DEBUG] Seg {ctx.segment_idx}: Added preprocessing '{preprocessing_code}', vpt_components = {vpt_components}")
                ctx.dprint(f"[VACEActivated] Seg {ctx.segment_idx}: Using VACE ControlNet with preprocessing '{preprocessing_code}'")
            else:
                ctx.dprint(f"[VPT_DEBUG] Seg {ctx.segment_idx}: No preprocessing (default 'M'), vpt_components = {vpt_components}")
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
                ctx.dprint(f"[GUIDE_DEBUG] Segment {ctx.segment_idx}: Found predecessor {task_dependency_id} with output: {raw_path_from_db}")
                # Resolve path (handle SQLite relative paths vs absolute paths)
                if db_ops.DB_TYPE == "sqlite" and db_ops.SQLITE_DB_PATH and raw_path_from_db.startswith("files/"):
                    sqlite_db_parent = Path(db_ops.SQLITE_DB_PATH).resolve().parent
                    return str((sqlite_db_parent / "public" / raw_path_from_db).resolve())
                else:
                    return raw_path_from_db
        
        return None


def create_travel_segment_context(
    task_id: str,
    segment_idx: int, 
    model_name: str,
    total_frames_for_segment: int,
    parsed_res_wh: Tuple[int, int],
    segment_processing_dir: Path,
    full_orchestrator_payload: Dict[str, Any],
    segment_params: Dict[str, Any],
    mask_active_frames: bool,
    debug_enabled: bool,
    dprint: Callable = print
) -> TravelSegmentContext:
    """Factory function to create TravelSegmentContext."""
    return TravelSegmentContext(
        task_id=task_id,
        segment_idx=segment_idx,
        model_name=model_name,
        total_frames_for_segment=total_frames_for_segment,
        parsed_res_wh=parsed_res_wh,
        segment_processing_dir=segment_processing_dir,
        full_orchestrator_payload=full_orchestrator_payload,
        segment_params=segment_params,
        mask_active_frames=mask_active_frames,
        debug_enabled=debug_enabled,
        dprint=dprint
    )
