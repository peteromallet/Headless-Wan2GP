"""
Structure Video Guidance Module

This module provides functionality to extract motion patterns from a structure video
and apply them to unguidanced frames in travel guide videos.

Key Components:
- GuidanceTracker: Logically tracks which frames have guidance
- Video loading: Using WGP.py patterns for consistency
- Optical flow extraction: Using RAFT via FlowAnnotator
- Flow adjustment: "adjust" (interpolate) vs "clip" (use as-is) semantics
- Motion application: Progressive warping with configurable strength
"""

import sys
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Callable
import traceback
import torch


@dataclass
class GuidanceTracker:
    """Tracks which frames in the guide video have guidance."""
    total_frames: int
    has_guidance: List[bool]  # True = has guidance, False = needs structure motion
    
    def __init__(self, total_frames: int):
        self.total_frames = total_frames
        # Initially, no frames have guidance
        self.has_guidance = [False] * total_frames
    
    def mark_guided(self, start_idx: int, end_idx: int):
        """Mark a range of frames as having guidance."""
        for i in range(start_idx, min(end_idx + 1, self.total_frames)):
            self.has_guidance[i] = True
    
    def mark_single_frame(self, idx: int):
        """Mark a single frame as having guidance."""
        if 0 <= idx < self.total_frames:
            self.has_guidance[idx] = True
    
    def get_unguidanced_ranges(self) -> List[Tuple[int, int]]:
        """
        Get continuous ranges of frames without guidance.
        
        Returns:
            List of (start_idx, end_idx) tuples for unguidanced ranges
        """
        unguidanced_ranges = []
        in_unguidanced_range = False
        range_start = None
        
        for i, has_guide in enumerate(self.has_guidance):
            if not has_guide and not in_unguidanced_range:
                # Start of unguidanced range
                range_start = i
                in_unguidanced_range = True
            elif has_guide and in_unguidanced_range:
                # End of unguidanced range
                unguidanced_ranges.append((range_start, i - 1))
                in_unguidanced_range = False
        
        # Handle case where video ends with unguidanced frames
        if in_unguidanced_range:
            unguidanced_ranges.append((range_start, self.total_frames - 1))
        
        return unguidanced_ranges
    
    def get_anchor_frame_index(self, unguidanced_range_start: int) -> Optional[int]:
        """
        Get the index of the last guided frame before an unguidanced range.
        This frame will be used as the anchor for structure motion warping.
        
        Args:
            unguidanced_range_start: Start index of the unguidanced range
            
        Returns:
            Index of anchor frame, or None if no guided frame exists before this range
        """
        # Search backward from range start
        for i in range(unguidanced_range_start - 1, -1, -1):
            if self.has_guidance[i]:
                return i
        return None
    
    def debug_summary(self) -> str:
        """Generate a visual summary of guidance state for debugging."""
        visual = []
        for i, has_guide in enumerate(self.has_guidance):
            if i % 10 == 0:
                visual.append(f"\n{i:3d}: ")
            visual.append("█" if has_guide else "░")
        
        unguidanced = self.get_unguidanced_ranges()
        summary = [
            "".join(visual),
            f"\nGuided frames: {sum(self.has_guidance)}/{self.total_frames}",
            f"Unguidanced ranges: {len(unguidanced)}"
        ]
        
        if unguidanced:
            summary.append("Ranges needing structure motion:")
            for start, end in unguidanced:
                summary.append(f"  - Frames {start}-{end} ({end - start + 1} frames)")
        
        return "\n".join(summary)


def _resample_frame_indices(video_fps: float, video_frames_count: int, max_target_frames_count: int, target_fps: float, start_target_frame: int) -> List[int]:
    """
    Calculate which frame indices to extract for FPS conversion.
    Ported from Wan2GP/shared/utils/utils.py to avoid importing wgp.py
    """
    import math
    
    video_frame_duration = 1 / video_fps
    target_frame_duration = 1 / target_fps
    
    target_time = start_target_frame * target_frame_duration
    frame_no = math.ceil(target_time / video_frame_duration)
    cur_time = frame_no * video_frame_duration
    frame_ids = []
    
    while True:
        if max_target_frames_count != 0 and len(frame_ids) >= max_target_frames_count:
            break
        diff = round((target_time - cur_time) / video_frame_duration, 5)
        add_frames_count = math.ceil(diff)
        frame_no += add_frames_count
        if frame_no >= video_frames_count:
            break
        frame_ids.append(frame_no)
        cur_time += add_frames_count * video_frame_duration
        target_time += target_frame_duration
    
    frame_ids = frame_ids[:max_target_frames_count]
    return frame_ids


def load_structure_video_frames(
    structure_video_path: str,
    target_frame_count: int,
    target_fps: int,
    target_resolution: Tuple[int, int],
    treatment: str = "adjust",
    crop_to_fit: bool = True,
    dprint: Callable = print
) -> List[np.ndarray]:
    """
    Load structure video frames with treatment mode.

    Args:
        structure_video_path: Path to structure video
        target_frame_count: Number of frames to load (note: loads target_frame_count+1 to get enough flows)
        target_fps: Target FPS (used for clip mode temporal sampling)
        target_resolution: (width, height) tuple
        treatment: "adjust" (stretch/compress entire video) or "clip" (temporal sample)
        crop_to_fit: If True, center-crop to match target aspect ratio before resizing
        dprint: Debug print function

    Returns:
        List of numpy uint8 arrays [H, W, C] in RGB format
    """
    import cv2
    from PIL import Image

    # Use decord directly instead of importing from wgp.py to avoid argparse conflicts
    try:
        import decord
        decord.bridge.set_bridge('torch')
    except ImportError:
        raise ImportError("decord is required for video processing. Install with: pip install decord")

    # Load N+1 frames to ensure we get N flows (RAFT produces N-1 flows for N frames)
    frames_to_load = target_frame_count + 1

    # Load video
    reader = decord.VideoReader(structure_video_path)
    video_fps = round(reader.get_avg_fps())
    video_frame_count = len(reader)

    dprint(f"[STRUCTURE_VIDEO] Loading frames from structure video:")
    dprint(f"  Video: {video_frame_count} frames @ {video_fps}fps")
    dprint(f"  Needed: {frames_to_load} frames")
    dprint(f"  Treatment: {treatment}")

    # Calculate frame indices based on treatment mode
    if treatment == "adjust":
        # ADJUST MODE: Stretch/compress entire video to match needed frame count
        # Linearly interpolate frame indices across the entire video
        if video_frame_count >= frames_to_load:
            # Compress: Sample evenly across video
            frame_indices = [int(i * (video_frame_count - 1) / (frames_to_load - 1)) for i in range(frames_to_load)]
            dprint(f"  Adjust mode: Compressing {video_frame_count} frames → {frames_to_load} (dropping {video_frame_count - frames_to_load})")
        else:
            # Stretch: Repeat frames to reach target count
            # Use linear interpolation indices (will repeat frames)
            frame_indices = [int(i * (video_frame_count - 1) / (frames_to_load - 1)) for i in range(frames_to_load)]
            duplicates = frames_to_load - len(set(frame_indices))
            dprint(f"  Adjust mode: Stretching {video_frame_count} frames → {frames_to_load} (duplicating {duplicates})")
    else:
        # CLIP MODE: Temporal sampling based on FPS
        frame_indices = _resample_frame_indices(
            video_fps=video_fps,
            video_frames_count=video_frame_count,
            max_target_frames_count=frames_to_load,
            target_fps=target_fps,
            start_target_frame=0
        )

        # If video is too short, loop back to start
        if len(frame_indices) < frames_to_load:
            dprint(f"  Clip mode: Video too short ({len(frame_indices)} < {frames_to_load}), looping frames")
            while len(frame_indices) < frames_to_load:
                remaining = frames_to_load - len(frame_indices)
                frame_indices.extend(frame_indices[:remaining])

        dprint(f"  Clip mode: Temporal sampling extracted {len(frame_indices)} frames")

    if not frame_indices:
        raise ValueError(f"No frames could be extracted from structure video: {structure_video_path}")

    # Extract frames using decord
    frames = reader.get_batch(frame_indices)  # Returns torch tensors [T, H, W, C]

    dprint(f"[STRUCTURE_VIDEO] Loaded {len(frames)} frames")
    
    # Process frames to target resolution (WGP pattern from line 3826-3830)
    w, h = target_resolution
    target_aspect = w / h
    processed_frames = []
    
    for i, frame in enumerate(frames):
        # Convert decord/torch tensor to numpy (WGP pattern line 3826)
        if hasattr(frame, 'cpu'):
            frame_np = frame.cpu().numpy()  # [H, W, C] uint8
        else:
            frame_np = np.array(frame)
        
        # Ensure uint8
        if frame_np.dtype != np.uint8:
            frame_np = np.clip(frame_np, 0, 255).astype(np.uint8)
        
        # Convert to PIL for processing (WGP pattern line 3830)
        frame_pil = Image.fromarray(frame_np)
        
        # Center crop to match target aspect ratio if requested
        if crop_to_fit:
            src_w, src_h = frame_pil.size
            src_aspect = src_w / src_h
            
            if abs(src_aspect - target_aspect) > 0.01:  # Different aspect ratios
                if src_aspect > target_aspect:
                    # Source is wider - crop width
                    new_w = int(src_h * target_aspect)
                    left = (src_w - new_w) // 2
                    frame_pil = frame_pil.crop((left, 0, left + new_w, src_h))
                else:
                    # Source is taller - crop height
                    new_h = int(src_w / target_aspect)
                    top = (src_h - new_h) // 2
                    frame_pil = frame_pil.crop((0, top, src_w, top + new_h))
                
                if i == 0:
                    dprint(f"[STRUCTURE_VIDEO] Center-cropped from {src_w}x{src_h} to {frame_pil.size[0]}x{frame_pil.size[1]}")
        
        # Resize to target resolution
        frame_resized = frame_pil.resize((w, h), resample=Image.Resampling.LANCZOS)
        frame_resized_np = np.array(frame_resized)
        
        processed_frames.append(frame_resized_np)
    
    dprint(f"[STRUCTURE_VIDEO] Preprocessed {len(processed_frames)} frames to {w}x{h}")
    
    return processed_frames


def extract_optical_flow_from_frames(
    frames: List[np.ndarray],
    dprint: Callable = print
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Extract optical flow from video frames using FlowAnnotator.
    
    Follows wgp.py pattern from get_preprocessor (line 3643-3648).
    CRITICAL: Explicitly cleans up RAFT model from GPU after use.
    
    Args:
        frames: List of numpy uint8 arrays [H, W, C]
        dprint: Debug print function
        
    Returns:
        (flow_fields, flow_visualizations)
        - flow_fields: List of flow arrays [H, W, 2] float32 (N-1 flows for N frames)
        - flow_visualizations: List of RGB visualization frames
    """
    import gc
    import torch
    
    if len(frames) < 2:
        raise ValueError(f"Need at least 2 frames for optical flow, got {len(frames)}")
    
    # Import FlowAnnotator (WGP pattern)
    wan_dir = Path(__file__).parent.parent / "Wan2GP"
    if str(wan_dir) not in sys.path:
        sys.path.insert(0, str(wan_dir))
    
    from Wan2GP.preprocessing.flow import FlowAnnotator
    
    # Initialize annotator (WGP pattern line 3643-3648)
    flow_model_path = wan_dir / 'ckpts' / 'flow' / 'raft-things.pth'
    
    # Ensure RAFT model is downloaded
    if not flow_model_path.exists():
        dprint(f"[OPTICAL_FLOW] RAFT model not found, downloading from Hugging Face...")
        try:
            from huggingface_hub import hf_hub_download
            flow_model_path.parent.mkdir(parents=True, exist_ok=True)
            hf_hub_download(
                repo_id="DeepBeepMeep/Wan2.1",
                filename="raft-things.pth",
                local_dir=str(wan_dir / 'ckpts'),
                subfolder="flow"
            )
            dprint(f"[OPTICAL_FLOW] RAFT model downloaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to download RAFT model: {e}. Please manually download from https://huggingface.co/DeepBeepMeep/Wan2.1/tree/main/flow")
    
    flow_cfg = {
        'PRETRAINED_MODEL': str(flow_model_path)
    }
    flow_annotator = FlowAnnotator(flow_cfg)
    
    try:
        dprint(f"[OPTICAL_FLOW] Extracting flow from {len(frames)} frames")
        dprint(f"[OPTICAL_FLOW] Will produce {len(frames)-1} flow fields")
        
        # Extract flow (FlowAnnotator.forward handles conversion internally)
        # Returns N-1 flows for N frames (line 44 in flow.py)
        flow_fields, flow_vis = flow_annotator.forward(frames)
        
        dprint(f"[OPTICAL_FLOW] Extracted {len(flow_fields)} optical flow fields")
        dprint(f"[OPTICAL_FLOW] Flow shape: {flow_fields[0].shape if flow_fields else 'N/A'}")
        
        return flow_fields, flow_vis
    
    finally:
        # CRITICAL: Clean up RAFT model from GPU
        # Pattern from wgp.py lines 5285-5299 (cleanup on generation end/error)
        del flow_annotator
        gc.collect()
        torch.cuda.empty_cache()
        dprint(f"[OPTICAL_FLOW] Cleaned up RAFT model from GPU memory")


def adjust_flow_field_count(
    flow_fields: List[np.ndarray],
    target_count: int,
    treatment: str,
    dprint: Callable = print
) -> List[np.ndarray]:
    """
    Adjust number of flow fields to match target count.
    
    Args:
        flow_fields: List of optical flow fields [H, W, 2]
        target_count: Desired number of flow fields
        treatment: "adjust" (interpolate) or "clip" (use what's available)
        dprint: Debug print function
        
    Returns:
        Adjusted list of flow fields (may be shorter than target_count for "clip")
    """
    source_count = len(flow_fields)
    
    if source_count == 0:
        raise ValueError("No flow fields provided")
    
    if source_count == target_count:
        dprint(f"[FLOW_ADJUST] Count matches ({source_count}), no adjustment needed")
        return flow_fields
    
    if treatment == "clip":
        if source_count >= target_count:
            # Truncate - use only what's needed
            adjusted = flow_fields[:target_count]
            dprint(f"[FLOW_ADJUST] Clipped from {source_count} to {target_count} flows")
        else:
            # CRITICAL: Don't pad, just return what we have
            # Remaining frames will be left unchanged (gray)
            adjusted = flow_fields
            dprint(f"[FLOW_ADJUST] Clip mode: Using all {source_count} available flows (target was {target_count})")
            dprint(f"[FLOW_ADJUST] Remaining {target_count - source_count} frames will be left unchanged")
        
        return adjusted
    
    elif treatment == "adjust":
        # Temporal interpolation to match target exactly
        adjusted_flows = []
        
        for target_idx in range(target_count):
            # Map target index to source space
            # CORRECTED: Avoid off-by-one error
            if target_count == 1:
                source_idx_float = 0.0
            else:
                # Map [0, target_count-1] to [0, source_count-1]
                source_idx_float = target_idx * (source_count - 1) / (target_count - 1)
            
            source_idx_low = int(np.floor(source_idx_float))
            source_idx_high = int(np.ceil(source_idx_float))
            
            # Clamp to valid range
            source_idx_low = np.clip(source_idx_low, 0, source_count - 1)
            source_idx_high = np.clip(source_idx_high, 0, source_count - 1)
            
            if source_idx_low == source_idx_high:
                # Exact match
                adjusted_flows.append(flow_fields[source_idx_low].copy())
            else:
                # Linear interpolation between two flows
                alpha = source_idx_float - source_idx_low
                flow_low = flow_fields[source_idx_low]
                flow_high = flow_fields[source_idx_high]
                interpolated = (1 - alpha) * flow_low + alpha * flow_high
                adjusted_flows.append(interpolated)
        
        dprint(f"[FLOW_ADJUST] Interpolated from {source_count} to {target_count} flows")
        return adjusted_flows
    
    else:
        raise ValueError(f"Invalid treatment: {treatment}. Must be 'adjust' or 'clip'")


def apply_optical_flow_warp(
    source_frame: np.ndarray,
    flow: np.ndarray,
    target_resolution: Tuple[int, int],
    motion_strength: float = 1.0
) -> np.ndarray:
    """
    Warp a frame using optical flow with adjustable motion strength.
    
    Args:
        source_frame: Frame to warp [H, W, C] uint8
        flow: Optical flow field [H, W, 2] float32
        target_resolution: (width, height)
        motion_strength: Strength multiplier for motion (0.0=no motion, 1.0=full, >1.0=amplified)
        
    Returns:
        Warped frame [H, W, C] uint8
    """
    import cv2
    
    w, h = target_resolution
    
    # Ensure source frame matches target resolution
    if source_frame.shape[:2] != (h, w):
        source_frame = cv2.resize(source_frame, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # Ensure flow matches target resolution
    if flow.shape[:2] != (h, w):
        # Resize flow field
        flow_resized = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # CRITICAL: Scale flow vectors when resizing
        # flow[:,:,0] is x-displacement, scale by width ratio
        # flow[:,:,1] is y-displacement, scale by height ratio
        original_h, original_w = flow.shape[:2]
        flow_resized[:, :, 0] *= w / original_w
        flow_resized[:, :, 1] *= h / original_h
        
        flow = flow_resized
    
    # SCALE FLOW VECTORS by motion strength
    # This physically reduces/amplifies the motion magnitude
    # 0.0 = no motion (stays at anchor), 1.0 = full motion, 2.0 = double motion
    flow_scaled = flow * motion_strength
    
    # Create proper coordinate grid using meshgrid
    # np.meshgrid with indexing='xy' gives (X, Y) where X varies along columns
    x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')
    
    # Apply scaled flow to coordinates
    # flow_scaled[:,:,0] is x-displacement, flow_scaled[:,:,1] is y-displacement
    map_x = (x_coords + flow_scaled[:, :, 0]).astype(np.float32)
    map_y = (y_coords + flow_scaled[:, :, 1]).astype(np.float32)
    
    # Warp frame using remap
    warped = cv2.remap(
        source_frame,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE  # Replicate edge pixels for out-of-bounds
    )
    
    return warped


def apply_structure_motion_with_tracking(
    frames_for_guide_list: List[np.ndarray],
    guidance_tracker: GuidanceTracker,
    structure_video_path: str | None,
    structure_video_treatment: str,
    parsed_res_wh: Tuple[int, int],
    fps_helpers: int,
    motion_strength: float = 1.0,
    structure_motion_video_url: str | None = None,
    segment_processing_dir: Path | None = None,
    structure_motion_frame_offset: int = 0,
    dprint: Callable = print
) -> List[np.ndarray]:
    """
    Apply structure motion to unguidanced frames.
    
    IMPORTANT: This function marks frames as guided on the tracker as it warps them.
    In "clip" mode, only frames that actually receive warps are marked - frames left
    unchanged when flows run out remain unguidanced.
    
    Two modes of operation:
    1. Pre-warped video (FASTER): If structure_motion_video_url is provided, downloads
       and extracts pre-computed motion frames directly (no GPU warping needed).
    2. Legacy path: If only structure_video_path is provided, extracts flows and
       applies warping per-segment (slower, more GPU work).
    
    Args:
        frames_for_guide_list: Current guide frames
        guidance_tracker: Tracks which frames have guidance (MUTATED by this function)
        structure_video_path: Path to structure video (legacy path, optional if URL provided)
        structure_video_treatment: "adjust" or "clip" (ignored for pre-warped video)
        parsed_res_wh: Target resolution (width, height)
        fps_helpers: Target FPS
        motion_strength: Motion strength multiplier (0.0=no motion, 1.0=full, >1.0=amplified)
        structure_motion_video_url: URL/path to pre-warped motion video (faster path)
        segment_processing_dir: Directory for downloads (required if using URL)
        dprint: Debug print function
        
    Returns:
        Updated frames list with structure motion applied
    """
    # Get unguidanced ranges from tracker (not pixel inspection!)
    unguidanced_ranges = guidance_tracker.get_unguidanced_ranges()
    
    if not unguidanced_ranges:
        dprint(f"[STRUCTURE_VIDEO] No unguidanced frames found")
        return frames_for_guide_list
    
    # Calculate total frames needed
    total_unguidanced = sum(end - start + 1 for start, end in unguidanced_ranges)
    
    dprint(f"[STRUCTURE_VIDEO] Processing {total_unguidanced} unguidanced frames across {len(unguidanced_ranges)} ranges")
    
    # ===== PRE-WARPED VIDEO PATH (FASTER) =====
    if structure_motion_video_url:
        dprint(f"[STRUCTURE_VIDEO] ========== FAST PATH ACTIVATED ==========")
        dprint(f"[STRUCTURE_VIDEO] Using pre-warped motion video (fast path)")
        dprint(f"[STRUCTURE_VIDEO] URL/Path: {structure_motion_video_url}")

        if not segment_processing_dir:
            raise ValueError("segment_processing_dir required when using structure_motion_video_url")

        try:
            # Download and extract THIS segment's portion of motion frames
            dprint(f"[STRUCTURE_VIDEO] Extracting frames starting at offset {structure_motion_frame_offset}")
            motion_frames = download_and_extract_motion_frames(
                structure_motion_video_url=structure_motion_video_url,
                frame_start=structure_motion_frame_offset,
                frame_count=total_unguidanced,
                download_dir=segment_processing_dir,
                dprint=dprint
            )

            dprint(f"[STRUCTURE_VIDEO] Successfully extracted {len(motion_frames)} motion frames")

            # Drop motion frames directly into unguidanced ranges
            updated_frames = frames_for_guide_list.copy()
            motion_frame_idx = 0
            frames_filled = 0

            for range_start, range_end in unguidanced_ranges:
                dprint(f"[STRUCTURE_VIDEO] Filling range {range_start}-{range_end}")

                for frame_idx in range(range_start, range_end + 1):
                    if motion_frame_idx < len(motion_frames):
                        updated_frames[frame_idx] = motion_frames[motion_frame_idx]
                        guidance_tracker.mark_single_frame(frame_idx)
                        motion_frame_idx += 1
                        frames_filled += 1
                    else:
                        dprint(f"[STRUCTURE_VIDEO] Warning: Ran out of motion frames at frame {frame_idx}")
                        break

            dprint(f"[STRUCTURE_VIDEO] ✓ FAST PATH SUCCESS: Filled {frames_filled} frames with flow visualizations")
            dprint(f"[STRUCTURE_VIDEO] ==========================================")
            return updated_frames

        except Exception as e:
            dprint(f"[ERROR] ✗ FAST PATH FAILED: {e}")
            dprint(f"[ERROR] Falling back to legacy path (will warp RGB frames)")
            traceback.print_exc()
            # Fall back to legacy path
            dprint(f"[STRUCTURE_VIDEO] Continuing with legacy warping path...")
    else:
        dprint(f"[STRUCTURE_VIDEO] ========== FAST PATH SKIPPED ==========")
        dprint(f"[STRUCTURE_VIDEO] structure_motion_video_url is None or empty")
        dprint(f"[STRUCTURE_VIDEO] Reason: Orchestrator didn't provide pre-computed flow video")
        dprint(f"[STRUCTURE_VIDEO] =========================================")
    
    # ===== LEGACY PATH: Extract flows and apply warping per-segment =====
    if not structure_video_path:
        dprint(f"[ERROR] Neither structure_motion_video_url nor structure_video_path provided")
        return frames_for_guide_list
    
    dprint(f"[STRUCTURE_VIDEO] Using legacy path: extracting flows and warping per-segment")
    
    try:
        # --- Step 1: Load structure video frames ---
        # Need N+1 frames to generate N optical flows
        structure_frames = load_structure_video_frames(
            structure_video_path,
            target_frame_count=total_unguidanced,  # Function adds +1 internally
            target_fps=fps_helpers,
            target_resolution=parsed_res_wh,
            crop_to_fit=True,  # Apply center cropping
            dprint=dprint
        )
        
        if len(structure_frames) < 2:
            dprint(f"[WARNING] Structure video has insufficient frames ({len(structure_frames)}). Need at least 2.")
            return frames_for_guide_list
        
        # --- Step 2: Extract optical flow ---
        flow_fields, flow_vis = extract_optical_flow_from_frames(
            structure_frames,
            dprint=dprint
        )
        
        # flow_fields contains N-1 flows for N frames
        dprint(f"[STRUCTURE_VIDEO] Extracted {len(flow_fields)} flow fields from {len(structure_frames)} frames")
        
        # --- Step 3: Adjust flow count to match needed frames ---
        flow_fields = adjust_flow_field_count(
            flow_fields,
            target_count=total_unguidanced,
            treatment=structure_video_treatment,
            dprint=dprint
        )
        
        # Track how many flows we actually have (may be < target in "clip" mode)
        available_flow_count = len(flow_fields)
        flows_applied = 0
        frames_skipped = 0
        
        # --- Step 4: Apply flow to unguidanced ranges ---
        updated_frames = frames_for_guide_list.copy()
        flow_idx = 0
        
        for range_idx, (start_idx, end_idx) in enumerate(unguidanced_ranges):
            range_length = end_idx - start_idx + 1
            
            # Get anchor frame (last guided frame before this range)
            anchor_idx = guidance_tracker.get_anchor_frame_index(start_idx)
            
            if anchor_idx is not None:
                current_frame = updated_frames[anchor_idx].copy()
                dprint(f"[STRUCTURE_VIDEO] Range {range_idx}: frames {start_idx}-{end_idx}, anchor=frame_{anchor_idx}")
            else:
                # No guided frame before - use gray
                current_frame = np.full((parsed_res_wh[1], parsed_res_wh[0], 3), 128, dtype=np.uint8)
                dprint(f"[STRUCTURE_VIDEO] Range {range_idx}: frames {start_idx}-{end_idx}, anchor=gray (no prior guidance)")
            
            # Apply motion progressively through range
            for offset in range(range_length):
                frame_idx = start_idx + offset
                
                # Check if we're out of flows
                if flow_idx >= available_flow_count:
                    if structure_video_treatment == "clip":
                        # CRITICAL: In "clip" mode, STOP applying motion when out of flows
                        # Leave remaining frames unchanged (gray/unguidanced)
                        frames_skipped += 1
                        dprint(f"[STRUCTURE_VIDEO] Clip mode: Out of flows at frame {frame_idx}, leaving unchanged")
                        # Don't modify updated_frames[frame_idx] - leave it as is
                        # Don't mark as guided - it remains unguidanced
                        continue  # ← Skip to next frame
                    else:
                        # In "adjust" mode, this shouldn't happen (we interpolated to match)
                        # But as safety fallback, repeat last flow
                        dprint(f"[WARNING] Exhausted flows in adjust mode at frame {frame_idx} (shouldn't happen)")
                        flow = flow_fields[-1]
                else:
                    flow = flow_fields[flow_idx]
                    flow_idx += 1
                
                # Warp current frame using flow with specified motion strength
                warped_frame = apply_optical_flow_warp(
                    current_frame,
                    flow,
                    parsed_res_wh,
                    motion_strength
                )
                
                updated_frames[frame_idx] = warped_frame
                current_frame = warped_frame  # Next warp uses this result
                
                # CRITICAL: Mark this frame as guided since we just warped it
                guidance_tracker.mark_single_frame(frame_idx)
                
                flows_applied += 1
        
        # Summary logging
        if structure_video_treatment == "clip" and frames_skipped > 0:
            dprint(f"[STRUCTURE_VIDEO] Clip mode summary:")
            dprint(f"  - Applied {flows_applied} flows to {flows_applied} frames")
            dprint(f"  - Left {frames_skipped} frames unchanged (ran out of flows)")
            dprint(f"  - Total unguidanced: {total_unguidanced} frames")
        else:
            dprint(f"[STRUCTURE_VIDEO] Applied {flows_applied} flows to {flows_applied} frames")
        
        return updated_frames
    
    except Exception as e:
        dprint(f"[ERROR] Structure motion application failed: {e}")
        traceback.print_exc()
        # Return original frames unchanged
        return frames_for_guide_list


def create_structure_motion_video(
    structure_video_path: str,
    max_frames_needed: int,
    target_resolution: Tuple[int, int],
    target_fps: int,
    motion_strength: float,
    output_path: Path,
    treatment: str = "adjust",
    dprint: Callable = print
) -> Path:
    """
    Create a video of flow visualizations from the structure video.

    This is the orchestrator-level function that:
    1. Extracts optical flows from the structure video
    2. Generates colorful flow visualizations (rainbow images showing motion)
    3. Encodes them as an H.264 video

    The resulting video contains flow visualizations that segments can use as
    VACE guide videos for motion conditioning, matching the format used by
    FlowVisAnnotator in wgp.py.

    Args:
        structure_video_path: Path to the source structure video
        max_frames_needed: Number of frames to generate (total unguidanced frames)
        target_resolution: (width, height) for output frames
        target_fps: FPS for the output video
        motion_strength: Unused (kept for compatibility)
        output_path: Where to save the output video
        treatment: "adjust" (stretch/compress entire video) or "clip" (temporal sample)
        dprint: Debug print function

    Returns:
        Path to the created video file

    Raises:
        ValueError: If structure video cannot be loaded or processed
    """
    dprint(f"[STRUCTURE_MOTION_VIDEO] Creating flow visualization video...")
    dprint(f"  Source: {structure_video_path}")
    dprint(f"  Frames: {max_frames_needed}")
    dprint(f"  Resolution: {target_resolution[0]}x{target_resolution[1]}")
    dprint(f"  Treatment: {treatment}")

    try:
        # Step 1: Load structure video frames with treatment mode
        dprint(f"[STRUCTURE_MOTION_VIDEO] Loading structure video frames...")
        structure_frames = load_structure_video_frames(
            structure_video_path,
            target_frame_count=max_frames_needed,
            target_fps=target_fps,
            target_resolution=target_resolution,
            treatment=treatment,
            crop_to_fit=True,  # Apply center cropping
            dprint=dprint
        )

        # Step 2: Extract optical flow visualizations
        dprint(f"[STRUCTURE_MOTION_VIDEO] Extracting optical flow visualizations...")
        flow_fields, flow_vis = extract_optical_flow_from_frames(structure_frames, dprint=dprint)

        if not flow_vis:
            raise ValueError("No optical flow visualizations extracted from structure video")

        dprint(f"[STRUCTURE_MOTION_VIDEO] Extracted {len(flow_vis)} flow visualizations")

        # Step 3: Duplicate first flow visualization to match frame count
        # This matches FlowVisAnnotator behavior: for N frames, return N visualizations
        # by duplicating the first one (flow_vis has N-1 items for N frames)
        dprint(f"[STRUCTURE_MOTION_VIDEO] Preparing flow visualization frames...")

        # Duplicate first visualization (matching FlowVisAnnotator pattern)
        motion_frames = [flow_vis[0]] + flow_vis

        dprint(f"[STRUCTURE_MOTION_VIDEO] Prepared {len(motion_frames)} flow visualization frames")
        
        # Step 4: Encode as video
        dprint(f"[STRUCTURE_MOTION_VIDEO] Encoding video to {output_path}")
        
        # Import video creation utilities
        wan_dir = Path(__file__).parent.parent / "Wan2GP"
        if str(wan_dir) not in sys.path:
            sys.path.insert(0, str(wan_dir))
        
        from shared.utils.audio_video import save_video
        
        # Convert to torch tensor format expected by save_video
        # save_video expects [T, H, W, C] in range [0, 255]
        video_tensor = np.stack(motion_frames, axis=0)  # [T, H, W, C]
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save video using WGP's video utilities
        # Note: save_video expects numpy array or torch tensor
        save_video(
            video_tensor,
            save_file=str(output_path),
            fps=target_fps,
            codec_type='libx264_8',  # Use libx264 with 8-bit encoding
            normalize=False,  # Already in [0, 255] uint8 range
            value_range=(0, 255)  # Specify value range for uint8 data
        )
        
        # Verify output exists
        if not output_path.exists():
            raise ValueError(f"Failed to create video at {output_path}")
        
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        dprint(f"[STRUCTURE_MOTION_VIDEO] Created video: {output_path.name} ({file_size_mb:.2f} MB)")
        
        return output_path
        
    except Exception as e:
        dprint(f"[ERROR] Failed to create structure motion video: {e}")
        traceback.print_exc()
        raise


def download_and_extract_motion_frames(
    structure_motion_video_url: str,
    frame_start: int,
    frame_count: int,
    download_dir: Path,
    dprint: Callable = print
) -> List[np.ndarray]:
    """
    Download pre-warped structure motion video and extract needed frames.
    
    This is the segment-level function that downloads the orchestrator's
    pre-computed motion video and extracts only the frames this segment needs.
    
    Args:
        structure_motion_video_url: URL or path to the pre-warped motion video
        frame_start: Starting frame index to extract
        frame_count: Number of frames to extract
        download_dir: Directory to download video to
        dprint: Debug print function
        
    Returns:
        List of numpy arrays [H, W, C] uint8 RGB
        
    Raises:
        ValueError: If video cannot be downloaded or frames extracted
    """
    import requests
    from urllib.parse import urlparse
    
    dprint(f"[STRUCTURE_MOTION] Extracting frames from pre-warped video")
    dprint(f"  URL/Path: {structure_motion_video_url}")
    dprint(f"  Frame range: {frame_start} to {frame_start + frame_count - 1}")
    
    try:
        # Determine if this is a URL or local path
        parsed = urlparse(structure_motion_video_url)
        is_url = parsed.scheme in ['http', 'https']
        
        if is_url:
            # Download the video
            dprint(f"[STRUCTURE_MOTION] Downloading video...")
            
            download_dir.mkdir(parents=True, exist_ok=True)
            local_video_path = download_dir / "structure_motion.mp4"
            
            response = requests.get(structure_motion_video_url, timeout=120)
            response.raise_for_status()
            
            with open(local_video_path, 'wb') as f:
                f.write(response.content)
            
            file_size_mb = len(response.content) / (1024 * 1024)
            dprint(f"[STRUCTURE_MOTION] Downloaded {file_size_mb:.2f} MB to {local_video_path.name}")
        else:
            # Use local path
            local_video_path = Path(structure_motion_video_url)
            if not local_video_path.exists():
                raise ValueError(f"Structure motion video not found: {local_video_path}")
            dprint(f"[STRUCTURE_MOTION] Using local video: {local_video_path}")
        
        # Extract frames using decord
        dprint(f"[STRUCTURE_MOTION] Extracting frames...")
        
        # Add Wan2GP to path
        wan_dir = Path(__file__).parent.parent / "Wan2GP"
        if str(wan_dir) not in sys.path:
            sys.path.insert(0, str(wan_dir))
        
        import decord
        decord.bridge.set_bridge('torch')
        
        vr = decord.VideoReader(str(local_video_path))
        total_frames = len(vr)
        
        # Validate frame range
        if frame_start >= total_frames:
            raise ValueError(f"frame_start {frame_start} >= total frames {total_frames}")
        
        # Adjust frame_count if it would exceed video length
        actual_frame_count = min(frame_count, total_frames - frame_start)
        if actual_frame_count < frame_count:
            dprint(f"[STRUCTURE_MOTION] Warning: Only {actual_frame_count} frames available (requested {frame_count})")
        
        # Extract frames
        frame_indices = list(range(frame_start, frame_start + actual_frame_count))
        frames_tensor = vr.get_batch(frame_indices)  # Returns torch tensor [T, H, W, C]
        
        # Convert to list of numpy arrays
        frames_list = []
        for i in range(len(frames_tensor)):
            frame_np = frames_tensor[i].cpu().numpy()
            
            # Ensure uint8
            if frame_np.dtype != np.uint8:
                frame_np = np.clip(frame_np, 0, 255).astype(np.uint8)
            
            frames_list.append(frame_np)
        
        dprint(f"[STRUCTURE_MOTION] Extracted {len(frames_list)} frames")
        
        return frames_list
        
    except Exception as e:
        dprint(f"[ERROR] Failed to extract motion frames: {e}")
        traceback.print_exc()
        raise

