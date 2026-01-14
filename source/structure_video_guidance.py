"""
Structure Video Guidance Module

This module provides functionality to extract structure patterns from a reference video
and apply them to unguidanced frames in travel guide videos.

Key Components:
- GuidanceTracker: Logically tracks which frames have guidance
- Video loading: Using WGP.py patterns for consistency
- Multi-type preprocessing: Optical flow, canny edges, or depth maps
- Flow adjustment: "adjust" (interpolate) vs "clip" (use as-is) semantics
- Motion application: Progressive warping with configurable strength

Supported Preprocessing Types:
- "flow" (default): Optical flow visualization using RAFT
- "canny": Canny edge detection for structural features
- "depth": Depth map estimation using Depth Anything V2
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
            dropped = video_frame_count - frames_to_load
            dprint(f"  Adjust mode: Your input video has {video_frame_count} frames so we'll drop {dropped} frames to compress your guide video to the {frames_to_load} frames your input images cover.")
        else:
            # Stretch: Repeat frames to reach target count
            # Use linear interpolation indices (will repeat frames)
            frame_indices = [int(i * (video_frame_count - 1) / (frames_to_load - 1)) for i in range(frames_to_load)]
            duplicates = frames_to_load - len(set(frame_indices))
            dprint(f"  Adjust mode: Your input video has {video_frame_count} frames so we'll duplicate {duplicates} frames to stretch your guide video to the {frames_to_load} frames your input images cover.")
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
            dprint(f"  Clip mode: Video too short ({len(frame_indices)} frames < {frames_to_load} needed), looping back to start to fill remaining frames")
            while len(frame_indices) < frames_to_load:
                remaining = frames_to_load - len(frame_indices)
                frame_indices.extend(frame_indices[:remaining])
        elif video_frame_count > frames_to_load:
            ignored = video_frame_count - frames_to_load
            dprint(f"  Clip mode: Your video will guide {frames_to_load} frames of your timeline. The last {ignored} frames of your video (frames {frames_to_load + 1}-{video_frame_count}) will be ignored.")

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


def apply_structure_motion_with_tracking(
    frames_for_guide_list: List[np.ndarray],
    guidance_tracker: GuidanceTracker,
    structure_video_path: str | None,  # Not used by segments (orchestrator uses it)
    structure_video_treatment: str,  # Not used by segments (orchestrator applies treatment)
    parsed_res_wh: Tuple[int, int],  # Not used by segments (orchestrator uses it)
    fps_helpers: int,  # Not used by segments (orchestrator uses it)
    structure_type: str = "flow",
    motion_strength: float = 1.0,  # Not used by segments (orchestrator applies it)
    canny_intensity: float = 1.0,  # Not used by segments (orchestrator applies it)
    depth_contrast: float = 1.0,  # Not used by segments (orchestrator applies it)
    structure_guidance_video_url: str | None = None,
    segment_processing_dir: Path | None = None,
    structure_guidance_frame_offset: int = 0,
    # Legacy parameters for backward compatibility
    structure_motion_video_url: str | None = None,
    structure_motion_frame_offset: int = 0,
    dprint: Callable = print
) -> List[np.ndarray]:
    """
    Apply structure guidance to unguidanced frames (called by segment workers).

    IMPORTANT: This function marks frames as guided on the tracker as it fills them.

    The orchestrator pre-computes all structure guidance (flow/canny/depth visualizations)
    with motion_strength/intensity/contrast already applied, then uploads to Supabase.
    Segments download and insert their portion of the pre-computed guidance frames.

    Args:
        frames_for_guide_list: Current guide frames
        guidance_tracker: Tracks which frames have guidance (MUTATED by this function)
        structure_video_path: Not used by segments (orchestrator uses it to create guidance video)
        structure_video_treatment: Not used by segments (orchestrator applies treatment when creating guidance video)
        parsed_res_wh: Target resolution - not used by segments (orchestrator uses it)
        fps_helpers: Target FPS - not used by segments (orchestrator uses it)
        structure_type: Type of preprocessing ("flow", "canny", or "depth") - for logging only
        motion_strength: Not used by segments (orchestrator applies it when creating guidance video)
        canny_intensity: Not used by segments (orchestrator applies it when creating guidance video)
        depth_contrast: Not used by segments (orchestrator applies it when creating guidance video)
        structure_guidance_video_url: URL/path to pre-computed guidance video from orchestrator (REQUIRED)
        segment_processing_dir: Directory for downloads (required)
        structure_guidance_frame_offset: Starting frame offset in the guidance video
        dprint: Debug print function

    Returns:
        Updated frames list with structure guidance applied
    """
    # Backward compatibility: merge old and new parameter names
    if structure_guidance_video_url is None and structure_motion_video_url is not None:
        structure_guidance_video_url = structure_motion_video_url
    if structure_guidance_frame_offset == 0 and structure_motion_frame_offset != 0:
        structure_guidance_frame_offset = structure_motion_frame_offset
    
    # Get unguidanced ranges from tracker (not pixel inspection!)
    unguidanced_ranges = guidance_tracker.get_unguidanced_ranges()
    
    if not unguidanced_ranges:
        dprint(f"[STRUCTURE_VIDEO] No unguidanced frames found")
        return frames_for_guide_list
    
    # Calculate total frames needed
    total_unguidanced = sum(end - start + 1 for start, end in unguidanced_ranges)
    
    dprint(f"[STRUCTURE_VIDEO] Processing {total_unguidanced} unguidanced frames across {len(unguidanced_ranges)} ranges")
    
    # ===== PRE-WARPED VIDEO PATH (FASTER) =====
    if structure_guidance_video_url:
        dprint(f"[STRUCTURE_VIDEO] ========== FAST PATH ACTIVATED ==========")
        dprint(f"[STRUCTURE_VIDEO] Using pre-warped guidance video (fast path)")
        dprint(f"[STRUCTURE_VIDEO] Type: {structure_type}")
        dprint(f"[STRUCTURE_VIDEO] URL/Path: {structure_guidance_video_url}")

        if not segment_processing_dir:
            raise ValueError("segment_processing_dir required when using structure_guidance_video_url")

        try:
            # Download and extract THIS segment's portion of guidance frames
            dprint(f"[STRUCTURE_VIDEO] Extracting frames starting at offset {structure_guidance_frame_offset}")
            guidance_frames = download_and_extract_motion_frames(
                structure_motion_video_url=structure_guidance_video_url,  # Function still uses old param name internally
                frame_start=structure_guidance_frame_offset,
                frame_count=total_unguidanced,
                download_dir=segment_processing_dir,
                dprint=dprint
            )

            dprint(f"[STRUCTURE_VIDEO] Successfully extracted {len(guidance_frames)} guidance frames")

            # Drop guidance frames directly into unguidanced ranges
            updated_frames = frames_for_guide_list.copy()
            guidance_frame_idx = 0
            frames_filled = 0

            for range_start, range_end in unguidanced_ranges:
                dprint(f"[STRUCTURE_VIDEO] Filling range {range_start}-{range_end}")

                for frame_idx in range(range_start, range_end + 1):
                    if guidance_frame_idx < len(guidance_frames):
                        updated_frames[frame_idx] = guidance_frames[guidance_frame_idx]
                        guidance_tracker.mark_single_frame(frame_idx)
                        guidance_frame_idx += 1
                        frames_filled += 1
                    else:
                        dprint(f"[STRUCTURE_VIDEO] Warning: Ran out of guidance frames at frame {frame_idx}")
                        break

            dprint(f"[STRUCTURE_VIDEO] ✓ FAST PATH SUCCESS: Filled {frames_filled} frames with {structure_type} visualizations")
            dprint(f"[STRUCTURE_VIDEO] ==========================================")
            return updated_frames

        except Exception as e:
            dprint(f"[ERROR] ✗ FAST PATH FAILED: {e}")
            dprint(f"[ERROR] Structure guidance could not be applied")
            traceback.print_exc()
            # Return original frames unchanged
            return frames_for_guide_list

    # No pre-computed guidance video provided
    dprint(f"[STRUCTURE_VIDEO] ========== NO GUIDANCE VIDEO ==========")
    dprint(f"[STRUCTURE_VIDEO] structure_guidance_video_url is None or empty")
    dprint(f"[STRUCTURE_VIDEO] Reason: Orchestrator didn't provide pre-computed guidance video")
    dprint(f"[STRUCTURE_VIDEO] Cannot apply structure guidance - returning frames unchanged")
    dprint(f"[STRUCTURE_VIDEO] =========================================")
    return frames_for_guide_list


def get_structure_preprocessor(
    structure_type: str,
    motion_strength: float = 1.0,
    canny_intensity: float = 1.0,
    depth_contrast: float = 1.0,
    dprint: Callable = print
):
    """
    Get preprocessor function for structure video guidance.

    We import and instantiate the preprocessor ourselves, then run it
    on the structure video frames to generate RGB visualizations.

    Args:
        structure_type: Type of preprocessing ("flow", "canny", or "depth")
        motion_strength: Only affects flow - scales flow vector magnitude
        canny_intensity: Only affects canny - scales edge boldness
        depth_contrast: Only affects depth - adjusts depth map contrast
        dprint: Debug print function

    Returns:
        Function that takes a list of frames (np.ndarray)
        and returns a list of processed frames (RGB visualizations).
    """
    dprint(f"[PREPROCESSOR_DEBUG] Initializing preprocessor with structure_type='{structure_type}'")

    if structure_type == "flow":
        # Add Wan2GP to path for imports
        wan_dir = Path(__file__).parent.parent / "Wan2GP"
        if str(wan_dir) not in sys.path:
            sys.path.insert(0, str(wan_dir))

        from Wan2GP.preprocessing.flow import FlowAnnotator

        # Ensure RAFT model is downloaded
        flow_model_path = wan_dir / "ckpts" / "flow" / "raft-things.pth"
        if not flow_model_path.exists():
            dprint(f"[FLOW] RAFT model not found, downloading from Hugging Face...")
            try:
                from huggingface_hub import hf_hub_download
                flow_model_path.parent.mkdir(parents=True, exist_ok=True)
                hf_hub_download(
                    repo_id="DeepBeepMeep/Wan2.1",
                    filename="raft-things.pth",
                    local_dir=str(wan_dir / 'ckpts'),
                    subfolder="flow"
                )
                dprint(f"[FLOW] RAFT model downloaded successfully")
            except Exception as e:
                raise RuntimeError(f"Failed to download RAFT model: {e}. Please manually download from https://huggingface.co/DeepBeepMeep/Wan2.1/tree/main/flow")

        cfg = {"PRETRAINED_MODEL": str(flow_model_path)}
        annotator = FlowAnnotator(cfg)

        # Import flow_viz for visualization
        from Wan2GP.preprocessing.raft.utils import flow_viz

        def process_with_motion_strength(frames):
            """Process frames with motion_strength applied to flow visualizations."""
            # Get raw flow fields from RAFT
            flow_fields, _ = annotator.forward(frames)

            # Scale flow fields by motion_strength
            scaled_flows = [flow * motion_strength for flow in flow_fields]

            # Generate visualizations from scaled flows
            flow_visualizations = [flow_viz.flow_to_image(flow) for flow in scaled_flows]

            # Match FlowVisAnnotator behavior: duplicate first frame
            return flow_visualizations[:1] + flow_visualizations

        if abs(motion_strength - 1.0) > 1e-6:
            dprint(f"[FLOW] Applying motion_strength={motion_strength} to flow visualizations")

        return process_with_motion_strength
    
    elif structure_type == "canny":
        wan_dir = Path(__file__).parent.parent / "Wan2GP"
        if str(wan_dir) not in sys.path:
            sys.path.insert(0, str(wan_dir))

        from Wan2GP.preprocessing.canny import CannyVideoAnnotator

        # Ensure scribble/canny model is downloaded
        canny_model_path = wan_dir / "ckpts" / "scribble" / "netG_A_latest.pth"
        if not canny_model_path.exists():
            dprint(f"[CANNY] Scribble model not found, downloading from Hugging Face...")
            try:
                from huggingface_hub import hf_hub_download
                canny_model_path.parent.mkdir(parents=True, exist_ok=True)
                hf_hub_download(
                    repo_id="DeepBeepMeep/Wan2.1",
                    filename="netG_A_latest.pth",
                    local_dir=str(wan_dir / 'ckpts'),
                    subfolder="scribble"
                )
                dprint(f"[CANNY] Scribble model downloaded successfully")
            except Exception as e:
                raise RuntimeError(f"Failed to download Scribble model: {e}. Please manually download from https://huggingface.co/DeepBeepMeep/Wan2.1/tree/main/scribble")

        cfg = {"PRETRAINED_MODEL": str(canny_model_path)}
        annotator = CannyVideoAnnotator(cfg)
        
        def process_canny(frames):
            # Get base canny edges
            edge_frames = annotator.forward(frames)
            
            # Apply intensity adjustment if not 1.0
            if abs(canny_intensity - 1.0) > 1e-6:
                adjusted_frames = []
                for frame in edge_frames:
                    # Scale pixel values by intensity factor
                    adjusted = (frame.astype(np.float32) * canny_intensity).clip(0, 255).astype(np.uint8)
                    adjusted_frames.append(adjusted)
                dprint(f"[STRUCTURE_PREPROCESS] Applied canny intensity: {canny_intensity}")
                return adjusted_frames
            return edge_frames
        
        return process_canny
    
    elif structure_type == "depth":
        wan_dir = Path(__file__).parent.parent / "Wan2GP"
        if str(wan_dir) not in sys.path:
            sys.path.insert(0, str(wan_dir))

        from Wan2GP.preprocessing.depth_anything_v2.depth import DepthV2VideoAnnotator

        variant = "vitl"  # Could be configurable

        # Ensure depth model is downloaded
        depth_model_path = wan_dir / "ckpts" / "depth" / f"depth_anything_v2_{variant}.pth"
        if not depth_model_path.exists():
            dprint(f"[DEPTH] Depth Anything V2 {variant} model not found, downloading from Hugging Face...")
            try:
                from huggingface_hub import hf_hub_download
                depth_model_path.parent.mkdir(parents=True, exist_ok=True)
                hf_hub_download(
                    repo_id="DeepBeepMeep/Wan2.1",
                    filename=f"depth_anything_v2_{variant}.pth",
                    local_dir=str(wan_dir / 'ckpts'),
                    subfolder="depth"
                )
                dprint(f"[DEPTH] Depth Anything V2 {variant} model downloaded successfully")
            except Exception as e:
                raise RuntimeError(f"Failed to download Depth Anything V2 model: {e}. Please manually download from https://huggingface.co/DeepBeepMeep/Wan2.1/tree/main/depth")

        cfg = {
            "PRETRAINED_MODEL": str(depth_model_path),
            "MODEL_VARIANT": variant
        }
        annotator = DepthV2VideoAnnotator(cfg)
        
        def process_depth(frames):
            # Get base depth maps
            depth_frames = annotator.forward(frames)
            
            # Apply contrast adjustment if not 1.0
            if abs(depth_contrast - 1.0) > 1e-6:
                adjusted_frames = []
                for frame in depth_frames:
                    # Convert to float, normalize, apply contrast, denormalize
                    frame_float = frame.astype(np.float32) / 255.0
                    # Apply contrast around midpoint (0.5)
                    adjusted = ((frame_float - 0.5) * depth_contrast + 0.5).clip(0, 1)
                    adjusted = (adjusted * 255).astype(np.uint8)
                    adjusted_frames.append(adjusted)
                dprint(f"[STRUCTURE_PREPROCESS] Applied depth contrast: {depth_contrast}")
                return adjusted_frames
            return depth_frames
        
        return process_depth

    elif structure_type in ("raw", "uni3c"):
        # Raw/uni3c use raw video frames as guidance - no preprocessing needed
        # For uni3c, frames are passed directly to WGP's uni3c encoder
        dprint(f"[STRUCTURE_PREPROCESS] {structure_type} type: returning frames without preprocessing")
        return lambda frames: frames

    else:
        raise ValueError(f"Unsupported structure_type: {structure_type}. Must be 'flow', 'canny', 'depth', 'raw', or 'uni3c'")


def process_structure_frames(
    frames: List[np.ndarray],
    structure_type: str,
    motion_strength: float,
    canny_intensity: float,
    depth_contrast: float,
    dprint: Callable
) -> List[np.ndarray]:
    """
    Process frames with chosen preprocessor, ensuring consistent output count.

    Handles the N-1 problem for optical flow (which returns N-1 flows for N frames).

    Args:
        frames: List of input frames to preprocess
        structure_type: Type of preprocessing ("flow", "canny", "depth", "raw", or "uni3c")
        motion_strength: Strength parameter for flow
        canny_intensity: Intensity parameter for canny
        depth_contrast: Contrast parameter for depth
        dprint: Debug print function

    Returns:
        List of RGB visualization frames (length = len(frames))
    """
    # Handle raw type - no preprocessing needed
    if structure_type == "raw":
        dprint(f"[STRUCTURE_PREPROCESS] Raw type: returning {len(frames)} frames without preprocessing")
        return frames

    dprint(f"[STRUCTURE_PREPROCESS] Processing {len(frames)} frames with '{structure_type}' preprocessor...")

    preprocessor = get_structure_preprocessor(
        structure_type,
        motion_strength,
        canny_intensity,
        depth_contrast,
        dprint
    )
    
    import time
    start_time = time.time()
    processed_frames = preprocessor(frames)
    duration = time.time() - start_time
    
    dprint(f"[STRUCTURE_PREPROCESS] Preprocessing completed in {duration:.2f}s")
    
    # Handle N-1 case for optical flow
    if structure_type == "flow" and len(processed_frames) == len(frames) - 1:
        # Duplicate last flow frame to match input count
        processed_frames.append(processed_frames[-1].copy())
        dprint(f"[STRUCTURE_PREPROCESS] Duplicated last flow frame ({len(frames)-1} → {len(frames)} frames)")
    
    # Validate output count
    if len(processed_frames) != len(frames):
        raise ValueError(
            f"Preprocessor '{structure_type}' returned {len(processed_frames)} frames "
            f"for {len(frames)} input frames. Expected {len(frames)} output frames."
        )
    
    return processed_frames


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


def create_structure_guidance_video(
    structure_video_path: str,
    max_frames_needed: int,
    target_resolution: Tuple[int, int],
    target_fps: int,
    output_path: Path,
    structure_type: str = "flow",
    motion_strength: float = 1.0,
    canny_intensity: float = 1.0,
    depth_contrast: float = 1.0,
    treatment: str = "adjust",
    dprint: Callable = print
) -> Path:
    """
    Create a video of preprocessed structure visualizations from the structure video.

    This is the NEW orchestrator-level function that:
    1. Loads and preprocesses frames from the structure video
    2. Applies the chosen preprocessor (flow, canny, depth, or raw)
    3. Encodes them as an H.264 video

    The resulting video contains structure visualizations that segments can use as
    VACE guide videos for structural conditioning, or raw frames for Uni3C.

    Args:
        structure_video_path: Path to the source structure video
        max_frames_needed: Number of frames to generate (total unguidanced frames)
        target_resolution: (width, height) for output frames
        target_fps: FPS for the output video
        output_path: Where to save the output video
        structure_type: Type of preprocessing:
            - "flow": Optical flow visualization (VACE)
            - "canny": Edge detection (VACE)
            - "depth": Depth map estimation (VACE)
            - "raw": No preprocessing - raw frames only (Uni3C)
            Default: "flow"
        motion_strength: Flow strength multiplier (only used for flow, also maps to uni3c_strength for raw)
        canny_intensity: Edge intensity multiplier (only used for canny)
        depth_contrast: Depth contrast adjustment (only used for depth)
        treatment: "adjust" (stretch/compress entire video) or "clip" (temporal sample)
        dprint: Debug print function

    Returns:
        Path to the created video file

    Raises:
        ValueError: If structure video cannot be loaded or processed
    """
    dprint(f"[STRUCTURE_GUIDANCE_VIDEO] Creating {structure_type} visualization video...")
    dprint(f"  Source: {structure_video_path}")
    dprint(f"  Type: {structure_type}")
    dprint(f"  Frames: {max_frames_needed}")
    dprint(f"  Resolution: {target_resolution[0]}x{target_resolution[1]}")
    dprint(f"  Treatment: {treatment}")
    
    # Log active strength parameter
    if structure_type == "flow" and abs(motion_strength - 1.0) > 1e-6:
        dprint(f"  Motion strength: {motion_strength}")
    elif structure_type == "canny" and abs(canny_intensity - 1e-6) > 1e-6:
        dprint(f"  Canny intensity: {canny_intensity}")
    elif structure_type == "raw":
        dprint(f"  Raw frames: No preprocessing applied (Uni3C mode)")
        if abs(motion_strength - 1.0) > 1e-6:
            dprint(f"  Uni3C strength: {motion_strength} (from motion_strength)")
    elif structure_type == "depth" and abs(depth_contrast - 1.0) > 1e-6:
        dprint(f"  Depth contrast: {depth_contrast}")

    try:
        # Step 1: Load structure video frames with treatment mode
        dprint(f"[STRUCTURE_GUIDANCE_VIDEO] Loading structure video frames...")
        structure_frames = load_structure_video_frames(
            structure_video_path,
            target_frame_count=max_frames_needed,
            target_fps=target_fps,
            target_resolution=target_resolution,
            treatment=treatment,
            crop_to_fit=True,  # Apply center cropping
            dprint=dprint
        )

        # Step 2: Process frames with chosen preprocessor
        dprint(f"[STRUCTURE_GUIDANCE_VIDEO] Processing with '{structure_type}' preprocessor...")
        processed_frames = process_structure_frames(
            structure_frames,
            structure_type,
            motion_strength,
            canny_intensity,
            depth_contrast,
            dprint
        )

        if not processed_frames:
            raise ValueError(f"No {structure_type} visualizations extracted from structure video")

        dprint(f"[STRUCTURE_GUIDANCE_VIDEO] Processed {len(processed_frames)} frames")
        
        # Step 3: Encode as video
        dprint(f"[STRUCTURE_GUIDANCE_VIDEO] Encoding video to {output_path}")
        
        # Import video creation utilities
        wan_dir = Path(__file__).parent.parent / "Wan2GP"
        if str(wan_dir) not in sys.path:
            sys.path.insert(0, str(wan_dir))
        
        from shared.utils.audio_video import save_video
        
        # Convert to numpy array format expected by save_video
        # save_video expects [T, H, W, C] in range [0, 255]
        video_tensor = np.stack(processed_frames, axis=0)  # [T, H, W, C]
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save video using WGP's video utilities
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
        dprint(f"[STRUCTURE_GUIDANCE_VIDEO] Created video: {output_path.name} ({file_size_mb:.2f} MB)")
        
        # Clean up GPU memory
        dprint(f"[STRUCTURE_GUIDANCE_VIDEO] Cleaning up GPU memory...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return output_path
        
    except Exception as e:
        dprint(f"[ERROR] Failed to create structure guidance video: {e}")
        traceback.print_exc()
        raise


# =============================================================================
# Multi-Structure Video Compositing
# =============================================================================

def create_neutral_frame(structure_type: str, resolution: Tuple[int, int]) -> np.ndarray:
    """
    Create a neutral frame for gaps in structure video coverage.
    
    These frames don't bias the generation - they represent "no guidance signal".
    
    IMPORTANT for Uni3C (structure_type="raw"):
        Black pixel frames are detected by _encode_uni3c_guide() in any2video.py
        and converted to zeros in latent space. This is critical because:
        - VAE-encoded black pixels ≠ zero latents
        - Zero latents = true "no control" 
        - VAE-encoded black = "control toward black output"
        
        The detection happens automatically during VAE encoding, so black frames
        created here will be properly handled as "no guidance".
    
    Args:
        structure_type: Type of structure preprocessing ("flow", "canny", "depth", "raw", or "uni3c")
        resolution: (width, height) tuple
        
    Returns:
        numpy array [H, W, C] uint8 RGB
    """
    w, h = resolution
    
    if structure_type == "flow":
        # Gray = center of HSV color wheel = no motion
        return np.full((h, w, 3), 128, dtype=np.uint8)
    elif structure_type == "canny":
        # Black = no edges detected
        return np.zeros((h, w, 3), dtype=np.uint8)
    elif structure_type == "depth":
        # Mid-gray = neutral depth (not close, not far)
        return np.full((h, w, 3), 128, dtype=np.uint8)
    elif structure_type == "raw":
        # Black = no structure signal for Uni3C
        # NOTE: These black frames will be detected during VAE encoding and
        # their latents will be zeroed for true "no control" behavior.
        return np.zeros((h, w, 3), dtype=np.uint8)
    else:
        # Default to black
        return np.zeros((h, w, 3), dtype=np.uint8)


def load_structure_video_frames_with_range(
    structure_video_path: str,
    target_frame_count: int,
    target_fps: int,
    target_resolution: Tuple[int, int],
    treatment: str = "adjust",
    crop_to_fit: bool = True,
    source_start_frame: int = 0,
    source_end_frame: Optional[int] = None,
    dprint: Callable = print
) -> List[np.ndarray]:
    """
    Load structure video frames with optional source range extraction.
    
    Extended version of load_structure_video_frames that supports extracting
    a specific range from the source video before applying treatment.
    
    Args:
        structure_video_path: Path to structure video
        target_frame_count: Number of output frames needed
        target_fps: Target FPS (used for clip mode)
        target_resolution: (width, height) tuple
        treatment: "adjust" (stretch/compress) or "clip" (temporal sample)
        crop_to_fit: Center-crop to match target aspect ratio
        source_start_frame: First frame to extract from source (default: 0)
        source_end_frame: Last frame (exclusive) to extract (default: None = end of video)
        dprint: Debug print function
        
    Returns:
        List of numpy uint8 arrays [H, W, C] in RGB format
    """
    import cv2
    from PIL import Image
    
    # Try decord first (faster), fall back to cv2
    use_decord = False
    try:
        import decord
        decord.bridge.set_bridge('torch')
        use_decord = True
    except ImportError:
        dprint("[STRUCTURE_VIDEO_RANGE] decord not available, using cv2 fallback")
    
    if use_decord:
        reader = decord.VideoReader(structure_video_path)
        video_fps = round(reader.get_avg_fps())
        total_video_frames = len(reader)
    else:
        cap = cv2.VideoCapture(structure_video_path)
        video_fps = round(cap.get(cv2.CAP_PROP_FPS))
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Resolve source range
    actual_start = source_start_frame
    actual_end = source_end_frame if source_end_frame is not None else total_video_frames
    
    # Clamp to valid range
    actual_start = max(0, min(actual_start, total_video_frames - 1))
    actual_end = max(actual_start + 1, min(actual_end, total_video_frames))
    
    effective_source_frames = actual_end - actual_start
    
    dprint(f"[STRUCTURE_VIDEO_RANGE] Loading from source video:")
    dprint(f"  Total source frames: {total_video_frames} @ {video_fps}fps")
    dprint(f"  Source range: [{actual_start}, {actual_end}) = {effective_source_frames} frames")
    dprint(f"  Target frames needed: {target_frame_count}")
    dprint(f"  Treatment: {treatment}")
    
    # Load N+1 frames for flow processing (RAFT needs N frames to produce N-1 flows)
    frames_to_load = target_frame_count + 1
    
    # Calculate frame indices based on treatment mode
    if treatment == "adjust":
        # ADJUST: Stretch/compress source range to match needed count
        if effective_source_frames >= frames_to_load:
            # Compress: sample evenly
            frame_indices = [
                actual_start + int(i * (effective_source_frames - 1) / (frames_to_load - 1))
                for i in range(frames_to_load)
            ]
            dprint(f"  Adjust: Compressing {effective_source_frames} → {frames_to_load} frames")
        else:
            # Stretch: repeat frames
            frame_indices = [
                actual_start + int(i * (effective_source_frames - 1) / (frames_to_load - 1))
                for i in range(frames_to_load)
            ]
            dprint(f"  Adjust: Stretching {effective_source_frames} → {frames_to_load} frames")
    else:
        # CLIP: Temporal sampling from the source range
        # Calculate indices relative to source range
        frame_indices = _resample_frame_indices(
            video_fps=video_fps,
            video_frames_count=effective_source_frames,
            max_target_frames_count=frames_to_load,
            target_fps=target_fps,
            start_target_frame=0
        )
        # Offset to actual source positions
        frame_indices = [actual_start + idx for idx in frame_indices]
        
        # Handle if source range is too short
        if len(frame_indices) < frames_to_load:
            dprint(f"  Clip: Source range too short, looping to fill {frames_to_load} frames")
            while len(frame_indices) < frames_to_load:
                remaining = frames_to_load - len(frame_indices)
                frame_indices.extend(frame_indices[:remaining])
        
        dprint(f"  Clip: Extracted {len(frame_indices)} frame indices")
    
    if not frame_indices:
        raise ValueError(f"No frames could be extracted from range [{actual_start}, {actual_end})")
    
    # Extract frames
    if use_decord:
        frames = reader.get_batch(frame_indices)
        raw_frames = []
        for frame in frames:
            if hasattr(frame, 'cpu'):
                frame_np = frame.cpu().numpy()
            else:
                frame_np = np.array(frame)
            raw_frames.append(frame_np)
    else:
        # cv2 fallback - read frames sequentially
        raw_frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame_bgr = cap.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                raw_frames.append(frame_rgb)
            else:
                # Fallback: use last frame or black
                if raw_frames:
                    raw_frames.append(raw_frames[-1].copy())
                else:
                    h_src = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    w_src = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    raw_frames.append(np.zeros((h_src, w_src, 3), dtype=np.uint8))
        cap.release()
    
    # Process frames to target resolution
    w, h = target_resolution
    target_aspect = w / h
    processed_frames = []
    
    for i, frame_np in enumerate(raw_frames):
        if frame_np.dtype != np.uint8:
            frame_np = np.clip(frame_np, 0, 255).astype(np.uint8)
        
        frame_pil = Image.fromarray(frame_np)
        
        # Center crop if needed
        if crop_to_fit:
            src_w, src_h = frame_pil.size
            src_aspect = src_w / src_h
            
            if abs(src_aspect - target_aspect) > 0.01:
                if src_aspect > target_aspect:
                    new_w = int(src_h * target_aspect)
                    left = (src_w - new_w) // 2
                    frame_pil = frame_pil.crop((left, 0, left + new_w, src_h))
                else:
                    new_h = int(src_w / target_aspect)
                    top = (src_h - new_h) // 2
                    frame_pil = frame_pil.crop((0, top, src_w, top + new_h))
        
        frame_resized = frame_pil.resize((w, h), resample=Image.Resampling.LANCZOS)
        processed_frames.append(np.array(frame_resized))
    
    dprint(f"[STRUCTURE_VIDEO_RANGE] Loaded {len(processed_frames)} frames at {w}x{h}")
    
    return processed_frames


def validate_structure_video_configs(
    configs: List[dict],
    total_frames: int,
    dprint: Callable = print
) -> List[dict]:
    """
    Validate and normalize structure video configurations.
    
    Args:
        configs: List of structure video config dicts
        total_frames: Total frames in the output timeline
        dprint: Debug print function
        
    Returns:
        Sorted and validated configs
        
    Raises:
        ValueError: If configs are invalid or overlap
    """
    if not configs:
        return []
    
    # Sort by start_frame
    sorted_configs = sorted(configs, key=lambda c: c.get("start_frame", 0))
    
    prev_end = -1
    for i, config in enumerate(sorted_configs):
        # Check required fields
        if "path" not in config:
            raise ValueError(f"Structure video config {i} missing 'path'")
        if "start_frame" not in config:
            raise ValueError(f"Structure video config {i} missing 'start_frame'")
        if "end_frame" not in config:
            raise ValueError(f"Structure video config {i} missing 'end_frame'")
        
        start = config["start_frame"]
        end = config["end_frame"]
        
        # Validate range
        if start < 0:
            raise ValueError(f"Config {i}: start_frame {start} < 0")
        if end > total_frames:
            raise ValueError(f"Config {i}: end_frame {end} > total_frames {total_frames}")
        if start >= end:
            raise ValueError(f"Config {i}: start_frame {start} >= end_frame {end}")
        
        # Check for overlap
        if start < prev_end:
            raise ValueError(f"Config {i}: frame range [{start}, {end}) overlaps with previous config ending at {prev_end}")
        
        prev_end = end
    
    dprint(f"[COMPOSITE] Validated {len(sorted_configs)} structure video configs")
    for i, cfg in enumerate(sorted_configs):
        dprint(f"  Config {i}: frames [{cfg['start_frame']}, {cfg['end_frame']}) from {Path(cfg['path']).name}")
    
    return sorted_configs


def create_composite_guidance_video(
    structure_configs: List[dict],
    total_frames: int,
    structure_type: str,
    target_resolution: Tuple[int, int],
    target_fps: int,
    output_path: Path,
    motion_strength: float = 1.0,
    canny_intensity: float = 1.0,
    depth_contrast: float = 1.0,
    download_dir: Optional[Path] = None,
    dprint: Callable = print
) -> Path:
    """
    Create a single composite guidance video from multiple structure video sources.
    
    This function:
    1. Creates a timeline filled with neutral frames
    2. For each config, loads and processes source frames
    3. Places processed frames at the correct timeline positions
    4. Encodes everything as a single video
    
    Args:
        structure_configs: List of config dicts, each with:
            - path: Source video path/URL
            - start_frame: Start position in output timeline
            - end_frame: End position (exclusive) in output timeline
            - treatment: Optional "adjust" or "clip" (default: "adjust")
            - source_start_frame: Optional start frame in source video
            - source_end_frame: Optional end frame in source video
        total_frames: Total frames in the output timeline
        structure_type: "flow", "canny", "depth", "raw", or "uni3c"
        target_resolution: (width, height) tuple
        target_fps: Output video FPS
        motion_strength: Flow motion strength
        canny_intensity: Canny edge intensity
        depth_contrast: Depth map contrast
        download_dir: Directory for downloading source videos
        dprint: Debug print function
        
    Returns:
        Path to the created composite guidance video
    """
    dprint(f"[COMPOSITE] Creating composite guidance video...")
    dprint(f"  Total frames: {total_frames}")
    dprint(f"  Structure type: {structure_type}")
    dprint(f"  Resolution: {target_resolution[0]}x{target_resolution[1]}")
    dprint(f"  Configs: {len(structure_configs)}")
    
    # Validate configs
    sorted_configs = validate_structure_video_configs(structure_configs, total_frames, dprint)
    
    if not sorted_configs:
        raise ValueError("No valid structure video configs provided")
    
    # Initialize timeline with neutral frames
    dprint(f"[COMPOSITE] Initializing {total_frames} neutral frames...")
    neutral_frame = create_neutral_frame(structure_type, target_resolution)
    composite_frames = [neutral_frame.copy() for _ in range(total_frames)]
    
    # Track which frame ranges are filled
    filled_ranges = []
    
    # Process each config
    for config_idx, config in enumerate(sorted_configs):
        source_path = config["path"]
        start_frame = config["start_frame"]
        end_frame = config["end_frame"]
        frames_needed = end_frame - start_frame
        treatment = config.get("treatment", "adjust")
        
        dprint(f"[COMPOSITE] Processing config {config_idx}: {Path(source_path).name}")
        dprint(f"  Timeline range: [{start_frame}, {end_frame}) = {frames_needed} frames")
        
        # Download if URL
        if source_path.startswith(("http://", "https://")):
            if download_dir is None:
                download_dir = Path("./temp_structure_downloads")
            download_dir.mkdir(parents=True, exist_ok=True)
            
            import requests
            local_filename = f"structure_src_{config_idx}_{Path(source_path).name}"
            local_path = download_dir / local_filename
            
            if not local_path.exists():
                dprint(f"  Downloading: {source_path}")
                response = requests.get(source_path, timeout=120)
                response.raise_for_status()
                with open(local_path, 'wb') as f:
                    f.write(response.content)
                dprint(f"  Downloaded: {local_path.name} ({len(response.content) / 1024 / 1024:.2f} MB)")
            else:
                dprint(f"  Using cached: {local_path.name}")
            
            source_path = str(local_path)
        
        # Load source frames with optional range extraction
        source_frames = load_structure_video_frames_with_range(
            structure_video_path=source_path,
            target_frame_count=frames_needed,
            target_fps=target_fps,
            target_resolution=target_resolution,
            treatment=treatment,
            crop_to_fit=True,
            source_start_frame=config.get("source_start_frame", 0),
            source_end_frame=config.get("source_end_frame"),
            dprint=dprint
        )
        
        # Process frames with chosen preprocessor
        dprint(f"  Processing with '{structure_type}' preprocessor...")
        processed_frames = process_structure_frames(
            source_frames,
            structure_type,
            motion_strength,
            canny_intensity,
            depth_contrast,
            dprint
        )
        
        # Ensure we have exactly the frames needed
        if len(processed_frames) > frames_needed:
            processed_frames = processed_frames[:frames_needed]
            dprint(f"  Trimmed to {frames_needed} frames")
        elif len(processed_frames) < frames_needed:
            # Pad with last frame if short
            while len(processed_frames) < frames_needed:
                processed_frames.append(processed_frames[-1].copy())
            dprint(f"  Padded to {frames_needed} frames")
        
        # Place processed frames into composite
        for i, frame in enumerate(processed_frames):
            frame_idx = start_frame + i
            if 0 <= frame_idx < total_frames:
                composite_frames[frame_idx] = frame
        
        filled_ranges.append((start_frame, end_frame))
        dprint(f"  Placed {len(processed_frames)} frames at positions {start_frame}-{end_frame-1}")
    
    # Log coverage summary
    total_filled = sum(end - start for start, end in filled_ranges)
    total_neutral = total_frames - total_filled
    dprint(f"[COMPOSITE] Coverage summary:")
    dprint(f"  Filled frames: {total_filled} ({100*total_filled/total_frames:.1f}%)")
    dprint(f"  Neutral frames: {total_neutral} ({100*total_neutral/total_frames:.1f}%)")
    
    # Encode as video
    dprint(f"[COMPOSITE] Encoding composite video to {output_path}...")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Try WGP's save_video first, fall back to cv2
    try:
        wan_dir = Path(__file__).parent.parent / "Wan2GP"
        if str(wan_dir) not in sys.path:
            sys.path.insert(0, str(wan_dir))
        
        from shared.utils.audio_video import save_video
        
        video_tensor = np.stack(composite_frames, axis=0)  # [T, H, W, C]
        
        save_video(
            video_tensor,
            save_file=str(output_path),
            fps=target_fps,
            codec_type='libx264_8',
            normalize=False,
            value_range=(0, 255)
        )
    except ImportError:
        # cv2 fallback for encoding
        import cv2
        dprint("[COMPOSITE] Using cv2 fallback for video encoding")
        
        w, h = target_resolution
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, target_fps, (w, h))
        
        for frame_rgb in composite_frames:
            # Convert RGB to BGR for cv2
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
        
        writer.release()
    
    if not output_path.exists():
        raise ValueError(f"Failed to create composite video at {output_path}")
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    dprint(f"[COMPOSITE] Created composite video: {output_path.name} ({file_size_mb:.2f} MB)")
    
    # Clean up GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return output_path


def segment_has_structure_overlap(
    segment_index: int,
    segment_frames_expanded: List[int],
    frame_overlap_expanded: List[int],
    structure_videos: List[dict]
) -> bool:
    """
    Check if a segment overlaps with any structure video config.
    
    Useful for orchestrators to skip structure guidance for segments with no overlap,
    avoiding unnecessary neutral guidance video creation.
    
    Args:
        segment_index: Index of the segment (0-based)
        segment_frames_expanded: List of frame counts per segment
        frame_overlap_expanded: List of overlap values between segments
        structure_videos: List of structure video configs
        
    Returns:
        True if segment overlaps with at least one config, False otherwise
    """
    if not structure_videos:
        return False
    
    seg_start, seg_frames = calculate_segment_stitched_position(
        segment_index, segment_frames_expanded, frame_overlap_expanded, lambda x: None
    )
    seg_end = seg_start + seg_frames
    
    for cfg in structure_videos:
        cfg_start = cfg.get("start_frame", 0)
        cfg_end = cfg.get("end_frame", 0)
        
        # Check overlap
        if cfg_start < seg_end and cfg_end > seg_start:
            return True
    
    return False


def calculate_segment_stitched_position(
    segment_index: int,
    segment_frames_expanded: List[int],
    frame_overlap_expanded: List[int],
    dprint: Callable = print
) -> Tuple[int, int]:
    """
    Calculate a segment's start position and frame count in the stitched timeline.
    
    Args:
        segment_index: Index of the segment (0-based)
        segment_frames_expanded: List of frame counts per segment
        frame_overlap_expanded: List of overlap values between segments
        dprint: Debug print function
        
    Returns:
        Tuple of (stitched_start, frame_count) for this segment
    """
    stitched_start = 0
    for idx in range(segment_index):
        segment_frames = segment_frames_expanded[idx]
        if idx == 0:
            stitched_start = segment_frames
        else:
            overlap = frame_overlap_expanded[idx - 1] if idx > 0 and idx - 1 < len(frame_overlap_expanded) else 0
            stitched_start += segment_frames - overlap
    
    # Adjust for the first segment's contribution
    if segment_index > 0:
        overlap = frame_overlap_expanded[segment_index - 1] if segment_index - 1 < len(frame_overlap_expanded) else 0
        stitched_start -= overlap
    else:
        stitched_start = 0
    
    frame_count = segment_frames_expanded[segment_index] if segment_index < len(segment_frames_expanded) else 81
    
    dprint(f"[SEGMENT_POS] Segment {segment_index}: stitched_start={stitched_start}, frame_count={frame_count}")
    return stitched_start, frame_count


def extract_segment_structure_guidance(
    structure_videos: List[dict],
    segment_index: int,
    segment_frames_expanded: List[int],
    frame_overlap_expanded: List[int],
    target_resolution: Tuple[int, int],
    target_fps: int,
    output_path: Path,
    motion_strength: float = 1.0,
    canny_intensity: float = 1.0,
    depth_contrast: float = 1.0,
    download_dir: Optional[Path] = None,
    dprint: Callable = print
) -> Optional[Path]:
    """
    Extract structure guidance for a single segment from the full structure_videos config.
    
    This is the segment-level function that allows standalone segments to compute
    their own portion of structure guidance without needing the orchestrator to
    pre-compute a full composite video.
    
    Args:
        structure_videos: Full array of structure video configs (same format as orchestrator)
        segment_index: Index of this segment (0-based)
        segment_frames_expanded: List of frame counts per segment
        frame_overlap_expanded: List of overlap values between segments
        target_resolution: (width, height) tuple
        target_fps: Target FPS
        output_path: Where to save the segment's guidance video
        motion_strength: Flow motion strength
        canny_intensity: Canny edge intensity
        depth_contrast: Depth map contrast
        download_dir: Directory for downloading source videos
        dprint: Debug print function
        
    Returns:
        Path to the created segment guidance video, or None if no configs apply
    """
    dprint(f"[SEGMENT_GUIDANCE] Extracting guidance for segment {segment_index}")
    
    if not structure_videos:
        dprint(f"[SEGMENT_GUIDANCE] No structure_videos provided, skipping")
        return None
    
    # Calculate this segment's position in the stitched timeline
    seg_start, seg_frames = calculate_segment_stitched_position(
        segment_index, segment_frames_expanded, frame_overlap_expanded, dprint
    )
    seg_end = seg_start + seg_frames
    
    dprint(f"[SEGMENT_GUIDANCE] Segment covers stitched frames [{seg_start}, {seg_end})")
    
    # Extract structure_type from configs (all must be same type)
    structure_types = set()
    for cfg in structure_videos:
        cfg_type = cfg.get("structure_type", cfg.get("type", "flow"))
        structure_types.add(cfg_type)

    if len(structure_types) > 1:
        raise ValueError(f"All structure_videos must have same type, found: {structure_types}")

    structure_type = structure_types.pop() if structure_types else "flow"
    
    # Find configs that overlap with this segment's frame range
    relevant_configs = []
    for cfg in structure_videos:
        cfg_start = cfg["start_frame"]
        cfg_end = cfg["end_frame"]
        
        # Check if this config overlaps with the segment's range
        if cfg_start < seg_end and cfg_end > seg_start:
            # Calculate overlap region in stitched timeline
            overlap_start = max(cfg_start, seg_start)
            overlap_end = min(cfg_end, seg_end)
            overlap_frames = overlap_end - overlap_start
            
            # Transform to segment-local coordinates
            local_start = overlap_start - seg_start
            local_end = overlap_end - seg_start
            
            # Calculate which portion of SOURCE video to use
            # The original config maps source range to stitched range [cfg_start, cfg_end)
            # We need the portion that corresponds to [overlap_start, overlap_end)
            cfg_duration = cfg_end - cfg_start
            src_start_orig = cfg.get("source_start_frame", 0)
            src_end_orig = cfg.get("source_end_frame")
            
            # If source_end not specified, we'll let the loader handle it
            # For proportional mapping, we need source video length
            if src_end_orig is not None:
                src_duration = src_end_orig - src_start_orig
                # Proportional mapping into source video
                overlap_start_in_cfg = overlap_start - cfg_start
                overlap_end_in_cfg = overlap_end - cfg_start
                
                new_src_start = src_start_orig + (overlap_start_in_cfg / cfg_duration) * src_duration
                new_src_end = src_start_orig + (overlap_end_in_cfg / cfg_duration) * src_duration
            else:
                # Can't do proportional without knowing source length
                # Let the loader handle the full source range
                new_src_start = src_start_orig
                new_src_end = src_end_orig
            
            transformed_cfg = {
                "path": cfg["path"],
                "start_frame": local_start,
                "end_frame": local_end,
                "treatment": cfg.get("treatment", "adjust"),
                "source_start_frame": int(new_src_start) if new_src_start is not None else 0,
                "source_end_frame": int(new_src_end) if new_src_end is not None else None,
                "structure_type": structure_type,
                "motion_strength": cfg.get("motion_strength", motion_strength),
            }
            
            dprint(f"[SEGMENT_GUIDANCE] Config overlaps: stitched [{cfg_start},{cfg_end}) -> local [{local_start},{local_end})")
            dprint(f"  Source frames: [{new_src_start}, {new_src_end})")
            relevant_configs.append(transformed_cfg)
    
    if not relevant_configs:
        # No overlap = no structure guidance needed for this segment
        # Return None so the segment proceeds without structure guidance entirely
        # This is cleaner than creating an all-neutral video that gets zeroed anyway
        dprint(f"[SEGMENT_GUIDANCE] No configs overlap with segment {segment_index}, skipping structure guidance")
        return None
    
    dprint(f"[SEGMENT_GUIDANCE] Found {len(relevant_configs)} overlapping configs")
    
    # Create mini-composite using the transformed configs
    return create_composite_guidance_video(
        structure_configs=relevant_configs,
        total_frames=seg_frames,
        structure_type=structure_type,
        target_resolution=target_resolution,
        target_fps=target_fps,
        output_path=output_path,
        motion_strength=motion_strength,
        canny_intensity=canny_intensity,
        depth_contrast=depth_contrast,
        download_dir=download_dir,
        dprint=dprint
    )


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
    from urllib.parse import urlparse
    import requests
    
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
            # Use local path (support plain paths and file:// URLs)
            if parsed.scheme == "file":
                local_video_path = Path(parsed.path)
            else:
                local_video_path = Path(structure_motion_video_url)
            if not local_video_path.exists():
                raise ValueError(f"Structure motion video not found: {local_video_path}")
            dprint(f"[STRUCTURE_MOTION] Using local video: {local_video_path}")
        
        # Extract frames (prefer decord, fall back to cv2)
        dprint(f"[STRUCTURE_MOTION] Extracting frames...")

        try:
            # Add Wan2GP to path (decord often installed alongside this stack)
            wan_dir = Path(__file__).parent.parent / "Wan2GP"
            if str(wan_dir) not in sys.path:
                sys.path.insert(0, str(wan_dir))

            import decord  # type: ignore
            decord.bridge.set_bridge('torch')

            vr = decord.VideoReader(str(local_video_path))
            total_frames = len(vr)

            if frame_start >= total_frames:
                raise ValueError(f"frame_start {frame_start} >= total frames {total_frames}")

            actual_frame_count = min(frame_count, total_frames - frame_start)
            if actual_frame_count < frame_count:
                dprint(f"[STRUCTURE_MOTION] Warning: Only {actual_frame_count} frames available (requested {frame_count})")

            frame_indices = list(range(frame_start, frame_start + actual_frame_count))
            frames_tensor = vr.get_batch(frame_indices)  # torch tensor [T, H, W, C]

            frames_list: List[np.ndarray] = []
            for i in range(len(frames_tensor)):
                frame_np = frames_tensor[i].cpu().numpy()
                if frame_np.dtype != np.uint8:
                    frame_np = np.clip(frame_np, 0, 255).astype(np.uint8)
                frames_list.append(frame_np)

            dprint(f"[STRUCTURE_MOTION] Extracted {len(frames_list)} frames (decord)")
            return frames_list

        except ModuleNotFoundError:
            import cv2
            dprint("[STRUCTURE_MOTION] decord not available, falling back to cv2")

            cap = cv2.VideoCapture(str(local_video_path))
            if not cap.isOpened():
                raise ValueError(f"cv2 failed to open video: {local_video_path}")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            if total_frames <= 0:
                # Some codecs don't report frame count; we'll read until EOF.
                total_frames = 10**9

            if frame_start >= total_frames:
                cap.release()
                raise ValueError(f"frame_start {frame_start} >= total frames {total_frames}")

            # Seek to frame_start
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)

            frames_list: List[np.ndarray] = []
            for _ in range(frame_count):
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                if frame_rgb.dtype != np.uint8:
                    frame_rgb = np.clip(frame_rgb, 0, 255).astype(np.uint8)
                frames_list.append(frame_rgb)

            cap.release()
            dprint(f"[STRUCTURE_MOTION] Extracted {len(frames_list)} frames (cv2)")
            return frames_list
        
    except Exception as e:
        dprint(f"[ERROR] Failed to extract motion frames: {e}")
        traceback.print_exc()
        raise


def create_trimmed_structure_video(
    structure_video_path: str,
    max_frames_needed: int,
    target_resolution: Tuple[int, int],
    target_fps: int,
    output_path: Path,
    treatment: str = "adjust",
    dprint: Callable = print
) -> Path:
    """
    Create a trimmed/adjusted version of the structure video without applying any style transfer.
    This preserves the original video content but clips/stretches it to match the generation length.

    Args:
        structure_video_path: Path to the source structure video
        max_frames_needed: Number of frames to generate
        target_resolution: (width, height) for output frames
        target_fps: FPS for the output video
        output_path: Where to save the output video
        treatment: "adjust" (stretch/compress entire video) or "clip" (temporal sample)
        dprint: Debug print function

    Returns:
        Path to the created video file
    """
    dprint(f"[TRIMMED_STRUCTURE_VIDEO] Creating trimmed structure video...")
    dprint(f"  Source: {structure_video_path}")
    dprint(f"  Frames: {max_frames_needed}")
    dprint(f"  Resolution: {target_resolution[0]}x{target_resolution[1]}")
    dprint(f"  Treatment: {treatment}")

    try:
        # Step 1: Load structure video frames with treatment mode
        # This handles the trimming/adjusting logic
        frames = load_structure_video_frames(
            structure_video_path,
            target_frame_count=max_frames_needed,
            target_fps=target_fps,
            target_resolution=target_resolution,
            treatment=treatment,
            crop_to_fit=True,
            dprint=dprint
        )

        # Step 2: Encode as video directly (no style transfer)
        dprint(f"[TRIMMED_STRUCTURE_VIDEO] Encoding video to {output_path}")
        
        # Import video creation utilities
        wan_dir = Path(__file__).parent.parent / "Wan2GP"
        if str(wan_dir) not in sys.path:
            sys.path.insert(0, str(wan_dir))
        
        from shared.utils.audio_video import save_video
        
        # Convert list of numpy arrays [H, W, C] to tensor [T, H, W, C]
        video_tensor = np.stack(frames, axis=0)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save video using WGP's video utilities
        save_video(
            video_tensor,
            save_file=str(output_path),
            fps=target_fps,
            codec_type='libx264_8',
            normalize=False,
            value_range=(0, 255)
        )
        
        return output_path

    except Exception as e:
        dprint(f"[TRIMMED_STRUCTURE_VIDEO] Error: {e}")
        raise

