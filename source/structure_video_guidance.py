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
    
    else:
        raise ValueError(f"Unsupported structure_type: {structure_type}. Must be 'flow', 'canny', or 'depth'")


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
        structure_type: Type of preprocessing ("flow", "canny", "depth", or "raw")
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

