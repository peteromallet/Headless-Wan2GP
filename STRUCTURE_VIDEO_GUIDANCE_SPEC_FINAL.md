# Structure Video Guidance - Final Implementation Spec

## Following WGP.py Patterns

This spec applies the same patterns used in `wgp.py` for preprocessing video guides with flow/motion.

---

## Key Patterns from WGP.py

### 1. **Decord Frame Conversion**
```python
# wgp.py line 3826: Convert decord frames to numpy
frame = Image.fromarray(video[frame_idx].cpu().numpy())  # .asnumpy() also works
```

### 2. **Flow Extraction with Proper Conversion**
```python
# preprocessing/flow.py line 8-16: Robust type conversion
def convert_to_numpy(image):
    if isinstance(image, Image.Image):
        image = np.array(image)
    elif isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    elif isinstance(image, np.ndarray):
        image = image.copy()
    else:
        raise f'Unsupported datatype{type(image)}'
    return image

# line 41: FlowAnnotator expects uint8 numpy arrays
frames = [torch.from_numpy(convert_to_numpy(frame).astype(np.uint8))
         .permute(2, 0, 1).float()[None].to(self.device) 
         for frame in frames]
```

### 3. **Video Loading with get_resampled_video**
```python
# wgp.py line 3539-3553: Decord video loading
def get_resampled_video(video_in, start_frame, max_frames, target_fps, bridge='torch'):
    from shared.utils.utils import resample
    import decord
    decord.bridge.set_bridge(bridge)  # 'torch' or 'numpy'
    reader = decord.VideoReader(video_in)
    fps = round(reader.get_avg_fps())
    if max_frames < 0:
        max_frames = max(len(reader) / fps * target_fps + max_frames, 0)
    
    frame_nos = resample(fps, len(reader), 
                        max_target_frames_count=max_frames, 
                        target_fps=target_fps, 
                        start_target_frame=start_frame)
    frames_list = reader.get_batch(frame_nos)
    return frames_list  # torch tensors if bridge='torch'
```

---

## Revised Implementation Following WGP Patterns

### 1. GuidanceTracker (No Changes - Already Correct)

```python
# Same as STRUCTURE_VIDEO_GUIDANCE_SPEC_REVISED.md
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class GuidanceTracker:
    """Tracks which frames in the guide video have guidance."""
    total_frames: int
    has_guidance: List[bool]
    
    # ... (implementation unchanged from revised spec)
```

### 2. Load Structure Video Frames (WGP Style)

```python
def load_structure_video_frames(
    structure_video_path: str,
    target_frame_count: int,
    target_fps: int,
    target_resolution: tuple[int, int],
    dprint=print
) -> list[np.ndarray]:
    """
    Load structure video frames following wgp.py patterns.
    
    Returns list of numpy uint8 arrays [H, W, C] in RGB format.
    """
    import sys
    from pathlib import Path
    
    # Add Wan2GP to path if not already
    wan_dir = Path(__file__).parent.parent.parent / "Wan2GP"
    if str(wan_dir) not in sys.path:
        sys.path.insert(0, str(wan_dir))
    
    from Wan2GP.wgp import get_resampled_video
    import cv2
    from PIL import Image
    
    # CRITICAL: Load with torch bridge (returns decord tensors on GPU)
    frames = get_resampled_video(
        structure_video_path,
        start_frame=0,
        max_frames=target_frame_count,
        target_fps=target_fps,
        bridge='torch'  # Returns torch tensors
    )
    
    if not frames or len(frames) == 0:
        raise ValueError(f"No frames loaded from structure video: {structure_video_path}")
    
    dprint(f"[STRUCTURE_VIDEO] Loaded {len(frames)} frames from {structure_video_path}")
    
    # Process frames to target resolution (WGP pattern from line 3826-3830)
    w, h = target_resolution
    processed_frames = []
    
    for i, frame in enumerate(frames):
        # Convert decord/torch tensor to numpy (WGP pattern)
        # frame is a torch tensor on GPU from decord
        if hasattr(frame, 'cpu'):
            frame_np = frame.cpu().numpy()  # [H, W, C] uint8
        else:
            frame_np = np.array(frame)
        
        # Ensure uint8
        if frame_np.dtype != np.uint8:
            frame_np = np.clip(frame_np, 0, 255).astype(np.uint8)
        
        # Resize to target resolution using PIL (WGP pattern)
        frame_pil = Image.fromarray(frame_np)
        frame_resized = frame_pil.resize((w, h), resample=Image.Resampling.LANCZOS)
        frame_resized_np = np.array(frame_resized)
        
        processed_frames.append(frame_resized_np)
    
    dprint(f"[STRUCTURE_VIDEO] Preprocessed {len(processed_frames)} frames to {w}x{h}")
    
    return processed_frames
```

### 3. Extract Optical Flow (WGP Style)

```python
def extract_optical_flow_from_frames(
    frames: list[np.ndarray],
    dprint=print
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Extract optical flow from video frames using FlowAnnotator.
    
    Follows wgp.py pattern from get_preprocessor (line 3643-3648).
    
    Args:
        frames: List of numpy uint8 arrays [H, W, C]
        
    Returns:
        (flow_fields, flow_visualizations)
        - flow_fields: List of flow arrays [H, W, 2] float32 (N-1 flows for N frames)
        - flow_visualizations: List of RGB visualization frames
    """
    if len(frames) < 2:
        raise ValueError(f"Need at least 2 frames for optical flow, got {len(frames)}")
    
    # Import FlowAnnotator (WGP pattern)
    import sys
    from pathlib import Path
    wan_dir = Path(__file__).parent.parent.parent / "Wan2GP"
    if str(wan_dir) not in sys.path:
        sys.path.insert(0, str(wan_dir))
    
    from Wan2GP.preprocessing.flow import FlowAnnotator
    
    # Initialize annotator (WGP pattern)
    flow_cfg = {
        'PRETRAINED_MODEL': 'ckpts/flow/raft-things.pth'
    }
    flow_annotator = FlowAnnotator(flow_cfg)
    
    dprint(f"[OPTICAL_FLOW] Extracting flow from {len(frames)} frames")
    dprint(f"[OPTICAL_FLOW] Will produce {len(frames)-1} flow fields")
    
    # Extract flow (handles conversion internally via convert_to_numpy)
    flow_fields, flow_vis = flow_annotator.forward(frames)
    
    dprint(f"[OPTICAL_FLOW] Extracted {len(flow_fields)} optical flow fields")
    dprint(f"[OPTICAL_FLOW] Flow shape: {flow_fields[0].shape if flow_fields else 'N/A'}")
    
    return flow_fields, flow_vis
```

### 4. Adjust Flow Length (Fixed Math)

```python
def adjust_flow_field_count(
    flow_fields: list[np.ndarray],
    target_count: int,
    treatment: str,
    dprint=print
) -> list[np.ndarray]:
    """
    Adjust number of flow fields to match target count.
    
    CORRECTED: Proper index mapping without off-by-one errors.
    
    Args:
        flow_fields: List of optical flow fields [H, W, 2]
        target_count: Desired number of flow fields
        treatment: "adjust" (interpolate) or "clip" (truncate/pad)
        
    Returns:
        Adjusted list of flow fields
    """
    source_count = len(flow_fields)
    
    if source_count == 0:
        raise ValueError("No flow fields provided")
    
    if source_count == target_count:
        dprint(f"[FLOW_ADJUST] Count matches ({source_count}), no adjustment needed")
        return flow_fields
    
    if treatment == "clip":
        if source_count >= target_count:
            # Truncate
            adjusted = flow_fields[:target_count]
            dprint(f"[FLOW_ADJUST] Clipped from {source_count} to {target_count} flows")
        else:
            # Pad by repeating last flow
            padding_count = target_count - source_count
            padding = [flow_fields[-1].copy() for _ in range(padding_count)]
            adjusted = flow_fields + padding
            dprint(f"[FLOW_ADJUST] Padded from {source_count} to {target_count} flows (repeated last)")
        
        return adjusted
    
    elif treatment == "adjust":
        # Temporal interpolation
        adjusted_flows = []
        
        for target_idx in range(target_count):
            # CORRECTED: Map target index to source space
            # Avoid off-by-one: when target_idx = target_count-1, should map to source_count-1
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
```

### 5. Apply Optical Flow Warp (Fixed Grid Creation)

```python
def apply_optical_flow_warp(
    source_frame: np.ndarray,
    flow: np.ndarray,
    target_resolution: tuple[int, int]
) -> np.ndarray:
    """
    Warp a frame using optical flow.
    
    CORRECTED: Proper meshgrid creation and flow scaling.
    
    Args:
        source_frame: Frame to warp [H, W, C] uint8
        flow: Optical flow field [H, W, 2] float32
        target_resolution: (width, height)
        
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
    
    # CORRECTED: Create proper coordinate grid using meshgrid
    # np.meshgrid with indexing='xy' gives (X, Y) where X varies along columns, Y along rows
    x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')
    
    # Apply flow to coordinates
    # flow[:,:,0] is x-displacement, flow[:,:,1] is y-displacement
    map_x = (x_coords + flow[:, :, 0]).astype(np.float32)
    map_y = (y_coords + flow[:, :, 1]).astype(np.float32)
    
    # Warp frame using remap
    warped = cv2.remap(
        source_frame,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE  # Replicate edge pixels for out-of-bounds
    )
    
    return warped
```

### 6. Main Structure Motion Application

```python
def apply_structure_motion_with_tracking(
    frames_for_guide_list: list[np.ndarray],
    guidance_tracker: GuidanceTracker,
    structure_video_path: str,
    structure_video_treatment: str,
    parsed_res_wh: tuple[int, int],
    fps_helpers: int,
    dprint=print
) -> list[np.ndarray]:
    """
    Apply structure motion to unguidanced frames.
    
    Follows WGP.py patterns for video loading, flow extraction, and frame processing.
    
    Args:
        frames_for_guide_list: Current guide frames
        guidance_tracker: Tracks which frames have guidance
        structure_video_path: Path to structure video
        structure_video_treatment: "adjust" or "clip"
        parsed_res_wh: Target resolution (width, height)
        fps_helpers: Target FPS
        dprint: Debug print function
        
    Returns:
        Updated frames list with structure motion applied
    """
    # Get unguidanced ranges from tracker
    unguidanced_ranges = guidance_tracker.get_unguidanced_ranges()
    
    if not unguidanced_ranges:
        dprint(f"[STRUCTURE_VIDEO] No unguidanced frames found")
        return frames_for_guide_list
    
    # Calculate total frames needed
    total_unguidanced = sum(end - start + 1 for start, end in unguidanced_ranges)
    
    dprint(f"[STRUCTURE_VIDEO] Processing {total_unguidanced} unguidanced frames across {len(unguidanced_ranges)} ranges")
    
    try:
        # --- Step 1: Load structure video frames (WGP pattern) ---
        # Need N+1 frames to generate N optical flows
        structure_frames = load_structure_video_frames(
            structure_video_path,
            target_frame_count=total_unguidanced + 1,
            target_fps=fps_helpers,
            target_resolution=parsed_res_wh,
            dprint=dprint
        )
        
        if len(structure_frames) < 2:
            dprint(f"[WARNING] Structure video has insufficient frames ({len(structure_frames)}). Need at least 2.")
            return frames_for_guide_list
        
        # --- Step 2: Extract optical flow (WGP pattern) ---
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
                
                if flow_idx >= len(flow_fields):
                    dprint(f"[WARNING] Exhausted flow fields at frame {frame_idx}. Using last available flow.")
                    flow = flow_fields[-1]
                else:
                    flow = flow_fields[flow_idx]
                    flow_idx += 1
                
                # Warp current frame using flow
                warped_frame = apply_optical_flow_warp(
                    current_frame,
                    flow,
                    parsed_res_wh
                )
                
                updated_frames[frame_idx] = warped_frame
                current_frame = warped_frame  # Next warp uses this result
        
        dprint(f"[STRUCTURE_VIDEO] Applied structure motion to {total_unguidanced} frames using {flow_idx} flow fields")
        
        return updated_frames
    
    except Exception as e:
        dprint(f"[ERROR] Structure motion application failed: {e}")
        import traceback
        traceback.print_exc()
        # Return original frames unchanged
        return frames_for_guide_list
```

---

## Integration into create_guide_video_for_travel_segment

```python
def create_guide_video_for_travel_segment(
    # ... existing parameters ...
    structure_video_path: str | None = None,
    structure_video_treatment: str = "adjust",
    # ... other parameters ...
    *,
    dprint=print
) -> Path | None:
    """Creates the guide video with optional structure motion."""
    
    try:
        # Initialize guidance tracker
        guidance_tracker = GuidanceTracker(total_frames_for_segment)
        
        # Initialize frames list (all gray initially)
        gray_frame_bgr = sm_create_color_frame(parsed_res_wh, (128, 128, 128))
        frames_for_guide_list = [gray_frame_bgr.copy() for _ in range(total_frames_for_segment)]
        
        # --- Build guide video and track guidance ---
        # (Same as revised spec - mark frames as guided during construction)
        
        # Handle overlap frames
        if frame_overlap_from_previous > 0:
            # ... populate overlap frames ...
            for i in range(len(overlap_frames)):
                frames_for_guide_list[i] = overlap_frames[i]
                guidance_tracker.mark_single_frame(i)
        
        # Handle keyframe guidance
        # ... populate keyframe fades ...
        # guidance_tracker.mark_guided(start, end) for fade ranges
        
        # --- Apply structure motion to unguidanced frames ---
        if structure_video_path:
            dprint(f"[GUIDANCE_TRACK] Pre-structure guidance summary:")
            dprint(guidance_tracker.debug_summary())
            
            frames_for_guide_list = apply_structure_motion_with_tracking(
                frames_for_guide_list=frames_for_guide_list,
                guidance_tracker=guidance_tracker,
                structure_video_path=structure_video_path,
                structure_video_treatment=structure_video_treatment,
                parsed_res_wh=parsed_res_wh,
                fps_helpers=fps_helpers,
                dprint=dprint
            )
            
            # Mark structure motion frames as guided
            unguidanced_ranges = guidance_tracker.get_unguidanced_ranges()
            for start_idx, end_idx in unguidanced_ranges:
                guidance_tracker.mark_guided(start_idx, end_idx)
            
            dprint(f"[GUIDANCE_TRACK] Post-structure guidance summary:")
            dprint(guidance_tracker.debug_summary())
        
        # --- Create video from frames ---
        guide_video_file_path = create_video_from_frames_list(
            frames_for_guide_list,
            actual_guide_video_path,
            fps_helpers,
            parsed_res_wh
        )
        
        return guide_video_file_path if guide_video_file_path and guide_video_file_path.exists() else None
    
    except Exception as e:
        dprint(f"ERROR creating guide video: {e}")
        traceback.print_exc()
        return None
```

---

## Key Improvements from WGP Patterns

### ✅ 1. Proper Decord Conversion
- Use `get_resampled_video(bridge='torch')` for efficient GPU loading
- Convert with `.cpu().numpy()` following WGP pattern (line 3826)
- Handle torch tensors properly before numpy operations

### ✅ 2. Robust Flow Extraction
- Use `FlowAnnotator` directly (WGP pattern line 3643-3648)
- Built-in `convert_to_numpy()` handles multiple input types
- Returns N-1 flows for N frames (documented clearly)

### ✅ 3. Fixed Math
- Proper index mapping: `source_idx_float = target_idx * (source_count - 1) / (target_count - 1)`
- No off-by-one errors
- Handles edge cases (single frame, exact matches)

### ✅ 4. Correct Grid Creation
- Use `np.meshgrid(indexing='xy')` for proper X/Y coordinates
- Scale flow vectors when resizing: `flow[:,:,0] *= w_ratio`, `flow[:,:,1] *= h_ratio`
- Border handling with `cv2.BORDER_REPLICATE`

### ✅ 5. Resolution Handling
- Pre-resize frames to target resolution (avoids flow scaling issues)
- Consistent resolution throughout pipeline
- Explicit width/height ordering

---

## Testing Checklist

- [ ] Test with structure video same length as unguidanced frames
- [ ] Test with structure video shorter than needed (padding/interpolation)
- [ ] Test with structure video longer than needed (clipping)
- [ ] Test with multiple unguidanced ranges in one segment
- [ ] Test with no unguidanced frames (should skip gracefully)
- [ ] Test with structure video at different resolution
- [ ] Test with structure video at different FPS
- [ ] Verify no memory leaks from decord/torch tensors
- [ ] Verify flow fields are properly cached/released
- [ ] Visual inspection: motion continuity across ranges
- [ ] Visual inspection: anchor frame transitions smooth

---

## Error Handling

```python
# Guard against missing model weights
if not Path("ckpts/flow/raft-things.pth").exists():
    raise FileNotFoundError(
        "RAFT model weights not found. "
        "Download from: https://github.com/princeton-vl/RAFT"
    )

# Guard against corrupted structure video
try:
    frames = load_structure_video_frames(...)
except Exception as e:
    dprint(f"[ERROR] Could not load structure video: {e}")
    return frames_for_guide_list  # Continue without structure motion

# Guard against flow extraction failure
try:
    flow_fields, _ = extract_optical_flow_from_frames(...)
except Exception as e:
    dprint(f"[ERROR] Optical flow extraction failed: {e}")
    return frames_for_guide_list  # Continue without structure motion
```

---

## Summary

This final spec follows WGP.py's battle-tested patterns:

1. ✅ **Decord handling**: Use `get_resampled_video(bridge='torch')` + `.cpu().numpy()`
2. ✅ **Flow extraction**: Use `FlowAnnotator` with built-in type conversion
3. ✅ **Fixed math**: Proper interpolation without off-by-one errors
4. ✅ **Fixed grids**: Correct meshgrid creation with proper flow scaling
5. ✅ **Guard lengths**: Check frame counts, handle N-1 flows for N frames
6. ✅ **Error handling**: Graceful fallbacks, don't crash entire pipeline

Ready for implementation! 🎬

