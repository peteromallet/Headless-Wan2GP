# Structure Video Guidance - Revised Approach

## Critical Fix: Logic-Based Guidance Tracking

### Problem with Original Approach

The spec proposed detecting unguidanced frames by inspecting pixel values (checking if frames are gray). This is fragile because:

1. **Uint8 wraparound:** `np.abs(frame - 128)` doesn't work reliably with uint8 data
2. **Fades aren't pure gray:** Fade transitions create near-gray values that fail threshold checks
3. **Color adjustments:** Saturation/brightness adjustments change "gray" frames to non-gray
4. **Fade edges:** First/last frames of fades might be close enough to 128 to be misclassified

### Solution: Track Guidance State During Construction

Instead of inspecting pixels, **maintain a parallel tracking structure** that records which frames have guidance as we build the guide video.

---

## Revised Implementation

### 1. Guidance Tracker Data Structure

```python
from dataclasses import dataclass
from typing import List, Tuple

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
    
    def get_anchor_frame_index(self, unguidanced_range_start: int) -> int | None:
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
```

### 2. Modified Guide Video Creation

Update `create_guide_video_for_travel_segment` to track guidance as it builds:

```python
def create_guide_video_for_travel_segment(
    segment_idx_for_logging: int,
    end_anchor_image_index: int,
    is_first_segment_from_scratch: bool,
    total_frames_for_segment: int,
    parsed_res_wh: tuple[int, int],
    fps_helpers: int,
    input_images_resolved_for_guide: list[str],
    path_to_previous_segment_video_output_for_guide: str | None,
    output_target_dir: Path,
    guide_video_base_name: str,
    segment_image_download_dir: Path | None,
    task_id_for_logging: str,
    full_orchestrator_payload: dict,
    segment_params: dict,
    single_image_journey: bool = False,
    predefined_output_path: Path | None = None,
    structure_video_path: str | None = None,
    structure_video_treatment: str = "adjust",
    *,
    dprint=print
) -> Path | None:
    """Creates the guide video with optional structure motion, tracking guidance state."""
    
    try:
        # Initialize guidance tracker
        guidance_tracker = GuidanceTracker(total_frames_for_segment)
        
        # Initialize frames list (all gray initially)
        gray_frame_bgr = sm_create_color_frame(parsed_res_wh, (128, 128, 128))
        frames_for_guide_list = [gray_frame_bgr.copy() for _ in range(total_frames_for_segment)]
        
        # --- STEP 1: Handle overlap frames from previous segment ---
        frame_overlap_from_previous = segment_params.get("frame_overlap_from_previous", 0)
        
        if frame_overlap_from_previous > 0 and path_to_previous_segment_video_output_for_guide:
            try:
                prev_frames = sm_extract_frames_from_video(
                    path_to_previous_segment_video_output_for_guide,
                    dprint_func=dprint
                )
                
                if prev_frames and len(prev_frames) >= frame_overlap_from_previous:
                    # Copy last N frames from previous segment
                    overlap_frames = prev_frames[-frame_overlap_from_previous:]
                    
                    for i, overlap_frame in enumerate(overlap_frames):
                        if i < total_frames_for_segment:
                            frames_for_guide_list[i] = overlap_frame
                            # CRITICAL: Mark these frames as guided
                            guidance_tracker.mark_single_frame(i)
                    
                    dprint(f"[GUIDANCE_TRACK] Marked frames 0-{len(overlap_frames)-1} as guided (overlap)")
                else:
                    dprint(f"[WARNING] Previous segment has insufficient frames for {frame_overlap_from_previous}-frame overlap")
            except Exception as e_overlap:
                dprint(f"[WARNING] Failed to extract overlap frames: {e_overlap}")
        
        # --- STEP 2: Handle keyframe guidance (fades between images) ---
        
        # Check for consolidated keyframe positions (frame consolidation optimization)
        consolidated_keyframe_positions = segment_params.get("consolidated_keyframe_positions")
        
        if consolidated_keyframe_positions and not single_image_journey:
            # Consolidated segments: explicit keyframe positions provided
            dprint(f"[GUIDANCE_TRACK] Using consolidated keyframe positions: {consolidated_keyframe_positions}")
            
            # Load keyframe images
            keyframe_images = []
            for img_path in input_images_resolved_for_guide:
                img = load_and_resize_image(img_path, parsed_res_wh)
                keyframe_images.append(img)
            
            # Apply keyframes at specified positions
            for kf_idx, kf_position in enumerate(consolidated_keyframe_positions):
                if 0 <= kf_position < total_frames_for_segment and kf_idx < len(keyframe_images):
                    frames_for_guide_list[kf_position] = keyframe_images[kf_idx]
                    guidance_tracker.mark_single_frame(kf_position)
                    dprint(f"[GUIDANCE_TRACK] Marked frame {kf_position} as guided (keyframe {kf_idx})")
            
            # Interpolate between keyframes
            for i in range(len(consolidated_keyframe_positions) - 1):
                start_pos = consolidated_keyframe_positions[i]
                end_pos = consolidated_keyframe_positions[i + 1]
                
                if start_pos < end_pos - 1:  # Only interpolate if there's a gap
                    start_img = keyframe_images[i]
                    end_img = keyframe_images[i + 1]
                    
                    # Interpolate frames between keyframes
                    for frame_idx in range(start_pos + 1, end_pos):
                        alpha = (frame_idx - start_pos) / (end_pos - start_pos)
                        blended = cv2.addWeighted(start_img, 1 - alpha, end_img, alpha, 0)
                        frames_for_guide_list[frame_idx] = blended
                        guidance_tracker.mark_single_frame(frame_idx)
                    
                    dprint(f"[GUIDANCE_TRACK] Marked frames {start_pos+1}-{end_pos-1} as guided (interpolation)")
        
        else:
            # Standard segments: fade from start to end anchor
            start_anchor_img = None
            end_anchor_img = None
            
            # Determine start anchor
            if not is_first_segment_from_scratch and path_to_previous_segment_video_output_for_guide:
                # Extract last frame from previous segment as start anchor
                start_anchor_img = sm_extract_last_frame_as_image(
                    path_to_previous_segment_video_output_for_guide,
                    output_target_dir,
                    task_id_for_logging
                )
            elif len(input_images_resolved_for_guide) > 0:
                # Use first input image
                start_anchor_img = load_and_resize_image(
                    input_images_resolved_for_guide[0],
                    parsed_res_wh
                )
            
            # Determine end anchor
            if 0 <= end_anchor_image_index < len(input_images_resolved_for_guide):
                end_anchor_img = load_and_resize_image(
                    input_images_resolved_for_guide[end_anchor_image_index],
                    parsed_res_wh
                )
            
            # Apply fade with guidance tracking
            if start_anchor_img is not None and end_anchor_img is not None:
                fade_start_frame = frame_overlap_from_previous
                fade_end_frame = total_frames_for_segment - 1
                fade_length = fade_end_frame - fade_start_frame + 1
                
                if fade_length > 0:
                    for i in range(fade_length):
                        frame_idx = fade_start_frame + i
                        alpha = i / (fade_length - 1) if fade_length > 1 else 1.0
                        
                        # Apply fade curve (ease_in_out, etc.)
                        # ... fade curve logic ...
                        
                        blended = cv2.addWeighted(start_anchor_img, 1 - alpha, end_anchor_img, alpha, 0)
                        frames_for_guide_list[frame_idx] = blended
                        guidance_tracker.mark_single_frame(frame_idx)
                    
                    dprint(f"[GUIDANCE_TRACK] Marked frames {fade_start_frame}-{fade_end_frame} as guided (fade)")
        
        # --- STEP 3: Apply structure motion to unguidanced frames ---
        
        if structure_video_path:
            unguidanced_ranges = guidance_tracker.get_unguidanced_ranges()
            
            if unguidanced_ranges:
                dprint(f"[STRUCTURE_VIDEO] Guidance summary before structure motion:")
                dprint(guidance_tracker.debug_summary())
                
                total_unguidanced = sum(end - start + 1 for start, end in unguidanced_ranges)
                dprint(f"[STRUCTURE_VIDEO] Applying structure motion to {total_unguidanced} unguidanced frames across {len(unguidanced_ranges)} ranges")
                
                # Extract and apply structure motion
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
                for start_idx, end_idx in unguidanced_ranges:
                    guidance_tracker.mark_guided(start_idx, end_idx)
                
                dprint(f"[STRUCTURE_VIDEO] Guidance summary after structure motion:")
                dprint(guidance_tracker.debug_summary())
            else:
                dprint(f"[STRUCTURE_VIDEO] No unguidanced frames found - structure video not needed")
        
        # --- STEP 4: Create video from frames ---
        
        if predefined_output_path:
            actual_guide_video_path = predefined_output_path
        else:
            actual_guide_video_path = sm_get_unique_target_path(output_target_dir, guide_video_base_name, ".mp4")
        
        guide_video_file_path = create_video_from_frames_list(
            frames_for_guide_list,
            actual_guide_video_path,
            fps_helpers,
            parsed_res_wh
        )
        
        if guide_video_file_path and guide_video_file_path.exists():
            dprint(f"[GUIDANCE_TRACK] Final guide video created with {sum(guidance_tracker.has_guidance)}/{total_frames_for_segment} guided frames")
            return guide_video_file_path
        
        return None
    
    except Exception as e:
        dprint(f"ERROR creating guide video for segment {segment_idx_for_logging}: {e}")
        traceback.print_exc()
        return None
```

### 3. Structure Motion Application with Tracking

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
    Apply structure motion to unguidanced frames, using guidance tracker.
    
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
    import cv2
    from Wan2GP.preprocessing.flow import FlowAnnotator
    
    # Get unguidanced ranges from tracker (not pixel inspection!)
    unguidanced_ranges = guidance_tracker.get_unguidanced_ranges()
    
    if not unguidanced_ranges:
        return frames_for_guide_list
    
    # Calculate total frames needed
    total_unguidanced = sum(end - start + 1 for start, end in unguidanced_ranges)
    
    try:
        # --- Extract optical flow from structure video ---
        
        # Load structure video frames at target resolution
        structure_frames = load_structure_video_frames(
            structure_video_path,
            target_frame_count=total_unguidanced + 1,  # Need N+1 frames for N flows
            target_fps=fps_helpers,
            target_resolution=parsed_res_wh,
            dprint=dprint
        )
        
        if len(structure_frames) < 2:
            dprint(f"[WARNING] Structure video has insufficient frames ({len(structure_frames)}). Skipping structure motion.")
            return frames_for_guide_list
        
        # Extract optical flow
        flow_cfg = {'PRETRAINED_MODEL': 'ckpts/flow/raft-things.pth'}
        flow_annotator = FlowAnnotator(flow_cfg)
        
        # Convert to format expected by RAFT (numpy uint8 [H, W, C])
        structure_frames_np = [frame.asnumpy() if hasattr(frame, 'asnumpy') else frame 
                               for frame in structure_frames]
        
        flow_fields, flow_vis = flow_annotator.forward(structure_frames_np)
        
        dprint(f"[STRUCTURE_VIDEO] Extracted {len(flow_fields)} optical flow fields from {len(structure_frames)} frames")
        
        # --- Adjust flow count to match needed frames ---
        
        flow_fields = adjust_structure_motion_length(
            flow_fields,
            total_unguidanced,
            structure_video_treatment,
            dprint=dprint
        )
        
        # --- Apply flow to unguidanced ranges ---
        
        updated_frames = frames_for_guide_list.copy()
        flow_idx = 0
        
        for range_idx, (start_idx, end_idx) in enumerate(unguidanced_ranges):
            range_length = end_idx - start_idx + 1
            
            # Get anchor frame (last guided frame before this range)
            anchor_idx = guidance_tracker.get_anchor_frame_index(start_idx)
            
            if anchor_idx is not None:
                current_frame = updated_frames[anchor_idx].copy()
                dprint(f"[STRUCTURE_VIDEO] Range {range_idx}: frames {start_idx}-{end_idx}, using anchor from frame {anchor_idx}")
            else:
                # No guided frame before this range - use gray
                current_frame = sm_create_color_frame(parsed_res_wh, (128, 128, 128))
                dprint(f"[STRUCTURE_VIDEO] Range {range_idx}: frames {start_idx}-{end_idx}, using gray anchor (no prior guidance)")
            
            # Apply motion progressively through the range
            for offset in range(range_length):
                frame_idx = start_idx + offset
                
                if flow_idx >= len(flow_fields):
                    dprint(f"[WARNING] Ran out of flow fields at frame {frame_idx}. Using last available flow.")
                    flow = flow_fields[-1]
                else:
                    flow = flow_fields[flow_idx]
                    flow_idx += 1
                
                # Warp current frame using optical flow
                warped_frame = apply_optical_flow_warp(current_frame, flow, parsed_res_wh)
                updated_frames[frame_idx] = warped_frame
                current_frame = warped_frame
        
        dprint(f"[STRUCTURE_VIDEO] Applied structure motion to {total_unguidanced} frames across {len(unguidanced_ranges)} ranges")
        
        return updated_frames
    
    except Exception as e:
        dprint(f"[ERROR] Structure motion application failed: {e}")
        traceback.print_exc()
        # Return original frames unchanged
        return frames_for_guide_list
```

### 4. Helper Functions

```python
def load_structure_video_frames(
    structure_video_path: str,
    target_frame_count: int,
    target_fps: int,
    target_resolution: tuple[int, int],
    dprint=print
) -> list[np.ndarray]:
    """
    Load and preprocess structure video frames.
    
    Loads at target resolution to avoid flow scaling issues.
    """
    from Wan2GP.wgp import get_resampled_video
    import cv2
    
    # Load frames using existing utility
    frames = get_resampled_video(
        structure_video_path,
        start_frame=0,
        max_frames=target_frame_count,
        target_fps=target_fps,
        bridge='torch'  # Returns torch tensors
    )
    
    # Convert to numpy and resize to target resolution
    w, h = target_resolution
    processed_frames = []
    
    for frame in frames:
        # Convert from torch tensor to numpy if needed
        if hasattr(frame, 'asnumpy'):
            frame_np = frame.asnumpy()
        elif hasattr(frame, 'cpu'):
            frame_np = frame.cpu().numpy()
        else:
            frame_np = frame
        
        # Ensure uint8
        if frame_np.dtype != np.uint8:
            frame_np = np.clip(frame_np, 0, 255).astype(np.uint8)
        
        # Resize to target resolution
        frame_resized = cv2.resize(frame_np, (w, h), interpolation=cv2.INTER_LINEAR)
        
        processed_frames.append(frame_resized)
    
    dprint(f"[STRUCTURE_VIDEO] Loaded {len(processed_frames)} frames from {structure_video_path}")
    dprint(f"[STRUCTURE_VIDEO] Preprocessed to resolution {w}x{h}")
    
    return processed_frames


def apply_optical_flow_warp(
    source_frame: np.ndarray,
    flow: np.ndarray,
    target_resolution: tuple[int, int]
) -> np.ndarray:
    """
    Warp a frame using optical flow.
    
    Args:
        source_frame: Frame to warp [H, W, C] uint8
        flow: Optical flow field [H, W, 2] float32
        target_resolution: (width, height)
        
    Returns:
        Warped frame [H, W, C] uint8
    """
    import cv2
    
    w, h = target_resolution
    
    # Ensure flow matches target resolution
    if flow.shape[:2] != (h, w):
        flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # Create coordinate grid
    y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    
    # Apply flow to coordinates
    map_x = (x_coords + flow[:, :, 0]).astype(np.float32)
    map_y = (y_coords + flow[:, :, 1]).astype(np.float32)
    
    # Warp frame
    warped = cv2.remap(
        source_frame,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )
    
    return warped


def adjust_structure_motion_length(
    flow_fields: list[np.ndarray],
    target_frame_count: int,
    treatment: str,
    dprint=print
) -> list[np.ndarray]:
    """
    Adjust number of flow fields to match target frame count.
    
    CORRECTED: Handles edge cases and proper indexing.
    """
    source_count = len(flow_fields)
    
    if source_count == 0:
        raise ValueError("No flow fields provided")
    
    if treatment == "clip":
        if source_count >= target_frame_count:
            return flow_fields[:target_frame_count]
        else:
            # Pad with last flow field (repeat motion) rather than zero
            padding = [flow_fields[-1].copy() for _ in range(target_frame_count - source_count)]
            dprint(f"[STRUCTURE_VIDEO] Padded {len(padding)} flow fields (repeating last flow)")
            return flow_fields + padding
    
    elif treatment == "adjust":
        if source_count == target_frame_count:
            return flow_fields
        
        # Temporal interpolation
        adjusted_flows = []
        
        for target_idx in range(target_frame_count):
            # Map target index to source space
            # CORRECTED: Avoid off-by-one error
            if target_frame_count == 1:
                source_idx_float = 0.0
            else:
                source_idx_float = (target_idx / (target_frame_count - 1)) * (source_count - 1)
            
            source_idx_low = int(np.floor(source_idx_float))
            source_idx_high = int(np.ceil(source_idx_float))
            
            # Clamp to valid range
            source_idx_low = max(0, min(source_idx_low, source_count - 1))
            source_idx_high = max(0, min(source_idx_high, source_count - 1))
            
            if source_idx_low == source_idx_high:
                adjusted_flows.append(flow_fields[source_idx_low].copy())
            else:
                # Linear interpolation between flows
                alpha = source_idx_float - source_idx_low
                flow_low = flow_fields[source_idx_low]
                flow_high = flow_fields[source_idx_high]
                interpolated_flow = (1 - alpha) * flow_low + alpha * flow_high
                adjusted_flows.append(interpolated_flow)
        
        dprint(f"[STRUCTURE_VIDEO] Adjusted flow count from {source_count} to {target_frame_count} via interpolation")
        return adjusted_flows
    
    else:
        raise ValueError(f"Invalid treatment: {treatment}. Must be 'adjust' or 'clip'")
```

---

## Key Advantages

### 1. **Robustness**
- No dependency on pixel values or thresholds
- Works regardless of color adjustments, fades, or compression artifacts
- Explicit tracking prevents misclassification

### 2. **Debuggability**
```python
# Easy to visualize guidance state
print(guidance_tracker.debug_summary())

# Output:
#   0: ████████████████████░░░░░░░░░░
#  30: ░░░░░░░░░░░░░░░░░░░░████████████
#  60: ████████████
# Guided frames: 48/73
# Unguidanced ranges: 1
# Ranges needing structure motion:
#   - Frames 20-40 (21 frames)
```

### 3. **Maintainability**
- Single source of truth for guidance state
- Easy to add new guidance types (e.g., pose guidance, depth guidance)
- Clear separation between guidance construction and structure motion application

### 4. **Performance**
- No expensive pixel-level comparison loops
- O(1) lookup for guidance state
- Efficient range identification

### 5. **Correctness**
- Anchor frame lookup is reliable (uses logical state, not pixel similarity)
- Clear handling of edge cases (no guided frames before range, etc.)
- Proper indexing without off-by-one errors

---

## Debug Output Example

```
[GUIDANCE_TRACK] Marked frames 0-19 as guided (overlap)
[GUIDANCE_TRACK] Marked frame 20 as guided (keyframe 0)
[GUIDANCE_TRACK] Marked frames 21-51 as guided (interpolation)
[GUIDANCE_TRACK] Marked frame 52 as guided (keyframe 1)
[GUIDANCE_TRACK] Marked frames 53-72 as guided (interpolation)
[STRUCTURE_VIDEO] Guidance summary before structure motion:
  0: ████████████████████████████████████████████████████████████████████████
 Guided frames: 73/73
 Unguidanced ranges: 0
[STRUCTURE_VIDEO] No unguidanced frames found - structure video not needed
[GUIDANCE_TRACK] Final guide video created with 73/73 guided frames
```

Or with gaps:

```
[GUIDANCE_TRACK] Marked frames 0-19 as guided (overlap)
[GUIDANCE_TRACK] Marked frame 30 as guided (keyframe 0)
[GUIDANCE_TRACK] Marked frame 60 as guided (keyframe 1)
[STRUCTURE_VIDEO] Guidance summary before structure motion:
  0: ████████████████████░░░░░░░░░░
 30: ████░░░░░░░░░░░░░░░░░░░░░░░░░░
 60: ████░░░░░░░░░
 Guided frames: 25/73
 Unguidanced ranges: 2
 Ranges needing structure motion:
   - Frames 20-29 (10 frames)
   - Frames 31-59 (29 frames)
[STRUCTURE_VIDEO] Range 0: frames 20-29, using anchor from frame 19
[STRUCTURE_VIDEO] Range 1: frames 31-59, using anchor from frame 30
[STRUCTURE_VIDEO] Applied structure motion to 39 frames across 2 ranges
[GUIDANCE_TRACK] Final guide video created with 73/73 guided frames
```

---

## Summary

**Instead of pixel inspection, track guidance state explicitly during construction:**

1. Create `GuidanceTracker` at the start
2. Mark frames as guided when populating them (overlaps, fades, keyframes)
3. Query tracker for unguidanced ranges
4. Apply structure motion only to those ranges
5. Mark structure motion frames as guided

This approach is **robust, debuggable, and maintainable** – no fragile threshold checks!

