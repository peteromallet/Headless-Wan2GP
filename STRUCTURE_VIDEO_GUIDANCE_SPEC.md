# Structure Video Guidance Integration Specification

## Overview

This document specifies how to integrate structure video guidance into the travel video creation system. Structure video guidance allows users to provide a reference video from which motion patterns are extracted and applied to frames in the guide video that don't already have keyframe guidance.

## Current Travel Video Creation Flow

### 1. Orchestrator Task (`_handle_travel_orchestrator_task`)
- Receives journey parameters (images, prompts, frame counts, overlaps)
- Creates N segment tasks and 1 stitch task
- Each segment has:
  - `segment_frames_target`: Total frames for this segment (e.g., 73)
  - `frame_overlap_from_previous`: Frames that overlap with previous segment (e.g., 20)
  - VACE refs: Keyframe images at specific positions

### 2. Segment Task (`_handle_travel_segment_task`)
- Processes one segment of the journey
- Creates guide video using `TravelSegmentProcessor`
- Guide video contains:
  - Frames from previous segment's output (for continuity)
  - Fades between keyframe images
  - Gray placeholder frames where no guidance exists

### 3. Guide Video Creation (`create_guide_video_for_travel_segment`)
- Creates a list of frames: `frames_for_guide_list[total_frames_for_segment]`
- Initially all frames are gray (128, 128, 128)
- Process:
  1. **Overlap frames** (first N frames): Copy from previous segment output
  2. **Keyframe guidance**: Fade between input images at specific positions
  3. **Remaining frames**: Stay gray (no guidance)

**Example for 73-frame segment with 20-frame overlap:**
```
[PREV_20] [FADE_START_IMG → END_IMG (frames 20-72)] [END_IMG]
```

**Example with gaps:**
```
[PREV_20] [IMG_A] [GRAY_30] [IMG_B] [GRAY_20] [IMG_C]
       ^          ^         ^       ^         ^
    overlap   keyframe   no      keyframe   keyframe
              @ frame 20 guidance @ frame 51 @ frame 72
```

## Proposed Structure Video Guidance

### Core Concept

Extract motion patterns from a user-provided structure video and apply them to gray (unguidanced) frames in the guide video, preserving existing keyframe guidance and overlaps.

### Input Parameters

Add to orchestrator payload:
```python
{
    "structure_video_path": str | None,  # Path to structure video
    "structure_video_treatment": str,     # "adjust" or "clip"
    # ... existing parameters
}
```

### Processing Flow

#### Step 1: Identify Unguidanced Frame Ranges

In `create_guide_video_for_travel_segment`, after creating the initial guide video with overlaps and keyframe fades, identify which frames are still gray (unguidanced).

```python
def identify_unguidanced_ranges(frames_for_guide_list: list, 
                                gray_threshold: int = 130) -> list[tuple[int, int]]:
    """
    Identify continuous ranges of gray/unguidanced frames.
    
    Returns:
        List of (start_idx, end_idx) tuples for unguidanced ranges
        
    Example:
        frames = [PREV, PREV, IMG_A, GRAY, GRAY, GRAY, IMG_B, GRAY, GRAY]
        returns: [(3, 5), (7, 8)]  # Two ranges of gray frames
    """
    unguidanced_ranges = []
    in_gray_range = False
    range_start = None
    
    for i, frame in enumerate(frames_for_guide_list):
        # Check if frame is gray (all channels near 128)
        is_gray = np.all(np.abs(frame - 128) < gray_threshold)
        
        if is_gray and not in_gray_range:
            range_start = i
            in_gray_range = True
        elif not is_gray and in_gray_range:
            unguidanced_ranges.append((range_start, i - 1))
            in_gray_range = False
    
    # Handle case where segment ends with gray frames
    if in_gray_range:
        unguidanced_ranges.append((range_start, len(frames_for_guide_list) - 1))
    
    return unguidanced_ranges
```

#### Step 2: Extract Motion from Structure Video

Use the existing RAFT optical flow extractor to extract motion patterns:

```python
def extract_structure_motion(structure_video_path: str,
                            target_frame_count: int,
                            target_fps: int = 16) -> list[np.ndarray]:
    """
    Extract motion patterns (optical flow) from structure video.
    
    Args:
        structure_video_path: Path to structure video
        target_frame_count: Number of frames needed
        target_fps: Target FPS for resampling
        
    Returns:
        List of optical flow fields (H x W x 2) representing motion between frames
    """
    from Wan2GP.preprocessing.flow import FlowAnnotator
    from Wan2GP.wgp import get_resampled_video
    
    # Load and resample structure video
    structure_frames = get_resampled_video(
        structure_video_path,
        start_frame=0,
        max_frames=target_frame_count + 1,  # Need N+1 frames for N flows
        target_fps=target_fps
    )
    
    # Extract optical flow between consecutive frames
    flow_cfg = {
        'PRETRAINED_MODEL': 'ckpts/flow/raft-things.pth'
    }
    flow_annotator = FlowAnnotator(flow_cfg)
    flow_fields, flow_vis = flow_annotator.forward(structure_frames)
    
    return flow_fields, flow_vis  # flow_vis for debugging
```

#### Step 3: Handle Frame Count Mismatches

The structure video may have a different number of frames than needed. Handle this based on `structure_video_treatment`:

```python
def adjust_structure_motion_length(flow_fields: list[np.ndarray],
                                   target_frame_count: int,
                                   treatment: str) -> list[np.ndarray]:
    """
    Adjust structure motion to match target frame count.
    
    Args:
        flow_fields: Extracted optical flow fields
        target_frame_count: Total frames needed across all unguidanced ranges
        treatment: "adjust" or "clip"
        
    Returns:
        Adjusted list of flow fields
    """
    source_count = len(flow_fields)
    
    if treatment == "clip":
        # Simple clipping: use what we have, truncate or pad
        if source_count >= target_frame_count:
            return flow_fields[:target_frame_count]
        else:
            # Pad with zeros (no motion) if too short
            padding = [np.zeros_like(flow_fields[0])] * (target_frame_count - source_count)
            return flow_fields + padding
            
    elif treatment == "adjust":
        # Temporal interpolation to match target length
        if source_count == target_frame_count:
            return flow_fields
            
        # Use linear interpolation to resample flow fields
        import cv2
        adjusted_flows = []
        
        for target_idx in range(target_frame_count):
            # Map target index to source space
            source_idx_float = (target_idx / target_frame_count) * source_count
            source_idx_low = int(np.floor(source_idx_float))
            source_idx_high = min(int(np.ceil(source_idx_float)), source_count - 1)
            
            if source_idx_low == source_idx_high:
                adjusted_flows.append(flow_fields[source_idx_low].copy())
            else:
                # Interpolate between two flow fields
                alpha = source_idx_float - source_idx_low
                flow_low = flow_fields[source_idx_low]
                flow_high = flow_fields[source_idx_high]
                interpolated_flow = (1 - alpha) * flow_low + alpha * flow_high
                adjusted_flows.append(interpolated_flow)
        
        return adjusted_flows
```

#### Step 4: Apply Structure Motion to Guide Video

Replace gray frames with motion-guided frames:

```python
def apply_structure_motion_to_guide(frames_for_guide_list: list[np.ndarray],
                                   unguidanced_ranges: list[tuple[int, int]],
                                   flow_fields: list[np.ndarray],
                                   parsed_res_wh: tuple[int, int]) -> list[np.ndarray]:
    """
    Apply extracted motion patterns to unguidanced frames in guide video.
    
    Args:
        frames_for_guide_list: Current guide frames (with overlaps and keyframes)
        unguidanced_ranges: List of (start, end) indices for gray frames
        flow_fields: Extracted and adjusted optical flow fields
        parsed_res_wh: Target resolution (width, height)
        
    Returns:
        Updated guide frames with structure motion applied
    """
    import cv2
    
    # Calculate total unguidanced frames
    total_unguidanced = sum(end - start + 1 for start, end in unguidanced_ranges)
    
    # Ensure we have enough flow fields
    if len(flow_fields) < total_unguidanced:
        raise ValueError(f"Not enough flow fields ({len(flow_fields)}) for unguidanced frames ({total_unguidanced})")
    
    flow_idx = 0
    updated_frames = frames_for_guide_list.copy()
    
    for start_idx, end_idx in unguidanced_ranges:
        range_length = end_idx - start_idx + 1
        
        # Get anchor frame (last non-gray frame before this range)
        if start_idx > 0:
            anchor_frame = updated_frames[start_idx - 1].copy()
        else:
            # If range starts at frame 0, use first keyframe or gray
            anchor_frame = create_color_frame(parsed_res_wh, (128, 128, 128))
        
        # Apply motion progressively through the range
        current_frame = anchor_frame
        for offset in range(range_length):
            frame_idx = start_idx + offset
            
            # Get optical flow for this transition
            flow = flow_fields[flow_idx]
            flow_idx += 1
            
            # Resize flow to match target resolution if needed
            if flow.shape[:2] != (parsed_res_wh[1], parsed_res_wh[0]):
                flow = cv2.resize(flow, parsed_res_wh, interpolation=cv2.INTER_LINEAR)
                # Scale flow vectors proportionally
                flow[:, :, 0] *= parsed_res_wh[0] / flow.shape[1]
                flow[:, :, 1] *= parsed_res_wh[1] / flow.shape[0]
            
            # Apply optical flow to warp the current frame
            h, w = parsed_res_wh[1], parsed_res_wh[0]
            flow_map = np.zeros((h, w, 2), dtype=np.float32)
            flow_map[:, :, 0] = np.arange(w)  # x coordinates
            flow_map[:, :, 1] = np.arange(h)[:, np.newaxis]  # y coordinates
            
            # Add flow to coordinates
            flow_map += flow
            
            # Warp frame using flow
            warped_frame = cv2.remap(
                current_frame,
                flow_map[:, :, 0],
                flow_map[:, :, 1],
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE
            )
            
            updated_frames[frame_idx] = warped_frame
            current_frame = warped_frame
    
    return updated_frames
```

#### Step 5: Handle Frame Overlap Alignment

**Critical:** Segments overlap with previous segments. When applying structure motion, we must account for the fact that:

1. First `frame_overlap_from_previous` frames come from the previous segment's output
2. These overlapping frames are NOT gray - they're real frames from previous generation
3. Structure motion should NOT overwrite overlap frames

**Example:**
```
Segment 1: 73 frames [IMG_A fade to IMG_B]
Segment 2: 73 frames, 20-frame overlap
  - Frames 0-19: From Segment 1's frames 53-72 (overlap)
  - Frames 20-72: New content [IMG_B fade to IMG_C]
```

The `identify_unguidanced_ranges` function already handles this because overlap frames are copied from actual video output (not gray), so they won't be identified as unguidanced.

### Integration Points

#### A. Orchestrator Level

Add structure video parameter validation in `_handle_travel_orchestrator_task`:

```python
# In orchestrator payload
structure_video_path = orchestrator_payload.get("structure_video_path")
structure_video_treatment = orchestrator_payload.get("structure_video_treatment", "adjust")

if structure_video_path:
    # Validate structure video exists and is accessible
    if not Path(structure_video_path).exists():
        raise ValueError(f"Structure video not found: {structure_video_path}")
    
    # Validate treatment option
    if structure_video_treatment not in ["adjust", "clip"]:
        raise ValueError(f"Invalid structure_video_treatment: {structure_video_treatment}. Must be 'adjust' or 'clip'")
    
    dprint(f"[STRUCTURE_VIDEO] Using structure video: {structure_video_path}")
    dprint(f"[STRUCTURE_VIDEO] Frame mismatch treatment: {structure_video_treatment}")

# Pass to segment payloads
segment_payload = {
    # ... existing fields
    "structure_video_path": structure_video_path,
    "structure_video_treatment": structure_video_treatment,
}
```

#### B. Segment Level

Modify `create_guide_video_for_travel_segment` in `video_utils.py`:

```python
def create_guide_video_for_travel_segment(
    # ... existing parameters
    structure_video_path: str | None = None,
    structure_video_treatment: str = "adjust",
    # ... other parameters
) -> Path | None:
    """Creates the guide video with optional structure motion."""
    
    # ... existing guide video creation logic ...
    
    # After creating base guide with overlaps and keyframes
    if structure_video_path:
        dprint(f"[STRUCTURE_VIDEO] Applying structure motion to segment {segment_idx_for_logging}")
        
        # 1. Identify unguidanced ranges
        unguidanced_ranges = identify_unguidanced_ranges(frames_for_guide_list)
        
        if unguidanced_ranges:
            total_unguidanced = sum(end - start + 1 for start, end in unguidanced_ranges)
            dprint(f"[STRUCTURE_VIDEO] Found {len(unguidanced_ranges)} unguidanced ranges, {total_unguidanced} total frames")
            
            # 2. Extract motion from structure video
            flow_fields, flow_vis = extract_structure_motion(
                structure_video_path,
                target_frame_count=total_unguidanced,
                target_fps=fps_helpers
            )
            
            # 3. Adjust motion length if needed
            flow_fields = adjust_structure_motion_length(
                flow_fields,
                total_unguidanced,
                structure_video_treatment
            )
            
            # 4. Apply structure motion
            frames_for_guide_list = apply_structure_motion_to_guide(
                frames_for_guide_list,
                unguidanced_ranges,
                flow_fields,
                parsed_res_wh
            )
            
            # 5. Debug: Save flow visualization if in debug mode
            if debug_mode:
                flow_vis_path = output_target_dir / f"seg{segment_idx_for_logging:02d}_flow_vis.mp4"
                create_video_from_frames_list(flow_vis, flow_vis_path, fps_helpers, parsed_res_wh)
                dprint(f"[STRUCTURE_VIDEO] Saved flow visualization to {flow_vis_path}")
        else:
            dprint(f"[STRUCTURE_VIDEO] No unguidanced frames found in segment {segment_idx_for_logging}")
    
    # ... rest of guide video creation ...
```

### API Usage Example

```python
# Client-side call to create travel video with structure guidance
result = travel_between_images(
    input_images=["image_a.jpg", "image_b.jpg", "image_c.jpg"],
    base_prompt="cinematic landscape",
    negative_prompt="blurry, low quality",
    segment_frames=[73, 73, 73],
    frame_overlap=[20, 20],
    structure_video_path="/path/to/reference_motion.mp4",
    structure_video_treatment="adjust",  # or "clip"
    # ... other parameters
)
```

### Visual Explanation

**Without Structure Video:**
```
Segment Guide Video (73 frames):
[────────────OVERLAP(20)─────────────][FADE IMG_B→IMG_C (53 frames)]
[🎬🎬🎬🎬🎬🎬🎬🎬🎬🎬🎬🎬🎬🎬🎬🎬🎬🎬🎬🎬][🖼️➡️⬜⬜⬜⬜⬜⬜⬜⬜⬜➡️🖼️]
       Previous video frames            Keyframe fade
```

**With Structure Video:**
```
Structure Video: 🌊🌊🌊🌊🌊🌊🌊🌊 (ocean wave motion, 40 frames)

Segment Guide Video (73 frames) with "adjust" treatment:
[────────────OVERLAP(20)─────────────][FADE START][──STRUCTURE MOTION──][FADE END]
[🎬🎬🎬🎬🎬🎬🎬🎬🎬🎬🎬🎬🎬🎬🎬🎬🎬🎬🎬🎬][🖼️][🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊][🖼️]
       Previous video frames          5fr  Structure motion (adjusted)  5fr
                                      fade  from 40→43 frames          fade
```

### Debug Output

When `debug_mode_enabled=True`, save intermediate files:

```
outputs/
  seg00_guide_raw.mp4              # Guide before structure motion
  seg00_guide_flow_vis.mp4         # Optical flow visualization
  seg00_guide_final.mp4            # Final guide with structure motion
  seg00_unguidanced_ranges.json   # [(start, end), ...] ranges
  seg00_structure_motion_log.txt  # Processing details
```

### Error Handling

```python
class StructureVideoError(Exception):
    """Raised when structure video processing fails."""
    pass

# In guide video creation:
try:
    if structure_video_path:
        # Apply structure motion
        pass
except StructureVideoError as e:
    dprint(f"[WARNING] Structure video processing failed: {e}")
    dprint(f"[WARNING] Continuing with standard guide video (gray frames)")
    # Continue without structure motion
except Exception as e:
    # Log but don't fail the entire segment
    dprint(f"[ERROR] Unexpected error in structure video processing: {e}")
    traceback.print_exc()
```

### Performance Considerations

1. **Optical Flow Extraction:** RAFT flow extraction is GPU-intensive. Extract once and cache.
2. **Flow Field Memory:** Each flow field is H×W×2 float32 (~1MB per frame at 512x512). For long segments, consider:
   - Processing in chunks
   - Using float16 precision
   - Streaming from disk

3. **Resizing:** If structure video resolution differs from target, resize flows efficiently using OpenCV.

### Testing Strategy

1. **Unit Tests:**
   - `test_identify_unguidanced_ranges()` - Various frame patterns
   - `test_extract_structure_motion()` - Different video lengths
   - `test_adjust_structure_motion_length()` - Both "adjust" and "clip" modes
   - `test_apply_structure_motion()` - Motion application

2. **Integration Tests:**
   - Single segment with structure video
   - Multi-segment journey with structure video
   - Frame count mismatches (structure video shorter/longer than needed)
   - Edge case: No unguidanced frames (structure video ignored)

3. **Visual Tests:**
   - Compare output videos with/without structure guidance
   - Verify motion continuity across segment boundaries
   - Check overlap frame alignment

### Future Enhancements

1. **Multiple Structure Videos:** Different motion patterns for different segments
2. **Motion Strength:** Blend factor between gray and structure motion (0.0 = gray, 1.0 = full motion)
3. **Motion Looping:** If structure video is short, loop the motion pattern
4. **Smart Motion Extraction:** Only extract motion from specific parts of structure video
5. **Motion Style Transfer:** Apply structure motion while preserving edge/color information

---

## Summary

This specification adds structure video guidance to travel video generation by:

1. **Extracting** optical flow motion from a reference structure video
2. **Identifying** which frames in the guide video lack guidance (are gray)
3. **Applying** the extracted motion to fill those unguidanced frames
4. **Preserving** existing keyframe guidance and segment overlaps

The system respects frame overlaps between segments and provides flexible handling of frame count mismatches through "adjust" (temporal interpolation) or "clip" (truncate/pad) strategies.

