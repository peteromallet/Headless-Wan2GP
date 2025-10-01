# Structure Video Guidance - Usage Guide

## Overview

Structure video guidance allows you to apply motion patterns from a reference video to the unguidanced (gray) frames in your travel video guide. This creates more dynamic and interesting transitions between keyframes.

## How It Works

1. **Guidance Tracking**: The system tracks which frames have guidance (overlaps, keyframes) vs which frames are unguidanced (gray placeholders)
2. **Motion Extraction**: Optical flow is extracted from your structure video using RAFT
3. **Selective Application**: Motion is applied ONLY to unguidanced frames, preserving your keyframes and overlaps
4. **Progressive Warping**: Each frame is progressively warped based on the previous frame's result

## Parameters

### `structure_video_path` (string, optional)
Path or URL to the structure video file. If not provided, no structure motion is applied.

**Supports:**
- Local file paths: `/path/to/my/structure_video.mp4`
- HTTP/HTTPS URLs: `https://example.com/structure.mp4`

URLs are automatically downloaded to the output directory before processing.

**Example:**
```python
"structure_video_path": "/path/to/my/structure_video.mp4"
# OR
"structure_video_path": "https://example.com/structure.mp4"
```

### `structure_video_treatment` (string, default: "adjust")
How to handle mismatches between structure video length and unguidanced frame count.

- **"adjust"**: Temporally interpolate the structure video to match exactly
  - Best for: Ensuring all unguidanced frames receive motion
  - Result: Smooth, continuous motion across all frames
  
- **"clip"**: Use structure video as-is, truncate if too long, stop when exhausted if too short
  - Best for: Preserving exact timing of structure video motion
  - Result: Some frames may remain gray if structure video is too short

**Example:**
```python
"structure_video_treatment": "adjust"  # or "clip"
```

### `structure_video_motion_strength` (float, default: 1.0)
Multiplier for motion intensity.

- `0.0`: No motion (frames stay at anchor)
- `1.0`: Full motion from structure video
- `> 1.0`: Amplified motion

**Example:**
```python
"structure_video_motion_strength": 0.7  # Subtle motion
"structure_video_motion_strength": 1.5  # Exaggerated motion
```

## Usage Examples

### Basic Usage

Add these parameters to your orchestrator payload when creating a travel video:

```python
orchestrator_payload = {
    # ... existing parameters ...
    "structure_video_path": "/path/to/structure.mp4",
    "structure_video_treatment": "adjust",
    "structure_video_motion_strength": 1.0,
}
```

### Example 1: Subtle Camera Pan

```python
{
    "structure_video_path": "./samples/slow_pan_right.mp4",
    "structure_video_treatment": "adjust",
    "structure_video_motion_strength": 0.5,  # Gentle motion
}
```

### Example 2: Cinematic Zoom

```python
{
    "structure_video_path": "./samples/zoom_in.mp4",
    "structure_video_treatment": "clip",  # Preserve exact zoom timing
    "structure_video_motion_strength": 1.0,
}
```

### Example 3: Dramatic Movement

```python
{
    "structure_video_path": "./samples/swirl_motion.mp4",
    "structure_video_treatment": "adjust",
    "structure_video_motion_strength": 1.3,  # Amplified for drama
}
```

### Example 4: Using a URL

```python
{
    "structure_video_path": "https://example.com/my-structure-video.mp4",
    "structure_video_treatment": "adjust",
    "structure_video_motion_strength": 1.0,
}
```
*Note: The video will be automatically downloaded to your output directory*

## Implementation Flow

```
1. Orchestrator receives parameters
   ├─> Validates structure_video_path exists
   ├─> Validates treatment mode
   └─> Passes to all segment tasks

2. Each Segment Handler
   ├─> Builds guide video (overlaps + keyframe fades)
   ├─> Tracks which frames are guided
   └─> Calls structure motion application

3. Structure Motion Application
   ├─> Identifies unguidanced frame ranges
   ├─> Loads structure video frames
   ├─> Extracts optical flow (RAFT)
   ├─> Adjusts flow count (interpolate or clip)
   ├─> Progressively warps each unguidanced frame
   └─> Marks warped frames as guided

4. Result
   └─> Guide video with motion-filled gaps
```

## Debugging

Enable debug logging to see guidance tracking:

```python
orchestrator_payload = {
    # ... parameters ...
    "debug_mode_enabled": True,
}
```

You'll see output like:

```
[GUIDANCE_TRACK] Pre-structure guidance summary:
  0: ████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░
 Guided frames: 20/73
 Unguidanced ranges: 1
 Ranges needing structure motion:
   - Frames 20-72 (53 frames)

[STRUCTURE_VIDEO] Processing 53 unguidanced frames across 1 ranges
[STRUCTURE_VIDEO] Loaded 54 frames from structure.mp4
[OPTICAL_FLOW] Extracted 53 optical flow fields
[STRUCTURE_VIDEO] Applied 53 flows to 53 frames

[GUIDANCE_TRACK] Post-structure guidance summary:
  0: █████████████████████████████████████████████████████████████████████
 Guided frames: 73/73
 Unguidanced ranges: 0
```

## Requirements

- RAFT model weights at `Wan2GP/ckpts/flow/raft-things.pth`
- Structure video should be:
  - Similar resolution to your target (will be resized automatically)
  - Similar FPS (will be resampled automatically)
  - At least 2 frames long

## Tips

1. **Structure Video Selection**:
   - Use videos with clear, consistent motion
   - Avoid videos with rapid cuts or scene changes
   - Simple motions (pans, zooms) work better than complex ones

2. **Motion Strength**:
   - Start with 1.0 and adjust based on results
   - Lower values (0.3-0.7) for subtle, natural motion
   - Higher values (1.2-2.0) for stylized, dramatic effects

3. **Treatment Mode**:
   - Use "adjust" for most cases (ensures all frames filled)
   - Use "clip" when timing of structure video is important
   - "clip" mode useful for matching specific beats or rhythms

4. **Performance**:
   - Optical flow extraction is GPU-intensive (~200ms per flow)
   - Flows are cached and reused across all unguidanced ranges
   - Consider using shorter structure videos for faster processing

## Troubleshooting

### Issue: "Structure video not found"
**Solution**: Check the path is absolute and file exists

### Issue: No motion visible
**Possible causes:**
- `motion_strength` set to 0.0
- Structure video has minimal motion
- All frames already have guidance (no unguidanced gaps)

**Solution**: Check guidance tracking debug output, try different structure video

### Issue: Motion looks stretched/distorted
**Possible causes:**
- Resolution mismatch between structure and target
- Very high `motion_strength` value

**Solution**: Use structure video close to target resolution, reduce motion_strength

### Issue: "Need at least 2 frames for optical flow"
**Solution**: Structure video is too short or couldn't be loaded

## Advanced: Per-Segment Structure Videos

You can also specify structure video parameters per segment (overrides orchestrator-level):

```python
segment_payload = {
    # ... segment parameters ...
    "structure_video_path": "/path/to/segment_specific_structure.mp4",
    "structure_video_treatment": "clip",
    "structure_video_motion_strength": 0.8,
}
```

This allows different motion patterns for different parts of your journey!

## Implementation Details

See `STRUCTURE_VIDEO_GUIDANCE_COMPLETE.md` for the complete technical specification.

Key files:
- `source/structure_video_guidance.py`: Core implementation
- `source/video_utils.py`: Integration into guide video creation
- `source/travel_segment_processor.py`: Parameter passing
- `source/sm_functions/travel_between_images.py`: Orchestrator integration

