# Structure Video - Multi-Type Preprocessing Design

**Version:** 1.0  
**Status:** Design Proposal  
**Date:** October 2, 2025

---

## Overview

This document proposes extending the structure video guidance system to support multiple preprocessing types beyond optical flow. The system will support:

- **Optical flow** (DEFAULT) - Motion patterns and directional movement (current implementation)
- **Canny edges** - Sharp edge detection for structural guidance
- **Depth maps** - 3D spatial structure and depth information

### Key Design Principles

1. **We run preprocessors ourselves** at orchestrator level - not relying on WGP's internal preprocessing
2. **Optical flow remains the default** - maintains current behavior when `structure_type` is not passed
3. **Pre-warped video approach** - core architecture unchanged:
   - Orchestrator extracts and preprocesses structure video ONCE
   - Segments download and extract their specific slice using frame offsets
   - Each segment passes its slice to WGP as `video_guide`
4. **Strength parameters for all types** - similar to `motion_strength` for flow:
   - `structure_canny_intensity` - adjusts edge boldness
   - `structure_depth_contrast` - adjusts depth map contrast
5. **Backward compatible** - existing code works without modifications

---

## Current Architecture Review

### How Optical Flow Currently Works

1. **Orchestrator Level** (`_handle_travel_orchestrator_task`):
   - Extracts optical flows from structure video using RAFT
   - Creates flow visualization video (colorful flow maps)
   - Uploads to shared storage
   - Passes URL + offsets to segments

2. **Segment Level** (`_handle_travel_segment_task`):
   - Downloads pre-warped video (or reads locally)
   - Uses as VACE guide video with appropriate `video_prompt_type`
   - WGP model interprets flow visualizations for motion conditioning

3. **Key Function**: `create_structure_motion_video()` in `source/structure_video_guidance.py`
   - Loads structure video frames
   - Calls `extract_optical_flow_from_frames()` (uses RAFT)
   - Generates flow visualizations
   - Encodes as H.264 video

---

## Proposed Changes

### 1. New Parameter: `structure_type`

**Location**: Orchestrator payload (client → orchestrator → segments)

**Values**:
- `"flow"` (DEFAULT) - Optical flow (current implementation, best for motion)
- `"canny"` - Canny edge detection (structural features)
- `"depth"` - Depth map estimation (3D spatial structure)

**Example**:
```json
{
  "structure_video_path": "https://example.com/structure.mp4",
  "structure_type": "flow",  // Optional - defaults to "flow" if not provided
  "structure_video_treatment": "adjust",
  "structure_video_motion_strength": 1.0,  // Only affects flow
  "structure_canny_intensity": 1.0,  // Optional - only affects canny (default: 1.0)
  "structure_depth_contrast": 1.0    // Optional - only affects depth (default: 1.0)
}
```

### 2. Preprocessor Integration Approach

**CRITICAL**: We run the preprocessors ourselves at orchestrator level, NOT relying on WGP's internal preprocessing.

**Current Flow (Optical Flow)**:
1. Orchestrator loads structure video frames
2. Orchestrator runs `FlowAnnotator` to extract optical flows
3. Orchestrator generates flow visualization frames (RGB)
4. Orchestrator encodes visualizations as H.264 video
5. Segments download the pre-processed video
6. Segments pass it to WGP as `video_guide` (WGP uses it as-is)

**Proposed Multi-Type Approach**:
1. Orchestrator loads structure video frames
2. Orchestrator runs chosen preprocessor (flow/canny/depth)
3. Orchestrator generates visualization frames (RGB)
4. Orchestrator encodes visualizations as H.264 video
5. Orchestrator uploads to shared storage
6. Segments download the pre-processed video (if remote)
7. Segments extract their slice using `structure_guidance_frame_offset`
8. Segments pass their slice to WGP as `video_guide` (WGP uses it as-is)

**Preprocessor Mapping**:

| `structure_type` | Preprocessor Class | Output | Description |
|------------------|-------------------|--------|-------------|
| `"flow"` | `FlowAnnotator` | RGB flow maps | Colorful directional motion |
| `"canny"` | `CannyVideoAnnotator` | RGB edge lines | Black/white edge detection |
| `"depth"` | `DepthV2VideoAnnotator` | RGB depth maps | Grayscale depth visualization |

**Note**: All preprocessors are available in `Wan2GP/preprocessing/` directory. We import and run them ourselves, then pass the resulting video to WGP as a standard guide video.

### 3. Strength/Intensity Parameters

Each preprocessor type supports an adjustable strength parameter:

| Parameter | Affects | Default | Range | Description |
|-----------|---------|---------|-------|-------------|
| `structure_video_motion_strength` | Flow only | `1.0` | `0.0-2.0+` | Scales flow vector magnitude (0.0=no motion, 1.0=full, 2.0=amplified) |
| `structure_canny_intensity` | Canny only | `1.0` | `0.0-2.0+` | Scales edge visualization intensity (higher=bolder edges) |
| `structure_depth_contrast` | Depth only | `1.0` | `0.0-2.0+` | Adjusts depth map contrast (higher=more pronounced depth) |

**Implementation Notes**:
- **Flow**: Already implemented - scales flow vectors before warping
- **Canny**: Post-process visualization - multiply edge pixel values by intensity factor
- **Depth**: Post-process visualization - apply contrast adjustment to depth maps

**Example Usage**:
```python
# Subtle motion guidance
{"structure_type": "flow", "structure_video_motion_strength": 0.5}

# Bold edge guidance
{"structure_type": "canny", "structure_canny_intensity": 1.5}

# High-contrast depth
{"structure_type": "depth", "structure_depth_contrast": 1.3}
```

---

## Implementation Plan

### Phase 1: Refactor Existing Optical Flow Code

**Goal**: Generalize `create_structure_motion_video()` to support any preprocessor type.

#### Current Function Signature:
```python
def create_structure_motion_video(
    structure_video_path: str,
    max_frames_needed: int,
    target_resolution: Tuple[int, int],
    target_fps: int,
    motion_strength: float,  # Only relevant for flow
    output_path: Path,
    treatment: str,
    dprint: Callable
) -> Path
```

#### Proposed New Signature:
```python
def create_structure_guidance_video(
    structure_video_path: str,
    max_frames_needed: int,
    target_resolution: Tuple[int, int],
    target_fps: int,
    structure_type: str = "flow",     # NEW: preprocessor type (default: flow)
    motion_strength: float = 1.0,     # Only used for flow
    canny_intensity: float = 1.0,     # NEW: Only used for canny
    depth_contrast: float = 1.0,      # NEW: Only used for depth
    output_path: Path,
    treatment: str = "adjust",
    dprint: Callable = print
) -> Path
```

#### Internal Changes:

1. **Frame Loading** (same for all types):
   - Load structure video frames
   - Resize to target resolution
   - Handle treatment mode (adjust/clip)

2. **Preprocessing** (type-dependent):
   - Call appropriate WGP preprocessor based on `structure_type`
   - Each preprocessor returns RGB visualization frames

3. **Video Encoding** (same for all types):
   - Encode visualizations as H.264 video
   - Return path to encoded video

---

### Phase 2: Preprocessor Integration

**Goal**: Reuse existing WGP preprocessors with minimal modifications.

#### Preprocessor Imports and Configuration

```python
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
    
    Returns a function that takes a list of frames (np.ndarray)
    and returns a list of processed frames (RGB visualizations).
    """
    if structure_type == "flow":
        from Wan2GP.preprocessing.flow import FlowVisAnnotator
        cfg = {"PRETRAINED_MODEL": "ckpts/flow/raft-things.pth"}
        annotator = FlowVisAnnotator(cfg)
        # FlowVisAnnotator.forward() returns (flow_fields, flow_visualizations)
        # We only need the visualizations (RGB images)
        # Note: motion_strength is applied during frame warping, not here
        return lambda frames: annotator.forward(frames)[1]  # [1] = visualizations
    
    elif structure_type == "canny":
        from Wan2GP.preprocessing.canny import CannyVideoAnnotator
        cfg = {"PRETRAINED_MODEL": "ckpts/scribble/netG_A_latest.pth"}
        annotator = CannyVideoAnnotator(cfg)
        
        def process_canny(frames):
            # Get base canny edges
            edge_frames = annotator.forward(frames)
            
            # Apply intensity adjustment if not 1.0
            if abs(canny_intensity - 1.0) > 1e-6:
                import numpy as np
                adjusted_frames = []
                for frame in edge_frames:
                    # Scale pixel values by intensity factor
                    adjusted = (frame.astype(np.float32) * canny_intensity).clip(0, 255).astype(np.uint8)
                    adjusted_frames.append(adjusted)
                return adjusted_frames
            return edge_frames
        
        return process_canny
    
    elif structure_type == "depth":
        from Wan2GP.preprocessing.depth_anything_v2.depth import DepthV2VideoAnnotator
        variant = "vitl"  # Could be configurable
        cfg = {
            "PRETRAINED_MODEL": f"ckpts/depth/depth_anything_v2_{variant}.pth",
            "MODEL_VARIANT": variant
        }
        annotator = DepthV2VideoAnnotator(cfg)
        
        def process_depth(frames):
            # Get base depth maps
            depth_frames = annotator.forward(frames)
            
            # Apply contrast adjustment if not 1.0
            if abs(depth_contrast - 1.0) > 1e-6:
                import numpy as np
                adjusted_frames = []
                for frame in depth_frames:
                    # Convert to float, normalize, apply contrast, denormalize
                    frame_float = frame.astype(np.float32) / 255.0
                    # Apply contrast around midpoint (0.5)
                    adjusted = ((frame_float - 0.5) * depth_contrast + 0.5).clip(0, 1)
                    adjusted = (adjusted * 255).astype(np.uint8)
                    adjusted_frames.append(adjusted)
                return adjusted_frames
            return depth_frames
        
        return process_depth
    
    else:
        raise ValueError(f"Unsupported structure_type: {structure_type}")
```

#### Key Considerations:

1. **Preprocessor Cleanup**: Each preprocessor may load models into GPU memory
   - Must clean up after use (like optical flow currently does)
   - Call `del annotator` + `gc.collect()` + `torch.cuda.empty_cache()`

2. **Frame Count**: 
   - Optical flow returns N-1 frames for N input frames
   - Canny/Depth return N frames for N input frames
   - Need to handle this discrepancy in frame count adjustment

3. **Memory Management**: 
   - Depth Anything V2 loads large models
   - May need batch processing for large frame counts

---

### Phase 3: Orchestrator Changes

**File**: `source/sm_functions/travel_between_images.py`

#### Changes to `_handle_travel_orchestrator_task()`:

**Current Code** (lines 658-770):
```python
structure_video_path = orchestrator_payload.get("structure_video_path")
structure_video_treatment = orchestrator_payload.get("structure_video_treatment", "adjust")
structure_video_motion_strength = orchestrator_payload.get("structure_video_motion_strength", 1.0)

if structure_video_path:
    # ... download/validation ...
    
    # Create pre-warped motion video
    structure_motion_video_path = create_structure_motion_video(
        structure_video_path=structure_video_path,
        max_frames_needed=total_flow_frames,
        target_resolution=target_resolution,
        target_fps=target_fps,
        motion_strength=structure_video_motion_strength,
        output_path=current_run_output_dir / structure_motion_filename,
        treatment=structure_video_treatment,
        dprint=dprint
    )
```

**Proposed Changes**:
```python
structure_video_path = orchestrator_payload.get("structure_video_path")
structure_type = orchestrator_payload.get("structure_type", "flow")  # NEW: default to flow (current behavior)
structure_video_treatment = orchestrator_payload.get("structure_video_treatment", "adjust")

# Extract strength parameters for each type
structure_video_motion_strength = orchestrator_payload.get("structure_video_motion_strength", 1.0)
structure_canny_intensity = orchestrator_payload.get("structure_canny_intensity", 1.0)
structure_depth_contrast = orchestrator_payload.get("structure_depth_contrast", 1.0)

# Validate structure_type
if structure_type not in ["flow", "canny", "depth"]:
    raise ValueError(f"Invalid structure_type: {structure_type}. Must be 'flow', 'canny', or 'depth'")

if structure_video_path:
    dprint(f"[STRUCTURE_VIDEO] Using preprocessor type: {structure_type}")
    
    # Log which strength parameter is being used
    if structure_type == "flow" and structure_video_motion_strength != 1.0:
        dprint(f"[STRUCTURE_VIDEO] Motion strength: {structure_video_motion_strength}")
    elif structure_type == "canny" and structure_canny_intensity != 1.0:
        dprint(f"[STRUCTURE_VIDEO] Canny intensity: {structure_canny_intensity}")
    elif structure_type == "depth" and structure_depth_contrast != 1.0:
        dprint(f"[STRUCTURE_VIDEO] Depth contrast: {structure_depth_contrast}")
    
    # Create pre-processed guidance video
    structure_guidance_video_path = create_structure_guidance_video(
        structure_video_path=structure_video_path,
        max_frames_needed=total_flow_frames,
        target_resolution=target_resolution,
        target_fps=target_fps,
        structure_type=structure_type,  # NEW
        motion_strength=structure_video_motion_strength,
        canny_intensity=structure_canny_intensity,      # NEW
        depth_contrast=structure_depth_contrast,        # NEW
        output_path=current_run_output_dir / structure_guidance_filename,
        treatment=structure_video_treatment,
        dprint=dprint
    )
```

#### Payload Updates:

**Segment Payload** (lines 894-900):
```python
segment_payload = {
    # ... existing params ...
    
    # Structure video guidance parameters (UPDATED)
    "structure_video_path": structure_video_path,
    "structure_type": structure_type,  # NEW
    "structure_video_treatment": structure_video_treatment,
    "structure_video_motion_strength": structure_video_motion_strength,  # Only relevant for flow
    "structure_guidance_video_url": orchestrator_payload.get("structure_guidance_video_url"),  # Renamed from structure_motion_video_url
    "structure_guidance_frame_offset": segment_flow_offsets[idx],  # Renamed from structure_motion_frame_offset
}
```

---

### Phase 4: Segment Changes

**File**: `source/travel_segment_processor.py` (used by segment handler)

#### Changes to Segment Processing:

**Current Behavior**:
- When structure video is used, the pre-processed video is passed as `video_guide` to WGP
- WGP uses the guide video as-is (no further preprocessing)
- The guide video already contains RGB visualizations (flow maps, edge lines, or depth maps)

**Proposed Change**:

The segment processor simply needs to:
1. Download the pre-processed guidance video (if remote URL)
2. Pass it to WGP as `video_guide`
3. Optionally store `structure_type` in metadata for debugging

**No video_prompt_type changes needed** because:
- We're passing a pre-processed RGB video to WGP
- WGP doesn't need to know what type of preprocessing was done
- The video is used directly as visual guidance

**Integration in `TravelSegmentProcessor.process_segment()`**:
```python
# If structure guidance video is available
if segment_params.get("structure_guidance_video_url"):
    structure_type = segment_params.get("structure_type", "flow")
    
    # Download structure guidance video (if remote URL)
    local_structure_video = download_structure_guidance_video(...)
    
    dprint(f"[STRUCTURE_GUIDANCE] Using {structure_type} preprocessing")
    
    # Use as guide video (WGP treats it as standard video guidance)
    video_guide = local_structure_video
    # Note: video_prompt_type is set by existing logic for VACE models
```

---

## Behavioral Differences by Type

### Optical Flow (DEFAULT)
**Best For**: 
- Motion-heavy scenes (running, flying, dancing)
- Camera movements (pans, zooms, rotations)
- Dynamic action sequences
- Fluid transitions
- Most general-purpose structure guidance

**Characteristics**:
- Colorful directional visualizations (rainbow flow maps)
- Encodes motion magnitude and direction
- Medium preprocessing speed (~50-100ms per frame, GPU required)
- RAFT model (~45MB)
- Returns N-1 frames for N input frames (special handling needed)
- **Current implementation** - battle-tested and reliable

**Example Use Case**: 
> "Travel with the energy and motion of a dance performance"

---

### Canny Edges
**Best For**: 
- Structural transitions (buildings, architecture)
- Sharp feature preservation
- Line-based compositions
- Abstract/geometric patterns
- When you want crisp, clean structural guidance

**Characteristics**:
- High contrast black/white edges
- No color information
- Fast preprocessing (~10-20ms per frame, CPU-friendly)
- Small model footprint (~1MB)
- Returns N frames for N input frames

**Example Use Case**: 
> "Travel through a cityscape preserving architectural lines"

---

### Depth Maps
**Best For**: 
- 3D spatial structure
- Foreground/background separation
- Parallax effects
- Scene depth preservation

**Characteristics**:
- Grayscale depth visualizations (near=bright, far=dark)
- Encodes 3D spatial information
- Medium preprocessing (~30-50ms per frame, GPU required)
- Depth Anything V2 model (~300MB for vitl, ~100MB for vitb)
- Returns N frames for N input frames

**Example Use Case**: 
> "Travel through a forest preserving depth relationships between trees"

---

## Frame Count Handling

### The N-1 Problem (Optical Flow)

Optical flow is unique: it produces N-1 flows for N frames (each flow represents the transition FROM frame[i] TO frame[i+1]).

**Current Flow Visualization Approach**:
- FlowVisAnnotator returns N-1 RGB visualizations
- We need N frames for video encoding
- Solution: Duplicate the last flow visualization

**Proposed Generalized Approach**:

```python
def process_structure_frames(
    frames: List[np.ndarray],
    structure_type: str,
    dprint: Callable
) -> List[np.ndarray]:
    """
    Process frames with chosen preprocessor, ensuring consistent output count.
    
    Returns: List of RGB visualization frames (length = len(frames))
    """
    preprocessor = get_structure_preprocessor(structure_type, dprint)
    
    processed_frames = preprocessor(frames)
    
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
```

---

## Naming Conventions

### Variable/Parameter Naming Changes

To reflect the generalized approach:

| Old Name (Flow-Specific) | New Name (Generic) | Notes |
|--------------------------|-------------------|-------|
| `structure_motion_video_url` | `structure_guidance_video_url` | More generic |
| `structure_motion_frame_offset` | `structure_guidance_frame_offset` | More generic |
| `structure_video_motion_strength` | `structure_video_motion_strength` | Keep (only affects flow) |
| `create_structure_motion_video()` | `create_structure_guidance_video()` | Function rename |

### File Naming

Generated guidance video files:
- Current: `structure_motion_{timestamp}_{uuid}.mp4`
- Proposed: `structure_{type}_{timestamp}_{uuid}.mp4`
  - Example: `structure_flow_143052_a3f8e2.mp4` (default)
  - Example: `structure_canny_143052_b4f9d3.mp4`
  - Example: `structure_depth_143052_c5e0a4.mp4`

---

## Backward Compatibility

### Handling Legacy Requests

**Scenario**: Old clients don't send `structure_type` parameter.

**Solution**: Default to flow (maintains current behavior).

```python
structure_type = orchestrator_payload.get("structure_type", "flow")
```

**Migration Path**:
1. Old clients (no `structure_type`) → Use flow (current behavior maintained)
2. Clients can explicitly request `"structure_type": "canny"` or `"structure_type": "depth"` for new preprocessors
3. No breaking changes - existing workflows continue to work identically

### Database Schema

No database schema changes needed:
- `structure_type` is stored in task `params` JSON column
- Existing `structure_video_path` field remains
- Guidance video URL stored same way (just different content)

---

## Parameter Validation

### Orchestrator Validation

```python
# Validate structure_type
valid_structure_types = ["flow", "canny", "depth"]
structure_type = orchestrator_payload.get("structure_type", "flow")

if structure_type not in valid_structure_types:
    raise ValueError(
        f"Invalid structure_type: '{structure_type}'. "
        f"Must be one of: {', '.join(valid_structure_types)}"
    )

# Warn if wrong strength parameters are set
motion_strength = orchestrator_payload.get("structure_video_motion_strength", 1.0)
canny_intensity = orchestrator_payload.get("structure_canny_intensity", 1.0)
depth_contrast = orchestrator_payload.get("structure_depth_contrast", 1.0)

if structure_type == "flow":
    if canny_intensity != 1.0:
        dprint(f"[WARNING] structure_canny_intensity is set but structure_type is 'flow'. Parameter will be ignored.")
    if depth_contrast != 1.0:
        dprint(f"[WARNING] structure_depth_contrast is set but structure_type is 'flow'. Parameter will be ignored.")
elif structure_type == "canny":
    if motion_strength != 1.0:
        dprint(f"[WARNING] structure_video_motion_strength is set but structure_type is 'canny'. Parameter will be ignored.")
    if depth_contrast != 1.0:
        dprint(f"[WARNING] structure_depth_contrast is set but structure_type is 'canny'. Parameter will be ignored.")
elif structure_type == "depth":
    if motion_strength != 1.0:
        dprint(f"[WARNING] structure_video_motion_strength is set but structure_type is 'depth'. Parameter will be ignored.")
    if canny_intensity != 1.0:
        dprint(f"[WARNING] structure_canny_intensity is set but structure_type is 'depth'. Parameter will be ignored.")
```

### Client-Side Validation

Update API documentation and client libraries:
- Default: `structure_type = "flow"` (maintains current behavior)
- Valid values: `["flow", "canny", "depth"]`
- Strength parameters are type-specific:
  - `structure_video_motion_strength` only affects `structure_type="flow"`
  - `structure_canny_intensity` only affects `structure_type="canny"`
  - `structure_depth_contrast` only affects `structure_type="depth"`

---

## Implementation Checklist

### Code Changes

- [ ] **Phase 1**: Refactor `source/structure_video_guidance.py`
  - [ ] Rename `create_structure_motion_video()` → `create_structure_guidance_video()`
  - [ ] Add `structure_type` parameter (default `"flow"`)
  - [ ] Add strength parameters: `canny_intensity`, `depth_contrast` (default `1.0`)
  - [ ] Implement `get_structure_preprocessor()` function with strength support
  - [ ] Implement `process_structure_frames()` with N-1 handling
  - [ ] Update function to use chosen preprocessor
  - [ ] Add GPU cleanup for all preprocessor types

- [ ] **Phase 2**: Update orchestrator
  - [ ] Add `structure_type` parameter extraction (default `"flow"`)
  - [ ] Add strength parameter extraction (`motion_strength`, `canny_intensity`, `depth_contrast`)
  - [ ] Add validation for `structure_type` values
  - [ ] Add warnings for unused strength parameters
  - [ ] Update function call to use new name/signature and pass strength params
  - [ ] Update segment payload with new parameter names
  - [ ] Update filename generation to include type

- [ ] **Phase 3**: Update segment processor
  - [ ] Update `TravelSegmentProcessor` to handle `structure_type` parameter
  - [ ] Update parameter names (motion_video → guidance_video)
  - [ ] Add logging for which preprocessor type is being used

- [ ] **Phase 4**: Testing
  - [ ] Test flow preprocessing (default, maintains current behavior)
  - [ ] Test canny preprocessing (new option)
  - [ ] Test depth preprocessing (new option)
  - [ ] Test with 1 segment journey
  - [ ] Test with multi-segment journey (3+ segments)
  - [ ] Test with different `treatment` modes (adjust/clip)
  - [ ] Test backward compatibility (no `structure_type` param → should default to flow)

### Documentation Updates

- [ ] Update `STRUCTURE_VIDEO_GUIDANCE_USAGE.md`
  - [ ] Add `structure_type` parameter documentation
  - [ ] Add examples for each type (flow, canny, depth)
  - [ ] Add guidance on when to use each type
  - [ ] Update parameter table

- [ ] Update client API documentation
  - [ ] Document default (flow - maintains current behavior)
  - [ ] Document all valid `structure_type` values
  - [ ] Add visual examples of each preprocessing type
  - [ ] Note: No migration needed - existing code continues to work

- [ ] Update `STRUCTURE.md`
  - [ ] Document multi-type support
  - [ ] Update architecture diagrams

---

## Performance Considerations

### Preprocessing Speed Comparison

Based on typical 720p video (1280×720 resolution):

| Type | Speed (per frame) | GPU Required | Model Size | Memory Usage |
|------|-------------------|--------------|------------|--------------|
| Flow | ~50-100ms | Yes | ~45MB | Medium (~500MB) |
| Canny | ~10-20ms | No (CPU) | ~1MB | Low (~100MB) |
| Depth | ~30-50ms | Yes | ~300MB (vitl) | High (~1.5GB) |

### Orchestrator Impact

For a typical journey with 3 segments × 73 frames = 219 frames:

| Type | Orchestrator Time | Notes |
|------|-------------------|-------|
| Flow | ~10-20 seconds | Current baseline (default) |
| Canny | ~2-4 seconds | Fastest option |
| Depth | ~6-10 seconds | Medium speed |

**Recommendation**: Flow is the default for most use cases (motion guidance). Use Canny for faster preprocessing when structural edges are sufficient. Use Depth for 3D spatial awareness.

---

## Example Usage

### Client Request Format

**Example 1: Optical Flow (Default)**
```json
{
  "input_image_paths": ["img1.jpg", "img2.jpg", "img3.jpg"],
  "prompts": ["Scene 1", "Scene 2", "Scene 3"],
  "structure_video_path": "https://cdn.example.com/reference_motion.mp4",
  "structure_type": "flow",
  "structure_video_treatment": "adjust",
  "structure_video_motion_strength": 0.8,
  "model_name": "hunyuan_i2v",
  "resolution": "768x576",
  "fps_helpers": 16
}
```

**Example 2: Canny Edges with High Intensity**
```json
{
  "input_image_paths": ["img1.jpg", "img2.jpg", "img3.jpg"],
  "prompts": ["Scene 1", "Scene 2", "Scene 3"],
  "structure_video_path": "https://cdn.example.com/architecture.mp4",
  "structure_type": "canny",
  "structure_video_treatment": "adjust",
  "structure_canny_intensity": 1.5,
  "model_name": "hunyuan_i2v",
  "resolution": "768x576",
  "fps_helpers": 16
}
```

**Example 3: Depth with Enhanced Contrast**
```json
{
  "input_image_paths": ["img1.jpg", "img2.jpg", "img3.jpg"],
  "prompts": ["Scene 1", "Scene 2", "Scene 3"],
  "structure_video_path": "https://cdn.example.com/forest.mp4",
  "structure_type": "depth",
  "structure_video_treatment": "adjust",
  "structure_depth_contrast": 1.3,
  "model_name": "hunyuan_i2v",
  "resolution": "768x576",
  "fps_helpers": 16
}
```

### Orchestrator Logging

```
[STRUCTURE_VIDEO] Using preprocessor type: flow
[STRUCTURE_VIDEO] Total flow frames across all segments: 167
[STRUCTURE_VIDEO] Creating pre-processed guidance video...
[STRUCTURE_VIDEO] Loaded 167 frames from structure video
[STRUCTURE_PREPROCESS] Processing with optical flow extraction...
[STRUCTURE_PREPROCESS] Processed 167 frames in 11.2 seconds
[STRUCTURE_VIDEO] Encoded guidance video: structure_flow_143052_a3f8e2.mp4
[STRUCTURE_VIDEO] Pre-processed guidance video path: https://storage.supabase.co/.../structure_flow_143052_a3f8e2.mp4
```

### Segment Processing

```
[STRUCTURE_GUIDANCE] Segment 0: Using flow preprocessing
[STRUCTURE_GUIDANCE] Downloaded guidance video from: https://storage.supabase.co/...
[STRUCTURE_GUIDANCE] Reading frames 0-72 from guidance video (offset: 0)
[WGP] Generating segment with optical flow guidance...
```

---

## Future Enhancements

### Additional Preprocessor Types

Consider adding in future versions:
- **Scribble** (`structure_type="scribble"`) - Simplified edge sketches
- **Pose** (`structure_type="pose"`) - Human pose skeletal tracking
- **Gray** (`structure_type="gray"`) - Grayscale luminance

### Advanced Options

- **Preprocessor-specific tuning**:
  ```json
  {
    "structure_type": "canny",
    "structure_canny_low_threshold": 100,
    "structure_canny_high_threshold": 200
  }
  ```

- **Hybrid preprocessing**:
  ```json
  {
    "structure_type": "canny+depth",  // Blend multiple types
    "structure_blend_ratio": 0.7      // 70% canny, 30% depth
  }
  ```

- **Per-segment type override**:
  ```json
  {
    "structure_types": ["canny", "flow", "canny"],  // Different type per segment
  }
  ```

---

## Summary

This design extends structure video guidance to support three preprocessing types:

1. **Optical flow** (DEFAULT) - Motion-focused, battle-tested, best for most cases
2. **Canny edges** (NEW) - Fast, CPU-friendly, great for structural features
3. **Depth maps** (NEW) - 3D spatial structure, GPU-required

### Key Benefits:
- ✅ Maintains pre-warped video architecture (efficient, distributed-friendly)
- ✅ We run preprocessors ourselves at orchestrator level (same pattern as current flow implementation)
- ✅ Segments extract their slice using frame offsets (not passed whole video)
- ✅ Backward compatible (defaults to flow when `structure_type` not passed)
- ✅ Strength parameters for all types (not just flow)
- ✅ Minimal code changes (generalize existing functions)
- ✅ Provides flexibility (canny for speed, depth for 3D awareness)

### Implementation Risk: LOW
- Core architecture unchanged
- All preprocessors already exist and tested in WGP
- Main work is parameter plumbing and validation

### Estimated Implementation Time: 4-6 hours
- Phase 1 (refactor): 2 hours
- Phase 2 (orchestrator): 1 hour  
- Phase 3 (segments): 1 hour
- Phase 4 (testing): 1-2 hours

