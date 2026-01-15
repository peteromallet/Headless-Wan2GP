# Multi-Structure Video Support

## Status: ✅ IMPLEMENTED (Orchestrator + Standalone Segments)

## Overview

This feature supports **multiple non-overlapping structure videos** in a single travel generation. Users can specify different structure videos for different **frame ranges**, with gaps (no guidance) allowed between them.

**Key design decision:** The orchestrator pre-composites all structure videos into **ONE guidance video** with neutral frames for gaps. Segments work exactly as they do today — **no segment handler changes required**.

**Constraint:** All structure video configs in a generation must use the **same type** (all flow, all canny, all depth, or all raw).

### ⚠️ Coordinate System: Stitched Timeline

**CRITICAL**: `start_frame` and `end_frame` must be specified in the **stitched output timeline**:
- Frame numbers align with where **images/keyframes appear** in the final video
- For a 5-segment × 81-frame generation with 10-frame overlaps:
  - **Timeline length = 365 frames** (5×81 - 4×10), not 405
  - **Image keyframes at**: `[0, 71, 142, 213, 284, 355]`

This differs from the legacy single structure video which used an internal "guidance timeline" (405 frames for the same config).

---

## Architecture

### Current: Single Structure Video

```
Source video  →  [process]  →  ONE guidance video (entire timeline)
                                        ↓
                        Segments extract via frame_offset
```

### New: Pre-Composite Multiple Sources

```
Timeline:        0──────50   50───70   71───────81
                 │ src A │   (gap)    │  src B   │
                     ↓         ↓           ↓
Processing:      [process]  [neutral]  [process]
                     ↓         ↓           ↓
Composite:       [ flow A ] [neutral] [ flow B ]
                 └──────────────────────────────┘
                      ONE guidance video
                              ↓
                 Segments use offsets as usual
                 (ZERO changes to segment handlers!)
```

**Why this is simple:**
- Segments download ONE video and extract frames by offset — same as today
- All compositing complexity is in the orchestrator
- No segment handler changes needed

---

## Neutral Frames (No-Guidance Regions)

When there's a gap between structure videos, we fill with neutral frames that don't bias generation:

| Type | Neutral Frame | Why |
|------|---------------|-----|
| flow | Uniform gray (RGB 128,128,128) | Gray = HSV center = no motion signal |
| canny | Black (RGB 0,0,0) | No edges = no structure signal |
| depth | Mid-gray (RGB 128,128,128) | Neutral depth |
| raw | Black (RGB 0,0,0) | No structure signal for Uni3C |

### ⚠️ Uni3C (raw type): Automatic Latent-Space Zeroing

For Uni3C (`structure_type="raw"`), black pixel frames require special handling:

**The Problem:**
- VAE-encoded black pixels ≠ zero in latent space
- Encoded black frames would tell Uni3C "control toward black output" rather than "no control"

**The Solution (Implemented):**
The `_encode_uni3c_guide()` function in `any2video.py` automatically:
1. **Detects** black/near-black frames before VAE encoding
2. **Encodes** the full video (VAE needs temporal context)
3. **Zeros** latent frames that correspond to empty pixel frames

This means you can create a guide video with black frames for "no control" regions, and they will be properly converted to zero latents during encoding.

**Configuration:**
```python
# Enable/disable via API parameter (default: True)
"uni3c_zero_empty_frames": True  # or False to use VAE-encoded black frames
```

**Example logs when this activates:**
```
[UNI3C] any2video:   zero_empty_frames: True
[UNI3C] any2video: Detected 51/81 empty pixel frames
[UNI3C] any2video:   Empty frame ranges: 30-80
[UNI3C] any2video: VAE encoded render_latent shape: (1, 16, 21, 60, 104)
[UNI3C] any2video: Zeroed 13/21 latent frames (no control signal)
```

---

## New Payload Format

```python
orchestrator_payload = {
    # ... existing fields ...
    
    # Global processing parameters (can be overridden per-config)
    "structure_video_motion_strength": 1.0,  # For flow
    "structure_canny_intensity": 1.0,        # For canny
    "structure_depth_contrast": 1.0,         # For depth
    
    # NEW: Array of structure video configurations
    "structure_videos": [
        {
            # Source video (required)
            "path": "https://storage.../action_sequence.mp4",
            
            # Frame range in OUTPUT timeline (required)
            "start_frame": 0,
            "end_frame": 50,  # Exclusive: covers frames 0-49
            
            # Structure type (required - all configs must use same type)
            "structure_type": "flow",  # "flow" | "canny" | "depth" | "raw"
            
            # Optional: Per-config strength (defaults to global)
            "motion_strength": 1.0,
            
            # Optional: Extract specific portion of SOURCE video
            # If omitted, entire source is stretched/clipped to fit the frame range
            "source_start_frame": 0,    # Default: 0
            "source_end_frame": 10,     # Default: end of source
            
            # Optional: Treatment mode
            "treatment": "adjust"  # "adjust" | "clip" (default: "adjust")
        },
        {
            "path": "https://storage.../slow_zoom.mp4",
            "start_frame": 71,
            "end_frame": 82,  # Covers frames 71-81
            "structure_type": "flow",  # Must match first config!
            "motion_strength": 0.8,
            "treatment": "clip",
            "source_start_frame": 0,
            "source_end_frame": 15
        }
        # Frames 50-70: filled with neutral frames (no guidance)
    ],
    
    # LEGACY (still supported for backwards compatibility):
    "structure_video_path": None  # Single string = applies to entire timeline
}
```

### Backwards Compatibility

If `structure_video_path` (string) is provided without `structure_videos` (array), auto-convert:

```python
if not structure_videos and orchestrator_payload.get("structure_video_path"):
    total_frames = sum(expanded_segment_frames)
    structure_videos = [{
        "path": orchestrator_payload["structure_video_path"],
        "start_frame": 0,
        "end_frame": total_frames,
        "treatment": orchestrator_payload.get("structure_video_treatment", "adjust"),
    }]
```

---

## Standalone Segment Support

Standalone segments (`individual_travel_segment` tasks) can also use multi-structure video. **No pre-computed composite needed** — the segment computes its own portion.

### How It Works

1. Segment receives full `structure_videos` array in `orchestrator_details`
2. Segment calculates its position in stitched timeline using `segment_index` + `segment_frames_expanded` + `frame_overlap_expanded`
3. Segment filters configs to those overlapping its frame range
4. Segment creates a mini-composite for just its portion

### Payload for Standalone Segment

```python
# individual_travel_segment task payload
{
    "orchestrator_details": {
        "structure_videos": [
            {"path": "...", "start_frame": 60, "end_frame": 184, "structure_type": "flow", ...},
            {"path": "...", "start_frame": 237, "end_frame": 334, "structure_type": "flow", ...}
        ],
        "segment_frames_expanded": [81, 81, 81, 81, 81],
        "frame_overlap_expanded": [10, 10, 10, 10],
        "parsed_resolution_wh": "768x576",
        "fps_helpers": 16,
        # ... other orchestrator details
    },
    "segment_index": 2,  # Which segment to generate (0-based)
    # ... other segment params
}
```

### Key Functions

| Function | Location | Purpose |
|----------|----------|---------|
| `calculate_segment_stitched_position()` | `structure_video_guidance.py` | Calculate segment's start frame and count in stitched timeline |
| `extract_segment_structure_guidance()` | `structure_video_guidance.py` | Create guidance for single segment from full config |

### Example: Segment 2 (stitched frames 142-222)

```
Full timeline:    0────────────────────────────────────────364
Config A:              60────────────184
Config B:                                    237──────334

Segment 2 range:            142─────────222

Overlap with A:             142────184  (42 frames of guidance)
No overlap with B

Segment 2's mini-composite:
  Local 0-41:   Config A guidance (for stitched 142-183)
  Local 42-80:  Neutral (for stitched 184-222)
```

---

## Implementation Tasks

### Phase 1: New Compositing Function

**File:** `source/structure_video_guidance.py`

#### Task 1.1: Add `create_composite_guidance_video()`

```python
def create_composite_guidance_video(
    structure_configs: list[dict],
    total_frames: int,
    structure_type: str,
    target_resolution: Tuple[int, int],
    target_fps: int,
    output_path: Path,
    motion_strength: float = 1.0,
    canny_intensity: float = 1.0,
    depth_contrast: float = 1.0,
    download_dir: Path = None,
    dprint: Callable = print
) -> Path:
    """
    Create a single composite guidance video from multiple structure video configs.
    
    Args:
        structure_configs: List of configs with path, start_frame, end_frame, etc.
        total_frames: Total frames in the output timeline
        structure_type: "flow" | "canny" | "depth" | "raw" (same for all)
        target_resolution: (width, height)
        target_fps: Target FPS
        output_path: Where to save composite video
        motion_strength/canny_intensity/depth_contrast: Processing params
        download_dir: Directory for downloading source videos
        dprint: Debug print function
        
    Returns:
        Path to the composite guidance video
    """
    # 1. Initialize output frame array with neutral frames
    neutral_frame = create_neutral_frame(structure_type, target_resolution)
    composite_frames = [neutral_frame.copy() for _ in range(total_frames)]
    
    # 2. Process each config and place frames at correct positions
    for config in structure_configs:
        # Download source if URL
        source_path = download_video_if_url(config["path"], download_dir)
        
        # Calculate how many frames this config needs
        start_frame = config["start_frame"]
        end_frame = config["end_frame"]
        frames_needed = end_frame - start_frame
        
        # Load and process source video frames
        source_frames = load_structure_video_frames(
            source_path,
            target_frame_count=frames_needed,
            target_fps=target_fps,
            target_resolution=target_resolution,
            treatment=config.get("treatment", "adjust"),
            source_start_frame=config.get("source_start_frame", 0),
            source_end_frame=config.get("source_end_frame"),
            dprint=dprint
        )
        
        # Process frames (flow/canny/depth/raw)
        processed_frames = process_structure_frames(
            source_frames,
            structure_type,
            motion_strength,
            canny_intensity,
            depth_contrast,
            dprint
        )
        
        # Place processed frames into composite at correct positions
        for i, frame in enumerate(processed_frames):
            frame_idx = start_frame + i
            if frame_idx < total_frames:
                composite_frames[frame_idx] = frame
        
        dprint(f"[COMPOSITE] Placed {len(processed_frames)} frames at positions {start_frame}-{end_frame-1}")
    
    # 3. Encode composite as video
    encode_frames_to_video(composite_frames, output_path, target_fps)
    
    return output_path


def create_neutral_frame(structure_type: str, resolution: Tuple[int, int]) -> np.ndarray:
    """Create a neutral frame for the given structure type."""
    w, h = resolution
    
    if structure_type == "flow":
        # Gray = no motion in HSV flow visualization
        return np.full((h, w, 3), 128, dtype=np.uint8)
    elif structure_type == "canny":
        # Black = no edges
        return np.zeros((h, w, 3), dtype=np.uint8)
    elif structure_type == "depth":
        # Mid-gray = neutral depth
        return np.full((h, w, 3), 128, dtype=np.uint8)
    elif structure_type == "raw":
        # Black = no structure signal
        return np.zeros((h, w, 3), dtype=np.uint8)
    else:
        return np.zeros((h, w, 3), dtype=np.uint8)
```

#### Task 1.2: Update `load_structure_video_frames()` with source range params

```python
def load_structure_video_frames(
    structure_video_path: str,
    target_frame_count: int,
    target_fps: int,
    target_resolution: Tuple[int, int],
    treatment: str = "adjust",
    crop_to_fit: bool = True,
    source_start_frame: int = 0,          # NEW
    source_end_frame: int | None = None,  # NEW
    dprint: Callable = print
) -> List[np.ndarray]:
```

Changes:
- If `source_start_frame` or `source_end_frame` provided, only use that range of the source video
- Apply treatment ("adjust"/"clip") within that range

---

### Phase 2: Orchestrator Changes

**File:** `source/sm_functions/travel_between_images.py`

#### Task 2.1: Add validation

```python
def validate_structure_video_configs(configs: list, total_frames: int, dprint) -> None:
    """
    Validate structure video configurations:
    - No overlapping frame ranges
    - Valid frame indices (0 <= frame < total_frames)
    - Required fields present
    - Configs sorted by start_frame
    """
    if not configs:
        return
    
    # Sort by start_frame for easier overlap checking
    sorted_configs = sorted(configs, key=lambda c: c.get("start_frame", 0))
    
    prev_end = -1
    for i, config in enumerate(sorted_configs):
        if "path" not in config:
            raise ValueError(f"Structure video config {i} missing 'path'")
        if "start_frame" not in config or "end_frame" not in config:
            raise ValueError(f"Structure video config {i} missing 'start_frame' or 'end_frame'")
        
        start = config["start_frame"]
        end = config["end_frame"]
        
        if start < 0 or end > total_frames or start >= end:
            raise ValueError(f"Config {i}: invalid frame range [{start}, {end}) for timeline of {total_frames} frames")
        
        if start < prev_end:
            raise ValueError(f"Config {i}: frame range [{start}, {end}) overlaps with previous config ending at {prev_end}")
        
        prev_end = end
        
    dprint(f"[STRUCTURE_VIDEO] Validated {len(configs)} configs, no overlaps")
```

#### Task 2.2: Replace single-video processing with composite creation

Replace the existing structure video processing block (~lines 1041-1239) with:

```python
# Normalize structure video configs
structure_videos = orchestrator_payload.get("structure_videos", [])
total_timeline_frames = sum(expanded_segment_frames)  # Calculate total output frames

# Backwards compatibility: single video path
if not structure_videos and orchestrator_payload.get("structure_video_path"):
    structure_videos = [{
        "path": orchestrator_payload["structure_video_path"],
        "start_frame": 0,
        "end_frame": total_timeline_frames,
        "treatment": orchestrator_payload.get("structure_video_treatment", "adjust"),
    }]

if structure_videos:
    # Validate configs
    validate_structure_video_configs(structure_videos, total_timeline_frames, dprint)
    
    structure_type = orchestrator_payload.get("structure_type", "flow")
    
    # Create composite guidance video
    timestamp_short = datetime.now().strftime("%H%M%S")
    unique_suffix = uuid.uuid4().hex[:6]
    composite_filename = f"structure_composite_{structure_type}_{timestamp_short}_{unique_suffix}.mp4"
    
    composite_guidance_path = create_composite_guidance_video(
        structure_configs=structure_videos,
        total_frames=total_timeline_frames,
        structure_type=structure_type,
        target_resolution=target_resolution,
        target_fps=target_fps,
        output_path=current_run_output_dir / composite_filename,
        motion_strength=orchestrator_payload.get("structure_video_motion_strength", 1.0),
        canny_intensity=orchestrator_payload.get("structure_canny_intensity", 1.0),
        depth_contrast=orchestrator_payload.get("structure_depth_contrast", 1.0),
        download_dir=current_run_output_dir,
        dprint=dprint
    )
    
    # Upload composite
    guidance_video_url = upload_and_get_final_output_location(
        local_file_path=composite_guidance_path,
        supabase_object_name=composite_filename,
        initial_db_location=str(composite_guidance_path),
        dprint=dprint
    )
    
    # Store for segments (same as current single-video behavior)
    orchestrator_payload["structure_guidance_video_url"] = guidance_video_url
    orchestrator_payload["structure_type"] = structure_type
    
    travel_logger.success(f"Created composite guidance video with {len(structure_videos)} sources: {guidance_video_url}", task_id=orchestrator_task_id_str)
```

#### Task 2.3: Segment payload assignment (NO CHANGES NEEDED!)

The existing code that assigns `structure_guidance_video_url` and `structure_guidance_frame_offset` to segments **continues to work as-is**:

```python
# This code doesn't change!
segment_payload = {
    ...
    "structure_guidance_video_url": orchestrator_payload.get("structure_guidance_video_url"),
    "structure_guidance_frame_offset": segment_flow_offsets[idx],
    ...
}
```

Segments don't know or care that the guidance video was composited from multiple sources.

---

### Phase 3: Segment Handler Updates

**None required!** Segments:
1. Download the (composite) guidance video
2. Extract frames starting at their offset
3. Apply guidance as usual

Neutral frames in gaps are just frames with no signal — the model generates freely there.

---

## Visual Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Pre-Composite Approach                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Input configs:                                                              │
│    Config 0: action.mp4 → frames 0-49                                       │
│    Config 1: zoom.mp4   → frames 71-81                                      │
│    (gap at frames 50-70)                                                    │
│                                                                              │
│  Orchestrator creates ONE composite video:                                   │
│    ┌────────────────────────────────────────────────────────────────────┐   │
│    │ [flow from action.mp4] [neutral gray] [flow from zoom.mp4]        │   │
│    │ frames 0-49            frames 50-70   frames 71-81                │   │
│    └────────────────────────────────────────────────────────────────────┘   │
│                              ↓                                               │
│                    guidance_composite_xxx.mp4                                │
│                              ↓                                               │
│    Segment 0: offset=0,  extracts frames 0-60   (guided + some neutral)     │
│    Segment 1: offset=53, extracts frames 53-113 (neutral + guided)          │
│    Segment 2: offset=106, extracts frames 106-166 (if applicable)           │
│                                                                              │
│  Segments work EXACTLY as they do today — no changes!                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Testing Plan

### Unit Tests

1. **`create_neutral_frame()`**: Verify correct colors for each type
2. **`validate_structure_video_configs()`**: 
   - Overlapping ranges → error
   - Out-of-bounds frames → error
   - Missing fields → error
3. **`create_composite_guidance_video()`**:
   - Single config → same as current behavior
   - Multiple configs → frames placed correctly
   - Gaps filled with neutral frames

### Integration Tests

1. **Backwards compatibility**: Single `structure_video_path` still works
2. **Multi-source generation**:
   - 2 sources with gap between
   - Sources with different treatments (adjust vs clip)
3. **Edge cases**:
   - Config covering entire timeline (equivalent to single video)
   - Config at very start or very end
   - Single-frame gaps

---

## Migration Notes

- No database schema changes
- No breaking API changes (backwards compatible)
- No segment handler changes
- Existing payloads continue to work
- New `structure_videos` array format available immediately

---

## Future Enhancements (Out of Scope)

- **Per-config type**: Allow mixing flow/canny/depth (would need per-segment type tracking)
- **Smooth transitions**: Blend guidance at boundaries instead of hard cuts
- **Per-frame strength**: Vary strength over time within a config
- **Preview**: Show which frames map to which sources before generation
