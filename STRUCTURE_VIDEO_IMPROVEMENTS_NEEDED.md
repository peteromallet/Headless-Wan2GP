# Structure Video - Current Issues & Improvements Needed

## Issue 1: Video Size Handling (No Cropping)

### Current Behavior

**File:** `source/structure_video_guidance.py`  
**Lines:** 173-176

```python
# Current: Simple resize (stretches/squashes if aspect ratio differs)
frame_pil = Image.fromarray(frame_np)
frame_resized = frame_pil.resize((w, h), resample=Image.Resampling.LANCZOS)
frame_resized_np = np.array(frame_resized)
```

### Problem

**Scenario:**
- Target resolution: 1280x720 (16:9 aspect ratio)
- Structure video: 1920x1920 (1:1 aspect ratio)

**Current Result:**
```
1920x1920 → resize to 1280x720
# Image gets stretched/squashed! Motion vectors distorted!
```

### Solution Needed: Center Crop + Resize

**Option 1: Center Crop to Match Aspect Ratio**
```python
def load_structure_video_frames(
    structure_video_path: str,
    target_frame_count: int,
    target_fps: int,
    target_resolution: Tuple[int, int],
    crop_to_fit: bool = True,  # ← NEW parameter
    dprint: Callable = print
) -> List[np.ndarray]:
    # ... existing load code ...
    
    for i, frame in enumerate(frames):
        frame_np = frame.cpu().numpy() if hasattr(frame, 'cpu') else np.array(frame)
        frame_pil = Image.fromarray(frame_np)
        
        if crop_to_fit:
            # Calculate target aspect ratio
            target_w, target_h = target_resolution
            target_aspect = target_w / target_h
            
            # Get current dimensions
            current_w, current_h = frame_pil.size
            current_aspect = current_w / current_h
            
            if abs(current_aspect - target_aspect) > 0.01:  # Aspect ratios differ
                # Crop to match target aspect ratio (center crop)
                if current_aspect > target_aspect:
                    # Too wide - crop width
                    new_width = int(current_h * target_aspect)
                    left = (current_w - new_width) // 2
                    frame_pil = frame_pil.crop((left, 0, left + new_width, current_h))
                else:
                    # Too tall - crop height
                    new_height = int(current_w / target_aspect)
                    top = (current_h - new_height) // 2
                    frame_pil = frame_pil.crop((0, top, current_w, top + new_height))
        
        # Now resize to exact target resolution
        frame_resized = frame_pil.resize(target_resolution, resample=Image.Resampling.LANCZOS)
        processed_frames.append(np.array(frame_resized))
```

**Behavior:**
```
Structure video: 1920x1920 (1:1)
Target: 1280x720 (16:9)

Step 1: Crop to 16:9 aspect ratio
  → 1920x1080 (cropped 420px from top and bottom)

Step 2: Resize to target
  → 1280x720 (downscaled)

Result: No distortion, motion vectors accurate ✓
```

---

## Issue 2: Motion Extraction Happens Per Segment (Inefficient)

### Current Behavior

**When:** Motion extraction happens **EVERY SEGMENT**

**File:** `source/structure_video_guidance.py`  
**Function:** `apply_structure_motion_with_tracking()`  
**Called From:** `source/video_utils.py` line 770 and 850

**Execution Flow:**
```
Segment 0:
  ├─ Load structure video frames (lines 435-441)
  ├─ Extract optical flow with RAFT (lines 448-451)  ← GPU intensive!
  └─ Apply motion

Segment 1:
  ├─ Load structure video frames AGAIN
  ├─ Extract optical flow with RAFT AGAIN  ← Redundant!
  └─ Apply motion

Segment 2:
  ├─ Load structure video frames AGAIN
  ├─ Extract optical flow with RAFT AGAIN  ← Redundant!
  └─ Apply motion
```

### Problem

**Performance Impact:**
- RAFT extraction is GPU-intensive (~5-10 seconds per video)
- For 3 segments, you're extracting the SAME flows 3 times
- Total waste: ~15-30 seconds per orchestration
- VRAM loaded/unloaded repeatedly

**Why It's Wasteful:**
```
Segment 0 needs flows for frames 20-62 (43 flows)
Segment 1 needs flows for frames 20-62 (43 flows)
Segment 2 needs flows for frames 20-62 (43 flows)

→ We extract 43 flows THREE times from the SAME structure video
→ All three segments could use the SAME flow cache
```

### Solution 1: Orchestrator-Level Flow Extraction (Best)

**Extract once, use for all segments**

#### Changes to `travel_between_images.py`

```python
def _handle_travel_orchestrator_task(...):
    # ... existing code ...
    
    # Extract and validate structure video parameters
    structure_video_path = orchestrator_payload.get("structure_video_path")
    structure_video_treatment = orchestrator_payload.get("structure_video_treatment", "adjust")
    structure_video_motion_strength = orchestrator_payload.get("structure_video_motion_strength", 1.0)
    
    # NEW: Pre-extract flows at orchestrator level
    structure_flow_cache_path = None
    if structure_video_path:
        # Download if URL
        structure_video_path = download_video_if_url(...)
        
        # NEW: Extract flows once and cache to disk
        structure_flow_cache_path = _extract_and_cache_structure_flows(
            structure_video_path=structure_video_path,
            output_dir=current_run_output_dir,
            orchestrator_task_id=orchestrator_task_id_str,
            dprint=dprint
        )
        
        dprint(f"[STRUCTURE_VIDEO] Cached flows to: {structure_flow_cache_path}")
    
    # Pass cache path to all segments
    for idx in range(num_segments):
        segment_payload = {
            # ... existing fields ...
            "structure_video_path": structure_video_path,
            "structure_flow_cache_path": structure_flow_cache_path,  # ← NEW
            "structure_video_treatment": structure_video_treatment,
            "structure_video_motion_strength": structure_video_motion_strength,
        }
```

#### New Helper Function

```python
def _extract_and_cache_structure_flows(
    structure_video_path: str,
    output_dir: Path,
    orchestrator_task_id: str,
    dprint=print
) -> str:
    """
    Extract optical flows from structure video and cache to disk.
    
    Returns path to cached flow file (.npz).
    """
    from source.structure_video_guidance import (
        load_structure_video_frames,
        extract_optical_flow_from_frames
    )
    
    # Load entire structure video
    dprint(f"[STRUCTURE_FLOW_CACHE] Loading full structure video: {structure_video_path}")
    
    # Get video info to determine how many frames to load
    import cv2
    cap = cv2.VideoCapture(structure_video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    dprint(f"[STRUCTURE_FLOW_CACHE] Structure video: {total_frames} frames @ {fps} fps")
    
    # Load all frames (at target resolution - we'll still need to resize per segment)
    # Actually, we should load at original resolution and resize later
    # OR: Load at the max resolution we'll need
    
    # For now, let's load at a reasonable resolution and cache flows at that resolution
    # Segments can then resize flows as needed
    
    # Load frames (using first segment's resolution as reference)
    # This is a simplification - ideally we'd load at native resolution
    from Wan2GP.wgp import get_resampled_video
    
    frames = get_resampled_video(
        structure_video_path,
        start_frame=0,
        max_frames=total_frames,
        target_fps=fps,
        bridge='torch'
    )
    
    # Convert to numpy
    frames_np = [
        frame.cpu().numpy() if hasattr(frame, 'cpu') else np.array(frame)
        for frame in frames
    ]
    
    # Extract flows
    dprint(f"[STRUCTURE_FLOW_CACHE] Extracting flows from {len(frames_np)} frames...")
    flow_fields, flow_vis = extract_optical_flow_from_frames(frames_np, dprint=dprint)
    
    # Cache to disk
    cache_filename = f"structure_flows_{orchestrator_task_id}.npz"
    cache_path = output_dir / cache_filename
    
    dprint(f"[STRUCTURE_FLOW_CACHE] Saving {len(flow_fields)} flows to {cache_path}")
    
    # Save as compressed numpy archive
    np.savez_compressed(
        cache_path,
        flows=np.array(flow_fields),  # Stack into single array
        metadata=np.array({
            'source_video': structure_video_path,
            'total_flows': len(flow_fields),
            'fps': fps
        }, dtype=object)
    )
    
    return str(cache_path)
```

#### Update `apply_structure_motion_with_tracking`

```python
def apply_structure_motion_with_tracking(
    frames_for_guide_list: List[np.ndarray],
    guidance_tracker: GuidanceTracker,
    structure_video_path: str,
    structure_flow_cache_path: str | None = None,  # ← NEW parameter
    structure_video_treatment: str,
    parsed_res_wh: Tuple[int, int],
    fps_helpers: int,
    motion_strength: float = 1.0,
    dprint: Callable = print
) -> List[np.ndarray]:
    # ... existing unguidanced range detection ...
    
    try:
        # NEW: Check if we have cached flows
        if structure_flow_cache_path and Path(structure_flow_cache_path).exists():
            dprint(f"[STRUCTURE_VIDEO] Loading cached flows from {structure_flow_cache_path}")
            
            # Load cached flows
            cache_data = np.load(structure_flow_cache_path, allow_pickle=True)
            flow_fields = list(cache_data['flows'])
            
            dprint(f"[STRUCTURE_VIDEO] Loaded {len(flow_fields)} cached flows")
            
        else:
            # Fallback: Extract flows per-segment (old behavior)
            dprint(f"[STRUCTURE_VIDEO] No cache found, extracting flows...")
            
            structure_frames = load_structure_video_frames(
                structure_video_path,
                target_frame_count=total_unguidanced,
                target_fps=fps_helpers,
                target_resolution=parsed_res_wh,
                dprint=dprint
            )
            
            flow_fields, flow_vis = extract_optical_flow_from_frames(
                structure_frames,
                dprint=dprint
            )
        
        # ... rest of existing logic ...
```

### Solution 2: Simple In-Memory Cache (Quick Fix)

**Store flows in orchestrator payload for reuse**

```python
# In orchestrator:
if structure_video_path and "_structure_flows_cache" not in orchestrator_payload:
    # Extract once
    flows = extract_flows(...)
    orchestrator_payload["_structure_flows_cache"] = flows  # Store in payload

# Each segment gets the cached flows via full_orchestrator_payload
```

**Pros:** Simple, no disk I/O  
**Cons:** Flows stored in DB payload (could be large)

---

## Issue 3: Where Flows Are NOT Saved

### Current State

**Flows are extracted fresh for each segment and then discarded.**

**No caching occurs between:**
- Different segments of the same orchestration
- Different orchestrations using the same structure video

### Storage Locations to Consider

#### Option A: Per-Orchestration Cache (Recommended)
```
outputs/
  └── run_20251001_abc123/
      ├── structure_flows_cache.npz  ← Cached flows for this run
      ├── segment_0_guide.mp4
      ├── segment_1_guide.mp4
      └── ...
```

**Pros:**
- Shared across all segments in one orchestration
- Cleaned up when run is cleaned up
- No global cache management needed

**Cons:**
- Not shared across different orchestrations

#### Option B: Global Structure Video Cache
```
outputs/
  └── .structure_flow_cache/
      ├── {video_hash}_720p_16fps.npz  ← Cached by video + resolution + fps
      ├── {video_hash}_1080p_24fps.npz
      └── ...
```

**Pros:**
- Shared across ALL orchestrations using same video
- Maximum efficiency

**Cons:**
- Cache management complexity (cleanup, size limits)
- Need to hash video to create cache key
- Resolution/FPS variations need separate caches

---

## Recommended Implementation Order

### Priority 1: Add Cropping (Quick Fix)
- Add `crop_to_fit` parameter to `load_structure_video_frames()`
- Default to `True` for correct aspect ratio handling
- **Impact:** Fixes motion distortion immediately
- **Effort:** ~30 minutes

### Priority 2: Orchestrator-Level Cache (Performance)
- Extract flows once at orchestrator level
- Save to run directory as `.npz` file
- Pass cache path to all segments
- **Impact:** 3x faster for 3-segment runs, scales linearly
- **Effort:** ~2 hours

### Priority 3: Global Cache (Optional Optimization)
- Implement hash-based global cache
- Add cache management (size limits, cleanup)
- **Impact:** Speeds up repeated structure videos across runs
- **Effort:** ~4 hours

---

## Summary

| Issue | Current Behavior | Problem | Fix Priority |
|-------|-----------------|---------|--------------|
| **Size Handling** | Simple resize | Distorts motion if aspect ratios differ | **HIGH** |
| **Extraction Timing** | Per-segment | 3x redundant work for 3 segments | **HIGH** |
| **Caching** | None | Wastes 15-30 seconds per run | **MEDIUM** |

**Quick wins:** Fix cropping + add per-orchestration cache = massive improvement with minimal effort.

