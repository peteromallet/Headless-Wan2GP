# Structure Flow Cache Design - Supabase Storage

## Architecture

```
Orchestrator (once):
├─ Extract ALL flows from structure video
├─ Save flows to .npz file locally
├─ Upload .npz to Supabase Storage
├─ Store Supabase URL in orchestrator_payload
└─ Pass URL to all segment payloads

Segment 0 (parallel):
├─ Download flows .npz from Supabase URL
├─ Load flows array
├─ Slice flows for frames 20-62 (43 flows)
└─ Apply motion

Segment 1 (parallel):
├─ Download flows .npz from Supabase URL
├─ Load flows array
├─ Slice flows for frames 20-62 (43 flows)
└─ Apply motion

Segment 2 (parallel):
├─ Download flows .npz from Supabase URL  
├─ Load flows array
├─ Slice flows for frames 20-62 (43 flows)
└─ Apply motion
```

**Benefits:**
- ✅ Flows extracted once (5-10 sec saved per segment)
- ✅ Works across distributed workers (segments can run on different machines)
- ✅ Follows existing Supabase upload pattern
- ✅ Cached flows accessible to all segments
- ✅ Automatic cleanup (orchestrator folder deleted → flows deleted)

---

## Implementation

### Part 1: Orchestrator - Extract & Upload Flows

**File:** `source/sm_functions/travel_between_images.py`

**New function:**
```python
def _extract_and_upload_structure_flows(
    structure_video_path: str,
    orchestrator_task_id: str,
    current_run_output_dir: Path,
    project_id: str | None,
    dprint=print
) -> str | None:
    """
    Extract optical flows from structure video, save to .npz, upload to Supabase.
    
    Returns:
        Supabase URL of uploaded flows file, or None if extraction failed
    """
    from source.structure_video_guidance import (
        load_structure_video_frames,
        extract_optical_flow_from_frames
    )
    from source.common_utils import upload_and_get_final_output_location
    import numpy as np
    import cv2
    
    try:
        dprint(f"[STRUCTURE_FLOW_CACHE] Starting flow extraction from structure video")
        
        # Get structure video info
        cap = cv2.VideoCapture(structure_video_path)
        if not cap.isOpened():
            dprint(f"[ERROR] Could not open structure video: {structure_video_path}")
            return None
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        dprint(f"[STRUCTURE_FLOW_CACHE] Structure video: {total_frames} frames, {width}x{height}, {fps} fps")
        
        # Load all frames at native resolution
        from Wan2GP.wgp import get_resampled_video
        from pathlib import Path
        
        dprint(f"[STRUCTURE_FLOW_CACHE] Loading {total_frames} frames...")
        frames = get_resampled_video(
            structure_video_path,
            start_frame=0,
            max_frames=total_frames,
            target_fps=fps,  # Keep original FPS
            bridge='torch'
        )
        
        if not frames or len(frames) < 2:
            dprint(f"[ERROR] Insufficient frames loaded: {len(frames) if frames else 0}")
            return None
        
        dprint(f"[STRUCTURE_FLOW_CACHE] Loaded {len(frames)} frames")
        
        # Convert to numpy
        frames_np = []
        for frame in frames:
            if hasattr(frame, 'cpu'):
                frame_np = frame.cpu().numpy()
            else:
                frame_np = np.array(frame)
            
            if frame_np.dtype != np.uint8:
                frame_np = np.clip(frame_np, 0, 255).astype(np.uint8)
            
            frames_np.append(frame_np)
        
        # Extract optical flows
        dprint(f"[STRUCTURE_FLOW_CACHE] Extracting optical flows from {len(frames_np)} frames...")
        flow_fields, flow_vis = extract_optical_flow_from_frames(frames_np, dprint=dprint)
        
        dprint(f"[STRUCTURE_FLOW_CACHE] Extracted {len(flow_fields)} flow fields")
        
        # Save flows to local .npz file
        flow_cache_filename = f"structure_flows_{orchestrator_task_id}.npz"
        local_flow_cache_path = current_run_output_dir / flow_cache_filename
        
        dprint(f"[STRUCTURE_FLOW_CACHE] Saving flows to {local_flow_cache_path}")
        
        # Stack flows into single array for efficient storage
        flows_array = np.stack(flow_fields)  # Shape: [N-1, H, W, 2]
        
        np.savez_compressed(
            local_flow_cache_path,
            flows=flows_array,
            source_video=structure_video_path,
            total_flows=len(flow_fields),
            original_fps=fps,
            original_resolution=(width, height)
        )
        
        dprint(f"[STRUCTURE_FLOW_CACHE] Saved {flows_array.nbytes / 1024 / 1024:.2f} MB to disk")
        
        # Upload to Supabase storage
        dprint(f"[STRUCTURE_FLOW_CACHE] Uploading flows to Supabase...")
        
        final_url = upload_and_get_final_output_location(
            local_path=str(local_flow_cache_path),
            task_id=orchestrator_task_id,
            project_id=project_id,
            is_final_output=False  # This is an intermediate cache file
        )
        
        if final_url:
            dprint(f"[STRUCTURE_FLOW_CACHE] Successfully uploaded to: {final_url}")
            return final_url
        else:
            dprint(f"[STRUCTURE_FLOW_CACHE] Upload failed, segments will extract flows individually")
            return None
            
    except Exception as e:
        dprint(f"[ERROR] Flow extraction/upload failed: {e}")
        import traceback
        traceback.print_exc()
        return None
```

**Integration in orchestrator handler:**
```python
def _handle_travel_orchestrator_task(...):
    # ... existing code ...
    
    # Extract and validate structure video parameters
    structure_video_path = orchestrator_payload.get("structure_video_path")
    structure_video_treatment = orchestrator_payload.get("structure_video_treatment", "adjust")
    structure_video_motion_strength = orchestrator_payload.get("structure_video_motion_strength", 1.0)
    structure_flow_cache_url = None  # ← NEW
    
    if structure_video_path:
        # Download if URL
        from ..common_utils import download_video_if_url
        structure_video_path = download_video_if_url(
            structure_video_path,
            download_target_dir=current_run_output_dir,
            task_id_for_logging=orchestrator_task_id_str,
            descriptive_name="structure_video"
        )
        
        # Validate structure video exists
        if not Path(structure_video_path).exists():
            raise ValueError(f"Structure video not found: {structure_video_path}")
        
        # Validate treatment mode
        if structure_video_treatment not in ["adjust", "clip"]:
            raise ValueError(f"Invalid structure_video_treatment: {structure_video_treatment}")
        
        dprint(f"[STRUCTURE_VIDEO] Using: {structure_video_path}")
        dprint(f"[STRUCTURE_VIDEO] Treatment: {structure_video_treatment}")
        dprint(f"[STRUCTURE_VIDEO] Motion strength: {structure_video_motion_strength}")
        
        # NEW: Extract flows and upload to Supabase
        structure_flow_cache_url = _extract_and_upload_structure_flows(
            structure_video_path=structure_video_path,
            orchestrator_task_id=orchestrator_task_id_str,
            current_run_output_dir=current_run_output_dir,
            project_id=orchestrator_project_id,
            dprint=dprint
        )
        
        if structure_flow_cache_url:
            dprint(f"[STRUCTURE_VIDEO] Flow cache available at: {structure_flow_cache_url}")
        else:
            dprint(f"[STRUCTURE_VIDEO] No flow cache - segments will extract flows individually")
    
    # Loop to queue all segment tasks
    for idx in range(num_segments):
        # ... existing code ...
        
        segment_payload = {
            # ... existing fields ...
            
            # Structure video guidance parameters
            "structure_video_path": structure_video_path,
            "structure_video_treatment": structure_video_treatment,
            "structure_video_motion_strength": structure_video_motion_strength,
            "structure_flow_cache_url": structure_flow_cache_url,  # ← NEW
        }
```

---

### Part 2: Segment - Download & Slice Flows

**File:** `source/structure_video_guidance.py`

**New function to download and load cached flows:**
```python
def download_and_load_cached_flows(
    flow_cache_url: str,
    download_target_dir: Path,
    task_id: str,
    dprint: Callable = print
) -> Optional[np.ndarray]:
    """
    Download cached flow .npz from Supabase and load flows array.
    
    Args:
        flow_cache_url: Supabase URL to flows .npz file
        download_target_dir: Directory to download to
        task_id: Task ID for logging
        dprint: Debug print function
        
    Returns:
        Flows array of shape [N, H, W, 2] or None if download/load failed
    """
    import requests
    import numpy as np
    from pathlib import Path
    
    try:
        # Download from Supabase
        local_cache_path = download_target_dir / "structure_flows_cache.npz"
        
        dprint(f"[STRUCTURE_FLOW_CACHE] Downloading cached flows from {flow_cache_url}")
        
        response = requests.get(flow_cache_url, timeout=60)
        response.raise_for_status()
        
        with open(local_cache_path, 'wb') as f:
            f.write(response.content)
        
        dprint(f"[STRUCTURE_FLOW_CACHE] Downloaded {len(response.content) / 1024 / 1024:.2f} MB")
        
        # Load flows
        dprint(f"[STRUCTURE_FLOW_CACHE] Loading flows from cache...")
        cache_data = np.load(local_cache_path, allow_pickle=True)
        
        flows_array = cache_data['flows']
        original_fps = float(cache_data['original_fps'])
        original_res = tuple(cache_data['original_resolution'])
        
        dprint(f"[STRUCTURE_FLOW_CACHE] Loaded {len(flows_array)} flows")
        dprint(f"[STRUCTURE_FLOW_CACHE] Original: {original_res[0]}x{original_res[1]} @ {original_fps} fps")
        
        return flows_array
        
    except Exception as e:
        dprint(f"[ERROR] Failed to download/load cached flows: {e}")
        import traceback
        traceback.print_exc()
        return None
```

**Update `apply_structure_motion_with_tracking`:**
```python
def apply_structure_motion_with_tracking(
    frames_for_guide_list: List[np.ndarray],
    guidance_tracker: GuidanceTracker,
    structure_video_path: str,
    structure_video_treatment: str,
    parsed_res_wh: Tuple[int, int],
    fps_helpers: int,
    motion_strength: float = 1.0,
    structure_flow_cache_url: str | None = None,  # ← NEW parameter
    segment_processing_dir: Path | None = None,   # ← NEW parameter
    task_id: str = "unknown",                      # ← NEW parameter
    dprint: Callable = print
) -> List[np.ndarray]:
    """
    Apply structure motion to unguidanced frames.
    
    New parameters:
        structure_flow_cache_url: Optional Supabase URL to cached flows
        segment_processing_dir: Directory for downloading cache
        task_id: Task ID for logging
    """
    # Get unguidanced ranges
    unguidanced_ranges = guidance_tracker.get_unguidanced_ranges()
    
    if not unguidanced_ranges:
        dprint(f"[STRUCTURE_VIDEO] No unguidanced frames found")
        return frames_for_guide_list
    
    total_unguidanced = sum(end - start + 1 for start, end in unguidanced_ranges)
    
    dprint(f"[STRUCTURE_VIDEO] Processing {total_unguidanced} unguidanced frames across {len(unguidanced_ranges)} ranges")
    
    try:
        # --- NEW: Try to use cached flows first ---
        flow_fields = None
        
        if structure_flow_cache_url and segment_processing_dir:
            dprint(f"[STRUCTURE_VIDEO] Attempting to use cached flows")
            
            flows_array = download_and_load_cached_flows(
                flow_cache_url=structure_flow_cache_url,
                download_target_dir=segment_processing_dir,
                task_id=task_id,
                dprint=dprint
            )
            
            if flows_array is not None:
                # Convert array back to list of flows
                flow_fields = [flows_array[i] for i in range(len(flows_array))]
                
                dprint(f"[STRUCTURE_VIDEO] Using {len(flow_fields)} cached flows")
                
                # Resize flows to target resolution if needed
                h, w = parsed_res_wh[1], parsed_res_wh[0]
                original_h, original_w = flow_fields[0].shape[:2]
                
                if (original_h, original_w) != (h, w):
                    dprint(f"[STRUCTURE_VIDEO] Resizing flows from {original_w}x{original_h} to {w}x{h}")
                    
                    import cv2
                    resized_flows = []
                    for flow in flow_fields:
                        # Resize flow field
                        flow_resized = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR)
                        
                        # Scale flow vectors proportionally
                        flow_resized[:, :, 0] *= w / original_w  # X displacement
                        flow_resized[:, :, 1] *= h / original_h  # Y displacement
                        
                        resized_flows.append(flow_resized)
                    
                    flow_fields = resized_flows
                    dprint(f"[STRUCTURE_VIDEO] Flow resize complete")
        
        # --- Fallback: Extract flows if cache unavailable ---
        if flow_fields is None:
            dprint(f"[STRUCTURE_VIDEO] No cache available, extracting flows from video")
            
            # Original extraction logic
            structure_frames = load_structure_video_frames(
                structure_video_path,
                target_frame_count=total_unguidanced,
                target_fps=fps_helpers,
                target_resolution=parsed_res_wh,
                dprint=dprint
            )
            
            if len(structure_frames) < 2:
                dprint(f"[WARNING] Structure video has insufficient frames")
                return frames_for_guide_list
            
            flow_fields, flow_vis = extract_optical_flow_from_frames(
                structure_frames,
                dprint=dprint
            )
            
            dprint(f"[STRUCTURE_VIDEO] Extracted {len(flow_fields)} flows from video")
        
        # --- Rest of existing logic (adjust flows, apply motion) ---
        flow_fields = adjust_flow_field_count(
            flow_fields,
            target_count=total_unguidanced,
            treatment=structure_video_treatment,
            dprint=dprint
        )
        
        # ... existing motion application code ...
        
    except Exception as e:
        dprint(f"[ERROR] Structure motion application failed: {e}")
        traceback.print_exc()
        return frames_for_guide_list
```

**Update call sites in `video_utils.py`:**
```python
# In create_guide_video_for_travel_segment()

# Extract parameters
structure_flow_cache_url = full_orchestrator_payload.get("structure_flow_cache_url")

# Apply structure motion
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
        motion_strength=structure_video_motion_strength,
        structure_flow_cache_url=structure_flow_cache_url,  # ← NEW
        segment_processing_dir=output_target_dir,           # ← NEW
        task_id=task_id_for_logging,                        # ← NEW
        dprint=dprint
    )
```

---

## Flow Slicing Strategy

**Question:** Do we need to slice flows for each segment?

**Answer:** No! The `adjust_flow_field_count()` function already handles this:

```python
# Each segment calculates its own unguidanced frame count
total_unguidanced = sum(end - start + 1 for start, end in unguidanced_ranges)
# e.g., Segment 0: 43 frames, Segment 1: 43 frames, Segment 2: 43 frames

# adjust_flow_field_count() slices/interpolates the cached flows
flow_fields = adjust_flow_field_count(
    flow_fields,           # All flows from cache (e.g., 200 flows)
    target_count=43,       # Segment needs 43 flows
    treatment="adjust",    # Interpolate to match
    dprint=dprint
)
# Returns: 43 flows interpolated from the full cache
```

**So each segment:**
1. Downloads the FULL flow cache (same .npz for all)
2. Calls `adjust_flow_field_count()` to get exactly the flows it needs
3. In "adjust" mode: interpolates from full cache to match segment's frame count
4. In "clip" mode: uses first N flows from cache

---

## Performance Analysis

### Before (Current)
```
Orchestrator: 0.5 sec
Segment 0: Load video (2s) + Extract flows (8s) + Apply (1s) = 11s
Segment 1: Load video (2s) + Extract flows (8s) + Apply (1s) = 11s
Segment 2: Load video (2s) + Extract flows (8s) + Apply (1s) = 11s

Total: 0.5 + 11 + 11 + 11 = 33.5 seconds
```

### After (With Cache)
```
Orchestrator: Extract flows (10s) + Upload (2s) = 12.5s
Segment 0: Download cache (1s) + Apply (1s) = 2s
Segment 1: Download cache (1s) + Apply (1s) = 2s
Segment 2: Download cache (1s) + Apply (1s) = 2s

Total: 12.5 + 2 + 2 + 2 = 18.5 seconds

Savings: 15 seconds (45% faster)
```

### Parallel Segments (Best Case)
```
Orchestrator: 12.5s
Segments (parallel): max(2s, 2s, 2s) = 2s

Total: 14.5 seconds

Savings: 19 seconds (57% faster)
```

---

## Storage Considerations

**Flow Cache Size:**
- Typical structure video: 100 frames @ 720p
- Flows: 99 flows × 720 × 1280 × 2 (x,y) × 4 bytes (float32) = ~700 MB
- Compressed (.npz): ~200-300 MB

**Supabase Storage:**
- Stored in task folder: `{orchestrator_task_id}/structure_flows_{orchestrator_task_id}.npz`
- Cleaned up when orchestrator folder deleted
- No global cache management needed

---

## Error Handling

**If cache upload fails:**
```python
if structure_flow_cache_url:
    # Try to use cache
else:
    # Fallback: Extract flows per segment (current behavior)
```

**If cache download fails:**
```python
flows_array = download_and_load_cached_flows(...)
if flows_array is None:
    # Fallback: Extract flows from video
```

**Result:** Graceful degradation - feature always works even if caching fails

---

## Summary

✅ **Advantages:**
- Flows extracted once at orchestrator level
- Uploaded to Supabase (persistent, distributed-worker compatible)
- Each segment downloads and uses cached flows
- 45-57% faster processing
- Graceful fallback if caching fails
- Follows existing Supabase upload patterns

✅ **Implementation Effort:**
- New: `_extract_and_upload_structure_flows()` in orchestrator
- New: `download_and_load_cached_flows()` in structure_video_guidance
- Modify: Add cache URL parameter to existing functions
- Estimated: 3-4 hours

✅ **Ready to implement?**

