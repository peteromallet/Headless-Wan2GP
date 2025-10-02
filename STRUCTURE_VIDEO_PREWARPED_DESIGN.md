# Structure Motion - Pre-Warped Video Approach

## The Simpler Idea

**Instead of caching flows, cache the pre-warped frames as a video!**

```
Current Approach (Flow Cache):
Orchestrator: Structure video → Extract flows → Save flows.npz → Upload
Segments: Download flows → Apply warping → Use frames

Simpler Approach (Pre-Warped Video):
Orchestrator: Structure video → Extract flows → Apply warping → Save video → Upload
Segments: Download video → Extract frames → Use directly
```

---

## Key Insight: Pre-Generate Motion-Applied Frames

### Orchestrator Does Heavy Lifting

```python
def _create_structure_motion_video(
    structure_video_path: str,
    max_frames_needed: int,  # Based on longest segment gap
    target_resolution: tuple[int, int],
    target_fps: int,
    output_path: Path,
    dprint=print
) -> Path:
    """
    Create a video of pre-warped frames showing structure motion.
    
    Process:
    1. Extract flows from structure video
    2. Start with gray/neutral frame
    3. Apply flows progressively (gray → warp → warp → warp...)
    4. Save as video (H.264 compressed)
    5. Upload to Supabase
    
    Result: A video showing motion evolution that segments can extract from
    """
    from source.structure_video_guidance import (
        load_structure_video_frames,
        extract_optical_flow_from_frames,
        apply_optical_flow_warp
    )
    
    # Load structure video and extract flows
    dprint(f"[STRUCTURE_MOTION_VIDEO] Loading structure video...")
    structure_frames = load_structure_video_frames(
        structure_video_path,
        target_frame_count=max_frames_needed,
        target_fps=target_fps,
        target_resolution=target_resolution,
        dprint=dprint
    )
    
    dprint(f"[STRUCTURE_MOTION_VIDEO] Extracting optical flows...")
    flow_fields, _ = extract_optical_flow_from_frames(structure_frames, dprint=dprint)
    
    # Apply flows progressively starting from gray
    dprint(f"[STRUCTURE_MOTION_VIDEO] Generating motion-applied frames...")
    
    w, h = target_resolution
    gray_frame = np.full((h, w, 3), 128, dtype=np.uint8)  # Neutral starting point
    
    motion_frames = [gray_frame]  # Start with gray
    current_frame = gray_frame
    
    for i, flow in enumerate(flow_fields):
        # Apply flow to current frame
        warped_frame = apply_optical_flow_warp(
            source_frame=current_frame,
            flow=flow,
            target_resolution=target_resolution,
            motion_strength=1.0  # Full motion
        )
        
        motion_frames.append(warped_frame)
        current_frame = warped_frame  # Chain forward
        
        if (i + 1) % 50 == 0:
            dprint(f"[STRUCTURE_MOTION_VIDEO] Generated {i+1}/{len(flow_fields)} frames")
    
    # Save as video
    dprint(f"[STRUCTURE_MOTION_VIDEO] Encoding video to {output_path}")
    
    from source.video_utils import create_video_from_frames_list
    video_path = create_video_from_frames_list(
        motion_frames,
        output_path,
        target_fps,
        target_resolution
    )
    
    dprint(f"[STRUCTURE_MOTION_VIDEO] Created {len(motion_frames)} frame video")
    
    return video_path
```

### Segments Extract Frames

```python
def load_frames_from_structure_motion_video(
    structure_motion_video_url: str,
    frame_start: int,
    frame_count: int,
    download_dir: Path,
    dprint=print
) -> List[np.ndarray]:
    """
    Download structure motion video and extract needed frames.
    
    Much simpler than flow application!
    """
    import requests
    from source.video_utils import extract_frames_from_video
    
    # Download video
    local_video_path = download_dir / "structure_motion.mp4"
    
    dprint(f"[STRUCTURE_MOTION] Downloading pre-warped video...")
    response = requests.get(structure_motion_video_url, timeout=60)
    response.raise_for_status()
    
    with open(local_video_path, 'wb') as f:
        f.write(response.content)
    
    dprint(f"[STRUCTURE_MOTION] Downloaded {len(response.content) / 1024 / 1024:.2f} MB")
    
    # Extract needed frames
    dprint(f"[STRUCTURE_MOTION] Extracting frames {frame_start} to {frame_start + frame_count}")
    
    frames = extract_frames_from_video(
        str(local_video_path),
        start_frame=frame_start,
        num_frames=frame_count,
        dprint_func=dprint
    )
    
    dprint(f"[STRUCTURE_MOTION] Extracted {len(frames)} frames")
    
    return frames
```

---

## Critical Question: What About Different Anchors?

### The Problem

**Current approach:**
```
Segment 0:
  anchor = last_overlap_frame (from previous segment output)
  frame_20 = warp(anchor, flow_0)  ← Evolves from specific content
  frame_21 = warp(frame_20, flow_1)
  
Segment 1:
  anchor = different_overlap_frame  ← Different starting point!
  frame_20 = warp(anchor, flow_0)
  frame_21 = warp(frame_20, flow_1)
```

Each segment starts from different content (their specific overlap frames).

**Pre-warped approach:**
```
Orchestrator generates:
  frame_0 = gray
  frame_1 = warp(gray, flow_0)
  frame_2 = warp(frame_1, flow_1)
  ... all frames evolved from gray

All segments use these same pre-warped frames.
```

All segments use frames evolved from the same neutral gray starting point.

### Does This Matter?

**For motion guidance: Probably not much!**

The purpose of structure motion is to provide **motion patterns**, not specific content:
- The AI generates new content based on the motion guidance
- Whether the motion evolved from gray or from a specific anchor might not significantly affect the final result
- The motion vectors are the same, just applied to different starting content

**However:**
- Segments starting from their actual overlap frames would have more visual continuity
- Progressive evolution from overlap → motion is more coherent
- But AI will override most of the visual content anyway

---

## Approach Comparison

### Flow Cache (Complex)

**Storage:** 200-300 MB (.npz flows)  
**Segment work:** Download → Apply warping (GPU work) → Use frames  
**Flexibility:** Can vary motion_strength per segment, starts from segment's anchor  
**Pros:** More flexible, preserves per-segment anchor evolution  
**Cons:** Larger file, segments still do warping  

### Pre-Warped Video (Simple)

**Storage:** 10-50 MB (H.264 compressed video)  
**Segment work:** Download → Extract frames → Use directly  
**Flexibility:** Fixed motion_strength, generic anchor (gray)  
**Pros:** Much smaller, no GPU work per segment, super simple  
**Cons:** All segments use same "evolved from gray" frames  

---

## Hybrid Approach: Best of Both?

**What if we pre-warp BUT still allow segment-specific anchors?**

### Option A: Store Multiple Versions
```
Orchestrator generates:
- structure_motion_gray.mp4 (evolved from gray)
- structure_motion_img0.mp4 (evolved from first keyframe)
- structure_motion_img1.mp4 (evolved from second keyframe)

Segments choose which video to use based on their anchor.
```

**Pros:** Flexibility  
**Cons:** Multiple large files

### Option B: Just Use Gray (Simplest)
```
Orchestrator generates:
- structure_motion.mp4 (evolved from gray, one size fits all)

All segments extract their needed frames.
Motion pattern is consistent, visual evolution from gray.
```

**Pros:** Dead simple, tiny file, fast  
**Cons:** Loses per-segment anchor specificity  

---

## Recommendation

### For Your Use Case: **Pre-Warped Video (Option B)**

**Why:**
1. **Much smaller:** 10-50 MB vs 200-300 MB
2. **Much faster:** No warping per segment (just frame extraction)
3. **Simpler code:** Just download & extract frames
4. **Good enough:** Motion guidance doesn't need per-anchor evolution
5. **Distributed-friendly:** Small files download fast

**Trade-off accepted:**
- All segments use frames evolved from gray instead of their specific anchors
- Probably won't affect final quality much since AI generates new content anyway

### Implementation

**Orchestrator:**
```python
# After downloading structure video
structure_motion_video_path = _create_structure_motion_video(
    structure_video_path=structure_video_path,
    max_frames_needed=calculate_max_gap_size(all_segments),
    target_resolution=parsed_res_wh,
    target_fps=fps_helpers,
    output_path=current_run_output_dir / "structure_motion.mp4",
    dprint=dprint
)

# Upload to Supabase
structure_motion_video_url = upload_and_get_final_output_location(
    local_path=str(structure_motion_video_path),
    task_id=orchestrator_task_id_str,
    project_id=orchestrator_project_id,
    is_final_output=False
)

# Pass to segments
orchestrator_payload["structure_motion_video_url"] = structure_motion_video_url
```

**Segments:**
```python
# In apply_structure_motion_with_tracking()

if structure_motion_video_url:
    # Download and extract needed frames
    motion_frames = load_frames_from_structure_motion_video(
        structure_motion_video_url,
        frame_start=0,  # Or calculate based on segment
        frame_count=total_unguidanced,
        download_dir=segment_processing_dir,
        dprint=dprint
    )
    
    # Drop frames directly into unguidanced ranges
    frame_idx = 0
    for range_start, range_end in unguidanced_ranges:
        for pos in range(range_start, range_end + 1):
            if frame_idx < len(motion_frames):
                updated_frames[pos] = motion_frames[frame_idx]
                guidance_tracker.mark_single_frame(pos)
                frame_idx += 1
```

---

## Performance Comparison

### Flow Cache
```
Orchestrator: Extract flows (10s) + Upload 300MB (3s) = 13s
Segment: Download 300MB (3s) + Warp (2s) = 5s per segment
Total (3 segments): 13 + 5 + 5 + 5 = 28s
```

### Pre-Warped Video
```
Orchestrator: Extract flows (10s) + Warp (5s) + Encode (2s) + Upload 30MB (0.5s) = 17.5s
Segment: Download 30MB (0.5s) + Extract frames (0.2s) = 0.7s per segment
Total (3 segments): 17.5 + 0.7 + 0.7 + 0.7 = 19.6s

Parallel segments: 17.5 + 0.7 = 18.2s
```

**Verdict:** Similar total time, but:
- 10x smaller files (30 MB vs 300 MB)
- Segments are 7x faster (0.7s vs 5s)
- Better for distributed workers (smaller downloads)

---

## Conclusion

✅ **Pre-warped video approach is better for your use case**

**Why:**
- Simpler implementation
- Smaller files (better for Supabase/bandwidth)
- Faster segments (no GPU warping)
- Good enough quality (motion pattern is what matters)

**Trade-off:**
- Less flexible (can't vary motion_strength per segment)
- Generic anchor evolution (gray → motion)
- But these probably don't matter for final quality

**Want me to implement this instead of flow caching?** It's actually simpler and likely better suited to your distributed architecture.

