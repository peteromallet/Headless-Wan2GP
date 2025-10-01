# Fix: Honor "clip" Semantics Correctly

## User's Original Request

> **"If Clip, it should just use the frames as we pass them - if there are too many it should just use the ones in range. If there are too few, it should also do this."**

### Meaning:
- **Too many flows**: Use only what's needed (truncate)
- **Too few flows**: Use only what's available, **leave remaining frames unchanged**

---

## Current Implementation (WRONG)

### Problem 1: Padding in "clip" mode
```python
if treatment == "clip":
    if source_count >= target_count:
        adjusted = flow_fields[:target_count]  # ✓ Correct
    else:
        # ✗ WRONG: Pads by repeating last flow
        padding = [flow_fields[-1].copy() for _ in range(target_count - source_count)]
        adjusted = flow_fields + padding
```

### Problem 2: Fallback repeats last flow
```python
for offset in range(range_length):
    if flow_idx >= len(flow_fields):
        # ✗ WRONG: Uses last flow repeatedly
        flow = flow_fields[-1]
    else:
        flow = flow_fields[flow_idx]
        flow_idx += 1
    
    # Always warps, even when out of flows
    warped_frame = apply_optical_flow_warp(current_frame, flow, parsed_res_wh)
    updated_frames[frame_idx] = warped_frame
```

---

## Corrected Implementation

### Fix 1: Don't pad in "clip" mode

```python
def adjust_flow_field_count(
    flow_fields: list[np.ndarray],
    target_count: int,
    treatment: str,
    dprint=print
) -> list[np.ndarray]:
    """
    Adjust number of flow fields to match target count.
    
    Args:
        flow_fields: List of optical flow fields [H, W, 2]
        target_count: Desired number of flow fields
        treatment: "adjust" (interpolate) or "clip" (use what's available)
        
    Returns:
        Adjusted list of flow fields (may be shorter than target_count for "clip")
    """
    source_count = len(flow_fields)
    
    if source_count == 0:
        raise ValueError("No flow fields provided")
    
    if source_count == target_count:
        dprint(f"[FLOW_ADJUST] Count matches ({source_count}), no adjustment needed")
        return flow_fields
    
    if treatment == "clip":
        if source_count >= target_count:
            # Truncate - use only what's needed
            adjusted = flow_fields[:target_count]
            dprint(f"[FLOW_ADJUST] Clipped from {source_count} to {target_count} flows")
        else:
            # ✅ FIXED: Don't pad, just return what we have
            adjusted = flow_fields
            dprint(f"[FLOW_ADJUST] Clip mode: Using all {source_count} available flows (target was {target_count})")
            dprint(f"[FLOW_ADJUST] Remaining {target_count - source_count} frames will be left unchanged")
        
        return adjusted
    
    elif treatment == "adjust":
        # Temporal interpolation (unchanged - this is correct)
        adjusted_flows = []
        
        for target_idx in range(target_count):
            if target_count == 1:
                source_idx_float = 0.0
            else:
                source_idx_float = target_idx * (source_count - 1) / (target_count - 1)
            
            source_idx_low = int(np.floor(source_idx_float))
            source_idx_high = int(np.ceil(source_idx_float))
            
            source_idx_low = np.clip(source_idx_low, 0, source_count - 1)
            source_idx_high = np.clip(source_idx_high, 0, source_count - 1)
            
            if source_idx_low == source_idx_high:
                adjusted_flows.append(flow_fields[source_idx_low].copy())
            else:
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

### Fix 2: Stop applying motion when out of flows in "clip" mode

```python
def apply_structure_motion_with_tracking(
    frames_for_guide_list: list[np.ndarray],
    guidance_tracker: GuidanceTracker,
    structure_video_path: str,
    structure_video_treatment: str,  # ← Need this to know how to handle exhaustion
    parsed_res_wh: tuple[int, int],
    fps_helpers: int,
    dprint=print
) -> list[np.ndarray]:
    """Apply structure motion to unguidanced frames."""
    
    unguidanced_ranges = guidance_tracker.get_unguidanced_ranges()
    
    if not unguidanced_ranges:
        return frames_for_guide_list
    
    total_unguidanced = sum(end - start + 1 for start, end in unguidanced_ranges)
    
    try:
        # Load structure video frames
        structure_frames = load_structure_video_frames(
            structure_video_path,
            target_frame_count=total_unguidanced + 1,
            target_fps=fps_helpers,
            target_resolution=parsed_res_wh,
            dprint=dprint
        )
        
        if len(structure_frames) < 2:
            dprint(f"[WARNING] Structure video has insufficient frames ({len(structure_frames)})")
            return frames_for_guide_list
        
        # Extract optical flow
        flow_fields, flow_vis = extract_optical_flow_from_frames(
            structure_frames,
            dprint=dprint
        )
        
        # Adjust flow count (may return fewer flows in "clip" mode)
        flow_fields = adjust_flow_field_count(
            flow_fields,
            target_count=total_unguidanced,
            treatment=structure_video_treatment,
            dprint=dprint
        )
        
        # Track how many flows we actually have
        available_flow_count = len(flow_fields)
        flows_applied = 0
        frames_skipped = 0
        
        # Apply flow to unguidanced ranges
        updated_frames = frames_for_guide_list.copy()
        flow_idx = 0
        
        for range_idx, (start_idx, end_idx) in enumerate(unguidanced_ranges):
            range_length = end_idx - start_idx + 1
            
            # Get anchor frame
            anchor_idx = guidance_tracker.get_anchor_frame_index(start_idx)
            
            if anchor_idx is not None:
                current_frame = updated_frames[anchor_idx].copy()
                dprint(f"[STRUCTURE_VIDEO] Range {range_idx}: frames {start_idx}-{end_idx}, anchor=frame_{anchor_idx}")
            else:
                current_frame = np.full((parsed_res_wh[1], parsed_res_wh[0], 3), 128, dtype=np.uint8)
                dprint(f"[STRUCTURE_VIDEO] Range {range_idx}: frames {start_idx}-{end_idx}, anchor=gray")
            
            # Apply motion progressively through range
            for offset in range(range_length):
                frame_idx = start_idx + offset
                
                # ✅ FIXED: Check if we're out of flows
                if flow_idx >= available_flow_count:
                    if structure_video_treatment == "clip":
                        # In "clip" mode, STOP applying motion when out of flows
                        # Leave remaining frames unchanged (gray/unguidanced)
                        frames_skipped += 1
                        dprint(f"[STRUCTURE_VIDEO] Clip mode: Out of flows at frame {frame_idx}, leaving unchanged")
                        # Don't modify updated_frames[frame_idx] - leave it as is
                        continue  # ← Skip to next frame
                    else:
                        # In "adjust" mode, this shouldn't happen (we interpolated to match)
                        # But as safety fallback, repeat last flow
                        dprint(f"[WARNING] Exhausted flows in adjust mode at frame {frame_idx} (shouldn't happen)")
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
                current_frame = warped_frame
                flows_applied += 1
        
        # Summary logging
        if structure_video_treatment == "clip" and frames_skipped > 0:
            dprint(f"[STRUCTURE_VIDEO] Clip mode summary:")
            dprint(f"  - Applied {flows_applied} flows to {flows_applied} frames")
            dprint(f"  - Left {frames_skipped} frames unchanged (ran out of flows)")
            dprint(f"  - Total unguidanced: {total_unguidanced} frames")
        else:
            dprint(f"[STRUCTURE_VIDEO] Applied {flows_applied} flows to {flows_applied} frames")
        
        return updated_frames
    
    except Exception as e:
        dprint(f"[ERROR] Structure motion application failed: {e}")
        import traceback
        traceback.print_exc()
        return frames_for_guide_list
```

---

## Semantic Comparison

| Mode | Structure Video Length | Behavior |
|------|----------------------|----------|
| **"clip"** | Too many frames | ✅ Truncate to needed count |
| **"clip"** | Too few frames | ✅ Use all available, **leave remaining unchanged** |
| **"adjust"** | Too many frames | ✅ Temporal downsampling (interpolate) |
| **"adjust"** | Too few frames | ✅ Temporal upsampling (interpolate) |

---

## Example Scenarios

### Scenario 1: "clip" mode, structure video too short

```
Unguidanced frames needed: 30
Structure video flows available: 20

BEFORE (WRONG):
- Apply flows 0-19 to frames 0-19
- Apply flow 19 (repeated) to frames 20-29  ✗ Wrong!

AFTER (CORRECT):
- Apply flows 0-19 to frames 0-19
- Leave frames 20-29 unchanged (gray)  ✓ Correct!
```

### Scenario 2: "clip" mode, structure video too long

```
Unguidanced frames needed: 20
Structure video flows available: 30

BOTH (CORRECT):
- Truncate to 20 flows
- Apply flows 0-19 to frames 0-19
- Discard flows 20-29 (unused)
```

### Scenario 3: "adjust" mode, any mismatch

```
Unguidanced frames needed: 30
Structure video flows available: 20 (or 40, doesn't matter)

BOTH (CORRECT):
- Temporally interpolate to exactly 30 flows
- Apply all 30 interpolated flows to frames 0-29
- No frames left unchanged
```

---

## Visual Debug Output

### "clip" mode with insufficient flows:
```
[FLOW_ADJUST] Clip mode: Using all 20 available flows (target was 30)
[FLOW_ADJUST] Remaining 10 frames will be left unchanged
[STRUCTURE_VIDEO] Range 0: frames 50-79, anchor=frame_49
[STRUCTURE_VIDEO] Clip mode: Out of flows at frame 70, leaving unchanged
[STRUCTURE_VIDEO] Clip mode: Out of flows at frame 71, leaving unchanged
...
[STRUCTURE_VIDEO] Clip mode summary:
  - Applied 20 flows to 20 frames
  - Left 10 frames unchanged (ran out of flows)
  - Total unguidanced: 30 frames
```

### "adjust" mode (always fills completely):
```
[FLOW_ADJUST] Interpolated from 20 to 30 flows
[STRUCTURE_VIDEO] Range 0: frames 50-79, anchor=frame_49
[STRUCTURE_VIDEO] Applied 30 flows to 30 frames
```

---

## Summary

The fix ensures:

1. ✅ **"clip" mode never pads or repeats flows** - uses only what's available
2. ✅ **Remaining frames stay unchanged** (gray/unguidanced) when flows run out in "clip" mode
3. ✅ **"adjust" mode always fills completely** via temporal interpolation
4. ✅ **Clear logging** shows what happened in each mode

This now **correctly honors your original request**!

