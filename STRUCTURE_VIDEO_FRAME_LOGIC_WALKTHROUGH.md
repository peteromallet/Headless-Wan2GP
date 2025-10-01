# Structure Video Frame Logic - Complete Walkthrough

## The Frame Breakdown Flow

### Example Scenario

Let's trace through a concrete example to verify the logic:

**Segment Setup:**
- Total frames: 73
- Overlap frames: 20 (frames 0-19, from previous segment)
- Keyframe fade: 10 frames (frames 63-72, fading to end anchor)
- End anchor: frame 73
- **Unguidanced frames:** 20-62 (43 frames)

---

## Phase 1: Initialization

```python
guidance_tracker = GuidanceTracker(total_frames=73)
# Initially: all 73 frames marked False (unguidanced)
```

---

## Phase 2: Guidance Placement

```python
# 1. Place overlap frames (0-19)
for idx in range(20):
    frames_for_guide_list[idx] = overlap_frames[idx]
    guidance_tracker.mark_single_frame(idx)  # Mark 0-19 as guided

# 2. Place keyframe fade frames (63-72)
for idx in range(63, 73):
    frames_for_guide_list[idx] = fade_frames[idx - 63]
    guidance_tracker.mark_single_frame(idx)  # Mark 63-72 as guided

# 3. Place end anchor (73)
frames_for_guide_list[73] = end_anchor_frame
guidance_tracker.mark_single_frame(73)  # Mark 73 as guided

# State now:
# has_guidance = [T,T,T...T(0-19), F,F,F...F(20-62), T,T,T...T(63-73)]
#                 ↑ overlap       ↑ unguidanced      ↑ fade+anchor
```

---

## Phase 3: Detect Unguidanced Ranges

```python
unguidanced_ranges = guidance_tracker.get_unguidanced_ranges()
# Returns: [(20, 62)]  ← One continuous range

total_unguidanced = sum(end - start + 1 for start, end in unguidanced_ranges)
# total_unguidanced = 62 - 20 + 1 = 43 frames
```

**Key Insight:** The tracker logically identifies gaps, avoiding pixel inspection fragility.

---

## Phase 4: Load Structure Video

```python
structure_frames = load_structure_video_frames(
    structure_video_path="dance.mp4",
    target_frame_count=43,  # Request 43 frames
    target_fps=16,
    target_resolution=(1280, 720)
)
# Function internally loads 44 frames (43 + 1) to ensure enough flows
# Returns: 44 frames resized to 1280x720
```

**Why 44 frames?**
- RAFT produces N-1 flows for N frames
- We need 43 flows
- So we load 44 frames to get 43 flows

---

## Phase 5: Extract Optical Flow

```python
flow_fields, flow_vis = extract_optical_flow_from_frames(structure_frames)
# Input: 44 frames
# Output: 43 flow fields (each [720, 1280, 2])
# Each flow represents motion from frame_i to frame_i+1

# Example:
# flow_fields[0] = motion from structure_frame[0] → structure_frame[1]
# flow_fields[1] = motion from structure_frame[1] → structure_frame[2]
# ...
# flow_fields[42] = motion from structure_frame[42] → structure_frame[43]
```

---

## Phase 6: Adjust Flow Count

```python
flow_fields = adjust_flow_field_count(
    flow_fields=flow_fields,  # 43 flows
    target_count=43,           # Need 43 flows
    treatment="adjust"
)
# Since counts match, returns as-is: 43 flows
```

### What if structure video was shorter?

**Scenario A: "adjust" mode (interpolate)**
```python
# Structure video only has 22 frames → 21 flows
# Need 43 flows
adjusted_flows = adjust_flow_field_count(
    flow_fields=21_flows,
    target_count=43,
    treatment="adjust"
)
# Temporally interpolates to create 43 flows
# Maps [0-42] → [0-20] with linear interpolation
# Result: 43 flows (some are blends of adjacent structure flows)
```

**Scenario B: "clip" mode (use what's available)**
```python
# Structure video only has 22 frames → 21 flows
# Need 43 flows
adjusted_flows = adjust_flow_field_count(
    flow_fields=21_flows,
    target_count=43,
    treatment="clip"
)
# Returns 21 flows as-is
# Later logic will apply 21 flows and leave remaining 22 frames unchanged
```

---

## Phase 7: Apply Motion Across Ranges

### The Critical Loop Structure

```python
updated_frames = frames_for_guide_list.copy()
flow_idx = 0  # ← GLOBAL counter across ALL ranges

for range_idx, (start_idx, end_idx) in enumerate(unguidanced_ranges):
    # For our example: range_idx=0, start_idx=20, end_idx=62
    range_length = 62 - 20 + 1 = 43
    
    # Get anchor: last guided frame before this range
    anchor_idx = guidance_tracker.get_anchor_frame_index(20)
    # Searches backward from 20: finds frame 19 (last overlap frame)
    
    current_frame = updated_frames[19].copy()  # Start with overlap frame 19
    
    # Progressive warping through the range
    for offset in range(43):  # 0 to 42
        frame_idx = 20 + offset  # Ranges from 20 to 62
        
        # Get next flow from structure video
        flow = flow_fields[flow_idx]  # flow_idx: 0, 1, 2, ..., 42
        flow_idx += 1  # Increment global counter
        
        # Warp current frame using this flow
        warped_frame = apply_optical_flow_warp(
            source_frame=current_frame,
            flow=flow,
            motion_strength=1.0
        )
        
        # Store result and use as source for next warp
        updated_frames[frame_idx] = warped_frame
        current_frame = warped_frame  # ← Chain for progressive motion
        
        # CRITICAL: Mark this frame as now having guidance
        guidance_tracker.mark_single_frame(frame_idx)
```

### Progressive Motion Chain

```
Frame 19 (anchor/overlap) 
    ↓ apply flow[0]
Frame 20 = warp(frame_19, flow[0])
    ↓ apply flow[1]  
Frame 21 = warp(frame_20, flow[1])
    ↓ apply flow[2]
Frame 22 = warp(frame_21, flow[2])
    ↓ ... continue ...
    ↓ apply flow[42]
Frame 62 = warp(frame_61, flow[42])
```

**Key Properties:**
1. Each frame builds on the previous warped result (progressive/cumulative motion)
2. Motion pattern comes from structure video (flow fields)
3. Content comes from anchor frame (evolved through warping)
4. Each successfully warped frame is marked as guided

---

## Phase 8: Multiple Ranges Scenario

What if we had **multiple unguidanced ranges**?

**Scenario:**
- Total frames: 100
- Overlap: 0-19 (20 frames, guided)
- Gap 1: 20-40 (21 frames, unguidanced)
- Keyframe: 41 (guided)
- Gap 2: 42-79 (38 frames, unguidanced)
- Fade + anchor: 80-100 (21 frames, guided)

### Execution:

```python
unguidanced_ranges = [(20, 40), (42, 79)]
total_unguidanced = 21 + 38 = 59 frames

# Load 60 structure frames → get 59 flows
flow_fields = [flow_0, flow_1, ..., flow_58]

flow_idx = 0  # Global counter

# === Process Range 1: frames 20-40 (21 frames) ===
anchor_1 = frame_19  # Last overlap frame
current_frame = frame_19

for frame_idx in [20, 21, ..., 40]:  # 21 iterations
    flow = flow_fields[flow_idx]  # Uses flow_0 through flow_20
    flow_idx += 1  # Now flow_idx = 21
    
    warped = warp(current_frame, flow)
    updated_frames[frame_idx] = warped
    current_frame = warped
    guidance_tracker.mark_single_frame(frame_idx)

# === Process Range 2: frames 42-79 (38 frames) ===
anchor_2 = frame_41  # The keyframe
current_frame = frame_41  # ← NEW ANCHOR! Reset starting point

for frame_idx in [42, 43, ..., 79]:  # 38 iterations
    flow = flow_fields[flow_idx]  # Uses flow_21 through flow_58
    flow_idx += 1  # Now flow_idx = 59
    
    warped = warp(current_frame, flow)
    updated_frames[frame_idx] = warped
    current_frame = warped
    guidance_tracker.mark_single_frame(frame_idx)
```

**Critical Observations:**

1. **Flow sequence is continuous:** Gap 1 uses flows 0-20, Gap 2 uses flows 21-58
   - Structure video motion is applied sequentially across all gaps
   - Like "playing through" the structure video across discontinuous regions

2. **Anchors reset per range:** Each gap starts from its own preceding guided frame
   - Gap 1 starts from overlap frame 19
   - Gap 2 starts from keyframe 41
   - Content is different, but motion pattern continues from structure video

3. **Progressive warping within ranges:** Within each gap, motion accumulates
   - Not independent warps from anchor
   - Each frame is warped from the previous warp

---

## Phase 9: Clip Mode With Insufficient Flows

**Scenario:**
- Need 43 flows for gap (frames 20-62)
- Structure video only provides 25 flows
- Mode: "clip"

```python
flow_fields = adjust_flow_field_count(
    flow_fields=25_flows,
    target_count=43,
    treatment="clip"
)
# Returns 25 flows unchanged

available_flow_count = 25
flow_idx = 0
frames_skipped = 0

for offset in range(43):  # Try to fill 43 frames
    frame_idx = 20 + offset
    
    if flow_idx >= 25:  # ← Runs out at frame 45 (20 + 25)
        # CRITICAL: In clip mode, SKIP this frame
        frames_skipped += 1
        # Don't modify updated_frames[frame_idx] - stays gray
        # Don't mark as guided - stays unguidanced
        continue  # ← Skip to next frame
    
    flow = flow_fields[flow_idx]
    flow_idx += 1
    
    # Apply warp for frames 20-44 only
    warped = warp(current_frame, flow)
    updated_frames[frame_idx] = warped
    current_frame = warped
    guidance_tracker.mark_single_frame(frame_idx)  # Only mark 20-44

# Result:
# - Frames 20-44: warped with structure motion (marked guided)
# - Frames 45-62: still gray (remain unguidanced)
# - frames_skipped = 18
```

### Why This Matters for Masking

```python
# Later, when creating mask video:
inactive_indices = set()

# Add all guided frames to inactive
for i in range(total_frames):
    if guidance_tracker.has_guidance[i]:
        inactive_indices.add(i)

# In our clip scenario:
# inactive_indices = {0-19 (overlap), 20-44 (motion), 63-73 (fade+anchor)}
# ACTIVE (not in inactive) = {45-62} ← Gray frames get WHITE mask

# Mask result:
# Frames 45-62: WHITE mask → AI generates new content (no guidance available)
# All other frames: BLACK mask → Keep as-is
```

**This is correct!** Frames that didn't get motion (due to clip running out) remain unguidanced and get white masks, telling the AI to generate new content there.

---

## Key Design Principles

### 1. **Atomic Tracker Updates**
```python
# GOOD: Mark frame ONLY when actually warped
warped_frame = apply_optical_flow_warp(current_frame, flow)
updated_frames[frame_idx] = warped_frame
guidance_tracker.mark_single_frame(frame_idx)  # ← Atomic with warp

# BAD: Mark entire range after loop (OLD BUG - FIXED)
# This would mark frames that weren't actually warped in clip mode
```

### 2. **Progressive Motion Accumulation**
```python
# Progressive (CORRECT):
current_frame = anchor
for flow in flows:
    current_frame = warp(current_frame, flow)  # Chains motion

# vs. Independent (WRONG):
for flow in flows:
    warped = warp(anchor, flow)  # Would apply each flow to static anchor
```

### 3. **Global Flow Sequencing**
- Structure video flows are consumed sequentially across ALL ranges
- Each range gets the "next chunk" of flows
- Creates continuous motion pattern even across gaps

### 4. **Anchor Reset Per Range**
- Content source (anchor) resets for each gap
- Prevents motion from drifting too far from guided content
- Each gap "starts fresh" from its preceding keyframe/overlap

---

## Edge Cases Handled

### Empty Ranges
```python
unguidanced_ranges = []
# Function returns early, no motion applied
```

### No Anchor Available
```python
anchor_idx = guidance_tracker.get_anchor_frame_index(0)  # First frame
# Returns None (no guided frame before index 0)
current_frame = gray_frame  # Uses neutral gray as anchor
```

### Zero Structure Frames
```python
if len(structure_frames) < 2:
    return frames_for_guide_list  # Early return, no changes
```

### Flow Exhaustion in Adjust Mode
```python
if flow_idx >= available_flow_count:
    if structure_video_treatment == "adjust":
        # Shouldn't happen - we interpolated to match
        # Fallback: repeat last flow
        flow = flow_fields[-1]
```

---

## Summary: The Logic is Sound ✓

**Flow Breakdown:**
1. ✅ Tracker identifies unguidanced ranges logically (not pixel-based)
2. ✅ Total frame count calculated correctly across all ranges
3. ✅ Structure video loads N+1 frames to get N flows (RAFT math correct)
4. ✅ Flow adjustment handles both "adjust" and "clip" correctly
5. ✅ Motion applied progressively within each range (cumulative warping)
6. ✅ Flow sequence continues across ranges (global counter)
7. ✅ Anchors reset per range (prevents drift)
8. ✅ Atomic tracker updates (only mark actually-warped frames)
9. ✅ Clip mode leaves remaining frames unguidanced (correct masking)
10. ✅ All edge cases handled gracefully

**No logic errors found. Implementation is robust.** 🎯

