# Detecting Identical Generation Parameters

This document explains how to detect when all LoRAs and prompts in a batch generation are identical, enabling optimizations like model caching and efficient frame allocation.

## Best Approach: Orchestrator Level Detection

**Location**: `source/sm_functions/travel_between_images.py:280-443`
**Why**: Catches identical parameters early, before task creation, enabling both model optimization and improved frame allocation strategies.

```python
def detect_identical_parameters(orchestrator_payload, num_segments, dprint=None):
    """
    Detect if all segments will have identical generation parameters.
    Returns analysis that enables both model caching and frame optimization.
    """
    # Extract parameter arrays
    expanded_base_prompts = orchestrator_payload["base_prompts_expanded"]
    expanded_negative_prompts = orchestrator_payload["negative_prompts_expanded"]
    additional_loras = orchestrator_payload.get("additional_loras", [])

    # Check parameter identity
    prompts_identical = len(set(expanded_base_prompts)) == 1
    negative_prompts_identical = len(set(expanded_negative_prompts)) == 1

    # LoRA consistency check
    lora_flags_consistent = all([
        orchestrator_payload.get("apply_causvid", False),
        orchestrator_payload.get("use_lighti2x_lora", False),
        orchestrator_payload.get("apply_reward_lora", False)
    ])

    is_identical = prompts_identical and negative_prompts_identical

    if dprint and is_identical:
        dprint(f"[IDENTICAL_DETECTION] All {num_segments} segments identical - enabling optimizations")
        dprint(f"  - Unique prompt: '{expanded_base_prompts[0][:50]}...'")
        dprint(f"  - LoRA count: {len(additional_loras)}")

    return {
        "is_identical": is_identical,
        "can_optimize_frames": is_identical,  # Key for frame allocation optimization
        "can_reuse_model": is_identical,      # Key for model caching
        "unique_prompt": expanded_base_prompts[0] if prompts_identical else None
    }
```

## Integration Point

Add this check in the orchestrator around **line 200**, after quantization but before segment creation:

```python
# After frame quantization logic
identity_analysis = detect_identical_parameters(orchestrator_payload, num_segments, dprint)

if identity_analysis["can_optimize_frames"]:
    # Enable frame consolidation logic (see frame allocation section)
    orchestrator_payload["_enable_frame_consolidation"] = True

if identity_analysis["can_reuse_model"]:
    # Enable model caching optimizations
    orchestrator_payload["_enable_model_reuse"] = True
```

This approach provides the foundation for both model optimization and the frame allocation improvements described below.

## Frame Consolidation Optimization

**CRITICAL REQUIREMENT**: This optimization ONLY triggers when ALL prompts AND LoRAs are identical across segments. This is essential because the consolidation changes the fundamental segment structure that guide/mask creation depends on.

### Current Frame Allocation
The current system creates one segment per keyframe transition:
```
Keyframes at: [0, 24, 65, 82]
Current allocation:
- Segment 0: frames 0-24 (24 frames)
- Segment 1: frames 24-65 (41 frames)
- Segment 2: frames 65-82 (17 frames)
```

### Optimized Frame Consolidation
With identical parameters, we can fit multiple keyframes in one segment:

**Example 1: All keyframes fit**
```
Keyframes at: [0, 24, 65, 82]
Optimized allocation (max 65 frames):
- Segment 0: frames 0-65 (includes keyframes 0, 24, 65) - 65 frames
- Segment 1: frames 65-82 (continues to final keyframe 82) - 17 frames
```

**Example 2: Automatic segment boundary (your case)**
```
Keyframes at: [0, 54, 91]
With 65-frame limit:
- Segment 0: frames 0-54 (includes keyframes 0, 54) - 54 frames
- Segment 1: frames 54-91 (includes keyframes 54, 91) - 37 frames

Algorithm walkthrough:
1. Start at keyframe 0, segment_start = 0
2. Check if keyframe 54 fits: 54 - 0 = 54 frames ≤ 65 ✅ → Include keyframe 54
3. Check if keyframe 91 fits: 91 - 0 = 91 frames > 65 ❌ → Stop, create boundary
4. Segment 0: 0→54 (54 frames, includes keyframes 0, 54)
5. Start segment 1 at frame 54
6. Check remaining keyframe 91: 91 - 54 = 37 frames ≤ 65 ✅ → Include keyframe 91
7. Segment 1: 54→91 (37 frames, includes keyframes 54, 91)
```

### Implementation

Add this function to the orchestrator after parameter identity detection:

```python
def optimize_frame_allocation_for_identical_params(orchestrator_payload, max_frames_per_segment=65, dprint=None):
    """
    When all parameters are identical, consolidate keyframes into fewer segments.

    Args:
        orchestrator_payload: Original orchestrator data
        max_frames_per_segment: Maximum frames per segment (model technical limit)
        dprint: Debug logging function

    Returns:
        Updated orchestrator_payload with optimized frame allocation
    """
    original_segment_frames = orchestrator_payload["segment_frames_expanded"]
    original_frame_overlaps = orchestrator_payload["frame_overlap_expanded"]
    original_base_prompts = orchestrator_payload["base_prompts_expanded"]

    if dprint:
        dprint(f"[FRAME_CONSOLIDATION] Original allocation: {len(original_segment_frames)} segments")
        dprint(f"  - Segment frames: {original_segment_frames}")
        dprint(f"  - Frame overlaps: {original_frame_overlaps}")

    # Calculate cumulative keyframe positions
    keyframe_positions = [0]  # Start with frame 0
    cumulative_pos = 0

    for i, segment_frames in enumerate(original_segment_frames):
        if i < len(original_frame_overlaps):
            overlap = original_frame_overlaps[i]
            cumulative_pos += segment_frames - overlap
        else:
            cumulative_pos += segment_frames
        keyframe_positions.append(cumulative_pos)

    if dprint:
        dprint(f"[FRAME_CONSOLIDATION] Keyframe positions: {keyframe_positions}")

    # Consolidate keyframes into optimized segments
    optimized_segments = []
    optimized_overlaps = []
    optimized_prompts = []

    current_segment_start = 0
    keyframe_idx = 0

    while keyframe_idx < len(keyframe_positions) - 1:
        # Find how many keyframes we can fit in this segment
        segment_end = current_segment_start
        keyframes_in_segment = []

        for next_keyframe_idx in range(keyframe_idx + 1, len(keyframe_positions)):
            potential_end = keyframe_positions[next_keyframe_idx]
            segment_length = potential_end - current_segment_start

            if segment_length <= max_frames_per_segment:
                segment_end = potential_end
                keyframes_in_segment.append(keyframe_positions[next_keyframe_idx])
                dprint(f"[CONSOLIDATION_LOGIC] Keyframe {potential_end} fits in segment (length: {segment_length} <= {max_frames_per_segment})")
            else:
                dprint(f"[CONSOLIDATION_LOGIC] Keyframe {potential_end} exceeds frame limit (length: {segment_length} > {max_frames_per_segment}) - creating segment boundary")
                break

        # If we couldn't fit any keyframes, take at least one
        if not keyframes_in_segment:
            keyframes_in_segment = [keyframe_positions[keyframe_idx + 1]]
            segment_end = keyframes_in_segment[0]

        segment_length = segment_end - current_segment_start
        optimized_segments.append(segment_length)

        # Use the first prompt (they're all identical anyway)
        optimized_prompts.append(original_base_prompts[0])

        # Add overlap for next segment (if there is one)
        if segment_end < keyframe_positions[-1]:
            # Calculate reasonable overlap based on segment length
            overlap = min(8, segment_length // 4)  # Max 8 frames or 25% of segment
            overlap = (overlap // 2) * 2  # Make even for quantization
            optimized_overlaps.append(overlap)

        if dprint:
            dprint(f"[FRAME_CONSOLIDATION] Segment {len(optimized_segments)-1}: frames {current_segment_start}-{segment_end} ({segment_length} frames)")
            dprint(f"  - Includes keyframes: {keyframes_in_segment}")

        # Move to next segment
        keyframe_idx = len([kf for kf in keyframe_positions if kf <= segment_end]) - 1
        current_segment_start = segment_end

    # Update orchestrator payload
    orchestrator_payload["segment_frames_expanded"] = optimized_segments
    orchestrator_payload["frame_overlap_expanded"] = optimized_overlaps
    orchestrator_payload["base_prompts_expanded"] = optimized_prompts
    orchestrator_payload["negative_prompts_expanded"] = [orchestrator_payload["negative_prompts_expanded"][0]] * len(optimized_segments)
    orchestrator_payload["num_new_segments_to_generate"] = len(optimized_segments)

    if dprint:
        dprint(f"[FRAME_CONSOLIDATION] Optimized to {len(optimized_segments)} segments (was {len(original_segment_frames)})")
        dprint(f"  - New segment frames: {optimized_segments}")
        dprint(f"  - New overlaps: {optimized_overlaps}")
        dprint(f"  - Efficiency: {(len(original_segment_frames) - len(optimized_segments))} fewer segments")

    return orchestrator_payload

# Usage in orchestrator after identity detection
if identity_analysis["can_optimize_frames"]:
    orchestrator_payload = optimize_frame_allocation_for_identical_params(
        orchestrator_payload,
        max_frames_per_segment=65,  # Model technical limit
        dprint=dprint
    )
```

### Benefits

1. **Fewer GPU Model Loads**: Reduced segments mean fewer model initialization cycles
2. **Better Memory Utilization**: Larger segments use GPU memory more efficiently
3. **Reduced Task Queue Overhead**: Fewer database operations and task scheduling
4. **Improved Generation Speed**: Less time spent on setup between segments

### Example Results

Original: `[0,24,65,82]` → 3 segments of 24, 41, 17 frames
Optimized: `[0,24,65,82]` → 2 segments of 65, 17 frames

This optimization works particularly well for:
- Static scene descriptions with identical prompts
- Style transfer videos with consistent LoRA applications
- Long journeys with repetitive generation parameters

## Compatibility Analysis: Guide Video and Mask Creation

### ✅ **SAFE - Guide Video Creation**
The frame consolidation is **fully compatible** with guide video creation:

**Why it works:**
- Guide video uses `total_frames_for_segment` parameter (line 610 in video_utils.py)
- Each consolidated segment still gets its own guide video with the correct frame count
- Start/end anchors work correctly - consolidation preserves keyframe positions in the guide
- Previous segment video linking remains intact - each segment references the previous segment's output

**Key insight**: Guide creation doesn't care about prompt/LoRA identity - it only needs:
- Frame count for the segment (`total_frames_for_segment`)
- Start anchor image (first keyframe in segment)
- End anchor image (last keyframe in segment)
- Previous segment video (for overlap frames)

All these remain valid after consolidation.

### ✅ **SAFE - Mask Video Creation**
The frame consolidation is **fully compatible** with mask video creation:

**Why it works:**
- Mask creation is based on frame indices within each segment (lines 176-210 in travel_segment_processor.py)
- **Overlap frames**: Still marked as inactive (black) at start of each segment
- **Anchor frames**: Still marked as inactive at segment start/end based on keyframe positions
- **Frame-level masking**: Works identically regardless of how many keyframes are in the segment

**Key insight**: Mask creation operates on **per-segment frame indices**, not global keyframe positions.

### ⚠️ **CRITICAL CONSTRAINT: Identical Parameters Only**

The optimization is **ONLY safe** when ALL prompts and LoRAs are identical because:

1. **Different prompts** would break semantic continuity within consolidated segments
2. **Different LoRAs** would create visual inconsistencies within single videos
3. **Guide video anchoring** assumes consistent style between start/end keyframes
4. **Mask frame logic** assumes uniform generation parameters across the segment

### Implementation Safety Check

Add this validation to the consolidation function:

```python
def validate_consolidation_safety(orchestrator_payload, dprint=None):
    """
    Verify that frame consolidation is safe by checking parameter identity.
    """
    # Get parameter arrays
    prompts = orchestrator_payload["base_prompts_expanded"]
    neg_prompts = orchestrator_payload["negative_prompts_expanded"]
    additional_loras = orchestrator_payload.get("additional_loras", [])

    # Critical safety checks
    all_prompts_identical = len(set(prompts)) == 1
    all_neg_prompts_identical = len(set(neg_prompts)) == 1

    # LoRA consistency (flags should be uniform if prompts are identical)
    lora_flags = [
        orchestrator_payload.get("apply_causvid", False),
        orchestrator_payload.get("use_lighti2x_lora", False),
        orchestrator_payload.get("apply_reward_lora", False)
    ]

    is_safe = all_prompts_identical and all_neg_prompts_identical

    if dprint:
        if is_safe:
            dprint(f"[CONSOLIDATION_SAFETY] ✅ Safe to consolidate - all parameters identical")
        else:
            dprint(f"[CONSOLIDATION_SAFETY] ❌ NOT safe to consolidate:")
            if not all_prompts_identical:
                dprint(f"  - Prompts differ: {len(set(prompts))} unique prompts")
            if not all_neg_prompts_identical:
                dprint(f"  - Negative prompts differ: {len(set(neg_prompts))} unique")

    return {
        "is_safe": is_safe,
        "prompts_identical": all_prompts_identical,
        "negative_prompts_identical": all_neg_prompts_identical,
        "can_consolidate": is_safe
    }

# Use in optimization function
safety_check = validate_consolidation_safety(orchestrator_payload, dprint)
if not safety_check["is_safe"]:
    if dprint:
        dprint(f"[FRAME_CONSOLIDATION] Skipping optimization - parameters not identical")
    return orchestrator_payload  # Return unchanged
```

This ensures the optimization only triggers when it's genuinely safe and beneficial.

## Failure Handling Analysis

### ❌ **CRITICAL CONCERN: Segment Failure Impact**

The frame consolidation optimization has a significant **failure amplification risk**:

**Current System (without consolidation):**
- Keyframes `[0, 24, 65, 82]` → 3 segments: 24, 41, 17 frames
- If segment 1 fails, segments 2 and 3 are still attempted
- **1 failed segment = 41 lost frames**

**With Frame Consolidation:**
- Keyframes `[0, 24, 65, 82]` → 2 segments: 65, 17 frames
- If segment 1 fails, segment 2 can still proceed
- **1 failed segment = 65 lost frames** (58% more loss)

### **Failure Propagation Logic**

Based on the worker.py analysis (lines 1468-1476):

```python
# When a task fails
if not task_succeeded:
    db_ops.update_task_status(task_id, STATUS_FAILED, error_message)
    # Task is marked as failed and processing continues to next task
```

**Key findings:**
1. **No automatic retry** - Failed tasks stay failed
2. **Dependency blocking** - Tasks with `dependant_on` pointing to failed tasks cannot proceed
3. **Manual intervention required** - Failed segments must be manually restarted or skipped

### **Impact on Consolidated Segments**

**Scenario: Consolidated segment with 3 keyframes fails**
- **Lost content**: All 3 keyframes worth of video generation
- **Downstream impact**: Next segment waits indefinitely for the failed segment
- **Recovery cost**: Must regenerate larger chunk of content

**Trade-off: Efficiency vs Failure Impact**

The consolidation algorithm maximizes efficiency by fitting as many keyframes as possible within the technical frame limit:

```python
# Usage - pure efficiency optimization
orchestrator_payload = optimize_frame_allocation_for_identical_params(
    orchestrator_payload,
    max_frames_per_segment=65,  # Model technical limit only
    dprint=dprint
)
```

**Benefits:**
- **Maximum consolidation** - Limited only by model capabilities, not artificial constraints
- **Optimal GPU utilization** - Larger segments use memory more efficiently
- **Minimal task overhead** - Fewest possible segments for the given keyframes

**Trade-off awareness:**
- Larger consolidated segments mean more content is lost if a segment fails
- This is acceptable when prioritizing efficiency over failure resilience
- The identical parameter constraint already ensures consolidation is safe from a generation quality perspective