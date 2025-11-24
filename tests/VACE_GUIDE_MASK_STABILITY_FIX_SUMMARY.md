# VACE Guide/Mask Stability Fix Summary

## Problem Statement

The join clips guide/mask creation was potentially unstable due to several issues:

1. **Frame count mismatches** between guide video, mask video, and VACE generation parameters
2. **VACE quantization** (4n+1 requirement) not properly reflected in mask generation
3. **Replace mode boundary clamping** - oversized gaps could request more frames than available context, causing negative preservation windows
4. **No validation** of actual guide frame counts vs theoretical calculations

## Root Causes

### 1. Inconsistent Frame Count Tracking
- **Issue**: `vace_frame_utils.py` used `total_frames` parameter but didn't validate it against actual guide construction
- **Impact**: Guide, mask, and VACE `video_length` could become misaligned, especially after quantization

### 2. Replace Mode Edge Cases
- **Issue**: When `gap_frame_count` exceeded available context (e.g., gap=20 but only 8+8=16 context frames), the code would create 20 gray frames anyway
- **Impact**: Guide/mask mismatch, unexpected frame counts, potential generation failures

### 3. Variable Scope Issues
- **Issue**: Preservation counts calculated in guide building weren't available for mask generation
- **Impact**: Mask generation couldn't correctly mark preserved vs regenerated regions

## Solutions Implemented

### 1. Authoritative Guide Frame Count
**File**: `source/vace_frame_utils.py`

- Guide construction now returns actual frame count: `(guide_video, mask_video, total_frames)`
- Mask generation uses this authoritative count
- Warning logged when theoretical vs actual counts differ

```python
guide_frame_count = len(guide_frames)
if total_frames is None:
    total_frames = guide_frame_count
elif total_frames != guide_frame_count:
    dprint(f"[VACE_UTILS] Task {task_id}: total_frames override ({total_frames}) "
           f"does not match constructed guide ({guide_frame_count}). Using guide frame count.")
    total_frames = guide_frame_count

return created_guide_video, created_mask_video, total_frames
```

### 2. Replace Mode Gap Clamping
**File**: `source/vace_frame_utils.py`

- Clamp `actual_gap_count` to available boundary frames
- Prevent negative preservation windows
- Log warning when clamping occurs

```python
# Clamp replacement counts to available context
max_replaceable_before = min(frames_to_replace_from_before, num_context_before)
max_replaceable_after = min(frames_to_replace_from_after, num_context_after)

# If gap exceeds available context, clamp it
if frames_to_replace_from_before > num_context_before or frames_to_replace_from_after > num_context_after:
    actual_gap_count = max_replaceable_before + max_replaceable_after
    dprint(f"[VACE_UTILS]   Warning: Requested gap ({gap_frame_count}) exceeds available "
           f"boundary frames. Clamping to {actual_gap_count} frames.")
```

### 3. Consistent Frame Count Propagation
**Files**: `source/sm_functions/join_clips.py`, `source/sm_functions/inpaint_frames.py`

- Both callers now capture the returned frame count
- Use actual count instead of theoretical quantized count
- Pass correct value to `prepare_vace_generation_params()`

```python
# join_clips.py
created_guide_video, created_mask_video, guide_frame_count = create_guide_and_mask_for_generation(...)
if guide_frame_count != quantized_total_frames:
    dprint(f"[JOIN_CLIPS] Task {task_id}: Guide/mask total frame count ({guide_frame_count}) "
           f"differs from quantized expectation ({quantized_total_frames}). Using actual count.")
total_frames = guide_frame_count
```

### 4. Centralized Variable Initialization
**File**: `source/vace_frame_utils.py`

- Preservation variables initialized at function scope
- Available throughout guide and mask generation
- Prevents scope-related bugs

```python
# Initialize preservation variables for use throughout function
num_preserved_before = 0
num_preserved_after = 0
actual_gap_count = gap_frame_count
max_replaceable_before = 0
max_replaceable_after = 0
```

## Test Coverage

Created comprehensive test suite: `tests/test_vace_guide_mask_stability.py`

### Test Cases (All Passing)
1. ✅ **Basic INSERT Mode** - Simple gap insertion
2. ✅ **INSERT Mode with Anchor Regeneration** - Regenerate boundary frames for smoothness
3. ✅ **Basic REPLACE Mode** - Replace boundary frames instead of inserting
4. ✅ **REPLACE Mode with Oversized Gap** - Clamping behavior (edge case)
5. ✅ **VACE Quantization Alignment** - Frame counts match 4n+1 requirement
6. ✅ **Gap Inserted Frames** - keep_bridging_images feature
7. ✅ **Minimal Context Edge Case** - 1 frame before/after
8. ✅ **Multiple Resolution Support** - 320x192 to 1920x1080

## Benefits

### Stability Improvements
- **Eliminated frame count mismatches** between guide, mask, and VACE parameters
- **Prevented crashes** from negative preservation windows
- **Graceful degradation** when gap exceeds available context

### Debugging Enhancements
- **Clear warnings** when clamping occurs
- **Detailed logging** of preservation vs regeneration regions
- **Frame count validation** at each step

### Maintainability
- **Single source of truth** for frame counts (guide construction)
- **Centralized variable management** prevents scope bugs
- **Comprehensive test coverage** prevents regressions

## Files Modified

1. **source/vace_frame_utils.py** - Core guide/mask creation logic
   - Added gap clamping for replace mode
   - Return actual frame count from guide construction
   - Centralized variable initialization

2. **source/sm_functions/join_clips.py** - Join clips task handler
   - Capture returned frame count
   - Use actual count instead of theoretical
   - Add validation logging

3. **source/sm_functions/inpaint_frames.py** - Frame inpainting task handler
   - Same updates as join_clips.py
   - Consistent frame count handling

4. **tests/test_vace_guide_mask_stability.py** - New comprehensive test suite
   - 8 test cases covering edge cases
   - Video property validation
   - Frame count verification

## Running the Tests

```bash
cd /Users/peteromalley/Documents/Headless-Wan2GP
python tests/test_vace_guide_mask_stability.py
```

Expected output: **8/8 tests passed**

## Next Steps

1. **Production Testing**: Run join_clips tasks with various configurations
2. **Monitor Logs**: Watch for clamping warnings in production
3. **Performance**: Verify no regression in generation speed
4. **Edge Cases**: Test with extreme aspect ratios and frame counts

## Related Issues

- Fixes instability in join clips guide/mask creation
- Addresses VACE quantization edge cases
- Prevents frame count mismatches in replace mode
- Improves error messages for debugging

