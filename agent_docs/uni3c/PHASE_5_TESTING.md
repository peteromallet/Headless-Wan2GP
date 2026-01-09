# Phase 5: Testing & Validation

← [Back to Start](./STARTING_POINT_AND_STATUS.md) | ← [Phase 4](./PHASE_4_HEADLESS_WIRING.md)

---

## Prerequisites
- All previous phases complete
- Can create tasks through normal headless flow

---

## Quick Start: Run Tests

### 1. Start Worker (on GPU machine with venv)

```bash
source venv/bin/activate && python worker.py \
  --supabase-url https://wczysqzxlwdndgxitrvc.supabase.co \
  --supabase-anon-key eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6IndjenlzcXp4bHdkbmRneGl0cnZjIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTE1MDI4NjgsImV4cCI6MjA2NzA3ODg2OH0.r-4RyHZiDibUjgdgDDM2Vo6x3YpgIO5-BTwfkB2qyYA \
  --supabase-access-token 3HKcoLeJAFFfTFFeRV6Eu7Lq \
  --debug \
  --wgp-profile 5
```

### 2. Create Test Task (from any machine with .env containing SUPABASE_SERVICE_ROLE_KEY)

```bash
# Basic Uni3C test (strength=1.0, guide video controls motion)
python create_test_task.py uni3c_basic

# Baseline without Uni3C (for comparison, same seed)
python create_test_task.py uni3c_baseline

# Strength=0 test (should match baseline)
python create_test_task.py uni3c_strength_test
```

### 3. Monitor Task

```bash
python debug.py task <task_id>
# Look for [UNI3C] logs at each layer
```

### Test Assets (pre-configured in templates)

- **Guide Video**: `https://wczysqzxlwdndgxitrvc.supabase.co/storage/v1/object/public/image_uploads/guidance-videos/onboarding/structure_video_optimized.mp4`
- **Source Image**: `https://wczysqzxlwdndgxitrvc.supabase.co/storage/v1/object/public/image_uploads/8a9fdac5-ed89-482c-aeca-c3dd7922d53c/41V0rWGAaFwJ4Y9AOqcVC.jpg`

---

## Tasks

| Task | Status | Notes |
|------|--------|-------|
| Create test task with sample guide video | 🔴 | |
| Run end-to-end generation | 🔴 | |
| Verify all 6 logging layers show correct values | 🔴 | |
| Human verify: motion from guide appears in output | 🔴 | |
| Test silent-failure detection | 🔴 | Break one layer, confirm logs catch it |
| Test frame policies | 🔴 | fit, trim, loop |
| Test strength variations | 🔴 | 0.0, 0.5, 1.0 |
| Document any gotchas discovered | 🔴 | |

---

## Test 1: Basic End-to-End

### Create Test Task

```python
# Using create_test_task.py or similar
task = {
    "type": "travel_between_images",  # or appropriate type
    "model_type": "wan_2_2_i2v_lightning_baseline_3_3",
    
    # Uni3C params
    "use_uni3c": True,
    "uni3c_guide_video": "/path/to/samples/video.mp4",
    "uni3c_strength": 1.0,
    "uni3c_start_percent": 0.0,
    "uni3c_end_percent": 1.0,
    
    # Standard params
    "source_image": "/path/to/image.png",
    "video_length": 49,
    # ... other params
}
```

### Expected Log Output

```
# Layer 1: task_conversion.py
[UNI3C] Task abc123: use_uni3c=True, guide_video=/path/to/video.mp4, strength=1.0

# Layer 2: task_registry.py
[UNI3C] Task abc123: Uni3C ENABLED
[UNI3C] Task abc123:   guide_video=/path/to/video.mp4
[UNI3C] Task abc123:   strength=1.0
[UNI3C] Task abc123:   start_percent=0.0
[UNI3C] Task abc123:   end_percent=1.0

# Layer 3: wgp.py
[UNI3C] generate_video called with model_type=wan_2_2_i2v_lightning_baseline_3_3
[UNI3C] generate_video: Uni3C ENABLED
[UNI3C]   guide_video: /path/to/video.mp4
[UNI3C]   strength: 1.0
[UNI3C]   step window: 0% - 100%
[UNI3C]   frame_policy: fit
[UNI3C]   keep_on_gpu: False

# Layer 4: any2video.py
[UNI3C] any2video: Loading guide video from /path/to/video.mp4
[UNI3C] any2video: Loaded 120 frames
[UNI3C] any2video: After frame policy 'fit': 49 frames
[UNI3C] any2video: Guide video tensor shape: torch.Size([3, 49, 480, 640])
[UNI3C] any2video: Encoded render_latent shape: torch.Size([1, 20, 13, 60, 80])
[UNI3C] any2video:   Expected channels: 20
[UNI3C] any2video:   Actual channels: 20

# Layer 5: model.py (first step)
[UNI3C] model.forward: Uni3C data present
[UNI3C]   render_latent shape: torch.Size([1, 20, 13, 60, 80])
[UNI3C]   step window: 0% - 100%
[UNI3C] VRAM before controlnet forward: 8.45 GB
[UNI3C] VRAM after controlnet forward: 10.23 GB
[UNI3C] VRAM after offload: 8.52 GB

# Layer 6: model.py (periodic - first, 25%, 50%, 75%, last)
[UNI3C] Step 0/20 (0% of Uni3C window): Applying residual
[UNI3C]   residual shape: torch.Size([1, 4680, 5120])
[UNI3C]   residual mean: 0.012345, std: 0.234567
[UNI3C]   residual min: -0.876543, max: 0.987654
...
[UNI3C] Step 5/20 (25% of Uni3C window): Applying residual
...
[UNI3C] Step 19/20 (100% of Uni3C window): Applying residual
...

# End Summary
[UNI3C] ========== GENERATION COMPLETE ==========
[UNI3C] Uni3C applied to 20 steps × 20 blocks
[UNI3C] Total residual injections: 400
```

### Verification Checklist

- [ ] Layer 1 log appears (task_conversion.py): `[UNI3C] Task X: use_uni3c=True`
- [ ] Layer 2 log appears (task_registry.py): `[UNI3C] Task X: Uni3C ENABLED`
- [ ] Layer 3 log shows model_type + `Uni3C ENABLED` (wgp.py)
- [ ] Layer 4 log shows guide video loading + encoding (any2video.py)
- [ ] Layer 5 log shows `Uni3C data present` + VRAM stats (model.py)
- [ ] Layer 6 logs appear at multiple steps (0%, 25%, 50%, 75%, 100%)
- [ ] Layer 6 residual mean is non-zero and not NaN
- [ ] End summary shows `GENERATION COMPLETE` with step/block counts
- [ ] No OOM errors in VRAM logs
- [ ] Output video exists and is playable
- [ ] Output visually reflects motion from guide video

---

## Test 2: Silent Failure Detection

### Break Layer 1 (param_whitelist)

1. Temporarily remove `use_uni3c` from `param_whitelist` in task_conversion.py
2. Create a task with `use_uni3c=true`
3. **Expected**: Warning log appears:
   ```
   [UNI3C] Task abc123: ⚠️ use_uni3c was in db_task_params but NOT in generation_params!
   ```
4. **Expected**: Layer 3 shows `Uni3C DISABLED`
5. Restore the whitelist

### Break Layer 3 (signature)

1. Temporarily remove `use_uni3c` from `generate_video()` signature in wgp.py
2. Create a task with `use_uni3c=true`
3. **Expected**: Layer 2 shows `Uni3C ENABLED`
4. **Expected**: Layer 3 shows `Uni3C DISABLED (use_uni3c=False)` — the param was dropped!
5. Restore the signature

---

## Test 3: Frame Policies

### Test `fit` (default)

```python
task = {
    "use_uni3c": True,
    "uni3c_guide_video": "video_with_60_frames.mp4",
    "uni3c_frame_policy": "fit",
    "video_length": 49,  # Different from guide
}
```
**Expected**: Guide resampled to 49 frames, log shows `After frame policy 'fit': 49 frames`

### Test `trim`

```python
task = {
    "uni3c_frame_policy": "trim",
    # Guide has 60 frames, target is 49
}
```
**Expected**: Guide trimmed to 49 frames

### Test `loop`

```python
task = {
    "uni3c_frame_policy": "loop",
    # Guide has 30 frames, target is 49
}
```
**Expected**: Guide looped to 49 frames

### Test `off` (strict)

```python
task = {
    "uni3c_frame_policy": "off",
    # Guide has 60 frames, target is 49
}
```
**Expected**: Task fails with error about frame count mismatch

---

## Test 4: Strength Variations

### strength = 0.0

```python
task = {"uni3c_strength": 0.0, ...}
```
**Expected**: Output should be identical to non-Uni3C generation (residuals multiplied by 0)

### strength = 0.5

```python
task = {"uni3c_strength": 0.5, ...}
```
**Expected**: Partial guidance effect, motion less pronounced than strength=1.0

### strength = 1.0 (default)

```python
task = {"uni3c_strength": 1.0, ...}
```
**Expected**: Full guidance effect, motion clearly follows guide

### strength > 1.0

```python
task = {"uni3c_strength": 2.0, ...}
```
**Expected**: May work but could produce artifacts. Document behavior.

---

## Test 5: Step Window

### Early only (0% - 50%)

```python
task = {
    "uni3c_start_percent": 0.0,
    "uni3c_end_percent": 0.5,
}
```
**Expected**: Log shows `Uni3C DEACTIVATED` at 50% of steps

### Late only (50% - 100%)

```python
task = {
    "uni3c_start_percent": 0.5,
    "uni3c_end_percent": 1.0,
}
```
**Expected**: Log shows `Uni3C ACTIVE` starting at step ~50%

---

## Definition of Done Verification

Run through the Definition of Done checklist in [STARTING_POINT_AND_STATUS.md](./STARTING_POINT_AND_STATUS.md) (Ctrl+F "Definition of Done"):

### Must Have

| Criterion | Test | Status |
|-----------|------|--------|
| Task with `use_uni3c=true` produces different output | Compare with/without | ☐ |
| Guide video motion reflected in output | Human verification | ☐ |
| `[UNI3C]` logs at all 6 layers | Check logs | ☐ |
| Silent param drop triggers warning | Break Layer 1 test | ☐ |
| Works with `wan_2_2_i2v_lightning_baseline_3_3` | Use this preset | ☐ |

### Should Have

| Criterion | Test | Status |
|-----------|------|--------|
| `strength=0` identical to no Uni3C | Compare outputs | ☐ |
| Step window gates correctly | Early/late tests | ☐ |
| Frame policy works | fit/trim/loop tests | ☐ |

---

## Debug Commands

Add to `debug.py` for quick diagnosis:

```python
# python debug.py uni3c <task_id>
def diagnose_uni3c(task_id: str):
    """Check if Uni3C params flowed through correctly for a task."""
    task = db_ops.get_task(task_id)
    params = task.get("params", {})
    
    print(f"[UNI3C_DIAG] Task {task_id}")
    print(f"  use_uni3c in params: {'use_uni3c' in params}")
    
    if 'use_uni3c' in params:
        print(f"    value: {params['use_uni3c']}")
        print(f"    guide_video: {params.get('uni3c_guide_video', 'NOT_SET')}")
        print(f"    strength: {params.get('uni3c_strength', 'NOT_SET')}")
        print(f"    frame_policy: {params.get('uni3c_frame_policy', 'NOT_SET')}")
    
    # Check task logs for [UNI3C] entries
    print(f"\n  Checking logs for [UNI3C] entries...")
    # Implementation depends on your logging setup
```

---

## Document Gotchas

After testing, update this section with any issues discovered:

### Known Issues

_(To be filled in during testing)_

### Workarounds

_(To be filled in during testing)_

---

## Completion

When all tests pass:

1. Update [STARTING_POINT_AND_STATUS.md](./STARTING_POINT_AND_STATUS.md):
   - Set Phase 5 status to ✅
   - Set Overall Status to ✅ Done
   - Check all Definition of Done boxes
   - Update "Last Updated" date

2. Commit with message:
   ```
   [UNI3C] Phase 5: Testing complete - Uni3C integration done
   
   - All Definition of Done criteria verified
   - Silent failure detection confirmed working
   - Frame policies tested
   - Strength variations documented
   ```

🎉 **Congratulations!** Uni3C integration is complete.

