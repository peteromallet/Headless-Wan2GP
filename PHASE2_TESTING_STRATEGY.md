# Phase 2 Testing Strategy

**Based on Phase 1 Lessons Learned**

## Phase 1 Key Learnings

### What Worked:
1. ✅ **Static validation** caught parameter flow issues early
2. ✅ **Integration testing** caught the critical `wgp.apply_changes()` timing bug
3. ✅ **Real generation testing** proved end-to-end functionality
4. ✅ **Production directory testing** with multiple generations validated consistency

### Critical Bugs Found (Before Production!):
1. Missing `image_save_path` - would have broken all image tasks
2. `wgp.apply_changes()` timing - would have broken EVERYTHING

### Key Insight:
**Testing at multiple levels is essential - each level caught different bugs!**

---

## Phase 2 Testing Levels

Phase 2 is **higher risk** than Phase 1 because:
- Changes affect ALL task types (not just config)
- File paths change for every task
- More moving parts (task handlers, directory mapping, etc.)

Therefore, we need even more thorough testing.

---

## Level 1: Static Validation (5 minutes)

**Goal:** Verify code structure without running anything

**Script:** `test_phase2_static.py`

**Checks:**
- ✅ `_get_task_type_directory()` function exists
- ✅ All expected task types have directory mappings
- ✅ `prepare_output_path()` has `task_type` parameter
- ✅ Parameter is optional (backwards compatibility)
- ✅ Task handlers pass task_type to prepare_output_path()

**Command:**
```bash
python test_phase2_static.py
```

**Expected Output:**
```
✅ All static checks passed
- Task type mapping function found
- All task types mapped
- prepare_output_path accepts task_type
- Backwards compatible (task_type optional)
- Task handlers updated
```

---

## Level 2: Unit Testing (10 minutes)

**Goal:** Test individual functions in isolation

**Script:** `test_phase2_units.py`

**Tests:**

### Test 2.1: Directory Mapping Function
```python
def test_task_type_mapping():
    """Test _get_task_type_directory() for all task types."""
    assert _get_task_type_directory('vace') == 'generation/vace'
    assert _get_task_type_directory('t2v') == 'generation/text_to_video'
    assert _get_task_type_directory('join_clips_orchestrator') == 'orchestrator_runs/join_clips'
    assert _get_task_type_directory('image_inpaint') == 'editing/inpaint'
    assert _get_task_type_directory('extract_frame') == 'specialized/frame_extraction'
    assert _get_task_type_directory('unknown_task') == 'misc'  # default
```

### Test 2.2: prepare_output_path() with task_type
```python
def test_prepare_output_path_with_task_type():
    """Test that prepare_output_path creates correct subdirectories."""
    base_dir = Path("/test/outputs")

    # Test with task_type
    output_dir, _ = prepare_output_path(
        task_id="test_123",
        filename="test.mp4",
        main_output_dir_base=base_dir,
        task_type="vace"
    )

    assert "generation/vace" in str(output_dir)
```

### Test 2.3: Backwards Compatibility
```python
def test_prepare_output_path_without_task_type():
    """Test backwards compatibility - works without task_type."""
    base_dir = Path("/test/outputs")

    # Test without task_type (old behavior)
    output_dir, _ = prepare_output_path(
        task_id="test_123",
        filename="test.mp4",
        main_output_dir_base=base_dir
    )

    # Should save to root, not crash
    assert output_dir == base_dir
```

**Command:**
```bash
python test_phase2_units.py
```

---

## Level 3: Integration Testing (30 minutes)

**Goal:** Test with real WGP initialization but mock tasks

**Script:** `test_phase2_integration.py`

**Tests:**

### Test 3.1: Configuration Persistence
```python
def test_config_with_task_type():
    """Verify task_type flows through to WGP correctly."""
    queue = HeadlessTaskQueue(
        wan_dir="./Wan2GP",
        main_output_dir="./test_phase2_outputs",
        debug_mode=True
    )
    queue.start()

    # Verify base config still works (from Phase 1)
    assert wgp.save_path == "./test_phase2_outputs"
    assert wgp.image_save_path == "./test_phase2_outputs"

    # TODO: Verify task_type parameter flows through
```

### Test 3.2: Directory Creation
```python
def test_subdirectory_creation():
    """Test that subdirectories are created correctly."""
    # Submit mock task with task_type
    # Verify subdirectory exists
    # Verify permissions are correct
```

**Command:**
```bash
python test_phase2_integration.py
```

---

## Level 4: Single Task Type Testing (1 hour)

**Goal:** Validate ONE task type end-to-end before touching others

**Strategy:** Pick a SIMPLE task type first (not orchestrator!)

**Recommended First Task:** `extract_frame`
- Simple (just extracts a frame)
- Fast (no model loading)
- Easy to verify (single output file)

**Script:** `test_phase2_single_task.py`

**Tests:**

### Test 4.1: Extract Frame Task
```python
def test_extract_frame_task():
    """Test that extract_frame saves to specialized/frame_extraction/"""
    queue = HeadlessTaskQueue(
        wan_dir="./Wan2GP",
        main_output_dir="./test_phase2_outputs",
        debug_mode=True
    )
    queue.start()

    task = GenerationTask(
        id="test_extract_frame",
        model="extract_frame",  # or whatever the model key is
        prompt="",  # extract frame doesn't need prompt
        parameters={
            "video_path": "/path/to/test/video.mp4",
            "frame_number": 10
        }
    )

    result = queue.wait_for_completion(task.id, timeout=60)

    assert result["success"]
    output_path = Path(result["output_path"])

    # Verify it's in the correct subdirectory
    assert "specialized/frame_extraction" in str(output_path)
    assert output_path.exists()

    queue.stop()
```

**Manual Verification:**
```bash
ls -la test_phase2_outputs/specialized/frame_extraction/
# Should see extracted frame file
```

**Command:**
```bash
python test_phase2_single_task.py --task-type extract_frame
```

**Once this passes, repeat for other simple tasks:**
- Flux image generation → `generation/flux`
- T2I → `generation/text_to_image`

---

## Level 5: Real Generation Testing (2-3 hours)

**Goal:** Test REAL generations for each task category

**Script:** `test_phase2_real_generations.py`

**Tests:**

### Test 5.1: Generation Tasks
```python
def test_generation_category():
    """Test image/video generation tasks."""
    test_cases = [
        {
            "task_type": "flux",
            "expected_dir": "generation/flux",
            "params": {
                "prompt": "test image",
                "video_length": 1,
                "resolution": "512x512",
                "num_inference_steps": 4,
                "seed": 42
            }
        },
        {
            "task_type": "t2v",
            "expected_dir": "generation/text_to_video",
            "params": {
                "prompt": "test video",
                "video_length": 17,
                "resolution": "512x512",
                "num_inference_steps": 4,
                "seed": 42
            }
        }
    ]

    for test_case in test_cases:
        result = run_generation_test(test_case)
        assert result["success"]
        assert test_case["expected_dir"] in result["output_path"]
```

### Test 5.2: Multiple Generations (Same Task Type)
```python
def test_multiple_generations_same_type():
    """Verify consistency across multiple runs of same task type."""
    for i in range(3):
        result = generate_flux_image(f"test {i}")
        assert "generation/flux" in result["output_path"]
```

### Test 5.3: Multiple Generations (Different Task Types)
```python
def test_multiple_generations_different_types():
    """Test task type switching."""
    # Generate Flux image
    result1 = generate_flux_image("test 1")
    assert "generation/flux" in result1["output_path"]

    # Generate T2V video
    result2 = generate_t2v_video("test 2")
    assert "generation/text_to_video" in result2["output_path"]

    # Generate Flux again
    result3 = generate_flux_image("test 3")
    assert "generation/flux" in result3["output_path"]
```

**Command:**
```bash
python test_phase2_real_generations.py --categories generation
```

**Expected Output:**
```
Testing Generation Category:
  ✓ Flux: outputs/generation/flux/2025-12-17...png
  ✓ T2V: outputs/generation/text_to_video/2025-12-17...mp4
  ✓ Multiple generations consistent
  ✓ Task type switching works

✅ Generation category: PASSED
```

---

## Level 6: Orchestrator Testing (2-3 hours)

**Goal:** Test complex orchestrator workflows

**Why Separate:** Orchestrators are complex (parent/child tasks, stitching, etc.)

**Script:** `test_phase2_orchestrators.py`

**Tests:**

### Test 6.1: Travel Orchestrator
```python
def test_travel_orchestrator():
    """Test travel orchestrator saves to orchestrator_runs/travel/"""
    # Submit travel orchestrator task
    # Verify: outputs/orchestrator_runs/travel/RUNID/
    # Verify: Child segments also in correct location
    # Verify: Final stitched output in correct location
```

### Test 6.2: Join Clips Orchestrator
```python
def test_join_clips_orchestrator():
    """Test join_clips saves to orchestrator_runs/join_clips/"""
    # Submit join_clips task
    # Verify: outputs/orchestrator_runs/join_clips/RUNID/
```

**Command:**
```bash
python test_phase2_orchestrators.py --type travel
python test_phase2_orchestrators.py --type join_clips
```

---

## Level 7: Backwards Compatibility Testing (30 minutes)

**Goal:** Ensure old code still works

**Script:** `test_phase2_backwards_compat.py`

**Tests:**

### Test 7.1: Without task_type Parameter
```python
def test_without_task_type():
    """Test that tasks work without task_type (old behavior)."""
    # Call prepare_output_path() without task_type
    # Should save to root of main_output_dir
    # Should NOT crash or error
```

### Test 7.2: Old Task Handlers (if any remain)
```python
def test_old_task_handlers():
    """Test any task handlers that haven't been updated yet."""
    # If some handlers still don't pass task_type
    # They should work (backwards compatible)
```

**Command:**
```bash
python test_phase2_backwards_compat.py
```

---

## Level 8: Full System Testing (4-6 hours)

**Goal:** Test everything together in production-like environment

**Script:** `test_phase2_full_system.py`

**Tests:**

### Test 8.1: All Task Types
```python
def test_all_task_types():
    """Run one test of EVERY task type."""
    task_types = [
        "flux", "t2v", "vace", "image_inpaint",
        "extract_frame", "travel_orchestrator", "join_clips_orchestrator"
    ]

    for task_type in task_types:
        result = run_task(task_type, get_test_params(task_type))
        verify_output_location(result, task_type)
```

### Test 8.2: Production Load Simulation
```python
def test_production_load():
    """Simulate real production workload."""
    # Submit 10 tasks of mixed types
    # Wait for all to complete
    # Verify all in correct directories
    # Verify no file conflicts
```

**Command:**
```bash
python test_phase2_full_system.py --duration 4h
```

---

## Summary: Testing Order for Phase 2

```
Day 1: Static + Unit + Integration
┌─────────────────────────────────────┐
│ 1. Static Validation (5 min)       │ ← Fast feedback
│ 2. Unit Tests (10 min)             │ ← Test functions in isolation
│ 3. Integration Tests (30 min)      │ ← Test with real WGP
└─────────────────────────────────────┘

Day 2: Single Task Type
┌─────────────────────────────────────┐
│ 4. Single Task (1 hour)            │ ← Validate approach on ONE task
│    - extract_frame first           │
│    - Then flux                     │
│    - Then t2v                      │
└─────────────────────────────────────┘

Day 3: Real Generations
┌─────────────────────────────────────┐
│ 5. Real Generations (2-3 hours)    │ ← Test all generation tasks
│    - Multiple same type            │
│    - Multiple different types      │
│    - Verify directories            │
└─────────────────────────────────────┘

Day 4: Orchestrators + Backwards Compat
┌─────────────────────────────────────┐
│ 6. Orchestrators (2-3 hours)       │ ← Complex workflows
│ 7. Backwards Compat (30 min)       │ ← Old code still works
└─────────────────────────────────────┘

Day 5: Full System
┌─────────────────────────────────────┐
│ 8. Full System (4-6 hours)         │ ← Everything together
│    - All task types                │
│    - Production load               │
│    - Final validation              │
└─────────────────────────────────────┘
```

## Risk Mitigation

### High-Risk Changes:
- Orchestrator task paths (complex workflows)
- Task handler updates (many files to change)
- Directory mapping (affects all tasks)

### Mitigation Strategy:
1. **Incremental Implementation:**
   - Do ONE task type at a time
   - Test thoroughly before moving to next
   - Commit after each task type works

2. **Quick Rollback:**
   - Git commit before each change
   - Tag known-good states
   - Keep Phase 1 as fallback

3. **Monitoring:**
   - Watch error rates after each task type update
   - Monitor file locations in production
   - Quick revert if issues found

## Success Criteria

Phase 2 is ready for production when:

- ✅ All 8 test levels pass
- ✅ All task types save to correct subdirectories
- ✅ Multiple generations consistent
- ✅ Backwards compatibility works
- ✅ No file conflicts or errors
- ✅ Production load test passes
- ✅ Orchestrators work correctly
- ✅ Old code without updates still works

## Confidence Target

**Target Confidence:** 99%+

Achieved through:
- 8 levels of testing
- Real generations for all task types
- Orchestrator workflow validation
- Backwards compatibility verification
- Production load simulation

**Note:** Phase 1 reached 99.5% confidence through similar thorough testing!
