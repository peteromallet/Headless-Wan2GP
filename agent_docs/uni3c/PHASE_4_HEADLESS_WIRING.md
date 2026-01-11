# Phase 4: Headless Param Wiring

← [Back to Start](./STARTING_POINT_AND_STATUS.md) | ← [Phase 3](./PHASE_3_MODEL_INTEGRATION.md)

---

## Prerequisites
- Phase 3 complete (Model integration works with hardcoded uni3c_data)

## Tasks

| Task | Status | Notes |
|------|--------|-------|
| Add Uni3C params to `task_conversion.py:param_whitelist` | ✅ Done | Added 7 params to whitelist |
| Add Layer 1 logging in `task_conversion.py` | ✅ Done | Detects whitelist failures with warning log |
| Add Uni3C resolution block in `task_registry.py` | ✅ Done | Like SVI block, with URL download support |
| Add Layer 2 logging in `task_registry.py` | ✅ Done | Logs all Uni3C params |
| Add params to `wgp.py:generate_video()` signature | ✅ Done | 7 params with defaults |
| Add Layer 3 logging in `wgp.py` | ✅ Done | Logs ENABLED/DISABLED status |
| Pass params through to `any2video.generate()` | ✅ Done | All 7 params passed via `wan_model.generate()` |

---

## ⚠️ CRITICAL: Why This Phase Has Silent Failures

Uni3C params can be **silently dropped** at TWO places:

1. **`task_conversion.py:param_whitelist`**: Only whitelisted params flow from DB to generation_params
2. **`wgp.py:generate_video()` signature**: Wan2GP uses `inspect.signature()` to filter unknown params

If you forget either one, the task will run **without Uni3C** and produce non-guided output. No errors. No crashes. Just wrong results.

**This is why we need logging at every layer.**

---

## What You Need to Know

### The Param Flow

```
DB Task Params
     ↓
task_conversion.py (param_whitelist filter) ← Layer 1
     ↓
task_registry.py (resolution + injection) ← Layer 2
     ↓
headless_wgp.py (merge with defaults)
     ↓
wgp.py generate_video() (signature filter) ← Layer 3
     ↓
any2video.py generate() ← Layer 4
     ↓
model.py forward() ← Layer 5, 6
```

→ See [Parameter Definitions](_reference/PARAM_DEFINITIONS.md) for the full parameter table.

---

## Code: `task_conversion.py` (Layer 1)

### Add to `param_whitelist`

```python
# In source/task_conversion.py, add to param_whitelist set:
param_whitelist = {
    # ... existing params ...
    
    # Uni3C motion guidance parameters
    "use_uni3c",
    "uni3c_guide_video",
    "uni3c_strength",
    "uni3c_start_percent",
    "uni3c_end_percent",
    "uni3c_keep_on_gpu",
    "uni3c_frame_policy",
}
```

### Add Layer 1 Logging

```python
# After param_whitelist extraction, add:
if "use_uni3c" in generation_params:
    headless_logger.info(
        f"[UNI3C] Task {task_id}: use_uni3c={generation_params.get('use_uni3c')}, "
        f"guide_video={generation_params.get('uni3c_guide_video', 'NOT_SET')}, "
        f"strength={generation_params.get('uni3c_strength', 'NOT_SET')}"
    )
else:
    # IMPORTANT: Detect when whitelist is missing the param
    if db_task_params.get("use_uni3c"):
        headless_logger.warning(
            f"[UNI3C] Task {task_id}: ⚠️ use_uni3c was in db_task_params but NOT in generation_params! "
            f"Check param_whitelist in task_conversion.py"
        )
```

---

## Code: `task_registry.py` (Layer 2)

Add Uni3C resolution block similar to SVI. **These utilities already exist in the codebase:**
- `_get_param()` → defined at `source/task_registry.py:67`
- `dprint_func` → passed into handler functions (from `make_task_dprint()`)
- `download_file()` → `source/common_utils.py:1190`

```python
# In _handle_travel_segment_via_queue or similar function
# (see existing SVI handling around line 220 for the pattern)

from source.common_utils import download_file

# Resolve use_uni3c with explicit-false override semantics
use_uni3c = segment_params.get("use_uni3c")
if use_uni3c is None:
    use_uni3c = orchestrator_details.get("use_uni3c", False)

if use_uni3c:
    # Resolve other Uni3C params (using existing _get_param helper)
    uni3c_guide = _get_param("uni3c_guide_video", segment_params, orchestrator_details, prefer_truthy=True)
    uni3c_strength = _get_param("uni3c_strength", segment_params, orchestrator_details, default=1.0)
    uni3c_start = _get_param("uni3c_start_percent", segment_params, orchestrator_details, default=0.0)
    uni3c_end = _get_param("uni3c_end_percent", segment_params, orchestrator_details, default=1.0)
    uni3c_keep_gpu = _get_param("uni3c_keep_on_gpu", segment_params, orchestrator_details, default=False)
    uni3c_frame_policy = _get_param("uni3c_frame_policy", segment_params, orchestrator_details, default="fit")
    
    # Download guide video if URL (using existing download_file helper)
    if uni3c_guide and uni3c_guide.startswith(("http://", "https://")):
        local_filename = Path(uni3c_guide).name or "guide_video.mp4"
        download_file(uni3c_guide, segment_processing_dir, local_filename)
        uni3c_guide = str(segment_processing_dir / local_filename)
    
    # Layer 2 logging (dprint_func is passed into handler)
    dprint_func(f"[UNI3C] Task {task_id}: Uni3C ENABLED")
    dprint_func(f"[UNI3C] Task {task_id}:   guide_video={uni3c_guide}")
    dprint_func(f"[UNI3C] Task {task_id}:   strength={uni3c_strength}")
    dprint_func(f"[UNI3C] Task {task_id}:   start_percent={uni3c_start}")
    dprint_func(f"[UNI3C] Task {task_id}:   end_percent={uni3c_end}")
    
    # Inject into generation_params
    generation_params["use_uni3c"] = True
    generation_params["uni3c_guide_video"] = uni3c_guide
    generation_params["uni3c_strength"] = uni3c_strength
    generation_params["uni3c_start_percent"] = uni3c_start
    generation_params["uni3c_end_percent"] = uni3c_end
    generation_params["uni3c_keep_on_gpu"] = uni3c_keep_gpu
    generation_params["uni3c_frame_policy"] = uni3c_frame_policy
```

---

## Code: `wgp.py` (Layer 3)

### Add to `generate_video()` Signature

```python
def generate_video(
    # ... existing params ...
    
    # Uni3C motion guidance
    use_uni3c: bool = False,
    uni3c_guide_video: str = None,
    uni3c_strength: float = 1.0,
    uni3c_start_percent: float = 0.0,
    uni3c_end_percent: float = 1.0,
    uni3c_keep_on_gpu: bool = False,
    uni3c_frame_policy: str = "fit",
):
    # Layer 3 logging - immediately at function entry
    # Include model_type for context (helps debug preset-specific issues)
    print(f"[UNI3C] generate_video called with model_type={model_type}")
    
    if use_uni3c:
        print(f"[UNI3C] generate_video: Uni3C ENABLED")
        print(f"[UNI3C]   guide_video: {uni3c_guide_video}")
        print(f"[UNI3C]   strength: {uni3c_strength}")
        print(f"[UNI3C]   step window: {uni3c_start_percent*100:.0f}% - {uni3c_end_percent*100:.0f}%")
        print(f"[UNI3C]   frame_policy: {uni3c_frame_policy}")
        print(f"[UNI3C]   keep_on_gpu: {uni3c_keep_on_gpu}")
    else:
        # Log when NOT using Uni3C (helps detect silent drops)
        print(f"[UNI3C] generate_video: Uni3C DISABLED (use_uni3c={use_uni3c})")
    
    # ... rest of function ...
```

### Pass to `wan_model.generate()`

```python
# Where generate_video calls the model
result = wan_model.generate(
    # ... existing params ...
    
    # Pass Uni3C params
    use_uni3c=use_uni3c,
    uni3c_guide_video=uni3c_guide_video,
    uni3c_strength=uni3c_strength,
    uni3c_start_percent=uni3c_start_percent,
    uni3c_end_percent=uni3c_end_percent,
    uni3c_keep_on_gpu=uni3c_keep_on_gpu,
    uni3c_frame_policy=uni3c_frame_policy,
)
```

---

## Code: `any2video.py` (Layer 4)

### Add to `generate()` Signature

```python
def generate(
    self,
    # ... existing params ...
    
    # Uni3C motion guidance
    use_uni3c: bool = False,
    uni3c_guide_video: str = None,
    uni3c_strength: float = 1.0,
    uni3c_start_percent: float = 0.0,
    uni3c_end_percent: float = 1.0,
    uni3c_keep_on_gpu: bool = False,
    uni3c_frame_policy: str = "fit",
):
```

### Build `uni3c_data` Dict

```python
# Inside generate(), after computing target dimensions
uni3c_data = None
if use_uni3c and uni3c_guide_video:
    print(f"[UNI3C] any2video: Loading guide video from {uni3c_guide_video}")
    
    # Load and encode guide video (from Phase 2)
    guide_tensor = self._load_uni3c_guide_video(
        uni3c_guide_video, 
        target_height, 
        target_width, 
        target_frames,
        uni3c_frame_policy
    )
    
    # Get expected channels from checkpoint config
    expected_channels = self.uni3c_config.get("in_channels", 20)
    render_latent = self._encode_uni3c_guide(guide_tensor, expected_channels)
    
    print(f"[UNI3C] any2video: Encoded render_latent shape: {render_latent.shape}")
    
    # Build uni3c_data dict
    uni3c_data = {
        "controlnet": self.uni3c_controlnet,
        "controlnet_weight": uni3c_strength,
        "start": uni3c_start_percent,
        "end": uni3c_end_percent,
        "render_latent": render_latent,
        "render_mask": None,
        "camera_embedding": None,
        "offload": not uni3c_keep_on_gpu,
    }
```

---

## Failure Mode Detection

| Layer | What to Check | Failure Symptom |
|-------|---------------|-----------------|
| 1 | `[UNI3C] Task X: use_uni3c=True` in logs | If missing but task had `use_uni3c`, param_whitelist is incomplete |
| 2 | `[UNI3C] Task X: Uni3C ENABLED` in logs | If missing, task_registry didn't inject params |
| 3 | `[UNI3C] generate_video: Uni3C ENABLED` | If shows `DISABLED` but Layer 2 shows `ENABLED`, signature filtering dropped params |
| 4 | `[UNI3C] any2video: Encoded render_latent shape` | If missing, guide video wasn't loaded |
| 5 | `[UNI3C] model.forward: Uni3C data present` | If missing, uni3c_data didn't reach model |
| 6 | `[UNI3C] Step X: Applying residual` | If residual mean=0 or NaN, numerical issue |
| VRAM | `[UNI3C] VRAM before/after controlnet` | If not logged or OOM, memory issue |
| End | `[UNI3C] GENERATION COMPLETE` | If missing, generation didn't finish or counter broken |

---

## Phase Gate

Before moving to Phase 5:

1. Create a task with `use_uni3c=true` and `uni3c_guide_video=<path>`
2. Run the task
3. Check logs show `[UNI3C]` at ALL layers (1 through 6)
4. Verify output is different from non-Uni3C task

---

## Watchouts

1. **Both whitelists required**: `param_whitelist` in task_conversion.py AND params in `generate_video()` signature. Missing either = silent failure.

2. **URL handling**: If `uni3c_guide_video` is a URL, download it to temp dir before passing to WGP.

3. **Explicit-false semantics**: Like SVI, `use_uni3c=false` in segment should override `use_uni3c=true` in orchestrator.

---

## Next Phase

→ [Phase 5: Testing & Validation](./PHASE_5_TESTING.md)

