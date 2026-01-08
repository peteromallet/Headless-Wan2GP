# SVI (Stable Video Infinity) Implementation Guide

## Overview

SVI is a technique for generating arbitrarily long videos by continuing from previous segments. It uses specialized LoRAs and encoding to maintain temporal coherence across segment boundaries.

## How SVI Works (Conceptually)

### Standard I2V Generation
```
[Start Image] → [Generate ~80 frames] → [Output Video]
```

### SVI Continuation
```
Segment 0: [Start Image] → [Generate 77 frames] → [Seg0 Output]
                                                      ↓
Segment 1: [Last 9 frames of Seg0] → [Generate 77 new frames] → [Seg1 Output]
                                                                     ↓
Segment 2: [Last 9 frames of Seg1] → [Generate 77 new frames] → [Seg2 Output]
```

The key insight: SVI uses the **last N frames** of the previous segment as context ("prefix") for generating the next segment, ensuring smooth transitions.

---

## Architecture: From Task to Generation

### 1. Orchestrator Creates Segments (`travel_between_images.py`)

```
User Request: "Generate 3-segment video with SVI"
                    ↓
Orchestrator splits into:
  - Segment 0: use_svi=False, svi2pro=False (normal I2V)
  - Segment 1: use_svi=True, svi2pro=True (SVI continuation)
  - Segment 2: use_svi=True, svi2pro=True (SVI continuation)
```

**Key segment payload fields for SVI:**
- `use_svi: true` - Enable SVI mode
- `svi2pro: true` - Use SVI Pro encoding (pixel-space concatenation)
- `video_prompt_type: "I"` - Image refs mode
- `additional_loras` - Includes SVI LoRAs with phase multipliers

### 2. Task Registry Configures Generation (`task_registry.py`)

When `use_svi=True`:
```python
generation_params["svi2pro"] = True
generation_params["video_prompt_type"] = "I"
generation_params["video_source"] = "/path/to/prefix_video.mp4"  # Last 9 frames of predecessor
generation_params["image_refs_paths"] = ["/path/to/anchor.png"]  # Last frame as anchor
generation_params["image_end"] = end_anchor_image  # Target end frame
generation_params["sliding_window_overlap"] = 4
```

### 3. Model Management Patches Model Definition (`headless_model_management.py`)

**Critical:** `svi2pro` is a model_def property, not a generation parameter. We must patch it into the loaded model:

```python
# Patch wan_model.model_def directly (not just wgp.models_def)
wgp.wan_model.model_def["svi2pro"] = True
wgp.wan_model.model_def["sliding_window"] = True  # Enables video_source loading
```

**Why patch both?**
- `wgp.models_def` is the global registry
- `wan_model.model_def` is the **actual object** used during generation (captured at model load time)

### 4. WGP Loads Video Source (`wgp.py`)

When `sliding_window=True` and `video_source` is provided:
```python
# Load video_source (9 frames)
prefix_video = preprocess_video(video_source, ...)

# Take last reuse_frames as context
pre_video_guide = prefix_video[:, -reuse_frames:]  # reuse_frames from sliding_window_overlap

# Pass to generate
wan_model.generate(input_video=pre_video_guide, ...)
```

**Current Issue:** We're setting `sliding_window=True` as a hack to enable `reuse_frames > 0`. This may have unintended side effects.

### 5. any2video.py Encodes with SVI Pro Path

When `svi_pro=True`:
```python
# Check model_def for svi2pro flag
svi_pro = model_def.get("svi2pro", False)

if svi_pro:
    # SVI Pro encoding path - kijai-style
    # Pixel-space concatenation with single VAE encode
    
    # input_video provides context frames (from video_source)
    control_video = input_video  # Shape: [C, prefix_frames, H, W]
    
    # Build pixel tensor: [context_frames | zeros | end_frame]
    enc = torch.concat([control_video, zeros_for_generation, end_frame], dim=1)
    
    # Single VAE encode for temporal coherence
    lat_y = self.vae.encode([enc], ...)
    
    # Build mask: 1=preserve (known frames), 0=generate (new content)
    msk = torch.zeros(...)
    msk[:, :control_pre_frames_count] = 1  # Preserve prefix
    msk[:, -1:] = 1  # Preserve end frame
```

---

## SVI LoRAs and Phase Multipliers

### LoRAs Used
1. **Lightning High Noise LoRA** (`wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors`)
   - Multiplier: `1.2;0` → Phase 1: 1.2, Phase 2: 0

2. **Custom Style LoRA** (e.g., `14b-i2v.safetensors`)
   - Multiplier: `0.50;0.50` → Both phases: 0.5

3. **Lightning Low Noise LoRA** (`wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors`)
   - Multiplier: `0;1` → Phase 1: 0, Phase 2: 1

4. **SVI High Noise LoRA** (`SVI_Wan2.2-I2V-A14B_high_noise_lora_v2.0_pro.safetensors`)
   - Multiplier: `1;0` → Phase 1: 1, Phase 2: 0

5. **SVI Low Noise LoRA** (`SVI_Wan2.2-I2V-A14B_low_noise_lora_v2.0_pro.safetensors`)
   - Multiplier: `0;1` → Phase 1: 0, Phase 2: 1

### Phase Logic (6 steps, 2 phases)
```
Steps 0-1 (Phase 1 - High Noise):
  - Lightning High: 1.2
  - SVI High: 1.0
  - Lightning Low: 0
  - SVI Low: 0

Steps 2-5 (Phase 2 - Low Noise):
  - Lightning High: 0
  - SVI High: 0
  - Lightning Low: 1.0
  - SVI Low: 1.0
```

---

## Frame Arithmetic

### Example: 3-Segment Generation

```
Target: 200 total frames
Overlap: 4 frames between segments

Segment 0:
  - Generates: 77 frames (0-76)
  - Output: 77 frames

Segment 1:
  - Prefix: Last 9 frames of Seg0 (frames 68-76)
  - Generates: 81 frames (77 new + 4 overlap)
  - After trim: 77 new frames (77-153)
  - Overlap frames 68-71 are used for blending

Segment 2:
  - Prefix: Last 9 frames of Seg1 (frames 145-153)
  - Generates: 81 frames
  - After trim: ~47 frames to reach 200 total
```

### Why 9 Prefix Frames?
```
prefix_min_context = 5  # Minimum context for model to "see"
overlap_size = 4        # Frames that overlap with previous segment
frames_needed = 5 + 4 = 9
```

---

## Key Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `svi2pro` | `true` | Enable SVI Pro encoding path |
| `sliding_window_overlap` | `4` | Frames to overlap between segments |
| `video_prompt_type` | `"I"` | Image refs mode (anchor image) |
| `image_prompt_type` | `"SV"` | Start image + Video continuation |
| `guidance_scale` | `1` | Low guidance for continuation |
| `guidance2_scale` | `1` | Low guidance phase 2 |
| `switch_threshold` | `883` | Timestep to switch LoRA phases |
| `num_inference_steps` | `6` | Fast generation with Lightning |

---

## Mask Semantics

In the SVI encoding path:
- **`1` = Preserve** (known/context frames)
- **`0` = Generate** (new content)

```
Frame layout: [prefix_frames | generate_frames | end_frame]
Mask layout:  [   1 1 1 1    |    0 0 0 ... 0  |     1    ]
```

---

## Known Issues & Workarounds

### Issue 1: `svi2pro` Not Being Read
**Symptom:** `svi_pro=False` even when `svi2pro=True` is in generation_params

**Root Cause:** `any2video.py` reads `svi2pro` from `self.model_def`, which is captured at model load time, not from kwargs.

**Fix:** Patch `wan_model.model_def["svi2pro"] = True` directly before generation.

### Issue 2: `video_source` Not Being Loaded (Grey/Brown Frames)
**Symptom:** Middle frames are grey/brown, only anchors look correct

**Root Cause:** The baseline model lacks `sliding_window: true`, causing `reuse_frames=0` and `video_source` to be ignored.

**Current Workaround:** Patch `sliding_window=True` into model_def alongside `svi2pro`.

**TODO:** This is a hack. We should either:
1. Create a proper SVI model definition with `sliding_window: true`
2. Or modify WGP to load `video_source` regardless of sliding_window when `svi2pro=True`

### Issue 3: Mask Shape Mismatch
**Symptom:** Artifacts at segment boundaries

**Root Cause:** Different mask packing between standard I2V and SVI Pro paths.

**Fix:** Use standard end-frame mask packing for SVI Pro when `any_end_frame=True`.

---

## Debugging Guide

### Key Log Tags
```
[SVI_CONFIG]           - Segment configuration in orchestrator
[SVI_MODE]             - SVI mode detection in task_registry
[SVI2PRO]              - Model definition patching
[SVI2PRO_DIAG]         - Object ID comparison for patch verification
[SVI_PAYLOAD]          - WGP payload configuration
[SVI_VIDEO_SOURCE_DIAG] - pre_video_guide shape before generate()
[SVI_ENCODING_STATUS]  - svi_pro flag value at generation time
[SVI_ENCODING_PATH]    - Which encoding path is taken
[SVI_MASK_FIX]         - Mask configuration details
[LORA_LOAD_CONFIRM]    - LoRAs loaded into transformer
[LORA_STEP_MULTIPLIERS] - Per-step LoRA multiplier arrays
```

### Verification Checklist
1. ✅ `[SVI2PRO] Patched wan_model.model_def: svi2pro=True, sliding_window=True`
2. ✅ `[SVI_VIDEO_SOURCE_DIAG] pre_video_guide shape: [3, 9, H, W]` (9 frames, not 1)
3. ✅ `[SVI_ENCODING_STATUS] svi_pro=True`
4. ✅ `[SVI_ENCODING_PATH] ✅ ENTERED SVI_PRO ENCODING PATH`
5. ✅ `[SVI_MASK_FIX] known=N/total` where N > 1

---

## File Locations

| File | Purpose |
|------|---------|
| `source/sm_functions/travel_between_images.py` | Orchestrator, segment creation |
| `source/task_registry.py` | Task → generation_params conversion |
| `headless_model_management.py` | Model patching, generation dispatch |
| `Wan2GP/wgp.py` | Video loading, parameter resolution |
| `Wan2GP/models/wan/any2video.py` | SVI encoding, VAE, denoising |
| `Wan2GP/shared/utils/loras_mutipliers.py` | Phase-based LoRA multipliers |

---

## Future Improvements

1. **Proper SVI Model Definition:** Create a dedicated model config with `sliding_window: true` and `svi2pro: true` built-in.

2. **Direct video_source Loading:** Modify WGP to load `video_source` as `input_video` when `svi2pro=True`, independent of `sliding_window` flag.

3. **Better Prefix Handling:** Instead of relying on `reuse_frames`, have explicit "SVI prefix frames" parameter.

4. **Validation:** Add pre-flight checks to verify video_source frame count matches expectations.

