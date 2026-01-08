# Original Wan2GP SVI Implementation

This document provides an in-depth analysis of how SVI (Stable Video Infinity) Pro is implemented in the original [Wan2GP repository](https://github.com/deepbeepmeep/Wan2GP).

## Table of Contents

1. [Overview](#overview)
2. [Architecture-Based Model Definition](#architecture-based-model-definition)
3. [Model Handler Setup](#model-handler-setup)
4. [The Sliding Window System](#the-sliding-window-system)
5. [Video Source and Prefix Video Flow](#video-source-and-prefix-video-flow)
6. [SVI Pro Encoding Path](#svi-pro-encoding-path)
7. [Mask Construction](#mask-construction)
8. [LoRA Configuration](#lora-configuration)
9. [Key Differences from Our Headless Approach](#key-differences-from-our-headless-approach)

---

## Overview

SVI Pro enables "infinite" video generation by:
1. Using **anchor images** (reference images) that maintain visual consistency across windows
2. **Overlapping frames** from previous segments to ensure smooth transitions
3. **Special LoRAs** that are trained for high-noise and low-noise phases to handle continuation
4. A **special VAE encoding strategy** that encodes prefix frames separately and concatenates in latent space

The key insight is that SVI Pro doesn't just "continue" a video - it re-encodes the overlap region in latent space alongside an anchor image, allowing the model to generate coherent content while maintaining consistency with both the anchor and the previous segment.

---

## Architecture-Based Model Definition

In the original Wan2GP, SVI Pro is implemented as a **dedicated architecture type**, not a runtime flag.

### Model Definition JSON

```json
// defaults/i2v_2_2_svi2pro.json
{
    "model": {
        "name": "Wan2.2 Image2video SVI 2 Pro 14B",
        "architecture": "i2v_2_2_svi2pro",  // <-- Dedicated architecture
        "description": "Wan 2.2 Image 2 Video Stable Video Infinity Pro 2...",
        "URLs": "i2v_2_2",
        "URLs2": "i2v_2_2",
        "group": "wan2_2",
        "loras": [
            "https://huggingface.co/DeepBeepMeep/Wan2.2/resolve/main/SVI_Wan2.2-I2V-A14B_high_noise_lora_v2.0_pro.safetensors",
            "https://huggingface.co/DeepBeepMeep/Wan2.2/resolve/main/SVI_Wan2.2-I2V-A14B_low_noise_lora_v2.0_pro.safetensors"
        ],
        "loras_multipliers": ["1;0", "0;1"]  // Phase-based: high_noise in phase1, low_noise in phase2
    },
    "guidance_phases": 2,
    "switch_threshold": 900,
    "guidance_scale": 3.5,
    "guidance2_scale": 3.5,
    "flow_shift": 5
}
```

### Architecture Detection Functions

In `models/wan/wan_handler.py`:

```python
# Line 45-46: Test if architecture is SVI Pro
def test_svi2pro(base_model_type):
    return base_model_type in ["i2v_2_2_svi2pro"]

# Line 15: SVI Pro is included in the i2v class
def test_class_i2v(base_model_type):
    return base_model_type in [
        "i2v", "i2v_2_2", "fun_inp_1.3B", "fun_inp", "flf2v_720p", 
        "fantasy", "multitalk", "infinitetalk", "i2v_2_2_multitalk", 
        "animate", "chrono_edit", "steadydancer", "wanmove", "scail", 
        "i2v_2_2_svi2pro"  # <-- SVI Pro is an i2v variant
    ]
```

---

## Model Handler Setup

When the model is loaded, `wan_handler.py` sets up the model definition with special flags:

```python
# Line 206: Set svi2pro flag based on architecture
extra_model_def["svi2pro"] = svi2pro = test_svi2pro(base_model_type)

# Line 261: sliding_window is auto-enabled for all i2v types (including svi2pro)
extra_model_def.update({
    "sliding_window": base_model_type in ["multitalk", "infinitetalk", "t2v", "t2v_2_2", "fantasy", "animate", "lynx"] 
                      or test_class_i2v(base_model_type)  # <-- i2v_2_2_svi2pro passes this test
                      or test_wan_5B(base_model_type) 
                      or vace_class,
    # ... other settings
})

# Line 300-309: SVI Pro gets special image ref choices
if svi2pro:
    extra_model_def["image_ref_choices"] = {
        "choices": [
            ("No Anchor Image", ""),
            ("Anchor Images For Each Window", "KI"),
        ],
        "letters_filter": "KI",
        "show_label": False,
    }
    extra_model_def["all_image_refs_are_background_ref"] = True
    extra_model_def["parent_model_type"] = "i2v_2_2"  # Inherits from i2v_2_2
```

### Why This Matters

Because `i2v_2_2_svi2pro` passes `test_class_i2v()`:
- **`sliding_window = True`** is automatically set
- This enables the sliding window continuation system
- `reuse_frames` gets calculated properly
- `video_source` gets loaded and processed

---

## The Sliding Window System

The sliding window system is the backbone of video continuation in Wan2GP.

### Key Variables

| Variable | Description |
|----------|-------------|
| `sliding_window_size` | Max frames per generation window (e.g., 129) |
| `sliding_window_overlap` | Frames to overlap between windows (e.g., 4-9) |
| `reuse_frames` | Actual overlap frames used (computed) |
| `video_source` | Path to prefix video (previous segment's output) |
| `prefix_video` | Loaded tensor of the prefix video |
| `pre_video_guide` | Last `reuse_frames` from prefix_video |

### Reuse Frames Calculation

In `wgp.py` (lines 5446-5454):

```python
def test_any_sliding_window(model_type):
    model_def = get_model_def(model_type)
    if model_def is None:
        return False
    return model_def.get("sliding_window", False)

# In the generate function:
if test_any_sliding_window(model_type):
    if video_source is not None:
        current_video_length += sliding_window_overlap - 1
    sliding_window = current_video_length > sliding_window_size
    reuse_frames = min(sliding_window_size - latent_size, sliding_window_overlap)
else:
    sliding_window = False
    sliding_window_size = current_video_length
    reuse_frames = 0  # <-- CRITICAL: Without sliding_window=True, no frame reuse!
```

### Why `sliding_window=True` is Essential

If `sliding_window=False`:
- `reuse_frames = 0`
- `video_source` is never loaded into `prefix_video`
- The model has no context from previous segments
- **Result: Grey/brown frames** because the model is generating "from nothing"

---

## Video Source and Prefix Video Flow

### Step 1: Loading the Prefix Video

In `wgp.py` (lines 5629-5649):

```python
if window_no == 1 and (video_source is not None or image_start is not None):
    if image_start is not None:
        # First segment: use start image
        image_start_tensor, new_height, new_width = calculate_dimensions_and_resize_image(...)
        image_start_tensor = convert_image_to_tensor(image_start_tensor)
        pre_video_guide = prefix_video = image_start_tensor.unsqueeze(1)
    else:
        # Continuation: load video_source (the prefix video)
        prefix_video = preprocess_video(
            width=width, 
            height=height,
            video_in=video_source,  # Path to previous segment's last N frames
            max_frames=parsed_keep_frames_video_source,
            start_frame=0,
            fit_canvas=sample_fit_canvas,
            fit_crop=fit_crop,
            target_fps=fps,
            block_size=block_size
        )
        prefix_video = prefix_video.permute(3, 0, 1, 2)  # -> [C, F, H, W]
        prefix_video = prefix_video.float().div_(127.5).sub_(1.)  # Normalize to [-1, 1]
        
        new_height, new_width = prefix_video.shape[-2:]
        pre_video_guide = prefix_video[:, -reuse_frames:]  # Take last reuse_frames
    
    pre_video_frame = convert_tensor_to_image(prefix_video[:, -1])  # Last frame as PIL
    source_video_overlap_frames_count = pre_video_guide.shape[1]
    source_video_frames_count = prefix_video.shape[1]
    guide_start_frame = prefix_video.shape[1]
```

### Step 2: Passing to Generate

In `wgp.py` (around line 5992):

```python
sample = wan_model.generate(
    # ... many parameters ...
    pre_video_frame=pre_video_frame,       # Last frame as image (for non-SVI)
    prefix_video=prefix_video,             # Full prefix video tensor (for SVI Pro)
    prefix_frames_count=source_video_overlap_frames_count if window_no <= 1 else reuse_frames,
    overlap_size=sliding_window_overlap,
    # ... more parameters ...
)
```

---

## SVI Pro Encoding Path

The magic happens in `models/wan/any2video.py`. SVI Pro uses a **completely different encoding strategy** than standard i2v.

### Standard I2V Encoding (Non-SVI)

```python
# Standard: encode input_video directly
if not svi_pro:
    lat_y = self.vae.encode([enc], VAE_tile_size, any_end_frame=any_end_frame)[0]
```

### SVI Pro Encoding

In `any2video.py` (lines 620-642):

```python
svi_pro = model_def.get("svi2pro", False)
svi_mode = 2 if svi_pro else 0

# In the i2v encoding section:
if svi_pro or svi_mode and svi_ref_pad_num != 0:
    use_extended_overlapped_latents = False
    
    # Get anchor image (image_ref)
    if input_ref_images is None or len(input_ref_images) == 0:
        if pre_video_frame is None: 
            raise Exception("Missing Reference Image")
        image_ref = pre_video_frame
    else:
        image_ref = input_ref_images[min(window_no, len(input_ref_images)) - 1]
    
    image_ref = convert_image_to_tensor(image_ref).unsqueeze(1).to(device=self.device, dtype=self.VAE_dtype)
    
    if svi_pro:
        # Check for existing overlapped_latents OR compute from prefix_video
        if overlapped_latents is not None:
            post_decode_pre_trim = 1
        elif prefix_video is not None and prefix_video.shape[1] >= (5 + overlap_size):
            # CRITICAL: Encode the tail of prefix_video to get overlap latents
            overlapped_latents = self.vae.encode(
                [torch.cat([prefix_video[:, -(5 + overlap_size):]], dim=1)],
                VAE_tile_size
            )[0][:, -overlap_size // 4:].unsqueeze(0)
            post_decode_pre_trim = 1
        
        # Encode anchor image
        image_ref_latents = self.vae.encode([image_ref], VAE_tile_size)[0]
        
        # Calculate padding needed
        pad_len = lat_frames + ref_images_count - image_ref_latents.shape[1] \
                  - (overlapped_latents.shape[2] if overlapped_latents is not None else 0)
        pad_latents = torch.zeros(
            image_ref_latents.shape[0], pad_len, lat_h, lat_w,
            device=image_ref_latents.device, dtype=image_ref_latents.dtype
        )
        
        # FINAL LATENT CONSTRUCTION:
        if overlapped_latents is None:
            lat_y = torch.concat([image_ref_latents, pad_latents], dim=1)
        else:
            # [anchor_latents, overlap_latents, padding] <-- The key innovation!
            lat_y = torch.concat([
                image_ref_latents,           # Anchor image (1 latent frame)
                overlapped_latents.squeeze(0),  # Overlap from previous segment
                pad_latents                  # Zeros for frames to generate
            ], dim=1)
        
        image_ref_latents = None
```

### Visual Representation

```
Standard I2V:
┌─────────────────────────────────────────────────────────────┐
│ [input_video_latents] [zeros_for_new_frames]                │
└─────────────────────────────────────────────────────────────┘
      ↑ all from input_video

SVI Pro:
┌─────────────────────────────────────────────────────────────┐
│ [anchor_img] [overlap_from_prefix] [zeros_for_new_frames]   │
└─────────────────────────────────────────────────────────────┘
      ↑              ↑                       ↑
   1 frame    ~2 latent frames           remaining
  (encoded    (encoded from             (to generate)
  separately)  prefix_video tail)
```

---

## Mask Construction

The mask tells the model which frames to preserve vs generate.

### SVI Pro Mask (lines 664-666)

```python
msk = torch.ones(1, frame_num + ref_images_count * 4, lat_h, lat_w, device=self.device)

if any_end_frame:
    msk[:, control_pre_frames_count: -1] = 0  # Preserve start + end, generate middle
else:
    # For SVI: Only preserve first frame (anchor), generate everything else
    msk[:, 1 if svi_mode else control_pre_frames_count:] = 0
    
msk = torch.concat([
    torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1),  # Expand first frame
    msk[:, 1:]
], dim=1)

msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
msk = msk.transpose(1, 2)[0]
```

### Mask Values

- `1` = Preserve (known/fixed content)
- `0` = Generate (model fills in)

For SVI Pro without end frame:
```
Frame:    [anchor] [overlap1] [overlap2] [gen1] [gen2] [gen3] ... [genN]
Mask:     [  1   ] [   0    ] [   0    ] [ 0  ] [ 0  ] [ 0  ] ... [ 0  ]
```

The overlap frames are "generated" but seeded with the overlap latents, so they stay consistent.

---

## LoRA Configuration

SVI Pro requires specific LoRAs for the two-phase denoising process.

### Default SVI LoRAs

```json
"loras": [
    "SVI_Wan2.2-I2V-A14B_high_noise_lora_v2.0_pro.safetensors",
    "SVI_Wan2.2-I2V-A14B_low_noise_lora_v2.0_pro.safetensors"
],
"loras_multipliers": ["1;0", "0;1"]
```

### Phase Multipliers Meaning

- `"1;0"` = Strength 1.0 in phase 1 (high noise), 0.0 in phase 2 (low noise)
- `"0;1"` = Strength 0.0 in phase 1, 1.0 in phase 2

### With Lightning Acceleration

When using Lightning LoRAs alongside SVI:

```python
# Example combined LoRAs
loras = [
    "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors",  # Lightning high
    "14b-i2v.safetensors",                                         # Base enhancement
    "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors",   # Lightning low
    "SVI_Wan2.2-I2V-A14B_high_noise_lora_v2.0_pro.safetensors",   # SVI high
    "SVI_Wan2.2-I2V-A14B_low_noise_lora_v2.0_pro.safetensors"     # SVI low
]

# Phase multipliers (phase1;phase2)
loras_multipliers = ["1.2;0", "0.5;0.5", "0;1.0", "1;0", "0;1"]
```

The step-by-step multipliers are computed by `update_loras_slists()` in `loras_multipliers.py`.

---

## Key Differences from Our Headless Approach

| Aspect | Original Wan2GP | Our Headless Approach |
|--------|-----------------|----------------------|
| **Architecture** | Dedicated `i2v_2_2_svi2pro` type | Dynamic patching of baseline model |
| **sliding_window** | Auto-set by architecture | Must be patched at runtime |
| **svi2pro flag** | Set during model init | Patched before generation |
| **Model definition** | Single object throughout | Must patch both `models_def` AND `wan_model.model_def` |
| **LoRAs** | Pre-configured in model JSON | Dynamically merged from phase_config |

### Our Required Patches

To achieve SVI Pro with a baseline model, we must patch:

1. **`wgp.models_def[model_key]["svi2pro"] = True`**
   - Enables SVI mode checks in `any2video.py`

2. **`wgp.models_def[model_key]["sliding_window"] = True`**
   - Enables `test_any_sliding_window()` to return True
   - Allows `reuse_frames` to be computed
   - Triggers `video_source` loading

3. **`wgp.wan_model.model_def["svi2pro"] = True`**
   - Direct patch for runtime since model may already be loaded
   - The `any2video.py` code reads from `self.model_def`

4. **`wgp.wan_model.model_def["sliding_window"] = True`**
   - Ensures consistency (though `wgp.py` uses `models_def`)

### Why Both Objects Must Be Patched

```python
# In wgp.py - uses models_def via get_model_def()
def test_any_sliding_window(model_type):
    model_def = get_model_def(model_type)  # <- reads from models_def
    return model_def.get("sliding_window", False)

# In any2video.py - uses self.model_def
svi_pro = model_def.get("svi2pro", False)  # <- reads from wan_model.model_def
```

These can be **different objects** depending on when the model was loaded!

---

## Summary

The original SVI Pro implementation relies on:

1. **Architecture-based configuration** - `i2v_2_2_svi2pro` gets all the right flags automatically
2. **Sliding window system** - `sliding_window=True` enables frame reuse and video_source loading
3. **Special encoding path** - Encodes anchor + overlap separately, concatenates in latent space
4. **Phase-aware LoRAs** - High-noise and low-noise LoRAs switch at `switch_threshold`

Our headless approach achieves the same result through runtime patching, but must carefully ensure:
- Both `models_def` and `wan_model.model_def` are patched
- Patching happens BEFORE `generate()` is called
- All SVI-related parameters (`sliding_window_overlap`, `video_source`, `image_refs`, etc.) are passed correctly

