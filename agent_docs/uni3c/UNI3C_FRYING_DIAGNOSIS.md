# Uni3C "Frying" Artifact Diagnosis Guide

This document provides a comprehensive comparison between our Uni3C implementation and Kijai's ComfyUI-WanVideoWrapper, identifying potential causes of "frying" artifacts.

## Symptom Description

"Frying" manifests as:
- High-frequency noise/detail enhancement
- Unnatural sharpening or texture
- Grainy artifacts in detailed areas
- Overcooked/burned look in certain regions
- Color banding or posterization in gradients

---

## Complete Code Path Comparison

### Overview: End-to-End Flow

**Our Flow**:
1. `any2video._load_guide_video()` - Load video from disk
2. `any2video._encode_guide_video_to_latent()` - Normalize & VAE encode
3. `wgp.generate()` - Pass render_latent through generation
4. `model._compute_uni3c_states()` - Concat & run controlnet
5. `model.forward()` - Apply residuals per block

**Kijai's Flow**:
1. `WanVideoEncode` node - Load & normalize video
2. `vae.encode()` - VAE encode
3. `WanVideoUni3C_embeds` node - Package embeds
4. `WanModel.forward()` - Concat, run controlnet, apply residuals

---

## Detailed Culprit Analysis

### CULPRIT 1: Input Video Normalization

#### Our Implementation
**File**: `Wan2GP/models/wan/any2video.py:420-430`
```python
# Stack and normalize: [F, H, W, C] -> [C, F, H, W], range [0,255] -> [-1, 1]
video = np.stack(frames, axis=0)  # [F, H, W, C]
video = video.astype(np.float32)
video = (video / 127.5) - 1.0  # [-1, 1] (Wan2GP convention)
video = torch.from_numpy(video).permute(3, 0, 1, 2)  # [C, F, H, W]

print(f"[UNI3C] any2video: Guide video tensor shape: {tuple(video.shape)}, dtype: {video.dtype}")
print(f"[UNI3C] any2video:   value range: [{video.min().item():.2f}, {video.max().item():.2f}]")
```

#### Kijai's Implementation
**File**: `ComfyUI-WanVideoWrapper/nodes.py:2246-2257`
```python
image = image.to(vae.dtype).to(device).unsqueeze(0).permute(0, 4, 1, 2, 3) # B, C, T, H, W

if noise_aug_strength > 0.0:
    image = add_noise_to_reference_video(image, ratio=noise_aug_strength)

if isinstance(vae, TAEHV):
    latents = vae.encode_video(image.permute(0, 2, 1, 3, 4), parallel=False)
    latents = latents.permute(0, 2, 1, 3, 4)
else:
    latents = vae.encode(image * 2.0 - 1.0, device=device, tiled=enable_vae_tiling, ...)
```

#### Key Differences
| Aspect | Our Implementation | Kijai's Implementation |
|--------|-------------------|----------------------|
| Input range | [0, 255] uint8 | [0, 1] float |
| Normalization formula | `/ 127.5 - 1.0` | `* 2.0 - 1.0` |
| Noise augmentation | Not implemented | Optional `noise_aug_strength` |
| Color space | BGR (OpenCV) | RGB (ComfyUI) |

#### Mathematical Verification
```
Our formula:    (pixel / 127.5) - 1.0
  - pixel=0:    (0 / 127.5) - 1.0 = -1.0 ✓
  - pixel=127.5: (127.5 / 127.5) - 1.0 = 0.0 ✓
  - pixel=255:  (255 / 127.5) - 1.0 = 1.0 ✓

Kijai formula: pixel * 2.0 - 1.0  (assuming [0,1] input)
  - pixel=0:    0 * 2.0 - 1.0 = -1.0 ✓
  - pixel=0.5:  0.5 * 2.0 - 1.0 = 0.0 ✓
  - pixel=1.0:  1.0 * 2.0 - 1.0 = 1.0 ✓
```

#### Color Channel Order - VERIFIED OK
**Our code** correctly converts BGR to RGB after loading:
```python
ret, frame = cap.read()  # BGR order from OpenCV
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Already present at line 406
```

**Kijai's code** expects RGB from ComfyUI.

**Status**: ✅ Already fixed - not the cause of frying.

#### Diagnostic Logs
```
[UNI3C] any2video: Guide video tensor shape: (3, 81, 720, 1280), dtype: torch.float32
[UNI3C] any2video:   value range: [-1.00, 1.00]
```
**Verify**: Range should be exactly [-1.0, 1.0] or very close.

---

### CULPRIT 2: VAE Z-Score Normalization

#### Our WanVAE Wrapper
**File**: `Wan2GP/models/wan/modules/vae.py:788-852`
```python
class WanVAE:
    def __init__(self, ...):
        mean = [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
                0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921]
        std = [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
               3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160]
        self.mean = torch.tensor(mean, dtype=dtype, device=device)
        self.std = torch.tensor(std, dtype=dtype, device=device)
        self.scale = [self.mean, 1.0 / self.std]  # [mean, 1/std]

    def encode(self, videos, tile_size=256, any_end_frame=False):
        scale = [u.to(device=self.device) for u in self.scale]
        if tile_size > 0:
            return [self.model.spatial_tiled_encode(u.to(self.dtype).unsqueeze(0), scale, tile_size, ...).float().squeeze(0) for u in videos]
        else:
            return [self.model.encode(u.to(self.dtype).unsqueeze(0), scale, ...).float().squeeze(0) for u in videos]
```

#### Kijai's WanVAE Wrapper
**File**: `ComfyUI-WanVideoWrapper/wanvideo/modules/vae.py:618-654`
```python
class WanVAE:
    def __init__(self, ...):
        mean = [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
                0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921]
        std = [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
               3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160]
        self.mean = torch.tensor(mean, dtype=dtype, device=device)
        self.std = torch.tensor(std, dtype=dtype, device=device)
        self.scale = [self.mean, 1.0 / self.std]

    def encode(self, videos):
        with torch.autocast(device_type=..., enabled=False):
            return [
                self.model.encode(u.unsqueeze(0), self.scale).float().squeeze(0)
                for u in videos
            ]
```

#### Z-Score Normalization in WanVAE_ Model
**Our implementation** (`vae.py:593-600`):
```python
mu, log_var = self.conv1(out).chunk(2, dim=1)
if scale != None:
    if isinstance(scale[0], torch.Tensor):
        mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(1, self.z_dim, 1, 1, 1)
    else:
        mu = (mu - scale[0]) * scale[1]
return mu  # Returns ONLY mean (deterministic)
```

#### Key Observations
1. **Mean/std values are IDENTICAL** between implementations
2. **Both return deterministic `mu`** (not sampled)
3. **Scale device handling**:
   - Ours: `scale = [u.to(device=self.device) for u in self.scale]`
   - Kijai: Uses `self.scale` directly (assumes same device)
4. **Autocast**:
   - Kijai: `torch.autocast(..., enabled=False)` - explicitly disables
   - Ours: No autocast context

#### POTENTIAL ISSUE: Autocast During Encode
Kijai explicitly disables autocast during VAE encode. We don't have this safeguard.

**Check**: Is autocast enabled during our VAE encode? If so, precision could be reduced.

#### Diagnostic Logs
```
[UNI3C_DIAG] Latent stats: mean=X.XXXX, std=X.XXXX
[UNI3C_DIAG] Latent range: min=X.XXXX, max=X.XXXX
```
**Expected values**:
- Mean: Should be ~0.0 (after z-score normalization)
- Std: Should be ~1.0
- Range: Typically [-3.0, 3.0] for well-behaved latents

---

### CULPRIT 3: Tiled VAE Encoding

#### Our Tiled Encode Implementation
**File**: `vae.py:695-734`
```python
def spatial_tiled_encode(self, x, scale, tile_size, any_end_frame=False):
    tile_sample_min_size = tile_size
    tile_latent_min_size = int(tile_sample_min_size / 8)
    tile_overlap_factor = 0.25

    overlap_size = int(tile_sample_min_size * (1 - tile_overlap_factor))
    blend_extent = int(tile_latent_min_size * tile_overlap_factor)
    row_limit = tile_latent_min_size - blend_extent

    # Split video into tiles and encode them separately
    rows = []
    for i in range(0, x.shape[-2], overlap_size):
        row = []
        for j in range(0, x.shape[-1], overlap_size):
            tile = x[:, :, :, i: i + tile_sample_min_size, j: j + tile_sample_min_size]
            tile = self.encode(tile, any_end_frame=any_end_frame)  # NO SCALE PASSED!
            row.append(tile)
        rows.append(row)

    # Blend tiles together
    result_rows = []
    for i, row in enumerate(rows):
        result_row = []
        for j, tile in enumerate(row):
            if i > 0:
                tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
            if j > 0:
                tile = self.blend_h(row[j - 1], tile, blend_extent)
            result_row.append(tile[:, :, :, :row_limit, :row_limit])
        result_rows.append(torch.cat(result_row, dim=-1))

    mu = torch.cat(result_rows, dim=-2)

    # Z-score normalization applied AFTER blending
    if isinstance(scale[0], torch.Tensor):
        mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(1, self.z_dim, 1, 1, 1)
    else:
        mu = (mu - scale[0]) * scale[1]

    return mu
```

#### CRITICAL ISSUE: Scale Not Passed to Tile Encode
At line 710, individual tiles are encoded WITHOUT the scale parameter:
```python
tile = self.encode(tile, any_end_frame=any_end_frame)  # MISSING: scale parameter!
```

**Impact**: Each tile is encoded without z-score normalization. Normalization is only applied after blending (lines 728-732).

**Mathematical Analysis**:
- Let `E(x)` = raw encoder output (before normalization)
- Let `N(x)` = normalized = `(x - mean) / std`
- Blending is linear: `blend(a, b) = w*a + (1-w)*b`

**Correct order**: `blend(N(tile1), N(tile2))` = blend normalized tiles
**Our order**: `N(blend(E(tile1), E(tile2)))` = normalize blended raw tiles

For linear operations: `N(blend(a, b)) = blend(N(a), N(b))`... BUT only if mean=0!
Since mean ≠ 0, the order matters:
- `N(blend(a,b)) = (w*a + (1-w)*b - mean) / std`
- `blend(N(a), N(b)) = w*(a-mean)/std + (1-w)*(b-mean)/std = (w*a + (1-w)*b - mean) / std`

Actually these ARE equivalent! So this is NOT a bug.

#### BUT: Blending Function Analysis
**File**: `vae.py` (blend functions not shown, need to verify)

The blend functions use linear interpolation, which preserves the mathematical equivalence. However, if there's any non-linear processing or clamping in the blend, it could cause issues.

---

### CULPRIT 4: dtype Conversion Chain

#### Full dtype Chain Analysis

**Our Path**:
```
1. Load video (cv2.read)           -> uint8 [0, 255]
2. Convert to float32              -> float32 [0.0, 255.0]
3. Normalize                       -> float32 [-1.0, 1.0]
4. Convert to VAE_dtype            -> float32 (VAE_dtype default)
5. VAE internal processing         -> float32
6. VAE output                      -> float32 (explicit .float())
7. Convert to model dtype          -> bfloat16 (self.dtype)
8. Concatenation                   -> float32 (hidden_states uses .float())
9. Controlnet input conversion     -> controlnet.dtype (fp16/fp8)
```

**Kijai's Path**:
```
1. ComfyUI image input             -> float32 [0.0, 1.0] (ComfyUI standard)
2. Convert to VAE dtype            -> vae.dtype
3. Normalize (* 2.0 - 1.0)         -> [-1.0, 1.0]
4. VAE encode (autocast disabled)  -> float32
5. Store in uni3c_data             -> varies
6. Forward: convert to base_dtype  -> bf16/fp16
7. Concat with float() hidden_states -> float32
8. Controlnet input                -> controlnet.dtype
```

#### POTENTIAL ISSUE: bfloat16 Conversion

At `any2video.py:577`:
```python
render_latent = latent.unsqueeze(0).to(self.dtype)  # self.dtype is typically bfloat16
```

This converts the VAE output from float32 to bfloat16, which has reduced precision for large values.

**bfloat16 characteristics**:
- 1 sign bit, 8 exponent bits, 7 mantissa bits
- Same range as float32 but less precision
- Can cause quantization artifacts for values outside [-1, 1]

**If latent values are large** (e.g., before z-score normalization), bfloat16 conversion could cause precision loss.

---

### CULPRIT 5: Autocast During ControlNet Forward

#### Kijai's Implementation
**File**: `model.py:3155-3163`
```python
if uni3c_data["offload"] or self.uni3c_controlnet.device != self.main_device:
    self.uni3c_controlnet.to(self.main_device)
with torch.autocast(device_type=mm.get_autocast_device(device), dtype=self.base_dtype, enabled=self.uni3c_controlnet.quantized):
    uni3c_controlnet_states = self.uni3c_controlnet(
        render_latent=render_latent.to(self.main_device, self.uni3c_controlnet.dtype),
        render_mask=uni3c_data["render_mask"],
        camera_embedding=uni3c_data["camera_embedding"],
        temb=e.to(self.main_device),
        out_device=self.offload_device if uni3c_data["offload"] else device)
```

#### Our Implementation
**File**: `model.py:1513-1518`
```python
controlnet_states = controlnet(
    render_latent=render_latent_input.to(main_device, controlnet.dtype),
    render_mask=uni3c_data.get("render_mask"),
    camera_embedding=uni3c_data.get("camera_embedding"),
    temb=temb.to(main_device, controlnet.dtype),
    out_device=offload_device if uni3c_data.get("offload") else main_device
)
```

#### Key Difference
Kijai uses `torch.autocast(..., enabled=self.uni3c_controlnet.quantized)`:
- When quantized (fp8): Autocast is enabled -> mixed precision
- When not quantized: Autocast is disabled -> full precision

We do NOT use autocast. This could lead to:
1. Different behavior for quantized models
2. Potentially different numerical results

---

### CULPRIT 6: Noisy Latent Extraction

#### Our Implementation
**File**: `model.py:1604-1605`
```python
# Save raw noisy latent for Uni3C (before patch embedding)
raw_noisy_latent = x_list[0].clone() if uni3c_data is not None else None
```

Then at lines 1466-1476:
```python
hidden_states = noisy_latent.unsqueeze(0).clone().float() if noisy_latent.dim() == 4 else noisy_latent.clone().float()

if hidden_states.shape[1] == 16:
    padding = torch.zeros_like(hidden_states[:, :4])
    hidden_states = torch.cat([hidden_states, padding], dim=1)

hidden_states = hidden_states.to(main_device)
```

#### Kijai's Implementation
**File**: `model.py:2471-2478`
```python
if uni3c_data is not None:
    render_latent = uni3c_data["render_latent"].to(self.base_dtype)
    hidden_states = x[0].unsqueeze(0).clone().float()
    if hidden_states.shape[1] == 16:
        hidden_states = torch.cat([hidden_states, torch.zeros_like(hidden_states[:, :4])], dim=1)
    if hidden_states.shape[2] != render_latent.shape[2]:
        render_latent = nn.functional.interpolate(render_latent, ...)
    render_latent = torch.cat([hidden_states[:, :20], render_latent], dim=1)
```

#### Comparison
| Aspect | Our Implementation | Kijai's Implementation |
|--------|-------------------|----------------------|
| Noisy latent source | `x_list[0]` saved earlier | `x[0]` directly in forward |
| `.clone()` | Yes | Yes |
| `.float()` | Yes | Yes |
| Channel check | `shape[1] == 16` | `shape[1] == 16` |
| Padding | `zeros_like([:,:4])` | `zeros_like([:,:4])` |
| Device handling | Explicit `.to(main_device)` | Implicit |

**These are equivalent.**

---

### CULPRIT 7: Temporal Resampling

#### Our Implementation
**File**: `model.py:1491-1499`
```python
if hidden_states.shape[2:] != render_latent.shape[2:]:
    if current_step == 0:
        print(f"[UNI3C] Resampling render_latent from {tuple(render_latent.shape[2:])} -> {tuple(hidden_states.shape[2:])}")
    render_latent = torch.nn.functional.interpolate(
        render_latent.float(),
        size=tuple(hidden_states.shape[2:]),
        mode="trilinear",
        align_corners=False,
    ).to(render_latent.dtype)
```

#### Kijai's Implementation
**File**: `model.py:2476-2477`
```python
if hidden_states.shape[2] != render_latent.shape[2]:
    render_latent = nn.functional.interpolate(render_latent,
        size=(hidden_states.shape[2], hidden_states.shape[3], hidden_states.shape[4]),
        mode='trilinear', align_corners=False)
```

#### Key Differences
| Aspect | Our Implementation | Kijai's Implementation |
|--------|-------------------|----------------------|
| Condition check | `shape[2:] != shape[2:]` (all dims) | `shape[2] != shape[2]` (temporal only) |
| Pre-conversion | `.float()` before interpolate | None |
| Post-conversion | `.to(render_latent.dtype)` after | None (stays in whatever dtype) |

**POTENTIAL ISSUE**: We convert to float32 before interpolation, then back. This extra conversion could:
1. Introduce rounding errors
2. Behave differently than Kijai for quantized latents

**POTENTIAL ISSUE**: Kijai only checks temporal dimension, but resamples all 3 (T, H, W). If H or W differ but T matches, Kijai would NOT resample but we WOULD. However, this scenario shouldn't happen in normal use.

---

### CULPRIT 8: Controlnet Residual Injection

#### Our Implementation
**File**: `model.py:1947-1960`
```python
# Apply residual to ALL streams in x_list
for i, x in enumerate(x_list):
    x_start = ref_images_count  # Skip prefix tokens
    apply_len = min(x.shape[1] - x_start, residual.shape[1])

    if block_idx == 0 and current_step_no == 0 and i == 0:
        print(f"[UNI3C DEBUG]   x.shape[1]={x.shape[1]}, x_start={x_start}, ...")

    if apply_len > 0:
        x_list[i][:, x_start:x_start + apply_len] += (
            residual[:, :apply_len].to(x) * uni3c_data.get("controlnet_weight", 1.0)
        )
```

#### Kijai's Implementation
**File**: `model.py:3275-3276`
```python
if uni3c_controlnet_states is not None and b < len(uni3c_controlnet_states):
    x[:, :self.original_seq_len] += uni3c_controlnet_states[b].to(x) * uni3c_data["controlnet_weight"]
```

#### Key Differences
| Aspect | Our Implementation | Kijai's Implementation |
|--------|-------------------|----------------------|
| Token range | `[x_start:x_start+apply_len]` | `[:original_seq_len]` |
| Skip prefix | Yes (`x_start = ref_images_count`) | No (starts at 0) |
| Length calc | `min(x.shape[1]-x_start, residual.shape[1])` | `original_seq_len` (fixed) |
| Multiple streams | Yes (iterates `x_list`) | No (single `x`) |

**POTENTIAL ISSUE**: We skip `ref_images_count` tokens at the start. Kijai injects from position 0.

**Question**: What is `ref_images_count` in our code? If it's non-zero when it shouldn't be, we'd be injecting into wrong positions.

**Question**: What is `self.original_seq_len` in Kijai's code? Need to verify it matches our intended range.

---

### CULPRIT 9: Guide Video Frame Handling

#### Our Frame Loading
**File**: `any2video.py:395-430`
```python
def _load_guide_video(self, guide_video_path: str, target_frames: int, frame_policy: str, height: int, width: int):
    cap = cv2.VideoCapture(guide_video_path)
    frames = []
    while True:
        ret, frame = cap.read()  # BGR!
        if not ret:
            break
        # Resize to match target dimensions
        if frame.shape[0] != height or frame.shape[1] != width:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LANCZOS4)
        frames.append(frame)
    cap.release()

    # Apply frame policy
    frames = self._apply_uni3c_frame_policy(frames, target_frames, frame_policy)

    # Stack and normalize
    video = np.stack(frames, axis=0)  # [F, H, W, C]
    video = video.astype(np.float32)
    video = (video / 127.5) - 1.0  # [-1, 1]
    video = torch.from_numpy(video).permute(3, 0, 1, 2)  # [C, F, H, W]
```

#### POTENTIAL ISSUES

1. **BGR vs RGB**: OpenCV loads as BGR, but Wan models expect RGB.
   ```python
   # MISSING: frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
   ```

2. **Interpolation Method**: We use `INTER_LANCZOS4`, which is high-quality but can cause ringing artifacts near edges.

3. **Frame Policy Application**: If `frame_policy` causes significant temporal resampling, it could introduce artifacts.

---

### CULPRIT 10: Empty Frame Detection & Zeroing

#### Our Implementation
**File**: `any2video.py:476-595`
```python
def _detect_empty_frames(self, guide_video: torch.Tensor, threshold: float = 0.02) -> list:
    """Detect which frames are "empty" (black/near-black)."""
    num_frames = guide_video.shape[1]
    empty_mask = []
    for f in range(num_frames):
        frame = guide_video[:, f]  # [C, H, W]
        # In [-1, 1] space, black = -1
        max_deviation = (frame + 1.0).abs().max().item()  # Distance from black
        is_empty = max_deviation < threshold
        empty_mask.append(is_empty)
    return empty_mask

# Later: Zero out latent frames for empty pixel frames
if zero_empty_frames and any(empty_pixel_mask):
    for lat_f, pixel_indices in enumerate(pixel_to_latent):
        if all(empty_pixel_mask[p] for p in pixel_indices if p < len(empty_pixel_mask)):
            render_latent[:, :, lat_f, :, :] = 0.0  # Hard zero
```

#### POTENTIAL ISSUE: Hard Zero vs VAE-Encoded Black

When we zero latent frames, we set them to literal 0.0. However, VAE-encoded black would NOT be 0.0 - it would be the z-score normalized value of whatever the VAE produces for black input.

**Impact**: Hard zeros might be treated differently than "natural" zero-control by the controlnet.

**Alternative**: Could encode a single black frame and use that value instead of 0.0.

---

## Summary: Ranked Culprits by Likelihood

### HIGH Probability

1. ~~**BGR vs RGB Color Order**~~ ✅ VERIFIED OK - already converted at line 406

2. **bfloat16 Precision Loss**
   - Converting VAE output to bfloat16 before storing
   - At `any2video.py:577`: `render_latent = latent.unsqueeze(0).to(self.dtype)`
   - Fix: Keep as float32 until concatenation, or verify values are in safe range

3. **Missing Autocast for Quantized ControlNet**
   - Kijai uses `torch.autocast(..., enabled=controlnet.quantized)`
   - We don't use autocast at all
   - Fix: Add autocast wrapper around controlnet forward

4. **ref_images_count Offset in Residual Injection** - LIKELY NOT THE ISSUE
   - We skip `ref_images_count` prefix tokens: `x[:, x_start:x_start + apply_len]`
   - Kijai injects from position 0: `x[:, :original_seq_len]`
   - **Key insight**: For pure t2v+uni3c, `ref_images_count=0` so we inject from 0 (same as Kijai)
   - For i2v+uni3c, our offset is probably CORRECT (Kijai may not support this combo)
   - Check logs: If `ref_images_count=0`, this is not the issue

### MEDIUM Probability

5. **Float32 Conversion During Temporal Interpolation**
   - We do `.float()` -> interpolate -> `.to(original_dtype)`
   - Kijai interpolates without explicit dtype conversion
   - Verify: Check if this causes precision issues

6. **Hard Zero vs Natural Zero for Empty Frames**
   - We zero empty latent frames with literal `0.0`
   - VAE-encoded black would NOT be 0.0
   - Test: Try `zero_empty_frames=False` or encode black frame for reference

7. **Missing Autocast During VAE Encode**
   - Kijai explicitly disables autocast: `torch.autocast(..., enabled=False)`
   - We don't have this safeguard
   - Verify: Check if autocast is active during our VAE encode

### LOW Probability

8. **Tiled vs Non-Tiled Encoding**
   - Mathematically equivalent for linear operations
   - Test: Try `tile_size=0` for guide video encoding

9. **Frame Resize Interpolation**
   - We use `INTER_LANCZOS4` which can cause ringing near edges
   - Test: Try `INTER_LINEAR` or `INTER_AREA` instead

---

## Clarification: Two Different "Skip" Mechanisms

There are two separate mechanisms that affect uni3c at sequence boundaries:

### 1. ref_images_count Offset (START of sequence)
- **What**: Skips injecting into reference image TOKENS at the start
- **When**: Only when `ref_images_count > 0` (i2v mode with prepended ref tokens)
- **Why**: Controlnet output doesn't include ref tokens, so we align injection position
- **For pure t2v+uni3c**: `ref_images_count=0`, so this does nothing (inject from 0)

### 2. Blackout Last Frame (END of guide video)
- **What**: Zeros the guide video's last LATENT FRAME before controlnet
- **When**: When `uni3c_blackout_last_frame=True`
- **Why**: Let i2v end anchor handle final frame without uni3c interference
- **Effect**: Controlnet still runs, still injects, but residual is based on zero input = weak control

**These are independent mechanisms affecting different things:**
- ref_images_count → WHERE we inject in hidden states (token position)
- blackout → WHAT we inject (guide signal content for last frame)

---

## Diagnostic Commands

### 1. Check Latent Statistics
Run a generation and look for:
```
[UNI3C_DIAG] Latent stats: mean=X.XXXX, std=X.XXXX
[UNI3C_DIAG] Latent range: min=X.XXXX, max=X.XXXX
```

**Expected**: mean ≈ 0.0, std ≈ 1.0, range ∈ [-3, 3]

### 2. Check Concatenation Statistics
```
[UNI3C_DIAG] Noisy latent (ch0-19): mean=X.XXXX, std=X.XXXX, range=[X.XXXX, X.XXXX]
[UNI3C_DIAG] Guide latent (ch20-35): mean=X.XXXX, std=X.XXXX, range=[X.XXXX, X.XXXX]
```

**Expected**: Both should have similar statistics if properly normalized.

### 3. Check Resampling
```
[UNI3C] Resampling render_latent from (T, H, W) -> (T', H', W')
```

**Ideally**: No resampling needed (frame counts match).

---

## Quick Fixes to Test

### Fix 1: Keep float32 Longer (HIGH PRIORITY)
**File**: `any2video.py:577`
```python
# BEFORE:
render_latent = latent.unsqueeze(0).to(self.dtype)  # Converts to bfloat16

# AFTER:
render_latent = latent.unsqueeze(0)  # Keep as float32, convert later in _compute_uni3c_states
```

### Fix 2: Add Autocast Wrapper for ControlNet (HIGH PRIORITY)
**File**: `model.py:1513`
```python
# BEFORE:
controlnet_states = controlnet(...)

# AFTER:
with torch.autocast(device_type='cuda', dtype=controlnet.dtype, enabled=getattr(controlnet, 'quantized', False)):
    controlnet_states = controlnet(...)
```

### Fix 3: Disable Autocast During VAE Encode
**File**: `vae.py:844-852`
```python
# In WanVAE.encode():
def encode(self, videos, tile_size=256, any_end_frame=False):
    scale = [u.to(device=self.device) for u in self.scale]
    with torch.autocast(device_type='cuda', enabled=False):  # ADD THIS LINE
        if tile_size > 0:
            return [self.model.spatial_tiled_encode(...)]
        else:
            return [self.model.encode(...)]
```

### Fix 4: Disable Empty Frame Zeroing (Test)
```python
# In any2video.generate() call or in _encode_guide_video_to_latent:
zero_empty_frames=False
```

### Fix 5: Force Non-Tiled Encoding (Test)
**File**: `any2video.py:576`
```python
# BEFORE:
latent = self.vae.encode([guide_video], tile_size=VAE_tile_size)[0]

# AFTER:
latent = self.vae.encode([guide_video], tile_size=0)[0]  # Force no tiling
```

### Fix 6: Verify ref_images_count
**File**: `model.py:1949`

Check if `ref_images_count` is being passed correctly for uni3c use case. For pure uni3c (no reference images), this should be 0, meaning we inject from position 0 like Kijai does.

```python
# Debug: Add this log at injection point
print(f"[UNI3C DEBUG] ref_images_count={ref_images_count}, x_start={x_start}")
```

---

## Fixes Already Implemented

The following fixes have been applied to the codebase:

### ✅ Fix 1: Autocast Wrapper for ControlNet (DONE)
**File**: `model.py:1513-1521`
```python
with torch.autocast(device_type='cuda', dtype=controlnet.dtype, enabled=getattr(controlnet, 'quantized', False)):
    controlnet_states = controlnet(...)
```

### ✅ Fix 2: Disable Autocast During VAE Encode (DONE)
**File**: `vae.py:849-854`
```python
with torch.autocast(device_type='cuda', enabled=False):
    # VAE encode operations...
```

### ✅ Fix 3: Enhanced Diagnostic Logging (DONE)
**Files**: `any2video.py:582-583`, `model.py:1507-1511`, `model.py:1956-1959`
- Latent statistics after VAE encode
- Comparison of noisy vs guide latent at concatenation
- ref_images_count and injection position logging

### ✅ Fix 4: BGR to RGB Conversion (ALREADY EXISTED)
**File**: `any2video.py:406`
```python
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
```
