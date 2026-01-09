# Phase 2: Guide Video → Latents

← [Back to Start](./STARTING_POINT_AND_STATUS.md) | ← [Phase 1](./PHASE_1_PORT_CONTROLNET.md)

---

## Prerequisites
- Phase 1 complete (ControlNet loads and forward pass works)
- Know `in_channels` from checkpoint (16 or 20?)

## Tasks

| Task | Status | Notes |
|------|--------|-------|
| Add guide video loading in `any2video.py` | 🔴 | Load frames, resize, normalize |
| Add frame resampling per `uni3c_frame_policy` | 🔴 | fit/trim/loop/off |
| Add VAE encode to `render_latent` | 🔴 | Use existing VAE |
| Add 16→20 channel padding if needed | 🔴 | Check `in_channels` from Phase 1 |
| Add Layer 4 logging | 🔴 | Log shapes at each step |

---

## What You Need to Know

### Where This Code Goes

All guide video preprocessing happens in `Wan2GP/models/wan/any2video.py` in the `generate()` method.

### Guide Video → Latent Pipeline

1. **Load** guide video frames from path/URL
2. **Resize** to match target output resolution
3. **Resample** frames to match target frame count (per `uni3c_frame_policy`)
4. **Normalize** to Wan2GP's expected range
5. **VAE encode** to latent space
6. **Pad channels** if checkpoint expects 20 but VAE produces 16

### Frame Alignment Policy

Wan2GP models quantize frame count to `4N+1` style grids. The guide must match.

| Policy | Behavior |
|--------|----------|
| `fit` (default) | Resample guide video to exactly match target `video_length` |
| `trim` | If guide longer, trim. If shorter, hold last frame. |
| `loop` | Loop guide video to fill required frames |
| `off` | Require exact match, error otherwise |

---

## Code: Guide Video Loading

Add to `any2video.py`:

> **Existing utilities you should use:**
> - `source/video_utils.py:extract_frames_from_video()` - extracts frames as numpy arrays
> - `source/structure_video_guidance.py:load_structure_video_frames()` - uses `decord`, handles treatment modes
>
> The snippet below shows the **contract** (what inputs/outputs are expected). Adapt it to use the existing utilities above rather than raw OpenCV.

```python
def _load_uni3c_guide_video(
    self,
    guide_video_path: str,
    target_height: int,
    target_width: int,
    target_frames: int,
    frame_policy: str = "fit"
) -> torch.Tensor:
    """
    Load and preprocess guide video for Uni3C.
    
    Returns:
        Tensor of shape [C, F, H, W] ready for VAE encoding
    """
    import cv2
    import numpy as np
    
    print(f"[UNI3C] any2video: Loading guide video from {guide_video_path}")
    
    # Load video frames
    cap = cv2.VideoCapture(guide_video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # BGR -> RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Resize to target
        frame = cv2.resize(frame, (target_width, target_height))
        frames.append(frame)
    cap.release()
    
    print(f"[UNI3C] any2video: Loaded {len(frames)} frames")
    
    # Apply frame policy
    frames = self._apply_frame_policy(frames, target_frames, frame_policy)
    print(f"[UNI3C] any2video: After frame policy '{frame_policy}': {len(frames)} frames")
    
    # Stack and normalize: [F, H, W, C] -> [C, F, H, W]
    video = np.stack(frames, axis=0)  # [F, H, W, C]
    video = video.astype(np.float32) / 255.0  # [0, 1]
    video = video * 2.0 - 1.0  # [-1, 1]
    video = torch.from_numpy(video).permute(3, 0, 1, 2)  # [C, F, H, W]
    
    print(f"[UNI3C] any2video: Guide video tensor shape: {video.shape}")
    return video


def _apply_frame_policy(
    self,
    frames: list,
    target_frames: int,
    policy: str
) -> list:
    """Apply frame alignment policy."""
    current = len(frames)
    
    if policy == "off":
        if current != target_frames:
            raise ValueError(
                f"[UNI3C] Frame count mismatch: guide has {current}, "
                f"target is {target_frames}. Use a different frame_policy."
            )
        return frames
    
    elif policy == "fit":
        # Resample to exact target count
        if current == target_frames:
            return frames
        indices = np.linspace(0, current - 1, target_frames).astype(int)
        return [frames[i] for i in indices]
    
    elif policy == "trim":
        if current >= target_frames:
            return frames[:target_frames]
        else:
            # Hold last frame
            return frames + [frames[-1]] * (target_frames - current)
    
    elif policy == "loop":
        if current >= target_frames:
            return frames[:target_frames]
        else:
            # Loop until filled
            result = []
            while len(result) < target_frames:
                result.extend(frames)
            return result[:target_frames]
    
    else:
        raise ValueError(f"[UNI3C] Unknown frame_policy: {policy}")
```

---

## Code: VAE Encoding with Padding

```python
def _encode_uni3c_guide(
    self,
    guide_video: torch.Tensor,  # [C, F, H, W]
    expected_channels: int = 20  # from checkpoint in_channels
) -> torch.Tensor:
    """
    VAE-encode guide video and pad channels if needed.
    
    Returns:
        render_latent: [B, C, F, H, W] ready for ControlNet
    """
    # Add batch dim: [C, F, H, W] -> [1, C, F, H, W]
    guide_video = guide_video.unsqueeze(0)
    
    # VAE encode
    # Note: Check Wan2GP's VAE API - might be self.vae.encode()
    render_latent = self.vae.encode([guide_video], tile_size=self.VAE_tile_size)[0]
    
    print(f"[UNI3C] any2video: Encoded render_latent shape: {render_latent.shape}")
    print(f"[UNI3C] any2video:   Expected channels: {expected_channels}")
    print(f"[UNI3C] any2video:   Actual channels: {render_latent.shape[1]}")
    
    # Pad 16 -> 20 if needed (T2V workaround from Kijai)
    if render_latent.shape[1] == 16 and expected_channels == 20:
        print(f"[UNI3C] any2video: Padding 16 -> 20 channels")
        padding = torch.zeros_like(render_latent[:, :4])
        render_latent = torch.cat([render_latent, padding], dim=1)
        print(f"[UNI3C] any2video:   After padding: {render_latent.shape}")
    
    return render_latent
```

---

## Code: Temporal Resampling (at inference time)

If the encoded latent's temporal dimension doesn't match the model's expectation, resample:

```python
# In model.py, before passing to ControlNet
if hidden_states.shape[2] != render_latent.shape[2]:
    print(f"[UNI3C] Temporal mismatch: hidden={hidden_states.shape[2]}, guide={render_latent.shape[2]}")
    render_latent = torch.nn.functional.interpolate(
        render_latent, 
        size=(hidden_states.shape[2], hidden_states.shape[3], hidden_states.shape[4]), 
        mode='trilinear', 
        align_corners=False
    )
    print(f"[UNI3C] After interpolation: {render_latent.shape}")
```

---

## Phase Gate

Before moving to Phase 3, verify:

```python
# Test that guide video loads and encodes correctly
guide_path = "samples/video.mp4"  # Use a test video
target_h, target_w = 480, 640
target_frames = 49  # typical 4N+1

# Load
guide_tensor = model._load_uni3c_guide_video(
    guide_path, target_h, target_w, target_frames, "fit"
)
print(f"Guide tensor: {guide_tensor.shape}")  # Should be [3, 49, 480, 640]

# Encode
render_latent = model._encode_uni3c_guide(guide_tensor, expected_channels=20)
print(f"Render latent: {render_latent.shape}")  # Should be [1, 20, F', H', W']

print("✅ Phase 2 gate passed!")
```

---

## Watchouts

1. **Color space**: Wan2GP might expect a different normalization range than [-1, 1]. Check existing video loading code.

2. **Frame count quantization**: The target frame count should come from Wan2GP's internal calculation, not just from the task's `video_length` param.

3. **Memory**: Guide video loading can be memory-intensive. Consider streaming frames if needed.

---

## Next Phase

→ [Phase 3: Model Integration](./PHASE_3_MODEL_INTEGRATION.md)

