# Uni3C Parameter Definitions

← [Back to Start](../STARTING_POINT_AND_STATUS.md)

> Reference for all Uni3C parameters: names, types, defaults, and behavior.

---

## Task Parameters

These are the parameters users can set on tasks:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_uni3c` | bool | `false` | Master enable flag. Must be `true` to activate Uni3C. |
| `uni3c_guide_video` | str | `None` | Path or URL to the motion guide video. Required when `use_uni3c=true`. |
| `uni3c_strength` | float | `1.0` | Scalar multiplier on Uni3C residuals. 0.0 = no effect, 1.0 = full effect. |
| `uni3c_start_percent` | float | `0.0` | Start applying Uni3C at this % of denoising steps. |
| `uni3c_end_percent` | float | `1.0` | Stop applying Uni3C at this % of denoising steps. |
| `uni3c_keep_on_gpu` | bool | `false` | If true, don't offload ControlNet between steps (faster, more VRAM). |
| `uni3c_frame_policy` | str | `"fit"` | How to align guide video frames to output frames. |

---

## Frame Policy Values

| Value | Behavior |
|-------|----------|
| `fit` | Resample guide video to exactly match target frame count (default) |
| `trim` | If guide longer, trim. If shorter, hold last frame. |
| `loop` | Loop guide video to fill required frames |
| `off` | Require exact frame match, error otherwise |

---

## Parameter Flow

```
┌─────────────────────┐
│   Task DB Params    │
│  (user-specified)   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  task_conversion.py │  ← Must be in param_whitelist
│   param_whitelist   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  task_registry.py   │  ← Resolution + injection
│ (generation_params) │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   headless_wgp.py   │  ← Merge with defaults
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│      wgp.py         │  ← Must be in generate_video() signature
│  generate_video()   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   any2video.py      │  ← Guide video loading + encoding
│     generate()      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│     model.py        │  ← Residual injection
│     forward()       │
└─────────────────────┘
```

---

## Where to Add Parameters

### `source/task_conversion.py`

Add to `param_whitelist` set:

```python
param_whitelist = {
    # ... existing ...
    "use_uni3c",
    "uni3c_guide_video",
    "uni3c_strength",
    "uni3c_start_percent",
    "uni3c_end_percent",
    "uni3c_keep_on_gpu",
    "uni3c_frame_policy",
}
```

### `Wan2GP/wgp.py`

Add to `generate_video()` function signature:

```python
def generate_video(
    # ... existing params ...
    use_uni3c: bool = False,
    uni3c_guide_video: str = None,
    uni3c_strength: float = 1.0,
    uni3c_start_percent: float = 0.0,
    uni3c_end_percent: float = 1.0,
    uni3c_keep_on_gpu: bool = False,
    uni3c_frame_policy: str = "fit",
):
```

### `Wan2GP/models/wan/any2video.py`

Add same params to `generate()` function signature.

---

## Precedence Rules

Following the same pattern as `use_svi`:

1. **Segment params** (highest priority)
2. **Orchestrator params**
3. **Model defaults** (lowest priority)

Explicit `false` in segment overrides `true` in orchestrator:

```python
# Segment says false → Uni3C disabled even if orchestrator says true
segment_params = {"use_uni3c": False}
orchestrator_params = {"use_uni3c": True}
# Result: use_uni3c = False
```

---

## Internal Parameters (not user-facing)

These are computed/derived during processing:

| Parameter | Type | Description |
|-----------|------|-------------|
| `uni3c_render_latent` | Tensor | VAE-encoded guide video latents [B, C, F, H, W] |
| `uni3c_data` | dict | Full dict passed to model forward (see below) |

### `uni3c_data` Dict

Constructed in `any2video.py`, passed to `model.py`:

```python
uni3c_data = {
    "controlnet": <WanControlNet>,    # Loaded model
    "controlnet_weight": float,        # From uni3c_strength
    "start": float,                    # From uni3c_start_percent
    "end": float,                      # From uni3c_end_percent
    "render_latent": Tensor,           # Encoded guide video
    "render_mask": None,               # Not implemented
    "camera_embedding": None,          # Not implemented
    "offload": bool,                   # Inverse of uni3c_keep_on_gpu
}
```

---

## Validation Rules

| Rule | Error if violated |
|------|-------------------|
| `use_uni3c=true` requires `uni3c_guide_video` | "Guide video required when use_uni3c is true" |
| `uni3c_strength` should be >= 0 | Warning only (negative values technically work) |
| `uni3c_start_percent` should be < `uni3c_end_percent` | Warning only (results in zero-length window) |
| `uni3c_frame_policy="off"` requires exact frame match | "Frame count mismatch" error |
| Guide video must be readable | "Failed to load guide video" error |

