# SVI + End Frame Support

This document describes how to use the SVI (Stable Video Infinity) encoding with end frame support in Wan2GP.

## Overview

SVI is a feature for image-to-video generation that uses an anchor image to maintain consistency across long video generations. This implementation extends SVI to support **end frames**, allowing you to generate videos that transition smoothly from a start image to a target end image.

### How It Works

The latent structure for SVI + end frame is:

```
lat_y = [anchor_latent | zeros | zeros | ... | end_latent]
         ↑              ↑                        ↑
         encoded        diffusion generates      encoded
         separately     freely here              separately
```

- **Anchor latent**: The starting/reference frame (encoded separately)
- **Zeros**: Middle frames where the diffusion model generates motion
- **End latent**: The target end frame (encoded separately)

The mask tells the model:
- `[1,1,1,1 | 0,0,0,...,0 | 1,1,1,1]`
- First 4 sub-frames: known (preserve anchor)
- Middle: unknown (generate)
- Last 4 sub-frames: known (preserve end)

## Model Configuration

Use the pre-configured model: `wan_2_2_i2v_lightning_svi_3_3`

This config includes:
- **Architecture**: `i2v_2_2` with `svi2pro: true`
- **LoRAs**: SVI Pro LoRAs + LightX2V acceleration LoRAs
- **Phases**: 2-phase setup with proper model switching
- **Steps**: 6 inference steps

## Usage via Worker

### Task Structure

Create a task in Supabase with the following structure:

```json
{
  "task_type": "i2v",
  "params": {
    "model_type": "wan_2_2_i2v_lightning_svi_3_3",
    "prompt": "A smooth cinematic transition between scenes",
    "image_start": "https://your-bucket.supabase.co/path/to/start.png",
    "image_end": "https://your-bucket.supabase.co/path/to/end.png",
    "image_refs": ["https://your-bucket.supabase.co/path/to/anchor.png"],
    "video_prompt_type": "I",
    "video_length": 81,
    "resolution": "768x576"
  }
}
```

### Key Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model_type` | string | Yes | Use `wan_2_2_i2v_lightning_svi_3_3` |
| `prompt` | string | Yes | Text description of the video |
| `image_start` | string | Yes | URL/path to starting frame |
| `image_end` | string | Yes | URL/path to target end frame |
| `image_refs` | array | Yes | SVI anchor image(s). Usually same as `image_start` |
| `video_prompt_type` | string | Yes | **Must be `"I"`** to enable `image_refs` passthrough |
| `video_length` | int | No | Number of frames (default: 81) |
| `resolution` | string | No | Output resolution (default: from model config) |

### Example: SQL Insert

```sql
INSERT INTO tasks (task_type, status, params, project_id)
VALUES (
  'i2v',
  'Queued',
  '{
    "model_type": "wan_2_2_i2v_lightning_svi_3_3",
    "prompt": "A person walking smoothly through a garden",
    "image_start": "https://example.supabase.co/storage/v1/object/public/images/start.png",
    "image_end": "https://example.supabase.co/storage/v1/object/public/images/end.png",
    "image_refs": ["https://example.supabase.co/storage/v1/object/public/images/start.png"],
    "video_prompt_type": "I",
    "video_length": 81,
    "resolution": "768x576"
  }'::jsonb,
  'your-project-id'
);
```

### Example: Python

```python
from supabase import create_client

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

task = supabase.table("tasks").insert({
    "task_type": "i2v",
    "status": "Queued",
    "project_id": "your-project-id",
    "params": {
        "model_type": "wan_2_2_i2v_lightning_svi_3_3",
        "prompt": "A smooth cinematic transition between scenes",
        "image_start": start_image_url,
        "image_end": end_image_url,
        "image_refs": [anchor_image_url],  # Usually same as image_start
        "video_prompt_type": "I",
        "video_length": 81,
        "resolution": "768x576"
    }
}).execute()
```

## Usage via Test Script

For local testing, use the provided test script:

```bash
python test_svi_with_end_frame.py \
  --start samples/video.mp4 \
  --end samples/end.png \
  --frames 81 \
  --prompt "A smooth transition"
```

### Test Script Options

| Option | Default | Description |
|--------|---------|-------------|
| `--start` | `samples/video.mp4` | Source video (uses last 3 frames) |
| `--end` | `samples/1.png` | Target end image |
| `--anchor` | (auto) | Anchor image (defaults to first start frame) |
| `--prompt` | (default) | Generation prompt |
| `--frames` | 81 | Output video length |
| `--num-start-frames` | 3 | Frames to extract from start video |
| `--use-regular-i2v` | false | Use regular I2V instead of SVI encoding |

## Alternative: Regular I2V with SVI LoRAs

If you want better end frame fidelity without SVI encoding, use:

```json
{
  "model_type": "wan_2_2_i2v_lightning_svi_endframe",
  "svi2pro": false
}
```

This uses regular I2V encoding (`[start | zeros | end]` in pixel space before VAE) while still applying the SVI + Lightning LoRAs.

## Technical Details

### LoRA Phase Configuration

The model uses 4 LoRAs with phase-based multipliers:

| LoRA | Phase 1 (High Noise) | Phase 2 (Low Noise) |
|------|---------------------|---------------------|
| SVI High Noise | 1.0 | 0.0 |
| SVI Low Noise | 0.0 | 1.0 |
| LightX2V High Noise | 1.2 | 0.0 |
| LightX2V Low Noise | 0.0 | 1.0 |

### Phase Switching

- **Phase 1** (Steps 0-1): High noise model + high noise LoRAs
- **Phase 2** (Steps 2-5): Low noise model + low noise LoRAs
- Switch threshold: 883 (noise level)

### Model Config Location

```
Wan2GP/defaults/wan_2_2_i2v_lightning_svi_3_3.json
```

## Troubleshooting

### "First frame repeated for all frames"

This was caused by encoding anchor-repeated pixels instead of using zeros. Fixed in the current implementation.

### "Undercooked" output

Check:
1. `model_switch_phase` should be `1` for 2-phase setups
2. `guidance_scale` may need adjustment (default is 1, try 3 for stronger conditioning)
3. Verify LoRAs are loading (check logs for "Lora ... was loaded")

### `image_refs` not being used

Ensure `video_prompt_type: "I"` is set. Without this, `image_refs` will be ignored.

## Files

- `Wan2GP/defaults/wan_2_2_i2v_lightning_svi_3_3.json` - Model configuration
- `Wan2GP/models/wan/any2video.py` - SVI encoding implementation
- `test_svi_with_end_frame.py` - Test script

