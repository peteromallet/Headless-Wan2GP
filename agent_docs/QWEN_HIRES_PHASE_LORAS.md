# Qwen Hires Fix - Phase-Based LoRA Multipliers

## Overview

The Qwen hires fix now supports **phase-based LoRA multipliers**, allowing you to define different LoRA strengths for each pass of the two-pass generation workflow.

This approach is inspired by the Lightning VACE models (see `Wan2GP/defaults/vace_14B_lightning_3p_2_2.json`).

## How It Works

### Two-Pass Workflow

1. **Pass 1** (Base Resolution): Generate at base resolution (e.g., 1152x864)
2. **Pass 2** (Hires Refinement): Upscale latents and refine at higher resolution (e.g., 2304x1728)

### Phase-Based Multiplier Format

Instead of a single strength value, you can specify strengths for each pass:

```
"pass1_strength;pass2_strength"
```

## Examples

### Basic Format

```python
"lora_multipliers": [
    "1.0;0",      # Lightning LoRA: ON for pass 1, OFF for pass 2
    "1.1;1.1",    # Style LoRA: Same strength in both passes
    "0.5;0.8"     # Scene LoRA: Different strengths per pass
]
```

### Automatic Conversion

When hires fix is enabled, the system **automatically converts** simple multipliers to phase-based format:

**Input:**
```python
{
    "lora_names": [
        "Qwen-Image-Edit-Lightning-8steps-V1.0-bf16.safetensors",
        "style_transfer_qwen_edit_2_000011250.safetensors",
        "in_scene_different_object_000010500.safetensors"
    ],
    "lora_multipliers": ["1.0", "1.1", "0.5"],
    "hires_scale": 2.0,
    "hires_steps": 6
}
```

**Converted to:**
```python
"lora_multipliers": [
    "1.0;0",      # Lightning auto-detected, disabled in pass 2
    "1.1;1.1",    # Standard LoRA, same strength both passes
    "0.5;0.5"     # Standard LoRA, same strength both passes
]
```

### Manual Override

You can manually specify phase-based multipliers to have full control:

```python
{
    "lora_names": [
        "Qwen-Image-Edit-Lightning-8steps-V1.0-bf16.safetensors",
        "style_transfer_qwen_edit_2_000011250.safetensors",
        "detail_enhancer.safetensors"
    ],
    "lora_multipliers": [
        "1.0;0",      # Lightning: OFF in pass 2
        "1.1;1.2",    # Style: Stronger in pass 2
        "0;0.8"       # Detail: ONLY in pass 2
    ],
    "hires_scale": 2.0,
    "hires_steps": 6
}
```

## Task Parameter Example

```python
task_params = {
    "prompt": "A beautiful landscape at sunset",
    "resolution": "1152x864",
    "num_inference_steps": 6,
    "seed": 12345,

    # Hires config
    "hires_scale": 2.0,
    "hires_steps": 6,
    "hires_denoise": 0.5,

    # LoRAs with phase-based multipliers
    "additional_loras": {
        "https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Edit-Lightning-8steps-V1.0-bf16.safetensors": "1.0;0",
        "https://huggingface.co/peteromallet/ad_motion_loras/resolve/main/style_transfer_qwen_edit_2_000011250.safetensors": "1.1;1.1",
        "https://huggingface.co/peteromallet/random_junk/resolve/main/in_scene_different_object_000010500.safetensors": "0.5;0.5"
    },

    "style_reference_image": "https://example.com/style.jpg",
    "style_reference_strength": 1.1
}
```

## Lightning LoRA Auto-Detection

The system automatically detects Lightning/accelerator LoRAs by checking if the filename contains:
- `"lightning"`
- `"distill"`
- `"accelerator"`

These are automatically disabled in pass 2 (set to `"X;0"`), since they're optimized for fast generation, not refinement.

## Benefits

1. **Better Quality**: Use Lightning LoRAs for fast pass 1, then disable them for cleaner pass 2 refinement
2. **Flexible Control**: Different LoRA strengths per pass for fine-tuning
3. **Backward Compatible**: Simple multipliers still work and are auto-converted
4. **Detail Enhancement**: Enable detail LoRAs only in pass 2 for final polish

## Comparison to Lightning VACE Models

This implementation follows the same pattern as Lightning VACE models:

**VACE 3-Phase Example:**
```json
"loras": [
    "Wan2.2-Lightning_T2V-v1.1-A14B-4steps-lora_HIGH_fp16.safetensors",
    "Wan2.2-Lightning_T2V-v1.1-A14B-4steps-lora_LOW_fp16.safetensors"
],
"loras_multipliers": [
    "0;1;0",  // Phase 1: OFF, Phase 2: ON, Phase 3: OFF
    "0;0;1"   // Phase 1: OFF, Phase 2: OFF, Phase 3: ON
]
```

**Qwen 2-Phase Hires:**
```python
"lora_names": [
    "Qwen-Image-Edit-Lightning-8steps-V1.0-bf16.safetensors",
    "style_transfer_qwen_edit_2_000011250.safetensors"
],
"lora_multipliers": [
    "1.0;0",    // Pass 1: ON, Pass 2: OFF
    "1.1;1.1"   // Pass 1: ON, Pass 2: ON
]
```

## Testing

See `test_qwen_hires.py` for a working example.
