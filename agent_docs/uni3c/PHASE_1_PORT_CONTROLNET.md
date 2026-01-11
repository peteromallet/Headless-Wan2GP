# Phase 1: Port Uni3C ControlNet

← [Back to Start](./STARTING_POINT_AND_STATUS.md)

---

## Prerequisites
- None (this is the first phase)

## Tasks

| Task | Status | Notes |
|------|--------|-------|
| Create `Wan2GP/models/wan/uni3c/` directory | 🔴 | |
| Create `__init__.py` | 🔴 | |
| Create `load.py` (checkpoint download + config inference) | 🔴 | See code below |
| Port `controlnet.py` (WanControlNet class) | 🔴 | From Kijai's implementation |
| **Unit test**: Forward pass on dummy latents → correct shapes | 🔴 | **Phase gate** |

---

## What You Need to Know

### Directory Structure

Create under Wan2GP's model organization:

```
Wan2GP/models/wan/
├── uni3c/                  # NEW
│   ├── __init__.py
│   ├── controlnet.py       # WanControlNet class
│   └── load.py             # checkpoint loader utilities
```

### Checkpoint Details

**File**: [`Wan21_Uni3C_controlnet_fp16.safetensors`](https://huggingface.co/Kijai/WanVideo_comfy/blob/main/Wan21_Uni3C_controlnet_fp16.safetensors)
- **Size**: ~2 GB
- **Location**: `Kijai/WanVideo_comfy` on HuggingFace
- **Format**: safetensors (fp16)
- **Storage**: `Wan2GP/ckpts/controlnets/`  
  (i.e. your loader should typically be called with `ckpts_dir="Wan2GP/ckpts"` when run from repo root)

### ControlNet Architecture (from Kijai's impl)

| Property | Value | Source |
|----------|-------|--------|
| Transformer blocks | 20 | `controlnet_cfg["num_layers"] = 20` |
| dim | 1024 | `"dim": 1024` |
| time_embed_dim | 5120 | `"time_embed_dim": 5120` |
| proj_out dimension | 5120 | `nn.Linear(self.dim, 5120)` |
| Patch embedding | Conv3D(1,2,2) | `self.patch_size = (1, 2, 2)` |
| num_heads | 16 | `"num_heads": 16` |

**Important clarification**: Patch embedding produces `conv_out_dim = 5120`, but the transformer hidden `dim = 1024`.  
So the port must include an explicit **projection into transformer space** (e.g. `proj_in: 5120 → 1024`) before entering the block stack.

---

## Code: `load.py`

```python
"""
Uni3C ControlNet checkpoint loader with auto-download.
"""

import os
from pathlib import Path
from typing import Optional

import torch
from safetensors.torch import load_file

# HuggingFace repo and filename
UNI3C_REPO_ID = "Kijai/WanVideo_comfy"
UNI3C_CHECKPOINT_FILENAME = "Wan21_Uni3C_controlnet_fp16.safetensors"

def get_uni3c_checkpoint_path(ckpts_dir: str = "ckpts") -> Path:
    """Get the path where Uni3C checkpoint should be stored."""
    return Path(ckpts_dir) / "controlnets" / UNI3C_CHECKPOINT_FILENAME


def download_uni3c_checkpoint_if_missing(
    ckpts_dir: str = "ckpts",
    force_download: bool = False
) -> str:
    """
    Download Uni3C checkpoint from HuggingFace if not present.
    
    Args:
        ckpts_dir: Base checkpoints directory
        force_download: If True, download even if file exists
        
    Returns:
        Path to the checkpoint file
    """
    from huggingface_hub import hf_hub_download
    
    target_path = get_uni3c_checkpoint_path(ckpts_dir)
    target_dir = target_path.parent
    
    if target_path.exists() and not force_download:
        print(f"[UNI3C] Checkpoint already exists: {target_path}")
        return str(target_path)
    
    print(f"[UNI3C] Downloading checkpoint from {UNI3C_REPO_ID}...")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Download using HuggingFace hub
    downloaded_path = hf_hub_download(
        repo_id=UNI3C_REPO_ID,
        filename=UNI3C_CHECKPOINT_FILENAME,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False
    )
    
    print(f"[UNI3C] Downloaded checkpoint to: {downloaded_path}")
    return downloaded_path


def load_uni3c_checkpoint(
    checkpoint_path: Optional[str] = None,
    device: str = "cpu",
    dtype: Optional[torch.dtype] = None
) -> dict:
    """
    Load Uni3C checkpoint state dict.
    
    Args:
        checkpoint_path: Path to checkpoint (auto-downloads if None)
        device: Device to load tensors to
        dtype: Optional dtype conversion
        
    Returns:
        State dict with model weights
    """
    if checkpoint_path is None:
        checkpoint_path = download_uni3c_checkpoint_if_missing()
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"[UNI3C] Checkpoint not found: {checkpoint_path}")
    
    print(f"[UNI3C] Loading checkpoint from: {checkpoint_path}")
    
    # Load safetensors
    state_dict = load_file(checkpoint_path, device=device)
    
    # Log checkpoint stats for debugging
    total_params = sum(t.numel() for t in state_dict.values())
    total_size_mb = sum(t.numel() * t.element_size() for t in state_dict.values()) / 1024 / 1024
    sample_dtype = next(iter(state_dict.values())).dtype
    print(f"[UNI3C] Checkpoint loaded: {total_params:,} params, {total_size_mb:.1f} MB, dtype={sample_dtype}")
    
    # Infer config from checkpoint (matching Kijai's approach)
    config = infer_config_from_checkpoint(state_dict)
    print(f"[UNI3C] Inferred config: in_channels={config['in_channels']}, ffn_dim={config['ffn_dim']}")
    
    return state_dict, config


def infer_config_from_checkpoint(state_dict: dict) -> dict:
    """
    Infer Uni3C config from checkpoint tensor shapes.
    
    This matches Kijai's approach in nodes.py:57-73.
    """
    # Required keys to verify checkpoint validity
    if "controlnet_patch_embedding.weight" not in state_dict:
        raise ValueError("[UNI3C] Invalid checkpoint: missing controlnet_patch_embedding.weight")
    
    in_channels = state_dict["controlnet_patch_embedding.weight"].shape[1]
    ffn_dim = state_dict["controlnet_blocks.0.ffn.0.bias"].shape[0]
    
    return {
        "in_channels": in_channels,
        "conv_out_dim": 5120,
        "time_embed_dim": 5120,
        "dim": 1024,
        "ffn_dim": ffn_dim,
        "num_heads": 16,
        "num_layers": 20,
        "add_channels": 7,
        "mid_channels": 256,
    }
```

---

## Code: `controlnet.py` (Port from Kijai)

Port the `WanControlNet` class from Kijai's [`uni3c/controlnet.py`](https://github.com/kijai/ComfyUI-WanVideoWrapper/blob/main/uni3c/controlnet.py).

Key components to port:
- `WanControlNet` class
- `WanControlnetTransformerBlock` 
- Patch embedding (Conv3D)
- RoPE (rotary positional embedding)
- `proj_out` layers (20 of them, one per block)

**Critical implementation detail**: The forward signature is:
```python
def forward(self, render_latent, render_mask, camera_embedding, temb, 
            freqs=None, out_device=None):
```

For our MVP, `render_mask` and `camera_embedding` can remain `None` (not implemented).

**Timestep embedding (`temb`) shape**: In Wan2GP, `e = self.time_embedding(sinusoidal_embedding_1d(...))` is **2D** and is documented in the Sense Check as matching Uni3C’s `time_embed_dim = 5120`.  
So for Phase 1 (load + forward), treat `temb` as shape `[B, 5120]`.

→ See [Kijai Appendix](./_reference/KIJAI_APPENDIX.md) for full code snippets.

---

## Phase Gate: Unit Test

Before moving to Phase 2, verify:

```python
# test_uni3c_controlnet.py
import torch
from Wan2GP.models.wan.uni3c import load_uni3c_checkpoint, WanControlNet

# Load checkpoint
state_dict, config = load_uni3c_checkpoint()

# Create model
model = WanControlNet(**config)
model.load_state_dict(state_dict)
model.eval()

# Test forward pass
batch_size = 1
channels = config["in_channels"]  # likely 16 or 20
frames = 13
height, width = 60, 80  # latent dimensions for 480x640

render_latent = torch.randn(batch_size, channels, frames, height, width)
temb = torch.randn(batch_size, 5120)  # time embedding

with torch.no_grad():
    controlnet_states = model(
        render_latent=render_latent,
        render_mask=None,
        camera_embedding=None,
        temb=temb
    )

# Verify outputs
assert len(controlnet_states) == 20, f"Expected 20 states, got {len(controlnet_states)}"
for i, state in enumerate(controlnet_states):
    # Each state should have dimension 5120 in the last dim
    print(f"State {i}: {state.shape}")
    
print("✅ Phase 1 gate passed!")
```

**Note (dev environment)**: Importing `Wan2GP.models.wan...` will pull in Wan2GP’s `shared/attention.py`, which assumes a CUDA-enabled PyTorch build.  
If you’re running Phase 1 locally on CPU-only torch, run the test in a CUDA env (recommended) or load the Uni3C modules directly (via `importlib`) to avoid importing the broader Wan2GP package.

> Note: The snippets in this doc are **implementation guidance**. You’ll need to adapt imports and utilities to match the repo’s final module locations and available dependencies (e.g. `huggingface_hub`, `safetensors`).

---

## Watchouts

1. **in_channels may be 20, not 16**: Check the checkpoint's `controlnet_patch_embedding.weight.shape[1]`. If it's 20 and Wan2GP VAE produces 16-channel latents, you'll need padding in Phase 2.

2. **Don't integrate yet**: This phase is just about getting the ControlNet to load and run forward. Integration happens in Phase 3.

---

## Next Phase

→ [Phase 2: Guide Video → Latents](./PHASE_2_GUIDE_VIDEO_LATENTS.md)

