# Kijai Implementation Reference

← [Back to Start](../STARTING_POINT_AND_STATUS.md)

> Code snippets from [`kijai/ComfyUI-WanVideoWrapper`](https://github.com/kijai/ComfyUI-WanVideoWrapper).
> Validated January 2026.

---

## `uni3c_data` Dict Structure

The integration passes a single dict through the forward chain:

```python
uni3c_data = {
    "controlnet": <WanControlNet instance>,
    "controlnet_weight": 1.0,  # strength multiplier
    "start": 0.0,              # start_percent
    "end": 1.0,                # end_percent
    "render_latent": <tensor [B,C,F,H,W]>,
    "render_mask": None,       # NOT IMPLEMENTED in nodes
    "camera_embedding": None,  # NOT IMPLEMENTED in nodes
    "offload": True,           # whether to offload between steps
}
```

---

## Key Code Snippets from `wanvideo/modules/model.py`

### 16→20 Channel Padding (line ~2473)

```python
hidden_states = x[0].unsqueeze(0).clone().float()
if hidden_states.shape[1] == 16:  # T2V workaround
    hidden_states = torch.cat([hidden_states, torch.zeros_like(hidden_states[:, :4])], dim=1)
```

### Temporal Resampling (line ~2476)

```python
if hidden_states.shape[2] != render_latent.shape[2]:
    render_latent = nn.functional.interpolate(
        render_latent, 
        size=(hidden_states.shape[2], hidden_states.shape[3], hidden_states.shape[4]), 
        mode='trilinear', 
        align_corners=False
    )
```

### Step-Percent Gating (line ~3152)

```python
if (uni3c_data["start"] <= current_step_percentage <= uni3c_data["end"]) or \
        (uni3c_data["end"] > 0 and current_step == 0 and current_step_percentage >= uni3c_data["start"]):
```

### Controlnet Forward Call (line ~3158)

```python
uni3c_controlnet_states = self.uni3c_controlnet(
    render_latent=render_latent.to(self.main_device, self.uni3c_controlnet.dtype),
    render_mask=uni3c_data["render_mask"],
    camera_embedding=uni3c_data["camera_embedding"],
    temb=e.to(self.main_device),  # <-- Uses pre-projection `e`
    out_device=self.offload_device if uni3c_data["offload"] else device
)
```

### Per-Block Residual Injection (line ~3275)

```python
if uni3c_controlnet_states is not None and b < len(uni3c_controlnet_states):
    x[:, :self.original_seq_len] += uni3c_controlnet_states[b].to(x) * uni3c_data["controlnet_weight"]
```

---

## ControlNet Config (from `uni3c/nodes.py`)

Inferred from checkpoint + hardcoded:

```python
controlnet_cfg = {
    "in_channels": <from checkpoint: controlnet_patch_embedding.weight.shape[1]>,
    "conv_out_dim": 5120,
    "time_embed_dim": 5120,
    "dim": 1024,
    "ffn_dim": <from checkpoint: controlnet_blocks.0.ffn.0.bias.shape[0]>,
    "num_heads": 16,
    "num_layers": 20,
    "add_channels": 7,
    "mid_channels": 256,
    "attention_mode": "sdpa",  # or "sageattn"
    "quantized": <True if fp8>,
    "base_dtype": <fp16/bf16/fp32>
}
```

---

## WanControlNet Class (from `uni3c/controlnet.py`)

### Key Architecture Points

```python
class WanControlNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        conv_out_dim: int = 5120,
        time_embed_dim: int = 5120,
        dim: int = 1024,
        ffn_dim: int = 8192,
        num_heads: int = 16,
        num_layers: int = 20,
        add_channels: int = 7,
        mid_channels: int = 256,
        attention_mode: str = "sdpa",
        quantized: bool = False,
        base_dtype: torch.dtype = torch.float16
    ):
        super().__init__()
        
        self.dim = dim
        self.patch_size = (1, 2, 2)  # Temporal, Height, Width
        
        # Patch embedding: Conv3D
        self.controlnet_patch_embedding = nn.Conv3d(
            in_channels, conv_out_dim, 
            kernel_size=self.patch_size, 
            stride=self.patch_size
        )
        
        # Time embedding
        self.time_embedding = nn.Sequential(
            nn.Linear(256, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # 20 transformer blocks
        self.controlnet_blocks = nn.ModuleList([
            WanControlnetTransformerBlock(dim, ffn_dim, num_heads, attention_mode)
            for _ in range(num_layers)
        ])
        
        # Per-block projection to output dimension
        self.proj_out = nn.ModuleList([
            nn.Linear(dim, 5120)  # Output matches main model's hidden dim
            for _ in range(num_layers)
        ])
```

### Forward Method

```python
def forward(
    self, 
    render_latent: torch.Tensor,      # [B, C, F, H, W]
    render_mask: torch.Tensor = None,  # Not implemented
    camera_embedding: torch.Tensor = None,  # Not implemented
    temb: torch.Tensor = None,         # [B, 5120] timestep embedding
    freqs: torch.Tensor = None,        # RoPE frequencies
    out_device: torch.device = None
) -> list:
    """
    Returns:
        List of 20 tensors, one per block, each shape [B, seq_len, 5120]
    """
    # Patch embedding
    hidden_states = self.controlnet_patch_embedding(render_latent)
    # Reshape: [B, C, F, H, W] -> [B, seq_len, dim]
    hidden_states = hidden_states.flatten(2).transpose(1, 2)
    
    # Add time embedding
    if temb is not None:
        hidden_states = hidden_states + self.time_embedding(temb)[:, None, :]
    
    # Run through blocks, collect outputs
    controlnet_states = []
    for i, block in enumerate(self.controlnet_blocks):
        hidden_states = block(hidden_states, freqs=freqs)
        # Project and collect
        state = self.proj_out[i](hidden_states)
        if out_device is not None:
            state = state.to(out_device)
        controlnet_states.append(state)
    
    return controlnet_states
```

---

## Offload Pattern

```python
# Before controlnet forward
if uni3c_data.get("offload"):
    self.uni3c_controlnet.to(self.main_device)

# Run forward
controlnet_states = self.uni3c_controlnet(...)

# After forward
if uni3c_data.get("offload"):
    self.uni3c_controlnet.to(self.offload_device)
```

---

## Source Repository

- **Repo**: [kijai/ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper)
- **Key files**:
  - `uni3c/controlnet.py` - WanControlNet class
  - `uni3c/nodes.py` - Config inference, checkpoint loading
  - `wanvideo/modules/model.py` - Integration into main transformer

---

## Checkpoint

- **File**: [`Wan21_Uni3C_controlnet_fp16.safetensors`](https://huggingface.co/Kijai/WanVideo_comfy/blob/main/Wan21_Uni3C_controlnet_fp16.safetensors)
- **Size**: ~2 GB
- **Format**: safetensors (fp16)
- **Location**: HuggingFace `Kijai/WanVideo_comfy`

