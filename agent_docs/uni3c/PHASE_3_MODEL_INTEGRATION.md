# Phase 3: Model Integration

← [Back to Start](./STARTING_POINT_AND_STATUS.md) | ← [Phase 2](./PHASE_2_GUIDE_VIDEO_LATENTS.md)

---

## Prerequisites
- Phase 1 complete (ControlNet works standalone)
- Phase 2 complete (Guide video encodes to latents)

## Tasks

| Task | Status | Notes |
|------|--------|-------|
| Add `uni3c_data` to `model.py` forward signature | 🔴 | Dict structure below |
| Add step-percent gating logic | 🔴 | Check `start` <= current <= `end` |
| Add controlnet forward call (before block loop) | 🔴 | Compute all residuals upfront |
| Add per-block residual injection (inside block loop) | 🔴 | After each block's forward |
| Add Layer 5 + 6 logging | 🔴 | See logging code below |
| Add offload logic | 🔴 | Move to/from GPU between steps |

---

## What You Need to Know

### Injection Point

The core integration happens in `Wan2GP/models/wan/modules/model.py` in the main transformer forward pass.

**Key locations**:
- Timestep embedding: `e = self.time_embedding(sinusoidal_embedding_1d(...))` at ~line 1597
- Block loop: `for block_idx, block in enumerate(self.blocks)` at ~line 1759

### The `uni3c_data` Dict

Pass a single dict through the forward chain:

```python
uni3c_data = {
    "controlnet": <WanControlNet instance>,
    "controlnet_weight": 1.0,    # strength multiplier
    "start": 0.0,                # start_percent
    "end": 1.0,                  # end_percent
    "render_latent": <tensor>,   # [B, C, F, H, W]
    "render_mask": None,         # NOT IMPLEMENTED
    "camera_embedding": None,    # NOT IMPLEMENTED
    "offload": True,             # whether to offload between steps
}
```

### Critical Pattern

1. **Before block loop**: Run Uni3C ControlNet forward → get list of 20 residuals
2. **Inside block loop**: After each block runs, add its corresponding residual

This is NOT like VACE (which uses `vace_context`). Uni3C is per-block residual addition.

---

## Code: Step-Percent Gating

Add at start of forward pass:

```python
# Near top of forward pass, after computing timestep embedding `e`
uni3c_controlnet_states = None

if uni3c_data is not None:
    # Get current step percentage from offload.shared_state
    # (set in Wan2GP/models/wan/modules/model.py around line 1578-1579)
    current_step_no = offload.shared_state.get("step_no", 0)
    max_steps = offload.shared_state.get("max_steps", 1)
    current_step_percentage = current_step_no / max_steps
    
    in_window = (uni3c_data["start"] <= current_step_percentage <= uni3c_data["end"])
    
    # Also handle edge case: step 0 with non-zero end
    if uni3c_data["end"] > 0 and current_step_no == 0 and current_step_percentage >= uni3c_data["start"]:
        in_window = True
    
    if in_window:
        # Log activation (first time only)
        if current_step_no == 0:
            print(f"[UNI3C] model.forward: Uni3C data present")
            print(f"[UNI3C]   render_latent shape: {uni3c_data['render_latent'].shape}")
            print(f"[UNI3C]   step window: {uni3c_data['start']*100:.0f}% - {uni3c_data['end']*100:.0f}%")
        
        # Compute controlnet states
        uni3c_controlnet_states = self._compute_uni3c_states(uni3c_data, e)
```

---

## Code: ControlNet Forward (Before Block Loop)

```python
def _compute_uni3c_states(self, uni3c_data: dict, temb: torch.Tensor) -> list:
    """
    Run Uni3C ControlNet forward pass.
    
    Args:
        uni3c_data: Dict with controlnet, render_latent, etc.
        temb: Timestep embedding (pre-projection `e`)
        
    Returns:
        List of 20 residual tensors, one per block
    """
    controlnet = uni3c_data["controlnet"]
    render_latent = uni3c_data["render_latent"]
    
    # Log VRAM before controlnet forward (helps debug OOM)
    if torch.cuda.is_available():
        vram_before = torch.cuda.memory_allocated() / 1024**3
        print(f"[UNI3C] VRAM before controlnet forward: {vram_before:.2f} GB")
    
    # Move controlnet to GPU if offloaded
    if uni3c_data.get("offload", True):
        controlnet = controlnet.to(self.main_device)
    
    # Ensure render_latent matches hidden_states shape
    # (temporal resampling if needed - see Phase 2)
    
    # Run controlnet forward
    controlnet_states = controlnet(
        render_latent=render_latent.to(self.main_device, controlnet.dtype),
        render_mask=uni3c_data.get("render_mask"),
        camera_embedding=uni3c_data.get("camera_embedding"),
        temb=temb.to(self.main_device),
        out_device=self.offload_device if uni3c_data.get("offload") else self.main_device
    )
    
    # Log VRAM after controlnet forward
    if torch.cuda.is_available():
        vram_after = torch.cuda.memory_allocated() / 1024**3
        print(f"[UNI3C] VRAM after controlnet forward: {vram_after:.2f} GB")
    
    # Offload controlnet back if configured
    if uni3c_data.get("offload", True):
        controlnet.to(self.offload_device)
        if torch.cuda.is_available():
            vram_offloaded = torch.cuda.memory_allocated() / 1024**3
            print(f"[UNI3C] VRAM after offload: {vram_offloaded:.2f} GB")
    
    return controlnet_states
```

---

## Code: Per-Block Residual Injection (Inside Block Loop)

```python
# Inside the main block loop, AFTER each block's forward pass
for block_idx, block in enumerate(self.blocks):
    # ... existing block forward code ...
    x = block(x, ...)
    
    # ADD: Uni3C residual injection
    if uni3c_controlnet_states is not None and block_idx < len(uni3c_controlnet_states):
        residual = uni3c_controlnet_states[block_idx]
        
        # Log first block at key steps (first, 25%, 50%, 75%, last) to catch drift
        if block_idx == 0:
            current_step = offload.shared_state.get("step_no", 0)
            first_active_step = int(uni3c_data["start"] * max_steps)
            last_active_step = int(uni3c_data["end"] * max_steps) - 1
            
            # Log at first step, every 25%, and last step
            steps_to_log = {first_active_step}
            for pct in [0.25, 0.5, 0.75]:
                steps_to_log.add(int(first_active_step + pct * (last_active_step - first_active_step)))
            steps_to_log.add(last_active_step)
            
            if current_step in steps_to_log:
                pct_done = (current_step - first_active_step) / max(1, last_active_step - first_active_step) * 100
                print(f"[UNI3C] Step {current_step}/{max_steps} ({pct_done:.0f}% of Uni3C window): Applying residual")
                print(f"[UNI3C]   residual shape: {residual.shape}")
                print(f"[UNI3C]   residual mean: {residual.mean().item():.6f}, std: {residual.std().item():.6f}")
                print(f"[UNI3C]   residual min: {residual.min().item():.6f}, max: {residual.max().item():.6f}")
        
        # Apply residual (only to original sequence length)
        x[:, :self.original_seq_len] += residual.to(x) * uni3c_data["controlnet_weight"]
```

---

## Code: Layer 5 + 6 Logging

Already embedded in the code above. Summary:

**Layer 5 (Per-Step):**
```python
if current_step_no == 0:
    print(f"[UNI3C] model.forward: Uni3C data present")
    print(f"[UNI3C]   controlnet loaded: {uni3c_data.get('controlnet') is not None}")
    print(f"[UNI3C]   render_latent shape: {uni3c_data['render_latent'].shape}")
    print(f"[UNI3C]   step window: {uni3c_data['start']*100:.0f}% - {uni3c_data['end']*100:.0f}%")
```

**Layer 6 (Per-Block, at key steps):**
```python
print(f"[UNI3C] Step {current_step}/{max_steps}: Applying residual")
print(f"[UNI3C]   residual mean: {residual.mean().item():.6f}, std: {residual.std().item():.6f}")
print(f"[UNI3C]   residual min: {residual.min().item():.6f}, max: {residual.max().item():.6f}")
```

**End Summary (after all steps complete):**

Add a summary log after the denoising loop completes (or use a counter to detect last step):

```python
# Track Uni3C usage across generation (add as instance variable or pass through)
if not hasattr(self, '_uni3c_stats'):
    self._uni3c_stats = {"steps_applied": 0, "blocks_per_step": 0}

if uni3c_controlnet_states is not None:
    self._uni3c_stats["steps_applied"] += 1
    self._uni3c_stats["blocks_per_step"] = len(uni3c_controlnet_states)

# At end of generation (detect by step == max_steps - 1)
if current_step_no == max_steps - 1 and hasattr(self, '_uni3c_stats'):
    stats = self._uni3c_stats
    print(f"[UNI3C] ========== GENERATION COMPLETE ==========")
    print(f"[UNI3C] Uni3C applied to {stats['steps_applied']} steps × {stats['blocks_per_step']} blocks")
    print(f"[UNI3C] Total residual injections: {stats['steps_applied'] * stats['blocks_per_step']}")
    del self._uni3c_stats  # Clean up for next generation
```

---

## Watchouts

### 1. `temb` Shape with Diffusion-Forcing

Wan2GP sometimes has `t.dim() == 2` (`_flag_df` for diffusion-forcing). In that case, `e` might be repeated differently.

**Guard**:
```python
if temb.dim() == 1:
    temb = temb.unsqueeze(0)  # Add batch dim
assert temb.dim() == 2, f"Expected temb [B, 5120], got {temb.shape}"
```

### 2. `original_seq_len`

The residual should only be applied to the original token sequence, not any padded/extended tokens. Check if `self.original_seq_len` exists; if not, you may need to track it.

### 3. Offload Device

Verify `self.main_device` and `self.offload_device` exist on the model. If not, use the device of the input tensor.

### 4. Step counter source ✅ Verified

Step counters are set in `Wan2GP/models/wan/modules/model.py` around lines 1578-1579:
- `offload.shared_state["step_no"]` = current step number
- `offload.shared_state["max_steps"]` = total steps

The code snippets in this doc use these correct keys.

### 5. VRAM During Controlnet Forward

The controlnet forward temporarily needs VRAM for all 20 blocks' outputs. If VRAM is tight, this could OOM. The offload flag helps but doesn't eliminate the peak.

---

## Phase Gate

Before moving to Phase 4, verify:

1. Run a generation with hardcoded `uni3c_data` (no headless wiring yet)
2. Check logs show `[UNI3C]` at Layer 5 and 6
3. Output should be visually different from non-Uni3C generation

---

## Next Phase

→ [Phase 4: Headless Param Wiring](./PHASE_4_HEADLESS_WIRING.md)

