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

**Wan2GP-specific nuance**: Wan2GP’s forward operates on an `x_list` (multi-stream) and calls `block(...)` inside an inner loop.  
So the residual injection must happen **inside that inner loop**, right after `x_list[i] = block(...)`, not once outside.

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
    # Use (max_steps - 1) so the last step maps to 1.0 when end_percent=1.0
    current_step_percentage = current_step_no / max(1, (max_steps - 1))
    
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
    
    # Wan2GP doesn't expose self.main_device/self.offload_device on the model.
    # Use the model's device (patch embedding weights) as the "main" device.
    main_device = self.patch_embedding.weight.device
    offload_device = torch.device("cpu")

    # Move controlnet to main device if offloaded
    if uni3c_data.get("offload", True):
        controlnet = controlnet.to(main_device)
    
    # Ensure render_latent matches hidden_states shape
    # (temporal resampling if needed - see Phase 2)
    
    # Run controlnet forward
    controlnet_states = controlnet(
        render_latent=render_latent.to(main_device, controlnet.dtype),
        render_mask=uni3c_data.get("render_mask"),
        camera_embedding=uni3c_data.get("camera_embedding"),
        temb=temb.to(main_device),
        out_device=offload_device if uni3c_data.get("offload") else main_device
    )
    
    # Log VRAM after controlnet forward
    if torch.cuda.is_available():
        vram_after = torch.cuda.memory_allocated() / 1024**3
        print(f"[UNI3C] VRAM after controlnet forward: {vram_after:.2f} GB")
    
    # Offload controlnet back if configured
    if uni3c_data.get("offload", True):
        controlnet.to(offload_device)
        if torch.cuda.is_available():
            vram_offloaded = torch.cuda.memory_allocated() / 1024**3
            print(f"[UNI3C] VRAM after offload: {vram_offloaded:.2f} GB")
    
    return controlnet_states
```

---

## Code: Per-Block Residual Injection (Inside Block Loop)

**Wan2GP Reality**: The block loop operates on `x_list` (multi-stream) with an inner loop. See `model.py` ~lines 1759-1779.

**Injection site**: After the inner stream loop completes (line ~1779), before skip-cache logic (line ~1781).

```python
# ACTUAL Wan2GP block loop structure (simplified for clarity):
if any(x_should_calc):
    for block_idx, block in enumerate(self.blocks):
        offload.shared_state["layer"] = block_idx
        # ... callback / interrupt checks ...

        # SLG path (single stream)
        if slg_layers is not None and block_idx in slg_layers:
            x_list[0] = block(x_list[0], ...)
        else:
            # Multi-stream inner loop
            for i, (x, context, ...) in enumerate(zip(x_list, context_list, ...)):
                if should_calc:
                    x_list[i] = block(x, context=context, ..., **kwargs)
                    del x
            context = hints = None

        # ========== ADD UNI3C INJECTION HERE ==========
        if uni3c_controlnet_states is not None and block_idx < len(uni3c_controlnet_states):
            residual = uni3c_controlnet_states[block_idx]
            ref_images_count = kwargs.get("ref_images_count", 0)
            
            # Apply to ALL streams in x_list
            for i, x in enumerate(x_list):
                x_start = ref_images_count
                apply_len = min(x.shape[1] - x_start, residual.shape[1])
                if apply_len > 0:
                    x_list[i][:, x_start:x_start + apply_len] += (
                        residual[:, :apply_len].to(x) * uni3c_data["controlnet_weight"]
                    )
            
            # Log at first block of key steps
            if block_idx == 0:
                current_step = offload.shared_state.get("step_no", 0)
                if current_step == 0 or current_step == max_steps - 1:
                    print(f"[UNI3C] Step {current_step}/{max_steps}: Applying block {block_idx} residual")
                    print(f"[UNI3C]   residual shape: {residual.shape}, mean: {residual.mean().item():.4f}")
        # ========== END UNI3C INJECTION ==========

    # (existing skip-cache logic follows at ~line 1781)
```

**Key points**:
1. Injection happens **after** all streams are updated by the block
2. Must iterate over `x_list` to apply to ALL streams (not just stream 0)
3. Use `ref_images_count` to skip prefix tokens
4. Clamp by `min(x_len, residual_len)` to avoid shape crashes

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

The residual should only be applied to the original token sequence, not any padded/extended tokens.

**Wan2GP reality**: `model.py` does **not** define `self.original_seq_len`.  
Instead, Wan2GP tracks prefix token counts via `ref_images_count` in kwargs and sometimes trims to `real_seq` in special paths (e.g. steadydancer).

**Recommended MVP rule**:
- Inject starting at `ref_images_count` (skip prefix tokens)
- Clamp by `min(x_len, residual_len)` to avoid shape crashes

### 3. Offload Device

Wan2GP doesn’t expose `self.main_device` / `self.offload_device` on the transformer. Use:
- main device: `self.patch_embedding.weight.device`
- offload device: `torch.device("cpu")` (or a configurable device if you add one)

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

