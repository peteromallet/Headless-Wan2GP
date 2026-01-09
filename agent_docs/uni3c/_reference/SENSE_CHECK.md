# Sense-Check Results

← [Back to Start](../STARTING_POINT_AND_STATUS.md)

> This document validates our Uni3C plan against source implementations.
> Use this when verifying assumptions about architecture or behavior.

---

## ✅ Verified Against ComfyUI-WanVideoWrapper (Kijai's implementation)

| Claim in Plan | Source Truth | Status |
|---------------|--------------|--------|
| **Uni3C has 20 transformer blocks** | `controlnet_cfg["num_layers"] = 20` in `nodes.py:67` | ✅ Correct |
| **dim = 1024, time_embed_dim = 5120** | `"dim": 1024, "time_embed_dim": 5120` in `nodes.py:63-64` | ✅ Correct |
| **proj_out outputs dimension 5120** | `nn.Linear(self.dim, 5120)` in `controlnet.py:317` | ✅ Correct |
| **Patch embedding is Conv3D(1,2,2)** | `self.patch_size = (1, 2, 2)` in `controlnet.py:290` | ✅ Correct |
| **Accepts render_latent + temb** | `forward(render_latent, render_mask, camera_embedding, temb, ...)` in `controlnet.py:337` | ✅ Correct |
| **Returns per-block controlnet_states list** | `controlnet_states.append(self.proj_out[i](hidden_states)...)` in `controlnet.py:367` | ✅ Correct |
| **Per-block addition in main transformer** | `x[:, :self.original_seq_len] += uni3c_controlnet_states[b].to(x) * uni3c_data["controlnet_weight"]` in `model.py:3276` | ✅ Correct |
| **Step-percent gating logic** | `if (uni3c_data["start"] <= current_step_percentage <= uni3c_data["end"])` in `model.py:3153` | ✅ Correct |
| **Offload support per step** | `if uni3c_data["offload"]: self.uni3c_controlnet.to(self.offload_device)` in `model.py:3164` | ✅ Correct |
| **render_mask / camera_embedding NOT exposed** | `raise NotImplementedError("render_mask is not implemented...")` in `nodes.py:166` | ✅ Correct (plan says defer these) |

---

## ✅ Verified Against Wan2GP Ground Truth

| Claim in Plan | Wan2GP Reality | Status |
|---------------|----------------|--------|
| **Injection point is model.py block loop** | Main loop at `Wan2GP/models/wan/modules/model.py:1759+` (`for block_idx, block in enumerate(self.blocks)`) | ✅ Correct |
| **Timestep embedding `e` = time_embedding output** | `e = self.time_embedding(sinusoidal_embedding_1d(...))` at line 1597 | ✅ Correct |
| **`e` has dimension = self.dim (5120 for 14B)** | Defined in model init; matches Uni3C `time_embed_dim = 5120` | ✅ Correct |
| **Do NOT add new architecture string** | SVI presets keep `"architecture": "i2v_2_2"` and use flags | ✅ Correct approach |
| **Param precedence: task > defaults** | `headless_wgp.py:_resolve_parameters()` confirms task overrides model defaults | ✅ Correct |
| **Must add params to wgp.generate_video() signature** | Current signature is explicit (not `**kwargs` passthrough); **unknown params are silently filtered** by `inspect.signature()` checks at call sites | ⚠️ Correct – but silent failure mode |
| **VACE uses vace_context, separate from Uni3C** | VACE path uses `self.vace_patch_embedding` and `vace_context` | ✅ Correct – Uni3C is independent |

---

## ⚠️ Corrections Made to Original Plan

| Original Claim | Correction |
|----------------|------------|
| "Implement in wgp.py sampling loop" | **Wrong**: Real loop is in `models/wan/modules/model.py` |
| "Add architecture `i2v_2_2_uni3c`" | **Wrong**: Wan2GP doesn't discover architectures this way |
| "Checkpoint at Kijai/WanVideo_comfy" | **Uncertain**: That repo is a wrapper; actual checkpoint compatibility unknown |
| "Python will throw if signature doesn't accept params" | **Wrong**: Wan2GP uses `inspect.signature()` filtering at call sites (`wgp.py:6714-6718`), so **unknown params are silently dropped**. This is a silent failure mode, not a crash. |

---

## ⚠️ Critical Implementation Details Confirmed

### 1. temb dimension must match

Uni3C expects `time_embed_dim = 5120`, which matches Wan2GP's `e = self.time_embedding(...)` output dimension for the 14B model.

### 2. Injection happens AFTER block forward, not before

```python
# AFTER block runs
if uni3c_controlnet_states is not None and b < len(uni3c_controlnet_states):
    x[:, :self.original_seq_len] += uni3c_controlnet_states[b].to(x) * uni3c_data["controlnet_weight"]
```

### 3. Latent temporal resampling is handled

The Comfy implementation does temporal interpolation if guide frames don't match:

```python
if hidden_states.shape[2] != render_latent.shape[2]:
    render_latent = nn.functional.interpolate(render_latent, size=(...), mode='trilinear')
```

### 4. Uni3C controlnet runs BEFORE the block loop

But outputs are applied inside it. This is the correct pattern.

### 5. Guide latent channel count may require padding

The Comfy wrapper explicitly pads 16→20 channels in a "T2V workaround":

```python
if hidden_states.shape[1] == 16:
    hidden_states = torch.cat([hidden_states, torch.zeros_like(hidden_states[:, :4])], dim=1)
```

**Action**: Check `in_channels` from the Uni3C checkpoint and implement padding if needed.

### 6. `temb` shape nuance for diffusion-forcing

Wan2GP sometimes has `t.dim() == 2` (`_flag_df`), and in that case it repeats `e0` later. Uni3C expects a 2D tensor (batch, 5120).

**Action**: Add a guard for `_flag_df` to avoid passing a flattened or mismatched embedding.

