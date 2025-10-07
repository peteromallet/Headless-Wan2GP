# Travel Orchestrator Run Comparison - COMPLETE ANALYSIS

**Analysis Date:** 2025-10-07
**Log File:** `debug_20251007_044629.log`
**Document Created By:** Manual chunk-by-chunk log analysis

This document compares the last 2 completed travel orchestrator runs to identify ALL major differences in parameters, implementation, LoRA application, and phase_config.

---

## Executive Summary

**CRITICAL FINDING:** The two runs are **COMPLETELY DIFFERENT** in their core configuration:

- **Run 1 (2c40e95b)**: NO phase_config, NO LoRAs, simple 6-step generation, NO VLM prompt enhancement
- **Run 2 (5b52c2bc)**: FULL phase_config with **3 phases**, **2 LoRAs** applied via semicolon multipliers, VLM-enhanced prompts

These are **NOT comparable runs** - they test entirely different generation workflows.

---

## Run 1: Orchestrator 2c40e95b (Simple Mode - NO phase_config)

### Overview
- **Run ID:** 20251007044548003
- **Orchestrator Task:** 2c40e95b-b57d-497f-ae67-cf348a336eee
- **Stitch Task:** c963d189-f046-470d-b77c-dc8b4e309c82
- **Number of Segments:** 2
- **Generation Mode:** `batch` (NOT timeline)
- **Advanced Mode:** `False`
- **Enhance Prompt:** `False` (NO VLM)

### Global Configuration (Run 1)
- **seed_base:** 789
- **model_name:** `lightning_baseline_2_2_2`
- **resolution:** 768x576
- **steps:** 6 (flat, no phases)
- **amount_of_motion:** 0.5
- **segment_frames_expanded:** [61, 61]
- **frame_overlap_expanded:** [6]
- **phase_config:** **❌ NONE** - No phase_config present
- **lora_multipliers:** **❌ NONE** - No LoRAs applied

### Segment 0: Task 415b7199-4b11-4eba-aaa4-f31fee06ac10

| Parameter | Value |
|-----------|-------|
| **Segment Index** | 0 |
| **Seed** | 789 |
| **Model** | `lightning_baseline_2_2_2` |
| **Resolution** | 768x576 |
| **Steps** | 6 |
| **Amount of Motion** | 0.5 |
| **Is First Segment** | True |
| **Is Last Segment** | False |
| **Frame Overlap (next)** | 6 |
| **Frame Overlap (prev)** | 0 |
| **Segment Frames Target** | 61 |
| **base_prompt** | `""` (empty) |
| **negative_prompt** | `"cut, blurry, fade"` |
| **phase_config** | ❌ None |
| **lora_names** | ❌ None |
| **lora_multipliers** | ❌ None |
| **use_causvid_lora** | False |
| **apply_reward_lora** | False |
| **use_lighti2x_lora** | False |
| **use_styleboost_loras** | False |

### Segment 1: Task 53586086-c122-40ff-9f70-61f3b4ff46ca

| Parameter | Value |
|-----------|-------|
| **Segment Index** | 1 |
| **Seed** | 789 |
| **Model** | `lightning_baseline_2_2_2` |
| **Resolution** | 768x576 |
| **Steps** | 6 |
| **Amount of Motion** | 0.5 |
| **Is First Segment** | False |
| **Is Last Segment** | True |
| **Frame Overlap (next)** | 0 |
| **Frame Overlap (prev)** | 6 |
| **Segment Frames Target** | 61 |
| **base_prompt** | `""` (empty) |
| **negative_prompt** | `"cut, blurry, fade"` |
| **phase_config** | ❌ None |
| **lora_names** | ❌ None |
| **lora_multipliers** | ❌ None |
| **use_causvid_lora** | False |
| **apply_reward_lora** | False |
| **use_lighti2x_lora** | False |
| **use_styleboost_loras** | False |

### Run 1 Log Confirmation
```
[04:46:40] INFO HEADLESS [Task 415b7199] [DEBUG] Checking for phase_config.
Keys in full_orchestrator_payload: ['steps', 'run_id', 'shot_id', 'seed_base', 'model_name',
'base_prompt', 'advanced_mode', 'apply_causvid', 'enhance_prompt', 'openai_api_key']
```

**NO phase_config key present** - Run 1 uses standard single-guidance generation.

---

## Run 2: Orchestrator 5b52c2bc (Advanced Mode - WITH phase_config)

### Overview
- **Run ID:** 20251007050206112
- **Orchestrator Task:** 5b52c2bc-9771-47c4-b20f-16e6e5046328
- **Stitch Task:** f4dcbe64-6b6c-449e-9823-6a3f56484a4f
- **Number of Segments:** 2
- **Generation Mode:** `timeline` (NOT batch)
- **Advanced Mode:** `True`
- **Enhance Prompt:** `True` (VLM ENABLED)

### Global Configuration (Run 2)
- **seed_base:** 789
- **model_name:** `lightning_baseline_2_2_2`
- **resolution:** 768x576
- **steps:** 6 (distributed across 3 phases as [2, 2, 2])
- **amount_of_motion:** ❌ NOT SET (phase_config takes precedence)
- **segment_frames_expanded:** [53, 45] ← DIFFERENT from Run 1
- **frame_overlap_expanded:** [6]
- **flow_shift:** 5 (Run 1 had no flow_shift)
- **sample_solver:** `euler`

### phase_config Details (Run 2)

```json
{
  "phases": [
    {
      "phase": 1,
      "guidance_scale": 3,
      "loras": [
        {
          "url": "https://huggingface.co/lightx2v/Wan2.2-Lightning/resolve/main/Wan2.2-T2V-A14B-4steps-lora-250928/high_noise_model.safetensors",
          "multiplier": "0.75"
        }
      ]
    },
    {
      "phase": 2,
      "guidance_scale": 1,
      "loras": [
        {
          "url": "https://huggingface.co/lightx2v/Wan2.2-Lightning/resolve/main/Wan2.2-T2V-A14B-4steps-lora-250928/high_noise_model.safetensors",
          "multiplier": "1.0"
        }
      ]
    },
    {
      "phase": 3,
      "guidance_scale": 1,
      "loras": [
        {
          "url": "https://huggingface.co/lightx2v/Wan2.2-Lightning/resolve/main/Wan2.2-T2V-A14B-4steps-lora-250928/low_noise_model.safetensors",
          "multiplier": "1.0"
        }
      ]
    }
  ],
  "flow_shift": 5,
  "num_phases": 3,
  "sample_solver": "euler",
  "steps_per_phase": [2, 2, 2],
  "model_switch_phase": 2
}
```

### LoRA Configuration (Run 2)

**lora_names:**
```python
[
  'https://huggingface.co/lightx2v/Wan2.2-Lightning/resolve/main/Wan2.2-T2V-A14B-4steps-lora-250928/high_noise_model.safetensors',
  'https://huggingface.co/lightx2v/Wan2.2-Lightning/resolve/main/Wan2.2-T2V-A14B-4steps-lora-250928/low_noise_model.safetensors'
]
```

**lora_multipliers (WGP Format - CRITICAL!):**
```python
['0.75;1.0;0', '0;0;1.0']
```

**Parsed Meaning:**
- **LoRA 0 (high_noise_model):** Phase 1 = 0.75, Phase 2 = 1.0, Phase 3 = 0 (off)
- **LoRA 1 (low_noise_model):** Phase 1 = 0 (off), Phase 2 = 0 (off), Phase 3 = 1.0

**additional_loras (Initial Values):**
```python
{
  'https://huggingface.co/lightx2v/Wan2.2-Lightning/resolve/main/Wan2.2-T2V-A14B-4steps-lora-250928/low_noise_model.safetensors': 0,
  'https://huggingface.co/lightx2v/Wan2.2-Lightning/resolve/main/Wan2.2-T2V-A14B-4steps-lora-250928/high_noise_model.safetensors': 0.75
}
```

### Multi-Phase Guidance (Run 2)

- **guidance_scale** (Phase 1): 3
- **guidance2_scale** (Phase 2): 1
- **guidance3_scale** (Phase 3): 1
- **guidance_phases:** 3
- **switch_threshold:** 917.4817575573454 (timestep for Phase 1→2)
- **switch_threshold2:** 663.2310443522647 (timestep for Phase 2→3)
- **model_switch_phase:** 2

**Generated Timesteps:**
```
['1000.0', '952.4', '882.5', '769.7', '556.8', '5.0']
```

### VLM-Enhanced Prompts (Run 2 ONLY)

Run 2 uses VLM (Qwen2.5-VL-7B-Instruct) to generate transition prompts:

**Segment 0 base_prompt:**
```
"The character begins walking towards the camera in the shallow water, gradually transitioning
into a run as the camera zooms in closer, capturing his determined expression against the
backdrop of the serene mountainous landscape."
```

**Segment 1 base_prompt:**
```
"The camera subtly zooms in on the character's face, intensifying their expression as they
appear to be speaking or reacting to something off-screen, with the mountainous backdrop
remaining static but slightly blurred to emphasize the focus on the character's evolving emotion."
```

### Segment 0: Task 6700be57-9994-40c1-a468-337e6a249e3e

| Parameter | Value |
|-----------|-------|
| **Segment Index** | 0 |
| **Seed** | 789 |
| **Model** | `lightning_baseline_2_2_2` |
| **Resolution** | 768x576 |
| **num_inference_steps** | 6 |
| **phase_config** | ✅ **3 phases** |
| **steps_per_phase** | [2, 2, 2] |
| **guidance_scale** | 3 (Phase 1) |
| **guidance2_scale** | 1 (Phase 2) |
| **guidance3_scale** | 1 (Phase 3) |
| **flow_shift** | 5 |
| **sample_solver** | euler |
| **Is First Segment** | True |
| **Is Last Segment** | False |
| **Frame Overlap (next)** | 6 |
| **Frame Overlap (prev)** | 0 |
| **Segment Frames Target** | 53 |
| **base_prompt** | ✅ VLM-enhanced (see above) |
| **negative_prompt** | `"cut, blurry, fade"` |
| **lora_names** | ✅ 2 LoRAs |
| **lora_multipliers** | ✅ `['0.75;1.0;0', '0;0;1.0']` |
| **switch_threshold** | 917.4817575573454 |
| **switch_threshold2** | 663.2310443522647 |
| **model_switch_phase** | 2 |

### Segment 1: Task 2fb5f646-1513-4eef-bbfd-3b9bf51ababf

| Parameter | Value |
|-----------|-------|
| **Segment Index** | 1 |
| **Seed** | 789 |
| **Model** | `lightning_baseline_2_2_2` |
| **Resolution** | 768x576 |
| **num_inference_steps** | 6 |
| **phase_config** | ✅ **3 phases** |
| **steps_per_phase** | [2, 2, 2] |
| **guidance_scale** | 3 (Phase 1) |
| **guidance2_scale** | 1 (Phase 2) |
| **guidance3_scale** | 1 (Phase 3) |
| **flow_shift** | 5 |
| **sample_solver** | euler |
| **Is First Segment** | False |
| **Is Last Segment** | True |
| **Frame Overlap (next)** | 0 |
| **Frame Overlap (prev)** | 6 |
| **Segment Frames Target** | 45 |
| **base_prompt** | ✅ VLM-enhanced (see above) |
| **negative_prompt** | `"cut, blurry, fade"` |
| **lora_names** | ✅ 2 LoRAs |
| **lora_multipliers** | ✅ `['0.75;1.0;0', '0;0;1.0']` |
| **switch_threshold** | 917.4817575573454 |
| **switch_threshold2** | 663.2310443522647 |
| **model_switch_phase** | 2 |

### Run 2 Log Confirmation
```
[05:06:42] INFO TRAVEL [Task 5b52c2bc] phase_config detected in orchestrator - parsing comprehensive phase configuration
[05:06:42] DEBUG HEADLESS [Task 5b52c2bc] Generated timesteps for phase_config: ['1000.0', '952.4', '882.5', '769.7', '556.8', '5.0']
[05:06:42] INFO HEADLESS [Task 5b52c2bc] phase_config parsed: 3 phases, steps=[2, 2, 2], thresholds=[917.4817575573454, 663.2310443522647], 2 LoRAs, lora_multipliers=['0.75;1.0;0', '0;0;1.0']
[05:06:42] INFO TRAVEL [Task 5b52c2bc] phase_config parsed: 3 phases, steps=6, 2 LoRAs, lora_multipliers=['0.75;1.0;0', '0;0;1.0']
```

**phase_config successfully parsed and applied** - Run 2 uses multi-phase generation with dynamic LoRA switching.

---

## Side-by-Side Comparison

| Feature | Run 1 (2c40e95b) | Run 2 (5b52c2bc) |
|---------|------------------|------------------|
| **Run ID** | 20251007044548003 | 20251007050206112 |
| **Generation Mode** | `batch` | `timeline` |
| **Advanced Mode** | False | **True** |
| **Enhance Prompt** | False | **True** (VLM) |
| **phase_config** | ❌ None | ✅ **3 phases** |
| **LoRAs Applied** | ❌ None | ✅ **2 LoRAs** |
| **lora_multipliers** | ❌ None | ✅ `['0.75;1.0;0', '0;0;1.0']` |
| **Guidance** | Single value | **Multi-phase** (3, 1, 1) |
| **flow_shift** | Not set | **5** |
| **sample_solver** | Not specified | **euler** |
| **Segment Frames** | [61, 61] | **[53, 45]** |
| **Base Prompts** | Empty strings | **VLM-enhanced** |
| **Steps Distribution** | 6 (flat) | **[2, 2, 2]** (phased) |
| **Model Switching** | No | **Yes** (phase 2) |
| **Timestep Thresholds** | No | **Yes** (917.48, 663.23) |

---

## Image References (Identical)

Both runs use the same 3 image URLs:

1. `c961c478-19b8-4084-9590-7a4682ac6f0f-u1_1a92ba99-df24-4070-85bb-f11d72b5c3e8.jpeg`
2. `e5f05af3-ab18-4e36-b1f6-84d2d02669e7-u2_bfe36092-2337-4be1-ac18-839a98c525ca.jpeg`
3. `e5f05af3-ab18-4e36-b1f6-84d2d02669e7-u2_bfe36092-2337-4be1-ac18-839a98c525ca.jpeg` (duplicate of #2)

**Image assignment is identical** between runs.

---

## Processing Flow Comparison

### Run 1 Processing Flow
1. **Orchestrator** creates 2 segment tasks (no phase_config)
2. **Seg0 (415b7199)** - Basic I2V generation
   - 6 flat steps
   - No LoRAs
   - No VLM prompt
3. **Seg1 (53586086)** - Basic I2V generation
   - 6 flat steps
   - No LoRAs
   - No VLM prompt
4. **Stitch (c963d189)** - Combines segments with 6-frame overlap

### Run 2 Processing Flow
1. **Orchestrator** parses phase_config, creates 2 segment tasks
2. **Seg0 (6700be57)** - Advanced multi-phase generation
   - Phase 1 (steps 0-1): guidance=3, high_noise_lora @ 0.75
   - Phase 2 (steps 2-3): guidance=1, high_noise_lora @ 1.0
   - Phase 3 (steps 4-5): guidance=1, low_noise_lora @ 1.0
   - VLM-enhanced prompt
3. **Seg1 (2fb5f646)** - Advanced multi-phase generation
   - Phase 1 (steps 0-1): guidance=3, high_noise_lora @ 0.75
   - Phase 2 (steps 2-3): guidance=1, high_noise_lora @ 1.0
   - Phase 3 (steps 4-5): guidance=1, low_noise_lora @ 1.0
   - VLM-enhanced prompt
4. **Stitch (f4dcbe64)** - Combines segments with 6-frame overlap

---

## Critical Differences Summary

### 1. **phase_config Presence**
- **Run 1:** NO phase_config → standard single-pass generation
- **Run 2:** FULL phase_config → 3-phase multi-guidance generation

### 2. **LoRA Application**
- **Run 1:** NO LoRAs applied
- **Run 2:** 2 LoRAs with dynamic switching:
  - `high_noise_model`: 0.75 → 1.0 → 0 (off)
  - `low_noise_model`: 0 → 0 → 1.0 (on in final phase)

### 3. **lora_multipliers Format**
- **Run 1:** N/A (no LoRAs)
- **Run 2:** `['0.75;1.0;0', '0;0;1.0']` ← **THIS IS THE SEMICOLON FORMAT WE'VE BEEN INVESTIGATING!**

### 4. **Prompt Enhancement**
- **Run 1:** Empty base_prompts, no VLM
- **Run 2:** VLM-generated descriptive prompts via Qwen2.5-VL-7B-Instruct

### 5. **Guidance Strategy**
- **Run 1:** Single guidance value throughout
- **Run 2:** Multi-phase guidance (3 → 1 → 1)

### 6. **Segment Frame Counts**
- **Run 1:** [61, 61] = 122 total frames before overlap
- **Run 2:** [53, 45] = 98 total frames before overlap

### 7. **Generation Mode**
- **Run 1:** `batch`
- **Run 2:** `timeline`

---

## Implications for phase_config Debugging

This comparison reveals that:

1. **Run 2 is testing phase_config with 3 phases** - this is the **3-sampler** configuration that **works**
2. **Run 1 has no phase_config** - this is the **control/baseline** run
3. **The semicolon multipliers `['0.75;1.0;0', '0;0;1.0']` are successfully parsed and applied in Run 2**
4. **There is NO 2-phase run in this log** - we need to find a 2-phase run that failed to compare

---

## Questions for Further Investigation

1. **Where is the 2-phase run that's failing?** This log shows a 0-phase (Run 1) and 3-phase (Run 2) comparison.
2. **Did the 2-phase format use `['1.0;0', '0;1.0']`?** We need to find logs showing 2-phase multipliers being converted to floats.
3. **Is the issue specific to 2-phase configs?** Run 2 shows 3-phase works correctly.

---

## Conclusion

These two runs are testing **completely different workflows**:

- **Run 1** = Baseline (no phase_config, no LoRAs, batch mode)
- **Run 2** = Advanced (3-phase config, 2 LoRAs, timeline mode, VLM prompts)

**Run 2 demonstrates that phase_config with 3 phases WORKS CORRECTLY** - the semicolon multipliers are parsed and applied successfully. To debug the 2-phase failure, we need to locate a log showing 2-phase `lora_multipliers: ['1.0;0', '0;1.0']` being incorrectly converted to `[1.0, 0.0]`.

---

**Document saved:** `/workspace/lol/Headless-Wan2GP/run_comparison_complete.md`
