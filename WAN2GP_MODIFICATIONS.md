# Wan2GP Modifications Reference

This document catalogs all changes made to the `Wan2GP/` folder since it was detached from the upstream repository ([deepbeepmeep/Wan2GP](https://github.com/deepbeepmeep/Wan2GP)).

**Baseline Commit:** `b54ad78` (August 5, 2025) - When Wan2GP was converted from submodule to regular files  
**Original Upstream Pinned:** Commit `026c2b0` (approximately v7.x)  
**Current Upstream:** v9.1 (as of January 2026)

---

## Table of Contents
1. [Overview](#overview)
2. [Critical Core File Modifications](#critical-core-file-modifications)
3. [New Model Handlers (models/ directory)](#new-model-handlers)
4. [Configuration Files (defaults/)](#configuration-files)
5. [Other Modified Files](#other-modified-files)
6. [Import Path Changes](#import-path-changes)
7. [Merge Strategy Recommendations](#merge-strategy-recommendations)

---

## Overview

### Categories of Changes
- **Core Python files modified:** 7 files
- **New model handler directories added:** 4 directories (`models/qwen/`, `models/wan/`, `models/flux/`, etc.)
- **New default configurations:** ~30 custom JSON configs
- **Modified configurations:** ~20 JSON files
- **Import path reorganization:** Changed from `wan.*` imports to `shared.*` and `models.*`

---

## Critical Core File Modifications

### 1. `wgp.py` - Main Entry Point
**Commits:** 40+ modifications
**Key Changes:**

#### Import Path Reorganization
```python
# BEFORE (upstream)
import wan
from wan.utils import notification_sound
from wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS
from wan.modules.attention import get_attention_modes

# AFTER (our version)
import numpy as np
import importlib
from shared.utils import notification_sound
from shared.utils.loras_mutipliers import preparse_loras_multipliers
from shared.match_archi import match_nvidia_architecture
from shared.attention import get_attention_modes
```

#### Argparse Conflict Prevention
Added protection for headless worker imports:
```python
print(f"[WGP_IMPORT_DEBUG] ===== wgp.py module loading started =====")
print(f"[WGP_IMPORT_DEBUG] Current working directory: {os.getcwd()}")
print(f"[WGP_IMPORT_DEBUG] sys.argv: {sys.argv}")
os.environ["GRADIO_LANG"] = "en"
```

#### New Functions Added
- `release_model()` - GPU memory cleanup
- `clear_gen_cache()` - Cache management
- `clean_image_list()` - Image processing utility

#### Settings Caching Removal
Removed WGP settings caching system entirely for headless compatibility.

#### Version Updates
```python
target_mmgp_version = "3.6.0"  # was 3.5.6
WanGP_version = "8.61"         # was 7.61  
settings_version = 2.35        # was 2.23
```

---

### 2. `wan/any2video.py` - Core Generation Logic
**Commits:** 6 modifications
**Key Changes:**

#### Debug Logging
Added extensive logging for switch_threshold debugging:
```python
print(f"[AUTO_QUANTIZE_LOG] WanAny2V init: quantizeTransformer={quantizeTransformer}")
print(f"[SWITCH_THRESHOLD_LOG] switch_threshold: {switch_threshold}")
print(f"[SWITCH_THRESHOLD_DEBUG] any2video.py starting generation loop:")
```

#### DPM++_SDE Sampler Support
```python
elif sample_solver == 'dpm++_sde':
    sample_scheduler = FlowDPMSolverMultistepScheduler(
        num_train_timesteps=self.num_train_timesteps,
        algorithm_type="sde-dpmsolver++",
        shift=1,
        use_dynamic_shifting=False)
```

---

### 3. `models/wan/any2video.py` - Extended Generation Features
**Commits:** 6 modifications
**Key Changes:**

#### Latent Noise Mask Support for VACE
```python
latent_noise_mask_strength = 0.0,  # New parameter
vid2vid_init_video = None,          # New parameter
vid2vid_init_strength = 0.7,        # New parameter
```

#### Vid2Vid Initialization
Added ~80 lines for video-to-video initialization:
```python
# Vid2vid initialization: Use provided video as starting point
if vid2vid_init_video is not None and vid2vid_init_strength < 1.0:
    # Load video frames, encode with VAE, blend with noise
```

#### NumPy UnboundLocalError Fix
```python
# Fixed: prevent UnboundLocalError for numpy in euler path
import numpy as np  # Added at module level
```

---

### 4. `models/qwen/qwen_main.py` - NEW FILE
**Purpose:** Qwen image generation model factory and pipeline

#### Key Features
- Model warmup to pre-compile CUDA kernels
- Two-pass hires fix workflow
- System prompt support for image editing
- Latent space upscaling

```python
class model_factory():
    def __init__(self, checkpoint_dir, model_filename, ...):
        # Warmup pass to prevent ~100 second delay on first generation
        print("🔥 Warming up Qwen model (compiling CUDA kernels)...")
```

---

### 5. `models/qwen/qwen_handler.py` - NEW FILE
**Purpose:** Qwen model handler with family-specific configuration

```python
class family_handler():
    @staticmethod
    def query_model_def(base_model_type, model_def):
        # Qwen-specific settings
        extra_model_def = {
            "image_outputs": True,
            "sample_solvers": [("Default", "default"), ("Lightning", "lightning")],
            "guidance_max_phases": 1,
        }
```

---

## New Model Handlers

### `Wan2GP/models/` Directory Structure
Entire directory was **added** (not in upstream structure):

```
models/
├── __init__.py
├── flux/              # Flux model handler (14 files)
│   ├── flux_handler.py
│   ├── flux_main.py
│   └── modules/
├── hyvideo/           # Hunyuan Video handler (30+ files)
│   ├── hunyuan_handler.py
│   └── modules/
├── ltx_video/         # LTX Video handler (20+ files)
│   ├── ltxv_handler.py
│   └── models/
├── qwen/              # Qwen Image handler (5 files)
│   ├── qwen_handler.py
│   ├── qwen_main.py
│   ├── pipeline_qwenimage.py
│   └── transformer_qwenimage.py
└── wan/               # Wan model handler (40+ files)
    ├── wan_handler.py
    ├── any2video.py
    └── modules/
```

**Note:** This is a reorganization where model code was moved from `wan/`, `flux/`, etc. into a centralized `models/` directory with handler classes.

---

## Configuration Files

### New Custom Configurations (defaults/)

#### Lightning Baseline Configs (for fast inference)
| File | Purpose |
|------|---------|
| `lightning_baseline_2_2_2.json` | 2-phase lightning, 2-2-2 steps |
| `lightning_baseline_3_3.json` | 2-phase lightning, 3-3 steps |
| `wan_2_2_i2v_lightning_baseline_2_2_2.json` | I2V specific, 2-phase |
| `wan_2_2_i2v_lightning_baseline_3_3.json` | I2V specific, 3-phase |
| `wan_2_2_vace_lightning_baseline_2_2_2.json` | VACE specific, 3-phase |
| `wan_2_2_vace_lightning_baseline_3_3.json` | VACE specific, 2-phase |

#### VACE Cocktail Configurations
| File | Purpose |
|------|---------|
| `vace_14B_cocktail.json` | Base cocktail config |
| `vace_14B_cocktail_2_2.json` | Wan 2.2 cocktail |
| `vace_14B_fake_cocktail_2_2.json` | Modified architecture (t2v base) |
| `vace_fun_14B_cocktail_lightning.json` | Lightning acceleration |
| `vace_fun_14B_cocktail_lightning_3phase.json` | 3-phase lightning |
| `vace_fun_14B_cocktail_lightning_3phase_light_distill.json` | Light distillation |

#### Qwen Configurations
| File | Purpose |
|------|---------|
| `qwen_image_20B.json` | Base Qwen image generation |
| `qwen_image_edit_20B.json` | Qwen image editing with inpainting |
| `qwen_image_hires.json` | Two-pass hires workflow |

#### Test/Reward Configurations
| File | Purpose |
|------|---------|
| `test_reward_p1_0_25.json` | Reward LoRA at 0.25 strength |
| `test_reward_p1_0_5.json` | Reward LoRA at 0.5 strength |
| `test_reward_p1_0_75.json` | Reward LoRA at 0.75 strength |
| `test_reward_p1_1_0.json` | Reward LoRA at 1.0 strength |

### Modified Configurations
Key parameters changed in existing configs:

```json
// Common changes across multiple configs:
{
  "guidance_phases": 2,           // Added for multi-phase support
  "switch_threshold": 558,        // Added for stage transitions
  "model_switch_phase": 1,        // Added for model switching
  "lock_guidance_phases": true    // Added for phase locking
}
```

---

## Other Modified Files

### Preprocessing
| File | Change |
|------|--------|
| `preprocessing/__init__.py` | New file for module init |
| `preprocessing/extract_vocals.py` | Modified |
| `preprocessing/face_preprocessor.py` | Modified |
| `preprocessing/matanyone/app.py` | Modified |
| `preprocessing/speakers_separator.py` | Modified |

### Shared Utilities
New `shared/` directory added with utilities extracted from `wan/`:

```
shared/
├── RGB_factors.py
├── attention.py               # get_attention_modes
├── extract_lora.py
├── gradio/gallery.py
├── match_archi.py             # match_nvidia_architecture
├── sage2_core.py
└── utils/
    ├── audio_video.py         # extract_audio_tracks, save_video
    ├── basic_flowmatch.py
    ├── fm_solvers.py
    ├── loras_mutipliers.py
    ├── prompt_extend.py
    ├── prompt_parser.py
    ├── qwen_vl_utils.py
    ├── utils.py               # Core utilities
    └── vace_preprocessor.py
```

### Postprocessing
| File | Change |
|------|--------|
| `postprocessing/mmaudio/data/av_utils.py` | Modified |
| `postprocessing/mmaudio/utils/logger.py` | Modified (likely non-interactive backend) |

### Core wan/ Directory
| File | Change |
|------|--------|
| `wan/any2video.py` | Debug logging, switch_threshold |
| `wan/modules/attention.py` | Modified |
| `wan/modules/t5.py` | Modified |
| `wan/modules/vae.py` | Modified |
| `wan/utils/prompt_extend.py` | Modified |

---

## Import Path Changes

### Pattern Summary
| Original Import | New Import |
|----------------|------------|
| `from wan.utils import X` | `from shared.utils import X` |
| `from wan.modules.attention import Y` | `from shared.attention import Y` |
| `from wan.configs import Z` | Direct JSON loading or `from models.wan.configs` |
| `import wan` | Removed - use specific handlers |

### Files Affected by Import Changes
- `wgp.py` (main entry)
- All files in `models/` subdirectories
- `headless_wgp.py` (in project root)
- `headless_model_management.py` (in project root)

---

## Merge Strategy Recommendations

### High Priority - Preserve Our Changes
1. **`wgp.py`** - Heavy modifications for headless mode
2. **`models/wan/any2video.py`** - Vid2vid init, latent noise mask
3. **`models/qwen/`** - Entire directory is custom
4. **`shared/`** - Entire directory is custom reorganization
5. **All custom `defaults/*.json`** - Lightning baselines, cocktail configs

### Medium Priority - Review Upstream Changes
1. **`wan/any2video.py`** - Merge our debug logging with upstream improvements
2. **`preprocessing/`** - Check for upstream bug fixes
3. **`postprocessing/`** - Check for upstream improvements

### Low Priority - Can Accept Upstream
1. **`docs/`** - Accept upstream documentation updates
2. **`requirements.txt`** - Merge dependencies carefully
3. **Model configs in `configs/`** - Review for new models

### Recommended Merge Process
1. Fetch latest upstream to a separate branch
2. Diff key files listed above against our version
3. For `wgp.py`: Manual merge required - many custom modifications
4. For `any2video.py`: Apply our additions to new upstream version
5. For `models/`: Keep our structure, update handler code if upstream has improvements
6. For `defaults/`: Keep our custom configs, add any new upstream defaults

### Critical Files to Back Up Before Merge
```
Wan2GP/wgp.py
Wan2GP/models/wan/any2video.py
Wan2GP/models/qwen/*
Wan2GP/shared/*
Wan2GP/defaults/lightning_baseline*.json
Wan2GP/defaults/wan_2_2_*.json
Wan2GP/defaults/vace_*cocktail*.json
Wan2GP/defaults/qwen_*.json
```

---

## Quick Reference: Key Commits

| Commit | Description |
|--------|-------------|
| `7895e06` | Two-pass hires fix for Qwen |
| `b5ce3d3` | Fix UnboundLocalError for numpy in euler path |
| `c778b86` | Vid2vid initialization for VACE replace mode |
| `8c1947b` | Remove hardcoded loras from lightning configs |
| `a6af637` | System_prompt support for Qwen Image Edit |
| `fdd6c57` | DPM++_SDE sampler support |
| `26795d3` | Argparse conflict prevention for headless |
| `6091580` | Remove WGP settings caching entirely |

---

*Document generated: January 4, 2026*
*For upstream comparison: https://github.com/deepbeepmeep/Wan2GP*

