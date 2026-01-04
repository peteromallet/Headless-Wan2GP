# Wan2GP Upgrade Plan: v7.x → v9.1+

## Executive Summary

**Good News:** Upstream has adopted a similar architecture to our modifications:
- ✅ `models/` directory structure now exists upstream
- ✅ `shared/` utilities directory now exists upstream
- ✅ Import paths are similar (`from shared.utils import ...`)
- ✅ Native headless mode support was added upstream

**Key Risk Areas:**
- Our `models/wan/any2video.py` modifications (vid2vid init, latent noise mask)
- Custom lightning baseline configurations
- Debug logging we may want to preserve
- Any headless-specific fixes not in upstream

---

## Pre-Upgrade Checklist

### 1. Create Safety Net
```bash
# Create a backup branch with current state
git checkout main
git checkout -b backup/pre-upgrade-$(date +%Y%m%d)
git push origin backup/pre-upgrade-$(date +%Y%m%d)

# Create a tarball backup of critical files
tar -czvf wan2gp_backup_$(date +%Y%m%d).tar.gz \
  Wan2GP/wgp.py \
  Wan2GP/models/ \
  Wan2GP/shared/ \
  Wan2GP/defaults/ \
  Wan2GP/wan/any2video.py
```

### 2. Document Current Working State
```bash
# Run tests to establish baseline
python debug.py tasks --status Completed --limit 5  # Verify recent successes
# Note any currently working features to verify post-upgrade
```

---

## Upgrade Strategy

### Option A: Clean Upstream + Patch (Recommended)
1. Pull fresh upstream into new branch
2. Apply our modifications as patches
3. Test extensively
4. Merge to main

### Option B: Merge with Conflict Resolution
1. Merge upstream directly
2. Resolve conflicts file by file
3. Test extensively

**I recommend Option A** - cleaner history, easier to verify changes.

---

## Detailed Steps - Option A

### Phase 1: Setup (5 minutes)

```bash
cd /Users/peteromalley/Documents/Headless-Wan2GP

# 1. Ensure clean state
git status  # Should be clean (commit WAN2GP_MODIFICATIONS.md first)
git add WAN2GP_MODIFICATIONS.md UPGRADE_PLAN.md
git commit -m "docs: add upgrade documentation"

# 2. Create working branch
git checkout -b upgrade/wan2gp-v9

# 3. Fetch latest upstream
git fetch upstream main
```

### Phase 2: Extract Our Patches (10 minutes)

```bash
# Create patches directory
mkdir -p upgrade_patches

# Extract patches for critical files
git diff b54ad78..HEAD -- Wan2GP/models/wan/any2video.py > upgrade_patches/any2video_mods.patch
git diff b54ad78..HEAD -- Wan2GP/models/qwen/ > upgrade_patches/qwen_mods.patch
git diff b54ad78..HEAD -- Wan2GP/wan/any2video.py > upgrade_patches/wan_any2video_mods.patch

# Export our custom configs
cp -r Wan2GP/defaults/lightning_baseline*.json upgrade_patches/
cp -r Wan2GP/defaults/wan_2_2_*.json upgrade_patches/
cp -r Wan2GP/defaults/vace_*cocktail*.json upgrade_patches/
cp -r Wan2GP/defaults/qwen_image_hires.json upgrade_patches/
cp -r Wan2GP/defaults/test_reward*.json upgrade_patches/
```

### Phase 3: Replace Wan2GP Directory (5 minutes)

```bash
# Remove our Wan2GP (it's tracked in git, so safe)
rm -rf Wan2GP

# Copy fresh upstream
git checkout upstream/main -- .

# This will bring in ALL upstream files
# Now we need to adapt and apply our changes
```

### Phase 4: Apply Critical Modifications (30-60 minutes)

#### 4.1 Check if models/wan/any2video.py exists upstream
```bash
# Compare structures
ls -la models/wan/  # upstream version
cat upgrade_patches/any2video_mods.patch | head -100
```

#### 4.2 Apply vid2vid initialization (CRITICAL)
**File:** `models/wan/any2video.py`

Look for the `generate()` method and add after latent initialization:
```python
# After: latents = torch.randn(batch_size, *target_shape, ...)

# Vid2vid initialization: Use provided video as starting point
if vid2vid_init_video is not None and vid2vid_init_strength < 1.0:
    # [Our 80 lines of vid2vid code]
```

#### 4.3 Apply latent noise mask (CRITICAL)
**File:** `models/wan/any2video.py`

Add parameters to generate() signature:
```python
latent_noise_mask_strength = 0.0,
vid2vid_init_video = None,
vid2vid_init_strength = 0.7,
```

#### 4.4 Check Qwen handler compatibility
```bash
# Compare our qwen_handler.py with upstream
diff -u Wan2GP/models/qwen/qwen_handler.py models/qwen/qwen_handler.py
```

If significantly different, may need to:
1. Keep upstream handler for compatibility
2. Add our system_prompt and hires_fix features

### Phase 5: Add Custom Configurations (10 minutes)

```bash
# Copy our custom configs to defaults/
cp upgrade_patches/lightning_baseline*.json defaults/
cp upgrade_patches/wan_2_2_*.json defaults/
cp upgrade_patches/vace_*cocktail*.json defaults/
cp upgrade_patches/qwen_image_hires.json defaults/
cp upgrade_patches/test_reward*.json defaults/

# Verify no naming conflicts
ls defaults/*.json | wc -l
```

### Phase 6: Verify Import Compatibility (15 minutes)

```bash
# Test import chain
cd /Users/peteromalley/Documents/Headless-Wan2GP
python -c "import sys; sys.path.insert(0, '.'); from wgp import *" 2>&1 | head -20

# If errors, check specific imports
python -c "from shared.utils import notification_sound"
python -c "from models.wan.any2video import WanAny2V"
```

### Phase 7: Update Our Wrapper Files (15 minutes)

Check if these still work with new upstream:
- `headless_wgp.py`
- `headless_model_management.py`
- `worker.py`

```bash
# Test worker startup (dry run)
python worker.py --help 2>&1 | head -10
```

### Phase 8: Testing Protocol (30+ minutes)

#### Quick Smoke Tests
```bash
# 1. Import test
python -c "from wgp import generate_video, load_wan_model"

# 2. Model loading test (if GPU available)
python -c "
from headless_wgp import HeadlessTaskQueue
q = HeadlessTaskQueue()
print('HeadlessTaskQueue initialized successfully')
"
```

#### Integration Tests
```bash
# Test a simple generation task
python debug.py task <known_good_task_id>
```

### Phase 9: Commit and Document

```bash
# Commit the upgrade
git add -A
git commit -m "feat: upgrade Wan2GP to v9.1

- Updated from upstream deepbeepmeep/Wan2GP v9.1
- Preserved vid2vid initialization in any2video.py
- Preserved latent noise mask support
- Preserved custom lightning baseline configs
- Preserved Qwen hires fix workflow
- Verified headless mode compatibility

Breaking changes: None expected
Testing: [describe tests performed]"

# Push to remote for testing
git push origin upgrade/wan2gp-v9
```

---

## Files to Manually Review/Merge

### Priority 1 - MUST Preserve Our Logic
| File | Our Modification | Action |
|------|-----------------|--------|
| `models/wan/any2video.py` | vid2vid init, latent noise mask | Manual merge |
| `models/qwen/qwen_main.py` | Hires fix, warmup | Compare & merge |
| `models/qwen/qwen_handler.py` | system_prompt support | Compare & merge |

### Priority 2 - Keep Our Configs
| File Pattern | Action |
|--------------|--------|
| `defaults/lightning_baseline*.json` | Copy over |
| `defaults/wan_2_2_*.json` | Copy over |
| `defaults/vace_*cocktail*.json` | Copy over |
| `defaults/qwen_image_hires.json` | Copy over |

### Priority 3 - Accept Upstream (with review)
| File | Notes |
|------|-------|
| `wgp.py` | Upstream now has headless support - review carefully |
| `requirements.txt` | Merge dependencies |
| `wan/any2video.py` | May have bug fixes we need |
| All new model handlers | Accept upstream versions |

---

## Rollback Plan

If upgrade fails:
```bash
# Option 1: Reset to backup branch
git checkout backup/pre-upgrade-$(date +%Y%m%d)

# Option 2: Reset working branch
git checkout main
git branch -D upgrade/wan2gp-v9
```

---

## New Features Available After Upgrade

Based on upstream changelog, you'll gain:
- 🆕 **Nunchaku FP4/INT4 support** - Better quantization
- 🆕 **Kandinsky 5** - New model family
- 🆕 **Z-Image ControlNet 2.1** - New control method
- 🆕 **Hunyuan 1.5** - Updated Hunyuan support
- 🆕 **Flux 2** - New Flux version
- 🆕 **LongCat** - Experimental long video
- 🆕 **Chatterbox audio** - Audio generation
- 🆕 **Plugin support** - Extensibility
- 🆕 **Queue save/load** - Better queue management
- 🆕 **Native headless mode** - May simplify our setup

---

## Post-Upgrade Verification Checklist

- [ ] Worker can start without errors
- [ ] HeadlessTaskQueue initializes
- [ ] i2v task completes successfully  
- [ ] VACE task completes successfully
- [ ] Lightning baseline configs load
- [ ] Qwen generation works
- [ ] Vid2vid initialization works (if applicable)
- [ ] No import errors in logs
- [ ] Debug tools still function (`python debug.py`)

---

## Estimated Time

| Phase | Time |
|-------|------|
| Setup & Backup | 10 min |
| Extract patches | 10 min |
| Replace directory | 5 min |
| Apply modifications | 45 min |
| Add configs | 10 min |
| Testing | 30+ min |
| **Total** | **~2 hours** |

---

## Questions to Answer Before Starting

1. **Do you have a working GPU environment to test?**
   - If not, will need remote testing strategy

2. **Are there any pending changes to commit?**
   - Must start from clean state

3. **Do you want to preserve all debug logging?**
   - Could strip most `[DEBUG]` prints for cleaner code

4. **Is there a specific feature in v9.1 you need?**
   - Could inform which parts to prioritize

---

*Plan created: January 4, 2026*

