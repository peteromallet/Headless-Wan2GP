# Phase 1: Gotchas and Fixes

## Date: 2025-12-17

## Summary

After thorough sense-checking of Phase 1 implementation, I found **1 CRITICAL GOTCHA** and several minor notes. Below is the complete analysis.

---

## 🚨 GOTCHA #1: `image_save_path` Not Being Set (CRITICAL)

### Problem

WGP has TWO separate configuration keys for output paths:
- `save_path` - For videos
- `image_save_path` - For images (t2i, flux, etc.)

**Our Phase 1 implementation only sets `save_path`, not `image_save_path`.**

### Impact

- **Video generation tasks** (vace, t2v, i2v, etc.) → Will use configured path ✅
- **Image generation tasks** (wan_2_2_t2i, flux) → Will use DEFAULT path ❌
  - Images will still go to `Wan2GP/outputs/` instead of configured directory
  - Partial fix only!

### Evidence

From `Wan2GP/wgp.py:2347-2348`:
```python
save_path = server_config.get("save_path", os.path.join(os.getcwd(), "outputs"))
image_save_path = server_config.get("image_save_path", os.path.join(os.getcwd(), "outputs"))
```

From `Wan2GP/wgp.py:4864`:
```python
os.makedirs(image_save_path, exist_ok=True)
```

From `Wan2GP/wgp.py:5436`:
```python
image_path = os.path.join(image_save_path, file_name)
```

### Current Code (headless_wgp.py:321)

```python
'server_config': {'save_path': absolute_outputs_path}
```

### Fix Required

```python
'server_config': {
    'save_path': absolute_outputs_path,
    'image_save_path': absolute_outputs_path  # ADD THIS LINE
}
```

### Severity

**HIGH** - This defeats the purpose of Phase 1 for image-only tasks.

### Fix Complexity

**TRIVIAL** - One line change in `headless_wgp.py`

---

## ✅ Non-Issues (Checked and Safe)

### 1. Multiple Instantiation Points

**Checked:** `headless_model_management.py` has a `main()` function that creates `HeadlessTaskQueue` without `main_output_dir`.

**Status:** ✅ SAFE
- Parameter is optional (defaults to `None`)
- Maintains backwards compatibility
- This entry point is rarely used (worker.py is primary)

### 2. File Upload Logic

**Checked:** Upload functions in `source/common_utils.py`

**Status:** ✅ SAFE
- Uses provided paths dynamically
- No hardcoded path expectations
- Works with any output location

### 3. Orchestrator Paths

**Checked:** All orchestrator files in `source/sm_functions/`

**Status:** ✅ SAFE
- No hardcoded `Wan2GP/outputs` references
- Use dynamic path parameters
- Compatible with Phase 1 changes

### 4. Directory Creation

**Checked:** WGP's directory creation in `wgp.py`

**Status:** ✅ SAFE
```python
os.makedirs(save_path, exist_ok=True)
os.makedirs(image_save_path, exist_ok=True)
```
- WGP creates directories automatically
- Uses `exist_ok=True` (no error if exists)
- Will work with custom paths

### 5. Test Files

**Checked:** References to `Wan2GP/outputs` in test files

**Status:** ✅ SAFE
- Only in test/documentation files
- Don't affect production code
- Will be updated in Phase 2

---

## 🔍 Additional Checks Performed

### Syntax Validation
✅ All Python files compile without errors

### Import Chain
✅ All imports work correctly with new signatures

### Backwards Compatibility
✅ Optional parameters maintain old behavior

### File System Impact
✅ No unexpected file movements during code-only changes

### Path Construction
✅ No hardcoded path assumptions found

### URL/Storage References
✅ No assumptions about specific output locations

---

## 🛠️ Recommended Fix

### Option A: Fix Now (Recommended)

Apply the fix immediately before any production use:

```bash
# Edit headless_wgp.py line 321
# Change from:
'server_config': {'save_path': absolute_outputs_path}

# To:
'server_config': {
    'save_path': absolute_outputs_path,
    'image_save_path': absolute_outputs_path
}
```

**Benefits:**
- Complete fix for Phase 1
- Trivial change (1 line)
- Zero risk
- Makes Phase 1 actually work for all task types

**Cost:**
- Need to amend commit or create new commit
- Another round of validation

### Option B: Document and Fix in Phase 2

Proceed with current implementation, document the issue:

**Benefits:**
- No need to revise Phase 1
- Still achieves partial consolidation (videos)

**Drawbacks:**
- Images still split across directories
- Phase 1 objectives not fully met
- User might report "images still in Wan2GP/outputs"

---

## 📋 Updated Testing Checklist

After applying fix:

- [ ] Re-run syntax check
- [ ] Test image generation task (t2i/flux)
- [ ] Verify images go to configured directory
- [ ] Test video generation task (vace/t2v)
- [ ] Verify videos go to configured directory
- [ ] Run file snapshot comparison
- [ ] Check for any other `server_config` keys we might have missed

---

## 🎯 Risk Assessment After Fix

**Original Phase 1:** Very Low Risk ✅ (but incomplete)
**With image_save_path fix:** Very Low Risk ✅ AND complete ✅

**Why still low risk:**
- Same backwards compatible design
- Same safety properties
- Just adds one more config key
- WGP already handles both paths identically

---

## 📊 Final Recommendation

**FIX NOW** - Apply the image_save_path fix before deployment.

**Reasoning:**
1. It's a trivial change (1 line)
2. Makes Phase 1 actually complete
3. Zero additional risk
4. Takes 2 minutes to implement and test
5. Avoids user confusion ("Why are images still in old location?")

**Alternative:**
If you're confident image-only tasks won't run immediately, you could deploy as-is and fix in Phase 2. But fixing now is cleaner.

---

## Summary

Found: **1 critical gotcha**, easily fixable
Verified: **5+ potential issues** - all safe
Recommendation: **Apply fix immediately** (1 line change)

Phase 1 is 95% complete, needs 1-line fix to be 100% complete.
