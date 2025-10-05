# Directory Management Fix - Summary

## Problem Identified

The "Orchestrator initialization failed" error was caused by `wgp.py` being imported while the current working directory was `/workspace/lol/Headless-Wan2GP` instead of `/workspace/lol/Headless-Wan2GP/Wan2GP`.

### Root Cause

1. **Wan2GP's design**: `wgp.py` uses relative paths at module import time:
   ```python
   # Line 2265 of wgp.py - executes when module loads
   models_def_paths = glob.glob(os.path.join("defaults", "*.json")) + glob.glob(os.path.join("finetunes", "*.json"))
   ```

2. **Old code issues**:
   - Redundant `os.chdir()` calls in multiple places
   - Silent exception swallowing with `try/except: pass`
   - No verification that directory change succeeded
   - Directory recalculation using `__file__` could give different paths

3. **Result**: When wgp.py was imported from wrong directory:
   - Found 0 model definition files
   - `models_def` remained empty
   - Subsequent calls to `model_def.get(...)` failed with `'NoneType' object has no attribute 'get'`

## Solution Implemented

### Key Principle
**Wan2GP owns the working directory. Once we enter Wan2GP/, we STAY there.**

### Changes Made

#### 1. `headless_wgp.py` - Simplified and Added Validation
- **Removed** redundant `os.chdir()` (caller does this)
- **Added** strict validation that cwd == wan_root before doing anything
- **Added** early check that `defaults/` directory exists
- **Contract**: Caller MUST `chdir(wan_root)` before creating WanOrchestrator

```python
# NEW: Fails fast with clear error if in wrong directory
if current_dir != self.wan_root:
    raise RuntimeError("CRITICAL: WanOrchestrator must be initialized from Wan2GP directory!")
```

#### 2. `headless_model_management.py` - Enhanced Logging and Verification
- **Added** detailed logging of directory changes
- **Added** verification that `os.chdir()` succeeded
- **Added** check that `defaults/` exists before proceeding
- **Added** clear error messages if setup fails

```python
# NEW: Verify directory change worked
os.chdir(self.wan_dir)
actual_cwd = os.getcwd()
if actual_cwd != self.wan_dir:
    raise RuntimeError(f"Directory change failed!")
```

#### 3. Cleared Python Bytecode Cache
- Removed `__pycache__/*.pyc` and `Wan2GP/__pycache__/*.pyc`
- Forces fresh import of modules with new code

## How It Works Now

### Correct Initialization Sequence

1. **worker.py:1612** - Calculates absolute path to Wan2GP
2. **worker.py:1615** - Creates `HeadlessTaskQueue(wan_dir=...)`
3. **headless_model_management.py:293** - `os.chdir(self.wan_dir)` → Changes to Wan2GP
4. **headless_model_management.py:295-309** - Verifies change succeeded, defaults/ exists
5. **headless_model_management.py:314** - `WanOrchestrator(self.wan_dir)` → Creates orchestrator
6. **headless_wgp.py:41** - Verifies cwd == wan_root (fails fast if not)
7. **headless_wgp.py:52** - Verifies defaults/ exists
8. **headless_wgp.py:154** - `from wgp import ...` → wgp.py imports with CORRECT cwd
9. **wgp.py:2265** - glob.glob("defaults/*.json") → Finds 71 model files ✅

### What Happens on Error

**Before fix**: Silent failure, mysterious `'NoneType' object has no attribute 'get'` later

**After fix**: Immediate, clear error message:
```
CRITICAL: WanOrchestrator must be initialized from Wan2GP directory!
  Current directory: /workspace/lol/Headless-Wan2GP
  Expected directory: /workspace/lol/Headless-Wan2GP/Wan2GP
  Caller must chdir() before creating WanOrchestrator instance.
```

## Testing

Created `test_directory_fix.py` which verifies:
1. ✅ Wrong directory is properly rejected
2. ✅ Correct directory passes validation
3. ✅ Clear error messages guide debugging

## Next Steps for User

1. **Restart the worker** to pick up the new code:
   ```bash
   # Kill existing worker
   pkill -f "python worker.py"

   # Start fresh
   source venv/bin/activate
   python worker.py --db-type supabase ...
   ```

2. **Look for new debug logs** in output:
   - `[LAZY_INIT] Changing to Wan2GP directory`
   - `[LAZY_INIT] Changed directory to: ...`
   - `[INIT_DEBUG] WanOrchestrator.__init__ called with wan_root`
   - `[INIT_DEBUG] Available models after WGP import: [...]`

3. **If it still fails**, the logs will now show EXACTLY where and why

## Files Modified

### Core Fixes
- `headless_wgp.py` - Lines 19-64: Simplified __init__, added validation
- `headless_model_management.py` - Lines 285-314: Enhanced directory change with verification
- Removed: `__pycache__/*.pyc`, `Wan2GP/__pycache__/*.pyc`

### Path Monitoring (NEW)
- `headless_wgp.py`:
  - Lines 16-47: Added `_verify_wgp_directory()` helper function
  - Lines 189, 301, 381, 1535: Added path verification after wgp operations
- `headless_model_management.py`:
  - Lines 541-551: Added path check after task generation

See `PATH_MONITORING_GUIDE.md` for detailed documentation.

## Key Insight

The fix respects Wan2GP's design:
- Wan2GP expects to run from its own directory
- Uses relative paths throughout
- We work WITH this design, not against it
- Single source of truth: One chdir, stay there, verify it worked
