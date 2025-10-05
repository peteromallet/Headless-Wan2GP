# Path Monitoring and Verification

## Overview

Added comprehensive path verification throughout the codebase to catch directory changes early and provide clear diagnostics.

## Why This Matters

Wan2GP's `wgp.py` uses **relative paths** extensively:
- Model definitions: `defaults/*.json`, `finetunes/*.json`
- Model files: `models/`, `ckpts/`
- LoRAs: `loras_*/`
- Outputs: `outputs/`

If the current working directory changes away from `Wan2GP/`, these operations will fail mysteriously.

## Path Check Points

### 1. Before wgp Import (headless_wgp.py:172-174)
```python
if current_dir != self.wan_root:
    raise RuntimeError("Directory changed unexpectedly before wgp import")
```
**What it catches**: Ensures we're in Wan2GP/ before importing wgp module

**Log pattern**: `[INIT_DEBUG] Current directory: /path/to/Wan2GP`

---

### 2. After wgp Module Import (headless_wgp.py:189)
```python
_verify_wgp_directory(_init_logger, "after importing wgp module")
```
**What it catches**: Verifies wgp's module-level code didn't change directories

**Log pattern**: `[PATH_CHECK] after importing wgp module: Still in Wan2GP directory ✓`

---

### 3. After wgp Setup and Monkeypatching (headless_wgp.py:301)
```python
_verify_wgp_directory(model_logger, "after wgp setup and monkeypatching")
```
**What it catches**: Ensures patching operations didn't affect directory

**Log pattern**: `[PATH_CHECK] after wgp setup and monkeypatching: Still in Wan2GP directory ✓`

---

### 4. After apply_changes() (headless_wgp.py:381)
```python
_verify_wgp_directory(orchestrator_logger, "after apply_changes()")
```
**What it catches**: apply_changes() does file operations that could change directory

**Log pattern**: `[PATH_CHECK] after apply_changes(): Still in Wan2GP directory ✓`

---

### 5. After wgp.generate_video() (headless_wgp.py:1535)
```python
_verify_wgp_directory(generation_logger, "after wgp.generate_video()")
```
**What it catches**: Generation saves outputs and could change directory

**Log pattern**: `[PATH_CHECK] after wgp.generate_video(): Still in Wan2GP directory ✓`

---

### 6. After Task Processing (headless_model_management.py:541-551)
```python
current_dir = os.getcwd()
if "Wan2GP" not in current_dir:
    self.logger.warning("[PATH_CHECK] After generation: Current directory changed!")
```
**What it catches**: Ensures we're still in Wan2GP after full task execution

**Log pattern**: `[PATH_CHECK] After generation: Still in Wan2GP ✓`

---

## Helper Function: _verify_wgp_directory()

Located in `headless_wgp.py`, lines 16-47:

```python
def _verify_wgp_directory(logger, context: str = ""):
    """Verify we're still in Wan2GP directory."""
    current_dir = os.getcwd()

    # Check directory contains "Wan2GP"
    if "Wan2GP" not in current_dir:
        logger.warning(f"[PATH_CHECK] {context}: Current directory may be wrong!")

    # Verify critical directories accessible
    if not os.path.exists("defaults"):
        logger.error(f"[PATH_CHECK] {context}: CRITICAL - defaults/ no longer accessible!")

    return current_dir
```

### What It Checks
1. Current directory contains "Wan2GP" in path
2. `defaults/` directory is accessible (relative path)
3. Logs appropriate warnings/errors if checks fail

## Log Levels

- **DEBUG**: Normal operation - path is correct
- **WARNING**: Directory doesn't contain "Wan2GP" (potential issue)
- **ERROR**: Critical directory (defaults/) not accessible (serious issue)

## Reading the Logs

### ✅ Normal Operation
```
[PATH_CHECK] after importing wgp module: Still in Wan2GP directory ✓ (/workspace/lol/Headless-Wan2GP/Wan2GP)
[PATH_CHECK] after wgp setup and monkeypatching: Still in Wan2GP directory ✓
[PATH_CHECK] after apply_changes(): Still in Wan2GP directory ✓
[PATH_CHECK] after wgp.generate_video(): Still in Wan2GP directory ✓
[PATH_CHECK] After generation: Still in Wan2GP ✓
```

### ⚠️ Warning - Wrong Directory
```
[PATH_CHECK] after wgp.generate_video(): Current directory may be wrong!
  Current: /workspace/lol/Headless-Wan2GP
  Expected: Path containing 'Wan2GP'
  This could cause issues with wgp.py's relative paths!
```

### ❌ Error - Critical Directory Missing
```
[PATH_CHECK] after apply_changes(): CRITICAL - defaults/ no longer accessible!
  Current directory: /some/wrong/path
```

## Troubleshooting

### If You See Path Warnings

1. **Check the log context** - Which operation changed the directory?
2. **Look for chdir calls** - Did custom code call `os.chdir()`?
3. **Check wgp.py updates** - Did upstream Wan2GP add chdir calls?

### If Subsequent Tasks Fail

Path warnings indicate the worker is in the wrong directory. This will cause:
- Model loading failures
- "File not found" errors
- "No model definitions found" errors

**Solution**: Restart the worker process (it will reinitialize in correct directory)

## Performance Impact

- **Minimal** - Each check is just:
  - `os.getcwd()` - Fast syscall
  - String comparison
  - Optional `os.path.exists()` check

- Only logs at DEBUG level for successful checks
- Uses logger conditionals, so disabled loggers have zero overhead

## Files Modified

- `headless_wgp.py`:
  - Lines 16-47: Added `_verify_wgp_directory()` helper
  - Lines 189, 301, 381, 1535: Added path checks after wgp operations

- `headless_model_management.py`:
  - Lines 541-551: Added path check after task generation

## Future Maintenance

When adding new wgp operations:
1. Add `_verify_wgp_directory(logger, "after new_operation")` after the call
2. Use descriptive context string
3. Consider if the operation might change directory (file I/O, subprocess calls)
