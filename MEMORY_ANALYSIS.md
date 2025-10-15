# Memory Usage Analysis & Improvements

## Latest Issue: VLM Model Loading Failure (2025-10-15)

### Critical Path Resolution Problem

The VLM model was failing to load, causing:
1. **Model not loading from local cache** - tried to download from HuggingFace instead
2. **Wan models offloaded to CPU** - caused 43-second text encoding delays (should be 1-2s)
3. **Fallback to generic prompts** - "cinematic transition" instead of VLM-generated descriptions

**Root Cause**: Incorrect relative path `ckpts/Qwen2.5-VL-7B-Instruct`
- Worker runs from: `/workspace/Headless-Wan2GP`
- Model located at: `/workspace/Headless-Wan2GP/Wan2GP/ckpts/Qwen2.5-VL-7B-Instruct`
- Relative path tried: `/workspace/Headless-Wan2GP/ckpts/...` (doesn't exist)

**Fix Applied**: Changed to absolute path resolution in `vlm_utils.py` lines 77 and 194:

Also added comprehensive RAM monitoring throughout the travel orchestrator process.
```python
model_path = str(Path(__file__).parent.parent / "Wan2GP" / "ckpts" / "Qwen2.5-VL-7B-Instruct")
extender = QwenPromptExpander(
    model_name=model_path,  # Now resolves correctly
    device=device,
    is_vl=True
)
```

### RAM Monitoring Added (2025-10-15)

To track RAM usage and identify potential memory leaks, comprehensive RAM monitoring has been added throughout the travel orchestrator process:

**New Helper Function** in `travel_between_images.py` (line 94):
```python
def log_ram_usage(label: str, task_id: str = "unknown", logger=None) -> dict:
    """
    Log current RAM usage with descriptive label.
    Shows both process RSS (Resident Set Size) and system-wide memory stats.
    """
    process = psutil.Process(os.getpid())
    rss_mb = process.memory_info().rss / 1024**2
    sys_mem = psutil.virtual_memory()

    logger.info(
        f"[RAM] {label}: Process={rss_mb:.0f}MB ({rss_mb/1024:.2f}GB) | "
        f"System={sys_mem.percent:.1f}% used, {sys_mem.available/1024**3:.1f}GB available"
    )
```

**RAM Monitoring Points Added**:

1. **Orchestrator Task**:
   - Start of orchestrator (line 139)
   - Before VLM loading (line 997)
   - After VLM cleanup (line 1105)
   - End of orchestrator - success (line 1409)
   - End of orchestrator - error (line 1419)

2. **Segment Task**:
   - Start of segment (line 1425)
   - End of segment - success (line 2106)
   - End of segment - error (line 2128)

3. **Stitch Task**:
   - Start of stitch (line 2496)
   - End of stitch - success (line 3313)
   - End of stitch - error (line 3333)

**Example Output with Task IDs**:
```
[17:16:01] INFO TRAVEL [Task ad02fd98-2c1a-4ec8-802d-89529b275257] [RAM] Orchestrator start: Process=5432MB (5.31GB) | System=45.2% used, 58.3GB/128.0GB available
[17:16:02] INFO TRAVEL [Task ad02fd98-2c1a-4ec8-802d-89529b275257] [RAM] Before VLM loading: Process=5450MB (5.32GB) | System=45.3% used, 58.2GB/128.0GB available
[17:17:17] INFO TRAVEL [Task ad02fd98-2c1a-4ec8-802d-89529b275257] [RAM] After VLM cleanup: Process=21234MB (20.74GB) | System=62.8% used, 41.9GB/128.0GB available
[17:17:20] INFO TRAVEL [Task ad02fd98-2c1a-4ec8-802d-89529b275257] [RAM] Orchestrator end (success): Process=6240MB (6.09GB) | System=46.1% used, 57.4GB/128.0GB available
[17:17:22] INFO HEADLESS [Task 7b97a7dd-b5e2-449d-ab6d-dec8e30bec9e] [RAM] Segment via queue - start: Process=6245MB (6.10GB) | System=46.2% used, 57.3GB/128.0GB available
[17:17:38] INFO HEADLESS [Task 7b97a7dd-b5e2-449d-ab6d-dec8e30bec9e] [RAM] Before queue submission: Process=6892MB (6.73GB) | System=47.8% used, 55.7GB/128.0GB available
[17:21:20] INFO HEADLESS [Task 7b97a7dd-b5e2-449d-ab6d-dec8e30bec9e] [RAM] Segment via queue - end (success with chain): Process=7124MB (6.96GB) | System=48.3% used, 55.2GB/128.0GB available
```

This allows tracking:
- Process memory growth over time
- VLM model memory footprint (~14-16GB expected)
- Whether cleanup properly frees memory
- System-wide memory pressure
- Memory leaks across multiple task runs

---

## Previous Issue: RAM Cleanup & Monitoring

### Problem Identified

The Qwen2.5-VL-7B vision-language model used for prompt enhancement was consuming 14-16GB of system RAM but wasn't being properly tracked or released after use.

### Root Causes

1. **No RAM monitoring** - Only GPU VRAM was being logged, not system RAM where the VLM model resides
2. **Incomplete tensor cleanup** - Intermediate tensors (`inputs`, `generated_ids`, `image_inputs`, etc.) weren't being deleted
3. **PIL image retention** - Combined images created for each transition weren't being explicitly deleted
4. **Single-pass garbage collection** - Python's GC wasn't running enough times to clean up circular references

## Changes Made

### 1. Added System RAM Monitoring (`source/vlm_utils.py`)

**Before:**
```python
# Only GPU VRAM was logged
if torch.cuda.is_available():
    gpu_mem_before = torch.cuda.memory_allocated() / 1024**3
    dprint(f"[VLM_BATCH] GPU memory BEFORE: {gpu_mem_before:.2f} GB")
```

**After:**
```python
# Now logs both system RAM and GPU VRAM
import psutil
import os

process = psutil.Process(os.getpid())
ram_before_mb = process.memory_info().rss / 1024**2
print(f"[VLM_BATCH] System RAM BEFORE loading: {ram_before_mb:.2f} MB ({ram_before_mb/1024:.2f} GB)")
```

Added RAM logging at:
- Before VLM model loading
- After VLM model loading (shows model RAM footprint)
- Before cleanup
- After cleanup (shows if memory was actually freed)

### 2. Improved PIL Image Cleanup (`source/vlm_utils.py:255`)

```python
# Clean up PIL images to free memory after each inference
del combined_img, start_img, end_img
```

### 3. Enhanced VLM Object Cleanup (`source/vlm_utils.py:277-306`)

**Before:**
```python
del extender.model
del extender.processor
del extender
gc.collect()  # Single pass
```

**After:**
```python
# Move model to CPU first
if hasattr(extender, 'model') and extender.model is not None:
    extender.model = extender.model.to('cpu')
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

# Delete all components individually
if hasattr(extender, 'model'):
    del extender.model
if hasattr(extender, 'processor'):
    del extender.processor
if hasattr(extender, 'tokenizer'):
    del extender.tokenizer
if hasattr(extender, 'process_vision_info'):
    del extender.process_vision_info

del extender

# Triple-pass garbage collection for better cleanup
collected_1 = gc.collect()
collected_2 = gc.collect()
collected_3 = gc.collect()
```

### 4. Added Intermediate Tensor Cleanup (`Wan2GP/wan/utils/prompt_extend.py`)

**In `extend_with_img()` (line 450-453):**
```python
# Clean up intermediate tensors before moving model back to CPU
del inputs, generated_ids, generated_ids_trimmed, image_inputs, video_inputs
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

**In `extend()` (line 389-392):**
```python
# Clean up intermediate tensors before moving model back to CPU
del model_inputs, generated_ids
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

### 5. Added Warning Detection (`source/vlm_utils.py:322-328`)

```python
# Check if RAM was actually freed
if ram_freed_mb < 1000:  # Less than 1GB freed
    print(f"[VLM_CLEANUP] ⚠️  WARNING: Expected to free ~14-16GB but only freed {ram_freed_mb:.2f} MB")
    print(f"[VLM_CLEANUP] ⚠️  Model may still be held in memory by Python or OS caching")
    print(f"[VLM_CLEANUP] ℹ️  RAM may be freed later by Python's garbage collector or OS")
else:
    print(f"[VLM_CLEANUP] ✅ Successfully freed {ram_freed_mb/1024:.2f} GB of system RAM")
```

## Expected Output

When running a task with `enhance_prompt=True`, you'll now see:

```
[VLM_BATCH] System RAM BEFORE loading: 5432.12 MB (5.31 GB)
[VLM_BATCH] System RAM AFTER loading: 21543.45 MB (21.04 GB)
[VLM_BATCH] System RAM increase from model load: 16111.33 MB (15.73 GB)
[VLM_BATCH] Processing pair 1/1: vlm_start_seg0_231017_6b4478.png → vlm_end_seg0_231018_ac1c14.png
[VLM_BATCH] Generated: The skeletal figure emerges from the fiery ground...
[VLM_BATCH] Completed 1/1 prompts
[VLM_CLEANUP] System RAM BEFORE cleanup: 21678.90 MB (21.17 GB)
[VLM_CLEANUP] GPU memory BEFORE cleanup: 0.01 GB
[VLM_CLEANUP] Cleaning up VLM model and processor...
[VLM_CLEANUP] ✅ Successfully deleted VLM objects
[VLM_CLEANUP] Garbage collected 256 objects (passes: 137, 89, 30)
[VLM_CLEANUP] GPU memory AFTER cleanup: 0.01 GB
[VLM_CLEANUP] GPU memory freed: 0.00 GB
[VLM_CLEANUP] System RAM AFTER cleanup: 6234.56 MB (6.09 GB)
[VLM_CLEANUP] System RAM freed: 15444.34 MB (15.08 GB)
[VLM_CLEANUP] ✅ Successfully freed 15.08 GB of system RAM
[VLM_CLEANUP] ✅ VLM cleanup complete
```

## Remaining Considerations

If RAM is not being freed immediately:

1. **Python's memory allocator** may hold onto freed memory for future allocations
2. **OS page caching** may keep the memory mapped but available for reuse
3. **Hugging Face transformers** uses memory-mapped files which can show as RSS until unmapped by the OS

The key metric is whether subsequent model loads reuse the freed space rather than allocating new memory.

## Files Modified

1. `/workspace/Headless-Wan2GP/source/vlm_utils.py` - Added RAM monitoring and improved cleanup
2. `/workspace/Headless-Wan2GP/Wan2GP/wan/utils/prompt_extend.py` - Added intermediate tensor cleanup

## Testing

Run a travel_between_images task with `enhance_prompt=True` and monitor:
1. Initial RAM usage
2. RAM increase when VLM loads (~14-16GB expected)
3. RAM decrease after cleanup (should match the increase)
4. Whether subsequent generations reuse the memory space
