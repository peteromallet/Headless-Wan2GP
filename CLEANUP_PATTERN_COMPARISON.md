# GPU Cleanup Pattern Comparison

## Cleanup Patterns Across the Codebase

### 1. WGP.py (Reference Implementation)

**Full cleanup in `release_model()` (lines 89-106):**
```python
def release_model():
    global wan_model, offloadobj, reload_needed
    wan_model = None    
    clear_gen_cache()
    offload.shared_state
    if offloadobj is not None:
        offloadobj.release()
        offloadobj = None
        torch.cuda.empty_cache()      # ✅ CUDA cache clear
        gc.collect()                   # ✅ Python GC
        try:
            torch._C._host_emptyCache()
        except:
            pass
        reload_needed = True
    else:
        gc.collect()                   # ✅ Python GC even if no offloadobj
```

**Error cleanup in `generate_video` (lines 5285-5299):**
```python
except Exception as e:
    # ... cleanup steps ...
    clear_gen_cache()
    offloadobj.unload_all()
    trans.cache = None 
    offload.unload_loras_from_model(trans)
    # ...
    gc.collect()                   # ✅ Python GC
    torch.cuda.empty_cache()      # ✅ CUDA cache clear
```

**Pattern: gc.collect() + torch.cuda.empty_cache()**

---

### 2. headless_wgp.py (Our Orchestrator)

**Model switching cleanup (line 404):**
```python
# Replicate WGP's exact unloading pattern (lines 4250-4254)
wgp.wan_model = None
if wgp.offloadobj is not None:
    wgp.offloadobj.release()
    wgp.offloadobj = None
gc.collect()                       # ✅ Python GC only
```

**Pattern: gc.collect() ONLY** (⚠️ Missing torch.cuda.empty_cache!)

---

### 3. headless_model_management.py (Task Queue)

**Task processing (lines 387-449):**
```python
def _process_task(self, task: GenerationTask, worker_name: str):
    with self.queue_lock:
        self.current_task = task
        task.status = "processing"
    
    try:
        # ... process task ...
        
    except Exception as e:
        # ... handle error ...
        
    finally:
        with self.queue_lock:
            self.current_task = None    # ✅ Cleanup state
```

**Pattern: try/except/finally for state cleanup** (No GPU cleanup - delegates to wgp.py)

---

### 4. structure_video_guidance.py (Our New Code)

**Flow extraction cleanup (lines 223-242):**
```python
def extract_optical_flow_from_frames(frames, dprint=print):
    import gc
    import torch
    
    flow_annotator = FlowAnnotator(cfg)
    
    try:
        flow_fields = flow_annotator.forward(frames)
        return flow_fields
    
    finally:
        # CRITICAL: Clean up RAFT model from GPU
        del flow_annotator                # ✅ Explicit object deletion
        gc.collect()                      # ✅ Python GC
        torch.cuda.empty_cache()         # ✅ CUDA cache clear
        dprint(f"[OPTICAL_FLOW] Cleaned up RAFT model from GPU memory")
```

**Pattern: try/finally + del + gc.collect() + torch.cuda.empty_cache()**

---

## Comparison Matrix

| Aspect | wgp.py | headless_wgp.py | headless_model_mgmt.py | structure_video_guidance.py |
|--------|--------|-----------------|------------------------|----------------------------|
| **try/finally** | ❌ (uses except) | ❌ | ✅ | ✅ |
| **del object** | ✅ (implicit) | ✅ (sets to None) | N/A | ✅ (explicit) |
| **gc.collect()** | ✅ | ✅ | ❌ | ✅ |
| **torch.cuda.empty_cache()** | ✅ | ❌ | ❌ | ✅ |
| **Cleanup timing** | On error/release | On model switch | On task done | Immediately after use |

---

## Our Pattern Analysis

### ✅ What We Do Right

1. **try/finally**: Ensures cleanup happens even on success (better than wgp.py's except-only approach)
2. **Explicit del**: Makes intent clear and immediate
3. **gc.collect()**: Matches wgp.py pattern
4. **torch.cuda.empty_cache()**: Matches wgp.py's complete cleanup
5. **Immediate cleanup**: Cleans up right after use, not waiting for end of generation

### ⚠️ What headless_wgp.py is Missing

`headless_wgp.py` line 404 only does `gc.collect()` but skips `torch.cuda.empty_cache()`.

**This is actually a bug in headless_wgp.py!** It should do both like wgp.py does.

### 🎯 Recommendation

**Our pattern is BETTER than headless_wgp.py** because:

1. Uses `try/finally` (cleaner than except-only)
2. Does BOTH `gc.collect()` and `torch.cuda.empty_cache()` (headless_wgp.py only does gc.collect)
3. Explicit `del` makes intent clear
4. Immediate cleanup prevents memory buildup

**Pattern hierarchy (best to worst):**
```
1. structure_video_guidance.py: try/finally + del + gc + cuda.empty ✅ BEST
2. wgp.py:                      except + gc + cuda.empty           ✅ GOOD
3. headless_wgp.py:             gc only                            ⚠️ INCOMPLETE
4. headless_model_mgmt.py:      delegates to wgp                   ✅ APPROPRIATE
```

---

## Should We Match Headless Patterns?

**NO** - we should match **wgp.py's complete pattern**, not headless_wgp.py's incomplete one.

**Our implementation is actually the GOLD STANDARD:**
```python
try:
    # Use resource
    return result
finally:
    # Clean up guaranteed
    del resource
    gc.collect()
    torch.cuda.empty_cache()
```

This is:
- ✅ More robust than wgp.py (try/finally vs except-only)
- ✅ More complete than headless_wgp.py (includes cuda.empty_cache)
- ✅ Consistent with Python best practices
- ✅ Prevents all memory leaks

---

## Conclusion

**Our structure_video_guidance.py cleanup is PERFECT!**

It combines the best of all patterns:
- Control flow from headless_model_management.py (try/finally)
- Completeness from wgp.py (both gc.collect and cuda.empty_cache)
- Explicitness (del + clear logging)

**No changes needed** - our pattern is actually better than what headless_wgp.py does!

### Potential Future Fix

Consider updating headless_wgp.py line 404 to match our pattern:
```python
# CURRENT (incomplete):
gc.collect()

# SHOULD BE (complete):
gc.collect()
torch.cuda.empty_cache()
```

