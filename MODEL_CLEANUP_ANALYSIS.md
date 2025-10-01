# Model Cleanup Analysis - Structure Video Guidance

## Problem Identified

Our `extract_optical_flow_from_frames` creates a RAFT model but doesn't explicitly clean it up:

```python
def extract_optical_flow_from_frames(frames, dprint=print):
    # Initialize annotator
    flow_annotator = FlowAnnotator(flow_cfg)  # Loads RAFT model to GPU
    
    # Extract flow
    flow_fields, flow_vis = flow_annotator.forward(frames)
    
    return flow_fields, flow_vis
    # FlowAnnotator goes out of scope here, but GPU memory not freed!
```

## How WGP.py Handles It

**Pattern in wgp.py (lines 5285-5299):**
```python
except Exception as e:
    # Cleanup on error
    clear_gen_cache()
    offloadobj.unload_all()
    trans.cache = None 
    offload.unload_loras_from_model(trans)
    gc.collect()
    torch.cuda.empty_cache()
```

**Observation:**
- WGP creates preprocessors within `generate_video` scope
- They get cleaned up at end of generation (or on error)
- Explicit `gc.collect()` + `torch.cuda.empty_cache()` calls

## Our Situation

**Timeline:**
1. Guide video creation (our code runs)
   - FlowAnnotator created → RAFT loaded to GPU
   - Flow extracted
   - **Function returns** → FlowAnnotator out of scope
   - **GPU memory still allocated** (not freed by Python GC)
   
2. VACE model generation (TravelSegmentProcessor)
   - Main VACE model loads to GPU
   - **Potential GPU OOM if RAFT still in memory**

## Solution Required

Add explicit cleanup after flow extraction:

```python
def extract_optical_flow_from_frames(frames, dprint=print):
    import gc
    import torch
    
    # Initialize annotator
    flow_annotator = FlowAnnotator(flow_cfg)
    
    try:
        # Extract flow
        flow_fields, flow_vis = flow_annotator.forward(frames)
        
        return flow_fields, flow_vis
    finally:
        # CRITICAL: Clean up RAFT model
        del flow_annotator
        gc.collect()
        torch.cuda.empty_cache()
        dprint(f"[OPTICAL_FLOW] Cleaned up RAFT model from GPU")
```

## Impact Assessment

**Without cleanup:**
- ❌ RAFT model (~200-300MB) stays in GPU memory
- ❌ Reduces available VRAM for main VACE generation
- ❌ May cause OOM on smaller GPUs (8-12GB)
- ❌ Memory leak compounds if multiple segments processed

**With cleanup:**
- ✅ RAFT freed immediately after use
- ✅ Maximum VRAM available for VACE model
- ✅ No memory leaks across segments
- ✅ Works reliably on all GPU sizes

## Urgency: **HIGH** 🚨

This is a production-critical bug that will cause GPU memory issues.

