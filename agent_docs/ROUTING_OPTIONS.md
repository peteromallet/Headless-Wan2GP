# Task Routing - Your Options

You said: **"I basically want every task_type to either be Wan2GP or Comfy"**

Here are your options:

## Option 1: Simple Two-Engine Routing (Recommended)

Replace the complex TaskRegistry with simple routing:

### How It Works

```python
def route_task(task_type, context):
    # ComfyUI tasks
    if task_type in COMFY_TASK_TYPES:
        return handle_comfy_task(...)

    # Everything else → Wan2GP
    return handle_wan2gp_task(...)
```

### To Enable

```bash
# Backup existing registry
mv source/task_registry.py source/task_registry_legacy.py

# Use simple registry
mv source/task_registry_simple.py source/task_registry.py

# Done! Now all tasks route to either ComfyUI or Wan2GP
```

### What Changes

**Before (11 different handlers):**
- `vace` → Wan2GP queue
- `flux` → Wan2GP queue
- `travel_orchestrator` → Special handler
- `magic_edit` → Special handler
- `join_clips_orchestrator` → Special handler
- `inpaint_frames` → Special handler
- `comfy` → ComfyUI handler
- etc...

**After (2 engines):**
- `vace` → Wan2GP engine
- `flux` → Wan2GP engine
- `travel_orchestrator` → Wan2GP engine (no special handling)
- `magic_edit` → Wan2GP engine (no special handling)
- `comfy` → ComfyUI engine
- etc...

### ⚠️ Impact on Specialized Handlers

Specialized handlers like `travel_orchestrator`, `magic_edit`, etc. will **stop working** because they won't have custom pre/post-processing.

**Solutions:**
1. **Refactor into task params** - Move orchestration logic into task parameters
2. **Keep existing handlers** - Use Option 2 instead

---

## Option 2: Hybrid (Safe Migration)

Keep existing specialized handlers but route new tasks simply:

### How It Works

```python
def route_task(task_type, context):
    # Legacy specialized handlers (backward compatibility)
    if task_type in LEGACY_HANDLERS:
        return legacy_handlers[task_type](context)

    # ComfyUI tasks
    if task_type in COMFY_TASK_TYPES:
        return handle_comfy_task(...)

    # Everything else → Wan2GP
    return handle_wan2gp_task(...)
```

### To Enable

Add this to the existing `task_registry.py`:

```python
# At the top of dispatch method
class TaskRegistry:
    @staticmethod
    def dispatch(task_type: str, context: Dict[str, Any]):
        # NEW: Route via engine router for non-specialized tasks
        if task_type in COMFY_TASK_TYPES:
            return route_task(task_type, context)

        if task_type in DIRECT_QUEUE_TASK_TYPES and context["task_queue"]:
            return route_task(task_type, context)

        # EXISTING: Keep specialized handlers for backward compatibility
        handlers = {
            "travel_orchestrator": lambda: ...,
            "magic_edit": lambda: ...,
            # etc...
        }

        if task_type in handlers:
            return handlers[task_type]()

        # FALLBACK: Use router
        return route_task(task_type, context)
```

### What Changes

**Nothing breaks!** Existing tasks continue working.

New tasks automatically route to engines without needing handler definitions.

---

## Comparison

| Aspect | Option 1 (Simple) | Option 2 (Hybrid) |
|--------|-------------------|-------------------|
| **Complexity** | Very low | Medium |
| **Code to maintain** | ~50 lines | ~500 lines |
| **Specialized handlers** | ❌ Removed | ✅ Kept |
| **Backward compatibility** | ❌ Breaking | ✅ Safe |
| **Future additions** | Just add to COMFY_TASK_TYPES | Still auto-routes |

---

## Which Should You Choose?

### Choose Option 1 (Simple) if:
- ✅ You don't need specialized orchestrators (`travel_orchestrator`, etc.)
- ✅ All orchestration can be done via task params
- ✅ You want the cleanest, simplest code
- ✅ You're okay with a breaking change

### Choose Option 2 (Hybrid) if:
- ✅ You have existing tasks using specialized handlers
- ✅ You need backward compatibility
- ✅ You want to migrate gradually
- ✅ You can't refactor orchestrators yet

---

## My Recommendation

**Start with Option 2 (Hybrid)** for safety, then migrate to Option 1 when ready:

### Phase 1: Now (Hybrid)
```python
# Keep existing specialized handlers
# Route new simple tasks (vace, flux, comfy) via engine router
# No breaking changes
```

### Phase 2: Later (Simple)
```python
# Refactor orchestrators into task params
# Remove specialized handlers
# Clean, simple two-engine system
```

---

## Current Status

**Already implemented:**
- ✅ `task_engine_router.py` - Two-engine routing logic
- ✅ `task_registry_simple.py` - Simple registry (Option 1)
- ✅ `task_registry.py` - Complex registry with specialized handlers (current)

**To switch to Option 1:**
```bash
mv source/task_registry.py source/task_registry_legacy.py
mv source/task_registry_simple.py source/task_registry.py
```

**To use Option 2:**
No changes needed! Current `task_registry.py` already works with the router via the comfy handler.

---

## Adding New Task Types

### With Simple Routing (Option 1)

**Add to ComfyUI:**
```python
# source/task_engine_router.py
COMFY_TASK_TYPES = {
    "comfy",
    "comfy_image",  # Add this
}
```

**Everything else automatically goes to Wan2GP!**

### With Hybrid (Option 2)

Same as above - new tasks auto-route. Specialized handlers stay separate.

---

## Summary

You want: **Every task_type to either be Wan2GP or Comfy**

✅ **Already implemented!** Just choose your migration path:
- **Option 1:** Clean slate, two engines only
- **Option 2:** Gradual migration, keep legacy handlers

Both options give you the simplicity you want - tasks route to engines, not specialized handlers.
