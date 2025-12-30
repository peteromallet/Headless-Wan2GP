# Simplified Task Routing - Just Two Engines

## Current Problem

The TaskRegistry has grown complex with many specialized handlers:
- `travel_orchestrator`, `magic_edit`, `join_clips`, `inpaint_frames`, etc.

Each handler does slightly different things, making the system hard to understand and maintain.

## Simplified Solution

**Two engines only:**
1. **Wan2GP** - Handles all AI generation tasks (vace, flux, t2v, i2v, etc.)
2. **ComfyUI** - Handles workflow-based tasks (comfy)

## Implementation

### Replace TaskRegistry.dispatch() with this:

```python
# source/task_registry.py

from source.task_engine_router import route_task, COMFY_TASK_TYPES

class TaskRegistry:
    """Registry for task handlers."""

    @staticmethod
    def dispatch(task_type: str, context: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Route task to appropriate engine: ComfyUI or Wan2GP.

        Simple routing:
        - task_type in COMFY_TASK_TYPES → ComfyUI
        - Everything else → Wan2GP
        """
        return route_task(task_type, context)
```

**That's the entire dispatch method!**

### Define ComfyUI Task Types

```python
# source/task_engine_router.py

# Which tasks use ComfyUI?
COMFY_TASK_TYPES = {
    "comfy",
    # Add more as needed:
    # "comfy_image",
    # "comfy_video",
}

# Everything else uses Wan2GP automatically
```

## Task Routing Logic

```python
def route_task(task_type: str, context: Dict[str, Any]):
    # Check if ComfyUI task
    if task_type in COMFY_TASK_TYPES:
        return handle_comfy_task(...)  # → ComfyUI engine

    # Everything else goes to Wan2GP
    return handle_wan2gp_task(...)  # → Wan2GP engine
```

## Benefits

✅ **Simple** - Just two code paths
✅ **Clear** - Task type determines engine
✅ **Maintainable** - No specialized handlers to maintain
✅ **Extensible** - Add task types by adding to COMFY_TASK_TYPES

## Task Examples

| Task Type | Engine | What It Does |
|-----------|--------|--------------|
| `vace` | Wan2GP | Video generation with control |
| `flux` | Wan2GP | Image generation |
| `t2v` | Wan2GP | Text to video |
| `i2v` | Wan2GP | Image to video |
| `comfy` | ComfyUI | Workflow-based processing |

All routing happens automatically based on task_type!

## What About Orchestrators?

Current specialized handlers like `travel_orchestrator`, `magic_edit`, etc. are really just:
1. Pre-processing (prepare inputs)
2. Call Wan2GP
3. Post-processing (stitch results)

**Two options:**

### Option 1: Keep As-Is (Backward Compatible)

```python
# Legacy specialized handlers
SPECIALIZED_HANDLERS = {
    "travel_orchestrator",
    "magic_edit",
    "join_clips_orchestrator",
}

def route_task(task_type, context):
    # Check specialized handlers first (for backward compatibility)
    if task_type in SPECIALIZED_HANDLERS:
        return legacy_handlers[task_type](context)

    # Then check ComfyUI
    if task_type in COMFY_TASK_TYPES:
        return handle_comfy_task(...)

    # Everything else → Wan2GP
    return handle_wan2gp_task(...)
```

### Option 2: Refactor to Wan2GP (Clean Slate)

Move orchestration logic into the task params:

```sql
-- Old way: travel_orchestrator task type
INSERT INTO tasks (task_type, params) VALUES ('travel_orchestrator', '{...}');

-- New way: vace task with orchestration params
INSERT INTO tasks (task_type, params) VALUES ('vace', '{
  "orchestration": {
    "type": "travel",
    "segments": [...],
    "stitch": true
  },
  ...
}');
```

Then Wan2GP handler reads orchestration params and does the work.

## Recommendation

**Start simple, add complexity only if needed:**

```python
# Phase 1: Two engines only (NOW)
def route_task(task_type, context):
    if task_type in COMFY_TASK_TYPES:
        return handle_comfy_task(...)
    return handle_wan2gp_task(...)

# Phase 2: Add specialized handlers if truly needed (LATER)
def route_task(task_type, context):
    if task_type in SPECIALIZED_HANDLERS:
        return handlers[task_type](context)
    if task_type in COMFY_TASK_TYPES:
        return handle_comfy_task(...)
    return handle_wan2gp_task(...)
```

**For now:** Just two engines. Add specialization only when you hit a clear use case that can't be handled by params.

## Migration Path

### Current Code (Complex)

```python
handlers = {
    "travel_orchestrator": lambda: ...,
    "travel_segment": lambda: ...,
    "travel_stitch": lambda: ...,
    "magic_edit": lambda: ...,
    "join_clips_orchestrator": lambda: ...,
    "join_clips_segment": lambda: ...,
    "inpaint_frames": lambda: ...,
    "create_visualization": lambda: ...,
    "extract_frame": lambda: ...,
    "rife_interpolate_images": lambda: ...,
    "comfy": lambda: handle_comfy_task(...),
    # 11 different handlers!
}
```

### New Code (Simple)

```python
def route_task(task_type, context):
    if task_type == "comfy":
        return handle_comfy_task(...)

    return handle_wan2gp_task(...)
    # 2 handlers total!
```

## Implementation Steps

1. **Create `task_engine_router.py`** (already done ✅)

2. **Update TaskRegistry to use simple routing:**

```python
# source/task_registry.py

from source.task_engine_router import route_task

class TaskRegistry:
    @staticmethod
    def dispatch(task_type: str, context: Dict[str, Any]):
        return route_task(task_type, context)
```

3. **Test with existing tasks:**
   - Create vace task → Should route to Wan2GP ✅
   - Create flux task → Should route to Wan2GP ✅
   - Create comfy task → Should route to ComfyUI ✅

4. **Deprecate specialized handlers** (optional, later)

## Summary

**Philosophy:** Tasks should be data-driven, not code-driven.

Instead of creating a new handler function for each task type, define behavior through task parameters and let the engines handle it.

**Two engines:**
- Wan2GP: AI generation
- ComfyUI: Workflow processing

**Everything else is just parameters.**
