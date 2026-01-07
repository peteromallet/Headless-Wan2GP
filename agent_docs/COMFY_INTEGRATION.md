# ComfyUI Integration

## Overview

ComfyUI tasks are handled through the **TaskRegistry** system, just like all other task types. The worker automatically routes `task_type='comfy'` to the ComfyUI handler with **lazy-loading** - ComfyUI only starts when the first comfy task is encountered.

## Architecture

```
Task Created (task_type='comfy')
    ↓
Database (tasks table)
    ↓
worker.py polls database
    ↓
TaskRegistry.dispatch('comfy', context)
    ↓
comfy_handler.handle_comfy_task()
    ↓ (first call only)
Lazy-start ComfyUI server
    ↓
Download video → Upload to ComfyUI → Run workflow → Save output
    ↓
Task marked complete
```

## Files

| File | Purpose |
|------|---------|
| `source/comfy_utils.py` | ComfyUI server management and API client |
| `source/comfy_handler.py` | Task handler (registered in TaskRegistry) |
| `source/task_registry.py` | Routes tasks to handlers (includes comfy) |
| `worker.py` | Main worker (unchanged, uses TaskRegistry) |

## How It Works

### 1. Task Routing

The worker uses `TaskRegistry.dispatch()` to route all tasks:

```python
# worker.py (existing code, unchanged)
while True:
    task_info = db_ops.get_oldest_queued_task_supabase(worker_id)

    if task_info:
        task_type = task_info['task_type']

        # ONE line routes ALL task types
        success, output = TaskRegistry.dispatch(task_type, context)
```

The TaskRegistry checks its handlers dictionary:

```python
# task_registry.py
handlers = {
    "travel_orchestrator": ...,
    "magic_edit": ...,
    "comfy": lambda: handle_comfy_task(...),  # ← Added!
    "rife_interpolate_images": ...,
}

if task_type in handlers:
    return handlers[task_type]()
```

### 2. Lazy-Loading

ComfyUI **does not start on worker init**. It starts only when needed:

```python
# comfy_handler.py
_comfy_manager = None  # Global, starts as None

async def _ensure_comfy_running():
    if _comfy_manager is not None:
        return True  # Already running

    # First comfy task - start ComfyUI now
    manager = ComfyUIManager(COMFY_PATH, COMFY_PORT)
    manager.start()
    await manager.wait_for_ready()

    _comfy_manager = manager
    return True
```

**Benefits:**
- Workers without ComfyUI: Process vace/flux/t2v normally ✅
- Workers with ComfyUI but no comfy tasks: Zero overhead ✅
- Workers with ComfyUI and comfy tasks: Auto-starts on first task ✅

### 3. Graceful Degradation

If ComfyUI is not available:

```python
# comfy_handler.py handles errors gracefully
try:
    manager = ComfyUIManager(COMFY_PATH, COMFY_PORT)
    manager.start()
except FileNotFoundError:
    # ComfyUI not installed
    return False, "ComfyUI not available on this worker"
except Exception as e:
    # Startup failed
    return False, f"ComfyUI startup failed: {e}"
```

The task fails with a clear error message, but **other tasks continue working**.

## Task Format

Create a ComfyUI task in Supabase:

```sql
INSERT INTO tasks (task_type, params, status)
VALUES (
  'comfy',
  '{
    "workflow": {
      "1": {
        "inputs": {
          "video": "input.mp4",
          "force_rate": 0,
          "custom_width": 0,
          "custom_height": 0,
          "frame_load_cap": 0,
          "skip_first_frames": 0,
          "select_every_nth": 1
        },
        "class_type": "VHS_LoadVideo"
      },
      "2": {
        "inputs": {
          "images": ["1", 0],
          "frame_rate": 30,
          "format": "video/h264-mp4",
          "save_output": true,
          "pingpong": false
        },
        "class_type": "VHS_VideoCombine"
      }
    },
    "video_url": "https://example.com/input.mp4",
    "video_node_id": "1",
    "video_input_field": "video"
  }',
  'queued'
);
```

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `workflow` | Object | Yes | ComfyUI workflow JSON |
| `video_url` | String | No | URL to download input video |
| `video_node_id` | String | No | Node ID to inject video filename |
| `video_input_field` | String | No | Field name for video (default: "video") |

## Environment Variables

```bash
# Optional - defaults to /workspace/ComfyUI
COMFY_PATH=/workspace/ComfyUI

# Optional - defaults to 8188
COMFY_PORT=8188
```

## Deployment

### Docker Image

Your RunPod Docker image needs:

1. **ComfyUI installed**
2. **Required custom nodes** (e.g., VHS for video)
3. **Worker code** (Headless-Wan2GP)

```dockerfile
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Install ComfyUI
WORKDIR /workspace
RUN git clone https://github.com/comfyanonymous/ComfyUI.git && \
    cd ComfyUI && \
    pip install -r requirements.txt

# Install VHS (Video Helper Suite)
WORKDIR /workspace/ComfyUI/custom_nodes
RUN git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git && \
    cd ComfyUI-VideoHelperSuite && \
    pip install -r requirements.txt

# Clone worker repository
WORKDIR /workspace
RUN git clone https://github.com/your-org/Headless-Wan2GP.git && \
    cd Headless-Wan2GP && \
    pip install -r requirements.txt

# Start worker (ComfyUI will lazy-load when needed)
WORKDIR /workspace/Headless-Wan2GP
CMD ["python", "worker.py", "--worker", "$WORKER_ID", "--supabase-url", "$SUPABASE_URL", "--supabase-access-token", "$SUPABASE_SERVICE_ROLE_KEY"]
```

**Key points:**
- Only ONE worker process (`worker.py`)
- ComfyUI installed but not running on startup
- ComfyUI starts automatically when first comfy task arrives

### Without ComfyUI

If you deploy a worker **without ComfyUI**:

```dockerfile
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Skip ComfyUI installation

# Clone worker repository
WORKDIR /workspace
RUN git clone https://github.com/your-org/Headless-Wan2GP.git && \
    cd Headless-Wan2GP && \
    pip install -r requirements.txt

# Start worker
WORKDIR /workspace/Headless-Wan2GP
CMD ["python", "worker.py", "--worker", "$WORKER_ID", ...]
```

**What happens:**
- Worker processes vace/flux/t2v tasks perfectly ✅
- If a comfy task is claimed, it fails with: "ComfyUI not available on this worker" ❌
- Worker continues processing other tasks ✅

## Testing

### Local Testing

```bash
cd /workspace/Headless-Wan2GP

# Start worker
python worker.py \
  --worker test-worker \
  --supabase-url https://your-project.supabase.co \
  --supabase-access-token your-key

# In another terminal, create a test task
psql -c "INSERT INTO tasks (task_type, params, status) VALUES ('comfy', '{...}', 'queued');"

# Watch logs
tail -f logs/worker.log
```

Expected output:
```
Worker test-worker started. Polling every 10s.
Processing comfy task abc-123
First ComfyUI task detected - starting ComfyUI server...
Starting ComfyUI at /workspace/ComfyUI
ComfyUI started with PID: 12345
ComfyUI is ready!
✅ ComfyUI started successfully and is ready for tasks
Downloading video from https://...
Uploaded video: input.mp4
Workflow queued: prompt-xyz
Workflow completed
Saved output: /workspace/Headless-Wan2GP/outputs/comfy/abc-123_output.mp4
Task abc-123 completed
```

### Production Testing

1. **Deploy Docker image** with ComfyUI
2. **Update Railway env var:** `RUNPOD_WORKER_IMAGE=your-registry/wan2gp-comfy:latest`
3. **Create test task:**
   ```sql
   INSERT INTO tasks (task_type, params, status)
   VALUES ('comfy', '{"workflow": {...}}', 'queued');
   ```
4. **Verify:**
   - Orchestrator spawns worker
   - Worker claims task
   - ComfyUI starts (check logs for "ComfyUI started")
   - Workflow executes
   - Output saved to `/outputs/comfy/`
   - Task status = Complete

## Monitoring

### Check if ComfyUI is running

```bash
# SSH to worker
ssh root@<runpod-ip>

# Check ComfyUI process
ps aux | grep ComfyUI

# Check ComfyUI API
curl localhost:8188/system_stats
```

### Check task distribution

```sql
SELECT
  task_type,
  status,
  COUNT(*) as count
FROM tasks
WHERE created_at > NOW() - INTERVAL '1 hour'
GROUP BY task_type, status
ORDER BY task_type, status;
```

### Check outputs

```bash
# On worker machine
ls -lh /workspace/Headless-Wan2GP/outputs/comfy/
```

## Troubleshooting

### Issue: "ComfyUI not available on this worker"

**Cause:** Worker doesn't have ComfyUI installed

**Solution:**
- Use Docker image with ComfyUI
- Or set `COMFY_PATH` to correct location

### Issue: "ComfyUI failed to become ready"

**Cause:** ComfyUI startup failed

**Check:**
```bash
# Check ComfyUI exists
ls -la /workspace/ComfyUI/main.py

# Try manual start
cd /workspace/ComfyUI
python main.py --listen 0.0.0.0 --port 8188

# Check dependencies
pip install -r /workspace/ComfyUI/requirements.txt
```

### Issue: "No outputs generated by workflow"

**Cause:** Workflow didn't produce video outputs

**Check:**
1. Workflow JSON is valid
2. Required custom nodes installed (e.g., VHS)
3. Output node is configured with `save_output: true`

### Issue: Task fails but other tasks work fine

**Expected behavior!** ComfyUI tasks fail gracefully without affecting other tasks.

**Fix:** Ensure ComfyUI is properly installed and configured on workers that should handle comfy tasks.

## Migration from Separate Workers

If you were using the old `comfy_worker.py` approach:

### Old Architecture (Separate Workers)
```bash
CMD ["sh", "-c", "python worker.py & python comfy_worker.py"]
```
- Two processes
- Both poll database
- comfy_worker.py filters by task_type

### New Architecture (Single Worker)
```bash
CMD ["python", "worker.py", ...]
```
- One process
- TaskRegistry routes tasks
- ComfyUI lazy-loads when needed

### Migration Steps

1. **Update Docker image** to use new CMD (remove `comfy_worker.py` reference)
2. **Deploy** - worker will automatically use TaskRegistry routing
3. **Test** - create a comfy task and verify it processes

**No database changes needed!** Task format is identical.

## Summary

✅ **Single worker** - One process handles all task types
✅ **Lazy-loading** - ComfyUI only starts when needed
✅ **Safe** - Graceful degradation if ComfyUI unavailable
✅ **Extensible** - Add new engines by registering handlers
✅ **Production-ready** - Zero risk to existing tasks

ComfyUI is now just another task handler in the registry, following the same pattern as `travel_orchestrator`, `magic_edit`, and all other specialized handlers.
