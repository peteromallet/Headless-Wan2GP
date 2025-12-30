# ComfyUI Worker Documentation

## Architecture

The ComfyUI worker (`comfy_worker.py`) runs alongside the Wan2GP worker (`worker.py`) on RunPod instances spawned by `gpu_orchestrator`.

```
Headless-Wan2GP-Orchestrator (orchestrator repo)
└── gpu_orchestrator/
    ├── main.py              # Spawns RunPod instances
    ├── control_loop.py      # Manages worker lifecycle
    └── runpod_client.py     # RunPod API client

Headless-Wan2GP (worker repo - THIS REPO)
├── worker.py               # Handles Wan2GP tasks (vace, flux, t2v, etc.)
└── comfy_worker.py         # Handles ComfyUI tasks (NEW!)
```

## How It Works

### 1. Task Creation

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

### 2. Orchestrator Spawns Worker

The `gpu_orchestrator` (running on Railway):
1. Detects queued tasks
2. Spawns RunPod GPU instance
3. Worker starts on the instance

### 3. Worker Processes Task

On the RunPod instance, you can run EITHER worker:

**For Wan2GP tasks:**
```bash
python worker.py --worker-id gpu-worker-123
```

**For ComfyUI tasks:**
```bash
python comfy_worker.py --worker-id comfy-worker-123
```

The worker:
1. Starts ComfyUI
2. Claims `task_type='comfy'` tasks
3. Downloads video from `video_url`
4. Uploads to ComfyUI
5. Runs workflow
6. Downloads output
7. Uploads to Supabase storage
8. Marks task complete

## Running on RunPod

### Option 1: Separate Workers

Run both workers on the same machine:

```bash
# Terminal 1: Wan2GP worker
python worker.py --worker-id gpu-worker-123

# Terminal 2: ComfyUI worker
python comfy_worker.py --worker-id comfy-worker-123
```

### Option 2: Single Worker (Future)

Could merge into one worker that handles both task types:

```python
if task_type in ['vace', 'flux', 't2v', 'i2v', ...]:
    # Use Wan2GP
    process_wgp_task(task)
elif task_type == 'comfy':
    # Use ComfyUI
    process_comfy_task(task)
```

## Environment Variables

```bash
# Supabase (required)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your-key

# ComfyUI (required)
COMFY_PATH=/workspace/ComfyUI

# Optional
COMFY_PORT=8188
POLL_INTERVAL=5
```

## Dependencies

The comfy_worker uses:
- `httpx` - HTTP client for ComfyUI API
- `asyncio` - Async task processing
- Headless-Wan2GP's existing:  - `source.db_operations` - Database operations
  - `source.logging_utils` - Logging

## Differences from worker.py

| Aspect | worker.py | comfy_worker.py |
|--------|-----------|----------------|
| **Engine** | Wan2GP (GPU model) | ComfyUI (workflow engine) |
| **Task Types** | vace, flux, t2v, i2v, qwen, etc. | comfy |
| **Input** | Prompts, images, parameters | Workflow JSON + video URL |
| **Output** | Generated videos/images | Workflow outputs |
| **Dependencies** | Wan2GP, HeadlessTaskQueue | ComfyUI server |

## Deployment

### RunPod Docker Image

Build an image with BOTH workers:

```dockerfile
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

# Install Wan2GP
WORKDIR /workspace
RUN git clone https://github.com/yourrepo/Headless-Wan2GP.git
WORKDIR /workspace/Headless-Wan2GP
RUN pip install -r requirements.txt

# Install ComfyUI
WORKDIR /workspace
RUN git clone https://github.com/comfyanonymous/ComfyUI.git
WORKDIR /workspace/ComfyUI
RUN pip install -r requirements.txt

# Start appropriate worker based on task type
WORKDIR /workspace/Headless-Wan2GP
CMD ["python", "comfy_worker.py"]
```

### Start Command Options

**For ComfyUI tasks only:**
```bash
CMD ["python", "comfy_worker.py"]
```

**For Wan2GP tasks only:**
```bash
CMD ["python", "worker.py"]
```

**For both (run in parallel):**
```bash
CMD ["sh", "-c", "python worker.py & python comfy_worker.py"]
```

## Monitoring

### Check Worker Status

```sql
SELECT id, status, current_task_id, last_heartbeat
FROM workers
WHERE id LIKE 'comfy-worker-%'
ORDER BY created_at DESC;
```

### Check Task Status

```sql
SELECT task_id, task_type, status, created_at, started_at, completed_at
FROM tasks
WHERE task_type = 'comfy'
ORDER BY created_at DESC
LIMIT 10;
```

## Troubleshooting

### Worker not claiming tasks

**Check:**
1. Worker is running: `ps aux | grep comfy_worker`
2. ComfyUI started: `curl localhost:8188/system_stats`
3. Database connection: Check Supabase credentials
4. Task type matches: `task_type='comfy'` in database

### ComfyUI fails to start

**Common issues:**
1. Port already in use: `lsof -i :8188`
2. Missing dependencies: `cd /workspace/ComfyUI && pip install -r requirements.txt`
3. Missing models/nodes: Check ComfyUI logs

### Workflow fails

**Check:**
1. Workflow JSON is valid
2. All nodes exist in ComfyUI
3. Required models are downloaded
4. Video format is supported

## Example: Complete Flow

```bash
# 1. Create task
psql> INSERT INTO tasks (...) VALUES ('comfy', '{"workflow": {...}}', 'queued');

# 2. Orchestrator detects task and spawns RunPod instance

# 3. Worker starts on RunPod
root@runpod:/workspace/Headless-Wan2GP# python comfy_worker.py

# Output:
# ComfyUI worker starting: comfy-worker-12345
# Starting ComfyUI at /workspace/ComfyUI
# ComfyUI started with PID: 678
# ComfyUI is ready!
# Worker ready, starting task loop
# Processing ComfyUI task abc-123
# Downloading video from https://example.com/input.mp4
# Uploaded video: input.mp4
# Workflow queued: prompt-xyz
# Workflow completed
# Uploaded result: https://supabase.co/storage/...
# Task abc-123 completed

# 4. Check result
psql> SELECT output_url FROM tasks WHERE task_id = 'abc-123';
# https://supabase.co/storage/v1/object/public/videos/abc-123_output.mp4
```

## Next Steps

- [ ] Merge worker.py and comfy_worker.py into unified worker
- [ ] Add heartbeat support for ComfyUI workers
- [ ] Support multiple output files
- [ ] Add workflow validation before execution
- [ ] GPU memory management between Wan2GP and ComfyUI
