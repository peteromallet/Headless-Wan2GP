# Testing the Worker

Quick guide to test the worker with real task configurations.

## Recent Fixes

### LoRA Directory Path Fix (2026-01-06)
**Issue**: LoRAs were being downloaded to `loras/` but WGP expected them in `loras/wan/` for Wan 2.2 models, causing "LoRAs files are missing or invalid" errors.

**Fix**: Updated `source/lora_utils.py:_download_lora_from_url()` to accept `model_type` parameter and download to the correct model-specific directory:
- Wan 2.2 models (VACE, etc.): `loras/wan/`
- Other models: `loras/`

**Files Changed**:
- `source/lora_utils.py` - Added model_type parameter and directory logic
- `headless_model_management.py` - Pass model_type when calling download function
- `TESTING.md` - Added documentation for monitoring, preload models, and this fix

## Automated Tests (Unit/Integration)

Run the LoRA flow test suite:

```bash
python -m pytest tests/test_lora_flow.py -v
```

This runs 18 tests covering:
- URL detection and PENDING status
- Phase config parsing (2 and 3 phases)
- Download simulation and WGP format conversion
- Deduplication and edge cases

## Prerequisites

1. **Environment**: Ensure `.env` has `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY`
2. **Venv**: `source venv/bin/activate`

## Create a Test Task

```bash
# List available test tasks
python create_test_task.py --list

# Create a travel_orchestrator task (tests 3-phase Lightning LoRAs)
python create_test_task.py travel_orchestrator

# Create a qwen_image_style task
python create_test_task.py qwen_image_style

# Dry run (see what would be created)
python create_test_task.py travel_orchestrator --dry-run
```

## Run the Worker

**Important**: Preload the correct model for your test task to avoid unnecessary model switches.

```bash
# For travel_orchestrator (VACE model)
source venv/bin/activate && \
python worker.py --supabase-url https://wczysqzxlwdndgxitrvc.supabase.co \
  --supabase-anon-key eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6IndjenlzcXp4bHdkbmRneGl0cnZjIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTE1MDI4NjgsImV4cCI6MjA2NzA3ODg2OH0.r-4RyHZiDibUjgdgDDM2Vo6x3YpgIO5-BTwfkB2qyYA \
  --supabase-access-token 3HKcoLeJAFFfTFFeRV6Eu7Lq --wgp-profile 1 \
  --preload-model wan_2_2_vace_lightning_baseline_2_2_2

# For qwen_image_style (Qwen model)
source venv/bin/activate && \
python worker.py --supabase-url https://wczysqzxlwdndgxitrvc.supabase.co \
  --supabase-anon-key eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6IndjenlzcXp4bHdkbmRneGl0cnZjIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTE1MDI4NjgsImV4cCI6MjA2NzA3ODg2OH0.r-4RyHZiDibUjgdgDDM2Vo6x3YpgIO5-BTwfkB2qyYA \
  --supabase-access-token 3HKcoLeJAFFfTFFeRV6Eu7Lq --wgp-profile 1 \
  --preload-model qwen_image_edit_20B
```

## Monitor Worker Logs in Real-Time

When running the worker, the output is written to a background process. You can monitor it in real-time:

```bash
# If you started the worker with the command above, the output is in a background shell
# Find the shell ID using: /tasks (in Claude Code) or ps aux | grep worker.py

# Monitor the output file directly (replace TASK_ID with actual background task ID)
tail -f /tmp/claude/-workspace/tasks/TASK_ID.output

# Or periodically check progress (recommended pattern):
sleep 60 && tail -50 /tmp/claude/-workspace/tasks/TASK_ID.output

# Repeatedly check progress every 60 seconds:
while true; do
  sleep 60
  echo "=== $(date) ==="
  tail -50 /tmp/claude/-workspace/tasks/TASK_ID.output
  echo ""
done

# Check process is still running
ps aux | grep worker.py | grep -v grep

# Check process CPU, RAM, and runtime
ps aux | grep -E "[p]ython worker.py"
# Output columns: USER PID %CPU %MEM VSZ RSS TTY STAT START TIME COMMAND
# - %CPU: Percentage of CPU (>100% means multi-core usage)
# - %MEM: Percentage of system memory
# - RSS: Resident memory in KB (divide by 1024 for MB, 1048576 for GB)
# - TIME: Total CPU time used
# - STAT: Process state (S=sleeping, R=running, l=multi-threaded)

# Example: Check if worker is loading models or idle
# - High CPU (>50%) + Growing RAM = Model loading
# - Low CPU (<5%) + Stable RAM = Idle/waiting for tasks
# - Medium CPU (10-30%) + Stable RAM = Processing/generating
```

**Recommended monitoring pattern**: Run `sleep 60 && tail -50` repeatedly to check in on progress every minute. **Send a status update after each check** to track progress over time. This is especially useful during:
- Model loading (2-5 minutes): Expect high CPU (>100%), growing RAM (15-50GB)
- Video generation (1-10 minutes depending on length): Expect medium CPU (10-30%), stable RAM
- LoRA downloads (a few seconds each): Expect network activity

**For automated monitoring**: After each sleep cycle, report:
- Timestamp
- New log lines (or "No new output")
- Process stats if available
- Any errors or warnings detected
```

### Worker Log Stages

1. **Initialization** (first 10-20 seconds)
   - Queue setup, orchestrator initialization
   - Look for: `✅ ORCHESTRATOR WanOrchestrator initialized`

2. **Model Loading** (2-5 minutes)
   - Large model files being loaded into memory
   - Look for: `Loading Model 'ckpts/wan2.2_image2video_14B_*.safetensors'`

3. **Task Processing** (varies)
   - Task picked from queue and executed
   - Look for: `Processing task`, `[LORA_PROCESS]`, `activated_loras:`

4. **Completion**
   - Task status updated in Supabase
   - Look for: `✅ Task completed`, `Task status: completed`

## Check Results

```bash
# Check task status
python debug.py task <task_id>

# Get full JSON output
python debug.py task <task_id> --json

# View just logs
python debug.py task <task_id> --logs-only

# Filter for LoRA-related logs
python debug.py task <task_id> --logs-only 2>&1 | grep -E "LORA|LoRA|lora"

# Filter for errors/warnings
python debug.py task <task_id> --logs-only 2>&1 | grep -E "ERROR|WARNING|❌"
```

### Key Log Tags to Look For

| Tag | Meaning |
|-----|---------|
| `[LORA_PROCESS]` | LoRA handling in queue |
| `[LORA_DOWNLOAD]` | URL download attempts (downloads to model-specific dir) |
| `[TASK_CONVERSION]` | TaskConfig parsing |
| `activated_loras:` | Final LoRAs sent to WGP |
| `LoRAs need downloading` | Pending URLs detected |
| `Downloaded` | Successful download |

**Note on LoRA directories**: LoRAs are automatically downloaded to the correct model-specific directory:
- Wan 2.2 models (VACE, etc.): `loras/wan/`
- Other models: `loras/`

This ensures WGP can find the LoRA files when validating them.

## Test Task Details

### `travel_orchestrator`
- **Model**: `wan_2_2_vace_lightning_baseline_2_2_2`
- **LoRAs**: 3-phase config with Lightning high/low noise + 14b-i2v
- **Tests**: Phase config parsing, LoRA URL download, multiplier formatting

### `qwen_image_style`
- **Model**: `qwen-image` (qwen_image_edit_20B)
- **LoRAs**: Lightning phases via `lightning_lora_strength_phase_1/2`
- **Tests**: Qwen handler, style reference, hires fix

## Typical Test Flow

**IMPORTANT**: Create the test task BEFORE launching the worker, so the worker can pick it up immediately.

```bash
# 1. Create task FIRST (before starting worker)
python create_test_task.py travel_orchestrator
# Note the task ID returned

# 2. THEN start worker with correct model preloaded (will pick up the task)
python worker.py --supabase-url ... --wgp-profile 1 \
  --preload-model wan_2_2_vace_lightning_baseline_2_2_2

# 3. Monitor progress (check every 60 seconds)
sleep 60 && tail -50 /tmp/claude/-workspace/tasks/<TASK_ID>.output

# 4. Check task status via debug.py
python debug.py task <task_id>

# 5. If failed, check logs
python debug.py task <task_id> --logs-only | grep -E "LORA|ERROR|WARNING"
```

