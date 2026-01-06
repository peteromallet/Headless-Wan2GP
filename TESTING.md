# Testing the Worker

Quick guide to test the worker with real task configurations.

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

```bash
source venv/bin/activate && \
python worker.py --supabase-url https://wczysqzxlwdndgxitrvc.supabase.co \
  --supabase-anon-key eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6IndjenlzcXp4bHdkbmRneGl0cnZjIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTE1MDI4NjgsImV4cCI6MjA2NzA3ODg2OH0.r-4RyHZiDibUjgdgDDM2Vo6x3YpgIO5-BTwfkB2qyYA \
  --supabase-access-token 3HKcoLeJAFFfTFFeRV6Eu7Lq --wgp-profile 4
```

## Check Results

```bash
# Check task status
python debug.py task <task_id>

# Get full JSON output
python debug.py task <task_id> --json

# View just logs
python debug.py task <task_id> --logs-only
```

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

```bash
# 1. Create task
python create_test_task.py travel_orchestrator
# Note the task ID

# 2. Start worker (will pick up the task)
python worker.py --supabase-url ... --wgp-profile 4

# 3. Monitor in another terminal
python debug.py task <task_id>

# 4. If failed, check logs
python debug.py task <task_id> --logs-only | grep -E "LORA|ERROR|WARNING"
```

