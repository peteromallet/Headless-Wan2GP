# Debug Tool

Simple tool for investigating tasks in the Headless-Wan2GP system.

## Usage

```bash
# Investigate a specific task
python debug.py task <task_id>

# List recent tasks
python debug.py tasks

# List failed tasks
python debug.py tasks --status Failed --limit 10

# Filter by task type
python debug.py tasks --type join_clips_segment

# Get JSON output
python debug.py tasks --json
```

## Requirements

Requires `.env` with:
- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`
