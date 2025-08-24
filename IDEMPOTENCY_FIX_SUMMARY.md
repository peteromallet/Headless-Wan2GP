# Orchestrator Task Idempotency Fix

## Problem Analysis

The issue with task 527bdb56-e0cb-430e-afc4-c9744d811b13 was that orchestrator tasks were creating duplicate child tasks when workers died and the orchestrator was re-processed. This happened because:

1. **No Idempotency Checks**: Orchestrators blindly created child tasks without checking if they already existed
2. **Worker Failures**: When a worker died before marking the orchestrator complete, the task was reset to "Queued"
3. **Re-processing**: A new worker would claim the reset orchestrator and re-create all child tasks
4. **No Cleanup**: The system had no mechanism to detect or remove duplicate child tasks

## Solution Implemented

### 1. New Database Functions (`source/db_operations.py`)

```python
# Query child tasks for an orchestrator
get_orchestrator_child_tasks(orchestrator_task_id: str) -> dict

# Clean up duplicate child tasks  
cleanup_duplicate_child_tasks(orchestrator_task_id: str, expected_segments: int) -> dict

# Helper to delete tasks by ID
_delete_task_by_id(task_id: str) -> bool
```

### 2. Travel Orchestrator Idempotency (`source/sm_functions/travel_between_images.py`)

**Before Task Creation:**
- Check for existing child tasks using `orchestrator_task_id_ref`
- If all expected tasks exist, return success and clean up duplicates
- If partial tasks exist, continue with missing tasks only
- Skip creating segments that already exist

**Smart Task Skipping:**
```python
# Skip if this segment already exists
if idx in existing_segment_indices:
    previous_segment_task_id = existing_segment_task_ids[idx]
    print(f"[IDEMPOTENCY] Skipping segment {idx} - already exists with ID {previous_segment_task_id}")
    continue
```

**Stitch Task Protection:**
```python
# Only create stitch task if it doesn't exist
if not stitch_already_exists:
    # Create stitch task
    stitch_created = 1
else:
    print(f"[IDEMPOTENCY] Skipping stitch task - already exists")
```

### 3. Different Perspective Orchestrator Idempotency (`source/sm_functions/different_perspective.py`)

Applied the same pattern:
- Check for existing child tasks before creation
- Clean up duplicates if all tasks exist
- Continue with missing tasks for partial completion

### 4. Logging and Monitoring

Added comprehensive logging with `[IDEMPOTENCY]` tags:
- Child task existence checks
- Duplicate cleanup actions
- Task skipping decisions
- Summary of actions taken

## Benefits

1. **Prevents Duplicate Tasks**: Orchestrators now check before creating child tasks
2. **Handles Worker Failures**: Re-processed orchestrators skip existing tasks
3. **Automatic Cleanup**: Detects and removes duplicate child tasks
4. **Resource Efficiency**: No wasted compute on duplicate tasks
5. **Data Consistency**: Prevents downstream issues from duplicate data

## Database Queries Added

### SQLite
```sql
-- Find child tasks by orchestrator reference
SELECT id, task_type, status, params 
FROM tasks 
WHERE JSON_EXTRACT(params, '$.orchestrator_task_id_ref') = ?
ORDER BY created_at ASC

-- Delete task by ID
DELETE FROM tasks WHERE id = ?
```

### Supabase
```javascript
// Find child tasks by orchestrator reference
supabase.from("tasks")
  .select("id, task_type, status, params")
  .contains("params", {"orchestrator_task_id_ref": orchestrator_task_id})
  .order("created_at", {ascending: true})

// Delete task by ID  
supabase.from("tasks").delete().eq("id", task_id)
```

## Testing

The fix handles these scenarios:

1. **Fresh Orchestrator**: Creates all child tasks normally
2. **Complete Re-run**: Skips all tasks, cleans up duplicates, returns success
3. **Partial Re-run**: Creates only missing tasks, preserves existing ones
4. **Worker Failure Recovery**: Gracefully handles interrupted orchestrators

## Impact on Existing System

- **Backward Compatible**: Existing orchestrators work unchanged
- **No Breaking Changes**: Only adds protective logic
- **Performance Improvement**: Reduces unnecessary task creation
- **Better Reliability**: Handles worker failures gracefully

## Monitoring

Look for these log patterns:
- `[IDEMPOTENCY] Checking for existing child tasks`
- `[IDEMPOTENCY] Found existing child tasks: X segments, Y stitch tasks`
- `[IDEMPOTENCY] Skipping segment X - already exists`
- `[IDEMPOTENT] Child tasks already exist for orchestrator`

## Future Enhancements

1. **Progress Tracking**: Store orchestrator progress in task parameters
2. **Dependency Validation**: Verify child task dependencies are intact
3. **Status Reconciliation**: Ensure child task statuses are consistent
4. **Orchestrator Resumption**: Smart resume from last successful step
