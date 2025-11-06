# Join Clips Orchestrator

A task orchestration system for progressively joining multiple video clips with seamless transitions.

## Overview

The `join_clips_orchestrator` takes a list of video clips and creates a dependency chain of `join_clips_child` tasks that progressively build them into a single seamless video.

### Sequential Join Pattern

```
Input: [clip_A, clip_B, clip_C, clip_D]

Creates dependency chain:
  join_0: clip_A + clip_B → AB.mp4 (no dependency)
  join_1: AB.mp4 + clip_C → ABC.mp4 (depends on join_0)
  join_2: ABC.mp4 + clip_D → ABCD.mp4 (depends on join_1)

Final output: ABCD.mp4 (seamless video with transitions)
```

## Architecture

### Components

1. **join_clips_orchestrator** (`source/sm_functions/join_clips_orchestrator.py`)
   - Creates chained `join_clips_child` tasks
   - Implements idempotency checking
   - Monitors child task completion
   - Returns `[ORCHESTRATOR_COMPLETE]` when all joins finish

2. **join_clips_child** (`source/sm_functions/join_clips.py`)
   - Wrapper around existing `join_clips` task
   - Fetches predecessor output via `get_predecessor_output_via_edge_function()`
   - Delegates to `_handle_join_clips_task()` for actual joining

3. **Edge Function** (`supabase/functions/get-predecessor-output/`)
   - Queries task's `dependant_on` field
   - Returns predecessor's `output_location`
   - Already existed - no new edge function needed!

### Database Structure

**Orchestrator Task:**
```json
{
  "task_type": "join_clips_orchestrator",
  "params": {
    "orchestrator_details": {
      "clip_list": [
        {"url": "path/to/clip1.mp4", "name": "morning"},
        {"url": "path/to/clip2.mp4", "name": "afternoon"},
        {"url": "path/to/clip3.mp4", "name": "evening"}
      ],
      "context_frame_count": 8,
      "gap_frame_count": 53,
      "replace_mode": false,
      "prompt": "smooth transition",
      "run_id": "run_abc123",
      "output_base_dir": "/workspace/outputs/"
    }
  }
}
```

**Child Task:**
```json
{
  "task_type": "join_clips_child",
  "dependant_on": "previous_join_task_id",  // DB field, not in params
  "params": {
    "orchestrator_task_id_ref": "orchestrator_uuid",
    "orchestrator_run_id": "run_abc123",
    "join_index": 1,
    "is_first_join": false,
    "is_last_join": false,
    "starting_video_path": null,  // Fetched from predecessor
    "ending_video_path": "path/to/clip3.mp4",
    "context_frame_count": 8,
    "gap_frame_count": 53,
    // ... other join_clips parameters
  }
}
```

## Usage

### Via Python API

```python
from source import db_operations as db_ops

# Build orchestrator payload
orchestrator_payload = {
    "clip_list": [
        {"url": "s3://bucket/morning.mp4", "name": "morning"},
        {"url": "s3://bucket/afternoon.mp4", "name": "afternoon"},
        {"url": "s3://bucket/evening.mp4", "name": "evening"},
    ],
    "context_frame_count": 8,
    "gap_frame_count": 53,
    "replace_mode": False,
    "blend_frames": 3,
    "prompt": "smooth cinematic transition",
    "model": "lightning_baseline_2_2_2",
    "run_id": "timelapse_001",
    "output_base_dir": "/workspace/outputs/",
}

# Submit orchestrator task
task_id = db_ops.add_task_to_db(
    task_payload={"orchestrator_details": orchestrator_payload},
    task_type_str="join_clips_orchestrator",
    dependant_on=None
)
```

### Via Test Script

```bash
python test_join_clips_orchestrator.py \
  --clip1 /path/to/morning.mp4 \
  --clip2 /path/to/afternoon.mp4 \
  --clip3 /path/to/evening.mp4 \
  --context-frames 8 \
  --gap-frames 53 \
  --prompt "smooth time transition" \
  --aspect-ratio "16:9"
```

## Parameters

### Orchestrator Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `clip_list` | `list[dict]` | Yes | - | List of clips with `url` and `name` fields |
| `run_id` | `str` | Yes | - | Unique identifier for this orchestrator run |
| `context_frame_count` | `int` | No | 8 | Frames to extract from each clip boundary |
| `gap_frame_count` | `int` | No | 53 | Frames to generate/replace at transitions |
| `replace_mode` | `bool` | No | False | Replace boundary frames (True) vs insert (False) |
| `blend_frames` | `int` | No | 3 | Frames for crossfade blending |
| `prompt` | `str` | No | "smooth transition" | Generation prompt |
| `negative_prompt` | `str` | No | "" | Negative prompt |
| `model` | `str` | No | "lightning_baseline_2_2_2" | Model for generation |
| `aspect_ratio` | `str` | No | None | Standardize clips (e.g., "16:9") |
| `output_base_dir` | `str` | No | "./outputs/" | Base output directory |
| `per_join_settings` | `list[dict]` | No | [] | Per-join overrides (see below) |

### Per-Join Overrides

You can override settings for individual joins:

```python
orchestrator_payload = {
    "clip_list": [...],
    "prompt": "smooth transition",  # Default for all
    "per_join_settings": [
        {"prompt": "fade to sunset", "gap_frame_count": 30},  # Override for join 0
        {"prompt": "night falls"},  # Override for join 1
    ]
}
```

## Execution Flow

1. **Orchestrator Creation**
   - Worker claims `join_clips_orchestrator` task
   - Orchestrator handler creates `join_clips_child` tasks in dependency chain
   - Task status: `In Progress` (waiting for children)

2. **Child Task Execution** (sequential due to dependencies)
   - Worker claims `join_clips_child` task when dependency satisfied
   - If `starting_video_path` is `None`, fetch from predecessor
   - Delegate to `_handle_join_clips_task()` for joining
   - Save output for next child to use

3. **Orchestrator Completion**
   - When last child completes, orchestrator polls and detects completion
   - Returns `[ORCHESTRATOR_COMPLETE]<final_output_path>`
   - Worker marks orchestrator as `Complete`

## Idempotency

The orchestrator implements idempotency checking:

- If orchestrator is re-run, checks for existing child tasks
- If all children exist and are complete, returns final output immediately
- If any child failed, marks orchestrator as failed
- If children are in progress, returns status message

This prevents duplicate task creation on retries.

## Error Handling

### Child Task Failure

If a `join_clips_child` task fails:
- Task marked as `Failed` in database
- Subsequent child tasks blocked (due to dependency)
- Orchestrator detects failure and marks itself as `Failed`

### Orchestrator Failure

If orchestrator creation fails:
- No child tasks created
- Orchestrator marked as `Failed`
- Error message stored in `output_location`

## Output Structure

```
{output_base_dir}/
└── join_clips_run_{run_id}/
    ├── join_0/
    │   └── joined_output.mp4  (clip_A + clip_B)
    ├── join_1/
    │   └── joined_output.mp4  (AB + clip_C)
    └── join_2/
        └── joined_output.mp4  (ABC + clip_D) ← Final output
```

## Comparison to Travel Orchestrator

| Feature | Travel Orchestrator | Join Clips Orchestrator |
|---------|---------------------|-------------------------|
| Purpose | Generate video segments between images | Join existing video clips |
| Child Tasks | `travel_segment` + `travel_stitch` | `join_clips_child` only |
| Pattern | Parallel segments → stitch | Sequential joins |
| Stitch Task | Yes (combines all segments) | No (progressive joining) |
| Final Output | From stitch task | From last child join |

## Future Enhancements

### Parallel Join + Stitch Pattern

For very long clip lists (10+ clips), implement parallel joining:

```
Input: [A, B, C, D, E, F, G, H]

Parallel Phase:
  join_0: A + B → AB (parallel)
  join_1: C + D → CD (parallel)
  join_2: E + F → EF (parallel)
  join_3: G + H → GH (parallel)

Stitch Phase:
  stitch: [AB, CD, EF, GH] → final.mp4
```

**Benefits:**
- Faster for large clip counts
- Better GPU utilization

**Tradeoff:**
- More complex stitch logic required

## Testing

Run the test script with example clips:

```bash
# Simple 2-clip test
python test_join_clips_orchestrator.py \
  --clip1 examples/clip1.mp4 \
  --clip2 examples/clip2.mp4

# 3-clip test with custom settings
python test_join_clips_orchestrator.py \
  --clip1 morning.mp4 \
  --clip2 afternoon.mp4 \
  --clip3 evening.mp4 \
  --context-frames 12 \
  --gap-frames 40 \
  --replace-mode \
  --prompt "smooth day progression"
```

## Troubleshooting

### "Failed to fetch predecessor output"

**Cause:** Previous join task hasn't completed or failed
**Solution:** Check dependency chain, ensure previous tasks succeeded

### "starting_video_path is required for first join"

**Cause:** First `join_clips_child` has `starting_video_path=None`
**Solution:** Bug in orchestrator - first join must have explicit path

### Orchestrator stuck "In Progress"

**Cause:** Child tasks not completing
**Solution:** Check worker logs, verify child tasks are being claimed and executed

## Code References

- Orchestrator handler: `source/sm_functions/join_clips_orchestrator.py:27`
- Child task handler: `source/sm_functions/join_clips.py:784`
- Worker integration: `worker.py:2626` (orchestrator), `worker.py:2641` (child)
- Database helper: `source/db_operations.py:856` (get child tasks)
- Edge function: `supabase/functions/get-predecessor-output/index.ts:22`
