# Task Visibility Debug Improvements

## Issue Summary
Segment 0 task was created successfully but never processed by the worker, while Segment 1 was claimed and completed normally. Investigation revealed potential database replication lag or consistency issues.

## Root Cause Analysis
- Task creation via Edge Function returned 200 (success)
- Task took 2+ seconds to become visible in database queries
- Task-counts showed inconsistent data: `queued_only: 0` but `eligible_queued: 1`
- Worker skipped claim attempts based on `queued_only: 0`
- Task either disappeared or remained invisible to read replicas

## Changes Made

### 1. Enhanced Task Creation Verification (`source/db_operations.py`)
**Function:** `add_task_to_db()`

**Changes:**
- Increased verification retries from 5 to 10 (handles longer replication delays)
- Added comprehensive logging with timing information
- Query additional fields (status, created_at, project_id, task_type) for debugging
- Added clear visual indicators (✅, ⚠️, ❌) for different outcomes
- Log detailed error when task is not visible after all retries

**New Log Output:**
```
[VERIFY] ✅ Task {id} verified visible and Queued (took 2.34s)
[VERIFY] ⚠️  Task {id} already In Progress (claimed in 0.5s - unusually fast)
[ERROR] ❌ Task {id} creation confirmed by Edge Function but NOT VISIBLE in DB after 10 attempts (10.2s)
```

### 2. Enhanced Task Claiming Logic (`source/db_operations.py`)
**Function:** `get_oldest_queued_task_supabase()`

**Changes:**
- Log all task-counts fields (queued_only, eligible_queued, active_only)
- Detect and warn about inconsistent counts
- Proceed with claim attempt if `eligible_queued > 0` even when `queued_only = 0`
- Add detailed claim success logging with task type and segment index

**New Log Output:**
```
[CLAIM_DEBUG] Task counts: queued_only=0, eligible_queued=1, active_only=1
[WARN] ⚠️  Task count inconsistency detected: eligible_queued=1 but queued_only=0
[WARN] This suggests tasks exist but aren't visible as 'Queued' status - possible replication lag or status corruption
[CLAIM] ✅ Claimed task {id} (type=travel_segment, segment_index=1)
```

### 3. Enhanced Orchestrator Logging (`source/sm_functions/travel_between_images.py`)
**Function:** `_handle_travel_orchestrator_task()`

**Changes:**
- Add visible print statements for segment creation start/completion
- Log task IDs immediately after creation
- Log dependency verification results at INFO level (not just DEBUG)
- Add warnings for dependency verification failures

**New Log Output:**
```
[ORCHESTRATOR] Creating segment 0 task...
[ORCHESTRATOR] ✅ Segment 0 created: task_id=970504f6-b476-417d-b5eb-758df5584731
[ORCHESTRATOR] Segment 0 dependency verified: dependant_on=None
[WARN] ⚠️  Segment 0 dependency verification failed: {error}
```

## What These Changes Will Reveal Next Time

### Scenario 1: Database Replication Lag
```
[VERIFY] Attempt 1/10: Task {id} not visible yet (0 rows)
[VERIFY] Attempt 2/10: Task {id} not visible yet (0 rows)
[VERIFY] ✅ Task {id} verified visible and Queued (took 2.34s)
```
**Diagnosis:** Normal replication lag, task eventually visible

### Scenario 2: Task Never Becomes Visible
```
[VERIFY] Attempt 10/10: Task {id} not visible yet (0 rows)
[ERROR] ❌ Task {id} creation confirmed by Edge Function but NOT VISIBLE in DB after 10 attempts (10.2s)
[ERROR] This task may be lost due to database replication lag or consistency issues!
```
**Diagnosis:** Serious database consistency problem, task lost

### Scenario 3: Task Count Inconsistency
```
[CLAIM_DEBUG] Task counts: queued_only=0, eligible_queued=1, active_only=1
[WARN] ⚠️  Task count inconsistency detected
[CLAIM_DEBUG] Proceeding with claim attempt despite queued_only=0 because eligible_queued=1
[CLAIM] ✅ Claimed task {id} (type=travel_segment, segment_index=0)
```
**Diagnosis:** Task-counts edge function has bugs or is querying stale replicas

### Scenario 4: Unusually Fast Claim
```
[VERIFY] ⚠️  Task {id} already In Progress (claimed in 0.5s - unusually fast)
```
**Diagnosis:** Task was claimed by another worker almost immediately (race condition)

## Monitoring Recommendations

1. **Watch for replication lag warnings** - if tasks consistently take >2s to become visible, investigate database replication health
2. **Track task count inconsistencies** - if `eligible_queued != queued_only` frequently, the task-counts edge function needs fixing
3. **Monitor lost tasks** - any `❌ NOT VISIBLE` errors indicate serious problems requiring immediate attention
4. **Check claim patterns** - if wrong segments are being claimed (e.g., Segment 1 before Segment 0), investigate ordering logic

## Next Steps if Issue Persists

1. **Add database read consistency hints** to force reads from primary
2. **Implement task-counts edge function** (currently empty locally) with proper logic
3. **Add task status change tracking** to log all transitions (Queued → In Progress → Complete)
4. **Consider adding a task heartbeat** to detect stuck tasks
5. **Investigate Supabase replication lag metrics** in dashboard

