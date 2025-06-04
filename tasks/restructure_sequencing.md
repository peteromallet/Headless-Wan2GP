# Restructuring Task Sequencing for `travel_between_images`

**Objective**
Create **all** segment-level tasks (and the final stitch task) up-front and rely on a *dependency chain* so a task only starts when the task it depends on has completed.  This removes the current _enqueue-while-running_ approach and makes the queue predictable, observable and more suitable for parallel workers.

---
## 1. High-level Workflow After the Change

```
travel_orchestrator  (root – no dependency)
│
├─ travel_segment_00                 (dependant_on = NULL)
│    └─ wgp_sub_00                   (dependant_on = travel_segment_00)†
│
├─ travel_segment_01                 (dependant_on = wgp_sub_00)
│    └─ wgp_sub_01                   (dependant_on = travel_segment_01)
│
├─ …
│
└─ travel_stitch                     (dependant_on = wgp_sub_last)
```
† If you decide to keep the nested **wgp_sub** tasks.  Alternatively the segment handler can perform WGP itself and then the chain simply connects the `travel_segment_*` tasks.

---
## 2. Database Schema Changes

1. **Add a dependency column.**
   ```sql
   ALTER TABLE tasks ADD COLUMN dependant_on TEXT NULL;
   CREATE INDEX IF NOT EXISTS idx_dependant_on ON tasks(dependant_on);
   ```
   *`dependant_on`* stores **one** upstream task_id.  A task with no prerequisites will have `NULL`.

2. **SQLite**
   * Update `_ensure_db_initialized` in `steerable_motion.py` and `init_db()` in `headless.py` so the `CREATE TABLE` includes the new column.
   * Extend `_migrate_sqlite_schema()` with an `ALTER TABLE` guard so existing DBs are upgraded automatically.

3. **Supabase / Postgres**
   * Add `dependant_on TEXT` to `func_initialize_tasks_table` SQL.
   * Create a new RPC migration (`func_migrate_tasks_add_dependant_on`) or extend `func_migrate_tasks_for_task_type` to perform `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`.

---
## 3. Python Helper Updates

### 3.1 `sm_functions.common_utils.add_task_to_db`
*Signature change*
```python
def add_task_to_db(task_payload: dict, db_path: str | Path, task_type_str: str, dependant_on: str | None = None):
```
* Insert the new argument into the `INSERT` statement:  
  `..., task_type, status, output_location, dependant_on, created_at, ...`.

### 3.2 Queue-picker logic

#### SQLite (`headless.get_oldest_queued_task`)
Replace the select with a JOIN so only tasks whose dependency is **either NULL or COMPLETE** are returned:
```sql
SELECT t.id, t.params, t.task_type
FROM   tasks AS t
LEFT JOIN tasks AS d             ON d.id = t.dependant_on
WHERE  t.status = 'Queued'
  AND (t.dependant_on IS NULL OR d.status = 'Complete')
ORDER BY t.created_at ASC
LIMIT  1;
```
Do the same inside `func_claim_task` for Supabase.

### 3.3 Status update helper (optional)
No change – a task's status is still promoted by the worker.

---
## 4. Creating the Dependency Chain

This section details how the dependency chain is created. The primary change occurs within `headless.py`'s `_handle_travel_orchestrator_task`. The `sm_functions/travel_between_images.py` (specifically `run_travel_between_images_task`) will largely continue its current role: preparing the detailed parameters for the *single* `travel_orchestrator` task. It's this orchestrator task, once picked up by a `headless.py` worker, that will then generate and enqueue the entire sequence of segment and stitch tasks with the appropriate dependencies.

### 4.1  Inside `_handle_travel_orchestrator_task`
1.  **Loop** over every planned segment instead of only enqueuing the first:
   ```python
   previous_leaf_id = None
   for idx in range(num_segments):
       seg_task_id = sm_generate_unique_task_id(f"travel_seg_{run_id}_{idx:02d}_")
       seg_payload  = build_segment_payload(idx, ...)
       add_task_to_db(seg_payload, db_path, "travel_segment", dependant_on=previous_leaf_id)

       # Optionally spawn a wgp_sub task that depends on the segment itself
       # wgp_id = sm_generate_unique_task_id(f"wgp_sub_{seg_task_id[:10]}")
       # wgp_payload = build_wgp_payload(...)
       # add_task_to_db(wgp_payload, db_path, "wgp", dependant_on=seg_task_id)

       previous_leaf_id = seg_task_id      # Chain continues from the travel_segment task
   ```
2.  After the loop, enqueue the **stitch** task with `dependant_on = previous_leaf_id`. The `task_id` for this stitch task should be deterministic, for example, `f"travel_stitch_{run_id}"`. This allows the initiating script (`steerable_motion.py` via `run_travel_between_images_task`) to predict and provide this ID to the user upfront.

3.  Remove all runtime enqueueing from `_handle_travel_segment_task` and the WGP chaining helper.  Their only job is now **processing**.

### 4.2  Guide video logic
Because a segment will only start once its predecessor is **Complete**, the previous segment's `output_location` is guaranteed to be populated.  The existing helper `get_task_output_location_from_db()` remains valid; no additional polling is necessary.

### 4.3 Reporting Final Task ID to User
The `run_travel_between_images_task` function (in `sm_functions/travel_between_images.py`) should be updated:
1. After generating the `orchestrator_task_id` (which will also serve as the `run_id` for the sequence), it should also construct the `final_stitch_task_id` using the deterministic pattern defined in section 4.1 (e.g., `f"travel_stitch_{orchestrator_task_id}"`).
2. Upon successfully enqueuing the `travel_orchestrator` task, it should print a message to the console providing both:
    * The `orchestrator_task_id` (e.g., "Orchestrator task ID: XXXX").
    * The `final_stitch_task_id` (e.g., "Final stitch task ID to monitor: YYYY").
This allows users or scripts calling `steerable_motion.py` to immediately know which task ID represents the ultimate completion of the entire travel sequence.

---
## 5. Selecting Ready Tasks (Worker behaviour)
With the JOIN logic in §3.2 the worker automatically skips blocked tasks; **no extra Python logic is required**.  Multiple workers can safely claim tasks in parallel – only one will satisfy the WHERE clause for any given ready task.

---
## 6. Migration Strategy

1.  Pull latest code.
2.  Stop all running workers.
3.  Run `python headless.py --migrate-only` *(add a simple CLI flag that calls `_run_db_migrations()` then exits).*  For Supabase execute the new RPC.
4.  Restart workers; they will now respect `dependant_on`.
5.  Re-run your original CLI command – you will see **all** tasks appear instantly in the DB but only the root task will begin.

---
## 7. Testing Checklist

1. **Unit** – write a sqlite in-memory test that seeds three fake tasks A→B→C and asserts the selection order.
2. **Integration** – run the full `travel_between_images` flow and query:
   ```sql
   SELECT id, dependant_on, status FROM tasks ORDER BY created_at;
   ```
   Observe the chain.
3. **Recovery** – mark task *B* as `Failed` and ensure *C* never starts.

---
## 8. Backwards Compatibility
* Old tasks without `dependant_on` column remain valid – the JOIN treats missing column as NULL because the migration adds it with `NULL` default.
* Existing orchestrator logic will continue to work until you switch to the new chain-generation code.

---
## 9. Follow-up Work
* Consider extending `dependant_on` to a JSON array so a task can wait for **multiple** predecessors (e.g. a future *merge* task).
* Add a `priority` column so urgent tasks can jump the queue without breaking the dependency guarantee.
* Build a simple dashboard that visualises the DAG using the `dependant_on` relationships.

---
### Appendix A – Files to touch

1. `steerable_motion.py`  – DB init/schema.
2. `headless.py`
   * `_migrate_sqlite_schema`
   * `init_db` (SQLite) and `init_db_supabase` (via RPC `func_initialize_tasks_table`)
   * `get_oldest_queued_task` / Supabase RPC
   * `_handle_travel_orchestrator_task`