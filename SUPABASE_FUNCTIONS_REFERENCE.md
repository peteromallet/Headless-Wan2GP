# Supabase Functions Reference

This document describes all database RPC functions and edge functions used by the Wan2GP headless worker system.

## Edge Functions

Edge functions are called via HTTP requests and handle authentication, file uploads, and cross-user operations.

### `create-task`
**URL:** `/functions/v1/create-task`  
**Method:** POST  
**Purpose:** Creates new tasks in the database  
**Used by:** `add_task_to_db()` in `db_operations.py`  
**Status:** ✅ **WORKING** (200 OK)

**Request:**
```json
{
  "task_id": "uuid-string",
  "params": {...},
  "task_type": "travel_segment",
  "project_id": "uuid-string",
  "dependant_on": "uuid-string" // optional
}
```

**Response:** `"Task queued"` (200) or error

---

### `claim-next-task`
**URL:** `/functions/v1/claim-next-task`  
**Method:** POST  
**Purpose:** Atomically claims the next available task  
**Used by:** `get_oldest_queued_task_supabase()` in `db_operations.py`  
**Status:** ❌ **FAILING** (500 - Foreign key constraint)

**Request:** `{}` (empty body)

**Response:**
```json
{
  "task_id": "uuid-string",
  "params": {...},
  "task_type": "travel_segment", 
  "project_id": "uuid-string"
}
```

**Known Issues:**
- **Error:** `tasks_worker_id_fkey` constraint violation
- **Root Cause:** Edge function generates dynamic worker IDs (`edge_${crypto.randomUUID()}`) that don't exist in workers table
- **Fix Required:** Apply `fix_worker_issue.sql` migration to auto-create workers

**Notes:**
- Generates dynamic worker ID: `edge_${crypto.randomUUID()}`
- Service role: sees all tasks
- User token: sees only their project's tasks

---

### `complete-task`
**URL:** `/functions/v1/complete-task`  
**Method:** POST  
**Purpose:** Marks task complete and uploads output files  
**Used by:** `update_task_status_supabase()` when status = "Complete"  
**Status:** ✅ **WORKING** (200 OK with file upload)

**Request:**
```json
{
  "task_id": "uuid-string",
  "file_data": "base64-encoded-file",
  "filename": "output.mp4"
}
```

**Response:**
```json
{
  "success": true,
  "public_url": "https://...supabase.co/storage/.../file.mp4",
  "message": "Task completed and file uploaded successfully"
}
```

---

### `get-predecessor-output`
**URL:** `/functions/v1/get-predecessor-output`  
**Method:** POST  
**Purpose:** Gets output location of a task's dependency  
**Used by:** `get_predecessor_output_supabase()` in `db_operations.py`  
**Status:** ✅ **WORKING** (200 OK)

**Request:**
```json
{
  "task_id": "uuid-string"
}
```

**Response:**
```json
{
  "predecessor_id": "uuid-string",
  "output_location": "https://...supabase.co/storage/.../file.mp4"
}
```

---

### `get-completed-segments`
**URL:** `/functions/v1/get-completed-segments`  
**Method:** POST  
**Purpose:** Gets all completed segments for stitching  
**Used by:** `get_completed_segments_supabase()` in `db_operations.py`  
**Status:** ✅ **WORKING** (200 OK)

**Request:**
```json
{
  "orchestrator_id": "uuid-string"
}
```

**Response:**
```json
[
  {
    "segment_index": 0,
    "output_location": "https://...supabase.co/storage/.../segment_0.mp4"
  },
  {
    "segment_index": 1, 
    "output_location": "https://...supabase.co/storage/.../segment_1.mp4"
  }
]
```

## Database RPC Functions

RPC functions are called directly via the Supabase client and require proper authentication.

### `func_claim_available_task`
**Signature:** `(worker_id_param TEXT)`  
**Returns:** Table with columns: `id`, `status`, `attempts`, `worker_id`, `generation_started_at`, `task_data`, `created_at`, `task_type`  
**Purpose:** Claims oldest available task with dependency checking  
**Used by:** `get_oldest_queued_task_supabase()` as RPC fallback  
**Status:** ❌ **FAILING** (Foreign key constraint)

**Usage:**
```python
response = client.rpc("func_claim_available_task", {
    "worker_id_param": "gpu-20250723_145828-38ab706b"
}).execute()
```

**Known Issues:**
- **Error:** `tasks_worker_id_fkey` constraint violation  
- **Details:** Worker ID `gpu-test-b1a5c74a` not present in workers table
- **Fix Required:** Apply `fix_worker_issue.sql` migration

**Notes:**
- Updates task status to "In Progress"
- Sets worker_id and generation_started_at
- Checks dependencies are complete
- Uses `FOR UPDATE SKIP LOCKED` for concurrency

---

### `func_update_task_status`
**Signature:** `(p_task_id TEXT, p_status TEXT, p_table_name TEXT DEFAULT 'tasks', p_output_location TEXT DEFAULT NULL)`  
**Returns:** `BOOLEAN`  
**Purpose:** Updates task status and output location  
**Used by:** `update_task_status_supabase()` in `db_operations.py`  
**Status:** ✅ **WORKING**

**Usage:**
```python
response = client.rpc("func_update_task_status", {
    "p_task_id": "uuid-string",
    "p_status": "Complete",
    "p_output_location": "/path/to/output.mp4"
}).execute()
```

**Notes:**
- Handles enum casting: `p_status::task_status`
- Sets `updated_at` timestamp
- Sets `generation_processed_at` when status = "Complete"

---

### `complete_task_with_timing`
**Signature:** `(p_task_id TEXT, p_output_location TEXT)`  
**Returns:** `BOOLEAN`  
**Purpose:** Completes task with timing information  
**Used by:** Edge functions for task completion  
**Status:** ✅ **WORKING**

**Notes:**
- Sets status to "Complete"
- Updates `generation_processed_at` and `updated_at`
- Handles TEXT to UUID conversion for task_id

---

### `func_mark_task_failed`
**Signature:** `(p_task_id TEXT, p_error_message TEXT)`  
**Returns:** `BOOLEAN`  
**Purpose:** Marks task as failed with error message  
**Used by:** Error handling in task processing  
**Status:** ❓ **UNTESTED**

---

### `func_initialize_tasks_table`
**Signature:** `(p_table_name TEXT DEFAULT 'tasks')`  
**Returns:** `BOOLEAN`  
**Purpose:** Initializes or validates task table schema  
**Used by:** Database setup and testing  
**Status:** ✅ **WORKING**

---

### `func_migrate_tasks_for_task_type`
**Signature:** `(p_table_name TEXT DEFAULT 'tasks')`  
**Returns:** `BOOLEAN`  
**Purpose:** Handles task table migrations  
**Used by:** Database migrations and testing  
**Status:** ✅ **WORKING**

## Storage Operations

### Supabase Storage Upload
**Purpose:** Direct file uploads to Supabase storage  
**Used by:** `upload_to_supabase_storage()` in `db_operations.py`  
**Status:** ❌ **FAILING** (API compatibility issue)

**Known Issues:**
- **Error:** `'UploadResponse' object has no attribute 'status_code'`
- **Root Cause:** Code expects old API response format
- **Fix Required:** Update response handling for newer Supabase client

## Worker ID Patterns

The system supports various worker ID patterns:

| Pattern | Example | Usage | Status |
|---------|---------|--------|--------|
| `gpu-*` | `gpu-20250723_145828-38ab706b` | Production GPU workers | ❌ Needs migration |
| `edge_*` | `edge_a1b2c3d4-...` | Edge function generated IDs | ❌ Needs migration |
| `worker_*` | `worker_12345` | Process-based workers | ❌ Needs migration |
| `test_*` | `test_worker_abc123` | Test workers | ❌ Needs migration |

## Test Results Summary

**Latest Test Run:** 18/22 tests passed (81.8% success rate)

### ✅ **Working Functions (18)**
- All 4 working edge functions (create, complete, predecessor, segments)
- 4/6 RPC functions (update, timing, initialize, migrate)
- All image generation tasks (5/5)
- All video generation tasks (3/3)
- Core database operations (6/7)

### ❌ **Failing Functions (4)**
1. **`claim-next-task` edge function** - Worker foreign key constraint
2. **`func_claim_available_task` RPC** - Worker foreign key constraint  
3. **Supabase storage upload** - API compatibility issue
4. **Task claiming in general** - Affects headless worker functionality

## Required Fixes

### 1. Worker Foreign Key Constraint
**Migration Required:** `fix_worker_issue.sql`
- Creates trigger to auto-insert missing workers
- Backfills existing worker IDs from tasks table
- Fixes both edge function and RPC claiming

### 2. Storage Upload API
**Code Update Required:** Update `upload_to_supabase_storage()` response handling
- Remove `.status_code` attribute access
- Update to newer Supabase client API format

## Authentication Flow

1. **Service Role Key**: Bypasses RLS, sees all tasks
2. **User JWT**: Sees only tasks in user's projects (via RLS)
3. **PAT (Personal Access Token)**: Uses edge functions only, no direct RPC

## Key Files

- **`source/db_operations.py`**: Main database interface
- **`supabase/functions/*/index.ts`**: Edge function implementations
- **`headless.py`**: Main worker that uses these functions
- **Migration files**: SQL scripts to create/update functions

## Dependencies

- All functions respect the `dependant_on` field
- Tasks only run when dependencies have status = "Complete"
- Edge functions perform manual dependency checking
- RPC functions use SQL-based dependency validation 