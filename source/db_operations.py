# source/sm_functions/db_operations.py
import os
import sys
import json
import time
import traceback
import datetime
import sqlite3
import urllib.parse
import threading
from pathlib import Path

try:
    from supabase import create_client, Client as SupabaseClient
except ImportError:
    SupabaseClient = None

# -----------------------------------------------------------------------------
# Global DB Configuration (will be set by headless.py)
# -----------------------------------------------------------------------------
DB_TYPE = "sqlite"
PG_TABLE_NAME = "tasks"
SQLITE_DB_PATH = "tasks.db"
SUPABASE_URL = None
SUPABASE_SERVICE_KEY = None
SUPABASE_VIDEO_BUCKET = "videos"
SUPABASE_CLIENT: SupabaseClient | None = None

sqlite_lock = threading.Lock()

SQLITE_MAX_RETRIES = 5
SQLITE_RETRY_DELAY = 0.5  # seconds

# -----------------------------------------------------------------------------
# Status Constants
# -----------------------------------------------------------------------------
STATUS_QUEUED = "Queued"
STATUS_IN_PROGRESS = "In Progress"
STATUS_COMPLETE = "Complete"
STATUS_FAILED = "Failed"

# -----------------------------------------------------------------------------
# Debug / Verbose Logging Helpers
# -----------------------------------------------------------------------------
debug_mode = False

def dprint(msg: str):
    """Print a debug message if debug_mode is enabled."""
    if debug_mode:
        print(f"[DEBUG {time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

# -----------------------------------------------------------------------------
# Internal Helpers
# -----------------------------------------------------------------------------

def execute_sqlite_with_retry(db_path_str: str, operation_func, *args, **kwargs):
    """Execute SQLite operations with retry logic for handling locks and I/O errors"""
    for attempt in range(SQLITE_MAX_RETRIES):
        try:
            with sqlite_lock:  # Ensure only one thread accesses SQLite at a time
                conn = sqlite3.connect(db_path_str, timeout=30.0)  # 30 second timeout
                conn.execute("PRAGMA journal_mode=WAL")  # Enable WAL mode for better concurrency
                conn.execute("PRAGMA synchronous=NORMAL")  # Balance between safety and performance
                conn.execute("PRAGMA busy_timeout=30000")  # 30 second busy timeout
                try:
                    result = operation_func(conn, *args, **kwargs)
                    conn.commit()
                    return result
                finally:
                    conn.close()
        except (sqlite3.OperationalError, sqlite3.DatabaseError) as e:
            error_msg = str(e).lower()
            if "database is locked" in error_msg or "disk i/o error" in error_msg or "database disk image is malformed" in error_msg:
                if attempt < SQLITE_MAX_RETRIES - 1:
                    wait_time = SQLITE_RETRY_DELAY * (2 ** attempt)  # Exponential backoff
                    print(f"SQLite error on attempt {attempt + 1}: {e}. Retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"SQLite error after {SQLITE_MAX_RETRIES} attempts: {e}")
                    raise
            else:
                # For other SQLite errors, don't retry
                raise
        except Exception as e:
            # For non-SQLite errors, don't retry
            raise
    
    raise sqlite3.OperationalError(f"Failed to execute SQLite operation after {SQLITE_MAX_RETRIES} attempts")

# -----------------------------------------------------------------------------
# Public Database Functions
# -----------------------------------------------------------------------------

def _migrate_sqlite_schema(db_path_str: str):
    """Applies necessary schema migrations to an existing SQLite database."""
    dprint(f"SQLite Migration: Checking schema for {db_path_str}...")
    try:
        def migration_operations(conn):
            cursor = conn.cursor()

            # --- Check if 'tasks' table exists before attempting migrations ---
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='tasks'")
            if cursor.fetchone() is None:
                dprint("SQLite Migration: 'tasks' table does not exist. Skipping schema migration steps. init_db will create it.")
                return True # Indicate success as there's nothing to migrate on a non-existent table

            # Check if task_type column exists
            cursor.execute(f"PRAGMA table_info(tasks)")
            columns = [row[1] for row in cursor.fetchall()]
            task_type_column_exists = 'task_type' in columns

            if not task_type_column_exists:
                dprint("SQLite Migration: 'task_type' column not found. Adding it.")
                cursor.execute("ALTER TABLE tasks ADD COLUMN task_type TEXT") # Add as nullable first
                conn.commit() # Commit alter table before data migration
                dprint("SQLite Migration: 'task_type' column added.")
            else:
                dprint("SQLite Migration: 'task_type' column already exists.")

            # --- Add/Rename dependant_on column if not exists --- (Section 2.2)
            dependant_on_column_exists = 'dependant_on' in columns
            depends_on_column_exists_old_name = 'depends_on' in columns # Check for the previously incorrect name

            if not dependant_on_column_exists:
                if depends_on_column_exists_old_name:
                    dprint("SQLite Migration: Found old 'depends_on' column. Renaming to 'dependant_on'.")
                    try:
                        # Ensure no other column is already named 'dependant_on' before renaming
                        # This scenario is unlikely if migrations are run sequentially but good for robustness
                        if 'dependant_on' not in columns:
                            cursor.execute("ALTER TABLE tasks RENAME COLUMN depends_on TO dependant_on")
                            dprint("SQLite Migration: Renamed 'depends_on' to 'dependant_on'.")
                            dependant_on_column_exists = True # Mark as existing now
                        else:
                             dprint("SQLite Migration: 'dependant_on' column already exists. Skipping rename of 'depends_on'.")
                    except sqlite3.OperationalError as e_rename:
                        dprint(f"SQLite Migration: Could not rename 'depends_on' to 'dependant_on' (perhaps 'dependant_on' already exists or other issue): {e_rename}. Will attempt to ADD 'dependant_on' if it truly doesn't exist after this.")
                        # Re-check columns after attempted rename or if it failed
                        cursor.execute(f"PRAGMA table_info(tasks)")
                        rechecked_columns = [row[1] for row in cursor.fetchall()]
                        if 'dependant_on' not in rechecked_columns:
                            dprint("SQLite Migration: 'dependant_on' still not found after rename attempt, adding new column.")
                            cursor.execute("ALTER TABLE tasks ADD COLUMN dependant_on TEXT NULL")
                            dprint("SQLite Migration: 'dependant_on' column added.")
                        else:
                            dprint("SQLite Migration: 'dependant_on' column now exists (possibly due to a concurrent migration or complex rename scenario).")
                            dependant_on_column_exists = True
                else:
                    dprint("SQLite Migration: 'dependant_on' column not found and no 'depends_on' to rename. Adding new 'dependant_on' column.")
                    cursor.execute("ALTER TABLE tasks ADD COLUMN dependant_on TEXT NULL")
                    dprint("SQLite Migration: 'dependant_on' column added.")
            else:
                dprint("SQLite Migration: 'dependant_on' column already exists.")
            # --- End add/rename dependant_on column ---

            # Ensure the index for dependant_on exists
            cursor.execute("PRAGMA index_list(tasks)")
            indexes = [row[1] for row in cursor.fetchall()]
            if 'idx_dependant_on' not in indexes:
                dprint("SQLite Migration: Creating 'idx_dependant_on' index.")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_dependant_on ON tasks(dependant_on)")
            else:
                dprint("SQLite Migration: 'idx_dependant_on' index already exists.")

            # Populate task_type from params if it's NULL (for old rows or newly added column)
            dprint("SQLite Migration: Attempting to populate NULL 'task_type' from 'params' JSON...")
            cursor.execute("SELECT id, params FROM tasks WHERE task_type IS NULL")
            rows_to_migrate = cursor.fetchall()
            
            migrated_count = 0
            for task_id, params_json_str in rows_to_migrate:
                try:
                    params_dict = json.loads(params_json_str)
                    # Attempt to get task_type from common old locations within params
                    # The user might need to adjust these keys if their old storage was different
                    old_task_type = params_dict.get("task_type") # Most likely if it was in params
                    
                    if old_task_type:
                        dprint(f"SQLite Migration: Found task_type '{old_task_type}' in params for task_id {task_id}. Updating row.")
                        cursor.execute("UPDATE tasks SET task_type = ? WHERE id = ?", (old_task_type, task_id))
                        migrated_count += 1
                    else:
                        # If task_type is not in params, it might be inferred from 'model' or other fields
                        # For instance, if 'model' field implied the task type for older tasks.
                        # This part is highly dependent on previous conventions.
                        # As a simple default, if not found, it will remain NULL unless a default is set.
                        # For 'travel_between_images' and 'different_perspective', these are typically set by steerable_motion.py
                        # and wouldn't exist as 'task_type' inside params for headless.py's default processing.
                        # Headless tasks like 'generate_openpose' *did* use task_type in params.
                        dprint(f"SQLite Migration: No 'task_type' key in params for task_id {task_id}. It will remain NULL or needs manual/specific migration logic if it was inferred differently.")
                except json.JSONDecodeError:
                    dprint(f"SQLite Migration: Could not parse params JSON for task_id {task_id}. Skipping 'task_type' population for this row.")
                except Exception as e_row:
                    dprint(f"SQLite Migration: Error processing row for task_id {task_id}: {e_row}")
            
            if migrated_count > 0:
                conn.commit()
            dprint(f"SQLite Migration: Populated 'task_type' for {migrated_count} rows from params.")

            # Default remaining NULL task_types for old standard tasks
            # This ensures rows that didn't have an explicit 'task_type' in their params (e.g. old default WGP tasks)
            # get a value, respecting the NOT NULL constraint if the table is new or fully validated.
            default_task_type_for_old_rows = "standard_wgp_task" 
            cursor.execute(
                f"UPDATE tasks SET task_type = ? WHERE task_type IS NULL", 
                (default_task_type_for_old_rows,)
            )
            updated_to_default_count = cursor.rowcount
            if updated_to_default_count > 0:
                conn.commit()
            dprint(f"SQLite Migration: Updated {updated_to_default_count} older rows with NULL task_type to default '{default_task_type_for_old_rows}'.")

            dprint("SQLite Migration: Schema check and population attempt complete.")
            return True

        execute_sqlite_with_retry(db_path_str, migration_operations)
        
    except Exception as e:
        print(f"[ERROR] SQLite Migration: Failed to migrate schema for {db_path_str}: {e}")
        traceback.print_exc()
        # Depending on severity, you might want to sys.exit(1)

def _init_db_sqlite(db_path_str: str):
    """Initialize the SQLite database with proper error handling"""
    def _init_operation(conn):
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                task_type TEXT NOT NULL,
                params TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'Queued',
                dependant_on TEXT NULL,
                output_location TEXT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NULL,
                project_id TEXT NOT NULL,
                FOREIGN KEY (project_id) REFERENCES projects(id)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_status_created ON tasks(status, created_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_dependant_on ON tasks(dependant_on)")
        return True
    
    try:
        execute_sqlite_with_retry(db_path_str, _init_operation)
        print(f"SQLite database initialized: {db_path_str}")
    except Exception as e:
        print(f"Failed to initialize SQLite database: {e}")
        sys.exit(1)

def get_oldest_queued_task_sqlite(db_path_str: str):
    """Get the oldest queued task with proper error handling"""
    def _get_operation(conn):
        cursor = conn.cursor()
        # Modified to select tasks with status 'Queued' only
        # Also fetch project_id
        sql_query = f"""
            SELECT t.id, t.params, t.task_type, t.project_id
            FROM   tasks AS t
            LEFT JOIN tasks AS d             ON d.id = t.dependant_on
            WHERE  t.status = ? 
              AND (t.dependant_on IS NULL OR d.status = ?)
            ORDER BY t.created_at ASC
            LIMIT  1
        """
        query_params = (STATUS_QUEUED, STATUS_COMPLETE)
        
        cursor.execute(sql_query, query_params)
        task_row = cursor.fetchone()
        if task_row:
            dprint(f"SQLite: Fetched raw task_row: {task_row}")
            return {"task_id": task_row[0], "params": json.loads(task_row[1]), "task_type": task_row[2], "project_id": task_row[3]}
        return None
    
    try:
        return execute_sqlite_with_retry(db_path_str, _get_operation)
    except Exception as e:
        print(f"Error getting oldest queued task: {e}")
        return None

def update_task_status_sqlite(db_path_str: str, task_id: str, status: str, output_location_val: str | None = None):
    """Updates a task's status and updated_at timestamp with proper error handling"""
    def _update_operation(conn, task_id, status, output_location_val):
        cursor = conn.cursor()
        
        if status == STATUS_COMPLETE and output_location_val is not None:
            # Step 1: Update output_location and updated_at for the location change
            current_utc_iso_ts_loc_update = datetime.datetime.utcnow().isoformat() + "Z"
            dprint(f"SQLite Update (Split Step 1): Updating output_location for {task_id} to {output_location_val}")
            cursor.execute("UPDATE tasks SET output_location = ?, updated_at = ? WHERE id = ?",
                           (output_location_val, current_utc_iso_ts_loc_update, task_id))
            conn.commit()  # Explicitly commit the output_location update

            # Step 2: Update status and updated_at for the status change
            current_utc_iso_ts_status_update = datetime.datetime.utcnow().isoformat() + "Z"
            dprint(f"SQLite Update (Split Step 2): Updating status for {task_id} to {status}")
            cursor.execute("UPDATE tasks SET status = ?, updated_at = ? WHERE id = ?",
                           (status, current_utc_iso_ts_status_update, task_id))
            # The final commit for this status update will be handled by execute_sqlite_with_retry

        elif status == STATUS_FAILED and output_location_val is not None: # output_location_val is error message here
            current_utc_iso_ts_fail_update = datetime.datetime.utcnow().isoformat() + "Z"
            dprint(f"SQLite Update (Single): Updating status to FAILED and output_location (error msg) for {task_id}")
            cursor.execute("UPDATE tasks SET status = ?, updated_at = ?, output_location = ? WHERE id = ?",
                           (status, current_utc_iso_ts_fail_update, output_location_val, task_id))

        else: # For "In Progress" or other statuses, or if output_location_val is None
            current_utc_iso_ts_progress_update = datetime.datetime.utcnow().isoformat() + "Z"
            dprint(f"SQLite Update (Single): Updating status for {task_id} to {status} (output_location_val: {output_location_val})")
            # If output_location_val is None even for COMPLETE or FAILED, it won't be set here.
            # This branch primarily handles IN_PROGRESS or status changes where output_location is not part of the update.
            # If status is COMPLETE/FAILED and output_location_val is None, only status and updated_at change.
            if output_location_val is not None and status in [STATUS_COMPLETE, STATUS_FAILED]:
                 # This case should ideally be caught by the specific branches above,
                 # but as a safeguard if logic changes:
                 dprint(f"SQLite Update (Single with output_location): Updating status, output_location for {task_id}")
                 cursor.execute("UPDATE tasks SET status = ?, updated_at = ?, output_location = ? WHERE id = ?",
                               (status, current_utc_iso_ts_progress_update, output_location_val, task_id))
            else:
                 cursor.execute("UPDATE tasks SET status = ?, updated_at = ? WHERE id = ?",
                               (status, current_utc_iso_ts_progress_update, task_id))
        return True
    
    try:
        execute_sqlite_with_retry(db_path_str, _update_operation, task_id, status, output_location_val)
        dprint(f"SQLite: Updated status of task {task_id} to {status}. Output: {output_location_val if output_location_val else 'N/A'}")
    except Exception as e:
        print(f"Error updating task status for {task_id}: {e}")
        # Don't raise here to avoid crashing the main loop

def init_db():
    """Initializes the database, dispatching to the correct implementation."""
    if DB_TYPE == "supabase":
        return init_db_supabase()
    else:
        return _init_db_sqlite(SQLITE_DB_PATH)

def get_oldest_queued_task():
    """Gets the oldest queued task, dispatching to the correct implementation."""
    if DB_TYPE == "supabase":
        return get_oldest_queued_task_supabase()
    else:
        return get_oldest_queued_task_sqlite(SQLITE_DB_PATH)

def update_task_status(task_id: str, status: str, output_location: str | None = None):
    """Updates a task's status, dispatching to the correct implementation."""
    if DB_TYPE == "supabase":
        return update_task_status_supabase(task_id, status, output_location)
    else:
        return update_task_status_sqlite(SQLITE_DB_PATH, task_id, status, output_location)

def init_db_supabase(): # Renamed from init_db_postgres
    """Initializes the PostgreSQL tasks table via Supabase RPC if it doesn't exist."""
    if not SUPABASE_CLIENT:
        print("[ERROR] Supabase client not initialized. Cannot initialize database table.")
        sys.exit(1)
    try:
        # RPC call to the SQL function func_initialize_tasks_table
        # IMPORTANT: The func_initialize_tasks_table SQL function itself
        # must be updated to include "task_type TEXT NOT NULL" and "dependant_on TEXT NULL"
        # along with an index "idx_dependant_on" on the "dependant_on" column in its
        # CREATE TABLE statement for the specified p_table_name.
        SUPABASE_CLIENT.rpc("func_initialize_tasks_table", {"p_table_name": PG_TABLE_NAME}).execute()
        print(f"Supabase RPC: Table '{PG_TABLE_NAME}' initialization requested.")
        # Note: RPC for DDL might not return specific confirmation beyond successful execution.
        # You might need to add a SELECT to confirm table existence if strict feedback is needed.
    except Exception as e: # Broader exception for Supabase/PostgREST errors
        print(f"[ERROR] Supabase RPC for table initialization failed: {e}")
        traceback.print_exc()
        sys.exit(1)

def get_oldest_queued_task_supabase(): # Renamed from get_oldest_queued_task_postgres
    """Fetches the oldest task via Supabase RPC using func_claim_task."""
    if not SUPABASE_CLIENT:
        print("[ERROR] Supabase client not initialized. Cannot get task.")
        return None
    try:
        worker_id = f"worker_{os.getpid()}" # Example worker ID
        dprint(f"DEBUG get_oldest_queued_task_supabase: About to call RPC func_claim_task.")
        dprint(f"DEBUG get_oldest_queued_task_supabase: PG_TABLE_NAME = '{PG_TABLE_NAME}' (type: {type(PG_TABLE_NAME)})")
        dprint(f"DEBUG get_oldest_queued_task_supabase: worker_id = '{worker_id}' (type: {type(worker_id)})")
        
        response = SUPABASE_CLIENT.rpc(
            "func_claim_task", 
            {"p_table_name": PG_TABLE_NAME, "p_worker_id": worker_id}
        ).execute()
        
        dprint(f"Supabase RPC func_claim_task response data: {response.data}")

        if response.data and len(response.data) > 0:
            task_data = response.data[0] # RPC should return a single row or empty
            dprint(f"Supabase RPC: Raw task_data from func_claim_task: {task_data}") # DEBUG ADDED
            # Ensure the RPC returns task_id_out, params_out, and now task_type_out
            if task_data.get("task_id_out") and task_data.get("params_out") is not None and task_data.get("task_type_out") is not None:
                dprint(f"Supabase RPC: Claimed task {task_data['task_id_out']} of type {task_data['task_type_out']}")
                return {"task_id": task_data["task_id_out"], "params": task_data["params_out"], "task_type": task_data["task_type_out"]}
            else:
                dprint("Supabase RPC: func_claim_task returned but no task was claimed or required fields (task_id_out, params_out, task_type_out) are missing.")
                return None
        else:
            dprint("Supabase RPC: No task claimed or empty response from func_claim_task.")
            return None
    except Exception as e:
        print(f"[ERROR] Supabase RPC func_claim_task failed: {e}")
        traceback.print_exc()
        return None

def update_task_status_supabase(task_id_str, status_str, output_location_val=None): # Renamed from update_task_status_postgres
    """Updates a task's status via Supabase RPC using func_update_task_status."""
    if not SUPABASE_CLIENT:
        print("[ERROR] Supabase client not initialized. Cannot update task status.")
        return
    try:
        params = {
            "p_table_name": PG_TABLE_NAME,
            "p_task_id": task_id_str,
            "p_status": status_str
        }
        if output_location_val is not None:
            params["p_output_location"] = output_location_val
        
        SUPABASE_CLIENT.rpc("func_update_task_status", params).execute()
        dprint(f"Supabase RPC: Updated status of task {task_id_str} to {status_str}. Output: {output_location_val if output_location_val else 'N/A'}")
    except Exception as e:
        print(f"[ERROR] Supabase RPC func_update_task_status for {task_id_str} failed: {e}")

def _migrate_supabase_schema():
    """Applies necessary schema migrations to an existing Supabase/PostgreSQL database via RPC."""
    if not SUPABASE_CLIENT:
        print("[ERROR] Supabase Migration: Supabase client not initialized. Cannot run migration.")
        return

    dprint(f"Supabase Migration: Requesting schema migration via RPC 'func_migrate_tasks_for_task_type' for table {PG_TABLE_NAME}...")
    try:

        # IMPORTANT: The func_migrate_tasks_for_task_type SQL function itself (or a new one like func_migrate_tasks_add_dependant_on)
        # must be extended to perform:
        # ALTER TABLE {p_table_name} ADD COLUMN IF NOT EXISTS dependant_on TEXT NULL;
        # CREATE INDEX IF NOT EXISTS idx_dependant_on ON {p_table_name}(dependant_on);
        # It should also handle renaming 'depends_on' to 'dependant_on' if the old (previously incorrect) misspelled column 'depends_on' exists.
        response = SUPABASE_CLIENT.rpc("func_migrate_tasks_for_task_type", {"p_table_name": PG_TABLE_NAME}).execute()
        
        # Improved response handling based on Supabase Python client v2+ structure
        if response.error:
            print(f"[ERROR] Supabase Migration: RPC 'func_migrate_tasks_for_task_type' returned an error: {response.error.message} (Code: {response.error.code}, Details: {response.error.details})")
        elif response.data:
            dprint(f"Supabase Migration: RPC 'func_migrate_tasks_for_task_type' executed. Response data: {response.data}")
        else:
            dprint("Supabase Migration: RPC 'func_migrate_tasks_for_task_type' executed. (No specific data or error in response, check RPC logs if issues)")
            
    except Exception as e:
        print(f"[ERROR] Supabase Migration: Failed to execute RPC 'func_migrate_tasks_for_task_type': {e}")
        traceback.print_exc()

def _run_db_migrations():
    """Runs database migrations based on the configured DB_TYPE."""
    dprint(f"DB Migrations: Running for DB_TYPE: {DB_TYPE}")
    if DB_TYPE == "sqlite":
        if SQLITE_DB_PATH:
            _migrate_sqlite_schema(SQLITE_DB_PATH)
        else:
            print("[ERROR] DB Migration: SQLITE_DB_PATH not set. Skipping SQLite migration.")
    elif DB_TYPE == "supabase":
        if SUPABASE_CLIENT and PG_TABLE_NAME:
            _migrate_supabase_schema()
        else:
            print("[ERROR] DB Migration: Supabase client or PG_TABLE_NAME not configured. Skipping Supabase migration.")
    else:
        dprint(f"DB Migrations: No migration logic for DB_TYPE '{DB_TYPE}'. Skipping migrations.")

def add_task_to_db(task_payload: dict, task_type_str: str, dependant_on: str | None = None, db_path: str | None = None):
    """
    Adds a new task to the database, dispatching to SQLite or Supabase.
    The `db_path` argument is for legacy SQLite compatibility and is ignored for Supabase.
    """
    task_id = task_payload.get("task_id")
    if not task_id:
        raise ValueError("task_id must be present in the task_payload.")

    # Shared logic: Sanitize payload and get project_id
    params_for_db = task_payload.copy()
    params_for_db.pop("task_type", None) # Ensure task_type is not duplicated in params
    params_json_str = json.dumps(params_for_db)
    project_id = task_payload.get("project_id", "default_project_id")

    if DB_TYPE == "supabase":
        if not SUPABASE_CLIENT:
            print("[ERROR] Supabase client not initialized. Cannot add task.")
            return
        try:
            rpc_params = {
                "p_task_id": task_id,
                "p_params": params_json_str,
                "p_task_type": task_type_str,
                "p_project_id": project_id,
                "p_dependant_on": dependant_on,
                "p_table_name": PG_TABLE_NAME,
            }
            dprint(f"Supabase RPC: Adding task {task_id} via func_add_task with params: {rpc_params}")
            SUPABASE_CLIENT.rpc("func_add_task", rpc_params).execute()
            print(f"Task {task_id} (Type: {task_type_str}) added to Supabase.")
        except Exception as e:
            print(f"[ERROR] Supabase RPC func_add_task for {task_id} failed: {e}")
            raise

    else: # Default to SQLite
        db_to_use = db_path if db_path else SQLITE_DB_PATH
        if not db_to_use:
            raise ValueError("SQLite DB path is not configured.")

        def _add_op(conn):
            cursor = conn.cursor()
            current_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute(
                f"INSERT INTO tasks (id, params, task_type, status, created_at, project_id, dependant_on) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (task_id, params_json_str, task_type_str, STATUS_QUEUED, current_timestamp, project_id, dependant_on)
            )
        try:
            execute_sqlite_with_retry(db_to_use, _add_op)
            print(f"Task {task_id} (Type: {task_type_str}) added to SQLite database {db_to_use}.")
        except Exception as e:
            print(f"SQLite error when adding task {task_id} (Type: {task_type_str}): {e}")
            raise

def poll_task_status(task_id: str, poll_interval_seconds: int = 10, timeout_seconds: int = 1800, db_path: str | None = None) -> str | None:
    """
    Polls the DB for task completion and returns the output_location.
    Dispatches to SQLite or Supabase. `db_path` is for legacy SQLite calls.
    """
    print(f"Polling for completion of task {task_id} (timeout: {timeout_seconds}s)...")
    start_time = time.time()
    last_status_print_time = 0

    while True:
        current_time = time.time()
        if current_time - start_time > timeout_seconds:
            print(f"Error: Timeout polling for task {task_id} after {timeout_seconds} seconds.")
            return None

        status = None
        output_location = None

        if DB_TYPE == "supabase":
            if not SUPABASE_CLIENT:
                print("[ERROR] Supabase client not initialized. Cannot poll status.")
                time.sleep(poll_interval_seconds)
                continue
            try:
                # Assuming a simple select here. An RPC could also be used.
                resp = SUPABASE_CLIENT.table(PG_TABLE_NAME).select("status, output_location").eq("id", task_id).single().execute()
                if resp.data:
                    status = resp.data.get("status")
                    output_location = resp.data.get("output_location")
            except Exception as e:
                print(f"Supabase error while polling task {task_id}: {e}. Retrying...")
        else: # SQLite
            db_to_use = db_path if db_path else SQLITE_DB_PATH
            if not db_to_use:
                raise ValueError("SQLite DB path is not configured for polling.")
            
            def _poll_op(conn):
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(f"SELECT status, output_location FROM tasks WHERE id = ?", (task_id,))
                return cursor.fetchone()
            try:
                row = execute_sqlite_with_retry(db_to_use, _poll_op)
                if row:
                    status = row["status"]
                    output_location = row["output_location"]
            except Exception as e:
                print(f"SQLite error while polling task {task_id}: {e}. Retrying...")

        if status:
            if current_time - last_status_print_time > poll_interval_seconds * 2 :
                 print(f"Task {task_id}: Status = {status} (Output: {output_location if output_location else 'N/A'})")
                 last_status_print_time = current_time

            if status == STATUS_COMPLETE:
                if output_location:
                    print(f"Task {task_id} completed successfully. Output: {output_location}")
                    return output_location
                else:
                    print(f"Error: Task {task_id} is COMPLETE but output_location is missing. Assuming failure.")
                    return None
            elif status == STATUS_FAILED:
                print(f"Error: Task {task_id} failed. Error details: {output_location}")
                return None
            elif status not in [STATUS_QUEUED, STATUS_IN_PROGRESS]:
                print(f"Warning: Task {task_id} has unknown status '{status}'. Treating as error.")
                return None
        else:
             if current_time - last_status_print_time > poll_interval_seconds * 2 :
                print(f"Task {task_id}: Not found in DB yet or status pending...")
                last_status_print_time = current_time

        time.sleep(poll_interval_seconds)

# Helper to query DB for a specific task's output (needed by segment handler)
def get_task_output_location_from_db(task_id_to_find: str) -> str | None:
    dprint(f"Querying DB for output location of task: {task_id_to_find}")
    if DB_TYPE == "sqlite":
        def _get_op(conn):
            cursor = conn.cursor()
            # Ensure we only get tasks that are actually complete with an output
            cursor.execute("SELECT output_location FROM tasks WHERE id = ? AND status = ? AND output_location IS NOT NULL", 
                           (task_id_to_find, STATUS_COMPLETE))
            row = cursor.fetchone()
            return row[0] if row else None
        try:
            return execute_sqlite_with_retry(SQLITE_DB_PATH, _get_op)
        except Exception as e:
            print(f"Error querying SQLite for task output {task_id_to_find}: {e}")
            return None
    elif DB_TYPE == "supabase" and SUPABASE_CLIENT:
        try:
            response = SUPABASE_CLIENT.table(PG_TABLE_NAME)\
                .select("output_location, status")\
                .eq("id", task_id_to_find)\
                .single()\
                .execute()

            if response.data:
                task_details = response.data
                if task_details.get("status") == STATUS_COMPLETE and task_details.get("output_location"):
                    return task_details.get("output_location")
                else:
                    dprint(f"Task {task_id_to_find} found but not complete or no output_location. Status: {task_details.get('status')}")
                    return None
            else:
                dprint(f"Task {task_id_to_find} not found in Supabase.")
                return None
        except Exception as e:
            print(f"Error querying Supabase for task output {task_id_to_find}: {e}")
            traceback.print_exc()
            return None
    dprint(f"DB type {DB_TYPE} not supported or client not init for get_task_output_location_from_db")
    return None

def get_task_params(task_id: str) -> str | None:
    """Gets the raw params JSON string for a given task ID."""
    if DB_TYPE == "sqlite":
        def _get_op(conn):
            cursor = conn.cursor()
            cursor.execute("SELECT params FROM tasks WHERE id = ?", (task_id,))
            row = cursor.fetchone()
            return row[0] if row else None
        return execute_sqlite_with_retry(SQLITE_DB_PATH, _get_op)
    elif DB_TYPE == "supabase" and SUPABASE_CLIENT:
        try:
            resp = SUPABASE_CLIENT.table(PG_TABLE_NAME).select("params").eq("id", task_id).single().execute()
            if resp.data:
                return resp.data.get("params")
            return None
        except Exception as e:
            dprint(f"Error getting task params for {task_id} from Supabase: {e}")
            return None
    return None

def get_task_dependency(task_id: str) -> str | None:
    """Gets the dependency task ID for a given task ID."""
    if DB_TYPE == "sqlite":
        def _get_op(conn):
            cursor = conn.cursor()
            cursor.execute("SELECT dependant_on FROM tasks WHERE id = ?", (task_id,))
            row = cursor.fetchone()
            return row[0] if row else None
        return execute_sqlite_with_retry(SQLITE_DB_PATH, _get_op)
    elif DB_TYPE == "supabase" and SUPABASE_CLIENT:
        try:
            response = SUPABASE_CLIENT.table(PG_TABLE_NAME).select("dependant_on").eq("id", task_id).single().execute()
            if response.data:
                return response.data.get("dependant_on")
            return None
        except Exception as e_supabase_dep:
            dprint(f"Error fetching dependant_on from Supabase for task {task_id}: {e_supabase_dep}")
            return None
    return None

def get_completed_segment_outputs_for_stitch(run_id: str) -> list:
    """Gets completed travel_segment outputs for a given run_id for stitching."""
    if DB_TYPE == "sqlite":
        def _get_op(conn):
            cursor = conn.cursor()
            sql_query = f"""
                SELECT json_extract(t.params, '$.segment_index') AS segment_idx, t.output_location
                FROM tasks t
                INNER JOIN (
                    SELECT 
                        json_extract(params, '$.segment_index') AS segment_idx, 
                        MAX(created_at) AS max_created_at
                    FROM tasks
                    WHERE json_extract(params, '$.orchestrator_run_id') = ?
                      AND task_type = 'travel_segment'
                      AND status = ?
                      AND output_location IS NOT NULL
                    GROUP BY segment_idx
                ) AS latest_tasks
                ON json_extract(t.params, '$.segment_index') = latest_tasks.segment_idx 
                AND t.created_at = latest_tasks.max_created_at
                WHERE json_extract(t.params, '$.orchestrator_run_id') = ?
                  AND t.task_type = 'travel_segment'
                  AND t.status = ?
                ORDER BY CAST(json_extract(t.params, '$.segment_index') AS INTEGER) ASC
            """
            cursor.execute(sql_query, (run_id, STATUS_COMPLETE, run_id, STATUS_COMPLETE))
            rows = cursor.fetchall()
            return rows
        return execute_sqlite_with_retry(SQLITE_DB_PATH, _get_op)
    elif DB_TYPE == "supabase" and SUPABASE_CLIENT:
        try:
            rpc_params = {"p_run_id": run_id, "p_gen_task_type": "travel_segment"}
            rpc_response = SUPABASE_CLIENT.rpc("func_get_completed_generation_segments_for_stitch", rpc_params).execute()

            if rpc_response.data:
                return [(item.get("segment_idx"), item.get("output_loc")) for item in rpc_response.data]
            elif rpc_response.error:
                dprint(f"Stitch Supabase: Error from RPC func_get_completed_generation_segments_for_stitch: {rpc_response.error}. Stitching may fail.")
                return []
        except Exception as e_supabase_fetch_gen:
             dprint(f"Stitch Supabase: Exception during generation segment fetch: {e_supabase_fetch_gen}. Stitching may fail.")
             return []
    return []

def get_initial_task_counts() -> tuple[int, int] | None:
    """Gets the total and queued task counts from the SQLite DB. Returns (None, None) on failure."""
    if DB_TYPE != "sqlite": return None
    
    def _get_counts_op(conn):
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {PG_TABLE_NAME}")
        total_tasks = cursor.fetchone()[0]
        cursor.execute(f"SELECT COUNT(*) FROM {PG_TABLE_NAME} WHERE status = ?", (STATUS_QUEUED,))
        queued_tasks = cursor.fetchone()[0]
        return total_tasks, queued_tasks
    
    try:
        return execute_sqlite_with_retry(SQLITE_DB_PATH, _get_counts_op)
    except Exception as e:
        print(f"SQLite error getting task counts: {e}")
        return None
        
def get_abs_path_from_db_path(db_path: str, dprint) -> Path | None:
    """Helper to resolve a path from the DB (which might be relative) to a usable absolute path."""
    if not db_path:
        return None
        
    resolved_path = None
    if DB_TYPE == "sqlite" and SQLITE_DB_PATH and isinstance(db_path, str) and db_path.startswith("files/"):
        sqlite_db_parent = Path(SQLITE_DB_PATH).resolve().parent
        resolved_path = (sqlite_db_parent / "public" / db_path).resolve()
        dprint(f"Resolved SQLite relative path '{db_path}' to '{resolved_path}'")
    else:
        # Path from DB is already absolute (Supabase) or a non-standard path
        resolved_path = Path(db_path).resolve()
    
    if resolved_path and resolved_path.exists():
        return resolved_path
    else:
        dprint(f"Warning: Resolved path '{resolved_path}' from DB path '{db_path}' does not exist.")
        return None

def upload_to_supabase_storage(local_file_path: Path, supabase_object_name: str, bucket_name: str) -> str | None:
    """Uploads a file to Supabase storage and returns its public URL."""
    if not SUPABASE_CLIENT:
        print("[ERROR] Supabase client not initialized. Cannot upload to storage.")
        return None
    if not bucket_name:
        print("[ERROR] Supabase bucket name not configured. Cannot upload to storage.")
        return None

    try:
        dprint(f"Uploading {local_file_path} to Supabase bucket '{bucket_name}' as '{supabase_object_name}'...")
        with open(local_file_path, 'rb') as f:
            # Upsert to overwrite if exists
            res = SUPABASE_CLIENT.storage.from_(bucket_name).upload(
                path=supabase_object_name,
                file=f,
                file_options={"cache-control": "3600", "upsert": "true"} 
            )
        
        dprint(f"Supabase upload response status: {res.status_code}")
        # A 200 status code indicates success.
        if res.status_code == 200:
            public_url_response = SUPABASE_CLIENT.storage.from_(bucket_name).get_public_url(supabase_object_name)
            dprint(f"Supabase get_public_url response: {public_url_response}")
            if isinstance(public_url_response, str):
                 print(f"Successfully uploaded to Supabase. Public URL: {public_url_response}")
                 return public_url_response
            else:
                 print(f"[ERROR] Supabase upload succeeded but failed to get public URL. Raw response: {public_url_response}")
                 return None
        else:
            try:
                error_details = res.json()
                print(f"[ERROR] Supabase upload failed with status {res.status_code}: {error_details}")
            except Exception:
                print(f"[ERROR] Supabase upload failed with status {res.status_code}. Response: {res.text}")
            return None

    except Exception as e:
        print(f"[ERROR] An exception occurred during Supabase upload: {e}")
        traceback.print_exc()
        return None 