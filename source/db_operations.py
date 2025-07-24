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
import httpx  # For calling Supabase Edge Function
from pathlib import Path
import base64 # Added for JWT decoding

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
SUPABASE_VIDEO_BUCKET = "image_uploads"
SUPABASE_CLIENT: SupabaseClient | None = None
SUPABASE_EDGE_COMPLETE_TASK_URL: str | None = None  # Optional override for edge function
SUPABASE_ACCESS_TOKEN: str | None = None # Will be set by headless.py
SUPABASE_EDGE_CREATE_TASK_URL: str | None = None # Will be set by headless.py
SUPABASE_EDGE_CLAIM_TASK_URL: str | None = None # Will be set by headless.py

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
        print(f"[DEBUG {datetime.datetime.now().isoformat()}] {msg}")

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
# Internal Helpers for Supabase
# -----------------------------------------------------------------------------

def _get_user_id_from_jwt(jwt_str: str) -> str | None:
    """Decodes a JWT and extracts the 'sub' (user ID) claim without validation."""
    if not jwt_str:
        return None
    try:
        # JWT is composed of header.payload.signature
        _, payload_b64, _ = jwt_str.split('.')
        # The payload is base64 encoded. It needs to be padded to be decoded correctly.
        payload_b64 += '=' * (-len(payload_b64) % 4)
        payload_json = base64.b64decode(payload_b64).decode('utf-8')
        payload = json.loads(payload_json)
        user_id = payload.get('sub')
        dprint(f"JWT Decode: Extracted user ID (sub): {user_id}")
        return user_id
    except Exception as e:
        dprint(f"[ERROR] Could not decode JWT to get user ID: {e}")
        return None

def _is_jwt_token(token_str: str) -> bool:
    """
    Checks if a token string looks like a JWT (has 3 parts separated by dots).
    """
    if not token_str:
        return False
    parts = token_str.split('.')
    return len(parts) == 3

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

            # --- Add generation_started_at column if not exists ---
            generation_started_at_column_exists = 'generation_started_at' in columns
            if not generation_started_at_column_exists:
                dprint("SQLite Migration: 'generation_started_at' column not found. Adding it.")
                cursor.execute("ALTER TABLE tasks ADD COLUMN generation_started_at TEXT NULL")
                dprint("SQLite Migration: 'generation_started_at' column added.")
            else:
                dprint("SQLite Migration: 'generation_started_at' column already exists.")

            # --- Add generation_processed_at column if not exists ---
            generation_processed_at_column_exists = 'generation_processed_at' in columns
            if not generation_processed_at_column_exists:
                dprint("SQLite Migration: 'generation_processed_at' column not found. Adding it.")
                cursor.execute("ALTER TABLE tasks ADD COLUMN generation_processed_at TEXT NULL")
                dprint("SQLite Migration: 'generation_processed_at' column added.")
            else:
                dprint("SQLite Migration: 'generation_processed_at' column already exists.")

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
                generation_started_at TEXT NULL,
                generation_processed_at TEXT NULL,
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
            task_id = task_row[0]
            dprint(f"SQLite: Fetched raw task_row: {task_row}")
            
            # Update status to IN_PROGRESS and set generation_started_at
            current_utc_iso_ts = datetime.datetime.utcnow().isoformat() + "Z"
            cursor.execute("""
                UPDATE tasks 
                SET status = ?, 
                    updated_at = ?, 
                    generation_started_at = ?
                WHERE id = ?
            """, (STATUS_IN_PROGRESS, current_utc_iso_ts, current_utc_iso_ts, task_id))
            
            return {"task_id": task_id, "params": json.loads(task_row[1]), "task_type": task_row[2], "project_id": task_row[3]}
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

            # Step 2: Update status, updated_at, and generation_processed_at for the completion
            current_utc_iso_ts_status_update = datetime.datetime.utcnow().isoformat() + "Z"
            dprint(f"SQLite Update (Split Step 2): Updating status for {task_id} to {status}")
            cursor.execute("UPDATE tasks SET status = ?, updated_at = ?, generation_processed_at = ? WHERE id = ?",
                           (status, current_utc_iso_ts_status_update, current_utc_iso_ts_status_update, task_id))
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
    print(f"[UPDATE_TASK_STATUS_DEBUG] Called with:")
    print(f"[UPDATE_TASK_STATUS_DEBUG]   task_id: '{task_id}'")
    print(f"[UPDATE_TASK_STATUS_DEBUG]   status: '{status}'")
    print(f"[UPDATE_TASK_STATUS_DEBUG]   output_location: '{output_location}'")
    print(f"[UPDATE_TASK_STATUS_DEBUG]   DB_TYPE: '{DB_TYPE}'")
    
    try:
        if DB_TYPE == "supabase":
            print(f"[UPDATE_TASK_STATUS_DEBUG] Dispatching to update_task_status_supabase")
            result = update_task_status_supabase(task_id, status, output_location)
            print(f"[UPDATE_TASK_STATUS_DEBUG] update_task_status_supabase completed successfully")
            return result
        else:
            print(f"[UPDATE_TASK_STATUS_DEBUG] Dispatching to update_task_status_sqlite")
            result = update_task_status_sqlite(SQLITE_DB_PATH, task_id, status, output_location)
            print(f"[UPDATE_TASK_STATUS_DEBUG] update_task_status_sqlite completed successfully")
            return result
    except Exception as e:
        print(f"[UPDATE_TASK_STATUS_DEBUG] ❌ Exception in update_task_status: {e}")
        print(f"[UPDATE_TASK_STATUS_DEBUG] Exception type: {type(e).__name__}")
        traceback.print_exc()
        raise

def init_db_supabase(): # Renamed from init_db_postgres
    """Check if the Supabase tasks table exists (assuming it's already set up)."""
    if not SUPABASE_CLIENT:
        print("[ERROR] Supabase client not initialized. Cannot check database table.")
        sys.exit(1)
    try:
        # Simply check if the tasks table exists by querying it
        # Since the table already exists, we don't need to create it
        result = SUPABASE_CLIENT.table(PG_TABLE_NAME).select("count", count="exact").limit(1).execute()
        print(f"Supabase: Table '{PG_TABLE_NAME}' exists and accessible (count: {result.count})")
        return True
    except Exception as e: 
        print(f"[ERROR] Supabase table check failed: {e}")
        # Don't exit - the table might exist but have different permissions
        # Let the actual operations try and fail gracefully
        return False

def get_oldest_queued_task_supabase(worker_id: str = None): # Renamed from get_oldest_queued_task_postgres
    """Fetches the oldest task via Supabase Edge Function only."""
    if not SUPABASE_CLIENT:
        print("[ERROR] Supabase client not initialized. Cannot get task.")
        return None
    
    # Use provided worker_id or use the specific GPU worker ID
    if not worker_id:
        worker_id = "gpu-20250723_221138-afa8403b"
        dprint(f"DEBUG: No worker_id provided, using default GPU worker: {worker_id}")
    else:
        dprint(f"DEBUG: Using provided worker_id: {worker_id}")
    
    # Use Edge Function exclusively
    edge_url = (
        SUPABASE_EDGE_CLAIM_TASK_URL 
        or os.getenv('SUPABASE_EDGE_CLAIM_TASK_URL')
        or (f"{SUPABASE_URL.rstrip('/')}/functions/v1/claim-next-task" if SUPABASE_URL else None)
    )
    
    if edge_url and SUPABASE_ACCESS_TOKEN:
        try:
            dprint(f"DEBUG get_oldest_queued_task_supabase: Calling Edge Function at {edge_url}")
            dprint(f"DEBUG: Using worker_id: {worker_id}")
            
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {SUPABASE_ACCESS_TOKEN}'
            }
            
            # Pass worker_id in the request body for edge function to use
            payload = {"worker_id": worker_id}
            
            resp = httpx.post(edge_url, json=payload, headers=headers, timeout=15)
            dprint(f"Edge Function response status: {resp.status_code}")
            
            if resp.status_code == 200:
                task_data = resp.json()
                dprint(f"Edge Function claimed task: {task_data}")
                return task_data  # Already in the expected format
            elif resp.status_code == 204:
                dprint("Edge Function: No queued tasks available")
                return None
            else:
                dprint(f"Edge Function returned {resp.status_code}: {resp.text}")
                return None
        except Exception as e_edge:
            dprint(f"Edge Function call failed: {e_edge}")
            return None
    else:
        dprint("ERROR: No edge function URL or access token available for task claiming")
        return None

def update_task_status_supabase(task_id_str, status_str, output_location_val=None): # Renamed from update_task_status_postgres
    """Updates a task's status via Supabase Edge Functions only."""
    dprint(f"[DEBUG] update_task_status_supabase called: task_id={task_id_str}, status={status_str}, output_location={output_location_val}")
    
    if not SUPABASE_CLIENT:
        print("[ERROR] Supabase client not initialized. Cannot update task status.")
        return

    # --- Use edge functions for ALL status updates ---
    if status_str == STATUS_COMPLETE and output_location_val is not None:
        # Use complete-task edge function for completion with file
        edge_url = (
            SUPABASE_EDGE_COMPLETE_TASK_URL
            or (os.getenv("SUPABASE_EDGE_COMPLETE_TASK_URL") or None)
            or (f"{SUPABASE_URL.rstrip('/')}/functions/v1/complete-task" if SUPABASE_URL else None)
        )
        
        if not edge_url:
            print(f"[ERROR] No complete-task edge function URL available")
            return

        try:
            # Check if output_location_val is a local file path
            output_path = Path(output_location_val)
            
            if output_path.exists() and output_path.is_file():
                # Read the file and encode as base64
                import base64
                with open(output_path, 'rb') as f:
                    file_data = base64.b64encode(f.read()).decode('utf-8')
                
                headers = {"Content-Type": "application/json"}
                if SUPABASE_ACCESS_TOKEN:
                    headers["Authorization"] = f"Bearer {SUPABASE_ACCESS_TOKEN}"

                payload = {
                    "task_id": task_id_str, 
                    "file_data": file_data,
                    "filename": output_path.name
                }
                dprint(f"[DEBUG] Calling complete-task Edge Function with file upload for task {task_id_str}")
                resp = httpx.post(edge_url, json=payload, headers=headers, timeout=60)
                
                if resp.status_code == 200:
                    dprint(f"[DEBUG] Edge function SUCCESS for task {task_id_str} → status COMPLETE with file upload")
                    return
                else:
                    print(f"[ERROR] complete-task edge function failed: {resp.status_code} - {resp.text}")
                    return
            else:
                # Not a local file, treat as URL
                payload = {"task_id": task_id_str, "output_location": output_location_val}
                
                headers = {"Content-Type": "application/json"}
                if SUPABASE_ACCESS_TOKEN:
                    headers["Authorization"] = f"Bearer {SUPABASE_ACCESS_TOKEN}"
                
                resp = httpx.post(edge_url, json=payload, headers=headers, timeout=30)

                if resp.status_code == 200:
                    dprint(f"[DEBUG] Edge function SUCCESS for task {task_id_str} → status COMPLETE")
                    return
                else:
                    print(f"[ERROR] complete-task edge function failed: {resp.status_code} - {resp.text}")
                    return
        except Exception as e_edge:
            print(f"[ERROR] complete-task edge function exception: {e_edge}")
            return
    else:
        # Use update-task-status edge function for all other status updates
        edge_url = (
            os.getenv("SUPABASE_EDGE_UPDATE_TASK_URL") 
            or (f"{SUPABASE_URL.rstrip('/')}/functions/v1/update-task-status" if SUPABASE_URL else None)
        )
        
        if not edge_url:
            print(f"[ERROR] No update-task-status edge function URL available")
            return
            
        try:
            headers = {"Content-Type": "application/json"}
            if SUPABASE_ACCESS_TOKEN:
                headers["Authorization"] = f"Bearer {SUPABASE_ACCESS_TOKEN}"
            
            payload = {
                "task_id": task_id_str,
                "status": status_str
            }
            
            if output_location_val:
                payload["output_location"] = output_location_val
            
            dprint(f"[DEBUG] Calling update-task-status Edge Function for task {task_id_str} → {status_str}")
            resp = httpx.post(edge_url, json=payload, headers=headers, timeout=30)
            
            if resp.status_code == 200:
                dprint(f"[DEBUG] Edge function SUCCESS for task {task_id_str} → status {status_str}")
                return
            else:
                print(f"[ERROR] update-task-status edge function failed: {resp.status_code} - {resp.text}")
                return
                
        except Exception as e:
            print(f"[ERROR] update-task-status edge function exception: {e}")
            return

def _migrate_supabase_schema():
    """Legacy migration function - no longer used. Edge Function architecture complete."""
    dprint("Supabase Migration: Migration to Edge Functions complete. Schema migrations handled externally.")
    return  # No-op - migrations complete

def _run_db_migrations():
    """Runs database migrations based on the configured DB_TYPE."""
    dprint(f"DB Migrations: Running for DB_TYPE: {DB_TYPE}")
    if DB_TYPE == "sqlite":
        if SQLITE_DB_PATH:
            _migrate_sqlite_schema(SQLITE_DB_PATH)
        else:
            print("[ERROR] DB Migration: SQLITE_DB_PATH not set. Skipping SQLite migration.")
    elif DB_TYPE == "supabase":
        # The Supabase schema is managed externally. Edge Function architecture complete.
        dprint("DB Migrations: Skipping Supabase migrations (table assumed to exist).")
        return
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
        # Use Edge Function exclusively
        
        # Build Edge URL – env var override > global constant > default pattern
        edge_url = (
            SUPABASE_EDGE_CREATE_TASK_URL  # may be set at runtime
            if "SUPABASE_EDGE_CREATE_TASK_URL" in globals() else None
        ) or (os.getenv("SUPABASE_EDGE_CREATE_TASK_URL") or None) or (
            f"{SUPABASE_URL.rstrip('/')}/functions/v1/create-task" if SUPABASE_URL else None
        )

        if not edge_url:
            raise ValueError("Edge Function URL for create-task is not configured")

        headers = {"Content-Type": "application/json"}
        if SUPABASE_ACCESS_TOKEN:
            headers["Authorization"] = f"Bearer {SUPABASE_ACCESS_TOKEN}"

        payload_edge = {
            "task_id": task_id,
            "params": params_for_db,  # pass JSON directly
            "task_type": task_type_str,
            "project_id": project_id,
            "dependant_on": dependant_on,
        }

        dprint(f"Supabase Edge call >>> POST {edge_url} payload={str(payload_edge)[:120]}…")

        try:
            resp = httpx.post(edge_url, json=payload_edge, headers=headers, timeout=30)

            if resp.status_code == 200:
                print(f"Task {task_id} (Type: {task_type_str}) queued via Edge Function.")
                return
            else:
                error_msg = f"Edge Function create-task failed: {resp.status_code} - {resp.text}"
                print(f"[ERROR] {error_msg}")
                raise RuntimeError(error_msg)
                
        except httpx.RequestError as e:
            error_msg = f"Edge Function create-task request failed: {e}"
            print(f"[ERROR] {error_msg}")
            raise RuntimeError(error_msg)

    else: # Default to SQLite
        db_to_use = db_path if db_path else SQLITE_DB_PATH
        if not db_to_use:
            raise ValueError("SQLite DB path is not configured.")

        def _add_op(conn):
            cursor = conn.cursor()
            current_timestamp = datetime.datetime.utcnow().isoformat() + "Z"
            cursor.execute(
                f"INSERT INTO tasks (id, params, task_type, status, created_at, project_id, dependant_on) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    task_id,
                    params_json_str,
                    task_type_str,
                    STATUS_QUEUED,
                    current_timestamp,
                    project_id,
                    dependant_on,
                ),
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
                # Direct table query for polling status
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

def get_predecessor_output_via_edge_function(task_id: str) -> tuple[str | None, str | None]:
    """
    Gets both the predecessor task ID and its output location in a single call using Edge Function.
    Returns: (predecessor_id, output_location) or (None, None) if no dependency or error.
    
    This replaces the separate calls to get_task_dependency() + get_task_output_location_from_db().
    """
    if DB_TYPE == "sqlite":
        # For SQLite, fall back to separate calls since we don't have Edge Functions
        predecessor_id = get_task_dependency(task_id)
        if predecessor_id:
            output_location = get_task_output_location_from_db(predecessor_id)
            return predecessor_id, output_location
        return None, None
        
    elif DB_TYPE == "supabase" and SUPABASE_URL and SUPABASE_ACCESS_TOKEN:
        # Use the new Edge Function for Supabase
        edge_url = f"{SUPABASE_URL.rstrip('/')}/functions/v1/get-predecessor-output"
        
        try:
            dprint(f"Calling Edge Function: {edge_url} for task {task_id}")
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {SUPABASE_ACCESS_TOKEN}'
            }
            
            resp = httpx.post(edge_url, json={"task_id": task_id}, headers=headers, timeout=15)
            dprint(f"Edge Function response status: {resp.status_code}")
            
            if resp.status_code == 200:
                result = resp.json()
                dprint(f"Edge Function result: {result}")
                
                if result is None:
                    # No dependency
                    return None, None
                
                predecessor_id = result.get("predecessor_id")
                output_location = result.get("output_location")
                return predecessor_id, output_location
                
            elif resp.status_code == 404:
                dprint(f"Edge Function: Task {task_id} not found")
                return None, None
            else:
                dprint(f"Edge Function returned {resp.status_code}: {resp.text}. Falling back to direct queries.")
                # Fall back to separate calls
                predecessor_id = get_task_dependency(task_id)
                if predecessor_id:
                    output_location = get_task_output_location_from_db(predecessor_id)
                    return predecessor_id, output_location
                return None, None
                
        except Exception as e_edge:
            dprint(f"Edge Function call failed: {e_edge}. Falling back to direct queries.")
            # Fall back to separate calls
            predecessor_id = get_task_dependency(task_id)
            if predecessor_id:
                output_location = get_task_output_location_from_db(predecessor_id)
                return predecessor_id, output_location
            return None, None
    
    # If we can't use Edge Function, fall back to separate calls
    predecessor_id = get_task_dependency(task_id)
    if predecessor_id:
        output_location = get_task_output_location_from_db(predecessor_id)
        return predecessor_id, output_location
    return None, None


def get_completed_segment_outputs_for_stitch(run_id: str, project_id: str | None = None) -> list:
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
    elif DB_TYPE == "supabase":
        edge_url = f"{SUPABASE_URL.rstrip('/')}/functions/v1/get-completed-segments"
        try:
            dprint(f"Calling Edge Function: {edge_url} for run_id {run_id}, project_id {project_id}")
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {SUPABASE_ACCESS_TOKEN}'
            }
            payload = {"run_id": run_id}
            if project_id:
                payload["project_id"] = project_id

            resp = httpx.post(edge_url, json=payload, headers=headers, timeout=15)
            if resp.status_code == 200:
                results = resp.json()
                sorted_results = sorted(results, key=lambda x: x['segment_index'])
                return [(r['segment_index'], r['output_location']) for r in sorted_results]
            else:
                dprint(f"Edge Function returned {resp.status_code}: {resp.text}. Falling back to direct query.")
        except Exception as e:
            dprint(f"Edge Function failed: {e}. Falling back to direct query.")
        
        # Fallback to direct query
        try:
            # First, let's debug by getting ALL completed tasks to see what's there
            debug_resp = SUPABASE_CLIENT.table(PG_TABLE_NAME).select("id, task_type, status, params, output_location")\
                .eq("status", STATUS_COMPLETE).execute()
            
            dprint(f"[DEBUG_STITCH] Looking for run_id: '{run_id}' (type: {type(run_id)})")
            dprint(f"[DEBUG_STITCH] Total completed tasks in DB: {len(debug_resp.data) if debug_resp.data else 0}")
            
            travel_segment_count = 0
            matching_run_id_count = 0
            
            if debug_resp.data:
                for task in debug_resp.data:
                    task_type = task.get("task_type", "")
                    if task_type == "travel_segment":
                        travel_segment_count += 1
                        params_raw = task.get("params", {})
                        try:
                            params_obj = params_raw if isinstance(params_raw, dict) else json.loads(params_raw)
                            task_run_id = params_obj.get("orchestrator_run_id")
                            dprint(f"[DEBUG_STITCH] Found travel_segment task {task.get('id')}: orchestrator_run_id='{task_run_id}' (type: {type(task_run_id)}), segment_index={params_obj.get('segment_index')}, output_location={task.get('output_location', 'None')}")
                            
                            if str(task_run_id) == str(run_id):
                                matching_run_id_count += 1
                                dprint(f"[DEBUG_STITCH] ✅ MATCH FOUND! Task {task.get('id')} matches run_id {run_id}")
                        except Exception as e_debug:
                            dprint(f"[DEBUG_STITCH] Error parsing params for task {task.get('id')}: {e_debug}")
            
            dprint(f"[DEBUG_STITCH] Travel_segment tasks found: {travel_segment_count}")
            dprint(f"[DEBUG_STITCH] Tasks matching run_id '{run_id}': {matching_run_id_count}")
            
            # Now do the actual query
            sel_resp = SUPABASE_CLIENT.table(PG_TABLE_NAME).select("params, output_location")\
                .eq("task_type", "travel_segment").eq("status", STATUS_COMPLETE).execute()
            
            results = []
            if sel_resp.data:
                import json
                for i, row in enumerate(sel_resp.data):
                    params_raw = row.get("params")
                    if params_raw is None: 
                        continue
                    try:
                        params_obj = params_raw if isinstance(params_raw, dict) else json.loads(params_raw)
                    except Exception as e:
                        continue
                    
                    row_run_id = params_obj.get("orchestrator_run_id")
                    
                    # Use string comparison to handle type mismatches
                    if str(row_run_id) == str(run_id):
                        seg_idx = params_obj.get("segment_index")
                        output_loc = row.get("output_location")
                        results.append((seg_idx, output_loc))
                        dprint(f"[DEBUG_STITCH] Added to results: segment_index={seg_idx}, output_location={output_loc}")
            
            sorted_results = sorted(results, key=lambda x: x[0] if x[0] is not None else 0)
            dprint(f"[DEBUG_STITCH] Final sorted results: {sorted_results}")
            return sorted_results
        except Exception as e_sel:
            dprint(f"Stitch Supabase: Direct select failed: {e_sel}")
            traceback.print_exc()
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

def upload_to_supabase_storage(file_path, bucket_name="image_uploads", custom_path=None):
    """Upload a file to Supabase storage with improved error handling."""
    if not SUPABASE_CLIENT:
        print("[ERROR] Supabase client not initialized. Cannot upload file.")
        return None

    try:
        # Ensure file exists
        if not os.path.exists(file_path):
            print(f"[ERROR] File does not exist: {file_path}")
            return None

        file_name = os.path.basename(file_path)
        storage_path = custom_path or f"system/{file_name}"
        
        dprint(f"Uploading {file_path} to {bucket_name}/{storage_path}")

        # Read file data
        with open(file_path, 'rb') as f:
            file_data = f.read()

        # Upload to Supabase storage
        res = SUPABASE_CLIENT.storage.from_(bucket_name).upload(storage_path, file_data)
        
        # Handle the new UploadResponse API
        if hasattr(res, 'error') and res.error:
            print(f"Supabase upload error: {res.error}")
            return None
        elif hasattr(res, 'data') and res.data:
            # Success case - get the public URL
            public_url = SUPABASE_CLIENT.storage.from_(bucket_name).get_public_url(storage_path)
            dprint(f"Supabase upload successful. Public URL: {public_url}")
            return public_url
        else:
            # Try to get public URL anyway (upload might have succeeded)
            try:
                public_url = SUPABASE_CLIENT.storage.from_(bucket_name).get_public_url(storage_path)
                dprint(f"Upload completed, public URL: {public_url}")
                return public_url
            except Exception as url_error:
                print(f"[ERROR] Could not get public URL after upload: {url_error}")
                return None

    except Exception as e:
        print(f"[ERROR] An exception occurred during Supabase upload: {e}")
        return None 

def mark_task_failed_supabase(task_id_str, error_message):
    """Marks a task as Failed with an error message using direct database update."""
    dprint(f"Marking task {task_id_str} as Failed with message: {error_message}")
    if not SUPABASE_CLIENT:
        print("[ERROR] Supabase client not initialized. Cannot mark task failed.")
        return
    
    # Use the standard update function which now uses direct database updates for non-COMPLETE statuses
    update_task_status_supabase(task_id_str, STATUS_FAILED, error_message) 